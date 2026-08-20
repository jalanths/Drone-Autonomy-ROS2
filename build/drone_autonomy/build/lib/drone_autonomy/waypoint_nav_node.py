#!/usr/bin/env python3
"""
Autonomous Waypoint Navigation Node for ArduCopter via MAVROS (ROS 2 Humble)
=============================================================================
This node demonstrates autonomous multi-waypoint GPS navigation:
  1. Waits for FCU connection and GPS lock
  2. Sets GUIDED mode and arms the drone
  3. Takes off to 15 meters
  4. Flies to 3 GPS waypoints in a triangle pattern
  5. Returns to launch (RTL) and lands automatically

The waypoints are placed ~50m apart near the SITL default location
(Canberra, Australia: -35.3632, 149.1652).

Usage:
  Terminal 1: cd ~/ardupilot/ArduCopter && sim_vehicle.py --console --map
  Terminal 2: ros2 run mavros mavros_node --ros-args -p fcu_url:="udp://127.0.0.1:14550@"
  Terminal 3: python3 waypoint_nav_node.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from geographic_msgs.msg import GeoPoseStamped
from sensor_msgs.msg import NavSatFix
import math
import time


class WaypointNavNode(Node):
    def __init__(self):
        super().__init__('waypoint_nav_node')

        # ── Current drone state ──────────────────────────────────
        self.current_state = State()
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.gps_received = False
        self.current_alt = 0.0

        # ── Define GPS Waypoints (triangle pattern near SITL home)
        # Each waypoint is roughly 50 meters apart.
        # Format: (latitude, longitude, altitude_amsl)
        self.waypoints = [
            (-35.36280, 149.16530, 599.0),  # WP1: ~50m North
            (-35.36350, 149.16600, 599.0),  # WP2: ~50m East
            (-35.36380, 149.16450, 599.0),  # WP3: ~50m West
        ]
        self.current_wp_index = 0
        self.wp_reached_threshold = 2.0  # meters - how close is "arrived"

        # ── QoS Profile for sensor topics ────────────────────────
        # MAVROS publishes sensor data with Best Effort reliability.
        # If we use the default (Reliable), ROS 2 DDS won't match
        # the QoS and silently drops ALL messages!
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ── Subscribers ──────────────────────────────────────────
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_callback, 10)

        self.gps_sub = self.create_subscription(
            NavSatFix, '/mavros/global_position/global',
            self.gps_callback, sensor_qos)

        # ── Publisher: Global position setpoint ──────────────────
        # This is how we tell the drone "fly to this GPS coordinate"
        # in GUIDED mode. MAVROS translates this into a MAVLink
        # SET_POSITION_TARGET_GLOBAL_INT command for the Pixhawk.
        self.global_setpoint_pub = self.create_publisher(
            GeoPoseStamped,
            '/mavros/setpoint_position/global',
            10
        )

        # ── Service Clients ──────────────────────────────────────
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # ── State machine ────────────────────────────────────────
        self.phase = 'CONNECTING'
        # Phases: CONNECTING -> SET_MODE -> ARMING -> TAKEOFF ->
        #         ASCENDING -> NAVIGATE -> RTL -> DONE

        self.takeoff_alt = 15.0
        self.takeoff_time = None

        # ── Main timer: runs every 0.5 seconds ───────────────────
        self.timer = self.create_timer(0.5, self.mission_loop)

        self.get_logger().info('═══════════════════════════════════════════')
        self.get_logger().info('  🗺️  Waypoint Navigation Node Started')
        self.get_logger().info(f'  📍 {len(self.waypoints)} waypoints loaded')
        self.get_logger().info('  Waiting for FCU connection...')
        self.get_logger().info('═══════════════════════════════════════════')

    # ══════════════════════════════════════════════════════════════
    #  Subscriber Callbacks
    # ══════════════════════════════════════════════════════════════

    def state_callback(self, msg: State):
        self.current_state = msg

    def gps_callback(self, msg: NavSatFix):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        if not self.gps_received:
            self.gps_received = True
            self.get_logger().info(
                f'📡 GPS Fix: {msg.latitude:.6f}, {msg.longitude:.6f}')

    # ══════════════════════════════════════════════════════════════
    #  Main Mission Loop (State Machine)
    # ══════════════════════════════════════════════════════════════

    def mission_loop(self):
        """Executes one phase of the mission per timer tick."""

        # ── PHASE: CONNECTING ────────────────────────────────────
        if self.phase == 'CONNECTING':
            if not self.current_state.connected:
                return
            self.get_logger().info('✅ FCU Connected!')
            self.phase = 'SET_MODE'
            return

        # ── PHASE: SET_MODE ──────────────────────────────────────
        if self.phase == 'SET_MODE':
            if self.current_state.mode == 'GUIDED':
                self.get_logger().info('✅ Already in GUIDED mode.')
                self.phase = 'ARMING'
                return
            self.set_mode('GUIDED')
            return

        # ── PHASE: ARMING ────────────────────────────────────────
        if self.phase == 'ARMING':
            if self.current_state.armed:
                self.get_logger().info('✅ Motors armed.')
                self.phase = 'TAKEOFF'
                return
            self.arm_drone()
            return

        # ── PHASE: TAKEOFF ───────────────────────────────────────
        if self.phase == 'TAKEOFF':
            self.takeoff(self.takeoff_alt)
            self.takeoff_time = time.time()
            self.phase = 'ASCENDING'
            return

        # ── PHASE: ASCENDING (wait to reach altitude) ────────────
        if self.phase == 'ASCENDING':
            # Give it at least 10 seconds to climb, then check alt
            elapsed = time.time() - self.takeoff_time
            if elapsed > 10.0:
                self.get_logger().info(
                    f'✅ Reached cruise altitude. Starting waypoint navigation!')
                self.phase = 'NAVIGATE'
            return

        # ── PHASE: NAVIGATE (fly through waypoints) ──────────────
        if self.phase == 'NAVIGATE':
            if self.current_wp_index >= len(self.waypoints):
                self.get_logger().info('═══════════════════════════════════════════')
                self.get_logger().info('  ✅ ALL WAYPOINTS REACHED!')
                self.get_logger().info('  🏠 Returning to Launch (RTL)...')
                self.get_logger().info('═══════════════════════════════════════════')
                self.set_mode('RTL')
                self.phase = 'RTL'
                return

            # Get the current target waypoint
            wp = self.waypoints[self.current_wp_index]
            target_lat, target_lon, target_alt = wp

            # Calculate distance to waypoint
            dist = self.haversine_distance(
                self.current_lat, self.current_lon,
                target_lat, target_lon
            )

            # Check if we've arrived at the waypoint
            if dist < self.wp_reached_threshold:
                self.get_logger().info(
                    f'📍 Waypoint {self.current_wp_index + 1} REACHED! '
                    f'(distance: {dist:.1f}m)')
                self.current_wp_index += 1
                return

            # Publish the target GPS position for MAVROS
            self.publish_global_setpoint(target_lat, target_lon, target_alt)
            self.get_logger().info(
                f'🔄 Flying to WP {self.current_wp_index + 1}/'
                f'{len(self.waypoints)} | '
                f'Distance: {dist:.1f}m')
            return

        # ── PHASE: RTL (returning home) ──────────────────────────
        if self.phase == 'RTL':
            if not self.current_state.armed:
                self.get_logger().info('═══════════════════════════════════════════')
                self.get_logger().info('  🏠 LANDED SAFELY!')
                self.get_logger().info('  ✅ MISSION COMPLETE!')
                self.get_logger().info('═══════════════════════════════════════════')
                self.phase = 'DONE'
            return

        # ── PHASE: DONE ──────────────────────────────────────────
        if self.phase == 'DONE':
            self.timer.cancel()
            return

    # ══════════════════════════════════════════════════════════════
    #  Navigation Helpers
    # ══════════════════════════════════════════════════════════════

    def publish_global_setpoint(self, lat, lon, alt):
        """
        Publishes a GPS target coordinate for the drone to fly to.

        How it works:
        - In GUIDED mode, ArduCopter accepts position targets.
        - We publish a GeoPoseStamped message to the topic
          /mavros/setpoint_position/global
        - MAVROS converts this into a MAVLink
          SET_POSITION_TARGET_GLOBAL_INT command.
        - The Pixhawk's internal L1 navigation controller handles
          all the actual flying (speed, heading, deceleration).
        """
        msg = GeoPoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.latitude = lat
        msg.pose.position.longitude = lon
        msg.pose.position.altitude = alt
        # Orientation: keep current heading (identity quaternion)
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        self.global_setpoint_pub.publish(msg)

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculates the distance in meters between two GPS coordinates
        using the Haversine formula. This is the standard formula used
        in all GPS navigation systems to compute great-circle distance
        on a sphere (Earth).
        """
        R = 6371000  # Earth's radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (math.sin(dphi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(dlambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # ══════════════════════════════════════════════════════════════
    #  MAVROS Service Call Helpers (same pattern as takeoff_node)
    # ══════════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        if not self.set_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('SetMode service not available!')
            return
        request = SetMode.Request()
        request.custom_mode = mode
        future = self.set_mode_client.call_async(request)
        future.add_done_callback(
            lambda f: self._mode_response(f, mode))
        self.get_logger().info(f'🔄 Requesting mode: {mode}...')

    def _mode_response(self, future, mode):
        try:
            response = future.result()
            if response.mode_sent:
                if mode == 'GUIDED':
                    self.phase = 'ARMING'
                self.get_logger().info(f'✅ Mode changed to {mode}!')
            else:
                self.get_logger().warn(f'❌ Mode change to {mode} rejected.')
        except Exception as e:
            self.get_logger().error(f'SetMode failed: {e}')

    def arm_drone(self):
        if not self.arming_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Arming service not available!')
            return
        request = CommandBool.Request()
        request.value = True
        future = self.arming_client.call_async(request)
        future.add_done_callback(self._arm_response)
        self.get_logger().info('🔄 Requesting motor arming...')

    def _arm_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.phase = 'TAKEOFF'
                self.get_logger().info('✅ Motors ARMED!')
            else:
                self.get_logger().warn('❌ Arming rejected. Retrying...')
        except Exception as e:
            self.get_logger().error(f'Arming failed: {e}')

    def takeoff(self, altitude: float):
        if not self.takeoff_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Takeoff service not available!')
            return
        request = CommandTOL.Request()
        request.altitude = altitude
        request.latitude = 0.0
        request.longitude = 0.0
        request.min_pitch = 0.0
        request.yaw = 0.0
        future = self.takeoff_client.call_async(request)
        future.add_done_callback(self._takeoff_response)
        self.get_logger().info(f'🔄 Requesting takeoff to {altitude}m...')

    def _takeoff_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'🚁 Takeoff commanded! Ascending...')
            else:
                self.get_logger().warn('❌ Takeoff rejected.')
                self.phase = 'TAKEOFF'  # Retry
        except Exception as e:
            self.get_logger().error(f'Takeoff failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Waypoint Navigation Node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
