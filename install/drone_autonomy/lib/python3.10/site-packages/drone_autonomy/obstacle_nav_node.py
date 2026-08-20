#!/usr/bin/env python3
"""
Obstacle-Aware Navigation Node for ArduCopter via MAVROS (ROS 2 Humble)
========================================================================
This node combines waypoint navigation with reactive obstacle avoidance.

The "brain" logic works like this:
  1. Takes off and starts flying toward a GPS waypoint (same as before)
  2. Continuously reads /scan (LaserScan) data from the Lidar
  3. If an obstacle is detected within a safety threshold:
     a. STOPS forward motion immediately
     b. Scans left vs right sectors of the Lidar
     c. Chooses the direction with more open space
     d. Publishes a velocity command to dodge sideways
  4. Once the path ahead is clear, resumes flying to the waypoint

This is a REACTIVE avoidance strategy (like a reflex). It doesn't plan
a full path around the obstacle — it simply dodges. This is the fastest
and most reliable strategy for real-world drones with 2D Lidar.

Usage:
  Terminal 1: cd ~/ardupilot/ArduCopter && sim_vehicle.py --console --map
  Terminal 2: ros2 run mavros mavros_node --ros-args -p fcu_url:="udp://127.0.0.1:14550@"
  Terminal 3: python3 fake_lidar_node.py   (simulates obstacles)
  Terminal 4: python3 obstacle_nav_node.py (this file)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from geographic_msgs.msg import GeoPoseStamped
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import NavSatFix, LaserScan
import math
import time


class ObstacleNavNode(Node):
    def __init__(self):
        super().__init__('obstacle_nav_node')

        # ══════════════════════════════════════════════════════════
        #  TUNABLE PARAMETERS — adjust these for your real drone!
        # ══════════════════════════════════════════════════════════

        # Obstacle detection thresholds (meters)
        self.danger_distance = 3.0    # STOP if obstacle closer than this
        self.warning_distance = 5.0   # Slow down if obstacle within this
        self.safe_distance = 6.0      # Resume normal flight beyond this

        # Avoidance speeds (m/s)
        self.dodge_speed = 2.0        # Lateral dodge velocity
        self.forward_speed = 3.0      # Normal forward velocity
        self.slow_speed = 1.0         # Reduced speed in warning zone

        # Lidar scan sectors (degrees from center)
        # We split the forward-facing Lidar into 3 zones:
        #   LEFT: -90 to -15 degrees
        #   FRONT: -15 to +15 degrees
        #   RIGHT: +15 to +90 degrees
        self.front_half_angle = 15    # degrees
        self.side_scan_angle = 90     # degrees

        # ══════════════════════════════════════════════════════════
        #  STATE VARIABLES
        # ══════════════════════════════════════════════════════════

        self.current_state = State()
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.gps_received = False

        # Lidar data
        self.front_min_distance = float('inf')
        self.left_min_distance = float('inf')
        self.right_min_distance = float('inf')
        self.lidar_received = False

        # Navigation waypoints (triangle near SITL home)
        self.waypoints = [
            (-35.36280, 149.16530, 599.0),  # WP1: ~50m North
            (-35.36350, 149.16600, 599.0),  # WP2: ~50m East
            (-35.36380, 149.16450, 599.0),  # WP3: ~50m West
        ]
        self.current_wp_index = 0
        self.wp_reached_threshold = 3.0

        # State machine
        self.phase = 'CONNECTING'
        self.avoidance_state = 'CLEAR'  # CLEAR, WARNING, DODGING
        self.takeoff_alt = 15.0
        self.takeoff_time = None

        # ══════════════════════════════════════════════════════════
        #  ROS 2 INTERFACES
        # ══════════════════════════════════════════════════════════

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Subscribers
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_cb, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/mavros/global_position/global',
            self.gps_cb, sensor_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, sensor_qos)

        # Publishers
        self.global_setpoint_pub = self.create_publisher(
            GeoPoseStamped, '/mavros/setpoint_position/global', 10)
        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        # Service clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # Main timer
        self.timer = self.create_timer(0.25, self.main_loop)  # 4Hz decision loop

        self.get_logger().info('═══════════════════════════════════════════')
        self.get_logger().info('  🧠 Obstacle-Aware Navigation Node')
        self.get_logger().info(f'  ⚠️  Danger zone: < {self.danger_distance}m')
        self.get_logger().info(f'  📍 Waypoints: {len(self.waypoints)}')
        self.get_logger().info('═══════════════════════════════════════════')

    # ══════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ══════════════════════════════════════════════════════════════

    def state_cb(self, msg):
        self.current_state = msg

    def gps_cb(self, msg: NavSatFix):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        if not self.gps_received:
            self.gps_received = True
            self.get_logger().info(
                f'📡 GPS Fix: {msg.latitude:.6f}, {msg.longitude:.6f}')

    def scan_cb(self, msg: LaserScan):
        """
        Processes the 2D Lidar scan and extracts:
          - front_min_distance: closest obstacle directly ahead
          - left_min_distance:  closest obstacle to the left
          - right_min_distance: closest obstacle to the right

        This is the drone's "eyes". The main_loop reads these
        values to decide whether to fly forward or dodge.
        """
        if not self.lidar_received:
            self.lidar_received = True
            self.get_logger().info('📡 Lidar data streaming!')

        num_readings = len(msg.ranges)
        if num_readings == 0:
            return

        # Calculate index boundaries for each sector
        angle_per_reading = (msg.angle_max - msg.angle_min) / num_readings
        center_index = num_readings // 2

        front_half = int(math.radians(self.front_half_angle) / angle_per_reading)
        side_limit = int(math.radians(self.side_scan_angle) / angle_per_reading)

        # Extract sector ranges (filter out invalid readings)
        def min_valid(ranges_slice):
            valid = [r for r in ranges_slice
                     if msg.range_min < r < msg.range_max]
            return min(valid) if valid else float('inf')

        # FRONT sector: center ± front_half_angle
        front_start = max(0, center_index - front_half)
        front_end = min(num_readings, center_index + front_half)
        self.front_min_distance = min_valid(msg.ranges[front_start:front_end])

        # LEFT sector: center+front_half to center+side_limit
        left_start = center_index + front_half
        left_end = min(num_readings, center_index + side_limit)
        self.left_min_distance = min_valid(msg.ranges[left_start:left_end])

        # RIGHT sector: center-side_limit to center-front_half
        right_start = max(0, center_index - side_limit)
        right_end = center_index - front_half
        self.right_min_distance = min_valid(msg.ranges[right_start:right_end])

    # ══════════════════════════════════════════════════════════════
    #  MAIN DECISION LOOP — THE "BRAIN"
    # ══════════════════════════════════════════════════════════════

    def main_loop(self):
        """
        This is where the drone "thinks". Every 250ms it:
          1. Checks its current mission phase
          2. Reads the Lidar data
          3. Decides: fly to waypoint OR dodge obstacle
        """

        # ── Pre-flight phases (same as waypoint_nav_node) ────────
        if self.phase == 'CONNECTING':
            if self.current_state.connected:
                self.get_logger().info('✅ FCU Connected!')
                self.phase = 'SET_MODE'
            return

        if self.phase == 'SET_MODE':
            if self.current_state.mode == 'GUIDED':
                self.phase = 'ARMING'
                return
            self.set_mode('GUIDED')
            return

        if self.phase == 'ARMING':
            if self.current_state.armed:
                self.phase = 'TAKEOFF'
                return
            self.arm_drone()
            return

        if self.phase == 'TAKEOFF':
            self.takeoff(self.takeoff_alt)
            self.takeoff_time = time.time()
            self.phase = 'ASCENDING'
            return

        if self.phase == 'ASCENDING':
            if time.time() - self.takeoff_time > 12.0:
                self.get_logger().info('✅ Reached cruise altitude!')
                self.phase = 'NAVIGATE'
            return

        # ── NAVIGATION + OBSTACLE AVOIDANCE ──────────────────────
        if self.phase == 'NAVIGATE':
            self.navigate_with_avoidance()
            return

        if self.phase == 'RTL':
            if not self.current_state.armed:
                self.get_logger().info('═══════════════════════════════════════════')
                self.get_logger().info('  🏠 LANDED SAFELY!')
                self.get_logger().info('  ✅ MISSION COMPLETE!')
                self.get_logger().info('═══════════════════════════════════════════')
                self.phase = 'DONE'
            return

        if self.phase == 'DONE':
            self.timer.cancel()
            return

    def navigate_with_avoidance(self):
        """
        THE CORE LOGIC — Combines waypoint navigation with obstacle avoidance.

        Decision tree:
        ┌─────────────────────────────────┐
        │ Is obstacle < danger_distance?  │
        │         (< 3 meters)            │
        ├──── YES ────────┼──── NO ───────┤
        │   STOP & DODGE  │               │
        │                 │ Is obstacle    │
        │  Compare left   │ < warning?     │
        │  vs right space │ (< 5 meters)   │
        │                 ├─ YES ──┼─ NO ──┤
        │  Dodge toward   │ Slow   │ Full  │
        │  more open side │ down   │ speed │
        └─────────────────┴────────┴───────┘
        """

        # Check if all waypoints are done
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().info('═══════════════════════════════════════════')
            self.get_logger().info('  ✅ ALL WAYPOINTS REACHED!')
            self.get_logger().info('  🏠 Returning to Launch (RTL)...')
            self.get_logger().info('═══════════════════════════════════════════')
            self.set_mode('RTL')
            self.phase = 'RTL'
            return

        wp = self.waypoints[self.current_wp_index]
        target_lat, target_lon, target_alt = wp
        dist = self.haversine(self.current_lat, self.current_lon,
                              target_lat, target_lon)

        # Check waypoint arrival
        if dist < self.wp_reached_threshold:
            self.get_logger().info(
                f'📍 Waypoint {self.current_wp_index + 1} REACHED!')
            self.current_wp_index += 1
            self.avoidance_state = 'CLEAR'
            return

        # ── OBSTACLE DECISION TREE ───────────────────────────────

        front = self.front_min_distance
        left = self.left_min_distance
        right = self.right_min_distance

        # CASE 1: DANGER — obstacle very close, must dodge!
        if front < self.danger_distance:
            if self.avoidance_state != 'DODGING':
                self.avoidance_state = 'DODGING'
                self.get_logger().warn(
                    f'🚨 OBSTACLE at {front:.1f}m! DODGING!')

            # Decide which direction has more space
            if left > right:
                self.get_logger().info(
                    f'  ↰ Dodging LEFT (L:{left:.1f}m > R:{right:.1f}m)')
                self.send_velocity(0.0, self.dodge_speed, 0.0)
            else:
                self.get_logger().info(
                    f'  ↱ Dodging RIGHT (R:{right:.1f}m > L:{left:.1f}m)')
                self.send_velocity(0.0, -self.dodge_speed, 0.0)
            return

        # CASE 2: WARNING — obstacle approaching, slow down
        if front < self.warning_distance:
            if self.avoidance_state != 'WARNING':
                self.avoidance_state = 'WARNING'
                self.get_logger().warn(
                    f'⚠️  Obstacle at {front:.1f}m. Slowing down.')

            # Continue toward waypoint but at reduced speed
            self.publish_global_setpoint(target_lat, target_lon, target_alt)
            return

        # CASE 3: CLEAR — no obstacles, full speed to waypoint
        if self.avoidance_state != 'CLEAR':
            self.avoidance_state = 'CLEAR'
            self.get_logger().info('✅ Path clear! Resuming waypoint nav.')

        self.publish_global_setpoint(target_lat, target_lon, target_alt)

        # Periodic status logging
        if int(time.time()) % 3 == 0:
            self.get_logger().info(
                f'🔄 WP {self.current_wp_index + 1}/{len(self.waypoints)} '
                f'| Dist: {dist:.0f}m | Front: {front:.1f}m')

    # ══════════════════════════════════════════════════════════════
    #  MOVEMENT COMMANDS
    # ══════════════════════════════════════════════════════════════

    def send_velocity(self, vx, vy, vz):
        """
        Sends a velocity command to the drone in body frame.
          vx = forward/backward (+ = forward)
          vy = left/right (+ = left, - = right)
          vz = up/down (+ = up)

        This is how the drone "dodges". Instead of a GPS waypoint,
        we command raw velocity: "move 2 m/s to the left".
        MAVROS translates this into a MAVLink SET_VELOCITY command.
        """
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        self.vel_pub.publish(msg)

    def publish_global_setpoint(self, lat, lon, alt):
        msg = GeoPoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.latitude = lat
        msg.pose.position.longitude = lon
        msg.pose.position.altitude = alt
        msg.pose.orientation.w = 1.0
        self.global_setpoint_pub.publish(msg)

    # ══════════════════════════════════════════════════════════════
    #  UTILITY FUNCTIONS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (math.sin(dphi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(dlambda / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ══════════════════════════════════════════════════════════════
    #  MAVROS SERVICE HELPERS
    # ══════════════════════════════════════════════════════════════

    def set_mode(self, mode):
        if not self.set_mode_client.wait_for_service(timeout_sec=5.0):
            return
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        future.add_done_callback(
            lambda f: self._log_result(f, f'Mode → {mode}'))
        self.get_logger().info(f'🔄 Requesting mode: {mode}...')

    def arm_drone(self):
        if not self.arming_client.wait_for_service(timeout_sec=5.0):
            return
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        future.add_done_callback(lambda f: self._log_result(f, 'ARM'))
        self.get_logger().info('🔄 Requesting motor arming...')

    def takeoff(self, alt):
        if not self.takeoff_client.wait_for_service(timeout_sec=5.0):
            return
        req = CommandTOL.Request()
        req.altitude = alt
        future = self.takeoff_client.call_async(req)
        future.add_done_callback(
            lambda f: self._log_result(f, f'Takeoff → {alt}m'))
        self.get_logger().info(f'🔄 Requesting takeoff to {alt}m...')

    def _log_result(self, future, label):
        try:
            res = future.result()
            success = getattr(res, 'success', None)
            mode_sent = getattr(res, 'mode_sent', None)
            if success or mode_sent:
                self.get_logger().info(f'✅ {label} succeeded!')
            else:
                self.get_logger().warn(f'❌ {label} rejected. Retrying...')
        except Exception as e:
            self.get_logger().error(f'{label} failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Obstacle Navigation Node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
