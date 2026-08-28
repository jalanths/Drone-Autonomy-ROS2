#!/usr/bin/env python3
"""
Autonomous Random Explorer with Obstacle Avoidance (V3.0)
=========================================================
The drone takes off to 4m, then randomly explores the entire Gazebo world
forever. It picks a random GPS target within ~80m, flies towards it, and
when the Nav2 Costmap detects an obstacle ahead it autonomously dodges
left/right/back/up. Once the path is clear, it picks a NEW random target
and keeps exploring. It never stops — Ctrl+C to land.

Algorithm:
  1. Take off to 4m
  2. Pick a random GPS target within EXPLORE_RADIUS
  3. Fly toward target
  4. If obstacle detected in costmap → DODGE (left/right/back/up)
  5. After dodge or target reached → pick new random target → goto 3
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from geographic_msgs.msg import GeoPoseStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import OccupancyGrid
import math
import time
import random
import numpy as np


class Nav2ObstacleNode(Node):
    def __init__(self):
        super().__init__('nav2_obstacle_node')

        # ── Drone State ──────────────────────────────────────────
        self.current_state = State()
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.home_lat = 0.0
        self.home_lon = 0.0
        self.home_alt = 0.0
        self.gps_received = False
        self.costmap_received = False

        # Costmap grid and metadata
        self.costmap = None
        self.map_resolution = 0.1

        # ── Explorer Config ──────────────────────────────────────
        self.phase = 'CONNECTING'
        self.takeoff_alt = 4.0
        self.takeoff_time = None
        self.cruise_alt = 0.0  # Set from home_alt + takeoff_alt after GPS fix

        # Random exploration parameters
        self.EXPLORE_RADIUS = 80.0       # meters: max distance for random target
        self.EXPLORE_MIN_RADIUS = 20.0   # meters: min distance for random target
        self.target_lat = 0.0
        self.target_lon = 0.0
        self.target_reached_threshold = 8.0  # meters
        self.targets_visited = 0
        self.explore_start_time = None

        # Avoidance State
        self.danger_cell_cost = 90
        self.dodge_speed = 2.5
        self.is_dodging = False
        self.dodge_start_time = None
        self.DODGE_DURATION = 2.0  # seconds to dodge before re-evaluating
        self.consecutive_dodges = 0

        # Stats
        self.total_distance = 0.0
        self.last_stat_lat = None
        self.last_stat_lon = None

        # ── ROS 2 Setup ──────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_cb, sensor_qos)
        self.create_subscription(OccupancyGrid, '/costmap/costmap', self.costmap_cb, map_qos)

        self.global_setpoint_pub = self.create_publisher(GeoPoseStamped, '/mavros/setpoint_position/global', 10)
        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        self.timer = self.create_timer(0.25, self.main_loop)

        self.get_logger().info('═══════════════════════════════════════════')
        self.get_logger().info('  🌍 Autonomous Random Explorer V3.0')
        self.get_logger().info('  🔭 Mode: Infinite Random Exploration')
        self.get_logger().info('  📏 Altitude: 4m  |  Radius: 80m')
        self.get_logger().info('  🛡️ Obstacle Avoidance: Nav2 Costmap')
        self.get_logger().info('═══════════════════════════════════════════')

    # ══════════════════════════════════════════════════════════════
    #  Callbacks
    # ══════════════════════════════════════════════════════════════

    def state_cb(self, msg):
        self.current_state = msg

    def gps_cb(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        if not self.gps_received:
            self.gps_received = True
            self.home_lat = msg.latitude
            self.home_lon = msg.longitude
            self.home_alt = msg.altitude
            self.cruise_alt = self.home_alt + self.takeoff_alt
            self.get_logger().info(f'📡 GPS Fix: {msg.latitude:.6f}, {msg.longitude:.6f}, Alt: {msg.altitude:.1f}m AMSL')
            self.get_logger().info(f'📏 Home Alt: {self.home_alt:.1f}m | Cruise Alt: {self.cruise_alt:.1f}m (home + {self.takeoff_alt}m)')

    def costmap_cb(self, msg: OccupancyGrid):
        if not self.costmap_received:
            self.get_logger().info('🗺️ Nav2 Costmap Received!')
            self.costmap_received = True
        self.map_resolution = msg.info.resolution
        self.costmap = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

    # ══════════════════════════════════════════════════════════════
    #  Random Target Generator
    # ══════════════════════════════════════════════════════════════

    def pick_random_target(self):
        """Pick a random GPS coordinate within EXPLORE_RADIUS of home."""
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(self.EXPLORE_MIN_RADIUS, self.EXPLORE_RADIUS)

        # Convert meters to lat/lon offset
        dlat = (dist * math.cos(angle)) / 111111.0
        dlon = (dist * math.sin(angle)) / (111111.0 * math.cos(math.radians(self.home_lat)))

        self.target_lat = self.home_lat + dlat
        self.target_lon = self.home_lon + dlon

        bearing_deg = math.degrees(angle) % 360
        compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int(bearing_deg / 45) % 8]

        self.get_logger().info(
            f'🎯 NEW TARGET #{self.targets_visited + 1}: '
            f'{dist:.0f}m {compass} (bearing {bearing_deg:.0f}°) | '
            f'GPS: ({self.target_lat:.6f}, {self.target_lon:.6f})')

    # ══════════════════════════════════════════════════════════════
    #  Costmap Obstacle Scanner
    # ══════════════════════════════════════════════════════════════

    def scan_costmap_sectors(self):
        """Scan front/left/right/back sectors of the local costmap."""
        if self.costmap is None:
            return 0, 0, 0, 0

        height, width = self.costmap.shape
        cy, cx = height // 2, width // 2

        search_radius = 80  # 8m lookahead at 0.1m resolution
        gap = 10  # ±1.0m corridor width

        front_cells = self.costmap[cy - gap: cy + gap, cx: cx + search_radius]
        left_cells  = self.costmap[cy + gap: cy + search_radius, cx - search_radius: cx + search_radius]
        right_cells = self.costmap[cy - search_radius: cy - gap, cx - search_radius: cx + search_radius]
        back_cells  = self.costmap[cy - gap: cy + gap, cx - search_radius: cx]

        def max_cost(cells):
            return int(np.max(cells)) if cells.size > 0 else 0

        return max_cost(front_cells), max_cost(left_cells), max_cost(right_cells), max_cost(back_cells)

    # ══════════════════════════════════════════════════════════════
    #  Main Loop — State Machine
    # ══════════════════════════════════════════════════════════════

    def main_loop(self):
        if not self.gps_received or not self.costmap_received:
            return

        # ── Startup Phases ────────────────────────────────────────
        if self.phase == 'CONNECTING':
            if self.current_state.connected:
                self.phase = 'SET_MODE'
            return
        if self.phase == 'SET_MODE':
            if self.current_state.mode == 'GUIDED':
                self.phase = 'ARMING'
            else:
                self.set_mode('GUIDED')
            return
        if self.phase == 'ARMING':
            if self.current_state.armed:
                self.phase = 'TAKEOFF'
            else:
                self.arm_drone()
            return
        if self.phase == 'TAKEOFF':
            self.takeoff(self.takeoff_alt)
            self.takeoff_time = time.time()
            self.phase = 'ASCENDING'
            return
        if self.phase == 'ASCENDING':
            if time.time() - self.takeoff_time > 10.0:
                # cruise_alt already set from home_alt + takeoff_alt in gps_cb
                self.explore_start_time = time.time()
                self.pick_random_target()
                self.phase = 'EXPLORE'
                self.get_logger().info(f'🚀 EXPLORATION STARTED! Cruise alt: {self.cruise_alt:.1f}m AMSL')
            return

        # ── EXPLORE: The main autonomous exploration loop ─────────
        if self.phase == 'EXPLORE':
            # Track distance traveled
            if self.last_stat_lat is not None:
                d = self.haversine(self.current_lat, self.current_lon, self.last_stat_lat, self.last_stat_lon)
                if d > 1.0:
                    self.total_distance += d
                    self.last_stat_lat = self.current_lat
                    self.last_stat_lon = self.current_lon
            else:
                self.last_stat_lat = self.current_lat
                self.last_stat_lon = self.current_lon

            # Check if current target reached
            dist_to_target = self.haversine(self.current_lat, self.current_lon, self.target_lat, self.target_lon)
            if dist_to_target < self.target_reached_threshold:
                self.targets_visited += 1
                elapsed = time.time() - self.explore_start_time
                self.get_logger().info(
                    f'📍 TARGET #{self.targets_visited} REACHED! '
                    f'[Total: {self.total_distance:.0f}m flown in {elapsed:.0f}s] '
                    f'Picking new random target...')
                self.consecutive_dodges = 0
                self.pick_random_target()
                return

            # ── Obstacle Avoidance ────────────────────────────────
            front_cost, left_cost, right_cost, back_cost = self.scan_costmap_sectors()

            if front_cost > self.danger_cell_cost:
                self.is_dodging = True
                self.dodge_start_time = time.time()
                self.consecutive_dodges += 1

                self.get_logger().warn(
                    f'🚨 OBSTACLE AHEAD! (F:{front_cost} L:{left_cost} R:{right_cost} B:{back_cost}) '
                    f'Dodge #{self.consecutive_dodges}')

                # If stuck after many consecutive dodges, pick a completely new target
                if self.consecutive_dodges > 8:
                    self.get_logger().warn('🔄 TOO MANY DODGES! Picking completely new random target...')
                    self.consecutive_dodges = 0
                    self.pick_random_target()
                    return

                # Dodge logic: escape to the safest direction
                if left_cost >= self.danger_cell_cost and right_cost >= self.danger_cell_cost:
                    if back_cost >= self.danger_cell_cost:
                        self.get_logger().warn('  🆘 TRAPPED 360°! Escaping UPWARD!')
                        self.send_velocity(0.0, 0.0, 1.0)
                    else:
                        self.get_logger().warn('  ⬅️ Front+Sides blocked! Reversing...')
                        self.send_velocity(-self.dodge_speed, 0.0, 0.0)
                elif left_cost < right_cost:
                    self.get_logger().info(f'  ↰ Dodging LEFT (L:{left_cost} < R:{right_cost})')
                    self.send_velocity(0.5, self.dodge_speed, 0.0)
                else:
                    self.get_logger().info(f'  ↱ Dodging RIGHT (R:{right_cost} < L:{left_cost})')
                    self.send_velocity(0.5, -self.dodge_speed, 0.0)

            else:
                # Path is clear — fly toward random target
                if self.is_dodging:
                    self.is_dodging = False
                    self.get_logger().info('✅ Path CLEAR after dodge! Resuming exploration.')

                if self.consecutive_dodges > 0 and not self.is_dodging:
                    self.consecutive_dodges = max(0, self.consecutive_dodges - 1)

                # Print status every ~3 seconds
                if int(time.time()) % 3 == 0:
                    home_dist = self.haversine(self.current_lat, self.current_lon, self.home_lat, self.home_lon)
                    global_max = int(np.max(self.costmap)) if self.costmap is not None else 0
                    self.get_logger().info(
                        f'🔭 EXPLORING → Target #{self.targets_visited + 1} | '
                        f'Dist: {dist_to_target:.0f}m | '
                        f'FromHome: {home_dist:.0f}m | '
                        f'MapMax: {global_max} | '
                        f'Total: {self.total_distance:.0f}m flown')

                self.publish_global_setpoint(self.target_lat, self.target_lon, self.cruise_alt)

    # ══════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════

    def send_velocity(self, vx, vy, vz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
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

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def set_mode(self, mode):
        req = SetMode.Request(custom_mode=mode)
        self.set_mode_client.call_async(req)

    def arm_drone(self):
        req = CommandBool.Request(value=True)
        self.arming_client.call_async(req)

    def takeoff(self, alt):
        req = CommandTOL.Request(altitude=alt)
        self.takeoff_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = Nav2ObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
