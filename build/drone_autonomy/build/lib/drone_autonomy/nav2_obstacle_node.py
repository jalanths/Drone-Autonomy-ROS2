#!/usr/bin/env python3
"""
Nav2 Obstacle-Aware Navigation Node (Version 2.0)
=================================================
This node subscribes to the 2D Costmap published by Nav2, rather than raw Lidar data.
By using the Costmap, we see "inflated" obstacles, ensuring the drone never clips
a wall with its propellers.

It searches the Costmap grid ahead of the drone. If the cost is too high (fatal obstacle),
it stops and dodges laterally to an area with lower cost.
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
import numpy as np


class Nav2ObstacleNode(Node):
    def __init__(self):
        super().__init__('nav2_obstacle_node')

        # ── Drone State ──────────────────────────────────────────
        self.current_state = State()
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.gps_received = False
        self.costmap_received = False

        # Costmap grid and metadata
        self.costmap = None
        self.map_resolution = 0.1
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

        # Nav points
        self.waypoints = [
            (-35.36280, 149.16530, 599.0),
            (-35.36350, 149.16600, 599.0),
            (-35.36380, 149.16450, 599.0),
        ]
        self.current_wp_index = 0
        self.wp_reached_threshold = 3.0

        self.phase = 'CONNECTING'
        self.takeoff_alt = 15.0
        self.takeoff_time = None
        
        # Avoidance State
        self.danger_cell_cost = 90  # 0-100 (100 = solid wall, >90 = inflated wall)
        self.dodge_speed = 2.0
        self.avoidance_state = 'CLEAR'

        # ── ROS 2 Setup ──────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        # Nav2 costmap publishes with TRANSIENT_LOCAL + RELIABLE
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
        self.get_logger().info('  🗺️  Nav2 Costmap Node Started (V2.0)')
        self.get_logger().info('═══════════════════════════════════════════')

    def state_cb(self, msg): self.current_state = msg
    
    def gps_cb(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        self.gps_received = True

    def costmap_cb(self, msg: OccupancyGrid):
        if not self.costmap_received:
            self.get_logger().info('🗺️ Nav2 Costmap Received!')
            self.costmap_received = True
            
        self.map_resolution = msg.info.resolution
        self.costmap = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

    def scan_costmap_sectors(self):
        """
        Analyzes the 2D Costmap matrix directly in front, left, and right of the drone.
        The local costmap is centered around the drone.
        """
        if self.costmap is None:
            return 0, 0, 0

        # costmap is a 2D numpy array [y, x] where the drone is perfectly in the center.
        # Cost is 0-100. -1 is unknown.
        height, width = self.costmap.shape
        cy, cx = height // 2, width // 2
        
        # Check a 2-meter grid (20 cells at 0.1m resolution) directly ahead (x direction)
        search_radius = 20
        # Use a 15-cell gap between sectors so inflated costs don't bleed across  
        gap = 15
        
        front_cells = self.costmap[cy - gap: cy + gap, cx: cx + search_radius]
        left_cells  = self.costmap[cy + gap: cy + search_radius, cx: cx + search_radius]
        right_cells = self.costmap[cy - search_radius: cy - gap, cx: cx + search_radius]
        
        def max_cost(cells):
            return np.max(cells) if cells.size > 0 else 0

        return max_cost(front_cells), max_cost(left_cells), max_cost(right_cells)

    def main_loop(self):
        if not self.gps_received or not self.costmap_received:
            return  # Wait for sensors

        if self.phase == 'CONNECTING':
            if self.current_state.connected: self.phase = 'SET_MODE'
            return
        if self.phase == 'SET_MODE':
            if self.current_state.mode == 'GUIDED': self.phase = 'ARMING'
            else: self.set_mode('GUIDED')
            return
        if self.phase == 'ARMING':
            if self.current_state.armed: self.phase = 'TAKEOFF'
            else: self.arm_drone()
            return
        if self.phase == 'TAKEOFF':
            self.takeoff(self.takeoff_alt)
            self.takeoff_time = time.time()
            self.phase = 'ASCENDING'
            return
        if self.phase == 'ASCENDING':
            if time.time() - self.takeoff_time > 12.0:
                self.phase = 'NAVIGATE'
            return

        if self.phase == 'NAVIGATE':
            # Are we done?
            if self.current_wp_index >= len(self.waypoints):
                self.get_logger().info('✅ ALL WAYPOINTS REACHED! Triggering RTL.')
                self.set_mode('RTL')
                self.phase = 'RTL'
                return

            target_lat, target_lon, target_alt = self.waypoints[self.current_wp_index]
            dist = self.haversine(self.current_lat, self.current_lon, target_lat, target_lon)

            if dist < self.wp_reached_threshold:
                self.get_logger().info(f'📍 Waypoint {self.current_wp_index + 1} REACHED!')
                self.current_wp_index += 1
                return

            # Nav2 Costmap Avoidance Logic
            front_cost, left_cost, right_cost = self.scan_costmap_sectors()

            if front_cost > self.danger_cell_cost:
                self.get_logger().warn(f'🚨 INFLATED WALL DETECTED AHEAD (Cost {front_cost}/100)! DODGING!')
                
                # Check if trapped!
                if left_cost >= self.danger_cell_cost and right_cost >= self.danger_cell_cost:
                    self.get_logger().warn(f'  ⬇️ Trapped! Dodging BACKWARDS (L:{left_cost} R:{right_cost})')
                    self.send_velocity(-self.dodge_speed, 0.0, 0.0)
                # Dodge to the side with lower cost
                elif left_cost < right_cost:
                    self.get_logger().info(f'  ↰ Dodging LEFT (L-Cost:{left_cost}  R-Cost:{right_cost})')
                    self.send_velocity(0.0, self.dodge_speed, 0.0)
                else:
                    self.get_logger().info(f'  ↱ Dodging RIGHT (R-Cost:{right_cost}  L-Cost:{left_cost})')
                    self.send_velocity(0.0, -self.dodge_speed, 0.0)
            else:
                if int(time.time()) % 3 == 0:
                    global_max = np.max(self.costmap) if self.costmap is not None else 0
                    self.get_logger().info(f'✅ Path clear. (MapMax:{global_max} F:{front_cost} L:{left_cost} R:{right_cost}) Navigatng to WP {self.current_wp_index + 1}. Dist: {int(dist)}m')
                    
                # If we're safely far from waypoints, send global GPS setpoints(target_lat, target_lon, target_alt)
                self.publish_global_setpoint(target_lat, target_lon, target_alt)

    # ── Helpers (same as before) ───────────────────────────────────
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
        msg.pose.position.latitude, msg.pose.position.longitude, msg.pose.position.altitude = lat, lon, alt
        msg.pose.orientation.w = 1.0
        self.global_setpoint_pub.publish(msg)

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
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
