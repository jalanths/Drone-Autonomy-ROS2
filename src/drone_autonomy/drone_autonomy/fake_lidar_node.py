#!/usr/bin/env python3
"""
Simulated Lidar Publisher for SITL Testing
===========================================
Since ArduPilot SITL (without Gazebo) doesn't provide a simulated Lidar,
this node publishes fake LaserScan data on /scan to test obstacle avoidance.

It simulates a 2D lidar (like the RPLiDAR C1M1-R2) with:
  - 360-degree field of view
  - 12m max range
  - A virtual "wall" obstacle that appears at a configurable distance
    directly ahead of the drone

When the real RPLiDAR is connected, you simply stop running this node
and launch sllidar_ros2 instead. The /scan topic format is identical.

Usage:
  python3 fake_lidar_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import time


class FakeLidarNode(Node):
    def __init__(self):
        super().__init__('fake_lidar_node')

        # ── Lidar specs (matching RPLiDAR C1M1-R2) ──────────────
        self.angle_min = -math.pi        # -180 degrees
        self.angle_max = math.pi         # +180 degrees
        self.num_readings = 360          # 1 degree resolution
        self.angle_increment = (self.angle_max - self.angle_min) / self.num_readings
        self.range_min = 0.15            # 15cm minimum range
        self.range_max = 12.0            # 12m maximum range
        self.scan_rate = 10.0            # 10 Hz scan rate

        # ── Dynamic Obstacle Sequencing ─────────────────────────────────
        # We will dynamically spawn obstacles based on elapsed time to force
        # the drone to dodge left, right, and backwards!

        # ── Timing ───────────────────────────────────────────────
        self.start_time = time.time()

        # ── Publisher ────────────────────────────────────────────
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.timer = self.create_timer(1.0 / self.scan_rate, self.publish_scan)

        self.get_logger().info('═══════════════════════════════════════════')
        self.get_logger().info('  📡 Fake Lidar Node — ENHANCED')
        self.get_logger().info('  🧱 Dynamic sequenced obstacles active!')
        self.get_logger().info('  Obstacles will cycle every 80 seconds...')
        self.get_logger().info('═══════════════════════════════════════════')

    def publish_scan(self):
        """Publishes one complete 360-degree LaserScan message."""
        elapsed = time.time() - self.start_time

        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'laser_link'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.time_increment = (1.0 / self.scan_rate) / self.num_readings
        scan.scan_time = 1.0 / self.scan_rate
        scan.range_min = self.range_min
        scan.range_max = self.range_max

        # Use 11.9m instead of 12.0m to guarantee Nav2 clears the path without treating it as out-of-bounds noise
        ranges = [11.9] * self.num_readings

        # ── Dynamic Obstacle Logic ──
        # Time-based sequence (loops every 80 seconds)
        cycle_time = elapsed % 80.0
        
        obs_active = False
        center = 180
        width = 0
        dist = 2.0

        if 10.0 <= cycle_time < 20.0:
            # Block Right-Front -> Forces Dodge LEFT
            obs_active = True
            center, width, dist = 145, 60, 2.0
            if cycle_time - 10.0 < 0.15: self.get_logger().info('🧱 SCRIPT: Blocking RIGHT, forcing left dodge!')
        
        elif 30.0 <= cycle_time < 40.0:
            # Block Left-Front -> Forces Dodge RIGHT
            obs_active = True
            center, width, dist = 215, 60, 2.0
            if cycle_time - 30.0 < 0.15: self.get_logger().info('🧱 SCRIPT: Blocking LEFT, forcing right dodge!')
            
        elif 50.0 <= cycle_time < 65.0:
            # Block Front, Left, AND Right -> Forces Dodge BACKWARDS
            obs_active = True
            center, width, dist = 180, 160, 1.8
            if cycle_time - 50.0 < 0.15: self.get_logger().info('🧱 SCRIPT: Trapping FRONT/LEFT/RIGHT, forcing BACKWARDS dodge!')
            
        elif 65.0 <= cycle_time < 80.0:
            # Clear sky
            pass
            if cycle_time - 65.0 < 0.15: self.get_logger().info('✅ SCRIPT: Clear skies!')

        if obs_active:
            half_w = width // 2
            for i in range(center - half_w, center + half_w):
                idx = i % self.num_readings
                noise = 0.05 * math.sin(idx * 0.5 + elapsed * 3)
                wall_dist = max(self.range_min, dist + noise)
                ranges[idx] = wall_dist

        scan.ranges = ranges
        scan.intensities = [100.0] * self.num_readings
        self.scan_pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = FakeLidarNode()

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
