#!/usr/bin/env python3
"""
Dynamic Obstacle Driver
=======================
Moves the `dyn_block_*` models in drone_city_dynamic.world back and forth
across the mission route, so the avoidance logic has to cope with obstacles
that MOVE rather than a frozen map it could have memorised.

WHY A SERVICE INSTEAD OF A GAZEBO <actor>
-----------------------------------------
Gazebo Classic's <actor> animation moves only the visual skeleton — the
collision geometry stays at the origin. A ray sensor tests against COLLISION
geometry, so an actor is completely invisible to the LiDAR: it would look
convincing in the GUI while the drone flew straight through it. Driving real
<model> entities through /gazebo/set_entity_state moves collisions too, so
the obstacles genuinely register on /scan and land in the costmap.

Each block patrols a straight segment on a triangle wave, at the drone's
cruise altitude, positioned to cut across a different mission leg.
"""

import math

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node


class DynamicObstacleNode(Node):

    # name, (ax, ay), (bx, by)   — all in GAZEBO world coordinates.
    # Traverse SPEED is a parameter (see `speed`), not a hardcoded period, so
    # the obstacles stay matched to what the LiDAR can actually resolve.
    PATROLS = [
        ('dyn_block_0', (20.0, -14.0), (20.0,  14.0)),   # cuts HOME -> WP1
        ('dyn_block_1', (20.0,  44.0), (20.0,  68.0)),   # cuts WP3  -> WP4
        ('dyn_block_2', (-25.0, 36.0), (-25.0, 60.0)),   # cuts WP4  -> WP5
        ('dyn_block_3', (-40.0,  4.0), (-18.0,   4.0)),  # cuts WP5  -> WP6
    ]

    def __init__(self):
        super().__init__('dynamic_obstacle_node')

        self.declare_parameter('altitude', 4.0)
        self.declare_parameter('enabled', True)
        # Traverse speed, matched to the sensor that has to track these.
        #
        # These used to run at 2.0-3.5 m/s. The RPLiDAR turns at 10 Hz (and the
        # simulation was measured delivering ~5.4 Hz), so the fastest block
        # jumped 0.65 m between consecutive scans, and against a drone doing
        # 2.5 m/s the closing rate reached ~6 m/s. The avoidance needs a few
        # consecutive frames to confirm a threat before acting, by which point
        # such an obstacle has already covered metres — it was effectively
        # unresolvable rather than merely difficult.
        #
        # 1.0 m/s is a brisk walking pace: it moves 0.1 m per scan, so the
        # drone gets roughly ten times as many looks at it on approach.
        self.declare_parameter('speed', 1.0)             # m/s
        self.altitude = self.get_parameter('altitude').value
        self.enabled = self.get_parameter('enabled').value
        self.speed = max(0.05, float(self.get_parameter('speed').value))

        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.t0 = None
        self.ready = False
        self.warned = False

        self.create_timer(0.05, self.tick)      # 20 Hz

        self.get_logger().info('═══════════════════════════════════════════════')
        self.get_logger().info('  🚧 Dynamic Obstacle Driver')
        self.get_logger().info(f'  📦 {len(self.PATROLS)} moving blocks @ '
                               f'{self.altitude:.1f} m, {self.speed:.1f} m/s')
        self.get_logger().info('═══════════════════════════════════════════════')

    def tick(self):
        if not self.enabled:
            return

        if not self.ready:
            if not self.cli.service_is_ready():
                if not self.warned:
                    self.warned = True
                    self.get_logger().info(
                        '⏳ Waiting for /gazebo/set_entity_state '
                        '(is libgazebo_ros_state.so loaded in the world?)')
                return
            self.ready = True
            self.get_logger().info('✅ Connected to Gazebo — obstacles now moving.')

        if self.t0 is None:
            self.t0 = self.now()
        t = self.now() - self.t0

        for name, a, b in self.PATROLS:
            # Period is derived from the requested speed: one full there-and-back
            # covers the path twice.
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            period = max(1.0, 2.0 * length / self.speed)
            # Triangle wave in [0, 1]: smooth there-and-back with no snap.
            phase = (t % period) / period
            s = 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)

            x = a[0] + (b[0] - a[0]) * s
            y = a[1] + (b[1] - a[1]) * s

            req = SetEntityState.Request()
            st = EntityState()
            st.name = name
            st.reference_frame = 'world'
            st.pose.position.x = float(x)
            st.pose.position.y = float(y)
            st.pose.position.z = float(self.altitude)
            st.pose.orientation.w = 1.0
            req.state = st
            self.cli.call_async(req)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleNode()
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
