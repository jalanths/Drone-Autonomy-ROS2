#!/usr/bin/env python3
"""
odom -> base_link TF broadcaster
================================
Nav2's costmap needs a TF chain from its global frame (`odom`) down to the
sensor frame before it will accept a single laser return:

    odom --(this node)--> base_link --(static TF)--> laser_link

MAVROS publishes the drone's ENU pose on /mavros/local_position/pose but does
NOT broadcast it as a transform, so without this node the chain is broken and
the costmap stays empty.

TIMESTAMP NOTE
--------------
The transform is stamped with the ORIGINAL message stamp, not `now()`.
Re-stamping with the current clock looks harmless but quietly breaks the
costmap: the Gazebo LiDAR stamps /scan with sim time, and if this TF is
stamped from a slightly different clock reading, tf2 has to extrapolate and
rejects the scan with "Lookup would require extrapolation into the future".
Preserving the source stamp keeps pose and scan on the same timeline.
"""

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.count = 0

        # MAVROS publishes sensor data BEST_EFFORT. Subscribing RELIABLE
        # silently matches nothing and drops every message.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos)

        self.get_logger().info('🚀 Odometry TF broadcaster started (odom -> base_link)')

    def pose_cb(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp      # see module docstring
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation

        self.tf_broadcaster.sendTransform(t)

        if self.count == 0:
            self.get_logger().info(
                f'✅ First transform published — drone at ENU '
                f'({msg.pose.position.x:.1f}, {msg.pose.position.y:.1f}, '
                f'{msg.pose.position.z:.1f})')
        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBroadcaster()
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
