"""
Autonomy layer: TF chain -> Nav2 costmap -> avoidance brain -> RViz2.

Run this AFTER sim_launch.py and ArduPilot SITL are both up.

    ros2 launch drone_autonomy autonomy_launch.py

THE TF CHAIN (this is what makes or breaks the costmap)
-------------------------------------------------------
    odom --[odom_tf_broadcaster]--> base_link --[static]--> laser_link

Nav2 refuses every laser return unless it can resolve the scan's frame_id
all the way up to the costmap's global frame. Break any link and the costmap
silently stays empty — no error, just no obstacles, and a drone that flies
straight into a wall.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('drone_autonomy')
    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    mission_params = os.path.join(pkg_share, 'config', 'mission_params.yaml')
    rviz_config = os.path.join(pkg_share, 'config', 'drone_mission.rviz')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Launch RViz2')
    params_arg = DeclareLaunchArgument(
        'mission_params', default_value=mission_params,
        description='YAML of waypoints / avoidance tuning')

    sim_time = {'use_sim_time': True}

    # ── TF: odom -> base_link (from the MAVROS local pose) ────────────────
    odom_tf = Node(
        package='drone_autonomy', executable='odom_tf_broadcaster',
        name='odom_tf_broadcaster', output='screen', parameters=[sim_time])

    # ── TF: base_link -> laser_link ───────────────────────────────────────
    # 0.25 m matches <pose>0 0 0.25</pose> on the laser_link in the Iris SDF.
    laser_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_base_to_laser', output='screen',
        arguments=['0', '0', '0.25', '0', '0', '0', 'base_link', 'laser_link'],
        parameters=[sim_time])

    # Belt-and-braces: Gazebo Classic sometimes prefixes a nested model's
    # frame with the parent model name. The SDF sets <frame_name>laser_link
    # </frame_name> explicitly so this should be unused, but an extra static
    # TF costs nothing and prevents a silent empty costmap if it ever is.
    laser_tf_prefixed = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='tf_base_to_laser_prefixed', output='screen',
        arguments=['0', '0', '0.25', '0', '0', '0',
                   'base_link', 'iris_demo::laser_link'],
        parameters=[sim_time])

    # ── Nav2 costmap ──────────────────────────────────────────────────────
    # The standalone executable builds Costmap2DROS("costmap"), and that
    # constructor puts the lifecycle node in a namespace equal to its own
    # name. Verified with `ros2 node list`, the node is therefore:
    #
    #     /costmap/costmap    -> publishes /costmap/costmap
    #
    # which is why nav2_params.yaml nests its key as `costmap: costmap:` and
    # why the lifecycle manager below must be given 'costmap/costmap'. Passing
    # plain 'costmap' leaves the manager stuck on
    # "Waiting for service costmap/get_state..." forever, and because a
    # lifecycle node publishes nothing until it is activated, the costmap
    # topic never appears at all.
    costmap = Node(
        package='nav2_costmap_2d', executable='nav2_costmap_2d',
        name='costmap', output='screen',
        parameters=[nav2_params, sim_time])

    # A lifecycle node does nothing at all until something configures and
    # activates it — that is this manager's entire job.
    lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_costmap', output='screen',
        parameters=[sim_time, {
            'autostart': True,
            'node_names': ['costmap/costmap'],
            'bond_timeout': 0.0,
        }])

    # ── The avoidance brain ───────────────────────────────────────────────
    mission = Node(
        package='drone_autonomy', executable='mission_avoidance_node',
        name='mission_avoidance_node', output='screen',
        parameters=[LaunchConfiguration('mission_params'), sim_time])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[sim_time])

    return LaunchDescription([
        rviz_arg, params_arg,
        odom_tf, laser_tf, laser_tf_prefixed,
        costmap, lifecycle,
        mission,
        rviz,
    ])
