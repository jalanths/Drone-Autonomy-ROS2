import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    try:
        pkg_dir = get_package_share_directory('drone_autonomy')
        nav2_params_path = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    except Exception:
        nav2_params_path = os.path.expanduser('~/Downloads/ROBO/drone_ws/src/drone_autonomy/config/nav2_params.yaml')

    # The static transform connecting the drone's body to the lidar.
    # We place the lidar dead center on top of the drone (z=0.1m up)
    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser_link'],
        output='screen'
    )

    # Costmap Node (instead of the full Nav2 stack, we just run the local costmap)
    costmap_node = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        output='screen',
        parameters=[nav2_params_path],
    )

    # Lifecycle Manager is required to activate the Costmap Node
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_costmap',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'autostart': True},
            {'node_names': ['costmap/costmap']},
            {'bond_timeout': 0.0}
        ]
    )

    return LaunchDescription([
        tf_node,
        costmap_node,
        lifecycle_manager_node
    ])
