import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'drone_autonomy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.rviz') + glob('config/*.parm')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jalanth',
    maintainer_email='capstone.aerix@gmail.com',
    description='Autonomous waypoint navigation with dynamic obstacle '
                'avoidance for the Tarot 650 (ROS 2 + MAVROS + Nav2 + Gazebo)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # ── Primary mission stack ──────────────────────────────────────
            'mission_avoidance_node = drone_autonomy.mission_avoidance_node:main',
            'odom_tf_broadcaster = drone_autonomy.odom_tf_broadcaster:main',
            'stream_rate_node = drone_autonomy.stream_rate_node:main',
            'dynamic_obstacle_node = drone_autonomy.dynamic_obstacle_node:main',

            # ── Simulation helper ─────────────────────────────────────────
            # Only needed for SITL without Gazebo; the Gazebo Iris model
            # carries a real ray sensor that publishes /scan directly.
            # NOTE: this entry point was previously registered under the name
            # `lidar_publisher_node`, pointing at a module that no longer
            # exists after the rename — `ros2 run` on it always failed.
            'virtual_lidar_node = drone_autonomy.virtual_lidar_node:main',

            # ── Earlier standalone demos (kept for reference) ─────────────
            'takeoff_node = drone_autonomy.takeoff_node:main',
            'waypoint_nav_node = drone_autonomy.waypoint_nav_node:main',
            'obstacle_nav_node = drone_autonomy.obstacle_nav_node:main',
            'nav2_obstacle_node = drone_autonomy.nav2_obstacle_node:main',
        ],
    },
)
