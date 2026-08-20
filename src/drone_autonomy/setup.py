from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'drone_autonomy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Install config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jalanth',
    maintainer_email='user@example.com',
    description='Autonomous logic for Tarot 650 with ROS 2 and MAVROS',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'takeoff_node = drone_autonomy.takeoff_node:main',
            'waypoint_nav_node = drone_autonomy.waypoint_nav_node:main',
            'virtual_lidar_node = drone_autonomy.virtual_lidar_node:main',
            'obstacle_nav_node = drone_autonomy.obstacle_nav_node:main',
            'nav2_obstacle_node = drone_autonomy.nav2_obstacle_node:main',
        ],
    },
)
