"""
Simulation layer: Gazebo Classic 11 + MAVROS + dynamic obstacles.

Brings up everything that talks to the simulator, but NOT the autonomy stack
(see autonomy_launch.py) and NOT ArduPilot SITL itself, which has to run in
its own terminal because it is interactive.

Ordering matters: Gazebo must be listening on UDP 9002/9003 before SITL
starts, because ArduPilotPlugin is the server side of that FDM link.

    Terminal 1:  ros2 launch drone_autonomy sim_launch.py
    Terminal 2:  cd ~/ardupilot/ArduCopter && sim_vehicle.py -f gazebo-iris --console --map
    Terminal 3:  ros2 launch drone_autonomy autonomy_launch.py

Or just run scripts/run_mission.sh, which sequences all three.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('drone_autonomy')
    mavros_share = get_package_share_directory('mavros')

    default_world = os.path.join(pkg_share, 'worlds', 'drone_city_dynamic.world')

    world_arg = DeclareLaunchArgument(
        'world', default_value=default_world,
        description='SDF world to load')
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='true',
        description='Run the Gazebo GUI (false = headless gzserver only)')
    # ArduPilot SITL serves MAVLink on TCP 5760 directly. The familiar
    # udp://127.0.0.1:14550 endpoint only exists when MAVProxy is running
    # (i.e. when you started SITL via sim_vehicle.py) — MAVProxy is what
    # forwards 5760 out to 14550.
    #
    # This matters more than it looks: ArduPilot SITL parks at
    # "Waiting for connection ...." and does NOT run its main loop until a
    # MAVLink client attaches. No client -> no servo packets to Gazebo ->
    # the ArduPilotPlugin logs "Broken ArduPilot connection" forever and the
    # drone never spawns its FDM link. Pointing MAVROS at a dead UDP port
    # therefore looks like a Gazebo failure when it is really a MAVLink one.
    #
    #   scripts/run_mission.sh           -> arducopter binary, tcp://…:5760
    #   sim_vehicle.py (MAVProxy console) -> udp://127.0.0.1:14550@
    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url', default_value='tcp://127.0.0.1:5760',
        description='MAVLink endpoint. tcp://127.0.0.1:5760 for a raw SITL '
                    'binary; udp://127.0.0.1:14550@ when using sim_vehicle.py')
    dynamic_arg = DeclareLaunchArgument(
        'dynamic_obstacles', default_value='true',
        description='Drive the moving blocks across the route')

    # ── Gazebo ────────────────────────────────────────────────────────────
    # gazebo_ros' launch file loads libgazebo_ros_init.so, which publishes
    # /clock. Every node in this project runs use_sim_time:=true, so without
    # that plugin the whole stack would sit frozen at t=0.
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('gazebo_ros'),
                                  'launch', 'gazebo.launch.py'])),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'verbose': 'true',
            'gui': LaunchConfiguration('gui'),
            'init': 'true',
            'factory': 'true',
        }.items())

    # ── MAVROS ────────────────────────────────────────────────────────────
    # config/mavros_pluginlists.yaml is REQUIRED, not optional tuning: stock
    # mavros_node aborts on startup in this build because several plugins
    # collide on the same topic names. That file pins the working allowlist —
    # read its header before changing anything in it.
    mavros = Node(
        package='mavros', executable='mavros_node', name='mavros',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'mavros_pluginlists.yaml'),
            os.path.join(mavros_share, 'launch', 'apm_config.yaml'),
            {
                'fcu_url': LaunchConfiguration('fcu_url'),
                'gcs_url': '',
                'target_system_id': 1,
                'target_component_id': 1,
                'use_sim_time': True,
            },
        ],
        # ══════════════════════════════════════════════════════════════════
        #  MANDATORY REMAPPINGS — this MAVROS build flattens plugin topics
        # ══════════════════════════════════════════════════════════════════
        # Each MAVROS plugin is supposed to own a sub-namespace, giving the
        # documented names (/mavros/local_position/pose, /mavros/cmd/arming,
        # …). In this build that sub-namespacing is broken: every plugin
        # creates its endpoints directly under the UAS node, so they land as
        #
        #     /mavros/mavros/pose                (want /mavros/local_position/pose)
        #     /mavros/mavros/global              (want /mavros/global_position/global)
        #     /mavros/mavros/cmd_vel_unstamped   (want /mavros/setpoint_velocity/…)
        #     /mavros/mavros/arming              (want /mavros/cmd/arming)
        #
        # Verified with `ros2 node info /mavros/mavros`. The damage is not
        # limited to our code — MAVROS breaks its OWN wiring this way: the
        # version plugin registers a client on /mavros/cmd/command while the
        # command plugin serves /mavros/mavros/command, which is the source of
        # the "VER: command plugin service call failed!" spam on startup.
        #
        # It is also the reason config/mavros_pluginlists.yaml is needed:
        # flattening makes plugin leaf names collide (three plugins claim
        # `status`, two claim `local`), which aborts the process outright.
        #
        # These remappings restore the documented names, so every downstream
        # node — and MAVROS itself — finds what it expects.
        remappings=[
            # Telemetry out
            ('/mavros/mavros/pose',   '/mavros/local_position/pose'),
            ('/mavros/mavros/global', '/mavros/global_position/global'),
            ('/mavros/mavros/data',   '/mavros/imu/data'),
            # Control in
            ('/mavros/mavros/cmd_vel_unstamped',
             '/mavros/setpoint_velocity/cmd_vel_unstamped'),
            ('/mavros/mavros/cmd_vel', '/mavros/setpoint_velocity/cmd_vel'),
            # Services
            ('/mavros/mavros/arming',  '/mavros/cmd/arming'),
            ('/mavros/mavros/takeoff', '/mavros/cmd/takeoff'),
            ('/mavros/mavros/land',    '/mavros/cmd/land'),
            ('/mavros/mavros/command', '/mavros/cmd/command'),
        ])

    # ── MAVLink stream rates ──────────────────────────────────────────────
    # MANDATORY. ArduPilot sends only HEARTBEAT until a GCS requests streams,
    # and MAVROS 2 never does. Without this node /mavros/local_position/pose
    # stays silent, the odom->base_link TF is never published, and the Nav2
    # costmap discards every laser return. See the node's docstring.
    stream_rates = Node(
        package='drone_autonomy', executable='stream_rate_node',
        name='stream_rate_node', output='screen',
        parameters=[{'use_sim_time': True, 'rate_hz': 20}])

    # ── Moving obstacles ──────────────────────────────────────────────────
    dynamic_obstacles = Node(
        package='drone_autonomy', executable='dynamic_obstacle_node',
        name='dynamic_obstacle_node', output='screen',
        condition=IfCondition(LaunchConfiguration('dynamic_obstacles')),
        # Kept level with cruise_altitude. If the blocks sit below the drone
        # they are simply overflown and dynamic avoidance is never exercised.
        parameters=[{'use_sim_time': True, 'altitude': 4.0, 'speed': 1.0}])

    return LaunchDescription([
        world_arg, gui_arg, fcu_url_arg, dynamic_arg,
        gazebo,
        mavros,
        stream_rates,
        dynamic_obstacles,
    ])
