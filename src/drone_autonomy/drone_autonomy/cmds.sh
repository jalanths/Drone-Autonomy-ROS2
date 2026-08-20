# Terminal 1 — ArduPilot Simulator
cd ~/ardupilot/ArduCopter && sim_vehicle.py --console --map
# Terminal 2 — MAVROS Bridge
ros2 run mavros mavros_node --ros-args -p fcu_url:="udp://127.0.0.1:14550@"
# Terminal 3 — Fake Lidar
cd ~/Downloads/ROBO/drone_ws && . install/setup.bash
ros2 run drone_autonomy virtual_lidar_node
# Terminal 4 — Nav2 Costmap Generator
cd ~/Downloads/ROBO/drone_ws && . install/setup.bash
ros2 launch drone_autonomy nav2_drone_launch.py
# Terminal 5 — Brain Node V2.0 (Nav2 Costmap Avoidance)
cd ~/Downloads/ROBO/drone_ws && . install/setup.bash
ros2 run drone_autonomy nav2_obstacle_node
# Terminal 6 — RViz2 Visualizer
rviz2