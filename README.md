# Drone Autonomy in ROS 2 🚁

A complete ROS 2 autonomous drone navigation framework utilizing ArduPilot, MAVROS, and Nav2 for real-time path planning and dynamic obstacle avoidance. 

## 🌟 Key Features
- **ROS 2 Integration:** Fully built on the ROS 2 ecosystem.
- **ArduCopter & MAVROS:** Seamless flight control and telemetry streaming to offboard navigation nodes.
- **Nav2 Autonomy Stack:** Real-time path planning globally and locally using the Nav2 framework.
- **Dynamic Obstacle Avoidance:** Utilizes a LiDAR (or `virtual_lidar_node` in SITL) with Nav2 costmaps to dynamically dodge obstacles during flight.

## 🏗️ Hardware & Software Architecture Diagram
```mermaid
graph TD
    subgraph Hardware Layer
        Lidar[LiDAR Sensor / Depth Camera]
        CC[Companion Computer <br> e.g. Raspberry Pi / Jetson]
        FC[Flight Controller <br> e.g. Pixhawk Cubepilot]
    end

    subgraph "ROS 2 Software Stack (Companion Computer)"
        FakeLidar[virtual_lidar_node.py]
        Nav2[Nav2 Costmap & Planners]
        ObstacleNode[nav2_obstacle_node.py]
        MavrosNode[MAVROS Node]
    end

    %% Connections
    Lidar -->|LaserScan / PointCloud2| Nav2
    FakeLidar -.->|Simulated Scan for testing| Nav2
    Nav2 -->|Planned Path / Twist| ObstacleNode
    ObstacleNode -->|cmd_vel| MavrosNode
    
    Lidar <-->|USB / I2C| CC
    MavrosNode <-->|UART / USB Serial MAVLink| FC
```
