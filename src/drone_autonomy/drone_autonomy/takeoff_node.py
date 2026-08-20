#!/usr/bin/env python3
"""
Autonomous Takeoff Node for ArduCopter via MAVROS (ROS 2 Humble)
================================================================
This node demonstrates the fundamental autonomous flight sequence:
  1. Waits for a stable MAVROS connection (heartbeat from FCU)
  2. Switches the flight mode to GUIDED
  3. Arms the motors
  4. Commands a 15-meter vertical takeoff

Usage:
  Terminal 1: cd ~/ardupilot/ArduCopter && sim_vehicle.py --console --map
  Terminal 2: ros2 run mavros mavros_node --ros-args -p fcu_url:="udp://127.0.0.1:14550@"
  Terminal 3: python3 takeoff_node.py
"""

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL


class AutonomousTakeoffNode(Node):
    def __init__(self):
        super().__init__('autonomous_takeoff_node')

        # ── Current drone state ──────────────────────────────────
        self.current_state = State()

        # ── Subscriber: listens to /mavros/state for connection
        #    and flight mode updates from the FCU ─────────────────
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            10
        )

        # ── Service Clients ──────────────────────────────────────
        # SetMode: changes the flight mode (e.g. STABILIZE -> GUIDED)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # CommandBool: arms or disarms the motors (True = arm)
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')

        # CommandTOL: commands takeoff or landing with a target altitude
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # ── State machine flags ──────────────────────────────────
        self.is_connected = False
        self.mode_set = False
        self.is_armed = False
        self.takeoff_sent = False

        # ── Main timer: runs the flight sequence every 1 second ──
        self.timer = self.create_timer(1.0, self.flight_sequence)

        self.get_logger().info('═══════════════════════════════════════════')
        self.get_logger().info('  Autonomous Takeoff Node Started')
        self.get_logger().info('  Waiting for FCU connection...')
        self.get_logger().info('═══════════════════════════════════════════')

    def state_callback(self, msg: State):
        """Called every time MAVROS publishes the drone's state."""
        self.current_state = msg

    def flight_sequence(self):
        """
        Main state machine. Executes one step per timer tick:
          Step 1: Wait for FCU connection
          Step 2: Set mode to GUIDED
          Step 3: Arm the motors
          Step 4: Command takeoff to 15 meters
        """

        # ── STEP 1: Wait for connection ──────────────────────────
        if not self.current_state.connected:
            self.get_logger().info('Waiting for FCU connection...')
            return

        if not self.is_connected:
            self.is_connected = True
            self.get_logger().info('✅ FCU Connected! Heartbeat received.')

        # ── STEP 2: Set flight mode to GUIDED ────────────────────
        if not self.mode_set:
            if self.current_state.mode == 'GUIDED':
                self.mode_set = True
                self.get_logger().info('✅ Already in GUIDED mode.')
            else:
                self.set_mode('GUIDED')
            return

        # ── STEP 3: Arm the motors ───────────────────────────────
        if not self.is_armed:
            if self.current_state.armed:
                self.is_armed = True
                self.get_logger().info('✅ Motors already armed.')
            else:
                self.arm_drone()
            return

        # ── STEP 4: Takeoff to 15 meters ─────────────────────────
        if not self.takeoff_sent:
            self.takeoff(altitude=15.0)
            return

    # ══════════════════════════════════════════════════════════════
    #  MAVROS Service Call Helpers
    # ══════════════════════════════════════════════════════════════

    def set_mode(self, mode: str):
        """
        Calls /mavros/set_mode to change the flight mode.
        
        How it works:
        - MAVROS exposes a SetMode service that sends a MAVLink
          SET_MODE command to the Pixhawk.
        - For autonomous flight, we use 'GUIDED' mode, which tells
          ArduCopter to accept position/velocity commands from
          an external computer (our ROS 2 node) instead of the RC.
        """
        if not self.set_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('SetMode service not available!')
            return

        request = SetMode.Request()
        request.custom_mode = mode

        future = self.set_mode_client.call_async(request)
        future.add_done_callback(self.set_mode_response)
        self.get_logger().info(f'🔄 Requesting mode change to {mode}...')

    def set_mode_response(self, future):
        try:
            response = future.result()
            if response.mode_sent:
                self.mode_set = True
                self.get_logger().info('✅ Mode changed to GUIDED!')
            else:
                self.get_logger().warn('❌ Mode change rejected by FCU.')
        except Exception as e:
            self.get_logger().error(f'SetMode service call failed: {e}')

    def arm_drone(self):
        """
        Calls /mavros/cmd/arming to arm the motors.
        
        How it works:
        - MAVROS exposes a CommandBool service. Sending True arms
          the motors, sending False disarms them.
        - The Pixhawk has safety checks (GPS lock, EKF health, etc.)
          and may reject the arm request if pre-flight checks fail.
        - In SITL simulation, these checks pass automatically.
        """
        if not self.arming_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Arming service not available!')
            return

        request = CommandBool.Request()
        request.value = True  # True = arm, False = disarm

        future = self.arming_client.call_async(request)
        future.add_done_callback(self.arm_response)
        self.get_logger().info('🔄 Requesting motor arming...')

    def arm_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.is_armed = True
                self.get_logger().info('✅ Motors ARMED! Props spinning.')
            else:
                self.get_logger().warn('❌ Arming rejected. Retrying...')
        except Exception as e:
            self.get_logger().error(f'Arming service call failed: {e}')

    def takeoff(self, altitude: float):
        """
        Calls /mavros/cmd/takeoff to command a vertical ascent.
        
        How it works:
        - MAVROS exposes a CommandTOL (Takeoff Or Land) service.
        - We specify a target altitude in meters. The Pixhawk's
          internal controller handles throttle, stabilization, and
          climb rate automatically.
        - The drone will ascend vertically and hold position at
          the target altitude until it receives further commands.
        """
        if not self.takeoff_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Takeoff service not available!')
            return

        request = CommandTOL.Request()
        request.altitude = altitude
        request.latitude = 0.0   # 0 = use current GPS position
        request.longitude = 0.0  # 0 = use current GPS position
        request.min_pitch = 0.0
        request.yaw = 0.0

        future = self.takeoff_client.call_async(request)
        future.add_done_callback(self.takeoff_response)
        self.get_logger().info(f'🔄 Requesting takeoff to {altitude}m...')

    def takeoff_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.takeoff_sent = True
                self.get_logger().info('═══════════════════════════════════════════')
                self.get_logger().info('  🚁 TAKEOFF SUCCESSFUL!')
                self.get_logger().info('  Drone is ascending to 15 meters.')
                self.get_logger().info('  Watch your MAVProxy map!')
                self.get_logger().info('═══════════════════════════════════════════')
            else:
                self.get_logger().warn('❌ Takeoff rejected. Retrying...')
        except Exception as e:
            self.get_logger().error(f'Takeoff service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousTakeoffNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Autonomous Takeoff Node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
