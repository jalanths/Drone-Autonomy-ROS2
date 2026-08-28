#!/usr/bin/env python3
"""
MAVLink stream-rate kicker
==========================
Asks the FCU to actually SEND telemetry. Without this the entire autonomy
stack starves, in a way that is very hard to diagnose from the symptoms.

THE PROBLEM
-----------
ArduPilot transmits nothing but HEARTBEAT (and TIMESYNC) until a GCS asks for
more, via REQUEST_DATA_STREAM or MAV_CMD_SET_MESSAGE_INTERVAL. MAVProxy does
this automatically, so the issue is invisible when SITL is started through
sim_vehicle.py — but MAVROS 2 does NOT request streams from an ArduPilot
target, so a direct MAVROS <-> SITL link sits like this forever:

    /mavros/state                    1 Hz   <- HEARTBEAT only
    /mavros/local_position/pose      SILENT
    /mavros/global_position/global   SILENT
    /mavros/imu/data                 SILENT

Verified directly with pymavlink against SITL: 8 s of listening yielded only
HEARTBEAT/TIMESYNC, and a single REQUEST_DATA_STREAM(ALL, 10 Hz) immediately
produced ATTITUDE, GLOBAL_POSITION_INT, VFR_HUD, SYS_STATUS and friends at
~12 Hz.

WHY IT MATTERS SO MUCH HERE
---------------------------
/mavros/local_position/pose is the source for the odom -> base_link transform.
No pose -> no TF -> Nav2 rejects every laser return -> the costmap is empty ->
the drone believes the world is clear and flies into a building. The visible
failure is "obstacle avoidance doesn't work"; the actual cause is an
unrequested MAVLink stream three layers down.

WHY MAV_CMD_SET_MESSAGE_INTERVAL AND NOT /mavros/set_stream_rate
----------------------------------------------------------------
Two other routes were tried on this machine and rejected:

  * SR0_* parameters in the SITL defaults file. Reading SR0_POSITION back off
    the FCU returned nothing and the streams stayed silent even after wiping
    the EEPROM to guarantee the defaults applied.
  * /mavros/set_stream_rate. The service returns success and does nothing —
    another casualty of this build's broken plugin namespacing.

Sending MAV_CMD_SET_MESSAGE_INTERVAL (command 511) through /mavros/cmd/command
works, and is the modern MAVLink 2 mechanism that supersedes the deprecated
REQUEST_DATA_STREAM. It behaves identically against real ArduPilot hardware.

NOTE: /mavros/cmd/command only exists because sim_launch.py remaps the
flattened /mavros/mavros/command onto it. See the remapping block there.
"""

import rclpy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandLong
from rclpy.node import Node

# MAVLink message IDs the autonomy stack depends on.
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MESSAGES = [
    (32,  'LOCAL_POSITION_NED'),    # -> /mavros/local_position/pose  (drives TF)
    (33,  'GLOBAL_POSITION_INT'),   # -> /mavros/global_position/global
    (30,  'ATTITUDE'),
    (31,  'ATTITUDE_QUATERNION'),
    (1,   'SYS_STATUS'),
    (24,  'GPS_RAW_INT'),
    (74,  'VFR_HUD'),
    (27,  'RAW_IMU'),
    (242, 'HOME_POSITION'),
    (245, 'EXTENDED_SYS_STATE'),
]


class StreamRateNode(Node):

    def __init__(self):
        super().__init__('stream_rate_node')

        self.declare_parameter('rate_hz', 20)
        self.declare_parameter('retry_period_s', 5.0)
        self.rate_hz = int(self.get_parameter('rate_hz').value)

        self.connected = False
        self.sent = 0

        self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.cli = self.create_client(CommandLong, '/mavros/cmd/command')

        # Re-send periodically rather than once: the FCU can reboot, the link
        # can drop, and a stream request made before the FCU was listening is
        # simply lost. Re-asserting is cheap and idempotent.
        self.create_timer(
            float(self.get_parameter('retry_period_s').value), self.kick)

        self.get_logger().info('═══════════════════════════════════════════════')
        self.get_logger().info('  📶 MAVLink stream-rate kicker')
        self.get_logger().info(f'  Requesting ALL streams @ {self.rate_hz} Hz')
        self.get_logger().info('═══════════════════════════════════════════════')

    def state_cb(self, msg: State):
        if msg.connected and not self.connected:
            self.connected = True
            self.get_logger().info('✅ FCU connected — requesting streams.')
            self.kick()
        elif not msg.connected and self.connected:
            self.connected = False
            self.get_logger().warn('⚠️  FCU link lost — will re-request on reconnect.')

    def kick(self):
        if not self.connected or not self.cli.service_is_ready():
            return

        interval_us = float(int(1_000_000 / max(1, self.rate_hz)))
        for msg_id, _name in MESSAGES:
            req = CommandLong.Request()
            req.broadcast = False
            req.command = MAV_CMD_SET_MESSAGE_INTERVAL
            req.confirmation = 0
            req.param1 = float(msg_id)      # message to configure
            req.param2 = interval_us        # interval in microseconds
            self.cli.call_async(req)

        self.sent += 1
        if self.sent <= 3:
            self.get_logger().info(
                f'📡 Requested {len(MESSAGES)} message streams @ '
                f'{self.rate_hz} Hz (attempt {self.sent})')


def main(args=None):
    rclpy.init(args=args)
    node = StreamRateNode()
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
