# ══════════════════════════════════════════════════════════════════════════
#  Manual bring-up cheatsheet
#  (scripts/run_mission.sh does all of this for you — prefer that)
# ══════════════════════════════════════════════════════════════════════════
#
#  ORDER MATTERS. ArduPilot SITL parks at "Waiting for connection ...." and
#  does not run its main loop until a MAVLink client attaches, so until
#  MAVROS connects Gazebo prints "Broken ArduPilot connection" on a loop.
#  Those warnings before step 3 are expected, not a fault.

WS=~/Downloads/ROBO/drone_ws
AP=~/ardupilot

# ── Terminal 1 — Gazebo + MAVROS + stream rates + moving obstacles ────────
cd $WS && . install/setup.bash
ros2 launch drone_autonomy sim_launch.py            # add gui:=false for headless

# ── Terminal 2 — ArduPilot SITL ──────────────────────────────────────────
# --wipe is required: --defaults only fills parameters absent from storage,
# and SITL persists eeprom.bin in the working directory.
mkdir -p /tmp/sitl && cd /tmp/sitl
$AP/build/sitl/bin/arducopter --model gazebo-iris --speedup 1 -I0 --wipe \
  --home -35.363262,149.165237,584,0 \
  --defaults "$AP/Tools/autotest/default_params/copter.parm,$AP/Tools/autotest/default_params/gazebo-iris.parm,$WS/src/drone_autonomy/config/sitl_defaults.parm"

# Alternative, if you want the MAVProxy console/map instead of the raw binary.
# MAVProxy also requests MAVLink streams by itself. Launch sim_launch.py with
# fcu_url:=udp://127.0.0.1:14550@ to match.
#   cd $AP/ArduCopter && sim_vehicle.py -f gazebo-iris --console --map

# ── Terminal 3 — wait for the FCU link, then the autonomy stack ──────────
cd $WS && . install/setup.bash
ros2 topic echo /mavros/state --once          # expect: connected: true
ros2 launch drone_autonomy autonomy_launch.py # TF + costmap + brain + RViz2

# ── Handy checks ─────────────────────────────────────────────────────────
ros2 topic hz /scan                            # LiDAR alive (~10 Hz sim time)
ros2 topic hz /mavros/local_position/pose      # silent => streams not requested
ros2 run tf2_ros tf2_echo odom base_link       # missing => costmap will be empty
ros2 topic echo /costmap/costmap --once | head # costmap actually publishing

# ── Offline test of the avoidance maths (no simulator needed) ────────────
python3 $WS/src/drone_autonomy/test/test_avoidance.py
