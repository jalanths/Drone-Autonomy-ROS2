#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  run_mission.sh — one-shot bring-up of the full autonomous stack
# ══════════════════════════════════════════════════════════════════════════
#
#      Gazebo Classic 11  <-- FDM (UDP 9002/9003) -->  ArduPilot SITL
#            |                                              |
#         /scan, /clock                              MAVLink (TCP 5760)
#            |                                              |
#            +--------------->  ROS 2  <--------------------+
#                     Nav2 costmap -> mission_avoidance_node -> RViz2
#
#  ORDERING IS NOT OPTIONAL
#  ------------------------
#   1. Gazebo first — the ArduPilot plugin is the SERVER on UDP 9002, so it
#      must be listening before SITL tries to talk to it.
#   2. SITL second.
#   3. MAVROS third — and this one is load-bearing: ArduPilot SITL parks at
#      "Waiting for connection ...." and does NOT run its main loop until a
#      MAVLink client attaches. Until MAVROS connects, SITL sends no servo
#      packets, and Gazebo prints "Broken ArduPilot connection" on a loop.
#      Those warnings before step 3 are EXPECTED, not a fault.
#   4. Autonomy stack last, once /mavros/state reports connected.
#
#  Usage:
#      ./scripts/run_mission.sh              # headless Gazebo + RViz2
#      ./scripts/run_mission.sh --gui        # Gazebo GUI as well
#      ./scripts/run_mission.sh --no-rviz    # no RViz2
#      ./scripts/run_mission.sh --stop       # kill everything and exit
# ══════════════════════════════════════════════════════════════════════════
# NOTE: deliberately NOT using `set -u`. ROS 2 and Gazebo setup scripts
# dereference unset variables internally (AMENT_TRACE_SETUP_FILES and friends),
# so `set -u` aborts the moment we source them.
set -o pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AP="${ARDUPILOT_HOME:-$HOME/ardupilot}"
APG="${ARDUPILOT_GAZEBO_HOME:-$HOME/ardupilot_gazebo}"
LOGDIR="${WS}/.run_logs"
RUNDIR="${LOGDIR}/sitl"

GUI=false
RVIZ=true
HOME_LOC="-35.363262,149.165237,584,0"

for a in "$@"; do
  case "$a" in
    --gui)      GUI=true ;;
    --no-rviz)  RVIZ=false ;;
    --stop)     echo "Stopping…"
                pkill -9 -f mission_avoidance_node 2>/dev/null
                pkill -9 -f dynamic_obstacle_node 2>/dev/null
                pkill -9 -f "ros2 launch drone_autonomy" 2>/dev/null
                pkill -9 rviz2 2>/dev/null; pkill -9 mavros_node 2>/dev/null
                pkill -9 arducopter 2>/dev/null
                pkill -9 gzserver 2>/dev/null; pkill -9 gzclient 2>/dev/null
                sleep 2; echo "Stopped."; exit 0 ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a"; exit 1 ;;
  esac
done

# ── Preflight ─────────────────────────────────────────────────────────────
[ -x "$AP/build/sitl/bin/arducopter" ] || {
  echo "❌ SITL binary missing: $AP/build/sitl/bin/arducopter"
  echo "   Build it with:  cd $AP && ./waf configure --board sitl && ./waf copter"
  exit 1; }
[ -f "$APG/build/libArduPilotPlugin.so" ] || {
  echo "❌ ArduPilot Gazebo plugin missing: $APG/build/libArduPilotPlugin.so"
  exit 1; }

# The Iris model lives OUTSIDE this repo, so its LiDAR rate can silently drift
# out of step with control_rate_hz. That coupling is not cosmetic: the avoidance
# loop is pinned to the scan rate so it acts on every fresh scan exactly once,
# and threat_persist_ticks is counted in scans. A 10 Hz sensor under a 20 Hz
# loop means half the ticks re-decide on stale data and every dynamic obstacle
# is confirmed twice as slowly as intended.
MODEL_SDF="$APG/models/iris_with_ardupilot/model.sdf"
if [ -f "$MODEL_SDF" ]; then
  LIDAR_HZ=$(grep -A4 '<sensor name="rplidar"' "$MODEL_SDF" \
             | grep -o '<update_rate>[0-9.]*' | grep -o '[0-9.]*$' | head -1)
  LOOP_HZ=$(grep -o 'control_rate_hz: *[0-9.]*' \
            "$WS/src/drone_autonomy/config/mission_params.yaml" \
            | grep -o '[0-9.]*$' | head -1)
  if [ -n "$LIDAR_HZ" ] && [ -n "$LOOP_HZ" ] \
     && [ "${LIDAR_HZ%.*}" != "${LOOP_HZ%.*}" ]; then
    echo "⚠️  LiDAR is ${LIDAR_HZ} Hz but control_rate_hz is ${LOOP_HZ} Hz."
    echo "   Set <update_rate> to ${LOOP_HZ%.*} in $MODEL_SDF"
  fi
fi

mkdir -p "$RUNDIR"

# ── Environment ───────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH="${GAZEBO_PLUGIN_PATH:-}:/opt/ros/humble/lib:$APG/build"
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:-}:$APG/models"
[ -f "$WS/install/setup.bash" ] || { echo "❌ Workspace not built. Run: colcon build"; exit 1; }
source "$WS/install/setup.bash"

echo "════════════════════════════════════════════════════════"
echo "  🚁 Autonomous 6-Waypoint Mission"
echo "  workspace : $WS"
echo "  logs      : $LOGDIR"
echo "════════════════════════════════════════════════════════"

cleanup() {
  echo ""; echo "Shutting down…"
  pkill -9 -f mission_avoidance_node 2>/dev/null
  pkill -9 -f "ros2 launch drone_autonomy" 2>/dev/null
  pkill -9 rviz2 2>/dev/null; pkill -9 mavros_node 2>/dev/null
  pkill -9 arducopter 2>/dev/null
  pkill -9 gzserver 2>/dev/null; pkill -9 gzclient 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# ── 1. Gazebo + MAVROS ────────────────────────────────────────────────────
echo "[1/4] Gazebo (gui=$GUI) + MAVROS…"
setsid ros2 launch drone_autonomy sim_launch.py \
    gui:="$GUI" > "$LOGDIR/sim.log" 2>&1 < /dev/null &
sleep 18
pgrep -x gzserver >/dev/null || { echo "❌ gzserver failed — see $LOGDIR/sim.log"; exit 1; }
echo "      ✅ Gazebo up"

# ── 2. ArduPilot SITL ─────────────────────────────────────────────────────
# Run the binary directly rather than via sim_vehicle.py: MAVProxy needs an
# interactive TTY and exits immediately under nohup/setsid, which would leave
# SITL with no MAVLink client (see header).
echo "[2/4] ArduPilot SITL…"
# --wipe is REQUIRED, not hygiene. `--defaults` only supplies values for
# parameters that are absent from storage, and SITL persists everything to
# eeprom.bin in the working directory. Without a wipe, the SR0_* stream rates
# in sitl_defaults.parm are shadowed by whatever the first-ever boot wrote,
# telemetry stays at HEARTBEAT-only, and /mavros/local_position/pose never
# publishes — which silently starves the odom->base_link TF and the costmap.
( cd "$RUNDIR" && setsid "$AP/build/sitl/bin/arducopter" \
    --model gazebo-iris --speedup 1 -I0 --wipe --home "$HOME_LOC" \
    --defaults "$AP/Tools/autotest/default_params/copter.parm,$AP/Tools/autotest/default_params/gazebo-iris.parm,$WS/src/drone_autonomy/config/sitl_defaults.parm" \
    > "$LOGDIR/sitl.log" 2>&1 < /dev/null & )
sleep 12
pgrep -x arducopter >/dev/null || { echo "❌ SITL failed — see $LOGDIR/sitl.log"; exit 1; }
echo "      ✅ SITL up (MAVLink on tcp://127.0.0.1:5760)"

# ── 3. Wait for the FCU link ──────────────────────────────────────────────
echo "[3/4] Waiting for MAVROS <-> FCU heartbeat…"
CONNECTED=false
for i in $(seq 1 40); do
  if timeout 4 ros2 topic echo /mavros/state --once 2>/dev/null | grep -q "connected: true"; then
    CONNECTED=true; break
  fi
  sleep 3
done
$CONNECTED || { echo "❌ No heartbeat. Check $LOGDIR/sim.log and $LOGDIR/sitl.log"; exit 1; }
echo "      ✅ FCU connected"

# ── 4. Autonomy stack ─────────────────────────────────────────────────────
echo "[4/4] Costmap + avoidance brain + RViz2…"
setsid ros2 launch drone_autonomy autonomy_launch.py \
    rviz:="$RVIZ" > "$LOGDIR/autonomy.log" 2>&1 < /dev/null &
sleep 8
echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ Stack running. Mission log follows (Ctrl-C to stop)."
echo "════════════════════════════════════════════════════════"
tail -f "$LOGDIR/autonomy.log"
