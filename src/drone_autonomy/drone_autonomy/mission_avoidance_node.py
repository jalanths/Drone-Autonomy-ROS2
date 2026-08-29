#!/usr/bin/env python3
"""
Autonomous 6-Waypoint Mission with Dynamic Obstacle Avoidance + Smart Retrace
=============================================================================
V4.0 — Gazebo Classic 11 + ArduPilot SITL + MAVROS + Nav2 costmap + RViz2

WHAT THIS NODE DOES
-------------------
Flies a 6-waypoint tour through the Gazebo city world, dodging whatever the
LiDAR finds on the way, then RETRACES the exact corridor it proved safe on
the way out to get home again.

    ARM -> TAKEOFF -> [WP1..WP6 with live avoidance] -> RETRACE -> LAND

WHY THE OLD ALGORITHM WAS REPLACED
----------------------------------
The previous `nav2_obstacle_node` carved the costmap into four axis-aligned
rectangles and called them Front/Left/Right/Back "relative to the drone's
nose". They were not. A numpy slice such as

    front = costmap[cy-gap : cy+gap,  cx : cx+radius]

always reads toward +x of the *costmap* frame, and the costmap lives in
`odom` (ENU). So "front" meant **due East**, permanently — no matter which
way the drone was pointing or travelling. The avoidance only behaved
sensibly on an eastbound leg; on a westbound leg it would happily steer
*into* obstacles because it was inspecting the wrong half of the map.

This node instead runs a proper VFH+ (Vector Field Histogram) style
polar analysis that is genuinely relative to the DIRECTION OF TRAVEL:

  1. POLAR HISTOGRAM  Obstacles inside the lookahead disc are binned by their
     true bearing from the drone (72 bins, 5 deg each), fusing TWO sources:
     the raw LaserScan as the low-latency primary, and the Nav2 costmap as
     persistent, inflated memory. The per-sector minimum is taken, so
     whichever source spots danger first wins. See build_histogram() for the
     flight data showing why the costmap alone was not enough.
  2. ANGULAR ENLARGEMENT  Each occupied bin is widened by
     gamma = asin(safety_radius / distance). A pillar 2 m away blocks a much
     wider arc than the same pillar 15 m away. This is what keeps the rotors
     off the concrete instead of relying on a hand-tuned rectangle.
  3. CORRIDOR SELECTION  A heading is legal only if a CORRIDOR around it is
     clear, not merely the single bin pointing down it. Legal headings are
     scored on deviation from the goal, plus hysteresis so the drone commits
     to one side of an obstacle instead of dithering across its centreline.
  4. TRAP ESCAPE  If no corridor survives, the drone is boxed in on all
     360 deg and climbs vertically to fly over the trap; the target altitude
     then bleeds steadily back down to cruise on every clear tick.

Everything is computed in ENU, which is the SAME frame the costmap uses and
the SAME frame MAVROS interprets /mavros/setpoint_velocity/cmd_vel_unstamped
in — so there is no hidden frame conversion left to get wrong.

SMART RETRACE
-------------
While outbound, the drone drops a breadcrumb every `breadcrumb_spacing`
metres, but ONLY at moments when its path was clear. The return leg walks
those breadcrumbs in reverse. That corridor is known-flyable because the
drone physically flew it, so the trip home avoids re-solving the maze.
Avoidance stays fully armed during the retrace, so obstacles that MOVED in
behind the drone are still dodged.

FRAME NOTE (Gazebo world  ->  MAVROS ENU)
-----------------------------------------
The ArduPilot plugin declares <gazeboXYZToNED>0 0 0 3.141593 0 0</...>,
a 180 deg roll, so NED_north = gz_x and NED_east = -gz_y. Converting NED to
the ENU frame MAVROS reports gives:

        enu_x (East)  = -gz_y
        enu_y (North) = +gz_x
        enu_z (Up)    = +gz_z

Waypoints are therefore authored in intuitive Gazebo world coordinates
(what you actually see in the simulator) and converted internally. Set the
`waypoint_frame` parameter to 'enu' to bypass the conversion.
"""

import heapq
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# ── Costmap value semantics ───────────────────────────────────────────────
# nav2's Costmap2DPublisher rescales the internal 0..255 costmap onto the
# OccupancyGrid 0..100 range before publishing: 254 (lethal) -> 100,
# 253 (inscribed) -> 99, 255 (unknown) -> -1. So all thresholds below live
# on the 0..100 scale, and -1 means "never observed", which for an aerial
# vehicle we treat as free rather than blocked.
UNKNOWN = -1


def wrap_pi(angle: float) -> float:
    """Wrap an angle into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    """Extract the ENU yaw (rotation about Up) from a geometry_msgs Quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def rp_from_quat(q):
    """Extract (roll, pitch) from a geometry_msgs Quaternion."""
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    return roll, pitch


def cpa(rel_pos, rel_vel):
    """
    Closest point of approach between two bodies moving at constant velocity.

    `rel_pos` is (them - us) and `rel_vel` is (their velocity - ours), both
    2-vectors in a common frame. Returns (t_cpa, d_cpa): the time at which the
    gap is smallest, and how big that gap will be.

    WHY THIS AND NOT A CLOSING RATE
    -------------------------------
    Everything else in this node measures range and how fast range is
    shrinking. That answers "is it getting nearer", which is not the question.
    A block crossing four metres in front of the drone is getting nearer the
    whole time and will never touch it; a block on a true collision course has
    a constant bearing and, if the speeds happen to match, an unremarkable
    closing rate. Range rate cannot separate those two. Relative geometry can:

        gap(t) = |rel_pos + rel_vel * t|

    is a parabola in t, and its minimum is the whole answer. Differentiating
    and solving for zero gives t_cpa = -(rel_pos . rel_vel) / |rel_vel|^2.

    Two degenerate cases matter, and both mean "no intercept":

      * t_cpa <= 0 — the minimum is in the PAST. The gap is opening and will
        keep opening. Returned as a negative time so the caller's `0 < t`
        test rejects it without a special case.
      * |rel_vel| ~ 0 — station keeping. There is no approach to have a
        closest point of, and the formula would divide by zero. Returned as
        infinite time, which fails any finite horizon test.

    In both cases d_cpa is reported as the CURRENT separation, which is the
    only honest answer when there is no future minimum to speak of.
    """
    rel_pos = np.asarray(rel_pos, dtype=float)
    rel_vel = np.asarray(rel_vel, dtype=float)
    vv = float(np.dot(rel_vel, rel_vel))
    now_gap = float(np.linalg.norm(rel_pos))
    if vv < 1e-6:
        return float('inf'), now_gap
    t = -float(np.dot(rel_pos, rel_vel)) / vv
    if t <= 0.0:
        return t, now_gap
    return t, float(np.linalg.norm(rel_pos + rel_vel * t))


class _Track:
    """
    One tracked object: a filtered position and velocity in the world frame.

    Deliberately not a dataclass and deliberately tiny — one of these exists
    per visible cluster and they are rebuilt constantly inside a 20 Hz loop.
    """

    __slots__ = ('pos', 'vel', 't', 'hits')

    def __init__(self, pos, t):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros(2)
        self.t = float(t)
        self.hits = 1


class MissionAvoidanceNode(Node):

    # ══════════════════════════════════════════════════════════════════════
    #  Construction
    # ══════════════════════════════════════════════════════════════════════

    def __init__(self):
        super().__init__('mission_avoidance_node')

        # ── Parameters ────────────────────────────────────────────────────
        # Waypoints are a flat [x1, y1, x2, y2, ...] list because ROS 2
        # parameters cannot express a list-of-lists.
        self.declare_parameter('waypoints', [
             40.0,   0.0,   # WP1  east along the main road (warm-up leg)
             60.0,  40.0,   # WP2  skyscraper corridor  (leg crosses a tower)
             40.0,  60.0,   # WP3  deep canyon crossing (leg crosses a tower)
              0.0,  55.0,   # WP4  open ground south of the bridge
            -50.0,  40.0,   # WP5  park / forest, trees sit below cruise alt
             -8.0, -32.0,   # WP6  suburban street grid (houses + roofs)
        ])
        self.declare_parameter('waypoint_frame', 'gazebo')   # 'gazebo' | 'enu'
        self.declare_parameter('cruise_altitude', 4.0)
        self.declare_parameter('wp_radius', 3.0)
        self.declare_parameter('final_wp_radius', 1.0)
        self.declare_parameter('final_approach_speed', 0.25)
        self.declare_parameter('final_approach_timeout_s', 30.0)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('max_accel', 2.0)      # m/s^2 setpoint slew
        self.declare_parameter('max_accel_z', 1.5)
        self.declare_parameter('contact_speed', 0.15)
        self.declare_parameter('contact_confirm_s', 2.0)
        self.declare_parameter('contact_escape_speed', 1.0)
        self.declare_parameter('use_global_planner', True)
        self.declare_parameter('plan_lethal_cost', 90)
        self.declare_parameter('plan_downsample', 2)
        self.declare_parameter('plan_carrot_dist', 6.0)
        self.declare_parameter('replan_period_s', 1.0)
        self.declare_parameter('plan_cost_penalty', 0.8)
        self.declare_parameter('plan_unknown_penalty', 0.6)
        self.declare_parameter('plan_max_expansions', 60000)
        self.declare_parameter('route_stall_limit', 3)
        self.declare_parameter('backoff_range', 1.5)
        self.declare_parameter('backoff_arc', 80.0)
        self.declare_parameter('backoff_speed', 0.6)
        self.declare_parameter('backoff_commit_s', 1.5)
        self.declare_parameter('backoff_release_margin', 0.5)
        self.declare_parameter('detour_max_drift', 8.0)
        self.declare_parameter('no_detour_radius', 25.0)
        self.declare_parameter('detour_leave_margin', 3.0)
        self.declare_parameter('detour_timeout_s', 70.0)
        self.declare_parameter('cruise_speed', 0.8)
        self.declare_parameter('min_speed', 0.4)
        self.declare_parameter('takeoff_settle_s', 12.0)

        # Avoidance geometry
        self.declare_parameter('lookahead', 10.0)        # m, steering horizon
        self.declare_parameter('safety_radius', 1.0)     # m, angular enlargement
        self.declare_parameter('blocked_cost', 60)       # 0..100 occupancy scale
        self.declare_parameter('costmap_alt_band', 1.5)  # m above cruise before
                                                        # the 2D costmap is ignored
        self.declare_parameter('sector_count', 72)       # 5 deg per bin
        # Ignore returns closer than this: the drone sees ITSELF. Measured in
        # flight, level at 5.8 m, 13 returns sat at 0.30-0.33 m in a fixed
        # body-frame arc (-129 to -141 deg) — a ~6 cm object that never moves
        # relative to the airframe. That is a propeller blade cutting the LiDAR
        # plane. It is invisible on the ground because the motors are not
        # spinning, which is why it only appeared once armed.
        #
        # The damage was severe and diffuse: blades sweep, so the return
        # flickers, which the time-to-contact guard reads as enormous closing
        # speed and fires "impact in 0.0 s" continuously; and the sector is
        # permanently blocked in the histogram, dragging the drone toward
        # 'trapped' and inflating dodge counts (114 dodges to reach WP1 versus
        # ~15 once filtered).
        #
        # Real 2D LiDARs have exactly this problem — you cannot see through
        # your own props — so the filter belongs in code, not only in the SDF.
        self.declare_parameter('self_filter_range', 0.7)  # m
        self.declare_parameter('max_beam_z_offset', 2.0)  # m — reject beams
                                                          # that hit terrain
        self.declare_parameter('goal_weight', 1.0)
        self.declare_parameter('hysteresis_weight', 0.8)
        self.declare_parameter('max_heading_rate', 45.0)  # deg/s course slew cap
        self.declare_parameter('yaw_gain', 0.4)
        self.declare_parameter('max_yaw_rate', 0.4)       # rad/s
        self.declare_parameter('min_valley_deg', 18.0)

        # Vertical trap escape
        self.declare_parameter('climb_speed', 1.5)
        self.declare_parameter('max_escape_altitude', 4.0)  # == cruise_altitude
        self.declare_parameter('escape_step', 3.0)
        self.declare_parameter('trap_confirm_s', 0.6)  # trap must persist this long
        self.declare_parameter('descend_rate', 0.6)   # m/s of target bleed-off
        self.declare_parameter('descend_clear_radius', 3.0)  # m; XY column must be
                                                             # free before sinking
        self.declare_parameter('stuck_timeout_s', 15.0)

        # Emergency brake straight off the raw LaserScan
        self.declare_parameter('evade_distance', 6.0)        # m, guard only looks
                                                            # inside this range
        self.declare_parameter('collision_horizon_s', 2.5)   # s time-to-contact
        self.declare_parameter('dynamic_closing_thresh', 0.8)  # m/s of closing
                                                              # our own motion
                                                              # cannot explain
        self.declare_parameter('hard_stop_distance', 0.7)    # m, fire regardless
        self.declare_parameter('threat_persist_ticks', 2)    # consecutive ticks
        self.declare_parameter('brake_hold_s', 1.5)          # s to stay stopped
                                                            # after last detection
        self.declare_parameter('brake_stop_speed', 0.25)      # m/s = 'stopped'
        self.declare_parameter('terrain_close_range', 2.0)   # m
        self.declare_parameter('terrain_close_sectors', 12)  # simultaneous sectors
        self.declare_parameter('terrain_below_margin', 0.2)  # m the hit must sit
                                                            # below to count as ground

        # Predicting where a dynamic obstacle will BE
        #
        # OFF by default. The reflexes above are the product of several
        # crashes and they work; this layer is additive and unflown, so it
        # does not get to change how the aircraft behaves until it is asked
        # for by name.
        self.declare_parameter('enable_prediction', False)
        self.declare_parameter('track_cluster_gap', 0.8)   # m of range step
                                                           # that splits two
                                                           # objects apart
        self.declare_parameter('track_min_points', 3)      # beams to be real
        self.declare_parameter('track_assoc_radius', 1.2)  # m a centroid may
                                                           # jump between frames
        self.declare_parameter('track_confirm_frames', 4)  # associations before
                                                           # the velocity is
                                                           # believed
        self.declare_parameter('track_max_age_s', 0.6)     # s unseen -> forget
        self.declare_parameter('track_min_speed', 0.35)    # m/s below which it
                                                           # is scenery
        self.declare_parameter('track_max_speed', 5.0)     # m/s above which it
                                                           # is an association
                                                           # blunder
        self.declare_parameter('track_alpha', 0.5)         # position gain
        self.declare_parameter('track_beta', 0.25)         # velocity gain
        self.declare_parameter('track_max_count', 12)
        self.declare_parameter('cpa_horizon_s', 5.0)       # s; ignore intercepts
                                                           # further off than this
        self.declare_parameter('cpa_miss_distance', 2.0)   # m; closer than this
                                                           # counts as a hit

        # Smart retrace
        self.declare_parameter('enable_retrace', True)
        self.declare_parameter('breadcrumb_spacing', 4.0)
        self.declare_parameter('retrace_lookahead', 8.0)   # m; chase a distant
                                                          # breadcrumb, not the next
        self.declare_parameter('evade_cooldown_s', 0.4)

        p = self.get_parameter
        self.cruise_alt = p('cruise_altitude').value
        self.wp_radius = p('wp_radius').value
        self.final_wp_radius = p('final_wp_radius').value
        self.final_approach_speed = p('final_approach_speed').value
        self.final_approach_timeout_s = p('final_approach_timeout_s').value
        self.rate_hz = float(p('control_rate_hz').value)
        self.dt = 1.0 / self.rate_hz
        self.max_accel = p('max_accel').value
        self.max_accel_z = p('max_accel_z').value
        self.contact_speed = p('contact_speed').value
        self.contact_confirm_s = p('contact_confirm_s').value
        self.contact_escape_speed = p('contact_escape_speed').value
        self.use_global_planner = p('use_global_planner').value
        self.plan_lethal_cost = p('plan_lethal_cost').value
        self.plan_downsample = max(1, int(p('plan_downsample').value))
        self.plan_carrot_dist = p('plan_carrot_dist').value
        self.replan_period_s = p('replan_period_s').value
        self.plan_cost_penalty = p('plan_cost_penalty').value
        self.plan_unknown_penalty = p('plan_unknown_penalty').value
        self.plan_max_expansions = int(p('plan_max_expansions').value)
        self.route_stall_limit = int(p('route_stall_limit').value)
        self.backoff_range = p('backoff_range').value
        self.backoff_arc = p('backoff_arc').value
        self.backoff_speed = p('backoff_speed').value
        self.backoff_commit_s = p('backoff_commit_s').value
        self.backoff_release_margin = p('backoff_release_margin').value
        # Latched escape bearings, keyed by the reflex that owns them.
        # See committed_bearing() for why these are latched at all.
        self._latched = {}
        self.backoff_dir = None
        self.detour_max_drift = p('detour_max_drift').value
        self.no_detour_radius = p('no_detour_radius').value
        self.detour_leave_margin = p('detour_leave_margin').value
        self.detour_timeout_s = p('detour_timeout_s').value
        self.cruise_speed = p('cruise_speed').value
        self.min_speed = p('min_speed').value
        self.takeoff_settle_s = p('takeoff_settle_s').value
        self.lookahead = p('lookahead').value
        self.safety_radius = p('safety_radius').value
        self.blocked_cost = p('blocked_cost').value
        self.costmap_alt_band = p('costmap_alt_band').value
        self.nbins = int(p('sector_count').value)
        self.self_filter_range = p('self_filter_range').value
        self.max_beam_z_offset = p('max_beam_z_offset').value
        self.goal_weight = p('goal_weight').value
        self.hysteresis_weight = p('hysteresis_weight').value
        self.max_heading_rate = p('max_heading_rate').value
        self.yaw_gain = p('yaw_gain').value
        self.max_yaw_rate = p('max_yaw_rate').value
        self.min_valley_bins = max(
            1, int(math.radians(p('min_valley_deg').value) / (2 * math.pi / self.nbins)))
        self.climb_speed = p('climb_speed').value
        self.max_escape_alt = p('max_escape_altitude').value
        self.escape_step = p('escape_step').value
        self.trap_confirm_s = p('trap_confirm_s').value
        self.descend_rate = p('descend_rate').value
        self.descend_clear_radius = p('descend_clear_radius').value
        self.stuck_timeout_s = p('stuck_timeout_s').value
        self.evade_distance = p('evade_distance').value
        self.collision_horizon_s = p('collision_horizon_s').value
        self.dynamic_closing_thresh = p('dynamic_closing_thresh').value
        self.hard_stop_distance = p('hard_stop_distance').value
        self.threat_persist_ticks = int(p('threat_persist_ticks').value)
        self.brake_hold_s = p('brake_hold_s').value
        self.brake_stop_speed = p('brake_stop_speed').value
        self.terrain_close_range = p('terrain_close_range').value
        self.terrain_close_sectors = int(p('terrain_close_sectors').value)
        self.enable_prediction = bool(p('enable_prediction').value)
        self.track_cluster_gap = float(p('track_cluster_gap').value)
        self.track_min_points = int(p('track_min_points').value)
        self.track_assoc_radius = float(p('track_assoc_radius').value)
        self.track_confirm_frames = int(p('track_confirm_frames').value)
        self.track_max_age_s = float(p('track_max_age_s').value)
        self.track_min_speed = float(p('track_min_speed').value)
        self.track_max_speed = float(p('track_max_speed').value)
        self.track_alpha = float(p('track_alpha').value)
        self.track_beta = float(p('track_beta').value)
        self.track_max_count = int(p('track_max_count').value)
        self.cpa_horizon_s = float(p('cpa_horizon_s').value)
        self.cpa_miss_distance = float(p('cpa_miss_distance').value)
        self.terrain_below_margin = p('terrain_below_margin').value
        self.enable_retrace = p('enable_retrace').value
        self.breadcrumb_spacing = p('breadcrumb_spacing').value
        self.retrace_lookahead = p('retrace_lookahead').value
        self.evade_cooldown_s = p('evade_cooldown_s').value

        self.waypoints = self._load_waypoints()

        # ── Live state ────────────────────────────────────────────────────
        self.state = State()
        self.pos = None              # np.array([x, y, z]) in ENU
        self.yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.pose_ok = False

        self.costmap = None          # np.int8 HxW
        self.map_res = 0.1
        self.map_origin = (0.0, 0.0)
        self.costmap_ok = False

        self.scan = None
        self.scan_ok = False

        # Mission bookkeeping
        self.phase = 'WAIT_FCU'
        self.wp_index = 0
        self.takeoff_time = None
        self.mission_start = None
        self.breadcrumbs = []        # outbound trail of proven-clear positions
        self.retrace_index = 0
        self.total_distance = 0.0
        self.last_pos = None

        # Avoidance memory
        self.target_alt = self.cruise_alt
        self.last_heading = None     # for hysteresis
        self.dodge_count = 0
        self.was_avoiding = False
        self.back_at_cruise = True
        self.hold_alt_logged = False
        self.terrain_logged = False
        self.sector_zoff = np.full(self.nbins, np.inf)
        self.tracks = []              # [_Track] — see cluster_scan/update_tracks
        self.prev_nearest = None      # previous polar ranges, for closing speed
        self.prev_nearest_t = 0.0
        self.last_evade_t = -99.0
        self.braking = False
        self.brake_until = 0.0
        self.vel_enu = np.zeros(2)    # own ENU velocity, for closing-rate maths
        self.vel_fused_t = -99.0      # last /mavros/.../velocity_local message
        self.vel_ok = False
        self.threat_ticks = np.zeros(int(p('sector_count').value), dtype=int)
        self.prev_vel_pos = None
        self.prev_vel_t = 0.0
        self.trap_ticks = 0
        self.best_dist = None       # closest approach to the current target
        self.progress_time = 0.0
        self.cmd = np.zeros(3)      # last ENU velocity SETPOINT, for slewing
        self.moving_since = None    # first tick of a commanded-but-still stall
        self.contact = False
        self.contact_heading = 0.0
        self.final_approach_since = None
        self.last_cmd_t = 0.0
        # Committed boundary following (tangent-bug). `detour_dir` is the
        # LATCHED turn direction: +1 = keep rounding the obstacle to the left
        # (counter-clockwise), -1 = to the right. Zero means not detouring.
        self.detour_dir = 0
        self.detour_entry_dist = None
        self.detour_since = 0.0
        # Global route (A* over the costmap) and the carrot chased along it.
        #
        # A* runs on its OWN thread, not in the control loop. Measured on this
        # 180x180 map it costs 50 ms for a long solvable route and 141 ms when
        # the goal is unreachable, against a control period of 50 ms — so
        # planning inline dropped one to three setpoint cycles every second,
        # and ArduPilot's GUIDED mode is fed by exactly that stream. The
        # control loop now only ever READS `path`; `planner_tick` writes it.
        self.path = []
        self.path_goal = None       # the goal `path` was actually planned for
        self.plan_request = None    # the goal the control loop wants a route to
        self.plan_target = None     # the goal the planner last worked on
        self.last_plan_t = 0.0
        self.plan_fail = 0
        self.plan_ms = 0.0
        # `path` and `path_goal` must agree, so they are swapped as a pair
        # under this lock. The lock is NEVER held across the search itself —
        # only across the assignment, which is microseconds.
        self._plan_lock = threading.Lock()
        # Bumped every time the control loop throws the route away. A search
        # that started before the bump is answering a question nobody is
        # asking any more, so its result is discarded instead of published.
        self.plan_epoch = 0
        self.route_stall = 0
        self.last_log = 0.0
        self._log_times = {}
        self.status = 'init'

        # ── ROS interfaces ────────────────────────────────────────────────
        # See the timer block below for why there are two callback groups.
        self.ctrl_cbg = MutuallyExclusiveCallbackGroup()
        self.plan_cbg = MutuallyExclusiveCallbackGroup()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1)

        self.create_subscription(State, '/mavros/state', self.state_cb, 10,
                                 callback_group=self.ctrl_cbg)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self.pose_cb, sensor_qos,
                                 callback_group=self.ctrl_cbg)
        self.create_subscription(OccupancyGrid, '/costmap/costmap',
                                 self.costmap_cb, map_qos,
                                 callback_group=self.ctrl_cbg)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, sensor_qos,
                                 callback_group=self.ctrl_cbg)
        # Own velocity, straight from the EKF. This used to be differenced
        # from consecutive pose messages, and that estimate feeds the three
        # sharpest thresholds in the node — dynamic_closing_thresh (0.8 m/s),
        # the back-off travel direction, and contact_speed (0.15 m/s). A two
        # sample difference of a pose that itself carries centimetre jitter is
        # a poor way to measure a fraction of a metre per second; the fused
        # estimate is far quieter and costs one subscription.
        #
        # TWO topic names, because MAVROS is not consistent about where it
        # puts this and the answer varies by version and namespace layout.
        # On this stack the pose lands on /mavros/local_position/pose but the
        # velocity lands on /mavros/mavros/velocity_local, and subscribing to
        # the matching /mavros/local_position/velocity_local found a topic
        # with ZERO publishers — the node fell back to differenced pose for a
        # whole flight without ever saying so. Subscribing to both costs
        # nothing: only one of them ever publishes, and vel_cb is idempotent.
        for topic in ('/mavros/local_position/velocity_local',
                      '/mavros/mavros/velocity_local'):
            self.create_subscription(TwistStamped, topic, self.vel_cb,
                                     sensor_qos, callback_group=self.ctrl_cbg)

        self.vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/mission/markers', 10)
        self.path_pub = self.create_publisher(Path, '/mission/travelled', 10)

        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_cli = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.land_cli = self.create_client(CommandTOL, '/mavros/cmd/land')

        # ArduPilot GUIDED expects a steady setpoint stream, and the rate is
        # matched to the LiDAR's update_rate so we act on every fresh scan and
        # never twice on the same one. Both were raised 10 -> 20 Hz: the
        # persistence filter counts SCANS, so this halves the time to confirm a
        # dynamic threat (200 ms -> 100 ms) without making it any noisier.
        #
        # The control timer, the markers and every subscription share ONE
        # mutually-exclusive group, which is exactly how they behaved under
        # the default single-threaded executor: no two of them ever interleave.
        # Only the planner sits in a group of its own, so its search runs in
        # parallel with the control loop instead of blocking it.
        self.create_timer(self.dt, self.main_loop, callback_group=self.ctrl_cbg)
        self.create_timer(0.5, self.publish_markers,
                          callback_group=self.ctrl_cbg)
        # Polled at 10 Hz but only PLANS when the route is stale or the goal
        # moved, so a forced invalidation is picked up within 100 ms instead
        # of waiting out a full replan_period_s.
        self.create_timer(0.1, self.planner_tick,
                          callback_group=self.plan_cbg)

        self.get_logger().info('═══════════════════════════════════════════════')
        self.get_logger().info('  🚁 6-Waypoint Autonomous Mission  (V4.0)')
        self.get_logger().info(f'  📍 Waypoints    : {len(self.waypoints)}')
        self.get_logger().info(f'  📏 Cruise alt   : {self.cruise_alt:.1f} m')
        self.get_logger().info(f'  🧠 Avoidance    : VFH+ polar histogram '
                               f'({self.nbins} bins, {self.lookahead:.0f} m horizon)')
        self.get_logger().info(f'  🔁 Smart retrace: {"ON" if self.enable_retrace else "OFF"}')
        # Printed unconditionally: "was prediction on for that flight?" must be
        # answerable from the log alone, without diffing a config file.
        self.get_logger().info(
            f'  🔮 Prediction   : {"ON" if self.enable_prediction else "OFF"}'
            + (f' (CPA horizon {self.cpa_horizon_s:.1f} s, '
               f'miss {self.cpa_miss_distance:.1f} m)'
               if self.enable_prediction else ''))
        self.get_logger().info('═══════════════════════════════════════════════')

    def _load_waypoints(self):
        """Flat parameter list -> list of ENU (x, y) tuples."""
        flat = list(self.get_parameter('waypoints').value)
        if len(flat) % 2 != 0:
            self.get_logger().error('waypoints must contain an even number of values!')
            flat = flat[:-1]
        frame = self.get_parameter('waypoint_frame').value.lower()

        pts = []
        for i in range(0, len(flat), 2):
            a, b = float(flat[i]), float(flat[i + 1])
            if frame == 'gazebo':
                # See module docstring: ENU = (-gz_y, +gz_x)
                pts.append((-b, a))
            else:
                pts.append((a, b))

        for i, (x, y) in enumerate(pts):
            self.get_logger().info(f'   WP{i + 1}: ENU ({x:7.1f}, {y:7.1f})')
        return pts

    # ══════════════════════════════════════════════════════════════════════
    #  Callbacks
    # ══════════════════════════════════════════════════════════════════════

    def now(self) -> float:
        """
        Seconds on the ROS clock.

        Every node in this stack runs use_sim_time:=true, so this follows
        Gazebo's /clock. The node used to measure durations with self.now()
        instead, which is wrong twice over: the simulation does not run at
        real-time speed (so a '12 second' takeoff settle was 12 wall seconds
        but ~6 sim seconds of flight), and a host clock step corrupts every
        elapsed-time calculation — one run reported a mission duration of
        22936 s after an NTP adjustment mid-flight.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        new = np.array([p.x, p.y, p.z])
        if self.last_pos is not None:
            self.total_distance += float(np.linalg.norm(new[:2] - self.last_pos[:2]))
        # Own velocity, needed to tell a static wall (closes at our speed)
        # from an obstacle actually moving toward us (closes faster).
        #
        # FALLBACK PATH ONLY. vel_cb takes the EKF's own estimate when MAVROS
        # is publishing it; this two-sample difference runs only if that topic
        # goes quiet for half a second, so the guards keep working on a
        # degraded estimate rather than on none at all.
        t = self.now()
        if t - self.vel_fused_t > 0.5:
            if self.prev_vel_pos is not None and t - self.prev_vel_t > 0.02:
                self.vel_enu = ((new[:2] - self.prev_vel_pos)
                                / (t - self.prev_vel_t))
            if self.prev_vel_pos is None or t - self.prev_vel_t > 0.02:
                self.prev_vel_pos, self.prev_vel_t = new[:2].copy(), t

        self.last_pos = new
        self.pos = new
        self.yaw = yaw_from_quat(msg.pose.orientation)
        self.roll, self.pitch = rp_from_quat(msg.pose.orientation)
        if not self.pose_ok:
            self.pose_ok = True
            self.get_logger().info(
                f'📡 Local pose online — ENU ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})')

    def vel_cb(self, msg: TwistStamped):
        """Own ENU velocity as the EKF estimates it.

        MAVROS publishes this in the same ENU frame as the local pose, so it
        drops straight into vel_enu with no conversion.
        """
        v = msg.twist.linear
        self.vel_enu = np.array([v.x, v.y])
        self.vel_fused_t = self.now()
        if not self.vel_ok:
            self.vel_ok = True
            self.get_logger().info(
                '📡 Fused velocity online — closing-rate maths now uses the '
                'EKF estimate instead of differenced pose.')

    def costmap_cb(self, msg: OccupancyGrid):
        self.map_res = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.costmap = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        if not self.costmap_ok:
            self.costmap_ok = True
            self.get_logger().info(
                f'🗺️  Costmap online — {msg.info.width}x{msg.info.height} '
                f'@ {self.map_res:.2f} m/cell')

    def scan_cb(self, msg: LaserScan):
        self.scan = msg
        if not self.scan_ok:
            self.scan_ok = True
            self.get_logger().info(
                f'📡 LiDAR online — {len(msg.ranges)} beams, '
                f'{msg.range_max:.0f} m range, frame "{msg.header.frame_id}"')

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1 — Polar histogram of the costmap around the drone
    # ══════════════════════════════════════════════════════════════════════

    def _bin_index(self, angles):
        """ENU bearings -> histogram bin indices, wrapping safely."""
        bin_w = 2.0 * math.pi / self.nbins
        return (np.floor(((angles + math.pi) % (2.0 * math.pi)) / bin_w)
                .astype(int) % self.nbins)

    def build_histogram(self):
        """
        Bin every obstacle within `lookahead` by its true ENU bearing from the
        drone, fusing TWO independent sources.

        Returns (blocked, nearest), both length `nbins`:
          blocked[i] : True if sector i is unsafe to fly through
          nearest[i] : distance to the closest obstacle in that sector (inf if none)

        WHY BOTH THE RAW SCAN AND THE COSTMAP
        -------------------------------------
        Originally this read the Nav2 costmap alone, and the drone flew into a
        skyscraper. Instrumenting scan-vs-costmap during the approach showed
        why — the costmap simply did not contain the building:

            pos=(-16.4, 48.4)  SCAN  7.62 m @181°   COSTMAP 16.11 m   630 cells
            pos=(-21.8, 51.0)  SCAN  2.25 m @181°   COSTMAP 22.37 m   157 cells
            pos=(-23.9, 52.3)  SCAN  0.30 m @204°   COSTMAP  9.92 m  8056 cells

        The LiDAR tracked the wall in perfectly, from 7.6 m down to contact,
        while the costmap reported open space and only caught up after impact.
        A costmap is a filtered, TF-gated, buffered product: the observation
        has to survive a TF lookup at the scan's timestamp, a buffer purge, a
        raytrace-clear pass and a publish cycle before it reaches us. At 3 m/s
        that pipeline delay is metres of travel, and nothing in it logs a
        warning when it loses.

        So the raw scan is now the PRIMARY sensor — one message, no TF, no
        buffering, the same data VFH was originally designed to consume — and
        the costmap is fused in as SECONDARY memory. The costmap still earns
        its place: it remembers obstacles that have passed out of the LiDAR's
        current sweep or fallen behind its 30 m range, and it carries the
        inflation layer. Taking the per-sector minimum means whichever source
        sees danger first wins, which is the correct bias for a safety check.

        ANGULAR ENLARGEMENT
        -------------------
        An obstacle at distance d occupies a half-angle of
        asin(safety_radius / d), so the same wall blocks a far wider arc when
        it is close. That one line replaces a whole table of hand-tuned
        distance thresholds.
        """
        blocked = np.zeros(self.nbins, dtype=bool)
        nearest = np.full(self.nbins, np.inf)
        self.sector_zoff = np.full(self.nbins, np.inf)
        if self.pos is None:
            return blocked, nearest

        bin_w = 2.0 * math.pi / self.nbins

        # ── Source 1: raw LaserScan (primary, ~0 latency) ─────────────────
        # The scan is body-fixed, so bearings are rotated into ENU by yaw to
        # match the costmap and the velocity commands.
        s = self.scan
        if s is not None and len(s.ranges) > 0:
            r = np.asarray(s.ranges, dtype=float)
            ang = s.angle_min + np.arange(r.size) * s.angle_increment + self.yaw
            # SELF-RETURN FILTER — see self_filter_range.
            ok = (np.isfinite(r) & (r > self.self_filter_range) &
                  (r < s.range_max) & (r <= self.lookahead))

            # TILT REJECTION — discard beams that hit the floor or ceiling.
            #
            # The LiDAR is only horizontal when the drone is level. Under hard
            # evasive banking the beam plane tips and sweeps the surface below,
            # and those terrain hits are indistinguishable from a wall unless
            # attitude is taken into account. That is what destroyed the drone
            # on the retrace: skimming 7.6 m over 7.5 m rooftops it reported
            # obstacles at 0.8-1.2 m from SIX bearings at once (12, -7, 67, 73,
            # -22, -27 deg). Nothing is 1 m away in six directions — it was
            # seeing the roof it was flying over. The code read that as a
            # 360-degree trap, banked harder, tipped the plane further, and
            # flipped (`Crash: Disarming: AngErr=165>30`).
            #
            # Each beam is rotated into the world frame to recover the height
            # of what it actually struck. The z-row of Rz(yaw)Ry(pitch)Rx(roll)
            # applied to a body-frame beam (cos a, sin a, 0) gives
            #     dz = -sin(pitch)*cos(a) + cos(pitch)*sin(roll)*sin(a)
            # so the hit sits r*dz above the drone. Anything far above or below
            # the flight plane is terrain, not an obstacle to steer around.
            ang_body = s.angle_min + np.arange(r.size) * s.angle_increment
            dz = (-math.sin(self.pitch) * np.cos(ang_body) +
                  math.cos(self.pitch) * math.sin(self.roll) * np.sin(ang_body))
            zoff = r * dz                      # height of the hit above the drone
            ok &= np.abs(zoff) <= self.max_beam_z_offset

            if np.any(ok):
                idx = self._bin_index(ang[ok])
                np.minimum.at(nearest, idx, r[ok])
                # Lowest hit per sector. Used to tell a surface underneath
                # (hits consistently below) from a canyon wall (hits at our own
                # level) — see the terrain check in run_leg().
                np.minimum.at(self.sector_zoff, idx, zoff[ok])

        # ── Source 2: Nav2 costmap (secondary, persistent + inflated) ─────
        #
        # ONLY while flying near cruise altitude. The costmap is a flat 2D
        # projection with NO notion of height: it accumulates every obstacle
        # ever observed, at whatever altitude the drone happened to be. For a
        # vehicle that deliberately climbs over things, that is poison.
        #
        # Observed failure: after an escape climb the drone sat at 22 m, 4.3 m
        # from WP1, with ZERO scan returns within the 10 m lookahead — visibly
        # empty sky — yet still reported AVOIDING and orbited the waypoint
        # indefinitely. The costmap was vetoing the goal direction using
        # buildings recorded at 4 m, which the drone was now 18 m above. And
        # because a horizontal LiDAR at 22 m returns nothing, no raytrace ever
        # arrives to clear those cells: the stale marks are immortal.
        #
        # The live scan has no such problem — it always describes the slice the
        # drone is actually flying through. So once we have climbed away from
        # cruise, the 2D map describes a plane we are no longer in, and the
        # scan alone is the trustworthy source.
        grid = self.costmap
        above_map_plane = self.pos[2] > self.cruise_alt + self.costmap_alt_band
        if above_map_plane:
            grid = None
        if grid is not None:
            # Unknown (-1) counts as free: an aerial LiDAR simply has not swept
            # there yet, and refusing to enter unobserved space would freeze
            # the drone permanently.
            rows, cols = np.nonzero((grid >= self.blocked_cost) & (grid != UNKNOWN))
            if rows.size:
                wx = self.map_origin[0] + (cols + 0.5) * self.map_res
                wy = self.map_origin[1] + (rows + 0.5) * self.map_res
                dx = wx - self.pos[0]
                dy = wy - self.pos[1]
                dist = np.hypot(dx, dy)
                keep = (dist <= self.lookahead) & (dist > 1e-3)
                if np.any(keep):
                    np.minimum.at(nearest,
                                  self._bin_index(np.arctan2(dy[keep], dx[keep])),
                                  dist[keep])

        # ── Angular enlargement (loop covers at most nbins entries) ───────
        for b in np.nonzero(np.isfinite(nearest))[0]:
            d = max(nearest[b], 1e-3)
            ratio = min(1.0, self.safety_radius / max(d, self.safety_radius * 0.5))
            widen = int(math.ceil(math.asin(ratio) / bin_w))
            for k in range(b - widen, b + widen + 1):
                blocked[k % self.nbins] = True

        return blocked, nearest

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — Pick a steering direction out of the free valleys
    # ══════════════════════════════════════════════════════════════════════

    def choose_heading(self, goal_bearing, blocked, nearest):
        """
        Select the best free bearing.

        Returns (heading, mode) where mode is one of:
          'clear'   straight at the goal, nothing in the way
          'dodge'   steering around an obstacle via the best free valley
          'trapped' every sector blocked — caller must escape vertically
        """
        bin_w = 2.0 * math.pi / self.nbins
        free = ~blocked

        def bin_of(angle):
            return int((wrap_pi(angle) + math.pi) / bin_w) % self.nbins

        def angle_of(b):
            return wrap_pi(-math.pi + (b + 0.5) * bin_w)

        # A heading is only flyable if a CORRIDOR around it is clear, not just
        # the single bin pointing down it. `inset` bins of clearance are
        # required on both sides.
        #
        # This check is the whole ballgame. An earlier version accepted the
        # goal bearing whenever its own bin happened to be free, even with
        # both neighbours blocked. The drone then aimed at a gap narrower than
        # itself and logged nonsense like "steering RIGHT (1° off goal)"
        # immediately before clipping a building. Requiring symmetric
        # clearance is what turns the histogram into an actual safety margin.
        inset = max(1, self.min_valley_bins // 2)

        def corridor_clear(b):
            return all(free[(b + k) % self.nbins] for k in range(-inset, inset + 1))

        goal_bin = bin_of(goal_bearing)
        if corridor_clear(goal_bin):
            return goal_bearing, 'clear'

        # Every bin whose corridor is clear is a legal escape. Scanning all
        # `nbins` directly is both simpler and safer than reconstructing
        # contiguous valleys, and at 72 bins it costs nothing.
        candidates = [b for b in range(self.nbins) if corridor_clear(b)]
        if not candidates:
            return None, 'trapped'

        # Prefer headings close to the goal, and (mildly) close to whatever we
        # chose last tick so the drone commits to one side of an obstacle
        # instead of oscillating across its centreline.
        best, best_score = None, float('inf')
        for b in candidates:
            angle = angle_of(b)
            score = self.goal_weight * abs(wrap_pi(angle - goal_bearing))
            if self.last_heading is not None:
                score += self.hysteresis_weight * abs(wrap_pi(angle - self.last_heading))
            # Mildly prefer directions with more open space ahead.
            d = nearest[b]
            if np.isfinite(d):
                score += 0.25 * (self.lookahead - min(d, self.lookahead)) / self.lookahead
            if score < best_score:
                best_score, best = score, angle

        return best, 'dodge'

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2a — Global route planning: A* over the costmap
    # ══════════════════════════════════════════════════════════════════════
    #
    # WHY A GLOBAL PLANNER IS NOT OPTIONAL
    # ------------------------------------
    # Everything above this point is a LOCAL planner: it sees one LiDAR scan
    # and picks a heading. That is provably insufficient for this mission.
    # Liang et al. (PLoS ONE 18(6):e0287177, 2023) build exactly this stack —
    # 360-degree LiDAR at 10 Hz into VFH into a PX4 quadrotor — and report the
    # same wall we hit: "Since the VFH algorithm is a local optimization
    # algorithm, the local flight path can be smoothed, but the smoothing
    # effect of the global flight trajectory is not as expected." Their stated
    # remedy is to add A* on top. Independent indoor stacks converge on the
    # same three-layer split: A* over an occupancy grid for the route, a
    # reactive layer for obstacles, pure pursuit to track the result.
    #
    # The failure this fixes: on the WP5->WP6 leg the drone sat at ENU
    # (-0.6, -14.1) with WP6 at (32, -8). Two 8 m houses at ENU (10,-15) and
    # (25,-15) straddle the straight line; WP6 is in the street at y = -8, so
    # the route is north into the corridor and then east. No goal-directed
    # local step proposes that, because every metre north increases the
    # distance to the target. A* over a map that contains both houses proposes
    # it immediately.
    #
    # The local layer is NOT replaced. A* supplies a carrot a few metres ahead
    # on a known-good route; build_histogram()/choose_heading() still fly the
    # drone there off the live scan, so moving obstacles and anything the map
    # has not seen are still handled reactively at 20 Hz.

    def _plan_grid(self):
        """Down-sample the costmap into a planning grid.

        Returns (cost, unknown, res, h, w, origin). The origin is snapshotted
        HERE, together with the array it belongs to: the planner runs on its
        own thread now, so costmap_cb can replace self.costmap mid-search and
        a separately-read origin could end up describing a different map.
        """
        if self.costmap is None:
            return None
        k = self.plan_downsample
        g = self.costmap
        origin = self.map_origin
        h, w = g.shape[0] // k, g.shape[1] // k
        if h < 2 or w < 2:
            return None
        crop = g[:h * k, :w * k]
        # Pool by MAX so a cell is only free when everything inside it is.
        # Unknown (-1) is treated as free here and penalised in the search:
        # an aerial LiDAR has swept almost nothing when the mission starts, so
        # a planner that refuses unobserved space would never return a first
        # route at all. The reactive layer is what keeps that honest.
        cost = np.where(crop < 0, 0, crop).astype(np.int16)
        cost = cost.reshape(h, k, w, k).max(axis=(1, 3))
        unknown = (crop == UNKNOWN).reshape(h, k, w, k).all(axis=(1, 3))
        return cost, unknown, self.map_res * k, h, w, origin

    def plan_path(self, goal_xy):
        """A* from the drone to goal_xy. Returns world-frame points or None."""
        pg = self._plan_grid()
        if pg is None:
            return None
        cost, unknown, res, h, w, origin = pg
        # NOTE the threshold: plan_lethal_cost (90), not blocked_cost (60).
        # The inflation layer paints a 2 m gradient around every building, and
        # a 7 m street is nothing BUT gradient. Refusing to plan through
        # inflation would close every corridor in the map; the gradient is
        # applied as a soft cost below instead, so the route prefers the
        # middle of the street without being forbidden from using it.
        blocked = cost >= self.plan_lethal_cost

        ox, oy = origin

        def to_cell(x, y):
            return int((y - oy) / res), int((x - ox) / res)

        def to_world(r, c):
            return (ox + (c + 0.5) * res, oy + (r + 0.5) * res)

        def nearest_free(r, c, radius=12):
            """Snap onto free space. Waypoints sit in streets and the drone
            hugs buildings, so both ends land in inflation routinely; that is
            not a reason to declare the route impossible."""
            if not (0 <= r < h and 0 <= c < w):
                return None
            if not blocked[r, c]:
                return r, c
            r0, r1 = max(0, r - radius), min(h, r + radius + 1)
            c0, c1 = max(0, c - radius), min(w, c + radius + 1)
            rs, cs = np.nonzero(~blocked[r0:r1, c0:c1])
            if rs.size == 0:
                return None
            d = (rs + r0 - r) ** 2 + (cs + c0 - c) ** 2
            i = int(np.argmin(d))
            return int(rs[i] + r0), int(cs[i] + c0)

        start = nearest_free(*to_cell(self.pos[0], self.pos[1]))
        goal = nearest_free(*to_cell(goal_xy[0], goal_xy[1]))
        if start is None or goal is None:
            return None
        if start == goal:
            return [to_world(*goal)]

        soft = cost.astype(np.float32) / 100.0
        SQ2 = math.sqrt(2.0)
        nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2))
        gr, gc = goal
        g_score = {start: 0.0}
        came = {}
        openh = [(0.0, start)]
        closed = set()
        found = False
        for _ in range(self.plan_max_expansions):
            if not openh:
                break
            _, cur = heapq.heappop(openh)
            if cur == goal:
                found = True
                break
            if cur in closed:
                continue
            closed.add(cur)
            cr, cc = cur
            base = g_score[cur]
            for dr, dc, step in nbrs:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w) or blocked[nr, nc]:
                    continue
                pen = 1.0 + self.plan_cost_penalty * float(soft[nr, nc])
                if unknown[nr, nc]:
                    pen += self.plan_unknown_penalty
                ng = base + step * pen
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    came[(nr, nc)] = cur
                    # Octile distance: admissible because every step costs at
                    # least its geometric length (pen >= 1.0).
                    ar, ac = abs(nr - gr), abs(nc - gc)
                    hcost = (ar + ac) + (SQ2 - 2.0) * min(ar, ac)
                    heapq.heappush(openh, (ng + hcost, (nr, nc)))
        if not found:
            return None

        cells = [goal]
        cur = goal
        while cur in came:
            cur = came[cur]
            cells.append(cur)
        cells.reverse()
        return [to_world(r, c) for r, c in cells]

    def carrot_on(self, path):
        """The point plan_carrot_dist along the route, measured from the point
        on it nearest the drone — so the carrot can never latch onto a piece
        of route already flown."""
        if not path:
            return None
        pts = np.asarray(path, dtype=float)
        i = int(np.argmin(np.hypot(pts[:, 0] - self.pos[0],
                                   pts[:, 1] - self.pos[1])))
        acc = 0.0
        for j in range(i, len(path) - 1):
            acc += math.hypot(path[j + 1][0] - path[j][0],
                              path[j + 1][1] - path[j][1])
            if acc >= self.plan_carrot_dist:
                return path[j + 1]
        return path[-1]

    def remaining_route(self):
        """Metres left to fly along the current route, or None if there is
        no route.

        This is the progress metric the livelock watchdog must use whenever a
        global route exists. Straight-line distance to the waypoint is the
        WRONG measure the moment a plan legitimately goes around a block: on
        the WP5->WP6 leg the correct route runs north into the street, which
        increases the distance to WP6 for ~20 m of flying. Judged on
        straight-line distance the drone looked stuck, check_stuck declared a
        local minimum, and the boundary follow hijacked the heading and
        dragged it from 30 m out to 49 m — fighting a planner that was right.
        Route length shrinks whenever the drone is actually making progress,
        whichever direction that progress points.
        """
        path, _ = self.route_snapshot()
        if not path:
            return None
        pts = np.asarray(path, dtype=float)
        i = int(np.argmin(np.hypot(pts[:, 0] - self.pos[0],
                                   pts[:, 1] - self.pos[1])))
        seg = pts[i:]
        total = float(np.hypot(seg[0, 0] - self.pos[0], seg[0, 1] - self.pos[1]))
        if len(seg) > 1:
            total += float(np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))))
        return total

    def update_plan(self, goal_xy):
        """Ask for a route to `goal_xy` and steer at whatever route is ready.

        NON-BLOCKING BY DESIGN — this runs in the 20 Hz control loop and must
        never search. It posts the goal for `planner_tick` and returns a
        carrot from the route that thread has already published, or None if
        there isn't one that matches the goal yet.
        """
        if not self.use_global_planner:
            return None
        self.plan_request = (float(goal_xy[0]), float(goal_xy[1]))
        path, goal = self.route_snapshot()
        if not path or goal is None:
            return None
        # The route in hand may still be the one planned toward the PREVIOUS
        # waypoint. Steering at that would fly the drone at a target it has
        # already left, so it is ignored until the planner catches up.
        if math.hypot(goal[0] - goal_xy[0], goal[1] - goal_xy[1]) > 0.5:
            return None
        return self.carrot_on(path)

    def route_snapshot(self):
        """The current route and the goal it was planned for, as a pair."""
        with self._plan_lock:
            return self.path, self.path_goal

    def invalidate_plan(self):
        """Throw the route away and make the planner start over.

        Called from the control loop whenever the existing route has been
        judged useless — a new waypoint, a stall, a runaway detour. Bumping
        the epoch is what stops a search already in flight from publishing
        its answer over the top of this decision.
        """
        self.plan_epoch += 1
        self.plan_target = None
        with self._plan_lock:
            self.path, self.path_goal = [], None

    def planner_tick(self):
        """Run A* on the planner thread. Never called from the control loop.

        The search costs 50-141 ms on this map against a 50 ms control period,
        which is precisely why it lives here instead of in `update_plan`.
        """
        if not self.use_global_planner or self.costmap is None:
            return
        goal = self.plan_request
        if goal is None:
            return
        now = self.now()
        moved = (self.plan_target is None or
                 math.hypot(goal[0] - self.plan_target[0],
                            goal[1] - self.plan_target[1]) > 0.5)
        if not (moved or now - self.last_plan_t >= self.replan_period_s):
            return
        epoch = self.plan_epoch
        self.last_plan_t = now
        self.plan_target = goal
        t0 = time.perf_counter()
        p = self.plan_path(goal)
        self.plan_ms = (time.perf_counter() - t0) * 1000.0
        # The control loop invalidated the route while this search was
        # running, so it answers a question that no longer applies. Drop it
        # rather than overwrite the fresh decision with a stale one.
        if epoch != self.plan_epoch:
            return
        with self._plan_lock:
            self.path = [] if p is None else p
            self.path_goal = None if p is None else goal
        if p is None:
            self.plan_fail += 1
            if self.plan_fail == 3:
                self.get_logger().warn(
                    '  🗺️  No global route to the target — falling back to '
                    'direct bearing plus boundary following.')
        else:
            if self.plan_fail >= 3:
                self.get_logger().info('  🗺️  Global route recovered.')
            self.plan_fail = 0
            self._throttled(
                f'  🗺️  Route replanned: {len(p)} cells, '
                f'{self.plan_ms:.0f} ms',
                20.0, key='replan')

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2b — Committed boundary following (tangent bug)
    # ══════════════════════════════════════════════════════════════════════
    #
    # WHY A PLAIN VFH+ CANNOT FINISH THIS MISSION
    # -------------------------------------------
    # VFH+ is memoryless: every tick it re-derives a heading from the goal
    # bearing and the current scan. That is fine against a convex obstacle,
    # and hopeless against a local minimum. On the WP5->WP6 leg the drone sat
    # at ENU (-0.6, -14.1) with WP6 at (32, -8). The straight line runs
    # through suburb_house_10 (ENU 10,-15) and suburb_house_6 (25,-15); WP6
    # itself sits in the street at y=-8, so the only route is NORTH into the
    # corridor first, then EAST. Goal attraction pulls east, the house pushes
    # back, and the drone oscillated 31 -> 36 -> 31 m from the waypoint
    # indefinitely, never once getting closer.
    #
    # The first attempt at a cure was a 4 s "break out toward the roomiest
    # bearing". It did not work and could not have: 4 s at 1.5 m/s is 6 m of
    # travel, after which goal attraction drags the drone straight back into
    # the same basin. The log shows exactly that — BREAKOUT, AVOIDING,
    # BREAKOUT, with the distance sawtoothing between 31 and 36 m.
    #
    # What actually escapes a local minimum is COMMITMENT plus a MEASURED
    # exit test, which is what the Bug family of planners provides. Latch a
    # turn direction, follow the obstacle boundary in that direction no
    # matter where the goal is, and only return to goal-seeking once the
    # drone is genuinely closer to the target than it was when it got stuck.
    # Because the direction cannot flip on a whim, the drone rounds the
    # building instead of bouncing off its face.

    def tangent_heading(self, goal_bearing, blocked, nearest, direction):
        """
        The obstacle-hugging heading: sweep away from the goal bearing in the
        LATCHED direction and take the first bearing whose corridor is clear.

        Sweeping from the goal (rather than picking the roomiest bearing)
        is what makes this boundary FOLLOWING rather than fleeing — the
        heading returned is the one that skims the obstacle edge as tightly
        as the clearance test allows, so the drone tracks around the building
        instead of running off into open ground.
        """
        bin_w = 2.0 * math.pi / self.nbins
        free = ~blocked
        inset = max(1, self.min_valley_bins // 2)
        goal_bin = int((wrap_pi(goal_bearing) + math.pi) / bin_w) % self.nbins
        step = 1 if direction > 0 else -1
        for k in range(self.nbins):
            b = (goal_bin + step * k) % self.nbins
            if all(free[(b + j) % self.nbins] for j in range(-inset, inset + 1)):
                return wrap_pi(-math.pi + (b + 0.5) * bin_w)
        return None

    def start_detour(self, dist_to_goal, goal_bearing, blocked, nearest):
        """Latch a turn direction and begin following the obstacle boundary."""
        # Pick the side whose tangent lies closer to the goal bearing: that is
        # the shorter way around, and it keeps the first move sensible.
        best_dir, best_off = 0, None
        for d in (1, -1):
            t = self.tangent_heading(goal_bearing, blocked, nearest, d)
            if t is None:
                continue
            off = abs(wrap_pi(t - goal_bearing))
            if best_off is None or off < best_off:
                best_dir, best_off = d, off
        if best_dir == 0:
            return False
        self.detour_dir = best_dir
        self.detour_entry_dist = dist_to_goal
        self.detour_since = self.now()
        self.dodge_count += 1
        self.get_logger().warn(
            f'  🔁 LOCAL MINIMUM at {dist_to_goal:.1f} m from the target — '
            f'following the boundary '
            f'{"LEFT" if best_dir > 0 else "RIGHT"} until '
            f'{dist_to_goal - self.detour_leave_margin:.1f} m')
        return True

    def end_detour(self, why):
        if self.detour_dir:
            self.get_logger().info(f'  ✅ Detour finished — {why}.')
        self.detour_dir = 0
        self.detour_entry_dist = None
        # The livelock watchdog must start from a clean slate, or it fires
        # again the instant the detour ends.
        self.best_dist = None
        self.progress_time = self.now()

    def flip_detour(self, dist_to_goal):
        """This way round is not working — commit to the other way."""
        self.detour_dir = -self.detour_dir
        self.detour_entry_dist = dist_to_goal
        self.detour_since = self.now()
        self.get_logger().warn(
            f'  🔁 Boundary follow timed out at {dist_to_goal:.1f} m — '
            f'reversing to '
            f'{"LEFT" if self.detour_dir > 0 else "RIGHT"}')

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3 — Emergency brake straight off the raw LaserScan
    # ══════════════════════════════════════════════════════════════════════

    def imminent_collision(self, nearest):
        """
        Omnidirectional time-to-contact guard. Returns (bearing, dist, ttc) for
        whatever is about to hit us, or None.

        WHY THE OLD CHECK MISSED A COLLISION
        ------------------------------------
        The previous version measured range only inside a +/-25 deg arc around
        the DIRECTION OF TRAVEL. That is blind to anything arriving from the
        side — which is exactly how the drone was lost on the WP3->WP4 leg.
        A 3 m obstacle moving at ~2.4 m/s crossed its path broadside; altitude
        held a rock-steady 4.0 m for the whole flight and then went 3.9 -> 0.6 m
        in one step, with ZERO emergency brakes logged, and ArduPilot reported
        `Crash: Disarming: AngErr=167>30`.

        The deeper problem is that VFH is a STATIC-GEOMETRY algorithm. It asks
        "which bearings are free right now" and has no velocity model at all,
        so it cannot know that a currently-free heading will be occupied in a
        second and a half. Against a moving obstacle the histogram is always
        one frame behind.

        This adds the missing derivative, cheaply: the per-sector nearest-range
        array is differenced against the previous tick to get a closing speed,
        and time-to-contact is dist / closing_speed. Reacting on TTC rather
        than on raw distance is what makes it usable in tight spaces — a wall
        the drone is flying ALONGSIDE has a closing speed near zero and an
        infinite TTC no matter how close it is, so corridors do not trip it,
        while anything genuinely converging is caught from any direction.
        """
        now = self.now()
        prev, prev_t = self.prev_nearest, self.prev_nearest_t
        self.prev_nearest, self.prev_nearest_t = nearest.copy(), now
        if prev is None:
            return None
        dt = now - prev_t
        if dt <= 1e-3:
            return None

        valid = np.isfinite(nearest) & np.isfinite(prev)
        if not np.any(valid):
            return None

        closing = np.zeros_like(nearest)
        closing[valid] = (prev[valid] - nearest[valid]) / dt

        # Subtract the closing rate our OWN motion accounts for.
        #
        # This is the discriminator that makes the guard usable. A STATIC wall
        # appears to close at exactly our speed toward it — so flying normally
        # at an obstacle produced a constant stream of false alarms: one run
        # logged 173 collision evades against just 12 genuine steering dodges,
        # all triggering at 4.9-6.0 m, with the guard fighting the VFH steering
        # that was already handling those walls perfectly well from 10 m out.
        #
        # Only an obstacle that is itself MOVING toward us closes faster than
        # our own velocity explains. Testing the excess means the guard fires
        # for the one job VFH genuinely cannot do — dynamic obstacles — and
        # stays silent for everything VFH already covers.
        bin_w = 2.0 * math.pi / self.nbins
        bearings = -math.pi + (np.arange(self.nbins) + 0.5) * bin_w
        own_closing = (self.vel_enu[0] * np.cos(bearings) +
                       self.vel_enu[1] * np.sin(bearings))
        excess = closing - own_closing

        ttc = np.full(self.nbins, np.inf)
        act = (valid & (closing > 0.3) & (nearest < self.evade_distance) &
               ((excess > self.dynamic_closing_thresh) |
                (nearest < self.hard_stop_distance)))
        ttc[act] = nearest[act] / np.maximum(closing[act], 1e-3)

        # PERSISTENCE FILTER — the single most important part of this guard.
        #
        # `nearest` is a min-per-sector statistic over a rotating, translating
        # sensor. Which beam lands in which bin changes constantly, so a bin's
        # value can jump metres between ticks for reasons that have nothing to
        # do with anything moving. Differencing it therefore manufactures
        # closing-rate spikes out of pure binning noise.
        #
        # Measured: 1002 evades against 81 genuine steering dodges in one run,
        # bearings sweeping smoothly (2, -7, -13, -18 deg ...) at a median
        # 4.6 m — the signature of a STATIC obstacle whose relative bearing
        # rotates as the drone flies past it. The thrashing drove ArduPilot
        # into an EKF failsafe and aborted the retrace, while the collision
        # this guard exists to prevent had happened exactly once.
        #
        # Requiring the SAME sector to look dangerous for several consecutive
        # ticks rejects binning noise, which is uncorrelated tick to tick,
        # while a genuinely converging obstacle persists trivially.
        flagged = np.isfinite(ttc) & (ttc <= self.collision_horizon_s)
        self.threat_ticks = np.where(flagged, self.threat_ticks + 1, 0)

        # Bypass the 0.3s persistence delay if the threat is inside the physical stopping
        # boundary. Waiting 0.3s at 1.5 m/s eats 0.45m of stopping distance!
        panic = flagged & (nearest < self.hard_stop_distance)
        ready = (flagged & (self.threat_ticks >= self.threat_persist_ticks)) | panic
        if not np.any(ready):
            return None
        # Cooldown: one evade may not re-fire every tick. Without it a single
        # persistent threat produces a burst of dozens of velocity reversals,
        # which is what turned this guard into the main cause of EKF failsafes.
        if now - self.last_evade_t < self.evade_cooldown_s:
            return None
        self.last_evade_t = now

        b = int(np.argmin(np.where(ready, ttc, np.inf)))
        bin_w = 2.0 * math.pi / self.nbins
        return wrap_pi(-math.pi + (b + 0.5) * bin_w), float(nearest[b]), float(ttc[b])

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3b — Where will it BE? (optional, enable_prediction)
    # ══════════════════════════════════════════════════════════════════════
    #
    # THE GAP THIS FILLS
    # ------------------
    # build_histogram() reduces the scan to one number per 5 deg sector, and
    # imminent_collision() differences that array. What survives is a RADIAL
    # range rate — how fast the gap is shrinking along each bearing. What is
    # destroyed is the TANGENTIAL component, and that is the half that says
    # where an obstacle is going.
    #
    # The consequence shows up as both kinds of error. A block crossing four
    # metres clear of the drone closes range the whole way in and reads as a
    # threat; a block on a genuine collision course holds a constant bearing,
    # so if the speeds are comparable its closing rate is unremarkable and it
    # reads as safe. The persistence filter, the cooldown and the committed
    # bearings above are all, in part, compensation for a signal that cannot
    # tell those two apart.
    #
    # Recovering the tangential component needs the scan treated as OBJECTS
    # rather than bearings: cluster the raw beams, follow each cluster's
    # centroid across frames, and the velocity falls out. Then the question
    # "will we collide" is answered by geometry instead of inference.
    #
    # WHAT THIS LAYER IS NOT
    # ----------------------
    # It is not a replacement for imminent_collision(). Tracking fails in ways
    # a reflex does not: an object passing behind a building drops its track,
    # two objects crossing can swap identities, and a centroid computed from
    # whichever face is visible drifts as the aspect changes. Every one of
    # those produces a confident, wrong velocity. So the reflex keeps first
    # refusal and this only ever ADDS a threat the reflex missed.

    def cluster_scan(self):
        """
        Group the raw scan into objects. Returns [(cx, cy, n_beams, min_range)]
        with centroids in WORLD coordinates.

        Beams are walked in angular order and split wherever the range steps by
        more than `track_cluster_gap` or drops out entirely. That is the whole
        algorithm — for a 2D lidar looking at convex blocks it is equivalent to
        the usual adjacency clustering and costs one pass.

        Filtering matches build_histogram() exactly — same self-return cut,
        same range limits, same tilt rejection — because a layer that
        disagreed with the histogram about what exists would be worse than no
        layer at all.
        """
        s = self.scan
        if s is None or self.pos is None or len(s.ranges) == 0:
            return []

        r = np.asarray(s.ranges, dtype=float)
        n = r.size
        ang_body = s.angle_min + np.arange(n) * s.angle_increment
        ang = ang_body + self.yaw

        ok = (np.isfinite(r) & (r > self.self_filter_range) &
              (r < s.range_max) & (r <= self.lookahead))
        dz = (-math.sin(self.pitch) * np.cos(ang_body) +
              math.cos(self.pitch) * math.sin(self.roll) * np.sin(ang_body))
        ok &= np.abs(r * dz) <= self.max_beam_z_offset
        if not np.any(ok):
            return []

        wx = self.pos[0] + r * np.cos(ang)
        wy = self.pos[1] + r * np.sin(ang)

        groups, cur = [], []
        for i in range(n):
            if not ok[i]:
                if cur:
                    groups.append(cur)
                    cur = []
                continue
            if cur and abs(r[i] - r[cur[-1]]) > self.track_cluster_gap:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)

        # THE SEAM. Beam 0 and beam n-1 are neighbours on a 360 deg scan, so an
        # object sitting behind the drone is split by the wrap into two pieces.
        # Left alone that produces two half-objects with centroids on either
        # side of the real one, each below the beam count that makes an object
        # credible, and both appearing and vanishing as the drone yaws — which
        # would inject exactly the phantom velocities this layer must not have.
        span = ang_body[-1] - ang_body[0] + s.angle_increment
        if (len(groups) > 1 and span >= 2.0 * math.pi - 1e-6
                and groups[0][0] == 0 and groups[-1][-1] == n - 1
                and abs(r[0] - r[n - 1]) <= self.track_cluster_gap):
            groups[0] = groups.pop() + groups[0]

        out = []
        for g in groups:
            if len(g) < self.track_min_points:
                continue
            idx = np.asarray(g)
            out.append((float(wx[idx].mean()), float(wy[idx].mean()),
                        len(g), float(r[idx].min())))
        return out

    def update_tracks(self):
        """
        Associate this frame's clusters with the existing tracks and update
        each one's position and velocity. No-op unless enable_prediction.

        Association is greedy nearest-neighbour against the DEAD-RECKONED
        position rather than the last measured one, so a track that is moving
        is matched where it should have got to. The gate is `track_assoc_radius`
        — at 1 m/s and 5-10 Hz an obstacle moves 0.1-0.2 m per frame, so 1.2 m
        is loose enough to survive a dropped frame and far tighter than the
        spacing between any two objects the drone actually meets.

        The filter is alpha-beta: a fixed-gain constant-velocity estimator.
        A full Kalman filter would buy a covariance this code has nothing to
        spend it on, while alpha-beta needs no tuning beyond two numbers and
        cannot go singular.

        Centroid drift is the known weakness. The centroid of the VISIBLE face
        moves as the aspect angle changes, so a static object seen from a
        moving drone reports a small spurious velocity. That is precisely what
        `track_min_speed` exists to swallow.
        """
        if not self.enable_prediction:
            return

        now = self.now()
        clusters = self.cluster_scan()

        # Forget anything not seen recently. Doing this BEFORE association
        # means an occluded object is dropped rather than coasted indefinitely
        # on a velocity nobody has confirmed since.
        self.tracks = [t for t in self.tracks
                       if now - t.t <= self.track_max_age_s]

        used = set()
        for tr in self.tracks:
            dt = now - tr.t
            if dt <= 1e-3:
                continue
            pred = tr.pos + tr.vel * dt
            best, best_d = -1, self.track_assoc_radius
            for i, c in enumerate(clusters):
                if i in used:
                    continue
                d = math.hypot(c[0] - pred[0], c[1] - pred[1])
                if d < best_d:
                    best, best_d = i, d
            if best < 0:
                continue                       # ages out if this persists
            used.add(best)
            resid = np.array([clusters[best][0], clusters[best][1]]) - pred
            tr.pos = pred + self.track_alpha * resid
            tr.vel = tr.vel + (self.track_beta / dt) * resid
            tr.t = now
            tr.hits += 1

        for i, c in enumerate(clusters):
            if i not in used:
                self.tracks.append(_Track((c[0], c[1]), now))

        # In a street the scan can hold dozens of clusters. Only the
        # best-established ones can ever clear track_confirm_frames anyway, so
        # capping by hit count bounds the O(tracks * clusters) association
        # without discarding anything that was going to matter.
        if len(self.tracks) > self.track_max_count:
            self.tracks.sort(key=lambda t: -t.hits)
            self.tracks = self.tracks[:self.track_max_count]

    def predict_threat(self):
        """
        Return (bearing, dist, t_cpa) for the soonest genuine intercept, or
        None. Same tuple shape as imminent_collision() so the brake state
        machine consumes it unchanged.

        Four gates, and the order is the point. Each one is a reason to say
        NOTHING, because the expensive error for this layer is the false
        positive: a drone that brakes for traffic that was never going to hit
        it stops making progress, and stopping in the open is how several of
        these flights have ended.

          1. UNCONFIRMED  — fewer than track_confirm_frames associations. A
                            two-frame velocity is a difference of two centroid
                            estimates and is mostly noise.
          2. NOT MOVING   — below track_min_speed it is scenery, which VFH and
                            the back-off reflex already handle better than a
                            predictor would. Above track_max_speed nothing in
                            this world moves that fast, so the association is
                            wrong and the velocity is fiction.
          3. NOT CLOSING  — t_cpa outside (0, cpa_horizon_s]. Negative means
                            the gap is already opening; too far ahead means
                            the constant-velocity assumption has expired
                            before the intercept arrives.
          4. WILL MISS    — d_cpa beyond cpa_miss_distance. This is the whole
                            reason the layer exists: something can be close,
                            and closing, and still pass harmlessly astern.
        """
        if not self.enable_prediction or self.pos is None:
            return None

        now = self.now()
        own_p = np.asarray(self.pos[:2], dtype=float)
        own_v = np.asarray(self.vel_enu[:2], dtype=float)

        best = None
        for tr in self.tracks:
            if tr.hits < self.track_confirm_frames:
                continue
            if now - tr.t > self.track_max_age_s:
                continue
            speed = float(np.linalg.norm(tr.vel))
            if not (self.track_min_speed <= speed <= self.track_max_speed):
                continue
            rel_p = tr.pos - own_p
            rel_v = tr.vel - own_v
            t_cpa, d_cpa = cpa(rel_p, rel_v)
            if not (0.0 < t_cpa <= self.cpa_horizon_s):
                continue
            if d_cpa > self.cpa_miss_distance:
                continue
            if best is None or t_cpa < best[2]:
                best = (math.atan2(float(rel_p[1]), float(rel_p[0])),
                        float(np.linalg.norm(rel_p)), float(t_cpa))
        return best

    # ══════════════════════════════════════════════════════════════════════
    #  Main state machine
    # ══════════════════════════════════════════════════════════════════════

    def main_loop(self):
        if not self.pose_ok:
            return

        if self.phase == 'WAIT_FCU':
            if self.state.connected:
                self.get_logger().info('✅ FCU connected.')
                self.phase = 'SET_GUIDED'
            return

        if self.phase == 'SET_GUIDED':
            if self.state.mode == 'GUIDED':
                self.get_logger().info('✅ GUIDED mode active.')
                self.phase = 'ARM'
            else:
                self._call(self.set_mode_cli, SetMode.Request(custom_mode='GUIDED'))
            return

        if self.phase == 'ARM':
            # Wait for perception BEFORE arming, never after.
            #
            # This check used to sit in TAKEOFF, i.e. after the motors were
            # already running. ArduPilot auto-disarms an armed-but-idle copter
            # after ~10 s (DISARM_DELAY), so whenever the costmap took longer
            # than that to activate, the sequence deadlocked: motors armed, FCU
            # disarmed them, and the state machine sat in TAKEOFF logging
            # "Waiting for the Nav2 costmap" forever at a dead aircraft. One run
            # burned its entire 10-minute budget without leaving the ground.
            #
            # Arming is also simply the wrong thing to do before the drone can
            # see: spinning props while blind buys nothing.
            if not self.costmap_ok:
                self._throttled('⏳ Waiting for the Nav2 costmap before arming...')
                return
            if not self.scan_ok:
                self._throttled('⏳ Waiting for the LiDAR before arming...')
                return
            if self.state.armed:
                self.get_logger().info('✅ Armed.')
                self.phase = 'TAKEOFF'
            else:
                self._call(self.arm_cli, CommandBool.Request(value=True))
            return

        if self.phase == 'TAKEOFF':
            # If the FCU disarmed us while we were getting here, go back and
            # re-arm rather than commanding takeoff into the void.
            if not self.state.armed:
                self.get_logger().warn('⚠️  Disarmed before takeoff — re-arming.')
                self.phase = 'ARM'
                return
            self._call(self.takeoff_cli,
                       CommandTOL.Request(altitude=float(self.cruise_alt)))
            self.takeoff_time = self.now()
            self.get_logger().info(f'🚀 Takeoff commanded to {self.cruise_alt:.1f} m.')
            self.phase = 'CLIMB'
            return

        if self.phase == 'CLIMB':
            reached = self.pos[2] >= self.cruise_alt * 0.9
            timeout = self.now() - self.takeoff_time > self.takeoff_settle_s
            if reached or timeout:
                self.mission_start = self.now()
                self.breadcrumbs = [self.pos[:2].copy()]
                self.get_logger().info('═══════════════════════════════════════════════')
                self.get_logger().info(f'  ✈️  MISSION START — {len(self.waypoints)} waypoints')
                self.get_logger().info('═══════════════════════════════════════════════')
                self.phase = 'MISSION'
            return

        # The FCU can take the aircraft away from us at any moment — an EKF
        # failsafe switches ArduPilot out of GUIDED into LAND, and the crash
        # detector disarms. Without this guard the node happily kept streaming
        # velocity setpoints and logging "CRUISE | ahead clear" for ~90 s after
        # ArduPilot had already landed and disarmed a tumbling airframe. Never
        # narrate a healthy flight over a vehicle that is no longer ours.
        if self.phase in ('MISSION', 'RETRACE') and self.check_fcu_control():
            return

        if self.phase == 'MISSION':
            self.run_leg(self.waypoints[self.wp_index], leg_kind='wp')
            return

        if self.phase == 'RETRACE':
            # Pure-pursuit style: skip every breadcrumb already inside the
            # lookahead radius and chase a genuinely distant one.
            #
            # Breadcrumbs are dropped 4 m apart while wp_radius is 3 m, so
            # steering straight at "the next one" meant the drone was
            # permanently arriving: 62 targets became 62 jerky hops, and the
            # bearing to a point 4 m away swings wildly with each step. That
            # thrash — not the outbound flight, which always completed — is why
            # aborts clustered in the retrace.
            while (self.retrace_index >= 0 and
                   float(np.linalg.norm(self.pos[:2] -
                                        self.breadcrumbs[self.retrace_index]))
                   < self.retrace_lookahead):
                self.retrace_index -= 1
            if self.retrace_index < 0:
                self.get_logger().info('🏁 Retrace complete — home reached.')
                self.phase = 'LAND'
                return
            self.run_leg(self.breadcrumbs[self.retrace_index], leg_kind='retrace')
            return

        if self.phase == 'ABORTED':
            # Control belongs to the FCU now (it is running its own failsafe).
            # Stay quiet rather than fighting it with stale setpoints.
            return

        if self.phase == 'LAND':
            self.send_velocity(0.0, 0.0, 0.0, 0.0)
            self._call(self.land_cli, CommandTOL.Request(altitude=0.0))
            self.get_logger().info('🛬 Landing commanded.')
            self.phase = 'LANDING'
            return

        if self.phase == 'LANDING':
            if not self.state.armed:
                el = self.now() - self.mission_start if self.mission_start else 0.0
                self.get_logger().info('═══════════════════════════════════════════════')
                self.get_logger().info('  ✅ MISSION COMPLETE — landed and disarmed')
                self.get_logger().info(f'  📏 Distance flown : {self.total_distance:.0f} m')
                self.get_logger().info(f'  ⏱️  Duration       : {el:.0f} s')
                self.get_logger().info(f'  🛡️  Obstacle dodges: {self.dodge_count}')
                self.get_logger().info('═══════════════════════════════════════════════')
                self.phase = 'DONE'
            return

    # ══════════════════════════════════════════════════════════════════════
    #  One navigation tick toward a single target
    # ══════════════════════════════════════════════════════════════════════

    def run_leg(self, target, leg_kind):
        tx, ty = float(target[0]), float(target[1])
        dx, dy = tx - self.pos[0], ty - self.pos[1]
        dist = math.hypot(dx, dy)

        # ── Arrival ───────────────────────────────────────────────────────
        # The LAST waypoint is the mission's stated destination, so it earns a
        # real terminal approach. With one radius for everything it was
        # declared "reached" up to wp_radius = 3 m out, which is most of a house
        # width — the drone was landing the mission a car's length from where it
        # was asked to be.
        #
        # The tight radius is not unconditional: a waypoint can sit inside an
        # obstacle's inflation, where no amount of patience will close the last
        # metre. After final_approach_timeout_s the tolerance reopens to
        # wp_radius so precision can never turn into a livelock.
        radius = self.wp_radius
        final = (leg_kind == 'wp' and self.wp_index == len(self.waypoints) - 1)
        if final and dist < self.wp_radius:
            if self.final_approach_since is None:
                self.final_approach_since = self.now()
                self.get_logger().info(
                    f'  🎯 FINAL APPROACH — closing on WP{self.wp_index + 1} to '
                    f'{self.final_wp_radius:.1f} m (now {dist:.2f} m)')
            if self.now() - self.final_approach_since <= self.final_approach_timeout_s:
                radius = self.final_wp_radius
            elif radius != self.final_wp_radius:
                self._throttled(
                    f'  🎯 Final approach timed out at {dist:.2f} m — '
                    f'accepting {self.wp_radius:.1f} m tolerance.', 10.0,
                    key='final_approach')
        if dist < radius:
            self.advance(leg_kind, miss=dist)
            return

        goal_bearing = math.atan2(dy, dx)

        # ── Global route ──────────────────────────────────────────────────
        # Steer at a carrot on the A* route rather than straight at the
        # waypoint. ARRIVAL is still judged against the real waypoint above —
        # the planner changes which way we go, never where we are going.
        carrot = self.update_plan((tx, ty))
        if carrot is not None:
            cdx, cdy = carrot[0] - self.pos[0], carrot[1] - self.pos[1]
            if math.hypot(cdx, cdy) > 0.5:
                goal_bearing = math.atan2(cdy, cdx)

        # ── Perceive ──────────────────────────────────────────────────────
        blocked, nearest = self.build_histogram()
        heading, mode = self.choose_heading(goal_bearing, blocked, nearest)

        # Time-to-contact is evaluated HERE, ahead of the trap and terrain
        # reflexes, because both of those can command ZERO lateral velocity
        # and neither may outrank an obstacle that is closing on us.
        #
        # This killed a flight on 2026-08-27. A 3 m dynamic block (top at
        # 3.0 m) drifted alongside the drone at 4.0 m. Its returns sit BELOW
        # the drone, which is exactly the signature the terrain check reads as
        # "surface beneath" — so the node logged "SURFACE BENEATH: 18 sectors
        # inside 2.0 m — climbing straight up, no lateral move" and stopped
        # moving sideways, while the TTC guard that had already fired three
        # EMERGENCY BRAKEs never got to run at all. It was pinned at the 4 m
        # ceiling where climbing buys nothing, the block closed to 0.7 m, and
        # the FCU took the aircraft in LAND.
        #
        # A moving obstacle is never terrain. When something is closing, the
        # answer is to get out of its way.
        threat = self.imminent_collision(nearest)

        # PREDICTIVE LAYER — strictly additive (enable_prediction, default off).
        #
        # The reflex above keeps first refusal: it is the guard that has been
        # earned over several crashes, and this may add a threat it missed but
        # may never remove one it found. What this buys is the case the reflex
        # is blind to by construction — an obstacle whose closing rate looks
        # ordinary because it is crossing rather than approaching, which is the
        # geometry of every dynamic block on this course.
        self.update_tracks()
        if threat is None:
            threat = self.predict_threat()
            if threat is not None:
                # Throttled, not gated on `braking`. The first version tested
                # `not self.braking` on the assumption that the flag stays set
                # for the whole encounter — it does not. The RESUME branch
                # below clears it on any tick the histogram reads 'clear',
                # which for crossing traffic is most of them, so the guard
                # reset constantly and one 279 m flight logged this 86 times.
                self._throttled(
                    f'  🔮 PREDICTED INTERCEPT — {threat[1]:.1f} m at '
                    f'{math.degrees(threat[0]):.0f}°, CPA in {threat[2]:.1f} s',
                    2.0, key='predict')

        # ── Proximity back-off: outranks EVERY other state ────────────────
        #
        # Five separate flights ended the same way. The logs differ in which
        # state was active — TRAPPED 360, TERRAIN-CLIMB, AVOIDING — but the
        # last two lines are always the same shape:
        #
        #     WP6/6 | 18.5 m to go | alt 3.9 m | ahead 0.8m | AVOIDING
        #     roll 17.9 -> 66.2 -> 95.9 deg, az -> 0.00   (disarmed, falling)
        #
        # Something was 0.8 m away and the drone was still commanding motion.
        # Every higher-level behaviour here — trap squeeze, terrain climb,
        # boundary follow, brake sidestep — can command a velocity that closes
        # the last metre, because each one is reasoning about where to GO. None
        # of them is responsible for simply not touching anything.
        #
        # This is that responsibility, and it is placed first so nothing can
        # override it: inside backoff_range, drive directly away from the
        # nearest return and do nothing else. It cannot deadlock, because
        # retreating always increases the distance that triggered it.
        # Only what we are MOVING TOWARD can be hit. The first version of
        # this retreated from the nearest return in any direction, which in a
        # 7 m street under a 2 m inflation radius meant retreating from walls
        # it was harmlessly passing abeam: 69 back-offs in 11 minutes and the
        # drone livelocked between 21 and 26 m from WP6, never closing. A
        # guard that stops the drone touching things must not also stop it
        # flying down corridors.
        if np.any(np.isfinite(nearest)):
            bin_w = 2.0 * math.pi / self.nbins
            bearings = -math.pi + (np.arange(self.nbins) + 0.5) * bin_w
            speed_now = float(np.linalg.norm(self.vel_enu[:2]))
            if speed_now > 0.2:
                course = math.atan2(float(self.vel_enu[1]), float(self.vel_enu[0]))
            elif self.last_heading is not None:
                course = self.last_heading
            else:
                course = 0.0
            ahead = np.array([abs(wrap_pi(b - course)) for b in bearings]) \
                < math.radians(self.backoff_arc)
            cand = np.where(ahead & np.isfinite(nearest), nearest, np.inf)
            b_min = int(np.argmin(cand))
            d_min = float(cand[b_min])
            # HYSTERESIS on the exit: once retreating, the drone must open a
            # real gap before the reflex lets go, or it chatters in and out of
            # back-off at the boundary and each transition is a velocity step.
            trigger = self.backoff_range
            if self.backoff_dir is not None:
                trigger += self.backoff_release_margin
            if d_min < trigger:
                bin_w = 2.0 * math.pi / self.nbins
                toward = wrap_pi(-math.pi + (b_min + 0.5) * bin_w)
                want = wrap_pi(toward + math.pi)
                away, held = self.committed_bearing('backoff', want)
                self.backoff_dir = away
                self.status = 'BACK-OFF'
                self._throttled(
                    f'  🚨 BACK-OFF — {d_min:.2f} m at '
                    f'{math.degrees(toward):.0f}°, retreating along '
                    f'{math.degrees(away):.0f}° (held {held:.1f} s)',
                    2.0, key='backoff')
                self.send_velocity(math.cos(away) * self.backoff_speed,
                                   math.sin(away) * self.backoff_speed,
                                   self.climb_vz(), 0.0)
                return
            self.release_bearing('backoff')
            self.backoff_dir = None

        # ── Resting on something ──────────────────────────────────────────
        # Checked before any other reflex, because none of them are meaningful
        # while the airframe is in contact with a surface.
        if self.check_contact():
            self.status = 'CONTACT-ESCAPE'
            room = np.where(np.isfinite(nearest), nearest, self.lookahead)
            b = int(np.argmax(room))
            bin_w = 2.0 * math.pi / self.nbins
            esc = wrap_pi(-math.pi + (b + 0.5) * bin_w)
            # The ONE case allowed to exceed the altitude cap. Sitting on a
            # 7.5 m roof with a 4 m ceiling, climb_bounded() returns 0 and the
            # drone can never get off — the cap would enforce the deadlock it
            # exists to prevent. The climb is only ever 1.5 m above wherever
            # contact happened, and the normal descent decay pulls it straight
            # back to cruise_altitude once the drone is flying again.
            self.target_alt = max(self.target_alt, self.pos[2] + 1.5)
            self.send_velocity(math.cos(esc) * self.contact_escape_speed,
                               math.sin(esc) * self.contact_escape_speed,
                               self.climb_speed, 0.0)
            return

        # ── Trapped: climb over the obstacle ──────────────────────────────
        if mode == 'trapped':
            # A trap must PERSIST before it earns a climb.
            #
            # Previously any single trapped tick immediately added escape_step
            # to the target altitude, and `trapped_since` was cleared by the
            # very next clear tick. In cluttered terrain the histogram flickers
            # in and out of "no legal corridor" many times a second, so the
            # altitude ratcheted up +3 m at a time far faster than it could
            # bleed off: one mission logged 207 trap events and spent most of
            # its length pinned at max_escape_altitude, cruising over the
            # course instead of weaving through it.
            #
            # Holding position for `trap_confirm_s` costs nothing — the drone
            # is stopped and climbing gently either way — and it distinguishes
            # a real dead end from momentary sensor geometry.
            self.trap_ticks += 1
            confirm = max(1, int(self.trap_confirm_s / self.dt))
            if self.trap_ticks == confirm:
                self.target_alt = min(self.pos[2] + self.escape_step,
                                      self.max_escape_alt)
                self.dodge_count += 1
                self.get_logger().warn(
                    f'  🆘 TRAPPED 360° — climbing to {self.target_alt:.1f} m to fly over')
            elif self.trap_ticks > confirm and self.trap_ticks % max(1, int(1.5 / self.dt)) == 0:
                # Still boxed in after climbing — go higher, IF there is
                # anywhere higher to go. Under the 4 m envelope there is not,
                # and the old code still logged "escalating to 4.0 m" while
                # sitting at 4.0 m: an escalation that cannot happen,
                # announced as though it had. That line cost real time during
                # the 2026-08-28 post-mortem, because it reads as the escape
                # working when the truth is that the squeeze below is the
                # only thing acting.
                room = self.max_escape_alt - self.target_alt
                if room > 0.5:
                    self.target_alt = min(self.target_alt + self.escape_step,
                                          self.max_escape_alt)
                    self.get_logger().warn(f'  🆘 Still trapped — escalating to '
                                           f'{self.target_alt:.1f} m')
                else:
                    self.get_logger().warn(
                        f'  🆘 Still trapped at the {self.max_escape_alt:.1f} m '
                        f'ceiling — no climb available, squeezing out sideways')
            # At the ceiling, climbing cannot help; hovering there is a
            # deadlock. Fall back to the roomiest bearing and push horizontally
            # — a best-effort squeeze beats freezing in mid-air.
            if self.pos[2] >= self.max_escape_alt - 0.5:
                b = int(np.argmax(np.where(np.isfinite(nearest), nearest, self.lookahead)))
                bin_w = 2.0 * math.pi / self.nbins
                want = wrap_pi(-math.pi + (b + 0.5) * bin_w)
                # Latched for the same reason the retreat is: "the roomiest
                # bearing" is an argmax over a noisy per-bin minimum, so it
                # hops between comparable gaps every tick and the squeeze
                # turns into a rotating velocity command.
                esc, _ = self.committed_bearing('squeeze', want)
                self.status = 'TRAPPED-SQUEEZE'
                self.send_velocity(math.cos(esc) * self.min_speed,
                                   math.sin(esc) * self.min_speed,
                                   0.0, 0.0)
                return
            self.status = 'TRAPPED-CLIMB'
            # Actively back away from the closest obstacle while climbing.
            #
            # This used to command zero horizontal velocity, but a zero COMMAND
            # is not zero MOTION: arriving at cruise_speed with WPNAV_ACCEL
            # braking, the drone coasts over a metre before it actually stops.
            # A trap is detected at close range by definition, so that coast
            # carried it into the very wall it was climbing to escape —
            # "TRAPPED 360 -> 4.5 m -> 7.5 m -> 10.5 m" then
            # `Crash: Disarming: AngErr=170>30`.
            #
            # Pushing away from the nearest return both kills the momentum and
            # buys clearance, at min_speed so the estimator stays undisturbed.
            b_near = int(np.argmin(np.where(np.isfinite(nearest), nearest, np.inf)))
            bin_w = 2.0 * math.pi / self.nbins
            away = wrap_pi(-math.pi + (b_near + 0.5) * bin_w + math.pi)
            # Yaw frozen: a vertical trap escape is the worst possible moment
            # to add rotation to the estimator's workload.
            self.send_velocity(math.cos(away) * self.min_speed,
                               math.sin(away) * self.min_speed,
                               self.climb_vz(), 0.0)
            return

        self.trap_ticks = 0

        # ── Terrain proximity: surrounded at knife range means we are ON
        #    something, not next to it ───────────────────────────────────────
        #
        # Real obstacles do not surround a drone at ~1 m on every side at once.
        # That pattern only occurs when the beam plane is grazing a surface the
        # drone is sitting just above. Skimming 7.6 m over 7.5 m rooftops, the
        # log showed obstacles at 0.8-1.2 m from SIX bearings simultaneously
        # (12, -7, 67, 73, -22, -27 deg). The code read it as a wall closing in,
        # dashed sideways at cruise speed, banked harder, tipped the beam plane
        # deeper into the roof, and flipped
        # (`Crash: Disarming: AngErr=165>30`).
        #
        # Note this cannot be solved by filtering on hit height: the roof was
        # only ~0.35 m below, which no sensible z-threshold can separate from a
        # wall at flight level. The giveaway is the COUNT of simultaneously
        # close sectors, not their geometry.
        #
        # The correct response to "I am on top of something" is to go straight
        # up — never to accelerate horizontally, which tips the beam plane
        # further and deepens the illusion.
        # Being surrounded is NOT enough — a narrow canyon between towers looks
        # identical by sector count alone. The discriminator is HEIGHT: a
        # surface underneath produces hits consistently BELOW the drone (the
        # banked beam plane digs into it), whereas canyon walls sit at our own
        # level. Without this the drone froze 23 m up inside an 8 m skyscraper
        # gap, "climbing over a roof" that was really two towers beside it.
        near_mask = nearest < self.terrain_close_range
        below = near_mask & (self.sector_zoff < -self.terrain_below_margin)
        close = int(np.count_nonzero(below))
        # And never take this branch when we cannot actually climb: at the
        # ceiling climb_bounded() is 0, so it would command zero velocity and
        # `return` before any logging — a silent hover that ran 74 minutes
        # producing no output whatsoever.
        # `climb_bounded() > 0` is not a strong enough gate. With
        # max_escape_altitude pinned to the 4 m cruise ceiling — the specified
        # envelope — it is true whenever the drone has sagged even a few
        # centimetres below 4.0 m, so this branch fired at 3.99 m, bought at
        # most 0.01 m of height, and surrendered ALL lateral mobility to do it.
        #
        # That is how the 2026-08-27 flight ended. dyn_block_3 is a 3 m cube
        # spanning z 2.5..5.5 m that tracks back and forth across the WP5->WP6
        # leg at 1 m/s. When it came alongside, the tilted beam plane read its
        # face as "surface beneath" (16, then 17, then 20 sectors), the node
        # commanded zero lateral velocity, and the block closed in and hit the
        # airframe: 2.8 g on the accelerometer, roll through 105 deg to 157 deg,
        # inverted, into the ground.
        #
        # Requiring REAL headroom makes the branch inert under the 4 m cap,
        # where climbing was never an answer, while leaving it intact if the
        # ceiling is ever raised.
        #
        # 2026-08-28: BOTH of those guards failed open at once and the same
        # thing happened again, this time on the WP1 leg.
        #
        #   EMERGENCY BRAKE - obstacle 5.1 m at 178 deg
        #   EMERGENCY BRAKE - obstacle 2.8 m at -172 deg
        #   BACK-OFF - 1.40 m at -178 deg, retreating along 2 deg
        #   SURFACE BENEATH: 16 sectors inside 2.0 m - climbing straight up,
        #                    no lateral move
        #   BACK-OFF - 0.78 m
        #   Crash: Disarming: AngErr=165>30
        #
        # `headroom > 0.5` was satisfied because braking had let the drone sag
        # below 3.5 m, and `threat is None` was satisfied because detection
        # flickers through the persistence filter — one tick without a report
        # is all this branch needs to surrender lateral movement.
        #
        # Height cannot separate the two cases: a 3 m block beside a drone at
        # 3.4 m genuinely does return hits below it. Two things can, and both
        # are checked now:
        #
        #   * the brake window, not the instantaneous threat. A moving block
        #     does not stop existing because one scan missed it, and
        #     brake_until already carries brake_hold_s of memory.
        #   * plain proximity. Ground is what you hover ABOVE; it is not what
        #     sits 1.4 m from you while the back-off reflex retreats from it.
        #     Anything inside backoff_range disqualifies the terrain reading
        #     outright, whatever the sector count says.
        headroom = self.max_escape_alt - self.pos[2]
        # `nearest` holds only inf or finite positives — it is seeded to inf
        # and written solely by np.minimum.at from ranges already filtered
        # finite and positive — so np.min gives the closest obstacle if there
        # is one and inf if the sky is empty. Both are exactly what this gate
        # wants, so the mask-and-branch the check used to carry was dead
        # weight on a 20 Hz path.
        closest = float(np.min(nearest))
        if (close >= self.terrain_close_sectors
                and headroom > 0.5
                and threat is None
                and self.now() >= self.brake_until
                and closest > self.backoff_range):
            self.status = 'TERRAIN-CLIMB'
            self.target_alt = min(max(self.target_alt, self.pos[2]) + self.escape_step,
                                  self.max_escape_alt)
            if not self.terrain_logged:
                self.terrain_logged = True
                self.get_logger().warn(
                    f'  ⬆️  SURFACE BENEATH: {close} sectors inside '
                    f'{self.terrain_close_range:.1f} m — climbing straight up, '
                    f'no lateral move')
            self.send_velocity(0.0, 0.0, self.climb_bounded(), 0.0)
            return
        self.terrain_logged = False

        # ── Dynamic obstacle: EMERGENCY BRAKE, then look for space ────────
        #
        # The previous reflex drove AWAY from the threat at speed. Every firing
        # was an instantaneous velocity reversal, and with the guard made more
        # sensitive it fired 68 times against 10 genuine steering decisions in
        # one flight — straight back into "Vibration compensation ON",
        # "EKF Failsafe: changed to Land Mode", crash-disarm.
        #
        # Braking is a far gentler input than reversing: commanding zero
        # horizontal velocity lets ArduPilot decelerate on its own smoothly,
        # instead of demanding thrust in the opposite direction. It is also the
        # correct behaviour. A moving obstacle crossing the route does not need
        # to be outrun — it needs to be waited out, or stepped around once the
        # drone is stationary enough to pick a gap deliberately.
        #
        #   1. BRAKE     kill forward motion, hold altitude
        #   2. ASSESS    stopped, and the way to the waypoint is still covered?
        #                look for a clear corridor
        #   3. SIDESTEP  ease into that corridor at min_speed
        #      or HOLD   nothing open yet — wait for the obstacle to pass
        #   4. RESUME    the moment the path to the waypoint clears
        if threat is not None:
            t_bearing, t_dist, t_ttc = threat
            if not self.braking:
                self.braking = True
                self.dodge_count += 1
                self.get_logger().warn(
                    f'  🛑 EMERGENCY BRAKE — obstacle {t_dist:.1f} m at '
                    f'{math.degrees(t_bearing):.0f}°, impact in {t_ttc:.1f} s')
            # Keep braking for a while after the last detection, so a single
            # crossing obstacle does not cause stop/go stuttering.
            self.brake_until = self.now() + self.brake_hold_s

        if self.now() < self.brake_until:
            speed_now = float(np.linalg.norm(self.vel_enu))

            # 1. BRAKE — still carrying momentum.
            if speed_now > self.brake_stop_speed:
                self.status = 'EMERGENCY-BRAKE'
                self.send_velocity(0.0, 0.0, self.climb_vz(), 0.0)
                return

            # 4. RESUME — the route opened up while we were stopped.
            if mode == 'clear':
                self.brake_until = 0.0
                self.braking = False
                self.get_logger().info('  ✅ Path clear again — resuming.')
            else:
                # 2/3. ASSESS then SIDESTEP, or HOLD if nothing is open.
                if heading is not None:
                    self.status = 'BRAKE-SIDESTEP'
                    self.send_velocity(math.cos(heading) * self.min_speed,
                                       math.sin(heading) * self.min_speed,
                                       self.climb_vz(), 0.0)
                else:
                    self.status = 'BRAKE-HOLD'
                    self.send_velocity(0.0, 0.0, self.climb_vz(), 0.0)
                return
        else:
            self.braking = False

        # ── Normal steering ───────────────────────────────────────────────
        if mode == 'dodge':
            if not self.was_avoiding:
                self.dodge_count += 1
                side = 'LEFT' if wrap_pi(heading - goal_bearing) > 0 else 'RIGHT'
                self.get_logger().warn(
                    f'  ⚠️  Obstacle on course — steering {side} '
                    f'({math.degrees(abs(wrap_pi(heading - goal_bearing))):.0f}° off goal)')
            self.was_avoiding = True
            self.status = 'AVOIDING'
        else:
            if self.was_avoiding:
                self.get_logger().info('  ✅ Path clear — resuming direct course.')
            self.was_avoiding = False
            self.status = 'CRUISE'
            self.drop_breadcrumb()

        # Sink back toward cruise altitude on every clear tick.
        #
        # This is a DECAY, not a latch. Two earlier designs both failed: a
        # "clear for N continuous seconds" test never completed because
        # avoidance fires in short bursts at 10 Hz, and a decaying-credit
        # variant lost the race too whenever dodges were frequent. Either way
        # the drone finished whole missions pinned at max_escape_altitude,
        # cruising over the course instead of through it.
        #
        # Bleeding the target down a little on each clear tick is
        # self-correcting: the trap handler snaps the target back up the
        # instant the drone is boxed in again, so the two behaviours simply
        # servo against each other and no state can get stuck.
        if mode == 'clear' and self.target_alt > self.cruise_alt and self.safe_to_descend():
            step = self.descend_rate * self.dt
            self.target_alt = max(self.cruise_alt, self.target_alt - step)
            if self.target_alt <= self.cruise_alt + 1e-6 and not self.back_at_cruise:
                self.back_at_cruise = True
                self.get_logger().info(
                    f'  ⬇️  Path clear — back down to cruise '
                    f'({self.cruise_alt:.1f} m)')
        elif mode != 'clear':
            self.back_at_cruise = False

        # Judge progress along the ROUTE when there is one. Straight-line
        # distance to the waypoint necessarily grows while rounding a block,
        # and the watchdog must not read that as being stuck.
        route_left = self.remaining_route()
        self.check_stuck(dist if route_left is None else route_left,
                         nearest, goal_bearing, blocked,
                         have_route=route_left is not None,
                         goal_dist=dist)

        # A healthy route outranks a boundary follow. The detour exists for
        # when the planner has no answer; once it does have one, following the
        # obstacle edge is strictly worse than following the plan.
        if self.detour_dir and route_left is not None and self.plan_fail == 0:
            self.end_detour('a global route is available again')

        # ── Committed boundary following owns the course ───────────────────
        # A heading re-derived from the goal on every tick is precisely what
        # walked the drone into the local minimum, so while a detour is live
        # the goal gets no vote until one of the exit tests passes.
        if self.detour_dir:
            if dist < self.detour_entry_dist - self.detour_leave_margin:
                self.end_detour(f'closed to {dist:.1f} m from the target')
            elif mode == 'clear' and dist < self.detour_entry_dist:
                self.end_detour('line to the target is open again')
            elif dist > self.detour_entry_dist + self.detour_max_drift:
                # A detour that is losing ground badly is not rounding an
                # obstacle, it is running away. Abort and replan.
                self.end_detour(f'drifted {dist - self.detour_entry_dist:.1f} m '
                                f'further from the target')
                self.invalidate_plan()
            elif self.now() - self.detour_since > self.detour_timeout_s:
                self.flip_detour(dist)
            else:
                t = self.tangent_heading(goal_bearing, blocked, nearest,
                                         self.detour_dir)
                if t is None:
                    self.end_detour('no clear tangent remains')
                else:
                    heading = t
                    self.status = ('DETOUR-L' if self.detour_dir > 0
                                   else 'DETOUR-R')

        # Rate-limit how fast the commanded course may swing.
        #
        # The unlimited version produced sequences like
        # LEFT(136 deg) -> RIGHT(55) -> LEFT(105) -> RIGHT(70) within a couple
        # of seconds: 61 of 137 dodges on one leg commanded >80 deg course
        # changes. Slamming the velocity vector around like that (on top of the
        # yaw slew) is what drove ArduPilot's estimator into
        # "EKF3 Roll/Pitch inconsistent" and finally an EKF failsafe.
        # Obstacles do not move that fast; the oscillation was ours.
        #
        # This now sits AFTER the detour override so the boundary-following
        # heading is slewed like any other; the old break-out bypassed the
        # limiter entirely and stepped the course in a single tick.
        if self.last_heading is not None:
            delta = wrap_pi(heading - self.last_heading)
            max_step = math.radians(self.max_heading_rate) * self.dt
            if abs(delta) > max_step:
                heading = wrap_pi(self.last_heading + math.copysign(max_step, delta))

        self.last_heading = heading

        # Speed: ease off when the way ahead is tight or the target is near.
        head_bin = int((wrap_pi(heading) + math.pi) / (2 * math.pi / self.nbins)) % self.nbins
        room = nearest[head_bin]
        speed = self.cruise_speed
        if np.isfinite(room):
            speed *= float(np.clip(room / self.lookahead, 0.25, 1.0))
        if mode == 'dodge':
            speed = min(speed, self.cruise_speed * 0.7)
        speed = max(self.min_speed, min(speed, self.cruise_speed))
        speed = min(speed, max(self.min_speed, dist))   # decelerate into the target
        # Inside the terminal zone, and ONLY there, the min_speed floor is
        # lifted. min_speed exists so the drone never stalls while travelling;
        # applied to the last metre it guarantees an overshoot, because 0.8 m/s
        # carries the drone straight through a 1 m ball between ticks.
        if final and dist < self.wp_radius:
            speed = min(speed, max(self.final_approach_speed, dist * 0.5))

        self.send_velocity(math.cos(heading) * speed,
                           math.sin(heading) * speed,
                           self.climb_vz(),
                           self.yaw_rate(heading, active_avoidance=(mode == 'dodge')))

        self.log_progress(leg_kind, dist, mode, room)

    # ══════════════════════════════════════════════════════════════════════
    #  Mission progression
    # ══════════════════════════════════════════════════════════════════════

    def advance(self, leg_kind, miss=None):
        # New target: the progress watchdog must start fresh, or the distance
        # to the *previous* waypoint would count as the bar to beat. Any
        # boundary follow ends with the leg that provoked it — its exit test
        # is written in terms of a target that no longer applies.
        self.best_dist = None
        self.detour_dir = 0
        self.detour_entry_dist = None
        self.invalidate_plan()
        if leg_kind == 'wp':
            self.wp_index += 1
            self.drop_breadcrumb(force=True)
            el = self.now() - self.mission_start
            acc = f', miss {miss:.2f} m' if miss is not None else ''
            self.get_logger().info(
                f'📍 WAYPOINT {self.wp_index}/{len(self.waypoints)} REACHED  '
                f'[{self.total_distance:.0f} m flown, {el:.0f} s, '
                f'{self.dodge_count} dodges{acc}]')
            self.final_approach_since = None
            if self.wp_index >= len(self.waypoints):
                if self.enable_retrace and len(self.breadcrumbs) > 1:
                    self.retrace_index = len(self.breadcrumbs) - 1
                    self.get_logger().info('═══════════════════════════════════════════════')
                    self.get_logger().info(
                        f'  🔁 SMART RETRACE — following {len(self.breadcrumbs)} '
                        f'proven-clear breadcrumbs home')
                    self.get_logger().info('═══════════════════════════════════════════════')
                    self.phase = 'RETRACE'
                else:
                    self.phase = 'LAND'
        else:
            self.retrace_index -= 1
            if self.retrace_index >= 0 and self.retrace_index % 3 == 0:
                self.get_logger().info(
                    f'  🔁 Retracing… {self.retrace_index} breadcrumbs to home')

    def drop_breadcrumb(self, force=False):
        """
        Record the current position, but only from a CLEAR moment — the trail
        must describe a corridor we know is flyable, not the inside of a dodge.
        """
        if self.phase != 'MISSION':
            return
        here = self.pos[:2].copy()
        if not self.breadcrumbs:
            self.breadcrumbs.append(here)
            return
        if force or np.linalg.norm(here - self.breadcrumbs[-1]) >= self.breadcrumb_spacing:
            self.breadcrumbs.append(here)

    def check_stuck(self, dist_to_goal, nearest=None,
                    goal_bearing=None, blocked=None, have_route=False,
                    goal_dist=None):
        """
        Livelock breaker: climb if we stop getting CLOSER TO THE GOAL.

        This deliberately measures progress toward the target, not raw
        displacement. The displacement version missed the failure mode it
        existed to catch: skimming an obstacle face, the drone tracked
        34.9 -> 37.0 -> 33.5 -> 50.2 -> 46.1 m from the waypoint, sliding
        15 m back and forth without ever approaching it. Displacement was
        large every single window, so the watchdog stayed silent while the
        drone orbited a local minimum indefinitely.

        A reactive planner like VFH has no global view and *will* find local
        minima. Climbing out is the cheap escape, but it is unavailable here:
        max_escape_altitude is pinned to the 4 m cruise ceiling on purpose.
        So when climbing is barred the watchdog hands over to committed
        boundary following (see start_detour), which escapes in the plane.
        """
        now = self.now()
        # A boundary follow is ALLOWED to move away from the goal — that is
        # the entire point of rounding a building. It carries its own timeout,
        # so the generic watchdog must keep quiet while one is running or it
        # would abort the escape it just ordered.
        if self.detour_dir:
            return
        if self.best_dist is None or dist_to_goal < self.best_dist - 1.0:
            # Genuine progress: reset the clock and the bar.
            self.best_dist = dist_to_goal
            self.progress_time = now
            return
        if now - self.progress_time < self.stuck_timeout_s:
            return
        self.progress_time = now
        self.dodge_count += 1

        # The climb target must be measured from where the drone ACTUALLY IS,
        # not from `target_alt`.
        #
        # `target_alt` decays back toward cruise on every clear tick, so the
        # two can diverge badly. On 2026-08-27 the drone was resting at 7.51 m
        # on a rooftop while target_alt had bled down to ~4 m; this line
        # computed 4 + 3 = 7.0 m, printed "climbing to 7.0 m to break out",
        # and handed climb_vz() a NEGATIVE error. The livelock breaker spent
        # the rest of the mission pressing the drone DOWN into the roof it was
        # supposed to be escaping, re-firing every 15 s forever.
        #
        # The other two climb sites (trap escape, terrain climb) already
        # clamped to self.pos[2]. This one did not.
        want = min(max(self.target_alt, self.pos[2]) + self.escape_step,
                   self.max_escape_alt)
        if want > self.pos[2] + 0.2:
            self.target_alt = want
            self.get_logger().warn(
                f'  🔄 No progress toward goal in {self.stuck_timeout_s:.0f} s '
                f'(best {self.best_dist:.1f} m, now {dist_to_goal:.1f} m) — '
                f'climbing to {self.target_alt:.1f} m to break out')
            return

        # Climbing is barred — normally because max_escape_altitude is pinned
        # to cruise_altitude, which is the specified envelope. A watchdog that
        # can only climb is then a watchdog that does nothing at all, so break
        # the local minimum sideways instead: commit to the roomiest bearing
        # well off the current course for a few seconds, long enough to leave
        # the basin the planner keeps falling back into.
        # With a global route in hand, a stall means the drone is failing to
        # FOLLOW the route, not that the route is wrong. The right response is
        # a fresh plan against the latest map — not a blind boundary follow,
        # which would only fight the planner. Only when replanning has failed
        # to help twice running do we fall back to the tangent bug.
        if have_route:
            self.route_stall += 1
            if self.route_stall < self.route_stall_limit:
                self.get_logger().warn(
                    f'  🔄 No progress along the route in '
                    f'{self.stuck_timeout_s:.0f} s — forcing a replan '
                    f'({self.route_stall}/{self.route_stall_limit}).')
                self.invalidate_plan()
                self.best_dist = None
                return
            # Close to the target, a boundary follow is strictly harmful.
            # It commits to one direction for up to detour_timeout_s, which
            # near the goal means walking AWAY from it: on 2026-08-27 the drone
            # closed to 16.8 m of WP6 and the detour dragged it back out to
            # 44.9 m. Within no_detour_radius, keep replanning instead — the
            # route is short there and a new map almost always yields one.
            if goal_dist is not None and goal_dist < self.no_detour_radius:
                self.get_logger().warn(
                    f'  🔄 Stalled {goal_dist:.1f} m from the target — inside '
                    f'{self.no_detour_radius:.0f} m, so replanning rather than '
                    f'detouring away.')
                self.invalidate_plan()
                self.best_dist = None
                self.route_stall = 0
                return
            self.get_logger().warn(
                '  🔄 Replanning did not help — falling back to boundary '
                'following.')
            self.route_stall = 0

        if goal_bearing is not None and blocked is not None:
            self.get_logger().warn(
                f'  🔄 No progress in {self.stuck_timeout_s:.0f} s '
                f'(best {self.best_dist:.1f} m, now {dist_to_goal:.1f} m) — '
                f'at the {self.max_escape_alt:.1f} m ceiling, so going AROUND.')
            self.start_detour(dist_to_goal, goal_bearing, blocked, nearest)

    # ══════════════════════════════════════════════════════════════════════
    #  Actuation
    # ══════════════════════════════════════════════════════════════════════

    def committed_bearing(self, key, want, commit_s=None):
        """Latch an escape bearing so the drone commits instead of spinning.

        Every reflex that picks a direction from the CURRENT scan — retreat
        from the nearest return, squeeze toward the roomiest gap — recomputes
        that direction each tick. With obstacles on several sides the winning
        bin hops between them and the commanded velocity vector rotates,
        which the airframe answers with a sequence of reversals.

        That is what killed the 2026-08-28 flight: retreats at -13 deg, then
        -68 deg, then -98 deg inside six seconds, on top of three emergency
        brakes, ending in "EKF variance: over thresholds" and a failsafe to
        LAND with the altitude estimate jumping 4.0 m -> -6.2 m.

        So a bearing, once chosen, is FLOWN. It is replaced only after
        commit_s has given it a chance to work, and only if the direction now
        wanted is genuinely opposed (>90 deg) rather than jitter. This is the
        same commitment the tangent-bug detour already makes for the same
        reason; these reflexes simply never got it.
        """
        if commit_s is None:
            commit_s = self.backoff_commit_s
        now = self.now()
        held = self._latched.get(key)
        if (held is None
                or (now - held[1] > commit_s
                    and abs(wrap_pi(want - held[0])) > math.radians(90.0))):
            self._latched[key] = (want, now)
            return want, 0.0
        return held[0], now - held[1]

    def release_bearing(self, key):
        """Forget a latched bearing, so the next entry picks a fresh one."""
        self._latched.pop(key, None)

    def climb_vz(self):
        """P-controller onto the current target altitude."""
        err = self.target_alt - self.pos[2]
        return float(np.clip(0.8 * err, -1.0, self.climb_speed))

    def climb_bounded(self) -> float:
        """
        Full-rate climb, but never above the escape ceiling.

        The reflex branches (terrain climb, evade) used to command
        `self.climb_speed` directly — a hardcoded +1.5 m/s with NO altitude
        limit. `max_escape_altitude` only ever constrained `target_alt`, which
        those branches bypass entirely, so every firing ratcheted the drone
        upward with nothing to pull it back.

        It showed up worst on the retrace, which re-crosses the dense districts
        and therefore re-triggers the reflexes: median altitude 4.1 m outbound
        versus 7.2 m returning, peaking at 50.3 m against a 22 m ceiling. It is
        also self-reinforcing — climbing over a tall building puts a new
        surface under the drone, which reads as terrain proximity, which climbs
        again.
        """
        if self.pos is None or self.pos[2] >= self.max_escape_alt:
            return 0.0
        return self.climb_speed

    def yaw_rate(self, heading, active_avoidance=False):
        """
        Gently turn the nose toward travel.

        The LiDAR is a full 360-degree sensor, so unlike a forward-facing
        camera the nose direction buys NOTHING for perception — it is purely
        cosmetic. It is not free, though: slewing yaw at up to 86 deg/s while
        the velocity vector is also swinging around is exactly the input that
        pushed ArduPilot's EKF into "Check mag field (xy diff:159>100)" and
        "EKF3 Roll/Pitch inconsistent", ending in an EKF failsafe, a forced
        LAND and a crash-disarm.

        So the gain is halved, the rate is capped well below the old limit,
        and yaw is frozen outright while avoidance is actively steering —
        that is precisely when the horizontal command is changing fastest and
        the estimator can least afford extra rotation.
        """
        if active_avoidance:
            return 0.0
        return float(np.clip(self.yaw_gain * wrap_pi(heading - self.yaw),
                             -self.max_yaw_rate, self.max_yaw_rate))

    def safe_to_descend(self) -> bool:
        """
        True only if nothing is remembered directly BENEATH the drone.

        A horizontal 2D LiDAR is blind downward. It reports "clear" while the
        drone hovers centimetres above a rooftop, because the beam plane passes
        over the roof entirely. Descending on that evidence is how the drone
        died: bleeding back toward cruise it went 8.9 -> 8.0 -> 7.7 m over the
        suburb, whose roofs top out at 7.5 m, flew into one and flipped
        (`Crash: Disarming: AngErr=150>30, Accel=0.1<3.0`).

        The Nav2 costmap is the right instrument here, and this is the mirror
        image of the rule in build_histogram(). There, ABOVE cruise, the
        costmap must be IGNORED for horizontal steering because it describes a
        plane we deliberately left. Here it is the only thing that knows what
        fills the XY column beneath us, having watched it from lower down.
        Same 2D map, opposite conclusions, because the question differs:
        "what may I fly THROUGH" vs "what may I descend INTO".

        Unknown (-1) counts as free, so the drone can still sink over terrain
        it has never observed. That is the residual limit of a horizontal-only
        sensor: a rooftop first encountered from above is genuinely invisible.
        A downward rangefinder is the real answer, and the Tarot 650 has one.
        """
        grid = self.costmap
        if grid is None or self.pos is None:
            return True

        r = int(math.ceil(self.descend_clear_radius / self.map_res))
        cx = int((self.pos[0] - self.map_origin[0]) / self.map_res)
        cy = int((self.pos[1] - self.map_origin[1]) / self.map_res)
        h, w = grid.shape
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return True

        patch = grid[y0:y1, x0:x1]
        blocked = bool(np.any((patch >= self.blocked_cost) & (patch != UNKNOWN)))
        if blocked and not self.hold_alt_logged:
            self.hold_alt_logged = True
            self.get_logger().info(
                f'  ⏸️  Holding {self.pos[2]:.1f} m — obstacle remembered below; '
                f'not descending until clear of it')
        elif not blocked:
            self.hold_alt_logged = False
        return not blocked

    def check_contact(self) -> bool:
        """
        True while the airframe is resting on a surface.

        THE FAILURE THIS EXISTS FOR
        ---------------------------
        A horizontal 2D LiDAR is geometrically incapable of seeing a flat
        surface it is level with — the beam plane passes straight over the
        roof, so a drone standing ON a building reports open air in all 360
        directions. Every other guard in this node is built on that scan, so
        every other guard was blind at once.

        On 2026-08-27 the drone settled dead centre on suburb_roof_3 (top
        exactly 7.50 m) at z = 7.51 m and stayed there for the rest of the
        mission. Nothing detected it. It remained armed and in GUIDED, the
        distance to WP6 sat frozen at 13.7 m, and the node printed
        "ahead clear | CRUISE" once a second while commanding cruise speed
        into a rooftop.

        THE SIGNAL
        ----------
        Not the scan — the physics. A flying multirotor always carries some
        body-rate noise; a landed one carries none. The IMU during that
        deadlock read roll -0.08°, pitch -0.17°, p/q/r all exactly 0.0 °/s and
        az 9.81 m/s²: pure gravity, no motion whatsoever. So the test is simply
        "am I asking for motion and not getting any", which needs no new sensor
        and holds against ANY unmodelled surface — a roof, a ledge, a canopy,
        or a wall the drone is being pushed into.
        """
        if self.pos is None:
            return False
        commanding = self.commanded_speed() > self.min_speed * 0.5
        moving = float(np.linalg.norm(self.vel_enu)) > self.contact_speed
        if not commanding or moving:
            self.moving_since = None
            if self.contact:
                self.contact = False
                self.get_logger().info(
                    f'  ✅ Free of the surface at {self.pos[2]:.2f} m — resuming.')
            return False

        now = self.now()
        if self.moving_since is None:
            self.moving_since = now
            return False
        if now - self.moving_since < self.contact_confirm_s:
            return False
        if not self.contact:
            self.contact = True
            self.dodge_count += 1
            self.get_logger().error(
                f'  🪨 SURFACE CONTACT at {self.pos[2]:.2f} m — commanding '
                f'{self.commanded_speed():.2f} m/s, measuring '
                f'{float(np.linalg.norm(self.vel_enu)):.2f} m/s. Resting on '
                f'something the horizontal LiDAR cannot see — climbing off.')
        return True

    def check_fcu_control(self) -> bool:
        """
        True if the FCU is no longer under our control, meaning the caller must
        stop flying. See the call site for why this exists.
        """
        if not self.state.armed:
            self.get_logger().error('═══════════════════════════════════════════════')
            self.get_logger().error('  ⛔ FCU DISARMED MID-MISSION — aborting.')
            self.get_logger().error('     Usually an ArduPilot crash-detect or a')
            self.get_logger().error('     failsafe. Check /mavros/statustext/recv.')
            self.get_logger().error('═══════════════════════════════════════════════')
            self.send_velocity(0.0, 0.0, 0.0, 0.0)
            self.phase = 'ABORTED'
            return True
        if self.state.mode != 'GUIDED':
            self.get_logger().error(
                f'⛔ FCU left GUIDED (now {self.state.mode!r}) — the autopilot has '
                f'taken control (EKF/battery/RC failsafe). Aborting mission.')
            self.send_velocity(0.0, 0.0, 0.0, 0.0)
            self.phase = 'ABORTED'
            return True
        return False

    def send_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """
        Publish an ENU velocity setpoint, RATE-LIMITED.

        MAVROS's setpoint_velocity plugin converts this Twist from ENU to NED
        and emits SET_POSITION_TARGET_LOCAL_NED. It is a WORLD-frame command,
        not body-frame — which is exactly why every calculation in this node
        stays in ENU alongside the costmap.

        WHY THE SLEW LIMIT
        ------------------
        Every state in this node used to hand the FCU a STEP. The emergency
        brake was the worst offender — cruise speed to zero in a single tick is
        a demand of ~15 m/s², 1.5 g — but the trap squeeze, the terrain climb
        and the brake→sidestep transition all did the same thing, sometimes
        reversing direction outright.

        Measured against the IMU on the 2026-08-27 flight, windows starting at
        each EMERGENCY BRAKE showed:

            attitude   6.7° median / 14.9° peak   (cruise: 0.2°)
            body rates  35 °/s median / 56 °/s peak (cruise: 1.2 °/s)
            roll-rate sign flips 1.7-2.5 per second

        A ~30x jump, and the sign flipping is the tell: ArduPilot was not
        stopping, it was RINGING — chasing an unreachable demand, overshooting,
        and correcting. That is the violent wobble, and it is also what has
        repeatedly pushed the estimator into "Vibration compensation ON" and
        an EKF failsafe.

        Capping how fast the SETPOINT may move converts every one of those
        steps into a ramp the vehicle can actually track. It costs almost
        nothing in stopping distance: 1.5 m/s at 2 m/s² stops in 0.75 s and
        0.56 m, and the 10 -> 20 Hz decision loop gives back more than that.

        This is deliberately the single choke point — every state in the node
        publishes through here, so none of them can reintroduce a step.
        """
        now = self.now()
        dt = now - self.last_cmd_t if self.last_cmd_t else self.dt
        dt = float(np.clip(dt, 1e-3, 0.5))
        self.last_cmd_t = now

        # Horizontal: limit the change as a VECTOR, so a direction reversal is
        # limited exactly as firmly as a speed change. Limiting the components
        # separately would let a 180° turn through at full rate.
        want_xy = np.array([float(vx), float(vy)])
        delta = want_xy - self.cmd[:2]
        step = float(np.linalg.norm(delta))
        lim = self.max_accel * dt
        if step > lim:
            delta *= lim / step
        self.cmd[:2] += delta
        self.cmd[2] += float(np.clip(float(vz) - self.cmd[2],
                                     -self.max_accel_z * dt,
                                     self.max_accel_z * dt))

        msg = Twist()
        msg.linear.x = float(self.cmd[0])
        msg.linear.y = float(self.cmd[1])
        msg.linear.z = float(self.cmd[2])
        msg.angular.z = float(yaw_rate)
        self.vel_pub.publish(msg)

    def commanded_speed(self) -> float:
        """Horizontal speed we are currently ASKING for (post-slew)."""
        return float(np.linalg.norm(self.cmd[:2]))

    def _call(self, client, request):
        if not client.service_is_ready():
            return
        client.call_async(request)

    # ══════════════════════════════════════════════════════════════════════
    #  Logging & RViz visualisation
    # ══════════════════════════════════════════════════════════════════════

    def _throttled(self, text, period=2.0, key=None):
        """Rate-limited log line.

        `key` gives a message its OWN clock. Without it every throttled
        message shared one timestamp, so the 2 s progress line starved every
        longer-period message in the node: the route-replanned line asks for
        20 s of quiet and never got it, because the progress line reset the
        shared clock every 2 s. The effect was a planner running perfectly
        with no evidence of it in the log.
        """
        k = 'default' if key is None else key
        now = self.now()
        if now - self._log_times.get(k, 0.0) > period:
            self._log_times[k] = now
            self.last_log = now
            self.get_logger().info(text)

    def log_progress(self, leg_kind, dist, mode, room):
        label = (f'WP{self.wp_index + 1}/{len(self.waypoints)}' if leg_kind == 'wp'
                 else f'RETRACE {self.retrace_index}')
        icon = '🔭' if mode == 'clear' else '🛡️'
        room_s = f'{room:.1f}m' if np.isfinite(room) else 'clear'
        self._throttled(
            f'{icon} {label} | {dist:5.1f} m to go | alt {self.pos[2]:4.1f} m | '
            f'ahead {room_s} | {self.status}')

    def publish_markers(self):
        if self.pos is None:
            return
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def base(ns, mid, mtype, scale, color):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
            m.pose.orientation.w = 1.0
            return m

        # Global A* route — cyan line, with the carrot as a small sphere.
        # Snapshotted, because the planner thread may swap it mid-draw.
        route, _ = self.route_snapshot()
        if len(route) > 1:
            m = base('route', 0, Marker.LINE_STRIP, (0.25, 0.0, 0.0),
                     (0.1, 0.85, 0.95, 0.9))
            m.points = [Point(x=float(x), y=float(y), z=float(self.cruise_alt))
                        for x, y in route]
            arr.markers.append(m)
            c = self.carrot_on(route)
            if c is not None:
                k = base('route', 1, Marker.SPHERE, (0.9, 0.9, 0.9),
                         (1.0, 0.85, 0.1, 0.95))
                k.pose.position = Point(x=float(c[0]), y=float(c[1]),
                                        z=float(self.cruise_alt))
                arr.markers.append(k)

        # Waypoints — green ahead, grey once visited
        for i, (x, y) in enumerate(self.waypoints):
            done = i < self.wp_index
            m = base('waypoints', i, Marker.SPHERE, (2.0, 2.0, 2.0),
                     (0.45, 0.45, 0.45, 0.55) if done else (0.1, 0.9, 0.3, 0.85))
            m.pose.position = Point(x=x, y=y, z=self.cruise_alt)
            arr.markers.append(m)

            t = base('wp_labels', i, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 2.0),
                     (1.0, 1.0, 1.0, 0.9))
            t.pose.position = Point(x=x, y=y, z=self.cruise_alt + 2.5)
            t.text = f'WP{i + 1}'
            arr.markers.append(t)

        # Planned route
        route = base('route', 0, Marker.LINE_STRIP, (0.25, 0.0, 0.0),
                     (0.2, 0.6, 1.0, 0.6))
        route.points = [Point(x=0.0, y=0.0, z=self.cruise_alt)] + [
            Point(x=x, y=y, z=self.cruise_alt) for x, y in self.waypoints]
        arr.markers.append(route)

        # Breadcrumb trail (the retrace corridor)
        if len(self.breadcrumbs) > 1:
            bc = base('breadcrumbs', 0, Marker.LINE_STRIP, (0.35, 0.0, 0.0),
                      (1.0, 0.65, 0.0, 0.9))
            bc.points = [Point(x=float(b[0]), y=float(b[1]), z=self.cruise_alt)
                         for b in self.breadcrumbs]
            arr.markers.append(bc)

        # Blocked sectors — a red fan showing exactly what the drone refuses
        # to fly into right now.
        blocked, nearest = self.build_histogram()
        fan = base('blocked_sectors', 0, Marker.LINE_LIST, (0.08, 0.0, 0.0),
                   (1.0, 0.15, 0.1, 0.85))
        bin_w = 2.0 * math.pi / self.nbins
        for b in range(self.nbins):
            if not blocked[b]:
                continue
            ang = -math.pi + (b + 0.5) * bin_w
            r = min(nearest[b], self.lookahead) if np.isfinite(nearest[b]) else self.lookahead
            fan.points.append(Point(x=float(self.pos[0]), y=float(self.pos[1]),
                                    z=float(self.pos[2])))
            fan.points.append(Point(x=float(self.pos[0] + math.cos(ang) * r),
                                    y=float(self.pos[1] + math.sin(ang) * r),
                                    z=float(self.pos[2])))
        arr.markers.append(fan)

        # Chosen heading
        if self.last_heading is not None:
            h = base('heading', 0, Marker.ARROW, (0.3, 0.6, 0.0),
                     (0.1, 1.0, 0.9, 0.95))
            h.points = [
                Point(x=float(self.pos[0]), y=float(self.pos[1]), z=float(self.pos[2])),
                Point(x=float(self.pos[0] + math.cos(self.last_heading) * 6.0),
                      y=float(self.pos[1] + math.sin(self.last_heading) * 6.0),
                      z=float(self.pos[2]))]
            arr.markers.append(h)

        # Status readout
        st = base('status', 0, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 1.6),
                  (1.0, 1.0, 0.3, 1.0))
        st.pose.position = Point(x=float(self.pos[0]), y=float(self.pos[1]),
                                 z=float(self.pos[2] + 3.0))
        st.text = (f'{self.phase} | {self.status}\n'
                   f'alt {self.pos[2]:.1f} m | dodges {self.dodge_count}')
        arr.markers.append(st)

        self.marker_pub.publish(arr)

        # Flown path
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = stamp
        for b in self.breadcrumbs:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position = Point(x=float(b[0]), y=float(b[1]), z=self.cruise_alt)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = MissionAvoidanceNode()
    # Two threads, and only two: one runs the control group (the 20 Hz loop,
    # the markers and every subscription, all mutually exclusive exactly as
    # they were under the default executor), the other runs A*. Adding more
    # would buy nothing — there is no third thing to overlap.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted — holding position.')
        node.send_velocity(0.0, 0.0, 0.0, 0.0)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
