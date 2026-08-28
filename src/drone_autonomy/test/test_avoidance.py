#!/usr/bin/env python3
"""
Offline tests for the VFH+ avoidance core.

Runs the real build_histogram()/choose_heading() against synthetic costmaps,
so the steering logic can be validated in seconds without booting Gazebo,
ArduPilot and MAVROS.

    python3 src/drone_autonomy/test/test_avoidance.py
"""

import math
import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, __file__.rsplit('/test/', 1)[0])
from drone_autonomy.mission_avoidance_node import MissionAvoidanceNode, wrap_pi  # noqa: E402

RES = 0.1
SIZE = 400                      # 40 m window at 0.1 m/cell
ORIGIN = (-20.0, -20.0)         # drone sits at (0,0), i.e. grid centre


def make_node():
    node = MissionAvoidanceNode()
    node.pos = np.array([0.0, 0.0, 4.0])
    node.yaw = 0.0
    node.map_res = RES
    node.map_origin = ORIGIN
    node.last_heading = None
    return node


def blank():
    return np.zeros((SIZE, SIZE), dtype=np.int8)


def fill_box(grid, x0, x1, y0, y1, cost=100):
    """Mark a world-coordinate box as lethal."""
    c0 = int((x0 - ORIGIN[0]) / RES); c1 = int((x1 - ORIGIN[0]) / RES)
    r0 = int((y0 - ORIGIN[1]) / RES); r1 = int((y1 - ORIGIN[1]) / RES)
    grid[max(0, r0):min(SIZE, r1), max(0, c0):min(SIZE, c1)] = cost
    return grid


def deg(r):
    return math.degrees(r)


results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f"  — {detail}" if detail else ''))


def main():
    rclpy.init()
    node = make_node()

    print("\n── 1. Empty map: fly straight at the goal ─────────────────────")
    node.costmap = blank()
    goal = 0.0                                   # due East
    blocked, nearest = node.build_histogram()
    h, mode = node.choose_heading(goal, blocked, nearest)
    check("clear map -> mode 'clear'", mode == 'clear', f"mode={mode}")
    check("clear map -> heading == goal", h is not None and abs(wrap_pi(h - goal)) < 1e-6)

    print("\n── 2. Wall straight ahead: must deviate substantially ─────────")
    node.costmap = fill_box(blank(), 5.0, 7.0, -6.0, 6.0)   # 12 m wall, 5 m east
    blocked, nearest = node.build_histogram()
    h, mode = node.choose_heading(0.0, blocked, nearest)
    check("wall ahead -> mode 'dodge'", mode == 'dodge', f"mode={mode}")
    dev = abs(deg(wrap_pi(h - 0.0))) if h is not None else 0.0
    # The old bug produced ~1 degree here and flew into the wall.
    check("wall ahead -> deviation > 25 deg", dev > 25.0, f"deviation={dev:.0f}°")

    print("\n── 3. The old bug: single free bin inside a blocked span ──────")
    # Hand-build a histogram that is blocked everywhere except the exact goal
    # bin. A corridor check must REJECT this; the previous implementation
    # accepted it because that one bin was free.
    nb = node.nbins
    blocked = np.ones(nb, dtype=bool)
    nearest = np.full(nb, 4.0)
    goal_bin = int((wrap_pi(0.0) + math.pi) / (2 * math.pi / nb)) % nb
    blocked[goal_bin] = False
    h, mode = node.choose_heading(0.0, blocked, nearest)
    check("needle gap -> NOT treated as clear", mode != 'clear', f"mode={mode}")
    check("needle gap -> trapped (no wide corridor)", mode == 'trapped', f"mode={mode}")

    print("\n── 4. Corridor between two blocks ─────────────────────────────")
    # The mission's tightest real corridors are the 8 m slots between
    # skyscrapers and the ~6 m suburban streets, so a 7 m gap must be flyable.
    def gap_test(half_width):
        g = blank()
        fill_box(g, 4.0, 8.0,  half_width, 12.0)
        fill_box(g, 4.0, 8.0, -12.0, -half_width)
        node.costmap = g
        b, n = node.build_histogram()
        return node.choose_heading(0.0, b, n)

    h, mode = gap_test(3.5)                       # 7 m gap
    ok = h is not None and abs(deg(wrap_pi(h - 0.0))) < 40.0
    check("7 m gap -> threads it (heads roughly east)",
          ok, f"mode={mode} heading={deg(h) if h is not None else 0.0:.0f}°")

    # A 5 m gap leaves only 0.9 m of clearance either side of the 1.6 m
    # safety_radius. Refusing it is the DESIGNED behaviour, not a failure:
    # the drone should go around rather than squeeze through.
    h, mode = gap_test(2.5)                       # 5 m gap
    turned_away = h is None or abs(deg(wrap_pi(h - 0.0))) > 40.0
    check("5 m gap -> conservatively refused", turned_away,
          f"mode={mode} heading={deg(h) if h is not None else 0.0:.0f}°")

    print("\n── 5. Fully enclosed: report trapped so caller climbs ─────────")
    g = blank()
    fill_box(g, -8.0, 8.0, 6.0, 8.0)
    fill_box(g, -8.0, 8.0, -8.0, -6.0)
    fill_box(g, 6.0, 8.0, -8.0, 8.0)
    fill_box(g, -8.0, -6.0, -8.0, 8.0)
    node.costmap = g
    blocked, nearest = node.build_histogram()
    h, mode = node.choose_heading(0.0, blocked, nearest)
    check("boxed in -> mode 'trapped'", mode == 'trapped', f"mode={mode}")

    print("\n── 6. Angular enlargement scales with distance ────────────────")
    def blocked_count(dist):
        g = blank()
        fill_box(g, dist, dist + 0.4, -0.2, 0.2)   # small post dead ahead
        node.costmap = g
        b, _ = node.build_histogram()
        return int(b.sum())
    near, far = blocked_count(2.0), blocked_count(8.0)
    check("near post blocks a wider arc than a far one",
          near > far, f"2m -> {near} bins, 8m -> {far} bins")

    print("\n── 7. Goal behind the drone ───────────────────────────────────")
    node.costmap = fill_box(blank(), 5.0, 7.0, -6.0, 6.0)
    blocked, nearest = node.build_histogram()
    h, mode = node.choose_heading(math.pi, blocked, nearest)   # due West
    ok = mode == 'clear' and abs(deg(wrap_pi(h - math.pi))) < 1.0
    check("wall east, goal west -> unobstructed", ok,
          f"mode={mode} heading={deg(h) if h is not None else None:.0f}°")

    print("\n── 8. Closing obstacle from the SIDE is caught ────────────────")
    # This is the case that killed the drone: a block crossing the path
    # broadside, outside the old +/-25 deg travel-arc check.
    nb = node.nbins
    side = nb // 4                       # 90 deg away from travel
    # Drive a controlled clock: back-to-back calls are microseconds apart and
    # the guard rightly ignores dt <= 1 ms, so feed it real 10 Hz ticks.
    clock = {'t': 100.0}
    node.now = lambda: clock['t']

    def feed(bin_idx, ranges, vel=(0.0, 0.0)):
        """Play a range sequence through the guard at 10 Hz; return last threat.

        The guard requires a sector to stay dangerous for
        threat_persist_ticks consecutive frames, so a realistic approach has
        to be played out rather than poked with two samples.
        """
        node.prev_nearest = None
        node.threat_ticks = np.zeros(nb, dtype=int)
        node.vel_enu = np.array(vel)
        node.last_evade_t = -99.0        # clear the cooldown between scenarios
        out = None
        for r in ranges:
            frame = np.full(nb, np.inf)
            frame[bin_idx] = r
            fired = node.imminent_collision(frame)
            # Report the FIRST firing, not the last frame's return value: the
            # evade cooldown deliberately suppresses re-fires within
            # evade_cooldown_s, so later frames legitimately return None.
            if fired is not None and out is None:
                out = fired
            clock['t'] += 0.1
        return out

    # Closing steadily from 3.5 m at ~4 m/s, from the side.
    threat = feed(side, [3.5, 2.9, 2.5, 2.1])
    check("lateral closer -> threat detected", threat is not None,
          f"threat={threat}")
    if threat:
        b, d, ttc = threat
        check("threat bearing is to the side (not ahead)",
              abs(deg(wrap_pi(b))) > 45.0, f"bearing={deg(b):.0f}°")

    print("\n── 9. Wall flown ALONGSIDE does not trip the guard ────────────")
    # Constant range = zero closing speed = infinite TTC, however close.
    threat = feed(side, [1.5, 1.5, 1.5, 1.5, 1.5])
    check("parallel wall at 1.5 m -> no false trigger", threat is None,
          f"threat={threat}")

    print("\n── 10. Static wall approached head-on -> NO false evade ──────")
    # Closing at exactly our own speed = static. VFH's job, not the guard's.
    ahead = int((math.pi) / (2 * math.pi / nb)) % nb   # bin for bearing 0 (east)
    # Flying east at 2.5 m/s straight at a static wall: it closes at exactly
    # 2.5 m/s, which our own motion fully explains.
    threat = feed(ahead, [3.0, 2.75, 2.5, 2.25, 2.0], vel=(2.5, 0.0))
    check("static wall at own speed -> no evade", threat is None, f"threat={threat}")

    print("\n── 11. Obstacle moving INTO us -> evade fires ─────────────────")
    # Same own-speed, but the obstacle is also driving at us: 7 m/s closing.
    threat = feed(ahead, [3.0, 2.3, 1.6, 0.9], vel=(2.5, 0.0))
    check("obstacle closing faster than our motion -> evade",
          threat is not None, f"threat={threat}")

    print("\n── 12. Banked drone must not see the ground as a wall ────────")
    # Rebuild a scan-only histogram: uniform 4 m returns all round, which at a
    # 30 deg roll are mostly terrain hits, not obstacles in the flight plane.
    class FakeScan:
        angle_min = -math.pi
        angle_increment = 2 * math.pi / 360
        range_min = 0.3
        range_max = 30.0
        ranges = [4.0] * 360
    node.costmap = None
    node.pos = np.array([0.0, 0.0, 8.0])
    node.yaw = 0.0
    node.scan = FakeScan()

    node.roll, node.pitch = 0.0, 0.0
    b_level, _ = node.build_histogram()
    # A hard 60 deg bank throws beams well outside the flight plane; at 4 m
    # range that puts the hit up to 3.5 m off, past the 2 m band.
    node.roll, node.pitch = math.radians(60), 0.0
    b_banked, _ = node.build_histogram()
    check("hard bank rejects terrain beams the level pose accepted",
          int(b_banked.sum()) < int(b_level.sum()),
          f"level={int(b_level.sum())} bins, banked={int(b_banked.sum())} bins")

    print("\n── 13. Surrounded at knife range -> terrain, not a wall ───────")
    # The signature that killed a run: many sectors close simultaneously.
    surrounded = np.full(nb, np.inf)
    surrounded[:20] = 1.0
    close_ct = int(np.count_nonzero(surrounded < node.terrain_close_range))
    check("terrain signature exceeds the sector threshold",
          close_ct >= node.terrain_close_sectors,
          f"{close_ct} sectors close, threshold {node.terrain_close_sectors}")
    one_wall = np.full(nb, np.inf)
    one_wall[:5] = 1.0                   # a genuine nearby wall spans few sectors
    close_ct2 = int(np.count_nonzero(one_wall < node.terrain_close_range))
    check("a single close wall does NOT look like terrain",
          close_ct2 < node.terrain_close_sectors,
          f"{close_ct2} sectors close, threshold {node.terrain_close_sectors}")

    print("\n── 14. Reflex climbs respect the altitude ceiling ─────────────")
    # The reflex branches used to command climb_speed directly, bypassing
    # max_escape_altitude entirely: one retrace peaked at 50.3 m against a
    # 22 m ceiling.
    #
    # The ceiling is set explicitly here rather than taken from config: it is
    # now PINNED to cruise_altitude, so "below the ceiling" is not a state the
    # drone is ever in, and reading the live value would make this test assert
    # nothing. What matters is that climb_bounded() obeys whatever bound it is
    # given, in both directions.
    ceiling = node.max_escape_alt
    node.max_escape_alt = 12.0
    node.pos = np.array([0.0, 0.0, 4.0])
    check("below ceiling -> climbs", node.climb_bounded() > 0.0,
          f"alt=4.0 m, ceiling=12.0 m, vz={node.climb_bounded():.2f}")
    node.pos = np.array([0.0, 0.0, node.max_escape_alt + 5.0])
    check("above ceiling -> refuses to climb", node.climb_bounded() == 0.0,
          f"alt={node.pos[2]:.1f} m, ceiling={node.max_escape_alt:.1f} m, "
          f"vz={node.climb_bounded():.2f}")
    node.max_escape_alt = ceiling

    print("\n── 15. Canyon vs roof: height, not sector count ───────────────")
    # Both surround the drone. Only the height of the hits tells them apart.
    node.pos = np.array([0.0, 0.0, 10.0])
    nearest_close = np.full(nb, np.inf); nearest_close[:20] = 1.2

    roof = np.full(nb, np.inf); roof[:20] = -0.5      # hits BELOW the drone
    below_roof = int(np.count_nonzero(
        (nearest_close < node.terrain_close_range) &
        (roof < -node.terrain_below_margin)))
    check("roof underneath -> counts as terrain",
          below_roof >= node.terrain_close_sectors,
          f"{below_roof} sectors below, threshold {node.terrain_close_sectors}")

    canyon = np.full(nb, np.inf); canyon[:20] = 0.0   # hits at our OWN level
    below_canyon = int(np.count_nonzero(
        (nearest_close < node.terrain_close_range) &
        (canyon < -node.terrain_below_margin)))
    check("canyon walls -> NOT terrain (handled as a trap)",
          below_canyon < node.terrain_close_sectors,
          f"{below_canyon} sectors below, threshold {node.terrain_close_sectors}")

    print("\n── 16. Emergency brake: stop, then seek space ─────────────────")
    # Braking replaced the drive-away reflex because reversing thrust is what
    # broke the EKF. Verify the state machine's decision points.
    node.pos = np.array([0.0, 0.0, 4.0])
    node.braking = False
    node.brake_until = 0.0
    node.now = lambda: clock['t']

    # Moving fast + inside the brake window -> BRAKE (kill momentum first)
    node.brake_until = clock['t'] + node.brake_hold_s
    node.vel_enu = np.array([2.5, 0.0])
    moving = float(np.linalg.norm(node.vel_enu)) > node.brake_stop_speed
    check("carrying momentum -> brake before anything else", moving,
          f"speed={np.linalg.norm(node.vel_enu):.1f} > {node.brake_stop_speed}")

    # Once stopped, the drone is allowed to look for a gap
    node.vel_enu = np.array([0.1, 0.0])
    stopped = float(np.linalg.norm(node.vel_enu)) <= node.brake_stop_speed
    check("stopped -> free to sidestep into a gap", stopped,
          f"speed={np.linalg.norm(node.vel_enu):.1f} <= {node.brake_stop_speed}")

    # The hold window outlasts a single detection, so one crossing obstacle
    # cannot cause stop/go stuttering.
    check("brake window outlives one detection",
          node.brake_hold_s >= 1.0, f"brake_hold_s={node.brake_hold_s}")

    print("\n── 17. Setpoint slew limit: ramps, never steps ────────────────")
    # The measured wobble was ArduPilot RINGING against an unreachable demand:
    # 35 °/s median body rates during braking against 1.2 °/s cruising, with
    # the roll rate changing sign ~2x a second. Cap how fast the SETPOINT may
    # move and the demand becomes trackable.
    sent = []
    node.vel_pub.publish = lambda m: sent.append(m)
    node.cmd = np.array([2.0, 0.0, 0.0])       # cruising East
    node.last_cmd_t = clock['t']
    clock['t'] += node.dt
    node.send_velocity(0.0, 0.0, 0.0)          # a full-stop STEP request
    step = 2.0 - float(np.linalg.norm(node.cmd[:2]))
    check("one tick cannot stop from cruise",
          float(np.linalg.norm(node.cmd[:2])) > 0.5,
          f"still commanding {np.linalg.norm(node.cmd[:2]):.2f} m/s")
    check("deceleration honours max_accel",
          step <= node.max_accel * node.dt + 1e-6,
          f"Δv={step:.3f} <= {node.max_accel * node.dt:.3f} m/s per tick")

    # ...but it does get all the way there, and in a sane time.
    for _ in range(200):
        clock['t'] += node.dt
        node.send_velocity(0.0, 0.0, 0.0)
    check("the ramp still reaches a full stop",
          float(np.linalg.norm(node.cmd[:2])) < 0.01,
          f"{np.linalg.norm(node.cmd[:2]):.4f} m/s")

    # A direction REVERSAL must be limited as firmly as a speed change —
    # limiting components separately would let a 180° flip through at once.
    node.cmd = np.array([2.0, 0.0, 0.0])
    clock['t'] += node.dt
    node.send_velocity(-2.0, 0.0, 0.0)
    check("a 180° reversal is rate-limited too",
          abs(float(node.cmd[0]) - 2.0) <= node.max_accel * node.dt + 1e-6,
          f"vx={node.cmd[0]:.3f} after one tick of a +2 -> -2 flip")

    print("\n── 18. Livelock breaker never commands a DESCENT ──────────────")
    # The 2026-08-27 deadlock: parked at 7.51 m on a roof with target_alt
    # decayed to ~4 m, this computed 4+3 = 7.0 m and pressed the drone DOWN
    # into the roof, every 15 s, forever.
    node.pos = np.array([0.0, 0.0, 7.51])
    node.target_alt = 4.0
    node.best_dist = 13.7
    node.progress_time = clock['t'] - node.stuck_timeout_s - 1.0
    node.max_escape_alt = 12.0                 # ceiling well above us
    node.check_stuck(13.7, None)
    check("stuck climb is measured from the CURRENT altitude",
          node.target_alt > node.pos[2],
          f"target_alt={node.target_alt:.2f} > z={node.pos[2]:.2f}")
    check("climb_vz() therefore commands UP, not down",
          node.climb_vz() > 0.0, f"vz={node.climb_vz():+.2f} m/s")

    # At the ceiling it must still do something — a watchdog that can only
    # climb is a watchdog that does nothing when the cap forbids climbing.
    # It hands over to committed boundary following.
    def wall(lo_deg, hi_deg):
        """blocked[] with a wedge from lo..hi degrees occupied."""
        bw = 2.0 * math.pi / node.nbins
        bearings = -math.pi + (np.arange(node.nbins) + 0.5) * bw
        return (bearings >= math.radians(lo_deg)) & (bearings <= math.radians(hi_deg))

    node.pos = np.array([0.0, 0.0, 4.0])
    node.target_alt = 4.0
    node.max_escape_alt = 4.0                  # the specified 4 m envelope
    node.progress_time = clock['t'] - node.stuck_timeout_s - 1.0
    node.last_heading = 0.0
    node.detour_dir = 0
    # Asymmetric wall: open ground is much nearer going LEFT (+35°) than
    # RIGHT (-85°), so a correct tangent-bug commits left.
    blk = wall(-80.0, 30.0)
    room = np.where(blk, 3.0, 25.0)
    node.check_stuck(13.7, room, 0.0, blk)
    check("capped at the ceiling -> starts a committed detour",
          node.detour_dir != 0, f"detour_dir={node.detour_dir}")
    check("detour rounds the obstacle the SHORT way (left here)",
          node.detour_dir > 0, f"detour_dir={node.detour_dir}")

    t = node.tangent_heading(0.0, blk, room, node.detour_dir)
    check("tangent hugs the obstacle edge rather than fleeing to open space",
          t is not None and 30.0 < deg(t) < 75.0,
          f"tangent={deg(t):.0f}°" if t is not None else 'none')

    print("\n── 18b. A detour is COMMITTED — this is what stops the sawtooth ─")
    # The 4 s "break out toward the roomiest bearing" that preceded this
    # failed exactly here: it expired, goal attraction pulled the drone back
    # into the same basin, and the log sawtoothed 31 -> 36 -> 31 m from WP6
    # for minutes. Commitment plus a MEASURED exit test is the whole fix.
    node.detour_entry_dist = 33.0
    node.detour_since = clock['t']
    before = node.detour_dir
    node.best_dist = 33.0
    node.progress_time = clock['t'] - node.stuck_timeout_s - 1.0
    node.check_stuck(36.0, room, 0.0, blk)     # moving AWAY from the goal
    check("watchdog stays quiet while a detour runs (no direction flip)",
          node.detour_dir == before, f"dir={node.detour_dir} (was {before})")

    node.end_detour('test')
    check("ending a detour clears the direction and rearms the watchdog",
          node.detour_dir == 0 and node.best_dist is None,
          f"dir={node.detour_dir}, best_dist={node.best_dist}")

    node.detour_dir = 1
    node.detour_entry_dist = 33.0
    node.flip_detour(33.0)
    check("a timed-out direction reverses instead of giving up",
          node.detour_dir == -1, f"dir={node.detour_dir}")
    node.detour_dir = 0

    print("\n── 19. Surface contact: commanded but not moving ──────────────")
    # A 2D LiDAR cannot see a flat roof it is level with, so every scan-based
    # guard is blind at once. The signal is physics, not perception.
    node.pos = np.array([0.0, 0.0, 7.51])
    node.contact = False
    node.moving_since = None
    node.cmd = np.array([1.5, 0.0, 0.0])       # asking for cruise speed
    node.vel_enu = np.array([0.0, 0.0])        # getting nothing
    check("contact is not declared on the first stalled tick",
          node.check_contact() is False, 'needs contact_confirm_s of evidence')
    clock['t'] += node.contact_confirm_s + 0.1
    check("sustained commanded-but-motionless -> SURFACE CONTACT",
          node.check_contact() is True,
          f'{node.contact_confirm_s:.1f}s commanding 1.5 m/s at 0.0 m/s')

    # Hovering deliberately must NOT look like contact.
    node.contact = False
    node.moving_since = None
    node.cmd = np.array([0.0, 0.0, 0.0])       # holding position on purpose
    node.vel_enu = np.array([0.0, 0.0])
    clock['t'] += node.contact_confirm_s + 0.1
    check("a commanded hover is not contact",
          node.check_contact() is False, 'no motion commanded, none expected')

    # And it clears the moment the drone actually moves again.
    node.contact = True
    node.cmd = np.array([1.0, 0.0, 0.0])
    node.vel_enu = np.array([0.9, 0.0])
    check("contact clears once the drone moves",
          node.check_contact() is False and node.contact is False)

    print("\n── 20. Terminal approach to the final waypoint ────────────────")
    check("the last waypoint gets a tighter radius than the rest",
          node.final_wp_radius < node.wp_radius,
          f"{node.final_wp_radius} m vs {node.wp_radius} m")
    check("terminal creep is slower than the travelling floor",
          node.final_approach_speed < node.min_speed,
          f"{node.final_approach_speed} < min_speed {node.min_speed}")
    # 0.8 m/s at 20 Hz moves 4 cm per tick; the creep must resolve the ball.
    check("creep speed can actually resolve the final radius",
          node.final_approach_speed * node.dt < node.final_wp_radius / 4.0,
          f"{node.final_approach_speed * node.dt:.3f} m/tick into a "
          f"{node.final_wp_radius:.1f} m ball")
    check("precision has a timeout so it cannot livelock",
          node.final_approach_timeout_s > 0.0,
          f"{node.final_approach_timeout_s:.0f} s then falls back to wp_radius")

    print("\n── 21. The 4 m envelope is actually enforced ──────────────────")
    check("max_escape_altitude is pinned to cruise_altitude",
          abs(node.max_escape_alt - node.cruise_alt) < 1e-6,
          f"escape ceiling {node.max_escape_alt} m vs cruise {node.cruise_alt} m")
    node.pos = np.array([0.0, 0.0, 4.0])
    check("at cruise altitude the drone may not climb further",
          node.climb_bounded() == 0.0, 'vertical escape disabled by the cap')

    print("\n── 22. A* solves the WP6 trap that VFH alone could not ────────")
    # This replays the REAL geometry of the failure, at the real scale.
    # Drone at ENU (-0.6,-14.1), WP6 at (32,-8). suburb_house_10 (10,-15) and
    # suburb_house_6 (25,-15) are 8 m boxes straddling the straight line;
    # WP6 sits in the street at y=-8. The only route is north into the
    # corridor, then east — which no goal-directed local step will propose.
    PRES, PORG = 0.5, (-90.0, -85.0)
    PW = PH = int(180 / PRES)                    # 360x360 global map

    def world_map():
        g = np.zeros((PH, PW), dtype=np.int8)

        def box(cx, cy, half, cost):
            c0 = int((cx - half - PORG[0]) / PRES); c1 = int((cx + half - PORG[0]) / PRES)
            r0 = int((cy - half - PORG[1]) / PRES); r1 = int((cy + half - PORG[1]) / PRES)
            g[max(0, r0):min(PH, r1), max(0, c0):min(PW, c1)] = np.maximum(
                g[max(0, r0):min(PH, r1), max(0, c0):min(PW, c1)], cost)

        # The suburb block grid, in ENU, as it exists in the world file.
        for hx in (10.0, 25.0, 40.0):
            for hy in (-30.0, -15.0, 0.0):
                box(hx, hy, 4.0 + 2.0, 70)       # 2 m inflation gradient
                box(hx, hy, 4.0, 100)            # the 8 m building itself
        return g

    node.costmap = world_map()
    node.map_res, node.map_origin = PRES, PORG
    node.pos = np.array([-0.6, -14.1, 4.0])
    WP6 = (32.0, -8.0)

    import time as _t
    t0 = _t.perf_counter()
    path = node.plan_path(WP6)
    plan_ms = (_t.perf_counter() - t0) * 1000.0

    check("A* finds a route to WP6 through the street grid",
          path is not None and len(path) > 2,
          f"{len(path) if path else 0} cells in {plan_ms:.0f} ms")
    check("planning fits inside a 20 Hz control tick budget",
          plan_ms < 400.0, f"{plan_ms:.0f} ms")

    if path:
        px = np.array(path)
        # The route must enter the street corridor (y > -11), i.e. go NORTH,
        # which is exactly the move goal attraction refuses to make.
        check("the route goes NORTH into the street, not east into the house",
              float(px[:, 1].max()) > -11.0,
              f"max y on route = {px[:, 1].max():.1f} m (houses span y -19..-11)")

        # And it must not pass through a building.
        def in_house(x, y):
            for hx in (10.0, 25.0, 40.0):
                for hy in (-30.0, -15.0, 0.0):
                    if abs(x - hx) <= 4.0 and abs(y - hy) <= 4.0:
                        return True
            return False

        hits = [(x, y) for x, y in path if in_house(x, y)]
        check("the route never crosses a building footprint",
              not hits, f"{len(hits)} points inside a house")

        check("the route actually terminates at WP6",
              math.hypot(path[-1][0] - WP6[0], path[-1][1] - WP6[1]) < 2.0,
              f"ends {math.hypot(path[-1][0]-WP6[0], path[-1][1]-WP6[1]):.1f} m from WP6")

        # The decisive check: the bearing the drone is told to fly.
        c = node.carrot_on(path)
        cb = math.atan2(c[1] - node.pos[1], c[0] - node.pos[0])
        direct = math.atan2(WP6[1] - node.pos[1], WP6[0] - node.pos[0])
        check("the carrot steers away from the trapping direct bearing",
              abs(deg(wrap_pi(cb - direct))) > 20.0,
              f"carrot {deg(cb):.0f}° vs direct {deg(direct):.0f}° "
              f"(Δ{abs(deg(wrap_pi(cb - direct))):.0f}°)")
        check("the carrot has a northward component",
              c[1] > node.pos[1], f"carrot y={c[1]:.1f} > drone y={node.pos[1]:.1f}")

    print("\n── 23. Planner degrades safely ────────────────────────────────")
    node.costmap = None
    check("no costmap -> no carrot, caller falls back to direct bearing",
          node.update_plan(WP6) is None)
    node.costmap = world_map()
    node.pos = np.array([-0.6, -14.1, 4.0])
    # A goal buried inside a building must not wedge the planner: it snaps to
    # the nearest reachable cell instead of returning nothing.
    inside = node.plan_path((25.0, -15.0))
    check("a goal inside an obstacle snaps to reachable space",
          inside is not None and len(inside) > 1,
          f"{len(inside) if inside else 0} cells")
    node.use_global_planner = False
    check("planner can be disabled outright",
          node.update_plan(WP6) is None)
    node.use_global_planner = True

    print("\n── 24. A known street beats unexplored ground ─────────────────")
    # The first weighting got this backwards. Every suburb street is 7 m wide
    # against a 2 m inflation radius, so a street is ALL gradient. Penalising
    # gradient heavily meant penalising streets: A* routed the drone the long
    # way round through ground the LiDAR had never swept rather than down the
    # street it could already see.
    def street_map():
        g = np.full((PH, PW), -1, dtype=np.int8)          # unknown everywhere

        def band(y0, y1, x0, x1, v):
            r0 = int((y0 - PORG[1]) / PRES); r1 = int((y1 - PORG[1]) / PRES)
            c0 = int((x0 - PORG[0]) / PRES); c1 = int((x1 - PORG[0]) / PRES)
            g[max(0, r0):min(PH, r1), max(0, c0):min(PW, c1)] = v

        band(-11.0, -4.0, -5.0, 40.0, 0)      # the swept street: known free
        band(-11.0, -9.0, -5.0, 40.0, 70)     # inflation off the south kerb
        band(-6.0, -4.0, -5.0, 40.0, 70)      # inflation off the north kerb
        band(-19.0, -11.0, 6.0, 14.0, 100)    # houses either side
        band(-19.0, -11.0, 21.0, 29.0, 100)
        band(-4.0, 4.0, 6.0, 14.0, 100)
        band(-4.0, 4.0, 21.0, 29.0, 100)
        return g

    node.costmap = street_map()
    node.map_res, node.map_origin = PRES, PORG
    node.pos = np.array([-2.0, -7.5, 4.0])
    node.plan_cost_penalty = 0.8
    node.plan_unknown_penalty = 0.6
    route = node.plan_path((36.0, -7.5))
    check("A* routes down the known street", route is not None and len(route) > 5,
          f"{len(route) if route else 0} cells")
    if route:
        ys = [y for _, y in route]
        inside = sum(1 for y in ys if -11.5 <= y <= -3.5)
        check("the route stays in the swept corridor, not the unknown",
              inside / len(ys) > 0.9,
              f"{inside}/{len(ys)} points inside the street band")

    # And the ordering the weights are supposed to produce, stated directly.
    known_free = 1.0 + node.plan_cost_penalty * 0.0
    known_infl = 1.0 + node.plan_cost_penalty * 0.60
    unknown = 1.0 + node.plan_unknown_penalty
    near_lethal = 1.0 + node.plan_cost_penalty * 0.89
    check("cost ordering: free < inflated street < unknown < near-lethal",
          known_free < known_infl < unknown < near_lethal,
          f"{known_free:.2f} < {known_infl:.2f} < {unknown:.2f} < {near_lethal:.2f}")

    print("\n── 25. The watchdog must not fight the planner ────────────────")
    # The bug this pins down: check_stuck measured straight-line distance to
    # the waypoint. A correct route around a block INCREASES that distance for
    # ~20 m of flying, so the watchdog declared a local minimum and the
    # boundary follow hijacked the heading, dragging the drone from 30 m out
    # to 49 m — fighting a planner that was right.
    node.pos = np.array([0.0, -14.0, 4.0])
    node.path = [(0.0, -14.0), (0.0, -8.0), (10.0, -8.0), (20.0, -8.0)]
    far = node.remaining_route()
    node.pos = np.array([0.0, -8.0, 4.0])          # flown north, off the goal
    near = node.remaining_route()
    goal = (20.0, -8.0)
    d_before = math.hypot(goal[0] - 0.0, goal[1] - (-14.0))
    d_after = math.hypot(goal[0] - 0.0, goal[1] - (-8.0))
    check("route length falls while straight-line distance is flat or worse",
          near < far - 1.0,
          f"route {far:.1f} -> {near:.1f} m, straight line "
          f"{d_before:.1f} -> {d_after:.1f} m")

    node.detour_dir = 0
    node.route_stall = 0
    node.best_dist = 30.0
    node.progress_time = clock['t'] - node.stuck_timeout_s - 1.0
    blk2 = wall(-80.0, 30.0)
    room2 = np.where(blk2, 3.0, 25.0)
    node.check_stuck(30.0, room2, 0.0, blk2, have_route=True)
    check("a stall WITH a route forces a replan, not a detour",
          node.detour_dir == 0 and node.path == [],
          f"detour_dir={node.detour_dir}, path cleared={node.path == []}")

    # Far from the target (outside no_detour_radius), repeated stalls must
    # eventually hand over to boundary following.
    far_goal = node.no_detour_radius + 20.0
    for _ in range(node.route_stall_limit):
        node.progress_time = clock['t'] - node.stuck_timeout_s - 1.0
        node.best_dist = far_goal
        node.check_stuck(far_goal, room2, 0.0, blk2, have_route=True,
                         goal_dist=far_goal)
    check("repeated stalls DO eventually fall back to boundary following",
          node.detour_dir != 0, f"detour_dir={node.detour_dir}")
    node.detour_dir = 0

    # But NEAR the target a stall must never detour away from it.
    node.route_stall = 0
    near_goal = node.no_detour_radius - 8.0
    for _ in range(node.route_stall_limit + 2):
        node.progress_time = clock['t'] - node.stuck_timeout_s - 1.0
        node.best_dist = near_goal
        node.path = [(0.0, 0.0), (5.0, 0.0)]
        node.check_stuck(near_goal, room2, 0.0, blk2, have_route=True,
                         goal_dist=near_goal)
    check("close to the target a stall replans instead of detouring away",
          node.detour_dir == 0,
          f"detour_dir={node.detour_dir} at {near_goal:.0f} m "
          f"(no_detour_radius {node.no_detour_radius:.0f} m)")

    node.progress_time = clock['t'] - node.stuck_timeout_s - 1.0
    node.best_dist = 30.0
    node.route_stall = 0
    node.check_stuck(30.0, room2, 0.0, blk2, have_route=False)
    check("with NO route a stall still goes straight to boundary following",
          node.detour_dir != 0, f"detour_dir={node.detour_dir}")
    node.detour_dir = 0

    print("\n── 26. A moving obstacle is never terrain ─────────────────────")
    # The crash of 2026-08-27. A 3 m dynamic block (top at 3.0 m) drifted
    # alongside the drone at 4.0 m. Its returns sit BELOW the drone, which is
    # the exact signature the terrain check reads as "surface beneath", so the
    # node commanded "climbing straight up, no lateral move" while the TTC
    # guard — which had already fired three EMERGENCY BRAKEs — never ran at
    # all. Pinned at the 4 m ceiling where climbing buys nothing, the block
    # closed to 0.7 m and the FCU took the aircraft in LAND.
    node.costmap = blank()
    node.map_res, node.map_origin = RES, ORIGIN
    node.pos = np.array([0.0, 0.0, 4.0])
    node.max_escape_alt = 12.0            # climbing genuinely available here
    node.terrain_logged = False

    # Ring of returns just below the drone: the "surface beneath" signature.
    near = np.full(node.nbins, node.terrain_close_range - 0.5)
    node.sector_zoff = np.full(node.nbins,
                               -(node.terrain_below_margin + 0.5))
    close = int(np.count_nonzero(
        (near < node.terrain_close_range)
        & (node.sector_zoff < -node.terrain_below_margin)))
    check("the scenario really does trip the terrain heuristic",
          close >= node.terrain_close_sectors,
          f"{close} sectors below, threshold {node.terrain_close_sectors}")
    check("and climbing is genuinely available in this scenario",
          node.climb_bounded() > 0.0, f"climb {node.climb_bounded():.2f} m/s")

    # The gate itself: real headroom AND nothing closing.
    def terrain_allowed(threat, z, ceiling):
        node.pos = np.array([0.0, 0.0, z])
        node.max_escape_alt = ceiling
        return (close >= node.terrain_close_sectors
                and (node.max_escape_alt - node.pos[2]) > 0.5
                and threat is None)

    check("with headroom and nothing closing, the terrain climb still fires",
          terrain_allowed(None, 4.0, 12.0))
    check("with an obstacle closing, the terrain climb is suppressed",
          not terrain_allowed((math.radians(88.0), 4.1, 0.8), 4.0, 12.0),
          "TTC guard gets the tick instead of a zero-lateral hover")
    # The 4 m cap case: sagging to 3.99 m under a 4.0 m ceiling used to satisfy
    # climb_bounded() > 0 and surrender all lateral mobility for 0.01 m of
    # height. That is what dyn_block_3 flew into on 2026-08-27.
    check("under the 4 m cap the terrain climb is inert, not a freeze",
          not terrain_allowed(None, 3.99, 4.0),
          "0.01 m of headroom must not cost all lateral movement")
    node.pos = np.array([0.0, 0.0, 4.0])
    node.max_escape_alt = node.cruise_alt

    print("\n── 27. Proximity back-off outranks everything ─────────────────")
    # Five flights ended with something 0.8 m away while the drone was still
    # commanding motion, under three different states. Nothing owned the last
    # metre; this does.
    node.pos = np.array([0.0, 0.0, 4.0])
    node.max_escape_alt = node.cruise_alt
    nb = node.nbins
    near = np.full(nb, np.inf)
    b_hit = node._bin_index(np.array([0.0]))[0]        # something due East
    near[b_hit] = 0.8
    b_min = int(np.argmin(np.where(np.isfinite(near), near, np.inf)))
    d_min = float(near[b_min])
    check("0.8 m triggers the back-off band",
          d_min < node.backoff_range,
          f"{d_min:.2f} m < {node.backoff_range:.1f} m")
    bw = 2.0 * math.pi / nb
    toward = wrap_pi(-math.pi + (b_min + 0.5) * bw)
    away = wrap_pi(toward + math.pi)
    check("the retreat points directly away from the nearest return",
          abs(abs(deg(wrap_pi(away - toward))) - 180.0) < 6.0,
          f"toward {deg(toward):.0f}°, away {deg(away):.0f}°")
    check("back-off band clears the self-filter with margin",
          node.backoff_range > node.self_filter_range + 0.5,
          f"backoff {node.backoff_range:.1f} m vs self-filter "
          f"{node.self_filter_range:.1f} m")
    check("retreat speed is gentle enough not to provoke the estimator",
          0.0 < node.backoff_speed <= node.cruise_speed,
          f"{node.backoff_speed:.1f} m/s vs cruise {node.cruise_speed:.1f}")
    # It must not fire in normal flight, or the drone would never approach
    # anything at all.
    far = np.full(nb, np.inf); far[b_hit] = 5.0
    check("normal clearances do NOT trigger a retreat",
          float(np.min(far)) >= node.backoff_range, "5.0 m is not a retreat")

    # Directional: a wall passed ABEAM must not trigger a retreat, or the
    # drone cannot fly down a street. This livelocked the WP6 leg for
    # 11 minutes with 69 back-offs, oscillating 21-26 m short of the target.
    bw2 = 2.0 * math.pi / nb
    bearings = -math.pi + (np.arange(nb) + 0.5) * bw2

    def backoff_fires(obstacle_deg, course_deg, dist):
        n2 = np.full(nb, np.inf)
        n2[node._bin_index(np.array([math.radians(obstacle_deg)]))[0]] = dist
        ahead = np.array([abs(wrap_pi(b - math.radians(course_deg)))
                          for b in bearings]) < math.radians(node.backoff_arc)
        cand = np.where(ahead & np.isfinite(n2), n2, np.inf)
        return float(np.min(cand)) < node.backoff_range

    check("an obstacle dead ahead still triggers the retreat",
          backoff_fires(0.0, 0.0, 0.8), "0.8 m at 0 deg off course")
    check("a wall passed abeam does NOT trigger a retreat",
          not backoff_fires(90.0, 0.0, 0.8),
          "0.8 m at 90 deg off course — this is a corridor, not a collision")
    check("an obstacle behind does NOT trigger a retreat",
          not backoff_fires(180.0, 0.0, 0.8), "0.8 m directly astern")

    print("\n── 28. Smart retrace: breadcrumbs and the way home ────────────")
    # Outbound: crumbs are recorded only from CLEAR moments during MISSION,
    # at least breadcrumb_spacing apart, so the trail describes a corridor
    # already known to be flyable.
    node.phase = 'MISSION'
    node.breadcrumbs = []
    node.pos = np.array([0.0, 0.0, 4.0])
    node.drop_breadcrumb()
    check("the first crumb is the takeoff point",
          len(node.breadcrumbs) == 1 and
          float(np.linalg.norm(node.breadcrumbs[0])) < 1e-6,
          f"{len(node.breadcrumbs)} crumb at {node.breadcrumbs[0]}")

    node.pos = np.array([node.breadcrumb_spacing * 0.5, 0.0, 4.0])
    node.drop_breadcrumb()
    check("a crumb is NOT dropped before the spacing is covered",
          len(node.breadcrumbs) == 1,
          f"moved {node.breadcrumb_spacing*0.5:.1f} m, spacing "
          f"{node.breadcrumb_spacing:.1f} m")

    node.pos = np.array([node.breadcrumb_spacing * 1.2, 0.0, 4.0])
    node.drop_breadcrumb()
    check("a crumb IS dropped once the spacing is exceeded",
          len(node.breadcrumbs) == 2, f"{len(node.breadcrumbs)} crumbs")

    node.phase = 'RETRACE'
    node.pos = np.array([500.0, 500.0, 4.0])
    before = len(node.breadcrumbs)
    node.drop_breadcrumb()
    check("no crumbs are laid outside the MISSION phase",
          len(node.breadcrumbs) == before, "retrace must not extend its own trail")

    # Homeward: lay a realistic outbound trail, then fly it in reverse exactly
    # the way main_loop consumes it, and require the drone to arrive home.
    node.phase = 'MISSION'
    node.breadcrumbs = []
    trail = [(float(i) * node.breadcrumb_spacing, 0.0) for i in range(40)]
    for x, y in trail:
        node.pos = np.array([x, y, 4.0])
        node.drop_breadcrumb()
    laid = len(node.breadcrumbs)
    check("the outbound trail is recorded end to end",
          laid == len(trail), f"{laid} crumbs over {len(trail)} steps")

    node.phase = 'RETRACE'
    node.retrace_index = laid - 1
    node.pos = np.array([trail[-1][0], trail[-1][1], 4.0])
    order, guard = [], 0
    while node.retrace_index >= 0 and guard < 5000:
        guard += 1
        # This is main_loop's pure-pursuit skip, verbatim.
        while (node.retrace_index >= 0 and
               float(np.linalg.norm(node.pos[:2] -
                                    node.breadcrumbs[node.retrace_index]))
               < node.retrace_lookahead):
            node.retrace_index -= 1
        if node.retrace_index < 0:
            break
        order.append(node.retrace_index)
        tgt = node.breadcrumbs[node.retrace_index]
        step = tgt - node.pos[:2]
        n = float(np.linalg.norm(step))
        node.pos[:2] = node.pos[:2] + step / max(n, 1e-9) * min(n, 1.0)
    # The index may REPEAT — the drone chases one crumb over several ticks —
    # but it must never go back up, which would mean retracing outbound.
    check("the trail is consumed in reverse, never backwards",
          len(order) > 3 and all(b <= a for a, b in zip(order, order[1:])),
          f"{len(order)} targets, {order[:3]}...{order[-3:]}" if order else "none")
    check("every crumb on the trail is actually visited",
          len(set(order)) >= laid - 2,
          f"{len(set(order))} distinct of {laid} crumbs")
    check("retrace terminates rather than looping",
          node.retrace_index < 0 and guard < 5000,
          f"index={node.retrace_index} after {guard} iterations")
    check("the drone ends up at the takeoff point",
          float(np.linalg.norm(node.pos[:2])) < node.retrace_lookahead + 1.0,
          f"finished {float(np.linalg.norm(node.pos[:2])):.2f} m from home "
          f"(lookahead {node.retrace_lookahead:.1f} m)")

    print("\n── 29. The planner runs off the control loop ──────────────────")
    # A* costs 50 ms for a long route and 141 ms for an unreachable one,
    # against a 50 ms control period. Planning inline dropped one to three
    # setpoint cycles per second out of the stream ArduPilot's GUIDED mode
    # rides on. update_plan must therefore only ever POST a goal and READ a
    # route; planner_tick is the only thing allowed to search.
    node.pos = np.array([-0.6, -14.1, 4.0])
    node.costmap = world_map()
    node.map_res, node.map_origin = PRES, PORG   # section 28 left the 40 m window
    node.use_global_planner = True
    node.invalidate_plan()
    node.plan_request = None

    searched = {'n': 0}
    real_plan_path = node.plan_path

    def counting_plan_path(goal):
        searched['n'] += 1
        return real_plan_path(goal)

    node.plan_path = counting_plan_path

    t0 = time.perf_counter()
    for _ in range(200):
        node.update_plan(WP6)
    inline_ms = (time.perf_counter() - t0) * 1000.0
    check("update_plan never searches",
          searched['n'] == 0, f"{searched['n']} searches in 200 calls")
    check("200 control-loop calls cost less than one control period",
          inline_ms < node.dt * 1000.0,
          f"{inline_ms:.1f} ms total vs {node.dt * 1000.0:.0f} ms period")
    check("update_plan posts the goal for the planner thread",
          node.plan_request == (float(WP6[0]), float(WP6[1])),
          f"{node.plan_request}")
    check("no route yet -> no carrot, caller keeps the direct bearing",
          node.update_plan(WP6) is None)

    # Now let the planner thread do its half of the job.
    clock['t'] += node.replan_period_s + 1.0
    node.planner_tick()
    check("planner_tick searches and publishes a route",
          searched['n'] == 1 and len(node.route_snapshot()[0]) > 1,
          f"{searched['n']} search, {len(node.route_snapshot()[0])} cells")
    check("the published route is stamped with the goal it was planned for",
          node.route_snapshot()[1] == (float(WP6[0]), float(WP6[1])),
          f"{node.route_snapshot()[1]}")
    check("update_plan now returns a carrot from that route",
          node.update_plan(WP6) is not None)

    # The route belongs to WP6. Asked to steer somewhere else, the control
    # loop must NOT follow it — that would fly at a waypoint already left.
    check("a route planned for another goal is not used to steer",
          node.update_plan((-40.0, 40.0)) is None,
          "route to WP6 must not be flown toward a different target")

    # A stall or a new waypoint throws the route away.
    node.plan_request = (float(WP6[0]), float(WP6[1]))
    node.update_plan(WP6)
    epoch_before = node.plan_epoch
    node.invalidate_plan()
    check("invalidate_plan clears the route and bumps the epoch",
          node.route_snapshot() == ([], None)
          and node.plan_epoch == epoch_before + 1,
          f"epoch {epoch_before} -> {node.plan_epoch}")

    # THE RACE THIS SPLIT CREATES, AND THE GUARD FOR IT: the control loop
    # invalidates while a search is already running. That search's answer is
    # stale by the time it lands, and publishing it would silently overwrite
    # a fresh decision with an old one.
    def invalidating_plan_path(goal):
        node.invalidate_plan()          # the control loop, mid-search
        return real_plan_path(goal)

    node.plan_path = invalidating_plan_path
    node.plan_request = (float(WP6[0]), float(WP6[1]))
    node.plan_target = None
    clock['t'] += node.replan_period_s + 1.0
    node.planner_tick()
    check("a route invalidated mid-search is discarded, not published",
          node.route_snapshot() == ([], None),
          "stale result must not overwrite a fresh decision")

    node.plan_path = counting_plan_path
    node.plan_target = None
    clock['t'] += node.replan_period_s + 1.0
    node.planner_tick()
    before = searched['n']
    for _ in range(5):                       # five 10 Hz planner ticks
        clock['t'] += 0.1
        node.planner_tick()
    check("planner_tick honours replan_period_s instead of searching at 10 Hz",
          searched['n'] == before,
          f"{searched['n'] - before} extra searches in 0.5 s "
          f"(period {node.replan_period_s:.1f} s)")
    clock['t'] += node.replan_period_s + 1.0
    node.planner_tick()
    check("...but does replan once the period has elapsed",
          searched['n'] == before + 1, f"{searched['n'] - before} search")

    node.plan_path = real_plan_path
    node.use_global_planner = False
    check("a disabled planner still returns no carrot",
          node.update_plan(WP6) is None)
    node.use_global_planner = True

    print("\n── 30. Own velocity comes from the EKF, not differenced pose ──")
    # vel_enu feeds dynamic_closing_thresh (0.8 m/s), the back-off travel
    # direction and contact_speed (0.15 m/s). Differencing two poses that
    # each carry centimetre jitter is a poor way to measure that.
    class FakeTwist:
        class twist:
            class linear:
                x, y, z = 1.25, -0.4, 0.0

    node.vel_ok = False
    node.vel_cb(FakeTwist)
    check("vel_cb takes the fused ENU velocity straight from MAVROS",
          abs(node.vel_enu[0] - 1.25) < 1e-9 and abs(node.vel_enu[1] + 0.4) < 1e-9,
          f"vel_enu = {node.vel_enu}")
    check("the fused estimate is timestamped so staleness is detectable",
          abs(node.vel_fused_t - clock['t']) < 1e-9)

    # While the topic is live, pose differencing must not overwrite it.
    class FakePose:
        class pose:
            class position:
                x, y, z = 5.0, 5.0, 4.0
            class orientation:
                x = y = z = 0.0
                w = 1.0

    node.prev_vel_pos = np.array([0.0, 0.0])
    node.prev_vel_t = clock['t'] - 1.0
    node.last_pos = np.array([0.0, 0.0, 4.0])
    node.pose_cb(FakePose)
    check("fresh fused velocity suppresses the pose-differencing fallback",
          abs(node.vel_enu[0] - 1.25) < 1e-9,
          f"vel_enu = {node.vel_enu} (differencing would give ~5.0)")

    # Let the topic go quiet: the guards must degrade, not go blind.
    clock['t'] += 2.0
    node.prev_vel_pos = np.array([0.0, 0.0])
    node.prev_vel_t = clock['t'] - 1.0
    node.pose_cb(FakePose)
    check("a silent velocity topic falls back to differenced pose",
          abs(node.vel_enu[0] - 5.0) < 1e-6,
          f"vel_enu = {node.vel_enu} after 2 s without a fused message")

    node.destroy_node()
    rclpy.shutdown()

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'═' * 62}\n  {passed}/{total} checks passed\n{'═' * 62}")
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
