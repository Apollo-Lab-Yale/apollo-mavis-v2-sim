"""ResetPlanner tests (03-sim §10/§14.2): swap, detour, failures, arm_order, and the
pinched-start escape phase + gate-resolution verification (2026-09-09 incident)."""

from __future__ import annotations

import time

import mujoco
import numpy as np
import pytest
from apollo_mavis_v2_core import PlanRequest

from apollo_mavis_v2_sim import REGISTRY, DigitalTwin, PlannerParams, ResetPlanner
from apollo_mavis_v2_sim.planner import (
    _LEVER_ARM_M,
    ESCAPE_RATE_MARGIN,
    ESCAPE_SUBSTEP_RAD,
    ESCAPE_SUBSTEP_RAIL_M,
    FINE_STEP_M,
    MIN_SPEED_SCALE,
    PLANNER_ESCAPE_EPS_M,
    RAIL_MAX_STEP_M,
    REARM_MARGIN_M,
    _tick_count,
    time_parameterize,
)

HOME_J = [0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0]

# Verified on guardrail_rail: yawed-toward-each-other arms; min monitored
# cross-arm clearance ~34 mm at rail 0.24 and ~14 mm at rail 0.26.
CLOSE_START_ARM0 = [1.35] + HOME_J[1:] + [0.24]
INSIDE_BAND_ARM0 = [1.35] + HOME_J[1:] + [0.27]  # 4.2-4.9 mm < delta = 8 mm, 7 pairs
#   (re-measured 2026-09-09: rail 0.265 is clear, 0.2675 is the first inside the shell)
PENETRATING_ARM0 = [1.35] + HOME_J[1:] + [0.30]  # link6-link6 raw overlap
START_ARM1 = [-1.35] + HOME_J[1:] + [0.48]
GOAL_ARM0 = HOME_J + [0.05]
GOAL_ARM1 = HOME_J + [0.45]
# Fingers/platform over arm1's static rail bed: in collision in ANY ordering.
BAD_GOAL_ARM0 = [1.5, 0.6, 0.0, 0.4, 0.0, 1.6, 0.0, 0.45]

# Verified on guardrail_rail (2026-09-08): arm1 starts yawed toward arm0 at rail
# 0.46, exactly where arm0's yawed goal at rail 0.26 wants to be (link7/link7
# overlap); arm1's goal is the retracted keyframe posture. Both starts are far
# apart, so the clearance heuristic ties and picks ["arm0", "arm1"] — which
# fails with goal_in_collision; only the reversed order (arm1 retracts first)
# plans. Executing arm0 first would drive it into arm1's start.
BLOCKING_START_ARM1 = [-1.35] + HOME_J[1:] + [0.46]
YAWED_GOAL_ARM0 = [1.35] + HOME_J[1:] + [0.26]
FAR_START_ARM0 = HOME_J + [0.05]

# Verified on guardrail_env: extended posture whose TCP x ~ 0.62 pierces the
# pedestal band; rail 0.075 / 0.575 put it left/right of the pedestal.
EXT_J = [0.0, 0.0, 0.0, 1.1, 0.0, 0.4, 0.0]
DETOUR_START = EXT_J + [0.075]
DETOUR_GOAL = EXT_J + [0.575]


@pytest.fixture(scope="module")
def rail_twin() -> DigitalTwin:
    return DigitalTwin(REGISTRY.build("guardrail_rail"), inflation_m=0.008)


@pytest.fixture(scope="module")
def env_twin() -> DigitalTwin:
    return DigitalTwin(REGISTRY.build("guardrail_env"), inflation_m=0.008)


def _assert_waypoints_collision_free(
    twin: DigitalTwin, req: PlanRequest, res, order: list[str] | None = None
) -> None:
    """Every waypoint AND every interpolated edge sample passes check_config,
    in the sequential execution context (earlier arms advance to goal), arms
    executed one after another in ``res.arm_order`` (the execution contract)
    unless an explicit ``order`` is given."""
    if order is None:
        order = res.arm_order
        assert sorted(order) == sorted(res.waypoints)  # a permutation of the planned arms
        assert list(res.waypoints) == order  # insertion order mirrors the contract
    ctx = np.array(twin._q_meas_full)
    for arm_id, q in req.q_start.items():
        ctx[twin.addr[arm_id].qpos_adr] = q
    for arm_id in order:
        wps = res.waypoints[arm_id]
        adr = twin.addr[arm_id].qpos_adr
        step = np.full(len(adr), req.max_step_rad)
        if twin.addr[arm_id].has_rail:
            step[-1] = RAIL_MAX_STEP_M
        for q_a, q_b in zip(wps[:-1], wps[1:], strict=False):
            q_a, q_b = np.asarray(q_a), np.asarray(q_b)
            n = max(1, int(np.ceil(np.max(np.abs(q_b - q_a) / step))))
            for i in range(n + 1):
                ctx[adr] = q_a + (q_b - q_a) * i / n
                assert twin.check_config(ctx), (arm_id, i)
        ctx[adr] = wps[-1]


def test_two_arm_close_swap_succeeds(rail_twin):
    planner = ResetPlanner(rail_twin)
    req = PlanRequest(
        q_start={"arm0": CLOSE_START_ARM0, "arm1": START_ARM1},
        q_goal={"arm0": GOAL_ARM0, "arm1": GOAL_ARM1},
    )
    t0 = time.monotonic()
    res = planner.plan(req)
    assert res.ok, res.failure
    assert time.monotonic() - t0 < 2 * req.timeout_s  # <= 2 orderings tried
    assert set(res.waypoints) == {"arm0", "arm1"}
    # first ordering succeeds here: arm_order reports the heuristic's pick verbatim
    assert res.arm_order == planner._heuristic_order(req, ["arm0", "arm1"])
    _assert_waypoints_collision_free(rail_twin, req, res)
    for arm_id in ("arm0", "arm1"):
        assert np.allclose(res.waypoints[arm_id][0], req.q_start[arm_id])
        assert np.allclose(res.waypoints[arm_id][-1], req.q_goal[arm_id])


@pytest.mark.parametrize("order", [["arm0", "arm1"], ["arm1", "arm0"]])
def test_explicit_arm_order_is_echoed_verbatim(rail_twin, order):
    """A request-level ``arm_order`` is used as given (no retry) and reported back."""
    planner = ResetPlanner(rail_twin)
    req = PlanRequest(
        q_start={"arm0": CLOSE_START_ARM0, "arm1": START_ARM1},
        q_goal={"arm0": GOAL_ARM0, "arm1": GOAL_ARM1},
        arm_order=order,
    )
    res = planner.plan(req)
    assert res.ok, res.failure
    assert res.arm_order == order
    assert list(res.waypoints) == order
    _assert_waypoints_collision_free(rail_twin, req, res)


def test_reversed_retry_reports_the_order_actually_used(rail_twin, monkeypatch):
    """Heuristic order fails, the reversed retry plans: ``arm_order`` is the retry's.

    The 2026-09-08 real-cell incident: waypoints from a sequential plan were
    executed simultaneously. This pins the contract from both sides — the
    reported order is collision-free executed one arm at a time, and the OTHER
    order (the one the heuristic tried first) is not.
    """
    planner = ResetPlanner(rail_twin)
    req = PlanRequest(
        q_start={"arm0": FAR_START_ARM0, "arm1": BLOCKING_START_ARM1},
        q_goal={"arm0": YAWED_GOAL_ARM0, "arm1": GOAL_ARM1},
    )
    assert planner._heuristic_order(req, ["arm0", "arm1"]) == ["arm0", "arm1"]
    tried: list[list[str]] = []
    real = planner._plan_ordered

    def spy(r, order):
        tried.append(list(order))
        return real(r, order)

    monkeypatch.setattr(planner, "_plan_ordered", spy)
    res = planner.plan(req)
    assert tried == [["arm0", "arm1"], ["arm1", "arm0"]]  # exactly one reversed retry
    assert res.ok, res.failure
    assert res.arm_order == ["arm1", "arm0"]
    assert list(res.waypoints) == res.arm_order
    _assert_waypoints_collision_free(rail_twin, req, res)
    # The heuristic's order is NOT safe: arm0's goal meets arm1 still at its start.
    ctx = np.array(rail_twin._q_meas_full)
    ctx[rail_twin.addr["arm0"].qpos_adr] = res.waypoints["arm0"][-1]
    ctx[rail_twin.addr["arm1"].qpos_adr] = req.q_start["arm1"]
    assert not rail_twin.check_config(ctx)


def test_explicit_failing_order_is_not_retried_and_reports_no_order(rail_twin, monkeypatch):
    planner = ResetPlanner(rail_twin)
    tried: list[list[str]] = []
    real = planner._plan_ordered
    monkeypatch.setattr(
        planner, "_plan_ordered", lambda r, o: (tried.append(list(o)), real(r, o))[1]
    )
    res = planner.plan(
        PlanRequest(
            q_start={"arm0": FAR_START_ARM0, "arm1": BLOCKING_START_ARM1},
            q_goal={"arm0": YAWED_GOAL_ARM0, "arm1": GOAL_ARM1},
            arm_order=["arm0", "arm1"],
        )
    )
    assert tried == [["arm0", "arm1"]]
    assert not res.ok and res.failure == "goal_in_collision"
    assert res.arm_order == [] and res.waypoints == {}  # nothing is safe to execute


def test_start_inside_inflation_shell_escapes(rail_twin):
    """A start parked inside the 8 mm shell can still plan out (escape phase, below)."""
    planner = ResetPlanner(rail_twin)
    req = PlanRequest(
        q_start={"arm0": INSIDE_BAND_ARM0, "arm1": START_ARM1},
        q_goal={"arm0": GOAL_ARM0, "arm1": GOAL_ARM1},
    )
    res = planner.plan(req)
    assert res.ok, res.failure


def test_goal_in_collision_reports_pair(rail_twin):
    planner = ResetPlanner(rail_twin)
    res = planner.plan(
        PlanRequest(
            q_start={"arm0": CLOSE_START_ARM0, "arm1": START_ARM1},
            q_goal={"arm0": BAD_GOAL_ARM0, "arm1": GOAL_ARM1},
        )
    )
    assert not res.ok and res.failure == "goal_in_collision"
    assert res.failing_pair is not None
    labels = sorted(res.failing_pair)
    assert labels[0].startswith("arm0_") and labels[1].startswith("arm1_")
    assert res.waypoints == {}
    assert res.arm_order == []


def test_start_in_collision_reports_pair(rail_twin):
    planner = ResetPlanner(rail_twin)
    res = planner.plan(
        PlanRequest(
            q_start={"arm0": PENETRATING_ARM0, "arm1": START_ARM1},
            q_goal={"arm0": GOAL_ARM0, "arm1": GOAL_ARM1},
        )
    )
    assert not res.ok and res.failure == "start_in_collision"
    assert res.failing_pair is not None
    labels = sorted(res.failing_pair)
    assert labels[0].startswith("arm0_") and labels[1].startswith("arm1_")
    assert res.arm_order == []


def test_pedestal_detour_within_budget(env_twin):
    planner = ResetPlanner(env_twin)
    req = PlanRequest(q_start={"arm0": DETOUR_START}, q_goal={"arm0": DETOUR_GOAL})
    t0 = time.monotonic()
    res = planner.plan(req)
    wall = time.monotonic() - t0
    assert res.ok, res.failure
    assert wall < req.timeout_s  # single-arm wall time < 5 s
    assert len(res.waypoints["arm0"]) >= 3  # a straight edge is blocked
    assert res.arm_order == ["arm0"]
    _assert_waypoints_collision_free(env_twin, req, res)


def test_starved_planner_times_out(env_twin):
    planner = ResetPlanner(env_twin, params=PlannerParams(max_iters=3))
    res = planner.plan(
        PlanRequest(q_start={"arm0": DETOUR_START}, q_goal={"arm0": DETOUR_GOAL})
    )
    assert not res.ok and res.failure == "timeout"
    assert res.arm_order == []


def test_unknown_arm_rejected(rail_twin):
    planner = ResetPlanner(rail_twin)
    with pytest.raises(ValueError):
        planner.plan(PlanRequest(q_start={"nope": HOME_J + [0.1]}, q_goal={"nope": HOME_J + [0.2]}))


def test_twin_plan_delegates(rail_twin):
    res = rail_twin.plan(
        PlanRequest(
            q_start={"arm0": CLOSE_START_ARM0, "arm1": START_ARM1},
            q_goal={"arm0": GOAL_ARM0, "arm1": GOAL_ARM1},
        )
    )
    assert res.ok
    assert sorted(res.arm_order) == ["arm0", "arm1"]
    assert list(res.waypoints) == res.arm_order  # the contract survives the facade


def test_time_parameterize_monotone_and_capped():
    wps = [[0.0] * 8, [0.3] * 7 + [0.05], [0.6] * 7 + [0.10]]
    timed = time_parameterize(wps, has_rail=True)
    ts = [t for t, _ in timed]
    assert ts[0] == 0.0 and all(b > a for a, b in zip(ts, ts[1:], strict=False))
    p = PlannerParams()
    # rail dim is the slowest here: 0.05 m at 0.1 m/s -> >= 0.5 s per segment
    assert ts[1] >= 0.05 / p.rail_vel_m_s - 1e-9


# -- pinched start: the escape phase mirrors the gate (2026-09-09 incident) ------------------
# Real cell, 2026-09-09 01:14 (var/logs/runtime.log): the Manipulation Arm's right finger
# sat 2.1 mm from the Perception Arm's link3, inside the 8 mm shell. The planner whitelisted
# the pair and planned ONE straight segment that first CLOSED it to 1.1 mm; the gate's T8
# rule held the first step and the plan was cancelled after plan_gate_hold_s. The tests below
# rebuild that start on the mavis_v2 twin by bisection (the seeded Perception Arm initial
# condition at carriage 0.30; the Manipulation Arm on the joint-space line from a posture
# reaching toward it to its own initial condition, stopped where the pair reads 2.0 mm).
DEG = np.pi / 180.0
MAVIS_VIEW_INIT = [x * DEG for x in (0.0, 0.8, 0.0, 28.9, 0.0, 28.2, 0.0)] + [0.30]
MAVIS_GRIP_INIT = [x * DEG for x in (-180.0, -12.0, -20.0, 30.0, -5.0, 35.0, -8.9)] + [0.30]
MAVIS_GRIP_FREE = [0.7, -1.1, 0.0, 0.3, 0.1, -0.4, 0.0, 0.30]  # 14 cm clear, reaching at view
PINCH_PAIR = ("grip_right_finger", "view_link3")
PINCH_DEPTH_M = 0.002  # the incident's 2.1 mm
# Verified on guardrail_rail (2026-09-09): arm0 yawed toward arm1 at rail 0.2725 has SEVEN
# cross-arm pairs inside the shell at 1.7-2.6 mm (link6/link7/gripper bases), none touching.
MULTI_PINCH_ARM0 = [1.35] + HOME_J[1:] + [0.2725]


@pytest.fixture(scope="module")
def mavis_twin() -> DigitalTwin:
    return DigitalTwin(REGISTRY.build("mavis_v2"), inflation_m=0.008)


def pinch_on_line(twin, arm, q_free, q_far, others, pair, depth_m, n_scan=400):
    """Config on the joint-space line ``q_free -> q_far`` where ``pair`` reads ``depth_m``
    (bisection; the line must enter the shell through that pair while closing)."""
    q_free, q_far = np.asarray(q_free, dtype=float), np.asarray(q_far, dtype=float)

    def dist(t: float) -> float:
        return twin.pair_distance(pair, {arm: q_free + t * (q_far - q_free), **others})

    assert dist(0.0) > twin.inflation_m
    ts = np.linspace(0.0, 1.0, n_scan + 1)
    inside = next(t for t in ts if dist(t) < depth_m)
    lo, hi = 0.0, float(inside)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dist(mid) > depth_m:
            lo = mid
        else:
            hi = mid
    q = q_free + hi * (q_far - q_free)
    assert abs(dist(hi) - depth_m) < 1e-6
    return q


def _violating_pairs(twin, q_by_arm) -> dict[tuple[str, str], float]:
    ctx = np.array(twin._q_meas_full)
    for arm_id, q in q_by_arm.items():
        ctx[twin.addr[arm_id].qpos_adr] = q
    out: dict[tuple[str, str], float] = {}
    for pair, d in twin.check_config_violations(ctx):
        key = tuple(sorted(pair))
        out[key] = min(d, out.get(key, np.inf))
    return out


def _lever_length(dq: np.ndarray) -> float:
    n = min(7, dq.shape[0])
    return float(np.sum(np.abs(dq[:n]) * _LEVER_ARM_M[:n]) + np.sum(np.abs(dq[7:])))


def _movable(planner, arm_id, pairs):
    """The pairs of ``pairs`` the planned arm's joints can move (the others are constants)."""
    return {p for p in pairs if planner._moves(arm_id, tuple(sorted(p)))}


def assert_escape_mirrors_gate(twin, req, res, arm_id) -> tuple[int, int]:
    """The gate's T8 rule replayed over the planned path of ``arm_id`` in its sequential
    context, tick by tick at the speed of the request (the runtime executor walks a segment
    in ``_tick_count`` equal ticks): while any start-pinched pair - inside the shell OR the
    gate's hysteresis band, and movable by this arm - is not yet re-armed
    (> δ + REARM_MARGIN_M) every such pair opens by ≥ ESCAPE_RATE_MARGIN x
    PLANNER_ESCAPE_EPS_M per tick and no OTHER movable pair enters the shell (checked at the
    escape's coarse resolution); afterwards every sample at the gate's resolution (arm-point
    travel ≤ FINE_STEP_M) is collision-free. Returns (escape segments, escape ticks)."""
    delta = twin.inflation_m
    planner = ResetPlanner(twin)
    speed = float(req.speed_scale)
    ctx = {a: np.asarray(q, dtype=float) for a, q in req.q_start.items()}
    for earlier in res.arm_order[: res.arm_order.index(arm_id)]:
        ctx[earlier] = np.asarray(req.q_goal[earlier], dtype=float)  # sequential execution
    wps = [np.asarray(w, dtype=float) for w in res.waypoints[arm_id]]
    assert np.allclose(wps[0], req.q_start[arm_id]) and np.allclose(wps[-1], req.q_goal[arm_id])
    # V0 as the planner sees it: this arm's movable pairs within δ + hysteresis at the start
    ctx_full = np.array(twin._q_meas_full)
    for other, q in ctx.items():
        ctx_full[twin.addr[other].qpos_adr] = q
    pinched = set(
        planner._contacts_within(
            wps[0], arm_id, planner.addr[arm_id], ctx_full, planner._band_model,
            mujoco.MjData(planner._band_model),
        )
    )
    assert pinched, "the start must be inside the shell or the band"
    active = set(pinched)
    d_prev = {p: twin.pair_distance(p, {**ctx, arm_id: wps[0]}) for p in pinched}
    sub = np.full(wps[0].shape[0], ESCAPE_SUBSTEP_RAD)
    if wps[0].shape[0] > 7:
        sub[7:] = ESCAPE_SUBSTEP_RAIL_M
    need = ESCAPE_RATE_MARGIN * PLANNER_ESCAPE_EPS_M - 1e-12
    escape_segments = escape_ticks = 0
    for i, (q_a, q_b) in enumerate(zip(wps[:-1], wps[1:], strict=False)):
        if active:
            escape_segments += 1
            # (a) no NEW movable violation at the coarse resolution
            n = max(
                1,
                int(np.ceil(np.max(np.abs(q_b - q_a) / sub))),
                int(np.ceil(_lever_length(q_b - q_a) / FINE_STEP_M)),
            )
            for k in range(1, n + 1):
                q = q_a + (q_b - q_a) * (k / n)
                viol = _movable(planner, arm_id, _violating_pairs(twin, {**ctx, arm_id: q}))
                assert viol <= active, (i, k, viol - active)
            # (b) the gate's strict opening on EVERY executor tick of the requested speed
            n = _tick_count(q_b - q_a, speed)
            for k in range(1, n + 1):
                q = q_a + (q_b - q_a) * (k / n)
                escape_ticks += 1
                d_now = {p: twin.pair_distance(p, {**ctx, arm_id: q}) for p in pinched}
                for p in active:
                    assert d_now[p] >= d_prev[p] + need, (i, k, p, d_prev[p], d_now[p])
                for p in pinched - active:
                    assert d_now[p] >= delta + REARM_MARGIN_M - 1e-9, (i, k, p)
                d_prev = d_now
                for p in list(active):
                    if d_now[p] > delta + REARM_MARGIN_M:
                        active.discard(p)
            continue
        n = max(1, int(np.ceil(_lever_length(q_b - q_a) / FINE_STEP_M)))
        for k in range(1, n + 1):
            q = q_a + (q_b - q_a) * (k / n)
            viol = _movable(planner, arm_id, _violating_pairs(twin, {**ctx, arm_id: q}))
            assert not viol, (arm_id, i, k, viol)  # the gate would hold here
    assert not active, "the escape never re-armed every pinched pair"
    return escape_segments, escape_ticks


def test_pinched_finger_escapes_then_plans_to_the_far_side(mavis_twin):
    """(a) The incident start: right finger 2.0 mm from view_link3; the goal is on the far
    side of the Perception Arm's link3 (the straight line would sweep the gripper through
    it). The plan opens the pair monotonically to δ + 5 mm first, creates no new violation
    while doing so, then routes around with every gate-resolution sample clear."""
    q_start = pinch_on_line(
        mavis_twin, "grip", MAVIS_GRIP_FREE, MAVIS_GRIP_INIT, {"view": MAVIS_VIEW_INIT},
        PINCH_PAIR, PINCH_DEPTH_M,
    )
    free, init = np.asarray(MAVIS_GRIP_FREE), np.asarray(MAVIS_GRIP_INIT)
    q_goal = free + 0.25 * (init - free)  # far side, 13.6 cm clear of the pair
    req = PlanRequest(
        q_start={"grip": q_start.tolist(), "view": MAVIS_VIEW_INIT},
        q_goal={"grip": q_goal.tolist(), "view": MAVIS_VIEW_INIT},
    )
    start_viol = _violating_pairs(mavis_twin, {"grip": q_start, "view": MAVIS_VIEW_INIT})
    assert set(start_viol) == {tuple(sorted(PINCH_PAIR))}
    assert 0.0 < start_viol[tuple(sorted(PINCH_PAIR))] < mavis_twin.inflation_m
    t0 = time.monotonic()
    res = ResetPlanner(mavis_twin).plan(req)
    assert res.ok, (res.failure, res.failing_pair)
    assert time.monotonic() - t0 < req.timeout_s
    assert res.arm_order[0] == "grip"  # the pinched arm is the deepest -> first
    segments, samples = assert_escape_mirrors_gate(mavis_twin, req, res, "grip")
    assert 1 <= segments <= 40 and samples >= 1


def test_start_pinched_against_two_or_more_pairs_escapes(rail_twin):
    """(b) Seven cross-arm pairs inside the shell at 1.7-2.6 mm at once: every one of them
    must open every sample until re-armed (the gate requires EVERY blocked pair to open)."""
    req = PlanRequest(
        q_start={"arm0": MULTI_PINCH_ARM0, "arm1": START_ARM1},
        q_goal={"arm0": GOAL_ARM0, "arm1": GOAL_ARM1},
    )
    start_viol = _violating_pairs(rail_twin, {"arm0": MULTI_PINCH_ARM0, "arm1": START_ARM1})
    assert len(start_viol) >= 2 and all(0.0 < d < 0.008 for d in start_viol.values())
    res = ResetPlanner(rail_twin).plan(req)
    assert res.ok, (res.failure, res.failing_pair)
    segments, _samples = assert_escape_mirrors_gate(rail_twin, req, res, "arm0")
    assert segments >= 1


def test_no_escape_names_the_tightest_pair(rail_twin):
    """(c) ``no_escape``: the pinched start cannot be opened. A geometrically boxed-in start
    (every opening of one pair closes another, none touching) needs a purpose-built scene;
    the escape budget set to zero exercises the same failure path - kind, tightest pair,
    nothing to execute - deterministically."""
    planner = ResetPlanner(rail_twin, params=PlannerParams(escape_max_steps=0))
    res = planner.plan(
        PlanRequest(
            q_start={"arm0": INSIDE_BAND_ARM0, "arm1": START_ARM1},
            q_goal={"arm0": GOAL_ARM0, "arm1": GOAL_ARM1},
        )
    )
    assert not res.ok and res.failure == "no_escape"
    start_viol = _violating_pairs(rail_twin, {"arm0": INSIDE_BAND_ARM0, "arm1": START_ARM1})
    assert res.failing_pair == min(start_viol, key=start_viol.get)
    assert res.waypoints == {} and res.arm_order == []
    # the same start with the budget restored escapes (the old whitelist test above)
    assert ResetPlanner(rail_twin).plan(
        PlanRequest(
            q_start={"arm0": INSIDE_BAND_ARM0, "arm1": START_ARM1},
            q_goal={"arm0": GOAL_ARM0, "arm1": GOAL_ARM1},
        )
    ).ok


def test_goal_inside_the_shell_is_refused_even_for_the_pinched_pair(mavis_twin):
    """The whitelist also excused the GOAL: a goal 1.0 mm from the pinched pair planned as a
    straight closing segment (the incident's shape). Now it is ``goal_in_collision``."""
    q_start = pinch_on_line(
        mavis_twin, "grip", MAVIS_GRIP_FREE, MAVIS_GRIP_INIT, {"view": MAVIS_VIEW_INIT},
        PINCH_PAIR, PINCH_DEPTH_M,
    )
    q_goal = pinch_on_line(
        mavis_twin, "grip", MAVIS_GRIP_FREE, MAVIS_GRIP_INIT, {"view": MAVIS_VIEW_INIT},
        PINCH_PAIR, 0.001,
    )
    res = ResetPlanner(mavis_twin).plan(
        PlanRequest(
            q_start={"grip": q_start.tolist(), "view": MAVIS_VIEW_INIT},
            q_goal={"grip": q_goal.tolist(), "view": MAVIS_VIEW_INIT},
            arm_order=["grip", "view"],
        )
    )
    assert not res.ok and res.failure == "goal_in_collision"
    assert res.failing_pair == tuple(sorted(PINCH_PAIR))


def test_path_is_verified_at_the_gate_resolution(mavis_twin, monkeypatch):
    """RRT edges are sampled at 0.05 rad; the gate checks every tick. Without the final
    fine pass the seeded route of test (a) grazes ``grip_left_finger`` / ``view_link4`` at
    7.9 mm between two validated samples (found by the first gate replay, 2026-09-09) - a
    permanent hold on the real cell. With it every sample ≤ FINE_STEP_M of arm travel apart
    is clear (asserted by ``assert_escape_mirrors_gate`` in test (a))."""
    q_start = pinch_on_line(
        mavis_twin, "grip", MAVIS_GRIP_FREE, MAVIS_GRIP_INIT, {"view": MAVIS_VIEW_INIT},
        PINCH_PAIR, PINCH_DEPTH_M,
    )
    free, init = np.asarray(MAVIS_GRIP_FREE), np.asarray(MAVIS_GRIP_INIT)
    req = PlanRequest(
        q_start={"grip": q_start.tolist(), "view": MAVIS_VIEW_INIT},
        q_goal={"grip": (free + 0.25 * (init - free)).tolist(), "view": MAVIS_VIEW_INIT},
        arm_order=["grip", "view"],
    )
    planner = ResetPlanner(mavis_twin)
    monkeypatch.setattr(planner, "_repair_fine", lambda path, *a, **k: list(path))
    res = planner.plan(req)
    assert res.ok
    grazes = []
    ctx = {"view": np.asarray(MAVIS_VIEW_INIT)}
    wps = [np.asarray(w) for w in res.waypoints["grip"]]
    for q_a, q_b in zip(wps[:-1], wps[1:], strict=False):
        n = max(1, int(np.ceil(_lever_length(q_b - q_a) / FINE_STEP_M)))
        for k in range(1, n):
            q = q_a + (q_b - q_a) * (k / n)
            viol = _violating_pairs(mavis_twin, {**ctx, "grip": q})
            if viol and tuple(sorted(PINCH_PAIR)) not in viol:
                grazes.append(viol)
    if not grazes:
        pytest.skip("the seeded RRT route no longer grazes the shell between its samples")
    assert all(0.0 < d < mavis_twin.inflation_m for g in grazes for d in g.values())
    # the shipped planner repairs exactly this
    res2 = ResetPlanner(mavis_twin).plan(req)
    assert res2.ok
    assert_escape_mirrors_gate(mavis_twin, req, res2, "grip")


# -- 2026-09-09 review: constants for the free arm, the band, the tick model, honesty ---------
# Found by random search on the mavis_v2 twin (2026-09-09 review): the Perception Arm pinched
# against ITSELF (view_link1 / view_link5 at 7.96 mm - an intra-arm pair the monitored set does
# not contain); a Manipulation Arm start with grip_link4 6.00 mm from view_link2 (inside the
# shell) and 9.73 mm from view_link3 (inside the gate's [δ, δ + 2 mm) hysteresis band).
VIEW_SELF_PINCH = [-0.405, -0.8213, -0.1354, -0.0209, 0.6008, 0.5786, 0.4693, 0.3]
GRIP_BAND_START = [-2.9805, -0.91488, -1.46179, 1.67498, 0.58877, 0.1508, -0.36369, 0.15581]
GRIP_LIFTED = (
    list(MAVIS_GRIP_INIT[:3]) + [MAVIS_GRIP_INIT[3] + 0.3] + list(MAVIS_GRIP_INIT[4:7]) + [0.45]
)


def test_pairs_the_arm_cannot_move_are_constants(mavis_twin):
    """``_moves``: the other arm's pairs, and pairs whose two bodies hang under the same set
    of this arm's joints (the gripper's own knuckles, gripper base vs finger: only the finger
    joints move them) are constants for the planned arm; a link pair or an arm-vs-world pair
    is movable."""
    planner = ResetPlanner(mavis_twin)
    assert not planner._moves("grip", ("view_link1", "view_link5"))
    assert not planner._moves("grip", ("table", "view_d435_mount"))
    assert not planner._moves("grip", ("grip_left_outer_knuckle", "grip_right_inner_knuckle"))
    assert not planner._moves("grip", ("grip_left_finger", "grip_xarm_gripper_base_link"))
    assert planner._moves("grip", ("grip_link3", "grip_link7"))
    assert planner._moves("grip", ("grip_left_finger", "table"))
    assert planner._moves("grip", ("grip_right_finger", "view_link3"))
    assert planner._moves("view", ("view_link1", "view_link5"))
    assert not planner._moves("view", ("grip_right_finger", "table"))


def test_the_free_arm_plans_while_the_other_arm_is_pinched(mavis_twin):
    """Review blocker: the first escape implementation built V0 from EVERY pair violating at
    the start, so the FREE arm failed ``no_escape`` naming the OTHER arm's pair (6/6 cases:
    joint-panel goto, rail-homing pre-positioning, two-arm returns with the free arm first).
    The gate never holds an arm outside the offending set; neither does the planner now."""
    planner = ResetPlanner(mavis_twin)
    viol = _violating_pairs(mavis_twin, {"grip": MAVIS_GRIP_INIT, "view": VIEW_SELF_PINCH})
    assert viol and all(a.startswith("view_") for p in viol for a in p), viol
    assert all(0.0 < d < mavis_twin.inflation_m for d in viol.values())
    single = PlanRequest(
        q_start={"grip": MAVIS_GRIP_INIT, "view": VIEW_SELF_PINCH}, q_goal={"grip": GRIP_LIFTED}
    )
    res = planner.plan(single)
    assert res.ok and res.arm_order == ["grip"], (res.failure, res.failing_pair)
    _assert_waypoints_collision_free_for(mavis_twin, planner, single, res, "grip")
    two = PlanRequest(
        q_start={"grip": MAVIS_GRIP_INIT, "view": VIEW_SELF_PINCH},
        q_goal={"grip": GRIP_LIFTED, "view": MAVIS_VIEW_INIT},
    )
    grip_first = two.model_copy(update={"arm_order": ["grip", "view"]})
    res2 = planner.plan(grip_first)
    assert res2.ok and res2.arm_order == ["grip", "view"], (res2.failure, res2.failing_pair)
    # the pinched view escapes its own intra-arm pair afterwards (grip parked at its goal)
    assert_escape_mirrors_gate(mavis_twin, grip_first, res2, "view")
    # the heuristic sees the intra-arm pinch (the monitored sweep does not) -> view first
    assert planner._heuristic_order(two, ["grip", "view"]) == ["view", "grip"]
    res3 = planner.plan(two)
    assert res3.ok and res3.arm_order == ["view", "grip"], (res3.failure, res3.failing_pair)


def _assert_waypoints_collision_free_for(twin, planner, req, res, arm_id) -> None:
    """Every edge sample of ``arm_id``'s path is free of MOVABLE violations (the other arm's
    constants are allowed to persist), in the sequential context."""
    ctx = {a: np.asarray(q, dtype=float) for a, q in req.q_start.items()}
    for earlier in res.arm_order[: res.arm_order.index(arm_id)]:
        ctx[earlier] = np.asarray(req.q_goal[earlier], dtype=float)
    wps = [np.asarray(w) for w in res.waypoints[arm_id]]
    step = np.full(wps[0].shape[0], req.max_step_rad)
    step[7:] = RAIL_MAX_STEP_M
    for q_a, q_b in zip(wps[:-1], wps[1:], strict=False):
        n = max(1, int(np.ceil(np.max(np.abs(q_b - q_a) / step))))
        for k in range(n + 1):
            q = q_a + (q_b - q_a) * (k / n)
            viol = _movable(planner, arm_id, _violating_pairs(twin, {**ctx, arm_id: q}))
            assert not viol, (arm_id, k, viol)


def test_an_arm_left_in_place_is_not_planned(mavis_twin):
    """An arm whose goal IS its start is left alone even when pinched: nothing will be
    commanded for it, and the arms after it are planned around it. Here the Perception Arm
    stays at its initial condition while the Manipulation Arm's finger sits 2 mm from its
    link3 - the view-first ordering used to fail ``goal_in_collision`` on the view's own
    (unmoved) goal against the grip's start."""
    q_start = pinch_on_line(
        mavis_twin, "grip", MAVIS_GRIP_FREE, MAVIS_GRIP_INIT, {"view": MAVIS_VIEW_INIT},
        PINCH_PAIR, PINCH_DEPTH_M,
    )
    req = PlanRequest(
        q_start={"grip": q_start.tolist(), "view": MAVIS_VIEW_INIT},
        q_goal={"grip": MAVIS_GRIP_INIT, "view": MAVIS_VIEW_INIT},
        arm_order=["view", "grip"],
    )
    res = ResetPlanner(mavis_twin).plan(req)
    assert res.ok, (res.failure, res.failing_pair)
    assert res.arm_order == ["view", "grip"]
    assert len(res.waypoints["view"]) == 2
    assert all(np.allclose(w, MAVIS_VIEW_INIT) for w in res.waypoints["view"])
    assert_escape_mirrors_gate(mavis_twin, req, res, "grip")


def test_goal_in_collision_is_reported_before_the_escape_is_tried(mavis_twin):
    """Honesty (review item 3): the goal is judged BEFORE the escape - it is the certain and
    actionable diagnosis. A start no step can open (escape budget 0) with a goal inside the
    shell reports ``goal_in_collision``, not ``no_escape``."""
    q_start = pinch_on_line(
        mavis_twin, "grip", MAVIS_GRIP_FREE, MAVIS_GRIP_INIT, {"view": MAVIS_VIEW_INIT},
        PINCH_PAIR, PINCH_DEPTH_M,
    )
    q_goal = pinch_on_line(
        mavis_twin, "grip", MAVIS_GRIP_FREE, MAVIS_GRIP_INIT, {"view": MAVIS_VIEW_INIT},
        PINCH_PAIR, 0.001,
    )
    planner = ResetPlanner(mavis_twin, params=PlannerParams(escape_max_steps=0))
    req = PlanRequest(
        q_start={"grip": q_start.tolist(), "view": MAVIS_VIEW_INIT},
        q_goal={"grip": q_goal.tolist(), "view": MAVIS_VIEW_INIT},
        arm_order=["grip", "view"],
    )
    res = planner.plan(req)
    assert not res.ok and res.failure == "goal_in_collision"
    assert res.failing_pair == tuple(sorted(PINCH_PAIR))
    # the same start with a clear goal is the budget-zero ``no_escape`` of test (c)
    clear_goal = {"grip": MAVIS_GRIP_INIT, "view": MAVIS_VIEW_INIT}
    res = planner.plan(req.model_copy(update={"q_goal": clear_goal}))
    assert not res.ok and res.failure == "no_escape"
    assert res.failing_pair == tuple(sorted(PINCH_PAIR))


def test_a_pair_inside_the_gate_band_is_escaped_like_a_pinched_one(mavis_twin):
    """Review item 5: the gate keeps a pair inside its hysteresis band [δ, δ + hysteresis)
    in ``_block_pairs`` and demands that it opens on every tick while blocked. ``grip_link4``
    starts 6.0 mm from ``view_link2`` (the shell) and 9.73 mm from ``view_link3`` (the band):
    both are escaped - the band pair opens monotonically from the first segment on until it
    is re-armed - instead of the band pair being merely tolerated (≥ δ)."""
    planner = ResetPlanner(mavis_twin)
    assert planner.hysteresis_m == 0.002  # DigitalTwin default = SafetyConfig default
    q_start = np.asarray(GRIP_BAND_START)
    viol = _violating_pairs(mavis_twin, {"grip": q_start, "view": MAVIS_VIEW_INIT})
    assert set(viol) == {("grip_link4", "view_link2")}
    band_pair = ("grip_link4", "view_link3")
    view = np.asarray(MAVIS_VIEW_INIT)
    d_b = mavis_twin.pair_distance(band_pair, {"grip": q_start, "view": view})
    assert mavis_twin.inflation_m <= d_b < mavis_twin.inflation_m + planner.hysteresis_m
    req = PlanRequest(
        q_start={"grip": q_start.tolist(), "view": MAVIS_VIEW_INIT},
        q_goal={"grip": MAVIS_GRIP_INIT, "view": MAVIS_VIEW_INIT},
    )
    res = planner.plan(req)
    assert res.ok, (res.failure, res.failing_pair)
    segments, ticks = assert_escape_mirrors_gate(mavis_twin, req, res, "grip")
    assert segments >= 1 and ticks >= 1
    wps = [np.asarray(w) for w in res.waypoints["grip"]]
    d_b1 = mavis_twin.pair_distance(band_pair, {"grip": wps[1], "view": view})
    assert d_b1 > d_b + PLANNER_ESCAPE_EPS_M


def test_escape_is_judged_per_executor_tick_at_the_requested_speed(mavis_twin):
    """Review item 6: the escape's acceptance is a RATE per executor tick of the session the
    plan runs at (``PlanRequest.speed_scale``; default = the slowest speed offered), not a
    fixed opening per planner sample. A plan judged for 10 % passes the tick-level replay at
    10 % and - ticks being unions of the judged ones - at 100 %; a plan judged for 100 %
    passes at 100 %."""
    q_start = pinch_on_line(
        mavis_twin, "grip", MAVIS_GRIP_FREE, MAVIS_GRIP_INIT, {"view": MAVIS_VIEW_INIT},
        PINCH_PAIR, PINCH_DEPTH_M,
    )
    assert PlanRequest(q_start={}, q_goal={}).speed_scale == MIN_SPEED_SCALE == 0.1
    for judged, replayed in ((0.1, 0.1), (0.1, 1.0), (1.0, 1.0)):
        req = PlanRequest(
            q_start={"grip": q_start.tolist(), "view": MAVIS_VIEW_INIT},
            q_goal={"grip": MAVIS_GRIP_INIT, "view": MAVIS_VIEW_INIT},
            speed_scale=judged,
        )
        res = ResetPlanner(mavis_twin).plan(req)
        assert res.ok, (judged, res.failure, res.failing_pair)
        assert_escape_mirrors_gate(
            mavis_twin, req.model_copy(update={"speed_scale": replayed}), res, "grip"
        )
    # the tick model itself: more ticks at a slower speed, never fewer than one
    dq = np.array([0.025, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002])
    assert _tick_count(dq, 0.1) >= 10 * (_tick_count(dq, 1.0) - 1) + 1 >= 1
    assert _tick_count(np.zeros(8), 0.1) == 1


# -- 2026-09-09 fuzz (runtime tests/test_return_fuzz_mavis_v2.py): held rail, margin, budget --
# A doubly pinched Manipulation Arm start found by the fuzz (seed 20260909 of the first sweep):
# ``grip_right_finger`` / ``view_link2`` at 7.4 mm (the shell) AND the intra-arm
# ``grip_link2`` / ``grip_link5`` at 4.2 mm; the goal is the folded keyframe posture with the
# carriage kept. The escape re-armed both pairs, then every RRT path skimmed the intra-arm
# pair at 8-10 mm for a whole segment, the fine repair ran out of nudges and all three seeds
# failed the same way - ``timeout`` after 0.7 s of a 5 s budget - while the RRT sampled the
# carriage freely (a 12.6 cm excursion on another case of the same sweep).
DOUBLE_PINCH_GRIP = [-2.03, 0.709, 1.865, 0.177, -1.288, -0.843, -0.746, 0.16]
DOUBLE_PINCH_VIEW = [-3.06, -0.007, -2.497, 2.48, -3.552, 0.881, 3.136, 0.437]
FOLDED_GRIP_KEEP_RAIL = [-np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16]


def test_doubly_pinched_start_plans_with_the_gate_margin(mavis_twin):
    """The RRT's predicate carries the fine margin (module docstring): the plan exists, is
    found well inside the budget, escapes both pairs and never skims the shell afterwards
    (``assert_escape_mirrors_gate`` checks every sample at the gate's resolution)."""
    planner = ResetPlanner(mavis_twin)
    viol = _violating_pairs(mavis_twin, {"grip": DOUBLE_PINCH_GRIP, "view": DOUBLE_PINCH_VIEW})
    assert set(viol) == {("grip_right_finger", "view_link2"), ("grip_link2", "grip_link5")}
    assert all(0.0 < d < mavis_twin.inflation_m for d in viol.values())
    req = PlanRequest(
        q_start={"grip": DOUBLE_PINCH_GRIP, "view": DOUBLE_PINCH_VIEW},
        q_goal={"grip": FOLDED_GRIP_KEEP_RAIL},
        speed_scale=1.0,
    )
    t0 = time.monotonic()
    res = planner.plan(req)
    wall = time.monotonic() - t0
    assert res.ok, (res.failure, res.failing_pair)
    assert wall < 0.6 * req.timeout_s, wall
    segments, ticks = assert_escape_mirrors_gate(mavis_twin, req, res, "grip")
    assert segments >= 1 and ticks >= 1


def test_a_carriage_the_request_keeps_is_held(mavis_twin):
    """Rail start == goal (the runtime's joints phase, the rail-homing pre-positioning, a
    joint-panel goto that leaves the slider alone): no waypoint moves the carriage, the
    escape included - the RRT used to sample the rail like any other dof."""
    planner = ResetPlanner(mavis_twin)
    for q_goal in (FOLDED_GRIP_KEEP_RAIL, MAVIS_GRIP_INIT[:7] + [DOUBLE_PINCH_GRIP[7]]):
        res = planner.plan(
            PlanRequest(
                q_start={"grip": DOUBLE_PINCH_GRIP, "view": DOUBLE_PINCH_VIEW},
                q_goal={"grip": q_goal},
                speed_scale=1.0,
            )
        )
        assert res.ok, (res.failure, res.failing_pair)
        rails = {w[7] for w in res.waypoints["grip"]}
        assert rails == {DOUBLE_PINCH_GRIP[7]}, rails
        assert len(res.waypoints["grip"]) >= 3  # not a straight line: the escape + a detour
    # a sub-millimetre carriage request is "keep the carriage" too: the goal snaps to the
    # start's rail (below the runtime's 2 mm arrival tolerance, nothing observable is lost)
    goal = list(FOLDED_GRIP_KEEP_RAIL)
    goal[7] += 0.0005
    res = planner.plan(
        PlanRequest(
            q_start={"grip": DOUBLE_PINCH_GRIP, "view": DOUBLE_PINCH_VIEW},
            q_goal={"grip": goal},
            speed_scale=1.0,
        )
    )
    assert res.ok, (res.failure, res.failing_pair)
    assert {w[7] for w in res.waypoints["grip"]} == {DOUBLE_PINCH_GRIP[7]}
    assert np.allclose(res.waypoints["grip"][-1][:7], goal[:7])
    # ... while a real carriage request still moves it
    goal[7] = DOUBLE_PINCH_GRIP[7] + 0.05
    res = planner.plan(
        PlanRequest(
            q_start={"grip": DOUBLE_PINCH_GRIP, "view": DOUBLE_PINCH_VIEW},
            q_goal={"grip": goal},
            speed_scale=1.0,
        )
    )
    assert res.ok, (res.failure, res.failing_pair)
    assert res.waypoints["grip"][-1][7] == pytest.approx(goal[7])


def test_a_held_carriage_freezes_the_next_arm_where_the_path_ends(rail_twin):
    """Sequential context: arm k+1 is validated against arm k where its path ENDS (the
    snapped rail), not at the request's unreachable sub-millimetre goal."""
    planner = ResetPlanner(rail_twin)
    goal0 = list(GOAL_ARM0)
    goal0[7] = CLOSE_START_ARM0[7] + 0.0004  # keep the carriage (within RAIL_HOLD_TOL_M)
    req = PlanRequest(
        q_start={"arm0": CLOSE_START_ARM0, "arm1": START_ARM1},
        q_goal={"arm0": goal0, "arm1": GOAL_ARM1},
        arm_order=["arm0", "arm1"],
    )
    res = planner.plan(req)
    assert res.ok, (res.failure, res.failing_pair)
    assert res.waypoints["arm0"][-1][7] == CLOSE_START_ARM0[7]
    _assert_waypoints_collision_free(rail_twin, req, res)


def test_starved_planner_spends_its_budget_before_timing_out(env_twin):
    """Re-seeding runs until the request deadline (an RRT that exhausts ``max_iters``
    restarts with a new seed): a starved planner reports ``timeout`` only once the budget
    is really spent, not after a fixed three seeds with most of it unused."""
    planner = ResetPlanner(env_twin, params=PlannerParams(max_iters=3))
    req = PlanRequest(q_start={"arm0": DETOUR_START}, q_goal={"arm0": DETOUR_GOAL}, timeout_s=0.6)
    t0 = time.monotonic()
    res = planner.plan(req)
    wall = time.monotonic() - t0
    assert not res.ok and res.failure == "timeout"
    assert res.failing_pair is not None
    assert 0.9 * req.timeout_s <= wall < 3.0 * req.timeout_s, wall


def test_two_arm_orderings_are_probed_before_either_gets_the_budget(rail_twin, monkeypatch):
    """A two-arm request without an explicit order: both orderings get a short probe
    (ORDER_PROBE_FRACTION of the per-arm budget) before the heuristic ordering may spend
    all of it; a probe that fails deterministically (``goal_in_collision``) is not retried
    with more budget; an explicit order is never probed."""
    from apollo_mavis_v2_sim.planner import ORDER_PROBE_FRACTION

    planner = ResetPlanner(rail_twin)
    seen: list[tuple[list[str], float]] = []
    real = planner._plan_ordered

    def spy(r, order):
        seen.append((list(order), r.timeout_s))
        return real(r, order)

    monkeypatch.setattr(planner, "_plan_ordered", spy)
    req = PlanRequest(
        q_start={"arm0": FAR_START_ARM0, "arm1": BLOCKING_START_ARM1},
        q_goal={"arm0": YAWED_GOAL_ARM0, "arm1": GOAL_ARM1},
    )
    res = planner.plan(req)
    assert res.ok and res.arm_order == ["arm1", "arm0"]
    probe_s = req.timeout_s * ORDER_PROBE_FRACTION
    assert seen == [(["arm0", "arm1"], probe_s), (["arm1", "arm0"], probe_s)]
    seen.clear()
    planner.plan(req.model_copy(update={"arm_order": ["arm1", "arm0"]}))
    assert seen == [(["arm1", "arm0"], req.timeout_s)]


# The 200-start fuzz sweep (runtime tests/test_return_fuzz_mavis_v2.py, seeds 20261013 /
# 20261014): the Perception Arm's camera mount 6.7 mm from ``grip_rail_base`` - the
# Manipulation Arm's STATIC rail (a child of ``world`` with no joint). The planner treated the
# pair as a constant for the Manipulation Arm (correct: no joint of it moves the rail) and
# planned it first; the gate, attributing the pair to both arms by name prefix, held that plan
# from its first moving tick. Pair ownership is kinematic now (twin ``_arms_of_pair``).
RAIL_BASE_PINCH_GRIP = [-2.148, -0.2721, 0.1854, 1.0892, 1.7294, 0.2123, 0.8133, 0.503]
RAIL_BASE_PINCH_VIEW = [0.9105, -1.2391, 2.4242, 0.6749, 0.8878, -0.1517, 2.2932, 0.0952]
RAIL_BASE_PAIR = ("grip_rail_base", "view_d435_mount")


def test_pair_ownership_is_kinematic(mavis_twin):
    """A label belongs to the arm whose joints move its body; the static rail base and the
    world geoms belong to no arm; a label is never shared by two arms."""
    arms = mavis_twin._arms_of_pair
    assert arms(RAIL_BASE_PAIR) == ["view"]
    assert arms(("grip_rail_base", "table")) == []
    assert arms(("grip_rail_platform", "view_link3")) == ["grip", "view"]  # the carriage moves
    assert arms(("grip_link_base", "view_link7")) == ["grip", "view"]  # the arm base rides it
    assert arms(("grip_left_finger", "table")) == ["grip"]
    assert arms(("grip_right_finger", "view_link3")) == ["grip", "view"]
    assert arms(("view_link1", "view_link5")) == ["view"]
    assert arms(("view_d435_mount", "obstacle")) == ["view"]
    owners = [mavis_twin._arms_of_label[label] for label in mavis_twin._geoms_of_label]
    assert all(len(o) <= 1 for o in owners)
    assert mavis_twin._arms_of_label["grip_rail_base"] == ()
    assert mavis_twin._arms_of_label["table"] == ()


def test_the_other_arms_pinch_against_this_arms_rail_base_is_the_other_arms(mavis_twin):
    """The planner's view of the fuzz case: the pair is movable by (and escaped by) the
    Perception Arm only, the Manipulation Arm's plan is a plain one, and the heuristic
    orders the pinched Perception Arm first (its depth, not the Manipulation Arm's)."""
    planner = ResetPlanner(mavis_twin)
    viol = _violating_pairs(
        mavis_twin, {"grip": RAIL_BASE_PINCH_GRIP, "view": RAIL_BASE_PINCH_VIEW}
    )
    assert RAIL_BASE_PAIR in viol and 0.0 < viol[RAIL_BASE_PAIR] < mavis_twin.inflation_m
    assert not planner._moves("grip", RAIL_BASE_PAIR) and planner._moves("view", RAIL_BASE_PAIR)
    req = PlanRequest(
        q_start={"grip": RAIL_BASE_PINCH_GRIP, "view": RAIL_BASE_PINCH_VIEW},
        q_goal={"grip": MAVIS_GRIP_INIT[:7] + [RAIL_BASE_PINCH_GRIP[7]],
                "view": MAVIS_VIEW_INIT[:7] + [RAIL_BASE_PINCH_VIEW[7]]},
        speed_scale=1.0,
    )
    assert planner._heuristic_order(req, ["grip", "view"]) == ["view", "grip"]
    res = planner.plan(req)
    assert res.ok, (res.failure, res.failing_pair)
    assert res.arm_order == ["view", "grip"]
    segments, ticks = assert_escape_mirrors_gate(mavis_twin, req, res, "view")
    assert segments >= 1 and ticks >= 1
    _assert_waypoints_collision_free_for(mavis_twin, planner, req, res, "grip")
