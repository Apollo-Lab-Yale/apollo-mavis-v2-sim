"""ResetPlanner tests (03-sim §10/§14.2): swap, detour, failures, hysteresis."""

from __future__ import annotations

import time

import numpy as np
import pytest
from apollo_mavis_v2_core import PlanRequest

from apollo_mavis_v2_sim import REGISTRY, DigitalTwin, PlannerParams, ResetPlanner
from apollo_mavis_v2_sim.planner import RAIL_MAX_STEP_M, time_parameterize

HOME_J = [0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0]

# Verified on guardrail_rail: yawed-toward-each-other arms; min monitored
# cross-arm clearance ~34 mm at rail 0.24 and ~14 mm at rail 0.26.
CLOSE_START_ARM0 = [1.35] + HOME_J[1:] + [0.24]
INSIDE_BAND_ARM0 = [1.35] + HOME_J[1:] + [0.265]  # ~4 mm < delta = 8 mm
PENETRATING_ARM0 = [1.35] + HOME_J[1:] + [0.30]  # link6-link6 raw overlap
START_ARM1 = [-1.35] + HOME_J[1:] + [0.48]
GOAL_ARM0 = HOME_J + [0.05]
GOAL_ARM1 = HOME_J + [0.45]
# Fingers/platform over arm1's static rail bed: in collision in ANY ordering.
BAD_GOAL_ARM0 = [1.5, 0.6, 0.0, 0.4, 0.0, 1.6, 0.0, 0.45]

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


def _assert_waypoints_collision_free(twin: DigitalTwin, req: PlanRequest, res) -> None:
    """Every waypoint AND every interpolated edge sample passes check_config,
    in the sequential execution context (earlier arms advance to goal)."""
    ctx = np.array(twin._q_meas_full)
    for arm_id, q in req.q_start.items():
        ctx[twin.addr[arm_id].qpos_adr] = q
    for arm_id, wps in res.waypoints.items():
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
    _assert_waypoints_collision_free(rail_twin, req, res)
    for arm_id in ("arm0", "arm1"):
        assert np.allclose(res.waypoints[arm_id][0], req.q_start[arm_id])
        assert np.allclose(res.waypoints[arm_id][-1], req.q_goal[arm_id])


def test_start_inside_inflation_shell_escapes(rail_twin):
    """Hysteresis: a start parked inside the 8 mm shell can still plan out."""
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


def test_pedestal_detour_within_budget(env_twin):
    planner = ResetPlanner(env_twin)
    req = PlanRequest(q_start={"arm0": DETOUR_START}, q_goal={"arm0": DETOUR_GOAL})
    t0 = time.monotonic()
    res = planner.plan(req)
    wall = time.monotonic() - t0
    assert res.ok, res.failure
    assert wall < req.timeout_s  # single-arm wall time < 5 s
    assert len(res.waypoints["arm0"]) >= 3  # a straight edge is blocked
    _assert_waypoints_collision_free(env_twin, req, res)


def test_starved_planner_times_out(env_twin):
    planner = ResetPlanner(env_twin, params=PlannerParams(max_iters=3))
    res = planner.plan(
        PlanRequest(q_start={"arm0": DETOUR_START}, q_goal={"arm0": DETOUR_GOAL})
    )
    assert not res.ok and res.failure == "timeout"


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


def test_time_parameterize_monotone_and_capped():
    wps = [[0.0] * 8, [0.3] * 7 + [0.05], [0.6] * 7 + [0.10]]
    timed = time_parameterize(wps, has_rail=True)
    ts = [t for t, _ in timed]
    assert ts[0] == 0.0 and all(b > a for a, b in zip(ts, ts[1:], strict=False))
    p = PlannerParams()
    # rail dim is the slowest here: 0.05 m at 0.1 m/s -> >= 0.5 s per segment
    assert ts[1] >= 0.05 / p.rail_vel_m_s - 1e-9
