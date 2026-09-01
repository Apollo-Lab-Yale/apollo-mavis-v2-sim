"""MinkIKSolver tests (03-sim §9/§14): tracking, refinements, divergence."""

from __future__ import annotations

import numpy as np
import pytest
from apollo_xarm7_core import IKSolver, Pose, se3

from apollo_xarm7_sim import (
    REGISTRY,
    DigitalTwin,
    IKParams,
    IKUnreachableError,
    MinkIKSolver,
    default_collision_pairs,
)

HOME = np.array([0, -0.247, 0, 0.909, 0, 1.15644, 0, 0.325])


@pytest.fixture()
def solver() -> MinkIKSolver:
    return MinkIKSolver(REGISTRY.build("single_rail"), IKParams(), collision_pairs=None)


def _tcp(solver: MinkIKSolver, arm_id: str = "arm0") -> tuple[np.ndarray, np.ndarray]:
    a = solver.scene.addressing[arm_id]
    d = solver.configuration.data
    return (
        d.site_xpos[a.tcp_site_id].copy(),
        se3.mat_to_quat(d.site_xmat[a.tcp_site_id].reshape(3, 3)),
    )


def test_implements_core_protocol(solver):
    assert isinstance(solver, IKSolver)


def test_circle_tracking_500_ticks(solver):
    pos0, quat0 = _tcp(solver)
    lo, hi = solver._jnt_range["arm0"]
    for k in range(500):
        t = k * 0.01
        target = Pose(
            pos0 + [0.05 * np.sin(1.26 * t), 0.05 * (1 - np.cos(1.26 * t)), 0.0], quat0
        )
        r = solver.solve("arm0", target, None)
        assert np.all(r.q >= lo - 1e-9) and np.all(r.q <= hi + 1e-9)
    assert r.pos_err_m < 5e-4  # < 0.5 mm after warm-up
    assert not r.diverged


def test_rail_preference_lateral_target(solver):
    """A small lateral (rail-direction) move should use joints, not the rail."""
    pos0, quat0 = _tcp(solver)
    target = Pose(pos0 + [0.0, 0.03, 0.0], quat0)
    for _ in range(300):
        r = solver.solve("arm0", target, None)
    assert r.pos_err_m < 5e-4
    assert abs(r.q[7] - HOME[7]) < 0.01  # rail absorbed < 1 cm of a 3 cm move


def test_unreachable_sets_diverged_without_raising(solver):
    pos0, quat0 = _tcp(solver)
    bad = Pose(pos0 + [2.0, 0.0, 0.0], quat0)
    results = [solver.solve("arm0", bad, None) for _ in range(12)]
    assert not results[8].diverged  # not before 10 consecutive ticks
    assert results[9].diverged or results[10].diverged
    assert all(np.all(np.isfinite(r.q)) for r in results)


def test_ecaa_weight_slew_and_floor(solver):
    base = solver.params.orientation_cost
    pos0, quat0 = _tcp(solver)
    target = Pose(pos0, quat0)
    solver.set_min_clearance(0.0)  # deep in the warn band -> relax orientation
    weights = []
    for _ in range(60):
        solver.solve("arm0", target, None)
        weights.append(solver._w_orient["arm0"])
    diffs = np.diff([base] + weights)
    assert np.all(diffs <= 1e-12)  # monotone decreasing toward the floor
    assert np.all(np.abs(diffs) <= 0.02 * base + 1e-12)  # <= 2 %/tick slew
    assert weights[-1] == pytest.approx(0.10 * base)  # 10 % floor
    solver.set_min_clearance(1.0)  # clear -> recover toward base, rate-limited
    up = []
    for _ in range(60):
        solver.solve("arm0", target, None)
        up.append(solver._w_orient["arm0"])
    dup = np.diff([weights[-1]] + up)
    assert np.all(dup >= -1e-12) and np.all(dup <= 0.02 * base + 1e-12)
    assert up[-1] == pytest.approx(base)


def test_flat_tolerance_frees_tool_roll(solver):
    pos0, quat0 = _tcp(solver)
    # roll target: rotate 0.8 rad about the TCP's local z (tool roll)
    z_local = se3.quat_rotate(quat0, np.array([0.0, 0.0, 1.0]))
    quat_roll = se3.quat_mul(se3.rotvec_to_quat(0.8 * z_local), quat0)
    target = Pose(pos0, quat_roll)
    q7_start = solver._q_warm["arm0"][6]
    solver.set_flat_tolerances("arm0", np.array([0, 0, 0, 0, 0, np.pi]))
    for _ in range(50):
        r = solver.solve("arm0", target, None)
    assert abs(solver._q_warm["arm0"][6] - q7_start) < 0.05  # roll NOT chased
    assert r.rot_err_rad < 1e-3  # shrunk error reports "within tolerance"
    solver.set_flat_tolerances("arm0", np.zeros(6))
    for _ in range(80):
        r = solver.solve("arm0", target, None)
    assert abs(solver._q_warm["arm0"][6] - q7_start) > 0.4  # now it rolls


def test_collision_row_cap():
    scene = REGISTRY.build("guardrail_env")
    twin = DigitalTwin(REGISTRY.build("guardrail_env"), inflation_m=0.008)
    solver = MinkIKSolver(
        twin.scene,
        IKParams(max_collision_rows=12),
        collision_pairs=default_collision_pairs(twin.scene, twin.allowed),
    )
    del scene
    q = HOME.copy()
    q[1] += 0.85  # fingers a few cm above the table: many candidate pairs
    solver.reset("arm0", q)
    pos0, quat0 = _tcp(solver)
    r = solver.solve("arm0", Pose(pos0 + [0.0, 0.0, -0.01], quat0), None)
    assert 1 <= r.active_collision_rows <= 12


def test_solve_to_convergence_places_rail(solver):
    pos0, quat0 = _tcp(solver)
    r = solver.solve_to_convergence("arm0", Pose(pos0 + [0.0, 0.45, 0.05], quat0), HOME)
    assert r.pos_err_m < 1e-3 and r.rot_err_rad < 0.01
    assert r.q[7] > HOME[7] + 0.05  # rail placed automatically
    # servo warm state untouched by the one-shot query
    assert np.allclose(solver._q_warm["arm0"], HOME)


def test_solve_to_convergence_unreachable_raises(solver):
    pos0, quat0 = _tcp(solver)
    with pytest.raises(IKUnreachableError) as ei:
        solver.solve_to_convergence("arm0", Pose(pos0 + [3.0, 0.0, 0.0], quat0), HOME)
    assert ei.value.best_result.pos_err_m > 1e-3


def test_reset_clears_history_and_reseeds(solver):
    pos0, quat0 = _tcp(solver)
    for _ in range(5):
        solver.solve("arm0", Pose(pos0 + [0.02, 0.0, 0.0], quat0), None)
    assert np.any(solver._dq_hist["arm0"][0] != 0.0)
    q_new = HOME + 0.1
    solver.reset("arm0", q_new)
    assert np.allclose(solver._q_warm["arm0"], q_new)
    assert not np.any(solver._dq_hist["arm0"][0])
    assert not np.any(solver._dq_hist["arm0"][1])
    assert solver._streak["arm0"] == 0


def test_far_qseed_triggers_reseed(solver):
    pos0, quat0 = _tcp(solver)
    q_seed = HOME.copy()
    q_seed[0] += 0.5  # far from the warm state -> external motion
    r = solver.solve("arm0", Pose(pos0, quat0), q_seed)
    assert np.max(np.abs(r.q - q_seed)) < 0.1  # solved FROM the seed
