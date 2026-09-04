"""Gripper direction and mapping — measured, never trusted from memory (03-sim §6)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from apollo_mavis_v2_sim.gripper import (
    ctrl_to_open_frac,
    driver_q_to_open_frac,
    open_frac_to_ctrl,
    open_frac_to_meters,
)


def _fingertip_gap(scene, ctrl_value: float) -> float:
    """Settle the gripper at a ctrl value and measure the finger-pad gap."""
    model, addr = scene.model, scene.addressing["arm0"]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[addr.gripper_ctrl_adr] = ctrl_value
    mujoco.mj_step(model, data, nstep=1000)  # 2 s: plenty to settle
    left = data.geom_xpos[model.geom("arm0_left_finger_pad_1").id]
    right = data.geom_xpos[model.geom("arm0_right_finger_pad_1").id]
    return float(np.linalg.norm(left - right))


def test_direction_ctrl0_open_ctrl255_closed(single_fixed_scene):
    gap_open = _fingertip_gap(single_fixed_scene, 0.0)
    gap_closed = _fingertip_gap(single_fixed_scene, 255.0)
    assert gap_open > gap_closed + 0.03, (gap_open, gap_closed)
    assert gap_open > 0.06  # near the 0.085 m span (pads sit inside the tips)


def test_gap_monotone_in_ctrl(single_fixed_scene):
    gaps = [_fingertip_gap(single_fixed_scene, c) for c in (0.0, 85.0, 170.0, 255.0)]
    assert all(a > b for a, b in zip(gaps, gaps[1:], strict=False)), gaps


def test_open_frac_to_ctrl_endpoints():
    assert open_frac_to_ctrl(1.0) == 0.0
    assert open_frac_to_ctrl(0.0) == 255.0
    assert open_frac_to_ctrl(0.5) == pytest.approx(127.5)
    # clamped outside [0, 1]
    assert open_frac_to_ctrl(1.5) == 0.0
    assert open_frac_to_ctrl(-0.5) == 255.0


def test_round_trips():
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert ctrl_to_open_frac(open_frac_to_ctrl(f)) == pytest.approx(f)
    assert driver_q_to_open_frac(0.0) == 1.0
    assert driver_q_to_open_frac(0.85) == 0.0
    assert open_frac_to_meters(1.0) == pytest.approx(0.085)
    assert open_frac_to_meters(0.0) == 0.0


def test_open_frac_matches_driver_equilibrium(single_fixed_scene):
    """ctrl -> settled driver angle -> open_frac round-trips approximately."""
    model, addr = single_fixed_scene.model, single_fixed_scene.addressing["arm0"]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[addr.gripper_ctrl_adr] = open_frac_to_ctrl(0.5)
    mujoco.mj_step(model, data, nstep=1500)
    measured = driver_q_to_open_frac(float(data.qpos[addr.gripper_driver_qpos_adr]))
    assert measured == pytest.approx(0.5, abs=0.15)  # soft servo: loose tolerance
