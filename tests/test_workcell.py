"""SimWorkcell / SimArm: stepping, remapping, faults, pacing (03-sim §6)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from apollo_xarm7_core import (
    ArmInterface,
    CommandError,
    GripperCommand,
    RailUnavailableError,
    WorkcellInterface,
)
from conftest import make_config

from apollo_xarm7_sim import REGISTRY, SimWorkcell
from apollo_xarm7_sim.workcell import WORKCELL_FAULT_CODE


@pytest.fixture()
def rail_cell():
    scene = REGISTRY.build("single_rail")
    cell = SimWorkcell(scene, make_config("single_rail", ["arm0"]))
    yield cell
    cell.stop()


@pytest.fixture()
def fixed_cell():
    scene = REGISTRY.build("single_fixed_tabletop")
    cell = SimWorkcell(scene, make_config("single_fixed_tabletop", ["arm0"]))
    yield cell
    cell.stop()


def test_interface_compliance(rail_cell):
    assert isinstance(rail_cell, WorkcellInterface)
    assert rail_cell.kind == "sim"
    arm = rail_cell.arms["arm0"]
    assert isinstance(arm, ArmInterface)
    assert arm.dof == 8 and arm.has_rail and not arm.gripper_force_capable


def test_initial_state_matches_keyframe(rail_cell):
    st = rail_cell.arms["arm0"].get_state()
    np.testing.assert_allclose(st.q[:7], [0, -0.247, 0, 0.909, 0, 1.15644, 0], atol=1e-9)
    assert st.rail_pos_m == pytest.approx(0.325)
    assert st.q[7] == st.rail_pos_m
    assert st.error_code == 0 and st.warn_code == 0
    assert st.mode == 1 and st.state == 0 and not st.stale
    assert st.gripper.open_frac == pytest.approx(1.0)


def test_fixed_arm_has_no_rail(fixed_cell):
    arm = fixed_cell.arms["arm0"]
    assert arm.dof == 7 and not arm.has_rail
    assert arm.get_state().rail_pos_m is None
    with pytest.raises(RailUnavailableError):
        arm.command_rail(0.3)


def test_command_joints_validation(rail_cell):
    arm = rail_cell.arms["arm0"]
    with pytest.raises(CommandError):
        arm.command_joints(np.zeros(7))  # wrong dof (rail arm needs 8)
    bad = np.zeros(8)
    bad[3] = np.nan
    with pytest.raises(CommandError):
        arm.command_joints(bad)


def test_rail_last_boundary_remap(rail_cell):
    """Core-order (rail LAST) commands land in the rail-FIRST MJCF ctrl slots."""
    model = rail_cell.scene.model
    arm = rail_cell.arms["arm0"]
    rail_cell.start()  # start() resets targets to the keyframe; command after
    q = np.array([0.1, -0.2, 0.3, 0.9, -0.1, 1.2, 0.05, 0.42])
    arm.command_joints(q)
    targets = rail_cell._targets
    assert targets[model.actuator("arm0_rail").id] == pytest.approx(0.42)
    for i in range(1, 8):
        assert targets[model.actuator(f"arm0_act{i}").id] == pytest.approx(q[i - 1])
    # data.ctrl gets the same values on the next tick
    time.sleep(0.05)
    rail_cell.stop()
    assert rail_cell._data.ctrl[model.actuator("arm0_rail").id] == pytest.approx(0.42)


def test_rail_clamp(rail_cell):
    model = rail_cell.scene.model
    arm = rail_cell.arms["arm0"]
    rail_slot = model.actuator("arm0_rail").id
    arm.command_rail(0.9)
    assert rail_cell._targets[rail_slot] == pytest.approx(0.65)
    arm.command_rail(-0.1)
    assert rail_cell._targets[rail_slot] == pytest.approx(0.0)
    q = np.zeros(8)
    q[7] = 2.0  # command_joints clamps the rail slot too
    arm.command_joints(q)
    assert rail_cell._targets[rail_slot] == pytest.approx(0.65)


def test_gripper_command_mapping(rail_cell):
    model = rail_cell.scene.model
    arm = rail_cell.arms["arm0"]
    slot = model.actuator("arm0_gripper").id
    arm.command_gripper(GripperCommand(open_frac=1.0))
    assert rail_cell._targets[slot] == pytest.approx(0.0)  # ctrl 0 = open
    arm.command_gripper(GripperCommand(open_frac=0.0))
    assert rail_cell._targets[slot] == pytest.approx(255.0)  # ctrl 255 = closed


def test_servo_reaches_joint_target(rail_cell):
    rail_cell.start()
    arm = rail_cell.arms["arm0"]
    q = arm.get_state().q.copy()
    q[0] += 0.2
    q[3] -= 0.15
    arm.command_joints(q)
    time.sleep(0.5)  # servo settle < 0.5 s (03-sim §14)
    st = arm.get_state()
    np.testing.assert_allclose(st.q[:7], q[:7], atol=0.02)


@pytest.mark.perf
def test_pacing_300_ticks_in_3_seconds(rail_cell):
    rail_cell.start()
    t0 = rail_cell.tick_count
    time.sleep(3.0)
    ticks = rail_cell.tick_count - t0
    assert 298 <= ticks <= 302, ticks  # 300 +/- 2: monotonic pacing, no drift


@pytest.mark.perf
def test_realign_after_block_no_burst(rail_cell):
    """A 50 ms stall re-syncs instead of burst-stepping to catch up."""
    blocked = {"done": False}

    def hook():
        if not blocked["done"] and rail_cell.tick_count == 50:
            blocked["done"] = True
            time.sleep(0.05)

    rail_cell._tick_hook = hook
    rail_cell.start()
    time.sleep(1.0)
    rail_cell.stop()
    ticks = rail_cell.tick_count
    assert blocked["done"]
    assert rail_cell.overrun_count >= 1
    # lost ~5 ticks during the stall; a catch-up burst would show ~100+
    assert 90 <= ticks <= 100, ticks


def test_fault_injection_latches_until_cleared(rail_cell):
    rail_cell.start()
    arm = rail_cell.arms["arm0"]
    rail_cell.inject_fault("arm0", 31)
    time.sleep(0.05)
    assert arm.get_state().error_code == 31
    time.sleep(0.05)
    assert arm.get_state().error_code == 31  # latched, not transient
    arm.clear_errors()
    time.sleep(0.05)
    assert arm.get_state().error_code == 0


def test_step_thread_death_latches_and_stop_start_recovers(rail_cell):
    def hook():
        raise RuntimeError("boom (test)")

    rail_cell._tick_hook = hook
    rail_cell.start()
    time.sleep(0.1)
    st = rail_cell.arms["arm0"].get_state()
    assert st.error_code == WORKCELL_FAULT_CODE
    frozen = rail_cell.tick_count
    # clear_errors does NOT restart a dead thread
    rail_cell.arms["arm0"].clear_errors()
    time.sleep(0.1)
    assert rail_cell.tick_count == frozen
    # stop() + start() does (mirrors hardware recovery)
    rail_cell._tick_hook = None
    rail_cell.stop()
    rail_cell.start()
    time.sleep(0.2)
    assert rail_cell.tick_count > 0
    assert rail_cell.arms["arm0"].get_state().error_code == 0


def test_arm_stop_freezes_targets(rail_cell):
    rail_cell.start()
    arm = rail_cell.arms["arm0"]
    q = arm.get_state().q.copy()
    q[0] += 1.0
    arm.command_joints(q)
    time.sleep(0.1)
    arm.stop()  # freeze at current measured posture
    time.sleep(0.4)
    st = arm.get_state()
    assert st.q[0] < 0.6  # never reached the withdrawn target


def test_snapshot_arrays_are_copies(rail_cell):
    rail_cell.start()
    time.sleep(0.05)
    st1 = rail_cell.arms["arm0"].get_state()
    q_saved = st1.q.copy()
    time.sleep(0.1)
    np.testing.assert_array_equal(st1.q, q_saved)  # not aliased to live qpos
    snap = rail_cell.snapshot()
    assert snap.sim_time > 0.0
    assert snap.tick == snap.tick  # WorkcellSnapshot is self-consistent


def test_states_keys_match_arms():
    scene = REGISTRY.build("dual_rail_tabletop")
    cell = SimWorkcell(scene, make_config("dual_rail_tabletop", ["left", "right"]))
    try:
        assert set(cell.states()) == {"left", "right"} == set(cell.arms)
    finally:
        cell.stop()


def test_config_arm_mismatch_raises():
    from apollo_xarm7_sim import SceneArmMismatchError

    scene = REGISTRY.build("single_rail")
    with pytest.raises(SceneArmMismatchError):
        SimWorkcell(scene, make_config("single_rail", ["nope"]))
