"""DigitalTwin tests (03-sim §8/§14): excludes, audit, thresholds, whitelist."""

from __future__ import annotations

import time

import mujoco
import numpy as np
import pytest
from apollo_xarm7_core import ArmState, DigitalTwinInterface, GripperState, Pose

from apollo_xarm7_sim import REGISTRY, DigitalTwin, TwinAuditError
from apollo_xarm7_sim.scenes.builder import build_scene
from apollo_xarm7_sim.scenes.descriptor import ArmSpec, EnvironmentSpec, SceneDescriptor

HOME = np.array([0, -0.247, 0, 0.909, 0, 1.15644, 0, 0.325])


@pytest.fixture(scope="module")
def twin() -> DigitalTwin:
    return DigitalTwin(REGISTRY.build("guardrail_env"), inflation_m=0.008)


def _q_desc(twin: DigitalTwin, j2: float) -> np.ndarray:
    q = HOME.copy()
    q[1] += j2  # tip forward/down toward the table
    return q


def _min_monitored(twin: DigitalTwin, q: np.ndarray) -> tuple[float, tuple[str, str]]:
    twin.data.qpos[twin.addr["arm0"].qpos_adr] = q
    mujoco.mj_kinematics(twin.model, twin.data)
    best, pair = 1.0, ("", "")
    for g1, g2 in twin.monitored_pairs:
        d = mujoco.mj_geomDistance(twin.model, twin.data, g1, g2, best, None)
        if d < best:
            best = d
            pair = (twin.allowed.label_of_geom(g1), twin.allowed.label_of_geom(g2))
    twin.data.qpos[:] = twin._q_meas_full
    return best, pair


def test_implements_core_protocol(twin):
    assert isinstance(twin, DigitalTwinInterface)


def test_link_base_link1_excluded_per_arm(twin):
    for arm_id in twin.addr.arms:
        assert twin.allowed.allows_labels(f"{arm_id}_link_base", f"{arm_id}_link1")


def test_audit_rejects_bad_scene():
    desc = SceneDescriptor(
        id="bad_cell",
        description="box inside the arm's home volume",
        arms=(ArmSpec(id="arm0", model="xarm7_fixed", wrist_cam=False),),
        environment=(
            EnvironmentSpec(
                name="intruder", type="box", size=(0.08, 0.08, 0.08), pos=(0.4, 0.0, 0.2)
            ),
        ),
    )
    with pytest.raises(TwinAuditError, match="intruder"):
        DigitalTwin(build_scene(desc), inflation_m=0.025)


@pytest.mark.parametrize(
    "scene_id",
    ["single_rail", "dual_rail_tabletop", "guardrail_face", "guardrail_rail"],
)
def test_registry_scenes_audit_clean_at_debug_delta(scene_id):
    DigitalTwin(REGISTRY.build(scene_id), inflation_m=0.025)  # no TwinAuditError


def test_no_false_alarms_near_home(twin):
    rng = np.random.default_rng(3)
    for _ in range(4):
        q = HOME + np.concatenate([rng.uniform(-0.2, 0.2, 7), rng.uniform(-0.1, 0.1, 1)])
        full = np.array(twin._q_meas_full)
        full[twin.addr["arm0"].qpos_adr] = q
        assert twin.check_config(full)


def test_detection_appears_at_delta_not_before(twin):
    """Two-body approach: contact detected below δ total, none above 1.1δ."""
    delta = twin.inflation_m
    dists = {}
    for j2 in np.arange(0.0, 1.2, 0.005):
        q = _q_desc(twin, j2)
        d, pair = _min_monitored(twin, q)
        if "table" in pair:
            dists[j2] = d
    above = next(j2 for j2, d in dists.items() if 1.1 * delta < d < 3 * delta)
    below = next(j2 for j2, d in dists.items() if 0.1 * delta < d < 0.9 * delta)
    rep_above = twin.check({"arm0": _q_desc(twin, above)})
    assert not rep_above.blocked
    rep_below = twin.check({"arm0": _q_desc(twin, below)})
    assert rep_below.blocked and rep_below.severity == "blocked"
    assert any("table" in p for p in rep_below.pairs)
    assert 0.0 < rep_below.min_clearance_m < delta


def test_check_is_stateless(twin):
    before = np.array(twin.data.qpos)
    twin.check({"arm0": _q_desc(twin, 0.9)})
    assert np.array_equal(twin.data.qpos, before)


def test_sync_writes_measured(twin):
    q = HOME.copy()
    q[0] = 0.3
    state = ArmState(
        arm_id="arm0", q=q, dq=np.zeros(8), ee_pose=Pose.identity(),
        gripper=GripperState(open_frac=1.0), rail_pos_m=float(q[7]),
        error_code=0, warn_code=0, mode=1, state=0, stale=False,
        t_mono=time.monotonic(), wallclock_ns=time.time_ns(),
    )
    twin.sync({"arm0": state})
    assert np.allclose(twin.data.qpos[twin.addr["arm0"].qpos_adr], q)
    twin.sync(
        {
            "arm0": ArmState(
                arm_id="arm0", q=HOME, dq=np.zeros(8), ee_pose=Pose.identity(),
                gripper=GripperState(open_frac=1.0), rail_pos_m=float(HOME[7]),
                error_code=0, warn_code=0, mode=1, state=0, stale=False,
                t_mono=time.monotonic(), wallclock_ns=time.time_ns(),
            )
        }
    )


def test_grasp_whitelist_suppresses_finger_pairs(twin):
    # depth where ONLY gripper bodies are inside the band vs the table
    for j2 in np.arange(0.3, 1.2, 0.005):
        rep = twin.check({"arm0": _q_desc(twin, j2)})
        if rep.blocked:
            break
    assert rep.blocked
    gripper_labels = set(twin._gripper_labels["arm0"])
    assert all(
        (set(p) - {"table"}) <= gripper_labels for p in rep.pairs
    ), f"non-gripper pairs at first contact: {rep.pairs}"
    twin.set_grasp_whitelist("arm0", ["table"])
    assert not twin.check({"arm0": _q_desc(twin, j2)}).blocked
    twin.set_grasp_whitelist("arm0", [])
    assert twin.check({"arm0": _q_desc(twin, j2)}).blocked


def test_clearance_sorted_and_recomputable(twin):
    rows = twin.clearance(0.2)
    assert rows, "expected clearance rows within 0.2 m"
    dists = [r.dist_m for r in rows]
    assert dists == sorted(dists)
    monitored = set(twin.monitored_pairs)

    def gid(name: str) -> int:
        try:
            return int(twin.model.geom(name).id)
        except KeyError:  # unnamed geom: PairClearance labels it "geom<id>"
            return int(name.removeprefix("geom"))

    for r in rows[:5]:
        g1 = gid(r.geom1)
        g2 = gid(r.geom2)
        assert (min(g1, g2), max(g1, g2)) in monitored
        mujoco.mj_kinematics(twin.model, twin.data)
        d = mujoco.mj_geomDistance(twin.model, twin.data, g1, g2, 0.2, None)
        assert d == pytest.approx(r.dist_m, abs=1e-9)
        assert r.fromto is not None and r.fromto.shape == (6,)
