"""Addressing: rail-first MJCF <-> rail-last core remapping (03-sim §5)."""

from __future__ import annotations

import mujoco
import numpy as np

from apollo_mavis_v2_sim.scenes.builder import HOME_Q


def test_qpos_adr_is_core_order_rail_last(triple_scene):
    model = triple_scene.model
    for arm_id in triple_scene.meta.arm_ids:
        addr = triple_scene.addressing[arm_id]
        rail_slot = model.joint(f"{arm_id}_rail_joint").qposadr[0]
        joint_slots = [model.joint(f"{arm_id}_joint{i}").qposadr[0] for i in range(1, 8)]
        # MJCF-internal: rail is the kinematic ancestor -> FIRST in qpos
        assert rail_slot == min([rail_slot, *joint_slots])
        # core order: joints q[0:7], rail LAST at q[7]
        assert list(addr.qpos_adr[:7]) == joint_slots
        assert addr.qpos_adr[7] == rail_slot


def test_write_core_vector_lands_in_mjcf_slots(triple_scene):
    """Rail-last core vectors write the rail-FIRST MJCF qpos slots correctly."""
    model = triple_scene.model
    data = mujoco.MjData(model)
    addr = triple_scene.addressing["arm1"]
    q_core = np.array([0.1, -0.2, 0.3, 0.9, -0.1, 1.2, 0.05, 0.42])  # rail LAST
    data.qpos[addr.qpos_adr] = q_core
    # MJCF layout check by name: rail slide holds q[7], joint_i holds q[i-1]
    assert data.qpos[model.joint("arm1_rail_joint").qposadr[0]] == 0.42
    for i in range(1, 8):
        assert data.qpos[model.joint(f"arm1_joint{i}").qposadr[0]] == q_core[i - 1]
    # and reading back through the same slices round-trips
    np.testing.assert_array_equal(data.qpos[addr.qpos_adr], q_core)


def test_ctrl_adr_is_core_order(triple_scene):
    model = triple_scene.model
    for arm_id in triple_scene.meta.arm_ids:
        addr = triple_scene.addressing[arm_id]
        assert list(addr.ctrl_adr[:7]) == [
            model.actuator(f"{arm_id}_act{i}").id for i in range(1, 8)
        ]
        assert addr.ctrl_adr[7] == model.actuator(f"{arm_id}_rail").id
        assert addr.gripper_ctrl_adr == model.actuator(f"{arm_id}_gripper").id


def test_rail_slot_moves_base_along_plus_y(triple_scene):
    """Perturb-and-check FK: the rail slot translates link_base along +Y."""
    model = triple_scene.model
    data = mujoco.MjData(model)
    addr = triple_scene.addressing["arm0"]
    q = np.array([*HOME_Q, 0.0])
    data.qpos[addr.qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    base0 = data.xpos[addr.base_body_id].copy()
    q[7] = 0.5
    data.qpos[addr.qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    delta = data.xpos[addr.base_body_id] - base0
    np.testing.assert_allclose(delta, [0.0, 0.5, 0.0], atol=1e-12)


def test_joint_perturbation_moves_only_that_arms_tcp(triple_scene):
    model = triple_scene.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    tcp = {a: data.site_xpos[triple_scene.addressing[a].tcp_site_id].copy()
           for a in triple_scene.meta.arm_ids}
    addr1 = triple_scene.addressing["arm1"]
    q = data.qpos[addr1.qpos_adr].copy()
    q[1] += 0.2  # joint2
    data.qpos[addr1.qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    moved = np.linalg.norm(data.site_xpos[addr1.tcp_site_id] - tcp["arm1"])
    assert moved > 0.01
    for other in ("arm0", "arm2"):
        a = triple_scene.addressing[other]
        np.testing.assert_allclose(data.site_xpos[a.tcp_site_id], tcp[other], atol=1e-12)


def test_geom_partition(triple_scene):
    model = triple_scene.model
    addr = triple_scene.addressing
    all_arm_geoms: set[int] = set()
    for arm_id in triple_scene.meta.arm_ids:
        geoms = set(int(g) for g in addr[arm_id].geom_ids)
        assert geoms.isdisjoint(all_arm_geoms)
        all_arm_geoms |= geoms
    env = set(int(g) for g in addr.env_geom_ids)
    assert env.isdisjoint(all_arm_geoms)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor in env
    assert len(addr.body_of_geom) == model.ngeom
