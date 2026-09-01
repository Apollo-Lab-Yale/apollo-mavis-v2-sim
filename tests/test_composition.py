"""Scene composition: registry, MjSpec attach, keyframes, round-trip (03-sim §5)."""

from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest
from apollo_xarm7_core import Pose

from apollo_xarm7_sim import REGISTRY, SceneArmMismatchError, SceneNotFoundError, SceneOverrides
from apollo_xarm7_sim.scenes.builder import HOME_Q, RAIL_HOME_M

ALL_SCENES = ["single_fixed_tabletop", "single_rail", "dual_rail_tabletop",
              "triple_rail_row", "dual_mixed"]


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_every_scene_builds(scene_id):
    built = REGISTRY.build(scene_id)
    meta = built.meta
    model = built.model
    # expected sizes: rail arm 14 qpos / 9 ctrl, fixed arm 13 / 8 (with gripper)
    nq = sum(14 if meta.rail[a] else 13 for a in meta.arm_ids)
    nu = sum(9 if meta.rail[a] else 8 for a in meta.arm_ids)
    assert model.nq == nq and model.nu == nu
    # prefixed actuator names, e.g. <arm_id>_act1 ... <arm_id>_gripper
    for arm_id in meta.arm_ids:
        for i in range(1, 8):
            assert model.actuator(f"{arm_id}_act{i}").id >= 0
        assert model.actuator(f"{arm_id}_gripper").id >= 0
        if meta.rail[arm_id]:
            assert model.actuator(f"{arm_id}_rail").id >= 0
        assert model.site(f"{arm_id}_link_tcp").id >= 0
    # cameras exist post-prefix
    for cam in meta.cameras:
        assert model.camera(cam).id >= 0


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_merged_keyframe_is_index_zero(scene_id):
    built = REGISTRY.build(scene_id)
    model = built.model
    assert model.key(0).name == "initial"
    assert model.key(0).qpos.shape == (model.nq,)
    # per-arm <arm_id>_home keys also exist, re-indexed to full nq
    names = [model.key(i).name for i in range(model.nkey)]
    for arm_id in built.meta.arm_ids:
        assert f"{arm_id}_home" in names
        assert model.key(f"{arm_id}_home").qpos.shape == (model.nq,)


def test_initial_keyframe_holds_all_arms_at_home(triple_scene):
    model = triple_scene.model
    key = model.key(0)
    for arm_id in triple_scene.meta.arm_ids:
        addr = triple_scene.addressing[arm_id]
        np.testing.assert_allclose(key.qpos[addr.qpos_adr][:7], HOME_Q, atol=1e-12)
        assert key.qpos[addr.qpos_adr][7] == pytest.approx(RAIL_HOME_M)
        np.testing.assert_allclose(key.ctrl[addr.ctrl_adr][:7], HOME_Q, atol=1e-12)


def test_rail_range_is_direct_0_to_065(triple_scene):
    model = triple_scene.model
    for arm_id in triple_scene.meta.arm_ids:
        rng = model.joint(f"{arm_id}_rail_joint").range
        assert tuple(rng) == (0.0, 0.65)
        act_rng = model.actuator(f"{arm_id}_rail").ctrlrange
        assert tuple(act_rng) == (0.0, 0.65)


def test_to_xml_round_trip_recompiles(triple_scene):
    spec2 = mujoco.MjSpec.from_string(triple_scene.xml)
    model2 = spec2.compile()
    assert model2.nq == triple_scene.model.nq
    assert model2.nu == triple_scene.model.nu
    assert model2.nkey == triple_scene.model.nkey
    assert model2.key(0).name == "initial"


def test_build_emits_no_attach_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any attach-conflict warning -> failure
        REGISTRY.build("dual_rail_tabletop")


def test_unknown_scene_raises():
    with pytest.raises(SceneNotFoundError):
        REGISTRY.build("no_such_scene")


def test_override_arm_subset():
    built = REGISTRY.build("triple_rail_row", SceneOverrides(arm_ids=("arm0", "arm2")))
    assert built.meta.arm_ids == ("arm0", "arm2")
    assert built.model.nq == 28  # 2 railed arms
    with pytest.raises(SceneArmMismatchError):
        REGISTRY.build("triple_rail_row", SceneOverrides(arm_ids=("arm0", "nope")))


def test_override_base_pose():
    pose = Pose(np.array([0.5, -0.2, 0.107188]), np.array([1.0, 0.0, 0.0, 0.0]))
    built = REGISTRY.build("single_rail", SceneOverrides(base_pose={"arm0": pose}))
    data = mujoco.MjData(built.model)
    mujoco.mj_forward(built.model, data)  # qpos 0 -> rail at 0 -> base == rail origin
    np.testing.assert_allclose(data.body("arm0_link_base").xpos, pose.position, atol=1e-12)


def test_override_geom_inflation():
    built = REGISTRY.build("single_rail", SceneOverrides(geom_inflation_m=0.008))
    model = built.model
    collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    np.testing.assert_allclose(model.geom_gap[collidable], 0.004)
    np.testing.assert_allclose(model.geom_margin[collidable], 0.0)


def test_gripper_none_builds_without_gripper():
    from apollo_xarm7_sim import SceneDescriptor
    from apollo_xarm7_sim.scenes.builder import build_scene

    desc = SceneDescriptor(
        id="_nogrip",
        description="railed arm without gripper",
        arms=(
            {"id": "a0", "model": "xarm7_on_rail", "gripper": "none",
             "base_pos": (0.0, -0.325, 0.107188)},
        ),
    )
    built = build_scene(desc)
    model = built.model
    assert model.nq == 8 and model.nu == 8  # rail + 7 joints, no gripper dims
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a0_gripper") < 0
    addr = built.addressing["a0"]
    assert not addr.has_gripper and addr.gripper_ctrl_adr is None
    assert model.site("a0_link_tcp").id == addr.tcp_site_id  # flange TCP fallback
    assert model.key(0).name == "initial" and model.key(0).qpos.shape == (8,)
    mujoco.MjSpec.from_string(built.xml).compile()  # round-trip still holds


def test_wrist_cam_false_removes_camera_and_mount():
    built = REGISTRY.build("dual_mixed")
    model = built.model
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fix0_wrist_cam") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fix0_d435_mount") < 0
    assert model.camera("rail0_wrist_cam").id >= 0
    assert built.addressing["fix0"].wrist_cam_id is None
