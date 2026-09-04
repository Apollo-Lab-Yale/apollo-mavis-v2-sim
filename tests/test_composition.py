"""Scene composition: registry, MjSpec attach, keyframes, round-trip (03-sim §5)."""

from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest
from apollo_mavis_v2_core import Pose

from apollo_mavis_v2_sim import REGISTRY, SceneArmMismatchError, SceneNotFoundError, SceneOverrides
from apollo_mavis_v2_sim.scenes import ArmSpec, SceneView, build_scene
from apollo_mavis_v2_sim.scenes.builder import (
    HOME_Q,
    MIC_MASS_KG,
    MIC_RADIUS_M,
    MIC_TIP_Z_M,
    RAIL_HOME_M,
    WRIST_CAM_Z_M,
)

ALL_SCENES = ["single_fixed_tabletop", "single_rail", "dual_rail_tabletop",
              "triple_rail_row", "dual_mixed", "mavis_v2"]


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_every_scene_builds(scene_id):
    built = REGISTRY.build(scene_id)
    meta = built.meta
    model = built.model
    # expected sizes: rail arm 8 qpos / 8 ctrl, fixed arm 7 / 7, plus 6 qpos / 1 ctrl
    # for an xArm gripper (mavis_v2's camera-only arm has none)
    grip = {a: built.addressing[a].has_gripper for a in meta.arm_ids}
    nq = sum((8 if meta.rail[a] else 7) + (6 if grip[a] else 0) for a in meta.arm_ids)
    nu = sum((8 if meta.rail[a] else 7) + (1 if grip[a] else 0) for a in meta.arm_ids)
    assert model.nq == nq and model.nu == nu
    # prefixed actuator names, e.g. <arm_id>_act1 ... <arm_id>_gripper
    for arm_id in meta.arm_ids:
        for i in range(1, 8):
            assert model.actuator(f"{arm_id}_act{i}").id >= 0
        if grip[arm_id]:
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
    from apollo_mavis_v2_sim import SceneDescriptor
    from apollo_mavis_v2_sim.scenes.builder import build_scene

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


def test_allowed_pairs_unknown_label_fails_the_build():
    from apollo_mavis_v2_sim import SceneCompileError, SceneDescriptor
    from apollo_mavis_v2_sim.scenes.builder import build_scene

    desc = SceneDescriptor(
        id="_badpair",
        description="typo in a scene-authored structural pair",
        arms=({"id": "a0", "model": "xarm7_on_rail", "base_pos": (0.0, -0.325, 0.107188)},),
        allowed_pairs=(("a0_rail_platform", "tabel"),),
    )
    with pytest.raises(SceneCompileError, match="tabel"):
        build_scene(desc)


def test_allowed_pairs_of_dropped_arms_are_skipped_by_arm_subset():
    from apollo_mavis_v2_sim import SceneDescriptor
    from apollo_mavis_v2_sim.scenes.builder import build_scene

    desc = SceneDescriptor(
        id="_subset",
        description="pair naming an arm that the override drops",
        arms=(
            {"id": "a0", "model": "xarm7_on_rail", "base_pos": (0.0, -0.325, 0.107188)},
            {"id": "b0", "model": "xarm7_on_rail", "base_pos": (1.0, -0.325, 0.107188)},
        ),
        allowed_pairs=(("a0_rail_platform", "b0_rail_base"),),
    )
    built = build_scene(desc, SceneOverrides(arm_ids=("a0",)))
    assert built.meta.arm_ids == ("a0",)


def test_arm_id_prefix_of_another_arm_is_rejected():
    from apollo_mavis_v2_sim import SceneDescriptor

    with pytest.raises(ValueError, match="prefix"):
        SceneDescriptor(
            id="_prefix",
            description="cam / cam2 would make pair labels ambiguous",
            arms=(
                {"id": "cam", "model": "xarm7_on_rail"},
                {"id": "cam_2", "model": "xarm7_on_rail", "base_pos": (1.0, 0.0, 0.107188)},
            ),
        )


def test_view_field_sets_the_free_camera_defaults():
    """``view`` -> ``<visual><global azimuth elevation>``; absent -> MuJoCo defaults untouched."""
    assert SceneView() == SceneView(azimuth=90.0, elevation=-45.0)  # MuJoCo's own defaults
    base = REGISTRY.descriptor("single_rail")
    assert base.view is None
    g = REGISTRY.build("single_rail").model.vis.global_
    assert (g.azimuth, g.elevation) == (90.0, -45.0)
    desc = base.model_copy(update={"view": SceneView(azimuth=-90.0, elevation=-30.0)})
    built = build_scene(desc)
    g = built.model.vis.global_
    assert (g.azimuth, g.elevation) == (-90.0, -30.0)
    g2 = mujoco.MjModel.from_xml_string(built.xml).vis.global_  # persisted with the episode XML
    assert (g2.azimuth, g2.elevation) == (-90.0, -30.0)


# -- microphone body (03-sim §3) ---------------------------------------------

CAM_ONLY_ARM = {
    "id": "a0", "model": "xarm7_on_rail", "gripper": "none", "wrist_cam": True,
    "base_pos": (0.0, -0.325, 0.107188),
}


def _cam_only_desc(**arm_extra):
    from apollo_mavis_v2_sim import SceneDescriptor

    return SceneDescriptor(
        id="_mic", description="camera-only railed arm", arms=({**CAM_ONLY_ARM, **arm_extra},)
    )


def test_microphone_flag_adds_welded_cylinder_under_link7():
    built = build_scene(_cam_only_desc(microphone=True))
    model = built.model
    assert built.meta.microphones == {"a0": True}
    body = model.body("a0_microphone")
    assert model.body(body.parentid[0]).name == "a0_link7"
    assert body.jntnum[0] == 0 and body.weldid[0] == model.body("a0_link7").weldid[0]
    assert body.mass[0] == pytest.approx(MIC_MASS_KG)  # explicit, not density-derived
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body.id]
    assert len(geoms) == 1
    g = model.geom("a0_microphone")
    assert g.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert MIC_RADIUS_M == 0.040 and MIC_TIP_Z_M == pytest.approx(WRIST_CAM_Z_M + 0.14)
    np.testing.assert_allclose(g.size[:2], [MIC_RADIUS_M, MIC_TIP_Z_M / 2])  # [0.040, 0.095]
    np.testing.assert_allclose(g.pos, [0.0, 0.0, MIC_TIP_Z_M / 2])  # z in [0, 0.19] in link7
    np.testing.assert_allclose(g.quat, [1.0, 0.0, 0.0, 0.0])  # axis = link7 +z (flange)
    assert g.contype[0] == 1 and g.conaffinity[0] == 1 and g.group[0] == 0
    assert g.rgba[3] == 1.0 and max(g.rgba[:3]) < 0.25  # dark housing
    # joint-less: sizes, keyframes and addressing unchanged vs the mic-less arm
    plain = build_scene(_cam_only_desc()).model
    assert (model.nq, model.nu, model.nkey) == (plain.nq, plain.nu, plain.nkey) == (8, 8, 2)
    np.testing.assert_allclose(model.key(0).qpos, plain.key(0).qpos)
    assert model.nbody == plain.nbody + 1 and model.ngeom == plain.ngeom + 1
    assert g.id in set(built.addressing["a0"].geom_ids.tolist())  # subtree scan picks it up
    assert mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_BODY, "a0_microphone") < 0
    # persisted XML round-trips with the mic
    model2 = mujoco.MjSpec.from_string(built.xml).compile()
    assert model2.body("a0_microphone").id >= 0 and (model2.nq, model2.nu) == (8, 8)
    np.testing.assert_allclose(model2.geom("a0_microphone").size[:2], g.size[:2])


def test_microphone_flag_works_on_the_fixed_child_model_too():
    built = build_scene(_cam_only_desc(microphone=True, model="xarm7_fixed", base_pos=(0, 0, 0)))
    assert built.model.body("a0_microphone").id >= 0 and built.model.nq == 7


@pytest.mark.parametrize(
    "bad", [{"gripper": "xarm"}, {"wrist_cam": False}, {"gripper": "xarm", "wrist_cam": False}]
)
def test_microphone_requires_camera_only_arm(bad):
    with pytest.raises(ValueError, match="microphone requires wrist_cam: true and gripper: none"):
        ArmSpec(**{**CAM_ONLY_ARM, "microphone": True, **bad})
    with pytest.raises(ValueError, match="microphone"):
        _cam_only_desc(microphone=True, **bad)


def test_microphone_override_toggles_the_same_scene():
    from apollo_mavis_v2_sim import SceneCompileError

    plain = REGISTRY.build("mavis_v2")
    assert plain.meta.microphones == {"view": False, "grip": False}
    assert mujoco.mj_name2id(plain.model, mujoco.mjtObj.mjOBJ_BODY, "view_microphone") < 0
    mic = REGISTRY.build("mavis_v2", SceneOverrides(microphones={"view": True}))
    assert mic.meta.microphones == {"view": True, "grip": False}
    assert mic.model.body("view_microphone").id >= 0
    assert (mic.model.nq, mic.model.nu) == (plain.model.nq, plain.model.nu) == (22, 17)
    assert mic.meta.cameras == plain.meta.cameras and mic.meta.arm_ids == plain.meta.arm_ids
    assert REGISTRY.meta("mavis_v2").microphones == {"view": False, "grip": False}  # unchanged
    # explicit off, unknown arm, and a mic on the gripper arm
    off = REGISTRY.build("mavis_v2", SceneOverrides(microphones={"view": False}))
    assert off.meta.microphones == {"view": False, "grip": False}
    with pytest.raises(SceneArmMismatchError, match="nope"):
        REGISTRY.build("mavis_v2", SceneOverrides(microphones={"nope": True}))
    with pytest.raises(SceneCompileError, match="microphone requires"):
        REGISTRY.build("mavis_v2", SceneOverrides(microphones={"grip": True}))
    # composes with an arm subset: the override may name a dropped arm's mate only
    sub = REGISTRY.build(
        "mavis_v2", SceneOverrides(arm_ids=("view",), microphones={"view": True})
    )
    assert sub.meta.microphones == {"view": True} and sub.model.body("view_microphone").id >= 0


def test_allowed_pairs_may_name_a_microphone_that_is_switched_off():
    from apollo_mavis_v2_sim import SceneDescriptor

    desc = SceneDescriptor(
        id="_micpair",
        description="structural pair naming the optional mic",
        arms=({**CAM_ONLY_ARM},),
        environment=(
            {"name": "table", "type": "box", "size": (0.3, 0.3, 0.01), "pos": (0.5, 0, 0.3)},
        ),
        allowed_pairs=(("a0_microphone", "table"),),
    )
    build_scene(desc)  # mic off: label skipped like a dropped arm's
    built = build_scene(desc, SceneOverrides(microphones={"a0": True}))
    assert built.meta.allowed_pairs == (("a0_microphone", "table"),)

