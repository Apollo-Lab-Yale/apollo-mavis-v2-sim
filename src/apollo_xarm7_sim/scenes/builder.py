"""Runtime scene composition via ``mujoco.MjSpec`` (design 03-sim §5).

One parent spec owns every physics option (children carry none — attach
keeps parent values); each arm child model is attached with prefix
``"<arm_id>_"`` at its base frame. The builder writes ONE merged keyframe
(index 0, ``initial``) before attaching, because attached per-arm
``<arm>_home`` keys zero the other arms. ``BuiltScene.xml`` is
``spec.to_xml()`` with an absolute meshdir — persisted with every episode
and re-compilable via ``MjSpec.from_string`` (phase-07 consumes it).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import mujoco
import numpy as np

from ..assets import asset_path
from ..errors import SceneArmMismatchError, SceneCompileError
from ..gripper import DRIVER_CLOSED_RAD, open_frac_to_ctrl
from .addressing import Addressing
from .descriptor import (
    ArmSpec,
    KeyframeArm,
    SceneDescriptor,
    SceneMeta,
    SceneOverrides,
)

# Menagerie home posture (j1..j7) and rail mid-travel default.
HOME_Q = (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0)
RAIL_HOME_M = 0.325
N_GRIPPER_JOINTS = 6  # driver/follower/spring-link x left/right


@dataclass(frozen=True)
class BuiltScene:
    meta: SceneMeta
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    xml: str  # spec.to_xml() — persisted per episode
    addressing: Addressing


def scene_meta(desc: SceneDescriptor, overrides: SceneOverrides | None = None) -> SceneMeta:
    """Registry row for a (possibly arm-subsetted) scene."""
    arms = _selected_arms(desc, overrides)
    cameras = tuple(c.name for c in desc.cameras) + tuple(
        f"{a.id}_wrist_cam" for a in arms if a.wrist_cam
    )
    return SceneMeta(
        id=desc.id,
        description=desc.description,
        n_arms=len(arms),
        arm_ids=tuple(a.id for a in arms),
        rail={a.id: a.has_rail for a in arms},
        wrist_cams={a.id: a.wrist_cam for a in arms},
        cameras=cameras,
        suitable_for=frozenset(desc.suitable_for),
        allowed_pairs=tuple((str(a), str(b)) for a, b in desc.allowed_pairs),
    )


def build_scene(desc: SceneDescriptor, overrides: SceneOverrides | None = None) -> BuiltScene:
    """Compose and compile a scene; raises the typed scene errors on failure."""
    arms = _selected_arms(desc, overrides)
    meta = scene_meta(desc, overrides)
    spec = mujoco.MjSpec()
    spec.modelname = desc.id
    # Absolute meshdir so spec.to_xml() re-compiles anywhere (episode replay).
    spec.meshdir = str(asset_path())
    _apply_options(spec, desc)
    _add_lights_cameras_environment(spec, desc)
    _add_initial_keyframe(spec, arms, desc.keyframe)  # BEFORE attach -> key index 0
    for arm in arms:
        child = mujoco.MjSpec.from_file(str(asset_path(f"{arm.model}.xml")))
        _customize_child(child, arm)
        pose = (overrides.base_pose.get(arm.id) if overrides else None) or None
        if pose is not None:
            pos, quat = list(pose.position), list(pose.orientation)
        else:
            pos, quat = list(arm.base_pos), list(arm.base_quat)
        frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        with warnings.catch_warnings():
            # Children carry no <option>; MuJoCo still warns about defaults.
            warnings.filterwarnings("ignore", message="Attach conflict")
            spec.attach(child, prefix=f"{arm.id}_", frame=frame)
    try:
        model = spec.compile()
    except (mujoco.FatalError, ValueError) as e:  # mujoco raises ValueError too
        raise SceneCompileError(f"scene {desc.id!r} failed to compile: {e}") from e
    if overrides is not None and overrides.geom_inflation_m is not None:
        _apply_inflation(model, overrides.geom_inflation_m)
    _validate_allowed_pairs(model, desc, arms)
    return BuiltScene(meta, spec, model, spec.to_xml(), Addressing(model, meta))


def _validate_allowed_pairs(
    model: mujoco.MjModel, desc: SceneDescriptor, arms: tuple[ArmSpec, ...]
) -> None:
    """Every scene-authored allowed-pair label must resolve to a pair label
    of the built model (world geom name or ``<arm_id>_<body>``); labels of
    arms dropped by an ``arm_ids`` override are skipped, typos fail loudly."""
    known = {model.body(b).name for b in range(1, model.nbody)}
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name:
                known.add(name)
    dropped = {a.id for a in desc.arms} - {a.id for a in arms}
    for pair in desc.allowed_pairs:
        for label in pair:
            if label in known or any(label.startswith(f"{d}_") for d in dropped):
                continue
            raise SceneCompileError(
                f"scene {desc.id!r}: allowed_pairs label {label!r} matches no world geom "
                "or arm body of the built model"
            )


def _apply_options(spec: mujoco.MjSpec, desc: SceneDescriptor) -> None:
    opt = desc.options
    spec.option.timestep = opt.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = (
        mujoco.mjtCone.mjCONE_ELLIPTIC
        if opt.cone == "elliptic"
        else mujoco.mjtCone.mjCONE_PYRAMIDAL
    )
    spec.option.impratio = opt.impratio
    if not opt.multiccd:  # multiccd is ON by default in 3.12 (disable bit)
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    spec.visual.global_.offwidth = desc.offscreen.width
    spec.visual.global_.offheight = desc.offscreen.height


def _add_lights_cameras_environment(spec: mujoco.MjSpec, desc: SceneDescriptor) -> None:
    spec.worldbody.add_light(
        pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    if not any(e.type == "plane" for e in desc.environment):
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.1],
            rgba=[0.35, 0.35, 0.35, 1.0],
        )
    for env in desc.environment:
        gtype = (
            mujoco.mjtGeom.mjGEOM_PLANE if env.type == "plane" else mujoco.mjtGeom.mjGEOM_BOX
        )
        spec.worldbody.add_geom(
            name=env.name, type=gtype, size=list(env.size), pos=list(env.pos),
            quat=list(env.quat), rgba=list(env.rgba),
        )
    for cam in desc.cameras:
        spec.worldbody.add_camera(
            name=cam.name, pos=list(cam.pos), xyaxes=list(cam.xyaxes), fovy=cam.fovy
        )


def _customize_child(child: mujoco.MjSpec, arm: ArmSpec) -> None:
    """Per-arm child edits: drop the wrist cam and/or gripper subtree."""
    modified = False
    if not arm.wrist_cam:
        child.delete(child.body("d435_mount"))  # no joints -> keyframes unaffected
        modified = True
    if arm.gripper == "none":
        child.delete(child.body("xarm_gripper_base_link"))
        # The TCP site lived on the gripper base; a gripper-less arm's TCP
        # is the link7 flange (phase-02 convention; see report).
        child.body("link7").add_site(name="link_tcp", pos=[0.0, 0.0, 0.0])
        _shrink_child_keyframes(child, arm)
        if arm.wrist_cam:
            # Camera-only arm: the D435 + stand IS the tool. Its mesh is
            # visual-only in the child (it overlaps the gripper hull when
            # both are mounted); with no gripper it must be collidable so
            # the twin protects the real camera body (~8 cm past the flange).
            cam_geom = child.geom("d435")
            cam_geom.contype = 1
            cam_geom.conaffinity = 1
        modified = True
    if modified:
        # Editing leaves keyframes "pending"; compiling the child finalizes
        # them so attach prefixes them correctly (MuJoCo warns otherwise).
        child.compile()


def _shrink_child_keyframes(child: mujoco.MjSpec, arm: ArmSpec) -> None:
    """Resize the child ``home`` key after the gripper joints were deleted."""
    n_q = 8 if arm.has_rail else 7
    n_ctrl = 8 if arm.has_rail else 7
    for key in child.keys:
        key.qpos = list(np.asarray(key.qpos, dtype=np.float64)[:n_q])
        key.ctrl = list(np.asarray(key.ctrl, dtype=np.float64)[:n_ctrl])


def _arm_initial(arm: ArmSpec, kf: KeyframeArm | None) -> tuple[list[float], list[float]]:
    """Per-arm (qpos, ctrl) blocks in MJCF order ([rail?, j1..j7, grip6?])."""
    if kf is not None:
        q_chain = [float(v) for v in kf.q]
        open_frac = kf.gripper
    else:
        q_chain = ([RAIL_HOME_M] if arm.has_rail else []) + list(HOME_Q)
        open_frac = 1.0
    qpos = list(q_chain)
    ctrl = list(q_chain)  # position servos: targets == initial posture
    if arm.gripper == "xarm":
        drv = (1.0 - open_frac) * DRIVER_CLOSED_RAD
        qpos += [drv] * N_GRIPPER_JOINTS
        ctrl += [open_frac_to_ctrl(open_frac)]
    return qpos, ctrl


def _add_initial_keyframe(
    spec: mujoco.MjSpec,
    arms: tuple[ArmSpec, ...],
    keyframe: dict[str, KeyframeArm] | None,
) -> None:
    qpos: list[float] = []
    ctrl: list[float] = []
    for arm in arms:
        kf = keyframe.get(arm.id) if keyframe else None
        q_a, c_a = _arm_initial(arm, kf)
        qpos += q_a
        ctrl += c_a
    spec.add_key(name="initial", qpos=qpos, ctrl=ctrl)


def _apply_inflation(model: mujoco.MjModel, inflation_m: float) -> None:
    """Twin inflation: gap = d/2 per collidable geom, margin 0 (03-sim §8)."""
    collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_gap[collidable] = inflation_m / 2.0
    model.geom_margin[collidable] = 0.0


def _selected_arms(
    desc: SceneDescriptor, overrides: SceneOverrides | None
) -> tuple[ArmSpec, ...]:
    if overrides is None or overrides.arm_ids is None:
        return desc.arms
    known = {a.id for a in desc.arms}
    unknown = [i for i in overrides.arm_ids if i not in known]
    if unknown or not overrides.arm_ids:
        raise SceneArmMismatchError(
            f"scene {desc.id!r} has arms {sorted(known)}, "
            f"override requested {list(overrides.arm_ids)}"
        )
    return tuple(a for a in desc.arms if a.id in set(overrides.arm_ids))


__all__ = ["HOME_Q", "RAIL_HOME_M", "BuiltScene", "scene_meta", "build_scene"]
