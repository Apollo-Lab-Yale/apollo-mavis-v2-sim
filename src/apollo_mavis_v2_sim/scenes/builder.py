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
from pydantic import ValidationError

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

# Microphone body (RODE NT-USB Mini + bracket) on a camera-only arm, link7 frame
# (03-sim §3). The link7 origin IS the flange face and +z the flange/tool axis.
# WRIST CAMERA POSE IN link7 -- MEASURED 2026-09-06 (03-sim 4.3, "wrist camera
# extrinsic"). It had been the reference model's guess (0.07, 0, 0.05), which put the
# twin's optical centre 17.1 mm sideways and 19.8 mm too far from the flange face:
# that single error was the whole visible overlay offset the operator reported, and it
# is why the twin's table-plane image scale was 2.7 % too large. Solved from four
# hand-drawn dots on the table whose world positions were tape-measured, with the
# camera ROTATION left at the model's value (independently confirmed: the observed
# perspective convergence of a 195 x 96 mm rectangle matches the twin's to 1.0226 vs
# 1.0226). Reprojection RMS 1.75 px over the four dots; the rail's own end face,
# which was NOT used in the fit, lands 1.3 px from where the frame shows it.
WRIST_CAM_POS_M = (0.06832, -0.02220, 0.02945)
# The MICROPHONE body is deliberately NOT tied to the camera pose. It is a physical
# part on the bracket: re-measuring where the lens's optical centre sits does not move
# it. Its length was originally referenced to the assumed camera plane z = 0.05, so
# that number lives on here as the mic's own reference plane. The mic geometry is
# STILL UNVERIFIED -- the real view_wrist image shows no occlusion where the twin
# shows ~12 % -- and needs its own measurement of the mount (CLAUDE.md).
MIC_REF_PLANE_Z_M = 0.05
MIC_AHEAD_OF_CAM_M = 0.14  # mic tip beyond that reference plane
MIC_RADIUS_M = 0.040
MIC_TIP_Z_M = MIC_REF_PLANE_Z_M + MIC_AHEAD_OF_CAM_M  # 0.19
MIC_HALF_LENGTH_M = MIC_TIP_Z_M / 2.0  # MuJoCo cylinder size = [radius, half-length]
WRIST_CAM_Z_M = MIC_REF_PLANE_Z_M  # deprecated alias, kept for the mic's own tests
MIC_MASS_KG = 0.45  # NT-USB Mini (~0.35 kg) + bracket -- estimate, to be weighed
MIC_RGBA = (0.12, 0.12, 0.13, 1.0)  # dark housing


@dataclass(frozen=True)
class BuiltScene:
    meta: SceneMeta
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    xml: str  # spec.to_xml() — persisted per episode
    addressing: Addressing


def scene_meta(desc: SceneDescriptor, overrides: SceneOverrides | None = None) -> SceneMeta:
    """Registry row for a (possibly arm-subsetted / mic-overridden) scene."""
    arms = _effective_arms(desc, overrides)
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
        title=desc.title,
        hidden=desc.hidden,
        microphones={a.id: a.microphone for a in arms},
        graspable=tuple(desc.graspable),
    )


def build_scene(desc: SceneDescriptor, overrides: SceneOverrides | None = None) -> BuiltScene:
    """Compose and compile a scene; raises the typed scene errors on failure."""
    arms = _effective_arms(desc, overrides)
    meta = scene_meta(desc, overrides)
    spec = mujoco.MjSpec()
    spec.modelname = desc.id
    # Absolute meshdir / texturedir so spec.to_xml() re-compiles anywhere (episode
    # replay); environment meshes and textures are asset-relative FILES (03-sim §4.1).
    spec.meshdir = str(asset_path())
    spec.texturedir = str(asset_path())
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
    _validate_graspable(model, desc)
    return BuiltScene(meta, spec, model, spec.to_xml(), Addressing(model, meta))


def _validate_allowed_pairs(
    model: mujoco.MjModel, desc: SceneDescriptor, arms: tuple[ArmSpec, ...]
) -> None:
    """Every scene-authored allowed-pair label must resolve to a pair label
    of the built model (world geom name or ``<arm_id>_<body>``); labels of
    arms dropped by an ``arm_ids`` override and ``<arm_id>_microphone`` of
    arms built without the mic are skipped, typos fail loudly."""
    known = {model.body(b).name for b in range(1, model.nbody)}
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name:
                known.add(name)
    dropped = {a.id for a in desc.arms} - {a.id for a in arms}
    optional = {f"{a.id}_microphone" for a in arms if not a.microphone}
    for pair in desc.allowed_pairs:
        for label in pair:
            if (
                label in known
                or label in optional
                or any(label.startswith(f"{d}_") for d in dropped)
            ):
                continue
            raise SceneCompileError(
                f"scene {desc.id!r}: allowed_pairs label {label!r} matches no world geom "
                "or arm body of the built model"
            )


def _validate_graspable(model: mujoco.MjModel, desc: SceneDescriptor) -> None:
    """Every ``graspable`` name must be a WORLD geom of the built model (a session
    whitelists it against a gripper by its twin pair label = the geom name)."""
    world_geoms = set()
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name:
                world_geoms.add(name)
    for label in desc.graspable:
        if label not in world_geoms:
            raise SceneCompileError(
                f"scene {desc.id!r}: graspable {label!r} matches no world geom of the built model"
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
    if desc.view is not None:  # default free camera (runtime "sim" stream)
        spec.visual.global_.azimuth = desc.view.azimuth
        spec.visual.global_.elevation = desc.view.elevation


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
    materials: dict[str, str] = {}  # texture path -> material name
    meshes: dict[tuple[str, tuple[float, float, float]], str] = {}  # (file, scale) -> mesh name
    taken: set[str] = set()  # asset identifiers handed out so far
    for env in desc.environment:
        kwargs: dict = dict(
            name=env.name, pos=list(env.pos), quat=list(env.quat), rgba=list(env.rgba)
        )
        if env.type == "mesh":
            assert env.mesh is not None  # EnvironmentSpec validator
            key = (env.mesh, tuple(env.scale))
            if key not in meshes:
                _require_asset(desc, env.mesh, f"environment geom {env.name!r} mesh")
                mesh_name = _asset_ident("mesh", env.mesh, taken)
                spec.add_mesh(name=mesh_name, file=env.mesh, scale=list(env.scale))
                meshes[key] = mesh_name
            kwargs.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=meshes[key])
        else:
            assert env.size is not None  # EnvironmentSpec validator
            kwargs.update(
                type=(
                    mujoco.mjtGeom.mjGEOM_PLANE
                    if env.type == "plane"
                    else mujoco.mjtGeom.mjGEOM_BOX
                ),
                size=list(env.size),
            )
        if env.texture is not None:
            if env.texture not in materials:
                # One 2D FILE texture + material per distinct PNG (buffer textures would
                # break spec.to_xml(), which is persisted with every episode).
                _require_asset(desc, env.texture, f"environment geom {env.name!r} texture")
                ident = _asset_ident("tex", env.texture, taken)
                spec.add_texture(
                    name=ident, type=mujoco.mjtTexture.mjTEXTURE_2D, file=env.texture
                )
                mat = spec.add_material(name=f"{ident}_mat")
                mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = ident
                mat.texuniform = False
                materials[env.texture] = mat.name
            kwargs["material"] = materials[env.texture]
        if not env.collidable:
            # Visual-only (AprilTag plates): never a contact, never a monitored pair,
            # never inflated; group 1 unless the author picks one.
            kwargs.update(contype=0, conaffinity=0, group=1 if env.group is None else env.group)
        elif env.group is not None:
            kwargs["group"] = env.group
        spec.worldbody.add_geom(**kwargs)
    for cam in desc.cameras:
        spec.worldbody.add_camera(
            name=cam.name, pos=list(cam.pos), xyaxes=list(cam.xyaxes), fovy=cam.fovy
        )


def _require_asset(desc: SceneDescriptor, rel: str, what: str) -> None:
    """Asset-relative mesh / texture files must exist BEFORE compile (MuJoCo's own
    error names the absolute path and arrives from deep inside the compiler)."""
    try:
        asset_path(*rel.split("/"))
    except FileNotFoundError as e:
        raise SceneCompileError(f"scene {desc.id!r}: {what} {rel!r} is not a vendored asset") from e


def _asset_ident(kind: str, rel: str, taken: set[str]) -> str:
    """Stable MJCF asset name for an asset-relative path (``tex_textures_tag_png``);
    a second path that sanitises to the same identifier gets a ``_2`` / ``_3`` suffix."""
    stem = "".join(c if c.isalnum() else "_" for c in rel).strip("_")
    name, n = f"{kind}_{stem}", 1
    while name in taken:
        n += 1
        name = f"{kind}_{stem}_{n}"
    taken.add(name)
    return name


def _customize_child(child: mujoco.MjSpec, arm: ArmSpec) -> None:
    """Per-arm child edits: drop the wrist cam and/or gripper subtree, add the mic."""
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
    if arm.microphone:  # ArmSpec validator: camera-only arm (wrist_cam, no gripper)
        _add_microphone(child)
        modified = True
    if modified:
        # Editing leaves keyframes "pending"; compiling the child finalizes
        # them so attach prefixes them correctly (MuJoCo warns otherwise).
        child.compile()


def _add_microphone(child: mujoco.MjSpec) -> None:
    """Joint-less ``microphone`` body under link7 with ONE collidable cylinder.

    A separate body (pair label ``<arm>_microphone``) rather than a geom on
    ``d435_mount`` so twin events name the mic, not the camera. No joint, no
    actuator, no site: nq/nu/keyframes and the addressing layer are untouched;
    the twin, IK avoidance rows and guardrail pick the geom up through the
    per-arm subtree scan. Explicit ``type``/``rgba`` override the child's
    ``xarm7`` default class (type=mesh, material white).
    """
    mic = child.body("link7").add_body(name="microphone")
    mic.add_geom(
        name="microphone",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[MIC_RADIUS_M, MIC_HALF_LENGTH_M, 0.0],
        pos=[0.0, 0.0, MIC_HALF_LENGTH_M],  # axis = link7 +z (flange axis), z in [0, tip]
        rgba=list(MIC_RGBA),
        contype=1,
        conaffinity=1,
        group=0,
        mass=MIC_MASS_KG,  # explicit (density default would give ~0.6 kg)
    )


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


def _effective_arms(
    desc: SceneDescriptor, overrides: SceneOverrides | None
) -> tuple[ArmSpec, ...]:
    """Selected arms with ``SceneOverrides.microphones`` applied.

    The override re-validates the ``ArmSpec`` (the camera-only gate lives in
    ONE place); an unknown arm id is a ``SceneArmMismatchError`` like
    ``arm_ids``, a mic on a gripper / camera-less arm a ``SceneCompileError``.
    """
    arms = _selected_arms(desc, overrides)
    if overrides is None or not overrides.microphones:
        return arms
    known = {a.id for a in desc.arms}
    unknown = sorted(i for i in overrides.microphones if i not in known)
    if unknown:
        raise SceneArmMismatchError(
            f"scene {desc.id!r} has arms {sorted(known)}, "
            f"microphones override names {unknown}"
        )
    out: list[ArmSpec] = []
    for arm in arms:
        if arm.id not in overrides.microphones:
            out.append(arm)
            continue
        try:
            out.append(
                ArmSpec.model_validate(
                    {**arm.model_dump(), "microphone": bool(overrides.microphones[arm.id])}
                )
            )
        except ValidationError as e:
            raise SceneCompileError(
                f"scene {desc.id!r}: microphones override rejected: {e}"
            ) from e
    return tuple(out)


__all__ = [
    "HOME_Q",
    "RAIL_HOME_M",
    "MIC_RADIUS_M",
    "MIC_TIP_Z_M",
    "MIC_MASS_KG",
    "WRIST_CAM_POS_M",
    "MIC_REF_PLANE_Z_M",
    "WRIST_CAM_Z_M",
    "BuiltScene",
    "scene_meta",
    "build_scene",
]
