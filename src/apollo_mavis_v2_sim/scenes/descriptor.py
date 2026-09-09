"""Scene descriptor schema (pydantic) + build-time overrides (03-sim §4).

One YAML file per scene under ``assets/scenes/``; the registry validates
each into a :class:`SceneDescriptor`. Descriptor ``keyframe`` vectors use
the MJCF-internal qpos order (rail slide FIRST) — scene authoring is the
only place outside :mod:`.addressing` where that ordering appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from apollo_mavis_v2_core import Pose
from pydantic import BaseModel, ConfigDict, Field, model_validator

ARM_MODELS = ("xarm7_on_rail", "xarm7_fixed")


class SceneOptions(BaseModel):
    """Physics options, authored on the PARENT spec only (03-sim §5)."""

    model_config = ConfigDict(frozen=True)
    timestep: float = 0.002
    integrator: Literal["implicitfast"] = "implicitfast"
    cone: Literal["elliptic", "pyramidal"] = "elliptic"
    impratio: float = 10.0
    multiccd: bool = True


class OffscreenSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    width: int = 1920
    height: int = 1080


class SceneView(BaseModel):
    """Default MuJoCo free camera (``<visual><global azimuth elevation>``).

    The free camera (``Renderer.update_scene(camera=-1)``, the runtime ``sim``
    stream) orbits the model statistic centre; azimuth 90 puts it at -Y looking
    +Y (MuJoCo default), -90 at +Y looking -Y. Defaults mirror MuJoCo's.
    """

    model_config = ConfigDict(frozen=True)
    azimuth: float = 90.0
    elevation: float = -45.0


class ArmSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    model: Literal["xarm7_on_rail", "xarm7_fixed"]
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    gripper: Literal["xarm", "none"] = "xarm"
    wrist_cam: bool = True
    # Microphone body (RODE NT-USB Mini + bracket) in front of the wrist camera:
    # a joint-less cylinder coaxial with the link7 flange (03-sim §3). Only a
    # camera-only arm can carry it -- on a gripper arm the 0.19 m cylinder would
    # run through the 0.172 m gripper and its jointed fingers.
    microphone: bool = False

    @property
    def has_rail(self) -> bool:
        return self.model == "xarm7_on_rail"

    @model_validator(mode="after")
    def _microphone_needs_camera_only_arm(self) -> ArmSpec:
        if self.microphone and not (self.wrist_cam and self.gripper == "none"):
            raise ValueError(
                f"arm {self.id!r}: microphone requires wrist_cam: true and gripper: none "
                f"(got wrist_cam={self.wrist_cam}, gripper={self.gripper!r})"
            )
        return self


class CameraSpec(BaseModel):
    """A named environment camera on the parent spec."""

    model_config = ConfigDict(frozen=True)
    name: str
    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float = 45.0


MJ_NGROUP = 6  # MuJoCo geom groups 0..5 (mjNGROUP)


def _asset_relative(path: str, what: str) -> str:
    """An asset-relative POSIX path under ``assets/`` (``textures/tag.png``): no
    absolute paths, no ``..`` -- the builder resolves it through ``asset_path``
    and persists only the relative name in the scene XML."""
    if not path or path.startswith("/") or ".." in path.split("/") or "\\" in path:
        raise ValueError(
            f"{what} must be an asset-relative path like 'textures/tag.png', got {path!r}"
        )
    return path


class EnvironmentSpec(BaseModel):
    """Static world geom: ``plane`` | ``box`` | ``mesh`` (03-sim §4.1).

    ``mesh`` geoms (implemented 2026-09-09 for file meshes, 16-gello §10) reference
    an asset-relative STL/OBJ through ``mesh`` and scale it with ``scale``; their
    collider is MuJoCo's convex hull of the mesh. ``texture`` (asset-relative PNG)
    becomes a 2D texture + material (``texuniform: false``) on the geom -- FILES only,
    because ``spec.to_xml()`` refuses buffer textures and the XML is persisted with
    every episode. ``collidable: false`` sets ``contype = conaffinity = 0`` (never a
    monitored pair, never in the twin's inflation) and puts the geom in group 1 unless
    ``group`` says otherwise; ``group`` alone overrides MuJoCo's default group 0.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    type: Literal["plane", "box", "mesh"]
    size: tuple[float, float, float] | None = None  # plane / box: required; mesh: use scale
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    mesh: str | None = None  # asset-relative STL/OBJ, type mesh only
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)  # mesh scale
    texture: str | None = None  # asset-relative PNG -> 2D texture + material
    collidable: bool = True
    group: int | None = None  # MuJoCo geom group 0..5; None = MuJoCo default (1 if not collidable)

    @model_validator(mode="after")
    def _check(self) -> EnvironmentSpec:
        if self.type == "mesh":
            if self.mesh is None:
                raise ValueError(f"environment geom {self.name!r}: type mesh requires 'mesh'")
            if self.size is not None:
                raise ValueError(
                    f"environment geom {self.name!r}: a mesh geom is sized by 'scale', not 'size'"
                )
            _asset_relative(self.mesh, f"environment geom {self.name!r} mesh")
        else:
            if self.mesh is not None:
                raise ValueError(
                    f"environment geom {self.name!r}: 'mesh' is only valid with type mesh "
                    f"(got type {self.type!r})"
                )
            if self.size is None:
                raise ValueError(
                    f"environment geom {self.name!r}: type {self.type} requires 'size'"
                )
        if self.texture is not None:
            _asset_relative(self.texture, f"environment geom {self.name!r} texture")
        if self.group is not None and not 0 <= self.group < MJ_NGROUP:
            raise ValueError(
                f"environment geom {self.name!r}: group must be in [0, {MJ_NGROUP - 1}], "
                f"got {self.group}"
            )
        return self


class KeyframeArm(BaseModel):
    """Per-arm initial state; ``q`` is MJCF qpos order (rail slide FIRST)."""

    model_config = ConfigDict(frozen=True)
    q: tuple[float, ...]
    gripper: float = 1.0  # open fraction, 1 = fully open

    @model_validator(mode="after")
    def _check(self) -> KeyframeArm:
        if len(self.q) not in (7, 8):
            raise ValueError(f"keyframe q must have 7 or 8 entries, got {len(self.q)}")
        if not 0.0 <= self.gripper <= 1.0:
            raise ValueError(f"keyframe gripper must be in [0, 1], got {self.gripper}")
        return self


class SceneDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    description: str
    title: str | None = None  # display name (UI); None -> callers fall back to description
    hidden: bool = False  # dev-only scene: kept for CI/tests, filtered from registry listings
    suitable_for: tuple[Literal["sim", "twin"], ...] = ("sim", "twin")
    options: SceneOptions = SceneOptions()
    offscreen: OffscreenSpec = OffscreenSpec()
    view: SceneView | None = None  # None -> MuJoCo's free-camera defaults untouched
    arms: tuple[ArmSpec, ...] = Field(min_length=1, max_length=3)
    cameras: tuple[CameraSpec, ...] = ()
    environment: tuple[EnvironmentSpec, ...] = ()
    keyframe: dict[str, KeyframeArm] | None = None  # None -> per-arm menagerie home
    # Structural collision-pair whitelist authored WITH the scene (11-safety
    # §6.3 source (a)): label pairs that are permanently inside the inflation
    # band by construction (e.g. a rail carriage 24 mm above the table it is
    # bolted to, two rails mounted side by side). Labels are the twin's pair
    # labels: world geoms by geom name ("table"), arm bodies as
    # "<arm_id>_<body>" ("grip_rail_platform"). Unknown labels fail the build.
    allowed_pairs: tuple[tuple[str, str], ...] = ()
    # World geom names a session may whitelist against a gripper (16-gello D7:
    # `fridge_door_handle`, ...): the twin's ``set_grasp_whitelist(arm, graspable)``
    # drops finger <-> handle pairs while every arm link stays gated against every
    # appliance body. Validated at build against the built model's world geoms.
    graspable: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _cross_field(self) -> SceneDescriptor:
        arm_ids = [a.id for a in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError(f"arm ids must be unique, got {arm_ids}")
        for a in arm_ids:
            if any(o != a and o.startswith(f"{a}_") for o in arm_ids):
                raise ValueError(f"arm id {a!r} is a prefix of another arm id (labels clash)")
        for pair in self.allowed_pairs:
            if len(pair) != 2 or pair[0] == pair[1] or not all(pair):
                raise ValueError(f"allowed_pairs entries must be two distinct labels, got {pair}")
        names = [c.name for c in self.cameras] + [e.name for e in self.environment]
        if len(set(names)) != len(names):
            raise ValueError(f"camera/environment names must be unique, got {names}")
        if len(set(self.graspable)) != len(self.graspable) or not all(self.graspable):
            raise ValueError(f"graspable names must be unique and non-empty, got {self.graspable}")
        env_by_name = {e.name: e for e in self.environment}
        for g in self.graspable:
            env = env_by_name.get(g)
            if env is not None and not env.collidable:
                raise ValueError(
                    f"graspable geom {g!r} is not collidable -- a grasp whitelist on it is a no-op"
                )
        if self.keyframe is not None:
            for arm_id, kf in self.keyframe.items():
                arm = next((a for a in self.arms if a.id == arm_id), None)
                if arm is None:
                    raise ValueError(f"keyframe references unknown arm {arm_id!r}")
                expect = 8 if arm.has_rail else 7
                if len(kf.q) != expect:
                    raise ValueError(
                        f"keyframe for arm {arm_id!r} needs {expect} qpos entries "
                        f"(MJCF order, rail first), got {len(kf.q)}"
                    )
        return self


@dataclass(frozen=True)
class SceneOverrides:
    """Runtime-supplied build overrides (03-sim §4)."""

    arm_ids: tuple[str, ...] | None = None  # subset of descriptor arms to instantiate
    base_pose: dict[str, Pose] = field(default_factory=dict)  # per-arm world pose
    geom_inflation_m: float | None = None  # twin only: TOTAL pair inflation (gap = d/2)
    # Per-arm microphone body on/off (same shape as base_pose); overrides the
    # descriptor's ArmSpec.microphone so ONE YAML serves the mic-less sim scene
    # and the hardware digital twin (ArmConfig.microphone -> {arm_id: True}).
    microphones: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneMeta:
    """Registry row (03-sim §4); runtime serializes this into core SceneInfo."""

    id: str
    description: str
    n_arms: int
    arm_ids: tuple[str, ...]
    rail: dict[str, bool]  # per arm id
    wrist_cams: dict[str, bool]  # per arm id
    cameras: tuple[str, ...]  # named MJCF cameras (post-prefix names)
    suitable_for: frozenset[str]  # {"sim", "twin"}
    allowed_pairs: tuple[tuple[str, str], ...] = ()  # scene-authored structural pairs
    title: str | None = None  # display name; runtime label = title or description
    hidden: bool = False  # filtered from SceneRegistry.list() by default
    microphones: dict[str, bool] = field(default_factory=dict)  # per arm id, post-override
    graspable: tuple[str, ...] = ()  # world geoms a session may whitelist against a gripper


__all__ = [
    "ARM_MODELS",
    "MJ_NGROUP",
    "SceneOptions",
    "OffscreenSpec",
    "SceneView",
    "ArmSpec",
    "CameraSpec",
    "EnvironmentSpec",
    "KeyframeArm",
    "SceneDescriptor",
    "SceneOverrides",
    "SceneMeta",
]
