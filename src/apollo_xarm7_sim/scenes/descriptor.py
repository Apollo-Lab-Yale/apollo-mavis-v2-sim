"""Scene descriptor schema (pydantic) + build-time overrides (03-sim §4).

One YAML file per scene under ``assets/scenes/``; the registry validates
each into a :class:`SceneDescriptor`. Descriptor ``keyframe`` vectors use
the MJCF-internal qpos order (rail slide FIRST) — scene authoring is the
only place outside :mod:`.addressing` where that ordering appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from apollo_xarm7_core import Pose
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

    @property
    def has_rail(self) -> bool:
        return self.model == "xarm7_on_rail"


class CameraSpec(BaseModel):
    """A named environment camera on the parent spec."""

    model_config = ConfigDict(frozen=True)
    name: str
    pos: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float = 45.0


class EnvironmentSpec(BaseModel):
    """Static environment geom (phase-02: plane | box; meshes are phase-03+)."""

    model_config = ConfigDict(frozen=True)
    name: str
    type: Literal["plane", "box"]
    size: tuple[float, float, float]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)


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


__all__ = [
    "ARM_MODELS",
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
