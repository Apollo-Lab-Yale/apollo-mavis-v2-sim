"""Scene descriptors, registry, MjSpec composition, and addressing."""

from .addressing import Addressing, ArmAddress
from .builder import BuiltScene, build_scene, scene_meta
from .descriptor import (
    ArmSpec,
    CameraSpec,
    EnvironmentSpec,
    KeyframeArm,
    OffscreenSpec,
    SceneDescriptor,
    SceneMeta,
    SceneOptions,
    SceneOverrides,
)
from .registry import REGISTRY, SceneRegistry

__all__ = [
    "Addressing",
    "ArmAddress",
    "BuiltScene",
    "build_scene",
    "scene_meta",
    "ArmSpec",
    "CameraSpec",
    "EnvironmentSpec",
    "KeyframeArm",
    "OffscreenSpec",
    "SceneDescriptor",
    "SceneMeta",
    "SceneOptions",
    "SceneOverrides",
    "SceneRegistry",
    "REGISTRY",
]
