"""apollo-xarm7-sim: MuJoCo simulation workcell for the apollo-xarm7 stack.

Public API re-exports (design 03-sim §1). This module NEVER touches
environment variables: ``MUJOCO_GL=egl`` must be exported by the runtime
entrypoint (or the test/benchmark harness) before mujoco initializes its GL
context; rendering is lazily initialized inside the render thread.
"""

from .assets import asset_path
from .cameras import SimCamera
from .errors import (
    SceneArmMismatchError,
    SceneCompileError,
    SceneError,
    SceneNotFoundError,
)
from .gripper import (
    DRIVER_CLOSED_RAD,
    GRIPPER_CTRL_MAX,
    GRIPPER_SPAN_M,
    ctrl_to_open_frac,
    driver_q_to_open_frac,
    open_frac_to_ctrl,
    open_frac_to_meters,
)
from .rendering import RenderService, StreamSpec
from .scenes import (
    REGISTRY,
    Addressing,
    ArmAddress,
    BuiltScene,
    SceneDescriptor,
    SceneMeta,
    SceneOverrides,
    SceneRegistry,
    build_scene,
)
from .workcell import CTRL_DT, SimArm, SimWorkcell, WorkcellSnapshot

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "asset_path",
    # scenes
    "REGISTRY",
    "SceneRegistry",
    "SceneDescriptor",
    "SceneMeta",
    "SceneOverrides",
    "BuiltScene",
    "build_scene",
    "Addressing",
    "ArmAddress",
    # workcell
    "CTRL_DT",
    "SimWorkcell",
    "SimArm",
    "WorkcellSnapshot",
    # cameras / rendering
    "SimCamera",
    "RenderService",
    "StreamSpec",
    # gripper
    "GRIPPER_CTRL_MAX",
    "DRIVER_CLOSED_RAD",
    "GRIPPER_SPAN_M",
    "open_frac_to_ctrl",
    "ctrl_to_open_frac",
    "driver_q_to_open_frac",
    "open_frac_to_meters",
    # errors
    "SceneError",
    "SceneNotFoundError",
    "SceneArmMismatchError",
    "SceneCompileError",
]
