"""SimWorkcell / SimArm — the sim IS the robot (design 03-sim §6).

Implements core's ``WorkcellInterface``/``ArmInterface`` over a composed
scene: a monotonic-paced stepping thread writes position-servo targets into
``data.ctrl`` and steps physics at 100 Hz (5 x 2 ms substeps), publishing
immutable :class:`ArmState` snapshots. Commands arrive in CORE order (rail
LAST at ``q[7]``); the write into ``data.ctrl`` goes through
``Addressing.ctrl_adr`` which hides the MJCF rail-first layout.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
from apollo_mavis_v2_core import (
    ArmInterface,
    ArmState,
    CameraInterface,
    CommandError,
    GripperCommand,
    GripperState,
    Pose,
    RailUnavailableError,
    WorkcellConfig,
    WorkcellInterface,
    se3,
)

from .errors import SceneArmMismatchError
from .gripper import driver_q_to_open_frac, open_frac_to_ctrl
from .scenes.addressing import ArmAddress
from .scenes.builder import BuiltScene

if TYPE_CHECKING:
    from .rendering import RenderService

logger = logging.getLogger(__name__)

CTRL_DT = 0.01  # 100 Hz command tick
WORKCELL_FAULT_CODE = 99  # error_code latched on every arm if the step thread dies
SIM_SOURCE = "sim"


@dataclass(frozen=True)
class WorkcellSnapshot:
    """Lock-free read of the latest published state (one per tick)."""

    states: dict[str, ArmState]
    sim_time: float
    tick: int
    t_mono: float


class SimArm(ArmInterface):
    """One simulated arm; a thin command/state facade over the workcell."""

    def __init__(self, workcell: SimWorkcell, addr: ArmAddress) -> None:
        self._cell = workcell
        self._addr = addr
        self._connected = False

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:  # no hardware: immediate CONNECTED
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    # -- state -------------------------------------------------------------
    def get_state(self) -> ArmState:
        return self._cell.snapshot().states[self._addr.arm_id]

    # -- commands (non-blocking latest-wins target writes) ------------------
    def command_joints(self, q: np.ndarray) -> None:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (self.dof,):
            raise CommandError(
                f"{self._addr.arm_id}: command_joints expects shape ({self.dof},), "
                f"got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise CommandError(f"{self._addr.arm_id}: command_joints got non-finite values")
        if self._addr.has_rail:
            arr = arr.copy()
            arr[7] = min(se3.RAIL_TRAVEL_M, max(0.0, arr[7]))
        self._cell._write_targets(self._addr.ctrl_adr, arr)

    def command_gripper(self, cmd: GripperCommand) -> None:
        if self._addr.gripper_ctrl_adr is None:
            raise CommandError(f"{self._addr.arm_id} has no gripper")
        self._cell._write_target(self._addr.gripper_ctrl_adr, open_frac_to_ctrl(cmd.open_frac))

    def command_rail(self, pos_m: float) -> None:
        if not self._addr.has_rail:
            raise RailUnavailableError(f"{self._addr.arm_id} has no rail")
        pos = min(se3.RAIL_TRAVEL_M, max(0.0, float(pos_m)))
        self._cell._write_target(int(self._addr.ctrl_adr[-1]), pos)

    def clear_errors(self) -> None:
        self._cell._clear_fault(self._addr.arm_id)

    def stop(self) -> None:
        """Software stop: freeze targets at the current measured posture."""
        state = self.get_state()
        self._cell._write_targets(self._addr.ctrl_adr, state.q)

    # -- properties ----------------------------------------------------------
    @property
    def dof(self) -> int:
        return self._addr.dof

    @property
    def has_rail(self) -> bool:
        return self._addr.has_rail

    @property
    def gripper_force_capable(self) -> bool:
        return False  # sim gripper is position-only


class SimWorkcell(WorkcellInterface):
    """A simulated workcell with its own physics-stepping thread."""

    def __init__(
        self,
        scene: BuiltScene,
        config: WorkcellConfig,
        render_service: RenderService | None = None,
        ctrl_hz: float = 100.0,
        depth_cameras: Iterable[str] = (),  # phase-12: cameras whose frames carry a depth sibling
    ) -> None:
        scene_arms = set(scene.meta.arm_ids)
        wanted = [a.id for a in config.arms]
        missing = [a for a in wanted if a not in scene_arms]
        if missing:
            raise SceneArmMismatchError(
                f"config arms {missing} not in scene {scene.meta.id!r} "
                f"(has {sorted(scene_arms)})"
            )
        self.scene = scene
        self.config = config
        self._model = scene.model
        self._data = mujoco.MjData(scene.model)
        self._render_service = render_service
        self._ctrl_dt = 1.0 / ctrl_hz
        self._nsub = max(1, round(self._ctrl_dt / scene.model.opt.timestep))
        self._cmd_lock = threading.Lock()
        self._targets = np.zeros(scene.model.nu)
        self._snapshot_slot: WorkcellSnapshot | None = None
        self._faults: dict[str, int] = {}
        self._fault_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._tick = 0
        self.overrun_count = 0
        self._tick_hook: Callable[[], None] | None = None  # tests only
        self.arms: dict[str, ArmInterface] = {
            arm_id: SimArm(self, scene.addressing[arm_id]) for arm_id in wanted
        }
        self.cameras: dict[str, CameraInterface] = {}
        if render_service is not None:
            render_service.register_source(SIM_SOURCE, scene.model)
            self.cameras = _make_sim_cameras(
                scene, config, render_service, depth_cameras=set(depth_cameras)
            )
        self._reset_and_publish()

    @property
    def kind(self) -> Literal["hardware", "sim"]:
        return "sim"

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        """Reset to keyframe 0, clear faults, and spawn the stepping thread."""
        if self._running:
            return
        with self._fault_lock:
            self._faults.clear()
        self._reset_and_publish()
        for arm in self.arms.values():
            arm.connect()
        for cam in self.cameras.values():
            cam.start()
        self._running = True
        self._thread = threading.Thread(
            target=self._step_loop, name="sim-step", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for cam in self.cameras.values():
            cam.stop()
        for arm in self.arms.values():
            arm.disconnect()

    def states(self) -> dict[str, ArmState]:
        return self.snapshot().states

    def snapshot(self) -> WorkcellSnapshot:
        snap = self._snapshot_slot
        assert snap is not None  # published in __init__
        return snap

    @property
    def tick_count(self) -> int:
        return self._tick

    def step_virtual(self, n_ticks: int = 1) -> None:
        """Advance physics synchronously by whole control ticks (no pacing).

        For virtual-tick harnesses (guardrail CI, tests): same ctrl-write +
        substep sequence as the stepping thread, but on the caller's thread
        and without wall-clock sleeps. Refuses while the thread runs.
        """
        if self._running:
            raise RuntimeError("step_virtual requires the stepping thread stopped")
        for _ in range(n_ticks):
            with self._cmd_lock:
                self._data.ctrl[:] = self._targets
            mujoco.mj_step(self._model, self._data, nstep=self._nsub)
            self._tick += 1
            self._publish_snapshot()
            if self._render_service is not None:
                self._render_service.submit_state(
                    SIM_SOURCE, self._data.qpos, self._data.time
                )

    def inject_fault(self, arm_id: str, code: int) -> None:
        """Tests only: latch an error code on one arm until clear_errors()."""
        if arm_id not in self.arms:
            raise KeyError(arm_id)
        with self._fault_lock:
            self._faults[arm_id] = int(code)
        if not self._running:  # thread not publishing -> republish now
            self._publish_snapshot()

    # -- internals -------------------------------------------------------------
    def _write_targets(self, ctrl_adr: np.ndarray, values: np.ndarray) -> None:
        with self._cmd_lock:
            self._targets[ctrl_adr] = values

    def _write_target(self, ctrl_adr: int, value: float) -> None:
        with self._cmd_lock:
            self._targets[ctrl_adr] = value

    def _clear_fault(self, arm_id: str) -> None:
        with self._fault_lock:
            self._faults.pop(arm_id, None)
        if not self._running:
            self._publish_snapshot()

    def _reset_and_publish(self) -> None:
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)
        with self._cmd_lock:
            self._targets[:] = self._data.ctrl  # keyframe ctrl = initial targets
        self._publish_snapshot()

    def _step_loop(self) -> None:
        model, data, nsub = self._model, self._data, self._nsub
        next_t = time.monotonic()
        try:
            while self._running:
                hook = self._tick_hook
                if hook is not None:
                    hook()
                with self._cmd_lock:
                    data.ctrl[:] = self._targets
                mujoco.mj_step(model, data, nstep=nsub)
                self._tick += 1
                self._publish_snapshot()
                if self._render_service is not None:
                    self._render_service.submit_state(SIM_SOURCE, data.qpos, data.time)
                # Drift-free monotonic pacing; re-sync when > 1 tick behind so a
                # stall never turns into a frame-chasing burst.
                next_t += self._ctrl_dt
                lag = time.monotonic() - next_t
                if lag > self._ctrl_dt:
                    next_t = time.monotonic()
                    self.overrun_count += 1
                elif lag < 0.0:
                    time.sleep(-lag)
        except Exception:
            logger.exception("sim step thread died; latching workcell fault")
            self._running = False
            with self._fault_lock:
                for arm_id in self.arms:
                    self._faults.setdefault(arm_id, WORKCELL_FAULT_CODE)
            self._publish_snapshot()

    def _publish_snapshot(self) -> None:
        data = self._data
        t_mono = time.monotonic()
        wallclock_ns = time.time_ns()
        with self._fault_lock:
            faults = dict(self._faults)
        states: dict[str, ArmState] = {}
        for arm_id in self.arms:
            addr = self.scene.addressing[arm_id]
            q = data.qpos[addr.qpos_adr].copy()
            dq = data.qvel[addr.dof_adr].copy()
            rail_pos = float(q[7]) if addr.has_rail else None
            if addr.has_rail:
                dq[7] = 0.0  # rail velocity unobservable on hardware; stay isomorphic
            if addr.gripper_driver_qpos_adr is not None:
                open_frac = driver_q_to_open_frac(data.qpos[addr.gripper_driver_qpos_adr])
            else:
                open_frac = 1.0
            states[arm_id] = ArmState(
                arm_id=arm_id,
                q=q,
                dq=dq,
                ee_pose=_tcp_in_base(data, addr),
                gripper=GripperState(open_frac=open_frac),
                rail_pos_m=rail_pos,
                error_code=faults.get(arm_id, 0),
                warn_code=0,
                mode=1,  # servo
                state=0,  # ready
                stale=False,
                t_mono=t_mono,
                wallclock_ns=wallclock_ns,
            )
        self._snapshot_slot = WorkcellSnapshot(
            states=states, sim_time=float(data.time), tick=self._tick, t_mono=t_mono
        )


def _tcp_in_base(data: mujoco.MjData, addr: ArmAddress) -> Pose:
    """TCP site pose expressed in the arm_base (link_base) frame."""
    p_site = data.site_xpos[addr.tcp_site_id]
    r_site = data.site_xmat[addr.tcp_site_id].reshape(3, 3)
    p_base = data.xpos[addr.base_body_id]
    r_base = data.xmat[addr.base_body_id].reshape(3, 3)
    return Pose(r_base.T @ (p_site - p_base), se3.mat_to_quat(r_base.T @ r_site))


def _make_sim_cameras(
    scene: BuiltScene,
    config: WorkcellConfig,
    service: RenderService,
    depth_cameras: set[str] | None = None,
) -> dict[str, CameraInterface]:
    from .cameras import SimCamera  # local import: keep module import light

    by_id = {c.id: c for c in config.cameras if c.kind == "sim"}
    depth = depth_cameras or set()
    cameras: dict[str, CameraInterface] = {}
    for name in scene.meta.cameras:
        cfg = by_id.get(name)
        cameras[name] = SimCamera(
            camera_id=name,
            render_service=service,
            mjcf_camera=name,
            resolution=cfg.resolution if cfg else (640, 480),
            fps=float(cfg.fps) if cfg else 30.0,
            depth=(name in depth) or bool(cfg and getattr(cfg, "depth", False)),
        )
    return cameras


__all__ = ["CTRL_DT", "WORKCELL_FAULT_CODE", "WorkcellSnapshot", "SimArm", "SimWorkcell"]
