"""Offscreen RenderService: one thread owns every Renderer (03-sim §7).

GL contexts are thread-affine, so ALL ``mujoco.Renderer`` instances live in
the single render thread, created lazily inside it — never in the stepping
or asyncio threads. ``MUJOCO_GL=egl`` must be exported before mujoco's GL
initialization; the RUNTIME entrypoint owns that (this module never touches
environment variables). Each source keeps a private ``MjData`` (never the
physics one); inputs arrive via depth-1 latest-qpos slots and outputs leave
via depth-1 latest-frame slots.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np
from apollo_mavis_v2_core import CameraFrame, LatestSlot

logger = logging.getLogger(__name__)

_IDLE_SLEEP_S = 0.02


@dataclass(frozen=True)
class StreamSpec:
    stream_id: str  # "sim", "twin", or a camera id e.g. "left_wrist_cam"
    source: str  # which registered model/state to render
    camera: str | int | None  # named MJCF camera; None = free camera
    width: int = 640
    height: int = 480
    fps: float = 30.0
    depth: bool = False  # reserved; depth rendering is off in v1 (03-sim §7)
    show_inflation: bool = False  # twin debug: geom group 3 visible


class _Source:
    """A renderable model + latest-qpos slot; MjData created in-thread."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.qpos_slot: LatestSlot = LatestSlot()
        self.data: mujoco.MjData | None = None  # created lazily by the render thread


class _Stream:
    def __init__(self, spec: StreamSpec) -> None:
        self.spec = spec
        self.frame_slot: LatestSlot = LatestSlot()
        self.buffer = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
        self.next_t = 0.0
        self.seq = 0
        self.dead = False
        self.scene_option = mujoco.MjvOption()
        if spec.show_inflation:
            self.scene_option.geomgroup[3] = 1


class RenderService:
    """Multi-stream offscreen renderer with per-stream fps pacing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: dict[str, _Source] = {}
        self._streams: dict[str, _Stream] = {}
        self._renderers: dict[tuple[str, int, int], mujoco.Renderer] = {}  # thread-owned
        self._thread: threading.Thread | None = None
        self._running = False

    # -- registration --------------------------------------------------------
    def register_source(self, source: str, model: mujoco.MjModel) -> None:
        with self._lock:
            self._sources[source] = _Source(model)

    def add_stream(self, spec: StreamSpec) -> None:
        if spec.depth:
            raise NotImplementedError("depth rendering is off in v1 (03-sim §7)")
        with self._lock:
            if spec.source not in self._sources:
                raise ValueError(f"unknown render source {spec.source!r}")
            if spec.stream_id in self._streams:
                raise ValueError(f"duplicate stream id {spec.stream_id!r}")
            self._streams[spec.stream_id] = _Stream(spec)

    def remove_stream(self, stream_id: str) -> None:
        with self._lock:
            self._streams.pop(stream_id, None)

    # -- data flow -------------------------------------------------------------
    def submit_state(self, source: str, qpos: np.ndarray, t: float) -> None:
        src = self._sources.get(source)
        if src is not None:
            src.qpos_slot.put((np.array(qpos, dtype=np.float64), float(t)))

    def latest(self, stream_id: str) -> CameraFrame | None:
        stream = self._streams.get(stream_id)
        if stream is None or stream.dead:
            return None
        got = stream.frame_slot.get()
        return got[0] if got is not None else None

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, name="sim-render", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # -- render thread ------------------------------------------------------------
    def _render_loop(self) -> None:
        try:
            while self._running:
                with self._lock:
                    streams = [s for s in self._streams.values() if not s.dead]
                if not streams:
                    time.sleep(_IDLE_SLEEP_S)
                    continue
                now = time.monotonic()
                due = [s for s in streams if s.next_t <= now]
                if not due:
                    wake = min(s.next_t for s in streams)
                    time.sleep(min(_IDLE_SLEEP_S, max(0.0, wake - now)))
                    continue
                for stream in due:
                    try:
                        self._render_stream(stream)
                    except Exception:
                        # A renderer exception kills only its stream (03-sim §12).
                        stream.dead = True
                        logger.exception("stream %r died", stream.spec.stream_id)
                        continue
                    period = 1.0 / stream.spec.fps
                    stream.next_t = (stream.next_t or now) + period
                    if stream.next_t < now - period:  # far behind: re-align, no bursts
                        stream.next_t = now + period
        finally:
            # Explicit close in the owning thread avoids benign-but-noisy
            # destructor-order EGLErrors at interpreter exit.
            for renderer in self._renderers.values():
                try:
                    renderer.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    logger.warning("renderer close failed", exc_info=True)
            self._renderers.clear()

    def _render_stream(self, stream: _Stream) -> None:
        spec = stream.spec
        src = self._sources[spec.source]
        if src.data is None:
            src.data = mujoco.MjData(src.model)
            mujoco.mj_forward(src.model, src.data)
        got = src.qpos_slot.get()
        if got is not None:
            qpos, _sim_t = got[0]
            src.data.qpos[:] = qpos
            mujoco.mj_forward(src.model, src.data)
        key = (spec.source, spec.height, spec.width)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(src.model, height=spec.height, width=spec.width)
            self._renderers[key] = renderer
        camera = spec.camera if spec.camera is not None else -1
        renderer.update_scene(src.data, camera=camera, scene_option=stream.scene_option)
        renderer.render(out=stream.buffer)  # zero-alloc into the stream buffer
        stream.seq += 1
        stream.frame_slot.put(
            CameraFrame(
                camera_id=spec.stream_id,
                rgb=stream.buffer.copy(),  # published frames are immutable
                t_mono=time.monotonic(),
                wallclock_ns=time.time_ns(),
                seq=stream.seq,
            )
        )


__all__ = ["StreamSpec", "RenderService"]
