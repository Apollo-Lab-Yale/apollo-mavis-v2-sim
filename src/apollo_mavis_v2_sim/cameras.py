"""SimCamera: core CameraInterface adapter over the RenderService (03-sim §7).

``start()`` registers a stream on the render service; ``latest()`` polls the
service's depth-1 frame slot. The capture "thread" is the render thread —
this class owns none.
"""

from __future__ import annotations

from apollo_mavis_v2_core import CameraFrame, CameraInterface

from .rendering import RenderService, StreamSpec

SIM_SOURCE = "sim"


class SimCamera(CameraInterface):
    """One simulated RGB(-D) camera stream rendered from a named MJCF camera.

    ``depth=True`` (phase-12) adds a seq-aligned uint16-mm depth image to every
    ``CameraFrame`` (``frame.depth``); the default colour-only stream is unchanged.
    """

    def __init__(
        self,
        camera_id: str,
        render_service: RenderService,
        mjcf_camera: str | int,
        resolution: tuple[int, int] = (640, 480),
        fps: float = 30.0,
        source: str = SIM_SOURCE,
        depth: bool = False,
    ) -> None:
        self._camera_id = camera_id
        self._service = render_service
        self._spec = StreamSpec(
            stream_id=camera_id,
            source=source,
            camera=mjcf_camera,
            width=resolution[0],
            height=resolution[1],
            fps=fps,
            depth=depth,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._service.add_stream(self._spec)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._service.remove_stream(self._camera_id)
        self._started = False

    def latest(self) -> CameraFrame | None:
        return self._service.latest(self._camera_id)

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._spec.width, self._spec.height)

    @property
    def fps(self) -> float:
        return self._spec.fps


__all__ = ["SimCamera", "SIM_SOURCE"]
