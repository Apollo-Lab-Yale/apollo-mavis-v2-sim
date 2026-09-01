"""RenderService + SimCamera: EGL headless smoke (03-sim §7)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from apollo_xarm7_core import CameraInterface
from conftest import make_config

from apollo_xarm7_sim import RenderService, SimCamera, SimWorkcell, StreamSpec

pytestmark = pytest.mark.egl


def _wait_frame(svc, stream_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = svc.latest(stream_id)
        if frame is not None:
            return frame
        time.sleep(0.02)
    return None


@pytest.fixture()
def service(single_fixed_scene):
    svc = RenderService()
    svc.register_source("sim", single_fixed_scene.model)
    svc.start()
    yield svc
    svc.stop()


def test_egl_frame_shape_and_content(service):
    service.add_stream(StreamSpec("view", "sim", camera="cam_front"))
    frame = _wait_frame(service, "view")
    assert frame is not None
    assert frame.rgb.shape == (480, 640, 3) and frame.rgb.dtype == np.uint8
    assert frame.rgb.any(), "frame is all black"


def test_full_hd_renders(service):
    """offwidth/offheight were raised to 1920x1080 in every scene spec."""
    service.add_stream(StreamSpec("hd", "sim", camera="cam_front", width=1920, height=1080))
    frame = _wait_frame(service, "hd")
    assert frame is not None and frame.rgb.shape == (1080, 1920, 3)


def test_free_camera_and_named_camera(service):
    service.add_stream(StreamSpec("free", "sim", camera=None))
    service.add_stream(StreamSpec("wrist", "sim", camera="arm0_wrist_cam"))
    assert _wait_frame(service, "free") is not None
    assert _wait_frame(service, "wrist") is not None


def test_submit_state_changes_frame(service, single_fixed_scene):
    service.add_stream(StreamSpec("view", "sim", camera="cam_front"))
    f1 = _wait_frame(service, "view")
    qpos = single_fixed_scene.model.key(0).qpos.copy()
    addr = single_fixed_scene.addressing["arm0"]
    qpos[addr.qpos_adr[0]] += 1.5  # swing joint1
    service.submit_state("sim", qpos, 0.0)
    time.sleep(0.2)
    f2 = service.latest("view")
    assert f2 is not None and f2.seq > f1.seq
    assert (f1.rgb != f2.rgb).any(), "moving the arm should change pixels"


def test_stream_error_isolates(service):
    service.add_stream(StreamSpec("good", "sim", camera="cam_front"))
    service.add_stream(StreamSpec("bad", "sim", camera="no_such_camera"))
    assert _wait_frame(service, "good") is not None
    time.sleep(0.2)
    assert service.latest("bad") is None  # dead stream -> None, service alive
    assert service.latest("good") is not None


def test_remove_stream_and_clean_shutdown(single_fixed_scene):
    svc = RenderService()
    svc.register_source("sim", single_fixed_scene.model)
    svc.start()
    svc.add_stream(StreamSpec("view", "sim", camera="cam_front"))
    assert _wait_frame(svc, "view") is not None
    svc.remove_stream("view")
    assert svc.latest("view") is None
    svc.stop()  # closes renderers in the render thread; idempotent
    svc.stop()


def test_sim_camera_interface(single_fixed_scene):
    svc = RenderService()
    svc.start()
    scene = single_fixed_scene
    cell = SimWorkcell(scene, make_config("single_fixed_tabletop", ["arm0"]), svc)
    try:
        assert set(cell.cameras) == {"cam_front", "arm0_wrist_cam"}
        cam = cell.cameras["arm0_wrist_cam"]
        assert isinstance(cam, CameraInterface)
        assert isinstance(cam, SimCamera)
        assert cam.resolution == (640, 480) and cam.fps == 30.0
        cell.start()  # starts cameras + feeds submit_state each tick
        deadline = time.monotonic() + 5.0
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = cam.latest()
            time.sleep(0.02)
        assert frame is not None
        assert frame.camera_id == "arm0_wrist_cam"
        assert frame.rgb.shape == (480, 640, 3)
    finally:
        cell.stop()
        svc.stop()


def test_depth_streams_rejected(service):
    with pytest.raises(NotImplementedError):
        service.add_stream(StreamSpec("d", "sim", camera=None, depth=True))
