"""Benchmark mj_step and offscreen rendering on the 3-arm scene.

Usage (EGL headless):

    MUJOCO_GL=egl uv run python benchmarks/bench_step_render.py

Thresholds (phase-02 acceptance; measured ~38 us step / ~0.61 ms frame on
the target RTX 4090 box): 3-arm mj_step < 100 us, 100 Hz tick (5 substeps)
< 0.5 ms, 640x480 RGB frame < 2 ms. Exit code 1 if any threshold fails.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco GL init; harness-owned

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from apollo_xarm7_sim import REGISTRY  # noqa: E402

STEP_THRESHOLD_US = 100.0
TICK_THRESHOLD_MS = 0.5
FRAME_THRESHOLD_MS = 2.0


def bench_step(model: mujoco.MjModel, n: int = 5000) -> tuple[float, float]:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data, nstep=200)  # warmup
    t0 = time.perf_counter()
    mujoco.mj_step(model, data, nstep=n)
    step_us = (time.perf_counter() - t0) / n * 1e6

    mujoco.mj_resetDataKeyframe(model, data, 0)
    n_ticks = 500
    t0 = time.perf_counter()
    for _ in range(n_ticks):
        mujoco.mj_step(model, data, nstep=5)
    tick_ms = (time.perf_counter() - t0) / n_ticks * 1e3
    return step_us, tick_ms


def bench_render(model: mujoco.MjModel, width: int, height: int, n: int) -> float:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    out = np.empty((height, width, 3), dtype=np.uint8)
    try:
        renderer.update_scene(data, camera="cam_front")
        renderer.render(out=out)  # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            renderer.update_scene(data, camera="cam_front")
            renderer.render(out=out)
        return (time.perf_counter() - t0) / n * 1e3
    finally:
        renderer.close()


def main() -> int:
    built = REGISTRY.build("triple_rail_row")
    model = built.model
    print(f"scene: {built.meta.id} (nq={model.nq}, nu={model.nu}, nmesh={model.nmesh})")

    step_us, tick_ms = bench_step(model)
    print(f"mj_step (3 arms):        {step_us:8.1f} us/step   (threshold < {STEP_THRESHOLD_US} us)")
    print(f"100 Hz tick (5 substeps):{tick_ms:8.3f} ms/tick   (threshold < {TICK_THRESHOLD_MS} ms)")

    frame_ms = bench_render(model, 640, 480, 300)
    print(f"render 640x480 RGB:      {frame_ms:8.3f} ms/frame  "
          f"(threshold < {FRAME_THRESHOLD_MS} ms, ~{1000.0 / frame_ms:.0f} FPS)")
    hd_ms = bench_render(model, 1920, 1080, 50)
    print(f"render 1920x1080 RGB:    {hd_ms:8.3f} ms/frame  (info only)")

    ok = (
        step_us < STEP_THRESHOLD_US
        and tick_ms < TICK_THRESHOLD_MS
        and frame_ms < FRAME_THRESHOLD_MS
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
