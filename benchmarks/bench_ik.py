"""Benchmark MinkIKSolver: servo warm-start loop + one-shot far target.

Usage:

    uv run python benchmarks/bench_ik.py

Reference numbers (research, this machine): ~116 us/step 7-DoF, ~113 us/step
8-DoF rail (~8.6/8.8 kHz); one-shot far target 18-28 QP steps, 2-4 ms,
<0.1 mm. Acceptance thresholds (phase-03): servo mean <= 0.5 ms/step with
tracking error <= 0.1 mm; one-shot <= 10 ms. Exit code 1 on failure.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from apollo_mavis_v2_core import Pose, se3

from apollo_mavis_v2_sim import REGISTRY
from apollo_mavis_v2_sim.ik import IKParams, MinkIKSolver, default_collision_pairs

SERVO_MEAN_THRESHOLD_MS = 0.5
SERVO_TRACK_THRESHOLD_MM = 0.1
ONESHOT_THRESHOLD_MS = 10.0
N_SERVO = 2000


def _tcp_pose(solver: MinkIKSolver, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
    a = solver.scene.addressing[arm_id]
    d = solver.configuration.data
    return (
        d.site_xpos[a.tcp_site_id].copy(),
        se3.mat_to_quat(d.site_xmat[a.tcp_site_id].reshape(3, 3)),
    )


def bench_servo(
    scene_id: str,
    arm_id: str,
    label: str,
    collision: bool,
    p99_threshold_ms: float | None = None,
) -> list[str]:
    """Single-arm cases: mean <= 0.5 ms (brief). Multi-arm + collision case:
    p99 < 1 ms — the design's own 3-arm perf bar (03-sim §14)."""
    scene = REGISTRY.build(scene_id)
    pairs = default_collision_pairs(scene) if collision else None
    solver = MinkIKSolver(scene, IKParams(), collision_pairs=pairs)
    pos0, quat0 = _tcp_pose(solver, arm_id)
    times = np.empty(N_SERVO)
    errs = np.empty(N_SERVO)
    for k in range(N_SERVO):
        t = k * 0.01
        target = Pose(
            pos0
            + [
                0.05 * np.sin(2 * np.pi * 0.2 * t),
                0.05 * (1 - np.cos(2 * np.pi * 0.2 * t)),
                0.02 * np.sin(2 * np.pi * 0.1 * t),
            ],
            quat0,
        )
        t0 = time.perf_counter()
        r = solver.solve(arm_id, target, None)
        times[k] = time.perf_counter() - t0
        errs[k] = r.pos_err_m
    mean_ms = float(times.mean() * 1e3)
    p99_ms = float(np.percentile(times, 99) * 1e3)
    track_mm = float(errs[200:].max() * 1e3)
    line = (
        f"servo {label:<28} mean {mean_ms * 1e3:6.0f} us  p99 {p99_ms * 1e3:6.0f} us "
        f"({1.0 / (mean_ms * 1e-3):,.0f} Hz)  track {track_mm:.4f} mm"
    )
    print(line)
    fails = []
    if p99_threshold_ms is not None:
        if p99_ms > p99_threshold_ms:
            fails.append(f"{label}: servo p99 {p99_ms:.3f} ms > {p99_threshold_ms}")
    elif mean_ms > SERVO_MEAN_THRESHOLD_MS:
        fails.append(f"{label}: servo mean {mean_ms:.3f} ms > {SERVO_MEAN_THRESHOLD_MS}")
    if track_mm > SERVO_TRACK_THRESHOLD_MM:
        fails.append(f"{label}: tracking {track_mm:.4f} mm > {SERVO_TRACK_THRESHOLD_MM}")
    return fails


def bench_oneshot(scene_id: str, arm_id: str) -> list[str]:
    scene = REGISTRY.build(scene_id)
    solver = MinkIKSolver(scene, IKParams(), collision_pairs=None)
    a = scene.addressing[arm_id]
    q0 = np.array(solver.configuration.data.qpos[a.qpos_adr])
    pos0, quat0 = _tcp_pose(solver, arm_id)
    fails: list[str] = []
    for dy in (0.45, 0.3, -0.25):
        target = Pose(pos0 + [0.0, dy, 0.05], quat0)
        t0 = time.perf_counter()
        r = solver.solve_to_convergence(arm_id, target, q0)
        ms = (time.perf_counter() - t0) * 1e3
        print(
            f"one-shot dy={dy:+.2f}: {ms:5.2f} ms  err {r.pos_err_m * 1e3:.4f} mm / "
            f"{r.rot_err_rad:.5f} rad  rail -> {r.q[7]:.3f} m"
        )
        if ms > ONESHOT_THRESHOLD_MS:
            fails.append(f"one-shot dy={dy}: {ms:.2f} ms > {ONESHOT_THRESHOLD_MS}")
    return fails


def main() -> int:
    fails: list[str] = []
    fails += bench_servo("single_fixed_tabletop", "arm0", "7-DoF fixed", False)
    fails += bench_servo("single_rail", "arm0", "8-DoF rail", False)
    fails += bench_servo("single_rail", "arm0", "8-DoF rail +collision", True)
    fails += bench_servo(
        "triple_rail_row", "arm1", "8-DoF (3-arm scene)", True, p99_threshold_ms=1.0
    )
    fails += bench_oneshot("single_rail", "arm0")
    for f in fails:
        print("FAIL:", f)
    print("bench_ik:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
