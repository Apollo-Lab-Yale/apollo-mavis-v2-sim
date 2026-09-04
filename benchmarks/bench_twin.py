"""Benchmark DigitalTwin.check + the mj_geomDistance clearance sweep.

Usage:

    uv run python benchmarks/bench_twin.py

Reference numbers (research, this machine, 3 arms + grippers, gap
inflation): check ~=240 us/tick at home, ~=750 us adversarial; arm0 x arm1
mj_geomDistance sweep (289 pairs, distmax=0.2) ~=0.29 ms. Acceptance
(phase-03): 3-arm check <= 1 ms/tick. Exit code 1 on failure.
"""

from __future__ import annotations

import sys
import time

import mujoco
import numpy as np

from apollo_mavis_v2_sim import REGISTRY
from apollo_mavis_v2_sim.twin import DigitalTwin

CHECK_THRESHOLD_MS = 1.0
N_CHECK = 2000


def _home_q(twin: DigitalTwin) -> dict[str, np.ndarray]:
    key0 = twin.model.key(0)
    return {
        arm_id: np.array(key0.qpos[a.qpos_adr]) for arm_id, a in twin.addr.arms.items()
    }


def _adversarial_q(twin: DigitalTwin) -> dict[str, np.ndarray]:
    """Rails converged + arms leaned toward each other: ~66 near contacts
    (matches the research adversarial profile of ~72)."""
    q = _home_q(twin)
    ids = sorted(q)
    for i, arm_id in enumerate(ids):
        qa = q[arm_id]
        qa[0] = 0.8 if i % 2 == 0 else -0.8  # yaw toward the neighbour
        qa[1] = 0.45  # lean forward
        qa[3] = 0.45
        if qa.shape[0] == 8:
            qa[7] = 0.48 if i % 2 == 0 else 0.17
    return q


def bench_check(twin: DigitalTwin, q: dict[str, np.ndarray], label: str) -> float:
    twin.check(q)  # warmup + contact count
    ncon = 0
    twin.data.qpos[:] = twin._q_meas_full
    for arm_id, qa in q.items():
        twin.data.qpos[twin.addr[arm_id].qpos_adr] = qa
    mujoco.mj_kinematics(twin.model, twin.data)
    mujoco.mj_collision(twin.model, twin.data)
    ncon = twin.data.ncon
    twin.data.qpos[:] = twin._q_meas_full
    t0 = time.perf_counter()
    for _ in range(N_CHECK):
        twin.check(q)
    us = (time.perf_counter() - t0) / N_CHECK * 1e6
    print(f"twin.check {label:<14} {us:7.1f} us/tick  ({ncon} contacts)")
    return us / 1e3


def bench_sweep(twin: DigitalTwin) -> None:
    ids = sorted(twin.addr.arms)
    g0 = set(int(g) for g in twin.addr[ids[0]].geom_ids)
    g1 = set(int(g) for g in twin.addr[ids[1]].geom_ids)
    pairs = [
        (a, b)
        for a, b in twin.monitored_pairs
        if (a in g0 and b in g1) or (a in g1 and b in g0)
    ]
    mujoco.mj_kinematics(twin.model, twin.data)
    t0 = time.perf_counter()
    n_rep = 200
    for _ in range(n_rep):
        for a, b in pairs:
            mujoco.mj_geomDistance(twin.model, twin.data, a, b, 0.2, None)
    ms = (time.perf_counter() - t0) / n_rep * 1e3
    print(f"mj_geomDistance sweep arm0 x arm1: {len(pairs)} pairs, {ms:.3f} ms")
    t0 = time.perf_counter()
    for _ in range(100):
        twin.clearance(0.05)
    print(f"twin.clearance(0.05) full monitored set: "
          f"{(time.perf_counter() - t0) / 100 * 1e3:.3f} ms")


def main() -> int:
    twin = DigitalTwin(REGISTRY.build("triple_rail_row"), inflation_m=0.0125)
    fails: list[str] = []
    home_ms = bench_check(twin, _home_q(twin), "home")
    adv_ms = bench_check(twin, _adversarial_q(twin), "adversarial")
    for label, ms in (("home", home_ms), ("adversarial", adv_ms)):
        if ms > CHECK_THRESHOLD_MS:
            fails.append(f"check {label}: {ms:.3f} ms > {CHECK_THRESHOLD_MS} ms")
    bench_sweep(twin)
    for f in fails:
        print("FAIL:", f)
    print("bench_twin:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
