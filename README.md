# apollo-xarm7-sim

MuJoCo simulation workcell for the apollo-xarm7 stack. Implements the core
interfaces (`WorkcellInterface`, `ArmInterface`, `CameraInterface`) over a
scene composed at runtime from vendored `mujoco_menagerie` assets, so the
runtime treats sim exactly like hardware.

Phase 02 scope: vendored assets, `xarm7_on_rail.xml`/`xarm7_fixed.xml` child
models, scene registry + `MjSpec` composition, `SimWorkcell` (monotonic-paced
stepping thread), gripper mapping, sim cameras, and the offscreen EGL
`RenderService`.

Phase 03 scope: `MinkIKSolver` (mink QP differential IK + four
RelaxedIK-derived refinements), `DigitalTwin` (kinematic-only gap-inflated
collision mirror), `ResetPlanner` (per-arm sequential RRT-Connect), and the
safety-layer CI script `tools/guardrail_check.py` (11-safety §5.1).

## Usage

```bash
uv sync
uv run pytest                    # CPU-only tests
MUJOCO_GL=egl uv run pytest      # + EGL rendering tests
MUJOCO_GL=egl uv run python benchmarks/bench_step_render.py
uv run python benchmarks/bench_ik.py
uv run python benchmarks/bench_twin.py
uv run python -m apollo_xarm7_sim.tools.guardrail_check --all   # safety CI
```

`MUJOCO_GL=egl` must be exported **before** the process imports mujoco's
rendering path; this package never touches environment variables — the
runtime entrypoint (or your shell) is responsible.

## Scenes

```python
from apollo_xarm7_sim.scenes.registry import REGISTRY
built = REGISTRY.build("triple_rail_row")
print(built.model.nq, built.meta.arm_ids)
```

Scene descriptors live in `src/apollo_xarm7_sim/assets/scenes/*.yaml`; the
composed `spec.to_xml()` is persisted with every episode for replay.

## Measured numbers (Threadripper PRO 5975WX + RTX 4090, single thread)

Phase-03 benchmarks, 2026-09-01 (`bench_ik.py` / `bench_twin.py`):

| Benchmark | Measured | Threshold |
|---|---|---|
| IK servo step, 7-DoF fixed | 185 µs mean / 207 µs p99, 0.054 mm tracking | ≤ 0.5 ms mean, ≤ 0.1 mm |
| IK servo step, 8-DoF rail | 182 µs mean / 205 µs p99 | ≤ 0.5 ms mean |
| IK servo + collision rows (1 arm) | 305 µs mean / 339 µs p99 | ≤ 0.5 ms mean |
| IK servo, 3-arm scene + collision rows | 544 µs mean / 590 µs p99 | p99 < 1 ms (03-sim §14) |
| IK one-shot far target (rail chain) | 1.9–2.3 ms, < 0.1 mm, rail auto-placed | ≤ 10 ms |
| Twin check, 3 arms, home | 14 µs/tick (0 contacts — structural pairs excluded) | ≤ 1 ms |
| Twin check, 3 arms, adversarial | 744 µs/tick (66 contacts) | ≤ 1 ms |
| `mj_geomDistance` arm0×arm1 sweep | 360 pairs, 0.15 ms (distmax 0.2) | ~0.29 ms ref |
| Guardrail CI (`--all`, 12 runs) | ~12.5 s virtual-tick | < 30 s |

## Fidelity notes

- The menagerie actuator gains (kp 1500/1000/800) have **not** been
  identified against the real arm — kept as-is; calibration is a phase-09
  topic.
- The rail travel is corrected to the real 0.65 m (mavis models 0.74 m);
  the rail/platform mesh geometry is unverified against real hardware.
