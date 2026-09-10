# apollo-mavis-v2-sim

MuJoCo simulation workcell for the apollo-mavis-v2 stack — the digital twin of
the MAVIS v2 cell (two xArm7 arms on linear tracks: grip arm with gripper + wrist
camera, view arm with wrist camera only; scene `mavis_v2`). Implements the core
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
uv run python -m apollo_mavis_v2_sim.tools.guardrail_check --all   # safety CI
```

`MUJOCO_GL=egl` must be exported **before** the process imports mujoco's
rendering path; this package never touches environment variables — the
runtime entrypoint (or your shell) is responsible.

## Scenes

```python
from apollo_mavis_v2_sim.scenes.registry import REGISTRY
built = REGISTRY.build("triple_rail_row")
print(built.model.nq, built.meta.arm_ids)
```

Scene descriptors live in `src/apollo_mavis_v2_sim/assets/scenes/*.yaml`; the
composed `spec.to_xml()` is persisted with every episode for replay. After
adding or editing a scene run `uv run python -m apollo_mavis_v2_sim.tools.gen_asset_manifest`
(scene YAMLs are hashed into `ASSET_MANIFEST.json`).

| scene | arms | listed | notes |
|---|---|---|---|
| `single_rail`, `single_fixed_tabletop` | 1 | hidden | dev defaults (runtime `configs/sim.yaml`) |
| `dual_rail_tabletop`, `dual_mixed`, `triple_rail_row` | 2–3 | hidden | composition coverage |
| `guardrail_env`, `guardrail_face`, `guardrail_rail` | 1–2 | hidden | safety CI cells |
| `mavis_v2` | 2 | **APOLLO MAVIS V2 Digital Twin** | **the Apollo lab cell** (measured 2026-09-02): camera-only arm `view` on the outer rail, gripper arm `grip` 39.0 cm inward, 1.215 × 0.62 m table, 0.16 × 0.16 × 0.24 m obstacle at the left end of the channel — twin reference for the real arms; runtime `configs/mavis_v2.yaml` |
| `mavis_v2_kitchen` | 2 | hidden — title "APOLLO MAVIS V2 Kitchen" | `mavis_v2` + the lab kitchen measured 2026-09-09 (GE GDE21ESKSS fridge, 30-inch GE range, counter, upper cabinets, wall as boxes; four tagStandard41h12 AprilTag plates, textured, non-collidable; `graspable` handles): the twin of the room the cell stands in. Selected explicitly in config (`digital_twin_scene` / `sim_scene`), never listed by `GET /api/scenes` |

`REGISTRY.list()` returns only the visible scene (`hidden: true` scenes are filtered
unless `list(include_hidden=True)`); `descriptor()/meta()/build()` resolve every id.
`gripper: none` + `wrist_cam: true` composes a camera-only arm (TCP at the
flange, D435 + stand collidable); such an arm may add `microphone: true` — or the
hardware twin passes `SceneOverrides(microphones={"view": True})` — for the RØDE
NT-USB Mini collision cylinder in front of the lens (8 cm diameter, 14 cm past the
camera plane; docs/design/03-sim §3). `allowed_pairs:` declares structural
near-contacts the twin must not treat as hazards (docs/design/03-sim §4.2).

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
| Guardrail CI (`--all`, 15 runs incl. `mavis_v2_rail_sweep`) | see `tests/test_guardrail.py` | < 30 s |

## Fidelity notes

- The menagerie actuator gains (kp 1500/1000/800) have **not** been
  identified against the real arm — kept as-is; calibration is a phase-09
  topic.
- The rail travel is corrected to the real 0.65 m (mavis models 0.74 m);
  the rail/platform mesh geometry is unverified against real hardware.
