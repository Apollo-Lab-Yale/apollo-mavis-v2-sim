# apollo-xarm7-sim

MuJoCo simulation workcell for the apollo-xarm7 stack. Implements the core
interfaces (`WorkcellInterface`, `ArmInterface`, `CameraInterface`) over a
scene composed at runtime from vendored `mujoco_menagerie` assets, so the
runtime treats sim exactly like hardware.

Phase 02 scope: vendored assets, `xarm7_on_rail.xml`/`xarm7_fixed.xml` child
models, scene registry + `MjSpec` composition, `SimWorkcell` (monotonic-paced
stepping thread), gripper mapping, sim cameras, and the offscreen EGL
`RenderService`. IK / digital twin / planner land in phase 03.

## Usage

```bash
uv sync
uv run pytest                    # CPU-only tests
MUJOCO_GL=egl uv run pytest      # + EGL rendering tests
MUJOCO_GL=egl uv run python benchmarks/bench_step_render.py
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

## Fidelity notes

- The menagerie actuator gains (kp 1500/1000/800) have **not** been
  identified against the real arm — kept as-is; calibration is a phase-09
  topic.
- The rail travel is corrected to the real 0.65 m (mavis models 0.74 m);
  the rail/platform mesh geometry is unverified against real hardware.
