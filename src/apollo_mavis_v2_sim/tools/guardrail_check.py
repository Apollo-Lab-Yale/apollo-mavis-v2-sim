"""Safety-layer CI regression — spec: 11-safety §5.1 (BINDING).

A ``SimWorkcell`` plays the real robot while the full hardware-mode safety
stack runs against it (``safety_debug``): a separate ``DigitalTwin`` built
from the same scene id, a hold-last-safe ``SafetyGate`` (§7 semantics — the
reference implementation runtime's phase-05 gate must match), and IK
collision-avoidance rows. Headless, virtual-tick paced (``step_virtual``,
no sleeps), fixed seeds; exit 0 = all assertions pass.

Run:  uv run python -m apollo_mavis_v2_sim.tools.guardrail_check --all
                                                    [--no-ik-avoidance]

Scenarios: ``env_table_descend`` / ``env_pedestal_sweep`` (arm↔environment),
``cross_arm_head_on`` / ``cross_arm_rail_converge`` (arm↔arm),
``mavis_v2_rail_sweep`` (the lab cell's Manipulation Arm ``grip`` along the
channel into its obstacle) and ``mavis_v2_rail_sweep_mic`` (same, with the hardware
twin's microphone body on the Perception Arm ``view`` via
``GuardrailScenario.overrides``). A scenario may start from its own posture
(``GuardrailScenario.start_q``, core order, rail LAST) instead of the scene
keyframe: the mavis_v2 keyframe is the cell's folded initial state (rails at
opposite ends), from which a +X sweep reaches nothing, so both mavis scenarios
teleport to a lowered ready pose in the channel first. Assertion contract A1–A5 in
``_assert_*`` below. Ground truth = the *physics* model with zero inflation: any
sim contact ``dist <= 0`` between geoms matching ``target_pair_prefixes`` is a
real-contact failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import mujoco
import numpy as np
from apollo_mavis_v2_core import (
    ArmConfig,
    CollisionEvent,
    CollisionReport,
    CommandSource,
    Pose,
    SafetyConfig,
    WorkcellConfig,
    se3,
)
from apollo_mavis_v2_core.schemas import PoseModel

from ..ik import IKParams, MinkIKSolver, default_collision_pairs
from ..scenes import REGISTRY, SceneOverrides
from ..twin import DigitalTwin
from ..workcell import CTRL_DT, SimWorkcell

DT = CTRL_DT  # 100 Hz virtual tick
SAFETY_DEBUG_INFLATION_M = 0.025  # un-surveyed-cell delta (11-safety §6.2)
ESCAPE_EPS_M = 1e-5  # strict-opening margin, §7 step 6
TARGET_LEASH_POS_M = 0.03  # teleop target leash around FK(q_meas) (04-runtime §6):
TARGET_LEASH_ROT_RAD = 0.5  # a gate hold must not let the target run away


@dataclass(frozen=True)
class GuardrailScenario:
    scenario_id: str
    scene_id: str
    driven_arm: str
    twist: np.ndarray  # (6,) constant world-frame twist via the teleop path
    target_pair_prefixes: tuple[str, str]  # e.g. ("arm0_", "table")
    max_ticks: int = 2000  # 20 s virtual @ 100 Hz
    escape_after_block_ticks: int = 50  # then reverse the twist
    graze_twist: np.ndarray | None = None  # tangential variant (A5, IK ON)
    graze_ticks: int = 700
    # Build overrides applied to BOTH the played robot and the twin (e.g. the
    # hardware digital twin's microphone body: SceneOverrides(microphones=...)).
    overrides: SceneOverrides | None = None
    # Scenario-local start posture per arm (CORE order: j1..j7, rail LAST), written
    # into the played robot before the run; arms not listed keep the scene keyframe.
    start_q: Mapping[str, np.ndarray] | None = None


@dataclass
class ScenarioResult:
    scenario_id: str
    variant: str  # "main" | "graze"
    ik_avoidance: bool
    ground_truth_contacts: int = 0
    blocked_events: list[tuple[int, CollisionEvent]] = field(default_factory=list)
    cleared_tick: int | None = None
    block_tick: int | None = None
    reversal_tick: int | None = None
    hold_max_dev: float = 0.0  # max |q_sent - q_sent(t_block)| while held
    first_block_clearance: float | None = None
    first_block_pair: tuple[str, str] | None = None
    recompute_diff: float | None = None  # A2 mj_geomDistance cross-check
    escape_clearances: list[float] = field(default_factory=list)
    resumed: bool = False
    ticks_run: int = 0


@dataclass
class _GateDecision:
    q_out: dict[str, np.ndarray]
    blocked: bool
    report: CollisionReport
    events: list[CollisionEvent]


class SafetyGate:
    """Hold-last-safe command gate — reference semantics of 11-safety §7.

    No staleness rung here (virtual sim states are always fresh); phase-05's
    runtime gate adds §7 step 1 and must otherwise match this behaviour.
    """

    def __init__(self, twin: DigitalTwin, cfg: SafetyConfig) -> None:
        self.twin = twin
        self.cfg = cfg
        self._last_safe: dict[str, np.ndarray] | None = None
        self._blocked = False
        self._block_pairs: set[tuple[str, str]] = set()

    def _unblock_threshold(self) -> float:
        return self.twin.inflation_m + self.cfg.min_clearance_m + self.cfg.hysteresis_m

    def filter(
        self,
        q_cmd: dict[str, np.ndarray],
        q_meas: dict[str, np.ndarray],
        source: CommandSource = CommandSource.TELEOP,
        ts: float = 0.0,
    ) -> _GateDecision:
        if self._last_safe is None:  # session start: seed from measured
            self._last_safe = {a: np.array(q) for a, q in q_meas.items()}
        report = self.twin.check(q_cmd)  # §7 step 2-3 (COMMANDED config)
        viols: dict[tuple[str, str], float] = {}
        for ev in report.violations:
            viols[ev.pairs[0]] = ev.min_clearance_m
        if self._blocked:  # hysteresis band (δ, δ+hyst]: invisible to contacts
            for pair in self._block_pairs - set(viols):
                d = self.twin.pair_distance(pair, q_cmd, self._unblock_threshold() + 0.01)
                if d < self._unblock_threshold():
                    viols[pair] = d
        if not viols:  # §7 step 4
            events = []
            if self._blocked:
                self._blocked = False
                self._block_pairs = set()
                events.append(
                    CollisionEvent(
                        ts=ts, kind="cleared", pairs=[], dists_m=[],
                        min_clearance_m=self._unblock_threshold(), source=source,
                    )
                )
            self._last_safe = {a: np.array(q) for a, q in q_cmd.items()}
            return _GateDecision(dict(q_cmd), False, report, events)

        # §7 steps 5-7: per-arm escape test, else hold-last-safe.
        offending = {a for pair in viols for a in self.twin._arms_of_pair(pair)}
        # Exact clearances via mj_geomDistance at q_cmd and q_meas (also
        # makes event dists A2-recomputable to machine precision).
        d_cmd = {p: self.twin.pair_distance(p, q_cmd, 0.5) for p in viols}
        d_meas = {p: self.twin.pair_distance(p, None, 0.5) for p in viols}
        meas_viols = {
            p for p, d in d_meas.items() if d < self.twin.inflation_m + self.cfg.min_clearance_m
        }
        q_out: dict[str, np.ndarray] = {}
        for arm_id, q in q_cmd.items():
            if arm_id not in offending:
                q_out[arm_id] = q  # step 2 checked all arms jointly
                continue
            arm_pairs = [p for p in viols if arm_id in self.twin._arms_of_pair(p)]
            opens = all(d_cmd[p] >= d_meas[p] + ESCAPE_EPS_M for p in arm_pairs)
            no_new = all(p in meas_viols for p in arm_pairs)  # step 6, clause 2
            if opens and no_new:  # step 6 (T8 escape)
                q_out[arm_id] = q
                self._last_safe[arm_id] = np.array(q)
            else:  # step 7: hold
                q_out[arm_id] = self._last_safe[arm_id]
        events = []
        new_pairs = set(viols) - self._block_pairs
        if not self._blocked or new_pairs:  # rising edge / pair-set change
            pairs = sorted(viols)
            events.append(
                CollisionEvent(
                    ts=ts,
                    kind="penetration" if min(d_cmd.values()) <= 0.0 else "blocked",
                    pairs=pairs,
                    dists_m=[d_cmd[p] for p in pairs],
                    min_clearance_m=min(d_cmd.values()),
                    source=source,
                    arm_ids=sorted(offending),
                )
            )
        self._blocked = True
        self._block_pairs = set(viols)
        return _GateDecision(q_out, True, report, events)


def _tw(*v: float) -> np.ndarray:
    return np.array(v, dtype=np.float64)


# mavis_v2 sweep start posture (CORE order: j1..j7, rail LAST) — the cell's
# pre-2026-09-04 keyframe, kept here so the scenario premise (03-sim §4.3 /
# §11) survives the keyframe becoming the folded initial state. Both rails at
# q = 0.5974 (the −X end): 'grip' elbow-up in the channel, tool straight down,
# TCP 4.5 cm below the obstacle top and ~0.9 m to its −X; 'view' parked ~1.4 m
# up over the workspace, D435 looking down (20 deg toward +X), clear of the
# sweep even with the microphone body on.
MAVIS_SWEEP_START_Q: Mapping[str, np.ndarray] = {
    "grip": _tw(4.6356, -0.4748, 0.6498, 0.3665, 0.3977, 0.7955, -0.3882, 0.5974),
    "view": _tw(4.7376, -1.4648, -0.0307, 0.8091, -0.0409, 1.917, 1.5726, 0.5974),
}


SCENARIOS: dict[str, GuardrailScenario] = {
    s.scenario_id: s
    for s in (
        GuardrailScenario(
            scenario_id="env_table_descend",
            scene_id="guardrail_env",
            driven_arm="arm0",
            twist=_tw(0.0, 0.0, -0.12, 0.0, 0.0, 0.0),
            target_pair_prefixes=("arm0_", "table"),
            graze_twist=_tw(0.0, 0.06, -0.06, 0.0, 0.0, 0.0),
            graze_ticks=340,
        ),
        GuardrailScenario(
            scenario_id="env_pedestal_sweep",
            scene_id="guardrail_env",
            driven_arm="arm0",
            twist=_tw(0.12, 0.0, 0.0, 0.0, 0.0, 0.0),
            target_pair_prefixes=("arm0_", "pedestal"),
            graze_twist=_tw(0.09, 0.085, 0.0, 0.0, 0.0, 0.0),
            graze_ticks=300,
        ),
        GuardrailScenario(
            scenario_id="cross_arm_head_on",
            scene_id="guardrail_face",
            driven_arm="arm0",
            twist=_tw(0.12, 0.0, 0.0, 0.0, 0.0, 0.0),
            target_pair_prefixes=("arm0_", "arm1_"),
            graze_twist=_tw(0.08, 0.10, 0.0, 0.0, 0.0, 0.0),
            graze_ticks=300,
        ),
        GuardrailScenario(
            scenario_id="cross_arm_rail_converge",
            scene_id="guardrail_rail",
            driven_arm="arm0",
            twist=_tw(0.0, 0.12, 0.0, 0.0, 0.0, 0.0),
            target_pair_prefixes=("arm0_", "arm1_"),
            graze_twist=_tw(0.0, 0.10, 0.02, 0.0, 0.0, 0.0),
            graze_ticks=1100,
        ),
        # Lab cell (mavis_v2): the untouchable obstacle stands at the +X end of
        # the channel (operator's left). The scene keyframe is the cell's INITIAL
        # STATE (both arms folded at the xArm zero, rails at opposite ends), which
        # a +X sweep cannot bring into contact with anything, so the scenario
        # starts from MAVIS_SWEEP_START_Q instead: the Manipulation Arm 'grip' at
        # the −X end (operator's right, rail q ~ 0.597 — travel is reversed by the
        # yaw +90 remount that turns each rail's plate to −Y) in an elbow-up ready
        # pose with the TCP 4.5 cm below the obstacle top, the Perception Arm
        # 'view' parked high over the workspace and out of the sweep. The TCP is
        # driven +X in world along the channel — the IK carries the rail toward
        # q=0 — until the obstacle's face is reached. Graze: same sweep while
        # rising 3 cm/s, clearing the top by ~8 cm.
        GuardrailScenario(
            scenario_id="mavis_v2_rail_sweep",
            scene_id="mavis_v2",
            driven_arm="grip",
            twist=_tw(0.12, 0.0, 0.0, 0.0, 0.0, 0.0),
            target_pair_prefixes=("grip_", "obstacle"),
            graze_twist=_tw(0.12, 0.0, 0.03, 0.0, 0.0, 0.0),
            graze_ticks=520,
            start_q=MAVIS_SWEEP_START_Q,
        ),
        # Same sweep with the deployment twin's geometry: the Perception Arm
        # carries the microphone cylinder (03-sim §3), as the hardware digital
        # twin builds it (ArmConfig.microphone -> SceneOverrides).
        GuardrailScenario(
            scenario_id="mavis_v2_rail_sweep_mic",
            scene_id="mavis_v2",
            driven_arm="grip",
            twist=_tw(0.12, 0.0, 0.0, 0.0, 0.0, 0.0),
            target_pair_prefixes=("grip_", "obstacle"),
            graze_twist=_tw(0.12, 0.0, 0.03, 0.0, 0.0, 0.0),
            graze_ticks=520,
            overrides=SceneOverrides(microphones={"view": True}),
            start_q=MAVIS_SWEEP_START_Q,
        ),
    )
}


def _build_real_robot_scene(scene_id: str, overrides: SceneOverrides | None = None):
    """Scene compiled with hardware-grade servo fidelity for the played robot.

    The menagerie position actuators sag 1-2 cm under gravity and the mavis
    rail spring-servo (k=50) lags ~0.12 m at approach speed; a real xArm7 +
    linear track hold commanded positions stiffly. Compile-time spec edits
    (``body_gravcomp`` is ignored when assigned post-compile): gravity
    compensation on every body + a 40x stiffer rail servo (same ctrl ==
    position equilibrium). Twin/IK still use the stock scene — only the
    "real" cell changes.
    """
    from ..scenes.addressing import Addressing
    from ..scenes.builder import BuiltScene

    scene = REGISTRY.build(scene_id, overrides)
    spec = scene.spec
    for body in spec.bodies:
        body.gravcomp = 1.0
    for joint in spec.joints:
        if joint.name.endswith("rail_joint"):
            joint.stiffness *= 40.0
            joint.damping *= 4.0
    for act in spec.actuators:
        if act.name.endswith("_rail"):
            act.gear[0] *= 40.0
    model = spec.compile()
    return BuiltScene(scene.meta, spec, model, spec.to_xml(), Addressing(model, scene.meta))


def _sim_config(scene) -> WorkcellConfig:
    return WorkcellConfig(
        kind="sim",
        sim_scene=scene.meta.id,
        arms=[
            ArmConfig(
                id=a,
                base_in_world=PoseModel(),
                gripper="xarm" if scene.addressing[a].has_gripper else "none",
            )
            for a in scene.meta.arm_ids
        ],
        safety=SafetyConfig(
            safety_debug=True, geom_inflation_m=SAFETY_DEBUG_INFLATION_M
        ),
    )


def _start_from(cell: SimWorkcell, q_by_arm: Mapping[str, np.ndarray]) -> None:
    """Teleport the played robot to a scenario-local posture (core order, rail LAST).

    Written straight into the physics state (as a keyframe reset would be) and
    into the servo targets, so the scenario premise does not depend on the
    scene keyframe; ``step_virtual`` afterwards lets the servos settle.
    """
    data = cell._data  # same private access as _ground_truth_contacts
    for arm_id, q in q_by_arm.items():
        addr = cell.scene.addressing[arm_id]
        data.qpos[addr.qpos_adr] = q
        cell.arms[arm_id].command_joints(np.asarray(q, dtype=np.float64))
    data.qvel[:] = 0.0
    mujoco.mj_forward(cell._model, data)


def _tcp_world(twin: DigitalTwin, arm_id: str) -> Pose:
    """TCP world pose at the twin's measured config."""
    mujoco.mj_kinematics(twin.model, twin.data)
    sid = twin.addr[arm_id].tcp_site_id
    pos = np.array(twin.data.site_xpos[sid])
    quat = se3.mat_to_quat(np.array(twin.data.site_xmat[sid]).reshape(3, 3))
    return Pose(pos, quat)


def _integrate_twist(pose: Pose, twist: np.ndarray, dt: float) -> Pose:
    pos = pose.position + twist[:3] * dt
    w = twist[3:] * dt
    quat = se3.quat_mul(se3.rotvec_to_quat(w), pose.orientation) if np.any(w) else (
        pose.orientation
    )
    return Pose(pos, quat)


def _pair_matches(pair: tuple[str, str], prefixes: tuple[str, str]) -> bool:
    a, b = pair
    p1, p2 = prefixes
    return (a.startswith(p1) and b.startswith(p2)) or (
        a.startswith(p2) and b.startswith(p1)
    )


def _ground_truth_contacts(
    cell: SimWorkcell, labels: tuple[str, ...], prefixes: tuple[str, str]
) -> int:
    """Sim contacts with dist <= 0 between geoms matching the target pair."""
    data = cell._data  # physics data, zero inflation: contacts are real
    n = 0
    for i in range(data.ncon):
        if data.contact.dist[i] > 0.0:
            continue
        pair = (
            labels[int(data.contact.geom[i][0])],
            labels[int(data.contact.geom[i][1])],
        )
        if _pair_matches(pair, prefixes):
            n += 1
    return n


def run_scenario(
    s: GuardrailScenario, ik_avoidance: bool, variant: str = "main"
) -> ScenarioResult:
    from ..twin import geom_labels  # local: tools stay import-light

    graze = variant == "graze"
    twist0 = s.graze_twist if graze else s.twist
    assert twist0 is not None
    max_ticks = s.graze_ticks if graze else s.max_ticks
    sim_scene = _build_real_robot_scene(s.scene_id, s.overrides)
    cfg = SafetyConfig(safety_debug=True, geom_inflation_m=SAFETY_DEBUG_INFLATION_M)
    cell = SimWorkcell(sim_scene, _sim_config(sim_scene))
    twin = DigitalTwin(
        REGISTRY.build(s.scene_id, s.overrides), inflation_m=cfg.geom_inflation_m
    )
    ik = MinkIKSolver(
        twin.scene,
        IKParams(min_distance_m=cfg.geom_inflation_m + 0.002),
        collision_pairs=(
            default_collision_pairs(twin.scene, twin.allowed) if ik_avoidance else None
        ),
    )
    gate = SafetyGate(twin, cfg)
    labels = geom_labels(sim_scene.model)
    res = ScenarioResult(s.scenario_id, variant, ik_avoidance)

    if s.start_q:
        _start_from(cell, s.start_q)
    cell.step_virtual(20)  # let servos settle onto the start posture
    states = cell.states()
    for arm_id in cell.arms:
        ik.reset(arm_id, states[arm_id].q)
    twin.sync(states)
    target = _tcp_world(twin, s.driven_arm)
    q_hold = {a: st.q.copy() for a, st in states.items() if a != s.driven_arm}

    twist = twist0.copy()
    q_block: np.ndarray | None = None
    q_episode: np.ndarray | None = None  # A3 reference: q_sent at the rising edge
    prev_blocked = False
    block_pairs: list[tuple[str, str]] = []
    q_cleared: np.ndarray | None = None
    for tick in range(max_ticks):
        res.ticks_run = tick + 1
        states = cell.states()
        q_meas = {a: st.q for a, st in states.items()}
        twin.sync(states)
        ik.sync_passive(states)
        target = _integrate_twist(target, twist, DT)
        # Teleop clamps (04-runtime §6 / 11-safety §8): leash the integrated
        # target to the measured TCP so gate holds cannot bank drift.
        target = se3.clamp_pose_to_leash(
            target, _tcp_world(twin, s.driven_arm), TARGET_LEASH_POS_M, TARGET_LEASH_ROT_RAD
        )
        r = ik.solve(s.driven_arm, target, q_meas[s.driven_arm])
        if not np.isfinite(r.pos_err_m) or r.pos_err_m > 0.05 or r.rot_err_rad > 0.5:
            target = _tcp_world(twin, s.driven_arm)  # divergence re-anchor (§8)
        q_cmd = dict(q_hold)
        q_cmd[s.driven_arm] = r.q
        dec = gate.filter(q_cmd, q_meas, ts=tick * DT)
        q_sent = dec.q_out
        for arm_id, q in q_sent.items():
            cell.arms[arm_id].command_joints(q)
        cell.step_virtual()
        res.ground_truth_contacts += _ground_truth_contacts(
            cell, labels, s.target_pair_prefixes
        )
        for ev in dec.events:
            if ev.kind in ("blocked", "penetration"):
                res.blocked_events.append((tick, ev))
            elif ev.kind == "cleared" and res.reversal_tick is not None:
                res.cleared_tick = tick
                q_cleared = q_sent[s.driven_arm].copy()
        if res.block_tick is None and dec.blocked:
            res.block_tick = tick
            q_block = q_sent[s.driven_arm].copy()
            ev0 = res.blocked_events[0][1]
            block_pairs = list(ev0.pairs)
            matches = [
                (d, p)
                for p, d in zip(ev0.pairs, ev0.dists_m, strict=True)
                if _pair_matches(p, s.target_pair_prefixes)
            ]
            if matches:
                d0, p0 = min(matches)
                res.first_block_pair, res.first_block_clearance = p0, d0
                res.recompute_diff = abs(twin.pair_distance(p0, q_cmd, 0.5) - d0)
        # A3: hold is a hold — deviation from the rising-edge q_sent, per
        # blocked episode, while the approach twist is still applied.
        if res.reversal_tick is None:
            if dec.blocked and not prev_blocked:
                q_episode = q_sent[s.driven_arm].copy()
            if dec.blocked and q_episode is not None:
                res.hold_max_dev = max(
                    res.hold_max_dev,
                    float(np.max(np.abs(q_sent[s.driven_arm] - q_episode))),
                )
        prev_blocked = dec.blocked
        if res.block_tick is not None and res.reversal_tick is None:
            assert q_block is not None
            if tick >= res.block_tick + s.escape_after_block_ticks:
                twist = -twist0  # reverse the input: escape phase (A4)
                res.reversal_tick = tick
        if res.reversal_tick is not None and res.cleared_tick is None and block_pairs:
            # A4 series on the MEASURED config: physical clearance must not
            # shrink during the escape (commanded configs mix hold/escape
            # reference frames and are not comparable tick-to-tick).
            res.escape_clearances.append(
                min(twin.pair_distance(p, None, 0.5) for p in block_pairs)
            )
        if res.cleared_tick is not None:
            assert q_cleared is not None
            if float(np.max(np.abs(q_sent[s.driven_arm] - q_cleared))) > 1e-4:
                res.resumed = True
            if res.resumed or tick > res.cleared_tick + 50:
                break
    return res


def _assert_gate_contract(s: GuardrailScenario, res: ScenarioResult) -> list[str]:
    """A1-A4 on a main-variant run (11-safety §5.1)."""
    fails: list[str] = []
    if not res.blocked_events:
        fails.append("A1: no CollisionEvent(kind='blocked') emitted")
    if res.ground_truth_contacts:
        fails.append(f"A1: {res.ground_truth_contacts} real contact tick(s)")
    band = SAFETY_DEBUG_INFLATION_M + SafetyConfig().min_clearance_m
    if res.first_block_pair is None:
        fails.append("A2: first blocked event has no pair matching target prefixes")
    else:
        c = res.first_block_clearance
        assert c is not None and res.recompute_diff is not None
        if not 0.0 < c <= band + 1e-9:
            fails.append(f"A2: first-block clearance {c:.4f} m outside (0, {band}]")
        if res.recompute_diff > 1e-6:
            fails.append(f"A2: mj_geomDistance recompute differs by {res.recompute_diff:.2e}")
    if res.hold_max_dev > 1e-4:
        fails.append(f"A3: q_sent moved {res.hold_max_dev:.2e} while held")
    if res.cleared_tick is None:
        fails.append("A4: no 'cleared' event after twist reversal")
    else:
        assert res.reversal_tick is not None
        if res.cleared_tick - res.reversal_tick > 100:
            fails.append(
                f"A4: cleared after {res.cleared_tick - res.reversal_tick} ticks (>100)"
            )
        drops = [
            b - a
            for a, b in zip(res.escape_clearances, res.escape_clearances[1:], strict=False)
        ]
        if drops and min(drops) < -1e-4:
            fails.append(f"A4: escape clearance decreased by {-min(drops):.2e} m")
        if not res.resumed:
            fails.append("A4: motion did not resume after 'cleared'")
    return fails


def _assert_graze(res: ScenarioResult) -> list[str]:
    """A5 (IK ON): graze variants glide — zero blocks, zero real contacts."""
    fails: list[str] = []
    if res.blocked_events:
        fails.append(f"A5: graze produced {len(res.blocked_events)} blocked event(s)")
    if res.ground_truth_contacts:
        fails.append(f"A5: graze made {res.ground_truth_contacts} real contact tick(s)")
    return fails


def _run_and_report(
    s: GuardrailScenario, ik_avoidance: bool, variant: str
) -> tuple[bool, str]:
    t0 = time.perf_counter()
    res = run_scenario(s, ik_avoidance, variant)
    if variant == "graze":
        fails = _assert_graze(res)
    elif ik_avoidance:
        # IK ON, head-on push: L2 holds standoff outside the gate band; only
        # the ground truth is asserted (the gate contract is proven with L2
        # off — layer independence, A5).
        fails = (
            [f"A5: {res.ground_truth_contacts} real contact tick(s) with IK on"]
            if res.ground_truth_contacts
            else []
        )
    else:
        fails = _assert_gate_contract(s, res)
    dt_s = time.perf_counter() - t0
    tag = f"{s.scenario_id:<28} {variant:<5} ik={'on ' if ik_avoidance else 'off'}"
    line = (
        f"{'PASS' if not fails else 'FAIL'}  {tag}  "
        f"ticks={res.ticks_run:<5} blocks={len(res.blocked_events)} "
        f"gt={res.ground_truth_contacts} {dt_s:5.1f}s"
    )
    for f in fails:
        line += f"\n      - {f}"
    return not fails, line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="guardrail_check", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--all", action="store_true", help="run every scenario (CI)")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), help="run one scenario")
    parser.add_argument(
        "--no-ik-avoidance",
        action="store_true",
        help="gate-only runs: L1 must satisfy A1-A4 without L2 (A5)",
    )
    args = parser.parse_args(argv)
    if not args.all and not args.scenario:
        parser.error("choose --all or --scenario <id>")
    ids = sorted(SCENARIOS) if args.all else [args.scenario]
    t0 = time.perf_counter()
    ok = True
    for sid in ids:
        s = SCENARIOS[sid]
        runs: list[tuple[bool, str]] = [_run_and_report(s, False, "main")]
        if not args.no_ik_avoidance:
            runs.append(_run_and_report(s, True, "main"))
            runs.append(_run_and_report(s, True, "graze"))
        for passed, line in runs:
            print(line)
            ok &= passed
    print(f"total {time.perf_counter() - t0:.1f}s -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
