"""ResetPlanner — per-arm sequential RRT-Connect in joint space (03-sim §10).

Serves profile loads / ``start_from``, joint-panel ``goto`` and safety-escape
motions. Request/response shapes are core's ``PlanRequest``/``PlanResult``
(01-core §6) — this module defines no wire shape of its own. The twin is the
validity checker (planner-private ``MjData``); arms are planned one at a time
with earlier arms frozen at their *goals* and later arms at their *starts*
(11-safety §9); a failed ordering is retried in reverse. Execution lives in
runtime (phase-05): this module only produces waypoints, all of which pass
``twin.check_config`` — in the sequential context they were validated in.
Callers MUST execute the arms one after another in ``PlanResult.arm_order``
(the ``waypoints`` dict's insertion order equals it, but ``arm_order`` is the
contract): arm k's path was checked with arms < k parked at their goals and
arms > k at their starts, so moving two arms at once passes through
combinations the planner never validated (2026-09-08 real-cell incident: a
two-arm ``reset_to_initial`` executed simultaneously sat gate-blocked at
5.2 mm until the budget ran out).

**Pinched start (2026-09-09).** An arm parked INSIDE the inflation shell (a
pair closer than δ, e.g. held there by the gate) or inside the gate's
hysteresis band (closer than ``δ + hysteresis_m``: the gate keeps demanding
that such a pair OPEN while it is blocked) is first walked OUT by an escape
phase that mirrors the runtime gate's T8 rule (11-safety §7.1 steps 5-7):
every pinched pair must open by at least ``PLANNER_ESCAPE_EPS_M`` per executor
tick — judged tick by tick at the speed the plan will run at
(``PlanRequest.speed_scale``, Cartesian cap and the executor's equal-tick rule
included), not per planner sample — no new pair may enter the shell, and only
once every pinched pair is past
``δ + REARM_MARGIN_M`` does the normal RRT-Connect take over (with the gate's
margin as its predicate, below). Before that day the planner WHITELISTED the
pinched pairs instead and happily planned a straight segment that first CLOSED
them (real cell, 01:14: ``grip_right_finger`` / ``view_link3`` 2.1 → 1.1 mm;
the gate held the first step and the plan was aborted after
``plan_gate_hold_s``). The gate is the safety authority; the planner produces
paths the gate will pass.

**Only pairs the planned arm's joints can move count.** A pair the planned
arm is not part of (the OTHER arm's intra-arm pinch, the other arm against the
table, …) cannot change while this arm moves, and the gate never holds an arm
that is not in the offending set (§7.1 step 5); a pair whose two bodies hang
under the SAME set of this arm's joints (the gripper's own knuckles, the
gripper base against a finger: rigid together, only the finger joints move
them) cannot change either. Such pairs are constants for this arm: they are
neither escaped, nor reported as this arm's ``start_in_collision``, nor
allowed to block its RRT. The same-day review found the first escape
implementation failing ``no_escape`` for the FREE arm whenever the other arm
sat inside the shell (joint-panel ``goto``, rail-homing pre-positioning, and
two-arm returns whenever the heuristic put the free arm first). An arm whose
goal IS its start is left alone (no escape, no checks: nothing will be
commanded) so that the other arm can still be planned around it.

**Held carriage (2026-09-09 fuzz).** A rail slot whose start and goal coincide
(within ``RAIL_HOLD_TOL_M``) is a request for NO carriage motion and is pinned
for the RRT, the escape and the repair nudges alike (the goal snaps to the
start's rail). Before, the RRT sampled the rail like any other dof and the
runtime's two-phase return - "the joints first with each carriage HELD where it
is" (04-runtime §10.5) - slid a carriage by up to 12.6 cm on the way to the
folded posture (measured on the mavis_v2 fuzz); the rail-homing pre-positioning
plans on a carriage whose position is UNKNOWN. Joints are not held this way: the
joint-panel ``goto`` relies on the other joints for its routing.

**The RRT plans with the gate's margin (2026-09-09 fuzz).** The validity
predicate of the RRT, the direct edge and the shortcut pass is the same one the
final verification applies (every movable pair ≥ ``δ + FINE_MARGIN_M``, pairs
already that tight at the endpoints ≥ δ), not the bare "≥ δ". With the bare
predicate a doubly pinched start (cross-arm 7.2 mm + ``grip_link2`` /
``grip_link5`` 4.2 mm) was escaped correctly, then every RRT path skimmed the
intra-arm pair at 8-10 mm for a whole segment, the repair pass ran out of nudges
and all three seeds failed the same way: ``timeout`` after 0.7 s of a 5 s
budget. Attempts are now re-seeded until the request deadline (an RRT that
exhausts ``max_iters`` restarts with a new seed; after a repair failure the
direct edge and the shortcut are skipped so the retries differ), and only a
budget really spent reports ``timeout``. Two-arm requests probe both orderings
with ``ORDER_PROBE_FRACTION`` of the budget before either gets all of it, and
``LOCAL_SAMPLE_FRACTION`` of the RRT samples come from the box spanned by start
and goal (see the constants).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from apollo_mavis_v2_core import PlanRequest, PlanResult

from .scenes.addressing import N_ARM_JOINTS, Addressing
from .twin import apply_inflation

if TYPE_CHECKING:
    from .twin import DigitalTwin

RAIL_MAX_STEP_M = 0.01  # edge-check resolution for the rail dim (11-safety §9)
# A rail slot asked to move by no more than this is HELD (module docstring): below the
# runtime's carriage arrival tolerance (2 mm), so nothing observable is lost.
RAIL_HOLD_TOL_M = 0.001
REARM_MARGIN_M = 0.005  # a pinched pair is re-armed past inflation + 5 mm (> the gate's
#   hysteresis band δ + hysteresis_m, 2 mm in the cell: once re-armed the gate has cleared it)
# Strict-opening margin of the gate's T8 escape rule — MUST equal
# ``apollo_mavis_v2_runtime.safety.gate.ESCAPE_EPS_M`` (1e-5; the sim reference gate in
# ``tools/guardrail_check.py`` carries the same value). Duplicated here because sim must
# not import the runtime (dependency direction, 00-overview).
PLANNER_ESCAPE_EPS_M = 1e-5
# Escape phase candidate resolution (coarse pass: collision check for NEW violations plus
# the opening rate of every pinched pair). The chosen candidate is then re-walked at the
# executor's smallest tick (``_executor_ticks`` below), pair distances only.
ESCAPE_SUBSTEP_RAD = 0.002
ESCAPE_SUBSTEP_RAIL_M = 0.00025
_FD_STEP_RAD = 1e-3  # finite-difference probe for the opening gradient
_FD_STEP_RAIL_M = 1e-4
# Final path verification at the gate's resolution (2026-09-09). RRT edges are sampled at
# ``max_step_rad`` (0.05 rad), but the gate checks EVERY executor tick, and between two
# 0.05 rad samples an arm point can travel several centimetres: the first replay of a plan
# through the real gate found ``grip_left_finger`` / ``view_link4`` at 7.92 mm between two
# validated samples (a permanent hold, cancelled after ``plan_gate_hold_s``). The final
# path is therefore re-sampled so that no arm point moves more than ``FINE_STEP_M``
# between samples (per-joint lever bound, the rail 1:1) and every sample must clear
# ``δ + FINE_MARGIN_M``; a distance is 1-Lipschitz in the displacement, so the ticks in
# between stay ≥ δ (``FINE_MARGIN_M >= FINE_STEP_M / 2``). Pairs already tighter than that at
# the path's endpoints (the escaped start / the goal, where the arm rests) only have to stay
# ≥ δ. A grazing sample is nudged off the pair along its opening gradient; failing that the
# RRT is re-run with the next seed.
FINE_STEP_M = 0.004  # = the hardware Cartesian cap per tick (ServoLimits.max_cart_step_m)
FINE_MARGIN_M = 0.002
FINE_NUDGE_RAD = 0.01
FINE_NUDGE_RAIL_M = 0.002
_FINE_NUDGES_MAX = 12  # nudges per grazing sample
_FINE_REPAIRS_MAX = 24  # grazing samples repaired per path
# RRT re-runs (next seed) when a path cannot be repaired or is not found: as many as the
# request deadline allows (2026-09-09 fuzz; was a fixed 2 that left most of the budget unused).
# Per-joint bound on the displacement of ANY arm point per rad of that joint — the same
# numbers the hardware servo streamer uses (apollo_mavis_v2_hardware ServoLimits.lever_arm_m).
_LEVER_ARM_M = np.array([1.20, 1.20, 1.00, 0.75, 0.44, 0.30, 0.10])
# The runtime executor's tick model the escape phase is judged against (04-runtime §7;
# runtime ``control/joint_panel.py::PlanExecutor.step``). The executor walks each straight
# segment in ``ceil(ratio)`` EQUAL ticks, ``ratio`` = the segment measured in caps (every
# joint ≤ slew, the rail ≤ its cap, ``sum|dq_j| * lever_j`` ≤ the Cartesian cap, all scaled
# by the session's ``speed_scale``), so a tick is never shorter than half a full tick and,
# for a segment of many ticks, practically a full one. The gate demands ≥ ``ESCAPE_EPS_M``
# of opening on EVERY tick, and the opening of a segment is shared among its ticks, so the
# slower the session the less opening per tick: the escape is judged at the speed the plan
# will run at (``PlanRequest.speed_scale``; the runtime passes the session's, the rail-homing
# job its 10 %; the core default is ``MIN_SPEED_SCALE``, the slowest speed the runtime
# offers) - a faster session's ticks are unions of the judged ones and open more, so a plan
# judged at a slower speed stays valid at any faster one. Duplicated from the hardware
# package's ``ServoLimits`` (0.6 rad/s, 50 mm/s rail, 4 mm/tick at 100 Hz) and the executor;
# pinned by runtime ``tests/test_plan_passes_gate.py`` (sim may import neither).
HW_SLEW_RAD_PER_TICK = 0.006
HW_RAIL_M_PER_TICK = 0.0005
HW_CART_STEP_M = FINE_STEP_M
MIN_SPEED_SCALE = 0.1  # = core PlanRequest.speed_scale default; the Hardware tab's smallest pick
# Required opening per tick, as a multiple of the gate's ESCAPE_EPS_M: the margin covers the
# phase offset between the executor's ticks and the planner's samples (a tick straddling two
# samples sees the two partial openings, ≥ the margin x eps to first order; the executor's
# caps on the cell equal the constants above, so at the judged speed the ticks ARE the
# samples) and the curvature between samples (~1e-7 m at these step sizes).
ESCAPE_RATE_MARGIN = 1.25
# Two-arm requests without an explicit order (2026-09-09 fuzz): each ordering is first
# PROBED with this fraction of the per-arm budget; only then does an ordering get the full
# budget. The RRT cannot prove a dead end - an arm whose goal is clear but walled in by the
# OTHER arm's start burns its whole budget - and the reversed ordering frees it in 0.1 s.
ORDER_PROBE_FRACTION = 0.2
# RRT sampling (2026-09-09 fuzz): this fraction of the samples is drawn from the box spanned by
# the start and the goal widened by the margins below (the rest from the full joint range,
# which keeps the planner probabilistically complete). Four of the seven joints have ±2π
# limits: uniform samples over two full turns grew the trees toward postures no return ever
# needs, and a 5 rad base rotation next to the other arm took 14 s to plan.
LOCAL_SAMPLE_FRACTION = 0.7
LOCAL_SAMPLE_MARGIN_RAD = 1.5
LOCAL_SAMPLE_MARGIN_RAIL_M = 0.15


def _executor_ticks(dq: np.ndarray, speed: float) -> float:
    """Full executor ticks over the joint delta ``dq`` at speed scale ``speed``: the
    ``ratio`` of runtime ``PlanExecutor.step`` (joint slew, rail cap, Cartesian cap over
    the 7 arm joints - the rail slot is not part of the Cartesian bound there either)."""
    n = min(N_ARM_JOINTS, dq.shape[0])
    s = max(1e-6, min(float(speed), 1.0))
    ticks = float(np.max(np.abs(dq[:n]))) / (HW_SLEW_RAD_PER_TICK * s) if n else 0.0
    if dq.shape[0] > N_ARM_JOINTS:
        ticks = max(ticks, float(np.max(np.abs(dq[N_ARM_JOINTS:]))) / (HW_RAIL_M_PER_TICK * s))
    ticks = max(ticks, float(np.sum(np.abs(dq[:n]) * _LEVER_ARM_M[:n])) / (HW_CART_STEP_M * s))
    return ticks


def _tick_count(dq: np.ndarray, speed: float) -> int:
    """Ticks the executor spends on the segment ``dq`` at speed scale ``speed`` (equal
    ticks, ``ceil(ratio)``); ≥ 1."""
    return max(1, int(math.ceil(_executor_ticks(dq, speed) - 1e-9)))


@dataclass
class PlannerParams:
    """Local tuning knobs; request-level knobs ride ``PlanRequest``."""

    rail_weight: float = 4.0  # rail dims x4 in the nearest-neighbour metric
    # RRT iterations per seed before a restart. The request DEADLINE bounds the planning
    # time (attempts are re-seeded until it passes), so this only decides how often the
    # trees are thrown away: 2000 restarted a 2.7 rad base rotation past the other arm
    # every ~0.5 s and never finished it in 5 s; 10000 plans it in 2 s (2026-09-09 fuzz).
    max_iters: int = 10000
    shortcut_attempts: int = 50
    joint_vel_rad_s: float = 0.6  # time-parameterization caps (11-safety §9)
    joint_acc_rad_s2: float = 2.0
    rail_vel_m_s: float = 0.1
    rng_seed: int = 0
    escape_max_steps: int = 400  # greedy escape budget for a pinched start (§10 item 3)
    escape_random_dirs: int = 24  # random directions tried per escape step (+ gradient, axes)


@dataclass
class _ArmFailure:
    failure: str  # "goal_in_collision" | "start_in_collision" | "no_escape" | "timeout"
    failing_pair: tuple[str, str] | None


class ResetPlanner:
    """Joint-space RRT-Connect over the (inflated) twin model."""

    def __init__(
        self,
        twin: DigitalTwin,
        addr: Addressing | None = None,
        params: PlannerParams | None = None,
    ) -> None:
        self.twin = twin
        self.addr = addr if addr is not None else twin.addr
        self.params = params or PlannerParams()
        self._jnt_range: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        model = twin.model
        for arm_id, a in self.addr.arms.items():
            lo, hi = [], []
            for i in range(1, N_ARM_JOINTS + 1):
                rng = model.joint(f"{arm_id}_joint{i}").range
                lo.append(float(rng[0]))
                hi.append(float(rng[1]))
            if a.has_rail:
                rng = model.joint(f"{arm_id}_rail_joint").range
                lo.append(float(rng[0]))
                hi.append(float(rng[1]))
            self._jnt_range[arm_id] = (np.array(lo), np.array(hi))
        # Private copy of the twin model inflated by a further FINE_MARGIN_M for the final
        # path verification (the twin's own model keeps the gate's δ).
        self._fine_model = copy.copy(model)
        apply_inflation(self._fine_model, twin.inflation_m + FINE_MARGIN_M)
        # The gate's hysteresis band (SafetyConfig.hysteresis_m, carried by the twin): a
        # pair inside [δ, δ + hysteresis) at the start is escaped like a pinched one, because
        # a blocked gate demands that every pair it holds in ``_block_pairs`` - band pairs
        # included - opens on every tick (11-safety §7.1 step 6). A second inflated copy
        # detects them; it is the fine model itself when the two margins coincide.
        self.hysteresis_m = float(getattr(twin, "hysteresis_m", FINE_MARGIN_M))
        if abs(self.hysteresis_m - FINE_MARGIN_M) < 1e-12:
            self._band_model = self._fine_model
        else:
            self._band_model = copy.copy(model)
            apply_inflation(self._band_model, twin.inflation_m + self.hysteresis_m)
        self._pair_arms: dict[tuple[str, str], list[str]] = {}
        # Per arm: the joint ids of its planned dofs, and per label the subset of them on
        # the label's kinematic path (``_moves``: a pair whose two labels share the same
        # subset cannot change with this arm's motion).
        self._dof_joints: dict[str, frozenset[int]] = {}
        for arm_id, a in self.addr.arms.items():
            names = [f"{arm_id}_joint{i}" for i in range(1, N_ARM_JOINTS + 1)]
            if a.has_rail:
                names.append(f"{arm_id}_rail_joint")
            self._dof_joints[arm_id] = frozenset(int(model.joint(n).id) for n in names)
        self._label_joints: dict[tuple[str, str], frozenset[int]] = {}

    # -- public entry ----------------------------------------------------------
    def plan(self, req: PlanRequest) -> PlanResult:
        arms = list(req.q_goal)
        unknown = [a for a in arms if a not in self.addr.arms]
        if unknown or set(req.q_start) < set(arms):
            raise ValueError(
                f"plan request arms {arms} must exist in the scene and in q_start"
            )
        order = list(req.arm_order) if req.arm_order else self._heuristic_order(req, arms)
        if req.arm_order is not None or len(order) < 2:
            return self._plan_ordered(req, order)[0]
        orders = [order, list(reversed(order))]
        # Probe pass (module constant ORDER_PROBE_FRACTION): a short budget per ordering
        # first, so a heuristic ordering the RRT cannot get through does not spend the
        # whole per-arm budget before the reverse is tried. A deterministic failure
        # (``goal_in_collision`` / ``start_in_collision`` / ``no_escape``) does not depend
        # on the budget and is not retried with more of it.
        probe = req.model_copy(update={"timeout_s": req.timeout_s * ORDER_PROBE_FRACTION})
        results: list[tuple[PlanResult, int]] = []
        for o in orders:
            res, planned = self._plan_ordered(probe, o)
            if res.ok:
                return res
            results.append((res, planned))
        for i, o in enumerate(orders):
            if results[i][0].failure != "timeout":
                continue  # deterministic: more budget changes nothing
            res, planned = self._plan_ordered(req, o)
            if res.ok:
                return res
            results[i] = (res, planned)
        # Both fail: report the ordering that got furthest (more arms planned before the
        # failure), the first ordering on a tie. A ``goal_in_collision`` of the reversed
        # ordering is often an artefact of that ordering (an arm's goal against the OTHER
        # arm's start, which the first ordering moves away), so it does not outrank the
        # heuristic ordering's diagnosis by kind alone.
        (first, first_planned), (second, second_planned) = results
        return second if second_planned > first_planned else first

    def _heuristic_order(self, req: PlanRequest, arms: list[str]) -> list[str]:
        """Deepest-in-warn-band arm first (min start clearance ascending).

        Monitored (cross-arm / arm-environment) pairs via ``mj_geomDistance`` within
        5 cm, plus every violating pair at the start configuration - which includes
        the INTRA-arm pairs the monitored set excludes, so an arm pinched against
        itself is planned first too (2026-09-09 review)."""
        data = mujoco.MjData(self.twin.model)
        qpos = self._context_qpos(req, data)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(self.twin.model, data)
        depth: dict[str, float] = {}
        label = self.twin.allowed.label_of_geom
        for g1, g2 in self.twin.monitored_pairs:
            dist = mujoco.mj_geomDistance(self.twin.model, data, g1, g2, 0.05, None)
            # attributed to the arms whose joints move the pair (the twin's kinematic
            # ownership): the other arm against this arm's static rail base is the OTHER
            # arm's depth, not this arm's
            for arm_id in self._arms_of(tuple(sorted((label(g1), label(g2))))):
                if arm_id in arms:
                    depth[arm_id] = min(depth.get(arm_id, 0.05), float(dist))
        for pair, dist in self.twin.check_config_violations(qpos, data=data):
            for arm_id in self._arms_of(tuple(sorted(pair))):
                if arm_id in arms:
                    depth[arm_id] = min(depth.get(arm_id, 0.05), float(dist))
        return sorted(arms, key=lambda a: depth.get(a, 0.05))

    def _context_qpos(self, req: PlanRequest, data: mujoco.MjData) -> np.ndarray:
        """Full qpos with every requested arm at its start config."""
        qpos = np.array(self.twin._q_meas_full)
        for arm_id, q in req.q_start.items():
            if arm_id in self.addr.arms:
                qpos[self.addr[arm_id].qpos_adr] = np.asarray(q, dtype=np.float64)
        return qpos

    def _arms_of(self, key: tuple[str, str]) -> list[str]:
        arms = self._pair_arms.get(key)
        if arms is None:
            arms = list(self.twin._arms_of_pair(key))
            self._pair_arms[key] = arms
        return arms

    def _joints_under(self, arm_id: str, label: str) -> frozenset[int]:
        """This arm's dof joints on the kinematic path of ``label`` (a body name; a
        world geom label has none)."""
        cache_key = (arm_id, label)
        found = self._label_joints.get(cache_key)
        if found is None:
            model = self.twin.model
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label)
            dofs = self._dof_joints[arm_id]
            joints: set[int] = set()
            while body > 0:
                adr = int(model.body_jntadr[body])
                for j in range(adr, adr + int(model.body_jntnum[body])):
                    if j in dofs:
                        joints.add(j)
                body = int(model.body_parentid[body])
            found = frozenset(joints)
            self._label_joints[cache_key] = found
        return found

    def _moves(self, arm_id: str, key: tuple[str, str]) -> bool:
        """True iff the (sorted) label pair's distance can change with the planned arm's
        motion: the arm is one side of it and the two labels do not hang under the same
        set of its joints (module docstring)."""
        if arm_id not in self._arms_of(key):
            return False
        return self._joints_under(arm_id, key[0]) != self._joints_under(arm_id, key[1])

    # -- ordered sequential planning -------------------------------------------
    def _plan_ordered(self, req: PlanRequest, order: list[str]) -> tuple[PlanResult, int]:
        """Plan ``order`` sequentially; returns the result and how many arms were planned
        before a failure (all of them on success)."""
        data = mujoco.MjData(self.twin.model)
        fine_data = mujoco.MjData(self._fine_model)
        band_data = (
            fine_data if self._band_model is self._fine_model else mujoco.MjData(self._band_model)
        )
        ctx = self._context_qpos(req, data)  # everyone at start
        waypoints: dict[str, list[list[float]]] = {}
        for arm_id in order:
            q_start = np.asarray(req.q_start[arm_id], dtype=np.float64)
            q_goal = np.asarray(req.q_goal[arm_id], dtype=np.float64)
            path_or_fail = self._plan_arm(
                arm_id, q_start, q_goal, ctx, data, fine_data, band_data, req
            )
            if isinstance(path_or_fail, _ArmFailure):
                return (
                    PlanResult(
                        ok=False,
                        failure=path_or_fail.failure,
                        failing_pair=path_or_fail.failing_pair,
                    ),
                    len(waypoints),
                )
            waypoints[arm_id] = [list(map(float, q)) for q in path_or_fail]
            # freeze at the goal for arms > k - where the path ENDS (a held rail snaps the goal)
            ctx[self.addr[arm_id].qpos_adr] = np.asarray(path_or_fail[-1], dtype=np.float64)
        # ``arm_order`` is the execution contract (module docstring); the failing
        # branch above deliberately leaves it empty — nothing is safe to execute.
        return PlanResult(ok=True, waypoints=waypoints, arm_order=list(order)), len(order)

    # -- single-arm escape + RRT-Connect -----------------------------------------
    def _plan_arm(
        self,
        arm_id: str,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        ctx: np.ndarray,
        data: mujoco.MjData,
        fine_data: mujoco.MjData,
        band_data: mujoco.MjData,
        req: PlanRequest,
    ) -> list[np.ndarray] | _ArmFailure:
        p = self.params
        a = self.addr[arm_id]
        rng = np.random.default_rng(p.rng_seed)
        deadline = time.monotonic() + req.timeout_s
        lo, hi = (np.array(v) for v in self._jnt_range[arm_id])
        step = np.full(a.dof, req.max_step_rad)
        metric_w = np.ones(a.dof)
        q_goal = np.array(q_goal, dtype=np.float64)
        if a.has_rail:
            step[N_ARM_JOINTS] = RAIL_MAX_STEP_M
            metric_w[N_ARM_JOINTS] = p.rail_weight
            if abs(q_goal[N_ARM_JOINTS] - q_start[N_ARM_JOINTS]) <= RAIL_HOLD_TOL_M:
                # Held carriage (module docstring): no carriage motion was asked for, so
                # none is planned - the rail slot is pinned for the RRT samples, the escape
                # candidates and the repair nudges alike, and the goal snaps to the start.
                q_goal[N_ARM_JOINTS] = q_start[N_ARM_JOINTS]
                lo[N_ARM_JOINTS] = hi[N_ARM_JOINTS] = q_start[N_ARM_JOINTS]

        if np.max(np.abs(q_goal - q_start)) < 1e-9:
            # Nothing will be commanded for this arm (module docstring): it stays where it
            # is, pinched or not, and the arms after it are planned around it.
            return [q_start.copy(), q_goal.copy()]

        def violations(q_arm: np.ndarray) -> list[tuple[tuple[str, str], float]]:
            """Violating pairs THIS arm can move (module docstring: the others are
            constants for it), sorted label pairs, in twin order."""
            ctx[a.qpos_adr] = q_arm
            out: list[tuple[tuple[str, str], float]] = []
            for pair, dist in self.twin.check_config_violations(ctx, data=data):
                key = tuple(sorted(pair))
                if self._moves(arm_id, key):
                    out.append((key, dist))
            return out

        # Real penetration (≤ 0 mm) of one of this arm's pairs: nothing to plan.
        start_viol = violations(q_start)
        deep = [(pair, d) for pair, d in start_viol if d <= 0.0]
        if deep:
            worst = min(deep, key=lambda v: v[1])
            return _ArmFailure("start_in_collision", worst[0])
        # The goal is judged BEFORE the escape: it is the certain, cheap and actionable
        # diagnosis (a goal inside the shell cannot be reached whatever the start does).
        goal_viol = violations(q_goal)
        if goal_viol:
            worst = min(goal_viol, key=lambda v: v[1])
            return _ArmFailure("goal_in_collision", worst[0])

        # Pinched start (11-safety §9, 03-sim §10 item 3): this arm's pairs inside the
        # shell OR the gate's hysteresis band are walked out by the escape phase first.
        pinched = self._contacts_within(q_start, arm_id, a, ctx, self._band_model, band_data)
        escape: list[np.ndarray] = [q_start.copy()]
        if pinched:
            speed = float(getattr(req, "speed_scale", MIN_SPEED_SCALE))
            esc = self._escape(
                arm_id, q_start, pinched, step, lo, hi, ctx, data, fine_data, rng, deadline, speed
            )
            if isinstance(esc, _ArmFailure):
                return esc
            escape = esc
        q_from = escape[-1]

        # Pairs the arm rests near at either end only have to stay ≥ δ along the path.
        tolerated = set(
            self._contacts_within(q_from, arm_id, a, ctx, self._fine_model, fine_data)
        ) | set(self._contacts_within(q_goal, arm_id, a, ctx, self._fine_model, fine_data))
        last_block: list[tuple[str, str] | None] = [None]

        def valid(q_arm: np.ndarray) -> bool:
            """The RRT's predicate IS the verification's (module docstring): every movable
            pair ≥ δ + FINE_MARGIN_M, the tolerated ones ≥ δ - so the paths it returns
            do not skim the shell and the fine pass only has the between-sample dips left."""
            pair = self._margin_violation(q_arm, arm_id, a, ctx, fine_data, tolerated)
            if pair is not None:
                last_block[0] = pair
                return False
            return True

        graze: tuple[str, str] | None = None
        direct = True  # the straight edge + the shortcut pass; off after a repair failure
        attempt = 0
        while attempt == 0 or time.monotonic() < deadline:
            if attempt:
                rng = np.random.default_rng(p.rng_seed + attempt)
            attempt += 1
            path = self._rrt_connect(
                q_from, q_goal, lo, hi, step, metric_w, valid, rng, deadline, direct=direct
            )
            if path is None:
                # ``max_iters`` exhausted (or the deadline): the next seed gets the rest of
                # the budget - the operator sees ``timeout`` only once it is really spent.
                continue
            if direct:
                path = self._shortcut(path, step, valid, rng)  # the escape is never shortcut
            repaired = self._repair_fine(
                path, arm_id, a, ctx, data, fine_data, tolerated, lo, hi, deadline
            )
            if isinstance(repaired, list):
                ctx[a.qpos_adr] = q_goal
                return escape[:-1] + repaired  # repaired[0] is the escaped configuration
            graze = repaired
            # A path that validates at the RRT resolution but grazes at the gate's cannot
            # be repaired: the next seeds must not rebuild the same straight edge.
            direct = False
        return _ArmFailure("timeout", graze or last_block[0])

    def _pair_dist_in(self, data: mujoco.MjData, pair: tuple[str, str]) -> float:
        best = 1.0
        geoms = self.twin._geoms_of_label
        for g1 in geoms.get(pair[0], ()):
            for g2 in geoms.get(pair[1], ()):
                best = min(
                    best,
                    mujoco.mj_geomDistance(
                        self.twin.model, data, int(g1), int(g2), best, None
                    ),
                )
        return float(best)

    # -- escape phase (pinched start) ---------------------------------------------
    def _escape(
        self,
        arm_id: str,
        q_start: np.ndarray,
        pinched: dict[tuple[str, str], float],
        step: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        ctx: np.ndarray,
        data: mujoco.MjData,
        fine_data: mujoco.MjData,
        rng: np.random.Generator,
        deadline: float,
        speed: float,
    ) -> list[np.ndarray] | _ArmFailure:
        """Greedy monotone opening from a start inside the inflation shell / the band.

        Mirrors the gate's T8 escape rule tick by tick (11-safety §7.1 steps 5-7):
        along every returned segment every pinched pair that is not yet re-armed
        opens by ≥ ``ESCAPE_RATE_MARGIN x PLANNER_ESCAPE_EPS_M`` per executor tick
        at the requested ``speed`` (``_tick_count``: the slower the session the more
        ticks per segment and the less opening per tick), no pair of this arm outside the
        pinched set comes closer than ``δ + FINE_MARGIN_M`` (a NEW violation makes
        the gate hold; pairs already that tight at the start only have to stay
        ≥ δ), a pair already re-armed stays past ``δ + REARM_MARGIN_M`` (never back
        into the gate's hysteresis band), and joint limits hold. Each step tries the
        finite-difference ascent direction of the tightest pair (and the joint
        ascent of all pinched pairs), the coordinate axes and ``escape_random_dirs``
        random directions at half the edge resolution; candidates are filtered at
        the coarse resolution (collision check + the same opening RATE), ranked by
        the largest smallest opening, and the best one whose tick-level re-walk
        passes is kept. Ends once every pinched pair is past ``δ + REARM_MARGIN_M``;
        ``no_escape`` (naming the tightest pair) when no candidate opens them all
        or the step budget runs out, ``timeout`` past the request deadline.
        """
        p = self.params
        a = self.addr[arm_id]
        model = self.twin.model
        rearm_m = self.twin.inflation_m + REARM_MARGIN_M
        need_tick = ESCAPE_RATE_MARGIN * PLANNER_ESCAPE_EPS_M
        esc_step = step / 2.0
        sub_res = np.full(a.dof, ESCAPE_SUBSTEP_RAD)
        fd = np.full(a.dof, _FD_STEP_RAD)
        if a.has_rail:
            sub_res[N_ARM_JOINTS] = ESCAPE_SUBSTEP_RAIL_M
            fd[N_ARM_JOINTS] = _FD_STEP_RAIL_M
        v0 = sorted(pinched)
        active = set(v0)
        rearmed: set[tuple[str, str]] = set()
        tolerated = (
            set(self._contacts_within(q_start, arm_id, a, ctx, self._fine_model, fine_data))
            - active
        )

        def kinematics(q_arm: np.ndarray) -> None:
            ctx[a.qpos_adr] = q_arm
            data.qpos[:] = ctx
            mujoco.mj_kinematics(model, data)

        def dists() -> dict[tuple[str, str], float]:
            return {pair: self._pair_dist_in(data, pair) for pair in v0}

        def segment_ok(
            q_a: np.ndarray, q_b: np.ndarray, d_a: dict[tuple[str, str], float]
        ) -> dict[tuple[str, str], float] | None:
            """Coarse pass over the straight segment a→b (samples ≤ ESCAPE_SUBSTEP apart
            AND ≤ FINE_STEP_M of arm-point travel): no new violation, every active pair
            opens at the tick rate, re-armed pairs stay out. The end distances, or None."""
            dq = q_b - q_a
            n = max(
                1,
                int(math.ceil(float(np.max(np.abs(dq) / sub_res)))),
                int(math.ceil(self._lever_length(dq) / FINE_STEP_M)),
            )
            need = need_tick * _tick_count(dq, speed) / n  # the tick rate per coarse sample
            prev = d_a
            for i in range(1, n + 1):
                q_i = q_a + dq * (i / n)
                if (
                    self._margin_violation(q_i, arm_id, a, ctx, fine_data, tolerated, active)
                    is not None
                ):
                    return None  # a new violation (or a re-armed pair back inside)
                kinematics(q_i)
                cur = dists()
                for pair in active:
                    if cur[pair] < prev[pair] + need:
                        return None
                for pair in rearmed:
                    if cur[pair] < rearm_m:
                        return None
                prev = cur
            return prev

        def ticks_ok(q_a: np.ndarray, q_b: np.ndarray, d_a: dict[tuple[str, str], float]) -> bool:
            """The gate's rule at the executor's ticks of the requested speed: every active
            pair opens by ≥ ESCAPE_RATE_MARGIN x eps on every one of them (pair distances
            only - the coarse pass already excluded new violations at ≤ FINE_STEP_M
            spacing)."""
            dq = q_b - q_a
            n = _tick_count(dq, speed)
            prev = d_a
            for i in range(1, n + 1):
                kinematics(q_a + dq * (i / n))
                cur = dists()
                for pair in active:
                    if cur[pair] < prev[pair] + need_tick:
                        return False
                prev = cur
            return True

        kinematics(q_start)
        d_now = dists()
        q = q_start.copy()
        path = [q.copy()]
        for _ in range(p.escape_max_steps):
            if not active:
                break
            tight = min(active, key=lambda pair: d_now[pair])
            if time.monotonic() > deadline:
                return _ArmFailure("timeout", tight)
            # Candidate directions (box-normalised: the fastest slot moves esc_step).
            grads = self._opening_gradients(q, active, fd, lo, hi, kinematics, dists)
            dirs: list[np.ndarray] = [grads[tight] * esc_step]
            if len(active) > 1:
                joint = np.zeros(a.dof)
                for pair in active:
                    g = grads[pair] * esc_step
                    m = float(np.max(np.abs(g)))
                    if m > 0.0:
                        joint += g / m
                dirs.append(joint)
            for j in range(a.dof):
                e = np.zeros(a.dof)
                e[j] = 1.0
                dirs.extend((e, -e))
            dirs.extend(rng.normal(size=a.dof) for _ in range(p.escape_random_dirs))
            cands: list[tuple[float, np.ndarray, dict[tuple[str, str], float]]] = []
            for u in dirs:
                m = float(np.max(np.abs(u)))
                if m <= 0.0:
                    continue
                q_c = np.clip(q + esc_step * (u / m), lo, hi)
                if np.max(np.abs(q_c - q)) < 1e-9:
                    continue
                d_c = segment_ok(q, q_c, d_now)
                if d_c is None:
                    continue
                score = min(d_c[pair] - d_now[pair] for pair in active)
                cands.append((score, q_c, d_c))
            cands.sort(key=lambda c: c[0], reverse=True)
            chosen: tuple[np.ndarray, dict[tuple[str, str], float]] | None = None
            for _score, q_c, d_c in cands:
                if ticks_ok(q, q_c, d_now):
                    chosen = (q_c, d_c)
                    break
            if chosen is None:
                return _ArmFailure("no_escape", tight)
            q, d_now = chosen
            path.append(q.copy())
            for pair in list(active):
                if d_now[pair] > rearm_m:
                    active.discard(pair)
                    rearmed.add(pair)
        else:
            if active:
                return _ArmFailure("no_escape", min(active, key=lambda pair: d_now[pair]))
        return path

    @staticmethod
    def _opening_gradients(
        q: np.ndarray,
        pairs: set[tuple[str, str]],
        fd: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        kinematics,
        dists,
    ) -> dict[tuple[str, str], np.ndarray]:
        """Central finite-difference d(dist)/dq per pair (one-sided at a limit)."""
        grads = {pair: np.zeros(q.shape[0]) for pair in pairs}
        for j in range(q.shape[0]):
            q_plus, q_minus = q.copy(), q.copy()
            q_plus[j] = min(q[j] + fd[j], hi[j])
            q_minus[j] = max(q[j] - fd[j], lo[j])
            span = q_plus[j] - q_minus[j]
            if span <= 0.0:
                continue
            kinematics(q_plus)
            d_plus = dists()
            kinematics(q_minus)
            d_minus = dists()
            for pair in pairs:
                grads[pair][j] = (d_plus[pair] - d_minus[pair]) / span
        return grads

    # -- final verification at the gate's resolution --------------------------------
    @staticmethod
    def _lever_length(dq: np.ndarray) -> float:
        """Upper bound on the displacement of any arm point over the joint delta ``dq``."""
        n = min(N_ARM_JOINTS, dq.shape[0])
        length = float(np.sum(np.abs(dq[:n]) * _LEVER_ARM_M[:n]))
        if dq.shape[0] > N_ARM_JOINTS:
            length += float(np.sum(np.abs(dq[N_ARM_JOINTS:])))  # rail: 1:1
        return length

    def _contacts_within(
        self,
        q_arm: np.ndarray,
        arm_id: str,
        a,
        ctx: np.ndarray,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> dict[tuple[str, str], float]:
        """This arm's movable pairs in contact on an inflated model copy (``model`` = the fine
        model: within ``δ + FINE_MARGIN_M``; the band model: within ``δ + hysteresis``),
        sorted label pair → distance (the contact distance IS the plain surface distance)."""
        ctx[a.qpos_adr] = q_arm
        data.qpos[:] = ctx
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        out: dict[tuple[str, str], float] = {}
        for pair, dist in self.twin._violations(data):
            key = tuple(sorted(pair))
            if self._moves(arm_id, key):
                out[key] = min(dist, out.get(key, np.inf))
        return out

    def _margin_violation(
        self,
        q_arm: np.ndarray,
        arm_id: str,
        a,
        ctx: np.ndarray,
        fine_data: mujoco.MjData,
        tolerated: set[tuple[str, str]],
        ignore: set[tuple[str, str]] = frozenset(),
    ) -> tuple[str, str] | None:
        """Tightest pair of THIS arm closer than ``δ + FINE_MARGIN_M`` at ``q_arm``
        (``tolerated`` pairs only have to stay ≥ δ; ``ignore`` = the escape's pinched
        pairs, judged by their opening instead); None when the sample is clear."""
        ctx[a.qpos_adr] = q_arm
        fine_data.qpos[:] = ctx
        mujoco.mj_kinematics(self._fine_model, fine_data)
        mujoco.mj_collision(self._fine_model, fine_data)
        worst: tuple[tuple[str, str], float] | None = None
        for pair, dist in self.twin._violations(fine_data):
            key = tuple(sorted(pair))
            if key in ignore or (key in tolerated and dist >= self.twin.inflation_m):
                continue
            if not self._moves(arm_id, key):
                continue  # a constant for this arm (module docstring)
            if worst is None or dist < worst[1]:
                worst = (key, dist)
        return None if worst is None else worst[0]

    def _fine_violation(
        self,
        path: list[np.ndarray],
        arm_id: str,
        a,
        ctx: np.ndarray,
        fine_data: mujoco.MjData,
        tolerated: set[tuple[str, str]],
        start: int = 0,
    ) -> tuple[int, np.ndarray, tuple[str, str], bool] | None:
        """First sample (spacing ≤ FINE_STEP_M of arm-point travel) the gate could hold
        at: (segment index, sample, pair, sample is the segment's end waypoint). The
        path's own endpoints are the caller's (plain-validated) start and goal."""
        for i in range(start, len(path) - 1):
            q_a, q_b = path[i], path[i + 1]
            n = max(1, int(math.ceil(self._lever_length(q_b - q_a) / FINE_STEP_M)))
            last = i == len(path) - 2
            for k in range(1, n + 1):
                if k == n and last:
                    break  # the goal itself
                q_k = q_a + (q_b - q_a) * (k / n)
                pair = self._margin_violation(q_k, arm_id, a, ctx, fine_data, tolerated)
                if pair is not None:
                    return i, q_k, pair, k == n
        return None

    def _repair_fine(
        self,
        path: list[np.ndarray],
        arm_id: str,
        a,
        ctx: np.ndarray,
        data: mujoco.MjData,
        fine_data: mujoco.MjData,
        tolerated: set[tuple[str, str]],
        lo: np.ndarray,
        hi: np.ndarray,
        deadline: float,
    ) -> list[np.ndarray] | tuple[str, str] | None:
        """Verify ``path`` at the gate's resolution, nudging grazing samples off their
        pair along its opening gradient (inserted as waypoints; an interior waypoint
        is replaced). Returns the repaired path, or the pair that could not be repaired."""
        model = self.twin.model
        nudge = np.full(a.dof, FINE_NUDGE_RAD)
        fd = np.full(a.dof, _FD_STEP_RAD)
        if a.has_rail:
            nudge[N_ARM_JOINTS] = FINE_NUDGE_RAIL_M
            fd[N_ARM_JOINTS] = _FD_STEP_RAIL_M

        def kinematics(q_arm: np.ndarray) -> None:
            ctx[a.qpos_adr] = q_arm
            data.qpos[:] = ctx
            mujoco.mj_kinematics(model, data)

        def dists_for(pair: tuple[str, str]):
            def dists() -> dict[tuple[str, str], float]:
                return {pair: self._pair_dist_in(data, pair)}

            return dists

        path = list(path)
        resume = 0
        for _ in range(_FINE_REPAIRS_MAX):
            hit = self._fine_violation(path, arm_id, a, ctx, fine_data, tolerated, resume)
            if hit is None:
                return path
            i, q_bad, pair, is_waypoint = hit
            if time.monotonic() > deadline:
                return pair
            q_new = q_bad.copy()
            fixed = False
            dists = dists_for(pair)
            for _ in range(_FINE_NUDGES_MAX):
                grads = self._opening_gradients(q_new, {pair}, fd, lo, hi, kinematics, dists)
                g = grads[pair] * nudge
                m = float(np.max(np.abs(g)))
                if m <= 0.0:
                    break
                q_new = np.clip(q_new + nudge * (g / m), lo, hi)
                if self._margin_violation(q_new, arm_id, a, ctx, fine_data, tolerated) is None:
                    fixed = True
                    break
            if not fixed:
                return pair
            if is_waypoint:
                path[i + 1] = q_new
            else:
                path.insert(i + 1, q_new)
            resume = i
        hit = self._fine_violation(path, arm_id, a, ctx, fine_data, tolerated, resume)
        return path if hit is None else hit[2]

    # -- RRT-Connect machinery ---------------------------------------------------
    def _rrt_connect(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        step: np.ndarray,
        w: np.ndarray,
        valid,
        rng: np.random.Generator,
        deadline: float,
        direct: bool = True,
    ) -> list[np.ndarray] | None:
        """RRT-Connect between two valid configurations; ``direct=False`` skips the
        straight-edge shortcut (a retry after the straight edge failed verification)."""
        if np.max(np.abs(q_goal - q_start) / step) < 1.0 or (
            direct and self._edge_valid(q_start, q_goal, step, valid)
        ):
            return [q_start.copy(), q_goal.copy()]
        trees = ([q_start.copy()], [q_goal.copy()])
        parents: tuple[list[int], list[int]] = ([-1], [-1])
        # the start/goal-local sampling box (module constants; a held rail stays pinned)
        margin = np.full(q_start.shape[0], LOCAL_SAMPLE_MARGIN_RAD)
        margin[N_ARM_JOINTS:] = LOCAL_SAMPLE_MARGIN_RAIL_M
        lo_loc = np.maximum(lo, np.minimum(q_start, q_goal) - margin)
        hi_loc = np.minimum(hi, np.maximum(q_start, q_goal) + margin)
        a = 0  # active tree index; tree 0 roots at start, tree 1 at goal
        for _ in range(self.params.max_iters):
            if time.monotonic() > deadline:
                return None
            b = 1 - a
            if rng.uniform() < LOCAL_SAMPLE_FRACTION:
                q_rand = rng.uniform(lo_loc, hi_loc)
            else:
                q_rand = rng.uniform(lo, hi)
            new_a, _ = self._extend(
                trees[a], parents[a], q_rand, step, w, valid, greedy=False
            )
            if new_a is not None:
                new_b, reached = self._extend(
                    trees[b], parents[b], trees[a][new_a], step, w, valid, greedy=True
                )
                if new_b is not None and reached:
                    path_a = self._trace(trees[a], parents[a], new_a)
                    path_b = self._trace(trees[b], parents[b], new_b)
                    if a == 1:
                        path_a, path_b = path_b, path_a  # path_a roots at start
                    return path_a + path_b[::-1][1:]
            a = b
        return None

    def _extend(
        self,
        nodes: list[np.ndarray],
        parents: list[int],
        q_target: np.ndarray,
        step: np.ndarray,
        w: np.ndarray,
        valid,
        greedy: bool,
    ) -> tuple[int | None, bool]:
        """Grow toward ``q_target``; returns (new node index, reached target?)."""
        arr = np.asarray(nodes)
        near_idx = int(np.argmin(np.max(w * np.abs(arr - q_target), axis=1)))
        q_near = nodes[near_idx]
        delta = q_target - q_near
        n = int(np.ceil(np.max(np.abs(delta) / step)))
        if n == 0:
            return near_idx, True
        unit = delta / n
        limit = n if greedy else 1
        done = 0
        for i in range(1, limit + 1):
            if not valid(q_near + unit * i):
                break
            done = i
        if done == 0:
            return None, False
        nodes.append(q_near + unit * done)
        parents.append(near_idx)
        return len(nodes) - 1, done == n

    @staticmethod
    def _trace(nodes: list[np.ndarray], parents: list[int], idx: int) -> list[np.ndarray]:
        path = []
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        return path[::-1]  # root first

    def _edge_valid(self, q_a: np.ndarray, q_b: np.ndarray, step: np.ndarray, valid) -> bool:
        n = int(np.ceil(np.max(np.abs(q_b - q_a) / step)))
        for i in range(1, n + 1):
            if not valid(q_a + (q_b - q_a) * (i / n)):
                return False
        return True

    def _shortcut(
        self,
        path: list[np.ndarray],
        step: np.ndarray,
        valid,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Random shortcut passes: keep a splice when the straight edge holds."""
        for _ in range(self.params.shortcut_attempts):
            if len(path) < 3:
                break
            i, j = sorted(int(x) for x in rng.integers(0, len(path), size=2))
            if j - i < 2:
                continue
            if self._edge_valid(path[i], path[j], step, valid):
                path = path[: i + 1] + path[j:]
        return path


def time_parameterize(
    waypoints: list[list[float]],
    has_rail: bool,
    params: PlannerParams | None = None,
) -> list[tuple[float, list[float]]]:
    """Attach monotone timestamps to sparse waypoints (11-safety §9 caps).

    Per segment the duration is the slowest dim under a trapezoidal bound:
    ``t = max(|Δ|/v, sqrt(4|Δ|/a))`` per dim. Runtime (phase-05) interpolates
    these into 100 Hz setpoints streamed through the same gate as teleop.
    """
    p = params or PlannerParams()
    if not waypoints:
        return []
    qs = [np.asarray(q, dtype=np.float64) for q in waypoints]
    vel = np.full(qs[0].shape, p.joint_vel_rad_s)
    acc = np.full(qs[0].shape, p.joint_acc_rad_s2)
    if has_rail:
        vel[N_ARM_JOINTS] = p.rail_vel_m_s
    out = [(0.0, list(map(float, qs[0])))]
    t = 0.0
    for q_prev, q_next in zip(qs[:-1], qs[1:], strict=True):
        delta = np.abs(q_next - q_prev)
        t += float(np.max(np.maximum(delta / vel, np.sqrt(4.0 * delta / acc))))
        out.append((t, list(map(float, q_next))))
    return out


__all__ = [
    "PlannerParams",
    "ResetPlanner",
    "time_parameterize",
    "RAIL_MAX_STEP_M",
    "RAIL_HOLD_TOL_M",
    "ORDER_PROBE_FRACTION",
    "LOCAL_SAMPLE_FRACTION",
    "REARM_MARGIN_M",
    "PLANNER_ESCAPE_EPS_M",
    "ESCAPE_RATE_MARGIN",
    "FINE_STEP_M",
    "FINE_MARGIN_M",
    "HW_SLEW_RAD_PER_TICK",
    "HW_RAIL_M_PER_TICK",
    "HW_CART_STEP_M",
    "MIN_SPEED_SCALE",
]
