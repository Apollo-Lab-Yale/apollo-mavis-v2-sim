"""ResetPlanner — per-arm sequential RRT-Connect in joint space (03-sim §10).

Serves profile loads / ``start_from``, joint-panel ``goto`` and safety-escape
motions. Request/response shapes are core's ``PlanRequest``/``PlanResult``
(01-core §6) — this module defines no wire shape of its own. The twin is the
validity checker (planner-private ``MjData``); arms are planned one at a time
with earlier arms frozen at their *goals* and later arms at their *starts*
(11-safety §9); a failed ordering is retried in reverse. Execution lives in
runtime (phase-05): this module only produces waypoints, all of which pass
``twin.check_config``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from apollo_mavis_v2_core import PlanRequest, PlanResult

from .scenes.addressing import N_ARM_JOINTS, Addressing

if TYPE_CHECKING:
    from .twin import DigitalTwin

RAIL_MAX_STEP_M = 0.01  # edge-check resolution for the rail dim (11-safety §9)
REARM_MARGIN_M = 0.005  # start-hysteresis: re-arm past inflation + 5 mm


@dataclass
class PlannerParams:
    """Local tuning knobs; request-level knobs ride ``PlanRequest``."""

    rail_weight: float = 4.0  # rail dims x4 in the nearest-neighbour metric
    max_iters: int = 2000
    shortcut_attempts: int = 50
    joint_vel_rad_s: float = 0.6  # time-parameterization caps (11-safety §9)
    joint_acc_rad_s2: float = 2.0
    rail_vel_m_s: float = 0.1
    rng_seed: int = 0


@dataclass
class _ArmFailure:
    failure: str  # "goal_in_collision" | "start_in_collision" | "timeout"
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

    # -- public entry ----------------------------------------------------------
    def plan(self, req: PlanRequest) -> PlanResult:
        arms = list(req.q_goal)
        unknown = [a for a in arms if a not in self.addr.arms]
        if unknown or set(req.q_start) < set(arms):
            raise ValueError(
                f"plan request arms {arms} must exist in the scene and in q_start"
            )
        order = list(req.arm_order) if req.arm_order else self._heuristic_order(req, arms)
        first = self._plan_ordered(req, order)
        if first.ok or len(order) < 2 or req.arm_order is not None:
            return first
        second = self._plan_ordered(req, list(reversed(order)))
        return second if second.ok else first  # both fail: report first ordering

    def _heuristic_order(self, req: PlanRequest, arms: list[str]) -> list[str]:
        """Deepest-in-warn-band arm first (min start clearance ascending)."""
        data = mujoco.MjData(self.twin.model)
        qpos = self._context_qpos(req, data)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(self.twin.model, data)
        depth: dict[str, float] = {}
        geoms_of_arm = {a: set(int(g) for g in self.addr[a].geom_ids) for a in arms}
        for g1, g2 in self.twin.monitored_pairs:
            dist = mujoco.mj_geomDistance(self.twin.model, data, g1, g2, 0.05, None)
            for arm_id in arms:
                if g1 in geoms_of_arm[arm_id] or g2 in geoms_of_arm[arm_id]:
                    depth[arm_id] = min(depth.get(arm_id, 0.05), float(dist))
        return sorted(arms, key=lambda a: depth.get(a, 0.05))

    def _context_qpos(self, req: PlanRequest, data: mujoco.MjData) -> np.ndarray:
        """Full qpos with every requested arm at its start config."""
        qpos = np.array(self.twin._q_meas_full)
        for arm_id, q in req.q_start.items():
            if arm_id in self.addr.arms:
                qpos[self.addr[arm_id].qpos_adr] = np.asarray(q, dtype=np.float64)
        return qpos

    # -- ordered sequential planning -------------------------------------------
    def _plan_ordered(self, req: PlanRequest, order: list[str]) -> PlanResult:
        data = mujoco.MjData(self.twin.model)
        ctx = self._context_qpos(req, data)  # everyone at start
        waypoints: dict[str, list[list[float]]] = {}
        for arm_id in order:
            q_start = np.asarray(req.q_start[arm_id], dtype=np.float64)
            q_goal = np.asarray(req.q_goal[arm_id], dtype=np.float64)
            path_or_fail = self._plan_arm(arm_id, q_start, q_goal, ctx, data, req)
            if isinstance(path_or_fail, _ArmFailure):
                return PlanResult(
                    ok=False,
                    failure=path_or_fail.failure,
                    failing_pair=path_or_fail.failing_pair,
                )
            waypoints[arm_id] = [list(map(float, q)) for q in path_or_fail]
            ctx[self.addr[arm_id].qpos_adr] = q_goal  # freeze at goal for arms > k
        return PlanResult(ok=True, waypoints=waypoints)

    # -- single-arm RRT-Connect -------------------------------------------------
    def _plan_arm(
        self,
        arm_id: str,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        ctx: np.ndarray,
        data: mujoco.MjData,
        req: PlanRequest,
    ) -> list[np.ndarray] | _ArmFailure:
        p = self.params
        a = self.addr[arm_id]
        rng = np.random.default_rng(p.rng_seed)
        deadline = time.monotonic() + req.timeout_s
        lo, hi = self._jnt_range[arm_id]
        step = np.full(a.dof, req.max_step_rad)
        metric_w = np.ones(a.dof)
        if a.has_rail:
            step[N_ARM_JOINTS] = RAIL_MAX_STEP_M
            metric_w[N_ARM_JOINTS] = p.rail_weight

        def violations(q_arm: np.ndarray) -> list[tuple[tuple[str, str], float]]:
            ctx[a.qpos_adr] = q_arm
            return self.twin.check_config_violations(ctx, data=data)

        # Start-state hysteresis (11-safety §9): pairs already inside the
        # inflation shell are whitelisted until first exceeding δ + 5 mm.
        start_viol = violations(q_start)
        deep = [(pair, d) for pair, d in start_viol if d <= 0.0]
        if deep:
            worst = min(deep, key=lambda v: v[1])
            return _ArmFailure("start_in_collision", tuple(sorted(worst[0])))
        whitelist = {tuple(sorted(pair)) for pair, _ in start_viol}
        last_block: list[tuple[str, str] | None] = [None]

        def valid(q_arm: np.ndarray) -> bool:
            viol = violations(q_arm)
            present = set()
            for pair, _dist in viol:
                key = tuple(sorted(pair))
                present.add(key)
                if key not in whitelist:
                    last_block[0] = key
                    return False
            for key in list(whitelist - present):
                # pair fell out of the detection window: re-arm once clear of
                # inflation + margin (kinematics for ctx already computed).
                if self._pair_dist_in(data, key) > self.twin.inflation_m + REARM_MARGIN_M:
                    whitelist.discard(key)
            return True

        goal_viol = violations(q_goal)
        goal_block = [
            (pair, d) for pair, d in goal_viol if tuple(sorted(pair)) not in whitelist
        ]
        if goal_block:
            worst = min(goal_block, key=lambda v: v[1])
            return _ArmFailure("goal_in_collision", tuple(sorted(worst[0])))

        path = self._rrt_connect(
            q_start, q_goal, lo, hi, step, metric_w, valid, rng, deadline
        )
        if path is None:
            return _ArmFailure("timeout", last_block[0])
        path = self._shortcut(path, step, valid, rng)
        ctx[a.qpos_adr] = q_goal
        return path

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
    ) -> list[np.ndarray] | None:
        if np.max(np.abs(q_goal - q_start) / step) < 1.0 or self._edge_valid(
            q_start, q_goal, step, valid
        ):
            return [q_start.copy(), q_goal.copy()]
        trees = ([q_start.copy()], [q_goal.copy()])
        parents: tuple[list[int], list[int]] = ([-1], [-1])
        a = 0  # active tree index; tree 0 roots at start, tree 1 at goal
        for _ in range(self.params.max_iters):
            if time.monotonic() > deadline:
                return None
            b = 1 - a
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


__all__ = ["PlannerParams", "ResetPlanner", "time_parameterize", "RAIL_MAX_STEP_M"]
