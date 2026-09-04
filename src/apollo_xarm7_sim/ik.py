"""MinkIKSolver — QP differential IK over the composite scene (03-sim §9).

Implements core's ``IKSolver`` Protocol with mink 1.3.0 (`qpsolvers` + daqp):
one ``mink.Configuration`` shared by all arms of the workcell (the model is
typically the digital twin's — the Configuration owns its own ``MjData``),
per-arm ``FrameTask`` on the ``<arm>_link_tcp`` site + one ``PostureTask``
with per-DoF costs (rail expensive, or pinned outright with ``lock_rail`` —
then the servo path never moves it and adopts the rail slot from the seed;
non-active arms pinned at 1e4 toward their measured configuration and their
Δq slice zeroed before integration)
+ ``ConfigurationLimit`` + ``VelocityLimit`` + a row-capped
``CollisionAvoidanceLimit`` (omitted in plain sim mode — pass
``collision_pairs=None``).

Four RelaxedIK/CollisionIK-derived refinements (research collision-ik §5a):
accel/jerk regularization over a 3-deep Δq history, adaptive orientation
weight (ECAA), per-DoF flat-bottom tolerances, and an active constraint-row
cap. Targets are TCP poses in the WORLD frame.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np
from apollo_xarm7_core import ArmState, IKResult, Pose
from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
from mink.limits.limit import Constraint
from mink.tasks.task import Objective

from .errors import IKUnreachableError
from .scenes.addressing import N_ARM_JOINTS
from .scenes.builder import HOME_Q, BuiltScene
from .twin import AllowedPairs, build_monitored_pairs

_PIN_COST = 1e4  # posture cost pinning non-active / non-IK dofs
_ECAA_A = 0.05  # ECAA shape constant: w_target = base * a / (a + s)
_ECAA_D_WARN = 0.05  # m; proximity score s = max(0, (d_warn - d_min)/d_warn)
_ECAA_SLEW = 0.02  # max weight change per tick, fraction of base
_ECAA_FLOOR = 0.10  # weight floor, fraction of base
_DIVERGE_TICKS = 10  # consecutive over-residual ticks before diverged=True


def _default_velocity_limits() -> dict[str, float]:
    # rad/s per joint (xArm7 firmware cap ~= pi rad/s), rail in m/s.
    limits = {f"joint{i}": 3.14 for i in range(1, N_ARM_JOINTS + 1)}
    limits["rail"] = 0.2
    return limits


@dataclass
class IKParams:
    """Tuning knobs (03-sim §9; collision params bind to 11-safety §8)."""

    dt: float = 0.01
    damping: float = 1e-3
    solver: str = "daqp"
    position_cost: float = 1.0
    orientation_cost: float = 0.5  # base; ECAA adapts the live weight
    posture_cost_joint: float = 5e-2
    posture_cost_rail: float = 5.0  # rail expensive -> prefer joints
    collision_gain: float = 0.85
    max_collision_rows: int = 12  # = safety.max_active_constraint_rows
    min_distance_m: float = 0.010  # = geom_inflation_m + 0.002 (OUTSIDE the gate)
    detection_distance_m: float = 0.05
    velocity_limits: dict[str, float] = field(default_factory=_default_velocity_limits)
    flat_tolerances: np.ndarray = field(default_factory=lambda: np.zeros(6))
    w_accel: float = 1e-2
    w_jerk: float = 1e-3
    pos_reject_m: float = 0.02
    rot_reject_rad: float = 0.35
    reseed_threshold: float = 0.05  # max|q_seed - q_warm| that forces a reset
    # Servo path (``solve``) only: exclude the rail from the differential QP.
    # The rail slot becomes an INPUT (rail keys / controller trackpad), adopted
    # from ``q_seed`` every tick and never moved to reach a target (04-runtime
    # §6 "Rail", ``control.rail_in_ik: false``). One-shot far targets
    # (``solve_to_convergence``) still place the rail.
    lock_rail: bool = False


def default_collision_pairs(
    scene: BuiltScene, allowed: AllowedPairs | None = None
) -> list[tuple[list[int], list[int]]]:
    """Self + cross-arm + arm↔environment geom pairs for the avoidance limit.

    Explicit singleton pairs (pre-filtered) rather than group products: mink
    does not honour MJCF ``<exclude>``s, so the pair list must respect the
    same ``AllowedPairs`` filter as the twin (permanent near-contacts like
    ``left_finger ↔ gripper_base`` would otherwise wedge the QP).
    """
    model = scene.model
    allowed = allowed if allowed is not None else AllowedPairs(model)
    pairs = set(build_monitored_pairs(model, scene.addressing, allowed))
    # Intra-arm self-collision pairs: skip welded + parent-child neighbours.
    weld = model.body_weldid[model.geom_bodyid]
    parent_of_weld = model.body_weldid[model.body_parentid]
    for a in scene.addressing.arms.values():
        geoms = a.geom_ids
        for i, ga in enumerate(geoms):
            for gb in geoms[i + 1 :]:
                ga_i, gb_i = int(ga), int(gb)
                b1, b2 = int(model.geom_bodyid[ga_i]), int(model.geom_bodyid[gb_i])
                if weld[ga_i] == weld[gb_i]:
                    continue
                if (
                    parent_of_weld[model.body_weldid[b1]] == model.body_weldid[b2]
                    or parent_of_weld[model.body_weldid[b2]] == model.body_weldid[b1]
                ):
                    continue  # parent-child (weld-aware), both directions
                if not (
                    model.geom_contype[ga_i] & model.geom_conaffinity[gb_i]
                    or model.geom_contype[gb_i] & model.geom_conaffinity[ga_i]
                ):
                    continue
                if allowed.allows(ga_i, gb_i):
                    continue
                pairs.add((min(ga_i, gb_i), max(ga_i, gb_i)))
    return [([g1], [g2]) for g1, g2 in sorted(pairs)]


class _FlatToleranceFrameTask(mink.FrameTask):
    """Refinement 3: per-DoF flat-bottom tolerance via error shrinkage.

    ``e_i <- 0 if |e_i| <= tol_i else e_i - tol_i * sign(e_i)`` on the 6-D
    task error (x y z rx ry rz, local frame) before the QP — RangedIK's
    swamp loss reduced to its useful core (e.g. ``tol[5] = pi`` frees tool
    roll during collection). Default tolerances are zero (exact tracking).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tol = np.zeros(6)

    def _shrink(self, err: np.ndarray) -> np.ndarray:
        if np.any(self.tol > 0.0):
            err = np.sign(err) * np.maximum(np.abs(err) - self.tol, 0.0)
        return err

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        return self._shrink(super().compute_error(configuration))

    def compute_qp_residual(self, configuration: mink.Configuration):
        # mink 1.3.0's fused error+jacobian fast path bypasses compute_error;
        # shrink there too or the QP would still chase the tolerated axes.
        error, jacobian = self._error_and_jacobian(configuration)
        return self._weighted_residual(self._shrink(error), jacobian)


class _SmoothnessObjective:
    """Refinement 1: accel/jerk regularization over a 3-deep Δq history.

    Adds ``w_accel * ||Δq - Δq₋₁||² + w_jerk * ||Δq - (2Δq₋₁ - Δq₋₂)||²``
    to the QP cost, restricted to the active arm's dof slice (linear in Δq
    given history — why RelaxedIK output looks smooth on real arms; mink
    alone only damps). Duck-typed as a mink task: only
    ``compute_qp_objective`` is required by ``build_ik``.
    """

    def __init__(self, nv: int, w_accel: float, w_jerk: float) -> None:
        self._nv = nv
        self._w_accel = w_accel
        self._w_jerk = w_jerk
        self._mask = np.zeros(nv, dtype=bool)
        self._dq1 = np.zeros(nv)  # Δq at t-1
        self._dq2 = np.zeros(nv)  # Δq at t-2

    def set_active(self, dof_adr: np.ndarray, dq1: np.ndarray, dq2: np.ndarray) -> None:
        self._mask[:] = False
        self._mask[dof_adr] = True
        self._dq1, self._dq2 = dq1, dq2

    def compute_qp_residual(self, configuration: mink.Configuration) -> None:
        return None  # dense-objective task: no low-rank residual form

    def compute_qp_objective(self, configuration: mink.Configuration):
        wa, wj = self._w_accel, self._w_jerk
        diag = np.where(self._mask, 2.0 * (wa + wj), 0.0)
        h_mat = np.diag(diag)
        jerk_ref = 2.0 * self._dq1 - self._dq2
        c = np.where(self._mask, -2.0 * (wa * self._dq1 + wj * jerk_ref), 0.0)
        return Objective(h_mat, c)


class _CappedCollisionLimit(mink.CollisionAvoidanceLimit):
    """Refinement 4: keep only the ``max_rows`` nearest pairs per tick.

    Reimplements the 1.3.0 row loop (version-pinned) so per-row distances
    are available for ranking by ``dist - min_dist``; bounds worst-case QP
    latency at 100 Hz with 3 arms + environment meshes. Exposes
    ``last_active_rows`` and ``last_min_dist`` for telemetry / ECAA fallback.
    """

    def __init__(self, *args, max_rows: int = 12, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_rows = max_rows
        self.last_active_rows = 0
        self.last_min_dist: float | None = None

    def compute_qp_inequalities(
        self, configuration: mink.Configuration, dt: float
    ) -> Constraint:
        model, data = self.model, configuration.data
        distmax = self.collision_detection_distance
        if self.broadphase and self.max_num_contacts >= self.broadphase_min_pairs:
            indices = self._broadphase_survivors(data)
        else:
            indices = range(self.max_num_contacts)
        # Pass 1: distances only (shared scratch fromto), rank by proximity.
        geom_id_pairs = self.geom_id_pairs
        geom_distance = mujoco.mj_geomDistance
        scratch = self._fromto
        hits: list[tuple[float, int]] = []
        for idx in indices:
            g1, g2 = geom_id_pairs[idx]
            dist = geom_distance(model, data, g1, g2, distmax, scratch)
            if dist < distmax:
                hits.append((dist, idx))
        hits.sort()
        hits = hits[: self.max_rows]
        self.last_active_rows = len(hits)
        self.last_min_dist = hits[0][0] if hits else None
        # Pass 2: witness segments + normal Jacobians for the kept rows only.
        n = max(len(hits), 1)
        coeff = np.zeros((n, model.nv))
        bound = np.full((n,), np.inf)
        dmin = self.minimum_distance_from_collisions
        for row, (dist, idx) in enumerate(hits):
            g1, g2 = geom_id_pairs[idx]
            geom_distance(model, data, g1, g2, distmax, scratch)
            jac = compute_contact_normal_jacobian(
                model, data, g1, g2, scratch, self._normal, self._jac1, self._jac2
            )
            if dist > dmin:
                bound[row] = (self.gain * (dist - dmin) / dt) + self.bound_relaxation
            else:
                bound[row] = self.bound_relaxation
            coeff[row] = (-1.0 if dist >= 0 else 1.0) * jac
        return Constraint(G=coeff, h=bound)


class MinkIKSolver:
    """Differential IK for every arm of one workcell (core ``IKSolver``).

    ``collision_pairs=None`` omits the ``CollisionAvoidanceLimit`` entirely
    (plain sim mode — gate off, 11-safety §5); pass
    :func:`default_collision_pairs` (or a custom list) for ``safety_debug``
    and hardware modes. All arm vectors are core order (rail LAST, ``q[7]``).
    """

    def __init__(
        self,
        scene: BuiltScene,
        params: IKParams | None = None,
        collision_pairs: list[tuple[list[int], list[int]]] | None = None,
        rng_seed: int = 0,
    ) -> None:
        self.scene = scene
        self.model = scene.model
        self.params = params or IKParams()
        p = self.params
        self.configuration = mink.Configuration(scene.model)
        self._rng = np.random.default_rng(rng_seed)
        nv = scene.model.nv

        self._tasks: dict[str, _FlatToleranceFrameTask] = {}
        self._jnt_range: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        vel_map: dict[str, float] = {}
        for arm_id, a in scene.addressing.arms.items():
            self._tasks[arm_id] = _FlatToleranceFrameTask(
                frame_name=f"{arm_id}_link_tcp",
                frame_type="site",
                position_cost=p.position_cost,
                orientation_cost=p.orientation_cost,
                lm_damping=1e-6,
            )
            self._tasks[arm_id].tol = np.array(p.flat_tolerances, dtype=np.float64)
            lo, hi = [], []
            for i in range(1, N_ARM_JOINTS + 1):
                rng = scene.model.joint(f"{arm_id}_joint{i}").range
                lo.append(float(rng[0]))
                hi.append(float(rng[1]))
                vel_map[f"{arm_id}_joint{i}"] = p.velocity_limits[f"joint{i}"]
            if a.has_rail:
                rng = scene.model.joint(f"{arm_id}_rail_joint").range
                lo.append(float(rng[0]))
                hi.append(float(rng[1]))
                vel_map[f"{arm_id}_rail_joint"] = p.velocity_limits["rail"]
            self._jnt_range[arm_id] = (np.array(lo), np.array(hi))

        self._posture = mink.PostureTask(scene.model, cost=_PIN_COST)
        self._smooth = _SmoothnessObjective(nv, p.w_accel, p.w_jerk)
        self._limits: list = [
            mink.ConfigurationLimit(scene.model),
            mink.VelocityLimit(scene.model, vel_map),
        ]
        self._collision_limit: _CappedCollisionLimit | None = None
        if collision_pairs is not None:
            self._collision_limit = _CappedCollisionLimit(
                scene.model,
                geom_pairs=collision_pairs,
                gain=p.collision_gain,
                minimum_distance_from_collisions=p.min_distance_m,
                collision_detection_distance=p.detection_distance_m,
                max_rows=p.max_collision_rows,
            )
            self._limits.append(self._collision_limit)
        # One-shot mode drops the VelocityLimit (positional query, no clock).
        self._limits_oneshot = [
            lim for lim in self._limits if not isinstance(lim, mink.VelocityLimit)
        ]
        self._one_shot = False

        # Per-arm solver state (core-order vectors of length dof).
        key0 = scene.model.key(0)
        self._q_warm: dict[str, np.ndarray] = {}
        self._q_meas: dict[str, np.ndarray] = {}
        for arm_id, a in scene.addressing.arms.items():
            q0 = np.array(key0.qpos[a.qpos_adr])
            self._q_warm[arm_id] = q0.copy()
            self._q_meas[arm_id] = q0.copy()
        self._dq_hist: dict[str, tuple[np.ndarray, np.ndarray]] = {
            arm_id: (np.zeros(nv), np.zeros(nv)) for arm_id in self._tasks
        }
        self._streak: dict[str, int] = dict.fromkeys(self._tasks, 0)
        self._w_orient: dict[str, float] = dict.fromkeys(
            self._tasks, p.orientation_cost
        )
        self._min_clearance: float | None = None
        self.configuration.update(self._compose_qpos(None))

    # -- state plumbing -------------------------------------------------------
    def _compose_qpos(self, active_arm: str | None) -> np.ndarray:
        """Full qpos: warm slice for the active arm, measured for the rest."""
        q = np.array(self.configuration.data.qpos)
        for arm_id, a in self.scene.addressing.arms.items():
            src = self._q_warm if arm_id == active_arm else self._q_meas
            q[a.qpos_adr] = src[arm_id]
        return q

    def sync_passive(self, states: Mapping[str, ArmState]) -> None:
        """Record measured configurations used to pin non-active arms."""
        for arm_id, state in states.items():
            if arm_id in self._q_meas:
                self._q_meas[arm_id] = np.array(state.q, dtype=np.float64)

    def set_min_clearance(self, d_min: float | None) -> None:
        """Feed the last twin clearance (m) to the ECAA weight adaptation."""
        self._min_clearance = d_min

    def set_flat_tolerances(self, arm_id: str, tol: np.ndarray) -> None:
        """Per-arm flat-bottom tolerances (x y z rx ry rz), refinement 3."""
        self._tasks[arm_id].tol = np.asarray(tol, dtype=np.float64).reshape(6).copy()

    def reset(self, arm_id: str, q_measured: np.ndarray) -> None:
        """Re-seed one arm and clear its accel/jerk history + residual streak.

        MANDATORY after recovery, takeover toggles, and planner handoffs.
        """
        a = self.scene.addressing[arm_id]
        q = np.asarray(q_measured, dtype=np.float64)
        if q.shape != (a.dof,):
            raise ValueError(f"{arm_id}: expected shape ({a.dof},), got {q.shape}")
        self._q_warm[arm_id] = q.copy()
        self._q_meas[arm_id] = q.copy()
        nv = self.model.nv
        self._dq_hist[arm_id] = (np.zeros(nv), np.zeros(nv))
        self._streak[arm_id] = 0
        self._w_orient[arm_id] = self.params.orientation_cost

    # -- one differential step (the 100 Hz path; never raises) ----------------
    def solve(self, arm_id: str, target: Pose, q_seed: np.ndarray | None) -> IKResult:
        t0 = time.perf_counter()
        p = self.params
        a = self.scene.addressing[arm_id]
        locked = p.lock_rail and a.has_rail and not self._one_shot
        if q_seed is not None:
            q_seed = np.asarray(q_seed, dtype=np.float64)
            if locked:  # the rail is an input, not a decision: follow it, no reset
                self._q_warm[arm_id][N_ARM_JOINTS] = q_seed[N_ARM_JOINTS]
            if np.max(np.abs(q_seed - self._q_warm[arm_id])) > p.reseed_threshold:
                self.reset(arm_id, q_seed)  # external motion / recovery / clamp
        self.configuration.update(self._compose_qpos(arm_id))
        active = a.dof_adr[:N_ARM_JOINTS] if locked else a.dof_adr

        task = self._tasks[arm_id]
        task.set_orientation_cost(self._update_ecaa(arm_id))
        task.set_target(_pose_to_se3(target))
        self._set_posture(arm_id, one_shot=self._one_shot)
        dq1, dq2 = self._dq_hist[arm_id]
        self._smooth.set_active(active, dq1, dq2)
        tasks = [task, self._posture, self._smooth]

        v = self._solve_qp(tasks, arm_id)
        if v is None:  # recovery ladder exhausted: hold (03-sim §12)
            self._streak[arm_id] += 1
            return IKResult(
                q=self._q_warm[arm_id].copy(),
                pos_err_m=float("nan"),
                rot_err_rad=float("nan"),
                diverged=True,
                active_collision_rows=self._active_rows(),
                solve_time_s=time.perf_counter() - t0,
            )
        mask = np.zeros(self.model.nv)
        mask[active] = 1.0
        v_masked = v * mask  # pin every non-active dof (and a locked rail) exactly
        self.configuration.integrate_inplace(v_masked, p.dt)

        err = task.compute_error(self.configuration)
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        if pos_err > p.pos_reject_m or rot_err > p.rot_reject_rad:
            self._streak[arm_id] += 1
        else:
            self._streak[arm_id] = 0

        q_out = np.array(self.configuration.data.qpos[a.qpos_adr])
        self._q_warm[arm_id] = q_out.copy()
        self._dq_hist[arm_id] = (v_masked * p.dt, dq1)
        return IKResult(
            q=q_out,
            pos_err_m=pos_err,
            rot_err_rad=rot_err,
            diverged=self._streak[arm_id] >= _DIVERGE_TICKS,
            active_collision_rows=self._active_rows(),
            solve_time_s=time.perf_counter() - t0,
        )

    def _active_rows(self) -> int:
        return self._collision_limit.last_active_rows if self._collision_limit else 0

    def _update_ecaa(self, arm_id: str) -> float:
        """Refinement 2: near-obstacle orientation-weight relaxation (ECAA)."""
        base = self.params.orientation_cost
        d_min = self._min_clearance
        if d_min is None and self._collision_limit is not None:
            d_min = self._collision_limit.last_min_dist
        s = 0.0 if d_min is None else max(0.0, (_ECAA_D_WARN - d_min) / _ECAA_D_WARN)
        target = base * _ECAA_A / (_ECAA_A + s)
        target = max(target, _ECAA_FLOOR * base)
        w = self._w_orient[arm_id]
        step = np.clip(target - w, -_ECAA_SLEW * base, _ECAA_SLEW * base)
        self._w_orient[arm_id] = w + step
        return self._w_orient[arm_id]

    def _set_posture(self, arm_id: str, one_shot: bool = False) -> None:
        """Per-DoF motion damping: target = current q, so the task penalizes
        Δq with weight cost² — joints cheap, rail expensive (servo mode),
        non-active dofs pinned. One-shot mode un-expenses the rail so far
        targets can place it (time-parameterization is the executor's job)."""
        a = self.scene.addressing[arm_id]
        cost = np.full(self.model.nv, _PIN_COST)
        cost[a.dof_adr[:N_ARM_JOINTS]] = self.params.posture_cost_joint
        if a.has_rail:
            if one_shot:
                rail_cost = self.params.posture_cost_joint
            elif self.params.lock_rail:
                rail_cost = _PIN_COST  # rail excluded from the servo solve
            else:
                rail_cost = self.params.posture_cost_rail
            cost[a.dof_adr[N_ARM_JOINTS]] = rail_cost
        self._posture.set_cost(cost)
        self._posture.set_target_from_configuration(self.configuration)

    def _solve_qp(self, tasks: list, arm_id: str) -> np.ndarray | None:
        """QP + the infeasibility recovery ladder (11-safety §8)."""
        p = self.params
        limits = self._limits_oneshot if self._one_shot else self._limits
        try:
            return mink.solve_ik(
                self.configuration, tasks, p.dt, p.solver, p.damping, limits=limits
            )
        except mink.NoSolutionFound:
            pass
        # Rung 1: retreat-only — drop the FrameTask, posture toward measured.
        self._posture.set_target(self._compose_qpos(None))
        try:
            return mink.solve_ik(
                self.configuration,
                [self._posture],
                p.dt,
                p.solver,
                p.damping,
                limits=limits,
            )
        except mink.NoSolutionFound:
            pass
        # Rung 2: 2 mm bound relaxation un-pins numerically conflicting rows.
        if self._collision_limit is not None:
            self._collision_limit.bound_relaxation = -0.002
            try:
                return mink.solve_ik(
                    self.configuration, tasks, p.dt, p.solver, p.damping, limits=limits
                )
            except mink.NoSolutionFound:
                pass
            finally:
                self._collision_limit.bound_relaxation = 0.0
        return None  # rung 3: hold (caller returns previous q, diverged=True)

    # -- one-shot far targets (goto / planner endpoints) -----------------------
    def solve_to_convergence(
        self,
        arm_id: str,
        target: Pose,
        q_seed: np.ndarray,
        max_steps: int = 50,
        n_restarts: int = 4,
    ) -> IKResult:
        """Iterate to < 1 mm / < 0.01 rad with multi-seed restarts.

        Restores the servo warm state afterwards; raises
        :class:`IKUnreachableError` (carrying the best attempt) when every
        seed fails. Among converged seeds, returns the one closest to
        ``q_seed`` in rail-weighted posture distance.
        """
        a = self.scene.addressing[arm_id]
        q_seed = np.asarray(q_seed, dtype=np.float64)
        snapshot = self._snapshot_state()
        seeds = [q_seed, self._home_seed(arm_id, q_seed)]
        lo, hi = self._jnt_range[arm_id]
        lo_r, hi_r = np.maximum(lo, -np.pi), np.minimum(hi, np.pi)
        for _ in range(max(0, n_restarts - 1)):
            seeds.append(self._rng.uniform(lo_r, hi_r))
        seeds = seeds[: 1 + n_restarts]
        converged: list[IKResult] = []
        best_fail: IKResult | None = None
        self._one_shot = True
        try:
            for seed in seeds:
                self.reset(arm_id, seed)
                result: IKResult | None = None
                polish = 0
                for _ in range(max_steps):
                    result = self.solve(arm_id, target, None)
                    if result.pos_err_m < 1e-3 and result.rot_err_rad < 0.01:
                        polish += 1  # criterion met: a few extra steps -> <0.1 mm
                        if polish > 10 or (
                            result.pos_err_m < 1e-4 and result.rot_err_rad < 1e-3
                        ):
                            break
                assert result is not None
                if result.pos_err_m < 1e-3 and result.rot_err_rad < 0.01:
                    if seed is seeds[0]:  # primary seed converged: done
                        return result
                    converged.append(result)
                elif best_fail is None or _fail_score(result) < _fail_score(best_fail):
                    best_fail = result
        finally:
            self._one_shot = False
            self._restore_state(snapshot)
        if converged:
            w = np.ones(a.dof)
            if a.has_rail:
                w[N_ARM_JOINTS] = 4.0  # rail excursions count 4x (03-sim §10)
            return min(converged, key=lambda r: float(np.linalg.norm(w * (r.q - q_seed))))
        assert best_fail is not None
        raise IKUnreachableError(best_fail)

    def _home_seed(self, arm_id: str, q_seed: np.ndarray) -> np.ndarray:
        a = self.scene.addressing[arm_id]
        home = np.array(HOME_Q, dtype=np.float64)
        if a.has_rail:
            home = np.append(home, q_seed[N_ARM_JOINTS])  # keep the seed's rail
        return home

    def _snapshot_state(self) -> tuple:
        return (
            np.array(self.configuration.data.qpos),
            {k: v.copy() for k, v in self._q_warm.items()},
            {k: v.copy() for k, v in self._q_meas.items()},
            {k: (h[0].copy(), h[1].copy()) for k, h in self._dq_hist.items()},
            dict(self._streak),
            dict(self._w_orient),
        )

    def _restore_state(self, snap: tuple) -> None:
        qpos, warm, meas, hist, streak, w_orient = snap
        self.configuration.update(qpos)
        self._q_warm = warm
        self._q_meas = meas
        self._dq_hist = hist
        self._streak = streak
        self._w_orient = w_orient


def _fail_score(r: IKResult) -> float:
    if not np.isfinite(r.pos_err_m):
        return float("inf")
    return r.pos_err_m + 0.1 * r.rot_err_rad


def _pose_to_se3(pose: Pose) -> mink.SE3:
    return mink.SE3(wxyz_xyz=np.concatenate([pose.orientation, pose.position]))


__all__ = [
    "IKParams",
    "MinkIKSolver",
    "default_collision_pairs",
]
