"""DigitalTwin — kinematic-only collision mirror (design 03-sim §8, 11-safety §6).

Implements core's ``DigitalTwinInterface`` over its OWN ``MjModel``/``MjData``
pair (build a second ``BuiltScene`` from the same scene id — never share the
physics model): per tick only ``mj_kinematics`` + ``mj_collision``, never
``mj_step``, so proximity contacts cannot perturb the robot.

Inflation (verified 3.12.0 semantics): ``geom_margin = 0`` and
``geom_gap = total/2`` per collidable geom — contacts are *detected* at
``margin + gap`` (sum over both geoms → total δ per pair) but generate forces
only inside ``margin``, i.e. detection-only (``efc_address == -1``).

Pair labels: a geom is labelled by its body name, except world-body geoms
(environment boxes/planes) which are labelled by their geom name — so events
read ("arm0_link5", "table") rather than ("arm0_link5", "world").
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from apollo_xarm7_core import (
    ArmState,
    CameraFrame,
    CollisionEvent,
    CollisionReport,
    PairClearance,
    PlanRequest,
    PlanResult,
)

from .errors import TwinAuditError
from .scenes.builder import BuiltScene

if TYPE_CHECKING:
    from .rendering import RenderService

TWIN_SOURCE = "twin"
DEFAULT_INFLATION_M = 0.008  # TOTAL pair inflation δ (SafetyConfig.geom_inflation_m)

Pair = frozenset  # frozenset({label1, label2})


def apply_inflation(model: mujoco.MjModel, total_gap_m: float) -> None:
    """Inflate every collidable geom by ``total_gap_m / 2`` (11-safety §6.2).

    ``margin`` stays 0 so the added contacts are detection-only; the per-pair
    detection threshold is the SUM of both geoms' gaps, hence δ/2 per geom
    yields the full δ between any two inflated geoms.
    """
    collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_gap[collidable] = total_gap_m / 2.0
    model.geom_margin[collidable] = 0.0


def geom_labels(model: mujoco.MjModel) -> tuple[str, ...]:
    """Per-geom pair label: owning body name, or geom name for world geoms."""
    labels = []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if body == 0:  # worldbody: label by the geom itself ("table", "floor")
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            labels.append(name if name else f"geom{g}")
        else:
            labels.append(model.body(body).name)
    return tuple(labels)


class AllowedPairs:
    """Label-pair whitelist: MJCF excludes + structural excludes + grasp lists.

    Sources (11-safety §6.3): (a) every MJCF ``<exclude>`` (re-asserted here —
    including the mandatory per-arm ``link_base ↔ link1`` and the rail chain
    authored in ``xarm7_on_rail.xml``), (b) ``safety.allowed_pairs_extra``,
    (c) session-scoped grasp whitelists per arm (``set_grasp``).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        extra: Iterable[tuple[str, str]] = (),
    ) -> None:
        self._labels = geom_labels(model)
        static: set[Pair] = set()
        for sig in model.exclude_signature:
            b1, b2 = int(sig) >> 16, int(sig) & 0xFFFF
            static.add(frozenset({model.body(b1).name, model.body(b2).name}))
        for a, b in extra:
            static.add(frozenset({a, b}))
        self._static = static
        self._grasp: dict[str, set[Pair]] = {}  # arm_id -> session pairs
        self._grasp_union: set[Pair] = set()

    def label_of_geom(self, geom_id: int) -> str:
        return self._labels[geom_id]

    def allows(self, geom1: int, geom2: int) -> bool:
        pair = frozenset({self._labels[geom1], self._labels[geom2]})
        return pair in self._static or pair in self._grasp_union

    def allows_labels(self, label1: str, label2: str) -> bool:
        pair = frozenset({label1, label2})
        return pair in self._static or pair in self._grasp_union

    def set_grasp(self, arm_id: str, pairs: set[Pair]) -> None:
        """Replace the session grasp whitelist for one arm."""
        self._grasp[arm_id] = pairs
        self._grasp_union = set().union(*self._grasp.values()) if self._grasp else set()

    @property
    def static_pairs(self) -> frozenset[Pair]:
        return frozenset(self._static)


def _contype_ok(model: mujoco.MjModel, g1: int, g2: int) -> bool:
    return bool(model.geom_contype[g1] & model.geom_conaffinity[g2]) or bool(
        model.geom_contype[g2] & model.geom_conaffinity[g1]
    )


def build_monitored_pairs(
    model: mujoco.MjModel,
    addressing,
    allowed: AllowedPairs,
) -> list[tuple[int, int]]:
    """Geom-id pairs for the ``clearance()`` sweep: cross-arm + arm↔env.

    Pairs survive the same filters MuJoCo's collision driver applies
    (contype/conaffinity, same-weld) plus the ``AllowedPairs`` label filter
    (which re-asserts every MJCF ``<exclude>``). Intra-arm pairs are excluded
    here — ``check()`` sees them through ``mj_collision``'s full contact list.
    """
    arm_geoms = {aid: a.geom_ids for aid, a in addressing.arms.items()}
    arm_ids = list(arm_geoms)
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for i, ai in enumerate(arm_ids):
        for aj in arm_ids[i + 1 :]:
            groups.append((arm_geoms[ai], arm_geoms[aj]))
        groups.append((arm_geoms[ai], addressing.env_geom_ids))
    weld = model.body_weldid[model.geom_bodyid]
    pairs: set[tuple[int, int]] = set()
    for geoms_a, geoms_b in groups:
        for ga in geoms_a:
            for gb in geoms_b:
                ga_i, gb_i = int(ga), int(gb)
                if weld[ga_i] == weld[gb_i]:
                    continue
                if not _contype_ok(model, ga_i, gb_i):
                    continue
                if allowed.allows(ga_i, gb_i):
                    continue
                pairs.add((min(ga_i, gb_i), max(ga_i, gb_i)))
    return sorted(pairs)


class DigitalTwin:
    """Kinematic mirror of the workcell for collision checking and planning.

    Pass a freshly built ``BuiltScene`` (same scene id as the sim/real cell,
    built separately): the twin takes ownership of its model and inflates it.
    Threading (03-sim §12): ``sync``/``check``/``clearance`` on the caller's
    control-loop thread only; ``plan`` may use a worker with a private
    ``MjData``; ``render`` goes through the render service's own thread.
    """

    def __init__(
        self,
        scene: BuiltScene,
        inflation_m: float = DEFAULT_INFLATION_M,
        render_service: RenderService | None = None,
        allowed_pairs_extra: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.scene = scene
        self.model = scene.model
        self.data = mujoco.MjData(scene.model)
        self.addr = scene.addressing
        self.inflation_m = float(inflation_m)
        apply_inflation(self.model, self.inflation_m)
        # Sources (11-safety §6.3): (a) scene-authored structural pairs,
        # (b) config allowed_pairs_extra, plus the built-in rail-vs-plane rule.
        extra = (
            list(scene.meta.allowed_pairs)
            + list(allowed_pairs_extra)
            + self._structural_extra()
        )
        self.allowed = AllowedPairs(self.model, extra=extra)
        self.monitored_pairs = build_monitored_pairs(self.model, self.addr, self.allowed)
        self._geoms_of_label = self._build_label_index()
        self._gripper_labels = self._build_gripper_labels()
        self._render_service = render_service
        if render_service is not None:
            render_service.register_source(TWIN_SOURCE, self.model)
        # Start at the scene keyframe (home); audit before arming the gate.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_kinematics(self.model, self.data)
        self._q_meas_full = np.array(self.data.qpos)
        self._audit_home()
        self._planner = None  # lazy ResetPlanner (plan())

    # -- init helpers --------------------------------------------------------
    def _structural_extra(self) -> list[tuple[str, str]]:
        """Scene-level structural pairs: rail platforms vs ground planes.

        A rail platform's z/x are mechanically fixed (slide along the rail
        axis only), so it can never reach a floor plane — yet its hull sits
        inside the inflation band above it permanently (at-home audit).
        """
        labels = geom_labels(self.model)
        planes = [
            labels[g]
            for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == 0
            and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        out = []
        for arm_id, a in self.addr.arms.items():
            if a.has_rail:
                out.extend((f"{arm_id}_rail_platform", p) for p in planes)
        return out

    def _build_label_index(self) -> dict[str, np.ndarray]:
        index: dict[str, list[int]] = {}
        collidable = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        for g in range(self.model.ngeom):
            if collidable[g]:
                index.setdefault(self.allowed.label_of_geom(g), []).append(g)
        return {label: np.array(gids, dtype=np.intp) for label, gids in index.items()}

    def _build_gripper_labels(self) -> dict[str, tuple[str, ...]]:
        """Per arm: labels of gripper bodies (grasp-whitelist scope)."""
        out: dict[str, tuple[str, ...]] = {}
        for arm_id, a in self.addr.arms.items():
            if not a.has_gripper:
                out[arm_id] = ()
                continue
            root = int(self.model.body(f"{arm_id}_xarm_gripper_base_link").id)
            labels = set()
            for g in a.geom_ids:
                body = int(self.model.geom_bodyid[g])
                b = body
                while b != 0:
                    if b == root:
                        labels.add(self.allowed.label_of_geom(int(g)))
                        break
                    b = int(self.model.body_parentid[b])
            out[arm_id] = tuple(sorted(labels))
        return out

    def _audit_home(self) -> None:
        """Refuse to arm the gate with unexplained at-home contacts (§6.3)."""
        mujoco.mj_collision(self.model, self.data)
        bad = self._violations()
        if bad:
            pairs = sorted({tuple(sorted(p)) for p, _ in bad})
            raise TwinAuditError(
                f"scene {self.scene.meta.id!r}: unexplained contacts at home under "
                f"inflation {self.inflation_m} m: {pairs}; add <exclude>/allowed "
                "pairs or fix the scene before arming the gate"
            )

    # -- per-tick core (control-loop thread) ---------------------------------
    def sync(self, states: Mapping[str, ArmState]) -> None:
        """Write measured q into the twin's qpos (incl. rail slots)."""
        for arm_id, state in states.items():
            self.data.qpos[self.addr[arm_id].qpos_adr] = state.q
        self._q_meas_full[:] = self.data.qpos
        if self._render_service is not None:
            self._render_service.submit_state(
                TWIN_SOURCE, self.data.qpos, time.monotonic()
            )

    def _violations(
        self, data: mujoco.MjData | None = None
    ) -> list[tuple[tuple[str, str], float]]:
        """Non-allowed contacts in ``data`` (default: own) as (label pair, dist)."""
        d = self.data if data is None else data
        ncon = d.ncon
        if ncon == 0:
            return []
        geoms = d.contact.geom[:ncon]
        dists = d.contact.dist[:ncon]
        out: list[tuple[tuple[str, str], float]] = []
        for (g1, g2), dist in zip(geoms, dists, strict=True):
            if self.allowed.allows(int(g1), int(g2)):
                continue
            pair = (
                self.allowed.label_of_geom(int(g1)),
                self.allowed.label_of_geom(int(g2)),
            )
            out.append((pair, float(dist)))
        return out

    def _arms_of_pair(self, pair: tuple[str, str]) -> list[str]:
        return sorted(
            arm_id
            for arm_id in self.addr.arms
            if any(label.startswith(f"{arm_id}_") for label in pair)
        )

    def check(self, q_by_arm: Mapping[str, np.ndarray]) -> CollisionReport:
        """Collision-check the COMMANDED config, all arms jointly (stateless)."""
        ts = time.monotonic()
        for arm_id, q in q_by_arm.items():
            self.data.qpos[self.addr[arm_id].qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        raw = self._violations()
        self.data.qpos[:] = self._q_meas_full  # stateless: restore measured
        if not raw:
            return CollisionReport(blocked=False, severity="ok", ts=ts)
        by_pair: dict[tuple[str, str], float] = {}
        for pair, dist in raw:
            key = tuple(sorted(pair))
            by_pair[key] = min(dist, by_pair.get(key, np.inf))
        events = [
            CollisionEvent(
                ts=ts,
                kind="penetration" if dist <= 0.0 else "blocked",
                pairs=[pair],
                dists_m=[dist],
                min_clearance_m=dist,
                arm_ids=self._arms_of_pair(pair),
            )
            for pair, dist in sorted(by_pair.items(), key=lambda kv: kv[1])
        ]
        return CollisionReport(
            blocked=True,
            severity="blocked",
            pairs=list(by_pair),
            min_clearance_m=min(by_pair.values()),
            violations=events,
            ts=ts,
        )

    # -- planner fast path ----------------------------------------------------
    def check_config(
        self, q_full: np.ndarray, *, data: mujoco.MjData | None = None
    ) -> bool:
        """True iff the FULL-model qpos vector is collision-free.

        ``data`` lets a planner worker use a private ``MjData`` (03-sim §10);
        the default uses the twin's own data (restored to measured afterwards).
        """
        detail = self.check_config_violations(q_full, data=data)
        return not detail

    def check_config_violations(
        self, q_full: np.ndarray, *, data: mujoco.MjData | None = None
    ) -> list[tuple[tuple[str, str], float]]:
        """Planner-detail variant of ``check_config``: violating (pair, dist)."""
        q_full = np.asarray(q_full, dtype=np.float64)
        if q_full.shape != (self.model.nq,):
            raise ValueError(f"q_full must have shape ({self.model.nq},), got {q_full.shape}")
        own = data is None
        d = self.data if own else data
        if own:
            saved = np.array(d.qpos)
        d.qpos[:] = q_full
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        out = self._violations(d)
        if own:
            d.qpos[:] = saved
        return out

    # -- clearance sweep (telemetry rate, not the 100 Hz gate) ----------------
    def clearance(self, distmax: float = 0.05) -> list[PairClearance]:
        """Measured-config clearances over the monitored pairs, ascending."""
        self.data.qpos[:] = self._q_meas_full
        mujoco.mj_kinematics(self.model, self.data)
        fromto = np.empty(6)
        out: list[PairClearance] = []
        for g1, g2 in self.monitored_pairs:
            dist = mujoco.mj_geomDistance(self.model, self.data, g1, g2, distmax, fromto)
            if dist >= distmax:
                continue
            out.append(
                PairClearance(
                    geom1=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                    or f"geom{g1}",
                    geom2=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
                    or f"geom{g2}",
                    body_pair=(
                        self.allowed.label_of_geom(g1),
                        self.allowed.label_of_geom(g2),
                    ),
                    dist_m=float(dist),
                    fromto=fromto.copy(),
                )
            )
        out.sort(key=lambda pc: pc.dist_m)
        return out

    def pair_distance(
        self,
        pair: tuple[str, str],
        q_by_arm: Mapping[str, np.ndarray] | None = None,
        distmax: float = 0.5,
    ) -> float:
        """Min ``mj_geomDistance`` between two labels at a given config.

        ``q_by_arm=None`` evaluates the measured config; otherwise the given
        commanded slots are written on top of measured (gate escape rule,
        11-safety §7 step 6). Stateless: measured qpos is restored.
        """
        if q_by_arm is not None:
            for arm_id, q in q_by_arm.items():
                self.data.qpos[self.addr[arm_id].qpos_adr] = q
        else:
            self.data.qpos[:] = self._q_meas_full
        mujoco.mj_kinematics(self.model, self.data)
        best = distmax
        for g1 in self._geoms_of_label.get(pair[0], ()):
            for g2 in self._geoms_of_label.get(pair[1], ()):
                dist = mujoco.mj_geomDistance(
                    self.model, self.data, int(g1), int(g2), distmax, None
                )
                best = min(best, dist)
        self.data.qpos[:] = self._q_meas_full
        return float(best)

    # -- session state ---------------------------------------------------------
    def set_grasp_whitelist(self, arm_id: str, body_names: list[str]) -> None:
        """Suppress collision pairs between the arm's gripper and held bodies."""
        if arm_id not in self.addr.arms:
            raise ValueError(f"unknown arm {arm_id!r}")
        gripper = self._gripper_labels[arm_id]
        pairs = {
            frozenset({g, b}) for g in gripper for b in body_names if g != b
        }
        self.allowed.set_grasp(arm_id, pairs)
        self.monitored_pairs = build_monitored_pairs(self.model, self.addr, self.allowed)

    # -- planning / rendering --------------------------------------------------
    def plan(self, req: PlanRequest) -> PlanResult:
        """Joint-space RRT-Connect over the twin model (03-sim §10)."""
        if self._planner is None:
            from .planner import ResetPlanner  # local: avoid import cycle

            self._planner = ResetPlanner(self, self.addr)
        return self._planner.plan(req)

    def render(self, view: str) -> CameraFrame | None:
        """Latest offscreen frame for a registered twin stream (render thread)."""
        if self._render_service is None:
            return None
        return self._render_service.latest(view)


__all__ = [
    "DEFAULT_INFLATION_M",
    "TWIN_SOURCE",
    "apply_inflation",
    "geom_labels",
    "AllowedPairs",
    "build_monitored_pairs",
    "DigitalTwin",
]
