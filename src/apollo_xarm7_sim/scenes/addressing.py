"""The rail-first <-> rail-last remapping boundary (design 03-sim §5).

MJCF stores the rail slide FIRST in ``qpos`` (it is the kinematic ancestor);
every external boundary (``ArmInterface``, ``IKSolver``, datasets) uses core
order — joints ``q[0:7]``, rail LAST at ``q[7]`` (01-core §4). ``Addressing``
precomputes per-arm index arrays IN CORE ORDER, so
``data.qpos[addr.qpos_adr]`` consumes/produces rail-last vectors directly.
All remapping lives here; nothing above this layer ever sees MJCF ordering.
Everything is resolved once at scene build — hot loops do zero name lookups.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .descriptor import SceneMeta

N_ARM_JOINTS = 7


@dataclass(frozen=True)
class ArmAddress:
    """Precomputed model indices for one arm (all vectors in core order)."""

    arm_id: str
    has_rail: bool
    has_gripper: bool
    dof: int  # 7, or 8 with rail (rail slot LAST)
    qpos_adr: np.ndarray  # (dof,) int — [j1..j7, rail?]
    dof_adr: np.ndarray  # (dof,) int — [j1..j7, rail?]
    ctrl_adr: np.ndarray  # (dof,) int — [act1..act7, rail?]
    gripper_ctrl_adr: int | None
    gripper_driver_qpos_adr: int | None  # left driver joint (open-frac readback)
    tcp_site_id: int
    wrist_cam_id: int | None
    base_body_id: int  # <arm>_link_base
    root_body_id: int  # <arm>_rail_base when railed, else link_base
    geom_ids: np.ndarray  # collidable geoms in the arm subtree


def _subtree_body_ids(model: mujoco.MjModel, root: int) -> set[int]:
    ids = set()
    stack = [root]
    while stack:
        b = stack.pop()
        ids.add(b)
        stack.extend(
            c for c in range(model.nbody) if model.body_parentid[c] == b and c != b
        )
    return ids


class Addressing:
    """Per-arm address book for a composed scene model."""

    def __init__(self, model: mujoco.MjModel, meta: SceneMeta) -> None:
        self.arms: dict[str, ArmAddress] = {}
        collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        arm_geoms: set[int] = set()
        for arm_id in meta.arm_ids:
            has_rail = meta.rail[arm_id]
            jnames = [f"{arm_id}_joint{i}" for i in range(1, N_ARM_JOINTS + 1)]
            if has_rail:
                jnames.append(f"{arm_id}_rail_joint")
            qpos_adr = np.array([model.joint(n).qposadr[0] for n in jnames], dtype=np.intp)
            dof_adr = np.array([model.joint(n).dofadr[0] for n in jnames], dtype=np.intp)
            anames = [f"{arm_id}_act{i}" for i in range(1, N_ARM_JOINTS + 1)]
            if has_rail:
                anames.append(f"{arm_id}_rail")
            ctrl_adr = np.array([model.actuator(n).id for n in anames], dtype=np.intp)
            has_gripper = _actuator_exists(model, f"{arm_id}_gripper")
            gripper_ctrl_adr = (
                int(model.actuator(f"{arm_id}_gripper").id) if has_gripper else None
            )
            gripper_driver_qpos_adr = (
                int(model.joint(f"{arm_id}_left_driver_joint").qposadr[0])
                if has_gripper
                else None
            )
            wrist_cam_id = (
                int(model.camera(f"{arm_id}_wrist_cam").id)
                if meta.wrist_cams.get(arm_id, False)
                else None
            )
            root_name = f"{arm_id}_rail_base" if has_rail else f"{arm_id}_link_base"
            root_body_id = int(model.body(root_name).id)
            subtree = _subtree_body_ids(model, root_body_id)
            geom_ids = np.array(
                sorted(
                    g
                    for g in range(model.ngeom)
                    if collidable[g] and int(model.geom_bodyid[g]) in subtree
                ),
                dtype=np.intp,
            )
            arm_geoms.update(int(g) for g in geom_ids)
            self.arms[arm_id] = ArmAddress(
                arm_id=arm_id,
                has_rail=has_rail,
                has_gripper=has_gripper,
                dof=N_ARM_JOINTS + (1 if has_rail else 0),
                qpos_adr=qpos_adr,
                dof_adr=dof_adr,
                ctrl_adr=ctrl_adr,
                gripper_ctrl_adr=gripper_ctrl_adr,
                gripper_driver_qpos_adr=gripper_driver_qpos_adr,
                tcp_site_id=int(model.site(f"{arm_id}_link_tcp").id),
                wrist_cam_id=wrist_cam_id,
                base_body_id=int(model.body(f"{arm_id}_link_base").id),
                root_body_id=root_body_id,
                geom_ids=geom_ids,
            )
        self.env_geom_ids: np.ndarray = np.array(
            sorted(g for g in range(model.ngeom) if collidable[g] and g not in arm_geoms),
            dtype=np.intp,
        )
        self.body_of_geom: tuple[str, ...] = tuple(
            model.body(int(model.geom_bodyid[g])).name for g in range(model.ngeom)
        )

    def __getitem__(self, arm_id: str) -> ArmAddress:
        return self.arms[arm_id]


def _actuator_exists(model: mujoco.MjModel, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0


__all__ = ["N_ARM_JOINTS", "ArmAddress", "Addressing"]
