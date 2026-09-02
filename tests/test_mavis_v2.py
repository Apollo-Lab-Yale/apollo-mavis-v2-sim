"""mavis_v2 — the Apollo lab cell (measured 2026-09-02), digital-twin reference.

Pins the scene to the tape measurements it was generated from (03-sim §4.3)
so a stray edit of the YAML cannot silently move the twin's obstacles.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import yaml

from apollo_xarm7_sim import REGISTRY, DigitalTwin
from apollo_xarm7_sim.assets import asset_path
from apollo_xarm7_sim.twin import geom_labels

TABLE_TOP_Z, HALF_L, HALF_W = 0.735, 0.6075, 0.315
RAIL_Z = 0.107188  # rail feet -> arm mounting plane (mavis rail mesh)
PLATFORM_TO_Q0_END = 0.098  # carriage edge beyond link_base toward the q=0 end
RAIL_ACROSS = (-0.120, 0.0724)  # rail mesh extent across the travel axis (rail frame x)


@pytest.fixture(scope="module")
def built():
    return REGISTRY.build("mavis_v2")


@pytest.fixture(scope="module")
def desc() -> dict:
    return yaml.safe_load(asset_path("scenes", "mavis_v2.yaml").read_text())


def test_meta_and_composition(built):
    m = built.meta
    assert m.arm_ids == ("view", "grip") and all(m.rail.values())
    assert m.cameras == ("cam_front", "cam_top", "view_wrist_cam", "grip_wrist_cam")
    assert m.suitable_for == {"sim", "twin"}
    assert not built.addressing["view"].has_gripper and built.addressing["grip"].has_gripper
    assert built.model.nq == 8 + 14 and built.model.nu == 8 + 9


def test_table_and_obstacle_match_measurements(built):
    model = built.model
    t, o = model.geom("table"), model.geom("obstacle")
    np.testing.assert_allclose(t.size, [HALF_L, HALF_W, 0.015])
    np.testing.assert_allclose(t.pos[2] + t.size[2], TABLE_TOP_Z)
    np.testing.assert_allclose(o.size, [0.08, 0.08, 0.13])
    np.testing.assert_allclose(o.pos[1] + o.size[1], HALF_W)  # flush with the back edge
    np.testing.assert_allclose(HALF_L - (o.pos[0] + o.size[0]), 0.292)  # +X face 29.2 cm from right
    np.testing.assert_allclose(o.pos[2] - o.size[2], TABLE_TOP_Z)  # standing on the table


def test_rails_run_along_x_with_zero_at_the_right_end(built):
    model, data = built.model, mujoco.MjData(built.model)
    for arm in ("view", "grip"):
        rail_base = model.body(f"{arm}_rail_base")
        np.testing.assert_allclose(rail_base.pos[2], TABLE_TOP_Z + RAIL_Z)
        for q, dx in ((0.0, 0.0), (0.65, -0.65)):  # travel increases toward -X
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[model.joint(f"{arm}_rail_joint").qposadr[0]] = q
            mujoco.mj_kinematics(model, data)
            lb = data.xpos[model.body(f"{arm}_link_base").id]
            np.testing.assert_allclose(lb[:2], rail_base.pos[:2] + [dx, 0.0], atol=1e-6)
    view, grip = model.body("view_rail_base").pos, model.body("grip_rail_base").pos
    assert view[0] == grip[0]  # both arms share the rail-zero X
    assert view[1] < grip[1]  # camera arm in front (-Y), gripper arm behind
    # rail zero 14.5 cm from the right table edge, measured to the carriage edge
    np.testing.assert_allclose(view[0] + PLATFORM_TO_Q0_END, HALF_L - 0.145, atol=1e-3)


def test_rail_edge_offsets_from_the_front_edge(desc):
    arms = {a["id"]: a for a in desc["arms"]}
    # yaw +90: rail-frame x -> world y, so the mesh spans [y0-0.120, y0+0.0724]
    view_y, grip_y = arms["view"]["base_pos"][1], arms["grip"]["base_pos"][1]
    np.testing.assert_allclose(view_y + RAIL_ACROSS[0], -HALF_W + 0.02, atol=1e-3)
    np.testing.assert_allclose(grip_y + RAIL_ACROSS[1], -HALF_W + 0.42, atol=1e-3)
    for a in arms.values():  # both arms face +Y (yaw +90 deg about Z), wxyz
        np.testing.assert_allclose(a["base_quat"], [0.70710678, 0, 0, 0.70710678], atol=1e-6)


def test_camera_only_arm_tool_is_collidable(built):
    model = built.model
    v, g = model.geom("view_d435"), model.geom("grip_d435")
    assert v.contype[0] == 1 and v.conaffinity[0] == 1  # camera IS the tool -> protected
    assert g.contype[0] == 0 and g.conaffinity[0] == 0  # next to the gripper hull: visual only
    assert "view_d435_mount" in geom_labels(model)


@pytest.mark.parametrize("delta", [0.008, 0.025])
def test_twin_audit_clean_and_structural_pairs_whitelisted(built, delta):
    twin = DigitalTwin(REGISTRY.build("mavis_v2"), inflation_m=delta)  # no TwinAuditError
    for a, b in built.meta.allowed_pairs:
        assert twin.allowed.allows_labels(a, b)
    labels = geom_labels(twin.model)
    monitored = {frozenset((labels[g1], labels[g2])) for g1, g2 in twin.monitored_pairs}
    assert frozenset(("view_rail_platform", "table")) not in monitored
    assert frozenset(("grip_rail_platform", "view_rail_base")) not in monitored
    assert frozenset(("grip_link5", "obstacle")) in monitored  # real hazards stay monitored
    assert frozenset(("view_d435_mount", "grip_link5")) in monitored


def test_keyframe_rails_at_zero_and_gripper_ready_above_the_obstacle(built):
    """Premise of the guardrail scenario ``mavis_v2_box_descend``."""
    model, data = built.model, mujoco.MjData(built.model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for arm in ("view", "grip"):
        assert data.qpos[model.joint(f"{arm}_rail_joint").qposadr[0]] == 0.0
    sid = model.site("grip_link_tcp").id
    tcp, tool_z = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)[:, 2]
    o = model.geom("obstacle")
    assert 0.08 <= tcp[2] - (o.pos[2] + o.size[2]) <= 0.15  # ~0.1 m above the box top
    assert abs(tcp[1] - o.pos[1]) <= o.size[1]  # inside the box's y span
    assert tcp[0] - 0.05 < o.pos[0] + o.size[0]  # gripper hull overlaps the box top in x
    assert tool_z[2] < -0.98  # tool pointing straight down
