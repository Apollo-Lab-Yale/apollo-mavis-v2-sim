"""mavis_v2 — the Apollo lab cell (tape-measured 2026-09-02), digital-twin reference.

Frame (user's definitions): +Y = OUTER edge (the long edge the arms face at rail
zero; camera rail side); facing the outer edge, +X is to the RIGHT (arms at rail
zero + obstacle). Pins the scene to the measurements it was generated from
(03-sim §4.3) so a stray edit of the YAML cannot silently move the twin's obstacles.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import yaml

from apollo_xarm7_sim import REGISTRY, DigitalTwin
from apollo_xarm7_sim.assets import asset_path
from apollo_xarm7_sim.twin import geom_labels

TABLE_TOP_Z, HALF_L, HALF_W = 0.735, 0.6075, 0.31
RAIL_Z = 0.107188  # rail feet -> arm mounting plane (mavis rail mesh)
RAIL_SPACING = 0.395  # between the two rail bases (identical rails, same orientation)
RAIL_ACROSS = (-0.120, 0.0724)  # rail mesh across the travel axis (rail frame x; plate on -x)
CARRIAGE_ACROSS = (-0.080, 0.090)
BASE_R = 0.063


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


def test_table_matches_measurements(built):
    t = built.model.geom("table")
    np.testing.assert_allclose(t.size, [HALF_L, HALF_W, 0.015])  # 1.215 x 0.62 x 0.03
    np.testing.assert_allclose(t.pos[2] + t.size[2], TABLE_TOP_Z)


def test_rails_same_orientation_zero_at_the_right_end(built):
    model, data = built.model, mujoco.MjData(built.model)
    for arm in ("view", "grip"):
        rail_base = model.body(f"{arm}_rail_base")
        np.testing.assert_allclose(rail_base.pos[2], TABLE_TOP_Z + RAIL_Z)
        # arm base +X -> world +Y (outer edge); travel increases toward -X
        np.testing.assert_allclose(rail_base.quat, [0.70710678, 0, 0, 0.70710678], atol=1e-6)
        for q, dx in ((0.0, 0.0), (0.65, -0.65)):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[model.joint(f"{arm}_rail_joint").qposadr[0]] = q
            mujoco.mj_kinematics(model, data)
            lb = data.xpos[model.body(f"{arm}_link_base").id]
            np.testing.assert_allclose(lb[:2], rail_base.pos[:2] + [dx, 0.0], atol=1e-6)
    view, grip = model.body("view_rail_base").pos, model.body("grip_rail_base").pos
    assert view[0] == grip[0]  # both arms share the rail-zero X
    assert view[1] > grip[1]  # camera arm on the outer rail, gripper arm inward
    # the rails' zero end is flush with the table's right edge (mesh: 0.2476 m past the base)
    np.testing.assert_allclose(view[0] + 0.2476, HALF_L, atol=1e-3)
    # ... which puts the carriage's right edge ~15 cm from it (user: "arm ~14 cm")
    assert 0.13 <= HALF_L - (view[0] + 0.098) <= 0.16


def test_rail_offsets_from_the_outer_edge(desc):
    arms = {a["id"]: a for a in desc["arms"]}
    view_y, grip_y = arms["view"]["base_pos"][1], arms["grip"]["base_pos"][1]
    np.testing.assert_allclose(HALF_W - (view_y + RAIL_ACROSS[1]), 0.026, atol=1e-3)  # 2.6 cm
    np.testing.assert_allclose(view_y - grip_y, RAIL_SPACING, atol=1e-3)  # 39.5 cm apart
    assert grip_y + RAIL_ACROSS[0] > -HALF_W  # plate edge stays on the table (0.66 cm)


def test_obstacle_at_the_left_end_of_the_channel(built):
    model = built.model
    o = model.geom("obstacle")
    np.testing.assert_allclose(o.size, [0.08, 0.08, 0.12])  # 16 x 16 x 24 cm
    np.testing.assert_allclose(o.pos[0] - o.size[0], -HALF_L)  # flush against the LEFT edge
    np.testing.assert_allclose(o.pos[2] - o.size[2], TABLE_TOP_Z)  # standing on the table
    np.testing.assert_allclose(HALF_W - (o.pos[1] + o.size[1]), 0.275)  # 27.5 cm from outer edge
    view_y = model.body("view_rail_base").pos[1]
    assert o.pos[1] + o.size[1] < view_y + RAIL_ACROSS[0]  # clear of the camera rail's plate
    # the gripper carriage never reaches it: 0.65 m of travel ends >5 cm short of its face
    grip_x = model.body("grip_rail_base").pos[0]
    assert (grip_x - 0.65 - 0.098) - (o.pos[0] + o.size[0]) > 0.05


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
    assert frozenset(("grip_link5", "obstacle")) in monitored  # real hazards stay monitored
    assert frozenset(("grip_rail_platform", "obstacle")) in monitored
    assert frozenset(("view_d435_mount", "grip_link5")) in monitored


def test_keyframe_rails_at_zero_and_gripper_ready_in_the_channel(built):
    """Premise of the guardrail scenario ``mavis_v2_rail_sweep`` (-X along the channel)."""
    model, data = built.model, mujoco.MjData(built.model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for arm in ("view", "grip"):
        assert data.qpos[model.joint(f"{arm}_rail_joint").qposadr[0]] == 0.0
    sid = model.site("grip_link_tcp").id
    tcp, tool_z = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)[:, 2]
    o = model.geom("obstacle")
    assert abs(tcp[1] - o.pos[1]) < o.size[1]  # inside the obstacle's y span
    assert TABLE_TOP_Z + 0.1 < tcp[2] < o.pos[2] + o.size[2]  # below the obstacle top
    assert tcp[0] > o.pos[0] + o.size[0] + 0.4  # far to the right of it (rail sweep to reach)
    assert tool_z[2] < -0.98  # tool pointing straight down
