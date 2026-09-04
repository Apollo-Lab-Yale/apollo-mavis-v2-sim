"""mavis_v2 — the Apollo lab cell (tape-measured 2026-09-02), digital-twin reference.

Frame (the operator is the authority on left/right): +Y = OUTER edge, where the
operator stands (the Perception Arm 'view' rides the rail next to it, so it is the
nearest arm; the Manipulation Arm 'grip' rides the inner rail). The operator faces
the arms (-Y); their RIGHT is -X and their LEFT is +X (the obstacle). Each chiral
rail is turned end-for-end (base_quat yaw +90) so its thin plate faces the interior
(-Y, away from the operator); the price is reversed travel -- rail zero (q=0) is at
the operator's LEFT (+X) and qpos increases toward -X. Initial state (keyframe,
user decision 2026-09-04): both arms at the xArm7 factory zero posture with joint 1
= pi, rails at OPPOSITE ends -- grip q = 0.65 (operator's right), view q = 0
(operator's left, next to the obstacle). Pins the scene to the measurements it was
generated from (03-sim §4.3) so a stray edit of the YAML cannot silently move the
twin's obstacles or its initial state.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import yaml

from apollo_mavis_v2_sim import REGISTRY, DigitalTwin, SceneOverrides
from apollo_mavis_v2_sim.assets import asset_path
from apollo_mavis_v2_sim.ik import default_collision_pairs
from apollo_mavis_v2_sim.scenes.builder import MIC_RADIUS_M, MIC_TIP_Z_M, WRIST_CAM_Z_M
from apollo_mavis_v2_sim.twin import geom_labels

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


def test_meta_and_composition(built, desc):
    m = built.meta
    assert m.arm_ids == ("view", "grip") and all(m.rail.values())
    assert m.cameras == ("cam_front", "cam_top", "view_wrist_cam", "grip_wrist_cam")
    assert m.suitable_for == {"sim", "twin"}
    assert not built.addressing["view"].has_gripper and built.addressing["grip"].has_gripper
    assert built.model.nq == 8 + 14 and built.model.nu == 8 + 9
    # the one scene the UI sees: titled, visible, mic OFF by default (hardware twin
    # switches it on through SceneOverrides.microphones)
    assert m.title == desc["title"] == "APOLLO MAVIS V2 Digital Twin" and not m.hidden
    assert m.microphones == {"view": False, "grip": False}
    arms = {a["id"]: a for a in desc["arms"]}
    assert arms["view"]["microphone"] is False and "microphone" not in arms["grip"]


def test_table_matches_measurements(built):
    t = built.model.geom("table")
    np.testing.assert_allclose(t.size, [HALF_L, HALF_W, 0.015])  # 1.215 x 0.62 x 0.03
    np.testing.assert_allclose(t.pos[2] + t.size[2], TABLE_TOP_Z)


def _rail_bar_world(model, data, arm):
    """World-space vertices of the ``{arm}_rail`` bar mesh at the current pose."""
    g = model.geom(f"{arm}_rail")
    mid = g.dataid[0]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[va : va + vn].reshape(-1, 3)
    R = data.geom_xmat[g.id].reshape(3, 3)
    return verts @ R.T + data.geom_xpos[g.id]


RAIL_BASE_TO_FAR_X = 0.845  # mavis mesh: base centre -> the -X (arm-rest) end


def test_rails_same_orientation_flush_with_the_operator_right_edge(built):
    model, data = built.model, mujoco.MjData(built.model)
    for arm in ("view", "grip"):
        rail_base = model.body(f"{arm}_rail_base")
        np.testing.assert_allclose(rail_base.pos[2], TABLE_TOP_Z + RAIL_Z)
        # base_quat yaw +90: each chiral rail turned end-for-end so its plate faces
        # -Y (interior); travel is reversed -> a rail qpos INCREASE drives -X
        np.testing.assert_allclose(rail_base.quat, [0.70710678, 0, 0, 0.70710678], atol=1e-6)
        for q, dx in ((0.0, 0.0), (0.65, -0.65)):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[model.joint(f"{arm}_rail_joint").qposadr[0]] = q
            mujoco.mj_kinematics(model, data)
            lb = data.xpos[model.body(f"{arm}_link_base").id]
            np.testing.assert_allclose(lb[:2], rail_base.pos[:2] + [dx, 0.0], atol=1e-6)
    view, grip = model.body("view_rail_base").pos, model.body("grip_rail_base").pos
    assert view[0] == grip[0]  # both rails share the base X
    assert view[1] > grip[1]  # Perception Arm on the outer rail, Manipulation Arm inward
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for arm in ("view", "grip"):
        # the thin plate now sits on the -Y (interior) side, away from the operator
        by = model.body(f"{arm}_rail_base").pos[1]
        bar = _rail_bar_world(model, data, arm)
        assert (by - bar[:, 1].min()) > (bar[:, 1].max() - by)  # more overhang toward -Y
        # each rail's -X end is flush with the table's -X edge
        np.testing.assert_allclose(bar[:, 0].min(), -HALF_L, atol=1e-3)
        np.testing.assert_allclose(model.body(f"{arm}_rail_base").pos[0] - RAIL_BASE_TO_FAR_X,
                                   -HALF_L, atol=1e-3)
    # x0 was derived from the 2026-09-02 measurement "arm ~14 cm from the edge" with
    # both arms at rest (rail q ~ 0.597): the mesh carriage's near edge is 15 cm in.
    # (Not the keyframe any more -- the initial state parks grip at q = 0.65, 9.7 cm.)
    for arm in ("view", "grip"):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[model.joint(f"{arm}_rail_joint").qposadr[0]] = 0.5974
        mujoco.mj_kinematics(model, data)
        lb = data.xpos[model.body(f"{arm}_link_base").id]
        assert 0.13 <= (lb[0] - 0.098) - (-HALF_L) <= 0.16


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
    # flush against the +X edge (operator's LEFT)
    np.testing.assert_allclose(o.pos[0] + o.size[0], HALF_L)
    np.testing.assert_allclose(o.pos[2] - o.size[2], TABLE_TOP_Z)  # standing on the table
    np.testing.assert_allclose(HALF_W - (o.pos[1] + o.size[1]), 0.275)  # 27.5 cm from outer edge
    view_y = model.body("view_rail_base").pos[1]
    assert o.pos[1] + o.size[1] < view_y + RAIL_ACROSS[0]  # clear of the camera rail's plate
    # the gripper carriage never reaches it: its max +X reach (rail q=0, base X) ends
    # >5 cm short of the obstacle's face
    grip_x = model.body("grip_rail_base").pos[0]
    assert (o.pos[0] - o.size[0]) - (grip_x + 0.098) > 0.05


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
    # intra-arm link2/link4: 1.78 cm apart at the factory-zero initial state, so the
    # scene whitelists it -- the audit and the gate's check() (mj_collision's full
    # contact list) ignore it; the clearance sweep never held intra-arm pairs anyway
    assert twin.allowed.allows_labels("grip_link2", "grip_link4")
    assert twin.allowed.allows_labels("view_link2", "view_link4")
    assert not any(all(x.startswith("grip_link") for x in p) for p in monitored)
    q0 = {a: twin._q_meas_full[twin.addr[a].qpos_adr] for a in ("view", "grip")}
    assert not twin.check(q0).blocked  # the gate is clean at the initial state


def test_keyframe_is_the_initial_state(built, desc):
    """Initial state (user decision 2026-09-04): both arms at the xArm7 factory zero
    posture with joint 1 = pi (joints 2-7 = 0), rails at OPPOSITE ends -- the
    Manipulation Arm 'grip' at q = 0.65 (the -X end, operator's right), the Perception
    Arm 'view' at q = 0 (the +X end, operator's left, next to the obstacle).

    The xArm7 zero is a FOLDED pose: the forearm hangs beside the upper arm and the
    tool points straight down (flange 12.05 cm above the mounting plane, 20.6 cm to
    the side). Joint 1 = pi puts that side at -Y, away from the operator.
    """
    kf = desc["keyframe"]
    np.testing.assert_allclose(kf["view"]["q"], [0.0, np.pi, 0, 0, 0, 0, 0, 0])  # rail FIRST
    np.testing.assert_allclose(kf["grip"]["q"], [0.65, np.pi, 0, 0, 0, 0, 0, 0])
    assert kf["grip"]["gripper"] == 1.0  # open
    model, data = built.model, mujoco.MjData(built.model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert data.qpos[model.joint("view_rail_joint").qposadr[0]] == 0.0
    assert data.qpos[model.joint("grip_rail_joint").qposadr[0]] == 0.65
    for arm in ("view", "grip"):
        assert data.qpos[model.joint(f"{arm}_joint1").qposadr[0]] == pytest.approx(np.pi)
        for j in range(2, 8):
            assert data.qpos[model.joint(f"{arm}_joint{j}").qposadr[0]] == 0.0
    x0 = model.body("grip_rail_base").pos[0]
    grip_base = data.xpos[model.body("grip_link_base").id]
    view_base = data.xpos[model.body("view_link_base").id]
    np.testing.assert_allclose(grip_base[0], x0 - 0.65)  # -0.4125: operator's right
    np.testing.assert_allclose(view_base[0], x0)  # +0.2375: operator's left
    assert grip_base[0] < 0 < view_base[0]
    o = model.geom("obstacle")
    # the Perception Arm's carriage stops short of the obstacle: 11.2 cm in x and
    # 9.7 cm outside its y span (carriage across-extent +Y of the base line)
    assert (o.pos[0] - o.size[0]) - (view_base[0] + 0.098) == pytest.approx(0.112, abs=1e-3)
    assert (view_base[1] + CARRIAGE_ACROSS[0]) - (o.pos[1] + o.size[1]) > 0.09
    for arm in ("view", "grip"):
        sid = model.site(f"{arm}_link_tcp").id
        tool_z = data.site_xmat[sid].reshape(3, 3)[:, 2]
        assert tool_z[2] < -0.98  # folded zero: tool points straight down
        flange = data.xpos[model.body(f"{arm}_link7").id]
        base = data.xpos[model.body(f"{arm}_link_base").id]
        np.testing.assert_allclose(flange - base, [0.0, -0.206, 0.1205], atol=1e-3)  # -Y side
    tcp = data.site_xpos[model.site("grip_link_tcp").id]
    assert tcp[1] < -HALF_W  # the gripper hangs just outside the table's inner edge
    assert tcp[2] > TABLE_TOP_Z + 0.05  # and above the table plane
    view_tcp = data.site_xpos[model.site("view_link_tcp").id]
    assert -0.111 < view_tcp[1] < 0.0916  # the D435 hangs into the channel, looking down


@pytest.mark.parametrize("delta", [0.008, 0.025])
def test_initial_state_clearances(delta):
    """Smallest monitored clearances at the keyframe (mic off) -- all outside the
    safety_debug band; the intra-arm link2/link4 pair (1.78 cm at the xArm zero) is
    whitelisted in the YAML so the audit and the gate ignore it (see the header)."""
    twin = DigitalTwin(REGISTRY.build("mavis_v2"), inflation_m=delta)
    for arm in ("view", "grip"):
        assert twin.allowed.allows_labels(f"{arm}_link2", f"{arm}_link4")
        assert twin.pair_distance((f"{arm}_link2", f"{arm}_link4"), None, 0.1) == pytest.approx(
            0.0178, abs=1e-3
        )
    assert twin.pair_distance(("grip_left_finger", "table"), None, 0.5) > 0.09
    assert twin.pair_distance(("grip_link_base", "table"), None, 0.5) > 0.10
    assert twin.pair_distance(("view_d435_mount", "grip_rail_base"), None, 0.5) > 0.12
    assert twin.pair_distance(("view_rail_platform", "obstacle"), None, 0.5) > 0.15
    nearest = twin.clearance(distmax=0.09)
    assert nearest == []  # nothing monitored within 9 cm at the initial state


def test_overviews_put_the_obstacle_on_the_image_left(built, desc):
    """Overview convention: every image is framed as the operator sees the cell --
    the red obstacle (the +X / empty end, operator's LEFT) on the LEFT, the arms on
    the RIGHT. All three overviews look from the +Y (operator) side toward -Y, so
    image right = -X and the top view has the near outer edge (+Y) at the bottom."""
    cams = {c["name"]: c for c in desc["cameras"]}
    assert cams["cam_front"]["pos"] == [0.0, 2.0, 1.9]
    assert cams["cam_front"]["xyaxes"] == [-1, 0, 0, 0, -0.55, 1]
    assert cams["cam_top"]["pos"] == [0.0, 0.0, 2.6]
    assert cams["cam_top"]["xyaxes"] == [-1, 0, 0, 0, -1, 0]
    model, data = built.model, mujoco.MjData(built.model)
    mujoco.mj_forward(model, data)
    table = model.geom("table")
    table_top = table.pos + [0.0, 0.0, table.size[2]]
    obstacle = model.geom("obstacle").pos
    for name in ("cam_front", "cam_top"):
        cid = model.camera(name).id
        rot = data.cam_xmat[cid].reshape(3, 3)  # columns x, y, z; right-handed: x cross y = z
        np.testing.assert_allclose(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2], atol=1e-6)
        assert rot[0, 0] < -0.99  # image right = -X
        look, to_table = -rot[:, 2], table_top - data.cam_xpos[cid]
        assert look @ to_table / np.linalg.norm(to_table) > 0.99  # aimed at the table centre
        # the obstacle projects LEFT of the table centre along the image-right axis
        assert (obstacle - table.pos) @ rot[:, 0] < 0
    top = data.cam_xmat[model.camera("cam_top").id].reshape(3, 3)
    np.testing.assert_allclose(top[:, 1], [0, -1, 0], atol=1e-9)  # image up = -Y (near +Y edge low)


def test_view_puts_the_obstacle_on_the_left(built, desc):
    """``view`` -> <visual><global azimuth elevation>: azimuth -90 puts MuJoCo's free
    camera (runtime ``sim`` stream) at +Y looking -Y, so the obstacle (the +X end)
    is on the LEFT of the image -- the operator's picture. From this side the
    camera-only arm (+Y) renders nearer than the gripper arm (-Y)."""
    assert desc["view"] == {"azimuth": -90.0, "elevation": -30.0}
    g = built.model.vis.global_
    assert (g.azimuth, g.elevation) == (-90.0, -30.0)
    az, el = np.radians(g.azimuth), np.radians(g.elevation)
    # mjv free camera: forward from azimuth/elevation, camera = lookat - distance * forward
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    np.testing.assert_allclose(forward, [0.0, -np.cos(el), np.sin(el)], atol=1e-12)
    assert forward[1] < 0  # looking toward -Y from the +Y side (camera = lookat - d * forward)
    view_y = built.model.body("view_rail_base").pos[1]
    grip_y = built.model.body("grip_rail_base").pos[1]
    assert (view_y * forward[1]) < (grip_y * forward[1])  # smaller depth along forward = nearer


# -- microphone body on the camera-only arm (03-sim §3) -----------------------


@pytest.fixture(scope="module")
def built_mic():
    return REGISTRY.build("mavis_v2", SceneOverrides(microphones={"view": True}))


def _mesh_verts_in_body(model, data, geom_name: str, body_name: str) -> np.ndarray:
    """Compiled mesh vertices of ``geom_name`` expressed in ``body_name``'s frame."""
    g = model.geom(geom_name)
    mid = g.dataid[0]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[va : va + vn].reshape(-1, 3)
    world = verts @ data.geom_xmat[g.id].reshape(3, 3).T + data.geom_xpos[g.id]
    b = model.body(body_name).id
    return (world - data.xpos[b]) @ data.xmat[b].reshape(3, 3)


def test_microphone_clears_the_side_mounted_camera(built_mic):
    """Recomputes the mic/camera gap from the COMPILED d435 mesh so a future edit
    of the camera pose or mount mesh fails here, not on the real arm.

    link7 frame: origin = flange face, +z = flange axis. wrist_cam sits at
    (0.07, 0, 0.05) looking +z; the D435 block starts at x = 0.055; the mic is a
    coaxial cylinder r = 0.040 from z = 0 to 0.19 (14 cm past the camera plane)
    -> 1.5 cm radial gap. Only the 3 mm mount plate lies inside the footprint
    (same welded body: MuJoCo never generates that pair)."""
    model, data = built_mic.model, mujoco.MjData(built_mic.model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    cam = model.camera("view_wrist_cam")
    assert model.body(cam.bodyid[0]).name == "view_d435_mount"
    np.testing.assert_allclose(cam.pos, [0.07, 0.0, WRIST_CAM_Z_M])
    look = data.cam_xmat[cam.id].reshape(3, 3)[:, 2]  # camera looks along -z of its frame
    flange_axis = data.xmat[model.body("view_link7").id].reshape(3, 3)[:, 2]
    assert -look @ flange_axis > 0.999  # LOOK = link7 +z: the mic points where the lens looks
    # d435_mount is welded at identity on link7 -> mount frame == link7 frame
    mount = model.body("view_d435_mount")
    np.testing.assert_allclose(mount.pos, 0.0, atol=1e-12)
    np.testing.assert_allclose(mount.quat, [1, 0, 0, 0], atol=1e-12)
    assert mount.weldid[0] == model.body("view_link7").weldid[0] == model.body(
        "view_microphone"
    ).weldid[0]
    verts = _mesh_verts_in_body(model, data, "view_d435", "view_link7")
    radial = np.hypot(verts[:, 0], verts[:, 1])
    block = verts[:, 2] > 0.004  # everything above the 3 mm mounting plate
    assert block.sum() > 1000
    assert verts[block, 0].min() == pytest.approx(0.055, abs=1e-3)  # camera block starts here
    gap = radial[block].min() - MIC_RADIUS_M
    assert gap == pytest.approx(0.015, abs=1e-3)  # 1.5 cm radial clearance
    assert verts[:, 2].max() < WRIST_CAM_Z_M  # the mesh front face is behind the camera plane
    assert MIC_TIP_Z_M == pytest.approx(WRIST_CAM_Z_M + 0.14)
    mic = model.geom("view_microphone")
    np.testing.assert_allclose(mic.pos[2] + mic.size[1], MIC_TIP_Z_M)  # tip at z = 0.19
    np.testing.assert_allclose(mic.pos[2] - mic.size[1], 0.0, atol=1e-12)  # base at the flange
    # the plate IS inside the footprint (radial >= hole radius 15 mm) -- same weld only
    plate = verts[:, 2] <= 0.0031
    assert radial[plate].min() < MIC_RADIUS_M


@pytest.mark.parametrize("delta", [0.008, 0.025])
def test_twin_audit_clean_with_microphone(built_mic, delta):
    twin = DigitalTwin(
        REGISTRY.build("mavis_v2", SceneOverrides(microphones={"view": True})), inflation_m=delta
    )  # no TwinAuditError at the keyframe (the parked camera arm clears its own shoulder)
    labels = geom_labels(twin.model)
    monitored = {frozenset((labels[g1], labels[g2])) for g1, g2 in twin.monitored_pairs}
    for other in ("table", "obstacle", "floor", "grip_link5", "grip_rail_platform"):
        assert frozenset(("view_microphone", other)) in monitored
    assert frozenset(("view_microphone", "view_d435_mount")) not in monitored  # same weld
    assert twin._arms_of_pair(("view_microphone", "obstacle")) == ["view"]
    mic_geom = int(built_mic.model.geom("view_microphone").id)
    assert any(mic_geom in a or mic_geom in b for a, b in default_collision_pairs(twin.scene))
    # keyframe clearances (folded initial state: the mic points DOWN at the table from
    # the channel, tip at z ~ 0.773): 3.8 cm to the table -- outside the 2.5 cm
    # safety_debug band but the tightest clearance of the whole initial state
    assert 0.03 < twin.pair_distance(("view_microphone", "table"), None, 1.0) < 0.045
    assert twin.pair_distance(("view_microphone", "grip_rail_base"), None, 1.0) > 0.07
    assert twin.pair_distance(("view_microphone", "obstacle"), None, 1.0) > 0.15
    assert twin.pair_distance(("view_microphone", "view_link1"), None, 1.0) > 0.1
    assert twin.pair_distance(("view_microphone", "grip_link6"), None, 1.0) > 0.5

