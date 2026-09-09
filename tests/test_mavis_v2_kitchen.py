"""mavis_v2_kitchen — the GELLO Manipulation digital twin (16-gello §3 / §10, phase-15).

The lab cell of ``mavis_v2`` plus the kitchen the Perception Arm's wrist camera saw
on 2026-09-09: a GE GDE21ESKSS refrigerator, a 30-inch GE range, the counter run,
the upper cabinets and the wall as dimensioned BOXES on the depth-measured faces, and
the four AprilTags (tagStandard41h12 ids 0 / 1 / 3 / 4) as non-collidable textured
plates. Pins: the cell blocks are byte-equal to ``mavis_v2``'s (one geometry
authority), the boxes match the §3 ranges, the twin audits clean at both inflations
with the microphone on and off, the handles are graspable, the plates never reach the
gate, and a render from ``view_wrist_cam`` at the GELLO hold posture reproduces the
REAL tag detections of the measurement frame (``tests/data/kitchen_tags_20260909.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from apollo_mavis_v2_sim import REGISTRY, DigitalTwin, SceneOverrides
from apollo_mavis_v2_sim.assets import asset_path
from apollo_mavis_v2_sim.twin import geom_labels

KITCHEN = "mavis_v2_kitchen"
GRASPABLE = ("fridge_door_handle", "fridge_drawer_handle", "range_handle")
TAG_PLATES = {"tag_0": 0, "tag_1": 1, "tag_3": 3, "tag_4": 4}
HOLD_POSTURE = (2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029)  # J1..J7 rad, rail 0.0
# 16-gello §3 "Resulting boxes": (x range, y range, z range) in world metres.
BOX_RANGES = {
    "fridge_body": ((-0.681, 0.075), (-1.856, -1.027), (0.0, 1.775)),
    "fridge_door_handle": ((0.015, 0.045), (-1.027, -0.977), (0.75, 1.65)),
    "fridge_drawer_handle": ((-0.545, -0.055), (-1.027, -0.977), (0.475, 0.510)),
    "range_body": ((0.480, 1.239), (-1.883, -1.222), (0.0, 0.914)),
    "range_backguard": ((0.480, 1.239), (-1.883, -1.783), (0.914, 1.194)),
    "range_handle": ((0.530, 1.189), (-1.222, -1.172), (0.600, 0.630)),
    "counter": ((0.075, 0.480), (-1.889, -1.279), (0.0, 0.914)),
    "upper_cabinet": ((0.075, 1.289), (-1.883, -1.578), (1.372, 2.134)),
    "kitchen_wall": ((-1.00, 1.50), (-1.933, -1.883), (0.0, 2.40)),
}
# 16-gello §3 "Tags": measured centres (m) and the face each plate stands on.
TAG_CENTRES = {
    "tag_0": ((0.075, -1.314, 1.426), (1.0, 0.0, 0.0)),
    "tag_4": ((0.076, -1.142, 0.543), (1.0, 0.0, 0.0)),
    "tag_1": ((-0.360, -1.027, 1.489), (0.0, 1.0, 0.0)),
    "tag_3": ((0.616, -1.222, 0.498), (0.0, 1.0, 0.0)),
}
PLATE_HALF = 0.1024  # 0.205 m sheet: 9-bit tag + one white bit of margin at 18.6 mm / bit
# Real view_wrist colour intrinsics (configs/mavis_v2.yaml) + the overlay's principal-point
# nudge for view_wrist ([21, 13]): the convention every kitchen measurement was made in.
FX, FY, CX, CY, W, H = 606.36, 606.38, 311.90 + 21.0, 249.45 + 13.0, 640, 480
REAL_TAGS = Path(__file__).parent / "data" / "kitchen_tags_20260909.json"


@pytest.fixture(scope="module")
def built():
    return REGISTRY.build(KITCHEN)


@pytest.fixture(scope="module")
def desc() -> dict:
    return yaml.safe_load(asset_path("scenes", f"{KITCHEN}.yaml").read_text())


@pytest.fixture(scope="module")
def cell() -> dict:
    return yaml.safe_load(asset_path("scenes", "mavis_v2.yaml").read_text())


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {r["name"]: r for r in rows}


# -- registry row / composition ------------------------------------------------------------


def test_meta_is_hidden_titled_and_declares_the_handles(built, desc):
    m = built.meta
    assert m.id == KITCHEN and m.hidden is True  # mavis_v2 stays the only listed scene
    assert m.title == desc["title"] == "APOLLO MAVIS V2 Kitchen (GELLO)"
    assert m.suitable_for == {"sim", "twin"}
    assert m.arm_ids == ("view", "grip") and all(m.rail.values())
    assert m.cameras == ("cam_front", "cam_top", "cam_kitchen", "view_wrist_cam", "grip_wrist_cam")
    assert m.graspable == GRASPABLE == tuple(desc["graspable"])
    assert m.microphones == {"view": False, "grip": False}
    assert built.model.nq == 8 + 14 and built.model.nu == 8 + 9  # same arms as mavis_v2
    assert [row.id for row in REGISTRY.list()] == ["mavis_v2"]


def test_cell_blocks_are_verbatim_copies_of_mavis_v2(desc, cell):
    """One geometry authority: arms, the two overview cameras, table, obstacle, the
    structural pairs and the Manipulation Arm's keyframe equal mavis_v2.yaml's."""
    assert desc["arms"] == cell["arms"]
    cams, cell_cams = _by_name(desc["cameras"]), _by_name(cell["cameras"])
    for name in ("cam_front", "cam_top"):
        assert cams[name] == cell_cams[name]
    env, cell_env = _by_name(desc["environment"]), _by_name(cell["environment"])
    for name in ("table", "obstacle"):
        assert env[name] == cell_env[name]
    assert desc["allowed_pairs"] == cell["allowed_pairs"]
    assert desc["keyframe"]["grip"] == cell["keyframe"]["grip"]


def test_keyframe_puts_the_perception_arm_at_the_gello_hold_posture(built, desc):
    kf = desc["keyframe"]
    np.testing.assert_allclose(kf["view"]["q"], [0.0, *HOLD_POSTURE])  # rail FIRST
    model, data = built.model, mujoco.MjData(built.model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    q_view = data.qpos[built.addressing["view"].qpos_adr]  # core order: j1..j7, rail
    np.testing.assert_allclose(q_view, [*HOLD_POSTURE, 0.0])
    q_grip = data.qpos[built.addressing["grip"].qpos_adr]
    np.testing.assert_allclose(q_grip, [np.pi, 0, 0, 0, 0, 0, 0, 0.65])
    # 16-gello §3 "Camera pose": the measurement camera, from the twin at this posture
    cid = model.camera("view_wrist_cam").id
    np.testing.assert_allclose(data.cam_xpos[cid], [0.460, 0.394, 1.579], atol=1e-3)
    axis = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(axis, [-0.243, -0.907, -0.343], atol=1e-3)


# -- the kitchen geometry pinned to the 2026-09-09 measurement -----------------------------


def test_appliance_boxes_match_the_measured_ranges(built):
    model = built.model
    for name, ranges in BOX_RANGES.items():
        g = model.geom(name)
        assert g.type[0] == mujoco.mjtGeom.mjGEOM_BOX and g.contype[0] == 1
        size = [(hi - lo) / 2 for lo, hi in ranges]
        pos = [(hi + lo) / 2 for lo, hi in ranges]
        np.testing.assert_allclose(g.size, size, atol=1e-4, err_msg=name)
        np.testing.assert_allclose(g.pos, pos, atol=1e-4, err_msg=name)
    # the anchor planes the depth image gave: fridge side x = 0.075, fridge door
    # y = -1.027, range front y = -1.222, drawer fronts y = -1.279, counter/range top 0.914
    fridge, rng = model.geom("fridge_body"), model.geom("range_body")
    counter = model.geom("counter")
    assert fridge.pos[0] + fridge.size[0] == pytest.approx(0.075, abs=1e-4)
    assert fridge.pos[1] + fridge.size[1] == pytest.approx(-1.027, abs=1e-4)
    assert rng.pos[1] + rng.size[1] == pytest.approx(-1.222, abs=1e-4)
    assert counter.pos[1] + counter.size[1] == pytest.approx(-1.279, abs=1e-4)
    assert rng.pos[2] + rng.size[2] == counter.pos[2] + counter.size[2] == pytest.approx(0.914)
    # the handles stand proud of their doors (graspable bars, 5 cm deep)
    door = model.geom("fridge_door_handle")
    assert door.pos[1] - door.size[1] == pytest.approx(-1.027, abs=1e-4)
    assert model.geom("range_handle").pos[1] - 0.025 == pytest.approx(-1.222, abs=1e-4)


def test_tag_plates_sit_on_their_faces_upright(built):
    """Each plate: 1 mm thick, 0.205 m square, standing <= 1 mm proud of its face at the
    measured centre, local +x = the outward face normal (the textured side), local +z up."""
    model, data = built.model, mujoco.MjData(built.model)
    mujoco.mj_forward(model, data)
    for name, (centre, normal) in TAG_CENTRES.items():
        g = model.geom(name)
        np.testing.assert_allclose(g.size, [0.0005, PLATE_HALF, PLATE_HALF])
        np.testing.assert_allclose(g.pos, centre, atol=1.5e-3, err_msg=name)
        rot = data.geom_xmat[g.id].reshape(3, 3)
        np.testing.assert_allclose(rot[:, 0], normal, atol=1e-6, err_msg=name)  # plate normal
        np.testing.assert_allclose(rot[:, 2], [0, 0, 1], atol=1e-6, err_msg=name)  # tag up
        face = np.array(centre) @ np.array(normal)
        proud = g.pos @ np.array(normal) - g.size[0] - face
        assert -0.0011 <= proud <= 0.0011, (name, proud)  # back face on / at the appliance face


def test_tag_plates_are_textured_visual_only_and_never_monitored(built):
    model = built.model
    xml = built.xml
    assert model.ntex == 4 and 'texturedir="' in xml
    for name, tag_id in TAG_PLATES.items():
        g = model.geom(name)
        assert g.contype[0] == 0 and g.conaffinity[0] == 0 and g.group[0] == 1
        assert g.matid[0] >= 0  # textured through a material
        tex = model.mat(int(g.matid[0])).texid[mujoco.mjtTextureRole.mjTEXROLE_RGB]
        assert model.tex_width[tex] == 704 and model.tex_height[tex] == 704
        assert f'file="textures/tagStandard41h12_{tag_id:05d}.png"' in xml
    env = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
        for g in built.addressing.env_geom_ids
    }
    assert set(BOX_RANGES) | {"table", "obstacle", "floor"} <= env
    assert not set(TAG_PLATES) & env
    twin = DigitalTwin(REGISTRY.build(KITCHEN), inflation_m=0.025)
    labels = geom_labels(twin.model)
    monitored_labels = {labels[g] for pair in twin.monitored_pairs for g in pair}
    assert not set(TAG_PLATES) & monitored_labels
    assert set(BOX_RANGES) <= monitored_labels
    for name in TAG_PLATES:
        assert twin.model.geom_gap[twin.model.geom(name).id] == 0.0  # never inflated
    mujoco.MjModel.from_xml_string(xml)  # the persisted scene XML re-compiles (file textures)


# -- twin audit + graspable handles -------------------------------------------------------


@pytest.mark.parametrize("mic", [False, True], ids=["mic_off", "mic_on"])
@pytest.mark.parametrize("delta", [0.008, 0.025])
def test_twin_audit_clean_at_the_keyframe(delta, mic):
    built = REGISTRY.build(KITCHEN, SceneOverrides(microphones={"view": mic}))
    twin = DigitalTwin(built, inflation_m=delta)  # no TwinAuditError
    for a, b in built.meta.allowed_pairs:
        assert twin.allowed.allows_labels(a, b)
    labels = geom_labels(twin.model)
    monitored = {frozenset((labels[g1], labels[g2])) for g1, g2 in twin.monitored_pairs}
    # every appliance body is gated against the arm links (the fridge is the one in reach)
    for link in ("grip_link5", "grip_link7", "grip_left_finger", "view_link7", "view_d435_mount"):
        assert frozenset((link, "fridge_body")) in monitored
        assert frozenset((link, "fridge_door_handle")) in monitored
    assert frozenset(("grip_link5", "obstacle")) in monitored  # the cell's own hazards stay
    if mic:
        assert frozenset(("view_microphone", "fridge_body")) in monitored
    q0 = {a: twin._q_meas_full[twin.addr[a].qpos_adr] for a in ("view", "grip")}
    assert not twin.check(q0).blocked  # the gate is clean at the initial state
    assert twin.clearance(distmax=0.05) == []  # nothing monitored within 5 cm at the keyframe


def test_graspable_whitelist_frees_the_fingers_but_not_the_arm(built):
    twin = DigitalTwin(REGISTRY.build(KITCHEN), inflation_m=0.008)
    labels = geom_labels(twin.model)
    gripper = set(twin._gripper_labels["grip"])
    assert {"grip_left_finger", "grip_right_finger"} <= gripper

    def monitored():
        return {frozenset((labels[g1], labels[g2])) for g1, g2 in twin.monitored_pairs}

    before = monitored()
    for handle in built.meta.graspable:
        assert frozenset(("grip_left_finger", handle)) in before
    twin.set_grasp_whitelist("grip", list(built.meta.graspable))
    after = monitored()
    for handle in built.meta.graspable:
        for label in gripper:
            assert frozenset((label, handle)) not in after, (label, handle)
        assert frozenset(("grip_link6", handle)) in after  # arm links stay gated vs handles
        assert twin.allowed.allows_labels("grip_left_finger", handle)
    assert frozenset(("grip_left_finger", "fridge_body")) in after  # only the handles open
    assert len(before) - len(after) == len(gripper) * len(built.meta.graspable)
    twin.set_grasp_whitelist("grip", [])
    assert monitored() == before
    twin.set_grasp_whitelist("view", list(built.meta.graspable))  # no gripper: a no-op
    assert monitored() == before


# -- the four tags seen from the measurement camera --------------------------------------


def _render_view_wrist(mic: bool, k: int) -> np.ndarray:
    """Render ``view_wrist_cam`` at the keyframe with the REAL colour intrinsics (+ the
    overlay nudge) applied to the MjSpec camera before compile — the twin-overlay recipe
    (runtime ``streams/twin_overlay.py`` ``apply_intrinsics``) — supersampled ``k`` times
    (the same pinhole camera: every intrinsic scales with the image)."""
    built = REGISTRY.build(KITCHEN, SceneOverrides(microphones={"view": mic}))
    spec = built.spec
    w, h = W * k, H * k
    cam = spec.camera("view_wrist_cam")
    cam.resolution = [w, h]
    cam.sensor_size = [w * 1e-5, h * 1e-5]
    cam.focal_pixel = [FX * k, FY * k]
    cam.principal_pixel = [w / 2 - CX * k, h / 2 - CY * k]  # MuJoCo's sign is OpenCV's opposite
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), w)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), h)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # view at the hold posture, grip at the cell's
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=h, width=w)
    try:
        renderer.update_scene(data, camera="view_wrist_cam")
        return renderer.render().copy()
    finally:
        renderer.close()


def _detect(rgb: np.ndarray, k: int) -> dict[int, dict]:
    from pupil_apriltags import Detector

    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    out = {}
    for t in Detector(families="tagStandard41h12").detect(gray):
        out[int(t.tag_id)] = {
            "hamming": int(t.hamming),
            "center": np.asarray(t.center) / k,
            "corners": np.asarray(t.corners) / k,
        }
    return out


@pytest.mark.egl
def test_tags_detected_from_view_wrist_cam_match_the_real_frame():
    """The kitchen twin, rendered from the camera that measured it, reproduces the real
    2026-09-09 detections: ids 0 / 1 / 3 / 4, centres and corners within 25 px, upright
    and unmirrored (the detector's corner order is bottom-left, bottom-right, top-right,
    top-left in the image, as in the real frame). The fridge-side plates (0, 4) are seen
    edge-on (~10 px wide at 640 x 480); MuJoCo's isotropic mipmapping blurs them below
    decodability at native resolution, so the four-tag assertion renders the same pinhole
    camera supersampled 2x and halves the pixel coordinates; the native render must still
    decode the two frontal tags (1 on the door, 3 on the range)."""
    real = {int(t["id"]): t for t in json.loads(REAL_TAGS.read_text())}
    assert set(real) == {0, 1, 3, 4}

    def check(found: dict[int, dict], ids: set[int]) -> None:
        assert ids <= set(found), (sorted(found), sorted(ids))
        for tag_id in ids:
            got, ref = found[tag_id], real[tag_id]
            assert got["hamming"] <= 2
            err = np.linalg.norm(got["center"] - np.asarray(ref["center"]))
            assert err < 25.0, (tag_id, err, got["center"], ref["center"])
            corner_err = np.linalg.norm(got["corners"] - np.asarray(ref["corners"]), axis=1)
            assert corner_err.max() < 25.0, (tag_id, corner_err)  # same corner ORDER = same pose
            c = got["corners"]
            assert (c[3] - c[0])[1] < 0 and (c[1] - c[0])[0] > 0, (tag_id, c)  # upright, unmirrored

    # mic OFF (the real frame shows no occlusion; the twin's modelled mic body, still
    # unverified per 03-sim §4.3, covers tag 4 at the bottom-centre of the image)
    check(_detect(_render_view_wrist(mic=False, k=2), 2), {0, 1, 3, 4})
    check(_detect(_render_view_wrist(mic=False, k=1), 1), {1, 3})
    # mic ON (the hardware twin): the three unoccluded tags still line up
    check(_detect(_render_view_wrist(mic=True, k=2), 2), {0, 1, 3})


@pytest.mark.egl
def test_cam_kitchen_frames_the_fridge_front_and_the_manipulation_arm(built, desc):
    """The GELLO launch-preview camera: operator side (+Y), looking -Y and down, image
    right = -X (the operator convention); the fridge door face and the Manipulation Arm at
    its keyframe project inside the frame."""
    cam = _by_name(desc["cameras"])["cam_kitchen"]
    assert cam["pos"][1] > 0.6 and cam["xyaxes"][:3] == [-1, 0, 0]
    model, data = built.model, mujoco.MjData(built.model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    cid = model.camera("cam_kitchen").id
    rot, pos = data.cam_xmat[cid].reshape(3, 3), data.cam_xpos[cid]
    assert rot[0, 0] < -0.99 and (-rot[:, 2])[1] < -0.7 and (-rot[:, 2])[2] < 0  # -Y and down
    fovy = np.radians(model.cam_fovy[cid])
    fovx = 2 * np.arctan(np.tan(fovy / 2) * 640 / 480)

    def in_frame(p):
        v = rot.T @ (np.asarray(p) - pos)  # camera frame: x right, y up, looks along -z
        depth = -v[2]
        if depth <= 0:
            return False
        return abs(v[0] / depth) < np.tan(fovx / 2) and abs(v[1] / depth) < np.tan(fovy / 2)

    fridge = model.geom("fridge_body")
    ymax = fridge.pos[1] + fridge.size[1]
    for x in (fridge.pos[0] - fridge.size[0], fridge.pos[0] + fridge.size[0]):
        for z in (0.3, fridge.pos[2] + fridge.size[2]):
            assert in_frame([x, ymax, z]), (x, z)  # the door face's corners
    for body in ("grip_link_base", "grip_link4", "grip_link7"):
        assert in_frame(data.xpos[model.body(body).id]), body
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera="cam_kitchen")
        rgb = renderer.render()
        assert rgb.any() and rgb.mean() > 20  # not black: the appliances fill the frame
    finally:
        renderer.close()
