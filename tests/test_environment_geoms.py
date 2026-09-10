"""``EnvironmentSpec`` mesh / texture / collidable / group (03-sim §4.1 / §4.4).

Textured plates and file meshes for the kitchen twin: the descriptor validates the
field combinations, the builder emits FILE textures (``spec.to_xml()`` refuses
buffer textures and the XML is persisted with every episode), and ``collidable:
false`` geoms never become contacts, monitored pairs or inflated twin geoms.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from apollo_mavis_v2_sim import SceneCompileError, SceneDescriptor, build_scene
from apollo_mavis_v2_sim.scenes import EnvironmentSpec, scene_meta

TAG_PNG = "textures/tagStandard41h12_00000.png"
ARM = {"id": "a0", "model": "xarm7_fixed", "base_pos": (0.0, 2.0, 0.0)}


def _desc(**kw) -> SceneDescriptor:
    return SceneDescriptor(id="_env", description="environment geom fixture", arms=(ARM,), **kw)


# -- descriptor validation ---------------------------------------------------------


def test_defaults_keep_the_phase_02_box_shape():
    e = EnvironmentSpec(name="table", type="box", size=(0.5, 0.5, 0.01))
    assert e.collidable is True and e.group is None and e.texture is None
    assert e.mesh is None and e.scale == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"type": "mesh"}, "requires 'mesh'"),
        ({"type": "mesh", "mesh": "rail/linear_motor_platform.stl", "size": (1, 1, 1)}, "scale"),
        ({"type": "box", "size": (1, 1, 1), "mesh": "rail/linear_motor_rail.stl"}, "only valid"),
        ({"type": "plane", "mesh": "rail/linear_motor_platform.stl"}, "only valid"),
        ({"type": "box"}, "requires 'size'"),
        ({"type": "plane"}, "requires 'size'"),
        ({"type": "mesh", "mesh": "/abs/path.stl"}, "asset-relative"),
        ({"type": "mesh", "mesh": "../escape.stl"}, "asset-relative"),
        ({"type": "box", "size": (1, 1, 1), "texture": "/etc/passwd.png"}, "asset-relative"),
        ({"type": "box", "size": (1, 1, 1), "group": 6}, "group"),
        ({"type": "box", "size": (1, 1, 1), "group": -1}, "group"),
    ],
)
def test_invalid_field_combinations_are_rejected(kw, match):
    with pytest.raises(ValidationError, match=match):
        EnvironmentSpec(name="g", **kw)


def test_graspable_names_must_be_unique_and_collidable():
    with pytest.raises(ValidationError, match="unique"):
        _desc(
            environment=({"name": "h", "type": "box", "size": (0.1, 0.1, 0.1)},),
            graspable=("h", "h"),
        )
    with pytest.raises(ValidationError, match="not collidable"):
        _desc(
            environment=(
                {"name": "h", "type": "box", "size": (0.1, 0.1, 0.1), "collidable": False},
            ),
            graspable=("h",),
        )


def test_graspable_must_resolve_to_a_world_geom_at_build():
    desc = _desc(
        environment=({"name": "handle", "type": "box", "size": (0.1, 0.1, 0.1)},),
        graspable=("handle", "a0_link7"),  # an arm body is not a world geom
    )
    with pytest.raises(SceneCompileError, match="a0_link7"):
        build_scene(desc)
    ok = _desc(
        environment=({"name": "handle", "type": "box", "size": (0.1, 0.1, 0.1)},),
        graspable=("handle",),
    )
    assert scene_meta(ok).graspable == ("handle",)
    assert build_scene(ok).meta.graspable == ("handle",)


def test_missing_texture_or_mesh_asset_fails_before_compile():
    with pytest.raises(SceneCompileError, match="not a vendored asset"):
        build_scene(
            _desc(
                environment=(
                    {"name": "p", "type": "box", "size": (0.001, 0.1, 0.1),
                     "texture": "textures/no_such_tag.png"},
                ),
            )
        )
    with pytest.raises(SceneCompileError, match="not a vendored asset"):
        build_scene(_desc(environment=({"name": "m", "type": "mesh", "mesh": "rail/nope.stl"},)))


# -- builder: textured box, mesh geom, collidable / group -------------------------------


@pytest.fixture(scope="module")
def textured():
    desc = SceneDescriptor(
        id="_textured",
        description="two tag plates sharing one PNG, one collidable box with a group",
        arms=(ARM,),
        environment=(
            {"name": "plate_a", "type": "box", "size": (0.0005, 0.1, 0.1), "pos": (0, 0, 1),
             "texture": TAG_PNG, "collidable": False},
            {"name": "plate_b", "type": "box", "size": (0.0005, 0.1, 0.1), "pos": (0, 0.5, 1),
             "texture": TAG_PNG, "collidable": False, "group": 2},
            {"name": "crate", "type": "box", "size": (0.1, 0.1, 0.1), "pos": (0.5, 0, 0.1),
             "group": 3},
            {"name": "plain", "type": "box", "size": (0.1, 0.1, 0.1), "pos": (-0.5, 0, 0.1)},
        ),
    )
    return build_scene(desc)


def test_textured_box_gets_one_file_texture_and_material_per_png(textured):
    model = textured.model
    assert model.ntex == 1  # both plates share the PNG -> one texture, one material
    tex_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, i) for i in range(model.ntex)
    ]
    assert tex_names == ["tex_textures_tagStandard41h12_00000_png"]
    assert model.tex_type[0] == mujoco.mjtTexture.mjTEXTURE_2D
    assert model.tex_width[0] == 704 and model.tex_height[0] == 704
    a, b = model.geom("plate_a"), model.geom("plate_b")
    assert a.matid[0] == b.matid[0] >= 0
    mat = model.mat(int(a.matid[0]))
    assert mat.name == "tex_textures_tagStandard41h12_00000_png_mat"
    assert mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGB] == 0
    assert mat.texuniform[0] == 0
    assert model.geom("plain").matid[0] == -1  # untextured world geoms keep plain rgba


def test_textured_xml_uses_file_textures_and_round_trips(textured):
    xml = textured.xml
    assert 'texturedir="' in xml and 'file="textures/tagStandard41h12_00000.png"' in xml
    assert "<texture" in xml and 'type="2d"' in xml
    model2 = mujoco.MjModel.from_xml_string(xml)  # re-compiles anywhere (episode replay)
    assert model2.ntex == 1 and model2.ngeom == textured.model.ngeom
    assert model2.geom("plate_a").matid[0] >= 0


def test_non_collidable_geoms_are_visual_only(textured):
    model = textured.model
    a, b = model.geom("plate_a"), model.geom("plate_b")
    assert a.contype[0] == 0 and a.conaffinity[0] == 0 and a.group[0] == 1  # default group 1
    assert b.contype[0] == 0 and b.conaffinity[0] == 0 and b.group[0] == 2  # explicit group
    crate, plain = model.geom("crate"), model.geom("plain")
    assert crate.contype[0] == 1 and crate.conaffinity[0] == 1 and crate.group[0] == 3
    assert plain.contype[0] == 1 and plain.conaffinity[0] == 1 and plain.group[0] == 0
    env = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
        for g in textured.addressing.env_geom_ids
    }
    assert {"crate", "plain", "floor"} <= env and not {"plate_a", "plate_b"} & env


def test_non_collidable_geoms_are_not_inflated_by_the_twin(textured):
    from apollo_mavis_v2_sim import DigitalTwin
    from apollo_mavis_v2_sim.twin import geom_labels

    twin = DigitalTwin(textured, inflation_m=0.025)
    model = twin.model
    assert model.geom_gap[model.geom("plate_a").id] == 0.0
    assert model.geom_gap[model.geom("crate").id] == pytest.approx(0.0125)
    labels = geom_labels(model)
    monitored = {labels[g] for pair in twin.monitored_pairs for g in pair}
    assert "crate" in monitored and "plate_a" not in monitored and "plate_b" not in monitored


def test_mesh_environment_geom_builds_with_scale_and_convex_hull():
    desc = _desc(
        environment=(
            {"name": "carriage", "type": "mesh", "mesh": "rail/linear_motor_platform.stl",
             "scale": (2.0, 2.0, 2.0), "pos": (0.0, -1.0, 0.5)},
            {"name": "carriage_small", "type": "mesh", "mesh": "rail/linear_motor_platform.stl",
             "pos": (0.0, -1.5, 0.5), "collidable": False},
        ),
        graspable=("carriage",),
    )
    built = build_scene(desc)
    model = built.model
    g = model.geom("carriage")
    assert g.type[0] == mujoco.mjtGeom.mjGEOM_MESH and g.contype[0] == 1
    assert model.nmesh == built.model.nmesh >= 2  # the arm's meshes + two scaled copies of ours
    big = model.mesh(int(g.dataid[0]))
    small = model.mesh(int(model.geom("carriage_small").dataid[0]))
    assert big.id != small.id  # (file, scale) pairs get their own mesh asset
    np.testing.assert_allclose(model.mesh_scale[big.id], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(model.mesh_scale[small.id], [1.0, 1.0, 1.0])
    # collider = the mesh's convex hull (MuJoCo's default for mesh geoms)
    assert model.mesh_graphadr[big.id] >= 0
    assert int(g.id) in set(built.addressing.env_geom_ids.tolist())
    assert built.meta.graspable == ("carriage",)
    mujoco.MjModel.from_xml_string(built.xml)  # round trip with meshdir


def test_scene_meta_carries_graspable(textured):
    assert textured.meta.graspable == ()
