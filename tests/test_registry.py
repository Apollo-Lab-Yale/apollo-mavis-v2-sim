"""Scene registry listing: ``hidden`` dev scenes and display ``title`` (03-sim §4).

The UI/API see exactly one scene (the lab cell); the eight dev/CI scenes and the
hidden kitchen twin stay in the package -- built by id -- but are filtered
from ``REGISTRY.list()``.
"""

from __future__ import annotations

import pytest

from apollo_mavis_v2_sim import REGISTRY

KITCHEN = "mavis_v2_kitchen"  # hidden: selected explicitly in config, never listed
DEV_SCENES = {
    "single_rail",
    "single_fixed_tabletop",
    "dual_rail_tabletop",
    "dual_mixed",
    "triple_rail_row",
    "guardrail_env",
    "guardrail_face",
    "guardrail_rail",
}


def test_list_hides_dev_scenes_by_default():
    rows = REGISTRY.list()
    assert [m.id for m in rows] == ["mavis_v2"]
    (m,) = rows
    assert m.title == "APOLLO MAVIS V2 Digital Twin"
    assert m.hidden is False
    assert m.microphones == {"view": False, "grip": False}  # YAML default: mic off


def test_list_include_hidden_returns_every_descriptor():
    rows = REGISTRY.list(include_hidden=True)
    assert {m.id for m in rows} == DEV_SCENES | {"mavis_v2", KITCHEN}
    assert all(m.hidden for m in rows if m.id in DEV_SCENES)
    assert all(m.title is None for m in rows if m.id in DEV_SCENES)
    assert [m.id for m in rows if not m.hidden] == ["mavis_v2"]
    # the kitchen twin (03-sim §4.4): hidden like a dev scene, titled like the lab cell
    (kitchen,) = [m for m in rows if m.id == KITCHEN]
    assert kitchen.hidden is True and kitchen.title == "APOLLO MAVIS V2 Kitchen"


@pytest.mark.parametrize("scene_id", sorted(DEV_SCENES))
def test_hidden_scenes_stay_addressable_by_id(scene_id):
    desc = REGISTRY.descriptor(scene_id)
    assert desc.hidden is True and desc.title is None
    meta = REGISTRY.meta(scene_id)
    assert meta.id == scene_id and meta.hidden is True
    assert REGISTRY.build(scene_id).model.nq > 0  # unfiltered build


def test_title_defaults_to_none_and_is_carried_by_meta():
    from apollo_mavis_v2_sim import SceneDescriptor
    from apollo_mavis_v2_sim.scenes import scene_meta

    base = SceneDescriptor(id="_t", description="d", arms=({"id": "a0", "model": "xarm7_fixed"},))
    assert base.title is None and base.hidden is False
    assert scene_meta(base).title is None and scene_meta(base).hidden is False
    titled = base.model_copy(update={"title": "T", "hidden": True})
    assert scene_meta(titled).title == "T" and scene_meta(titled).hidden is True
