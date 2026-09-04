"""Inflation-semantics CI sentinel (11-safety §6.2 / phase-03 acceptance).

Pins the MuJoCo behaviours L1 depends on: ``margin=0, gap=δ`` contacts are
detection-only (``efc_address == -1``, zero constraint force) and the pair
threshold sums both geoms' gaps. Exercises the actual ``apply_inflation``
used by the twin, on primitives AND on a composed scene. Any failure here
means upstream drift: re-audit the twin before unpinning ``mujoco==3.12.0``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from apollo_mavis_v2_sim import REGISTRY
from apollo_mavis_v2_sim.twin import apply_inflation

_TWO_SPHERES = """
<mujoco>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="slide" axis="1 0 0"/>
      <geom name="ga" type="sphere" size="0.05"/>
    </body>
    <body name="b" pos="{x2} 0 1">
      <joint name="jb" type="slide" axis="1 0 0"/>
      <geom name="gb" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _inflated_spheres(x2: float, delta: float):
    model = mujoco.MjModel.from_xml_string(_TWO_SPHERES.format(x2=x2))
    apply_inflation(model, delta)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_apply_inflation_sets_gap_half_margin_zero():
    model = mujoco.MjModel.from_xml_string(_TWO_SPHERES.format(x2=0.5))
    apply_inflation(model, 0.008)
    assert np.allclose(model.geom_gap, 0.004)  # δ/2 per geom
    assert np.allclose(model.geom_margin, 0.0)


def test_detection_only_contact_with_zero_force():
    # surface dist 0.03 < δ = 0.04 -> detected, but margin 0 -> no efc rows
    model, data = _inflated_spheres(x2=0.13, delta=0.04)
    assert data.ncon == 1
    c = data.contact[0]
    assert c.dist == pytest.approx(0.03, abs=1e-9)
    assert c.efc_address == -1
    assert np.allclose(data.qfrc_constraint, 0.0)


def test_pair_threshold_sums_both_gaps():
    # δ = 0.04 total -> per-geom 0.02: no contact at 0.045, contact at 0.035
    model, data = _inflated_spheres(x2=0.145, delta=0.04)
    assert data.ncon == 0
    model, data = _inflated_spheres(x2=0.135, delta=0.04)
    assert data.ncon == 1
    assert data.contact[0].dist == pytest.approx(0.035, abs=1e-9)


def test_scene_inflation_is_dynamics_neutral():
    """Inflating a composed scene adds no constraint force at home."""
    scene = REGISTRY.build("single_rail")
    apply_inflation(scene.model, 0.025)
    data = mujoco.MjData(scene.model)
    mujoco.mj_resetDataKeyframe(scene.model, data, 0)
    mujoco.mj_forward(scene.model, data)
    ncon = data.ncon
    active = [i for i in range(ncon) if data.contact.efc_address[i] >= 0]
    # every detected contact must be detection-only (excluded from efc)
    assert active == []
    collidable = (scene.model.geom_contype != 0) | (scene.model.geom_conaffinity != 0)
    assert np.allclose(scene.model.geom_gap[collidable], 0.0125)
    assert not np.any(scene.model.geom_gap[~collidable])
