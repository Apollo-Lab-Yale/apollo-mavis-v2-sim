"""mujoco==3.12.0 pin keeper (03-sim §13).

margin/gap semantics have changed across MuJoCo versions and attach keyframe
re-indexing only recently became reliable. Any failure here = upstream
drift; re-audit twin inflation (§8) and the builder (§5) before unpinning.
"""

from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest

from apollo_mavis_v2_sim.assets import asset_path

_TWO_SPHERES = """
<mujoco>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="slide" axis="1 0 0"/>
      <geom name="ga" type="sphere" size="0.05" margin="{m1}" gap="{g1}"/>
    </body>
    <body name="b" pos="{x2} 0 1">
      <joint name="jb" type="slide" axis="1 0 0"/>
      <geom name="gb" type="sphere" size="0.05" margin="{m2}" gap="{g2}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _spheres(x2: float, m1=0.0, g1=0.0, m2=0.0, g2=0.0):
    xml = _TWO_SPHERES.format(x2=x2, m1=m1, g1=g1, m2=m2, g2=g2)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_gap_is_detection_only():
    # surface dist 0.03; gap 0.02 EACH (pair threshold 0.04 > 0.03), margin 0
    model, data = _spheres(x2=0.13, g1=0.02, g2=0.02)
    assert data.ncon == 1
    c = data.contact[0]
    assert c.dist == pytest.approx(0.03, abs=1e-9)
    assert c.efc_address == -1  # detection-only: no constraint rows
    assert np.allclose(data.qfrc_constraint, 0.0)  # zero dynamics effect


def test_pair_threshold_sums_both():
    # gap on ONE geom only -> threshold 0.02: no contact at 0.03 separation...
    model, data = _spheres(x2=0.13, g1=0.02)
    assert data.ncon == 0
    # ...but detected at 0.015
    model, data = _spheres(x2=0.115, g1=0.02)
    assert data.ncon == 1
    assert data.contact[0].dist == pytest.approx(0.015, abs=1e-9)


def test_geom_distance_signed():
    model, data = _spheres(x2=0.13)
    fromto = np.zeros(6)
    d = mujoco.mj_geomDistance(model, data, 0, 1, 0.2, fromto)
    assert d == pytest.approx(0.03, abs=1e-9)
    # negative when penetrating
    model, data = _spheres(x2=0.09)
    assert mujoco.mj_geomDistance(model, data, 0, 1, 0.2, None) == pytest.approx(-0.01, abs=1e-9)
    # returns distmax when nothing within distmax
    model, data = _spheres(x2=0.13)
    assert mujoco.mj_geomDistance(model, data, 0, 1, 0.02, None) == 0.02


def test_margin_generates_forces():
    # margin (not gap) -> the contact HAS efc rows: NOT pure inflation
    model, data = _spheres(x2=0.13, m1=0.02, m2=0.02)
    assert data.ncon == 1
    assert data.contact[0].efc_address >= 0


def test_attach_keyframes_and_names():
    xarm = str(asset_path("ufactory_xarm7", "xarm7.xml"))
    spec = mujoco.MjSpec()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict")
        for i in range(3):
            child = mujoco.MjSpec.from_file(xarm)
            frame = spec.worldbody.add_frame(pos=[0.0, 0.9 * i, 0.0])
            spec.attach(child, prefix=f"arm{i}_", frame=frame)
    model = spec.compile()
    # all child assets copy per attach, referenced or not: 3 x 16 meshes
    assert model.nmesh == 48
    assert model.nkey == 3
    for i in range(3):
        assert model.key(f"arm{i}_home").qpos.shape == (model.nq,)
        assert model.actuator(f"arm{i}_act1").id >= 0
        assert model.actuator(f"arm{i}_gripper").id >= 0
    # to_xml round-trips and recompiles to the same sizes
    spec.meshdir = str(asset_path("ufactory_xarm7", "assets"))
    model2 = mujoco.MjSpec.from_string(spec.to_xml()).compile()
    for attr in ("nq", "nu", "nkey", "ngeom", "nmesh", "nbody"):
        assert getattr(model2, attr) == getattr(model, attr), attr
