"""Shared fixtures. Sets MUJOCO_GL=egl for the whole test process.

The package itself never touches environment variables (03-sim §7); the
test harness owns its own process env, and it must be set BEFORE mujoco
initializes a GL context, hence the setdefault at conftest import time.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # noqa: E402 - must precede mujoco GL init

import pytest
from apollo_xarm7_core import ArmConfig, WorkcellConfig
from apollo_xarm7_core.schemas import PoseModel

from apollo_xarm7_sim import REGISTRY, BuiltScene


def make_config(scene_id: str, arm_ids: list[str]) -> WorkcellConfig:
    return WorkcellConfig(
        kind="sim",
        sim_scene=scene_id,
        arms=[ArmConfig(id=a, base_in_world=PoseModel()) for a in arm_ids],
    )


@pytest.fixture(scope="session")
def triple_scene() -> BuiltScene:
    return REGISTRY.build("triple_rail_row")


@pytest.fixture(scope="session")
def dual_scene() -> BuiltScene:
    return REGISTRY.build("dual_rail_tabletop")


@pytest.fixture(scope="session")
def single_fixed_scene() -> BuiltScene:
    return REGISTRY.build("single_fixed_tabletop")
