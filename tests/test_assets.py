"""Vendored assets: presence, provenance, and manifest hashes (03-sim §2/§14)."""

from __future__ import annotations

import json

import pytest

from apollo_xarm7_sim.assets import asset_path
from apollo_xarm7_sim.tools.gen_asset_manifest import MANIFEST_NAME, compute_manifest

ARM_STLS = [
    "link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "end_tool", "base_link",
    "left_outer_knuckle", "left_finger", "left_inner_knuckle",
    "right_outer_knuckle", "right_finger", "right_inner_knuckle",
]


def test_menagerie_vendored_complete():
    for name in ("xarm7.xml", "xarm7_nohand.xml", "hand.xml", "LICENSE", "UPSTREAM"):
        assert asset_path("ufactory_xarm7", name).is_file()
    for stl in ARM_STLS:
        assert asset_path("ufactory_xarm7", "assets", f"{stl}.stl").is_file()


def test_rail_and_camera_assets_vendored():
    for name in ("linear_motor_rail.stl", "linear_motor_platform.stl", "LICENSE", "UPSTREAM"):
        assert asset_path("rail", name).is_file()
    for name in ("d435_with_cam_stand.stl", "LICENSE", "UPSTREAM"):
        assert asset_path("cameras", name).is_file()


def test_upstream_files_record_commit_hashes():
    for d in ("ufactory_xarm7", "rail", "cameras"):
        text = asset_path(d, "UPSTREAM").read_text(encoding="utf-8")
        assert "commit:" in text and "source:" in text and "license:" in text


def test_child_models_present():
    assert asset_path("xarm7_on_rail.xml").is_file()
    assert asset_path("xarm7_fixed.xml").is_file()


def test_asset_manifest_hashes_match():
    manifest = json.loads(asset_path(MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest == compute_manifest(), (
        "asset drift: regenerate with "
        "`uv run python -m apollo_xarm7_sim.tools.gen_asset_manifest`"
    )
    # every vendored binary is covered
    assert "rail/linear_motor_rail.stl" in manifest
    assert "ufactory_xarm7/assets/link_base.stl" in manifest
    assert "xarm7_on_rail.xml" in manifest


def test_asset_path_raises_on_missing():
    with pytest.raises(FileNotFoundError):
        asset_path("no_such_dir", "nope.stl")
