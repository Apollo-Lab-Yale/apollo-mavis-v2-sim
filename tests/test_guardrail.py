"""Guardrail CI regression as a pytest (11-safety §5.1) — this IS the safety CI.

Runs the full matrix (6 scenarios x {gate-only main, IK-on main, IK-on
graze}) through the script's own ``main`` so the CI entry point and the test
exercise identical code. Budget: < 30 s virtual-tick wall time.
"""

from __future__ import annotations

import time

from apollo_mavis_v2_sim.tools.guardrail_check import MAVIS_SWEEP_START_Q, SCENARIOS, main


def test_scenario_table_is_complete():
    assert set(SCENARIOS) == {
        "env_table_descend",
        "env_pedestal_sweep",
        "cross_arm_head_on",
        "cross_arm_rail_converge",
        "mavis_v2_rail_sweep",
        "mavis_v2_rail_sweep_mic",
    }
    for s in SCENARIOS.values():
        assert s.twist.shape == (6,)
        assert s.graze_twist is not None and s.graze_twist.shape == (6,)
        assert len(s.target_pair_prefixes) == 2
    # the mic variant is the deployment twin's geometry: mic on the Perception Arm only
    mic = SCENARIOS["mavis_v2_rail_sweep_mic"]
    assert mic.overrides is not None and mic.overrides.microphones == {"view": True}
    assert SCENARIOS["mavis_v2_rail_sweep"].overrides is None
    # Both mavis scenarios start from their own lowered ready pose (the pre-2026-09-04
    # keyframe), not from the scene keyframe: the cell's initial state folds both arms
    # at the xArm zero with the rails at opposite ends, where a +X sweep hits nothing.
    for sid in ("mavis_v2_rail_sweep", "mavis_v2_rail_sweep_mic"):
        start = SCENARIOS[sid].start_q
        assert start is not None and start.keys() == {"grip", "view"}
        assert start is MAVIS_SWEEP_START_Q
        for q in start.values():
            assert q.shape == (8,) and 0.0 <= q[7] <= 0.65  # core order, rail LAST
    for sid in ("env_table_descend", "env_pedestal_sweep", "cross_arm_head_on",
                "cross_arm_rail_converge"):
        assert SCENARIOS[sid].start_q is None  # those cells still start at the keyframe


def test_all_scenarios_pass_within_budget(capsys):
    t0 = time.monotonic()
    rc = main(["--all"])
    wall = time.monotonic() - t0
    out = capsys.readouterr().out
    assert rc == 0, f"guardrail failed:\n{out}"
    assert wall < 30.0, f"guardrail took {wall:.1f} s (budget 30 s)"
    assert out.count("PASS") >= 18  # 6 scenarios x 3 runs + summary
