"""Guardrail CI regression as a pytest (11-safety §5.1) — this IS the safety CI.

Runs the full matrix (4 scenarios x {gate-only main, IK-on main, IK-on
graze}) through the script's own ``main`` so the CI entry point and the test
exercise identical code. Budget: < 30 s virtual-tick wall time.
"""

from __future__ import annotations

import time

from apollo_xarm7_sim.tools.guardrail_check import SCENARIOS, main


def test_scenario_table_is_complete():
    assert set(SCENARIOS) == {
        "env_table_descend",
        "env_pedestal_sweep",
        "cross_arm_head_on",
        "cross_arm_rail_converge",
    }
    for s in SCENARIOS.values():
        assert s.twist.shape == (6,)
        assert s.graze_twist is not None and s.graze_twist.shape == (6,)
        assert len(s.target_pair_prefixes) == 2


def test_all_scenarios_pass_within_budget(capsys):
    t0 = time.monotonic()
    rc = main(["--all"])
    wall = time.monotonic() - t0
    out = capsys.readouterr().out
    assert rc == 0, f"guardrail failed:\n{out}"
    assert wall < 30.0, f"guardrail took {wall:.1f} s (budget 30 s)"
    assert out.count("PASS") >= 12  # 4 scenarios x 3 runs + summary
