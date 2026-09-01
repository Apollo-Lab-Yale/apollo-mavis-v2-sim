"""xArm gripper mapping: open fraction <-> MJCF ctrl <-> meters (03-sim §6).

The menagerie gripper actuator has ``ctrlrange="0 255"`` with equilibrium
driver angle ``q_drv = ctrl * 0.85 / 255`` rad, where 0.85 rad is FULLY
CLOSED. Hence **ctrl 0 = open, ctrl 255 = closed** — the direction is
asserted by ``test_gripper_mapping`` via fingertip-gap measurement, never
trusted from memory. The real gripper spans 0-850 pulses over a 0.085 m
opening (850 = open); hardware maps the same open fraction to pulses
(``f * 850``), so both workcells agree at the ``GripperCommand`` boundary.
"""

from __future__ import annotations

GRIPPER_CTRL_MAX = 255.0  # menagerie actuator ctrlrange upper bound (= closed)
DRIVER_CLOSED_RAD = 0.85  # driver joint angle at fully closed
GRIPPER_SPAN_M = 0.085  # full mechanical opening in meters


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def open_frac_to_ctrl(open_frac: float) -> float:
    """Open fraction (1 = fully open) -> actuator ctrl (0 = open, 255 = closed)."""
    return (1.0 - _clamp01(open_frac)) * GRIPPER_CTRL_MAX


def ctrl_to_open_frac(ctrl: float) -> float:
    """Actuator ctrl (0-255) -> open fraction (inverse of :func:`open_frac_to_ctrl`)."""
    return 1.0 - min(GRIPPER_CTRL_MAX, max(0.0, float(ctrl))) / GRIPPER_CTRL_MAX


def driver_q_to_open_frac(q_driver_rad: float) -> float:
    """Measured driver joint angle (rad) -> open fraction, clamped to [0, 1]."""
    return _clamp01(1.0 - float(q_driver_rad) / DRIVER_CLOSED_RAD)


def open_frac_to_meters(open_frac: float) -> float:
    """Open fraction -> fingertip opening in meters (0.085 m span)."""
    return _clamp01(open_frac) * GRIPPER_SPAN_M


__all__ = [
    "GRIPPER_CTRL_MAX",
    "DRIVER_CLOSED_RAD",
    "GRIPPER_SPAN_M",
    "open_frac_to_ctrl",
    "ctrl_to_open_frac",
    "driver_q_to_open_frac",
    "open_frac_to_meters",
]
