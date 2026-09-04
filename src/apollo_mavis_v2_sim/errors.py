"""Sim-package exceptions (design 03-sim §12), rooted in core's hierarchy."""

from __future__ import annotations

from apollo_mavis_v2_core import ApolloError


class SceneError(ApolloError):
    """Base class for scene-registry / composition failures."""


class SceneNotFoundError(SceneError, KeyError):
    """Unknown scene id."""


class SceneArmMismatchError(SceneError):
    """Requested arm ids do not match the scene descriptor's arm ids."""


class SceneCompileError(SceneError):
    """``MjSpec.compile()`` failed (wraps ``mujoco.FatalError``/``ValueError``)."""


class TwinAuditError(ApolloError):
    """Unexplained at-home contacts under inflation: refuse to arm the gate."""


class IKUnreachableError(ApolloError):
    """``solve_to_convergence`` failed from every restart seed.

    Carries the best (non-converged) :class:`~apollo_mavis_v2_core.IKResult`.
    """

    def __init__(self, best_result) -> None:
        self.best_result = best_result
        super().__init__(
            "IK did not converge from any restart seed "
            f"(best pos_err={best_result.pos_err_m:.4f} m, "
            f"rot_err={best_result.rot_err_rad:.4f} rad)"
        )


__all__ = [
    "SceneError",
    "SceneNotFoundError",
    "SceneArmMismatchError",
    "SceneCompileError",
    "TwinAuditError",
    "IKUnreachableError",
]
