"""Sim-package exceptions (design 03-sim §12), rooted in core's hierarchy."""

from __future__ import annotations

from apollo_xarm7_core import ApolloError


class SceneError(ApolloError):
    """Base class for scene-registry / composition failures."""


class SceneNotFoundError(SceneError, KeyError):
    """Unknown scene id."""


class SceneArmMismatchError(SceneError):
    """Requested arm ids do not match the scene descriptor's arm ids."""


class SceneCompileError(SceneError):
    """``MjSpec.compile()`` failed (wraps ``mujoco.FatalError``/``ValueError``)."""


__all__ = [
    "SceneError",
    "SceneNotFoundError",
    "SceneArmMismatchError",
    "SceneCompileError",
]
