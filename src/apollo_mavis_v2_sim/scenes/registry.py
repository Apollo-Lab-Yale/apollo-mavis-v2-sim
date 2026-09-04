"""Scene registry over the YAML descriptors in ``assets/scenes`` (03-sim §4).

``REGISTRY`` is instantiated at import; descriptor scanning is lazy (first
access) so importing the package does no file I/O.
"""

from __future__ import annotations

import yaml
from pydantic import ValidationError

from ..assets import asset_path
from ..errors import SceneCompileError, SceneNotFoundError
from .builder import BuiltScene, build_scene, scene_meta
from .descriptor import SceneDescriptor, SceneMeta, SceneOverrides


class SceneRegistry:
    """id -> descriptor lookup with build-on-demand composition."""

    def __init__(self) -> None:
        self._descriptors: dict[str, SceneDescriptor] | None = None

    def _load(self) -> dict[str, SceneDescriptor]:
        if self._descriptors is None:
            descriptors: dict[str, SceneDescriptor] = {}
            scene_dir = asset_path("scenes")
            for path in sorted(scene_dir.glob("*.yaml")):
                try:
                    data = yaml.safe_load(path.read_text(encoding="utf-8"))
                    desc = SceneDescriptor.model_validate(data)
                except (yaml.YAMLError, ValidationError) as e:
                    raise SceneCompileError(f"bad scene descriptor {path.name}: {e}") from e
                if desc.id != path.stem:
                    raise SceneCompileError(
                        f"scene descriptor {path.name}: id {desc.id!r} != filename stem"
                    )
                descriptors[desc.id] = desc
            self._descriptors = descriptors
        return self._descriptors

    def list(self) -> list[SceneMeta]:
        return [scene_meta(d) for d in self._load().values()]

    def descriptor(self, scene_id: str) -> SceneDescriptor:
        try:
            return self._load()[scene_id]
        except KeyError:
            raise SceneNotFoundError(
                f"unknown scene {scene_id!r}; known: {sorted(self._load())}"
            ) from None

    def meta(self, scene_id: str) -> SceneMeta:
        return scene_meta(self.descriptor(scene_id))

    def build(self, scene_id: str, overrides: SceneOverrides | None = None) -> BuiltScene:
        return build_scene(self.descriptor(scene_id), overrides)


REGISTRY = SceneRegistry()

__all__ = ["SceneRegistry", "REGISTRY"]
