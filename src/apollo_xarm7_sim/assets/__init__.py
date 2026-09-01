"""Package-data access for vendored MJCF/mesh assets (design 03-sim §2).

``asset_path`` is the canonical accessor; nothing else in the package
hardcodes asset locations.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_ASSETS = resources.files("apollo_xarm7_sim") / "assets"


def asset_path(*parts: str) -> Path:
    """Absolute path to a vendored asset, e.g. ``asset_path("xarm7_on_rail.xml")``.

    Raises :class:`FileNotFoundError` when the asset does not exist so a
    typo'd path fails at resolution time instead of deep inside MuJoCo.
    """
    entry = _ASSETS.joinpath(*parts)
    path = Path(str(entry))
    if not path.exists():
        raise FileNotFoundError(f"no such asset: {'/'.join(parts)} (under {_ASSETS})")
    return path


__all__ = ["asset_path"]
