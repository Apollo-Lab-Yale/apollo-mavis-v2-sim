"""Regenerate ``assets/ASSET_MANIFEST.json`` (sha256 per vendored asset).

Recorded episodes persist the composed scene XML plus this manifest's
hashes instead of copying STLs — replay verifies the asset set matches
(design 03-sim §5). Run after any asset change:

    uv run python -m apollo_xarm7_sim.tools.gen_asset_manifest
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from ..assets import asset_path

MANIFEST_NAME = "ASSET_MANIFEST.json"
_EXCLUDE = {MANIFEST_NAME, "__init__.py", "__pycache__"}


def compute_manifest() -> dict[str, str]:
    """Relative asset path -> sha256, sorted, for everything under assets/."""
    root = asset_path()
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in _EXCLUDE for part in rel.split("/")):
            continue
        entries[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def write_manifest() -> Path:
    root = asset_path()
    out = Path(str(root)) / MANIFEST_NAME
    payload = json.dumps(compute_manifest(), indent=2, sort_keys=True) + "\n"
    out.write_text(payload, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    out = write_manifest()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
