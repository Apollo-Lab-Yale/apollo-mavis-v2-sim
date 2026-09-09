"""Import guard: the sim package pulls in neither dora nor pyarrow (phase-12).

The external bus lives behind the runtime's ``dora_bridge``; sim publishes plain
``CameraFrame``s (optionally with a depth sibling) and never touches the codec.
Mirrors core's test_import_guard; runs in a subprocess so this process's own
imports cannot mask a leak. Importing the package needs no GL context.
"""

from __future__ import annotations

import json
import subprocess
import sys

FORBIDDEN = {"dora", "pyarrow", "xarm", "fastapi", "torch", "lerobot", "cv2"}

_CHILD = """
import json
import sys
import apollo_mavis_v2_sim  # noqa: F401
import apollo_mavis_v2_sim.rendering  # noqa: F401
import apollo_mavis_v2_sim.cameras  # noqa: F401
print(json.dumps(sorted(sys.modules)))
"""


def test_sim_import_pulls_no_dora_or_pyarrow() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    top_level = {m.split(".")[0] for m in json.loads(proc.stdout)}
    leaked = top_level & FORBIDDEN
    assert not leaked, f"sim imported banned modules: {sorted(leaked)}"
