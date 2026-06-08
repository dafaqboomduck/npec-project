"""Make the repository's `scripts/` directory importable so test modules
can `import inventory_dataset` etc. without any path hacks at the test
module top.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
