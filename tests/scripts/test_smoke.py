"""Smoke tests for the scripts/ folder.

For each Python script we either:
1. Run it with `--help` (if it uses argparse/typer) to confirm the parser
   wires up cleanly, or
2. Compile it with the `compile()` builtin to confirm it's at least
   syntactically valid.

Scripts that import Azure SDKs or other live external services at import
time are exercised via compile() rather than execution, since they would
otherwise try to contact a real workspace.
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_SCRIPT_FILES = sorted(p for p in _SCRIPTS.glob("*.py") if p.name != "__init__.py")

# Scripts whose --help should succeed quickly without hitting external services.
_HELP_SAFE = {
    "inventory_dataset.py",
    "dataset_deep_check.py",
}


@pytest.mark.parametrize("script", _SCRIPT_FILES, ids=lambda p: p.name)
def test_script_compiles(script: Path) -> None:
    """Every script must parse — catches typos/import-time syntax errors."""
    py_compile.compile(str(script), doraise=True)


@pytest.mark.parametrize(
    "script",
    [p for p in _SCRIPT_FILES if p.name in _HELP_SAFE],
    ids=lambda p: p.name,
)
def test_script_help(script: Path) -> None:
    """Pure-local scripts: --help exits 0 and prints usage."""
    out = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "usage" in (out.stdout + out.stderr).lower()
