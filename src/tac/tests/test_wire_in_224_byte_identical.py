# SPDX-License-Identifier: MIT
"""Regression guard for the #224 CONSOLIDATED wire-in default-off byte-identical bar.

Runs tools/wire_in_224_byte_identical_smoke.py (the deterministic same-process forward-
equivalence proof that the render compose hooks are no-ops at their defaults and the new
witness accessors are additive) and asserts it PASSES. MLX-only; skipped where MLX is
unavailable (e.g. a non-Mac CI runner)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mlx_available(), reason="MLX not available (non-Mac runner)")
def test_wire_in_224_default_off_byte_identical_smoke():
    smoke = REPO / "tools" / "wire_in_224_byte_identical_smoke.py"
    assert smoke.exists(), smoke
    env = dict(os.environ)
    env["PYTHONPATH"] = f"src:upstream:{REPO}"
    # the smoke's render hooks are GPU-free; drop the GPU-only custom kernel flag so it runs on CPU.
    env.pop("TAC_MLX_CUSTOM_GROUPED_BACKWARD", None)
    r = subprocess.run(
        [sys.executable, str(smoke)], cwd=str(REPO), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"smoke failed rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "RESULT: PASS" in r.stdout, r.stdout
