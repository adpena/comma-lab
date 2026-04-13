#!/usr/bin/env python3
"""Kaggle kernel bootstrap for asymmetric warp renderer training.

This file is intentionally minimal — all Kaggle logic lives in
tac.deploy.kaggle.runner (installed from the dataset wheel below).

Bootstrap sequence:
  1. Find tac wheel in /kaggle/input (stdlib only — tac not yet importable)
  2. pip install tac wheel
  3. Delegate everything to tac.deploy.kaggle.runner.main()
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _find_wheel(input_root: Path) -> Path:
    """Locate the tac wheel using stdlib only (tac not yet importable)."""
    for pattern in ("tac-*.whl", "comma_video_lab_ball_pack-*.whl"):
        candidates = sorted(input_root.rglob(pattern))
        if candidates:
            return candidates[-1]
    raise ImportError(
        f"tac wheel not found in {input_root}.\n"
        f"  Upload with: kaggle datasets version -p dist/ -m 'tac vX.Y.Z'\n"
        f"  to the comma-lab-private-assets dataset."
    )


def _bootstrap_tac(input_root: Path) -> None:
    """Install tac wheel if not already importable."""
    try:
        import tac  # noqa: F401
        from tac.deploy.kaggle import runner  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass
    wheel = _find_wheel(input_root)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel)]
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    _input_root = Path("/kaggle/input")
    _bootstrap_tac(_input_root)
    from tac.deploy.kaggle.runner import main  # noqa: E402
    raise SystemExit(main(launcher_path=Path(__file__)))
