#!/usr/bin/env python3
"""Kaggle kernel bootstrap for asymmetric warp renderer training.

Intentionally minimal — all Kaggle logic lives in tac.deploy.kaggle.runner
(installed from the dataset wheel below).

Bootstrap sequence:
  1. Resolve input root (CLOUD_INPUT_ROOT env → /kaggle/input default)
  2. Find + install tac wheel via uv (preferred) or pip (fallback)  [stdlib only]
  3. Full post-install verification via tac.deploy.cloud_bootstrap
  4. Delegate to tac.deploy.kaggle.runner.main()
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Pre-tac bootstrap stub (stdlib only — tac not yet importable at this point)
# Mirrors tac.deploy.cloud_bootstrap.BOOTSTRAP_STUB; must stay in sync.
# ---------------------------------------------------------------------------

def _pre_install(input_root: Path) -> None:
    """Stage 1: minimal stdlib-only wheel install. Idempotent."""
    try:
        import tac  # noqa: F401
        return  # already installed
    except ImportError:
        pass

    wheel: Path | None = None
    for pat in ("tac-*.whl", "comma_video_lab_ball_pack-*.whl"):
        hits = sorted(input_root.rglob(pat))
        if hits:
            wheel = hits[-1]
            break
    if wheel is None:
        raise ImportError(
            f"tac wheel not found in {input_root}.\n"
            f"  Upload with: kaggle datasets version -p dist/ -m 'tac vX.Y.Z'\n"
            f"  to the comma-lab-private-assets dataset."
        )

    uv = shutil.which("uv") or next(
        (str(c) for c in (
            Path.home() / ".local" / "bin" / "uv",
            Path.home() / ".cargo" / "bin" / "uv",
            Path("/usr/local/bin/uv"),
            Path("/opt/conda/bin/uv"),
        ) if Path(c).exists()),
        None,
    )
    if uv:
        subprocess.check_call([uv, "pip", "install", "--system", "-q", "--no-deps", str(wheel)])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel)])


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # Provider-agnostic input root — CLOUD_INPUT_ROOT env override supported
    _input_root = Path(os.environ.get("CLOUD_INPUT_ROOT", "/kaggle/input"))

    # Stage 1: stdlib pre-install
    _pre_install(_input_root)

    # Stage 2: full verification via cloud_bootstrap (tac now importable)
    from tac.deploy.cloud_bootstrap import bootstrap as _cb_bootstrap
    _cb_bootstrap(_input_root, verify_submodule="tac.deploy.kaggle.runner")

    from tac.deploy.kaggle.runner import main  # noqa: E402
    raise SystemExit(main(launcher_path=Path(__file__)))
