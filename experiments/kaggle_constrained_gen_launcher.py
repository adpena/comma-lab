#!/usr/bin/env python3
"""Kaggle kernel bootstrap for constrained generation inflate experiment.

Pre-registered hypothesis: Constrained optimization from noise scores < 0.80 proxy.
Kill: proxy > 1.50 after 500 steps.

Bootstrap sequence:
  1. Find tac wheel from dataset mount (stdlib only — tac not yet importable)
  2. Install wheel via uv (preferred) or pip fallback
  3. Delegate to tac.deploy.kaggle.runner_constrained.main()
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Self-contained uv/pip bootstrap (runs before tac is importable)
# ---------------------------------------------------------------------------

def _try_ensure_uv() -> str | None:
    """Locate uv binary or bootstrap it. Returns path or None (never raises)."""
    existing = shutil.which("uv")
    if existing:
        return existing

    candidates = [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
        Path("/opt/conda/bin/uv"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    if os.environ.get("CLOUD_SKIP_UV_INSTALL", "").lower() in ("1", "true", "yes"):
        return None

    print("  [launcher] uv not found — bootstrapping ...")
    result = subprocess.run(
        ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("  [launcher] uv bootstrap failed — falling back to pip")
        return None

    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which("uv")


def _install_wheel(wheel: Path) -> None:
    """Install wheel via uv (preferred) or pip fallback."""
    uv = _try_ensure_uv()
    if uv is not None:
        subprocess.check_call(
            [uv, "pip", "install", "--system", "-q", "--no-deps", str(wheel)]
        )
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel)]
        )


# ---------------------------------------------------------------------------
# tac wheel discovery + install
# ---------------------------------------------------------------------------

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
        from tac.constrained_gen import constrained_generate  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass
    wheel = _find_wheel(input_root)
    _install_wheel(wheel)
    try:
        from tac.constrained_gen import constrained_generate  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            f"Installed {wheel.name} but tac.constrained_gen is not importable.\n"
            f"  This wheel may be missing the constrained_gen module.\n"
            f"  Rebuild and upload: uv build && kaggle datasets version -p dist/ -m 'tac vX.Y.Z'"
        ) from exc


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    _input_root = Path("/kaggle/input")
    _bootstrap_tac(_input_root)
    from tac.deploy.kaggle.runner_constrained import main  # noqa: E402
    raise SystemExit(main(launcher_path=Path(__file__)))
