# SPDX-License-Identifier: MIT
"""Provider-agnostic cloud path resolution and uv bootstrap.

All experiment runners import from this module for platform-specific paths
and tooling setup. Platform auto-detection; env vars override.

Platform detection (first match wins):
    1. CLOUD_PLATFORM env var: "kaggle" | "modal" | "lightning" | "local"
    2. KAGGLE_KERNEL_RUN_TYPE env var present → kaggle
    3. MODAL_IS_CONTAINER=true → modal
    4. /teamspace directory exists → lightning
    5. fallback → local

Path resolution:
    Kaggle:    input=/kaggle/input         working=/kaggle/working
    Modal:     input=/vol/input            working=/vol/working      (customize via env)
    Lightning: input=/teamspace/input      working=/teamspace/working
    Local:     input=CWD/data/input        working=CWD/data/working

Override any path via:
    CLOUD_INPUT_ROOT   — override input_root for this run
    CLOUD_WORKING_DIR  — override working_dir for this run

Usage::

    from tac.deploy.cloud_paths import CloudPaths, ensure_uv, uv_pip_install

    paths = CloudPaths.detect()
    uv = ensure_uv()
    uv_pip_install(uv, "av", "safetensors", "timm")

    input_root = paths.input_root
    working_dir = paths.working_dir
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Platform constants
# ---------------------------------------------------------------------------

_KAGGLE_INPUT = Path("/kaggle/input")
_KAGGLE_WORKING = Path("/kaggle/working")

_MODAL_INPUT = Path("/vol/input")
_MODAL_WORKING = Path("/vol/working")

_LIGHTNING_ROOT = Path("/teamspace")
_LIGHTNING_INPUT = _LIGHTNING_ROOT / "input"
_LIGHTNING_WORKING = _LIGHTNING_ROOT / "working"


# ---------------------------------------------------------------------------
# CloudPaths dataclass
# ---------------------------------------------------------------------------

@dataclass
class CloudPaths:
    """Resolved platform-specific paths for a cloud training/eval run.

    Attributes:
        platform: detected platform name
        input_root: root directory where asset datasets are mounted/available
        working_dir: writable output directory for this run
    """
    platform: str
    input_root: Path
    working_dir: Path

    @classmethod
    def detect(cls) -> "CloudPaths":
        """Auto-detect platform and return appropriate path defaults.

        Respects CLOUD_PLATFORM, CLOUD_INPUT_ROOT, CLOUD_WORKING_DIR overrides.
        """
        platform = os.environ.get("CLOUD_PLATFORM", "").strip().lower()

        if not platform:
            if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
                platform = "kaggle"
            elif os.environ.get("MODAL_IS_CONTAINER", "").lower() in ("1", "true", "yes"):
                platform = "modal"
            elif _LIGHTNING_ROOT.exists() and (
                os.environ.get("LIGHTNING_STUDIO_ID") or (_LIGHTNING_ROOT / "studios").exists()
            ):
                platform = "lightning"
            else:
                platform = "local"

        if platform == "kaggle":
            default_input = _KAGGLE_INPUT
            default_working = _KAGGLE_WORKING
        elif platform == "modal":
            default_input = _MODAL_INPUT
            default_working = _MODAL_WORKING
        elif platform == "lightning":
            default_input = _LIGHTNING_INPUT
            default_working = _LIGHTNING_WORKING
        else:
            cwd = Path.cwd()
            default_input = cwd / "data" / "input"
            default_working = cwd / "data" / "working"

        input_root = Path(os.environ.get("CLOUD_INPUT_ROOT") or default_input)
        working_dir = Path(os.environ.get("CLOUD_WORKING_DIR") or default_working)
        working_dir.mkdir(parents=True, exist_ok=True)

        return cls(platform=platform, input_root=input_root, working_dir=working_dir)

    def asset_path(self, slug: str, rel: str) -> Path:
        """Resolve a path to an asset inside a dataset/volume.

        Args:
            slug: dataset slug (e.g. "comma-lab-private-assets")
            rel:  relative path inside the dataset (e.g. "0.mkv")
        """
        return self.input_root / slug / rel

    def __str__(self) -> str:
        return (
            f"CloudPaths(platform={self.platform!r}, "
            f"input_root={self.input_root}, working_dir={self.working_dir})"
        )


# ---------------------------------------------------------------------------
# uv bootstrap
# ---------------------------------------------------------------------------

def _uv_candidates() -> list[Path]:
    return [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
        Path("/opt/conda/bin/uv"),
    ]


def try_ensure_uv() -> str | None:
    """Return path to uv binary if available, else None (never raises).

    Preferred over ensure_uv() when a pip fallback is acceptable — the
    caller should use install_packages() which handles the None case.

    Attempts:
        1. Check PATH (shutil.which)
        2. Check common install locations
        3. Bootstrap via astral.sh installer (if CLOUD_SKIP_UV_INSTALL is not set)
    """
    # Honour explicit override first
    uv_override = os.environ.get("CLOUD_UV_PATH", "").strip()
    if uv_override and Path(uv_override).exists():
        return uv_override

    existing = shutil.which("uv")
    if existing:
        return existing

    for candidate in _uv_candidates():
        if candidate.exists():
            return str(candidate)

    if os.environ.get("CLOUD_SKIP_UV_INSTALL", "").lower() in ("1", "true", "yes"):
        print("  [cloud_paths] uv not found and CLOUD_SKIP_UV_INSTALL set — using pip fallback")
        return None

    print("  [cloud_paths] uv not found — bootstrapping via astral.sh ...")
    result = subprocess.run(
        ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [cloud_paths] uv bootstrap failed (stdout={result.stdout!r}) — using pip fallback")
        return None

    for candidate in _uv_candidates():
        if candidate.exists():
            print(f"  [cloud_paths] uv installed at {candidate}")
            return str(candidate)

    refreshed = shutil.which("uv")
    if refreshed:
        return refreshed

    print("  [cloud_paths] uv installed but binary not found on PATH — using pip fallback")
    return None


def ensure_uv() -> str:
    """Return path to the uv binary, installing uv if not present.

    Raises RuntimeError if uv cannot be found or bootstrapped. Use
    try_ensure_uv() instead when a pip fallback is acceptable.

    Returns:
        Absolute path to the uv binary.

    Raises:
        RuntimeError: if uv cannot be found or installed.
    """
    result = try_ensure_uv()
    if result is None:
        raise RuntimeError(
            "uv not available and could not be installed. "
            "Set CLOUD_SKIP_UV_INSTALL=1 to use pip fallback, "
            "or set CLOUD_UV_PATH to the uv binary path."
        )
    return result


def uv_pip_install(uv: str, *packages: str, no_deps: bool = False) -> None:
    """Install packages using uv pip install.

    Preferred over pip_install() when uv is available.

    Args:
        uv: path to the uv binary (from ensure_uv() / try_ensure_uv()).
        *packages: package names or wheel paths to install.
        no_deps: if True, pass --no-deps (use for pre-built wheels only).
    """
    if not packages:
        return
    cmd = [uv, "pip", "install", "--system", "-q"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)
    subprocess.check_call(cmd)


def pip_install(*packages: str, no_deps: bool = False) -> None:
    """Install packages using the current interpreter's pip.

    Fallback for environments where uv is not available. Prefer
    uv_pip_install() when uv is present.

    Args:
        *packages: package names or wheel paths to install.
        no_deps: if True, pass --no-deps (use for pre-built wheels only).
    """
    if not packages:
        return
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)
    subprocess.check_call(cmd)


def install_packages(
    *packages: str, uv: str | None = None, no_deps: bool = False
) -> None:
    """Install packages using uv if available, else pip.

    This is the preferred install entrypoint for cloud runners that must
    work across environments where uv may not be present (Kaggle, Modal,
    Lightning, local).

    Args:
        *packages: package names or wheel paths to install.
        uv: uv binary path from try_ensure_uv(). None → use pip fallback.
        no_deps: if True, pass --no-deps.
    """
    if not packages:
        return
    if uv is not None:
        uv_pip_install(uv, *packages, no_deps=no_deps)
    else:
        pip_install(*packages, no_deps=no_deps)


def _is_importable(pkg: str) -> bool:
    """Check if a package is importable (without installing it)."""
    try:
        __import__(pkg.replace("-", "_"))
        return True
    except ImportError:
        return False


def ensure_packages(uv: str | None, *packages: str) -> None:
    """Install any packages not yet importable, using uv or pip fallback.

    Args:
        uv: uv binary path from try_ensure_uv(). None → pip fallback.
        *packages: package names (pypi format).
    """
    missing = [p for p in packages if not _is_importable(p)]
    if missing:
        install_packages(*missing, uv=uv)
