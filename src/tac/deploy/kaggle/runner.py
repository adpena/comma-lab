"""Kaggle environment setup and training launch for the asymmetric warp renderer.

All Kaggle-specific logic lives here so it can be tested and reviewed independently
from the bootstrap stub (kaggle_asym_warp_launcher.py), which only installs this
package before delegating to main().

Provider contract (mirrors Lightning/Modal patterns):
  - Strip Modal-specific flags (--raft-flow-path, --pose-targets-path, --max-hours)
    from build_flags() output.
  - Re-inject Kaggle-local paths + overrides.
  - Fail hard with actionable errors if required assets are missing.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Glob patterns to search for the tac wheel, in priority order.
#: The package was renamed from comma_video_lab_ball_pack → tac at v1.0.0.
WHEEL_GLOBS: tuple[str, ...] = ("tac-*.whl", "comma_video_lab_ball_pack-*.whl")

#: Kaggle dataset slug that hosts the tac wheel + supervision assets.
ASSET_DATASET_SLUG: str = "comma-lab-private-assets"

#: Upstream scorer repo (public, cloned at runtime).
UPSTREAM_REPO: str = "https://github.com/commaai/comma_video_compression_challenge.git"

#: Python deps not included in the tac wheel.
PIP_DEPS: tuple[str, ...] = (
    "av",
    "safetensors",
    "timm",
    "einops",
    "segmentation-models-pytorch",
    "pydantic",
    "click",
)

#: Flags whose values come from Modal-volume paths (/results/...) and must be
#: stripped before re-injecting Kaggle-local equivalents.
#: NOTE: all entries here take a value argument — never add boolean flags.
_STRIP_FLAGS: frozenset[str] = frozenset({
    "--raft-flow-path",
    "--pose-targets-path",
    "--max-hours",
    "--device",
})

#: Kaggle GPU session budget (T4 sessions last up to ~9h; leave 30 min margin).
KAGGLE_MAX_HOURS: float = 8.5

#: Kaggle default device.
KAGGLE_DEVICE: str = "cuda"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def find_tac_wheel(input_root: Path) -> Path:
    """Locate the tac wheel in the Kaggle dataset mount.

    Tries each pattern in WHEEL_GLOBS in order and returns the highest-sorted
    (latest version) match.

    Raises:
        ImportError: with instructions if no wheel is found.
    """
    for pattern in WHEEL_GLOBS:
        candidates = sorted(input_root.rglob(pattern))
        if candidates:
            return candidates[-1]
    raise ImportError(
        f"tac wheel not found in {input_root}.\n"
        f"  Expected: {input_root}/{ASSET_DATASET_SLUG}/tac-*.whl\n"
        f"  Upload with: kaggle datasets version -p dist/ -m 'tac vX.Y.Z'\n"
        f"  Searched patterns: {WHEEL_GLOBS}"
    )


def ensure_tac(input_root: Path | None = None) -> None:
    """Install tac from the dataset wheel if not already importable."""
    try:
        import tac  # noqa: F401
        from tac.deploy.kaggle import runner  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass
    if input_root is None:
        input_root = Path("/kaggle/input")
    wheel = find_tac_wheel(input_root)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel)]
    )
    # Verify the installed wheel has the deploy subpackages (added in tac v1.0.0).
    # Old wheels (comma_video_lab_ball_pack < v1.0.0) install tac but lack
    # tac.deploy.kaggle.runner, producing a cryptic ImportError later.
    try:
        from tac.deploy.kaggle import runner as _r  # noqa: F401,F811
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            f"Installed {wheel.name} but tac.deploy.kaggle.runner is not available.\n"
            f"  This wheel is pre-v1.0.0 and lacks the deploy subpackages.\n"
            f"  Upload tac-1.0.0+ to the comma-lab-private-assets dataset:\n"
            f"    kaggle datasets version -p dist/ -m 'tac v1.0.0'\n"
            f"  (Found wheel at: {wheel})"
        )


def ensure_upstream(working_dir: Path | None = None) -> Path:
    """Clone the upstream scorer repo if not already present.

    Returns:
        Path to the cloned upstream directory.
    """
    if working_dir is None:
        working_dir = Path("/kaggle/working")
    upstream = working_dir / "upstream"
    if not (upstream / "models").exists():
        subprocess.check_call([
            "git", "clone", "--depth", "1", UPSTREAM_REPO, str(upstream),
        ])
        subprocess.check_call(["git", "lfs", "pull"], cwd=upstream)
    return upstream


def ensure_deps() -> None:
    """Install missing Python dependencies via pip."""
    missing = []
    for dep in PIP_DEPS:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing.append(dep)
    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing
        )


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------

def resolve_training_script(launcher_path: Path) -> Path:
    """Find train_renderer_fridrich.py relative to the launcher.

    write_bundle() preserves repo-relative paths, so the script lands at
    experiments/train_renderer_fridrich.py inside the bundle directory.
    We probe both the flat layout (legacy) and the repo-relative layout.

    Raises:
        FileNotFoundError: with both probed paths if neither exists.
    """
    flat = launcher_path.parent / "train_renderer_fridrich.py"
    nested = launcher_path.parent / "experiments" / "train_renderer_fridrich.py"
    if flat.exists():
        return flat
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"Training script not found. Tried:\n"
        f"  {flat}\n"
        f"  {nested}\n"
        f"Ensure the kernel bundle includes experiments/train_renderer_fridrich.py."
    )


def resolve_supervision_assets(
    variant: str,
    asset_root: Path,
) -> dict[str, Path]:
    """Resolve and validate supervision asset paths for a given variant.

    Returns a dict with keys 'raft_flow' and/or 'posenet_targets' as needed.

    Raises:
        FileNotFoundError: with upload instructions if a required asset is missing.
    """
    assets: dict[str, Path] = {}

    if variant in ("supervised", "raft_only"):
        raft_flow = asset_root / "raft_flow.pt"
        if not raft_flow.exists():
            raise FileNotFoundError(
                f"raft_flow.pt not found at {raft_flow}\n"
                f"  Download from Modal volume:\n"
                f"    modal volume get comma-lab-results /results/raft_flow.pt ./\n"
                f"  Then add to the dataset:\n"
                f"    kaggle datasets version -p <dir-with-raft_flow.pt> -m 'add raft_flow.pt'\n"
                f"  Or use variant='base' to train without RAFT flow supervision."
            )
        assets["raft_flow"] = raft_flow

    if variant == "supervised":
        targets = asset_root / "posenet_targets.bin"
        if not targets.exists():
            raise FileNotFoundError(
                f"posenet_targets.bin not found at {targets}\n"
                f"  Download from Modal volume:\n"
                f"    modal volume get comma-lab-results /results/posenet_targets.bin ./\n"
                f"  Then add to the dataset:\n"
                f"    kaggle datasets version -p <dir> -m 'add posenet_targets.bin'\n"
                f"  Or use variant='raft_only' for RAFT-only supervision."
            )
        assets["posenet_targets"] = targets

    return assets


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def _strip_flags(flags: list[str], to_strip: frozenset[str]) -> list[str]:
    """Remove flag+value pairs for all flags in to_strip.

    Assumes every flag in to_strip takes exactly one value argument (no boolean flags).
    """
    clean: list[str] = []
    skip_next = False
    for f in flags:
        if skip_next:
            skip_next = False
            continue
        if f in to_strip:
            skip_next = True
            continue
        clean.append(f)
    return clean


def build_kaggle_command(
    variant: str,
    script_path: Path,
    asset_root: Path,
    resume_from: str | None = None,
) -> list[str]:
    """Build the full training command for Kaggle.

    Imports canonical flags from deploy_config, strips Modal-specific paths,
    and injects Kaggle-local equivalents.

    Args:
        variant: one of ALL_VARIANTS ('base', 'supervised', 'raft_only')
        script_path: absolute path to train_renderer_fridrich.py in the bundle
        asset_root: path to the comma-lab-private-assets dataset mount
        resume_from: optional checkpoint path on the Kaggle filesystem

    Returns:
        Full argv list starting with sys.executable.
    """
    from tac.deploy.deploy_config import ALL_VARIANTS, build_flags

    if variant not in ALL_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Valid: {list(ALL_VARIANTS)}"
        )

    # Do NOT pass provider_script_path — build_flags would prepend ["python", path]
    # which we'd then have to strip out. Instead, get bare flags and build the
    # full command ourselves so the argv is unambiguous.
    base_flags = build_flags(
        variant=variant,
        resume_from=resume_from,
    )
    clean_flags = _strip_flags(base_flags, _STRIP_FLAGS)

    # Kaggle-specific overrides
    overrides: list[str] = [
        "--device", KAGGLE_DEVICE,
        "--max-hours", str(KAGGLE_MAX_HOURS),
    ]

    assets = resolve_supervision_assets(variant, asset_root)
    if "raft_flow" in assets:
        overrides += ["--raft-flow-path", str(assets["raft_flow"])]
    if "posenet_targets" in assets:
        overrides += ["--pose-targets-path", str(assets["posenet_targets"])]

    return [sys.executable, str(script_path)] + clean_flags + overrides


# ---------------------------------------------------------------------------
# Environment wiring
# ---------------------------------------------------------------------------

def set_training_env(upstream: Path) -> None:
    """Set environment variables required by train_renderer_fridrich.py."""
    os.environ["UPSTREAM_ROOT"] = str(upstream)
    os.environ["TAC_UPSTREAM_DIR"] = str(upstream)
    os.environ["TAC_MODELS_DIR"] = str(upstream / "models")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{upstream}:{existing}" if existing else str(upstream)


def save_manifest(
    variant: str,
    cmd: list[str],
    working_dir: Path | None = None,
    script_path: Path | None = None,
) -> Path:
    """Write a deployment manifest JSON for reproducibility."""
    if working_dir is None:
        working_dir = Path("/kaggle/working")
    manifest = {
        "variant": variant,
        "full_command": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "provider": "kaggle",
        "gpu": "T4",
        "script": str(script_path) if script_path else None,
        "wheel_globs": list(WHEEL_GLOBS),
    }
    path = working_dir / f"deployment_manifest_{variant}.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(
    variant: str | None = None,
    input_root: Path | None = None,
    working_dir: Path | None = None,
    launcher_path: Path | None = None,
) -> int:
    """Full Kaggle asymmetric warp training launch sequence.

    Args:
        variant: override ASYM_VARIANT env var (default: reads from env, falls back to 'base')
        input_root: Kaggle dataset mount root (default: /kaggle/input)
        working_dir: Kaggle working directory (default: /kaggle/working)
        launcher_path: path to the calling launcher script (for script resolution)
    """
    if input_root is None:
        input_root = Path("/kaggle/input")
    if working_dir is None:
        working_dir = Path("/kaggle/working")
    if launcher_path is None:
        # Caller should pass __file__; fall back to working_dir
        launcher_path = working_dir / "kaggle_asym_warp_launcher.py"

    print("=== Kaggle asymmetric warp runner ===")

    ensure_deps()
    ensure_tac(input_root)
    upstream = ensure_upstream(working_dir)

    asset_root = input_root / ASSET_DATASET_SLUG
    script_path = resolve_training_script(Path(launcher_path))

    if variant is None:
        variant = os.environ.get("ASYM_VARIANT", "base").strip()

    print(f"  Variant: {variant}")
    print(f"  Script:  {script_path}")
    print(f"  Assets:  {asset_root}")

    cmd = build_kaggle_command(variant, script_path, asset_root)
    set_training_env(upstream)
    manifest_path = save_manifest(variant, cmd, working_dir, script_path)

    print(f"  Manifest: {manifest_path}")
    print(f"  Command:  {' '.join(cmd)}")
    print("  ---")

    result = subprocess.run(cmd)
    return result.returncode
