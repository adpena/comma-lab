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

# Re-export canonical bootstrap from cloud_bootstrap.
# Import at module level so the bound references survive test mocks on __import__.
from tac.deploy.cloud_bootstrap import (  # noqa: E402,F401
    WHEEL_GLOBS as WHEEL_GLOBS,
    find_wheel as find_tac_wheel,
    bootstrap as _cb_bootstrap,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

def ensure_tac(input_root: Path | None = None) -> None:
    """Ensure tac is installed with the deploy subpackages (added in v1.0.0).

    Delegates to :func:`tac.deploy.cloud_bootstrap.bootstrap` with
    ``verify_submodule="tac.deploy.kaggle.runner"``.  Idempotent — safe to
    call at the start of any Kaggle runner.
    """
    if input_root is None:
        input_root = Path(os.environ.get("CLOUD_INPUT_ROOT", "/kaggle/input"))
    _cb_bootstrap(input_root, verify_submodule="tac.deploy.kaggle.runner")


def ensure_upstream(working_dir: Path | None = None) -> Path:
    """Clone the upstream scorer repo if not already present.

    git lfs pull is retried up to 3 times — it intermittently times out or
    rate-limits on Kaggle's network, which would otherwise fail the whole run.

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
        # Retry git lfs pull up to 3 times — network failures are common on Kaggle
        for attempt in range(1, 4):
            try:
                subprocess.check_call(["git", "lfs", "pull"], cwd=upstream)
                break
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise RuntimeError(
                        "git lfs pull failed after 3 attempts.\n"
                        "  Check upstream repo LFS quota and Kaggle network access.\n"
                        f"  Upstream: {UPSTREAM_REPO}"
                    )
                print(f"  git lfs pull attempt {attempt} failed — retrying...")
                import time as _time
                _time.sleep(5 * attempt)
    return upstream


def ensure_deps() -> None:
    """Install missing Python dependencies — uv preferred, pip fallback."""
    from tac.deploy.cloud_paths import try_ensure_uv, ensure_packages
    ensure_packages(try_ensure_uv(), *PIP_DEPS)


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

    Reads configuration from environment variables:
        ASYM_VARIANT   — variant to train ('base', 'supervised', 'raft_only')
        RESUME_FROM    — optional path to a .pt checkpoint inside the dataset mount,
                         e.g. /kaggle/input/comma-lab-private-assets/renderer_best_v3.pt

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

    # Kaggle mount path varies across platform versions
    asset_root = input_root / ASSET_DATASET_SLUG
    if not asset_root.exists():
        asset_root = input_root / "datasets" / "adpena" / ASSET_DATASET_SLUG
    script_path = resolve_training_script(Path(launcher_path))

    if variant is None:
        variant = os.environ.get("ASYM_VARIANT", "base").strip()

    resume_from: str | None = os.environ.get("RESUME_FROM") or None
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            print(
                f"  WARNING: RESUME_FROM checkpoint not found: {resume_path}\n"
                f"  The launcher preamble sets RESUME_FROM unconditionally; if the\n"
                f"  checkpoint was not uploaded to the dataset, training starts from scratch.\n"
                f"  Upload {resume_path.name} to {input_root / ASSET_DATASET_SLUG} to resume."
            )
            resume_from = None

    print(f"  Variant:     {variant}")
    print(f"  Script:      {script_path}")
    print(f"  Assets:      {asset_root}")
    print(f"  Resume from: {resume_from or '(none — training from scratch)'}")

    cmd = build_kaggle_command(variant, script_path, asset_root, resume_from=resume_from)
    set_training_env(upstream)
    manifest_path = save_manifest(variant, cmd, working_dir, script_path)

    print(f"  Manifest: {manifest_path}")
    print(f"  Command:  {' '.join(cmd)}")
    print("  ---")

    result = subprocess.run(cmd)
    return result.returncode
