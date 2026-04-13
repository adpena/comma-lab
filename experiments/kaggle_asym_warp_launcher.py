#!/usr/bin/env python3
"""Kaggle kernel launcher for asymmetric warp renderer training.

Reads ASYM_VARIANT env var (base | supervised | raft_only) and delegates
to train_renderer_fridrich.py with the canonical flags from deploy_config.
This keeps Kaggle in strict parity with Modal and Lightning.

Dataset requirements (from comma-lab-private-assets):
  - posenet_targets.bin  (required for supervised variant)
  - raft_flow.pt         (required for supervised / raft_only variants)
  - archive.zip          (training archive)
  - upstream scorer (cloned from GitHub or from dataset)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()

# Flags that the deploy_config VARIANT_FLAGS inject with /results/ paths,
# which must be replaced with Kaggle-local asset paths.
_SUPERVISION_ASSET_FLAGS = {"--raft-flow-path", "--pose-targets-path"}

# Flags whose values we override per-provider; strip from BASE_FLAGS before
# appending our Kaggle-specific values.
# NOTE: _strip_flags assumes all flags in these sets take a following value argument.
# Never add boolean (valueless) flags here — they would incorrectly skip the next token.
_KAGGLE_OVERRIDE_FLAGS = {"--device", "--max-hours"}


def ensure_tac() -> None:
    """Install tac wheel via pip (Kaggle standard — uv not pre-installed/supported)."""
    try:
        import tac  # noqa: F401
        from tac.deploy import deploy_config  # noqa: F401 — need deploy subpackage
        return
    except (ImportError, ModuleNotFoundError):
        pass
    input_root = Path("/kaggle/input")
    candidates = sorted(input_root.rglob("comma_video_lab_ball_pack-*.whl"))
    if not candidates:
        raise ImportError(
            f"tac wheel not found in {input_root}; upload comma_video_lab_ball_pack-*.whl "
            f"to the comma-lab-private-assets dataset"
        )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(candidates[0])])


def ensure_upstream() -> Path:
    upstream = Path("/kaggle/working/upstream")
    if not (upstream / "models").exists():
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/commaai/comma_video_compression_challenge.git",
            str(upstream),
        ])
        subprocess.check_call(["git", "lfs", "pull"], cwd=upstream)
    return upstream


def ensure_deps() -> None:
    """Install missing deps via pip (Kaggle standard toolchain)."""
    deps = ["av", "safetensors", "timm", "einops", "segmentation-models-pytorch", "pydantic", "click"]
    missing = []
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing.append(dep)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)


def _strip_flags(flags: list[str], to_strip: set[str]) -> list[str]:
    """Remove flag+value pairs for all flags in to_strip."""
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


# ---------------------------------------------------------------------------
def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    print("=== Kaggle asymmetric warp launcher ===")

    ensure_deps()
    ensure_tac()
    upstream = ensure_upstream()

    # Locate asset files from the Kaggle dataset mount
    input_root = Path("/kaggle/input")
    asset_root = input_root / "comma-lab-private-assets"

    # Locate training script — bundled alongside this launcher
    script_path = SCRIPT_PATH.parent / "train_renderer_fridrich.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Variant from env var (set via bootstrap_preamble in kernel metadata)
    variant = os.environ.get("ASYM_VARIANT", "base").strip()

    # Import canonical flags from tac (installed via wheel above)
    from tac.deploy.deploy_config import ALL_VARIANTS, build_flags  # noqa: E402
    if variant not in ALL_VARIANTS:
        print(f"ERROR: Unknown ASYM_VARIANT={variant!r}. Valid: {ALL_VARIANTS}")
        return 1

    # Resolve asset paths on Kaggle (dataset mounts at /kaggle/input/)
    posenet_targets = asset_root / "posenet_targets.bin"
    raft_flow = asset_root / "raft_flow.pt"

    # Validate supervision assets before building the command
    if variant in ("supervised", "raft_only"):
        if not raft_flow.exists():
            print(f"ERROR: raft_flow.pt not found at {raft_flow}")
            return 1
    if variant == "supervised":
        if not posenet_targets.exists():
            print(f"ERROR: posenet_targets.bin not found at {posenet_targets}")
            return 1

    # Build flags from deploy_config, then strip provider-specific paths + overrides
    base_flags = build_flags(variant=variant)

    # Strip flags whose values must be replaced on Kaggle
    clean_flags = _strip_flags(base_flags, _SUPERVISION_ASSET_FLAGS | _KAGGLE_OVERRIDE_FLAGS)

    # Append Kaggle-specific overrides
    kaggle_overrides: list[str] = [
        "--device", "cuda",
        "--max-hours", "8.5",   # Kaggle T4: up to ~9h sessions
    ]
    if variant in ("supervised", "raft_only"):
        kaggle_overrides += ["--raft-flow-path", str(raft_flow)]
    if variant == "supervised":
        kaggle_overrides += ["--pose-targets-path", str(posenet_targets)]

    cmd = [sys.executable, str(script_path)] + clean_flags + kaggle_overrides

    # Set environment for the training subprocess
    os.environ["UPSTREAM_ROOT"] = str(upstream)       # used by train_renderer_fridrich.py
    os.environ["TAC_MODELS_DIR"] = str(upstream / "models")
    # PYTHONPATH: upstream for scorer imports; tac is already in site-packages
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{upstream}:{existing}" if existing else str(upstream)

    # Save deployment manifest
    import json
    import socket
    import time
    manifest = {
        "variant": variant,
        "full_command": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "provider": "kaggle",
        "gpu": "T4",
        "script": str(script_path),
    }
    manifest_path = Path("/kaggle/working") / f"deployment_manifest_{variant}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest saved: {manifest_path}")
    print(f"  Variant: {variant}")
    print(f"  Command: {' '.join(cmd)}")
    print("  ---")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
