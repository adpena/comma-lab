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

# --- Bootstrap: install tac wheel from Kaggle dataset --
SCRIPT_PATH = Path(__file__).resolve()


def ensure_tac() -> None:
    try:
        import tac  # noqa: F401
        return
    except ImportError:
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
    for dep in ["av", "safetensors", "timm", "einops", "segmentation-models-pytorch", "pydantic", "click"]:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])


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
    working = Path("/kaggle/working")

    # Locate training script — bundled alongside this launcher
    script_path = SCRIPT_PATH.parent / "train_renderer_fridrich.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Variant from env var (set in kernel metadata or overridden at launch)
    variant = os.environ.get("ASYM_VARIANT", "base").strip()
    output_dir = working / f"renderer_{variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import canonical flags from tac (installed via wheel above)
    from tac.deploy.deploy_config import ALL_VARIANTS, build_flags  # noqa: E402
    if variant not in ALL_VARIANTS:
        print(f"ERROR: Unknown ASYM_VARIANT={variant!r}. Valid: {ALL_VARIANTS}")
        return 1

    # Resolve asset paths
    posenet_targets = asset_root / "posenet_targets.bin"
    raft_flow = asset_root / "raft_flow.pt"

    # Build command — use Kaggle-specific asset paths for supervision flags
    extra_overrides: list[str] = []
    if variant in ("supervised", "raft_only"):
        if not raft_flow.exists():
            print(f"ERROR: raft_flow.pt not found at {raft_flow}")
            return 1
        extra_overrides += ["--raft-flow-path", str(raft_flow)]
    if variant == "supervised":
        if not posenet_targets.exists():
            print(f"ERROR: posenet_targets.bin not found at {posenet_targets}")
            return 1
        extra_overrides += ["--pose-targets-path", str(posenet_targets)]

    # Build base flags from deploy_config (strips the /results/ asset paths,
    # which we override above with the Kaggle-local paths)
    flags = build_flags(variant=variant, extra=extra_overrides)
    # Remove any /results/ paths that don't exist on Kaggle
    clean_flags: list[str] = []
    skip_next = False
    for f in flags:
        if skip_next:
            skip_next = False
            continue
        if f in ("--raft-flow-path", "--pose-targets-path"):
            skip_next = True  # skip the /results/... value; we've added overrides above
            continue
        clean_flags.append(f)

    cmd = [sys.executable, str(script_path)] + clean_flags + [
        "--output-dir", str(output_dir),
        "--device", "cuda",
        "--max-hours", "8.5",  # Kaggle T4 sessions: up to ~9h
    ]

    os.environ["PYTHONPATH"] = f"{SCRIPT_PATH.parent.parent / 'src'}:{upstream}"
    os.environ["TAC_UPSTREAM_DIR"] = str(upstream)
    os.environ["TAC_MODELS_DIR"] = str(upstream / "models")

    # Save deployment manifest
    import json, time, socket
    manifest = {
        "variant": variant,
        "full_command": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "provider": "kaggle",
        "gpu": "T4",
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest saved: {manifest_path}")
    print(f"  Variant: {variant}")
    print(f"  Command: {' '.join(cmd)}")
    print("  ---")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
