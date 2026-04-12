"""Deploy the dilated h64 trainer to Modal A10G.

Installs tac as a package, uses precomputed data volume, runs training
via the canonical CLI (same as local).

Usage:
    .venv/bin/modal run src/tac/deploy/modal/modal_dilated_h64_deploy.py
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "comma-lab-dilated-h64"
REPO_ROOT = Path(__file__).resolve().parents[4]  # src/tac/deploy/modal -> repo root
PRECOMPUTED_VOL = "tac-precomputed"
RESULTS_VOL = "comma-lab-dilated-h64"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "ffmpeg")
    .pip_install(
        "torch==2.6.*",
        "torchvision",
        "av",
        "numpy",
        "pydantic>=2.0",
        "safetensors",
        "timm",
        "einops",
        "segmentation-models-pytorch",
    )
    .add_local_dir(str(REPO_ROOT / "src"), "/root/src")
    .env({"PYTHONPATH": "/root/src"})
)

precomputed_vol = modal.Volume.from_name(PRECOMPUTED_VOL, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 6,
    volumes={"/data": precomputed_vol, "/results": results_vol},
    memory=32768,
)
def train_dilated_h64(tag: str = "dilated_h64_long1000_modal"):
    """Run dilated h64 training via the canonical tac CLI."""
    import os
    import subprocess
    import sys

    os.makedirs(f"/results/{tag}", exist_ok=True)

    precomputed = "/data/precomputed"
    has_precomputed = os.path.exists(f"{precomputed}/comp_frames.pt")

    print(f"=== tac lossy training: proven_baseline | tag: {tag} ===")
    print(f"  GPU: CUDA ({os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')})")
    print(f"  Precomputed: {'YES' if has_precomputed else 'NO (will decode video)'}")

    cmd = [
        sys.executable, "-m", "tac", "lossy",
        "--profile", "proven_baseline",
        "--tag", tag,
        "--output-dir", f"/results/{tag}",
        "--hidden", "64",
        "--alpha", "20",
        "--epochs", "2500",
        "--eval-subsample", "1",
        "--checkpoint-eval-every", "10",
        "--checkpoint-select-int8",
        "--per-channel-int8",
    ]
    if has_precomputed:
        cmd.extend(["--precomputed", precomputed])

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/root/src"})

    results_vol.commit()

    print(f"\n=== dilated h64 complete (exit {result.returncode}) ===")
    return {"tag": tag, "exit_code": result.returncode}


@app.local_entrypoint()
def main():
    # Pre-flight cost estimate
    from tac.cost_tracker import print_cost_estimate
    print("\n--- Cost Estimate ---")
    print_cost_estimate(gpu="a10g", estimated_hours=6.0, platform="modal")
    print()

    result = train_dilated_h64.remote()
    print(f"Modal dilated h64 exit code: {result}")
