"""Deploy h=96 training to Modal A10G GPU.

Installs tac as a package, uses precomputed data volume, runs training
via the canonical CLI (same as local).

Usage:
    .venv/bin/modal run src/tac/deploy/modal/modal_h96_deploy.py
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "comma-lab-h96"
REPO_ROOT = Path(__file__).resolve().parents[4]  # src/tac/deploy/modal -> repo root
PRECOMPUTED_VOL = "tac-precomputed"
RESULTS_VOL = "comma-lab-weights"

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
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(REPO_ROOT / "src"), "/root/src")
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
def train_h96(tag: str = "h96_council_modal"):
    """Run h=96 training via the canonical tac CLI."""
    import os
    import subprocess
    import sys

    os.makedirs(f"/results/{tag}", exist_ok=True)

    precomputed = "/data/precomputed"
    has_precomputed = os.path.exists(f"{precomputed}/comp_frames.pt")

    print(f"=== tac lossy training: h96_council | tag: {tag} ===")
    print(f"  GPU: CUDA ({os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')})")
    print(f"  Precomputed: {'YES' if has_precomputed else 'NO (will decode video)'}")

    cmd = [
        sys.executable, "-m", "tac", "lossy",
        "--profile", "h96_council",
        "--tag", tag,
        "--output-dir", f"/results/{tag}",
        "--hidden", "96",
        "--epochs", "2500",
        "--alpha", "20",
    ]
    if has_precomputed:
        cmd.extend(["--precomputed", precomputed])

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/root/src"})

    results_vol.commit()

    print(f"\n=== h96 complete (exit {result.returncode}) ===")
    return {"tag": tag, "exit_code": result.returncode}


@app.local_entrypoint()
def main():
    # Pre-flight cost estimate
    from tac.cost_tracker import print_cost_estimate
    print("\n--- Cost Estimate ---")
    print_cost_estimate(gpu="a10g", estimated_hours=6.0, platform="modal")
    print()

    print("Deploying h=96 training to Modal A10G...")
    result = train_h96.remote()
    print(f"Training completed: {result}")
    print("Download results with:")
    print("  modal volume get comma-lab-weights /results/ ./modal_weights/")
