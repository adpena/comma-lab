"""Deploy h=96 training to Modal A10G with hardened tac library.

Uses the portable train_tac.py + full tac library. All paths explicit.

Usage:
    .venv/bin/modal run experiments/modal_h96_v2_deploy.py
    .venv/bin/modal volume get comma-lab-weights weights/ ./modal_weights/
"""
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent

app = modal.App("comma-lab-h96-v2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "av", "safetensors", "timm", "einops",
        "segmentation-models-pytorch", "numpy", "pydantic",
    )
    .run_commands(
        "apt-get update && apt-get install -y git git-lfs unzip",
        # Clone upstream for scorer models + GT video
        "git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git /upstream",
        "cd /upstream && git lfs pull",
    )
    # Upload tac library + training script
    .add_local_dir(str(REPO / "src" / "tac"), "/app/src/tac")
    .add_local_file(str(REPO / "experiments" / "train_tac.py"), "/app/train_tac.py")
)

vol = modal.Volume.from_name("comma-lab-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 8,
    volumes={"/vol": vol},
)
def train_h96():
    import os
    import subprocess
    import sys

    print("=== Modal h=96 v2 (hardened tac library) ===")
    gpu = os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read().strip()
    print(f"GPU: {gpu}")

    os.makedirs("/vol/weights", exist_ok=True)

    # Extract compressed video from archive on volume
    archive_path = "/vol/archive.zip"
    if not os.path.exists(archive_path):
        print("ERROR: archive.zip not on volume. Upload first:")
        print("  .venv/bin/modal volume put comma-lab-weights submissions/robust_current/archive.zip archive.zip")
        return 1

    # Compute saliency on the fly (train_tac.py needs it)
    # For now, generate a uniform saliency map if not available
    saliency_path = "/vol/saliency.npy"
    if not os.path.exists(saliency_path):
        print("Generating uniform saliency (no pre-computed map on volume)...")
        import numpy as np
        sal = np.ones((30, 874, 1164), dtype=np.float32)
        np.save(saliency_path, sal)

    result = subprocess.run(
        [
            sys.executable, "/app/train_tac.py",
            "--tag", "h96_modal_v2",
            "--hidden", "96",
            "--epochs", "2500",
            "--alpha", "20",
            "--sal-lambda", "1.0",
            "--subsample", "4",
            "--output-dir", "/vol/weights",
            "--archive", archive_path,
            "--gt-video", "/upstream/videos/0.mkv",
            "--saliency", saliency_path,
            "--models-dir", "/upstream/models",
            "--upstream-dir", "/upstream",
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/app/src:/upstream",
            "PYTHONUNBUFFERED": "1",
        },
    )

    vol.commit()
    print(f"Exit code: {result.returncode}")
    return result.returncode


@app.local_entrypoint()
def main():
    # Upload archive + saliency to volume
    import subprocess as sp

    archive = REPO / "submissions" / "robust_current" / "archive.zip"
    saliency = REPO / "experiments" / "masks" / "posenet_saliency.npy"

    print("Uploading data to Modal volume...")
    sp.run([".venv/bin/modal", "volume", "put", "comma-lab-weights", str(archive), "archive.zip"])
    if saliency.exists():
        sp.run([".venv/bin/modal", "volume", "put", "comma-lab-weights", str(saliency), "saliency.npy"])

    print("Deploying h=96 v2 to Modal A10G...")
    result = train_h96.remote()
    print(f"Training completed: exit code {result}")
    print("Download: .venv/bin/modal volume get comma-lab-weights weights/ ./modal_weights/")
