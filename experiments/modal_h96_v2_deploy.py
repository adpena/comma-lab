"""Deploy h=96 training to Modal A10G with hardened tac library.

Usage:
    .venv/bin/modal run experiments/modal_h96_v2_deploy.py

Uses the full tac library (hardened eval, atomic saves, pydantic config).
Results persist to Modal Volume — download with:
    .venv/bin/modal volume get comma-lab-weights weights/ ./modal_weights/
"""
import modal
from pathlib import Path

REPO = Path(__file__).parent.parent

app = modal.App("comma-lab-h96-v2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "av", "safetensors", "timm", "einops",
        "segmentation-models-pytorch", "numpy", "pydantic",
    )
    .run_commands("apt-get update && apt-get install -y git git-lfs")
    # Upload tac library
    .add_local_dir(str(REPO / "src" / "tac"), "/root/src/tac")
    # Upload training script
    .add_local_file(str(REPO / "experiments" / "train_tac.py"), "/root/train_tac.py")
)

vol = modal.Volume.from_name("comma-lab-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 8,  # 8 hours
    volumes={"/results": vol},
)
def train_h96():
    import subprocess
    import os
    import sys

    print("=== Modal h=96 v2 (hardened tac library) ===")
    gpu = os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read().strip()
    print(f"GPU: {gpu}")

    # Setup paths
    os.makedirs("/results/weights", exist_ok=True)
    os.makedirs("/root/workspace", exist_ok=True)

    # Clone upstream for scorer models + video
    print("Cloning upstream repo...")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/commaai/comma_video_compression_challenge.git",
        "/root/workspace/upstream/comma_video_compression_challenge"
    ], check=True)
    subprocess.run(["git", "lfs", "pull"],
                   cwd="/root/workspace/upstream/comma_video_compression_challenge", check=True)

    # Create the submission archive by encoding locally
    # Actually, we need the archive.zip — upload it
    print("ERROR: Need to upload archive.zip to Modal volume first")
    print("Run: modal volume put comma-lab-weights submissions/robust_current/archive.zip archive.zip")

    # Check if archive exists on volume
    archive_path = "/results/archive.zip"
    if not os.path.exists(archive_path):
        print(f"Archive not found at {archive_path}")
        print("Upload it first with: .venv/bin/modal volume put comma-lab-weights submissions/robust_current/archive.zip archive.zip")
        return 1

    # Ensure tac is importable
    sys.path.insert(0, "/root/src/..")

    # Run training
    result = subprocess.run([
        sys.executable, "/root/train_tac.py",
        "--hidden", "96",
        "--epochs", "2500",
        "--alpha", "20",
        "--sal-lambda", "1.0",
        "--subsample", "4",
        "--tag", "h96_modal_v2",
        "--output-dir", "/results/weights",
    ], env={
        **os.environ,
        "PYTHONPATH": "/root/src/..:/root/workspace/upstream/comma_video_compression_challenge",
        "PYTHONUNBUFFERED": "1",
    })

    vol.commit()
    print(f"Training exit code: {result.returncode}")
    return result.returncode


@app.local_entrypoint()
def main():
    # First upload the archive to the volume
    print("Uploading archive.zip to Modal volume...")
    import subprocess as sp
    archive = REPO / "submissions" / "robust_current" / "archive.zip"
    sp.run([
        ".venv/bin/modal", "volume", "put", "comma-lab-weights",
        str(archive), "archive.zip"
    ])

    print("Deploying h=96 v2 training to Modal A10G...")
    result = train_h96.remote()
    print(f"Training completed with exit code: {result}")
    print("Download with: .venv/bin/modal volume get comma-lab-weights weights/ ./modal_weights/")
