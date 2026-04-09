"""Deploy h=96 training to Modal A10G GPU.

Usage:
    .venv/bin/python -m modal run experiments/modal_h96_deploy.py
"""
import modal
from pathlib import Path

app = modal.App("comma-lab-h96")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "av", "safetensors", "timm", "einops",
        "segmentation-models-pytorch", "numpy",
    )
    .run_commands("apt-get update && apt-get install -y git git-lfs")
    .add_local_file("experiments/cloud_h96_trainer.py", "/root/cloud_h96_trainer.py")
)

vol = modal.Volume.from_name("comma-lab-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 6,
    volumes={"/results": vol},
)
def train_h96():
    import subprocess
    import os

    print("Starting h=96 training on Modal A10G...")
    print(f"GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()}")

    # Set env so the script saves to the persistent volume
    import os
    os.environ["PERSIST_DIR"] = "/results"

    result = subprocess.run(
        ["python", "/root/cloud_h96_trainer.py",
         "--hidden", "96", "--epochs", "2500", "--alpha", "20"],
        cwd="/tmp",
        env={**os.environ},
    )
    vol.commit()
    return result.returncode


@app.local_entrypoint()
def main():
    print("Deploying h=96 training to Modal A10G...")
    result = train_h96.remote()
    print(f"Training completed with exit code: {result}")
