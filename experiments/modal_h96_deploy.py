"""Deploy h=96 training to Modal A10G GPU with persistent results.

Usage:
    .venv/bin/python -m modal run experiments/modal_h96_deploy.py

Results are saved to Modal Volume 'comma-lab-weights' and can be
downloaded after training completes.
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
    timeout=3600 * 6,  # 6 hours
    volumes={"/results": vol},
)
def train_h96():
    import subprocess
    import os
    import shutil
    import glob

    print("Starting h=96 training on Modal A10G...")
    gpu = os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read().strip()
    print(f"GPU: {gpu}")

    result = subprocess.run(
        ["python", "/root/cloud_h96_trainer.py",
         "--hidden", "96", "--epochs", "2500", "--alpha", "20"],
        cwd="/tmp",
    )

    # Copy ALL weight files to the persistent volume
    print("Copying results to persistent volume...")
    os.makedirs("/results/weights", exist_ok=True)

    # Find all .pt and .json checkpoint files the trainer produced
    for pattern in ["/tmp/**/*.pt", "/tmp/**/*.json", "/tmp/**/postfilter_*"]:
        for f in glob.glob(pattern, recursive=True):
            dest = f"/results/weights/{os.path.basename(f)}"
            shutil.copy2(f, dest)
            print(f"  Saved: {dest} ({os.path.getsize(dest)} bytes)")

    # Also save the training log
    for log in glob.glob("/tmp/**/*.log", recursive=True):
        dest = f"/results/{os.path.basename(log)}"
        shutil.copy2(log, dest)
        print(f"  Log: {dest}")

    vol.commit()
    print(f"Training exit code: {result.returncode}")
    print("Results committed to Modal Volume 'comma-lab-weights'")
    return result.returncode


@app.local_entrypoint()
def main():
    print("Deploying h=96 training to Modal A10G...")
    result = train_h96.remote()
    print(f"Training completed with exit code: {result}")
    print("Download results with:")
    print("  modal volume get comma-lab-weights /results/weights/ ./modal_weights/")
