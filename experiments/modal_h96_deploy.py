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

    os.makedirs("/results/weights", exist_ok=True)

    # Symlink the output dir so checkpoints save directly to the volume
    # This means EVERY checkpoint is persisted immediately — no signal loss
    os.environ["POSTFILTER_OUTPUT_DIR"] = "/results/weights"

    # Extract compressed video from archive.zip on the volume
    archive_path = "/results/archive.zip"
    if not os.path.exists(archive_path):
        print("ERROR: archive.zip not found on volume!")
        print("Upload with: .venv/bin/modal volume put comma-lab-weights submissions/robust_current/archive.zip archive.zip")
        return 1

    os.makedirs("/tmp/archive", exist_ok=True)
    subprocess.run(["unzip", "-o", archive_path, "-d", "/tmp/archive"], check=True)
    compressed_mkv = "/tmp/archive/0.mkv"
    print(f"Compressed video: {compressed_mkv} ({os.path.getsize(compressed_mkv)} bytes)")

    result = subprocess.run(
        ["python", "/root/cloud_h96_trainer.py",
         "--hidden", "96", "--epochs", "2500", "--alpha", "20",
         "--compressed-video", compressed_mkv],
        cwd="/tmp",
    )

    # Also copy any files the trainer put elsewhere
    for pattern in ["/tmp/**/*.pt", "/tmp/**/*.json"]:
        for f in glob.glob(pattern, recursive=True):
            dest = f"/results/weights/{os.path.basename(f)}"
            if not os.path.exists(dest):
                shutil.copy2(f, dest)

    # Commit volume to persist everything
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
