"""Deploy the self-contained dilated h64 trainer to Modal.

Usage:
    uv run --with modal modal run experiments/modal_dilated_h64_deploy.py
"""
from __future__ import annotations

import modal


APP_NAME = "comma-lab-dilated-h64"
TRAINER_PATH = "experiments/train_postfilter_dilated_h64.py"
REMOTE_TRAINER = "/root/train_postfilter_dilated_h64.py"
ARCHIVE_PATH = "reports/raw/2026-04-06-av1-roi-experiments/decode_base_archive.zip"
REMOTE_ARCHIVE = "/root/decode_base_archive.zip"
VOLUME_NAME = "comma-lab-dilated-h64"


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "av",
        "safetensors",
        "timm",
        "einops",
        "segmentation-models-pytorch",
        "numpy",
    )
    .run_commands("apt-get update && apt-get install -y git git-lfs")
    .add_local_file(TRAINER_PATH, REMOTE_TRAINER)
    .add_local_file(ARCHIVE_PATH, REMOTE_ARCHIVE)
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 6,
    volumes={"/results": vol},
)
def train_dilated_h64():
    import glob
    import os
    import shutil
    import subprocess

    os.makedirs("/results/weights", exist_ok=True)
    os.environ["POSTFILTER_OUTPUT_DIR"] = "/results/weights"

    result = subprocess.run(
        [
            "python",
            REMOTE_TRAINER,
            "--hidden",
            "64",
            "--alpha",
            "20",
            "--epochs",
            "2500",
            "--eval-subsample",
            "1",
            "--checkpoint-eval-every",
            "10",
            "--checkpoint-select-int8",
            "--per-channel-int8",
            "--tag",
            "dilated_h64_long1000_modal",
        ],
        cwd="/tmp",
    )

    for pattern in ["/tmp/**/*.pt", "/tmp/**/*.json"]:
        for path in glob.glob(pattern, recursive=True):
            dest = f"/results/weights/{os.path.basename(path)}"
            if not os.path.exists(dest):
                shutil.copy2(path, dest)

    vol.commit()
    return result.returncode


@app.local_entrypoint()
def main():
    result = train_dilated_h64.remote()
    print(f"Modal dilated h64 exit code: {result}")
