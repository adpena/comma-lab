"""Deploy dilated h=64 + KL distill + hard-frame curriculum to Modal A10G.

KL distill is showing the best early signal (1.467 at ep 7 on MPS).
Adding hard-frame curriculum (0.5 ratio) to focus on worst SegNet pairs.
This combination is not running anywhere else.

Usage:
    .venv/bin/modal run experiments/modal_dilated_kl_hardframe_deploy.py
"""
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent

app = modal.App("comma-lab-dilated-kl-hardframe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "av", "safetensors", "timm", "einops",
        "segmentation-models-pytorch", "numpy", "pydantic",
    )
    .run_commands(
        "apt-get update && apt-get install -y git git-lfs unzip",
        "git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git /upstream",
        "cd /upstream && git lfs pull",
    )
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
def train():
    import os
    import subprocess
    import sys

    print("=== Modal dilated h=64 + KL distill + hard-frame curriculum ===")
    gpu = os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read().strip()
    print(f"GPU: {gpu}")

    os.makedirs("/vol/weights_kl_hf", exist_ok=True)

    archive_path = "/vol/archive.zip"
    if not os.path.exists(archive_path):
        print("ERROR: archive.zip not on volume")
        return 1

    saliency_path = "/vol/saliency.npy"
    if not os.path.exists(saliency_path):
        import numpy as np
        print("Generating uniform saliency...")
        np.save(saliency_path, np.ones((30, 874, 1164), dtype=np.float32))

    # Precompute if not available
    precomputed_dir = "/vol/precomputed"
    precomputed_args = []
    if os.path.exists(os.path.join(precomputed_dir, "comp_frames.pt")):
        print(f"Using precomputed frames from {precomputed_dir}")
        precomputed_args = ["--precomputed", precomputed_dir]
    else:
        os.makedirs(precomputed_dir, exist_ok=True)
        sys.path.insert(0, "/app/src")
        sys.path.insert(0, "/upstream")
        import torch
        from tac.data import decode_archive, decode_video
        print("Precomputing frames...")
        comp = decode_archive(archive_path)
        gt = decode_video("/upstream/videos/0.mkv")
        torch.save(torch.stack(comp), os.path.join(precomputed_dir, "comp_frames.pt"))
        torch.save(torch.stack(gt), os.path.join(precomputed_dir, "gt_frames.pt"))
        vol.commit()
        print("Saved precomputed frames")
        precomputed_args = ["--precomputed", precomputed_dir]

    resume_path = "/vol/weights_kl_hf/training_state_dilated_kl_hardframe.pt"
    resume_args = []
    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        resume_args = ["--resume-from", resume_path]

    result = subprocess.run(
        [
            sys.executable, "/app/train_tac.py",
            "--tag", "dilated_kl_hardframe",
            "--variant", "dilated",
            "--hidden", "64",
            "--epochs", "2500",
            "--alpha", "20",
            "--sal-lambda", "1.0",
            "--loss-mode", "kl_distill",
            "--temperature-start", "5.0",
            "--temperature-end", "1.0",
            "--hard-frame-ratio", "0.5",
            "--subsample", "4",
            "--eval-every", "5",
            "--output-dir", "/vol/weights_kl_hf",
            "--archive", archive_path,
            "--gt-video", "/upstream/videos/0.mkv",
            "--saliency", saliency_path,
            "--models-dir", "/upstream/models",
            "--upstream-dir", "/upstream",
            *precomputed_args,
            *resume_args,
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
    import subprocess as sp

    archive = REPO / "submissions" / "robust_current" / "archive.zip"
    saliency = REPO / "experiments" / "masks" / "posenet_saliency.npy"

    print("Uploading data to Modal volume...")
    sp.run([".venv/bin/modal", "volume", "put", "comma-lab-weights",
            str(archive), "archive.zip", "--force"])
    if saliency.exists():
        sp.run([".venv/bin/modal", "volume", "put", "comma-lab-weights",
                str(saliency), "saliency.npy", "--force"])

    print("Deploying KL distill + hard-frame to Modal A10G...")
    result = train.remote()
    print(f"Training completed: exit code {result}")
