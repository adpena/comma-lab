"""Nuclear training: h=96/128 on H100 with all optimizations.

The final push. Combines:
- Larger model (h=96 or h=128)
- Temperature-annealed SegNet loss (unlock headroom)
- Full 5000 epochs
- H100 GPU (~8s/epoch)
- Hardened tac library with uint8-compliant eval
- Resume from any prior training state

Cost: ~$54 for h=128/5000ep on H100, ~$27 for h=96/5000ep

Usage:
    # h=96, temperature loss, 5000 epochs
    .venv/bin/modal run deploy/modal/modal_nuclear_deploy.py

    # h=128, standard loss, 5000 epochs
    .venv/bin/modal run deploy/modal/modal_nuclear_deploy.py --hidden 128

    # Download results
    .venv/bin/modal volume get comma-lab-weights weights/ ./modal_weights/ --force
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2]

app = modal.App("comma-lab-nuclear")

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
    gpu="H100",
    timeout=3600 * 12,  # 12 hours
    volumes={"/vol": vol},
)
def train_nuclear(
    hidden: int = 96,
    epochs: int = 5000,
    loss_mode: str = "temperature",
    tag: str = "",
):
    import os
    import subprocess
    import sys

    if not tag:
        tag = f"nuclear_h{hidden}_{loss_mode}"

    print(f"=== NUCLEAR RUN: h={hidden}, {loss_mode}, {epochs} epochs ===")
    gpu = os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read().strip()
    print(f"GPU: {gpu}")

    os.makedirs("/vol/weights", exist_ok=True)

    archive_path = "/vol/archive.zip"
    if not os.path.exists(archive_path):
        print("ERROR: archive.zip not on volume")
        return 1

    saliency_path = "/vol/saliency.npy"
    if not os.path.exists(saliency_path):
        import numpy as np
        print("Generating uniform saliency...")
        np.save(saliency_path, np.ones((30, 874, 1164), dtype=np.float32))

    # Check for resume state
    resume_path = f"/vol/weights/training_state_{tag}.pt"
    resume_args = ["--resume-from", resume_path] if os.path.exists(resume_path) else []
    if resume_args:
        print(f"Resuming from {resume_path}")

    # Loss mode args
    loss_args = ["--loss-mode", loss_mode]
    if loss_mode == "temperature":
        loss_args += ["--temperature-start", "1.0", "--temperature-end", "0.05"]

    result = subprocess.run(
        [
            sys.executable, "/app/train_tac.py",
            "--tag", tag,
            "--hidden", str(hidden),
            "--epochs", str(epochs),
            "--alpha", "20",
            "--sal-lambda", "1.0",
            "--subsample", "4",
            "--output-dir", "/vol/weights",
            "--archive", archive_path,
            "--gt-video", "/upstream/videos/0.mkv",
            "--saliency", saliency_path,
            "--models-dir", "/upstream/models",
            "--upstream-dir", "/upstream",
            *loss_args,
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
def main(
    hidden: int = 96,
    epochs: int = 5000,
    loss_mode: str = "temperature",
    tag: str = "",
):
    import subprocess as sp

    archive = REPO / "submissions" / "robust_current" / "archive.zip"
    saliency = REPO / "experiments" / "masks" / "posenet_saliency.npy"

    print("Uploading data to Modal volume...")
    sp.run([".venv/bin/modal", "volume", "put", "comma-lab-weights",
            str(archive), "archive.zip", "--force"])
    if saliency.exists():
        sp.run([".venv/bin/modal", "volume", "put", "comma-lab-weights",
                str(saliency), "saliency.npy", "--force"])

    if not tag:
        tag = f"nuclear_h{hidden}_{loss_mode}"

    # Pre-flight cost estimate
    from tac.cost_tracker import print_cost_estimate
    print("\n--- Cost Estimate ---")
    print_cost_estimate(gpu="h100", estimated_hours=12.0, platform="modal")
    print()

    print(f"Deploying NUCLEAR run: h={hidden}, {loss_mode}, {epochs}ep on H100...")
    result = train_nuclear.remote(hidden=hidden, epochs=epochs, loss_mode=loss_mode, tag=tag)
    print(f"Completed: exit code {result}")
    print("Download: .venv/bin/modal volume get comma-lab-weights weights/ ./modal_weights/ --force")
