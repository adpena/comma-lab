#!/usr/bin/env python3
"""Universal cloud training deployer for GCP and Lightning AI.

Creates a self-contained training package that works anywhere with:
- Python 3.11+
- A GPU (T4, A10G, or better)
- ~8GB disk space

Usage:
    # Package everything for upload
    python deploy/cloud_deploy.py package

    # Deploy to GCP (requires gcloud CLI + project with GPU quota)
    python deploy/cloud_deploy.py gcp --project YOUR_PROJECT

    # Deploy to Lightning AI (requires lightning-sdk CLI)
    python deploy/cloud_deploy.py lightning

    # Generate a Colab notebook
    python deploy/cloud_deploy.py colab
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).parent.parent


def package(output_dir: str = "experiments/cloud_package"):
    """Create a self-contained training package."""
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Copy tac library
    shutil.copytree(REPO / "src" / "tac", out / "src" / "tac")
    # Copy training script
    shutil.copy2(REPO / "experiments" / "train_tac.py", out / "train_tac.py")
    # Copy data files
    shutil.copy2(REPO / "submissions" / "robust_current" / "archive.zip", out / "archive.zip")
    sal_path = REPO / "experiments" / "masks" / "posenet_saliency.npy"
    if sal_path.exists():
        shutil.copy2(sal_path, out / "saliency.npy")

    # Create requirements.txt
    (out / "requirements.txt").write_text(
        "torch\nav\nsafetensors\ntimm\neinops\nsegmentation-models-pytorch\nnumpy\npydantic\n"
    )

    # Create setup script
    (out / "setup.sh").write_text("""#!/bin/bash
set -euo pipefail
echo "=== Setting up tac training environment ==="

# Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone upstream for scorer models
if [ ! -d upstream ]; then
    git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git upstream
    cd upstream && git lfs pull && cd ..
fi

# Install deps
uv pip install --system -r requirements.txt

echo "=== Setup complete ==="
""")
    os.chmod(out / "setup.sh", 0o755)

    # Create run script
    (out / "run.sh").write_text("""#!/bin/bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:$(pwd)/upstream:$PYTHONPATH"
export PYTHONUNBUFFERED=1

TAG="${TAG:-cloud_h64_standard}"
HIDDEN="${HIDDEN:-64}"
EPOCHS="${EPOCHS:-2500}"
LOSS="${LOSS:-standard}"
TEMP_START="${TEMP_START:-1.0}"
TEMP_END="${TEMP_END:-0.05}"

echo "=== Training: tag=$TAG h=$HIDDEN epochs=$EPOCHS loss=$LOSS ==="

python3 train_tac.py \
    --tag "$TAG" \
    --hidden "$HIDDEN" \
    --epochs "$EPOCHS" \
    --loss-mode "$LOSS" \
    --temperature-start "$TEMP_START" \
    --temperature-end "$TEMP_END" \
    --alpha 20 \
    --sal-lambda 1.0 \
    --subsample 4 \
    --output-dir ./weights \
    --archive ./archive.zip \
    --gt-video ./upstream/videos/0.mkv \
    --saliency ./saliency.npy \
    --models-dir ./upstream/models \
    --upstream-dir ./upstream
""")
    os.chmod(out / "run.sh", 0o755)

    pkg_size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Package created at {out}/ ({pkg_size:.1f} MB)")
    print(f"Contents: {list(out.iterdir())}")
    return out


def deploy_gcp(project: str, zone: str = "us-central1-a", machine: str = "n1-standard-4"):
    """Deploy to GCP Compute Engine with T4 GPU."""
    pkg = package()

    print(f"Deploying to GCP project={project}, zone={zone}")

    # Create startup script
    startup = """#!/bin/bash
set -euo pipefail
cd /opt/training
bash setup.sh
TAG=gcp_h64_standard HIDDEN=64 EPOCHS=2500 bash run.sh 2>&1 | tee /opt/training/train.log
# Copy results to GCS when done
gsutil -m cp -r /opt/training/weights/ gs://{project}-tac-weights/
"""

    startup_path = pkg / "startup.sh"
    startup_path.write_text(startup.format(project=project))

    print("To deploy manually:")
    print("  1. gcloud compute instances create tac-trainer \\")
    print(f"       --project={project} --zone={zone} \\")
    print(f"       --machine-type={machine} --accelerator=type=nvidia-tesla-t4,count=1 \\")
    print("       --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \\")
    print("       --boot-disk-size=50GB --maintenance-policy=TERMINATE")
    print(f"  2. gcloud compute scp --recurse {pkg}/ tac-trainer:/opt/training/ --project={project} --zone={zone}")
    print(f"  3. gcloud compute ssh tac-trainer --project={project} --zone={zone} -- 'cd /opt/training && bash setup.sh && bash run.sh'")
    print(f"  4. gcloud compute scp --recurse tac-trainer:/opt/training/weights/ ./gcp_weights/ --project={project} --zone={zone}")


def deploy_lightning():
    """Deploy to Lightning AI studio."""
    pkg = package()

    print("Lightning AI deployment:")
    print("  1. Go to https://lightning.ai/studios")
    print("  2. Create new Studio with GPU (T4 or A10G)")
    print(f"  3. Upload {pkg}/ to the studio")
    print("  4. Open terminal and run:")
    print("     cd cloud_package && bash setup.sh && bash run.sh")
    print()
    print("  Or with kl_distill:")
    print("     TAG=lightning_h64_kl LOSS=kl_distill TEMP_START=5.0 TEMP_END=1.0 bash run.sh")

    # Also create a Lightning CLI config if the SDK CLI is installed.
    # Do not execute `lightning --version` here: if a poisoned PyPI
    # `lightning` console script is first on PATH, running it could import the
    # compromised package. Inspect installed package metadata instead.
    try:
        version = importlib.metadata.version("lightning-sdk")
        print(f"\nLightning SDK detected ({version})! Auto-deploy coming soon.")
    except importlib.metadata.PackageNotFoundError:
        print("\nInstall Lightning SDK CLI for auto-deploy: uv pip install lightning-sdk")


def generate_colab():
    """Generate a Colab notebook for training."""
    pkg = package()

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# tac Training — comma video compression challenge\n",
                           "Hardened tac v0.8.0 with pydantic validation, 61 tests, atomic saves."]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Setup\n",
                    "!nvidia-smi\n",
                    "!pip install torch av safetensors timm einops segmentation-models-pytorch numpy pydantic\n",
                    "!git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git upstream\n",
                    "!cd upstream && git lfs pull\n",
                ],
                "execution_count": None, "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Upload archive.zip and saliency.npy from cloud_package/\n",
                    "from google.colab import files\n",
                    "uploaded = files.upload()  # upload archive.zip\n",
                ],
                "execution_count": None, "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Upload tac library\n",
                    "# Upload the cloud_package/src/ directory\n",
                    "import sys\n",
                    "sys.path.insert(0, 'src')\n",
                    "sys.path.insert(0, 'upstream')\n",
                ],
                "execution_count": None, "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Train\n",
                    "!PYTHONPATH=src:upstream PYTHONUNBUFFERED=1 python train_tac.py \\\n",
                    "    --tag colab_h64_standard --hidden 64 --epochs 2500 \\\n",
                    "    --alpha 20 --sal-lambda 1.0 --subsample 4 \\\n",
                    "    --output-dir ./weights \\\n",
                    "    --archive ./archive.zip \\\n",
                    "    --gt-video ./upstream/videos/0.mkv \\\n",
                    "    --saliency ./saliency.npy \\\n",
                    "    --models-dir ./upstream/models \\\n",
                    "    --upstream-dir ./upstream\n",
                ],
                "execution_count": None, "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Download results\n",
                    "from google.colab import files\n",
                    "import glob\n",
                    "for f in glob.glob('weights/*best*'):\n",
                    "    files.download(f)\n",
                ],
                "execution_count": None, "outputs": []
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    nb_path = pkg / "tac_training.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=2))
    print(f"Colab notebook: {nb_path}")
    print("Upload to https://colab.research.google.com/")


def main():
    parser = argparse.ArgumentParser(description="Universal cloud training deployer")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("package", help="Create self-contained training package")

    gcp_p = sub.add_parser("gcp", help="Deploy to Google Cloud")
    gcp_p.add_argument("--project", required=True)
    gcp_p.add_argument("--zone", default="us-central1-a")

    sub.add_parser("lightning", help="Deploy to Lightning AI")
    sub.add_parser("colab", help="Generate Colab notebook")

    args = parser.parse_args()

    if args.command == "package":
        package()
    elif args.command == "gcp":
        deploy_gcp(args.project, args.zone)
    elif args.command == "lightning":
        deploy_lightning()
    elif args.command == "colab":
        generate_colab()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
