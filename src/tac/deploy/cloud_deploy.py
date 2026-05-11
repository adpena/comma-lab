#!/usr/bin/env python3
"""Compatibility CLI for provider-neutral cloud training bundles.

The canonical implementation lives in :mod:`tac.deploy.build_bundle` and the
training command is derived from :mod:`tac.deploy.deploy_config`. This wrapper
keeps the old ``cloud_deploy.py`` entry point discoverable without carrying a
second copy of bundle-building logic.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from tac.deploy.build_bundle import build as build_bundle


def package(output_dir: str = "experiments/cloud_package") -> Path:
    """Create a self-contained training package via the canonical builder."""

    return build_bundle(output=output_dir, include_saliency=True, include_precomputed=False)


def deploy_gcp(project: str, zone: str = "us-central1-a", machine: str = "n1-standard-4") -> None:
    """Print the manual GCP launch commands for the canonical bundle."""

    pkg = package()
    startup = """#!/bin/bash
set -euo pipefail
cd /opt/training
bash setup.sh
TAG=gcp_base VARIANT=base bash train.sh 2>&1 | tee /opt/training/train.log
gsutil -m cp -r /opt/training/weights/ gs://{project}-tac-weights/
"""
    startup_path = pkg / "startup.sh"
    startup_path.write_text(startup.format(project=project), encoding="utf-8")

    print(f"Deploying to GCP project={project}, zone={zone}")
    print("Manual launch:")
    print("  1. gcloud compute instances create tac-trainer \\")
    print(f"       --project={project} --zone={zone} \\")
    print(f"       --machine-type={machine} --accelerator=type=nvidia-tesla-t4,count=1 \\")
    print("       --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \\")
    print("       --boot-disk-size=60GB --maintenance-policy=TERMINATE")
    print(f"  2. gcloud compute scp --recurse {pkg}/ tac-trainer:/opt/training/ --project={project} --zone={zone}")
    print(f"  3. gcloud compute ssh tac-trainer --project={project} --zone={zone} -- 'cd /opt/training && bash setup.sh && TAG=gcp_base VARIANT=base bash train.sh'")
    print(f"  4. gcloud compute scp --recurse tac-trainer:/opt/training/weights/ ./gcp_weights/ --project={project} --zone={zone}")


def deploy_lightning() -> None:
    """Print the manual Lightning Studio launch commands for the canonical bundle."""

    pkg = package()
    print("Lightning AI deployment:")
    print("  1. Create or open a Studio with GPU (T4 or better)")
    print(f"  2. Upload {pkg}/ to the Studio")
    print("  3. Run: cd cloud_package && bash setup.sh && TAG=lightning_base VARIANT=base bash train.sh")
    print("  4. Use `scripts/launch_lightning_batch_job.py` for promotion-grade exact eval")

    try:
        version = importlib.metadata.version("lightning-sdk")
        print(f"\nLightning SDK detected ({version}).")
    except importlib.metadata.PackageNotFoundError:
        print("\nInstall Lightning SDK for batch jobs: uv pip install lightning-sdk")


def generate_colab() -> None:
    """Generate a small Colab notebook that runs the canonical bundle."""

    pkg = package()
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# tac training bundle\n",
                    "Runs the canonical deploy bundle with provider-agnostic flags.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["!nvidia-smi\n", "!bash setup.sh\n"],
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["!TAG=colab_base VARIANT=base bash train.sh\n"],
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from google.colab import files\n",
                    "import glob\n",
                    "for f in glob.glob('weights/*best*'):\n",
                    "    files.download(f)\n",
                ],
                "execution_count": None,
                "outputs": [],
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
    nb_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(f"Colab notebook: {nb_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("package", help="Create self-contained training package")

    gcp_p = sub.add_parser("gcp", help="Print Google Cloud launch commands")
    gcp_p.add_argument("--project", required=True)
    gcp_p.add_argument("--zone", default="us-central1-a")

    sub.add_parser("lightning", help="Print Lightning Studio launch commands")
    sub.add_parser("colab", help="Generate Colab notebook in the bundle")

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
