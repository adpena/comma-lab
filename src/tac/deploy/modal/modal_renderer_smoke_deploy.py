"""Deploy GPU renderer smoke tests to Modal A10G.

Runs MaskRenderer + DP-SIMS + Wavelet smoke profiles in parallel.
Each smoke test: ~200 epochs, ~15-30 minutes, ~$0.25-0.50.

Usage:
    modal run src/tac/deploy/modal/modal_renderer_smoke_deploy.py
"""
from __future__ import annotations

import modal

app = modal.App("tac-renderer-smoke")

# Image with PyTorch + tac dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "numpy", "pydantic", "av")
    .pip_install("uv")
)

# Precomputed data volume
precomputed_vol = modal.Volume.from_name("tac-precomputed", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": precomputed_vol},
)
def train_renderer_smoke(profile: str, tag: str):
    """Run a single renderer smoke test."""
    import subprocess
    import sys

    # Install tac from the repo
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "/root/tac"],
        check=True, capture_output=True,
    )

    from tac.experiments.train_renderer import train
    from tac.profiles import PROFILES

    profile_config = PROFILES[profile]
    print(f"Running {profile} smoke test on A10G...")
    print(f"  Config: {profile_config}")

    # Train with precomputed data
    import argparse
    args = argparse.Namespace(
        profile=profile,
        precomputed="/data/precomputed",
        tag=tag,
        output_dir="/root/output",
        **{k.replace("-", "_"): v for k, v in profile_config.items()
           if k not in ("profile",)},
    )

    train(args)
    print(f"Smoke test {profile} complete.")


@app.local_entrypoint()
def main():
    """Launch all 3 smoke tests in parallel."""
    profiles = [
        ("mask_renderer_smoke", "mask_renderer_smoke_modal"),
        ("dp_sims_smoke", "dp_sims_smoke_modal"),
        ("wavelet_renderer_smoke", "wavelet_smoke_modal"),
    ]

    # Launch in parallel
    futures = []
    for profile, tag in profiles:
        print(f"Launching {profile}...")
        futures.append(train_renderer_smoke.spawn(profile=profile, tag=tag))

    # Collect results
    for f, (profile, tag) in zip(futures, profiles):
        try:
            f.get()
            print(f"  {profile}: COMPLETE")
        except Exception as e:
            print(f"  {profile}: FAILED — {e}")
