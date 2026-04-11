"""Deploy GPU renderer smoke tests to Modal A10G.

Installs tac as a package, uses precomputed data volume, runs renderer
smoke tests via the canonical CLI (same as local).

Usage:
    .venv/bin/modal run src/tac/deploy/modal/modal_renderer_smoke_deploy.py
    .venv/bin/modal run src/tac/deploy/modal/modal_renderer_smoke_deploy.py --profile dp_sims_smoke
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tac-renderer-smoke"
REPO_ROOT = Path(__file__).resolve().parents[4]  # src/tac/deploy/modal -> repo root
PRECOMPUTED_VOL = "tac-precomputed"
RESULTS_VOL = "tac-renderer-results"

app = modal.App(APP_NAME)

# Image: PyTorch + scorer dependencies + tac source
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "ffmpeg")
    .pip_install(
        "torch==2.6.*",
        "torchvision",
        "av",
        "numpy",
        "pydantic>=2.0",
        "safetensors",
        "timm",
        "einops",
        "segmentation-models-pytorch",
    )
    # Copy tac source into the image
    .add_local_dir(str(REPO_ROOT / "src"), "/root/src")
    .env({"PYTHONPATH": "/root/src"})
)

precomputed_vol = modal.Volume.from_name(PRECOMPUTED_VOL, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 2,
    volumes={"/data": precomputed_vol, "/results": results_vol},
    memory=32768,
)
def train_renderer(profile: str, tag: str, extra_args: list[str] | None = None):
    """Run renderer training via the canonical CLI."""
    import os
    import subprocess
    import sys

    os.makedirs(f"/results/{tag}", exist_ok=True)

    # Check precomputed data
    precomputed = "/data/precomputed"
    has_precomputed = os.path.exists(f"{precomputed}/comp_frames.pt")

    print(f"=== tac renderer training: {profile} | tag: {tag} ===")
    print(f"  GPU: CUDA ({os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')})")
    print(f"  Precomputed: {'YES' if has_precomputed else 'NO (will decode video)'}")

    # Build command — use the canonical CLI, same as local
    cmd = [
        sys.executable, "-m", "tac.experiments.train_renderer",
        "--profile", profile,
        "--tag", tag,
        "--output-dir", f"/results/{tag}",
        "--wall-clock-timeout", "7000",
    ]
    if has_precomputed:
        cmd.extend(["--precomputed", precomputed])
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Command: {' '.join(cmd)}")
    print()

    # Run training
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/root/src"})

    # Persist results
    results_vol.commit()

    print(f"\n=== {profile} complete (exit {result.returncode}) ===")
    return {"profile": profile, "tag": tag, "exit_code": result.returncode}


@app.local_entrypoint()
def main(
    profile: str = "",
    tag: str = "",
):
    """Launch renderer smoke tests. No args = all 3 in parallel."""
    if profile:
        profiles = [(profile, tag or f"{profile}_modal")]
    else:
        profiles = [
            ("mask_renderer_smoke", "mask_renderer_smoke_modal"),
            ("dp_sims_smoke", "dp_sims_smoke_modal"),
            ("wavelet_renderer_smoke", "wavelet_smoke_modal"),
        ]

    print(f"Launching {len(profiles)} renderer experiment(s) on Modal A10G...")
    for p, t in profiles:
        print(f"  {p} -> {t}")
    print()

    # Launch in parallel
    handles = []
    for p, t in profiles:
        handles.append(train_renderer.spawn(profile=p, tag=t))

    # Collect results
    for h, (p, t) in zip(handles, profiles):
        try:
            result = h.get()
            status = "OK" if result["exit_code"] == 0 else f"FAILED (exit {result['exit_code']})"
            print(f"  {p}: {status}")
        except Exception as e:
            print(f"  {p}: ERROR — {e}")

    print(f"\nResults: .venv/bin/modal volume ls {RESULTS_VOL}")
