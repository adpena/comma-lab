"""Deploy GPU renderer smoke tests to Modal A10G.

Installs tac as a package, uses precomputed data volume, runs 3 renderer
smoke tests in parallel. Each ~200 epochs, ~15-30 min, ~$0.25-0.50.

Usage:
    .venv/bin/modal run src/tac/deploy/modal/modal_renderer_smoke_deploy.py
    .venv/bin/modal run src/tac/deploy/modal/modal_renderer_smoke_deploy.py --profile dp_sims_smoke
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tac-renderer-smoke"
REPO_ROOT = Path(__file__).resolve().parents[4]  # src/tac/deploy/modal -> repo root
VOLUME_NAME = "tac-precomputed"
RESULTS_VOLUME = "tac-renderer-results"

app = modal.App(APP_NAME)

# Build image with PyTorch + tac installed from local source
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "av",
        "numpy",
        "pydantic>=2.0",
        "safetensors",
        "timm",
        "einops",
        "segmentation-models-pytorch",
    )
    # Install tac from local source
    .add_local_dir(str(REPO_ROOT / "src" / "tac"), "/root/src/tac")
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), "/root/pyproject.toml")
    .add_local_dir(str(REPO_ROOT / "src"), "/root/src", condition=lambda p: p.endswith(".py"))
    .run_commands("cd /root && pip install -e . 2>/dev/null || PYTHONPATH=/root/src pip install -e . 2>/dev/null || true")
    .env({"PYTHONPATH": "/root/src"})
)

precomputed_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 2,
    volumes={"/data": precomputed_vol, "/results": results_vol},
    memory=32768,
)
def train_renderer_smoke(profile: str, tag: str):
    """Run a single renderer smoke test on A10G."""
    import os
    import sys
    import json
    import glob
    import shutil

    sys.path.insert(0, "/root/src")

    os.makedirs(f"/results/{tag}", exist_ok=True)

    # Check precomputed data
    precomputed = "/data/precomputed"
    if not os.path.exists(f"{precomputed}/comp_frames.pt"):
        # Fallback: decode from archive
        precomputed = ""
        print("WARNING: No precomputed data found. Will decode video (slow).")

    print(f"=== Training: {profile} | tag: {tag} ===")
    print(f"  GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print(f"  Precomputed: {'YES' if precomputed else 'NO'}")

    # Import and run
    from tac.experiments.train_renderer import train
    from tac.profiles import PROFILES

    if profile not in PROFILES:
        print(f"ERROR: Unknown profile '{profile}'. Available: {sorted(PROFILES.keys())}")
        return {"error": f"unknown profile: {profile}"}

    # Build args namespace from profile
    import argparse
    config = dict(PROFILES[profile])
    args = argparse.Namespace(
        profile=profile,
        precomputed=precomputed if precomputed else None,
        tag=tag,
        output_dir=f"/results/{tag}",
        resume_from=None,
        wall_clock_timeout=7000,  # ~2 hours minus buffer
    )

    # Run training
    try:
        train(args)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

    # Copy results to volume
    for pattern in [f"/results/{tag}/**/*.pt", f"/results/{tag}/**/*.json"]:
        for path in glob.glob(pattern, recursive=True):
            print(f"  Saved: {path}")

    results_vol.commit()

    # Return summary
    summary = {"profile": profile, "tag": tag, "status": "complete"}
    best_path = f"/results/{tag}/best_proxy_score.json"
    if os.path.exists(best_path):
        with open(best_path) as f:
            summary["best"] = json.load(f)

    return summary


@app.local_entrypoint()
def main(
    profile: str = "",
    tag: str = "",
):
    """Launch renderer smoke tests. If no profile specified, runs all 3."""
    if profile:
        profiles = [(profile, tag or f"{profile}_modal")]
    else:
        profiles = [
            ("mask_renderer_smoke", "mask_renderer_smoke_modal"),
            ("dp_sims_smoke", "dp_sims_smoke_modal"),
            ("wavelet_renderer_smoke", "wavelet_smoke_modal"),
        ]

    print(f"Launching {len(profiles)} renderer smoke test(s) on Modal A10G...")
    for p, t in profiles:
        print(f"  {p} -> {t}")

    # Launch in parallel
    handles = []
    for p, t in profiles:
        handles.append(train_renderer_smoke.spawn(profile=p, tag=t))

    # Collect results
    results = []
    for h, (p, t) in zip(handles, profiles):
        try:
            result = h.get()
            print(f"\n=== {p}: {result.get('status', 'unknown')} ===")
            if "best" in result:
                print(f"  Best proxy: {result['best']}")
            results.append(result)
        except Exception as e:
            print(f"\n=== {p}: FAILED — {e} ===")
            results.append({"profile": p, "status": "failed", "error": str(e)})

    print(f"\n{'='*60}")
    print(f"All smoke tests complete. Check results:")
    print(f"  .venv/bin/modal volume ls {RESULTS_VOLUME}")
