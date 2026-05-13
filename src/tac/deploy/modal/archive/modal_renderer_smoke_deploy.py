"""Deploy GPU renderer smoke tests to Modal A10G.

Uses precomputed data volume + tac source. Runs training via subprocess
with the canonical CLI (identical to local execution).

Usage:
    .venv/bin/modal run src/tac/deploy/modal/modal_renderer_smoke_deploy.py
    .venv/bin/modal run src/tac/deploy/modal/modal_renderer_smoke_deploy.py --profile dp_sims_smoke
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tac-renderer-smoke"
PRECOMPUTED_VOL = "tac-precomputed"
RESULTS_VOL = "tac-renderer-results"

# REPO_ROOT only needed locally for add_local_dir — guard for container env
_script = Path(__file__).resolve()
try:
    REPO_ROOT = _script.parents[4]  # src/tac/deploy/modal -> repo root
except IndexError:
    REPO_ROOT = Path("/root")  # inside Modal container

app = modal.App(APP_NAME)

# Build image: PyTorch + scorer deps + tac source (only src/tac/, no pycache)
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
    # Clone upstream scorer repo (PoseNet/SegNet model definitions)
    .run_commands(
        "git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git /root/upstream",
        "cd /root/upstream && git lfs pull",
    )
    .env({
        "PYTHONPATH": "/root/src:/root/upstream",
        "PYTHONUNBUFFERED": "1",
        "TAC_UPSTREAM_DIR": "/root/upstream",
        "TAC_MODELS_DIR": "/root/upstream/models",
    })
    # add_local_dir must be LAST — Modal mounts these at startup, not during build
    .add_local_dir(str(REPO_ROOT / "src" / "tac"), "/root/src/tac")
)

precomputed_vol = modal.Volume.from_name(PRECOMPUTED_VOL, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)


def _run_with_periodic_commits(cmd: list[str], env: dict, commit_interval: int = 300):
    """Run a subprocess with periodic Modal volume commits.

    Commits results_vol every commit_interval seconds during training,
    so partial results survive if the local CLI is killed or times out.
    """
    import subprocess
    import threading
    import time

    training_done = threading.Event()

    def _periodic_commit():
        while not training_done.is_set():
            training_done.wait(timeout=commit_interval)
            if not training_done.is_set():
                try:
                    results_vol.commit()
                    print(f"  [volume] Periodic commit at {time.strftime('%H:%M:%S')}")
                except Exception as e:
                    print(f"  [volume] Commit failed: {e}")

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    try:
        result = subprocess.run(cmd, env=env)
    finally:
        training_done.set()
        results_vol.commit()
        print("  [volume] Final commit done")

    return result


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 2,
    volumes={"/data": precomputed_vol, "/results": results_vol},
    memory=32768,
)
def train_renderer(profile: str, tag: str, extra_args: list[str] | None = None):
    """Run renderer training via canonical CLI on A10G."""
    import os
    import sys

    os.makedirs(f"/results/{tag}", exist_ok=True)

    # Check precomputed data
    precomputed = "/data/precomputed"
    has_precomputed = os.path.exists(f"{precomputed}/comp_frames.pt")

    print(f"=== tac renderer training: {profile} | tag: {tag} ===")
    print("  GPU: CUDA")
    print(f"  Precomputed: {'YES' if has_precomputed else 'NO'}")

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
    result = _run_with_periodic_commits(cmd, env={**os.environ, "PYTHONPATH": "/root/src:/root/upstream"})
    return {"profile": profile, "tag": tag, "exit_code": result.returncode}


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 2,
    volumes={"/data": precomputed_vol, "/results": results_vol},
    memory=32768,
)
def train_postfilter(profile: str, tag: str, extra_args: list[str] | None = None):
    """Run CPU-lane postfilter training on A10G (faster than MPS)."""
    import os
    import sys

    os.makedirs(f"/results/{tag}", exist_ok=True)

    precomputed = "/data/precomputed"
    has_precomputed = os.path.exists(f"{precomputed}/comp_frames.pt")

    print(f"=== tac postfilter training: {profile} | tag: {tag} ===")
    print("  GPU: CUDA")
    print(f"  Precomputed: {'YES' if has_precomputed else 'NO'}")

    cmd = [
        sys.executable, "-m", "tac",
        "lossy",
        "--profile", profile,
        "--tag", tag,
        "--output-dir", f"/results/{tag}",
    ]
    if has_precomputed:
        cmd.extend(["--precomputed", precomputed])
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Command: {' '.join(cmd)}")
    result = _run_with_periodic_commits(cmd, env={**os.environ, "PYTHONPATH": "/root/src:/root/upstream"})
    return {"profile": profile, "tag": tag, "exit_code": result.returncode}


@app.local_entrypoint()
def main(
    profile: str = "",
    tag: str = "",
    lane: str = "gpu",
):
    """Launch training on Modal A10G.

    Args:
        profile: tac profile name (e.g., mask_renderer_smoke, proven_baseline)
        tag: experiment tag for output directory
        lane: 'gpu' for renderer, 'cpu' for postfilter
    """
    if not profile:
        # Pre-flight cost estimate
        from tac.cost_tracker import print_cost_estimate
        print("\n--- Cost Estimate (per test) ---")
        print_cost_estimate(gpu="a10g", estimated_hours=2.0, platform="modal")
        print(f"  x3 parallel = ~${3 * 2.0 * 1.04:.2f} total\n")

        # Default: all 3 GPU renderer smoke tests in parallel
        profiles = [
            ("mask_renderer_smoke", "mask_renderer_smoke_a10g"),
            ("dp_sims_smoke", "dp_sims_smoke_a10g"),
            ("wavelet_renderer_smoke", "wavelet_smoke_a10g"),
        ]
        print("Launching all 3 GPU renderer smoke tests on Modal A10G...")
        handles = []
        for p, t in profiles:
            print(f"  {p} -> {t}")
            handles.append(train_renderer.spawn(profile=p, tag=t))

        for h, (p, t) in zip(handles, profiles):
            try:
                result = h.get()
                status = "OK" if result["exit_code"] == 0 else f"FAILED (exit {result['exit_code']})"
                print(f"  {p}: {status}")
            except Exception as e:
                print(f"  {p}: ERROR — {e}")
    else:
        t = tag or f"{profile}_a10g"
        print(f"Launching {lane} lane: {profile} -> {t}")
        fn = train_postfilter if lane == "cpu" else train_renderer
        result = fn.remote(profile=profile, tag=t)
        status = "OK" if result["exit_code"] == 0 else f"FAILED (exit {result['exit_code']})"
        print(f"  {profile}: {status}")

    print(f"\nResults: .venv/bin/modal volume ls {RESULTS_VOL}")
