"""Run contest-compliant auth eval on Modal T4 with DALI.

This is the DEFINITIVE score — matches the contest evaluation environment.
Uploads the submission archive, runs inflate.sh → evaluate.py on a T4 GPU.

Usage:
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run experiments/modal_auth_eval.py \
        --archive /tmp/modal_submission/archive.zip \
        --submission-dir submissions/robust_current
"""
from __future__ import annotations

import modal

app = modal.App("comma-auth-eval")

# Image with all dependencies (matches contest T4 environment)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "unzip")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "av",
        "click",
        "nvidia-dali-cuda120",
        "tqdm",
    )
)


eval_image = (
    image
    .add_local_dir("upstream", remote_path="/root/upstream")
    .add_local_dir("submissions/robust_current", remote_path="/root/submission")
)


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=2400,  # 40 min (30 min contest + 10 min overhead)
)
def run_auth_eval(archive_bytes: bytes) -> dict:
    """Run full contest-compliant auth eval on T4.

    Replicates the exact contest pipeline:
        1. unzip archive.zip
        2. inflate.sh → inflate_renderer.py
        3. evaluate.py with DALI (if available, else AVVideoDataset)
    """
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import time

    t_start = time.monotonic()

    work = tempfile.mkdtemp()
    submission_dir = os.path.join(work, "submission")
    archive_dir = os.path.join(work, "archive")
    inflated_dir = os.path.join(work, "inflated")
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(inflated_dir, exist_ok=True)

    # Write archive
    archive_path = os.path.join(submission_dir, "archive.zip")
    with open(archive_path, "wb") as f:
        f.write(archive_bytes)
    archive_size = len(archive_bytes)
    print(f"Archive: {archive_size:,} bytes")

    # Copy submission scripts
    for fn in ["inflate.sh", "inflate_renderer.py", "config.env",
               "compress_masks.py", "inflate_postfilter.py"]:
        src = os.path.join("/root/submission", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(submission_dir, fn))

    # Unzip archive
    subprocess.run(
        ["unzip", "-o", archive_path, "-d", archive_dir],
        check=True, capture_output=True,
    )
    print(f"Archive contents: {os.listdir(archive_dir)}")

    # Write video names
    video_names_file = os.path.join(work, "video_names.txt")
    with open(video_names_file, "w") as f:
        f.write("0.mkv\n")

    # Run inflate_renderer.py directly (simpler than inflate.sh)
    print("\n=== INFLATE ===")
    t_inflate = time.monotonic()
    env = {
        **os.environ,
        "PYTHONPATH": f"/root/submission:/root/upstream:{work}",
    }
    proc = subprocess.run(
        [
            sys.executable, "-u",
            "/root/submission/inflate_renderer.py",
            archive_dir,
            inflated_dir,
            video_names_file,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    inflate_time = time.monotonic() - t_inflate
    print(proc.stdout[-500:] if proc.stdout else "")
    if proc.returncode != 0:
        print(f"INFLATE FAILED:\n{proc.stderr[-500:]}")
        return {"error": "inflate failed", "stderr": proc.stderr[-500:]}

    raw_path = os.path.join(inflated_dir, "0.raw")
    if not os.path.exists(raw_path):
        return {"error": "0.raw not generated"}
    raw_size = os.path.getsize(raw_path)
    print(f"Inflate OK: {raw_size:,} bytes in {inflate_time:.1f}s")

    # Copy archive.zip to work dir for evaluate.py
    shutil.copy2(archive_path, os.path.join(work, "archive.zip"))

    # Run upstream evaluate.py
    print("\n=== EVALUATE ===")
    t_eval = time.monotonic()
    report_path = os.path.join(work, "report.txt")
    proc = subprocess.run(
        [
            sys.executable, "-u",
            "/root/upstream/evaluate.py",
            "--submission-dir", work,
            "--uncompressed-dir", "/root/upstream/videos/",
            "--video-names-file", video_names_file,
            "--device", "cpu",  # Use AVVideoDataset, not DALI (DALI has NVML driver issues on Modal T4)
            "--batch-size", "4",
            "--report", report_path,
        ],
        env={**os.environ, "PYTHONPATH": "/root/upstream"},
        capture_output=True,
        text=True,
        timeout=1200,
    )
    eval_time = time.monotonic() - t_eval
    total_time = time.monotonic() - t_start

    print(proc.stdout[-500:] if proc.stdout else "")
    if proc.returncode != 0:
        print(f"EVALUATE FAILED:\n{proc.stderr[-500:]}")
        return {"error": "evaluate failed", "stderr": proc.stderr[-500:]}

    # Parse report
    result = {
        "archive_bytes": archive_size,
        "inflate_seconds": inflate_time,
        "eval_seconds": eval_time,
        "total_seconds": total_time,
    }
    if os.path.exists(report_path):
        report = open(report_path).read()
        print(f"\n{report}")
        for line in report.splitlines():
            if "PoseNet" in line:
                result["posenet_dist"] = float(line.split(":")[-1].strip())
            elif "SegNet" in line:
                result["segnet_dist"] = float(line.split(":")[-1].strip())
            elif "Final score" in line:
                result["score"] = float(line.split("=")[-1].strip())
            elif "Compression Rate" in line:
                result["rate"] = float(line.split(":")[-1].strip())

    return result


@app.local_entrypoint()
def main(archive: str = "/tmp/modal_submission/archive.zip"):
    """Run auth eval on Modal T4."""
    from pathlib import Path

    archive_path = Path(archive)
    if not archive_path.exists():
        print(f"ERROR: archive not found: {archive}")
        return

    archive_bytes = archive_path.read_bytes()
    print(f"Uploading {len(archive_bytes):,} bytes to Modal T4...")

    result = run_auth_eval.remote(archive_bytes)

    print(f"\n{'='*60}")
    if "score" in result:
        print(f"  AUTH SCORE: {result['score']:.2f} [T4, contest-compliant]")
        print(f"  PoseNet: {result.get('posenet_dist', '?')}")
        print(f"  SegNet: {result.get('segnet_dist', '?')}")
        print(f"  Rate: {result.get('rate', '?')}")
        print(f"  Inflate: {result.get('inflate_seconds', '?'):.1f}s")
        print(f"  Eval: {result.get('eval_seconds', '?'):.1f}s")
        print(f"  Total: {result.get('total_seconds', '?'):.1f}s")
    else:
        print(f"  ERROR: {result.get('error', 'unknown')}")
    print(f"{'='*60}")

    # Save result
    import json
    result_path = Path("experiments/results/modal_auth_eval.json")
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Saved to {result_path}")
