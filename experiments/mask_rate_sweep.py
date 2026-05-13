#!/usr/bin/env python3
"""Mask compression sweep: find Pareto-optimal mask size vs score.

Runs full e2e pipeline (inflate_renderer.py → upstream evaluate.py) for each
candidate mask encoding. No shortcuts, no proxies, no approximations.

Each candidate:
  1. Encode masks at given resolution + CRF
  2. Build archive with submission_archive
  3. Run inflate_renderer.py
  4. Run upstream evaluate.py
  5. Record the TRUE upstream score

Sweep dimensions:
  - Resolution: 384x512 (native), 192x256 (1/2), 96x128 (1/4), 48x64 (1/8)
  - CRF: 20, 30, 40, 50, 63 (max compression)
  - With upsample-before-renderer fix for sub-native resolutions
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = PROJECT_ROOT / "upstream"
SUBMISSION_DIR = PROJECT_ROOT / "submissions" / "robust_current"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(UPSTREAM))

ORIGINAL_VIDEO_BYTES = 37_545_489
NUM_FRAMES = 1200
NUM_CLASSES = 5
SEG_H, SEG_W = 384, 512


def extract_fresh_masks(device: str = "mps") -> torch.Tensor:
    """Extract SegNet masks from GT video at native 384x512."""
    from tac.eval.auth_eval import AuthEvaluator

    evaluator = AuthEvaluator(upstream_dir=UPSTREAM, device=device)
    evaluator.load_scorers()
    gt_frames = evaluator.decode_gt_video("0.mkv")
    masks = evaluator.extract_masks(gt_frames, batch_size=4)
    return masks  # (1200, 384, 512) long


def downscale_masks(masks: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Downscale masks using nearest neighbor (preserves class indices)."""
    if masks.shape[1] == target_h and masks.shape[2] == target_w:
        return masks
    return F.interpolate(
        masks.float().unsqueeze(1),
        size=(target_h, target_w),
        mode="nearest",
    ).squeeze(1).long()


def encode_masks_to_mkv(
    masks: torch.Tensor, output_path: Path, crf: int = 20
) -> int:
    """Encode masks as AV1 monochrome. Returns file size."""
    from tac.mask_codec import encode_masks_monochrome
    return encode_masks_monochrome(masks, output_path, crf=crf)


def build_archive(
    masks_mkv: Path,
    renderer_bin: Path,
    optimized_poses: Path,
    output_path: Path,
) -> int:
    """Build submission archive. Returns archive size in bytes."""
    import zipfile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(renderer_bin, arcname="renderer.bin")
        zf.write(masks_mkv, arcname="masks.mkv")
        zf.write(optimized_poses, arcname="optimized_poses.pt")

    return output_path.stat().st_size


def run_inflate(archive_dir: Path, inflated_dir: Path, video_names_file: Path) -> float:
    """Run inflate_renderer.py. Returns wall-clock seconds."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{UPSTREAM}:{PROJECT_ROOT}"

    t0 = time.monotonic()
    proc = subprocess.run(
        [
            str(VENV_PYTHON), "-u",
            str(SUBMISSION_DIR / "inflate_renderer.py"),
            str(archive_dir),
            str(inflated_dir),
            str(video_names_file),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        print(f"  INFLATE FAILED:\n{proc.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError(f"inflate_renderer.py failed: {proc.returncode}")

    return elapsed


def run_upstream_evaluate(
    work_dir: Path, video_names_file: Path, device: str = "mps"
) -> dict:
    """Run upstream evaluate.py. Returns parsed score dict."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{UPSTREAM}:{PROJECT_ROOT}"

    report_path = work_dir / "report.txt"

    proc = subprocess.run(
        [
            str(VENV_PYTHON), "-u",
            str(UPSTREAM / "evaluate.py"),
            "--submission-dir", str(work_dir),
            "--uncompressed-dir", str(UPSTREAM / "videos"),
            "--video-names-file", str(video_names_file),
            "--device", device,
            "--report", str(report_path),
            "--batch-size", "4",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if proc.returncode != 0:
        print(f"  EVALUATE FAILED:\n{proc.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError(f"evaluate.py failed: {proc.returncode}")

    # Parse report
    report = report_path.read_text()
    result = {}
    for line in report.splitlines():
        if "PoseNet Distortion" in line:
            result["posenet_dist"] = float(line.split(":")[-1].strip())
        elif "SegNet Distortion" in line:
            result["segnet_dist"] = float(line.split(":")[-1].strip())
        elif "Submission file size" in line:
            result["archive_bytes"] = int(line.split(":")[-1].strip().replace(",", ""))
        elif "Original uncompressed size" in line:
            result["uncompressed_bytes"] = int(line.split(":")[-1].strip().replace(",", ""))
        elif "Compression Rate" in line:
            result["rate"] = float(line.split(":")[-1].strip())
        elif "Final score" in line:
            result["score"] = float(line.split("=")[-1].strip())

    return result


def run_single_candidate(
    masks_mkv: Path,
    renderer_bin: Path,
    optimized_poses: Path,
    label: str,
    device: str = "mps",
) -> dict:
    """Full e2e eval for one mask candidate."""
    work_dir = Path(tempfile.mkdtemp())
    archive_dir = work_dir / "archive"
    inflated_dir = work_dir / "inflated"
    archive_dir.mkdir()
    inflated_dir.mkdir()

    try:
        # Build archive
        archive_path = work_dir / "archive.zip"
        archive_bytes = build_archive(masks_mkv, renderer_bin, optimized_poses, archive_path)

        # Extract archive for inflate
        subprocess.run(
            ["unzip", "-q", "-o", str(archive_path), "-d", str(archive_dir)],
            check=True,
        )

        # Write video names
        vn = work_dir / "video_names.txt"
        vn.write_text("0.mkv\n")

        # Inflate
        inflate_time = run_inflate(archive_dir, inflated_dir, vn)

        # Check output
        raw_path = inflated_dir / "0.raw"
        if not raw_path.exists():
            raise RuntimeError("0.raw not generated")
        raw_size = raw_path.stat().st_size
        expected = 1164 * 874 * 3 * 1200
        if raw_size != expected:
            raise RuntimeError(f"Wrong raw size: {raw_size} != {expected}")

        # Evaluate
        result = run_upstream_evaluate(work_dir, vn, device)
        result["label"] = label
        result["inflate_time_s"] = inflate_time
        result["masks_mkv_bytes"] = masks_mkv.stat().st_size

        rate_term = 25 * archive_bytes / ORIGINAL_VIDEO_BYTES
        segnet_term = 100 * result["segnet_dist"]
        posenet_term = math.sqrt(10 * result["posenet_dist"])

        print(f"  [{label}] score={result['score']:.2f} "
              f"(seg={segnet_term:.4f} pose={posenet_term:.4f} rate={rate_term:.4f}) "
              f"archive={archive_bytes:,}B masks={masks_mkv.stat().st_size:,}B "
              f"inflate={inflate_time:.1f}s")

        return result

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mask compression sweep")
    parser.add_argument("--device", default="mps", help="Torch device")
    parser.add_argument("--output", default="experiments/results/mask_sweep.json")
    parser.add_argument("--resolutions", nargs="+", type=str,
                        default=["384x512", "192x256", "96x128", "48x64"])
    parser.add_argument("--crfs", nargs="+", type=int,
                        default=[15, 20, 30, 40, 50, 63])
    args = parser.parse_args()

    renderer_bin = SUBMISSION_DIR / "renderer.bin"
    optimized_poses = SUBMISSION_DIR / "optimized_poses.pt"

    assert renderer_bin.exists(), f"renderer.bin not found: {renderer_bin}"
    assert optimized_poses.exists(), f"optimized_poses.pt not found: {optimized_poses}"

    # Extract fresh masks at native resolution
    print("Extracting fresh SegNet masks at 384x512 ...", flush=True)
    native_masks = extract_fresh_masks(args.device)
    print(f"  Got {native_masks.shape} masks", flush=True)

    results = []
    temp_dir = Path(tempfile.mkdtemp())

    try:
        for res_str in args.resolutions:
            h, w = map(int, res_str.split("x"))

            # Downscale masks
            masks = downscale_masks(native_masks, h, w)

            for crf in args.crfs:
                label = f"{h}x{w}_crf{crf}"
                print(f"\n=== {label} ===", flush=True)

                # Encode
                mkv_path = temp_dir / f"masks_{label}.mkv"
                try:
                    encode_masks_to_mkv(masks, mkv_path, crf=crf)
                except Exception as e:
                    print(f"  ENCODE FAILED: {e}", file=sys.stderr)
                    continue

                # Full e2e eval
                try:
                    result = run_single_candidate(
                        mkv_path, renderer_bin, optimized_poses,
                        label, args.device,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"  E2E FAILED: {e}", file=sys.stderr)
                    results.append({"label": label, "error": str(e)})

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n=== RESULTS SAVED: {output_path} ===")

        # Print summary table
        print(f"\n{'Label':<20} {'Score':>8} {'SegNet':>8} {'PoseNet':>8} {'Rate':>8} {'Masks':>10} {'Archive':>10}")
        print("-" * 85)
        for r in sorted(results, key=lambda x: x.get("score", 999)):
            if "error" in r:
                print(f"{r['label']:<20} ERROR: {r['error'][:50]}")
                continue
            rate_term = 25 * r["archive_bytes"] / ORIGINAL_VIDEO_BYTES
            seg_term = 100 * r["segnet_dist"]
            pose_term = math.sqrt(10 * r["posenet_dist"])
            print(f"{r['label']:<20} {r['score']:>8.2f} {seg_term:>8.4f} {pose_term:>8.4f} "
                  f"{rate_term:>8.4f} {r['masks_mkv_bytes']:>9,}B {r['archive_bytes']:>9,}B")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
