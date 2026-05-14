#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cross-device divergence diagnostic — Stream 1 [macOS-CPU advisory] + [MPS-research-signal].

Operator directive 2026-05-13 AGGRESSIVE LOCAL HARDWARE SWEEP.

Given an inflated archive (raw frames already produced by the archive's
own ``inflate.sh``) + the GT video, this tool runs the upstream
``DistortionNet.compute_distortion`` per pair on TWO devices (macOS-CPU,
MPS) and reports the divergence per pair + aggregate.

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: MPS output is
``[MPS-research-signal]`` permanently (score_claim=False,
promotion_eligible=False, ready_for_exact_eval_dispatch=False).

Per CLAUDE.md "Apples-to-apples evidence discipline": the cross-device
comparison is per-pair (B=1) — NOT a substitute for full-batch contest
eval. The intent is to detect substrate-specific cross-device divergence
as a PRE-DISPATCH bug detector: substrates whose CPU↔MPS Δscore exceeds
the documented A1 baseline drift band (~0.034 on A1) are likely buggy.

Output: JSON with per-pair (CPU pose, CPU seg, MPS pose, MPS seg) +
aggregate divergence summary.

Usage:
    .venv/bin/python tools/cross_device_divergence_diagnostic.py \\
        --archive-dir <inflated raw frames dir> \\
        --uncompressed-dir upstream/videos \\
        --video-names-file upstream/public_test_video_names.txt \\
        --output-json <path> \\
        --n-pairs 50  # restrict for speed; default 600 = full eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "upstream"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-dir", type=Path, required=True,
                   help="Directory containing inflated <name>.raw files.")
    p.add_argument("--uncompressed-dir", type=Path, required=True,
                   help="Directory containing GT <name>.mkv files.")
    p.add_argument("--video-names-file", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--n-pairs", type=int, default=600,
                   help="Number of pairs to evaluate per video.")
    p.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Per-device batch size for the seq_len=2 pair stream.")
    args = p.parse_args()

    from frame_utils import AVVideoDataset, TensorVideoDataset, camera_size, seq_len
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    video_names = [
        line.strip()
        for line in args.video_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results: dict = {
        "schema": "cross_device_divergence_diagnostic_v1",
        "archive_dir": str(args.archive_dir),
        "uncompressed_dir": str(args.uncompressed_dir),
        "video_names": video_names,
        "n_pairs": args.n_pairs,
        "devices": args.devices,
        "per_device_results": {},
        "evidence_tags": {
            "cpu": "[macOS-CPU advisory]",
            "mps": "[MPS-research-signal]",
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    # Re-score on each device. We process the same N pairs in the same order
    # by using a fixed seed and batch_size; this is the canonical contest path
    # (zip(ds_gt, ds_comp)).
    for device_str in args.devices:
        if device_str == "mps" and not torch.backends.mps.is_available():
            print(f"[{device_str}] MPS not available; skipping", flush=True)
            results["per_device_results"][device_str] = {"error": "mps_unavailable"}
            continue

        device = torch.device(device_str)
        torch.manual_seed(args.seed)
        print(f"\n=== Device: {device_str} ===", flush=True)
        t0 = time.time()

        try:
            distortion_net = DistortionNet().eval().to(device=device)
            distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

            ds_gt = AVVideoDataset(
                video_names, data_dir=args.uncompressed_dir,
                batch_size=args.batch_size, device=device,
                num_threads=2, seed=args.seed, prefetch_queue_depth=4,
            )
            ds_gt.prepare_data()
            dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)

            ds_comp = TensorVideoDataset(
                video_names, data_dir=args.archive_dir,
                batch_size=args.batch_size, device=device,
                num_threads=2, seed=args.seed, prefetch_queue_depth=4,
            )
            ds_comp.prepare_data()
            dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

            pose_dists_all: list[float] = []
            seg_dists_all: list[float] = []
            pair_count = 0

            with torch.inference_mode():
                for (_, _, batch_gt), (_, _, batch_comp) in zip(dl_gt, dl_comp, strict=False):
                    batch_gt = batch_gt.to(device)
                    batch_comp = batch_comp.to(device)
                    assert list(batch_comp.shape)[1:] == [seq_len, camera_size[1], camera_size[0], 3], \
                        f"unexpected batch shape: {batch_comp.shape}"
                    pose_d, seg_d = distortion_net.compute_distortion(batch_gt, batch_comp)
                    # Sync MPS before transfer to CPU to ensure accurate timing
                    if device_str == "mps":
                        torch.mps.synchronize()
                    pose_dists_all.extend(pose_d.detach().cpu().float().tolist())
                    seg_dists_all.extend(seg_d.detach().cpu().float().tolist())
                    pair_count += int(batch_gt.shape[0])
                    if pair_count >= args.n_pairs:
                        break

            # Trim to exactly n_pairs (cross-device alignment)
            pose_dists_all = pose_dists_all[:args.n_pairs]
            seg_dists_all = seg_dists_all[:args.n_pairs]

            avg_pose = sum(pose_dists_all) / len(pose_dists_all)
            avg_seg = sum(seg_dists_all) / len(seg_dists_all)

            elapsed = time.time() - t0
            print(
                f"[{device_str}] n_pairs={len(pose_dists_all)} "
                f"avg_pose={avg_pose:.6e} avg_seg={avg_seg:.6e} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

            results["per_device_results"][device_str] = {
                "device": device_str,
                "n_pairs": len(pose_dists_all),
                "avg_posenet_dist": avg_pose,
                "avg_segnet_dist": avg_seg,
                "per_pair_pose": pose_dists_all,
                "per_pair_seg": seg_dists_all,
                "elapsed_seconds": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - t0
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[{device_str}] ERROR: {err_msg}", flush=True)
            results["per_device_results"][device_str] = {
                "device": device_str,
                "error": err_msg,
                "elapsed_seconds": elapsed,
            }

    # Cross-device divergence summary
    if all(d in results["per_device_results"] and "avg_posenet_dist" in results["per_device_results"][d]
           for d in args.devices):
        cpu_r = results["per_device_results"]["cpu"]
        mps_r = results["per_device_results"].get("mps", {})
        if "avg_posenet_dist" in mps_r:
            d_pose_avg = mps_r["avg_posenet_dist"] - cpu_r["avg_posenet_dist"]
            d_seg_avg = mps_r["avg_segnet_dist"] - cpu_r["avg_segnet_dist"]
            # Score function: 100*seg + sqrt(10*pose) + 25*rate (rate same for both)
            import math
            cpu_score_components = (
                100 * cpu_r["avg_segnet_dist"]
                + math.sqrt(10 * cpu_r["avg_posenet_dist"])
            )
            mps_score_components = (
                100 * mps_r["avg_segnet_dist"]
                + math.sqrt(10 * mps_r["avg_posenet_dist"])
            )
            results["divergence_summary"] = {
                "delta_avg_pose_mps_minus_cpu": d_pose_avg,
                "delta_avg_seg_mps_minus_cpu": d_seg_avg,
                "cpu_score_no_rate": cpu_score_components,
                "mps_score_no_rate": mps_score_components,
                "delta_score_no_rate_mps_minus_cpu": mps_score_components - cpu_score_components,
                "ratio_pose_mps_over_cpu": (
                    mps_r["avg_posenet_dist"] / cpu_r["avg_posenet_dist"]
                    if cpu_r["avg_posenet_dist"] > 0 else None
                ),
                "ratio_seg_mps_over_cpu": (
                    mps_r["avg_segnet_dist"] / cpu_r["avg_segnet_dist"]
                    if cpu_r["avg_segnet_dist"] > 0 else None
                ),
            }
            print("\n=== Divergence summary ===")
            print(json.dumps(results["divergence_summary"], indent=2))

    # Strip per-pair from JSON if too large; keep aggregate
    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
