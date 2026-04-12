#!/usr/bin/env python
"""Diagnose PoseNet scoring divergence between local venv and upstream venv.

The SAME archive (md5 463b6fdb) scores:
  Auth:  PoseNet 0.00218, SegNet 0.00610, Score 1.33
  Local: PoseNet 0.06256, SegNet 0.00565, Score 1.93
  => PoseNet is 29x inflated locally. Why?

This script loads PoseNet + SegNet with the UPSTREAM weights, feeds
them ONE batch of frame pairs (ground truth vs compressed), and prints
the raw model outputs and per-pair distortions. Run with BOTH venvs
to isolate torch/timm version effects.

Usage:
  # Our venv (torch 2.11.0, timm 1.0.26)
  .venv/bin/python submissions/robust_current/diagnose_scorer.py \
    --upstream-dir workspace/upstream/comma_video_compression_challenge \
    --submission-dir /tmp/verify_133/submission

  # Upstream venv (torch 2.10.0, timm 1.0.22)
  workspace/upstream/comma_video_compression_challenge/.venv/bin/python \
    submissions/robust_current/diagnose_scorer.py \
    --upstream-dir workspace/upstream/comma_video_compression_challenge \
    --submission-dir /tmp/verify_133/submission
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import click
import torch
import numpy as np


def _add_upstream_to_path(upstream_dir: Path) -> None:
    """Add upstream dir to sys.path so we can import modules/frame_utils."""
    ud = str(upstream_dir.resolve())
    if ud not in sys.path:
        sys.path.insert(0, ud)


@click.command()
@click.option(
    "--upstream-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to upstream challenge repo.",
)
@click.option(
    "--submission-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to submission dir (with archive.zip and inflated/).",
)
@click.option("--device", default="cpu", help="Torch device.")
@click.option("--batch-size", default=16, help="Batch size for data loading.")
@click.option("--max-batches", default=3, help="Max batches to process (0=all).")
@click.option("--full-eval", is_flag=True, help="Run full evaluation over all batches.")
def diagnose(
    upstream_dir: str,
    submission_dir: str,
    device: str,
    batch_size: int,
    max_batches: int,
    full_eval: bool,
) -> None:
    """Diagnose PoseNet scoring divergence between venvs."""
    upstream = Path(upstream_dir)
    submission = Path(submission_dir)

    _add_upstream_to_path(upstream)

    # Import upstream modules
    import timm as timm_mod
    from modules import DistortionNet, segnet_sd_path, posenet_sd_path
    from frame_utils import (
        AVVideoDataset, TensorVideoDataset,
        camera_size, seq_len, rgb_to_yuv6,
    )

    click.echo("=" * 60)
    click.echo("  SCORER DIVERGENCE DIAGNOSTIC")
    click.echo("=" * 60)
    click.echo(f"  Python:     {sys.version.split()[0]}")
    click.echo(f"  torch:      {torch.__version__}")
    click.echo(f"  timm:       {timm_mod.__version__}")
    click.echo(f"  device:     {device}")
    click.echo(f"  upstream:   {upstream}")
    click.echo(f"  submission: {submission}")
    click.echo(f"  batch_size: {batch_size}")
    click.echo(f"  max_batches:{max_batches} (0=all)")
    click.echo("=" * 60)

    dev = torch.device(device)

    # Load models
    click.echo("\nLoading DistortionNet...")
    t0 = time.time()
    dnet = DistortionNet().eval().to(device=dev)
    dnet.load_state_dicts(posenet_sd_path, segnet_sd_path, dev)
    click.echo(f"  Loaded in {time.time() - t0:.1f}s")

    # Load video names
    video_names_file = upstream / "public_test_video_names.txt"
    test_video_names = [
        ln.strip() for ln in video_names_file.read_text().splitlines() if ln.strip()
    ]
    click.echo(f"  Videos: {test_video_names}")

    # Load ground truth
    click.echo("\nLoading ground truth video...")
    ds_gt = AVVideoDataset(
        test_video_names,
        data_dir=upstream / "videos",
        batch_size=batch_size,
        device=dev,
    )
    ds_gt.prepare_data()

    # Load compressed (inflated)
    # Check if inflated/ has .raw files
    inflated_dir = submission / "inflated"
    raw_files = list(inflated_dir.glob("*.raw"))
    if raw_files:
        click.echo(f"Loading compressed from .raw files: {[f.name for f in raw_files]}")
        ds_comp = TensorVideoDataset(
            test_video_names,
            data_dir=inflated_dir,
            batch_size=batch_size,
            device=dev,
        )
    else:
        click.echo(f"No .raw files in {inflated_dir}, trying AVVideoDataset...")
        # Maybe inflated has .hevc or .mkv files
        ds_comp = AVVideoDataset(
            test_video_names,
            data_dir=inflated_dir,
            batch_size=batch_size,
            device=dev,
        )
    ds_comp.prepare_data()

    dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)
    dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

    # Process batches
    click.echo("\n--- Per-batch distortions ---")
    all_pose_dists = []
    all_seg_dists = []
    total_pairs = 0
    batch_idx = 0

    with torch.inference_mode():
        for (_, _, batch_gt), (_, _, batch_comp) in zip(dl_gt, dl_comp):
            batch_gt = batch_gt.to(dev)
            batch_comp = batch_comp.to(dev)
            bs = batch_gt.shape[0]

            # Raw model outputs for first batch
            if batch_idx == 0:
                click.echo(f"\n  Batch 0 shape: gt={batch_gt.shape}, comp={batch_comp.shape}")
                click.echo(f"  Batch 0 dtype: gt={batch_gt.dtype}, comp={batch_comp.dtype}")
                click.echo(f"  Batch 0 range: gt=[{batch_gt.float().min():.0f}, {batch_gt.float().max():.0f}], "
                           f"comp=[{batch_comp.float().min():.0f}, {batch_comp.float().max():.0f}]")

                # Get raw PoseNet outputs for first pair
                posenet_out_gt, segnet_out_gt = dnet(batch_gt)
                posenet_out_comp, segnet_out_comp = dnet(batch_comp)

                # Print raw pose vectors for first sample
                click.echo(f"\n  --- Raw PoseNet output (sample 0, first 6 of 12) ---")
                p_gt = posenet_out_gt['pose'][0, :6].cpu().numpy()
                p_comp = posenet_out_comp['pose'][0, :6].cpu().numpy()
                click.echo(f"  GT:   {np.array2string(p_gt, precision=6, suppress_small=True)}")
                click.echo(f"  Comp: {np.array2string(p_comp, precision=6, suppress_small=True)}")
                click.echo(f"  Diff: {np.array2string(p_gt - p_comp, precision=6, suppress_small=True)}")
                click.echo(f"  MSE(first 6): {((p_gt - p_comp) ** 2).mean():.8f}")

                # SegNet argmax comparison
                seg_gt_argmax = segnet_out_gt.argmax(dim=1)[0].cpu()
                seg_comp_argmax = segnet_out_comp.argmax(dim=1)[0].cpu()
                seg_diff = (seg_gt_argmax != seg_comp_argmax).float().mean().item()
                click.echo(f"\n  --- SegNet (sample 0) ---")
                click.echo(f"  Argmax mismatch rate: {seg_diff:.6f}")

                # Per-pair distortions using the official method
                pose_dist = dnet.posenet.compute_distortion(posenet_out_gt, posenet_out_comp)
                seg_dist = dnet.segnet.compute_distortion(segnet_out_gt, segnet_out_comp)
            else:
                # Standard path
                pose_dist, seg_dist = dnet.compute_distortion(batch_gt, batch_comp)

            # Per-pair breakdown
            pose_per_pair = pose_dist.cpu().numpy()
            seg_per_pair = seg_dist.cpu().numpy()

            all_pose_dists.extend(pose_per_pair.tolist())
            all_seg_dists.extend(seg_per_pair.tolist())
            total_pairs += bs

            # Print per-pair for first few batches
            if batch_idx < 3:
                click.echo(f"\n  Batch {batch_idx} ({bs} pairs):")
                for i in range(min(bs, 4)):
                    click.echo(f"    pair {total_pairs - bs + i}: "
                               f"pose={pose_per_pair[i]:.8f}, seg={seg_per_pair[i]:.8f}")
                if bs > 4:
                    click.echo(f"    ... ({bs - 4} more)")
                click.echo(f"    batch mean: pose={pose_per_pair.mean():.8f}, seg={seg_per_pair.mean():.8f}")

            batch_idx += 1
            if not full_eval and max_batches > 0 and batch_idx >= max_batches:
                break

    # Summary
    pose_arr = np.array(all_pose_dists)
    seg_arr = np.array(all_seg_dists)
    mean_pose = pose_arr.mean()
    mean_seg = seg_arr.mean()

    click.echo("\n" + "=" * 60)
    click.echo(f"  SUMMARY ({total_pairs} pairs, {batch_idx} batches)")
    click.echo("=" * 60)
    click.echo(f"  PoseNet mean:   {mean_pose:.8f}")
    click.echo(f"  PoseNet std:    {pose_arr.std():.8f}")
    click.echo(f"  PoseNet min:    {pose_arr.min():.8f}")
    click.echo(f"  PoseNet max:    {pose_arr.max():.8f}")
    click.echo(f"  PoseNet median: {np.median(pose_arr):.8f}")
    click.echo(f"  SegNet mean:    {mean_seg:.8f}")
    click.echo(f"  SegNet std:     {seg_arr.std():.8f}")

    # Compute what the score would be (need rate for that)
    archive_zip = submission / "archive.zip"
    videos_dir = upstream / "videos"
    if archive_zip.exists():
        compressed_size = archive_zip.stat().st_size
        uncompressed_size = sum(
            f.stat().st_size for f in videos_dir.rglob("*") if f.is_file()
        )
        rate = compressed_size / uncompressed_size
    else:
        rate = 0.02302  # fallback known value

    if full_eval or max_batches == 0:
        score = 100 * mean_seg + math.sqrt(10 * mean_pose) + 25 * rate
        click.echo(f"\n  Rate:  {rate:.8f}")
        click.echo(f"  Score: 100*{mean_seg:.6f} + sqrt(10*{mean_pose:.6f}) + 25*{rate:.6f}")
        click.echo(f"       = {100*mean_seg:.4f} + {math.sqrt(10*mean_pose):.4f} + {25*rate:.4f}")
        click.echo(f"       = {score:.4f}")
    else:
        click.echo(f"\n  (Partial eval: {batch_idx}/{max_batches} batches. Use --full-eval for final score.)")
        score_est = 100 * mean_seg + math.sqrt(10 * mean_pose) + 25 * rate
        click.echo(f"  Estimated score: {score_est:.4f} (partial, NOT authoritative)")

    # Save per-pair data for analysis
    output = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "timm": timm_mod.__version__,
        "device": device,
        "total_pairs": total_pairs,
        "batches_processed": batch_idx,
        "full_eval": full_eval,
        "posenet_mean": float(mean_pose),
        "posenet_std": float(pose_arr.std()),
        "posenet_min": float(pose_arr.min()),
        "posenet_max": float(pose_arr.max()),
        "segnet_mean": float(mean_seg),
        "segnet_std": float(seg_arr.std()),
        "rate": float(rate),
        "per_pair_posenet": [float(x) for x in all_pose_dists],
        "per_pair_segnet": [float(x) for x in all_seg_dists],
    }
    out_path = Path(f"/tmp/diagnose_scorer_{torch.__version__}.json")
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    click.echo(f"\n  Per-pair data saved to: {out_path}")
    click.echo("=" * 60)


if __name__ == "__main__":
    diagnose()
