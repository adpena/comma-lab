#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Mask-Conditioned SIREN Experiment — the dinner conversation turned into code.

A SIREN network whose weights ARE the compressed video, conditioned on
segmentation masks so the network allocates capacity per-class: sky gets
smooth blue (few parameters), road gets textured gray (more parameters),
class boundaries stay sharp (most parameters).

Three-phase training:
1. Memorize: MSE pixel reconstruction at scorer resolution (512x384)
2. Constrain: add Fridrich-style scorer losses (PoseNet + SegNet)
3. Fine-tune: scorer-only loss with reduced LR

Archive = masks (zlib-compressed, ~300B) + SIREN weights (fp16, ~200KB)
Rate = 25 * archive_KB / 37500KB

Run on Lightning T4:
    PYTHONPATH=src:/home/zeus/content/upstream python experiments/siren_mask_experiment.py --device cuda

Local CPU smoke test (10 frames, 50 steps):
    PYTHONPATH=src python experiments/siren_mask_experiment.py --smoke --device cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_CANDIDATE_UPSTREAM = [
    Path("/home/zeus/content/upstream"),
    Path(__file__).resolve().parent.parent / "upstream",
    Path(os.environ.get("UPSTREAM_ROOT", "")) if os.environ.get("UPSTREAM_ROOT") else None,
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break

if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

_CANDIDATE_WEIGHTS = [
    Path("/home/zeus/content/upstream/models"),
    Path("/home/zeus/content/pact/upstream/models"),
    Path(__file__).resolve().parent.parent / "upstream" / "models",
]
WEIGHTS_DIR: Path | None = None
for _p in _CANDIDATE_WEIGHTS:
    if _p is not None and (_p / "posenet.safetensors").exists():
        WEIGHTS_DIR = _p
        break

_CANDIDATE_GT = [
    Path("/home/zeus/content/upstream/videos/0.mkv"),
    Path(__file__).resolve().parent.parent / "upstream" / "videos" / "0.mkv",
]
GT_VIDEO: Path | None = None
for _p in _CANDIDATE_GT:
    if _p is not None and _p.exists():
        GT_VIDEO = _p
        break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scorers(device: str) -> tuple[nn.Module, nn.Module]:
    """Load frozen PoseNet + SegNet."""
    if WEIGHTS_DIR is None:
        raise FileNotFoundError(
            "Scorer weights not found. Searched: " +
            ", ".join(str(p) for p in _CANDIDATE_WEIGHTS)
        )
    from tac.scorer import load_scorers
    return load_scorers(
        WEIGHTS_DIR / "posenet.safetensors",
        WEIGHTS_DIR / "segnet.safetensors",
        device=device,
        upstream_dir=UPSTREAM_ROOT,
    )


def _load_gt_frames(n_frames: int, target_h: int = 874, target_w: int = 1164) -> list[torch.Tensor]:
    """Load n ground truth frames as (H, W, 3) uint8 tensors at native resolution."""
    if GT_VIDEO is None:
        raise FileNotFoundError(
            "GT video not found. Searched: " +
            ", ".join(str(p) for p in _CANDIDATE_GT)
        )
    from tac.data import decode_video
    frames = decode_video(str(GT_VIDEO), target_h=target_h, target_w=target_w)
    return frames[:n_frames]


def _resize_frames_to_scorer(frames: list[torch.Tensor], h: int = 384, w: int = 512) -> torch.Tensor:
    """Resize (H, W, 3) frames to scorer resolution, return (N, H, W, 3) uint8."""
    out = []
    for f in frames:
        chw = f.float().permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        resized = F.interpolate(chw, size=(h, w), mode="bilinear", align_corners=False)
        out.append(resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte())
    return torch.stack(out)


def _extract_masks(
    frames_scorer: torch.Tensor,
    segnet: nn.Module,
    device: str,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract segmentation masks from frames using SegNet.

    Args:
        frames_scorer: (N, H, W, 3) uint8 at scorer resolution.
        segnet: frozen SegNet model.
        device: compute device.
        batch_size: frames per forward pass.

    Returns:
        (N, H, W) long tensor with class indices.
    """
    N, H, W, C = frames_scorer.shape
    masks = torch.zeros(N, H, W, dtype=torch.long)

    for i in range(0, N, batch_size):
        batch = frames_scorer[i:i + batch_size].float().to(device)
        # SegNet expects (B, T=1, C, H, W)
        batch_btchw = batch.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
        with torch.no_grad():
            seg_in = segnet.preprocess_input(batch_btchw)
            seg_out = segnet(seg_in)
        # seg_out shape varies: (B, H_out, W_out, num_classes) or similar
        if seg_out.ndim == 4 and seg_out.shape[-1] > seg_out.shape[1]:
            # (B, H, W, C) format
            seg_labels = seg_out.argmax(dim=-1)
        elif seg_out.ndim == 4:
            # (B, C, H, W) format
            seg_labels = seg_out.argmax(dim=1)
        else:
            seg_labels = seg_out.argmax(dim=-1)

        # Resize to scorer resolution if needed
        sh = seg_labels.shape[-2]
        sw = seg_labels.shape[-1]
        if sh != H or sw != W:
            seg_labels = F.interpolate(
                seg_labels.float().unsqueeze(1), size=(H, W), mode="nearest",
            ).squeeze(1).long()

        masks[i:i + batch_size] = seg_labels.cpu()

    return masks


def _extract_gt_poses(
    frames_scorer: torch.Tensor,
    posenet: nn.Module,
    device: str,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract GT PoseNet targets from consecutive frame pairs.

    Args:
        frames_scorer: (N, H, W, 3) uint8 at scorer resolution.
        posenet: frozen PoseNet model.
        device: compute device.
        batch_size: pairs per forward pass.

    Returns:
        (N-1, 6) float tensor of pose outputs.
    """
    N = frames_scorer.shape[0]
    poses = []

    for i in range(0, N - 1, batch_size):
        end = min(i + batch_size, N - 1)
        frame1 = frames_scorer[i:end].float().to(device)
        frame2 = frames_scorer[i + 1:end + 1].float().to(device)
        # Build (B, 2, C, H, W) pairs
        pairs = torch.stack([
            frame1.permute(0, 3, 1, 2),
            frame2.permute(0, 3, 1, 2),
        ], dim=1).contiguous()

        with torch.no_grad():
            pose_in = posenet.preprocess_input(pairs)
            pose_out = posenet(pose_in)
            pose = pose_out["pose"][..., :6] if isinstance(pose_out, dict) else pose_out[..., :6]
        poses.append(pose.cpu())

    return torch.cat(poses, dim=0)


def _score_frames(
    gen_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    device: str,
    batch_size: int = 4,
) -> dict[str, float]:
    """Score generated frames against GT using official scorer formula.

    Returns dict with seg_dist, pose_dist, and component scores.
    """
    N = gen_frames.shape[0]

    # SegNet: per-frame comparison
    seg_dists = []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        gen_batch = gen_frames[i:end].float().to(device)
        gt_batch = gt_frames[i:end].float().to(device)

        gen_btchw = gen_batch.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
        gt_btchw = gt_batch.permute(0, 3, 1, 2).unsqueeze(1).contiguous()

        with torch.no_grad():
            gen_seg_in = segnet.preprocess_input(gen_btchw)
            gen_seg_out = segnet(gen_seg_in)
            gt_seg_in = segnet.preprocess_input(gt_btchw)
            gt_seg_out = segnet(gt_seg_in)

        gen_labels = gen_seg_out.argmax(dim=-1) if gen_seg_out.shape[-1] < gen_seg_out.shape[1] or gen_seg_out.ndim == 3 else gen_seg_out.argmax(dim=1 if gen_seg_out.ndim == 4 and gen_seg_out.shape[1] < gen_seg_out.shape[-1] else -1)
        gt_labels = gt_seg_out.argmax(dim=-1) if gt_seg_out.shape[-1] < gt_seg_out.shape[1] or gt_seg_out.ndim == 3 else gt_seg_out.argmax(dim=1 if gt_seg_out.ndim == 4 and gt_seg_out.shape[1] < gt_seg_out.shape[-1] else -1)

        disagreement = (gen_labels != gt_labels).float().mean(dim=(-2, -1))
        seg_dists.append(disagreement.cpu())

    seg_dist = torch.cat(seg_dists).mean().item()

    # PoseNet: consecutive pair comparison
    pose_dists = []
    for i in range(0, N - 1, batch_size):
        end = min(i + batch_size, N - 1)
        gen1 = gen_frames[i:end].float().to(device)
        gen2 = gen_frames[i + 1:end + 1].float().to(device)
        gt1 = gt_frames[i:end].float().to(device)
        gt2 = gt_frames[i + 1:end + 1].float().to(device)

        gen_pairs = torch.stack([gen1.permute(0, 3, 1, 2), gen2.permute(0, 3, 1, 2)], dim=1).contiguous()
        gt_pairs = torch.stack([gt1.permute(0, 3, 1, 2), gt2.permute(0, 3, 1, 2)], dim=1).contiguous()

        with torch.no_grad():
            gen_pose_in = posenet.preprocess_input(gen_pairs)
            gen_pose_out = posenet(gen_pose_in)
            gen_pose = gen_pose_out["pose"][..., :6] if isinstance(gen_pose_out, dict) else gen_pose_out[..., :6]

            gt_pose_in = posenet.preprocess_input(gt_pairs)
            gt_pose_out = posenet(gt_pose_in)
            gt_pose = gt_pose_out["pose"][..., :6] if isinstance(gt_pose_out, dict) else gt_pose_out[..., :6]

        pair_dist = (gen_pose - gt_pose).pow(2).mean(dim=-1)
        pose_dists.append(pair_dist.cpu())

    pose_dist = torch.cat(pose_dists).mean().item()

    return {
        "seg_dist": seg_dist,
        "pose_dist": pose_dist,
    }


def test_bilinear_roundtrip(
    gt_frames_native: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    device: str,
    n_test: int = 10,
) -> dict[str, float]:
    """Contrarian's requirement: verify bilinear round-trip is scorer-invisible.

    Load GT at 1164x874 -> downscale to 512x384 -> upscale back to 1164x874.
    Score round-tripped vs original. If distortion < 0.01, the round-trip is safe.
    """
    print("\n=== Bilinear Round-Trip Test (Contrarian's Check) ===")
    frames = gt_frames_native[:n_test]
    native_stack = torch.stack(frames)  # (N, 874, 1164, 3)

    # Downscale to scorer resolution
    native_chw = native_stack.float().permute(0, 3, 1, 2)  # (N, 3, 874, 1164)
    scorer_res = F.interpolate(native_chw, size=(384, 512), mode="bilinear", align_corners=False)

    # Upscale back
    roundtripped = F.interpolate(scorer_res, size=(874, 1164), mode="bilinear", align_corners=False)
    roundtripped_hwc = roundtripped.permute(0, 2, 3, 1).clamp(0, 255).byte()

    # Score at native resolution (pass scorer-res frames since that's what scorer sees)
    scorer_original = scorer_res.permute(0, 2, 3, 1).clamp(0, 255).byte()
    scorer_roundtripped = F.interpolate(roundtripped, size=(384, 512), mode="bilinear", align_corners=False)
    scorer_roundtripped_hwc = scorer_roundtripped.permute(0, 2, 3, 1).clamp(0, 255).byte()

    result = _score_frames(scorer_roundtripped_hwc, scorer_original, posenet, segnet, device)

    pixel_mse = (native_stack.float() - roundtripped_hwc.float()).pow(2).mean().item()
    psnr = 10 * math.log10(255 ** 2 / max(pixel_mse, 1e-10))

    print(f"  Round-trip PSNR: {psnr:.1f} dB")
    print(f"  Scorer seg_dist: {result['seg_dist']:.6f}")
    print(f"  Scorer pose_dist: {result['pose_dist']:.6f}")

    safe = result["seg_dist"] < 0.01 and result["pose_dist"] < 0.01
    print(f"  Verdict: {'SAFE' if safe else 'UNSAFE'} for scorer")
    result["psnr"] = psnr
    result["safe"] = safe
    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full mask-conditioned SIREN experiment."""
    device = args.device
    results: dict[str, Any] = {
        "experiment": "mask_conditioned_siren",
        "timestamp": time.strftime("%Y%m%dT%H%M%SZ"),
        "device": device,
        "args": vars(args),
    }

    print("=" * 70)
    print("Mask-Conditioned SIREN Experiment")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ── Step 0: Load GT video ──
    n_frames = args.n_frames
    print(f"\n[1/8] Loading {n_frames} GT frames...")
    t0 = time.time()
    gt_frames_native = _load_gt_frames(n_frames)
    print(f"  Loaded {len(gt_frames_native)} frames at {gt_frames_native[0].shape} in {time.time()-t0:.1f}s")

    # Resize to scorer resolution
    print(f"\n[2/8] Resizing to scorer resolution (512x384)...")
    gt_scorer = _resize_frames_to_scorer(gt_frames_native)
    print(f"  Shape: {gt_scorer.shape}")
    results["gt_shape"] = list(gt_scorer.shape)

    # ── Step 1: Load scorers ──
    print(f"\n[3/8] Loading scorers...")
    t0 = time.time()
    posenet, segnet = _load_scorers(device)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Step 1.5: Bilinear round-trip test ──
    if not args.skip_roundtrip:
        rt_result = test_bilinear_roundtrip(gt_frames_native, posenet, segnet, device)
        results["roundtrip_test"] = rt_result
        if not rt_result["safe"]:
            print("  WARNING: Bilinear round-trip introduces scorer-visible artifacts!")

    # ── Step 2: Extract masks ──
    print(f"\n[4/8] Extracting masks from GT frames...")
    t0 = time.time()
    masks = _extract_masks(gt_scorer, segnet, device)
    print(f"  Masks shape: {masks.shape}, classes: {masks.unique().tolist()}")
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # Mask statistics
    for c in range(5):
        pct = (masks == c).float().mean().item() * 100
        print(f"    Class {c}: {pct:.1f}%")

    # ── Step 3: Extract GT PoseNet targets ──
    print(f"\n[5/8] Extracting GT PoseNet targets...")
    t0 = time.time()
    gt_poses = _extract_gt_poses(gt_scorer, posenet, device)
    print(f"  Poses shape: {gt_poses.shape}")
    print(f"  Extracted in {time.time()-t0:.1f}s")
    results["gt_poses_mean"] = gt_poses.mean(dim=0).tolist()

    # ── Step 4: Train MaskConditionedSIREN ──
    print(f"\n[6/8] Training MaskConditionedSIREN...")
    print(f"  hidden={args.hidden}, layers={args.layers}, omega_0={args.omega_0}")
    print(f"  steps={args.num_steps}, batch_pixels={args.batch_pixels}")

    from tac.network_codec import MaskConditionedSIREN, train_mask_conditioned_siren

    t0 = time.time()
    model = train_mask_conditioned_siren(
        gt_scorer,
        masks,
        posenet=posenet if not args.pixel_only else None,
        segnet=segnet if not args.pixel_only else None,
        hidden=args.hidden,
        layers=args.layers,
        omega_0=args.omega_0,
        num_classes=5,
        pos_encoding_freqs=args.pos_freqs,
        num_steps=args.num_steps,
        batch_pixels=args.batch_pixels,
        lr=args.lr,
        device=device,
        log_every=args.log_every,
        scorer_ramp_start=args.scorer_ramp_start,
        scorer_weight_max=args.scorer_weight_max,
        tv_weight=args.tv_weight,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
    results["train_time_s"] = train_time
    results["param_count"] = model.param_count()
    results["size_fp16_kb"] = model.size_bytes_fp16() / 1024

    # ── Step 5: Export archive ──
    print(f"\n[7/8] Exporting archive...")
    from tac.network_codec import export_mask_siren_archive, inflate_mask_siren_archive

    archive = export_mask_siren_archive(model, masks, use_fp16=True)
    archive_kb = len(archive) / 1024
    print(f"  Archive size: {len(archive)} bytes ({archive_kb:.1f} KB)")
    results["archive_bytes"] = len(archive)
    results["archive_kb"] = archive_kb

    # Rate calculation: rate = 25 * archive_bytes / gt_video_bytes
    gt_video_bytes = n_frames * 1164 * 874 * 3 / 2  # YUV420 size
    rate = 25.0 * len(archive) / gt_video_bytes
    print(f"  Rate component: {rate:.4f}")
    results["rate"] = rate

    # ── Step 6: Inflate and score ──
    print(f"\n[8/8] Inflating and scoring...")
    t0 = time.time()
    inflated = inflate_mask_siren_archive(archive, device=device)
    inflate_time = time.time() - t0
    print(f"  Inflate time: {inflate_time:.1f}s")
    print(f"  Inflated shape: {inflated.shape}")
    results["inflate_time_s"] = inflate_time

    # Pixel quality vs GT at scorer resolution
    pixel_mse = (inflated.float() - gt_scorer.float()).pow(2).mean().item()
    psnr = 10 * math.log10(255 ** 2 / max(pixel_mse, 1e-10))
    print(f"  Reconstruction PSNR: {psnr:.1f} dB")
    results["psnr"] = psnr

    # Score with scorers
    print(f"\n  Scoring with scorers...")
    score_result = _score_frames(inflated, gt_scorer, posenet, segnet, device)
    seg_dist = score_result["seg_dist"]
    pose_dist = score_result["pose_dist"]

    # Competition score formula: S = 100*seg + sqrt(10*pose) + 25*rate
    seg_component = 100.0 * seg_dist
    pose_component = math.sqrt(10.0 * pose_dist)
    rate_component = rate
    total_score = seg_component + pose_component + rate_component

    print(f"\n  === SCORE BREAKDOWN ===")
    print(f"  SegNet distortion:  {seg_dist:.6f}  -> 100*seg = {seg_component:.4f}")
    print(f"  PoseNet distortion: {pose_dist:.6f}  -> sqrt(10*pose) = {pose_component:.4f}")
    print(f"  Rate:               {rate:.6f}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL SCORE:        {total_score:.4f}")
    print(f"  (Current best: 1.33, Target: <0.60)")

    results["seg_dist"] = seg_dist
    results["pose_dist"] = pose_dist
    results["seg_component"] = seg_component
    results["pose_component"] = pose_component
    results["rate_component"] = rate_component
    results["total_score"] = total_score

    # ── Contrarian's capacity check ──
    compression_ratio = (n_frames * 512 * 384 * 3) / len(archive)
    print(f"\n  Compression ratio: {compression_ratio:.0f}:1")
    print(f"  (707M values from {model.param_count():,} params = {compression_ratio:.0f}:1)")
    results["compression_ratio"] = compression_ratio

    # Test mask conditioning vs unconditional (ablation)
    if args.ablation:
        print(f"\n  === ABLATION: mask conditioning effect ===")
        # Generate frames with uniform mask (all zeros) to see effect
        uniform_masks = torch.zeros_like(masks)
        model_cpu = model.cpu().eval()
        with torch.no_grad():
            frame_cond = model_cpu.generate_frame(0, masks[0].cpu(), device="cpu").float()
            frame_uncond = model_cpu.generate_frame(0, uniform_masks[0].cpu(), device="cpu").float()
        diff = (frame_cond - frame_uncond).abs().mean().item()
        print(f"  Mean pixel diff (conditioned vs uniform mask): {diff:.2f}")
        results["ablation_mask_diff"] = diff
        model = model.to(device)

    # ── Save results ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"siren_mask_results_{results['timestamp']}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Save archive
    archive_path = output_dir / "siren_mask_archive.bin"
    with open(archive_path, "wb") as f:
        f.write(archive)
    print(f"  Archive saved: {archive_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mask-Conditioned SIREN Experiment")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-frames", type=int, default=1200, help="Number of GT frames to use")
    parser.add_argument("--hidden", type=int, default=128, help="SIREN hidden width")
    parser.add_argument("--layers", type=int, default=5, help="Number of SIREN hidden layers")
    parser.add_argument("--omega-0", type=float, default=30.0, help="SIREN omega_0")
    parser.add_argument("--pos-freqs", type=int, default=8, help="Positional encoding frequencies")
    parser.add_argument("--num-steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch-pixels", type=int, default=8192, help="Pixels per batch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--scorer-ramp-start", type=float, default=0.6, help="Fraction before scorer loss")
    parser.add_argument("--scorer-weight-max", type=float, default=10.0, help="Max scorer weight")
    parser.add_argument("--tv-weight", type=float, default=0.01, help="Total variation weight")
    parser.add_argument("--log-every", type=int, default=100, help="Log interval")
    parser.add_argument("--pixel-only", action="store_true", help="Skip scorer loss (Phase 1 only)")
    parser.add_argument("--skip-roundtrip", action="store_true", help="Skip bilinear round-trip test")
    parser.add_argument("--ablation", action="store_true", help="Run mask conditioning ablation")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (10 frames, 50 steps)")
    parser.add_argument("--output-dir", default="experiments/results/siren_mask", help="Output directory")
    args = parser.parse_args()

    if args.smoke:
        args.n_frames = 10
        args.num_steps = 50
        args.batch_pixels = 1024
        args.log_every = 10
        args.hidden = 32
        args.layers = 3

    try:
        results = run_experiment(args)
        print(f"\nExperiment complete. Total score: {results['total_score']:.4f}")
    except Exception as e:
        print(f"\nExperiment FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
