#!/usr/bin/env python3
"""FP4 Archive Rate Optimization: quantize distilled renderer to 4-bit.

Quantizes the distilled renderer to FP4 (4-bit floating point) with custom
codebook, measuring quality loss vs FP16 on the proxy scorer.

The competitor (mask2mask, score 0.60) uses this exact scheme:
    - 8-value codebook: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    - Per-block scaling (block_size=32)
    - 4 bits per weight → ~143KB for 287K params

This experiment:
    1. Post-training quantization (PTQ): quantize and measure quality
    2. Quality-aware quantization: find optimal block size per layer
    3. Rate-quality tradeoff: sweep codebook and measure archive size vs score
    4. QAT fine-tuning: brief STE-based training to recover quality

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/quantize_distilled.py \
        --device mps --smoke

    # Full analysis (all quantization levels):
    PYTHONPATH=src:upstream python experiments/quantize_distilled.py \
        --device cuda \
        --checkpoint experiments/results/v5_lagrangian_renderer/renderer_best.pt

    # QAT fine-tuning after PTQ:
    PYTHONPATH=src:upstream python experiments/quantize_distilled.py \
        --device cuda --qat --qat-epochs 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_CANDIDATE_UPSTREAM = [
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
    Path("/kaggle/working/upstream"),
    Path(__file__).resolve().parent.parent / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

RESULTS_DIR = (
    Path(os.environ["TAC_RESULTS_DIR"])
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "quantization"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FP4 quantization analysis and rate optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to distilled renderer checkpoint")
    p.add_argument("--block-sizes", type=str, default="16,32,64,128",
                   help="Comma-separated block sizes to sweep")
    p.add_argument("--qat", action="store_true",
                   help="Run quantization-aware fine-tuning after PTQ analysis")
    p.add_argument("--qat-epochs", type=int, default=500,
                   help="QAT fine-tuning epochs")
    p.add_argument("--qat-lr", type=float, default=0.0003,
                   help="QAT learning rate (lower than initial training)")
    p.add_argument("--batch-size", type=int, default=8, help="Pairs per batch for eval")
    p.add_argument("--simulate-resize", action="store_true",
                   help="Apply eval roundtrip in scoring")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames, fewer sweeps")
    return p.parse_args()


def compute_archive_size(state_dict: dict, block_size: int = 32) -> int:
    """Compute archive size for FP4 quantized model.

    FP4 layout per block:
        - 4 bits per weight (block_size weights)
        - 1 fp16 scale per block (2 bytes)
        - 1 bit per weight for sign (stored in packed byte)
    Total per block: block_size * 4/8 + 2 + block_size/8 bytes
                   = block_size * 0.5 + 2 + block_size * 0.125 bytes
                   = block_size * 0.625 + 2 bytes

    Args:
        state_dict: model state dict
        block_size: quantization block size

    Returns:
        Total archive size in bytes
    """
    total_params = 0
    for k, v in state_dict.items():
        if "weight" in k and v.ndim >= 2:
            total_params += v.numel()

    n_blocks = (total_params + block_size - 1) // block_size
    # 4 bits per weight + 16-bit scale per block
    weight_bytes = (total_params * 4 + 7) // 8  # 4-bit packed
    scale_bytes = n_blocks * 2  # fp16 scales
    sign_bytes = (total_params + 7) // 8  # 1 bit per weight

    return weight_bytes + scale_bytes + sign_bytes


def evaluate_model(
    model: nn.Module,
    gt_frames: list[torch.Tensor],
    gt_masks: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    simulate_resize: bool = False,
    max_pairs: int | None = None,
) -> dict[str, float]:
    """Evaluate model quality using proxy scorer.

    Returns:
        Dict with mean pose_dist, seg_dist, and proxy_score.
    """
    from tac.losses import scorer_forward_pair

    n_pairs = len(gt_frames) // 2
    if max_pairs:
        n_pairs = min(n_pairs, max_pairs)

    total_pose = 0.0
    total_seg = 0.0

    model.eval()
    with torch.no_grad():
        for i in range(n_pairs):
            m0 = gt_masks[i * 2].to(device)
            m1 = gt_masks[i * 2 + 1].to(device)
            masks = torch.stack([m0, m1], dim=0)

            rgb = model(masks)  # (2, 3, H, W)

            if simulate_resize:
                _, _, H, W = rgb.shape
                small = F.interpolate(rgb, size=(384, 512), mode="bilinear", align_corners=False)
                rgb = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)

            pair_chw = rgb.unsqueeze(0)
            fp_out, fs_out = scorer_forward_pair(pair_chw, posenet, segnet)

            # GT scorer outputs
            f0 = gt_frames[i * 2].float().to(device)
            f1 = gt_frames[i * 2 + 1].float().to(device)
            gt_pair = torch.stack([f0, f1], dim=0).unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()
            gp_out, gs_out = scorer_forward_pair(gt_pair, posenet, segnet)

            pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
            pred_soft = F.softmax(fs_out, dim=1)
            gt_soft = F.softmax(gs_out, dim=1)
            seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

            total_pose += pose_dist.item()
            total_seg += seg_dist.item()

    mean_pose = total_pose / n_pairs
    mean_seg = total_seg / n_pairs
    # Proxy score formula: 100 * seg + sqrt(10 * pose)
    proxy_score = 100.0 * mean_seg + (10.0 * mean_pose) ** 0.5

    return {
        "mean_pose_dist": mean_pose,
        "mean_seg_dist": mean_seg,
        "proxy_score": proxy_score,
        "n_pairs_evaluated": n_pairs,
    }


def main() -> None:
    args = parse_args()

    if args.smoke:
        n_frames = 20
        block_sizes = [32, 64]
    else:
        n_frames = 1200
        block_sizes = [int(x) for x in args.block_sizes.split(",")]

    device = torch.device(args.device)

    # Resolve paths
    from tac.utils import find_project_root
    root = find_project_root()
    upstream = Path(args.upstream) if args.upstream else root / "upstream"
    video_path = Path(args.video) if args.video else upstream / "videos" / "0.mkv"
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        root / "experiments" / "results" / "v5_lagrangian_renderer" / "renderer_best.pt"
    )

    # Verify checkpoint
    from tac.checkpoint import verify_checkpoint_identity
    verify_checkpoint_identity(str(checkpoint))

    # Load renderer
    from tac.renderer import MaskRenderer
    print(f"[quantize] Loading renderer from {checkpoint}")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        renderer_state = state["model_state_dict"]
        config = state.get("config", {})
    else:
        renderer_state = state
        config = {}

    model = MaskRenderer(
        embed_dim=config.get("embed_dim", 6),
        base_ch=config.get("base_ch", 36),
        mid_ch=config.get("mid_ch", 60),
        depth=config.get("depth", 1),
        pose_dim=config.get("pose_dim", 0),
    )
    model.load_state_dict(renderer_state, strict=False)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[quantize] Model: {n_params} params")

    # Load scorers
    from tac.scorer import load_differentiable_scorers, extract_gt_masks
    posenet, segnet = load_differentiable_scorers(str(upstream), device=device)

    # Decode video and extract masks
    from tac.data import decode_video
    print(f"[quantize] Decoding video: {video_path}")
    gt_frames = decode_video(str(video_path))[:n_frames]
    print(f"[quantize] Extracting SegNet masks...")
    gt_masks = extract_gt_masks(gt_frames, segnet, device=device)

    # Baseline: FP32 quality
    print(f"\n[quantize] === BASELINE (FP32) ===")
    baseline = evaluate_model(
        model, gt_frames, gt_masks, posenet, segnet, device,
        simulate_resize=args.simulate_resize,
        max_pairs=min(len(gt_frames) // 2, 50) if args.smoke else None,
    )
    print(f"  Proxy score: {baseline['proxy_score']:.4f}")
    print(f"  PoseNet: {baseline['mean_pose_dist']:.6f}, SegNet: {baseline['mean_seg_dist']:.4f}")
    fp32_bytes = n_params * 4
    print(f"  Archive (FP32): {fp32_bytes / 1024:.1f}KB")

    # FP4 quantization sweep
    from tac.fp4_quantize import quantize_fp4, dequantize_fp4

    results = {
        "baseline_fp32": baseline,
        "baseline_fp32_bytes": fp32_bytes,
        "quantization_sweep": [],
    }

    print(f"\n[quantize] === FP4 QUANTIZATION SWEEP ===")
    for bs in block_sizes:
        print(f"\n  Block size = {bs}:")

        # Quantize
        t0 = time.time()
        packed = quantize_fp4(renderer_state, block_size=bs)
        quant_time = time.time() - t0

        # Dequantize
        restored_state = dequantize_fp4(packed)
        model.load_state_dict(restored_state, strict=False)
        model = model.to(device).eval()

        # Measure quality
        quality = evaluate_model(
            model, gt_frames, gt_masks, posenet, segnet, device,
            simulate_resize=args.simulate_resize,
            max_pairs=min(len(gt_frames) // 2, 50) if args.smoke else None,
        )

        # Compute archive size
        archive_bytes = compute_archive_size(renderer_state, block_size=bs)

        # Quality degradation
        pose_degrad = quality["mean_pose_dist"] / max(baseline["mean_pose_dist"], 1e-8)
        seg_degrad = quality["mean_seg_dist"] / max(baseline["mean_seg_dist"], 1e-8)
        score_degrad = quality["proxy_score"] - baseline["proxy_score"]

        result = {
            "block_size": bs,
            "archive_bytes": archive_bytes,
            "archive_kb": archive_bytes / 1024,
            "compression_ratio": fp32_bytes / archive_bytes,
            "proxy_score": quality["proxy_score"],
            "score_degradation": score_degrad,
            "pose_dist": quality["mean_pose_dist"],
            "seg_dist": quality["mean_seg_dist"],
            "pose_degradation_factor": pose_degrad,
            "seg_degradation_factor": seg_degrad,
            "quant_time_s": quant_time,
        }
        results["quantization_sweep"].append(result)

        print(f"    Archive: {archive_bytes/1024:.1f}KB ({fp32_bytes/archive_bytes:.1f}x compression)")
        print(f"    Proxy score: {quality['proxy_score']:.4f} (Δ={score_degrad:+.4f})")
        print(f"    PoseNet: {quality['mean_pose_dist']:.6f} ({pose_degrad:.2f}x)")
        print(f"    SegNet: {quality['mean_seg_dist']:.4f} ({seg_degrad:.2f}x)")

    # Int8 baseline for comparison
    print(f"\n[quantize] === INT8 BASELINE ===")
    int8_state = {}
    for k, v in renderer_state.items():
        if v.ndim >= 2:
            scale = v.abs().max() / 127.0
            int8_state[k] = ((v / scale).round().clamp(-127, 127) * scale).float()
        else:
            int8_state[k] = v
    model.load_state_dict(int8_state, strict=False)
    model = model.to(device).eval()

    int8_quality = evaluate_model(
        model, gt_frames, gt_masks, posenet, segnet, device,
        simulate_resize=args.simulate_resize,
        max_pairs=min(len(gt_frames) // 2, 50) if args.smoke else None,
    )
    int8_bytes = n_params  # 1 byte per param + scales
    int8_degrad = int8_quality["proxy_score"] - baseline["proxy_score"]
    print(f"  Archive: {int8_bytes/1024:.1f}KB")
    print(f"  Proxy score: {int8_quality['proxy_score']:.4f} (Δ={int8_degrad:+.4f})")
    results["int8"] = {
        "archive_bytes": int8_bytes,
        "proxy_score": int8_quality["proxy_score"],
        "score_degradation": int8_degrad,
    }

    # QAT fine-tuning (optional)
    if args.qat:
        from tac.fp4_quantize import FakeQuantFP4

        print(f"\n[quantize] === QAT FINE-TUNING ===")
        # Reload FP32 weights
        model.load_state_dict(renderer_state, strict=False)
        model = model.to(device).train()

        # Insert fake quantization
        fake_quant = FakeQuantFP4(block_size=32)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.qat_lr)
        from tac.losses import scorer_forward_pair

        n_pairs = len(gt_frames) // 2
        qat_history = []

        for epoch in range(args.qat_epochs):
            perm = torch.randperm(n_pairs)
            epoch_loss = 0.0

            for batch_start in range(0, n_pairs, args.batch_size):
                batch_idx = perm[batch_start:batch_start + args.batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

                for idx in batch_idx:
                    i = idx.item()
                    m0 = gt_masks[i * 2].to(device)
                    m1 = gt_masks[i * 2 + 1].to(device)
                    masks = torch.stack([m0, m1], dim=0)

                    # Apply fake quantization to weights before forward
                    with fake_quant.apply_to(model):
                        rgb = model(masks)

                    if args.simulate_resize:
                        _, _, H, W = rgb.shape
                        small = F.interpolate(rgb, size=(384, 512), mode="bilinear", align_corners=False)
                        rgb = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)

                    pair_chw = rgb.unsqueeze(0)
                    fp_out, fs_out = scorer_forward_pair(pair_chw, posenet, segnet)

                    f0 = gt_frames[i * 2].float().to(device)
                    f1 = gt_frames[i * 2 + 1].float().to(device)
                    gt_pair = torch.stack([f0, f1], dim=0).unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()
                    with torch.no_grad():
                        gp_out, gs_out = scorer_forward_pair(gt_pair, posenet, segnet)

                    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
                    pred_soft = F.softmax(fs_out, dim=1)
                    gt_soft = F.softmax(gs_out, dim=1)
                    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

                    loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
                    batch_loss = batch_loss + loss

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            if epoch % 100 == 0:
                print(f"  QAT epoch {epoch}/{args.qat_epochs} loss={epoch_loss/n_pairs:.4f}")
                qat_history.append({"epoch": epoch, "loss": epoch_loss / n_pairs})

        # Evaluate after QAT
        model.eval()
        packed_qat = quantize_fp4(model.state_dict(), block_size=32)
        restored_qat = dequantize_fp4(packed_qat)
        model.load_state_dict(restored_qat, strict=False)

        qat_quality = evaluate_model(
            model, gt_frames, gt_masks, posenet, segnet, device,
            simulate_resize=args.simulate_resize,
        )
        print(f"  QAT result: proxy={qat_quality['proxy_score']:.4f}")
        results["qat"] = {
            "proxy_score": qat_quality["proxy_score"],
            "epochs": args.qat_epochs,
            "history": qat_history,
        }
        torch.save(model.state_dict(), output_dir / "renderer_qat.pt")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"[quantize] SUMMARY")
    print(f"  FP32 baseline: {baseline['proxy_score']:.4f} ({fp32_bytes/1024:.1f}KB)")
    print(f"  INT8:          {int8_quality['proxy_score']:.4f} ({int8_bytes/1024:.1f}KB)")
    for r in results["quantization_sweep"]:
        print(f"  FP4 bs={r['block_size']:3d}:    {r['proxy_score']:.4f} ({r['archive_kb']:.1f}KB)")
    if args.qat and "qat" in results:
        print(f"  FP4+QAT:       {results['qat']['proxy_score']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
