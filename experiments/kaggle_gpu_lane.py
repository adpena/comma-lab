"""GPU Lane Experiment: Coupled Trajectory + Fridrich Constrained Optimization

Run on Kaggle P100 (16GB VRAM). This is the viability test for the GPU lane.
If this scores below 0.80, GPU lane is the path to victory.

Setup on Kaggle:
    !pip install safetensors timm einops segmentation-models-pytorch av click
    !git clone https://github.com/commaai/comma_video_compression_challenge.git /kaggle/working/upstream
    !cd /kaggle/working/upstream && git lfs pull

    # Upload src/tac/ directory to Kaggle dataset or inline
    # Upload src/ to /kaggle/working/src/
    import sys
    sys.path.insert(0, '/kaggle/working/upstream')
    sys.path.insert(0, '/kaggle/working/src')

Usage:
    python kaggle_gpu_lane.py --num-steps 1000 --device cuda
    python kaggle_gpu_lane.py --num-steps 100 --device cuda   # smoke test
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import click
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: on Kaggle, upstream repo lives at /kaggle/working/upstream
# and our tac source lives at /kaggle/working/src (or wherever uploaded).
# Locally, adjust as needed.
# ---------------------------------------------------------------------------
KAGGLE_UPSTREAM = Path(os.environ.get(
    "UPSTREAM_DIR", "/kaggle/working/upstream"
))
KAGGLE_VIDEO = Path(os.environ.get(
    "VIDEO_PATH",
    "/kaggle/working/upstream/data/target_video.mp4",
))
KAGGLE_POSENET = Path(os.environ.get(
    "POSENET_PATH",
    "/kaggle/working/upstream/models/posenet.safetensors",
))
KAGGLE_SEGNET = Path(os.environ.get(
    "SEGNET_PATH",
    "/kaggle/working/upstream/models/segnet.safetensors",
))
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR", "/kaggle/working/gpu_lane_output"
))

# Ensure upstream is importable (for modules.py with PoseNet/SegNet defs)
if str(KAGGLE_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(KAGGLE_UPSTREAM))


# ---------------------------------------------------------------------------
# Imports from tac (our library)
# ---------------------------------------------------------------------------
from tac.camera import (
    CAMERA_H,
    CAMERA_W,
    SEGNET_INPUT_H,
    SEGNET_INPUT_W,
)
from tac.constrained_gen import (
    coupled_trajectory_optimize,
    build_constrained_archive,
    gpu_lane_full_pipeline,
)
from tac.fridrich import (
    estimate_detection_boundary,
    compute_pixel_cost_map,
    fridrich_constrained_optimize,
    optimal_quantization_stc,
)
from tac.scorer import load_scorers, comma_score
from tac.scorer_targets import extract_posenet_targets
from tac.mask_codec import extract_masks
from tac.data import decode_video


# ---------------------------------------------------------------------------
# VRAM-safe batched scorer evaluation
# ---------------------------------------------------------------------------

def batched_scorer_eval(
    frames: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    gt_frames: list[torch.Tensor],
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, float]:
    """Evaluate proxy score in VRAM-safe batches.

    Args:
        frames: (N, H, W, 3) float tensor of optimized frames.
        posenet: frozen PoseNet.
        segnet: frozen SegNet.
        gt_frames: list of (H, W, 3) uint8 ground truth tensors.
        device: computation device.
        batch_size: max frames per scorer forward pass.

    Returns:
        Dict with seg_dist, pose_dist, and per-component details.
    """
    N = frames.shape[0]
    n_pairs = N // 2

    # SegNet distortion: compare argmax(segnet(optimized)) vs argmax(segnet(gt))
    seg_total = 0.0
    seg_count = 0
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_opt = frames[i:end].to(device)  # (B, H, W, 3)
        batch_gt = torch.stack(gt_frames[i:end]).float().to(device)

        # SegNet expects (B, T=1, C, H, W)
        opt_chw = batch_opt.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
        gt_chw = batch_gt.permute(0, 3, 1, 2).unsqueeze(1).contiguous()

        # Resize to SegNet resolution
        B = opt_chw.shape[0]
        opt_resized = F.interpolate(
            opt_chw.squeeze(1),
            size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(1)
        gt_resized = F.interpolate(
            gt_chw.squeeze(1),
            size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(1)

        with torch.no_grad():
            opt_seg_in = segnet.preprocess_input(opt_resized)
            gt_seg_in = segnet.preprocess_input(gt_resized)
            opt_logits = segnet(opt_seg_in)
            gt_logits = segnet(gt_seg_in)

        opt_masks = opt_logits.argmax(dim=1)
        gt_masks = gt_logits.argmax(dim=1)
        seg_total += (opt_masks != gt_masks).float().mean().item() * B
        seg_count += B

        del batch_opt, batch_gt, opt_chw, gt_chw
        if device == "cuda":
            torch.cuda.empty_cache()

    seg_dist = seg_total / max(seg_count, 1)

    # PoseNet distortion: compare pose outputs on optimized vs GT pairs
    pose_total = 0.0
    pose_count = 0
    for pair_idx in range(0, n_pairs, batch_size):
        pair_end = min(pair_idx + batch_size, n_pairs)
        opt_pairs = []
        gt_pairs = []
        for pi in range(pair_idx, pair_end):
            fi = pi * 2
            # Optimized pair
            f0_opt = frames[fi].to(device)
            f1_opt = frames[fi + 1].to(device)
            opt_pairs.append(torch.stack([f0_opt, f1_opt]).unsqueeze(0))
            # GT pair
            f0_gt = gt_frames[fi].float().to(device)
            f1_gt = gt_frames[fi + 1].float().to(device)
            gt_pairs.append(torch.stack([f0_gt, f1_gt]).unsqueeze(0))

        opt_batch = torch.cat(opt_pairs, dim=0)  # (B, 2, H, W, 3)
        gt_batch = torch.cat(gt_pairs, dim=0)

        # Convert to (B, 2, C, H, W)
        opt_bchw = opt_batch.permute(0, 1, 4, 2, 3).contiguous()
        gt_bchw = gt_batch.permute(0, 1, 4, 2, 3).contiguous()

        with torch.no_grad():
            opt_pose_in = posenet.preprocess_input(opt_bchw)
            gt_pose_in = posenet.preprocess_input(gt_bchw)
            opt_pose = posenet(opt_pose_in)["pose"][..., :6]
            gt_pose = posenet(gt_pose_in)["pose"][..., :6]

        pair_dist = (opt_pose - gt_pose).pow(2).mean().item()
        B = pair_end - pair_idx
        pose_total += pair_dist * B
        pose_count += B

        del opt_batch, gt_batch, opt_bchw, gt_bchw
        if device == "cuda":
            torch.cuda.empty_cache()

    pose_dist = pose_total / max(pose_count, 1)

    return {
        "seg_dist": seg_dist,
        "pose_dist": pose_dist,
        "n_frames": N,
        "n_pairs": n_pairs,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@click.command()
@click.option("--num-steps", default=1000, help="Coupled trajectory optimization steps")
@click.option("--fridrich-steps", default=500, help="Fridrich constrained refinement steps")
@click.option("--lr", default=0.01, help="Optimization learning rate")
@click.option("--seg-weight", default=100.0, help="SegNet constraint weight")
@click.option("--pose-weight", default=10.0, help="PoseNet constraint weight")
@click.option("--compress-weight", default=1.0, help="Compressibility weight")
@click.option("--noise-seed", default=42, help="Deterministic noise seed")
@click.option("--device", default="cuda", help="Computation device")
@click.option("--batch-size", default=8, help="Max frames per scorer batch (P100 16GB -> 8)")
@click.option("--save-every", default=100, help="Save checkpoint every N steps")
@click.option("--skip-fridrich", is_flag=True, help="Skip Fridrich refinement pass")
@click.option("--use-full-pipeline", is_flag=True, help="Use gpu_lane_full_pipeline (all stages)")
@click.option("--video-path", default=None, help="Override video path")
@click.option("--posenet-path", default=None, help="Override posenet path")
@click.option("--segnet-path", default=None, help="Override segnet path")
@click.option("--output-dir", default=None, help="Override output directory")
@click.option("--upstream-dir", default=None, help="Override upstream directory")
def main(
    num_steps: int,
    fridrich_steps: int,
    lr: float,
    seg_weight: float,
    pose_weight: float,
    compress_weight: float,
    noise_seed: int,
    device: str,
    batch_size: int,
    save_every: int,
    skip_fridrich: bool,
    use_full_pipeline: bool,
    video_path: str | None,
    posenet_path: str | None,
    segnet_path: str | None,
    output_dir: str | None,
    upstream_dir: str | None,
) -> None:
    """GPU Lane: Coupled Trajectory + Fridrich optimization on Kaggle P100."""
    video = Path(video_path) if video_path else KAGGLE_VIDEO
    pnet = Path(posenet_path) if posenet_path else KAGGLE_POSENET
    snet = Path(segnet_path) if segnet_path else KAGGLE_SEGNET
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    upstream = Path(upstream_dir) if upstream_dir else KAGGLE_UPSTREAM

    out.mkdir(parents=True, exist_ok=True)

    print(f"[gpu_lane] Device: {device}")
    print(f"[gpu_lane] Video: {video}")
    print(f"[gpu_lane] Scorers: {pnet}, {snet}")
    print(f"[gpu_lane] Output: {out}")
    print(f"[gpu_lane] Steps: {num_steps} coupled + {fridrich_steps} Fridrich")
    print(f"[gpu_lane] Batch size: {batch_size}")

    # ----------------------------------------------------------------
    # Step 1: Load scorer models
    # ----------------------------------------------------------------
    t0 = time.time()
    print("\n[gpu_lane] Step 1: Loading scorer models...")
    posenet, segnet = load_scorers(pnet, snet, device=device, upstream_dir=upstream)
    from tac.scorer import make_scorers_differentiable
    make_scorers_differentiable(posenet, segnet)
    print(f"  Loaded + patched differentiable in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------
    # Step 2: Decode source video
    # ----------------------------------------------------------------
    t0 = time.time()
    print("\n[gpu_lane] Step 2: Decoding video...")
    gt_frames = decode_video(video, target_h=CAMERA_H, target_w=CAMERA_W)
    n_frames = len(gt_frames)
    print(f"  Decoded {n_frames} frames in {time.time() - t0:.1f}s")
    print(f"  Frame shape: {gt_frames[0].shape}")

    # ----------------------------------------------------------------
    # Step 3: Extract masks and pose targets from GT
    # ----------------------------------------------------------------
    t0 = time.time()
    print("\n[gpu_lane] Step 3: Extracting masks...")
    masks = extract_masks(gt_frames, segnet, device=device, batch_size=batch_size)
    print(f"  Masks shape: {masks.shape} in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("\n[gpu_lane] Step 4: Extracting PoseNet targets from GT...")
    pose_result = extract_posenet_targets(
        gt_frames, posenet, device=device, batch_size=batch_size,
    )
    pose_targets = pose_result["targets"]  # (n_pairs, 6)
    print(f"  Pose targets shape: {pose_targets.shape} in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------
    # Step 5: Run the GPU lane pipeline
    # ----------------------------------------------------------------
    if use_full_pipeline:
        # Use the integrated pipeline with all Fridrich stages
        t0 = time.time()
        print("\n[gpu_lane] Step 5: Running full GPU lane pipeline...")
        result = gpu_lane_full_pipeline(
            masks=masks,
            posenet=posenet,
            segnet=segnet,
            device=device,
            expected_pose=pose_targets,
            num_steps=num_steps,
            fridrich_steps=fridrich_steps,
            lr=lr,
            seg_weight=seg_weight,
            pose_weight=pose_weight,
            compress_weight=compress_weight,
            noise_seed=noise_seed,
            batch_size=batch_size,
            log_every=save_every,
        )
        optimized_frames = result["optimized_frames"]
        print(f"  Full pipeline completed in {time.time() - t0:.1f}s")
        print(f"  Diagnostics: {json.dumps(result['diagnostics'], indent=2, default=str)}")

    else:
        # Manual step-by-step pipeline for more control

        # Step 5a: Coupled trajectory optimization (4D-Var)
        t0 = time.time()
        print(f"\n[gpu_lane] Step 5a: Coupled trajectory optimization ({num_steps} steps)...")
        optimized_frames = coupled_trajectory_optimize(
            masks=masks,
            expected_pose=pose_targets,
            posenet=posenet,
            segnet=segnet,
            num_steps=num_steps,
            lr=lr,
            seg_weight=seg_weight,
            pose_weight=pose_weight,
            compress_weight=compress_weight,
            noise_seed=noise_seed,
            device=device,
            log_every=save_every,
        )
        elapsed = time.time() - t0
        print(f"  Coupled trajectory done in {elapsed:.1f}s ({elapsed/num_steps:.3f}s/step)")

        # Save intermediate checkpoint
        ckpt_path = out / "coupled_trajectory_frames.pt"
        torch.save(optimized_frames.cpu(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        # Step 5b: Fridrich constrained refinement
        if not skip_fridrich:
            t0 = time.time()
            print(f"\n[gpu_lane] Step 5b: Fridrich refinement ({fridrich_steps} steps)...")

            # Estimate detection boundary
            print("  Estimating detection boundary...")
            boundary = estimate_detection_boundary(
                optimized_frames.permute(0, 3, 1, 2).contiguous(),
                posenet, segnet,
                num_probes=20,
                max_magnitude=30.0,
                device=device,
                subsample_frames=4,
            )
            seg_boundary = boundary["seg_boundary"]
            pose_boundary = boundary["pose_boundary"]
            print(f"  Detection boundary: seg={seg_boundary:.6f}, pose={pose_boundary:.6f}")

            # Compute pixel cost map
            print("  Computing pixel cost map...")
            cost_map = compute_pixel_cost_map(
                optimized_frames.permute(0, 3, 1, 2).contiguous(),
                posenet, segnet,
                method="hybrid",
                device=device,
                subsample_frames=1,
            )
            print(f"  Cost map: shape={cost_map.shape}, range=[{cost_map.min():.4f}, {cost_map.max():.4f}]")

            # Fridrich constrained optimization
            print("  Running Fridrich constrained optimization...")
            optimized_bchw = fridrich_constrained_optimize(
                optimized_frames,
                posenet, segnet,
                pixel_costs=cost_map,
                seg_boundary=seg_boundary,
                pose_boundary=pose_boundary,
                num_steps=fridrich_steps,
                lr=lr,
                device=device,
                batch_size=batch_size,
            )
            # Convert back to (N, H, W, 3)
            optimized_frames = optimized_bchw.permute(0, 2, 3, 1).contiguous()

            # STC quantization
            print("  Applying STC quantization...")
            optimized_bchw_quant = optimal_quantization_stc(
                optimized_bchw.float(),
                cost_map,
                target_rate_reduction=0.1,
            )
            optimized_frames = optimized_bchw_quant.permute(0, 2, 3, 1).contiguous().float()

            elapsed = time.time() - t0
            print(f"  Fridrich refinement done in {elapsed:.1f}s")

            # Save Fridrich checkpoint
            ckpt_path = out / "fridrich_refined_frames.pt"
            torch.save(optimized_frames.cpu(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ----------------------------------------------------------------
    # Step 6: Build archive and compute proxy score
    # ----------------------------------------------------------------
    t0 = time.time()
    print("\n[gpu_lane] Step 6: Building archive...")
    archive_dir = out / "archive"
    build_constrained_archive(masks, pose_targets, noise_seed, archive_dir)

    # Also save optimized frames as the inflated output
    frames_path = out / "optimized_frames.pt"
    torch.save(optimized_frames.cpu(), frames_path)

    # Compute archive size (rate)
    archive_size = sum(f.stat().st_size for f in archive_dir.iterdir() if f.is_file())
    # Rate = archive_size / (n_frames * H * W * 1.5)  [YUV420 bytes per pixel]
    total_yuv_bytes = n_frames * CAMERA_H * CAMERA_W * 1.5
    rate = archive_size / total_yuv_bytes
    print(f"  Archive: {archive_size} bytes, rate={rate:.6f}")

    # ----------------------------------------------------------------
    # Step 7: Compute proxy score
    # ----------------------------------------------------------------
    print("\n[gpu_lane] Step 7: Computing proxy score...")
    eval_result = batched_scorer_eval(
        optimized_frames,
        posenet, segnet,
        gt_frames,
        device=device,
        batch_size=batch_size,
    )
    seg_dist = eval_result["seg_dist"]
    pose_dist = eval_result["pose_dist"]
    proxy = comma_score(pose_dist, seg_dist, rate)

    print(f"\n{'='*60}")
    print(f"  GPU LANE PROXY SCORE: {proxy:.4f}")
    print("  Components:")
    print(f"    SegNet distortion:  {seg_dist:.6f} (100x = {100*seg_dist:.4f})")
    print(f"    PoseNet distortion: {pose_dist:.6f} (sqrt(10x) = {math.sqrt(10*pose_dist):.4f})")
    print(f"    Rate:               {rate:.6f} (25x = {25*rate:.4f})")
    print(f"{'='*60}")

    # Save results summary
    elapsed_total = time.time() - t0

    # Cost tracking
    cost_info = {}
    try:
        from tac.cost_tracker import CostRecord, collect_replicability_metadata
        gpu_label = "p100" if "p100" in device.lower() or Path("/kaggle").exists() else device
        cost_rec = CostRecord.from_run(
            platform="kaggle" if Path("/kaggle").exists() else "local",
            gpu=gpu_label,
            runtime_seconds=elapsed_total,
        )
        cost_info = cost_rec.to_dict()
        replicability = collect_replicability_metadata(device)
    except ImportError:
        replicability = {}

    results = {
        "proxy_score": proxy,
        "seg_dist": seg_dist,
        "pose_dist": pose_dist,
        "rate": rate,
        "archive_bytes": archive_size,
        "n_frames": n_frames,
        "num_steps": num_steps,
        "fridrich_steps": fridrich_steps if not skip_fridrich else 0,
        "device": device,
        "elapsed_total": elapsed_total,
        "cost": cost_info,
        "replicability": replicability,
    }
    results_path = out / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
