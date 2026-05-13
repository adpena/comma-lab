#!/usr/bin/env python3
"""YUV Null Space Exploitation: identify pixels invisible to PoseNet.

PoseNet's preprocess_input converts RGB to YUV6 with 4:2:0 chroma subsampling.
Within each 2x2 block, perturbations that sum to zero in chroma channels are
invisible to PoseNet because the subsampling averages them out.

This script:
    1. Analyzes the YUV6 transform to identify exact null space dimensions
    2. Demonstrates: perturb a frame in the null space, verify PoseNet output is identical
    3. Quantifies: how many free dimensions exist per frame (~294K)
    4. Tests: use null space perturbations to improve SegNet WITHOUT affecting PoseNet

Theoretical basis:
    RGB→YUV: linear transform (BT.601 matrix)
    YUV 4:2:0: U and V are 2x2 averaged → subsampled to (H/2, W/2)
    Null space = perturbations ΔR, ΔG, ΔB in a 2x2 block such that:
        - ΔY remains unchanged (preserve luma — OR: we can change luma freely since
          PoseNet uses all of it, so null space is only in chroma)
        - Actually: PoseNet uses full-res Y + subsampled U + subsampled V = "YUV6"
        - The null space is in U/V: within each 2x2 block, perturbations to R,G,B
          that create chroma differences that average to zero across the 2x2 block

    Per 2x2 block (4 pixels × 3 channels = 12 DOF input):
        PoseNet sees: 4 Y values + 1 U + 1 V = 6 values → 12 - 6 = 6 null dims per block
        Total null dims per frame: (H/2 × W/2) × 6 = 437 × 582 × 6 / 4 ≈ 190K

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/analysis/yuv_null_space.py \
        --device mps --smoke

    # Full analysis:
    PYTHONPATH=src:upstream python experiments/analysis/yuv_null_space.py \
        --device mps --n-frames 1200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.utils import find_project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YUV null space analysis for PoseNet-invisible perturbations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="mps", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames to analyze")
    p.add_argument("--perturbation-magnitude", type=float, default=5.0,
                   help="Magnitude of null-space perturbation (pixel units)")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 20 frames")
    return p.parse_args()


def compute_yuv6_transform_matrix() -> torch.Tensor:
    """Compute the RGB→YUV BT.601 transform matrix.

    Returns:
        (3, 3) transform matrix where YUV = M @ RGB
    """
    # BT.601 limited range:
    # Y = 0.299*R + 0.587*G + 0.114*B  (then scaled to limited range)
    # U = -0.169*R - 0.331*G + 0.500*B + 128
    # V = 0.500*R - 0.419*G - 0.081*B + 128
    M = torch.tensor([
        [0.299, 0.587, 0.114],    # Y
        [-0.169, -0.331, 0.500],  # U (Cb)
        [0.500, -0.419, -0.081],  # V (Cr)
    ], dtype=torch.float32)
    return M


def compute_null_space_basis_per_block() -> torch.Tensor:
    """Compute null space basis for a 2x2 pixel block in YUV6 representation.

    PoseNet sees per 2x2 block:
        - 4 Y values (full resolution luma): Y_00, Y_01, Y_10, Y_11
        - 1 U value (average of 4 U values): mean(U_00, U_01, U_10, U_11)
        - 1 V value (average of 4 V values): mean(V_00, V_01, V_10, V_11)
        = 6 observed values

    Input DOF per 2x2 block:
        - 4 pixels × 3 RGB channels = 12

    Null space dim = 12 - 6 = 6

    Returns:
        (6, 12) null space basis vectors (each row is one basis vector)
        The 12-dim vector is [R00, G00, B00, R01, G01, B01, R10, G10, B10, R11, G11, B11]
    """
    M = compute_yuv6_transform_matrix()  # (3, 3)

    # Build the observation matrix A such that observed = A @ pixel_vector
    # pixel_vector is 12D: [R00, G00, B00, R01, G01, B01, R10, G10, B10, R11, G11, B11]
    # observed is 6D: [Y00, Y01, Y10, Y11, U_avg, V_avg]
    A = torch.zeros(6, 12)

    # Y values (full res): Y_ij = M[0,:] @ RGB_ij
    for i, offset in enumerate([0, 3, 6, 9]):
        A[i, offset:offset + 3] = M[0]  # Y row of transform

    # U_avg = mean(M[1,:] @ RGB_ij for all ij)
    for offset in [0, 3, 6, 9]:
        A[4, offset:offset + 3] = M[1] / 4.0  # U row, averaged

    # V_avg = mean(M[2,:] @ RGB_ij for all ij)
    for offset in [0, 3, 6, 9]:
        A[5, offset:offset + 3] = M[2] / 4.0  # V row, averaged

    # Null space = kernel of A
    # SVD: A = U S V^T, null space = columns of V corresponding to zero singular values
    _, S, Vh = torch.linalg.svd(A, full_matrices=True)
    # Rank of A should be 6, so null space has dim 12 - 6 = 6
    rank = (S > 1e-6).sum().item()
    null_basis = Vh[rank:]  # (6, 12) — the last 6 rows of V^T

    return null_basis


def generate_null_space_perturbation(
    frame: torch.Tensor,
    magnitude: float = 5.0,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a random perturbation in PoseNet's null space.

    Args:
        frame: (H, W, 3) float tensor
        magnitude: L-inf magnitude of the perturbation
        seed: random seed for reproducibility

    Returns:
        (H, W, 3) perturbation tensor (add to frame for perturbed version)
    """
    H, W, _ = frame.shape
    assert H % 2 == 0 and W % 2 == 0, f"Frame dims must be even, got {H}x{W}"

    null_basis = compute_null_space_basis_per_block()  # (6, 12)

    # Generate random coefficients in null space
    gen = torch.Generator()
    gen.manual_seed(seed)
    n_blocks = (H // 2) * (W // 2)
    # Random coefficients for each block's 6 null space dimensions
    coeffs = torch.randn(n_blocks, 6, generator=gen) * magnitude

    # Project to pixel space: perturbation = coeffs @ null_basis
    # Result: (n_blocks, 12) — reshape to (H, W, 3)
    delta_flat = coeffs @ null_basis  # (n_blocks, 12)

    # Vectorized reshape: each block's 12 values are [R00,G00,B00, R01,G01,B01, R10,G10,B10, R11,G11,B11]
    # Reshape to (H//2, W//2, 4_pixels, 3_channels), then scatter into (H, W, 3)
    bH, bW = H // 2, W // 2
    delta_blocks = delta_flat.reshape(bH, bW, 4, 3)  # 4 pixels per block, 3 channels

    perturbation = torch.zeros(H, W, 3)
    perturbation[0::2, 0::2] = delta_blocks[:, :, 0]  # pixel (0,0)
    perturbation[0::2, 1::2] = delta_blocks[:, :, 1]  # pixel (0,1)
    perturbation[1::2, 0::2] = delta_blocks[:, :, 2]  # pixel (1,0)
    perturbation[1::2, 1::2] = delta_blocks[:, :, 3]  # pixel (1,1)

    # Clamp to maintain valid pixel range
    # Scale so max perturbation is roughly magnitude
    max_pert = perturbation.abs().max()
    if max_pert > 0:
        perturbation = perturbation * (magnitude / max_pert)

    return perturbation


def verify_posenet_invariance(
    frame0: torch.Tensor,
    frame1: torch.Tensor,
    perturbation: torch.Tensor,
    posenet: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Verify that null-space perturbation does not change PoseNet output.

    Args:
        frame0, frame1: (H, W, 3) float tensors (original pair)
        perturbation: (H, W, 3) null-space perturbation
        posenet: PoseNet model
        device: compute device

    Returns:
        Dict with original/perturbed outputs and difference.
    """
    # Original pair
    f0 = frame0.to(device).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)  # (1,1,3,H,W)
    f1 = frame1.to(device).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)
    pair_orig = torch.cat([f0, f1], dim=1)  # (1, 2, 3, H, W)

    # Perturbed pair (apply perturbation to frame0 only)
    f0_pert = (frame0 + perturbation).clamp(0, 255).to(device)
    f0p = f0_pert.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)
    pair_pert = torch.cat([f0p, f1], dim=1)

    with torch.no_grad():
        orig_in = posenet.preprocess_input(pair_orig)
        pert_in = posenet.preprocess_input(pair_pert)
        orig_out = posenet(orig_in)["pose"][..., :6]
        pert_out = posenet(pert_in)["pose"][..., :6]

    diff = (orig_out - pert_out).abs()
    return {
        "max_pose_diff": diff.max().item(),
        "mean_pose_diff": diff.mean().item(),
        "l2_pose_diff": diff.pow(2).sum().sqrt().item(),
        "perturbation_l2": perturbation.pow(2).sum().sqrt().item(),
        "perturbation_linf": perturbation.abs().max().item(),
    }


def check_segnet_effect(
    frame0: torch.Tensor,
    frame1: torch.Tensor,
    perturbation: torch.Tensor,
    segnet: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Check if null-space perturbation changes SegNet output.

    If SegNet IS affected but PoseNet is NOT, we have a free lever for SegNet improvement.

    Returns:
        Dict with SegNet agreement/disagreement metrics.
    """
    # Original
    f0 = frame0.to(device).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)
    f1 = frame1.to(device).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)
    pair_orig = torch.cat([f0, f1], dim=1)

    # Perturbed
    f0_pert = (frame0 + perturbation).clamp(0, 255).to(device)
    f0p = f0_pert.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)
    pair_pert = torch.cat([f0p, f1], dim=1)

    with torch.no_grad():
        orig_in = segnet.preprocess_input(pair_orig)
        pert_in = segnet.preprocess_input(pair_pert)
        orig_out = F.softmax(segnet(orig_in), dim=1)
        pert_out = F.softmax(segnet(pert_in), dim=1)

    # Argmax agreement
    orig_class = orig_out.argmax(dim=1)
    pert_class = pert_out.argmax(dim=1)
    agreement = (orig_class == pert_class).float().mean().item()

    # Soft distance
    soft_dist = 1.0 - (orig_out * pert_out).sum(dim=1).mean().item()

    return {
        "argmax_agreement": agreement,
        "argmax_disagreement_pct": (1.0 - agreement) * 100,
        "soft_distance": soft_dist,
    }


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.n_frames = 20

    device = torch.device(args.device)

    # Resolve paths
    root = find_project_root()
    upstream = Path(args.upstream) if args.upstream else root / "upstream"
    video_path = Path(args.video) if args.video else upstream / "videos" / "0.mkv"
    output_dir = Path(args.output_dir) if args.output_dir else (
        root / "experiments" / "results" / "yuv_null_space"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Theoretical analysis
    print("[null-space] === THEORETICAL ANALYSIS ===")
    null_basis = compute_null_space_basis_per_block()
    print(f"  Null space basis shape: {null_basis.shape}")
    print(f"  Null space dimensionality per 2x2 block: {null_basis.shape[0]}")

    # Verify orthogonality to observation matrix
    M = compute_yuv6_transform_matrix()
    A = torch.zeros(6, 12)
    for i, offset in enumerate([0, 3, 6, 9]):
        A[i, offset:offset + 3] = M[0]
    for offset in [0, 3, 6, 9]:
        A[4, offset:offset + 3] = M[1] / 4.0
        A[5, offset:offset + 3] = M[2] / 4.0

    # Check A @ null_basis^T ≈ 0
    check = A @ null_basis.T
    max_violation = check.abs().max().item()
    print(f"  Null space verification: max |A @ null_basis^T| = {max_violation:.2e} (should be ~0)")

    # Frame-level analysis
    H, W = 874, 1164
    n_blocks = (H // 2) * (W // 2)
    null_dims_per_frame = n_blocks * 6
    total_dims_per_frame = H * W * 3
    print(f"\n  Frame dimensions: {H}x{W}x3 = {total_dims_per_frame:,} total DOF")
    print(f"  Number of 2x2 blocks: {n_blocks:,}")
    print(f"  Null space dims per frame: {null_dims_per_frame:,}")
    print(f"  Observable dims per frame: {total_dims_per_frame - null_dims_per_frame:,}")
    print(f"  Free fraction: {null_dims_per_frame/total_dims_per_frame:.1%}")

    # Load models
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(str(upstream), device=device)

    # Decode video
    from tac.data import decode_video
    print(f"\n[null-space] Decoding video: {video_path}")
    gt_frames = decode_video(str(video_path))[:args.n_frames]
    n_pairs = len(gt_frames) // 2

    # Empirical verification
    print("\n[null-space] === EMPIRICAL VERIFICATION ===")
    print(f"  Perturbation magnitude: {args.perturbation_magnitude}")

    posenet_diffs = []
    segnet_effects = []
    t0 = time.time()

    for pair_idx in range(n_pairs):
        f0 = gt_frames[pair_idx * 2].float()
        f1 = gt_frames[pair_idx * 2 + 1].float()

        # Generate null-space perturbation
        perturbation = generate_null_space_perturbation(
            f0, magnitude=args.perturbation_magnitude, seed=pair_idx
        )

        # Verify PoseNet invariance
        pose_result = verify_posenet_invariance(f0, f1, perturbation, posenet, device)
        posenet_diffs.append(pose_result)

        # Check SegNet effect
        seg_result = check_segnet_effect(f0, f1, perturbation, segnet, device)
        segnet_effects.append(seg_result)

        if pair_idx % 20 == 0:
            print(f"  Pair {pair_idx}/{n_pairs} | PoseNet Δ={pose_result['max_pose_diff']:.2e} "
                  f"| SegNet disagree={seg_result['argmax_disagreement_pct']:.1f}%")

    total_time = time.time() - t0

    # Aggregate results
    mean_pose_diff = np.mean([d["max_pose_diff"] for d in posenet_diffs])
    max_pose_diff = np.max([d["max_pose_diff"] for d in posenet_diffs])
    mean_seg_disagree = np.mean([s["argmax_disagreement_pct"] for s in segnet_effects])
    mean_seg_soft = np.mean([s["soft_distance"] for s in segnet_effects])

    results = {
        "theoretical": {
            "null_space_dim_per_block": 6,
            "null_space_dim_per_frame": null_dims_per_frame,
            "total_dim_per_frame": total_dims_per_frame,
            "free_fraction": null_dims_per_frame / total_dims_per_frame,
            "null_space_verification_max_error": max_violation,
        },
        "empirical": {
            "n_pairs_tested": n_pairs,
            "perturbation_magnitude": args.perturbation_magnitude,
            "posenet_invariance": {
                "mean_max_diff": mean_pose_diff,
                "worst_case_max_diff": max_pose_diff,
                "is_invariant": max_pose_diff < 1e-4,
            },
            "segnet_effect": {
                "mean_argmax_disagreement_pct": mean_seg_disagree,
                "mean_soft_distance": mean_seg_soft,
                "is_affected": mean_seg_disagree > 0.1,
            },
            "total_time_s": total_time,
        },
        "conclusion": {
            "posenet_null_space_confirmed": max_pose_diff < 1e-4,
            "segnet_affected_by_null_space": mean_seg_disagree > 0.1,
            "exploitable": max_pose_diff < 1e-4 and mean_seg_disagree > 0.1,
            "free_dimensions_per_frame": null_dims_per_frame,
        },
    }

    # Convert numpy/torch bools to Python bools for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy/torch scalar
            return obj.item()
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    with open(output_dir / "results.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    # Save null space basis for use in training
    torch.save(null_basis, output_dir / "null_space_basis.pt")

    # Summary
    print(f"\n{'='*60}")
    print("[null-space] RESULTS")
    print(f"  PoseNet invariance confirmed: {max_pose_diff < 1e-4}")
    print(f"    Max pose diff: {max_pose_diff:.2e} (threshold: 1e-4)")
    print(f"    Mean pose diff: {mean_pose_diff:.2e}")
    print(f"  SegNet affected: {mean_seg_disagree > 0.1}")
    print(f"    Mean argmax disagreement: {mean_seg_disagree:.2f}%")
    print(f"    Mean soft distance: {mean_seg_soft:.6f}")
    print(f"  Free dimensions per frame: {null_dims_per_frame:,} ({null_dims_per_frame/total_dims_per_frame:.1%})")
    print(f"  EXPLOITABLE: {results['conclusion']['exploitable']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
