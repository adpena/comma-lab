# SPDX-License-Identifier: MIT
"""Validate adaptive per-pixel YUV null space projection.

Verifies that:
1. The null space basis is orthonormal
2. Projected gradients are truly in PoseNet's null space (PoseNet output unchanged)
3. SegNet can still improve (projected gradient has nonzero SegNet component)

Usage:
    .venv/bin/python experiments/validate_null_space_projection.py [--device mps|cuda|cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_basis_orthonormality(basis: torch.Tensor) -> float:
    """Check that basis @ basis.T == I_K. Returns max deviation."""
    gram = basis @ basis.T
    identity = torch.eye(basis.shape[0], device=basis.device, dtype=basis.dtype)
    max_err = (gram - identity).abs().max().item()
    return max_err


def validate_posenet_invariance(
    basis: torch.Tensor,
    device: torch.device,
    n_trials: int = 20,
    magnitude: float = 5.0,
) -> dict:
    """Verify that null space perturbations don't affect PoseNet output.

    Creates random 2x2 blocks, perturbs them along null space directions,
    and checks that rgb_to_yuv6 output is unchanged.
    """
    from tac.constrained_gen import rgb_to_yuv6

    max_diff = 0.0
    for _ in range(n_trials):
        # Random image: (1, 3, 8, 8) -- 4x4 = 16 blocks of 2x2
        img = torch.rand(1, 3, 8, 8, device=device) * 255.0

        # Get YUV6 before perturbation
        yuv_before = rgb_to_yuv6(img)

        # Generate random perturbation in null space
        B, C, H, W = img.shape
        blocks = img.reshape(B, 3, H // 2, 2, W // 2, 2)
        blocks = blocks.permute(0, 2, 4, 3, 5, 1).reshape(B, H // 2, W // 2, 12)

        # Random coefficients for null space directions
        null_basis = basis.to(device, img.dtype)
        K = null_basis.shape[0]
        coeffs = torch.randn(B, H // 2, W // 2, K, device=device) * magnitude

        # Project: perturbation = coeffs @ null_basis -> (B, H//2, W//2, 12)
        perturbation = coeffs @ null_basis

        # Apply perturbation
        blocks_perturbed = blocks + perturbation

        # Reshape back to image
        blocks_perturbed = blocks_perturbed.reshape(B, H // 2, W // 2, 2, 2, 3)
        blocks_perturbed = blocks_perturbed.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, H, W)
        # Check WITHOUT clamping first (pure null space property)
        yuv_after_noclamp = rgb_to_yuv6(blocks_perturbed)
        diff_noclamp = (yuv_before - yuv_after_noclamp).abs().max().item()
        max_diff = max(max_diff, diff_noclamp)

    return {"max_posenet_diff": max_diff, "n_trials": n_trials, "magnitude": magnitude}


def validate_segnet_improvement(
    basis: torch.Tensor,
    device: torch.device,
) -> dict:
    """Verify that projection retains nonzero SegNet-useful gradient component."""
    from tac.scorer_exploits import project_segnet_grad_to_posenet_null_space

    # Random SegNet gradient
    grad = torch.randn(1, 3, 16, 16, device=device)
    grad_norm = grad.norm().item()

    # Project into null space
    projected = project_segnet_grad_to_posenet_null_space(
        grad, basis.to(device), max_magnitude=100.0,
    )
    projected_norm = projected.norm().item()

    # The projection should retain a meaningful fraction (theory: 50% = 6/12 DOF)
    retention = projected_norm / grad_norm if grad_norm > 0 else 0.0

    return {
        "grad_norm": grad_norm,
        "projected_norm": projected_norm,
        "retention_fraction": retention,
        "expected_retention": "~0.5 (6/12 DOF in null space)",
    }


def validate_full_pipeline(
    basis: torch.Tensor,
    device: torch.device,
) -> dict:
    """End-to-end test: apply null space projection to frames, verify PoseNet unchanged."""
    from tac.constrained_gen import rgb_to_yuv6
    from tac.scorer_exploits import project_segnet_grad_to_posenet_null_space

    # Simulated frames and gradient
    frames = torch.rand(2, 3, 16, 16, device=device) * 255.0
    segnet_grad = torch.randn(2, 3, 16, 16, device=device)

    # Get YUV6 before
    yuv_before = rgb_to_yuv6(frames)

    # Project gradient and apply
    projected = project_segnet_grad_to_posenet_null_space(
        segnet_grad, basis.to(device), max_magnitude=5.0,
    )
    frames_after = (frames - 0.5 * projected).clamp(0.0, 255.0)

    # Get YUV6 after
    yuv_after = rgb_to_yuv6(frames_after)

    diff = (yuv_before - yuv_after).abs().max().item()

    return {
        "max_yuv_diff_after_projection": diff,
        "pass": diff < 0.1,  # Should be near-zero (clamping introduces small errors)
    }


def main():
    parser = argparse.ArgumentParser(description="Validate YUV null space projection")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for validation (cpu, mps, cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[validate] Device: {device}")

    # Load basis
    basis_path = Path("experiments/results/yuv_null_space/null_space_basis.pt")
    if basis_path.exists():
        basis = torch.load(basis_path, map_location=device, weights_only=True)
        print(f"[validate] Loaded precomputed basis: shape {basis.shape}")
    else:
        print("[validate] Precomputed basis not found, computing analytically...")
        from tac.scorer_exploits import _compute_yuv420_null_space_basis
        basis = _compute_yuv420_null_space_basis(device=device)
        print(f"[validate] Computed basis: shape {basis.shape}")

    # Test 1: Orthonormality
    orth_err = validate_basis_orthonormality(basis)
    status = "PASS" if orth_err < 1e-5 else "FAIL"
    print(f"\n[Test 1] Orthonormality: max error = {orth_err:.2e} [{status}]")

    # Test 2: PoseNet invariance
    result = validate_posenet_invariance(basis, device)
    status = "PASS" if result["max_posenet_diff"] < 0.01 else "FAIL"
    print(f"[Test 2] PoseNet invariance: max diff = {result['max_posenet_diff']:.6f} "
          f"(mag={result['magnitude']}, trials={result['n_trials']}) [{status}]")

    # Test 3: SegNet gradient retention
    result = validate_segnet_improvement(basis, device)
    status = "PASS" if result["retention_fraction"] > 0.3 else "FAIL"
    print(f"[Test 3] SegNet gradient retention: {result['retention_fraction']:.3f} "
          f"({result['expected_retention']}) [{status}]")

    # Test 4: Full pipeline
    result = validate_full_pipeline(basis, device)
    status = "PASS" if result["pass"] else "FAIL"
    print(f"[Test 4] Full pipeline PoseNet invariance: "
          f"max YUV diff = {result['max_yuv_diff_after_projection']:.6f} [{status}]")

    print("\n[validate] All tests complete.")


if __name__ == "__main__":
    main()
