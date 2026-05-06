#!/usr/bin/env python3
# LANE_GP_BASIS_FIT_KILL_ACKNOWLEDGED:
# Read .omx/research/council_lane_gp_v4_design_20260430.md before adding
# ANY new smooth-basis pose-fit experiment. Lane GP v3 (89.67 [Modal-T4-CPU])
# was retired as the measured smooth-basis implementation per Council #271 +
# Lane GP v4 design verdict. The baseline Lane G v3 pose trajectory is
# approximately white-noise in dims 1-5 (diff_std > signal_std) with
# uniformly-distributed spectral support — no reviewed smooth basis
# (polynomial / B-spline / DCT / natural cubic) can fit it below RMSE ≈ 1.2
# (near signal std). The Runge-phenomenon diagnosis in
# project_lane_gp_v3_landed_runge_phenomenon_20260429.md was incomplete; the
# trajectory is structurally incompressible by the reviewed smooth bases at any
# K. This file is RETAINED for archival/historical reasons (Lane GP v3
# reproducer); new smooth-basis work requires explicit reactivation criteria.
# See preflight.py Check 91.
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tac.pose_gaussian_process import fit_pose_gp, reconstruct_poses, save_pose_gp


def _load_pose_tensor(path: Path) -> torch.Tensor:
    # Audit Finding 11 (2026-05-06): map_location="cpu" — `fit_pose_gp` does
    # numpy fitting; CUDA load was wasteful overhead and broke CPU-only machines.
    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        for key in ("poses", "optimized_poses", "gt_poses"):
            value = data.get(key)
            if value is not None:
                return torch.as_tensor(value, dtype=torch.float32)
        raise ValueError(f"{path} dict has no poses, optimized_poses, or gt_poses key")
    return torch.as_tensor(data, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Lane GP pose polynomial")
    parser.add_argument("--poses", required=True, help="Path to optimized_poses.pt")
    parser.add_argument("--output", required=True, help="Path for pose_gp.bin")
    parser.add_argument("--n-pairs", type=int, default=600)
    args = parser.parse_args()

    baseline = _load_pose_tensor(Path(args.poses))
    model = fit_pose_gp(baseline)
    save_pose_gp(model, args.output)
    # Pass baseline_poses=baseline so dims 1-5 are preserved (NOT zero-padded).
    # Without this kwarg, reconstruct_poses defaults dims 1-5 to ZERO which is
    # OFF-MANIFOLD for 6-DOF-trained renderers and CATASTROPHICALLY degrades
    # scores (Lane GP v2 audit, 2026-04-29). The earlier "Fix A" was advertised
    # as landed but the fit_pose_gp.py call site never passed the kwarg —
    # discovered by Lane MM/GP pipeline audit codex 2026-04-29 PM. Lane GP v3
    # = 89.67 score is partially attributable to this bug, not just the
    # Runge-phenomenon polynomial-fit limitation.
    reconstructed = reconstruct_poses(model, int(args.n_pairs), baseline_poses=baseline).to(baseline.device)
    compare_n = min(int(args.n_pairs), baseline.shape[0], reconstructed.shape[0])
    rmse = torch.sqrt(torch.mean((reconstructed[:compare_n, 0] - baseline[:compare_n, 0]).square()))

    coeff_summary = ", ".join(f"{v:.6g}" for v in model.poly_coeffs.tolist())
    sigma_summary = ", ".join(f"{v:.6g}" for v in model.sigma.tolist())
    out_path = Path(args.output)
    print(f"pose_gp: wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"pose_gp: degree10 coeffs=[{coeff_summary}]")
    print(f"pose_gp: sigma_dims_1_5=[{sigma_summary}]")
    print(f"pose_gp: reconstructed dim0 RMSE vs baseline={rmse.item():.6f} over {compare_n} pairs")


if __name__ == "__main__":
    main()
