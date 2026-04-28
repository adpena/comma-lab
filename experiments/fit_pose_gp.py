#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tac.pose_gaussian_process import fit_pose_gp, reconstruct_poses, save_pose_gp


def _load_pose_tensor(path: Path) -> torch.Tensor:
    data = torch.load(str(path), map_location="cuda", weights_only=True)
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
    reconstructed = reconstruct_poses(model, int(args.n_pairs)).to(baseline.device)
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
