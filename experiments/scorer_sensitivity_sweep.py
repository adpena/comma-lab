#!/usr/bin/env python3
"""Yousfi's scorer-Jacobian sensitivity sweep.

Unlike pixel-level sensitivity (Hotz), this measures how quantization noise
in each layer propagates through the ACTUAL SegNet + PoseNet scorers.

A layer that's pixel-sensitive but scorer-insensitive can safely use fewer bits.
A layer that's pixel-insensitive but scorer-sensitive needs MORE bits.

Usage:
    PYTHONPATH=src:upstream:$PWD python experiments/scorer_sensitivity_sweep.py \
        --checkpoint submissions/robust_current/renderer.bin \
        --device mps --n-pairs 5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "upstream"))


def quantize_to_bits(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return tensor
    n_levels = 2 ** bits
    vmin, vmax = tensor.min(), tensor.max()
    if vmax - vmin < 1e-10:
        return tensor
    scale = (vmax - vmin) / (n_levels - 1)
    quantized = ((tensor - vmin) / scale).round().clamp(0, n_levels - 1)
    return quantized * scale + vmin


def main():
    parser = argparse.ArgumentParser(description="Scorer-Jacobian sensitivity sweep (Yousfi)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--output", default="experiments/results/scorer_sensitivity_sweep.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    from tac.renderer_export import load_any_renderer_checkpoint
    model = load_any_renderer_checkpoint(args.checkpoint, device=str(device))
    if isinstance(model, tuple):
        model = model[0]
    model.eval()

    # Load scorer
    from modules import DistortionNet
    dn = DistortionNet().eval().to(device)
    dn.load_state_dicts(
        Path("upstream/models/posenet.safetensors"),
        Path("upstream/models/segnet.safetensors"),
        device,
    )

    # Load data
    import av
    gt_frames = []
    with av.open("upstream/videos/0.mkv") as container:
        for frame in container.decode(container.streams.video[0]):
            gt_frames.append(frame.to_ndarray(format="rgb24"))
            if len(gt_frames) >= args.n_pairs * 2:
                break

    from tac.mask_codec import decode_masks
    masks = decode_masks("submissions/robust_current/masks_crf50.mkv")

    poses_path = Path("experiments/results/gt_poses.pt")
    poses = torch.load(str(poses_path), map_location="cpu", weights_only=True).float() if poses_path.exists() else None

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Scorer: DistortionNet (SegNet + PoseNet)")
    print(f"Pairs: {args.n_pairs}")
    print()

    param_groups = {name: param for name, param in model.named_parameters() if param.ndim >= 2}
    original_state = {name: param.data.clone() for name, param in param_groups.items()}

    # Measure float baseline scorer distortion
    print("Measuring float baseline...")
    def measure_scorer(mdl):
        pd_sum, sd_sum = 0.0, 0.0
        n = 0
        with torch.inference_mode():
            for i in range(args.n_pairs):
                m_t = masks[2*i:2*i+1].to(device, dtype=torch.long)
                m_t1 = masks[2*i+1:2*i+2].to(device, dtype=torch.long)
                p = poses[i:i+1].to(device) if poses is not None else None
                kw = {"pose": p} if p is not None else {}
                pairs = mdl(m_t, m_t1, **kw)
                chw = pairs[0].permute(0, 3, 1, 2).float().contiguous()
                cam = F.interpolate(chw, size=(874, 1164), mode="bilinear", align_corners=False)
                cam = cam.round().clamp(0, 255).to(torch.uint8).float()
                gt_p = torch.stack([
                    torch.from_numpy(gt_frames[2*i]).float(),
                    torch.from_numpy(gt_frames[2*i+1]).float(),
                ]).unsqueeze(0).to(device)
                comp_p = cam.permute(0, 2, 3, 1).unsqueeze(0).contiguous()
                pd, sd = dn.compute_distortion(gt_p, comp_p)
                pd_sum += pd.sum().item()
                sd_sum += sd.sum().item()
                n += pd.shape[0]
        return pd_sum / max(n, 1), sd_sum / max(n, 1)

    base_pd, base_sd = measure_scorer(model)
    base_score = 100 * base_sd + math.sqrt(10 * base_pd)
    print(f"  Baseline: pose={base_pd:.6f} seg={base_sd:.6f} score={base_score:.4f}")

    # Sweep each layer at each bit-depth
    candidate_bits = [2, 4, 8]  # fewer candidates for speed (scorer eval is expensive)
    results = {}

    for name, param in param_groups.items():
        layer_results = {}
        for bits in candidate_bits:
            # Quantize only this layer
            param.data = quantize_to_bits(original_state[name], bits)
            pd, sd = measure_scorer(model)
            score = 100 * sd + math.sqrt(10 * pd)
            delta = score - base_score
            layer_results[bits] = {
                "pose_d": pd,
                "seg_d": sd,
                "score": score,
                "delta": delta,
            }
            # Restore
            param.data = original_state[name].clone()

        # Scorer sensitivity: how much does 2-bit degrade the SCORE (not pixels)
        scorer_sensitivity = layer_results[2]["delta"] - layer_results[8]["delta"]
        results[name] = {
            "n_params": param.numel(),
            "scorer_sensitivity": scorer_sensitivity,
            "bit_results": layer_results,
        }
        print(f"  {name}: {param.numel():>6} params | "
              f"scorer_sens={scorer_sensitivity:.4f} | "
              f"2b_delta={layer_results[2]['delta']:+.4f} | "
              f"8b_delta={layer_results[8]['delta']:+.4f}")

    # Budget-constrained allocation (knapsack):
    # Target: same or smaller archive than uniform FP4.
    # Strategy: start at 2 bits everywhere, upgrade layers with highest
    # scorer sensitivity until budget is exhausted.
    total_params = sum(p.numel() for p in param_groups.values())
    budget_bits = total_params * 4  # uniform FP4 budget (target: stay at or below)

    print(f"\n{'='*60}")
    print("BUDGET-CONSTRAINED ALLOCATION (knapsack, Yousfi approach):")
    print(f"  Budget: {budget_bits / 8 / 1024:.1f} KB (uniform FP4 equivalent)")

    # Start all at 2 bits
    allocation = {name: 2 for name in results}
    used_bits = sum(results[n]["n_params"] * 2 for n in results)

    # Layers with NEGATIVE sensitivity: keep at 2 bits (quantization HELPS)
    # Layers with positive sensitivity: upgrade in order of sensitivity
    upgradeable = [
        (name, info) for name, info in results.items()
        if info["scorer_sensitivity"] > 0
    ]
    upgradeable.sort(key=lambda x: -x[1]["scorer_sensitivity"])

    upgrade_sequence = [(2, 4), (4, 8)]  # upgrade path: 2→4→8
    for from_bits, to_bits in upgrade_sequence:
        for name, info in upgradeable:
            if allocation[name] != from_bits:
                continue
            extra_bits = info["n_params"] * (to_bits - from_bits)
            if used_bits + extra_bits <= budget_bits:
                allocation[name] = to_bits
                used_bits += extra_bits

    for name in sorted(allocation.keys(), key=lambda n: -results[n]["scorer_sensitivity"]):
        info = results[name]
        print(f"  {name}: {allocation[name]}b "
              f"(scorer_sens={info['scorer_sensitivity']:.5f}, {info['n_params']} params)")

    total_uniform = total_params * 4
    total_mixed = sum(results[n]["n_params"] * allocation[n] for n in allocation)
    print(f"\nUniform FP4: {total_uniform / 8 / 1024:.1f} KB")
    print(f"Scorer-optimal: {total_mixed / 8 / 1024:.1f} KB")
    print(f"Savings: {(total_uniform - total_mixed) / 8 / 1024:.1f} KB "
          f"({(1 - total_mixed / total_uniform) * 100:.1f}%)")

    # Layers that BENEFIT from 2-bit (negative sensitivity)
    beneficial = [(n, results[n]) for n in allocation if results[n]["scorer_sensitivity"] < 0]
    if beneficial:
        print(f"\n  Layers where 2-bit IMPROVES score ({len(beneficial)}):")
        for n, info in sorted(beneficial, key=lambda x: x[1]["scorer_sensitivity"]):
            print(f"    {n}: sens={info['scorer_sensitivity']:.5f} (FREE quality)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "baseline": {"pose_d": base_pd, "seg_d": base_sd, "score": base_score},
        "allocation": allocation,
        "uniform_kb": total_uniform / 8 / 1024,
        "mixed_kb": total_mixed / 8 / 1024,
        "layer_results": {k: {
            "n_params": v["n_params"],
            "scorer_sensitivity": v["scorer_sensitivity"],
            "allocated_bits": allocation[k],
        } for k, v in results.items()},
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
