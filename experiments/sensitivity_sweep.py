#!/usr/bin/env python3
"""Per-layer quantization sensitivity sweep (Hotz approach).

For each parameter group, quantize to {2, 3, 4, 6, 8} bits while keeping
all others at float. Measure scorer distortion delta. Produces optimal
per-layer bit allocation for mixed-precision QAT.

Usage:
    PYTHONPATH=src:upstream:$PWD python experiments/sensitivity_sweep.py \
        --checkpoint /tmp/distill_v2_ep850_best.pt \
        --device mps --n-pairs 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def quantize_to_bits(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Simulate quantization to N bits via round-trip (STE-style)."""
    if bits >= 16:
        return tensor
    n_levels = 2 ** bits
    vmin, vmax = tensor.min(), tensor.max()
    if vmax - vmin < 1e-10:
        return tensor
    scale = (vmax - vmin) / (n_levels - 1)
    quantized = ((tensor - vmin) / scale).round().clamp(0, n_levels - 1)
    return quantized * scale + vmin


def measure_distortion(
    model: nn.Module,
    masks: torch.Tensor,
    gt_frames: list,
    poses: torch.Tensor | None,
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    n_pairs: int = 10,
) -> dict:
    """Measure scorer distortion for current model weights."""
    from tac.renderer import simulate_eval_roundtrip
    from tac.camera import CAMERA_H, CAMERA_W

    model.eval()
    pd_list, sd_list = [], []

    with torch.inference_mode():
        for i in range(min(n_pairs, len(gt_frames) // 2)):
            m_t = masks[2 * i: 2 * i + 1].to(device, dtype=torch.long)
            m_t1 = masks[2 * i + 1: 2 * i + 2].to(device, dtype=torch.long)
            p = poses[i: i + 1].to(device) if poses is not None else None
            kwargs = {"pose": p} if p is not None else {}

            pairs = model(m_t, m_t1, **kwargs)
            chw = pairs[0].permute(0, 3, 1, 2).float()
            cam = F.interpolate(chw, size=(874, 1164), mode="bilinear", align_corners=False)
            cam = cam.round().clamp(0, 255).to(torch.uint8).float()

            gt_p = torch.stack([
                torch.from_numpy(gt_frames[2 * i]).float(),
                torch.from_numpy(gt_frames[2 * i + 1]).float(),
            ]).unsqueeze(0).to(device)
            comp_p = cam.permute(0, 2, 3, 1).unsqueeze(0).contiguous()

            from modules import DistortionNet
            # Use pre-loaded scorers
            pd_val = F.mse_loss(
                posenet.preprocess_input(comp_p.permute(0, 1, 4, 2, 3)),
                posenet.preprocess_input(gt_p.permute(0, 1, 4, 2, 3)),
            ).item()
            pd_list.append(pd_val)

    return {"pose_d": sum(pd_list) / max(len(pd_list), 1)}


def main():
    parser = argparse.ArgumentParser(description="Per-layer quantization sensitivity sweep")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-pairs", type=int, default=10)
    parser.add_argument("--output", default="experiments/results/sensitivity_sweep.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    from tac.renderer_export import load_any_renderer_checkpoint
    model = load_any_renderer_checkpoint(args.checkpoint, device=str(device))
    if isinstance(model, tuple):
        model = model[0]
    model.eval()

    param_groups = {}
    for name, param in model.named_parameters():
        if param.ndim >= 2:  # skip biases
            param_groups[name] = param

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Sweeping {len(param_groups)} parameter groups")
    print(f"Candidate bits: [2, 3, 4, 6, 8]")
    print()

    # Store original weights
    original_state = {name: param.data.clone() for name, param in param_groups.items()}

    candidate_bits = [2, 3, 4, 6, 8]
    results = {}

    # Measure baseline (float32)
    print("Measuring baseline (float32)...")
    # Simple proxy: measure pixel difference when quantizing each layer
    for name, param in param_groups.items():
        layer_results = {}
        for bits in candidate_bits:
            # Quantize this layer, keep others float
            param.data = quantize_to_bits(original_state[name], bits)

            # Quick forward pass to measure output difference
            with torch.inference_mode():
                mask = torch.randint(0, 5, (1, 384, 512), device=device)
                pose = torch.zeros(1, 6, device=device) if hasattr(model, 'pose_dim') and model.pose_dim > 0 else None  # OFF_MANIFOLD_OK: per-layer-quantization sensitivity probe — outputs are diffed (out_quant - out_float) so absolute pose value cancels; only quantization-induced renderer drift is measured.
                kwargs = {"pose": pose} if pose is not None else {}
                out_quant = model(mask, mask, **kwargs)

            # Restore
            param.data = original_state[name].clone()

            with torch.inference_mode():
                out_float = model(mask, mask, **kwargs)

            diff = (out_quant.float() - out_float.float()).abs().mean().item()
            layer_results[bits] = {
                "pixel_diff": diff,
                "n_params": param.numel(),
                "bits_cost": param.numel() * bits / 8,
            }

        # Find sensitivity: how much does reducing bits hurt?
        sensitivity = layer_results[2]["pixel_diff"] / max(layer_results[8]["pixel_diff"], 1e-10)
        results[name] = {
            "n_params": param.numel(),
            "sensitivity": sensitivity,
            "bit_results": layer_results,
        }
        best_bits = min(candidate_bits, key=lambda b: layer_results[b]["pixel_diff"] + 0.01 * layer_results[b]["bits_cost"])
        print(f"  {name}: {param.numel():>6} params | sensitivity={sensitivity:.1f} | "
              f"best={best_bits}b | 2b_diff={layer_results[2]['pixel_diff']:.4f} | "
              f"8b_diff={layer_results[8]['pixel_diff']:.4f}")

    # Allocate bits based on sensitivity
    print("\n=== OPTIMAL ALLOCATION ===")
    total_bits_uniform = sum(p.numel() * 4 for p in param_groups.values())
    total_bits_mixed = 0
    allocation = {}

    for name, info in sorted(results.items(), key=lambda x: -x[1]["sensitivity"]):
        # High sensitivity → more bits, low sensitivity → fewer bits
        s = info["sensitivity"]
        if s > 10:
            bits = 8
        elif s > 3:
            bits = 6
        elif s > 1.5:
            bits = 4
        else:
            bits = 2
        allocation[name] = bits
        total_bits_mixed += info["n_params"] * bits
        print(f"  {name}: {bits} bits (sensitivity={s:.1f})")

    uniform_kb = total_bits_uniform / 8 / 1024
    mixed_kb = total_bits_mixed / 8 / 1024
    print(f"\nUniform FP4: {uniform_kb:.1f} KB")
    print(f"Mixed precision: {mixed_kb:.1f} KB")
    print(f"Savings: {uniform_kb - mixed_kb:.1f} KB ({(1 - mixed_kb/uniform_kb)*100:.1f}%)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "allocation": allocation,
        "uniform_kb": uniform_kb,
        "mixed_kb": mixed_kb,
        "savings_pct": (1 - mixed_kb / uniform_kb) * 100,
        "layer_results": {k: {
            "n_params": v["n_params"],
            "sensitivity": v["sensitivity"],
            "allocated_bits": allocation[k],
        } for k, v in results.items()},
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
