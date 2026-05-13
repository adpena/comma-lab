#!/usr/bin/env python3
"""INT4 smoke test: quantize dilated h=64 model to uniform 4-bit and compare with INT8.

Technique 5 from cross-cultural research survey.

This script:
1. Loads the best dilated h=64 INT8 checkpoint
2. Dequantizes to FP32
3. Re-quantizes to INT4 (uniform symmetric 4-bit, 15 levels)
4. Compares parameter distributions, output quality, and file sizes

Usage:
    python -m tac.experiments.benchmark_int4 [--checkpoint PATH]
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from tac.architectures import build_postfilter

# R41 fix: PROJECT_ROOT was used in CLI defaults without being defined.
# Resolve from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── INT4 Quantization (uniform symmetric) ─────────────────────────────


def quantize_int4_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to symmetric INT4 (4-bit, 15 levels: -7 to +7).

    Per-channel quantization: each output channel gets its own scale.

    Args:
        weight: (out_ch, ...) float tensor

    Returns:
        (quantized_int4, scales) where quantized is in [-7, 7] and
        scales is per-channel float
    """
    if weight.ndim < 2:
        # Per-tensor for 1D (bias)
        amax = weight.abs().max().clamp(min=1e-10)
        scale = amax / 7.0
        quantized = (weight / scale).round().clamp(-7, 7)
        return quantized, scale.unsqueeze(0)

    flat = weight.detach().reshape(weight.shape[0], -1)
    amax = flat.abs().amax(dim=1).clamp(min=1e-10)
    scale = amax / 7.0  # (out_ch,)
    scale_view = scale.reshape(-1, *([1] * (weight.ndim - 1)))
    quantized = (weight / scale_view).round().clamp(-7, 7)
    return quantized, scale


def dequantize_int4_per_channel(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT4 back to float."""
    if quantized.ndim < 2:
        return quantized * scale[0]
    scale_view = scale.reshape(-1, *([1] * (quantized.ndim - 1)))
    return quantized * scale_view


def quantize_model_int4(model: nn.Module) -> tuple[nn.Module, dict]:
    """Quantize all Conv2d/Linear weights in model to INT4.

    Returns:
        (quantized_model, stats) where stats contains per-layer info
    """
    stats = {}
    total_params = 0
    total_error = 0.0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            q, s = quantize_int4_per_channel(w)
            w_recon = dequantize_int4_per_channel(q, s)

            # Compute quantization error
            mse = (w - w_recon).pow(2).mean().item()
            max_err = (w - w_recon).abs().max().item()
            n_params = w.numel()

            stats[name] = {
                "shape": list(w.shape),
                "params": n_params,
                "mse": mse,
                "max_error": max_err,
                "scale_range": (s.min().item(), s.max().item()),
            }

            total_params += n_params
            total_error += mse * n_params

            # Replace with dequantized weights
            module.weight.data = w_recon

            # Also quantize bias if present
            if module.bias is not None:
                b = module.bias.data
                bq, bs = quantize_int4_per_channel(b)
                b_recon = dequantize_int4_per_channel(bq, bs)
                module.bias.data = b_recon

    stats["_summary"] = {
        "total_params": total_params,
        "avg_mse": total_error / max(total_params, 1),
    }

    return model, stats


def save_int4_checkpoint(model: nn.Module, path: str | Path) -> int:
    """Save model with INT4-packed weights.

    Packs weights as int4 (nibbles) for size measurement.
    Format: per-channel scales (fp16) + packed int4 weights (nibble pairs).

    Returns:
        File size in bytes
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    packed_data = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            q, s = quantize_int4_per_channel(w)

            # Pack two int4 values per byte
            q_flat = (q.reshape(-1).to(torch.int8) + 7).to(torch.uint8)  # shift to [0, 14]
            if q_flat.numel() % 2 != 0:
                q_flat = torch.cat([q_flat, torch.zeros(1, dtype=torch.uint8)])
            packed = (q_flat[0::2] << 4) | q_flat[1::2]

            packed_data[f"{name}.weight.packed"] = packed
            packed_data[f"{name}.weight.scale"] = s.half()
            packed_data[f"{name}.weight.shape"] = torch.tensor(list(w.shape))

            if module.bias is not None:
                packed_data[f"{name}.bias"] = module.bias.data.half()

    torch.save(packed_data, path)
    return path.stat().st_size


def compute_output_difference(
    model_a: nn.Module,
    model_b: nn.Module,
    input_tensor: torch.Tensor,
) -> dict:
    """Compare outputs of two models on the same input.

    Returns:
        Dict with MSE, PSNR, max absolute difference
    """
    with torch.no_grad():
        out_a = model_a(input_tensor)
        out_b = model_b(input_tensor)

    diff = (out_a - out_b).float()
    mse = diff.pow(2).mean().item()
    max_diff = diff.abs().max().item()
    # PSNR assumes pixel values in [0, 255] range (MAX_I = 255)
    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10)) if mse > 0 else float("inf")

    return {
        "mse": mse,
        "psnr_db": psnr,
        "max_abs_diff": max_diff,
        "mean_abs_diff": diff.abs().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="INT4 vs INT8 benchmark")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "submissions/robust_current/postfilter_int8.pt"),
        help="Path to INT8 checkpoint",
    )
    parser.add_argument("--variant", default="dilated", help="Architecture variant")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of random frames to test")
    args = parser.parse_args()

    print("=" * 70)
    print("INT4 vs INT8 Quantization Benchmark")
    print("=" * 70)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Looking for alternative checkpoints...")
        alternatives = list(PROJECT_ROOT.glob("**/*int8*.pt"))
        for alt in alternatives[:5]:
            print(f"  Found: {alt}")
        if alternatives:
            checkpoint_path = alternatives[0]
            print(f"Using: {checkpoint_path}")
        else:
            print("No INT8 checkpoints found. Building fresh model for benchmark.")
            checkpoint_path = None

    # Build models
    model_int8 = build_postfilter(args.variant, hidden=args.hidden)
    model_int4 = build_postfilter(args.variant, hidden=args.hidden)

    if checkpoint_path and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # INT8 checkpoint: dequantize
        # Supports two formats:
        #   1. key_scale / key (legacy)
        #   2. key.q / key.s (current)
        fp32_state = {}

        # Detect format
        has_dot_q = any(k.endswith(".q") for k in state if isinstance(state[k], torch.Tensor))

        if has_dot_q:
            # Format: param.q (int8 quantized) + param.s (scale)
            processed = set()
            for key in state:
                if key.startswith("__"):
                    continue
                if key.endswith(".q"):
                    base = key[:-2]  # strip .q
                    processed.add(base)
                    q = state[key].float()
                    s_key = base + ".s"
                    if s_key in state:
                        s = state[s_key].float()
                        if s.ndim == 0:
                            fp32_state[base] = q * s
                        elif q.ndim >= 2:
                            s_view = s.reshape(-1, *([1] * (q.ndim - 1)))
                            fp32_state[base] = q * s_view
                        else:
                            fp32_state[base] = q * s
                    else:
                        fp32_state[base] = q
        else:
            # Legacy format: key_scale / key
            for key, val in state.items():
                if key.startswith("__") or not isinstance(val, torch.Tensor):
                    continue
                if key.endswith("_scale"):
                    continue
                if key + "_scale" in state:
                    scale = state[key + "_scale"]
                    if val.ndim >= 2:
                        scale_view = scale.reshape(-1, *([1] * (val.ndim - 1)))
                        fp32_state[key] = val.float() * scale_view
                    else:
                        fp32_state[key] = val.float() * scale
                else:
                    fp32_state[key] = val.float()

        info = model_int8.load_state_dict(fp32_state, strict=False)
        if info.missing_keys:
            print(f"  Warning: {len(info.missing_keys)} missing keys: {info.missing_keys[:5]}")
        if info.unexpected_keys:
            print(f"  Warning: {len(info.unexpected_keys)} unexpected keys: {info.unexpected_keys[:5]}")
        model_int4.load_state_dict(fp32_state, strict=False)
        print(f"\nLoaded checkpoint: {checkpoint_path}")
    else:
        print("\nUsing randomly initialized model (no checkpoint)")

    # Count parameters
    total_params = sum(p.numel() for p in model_int8.parameters())
    print(f"Model: {args.variant} h={args.hidden}, {total_params:,} params")

    # Quantize to INT4
    print("\n--- INT4 Quantization ---")
    model_int4, int4_stats = quantize_model_int4(model_int4)
    print(f"Total params: {int4_stats['_summary']['total_params']:,}")
    print(f"Average MSE: {int4_stats['_summary']['avg_mse']:.6e}")

    for name, info in sorted(int4_stats.items()):
        if name.startswith("_"):
            continue
        print(f"  {name}: shape={info['shape']}, MSE={info['mse']:.6e}, "
              f"max_err={info['max_error']:.4f}")

    # File size comparison
    print("\n--- File Size Comparison ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # INT8 size
        int8_path = Path(tmpdir) / "model_int8.pt"
        int8_state = {}
        for name, module in model_int8.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight.data
                if w.ndim >= 2:
                    flat = w.reshape(w.shape[0], -1)
                    scale = flat.abs().amax(dim=1).clamp(min=1e-10) / 127.0
                    scale_view = scale.reshape(-1, *([1] * (w.ndim - 1)))
                    q = (w / scale_view).round().clamp(-128, 127).to(torch.int8)
                else:
                    scale = w.abs().max().clamp(min=1e-10) / 127.0
                    q = (w / scale).round().clamp(-128, 127).to(torch.int8)
                int8_state[f"{name}.weight"] = q
                int8_state[f"{name}.weight_scale"] = scale.half() if w.ndim >= 2 else scale.half()
                if module.bias is not None:
                    int8_state[f"{name}.bias"] = module.bias.data.half()
        torch.save(int8_state, int8_path)
        int8_size = int8_path.stat().st_size

        # INT4 size
        int4_path = Path(tmpdir) / "model_int4.pt"
        int4_size = save_int4_checkpoint(model_int4, int4_path)

        print(f"INT8 checkpoint: {int8_size:,} bytes ({int8_size/1024:.1f} KB)")
        print(f"INT4 checkpoint: {int4_size:,} bytes ({int4_size/1024:.1f} KB)")
        print(f"Reduction: {(1 - int4_size/int8_size)*100:.1f}%")

    # Theoretical sizes
    int8_theoretical = total_params * 1  # 1 byte per param
    int4_theoretical = total_params * 0.5  # 0.5 bytes per param (nibble packing)
    print("\nTheoretical sizes (weights only):")
    print(f"  INT8: {int8_theoretical:,} bytes ({int8_theoretical/1024:.1f} KB)")
    print(f"  INT4: {int4_theoretical:,} bytes ({int4_theoretical/1024:.1f} KB)")

    # Output quality comparison
    print(f"\n--- Output Quality (random {args.num_frames} frames) ---")
    # Use 256x192 for quick benchmark (quarter of SegNet resolution)
    H, W = 256, 192
    test_input = torch.rand(args.num_frames, 3, H, W) * 255.0

    quality = compute_output_difference(model_int8, model_int4, test_input)
    print(f"MSE: {quality['mse']:.4f}")
    print(f"PSNR: {quality['psnr_db']:.1f} dB")
    print(f"Max absolute diff: {quality['max_abs_diff']:.2f} / 255")
    print(f"Mean absolute diff: {quality['mean_abs_diff']:.2f} / 255")

    # Quality assessment
    print("\n--- Assessment ---")
    if quality["psnr_db"] > 40:
        print("EXCELLENT: INT4 quality degradation is negligible (>40dB PSNR)")
    elif quality["psnr_db"] > 30:
        print("GOOD: INT4 quality loss is minor (30-40dB PSNR)")
    elif quality["psnr_db"] > 20:
        print("MARGINAL: INT4 quality loss is noticeable (20-30dB PSNR)")
    else:
        print("POOR: INT4 quality loss is severe (<20dB PSNR)")

    rate_saving_kb = (int8_theoretical - int4_theoretical) / 1024
    rate_weight = 25  # contest formula multiplier for rate term
    n_frames = 1200  # video frame count in contest evaluation
    frame_h = 874  # frame height (pixels)
    frame_w = 1164  # frame width (pixels)
    n_channels = 3  # RGB channels
    total_pixels = n_frames * frame_w * frame_h * n_channels
    print(f"\nRate impact: INT4 saves ~{rate_saving_kb:.1f} KB in archive")
    print(f"At {rate_weight}*rate scoring: saving ~{rate_weight * (int8_theoretical - int4_theoretical) / total_pixels:.6f} score points")

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
