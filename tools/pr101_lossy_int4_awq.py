#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR101 lossy int4 + AWQ (Activation-aware Weight Quantization)
— audit criterion #5 for the lossy_int4 lane.

Per the 2026-05-08 audit memo (`feedback_implementation_vs_model_gap_audit_20260508.md`):

    - naive PTQ at int4: 37.42% rel_err (criterion #0 not dispatchable)
    - QAT at int4: 28.48% rel_err (criterion #1 not dispatchable)
    - per-channel scales: 30.41% rel_err (criterion #2 not dispatchable)
    - mixed-precision int4/int6/int8: 4.895% rel_err (criterion #3 conditional)
    - GPTQ Hessian-aware: 40.11% rel_err (criterion #4 not dispatchable)
    - AWQ activation-aware scaling: NOT TESTED (this tool — criterion #5)

AWQ (Lin et al. 2023, "AWQ: Activation-Aware Weight Quantization for
On-Device LLM Compression and Acceleration", https://arxiv.org/abs/2306.00978)
identifies salient input channels by their activation magnitudes and applies
PER-INPUT-CHANNEL scaling factors to the weight matrix before quantization:

    W'[:, j] = W[:, j] * s_j
    W'_q  = quant(W')
    W_recon[:, j] = W'_q[:, j] / s_j

Salient channels (high activation magnitude) get larger s_j, which expands
their range in the quantization grid — yielding finer effective resolution
on the channels that matter most for downstream output. The unscaling at
recon time is mathematically equivalent (modulo quantization error) but the
ERROR distribution is shifted: salient channels see less relative error.

The optimal scale per channel is parameterized as:

    s_j = (mean(|x_j|))^α  with α ∈ [0, 1]

which is grid-searched over a discrete set of α values per layer. α=0 is
the naive PTQ baseline (no scaling); α=1 is full activation-magnitude
scaling. The paper finds α near 0.5-0.8 typically optimal for transformer
weights; we replicate that grid-search per-tensor.

For PR101's substrate

    AWQ was designed for transformer LLMs where activation magnitudes
    correlate with channel importance. PR101 is a per-video overfit
    HNeRVDecoder — the activation magnitudes do still indicate which
    input channels carry signal at each layer, but the salience-vs-
    importance correlation is empirically untested for this substrate.
    We use SYNTHETIC activations from the renderer's forward pass on
    deterministic latents (uniform[-1, 1]^28 with a fixed seed) for
    activation-magnitude profiling. This is the closest available
    substrate-faithful calibration set; the substrate gap (synthetic vs
    PR106-video latents) is itself part of the result.

CLAUDE.md compliance (NON-NEGOTIABLES)

    * Pure CPU prep — no scorer load, no contest-CUDA dispatch from
      this tool. Evidence tagged ``[CPU-prep faithful audit-criterion-5 test]``.
    * Score never claimed; only BYTES + ROUNDTRIP rel_err are anchored.
    * promotion_eligible=False and ready_for_exact_eval_dispatch=False always.
      A good local rel_err only means cuda_eval_worth_testing=True; a separate
      byte-closed packet, runtime decoder, dispatch claim, and exact CUDA auth
      eval must flip any dispatch flag.
    * No /tmp paths in any persisted artifact.
    * Never invent CLI flags — uses the same argparse surface conventions
      as the prior 4 lossy_int4 variants.
    * This tool only adjudicates the measured AWQ config. It does not
      falsify the lossy-int4 family or downstream byte-closed-runtime
      lanes; only this measured (calibration_size, alpha grid) point is
      anchored.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA  # noqa: E402

TOOL_NAME = "tools/pr101_lossy_int4_awq.py"
SCHEMA_VERSION = "pr101_lossy_int4_awq.v1"
INT4_RANGE = 7
DEFAULT_BLOCK_SIZE = 1024
ARCHIVE_OVERHEAD_BYTES = 16_094
PR101_BROTLI_BASELINE_BYTES = 178_144
DISPATCH_THRESHOLD_PCT = 5.0
EVIDENCE_GRADE = "[CPU-prep faithful audit-criterion-5 test]"
DEFAULT_STATE_DICT_PATH = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)
DEFAULT_CALIBRATION_SAMPLES = 64
DEFAULT_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LOCAL_AWQ_DISPATCH_BLOCKERS = (
    "local_awq_proxy_signal_not_exact_eval_dispatch_ready",
    "byte_closed_int4_candidate_packet_missing",
    "no_int4_decoder_runtime_built",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "synthetic_activations_substrate_mismatch_vs_pr106_video_latents",
    "cpu_proxy_rel_err_not_score_evidence",
    "awq_inverse_scaling_assumes_linear_runtime_absorption_not_yet_built",
)


# ----------------------------------------------------------------------------
# Utility (mirrors patterns from pr101_lossy_int4_gptq.py for consistency)
# ----------------------------------------------------------------------------

def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_device(prefer: str = "cpu") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def local_awq_dispatch_contract(*, cuda_eval_worth_testing: bool) -> dict[str, Any]:
    """Return fail-closed dispatch metadata for local AWQ research signals."""
    return {
        "cuda_eval_worth_testing": bool(cuda_eval_worth_testing),
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "dispatch_blockers": list(LOCAL_AWQ_DISPATCH_BLOCKERS),
    }


# ----------------------------------------------------------------------------
# HNeRVDecoder — exact replica of submissions/pr103_pr106_final_runtime/inflate.py
# ----------------------------------------------------------------------------

class HNeRVDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        base_channels: int = BASE_CHANNELS,
        eval_size: tuple[int, int] = EVAL_SIZE,
    ) -> None:
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        c = base_channels
        self.channels = [c, c, c, int(c * 0.75), int(c * 0.58), int(c * 0.5), int(c * 0.5)]
        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(6):
            in_ch, out_ch = self.channels[i], self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        x = self.stem(z).view(b, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips, strict=True):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


def load_state_into_decoder(state: dict[str, torch.Tensor]) -> HNeRVDecoder:
    decoder = HNeRVDecoder()
    expected_state = decoder.state_dict()
    payload: dict[str, torch.Tensor] = {}
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        if name not in expected_state:
            raise SystemExit(f"HNeRVDecoder has no parameter {name!r} (schema drift)")
        payload[name] = state[name].to(torch.float32)
    decoder.load_state_dict(payload, strict=False)
    decoder.eval()
    return decoder


# ----------------------------------------------------------------------------
# Calibration: per-input-channel activation magnitudes (mean abs).
# For Linear: input is (B, in_features). For Conv2d: input after F.unfold is
# (B*L, in_ch * kH * kW). We aggregate by INPUT POSITION (col index of the
# flattened reshape) to match the AWQ formulation: one scale per "input
# channel" of the lowered weight matrix.
# ----------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    n_samples: int = DEFAULT_CALIBRATION_SAMPLES
    seed: int = 2026
    latent_low: float = -1.0
    latent_high: float = 1.0


def collect_activation_magnitudes(
    decoder: HNeRVDecoder,
    cfg: CalibrationConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return a dict ``"<module_name>.weight" -> per-input-channel magnitude``
    where each entry is an array of shape ``(in_features_per_position,)``
    containing ``mean(|x_j|)`` over all calibration samples (rows).

    For Conv2d we use F.unfold so the per-input-channel index matches the
    weight matrix's column index in the lowered ``W @ X^T`` formulation that
    AWQ assumes.
    """
    rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
    z = (
        cfg.latent_low
        + (cfg.latent_high - cfg.latent_low)
        * torch.rand(cfg.n_samples, LATENT_DIM, generator=rng)
    ).to(device)

    abs_sum: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            x = inputs[0]
            if isinstance(module, nn.Linear):
                rows = x.detach().reshape(-1, x.shape[-1])
            elif isinstance(module, nn.Conv2d):
                kH, kW = module.kernel_size
                pH = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                pW = module.padding[1] if isinstance(module.padding, tuple) else module.padding
                sH = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                sW = module.stride[1] if isinstance(module.stride, tuple) else module.stride
                dH = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
                dW = module.dilation[1] if isinstance(module.dilation, tuple) else module.dilation
                unfolded = F.unfold(
                    x.detach(),
                    kernel_size=(kH, kW),
                    padding=(pH, pW),
                    stride=(sH, sW),
                    dilation=(dH, dW),
                )
                rows = unfolded.permute(0, 2, 1).reshape(-1, unfolded.shape[1])
            else:  # pragma: no cover
                return
            rows_np = rows.cpu().numpy().astype(np.float64)
            key = f"{name}.weight"
            row_abs_sum = np.abs(rows_np).sum(axis=0)
            if key in abs_sum:
                abs_sum[key] = abs_sum[key] + row_abs_sum
                counts[key] = counts[key] + rows_np.shape[0]
            else:
                abs_sum[key] = row_abs_sum
                counts[key] = rows_np.shape[0]
        return hook

    for name, module in decoder.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    decoder.eval()
    with torch.no_grad():
        decoder(z)

    for h in handles:
        h.remove()

    out: dict[str, np.ndarray] = {}
    for k, total_abs in abs_sum.items():
        n = max(counts[k], 1)
        out[k] = (total_abs / n).astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# Encoder math: same per-block int4 grid as pr101_lossy_int4_block_sweep.
# ----------------------------------------------------------------------------

def quantize_block_int4_with_scale(block: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-block symmetric int4 quant. Returns (codes [-7,+7], scale_fp16_as_fp32)."""
    abs_max = float(np.abs(block).max())
    if abs_max <= 0.0:
        scale = float(np.float16(1e-6))
        codes = np.zeros_like(block, dtype=np.int8)
        return codes, scale
    scale_f = abs_max / INT4_RANGE
    scale_fp16 = float(np.float16(scale_f))
    codes_f = block / scale_fp16
    codes = np.clip(np.round(codes_f).astype(np.int8), -INT4_RANGE, +INT4_RANGE)
    return codes.astype(np.int8), scale_fp16


def naive_int4_quantize_tensor(tensor: np.ndarray, block_size: int) -> np.ndarray:
    """Per-block symmetric int4 quant + dequant, no calibration."""
    flat = tensor.astype(np.float32).reshape(-1)
    n = flat.size
    n_full = n // block_size
    tail = n - n_full * block_size
    out = np.zeros_like(flat)
    if n_full > 0:
        for b in range(n_full):
            block = flat[b * block_size : (b + 1) * block_size]
            codes, scale = quantize_block_int4_with_scale(block)
            out[b * block_size : (b + 1) * block_size] = codes.astype(np.float32) * scale
    if tail:
        block = flat[n_full * block_size :]
        codes, scale = quantize_block_int4_with_scale(block)
        out[n_full * block_size :] = codes.astype(np.float32) * scale
    return out.reshape(tensor.shape).astype(np.float32)


# ----------------------------------------------------------------------------
# AWQ core
# ----------------------------------------------------------------------------

@dataclass
class AWQConfig:
    block_size_encoder: int = DEFAULT_BLOCK_SIZE
    alpha_grid: tuple[float, ...] = tuple(DEFAULT_ALPHA_GRID)
    eps: float = 1e-6
    # The per-tensor weight prior on the loss: 'uniform' weights every weight
    # element equally; 'salient' weights by activation magnitude. AWQ paper
    # uses uniform weight (the salience comes in via the SCALING, not the
    # loss); we follow that.
    error_metric: str = "uniform"


def compute_awq_scales(
    activation_mag: np.ndarray, alpha: float, eps: float = 1e-6,
) -> np.ndarray:
    """Compute per-input-channel scales s_j = (mean|x_j|)^alpha.

    Normalize so that the geometric mean of s is 1 — this preserves the
    overall weight magnitude (and hence the quantization range) while
    redistributing fidelity across input channels. AWQ's reference impl
    normalizes by the sqrt of the product of clamped scales; we use the
    geometric-mean form which is mathematically equivalent up to sign.
    """
    mag = np.maximum(activation_mag.astype(np.float64), eps)
    if alpha == 0.0:
        return np.ones_like(mag, dtype=np.float64)
    s = mag ** alpha
    # Clamp at the bottom to avoid pathological zero-scaling
    s = np.maximum(s, eps)
    # Geometric-mean normalization: log mean of s = 0
    log_mean = float(np.log(s).mean())
    s = s / np.exp(log_mean)
    return s


def awq_quantize_tensor(
    weight: np.ndarray,  # (out_features, in_features)
    activation_mag: np.ndarray,  # (in_features,)
    *,
    cfg: AWQConfig,
    original_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run AWQ on ONE weight tensor: grid-search alpha over cfg.alpha_grid,
    pick the one with the lowest weight-roundtrip MSE, return the
    AWQ-recovered weight tensor (shape (out_features, in_features)).

    The recon path:
        W' = W * s   (elementwise per-column scaling)
        W'_packed = encode_int4(W')          -- per the encoder's grid
        W'_recon = decode_int4(W'_packed)    -- exactly the dequantized form
        W_recon = W'_recon / s

    Note: the int4 grid scales per BLOCK in the encoder's flattened order
    (which interleaves columns). When we scale columns and re-encode, the
    block-level max-abs values change; the encoder's scale grid adapts.
    This is the key AWQ insight: by scaling salient columns UP, we shift
    block-level max-abs to those columns, which (after divide-by-s) yields
    smaller residual error for them. The non-salient columns end up with
    coarser quantization but they matter less.

    We assume the inverse-scaling at recon time can be absorbed into the
    runtime decoder (or into the previous layer's bias/weight at runtime
    setup). The dispatch contract notes this as a blocker until the
    runtime decoder is built.
    """
    out_features, in_features = weight.shape
    if activation_mag.shape[0] != in_features:
        raise ValueError(
            f"activation magnitudes have {activation_mag.shape[0]} cols but "
            f"weight has {in_features} input features"
        )

    best_alpha: float | None = None
    best_loss: float = float("inf")
    best_recon: np.ndarray | None = None
    losses: dict[float, float] = {}

    for alpha in cfg.alpha_grid:
        s = compute_awq_scales(activation_mag, alpha, cfg.eps)
        # Scale columns of W (broadcast (1, in_features) over rows)
        w_scaled = weight.astype(np.float64) * s[None, :]
        # Reshape to original tensor shape for encoder-grid quantization
        w_scaled_full = w_scaled.reshape(original_shape).astype(np.float32)
        w_quant = naive_int4_quantize_tensor(w_scaled_full, cfg.block_size_encoder)
        # Reshape back to (out, in_features) and unscale
        w_quant_2d = w_quant.reshape(out_features, in_features).astype(np.float64)
        w_recon = w_quant_2d / s[None, :]
        # Loss: MSE in the un-scaled space (matches what the wire format
        # actually reconstructs at recon time).
        if cfg.error_metric == "uniform":
            err = float(np.mean((w_recon - weight.astype(np.float64)) ** 2))
        elif cfg.error_metric == "salient":
            # Weight by activation magnitude squared — gives more weight to
            # salient channels in the loss.
            w_per_col = activation_mag.astype(np.float64) ** 2
            w_per_col = w_per_col / max(float(w_per_col.sum()), cfg.eps)
            sq_err_per_col = np.mean((w_recon - weight.astype(np.float64)) ** 2, axis=0)
            err = float(np.sum(w_per_col * sq_err_per_col))
        else:
            raise ValueError(f"unknown error_metric {cfg.error_metric!r}")
        losses[alpha] = err
        if err < best_loss:
            best_loss = err
            best_alpha = alpha
            best_recon = w_recon.astype(np.float32)
    assert best_recon is not None and best_alpha is not None
    return best_recon, {
        "alpha_grid": list(cfg.alpha_grid),
        "best_alpha": best_alpha,
        "losses_per_alpha": losses,
        "n_columns_in": int(in_features),
        "n_rows_out": int(out_features),
        "error_metric": cfg.error_metric,
    }


# ----------------------------------------------------------------------------
# Per-tensor dispatch: AWQ for weights, naive PTQ for biases.
# ----------------------------------------------------------------------------

def quantize_pr101_state(
    fp32_state: dict[str, np.ndarray],
    activation_magnitudes: dict[str, np.ndarray],
    *,
    cfg: AWQConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Apply AWQ to all weight tensors and naive PTQ to bias tensors."""
    quantized: dict[str, np.ndarray] = {}
    per_tensor_stats: dict[str, dict[str, Any]] = {}

    for name, shape in FIXED_STATE_SCHEMA:
        tensor = fp32_state[name]
        if name.endswith(".bias") or tensor.ndim <= 1:
            quantized[name] = naive_int4_quantize_tensor(tensor, cfg.block_size_encoder)
            per_tensor_stats[name] = {"path": "naive_ptq_bias", "shape": list(shape)}
            continue
        if name not in activation_magnitudes:
            quantized[name] = naive_int4_quantize_tensor(tensor, cfg.block_size_encoder)
            per_tensor_stats[name] = {"path": "naive_ptq_no_calibration", "shape": list(shape)}
            continue
        if tensor.ndim == 4:
            out_ch = tensor.shape[0]
            w2d = tensor.reshape(out_ch, -1).astype(np.float32)
        elif tensor.ndim == 2:
            w2d = tensor.astype(np.float32)
        else:
            quantized[name] = naive_int4_quantize_tensor(tensor, cfg.block_size_encoder)
            per_tensor_stats[name] = {"path": "naive_ptq_unsupported_ndim", "shape": list(shape)}
            continue
        w_q, stats = awq_quantize_tensor(
            w2d, activation_magnitudes[name],
            cfg=cfg, original_shape=tuple(shape),
        )
        quantized[name] = w_q.reshape(shape).astype(np.float32)
        per_tensor_stats[name] = {"path": "awq", "shape": list(shape), **stats}
    return quantized, per_tensor_stats


# ----------------------------------------------------------------------------
# Encode / measure
# ----------------------------------------------------------------------------

def encode_quantized_state_to_archive_bytes(
    quantized: dict[str, np.ndarray],
    *,
    block_size: int,
) -> tuple[int, int, int]:
    from pr101_lossy_int4_block_sweep import encode_tensor

    payloads: list[bytes] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        flat = quantized[name].astype(np.float32).flatten()
        payload, _stats = encode_tensor(flat, block_size=block_size)
        payloads.append(payload)
    full_payload = b"".join(payloads)
    compressed = brotli.compress(
        full_payload, quality=11, lgwin=16, lgblock=19, mode=brotli.MODE_GENERIC
    )
    archive_bytes = len(compressed) + ARCHIVE_OVERHEAD_BYTES
    return len(full_payload), len(compressed), archive_bytes


def measure_roundtrip_rel_err(
    quantized: dict[str, np.ndarray],
    originals_fp32: dict[str, np.ndarray],
) -> dict:
    per_tensor: list[dict] = []
    total_elements = 0
    weighted_rel_err_sum = 0.0
    weighted_abs_err_sum = 0.0
    n_nontrivial_total = 0
    max_p99 = 0.0
    max_max = 0.0

    for name, _shape in FIXED_STATE_SCHEMA:
        orig = originals_fp32[name].flatten().astype(np.float64)
        recon = quantized[name].astype(np.float32).flatten().astype(np.float64)
        abs_err = np.abs(recon - orig)
        eps = 1e-8
        mask = np.abs(orig) > eps
        rel_err_pct = np.zeros_like(abs_err)
        rel_err_pct[mask] = 100.0 * abs_err[mask] / np.abs(orig[mask])

        n_elements = int(orig.size)
        n_nontrivial = int(mask.sum())
        stats = {
            "name": name,
            "n_elements": n_elements,
            "n_nontrivial": n_nontrivial,
            "abs_err_mean": float(abs_err.mean()),
            "abs_err_max": float(abs_err.max()),
            "rel_err_pct_mean": float(rel_err_pct[mask].mean()) if mask.any() else 0.0,
            "rel_err_pct_p50": float(np.percentile(rel_err_pct[mask], 50)) if mask.any() else 0.0,
            "rel_err_pct_p90": float(np.percentile(rel_err_pct[mask], 90)) if mask.any() else 0.0,
            "rel_err_pct_p99": float(np.percentile(rel_err_pct[mask], 99)) if mask.any() else 0.0,
            "rel_err_pct_max": float(rel_err_pct[mask].max()) if mask.any() else 0.0,
        }
        per_tensor.append(stats)
        total_elements += n_elements
        n_nontrivial_total += n_nontrivial
        weighted_rel_err_sum += stats["rel_err_pct_mean"] * n_nontrivial
        weighted_abs_err_sum += stats["abs_err_mean"] * n_elements
        max_p99 = max(max_p99, stats["rel_err_pct_p99"])
        max_max = max(max_max, stats["rel_err_pct_max"])

    return {
        "n_total_elements": total_elements,
        "n_nontrivial_elements": n_nontrivial_total,
        "weighted_avg_rel_err_pct": weighted_rel_err_sum / max(n_nontrivial_total, 1),
        "weighted_avg_abs_err": weighted_abs_err_sum / max(total_elements, 1),
        "max_p99_rel_err_pct": max_p99,
        "max_max_rel_err_pct": max_max,
        "per_tensor": per_tensor,
    }


def classify_cpu_proxy_candidate(
    *,
    weighted_avg_rel_err_pct: float,
    archive_bytes: int,
) -> tuple[str, bool]:
    if archive_bytes >= PR101_BROTLI_BASELINE_BYTES:
        if weighted_avg_rel_err_pct >= DISPATCH_THRESHOLD_PCT:
            return "MEASURED_CONFIG_DOMINATED_AND_LOSSY", False
        return "MEASURED_CONFIG_DOMINATED_BY_PR101_BROTLI_BASELINE", False
    if weighted_avg_rel_err_pct < 2.0:
        return "CUDA-EVAL-WORTH-TESTING", True
    if weighted_avg_rel_err_pct < DISPATCH_THRESHOLD_PCT:
        return "CONDITIONAL-CUDA-EVAL-WORTH-TESTING", True
    return "MEASURED_CONFIG_NOT_DISPATCHABLE", False


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT_PATH)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES)
    p.add_argument("--alpha-grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID,
                   help="AWQ alpha values to grid-search per-tensor")
    p.add_argument("--error-metric", choices=["uniform", "salient"], default="uniform",
                   help="AWQ inner-loss weighting: uniform = paper standard")
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--evidence-jsonl", type=Path,
                   default=REPO_ROOT / "reports/cathedral_autopilot_evidence.jsonl")
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device("cpu") if args.device == "auto" else select_device(args.device)
    print(f"[AWQ] device: {device}")

    sd = torch.load(args.state_dict, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {args.state_dict} is not a dict")

    fp32_state: dict[str, np.ndarray] = {}
    fp32_state_torch: dict[str, torch.Tensor] = {}
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in sd:
            raise SystemExit(f"state_dict missing {name!r}")
        t = sd[name].detach().cpu().to(torch.float32)
        fp32_state_torch[name] = t.clone()
        fp32_state[name] = t.numpy().copy()

    # ------------------------------------------------------------------
    # Build decoder + collect per-input-channel activation magnitudes.
    # ------------------------------------------------------------------
    print(f"[AWQ] building HNeRVDecoder + collecting {args.calibration_samples} synthetic-latent activation magnitudes")
    decoder = load_state_into_decoder(fp32_state_torch).to(device)
    cal_cfg = CalibrationConfig(n_samples=args.calibration_samples, seed=args.seed)
    activation_mag = collect_activation_magnitudes(decoder, cal_cfg, device)
    activation_summary = {
        k: {"len": int(v.shape[0]), "mean": float(v.mean()), "max": float(v.max())}
        for k, v in activation_mag.items()
    }
    print(f"[AWQ] collected magnitudes for {len(activation_mag)} weight tensors")

    # ------------------------------------------------------------------
    # Pre-AWQ naive PTQ baseline.
    # ------------------------------------------------------------------
    pre_quant: dict[str, np.ndarray] = {}
    for name, _shape in FIXED_STATE_SCHEMA:
        pre_quant[name] = naive_int4_quantize_tensor(fp32_state[name], args.block_size)
    pre_metrics = measure_roundtrip_rel_err(pre_quant, fp32_state)
    pre_raw, pre_brotli, pre_archive = encode_quantized_state_to_archive_bytes(
        pre_quant, block_size=args.block_size,
    )
    print(f"\n[AWQ] naive-PTQ baseline rel_err = {pre_metrics['weighted_avg_rel_err_pct']:.4f}%, "
          f"archive_bytes = {pre_archive:,} B")

    # ------------------------------------------------------------------
    # AWQ.
    # ------------------------------------------------------------------
    cfg = AWQConfig(
        block_size_encoder=args.block_size,
        alpha_grid=tuple(args.alpha_grid),
        error_metric=args.error_metric,
    )
    print(f"\n[AWQ] running AWQ (alpha_grid={list(cfg.alpha_grid)}, error_metric={cfg.error_metric})")
    awq_quant, per_tensor_stats = quantize_pr101_state(
        fp32_state, activation_mag, cfg=cfg,
    )

    post_metrics = measure_roundtrip_rel_err(awq_quant, fp32_state)
    raw_bytes, brotli_bytes, archive_bytes = encode_quantized_state_to_archive_bytes(
        awq_quant, block_size=args.block_size,
    )
    rel_err_pct = post_metrics["weighted_avg_rel_err_pct"]
    print(f"\n[AWQ] post-AWQ rel_err = {rel_err_pct:.4f}% "
          f"(naive baseline {pre_metrics['weighted_avg_rel_err_pct']:.4f}%)")
    print(f"[AWQ] post-AWQ archive_bytes = {archive_bytes:,} B "
          f"(naive {pre_archive:,} B; PR101 brotli baseline {PR101_BROTLI_BASELINE_BYTES:,} B)")

    verdict, cuda_eval_worth_testing = classify_cpu_proxy_candidate(
        weighted_avg_rel_err_pct=rel_err_pct,
        archive_bytes=archive_bytes,
    )
    dispatch_contract = local_awq_dispatch_contract(
        cuda_eval_worth_testing=cuda_eval_worth_testing,
    )

    if device.type == "cuda":
        evidence_grade = "[CUDA-research-signal faithful audit-criterion-5 test]"
    elif device.type == "mps":
        evidence_grade = "[MPS-research-signal faithful audit-criterion-5 test]"
    else:
        evidence_grade = EVIDENCE_GRADE

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"reports/raw/pr101_lossy_int4_awq_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": evidence_grade,
        "evidence_semantics": (
            "awq_calibrated_int4_byte_anchor_plus_roundtrip_rel_err_no_score"
        ),
        "score_claim": False,
        "promotion_eligible": dispatch_contract["promotion_eligible"],
        "rank_or_kill_eligible": dispatch_contract["rank_or_kill_eligible"],
        "ready_for_exact_eval_dispatch": dispatch_contract["ready_for_exact_eval_dispatch"],
        "cuda_eval_worth_testing": dispatch_contract["cuda_eval_worth_testing"],
        "dispatch_attempted": dispatch_contract["dispatch_attempted"],
        "proxy_row": True,
        "family_falsified": False,
        "falsification_scope": "measured_configuration_only",
        "input_state_dict": repo_relative(args.state_dict),
        "input_state_dict_sha256": sha256_file(args.state_dict),
        "device": str(device),
        "seed": args.seed,
        "block_size": args.block_size,
        "awq": {
            "alpha_grid": list(cfg.alpha_grid),
            "error_metric": cfg.error_metric,
            "calibration_samples": args.calibration_samples,
            "calibration_latent_distribution": "uniform[-1,1]^28",
            "calibration_substrate": "synthetic_latents_not_pr106_video_latents",
            "per_tensor_stats": per_tensor_stats,
            "activation_summary": activation_summary,
        },
        "naive_ptq_baseline": {
            "weighted_avg_rel_err_pct": pre_metrics["weighted_avg_rel_err_pct"],
            "max_p99_rel_err_pct": pre_metrics["max_p99_rel_err_pct"],
            "max_max_rel_err_pct": pre_metrics["max_max_rel_err_pct"],
            "archive_bytes": pre_archive,
        },
        "awq_post_calibration": {
            "weighted_avg_rel_err_pct": post_metrics["weighted_avg_rel_err_pct"],
            "weighted_avg_abs_err": post_metrics["weighted_avg_abs_err"],
            "max_p99_rel_err_pct": post_metrics["max_p99_rel_err_pct"],
            "max_max_rel_err_pct": post_metrics["max_max_rel_err_pct"],
            "per_tensor": post_metrics["per_tensor"],
        },
        "archive_bytes": archive_bytes,
        "raw_payload_bytes": raw_bytes,
        "brotli_bytes": brotli_bytes,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "verdict": verdict,
        "dispatch_blockers": dispatch_contract["dispatch_blockers"],
        "supersedes_prior_FALSIFIED_tag": False,
        "reactivation_criteria_tested": ["AWQ_calibration"],
        "reactivation_criteria_remaining": [],  # final audit criterion
        "n_total_elements": post_metrics["n_total_elements"],
        "n_nontrivial_elements": post_metrics["n_nontrivial_elements"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[AWQ] manifest: {manifest_path}")

    # ------------------------------------------------------------------
    # Cathedral evidence row.
    # ------------------------------------------------------------------
    if cuda_eval_worth_testing:
        dispatch_verdict = "CUDA-EVAL-WORTH-TESTING-pending-runtime-packet"
    else:
        dispatch_verdict = "DEFERRED-pending-research"
    evidence_row = {
        "technique": "lossy_int4_quantization_awq",
        "empirical_archive_bytes": archive_bytes,
        "empirical_distortion_increase_pct": rel_err_pct,
        "evidence_grade": evidence_grade,
        "evidence_marker": evidence_grade,
        "evidence_semantics": (
            "awq_calibrated_int4_byte_anchor_plus_roundtrip_rel_err_no_score"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": dispatch_contract["ready_for_exact_eval_dispatch"],
        "dispatch_attempted": dispatch_contract["dispatch_attempted"],
        "proxy_row": True,
        "cuda_eval_worth_testing": dispatch_contract["cuda_eval_worth_testing"],
        "family_falsified": False,
        "falsification_scope": "measured_configuration_only",
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "dispatch_blockers": manifest["dispatch_blockers"],
        "source": (
            f"{evidence_grade} {repo_relative(manifest_path)} "
            f"(AWQ samples={args.calibration_samples} alphas={len(cfg.alpha_grid)} "
            f"metric={cfg.error_metric}; rel_err={rel_err_pct:.2f}%)"
        ),
        "contest_dispatch_verdict": dispatch_verdict,
        "supersedes_prior_FALSIFIED_tag": False,
        "reactivation_criteria_tested": ["AWQ_calibration"],
        "reactivation_criteria_remaining": [],
        "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evidence_row) + "\n")
    print(f"[AWQ] evidence row appended: {args.evidence_jsonl}")

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"  archive_bytes:          {archive_bytes:,} B")
    print(f"  vs PR101 brotli base:   {(archive_bytes - PR101_BROTLI_BASELINE_BYTES):+,} B")
    print(f"  vs naive PTQ archive:   {(archive_bytes - pre_archive):+,} B")
    print(f"  rel_err_pct:            {rel_err_pct:.4f}%")
    print(f"  naive PTQ baseline:     {pre_metrics['weighted_avg_rel_err_pct']:.4f}%")
    print(f"  improvement:            {(pre_metrics['weighted_avg_rel_err_pct'] - rel_err_pct):+.3f}pp")
    print(f"  evidence_grade:         {evidence_grade}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
