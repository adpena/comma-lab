#!/usr/bin/env python3
"""Pre-Computed Gradient Corrections: one-step TTO without scorer at inflate time.

At COMPRESS time (unlimited budget):
1. Render all 1200 frames with current renderer
2. Compute full scorer gradient at each pixel: grad = d(score)/d(pixel)
3. The gradient tells us: "move this pixel THIS direction to improve the score"
4. Sparsify (keep top K% of pixels by gradient magnitude)
5. Quantize to int8 and compress with zlib
6. Store in archive

At INFLATE time (no scorer needed):
    frames = renderer_output + alpha * stored_gradient_correction

This is ONE-STEP TTO without any scorer at inflate time. Contest-compliant because
the gradient correction is just a pre-computed array of pixel adjustments.

Raw gradient: 1200 x 384 x 512 x 3 x float32 = ~2.8 GB
After top-5% sparsification + int8 quantization + zlib: ~50-100 KB expected.

Usage:
    # Smoke test (local MPS, 20 frames):
    PYTHONPATH=src:upstream python experiments/precompute_gradient_corrections.py \
        --checkpoint path/to/renderer_best.pt --device mps --smoke

    # Full run (4090):
    PYTHONPATH=src:upstream python experiments/precompute_gradient_corrections.py \
        --checkpoint path/to/renderer_best.pt --device cuda

    # Apply corrections at inflate time:
    # Load corrections.npz, decompress, add alpha * corrections to rendered frames
"""
from __future__ import annotations

# DX-fix 2026-04-25: line-buffer stdout/stderr so progress logs flush
# immediately when piped to log files (Python buffers ~8KB by default,
# making long-running scripts appear silent for hours per the optimize_poses
# incident on the A100 today).
import sys as _dx_sys
try:
    _dx_sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    _dx_sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass


import argparse
import heapq
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import struct

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_CANDIDATE_UPSTREAM = [
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
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
    Path(os.environ.get("TAC_RESULTS_DIR", ""))
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "gradient_corrections"
)

from tac.renderer import simulate_eval_roundtrip  # canonical impl (no local copy)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEGNET_INPUT_H, SEGNET_INPUT_W = 384, 512
NUM_FRAMES = 1200
NUM_PAIRS = NUM_FRAMES // 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute gradient corrections for one-step TTO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to renderer .pt or .bin checkpoint")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=NUM_FRAMES,
                   help="Number of frames to process")
    p.add_argument("--batch-pairs", type=int, default=10,
                   help="Pairs per gradient computation (limited by VRAM)")
    p.add_argument("--seg-weight", type=float, default=100.0,
                   help="SegNet loss weight (hinge)")
    p.add_argument("--pose-weight", type=float, default=10.0,
                   help="PoseNet loss weight (MSE)")
    p.add_argument("--hinge-margin", type=float, default=0.5,
                   help="Margin for SegNet hinge loss")
    p.add_argument("--top-k-pct", type=float, default=5.0,
                   help="Keep top K%% of pixels by gradient magnitude")
    p.add_argument("--allocation-strategy", type=str, default="greedy",
                   choices=["greedy", "fixed-budget"],
                   help="Pixel correction allocation strategy. 'fixed-budget' "
                        "preserves V1 top-k behavior; 'greedy' performs "
                        "rate-capped water-fill by gain/byte.")
    p.add_argument("--rate-cap-bytes", type=int, default=50000,
                   help="Estimated byte cap for greedy correction allocation")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Step size for gradient correction (applied at inflate time)")
    p.add_argument("--n-steps", type=int, default=1,
                   help="Number of gradient steps to average (1=single step, >1=averaged)")
    p.add_argument("--upstream", type=str, default=None,
                   help="Path to upstream repo (auto-detected if None)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: timestamped)")
    p.add_argument("--video", type=str, default=None,
                   help="Path to GT video (default: upstream/videos/0.mkv)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 20 frames")
    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate contest eval resolution roundtrip in loss. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")
    p.add_argument("--gt-poses-path", type=str, default=None,
                   help="Path to GT or optimized poses")
    p.add_argument("--quantize-bits", type=int, default=8,
                   choices=[4, 8, 16], help="Quantization bits for gradient storage")
    p.add_argument("--compression", type=str, default="zlib",
                   choices=["zlib", "none"], help="Compression for sparse gradient data")
    return p.parse_args()


# Magic-byte registry for content-based renderer-checkpoint dispatch.
#
# 2026-04-26 DEN-V2 bug class: load_renderer() did
# `torch.load(path, weights_only=False)` on a path that turned out to be an
# FP4-format .bin file (header `b'FP4A'`, NOT a PyTorch pickle). torch.load
# crashed with `ValueError: could not convert string to float: 'P4AV'` after
# parsing the magic as ASCII text. The pipeline soft-skipped, but only after a
# full subprocess crash — every DEN-class deploy hit this. Same root pattern
# as the 2026-04-26 SHIRAZ pose-loader bug (suffix-based dispatch on a file
# whose actual format differed from its extension).
#
# Permanent fix: content-detection. Read the first 4 bytes, dispatch on the
# magic. New formats register here; consumers never need to change. Mirrors
# the style of `tac.submission_archive.load_optimized_poses` and
# `tac.renderer_export.detect_checkpoint_type`.
_RENDERER_MAGIC_FP4A = b"FP4A"
_RENDERER_MAGIC_ASYM = b"ASYM"
_RENDERER_MAGIC_DPSM = b"DPSM"
_RENDERER_MAGIC_INT4_LZMA2 = b"I4LZ"
# Lane I (Cool-Chic / C3 residual) — added 2026-04-27. Both formats route
# through tac.renderer_export.load_any_renderer_checkpoint which dispatches
# to load_coolchic_renderer / load_c3_residual_renderer respectively.
_RENDERER_MAGIC_COOLCHIC = b"CCh1"
_RENDERER_MAGIC_C3_RESIDUAL = b"C3R1"
_RENDERER_PICKLE_MAGICS = (
    b"\x80\x02",      # pickle protocol 2
    b"\x80\x03",      # pickle protocol 3
    b"\x80\x04",      # pickle protocol 4
    b"\x80\x05",      # pickle protocol 5
    b"PK\x03\x04",    # ZIP (PyTorch >=1.6 default torch.save container)
)


def _looks_like_pytorch_pickle(raw: bytes) -> bool:
    """True iff the first bytes match a known torch.save container header."""
    return any(raw.startswith(m) for m in _RENDERER_PICKLE_MAGICS)


def load_renderer(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load an AsymmetricPairGenerator-compatible renderer with content-based
    format detection.

    This is the canonical loader for downstream consumers (engineered_quant_noise,
    pair_difficulty_map, etc.). Dispatch is on the file's first 4 magic bytes,
    NOT the suffix — suffix-based dispatch is what produced the 2026-04-26
    DEN-V2 crash where an FP4-format .bin file got handed to torch.load and
    raised "could not convert string to float: 'P4AV'".

    Supported formats:
        - FP4A (.bin) — FP4-quantized AsymmetricPairGenerator → loaded via
          tac.renderer_export.load_asymmetric_checkpoint_fp4
        - ASYM (.bin) — float ASYM export → loaded via
          tac.renderer_export.load_asymmetric_checkpoint
        - DPSM (.bin) — DP-SIMS renderer → loaded via
          tac.renderer_export.load_renderer_checkpoint
        - I4LZ (.bin) — int4 + LZMA2 → routed through load_any_renderer_checkpoint
        - PyTorch pickle / zip (.pt) — legacy training-checkpoint dict with
          `model_state_dict` / `state_dict` / raw state dict + `model_config`

    Anything else raises a RuntimeError that names both the magic seen and the
    accepted formats. Never silently degrade to torch.load.
    """
    from pathlib import Path

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Renderer checkpoint not found: {ckpt_path}")

    # Read just enough to dispatch — full file is read again by the format
    # loader. Cheap on every supported size.
    with ckpt_path.open("rb") as fh:
        magic = fh.read(4)

    if len(magic) < 4:
        raise RuntimeError(
            f"Renderer checkpoint {ckpt_path} is too short ({len(magic)}B); "
            f"expected at least 4 magic bytes (FP4A/ASYM/DPSM/I4LZ/CCh1/C3R1) "
            f"or a PyTorch pickle/zip header."
        )

    # Branch A: native binary export formats. Defer to the canonical
    # auto-detecting loader so I4LZ/ASYM/DPSM/FP4A all stay correct without
    # us reimplementing arch inference here.
    if magic in (_RENDERER_MAGIC_FP4A, _RENDERER_MAGIC_ASYM,
                 _RENDERER_MAGIC_DPSM, _RENDERER_MAGIC_INT4_LZMA2,
                 _RENDERER_MAGIC_COOLCHIC, _RENDERER_MAGIC_C3_RESIDUAL):
        from tac.renderer_export import load_any_renderer_checkpoint
        device_str = str(device) if not isinstance(device, str) else device
        model = load_any_renderer_checkpoint(ckpt_path, device=device_str).eval()
        for p_param in model.parameters():
            p_param.requires_grad = False
        n_params = sum(p_param.numel() for p_param in model.parameters())
        print(f"[renderer] Loaded {n_params:,} params from {ckpt_path} "
              f"(format={magic.decode('ascii', errors='replace')})")
        return model

    # Branch B: legacy PyTorch pickle / zip checkpoint. This is the original
    # training-checkpoint format: a dict with `config` / `model_config` and
    # one of `model_state_dict` / `state_dict` / a raw state dict.
    if _looks_like_pytorch_pickle(magic):
        from tac.renderer import AsymmetricPairGenerator

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model_cfg = ckpt.get("model_config", ckpt.get("config", {}))
        model = AsymmetricPairGenerator(
            num_classes=model_cfg.get("num_classes", 5),
            embed_dim=model_cfg.get("embed_dim", 6),
            base_ch=model_cfg.get("base_ch", 36),
            mid_ch=model_cfg.get("mid_ch", 60),
            motion_hidden=model_cfg.get("motion_hidden", 32),
            depth=model_cfg.get("depth", 1),
            max_flow_px=model_cfg.get("max_flow_px", 20.0),
            max_residual=model_cfg.get("max_residual", 20.0),
            flow_only=model_cfg.get("flow_only", False),
            pose_dim=model_cfg.get("pose_dim", 0),
            use_dsconv=model_cfg.get("use_dsconv", False),
        )

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

        model = model.eval().to(device)
        for p_param in model.parameters():
            p_param.requires_grad = False

        n_params = sum(p_param.numel() for p_param in model.parameters())
        print(f"[renderer] Loaded {n_params:,} params from {ckpt_path} "
              f"(format=pytorch)")
        return model

    # Unknown — fail loudly with the exact magic bytes seen and the accepted set.
    raise RuntimeError(
        f"Unrecognized renderer checkpoint format at {ckpt_path}: "
        f"magic bytes {magic!r}. Accepted formats: "
        f"FP4A/ASYM/DPSM/I4LZ/CCh1/C3R1 binary exports, or a PyTorch pickle/zip "
        f"(magics {[m for m in _RENDERER_PICKLE_MAGICS]}). "
        f"This is the 2026-04-26 DEN-V2 bug class — do NOT add a fallback to "
        f"torch.load; instead fix the producer or register a new magic in "
        f"experiments/precompute_gradient_corrections.load_renderer."
    )


def segnet_hinge_loss(
    logits: torch.Tensor,
    gt_masks: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Hinge loss on SegNet logits."""
    correct = logits.gather(1, gt_masks.unsqueeze(1)).squeeze(1)
    mask_inf = torch.zeros_like(logits)
    mask_inf.scatter_(1, gt_masks.unsqueeze(1), float("-inf"))
    runner_up = (logits + mask_inf).max(dim=1).values
    loss = F.relu(margin - (correct - runner_up))
    return loss.mean()


def compute_frame_gradients(
    rendered_frames: torch.Tensor,
    gt_masks: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    seg_weight: float,
    pose_weight: float,
    hinge_margin: float,
    eval_roundtrip: bool,
) -> torch.Tensor:
    """Compute d(loss)/d(pixels) for a batch of frame pairs.

    Args:
        rendered_frames: (2*B, H, W, 3) float in [0, 255], rendered output
        gt_masks: (2*B, H, W) long, GT SegNet class labels
        pose_targets: (B, 6) GT PoseNet outputs
        posenet, segnet: frozen differentiable scorers
        device: computation device
        seg_weight, pose_weight: loss weights
        hinge_margin: SegNet hinge loss margin
        eval_roundtrip: simulate resolution roundtrip

    Returns:
        (2*B, H, W, 3) gradient tensor (negative gradient = improvement direction)
    """
    # Make frames require grad so we can compute d(loss)/d(frames)
    frames_hwc = rendered_frames.clone().detach().to(device).requires_grad_(True)
    frames_chw = frames_hwc.permute(0, 3, 1, 2).contiguous()

    if eval_roundtrip:
        frames_chw = simulate_eval_roundtrip(frames_chw)

    # SegNet loss
    seg_in = segnet.preprocess_input(frames_chw.unsqueeze(1))
    seg_logits = segnet(seg_in)
    seg_loss = segnet_hinge_loss(seg_logits, gt_masks.to(device), margin=hinge_margin)

    # PoseNet loss
    N = frames_hwc.shape[0]
    B = N // 2
    frame_t_chw = frames_chw[:B]
    frame_t1_chw = frames_chw[B:]
    pairs_chw = torch.stack([frame_t_chw, frame_t1_chw], dim=1)

    if eval_roundtrip:
        B_p, T_p, C_p, H_p, W_p = pairs_chw.shape
        flat = pairs_chw.reshape(B_p * T_p, C_p, H_p, W_p)
        flat = simulate_eval_roundtrip(flat)
        pairs_chw = flat.reshape(B_p, T_p, C_p, H_p, W_p)

    pose_in = posenet.preprocess_input(pairs_chw)
    pose_out = posenet(pose_in)["pose"][..., :6]
    pose_loss = F.mse_loss(pose_out, pose_targets.to(device))

    total_loss = seg_weight * seg_loss + pose_weight * pose_loss
    total_loss.backward()

    # Return negative gradient (descent direction = improvement)
    grad = -frames_hwc.grad.detach().cpu()
    return grad


def estimated_sparse_bytes(
    sparse_data: dict,
    byte_overhead: int = 100,
    byte_per_pixel: int = 1,
) -> int:
    """Estimate sparse correction bytes using the EC-V2 arithmetic model."""
    n_kept = int(sparse_data.get("n_kept", len(sparse_data.get("indices", []))))
    return int(byte_overhead + n_kept * byte_per_pixel)


def greedy_waterfill_correction_map(
    gradients: np.ndarray,
    rate_cap_bytes: int = 50000,
    margin_deltas: np.ndarray | None = None,
    byte_overhead: int = 100,
    byte_per_pixel: int = 1,
    return_metadata: bool = False,
):
    """Allocate int8 pixel corrections by descending gain/byte.

    The gain proxy is ``abs(margin_deltas)`` when supplied, otherwise the
    per-pixel gradient magnitude. Byte cost follows the lane spec's fast
    arithmetic model: ``byte_overhead`` fixed bytes plus ``byte_per_pixel``
    bytes per kept pixel.

    Codex finding 1 fix (2026-04-28):
        * The dense ``corrections`` map now stores the **original gradient
          values** for selected pixels (cast back to ``int8`` via per-tensor
          symmetric quantization), NOT sign-only ±127. The previous code
          discarded magnitude → ``apply_corrections`` clamped pixels to
          ±127 instead of nudging them by the gradient direction.
        * The ``rate_cap_bytes`` accounting argument matches the *fast*
          per-pixel arithmetic model used during the greedy selection;
          callers who need the cap to track the real packed format
          (uint32 index + ``C`` int8 channels + JSON header) should pass
          ``byte_per_pixel=4 + C`` and adjust ``byte_overhead`` for the
          header. ``sparsify_and_quantize`` performs an additional drop-
          tail-and-repack pass on the actual ``pack_sparse_corrections``
          output to enforce a hard byte cap end-to-end.
    """
    gradients = np.asarray(gradients)
    if gradients.ndim != 4:
        raise ValueError(f"gradients must have shape (N,H,W,C), got {gradients.shape}")
    if byte_per_pixel <= 0:
        raise ValueError("byte_per_pixel must be positive")

    N, H, W, C = gradients.shape
    corrections = np.zeros((N, H, W, C), dtype=np.int8)
    cap = int(rate_cap_bytes)

    if margin_deltas is None:
        gains = np.sqrt((gradients.astype(np.float32) ** 2).sum(axis=-1))
    else:
        gains = np.abs(np.asarray(margin_deltas, dtype=np.float32))
        if gains.shape != gradients.shape[:-1]:
            raise ValueError(
                f"margin_deltas must have shape {gradients.shape[:-1]}, got {gains.shape}"
            )

    flat_gains = gains.reshape(-1)
    flat_grads = gradients.reshape(-1, C)
    # Per-tensor symmetric quantization scale across all candidate gradient
    # magnitudes. Using a single scale lets the dense int8 map preserve
    # *relative* gradient magnitude across selected pixels — required for
    # apply_corrections to perform a real gradient step (codex finding 1).
    grad_max = float(np.abs(flat_grads).max()) if flat_grads.size else 0.0
    quant_scale = grad_max if grad_max > 1e-12 else 1.0

    heap: list[tuple[float, int, np.ndarray, float]] = []
    for flat_idx, gain in enumerate(flat_gains):
        gain_f = float(gain)
        if not math.isfinite(gain_f) or gain_f <= 0.0:
            continue
        raw = flat_grads[flat_idx]
        if not np.any(raw):
            continue
        # Preserve gradient magnitude (codex finding 1): quantize the original
        # gradient values to int8 using a shared symmetric scale, instead of
        # writing sign-only ±127.
        encoded = np.clip(
            np.round(raw.astype(np.float32) / quant_scale * 127.0),
            -127,
            127,
        ).astype(np.int8)
        ratio = gain_f / float(byte_per_pixel)
        # heapq is a min-heap. Negate ratio for max-heap behavior and include
        # flat_idx as the next tuple item for deterministic tie-breaking.
        heapq.heappush(heap, (-ratio, flat_idx, encoded, gain_f))

    selected: list[int] = []
    total_gain = 0.0
    used_bytes = int(byte_overhead)
    flat_corr = corrections.reshape(-1, C)
    while heap:
        _neg_ratio, flat_idx, encoded, gain_f = heapq.heappop(heap)
        next_bytes = used_bytes + int(byte_per_pixel)
        if next_bytes > cap:
            break
        flat_corr[flat_idx] = encoded
        used_bytes = next_bytes
        total_gain += gain_f
        selected.append(int(flat_idx))

    meta = {
        "selected_indices": selected,
        "estimated_bytes": used_bytes,
        "total_gain": float(total_gain),
        "byte_overhead": int(byte_overhead),
        "byte_per_pixel": int(byte_per_pixel),
        "rate_cap_bytes": cap,
        "quant_scale": float(quant_scale),
    }
    if return_metadata:
        return corrections, meta
    return corrections


def sparse_to_dense_int8(sparse_data: dict) -> np.ndarray:
    """Return a dense int8 correction map from a sparse correction dict."""
    shape = tuple(sparse_data["shape"])
    if len(shape) != 4:
        raise ValueError(f"sparse shape must be 4D, got {shape}")
    dense = np.zeros(shape, dtype=np.int8)
    if sparse_data["n_kept"] == 0:
        return dense
    flat = dense.reshape(-1, shape[-1])
    values = sparse_data["values"]
    if values.dtype != np.int8:
        values = np.clip(np.rint(values.astype(np.float32)), -127, 127).astype(np.int8)
    flat[sparse_data["indices"]] = np.clip(values, -127, 127).astype(np.int8)
    return dense


def sparsify_and_quantize(
    gradients: np.ndarray,
    top_k_pct: float = 5.0,
    quantize_bits: int = 8,
    allocation_strategy: str = "greedy",
    rate_cap_bytes: int = 50000,
) -> dict:
    """Sparsify gradient tensor and quantize to low precision.

    Args:
        gradients: (N, H, W, 3) float32 gradient tensor
        top_k_pct: keep top K% of pixels by gradient magnitude (fixed-budget)
        quantize_bits: 4, 8, or 16 bit quantization
        allocation_strategy: "greedy" for EC-V2, "fixed-budget" for V1

    Returns:
        dict with sparse representation:
        - indices: (K,) uint32 flat indices of kept pixels
        - values: (K, 3) quantized gradient values
        - scale: float, max absolute value for dequantization
        - shape: original tensor shape
        - top_k_pct: sparsity level
    """
    N, H, W, C = gradients.shape

    if allocation_strategy not in {"greedy", "fixed-budget"}:
        raise ValueError(
            "allocation_strategy must be 'greedy' or 'fixed-budget', "
            f"got {allocation_strategy!r}"
        )

    magnitudes = np.sqrt((gradients ** 2).sum(axis=-1))  # (N, H, W)
    flat_mags = magnitudes.reshape(-1)
    flat_grads = gradients.reshape(-1, C)

    if allocation_strategy == "greedy":
        correction_map, greedy_meta = greedy_waterfill_correction_map(
            gradients,
            rate_cap_bytes=rate_cap_bytes,
            margin_deltas=magnitudes,
            return_metadata=True,
        )
        top_k_indices = np.asarray(greedy_meta["selected_indices"], dtype=np.uint32)
        k = int(top_k_indices.size)
        # Codex finding 1 fix: pull the ORIGINAL gradient values for the
        # selected indices, not the int8 dense map (which was previously
        # ±127 sign-only and discarded magnitude). The downstream quantize
        # step now operates on real gradient magnitudes.
        kept_grads = flat_grads[top_k_indices].astype(np.float32)
    else:
        greedy_meta = None
        # Keep top K% by magnitude. This is the V1 fixed-budget behavior.
        k = max(1, int(len(flat_mags) * top_k_pct / 100.0))
        # Use argpartition for O(n) instead of O(n log n) full sort
        top_k_indices = np.argpartition(flat_mags, -k)[-k:]
        # Sort the top-k indices by magnitude (descending) for better compression
        sorted_order = np.argsort(flat_mags[top_k_indices])[::-1]
        top_k_indices = top_k_indices[sorted_order]
        kept_grads = flat_grads[top_k_indices]  # (K, 3)

    # Quantize
    scale = np.abs(kept_grads).max() if k > 0 else 0.0
    if scale < 1e-10:
        scale = 1.0  # avoid division by zero

    if quantize_bits == 8:
        normalized = (kept_grads / scale * 127.0).clip(-127, 127).astype(np.int8)
    elif quantize_bits == 4:
        normalized = (kept_grads / scale * 7.0).clip(-7, 7).astype(np.int8)
    elif quantize_bits == 16:
        # fp16: store raw values (no scale normalization needed)
        normalized = kept_grads.astype(np.float16)
    else:
        raise ValueError(f"Unsupported quantize_bits={quantize_bits}")

    out = {
        "indices": top_k_indices.astype(np.uint32),
        "values": normalized,
        "scale": float(scale),
        "shape": list(gradients.shape),
        "top_k_pct": top_k_pct,
        "quantize_bits": quantize_bits,
        "n_kept": k,
        "n_total": len(flat_mags),
        "allocation_strategy": allocation_strategy,
        "estimated_bytes": None,
        "total_gain": float(flat_mags[top_k_indices].sum()) if k > 0 else 0.0,
    }
    out["estimated_bytes"] = (
        greedy_meta["estimated_bytes"]
        if greedy_meta is not None
        else estimated_sparse_bytes(out)
    )
    if greedy_meta is not None:
        out["total_gain"] = greedy_meta["total_gain"]
        out["rate_cap_bytes"] = int(rate_cap_bytes)

    # Codex finding 1 bug 2 fix: enforce a HARD cap on the actual packed-byte
    # size by dropping tail entries (lowest magnitude first) and repacking
    # until ``len(pack_sparse_corrections(...)) <= rate_cap_bytes``. The
    # greedy allocator's internal accounting uses a fast 1-byte-per-pixel
    # arithmetic model; the real packed format is 4-byte uint32 indices
    # plus C-byte int8 channels plus a JSON header — typically ~7×–13×
    # the fast model. Without this loop a 50 KB cap could materialise as
    # ~350 KB on disk, silently busting the archive size.
    out["packed_bytes"] = enforce_packed_byte_cap(out, rate_cap_bytes=rate_cap_bytes)
    return out


def enforce_packed_byte_cap(
    sparse_data: dict,
    rate_cap_bytes: int,
    compression: str = "zlib",
    max_iters: int = 64,
) -> int:
    """Drop tail entries until the packed (and optionally zlib-compressed)
    correction blob fits inside ``rate_cap_bytes``.

    Mutates ``sparse_data`` in place (``indices``, ``values``, ``n_kept``).
    Returns the final packed byte size. Idempotent if already under cap.

    The "tail" is defined as the entries with the smallest absolute value
    (l-infinity over channels), which are the lowest-gain selections per
    the EC-V2 greedy ranking. Dropping them preserves the highest-gain
    corrections.
    """
    if rate_cap_bytes <= 0 or sparse_data["n_kept"] == 0:
        sparse_data["packed_bytes"] = 0
        return 0
    indices = np.asarray(sparse_data["indices"]).copy()
    values = np.asarray(sparse_data["values"]).copy()
    # Rank by gain (max abs across channels). Smallest = candidate to drop.
    if values.ndim == 1:
        gain = np.abs(values).astype(np.float32)
    else:
        gain = np.abs(values).max(axis=-1).astype(np.float32)
    order = np.argsort(-gain)  # descending: highest gain first
    indices = indices[order]
    values = values[order]
    n = int(indices.shape[0])

    for _ in range(max_iters):
        sparse_data["indices"] = indices[:n]
        sparse_data["values"] = values[:n]
        sparse_data["n_kept"] = n
        packed = pack_sparse_corrections(sparse_data, compression=compression)
        size = len(packed)
        if size <= rate_cap_bytes or n <= 0:
            sparse_data["packed_bytes"] = size
            return size
        # Drop ~10% of remaining entries (at least 1) and try again.
        drop = max(1, n // 10)
        n = max(0, n - drop)
    # Final state.
    sparse_data["indices"] = indices[:n]
    sparse_data["values"] = values[:n]
    sparse_data["n_kept"] = n
    packed = pack_sparse_corrections(sparse_data, compression=compression)
    sparse_data["packed_bytes"] = len(packed)
    return len(packed)


def pack_sparse_corrections(sparse_data: dict, compression: str = "zlib") -> bytes:
    """Pack sparse gradient corrections into a compact binary format.

    Format:
        [header_len: uint32] [json_header] [indices_data] [values_data]

    Returns:
        bytes object ready for archive inclusion
    """
    header = {
        "scale": sparse_data["scale"],
        "shape": sparse_data["shape"],
        "top_k_pct": sparse_data["top_k_pct"],
        "quantize_bits": sparse_data["quantize_bits"],
        "n_kept": sparse_data["n_kept"],
        "n_total": sparse_data["n_total"],
    }
    header_bytes = json.dumps(header).encode("utf-8")

    indices_bytes = sparse_data["indices"].tobytes()
    values_bytes = sparse_data["values"].tobytes()

    # Pack: header_len (4B) + header + indices + values
    data = struct.pack("<I", len(header_bytes)) + header_bytes + indices_bytes + values_bytes

    if compression == "zlib":
        data = zlib.compress(data, level=9)

    return data


def unpack_sparse_corrections(data: bytes, compressed: bool = True) -> dict:
    """Unpack sparse gradient corrections from binary format.

    Returns dict compatible with apply_corrections().
    """
    if compressed:
        data = zlib.decompress(data)

    header_len = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4:4 + header_len].decode("utf-8"))

    offset = 4 + header_len
    n_kept = header["n_kept"]
    indices_size = n_kept * 4  # uint32
    indices = np.frombuffer(data[offset:offset + indices_size], dtype=np.uint32)
    offset += indices_size

    qbits = header["quantize_bits"]
    if qbits in (4, 8):
        values = np.frombuffer(data[offset:], dtype=np.int8).reshape(n_kept, 3)
    elif qbits == 16:
        values = np.frombuffer(data[offset:], dtype=np.float16).reshape(n_kept, 3)
    else:
        raise ValueError(f"Unsupported quantize_bits={qbits}")

    return {
        "indices": indices,
        "values": values,
        "scale": header["scale"],
        "shape": header["shape"],
        "quantize_bits": qbits,
        "n_kept": n_kept,
        "n_total": header["n_total"],
    }


def apply_corrections(
    frames: np.ndarray,
    corrections: dict,
    alpha: float = 1.0,
) -> np.ndarray:
    """Apply pre-computed gradient corrections to rendered frames.

    This is the INFLATE-TIME function. No scorer needed.

    Args:
        frames: (N, H, W, 3) float32 rendered frames
        corrections: dict from unpack_sparse_corrections()
        alpha: step size multiplier

    Returns:
        (N, H, W, 3) corrected frames
    """
    N, H, W, C = frames.shape
    assert N * H * W == corrections["n_total"], (
        f"Resolution mismatch: {N * H * W} vs {corrections['n_total']}"
    )
    flat_frames = frames.reshape(-1, C).copy()

    indices = corrections["indices"]
    values = corrections["values"]
    scale = corrections["scale"]
    qbits = corrections["quantize_bits"]

    # Dequantize
    if qbits == 8:
        dequant = values.astype(np.float32) / 127.0 * scale
    elif qbits == 4:
        dequant = values.astype(np.float32) / 7.0 * scale
    elif qbits == 16:
        # fp16: stored as raw float16, no scale normalization was applied
        dequant = values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported quantize_bits={qbits}")

    # Apply corrections
    flat_frames[indices] += alpha * dequant
    flat_frames = np.clip(flat_frames, 0, 255)

    return flat_frames.reshape(N, H, W, C)


def _enforce_eval_roundtrip(args) -> None:
    """CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True; only escape hatch
    is TAC_ALLOW_NO_ROUNDTRIP=1 env var with loud banner.

    2026-04-27 codex R5-4 #4: delegated to the centralised
    `tac.eval_roundtrip_gate.enforce_eval_roundtrip` helper. The previous
    per-script copies were sticky — they only printed the warning when
    `args.eval_roundtrip` was already False, so a leftover env var in a
    shell / tmux session silently relaxed later runs without acknowledgement.
    The centralised helper warns whenever the env var is present and
    records it in run provenance.
    """
    from tac.eval_roundtrip_gate import enforce_eval_roundtrip
    output_dir = getattr(args, "output_dir", None)
    enforce_eval_roundtrip(args, output_dir=output_dir, write_provenance=output_dir is not None)


def main():
    args = parse_args()

    if args.smoke:
        args.n_frames = 20
        args.batch_pairs = 5
        print("[smoke] Smoke test: 20 frames, 5 pairs/batch")

    args.n_frames = args.n_frames - (args.n_frames % 2)
    n_pairs = args.n_frames // 2

    device = torch.device(args.device)
    upstream = Path(args.upstream) if args.upstream else UPSTREAM_ROOT
    if upstream is None:
        print("ERROR: Cannot find upstream root. Set --upstream or TAC_UPSTREAM_DIR.",
              file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = str(RESULTS_DIR / f"grad_corr_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # codex R5-r6 #3: gate AFTER output_dir resolution so sidecar lands.
    _enforce_eval_roundtrip(args)

    video_path = args.video or str(upstream / "videos" / "0.mkv")

    print(f"[config] device={device}, n_frames={args.n_frames}")
    print(f"[config] batch_pairs={args.batch_pairs}, n_steps={args.n_steps}")
    print(f"[config] seg_weight={args.seg_weight}, pose_weight={args.pose_weight}")
    print(f"[config] top_k_pct={args.top_k_pct}%, quantize_bits={args.quantize_bits}")
    print(f"[config] allocation_strategy={args.allocation_strategy}, "
          f"rate_cap_bytes={args.rate_cap_bytes}")
    print(f"[config] alpha={args.alpha}, compression={args.compression}")
    print(f"[config] checkpoint={args.checkpoint}")
    print(f"[config] output_dir={output_dir}")

    t_total = time.monotonic()

    # -- Step 1: Load scorers --
    print("\n[1/6] Loading differentiable scorers...")
    t0 = time.monotonic()
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    print(f"[1/6] Scorers loaded in {time.monotonic() - t0:.1f}s")

    # -- Step 2: Load renderer --
    print("\n[2/6] Loading renderer...")
    t0 = time.monotonic()
    renderer = load_renderer(args.checkpoint, device)
    print(f"[2/6] Renderer loaded in {time.monotonic() - t0:.1f}s")

    # -- Step 3: Decode GT video + extract masks + targets --
    print(f"\n[3/6] Decoding GT video ({args.n_frames} frames)...")
    t0 = time.monotonic()
    from tac.data import load_gt_video
    gt_frames = load_gt_video(video_path, n_frames=args.n_frames)
    args.n_frames = len(gt_frames)
    n_pairs = args.n_frames // 2
    print(f"[3/6] Decoded {args.n_frames} frames in {time.monotonic() - t0:.1f}s")

    print("\n[4/6] Extracting GT masks and pose targets...")
    t0 = time.monotonic()
    from tac.scorer import extract_gt_masks, extract_gt_pose_targets
    gt_masks = extract_gt_masks(gt_frames, segnet, device)
    pose_targets = extract_gt_pose_targets(gt_frames, posenet, device)
    print(f"[4/6] Masks: {gt_masks.shape}, Poses: {pose_targets.shape} "
          f"in {time.monotonic() - t0:.1f}s")

    # Load poses for FiLM conditioning
    renderer_pose_dim = getattr(renderer, "pose_dim", 0)
    init_poses: torch.Tensor | None = None
    if renderer_pose_dim > 0:
        if args.gt_poses_path and Path(args.gt_poses_path).exists():
            init_poses = torch.load(args.gt_poses_path, map_location="cpu",
                                    weights_only=True).float()[:n_pairs]
        else:
            init_poses = pose_targets[:n_pairs].clone()

    # -- Step 4: Render all frames --
    print(f"\n[5/6] Rendering {args.n_frames} frames and computing gradients...")
    t0 = time.monotonic()
    n_batches = math.ceil(n_pairs / args.batch_pairs)

    all_gradients = []
    all_rendered = []
    batch_stats = []

    for batch_idx in range(n_batches):
        pair_start = batch_idx * args.batch_pairs
        pair_end = min(pair_start + args.batch_pairs, n_pairs)
        frame_start = 2 * pair_start
        frame_end = 2 * pair_end
        n_batch_pairs = pair_end - pair_start

        # Render frames (no grad through renderer, grad through pixel output)
        batch_masks_t = gt_masks[frame_start:frame_end:2].to(device)
        batch_masks_t1 = gt_masks[frame_start + 1:frame_end + 1:2].to(device)
        batch_pose = None
        if init_poses is not None:
            batch_pose = init_poses[pair_start:pair_end].to(device)

        with torch.no_grad():
            pairs = renderer(batch_masks_t, batch_masks_t1, pose=batch_pose)
            frame_t = pairs[:, 0]   # (B, H, W, 3)
            frame_t1 = pairs[:, 1]  # (B, H, W, 3)
            # Interleave to match gt_masks ordering: [f0_t, f0_t1, f1_t, f1_t1, ...]
            rendered = torch.stack([frame_t, frame_t1], dim=1).reshape(-1, *frame_t.shape[1:])  # (2*B, H, W, 3)

        all_rendered.append(rendered.cpu())

        # Compute gradients (possibly multi-step averaged)
        batch_grad_accum = torch.zeros_like(rendered, device="cpu")
        for step in range(args.n_steps):
            grad = compute_frame_gradients(
                rendered_frames=rendered,
                gt_masks=gt_masks[frame_start:frame_end],
                pose_targets=pose_targets[pair_start:pair_end],
                posenet=posenet,
                segnet=segnet,
                device=device,
                seg_weight=args.seg_weight,
                pose_weight=args.pose_weight,
                hinge_margin=args.hinge_margin,
                eval_roundtrip=args.eval_roundtrip,
            )
            batch_grad_accum += grad

        batch_grad_accum /= args.n_steps
        all_gradients.append(batch_grad_accum)

        grad_mag = batch_grad_accum.norm(dim=-1)
        stats = {
            "batch": batch_idx,
            "pairs": f"{pair_start}-{pair_end}",
            "grad_mean": grad_mag.mean().item(),
            "grad_max": grad_mag.max().item(),
            "grad_nonzero_pct": (grad_mag > 1e-6).float().mean().item() * 100,
        }
        batch_stats.append(stats)

        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            print(f"  batch {batch_idx + 1}/{n_batches}: "
                  f"|grad|_mean={stats['grad_mean']:.4f}, "
                  f"|grad|_max={stats['grad_max']:.4f}, "
                  f"nonzero={stats['grad_nonzero_pct']:.1f}%")

        # Free GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    dt_grad = time.monotonic() - t0
    print(f"[5/6] Gradients computed in {dt_grad:.1f}s")

    # -- Step 5: Sparsify, quantize, compress --
    print(f"\n[6/6] Sparsifying (top {args.top_k_pct}%), "
          f"quantizing ({args.quantize_bits}-bit), compressing...")
    t0 = time.monotonic()

    all_grads_np = torch.cat(all_gradients, dim=0).numpy()
    all_rendered_np = torch.cat(all_rendered, dim=0).numpy()

    sparse_data = sparsify_and_quantize(
        all_grads_np,
        top_k_pct=args.top_k_pct,
        quantize_bits=args.quantize_bits,
        allocation_strategy=args.allocation_strategy,
        rate_cap_bytes=args.rate_cap_bytes,
    )

    packed = pack_sparse_corrections(sparse_data, compression=args.compression)
    packed_size = len(packed)

    print(f"  Raw gradient: {all_grads_np.nbytes:,} bytes")
    print(f"  Sparse ({sparse_data['n_kept']:,} / {sparse_data['n_total']:,} pixels, "
          f"{args.top_k_pct}% / {args.allocation_strategy})")
    print(f"  Estimated sparse bytes: {sparse_data['estimated_bytes']:,} "
          f"(cap={args.rate_cap_bytes:,})")
    print(f"  Packed: {packed_size:,} bytes ({packed_size / 1024:.1f} KB)")
    print(f"  Compression ratio: {all_grads_np.nbytes / max(packed_size, 1):.0f}x")

    # Save packed corrections
    corrections_path = output_dir / "gradient_corrections.bin"
    with open(corrections_path, "wb") as f:
        f.write(packed)
    print(f"  Saved: {corrections_path}")
    correction_map_path = output_dir / "correction_map.npy"
    np.save(correction_map_path, sparse_to_dense_int8(sparse_data))
    print(f"  Saved dense int8 correction map: {correction_map_path}")

    # -- Step 6: Validate by applying corrections --
    print("\n  Validating corrections (apply + re-score)...")

    # Unpack and apply
    unpacked = unpack_sparse_corrections(packed, compressed=(args.compression == "zlib"))
    corrected_np = apply_corrections(all_rendered_np, unpacked, alpha=args.alpha)

    # Score original vs corrected
    from tac.scorer import compute_proxy_score
    original_frames_t = torch.from_numpy(all_rendered_np).float()
    corrected_frames_t = torch.from_numpy(corrected_np).float()

    orig_score = compute_proxy_score(
        original_frames_t, gt_frames, posenet, segnet, device, rate=0.0,
    )
    corr_score = compute_proxy_score(
        corrected_frames_t, gt_frames, posenet, segnet, device, rate=0.0,
    )

    # Rate cost of corrections
    # Compute original rate from actual archive size if available, else estimate
    archive_path = Path(args.checkpoint).parent / "archive.zip"
    if archive_path.exists():
        archive_bytes = archive_path.stat().st_size
        original_rate = archive_bytes / 37_545_489
    else:
        # Fallback: estimate from typical renderer archive (~187KB)
        original_rate = 187_000 / 37_545_489
    correction_rate = packed_size / 37_545_489
    total_rate = original_rate + correction_rate

    print(f"\n  Original:  score={orig_score['score']:.4f} "
          f"(pose={orig_score['pose']:.6f}, seg={orig_score['seg']:.6f})")
    print(f"  Corrected: score={corr_score['score']:.4f} "
          f"(pose={corr_score['pose']:.6f}, seg={corr_score['seg']:.6f})")
    print(f"  Delta:     {corr_score['score'] - orig_score['score']:+.4f}")
    print(f"  Rate cost: {correction_rate:.6f} ({packed_size / 1024:.1f} KB)")
    print(f"  Total rate: {total_rate:.6f} (archive + corrections)")

    dt_pack = time.monotonic() - t0
    total_time = time.monotonic() - t_total

    # Save summary
    summary = {
        "config": vars(args),
        "original_score": orig_score,
        "corrected_score": corr_score,
        "delta_score": corr_score["score"] - orig_score["score"],
        "packed_size_bytes": packed_size,
        "packed_size_kb": packed_size / 1024,
        "correction_rate": correction_rate,
        "sparse_stats": {
            "n_kept": sparse_data["n_kept"],
            "n_total": sparse_data["n_total"],
            "sparsity_pct": args.top_k_pct,
            "scale": sparse_data["scale"],
            "allocation_strategy": args.allocation_strategy,
            "estimated_bytes": sparse_data["estimated_bytes"],
            "rate_cap_bytes": args.rate_cap_bytes,
            "total_gain": sparse_data.get("total_gain", 0.0),
        },
        "gradient_stats": batch_stats,
        "total_time_s": total_time,
        "gradient_time_s": dt_grad,
        "pack_time_s": dt_pack,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save raw gradient stats for analysis
    grad_magnitudes = np.sqrt((all_grads_np ** 2).sum(axis=-1))
    np.save(output_dir / "gradient_magnitudes.npy", grad_magnitudes)
    print(f"\n  Gradient magnitude stats:")
    print(f"    mean={grad_magnitudes.mean():.4f}, "
          f"max={grad_magnitudes.max():.4f}, "
          f"std={grad_magnitudes.std():.4f}")
    print(f"    >0.01: {(grad_magnitudes > 0.01).mean() * 100:.1f}%")
    print(f"    >0.1:  {(grad_magnitudes > 0.1).mean() * 100:.1f}%")
    print(f"    >1.0:  {(grad_magnitudes > 1.0).mean() * 100:.1f}%")

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
