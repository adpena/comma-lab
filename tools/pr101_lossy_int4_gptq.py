#!/usr/bin/env python3
"""PR101 lossy int4 + GPTQ (Hessian-aware sequential calibrated quantization)
— audit criterion #4 for the lossy_int4 lane.

Per the 2026-05-08 audit memo (`feedback_implementation_vs_model_gap_audit_20260508.md`):

    - naive PTQ at int4: 37.42% rel_err (criterion #0 not dispatchable)
    - QAT at int4: 28.48% rel_err (criterion #1 not dispatchable)
    - per-channel scales: 30.41% rel_err (criterion #2 not dispatchable)
    - mixed-precision int4/int6/int8: 4.895% rel_err (criterion #3 conditional)
    - GPTQ Hessian-aware quantization: NOT TESTED (this tool — criterion #4)
    - AWQ activation-aware scaling: NOT TESTED (criterion #5; companion tool)

GPTQ (Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", https://arxiv.org/abs/2210.17323) uses
Optimal Brain Surgeon (OBS) style local-error minimization with calibration
data:

    1. For each weight matrix W (shape [out_features, in_features]) we
       gather a calibration activation matrix X (shape [n_samples, in_features])
       using forward hooks on the renderer's HNeRVDecoder forward pass.
    2. Compute the per-input-channel Hessian H = 2 * X^T @ X / n_samples
       + damping * mean(diag(H)) * I  (per the paper, damping = 0.01).
    3. Cholesky-invert H to get H_inv = (L L^T)^-1 = U^T U where U is the
       upper-Cholesky factor of H_inv. The diagonal of U gives per-column
       quantization step weights.
    4. Sequentially quantize columns of W in their natural order (paper
       suggests column-permutation by inverse Hessian diagonal but the
       pure-order variant matches "vanilla GPTQ"). For column j:
            error_j = (w[:, j] - quant(w[:, j])) / U[j, j]
            w[:, j+1:] -= error_j[:, None] * U[j, j+1:][None, :]
       This propagates the quantization error to remaining columns so the
       overall output error is OBS-optimal.
    5. Snap final quantized weights through the same int4 grid the encoder
       uses, feed into `pr101_lossy_int4_block_sweep.encode_tensor` to
       produce the actual archive bytes, and measure roundtrip rel_err.

Substrate-mismatch caveat (DOCUMENTED EXPLICITLY)

    GPTQ was designed for transformer LLMs where calibration activations
    come from a held-out text corpus. PR101 is a per-video overfit
    HNeRVDecoder — there is no natural "calibration corpus" of activations.
    We use SYNTHETIC activations from the renderer's forward pass on
    deterministic latent samples (sampled uniformly in [-1, 1]^28 with a
    fixed seed). This is the closest available substrate-faithful
    calibration set; it tests GPTQ's Hessian-aware error-propagation
    machinery on the actual weight tensors but cannot claim "test
    activations match deployment activations" the way an LLM GPTQ run
    would. The substrate gap is itself a finding: a faithful-activation
    GPTQ would require pre-computed PR106 frame latents, which are not
    distributed alongside the state_dict.

CLAUDE.md compliance (NON-NEGOTIABLES)

    * Pure CPU/MPS prep — no scorer load, no contest-CUDA dispatch from
      this tool. Evidence tagged ``[CPU-prep faithful audit-criterion-4 test]``.
    * Score never claimed; only BYTES + ROUNDTRIP rel_err are anchored.
    * promotion_eligible=False and ready_for_exact_eval_dispatch=False always.
      A good local rel_err only means cuda_eval_worth_testing=True; a separate
      byte-closed packet, runtime decoder, dispatch claim, and exact CUDA auth
      eval must flip any dispatch flag.
    * No /tmp paths in any persisted artifact.
    * Never invent CLI flags — uses the same argparse surface conventions
      as the prior 4 lossy_int4 variants.
    * This tool only adjudicates the measured GPTQ config. It does not
      falsify the lossy-int4 family or downstream byte-closed-runtime
      lanes; only this measured (calibration_size, damping, blocksize)
      point is anchored.
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

TOOL_NAME = "tools/pr101_lossy_int4_gptq.py"
SCHEMA_VERSION = "pr101_lossy_int4_gptq.v1"
INT4_RANGE = 7  # symmetric: [-7, +7], 15 levels (with zero)
DEFAULT_BLOCK_SIZE = 1024  # matches the 100,799 B archive anchor
ARCHIVE_OVERHEAD_BYTES = 16_094  # PR101 zip overhead constant
PR101_BROTLI_BASELINE_BYTES = 178_144
DISPATCH_THRESHOLD_PCT = 5.0
EVIDENCE_GRADE = "[CPU-prep faithful audit-criterion-4 test]"
DEFAULT_STATE_DICT_PATH = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)
DEFAULT_CALIBRATION_SAMPLES = 64  # GPTQ paper uses 128; 64 is enough for 28-D latents
DEFAULT_DAMPING_FRACTION = 0.01  # per-paper damping = 1% of mean(diag(H))
DEFAULT_BLOCK_SIZE_GPTQ = 128  # GPTQ "lazy batch update" block size for column updates
LOCAL_GPTQ_DISPATCH_BLOCKERS = (
    "local_gptq_proxy_signal_not_exact_eval_dispatch_ready",
    "byte_closed_int4_candidate_packet_missing",
    "no_int4_decoder_runtime_built",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "synthetic_activations_substrate_mismatch_vs_pr106_video_latents",
    "cpu_proxy_rel_err_not_score_evidence",
)


# ----------------------------------------------------------------------------
# Utility (mirrors patterns from pr101_lossy_int4_qat.py for consistency)
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
    """GPTQ Hessian inversion is solidly CPU-bounded for our small tensors
    (max ~3.5K cols). MPS allowed only as research-signal accelerator;
    contest-CUDA is for the eval lane downstream of this tool, not here.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def local_gptq_dispatch_contract(*, cuda_eval_worth_testing: bool) -> dict[str, Any]:
    """Return fail-closed dispatch metadata for local GPTQ research signals.

    Ready-for-exact-eval-dispatch is ALWAYS False from CPU/MPS evidence — only
    a [contest-CUDA] auth eval on the byte-closed runtime packet can flip it.
    """
    return {
        "cuda_eval_worth_testing": bool(cuda_eval_worth_testing),
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "dispatch_blockers": list(LOCAL_GPTQ_DISPATCH_BLOCKERS),
    }


# ----------------------------------------------------------------------------
# HNeRVDecoder — exact replica of submissions/pr103_pr106_final_runtime/inflate.py
# (re-instantiated locally so this tool stays self-contained and we don't have
# to import a submissions tree).
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
    """Build an HNeRVDecoder and inject the PR101 state_dict.

    PR101's state_dict layout matches FIXED_STATE_SCHEMA — same names as
    HNeRVDecoder's submodules. We use ``strict=False`` because the schema
    excludes the (Identity-aliased) skips.0 / skips.1 entries which have no
    parameters by construction.
    """
    decoder = HNeRVDecoder()
    expected_state = decoder.state_dict()
    payload: dict[str, torch.Tensor] = {}
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        if name not in expected_state:
            raise SystemExit(f"HNeRVDecoder has no parameter {name!r} (schema drift)")
        payload[name] = state[name].to(torch.float32)
    missing_keys, unexpected_keys = decoder.load_state_dict(payload, strict=False)
    # The decoder also has e.g. skips.0/1 which are Identity — no params expected.
    # `missing_keys` is therefore allowed to contain those identity entries.
    decoder.eval()
    return decoder


# ----------------------------------------------------------------------------
# Calibration activation collection
# ----------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    n_samples: int = DEFAULT_CALIBRATION_SAMPLES
    seed: int = 2026
    latent_low: float = -1.0
    latent_high: float = 1.0


def collect_calibration_activations(
    decoder: HNeRVDecoder,
    cfg: CalibrationConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run the decoder forward on synthetic latents and record the input
    activation matrix to each parameter-bearing layer.

    Returns a dict keyed by ``<module_name>.weight`` mapping to a numpy
    array of shape ``(n_total_positions, in_features_per_position)`` that is
    the matrix the layer's weight multiplies (Linear: rows = batch elements;
    Conv2d: rows = batch * H * W spatial positions, cols = in_ch * kH * kW).

    For Conv2d layers we use ``F.unfold`` semantics so the recorded matrix
    is exactly the one in the lowered ``W @ X^T`` formulation GPTQ assumes.
    """
    rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
    z = (
        cfg.latent_low
        + (cfg.latent_high - cfg.latent_low)
        * torch.rand(cfg.n_samples, LATENT_DIM, generator=rng)
    ).to(device)

    activations: dict[str, list[np.ndarray]] = {}
    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            x = inputs[0]
            if isinstance(module, nn.Linear):
                # x shape: (B, in_features) — use as-is (each row is a sample).
                rows = x.detach().reshape(-1, x.shape[-1])
            elif isinstance(module, nn.Conv2d):
                # Unfold to (B, in_ch * kH * kW, L) where L = number of output positions.
                # The GPTQ Hessian wants rows = positions, cols = in_ch * kH * kW.
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
                # unfolded: (B, in_ch * kH * kW, L) -> permute to (B*L, C*kH*kW)
                rows = unfolded.permute(0, 2, 1).reshape(-1, unfolded.shape[1])
            else:  # pragma: no cover - guarded by registration loop below
                return
            activations.setdefault(f"{name}.weight", []).append(rows.cpu().numpy().astype(np.float32))
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
    for k, lst in activations.items():
        if not lst:
            continue
        out[k] = np.concatenate(lst, axis=0)
    return out


# ----------------------------------------------------------------------------
# Encoder math: same per-block int4 grid as pr101_lossy_int4_block_sweep.
# We use it both for snapping the GPTQ-quantized weights through the wire
# format AND for the iterative quantization step inside GPTQ. Important: the
# encoder pre-snaps the per-block scale to fp16 to match the on-disk format.
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


def quantize_value_with_known_scale(value: float, scale: float) -> float:
    """Snap a single fp32 value through the symmetric int4 grid given a
    pre-computed fp16 scale. Returns the dequantized fp32 value.

    GPTQ proceeds in column order; once we've fixed a per-block scale (computed
    from the unmodified block of the matrix), each subsequent column update is
    measured against THAT scale. This matches the standard GPTQ "fixed grid
    per group" protocol.
    """
    if scale <= 0.0:
        return 0.0
    code = int(np.clip(np.round(value / scale), -INT4_RANGE, +INT4_RANGE))
    return float(code * scale)


def quantize_vector_with_known_scales(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Vectorized version of :func:`quantize_value_with_known_scale` for one
    block worth of elements. ``scales`` has the same shape as ``values``
    (per-element scale, replicated within a block).
    """
    safe = np.where(scales > 0, scales, 1.0)
    codes = np.clip(np.round(values / safe).astype(np.int32), -INT4_RANGE, +INT4_RANGE)
    return (codes * scales).astype(np.float32)


# ----------------------------------------------------------------------------
# GPTQ core: per-tensor sequential calibrated quantization
# ----------------------------------------------------------------------------

@dataclass
class GPTQConfig:
    block_size_encoder: int = DEFAULT_BLOCK_SIZE  # int4 wire-format block size
    block_size_gptq: int = DEFAULT_BLOCK_SIZE_GPTQ  # GPTQ lazy-update block size
    damping_fraction: float = DEFAULT_DAMPING_FRACTION
    actorder: bool = False  # column reordering by descending diag(H)


def _build_hessian(activations: np.ndarray, damping_fraction: float) -> np.ndarray:
    """Compute H = 2 X^T X / n + damping * mean(diag(H)) * I.

    activations shape: (n_samples, in_features). We average over n_samples so
    Hessian is scale-invariant to the calibration set size; the factor of 2
    is conventional (matches Frantar's reference implementation).
    """
    n = max(activations.shape[0], 1)
    h = (2.0 / n) * (activations.astype(np.float64).T @ activations.astype(np.float64))
    diag_mean = float(np.diag(h).mean())
    if diag_mean <= 0.0:
        diag_mean = 1.0  # all-zero activations: fall back to identity damping
    h += damping_fraction * diag_mean * np.eye(h.shape[0])
    return h


def _cholesky_inverse_upper(h: np.ndarray) -> np.ndarray:
    """Return the upper-triangular Cholesky factor U such that
    H_inv = U^T @ U (Frantar's "Hinv = (Cholesky upper)" convention).

    GPTQ's column-by-column update only needs U[j, j:] (the j-th row of U
    starting at column j). We build U via:
        L = cholesky(H)        (lower, H = L L^T)
        H_inv = L^-T @ L^-1
        U = cholesky(H_inv).T  (upper of an upper Cholesky factor)

    For numerical stability we instead invert L directly: L_inv = solve(L, I)
    so L_inv has shape (n,n) lower-triangular. Then H_inv = L_inv^T @ L_inv
    and we cholesky-decompose that to get U.
    """
    n = h.shape[0]
    # Try Cholesky; if it fails, add another small jitter.
    for jitter in (0.0, 1e-7, 1e-5, 1e-3):
        try:
            l_chol = np.linalg.cholesky(h + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            continue
    else:  # pragma: no cover - extreme degenerate case
        raise RuntimeError("GPTQ Hessian non-PSD even with jitter")
    # Solve L Y = I to get L^{-1}
    l_inv = np.linalg.solve(l_chol, np.eye(n))
    h_inv = l_inv.T @ l_inv
    # Now cholesky H_inv to get its upper factor
    for jitter in (0.0, 1e-9, 1e-7, 1e-5):
        try:
            u = np.linalg.cholesky(h_inv + jitter * np.eye(n)).T
            return u
        except np.linalg.LinAlgError:
            continue
    raise RuntimeError("GPTQ H_inv non-PSD even with jitter")


def _per_block_max_abs_scales(
    weight_row_major: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-block fp16 scales over the FLATTENED weight tensor in the
    encoder's serialization order (row-major flatten of the original tensor
    shape — this is what ``encode_tensor`` does).

    Returns (scales_per_element, n_blocks). ``scales_per_element`` has shape
    (numel,) so each weight element knows its block-scale at quantize time.
    """
    flat = weight_row_major.reshape(-1).astype(np.float64)
    n = flat.size
    n_full = n // block_size
    tail = n - n_full * block_size
    n_blocks = n_full + (1 if tail else 0)
    scale_per_block = np.zeros(n_blocks, dtype=np.float32)
    if n_full > 0:
        body = flat[: n_full * block_size].reshape(n_full, block_size)
        abs_max = np.maximum(np.abs(body).max(axis=1), 1e-12)
        scale_per_block[:n_full] = (abs_max / INT4_RANGE).astype(np.float16).astype(np.float32)
    if tail:
        tail_block = flat[n_full * block_size :]
        abs_max_t = max(float(np.abs(tail_block).max()), 1e-12)
        scale_per_block[n_full] = float(np.float16(abs_max_t / INT4_RANGE))
    scales_per_elem = np.empty(n, dtype=np.float32)
    if n_full > 0:
        scales_per_elem[: n_full * block_size] = np.repeat(
            scale_per_block[:n_full], block_size
        )
    if tail:
        scales_per_elem[n_full * block_size :] = scale_per_block[n_full]
    return scales_per_elem, scale_per_block


def gptq_quantize_tensor(
    weight: np.ndarray,
    activations: np.ndarray,
    *,
    cfg: GPTQConfig,
    original_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run GPTQ on ONE weight tensor.

    Inputs
    ------
    weight: shape (out_features, in_features). For Conv2d this is
        ``W.reshape(out_ch, in_ch * kH * kW)``.
    activations: shape (n_samples, in_features). Rows are calibration samples.

    Returns
    -------
    weight_quantized: same shape as ``weight``, post-GPTQ.
    stats: per-tensor diagnostic dict.

    The encoder's int4 wire format flattens the ORIGINAL tensor shape
    row-major (e.g. for Conv2d (out_ch, in_ch, kH, kW)). The block-scales it
    derives are based on that flattened order — which intermixes columns of
    the (out, in_full) reshape. To preserve byte-faithfulness AND OBS-style
    accounting, we:

      1. Compute per-element scales by max-abs over each encoder-block of the
         INITIAL weights (the GPTQ paper calls this "static" quantization
         grid; once chosen, scales don't move). The encoder sees these same
         scales when it quantizes the final weights, so byte counts are
         deterministic given the final weights.
      2. Run column-by-column GPTQ updates on the (out, in_full) reshape;
         when we quantize column j we look up that column's elements'
         per-element scales (computed in step 1) and snap each through the
         int4 grid using those scales.
      3. Propagate residual error to remaining columns via H_inv.

    The "static-scale" assumption introduces a small mismatch between the
    OBS-optimal grid (which would need a re-derivation as columns change)
    and the encoder's actual grid (which sees the FINAL weights). In
    practice this is the standard GPTQ-with-fixed-grouping protocol used in
    the open-source GPTQ-for-LLaMa, AutoGPTQ, and ExLlama implementations.
    """
    out_features, in_features = weight.shape
    if activations.shape[1] != in_features:
        raise ValueError(
            f"calibration activations have {activations.shape[1]} cols but "
            f"weight has {in_features} input features"
        )

    # Step 1: build per-element scales using the encoder's flattening order.
    # The encoder sees the tensor as ``original_shape`` flattened row-major.
    # For Linear: original_shape == (out, in), and reshape(out, in_full)
    # already equals the flattened row-major form by row.
    # For Conv2d: original_shape == (out_ch, in_ch, kH, kW), and the
    # reshape(out, in_ch * kH * kW) ALSO matches row-major flatten.
    # So in BOTH cases the (out, in_full) matrix's row-major flatten ==
    # the encoder's flat order. We compute scales over that.
    weight_for_scales = weight.copy().reshape(original_shape)
    scales_per_elem_flat, scale_per_block = _per_block_max_abs_scales(
        weight_for_scales, cfg.block_size_encoder,
    )
    # Reshape back to (out, in_full) so we can index by (row, col) cheaply.
    scales_matrix = scales_per_elem_flat.reshape(weight.shape).astype(np.float64)

    # Step 2: build Hessian + Cholesky-upper of H_inv.
    h = _build_hessian(activations, cfg.damping_fraction)
    u = _cholesky_inverse_upper(h)  # shape (in_features, in_features) upper

    # Optional column reordering by descending Hessian diagonal (act-order).
    perm = np.arange(in_features, dtype=np.int64)
    if cfg.actorder:
        diag_h = np.diag(h)
        perm = np.argsort(-diag_h)
        # Permute weight COLUMNS, scales COLUMNS, and Hessian (rows AND cols).
        weight_p = weight[:, perm].astype(np.float64).copy()
        scales_p = scales_matrix[:, perm].copy()
        h_p = h[np.ix_(perm, perm)]
        u = _cholesky_inverse_upper(h_p)
    else:
        weight_p = weight.astype(np.float64).copy()
        scales_p = scales_matrix.copy()

    # Step 3: column-by-column quantization with lazy-batch updates.
    # We process columns in groups of cfg.block_size_gptq for efficiency
    # (matches Frantar's "Quant-Block-Size" parameter; the math is the same
    # as one-by-one but the matmul updates are batched).
    quantized_p = weight_p.copy()
    bg = max(1, int(cfg.block_size_gptq))
    losses_per_col_sq = np.zeros(in_features, dtype=np.float64)
    for col_start in range(0, in_features, bg):
        col_end = min(in_features, col_start + bg)
        # Per-column quantization with intra-block error accumulation.
        # We need an (out, group_size) buffer of accumulated propagated error.
        group_w = quantized_p[:, col_start:col_end].copy()  # (out, group)
        group_q = np.zeros_like(group_w)
        group_err = np.zeros_like(group_w)
        u_block = u[col_start:col_end, col_start:col_end]
        u_after = u[col_start:col_end, col_end:]
        for j in range(col_end - col_start):
            col_idx = col_start + j
            d = float(u_block[j, j])
            if d <= 0.0 or not np.isfinite(d):
                # Degenerate column — skip update, just snap as-is.
                col_w = group_w[:, j]
                col_scales = scales_p[:, col_idx]
                col_q = quantize_vector_with_known_scales(col_w.astype(np.float32),
                                                           col_scales.astype(np.float32))
                group_q[:, j] = col_q
                continue
            col_w = group_w[:, j]  # current (post-prop) column
            col_scales = scales_p[:, col_idx]
            col_q = quantize_vector_with_known_scales(
                col_w.astype(np.float32), col_scales.astype(np.float32),
            ).astype(np.float64)
            group_q[:, j] = col_q
            err = (col_w - col_q) / d  # (out,)
            losses_per_col_sq[col_idx] = float(np.mean(err ** 2))
            group_err[:, j] = err
            # Propagate to remaining columns within this group.
            if j + 1 < (col_end - col_start):
                group_w[:, j + 1 :] -= np.outer(err, u_block[j, j + 1 :])
        # Now propagate the GROUP's accumulated error to columns AFTER the group.
        if u_after.size:
            # group_err shape (out, group); u_after shape (group, after_cols).
            quantized_p[:, col_end:] -= group_err @ u_after
        quantized_p[:, col_start:col_end] = group_q

    # Undo the act-order permutation, if any.
    if cfg.actorder:
        quantized = np.empty_like(quantized_p)
        quantized[:, perm] = quantized_p
    else:
        quantized = quantized_p

    return quantized.astype(np.float32), {
        "n_blocks_encoder": int(scale_per_block.size),
        "n_columns_in": int(in_features),
        "n_rows_out": int(out_features),
        "actorder": bool(cfg.actorder),
        "damping_fraction": float(cfg.damping_fraction),
        "block_size_gptq": int(cfg.block_size_gptq),
        "mean_col_loss_sq": float(losses_per_col_sq.mean()),
        "max_col_loss_sq": float(losses_per_col_sq.max()),
    }


# ----------------------------------------------------------------------------
# Encode / measure: same encoder pipeline as pr101_lossy_int4_qat.py so the
# tools agree byte-for-byte on what a "quantized state -> archive bytes"
# count means.
# ----------------------------------------------------------------------------

def encode_quantized_state_to_archive_bytes(
    quantized: dict[str, np.ndarray],
    *,
    block_size: int,
) -> tuple[int, int, int]:
    """Re-use the canonical encoder to produce the actual archive byte count.

    Returns (raw_payload_bytes, brotli_bytes, archive_bytes).
    """
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
    """Same math as `pr101_lossy_int4_roundtrip_test.rel_err_stats` but on
    the GPTQ-quantized values vs ORIGINAL fp32 weights.
    """
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


# ----------------------------------------------------------------------------
# Bias quantization — biases have no associated input activations so GPTQ's
# Hessian machinery does not apply. We reuse the encoder's per-block max-abs
# scheme directly (same as the naive PTQ tool); biases are typically small
# (1-D per-channel) so this is the correct accounting.
# ----------------------------------------------------------------------------

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
# Per-tensor dispatch: GPTQ for weights, naive PTQ for biases.
# ----------------------------------------------------------------------------

def quantize_pr101_state(
    fp32_state: dict[str, np.ndarray],
    activations: dict[str, np.ndarray],
    *,
    cfg: GPTQConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Apply GPTQ to all weight tensors and naive PTQ to bias tensors.

    Returns (quantized_state, per_tensor_stats).
    """
    quantized: dict[str, np.ndarray] = {}
    per_tensor_stats: dict[str, dict[str, Any]] = {}

    for name, shape in FIXED_STATE_SCHEMA:
        tensor = fp32_state[name]
        if name.endswith(".bias") or tensor.ndim <= 1:
            # No GPTQ for biases — use the encoder's PTQ grid directly.
            quantized[name] = naive_int4_quantize_tensor(tensor, cfg.block_size_encoder)
            per_tensor_stats[name] = {"path": "naive_ptq_bias", "shape": list(shape)}
            continue
        # GPTQ path.
        if name not in activations:
            # No calibration available for this layer (shouldn't happen for
            # the named conv/linear params, but be defensive).
            quantized[name] = naive_int4_quantize_tensor(tensor, cfg.block_size_encoder)
            per_tensor_stats[name] = {"path": "naive_ptq_no_calibration", "shape": list(shape)}
            continue
        if tensor.ndim == 4:
            # Conv2d: (out_ch, in_ch, kH, kW) -> (out_ch, in_ch * kH * kW)
            out_ch = tensor.shape[0]
            w2d = tensor.reshape(out_ch, -1).astype(np.float32)
        elif tensor.ndim == 2:
            w2d = tensor.astype(np.float32)
        else:
            quantized[name] = naive_int4_quantize_tensor(tensor, cfg.block_size_encoder)
            per_tensor_stats[name] = {"path": "naive_ptq_unsupported_ndim", "shape": list(shape)}
            continue
        w_q, stats = gptq_quantize_tensor(
            w2d, activations[name], cfg=cfg, original_shape=tuple(shape),
        )
        quantized[name] = w_q.reshape(shape).astype(np.float32)
        per_tensor_stats[name] = {"path": "gptq", "shape": list(shape), **stats}
    return quantized, per_tensor_stats


def classify_cpu_proxy_candidate(
    *,
    weighted_avg_rel_err_pct: float,
    archive_bytes: int,
) -> tuple[str, bool]:
    """Fail-closed CPU proxy classification.

    Reuses the same threshold semantics as the prior 4 lossy_int4 variants:
    rel_err < 2% AND archive < baseline → CUDA-EVAL-WORTH-TESTING; rel_err
    < 5% → CONDITIONAL; otherwise NOT_DISPATCHABLE. We never set
    ready_for_exact_eval_dispatch here — see the dispatch contract.
    """
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
# Main entry point
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT_PATH)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                   help="Encoder int4 block size (matches the wire format)")
    p.add_argument("--block-size-gptq", type=int, default=DEFAULT_BLOCK_SIZE_GPTQ,
                   help="GPTQ lazy-batch column update block size")
    p.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES,
                   help="Number of synthetic latent samples for activation collection")
    p.add_argument("--damping-fraction", type=float, default=DEFAULT_DAMPING_FRACTION,
                   help="GPTQ Hessian damping = damping_fraction * mean(diag(H))")
    p.add_argument("--actorder", action="store_true",
                   help="Reorder columns by descending Hessian diag (paper variant)")
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu",
                   help="Forward-pass device for activation collection (CPU is fine)")
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
    print(f"[GPTQ] device: {device}")

    sd = torch.load(args.state_dict, map_location="cpu", weights_only=False)
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
    # Build decoder + collect calibration activations.
    # ------------------------------------------------------------------
    print(f"[GPTQ] building HNeRVDecoder + collecting {args.calibration_samples} synthetic-latent activations")
    decoder = load_state_into_decoder(fp32_state_torch).to(device)
    cal_cfg = CalibrationConfig(n_samples=args.calibration_samples, seed=args.seed)
    activations = collect_calibration_activations(decoder, cal_cfg, device)
    activation_summary = {
        k: {"shape": list(v.shape), "abs_mean": float(np.abs(v).mean())}
        for k, v in activations.items()
    }
    print(f"[GPTQ] collected activations for {len(activations)} weight tensors")

    # ------------------------------------------------------------------
    # Pre-train naive PTQ baseline (the falsification baseline).
    # ------------------------------------------------------------------
    pre_quant: dict[str, np.ndarray] = {}
    for name, _shape in FIXED_STATE_SCHEMA:
        pre_quant[name] = naive_int4_quantize_tensor(fp32_state[name], args.block_size)
    pre_metrics = measure_roundtrip_rel_err(pre_quant, fp32_state)
    pre_raw, pre_brotli, pre_archive = encode_quantized_state_to_archive_bytes(
        pre_quant, block_size=args.block_size,
    )
    print(f"\n[GPTQ] naive-PTQ baseline rel_err = {pre_metrics['weighted_avg_rel_err_pct']:.4f}%, "
          f"archive_bytes = {pre_archive:,} B")

    # ------------------------------------------------------------------
    # GPTQ.
    # ------------------------------------------------------------------
    cfg = GPTQConfig(
        block_size_encoder=args.block_size,
        block_size_gptq=args.block_size_gptq,
        damping_fraction=args.damping_fraction,
        actorder=args.actorder,
    )
    print(f"\n[GPTQ] running GPTQ (damping={args.damping_fraction}, "
          f"block_size_gptq={args.block_size_gptq}, actorder={args.actorder})")
    gptq_quant, per_tensor_stats = quantize_pr101_state(
        fp32_state, activations, cfg=cfg,
    )

    post_metrics = measure_roundtrip_rel_err(gptq_quant, fp32_state)
    raw_bytes, brotli_bytes, archive_bytes = encode_quantized_state_to_archive_bytes(
        gptq_quant, block_size=args.block_size,
    )
    rel_err_pct = post_metrics["weighted_avg_rel_err_pct"]
    print(f"\n[GPTQ] post-GPTQ rel_err = {rel_err_pct:.4f}% "
          f"(naive baseline {pre_metrics['weighted_avg_rel_err_pct']:.4f}%)")
    print(f"[GPTQ] post-GPTQ archive_bytes = {archive_bytes:,} B "
          f"(naive {pre_archive:,} B; PR101 brotli baseline {PR101_BROTLI_BASELINE_BYTES:,} B)")

    verdict, cuda_eval_worth_testing = classify_cpu_proxy_candidate(
        weighted_avg_rel_err_pct=rel_err_pct,
        archive_bytes=archive_bytes,
    )
    dispatch_contract = local_gptq_dispatch_contract(
        cuda_eval_worth_testing=cuda_eval_worth_testing,
    )

    if device.type == "cuda":
        evidence_grade = "[CUDA-research-signal faithful audit-criterion-4 test]"
    elif device.type == "mps":
        evidence_grade = "[MPS-research-signal faithful audit-criterion-4 test]"
    else:
        evidence_grade = EVIDENCE_GRADE

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"reports/raw/pr101_lossy_int4_gptq_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": evidence_grade,
        "evidence_semantics": (
            "gptq_calibrated_int4_byte_anchor_plus_roundtrip_rel_err_no_score"
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
        "gptq": {
            "block_size_gptq": args.block_size_gptq,
            "damping_fraction": args.damping_fraction,
            "actorder": args.actorder,
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
        "gptq_post_calibration": {
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
        "supersedes_prior_FALSIFIED_tag": False,  # this is criterion #4, separate from prior
        "reactivation_criteria_tested": ["GPTQ_calibration"],
        "reactivation_criteria_remaining": ["AWQ_calibration"],
        "n_total_elements": post_metrics["n_total_elements"],
        "n_nontrivial_elements": post_metrics["n_nontrivial_elements"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[GPTQ] manifest: {manifest_path}")

    # ------------------------------------------------------------------
    # Cathedral evidence row.
    # ------------------------------------------------------------------
    if cuda_eval_worth_testing:
        dispatch_verdict = "CUDA-EVAL-WORTH-TESTING-pending-runtime-packet"
    else:
        dispatch_verdict = "DEFERRED-pending-research"
    evidence_row = {
        "technique": "lossy_int4_quantization_gptq",
        "empirical_archive_bytes": archive_bytes,
        "empirical_distortion_increase_pct": rel_err_pct,
        "evidence_grade": evidence_grade,
        "evidence_marker": evidence_grade,
        "evidence_semantics": (
            "gptq_calibrated_int4_byte_anchor_plus_roundtrip_rel_err_no_score"
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
            f"(GPTQ samples={args.calibration_samples} damping={args.damping_fraction} "
            f"block_size_gptq={args.block_size_gptq}; rel_err={rel_err_pct:.2f}%)"
        ),
        "contest_dispatch_verdict": dispatch_verdict,
        "supersedes_prior_FALSIFIED_tag": False,
        "reactivation_criteria_tested": ["GPTQ_calibration"],
        "reactivation_criteria_remaining": ["AWQ_calibration"],
        "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evidence_row) + "\n")
    print(f"[GPTQ] evidence row appended: {args.evidence_jsonl}")

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
