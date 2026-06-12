# SPDX-License-Identifier: MIT
"""Differentiable brotli-byte rate surrogate (Lever 1) — the dominant-term attack.

The contest score `S = 100·d_seg + sqrt(10·d_pose) + 25·|archive.zip|/37_545_489`
is **rate-dominated** at the frontier (`25·177169/37.5e6 = 0.118`, 61.7 % of S).
The archive codec is **brotli q=11 over the zigzag-INT8 weight byte stream** (decoder)
plus **per-dim delta → uint8 → brotli** (latents). The existing C1a regularizer
(``tac.losses.cat_entropy_v2``) penalizes only the **memoryless** per-weight Shannon
entropy `H(W)` — a brotli lower bound that IGNORES the inter-symbol context (LZ
matches + order-N modeling) brotli actually exploits. The gap between `H(W)·numel/8`
and the real brotli byte count is exactly that redundancy.

This module closes that gap with two differentiable surrogates, both mirroring the
cat_entropy_v2 soft-histogram machinery (so the INT8 grid matches the codec):

* **(1a) Conditional (order-1) weight entropy** ``H(W_i | W_{i-1})`` along the
  brotli scan order, via a soft 2-D joint histogram of adjacent zigzag symbols.
  Order-1 conditional entropy is a TIGHTER brotli proxy than the marginal `H(W)`
  because brotli's order-N context modeling is bounded below by it and approaches
  it on scan-order-smooth INT8 streams. Penalizing it biases the decoder toward
  long zigzag runs (long brotli LZ matches).

  **MED-1 scan-order validation (2026-06-12,
  ``experiments/probe_lever1_entropy_vs_real_brotli.py``).** The order-1 conditional
  entropy is only a PREDICTIVE proxy for real brotli bytes when computed over the
  CODEC's ACTUAL scan order — the FULL ``state_dict()`` (every tensor, weights AND
  biases, in state-dict order) as ONE concatenated stream, exactly what
  ``codec.encode_decoder(quantize_state_dict(sd))`` brotli-compresses. With
  ``RateSurrogateConfig.codec_scan_order=True`` the probe measured **Spearman 0.90 /
  Pearson 0.999** vs real brotli decoder bytes across 8 real weight configs (basin
  EMA, sorted/smoothed, shuffled, 4 noise levels, random-init). The LEGACY per-tensor
  mode (``codec_scan_order=False``: named_modules Conv2d/Linear *weights only*,
  first-2000-symbols-per-tensor) measured **Spearman -0.14** — it is NOT a reliable
  rank proxy in the realistic high-entropy regime (the basin's trained INT8 weights
  use 195-228 of 255 symbols, ~1.5% zeros, so the truncated weights-only subset does
  not track the deployed stream's order-1 redundancy). The driver therefore uses
  ``codec_scan_order=True``. (Note: the marginal-vs-conditional inequality
  ``H(W|W_prev) <= H(W)`` still holds in BOTH modes — that mathematical lower-bound
  property was never the issue; the issue was that the per-tensor MODE measured the
  wrong subset/order, so its VALUE did not track real bytes. The fix is the
  deploy-faithful scan order, not the inequality.)

* **(1b) Latent temporal-delta entropy** ``H(quantize(z[p] − z[p−1]))`` per latent
  dim — the quantity the codec actually delta-codes. The latents path has NO
  training-time rate term today (only the decoder weights see C1a). Dashcam video
  is temporally redundant (CLAUDE.md L25 temporal-delta latent coding), so this is
  a real, currently-unexploited byte lever.

NO-FAKE discipline (this module ACTUALLY computes the entropies on the real
weights/latents the codec compresses — verified in
``src/tac/tests/test_rate_surrogate.py``):
- a smoother (longer-run) INT8 weight stream has LOWER conditional entropy than a
  high-entropy random stream (the mechanism, not a constant);
- the gradient flows to the real weights/latents so the optimizer can descend it;
- conditional entropy ``H(W|W_prev) <= H(W)`` marginal (the true-bound property).

Score-claim discipline (sister of cat_entropy_v2): this changes TRAINING DYNAMICS,
NOT archive bytes directly; NO score is asserted from its addition. The empirical
bit-spend proof is the paired A/B (lever-on vs off, same epoch budget) measuring
``archive_bytes`` at equal ``d_seg/d_pose`` — per CLAUDE.md Catalog #304
(closed-form-rate-without-empirical-bit-spend-proof: the A/B IS the proof).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Same INT8-symmetric grid + soft-assignment bandwidth as cat_entropy_v2 (the codec
# quantizes to {-127..127}; the surrogate MUST share that grid to be a real proxy).
_BIN_MIN = -127
_BIN_MAX = 127
_TARGET_LAYERS: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear)


@dataclass(frozen=True)
class RateSurrogateConfig:
    """Configuration for the differentiable rate surrogate.

    The INT8 grid + ``sigma`` mirror ``CatEntropyV2Config`` (the codec contract).
    """

    #: Soft-assignment bandwidth for the INT8 bin kernel (smaller = sharper grid
    #: commitment). Matches cat_entropy_v2's default.
    sigma: float = 0.2
    #: INT8-symmetric bin range {-127, ..., 127} — the codec's quantization grid.
    bin_min: int = _BIN_MIN
    bin_max: int = _BIN_MAX
    #: Numerical-stability epsilon inside log2 / division.
    eps: float = 1e-12
    #: ``w.abs().max()`` floor — tensors below this are skipped (effectively zero).
    max_abs_floor: float = 1e-12
    #: Per-tensor cap on the number of adjacent (prev, cur) pairs used for the
    #: order-1 joint histogram (the joint is over a contiguous scan-order slice, so
    #: the run structure is preserved — we take the FIRST ``pair_sample_size``
    #: adjacencies, NOT a random subsample which would destroy adjacency).
    pair_sample_size: int = 2000
    #: Number of uint8 bins for the latent-delta histogram (1b). The codec maps
    #: per-dim minmax → uint8, so the delta lives on a 256-level grid; we use a
    #: coarser soft histogram (the deltas are concentrated near zero).
    lat_bins: int = 256
    #: Soft-assignment bandwidth (in uint8-grid units) for the latent-delta hist.
    lat_sigma: float = 1.0
    #: When True, ``conditional_weight_entropy`` iterates the decoder's FULL
    #: ``state_dict()`` (every tensor — weights AND biases — in state-dict order)
    #: as ONE concatenated symbol stream, matching the EXACT density the vendored
    #: codec brotli-compresses (``codec.encode_decoder(quantize_state_dict(sd))``
    #: concatenates every tensor in ``sd.items()`` order into a single brotli
    #: stream). This is the DEPLOY-FAITHFUL proxy: the MED-1 probe
    #: (``experiments/probe_lever1_entropy_vs_real_brotli.py``) measured Spearman
    #: 0.90 / Pearson 0.999 between this and real brotli decoder bytes, vs Spearman
    #: -0.14 for the per-tensor (named_modules-weights-only, first-2000-truncated)
    #: mode. When False (the module-standalone default), the legacy per-tensor
    #: size-weighted average is used (single-tensor decoders are identical either
    #: way; the difference only appears on multi-tensor state dicts).
    codec_scan_order: bool = False
    #: Cap on the number of contiguous symbols used for the single-stream
    #: codec-scan-order conditional entropy (preserves adjacency across the WHOLE
    #: stream order; the basin's ~83K-weight decoder fits comfortably). Only used
    #: when ``codec_scan_order=True``.
    stream_sample_size: int = 60000


def _iter_quantizable_weights(decoder: nn.Module):
    """Yield ``(name, weight_tensor)`` for every quantizable Conv2d/Linear layer.

    Mirrors cat_entropy_v2's iteration so the SAME tensor set is regularized — the
    conditional surrogate is the order-1 refinement of the SAME marginal C1a sees.
    """
    for name, mod in decoder.named_modules():
        if isinstance(mod, _TARGET_LAYERS) and hasattr(mod, "weight"):
            w = mod.weight
            if w is None:
                continue
            yield name, w


def _codec_stream_normalized(decoder: nn.Module, cfg: RateSurrogateConfig):
    """Yield each ``state_dict()`` tensor (weights AND biases, in state-dict order),
    normalized to the INT8 grid (``t / (|t|.max()/127)``), as the codec sees them.

    The vendored ``codec.quantize_state_dict`` iterates ``sd.items()`` and quantizes
    EVERY tensor per-tensor-symmetric INT8 with its OWN ``|t|.max()/127`` scale, then
    ``encode_decoder`` concatenates the zigzag-INT8 of each (in the same order) into a
    SINGLE brotli stream. To track that exact density we (a) walk the gradient-carrying
    parameters in the SAME order ``state_dict()`` yields them, (b) per-tensor INT8-grid-
    normalize with the SAME ``|t|.max()/127`` scale (grad broken through the scale, like
    C1a). Tensors below the abs floor (e.g. a zero-init FiLM fc2) are skipped — they
    quantize to all-zero and contribute no order-1 structure.

    Critically this walks ``state_dict()`` (the deploy contract) — which includes biases
    the per-tensor named_modules-weights-only path omitted — so the surrogate's symbol
    stream is the one the codec compresses, not an abstract subset.
    """
    n_levels = float(max(cfg.bin_max, -cfg.bin_min))  # 127.0
    # Map state-dict NAME -> the live, gradient-carrying parameter. ``state_dict()``
    # returns DETACHED tensors (no grad_fn), so to keep the gradient we look the
    # parameter up by name (``named_parameters()`` shares the exact state_dict key for
    # every parameter). Buffers (e.g. a stored-pose buffer) are not in
    # ``named_parameters`` → they fall through as the detached state_dict tensor (no
    # grad, but the codec still quantizes them, so they belong in the stream). We iterate
    # ``state_dict()`` to get the EXACT codec write order.
    name_to_param = dict(decoder.named_parameters())
    for name, t in decoder.state_dict().items():
        live = name_to_param.get(name)
        w = live if live is not None else t  # buffers fall through (no grad)
        if w.numel() < 1:
            continue
        ma = w.abs().max().detach()
        if ma.item() < cfg.max_abs_floor:
            continue
        yield (w / (ma / n_levels)).flatten()


def _stream_conditional_entropy(
    stream: torch.Tensor, bins: torch.Tensor, cfg: RateSurrogateConfig
) -> torch.Tensor:
    """Order-1 conditional entropy ``H(cur|prev)`` of a single 1-D symbol stream
    (differentiable, soft-INT8-binned). Shared by the codec-scan-order path."""
    if stream.numel() < 2:
        return torch.zeros((), device=stream.device, dtype=torch.float32)
    if stream.numel() > cfg.stream_sample_size:
        stream = stream[: cfg.stream_sample_size]
    sa = _soft_bin_assignment(stream, bins, cfg.sigma, cfg.eps)  # (M, K)
    prev = sa[:-1]
    cur = sa[1:]
    joint = (prev.unsqueeze(2) * cur.unsqueeze(1)).mean(dim=0)  # (K, K)
    joint = joint / (joint.sum() + cfg.eps)
    marg_prev = joint.sum(dim=1)
    cond = joint / (marg_prev.unsqueeze(1) + cfg.eps)
    h_cond_per_a = -(cond * torch.log2(cond + cfg.eps)).sum(dim=1)
    return (marg_prev * h_cond_per_a).sum()


def _soft_bin_assignment(
    wn: torch.Tensor, bins: torch.Tensor, sigma: float, eps: float
) -> torch.Tensor:
    """Soft INT8-bin assignment ``sa[i, b] = softmax_b(-((wn[i]-b)/sigma)^2/2)``.

    Identical kernel to cat_entropy_v2 (so the grid commitment matches). Returns a
    ``(len(wn), len(bins))`` row-stochastic soft-histogram assignment matrix."""
    sa = torch.exp(-0.5 * ((wn.unsqueeze(1) - bins.unsqueeze(0)) / sigma).pow(2))
    return sa / (sa.sum(dim=1, keepdim=True) + eps)


def conditional_weight_entropy(
    decoder: nn.Module,
    config: RateSurrogateConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Order-1 conditional weight entropy ``H(W_i | W_{i-1})`` (bits/weight),
    size-weighted across decoder Conv2d/Linear tensors — Lever 1a.

    For each tensor the weights are normalized to the INT8 grid (``w / (max/127)``),
    soft-assigned to bins, and the **soft joint histogram of ADJACENT scan-order
    symbols** ``J[a,b] = mean_i sa_{i-1}[a]·sa_i[b]`` is formed. The conditional
    entropy ``H(cur|prev) = Σ_a J[a,·]·(-Σ_b (J[a,b]/J[a,·]) log2 (J[a,b]/J[a,·]))``
    is the differentiable order-1 Markov entropy. It is ``<= H(W)`` marginal (the
    cat_entropy_v2 quantity) by the conditioning inequality — a TRUE, conservative
    brotli lower bound.

    Returns a SCALAR ``torch.Tensor`` (grad-enabled when weights ``requires_grad``);
    ``0.0`` if no eligible tensor has ``>= 2`` elements above the abs floor.

    Two modes (``RateSurrogateConfig.codec_scan_order``):

    * ``codec_scan_order=True`` (the DEPLOY-FAITHFUL proxy, used by the driver):
      iterate the FULL ``state_dict()`` (every tensor — weights AND biases — in
      state-dict order) as ONE concatenated symbol stream, matching the exact density
      the vendored codec brotli-compresses. The MED-1 probe measured Spearman 0.90 /
      Pearson 0.999 vs real brotli decoder bytes across 8 real weight configs.
    * ``codec_scan_order=False`` (the module-standalone default): the legacy per-tensor
      size-weighted average over named_modules Conv2d/Linear weights only, truncated to
      the first ``pair_sample_size`` symbols per tensor. The MED-1 probe measured
      Spearman -0.14 vs real brotli bytes (NOT a reliable rank proxy in the realistic
      high-entropy regime) — kept only for backward compatibility / single-tensor
      callers (where the two modes coincide).
    """
    cfg = config or RateSurrogateConfig()
    if device is None:
        try:
            device_ = next(decoder.parameters()).device
        except StopIteration:
            device_ = torch.device("cpu")
    else:
        device_ = torch.device(device)

    bins = torch.arange(cfg.bin_min, cfg.bin_max + 1, device=device_, dtype=torch.float32)
    n_levels = float(max(cfg.bin_max, -cfg.bin_min))  # 127.0

    if cfg.codec_scan_order:
        # DEPLOY-FAITHFUL: one concatenated stream over the FULL state_dict, in the
        # exact order the codec writes it (the density brotli actually consumes).
        chunks = list(_codec_stream_normalized(decoder, cfg))
        if not chunks:
            return torch.zeros((), device=device_, dtype=torch.float32)
        stream = torch.cat([c.to(device_) for c in chunks])
        return _stream_conditional_entropy(stream, bins, cfg)

    weighted_entropy = torch.zeros((), device=device_, dtype=torch.float32)
    total_numel = 0
    for _, w in _iter_quantizable_weights(decoder):
        numel = w.numel()
        if numel < 2:
            continue
        ma = w.abs().max().detach()  # break grad through the scale (matches C1a)
        if ma.item() < cfg.max_abs_floor:
            continue
        wn = (w / (ma / n_levels)).flatten()
        # Take the FIRST contiguous slice (preserve adjacency — a random subsample
        # would destroy the scan-order run structure the conditional entropy reads).
        if wn.numel() > cfg.pair_sample_size + 1:
            wn = wn[: cfg.pair_sample_size + 1]
        sa = _soft_bin_assignment(wn, bins, cfg.sigma, cfg.eps)  # (M, K)
        prev = sa[:-1]  # (M-1, K)
        cur = sa[1:]  # (M-1, K)
        # Soft joint J[a,b] = mean_i prev[i,a] * cur[i,b].
        joint = (prev.unsqueeze(2) * cur.unsqueeze(1)).mean(dim=0)  # (K, K)
        joint = joint / (joint.sum() + cfg.eps)
        marg_prev = joint.sum(dim=1)  # (K,) P(prev=a)
        # Conditional P(cur=b | prev=a) = joint[a,b] / marg_prev[a].
        cond = joint / (marg_prev.unsqueeze(1) + cfg.eps)  # (K, K)
        h_cond_per_a = -(cond * torch.log2(cond + cfg.eps)).sum(dim=1)  # (K,)
        h_cond = (marg_prev * h_cond_per_a).sum()  # scalar bits/symbol
        weighted_entropy = weighted_entropy + numel * h_cond
        total_numel += numel

    return weighted_entropy / max(total_numel, 1)


def latent_delta_entropy(
    latents: torch.Tensor,
    config: RateSurrogateConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Temporal-delta entropy of the latents (Lever 1b) — bits per latent scalar.

    The codec maps the latents per-dim to uint8 (minmax), 1st-order temporal-delta
    codes them, zigzags, and brotli-compresses. This surrogate penalizes the soft
    Shannon entropy of the per-dim **temporal delta** ``dz[p] = z[p] − z[p−1]``
    mapped to the uint8 grid — rewarding temporally-smooth latents (small deltas →
    mostly-zero hi-byte stream → tiny brotli).

    ``latents`` is ``(n_pairs, latent_dim)`` (the FULL latent tensor; the delta is a
    GLOBAL temporal quantity, like C1a reads all the weights). Returns a SCALAR
    grad-enabled ``torch.Tensor``; ``0.0`` if ``n_pairs < 2``.
    """
    cfg = config or RateSurrogateConfig()
    if device is None:
        device_ = latents.device
    else:
        device_ = torch.device(device)
    z = latents.to(device_)
    if z.ndim != 2:
        raise ValueError(
            f"latent_delta_entropy expects (n_pairs, latent_dim); got {tuple(z.shape)}"
        )
    n_pairs, latent_dim = z.shape
    if n_pairs < 2:
        return torch.zeros((), device=device_, dtype=torch.float32)

    # Per-dim minmax → uint8 grid (matches the codec's per-dim minmax quantization).
    # Use a DETACHED min/max for the scale (the codec computes them from the data;
    # we want the gradient to flow through the delta values, not the range).
    zmin = z.min(dim=0, keepdim=True).values.detach()
    zmax = z.max(dim=0, keepdim=True).values.detach()
    span = (zmax - zmin).clamp_min(cfg.eps)
    z_u8 = (z - zmin) / span * (cfg.lat_bins - 1)  # (n_pairs, latent_dim) in [0, 255]
    dz = z_u8[1:] - z_u8[:-1]  # (n_pairs-1, latent_dim) signed deltas on the u8 grid

    # Soft histogram of the deltas over the signed grid [-(lat_bins-1), lat_bins-1].
    # The deltas concentrate near 0 (temporal smoothness); a soft hist over a grid
    # centered at 0 captures that concentration. Flatten across dims+pairs (the
    # codec's per-dim streams are concatenated before brotli, so a pooled entropy is
    # the right global proxy).
    grid = torch.arange(
        -(cfg.lat_bins - 1), cfg.lat_bins, device=device_, dtype=torch.float32
    )
    dz_flat = dz.reshape(-1)
    sa = torch.exp(-0.5 * ((dz_flat.unsqueeze(1) - grid.unsqueeze(0)) / cfg.lat_sigma).pow(2))
    sa = sa / (sa.sum(dim=1, keepdim=True) + cfg.eps)
    bp = sa.mean(dim=0)
    bp = bp / (bp.sum() + cfg.eps)
    entropy = -(bp * torch.log2(bp + cfg.eps)).sum()  # bits/delta scalar
    return entropy


def brotli_rate_surrogate(
    decoder: nn.Module,
    latents: torch.Tensor | None,
    config: RateSurrogateConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BOTH rate surrogates as ``(H_cond_weights, R_lat)``.

    The driver adds ``rate_lambda_w · H_cond + rate_lambda_lat · R_lat`` to the
    loss. ``latents=None`` returns ``R_lat = 0`` (decoder-only). The two terms are
    independent (weights vs latents) and each is a true bound, so they compose
    additively in the Lagrangian.
    """
    cfg = config or RateSurrogateConfig()
    h_cond = conditional_weight_entropy(decoder, cfg, device=device)
    if latents is None:
        r_lat = torch.zeros_like(h_cond)
    else:
        r_lat = latent_delta_entropy(latents, cfg, device=device)
    return h_cond, r_lat


# Score-claim discipline metadata (sister of cat_entropy_v2).
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False

__all__ = [
    "RateSurrogateConfig",
    "conditional_weight_entropy",
    "latent_delta_entropy",
    "brotli_rate_surrogate",
    "SCORE_CLAIM",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
]
