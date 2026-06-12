# SPDX-License-Identifier: MIT
"""cool_chic AR-Gaussian entropy coder — REAL lossless range coding of the latents.

This is the missing primitive that makes Cool-Chic a real rate carrier (TRACK B
step 1). The L0 SKETCH stored latents as RAW int16 (``archive.py``
``LATENT_*_BLOB``, 99.92 % of the 18.4 MB smoke archive) while the AR prior net
was INERT at the byte level (a train-time rate proxy that never drove a coder).

This module closes that gap: it consumes the AR prior's per-element conditional
Gaussian density ``(mean, log_scale)`` exactly as ``architecture.compute_ar_log_prob``
does, derives a per-symbol discrete PMF over an integer quantization grid, and
RANGE-CODES every latent integer against that PMF. Decode reconstructs the exact
integer grid (lossless w.r.t. the grid) by replaying the SAME causal AR chain.

Reuse target (SEARCH-FIRST, do NOT reimplement a coder):
    ``tac.lossless.range_coder`` — pure-Python CACM87 arithmetic coder with a
    per-symbol adaptive-CDF API (``RangeEncoder.encode(symbol, cumulative,
    total)`` / ``RangeDecoder.target(total)`` + ``update(...)``). Its test
    ``test_incremental_range_coder_roundtrips_dynamic_frequencies`` already
    proves it round-trips with a DIFFERENT frequency table per symbol — exactly
    what a per-latent Gaussian AR prior emits. No external deps, byte-clean,
    deterministic. We add ONLY the Gaussian→PMF discretization + the AR-causal
    driver + tail-escape; the entropy-coding math is the reused primitive.

Why this is the rate lever (the Track-B thesis):
    Cool-Chic spends ~all its bytes on per-pair LATENTS, not decoder weights.
    HNeRV's floor is its decoder-weight blob (~91 % of its archive). If the
    AR-coded latent rate falls below that decoder-weight floor, Cool-Chic is a
    structurally lower-floor carrier. This coder is what lets us MEASURE that.

Full-stack synergy (Layer 1 = this coder ↔ Layer 2 = in-curriculum rate lever):
    The discrete PMF below is the EXACT discretization of the SAME Gaussian
    ``log p(z | z_{t-1})`` that ``compute_ar_log_prob`` returns and the Layer-2
    differentiable rate surrogate trains. So the train-time predicted rate
    (``-log2 p`` summed) and the deploy-time coded bytes are the SAME quantity
    up to (a) the bin-width discretization (``+ log2(q)`` per element, a constant
    offset) and (b) the finite range-coder overflow (< 2 bytes total). Training
    against ``-log2 p`` therefore directly minimizes the bytes THIS coder emits —
    the train/deploy rate gap is closed by construction. See
    ``ar_gaussian_predicted_bits`` (the differentiable-equivalent estimator) which
    matches ``encode_latents``' actual byte count within the discretization offset.

CLAUDE.md compliance:
- NO FAKE: this ACTUALLY range-codes and the decoder reconstructs the exact
  integer grid; round-trip is asserted in tests on REAL trained latents. The
  bytes ACTUALLY shrink (measured, not claimed).
- No silent device defaults (operates on CPU tensors / numpy; caller controls).
- No scorer load, no /tmp paths.
- Deterministic (fixed grid, fixed tail-escape, fixed coder).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch

from tac.lossless.range_coder import RangeDecoder, RangeEncoder

# --------------------------------------------------------------------------- #
# Grid + coder constants                                                      #
# --------------------------------------------------------------------------- #

# Range-coder frequency-table total. The CACM87 coder needs every symbol to
# have frequency >= 1, and TOTAL must be small enough that ``range // total``
# never underflows (STATE_BITS=32, so TOTAL <= 2**16 is the safe ceiling per
# the coder's invariants). 1<<14 leaves headroom for the escape symbol while
# keeping ~14 bits of PMF resolution (well below the per-symbol Gaussian
# entropy, so quantizing the PMF to the table costs < 0.01 bit/symbol).
RANGE_CODER_TOTAL: int = 1 << 14

# Coding-window half-width in (median) std units. The AR Gaussian is coded on the
# integer grid ``k`` over ``[center - WINDOW_SIGMAS*med_sigma, center + ...]``;
# anything outside is sent via the ESCAPE symbol + a raw int32 (lossless for
# outliers). 5 median-sigma captures > 1 - 5.7e-7 of a Gaussian's mass; elements
# whose own sigma exceeds the median (heavy tail) escape — rare and lossless.
WINDOW_SIGMAS: float = 5.0

# Hard cap on the symbol-window width (number of integer bins). Caps the per-symbol
# PMF size + the range-coder table-validation cost; wide-sigma pairs escape past it.
MAX_WINDOW_BINS: int = 1024

# Section magic for the entropy-coded blobs (sanity tag inside the section).
ECCV_MAGIC: bytes = b"ECC1"


# --------------------------------------------------------------------------- #
# Grid quantization (the latent -> integer mapping the coder is lossless over) #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LatentGrid:
    """Uniform scalar quantization grid: ``k = round(z / step)``; ``z_hat = k*step``.

    A SINGLE step per latent tensor (coarse / fine). This is the score-relevant
    quantization; the entropy coder is LOSSLESS w.r.t. the integers ``k`` (decode
    reconstructs exact ``k`` -> exact ``z_hat``). Choosing ``step`` is the rate /
    distortion operating point (Layer-2 can train against it).
    """

    step: float

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """float latents -> int64 grid symbols ``k``."""
        if self.step <= 0:
            raise ValueError("grid step must be positive")
        return torch.round(z.detach().to(torch.float32) / self.step).to(torch.int64)

    def dequantize(self, k: torch.Tensor | np.ndarray) -> torch.Tensor:
        """int grid symbols ``k`` -> reconstructed float latents ``z_hat``."""
        if isinstance(k, np.ndarray):
            k = torch.from_numpy(k.astype(np.int64))
        return k.to(torch.float32) * float(self.step)


def choose_grid_step(z: torch.Tensor, *, bits_per_std: float = 5.0) -> float:
    """Pick a grid step giving ~``2**bits_per_std`` levels per std of the latents.

    The L0 int16 grid wasted ~13K levels/std (1.4e-5 step on std~0.07). A
    score-relevant grid needs far fewer; ``bits_per_std=5`` -> 32 levels/std is a
    conservative default (caller / Layer-2 may sweep). Returned step is a clean
    float; provenance stored in the section header so decode is exact.
    """
    std = float(z.detach().to(torch.float32).std().clamp(min=1e-8))
    levels_per_std = 2.0 ** bits_per_std
    return std / levels_per_std


# --------------------------------------------------------------------------- #
# Gaussian -> discrete PMF (the per-symbol CDF the range coder consumes)       #
# --------------------------------------------------------------------------- #

def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF Φ(x) = 0.5*(1 + erf(x/sqrt2)), vectorized via torch.erf.

    torch.erf is deterministic, a project dep, and vectorized (no per-element
    Python loop, no scipy dep). float64 throughout for grid-stable CDFs.
    """
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    out = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
    return out.numpy()


def _gaussian_bin_pmf_batch(
    mean: np.ndarray,
    sigma: np.ndarray,
    k_grid: np.ndarray,
    step: float,
) -> np.ndarray:
    """Vectorized discrete PMF for a BATCH of Gaussians over a shared bin window.

    Args:
        mean, sigma: ``(E,)`` per-element Gaussian params (E = elements in a pair).
        k_grid: ``(E, W)`` integer bin indices per element (each row its own
            window ``[k_center-half, k_center+half]``).
        step: grid step.

    Returns:
        ``(E, W)`` non-normalized PMF (bin mass), float64. Bin ``k`` covers the
        real interval ``[(k-0.5)*step, (k+0.5)*step)``. EXACT discretization of
        the per-element continuous Gaussian density.
    """
    mean_col = mean[:, None]
    sigma_col = np.maximum(sigma[:, None], 1e-8)
    lower = (k_grid.astype(np.float64) - 0.5) * step
    upper = (k_grid.astype(np.float64) + 0.5) * step
    z_lower = (lower - mean_col) / sigma_col
    z_upper = (upper - mean_col) / sigma_col
    return _phi(z_upper) - _phi(z_lower)


def _pmf_rows_to_freq_tables(pmf: np.ndarray, *, total: int) -> np.ndarray:
    """Vectorized: ``(E, W)`` PMF rows -> ``(E, W)`` integer freq tables (each sums to ``total``).

    Every entry >= 1 (range-coder invariant). Same min-1 + residual
    redistribution discipline as ``tac.lossless.range_coder.normalize_probability_rows``
    (the reused coder's own normalizer), batched here so a whole pair is built in
    one numpy pass. Deterministic stable-argsort tie-breaking matches the scalar
    path so encode/decode agree bit-for-bit.
    """
    p = np.clip(np.asarray(pmf, dtype=np.float64), 0.0, None)
    e, w = p.shape
    s = p.sum(axis=1, keepdims=True)
    # degenerate rows (all-zero mass) -> uniform fallback (still lossless)
    degenerate = (s[:, 0] <= 0.0)
    if np.any(degenerate):
        p[degenerate] = 1.0
        s = p.sum(axis=1, keepdims=True)
    scaled = np.maximum(1, np.rint(p / s * total).astype(np.int64))
    deltas = total - scaled.sum(axis=1)
    for row in range(e):
        d = int(deltas[row])
        if d > 0:
            order = np.argsort(-p[row], kind="stable")
            scaled[row, order[:d]] += 1
        elif d < 0:
            order = np.argsort(p[row], kind="stable")
            remaining = -d
            for col in order.tolist():
                take = min(int(scaled[row, col]) - 1, remaining)
                if take > 0:
                    scaled[row, col] -= take
                    remaining -= take
                if remaining == 0:
                    break
    if np.any(scaled.sum(axis=1) != total) or np.any(scaled <= 0):
        raise ValueError("failed to normalize batched Gaussian PMF into positive freq tables")
    return scaled


def _cumulative(freqs) -> tuple[list[int], int]:
    cum = [0]
    run = 0
    for f in freqs:
        run += int(f)
        cum.append(run)
    return cum, run


# --------------------------------------------------------------------------- #
# AR prior driver — exact (mean, sigma) per element, replayable at decode time #
# --------------------------------------------------------------------------- #

class _ARPriorEvaluator:
    """Replays the AR prior chain to produce per-element (mean, sigma) tensors.

    Mirrors ``CoolChicSubstrate._ar_log_prob_chain`` EXACTLY: pair ``t`` is
    conditioned on the reconstructed previous-pair latent ``z_hat_{t-1}`` (zero
    context for ``t=0``). Because decode reconstructs ``z_hat`` losslessly, the
    decoder computes bit-identical params — the AR causal chain is decodable.

    We hold the prior net (a tiny torch ``_ARPriorNet``) and run it on CPU under
    ``no_grad``. The net is loaded from the archive's AR_PRIOR_BLOB at decode.
    """

    def __init__(self, prior_net: torch.nn.Module) -> None:
        self.net = prior_net.eval()

    @torch.no_grad()
    def params_for_pair(self, prev_latent: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """prev_latent ``(1, C, h, w)`` float32 -> (mean, sigma) numpy arrays."""
        mean, log_scale = self.net(prev_latent)
        sigma = torch.exp(log_scale)
        return (
            mean.squeeze(0).to(torch.float64).numpy(),
            sigma.squeeze(0).to(torch.float64).numpy(),
        )


# --------------------------------------------------------------------------- #
# Encode / decode one latent chain (coarse OR fine)                            #
# --------------------------------------------------------------------------- #

def _pair_window(
    mean_flat: np.ndarray, sigma_flat: np.ndarray, step: float
) -> tuple[np.ndarray, int, np.ndarray]:
    """Per-element window centers + shared half-width + ``(E, W)`` bin-index grid.

    Center of element ``j`` is ``round(mu_j / step)`` (the Gaussian mean in grid
    units); half-width is SHARED across the pair (max over elements, capped at
    ``MAX_WINDOW_BINS//2``) so the windows form a clean rectangular ``(E, W)``
    array for the vectorized PMF. Encode and decode call this identically.
    """
    k_center = np.rint(mean_flat / step).astype(np.int64)  # (E,)
    # Shared half-width sized to the pair's MEDIAN sigma (not max): rare wide-sigma
    # elements fall outside their window and are sent losslessly via ESCAPE. This
    # keeps the rectangular (E, W) window tight (a single huge-sigma element no
    # longer dilates the whole pair's table) — faster AND closer to the true
    # per-element entropy. Median is robust to the heavy sigma tail.
    med_sigma = float(np.median(sigma_flat))
    half = min(MAX_WINDOW_BINS // 2, max(2, math.ceil(WINDOW_SIGMAS * med_sigma / step)))
    offsets = np.arange(-half, half + 1, dtype=np.int64)  # (W,)
    k_grid = k_center[:, None] + offsets[None, :]  # (E, W)
    return k_center, half, k_grid


def encode_latent_chain(
    latents: torch.Tensor,
    prior_net: torch.nn.Module,
    grid: LatentGrid,
) -> bytes:
    """Range-code a ``(num_pairs, C, h, w)`` latent chain against its AR Gaussian.

    Lossless w.r.t. the integer grid: decode reconstructs the exact ``k`` symbols.
    Returns the entropy-coded blob (header + range stream).
    """
    if latents.dim() != 4:
        raise ValueError("latents must be 4-D (num_pairs, C, h, w)")
    num_pairs, c, h, w = latents.shape
    k_all = grid.quantize(latents).numpy()  # (num_pairs, C, h, w) int64
    evaluator = _ARPriorEvaluator(prior_net)

    enc = RangeEncoder()
    escapes: list[tuple[int, int]] = []  # (flat_position, raw_k) for out-of-window
    flat_pos = 0

    zero_prev = torch.zeros((1, c, h, w), dtype=torch.float32)
    prev_latent = zero_prev
    for t in range(num_pairs):
        mean_arr, sigma_arr = evaluator.params_for_pair(prev_latent)
        k_pair = k_all[t]  # (C, h, w)
        mean_flat = mean_arr.ravel()
        sigma_flat = np.maximum(sigma_arr.ravel(), 1e-8)
        k_flat = k_pair.ravel().astype(np.int64)

        k_center, half, k_grid = _pair_window(mean_flat, sigma_flat, grid.step)
        pmf = _gaussian_bin_pmf_batch(mean_flat, sigma_flat, k_grid, grid.step)
        freqs = _pmf_rows_to_freq_tables(pmf, total=RANGE_CODER_TOTAL - 1)  # (E, W)
        # escape slot column (freq 1) appended -> cumulative built per row below
        cum = np.concatenate(
            [np.zeros((freqs.shape[0], 1), np.int64), np.cumsum(freqs, axis=1)], axis=1
        )
        cum = np.concatenate([cum, (cum[:, -1:] + 1)], axis=1)  # escape -> +1
        totals = cum[:, -1]
        in_window = (k_flat >= k_grid[:, 0]) & (k_flat <= k_grid[:, -1])
        sym = np.where(in_window, k_flat - k_grid[:, 0], freqs.shape[1])  # escape index = W

        for j in range(k_flat.shape[0]):
            if not in_window[j]:
                escapes.append((flat_pos, int(k_flat[j])))
            enc.encode(symbol=int(sym[j]), cumulative=cum[j].tolist(), total=int(totals[j]))
            flat_pos += 1

        # reconstruct z_hat for this pair to drive the next AR step (causal, exact)
        prev_latent = grid.dequantize(torch.from_numpy(k_pair)).unsqueeze(0)

    stream = enc.finish()

    # Header: MAGIC(4) + num_pairs(u32) + C(u32) + h(u32) + w(u32) + step(f64)
    #         + n_escapes(u32) + stream_len(u32) ; then escapes (u32 pos + i32 k)*
    header = struct.pack(
        "<4sIIIIdII",
        ECCV_MAGIC, num_pairs, c, h, w, float(grid.step), len(escapes), len(stream),
    )
    esc_bytes = b"".join(struct.pack("<Ii", pos, k) for pos, k in escapes)
    return header + esc_bytes + stream


_ECCV_HEADER_FMT = "<4sIIIIdII"
_ECCV_HEADER_SIZE = struct.calcsize(_ECCV_HEADER_FMT)


def decode_latent_chain(blob: bytes, prior_net: torch.nn.Module) -> torch.Tensor:
    """Inverse of ``encode_latent_chain``: reconstruct the exact ``z_hat`` tensor.

    Replays the SAME causal AR chain; reconstructs each ``k`` symbol, applies
    escapes, dequantizes. Returns ``(num_pairs, C, h, w)`` float32 ``z_hat``.
    """
    if len(blob) < _ECCV_HEADER_SIZE:
        raise ValueError("entropy blob too short for header")
    magic, num_pairs, c, h, w, step, n_esc, stream_len = struct.unpack(
        _ECCV_HEADER_FMT, blob[:_ECCV_HEADER_SIZE]
    )
    if magic != ECCV_MAGIC:
        raise ValueError(f"bad entropy section magic: {magic!r}")
    grid = LatentGrid(step=float(step))

    pos = _ECCV_HEADER_SIZE
    escapes: dict[int, int] = {}
    for _ in range(n_esc):
        ep, ek = struct.unpack("<Ii", blob[pos : pos + 8])
        escapes[ep] = ek
        pos += 8
    stream = blob[pos : pos + stream_len]
    if len(stream) != stream_len:
        raise ValueError("entropy stream length mismatch")

    dec = RangeDecoder(stream) if stream_len > 0 else None
    evaluator = _ARPriorEvaluator(prior_net)

    elems_per_pair = c * h * w
    out_k = np.empty((num_pairs, c, h, w), dtype=np.int64)
    flat_pos = 0
    zero_prev = torch.zeros((1, c, h, w), dtype=torch.float32)
    prev_latent = zero_prev
    for t in range(num_pairs):
        mean_arr, sigma_arr = evaluator.params_for_pair(prev_latent)
        mean_flat = mean_arr.ravel()
        sigma_flat = np.maximum(sigma_arr.ravel(), 1e-8)

        # IDENTICAL window + freq tables as encode (deterministic agreement)
        k_center, half, k_grid = _pair_window(mean_flat, sigma_flat, grid.step)
        pmf = _gaussian_bin_pmf_batch(mean_flat, sigma_flat, k_grid, grid.step)
        freqs = _pmf_rows_to_freq_tables(pmf, total=RANGE_CODER_TOTAL - 1)  # (E, W)
        cum = np.concatenate(
            [np.zeros((freqs.shape[0], 1), np.int64), np.cumsum(freqs, axis=1)], axis=1
        )
        cum = np.concatenate([cum, (cum[:, -1:] + 1)], axis=1)
        totals = cum[:, -1]
        escape_idx = freqs.shape[1]

        k_flat = np.empty(elems_per_pair, dtype=np.int64)
        for j in range(elems_per_pair):
            total = int(totals[j])
            target = dec.target(total)  # type: ignore[union-attr]
            row = cum[j]
            # symbol = largest index with cum[idx] <= target
            sym = int(np.searchsorted(row, target, side="right") - 1)
            dec.update(low_count=int(row[sym]), high_count=int(row[sym + 1]), total=total)  # type: ignore[union-attr]
            if sym == escape_idx:
                k_flat[j] = escapes[flat_pos]
            else:
                k_flat[j] = int(k_grid[j, 0]) + sym
            flat_pos += 1
        out_k[t] = k_flat.reshape(c, h, w)
        prev_latent = grid.dequantize(torch.from_numpy(out_k[t])).unsqueeze(0)

    return grid.dequantize(out_k)


# --------------------------------------------------------------------------- #
# Differentiable-equivalent bit estimator (the Layer-1 ↔ Layer-2 unifier)      #
# --------------------------------------------------------------------------- #

def ar_gaussian_predicted_bits(
    latents: torch.Tensor,
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    grid: LatentGrid,
) -> torch.Tensor:
    """Predicted coded BITS for a latent tensor under its AR Gaussian + grid.

    This is the SAME quantity ``compute_ar_log_prob`` builds (continuous NLL),
    converted to the discrete bin-rate that THIS coder actually emits:

        bits = -log2 P(bin_k) ≈ -log2( pdf(z) * step )
             = (-ln p(z) - ln(step)) / ln 2     (per element, for fine grids)

    Equivalently, ``-log2 p_continuous + log2(1/step)``: the continuous AR NLL
    (the Layer-2 surrogate) plus a CONSTANT per-element offset ``-log2(step)``.
    So minimizing the Layer-2 continuous NLL == minimizing this coder's bytes, up
    to a constant — the train/deploy rate gap is closed by construction.

    Differentiable in (latents, mean, log_scale) so a Layer-2 loss can backprop
    through it. Returns a scalar tensor (total bits across all elements).
    """
    sigma = torch.exp(log_scale)
    log_two_pi = math.log(2.0 * math.pi)
    diff = (latents - mean) / sigma
    # continuous Gaussian log-density (nats)
    ln_p = -0.5 * diff * diff - log_scale - 0.5 * log_two_pi
    # discrete bin bits = -log2( p * step ) = (-ln_p - ln step) / ln2
    bits = (-ln_p - math.log(grid.step)) / math.log(2.0)
    return bits.clamp(min=0.0).sum()
