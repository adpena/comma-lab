"""Differentiable Shannon-entropy loss terms for delta-epsilon-zeta joint substrate training.

Per Grand Council 2026-05-07 verdict (`.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md`),
the strategic follow-on after PR103-on-PR106 standalone (-661 B) is:

> Joint substrate training: reduce the native description length of the
> HNeRV/sidechannel system instead of relying only on post-hoc entropy recodes.

Per Path B per-tensor Shannon analysis on PR106 substrate
(`experiments/results/lane_per_tensor_shannon_pr106_20260507T173846Z/per_tensor_shannon.json`):

- H0 floor (28 tensors): 167,570 B
- H2 floor (28 tensors): **88,983 B**
- brotli total: 170,096 B (sits at 1.015x H0; near-optimal for i.i.d. coding)
- AC total: 161,388 B (PR103 indices)

The delta-epsilon-zeta training target is **H2 -> H0 collapse**: train weights so that
second-order conditional entropy collapses to first-order (i.e., weights
are conditionally uniform given context). At the limit `H2 = H0`, AC and
brotli are equivalent and both achieve the i.i.d. floor. Going further
toward `H2 -> 0` would require structured weights - the **physics floor**
is the architecture's intrinsic complexity.

This module provides **torch-differentiable surrogates** for H0 and H2
that can be added to a training loss. The surrogates use soft histograms
(temperature-controlled softmax over int bins) so gradients flow back to
the float weights.

Strict-scorer-rule: pure CPU + numpy + torch (no MPS dependencies, no
scorer load). The surrogates are entropy estimators; they do NOT call
the pipeline's `encode()`, which is non-differentiable (integer codec
output bytes).

Cross-references:

- Council deliberation: `.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md`
- Numpy reference (non-differentiable): :mod:`tools.per_tensor_shannon_analysis`
- Pipeline-aware callback (non-differentiable empirical signal):
  :mod:`tac.codec_pipeline_deltaepszeta_callback`
- Substrate-mismatch reframe:
  ``feedback_op1_substrate_mismatch_codec_engineering_reframe_20260507.md``
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Soft-histogram building blocks
# ---------------------------------------------------------------------------

def _soft_assignment(
    weights: torch.Tensor,
    *,
    n_bits: int = 8,
    quant_range: float = 127.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute a soft assignment of float weights to integer bins.

    For each weight ``w``, returns a probability vector over ``2**n_bits``
    bins. Uses ``softmax(-(w_scaled - bin_center)^2 / temperature)`` so the
    assignment is differentiable in ``w``.

    Args:
        weights: tensor of arbitrary shape (float). Will be flattened.
        n_bits: number of bits per symbol; alphabet size = ``2**n_bits``.
        quant_range: the symmetric range over which weights are quantized
            (default 127.0, matching PR101's `N_QUANT`).
        temperature: softmax temperature. Lower = harder assignment, higher
            = softer (better gradient flow but biased entropy estimate).

    Returns:
        tensor of shape ``(N, alphabet_size)`` where ``N = weights.numel()``.
        Rows sum to 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")

    alphabet_size = 1 << n_bits
    flat = weights.reshape(-1)
    # Map weights into [0, alphabet_size-1] via the quantization scale.
    # We use the same scale per-tensor as PR101's int8 zigzag pipeline:
    # tensor.max() / quant_range -> scale, then w / scale in [-127, 127],
    # offset to [0, 254] for u8. Bin centers at integer u8 positions.
    scale = flat.detach().abs().max().clamp_min(1e-12) / quant_range
    scaled = flat / scale  # in [-127, 127] approximately
    offset_u8 = scaled + (alphabet_size / 2.0)  # in [1, 255] for n_bits=8

    bin_centers = torch.arange(
        alphabet_size, dtype=weights.dtype, device=weights.device
    )
    # Squared distance from each weight to each bin, shape (N, alphabet_size).
    diff = offset_u8.unsqueeze(1) - bin_centers.unsqueeze(0)
    logits = -(diff * diff) / (temperature * temperature)
    return torch.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# H0 proxy (zero-th-order / i.i.d. entropy)
# ---------------------------------------------------------------------------

def shannon_h0_loss(
    weights: torch.Tensor,
    *,
    n_bits: int = 8,
    quant_range: float = 127.0,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return a differentiable scalar = H0 in bits/symbol.

    For an empirical histogram p over the alphabet, ``H0 = -Sum p_i log2 p_i``.
    Numpy reference: :func:`tools.per_tensor_shannon_analysis.shannon_entropy_h0`.

    Maximum value = ``n_bits`` (uniform distribution); minimum = 0 (delta).
    Differentiable in ``weights`` via the soft-assignment scheme.

    Args:
        weights: float tensor of arbitrary shape. Will be flattened.
        n_bits: alphabet size (2**n_bits). Default 8 = u8 alphabet.
        quant_range: symmetric quant range; matches PR101's N_QUANT.
        temperature: soft-assignment temperature; lower = closer to true
            (non-differentiable) histogram.
        eps: numerical floor on probabilities for log stability.

    Returns:
        scalar torch tensor (in bits per symbol). Backprop-friendly.
    """
    soft = _soft_assignment(
        weights,
        n_bits=n_bits,
        quant_range=quant_range,
        temperature=temperature,
    )
    # Aggregate to a single histogram across the entire weight tensor.
    p = soft.mean(dim=0).clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    h0_nats = -(p * torch.log(p)).sum()
    return h0_nats / math.log(2.0)  # convert nats -> bits


# ---------------------------------------------------------------------------
# H2 proxy (conditional second-order entropy)
# ---------------------------------------------------------------------------

def shannon_h2_loss(
    weights: torch.Tensor,
    *,
    n_bits: int = 8,
    quant_range: float = 127.0,
    temperature: float = 1.0,
    eps: float = 1e-12,
    max_alphabet_for_trigram: int = 32,
) -> torch.Tensor:
    """Return a differentiable scalar = H2 in bits/symbol (lower bound).

    H2 is the conditional entropy ``H(X_t | X_{t-1}, X_{t-2})`` computed
    via the empirical trigram distribution:

        H2 = -Sum_{x_t, x_{t-1}, x_{t-2}} p(x_t, x_{t-1}, x_{t-2}) log2 p(x_t | x_{t-1}, x_{t-2})

    Numpy reference: :func:`tools.per_tensor_shannon_analysis.shannon_entropy_h2`.

    For tractability the trigram table is computed over a reduced alphabet
    (``max_alphabet_for_trigram``-bin re-quantization) to keep memory
    bounded. The reduced alphabet is a 2x or 4x downsampled version of
    the full ``2**n_bits`` alphabet; the H2 value is reported in bits per
    symbol of the FULL alphabet.

    The differentiable surrogate uses the same soft-assignment as H0 but
    aggregates conditional probabilities. Lower = more redundant
    (compressible by context-aware coder); higher = more random.

    Important: H2 <= H1 <= H0 always (conditioning never increases entropy).
    A delta-epsilon-zeta training objective minimizing H2 pushes toward conditionally
    uniform weights.
    """
    if max_alphabet_for_trigram > 64:
        raise ValueError(
            f"max_alphabet_for_trigram > 64 is intractable for trigram "
            f"tables on most weight tensors; got {max_alphabet_for_trigram}"
        )
    full_alphabet = 1 << n_bits
    if max_alphabet_for_trigram > full_alphabet:
        max_alphabet_for_trigram = full_alphabet

    # Soft-assign to the reduced alphabet directly.
    reduced_bits = max(1, int(math.log2(max_alphabet_for_trigram)))
    soft = _soft_assignment(
        weights,
        n_bits=reduced_bits,
        quant_range=quant_range,
        temperature=temperature,
    )  # shape (N, k)
    n = soft.shape[0]
    if n < 3:
        # Degenerate; return H0 on the same alphabet.
        return shannon_h0_loss(
            weights,
            n_bits=reduced_bits,
            quant_range=quant_range,
            temperature=temperature,
            eps=eps,
        )

    # Trigram distribution: p(x_t, x_{t-1}, x_{t-2}) approximated as the
    # tensor-product of consecutive soft assignments along the flattened
    # sequence. Shape (n-2, k, k, k).
    a = soft[2:]      # x_t
    b = soft[1:-1]    # x_{t-1}
    c = soft[:-2]     # x_{t-2}
    # Outer product over the last dim, summed over time.
    # tri[i, j, k_] = E_t[a_t,i * b_t,j * c_t,k_]
    tri = torch.einsum('ti,tj,tk->ijk', a, b, c) / a.shape[0]
    tri = tri.clamp_min(eps)
    # Marginal over context: p(x_{t-1}, x_{t-2})
    context_marginal = tri.sum(dim=0).clamp_min(eps)  # (k, k)
    # Joint p(x_t, x_{t-1}, x_{t-2}) -> conditional p(x_t | x_{t-1}, x_{t-2})
    cond = tri / context_marginal.unsqueeze(0)        # (k, k, k)
    cond = cond.clamp_min(eps)
    # H2 = -Sum p(x_t, x_{t-1}, x_{t-2}) log p(x_t | x_{t-1}, x_{t-2})
    h2_nats = -(tri * torch.log(cond)).sum()
    h2_bits_per_reduced_symbol = h2_nats / math.log(2.0)
    # Scale to bits per FULL-alphabet symbol: full_bits / reduced_bits ratio.
    # Since we compress alphabet by `full_alphabet/k`, each reduced symbol
    # carries `log2(k) / log2(full_alphabet)` of a full symbol's worth of
    # information; report H2 in full-alphabet-symbol bits via:
    return h2_bits_per_reduced_symbol * (n_bits / max(1, reduced_bits))


# ---------------------------------------------------------------------------
# Headroom / target ratio (per Path B finding: H2/H0 ~= 0.531 on PR106 -> 1.91x headroom)
# ---------------------------------------------------------------------------

def shannon_h2_h0_ratio(
    weights: torch.Tensor,
    *,
    n_bits: int = 8,
    temperature: float = 1.0,
    max_alphabet_for_trigram: int = 32,
) -> torch.Tensor:
    """Return the H2 / H0 ratio (compressibility headroom).

    Path B PR106 result: ratio ~= 0.531 (H2 = 88,983 B, H0 = 167,570 B
    aggregate over 28 tensors). Lower = more compressible by context-aware
    coding; the i.i.d.-uniform regime has ratio = 1.0.

    Use this as a delta-epsilon-zeta training-target signal: drive the ratio toward 1.0
    (uniform conditional) OR drive H2 itself toward zero (structured
    weights). The two regimes reflect different design choices.

    Returns:
        scalar torch tensor in [0, 1] approximately (above 1 indicates
        soft-histogram artifacts; clamp interpretation).
    """
    h0 = shannon_h0_loss(
        weights, n_bits=n_bits, temperature=temperature
    )
    h2 = shannon_h2_loss(
        weights,
        n_bits=n_bits,
        temperature=temperature,
        max_alphabet_for_trigram=max_alphabet_for_trigram,
    )
    return h2 / h0.clamp_min(1e-6)


__all__ = [
    "shannon_h0_loss",
    "shannon_h2_h0_ratio",
    "shannon_h2_loss",
]
