# SPDX-License-Identifier: MIT
"""Lane Ω water-fill bit allocator (per-weight bit-depth from Hessian importance).

Given a per-weight Fisher importance tensor I(w_ij) and a total bit budget B,
allocate per-weight bit-depths bits[w_ij] ∈ [min_bits, max_bits] so that:

    sum(bits) ≤ B
    bits[w] ≈ round(c * I(w)^alpha)  clamped to [min_bits, max_bits]

The constant `c` is found by binary search so the budget is satisfied while
keeping the allocation monotonically nondecreasing in importance.

This is the cousin of HAWQ (Dong 2019) and OBQ (Frantar 2022) — same
"importance → bit-budget" idea but applied per-weight (not per-layer or
per-block) and driven by the contest's hard-pair Fisher signal (memory:
project_lane_omega_bit_budget_hessian_aware_quantization +
project_lane_w_hard_pair_self_compress_premise_20260427).

CLAUDE.md compliance:
    * Pure CPU function (no GPU dependence). Deterministic for a given input.
    * No CLI here — call sites in profile_hessian_per_weight.py /
      remote_lane_omega_hessian_qat.sh consume this directly.
    * No global state.
    * Tested to: budget conservation, monotonic in importance, deterministic.

WAVE-3-FORENSIC-FIX-2 2026-05-20 history note (per Catalog #290 canonical-vs-unique
decision):
    * Originally landed at ``src/tac/bit_allocator.py`` (Lane Ω water-fill,
      sister wave 2026-04-27).
    * 2026-05-19 WAVE-2-PREREQ-BIT-ALLOCATOR sister wave created the package
      ``src/tac/bit_allocator/`` (per_byte / per_class / per_axis / pareto_dual)
      which structurally shadowed this module per Python package precedence.
    * 2026-05-20 WAVE-3-FORENSIC-FIX-2 renamed this module to
      ``src/tac/bit_allocator/lane_omega.py`` and re-exported ``allocate_bits``
      / ``allocation_report`` / ``DEFAULT_ALPHA`` / ``DEFAULT_MIN_BITS`` /
      ``DEFAULT_MAX_BITS`` from the package's ``__init__.py`` so existing
      callers (``from tac.bit_allocator import allocate_bits``) continue to
      work without modification. The canonical-vs-unique decision is Option B
      (preserve Lane Ω historical anchor as a distinct submodule; per-weight
      Fisher water-fill is mathematically distinct from per-byte / per-class
      / per-axis / Pareto-dual allocators in the sister package).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

__all__ = [
    "allocate_bits",
    "allocation_report",
    "DEFAULT_MIN_BITS",
    "DEFAULT_MAX_BITS",
    "DEFAULT_ALPHA",
]

# Tunables. The defaults are conservative (alpha=0.5 = sqrt-importance scaling
# matches HAWQ Section 3; min_bits=1 because a 0-bit weight is effectively
# pruned which we do NOT want to do silently — the export format DOES support
# storing a zero-bit weight as a single channel-mean scalar, but that's a
# different feature). Operators override per profile.
DEFAULT_ALPHA: float = 0.5
DEFAULT_MIN_BITS: int = 1
DEFAULT_MAX_BITS: int = 8


def _flatten_importance(
    importance: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int]]]:
    """Flatten layer-wise importance tensors into a single 1D tensor.

    Returns:
        flat: 1D tensor of all importances concatenated in deterministic order.
        meta: list of (layer_name, original_shape, numel) per layer in the
              same order, so we can scatter the 1D bits back into per-layer
              tensors.
    """
    flat_chunks: list[torch.Tensor] = []
    meta: list[tuple[str, torch.Size, int]] = []
    for name in sorted(importance.keys()):  # deterministic order
        t = importance[name]
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"importance[{name!r}] must be a torch.Tensor, got {type(t)}"
            )
        if not torch.isfinite(t).all():
            raise ValueError(
                f"importance[{name!r}] contains non-finite values "
                f"(min={t.min().item()}, max={t.max().item()})"
            )
        if (t < 0).any():
            raise ValueError(
                f"importance[{name!r}] contains negative values — Fisher "
                f"importance is |∂L/∂w|² and must be ≥ 0"
            )
        flat_chunks.append(t.detach().reshape(-1).to(torch.float64).cpu())
        meta.append((name, t.shape, t.numel()))
    if not flat_chunks:
        raise ValueError("importance dict is empty — nothing to allocate")
    return torch.cat(flat_chunks, dim=0), meta


def _bits_for_c(
    flat_imp: torch.Tensor,
    c: float,
    alpha: float,
    min_bits: int,
    max_bits: int,
) -> torch.Tensor:
    """Compute per-weight bits as round(c * I^alpha) clamped to [min_bits, max_bits]."""
    raw = c * flat_imp.pow(alpha)
    rounded = raw.round().clamp(min_bits, max_bits).to(torch.int64)
    return rounded


def allocate_bits(
    importance: Mapping[str, torch.Tensor],
    total_bits: int,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_bits: int = DEFAULT_MIN_BITS,
    max_bits: int = DEFAULT_MAX_BITS,
    bisect_iters: int = 64,
) -> "OrderedDict[str, torch.Tensor]":
    """Water-fill per-weight bit allocation under a total bit budget.

    Args:
        importance: layer_name → tensor of per-element Fisher importance.
            Tensor shape must match the corresponding state_dict tensor.
            Values must be non-negative and finite.
        total_bits: total bit budget across all weights (sum of bits ≤ this).
        alpha: scaling exponent. alpha=0.5 → sqrt-importance (HAWQ default).
            alpha=1.0 → linear-importance (more aggressive on the tail).
        min_bits: floor per weight (default 1).
        max_bits: ceiling per weight (default 8).
        bisect_iters: number of binary search iterations for c (default 64,
            gives ~1e-19 relative precision in c — way more than needed).

    Returns:
        OrderedDict mapping layer_name → uint8 tensor of bit-depths, same
        shape as the input importance tensor for that layer.

    Raises:
        ValueError if total_bits < min_bits * num_weights (budget too tight).
        TypeError if importance contains non-tensor values.
    """
    if min_bits < 0 or max_bits < min_bits:
        raise ValueError(
            f"invalid bit range [min_bits={min_bits}, max_bits={max_bits}]"
        )
    if total_bits <= 0:
        raise ValueError(f"total_bits must be positive, got {total_bits}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0 for monotonic mapping, got {alpha}")

    flat_imp, meta = _flatten_importance(importance)
    n_weights = flat_imp.numel()

    # Sanity check: even at min_bits, can we fit the budget?
    floor_total = n_weights * min_bits
    if floor_total > total_bits:
        raise ValueError(
            f"infeasible budget: {n_weights} weights × min_bits={min_bits} = "
            f"{floor_total} bits > total_bits={total_bits}. "
            f"Either lower min_bits, raise total_bits, or shrink the model."
        )

    # If even max-bits-everywhere fits, no allocation needed — give everyone max.
    ceiling_total = n_weights * max_bits
    if ceiling_total <= total_bits:
        flat_bits = torch.full((n_weights,), max_bits, dtype=torch.int64)
        return _scatter_back(flat_bits, meta)

    # Binary search c so the resulting bit-sum lands ≤ total_bits and as close
    # as possible from below. The bits-as-function-of-c curve is monotonic
    # (raw = c * I^alpha; round + clamp preserves monotonicity in c).
    # Bracket: c_lo = 0 (everyone clamps to min_bits) → bit_sum = floor_total.
    #          c_hi: choose so even the smallest nonzero importance gets
    #                clamped to max_bits.
    # If all importances are 0, fall back to flat min_bits.
    max_imp = float(flat_imp.max().item())
    if max_imp == 0.0:
        flat_bits = torch.full((n_weights,), min_bits, dtype=torch.int64)
        return _scatter_back(flat_bits, meta)

    c_lo = 0.0
    # Choose c_hi so c_hi * max_imp^alpha = max_bits + 1 (everyone saturates)
    c_hi = float(max_bits + 1) / max(max_imp ** alpha, 1e-30)

    # Round 26 / Codex finding 3 fix — bracket growth.
    #
    # The previous bracket had a subtle blind spot: if the highest-importance
    # weight saturates at max_bits well before c_hi is reached, the search
    # interval can collapse to a region where the *low-importance* weights
    # never escape min_bits. Concrete counterexample (importance=[100, 1],
    # alpha=0.5, total_bits=12, max_bits=8): c_hi was ~0.9, the high-imp
    # weight already clamped to 8, the low-imp weight stayed at 1, total=9
    # — leaving 3 bits unspent forever.
    #
    # Fix: if c_hi is feasible (sum ≤ total_bits) — meaning even the upper
    # bracket leaves slack — grow c_hi exponentially until either (a) it
    # exceeds budget so we have a real upper bound for bisection, or (b)
    # we hit a hard cap on growth iterations (safety guard for the
    # all-saturated case where every weight is already at max_bits and
    # growing c further can never spend more bits).
    best_bits: torch.Tensor | None = None
    SAFETY_GROWTH_ITERS = 48
    growth = 0
    prev_upper_sum: int | None = None
    while True:
        upper_candidate = _bits_for_c(flat_imp, c_hi, alpha, min_bits, max_bits)
        upper_sum = int(upper_candidate.sum().item())
        if upper_sum > total_bits:
            break  # bracket is now valid for bisection
        if upper_sum == ceiling_total:
            # Every weight already at max_bits — growing c more cannot
            # increase the sum. Accept this saturated allocation directly.
            best_bits = upper_candidate
            c_lo = c_hi  # collapse bracket so the bisection loop is a no-op
            break
        # Slack remains AND not all saturated → grow c_hi.
        # The current c_hi is feasible (sum ≤ total_bits) so it's a valid
        # candidate; record it as the running best before stretching.
        best_bits = upper_candidate
        # If growth doesn't change the sum (e.g. _bits_for_c is degenerate
        # like a monkeypatched constant), stop — further doubling is wasted.
        if prev_upper_sum is not None and upper_sum == prev_upper_sum:
            break
        prev_upper_sum = upper_sum
        c_lo = c_hi
        c_hi *= 2.0
        growth += 1
        if growth >= SAFETY_GROWTH_ITERS:
            # Safety cap. Should be unreachable in practice (each doubling
            # roughly halves the slack), but guarantees termination.
            break

    for _ in range(bisect_iters):
        c_mid = 0.5 * (c_lo + c_hi)
        candidate = _bits_for_c(flat_imp, c_mid, alpha, min_bits, max_bits)
        bit_sum = int(candidate.sum().item())
        if bit_sum <= total_bits:
            # feasible — try to spend more
            best_bits = candidate
            c_lo = c_mid
        else:
            c_hi = c_mid
        # Early exit if bracket is tight enough.
        if c_hi - c_lo < 1e-15 * max(c_hi, 1e-30):
            break

    if best_bits is None:
        # All midpoints exceeded budget → fall back to floor allocation.
        # This only triggers if total_bits == floor_total exactly with rounding.
        best_bits = torch.full((n_weights,), min_bits, dtype=torch.int64)

    # Final guarantee
    final_sum = int(best_bits.sum().item())
    assert final_sum <= total_bits, (
        f"water-fill produced sum={final_sum} > budget={total_bits}; "
        f"bisection bug"
    )
    return _scatter_back(best_bits, meta)


def _scatter_back(
    flat_bits: torch.Tensor,
    meta: list[tuple[str, torch.Size, int]],
) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    offset = 0
    for name, shape, n in meta:
        chunk = flat_bits[offset : offset + n].to(torch.uint8).reshape(shape)
        out[name] = chunk.contiguous()
        offset += n
    assert offset == flat_bits.numel()
    return out


def allocation_report(
    bits_per_weight: Mapping[str, torch.Tensor],
    importance: Mapping[str, torch.Tensor] | None = None,
) -> dict:
    """Compute summary statistics for a bit allocation.

    Used in provenance JSON so a fresh agent can see at a glance whether the
    allocation is sensible (e.g. mean ≈ 2 bits is the regime we predicted;
    mean ≈ 7 bits means the budget is too generous to discriminate).

    Args:
        bits_per_weight: layer_name → uint8 tensor of bit-depths.
        importance: optional matching importance tensor for per-layer
            importance-weighted bit averages.

    Returns:
        dict with: total_weights, total_bits, mean_bits, min_bits, max_bits,
        bits_histogram (list of 9 ints for bits 0..8), per_layer (dict of
        {layer: {n, mean_bits, total_bits}}).
    """
    total_weights = 0
    total_bits = 0
    all_bits_chunks: list[torch.Tensor] = []
    per_layer: dict[str, dict] = {}
    for name in sorted(bits_per_weight.keys()):
        b = bits_per_weight[name].reshape(-1).to(torch.int64)
        n = int(b.numel())
        s = int(b.sum().item())
        per_layer[name] = {
            "n": n,
            "mean_bits": float(b.float().mean().item()),
            "total_bits": s,
            "min_bits": int(b.min().item()),
            "max_bits": int(b.max().item()),
        }
        total_weights += n
        total_bits += s
        all_bits_chunks.append(b)
    all_bits = torch.cat(all_bits_chunks, dim=0) if all_bits_chunks else torch.empty(0, dtype=torch.int64)
    histogram = [int((all_bits == v).sum().item()) for v in range(0, 9)]
    return {
        "total_weights": total_weights,
        "total_bits": total_bits,
        "mean_bits": float(all_bits.float().mean().item()) if total_weights else 0.0,
        "median_bits": float(all_bits.float().median().item()) if total_weights else 0.0,
        "min_bits": int(all_bits.min().item()) if total_weights else 0,
        "max_bits": int(all_bits.max().item()) if total_weights else 0,
        "bits_histogram_0_8": histogram,
        "per_layer": per_layer,
    }
