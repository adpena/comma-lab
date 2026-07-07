# SPDX-License-Identifier: MIT
"""MLX port of the Ballé-style rate-in-the-loss WEIGHT-ENTROPY penalty for the LEVELSET witness
(``--weight-entropy-penalty-lambda`` — the fold path named by council draft
``DRAFT_derived_optimal_next_run_for_council_20260707.md`` §22(2): the −19.6% archive-bytes lever
existed on the TORCH VEHICLE ONLY, so a DSL ``Lever`` could not legally hold it; THIS module is
the trainer-side term that makes the flag real on the capstone).

WHY THIS IS THE RATE LEVER (same math as ``tac.torch_vehicle.weight_entropy_penalty``)
--------------------------------------------------------------------------------------
The contest rate term scores ONLY ``archive.zip`` bytes; the witness's counted payload is the
int8+brotli blob of its LEARNED params (``lever_b_levelset_generator.quantize_levelset_blob``:
per-tensor symmetric grid ``q = round(w / s * 127)`` with ``s = max|w| + 1e-8``; the curvelet
bank ``B``/``*_B`` is a FREE deterministic table, rule 118). brotli sits near the order-0
entropy floor, so the byte floor is ``Σ_t H(symbols_t)·numel_t/8`` — set by TRAINING. Adding
``λ·rate_term`` (the expected symbol codelength mapped onto the contest rate scale) pulls the
weight-symbol distribution toward low entropy → a lower deployed byte floor.

WHAT THE MLX SURROGATE IS (deterministic, state-free — deliberately NOT the torch learned prior)
------------------------------------------------------------------------------------------------
The torch vehicle's term is a LEARNED per-channel Ballé prior with ``U(-0.5,0.5)`` STE noise —
both of which carry state (learnable prior params in the optimizer/checkpoint) and RNG. The
levelset trainer's deterministic-reproducibility spine forbids new RNG and state the resume path
does not carry, so this port uses the DETERMINISTIC soft-histogram marginal-entropy surrogate
(the ``cat_entropy_v2`` / ``tac.losses.rate_surrogate`` kernel — Gaussian soft-bin assignment on
the int8 grid, σ=0.2, bins {-127..127}): per counted tensor,

    grid_i = w_i · 127 / stop_grad(max|w| + 1e-8)          (the EXACT codec grid)
    p_b    = mean_i softbin(grid_i, b)                      (row-normalized Gaussian kernel)
    H_t    = −Σ_b p_b log2(p_b + ε)                         (bits/weight, differentiable)
    total_bits = Σ_t H_t · numel_t ;  rate_term = total_bits/8/37_545_489 · 25

No RNG, no learnable prior, no buffers → the term is a pure function of the CURRENT weights
(resume-safe by construction; nothing new to checkpoint). The gradient flows to EVERY counted
element through its soft assignment (the scale is ``stop_gradient``-ed — only the SHAPE of the
symbol distribution is penalized, exactly as the torch lever detaches its scale).

SCORE-CLAIM DISCIPLINE (borrowed-number firewall — NO-FAKE #8)
--------------------------------------------------------------
The torch vehicle's MEASURED −19.6% (live-decoder bytes, λ50, 2026-06-20; EMA-lag caveat; ema0.9
translation proof) is attributed to the TORCH vehicle and its learned-prior term. THIS MLX lever
is NEVER-FIRED until its own n600 A/B lands: NO byte or score number transfers. The NO-FAKE
headline metric a λ>0 run must lower is :func:`measured_symbol_entropy_bits_numpy` — the HARD
(codec-exact) symbol entropy, not the surrogate.

λ=0 (the default) is a TRUE no-op: the trainer's guard skips the branch entirely (the term is
never constructed — no graph/memory change), mirroring the ``code_nuc_w`` lever pattern.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# The contest rate-term constants (S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/N). Used ONLY to
# map bits onto the score's rate scale so λ is comparable to the d_seg/d_pose terms; NOT a score
# claim (real bytes come from the byte-closed quantize_levelset_blob / archive.zip stat).
_TOTAL_VIDEO_BYTES = 37_545_489
_RATE_COEFF = 25.0
# The levelset codec grid (lever_b_levelset_generator._int8_symmetric): s = max|w| + 1e-8,
# q = clip(round(w / s * 127), -127, 127).
_N_QUANT = 127
_SCALE_EPS = 1e-8
# Soft-bin kernel bandwidth — matches cat_entropy_v2 / RateSurrogateConfig.sigma (the torch
# deterministic-surrogate contract) so the two vehicles' memoryless surrogates are the same math.
DEFAULT_SIGMA = 0.2
_LOG_EPS = 1e-12


def is_counted_param(name: str) -> bool:
    """True iff the named witness param is COUNTED in archive.zip (the learned payload).

    Mirrors ``quantize_levelset_blob`` / ``save_levelset_npz`` membership EXACTLY: everything is
    counted EXCEPT the free deterministic curvelet bank (``B`` / ``*_B`` — rule 118, regenerated
    at decode from cfg scalars). Penalizing a free table would be a fake rate lever (it ships no
    bytes); the predicate is shared so the penalty's tensor set provably matches the codec's.
    """
    return not (name == "B" or name.endswith("_B"))


def counted_param_items(params_flat: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Filter a ``tree_flatten``-style ``[(name, array), ...]`` list to the COUNTED set,
    deterministically sorted by name (stable across runs/hosts)."""
    return sorted(
        ((n, a) for n, a in params_flat if is_counted_param(n)),
        key=lambda kv: kv[0],
    )


# ---------------------------------------------------------------------------
# MLX differentiable surrogate (the trainer's loss term)
# ---------------------------------------------------------------------------
def soft_symbol_entropy_bits_mlx(w, *, sigma: float = DEFAULT_SIGMA):
    """Differentiable soft-histogram symbol entropy (bits/weight) of ONE tensor on the codec
    grid. Returns a scalar mx array; gradient flows to every element of ``w`` (the per-tensor
    scale is ``stop_gradient``-ed — only the distribution SHAPE is penalized). A degenerate
    all-~zero tensor maps to the all-zero symbol stream (entropy → ~0), matching the codec."""
    import mlx.core as mx

    ma = mx.stop_gradient(mx.max(mx.abs(w))) + _SCALE_EPS
    grid = mx.reshape(w * (float(_N_QUANT) / ma), (-1, 1))            # (numel, 1)
    bins = mx.arange(-_N_QUANT, _N_QUANT + 1, dtype=grid.dtype)[None]  # (1, 255)
    # Row-normalized Gaussian soft assignment (identical kernel to cat_entropy_v2 /
    # rate_surrogate._soft_bin_assignment), then the marginal soft histogram.
    sa = mx.exp(-0.5 * mx.square((grid - bins) / float(sigma)))
    sa = sa / (mx.sum(sa, axis=1, keepdims=True) + _LOG_EPS)
    p = mx.mean(sa, axis=0)                                           # (255,) soft histogram
    return -mx.sum(p * mx.log2(p + _LOG_EPS))                         # bits/weight (scalar)


def weight_entropy_rate_term_mlx(model, *, sigma: float = DEFAULT_SIGMA):
    """``(total_bits, rate_term)`` of the witness model's COUNTED weights under the deterministic
    soft-histogram surrogate — the MLX twin of the torch ``WeightEntropyPenalty.rate_bits``.

    ``total_bits = Σ_t H_t·numel_t`` (differentiable; gradient to every counted param);
    ``rate_term = total_bits/8/37_545_489·25`` (the contest rate scale, so λ is chosen in score
    units). Iterates ``tree_flatten(model.parameters())`` filtered by :func:`is_counted_param`
    in sorted-name order (deterministic). Raises if the model exposes NO counted params
    (refuse-not-silently-skip, matching the torch construction guard)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    counted = counted_param_items(tree_flatten(model.parameters()))
    if not counted:
        raise ValueError(
            "weight_entropy_rate_term_mlx: the model exposes NO counted params to penalize "
            "(every param matched the free-bank exclusion). The penalty would be a silent no-op."
        )
    total_bits = mx.zeros(())
    for _name, arr in counted:
        h_t = soft_symbol_entropy_bits_mlx(arr, sigma=sigma)
        total_bits = total_bits + h_t * float(arr.size)
    rate_term = total_bits / 8.0 / float(_TOTAL_VIDEO_BYTES) * _RATE_COEFF
    return total_bits, rate_term


# ---------------------------------------------------------------------------
# numpy reference twins (parity + the NO-FAKE hard metric)
# ---------------------------------------------------------------------------
def soft_symbol_entropy_bits_numpy(w: np.ndarray, *, sigma: float = DEFAULT_SIGMA) -> float:
    """numpy fp reference of :func:`soft_symbol_entropy_bits_mlx` (same math, one backend
    contract — the parity surface the unit test pins)."""
    a = np.asarray(w, np.float32).reshape(-1, 1).astype(np.float64)
    ma = float(np.abs(a).max()) + _SCALE_EPS
    grid = a * (float(_N_QUANT) / ma)
    bins = np.arange(-_N_QUANT, _N_QUANT + 1, dtype=np.float64)[None]
    sa = np.exp(-0.5 * np.square((grid - bins) / float(sigma)))
    sa = sa / (sa.sum(axis=1, keepdims=True) + _LOG_EPS)
    p = sa.mean(axis=0)
    return float(-(p * np.log2(p + _LOG_EPS)).sum())


def measured_symbol_entropy_bits_numpy(params: dict[str, np.ndarray]) -> float:
    """The REAL (codec-exact, HARD-quantized) mean weight symbol-entropy in BITS/WEIGHT,
    size-weighted over the COUNTED tensors — the NO-FAKE headline metric a λ>0 run must LOWER
    (the MLX twin of the torch ``measure_decoder_weight_symbol_entropy``).

    Quantizes EXACTLY as ``lever_b_levelset_generator._int8_symmetric`` does
    (``q = clip(round(w/(max|w|+1e-8)*127), -127, 127)``) and computes the exact histogram
    Shannon entropy — no soft assignment, no surrogate. A lever that lowers the soft surrogate
    but NOT this measured entropy would be a fake."""
    total_numel = 0
    weighted = 0.0
    for name, arr in sorted(params.items()):
        if not is_counted_param(name):
            continue
        a = np.asarray(arr, np.float32)
        if a.size == 0:
            continue
        s = float(np.abs(a).max()) + _SCALE_EPS
        q = np.clip(np.round(a / s * _N_QUANT), -_N_QUANT, _N_QUANT).astype(np.int64).ravel()
        counts = np.bincount(q + _N_QUANT, minlength=2 * _N_QUANT + 1).astype(np.float64)
        p = counts / counts.sum()
        nz = p[p > 0]
        weighted += a.size * float(-(nz * np.log2(nz)).sum())
        total_numel += a.size
    return weighted / max(total_numel, 1)


__all__ = [
    "DEFAULT_SIGMA",
    "counted_param_items",
    "is_counted_param",
    "measured_symbol_entropy_bits_numpy",
    "soft_symbol_entropy_bits_mlx",
    "soft_symbol_entropy_bits_numpy",
    "weight_entropy_rate_term_mlx",
]
