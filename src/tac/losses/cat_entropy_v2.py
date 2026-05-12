"""Size-weighted soft-histogram categorical-entropy regularizer (PR95 cat_entropy_v2).

Council deliberation 2 (OOO commit 328bf2f9, 8/10 FOR cat_entropy_v2 FIRST):
this is the Phase 1 highest-Shannon-MDL-EV port from PR95's 4 training primitives.
Phases 2-4 (8-stage curriculum, Muon optimizer, dual-RGB-head decoder) are
DEFERRED pending substrate decision.

Source port:
    ``experiments/results/public_pr_intake_full/public_pr98_intake_20260505_auto/
      source/submissions/hnerv_muon_finetuned_from_pr95/src/losses.py:79-113``
    (READ-ONLY public intake clone per Catalog #109)

What it computes
────────────────
For each Conv2d / Linear weight tensor ``w`` in the supplied decoder:

1. Per-tensor symmetric scale: ``s = w.abs().max() / 127`` (matches INT8
   symmetric-fake-quant in the same PR95 module). Tensors with ``s < 1e-12``
   are skipped (they are effectively zero and contribute no entropy signal).
2. Normalize: ``w_n = w / s`` (flattened to 1D). Optional random subsample
   of ``sample_size`` elements when the tensor is larger (PR95 default 2000).
3. Soft-assignment to integer bins ``{-127, ..., 127}`` via Gaussian kernel
   with bandwidth ``sigma`` (PR95 default 0.2):
       ``sa[i, b] = exp(-0.5 * ((w_n[i] - b) / sigma)^2)``
   then row-normalized to a per-sample categorical distribution.
4. Aggregate the per-bin mean ``bp = sa.mean(0)`` (marginal categorical
   distribution over the 255 bins).
5. Compute Shannon categorical entropy in bits:
       ``H = -sum(bp * log2(bp))``
6. Weight each tensor's entropy by its ``numel`` and return the weighted
   mean entropy across all eligible tensors.

Pushing this regularizer DOWN with a large lambda and a small sigma sharpens
the post-INT8 weight distribution at integer grid points, improving the
quantize-then-decompress roundtrip's bit-rate / distortion tradeoff for any
substrate that ships INT8-quantized Conv2d/Linear weights (HNeRV-family,
balle_renderer, sane_hnerv, post-PR95 hnerv_muon variants).

Substrate-independence (per Contrarian's coupling caveat in the OOO council)
────────────────────────────────────────────────────────────────────────────
The Phase 1 port is INTENTIONALLY architecture-blind: it walks ``named_modules``
and matches any ``nn.Conv2d`` / ``nn.Linear`` layer. It does NOT inspect:

- channel topology (FiLM / hyperprior / nonlinear-transform-coding etc.),
- layer order (skip connections, residual blocks),
- training stage (Stage 5+ curriculum gating — that is Phase 2).

This means it is safe to call on any decoder ``nn.Module`` and lambda-schedule
yourself. Phase 2 (8-stage curriculum) would wire the stage selector around
this primitive.

target_substrate_hint = "any_with_categorical_outputs"  — HNeRV-family yes,
but NOT coupled to layout. Trainers that want to use it MUST decide their own
``lambda`` schedule + module-filter and tag the run with the lane registry's
``score_aware_loss=true`` field.

This port preserves the math byte-faithfully — no algorithmic changes vs PR95.

Score-claim discipline
──────────────────────
- ``score_claim = false``
- ``promotion_eligible = false``
- ``ready_for_exact_eval_dispatch = false``

Per CLAUDE.md "FORBIDDEN forbidden_score_claim_with_byte_change_unless_inflate_consumes":
this primitive changes training dynamics, not archive bytes; no score is
asserted from its addition alone. Any downstream submission that includes a
``cat_entropy_v2``-trained decoder MUST go through the contest-CUDA + contest-CPU
dual-eval gate per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CatEntropyV2Config:
    """Configuration for :func:`cat_entropy_v2`.

    The defaults mirror PR95's training primitive exactly. Override only
    when ablating; document the reason in the trainer's lane-evidence row.
    """

    #: Soft-assignment bandwidth (smaller = sharper grid commitment).
    sigma: float = 0.2
    #: Per-tensor random subsample budget; tensors larger than this get
    #: ``sample_size`` random elements (deterministic w.r.t. the supplied
    #: ``torch.Generator`` when one is passed to :func:`cat_entropy_v2`).
    sample_size: int = 2000
    #: Bin range: ``{-127, ..., 127}`` (INT8-symmetric grid). PR95 hard-codes
    #: this; we expose it for ablation but the contract is "must match the
    #: trainer's eventual INT8 fake-quant grid."
    bin_min: int = -127
    bin_max: int = 127
    #: Numerical-stability epsilon added inside log2 and division.
    eps: float = 1e-12
    #: ``w.abs().max()`` floor — tensors below this are skipped (treated as
    #: zero-valued, no entropy signal).
    max_abs_floor: float = 1e-12


# Module classes that contribute entropy. PR95 uses (Conv2d, Linear);
# we keep that exactly.
_TARGET_LAYERS: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear)


def _iter_quantizable_weights(
    decoder: nn.Module,
) -> Iterable[tuple[str, nn.Module, torch.Tensor]]:
    """Yield ``(name, module, weight_tensor)`` for every quantizable layer."""
    for name, mod in decoder.named_modules():
        if isinstance(mod, _TARGET_LAYERS) and hasattr(mod, "weight"):
            w = mod.weight
            if w is None:
                continue
            yield name, mod, w


def cat_entropy_v2(
    decoder: nn.Module,
    config: CatEntropyV2Config | None = None,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Size-weighted soft-histogram categorical entropy of decoder weights.

    Returns a SCALAR ``torch.Tensor`` (with grad enabled when the input
    ``decoder`` weights have ``requires_grad=True``) in **bits per weight**,
    averaged across all eligible Conv2d/Linear tensors with tensor-numel
    weighting.

    The math matches PR95 exactly:

        ``H_decoder = sum_t numel(t) * H_t / sum_t numel(t)``

    where ``H_t`` is the Shannon categorical entropy of the soft-histogram of
    the symmetric-INT8-normalized weights of layer ``t``.

    Parameters
    ----------
    decoder
        Any ``nn.Module``. Walked via ``named_modules()`` and any Conv2d/Linear
        weight is included. Other layer types are silently ignored.
    config
        :class:`CatEntropyV2Config`. Defaults match PR95's call site.
    device
        Optional override for the bin-tensor device. When ``None`` (default),
        the device of the first parameter is used.
    generator
        Optional ``torch.Generator`` for reproducible subsampling.

    Returns
    -------
    ``torch.Tensor``
        Scalar entropy in bits/weight (PR95's units), differentiable w.r.t.
        the decoder weights when ``requires_grad`` is set.

    Notes
    -----
    - When the decoder has no Conv2d/Linear weights (or all tensors are below
      ``max_abs_floor``), the function returns ``0.0`` as a scalar on the
      resolved device. This matches PR95's ``weighted_entropy / max(total_numel, 1)``.
    - The subsample uses ``torch.randperm`` on the tensor's own device when
      ``generator`` is ``None``; deterministic ablations should pass an
      explicit ``torch.Generator``.
    """
    cfg = config or CatEntropyV2Config()

    # Resolve device.
    if device is None:
        try:
            device_ = next(decoder.parameters()).device
        except StopIteration:
            device_ = torch.device("cpu")
    else:
        device_ = torch.device(device)

    bins = torch.arange(
        cfg.bin_min, cfg.bin_max + 1, device=device_, dtype=torch.float32
    )
    # 255 bins for {-127, ..., 127}.
    weighted_entropy = torch.zeros((), device=device_, dtype=torch.float32)
    total_numel = 0

    n_levels = float(max(cfg.bin_max, -cfg.bin_min))  # 127.0 in PR95 defaults

    for _, _, w in _iter_quantizable_weights(decoder):
        numel = w.numel()
        # PR95 uses .detach() to break the gradient through ma. We follow.
        ma = w.abs().max().detach()
        if ma.item() < cfg.max_abs_floor:
            continue
        # Normalize to the INT8 grid scale.
        wn = (w / (ma / n_levels)).flatten()
        if wn.numel() > cfg.sample_size:
            if generator is not None:
                idx = torch.randperm(
                    wn.numel(), device=wn.device, generator=generator
                )[: cfg.sample_size]
            else:
                idx = torch.randperm(wn.numel(), device=wn.device)[: cfg.sample_size]
            wn = wn[idx]
        # Soft-assignment kernel (broadcast over bins).
        sa = torch.exp(
            -0.5 * ((wn.unsqueeze(1) - bins.unsqueeze(0)) / cfg.sigma).pow(2)
        )
        sa = sa / (sa.sum(dim=1, keepdim=True) + cfg.eps)
        # Marginal bin probability.
        bp = sa.mean(dim=0)
        bp = bp / (bp.sum() + cfg.eps)
        # Shannon entropy in bits.
        entropy = -(bp * torch.log2(bp + cfg.eps)).sum()
        weighted_entropy = weighted_entropy + numel * entropy
        total_numel += numel

    return weighted_entropy / max(total_numel, 1)


# Target-substrate-hint metadata for lane-registry evidence rows + downstream
# bit-allocator routing per CLAUDE.md Catalog #124 representation lane gate.
TARGET_SUBSTRATE_HINT = "any_with_categorical_outputs"
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False
SOURCE_PORT_REFERENCE = (
    "experiments/results/public_pr_intake_full/public_pr98_intake_20260505_auto/"
    "source/submissions/hnerv_muon_finetuned_from_pr95/src/losses.py:79-113"
)
COUNCIL_DECISION_REFERENCE = "OOO commit 328bf2f9 deliberation 2 (8/10 FOR Phase 1)"

__all__ = [
    "CatEntropyV2Config",
    "cat_entropy_v2",
    "TARGET_SUBSTRATE_HINT",
    "SCORE_CLAIM",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SOURCE_PORT_REFERENCE",
    "COUNCIL_DECISION_REFERENCE",
]
