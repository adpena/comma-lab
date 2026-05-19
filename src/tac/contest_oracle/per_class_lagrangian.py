# SPDX-License-Identifier: MIT
"""Impl 5 -- 5-class imbalance-corrected per-class Lagrangian (NEW; genuinely missing).

The SegNet 5-class output is fixed by ``upstream/modules.py``
(``smp.Unet(.., classes=5)``); rare classes have proportionally HIGHER
per-pixel-improvement EV than common classes (a 1 percentage-point reduction
in argmax-disagreement on a 1% class is 100x more leveraged than the same
reduction on a 50% class).

The canonical per-class Lagrangian assigns inverse-class-frequency weights
to the score-aware seg loss so substrate trainers backprop with the
empirically-optimal per-class multipliers; this extincts the implicit-
uniform-weighting cargo-cult per Catalog #303.

Citations:
  - Lin, Goyal, Girshick, He, Dollar 2017 *Focal Loss for Dense Object
    Detection* (arxiv 1708.02002) -- class-imbalance correction theory.
  - Cui, Jia, Lin, Song, Belongie 2019 *Class-Balanced Loss Based on
    Effective Number of Samples* (CVPR) -- effective-number formulation.
  - ``upstream/modules.py`` -- SegNet 5-class architecture.
  - CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L6 --
    score-domain Lagrangian (not weight-domain proxies).

Catalog #125 hook 1 (sensitivity_map): ACTIVE -- per-class sensitivities
ARE per-class Lagrangian multipliers (canonical algebraic identity).
Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- autopilot
ranker consumes per-class multipliers for substrate score-aware loss
configuration recommendations.
Catalog #305 observability surface: decomposable_per_signal, cite_able,
counterfactual_able.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from .constants import SEGNET_NUM_CLASSES


class PerClassLagrangianError(ValueError):
    """Raised when class-frequency input is invalid (e.g. negative / wrong length)."""


# Effective-number canonical hyperparameter (Cui et al 2019 CVPR).
# beta = 0.9999 is the canonical value from the paper for ImageNet-scale
# imbalanced datasets. We expose it as configurable but default to canonical.
DEFAULT_EFFECTIVE_NUMBER_BETA: Final[float] = 0.9999
"""Cui et al 2019 canonical beta for effective-number class-imbalance correction."""


@dataclass(frozen=True, slots=True)
class PerClassLambdaSeg:
    """Per-class Lagrangian multipliers for the SegNet 5-class seg loss."""

    class_frequencies: tuple[float, ...]
    """Tuple of ``SEGNET_NUM_CLASSES`` floats summing to 1.0 (the empirical
    distribution of class assignments across the per-pair pixel cells)."""

    lambda_per_class_inverse_frequency: tuple[float, ...]
    """Canonical inverse-frequency weights ``lambda_c = 1 / max(freq_c, eps)``
    normalized to mean 1.0; the rare classes get proportionally higher
    multipliers."""

    lambda_per_class_effective_number: tuple[float, ...]
    """Cui et al 2019 effective-number weights
    ``lambda_c = (1 - beta) / (1 - beta^n_c)`` normalized to mean 1.0;
    more stable for very-rare classes (avoid the inverse-frequency
    boundary case of ``freq -> 0 -> lambda -> inf``)."""

    method_default: str = "effective_number"
    """Canonical default; inverse-frequency is the simpler closed-form sister."""


def compute_per_class_lambda_seg(
    *,
    class_frequencies: Sequence[float],
    beta: float = DEFAULT_EFFECTIVE_NUMBER_BETA,
    epsilon: float = 1.0e-6,
) -> PerClassLambdaSeg:
    """Closed-form per-class Lagrangian multipliers from class frequencies.

    Args:
        class_frequencies: Sequence of ``SEGNET_NUM_CLASSES`` floats.
            Each entry is the empirical fraction of pixel-cells assigned
            to that class. Should sum to ~1.0.
        beta: Cui et al 2019 effective-number hyperparameter; canonical 0.9999.
        epsilon: Floor on frequency to avoid divide-by-zero in inverse-
            frequency weights; canonical 1e-6.

    Returns:
        ``PerClassLambdaSeg`` carrying both inverse-frequency AND
        effective-number canonical multiplier tuples.

    Raises:
        PerClassLagrangianError: if input has wrong length or any negative
            entry.
    """
    if len(class_frequencies) != SEGNET_NUM_CLASSES:
        raise PerClassLagrangianError(
            f"class_frequencies must have length {SEGNET_NUM_CLASSES} "
            f"(got {len(class_frequencies)})"
        )
    for i, f in enumerate(class_frequencies):
        if f < 0:
            raise PerClassLagrangianError(
                f"class_frequencies[{i}]={f} must be >= 0"
            )
    total = sum(class_frequencies)
    if total <= 0:
        raise PerClassLagrangianError(
            f"class_frequencies must sum to > 0 (got total={total})"
        )

    # Normalize to a proper distribution
    freqs = tuple(float(f) / total for f in class_frequencies)

    # Inverse-frequency weights: lambda_c = 1 / max(freq_c, eps); normalize mean to 1.
    inv_freq = tuple(1.0 / max(f, epsilon) for f in freqs)
    mean_inv = sum(inv_freq) / len(inv_freq)
    lambda_inv_normalized = tuple(w / mean_inv for w in inv_freq)

    # Effective-number weights: lambda_c = (1-beta) / (1 - beta^n_c)
    # where n_c is the count proxy. Cui et al treat n_c as relative-frequency
    # weighted by a fixed total-sample reference; we use class-frequency directly
    # as n_c surrogate (the normalization-invariant lambda ratios are what
    # downstream consumers care about).
    if not (0.0 < beta < 1.0):
        raise PerClassLagrangianError(f"beta must be in (0, 1) (got {beta})")
    eff_num = tuple(
        (1.0 - beta) / (1.0 - beta ** max(f * 1.0e6, 1.0))  # scale n_c proxy
        for f in freqs
    )
    mean_eff = sum(eff_num) / len(eff_num)
    lambda_eff_normalized = tuple(w / mean_eff for w in eff_num)

    return PerClassLambdaSeg(
        class_frequencies=freqs,
        lambda_per_class_inverse_frequency=lambda_inv_normalized,
        lambda_per_class_effective_number=lambda_eff_normalized,
    )


def apply_per_class_lambda_to_seg_loss(
    *,
    per_class_argmax_disagreement: Sequence[float],
    lambda_per_class: Sequence[float],
) -> float:
    """Apply per-class Lagrangian multipliers to a per-class seg-loss vector.

    Returns the canonical scalar loss ``L = sum_c lambda_c * d_seg_c``;
    drop-in for the existing uniform-weighted ``L = sum_c d_seg_c``.

    Args:
        per_class_argmax_disagreement: 5-tuple of per-class argmax-disagreement
            rates from the SegNet scorer.
        lambda_per_class: 5-tuple of per-class Lagrangian multipliers from
            ``compute_per_class_lambda_seg``.

    Returns:
        Scalar weighted seg loss.

    Raises:
        PerClassLagrangianError: if length mismatch or negative inputs.
    """
    if len(per_class_argmax_disagreement) != SEGNET_NUM_CLASSES:
        raise PerClassLagrangianError(
            f"per_class_argmax_disagreement must have length {SEGNET_NUM_CLASSES} "
            f"(got {len(per_class_argmax_disagreement)})"
        )
    if len(lambda_per_class) != SEGNET_NUM_CLASSES:
        raise PerClassLagrangianError(
            f"lambda_per_class must have length {SEGNET_NUM_CLASSES} "
            f"(got {len(lambda_per_class)})"
        )
    for i, (d, lam) in enumerate(
        zip(per_class_argmax_disagreement, lambda_per_class)
    ):
        if d < 0:
            raise PerClassLagrangianError(
                f"per_class_argmax_disagreement[{i}]={d} must be >= 0"
            )
        if lam < 0:
            raise PerClassLagrangianError(
                f"lambda_per_class[{i}]={lam} must be >= 0"
            )
    return float(sum(d * lam for d, lam in zip(per_class_argmax_disagreement, lambda_per_class)))


__all__ = [
    "DEFAULT_EFFECTIVE_NUMBER_BETA",
    "PerClassLagrangianError",
    "PerClassLambdaSeg",
    "apply_per_class_lambda_to_seg_loss",
    "compute_per_class_lambda_seg",
]
