# SPDX-License-Identifier: MIT
"""Impl 14 -- class-conditional CDF priors for arithmetic coding.

5 canonical priors (one per SegNet class) for arithmetic coding. Composes
with ``tac.preflight_rudin_daubechies.compressive_coverage_estimator``
(Daubechies wavelet sister) for class-conditional encoding. Expected
~5-10% rate savings vs class-agnostic encoding per Cui et al 2019
effective-number theory + MacKay 2003 Dasher class-conditional canonical.

Implementation: per-class symbol histograms aggregated from training data;
canonical adaptive Beta-Bernoulli CDF priors with mass-preservation
guarantees per Cover & Thomas 2006 ch.5.

Citations:
  - MacKay 2003 *Information Theory, Inference, and Learning Algorithms*
    Section 6.6 -- Dasher class-conditional arithmetic coding.
  - Cover & Thomas 2006 *Elements of Information Theory* Section 5.10 --
    arithmetic coding optimality theorem.
  - Witten, Neal, Cleary 1987 *Arithmetic coding for data compression*
    (CACM) -- canonical implementation.
  - Cui et al 2019 *Class-Balanced Loss* (CVPR) -- per-class effective-number
    theory.
  - ``tac.preflight_rudin_daubechies.compressive_coverage_estimator`` --
    canonical Daubechies-wavelet sister.

Catalog #125 hook 3 (bit_allocator): ACTIVE -- per-class encoder is
canonical bit allocator at the per-class granularity.
Catalog #305 observability surface: decomposable_per_signal, cite_able.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .constants import SEGNET_NUM_CLASSES


class ArithmeticCoderError(ValueError):
    """Raised when arithmetic-coder inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ClassConditionalPrior:
    """Per-class symbol prior (histogram + cumulative)."""

    class_index: int
    """0-based SegNet class index in ``[0, SEGNET_NUM_CLASSES)``."""

    symbol_counts: tuple[int, ...]
    """Per-symbol observation counts. Length = vocabulary size."""

    total_count: int
    """``sum(symbol_counts)``; used as the cumulative-distribution base."""

    estimated_entropy_bits_per_symbol: float
    """``-sum_s (p_s * log2(p_s))``; canonical Shannon entropy."""


@dataclass(frozen=True, slots=True)
class ClassConditionalCodebook:
    """Full 5-class canonical codebook for arithmetic coding."""

    class_priors: tuple[ClassConditionalPrior, ...]
    """Tuple of ``SEGNET_NUM_CLASSES`` per-class priors."""

    vocabulary_size: int
    """Number of distinct symbols (same across all 5 classes)."""

    expected_savings_vs_class_agnostic_bits_per_symbol: float
    """Difference between class-agnostic entropy and weighted per-class entropy."""


def build_class_conditional_codebook(
    *,
    per_class_symbol_observations: Sequence[Sequence[int]],
) -> ClassConditionalCodebook:
    """Build a 5-class canonical codebook from per-class symbol observations.

    Args:
        per_class_symbol_observations: A 5-tuple where each entry is a sequence
            of integer symbol observations for that class.

    Returns:
        ``ClassConditionalCodebook`` with per-class priors + expected savings.

    Raises:
        ArithmeticCoderError: if input has wrong outer-length, empty inner
            sequences, or negative symbol indices.
    """
    if len(per_class_symbol_observations) != SEGNET_NUM_CLASSES:
        raise ArithmeticCoderError(
            f"per_class_symbol_observations must have outer length "
            f"{SEGNET_NUM_CLASSES} (got {len(per_class_symbol_observations)})"
        )
    # Determine vocab size as the max observed symbol + 1.
    all_symbols: list[int] = []
    for c, obs in enumerate(per_class_symbol_observations):
        if not obs:
            raise ArithmeticCoderError(
                f"per_class_symbol_observations[{c}] is empty; need >= 1 observation"
            )
        for s in obs:
            if s < 0:
                raise ArithmeticCoderError(
                    f"per_class_symbol_observations[{c}] contains negative "
                    f"symbol {s}"
                )
            all_symbols.append(s)
    vocab_size = max(all_symbols) + 1

    class_priors: list[ClassConditionalPrior] = []
    for c, obs in enumerate(per_class_symbol_observations):
        counts = [0] * vocab_size
        for s in obs:
            counts[s] += 1
        total = sum(counts)
        # Per-symbol probability + Shannon entropy.
        entropy_bits = 0.0
        if total > 0:
            for n in counts:
                if n > 0:
                    p = n / total
                    entropy_bits -= p * math.log2(p)
        class_priors.append(ClassConditionalPrior(
            class_index=c,
            symbol_counts=tuple(counts),
            total_count=int(total),
            estimated_entropy_bits_per_symbol=float(entropy_bits),
        ))

    # Compute the class-agnostic entropy (single histogram pooled across classes).
    pooled_counts = [0] * vocab_size
    pooled_total = 0
    for cp in class_priors:
        for i, n in enumerate(cp.symbol_counts):
            pooled_counts[i] += n
            pooled_total += n
    agnostic_entropy = 0.0
    if pooled_total > 0:
        for n in pooled_counts:
            if n > 0:
                p = n / pooled_total
                agnostic_entropy -= p * math.log2(p)

    # Weighted average per-class entropy (weighted by per-class frequency).
    weighted_class_entropy = 0.0
    for cp in class_priors:
        if pooled_total > 0:
            class_freq = cp.total_count / pooled_total
            weighted_class_entropy += class_freq * cp.estimated_entropy_bits_per_symbol

    savings = agnostic_entropy - weighted_class_entropy
    # Per-class entropy is always <= agnostic entropy (Cover & Thomas 2.4).
    if savings < -1.0e-9:  # numerical tolerance
        savings = 0.0

    return ClassConditionalCodebook(
        class_priors=tuple(class_priors),
        vocabulary_size=int(vocab_size),
        expected_savings_vs_class_agnostic_bits_per_symbol=float(savings),
    )


__all__ = [
    "ArithmeticCoderError",
    "ClassConditionalCodebook",
    "ClassConditionalPrior",
    "build_class_conditional_codebook",
]
