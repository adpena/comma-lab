# SPDX-License-Identifier: MIT
"""Training pair weights derived from canonical per-pair master gradients.

This is the training-time execution hook for per-pair master-gradient signal:
it converts a per-pair gradient tensor into bounded, mean-normalized weights
that trainers can pass to samplers, loss masks, distillation focus, or
curriculum stage selection. It is planning-side signal only; promotion still
requires byte-closed archive/runtime custody plus exact same-axis eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.master_gradient_consumers import (
    load_per_pair_gradient_from_anchor,
    per_pair_difficulty_atlas,
)

__all__ = [
    "MasterGradientPairWeights",
    "derive_master_gradient_pair_weights",
    "load_master_gradient_pair_weights_for_archive",
]


@dataclass(frozen=True)
class MasterGradientPairWeights:
    """Bounded pair weights for training curriculum consumers."""

    archive_sha256: str
    pair_weights: tuple[float, ...]
    top_k_hardest_pair_indices: tuple[int, ...]
    bottom_k_easiest_pair_indices: tuple[int, ...]
    measurement_axis: str
    measurement_hardware: str
    normalization: str
    min_weight: float
    max_weight: float
    rationale: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    evidence_grade: str = "[predicted; training pair weights from per-pair master-gradient v1]"

    def as_policy(self) -> dict[str, Any]:
        """Return a JSON-safe policy block for trainer/pipeline callers."""
        return {
            "archive_sha256": self.archive_sha256,
            "pair_weights": list(self.pair_weights),
            "top_k_hardest_pair_indices": list(self.top_k_hardest_pair_indices),
            "bottom_k_easiest_pair_indices": list(self.bottom_k_easiest_pair_indices),
            "measurement_axis": self.measurement_axis,
            "measurement_hardware": self.measurement_hardware,
            "normalization": self.normalization,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "rationale": self.rationale,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "evidence_grade": self.evidence_grade,
        }


def _validate_weight_bounds(min_weight: float, max_weight: float) -> None:
    if not np.isfinite(min_weight) or not np.isfinite(max_weight):
        raise ValueError("min_weight and max_weight must be finite")
    if min_weight <= 0.0:
        raise ValueError(f"min_weight must be > 0; got {min_weight!r}")
    if max_weight < min_weight:
        raise ValueError(
            f"max_weight must be >= min_weight; got min={min_weight!r}, max={max_weight!r}"
        )


def derive_master_gradient_pair_weights(
    per_pair_gradient: np.ndarray,
    *,
    archive_sha256: str,
    measurement_axis: str,
    measurement_hardware: str,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    top_k: int = 50,
    bottom_k: int = 50,
) -> MasterGradientPairWeights:
    """Derive bounded mean-normalized pair weights from a per-pair gradient.

    The base signal is the per-pair L2 gradient norm from
    :func:`tac.master_gradient_consumers.per_pair_difficulty_atlas`. We divide
    by the mean positive norm and clip to ``[min_weight, max_weight]`` so one
    extreme pair cannot collapse the sampler or loss surface.
    """
    _validate_weight_bounds(min_weight, max_weight)
    if not isinstance(archive_sha256, str) or len(archive_sha256) < 12:
        raise ValueError(f"archive_sha256 must be a 12+ char string; got {archive_sha256!r}")

    atlas = per_pair_difficulty_atlas(
        per_pair_gradient,
        archive_sha256=archive_sha256,
        measurement_axis=measurement_axis,
        measurement_hardware=measurement_hardware,
        top_k=top_k,
        bottom_k=bottom_k,
        write_sidecar=False,
    )
    if atlas.n_pairs <= 0:
        raise ValueError("per_pair_gradient must contain at least one pair")

    norms = np.zeros(atlas.n_pairs, dtype=np.float64)
    for entry in atlas.entries:
        norms[int(entry.pair_index)] = float(entry.gradient_norm_l2)
    positive = norms[norms > 0.0]
    if positive.size == 0:
        weights = np.ones(atlas.n_pairs, dtype=np.float64)
        normalization = "uniform_zero_gradient"
    else:
        mean_norm = float(positive.mean())
        weights = norms / mean_norm
        weights = np.where(norms > 0.0, weights, min_weight)
        normalization = "l2_norm_divided_by_positive_mean_then_clipped"
    weights = np.clip(weights, min_weight, max_weight)
    # Preserve a mean near 1.0 after clipping without violating caps.
    mean_weight = float(weights.mean()) if weights.size else 1.0
    if mean_weight > 0.0:
        weights = np.clip(weights / mean_weight, min_weight, max_weight)

    rationale = (
        "Training curriculum pair weights derived from canonical per-pair "
        f"master-gradient difficulty atlas; axis={measurement_axis}, "
        f"hardware={measurement_hardware}. Advisory axes remain planning-only "
        "and cannot become contest promotion evidence without exact same-axis "
        "archive/runtime eval custody."
    )
    return MasterGradientPairWeights(
        archive_sha256=archive_sha256,
        pair_weights=tuple(float(x) for x in weights.tolist()),
        top_k_hardest_pair_indices=atlas.top_k_hardest_pair_indices,
        bottom_k_easiest_pair_indices=atlas.bottom_k_easiest_pair_indices,
        measurement_axis=measurement_axis,
        measurement_hardware=measurement_hardware,
        normalization=normalization,
        min_weight=float(min_weight),
        max_weight=float(max_weight),
        rationale=rationale,
    )


def load_master_gradient_pair_weights_for_archive(
    archive_sha256: str,
    *,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    top_k: int = 50,
    bottom_k: int = 50,
) -> MasterGradientPairWeights:
    """Load canonical per-pair gradient anchor and derive training weights."""
    per_pair, anchor = load_per_pair_gradient_from_anchor(
        archive_sha256=archive_sha256
    )
    return derive_master_gradient_pair_weights(
        per_pair,
        archive_sha256=archive_sha256,
        measurement_axis=str(anchor.get("measurement_axis", "")),
        measurement_hardware=str(anchor.get("measurement_hardware", "")),
        min_weight=min_weight,
        max_weight=max_weight,
        top_k=top_k,
        bottom_k=bottom_k,
    )
