# SPDX-License-Identifier: MIT
"""Unified prediction interface — TRACK A + TRACK B ensemble (operator override).

Per operator-frontier-override 2026-05-19 verbatim *"we shoud pursue PP in
parallel"* (capture at
``.omx/research/operator_authorizations/findings_lagrangian_pp_parallel_pursuit_plus_all_voices_matter_override_20260519T080000Z.md``):

Both tracks emit predictions through THIS canonical interface:

- TRACK A: closed-form Gaussian (per slot 20 + supplemental consolidated;
  ``tac.findings_lagrangian.posterior.GaussianPosterior`` → ScalarPrediction).
- TRACK B: NumPyro hierarchical posteriors over architectures (per operator
  override; ``tac.findings_lagrangian_pp.pp_posterior`` → ScalarPrediction).
- ENSEMBLE: weighted by predicted-vs-empirical residual; cathedral autopilot
  ranker consumes the ensemble for ranking. The ensemble weighting
  EMPIRICALLY TESTS slot 20's CARGO-CULTED-PP Assumption-Adversary
  classification: if track B's residuals are NO BETTER than track A,
  the council's REFUSE was right; if PP demonstrably outperforms,
  RECLASSIFIED-AS-HARD-EARNED.

Per Catalog #323 canonical Provenance umbrella: every prediction carries
``[predicted]`` axis tag + Provenance for downstream consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


__all__ = [
    "ScalarPrediction",
    "UnifiedPrediction",
    "ensemble_prediction_from_tracks",
    "EnsembleError",
]


class EnsembleError(ValueError):
    """Raised when ensemble inputs are malformed."""


@dataclass(frozen=True)
class ScalarPrediction:
    """One scalar prediction with uncertainty + provenance.

    Both TRACK A (closed-form Gaussian) and TRACK B (NumPyro hierarchical)
    emit ScalarPrediction. The cathedral autopilot ranker consumes
    ``predicted_value`` for ranking + ``uncertainty_sigma`` for the Q7
    σ-aware downweighting per Catalog #125 hook #4.
    """

    predicted_value: float
    uncertainty_sigma: float
    axis_tag: str  # one of "[predicted]", "[contest-CUDA]", "[contest-CPU]" etc.
    source_track: str  # one of "track_a_handrolled", "track_b_numpyro", "ensemble"
    equation_id: str
    n_anchors_consumed: int
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.predicted_value, (int, float)):
            raise EnsembleError("predicted_value must be numeric")
        if self.predicted_value != self.predicted_value:  # NaN
            raise EnsembleError("predicted_value is NaN")
        if not isinstance(self.uncertainty_sigma, (int, float)):
            raise EnsembleError("uncertainty_sigma must be numeric")
        if self.uncertainty_sigma < 0:
            raise EnsembleError(
                f"uncertainty_sigma={self.uncertainty_sigma} must be >= 0"
            )
        if not isinstance(self.axis_tag, str) or not self.axis_tag.strip():
            raise EnsembleError("axis_tag must be non-empty string")
        if not isinstance(self.source_track, str) or self.source_track not in {
            "track_a_handrolled",
            "track_b_numpyro",
            "ensemble",
        }:
            raise EnsembleError(
                f"source_track={self.source_track!r} must be one of "
                "{track_a_handrolled, track_b_numpyro, ensemble}"
            )
        if not isinstance(self.equation_id, str) or not self.equation_id.strip():
            raise EnsembleError("equation_id must be non-empty string")
        if not isinstance(self.n_anchors_consumed, int) or self.n_anchors_consumed < 0:
            raise EnsembleError(
                f"n_anchors_consumed={self.n_anchors_consumed} must be non-negative int"
            )
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise EnsembleError("rationale must be non-empty string")

    @property
    def is_promotable(self) -> bool:
        """False for [predicted]; True only for [contest-CUDA]/[contest-CPU].

        Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable:
        predictions from the findings Lagrangian are NEVER directly promotable;
        they may inform dispatch ranking but cannot constitute a score claim
        until paired Linux x86_64 contest-axis empirical anchor lands.
        """
        return self.axis_tag in {"[contest-CUDA]", "[contest-CPU]"}

    def as_dict(self) -> dict[str, object]:
        """JSON-safe serialization."""
        return {
            "predicted_value": float(self.predicted_value),
            "uncertainty_sigma": float(self.uncertainty_sigma),
            "axis_tag": self.axis_tag,
            "source_track": self.source_track,
            "equation_id": self.equation_id,
            "n_anchors_consumed": int(self.n_anchors_consumed),
            "rationale": self.rationale,
            "is_promotable": self.is_promotable,
        }


@dataclass(frozen=True)
class UnifiedPrediction:
    """Canonical interface consumed by cathedral autopilot ranker.

    Both TRACK A + TRACK B emit ScalarPrediction; the ensemble combines them.
    The ensemble_prediction is what the cathedral autopilot ranker actually
    uses; the per-track predictions are preserved for empirical comparison
    (the test hypothesis for slot 20's CARGO-CULTED-PP classification).
    """

    track_a_prediction: ScalarPrediction
    track_b_prediction: Optional[ScalarPrediction]  # None if TRACK B not built yet
    ensemble_prediction: ScalarPrediction
    equation_id: str
    track_a_weight: float = 0.5
    track_b_weight: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.track_a_prediction, ScalarPrediction):
            raise EnsembleError("track_a_prediction must be ScalarPrediction")
        if self.track_b_prediction is not None and not isinstance(
            self.track_b_prediction, ScalarPrediction
        ):
            raise EnsembleError("track_b_prediction must be ScalarPrediction or None")
        if not isinstance(self.ensemble_prediction, ScalarPrediction):
            raise EnsembleError("ensemble_prediction must be ScalarPrediction")
        if self.ensemble_prediction.source_track != "ensemble":
            raise EnsembleError(
                f"ensemble_prediction.source_track must be 'ensemble', got "
                f"{self.ensemble_prediction.source_track!r}"
            )
        if not isinstance(self.equation_id, str) or not self.equation_id.strip():
            raise EnsembleError("equation_id must be non-empty string")
        if self.track_a_prediction.equation_id != self.equation_id:
            raise EnsembleError(
                f"track_a_prediction.equation_id={self.track_a_prediction.equation_id!r} "
                f"!= {self.equation_id!r}"
            )
        if (
            self.track_b_prediction is not None
            and self.track_b_prediction.equation_id != self.equation_id
        ):
            raise EnsembleError(
                f"track_b_prediction.equation_id={self.track_b_prediction.equation_id!r} "
                f"!= {self.equation_id!r}"
            )
        if not 0.0 <= self.track_a_weight <= 1.0:
            raise EnsembleError(
                f"track_a_weight={self.track_a_weight} must be in [0, 1]"
            )
        if not 0.0 <= self.track_b_weight <= 1.0:
            raise EnsembleError(
                f"track_b_weight={self.track_b_weight} must be in [0, 1]"
            )

    def as_dict(self) -> dict[str, object]:
        """JSON-safe serialization."""
        return {
            "equation_id": self.equation_id,
            "track_a_weight": float(self.track_a_weight),
            "track_b_weight": float(self.track_b_weight),
            "track_a_prediction": self.track_a_prediction.as_dict(),
            "track_b_prediction": (
                self.track_b_prediction.as_dict() if self.track_b_prediction else None
            ),
            "ensemble_prediction": self.ensemble_prediction.as_dict(),
        }


def ensemble_prediction_from_tracks(
    track_a: ScalarPrediction,
    track_b: Optional[ScalarPrediction],
    *,
    track_a_weight: float = 0.5,
    track_b_weight: float = 0.5,
) -> UnifiedPrediction:
    """Combine TRACK A + TRACK B predictions into a weighted ensemble.

    Per operator override 2026-05-19: cathedral autopilot consumes
    ``UnifiedPrediction.ensemble_prediction`` for ranking. The ensemble
    weighting EMPIRICALLY TESTS slot 20's CARGO-CULTED-PP classification.

    Default weighting: 50/50. The operator/autopilot may adjust weights
    based on observed predicted-vs-empirical residuals (down-weight the
    track whose predictions are systematically biased).

    If TRACK B is None (not yet built or unavailable), the ensemble
    degenerates to TRACK A's prediction with explicit cite-chain.

    Args:
        track_a: TRACK A closed-form Gaussian prediction.
        track_b: TRACK B NumPyro hierarchical prediction (may be None).
        track_a_weight, track_b_weight: ensemble weights.

    Returns:
        UnifiedPrediction with all 3 predictions + weights.
    """
    if not isinstance(track_a, ScalarPrediction):
        raise EnsembleError("track_a must be ScalarPrediction")
    if track_b is not None and not isinstance(track_b, ScalarPrediction):
        raise EnsembleError("track_b must be ScalarPrediction or None")
    if track_a.source_track != "track_a_handrolled":
        raise EnsembleError(
            f"track_a.source_track must be 'track_a_handrolled', got {track_a.source_track!r}"
        )
    if track_b is not None and track_b.source_track != "track_b_numpyro":
        raise EnsembleError(
            f"track_b.source_track must be 'track_b_numpyro', got {track_b.source_track!r}"
        )
    if track_b is not None and track_a.equation_id != track_b.equation_id:
        raise EnsembleError(
            f"track_a.equation_id={track_a.equation_id!r} != track_b.equation_id={track_b.equation_id!r}"
        )

    if track_b is None:
        # Ensemble degenerates to track A.
        ensemble = ScalarPrediction(
            predicted_value=track_a.predicted_value,
            uncertainty_sigma=track_a.uncertainty_sigma,
            axis_tag=track_a.axis_tag,
            source_track="ensemble",
            equation_id=track_a.equation_id,
            n_anchors_consumed=track_a.n_anchors_consumed,
            rationale=(
                f"Ensemble degenerated to TRACK A (closed-form Gaussian); "
                f"TRACK B (NumPyro) not yet built or unavailable. "
                f"Per operator-frontier-override 2026-05-19 + slot 20 + "
                f"supplemental: track_a is the canonical fallback."
            ),
        )
        return UnifiedPrediction(
            track_a_prediction=track_a,
            track_b_prediction=None,
            ensemble_prediction=ensemble,
            equation_id=track_a.equation_id,
            track_a_weight=1.0,
            track_b_weight=0.0,
        )

    # Normalize weights.
    total = track_a_weight + track_b_weight
    if total <= 0:
        raise EnsembleError(
            f"track_a_weight + track_b_weight = {total} must be > 0"
        )
    a_w = track_a_weight / total
    b_w = track_b_weight / total

    ensemble_value = a_w * track_a.predicted_value + b_w * track_b.predicted_value
    # Combined sigma per independent-source uncertainty propagation:
    # sigma_ensemble = sqrt(a_w^2 * sigma_a^2 + b_w^2 * sigma_b^2)
    ensemble_sigma = (
        a_w**2 * track_a.uncertainty_sigma**2
        + b_w**2 * track_b.uncertainty_sigma**2
    ) ** 0.5
    # Conservative axis_tag: take the LESS-promotable of the two.
    if "[contest" in track_a.axis_tag and "[contest" in track_b.axis_tag:
        ensemble_axis = track_a.axis_tag
    else:
        ensemble_axis = "[predicted]"

    ensemble = ScalarPrediction(
        predicted_value=float(ensemble_value),
        uncertainty_sigma=float(ensemble_sigma),
        axis_tag=ensemble_axis,
        source_track="ensemble",
        equation_id=track_a.equation_id,
        n_anchors_consumed=max(track_a.n_anchors_consumed, track_b.n_anchors_consumed),
        rationale=(
            f"Ensemble of TRACK A (closed-form Gaussian; weight={a_w:.2f}) + "
            f"TRACK B (NumPyro hierarchical; weight={b_w:.2f}). "
            f"Per operator-frontier-override 2026-05-19; tests slot 20's "
            f"CARGO-CULTED-PP Assumption-Adversary classification empirically."
        ),
    )
    return UnifiedPrediction(
        track_a_prediction=track_a,
        track_b_prediction=track_b,
        ensemble_prediction=ensemble,
        equation_id=track_a.equation_id,
        track_a_weight=a_w,
        track_b_weight=b_w,
    )
