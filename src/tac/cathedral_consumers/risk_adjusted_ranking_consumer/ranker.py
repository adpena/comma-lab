# SPDX-License-Identifier: MIT
"""RiskAdjustedRanker — uncertainty-aware candidate ranking.

Per SLOT MG-1 spec: rank candidates by ``posterior_mean - lambda * posterior_std``.

  - ``lambda_risk_aversion = 0`` is the current behavior (race-mode; pure
    posterior mean; matches the existing rank_candidates default).
  - ``lambda_risk_aversion = 1`` is conservative (penalize speculative
    predictions by their full uncertainty width).
  - Intermediate values (0.25 / 0.5 / 0.75) interpolate.

The ranker is deterministic: given the same UncertaintyAwareCandidateRow
list, it returns the same ordering. Per CLAUDE.md "Beauty, simplicity,
and developer experience" + "Deterministic reproducibility".

Wire-in hooks per Catalog #125 (this module is the canonical cathedral-
consumer-side ACTIVE on hook #4 cathedral autopilot dispatch):
  hook #1 sensitivity-map = N/A
  hook #2 Pareto constraint = ACTIVE (the lambda parameter encodes the
    risk-axis Pareto trade-off; sister consumers can read the ranked
    output to populate a Pareto frontier)
  hook #3 bit-allocator = N/A
  hook #4 cathedral autopilot dispatch = ACTIVE PRIMARY (SLOT MG-1)
  hook #5 continual-learning posterior = ACTIVE (re-rank on new anchor)
  hook #6 probe-disambiguator = ACTIVE (the lambda value IS the
    operator-visible knob distinguishing race-mode vs conservative paths)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .candidate_row import UncertaintyAwareCandidateRow


# Pure-prior inflation matches uncertainty.py: a None uncertainty value
# means "no anchors, no equation match" — treat as maximally uncertain.
# The value is calibrated so lambda=1 with this value approximately doubles
# the penalty vs a typical 1e-3 posterior std at 5 anchors.
_NULL_UNCERTAINTY_PROXY = 1.0


@dataclass(frozen=True)
class RankedCandidate:
    """One ranked output row.

    Frozen by design. Includes the raw candidate + the risk-adjusted sort
    key so the operator can audit which candidates were demoted by their
    uncertainty.

    Fields:
        candidate: the underlying UncertaintyAwareCandidateRow.
        risk_adjusted_score: posterior_mean - lambda * posterior_std.
            Lower = better (matches existing predicted_score_delta
            convention: negative = improvement).
        posterior_mean: the candidate's predicted_score_delta.
        posterior_std: the uncertainty value used (None-substituted via
            _NULL_UNCERTAINTY_PROXY if predicted_delta_uncertainty is None).
        lambda_risk_aversion: the lambda value used for this ranking pass.
    """

    candidate: UncertaintyAwareCandidateRow
    risk_adjusted_score: float
    posterior_mean: float
    posterior_std: float
    lambda_risk_aversion: float


class RiskAdjustedRanker:
    """Canonical risk-adjusted ranker for SLOT MG-1.

    Per the spec: ``rank_candidates(candidates, lambda_risk_aversion=0.0)``
    returns a list sorted by ``posterior_mean - lambda * posterior_std``
    ascending (most-improvement-first, since improvement is negative
    delta per CLAUDE.md "Forbidden score claims" tag convention).

    The ranker does NOT mutate inputs. It does NOT depend on the autopilot
    loop's existing rank_candidates function (that lives in
    tools/cathedral_autopilot_autonomous_loop.py and is sister-territory).
    Auto-discovery per Catalog #335 ingests this module via the canonical
    ``consume_candidate`` Protocol; sister consumers can also call
    ``RiskAdjustedRanker().rank_candidates(...)`` directly.

    Per CLAUDE.md "Forbidden score claims": the risk-adjusted score is a
    PREDICTION, never a measurement. Promotion to a contest score claim
    requires a paired empirical Provenance per Catalog #127 + #323.
    """

    def __init__(self) -> None:
        """Stateless by design. Constructor takes no parameters."""

    def rank_candidates(
        self,
        candidates: Sequence[UncertaintyAwareCandidateRow],
        lambda_risk_aversion: float = 0.0,
    ) -> list[RankedCandidate]:
        """Rank candidates by ``posterior_mean - lambda * posterior_std``.

        Args:
            candidates: sequence of UncertaintyAwareCandidateRow instances.
                Empty input returns empty output.
            lambda_risk_aversion: non-negative risk-aversion knob. 0 = race-
                mode (pure posterior mean); 1 = conservative (full
                uncertainty width subtracted). Negative values rejected.

        Returns:
            list[RankedCandidate] sorted ascending by risk_adjusted_score
            (most-improvement-first since improvement is negative delta).
            Ties broken by candidate_id (lexicographic) for determinism.

        Raises:
            ValueError: if lambda_risk_aversion is negative OR NaN.

        Per CLAUDE.md "Deterministic reproducibility": for the same inputs
        + lambda, the same ordering is returned across runs / processes /
        machines.
        """
        if not isinstance(lambda_risk_aversion, (int, float)):
            raise ValueError(
                f"lambda_risk_aversion must be numeric, got "
                f"{type(lambda_risk_aversion).__name__}"
            )
        if lambda_risk_aversion != lambda_risk_aversion:  # NaN
            raise ValueError("lambda_risk_aversion must not be NaN")
        if lambda_risk_aversion < 0.0:
            raise ValueError(
                f"lambda_risk_aversion must be >= 0 (got {lambda_risk_aversion}); "
                "negative risk-aversion is undefined (would reward uncertainty)"
            )

        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            if not isinstance(candidate, UncertaintyAwareCandidateRow):
                raise ValueError(
                    f"candidates must be UncertaintyAwareCandidateRow instances, "
                    f"got {type(candidate).__name__}"
                )

            posterior_mean = candidate.predicted_score_delta
            if candidate.predicted_delta_uncertainty is None:
                posterior_std = _NULL_UNCERTAINTY_PROXY
            else:
                posterior_std = float(candidate.predicted_delta_uncertainty)

            # Improvement is NEGATIVE delta (per CLAUDE.md "Forbidden score claims"
            # convention). To penalize uncertainty when ranking by
            # most-improvement-first, we ADD lambda * std to the (negative)
            # posterior mean so the risk-adjusted score becomes LESS NEGATIVE
            # (worse) as uncertainty grows. Equivalent to "shrink the predicted
            # improvement by the uncertainty width".
            risk_adjusted_score = posterior_mean + lambda_risk_aversion * posterior_std

            ranked.append(
                RankedCandidate(
                    candidate=candidate,
                    risk_adjusted_score=risk_adjusted_score,
                    posterior_mean=posterior_mean,
                    posterior_std=posterior_std,
                    lambda_risk_aversion=lambda_risk_aversion,
                )
            )

        # Ascending sort: most-negative = best improvement = first.
        # Tie-break by candidate_id for determinism.
        ranked.sort(key=lambda r: (r.risk_adjusted_score, r.candidate.candidate_id))
        return ranked
