# SPDX-License-Identifier: MIT
"""Canonical 3-metric trichotomy ranking for cathedral autopilot (GAP 1).

Per operator binding correction 2026-05-28 (3rd in sequence) verbatim:
*"we don't ujst want the safest score lowering; we want highest EV in
shortest wall clock which is not necessarily the safest and most
incredmenetal and closest to ready"*.

Per ``feedback_canonical_ev_metric_trichotomy_hygiene_vs_frontier_vs_highest_ev_shortest_wall_clock_20260528.md``:
the canonical METRIC TRICHOTOMY is

  - **HYGIENE-EV** — lesson-honored vector per PR-95-parity 13 lessons;
    correlates with discipline quality.
  - **FRONTIER-BREAKING-EV** — predicted ΔS magnitude × empirical
    confidence × cost-class; correlates with predicted outcome.
  - **HIGHEST-EV-SHORTEST-WALL-CLOCK** — operator's canonical metric =
    ``(predicted ΔS magnitude × probability materializes) / wall-clock-to-validation``;
    variance-acceptable upside-per-time.

These are ORTHOGONAL: conflating ANY two produces canonical anti-pattern
recurrence (cf. anti-pattern
``hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation_canonical_rediscovery_v1``).

This helper replaces the cathedral autopilot ranker's single composite
``predicted_delta`` ranking with independent 3-metric ranking. Per
CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
all three metrics are surfaced + the operator-canonical-metric routing
default is ``highest_ev_shortest_wall_clock`` (variance-acceptance
dominates safest-incremental per EV math).

Per Catalog #341 (Tier A canonical-routing-markers): the ranking is
OBSERVABILITY-ONLY at landing — every CandidateWithThreeMetric row
carries ``promotable=False`` + ``axis_tag="[predicted]"`` + the canonical
Provenance per Catalog #323 + #356.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


# Canonical operator-default metric per operator binding correction
# 2026-05-28 ~23:40Z. The 3rd metric in the trichotomy IS what the
# operator wants for score-lowering work.
HIGHEST_EV_SHORTEST_WALL_CLOCK = "highest_ev_shortest_wall_clock"
HYGIENE_EV = "hygiene_ev"
FRONTIER_BREAKING_EV = "frontier_breaking_ev"

VALID_CANONICAL_METRICS = frozenset(
    {HIGHEST_EV_SHORTEST_WALL_CLOCK, HYGIENE_EV, FRONTIER_BREAKING_EV}
)

DEFAULT_OPERATOR_CANONICAL_METRIC = HIGHEST_EV_SHORTEST_WALL_CLOCK


@dataclass(frozen=True)
class CandidateWithThreeMetric:
    """One candidate row with the canonical 3-metric trichotomy decomposition.

    All three metrics are computed INDEPENDENTLY per the canonical
    formulas; the operator-canonical-metric routing default selects
    ``highest_ev_shortest_wall_clock_ev`` as the primary ranking key
    UNLESS operator says otherwise.

    Per Catalog #287/#323 + #341 + #356: every row carries
    ``promotable=False`` + ``axis_tag="[predicted]"`` so the ranking
    is observability-only. The cathedral autopilot ranker consumes
    these rows for ORDERING (Tier A); they are NEVER promoted to score
    claims.

    Args:
        candidate_id: stable id for the candidate (mirrors
            ``CandidateRow.candidate_id``).
        hygiene_ev: lesson-honored EV per PR-95-parity 13 lessons; the
            ratio ``lessons_honored / lessons_total`` for the candidate
            (1.0 = all 13 lessons honored; 0.0 = none honored).
        frontier_breaking_ev: predicted ΔS magnitude × empirical
            confidence × cost-class scaling. Negative ΔS values are
            beneficial (score reduction); the EV is computed on the
            magnitude.
        highest_ev_shortest_wall_clock_ev: operator's canonical metric
            = ``|predicted ΔS magnitude| × probability_materializes /
            wall_clock_to_validation_hours``. Variance-acceptable;
            high-leverage low-probability candidates can dominate.
        rank_per_canonical_metric: integer rank ordered by the operator
            canonical metric (1 = top; ties broken by candidate_id).
        rationale: operator-facing readable string explaining the
            ranking + each metric value + why one dominates the other.
        promotable: ALWAYS False per Tier A canonical-routing markers.
        axis_tag: ``"[predicted]"`` per Catalog #287/#341.
    """

    candidate_id: str
    hygiene_ev: float
    frontier_breaking_ev: float
    highest_ev_shortest_wall_clock_ev: float
    rank_per_canonical_metric: int
    rationale: str
    promotable: bool = False
    axis_tag: str = "[predicted]"

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        for fname in (
            "hygiene_ev",
            "frontier_breaking_ev",
            "highest_ev_shortest_wall_clock_ev",
        ):
            value = getattr(self, fname)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"CandidateWithThreeMetric.{fname} must be numeric, got "
                    f"{type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(
                    f"CandidateWithThreeMetric.{fname} is NaN; consumers must "
                    "emit finite values (use 0.0 for no-signal)"
                )
            if math.isinf(value):
                raise ValueError(
                    f"CandidateWithThreeMetric.{fname} is infinite"
                )
            object.__setattr__(self, fname, float(value))
        if self.promotable is not False:
            raise ValueError(
                "CandidateWithThreeMetric.promotable MUST be False per "
                "Tier A canonical-routing markers (Catalog #341); this "
                "ranking is observability-only by construction"
            )
        if self.axis_tag != "[predicted]":
            raise ValueError(
                f"CandidateWithThreeMetric.axis_tag={self.axis_tag!r} must "
                "be '[predicted]' per Catalog #287/#341"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict per cathedral consumer contract."""
        return {
            "candidate_id": self.candidate_id,
            "hygiene_ev": self.hygiene_ev,
            "frontier_breaking_ev": self.frontier_breaking_ev,
            "highest_ev_shortest_wall_clock_ev": self.highest_ev_shortest_wall_clock_ev,
            "rank_per_canonical_metric": self.rank_per_canonical_metric,
            "rationale": self.rationale,
            "promotable": self.promotable,
            "axis_tag": self.axis_tag,
        }


@dataclass(frozen=True)
class ThreeMetricTrichotomyRankingResult:
    """Result of canonical 3-metric trichotomy ranking.

    Carries:
      - ``operator_canonical_metric`` — which metric was the routing default
      - ``candidates_with_three_metrics`` — per-candidate trichotomy rows
        in operator-canonical-metric rank order
      - ``per_metric_top_candidate`` — dict of metric -> top candidate id
        (surfaces orthogonality: top per HYGIENE-EV may differ from top
        per FRONTIER-BREAKING-EV may differ from top per HIGHEST-EV-
        SHORTEST-WALL-CLOCK)
      - ``rationale`` — operator-facing readable summary explaining the
        ranking + the orthogonality of the three metrics
      - ``promotable`` — ALWAYS False per Tier A canonical-routing markers
      - ``axis_tag`` — ``"[predicted]"``
    """

    operator_canonical_metric: str
    candidates_with_three_metrics: tuple[CandidateWithThreeMetric, ...]
    per_metric_top_candidate: Mapping[str, str]
    rationale: str
    promotable: bool = False
    axis_tag: str = "[predicted]"

    def __post_init__(self) -> None:
        if self.operator_canonical_metric not in VALID_CANONICAL_METRICS:
            raise ValueError(
                f"operator_canonical_metric={self.operator_canonical_metric!r} "
                f"must be one of {sorted(VALID_CANONICAL_METRICS)}"
            )
        if not isinstance(self.candidates_with_three_metrics, tuple):
            raise ValueError(
                "candidates_with_three_metrics must be a tuple (frozen)"
            )
        for i, c in enumerate(self.candidates_with_three_metrics):
            if not isinstance(c, CandidateWithThreeMetric):
                raise ValueError(
                    f"candidates_with_three_metrics[{i}] must be "
                    f"CandidateWithThreeMetric, got {type(c).__name__}"
                )
        if not isinstance(self.per_metric_top_candidate, Mapping):
            raise ValueError("per_metric_top_candidate must be a Mapping")
        if self.promotable is not False:
            raise ValueError(
                "ThreeMetricTrichotomyRankingResult.promotable MUST be False "
                "per Tier A canonical-routing markers (Catalog #341)"
            )
        if self.axis_tag != "[predicted]":
            raise ValueError(
                f"axis_tag={self.axis_tag!r} must be '[predicted]'"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict per cathedral consumer contract."""
        return {
            "operator_canonical_metric": self.operator_canonical_metric,
            "candidates_with_three_metrics": [
                c.as_dict() for c in self.candidates_with_three_metrics
            ],
            "per_metric_top_candidate": dict(self.per_metric_top_candidate),
            "rationale": self.rationale,
            "promotable": self.promotable,
            "axis_tag": self.axis_tag,
        }


def _compute_hygiene_ev(candidate: Mapping[str, Any]) -> float:
    """Compute HYGIENE-EV per PR-95-parity 13-lesson honored ratio.

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
    13 inviolable lessons. The candidate may carry an explicit
    ``hygiene_lessons_honored`` int and ``hygiene_lessons_total`` int
    (defaulting to 13 per HNeRV parity L1-L13); the ratio yields the
    canonical HYGIENE-EV in [0.0, 1.0].

    If neither field is present, returns 0.0 (no signal). Per Catalog
    #287: this is observability-only, never promoted.
    """
    honored = candidate.get("hygiene_lessons_honored")
    total = candidate.get("hygiene_lessons_total", 13)
    if honored is None or not isinstance(honored, (int, float)):
        # Try alternate fields surfaced by cathedral consumer rows.
        honored = candidate.get("lessons_honored_count")
        if honored is None or not isinstance(honored, (int, float)):
            return 0.0
    if not isinstance(total, (int, float)) or total <= 0:
        total = 13
    ratio = max(0.0, min(1.0, float(honored) / float(total)))
    return ratio


def _compute_frontier_breaking_ev(candidate: Mapping[str, Any]) -> float:
    """Compute FRONTIER-BREAKING-EV per predicted ΔS × confidence × cost-class.

    ``frontier_breaking_ev = |predicted_score_delta| × confidence × cost_class_scale``

    Where ``cost_class_scale`` is the inverse of the cost band (smoke=1.0,
    full=0.5, long_burn=0.1) reflecting that cheaper validations are
    proportionally more valuable per dollar. Per the canonical Z1
    empirical revision (cf. ``apply_z1_empirical_revision_to_candidate_delta``)
    the predicted_score_delta is already adjusted for within-class
    saturation; this helper consumes the adjusted value when present.

    Returns 0.0 if no signal; observability-only per Catalog #287.
    """
    delta = candidate.get("predicted_score_delta_z1_adjusted")
    if delta is None:
        delta = candidate.get("predicted_score_delta")
    if delta is None or not isinstance(delta, (int, float)):
        return 0.0
    confidence = candidate.get("empirical_confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))
    cost = candidate.get("estimated_dispatch_cost_usd", 1.0)
    if not isinstance(cost, (int, float)) or cost <= 0:
        cost_class_scale = 0.5
    elif cost < 1.0:
        cost_class_scale = 1.0  # smoke class
    elif cost < 10.0:
        cost_class_scale = 0.5  # full class
    else:
        cost_class_scale = 0.1  # long_burn class
    return abs(float(delta)) * confidence * cost_class_scale


def _compute_highest_ev_shortest_wall_clock_ev(
    candidate: Mapping[str, Any],
) -> float:
    """Compute HIGHEST-EV-SHORTEST-WALL-CLOCK per operator's canonical formula.

    ``EV = (|predicted ΔS magnitude| × probability_materializes) / wall_clock_to_validation_hours``

    The operator's canonical metric formalized per
    ``meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1``
    canonical equation. Variance-acceptable: a low-confidence (15%)
    high-leverage (+374%) candidate dominates a high-confidence (90%)
    low-leverage (+5%) candidate per the EV math.

    Returns 0.0 if no signal; observability-only per Catalog #287.
    """
    delta = candidate.get("predicted_score_delta_z1_adjusted")
    if delta is None:
        delta = candidate.get("predicted_score_delta")
    if delta is None or not isinstance(delta, (int, float)):
        return 0.0
    prob = candidate.get(
        "probability_materializes", candidate.get("empirical_confidence", 0.5)
    )
    if not isinstance(prob, (int, float)):
        prob = 0.5
    prob = max(0.0, min(1.0, float(prob)))
    wc_hours = candidate.get("wall_clock_to_validation_hours")
    if not isinstance(wc_hours, (int, float)) or wc_hours <= 0:
        # Estimate from cost-class proxy: smoke ~0.5h, full ~3h,
        # long_burn ~12h. Per CLAUDE.md "Production-hardened dispatch
        # optimization protocol" Tier 2.
        cost = candidate.get("estimated_dispatch_cost_usd", 1.0)
        if not isinstance(cost, (int, float)) or cost <= 0:
            wc_hours = 3.0
        elif cost < 1.0:
            wc_hours = 0.5
        elif cost < 10.0:
            wc_hours = 3.0
        else:
            wc_hours = 12.0
    return abs(float(delta)) * prob / float(wc_hours)


def rank_candidates_via_three_metric_trichotomy(
    candidates: list[Mapping[str, Any]],
    *,
    operator_canonical_metric: str = DEFAULT_OPERATOR_CANONICAL_METRIC,
) -> ThreeMetricTrichotomyRankingResult:
    """Rank candidates per the canonical 3-metric trichotomy.

    Per ``feedback_canonical_ev_metric_trichotomy_hygiene_vs_frontier_vs_highest_ev_shortest_wall_clock_20260528.md``
    + operator binding correction 2026-05-28 ~23:40Z: the canonical
    routing default is ``highest_ev_shortest_wall_clock`` (operator's
    canonical metric) UNLESS operator explicitly says otherwise.

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + variance-acceptance
    default: this helper replaces the cathedral autopilot ranker's
    single composite ``predicted_delta`` ranking with INDEPENDENT
    3-metric ranking. All three metrics surface per-candidate; the
    routing default selects ONE metric for primary ordering but the
    operator-facing rationale surfaces all three so the ORTHOGONALITY
    is visible.

    Per Catalog #287/#323/#341: every row is observability-only
    (``promotable=False`` + ``axis_tag="[predicted]"``).

    Per ``meta_orchestrator_three_metric_trichotomy_orthogonality_v1``
    canonical equation: conflating ANY two metrics produces canonical
    anti-pattern recurrence (the 3-correction sequence today is the
    empirical anchor).

    Args:
        candidates: list of candidate mappings (typically cathedral
            autopilot ``CandidateRow.as_dict()`` output).
        operator_canonical_metric: one of
            ``VALID_CANONICAL_METRICS``. Default is
            ``HIGHEST_EV_SHORTEST_WALL_CLOCK`` per operator binding
            correction 2026-05-28 ~23:40Z.

    Returns:
        :class:`ThreeMetricTrichotomyRankingResult` with per-candidate
        trichotomy rows in operator-canonical-metric rank order.
    """
    if operator_canonical_metric not in VALID_CANONICAL_METRICS:
        raise ValueError(
            f"operator_canonical_metric={operator_canonical_metric!r} must "
            f"be one of {sorted(VALID_CANONICAL_METRICS)}"
        )

    # Compute all 3 metrics per candidate.
    rows_unranked: list[CandidateWithThreeMetric] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        cid = candidate.get("candidate_id")
        if not cid or not isinstance(cid, str):
            continue
        hygiene = _compute_hygiene_ev(candidate)
        frontier = _compute_frontier_breaking_ev(candidate)
        highest_ev = _compute_highest_ev_shortest_wall_clock_ev(candidate)
        rationale = (
            f"hygiene_ev={hygiene:.4f} (lessons-honored ratio per PR-95-parity 13 lessons); "
            f"frontier_breaking_ev={frontier:.6f} (|ΔS|×confidence×cost-class); "
            f"highest_ev_shortest_wall_clock_ev={highest_ev:.6f} (|ΔS|×prob/wc); "
            f"operator-canonical-routing default={operator_canonical_metric}"
        )
        rows_unranked.append(
            CandidateWithThreeMetric(
                candidate_id=cid,
                hygiene_ev=hygiene,
                frontier_breaking_ev=frontier,
                highest_ev_shortest_wall_clock_ev=highest_ev,
                rank_per_canonical_metric=0,  # set below
                rationale=rationale,
            )
        )

    # Sort by operator-canonical metric (descending; higher EV first).
    # Tie-break by candidate_id for determinism per Catalog #371 no-drift
    # invariant.
    def _sort_key(c: CandidateWithThreeMetric) -> tuple[float, str]:
        if operator_canonical_metric == HIGHEST_EV_SHORTEST_WALL_CLOCK:
            primary = c.highest_ev_shortest_wall_clock_ev
        elif operator_canonical_metric == FRONTIER_BREAKING_EV:
            primary = c.frontier_breaking_ev
        else:
            primary = c.hygiene_ev
        # Negate for descending sort; candidate_id ascending for tie-break.
        return (-primary, c.candidate_id)

    rows_unranked.sort(key=_sort_key)
    rows_ranked = tuple(
        CandidateWithThreeMetric(
            candidate_id=c.candidate_id,
            hygiene_ev=c.hygiene_ev,
            frontier_breaking_ev=c.frontier_breaking_ev,
            highest_ev_shortest_wall_clock_ev=c.highest_ev_shortest_wall_clock_ev,
            rank_per_canonical_metric=rank_i,
            rationale=c.rationale,
        )
        for rank_i, c in enumerate(rows_unranked, start=1)
    )

    # Compute per-metric top candidate (surfaces orthogonality).
    per_metric_top: dict[str, str] = {}
    if rows_ranked:
        top_hygiene = max(rows_ranked, key=lambda r: (r.hygiene_ev, r.candidate_id))
        top_frontier = max(
            rows_ranked, key=lambda r: (r.frontier_breaking_ev, r.candidate_id)
        )
        top_highest = max(
            rows_ranked,
            key=lambda r: (r.highest_ev_shortest_wall_clock_ev, r.candidate_id),
        )
        per_metric_top = {
            HYGIENE_EV: top_hygiene.candidate_id,
            FRONTIER_BREAKING_EV: top_frontier.candidate_id,
            HIGHEST_EV_SHORTEST_WALL_CLOCK: top_highest.candidate_id,
        }

    n = len(rows_ranked)
    distinct_top = len(set(per_metric_top.values())) if per_metric_top else 0
    orthogonality_signal = (
        "distinct tops across all 3 metrics (orthogonality empirically confirmed)"
        if distinct_top == 3
        else f"{distinct_top} distinct top(s) across 3 metrics"
        if distinct_top > 0
        else "(no candidates ranked)"
    )
    rationale = (
        f"Ranked {n} candidate(s) per canonical 3-metric trichotomy "
        f"(routing default: {operator_canonical_metric}). "
        f"Per-metric tops: {per_metric_top}. "
        f"Orthogonality: {orthogonality_signal}. "
        "Per CLAUDE.md 'Apples-to-apples evidence discipline' + Catalog "
        "#287/#323/#341: observability-only (no promotion authority)."
    )

    return ThreeMetricTrichotomyRankingResult(
        operator_canonical_metric=operator_canonical_metric,
        candidates_with_three_metrics=rows_ranked,
        per_metric_top_candidate=per_metric_top,
        rationale=rationale,
    )


__all__ = [
    "CandidateWithThreeMetric",
    "DEFAULT_OPERATOR_CANONICAL_METRIC",
    "FRONTIER_BREAKING_EV",
    "HIGHEST_EV_SHORTEST_WALL_CLOCK",
    "HYGIENE_EV",
    "ThreeMetricTrichotomyRankingResult",
    "VALID_CANONICAL_METRICS",
    "rank_candidates_via_three_metric_trichotomy",
]
