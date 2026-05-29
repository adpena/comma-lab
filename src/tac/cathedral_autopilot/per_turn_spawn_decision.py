# SPDX-License-Identifier: MIT
"""Per-turn main-thread spawn-decision helper for cathedral autopilot (GAP 3).

Per ``feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md``
GAP 3 + Catalog #376 + Catalog #378 sister sub-agent + parent-spawn-
decision PV gap extinction.

The cathedral autopilot runs as PER-ITERATION continuous loop, NOT a
per-TURN main-thread spawn-decision helper. This helper closes that gap:
runs at main-thread spawn-decision time (NOT continuous-loop time),
routes through the canonical 3-metric trichotomy ranking, respects
cap=1-per-turn discipline under throttle + simultaneous-multi-subagent-
spawn-rate-limit cascade.

Per ``manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1``
canonical anti-pattern: main-thread queue priority shifts per-turn
based on landed sister vs canonical metric. THIS helper makes the
ranking deterministic per canonical metric (NO per-turn drift).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CandidateSelection:
    """Canonical next-spawn selection for main-thread per-turn decision.

    Args:
        selected_candidate_id: id of the next-spawn candidate selected
            per the canonical 3-metric trichotomy ranking + cap-window
            discipline. None if no eligible candidate exists in the queue.
        operator_canonical_metric: which metric the routing default was
            (mirrors :data:`tac.cathedral_autopilot.three_metric_trichotomy.DEFAULT_OPERATOR_CANONICAL_METRIC`).
        cap_window_remaining: cap-window slots remaining AFTER this
            spawn lands (decremented by 1 from caller's input). None if
            no spawn is selected.
        in_flight_sister_count: count of in-flight sister subagents at
            decision time (for audit visibility).
        rationale: operator-facing readable rationale explaining the
            selection + the 3-metric trichotomy values for the selected
            candidate + why cap discipline allows the spawn.
        promotable: ALWAYS False per Tier A canonical-routing markers.
        axis_tag: ``"[predicted]"`` per Catalog #287/#341.
        recommendation: one of "PROCEED" (spawn the selected candidate),
            "WAIT_CAP_EXCEEDED" (cap window is full; do NOT spawn),
            "WAIT_NO_ELIGIBLE_CANDIDATES" (queue empty / all in-flight),
            "STAND_DOWN_PV_FAILURE" (Catalog #378 PV would refuse this
            spawn).
    """

    selected_candidate_id: str | None
    operator_canonical_metric: str
    cap_window_remaining: int | None
    in_flight_sister_count: int
    rationale: str
    recommendation: str
    promotable: bool = False
    axis_tag: str = "[predicted]"

    def __post_init__(self) -> None:
        if self.selected_candidate_id is not None and not isinstance(
            self.selected_candidate_id, str
        ):
            raise ValueError("selected_candidate_id must be a string or None")
        if not isinstance(self.operator_canonical_metric, str):
            raise ValueError("operator_canonical_metric must be a string")
        if self.cap_window_remaining is not None:
            if not isinstance(self.cap_window_remaining, int) or self.cap_window_remaining < 0:
                raise ValueError(
                    "cap_window_remaining must be a non-negative int or None"
                )
        if not isinstance(self.in_flight_sister_count, int) or self.in_flight_sister_count < 0:
            raise ValueError(
                "in_flight_sister_count must be a non-negative int"
            )
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        if self.recommendation not in _VALID_RECOMMENDATIONS:
            raise ValueError(
                f"recommendation={self.recommendation!r} must be one of "
                f"{sorted(_VALID_RECOMMENDATIONS)}"
            )
        if self.promotable is not False:
            raise ValueError(
                "CandidateSelection.promotable MUST be False per Tier A "
                "canonical-routing markers (Catalog #341)"
            )
        if self.axis_tag != "[predicted]":
            raise ValueError(
                f"axis_tag={self.axis_tag!r} must be '[predicted]'"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict per cathedral consumer contract."""
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "operator_canonical_metric": self.operator_canonical_metric,
            "cap_window_remaining": self.cap_window_remaining,
            "in_flight_sister_count": self.in_flight_sister_count,
            "rationale": self.rationale,
            "recommendation": self.recommendation,
            "promotable": self.promotable,
            "axis_tag": self.axis_tag,
        }


_RECOMMENDATION_PROCEED = "PROCEED"
_RECOMMENDATION_WAIT_CAP = "WAIT_CAP_EXCEEDED"
_RECOMMENDATION_WAIT_QUEUE = "WAIT_NO_ELIGIBLE_CANDIDATES"
_RECOMMENDATION_STAND_DOWN_PV = "STAND_DOWN_PV_FAILURE"

_VALID_RECOMMENDATIONS = frozenset(
    {
        _RECOMMENDATION_PROCEED,
        _RECOMMENDATION_WAIT_CAP,
        _RECOMMENDATION_WAIT_QUEUE,
        _RECOMMENDATION_STAND_DOWN_PV,
    }
)


def select_canonical_next_spawn_for_main_thread(
    in_flight_subagents: list[Mapping[str, Any]],
    canonical_queue: list[Mapping[str, Any]],
    *,
    cap_window_remaining: int = 1,
    operator_canonical_metric: str | None = None,
) -> CandidateSelection:
    """Select the canonical next-spawn for the main thread per-turn.

    Per ``feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md``
    GAP 3: the cathedral autopilot's per-iteration continuous loop
    runs at autopilot-loop cadence; the main-thread spawn-decision needs
    a per-TURN helper. THIS helper closes that gap.

    Per Catalog #376 (subagent SPAWN-PV) + Catalog #378 (main-thread
    PARENT-spawn-decision PV): if the PV would refuse the spawn (e.g.
    duplicate-work-on-disk OR sister-in-flight on same files), this
    helper returns STAND_DOWN_PV_FAILURE WITHOUT selecting a candidate.

    Per ``manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1``
    canonical anti-pattern: the ranking is DETERMINISTIC per canonical
    metric (NO per-turn drift based on landed sister). The helper
    consults the 3-metric trichotomy ranking + cap-window discipline +
    in-flight-sister exclusion via candidate_id.

    Args:
        in_flight_subagents: list of in-flight subagent mappings
            (each carries ``candidate_id`` / ``subagent_id`` / ``files_touched``).
        canonical_queue: list of canonical-queue candidate mappings
            (each suitable for ``rank_candidates_via_three_metric_trichotomy``).
        cap_window_remaining: cap-window slots remaining (default 1
            per cap=1-per-turn discipline under throttle).
        operator_canonical_metric: optional override for the routing
            default; defaults to
            :data:`tac.cathedral_autopilot.three_metric_trichotomy.DEFAULT_OPERATOR_CANONICAL_METRIC`.

    Returns:
        :class:`CandidateSelection` with the selected next-spawn (if any)
        + 3-metric trichotomy rationale + cap-window decrement.
    """
    from tac.cathedral_autopilot.three_metric_trichotomy import (
        DEFAULT_OPERATOR_CANONICAL_METRIC,
        rank_candidates_via_three_metric_trichotomy,
        VALID_CANONICAL_METRICS,
    )

    if operator_canonical_metric is None:
        operator_canonical_metric = DEFAULT_OPERATOR_CANONICAL_METRIC
    if operator_canonical_metric not in VALID_CANONICAL_METRICS:
        raise ValueError(
            f"operator_canonical_metric={operator_canonical_metric!r} must "
            f"be one of {sorted(VALID_CANONICAL_METRICS)}"
        )

    if not isinstance(cap_window_remaining, int) or cap_window_remaining < 0:
        raise ValueError(
            f"cap_window_remaining={cap_window_remaining!r} must be a "
            "non-negative int"
        )

    in_flight_count = len(in_flight_subagents)

    if cap_window_remaining < 1:
        return CandidateSelection(
            selected_candidate_id=None,
            operator_canonical_metric=operator_canonical_metric,
            cap_window_remaining=cap_window_remaining,
            in_flight_sister_count=in_flight_count,
            rationale=(
                f"cap_window_remaining={cap_window_remaining} (< 1); "
                f"{in_flight_count} sister(s) in flight. Cap=1-per-turn "
                "discipline refuses additional spawns per CLAUDE.md + sister "
                "rate-limit cascade anti-pattern. WAIT for next cap-window."
            ),
            recommendation=_RECOMMENDATION_WAIT_CAP,
        )

    in_flight_ids = {
        s.get("candidate_id")
        for s in in_flight_subagents
        if isinstance(s, Mapping)
    }
    in_flight_ids.discard(None)

    eligible_queue = [
        c
        for c in canonical_queue
        if isinstance(c, Mapping) and c.get("candidate_id") not in in_flight_ids
    ]

    if not eligible_queue:
        return CandidateSelection(
            selected_candidate_id=None,
            operator_canonical_metric=operator_canonical_metric,
            cap_window_remaining=cap_window_remaining,
            in_flight_sister_count=in_flight_count,
            rationale=(
                f"canonical queue depth={len(canonical_queue)}; "
                f"{in_flight_count} sister(s) in flight; ZERO eligible "
                "candidates after in-flight exclusion. WAIT for queue refill "
                "or sister completion."
            ),
            recommendation=_RECOMMENDATION_WAIT_QUEUE,
        )

    ranking = rank_candidates_via_three_metric_trichotomy(
        eligible_queue, operator_canonical_metric=operator_canonical_metric
    )
    if not ranking.candidates_with_three_metrics:
        return CandidateSelection(
            selected_candidate_id=None,
            operator_canonical_metric=operator_canonical_metric,
            cap_window_remaining=cap_window_remaining,
            in_flight_sister_count=in_flight_count,
            rationale=(
                "3-metric trichotomy ranking yielded ZERO ranked candidates "
                f"from {len(eligible_queue)} eligible queue entries (signal "
                "loss; investigate per Catalog #371 auto-recalibrator)."
            ),
            recommendation=_RECOMMENDATION_WAIT_QUEUE,
        )

    top = ranking.candidates_with_three_metrics[0]

    # Per Catalog #378 PV: best-effort guard against duplicate work on
    # disk OR sister-in-flight files overlap. THIS helper does NOT
    # actually invoke the PV guard (that's the responsibility of the
    # caller per the sister Catalog #340 staging surface) but it
    # surfaces the PROCEED recommendation contingent on the caller
    # honoring PV. The recommendation token PROCEED encodes "ranking
    # complete; caller must invoke Catalog #378 PV before Agent.spawn()".
    rationale = (
        f"Selected next-spawn candidate_id={top.candidate_id!r} per canonical "
        f"3-metric trichotomy ranking (routing default: {operator_canonical_metric}). "
        f"hygiene_ev={top.hygiene_ev:.4f}, "
        f"frontier_breaking_ev={top.frontier_breaking_ev:.6f}, "
        f"highest_ev_shortest_wall_clock_ev={top.highest_ev_shortest_wall_clock_ev:.6f}. "
        f"Cap-window remaining (before this spawn): {cap_window_remaining}; "
        f"{in_flight_count} sister(s) in flight. PROCEED contingent on caller "
        "invoking Catalog #378 verify_head_state_before_main_thread_spawn PV "
        "before Agent.spawn(); cap-window decrements to "
        f"{max(0, cap_window_remaining - 1)} after this spawn lands."
    )

    return CandidateSelection(
        selected_candidate_id=top.candidate_id,
        operator_canonical_metric=operator_canonical_metric,
        cap_window_remaining=max(0, cap_window_remaining - 1),
        in_flight_sister_count=in_flight_count,
        rationale=rationale,
        recommendation=_RECOMMENDATION_PROCEED,
    )


__all__ = [
    "CandidateSelection",
    "select_canonical_next_spawn_for_main_thread",
]
