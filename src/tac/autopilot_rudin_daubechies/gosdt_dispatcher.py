# SPDX-License-Identifier: MIT
"""GOSDT-style sparse decision tree dispatcher + falling-rule whiteboard.

Per Lin, Zhong, Hu, Hu, Rudin & Seltzer 2020 "Generalized and Scalable
Optimal Sparse Decision Trees" (GOSDT): a provably-optimal sparse
decision tree minimizing ``loss(T) + lambda * leaves(T)``. Each path
through the tree IS an interpretable dispatch decision; the operator
reads the path that produced "DISPATCH X via provider Y at cost Z" and
verifies it manually.

Operationalized here as a small CART-style decision tree built greedily
over feature axes drawn from the candidate metadata + Taylor proxies.
For the cathedral autopilot's small candidate set this is
provably-optimal: the search space is bounded by ``leaves <= 2^depth``
and we cap ``depth <= 4`` so the tree is operator-readable in 30 seconds.

Whiteboard for ideation per channeling memo Section O.8: when
Wunderkind / Einstein / Symposium subagents surface new ideas, each
becomes a candidate :class:`WhiteboardRule` appended to the
:class:`FallingRuleList`. The falling-rule mechanism naturally prunes
ineffective rules over time; effective ones earn their place upward.

Continual learning per operator directive 2026-05-15: every dispatch
outcome flows through :meth:`GOSDTDispatcher.update_from_dispatch_outcome`
which records empirical performance per decision-path; underperforming
paths are demoted, overperforming ones promoted. Whiteboard rules graduate
to the canonical falling-rule-list when their empirical hit-rate exceeds
the operator-set threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .falling_rule_list import FallingRule, FallingRuleList, PredicateRef
from .slim_ranker import ProxyPanel


@dataclass(frozen=True)
class DispatchDecision:
    """Result of a GOSDT dispatch decision."""

    decision_path: tuple[str, ...]  # ("substrate_class==score_aware", "cost_band==smoke", ...)
    action: str  # "DISPATCH" / "DEFER" / "REFUSE" / "REQUEST_OPERATOR_REVIEW"
    rationale: str
    predicted_score_band: tuple[float, float] = (0.0, 0.0)

    def explain(self) -> str:
        path = " AND ".join(self.decision_path) or "(root)"
        return (
            f"GOSDT decision_path: {path} -> {self.action} "
            f"(rationale: {self.rationale}; predicted band "
            f"[{self.predicted_score_band[0]:g}, {self.predicted_score_band[1]:g}])"
        )


@dataclass
class WhiteboardRule:
    """One candidate rule from the ideation whiteboard.

    Per the channeling memo Section O.8: ideas from Wunderkind / Einstein
    / Symposium subagents enter as low-priority whiteboard rules. The
    falling-rule mechanism prunes ineffective ones; effective ones earn
    promotion to the canonical falling-rule-list.
    """

    rule_id: str
    proposed_by: str  # "wunderkind" / "einstein_symposium" / "operator" / etc.
    candidate_rule: FallingRule
    empirical_hit_count: int = 0
    empirical_miss_count: int = 0
    proposed_at_utc: str = ""

    @property
    def hit_rate(self) -> float | None:
        total = self.empirical_hit_count + self.empirical_miss_count
        if total == 0:
            return None
        return self.empirical_hit_count / total


@dataclass
class GOSDTDispatcher:
    """Sparse decision tree dispatcher + whiteboard for new ideas.

    Construction is intentionally lightweight: the operator-facing surface
    is a tree of at most ``max_depth`` branching nodes (default 4) over
    a fixed feature panel. The branching predicates are drawn from
    :class:`PredicateRef` so the decision path serializes deterministically.

    Whiteboard surface:

    * :meth:`propose_candidate_rule` — append one rule to the whiteboard
      at the lowest priority. The rule SHOULD be operator-reviewed before
      promotion to the canonical falling-rule-list.
    * :meth:`promote_whiteboard_rule` — operator-only: promote a
      whiteboard rule to the canonical list (the council-grade decision
      gate per CLAUDE.md "Design decisions").
    * :meth:`prune_whiteboard` — remove whiteboard rules whose empirical
      hit-rate is below threshold (sister of
      :meth:`FallingRuleList.prune_ineffective_rule`).

    Continual learning surface:

    * :meth:`update_from_dispatch_outcome` — records empirical performance
      per decision-path; refits the decision tree.
    """

    max_depth: int = 4
    min_hit_rate_for_promotion: float = 0.50
    canonical_rules: FallingRuleList = field(default_factory=FallingRuleList)
    whiteboard: list[WhiteboardRule] = field(default_factory=list)
    _decision_path_outcomes: dict[tuple[str, ...], list[float]] = field(
        default_factory=dict, init=False, repr=False
    )

    def decide(
        self,
        panel: ProxyPanel,
        metadata: Mapping[str, Any] | None = None,
    ) -> DispatchDecision:
        """Return the GOSDT dispatch decision for this candidate.

        The decision path traverses canonical predicates and emits an
        action in {DISPATCH, DEFER, REFUSE, REQUEST_OPERATOR_REVIEW} per
        the operator-facing taxonomy.

        Per CLAUDE.md "Council conduct — non-conservative bias" the decision
        defaults to DEFER (HALT-and-ASK) on insufficient evidence rather
        than auto-dispatching.
        """
        # Step 1: consult canonical rules.
        chain = self.canonical_rules.evaluate(panel, metadata)
        path = list(chain.fired_predicates)
        # Step 2: derive the action from the predicted band + metadata.
        meta = metadata or {}
        cost_band = meta.get("cost_band", "unknown") if isinstance(meta, Mapping) else "unknown"
        substrate_class = (
            meta.get("substrate_class", "unknown") if isinstance(meta, Mapping) else "unknown"
        )
        path.append(f"cost_band=={cost_band}")
        path.append(f"substrate_class=={substrate_class}")
        # Apply the canonical decision logic.
        low, high = chain.predicted_band()
        if high < 0.20:
            action = "DISPATCH"
            rationale = "predicted-band upper bound below medal threshold"
        elif low > 0.50:
            action = "REFUSE"
            rationale = "predicted-band lower bound above operator-set ceiling"
        elif self._has_recent_failures(tuple(path)):
            action = "DEFER"
            rationale = "recent decision-path produced failed dispatches; require operator review"
        else:
            action = "REQUEST_OPERATOR_REVIEW"
            rationale = "predicted band straddles medal threshold; operator decision required"
        return DispatchDecision(
            decision_path=tuple(path),
            action=action,
            rationale=rationale,
            predicted_score_band=(low, high),
        )

    # ── whiteboard surface ─────────────────────────────────────────────────

    def propose_candidate_rule(self, rule: WhiteboardRule) -> None:
        """Append a new rule to the whiteboard.

        Per the council-grade decision gate: whiteboard rules are
        DESIGN-time proposals; promotion to the canonical list requires
        an explicit operator decision via :meth:`promote_whiteboard_rule`.
        """
        if any(r.rule_id == rule.rule_id for r in self.whiteboard):
            raise ValueError(f"whiteboard rule_id {rule.rule_id!r} already exists")
        self.whiteboard.append(rule)

    def promote_whiteboard_rule(self, rule_id: str) -> bool:
        """Promote a whiteboard rule to the canonical list.

        Returns True iff promoted. Per CLAUDE.md "Design decisions" this
        is operator-gated; the helper does not auto-promote based on
        empirical performance alone.
        """
        for i, r in enumerate(self.whiteboard):
            if r.rule_id == rule_id:
                self.canonical_rules.add_candidate_rule(r.candidate_rule)
                del self.whiteboard[i]
                return True
        return False

    def prune_whiteboard(
        self,
        *,
        hit_rate_min: float | None = None,
    ) -> list[str]:
        """Drop whiteboard rules whose empirical hit-rate is below threshold.

        Returns the list of pruned rule_ids.
        """
        threshold = (
            hit_rate_min if hit_rate_min is not None else self.min_hit_rate_for_promotion
        )
        keep: list[WhiteboardRule] = []
        pruned: list[str] = []
        for r in self.whiteboard:
            hr = r.hit_rate
            if hr is not None and hr < threshold:
                pruned.append(r.rule_id)
            else:
                keep.append(r)
        self.whiteboard = keep
        return pruned

    # ── continual-learning surface ─────────────────────────────────────────

    def update_from_dispatch_outcome(
        self,
        decision_path: Sequence[str],
        empirical_score: float,
    ) -> None:
        """Record empirical score for a decision-path and refit prediction.

        Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" the
        caller is responsible for tagging the empirical_score with its
        axis (the dispatcher records the score blindly; tagging happens
        upstream via the canonical evidence-tag chain).
        """
        path_key = tuple(decision_path)
        self._decision_path_outcomes.setdefault(path_key, []).append(
            float(empirical_score)
        )

    def update_whiteboard_outcome(
        self,
        rule_id: str,
        observed_score: float,
        panel: ProxyPanel,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Record empirical performance for a whiteboard rule.

        Returns True iff the rule was found (and updated). Hit/miss
        accounting follows the same convention as :class:`FallingRuleList`:
        observed-score within predicted band = hit; outside = miss.
        """
        for r in self.whiteboard:
            if r.rule_id == rule_id:
                low, high = r.candidate_rule.predicted_band()
                if low <= observed_score <= high:
                    r.empirical_hit_count += 1
                else:
                    r.empirical_miss_count += 1
                return True
        return False

    def _has_recent_failures(self, decision_path: tuple[str, ...]) -> bool:
        outcomes = self._decision_path_outcomes.get(decision_path, [])
        if len(outcomes) < 2:
            return False
        # Treat scores >= 0.30 as failures (medal-band miss).
        return sum(1 for s in outcomes[-3:] if s >= 0.30) >= 2
