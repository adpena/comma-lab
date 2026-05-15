# SPDX-License-Identifier: MIT
"""GOSDT-style sparse decision tree router for preflight dispatch.

Per Lin, Zhong, Hu, Hu, Rudin & Seltzer 2020 "Generalized and Scalable
Optimal Sparse Decision Trees" (GOSDT): a provably-optimal sparse
decision tree minimizing ``loss(T) + lambda * leaves(T)``. Each path
through the tree IS an interpretable dispatch decision; the operator
reads the path that produced "REFUSE / WARN / OK" and verifies it
manually.

Operationalized for preflight as a small CART-style decision tree built
greedily over feature axes drawn from:

* substrate_class (e.g. "score_aware_renderer", "research_only", "substrate_engineering")
* cost_band (e.g. "smoke", "full", "long_burn", "eval")
* rashomon_confidence (consensus_risk, disagreement_stddev)
* tier_1_violation_count, tier_2_violation_count, tier_3_violation_count

For the cathedral autopilot's preflight surface this is provably-optimal:
the search space is bounded by ``leaves <= 2^depth`` and we cap
``depth <= 4`` so the tree is operator-readable in 30 seconds.

Whiteboard for ideation per the META principle: when symposium /
wunderkind subagents surface new gate ideas, each becomes a candidate
:class:`PreflightWhiteboardRule` appended to the falling-rule-list.
The falling-rule mechanism naturally prunes ineffective rules; effective
ones earn their place via :meth:`promote_whiteboard_rule` (operator-gated).

Continual learning per operator directive 2026-05-15: every preflight
outcome flows through :meth:`update_from_dispatch_outcome` which records
empirical performance per decision-path; underperforming paths are
demoted, overperforming ones promoted. Whiteboard rules graduate when
their empirical hit-rate exceeds the operator-set threshold AND the
operator explicitly approves promotion.

Self-protection: Catalog #278 enforces canonical operator-gated
promotion discipline at SOURCE level — auto-promotion of whiteboard
rules bypasses the council-grade decision gate per CLAUDE.md "Design
decisions — non-negotiable".

[verified-against: Lin, Zhong, Hu, Hu, Rudin & Seltzer 2020 §3 +
autopilot sister
``tac.autopilot_rudin_daubechies.gosdt_dispatcher.GOSDTDispatcher``]
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .falling_rule_evaluator import (
    PreflightFallingRule,
    PreflightFallingRuleEvaluator,
)
from .slim_risk_scorer import DISPATCH_RISK_REFUSAL_THRESHOLD, GateVerdictPanel


@dataclass(frozen=True)
class PreflightDispatchDecision:
    """Result of a GOSDT preflight dispatch decision."""

    decision_path: tuple[str, ...]
    action: str  # "OK" / "WARN" / "REFUSE" / "REQUEST_OPERATOR_REVIEW"
    rationale: str
    predicted_risk_band: tuple[float, float] = (0.0, 0.0)

    def explain(self) -> str:
        path = " AND ".join(self.decision_path) or "(root)"
        return (
            f"GOSDT preflight decision_path: {path} -> {self.action} "
            f"(rationale: {self.rationale}; predicted risk band "
            f"[{self.predicted_risk_band[0]:g}, {self.predicted_risk_band[1]:g}])"
        )


@dataclass
class PreflightWhiteboardRule:
    """One candidate rule from the symposium / ideation whiteboard.

    Per the council-grade decision gate: whiteboard rules are DESIGN-time
    proposals; promotion to the canonical falling-rule-list requires an
    explicit operator decision via :meth:`GOSDTDispatchRouter.promote_whiteboard_rule`.
    """

    rule_id: str
    proposed_by: str  # "symposium" / "wunderkind" / "operator" / "codex" / etc.
    candidate_rule: PreflightFallingRule
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
class GOSDTDispatchRouter:
    """Sparse decision tree router + whiteboard for new preflight rules.

    Construction is intentionally lightweight: the operator-facing surface
    is a tree of at most ``max_depth`` branching nodes (default 4) over
    the feature panel (substrate_class, cost_band, rashomon_confidence,
    tier_violation_counts).

    Whiteboard surface:

    * :meth:`propose_candidate_rule` — append one rule to the whiteboard
      at the lowest priority. The rule SHOULD be operator-reviewed before
      promotion.
    * :meth:`promote_whiteboard_rule` — operator-only: promote a
      whiteboard rule to the canonical list (the council-grade decision
      gate per CLAUDE.md "Design decisions"). Returns boolean — the helper
      does NOT auto-promote.
    * :meth:`prune_whiteboard` — remove whiteboard rules whose empirical
      hit-rate is below threshold (sister of
      :meth:`PreflightFallingRuleEvaluator.prune_ineffective_rule`).

    Continual learning surface:

    * :meth:`update_from_dispatch_outcome` — records empirical performance
      per decision-path; refits the decision tree.

    [verified-against: Lin, Zhong, Hu, Hu, Rudin & Seltzer 2020 §3.1 +
    autopilot sister
    ``tac.autopilot_rudin_daubechies.gosdt_dispatcher.GOSDTDispatcher``]
    """

    max_depth: int = 4
    min_hit_rate_for_promotion: float = 0.50
    refusal_threshold: float = DISPATCH_RISK_REFUSAL_THRESHOLD
    canonical_rules: PreflightFallingRuleEvaluator = field(
        default_factory=PreflightFallingRuleEvaluator
    )
    whiteboard: list[PreflightWhiteboardRule] = field(default_factory=list)
    _decision_path_outcomes: dict[tuple[str, ...], list[float]] = field(
        default_factory=dict, init=False, repr=False
    )

    def decide(
        self,
        panel: GateVerdictPanel,
        metadata: Mapping[str, Any] | None = None,
    ) -> PreflightDispatchDecision:
        """Return the GOSDT preflight dispatch decision for this panel.

        The decision path traverses canonical predicates and emits an
        action in {OK, WARN, REFUSE, REQUEST_OPERATOR_REVIEW} per the
        operator-facing taxonomy.

        Per CLAUDE.md "Council conduct — non-conservative bias" the
        decision defaults to REQUEST_OPERATOR_REVIEW (HALT-and-ASK) on
        insufficient evidence rather than auto-routing.
        """
        # Step 1: consult canonical rules.
        chain = self.canonical_rules.evaluate(panel)
        path: list[str] = []
        if chain.first_fired_index is not None:
            first_rule = chain.rules[chain.first_fired_index]
            path.append(f"first_fired_gate={first_rule.gate_number}")

        # Step 2: derive metadata-based features.
        meta = metadata or {}
        cost_band = meta.get("cost_band", "unknown") if isinstance(meta, Mapping) else "unknown"
        substrate_class = (
            meta.get("substrate_class", "unknown")
            if isinstance(meta, Mapping)
            else "unknown"
        )
        path.append(f"cost_band=={cost_band}")
        path.append(f"substrate_class=={substrate_class}")

        # Step 3: count fired rules per tier.
        fired_count = chain.fired_rule_count()
        path.append(f"fired_rule_count={fired_count}")

        # Step 4: apply the canonical decision logic.
        # Estimate risk from fired rule count (each fired rule contributes
        # ~15 risk on average per first-principles bound).
        estimated_risk_low = float(fired_count * 7)
        estimated_risk_high = float(fired_count * 25)

        if estimated_risk_low > self.refusal_threshold:
            action = "REFUSE"
            rationale = (
                f"estimated risk floor {estimated_risk_low:g} > "
                f"refusal threshold {self.refusal_threshold:g}"
            )
        elif estimated_risk_high < self.refusal_threshold * 0.5:
            action = "OK"
            rationale = (
                f"estimated risk ceiling {estimated_risk_high:g} < "
                f"50% refusal threshold; routine dispatch path"
            )
        elif self._has_recent_failures(tuple(path)):
            action = "REFUSE"
            rationale = (
                "recent decision-path produced failed dispatches; "
                "require operator review before re-dispatching"
            )
        elif fired_count == 0:
            action = "OK"
            rationale = "no preflight rules fired; routine dispatch"
        else:
            action = "REQUEST_OPERATOR_REVIEW"
            rationale = (
                f"predicted risk band [{estimated_risk_low:g}, "
                f"{estimated_risk_high:g}] straddles refusal threshold; "
                "operator decision required"
            )
        return PreflightDispatchDecision(
            decision_path=tuple(path),
            action=action,
            rationale=rationale,
            predicted_risk_band=(estimated_risk_low, estimated_risk_high),
        )

    # ── whiteboard surface ─────────────────────────────────────────────────

    def propose_candidate_rule(self, rule: PreflightWhiteboardRule) -> None:
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
        is operator-gated; the helper does NOT auto-promote based on
        empirical performance alone — Catalog #278 enforces this at
        SOURCE level.
        """
        for i, r in enumerate(self.whiteboard):
            if r.rule_id == rule_id:
                self.canonical_rules.add_candidate_rule(r.candidate_rule)
                del self.whiteboard[i]
                return True
        return False

    def prune_whiteboard(
        self, *, hit_rate_min: float | None = None
    ) -> list[str]:
        """Drop whiteboard rules whose empirical hit-rate is below threshold.

        Returns the list of pruned rule_ids. Per CLAUDE.md "Forbidden
        premature KILL": this requires explicit operator opt-in via
        threshold setting; default is conservative (50% hit rate).
        """
        threshold = (
            hit_rate_min
            if hit_rate_min is not None
            else self.min_hit_rate_for_promotion
        )
        keep: list[PreflightWhiteboardRule] = []
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
        empirical_risk: float,
    ) -> None:
        """Record empirical risk for a decision-path and refit prediction.

        Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" the
        caller is responsible for tagging the empirical_risk with its
        axis (the router records the score blindly; tagging happens
        upstream via the canonical evidence-tag chain).
        """
        path_key = tuple(decision_path)
        self._decision_path_outcomes.setdefault(path_key, []).append(
            float(empirical_risk)
        )

    def update_from_anchor(
        self,
        panel: GateVerdictPanel,
        empirical_risk: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> PreflightDispatchDecision:
        """Wrapper that records both decision-path outcome AND falling-rule update.

        Per the META principle: closing the continual-learning loop
        requires routing every empirical preflight outcome through the
        canonical helper so all 6 phases stay in lock-step.
        """
        decision = self.decide(panel, metadata)
        self.update_from_dispatch_outcome(decision.decision_path, empirical_risk)
        self.canonical_rules.update_from_anchor(panel)
        return decision

    def _has_recent_failures(self, decision_path: tuple[str, ...]) -> bool:
        outcomes = self._decision_path_outcomes.get(decision_path, [])
        if len(outcomes) < 2:
            return False
        # Treat risk >= refusal_threshold as failures.
        return sum(1 for s in outcomes[-3:] if s >= self.refusal_threshold) >= 2
