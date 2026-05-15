# SPDX-License-Identifier: MIT
"""Falling-rule-list evaluator over preflight catalog gates.

Per Wang & Rudin 2015 "Falling Rule Lists": a rule list whose rules are
ordered by support such that the FIRST matching rule wins. The "falling"
property: rules earlier in the list have HIGHER priority; rules at the
bottom catch the residual.

Each rule has the shape::

    if <gate_predicate> then <verdict> with rationale "<auditable string>"

Rules are sorted by HIT-RATE — gates that fire frequently in the empirical
preflight history sort to the FRONT. Operator reads rule chain, not opaque
pass/fail.

Each existing catalog gate becomes a rule. The rule chain IS the
documentation: the operator reads the conjunction that produced the
preflight verdict.

Continual learning per operator directive 2026-05-15: rules earn or lose
their position via empirical performance. :meth:`PreflightFallingRuleEvaluator.add_candidate_rule`
appends a low-priority rule; :meth:`prune_ineffective_rule` drops rules
whose empirical hit-rate is below threshold OR whose rationale rolled.

Self-protection: Catalog #274 enforces canonical falling-rule discipline
at SOURCE level — the fine-rule-overrides-coarse-gate antipattern inverts
the canonical "first-match-wins; higher-priority rules earlier" rule.

[verified-against: Wang & Rudin 2015 §3 + canonical autopilot sister
``tac.autopilot_rudin_daubechies.falling_rule_list.FallingRuleList``]
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .slim_risk_scorer import GateVerdictPanel


@dataclass(frozen=True)
class GateRuleVerdict:
    """One rule's verdict for a panel."""

    gate_number: int
    rule_fired: bool
    verdict: str  # "VIOLATED" | "PASSED" | "WAIVED" | "EXEMPT" | "NOT_RUN"
    rationale: str
    recommended_fix: str = ""

    def explain(self) -> str:
        if self.rule_fired:
            fix_line = (
                f" -> recommended fix: {self.recommended_fix}"
                if self.recommended_fix
                else ""
            )
            return (
                f"gate_{self.gate_number} FIRED verdict={self.verdict} "
                f"(rationale: {self.rationale}){fix_line}"
            )
        return f"gate_{self.gate_number} did not fire (verdict={self.verdict})"


@dataclass
class PreflightFallingRule:
    """One operator-readable falling rule wrapping a catalog gate.

    Per the META principle landed in this package: every preflight gate's
    failure message MUST cite the rule chain that fired AND the recommended
    fix as a rule chain. Comment-only failure descriptions are FORBIDDEN
    per CLAUDE.md 'Comment-only contracts are FORBIDDEN' extended to gate
    output. The catalog table itself is a falling-rule list with hit-rate
    sorting.
    """

    gate_number: int
    gate_name: str
    rationale_template: str
    recommended_fix: str
    empirical_hit_count: int = 0
    empirical_miss_count: int = 0

    @property
    def hit_rate(self) -> float | None:
        total = self.empirical_hit_count + self.empirical_miss_count
        if total == 0:
            return None
        return self.empirical_hit_count / total

    def evaluate(self, panel: GateVerdictPanel) -> GateRuleVerdict:
        """Return the rule verdict for ``panel``."""
        verdict_label = str(panel.verdicts.get(str(self.gate_number), "NOT_RUN")).upper()
        rule_fired = verdict_label == "VIOLATED"
        rationale = (
            self.rationale_template.format(gate_number=self.gate_number)
            if rule_fired
            else f"gate_{self.gate_number}={verdict_label}"
        )
        return GateRuleVerdict(
            gate_number=self.gate_number,
            rule_fired=rule_fired,
            verdict=verdict_label,
            rationale=rationale,
            recommended_fix=self.recommended_fix if rule_fired else "",
        )


@dataclass
class PreflightRuleChain:
    """The chain of rule verdicts that produced the overall preflight decision."""

    rules: tuple[GateRuleVerdict, ...]
    first_fired_index: int | None = None  # first-match-wins index

    def fired_rule_count(self) -> int:
        return sum(1 for r in self.rules if r.rule_fired)

    def explain(self) -> str:
        if not self.rules:
            return "preflight rule chain: (empty)"
        lines = []
        for i, r in enumerate(self.rules):
            marker = "→" if i == self.first_fired_index else " "
            lines.append(f"  {marker} [{i}] {r.explain()}")
        first_label = (
            f"first-match index={self.first_fired_index}"
            if self.first_fired_index is not None
            else "no rule fired (clean preflight)"
        )
        return f"preflight rule chain ({first_label}):\n" + "\n".join(lines)


class PreflightFallingRuleEvaluator:
    """Falling-rule list evaluator over preflight catalog gates.

    Per Rudin's interpretability principle: rules are sorted by empirical
    hit-rate (most-frequent-firing rules at the FRONT of the list). The
    operator reads the rule chain top-to-bottom; the FIRST rule that fires
    is the canonical verdict.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    rules with empirical hit-rate below the prune threshold are NOT killed
    automatically; they are demoted to the back of the list. Explicit
    operator opt-in to :meth:`prune_ineffective_rule` is required for
    removal.

    Continual learning: every preflight outcome flows through
    :meth:`update_from_anchor` which records hit/miss stats per rule and
    re-sorts the list by empirical hit-rate.

    [verified-against: Wang & Rudin 2015 §3.2 + autopilot sister
    ``tac.autopilot_rudin_daubechies.falling_rule_list.FallingRuleList``]
    """

    def __init__(
        self,
        *,
        rules: Sequence[PreflightFallingRule] | None = None,
        prune_hit_rate_threshold: float = 0.05,
    ) -> None:
        self._rules: list[PreflightFallingRule] = list(rules) if rules else []
        self.prune_hit_rate_threshold = float(prune_hit_rate_threshold)
        self._anchors_seen: int = 0

    @property
    def rules(self) -> tuple[PreflightFallingRule, ...]:
        return tuple(self._rules)

    @property
    def n_anchors(self) -> int:
        return self._anchors_seen

    def add_candidate_rule(self, rule: PreflightFallingRule) -> None:
        """Append a new rule at the BACK of the list (lowest priority).

        Per the falling-rule discipline: rules earn their place by empirical
        performance. New rules start at the back; high-hit-rate rules
        promote up over time via :meth:`_resort_by_hit_rate`.
        """
        if any(r.gate_number == rule.gate_number for r in self._rules):
            raise ValueError(
                f"rule for gate_{rule.gate_number} already registered"
            )
        self._rules.append(rule)

    def prune_ineffective_rule(self, gate_number: int) -> bool:
        """Drop a rule whose empirical hit-rate is below threshold.

        Returns True iff dropped. Per CLAUDE.md "Forbidden premature KILL":
        explicit operator opt-in is required; this is NOT an auto-prune
        helper.
        """
        for i, r in enumerate(self._rules):
            if r.gate_number == gate_number:
                hr = r.hit_rate
                if hr is None or hr >= self.prune_hit_rate_threshold:
                    return False
                del self._rules[i]
                return True
        return False

    def evaluate(self, panel: GateVerdictPanel) -> PreflightRuleChain:
        """Evaluate all rules against ``panel`` and return the chain.

        First-match-wins: ``first_fired_index`` records the first rule
        that fired (canonical preflight verdict). All other fired rules
        remain visible in the chain for the operator-facing explain().
        """
        verdicts: list[GateRuleVerdict] = []
        first_fired: int | None = None
        for i, rule in enumerate(self._rules):
            v = rule.evaluate(panel)
            verdicts.append(v)
            if v.rule_fired and first_fired is None:
                first_fired = i
        return PreflightRuleChain(
            rules=tuple(verdicts),
            first_fired_index=first_fired,
        )

    def update_from_anchor(
        self,
        panel: GateVerdictPanel,
        observed_violated_gate_numbers: Sequence[int] | None = None,
    ) -> PreflightRuleChain:
        """Record hit/miss stats per rule and refit sort order.

        ``observed_violated_gate_numbers`` defaults to ``panel.violated_gate_numbers()``
        — the SAME set the panel itself records. The optional override
        exists for re-classifying a violation as WAIVED-after-the-fact via
        an explicit operator review.
        """
        chain = self.evaluate(panel)
        observed = set(
            observed_violated_gate_numbers
            if observed_violated_gate_numbers is not None
            else panel.violated_gate_numbers()
        )
        for rule in self._rules:
            if rule.gate_number in observed:
                rule.empirical_hit_count += 1
            else:
                rule.empirical_miss_count += 1
        self._anchors_seen += 1
        self._resort_by_hit_rate()
        return chain

    def _resort_by_hit_rate(self) -> None:
        """Sort rules by empirical hit-rate descending (highest first).

        Rules with no data are sorted to the BACK (per Wang & Rudin 2015
        "Falling Rule Lists" §3.3: unobserved rules earn their place by
        empirical evidence).
        """
        def _sort_key(rule: PreflightFallingRule) -> tuple[float, int]:
            hr = rule.hit_rate
            # Rules with no data sort to back via -1.0
            sort_hr = hr if hr is not None else -1.0
            return (-sort_hr, rule.gate_number)
        self._rules.sort(key=_sort_key)
