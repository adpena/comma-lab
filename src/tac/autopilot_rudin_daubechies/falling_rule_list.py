# SPDX-License-Identifier: MIT
"""Falling-rule-list ranking surface — Rudin's canonical interpretable form.

Per Wang & Rudin 2015 "Falling Rule Lists": a rule list whose rules are
ordered by support such that the FIRST matching rule wins. The "falling"
property: rules earlier in the list have HIGHER predicted-band priority;
rules at the bottom catch the residual.

Each rule has the shape::

    if predicate(panel) then predicted_score in [a, b]
        else continue to next rule

The predicate references Taylor proxy values + SubstrateContract fields per
the META layer (Catalog #241/#242). The rule chain IS the documentation:
the operator reads the conjunction that produced the ranking decision.

Continual learning per operator directive 2026-05-15: rules earn or lose
their place via empirical performance. :meth:`FallingRuleList.add_candidate_rule`
appends a low-priority rule; :meth:`prune_ineffective_rule` drops rules whose
empirical hit-rate is below threshold OR whose predicted-band missed the
empirical anchor by more than tolerance.

Per CLAUDE.md "Council conduct — non-conservative bias" the falling-rule
mechanism is the canonical operator-facing transparency layer; black-box
rankers are the anti-pattern.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .slim_ranker import ProxyPanel


# Canonical predicate tokens (operator-readable).
_PREDICATE_OP_REGEX = re.compile(
    r"^\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_.]*)\s*"
    r"(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<rhs>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PredicateRef:
    """One operator-readable predicate.

    The predicate string is parsed into ``(lhs, op, rhs)`` so the rule chain
    serializes deterministically. Supported operators: ``< <= > >= == !=``.
    LHS may reference any :class:`ProxyPanel` attribute or any nested key
    of the candidate metadata mapping (dot-separated for nesting).
    """

    expression: str
    lhs: str
    op: str
    rhs: float

    @classmethod
    def parse(cls, expression: str) -> "PredicateRef":
        m = _PREDICATE_OP_REGEX.match(expression)
        if not m:
            raise ValueError(
                f"unparseable predicate {expression!r}; expected '<lhs> <op> <rhs>'"
            )
        return cls(
            expression=expression.strip(),
            lhs=m.group("lhs").strip(),
            op=m.group("op").strip(),
            rhs=float(m.group("rhs")),
        )

    def evaluate(
        self,
        panel: ProxyPanel,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Evaluate the predicate against panel and metadata.

        Returns False if the LHS resolves to None / missing (per Rudin's
        canonical falling-rule semantics: an unmeasurable predicate cannot
        fire).
        """
        value = self._resolve_lhs(panel, metadata)
        if value is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        if self.op == "<":
            return v < self.rhs
        if self.op == "<=":
            return v <= self.rhs
        if self.op == ">":
            return v > self.rhs
        if self.op == ">=":
            return v >= self.rhs
        if self.op == "==":
            return v == self.rhs
        if self.op == "!=":
            return v != self.rhs
        raise ValueError(f"unknown operator {self.op!r}")

    def _resolve_lhs(
        self,
        panel: ProxyPanel,
        metadata: Mapping[str, Any] | None,
    ) -> Any:
        # Try ProxyPanel first.
        if hasattr(panel, self.lhs):
            return getattr(panel, self.lhs)
        # Then try metadata via dotted path.
        if metadata is None:
            return None
        cur: Any = metadata
        for part in self.lhs.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur


@dataclass(frozen=True)
class FallingRule:
    """One typed rule in a falling-rule list.

    All predicates must hold for the rule to fire (conjunction). On firing,
    the rule emits the (low, high) predicted-score band. Lower bands at
    EARLIER positions per Rudin's "falling" discipline.
    """

    name: str
    predicates: tuple[PredicateRef, ...]
    predicted_score_low: float
    predicted_score_high: float
    rule_id: str = ""

    def __post_init__(self) -> None:
        if self.predicted_score_low > self.predicted_score_high:
            raise ValueError(
                f"rule {self.name!r}: low {self.predicted_score_low} > "
                f"high {self.predicted_score_high}"
            )

    def fires_on(
        self,
        panel: ProxyPanel,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        return all(p.evaluate(panel, metadata) for p in self.predicates)

    def predicted_band(self) -> tuple[float, float]:
        return (self.predicted_score_low, self.predicted_score_high)


@dataclass
class RuleChain:
    """Result of evaluating a falling-rule list against one panel."""

    rule_name: str
    rule_id: str
    predicted_score_low: float
    predicted_score_high: float
    fired_predicates: tuple[str, ...]
    rule_position: int  # 0-based index in the list

    def predicted_band(self) -> tuple[float, float]:
        return (self.predicted_score_low, self.predicted_score_high)

    def explain(self) -> str:
        """Operator-readable rule chain explanation."""
        preds = " AND ".join(self.fired_predicates) or "(no predicates)"
        return (
            f"rule[{self.rule_position}] {self.rule_name!r} fired: "
            f"{preds} -> predicted_score in "
            f"[{self.predicted_score_low:g}, {self.predicted_score_high:g}]"
        )


@dataclass
class FallingRuleList:
    """Sequence of falling rules with continual-learning hooks.

    Per Rudin "Falling Rule Lists" canonical algorithm: rules ordered by
    decreasing predicted-band priority; first matching rule wins. The
    bottom rule is the catch-all (the "default band").

    Continual learning surface:

    * :meth:`add_candidate_rule` appends a low-priority rule (last position
      before the default catch-all).
    * :meth:`prune_ineffective_rule` removes rules whose empirical hit-rate
      < threshold OR whose predicted band missed empirical anchors.
    * :meth:`update_from_anchor` records empirical performance per rule for
      pruning decisions.
    """

    rules: list[FallingRule] = field(default_factory=list)
    default_band_low: float = 0.10
    default_band_high: float = 0.30
    _hit_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _miss_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _empirical_score_sum: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _empirical_score_n: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def evaluate(
        self,
        panel: ProxyPanel,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuleChain:
        """Return the first-firing rule's chain (or the catch-all band)."""
        for i, rule in enumerate(self.rules):
            if rule.fires_on(panel, metadata):
                return RuleChain(
                    rule_name=rule.name,
                    rule_id=rule.rule_id or rule.name,
                    predicted_score_low=rule.predicted_score_low,
                    predicted_score_high=rule.predicted_score_high,
                    fired_predicates=tuple(p.expression for p in rule.predicates),
                    rule_position=i,
                )
        return RuleChain(
            rule_name="<catch-all default>",
            rule_id="__default__",
            predicted_score_low=self.default_band_low,
            predicted_score_high=self.default_band_high,
            fired_predicates=(),
            rule_position=len(self.rules),
        )

    # ── continual-learning surface ─────────────────────────────────────────

    def add_candidate_rule(self, rule: FallingRule) -> None:
        """Append a candidate rule at the END (low priority before catch-all).

        Per Rudin's falling discipline: new rules earn promotion via empirical
        performance. They start at the bottom; if their hit-rate is high AND
        their predicted band tracks empirical anchors, they may be promoted
        upward by an explicit operator decision (NOT auto-promotion — operator
        retains the rule-list ordering authority per the council-grade
        decision rule in CLAUDE.md).
        """
        if rule.rule_id and any(r.rule_id == rule.rule_id for r in self.rules):
            raise ValueError(f"rule_id {rule.rule_id!r} already in list")
        self.rules.append(rule)

    def prune_ineffective_rule(
        self,
        rule_id: str,
        *,
        hit_rate_min: float = 0.05,
        empirical_band_tolerance: float = 0.05,
    ) -> bool:
        """Drop a rule whose empirical performance is below threshold.

        Returns True iff the rule was removed.
        """
        rule_idx: int | None = None
        for i, r in enumerate(self.rules):
            if (r.rule_id or r.name) == rule_id:
                rule_idx = i
                break
        if rule_idx is None:
            return False
        hits = self._hit_count.get(rule_id, 0)
        misses = self._miss_count.get(rule_id, 0)
        total = hits + misses
        if total == 0:
            return False  # no evidence yet
        hit_rate = hits / total if total > 0 else 0.0
        n = self._empirical_score_n.get(rule_id, 0)
        mean_emp = (
            self._empirical_score_sum.get(rule_id, 0.0) / n if n > 0 else None
        )
        rule = self.rules[rule_idx]
        out_of_band = False
        if mean_emp is not None:
            low, high = rule.predicted_band()
            if (
                mean_emp < low - empirical_band_tolerance
                or mean_emp > high + empirical_band_tolerance
            ):
                out_of_band = True
        if hit_rate < hit_rate_min or out_of_band:
            del self.rules[rule_idx]
            return True
        return False

    def update_from_anchor(
        self,
        observed_score: float,
        panel: ProxyPanel,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuleChain:
        """Record empirical performance for the rule that fires on this panel.

        Returns the rule chain that fired so the caller (autopilot) can log
        the per-anchor empirical-vs-predicted comparison.
        """
        chain = self.evaluate(panel, metadata)
        rid = chain.rule_id
        # Hit if observed score within band; miss otherwise.
        within = chain.predicted_score_low <= observed_score <= chain.predicted_score_high
        if within:
            self._hit_count[rid] = self._hit_count.get(rid, 0) + 1
        else:
            self._miss_count[rid] = self._miss_count.get(rid, 0) + 1
        self._empirical_score_sum[rid] = (
            self._empirical_score_sum.get(rid, 0.0) + float(observed_score)
        )
        self._empirical_score_n[rid] = self._empirical_score_n.get(rid, 0) + 1
        return chain

    def hit_rate(self, rule_id: str) -> float | None:
        hits = self._hit_count.get(rule_id, 0)
        misses = self._miss_count.get(rule_id, 0)
        total = hits + misses
        if total == 0:
            return None
        return hits / total

    def empirical_mean(self, rule_id: str) -> float | None:
        n = self._empirical_score_n.get(rule_id, 0)
        if n == 0:
            return None
        return self._empirical_score_sum.get(rule_id, 0.0) / n
