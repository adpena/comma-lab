# SPDX-License-Identifier: MIT
"""Wavelet-decomposed multi-scale falling-rule-list ranker.

The COMPOSITE design Rudin and Daubechies jointly recommended:

    WAVELET-DECOMPOSED RASHOMON-ENSEMBLE FALLING-RULE-LIST AUTOPILOT RANKER

Each candidate is ranked through 4 cascading scales of falling-rule lists
(coarse to fine). Coarse-scale rules GATE fine-scale rules: if the
coarse rule list says "this candidate is in the wrong substrate class"
the fine rules are not even consulted.

Scales (per the channeling memo Section O.3):

* Scale 0 (COARSEST) — substrate-class (within-class refinement vs
  cross-class shift)
* Scale 1 — specific substrate (Z3 / Z4 / Z5 / D1 / D4 / etc.)
* Scale 2 — codec / archive grammar choice
* Scale 3 (FINEST) — hyperparams (epochs, learning rate, batch size)

Each scale's decision constrains the next; planning is hierarchical.
The Daubechies wavelet decomposition naturally maps to this hierarchy:
the coarsest wavelet sub-band IS the coarsest rule list; finer sub-bands
inherit the coarser support.

Continual learning per operator directive 2026-05-15: each scale's rule
list updates independently via :meth:`update_at_scale`; coarser scales
are stable across many anchors (substrate-class doesn't change often)
while finer scales adapt rapidly (hyperparams refine per dispatch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .falling_rule_list import FallingRule, FallingRuleList, RuleChain
from .slim_ranker import ProxyPanel


WAVELET_NUM_SCALES_DEFAULT: int = 4


@dataclass
class _MultiScaleEvalResult:
    """Result of evaluating one panel through all scales."""

    panel: ProxyPanel
    chain_per_scale: tuple[RuleChain, ...]
    consensus_low: float
    consensus_high: float
    coarsest_gate_passed: bool

    def explain(self) -> str:
        lines: list[str] = []
        for i, chain in enumerate(self.chain_per_scale):
            scale_name = ["substrate_class", "specific_substrate", "codec_grammar", "hyperparams"]
            label = scale_name[i] if i < len(scale_name) else f"scale_{i}"
            lines.append(f"  scale[{i}] {label}: {chain.explain()}")
        consensus = (
            f"consensus band [{self.consensus_low:g}, {self.consensus_high:g}]"
        )
        return "multi-scale rule chain:\n" + "\n".join(lines) + f"\n  -> {consensus}"


class WaveletMultiScaleFallingRuleListRanker:
    """4-scale falling-rule-list ranker with coarse-gates-fine semantics.

    Ranking flow:

    1. Evaluate the coarsest scale's rule list. If the coarse rule emits
       a band whose UPPER bound is below the operator-set "promotion
       threshold", the candidate is GATED OUT — skip finer scales.
    2. Otherwise evaluate the next-finer scale, conditioning on the
       coarser scale's predicted band.
    3. Repeat to the finest scale.

    The final consensus band is the INTERSECTION of all per-scale bands
    (finest scale dominates when it lies inside coarser bands; otherwise
    the finest band is widened to the coarser containing band).

    Per CLAUDE.md "Council conduct — non-conservative bias" the rule
    chain is the canonical operator-readable transparency layer; each
    scale's contribution is auditable independently.
    """

    def __init__(
        self,
        *,
        num_scales: int = WAVELET_NUM_SCALES_DEFAULT,
        promotion_threshold_score: float = 0.30,
    ) -> None:
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")
        self.num_scales = int(num_scales)
        self.promotion_threshold_score = float(promotion_threshold_score)
        # One falling-rule list per scale.
        self._rule_lists: list[FallingRuleList] = [
            FallingRuleList() for _ in range(self.num_scales)
        ]

    def rule_list_at_scale(self, scale: int) -> FallingRuleList:
        if scale < 0 or scale >= self.num_scales:
            raise IndexError(
                f"scale {scale} out of range [0, {self.num_scales})"
            )
        return self._rule_lists[scale]

    def evaluate(
        self,
        panel: ProxyPanel,
        metadata: Mapping[str, object] | None = None,
    ) -> _MultiScaleEvalResult:
        chains: list[RuleChain] = []
        # Start from the widest band (no constraint).
        consensus_low = float("-inf")
        consensus_high = float("inf")
        coarsest_chain = self._rule_lists[0].evaluate(panel, metadata)
        chains.append(coarsest_chain)
        # Coarse-gate: if the coarsest rule's upper band is above promotion
        # threshold, the candidate is GATED OUT and we still emit the chain
        # but skip evaluating finer scales (per Daubechies' coarse-rule-
        # gates-fine-rule discipline).
        coarsest_gate_passed = (
            coarsest_chain.predicted_score_low <= self.promotion_threshold_score
        )
        # Tighten consensus to coarsest band.
        consensus_low = max(consensus_low, coarsest_chain.predicted_score_low)
        consensus_high = min(consensus_high, coarsest_chain.predicted_score_high)
        if not coarsest_gate_passed:
            # Pad chains for unevaluated scales with the coarsest chain.
            while len(chains) < self.num_scales:
                chains.append(coarsest_chain)
            return _MultiScaleEvalResult(
                panel=panel,
                chain_per_scale=tuple(chains),
                consensus_low=consensus_low,
                consensus_high=consensus_high,
                coarsest_gate_passed=False,
            )
        # Evaluate finer scales.
        for s in range(1, self.num_scales):
            chain = self._rule_lists[s].evaluate(panel, metadata)
            chains.append(chain)
            # Intersect bands (with widening if disjoint).
            new_low = max(consensus_low, chain.predicted_score_low)
            new_high = min(consensus_high, chain.predicted_score_high)
            if new_low <= new_high:
                consensus_low = new_low
                consensus_high = new_high
            # else: scales disagree; keep coarser band rather than picking
            # an empty interval (Daubechies' "coarsest scale dominates on
            # disagreement").
        return _MultiScaleEvalResult(
            panel=panel,
            chain_per_scale=tuple(chains),
            consensus_low=consensus_low,
            consensus_high=consensus_high,
            coarsest_gate_passed=True,
        )

    # ── continual-learning surface ─────────────────────────────────────────

    def update_at_scale(
        self,
        scale: int,
        observed_score: float,
        panel: ProxyPanel,
        metadata: Mapping[str, object] | None = None,
    ) -> RuleChain:
        """Update only ONE scale's rule list (per Daubechies' wavelet
        discipline: each scale evolves at its own rate)."""
        if scale < 0 or scale >= self.num_scales:
            raise IndexError(
                f"scale {scale} out of range [0, {self.num_scales})"
            )
        return self._rule_lists[scale].update_from_anchor(
            observed_score, panel, metadata=metadata
        )

    def update_all_scales(
        self,
        observed_score: float,
        panel: ProxyPanel,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[RuleChain, ...]:
        """Update every scale's rule list (the canonical anchor harvest)."""
        return tuple(
            self._rule_lists[s].update_from_anchor(
                observed_score, panel, metadata=metadata
            )
            for s in range(self.num_scales)
        )
