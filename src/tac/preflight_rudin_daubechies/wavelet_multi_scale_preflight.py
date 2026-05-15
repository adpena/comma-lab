# SPDX-License-Identifier: MIT
"""Wavelet-decomposed multi-scale preflight ranker.

Per Daubechies 1988 "Orthonormal bases of compactly supported wavelets":
hierarchical ranking decomposes the preflight surface into nested scales
(coarse → fine → finest), where coarse-scale rules GATE fine-scale rules.

For preflight: each catalog gate is classified into ONE of 4 scales:

* Scale 0 (COARSE): file existence / schema compliance
  Examples: Catalog #11 (remote scripts NVDEC probe), #93 (build_manifest
  custody), #109 (intake clones pristine), #146 (Phase 1 trainer runtime).
* Scale 1 (MID): integration contract / wire-up
  Examples: Catalog #117 (subagent commit serializer), #125 (subagent
  landing has solver wire-in), #151 (operator wrapper threads trainer
  flags), #167 (smoke before full pattern).
* Scale 2 (FINE): byte-mutation / semantic correctness
  Examples: Catalog #1 (no MPS fallback default), #5 (no eval_roundtrip
  False), #14 (loader format safety), #205 (inflate device fork).
* Scale 3 (FINEST): per-substrate distinguishing-feature contract
  Examples: Catalog #164 (substrate score-aware loss preprocess), #170
  (substrate min vram), #220 (substrate L1 scaffold byte addition).

Coarse-scale failure SHORT-CIRCUITS finer scale evaluation: if a file
doesn't exist at scale 0, the integration contract at scale 1 cannot
possibly pass. The Daubechies wavelet decomposition naturally maps to
this hierarchy: the coarsest sub-band IS the file-existence layer; finer
sub-bands inherit support from the coarser ones.

Continual learning per operator directive 2026-05-15: each scale's gate
classification updates independently via :meth:`reclassify_gate_to_scale`;
coarser scales are stable across many preflight cycles (file existence
rarely changes class) while finer scales adapt rapidly.

Self-protection: Catalog #277 enforces canonical hierarchical-gating
discipline at SOURCE level — bypassing the coarse-gates-fine semantics
defeats the wavelet hierarchical-planning discipline.

[verified-against: Daubechies 1988 §3 + Mallat 1989 "A theory for
multiresolution signal decomposition" + autopilot sister
``tac.autopilot_rudin_daubechies.wavelet_multi_scale_ranker.WaveletMultiScaleFallingRuleListRanker``]
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .falling_rule_evaluator import (
    PreflightFallingRule,
    PreflightFallingRuleEvaluator,
    PreflightRuleChain,
)

if TYPE_CHECKING:
    from .slim_risk_scorer import GateVerdictPanel

PREFLIGHT_WAVELET_NUM_SCALES_DEFAULT: int = 4

# Canonical scale labels (operator-readable).
SCALE_LABELS: tuple[str, ...] = (
    "file_existence",
    "integration_contract",
    "byte_mutation",
    "per_substrate_feature",
)

# Canonical scale-by-canonical-gate-number classification. Sample 20 mapped
# below per the per-tier-meta + sister classification design memo. Future
# subagent backfill can extend via `reclassify_gate_to_scale(...)`.
_DEFAULT_SCALE_BY_GATE: Mapping[int, int] = {
    # Scale 0 (COARSE): file existence / schema compliance
    11: 0,    # remote scripts NVDEC probe
    93: 0,    # build_manifest archive custody clean
    109: 0,   # public PR intake clones pristine
    110: 0,   # recovery_metadata append-only
    146: 0,   # Phase 1 trainer runtime emits contest-compliant inflate
    # Scale 1 (MID): integration contract / wire-up
    117: 1,   # subagent commit serializer uses lock
    125: 1,   # subagent landing has solver wire-in
    126: 1,   # lane pre-registered before work starts
    151: 1,   # operator wrapper threads trainer tier required flags
    167: 1,   # substrate dispatch uses smoke-before-full pattern
    # Scale 2 (FINE): byte-mutation / semantic correctness
    1: 2,     # no MPS fallback default
    5: 2,     # no eval_roundtrip False
    7: 2,     # training scripts have auth eval
    14: 2,    # preflight loader format safety
    205: 2,   # inflate device fork canonical helper
    # Scale 3 (FINEST): per-substrate distinguishing-feature contract
    164: 3,   # substrate score-aware loss calls preprocess input
    170: 3,   # substrate recipes declare min_vram_gb floor
    220: 3,   # substrate L1 scaffold byte addition operational mechanism
    227: 3,   # substrate class promotion requires Tier C evidence
    240: 3,   # substrate contest CUDA chain complete
}


@dataclass
class PreflightScaleClassification:
    """Per-scale evaluation result for ONE preflight panel."""

    panel: GateVerdictPanel
    chain_per_scale: tuple[PreflightRuleChain, ...]
    coarsest_gate_passed: bool
    short_circuited_at_scale: int | None = None

    def explain(self) -> str:
        lines: list[str] = []
        for i, chain in enumerate(self.chain_per_scale):
            label = SCALE_LABELS[i] if i < len(SCALE_LABELS) else f"scale_{i}"
            short = "" if self.short_circuited_at_scale != i else " (SHORT-CIRCUITED)"
            lines.append(f"  scale[{i}] {label}{short}: {chain.fired_rule_count()} rules fired")
        verdict = (
            "GATED-OUT"
            if not self.coarsest_gate_passed
            else "PASSED"
            if all(c.first_fired_index is None for c in self.chain_per_scale)
            else "WARN"
        )
        return f"multi-scale preflight (verdict {verdict}):\n" + "\n".join(lines)


class WaveletMultiScalePreflightRanker:
    """4-scale falling-rule-list ranker for the preflight surface.

    Ranking flow:

    1. Evaluate the coarsest scale (file existence). If ANY file-existence
       rule fires (e.g. the trainer file doesn't exist), the candidate is
       GATED OUT — skip finer scales since they cannot possibly pass.
    2. Otherwise evaluate the next-finer scale (integration contract).
    3. Repeat to the finest scale (per-substrate feature).

    The final verdict is PASSED if no rule fires at any scale; WARN if
    fine/finest scale rules fire but coarse passes; GATED-OUT if any
    coarse-scale rule fires.

    Per CLAUDE.md "Council conduct — non-conservative bias" the rule
    chain is the canonical operator-readable transparency layer; each
    scale's contribution is auditable independently.

    [verified-against: Daubechies 1988 §3 + autopilot sister
    ``tac.autopilot_rudin_daubechies.wavelet_multi_scale_ranker.WaveletMultiScaleFallingRuleListRanker``]
    """

    def __init__(
        self,
        *,
        num_scales: int = PREFLIGHT_WAVELET_NUM_SCALES_DEFAULT,
        scale_by_gate: Mapping[int, int] | None = None,
    ) -> None:
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")
        self.num_scales = int(num_scales)
        # One falling-rule evaluator per scale.
        self._scale_evaluators: list[PreflightFallingRuleEvaluator] = [
            PreflightFallingRuleEvaluator() for _ in range(self.num_scales)
        ]
        self._scale_by_gate: dict[int, int] = dict(
            scale_by_gate if scale_by_gate is not None else _DEFAULT_SCALE_BY_GATE
        )

    def evaluator_at_scale(self, scale: int) -> PreflightFallingRuleEvaluator:
        if scale < 0 or scale >= self.num_scales:
            raise IndexError(
                f"scale {scale} out of range [0, {self.num_scales})"
            )
        return self._scale_evaluators[scale]

    def classify_gate_to_scale(self, gate_number: int) -> int:
        """Return the canonical scale # for ``gate_number`` (default scale 2)."""
        return self._scale_by_gate.get(gate_number, 2)

    def reclassify_gate_to_scale(self, gate_number: int, new_scale: int) -> None:
        """Reclassify ``gate_number`` to ``new_scale``."""
        if new_scale < 0 or new_scale >= self.num_scales:
            raise IndexError(
                f"new_scale {new_scale} out of range [0, {self.num_scales})"
            )
        self._scale_by_gate[gate_number] = new_scale

    def add_rule_at_scale(
        self, scale: int, rule: PreflightFallingRule
    ) -> None:
        """Add ``rule`` to the evaluator at ``scale``.

        The rule's ``gate_number`` is also registered in the
        scale-by-gate index so future panels route consistently.
        """
        self._scale_evaluators[scale].add_candidate_rule(rule)
        self._scale_by_gate[rule.gate_number] = scale

    def evaluate(
        self, panel: GateVerdictPanel
    ) -> PreflightScaleClassification:
        """Evaluate ``panel`` through all scales with coarse-gates-fine semantics."""
        chains: list[PreflightRuleChain] = []
        # Scale 0 (COARSEST) — file-existence
        coarsest = self._scale_evaluators[0].evaluate(panel)
        chains.append(coarsest)
        coarsest_gate_passed = coarsest.first_fired_index is None
        if not coarsest_gate_passed:
            # Pad chains for unevaluated scales with the coarsest chain (placeholder).
            while len(chains) < self.num_scales:
                chains.append(coarsest)
            return PreflightScaleClassification(
                panel=panel,
                chain_per_scale=tuple(chains),
                coarsest_gate_passed=False,
                short_circuited_at_scale=0,
            )
        # Evaluate finer scales.
        for s in range(1, self.num_scales):
            chain = self._scale_evaluators[s].evaluate(panel)
            chains.append(chain)
        return PreflightScaleClassification(
            panel=panel,
            chain_per_scale=tuple(chains),
            coarsest_gate_passed=True,
            short_circuited_at_scale=None,
        )

    # ── continual-learning surface ─────────────────────────────────────────

    def update_at_scale(
        self,
        scale: int,
        panel: GateVerdictPanel,
        observed_violated_gate_numbers: Sequence[int] | None = None,
    ) -> PreflightRuleChain:
        """Update only ONE scale's rule list.

        Per Daubechies' wavelet discipline: each scale evolves at its own
        rate; coarser scales are stable, finer scales adapt rapidly.
        """
        if scale < 0 or scale >= self.num_scales:
            raise IndexError(
                f"scale {scale} out of range [0, {self.num_scales})"
            )
        return self._scale_evaluators[scale].update_from_anchor(
            panel, observed_violated_gate_numbers
        )

    def update_all_scales(
        self,
        panel: GateVerdictPanel,
        observed_violated_gate_numbers: Sequence[int] | None = None,
    ) -> tuple[PreflightRuleChain, ...]:
        """Update every scale's rule list (the canonical anchor harvest)."""
        return tuple(
            self._scale_evaluators[s].update_from_anchor(
                panel, observed_violated_gate_numbers
            )
            for s in range(self.num_scales)
        )
