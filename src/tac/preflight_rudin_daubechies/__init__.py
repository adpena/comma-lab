# SPDX-License-Identifier: MIT
"""Rudin-Daubechies preflight composite — autopilot-mirror for ranking + routing
of expensive preflight gate operations.

Per operator approval 2026-05-15: mirror the recently-landed
``tac.autopilot_rudin_daubechies.*`` 6-phase composite (commit ``a2e479565``)
to the preflight surface as ``tac.preflight_rudin_daubechies.*``.

The autopilot mirrors to preflight because BOTH are RANKING + ROUTING decisions
about expensive operations consuming empirical evidence:

* autopilot ranks DISPATCH candidates (paid GPU jobs) by predicted score
  consuming Taylor proxies + cost-band posterior + contest-CUDA anchors
* preflight ranks GATE candidates (~270 catalog gates) by violation risk
  consuming staged-file risk vectors + per-gate hit-rate + recent failure
  evidence

Six phases ship independently and compose:

* Phase 1 — :mod:`.slim_risk_scorer` — Sparse Linear Integer Model over ~270
  catalog gate verdicts. Reuses
  :class:`tac.autopilot_rudin_daubechies.slim_ranker.SLIMCoefficient`
  integer-contract validator. [verified-against: Ustun & Rudin 2016, "Supersparse
  linear integer models for optimized medical scoring systems"]
* Phase 2 — :mod:`.falling_rule_evaluator` — falling-rule list per gate
  predicate; first-match-wins; rules never-firing get auto-pruned. Each existing
  catalog gate becomes a rule. [verified-against: Wang & Rudin 2015, "Falling
  Rule Lists"]
* Phase 3 — :mod:`.rashomon_preflight_ensemble` — K=8 near-optimal preflight
  orderings via bootstrap diversity; consensus = HIGH-CONFIDENCE verdict;
  disagreement queue = next gate to add. [verified-against: Semenova, Rudin &
  Parr 2020, "On the Existence of Simpler Machine Learning Models"]
* Phase 4 — :mod:`.compressive_coverage_estimator` — sample 8 representative
  trainer/recipe fixtures; L1-reconstruct full coverage manifest of all 270
  gates with bounded uncertainty. [verified-against: Daubechies, DeVore,
  Fornasier & Gunturk 2010, "Iteratively Reweighted Least Squares Minimization
  for Sparse Recovery"]
* Phase 5 — :mod:`.wavelet_multi_scale_preflight` — 4 scales (COARSE: file
  existence / schema compliance → MID: integration contract / wire-up → FINE:
  byte-mutation / semantic correctness → FINEST: per-substrate distinguishing-
  feature contract). Coarse GATES fine. [verified-against: Daubechies 1988,
  "Orthonormal bases of compactly supported wavelets"]
* Phase 6 — :mod:`.gosdt_dispatch_router` — sparse decision tree branching on
  (substrate-class / cost-band / Rashomon-confidence / Tier-1+2+3 verdicts)
  into auditable sparse leaves. Operator-gated whiteboard for new rules.
  [verified-against: Lin, Zhong, Hu, Hu, Rudin & Seltzer 2020, "Generalized
  and Scalable Optimal Sparse Decision Trees"]

The binding contract per operator directive 2026-05-15: **continual learning
is the goal**. Every helper exposes ``update_from_anchor(...)`` so each
preflight outcome (PASSED / VIOLATED / WAIVED / EXEMPT) closes the loop:

    staged-files -> per-gate verdicts -> SLIM/Rashomon/Wavelet/GOSDT update
        -> next preflight rank -> faster operator iteration cycle.

Per CLAUDE.md "Subagent coherence-by-default" the package COMPOSES with the
existing canonical infrastructure (``tac.preflight``, ``tac.continual_learning``,
``tac.cost_band_calibration``, ``tac.autopilot_rudin_daubechies``) rather than
duplicating it. Catalog #273-#278 enforce this self-protection.

Per CLAUDE.md "Apples-to-apples evidence discipline" every prediction emitted
by this package carries an explicit axis label (``[preflight-risk; cold-start]``
or ``[preflight-risk; n=K-anchor-posterior]``) and confidence tag.

Per CLAUDE.md "Council conduct" the design is non-conservative: a near-optimal
interpretable preflight ranker is the canonical operator-facing transparency
layer for "which gates fire most often / which gate to add next", not a
bag-of-checks black box.

Cross-ref `feedback_rudin_daubechies_autopilot_full_implementation_landed_20260515.md`
(the autopilot sister this composite mirrors) + the META principle below.

The CLAUDE.md META principle (NEW non-negotiable) anchored to this package:

    Every preflight gate's failure message MUST cite the rule chain that fired
    AND the recommended fix as a rule chain. Comment-only failure descriptions
    are FORBIDDEN per CLAUDE.md 'Comment-only contracts are FORBIDDEN' extended
    to gate output. The catalog table itself is a falling-rule list with
    hit-rate sorting.
"""
from __future__ import annotations

from .compressive_coverage_estimator import (
    CompressiveCoverageEstimator,
    CoverageCell,
)
from .falling_rule_evaluator import (
    GateRuleVerdict,
    PreflightFallingRule,
    PreflightFallingRuleEvaluator,
    PreflightRuleChain,
)
from .gosdt_dispatch_router import (
    GOSDTDispatchRouter,
    PreflightDispatchDecision,
    PreflightWhiteboardRule,
)
from .rashomon_preflight_ensemble import (
    DEFAULT_PREFLIGHT_RASHOMON_SIZE,
    PreflightRashomonEnsemble,
    PreflightRashomonMember,
)
from .slim_risk_scorer import (
    DEFAULT_PREFLIGHT_INTEGER_BOUND,
    DEFAULT_PREFLIGHT_SPARSITY_TARGET,
    GateVerdictPanel,
    PreflightSLIMRiskScorer,
    PreflightSLIMTrainingError,
    explain_preflight_risk_prediction,
    predict_dispatch_risk_score_with_rationale,
)
from .wavelet_multi_scale_preflight import (
    PREFLIGHT_WAVELET_NUM_SCALES_DEFAULT,
    PreflightScaleClassification,
    WaveletMultiScalePreflightRanker,
)

__all__ = [
    "DEFAULT_PREFLIGHT_INTEGER_BOUND",
    "DEFAULT_PREFLIGHT_RASHOMON_SIZE",
    "DEFAULT_PREFLIGHT_SPARSITY_TARGET",
    "PREFLIGHT_WAVELET_NUM_SCALES_DEFAULT",
    "CompressiveCoverageEstimator",
    "CoverageCell",
    "GOSDTDispatchRouter",
    "GateRuleVerdict",
    "GateVerdictPanel",
    "PreflightDispatchDecision",
    "PreflightFallingRule",
    "PreflightFallingRuleEvaluator",
    "PreflightRashomonEnsemble",
    "PreflightRashomonMember",
    "PreflightRuleChain",
    "PreflightSLIMRiskScorer",
    "PreflightSLIMTrainingError",
    "PreflightScaleClassification",
    "PreflightWhiteboardRule",
    "WaveletMultiScalePreflightRanker",
    "explain_preflight_risk_prediction",
    "predict_dispatch_risk_score_with_rationale",
]
