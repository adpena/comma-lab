# SPDX-License-Identifier: MIT
"""Rudin-Daubechies autopilot ranker package.

Implements the WAVELET-DECOMPOSED RASHOMON-ENSEMBLE FALLING-RULE-LIST AUTOPILOT
RANKER per the channeling memo
``feedback_rudin_daubechies_recommendations_for_completing_cathedral_autopilot_nervous_system_20260515``.

Six phases ship independently and compose:

* Phase 1 — :mod:`.slim_ranker` — Sparse Linear Integer Model over Taylor proxies
* Phase 2 — :mod:`.falling_rule_list` — interpretable rule chain ranking surface
* Phase 3 — :mod:`.rashomon_ensemble` — K=8 near-optimal ensemble + disagreement queue
* Phase 4 — :mod:`.compressive_landscape` — L1 reconstruction from few anchors
* Phase 5 — :mod:`.wavelet_multi_scale_ranker` — coarse-gates-fine multi-scale rules
* Phase 6 — :mod:`.gosdt_dispatcher` — sparse decision tree + whiteboard

The binding contract per operator directive 2026-05-15: **continual learning is
the goal**. Every helper exposes ``update_from_anchor(...)`` so each empirical
anchor (CUDA / CPU / GHA Linux x86_64) closes the loop:

    harvest -> call_id_ledger -> continual_learning posterior
        -> cost_band_calibration -> SLIM/Rashomon/Wavelet/GOSDT update
        -> autopilot re-rank -> next dispatch.

Per CLAUDE.md "Subagent coherence-by-default" the package COMPOSES with the
existing canonical infrastructure (``tac.cost_band_calibration``,
``tac.continual_learning``, ``tac.sensitivity_map``,
``tac.substrate_registry``) rather than duplicating it.

Per CLAUDE.md "Apples-to-apples evidence discipline" every prediction emitted
by this package carries an explicit axis label and confidence tag.

Per CLAUDE.md "Council conduct" the design is non-conservative: a
near-optimal interpretable ranker is the canonical operator-facing transparency
layer, not a "safe" black box.
"""
from __future__ import annotations

from .compressive_landscape import (
    CompressiveSensingLandscapeRecovery,
    LandscapeCell,
)
from .compressive_sensing_lattice_recovery import (
    BayesianSequentialKSelector,
    CoherenceMinimizingSelector,
    DaubechiesDb4LatticeBasis,
    FrontierPursuitClass,
    LatticePhaseTransitionMonitor,
    SparseSignalPosterior,
    SubstrateLatticeNode,
    SubstrateLatticeRecovery,
    TreeStructuredSparsityPrior,
    classify_predicted_band,
    compute_pairwise_coherence,
    diff_sparse_signal_posteriors,
)
from .falling_rule_list import (
    FallingRule,
    FallingRuleList,
    PredicateRef,
    RuleChain,
)
from .gosdt_dispatcher import (
    DispatchDecision,
    GOSDTDispatcher,
    WhiteboardRule,
)
from .rashomon_ensemble import (
    DEFAULT_RASHOMON_ENSEMBLE_SIZE,
    RashomonEnsembleRanker,
    RashomonMember,
)
from .slim_ranker import (
    DEFAULT_INTEGER_COEFFICIENT_BOUND,
    DEFAULT_SPARSITY_TARGET,
    ProxyPanel,
    SLIMCoefficient,
    SLIMRanker,
    SLIMTrainingError,
    explain_slim_prediction,
)
from .wavelet_multi_scale_ranker import (
    WAVELET_NUM_SCALES_DEFAULT,
    WaveletMultiScaleFallingRuleListRanker,
)

__all__ = [
    # Phase 1
    "DEFAULT_INTEGER_COEFFICIENT_BOUND",
    # Phase 3
    "DEFAULT_RASHOMON_ENSEMBLE_SIZE",
    "DEFAULT_SPARSITY_TARGET",
    # Phase 5
    "WAVELET_NUM_SCALES_DEFAULT",
    # Compressive-sensing lattice recovery (operator approval 2026-05-16):
    "BayesianSequentialKSelector",
    "CoherenceMinimizingSelector",
    # Phase 4
    "CompressiveSensingLandscapeRecovery",
    "DaubechiesDb4LatticeBasis",
    "DispatchDecision",
    # Phase 2
    "FallingRule",
    "FallingRuleList",
    "FrontierPursuitClass",
    # Phase 6
    "GOSDTDispatcher",
    "LandscapeCell",
    "LatticePhaseTransitionMonitor",
    "PredicateRef",
    "ProxyPanel",
    "RashomonEnsembleRanker",
    "RashomonMember",
    "RuleChain",
    "SLIMCoefficient",
    "SLIMRanker",
    "SLIMTrainingError",
    "SparseSignalPosterior",
    "SubstrateLatticeNode",
    "SubstrateLatticeRecovery",
    "TreeStructuredSparsityPrior",
    "WaveletMultiScaleFallingRuleListRanker",
    "WhiteboardRule",
    "classify_predicted_band",
    "compute_pairwise_coherence",
    "diff_sparse_signal_posteriors",
    "explain_slim_prediction",
]
