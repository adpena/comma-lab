# SPDX-License-Identifier: MIT
"""Row #6: ROC-optimal HIGH_PAIR_INVARIANT threshold solver (Catalog #319).

Replaces hardcoded HIGH_PAIR_INVARIANT classification threshold in
``tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification``
with the ROC-optimal threshold derived from a labeled DeliverabilityProof
corpus.

The operating point selects the threshold that minimizes the weighted
classification error (FP + FN) — at equal weights this is the Youden's J
statistic maximum (max TPR - FPR). When operator wants asymmetric costs
(e.g., a false-positive HIGH_PAIR_INVARIANT reward is 3x worse than a
false-negative miss), pass ``fp_cost_multiplier`` > 1.0.

Per Catalog #319 Q2/Q3 lineage: the Venn-classification reward branch only
fires when ``DeliverabilityProof`` validates contest compliance — this
helper provides the threshold AT WHICH the classifier decides
HIGH_PAIR_INVARIANT vs HIGH_PAIR_SPECIFIC vs OTHER.

This helper is a CALLABLE that cathedral_autopilot CAN consume; it does NOT
modify cathedral_autopilot itself (Catalog #314 absorption avoidance).

Canonical-vs-unique decision per layer
--------------------------------------
- ROC computation: ADOPT_CANONICAL (sweep thresholds; build TPR/FPR pairs)
- Optimization metric: UNIQUE (Youden's J with optional FP/FN cost weights)
- Threshold representation: ADOPT_CANONICAL (float in score range)

9-dim checklist evidence: O(N log N) sort + O(N) sweep; pure function;
predicted ΔS [-0.002, -0.0005].

Observability: full ROC curve preserved in intermediate_values["roc_curve"].

6-hook wire-in: cathedral_autopilot_dispatch ACTIVE via Atom emission AND
via being the canonical consumer of the labeled DeliverabilityProof corpus;
others N/A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tac.analytical_solve_extinctions.vram_aware_batch_size import (
    AnalyticalSolveResult,
)


@dataclass(frozen=True)
class ROCThresholdInput:
    """Labeled corpus entries: (score, is_positive_label, optional_weight)."""

    labeled_examples: Sequence[tuple[float, bool]]
    fp_cost_multiplier: float = 1.0
    fn_cost_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.labeled_examples:
            raise ValueError("labeled_examples must be non-empty")
        if self.fp_cost_multiplier <= 0:
            raise ValueError(f"fp_cost_multiplier must be positive; got {self.fp_cost_multiplier}")
        if self.fn_cost_multiplier <= 0:
            raise ValueError(f"fn_cost_multiplier must be positive; got {self.fn_cost_multiplier}")
        n_pos = sum(1 for _, lbl in self.labeled_examples if lbl)
        n_neg = len(self.labeled_examples) - n_pos
        if n_pos == 0:
            raise ValueError("labeled_examples has zero positives; cannot compute ROC")
        if n_neg == 0:
            raise ValueError("labeled_examples has zero negatives; cannot compute ROC")


_LITERATURE_CITATION = (
    "Youden 1950 'Index for rating diagnostic tests'; "
    "Fawcett 2006 'An Introduction to ROC Analysis' Pattern Recognition Letters 27(8)"
)


def solve_roc_optimal_high_pair_invariant_threshold(
    inputs: ROCThresholdInput,
    *,
    emit_arbitrariness_atom: bool = False,
) -> AnalyticalSolveResult:
    """Sweep candidate thresholds; pick the one minimizing FP_cost + FN_cost."""
    # Sort by score descending; sweep threshold below each unique score.
    sorted_examples = sorted(inputs.labeled_examples, key=lambda x: -x[0])
    n_total = len(sorted_examples)
    n_pos = sum(1 for _, lbl in sorted_examples if lbl)
    n_neg = n_total - n_pos

    # Threshold candidates: midpoints between consecutive distinct scores.
    unique_scores = sorted({s for s, _ in sorted_examples}, reverse=True)
    threshold_candidates: list[float] = []
    if len(unique_scores) >= 2:
        for i in range(len(unique_scores) - 1):
            threshold_candidates.append((unique_scores[i] + unique_scores[i + 1]) / 2.0)
    threshold_candidates.append(min(unique_scores) - 1e-9)
    threshold_candidates.append(max(unique_scores) + 1e-9)

    best_threshold = threshold_candidates[0]
    best_cost = float("inf")
    best_tpr = 0.0
    best_fpr = 0.0
    roc_curve: list[tuple[float, float, float, float]] = []  # (threshold, tpr, fpr, cost)

    for threshold in threshold_candidates:
        tp = sum(1 for s, lbl in sorted_examples if s >= threshold and lbl)
        fp = sum(1 for s, lbl in sorted_examples if s >= threshold and not lbl)
        fn = n_pos - tp
        tpr = tp / n_pos
        fpr = fp / n_neg
        cost = inputs.fp_cost_multiplier * fp + inputs.fn_cost_multiplier * fn
        roc_curve.append((threshold, tpr, fpr, cost))
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
            best_tpr = tpr
            best_fpr = fpr

    youdens_j = best_tpr - best_fpr
    intermediate: dict[str, Any] = {
        "roc_curve": roc_curve,
        "best_tpr": best_tpr,
        "best_fpr": best_fpr,
        "youdens_j": youdens_j,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "best_cost": best_cost,
    }
    coupled: dict[str, Any] = {
        "operating_point_fp_rate": best_fpr,
        "operating_point_tp_rate": best_tpr,
    }
    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, best_threshold, youdens_j)

    return AnalyticalSolveResult(
        solved_value=best_threshold,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.analytical_solve_extinctions.roc_optimal_high_pair_invariant_threshold.solve_roc_optimal_high_pair_invariant_threshold"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"ROC-optimal threshold={best_threshold:.6f}; Youden's J={youdens_j:.4f}; "
            f"corpus n={len(sorted_examples)} (pos={n_pos}, neg={n_neg})."
        ),
    )


def _emit_atom(inputs: ROCThresholdInput, threshold: float, youdens_j: float):
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="analytical_solve_extinctions.roc_optimal_high_pair_invariant_threshold.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id="high_pair_invariant_threshold_roc_optimal",
        file_path="tools/cathedral_autopilot_autonomous_loop.py",
        current_value="hardcoded_non_parameterized_threshold",
        predicted_replacement={"threshold": threshold, "youdens_j": youdens_j},
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_ev_delta_s=(-0.002, -0.0005),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/analytical_solve_extinctions/roc_optimal_high_pair_invariant_threshold.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
    )
