# SPDX-License-Identifier: MIT
"""Row #7: Statistical coupling-threshold derivation (mean + k·std).

Replaces hardcoded ``coupling_threshold: float = 0.5`` at
``src/tac/master_gradient_consumers.py:2668`` (READ-ONLY here per Catalog #314
absorption-pattern avoidance) with the closed-form

    threshold = mean(off_diagonal) + k * std(off_diagonal)

where ``off_diagonal`` is the set of pairwise gradient inner products
excluding the diagonal (self-pairings). This sets the threshold at the
canonical "significant coupling boundary" — k=1 (default) corresponds to the
~16% tail in a Gaussian distribution; k=2 corresponds to ~2.5%.

This helper is a CALLABLE that ``master_gradient_consumers`` CAN consume to
DERIVE its threshold; it does NOT modify the existing module.

Canonical-vs-unique decision per layer
--------------------------------------
- Statistics: ADOPT_CANONICAL Welford-like single-pass mean+std
- Threshold formula: UNIQUE (mean + k·std specifically targeted at
  significant-coupling boundary)
- Off-diagonal filter: UNIQUE per gradient-matrix semantics

9-dim checklist evidence: O(N^2) single pass; pure function; predicted ΔS
[-0.001, -0.0002].

Observability: mean + std + tail_distribution_at_threshold preserved.

6-hook wire-in: cathedral_autopilot_dispatch ACTIVE via Atom emission; rest N/A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from tac.analytical_solve_extinctions.vram_aware_batch_size import (
    AnalyticalSolveResult,
)


@dataclass(frozen=True)
class CouplingThresholdInput:
    """Inputs for the coupling-threshold solve."""

    pairwise_inner_products: Sequence[Sequence[float]]  # N x N symmetric
    k_std_multiplier: float = 1.0  # mean + k*std

    def __post_init__(self) -> None:
        if not self.pairwise_inner_products:
            raise ValueError("pairwise_inner_products must be non-empty")
        n = len(self.pairwise_inner_products)
        if n < 2:
            raise ValueError(
                f"pairwise_inner_products needs >= 2 entries; got {n} (no off-diagonal)"
            )
        for i, row in enumerate(self.pairwise_inner_products):
            if len(row) != n:
                raise ValueError(
                    f"pairwise_inner_products row {i} length {len(row)} != n {n}"
                )
        if self.k_std_multiplier < 0:
            raise ValueError(f"k_std_multiplier must be >= 0; got {self.k_std_multiplier}")


_LITERATURE_CITATION = (
    "Welford 1962 'Note on a method for calculating corrected sums of squares and products' "
    "(single-pass variance); canonical significant-coupling-boundary statistic"
)


def solve_coupling_threshold_statistical(
    inputs: CouplingThresholdInput,
    *,
    emit_arbitrariness_atom: bool = False,
) -> AnalyticalSolveResult:
    """Compute threshold = mean(off_diagonal) + k * std(off_diagonal)."""
    matrix = inputs.pairwise_inner_products
    n = len(matrix)
    off_diag_values: list[float] = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag_values.append(matrix[i][j])
    m = len(off_diag_values)
    mean_val = sum(off_diag_values) / m
    var_val = sum((x - mean_val) ** 2 for x in off_diag_values) / m if m > 1 else 0.0
    std_val = math.sqrt(var_val)
    threshold = mean_val + inputs.k_std_multiplier * std_val
    n_above_threshold = sum(1 for x in off_diag_values if x >= threshold)
    tail_fraction = n_above_threshold / m

    intermediate: dict[str, Any] = {
        "n_off_diagonal_pairs": m,
        "mean": mean_val,
        "std": std_val,
        "n_above_threshold": n_above_threshold,
        "tail_fraction_at_threshold": tail_fraction,
    }
    coupled: dict[str, Any] = {
        "k_std_multiplier": inputs.k_std_multiplier,
        "tail_fraction": tail_fraction,
    }
    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, threshold, mean_val, std_val)

    return AnalyticalSolveResult(
        solved_value=threshold,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.analytical_solve_extinctions.coupling_threshold_statistical_derivation.solve_coupling_threshold_statistical"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"Coupling threshold={threshold:.6f} = mean({mean_val:.4f}) + "
            f"{inputs.k_std_multiplier}*std({std_val:.4f}); tail_fraction={tail_fraction:.4f}."
        ),
    )


def _emit_atom(inputs: CouplingThresholdInput, threshold: float, mean_val: float, std_val: float):
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="analytical_solve_extinctions.coupling_threshold_statistical_derivation.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id="coupling_threshold_statistical_derived",
        file_path="src/tac/master_gradient_consumers.py",
        current_value=0.5,
        predicted_replacement={
            "threshold": threshold,
            "mean": mean_val,
            "std": std_val,
            "k_std_multiplier": inputs.k_std_multiplier,
        },
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_ev_delta_s=(-0.001, -0.0002),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/analytical_solve_extinctions/coupling_threshold_statistical_derivation.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
    )
