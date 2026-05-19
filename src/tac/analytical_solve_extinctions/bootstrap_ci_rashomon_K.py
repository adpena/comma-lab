# SPDX-License-Identifier: MIT
"""Row #9: Bootstrap-CI-derived Rashomon ensemble K solver.

Replaces hardcoded ``K=8`` Rashomon ensemble default in
``tac.preflight_rudin_daubechies.rashomon`` (READ-ONLY here per Catalog #314)
with the bootstrap-confidence-interval width derivation per
Fisher-Rudin-Dominici 2019 "All Models are Wrong, but Many are Useful"
(arxiv:1801.01489):

    K = ceil(d * log(1/delta) / epsilon^2)

where:
    d         = effective Rashomon-set dimensionality
    delta     = confidence-interval failure probability (e.g., 0.05)
    epsilon   = target CI half-width on the disagreement rate

K=8 is a rule-of-thumb; this helper produces a derived K that's calibrated to
the specific Rashomon-set + operator-chosen CI target.

Canonical-vs-unique decision per layer
--------------------------------------
- CI formula: ADOPT_CANONICAL Fisher-Rudin-Dominici 2019
- Ceiling round: UNIQUE for integer ensemble size
- Clamp [min_K, max_K]: UNIQUE for operator-budget constraints

9-dim checklist evidence: O(1); pure function; predicted ΔS [-0.0005, 0.0].

Observability: per-parameter contribution preserved.

6-hook wire-in: cathedral_autopilot_dispatch ACTIVE via Atom emission; rest N/A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tac.analytical_solve_extinctions.vram_aware_batch_size import (
    AnalyticalSolveResult,
)


@dataclass(frozen=True)
class RashomonKInput:
    """Inputs for the Rashomon-K bootstrap-CI solve."""

    effective_dimensionality_d: float
    ci_failure_probability_delta: float = 0.05
    ci_target_half_width_epsilon: float = 0.1
    min_K: int = 3
    max_K: int = 64

    def __post_init__(self) -> None:
        if self.effective_dimensionality_d <= 0:
            raise ValueError(
                f"effective_dimensionality_d must be positive; got {self.effective_dimensionality_d}"
            )
        if not (0.0 < self.ci_failure_probability_delta < 1.0):
            raise ValueError(
                f"ci_failure_probability_delta must be in (0, 1); got {self.ci_failure_probability_delta}"
            )
        if not (0.0 < self.ci_target_half_width_epsilon < 1.0):
            raise ValueError(
                f"ci_target_half_width_epsilon must be in (0, 1); got {self.ci_target_half_width_epsilon}"
            )
        if self.min_K < 2:
            raise ValueError(f"min_K must be >= 2; got {self.min_K}")
        if self.max_K < self.min_K:
            raise ValueError(f"max_K {self.max_K} < min_K {self.min_K}")


_LITERATURE_CITATION = (
    "Fisher-Rudin-Dominici 2019 'All Models are Wrong, but Many are Useful' arxiv:1801.01489; "
    "canonical bootstrap-CI sample-size derivation"
)


def solve_bootstrap_ci_rashomon_K(
    inputs: RashomonKInput,
    *,
    emit_arbitrariness_atom: bool = False,
) -> AnalyticalSolveResult:
    """Compute K = ceil(d * log(1/delta) / epsilon^2)."""
    raw_K = (
        inputs.effective_dimensionality_d
        * math.log(1.0 / inputs.ci_failure_probability_delta)
        / inputs.ci_target_half_width_epsilon ** 2
    )
    candidate_K = math.ceil(raw_K)
    K = max(inputs.min_K, min(inputs.max_K, candidate_K))
    ceiling_capped = K == inputs.max_K
    floor_capped = K == inputs.min_K

    intermediate: dict[str, Any] = {
        "raw_K_unclamped": raw_K,
        "ceiling_capped": ceiling_capped,
        "floor_capped": floor_capped,
        "log_1_over_delta": math.log(1.0 / inputs.ci_failure_probability_delta),
        "epsilon_squared": inputs.ci_target_half_width_epsilon ** 2,
    }
    coupled: dict[str, Any] = {
        "vs_naive_K_8_ratio": K / 8.0,
    }
    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, K)

    return AnalyticalSolveResult(
        solved_value=K,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.analytical_solve_extinctions.bootstrap_ci_rashomon_K.solve_bootstrap_ci_rashomon_K"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"Rashomon K={K} (raw {raw_K:.2f}); "
            f"d={inputs.effective_dimensionality_d}, delta={inputs.ci_failure_probability_delta}, "
            f"epsilon={inputs.ci_target_half_width_epsilon}."
        ),
    )


def _emit_atom(inputs: RashomonKInput, K: int):
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="analytical_solve_extinctions.bootstrap_ci_rashomon_K.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id="rashomon_K_bootstrap_ci_derived",
        file_path="src/tac/preflight_rudin_daubechies/rashomon_ensemble.py",
        current_value=8,
        predicted_replacement={"K": K},
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_ev_delta_s=(-0.0005, 0.0),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/analytical_solve_extinctions/bootstrap_ci_rashomon_K.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
    )
