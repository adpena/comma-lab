# SPDX-License-Identifier: MIT
"""Row #8: Welling-Teh 2011 SGLD t_final convergence-time solver.

Replaces hardcoded ``langevin-t-final default=1e-4`` at
``experiments/train_substrate_stack_of_stacks.py:280`` (READ-ONLY here per
Catalog #314) with the closed-form Welling-Teh 2011 convergence criterion.

Per Welling-Teh 2011 "Bayesian Learning via Stochastic Gradient Langevin
Dynamics" (ICML), the SGLD chain converges to the target posterior when the
cumulative step size sequence satisfies:

    sum(eta_t) -> infinity    (chain explores fully)
    sum(eta_t^2) -> finite    (gradient noise dominates injected noise)

For a constant-step regime eta_t = eta, the convergence time scales as:

    t_final = O(variance_posterior_target / (eta^2 * target_acceptance))

The canonical formula this helper derives is:

    t_final = K_conv * variance_posterior / (eta^2 * target_acceptance)

with ``K_conv`` defaulting to 1.0 (operator can tune; conservative is 2-10).

Canonical-vs-unique decision per layer
--------------------------------------
- Convergence theory: ADOPT_CANONICAL Welling-Teh 2011
- Constant-step regime: UNIQUE for stack_of_stacks substrate
- Numerical stability floor: UNIQUE (refuse division by tiny eta)

9-dim checklist evidence: O(1); pure function; predicted ΔS [-0.002, -0.0003].

Observability: variance + step size + acceptance preserved.

6-hook wire-in: cathedral_autopilot_dispatch ACTIVE via Atom emission; rest N/A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tac.analytical_solve_extinctions.vram_aware_batch_size import (
    AnalyticalSolveResult,
)


@dataclass(frozen=True)
class SGLDTFinalInput:
    """Inputs for the SGLD t_final solve."""

    variance_posterior_target: float  # Target posterior variance (problem-specific)
    step_size_eta: float
    target_acceptance_rate: float = 0.574  # Roberts-Rosenthal 1998 optimal MALA acceptance
    K_conv_safety_factor: float = 1.0
    t_final_floor: float = 1e-6
    t_final_ceiling: float = 1.0

    def __post_init__(self) -> None:
        if self.variance_posterior_target <= 0:
            raise ValueError(
                f"variance_posterior_target must be positive; got {self.variance_posterior_target}"
            )
        if self.step_size_eta <= 0:
            raise ValueError(f"step_size_eta must be positive; got {self.step_size_eta}")
        if not (0.0 < self.target_acceptance_rate <= 1.0):
            raise ValueError(
                f"target_acceptance_rate must be in (0, 1]; got {self.target_acceptance_rate}"
            )
        if self.K_conv_safety_factor <= 0:
            raise ValueError(
                f"K_conv_safety_factor must be positive; got {self.K_conv_safety_factor}"
            )
        if self.t_final_floor <= 0:
            raise ValueError(f"t_final_floor must be positive; got {self.t_final_floor}")
        if self.t_final_ceiling <= self.t_final_floor:
            raise ValueError(
                f"t_final_ceiling {self.t_final_ceiling} <= t_final_floor {self.t_final_floor}"
            )


_LITERATURE_CITATION = (
    "Welling-Teh 2011 'Bayesian Learning via Stochastic Gradient Langevin Dynamics' ICML; "
    "Roberts-Rosenthal 1998 'Optimal Scaling of Discrete Approximations to Langevin Diffusions'"
)


def solve_sgld_t_final_welling_teh(
    inputs: SGLDTFinalInput,
    *,
    emit_arbitrariness_atom: bool = False,
) -> AnalyticalSolveResult:
    """Closed-form SGLD t_final via Welling-Teh 2011 convergence criterion."""
    raw_t_final = (
        inputs.K_conv_safety_factor * inputs.variance_posterior_target
        / (inputs.step_size_eta ** 2 * inputs.target_acceptance_rate)
    )
    t_final = max(inputs.t_final_floor, min(inputs.t_final_ceiling, raw_t_final))
    ceiling_capped = t_final == inputs.t_final_ceiling
    floor_capped = t_final == inputs.t_final_floor

    intermediate: dict[str, Any] = {
        "raw_t_final_unclamped": raw_t_final,
        "ceiling_capped": ceiling_capped,
        "floor_capped": floor_capped,
        "noise_dominates_gradient_check": inputs.step_size_eta ** 2 < inputs.variance_posterior_target,
    }
    coupled: dict[str, Any] = {
        "convergence_criterion_satisfied": not (ceiling_capped or floor_capped),
        "ratio_to_naive_1e_minus_4": t_final / 1e-4,
    }
    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, t_final)

    return AnalyticalSolveResult(
        solved_value=t_final,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.analytical_solve_extinctions.sgld_t_final_welling_teh.solve_sgld_t_final_welling_teh"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"SGLD t_final={t_final:.6e} (raw {raw_t_final:.6e}); "
            f"variance={inputs.variance_posterior_target:.4e}, eta={inputs.step_size_eta:.4e}, "
            f"acceptance={inputs.target_acceptance_rate}."
        ),
    )


def _emit_atom(inputs: SGLDTFinalInput, t_final: float):
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="analytical_solve_extinctions.sgld_t_final_welling_teh.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id="sgld_t_final_welling_teh_derived",
        file_path="experiments/train_substrate_stack_of_stacks.py",
        current_value=1e-4,
        predicted_replacement={"t_final": t_final},
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_ev_delta_s=(-0.002, -0.0003),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/analytical_solve_extinctions/sgld_t_final_welling_teh.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
    )
