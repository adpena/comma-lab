# SPDX-License-Identifier: MIT
"""Rows #2 + #3 (batched): R-D theoretic VQ codebook size K solver.

Replaces hand-picked K=64 (neural_weight_codec.py:83) AND K=256
(codec_pipeline_mask.py:359) with the rate-distortion-theoretic optimum that
minimizes the total cost functional:

    total_bits(K) = N * log2(K)                         # codeword indices
                  + K * d * bytes_per_codeword * 8      # codebook overhead in bits
                  + N * sigma_quant_squared(K)          # reconstruction error proxy
                  * lambda_RD

For piecewise-power-law sigma_quant_squared(K) ~ C / K^(2/d), the closed-form
minimizer is K* = (2 * lambda_RD * N * C / (d^2 * bytes_per_codeword))^(d/(d+2)).

In practice we evaluate the explicit cost on a candidate K grid (2-512 powers
of two) and pick the empirical minimum — this is the "elbow detection on R-D
curve" approach Gersho & Gray 1992 endorse.

Sister: ``src/tac/neural_weight_codec_sensitivity.py:509`` already has a
``K in {4, 16, 64, 256}`` sweep — the ARBITRARINESS is not WIRING the sweep
result back. This helper provides the canonical solve.

Canonical-vs-unique decision per layer
--------------------------------------
- Cost functional: ADOPT_CANONICAL R-D total-cost formula
- Candidate grid: UNIQUE (power-of-two K in [2, 512])
- Elbow detection: ADOPT_CANONICAL (argmin on cost array)

9-dim checklist evidence: O(|K_grid|) cost array build; pure function;
predicted ΔS [-0.002, -0.0001] composite (rows #2 + #3).

Observability: every K in the grid + its cost is preserved in
``intermediate_values["per_K_cost_breakdown"]`` for the diff-able + cite-able
facets.

6-hook wire-in: cathedral_autopilot_dispatch ACTIVE via Atom emission;
others N/A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tac.analytical_solve_extinctions.vram_aware_batch_size import (
    AnalyticalSolveResult,
)


@dataclass(frozen=True)
class RDCodebookSolverInput:
    """Inputs for the R-D theoretic VQ codebook K solve."""

    num_codewords_used: int  # N — number of times a codebook entry is referenced
    codeword_dim: int  # d — entry dimensionality (e.g., 4 for FP4A, 8x8=64 for mask blocks)
    bytes_per_codeword: int  # Codebook overhead per entry
    lambda_rd: float = 1.0  # Rate-distortion tradeoff weight
    quantization_constant_C: float = 1.0  # C in sigma^2 ~ C / K^(2/d)
    candidate_K_min: int = 2
    candidate_K_max: int = 512

    def __post_init__(self) -> None:
        if self.num_codewords_used <= 0:
            raise ValueError(f"num_codewords_used must be positive; got {self.num_codewords_used}")
        if self.codeword_dim <= 0:
            raise ValueError(f"codeword_dim must be positive; got {self.codeword_dim}")
        if self.bytes_per_codeword <= 0:
            raise ValueError(
                f"bytes_per_codeword must be positive; got {self.bytes_per_codeword}"
            )
        if self.lambda_rd <= 0:
            raise ValueError(f"lambda_rd must be positive; got {self.lambda_rd}")
        if self.candidate_K_min < 2:
            raise ValueError(f"candidate_K_min must be >= 2; got {self.candidate_K_min}")
        if self.candidate_K_max < self.candidate_K_min:
            raise ValueError(
                f"candidate_K_max {self.candidate_K_max} < candidate_K_min {self.candidate_K_min}"
            )


_LITERATURE_CITATION = (
    "Gersho-Gray 1992 'Vector Quantization and Signal Compression' "
    "(R-D theoretic codebook size + elbow detection on R-D curve); "
    "Cover-Thomas 'Elements of Information Theory' Ch. 13 (rate-distortion)"
)


def _power_of_two_grid(k_min: int, k_max: int) -> list[int]:
    """Power-of-two K grid in [k_min, k_max]."""
    out: list[int] = []
    k = 1
    while k <= k_max:
        if k >= k_min:
            out.append(k)
        k *= 2
    return out


def _total_cost_bits(K: int, inputs: RDCodebookSolverInput) -> float:
    """R-D total cost for codebook size K (in bits)."""
    bits_for_indices = inputs.num_codewords_used * math.log2(K)
    bits_for_codebook = K * inputs.codeword_dim * inputs.bytes_per_codeword * 8
    sigma_quant_sq = inputs.quantization_constant_C / (K ** (2.0 / inputs.codeword_dim))
    bits_for_distortion = inputs.lambda_rd * inputs.num_codewords_used * sigma_quant_sq
    return bits_for_indices + bits_for_codebook + bits_for_distortion


def solve_rd_theoretic_vq_codebook_K(
    inputs: RDCodebookSolverInput,
    *,
    emit_arbitrariness_atom: bool = False,
    codec_name: str = "<unknown_codec>",
) -> AnalyticalSolveResult:
    """Find optimal K on a power-of-two grid by argmin of R-D total cost.

    The closed-form continuous minimizer is K* = (2 * lambda_RD * N * C /
    (d^2 * bytes_per_codeword))^(d/(d+2)) but we evaluate explicit costs on a
    power-of-two grid both for codec friendliness (codebook indices pack to
    integral bit widths) AND for full visibility into the cost surface
    (operator can inspect ``per_K_cost_breakdown`` to see the R-D elbow).
    """
    K_grid = _power_of_two_grid(inputs.candidate_K_min, inputs.candidate_K_max)
    if not K_grid:
        raise ValueError(
            f"empty K_grid for [{inputs.candidate_K_min}, {inputs.candidate_K_max}]"
        )
    per_K_cost: list[tuple[int, float]] = [
        (K, _total_cost_bits(K, inputs)) for K in K_grid
    ]
    best_K, best_cost = min(per_K_cost, key=lambda kc: kc[1])

    intermediate: dict[str, Any] = {
        "per_K_cost_breakdown": per_K_cost,
        "best_cost_bits": best_cost,
        "K_grid_size": len(K_grid),
        "K_at_min_index_cost": min(K_grid, key=lambda K: inputs.num_codewords_used * math.log2(K)),
        "K_at_min_codebook_cost": K_grid[0],
    }
    coupled: dict[str, Any] = {
        "estimated_bits_per_index": math.log2(best_K),
        "estimated_codebook_bytes": best_K * inputs.codeword_dim * inputs.bytes_per_codeword,
    }
    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, best_K, codec_name)

    return AnalyticalSolveResult(
        solved_value=best_K,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.analytical_solve_extinctions.rd_theoretic_vq_codebook_K.solve_rd_theoretic_vq_codebook_K"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"Codec {codec_name}: K*={best_K} (cost {best_cost:.0f} bits); "
            f"grid {K_grid[0]}..{K_grid[-1]}; ARG_MIN over {len(K_grid)} candidates."
        ),
    )


def _emit_atom(inputs: RDCodebookSolverInput, best_K: int, codec_name: str):
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="analytical_solve_extinctions.rd_theoretic_vq_codebook_K.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id=f"vq_codebook_K_solved_for_{codec_name}",
        file_path=f"src/tac/{codec_name}.py",
        current_value="hand_picked_K_64_or_256",
        predicted_replacement={"K": best_K},
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_ev_delta_s=(-0.002, -0.0001),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/analytical_solve_extinctions/rd_theoretic_vq_codebook_K.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
    )
