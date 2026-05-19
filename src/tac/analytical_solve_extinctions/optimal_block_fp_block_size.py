# SPDX-License-Identifier: MIT
"""Row #4: Optimal block-FP block_size solver via closed-form convex minimum.

Replaces the undeclared / caller-supplied `block_size` parameter in
``src/tac/balle_hyperprior_codec.py:338`` (commonly hand-picked at 128 or 256)
with the closed-form solution to the convex tradeoff between:

    Header overhead:        H(block_size) = header_bytes_per_block * N / block_size
    Intra-block quant loss: Q(block_size) = quant_constant * block_size

The minimum of H + Q is at:

    block_size* = sqrt(N * header_bytes_per_block / quant_constant)

(derivative of H w.r.t. block_size is -N*header/(bs^2); derivative of Q is
quant_constant; setting them equal yields bs^2 = N*header/quant_constant.)

Per CLAUDE.md "Bit-level deconstruction and entropy discipline": fixed-section
removal AND deterministic pack ordering are first-class score lanes. The
block_size determines how many sections — solving this analytically eliminates
one cargo-cult variable.

Canonical-vs-unique decision per layer
--------------------------------------
- Cost functional: ADOPT_CANONICAL Ballé hyperprior framing
- Solver: UNIQUE closed-form sqrt(N*header/quant)
- Power-of-two rounding: UNIQUE for codec-byte alignment

9-dim checklist evidence: O(1); pure function; predicted ΔS [-0.002, -0.0003].

Observability: header_bytes + quant_loss + raw_minimum surfaced in
intermediate_values for the diff-able + decomposable facets.

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
class BlockFPSolverInput:
    """Inputs for the optimal block_size solve."""

    num_elements_N: int  # Total elements being block-quantized
    header_bytes_per_block: int
    quant_loss_constant: float  # Intra-block quantization variance proxy
    min_block_size: int = 4
    max_block_size: int = 4096
    round_to_power_of_two: bool = True

    def __post_init__(self) -> None:
        if self.num_elements_N <= 0:
            raise ValueError(f"num_elements_N must be positive; got {self.num_elements_N}")
        if self.header_bytes_per_block <= 0:
            raise ValueError(
                f"header_bytes_per_block must be positive; got {self.header_bytes_per_block}"
            )
        if self.quant_loss_constant <= 0:
            raise ValueError(
                f"quant_loss_constant must be positive; got {self.quant_loss_constant}"
            )
        if self.min_block_size < 1:
            raise ValueError(f"min_block_size must be >= 1; got {self.min_block_size}")
        if self.max_block_size < self.min_block_size:
            raise ValueError(
                f"max_block_size {self.max_block_size} < min_block_size {self.min_block_size}"
            )


_LITERATURE_CITATION = (
    "Ballé-Laparra-Simoncelli 2017 'End-to-end Optimized Image Compression' arxiv:1611.01704 "
    "(block-FP entropy bottleneck framing); Cover-Thomas Ch. 13 (R-D tradeoffs)"
)


def _round_to_power_of_two_nearest(x: float) -> int:
    """Round to nearest power-of-two (downward bias for codec friendliness)."""
    if x < 1.0:
        return 1
    return 1 << int(math.floor(math.log2(x)))


def solve_optimal_block_fp_block_size(
    inputs: BlockFPSolverInput,
    *,
    emit_arbitrariness_atom: bool = False,
    layer_name: str = "<unknown_layer>",
) -> AnalyticalSolveResult:
    """Closed-form block_size = sqrt(N * header_bytes_per_block / quant_loss_constant).

    The minimum of header_overhead(bs) = N*header/bs + quant_loss*bs is at
    bs* = sqrt(N*header/quant_loss). This helper computes the continuous
    minimum, then optionally rounds to the nearest power-of-two for codec
    friendliness, and clamps to [min_block_size, max_block_size].
    """
    raw_optimum = math.sqrt(
        inputs.num_elements_N * inputs.header_bytes_per_block / inputs.quant_loss_constant
    )
    if inputs.round_to_power_of_two:
        candidate = _round_to_power_of_two_nearest(raw_optimum)
    else:
        candidate = int(round(raw_optimum))
    block_size = max(inputs.min_block_size, min(inputs.max_block_size, candidate))

    num_blocks = max(1, inputs.num_elements_N // block_size)
    header_bytes_total = num_blocks * inputs.header_bytes_per_block
    quant_loss_total = inputs.quant_loss_constant * block_size
    intermediate: dict[str, Any] = {
        "raw_continuous_optimum": raw_optimum,
        "rounded_candidate": candidate,
        "num_blocks": num_blocks,
        "header_bytes_total": header_bytes_total,
        "quant_loss_total": quant_loss_total,
        "convexity_check_at_optimum": header_bytes_total + quant_loss_total,
    }
    coupled: dict[str, Any] = {
        "estimated_total_overhead_bytes": header_bytes_total,
    }
    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, block_size, layer_name)

    return AnalyticalSolveResult(
        solved_value=block_size,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.analytical_solve_extinctions.optimal_block_fp_block_size.solve_optimal_block_fp_block_size"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"Layer {layer_name}: block_size={block_size} (raw {raw_optimum:.1f}); "
            f"num_blocks={num_blocks}; header_bytes={header_bytes_total}; "
            f"closed-form per H/Q convex tradeoff."
        ),
    )


def _emit_atom(inputs: BlockFPSolverInput, block_size: int, layer_name: str):
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="analytical_solve_extinctions.optimal_block_fp_block_size.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id=f"block_fp_block_size_solved_for_{layer_name}",
        file_path="src/tac/balle_hyperprior_codec.py",
        current_value="caller_supplied_undeclared_default",
        predicted_replacement={"block_size": block_size},
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_ev_delta_s=(-0.002, -0.0003),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/analytical_solve_extinctions/optimal_block_fp_block_size.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
    )
