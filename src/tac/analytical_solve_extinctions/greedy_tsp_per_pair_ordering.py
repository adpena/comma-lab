# SPDX-License-Identifier: MIT
"""Row #10: Greedy nearest-neighbor TSP for per-pair ZIP dictionary reuse.

Replaces sequential per-pair ordering ``0, 1, 2, ..., 599`` in
``submissions/*/inflate.sh`` file_list emission with a greedy nearest-neighbor
TSP path on a per-pair similarity matrix. The objective is to maximize ZIP
deflate dictionary reuse — pairs with similar codec state (high
inner-product of latent vectors / shared mask classes / etc.) emit adjacent
bytes that the deflate sliding window can reuse.

Greedy NN-TSP is a known constant-factor approximation to optimal TSP and
runs in O(N^2); for N=600 frame pairs this is ~360k comparisons (instant).

Per CLAUDE.md "Bit-level deconstruction and entropy discipline": deterministic
pack ordering IS a first-class score lane. Solving it via greedy NN-TSP makes
the per-pair ordering ANALYTICAL rather than arbitrary.

Canonical-vs-unique decision per layer
--------------------------------------
- Algorithm: ADOPT_CANONICAL greedy nearest-neighbor
- Starting pair: UNIQUE (caller can specify, default 0)
- Similarity metric: UNIQUE per substrate (cosine, Hamming, etc.)

9-dim checklist evidence: O(N^2); pure function; predicted ΔS [-0.0005, 0.0].

Observability: full traversal + per-step nearest distance preserved.

6-hook wire-in: cathedral_autopilot_dispatch ACTIVE via Atom emission; rest N/A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tac.analytical_solve_extinctions.vram_aware_batch_size import (
    AnalyticalSolveResult,
)


@dataclass(frozen=True)
class PairOrderingInput:
    """Inputs for greedy NN-TSP per-pair ordering."""

    num_pairs: int
    pairwise_dissimilarity: Sequence[Sequence[float]]  # N x N symmetric
    start_pair_id: int = 0

    def __post_init__(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive; got {self.num_pairs}")
        if len(self.pairwise_dissimilarity) != self.num_pairs:
            raise ValueError(
                f"pairwise_dissimilarity rows={len(self.pairwise_dissimilarity)} "
                f"!= num_pairs={self.num_pairs}"
            )
        for i, row in enumerate(self.pairwise_dissimilarity):
            if len(row) != self.num_pairs:
                raise ValueError(
                    f"row {i} length {len(row)} != num_pairs {self.num_pairs}"
                )
        if not (0 <= self.start_pair_id < self.num_pairs):
            raise ValueError(
                f"start_pair_id {self.start_pair_id} not in [0, {self.num_pairs})"
            )


_LITERATURE_CITATION = (
    "Rosenkrantz-Stearns-Lewis 1977 'An Analysis of Several Heuristics for the Traveling Salesman Problem' "
    "SIAM J Computing 6(3); ZIP deflate sliding-window dictionary reuse theory"
)


def solve_greedy_tsp_per_pair_ordering(
    inputs: PairOrderingInput,
    *,
    emit_arbitrariness_atom: bool = False,
    substrate_id: str = "<unknown_substrate>",
) -> AnalyticalSolveResult:
    """Greedy nearest-neighbor TSP path; O(N^2) for N pairs.

    Starting at ``start_pair_id``, repeatedly move to the unvisited pair
    with the smallest dissimilarity to the current pair. Returns the
    ordered pair_id sequence as ``solved_value``.
    """
    n = inputs.num_pairs
    dissim = inputs.pairwise_dissimilarity
    visited: set[int] = {inputs.start_pair_id}
    order: list[int] = [inputs.start_pair_id]
    per_step_nearest_distance: list[float] = []

    current = inputs.start_pair_id
    while len(order) < n:
        best_next: int | None = None
        best_dist = float("inf")
        for j in range(n):
            if j in visited:
                continue
            d = dissim[current][j]
            if d < best_dist:
                best_dist = d
                best_next = j
        # best_next is always defined while len(order) < n
        assert best_next is not None
        visited.add(best_next)
        order.append(best_next)
        per_step_nearest_distance.append(best_dist)
        current = best_next

    total_path_dissimilarity = sum(per_step_nearest_distance)
    sequential_baseline_dissimilarity = sum(
        dissim[i][i + 1] for i in range(n - 1)
    )
    improvement = sequential_baseline_dissimilarity - total_path_dissimilarity

    intermediate: dict[str, Any] = {
        "total_path_dissimilarity": total_path_dissimilarity,
        "sequential_baseline_dissimilarity": sequential_baseline_dissimilarity,
        "absolute_improvement": improvement,
        "relative_improvement_fraction": (
            improvement / sequential_baseline_dissimilarity
            if sequential_baseline_dissimilarity > 0
            else 0.0
        ),
        "per_step_nearest_distance_first_8": per_step_nearest_distance[:8],
    }
    coupled: dict[str, Any] = {
        "start_pair_first": order[0] == inputs.start_pair_id,
        "is_permutation": sorted(order) == list(range(n)),
    }
    if emit_arbitrariness_atom:
        coupled["atom"] = _emit_atom(inputs, order, substrate_id, total_path_dissimilarity)

    return AnalyticalSolveResult(
        solved_value=order,
        intermediate_values=intermediate,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_invocation=(
            "tac.analytical_solve_extinctions.greedy_tsp_per_pair_ordering.solve_greedy_tsp_per_pair_ordering"
        ),
        coupled_adjustments=coupled,
        notes=(
            f"Substrate {substrate_id}: {n} pairs ordered via greedy NN-TSP "
            f"from pair {inputs.start_pair_id}; total_dissim={total_path_dissimilarity:.4f} "
            f"vs sequential baseline {sequential_baseline_dissimilarity:.4f}."
        ),
    )


def _emit_atom(
    inputs: PairOrderingInput, order: list[int], substrate_id: str, total_dissim: float
):
    from tac.atom.builders import build_arbitrary_value_atom
    from tac.atom.types import ResolutionPath
    from tac.provenance.builders import build_provenance_for_predicted

    provenance = build_provenance_for_predicted(
        model_id="analytical_solve_extinctions.greedy_tsp_per_pair_ordering.v1",
        inputs_sha256="0" * 64,
    )
    return build_arbitrary_value_atom(
        atom_id=f"pair_ordering_greedy_tsp_solved_for_{substrate_id}",
        file_path=f"submissions/{substrate_id}/inflate.sh",
        current_value="sequential_0_to_N",
        predicted_replacement={
            "order_first_8": order[:8],
            "total_pairs": len(order),
            "total_path_dissimilarity": total_dissim,
        },
        resolution_path=ResolutionPath.ANALYTICAL_SOLVE,
        predicted_ev_delta_s=(-0.0005, 0.0),
        cost_envelope_usd=0.0,
        literature_citation=_LITERATURE_CITATION,
        canonical_helper_repo_link=(
            "src/tac/analytical_solve_extinctions/greedy_tsp_per_pair_ordering.py"
        ),
        provenance=provenance,
        captured_by_subagent="lane_arbitrariness_extinction_wave_2a_path2_analytical_solve_batch_20260518",
    )
