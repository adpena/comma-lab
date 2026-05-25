"""Probe disambiguator for CUDA-aware DQS1 variant via per-axis Pareto pair ordering.

CANONICAL: design memo `.omx/research/cuda_axis_dqs1_design_memo_per_axis_pareto_pair_ordering_20260525.md`.

This is a SKELETON per the CUDA-AXIS-DQS1-DESIGN subagent's DELIVERABLE 2 (2026-05-25).
The helper bodies are intentionally `NotImplementedError` stubs; BUILD-1 through BUILD-4 sister
subagents per the design memo's operator-routable populate them in subsequent landings.

PURPOSE per Catalog #296 acceptance cascade (b):
    The design memo's predicted ΔS band [ΔCPU ∈ [-5e-7, +5e-7] AND ΔCUDA ∈ [-1e-6, +5e-7]]
    is paired with this probe-disambiguator path; once BUILD-1+BUILD-2 land helper bodies,
    the disambiguator empirically resolves the prediction by enumerating per-axis pair-deltas,
    constructing per-axis Pareto frontier + Dykstra alternating-projection feasibility set,
    and recommending minimax-optimal pair-drop.

CANONICAL EQUATION CANDIDATE per Catalog #344 RATIFY-N:
    `cuda_axis_dqs1_per_axis_pareto_pair_ordering_v1` (FORMALIZATION_PENDING until BUILD-1 +
    DISPATCH produce >= 3 paired CPU+CUDA empirical anchors per equation #36 residual discipline).

SISTER OF:
    - `pairset_component_marginal_score_decomposition_v1` (equation #36; per-pair per-axis Δ COMPOSITION surface)
    - `cpu_cuda_score_gap_v1` (equation #17; CPU-CUDA axis-split surface)
    - `pose_axis_cuda_amplification_v1` (equation #18; pose-axis amplification surface)
    - `cuda_axis_dqs1_regression_segnet_shift_v1` (DQS1-LOOP-CLOSURE-ASSIST Candidate 3 QUEUED)

6-HOOK WIRE-IN DECLARATION per Catalog #125 (see design memo for full rationale):
    Hook #1 sensitivity-map = ACTIVE | Hook #2 Pareto constraint = ACTIVE
    Hook #3 bit-allocator = ACTIVE | Hook #4 cathedral autopilot dispatch = ACTIVE
    Hook #5 continual-learning posterior = ACTIVE | Hook #6 probe-disambiguator = ACTIVE PRIMARY

CANONICAL PROVENANCE per Catalog #323:
    Every emitted `PerAxisPairDelta` row carries grade=`[predicted]`, score_claim=False,
    promotable=False per Catalog #341 canonical non-promotable markers (Tier A observability-only
    until BUILD-4 + DISPATCH ratifies; then UPGRADE-eligible to Tier B score-contributing per
    Catalog #357 once the canonical equation candidate is registered with >= 3 paired anchors).
"""

# SPDX-License-Identifier: MIT
# CHECKPOINT_DISCIPLINE_WAIVED:skeleton-scaffold-only-no-runtime-execution-per-design-memo-build-1-through-build-4-sister-subagents-populate-helper-bodies

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

__all__ = [
    "PerAxisPairDelta",
    "PerAxisPairDeltaComponents",
    "DykstraFeasibilityVerdict",
    "PerAxisParetoRankingResult",
    "build_per_axis_pair_delta_from_master_gradient_ledger",
    "build_per_axis_pareto_ranking",
    "find_dykstra_feasible_intersection",
    "find_minimax_optimal_drop_one",
    "emit_axis_decomposition_for_canonical_helper",
    "FallbackMode",
    "DEFAULT_EPSILON_CPU",
    "DEFAULT_EPSILON_CUDA",
]


# Canonical epsilon thresholds per design memo Section 1.3 Dykstra alternating-projections.
# - ε_CPU = 0 (strict CPU improvement; per equation #36 sign convention ΔS < 0 = improvement)
# - ε_CUDA = +5e-7 (CUDA near-neutrality acceptable per current frontier-gap analysis:
#   CUDA frontier still PR106 format0d 0.20533002, NOT a DQS1 archive; small CUDA regression
#   acceptable if CPU improvement materially advances frontier).
DEFAULT_EPSILON_CPU: float = 0.0
DEFAULT_EPSILON_CUDA: float = 5.0e-7

FallbackMode = Literal[
    "minimax_global",  # Section 1.4 fallback 1: argmin max(ΔCPU, ΔCUDA) over all pairs
    "mean_weighted",   # Section 1.4 fallback 2: argmin (w_CPU*ΔCPU + w_CUDA*ΔCUDA) over all pairs
    "lexicographic",   # Section 1.4 fallback 3: argmin ΔCPU then tiebreak by ΔCUDA over top-K
    "dykstra_strict_intersection",  # No fallback: return EMPTY if Dykstra-feasible set is empty
]


@dataclass(frozen=True)
class PerAxisPairDeltaComponents:
    """Per-pair per-axis Δ component decomposition per canonical equation #36.

    Sign conventions per equation #36:
    - delta_rate < 0: rate-term decrease (favorable; byte-drop reduces archive_bytes)
    - delta_segnet_axis > 0: SegNet distortion increase (unfavorable per CLAUDE.md "SegNet vs PoseNet importance")
    - delta_posenet_axis: signed (sqrt(10 * pose_avg_axis_new) - sqrt(10 * pose_avg_axis_baseline);
      non-linear per canonical contest formula)
    - delta_total_axis = delta_rate + delta_segnet_axis + delta_posenet_axis
    """

    delta_rate: float  # Axis-agnostic rate term Δ per equation #36 (negative = favorable byte-drop)
    delta_segnet_cpu: float
    delta_segnet_cuda: float
    delta_posenet_cpu: float
    delta_posenet_cuda: float

    def delta_cpu(self) -> float:
        """ΔS_CPU(pair_i) = ΔRate + ΔSegNet_CPU + ΔPoseNet_CPU per equation #36."""
        return self.delta_rate + self.delta_segnet_cpu + self.delta_posenet_cpu

    def delta_cuda(self) -> float:
        """ΔS_CUDA(pair_i) = ΔRate + ΔSegNet_CUDA + ΔPoseNet_CUDA per equation #36."""
        return self.delta_rate + self.delta_segnet_cuda + self.delta_posenet_cuda

    def minimax_score(self) -> float:
        """max(ΔCPU, ΔCUDA) per Section 1.4 minimax aggregation."""
        return max(self.delta_cpu(), self.delta_cuda())


@dataclass(frozen=True)
class DykstraFeasibilityVerdict:
    """Per-pair Dykstra alternating-projections feasibility verdict per Section 1.3."""

    in_cpu_improvement_polytope: bool  # ΔCPU < epsilon_cpu
    in_cuda_improvement_polytope: bool  # ΔCUDA <= epsilon_cuda
    in_feasible_intersection: bool  # both polytope memberships true
    epsilon_cpu_used: float
    epsilon_cuda_used: float


@dataclass(frozen=True)
class PerAxisPairDelta:
    """Canonical per-pair per-axis Δ decomposition row.

    BUILD-1 sister subagent populates this dataclass from the 600-pair fp64 master-gradient
    ledger × paired CPU+CUDA scorer responses for the FRONTIER archive `7a0da5d0fc32`.
    BUILD-2 sister subagent populates `pareto_rank` + `dykstra_feasibility` + `minimax_rank` fields.
    BUILD-4 sister subagent emits canonical `AxisDecomposition` per Catalog #356 via
    `emit_axis_decomposition_for_canonical_helper(...)`.
    """

    pair_idx: int  # Canonical pair index in DQS1 600-pair selector space (0-599)
    components: PerAxisPairDeltaComponents
    pareto_rank: int | None = None  # Fonseca-Fleming non-dominated layer; None = not yet ranked
    dykstra_feasibility: DykstraFeasibilityVerdict | None = None  # None = not yet checked
    minimax_rank: int | None = None  # Rank in ascending minimax_score order; None = not yet ranked
    archive_sha256_short: str | None = None  # First 12 chars of FRONTIER archive sha per Catalog #287
    canonical_provenance: Mapping[str, Any] | None = None  # Per Catalog #323 + #341 canonical markers

    def as_dict(self) -> Mapping[str, Any]:
        """Canonical JSONL round-trip per Catalog #131 + #245 pattern (BUILD-2 wires emit)."""
        raise NotImplementedError(
            "BUILD-2 sister subagent populates as_dict() per canonical JSONL append-only schema; "
            "see design memo DELIVERABLE 3 §3.2 BUILD-2 cost ~2-4h $0 GPU."
        )


@dataclass(frozen=True)
class PerAxisParetoRankingResult:
    """BUILD-2 output: ranked candidate list + Pareto/Dykstra/minimax verdicts.

    Canonical Provenance per Catalog #323 routes through `canonical_provenance` field;
    Catalog #341 canonical non-promotable markers carried per row.
    """

    candidates: Sequence[PerAxisPairDelta]
    feasible_intersection: Sequence[PerAxisPairDelta]
    minimax_optimal: PerAxisPairDelta | None  # None if dykstra_strict_intersection fallback + empty set
    fallback_mode_used: FallbackMode
    epsilon_cpu_used: float
    epsilon_cuda_used: float
    canonical_equation_id: str = "cuda_axis_dqs1_per_axis_pareto_pair_ordering_v1"  # FORMALIZATION_PENDING per Catalog #344
    canonical_provenance: Mapping[str, Any] | None = None


def build_per_axis_pair_delta_from_master_gradient_ledger(
    archive_sha256: str,
    *,
    master_gradient_ledger_path: Path,
    paired_cpu_cuda_anchor_path: Path,
    pair_indices: Iterable[int] | None = None,
) -> list[PerAxisPairDelta]:
    """BUILD-1: populate per-axis pair-delta table from master-gradient ledger × paired anchors.

    Sister subagent populates per design memo DELIVERABLE 3 §3.2 BUILD-1:
        Inputs:
            - `.omx/state/master_gradient_ledger.jsonl` (or canonical helper output)
            - paired CPU+CUDA scorer responses (CPU from local macOS smoke if MPS-VIABLE
              per Catalog #341; CUDA from harvested pair0371 anchor + paired anchors from
              sister DISPATCH cascade)
        Outputs:
            - NEW `.omx/state/cuda_axis_dqs1_per_axis_pair_delta_table_<archive_sha_short>_<utc>.jsonl`
              (canonical fcntl-locked JSONL per Catalog #131 + #245 pattern)
        Cost: ~2-4h wall-clock; $0 GPU
        Validation: per-pair residual against equation #36 < 5e-7

    Args:
        archive_sha256: FRONTIER archive sha (canonical: `7a0da5d0fc32...`)
        master_gradient_ledger_path: Path to canonical master-gradient ledger JSONL
        paired_cpu_cuda_anchor_path: Path to paired CPU+CUDA scorer response JSONL
        pair_indices: Optional subset of pair indices to populate (default: all 600)

    Returns:
        List of `PerAxisPairDelta` rows (one per pair) with `components` populated;
        `pareto_rank` + `dykstra_feasibility` + `minimax_rank` left None for BUILD-2.

    Raises:
        NotImplementedError: BUILD-1 sister subagent populates this body.
    """
    raise NotImplementedError(
        "BUILD-1 sister subagent populates this body per design memo DELIVERABLE 3 §3.2 BUILD-1. "
        "Cost: ~2-4h wall-clock; $0 GPU. Sister is spawnable independently; DISJOINT from "
        "PR95-STAGE-2-MLX-BUILD + DROP-MANY-BEAM-BUILD-1 + COMBINED-TIER-1-WAVE-2."
    )


def build_per_axis_pareto_ranking(
    candidates: Sequence[PerAxisPairDelta],
) -> Sequence[PerAxisPairDelta]:
    """BUILD-2: populate per-axis Pareto rank per Fonseca-Fleming 1995 NSGA-style.

    Sister subagent populates per design memo DELIVERABLE 3 §3.2 BUILD-2:
        A pair `pair_i` is CPU-CUDA-Pareto-non-dominated iff
            ∀ pair_j ≠ pair_i: ΔCPU(pair_j) > ΔCPU(pair_i) OR ΔCUDA(pair_j) > ΔCUDA(pair_i)
        (no other pair strictly improves BOTH axes)

        Per-pair Pareto rank = depth in successive non-dominated layers.

    Args:
        candidates: BUILD-1 output (per-axis pair-delta table)

    Returns:
        Re-ranked candidate sequence with `pareto_rank` populated per row.

    Raises:
        NotImplementedError: BUILD-2 sister subagent populates this body.
    """
    raise NotImplementedError(
        "BUILD-2 sister subagent populates this body per design memo DELIVERABLE 3 §3.2 BUILD-2. "
        "Cost: ~2-4h wall-clock; $0 GPU (pure Python algorithm on BUILD-1 table). "
        "Spawnable after BUILD-1 lands. Validation: per-pair Pareto-rank monotonicity."
    )


def find_dykstra_feasible_intersection(
    candidates: Sequence[PerAxisPairDelta],
    *,
    epsilon_cpu: float = DEFAULT_EPSILON_CPU,
    epsilon_cuda: float = DEFAULT_EPSILON_CUDA,
) -> Sequence[PerAxisPairDelta]:
    """BUILD-2: Dykstra alternating-projection feasibility per design memo Section 1.3.

    Sister subagent populates per design memo DELIVERABLE 3 §3.2 BUILD-2:
        CPU-improvement polytope C_CPU = {pair_i : ΔCPU(pair_i) < epsilon_cpu}
        CUDA-improvement polytope C_CUDA = {pair_i : ΔCUDA(pair_i) <= epsilon_cuda}
        Feasible intersection F = C_CPU ∩ C_CUDA

        Algorithm (Dykstra-1983 alternating-projections; convergent for closed convex sets;
        here finite discrete pair set with Dykstra correction term q_k vanishing per discrete case):
        1. Initialize F_0 = {0, ..., 599} (all candidate pairs)
        2. Project onto C_CPU: F_1 = F_0 ∩ C_CPU
        3. Project onto C_CUDA: F_2 = F_1 ∩ C_CUDA
        4. Verify F_2 idempotent (one round suffices for finite discrete sets)
        5. If F_2 == ∅, caller falls back to minimax over full candidate set per Section 1.4

    Args:
        candidates: BUILD-1 output (per-axis pair-delta table)
        epsilon_cpu: CPU improvement polytope threshold (default 0.0 = strict improvement)
        epsilon_cuda: CUDA improvement polytope threshold (default +5e-7 = near-neutrality)

    Returns:
        Subset of candidates in Dykstra-feasible intersection (may be empty);
        each carries populated `dykstra_feasibility` verdict.

    Raises:
        NotImplementedError: BUILD-2 sister subagent populates this body.
    """
    raise NotImplementedError(
        "BUILD-2 sister subagent populates this body per design memo DELIVERABLE 3 §3.2 BUILD-2 "
        "+ Section 1.3 Dykstra alternating-projections algorithm. Cost: ~2-4h wall-clock; $0 GPU. "
        "Validation: Dykstra-feasibility set membership invariant + idempotence under second projection."
    )


def find_minimax_optimal_drop_one(
    candidates: Sequence[PerAxisPairDelta],
    *,
    fallback_mode: FallbackMode = "minimax_global",
    mean_weights: tuple[float, float] = (0.5, 0.5),  # (w_CPU, w_CUDA) for mean_weighted fallback
    lexicographic_top_k: int = 10,  # K for lexicographic fallback's top-K filter
    epsilon_cpu: float = DEFAULT_EPSILON_CPU,
    epsilon_cuda: float = DEFAULT_EPSILON_CUDA,
) -> PerAxisParetoRankingResult:
    """BUILD-2: minimax-optimal pair-drop selection per design memo Section 1.4.

    Sister subagent populates per design memo DELIVERABLE 3 §3.2 BUILD-2:
        Primary: find Dykstra-feasible intersection F (Section 1.3); if non-empty,
            pair* = argmin_{pair_i ∈ F} max(ΔCPU(pair_i), ΔCUDA(pair_i))  (minimax within F)
        Fallback when F == ∅ (one of):
            - minimax_global: pair* = argmin over all pairs of max(ΔCPU, ΔCUDA)
            - mean_weighted: pair* = argmin (w_CPU * ΔCPU + w_CUDA * ΔCUDA)
            - lexicographic: argmin ΔCPU filtered to top-K then argmin ΔCUDA
            - dykstra_strict_intersection: return None (no fallback)

    Args:
        candidates: BUILD-1 output (per-axis pair-delta table)
        fallback_mode: aggregation strategy when Dykstra-feasible set is empty
        mean_weights: (w_CPU, w_CUDA) for mean_weighted fallback (default uniform)
        lexicographic_top_k: K for lexicographic fallback's top-K filter
        epsilon_cpu: Dykstra CPU polytope threshold
        epsilon_cuda: Dykstra CUDA polytope threshold

    Returns:
        `PerAxisParetoRankingResult` with `minimax_optimal` = selected pair OR None;
        canonical Provenance per Catalog #323 + Catalog #341 non-promotable markers
        (`score_claim=False`, `promotable=False`, `axis_tag="[predicted]"`).

    Raises:
        NotImplementedError: BUILD-2 sister subagent populates this body.
    """
    raise NotImplementedError(
        "BUILD-2 sister subagent populates this body per design memo DELIVERABLE 3 §3.2 BUILD-2 "
        "+ Section 1.4 multi-axis aggregation strategies. Cost: ~2-4h wall-clock; $0 GPU. "
        "Validation: minimax score = max(ΔCPU, ΔCUDA) exactly; per-fallback-mode regression tests."
    )


def emit_axis_decomposition_for_canonical_helper(
    per_axis_pair_delta: PerAxisPairDelta,
    *,
    archive_sha256: str,
    canonical_baseline_pose: float | None = None,
) -> Any:
    """BUILD-4: emit canonical `AxisDecomposition` per Catalog #356.

    Sister subagent populates per design memo DELIVERABLE 3 §3.2 BUILD-4:
        Wires per-axis Δ predictions through canonical `tac.cathedral.consumer_contract.AxisDecomposition`
        + canonical `tac.score_composition.compose_score_from_axes` helper for canonical contest
        formula application; canonical baseline pose loaded via
        `tac.score_composition.load_baseline_pose_from_canonical_frontier_pointer` (Catalog #343 sister).

        Canonical Provenance per Catalog #323 with:
            score_claim=False
            promotable=False
            axis_tag="[predicted]"
        per Catalog #341 canonical non-promotable markers (Tier A observability-only until DISPATCH
        produces >= 3 paired CPU+CUDA empirical anchors per Catalog #344 RATIFY-N for equation
        candidate `cuda_axis_dqs1_per_axis_pareto_pair_ordering_v1`).

    Args:
        per_axis_pair_delta: BUILD-2 output for a single ranked candidate
        archive_sha256: FRONTIER archive sha for canonical Provenance
        canonical_baseline_pose: Optional explicit baseline pose; default loads from canonical frontier pointer

    Returns:
        `tac.cathedral.consumer_contract.AxisDecomposition` instance per Catalog #356 frozen dataclass.

    Raises:
        NotImplementedError: BUILD-4 sister subagent populates this body.
    """
    raise NotImplementedError(
        "BUILD-4 sister subagent populates this body per design memo DELIVERABLE 3 §3.2 BUILD-4. "
        "Cost: ~1-2h wall-clock; $0 GPU. Spawnable after BUILD-2 lands. "
        "Routes through `tac.score_composition.compose_score_from_axes` + canonical AxisDecomposition "
        "per Catalog #356 + canonical Provenance per Catalog #323 + canonical non-promotable markers "
        "per Catalog #341. Validation: per-axis compose_scalar_delta(decomposition) matches predicted "
        "ΔCPU and ΔCUDA within 1e-9; canonical Provenance valid per tac.provenance.audit_score_claim_dict."
    )
