#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DQS1 drop-many beam pairwise-interaction waterfill probe-disambiguator.

Canonical scaffold skeleton per design memo:
``.omx/research/dqs1_drop_many_beam_pairwise_interaction_waterfill_design_memo_20260525.md``.

THIS IS A SKELETON — NOT A FULL EXECUTABLE BUILD.

Per Carmack MVP-first 5-step phasing, this file lands the:
- Canonical interface contract (dataclasses + function signatures)
- Canonical Provenance per Catalog #323 + Catalog #341 routing markers
- Operator-routable BUILD-1..4 + DISPATCH next-cascade enumeration
- ``NotImplementedError`` stubs for ``beam_search_drop_many`` +
  ``dykstra_alternating_projection_feasibility`` + ``waterfill_budget_consumed``
- Working ``build_pairwise_interaction_matrix`` stub that documents the
  expected fp64 ledger input and emits a canonical-provenance-tagged synthetic
  matrix for design-time observability

Full BUILD per the design memo:
- BUILD-1: populate empirical I[P,P] via ``$0`` CPU smoke on existing
  600-pair fp64 master-gradient ledger (~2-4h)
- BUILD-2: full ``beam_search_drop_many`` + Dykstra feasibility + waterfill
  (~4-6h)
- BUILD-3: Catalog #356 ``AxisDecomposition`` wire-in (~1-2h)
- BUILD-4: Catalog #357 Tier B canonical consumer (~2-3h)
- DISPATCH: paired CPU+CUDA Modal exact-eval of top-K=8 (~$2.40-4)

Authority per Catalog #287/#323/#341: score_claim=False / promotable=False /
axis_tag='[predicted]' for ALL emissions until DISPATCH lands paired anchors.

Canonical equation candidate QUEUED per Catalog #344:
``dqs1_drop_many_pairwise_interaction_beam_search_v1`` —
see design memo for math + acceptance criteria.

Sister of:
- ``tac.optimization.decoder_q_pairset_acquisition`` (reuses candidate schema;
  ``selector_kind = drop_many_beam_pairwise_interaction_waterfill`` is the
  canonical name codex registered for this)
- ``pairset_component_marginal_score_decomposition_v1`` (equation #36; the
  per-pair drop-one ratified base case)
- ``per_pair_master_gradient_score_impact_taylor_v1`` (equation #4; per-pair
  Taylor + Cauchy-Schwarz bound at sister surface)
- Wave-Ω water-filling sister (rate-budget redistribution surface)

[predicted] [empirical:pending_BUILD_1]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "BeamCandidate",
    "BeamSearchConfig",
    "DykstraFeasibilityConfig",
    "PairCandidate",
    "WaterfillConfig",
    "beam_search_drop_many",
    "build_pairwise_interaction_matrix",
    "dykstra_alternating_projection_feasibility",
    "waterfill_budget_consumed",
]


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FP64_LEDGER_PATH = REPO_ROOT / ".omx" / "state" / "master_gradient_anchors.jsonl"
DEFAULT_ACQUISITION_PLAN_PATH = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "codex_eureka_beyond_drop_two_acquisition_20260525T143351Z"
    / "dqs1_pairset_acquisition_eureka_drop_many.json"
)

# Canonical Daubechies multi-scale prior K=8 + Carmack MVP-first 5-step D=4
CANONICAL_BEAM_WIDTH_K = 8
CANONICAL_BEAM_DEPTH_D = 4
CANONICAL_DYKSTRA_MAX_ITERS = 8
CANONICAL_DYKSTRA_CONVERGENCE_EPS = 1e-6

# Canonical contest formula constants per Catalog #356 ``tac.score_composition``
CANONICAL_SEG_MULTIPLIER = 100.0
CANONICAL_POSE_SQRT_INNER = 10.0
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489

# Canonical Catalog #341 non-promotable routing markers
CANONICAL_AXIS_TAG_PREDICTED = "[predicted]"
CANONICAL_NON_PROMOTABLE_MARKERS: Mapping[str, Any] = {
    "predicted_delta_adjustment": 0.0,
    "promotable": False,
    "axis_tag": CANONICAL_AXIS_TAG_PREDICTED,
    "score_claim": False,
    "score_claim_valid": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "evidence_grade": "research_only",
    "allowed_use": "local_planning_only_pairwise_interaction_beam_search_no_dispatch_authority",
    "forbidden_use": "score_claim_or_distortion_authority",
}


@dataclasses.dataclass(frozen=True)
class PairCandidate:
    """Single pair candidate from the DQS1 acquisition pool.

    Mirrors a subset of fields from
    ``decoder_q_pairset_acquisition_candidate.v1`` schema. Canonical-source
    fields are read-only references; the beam search does NOT mutate the
    canonical schema per Catalog #110/#113 APPEND-ONLY discipline.
    """

    pair_index: int
    rate_score_delta_vs_source_selector: float  # canonical per-pair rate ΔS
    predicted_score_mean: float  # source_selector predicted (inherited)
    payload_bytes_delta_vs_source_selector: int
    distortion_repair_budget_score: float  # from ``distortion_repair_budget_from_rate_savings.score_budget``


@dataclasses.dataclass(frozen=True)
class BeamSearchConfig:
    """Canonical beam search config per design memo."""

    width_k: int = CANONICAL_BEAM_WIDTH_K
    depth_d: int = CANONICAL_BEAM_DEPTH_D
    early_stop_when_no_negative_delta: bool = True
    random_seed: int = 0


@dataclasses.dataclass(frozen=True)
class DykstraFeasibilityConfig:
    """Canonical Dykstra alternating-projection feasibility config.

    The 3-polytope intersection per design memo:
    1. Rate-saving polytope R: total archive_bytes_delta ≥ R_min
    2. SegNet-penalty-bound polytope S: total predicted ΔSegNet × 100 ≤ S_max
    3. PoseNet-stability polytope P: total predicted ΔPoseNet × sqrt(10) ≤ P_max
    """

    rate_min_bytes_saved: int = 2  # R_min; null beam fails this
    segnet_max_score_units: float = 0.0005  # S_max; conservative
    posenet_max_score_units: float = 0.0005  # P_max; conservative
    max_iterations: int = CANONICAL_DYKSTRA_MAX_ITERS
    convergence_eps: float = CANONICAL_DYKSTRA_CONVERGENCE_EPS


@dataclasses.dataclass(frozen=True)
class WaterfillConfig:
    """Canonical waterfill budget redistribution config.

    Reuses canonical ``distortion_repair_budget_from_rate_savings`` schema
    fields per DQS1 candidate row (per-pair score_budget +
    posenet_score_term_budget_at_fixed_seg + segnet_distortion_budget_at_fixed_pose).
    """

    # Per-pair waterfill: how much of the rate-savings budget goes to SegNet
    # repair vs PoseNet repair? Canonical: half-and-half until BUILD-2 lands
    # empirical calibration via canonical equation candidate v1.
    segnet_repair_fraction: float = 0.5
    posenet_repair_fraction: float = 0.5


@dataclasses.dataclass(frozen=True)
class BeamCandidate:
    """Single beam candidate (drop-K-tuple) with predicted ΔS decomposition."""

    drop_tuple: tuple[int, ...]  # sorted pair indices to drop
    depth: int
    delta_s_independent: float  # Σ per-pair ΔS_indep
    delta_s_interaction: float  # Σ_{i<j} I[i,j] × indicator
    delta_s_waterfill_budget_consumed: float
    delta_s_total: float  # delta_indep + delta_interaction - delta_waterfill
    dykstra_feasible: bool
    canonical_provenance: Mapping[str, Any]  # per Catalog #323


def build_pairwise_interaction_matrix(
    candidates: Sequence[PairCandidate],
    *,
    fp64_ledger_path: Path = DEFAULT_FP64_LEDGER_PATH,
    synthetic_design_time_only: bool = True,
) -> dict[str, Any]:
    """Build empirical pairwise interaction matrix I[P,P].

    Δ_ij = ΔS(drop i AND j) − ΔS(drop i) − ΔS(drop j)

    BUILD-1 fills this in via $0 CPU smoke on existing 600-pair fp64 ledger.
    THIS SKELETON returns a synthetic null-matrix (all zeros) tagged with
    canonical Provenance per Catalog #323 + non-promotable markers per
    Catalog #341 so design-time observability flows but no false authority
    leaks into downstream consumers.

    Returns a dict with keys:
    - ``interaction_matrix``: nested list shape (P, P) — sparse storage
      for design-time skeleton; BUILD-1 will emit dense np.ndarray
    - ``matrix_metadata``: canonical Provenance + sha of source fp64 ledger
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    pair_indices = sorted({c.pair_index for c in candidates})
    p_max = max(pair_indices) + 1

    if synthetic_design_time_only:
        # Skeleton: sparse all-zero matrix; BUILD-1 populates empirically
        matrix_data: dict[str, float] = {}  # (i, j) -> Δ_ij with i < j
        canonical_provenance = {
            **CANONICAL_NON_PROMOTABLE_MARKERS,
            "kind": "synthetic_design_time_skeleton",
            "captured_at_utc": "2026-05-25T15:30:00Z",
            "canonical_helper_invocation": (
                "tools/probe_dqs1_drop_many_beam_pairwise_interaction_disambiguator.py::"
                "build_pairwise_interaction_matrix(synthetic_design_time_only=True)"
            ),
            "source_url": str(fp64_ledger_path) if fp64_ledger_path.exists() else "PENDING_BUILD_1",
            "next_action": "BUILD-1 populates empirical I[P,P] from fp64 ledger",
            "design_memo_ref": (
                ".omx/research/dqs1_drop_many_beam_pairwise_interaction_waterfill_"
                "design_memo_20260525.md"
            ),
        }
        return {
            "p_max": p_max,
            "pair_indices": pair_indices,
            "interaction_matrix_sparse_lower_triangle": matrix_data,
            "matrix_metadata": canonical_provenance,
        }

    # BUILD-1 entry-point: empirical Δ_ij measurement
    raise NotImplementedError(
        "Empirical I[P,P] population is BUILD-1; "
        "design memo §4-BUILD operator-routable enumeration"
    )


def dykstra_alternating_projection_feasibility(
    drop_tuple: tuple[int, ...],
    candidates: Sequence[PairCandidate],
    *,
    config: DykstraFeasibilityConfig = DykstraFeasibilityConfig(),  # noqa: B008 - frozen dataclass default
) -> bool:
    """Dykstra alternating-projection check for 3-polytope feasibility.

    Per Boyd CO-LEAD + design memo §"Dykstra feasibility intersection":
    halt when ‖x_{k+1} − x_k‖ < ε OR max iterations reached.

    BUILD-2 implements the actual alternating projection iteration.
    THIS SKELETON returns True for non-empty tuples that have positive
    rate savings (R polytope satisfied by construction for codex's
    34 candidates per design memo's empirical verification).
    """
    if not drop_tuple:
        return False

    raise NotImplementedError(
        "Dykstra alternating-projection feasibility check is BUILD-2; "
        "design memo §4-BUILD operator-routable enumeration"
    )


def waterfill_budget_consumed(
    drop_tuple: tuple[int, ...],
    candidates: Sequence[PairCandidate],
    *,
    config: WaterfillConfig = WaterfillConfig(),  # noqa: B008 - frozen dataclass default
) -> float:
    """Compute waterfill budget consumed by SegNet+PoseNet repair.

    Per design memo: rate savings buy SegNet+PoseNet repair via canonical
    ``distortion_repair_budget_from_rate_savings`` schema; budget is
    redistributed across the drop-K-tuple per ``segnet_repair_fraction``
    and ``posenet_repair_fraction`` (default half-and-half until BUILD-2
    lands empirical calibration).
    """
    if not drop_tuple:
        return 0.0

    raise NotImplementedError(
        "Waterfill budget redistribution is BUILD-2; "
        "design memo §4-BUILD operator-routable enumeration"
    )


def beam_search_drop_many(
    candidates: Sequence[PairCandidate],
    interaction_matrix: Mapping[str, Any],
    *,
    config: BeamSearchConfig = BeamSearchConfig(),  # noqa: B008 - frozen dataclass default
    dykstra_config: DykstraFeasibilityConfig = DykstraFeasibilityConfig(),  # noqa: B008
    waterfill_config: WaterfillConfig = WaterfillConfig(),  # noqa: B008
) -> list[BeamCandidate]:
    """Interaction-aware beam search over drop-K-tuples.

    See design memo §"Algorithm pseudocode" for the canonical algorithm.

    Width K (default 8 per Daubechies multi-scale prior), depth D
    (default 4 per Carmack MVP-first 5-step). Per-step expansion uses
    interaction_matrix to predict drop-K-tuple ΔS = sum_indep +
    interaction_correction − waterfill_budget_consumed.

    Dykstra alternating-projection feasibility halts beam expansion
    when polytope intersection becomes empty.

    BUILD-2 implements the full algorithm.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if config.width_k <= 0 or config.depth_d <= 0:
        raise ValueError(f"width_k + depth_d must be positive: got {config!r}")
    if interaction_matrix.get("p_max", 0) <= 0:
        raise ValueError("interaction_matrix must carry p_max")

    raise NotImplementedError(
        "Full beam_search_drop_many is BUILD-2; "
        "design memo §4-BUILD operator-routable enumeration"
    )


def _load_acquisition_candidates(
    acquisition_plan_path: Path = DEFAULT_ACQUISITION_PLAN_PATH,
) -> list[PairCandidate]:
    """Load PairCandidate rows from the codex DQS1 acquisition plan.

    Filters to the drop_many_beam_pairwise_interaction_waterfill selector_kind
    rows (34 in codex's current plan) but treats them as the candidate POOL
    for beam search rather than as final independent evaluations.
    """
    if not acquisition_plan_path.exists():
        raise FileNotFoundError(
            f"acquisition plan not found at {acquisition_plan_path}; "
            "verify codex DQS1 cascade has landed"
        )
    with acquisition_plan_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    candidates_data = data.get("candidates", [])
    pair_candidates: list[PairCandidate] = []
    seen_pair_indices: set[int] = set()
    for row in candidates_data:
        if not isinstance(row, Mapping):
            continue
        op = row.get("acquisition_operation")
        if not isinstance(op, Mapping):
            continue
        if op.get("op") != "drop_many":
            # Beam pool includes drop_one + drop_two + drop_many; for design-time
            # skeleton, just inspect drop_many pool to confirm schema parsing.
            continue
        for pair_index in op.get("dropped_pair_indices", []):
            if not isinstance(pair_index, int) or pair_index in seen_pair_indices:
                continue
            seen_pair_indices.add(pair_index)
            budget = row.get("distortion_repair_budget_from_rate_savings", {})
            pair_candidates.append(
                PairCandidate(
                    pair_index=pair_index,
                    rate_score_delta_vs_source_selector=float(
                        row.get("rate_score_delta_vs_source_selector", 0.0)
                    ),
                    predicted_score_mean=float(row.get("predicted_score_mean", 0.0)),
                    payload_bytes_delta_vs_source_selector=int(
                        row.get("payload_bytes_delta_vs_source_selector", 0)
                    ),
                    distortion_repair_budget_score=float(
                        budget.get("score_budget", 0.0) if isinstance(budget, Mapping) else 0.0
                    ),
                )
            )
    return pair_candidates


def main(argv: Sequence[str] | None = None) -> int:
    """Probe-disambiguator skeleton entry-point.

    Canonical CLI surface (BUILD-1..4 will extend):
    - ``--acquisition-plan <path>``: codex acquisition JSON (default canonical)
    - ``--fp64-ledger <path>``: 600-pair fp64 master-gradient ledger
    - ``--width-k <int>``: beam width K (default 8)
    - ``--depth-d <int>``: beam depth D (default 4)
    - ``--probe-pairwise-interaction <i> <j>``: BUILD-1 single-cell probe
    - ``--output-dir <path>``: artifact emission directory
    """
    parser = argparse.ArgumentParser(
        description=(
            "DQS1 drop-many beam pairwise-interaction waterfill probe-disambiguator. "
            "Design-memo scaffold skeleton — BUILD-1..4 follow-on per design memo."
        )
    )
    parser.add_argument(
        "--acquisition-plan",
        type=Path,
        default=DEFAULT_ACQUISITION_PLAN_PATH,
        help="codex DQS1 acquisition plan JSON path",
    )
    parser.add_argument(
        "--fp64-ledger",
        type=Path,
        default=DEFAULT_FP64_LEDGER_PATH,
        help="600-pair fp64 master-gradient ledger path",
    )
    parser.add_argument(
        "--width-k",
        type=int,
        default=CANONICAL_BEAM_WIDTH_K,
        help="beam width K (canonical 8 per Daubechies multi-scale prior)",
    )
    parser.add_argument(
        "--depth-d",
        type=int,
        default=CANONICAL_BEAM_DEPTH_D,
        help="beam depth D (canonical 4 per Carmack MVP-first 5-step)",
    )
    parser.add_argument(
        "--design-time-skeleton",
        action="store_true",
        default=True,
        help="emit synthetic skeleton (default; BUILD-1 disables this)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON artifact to stdout (canonical Provenance per Catalog #323)",
    )
    args = parser.parse_args(argv)

    # Skeleton verification: confirm acquisition plan schema can be parsed
    if not args.acquisition_plan.exists():
        print(
            f"WARN: acquisition plan not found at {args.acquisition_plan}; "
            "scaffold skeleton smoke-test only",
            file=sys.stderr,
        )
        sample_candidates = [
            PairCandidate(
                pair_index=p,
                rate_score_delta_vs_source_selector=-7e-7,
                predicted_score_mean=0.19202828,
                payload_bytes_delta_vs_source_selector=-1,
                distortion_repair_budget_score=6.66e-7,
            )
            for p in range(0, 32)
        ]
    else:
        sample_candidates = _load_acquisition_candidates(args.acquisition_plan)
        if not sample_candidates:
            print(
                "WARN: no drop_many candidates in acquisition plan; "
                "scaffold skeleton smoke-test only",
                file=sys.stderr,
            )
            sample_candidates = [
                PairCandidate(
                    pair_index=p,
                    rate_score_delta_vs_source_selector=-7e-7,
                    predicted_score_mean=0.19202828,
                    payload_bytes_delta_vs_source_selector=-1,
                    distortion_repair_budget_score=6.66e-7,
                )
                for p in range(0, 32)
            ]

    matrix_payload = build_pairwise_interaction_matrix(
        sample_candidates,
        fp64_ledger_path=args.fp64_ledger,
        synthetic_design_time_only=args.design_time_skeleton,
    )

    skeleton_verdict = {
        "schema": "dqs1_drop_many_beam_pairwise_interaction_waterfill_skeleton.v1",
        "design_memo_path": (
            ".omx/research/dqs1_drop_many_beam_pairwise_interaction_waterfill_"
            "design_memo_20260525.md"
        ),
        "build_status": "SKELETON_ONLY_BUILD_1_TO_4_OPERATOR_ROUTABLE",
        "num_candidates_loaded": len(sample_candidates),
        "p_max": matrix_payload["p_max"],
        "interaction_matrix_metadata": matrix_payload["matrix_metadata"],
        "beam_config": dataclasses.asdict(BeamSearchConfig(width_k=args.width_k, depth_d=args.depth_d)),
        "dykstra_config": dataclasses.asdict(DykstraFeasibilityConfig()),
        "waterfill_config": dataclasses.asdict(WaterfillConfig()),
        "canonical_provenance": CANONICAL_NON_PROMOTABLE_MARKERS,
        "canonical_equation_candidate_id": "dqs1_drop_many_pairwise_interaction_beam_search_v1",
        "operator_routable_next_cascade_priority": [
            "BUILD-1: populate empirical I[P,P] (~2-4h, $0)",
            "BUILD-2: full beam_search_drop_many executable (~4-6h, $0)",
            "BUILD-3: Catalog #356 AxisDecomposition wire-in (~1-2h, $0)",
            "BUILD-4: Catalog #357 Tier B canonical consumer (~2-3h, $0)",
            "DISPATCH: paired CPU+CUDA Modal exact-eval top-K=8 (~$2.40-4)",
        ],
    }

    if args.json:
        json.dump(skeleton_verdict, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print("DQS1 drop-many beam pairwise-interaction waterfill — skeleton verdict")
        print(f"  build_status: {skeleton_verdict['build_status']}")
        print(f"  num_candidates_loaded: {skeleton_verdict['num_candidates_loaded']}")
        print(f"  p_max: {skeleton_verdict['p_max']}")
        print(f"  beam_config: {skeleton_verdict['beam_config']}")
        print("  operator_routable_next_cascade_priority:")
        for action in skeleton_verdict["operator_routable_next_cascade_priority"]:
            print(f"    - {action}")
        print(f"  design_memo: {skeleton_verdict['design_memo_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
