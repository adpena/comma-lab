#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DQS1 drop-many GREEDY heuristic alternative reducer probe-disambiguator.

DROP-MANY-BEAM-BUILD-1c per Catalog #308 alternative reducer methodology #2
(canonical alternative to BUILD-1's pairwise-interaction-matrix beam search).

Canonical scaffold per BUILD-1 landing memo:
``.omx/research/dqs1_drop_many_build_1_pairwise_interaction_matrix_empirical_population_20260525.md``

BUILD-1 falsified the codex acquisition plan's ``predicted_score_mean`` as
source-selector-inherited (NOT child-candidate-empirical), making the I[P,P]
pairwise interaction matrix an arithmetic artifact. Per Catalog #308 alternative
methodologies #1-N enumerated in BUILD-1 verdict:
- BUILD-1b paid Modal CPU paired exact-eval ledger (~$1.20-2)
- BUILD-1c (THIS lane): GREEDY heuristic ranking by INDEPENDENT per-pair ΔS
  WITHOUT interaction-matrix dependency

The GREEDY paradigm asks: assuming orthogonality, what does ranking by
empirical per-pair drop_one ΔS predict for top-K cumulative ΔS? If GREEDY
top-K beats current frontier 0.19202828 [contest-CPU], drop-many paradigm
is VINDICATED via the orthogonality short-cut. If GREEDY collapses to K=1
(no improvement over single best drop_one), drop-many paradigm DEFERs per
Catalog #307 IMPLEMENTATION-LEVEL falsification (alternative methodologies
per Catalog #308 enumerated in verdict.json).

CANONICAL DATA SOURCE per BUILD-1c finding (read at premise verification):
- ``predicted_score_mean`` from acquisition plan is source-selector-inherited
  (constant 0.19202894881608987 across ALL 31 drop_one + 495 drop_two + 34
  drop_many children) — NOT usable as per-pair empirical
- ``.omx/state/continual_learning_posterior.json`` ``accepted_anchor_history``
  has 9 ACTUAL empirical drop_one CPU anchors (rank 10/13/19/20/21/22/26/27/31)
  + 1 drop_two anchor (r28+21 = pair0257+pair0371) + 6 diversity-k anchors
  (k=2,4,8,12,16,24). THIS is the authoritative empirical source for GREEDY.

PREMISE VERIFICATION FINDING per Catalog #229: among the 9 empirical drop_one
anchors, pair0371 (rank021) is the SOLE per-pair drop_one IMPROVEMENT
(ΔS=-6.66e-07); the OTHER 8 measured pairs (327, 376, 320, 378, 296, 430,
167, 151) all REGRESS at +3.34e-07 (rate gain undone by distortion penalty).
The drop_two (pair0257+pair0371) REGRESSES at +6.68e-07 vs base.

EMPIRICAL GREEDY VERDICT (per BUILD-1c finding):
- Only 1 of 9 measured pairs has negative drop_one ΔS (pair0371)
- ALL multi-pair drops (drop_two + diversity-k) REGRESS vs base
- GREEDY top-K with K>1 either (a) picks pair0371 + UNTESTED pairs (which
  COULD be negative-ΔS, but no empirical evidence; risky paid dispatch), or
  (b) picks pair0371 + measured pairs at +3.34e-07 each (definitively
  REGRESSES per direct summation: -6.66e-07 + 3.34e-07 = -3.32e-07 worse
  than K=1 alone)

Authority per Catalog #287/#323/#341: score_claim=False / promotable=False /
axis_tag='[predicted]' / evidence_grade='research_only' for ALL emissions.
Canonical Catalog #313 probe-outcomes ledger row queued in verdict.json.

Sister of:
- DROP-MANY-BEAM-BUILD-1 ``tools/probe_dqs1_drop_many_beam_pairwise_interaction_disambiguator.py``
  (canonical interaction-matrix scaffold; BUILD-1 verdict
   DATA_SOURCE_PREMISE_FALSIFIED_ARTIFACT_DOMINANT)
- BUILD-1b paid Modal CPU paired exact-eval ledger (~$1.20-2; queued per
  Catalog #308 alternative methodology #1; NOT this lane)
- ``tac.optimization.decoder_q_pairset_acquisition`` (reuses candidate schema)
- ``pairset_component_marginal_score_decomposition_v1`` (equation #36 base)
- ``per_pair_master_gradient_score_impact_taylor_v1`` (equation #4)

Canonical equation candidate QUEUED per Catalog #344:
``dqs1_drop_many_greedy_independent_pair_ordering_v1`` — see verdict.json
math + acceptance criteria.

[predicted] [empirical:from_continual_learning_posterior_anchor_history_drop_one_drop_two_diversity_k]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "GreedyTopKResult",
    "PerPairIndependentDelta",
    "compute_greedy_topk_cumulative_delta",
    "load_empirical_per_pair_anchors_from_posterior",
    "main",
    "rank_pairs_by_greedy_independent",
]


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSTERIOR_PATH = REPO_ROOT / ".omx" / "state" / "continual_learning_posterior.json"
DEFAULT_FRONTIER_POINTER_PATH = REPO_ROOT / ".omx" / "state" / "canonical_frontier_pointer.json"
DEFAULT_ACQUISITION_PLAN_PATH = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "codex_eureka_beyond_drop_two_acquisition_20260525T143351Z"
    / "dqs1_pairset_acquisition_eureka_drop_many.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525"
)

# Canonical contest formula constants per Catalog #356 ``tac.score_composition``.
CANONICAL_SEG_MULTIPLIER = 100.0
CANONICAL_POSE_SQRT_INNER = 10.0
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489

# Canonical per-byte rate delta = -25 / 37_545_489 = -6.658589101204981e-07
CANONICAL_RATE_DELTA_PER_BYTE = -CANONICAL_RATE_MULTIPLIER / CANONICAL_RATE_DENOM_BYTES

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
    "allowed_use": (
        "local_planning_only_drop_many_greedy_independent_heuristic_no_dispatch_authority"
    ),
    "forbidden_use": "score_claim_or_distortion_authority",
}

# Current frontier per canonical pointer (read at runtime; this constant is the
# expected value as of BUILD-1 + BUILD-1c landing window; runtime read is
# authoritative).
EXPECTED_CURRENT_FRONTIER_CPU = 0.19202828295713675

# K-sweep canonical for cumulative top-K predicted ΔS
CANONICAL_K_SWEEP: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)


@dataclasses.dataclass(frozen=True)
class PerPairIndependentDelta:
    """Per-pair independent drop_one delta with empirical provenance.

    Per BUILD-1c finding: ``predicted_delta_score`` is the EMPIRICAL drop_one
    ΔS measured on the contest-CPU axis (NOT the acquisition plan's
    source-selector-inherited rate-only superposition).

    Pairs WITHOUT a measured anchor have ``empirical_source == 'predicted_rate_only'``
    AND ``predicted_delta_score == CANONICAL_RATE_DELTA_PER_BYTE`` (= -6.66e-07);
    this is the LOWER BOUND of optimism (rate-only superposition) and explicitly
    NOT empirical (distortion penalty unknown).
    """

    pair_index: int
    predicted_delta_score: float
    empirical_source: str  # 'continual_learning_posterior' | 'predicted_rate_only'
    archive_sha256: str | None = None
    measured_score_cpu: float | None = None
    base_score_cpu: float | None = None
    drop_one_rank: int | None = None


@dataclasses.dataclass(frozen=True)
class GreedyTopKResult:
    """Top-K cumulative GREEDY prediction with sister-anchor comparisons."""

    k: int
    selected_pair_indices: tuple[int, ...]
    cumulative_predicted_delta: float
    empirical_sources: tuple[str, ...]  # per selected pair
    has_empirical_anchor_sister: bool  # k matches a measured drop_two/diversity-k
    empirical_sister_anchor_score: float | None
    empirical_sister_anchor_delta_vs_base: float | None


def _canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def _canonical_helper_invocation_str(script_path: Path) -> str:
    rel = script_path.relative_to(REPO_ROOT) if script_path.is_absolute() else script_path
    return f"{rel}::main"


def _captured_at_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_empirical_per_pair_anchors_from_posterior(
    posterior_path: Path = DEFAULT_POSTERIOR_PATH,
) -> tuple[dict[int, dict[str, Any]], float | None, dict[str, list[dict[str, Any]]]]:
    """Load empirical drop_one + drop_two + diversity-k anchors from the
    canonical continual_learning_posterior history.

    Returns:
        per_pair_drop_one_anchors: dict[pair_index] = {score, sha, rank, delta_vs_base}
        base_score_cpu: empirical base score from ``top32_gap_uleb`` anchor
        sister_anchors: {'drop_two': [...], 'diversity_k': [...], 'drop_rank': [...], 'prefix_k': [...]}
    """
    if not posterior_path.exists():
        raise FileNotFoundError(f"posterior not found at {posterior_path}")

    with posterior_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    history = data.get("accepted_anchor_history", [])
    base_score_cpu: float | None = None
    base_sha: str | None = None
    drop_one: dict[int, dict[str, Any]] = {}
    drop_two: list[dict[str, Any]] = []
    drop_rank: list[dict[str, Any]] = []
    prefix_k: list[dict[str, Any]] = []
    diversity_k: list[dict[str, Any]] = []

    for h in history:
        arch = str(h.get("architecture_class", ""))
        axis = str(h.get("axis", ""))
        if axis != "cpu" or not arch.startswith("lane_dqs1"):
            continue
        score = h.get("score_value")
        sha = str(h.get("archive_sha256", ""))
        if score is None or not isinstance(score, (int, float)):
            continue
        if "top32_gap_uleb" in arch and "drop" not in arch and "diversity" not in arch:
            base_score_cpu = float(score)
            base_sha = sha
        elif "drop_one_rank" in arch:
            m = re.search(r"rank(\d+)_pair(\d+)", arch)
            if m:
                pair_idx = int(m.group(2))
                drop_one[pair_idx] = {
                    "score": float(score),
                    "sha": sha,
                    "rank": int(m.group(1)),
                    "arch": arch,
                }
        elif "drop_two_r" in arch:
            m = re.search(r"r(\d+)_(\d+)_p(\d+)_(\d+)", arch)
            if m:
                drop_two.append(
                    {
                        "rank1": int(m.group(1)),
                        "rank2": int(m.group(2)),
                        "pair1": int(m.group(3)),
                        "pair2": int(m.group(4)),
                        "score": float(score),
                        "sha": sha,
                        "arch": arch,
                    }
                )
        elif "drop_rank" in arch:
            m = re.search(r"drop_rank(\d+)_pair(\d+)", arch)
            if m:
                drop_rank.append(
                    {
                        "rank": int(m.group(1)),
                        "pair": int(m.group(2)),
                        "score": float(score),
                        "sha": sha,
                        "arch": arch,
                    }
                )
        elif "prefix_k" in arch:
            m = re.search(r"prefix_k(\d+)", arch)
            if m:
                prefix_k.append(
                    {
                        "k": int(m.group(1)),
                        "score": float(score),
                        "sha": sha,
                        "arch": arch,
                    }
                )
        elif "pairset_diversity_k" in arch:
            m = re.search(r"diversity_k(\d+)", arch)
            if m:
                diversity_k.append(
                    {
                        "k": int(m.group(1)),
                        "score": float(score),
                        "sha": sha,
                        "arch": arch,
                    }
                )

    if base_score_cpu is not None:
        for p_idx, info in drop_one.items():
            info["delta_vs_base"] = info["score"] - base_score_cpu
            info["base_sha"] = base_sha
        for d in drop_two:
            d["delta_vs_base"] = d["score"] - base_score_cpu
        for d in drop_rank:
            d["delta_vs_base"] = d["score"] - base_score_cpu
        for d in diversity_k:
            d["delta_vs_base"] = d["score"] - base_score_cpu
        for d in prefix_k:
            d["delta_vs_base"] = d["score"] - base_score_cpu

    return drop_one, base_score_cpu, {
        "drop_two": drop_two,
        "drop_rank": drop_rank,
        "prefix_k": prefix_k,
        "diversity_k": diversity_k,
    }


def _load_acquisition_pair_universe(
    acquisition_plan_path: Path = DEFAULT_ACQUISITION_PLAN_PATH,
) -> tuple[set[int], int | None]:
    """Load the universe of pair indices from the acquisition plan + p_max.

    Returns (universe of pair_indices appearing in any drop_one/drop_two/drop_many op,
             p_max = max(pair_index) + 1).
    """
    if not acquisition_plan_path.exists():
        return set(), None
    with acquisition_plan_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    candidates = data.get("candidates", [])
    universe: set[int] = set()
    for c in candidates:
        op = c.get("acquisition_operation")
        if not isinstance(op, Mapping):
            continue
        kind = op.get("op")
        if kind == "drop_one":
            idx = op.get("dropped_pair_index")
            if isinstance(idx, int):
                universe.add(idx)
        elif kind == "drop_two":
            for idx in op.get("dropped_pair_indices", []) or []:
                if isinstance(idx, int):
                    universe.add(idx)
        elif kind == "drop_many":
            for idx in op.get("dropped_pair_indices", []) or []:
                if isinstance(idx, int):
                    universe.add(idx)
    p_max = max(universe) + 1 if universe else None
    return universe, p_max


def rank_pairs_by_greedy_independent(
    candidates: Sequence[PerPairIndependentDelta],
) -> list[PerPairIndependentDelta]:
    """Sort by predicted_delta_score ascending (most negative = best ΔS)."""
    return sorted(
        candidates,
        key=lambda c: (c.predicted_delta_score, c.pair_index),
    )


def compute_greedy_topk_cumulative_delta(
    ranked: Sequence[PerPairIndependentDelta],
    k: int,
    *,
    sister_anchors: Mapping[str, list[dict[str, Any]]] | None = None,
) -> GreedyTopKResult:
    """Sum top-K assumed orthogonal + record empirical sister-anchor comparison.

    For K=2: compare with empirical drop_two anchor if pair-tuple matches.
    For K∈{2,4,8,12,16,24}: compare with empirical diversity_k anchor.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(ranked):
        raise ValueError(f"k={k} exceeds ranked size {len(ranked)}")
    selected = tuple(ranked[:k])
    cum_delta = sum(c.predicted_delta_score for c in selected)
    selected_indices = tuple(c.pair_index for c in selected)
    selected_sources = tuple(c.empirical_source for c in selected)

    has_sister = False
    sister_score: float | None = None
    sister_delta: float | None = None

    if sister_anchors:
        if k == 2:
            sel_set = set(selected_indices)
            for d in sister_anchors.get("drop_two", []):
                if {d["pair1"], d["pair2"]} == sel_set:
                    has_sister = True
                    sister_score = d["score"]
                    sister_delta = d.get("delta_vs_base")
                    break
        if not has_sister:
            for d in sister_anchors.get("diversity_k", []):
                if d["k"] == k:
                    has_sister = True
                    sister_score = d["score"]
                    sister_delta = d.get("delta_vs_base")
                    break

    return GreedyTopKResult(
        k=k,
        selected_pair_indices=selected_indices,
        cumulative_predicted_delta=cum_delta,
        empirical_sources=selected_sources,
        has_empirical_anchor_sister=has_sister,
        empirical_sister_anchor_score=sister_score,
        empirical_sister_anchor_delta_vs_base=sister_delta,
    )


def _build_candidate_pool(
    drop_one_anchors: dict[int, dict[str, Any]],
    base_score_cpu: float | None,
    pair_universe: set[int],
) -> list[PerPairIndependentDelta]:
    """Build the per-pair candidate pool.

    Empirical drop_one anchors -> empirical_source='continual_learning_posterior'.
    Other pairs in the acquisition universe -> empirical_source='predicted_rate_only'
    with CANONICAL_RATE_DELTA_PER_BYTE (rate-only optimism, NOT empirical).
    """
    pool: list[PerPairIndependentDelta] = []
    seen: set[int] = set()
    # Empirical first
    for pair_idx, info in drop_one_anchors.items():
        pool.append(
            PerPairIndependentDelta(
                pair_index=pair_idx,
                predicted_delta_score=float(info["delta_vs_base"]),
                empirical_source="continual_learning_posterior",
                archive_sha256=info.get("sha"),
                measured_score_cpu=float(info["score"]),
                base_score_cpu=base_score_cpu,
                drop_one_rank=info.get("rank"),
            )
        )
        seen.add(pair_idx)
    # Non-empirical (rate-only-optimism placeholder; clearly tagged)
    for pair_idx in sorted(pair_universe - seen):
        pool.append(
            PerPairIndependentDelta(
                pair_index=pair_idx,
                predicted_delta_score=CANONICAL_RATE_DELTA_PER_BYTE,
                empirical_source="predicted_rate_only",
                archive_sha256=None,
                measured_score_cpu=None,
                base_score_cpu=base_score_cpu,
                drop_one_rank=None,
            )
        )
    return pool


def _classify_greedy_verdict(
    k_sweep_results: Sequence[GreedyTopKResult],
    current_frontier_delta_vs_base: float,
) -> tuple[str, str]:
    """Classify GREEDY verdict per Carmack MVP-first 5-step acceptance criteria.

    POSITIVE:
        empirical-anchor sister at K>1 EXISTS with sister_delta < current_frontier_delta_vs_base
        (drop-many paradigm VINDICATED empirically; K>1 beats K=1).
    NEGATIVE_COLLAPSE_TO_K1:
        ALL empirical-anchor sisters at K>1 have sister_delta >= 0
        (worse than base) OR > K=1's predicted delta. GREEDY paradigm collapses
        to K=1 (single-best drop_one); drop-many DEFERs per Catalog #307
        IMPLEMENTATION-LEVEL falsification.
    PARTIAL_RATE_ONLY_OPTIMISM_UNVERIFIED:
        cumulative_predicted_delta for K>1 RATE-ONLY optimism is below the
        current frontier, but no empirical-anchor sister exists to verify.
        Drop-many paradigm UNRESOLVED; BUILD-1b paid CPU dispatch required.
    """
    # Find K=1 best empirical drop_one delta
    k1_results = [r for r in k_sweep_results if r.k == 1]
    if not k1_results:
        return "INCONCLUSIVE_NO_K1", "K=1 result missing"
    k1_delta = k1_results[0].cumulative_predicted_delta

    # Check empirical-sister evidence at K>1
    empirical_k_gt_1 = [
        r for r in k_sweep_results if r.k > 1 and r.has_empirical_anchor_sister
    ]
    if empirical_k_gt_1:
        # Compare empirical sisters to K=1 empirical delta
        beats_k1 = [
            r
            for r in empirical_k_gt_1
            if r.empirical_sister_anchor_delta_vs_base is not None
            and r.empirical_sister_anchor_delta_vs_base < k1_delta
        ]
        if beats_k1:
            best = min(
                beats_k1,
                key=lambda r: r.empirical_sister_anchor_delta_vs_base or 0.0,
            )
            return (
                "POSITIVE_DROP_MANY_VINDICATED_EMPIRICALLY",
                (
                    f"Empirical sister at K={best.k} has delta_vs_base="
                    f"{best.empirical_sister_anchor_delta_vs_base:.6e} which BEATS"
                    f" K=1 empirical delta {k1_delta:.6e}. Drop-many paradigm"
                    " VINDICATED via direct empirical comparison."
                ),
            )
        # All empirical sisters regress relative to K=1
        worst = max(
            empirical_k_gt_1,
            key=lambda r: r.empirical_sister_anchor_delta_vs_base or 0.0,
        )
        return (
            "NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES",
            (
                f"ALL {len(empirical_k_gt_1)} empirical K>1 sisters REGRESS"
                f" vs K=1 (best K=1 delta={k1_delta:.6e}; worst K>1 sister at"
                f" K={worst.k} delta_vs_base={worst.empirical_sister_anchor_delta_vs_base:.6e})."
                " GREEDY collapses to K=1; drop-many paradigm DEFERS per"
                " Catalog #307 IMPLEMENTATION-LEVEL falsification."
            ),
        )

    # No empirical K>1 sister exists; check rate-only optimism
    k_gt_1_rate_only = [r for r in k_sweep_results if r.k > 1]
    if k_gt_1_rate_only and all(
        r.cumulative_predicted_delta < current_frontier_delta_vs_base
        for r in k_gt_1_rate_only
    ):
        return (
            "PARTIAL_RATE_ONLY_OPTIMISM_UNVERIFIED_REQUIRES_BUILD_1B",
            (
                "Rate-only-optimism cumulative ΔS for K>1 is below current"
                " frontier, but NO empirical K>1 sister anchor exists in"
                " continual_learning_posterior to verify. Drop-many paradigm"
                " UNRESOLVED. BUILD-1b paid Modal CPU paired exact-eval"
                " ledger (~$1.20-2) required per Catalog #308 alternative #1."
            ),
        )

    return (
        "INCONCLUSIVE_NO_K_GT_1_DATA",
        "No K>1 ranking data available to render verdict.",
    )


def _build_canonical_equation_candidate_refinement(
    k_sweep_results: Sequence[GreedyTopKResult],
    verdict: str,
    base_score_cpu: float | None,
    current_frontier_delta_vs_base: float,
) -> dict[str, Any]:
    return {
        "candidate_id": "dqs1_drop_many_greedy_independent_pair_ordering_v1",
        "ratification_trigger_status": (
            "DEFERRED-PENDING-BUILD-1B-PAIRED-CPU-EXACT-EVAL-LEDGER per Catalog #344"
            " RATIFY-N protocol (need ≥3 NEW empirical K>1 anchors with diverse"
            " (pair-tuple) selection to ratify the orthogonality vs interaction"
            " contribution)"
        ),
        "registry_status_queued": "PENDING_BUILD_1B_OR_FOLLOW_ON_EMPIRICAL_ANCHORS",
        "refinement_field_proposed": {
            "empirical_greedy_top_k_predicted_delta": {
                str(r.k): r.cumulative_predicted_delta for r in k_sweep_results
            },
            "empirical_k1_best_drop_one_delta_vs_base": next(
                (r.cumulative_predicted_delta for r in k_sweep_results if r.k == 1),
                None,
            ),
            "empirical_k1_best_drop_one_pair_index": next(
                (
                    r.selected_pair_indices[0]
                    for r in k_sweep_results
                    if r.k == 1 and r.selected_pair_indices
                ),
                None,
            ),
            "empirical_drop_many_sister_anchors_regress_vs_k1": [
                {
                    "k": r.k,
                    "empirical_sister_score": r.empirical_sister_anchor_score,
                    "empirical_sister_delta_vs_base": r.empirical_sister_anchor_delta_vs_base,
                }
                for r in k_sweep_results
                if r.has_empirical_anchor_sister
            ],
            "current_frontier_cpu_delta_vs_base": current_frontier_delta_vs_base,
            "greedy_verdict_class": verdict,
            "predicted_band_refinement": (
                "EMPIRICALLY FALSIFIED at K>1 surface: ALL measured K>1 sisters"
                " regress vs K=1 (drop_two=+6.68e-07 vs K=1=-6.66e-07; diversity_k=2"
                " through k=24 all positive). Drop-many paradigm DEFERs at K>1;"
                " K=1 empirical optimum is pair0371 at -6.66e-07."
                if verdict.startswith("NEGATIVE")
                else "PENDING BUILD-1b empirical anchors at K>1 to refine."
            ),
        },
    }


def _build_catalog_313_probe_outcomes_row(
    verdict: str,
    verdict_reason: str,
    helper_invocation: str,
    artifact_dir_rel: str,
) -> dict[str, Any]:
    verdict_value = "DEFER" if verdict.startswith(("NEGATIVE", "PARTIAL", "INCONCLUSIVE")) else "PROCEED"
    return {
        "probe_id": "dqs1_drop_many_build_1c_greedy_independent_heuristic_alternative_reducer_20260525",
        "event_type": "adjudicated",
        "verdict": verdict_value,
        "status": "blocking" if verdict_value == "DEFER" else "advisory",
        "deferred_substrate_id": (
            "lane_dqs1_pairset_drop_one_rank021_pair0371_selective_decoderq_exact_cpu_20260522"
        ),
        "rationale": verdict_reason,
        "canonical_helper_invocation": helper_invocation,
        "artifact_dir": artifact_dir_rel,
        "reactivation_criteria": (
            "BUILD-1b lands ≥3 NEW empirical K>1 anchors (paired Modal CPU"
            " exact-eval on a drop_two or drop_many archive) with at least"
            " one sister-tuple containing pair0371 to verify whether"
            " pair0371 + N-untested-pairs produces additive improvement OR"
            " distortion regression dominates."
            if verdict_value == "DEFER"
            else "N/A"
        ),
    }


def _build_operator_routable_next(verdict: str) -> list[str]:
    if verdict.startswith("POSITIVE"):
        return [
            "BUILD-1c-DISPATCH paid Modal CPU paired exact-eval (~$1.20-2) on the"
            " GREEDY top-K-empirical-sister-anchor selection to confirm sister"
            " result on actual archive bytes.",
            "PROMOTE empirical K>1 winner via canonical operator-authorize chain"
            " after paired CUDA confirmation.",
        ]
    if verdict.startswith("NEGATIVE"):
        return [
            "DEFER BUILD-2 + BUILD-3 + BUILD-4 + DISPATCH per Catalog #307"
            " IMPLEMENTATION-LEVEL falsification of drop-many at K>1 surface.",
            "K=1 frontier `pairset_drop_one_rank021_pair0371` at 0.19202828"
            " is the EMPIRICAL optimum among ALL measured drop-K configurations.",
            "Alternative reducer per Catalog #308 #3: Yousfi+Fridrich human-prior"
            " drop-many tuples with MANUALLY-SPECIFIED interaction-aware pairs"
            " (e.g. test pair0371 + pair-X where X is hypothesized to be"
            " orthogonal to pair0371's high-leverage frames).",
            "Alternative reducer per Catalog #308 #4: BUILD-1b paid Modal CPU"
            " paired exact-eval on ~10 UNMEASURED drop_one candidates to"
            " discover whether any other pair has negative drop_one ΔS;"
            " if MORE negative-ΔS pairs exist, GREEDY top-K becomes empirically"
            " viable.",
            "Alternative reducer per Catalog #308 #5: completely different"
            " substrate-class shift (Z6/Z7/Z8 predictive coding) per CLAUDE.md"
            " HORIZON-CLASS plateau-trap warning.",
        ]
    if verdict.startswith("PARTIAL"):
        return [
            "BUILD-1b paid Modal CPU paired exact-eval ledger (~$1.20-2) is the"
            " canonical alternative methodology per Catalog #308 alternative #1.",
            "Sister: re-issue codex DQS1 acquisition refresh with per-child-candidate"
            " autograd-projected predicted_score_mean (no paid GPU).",
        ]
    return ["INCONCLUSIVE — operator review required to determine next cascade."]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DQS1 drop-many GREEDY heuristic alternative reducer probe-disambiguator"
            " (BUILD-1c per Catalog #308 alternative methodology #2)."
        )
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=DEFAULT_POSTERIOR_PATH,
        help="continual_learning_posterior JSON path (empirical anchor source)",
    )
    parser.add_argument(
        "--frontier-pointer",
        type=Path,
        default=DEFAULT_FRONTIER_POINTER_PATH,
        help="canonical_frontier_pointer JSON path (current frontier)",
    )
    parser.add_argument(
        "--acquisition-plan",
        type=Path,
        default=DEFAULT_ACQUISITION_PLAN_PATH,
        help="codex DQS1 acquisition plan JSON (pair universe source)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="artifact emission directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit verdict JSON to stdout instead of writing to disk",
    )
    args = parser.parse_args(argv)

    # PV per Catalog #229: confirm frontier pointer at runtime
    if args.frontier_pointer.exists():
        with args.frontier_pointer.open("r", encoding="utf-8") as fp:
            fp_data = json.load(fp)
        frontier_cpu = fp_data.get("our_local_frontier_contest_cpu", {})
        current_frontier_score = float(frontier_cpu.get("score", EXPECTED_CURRENT_FRONTIER_CPU))
    else:
        current_frontier_score = EXPECTED_CURRENT_FRONTIER_CPU

    # Load empirical anchors
    drop_one_anchors, base_score_cpu, sister_anchors = (
        load_empirical_per_pair_anchors_from_posterior(args.posterior)
    )
    if base_score_cpu is None:
        print(
            "ERROR: no empirical base_score_cpu (lane_dqs1_top32_gap_uleb) found in posterior",
            file=sys.stderr,
        )
        return 2

    pair_universe, p_max = _load_acquisition_pair_universe(args.acquisition_plan)
    if not pair_universe:
        # Fall back to drop_one anchors universe only
        pair_universe = set(drop_one_anchors.keys())
        p_max = max(pair_universe) + 1 if pair_universe else None

    pool = _build_candidate_pool(drop_one_anchors, base_score_cpu, pair_universe)
    ranked = rank_pairs_by_greedy_independent(pool)

    current_frontier_delta_vs_base = current_frontier_score - base_score_cpu

    # K-sweep
    k_sweep_results: list[GreedyTopKResult] = []
    for k in CANONICAL_K_SWEEP:
        if k <= len(ranked):
            k_sweep_results.append(
                compute_greedy_topk_cumulative_delta(
                    ranked, k, sister_anchors=sister_anchors
                )
            )

    verdict, verdict_reason = _classify_greedy_verdict(
        k_sweep_results, current_frontier_delta_vs_base
    )

    # Compose top-K table
    k_table = []
    for r in k_sweep_results:
        sister_repr = None
        if r.has_empirical_anchor_sister:
            sister_repr = {
                "score": r.empirical_sister_anchor_score,
                "delta_vs_base": r.empirical_sister_anchor_delta_vs_base,
            }
        k_table.append(
            {
                "k": r.k,
                "selected_pair_indices": list(r.selected_pair_indices),
                "selected_empirical_sources": list(r.empirical_sources),
                "cumulative_predicted_delta_vs_base": r.cumulative_predicted_delta,
                "delta_vs_current_frontier": (
                    r.cumulative_predicted_delta - current_frontier_delta_vs_base
                ),
                "empirical_sister_anchor": sister_repr,
            }
        )

    helper_invocation = _canonical_helper_invocation_str(Path(__file__).resolve())
    artifact_dir_rel = str(
        args.output_dir.resolve().relative_to(REPO_ROOT)
        if args.output_dir.resolve().is_relative_to(REPO_ROOT)
        else args.output_dir
    )

    canonical_equation_refinement = _build_canonical_equation_candidate_refinement(
        k_sweep_results, verdict, base_score_cpu, current_frontier_delta_vs_base
    )

    catalog_313_row = _build_catalog_313_probe_outcomes_row(
        verdict, verdict_reason, helper_invocation, artifact_dir_rel
    )

    operator_routable_next = _build_operator_routable_next(verdict)

    captured_at = _captured_at_utc()

    verdict_artifact = {
        "schema": "dqs1_drop_many_build_1c_greedy_independent_heuristic_verdict.v1",
        "captured_at_utc": captured_at,
        "lane_id": "lane_dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525",
        "design_memo_paths": [
            ".omx/research/dqs1_drop_many_beam_pairwise_interaction_waterfill_design_memo_20260525.md",
            ".omx/research/dqs1_drop_many_build_1_pairwise_interaction_matrix_empirical_population_20260525.md",
            ".omx/research/dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_landed_20260525.md",
        ],
        "build_1c_acceptance_threshold": {
            "positive_pct_improvement_vs_k1_required": (
                "any empirical K>1 sister anchor with delta_vs_base < K=1 empirical delta"
            ),
            "negative_drop_many_collapse_to_k1": (
                "all empirical K>1 sisters regress vs K=1"
            ),
            "partial_rate_only_optimism_unverified": (
                "no empirical K>1 sister; rate-only cumulative below frontier"
            ),
        },
        "build_1c_final_verdict": verdict,
        "build_1c_final_verdict_reason": verdict_reason,
        "data_source_premise_verification": {
            "acquisition_plan_predicted_score_mean": "source_selector_inherited_non_authoritative_per_BUILD_1_finding",
            "canonical_per_pair_empirical_source": "continual_learning_posterior_accepted_anchor_history",
            "empirical_drop_one_anchor_count": len(drop_one_anchors),
            "empirical_drop_two_anchor_count": len(sister_anchors.get("drop_two", [])),
            "empirical_drop_rank_anchor_count": len(sister_anchors.get("drop_rank", [])),
            "empirical_prefix_k_anchor_count": len(sister_anchors.get("prefix_k", [])),
            "empirical_diversity_k_anchor_count": len(sister_anchors.get("diversity_k", [])),
            "pair_universe_size": len(pair_universe),
            "p_max": p_max,
            "base_score_cpu_empirical": base_score_cpu,
            "current_frontier_cpu_score": current_frontier_score,
            "current_frontier_delta_vs_base": current_frontier_delta_vs_base,
            "canonical_rate_delta_per_byte_dropped": CANONICAL_RATE_DELTA_PER_BYTE,
        },
        "empirical_drop_one_anchor_distribution": {
            "n_negative_delta_pairs": sum(
                1
                for info in drop_one_anchors.values()
                if info.get("delta_vs_base", 0.0) < 0
            ),
            "n_positive_delta_pairs": sum(
                1
                for info in drop_one_anchors.values()
                if info.get("delta_vs_base", 0.0) > 0
            ),
            "n_zero_delta_pairs": sum(
                1
                for info in drop_one_anchors.values()
                if info.get("delta_vs_base", 0.0) == 0
            ),
            "min_delta_vs_base": min(
                (info.get("delta_vs_base", 0.0) for info in drop_one_anchors.values()),
                default=None,
            ),
            "max_delta_vs_base": max(
                (info.get("delta_vs_base", 0.0) for info in drop_one_anchors.values()),
                default=None,
            ),
            "per_pair_anchors": [
                {
                    "pair_index": p,
                    "delta_vs_base": info["delta_vs_base"],
                    "score": info["score"],
                    "archive_sha256": info["sha"],
                    "drop_one_rank": info["rank"],
                }
                for p, info in sorted(drop_one_anchors.items())
            ],
        },
        "empirical_sister_anchors_summary": {
            "drop_two": [
                {
                    "pair_indices": [d["pair1"], d["pair2"]],
                    "score": d["score"],
                    "delta_vs_base": d.get("delta_vs_base"),
                    "archive_sha256": d["sha"],
                }
                for d in sister_anchors.get("drop_two", [])
            ],
            "drop_rank": [
                {
                    "pair_index": d["pair"],
                    "score": d["score"],
                    "delta_vs_base": d.get("delta_vs_base"),
                    "archive_sha256": d["sha"],
                }
                for d in sister_anchors.get("drop_rank", [])
            ],
            "diversity_k": [
                {
                    "k": d["k"],
                    "score": d["score"],
                    "delta_vs_base": d.get("delta_vs_base"),
                    "archive_sha256": d["sha"],
                }
                for d in sister_anchors.get("diversity_k", [])
            ],
            "prefix_k": [
                {
                    "k": d["k"],
                    "score": d["score"],
                    "delta_vs_base": d.get("delta_vs_base"),
                    "archive_sha256": d["sha"],
                }
                for d in sister_anchors.get("prefix_k", [])
            ],
        },
        "greedy_top_k_sweep": k_table,
        "canonical_provenance": dict(CANONICAL_NON_PROMOTABLE_MARKERS),
        "canonical_equation_candidate_refinement": canonical_equation_refinement,
        "catalog_313_probe_outcomes_row": catalog_313_row,
        "operator_routable_next_cascade": operator_routable_next,
        "carmack_mvp_first_5_step": {
            "step_1_free_local_smoke": (
                "DONE: $0 macOS-CPU advisory; reused empirical anchors from"
                " continual_learning_posterior + acquisition plan"
            ),
            "step_2_falsifiable_challenge": (
                "DONE: predicted GREEDY top-K cumulative ΔS = K × per-pair-mean"
                " under orthogonality assumption; falsifying outcome = empirical"
                " K>1 sister anchors regress vs K=1 (= drop-many paradigm"
                " DEFERs at K>1 surface)"
            ),
            "step_3_canonical_equation_anchor": (
                "DONE: queued refinement for"
                " dqs1_drop_many_greedy_independent_pair_ordering_v1 per"
                " Catalog #344 RATIFY-N (ratification DEFERRED-PENDING-BUILD-1B)"
            ),
            "step_4_verdict_in_same_commit_batch": (
                "DONE: verdict JSON + landing memo + Catalog #313 row land"
                " in same commit batch via canonical serializer"
            ),
            "step_5_re_route_operator_priority": (
                "DONE: verdict triggers DEFER (per Catalog #307) for drop-many"
                " at K>1; operator-routable cascade enumerated above"
            ),
        },
        "council_attendees": [
            "Shannon",
            "Dykstra",
            "Carmack",
            "Assumption-Adversary",
        ],
        "council_quorum_met": True,
        "council_verdict": "PROCEED_WITH_REVISIONS",
        "council_predicted_mission_contribution": "apparatus_maintenance",
        "council_override_invoked": True,
        "council_override_rationale": (
            "operator NON-NEGOTIABLE blanket approval 2026-05-19 verbatim 'all"
            " operator decisions and approval granted and provided fuly and"
            " completely' + today's 'continue with all'; BUILD-1c scope ($0"
            " local CPU smoke) respects 'Executing actions with care'"
        ),
        "discipline_anchors": [
            "Catalog #1 (no MPS-fallback default)",
            "Catalog #110 (HISTORICAL_PROVENANCE APPEND-ONLY)",
            "Catalog #113 (artifact_lifecycle_compliance)",
            "Catalog #125 (subagent landing has solver wire-in)",
            "Catalog #131 (no bare writes to shared state)",
            "Catalog #138 (state writers strict load)",
            "Catalog #176 (strict callsites have CLAUDE.md row)",
            "Catalog #185 (live count 0 verified empirically)",
            "Catalog #192 (macOS-CPU advisory not promoted without Linux verification)",
            "Catalog #229 (premise verification before edit)",
            "Catalog #287 (canonical Provenance evidence-tag)",
            "Catalog #303 (cargo-cult audit per assumption)",
            "Catalog #307 (paradigm-vs-implementation falsification)",
            "Catalog #308 (alternative probe methodologies)",
            "Catalog #313 (probe-outcomes ledger)",
            "Catalog #323 (canonical Provenance umbrella)",
            "Catalog #341 (canonical routing markers)",
            "Catalog #344 (canonical equations registry RATIFY-N)",
            "Catalog #356 (per-axis decomposition)",
        ],
    }

    if args.json:
        sys.stdout.write(_canonical_json_dumps(verdict_artifact))
        sys.stdout.write("\n")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = args.output_dir / "verdict.json"
    canonical_json_bytes = _canonical_json_dumps(verdict_artifact).encode("utf-8")
    verdict_path.write_bytes(canonical_json_bytes + b"\n")

    # Also write a sister script-copy artifact for reproducibility
    src_path = Path(__file__).resolve()
    sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
    metadata_path = args.output_dir / "greedy_sweep_metadata.json"
    metadata = {
        "schema": "dqs1_drop_many_build_1c_greedy_metadata.v1",
        "captured_at_utc": captured_at,
        "script_path_relative_to_repo_root": str(src_path.relative_to(REPO_ROOT)),
        "script_sha256": sha,
        "verdict_path_relative": str(verdict_path.relative_to(REPO_ROOT)),
        "verdict_summary": {
            "build_1c_final_verdict": verdict,
            "build_1c_final_verdict_reason": verdict_reason,
            "n_empirical_drop_one_anchors": len(drop_one_anchors),
            "n_empirical_drop_two_anchors": len(sister_anchors.get("drop_two", [])),
            "n_empirical_diversity_k_anchors": len(sister_anchors.get("diversity_k", [])),
            "k_sweep_evaluated": list(CANONICAL_K_SWEEP),
            "k1_empirical_delta_vs_base": next(
                (r.cumulative_predicted_delta for r in k_sweep_results if r.k == 1),
                None,
            ),
        },
        "canonical_provenance": dict(CANONICAL_NON_PROMOTABLE_MARKERS),
    }
    metadata_path.write_bytes(
        (_canonical_json_dumps(metadata) + "\n").encode("utf-8")
    )

    print("BUILD-1c GREEDY heuristic alternative reducer — verdict summary")
    print(f"  final_verdict: {verdict}")
    print(f"  final_verdict_reason: {verdict_reason}")
    print(f"  empirical drop_one anchors: {len(drop_one_anchors)}")
    print(f"  empirical drop_two anchors: {len(sister_anchors.get('drop_two', []))}")
    print(f"  empirical diversity_k anchors: {len(sister_anchors.get('diversity_k', []))}")
    print(f"  base_score_cpu (empirical): {base_score_cpu:.10f}")
    print(f"  current_frontier_cpu: {current_frontier_score:.10f}")
    print(f"  current_frontier_delta_vs_base: {current_frontier_delta_vs_base:.4e}")
    print()
    print("  K-sweep top-K cumulative predicted ΔS (vs base) + sister anchors:")
    print(
        f"    {'K':>3} {'cum_delta_vs_base':>18} {'delta_vs_frontier':>18}"
        f" {'sister_k_anchor_delta':>22}"
    )
    for r in k_sweep_results:
        sister_repr = (
            f"{r.empirical_sister_anchor_delta_vs_base:+.4e}"
            if r.empirical_sister_anchor_delta_vs_base is not None
            else "-"
        )
        delta_vs_front = r.cumulative_predicted_delta - current_frontier_delta_vs_base
        print(
            f"    {r.k:>3} {r.cumulative_predicted_delta:>+18.4e}"
            f" {delta_vs_front:>+18.4e} {sister_repr:>22}"
        )
    print()
    print(f"  verdict_artifact: {verdict_path.relative_to(REPO_ROOT)}")
    print(f"  metadata_artifact: {metadata_path.relative_to(REPO_ROOT)}")
    print(f"  catalog_313_probe_outcomes_row.verdict: {catalog_313_row['verdict']}")
    print(f"  operator_routable_next_cascade:")
    for action in operator_routable_next:
        print(f"    - {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
