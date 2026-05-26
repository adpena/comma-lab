# SPDX-License-Identifier: MIT
"""META-LIFT-1 cross-substrate master-gradient pattern exploitation analyzer.

Per operator META-critique 2026-05-26 verbatim: *"Were those as fractal
optimized as possible? We are making progress but still too leaf and low level
when we should be exploiting patterns from math"*.

Today's Cascade A FEC10 + Cascade C' + UNIWARD 7th-order + FEC8 paired-axis
landings are LEAF-level: per-substrate master-gradient anchors are extracted
and consumed substrate-by-substrate via Catalog #354 8-exploit consumers
(``master_gradient_aggregate_consumer`` / ``master_gradient_per_pair_consumer``
/ ``cross_substrate_similarity_consumer`` / etc.). The empirical ledger at
``.omx/state/master_gradient_anchors.jsonl`` carries 11 rows across multiple
archives, but there is NO single canonical analyzer that exploits the
cross-substrate Cauchy-Schwarz + Taylor expansion META-pattern to identify
which substrate × which byte-region has the highest leverage IN AGGREGATE
across the full archive corpus.

This module is the META-LIFT (per the 11th standing directive ORDER
discipline: ONE canonical analyzer ACROSS substrates FIRST, then per-substrate
consumption SECOND). It loads canonical master-gradient anchors per substrate
via :mod:`tac.master_gradient_consumers` (NEVER duplicating producer logic
per Catalog #230 sister-disjoint) and emits a ranked cross-substrate
byte-saving opportunity list bounded by the Cauchy-Schwarz inequality.

Mathematical grounding (per canonical equation #344 family):

  Per-substrate Taylor expansion (canonical equation
  ``per_pair_master_gradient_score_impact_taylor_v1``):

    ΔS_substrate_i ≈ <∇S_i, Δθ_i> + (1/2) <Δθ_i, H_i Δθ_i> + O(||Δθ||³)

  where ``∇S_i`` is the per-byte master-gradient for substrate i (loaded
  via :func:`tac.master_gradient_consumers.load_aggregate_gradient_from_anchor`)
  and ``Δθ_i`` is the byte-perturbation vector for that substrate.

  Aggregate cross-substrate bound (canonical equation
  ``cross_substrate_master_gradient_aggregate_ranking_taylor_savings_v1``;
  FORMALIZATION_PENDING per Catalog #344 until paired-CUDA empirical
  anchor lands):

    ΔS_aggregate ≤ Σ_i ||∇S_i||_2 · ||Δθ_i||_2   (Cauchy-Schwarz upper bound)

  Per-axis decomposition (per Catalog #356 + CLAUDE.md "SegNet vs PoseNet
  importance — operating-point dependent"):

    ΔS = 100·Δd_seg + sqrt(10·d_pose_new) - sqrt(10·d_pose_old) + 25·Δarchive_bytes/N

  so per-axis ranking differs in magnitude AND sign across operating points;
  the analyzer surfaces per-axis ranked opportunities separately and computes
  the canonical contest-score composition via :func:`tac.score_composition.compose_score_from_axes`.

The analyzer is OBSERVABILITY-ONLY by construction (per Catalog #341
non-promotable routing markers + CLAUDE.md "Apples-to-apples evidence
discipline"): every emitted opportunity carries ``axis_tag=[predicted]`` +
``score_claim=False`` + ``promotable=False``. Promotion of a cross-substrate
ranking to a contest score signal REQUIRES paired-CUDA empirical anchor per
CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiable.

Sister architecture (Catalog #230 sister-disjoint):

  - :mod:`tac.master_gradient` — canonical authority helper (READ-ONLY here)
  - :mod:`tac.master_gradient_consumers` — per-substrate consumer surfaces
    (READ-ONLY here; ``load_aggregate_gradient_from_anchor`` /
    ``load_per_pair_gradient_from_anchor`` consumed)
  - :mod:`tac.cathedral_consumers.master_gradient_aggregate_consumer` (Catalog
    #354 exploit #1) — Tier A observability-only per-substrate annotation
  - :mod:`tac.cathedral_consumers.cross_substrate_similarity_consumer` —
    SUB_ADDITIVE / SUPER_ADDITIVE / ANTAGONISTIC classification matrix
    (sister surface; #356 different axis)
  - :mod:`tac.cathedral_consumers.cross_substrate_master_gradient_analyzer_consumer`
    (THIS landing) — Tier A observability-only annotation that surfaces
    cross-substrate ranked opportunities for the cathedral autopilot ranker

Per-axis decomposition is enabled per Catalog #356 (consumers MAY emit
``predicted_axis_decomposition`` when per-axis signal is available); the
analyzer's output carries per-axis Taylor projections so downstream Pareto +
bit-allocator consumers can route per-axis without recomputation.

Canonical fcntl-locked JSONL ledger at
``.omx/state/cross_substrate_master_gradient_analyses.jsonl`` per
Catalog #131/#138/#245 sister discipline. APPEND-ONLY per
Catalog #110/#113 HISTORICAL_PROVENANCE.

The 6-hook wire-in declaration per Catalog #125:

  * Hook #1 SENSITIVITY_MAP — ACTIVE (cross-substrate ranked
    opportunities surface aggregate per-byte sensitivity across substrates)
  * Hook #2 PARETO_CONSTRAINT — ACTIVE (Cauchy-Schwarz upper bound is the
    canonical Pareto feasibility boundary; future Dykstra alternating
    projections per Dim 1 Phase 4 can consume the canonical bound)
  * Hook #3 BIT_ALLOCATOR — ACTIVE (per-axis ranked opportunities feed
    bit allocator priority cascade per Dim 6 Step 6.5)
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE (sister consumer
    auto-discovered per Catalog #335/#336/#337)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (per-analysis canonical
    posterior anchor via ``append_analysis_locked``; sister of
    ``master_gradient.append_anchor_locked``)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (the analyzer IS the canonical
    disambiguator between competing reactivation paths across substrates;
    a substrate with high per-axis Taylor projection in the seg axis but
    low in the pose axis routes differently than the inverse per CLAUDE.md
    "SegNet vs PoseNet importance — operating-point dependent")
"""
from __future__ import annotations

from tac.cross_substrate_master_gradient_analyzer.analyzer import (
    CANONICAL_EQUATION_ID,
    CROSS_SUBSTRATE_ANALYSES_LEDGER_PATH,
    PREDICTED_AXIS_TAG,
    SCHEMA_VERSION,
    VALID_AXIS_LABELS,
    CrossSubstrateAxisProjection,
    CrossSubstrateMasterGradientAnalysis,
    CrossSubstrateMasterGradientAnalysisCorruptError,
    CrossSubstrateMasterGradientOpportunity,
    CrossSubstrateSubstrateRow,
    analyze_cross_substrate_master_gradients,
    append_analysis_locked,
    compute_cauchy_schwarz_cross_substrate_bound,
    load_analyses_strict,
    rank_byte_saving_opportunities_by_cross_substrate_taylor_residual,
)

__all__ = [
    "CANONICAL_EQUATION_ID",
    "CROSS_SUBSTRATE_ANALYSES_LEDGER_PATH",
    "PREDICTED_AXIS_TAG",
    "SCHEMA_VERSION",
    "VALID_AXIS_LABELS",
    "CrossSubstrateAxisProjection",
    "CrossSubstrateMasterGradientAnalysis",
    "CrossSubstrateMasterGradientAnalysisCorruptError",
    "CrossSubstrateMasterGradientOpportunity",
    "CrossSubstrateSubstrateRow",
    "analyze_cross_substrate_master_gradients",
    "append_analysis_locked",
    "compute_cauchy_schwarz_cross_substrate_bound",
    "load_analyses_strict",
    "rank_byte_saving_opportunities_by_cross_substrate_taylor_residual",
]
