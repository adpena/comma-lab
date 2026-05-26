# SPDX-License-Identifier: MIT
"""tac.master_gradient_comparison - canonical multi-granularity comparison surface.

Per SLOT MG-3 operator directive 2026-05-19 (parallel slot of the 5-slot
master-gradient batch). Produces the canonical multi-granularity comparison
surface for the 10-exploit enumeration documented in the parent operator
dialogue:

1. Per-pixel scorer-weighted reconstruction-error encoder loss
2. Per-pair difficulty atlas + bit-budget allocation
3. Per-byte top-K / bottom-K importance ranking (consumed by sister
   ``per_byte_sensitivity_consumer`` via DuckDB; this module is the producer)
4. Substrate-fit diagnostic via M_inflated vs M_contest residual
5. SegNet-class decomposition (NSCS06 v6 -> v7 anchor; 44% improvement)  # DOCSTRING_PERCENT_CLAIM_OK:canonical_NSCS06_v6_to_v7_empirical_44pct_anchor_per_omx_research_nscs06_path_a_chroma_optical_flow_redesign_20260516_md_cargo_cult_unwind_methodology_105_15_to_58_89_contest_CUDA
6. Cross-archive substrate-shift detector via gradient correlation
7. Information-theoretic floor via Cramer-Rao lower bound
8. Pareto facets emission for ``tac.optimization.dykstra_feasibility``
9. Symmetry-breaking via gradient-similarity equivalence classes
10. Chain-rule derivation of per-byte from per-pixel (Catalog #318 fail-closed)

Per Catalog #318 master-gradient raw-byte-authority guard: this module NEVER
performs raw archive-byte finite differences. All per-byte sensitivity is
DERIVED VIA CHAIN RULE from per-pixel scorer gradients composed with the
inflate Jacobian. The chain-rule path is the only contest-faithful score
derivative on a ZIP + entropy-coded packet.

Per CLAUDE.md "Bit-level deconstruction and entropy discipline" + Catalog
#220 / #272 / #105 / #139: byte-mutation closure proofs remain the canonical
score-impact verification. This module emits SENSITIVITY SURFACES, not score
claims. All persisted artifacts carry canonical Provenance per Catalog #323
with ``ProvenanceKind.PREDICTED_FROM_MODEL`` + ``ProvenanceEvidenceGrade.PREDICTED``
+ ``promotion_eligible=False``.

Cross-references:
* Catalog #318 raw-byte-authority guard (THIS module's chain-rule discipline
  is sister to that gate; SLOT MG-3 #352 enforces chain-rule routing
  structurally)
* Catalog #323 canonical Provenance umbrella (every persisted row)
* Catalog #287 placeholder-rationale rejection (waivers MUST be substantive)
* Catalog #305 observability surface declaration (this module IS the
  observability surface for the 10-exploit enumeration)
* Catalog #125 6-hook wire-in non-negotiable
* ``tac.master_gradient`` (the canonical per-pair / aggregate gradient
  helper; this module CONSUMES its outputs)
* ``tac.master_gradient_consumers.load_aggregate_gradient_from_anchor`` and
  sister ``load_per_pair_gradient_from_anchor`` (canonical loaders this
  module routes through, never the raw JSONL)
* ``src/tac/cathedral_consumers/per_pair_difficulty_atlas_consumer``
  (downstream consumer that consumes ``compute_per_pair_difficulty_atlas``
  output)
* ``tools/compare_master_gradient_surfaces.py`` (operator-facing CLI)

Cited canonical references (per Catalog #287 evidence-tag discipline):
* Cramer-Rao lower bound: Rao (1945) + Cramer (1946) - information-theoretic
  variance lower bound on unbiased estimators.
* Fisher information matrix: Fisher (1922) - per-parameter score sensitivity.
* Chain rule for compositions: standard calculus; canonical citation
  Goodfellow / Bengio / Courville (2016) Deep Learning Ch. 6.5 for
  multi-variable backprop.
* NSCS06 v6 -> v7 44% improvement via SegNet-class chroma anchors:  # DOCSTRING_PERCENT_CLAIM_OK:canonical_anchor_citation_explicitly_references_omx_research_nscs06_path_a_chroma_optical_flow_redesign_20260516_md_empirical_44pct_105_15_to_58_89_contest_CUDA
  ``.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md``.
"""

from __future__ import annotations

from tac.master_gradient_comparison.multi_granularity import (
    ArchiveByteGradientTensor,
    ContestGradientTensor,
    EquivalenceClass,
    InflatedGradientTensor,
    M_ARCHIVE_VIA_CHAIN_RULE_PROVENANCE_KIND,
    M_CONTEST_PROVENANCE_KIND,
    M_INFLATED_PROVENANCE_KIND,
    MultiGranularityComparisonError,
    PerPairDifficulty,
    PerPixelReconstructionError,
    cluster_pairs_by_gradient_similarity,
    compute_per_pair_difficulty_atlas,
    compute_score_weighted_reconstruction_error,
    decompose_M_contest_per_segnet_class,
    estimate_information_theoretic_floor,
    extract_M_archive_via_chain_rule,
    extract_M_contest,
    extract_M_inflated,
    persist_comparison_artifact,
)


__all__ = [
    "ArchiveByteGradientTensor",
    "ContestGradientTensor",
    "EquivalenceClass",
    "InflatedGradientTensor",
    "M_ARCHIVE_VIA_CHAIN_RULE_PROVENANCE_KIND",
    "M_CONTEST_PROVENANCE_KIND",
    "M_INFLATED_PROVENANCE_KIND",
    "MultiGranularityComparisonError",
    "PerPairDifficulty",
    "PerPixelReconstructionError",
    "cluster_pairs_by_gradient_similarity",
    "compute_per_pair_difficulty_atlas",
    "compute_score_weighted_reconstruction_error",
    "decompose_M_contest_per_segnet_class",
    "estimate_information_theoretic_floor",
    "extract_M_archive_via_chain_rule",
    "extract_M_contest",
    "extract_M_inflated",
    "persist_comparison_artifact",
]
