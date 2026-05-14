# SPDX-License-Identifier: MIT
"""Composition cell registry: (substrate x primitive x order) tuples.

Per operator directive 2026-05-12 ("stacking and composition on
everything"), this package extends
:mod:`tac.optimization.substrate_composition_matrix` (the pairwise
SUBSTRATE x SUBSTRATE matrix) with a NEW second dimension —
packet-compiler PRIMITIVES — and enumerates the cross-product as typed
:class:`CompositionCell` rows the cathedral autopilot can rank.

Public surface
--------------
- :class:`CompositionCell` — one (substrate x ordered-primitive-pipeline) row.
- :class:`PrimitiveRow` — one packet-compiler primitive inventory row.
- :class:`PrimitiveCategory` — top-level primitive taxonomy.
- :class:`PrimitiveOrderSensitivity` — pipeline ordering rule per category.
- :func:`canonical_primitive_inventory` — 14 packet-compiler primitives.
- :func:`primitive_compatibility` — substrate_class x primitive_category gate.
- :func:`validate_pipeline_ordering` — pipeline ordering / MX validator.
- :func:`enumerate_cells` — the autopilot's substrate-composition input.
- :func:`autopilot_ranking_input` — autopilot CandidateRow-compatible dicts.
- :func:`serialize_enumeration` — JSON-safe enumeration payload.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
Every cell carries ``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False`` until an empirical anchor is
posted via :mod:`tac.continual_learning`. Predicted deltas are derived
from primitive metadata; no number here is an authoritative measurement.

Cross-references
----------------
- :mod:`tac.optimization.substrate_composition_matrix` — pairwise
  SUBSTRATE x SUBSTRATE matrix; this module re-uses its taxonomy.
- :mod:`tac.optimization.autopilot_dispatch_ranking` — consumer.
- :mod:`tac.packet_compiler` — primitive symbol surface.
- :mod:`tac.continual_learning` — posterior anchors (wire-in hook 5).
"""

from __future__ import annotations

from tac.composition.adapters import (
    ADAPTER_COMPOSITION_SCHEMA_VERSION,
    AdapterCompositionError,
    AdapterRecord,
    DeterministicHypernetworkComposer,
    FrozenLinearLayer,
    HypernetworkWeights,
    PoEGatingResult,
    TropicalSelectionResult,
    adapter_delta_matrix,
    adapter_record_from_pr95,
    compose_tropical_adapter_delta,
    fold_adapter_chain,
    fold_adapter_record_into_weight,
    fold_tropical_adapter_metadata,
    product_of_experts_gating,
    tropical_adapter_weights,
)
from tac.composition.bregman_mixing import (
    BregmanError,
    BregmanGenerator,
    BregmanMixer,
    BregmanMixerSpec,
)
from tac.composition.distillation import (
    DISTILLATION_CHAIN_SCHEMA_VERSION,
    DistillationChainError,
    DistillationStage,
    DistillationStageChain,
    build_distillation_stage_chain,
)
from tac.composition.distillation_chain import (
    DistillationChain,
    DistillationError,
    DistillationLevel,
    distillation_loss,
)
from tac.composition.enumerate import (
    ENUMERATION_SCHEMA,
    autopilot_ranking_input,
    enumerate_cells,
    enumerate_substrate_incompatible_cells,
    serialize_enumeration,
)
from tac.composition.frontier_primitives import (
    FRONTIER_PRIMITIVES_SCHEMA_VERSION,
    CheckpointBarycenter,
    CompositionPrimitiveError,
    DiagonalGaussian,
    MERAHierarchyMetadata,
    MERALevelMetadata,
    SinkhornResult,
    bregman_barycenter,
    build_mera_hierarchy_metadata,
    checkpoint_diagonal_gaussian_barycenter,
    metadata_sha256,
    normalize_weights,
    sinkhorn_transport_plan,
    tensor_sha256,
    wasserstein_diagonal_gaussian_barycenter,
)
from tac.composition.hypernetwork import (
    Hypernetwork,
    HypernetworkError,
    HypernetworkSpec,
)
from tac.composition.product_of_experts import (
    ProductOfExpertsComposer,
    ProductOfExpertsError,
    ProductOfExpertsSpec,
)
from tac.composition.registry import (
    PLANNING_ONLY,
    PROMOTION_ELIGIBLE,
    READY_FOR_EXACT_EVAL_DISPATCH,
    SCHEMA_VERSION,
    SCORE_CLAIM,
    CompositionCell,
    PrimitiveCategory,
    PrimitiveOrderSensitivity,
    PrimitiveRow,
    RefusedReason,
    ScoreAxis,
    SemanticConstraint,
    SubstrateClass,
    SubstrateRow,
    canonical_primitive_inventory,
    canonical_substrate_inventory,
    classify_pipeline_violation,
    compute_semantic_warning,
    detect_dependency_violation,
    detect_substrate_semantic_incompatibility,
    primitive_compatibility,
    serialize_primitive_inventory,
    validate_pipeline_ordering,
)
from tac.composition.sinkhorn_ot_mixing import (
    SinkhornError,
    SinkhornOTMixer,
    SinkhornOTMixerSpec,
)

__all__ = [
    "ADAPTER_COMPOSITION_SCHEMA_VERSION",
    "DISTILLATION_CHAIN_SCHEMA_VERSION",
    "ENUMERATION_SCHEMA",
    "FRONTIER_PRIMITIVES_SCHEMA_VERSION",
    "PLANNING_ONLY",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SCHEMA_VERSION",
    "SCORE_CLAIM",
    "AdapterCompositionError",
    "AdapterRecord",
    "BregmanError",
    "BregmanGenerator",
    "BregmanMixer",
    "BregmanMixerSpec",
    "CheckpointBarycenter",
    "CompositionCell",
    "CompositionPrimitiveError",
    "DeterministicHypernetworkComposer",
    "DiagonalGaussian",
    "DistillationChain",
    "DistillationChainError",
    "DistillationError",
    "DistillationLevel",
    "DistillationStage",
    "DistillationStageChain",
    "FrozenLinearLayer",
    "Hypernetwork",
    "HypernetworkError",
    "HypernetworkSpec",
    "HypernetworkWeights",
    "MERAHierarchyMetadata",
    "MERALevelMetadata",
    "PoEGatingResult",
    "PrimitiveCategory",
    "PrimitiveOrderSensitivity",
    "PrimitiveRow",
    "ProductOfExpertsComposer",
    "ProductOfExpertsError",
    "ProductOfExpertsSpec",
    "RefusedReason",
    "ScoreAxis",
    "SemanticConstraint",
    "SinkhornError",
    "SinkhornOTMixer",
    "SinkhornOTMixerSpec",
    "SinkhornResult",
    "SubstrateClass",
    "SubstrateRow",
    "TropicalSelectionResult",
    "adapter_delta_matrix",
    "adapter_record_from_pr95",
    "autopilot_ranking_input",
    "bregman_barycenter",
    "build_distillation_stage_chain",
    "build_mera_hierarchy_metadata",
    "canonical_primitive_inventory",
    "canonical_substrate_inventory",
    "checkpoint_diagonal_gaussian_barycenter",
    "classify_pipeline_violation",
    "compose_tropical_adapter_delta",
    "compute_semantic_warning",
    "detect_dependency_violation",
    "detect_substrate_semantic_incompatibility",
    "distillation_loss",
    "enumerate_cells",
    "enumerate_substrate_incompatible_cells",
    "fold_adapter_chain",
    "fold_adapter_record_into_weight",
    "fold_tropical_adapter_metadata",
    "metadata_sha256",
    "normalize_weights",
    "primitive_compatibility",
    "product_of_experts_gating",
    "serialize_enumeration",
    "serialize_primitive_inventory",
    "sinkhorn_transport_plan",
    "tensor_sha256",
    "tropical_adapter_weights",
    "validate_pipeline_ordering",
    "wasserstein_diagonal_gaussian_barycenter",
]
