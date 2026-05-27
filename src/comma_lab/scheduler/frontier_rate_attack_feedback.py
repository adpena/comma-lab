# SPDX-License-Identifier: MIT
"""Compile frontier materializer feedback into queue-owned follow-up surfaces."""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.storage_tiers import DEFAULT_RESERVE_FREE_GB
from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    read_strict_single_member_zip,
)
from tac.optimization.dqs1_materializer_feedback_bridge import (
    DQS1_OBSERVATION_SOURCE_SCHEMA,
    DQS1_OBSERVATION_SWEEP_CONFIG_ID,
    FALSE_AUTHORITY,
    build_dqs1_materializer_feedback_bridge,
)
from tac.optimization.local_cpu_contest_drift import (
    EUREKA_SIGNAL_SCHEMA,
    LocalCPUContestDriftError,
    require_eureka_false_authority,
)
from tac.optimization.materializer_feedback import (
    FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
    FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
    materializer_observation_feedback_rows,
    materializer_observation_feedback_rows_from_queue_observation,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    MLXDynamicSweepObservationError,
    load_observation_rows,
    observation_duplicate_key,
)
from tac.optimization.pairset_component_marginal import (
    component_marginal_status,
    rate_delta_for_archive_byte_delta,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.optimization.repair_dynamics_palette_probe import (
    REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA,
)

from .byte_shaving_campaign_queue import (
    MATERIALIZER_BACKLOG_SCHEMA,
    build_materializer_execution_queue,
    build_materializer_work_queue,
    materializer_contexts_from_payload,
)
from .byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND,
    ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND,
    ARCHIVE_SECTION_REORDER_TARGET_KIND,
    BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
    INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_REORDER_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
    TENSOR_PRUNE_TARGET_KIND,
    TENSOR_QUANTIZE_TARGET_KIND,
    TENSOR_SHARED_CODEBOOK_TARGET_KIND,
    registry_manifest,
)
from .dqs1_local_first_queue import (
    DEFAULT_MLX_REFERENCE_CACHE_DIR,
    DEFAULT_QUEUE_ID,
    DEFAULT_RESULTS_ROOT,
    PAIR_FRAME_GEOMETRY_QUEUE_REQUEST_SCHEMA,
    build_queue_from_action_summary,
)
from .experiment_queue import (
    DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
    QUEUE_SCHEMA,
    ExperimentQueueError,
    normalize_queue_definition,
)
from .final_byte_operation_contexts import build_final_byte_operation_contexts
from .frontier_rate_attack_target_profile import (
    TARGET_OPTIMIZATION_PROFILE_QUEUE_METADATA_SCHEMA,
    TARGET_OPTIMIZATION_PROFILE_SCHEMA,
    build_frontier_target_optimization_profile,
    target_optimization_profile_queue_metadata,
)

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_CONSUMPTION_PROOF_REVALIDATION_SCHEMA = (
    "frontier_rate_attack_runtime_consumption_proof_revalidation.v1"
)
RUNTIME_CONSUMPTION_PROOF_SUCCESS_FIELDS = (
    "passed",
    "runtime_consumption_proof_passed",
    "full_frame_inflate_parity_satisfied",
    "source_runtime_unpacker_parse_satisfied",
)

FEEDBACK_REFRESH_SCHEMA = "frontier_rate_attack_feedback_refresh.v1"
FRONTIER_RATE_ATTACK_FEEDBACK_REFRESH_SCHEMA = FEEDBACK_REFRESH_SCHEMA
PAIR_FRAME_GEOMETRY_DISCOVERY_SCHEMA = (
    "frontier_rate_attack_pair_frame_geometry_discovery.v1"
)
DISCOVERED_PAIR_FRAME_GEOMETRY_SCHEMA = (
    "frontier_rate_attack_discovered_pair_frame_geometry_lattice.v1"
)
PAIR_FRAME_GEOMETRY_LATTICE_SCHEMA = "pair_frame_scorer_geometry_lattice.v1"
MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA = (
    "frontier_rate_attack_materializer_feedback_discovery.v1"
)
DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA = (
    "frontier_rate_attack_discovered_materializer_feedback.v1"
)
LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA = "frontier_rate_attack_local_cpu_eureka_discovery.v1"
LOCAL_CPU_EUREKA_PLANNER_HINT_SCHEMA = "frontier_rate_attack_local_cpu_eureka_planner_hint.v1"
LOCAL_CPU_EUREKA_PAIRSET_PROFILE_SCHEMA = (
    "frontier_rate_attack_local_cpu_eureka_pairset_acquisition_profile.v1"
)
DQS1_OBSERVATION_DISCOVERY_SCHEMA = "frontier_rate_attack_dqs1_observation_discovery.v1"
OPERATION_PORTFOLIO_SCHEMA = "frontier_rate_attack_operation_portfolio.v1"
OPERATION_PORTFOLIO_ROW_SCHEMA = "frontier_rate_attack_operation_portfolio_row.v1"
OPERATION_PORTFOLIO_TAXONOMY_SCHEMA = (
    "frontier_rate_attack_operation_portfolio_taxonomy.v1"
)
OPERATION_MATERIALIZER_BRIDGE_SCHEMA = (
    "frontier_rate_attack_operation_materializer_bridge.v1"
)
OPERATION_MATERIALIZER_BRIDGE_ROW_SCHEMA = (
    "frontier_rate_attack_operation_materializer_bridge_row.v1"
)
AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA = (
    "frontier_rate_attack_autonomous_chain_optimization.v1"
)
AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA = (
    "frontier_rate_attack_autonomous_chain_optimization_row.v1"
)
AUTONOMOUS_CHAIN_WORK_ORDER_SCHEMA = (
    "frontier_rate_attack_autonomous_chain_work_order.v1"
)
AUTONOMOUS_CHAIN_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_autonomous_chain_optimization_queue_metadata.v1"
)
AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_STEPS = 8
AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_EXPERIMENTS = 2
AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_PARALLEL = 1
REPAIR_CAMPAIGN_SCORE_QUEUE_MAX_STEPS = 12
REPAIR_CAMPAIGN_SCORE_QUEUE_MAX_EXPERIMENTS = 2
REPAIR_CAMPAIGN_SCORE_QUEUE_MAX_PARALLEL = 1
RATE_BUDGET_PRESERVATION_PLAN_SCHEMA = (
    "frontier_rate_attack_rate_budget_preservation_plan.v1"
)
RATE_BUDGET_PRESERVATION_ROW_SCHEMA = (
    "frontier_rate_attack_rate_budget_preservation_row.v1"
)
OPERATOR_ACTION_LEDGER_SCHEMA = "frontier_rate_attack_operator_action_ledger.v1"
OPERATOR_ACTION_TERM_SCHEMA = "frontier_rate_attack_operator_action_term.v1"
OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA = (
    "frontier_rate_attack_operation_chain_compiler_work_order.v1"
)
OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA = (
    "frontier_rate_attack_operation_chain_compiler_work_orders.v1"
)
OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA = (
    "frontier_rate_attack_operation_chain_compiler_stage_plan.v1"
)
OPERATION_CHAIN_COMPILER_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_operation_chain_compiler_queue_metadata.v1"
)
BYTE_RANGE_STAGE_INPUTS_SCHEMA = "frontier_rate_attack_byte_range_stage_inputs.v1"
TARGETED_DROP_MANY_STAGE_INPUTS_SCHEMA = (
    "frontier_rate_attack_targeted_drop_many_stage_inputs.v1"
)
TARGET_OPTIMIZATION_PROFILE_METADATA_SCHEMA = (
    "frontier_target_optimization_profile_metadata.v1"
)
TARGET_OPTIMIZATION_PROFILE_METADATA_SCHEMAS = frozenset(
    {
        TARGET_OPTIMIZATION_PROFILE_METADATA_SCHEMA,
        TARGET_OPTIMIZATION_PROFILE_QUEUE_METADATA_SCHEMA,
        TARGET_OPTIMIZATION_PROFILE_SCHEMA,
    }
)
MATERIALIZER_EXACT_READINESS_BRIDGE_SCHEMA = (
    "materializer_chain_exact_readiness_bridge_report.v1"
)
MATERIALIZER_NON_RATE_POSITIVE_SKIP_BLOCKER = (
    "materializer_candidate_not_rate_positive_for_exact_readiness"
)
RECEIVER_REPAIR_BACKLOG_SCHEMA = "frontier_rate_attack_receiver_repair_backlog.v1"
RECEIVER_REPAIR_ROW_SCHEMA = "frontier_rate_attack_receiver_repair_row.v1"
RECEIVER_REPAIR_WORK_ORDER_SCHEMA = (
    "frontier_rate_attack_receiver_repair_work_order.v1"
)
RECEIVER_REPAIR_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_receiver_repair_queue_metadata.v1"
)
MATERIALIZER_EXACT_READINESS_BRIDGE_TOOL = (
    "tools/run_materializer_exact_readiness_bridge.py"
)
MATERIALIZER_CHAIN_HARVEST_TOOL = "tools/harvest_materializer_chain_candidates.py"
MATERIALIZER_SUBMISSION_CLOSURE_TOOL = (
    "tools/build_materializer_submission_closure.py"
)
BYTE_RANGE_STAGE_INPUTS_TOOL = "tools/build_frontier_byte_range_stage_inputs.py"
TARGETED_DROP_MANY_STAGE_INPUTS_TOOL = (
    "tools/build_frontier_targeted_drop_many_stage_inputs.py"
)
DQS1_LOCAL_FIRST_QUEUE_TOOL = "tools/build_dqs1_local_first_queue.py"
BYTE_RANGE_CHAIN_TOOL = "tools/run_byte_range_entropy_recode_chain.py"
MATERIALIZER_CHAIN_HARVEST_REPORT_SCHEMA = "materializer_chain_harvest_report.v1"
MATERIALIZER_SUBMISSION_CLOSURE_REPORT_SCHEMA = (
    "materializer_submission_runtime_closure_report.v1"
)
RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA = (
    "frontier_rate_attack_receiver_closed_correction_budget.v1"
)
RECEIVER_CLOSED_RATE_PACKET_SIGNAL_SCHEMA = (
    "frontier_rate_attack_receiver_closed_rate_packet_materialization_signal.v1"
)
RECEIVER_CLOSED_CORRECTION_BUDGET_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_receiver_closed_correction_budget_queue_metadata.v1"
)
REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA = (
    "frontier_rate_attack_repair_budget_waterfill_work_order.v1"
)
REPAIR_BUDGET_WATERFILL_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_repair_budget_waterfill_queue_metadata.v1"
)
REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA = (
    "frontier_rate_attack_repair_budget_waterfill_allocation_action_term.v1"
)
REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA = (
    "frontier_rate_attack_repair_budget_typed_response_ledger.v1"
)
REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA = (
    "frontier_rate_attack_repair_budget_typed_response_row.v1"
)
REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA = (
    "frontier_rate_attack_repair_cascade_opportunity_row.v1"
)
REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA = (
    "frontier_rate_attack_repair_budget_materialization_plan.v1"
)
REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA = (
    "frontier_rate_attack_repair_budget_materialization_plan_row.v1"
)
REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA = (
    "frontier_rate_attack_repair_budget_materializer_binding_report.v1"
)
REPAIR_BUDGET_MATERIALIZER_BINDING_ROW_SCHEMA = (
    "frontier_rate_attack_repair_budget_materializer_binding_row.v1"
)
REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFEST_SCHEMA = (
    "frontier_rate_attack_repair_budget_child_component_replay_manifest.v1"
)
REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFESTS_SCHEMA = (
    "frontier_rate_attack_repair_budget_child_component_replay_manifests.v1"
)
REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA = (
    "frontier_rate_attack_repair_dynamics_palette_prior.v1"
)
REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA = (
    "frontier_rate_attack_repair_budget_materialization_execution_report.v1"
)
REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA = (
    "frontier_rate_attack_repair_budget_materialization_execution_row.v1"
)
TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_acquisition.v1"
)
TARGETED_COMPONENT_CORRECTION_ACQUISITION_ROW_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_acquisition_row.v1"
)
TARGETED_COMPONENT_CORRECTION_WORK_ORDER_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_work_order.v1"
)
TARGETED_COMPONENT_CORRECTION_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_queue_metadata.v1"
)
TARGETED_COMPONENT_CORRECTION_RESPONSE_ROW_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_response_row.v1"
)
TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_response_harvest.v1"
)
TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_materialization_request_row.v1"
)
TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_materialization_requests.v1"
)
TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_materialization_queue_metadata.v1"
)
TARGETED_COMPONENT_CORRECTION_CHAIN_MATERIALIZER_HANDOFF_SCHEMA = (
    "frontier_rate_attack_targeted_component_correction_chain_materializer_handoff.v1"
)
TARGETED_COMPONENT_CHAIN_MATERIALIZER_CONTEXT_CLOSURE_PLAN_SCHEMA = (
    "frontier_rate_attack_targeted_component_chain_materializer_context_closure_plan.v1"
)
QUEUE_FALSE_AUTHORITY_FALSE_OR_MISSING_FIELDS = tuple(
    field
    for field in DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS
    if field != "dispatch_ready"
)

_DEFAULT_BYTE_RANGE_SCHEMA_MANIFEST_PATHS = (
    "experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json",
    "experiments/results/hnerv_pr103_lc_ac_schema_20260507_codex/manifest.json",
)
_DEFAULT_BYTE_RANGE_BEAM_PROBE_REPORT_PATHS = (
    ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_beam_probe_mid32.json",
    ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_0_weight_beam_probe_mid32.json",
    ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_1_weight_beam_probe_mid32.json",
    ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_2_weight_beam_probe_mid32.json",
    ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_3_weight_beam_probe_mid32.json",
    ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/latent_hi_bytes_beam_probe.json",
)
_DEFAULT_BYTE_RANGE_GLOBAL_COMBO_REPORT_PATHS = (
    ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_probe.json",
    ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_plus_latent_hi_probe.json",
)
_DEFAULT_BYTE_RANGE_SOURCE_RUNTIME_DIR_PATHS = (
    "submissions/hnerv_lc_ac",
    "experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/source/submissions/hnerv_lc_ac",
    "experiments/results/public_pr_archive_release_view/public_pr103_intake_20260505_auto/source/submissions/hnerv_lc_ac",
)
_DEFAULT_SELECTOR_PARETO_PATHS = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/dqs1_gap_uleb_selector_pareto.json",
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/dqs1_gap_uleb_selector_pareto_20260522.json",
)
_DEFAULT_PAIR_FRAME_GEOMETRY_LATTICE_PATHS = (
    ".omx/research/codex_pair_frame_geometry_lattice_20260525T151102Z/pair_frame_scorer_geometry_lattice.json",
    ".omx/research/codex_pair_frame_geometry_lattice_20260525T151102Z/dqs1_pairset_acquisition_geometry_lattice.json",
)

_OPERATION_LEVELS = (
    "bit",
    "byte",
    "packet_member",
    "archive_section",
    "tensor_channel",
    "pixel",
    "region",
    "boundary",
    "frame",
    "pair",
    "batch",
    "full_video",
    "scorer_axis",
    "receiver_runtime",
    "training_substrate",
)

_TARGETED_COMPONENT_CORRECTION_FAMILY_SEEDS: tuple[dict[str, Any], ...] = (
    {
        "correction_family": "drop_within_selected_set_masked_boundary",
        "operation_levels": [
            "pixel",
            "region",
            "boundary",
            "frame",
            "pair",
            "batch",
            "scorer_axis",
        ],
        "priority_base": 96.0,
        "recommended_next_action": (
            "probe_drop_many_within_component_safe_pair_set_under_mask_boundary_guard"
        ),
        "targeted_dimensions": [
            "pair",
            "frame",
            "region",
            "boundary",
            "batch",
        ],
        "required_priors": [
            "component_marginal_rows",
            "pair_frame_geometry_lattice",
            "segnet_boundary_marginals",
        ],
    },
    {
        "correction_family": "segnet_posenet_waterfill_region_repair",
        "operation_levels": [
            "pixel",
            "region",
            "boundary",
            "frame",
            "pair",
            "full_video",
            "scorer_axis",
        ],
        "priority_base": 93.0,
        "recommended_next_action": (
            "allocate_receiver_closed_rate_credit_to_geometry_safe_repair_atoms"
        ),
        "targeted_dimensions": [
            "pixel",
            "region",
            "boundary",
            "frame",
            "full_video",
        ],
        "required_priors": [
            "component_marginal_rows",
            "master_gradient_or_inverse_scorer",
            "segnet_boundary_marginals",
        ],
    },
    {
        "correction_family": "pose_stable_pair_frame_motion_correction",
        "operation_levels": [
            "region",
            "frame",
            "pair",
            "batch",
            "full_video",
            "scorer_axis",
        ],
        "priority_base": 90.0,
        "recommended_next_action": (
            "fit_pose_stable_pair_frame_correction_atoms_under_component_guard"
        ),
        "targeted_dimensions": [
            "region",
            "frame",
            "pair",
            "batch",
            "full_video",
        ],
        "required_priors": [
            "component_marginal_rows",
            "lapose_motion_records",
            "master_gradient_or_inverse_scorer",
        ],
    },
    {
        "correction_family": "inverse_scorer_cell_basis_expansion",
        "operation_levels": [
            "bit",
            "byte",
            "pixel",
            "region",
            "boundary",
            "frame",
            "pair",
            "batch",
            "scorer_axis",
        ],
        "priority_base": 88.0,
        "recommended_next_action": (
            "compile_inverse_scorer_cells_into_component_guarded_correction_basis"
        ),
        "targeted_dimensions": [
            "bit",
            "byte",
            "pixel",
            "region",
            "boundary",
            "frame",
            "pair",
            "batch",
        ],
        "required_priors": [
            "component_marginal_rows",
            "master_gradient_or_inverse_scorer",
            "byte_closed_materializer_context",
        ],
    },
    {
        "correction_family": "full_video_batch_residual_budget_reallocation",
        "operation_levels": [
            "tensor_channel",
            "pixel",
            "frame",
            "pair",
            "batch",
            "full_video",
            "training_substrate",
            "scorer_axis",
        ],
        "priority_base": 84.0,
        "recommended_next_action": (
            "rebalance_full_video_residual_substrate_budget_after_rate_attack"
        ),
        "targeted_dimensions": [
            "tensor_channel",
            "pixel",
            "frame",
            "pair",
            "batch",
            "full_video",
        ],
        "required_priors": [
            "component_marginal_rows",
            "receiver_closed_rate_budget",
            "local_mlx_or_exact_axis_component_probe",
        ],
    },
)

_TARGET_OPERATION_METADATA: dict[str, dict[str, Any]] = {
    "packet_member_zip_header_elide_v1": {
        "operation_family": "packet_member_header_elision",
        "levels": ["byte", "packet_member", "receiver_runtime"],
        "queue_consumer": "frontier_final_rate_attack_materializer_queue",
    },
    "packet_member_recompress_v1": {
        "operation_family": "packet_member_compression_search",
        "levels": ["bit", "byte", "packet_member"],
        "queue_consumer": "frontier_final_rate_attack_materializer_queue",
    },
    "packet_member_merge_v1": {
        "operation_family": "packet_member_receiver_transform",
        "levels": ["byte", "packet_member", "receiver_runtime"],
        "queue_consumer": "frontier_final_rate_attack_materializer_queue",
    },
    "renderer_payload_dfl1_v1": {
        "operation_family": "renderer_payload_native_transform",
        "levels": ["byte", "packet_member", "full_video", "receiver_runtime"],
        "queue_consumer": "frontier_final_rate_attack_materializer_queue",
    },
    "archive_section_entropy_recode_v1": {
        "operation_family": "archive_section_entropy_recode",
        "levels": ["bit", "byte", "archive_section"],
        "queue_consumer": "frontier_final_rate_attack_materializer_queue",
    },
    "tensor_factorize_v1": {
        "operation_family": "tensor_channel_factorization",
        "levels": ["byte", "tensor_channel", "receiver_runtime"],
        "queue_consumer": "frontier_final_rate_attack_materializer_queue",
    },
}

_EUREKA_FAMILY_METADATA: dict[str, dict[str, Any]] = {
    "learned_multi_drop": {
        "levels": ["frame", "pair", "batch", "full_video", "scorer_axis"],
        "queue_consumer": "decoder_q_pairset_acquisition",
        "queue_executable": True,
    },
    "drop_many_beam_pairwise_interaction_waterfill": {
        "levels": ["frame", "pair", "batch", "scorer_axis"],
        "queue_consumer": "decoder_q_pairset_acquisition",
        "queue_executable": True,
    },
    "within_selected_set_mask_feather_probe": {
        "levels": ["pixel", "region", "boundary", "frame", "scorer_axis"],
        "queue_consumer": "inverse_scorer_action_surface",
        "queue_executable": False,
    },
    "master_gradient_constrained_low_sensitivity_drop": {
        "levels": ["frame", "pair", "batch", "scorer_axis"],
        "queue_consumer": "master_gradient_anchors",
        "queue_executable": False,
    },
    "inverse_scorer_null_direction_masked_variant": {
        "levels": ["pixel", "region", "boundary", "pair", "scorer_axis"],
        "queue_consumer": "inverse_scorer_action_surface",
        "queue_executable": False,
    },
}

_BROAD_OPERATION_SEEDS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "bitstream_range_ans_recode_wave",
        "operation_family": "range_ans_entropy_recode",
        "operation_levels": ["bit", "byte", "archive_section", "packet_member"],
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": "compile_entropy_coder_candidates_from_packet_sections",
        "blockers": [
            "requires_section_histogram_or_payload_grammar_manifest",
            "requires_receiver_decode_parity_before_exact_readiness",
        ],
        "priority_score": 5.0,
    },
    {
        "operation_id": "region_boundary_mask_feather_waterfill",
        "operation_family": "mask_region_boundary_feather_search",
        "operation_levels": ["pixel", "region", "boundary", "frame", "scorer_axis"],
        "queue_consumer": "inverse_scorer_action_surface",
        "recommended_next_action": "compile_mask_region_boundary_ops_from_inverse_scorer_cells",
        "blockers": [
            "requires_inverse_scorer_cell_to_runtime_materializer_binding",
            "requires_segnet_posenet_geometry_guard",
        ],
        "priority_score": 7.0,
    },
    {
        "operation_id": "pair_batch_full_video_low_impact_drop_surface",
        "operation_family": "full_video_low_impact_drop_search",
        "operation_levels": ["frame", "pair", "batch", "full_video", "scorer_axis"],
        "queue_consumer": "decoder_q_pairset_acquisition",
        "recommended_next_action": "expand_drop_two_near_misses_to_drop_many_and_full_board_controls",
        "blockers": ["requires_pair_frame_geometry_lattice_or_master_gradient_binding"],
        "priority_score": 6.5,
    },
    {
        "operation_id": "segnet_posenet_geometry_tradeoff_surface",
        "operation_family": "component_geometry_tradeoff_optimizer",
        "operation_levels": ["pixel", "region", "frame", "pair", "batch", "scorer_axis"],
        "queue_consumer": "pair_frame_scorer_geometry_lattice",
        "recommended_next_action": "fit_component_tradeoff_surface_before_paid_exact_eval",
        "blockers": ["requires_component_deltas_or_local_advisory_harvest_rows"],
        "priority_score": 6.0,
    },
    {
        "operation_id": "pr95_hnerv_mlx_export_first_campaign",
        "operation_family": "pr95_hnerv_mlx_training_substrate",
        "operation_levels": ["training_substrate", "full_video", "receiver_runtime", "byte"],
        "queue_consumer": "long_burn_campaign_dispatch",
        "recommended_next_action": "launch_timing_smoke_then_export_closed_archive_runtime",
        "blockers": [
            "requires_score_aware_training_smoke",
            "requires_export_archive_grammar_and_inflate_runtime_closure",
        ],
        "priority_score": 8.0,
    },
)

_REGISTERED_MISSING_MATERIALIZER_SEEDS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "materializer_backlog_byte_range_entropy_recode_v1",
        "operation_family": "byte_range_entropy_recode",
        "operation_levels": ["bit", "byte", "archive_section", "receiver_runtime"],
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": "bind_payload_grammar_and_receiver_proof_then_compile",
    },
    {
        "operation_id": "materializer_backlog_archive_section_header_elide_v1",
        "operation_family": "archive_section_header_elide",
        "operation_levels": ["byte", "archive_section", "receiver_runtime"],
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": "prove_implicit_section_constants_and_elide_local_headers",
    },
    {
        "operation_id": "materializer_backlog_archive_section_reorder_v1",
        "operation_family": "archive_section_reorder",
        "operation_levels": ["byte", "archive_section", "receiver_runtime"],
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": "test_order_independence_and_compression_context_ordering",
    },
    {
        "operation_id": "materializer_backlog_archive_section_proceduralize_v1",
        "operation_family": "archive_section_proceduralize",
        "operation_levels": ["byte", "archive_section", "receiver_runtime"],
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": "replace_stored_sections_with_deterministic_receiver_generation",
    },
    {
        "operation_id": "materializer_backlog_tensor_quantize_v1",
        "operation_family": "score_aware_tensor_quantize",
        "operation_levels": ["tensor_channel", "byte", "scorer_axis", "receiver_runtime"],
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": "search_per_tensor_channel_quantization_under_component_sensitivity",
    },
    {
        "operation_id": "materializer_backlog_tensor_prune_v1",
        "operation_family": "score_aware_tensor_prune",
        "operation_levels": ["tensor_channel", "frame", "byte", "scorer_axis"],
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": "prune_low_sensitivity_weights_channels_frames_or_atoms",
    },
    {
        "operation_id": "materializer_backlog_tensor_shared_codebook_v1",
        "operation_family": "tensor_shared_codebook",
        "operation_levels": ["bit", "byte", "tensor_channel", "archive_section"],
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": "build_shared_dictionary_codebook_factor_streams",
    },
    {
        "operation_id": "materializer_backlog_packet_member_reorder_v1",
        "operation_family": "packet_member_reorder",
        "operation_levels": ["byte", "packet_member", "receiver_runtime"],
        "queue_consumer": "frontier_final_rate_attack_materializer_queue",
        "recommended_next_action": "require_lookup_proof_before_executable_member_order_search",
    },
    {
        "operation_id": "materializer_backlog_inverse_steganalysis_high_level_operation_set_v1",
        "operation_family": "inverse_steganalysis_high_level_operation_set",
        "operation_levels": [
            "pixel",
            "region",
            "boundary",
            "frame",
            "pair",
            "batch",
            "scorer_axis",
            "receiver_runtime",
        ],
        "queue_consumer": "inverse_scorer_action_surface",
        "recommended_next_action": "compile_high_level_action_sets_into_chained_materializers",
    },
)

_MISSING_CLASS_SEEDS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "missing_class_range_ans_packet_compiler_target",
        "operation_family": "range_ans_entropy_coding",
        "operation_levels": ["bit", "byte", "archive_section", "packet_member"],
        "queue_consumer": "packet_compiler",
        "recommended_next_action": "make_range_ans_first_class_packetir_target",
    },
    {
        "operation_id": "missing_class_hnerv_pr95_mlx_export_archives",
        "operation_family": "hnerv_pr95_mlx_export_archive",
        "operation_levels": ["training_substrate", "full_video", "byte", "receiver_runtime"],
        "queue_consumer": "long_burn_campaign_dispatch",
        "recommended_next_action": "wire_weight_layout_activation_table_coordinate_encoding_export",
    },
    {
        "operation_id": "missing_class_minimal_runtime_receiver",
        "operation_family": "runtimeless_minimal_receiver",
        "operation_levels": ["receiver_runtime", "byte", "full_video"],
        "queue_consumer": "runtime_closure",
        "recommended_next_action": "prototype_minimal_legal_receiver_runtime_with_custody_proof",
    },
    {
        "operation_id": "missing_class_frame_pair_batch_materializers",
        "operation_family": "frame_pair_batch_materializers",
        "operation_levels": ["frame", "pair", "batch", "full_video", "scorer_axis"],
        "queue_consumer": "decoder_q_pairset_acquisition",
        "recommended_next_action": "materialize_grouped_frame_pair_batch_ops_beyond_dqs1",
    },
    {
        "operation_id": "missing_class_segnet_posenet_geometry_transforms",
        "operation_family": "segnet_posenet_geometry_transform",
        "operation_levels": ["pixel", "region", "boundary", "frame", "scorer_axis"],
        "queue_consumer": "inverse_scorer_action_surface",
        "recommended_next_action": "build_mask_feather_boundary_pose_stable_region_materializers",
    },
    {
        "operation_id": "missing_class_residual_substrate_materializers",
        "operation_family": "residual_substrate_materializers",
        "operation_levels": ["training_substrate", "full_video", "byte", "receiver_runtime"],
        "queue_consumer": "long_burn_campaign_dispatch",
        "recommended_next_action": "bind_siren_finer_wire_coolchic_compressai_candidates_to_archive_runtime",
    },
)


class FrontierRateAttackFeedbackError(ExperimentQueueError):
    """Raised when frontier feedback discovery or compilation is unsafe."""


DEFAULT_TARGETED_CHAIN_FULL_FRAME_FILE_LIST = Path(
    "upstream/public_test_video_names.txt"
)
RENDERER_PAYLOAD_DFL1_REQUIRED_MEMBERS = (
    "renderer.bin",
    "masks.mkv",
    "optimized_poses.pt",
)
def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrontierRateAttackFeedbackError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackError(f"{path}: expected JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FrontierRateAttackFeedbackError(f"{path}: cannot read JSONL") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}:{index}: invalid JSONL row: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise FrontierRateAttackFeedbackError(
                f"{path}:{index}: expected JSON object row"
            )
        rows.append(row)
    return rows


def _sha256_or_none(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
        return text
    return None


def _proof_candidate_archive_sha256(proof: Mapping[str, Any]) -> str | None:
    candidate_archive = proof.get("candidate_archive")
    if isinstance(candidate_archive, Mapping):
        nested = _sha256_or_none(candidate_archive.get("sha256"))
        if nested is not None:
            return nested
    for key in ("candidate_archive_sha256", "archive_sha256"):
        value = _sha256_or_none(proof.get(key))
        if value is not None:
            return value
    return None


def _runtime_consumption_proof_revalidation(
    *,
    proof_path_text: str | Path | None,
    repo_root: str | Path = _DEFAULT_REPO_ROOT,
    expected_candidate_archive_sha256: str | None = None,
    context: str = "runtime_consumption_proof",
) -> dict[str, Any]:
    """Revalidate receiver proof from the JSON artifact, never from queue booleans."""

    blockers: list[str] = []
    proof_payload: dict[str, Any] = {}
    proof_path = str(proof_path_text or "").strip()
    resolved_path: Path | None = None
    if not proof_path:
        blockers.append(f"{context}_path_missing")
    else:
        resolved_path = _resolve_path(proof_path, repo_root=Path(repo_root))
        if not resolved_path.is_file():
            blockers.append(f"{context}_file_missing")
        elif resolved_path.suffix != ".json":
            blockers.append(f"{context}_json_file_required")
        else:
            try:
                proof_payload = _load_json(resolved_path)
            except FrontierRateAttackFeedbackError:
                blockers.append(f"{context}_json_invalid")
            else:
                try:
                    require_no_truthy_authority_fields(
                        proof_payload,
                        context=f"{context}:{proof_path}",
                    )
                except ValueError:
                    blockers.append(f"{context}_truthy_authority_present")
                payload_blockers = _string_list(proof_payload.get("blockers"))
                if payload_blockers:
                    blockers.append(f"{context}_json_blockers_present")
                    blockers.extend(
                        f"{context}:{blocker}" for blocker in payload_blockers
                    )
                if not any(
                    proof_payload.get(key) is True
                    for key in RUNTIME_CONSUMPTION_PROOF_SUCCESS_FIELDS
                ):
                    blockers.append(f"{context}_success_flag_missing")
                if not (
                    proof_payload.get("receiver_contract_satisfied") is True
                    or proof_payload.get("runtime_consumption_proof_passed") is True
                ):
                    blockers.append(f"{context}_receiver_contract_not_satisfied")
                for key in (
                    "passed",
                    "runtime_consumption_proof_passed",
                    "receiver_contract_satisfied",
                ):
                    if key in proof_payload and proof_payload.get(key) is not True:
                        blockers.append(f"{context}_{key}_not_true")
                expected_sha = _sha256_or_none(expected_candidate_archive_sha256)
                proof_sha = _proof_candidate_archive_sha256(proof_payload)
                if expected_sha is not None:
                    if proof_sha is None:
                        blockers.append(f"{context}_candidate_archive_sha256_missing")
                    elif proof_sha != expected_sha:
                        blockers.append(f"{context}_candidate_archive_sha256_mismatch")
    valid = not blockers
    return {
        "schema": RUNTIME_CONSUMPTION_PROOF_REVALIDATION_SCHEMA,
        "context": context,
        "proof_path": proof_path or None,
        "resolved_proof_path": str(resolved_path) if resolved_path is not None else None,
        "proof_file_present": bool(resolved_path is not None and resolved_path.is_file()),
        "proof_present": valid,
        "proof_valid": valid,
        "proof_schema": proof_payload.get("schema"),
        "expected_candidate_archive_sha256": _sha256_or_none(
            expected_candidate_archive_sha256
        ),
        "candidate_archive_sha256": _proof_candidate_archive_sha256(proof_payload),
        "receiver_contract_satisfied": bool(
            valid
            and (
                proof_payload.get("receiver_contract_satisfied") is True
                or proof_payload.get("runtime_consumption_proof_passed") is True
            )
        ),
        "runtime_consumption_proof_passed": bool(
            valid
            and any(
                proof_payload.get(key) is True
                for key in RUNTIME_CONSUMPTION_PROOF_SUCCESS_FIELDS
            )
        ),
        "blockers": _unique_strings(blockers),
        **FALSE_AUTHORITY,
    }


def _candidate_archive_revalidation(
    *,
    archive_path_text: str | Path | None,
    archive_sha256: str | None,
    archive_bytes: Any = None,
    repo_root: str | Path = _DEFAULT_REPO_ROOT,
    context: str = "candidate_archive",
) -> dict[str, Any]:
    blockers: list[str] = []
    archive_path = str(archive_path_text or "").strip()
    expected_sha = _sha256_or_none(archive_sha256)
    resolved_path: Path | None = None
    actual_sha: str | None = None
    if not archive_path:
        blockers.append(f"{context}_path_missing")
    elif expected_sha is None:
        blockers.append(f"{context}_sha256_missing")
    else:
        resolved_path = _resolve_path(archive_path, repo_root=Path(repo_root))
        if not resolved_path.is_file():
            blockers.append(f"{context}_file_missing")
        else:
            actual_sha = _sha256_file(resolved_path)
            if actual_sha != expected_sha:
                blockers.append(f"{context}_file_sha256_mismatch")
            elif (
                isinstance(archive_bytes, int)
                and not isinstance(archive_bytes, bool)
                and resolved_path.stat().st_size != archive_bytes
            ):
                blockers.append(f"{context}_file_bytes_mismatch")
    valid = not blockers
    return {
        "schema": "frontier_rate_attack_candidate_archive_revalidation.v1",
        "context": context,
        "archive_path": archive_path or None,
        "resolved_archive_path": (
            str(resolved_path) if resolved_path is not None else None
        ),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "archive_file_present": bool(
            resolved_path is not None and resolved_path.is_file()
        ),
        "archive_valid": valid,
        "blockers": _unique_strings(blockers),
        **FALSE_AUTHORITY,
    }


def _finite_float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _is_materializer_feedback_payload(payload: Mapping[str, Any]) -> bool:
    schema = str(payload.get("schema") or "")
    observation_kind = str(payload.get("observation_kind") or "")
    if schema == "experiment_queue_observation.v1":
        return bool(
            materializer_observation_feedback_rows_from_queue_observation(payload)
        )
    if schema in {
        FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
        FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
    }:
        return True
    if observation_kind == "family_agnostic_materializer_empirical_observation":
        return True
    observations = payload.get("observations")
    if observations is None:
        observations = payload.get("rows")
    if isinstance(observations, list):
        return any(
            isinstance(row, Mapping) and _is_materializer_feedback_payload(row)
            for row in observations
        )
    return False


def _materializer_feedback_paths(root: Path, *, max_files: int) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        raise FrontierRateAttackFeedbackError(f"feedback root does not exist: {root}")
    candidates: list[Path] = []
    scanned_candidates = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        is_candidate = path.name in {"sweep.json", "observations.jsonl"} or (
            path.suffix in {".json", ".jsonl"} and "materializer" in rel_path
        )
        if not is_candidate:
            continue
        scanned_candidates += 1
        if scanned_candidates > max_files:
            raise FrontierRateAttackFeedbackError(
                f"{root}: materializer feedback discovery exceeded max_files={max_files}"
            )
        candidates.append(path)
    return candidates


def _payload_from_materializer_feedback_path(path: Path) -> dict[str, Any] | None:
    if path.suffix == ".jsonl":
        rows = _load_jsonl(path)
        materializer_rows = [
            row for row in rows if _is_materializer_feedback_payload(row)
        ]
        if not materializer_rows:
            return None
        for index, row in enumerate(materializer_rows):
            require_no_truthy_authority_fields(
                row,
                context=f"frontier_rate_attack_feedback.jsonl[{index}]",
            )
        return {
            "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
            "source_format": "jsonl_observation_rows",
            "observations": materializer_rows,
            **FALSE_AUTHORITY,
        }
    payload = _load_json(path)
    if not _is_materializer_feedback_payload(payload):
        return None
    require_no_truthy_authority_fields(
        payload,
        context="frontier_rate_attack_feedback.materializer_payload",
    )
    return payload


def _pair_frame_geometry_paths(root: Path, *, max_files: int) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        raise FrontierRateAttackFeedbackError(
            f"pair-frame geometry root does not exist: {root}"
        )
    candidates: list[Path] = []
    scanned_files = 0
    patterns = (
        "*pair_frame*geometry*lattice*.json",
        "*pair_frame_lattice*.json",
    )
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            scanned_files += 1
            if scanned_files > max_files:
                raise FrontierRateAttackFeedbackError(
                    f"{root}: pair-frame geometry discovery exceeded max_files={max_files}"
                )
            key = path.resolve(strict=False).as_posix()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)
    return candidates


def _pair_frame_geometry_requests(payload: Mapping[str, Any], *, path: Path) -> list[dict[str, Any]]:
    if payload.get("schema") != PAIR_FRAME_GEOMETRY_LATTICE_SCHEMA:
        return []
    require_no_truthy_authority_fields(
        payload,
        context=f"{path} pair-frame geometry lattice",
    )
    requests = payload.get("queue_executable_pairset_drop_requests")
    if not isinstance(requests, list):
        return []
    out: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            continue
        if request.get("schema") != PAIR_FRAME_GEOMETRY_QUEUE_REQUEST_SCHEMA:
            continue
        require_no_truthy_authority_fields(
            request,
            context=f"{path} pair-frame geometry request[{index}]",
        )
        if request.get("queue_executable") is not True:
            continue
        out.append(dict(request))
    return out


def discover_pair_frame_geometry_queue_requests(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    pair_frame_geometry_paths: Sequence[str | Path] = (),
    max_files_per_root: int = 256,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[str, Any]]:
    """Discover queue-executable pair-frame geometry requests.

    These requests are local DQS1 starts, not scorer authority.  Discovery is
    deliberately conservative: only typed lattice JSONs with false authority and
    typed queue-executable request rows are forwarded to the queue builder.
    """

    repo = Path(repo_root)
    default_roots = not frontier_artifact_roots
    roots: Sequence[str | Path] = (
        frontier_artifact_roots if not default_roots else (repo / ".omx" / "research",)
    )
    paths: list[Path] = []
    seen_paths: set[str] = set()
    for value in pair_frame_geometry_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            paths.append(path)
    for value in roots:
        root = _resolve_path(value, repo_root=repo)
        if default_roots and not root.exists():
            continue
        for path in _pair_frame_geometry_paths(root, max_files=max_files_per_root):
            if path.as_posix() in seen_paths:
                continue
            seen_paths.add(path.as_posix())
            paths.append(path)

    requests: list[dict[str, Any]] = []
    source_paths: list[str] = []
    discovered: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()
    duplicate_request_count = 0
    for path in paths:
        rel_path = _repo_rel(path, repo)
        payload = _load_json(path)
        try:
            path_requests = _pair_frame_geometry_requests(payload, path=path)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(f"{path}: {exc}") from exc
        if not path_requests:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": "no_queue_executable_pair_frame_geometry_requests",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        unique_requests: list[dict[str, Any]] = []
        for request in path_requests:
            request_id = str(request.get("candidate_id") or "")
            if request_id in seen_request_ids:
                duplicate_request_count += 1
                continue
            seen_request_ids.add(request_id)
            unique_requests.append(request)
        if not unique_requests:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": "duplicate_pair_frame_geometry_requests",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        requests.extend(unique_requests)
        source_paths.extend([rel_path] * len(unique_requests))
        discovered.append(
            {
                "schema": DISCOVERED_PAIR_FRAME_GEOMETRY_SCHEMA,
                "path": rel_path,
                "request_count": len(unique_requests),
                "candidate_ids": [
                    str(request.get("candidate_id")) for request in unique_requests
                ],
                "drop_counts": [
                    len(request.get("dropped_pair_indices") or [])
                    for request in unique_requests
                ],
                **FALSE_AUTHORITY,
            }
        )

    discovery = {
        "schema": PAIR_FRAME_GEOMETRY_DISCOVERY_SCHEMA,
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in roots
        ],
        "explicit_pair_frame_geometry_paths": [
            _repo_rel(_resolve_path(path, repo_root=repo), repo)
            for path in pair_frame_geometry_paths
        ],
        "scanned_candidate_path_count": len(paths),
        "discovered_lattice_count": len(discovered),
        "queue_executable_request_count": len(requests),
        "duplicate_request_count": duplicate_request_count,
        "discovered_lattices": discovered,
        "ignored_lattices": ignored,
        **FALSE_AUTHORITY,
    }
    return tuple(requests), tuple(source_paths), discovery


def _materializer_observation_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("observation_id") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("target_kind") or ""),
        str(row.get("materializer_id") or ""),
        str(row.get("source_archive_sha256") or ""),
        str(row.get("candidate_archive_sha256") or ""),
        str(row.get("saved_bytes") or ""),
        str(row.get("selected_member_name") or ""),
        ",".join(str(item) for item in row.get("selected_member_names") or []),
        str(row.get("receiver_contract_satisfied")),
        str(row.get("inflate_parity_satisfied")),
        ",".join(str(item) for item in row.get("readiness_blockers") or []),
        ",".join(str(item) for item in row.get("receiver_verification_blockers") or []),
    )


def discover_materializer_feedback_payloads(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    materializer_feedback_paths: Sequence[str | Path] = (),
    max_files_per_root: int = 256,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[str, Any]]:
    """Discover family-agnostic materializer feedback under frontier roots."""

    repo = Path(repo_root)
    default_roots = not frontier_artifact_roots
    roots: Sequence[str | Path] = (
        frontier_artifact_roots if not default_roots else (repo / ".omx" / "research",)
    )
    paths: list[Path] = []
    seen_paths: set[str] = set()
    for value in materializer_feedback_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            paths.append(path)
    for value in roots:
        root = _resolve_path(value, repo_root=repo)
        if default_roots and not root.exists():
            continue
        for path in _materializer_feedback_paths(root, max_files=max_files_per_root):
            if path.as_posix() in seen_paths:
                continue
            seen_paths.add(path.as_posix())
            paths.append(path)

    payloads: list[dict[str, Any]] = []
    source_paths: list[str] = []
    discovered: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    seen_observation_keys: set[tuple[str, ...]] = set()
    duplicate_observation_count = 0
    for path in paths:
        try:
            payload = _payload_from_materializer_feedback_path(path)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(f"{path}: {exc}") from exc
        rel_path = _repo_rel(path, repo)
        if payload is None:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": "not_family_agnostic_materializer_feedback",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        try:
            rows = materializer_observation_feedback_rows(payload, source_path=rel_path)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(f"{path}: {exc}") from exc
        unique_rows: list[dict[str, Any]] = []
        duplicate_rows = 0
        for row in rows:
            key = _materializer_observation_key(row)
            if key in seen_observation_keys:
                duplicate_rows += 1
                duplicate_observation_count += 1
                continue
            seen_observation_keys.add(key)
            unique_rows.append(row)
        if not unique_rows:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": (
                        "duplicate_materializer_observations"
                        if duplicate_rows
                        else "materializer_feedback_has_no_observation_rows"
                    ),
                    "duplicate_observation_count": duplicate_rows,
                    **FALSE_AUTHORITY,
                }
            )
            continue
        target_kinds = sorted(
            {
                str(row.get("target_kind"))
                for row in unique_rows
                if str(row.get("target_kind") or "").strip()
            }
        )
        payloads.append(
            {
                "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
                "source_payload_schema": payload.get("schema"),
                "source_format": payload.get("source_format") or path.suffix.lstrip("."),
                "observations": unique_rows,
                **FALSE_AUTHORITY,
            }
        )
        source_paths.append(rel_path)
        discovered.append(
            {
                "schema": DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA,
                "path": rel_path,
                "payload_schema": payload.get("schema"),
                "observation_count": len(unique_rows),
                "duplicate_observation_count": duplicate_rows,
                "target_kinds": target_kinds,
                "rate_positive_count": sum(
                    1 for row in unique_rows if row.get("rate_positive") is True
                ),
                "receiver_positive_rate_saving_count": sum(
                    1
                    for row in unique_rows
                    if row.get("rate_positive") is True
                    and (
                        row.get("receiver_contract_satisfied") is True
                        or row.get("inflate_parity_satisfied") is True
                    )
                ),
                **FALSE_AUTHORITY,
            }
        )

    discovery = {
        "schema": MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA,
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in roots
        ],
        "explicit_materializer_feedback_paths": [
            _repo_rel(_resolve_path(path, repo_root=repo), repo)
            for path in materializer_feedback_paths
        ],
        "scanned_candidate_path_count": len(paths),
        "discovered_feedback_count": len(payloads),
        "duplicate_observation_count": duplicate_observation_count,
        "discovered_feedback": discovered,
        "ignored_feedback_candidates": ignored,
        **FALSE_AUTHORITY,
    }
    return tuple(payloads), tuple(source_paths), discovery


def _eureka_signal_paths(root: Path, *, max_files: int) -> list[Path]:
    if root.is_file():
        return [root] if root.name.startswith("local_cpu_contest_drift_eureka_") else []
    if not root.exists():
        raise FrontierRateAttackFeedbackError(f"eureka root does not exist: {root}")
    candidates: list[Path] = []
    scanned_files = 0
    for path in sorted(root.rglob("local_cpu_contest_drift_eureka_*.json")):
        if not path.is_file():
            continue
        scanned_files += 1
        if scanned_files > max_files:
            raise FrontierRateAttackFeedbackError(
                f"{root}: eureka discovery exceeded max_files={max_files}"
            )
        candidates.append(path)
    return candidates


def _eureka_candidate_family(candidate_id: str) -> str:
    if candidate_id.startswith("pairset_drop_one_"):
        return "decoder_q_pairset_drop_one"
    if candidate_id.startswith("pairset_drop_two_"):
        return "decoder_q_pairset_drop_two"
    if candidate_id.startswith("pairset_component_combo_"):
        return "decoder_q_learned_multi_drop"
    if candidate_id.startswith("pairset_"):
        return "decoder_q_pairset"
    return "unknown"


def _eureka_gap_row(payload: Mapping[str, Any], *, path: Path, repo_root: Path) -> dict[str, Any]:
    try:
        require_eureka_false_authority(
            payload,
            context=f"{path} local CPU eureka signal",
        )
    except LocalCPUContestDriftError as exc:
        raise FrontierRateAttackFeedbackError(str(exc)) from exc
    require_no_truthy_authority_fields(
        payload,
        context=f"{path} local CPU eureka signal",
    )
    candidate_id = str(payload.get("candidate_id") or "")
    auth_frontier = _finite_float_or_none(payload.get("auth_frontier_score"))
    projected = _finite_float_or_none(payload.get("projected_contest_score"))
    conservative = _finite_float_or_none(
        payload.get("conservative_projected_contest_score")
    )
    eureka_margin = _finite_float_or_none(payload.get("eureka_margin"))
    projected_gap = None
    conservative_gap = None
    if auth_frontier is not None and projected is not None:
        projected_gap = projected - auth_frontier
    if auth_frontier is not None and conservative is not None:
        conservative_gap = conservative - auth_frontier
    return {
        "schema": EUREKA_SIGNAL_SCHEMA,
        "path": _repo_rel(path, repo_root),
        "candidate_id": candidate_id,
        "candidate_family": _eureka_candidate_family(candidate_id),
        "candidate_archive_sha256": str(payload.get("candidate_archive_sha256") or ""),
        "local_score": _finite_float_or_none(payload.get("local_score")),
        "projected_contest_score": projected,
        "conservative_projected_contest_score": conservative,
        "auth_frontier_score": auth_frontier,
        "projected_gap_vs_auth_frontier": projected_gap,
        "conservative_gap_vs_auth_frontier": conservative_gap,
        "eureka_margin": eureka_margin,
        "eureka_trigger": payload.get("eureka_trigger") is True,
        "recommended_action": str(payload.get("recommended_action") or ""),
        "trust_region": str(payload.get("trust_region") or ""),
        "candidate_trust_region_matches_calibration": (
            payload.get("candidate_trust_region_matches_calibration") is True
        ),
        "source_artifact": str(payload.get("source_artifact") or ""),
        **FALSE_AUTHORITY,
    }


def _eureka_sort_key(row: Mapping[str, Any]) -> tuple[float, float, str]:
    conservative_gap = row.get("conservative_gap_vs_auth_frontier")
    projected_gap = row.get("projected_gap_vs_auth_frontier")
    return (
        float(conservative_gap) if conservative_gap is not None else float("inf"),
        float(projected_gap) if projected_gap is not None else float("inf"),
        str(row.get("candidate_id") or ""),
    )


def _eureka_planner_hints(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    near_rows = [
        row
        for row in rows
        if row.get("recommended_action") == "observe_only"
        and row.get("eureka_trigger") is False
        and (
            (
                row.get("conservative_gap_vs_auth_frontier") is not None
                and float(row["conservative_gap_vs_auth_frontier"]) <= 1.0e-5
            )
            or (
                row.get("projected_gap_vs_auth_frontier") is not None
                and float(row["projected_gap_vs_auth_frontier"]) <= 5.0e-6
            )
        )
    ]
    drop_two_near = [
        row
        for row in near_rows
        if row.get("candidate_family") == "decoder_q_pairset_drop_two"
    ]
    hints: list[dict[str, Any]] = []
    if drop_two_near:
        source_ids = [
            str(row.get("candidate_id"))
            for row in sorted(drop_two_near, key=_eureka_sort_key)[:8]
        ]
        hints.append(
            {
                "schema": LOCAL_CPU_EUREKA_PLANNER_HINT_SCHEMA,
                "hint_id": "dqs1_expand_beyond_drop_two_near_boundary",
                "trigger": "near_frontier_observe_only_drop_two_cluster",
                "source_candidate_ids": source_ids,
                "source_signal_paths": [
                    str(row.get("path"))
                    for row in sorted(drop_two_near, key=_eureka_sort_key)[:8]
                    if str(row.get("path") or "")
                ],
                "recommended_candidate_families": [
                    "learned_multi_drop",
                    "drop_many_beam_pairwise_interaction_waterfill",
                    "within_selected_set_mask_feather_probe",
                    "master_gradient_constrained_low_sensitivity_drop",
                    "inverse_scorer_null_direction_masked_variant",
                ],
                "pairset_acquisition_profile": {
                    "schema": LOCAL_CPU_EUREKA_PAIRSET_PROFILE_SCHEMA,
                    "active": True,
                    "max_drop_two": 512,
                    "max_drop_many": 96,
                    "drop_many_counts": [3, 4, 6, 8],
                    "candidate_family": "dqs1_pairset_drop_many_local_first",
                    "rate_distortion_levels_considered": [
                        "bit",
                        "byte",
                        "packet_member",
                        "tensor_channel",
                        "pixel",
                        "region",
                        "boundary",
                        "frame",
                        "pair",
                        "batch",
                        "full_video",
                        "scorer_axis",
                        "receiver_runtime",
                    ],
                    "starting_point_policy": (
                        "expand from near-frontier drop-two rows into bounded "
                        "drop-many local probes before exact-axis authority"
                    ),
                    "blocked_family_requests": [
                        {
                            "family": "global_low_impact_full_pair_drop_probe",
                            "blocker": (
                                "requires pair-frame scorer-geometry lattice "
                                "binding before full-board pair/frame drops are "
                                "queue-executable"
                            ),
                            **FALSE_AUTHORITY,
                        },
                        {
                            "family": "within_selected_set_mask_feather_probe",
                            "blocker": (
                                "requires receiver/materializer support for "
                                "non-pair-drop mask semantics"
                            ),
                            **FALSE_AUTHORITY,
                        },
                        {
                            "family": "inverse_scorer_null_direction_masked_variant",
                            "blocker": (
                                "requires inverse-scorer action cell to runtime "
                                "materializer binding"
                            ),
                            **FALSE_AUTHORITY,
                        },
                    ],
                    **FALSE_AUTHORITY,
                },
                "rationale": (
                    "drop-two local CPU drift rows are close enough to the frontier "
                    "to guide acquisition, but too conservative to treat as the "
                    "optimization endpoint"
                ),
                "planner_consumers": [
                    "pairset_component_marginal_model",
                    "master_gradient",
                    "inverse_scorer_action_surface",
                    "frontier_rate_attack_feedback_cycle",
                ],
                "forbidden_use": "score_claim_or_exact_eval_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
    return hints


def discover_local_cpu_eureka_planning_signals(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    max_files_per_root: int = 256,
    strict_authority: bool = True,
) -> dict[str, Any]:
    """Discover local advisory eureka rows and compile acquisition hints.

    These rows are not observations for score/rank.  They are acquisition
    priors for the next local queue cycle, especially when near-boundary
    drop-two rows imply the search should expand beyond drop-two.
    """

    repo = Path(repo_root)
    explicit_roots = bool(frontier_artifact_roots)
    roots: Sequence[str | Path] = (
        frontier_artifact_roots if explicit_roots else (repo / ".omx" / "research",)
    )
    paths: list[Path] = []
    seen_paths: set[str] = set()
    for value in roots:
        root = _resolve_path(value, repo_root=repo)
        if not root.exists() and not explicit_roots:
            continue
        for path in _eureka_signal_paths(root, max_files=max_files_per_root):
            key = path.resolve(strict=False).as_posix()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            paths.append(path)

    rows: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    seen_signal_keys: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    for path in paths:
        payload = _load_json(path)
        if payload.get("schema") != EUREKA_SIGNAL_SCHEMA:
            ignored.append(
                {
                    "path": _repo_rel(path, repo),
                    "reason": "not_local_cpu_contest_drift_eureka_signal",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        try:
            row = _eureka_gap_row(payload, path=path, repo_root=repo)
        except FrontierRateAttackFeedbackError as exc:
            if explicit_roots and strict_authority:
                raise
            ignored.append(
                {
                    "path": _repo_rel(path, repo),
                    "reason": str(exc),
                    **FALSE_AUTHORITY,
                }
            )
            continue
        key = (
            str(row.get("candidate_id") or ""),
            str(row.get("candidate_archive_sha256") or ""),
            str(row.get("source_artifact") or ""),
        )
        if key in seen_signal_keys:
            duplicate_count += 1
            continue
        seen_signal_keys.add(key)
        rows.append(row)

    rows = sorted(rows, key=_eureka_sort_key)
    family_counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("candidate_family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
    hints = _eureka_planner_hints(rows)
    best = rows[0] if rows else None
    return {
        "schema": LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA,
        "active": bool(rows),
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in roots
        ],
        "signal_count": len(rows),
        "duplicate_signal_count": duplicate_count,
        "ignored_signal_candidates": ignored,
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "near_frontier_observe_only_count": len(
            [
                row
                for row in rows
                if row.get("recommended_action") == "observe_only"
                and row.get("eureka_trigger") is False
                and row.get("projected_gap_vs_auth_frontier") is not None
                and float(row["projected_gap_vs_auth_frontier"]) <= 5.0e-6
            ]
        ),
        "best_projected_gap_vs_auth_frontier": (
            None if best is None else best.get("projected_gap_vs_auth_frontier")
        ),
        "best_conservative_gap_vs_auth_frontier": (
            None if best is None else best.get("conservative_gap_vs_auth_frontier")
        ),
        "planner_hint_count": len(hints),
        "planner_hints": hints,
        "signal_rows": rows[:32],
        "allowed_use": "local_advisory_acquisition_prior_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def load_dqs1_observations(
    *,
    repo_root: str | Path,
    observation_paths: Sequence[str | Path],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Load and dedupe canonical DQS1 local-first observation JSONL rows."""

    repo = Path(repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []
    seen_rows: set[tuple[tuple[str, str | None], ...]] = set()
    seen_paths: set[str] = set()
    for value in observation_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.suffix != ".jsonl":
            raise FrontierRateAttackFeedbackError(
                f"{path}: DQS1 observations must be JSONL rows"
            )
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            source_paths.append(_repo_rel(path, repo))
        try:
            loaded = load_observation_rows(path)
        except OSError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}: cannot read DQS1 observation JSONL"
            ) from exc
        except MLXDynamicSweepObservationError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}: invalid DQS1 observation JSONL: {exc}"
            ) from exc
        for row in loaded:
            if (
                row.get("source_schema") != DQS1_OBSERVATION_SOURCE_SCHEMA
                or row.get("sweep_config_id") != DQS1_OBSERVATION_SWEEP_CONFIG_ID
            ):
                raise FrontierRateAttackFeedbackError(
                    f"{path}: non-local-first DQS1 observation row refused "
                    f"for candidate {row.get('candidate_id')!r}"
                )
            key = observation_duplicate_key(row)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            rows.append(row)
    return tuple(rows), tuple(source_paths)


def discover_dqs1_observation_jsonl_paths(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Find append-only DQS1 local-first observation JSONLs for queue feedback."""

    repo = Path(repo_root)
    roots: Sequence[str | Path] = (
        frontier_artifact_roots
        if frontier_artifact_roots
        else (repo / ".omx" / "research",)
    )
    paths: list[Path] = []
    seen: set[str] = set()
    for value in roots:
        root = _resolve_path(value, repo_root=repo)
        if root.is_file():
            candidates = [root] if root.name.startswith("dqs1_local_first_harvest_observations_") and root.suffix == ".jsonl" else []
        elif root.exists():
            candidates = list(root.rglob("dqs1_local_first_harvest_observations_*.jsonl"))
        else:
            candidates = []
        for path in sorted(candidates):
            key = path.resolve(strict=False).as_posix()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return {
        "schema": DQS1_OBSERVATION_DISCOVERY_SCHEMA,
        "active": bool(paths),
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in roots
        ],
        "discovered_observation_count": len(paths),
        "discovered_observation_jsonl_paths": [_repo_rel(path, repo) for path in paths],
        "allowed_use": "local_advisory_observation_replanning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [str(value)]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _unique_strings(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _finite_int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _normalized_file_list_entries(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalized_file_list_sha256(entries: Sequence[str]) -> str:
    payload = "\n".join(str(entry).strip() for entry in entries if str(entry).strip())
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


def _zip_contains_all_members(
    path_value: str | Path | None,
    *,
    repo_root: Path,
    member_names: Sequence[str],
) -> bool:
    if path_value in (None, ""):
        return False
    try:
        archive_path = _resolve_path(str(path_value), repo_root=repo_root)
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return all(name in names for name in member_names)


def _component_deltas(row: Mapping[str, Any]) -> dict[str, float] | None:
    raw = row.get("component_deltas")
    if not isinstance(raw, Mapping):
        raw = row.get("component_axis_deltas")
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, float] = {}
    for key in ("segnet_delta", "posenet_delta", "rate_delta"):
        value = _finite_float_or_none(raw.get(key))
        if value is None:
            return None
        out[key] = value
    return out


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _component_behavior_summary(
    dqs1_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for observation in dqs1_observations:
        components = _component_deltas(observation)
        if components is None:
            continue
        score_delta = _finite_float_or_none(observation.get("score_delta_vs_baseline"))
        archive_delta = _finite_int_or_none(
            observation.get("archive_byte_delta_vs_baseline")
        )
        try:
            status = component_marginal_status(
                segnet_delta=components["segnet_delta"],
                posenet_delta=components["posenet_delta"],
                rate_delta=components["rate_delta"],
            )
        except ValueError:
            status = "component_delta_invalid"
        rows.append(
            {
                "candidate_id": observation.get("candidate_id"),
                "family": observation.get("family"),
                "score_delta_vs_baseline": score_delta,
                "archive_byte_delta_vs_baseline": archive_delta,
                "selected_pair_indices": observation.get("selected_pair_indices"),
                "component_deltas": components,
                "component_marginal_status": status,
                **FALSE_AUTHORITY,
            }
        )
    if not rows:
        return {
            "schema": "frontier_rate_attack_component_behavior_summary.v1",
            "active": False,
            "inactive_reason": "no_dqs1_observation_component_deltas",
            "observation_count": 0,
            "allowed_use": "local_advisory_component_behavior_planning_only",
            "forbidden_use": "score_claim_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
    score_values = [
        float(row["score_delta_vs_baseline"])
        for row in rows
        if row.get("score_delta_vs_baseline") is not None
    ]
    seg_values = [float(row["component_deltas"]["segnet_delta"]) for row in rows]
    pose_values = [float(row["component_deltas"]["posenet_delta"]) for row in rows]
    rate_values = [float(row["component_deltas"]["rate_delta"]) for row in rows]
    best = min(
        rows,
        key=lambda row: (
            float(row["score_delta_vs_baseline"])
            if row.get("score_delta_vs_baseline") is not None
            else float("inf"),
            str(row.get("candidate_id") or ""),
        ),
    )
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("component_marginal_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema": "frontier_rate_attack_component_behavior_summary.v1",
        "active": True,
        "observation_count": len(rows),
        "best_candidate_id": best.get("candidate_id"),
        "best_family": best.get("family"),
        "best_score_delta_vs_baseline": best.get("score_delta_vs_baseline"),
        "best_archive_byte_delta_vs_baseline": best.get(
            "archive_byte_delta_vs_baseline"
        ),
        "best_component_deltas": best.get("component_deltas"),
        "best_selected_pair_indices": best.get("selected_pair_indices"),
        "component_marginal_status_counts": dict(sorted(status_counts.items())),
        "score_delta_min": min(score_values) if score_values else None,
        "score_delta_mean": _mean(score_values),
        "segnet_delta_mean": _mean(seg_values),
        "posenet_delta_mean": _mean(pose_values),
        "rate_delta_mean": _mean(rate_values),
        "negative_score_delta_count": sum(
            1
            for row in rows
            if row.get("score_delta_vs_baseline") is not None
            and float(row["score_delta_vs_baseline"]) < 0.0
        ),
        "pose_regression_count": sum(
            1 for row in rows if float(row["component_deltas"]["posenet_delta"]) > 0.0
        ),
        "segnet_regression_count": sum(
            1 for row in rows if float(row["component_deltas"]["segnet_delta"]) > 0.0
        ),
        "local_video_behavior_rows": rows[:16],
        "allowed_use": "local_advisory_component_behavior_planning_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _master_gradient_summary(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / ".omx" / "state" / "master_gradient_anchors.jsonl"
    if not path.is_file():
        return {
            "schema": "frontier_rate_attack_master_gradient_summary.v1",
            "active": False,
            "inactive_reason": "missing_master_gradient_anchors_jsonl",
            **FALSE_AUTHORITY,
        }
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        rows.append(
            {
                "archive_sha256": payload.get("archive_sha256"),
                "measurement_axis": payload.get("measurement_axis"),
                "measurement_method": payload.get("measurement_method"),
                "gradient_tensor_kind": payload.get("gradient_tensor_kind"),
                "gradient_byte_domain": payload.get("gradient_byte_domain"),
                "n_pairs_used": payload.get("n_pairs_used"),
                "n_pairs_total": payload.get("n_pairs_total"),
                "gradient_array_path": payload.get("gradient_array_path"),
                **FALSE_AUTHORITY,
            }
        )
    return {
        "schema": "frontier_rate_attack_master_gradient_summary.v1",
        "active": bool(rows),
        "anchor_count": len(rows),
        "latest_anchor": rows[-1] if rows else None,
        "anchors": rows[-8:],
        "allowed_use": "local_advisory_gradient_prior_for_acquisition_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _operation_row(
    *,
    operation_id: str,
    operation_family: str,
    operation_levels: Sequence[str],
    queue_consumer: str,
    recommended_next_action: str,
    priority_score: float,
    evidence_sources: Sequence[str] = (),
    evidence_summary: Mapping[str, Any] | None = None,
    blockers: Sequence[str] = (),
    queue_executable: bool = False,
    followup_signal: bool | None = None,
    source_kind: str = "frontier_feedback",
    suppression_keys: Sequence[str] = (),
    component_behavior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    levels = _unique_strings([level for level in operation_levels if level in _OPERATION_LEVELS])
    if not levels:
        levels = ["full_video"]
    safe_priority = _finite_float_or_none(priority_score)
    if safe_priority is None:
        safe_priority = 0.0
    normalized_blockers = _unique_strings(blockers)
    executable = bool(queue_executable) and not normalized_blockers
    signal = bool(executable if followup_signal is None else followup_signal)
    return {
        "schema": OPERATION_PORTFOLIO_ROW_SCHEMA,
        "operation_id": operation_id,
        "operation_family": operation_family,
        "operation_levels": levels,
        "queue_consumer": queue_consumer,
        "queue_executable": executable,
        "followup_signal": signal,
        "source_kind": source_kind,
        "priority_score": float(safe_priority),
        "recommended_next_action": recommended_next_action,
        "blockers": normalized_blockers,
        "suppression_keys": _unique_strings(suppression_keys),
        "evidence_sources": _unique_strings(evidence_sources),
        "evidence_summary": dict(evidence_summary or {}),
        "component_behavior": dict(component_behavior or {}),
        "allowed_use": "operation_portfolio_acquisition_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _materializer_target_root_for_source_path(path: Path) -> Path | None:
    if path.name == "candidate.json" and path.parent.parent.name == "rows":
        return path.parent.parent.parent
    if path.name in {"sweep.json", "observations.jsonl"}:
        return path.parent
    return None


def _exact_readiness_bridge_paths_for_sources(
    source_paths: Sequence[Any],
    *,
    repo_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for value in source_paths:
        text = str(value or "").strip()
        if not text:
            continue
        source_path = _resolve_path(text, repo_root=repo_root)
        target_root = _materializer_target_root_for_source_path(source_path)
        if target_root is None:
            continue
        bridge_path = (
            target_root / "exact_eval_handoff" / "exact_readiness_bridge_report.json"
        )
        key = bridge_path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        paths.append(bridge_path)
    return paths


def _exact_readiness_bridge_summary(
    bridge_paths: Sequence[Path],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    skip_reason_counts: dict[str, int] = {}
    missing_paths: list[str] = []
    invalid_paths: list[dict[str, str]] = []
    for path in bridge_paths:
        if not path.is_file():
            missing_paths.append(_repo_rel(path, repo_root))
            continue
        try:
            payload = _load_json(path)
            require_no_truthy_authority_fields(
                payload,
                context=f"{path} exact-readiness bridge",
            )
        except (FrontierRateAttackFeedbackError, ValueError) as exc:
            invalid_paths.append({"path": _repo_rel(path, repo_root), "reason": str(exc)})
            continue
        if payload.get("schema") != MATERIALIZER_EXACT_READINESS_BRIDGE_SCHEMA:
            invalid_paths.append(
                {
                    "path": _repo_rel(path, repo_root),
                    "reason": "schema_mismatch",
                    "schema": str(payload.get("schema") or ""),
                }
            )
            continue
        row_blockers: list[str] = []
        row_skip_reasons: list[str] = []
        for row in payload.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            row_is_skipped = str(row.get("readiness_verdict") or "").startswith(
                "skipped"
            )
            for blocker in _string_list(row.get("blockers")):
                if row_is_skipped or blocker == MATERIALIZER_NON_RATE_POSITIVE_SKIP_BLOCKER:
                    row_skip_reasons.append(blocker)
                    skip_reason_counts[blocker] = skip_reason_counts.get(blocker, 0) + 1
                else:
                    row_blockers.append(blocker)
                    blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        for blocker in _string_list(payload.get("dispatch_blockers")):
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        bridge_rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        skipped_candidate_count = _finite_int_or_none(
            payload.get("skipped_candidate_count")
        )
        if skipped_candidate_count is None:
            skipped_candidate_count = sum(
                1
                for row in bridge_rows
                if isinstance(row, Mapping)
                and str(row.get("readiness_verdict") or "").startswith("skipped")
            )
        reports.append(
            {
                "path": _repo_rel(path, repo_root),
                "candidate_count": _finite_int_or_none(payload.get("candidate_count")) or 0,
                "ready_candidate_count": _finite_int_or_none(
                    payload.get("ready_candidate_count")
                )
                or 0,
                "blocked_candidate_count": _finite_int_or_none(
                    payload.get("blocked_candidate_count")
                )
                or 0,
                "skipped_candidate_count": skipped_candidate_count,
                "candidate_ids": [
                    str(row.get("candidate_id") or "")
                    for row in payload.get("rows") or []
                    if isinstance(row, Mapping) and row.get("candidate_id")
                ],
                "row_blockers_sample": _unique_strings(row_blockers)[:12],
                "row_skip_reasons_sample": _unique_strings(row_skip_reasons)[:12],
                **FALSE_AUTHORITY,
            }
        )
    ready_count = sum(int(report["ready_candidate_count"]) for report in reports)
    candidate_count = sum(int(report["candidate_count"]) for report in reports)
    blocked_count = sum(int(report["blocked_candidate_count"]) for report in reports)
    skipped_count = sum(int(report["skipped_candidate_count"]) for report in reports)
    actionable_count = max(0, candidate_count - skipped_count)
    return {
        "schema": "frontier_rate_attack_materializer_exact_readiness_bridge_summary.v1",
        "bridge_report_count": len(reports),
        "candidate_count": candidate_count,
        "actionable_candidate_count": actionable_count,
        "ready_candidate_count": ready_count,
        "blocked_candidate_count": blocked_count,
        "skipped_candidate_count": skipped_count,
        "missing_bridge_report_paths": missing_paths,
        "invalid_bridge_report_paths": invalid_paths,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        "top_blockers": [
            blocker
            for blocker, _count in sorted(
                blocker_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:16]
        ],
        "top_skip_reasons": [
            reason
            for reason, _count in sorted(
                skip_reason_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:16]
        ],
        "reports": reports,
        "ready_for_chain_exact_readiness": bool(
            actionable_count and ready_count == actionable_count
        ),
        "allowed_use": "operation_portfolio_exact_readiness_planning_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


_RECEIVER_CLOSED_BUDGET_SAFE_BLOCKERS = frozenset(
    {
        "bridge_report_is_not_dispatch_authority",
        "optimizer_candidate_queue_is_planning_only",
        "requires_exact_eval_readiness_gate",
        "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
        "requires_non_proxy_score_evidence_before_promotion",
        "use_per_candidate_exact_ready_queue_only_when_present",
        "lane_claim_required_before_gpu_or_remote_eval",
    }
)
_RECEIVER_CLOSED_BUDGET_SAFE_BLOCKER_PREFIXES = (
    "above_active_floor_archive_bytes_without_operator_override",
)


def _receiver_closed_budget_safe_blocker(blocker: str) -> bool:
    lowered = str(blocker or "").strip().lower()
    if not lowered:
        return True
    if lowered in _RECEIVER_CLOSED_BUDGET_SAFE_BLOCKERS:
        return True
    return any(
        lowered.startswith(prefix)
        for prefix in _RECEIVER_CLOSED_BUDGET_SAFE_BLOCKER_PREFIXES
    )


def _paired_submission_closure_bridge_paths(closure_report_path: Path) -> list[Path]:
    closure_dir = closure_report_path.parent
    paths: list[Path] = []
    if closure_dir.name.startswith("submission_closure"):
        suffix = closure_dir.name.removeprefix("submission_closure")
        paths.extend(
            [
                closure_dir.parent / f"exact_readiness_bridge{suffix}" / (
                    "exact_readiness_bridge_report.json"
                ),
                closure_dir.parent / "exact_readiness_bridge_report.json",
            ]
        )
    for parent in closure_report_path.parents:
        if parent.name.startswith("submission_closure"):
            suffix = parent.name.removeprefix("submission_closure")
            paths.extend(
                [
                    parent.parent
                    / f"exact_readiness_bridge{suffix}"
                    / "exact_readiness_bridge_report.json",
                    parent.parent / "exact_readiness_bridge_report.json",
                ]
            )
            break
    paths.append(closure_dir / "exact_readiness_bridge_report.json")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _paired_submission_closure_bridge_path(closure_report_path: Path) -> Path:
    return _paired_submission_closure_bridge_paths(closure_report_path)[0]


def _discover_submission_closure_report_paths(
    *,
    repo_root: Path,
    frontier_artifact_roots: Sequence[str | Path],
    results_root: str | Path,
) -> list[Path]:
    roots: list[Path] = []
    for root in frontier_artifact_roots:
        roots.append(_resolve_path(root, repo_root=repo_root))
    resolved_results = _resolve_path(results_root, repo_root=repo_root)
    roots.extend(
        [
            resolved_results / "frontier_final_rate_attack",
            resolved_results / "frontier_operation_chain_compiler",
            resolved_results / "frontier_operation_portfolio",
            resolved_results / "frontier_operation_portfolio" / "frontier_receiver_repair",
        ]
    )
    paths: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        candidates: list[Path] = []
        if root.is_file() and root.name == "submission_closure_report.json":
            candidates = [root]
        elif root.is_dir():
            candidates = list(root.rglob("submission_closure_report.json"))
        for path in candidates:
            key = path.resolve(strict=False).as_posix()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths, key=lambda path: path.resolve(strict=False).as_posix())


def _bridge_rows_for_candidate(
    payload: Mapping[str, Any],
    candidate_id: str,
) -> list[Mapping[str, Any]]:
    rows = [row for row in payload.get("rows") or [] if isinstance(row, Mapping)]
    if not candidate_id:
        return rows
    matched = [row for row in rows if str(row.get("candidate_id") or "") == candidate_id]
    return matched


def _path_text_from_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        nested = value.get("path")
        return str(nested) if nested else None
    if value:
        return str(value)
    return None


def _source_queue_candidate_entries(
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    for key in ("top_k_forensic", "top_k", "dispatch_ready", "rows"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            entries.extend(row for row in value if isinstance(row, Mapping))
    return entries


def _submission_closure_reference_eval_context(
    closure: Mapping[str, Any],
    closure_report_path: Path,
    *,
    repo_root: Path,
    candidate_id: str,
) -> dict[str, Any]:
    source_queue_text = _path_text_from_value(
        closure.get("closed_source_queue_path") or closure.get("source_queue_path")
    )
    source_runtime_text = _path_text_from_value(
        closure.get("source_submission_dir")
        or closure.get("source_reference_runtime_dir")
    )
    source_inflate_text = _path_text_from_value(
        closure.get("source_inflate_sh_path")
        or closure.get("source_reference_inflate_sh_path")
    )
    source_archive_text = _path_text_from_value(closure.get("source_archive_path"))
    source_archive_sha256 = closure.get("source_archive_sha256")
    source_archive_bytes = closure.get("source_archive_bytes")
    source_queue_path: Path | None = None
    if source_queue_text:
        source_queue_path = _resolve_path(source_queue_text, repo_root=repo_root)
        if not source_queue_path.exists():
            source_queue_path = (
                closure_report_path.parent / source_queue_text
            ).resolve(strict=False)
    if source_queue_path is not None and source_queue_path.is_file():
        try:
            queue = _load_json(source_queue_path)
        except FrontierRateAttackFeedbackError:
            queue = {}
        candidate_archive_sha = str(closure.get("archive_sha256") or "")
        entries = _source_queue_candidate_entries(queue)
        matched = [
            entry
            for entry in entries
            if str(entry.get("candidate_id") or "") == candidate_id
            or str(entry.get("archive_sha256") or "") == candidate_archive_sha
            or str(entry.get("candidate_archive_sha256") or "") == candidate_archive_sha
        ]
        for entry in matched or entries[:1]:
            source_archive_text = source_archive_text or _path_text_from_value(
                entry.get("source_archive_path") or entry.get("source_archive")
            )
            source_archive_sha256 = source_archive_sha256 or entry.get(
                "source_archive_sha256"
            )
            source_archive_bytes = source_archive_bytes or entry.get(
                "source_archive_bytes"
            )
            source_inflate_text = source_inflate_text or _path_text_from_value(
                entry.get("source_inflate_sh_path")
                or entry.get("source_reference_inflate_sh_path")
            )
            source_runtime_text = (
                source_runtime_text
                or _path_text_from_value(entry.get("source_submission_dir"))
                or _path_text_from_value(entry.get("source_reference_runtime_dir"))
                or _path_text_from_value(entry.get("packet_member_merge_source_runtime_dir"))
            )
            runtime = entry.get("packet_member_merge_receiver_runtime")
            if isinstance(runtime, Mapping):
                source_runtime_text = source_runtime_text or _path_text_from_value(
                    runtime.get("source_runtime_dir")
                )
                entrypoint = runtime.get("entrypoint")
                if isinstance(entrypoint, Mapping):
                    source_runtime = entrypoint.get("source_runtime")
                    if isinstance(source_runtime, Mapping):
                        source_runtime_inflate = _path_text_from_value(
                            source_runtime.get("path")
                        )
                        if source_runtime_text is None:
                            source_inflate_text = (
                                source_inflate_text or source_runtime_inflate
                            )
            if source_archive_text and (source_runtime_text or source_inflate_text):
                break
    source_archive_path = (
        _resolve_path(source_archive_text, repo_root=repo_root)
        if source_archive_text
        else None
    )
    source_runtime_dir = (
        _resolve_path(source_runtime_text, repo_root=repo_root)
        if source_runtime_text
        else None
    )
    explicit_source_inflate = (
        _resolve_path(source_inflate_text, repo_root=repo_root)
        if source_inflate_text
        else None
    )
    if source_runtime_dir is None and explicit_source_inflate is not None:
        source_runtime_dir = explicit_source_inflate.parent
    source_inflate_sh = (
        explicit_source_inflate
        if explicit_source_inflate is not None
        else source_runtime_dir / "inflate.sh"
        if source_runtime_dir is not None
        else None
    )
    return {
        "source_archive_path": (
            None
            if source_archive_path is None
            else _repo_rel(source_archive_path, repo_root)
        ),
        "source_archive_sha256": source_archive_sha256,
        "source_archive_bytes": _finite_int_or_none(source_archive_bytes),
        "source_submission_dir": (
            None
            if source_runtime_dir is None
            else _repo_rel(source_runtime_dir, repo_root)
        ),
        "source_inflate_sh_path": (
            None
            if source_inflate_sh is None
            else _repo_rel(source_inflate_sh, repo_root)
        ),
        "source_queue_path": (
            None
            if source_queue_path is None
            else _repo_rel(source_queue_path, repo_root)
        ),
        "reference_eval_role": "receiver_closed_source_reference",
        "allowed_use": "receiver_closed_source_reference_for_component_delta_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _submission_closure_budget_row(
    closure_report_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    try:
        closure = _load_json(closure_report_path)
        require_no_truthy_authority_fields(
            closure,
            context=f"{closure_report_path} submission closure",
        )
    except (FrontierRateAttackFeedbackError, ValueError) as exc:
        return {
            "closure_report_path": _repo_rel(closure_report_path, repo_root),
            "generated_at_utc": None,
            "receiver_closed": False,
            "critical_blockers": [f"invalid_submission_closure_report:{exc}"],
            "saved_bytes_at_risk": 0,
            **FALSE_AUTHORITY,
        }
    if closure.get("schema") != MATERIALIZER_SUBMISSION_CLOSURE_REPORT_SCHEMA:
        return {
            "closure_report_path": _repo_rel(closure_report_path, repo_root),
            "generated_at_utc": None,
            "receiver_closed": False,
            "critical_blockers": ["submission_closure_report_schema_mismatch"],
            "saved_bytes_at_risk": 0,
            **FALSE_AUTHORITY,
        }

    candidate_id = str(closure.get("candidate_id") or "")
    target_kind = str(closure.get("target_kind") or "")
    signal = closure.get("targeted_correction_budget_signal")
    saved = _finite_int_or_none(closure.get("saved_bytes_at_risk"))
    if saved is None and isinstance(signal, Mapping):
        saved = _finite_int_or_none(signal.get("saved_bytes_at_risk"))
    if saved is None:
        saved = 0

    bridge_paths = _paired_submission_closure_bridge_paths(closure_report_path)
    bridge_path = next(
        (candidate for candidate in bridge_paths if candidate.is_file()),
        bridge_paths[0],
    )
    bridge_blockers: list[str] = []
    critical_blockers: list[str] = []
    bridge_candidate_count = 0
    if not bridge_path.is_file():
        critical_blockers.append("paired_exact_readiness_bridge_report_missing")
    else:
        try:
            bridge = _load_json(bridge_path)
            require_no_truthy_authority_fields(
                bridge,
                context=f"{bridge_path} receiver-closed budget bridge",
            )
        except (FrontierRateAttackFeedbackError, ValueError) as exc:
            bridge = {}
            critical_blockers.append(f"invalid_exact_readiness_bridge_report:{exc}")
        if bridge and bridge.get("schema") != MATERIALIZER_EXACT_READINESS_BRIDGE_SCHEMA:
            critical_blockers.append("paired_exact_readiness_bridge_schema_mismatch")
        elif bridge:
            rows = _bridge_rows_for_candidate(bridge, candidate_id)
            bridge_candidate_count = len(rows)
            if not rows:
                critical_blockers.append("paired_exact_readiness_bridge_candidate_missing")
            for row in rows:
                bridge_blockers.extend(_string_list(row.get("blockers")))
            bridge_blockers.extend(_string_list(bridge.get("dispatch_blockers")))
            critical_blockers.extend(
                blocker
                for blocker in bridge_blockers
                if not _receiver_closed_budget_safe_blocker(blocker)
            )

    if saved <= 0:
        critical_blockers.append("saved_bytes_at_risk_missing_or_non_positive")
    receiver_closed = not critical_blockers
    candidate_archive_text = _path_text_from_value(
        closure.get("candidate_archive_path") or closure.get("archive_path")
    )
    candidate_archive_path = (
        _resolve_path(candidate_archive_text, repo_root=repo_root)
        if candidate_archive_text
        else None
    )
    candidate_submission_text = _path_text_from_value(closure.get("submission_dir"))
    candidate_submission_dir = (
        _resolve_path(candidate_submission_text, repo_root=repo_root)
        if candidate_submission_text
        else None
    )
    candidate_inflate_text = _path_text_from_value(
        closure.get("candidate_inflate_sh_path") or closure.get("inflate_sh_path")
    )
    candidate_inflate_sh = (
        _resolve_path(candidate_inflate_text, repo_root=repo_root)
        if candidate_inflate_text
        else None
    )
    if candidate_inflate_sh is None and candidate_submission_dir is not None:
        candidate_inflate_sh = candidate_submission_dir / "inflate.sh"
    reference_eval_context = _submission_closure_reference_eval_context(
        closure,
        closure_report_path,
        repo_root=repo_root,
        candidate_id=candidate_id,
    )
    return {
        "schema": "frontier_rate_attack_receiver_closed_correction_budget_row.v1",
        "candidate_id": candidate_id,
        "target_kind": target_kind,
        "archive_path": (
            None
            if candidate_archive_path is None
            else _repo_rel(candidate_archive_path, repo_root)
        ),
        "archive_sha256": closure.get("archive_sha256"),
        "archive_bytes": closure.get("archive_bytes"),
        "candidate_archive_path": (
            None
            if candidate_archive_path is None
            else _repo_rel(candidate_archive_path, repo_root)
        ),
        "candidate_archive_sha256": closure.get(
            "candidate_archive_sha256", closure.get("archive_sha256")
        ),
        "candidate_archive_bytes": _finite_int_or_none(
            closure.get("candidate_archive_bytes") or closure.get("archive_bytes")
        ),
        "generated_at_utc": closure.get("generated_at_utc"),
        "saved_bytes_at_risk": saved,
        "receiver_closed": receiver_closed,
        "release_to_targeted_correction_planning": receiver_closed,
        "ready_for_budget_spend": False,
        "correction_budget_gate": (
            "receiver_static_runtime_closed_active_floor_only"
            if receiver_closed
            else "receiver_closure_or_bridge_blocker_present"
        ),
        "closure_report_path": _repo_rel(closure_report_path, repo_root),
        "paired_exact_readiness_bridge_report_path": _repo_rel(bridge_path, repo_root),
        "closed_source_queue_path": closure.get("closed_source_queue_path"),
        "submission_dir": closure.get("submission_dir"),
        "candidate_inflate_sh_path": (
            None
            if candidate_inflate_sh is None
            else _repo_rel(candidate_inflate_sh, repo_root)
        ),
        "source_archive_path": reference_eval_context.get("source_archive_path"),
        "source_archive_sha256": reference_eval_context.get("source_archive_sha256"),
        "source_archive_bytes": reference_eval_context.get("source_archive_bytes"),
        "source_submission_dir": reference_eval_context.get("source_submission_dir"),
        "source_inflate_sh_path": reference_eval_context.get("source_inflate_sh_path"),
        "reference_component_eval_context": reference_eval_context,
        "bridge_candidate_count": bridge_candidate_count,
        "bridge_blockers": _unique_strings(bridge_blockers),
        "critical_blockers": _unique_strings(critical_blockers),
        "active_rate_floor_blocked": any(
            str(blocker).startswith(
                "above_active_floor_archive_bytes_without_operator_override"
            )
            for blocker in bridge_blockers
        ),
        "allowed_use": "receiver_closed_rate_budget_for_targeted_correction_planning_only",
        "forbidden_use": "score_claim_or_dispatch_or_promotion_authority",
        **FALSE_AUTHORITY,
    }


def _load_receiver_closed_rate_packet_payload(
    manifest_path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], Path]:
    payload = _load_json(manifest_path)
    if payload.get("schema") == "pr101_frame_exploit_selector_archive_manifest_v1":
        sibling_packet_manifest = manifest_path.parent / "packet_manifest.json"
        if sibling_packet_manifest.is_file():
            return _load_json(sibling_packet_manifest), sibling_packet_manifest
    return payload, manifest_path


def _rate_packet_archive_info(payload: Mapping[str, Any]) -> dict[str, Any]:
    archive = payload.get("archive")
    archive_map = archive if isinstance(archive, Mapping) else {}
    selector_pack = archive_map.get("selector_pack_manifest")
    if not isinstance(selector_pack, Mapping):
        selector_pack = payload.get("selector_pack_manifest")
    selector_pack_map = selector_pack if isinstance(selector_pack, Mapping) else {}
    return {
        "archive_path": archive_map.get("path") or payload.get("archive_path"),
        "archive_sha256": archive_map.get("sha256") or payload.get("archive_sha256"),
        "archive_bytes": _finite_int_or_none(
            archive_map.get("bytes") or payload.get("archive_bytes")
        ),
        "source_archive_path": (
            archive_map.get("source")
            or payload.get("source_archive")
            or payload.get("source_archive_path")
        ),
        "source_archive_sha256": payload.get("source_archive_sha256"),
        "source_archive_bytes": _finite_int_or_none(payload.get("source_archive_bytes")),
        "selector_pack_manifest": dict(selector_pack_map),
        "compact_selector_codec": (
            payload.get("compact_selector_codec")
            or selector_pack_map.get("compact_selector_codec")
        ),
        "selector_policy_mode": payload.get("selector_policy_mode"),
        "selector_payload_wire_bytes": _finite_int_or_none(
            selector_pack_map.get("selector_payload_wire_bytes")
            or selector_pack_map.get("selector_payload_bytes")
        ),
        "selector_code_bits_total": _finite_int_or_none(
            selector_pack_map.get("selector_code_bits_total")
        ),
        "selector_avg_bits_per_pair": _finite_float_or_none(
            selector_pack_map.get("selector_avg_bits_per_pair")
            or selector_pack_map.get("selector_bits_per_pair")
        ),
        "palette_size": _finite_int_or_none(selector_pack_map.get("palette_size")),
        "n_pairs": _finite_int_or_none(selector_pack_map.get("n_pairs")),
        "compact_palette_mode_ids": list(
            selector_pack_map.get("compact_palette_mode_ids") or []
        ),
    }


def _rate_packet_runtime_info(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime")
    runtime_map = runtime if isinstance(runtime, Mapping) else {}
    manifest = runtime_map.get("manifest")
    manifest_map = manifest if isinstance(manifest, Mapping) else {}
    return {
        "submission_dir": runtime_map.get("path"),
        "runtime_content_tree_sha256": runtime_map.get("runtime_content_tree_sha256")
        or manifest_map.get("runtime_content_tree_sha256"),
        "runtime_tree_sha256": runtime_map.get("runtime_tree_sha256")
        or manifest_map.get("runtime_tree_sha256"),
        "runtime_file_count": _finite_int_or_none(
            runtime_map.get("runtime_file_count") or manifest_map.get("runtime_file_count")
        ),
    }


def _resolved_manifest_path_value(value: object, *, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _resolve_path(value, repo_root=repo_root)


def _manifest_path_display(value: object, *, repo_root: Path) -> str | None:
    resolved = _resolved_manifest_path_value(value, repo_root=repo_root)
    if resolved is None:
        return None
    return _repo_rel(resolved, repo_root)


def _verify_rate_packet_archive_file(
    info: Mapping[str, Any],
    *,
    repo_root: Path,
    role: str,
    critical_blockers: list[str],
) -> tuple[bool, bool]:
    archive_path = _resolved_manifest_path_value(
        info.get("archive_path"),
        repo_root=repo_root,
    )
    if archive_path is None:
        return False, False
    if not archive_path.is_file():
        critical_blockers.append(f"{role}_archive_file_missing")
        return False, False

    bytes_verified = False
    sha_verified = False
    expected_bytes = _finite_int_or_none(info.get("archive_bytes"))
    if expected_bytes is not None:
        actual_bytes = archive_path.stat().st_size
        if actual_bytes != expected_bytes:
            critical_blockers.append(
                f"{role}_archive_file_bytes_mismatch:{actual_bytes}!={expected_bytes}"
            )
        else:
            bytes_verified = True

    expected_sha = str(info.get("archive_sha256") or "").strip().lower()
    if expected_sha:
        actual_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            critical_blockers.append(f"{role}_archive_file_sha256_mismatch")
        else:
            sha_verified = True
    return bytes_verified, sha_verified


def _verify_rate_packet_runtime_dir(
    runtime_info: Mapping[str, Any],
    *,
    repo_root: Path,
    role: str,
    critical_blockers: list[str],
) -> bool:
    runtime_dir = _resolved_manifest_path_value(
        runtime_info.get("submission_dir"),
        repo_root=repo_root,
    )
    if runtime_dir is None:
        return False
    if not runtime_dir.is_dir():
        critical_blockers.append(f"{role}_submission_dir_missing_on_disk")
        return False
    return True


def _receiver_closed_rate_packet_budget_row(
    packet_manifest_path: Path,
    *,
    repo_root: Path,
    parent_manifest_path: Path | None = None,
) -> dict[str, Any]:
    bridge_blockers = [
        "packet_manifest_is_not_score_or_dispatch_authority",
        "component_eval_required_before_budget_spend",
        "exact_auth_eval_required_before_score_or_promotion_claim",
    ]
    critical_blockers: list[str] = []
    try:
        packet, resolved_packet_path = _load_receiver_closed_rate_packet_payload(
            packet_manifest_path,
            repo_root=repo_root,
        )
        require_no_truthy_authority_fields(
            packet,
            context=f"{resolved_packet_path} receiver-closed rate packet",
        )
    except (FrontierRateAttackFeedbackError, ValueError) as exc:
        return {
            "schema": RECEIVER_CLOSED_RATE_PACKET_SIGNAL_SCHEMA,
            "source_kind": "receiver_closed_rate_packet_materialization",
            "rate_packet_manifest_path": _repo_rel(packet_manifest_path, repo_root),
            "candidate_id": _bounded_content_key(
                "receiver_closed_rate_packet",
                (packet_manifest_path,),
            ),
            "target_kind": "pr101_frame_exploit_selector_packet_recode_v1",
            "receiver_closed": False,
            "release_to_targeted_correction_planning": False,
            "ready_for_budget_spend": False,
            "saved_bytes_at_risk": 0,
            "bridge_blockers": bridge_blockers,
            "critical_blockers": [f"invalid_rate_packet_manifest:{exc}"],
            "allowed_use": "receiver_closed_rate_budget_for_targeted_correction_planning_only",
            "forbidden_use": "score_claim_or_dispatch_or_promotion_authority",
            **FALSE_AUTHORITY,
        }

    packet_info = _rate_packet_archive_info(packet)
    runtime_info = _rate_packet_runtime_info(packet)
    if packet.get("schema") != "pr101_frame_exploit_selector_packet_manifest_v1":
        critical_blockers.append("rate_packet_manifest_schema_mismatch")
    if packet.get("candidate_archive_built") is False:
        critical_blockers.append("candidate_archive_not_built")
    if not packet_info.get("archive_path"):
        critical_blockers.append("candidate_archive_path_missing")
    if not packet_info.get("archive_sha256"):
        critical_blockers.append("candidate_archive_sha256_missing")
    if not packet_info.get("archive_bytes"):
        critical_blockers.append("candidate_archive_bytes_missing")
    if not runtime_info.get("submission_dir"):
        critical_blockers.append("candidate_submission_dir_missing")
    if not runtime_info.get("runtime_content_tree_sha256"):
        critical_blockers.append("candidate_runtime_content_tree_sha256_missing")
    (
        candidate_archive_file_bytes_verified,
        candidate_archive_file_sha256_verified,
    ) = _verify_rate_packet_archive_file(
        packet_info,
        repo_root=repo_root,
        role="candidate",
        critical_blockers=critical_blockers,
    )
    candidate_runtime_dir_verified = _verify_rate_packet_runtime_dir(
        runtime_info,
        repo_root=repo_root,
        role="candidate",
        critical_blockers=critical_blockers,
    )

    parent: dict[str, Any] = {}
    resolved_parent_path: Path | None = None
    parent_info: dict[str, Any] = {}
    parent_runtime_info: dict[str, Any] = {}
    parent_archive_file_bytes_verified = False
    parent_archive_file_sha256_verified = False
    parent_runtime_dir_verified = False
    if parent_manifest_path is None:
        critical_blockers.append("parent_rate_packet_manifest_required_for_saved_bytes")
    else:
        try:
            parent, resolved_parent_path = _load_receiver_closed_rate_packet_payload(
                parent_manifest_path,
                repo_root=repo_root,
            )
            require_no_truthy_authority_fields(
                parent,
                context=f"{resolved_parent_path} receiver-closed rate packet parent",
            )
            parent_info = _rate_packet_archive_info(parent)
            parent_runtime_info = _rate_packet_runtime_info(parent)
            if parent.get("schema") != "pr101_frame_exploit_selector_packet_manifest_v1":
                critical_blockers.append("parent_rate_packet_manifest_schema_mismatch")
            if not parent_info.get("archive_path"):
                critical_blockers.append("parent_archive_path_missing")
            if not parent_info.get("archive_sha256"):
                critical_blockers.append("parent_archive_sha256_missing")
            if not parent_info.get("archive_bytes"):
                critical_blockers.append("parent_archive_bytes_missing")
            if not parent_runtime_info.get("submission_dir"):
                critical_blockers.append("parent_submission_dir_missing")
            if not parent_runtime_info.get("runtime_content_tree_sha256"):
                critical_blockers.append("parent_runtime_content_tree_sha256_missing")
            (
                parent_archive_file_bytes_verified,
                parent_archive_file_sha256_verified,
            ) = _verify_rate_packet_archive_file(
                parent_info,
                repo_root=repo_root,
                role="parent",
                critical_blockers=critical_blockers,
            )
            parent_runtime_dir_verified = _verify_rate_packet_runtime_dir(
                parent_runtime_info,
                repo_root=repo_root,
                role="parent",
                critical_blockers=critical_blockers,
            )
        except (FrontierRateAttackFeedbackError, ValueError) as exc:
            critical_blockers.append(f"invalid_parent_rate_packet_manifest:{exc}")

    candidate_bytes = _finite_int_or_none(packet_info.get("archive_bytes"))
    parent_bytes = _finite_int_or_none(parent_info.get("archive_bytes"))
    archive_byte_delta_vs_parent: int | None = None
    saved_bytes = 0
    if candidate_bytes is not None and parent_bytes is not None:
        archive_byte_delta_vs_parent = candidate_bytes - parent_bytes
        saved_bytes = max(0, parent_bytes - candidate_bytes)
    if saved_bytes <= 0:
        critical_blockers.append("saved_bytes_vs_parent_missing_or_non_positive")

    candidate_codec = str(packet_info.get("compact_selector_codec") or "unknown_codec")
    parent_codec = str(parent_info.get("compact_selector_codec") or "unknown_parent_codec")
    lane_id = str(packet.get("lane_id") or "")
    candidate_id = _bounded_content_key(
        "receiver_closed_rate_packet",
        (
            lane_id,
            candidate_codec,
            packet_info.get("archive_sha256"),
            resolved_packet_path,
            parent_info.get("archive_sha256"),
        ),
    )
    target_kind = "pr101_frame_exploit_selector_packet_recode_v1"
    receiver_closed = not critical_blockers
    candidate_archive_path = _manifest_path_display(
        packet_info.get("archive_path"),
        repo_root=repo_root,
    )
    candidate_submission_dir = _manifest_path_display(
        runtime_info.get("submission_dir"),
        repo_root=repo_root,
    )
    parent_archive_path = _manifest_path_display(
        parent_info.get("archive_path"),
        repo_root=repo_root,
    )
    parent_submission_dir = _manifest_path_display(
        parent_runtime_info.get("submission_dir"),
        repo_root=repo_root,
    )
    parent_inflate_sh_path = (
        f"{parent_submission_dir}/inflate.sh" if parent_submission_dir else None
    )
    return {
        "schema": RECEIVER_CLOSED_RATE_PACKET_SIGNAL_SCHEMA,
        "source_kind": "receiver_closed_rate_packet_materialization",
        "candidate_id": candidate_id,
        "lane_id": lane_id or None,
        "target_kind": target_kind,
        "candidate_compact_selector_codec": candidate_codec,
        "parent_compact_selector_codec": parent_codec,
        "selector_policy_mode": packet_info.get("selector_policy_mode")
        or parent_info.get("selector_policy_mode"),
        "archive_sha256": packet_info.get("archive_sha256"),
        "archive_bytes": candidate_bytes,
        "archive_path": candidate_archive_path,
        "archive_file_sha256_verified": candidate_archive_file_sha256_verified,
        "archive_file_bytes_verified": candidate_archive_file_bytes_verified,
        "submission_dir": candidate_submission_dir,
        "submission_dir_verified": candidate_runtime_dir_verified,
        "runtime_content_tree_sha256": runtime_info.get(
            "runtime_content_tree_sha256"
        ),
        "runtime_tree_sha256": runtime_info.get("runtime_tree_sha256"),
        "runtime_file_count": runtime_info.get("runtime_file_count"),
        "source_archive_path": parent_archive_path,
        "source_archive_sha256": parent_info.get("archive_sha256"),
        "source_archive_bytes": parent_bytes,
        "source_archive_file_sha256_verified": parent_archive_file_sha256_verified,
        "source_archive_file_bytes_verified": parent_archive_file_bytes_verified,
        "source_submission_dir": parent_submission_dir,
        "source_submission_dir_verified": parent_runtime_dir_verified,
        "source_inflate_sh_path": parent_inflate_sh_path,
        "reference_component_eval_context": {
            "source_archive_path": parent_archive_path,
            "source_archive_sha256": parent_info.get("archive_sha256"),
            "source_archive_bytes": parent_bytes,
            "source_archive_file_sha256_verified": parent_archive_file_sha256_verified,
            "source_archive_file_bytes_verified": parent_archive_file_bytes_verified,
            "source_submission_dir": parent_submission_dir,
            "source_submission_dir_verified": parent_runtime_dir_verified,
            "source_inflate_sh_path": parent_inflate_sh_path,
            "reference_eval_role": "receiver_closed_parent_rate_packet_reference",
            "allowed_use": "parent_rate_packet_reference_for_component_delta_only",
            "forbidden_use": "score_claim_or_dispatch_authority",
            **FALSE_AUTHORITY,
        },
        "rate_packet_manifest_path": _repo_rel(resolved_packet_path, repo_root),
        "parent_rate_packet_manifest_path": (
            None
            if resolved_parent_path is None
            else _repo_rel(resolved_parent_path, repo_root)
        ),
        "saved_bytes_at_risk": saved_bytes,
        "archive_byte_delta_vs_parent": archive_byte_delta_vs_parent,
        "selector_payload_wire_bytes": packet_info.get("selector_payload_wire_bytes"),
        "parent_selector_payload_wire_bytes": parent_info.get(
            "selector_payload_wire_bytes"
        ),
        "selector_payload_wire_delta_bytes": (
            int(packet_info["selector_payload_wire_bytes"])
            - int(parent_info["selector_payload_wire_bytes"])
            if packet_info.get("selector_payload_wire_bytes") is not None
            and parent_info.get("selector_payload_wire_bytes") is not None
            else None
        ),
        "selector_code_bits_total": packet_info.get("selector_code_bits_total"),
        "parent_selector_code_bits_total": parent_info.get("selector_code_bits_total"),
        "selector_avg_bits_per_pair": packet_info.get("selector_avg_bits_per_pair"),
        "parent_selector_avg_bits_per_pair": parent_info.get(
            "selector_avg_bits_per_pair"
        ),
        "palette_size": packet_info.get("palette_size"),
        "n_pairs": packet_info.get("n_pairs"),
        "compact_palette_mode_ids": packet_info.get("compact_palette_mode_ids"),
        "entropy_position": "at_entropy_coder_integer_codeword_boundary",
        "receiver_closed": receiver_closed,
        "release_to_targeted_correction_planning": receiver_closed,
        "ready_for_budget_spend": False,
        "correction_budget_gate": (
            "receiver_static_runtime_closed_active_floor_only"
            if receiver_closed
            else "receiver_rate_packet_parent_or_runtime_blocker_present"
        ),
        "bridge_candidate_count": 1 if receiver_closed else 0,
        "bridge_blockers": _unique_strings(bridge_blockers),
        "critical_blockers": _unique_strings(critical_blockers),
        "active_rate_floor_blocked": False,
        "allowed_use": "receiver_closed_rate_budget_for_targeted_correction_planning_only",
        "forbidden_use": "score_claim_or_dispatch_or_promotion_authority",
        **FALSE_AUTHORITY,
    }


def build_receiver_closed_correction_budget(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    receiver_closed_rate_packet_paths: Sequence[str | Path] = (),
    receiver_closed_rate_parent_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Harvest receiver-closed materializer bytes into repair-budget signal."""

    repo = Path(repo_root)
    packet_paths = [
        _resolve_path(path, repo_root=repo) for path in receiver_closed_rate_packet_paths
    ]
    parent_paths = [
        _resolve_path(path, repo_root=repo) for path in receiver_closed_rate_parent_paths
    ]
    if parent_paths and len(parent_paths) not in (1, len(packet_paths)):
        raise FrontierRateAttackFeedbackError(
            "receiver_closed_rate_parent_paths must be empty, length 1, or match "
            "receiver_closed_rate_packet_paths"
        )
    paths = _discover_submission_closure_report_paths(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
        results_root=results_root,
    )
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    for path in paths:
        row = _submission_closure_budget_row(path, repo_root=repo)
        key = (
            str(row.get("candidate_id") or row.get("closure_report_path") or ""),
            str(row.get("target_kind") or ""),
            str(row.get("archive_sha256") or ""),
        )
        existing = rows_by_key.get(key)
        if existing is None:
            rows_by_key[key] = row
            continue
        duplicate_count += 1
        row_order = (
            str(row.get("generated_at_utc") or ""),
            str(row.get("closure_report_path") or ""),
        )
        existing_order = (
            str(existing.get("generated_at_utc") or ""),
            str(existing.get("closure_report_path") or ""),
        )
        if row_order > existing_order:
            rows_by_key[key] = row

    for index, packet_path in enumerate(packet_paths):
        parent_path = (
            parent_paths[index]
            if len(parent_paths) == len(packet_paths)
            else parent_paths[0]
            if parent_paths
            else None
        )
        row = _receiver_closed_rate_packet_budget_row(
            packet_path,
            repo_root=repo,
            parent_manifest_path=parent_path,
        )
        key = (
            str(row.get("candidate_id") or row.get("rate_packet_manifest_path") or ""),
            str(row.get("target_kind") or ""),
            str(row.get("archive_sha256") or ""),
        )
        existing = rows_by_key.get(key)
        if existing is None:
            rows_by_key[key] = row
            continue
        duplicate_count += 1
        if str(row.get("rate_packet_manifest_path") or "") > str(
            existing.get("rate_packet_manifest_path") or ""
        ):
            rows_by_key[key] = row

    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            str(row.get("target_kind") or ""),
            str(row.get("candidate_id") or ""),
            str(row.get("closure_report_path") or ""),
            str(row.get("rate_packet_manifest_path") or ""),
        ),
    )
    closed_rows = [row for row in rows if row.get("receiver_closed") is True]
    blocked_rows = [row for row in rows if row.get("receiver_closed") is not True]
    blocker_counts: dict[str, int] = {}
    for row in rows:
        for blocker in _string_list(row.get("critical_blockers")):
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        for blocker in _string_list(row.get("bridge_blockers")):
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    saved_values = [
        int(row.get("saved_bytes_at_risk") or 0)
        for row in closed_rows
        if int(row.get("saved_bytes_at_risk") or 0) > 0
    ]
    deduped_closure_count = sum(1 for row in rows if row.get("closure_report_path"))
    blockers: list[str] = []
    if not closed_rows:
        blockers.append("no_receiver_closed_materializer_budget_candidates_discovered")
    if any(row.get("active_rate_floor_blocked") is True for row in closed_rows):
        blockers.append("active_rate_floor_override_required_before_exact_dispatch")
    if closed_rows:
        blockers.append("segnet_posenet_component_eval_required_before_budget_spend")
        blockers.append("exact_auth_eval_required_before_score_or_promotion_claim")
    if blocked_rows:
        blockers.append("some_submission_closure_reports_still_have_receiver_blockers")
    return {
        "schema": RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA,
        "generated_at_utc": _utc_now(),
        "active": bool(closed_rows),
        "closure_report_count": len(paths),
        "rate_packet_manifest_count": len(packet_paths),
        "deduped_closure_report_count": deduped_closure_count,
        "deduped_budget_row_count": len(rows),
        "duplicate_closure_report_count": duplicate_count,
        "duplicate_budget_row_count": duplicate_count,
        "receiver_closed_candidate_count": len(closed_rows),
        "blocked_candidate_count": len(blocked_rows),
        "receiver_closed_saved_bytes_total": sum(saved_values),
        "receiver_closed_saved_bytes_max": max(saved_values) if saved_values else 0,
        "receiver_closed_budget_source_targets": _unique_strings(
            [row.get("target_kind") for row in closed_rows]
        ),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "top_blockers": [
            blocker
            for blocker, _count in sorted(
                blocker_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:16]
        ],
        "blockers": _unique_strings(blockers),
        "rows": rows,
        "recommended_next_action": (
            "feed_receiver_closed_rate_budget_into_targeted_segnet_posenet_"
            "correction_acquisition_under_component_eval_gate"
            if closed_rows
            else "run_receiver_repair_queue_until_static_runtime_closure_bridge_is_closed"
        ),
        "allowed_use": "receiver_closed_rate_budget_for_targeted_correction_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _receiver_closed_correction_budget_queue_metadata(
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RECEIVER_CLOSED_CORRECTION_BUDGET_QUEUE_METADATA_SCHEMA,
        "receiver_closed_correction_budget_schema": budget.get("schema"),
        "active": budget.get("active") is True,
        "receiver_closed_candidate_count": budget.get("receiver_closed_candidate_count"),
        "receiver_closed_saved_bytes_total": budget.get(
            "receiver_closed_saved_bytes_total"
        ),
        "receiver_closed_saved_bytes_max": budget.get("receiver_closed_saved_bytes_max"),
        "receiver_closed_budget_source_targets": list(
            budget.get("receiver_closed_budget_source_targets") or []
        ),
        "blockers": list(budget.get("blockers") or []),
        "allowed_use": "queue_metadata_pointer_to_receiver_closed_correction_budget",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _operation_portfolio_taxonomy() -> dict[str, Any]:
    return {
        "schema": OPERATION_PORTFOLIO_TAXONOMY_SCHEMA,
        "operation_levels": list(_OPERATION_LEVELS),
        "executable_target_kinds": sorted(_TARGET_OPERATION_METADATA),
        "registered_missing_materializers": [
            {
                "operation_id": str(seed["operation_id"]),
                "operation_family": str(seed["operation_family"]),
                "operation_levels": list(seed["operation_levels"]),
                "queue_consumer": str(seed["queue_consumer"]),
            }
            for seed in _REGISTERED_MISSING_MATERIALIZER_SEEDS
        ],
        "missing_entire_classes": [
            {
                "operation_id": str(seed["operation_id"]),
                "operation_family": str(seed["operation_family"]),
                "operation_levels": list(seed["operation_levels"]),
                "queue_consumer": str(seed["queue_consumer"]),
            }
            for seed in _MISSING_CLASS_SEEDS
        ],
        "allowed_use": "taxonomy_for_operation_portfolio_consumers_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _first_row_text(row: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _looks_like_section_manifest_path(value: str) -> bool:
    name = Path(value).name.lower()
    if name in {"sections.json", "section_manifest.json", "section_manifest.jsonl"}:
        return True
    if not name.endswith(".json"):
        return False
    if any(token in name for token in ("runtime", "proof", "candidate", "output")):
        return False
    return "section_manifest" in name or "sections" in name


def _section_manifest_from_feedback(row: Mapping[str, Any]) -> str | None:
    explicit = _first_row_text(
        row,
        (
            "section_manifest",
            "section_manifest_path",
            "parser_section_manifest",
            "packet_ir_manifest",
        ),
    )
    if explicit is not None:
        return explicit
    for path in _string_list(row.get("expected_artifact_paths")):
        if _looks_like_section_manifest_path(path):
            return path
    return None


def _passed_receiver_or_runtime_proof(row: Mapping[str, Any]) -> bool:
    if row.get("receiver_contract_satisfied") is True:
        return True
    if row.get("inflate_parity_satisfied") is True:
        return True
    status = str(row.get("runtime_consumption_proof_status") or "").strip().lower()
    return status in {"passed", "satisfied", "verified", "present_and_passed"}


def _materializer_feedback_context_hint(row: Mapping[str, Any]) -> dict[str, Any]:
    hint: dict[str, Any] = {}
    archive_path = _first_row_text(
        row,
        ("source_archive_path", "archive_path", "source_archive"),
    )
    if archive_path is not None:
        hint["archive_path"] = archive_path
        hint["source_archive"] = archive_path

    section_manifest = _section_manifest_from_feedback(row)
    if section_manifest is not None:
        hint["section_manifest"] = section_manifest

    packet_member_manifest = _first_row_text(
        row,
        ("packet_member_manifest", "member_manifest", "source_manifest_path"),
    )
    if packet_member_manifest is not None:
        hint["packet_member_manifest"] = packet_member_manifest

    tensor_manifest = _first_row_text(row, ("tensor_manifest", "tensor_manifest_path"))
    if tensor_manifest is not None:
        hint["tensor_manifest"] = tensor_manifest

    for out_key, in_keys in (
        ("candidate_id", ("candidate_id", "observation_id")),
        ("target_kind", ("target_kind",)),
        ("source_materializer_feedback_path", ("source_path",)),
        ("observed_candidate_archive_path", ("candidate_archive_path",)),
        ("observed_materializer_manifest_path", ("manifest_path",)),
    ):
        value = _first_row_text(row, in_keys)
        if value is not None:
            hint[out_key] = value

    runtime_proof = _first_row_text(
        row,
        ("runtime_consumption_proof", "runtime_consumption_proof_path"),
    )
    if runtime_proof is not None:
        if _passed_receiver_or_runtime_proof(row):
            hint["runtime_consumption_proof"] = runtime_proof
        else:
            hint["observed_runtime_consumption_proof_path"] = runtime_proof

    expected_paths = _string_list(row.get("expected_artifact_paths"))
    if expected_paths:
        hint["feedback_expected_artifact_paths"] = expected_paths[:12]
    blockers = _unique_strings(
        [
            *_string_list(row.get("readiness_blockers")),
            *_string_list(row.get("receiver_verification_blockers")),
        ]
    )
    if blockers:
        hint["feedback_readiness_blockers"] = blockers
    saved = _finite_int_or_none(row.get("saved_bytes"))
    if saved is not None:
        hint["feedback_saved_bytes"] = saved
    if row.get("rate_positive") is not None:
        hint["feedback_rate_positive"] = row.get("rate_positive") is True
    if row.get("receiver_contract_satisfied") is not None:
        hint["feedback_receiver_contract_satisfied"] = (
            row.get("receiver_contract_satisfied") is True
        )
    return hint


def _materializer_context_hint_score(hint: Mapping[str, Any]) -> tuple[int, int, int, int]:
    completeness = sum(
        1
        for key in (
            "archive_path",
            "section_manifest",
            "packet_member_manifest",
            "tensor_manifest",
            "runtime_consumption_proof",
        )
        if hint.get(key)
    )
    receiver_ok = 1 if hint.get("feedback_receiver_contract_satisfied") is True else 0
    rate_ok = 1 if hint.get("feedback_rate_positive") is True else 0
    saved = _finite_int_or_none(hint.get("feedback_saved_bytes")) or 0
    return (completeness, receiver_ok, rate_ok, saved)


def _portfolio_materializer_context_hint(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence = row.get("evidence_summary")
    if not isinstance(evidence, Mapping):
        return {}
    hint = evidence.get("best_context_hint")
    if isinstance(hint, Mapping):
        return dict(hint)
    return {}


def _materializer_operation_rows(
    payloads: Sequence[Mapping[str, Any]],
    source_paths: Sequence[str],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for payload_index, payload in enumerate(payloads):
        rows = materializer_observation_feedback_rows(
            payload,
            source_path=(
                source_paths[payload_index]
                if payload_index < len(source_paths)
                else None
            ),
        )
        for row in rows:
            target = str(row.get("target_kind") or "unknown_materializer")
            group = grouped.setdefault(
                target,
                {
                    "rows": [],
                    "source_paths": [],
                    "saved_bytes": [],
                    "blockers": [],
                    "suppression_keys": [],
                },
            )
            group["rows"].append(row)
            if row.get("source_path"):
                group["source_paths"].append(row.get("source_path"))
            saved = _finite_int_or_none(row.get("saved_bytes"))
            if saved is not None:
                group["saved_bytes"].append(saved)
            group["blockers"].extend(_string_list(row.get("readiness_blockers")))
            group["blockers"].extend(_string_list(row.get("receiver_verification_blockers")))
            if saved is not None and saved <= 0:
                group["suppression_keys"].append(
                    ":".join(
                        _unique_strings(
                            [
                                target,
                                row.get("source_archive_sha256"),
                                row.get("selected_member_name"),
                                ",".join(_string_list(row.get("selected_member_names"))),
                            ]
                        )
                    )
                )
    out: list[dict[str, Any]] = []
    for target, group in sorted(grouped.items()):
        rows = [row for row in group["rows"] if isinstance(row, Mapping)]
        saved_values = [int(value) for value in group["saved_bytes"]]
        positive_rows = [row for row in rows if row.get("rate_positive") is True]
        context_hints = [
            hint
            for hint in (_materializer_feedback_context_hint(row) for row in rows)
            if hint
        ]
        context_hints = sorted(
            context_hints,
            key=_materializer_context_hint_score,
            reverse=True,
        )
        receiver_positive = [
            row
            for row in positive_rows
            if row.get("receiver_contract_satisfied") is True
            or row.get("inflate_parity_satisfied") is True
        ]
        receiver_negative = [
            row
            for row in positive_rows
            if row.get("receiver_contract_satisfied") is False
            and row.get("inflate_parity_satisfied") is not True
        ]
        negative_count = len([value for value in saved_values if value <= 0])
        max_saved = max(saved_values) if saved_values else 0
        metadata = _TARGET_OPERATION_METADATA.get(target, {})
        blockers = list(group["blockers"])
        exact_readiness_bridge = _exact_readiness_bridge_summary(
            _exact_readiness_bridge_paths_for_sources(
                group["source_paths"],
                repo_root=repo_root,
            ),
            repo_root=repo_root,
        )
        actionable_bridge_count = int(
            exact_readiness_bridge.get("actionable_candidate_count") or 0
        )
        if exact_readiness_bridge["bridge_report_count"] and actionable_bridge_count:
            blockers.append("exact_readiness_bridge_report_not_ready")
            if exact_readiness_bridge["ready_candidate_count"]:
                blockers.append("exact_readiness_bridge_has_ready_candidates_pending_authority")
        blockers.extend(
            f"exact_readiness_bridge:{blocker}"
            for blocker in exact_readiness_bridge["top_blockers"][:6]
        )
        followup_signal = bool(
            receiver_positive or receiver_negative or positive_rows or negative_count
        )
        queue_executable = followup_signal
        if receiver_negative and not receiver_positive:
            blockers.append("receiver_runtime_or_inflate_parity_missing_for_rate_positive_candidate")
            action = "repair_receiver_runtime_then_chain_rate_positive_candidate"
        elif receiver_positive:
            blockers.append("exact_readiness_and_auth_axis_eval_still_required")
            action = "chain_receiver_positive_candidate_to_exact_readiness_handoff"
        elif negative_count:
            blockers.append("same_archive_target_member_negative_rate_feedback")
            action = "suppress_repeat_and_change_codec_parameters_or_target"
            queue_executable = False
        else:
            blockers.append("materializer_feedback_has_no_positive_rate_signal")
            action = "keep_as_low_priority_materializer_probe"
        priority = float(max(max_saved, 0)) / 16.0
        if receiver_negative:
            priority += 40.0
        if receiver_positive:
            priority += 30.0
        if negative_count and not positive_rows:
            priority -= 20.0
        out.append(
            _operation_row(
                operation_id=f"materializer_{target}",
                operation_family=str(
                    metadata.get("operation_family") or f"{target}_operation"
                ),
                operation_levels=metadata.get("levels", ["byte", "packet_member"]),
                queue_consumer=str(
                    metadata.get("queue_consumer")
                    or "frontier_final_rate_attack_materializer_queue"
                ),
                recommended_next_action=action,
                priority_score=priority,
                evidence_sources=group["source_paths"],
                evidence_summary={
                    "target_kind": target,
                    "observation_count": len(rows),
                    "rate_positive_count": len(positive_rows),
                    "receiver_positive_rate_saving_count": len(receiver_positive),
                    "receiver_negative_rate_saving_count": len(receiver_negative),
                    "negative_or_zero_rate_count": negative_count,
                    "max_saved_bytes": max_saved,
                    "total_positive_saved_bytes": sum(
                        value for value in saved_values if value > 0
                    ),
                    "exact_readiness_bridge": exact_readiness_bridge,
                    "context_hint_count": len(context_hints),
                    "best_context_hint": context_hints[0] if context_hints else {},
                    "context_hint_samples": context_hints[:5],
                },
                blockers=blockers,
                queue_executable=queue_executable,
                followup_signal=followup_signal,
                source_kind="materializer_feedback",
                suppression_keys=group["suppression_keys"],
            )
        )
    return out


def _materializer_chain_operation_rows(
    materializer_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_target = {
        str(row.get("evidence_summary", {}).get("target_kind") or ""): row
        for row in materializer_rows
        if isinstance(row.get("evidence_summary"), Mapping)
    }
    chain_targets = (
        "renderer_payload_dfl1_v1",
        "packet_member_merge_v1",
        "packet_member_zip_header_elide_v1",
    )
    present = [target for target in chain_targets if target in by_target]
    if len(present) < 2:
        return []
    total_positive_saved = sum(
        int(by_target[target]["evidence_summary"].get("total_positive_saved_bytes") or 0)
        for target in present
    )
    receiver_blockers = [
        blocker
        for target in present
        for blocker in _string_list(by_target[target].get("blockers"))
        if "receiver" in blocker or "runtime" in blocker or "parity" in blocker
    ]
    exact_bridge_summaries: list[Mapping[str, Any]] = []
    seen_bridge_summary_keys: set[str] = set()
    for target in present:
        if not isinstance(by_target[target].get("evidence_summary"), Mapping):
            continue
        summary = by_target[target]["evidence_summary"].get("exact_readiness_bridge")
        if not isinstance(summary, Mapping):
            continue
        report_paths = [
            str(report.get("path") or "")
            for report in summary.get("reports") or []
            if isinstance(report, Mapping) and report.get("path")
        ]
        key = "|".join(sorted(report_paths)) or target
        if key in seen_bridge_summary_keys:
            continue
        seen_bridge_summary_keys.add(key)
        exact_bridge_summaries.append(summary)
    exact_ready_candidates = sum(
        int(summary.get("ready_candidate_count") or 0)
        for summary in exact_bridge_summaries
    )
    exact_candidate_count = sum(
        int(summary.get("candidate_count") or 0) for summary in exact_bridge_summaries
    )
    exact_skipped_candidates = sum(
        int(summary.get("skipped_candidate_count") or 0)
        for summary in exact_bridge_summaries
    )
    exact_actionable_candidates = sum(
        int(
            summary.get("actionable_candidate_count")
            if summary.get("actionable_candidate_count") is not None
            else max(
                0,
                int(summary.get("candidate_count") or 0)
                - int(summary.get("skipped_candidate_count") or 0),
            )
        )
        for summary in exact_bridge_summaries
    )
    exact_blocker_counts: dict[str, int] = {}
    for summary in exact_bridge_summaries:
        for blocker, count in (summary.get("blocker_counts") or {}).items():
            parsed = _finite_int_or_none(count)
            exact_blocker_counts[str(blocker)] = exact_blocker_counts.get(
                str(blocker), 0
            ) + (parsed or 0)
    exact_bridge_reports = [
        dict(report)
        for summary in exact_bridge_summaries
        for report in summary.get("reports") or []
        if isinstance(report, Mapping)
    ]
    blockers = [
        "chain_requires_single_runtime_consumption_proof",
        "chain_requires_exact_readiness_handoff_after_composition",
        *receiver_blockers,
    ]
    if (
        exact_bridge_summaries
        and exact_actionable_candidates
        and exact_ready_candidates < exact_actionable_candidates
    ):
        blockers.append("chain_exact_readiness_bridges_have_no_ready_candidate")
    return [
        _operation_row(
            operation_id="chain_dfl1_merge_header_elide_minimal_envelope",
            operation_family="combined_receiver_byte_chain",
            operation_levels=["byte", "packet_member", "full_video", "receiver_runtime"],
            queue_consumer="frontier_final_rate_attack_materializer_queue",
            recommended_next_action=(
                "compose_dfl1_payload_merge_and_zip_header_elide_with_single_receiver_proof"
            ),
            priority_score=95.0 + float(max(total_positive_saved, 0)) / 8.0,
            evidence_sources=[
                source
                for target in present
                for source in _string_list(by_target[target].get("evidence_sources"))
            ],
            evidence_summary={
                "operator_model": "score_R_T_of_T_archive_equals_distortion_plus_rate_under_exact_runtime_proof",
                "chain_targets": list(present),
                "missing_chain_targets": [
                    target for target in chain_targets if target not in by_target
                ],
                "total_positive_saved_bytes_from_observed_parts": total_positive_saved,
                "targeted_correction_budget_model": (
                    "rate_positive_bytes_can_fund_segnet_posenet_repair_only_after_"
                    "single_receiver_runtime_proof_and_component_guard"
                ),
                "exact_readiness_bridge_summary": {
                    "schema": (
                        "frontier_rate_attack_chain_exact_readiness_bridge_summary.v1"
                    ),
                    "bridge_report_count": len(exact_bridge_summaries),
                    "candidate_count": exact_candidate_count,
                    "actionable_candidate_count": exact_actionable_candidates,
                    "ready_candidate_count": exact_ready_candidates,
                    "blocked_candidate_count": sum(
                        int(summary.get("blocked_candidate_count") or 0)
                        for summary in exact_bridge_summaries
                    ),
                    "skipped_candidate_count": exact_skipped_candidates,
                    "top_blockers": [
                        blocker
                        for blocker, _count in sorted(
                            exact_blocker_counts.items(),
                            key=lambda item: (-item[1], item[0]),
                        )[:16]
                    ],
                    "reports": exact_bridge_reports,
                    **FALSE_AUTHORITY,
                },
                "synergy_terms_to_measure": [
                    "header_elide_after_merge_member_name_constants",
                    "dfl1_binary_header_after_payload_member_rename",
                    "central_directory_minimization_after_member_merge",
                    "receiver_runtime_constant_lengths_after_chain",
                ],
                "antagonism_terms_to_measure": [
                    "payload_merge_blocks_member_local_recompression",
                    "dfl1_envelope_alignment_changes_zip_deflate_context",
                    "runtime_adapter_size_vs_archive_byte_gain",
                ],
            },
            blockers=blockers,
            queue_executable=False,
            followup_signal=True,
            source_kind="materializer_chain_acquisition",
        )
    ]


def _registered_materializer_chain_operation_rows(
    backlog_rows: Sequence[Mapping[str, Any]],
    *,
    targeted_correction_budget: Mapping[str, Any],
) -> list[dict[str, Any]]:
    registered_rows = [
        row
        for row in backlog_rows
        if str(row.get("operation_id") or "").startswith("materializer_backlog_")
    ]
    if len(registered_rows) < 2:
        return []
    chain_targets = [
        str(row.get("operation_id") or "").removeprefix("materializer_backlog_")
        for row in registered_rows
    ]
    materializer_levels = _unique_strings(
        [
            level
            for row in registered_rows
            for level in _string_list(row.get("operation_levels"))
        ]
    )
    missing_contracts = [
        "payload_grammar_schema_manifest",
        "archive_section_header_elision_contract",
        "archive_section_order_independence_contract",
        "tensor_sensitivity_rank_quant_prune_contract",
        "shared_codebook_dictionary_contract",
        "single_composed_receiver_runtime_consumption_proof",
        "chain_exact_readiness_handoff_after_composition",
    ]
    receiver_closed_saved = _finite_int_or_none(
        targeted_correction_budget.get("receiver_closed_materializer_saved_bytes_total")
    ) or 0
    local_drop_saved = _finite_int_or_none(
        targeted_correction_budget.get("local_drop_saved_bytes_total")
    ) or 0
    return [
        _operation_row(
            operation_id="chain_registered_multisurface_materializer_program",
            operation_family="registered_multisurface_materializer_chain",
            operation_levels=materializer_levels,
            queue_consumer="frontier_materializer_chain_acquisition_queue",
            recommended_next_action=(
                "compile_registered_bit_byte_archive_tensor_receiver_ops_into_"
                "one_scored_chain_plan_before_more_leaf_materializer_work"
            ),
            priority_score=72.0
            + min(float(max(receiver_closed_saved + local_drop_saved, 0)) / 64.0, 12.0),
            evidence_summary={
                "schema": "frontier_rate_attack_registered_materializer_chain_summary.v1",
                "operator_model": (
                    "minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes_"
                    "for_score_R_T_of_T_archive_under_single_receiver_proof"
                ),
                "chain_targets": chain_targets,
                "source_operation_ids": [
                    str(row.get("operation_id") or "") for row in registered_rows
                ],
                "materializer_levels": materializer_levels,
                "stage_plan": [
                    {
                        "stage": "payload_grammar_and_entropy",
                        "targets": ["byte_range_entropy_recode_v1"],
                        "required_before_execution": [
                            "schema_manifest",
                            "beam_probe_reports",
                            "source_runtime_dir",
                        ],
                    },
                    {
                        "stage": "archive_section_receiver_contracts",
                        "targets": [
                            "archive_section_header_elide_v1",
                            "archive_section_reorder_v1",
                            "archive_section_proceduralize_v1",
                        ],
                        "required_before_execution": [
                            "header_elision_contract",
                            "section_order_contract",
                            "runtime_consumption_proof",
                        ],
                    },
                    {
                        "stage": "tensor_scorer_sensitive_layout",
                        "targets": [
                            "tensor_quantize_v1",
                            "tensor_prune_v1",
                            "tensor_shared_codebook_v1",
                        ],
                        "required_before_execution": [
                            "component_sensitivity_rows",
                            "receiver_exact_reconstruction_contract",
                        ],
                    },
                    {
                        "stage": "packet_member_lookup_and_high_level_action_sets",
                        "targets": [
                            "packet_member_reorder_v1",
                            "inverse_steganalysis_high_level_operation_set_v1",
                        ],
                        "required_before_execution": [
                            "member_lookup_proof",
                            "inverse_scorer_action_surface_binding",
                        ],
                    },
                ],
                "synergy_terms_to_measure": [
                    "entropy_recode_after_section_reorder",
                    "header_elide_after_proceduralized_section_constants",
                    "shared_codebook_after_tensor_quant_prune",
                    "component_repair_budget_after_receiver_closed_rate_savings",
                    "single_receiver_adapter_amortizes_multiple_archive_ops",
                ],
                "antagonism_terms_to_measure": [
                    "section_reorder_breaks_payload_offsets_without_lookup_table",
                    "tensor_prune_changes_posenet_geometry_more_than_rate_credit",
                    "quantization_noise_consumes_segnet_boundary_budget",
                    "receiver_adapter_bytes_exceed_chained_rate_savings",
                ],
                "targeted_correction_budget": {
                    "schema": targeted_correction_budget.get("schema"),
                    "active": targeted_correction_budget.get("active"),
                    "receiver_closed_materializer_saved_bytes_total": receiver_closed_saved,
                    "local_drop_saved_bytes_total": local_drop_saved,
                    "blockers": _string_list(targeted_correction_budget.get("blockers")),
                    **FALSE_AUTHORITY,
                },
                "missing_contracts": missing_contracts,
                "queue_work_order_schema": OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA,
                **FALSE_AUTHORITY,
            },
            blockers=[
                "chain_requires_registered_materializer_context_binding",
                "chain_requires_single_composed_receiver_runtime_consumption_proof",
                "chain_requires_component_marginal_rows_before_budget_reallocation",
                "chain_requires_exact_readiness_handoff_after_composition",
                *[
                    f"chain_missing_contract:{contract}"
                    for contract in missing_contracts
                ],
            ],
            queue_executable=False,
            followup_signal=True,
            source_kind="registered_materializer_chain_acquisition",
            suppression_keys=chain_targets,
        )
    ]


def _targeted_correction_budget_summary(
    *,
    component_summary: Mapping[str, Any],
    materializer_rows: Sequence[Mapping[str, Any]],
    receiver_closed_correction_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    local_rows = [
        row
        for row in component_summary.get("local_video_behavior_rows") or []
        if isinstance(row, Mapping)
    ]
    local_byte_savings: list[int] = []
    local_rate_credits: list[float] = []
    score_credit_per_byte: list[float] = []
    for row in local_rows:
        archive_delta = _finite_int_or_none(row.get("archive_byte_delta_vs_baseline"))
        components = _component_deltas(row)
        if archive_delta is None or archive_delta >= 0 or components is None:
            continue
        saved_bytes = abs(archive_delta)
        local_byte_savings.append(saved_bytes)
        rate_delta = float(components["rate_delta"])
        if rate_delta < 0.0:
            credit = abs(rate_delta)
            local_rate_credits.append(credit)
            score_credit_per_byte.append(credit / float(saved_bytes))

    materializer_saved_bytes: list[int] = []
    materializer_targets: list[str] = []
    for row in materializer_rows:
        summary = row.get("evidence_summary")
        if not isinstance(summary, Mapping):
            continue
        saved = _finite_int_or_none(summary.get("total_positive_saved_bytes"))
        if saved is None or saved <= 0:
            continue
        materializer_saved_bytes.append(saved)
        target = str(summary.get("target_kind") or row.get("operation_id") or "")
        if target:
            materializer_targets.append(target)
    receiver_closed = (
        receiver_closed_correction_budget
        if isinstance(receiver_closed_correction_budget, Mapping)
        else {}
    )
    receiver_closed_saved_total = (
        _finite_int_or_none(receiver_closed.get("receiver_closed_saved_bytes_total"))
        or 0
    )
    receiver_closed_saved_max = (
        _finite_int_or_none(receiver_closed.get("receiver_closed_saved_bytes_max"))
        or 0
    )
    receiver_closed_candidate_count = (
        _finite_int_or_none(receiver_closed.get("receiver_closed_candidate_count"))
        or 0
    )
    estimated_credit_per_byte = max(score_credit_per_byte) if score_credit_per_byte else None
    receiver_closed_credit = (
        float(receiver_closed_saved_total) * float(estimated_credit_per_byte)
        if estimated_credit_per_byte is not None and receiver_closed_saved_total > 0
        else None
    )

    active = bool(local_byte_savings or materializer_saved_bytes or receiver_closed_saved_total)
    blockers: list[str] = []
    if not local_rate_credits:
        blockers.append("requires_component_measured_rate_credit_for_repair_budget")
    if materializer_saved_bytes:
        remaining_unclosed = max(sum(materializer_saved_bytes) - receiver_closed_saved_total, 0)
        if receiver_closed_saved_total <= 0:
            blockers.append(
                "materializer_saved_bytes_require_receiver_runtime_proof_before_spend"
            )
        elif remaining_unclosed:
            blockers.append(
                "some_materializer_saved_bytes_still_require_receiver_runtime_proof_before_spend"
            )
    if receiver_closed_saved_total > 0:
        blockers.append(
            "receiver_closed_materializer_bytes_require_component_eval_before_correction_spend"
        )
        blockers.append(
            "receiver_closed_materializer_bytes_still_require_exact_auth_eval_before_score_claim"
        )
    if component_summary.get("active") is not True:
        blockers.append("requires_segnet_posenet_component_behavior_rows")
    return {
        "schema": "frontier_rate_attack_targeted_correction_budget_summary.v1",
        "active": active,
        "local_drop_budget_candidate_count": len(local_byte_savings),
        "local_drop_saved_bytes_max": max(local_byte_savings) if local_byte_savings else 0,
        "local_drop_saved_bytes_total": sum(local_byte_savings),
        "local_drop_rate_credit_score_units_max": (
            max(local_rate_credits) if local_rate_credits else 0.0
        ),
        "local_drop_rate_credit_score_units_total": sum(local_rate_credits),
        "estimated_score_credit_per_saved_byte_max": (
            max(score_credit_per_byte) if score_credit_per_byte else None
        ),
        "materializer_rate_positive_saved_bytes_total": sum(materializer_saved_bytes),
        "materializer_rate_positive_saved_bytes_max": (
            max(materializer_saved_bytes) if materializer_saved_bytes else 0
        ),
        "materializer_budget_source_targets": _unique_strings(materializer_targets),
        "receiver_closed_materializer_budget_candidate_count": (
            receiver_closed_candidate_count
        ),
        "receiver_closed_materializer_saved_bytes_total": receiver_closed_saved_total,
        "receiver_closed_materializer_saved_bytes_max": receiver_closed_saved_max,
        "receiver_closed_materializer_budget_source_targets": list(
            receiver_closed.get("receiver_closed_budget_source_targets") or []
        ),
        "receiver_closed_materializer_estimated_score_credit_units_total": (
            receiver_closed_credit
        ),
        "receiver_closed_correction_budget_schema": receiver_closed.get("schema"),
        "receiver_closed_rate_budget_planning_active": receiver_closed_saved_total > 0,
        "recommended_next_action": (
            "feed_receiver_closed_rate_budget_into_targeted_segnet_posenet_repairs_"
            "and_accept_only_if_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes_improves"
            if receiver_closed_saved_total > 0
            else "compose_rate_positive_ops_with_targeted_segnet_posenet_repairs_and_accept_"
            "only_if_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes_improves"
        ),
        "blockers": _unique_strings(blockers),
        "allowed_use": "local_advisory_repair_budget_acquisition_planning_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _component_rate_preservation_rows(
    component_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in component_summary.get("local_video_behavior_rows") or []:
        if not isinstance(row, Mapping):
            continue
        archive_delta = _finite_int_or_none(row.get("archive_byte_delta_vs_baseline"))
        components = row.get("component_deltas")
        if archive_delta is None or archive_delta >= 0 or not isinstance(components, Mapping):
            continue
        saved_bytes = abs(archive_delta)
        rate_delta = _finite_float_or_none(components.get("rate_delta"))
        if rate_delta is None or rate_delta >= 0.0:
            rate_delta = -_rate_credit_score_units_for_saved_bytes(saved_bytes)
        rate_credit = abs(rate_delta)
        seg_delta = _finite_float_or_none(components.get("segnet_delta")) or 0.0
        pose_delta = _finite_float_or_none(components.get("posenet_delta")) or 0.0
        scorer_penalty = seg_delta + pose_delta
        score_delta = _finite_float_or_none(row.get("score_delta_vs_baseline"))
        if score_delta is None:
            score_delta = scorer_penalty - rate_credit
        candidate_id = str(row.get("candidate_id") or "unknown_candidate")
        distortion_debt = max(0.0, scorer_penalty)
        rows.append(
            {
                "schema": RATE_BUDGET_PRESERVATION_ROW_SCHEMA,
                "preservation_id": _bounded_content_key(
                    "rate_budget_preserve",
                    (candidate_id, row.get("family"), archive_delta, score_delta),
                ),
                "source_kind": "dqs1_component_rate_budget",
                "candidate_id": candidate_id,
                "family": row.get("family"),
                "target_kind": "dqs1_pairset_drop_pair_or_group",
                "selected_pair_indices": row.get("selected_pair_indices"),
                "saved_bytes": saved_bytes,
                "archive_byte_delta_vs_baseline": archive_delta,
                "rate_credit_score_units": rate_credit,
                "segnet_delta_score_units": seg_delta,
                "posenet_delta_score_units": pose_delta,
                "distortion_debt_score_units": distortion_debt,
                "net_score_delta_score_units": score_delta,
                "rate_only_archive_preservation_required": True,
                "preserve_as_rate_only_candidate": True,
                "budget_reinvestment_candidate": distortion_debt > 0.0,
                "budget_reinvestment_mode": (
                    "waterfill_distortion_repair_after_preserving_rate_only_floor"
                    if distortion_debt > 0.0
                    else "optional_recheck_only_no_positive_distortion_debt"
                ),
                "minimum_repair_score_units_to_beat_rate_only_floor": max(
                    0.0,
                    score_delta,
                ),
                "recommended_next_action": (
                    "preserve_rate_only_archive_then_waterfill_component_repair"
                    if distortion_debt > 0.0
                    else "preserve_rate_only_archive_as_floor_candidate"
                ),
                "allowed_use": "rate_only_floor_preservation_and_reinvestment_planning",
                "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _materializer_rate_preservation_rows(
    materializer_rows: Sequence[Mapping[str, Any]],
    receiver_closed_correction_budget: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    receiver_closed = (
        receiver_closed_correction_budget
        if isinstance(receiver_closed_correction_budget, Mapping)
        else {}
    )
    closed_by_target: dict[str, int] = {}
    for budget_row in receiver_closed.get("rows") or []:
        if not isinstance(budget_row, Mapping) or budget_row.get("receiver_closed") is not True:
            continue
        target = str(budget_row.get("target_kind") or "")
        saved = _finite_int_or_none(budget_row.get("saved_bytes_at_risk")) or 0
        if target and saved > 0:
            closed_by_target[target] = max(saved, closed_by_target.get(target, 0))
    matched_receiver_targets: set[str] = set()
    for row in materializer_rows:
        evidence = row.get("evidence_summary")
        if not isinstance(evidence, Mapping):
            continue
        target_kind = str(evidence.get("target_kind") or row.get("operation_id") or "")
        saved = _finite_int_or_none(evidence.get("total_positive_saved_bytes")) or 0
        if saved <= 0:
            saved = closed_by_target.get(target_kind, 0)
        if saved <= 0:
            continue
        matched_receiver_targets.add(target_kind)
        rate_credit = _rate_credit_score_units_for_saved_bytes(saved)
        candidate_id = str(row.get("operation_id") or target_kind or "materializer")
        rows.append(
            {
                "schema": RATE_BUDGET_PRESERVATION_ROW_SCHEMA,
                "preservation_id": _bounded_content_key(
                    "rate_budget_preserve",
                    (candidate_id, target_kind, saved),
                ),
                "source_kind": "receiver_closed_materializer_rate_budget",
                "candidate_id": candidate_id,
                "family": row.get("operation_family"),
                "target_kind": target_kind,
                "saved_bytes": saved,
                "archive_byte_delta_vs_baseline": -saved,
                "rate_credit_score_units": rate_credit,
                "segnet_delta_score_units": 0.0,
                "posenet_delta_score_units": 0.0,
                "distortion_debt_score_units": 0.0,
                "net_score_delta_score_units": -rate_credit,
                "receiver_closed_saved_bytes": closed_by_target.get(target_kind, 0),
                "rate_only_archive_preservation_required": True,
                "preserve_as_rate_only_candidate": True,
                "budget_reinvestment_candidate": True,
                "budget_reinvestment_mode": (
                    "preserve_lossless_rate_win_then_optionally_spend_credit_on_"
                    "component_guarded_repair"
                ),
                "minimum_repair_score_units_to_beat_rate_only_floor": 0.0,
                "recommended_next_action": (
                    "preserve_receiver_closed_rate_only_archive_before_any_"
                    "distortion_reinvestment"
                ),
                "allowed_use": "rate_only_floor_preservation_and_reinvestment_planning",
                "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
                **FALSE_AUTHORITY,
            }
        )
    for budget_row in receiver_closed.get("rows") or []:
        if not isinstance(budget_row, Mapping) or budget_row.get("receiver_closed") is not True:
            continue
        target_kind = str(budget_row.get("target_kind") or "")
        if not target_kind or target_kind in matched_receiver_targets:
            continue
        saved = _finite_int_or_none(budget_row.get("saved_bytes_at_risk")) or 0
        if saved <= 0:
            continue
        rate_credit = _rate_credit_score_units_for_saved_bytes(saved)
        candidate_id = str(
            budget_row.get("candidate_id") or target_kind or "receiver_closed_rate"
        )
        archive_delta = _finite_int_or_none(
            budget_row.get("archive_byte_delta_vs_parent")
        )
        if archive_delta is None:
            archive_delta = -saved
        rows.append(
            {
                "schema": RATE_BUDGET_PRESERVATION_ROW_SCHEMA,
                "preservation_id": _bounded_content_key(
                    "rate_budget_preserve",
                    (candidate_id, target_kind, saved, budget_row.get("archive_sha256")),
                ),
                "source_kind": str(
                    budget_row.get("source_kind")
                    or "receiver_closed_materializer_rate_budget"
                ),
                "candidate_id": candidate_id,
                "family": "receiver_closed_rate_packet",
                "target_kind": target_kind,
                "saved_bytes": saved,
                "archive_byte_delta_vs_baseline": archive_delta,
                "rate_credit_score_units": rate_credit,
                "segnet_delta_score_units": 0.0,
                "posenet_delta_score_units": 0.0,
                "distortion_debt_score_units": 0.0,
                "net_score_delta_score_units": -rate_credit,
                "receiver_closed_saved_bytes": saved,
                "rate_packet_manifest_path": budget_row.get("rate_packet_manifest_path"),
                "parent_rate_packet_manifest_path": budget_row.get(
                    "parent_rate_packet_manifest_path"
                ),
                "candidate_compact_selector_codec": budget_row.get(
                    "candidate_compact_selector_codec"
                ),
                "parent_compact_selector_codec": budget_row.get(
                    "parent_compact_selector_codec"
                ),
                "selector_policy_mode": budget_row.get("selector_policy_mode"),
                "archive_byte_delta_vs_parent": budget_row.get(
                    "archive_byte_delta_vs_parent"
                ),
                "selector_payload_wire_bytes": budget_row.get(
                    "selector_payload_wire_bytes"
                ),
                "parent_selector_payload_wire_bytes": budget_row.get(
                    "parent_selector_payload_wire_bytes"
                ),
                "selector_payload_wire_delta_bytes": budget_row.get(
                    "selector_payload_wire_delta_bytes"
                ),
                "selector_code_bits_total": budget_row.get("selector_code_bits_total"),
                "parent_selector_code_bits_total": budget_row.get(
                    "parent_selector_code_bits_total"
                ),
                "selector_avg_bits_per_pair": budget_row.get(
                    "selector_avg_bits_per_pair"
                ),
                "parent_selector_avg_bits_per_pair": budget_row.get(
                    "parent_selector_avg_bits_per_pair"
                ),
                "palette_size": budget_row.get("palette_size"),
                "n_pairs": budget_row.get("n_pairs"),
                "compact_palette_mode_ids": budget_row.get(
                    "compact_palette_mode_ids"
                ),
                "entropy_position": budget_row.get("entropy_position"),
                "rate_only_archive_preservation_required": True,
                "preserve_as_rate_only_candidate": True,
                "budget_reinvestment_candidate": True,
                "budget_reinvestment_mode": (
                    "preserve_receiver_closed_rate_packet_floor_then_optionally_"
                    "spend_credit_on_component_guarded_repair"
                ),
                "minimum_repair_score_units_to_beat_rate_only_floor": 0.0,
                "recommended_next_action": (
                    "feed_receiver_closed_rate_packet_credit_into_repair_waterfill_"
                    "after_preserving_parent_and_child_rate_only_floors"
                ),
                "allowed_use": "rate_only_floor_preservation_and_reinvestment_planning",
                "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _rate_budget_operator_action_term(
    row: Mapping[str, Any],
    *,
    rank: int,
) -> dict[str, Any]:
    saved_bytes = _finite_int_or_none(row.get("saved_bytes")) or 0
    archive_byte_delta = _finite_int_or_none(row.get("archive_byte_delta_vs_baseline"))
    if archive_byte_delta is None:
        archive_byte_delta = -saved_bytes
    segnet_delta = _finite_float_or_none(row.get("segnet_delta_score_units")) or 0.0
    posenet_delta = _finite_float_or_none(row.get("posenet_delta_score_units")) or 0.0
    rate_delta = rate_delta_for_archive_byte_delta(archive_byte_delta)
    distortion_debt = (
        _finite_float_or_none(row.get("distortion_debt_score_units")) or 0.0
    )
    objective_delta = segnet_delta + posenet_delta + rate_delta
    operator_action_id = _bounded_content_key(
        "operator_action_term",
        (
            row.get("preservation_id"),
            row.get("candidate_id"),
            row.get("target_kind"),
            archive_byte_delta,
            segnet_delta,
            posenet_delta,
            rank,
        ),
    )
    rate_packet_context = _targeted_rate_packet_context(row)
    return {
        "schema": OPERATOR_ACTION_TERM_SCHEMA,
        "rank": rank,
        "operator_action_id": operator_action_id,
        "preservation_id": row.get("preservation_id"),
        "candidate_id": row.get("candidate_id"),
        "source_kind": row.get("source_kind"),
        "target_kind": row.get("target_kind"),
        "T_i": {
            "operator_id": row.get("candidate_id") or operator_action_id,
            "family": row.get("family"),
            "target_kind": row.get("target_kind"),
            "archive_byte_delta_vs_baseline": archive_byte_delta,
            "saved_bytes": saved_bytes,
            "segnet_delta_score_units": segnet_delta,
            "posenet_delta_score_units": posenet_delta,
            "lambda_delta_bytes_score_units": rate_delta,
            "objective_delta_score_units": objective_delta,
            "distortion_debt_score_units": distortion_debt,
            **_targeted_rate_packet_context_fields(row),
            "receiver_closed_rate_packet_context": dict(rate_packet_context),
            **FALSE_AUTHORITY,
        },
        "R_i": {
            "receiver_proof_kind": "receiver_consumed_materialized_runtime_output",
            "receiver_closed_saved_bytes": row.get("receiver_closed_saved_bytes"),
            "receiver_closed_rate_packet_context": dict(rate_packet_context),
            "parser_only_proof_rejected": True,
            "deterministic_adapter_only": True,
            "receiver_optimizes_or_inspects_scorer": False,
            **FALSE_AUTHORITY,
        },
        "interaction_terms": {
            "schema": "frontier_rate_attack_operator_interaction_terms.v1",
            "status": "unmeasured_until_chain_materialization_and_component_replay",
            "assumed_independent_for_planning_only": True,
            "synergy_or_antagonism_score_units": None,
            "must_remeasure_before_promotion": True,
            **FALSE_AUTHORITY,
        },
        "legal_runtime_constraints": [
            "preserve_rate_only_floor_archive_before_distortion_budget_spend",
            "receiver_consumes_materialized_runtime_output",
            "component_response_replayed_before_budget_spend",
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ],
        "rate_only_floor_hard_constraint": True,
        "budget_spend_allowed": False,
        "allowed_use": "typed_operator_action_functional_ledger_term_for_queue_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _rate_budget_operator_action_ledger(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    terms = [
        _rate_budget_operator_action_term(row, rank=rank)
        for rank, row in enumerate(rows, start=1)
    ]
    archive_delta_total = sum(
        int(term["T_i"].get("archive_byte_delta_vs_baseline") or 0)
        for term in terms
    )
    segnet_delta_total = sum(
        float(term["T_i"].get("segnet_delta_score_units") or 0.0)
        for term in terms
    )
    posenet_delta_total = sum(
        float(term["T_i"].get("posenet_delta_score_units") or 0.0)
        for term in terms
    )
    rate_delta_total = rate_delta_for_archive_byte_delta(archive_delta_total)
    objective_delta_total = segnet_delta_total + posenet_delta_total + rate_delta_total
    return {
        "schema": OPERATOR_ACTION_LEDGER_SCHEMA,
        "term_schema": OPERATOR_ACTION_TERM_SCHEMA,
        "objective": "minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes",
        "objective_equation": "delta_S = delta_SegNet + delta_PoseNet + lambda * delta_bytes",
        "lambda_rate_score_per_byte": rate_delta_for_archive_byte_delta(1),
        "term_count": len(terms),
        "operator_action_ids": [
            str(term.get("operator_action_id") or "") for term in terms
        ],
        "cumulative": {
            "schema": "frontier_rate_attack_operator_action_cumulative_terms.v1",
            "archive_byte_delta_vs_baseline_total": archive_delta_total,
            "saved_bytes_total": max(0, -archive_delta_total),
            "segnet_delta_score_units_total": segnet_delta_total,
            "posenet_delta_score_units_total": posenet_delta_total,
            "lambda_delta_bytes_score_units_total": rate_delta_total,
            "objective_delta_score_units_total": objective_delta_total,
            **FALSE_AUTHORITY,
        },
        "rate_only_floor_hard_constraints": [
            "emit_cumulative_rate_only_archive_before_any_distortion_spend",
            "preserve_each_parent_rate_only_candidate_even_if_child_regresses",
            "budget_spend_child_must_reference_parent_rate_only_floor",
        ],
        "terms": terms,
        "budget_spend_allowed": False,
        "allowed_use": "canonical_action_functional_ledger_for_frontier_queue_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def build_frontier_rate_budget_preservation_plan(
    *,
    component_summary: Mapping[str, Any],
    materializer_rows: Sequence[Mapping[str, Any]],
    receiver_closed_correction_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve rate-only floors before spending byte credit on distortion repair."""

    rows = [
        *_component_rate_preservation_rows(component_summary),
        *_materializer_rate_preservation_rows(
            materializer_rows,
            receiver_closed_correction_budget,
        ),
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            -int(row.get("saved_bytes") or 0),
            float(row.get("net_score_delta_score_units") or 0.0),
            str(row.get("preservation_id") or ""),
        ),
    )
    rate_regression_rows = [
        row
        for row in rows
        if float(row.get("distortion_debt_score_units") or 0.0) > 0.0
        and float(row.get("net_score_delta_score_units") or 0.0) > 0.0
    ]
    total_saved = sum(int(row.get("saved_bytes") or 0) for row in rows)
    max_saved = max([int(row.get("saved_bytes") or 0) for row in rows] or [0])
    total_rate_credit = sum(float(row.get("rate_credit_score_units") or 0.0) for row in rows)
    total_distortion_debt = sum(
        float(row.get("distortion_debt_score_units") or 0.0) for row in rows
    )
    target_totals: dict[str, int] = {}
    for row in rows:
        target = str(row.get("target_kind") or "unknown_target")
        target_totals[target] = target_totals.get(target, 0) + int(
            row.get("saved_bytes") or 0
        )
    operator_action_ledger = _rate_budget_operator_action_ledger(rows)
    payload = {
        "schema": RATE_BUDGET_PRESERVATION_PLAN_SCHEMA,
        "generated_at_utc": _utc_now(),
        "active": bool(rows),
        "row_count": len(rows),
        "rate_only_candidate_count": len(rows),
        "rate_only_saved_bytes_total": total_saved,
        "rate_only_saved_bytes_max": max_saved,
        "rate_credit_score_units_total": total_rate_credit,
        "distortion_debt_score_units_total": total_distortion_debt,
        "rate_positive_distortion_regression_count": len(rate_regression_rows),
        "rate_positive_distortion_regression_candidate_ids": [
            str(row.get("candidate_id") or "") for row in rate_regression_rows[:16]
        ],
        "source_kinds": _unique_strings(row.get("source_kind") for row in rows),
        "action_functional": {
            "schema": "frontier_rate_attack_operator_action_functional.v1",
            "operator_semantics": (
                "each materializer is an operator T with receiver R_T; evaluate "
                "S(T)=delta_segnet(R_T(T(a)))+delta_posenet(R_T(T(a)))+"
                "lambda_rate*delta_archive_bytes(T(a))"
            ),
            "objective": "minimize_S_under_receiver_and_exact_readiness_constraints",
            "lambda_rate_score_per_byte": rate_delta_for_archive_byte_delta(1),
            "operator_action_ledger_schema": OPERATOR_ACTION_LEDGER_SCHEMA,
            "operator_action_term_schema": OPERATOR_ACTION_TERM_SCHEMA,
            "state_variables": [
                "archive_bytes",
                "decoded_video_frames",
                "segnet_component_distance",
                "posenet_component_distance",
                "receiver_runtime_contract",
                "operator_synergy_or_antagonism_terms",
            ],
            "constraints": [
                "preserve_rate_only_floor_archive_before_distortion_budget_spend",
                "receiver_consumes_materialized_runtime_output",
                "component_response_replayed_before_budget_spend",
                "exact_auth_eval_required_before_score_or_promotion_claim",
            ],
            "composition_law": (
                "compose operators in packet_archive_tensor_frame_pair_region bases, "
                "then remeasure interaction terms instead of assuming independent "
                "delta additivity"
            ),
            "discrete_solver": (
                "bounded_exact_knapsack_for_small_operator_sets; otherwise "
                "lagrangian_waterfill_over_measured_marginal_score_per_byte"
            ),
            "allowed_use": "canonical_planning_model_for_local_queue_acquisition",
            "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
            **FALSE_AUTHORITY,
        },
        "cumulative_rate_attack": {
            "schema": "frontier_rate_attack_cumulative_rate_attack_ledger.v1",
            "operator_count": len(rows),
            "operator_action_ledger_schema": operator_action_ledger.get("schema"),
            "operator_action_term_schema": OPERATOR_ACTION_TERM_SCHEMA,
            "operator_action_term_count": operator_action_ledger.get("term_count"),
            "operator_action_ids": list(
                operator_action_ledger.get("operator_action_ids") or []
            ),
            "saved_bytes_total": total_saved,
            "rate_credit_score_units_total": total_rate_credit,
            "targets_by_saved_bytes": [
                {"target_kind": target, "saved_bytes": saved}
                for target, saved in sorted(
                    target_totals.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "composition_policy": (
                "sum_receiver_consumed_rate_deltas_across_all_packet_archive_tensor_"
                "frame_pair_runtime_ops_then_measure_synergy_or_antagonism_before_"
                "promotion"
            ),
            "preserve_per_op_provenance": True,
            "preserve_cumulative_rate_only_archive": True,
            "emit_cumulative_rate_only_before_any_distortion_spend": True,
            **FALSE_AUTHORITY,
        },
        "waterfill_solver": {
            "schema": "frontier_rate_attack_budget_waterfill_solver.v1",
            "objective": (
                "minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes_"
                "while_preserving_each_rate_only_floor_candidate"
            ),
            "lambda_rate_score_per_byte": rate_delta_for_archive_byte_delta(1),
            "input_operator_action_ledger_schema": operator_action_ledger.get("schema"),
            "repair_allocation_action_term_schema": (
                REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
            ),
            "constraint": "0 <= spent_extra_bytes <= saved_bytes_for_candidate",
            "state_variables": [
                "saved_bytes_by_rate_only_candidate",
                "segnet_delta_by_repair_operator",
                "posenet_delta_by_repair_operator",
                "extra_bytes_by_repair_operator",
                "operator_synergy_or_antagonism_terms",
            ],
            "solver_policy": (
                "exact_integer_knapsack_for_bounded_candidate_sets_else_"
                "lagrangian_waterfill_sorted_by_component_score_improvement_per_byte"
            ),
            "acceptance_rule": (
                "emit_rate_only_archive_first; emit_spent_budget_archive_only_if_"
                "component_repair_delta_plus_extra_rate_delta_is_negative_relative_"
                "to_the_preserved_rate_only_floor"
            ),
            "rebrotli_default_after_rate_attack": True,
            "rebrotli_policy": (
                "after every receiver-consumed rate transform, enqueue packet_member_"
                "recompress_v1 or section Brotli retune where the runtime proof keeps "
                "payload semantics fixed"
            ),
            "budget_spend_allowed": False,
            "allowed_use": "mathematical_planning_surface_for_local_queue_only",
            "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
            **FALSE_AUTHORITY,
        },
        "operator_action_ledger": operator_action_ledger,
        "rows": rows,
        "blockers": _unique_strings(
            [
                "rate_only_archive_runtime_preservation_required_before_budget_spend",
                "component_replay_required_before_spent_budget_archive_promotion",
                "exact_auth_eval_required_before_score_or_promotion_claim",
                *(
                    ["no_rate_positive_rows_to_preserve"]
                    if not rows
                    else []
                ),
            ]
        ),
        "recommended_next_action": (
            "run_rate_attack_materializers_then_emit_rate_only_archive_and_budget_"
            "spent_archive_as_separate_candidates"
            if rows
            else "discover_receiver_closed_or_drop_many_rate_positive_candidates"
        ),
        "allowed_use": "rate_attack_preservation_and_budget_reinvestment_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="frontier_rate_budget_preservation_plan",
    )
    return payload


def _dqs1_component_operation_rows(
    observations: Sequence[Mapping[str, Any]],
    component_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not observations or component_summary.get("active") is not True:
        return []
    rows: list[dict[str, Any]] = []
    best_delta = _finite_float_or_none(component_summary.get("best_score_delta_vs_baseline"))
    best_candidate = str(component_summary.get("best_candidate_id") or "")
    pose_regressions = int(component_summary.get("pose_regression_count") or 0)
    seg_regressions = int(component_summary.get("segnet_regression_count") or 0)
    blockers: list[str] = []
    if pose_regressions:
        blockers.append("posenet_regression_rows_require_geometry_guard")
    if seg_regressions:
        blockers.append("segnet_regression_rows_require_mask_boundary_guard")
    if best_delta is None or best_delta >= 0.0:
        blockers.append("no_negative_local_component_score_delta_observed")
    rows.append(
        _operation_row(
            operation_id="dqs1_component_coupled_pair_batch_expansion",
            operation_family="decoder_q_component_coupled_pairset_search",
            operation_levels=["frame", "pair", "batch", "full_video", "scorer_axis"],
            queue_consumer="decoder_q_pairset_acquisition",
            recommended_next_action=(
                "expand_best_component_behavior_into_drop_many_pair_batch_surface"
                if best_delta is not None and best_delta < 0.0
                else "collect_more_component_harvest_rows_before_expansion"
            ),
            priority_score=75.0 + abs(best_delta or 0.0) * 1_000_000.0,
            evidence_summary={
                "best_candidate_id": best_candidate,
                "best_score_delta_vs_baseline": best_delta,
                "best_component_deltas": component_summary.get("best_component_deltas"),
                "best_selected_pair_indices": component_summary.get(
                    "best_selected_pair_indices"
                ),
                "negative_score_delta_count": component_summary.get(
                    "negative_score_delta_count"
                ),
                "component_marginal_status_counts": component_summary.get(
                    "component_marginal_status_counts"
                ),
                "targeted_correction_budget": {
                    "saved_bytes": max(
                        0,
                        -int(component_summary.get("best_archive_byte_delta_vs_baseline") or 0),
                    ),
                    "rate_credit_score_units": max(
                        0.0,
                        -float(
                            (
                                component_summary.get("best_component_deltas") or {}
                            ).get("rate_delta")
                            or 0.0
                        ),
                    ),
                    "spend_policy": (
                        "use_freed_rate_only_for_component_guarded_segnet_posenet_"
                        "corrections_that_improve_total_lagrangian"
                    ),
                },
            },
            blockers=blockers,
            queue_executable=best_delta is not None and best_delta < 0.0,
            followup_signal=best_delta is not None,
            source_kind="dqs1_component_behavior",
            component_behavior=component_summary,
        )
    )
    return rows


def _backlog_operation_rows(
    *,
    component_summary: Mapping[str, Any],
    master_gradient: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(
        (*_REGISTERED_MISSING_MATERIALIZER_SEEDS, *_MISSING_CLASS_SEEDS),
        start=1,
    ):
        blockers = [
            "missing_executable_materializer_or_receiver_binding",
            "requires_queue_context_and_runtime_consumption_proof",
        ]
        if "scorer_axis" in seed["operation_levels"] and component_summary.get("active") is not True:
            blockers.append("requires_segnet_posenet_component_behavior_rows")
        if (
            seed["operation_id"]
            in {
                "materializer_backlog_tensor_quantize_v1",
                "materializer_backlog_tensor_prune_v1",
                "materializer_backlog_inverse_steganalysis_high_level_operation_set_v1",
                "missing_class_segnet_posenet_geometry_transforms",
            }
            and master_gradient.get("active") is not True
        ):
            blockers.append("requires_master_gradient_or_inverse_scorer_binding")
        rows.append(
            _operation_row(
                operation_id=str(seed["operation_id"]),
                operation_family=str(seed["operation_family"]),
                operation_levels=seed["operation_levels"],
                queue_consumer=str(seed["queue_consumer"]),
                recommended_next_action=str(seed["recommended_next_action"]),
                priority_score=35.0 - float(index) / 10.0,
                evidence_summary={
                    "registered_backlog": seed["operation_id"].startswith(
                        "materializer_backlog_"
                    ),
                    "missing_entire_class": seed["operation_id"].startswith(
                        "missing_class_"
                    ),
                    "component_behavior_active": component_summary.get("active"),
                    "master_gradient_active": master_gradient.get("active"),
                },
                blockers=blockers,
                queue_executable=False,
                source_kind="operation_backlog",
            )
        )
    return rows


def _eureka_operation_rows(eureka_planning: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    hints = eureka_planning.get("planner_hints")
    if not isinstance(hints, Sequence) or isinstance(hints, (str, bytes, bytearray)):
        return out
    for hint in hints:
        if not isinstance(hint, Mapping):
            continue
        source_ids = _string_list(hint.get("source_candidate_ids"))
        for family in _string_list(hint.get("recommended_candidate_families")):
            metadata = _EUREKA_FAMILY_METADATA.get(family, {})
            blockers: list[str] = []
            profile = hint.get("pairset_acquisition_profile")
            if isinstance(profile, Mapping):
                for blocked in profile.get("blocked_family_requests") or []:
                    if isinstance(blocked, Mapping) and blocked.get("family") == family:
                        blockers.extend(_string_list(blocked.get("blocker")))
            queue_executable = bool(metadata.get("queue_executable")) and not blockers
            out.append(
                _operation_row(
                    operation_id=f"eureka_{family}",
                    operation_family=family,
                    operation_levels=metadata.get("levels", ["pair", "frame"]),
                    queue_consumer=str(metadata.get("queue_consumer") or "frontier_feedback_cycle"),
                    recommended_next_action=(
                        "materialize_local_queue_wave_from_near_frontier_eureka_cluster"
                        if queue_executable
                        else "compile_blocked_eureka_family_into_runtime_materializer"
                    ),
                    priority_score=65.0 if queue_executable else 45.0,
                    evidence_sources=[
                        str(row.get("path"))
                        for row in eureka_planning.get("signal_rows") or []
                        if isinstance(row, Mapping)
                        and str(row.get("path") or "")
                    ],
                    evidence_summary={
                        "hint_id": hint.get("hint_id"),
                        "trigger": hint.get("trigger"),
                        "source_candidate_ids": source_ids,
                        "best_projected_gap_vs_auth_frontier": eureka_planning.get(
                            "best_projected_gap_vs_auth_frontier"
                        ),
                        "best_conservative_gap_vs_auth_frontier": eureka_planning.get(
                            "best_conservative_gap_vs_auth_frontier"
                        ),
                    },
                    blockers=blockers,
                    queue_executable=queue_executable,
                    followup_signal=True,
                    source_kind="local_cpu_eureka_planning",
                )
            )
    return out


def _pair_frame_operation_rows(
    pair_frame_requests: Sequence[Mapping[str, Any]],
    source_paths: Sequence[str],
) -> list[dict[str, Any]]:
    if not pair_frame_requests:
        return []
    drop_counts = [
        len(request.get("dropped_pair_indices") or [])
        for request in pair_frame_requests
        if isinstance(request, Mapping)
    ]
    return [
        _operation_row(
            operation_id="pair_frame_geometry_queue_requests",
            operation_family="pair_frame_geometry_low_impact_drop_surface",
            operation_levels=["frame", "pair", "batch", "scorer_axis"],
            queue_consumer="dqs1_local_first_queue",
            recommended_next_action="execute_geometry_bound_pair_drop_requests_locally",
            priority_score=70.0 + float(max(drop_counts or [0])),
            evidence_sources=source_paths,
            evidence_summary={
                "request_count": len(pair_frame_requests),
                "max_drop_count": max(drop_counts or [0]),
                "candidate_ids": [
                    request.get("candidate_id") for request in pair_frame_requests
                ],
            },
            queue_executable=True,
            source_kind="pair_frame_geometry_lattice",
        )
    ]


def _seed_operation_rows(
    *,
    component_summary: Mapping[str, Any],
    master_gradient: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in _BROAD_OPERATION_SEEDS:
        blockers = _string_list(seed.get("blockers"))
        if seed["operation_id"] == "segnet_posenet_geometry_tradeoff_surface" and component_summary.get("active") is True:
            blockers = [
                blocker
                for blocker in blockers
                if blocker != "requires_component_deltas_or_local_advisory_harvest_rows"
            ]
        if seed["operation_id"] == "pair_batch_full_video_low_impact_drop_surface" and master_gradient.get("active") is True:
            blockers = [
                blocker
                for blocker in blockers
                if blocker != "requires_pair_frame_geometry_lattice_or_master_gradient_binding"
            ]
        rows.append(
            _operation_row(
                operation_id=str(seed["operation_id"]),
                operation_family=str(seed["operation_family"]),
                operation_levels=seed["operation_levels"],
                queue_consumer=str(seed["queue_consumer"]),
                recommended_next_action=str(seed["recommended_next_action"]),
                priority_score=float(seed["priority_score"]),
                evidence_summary={
                    "component_behavior_active": component_summary.get("active"),
                    "master_gradient_active": master_gradient.get("active"),
                    "master_gradient_anchor_count": master_gradient.get("anchor_count"),
                },
                blockers=blockers,
                queue_executable=not blockers
                and seed["queue_consumer"]
                in {
                    "decoder_q_pairset_acquisition",
                    "pair_frame_scorer_geometry_lattice",
                },
                source_kind="broad_operation_seed",
                component_behavior=(
                    component_summary
                    if seed["operation_id"] == "segnet_posenet_geometry_tradeoff_surface"
                    else {}
                ),
            )
        )
    return rows


def build_frontier_operation_portfolio(
    *,
    repo_root: str | Path,
    materializer_feedback_payloads: Sequence[Mapping[str, Any]],
    materializer_feedback_source_paths: Sequence[str],
    dqs1_observations: Sequence[Mapping[str, Any]],
    eureka_planning: Mapping[str, Any],
    pair_frame_requests: Sequence[Mapping[str, Any]],
    pair_frame_source_paths: Sequence[str],
    receiver_closed_correction_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose many operation families into one queue-owned acquisition surface."""

    repo = Path(repo_root)
    component_summary = _component_behavior_summary(dqs1_observations)
    master_gradient = _master_gradient_summary(repo_root=repo)
    materializer_rows = _materializer_operation_rows(
        materializer_feedback_payloads,
        materializer_feedback_source_paths,
        repo_root=repo,
    )
    targeted_correction_budget = _targeted_correction_budget_summary(
        component_summary=component_summary,
        materializer_rows=materializer_rows,
        receiver_closed_correction_budget=receiver_closed_correction_budget,
    )
    rate_budget_preservation_plan = build_frontier_rate_budget_preservation_plan(
        component_summary=component_summary,
        materializer_rows=materializer_rows,
        receiver_closed_correction_budget=receiver_closed_correction_budget,
    )
    backlog_rows = _backlog_operation_rows(
        component_summary=component_summary,
        master_gradient=master_gradient,
    )
    rows = [
        *materializer_rows,
        *_materializer_chain_operation_rows(materializer_rows),
        *_registered_materializer_chain_operation_rows(
            backlog_rows,
            targeted_correction_budget=targeted_correction_budget,
        ),
        *_dqs1_component_operation_rows(dqs1_observations, component_summary),
        *_eureka_operation_rows(eureka_planning),
        *_pair_frame_operation_rows(pair_frame_requests, pair_frame_source_paths),
        *_seed_operation_rows(
            component_summary=component_summary,
            master_gradient=master_gradient,
        ),
        *backlog_rows,
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get("priority_score") or 0.0),
            str(row.get("operation_id") or ""),
        ),
    )
    level_counts: dict[str, int] = {}
    for row in rows:
        for level in _string_list(row.get("operation_levels")):
            level_counts[level] = level_counts.get(level, 0) + 1
    queue_executable_rows = [
        row for row in rows if row.get("queue_executable") is True
    ]
    followup_signal_rows = [
        row for row in rows if row.get("followup_signal") is True
    ]
    return {
        "schema": OPERATION_PORTFOLIO_SCHEMA,
        "generated_at_utc": _utc_now(),
        "taxonomy": _operation_portfolio_taxonomy(),
        "operation_level_taxonomy": list(_OPERATION_LEVELS),
        "operation_count": len(rows),
        "queue_executable_operation_count": len(queue_executable_rows),
        "followup_signal_operation_count": len(followup_signal_rows),
        "blocked_operation_count": sum(1 for row in rows if row.get("blockers")),
        "suppression_key_count": sum(
            len(_string_list(row.get("suppression_keys"))) for row in rows
        ),
        "operation_level_counts": dict(sorted(level_counts.items())),
        "top_operation_ids": [
            str(row.get("operation_id") or "") for row in rows[:8]
        ],
        "top_queue_executable_operation_ids": [
            str(row.get("operation_id") or "") for row in queue_executable_rows[:8]
        ],
        "top_followup_signal_operation_ids": [
            str(row.get("operation_id") or "") for row in followup_signal_rows[:8]
        ],
        "component_behavior_summary": component_summary,
        "targeted_correction_budget_summary": targeted_correction_budget,
        "rate_budget_preservation_plan": rate_budget_preservation_plan,
        "receiver_closed_correction_budget": dict(receiver_closed_correction_budget or {}),
        "master_gradient_summary": master_gradient,
        "rows": rows,
        "allowed_use": "queue_owned_many_operation_acquisition_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _operation_portfolio_queue_metadata(portfolio: Mapping[str, Any]) -> dict[str, Any]:
    component = (
        portfolio.get("component_behavior_summary")
        if isinstance(portfolio.get("component_behavior_summary"), Mapping)
        else {}
    )
    correction_budget = (
        portfolio.get("targeted_correction_budget_summary")
        if isinstance(portfolio.get("targeted_correction_budget_summary"), Mapping)
        else {}
    )
    preservation_plan = (
        portfolio.get("rate_budget_preservation_plan")
        if isinstance(portfolio.get("rate_budget_preservation_plan"), Mapping)
        else {}
    )
    receiver_closed = (
        portfolio.get("receiver_closed_correction_budget")
        if isinstance(portfolio.get("receiver_closed_correction_budget"), Mapping)
        else {}
    )
    return {
        "schema": "frontier_rate_attack_operation_portfolio_queue_metadata.v1",
        "operation_portfolio_schema": portfolio.get("schema"),
        "operation_count": portfolio.get("operation_count"),
        "queue_executable_operation_count": portfolio.get(
            "queue_executable_operation_count"
        ),
        "followup_signal_operation_count": portfolio.get(
            "followup_signal_operation_count"
        ),
        "blocked_operation_count": portfolio.get("blocked_operation_count"),
        "top_operation_ids": list(portfolio.get("top_operation_ids") or []),
        "top_queue_executable_operation_ids": list(
            portfolio.get("top_queue_executable_operation_ids") or []
        ),
        "top_followup_signal_operation_ids": list(
            portfolio.get("top_followup_signal_operation_ids") or []
        ),
        "component_behavior_active": component.get("active") is True,
        "best_component_behavior_candidate_id": component.get("best_candidate_id"),
        "best_component_behavior_score_delta_vs_baseline": component.get(
            "best_score_delta_vs_baseline"
        ),
        "targeted_correction_budget_active": correction_budget.get("active") is True,
        "targeted_correction_budget_saved_bytes_total": correction_budget.get(
            "materializer_rate_positive_saved_bytes_total"
        ),
        "rate_budget_preservation_active": preservation_plan.get("active") is True,
        "rate_only_candidate_count": preservation_plan.get("rate_only_candidate_count"),
        "rate_only_saved_bytes_total": preservation_plan.get(
            "rate_only_saved_bytes_total"
        ),
        "rate_positive_distortion_regression_count": preservation_plan.get(
            "rate_positive_distortion_regression_count"
        ),
        "receiver_closed_correction_budget_active": receiver_closed.get("active")
        is True,
        "receiver_closed_materializer_saved_bytes_total": receiver_closed.get(
            "receiver_closed_saved_bytes_total"
        ),
        "receiver_closed_materializer_budget_candidate_count": receiver_closed.get(
            "receiver_closed_candidate_count"
        ),
        "allowed_use": "queue_metadata_pointer_to_operation_portfolio_artifact",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _targeted_component_correction_wire_hooks(
    correction_family: str,
) -> dict[str, Any]:
    return {
        "schema": "frontier_rate_attack_targeted_component_wire_hooks.v1",
        "correction_family": correction_family,
        "sensitivity_map": [
            "component_deltas_by_pair_frame_axis",
            "master_gradient_pair_region_boundary_prior",
        ],
        "pareto_constraint": [
            "accept_only_if_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes_improves",
            "preserve_receiver_closed_static_runtime_contract",
        ],
        "bit_allocator": [
            "spend_receiver_closed_rate_credit_on_highest_component_repair_roi",
            "reserve_exact_rate_floor_for_byte_closed_materializers",
        ],
        "cathedral_autopilot_dispatch": [
            "frontier_targeted_component_correction_acquisition_queue",
            "dqs1_local_first_component_guarded_followup_queue",
        ],
        "continual_learning": [
            "append_component_eval_observation_rows_after_local_or_exact_probe",
            "update_pairset_component_marginal_model_before_next_acquisition",
        ],
        "probe_disambiguator": [
            "drop_within_set_vs_boundary_feather_masking",
            "segnet_repair_vs_posenet_regression_tradeoff",
        ],
        "allowed_use": "wire_in_plan_for_queue_owned_component_correction_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _targeted_component_correction_acquisition_queue_metadata(
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": TARGETED_COMPONENT_CORRECTION_QUEUE_METADATA_SCHEMA,
        "targeted_component_correction_acquisition_schema": acquisition.get("schema"),
        "active": acquisition.get("active") is True,
        "row_count": acquisition.get("row_count"),
        "queue_actionable_acquisition_count": acquisition.get(
            "queue_actionable_acquisition_count"
        ),
        "receiver_closed_saved_bytes_total": acquisition.get(
            "receiver_closed_saved_bytes_total"
        ),
        "receiver_closed_rate_credit_score_units_total": acquisition.get(
            "receiver_closed_rate_credit_score_units_total"
        ),
        "best_component_behavior_candidate_id": acquisition.get(
            "best_component_behavior_candidate_id"
        ),
        "top_acquisition_ids": list(acquisition.get("top_acquisition_ids") or []),
        "top_correction_families": list(
            acquisition.get("top_correction_families") or []
        ),
        "blockers": list(acquisition.get("blockers") or []),
        "allowed_use": "queue_metadata_pointer_to_component_correction_acquisition",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _targeted_component_correction_row_by_id(
    acquisition: Mapping[str, Any],
    acquisition_id: str,
) -> Mapping[str, Any]:
    for row in acquisition.get("rows") or []:
        if isinstance(row, Mapping) and row.get("acquisition_id") == acquisition_id:
            return row
    raise FrontierRateAttackFeedbackError(
        f"unknown targeted component correction acquisition id: {acquisition_id}"
    )


def _targeted_component_correction_command_hints(
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    family = str(row.get("correction_family") or "")
    hints: list[dict[str, Any]] = [
        {
            "action_id": "build_pair_frame_geometry_lattice_for_component_guard",
            "command_template": [
                ".venv/bin/python",
                "tools/build_pair_frame_scorer_geometry_lattice.py",
                "--pairset-acquisition",
                "<pairset_acquisition.json>",
                "--pair-component-xray",
                "<pair_component_xray.json>",
                "--drop-counts",
                "2,3,4,6,8,12,16",
                "--max-requests",
                "32",
                "--json-out",
                "<component_correction_dir>/pair_frame_geometry_lattice.json",
            ],
            "blocked_until": [
                "pairset_acquisition_path",
                "pair_component_or_component_marginal_xray",
            ],
            **FALSE_AUTHORITY,
        },
        {
            "action_id": "run_feedback_cycle_after_component_probe",
            "command_template": [
                ".venv/bin/python",
                "tools/run_frontier_rate_attack_feedback_cycle.py",
                "--action-summary",
                "latest",
                "--dqs1-observation-jsonl",
                "<component_probe_observations.jsonl>",
                "--output-dir",
                "<component_correction_dir>/post_probe_feedback_cycle",
            ],
            "blocked_until": [
                "component_probe_observations_jsonl",
                "false_authority_observation_rows",
            ],
            **FALSE_AUTHORITY,
        },
    ]
    if "boundary" in _string_list(row.get("operation_levels")):
        hints.append(
            {
                "action_id": "build_segnet_boundary_marginals",
                "command_template": [
                    ".venv/bin/python",
                    "tools/build_segnet_boundary_marginals.py",
                    "--json-out",
                    "<component_correction_dir>/segnet_boundary_marginals.json",
                    "--device",
                    "cpu",
                    "--max-pairs",
                    "600",
                ],
                "blocked_until": ["upstream_video_and_segnet_runtime_available"],
                **FALSE_AUTHORITY,
            }
        )
    if family in {
        "drop_within_selected_set_masked_boundary",
        "inverse_scorer_cell_basis_expansion",
    }:
        hints.append(
            {
                "action_id": "plan_decoder_q_pairset_acquisition_drop_many",
                "command_template": [
                    ".venv/bin/python",
                    "tools/plan_decoder_q_pairset_acquisition.py",
                    "--selector-pareto",
                    "<selector_pareto.json>",
                    "--drop-many-counts",
                    "2,3,4,6,8,12,16",
                    "--max-drop-many",
                    "64",
                    "--pair-frame-geometry-lattice-json",
                    "<component_correction_dir>/pair_frame_geometry_lattice.json",
                    "--json-out",
                    "<component_correction_dir>/pairset_acquisition.json",
                ],
                "blocked_until": ["selector_pareto", "pair_frame_geometry_lattice"],
                **FALSE_AUTHORITY,
            }
        )
    if family in {
        "segnet_posenet_waterfill_region_repair",
        "pose_stable_pair_frame_motion_correction",
        "full_video_batch_residual_budget_reallocation",
    }:
        hints.append(
            {
                "action_id": "plan_component_repair_waterbucket",
                "command_template": [
                    ".venv/bin/python",
                    "tools/plan_decoder_q_signed_waterbucket.py",
                    "--feasibility-json",
                    "<fixed_length_feasibility.json>",
                    "--output-dir",
                    "<component_correction_dir>/signed_waterbucket",
                ],
                "blocked_until": [
                    "fixed_length_runtime_compatible_component_atoms",
                    "signed_component_calibration",
                ],
                **FALSE_AUTHORITY,
            }
        )
    if family.startswith("repair_dynamics_"):
        hints.append(
            {
                "action_id": "build_repair_dynamics_palette_probe_matrix",
                "command_template": [
                    ".venv/bin/python",
                    "tools/build_repair_dynamics_palette_probe_matrix.py",
                    "--work-order",
                    "<component_correction_dir>/repair_dynamics_work_order.json",
                    "--matrix-out",
                    "<component_correction_dir>/repair_dynamics_palette_probe_matrix.json",
                    "--probe-output-dir",
                    "<component_correction_dir>/repair_dynamics_palette_probe",
                    "--device",
                    "mlx",
                    "--n-pairs",
                    "48",
                ],
                "blocked_until": [
                    "repair_dynamics_work_order",
                    "repair_dynamics_palette_prior",
                    "receiver_closed_archive",
                ],
                "output_contract": (
                    "false_authority_repair_dynamics_palette_probe_matrix"
                ),
                **FALSE_AUTHORITY,
            }
        )
        hints.append(
            {
                "action_id": "build_repair_dynamics_component_work_order",
                "command_template": [
                    ".venv/bin/python",
                    "tools/build_frontier_targeted_component_correction_work_order.py",
                    "--targeted-component-correction-acquisition",
                    "<targeted_component_correction_acquisition.json>",
                    "--acquisition-id",
                    str(row.get("acquisition_id") or "<repair_dynamics_acquisition_id>"),
                    "--work-order-out",
                    "<component_correction_dir>/repair_dynamics_work_order.json",
                ],
                "blocked_until": [
                    "targeted_component_correction_acquisition",
                    "repair_dynamics_acquisition_id",
                ],
                **FALSE_AUTHORITY,
            }
        )
    return hints


def _slug_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in out.split("_") if part) or "unknown"


def _bounded_content_key(
    prefix: str,
    parts: Sequence[Any],
    *,
    slug_chars: int = 48,
) -> str:
    raw = "\0".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    display_source = next((str(part or "") for part in parts if str(part or "")), "")
    display = _slug_token(display_source)[:slug_chars].strip("_")
    stem = _slug_token(prefix)
    return f"{stem}_{display}_{digest}" if display else f"{stem}_{digest}"


def _rate_credit_score_units_for_saved_bytes(saved_bytes: int) -> float:
    return abs(rate_delta_for_archive_byte_delta(-int(saved_bytes)))


def _materializer_registry_adapters_by_target() -> dict[str, Mapping[str, Any]]:
    return {
        str(adapter.get("target_kind") or ""): adapter
        for adapter in registry_manifest().get("adapters", [])
        if isinstance(adapter, Mapping) and adapter.get("target_kind")
    }


def _portfolio_row_materializer_target_kind(
    row: Mapping[str, Any],
    *,
    adapters_by_target: Mapping[str, Mapping[str, Any]],
) -> str | None:
    evidence = row.get("evidence_summary")
    if isinstance(evidence, Mapping):
        target = str(evidence.get("target_kind") or "").strip()
        if target in adapters_by_target:
            return target
    operation_id = str(row.get("operation_id") or "").strip()
    for prefix in ("materializer_", "materializer_backlog_"):
        if not operation_id.startswith(prefix):
            continue
        target = operation_id.removeprefix(prefix)
        if target in adapters_by_target:
            return target
    return None


def _portfolio_row_candidate_saved_bytes(row: Mapping[str, Any]) -> int:
    evidence = row.get("evidence_summary")
    if not isinstance(evidence, Mapping):
        evidence = {}
    for key in (
        "total_positive_saved_bytes",
        "max_saved_bytes",
        "saved_bytes_at_risk",
        "receiver_closed_saved_bytes_total",
    ):
        value = _finite_int_or_none(evidence.get(key))
        if value is not None and value > 0:
            return value
    return 0


def _portfolio_materializer_suggested_row(
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "byte_shaving_suggested_materializer.v1",
        "materializer_id": adapter.get("materializer_id"),
        "target_kind": adapter.get("target_kind"),
        "receiver_contract_id": adapter.get("receiver_contract_id"),
        "receiver_contract_kind": adapter.get("receiver_contract_kind"),
        "cooperative_receiver_required": (
            adapter.get("cooperative_receiver_required") is True
        ),
        "materialization_resource_kind": (
            adapter.get("materialization_resource_kind") or "local_cpu"
        ),
        "required_context_fields": list(adapter.get("required_context_fields") or []),
        "executable": adapter.get("executable") is True,
        "emits_candidate_archive": adapter.get("emits_candidate_archive") is not False,
        "planning_only": adapter.get("planning_only") is True,
        **FALSE_AUTHORITY,
    }


def _portfolio_materializer_backlog_row(
    *,
    row: Mapping[str, Any],
    adapter: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    operation_id = str(row.get("operation_id") or f"operation_{rank}")
    target_kind = str(adapter.get("target_kind") or "unknown_target")
    backlog_key = (
        "frontier_operation_portfolio:"
        f"{_slug_token(operation_id)}:{_slug_token(target_kind)}"
    )
    blockers = _string_list(row.get("blockers"))
    blocker_counts = {blocker: blockers.count(blocker) for blocker in blockers}
    saved_bytes = _portfolio_row_candidate_saved_bytes(row)
    priority = _finite_float_or_none(row.get("priority_score")) or 0.0
    suggested = _portfolio_materializer_suggested_row(adapter)
    context_hint = _portfolio_materializer_context_hint(row)
    operation_params = {
        "frontier_operation_id": operation_id,
        "frontier_operation_family": row.get("operation_family"),
        "frontier_queue_consumer": row.get("queue_consumer"),
        **context_hint,
    }
    return {
        "schema": "byte_shaving_materializer_backlog_row.v1",
        "backlog_key": backlog_key,
        "gap_class": "frontier_operation_portfolio_context_binding",
        "target_kind": target_kind,
        "materializer_id": adapter.get("materializer_id"),
        "receiver_contract_id": adapter.get("receiver_contract_id"),
        "receiver_contract_kind": adapter.get("receiver_contract_kind"),
        "receiver_contract_status": (
            "receiver_context_required_before_exact_readiness"
        ),
        "cooperative_receiver_required": (
            adapter.get("cooperative_receiver_required") is True
        ),
        "materialization_resource_kind": (
            adapter.get("materialization_resource_kind") or "local_cpu"
        ),
        "suggested_materializer_count": 1,
        "suggested_materializers": [suggested],
        "unit_kind": adapter.get("unit_kind"),
        "operation_family": adapter.get("operation_family"),
        "blocked_row_count": 1,
        "blocked_resolution_count": 1,
        "selected_operation_count": 1,
        "affected_unit_count": 1,
        "candidate_saved_bytes_sum": saved_bytes,
        "expected_score_gain_sum": priority,
        "best_expected_score_gain": priority,
        "best_expected_delta_score": None,
        "best_candidate_saved_bytes": saved_bytes,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "source_unit_ids": [operation_id],
        "source_selection_ids": [operation_id],
        "source_selection_samples": [
            {
                "selection_id": operation_id,
                "selection_kind": row.get("source_kind"),
                "unit_id": operation_id,
                "candidate_saved_bytes": saved_bytes,
                "expected_score_gain": priority,
                "expected_delta_score": None,
            }
        ],
        "source_packet_ir_operation_indices": [],
        "source_packet_ir_operation_indices_by_unit": {},
        "operation_params": operation_params,
        "source_operation_params_by_unit": {
            operation_id: dict(operation_params)
        },
        "frontier_operation_portfolio_row": {
            "operation_id": operation_id,
            "operation_family": row.get("operation_family"),
            "operation_levels": list(row.get("operation_levels") or []),
            "queue_consumer": row.get("queue_consumer"),
            "recommended_next_action": row.get("recommended_next_action"),
            "evidence_sources": list(row.get("evidence_sources") or []),
            "evidence_summary": dict(
                row.get("evidence_summary")
                if isinstance(row.get("evidence_summary"), Mapping)
                else {}
            ),
            "blockers": blockers,
            **FALSE_AUTHORITY,
        },
        "backlog_rank": rank,
        "implementation_priority_score": priority + float(saved_bytes) * 1e-9,
        **FALSE_AUTHORITY,
    }


def _operation_materializer_bridge_selection_key(
    item: tuple[Mapping[str, Any], Mapping[str, Any], str],
) -> tuple[int, int, int, int, float, float, str]:
    row, adapter, target_kind = item
    source_kind = str(row.get("source_kind") or "")
    evidence = row.get("evidence_summary")
    has_target_evidence = (
        isinstance(evidence, Mapping)
        and str(evidence.get("target_kind") or "") == target_kind
    )
    concrete_signal = source_kind == "materializer_feedback" or has_target_evidence
    saved_bytes = _portfolio_row_candidate_saved_bytes(row)
    return (
        0 if concrete_signal else 1,
        0 if row.get("queue_executable") is True else 1,
        0 if row.get("followup_signal") is True else 1,
        0 if adapter.get("executable") is True else 1,
        -float(saved_bytes),
        -float(_finite_float_or_none(row.get("priority_score")) or 0.0),
        str(row.get("operation_id") or ""),
    )


def build_frontier_operation_materializer_bridge(
    *,
    repo_root: str | Path,
    operation_portfolio: Mapping[str, Any],
    default_output_root: str | Path | None = None,
    candidate_limit: int | None = None,
) -> dict[str, Any]:
    """Compile operation-portfolio rows into materializer compiler surfaces."""

    repo = Path(repo_root)
    if candidate_limit is not None and candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    require_no_truthy_authority_fields(
        operation_portfolio,
        context="operation_materializer_bridge_operation_portfolio",
    )
    adapters_by_target = _materializer_registry_adapters_by_target()
    selected_rows: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = []
    bridge_rows: list[dict[str, Any]] = []
    for portfolio_index, source_row in enumerate(operation_portfolio.get("rows") or []):
        if not isinstance(source_row, Mapping):
            continue
        require_no_truthy_authority_fields(
            source_row,
            context=f"operation_materializer_bridge.rows.{portfolio_index}",
        )
        queue_consumer = str(source_row.get("queue_consumer") or "")
        operation_id = str(source_row.get("operation_id") or f"operation_{portfolio_index}")
        evidence = (
            source_row.get("evidence_summary")
            if isinstance(source_row.get("evidence_summary"), Mapping)
            else {}
        )
        chain_targets = _string_list(evidence.get("chain_targets"))
        materializer_consumer = queue_consumer in {
            "frontier_final_rate_attack_materializer_queue",
            "byte_shaving_campaign_queue",
        }
        target_kind = _portfolio_row_materializer_target_kind(
            source_row,
            adapters_by_target=adapters_by_target,
        )
        if not materializer_consumer and target_kind is None and not chain_targets:
            continue
        adapter = adapters_by_target.get(target_kind or "")
        bridge_blockers = _string_list(source_row.get("blockers"))
        chain_compiler_work_order: dict[str, Any] | None = None
        if adapter is None and chain_targets:
            bridge_blockers.append(
                "operation_portfolio_chain_requires_chain_compiler_work_order"
            )
            chain_compiler_work_order = {
                "schema": OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA,
                "source_operation_id": operation_id,
                "source_operation_family": source_row.get("operation_family"),
                "chain_targets": chain_targets,
                "required_before_execution": [
                    "per_stage_materializer_contexts",
                    "single_composed_receiver_runtime_consumption_proof",
                    "chain_exact_readiness_bridge",
                    "targeted_component_budget_spend_gate",
                ],
                "targeted_correction_budget": (
                    evidence.get("targeted_correction_budget")
                    if isinstance(evidence.get("targeted_correction_budget"), Mapping)
                    else {}
                ),
                "stage_plan": (
                    evidence.get("stage_plan")
                    if isinstance(evidence.get("stage_plan"), Sequence)
                    and not isinstance(evidence.get("stage_plan"), (str, bytes, bytearray))
                    else []
                ),
                "allowed_use": (
                    "operation_chain_compiler_work_order_planning_only"
                ),
                "forbidden_use": (
                    "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                ),
                **FALSE_AUTHORITY,
            }
        elif adapter is None:
            bridge_blockers.append(
                "operation_portfolio_row_has_no_registered_materializer_target"
            )
        bridge_row = {
            "schema": OPERATION_MATERIALIZER_BRIDGE_ROW_SCHEMA,
            "source_operation_id": operation_id,
            "source_operation_family": source_row.get("operation_family"),
            "source_kind": source_row.get("source_kind"),
            "queue_consumer": queue_consumer,
            "operation_levels": list(source_row.get("operation_levels") or []),
            "target_kind": target_kind,
            "materializer_id": None if adapter is None else adapter.get("materializer_id"),
            "unit_kind": None if adapter is None else adapter.get("unit_kind"),
            "operation_family": (
                None if adapter is None else adapter.get("operation_family")
            ),
            "receiver_contract_id": (
                None if adapter is None else adapter.get("receiver_contract_id")
            ),
            "receiver_contract_kind": (
                None if adapter is None else adapter.get("receiver_contract_kind")
            ),
            "chain_targets": chain_targets,
            "chain_compiler_work_order": chain_compiler_work_order,
            "candidate_saved_bytes": _portfolio_row_candidate_saved_bytes(source_row),
            "priority_score": source_row.get("priority_score"),
            "materializer_backlog_key": (
                None
                if adapter is None
                else (
                    "frontier_operation_portfolio:"
                    f"{_slug_token(operation_id)}:{_slug_token(target_kind)}"
                )
            ),
            "queue_actionable": adapter is not None,
            "blockers": _unique_strings(bridge_blockers),
            "allowed_use": "frontier_operation_to_materializer_compiler_bridge_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        }
        bridge_rows.append(bridge_row)
        if adapter is not None:
            selected_rows.append((source_row, adapter, target_kind or ""))
    selected_rows = sorted(selected_rows, key=_operation_materializer_bridge_selection_key)
    if candidate_limit is not None:
        selected_rows = selected_rows[:candidate_limit]

    backlog_rows = [
        _portfolio_materializer_backlog_row(row=row, adapter=adapter, rank=rank)
        for rank, (row, adapter, _target_kind) in enumerate(selected_rows, start=1)
    ]
    materializer_backlog = {
        "schema": MATERIALIZER_BACKLOG_SCHEMA,
        "tool": "comma_lab.scheduler.frontier_rate_attack_feedback",
        "generated_at_utc": _utc_now(),
        "backlog_row_count": len(backlog_rows),
        "rows": backlog_rows,
        "allowed_use": "frontier_operation_portfolio_to_materializer_backlog_only",
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    materializer_contexts = build_final_byte_operation_contexts(
        materializer_backlog,
        artifact_map=None,
        repo_root=repo,
        default_output_root=default_output_root,
    )
    materializer_work_queue = build_materializer_work_queue(
        materializer_backlog,
        repo_root=repo,
        contexts=materializer_contexts_from_payload(materializer_contexts),
        source_plan_path=None,
        limit=candidate_limit,
    )
    bridge = {
        "schema": OPERATION_MATERIALIZER_BRIDGE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "operation_portfolio_schema": operation_portfolio.get("schema"),
        "operation_count": operation_portfolio.get("operation_count"),
        "bridge_row_count": len(bridge_rows),
        "materializer_backlog_row_count": len(backlog_rows),
        "context_row_count": materializer_contexts.get("row_count"),
        "blocked_context_count": materializer_contexts.get("blocked_context_count"),
        "work_queue_row_count": materializer_work_queue.get("row_count"),
        "executable_work_row_count": materializer_work_queue.get(
            "executable_row_count"
        ),
        "blocked_work_row_count": materializer_work_queue.get("blocked_row_count"),
        "top_source_operation_ids": [
            str(row.get("source_operation_id") or "") for row in bridge_rows[:8]
        ],
        "top_materializer_targets": _unique_strings(
            row.get("target_kind") for row in bridge_rows[:8]
        ),
        "selected_source_operation_ids": [
            str(row.get("operation_id") or "")
            for row, _adapter, _target_kind in selected_rows
        ],
        "selected_materializer_targets": _unique_strings(
            row.get("target_kind") for row in backlog_rows
        ),
        "materializer_backlog": materializer_backlog,
        "materializer_contexts": materializer_contexts,
        "materializer_work_queue": materializer_work_queue,
        "rows": bridge_rows,
        "allowed_use": (
            "queue_owned_operation_portfolio_materializer_compiler_bridge_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        bridge,
        context="operation_materializer_bridge",
    )
    return bridge


def _autonomous_chain_target_class(target_kind: str) -> str:
    if target_kind.startswith("packet_member_"):
        return "packet_member"
    if target_kind.startswith("archive_section_"):
        return "archive_section"
    if target_kind.startswith("tensor_"):
        return "tensor"
    if target_kind.startswith("byte_range_"):
        return "byte_range"
    if target_kind.startswith("renderer_payload_"):
        return "receiver_runtime_payload"
    if target_kind.startswith("inverse_steganalysis_"):
        return "inverse_scorer"
    if target_kind.endswith("_component_response"):
        return "component_response"
    return target_kind or "unknown_target"


def _autonomous_chain_targets_from_bridge(
    operation_materializer_bridge: Mapping[str, Any],
) -> list[str]:
    targets: list[str] = []
    targets.extend(_string_list(operation_materializer_bridge.get("top_materializer_targets")))
    targets.extend(
        _string_list(operation_materializer_bridge.get("selected_materializer_targets"))
    )
    for row in operation_materializer_bridge.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        targets.extend(_string_list(row.get("target_kind")))
        targets.extend(_string_list(row.get("chain_targets")))
    return _unique_strings(target for target in targets if target)


def _autonomous_chain_targets_from_handoff(
    targeted_component_correction_chain_materializer_handoff: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    registered = _string_list(
        targeted_component_correction_chain_materializer_handoff.get(
            "registered_chain_targets"
        )
    )
    unregistered = _string_list(
        targeted_component_correction_chain_materializer_handoff.get(
            "unregistered_chain_targets"
        )
    )
    return _unique_strings(registered), _unique_strings(unregistered)


def _autonomous_chain_levels_from_payloads(
    *,
    operation_portfolio: Mapping[str, Any],
    operation_materializer_bridge: Mapping[str, Any],
    targeted_component_correction_chain_materializer_handoff: Mapping[str, Any],
) -> list[str]:
    levels: list[str] = []
    for row in operation_portfolio.get("rows") or []:
        if isinstance(row, Mapping):
            levels.extend(_string_list(row.get("operation_levels")))
    for row in operation_materializer_bridge.get("rows") or []:
        if isinstance(row, Mapping):
            levels.extend(_string_list(row.get("operation_levels")))
    backlog = targeted_component_correction_chain_materializer_handoff.get(
        "materializer_backlog"
    )
    if isinstance(backlog, Mapping):
        for row in backlog.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            portfolio_row = row.get("frontier_operation_portfolio_row")
            if isinstance(portfolio_row, Mapping):
                levels.extend(_string_list(portfolio_row.get("operation_levels")))
    return _unique_strings(level for level in levels if level in _OPERATION_LEVELS)


def _autonomous_chain_correction_families(
    targeted_component_correction_chain_materializer_handoff: Mapping[str, Any],
) -> list[str]:
    families: list[str] = []
    backlog = targeted_component_correction_chain_materializer_handoff.get(
        "materializer_backlog"
    )
    if not isinstance(backlog, Mapping):
        return families
    for row in backlog.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        params = row.get("operation_params")
        if isinstance(params, Mapping):
            families.extend(_string_list(params.get("accepted_correction_families")))
            context = params.get("receiver_runtime_binding_context")
            if isinstance(context, Mapping):
                families.extend(_string_list(context.get("accepted_correction_families")))
        portfolio_row = row.get("frontier_operation_portfolio_row")
        if isinstance(portfolio_row, Mapping):
            families.extend(_string_list(portfolio_row.get("operation_family")))
    return _unique_strings(families)


def _autonomous_chain_receiver_closure_summary(
    targeted_component_correction_chain_materializer_handoff: Mapping[str, Any],
) -> dict[str, Any]:
    missing_fields: list[str] = []
    provided_fields: list[str] = []
    proof_targets: list[str] = []
    plans = [
        plan
        for plan in targeted_component_correction_chain_materializer_handoff.get(
            "context_closure_plans"
        )
        or []
        if isinstance(plan, Mapping)
    ]
    for plan in plans:
        target_kind = str(plan.get("target_kind") or "")
        missing_fields.extend(_string_list(plan.get("missing_context_fields")))
        provided_fields.extend(_string_list(plan.get("provided_context_fields")))
        if isinstance(plan.get("receiver_proof_request"), Mapping):
            proof_targets.append(target_kind)
    work_queue = targeted_component_correction_chain_materializer_handoff.get(
        "materializer_work_queue"
    )
    work_queue_row_count = (
        work_queue.get("row_count") if isinstance(work_queue, Mapping) else 0
    )
    executable_work_queue_row_count = (
        work_queue.get("executable_row_count") if isinstance(work_queue, Mapping) else 0
    )
    blocked_work_queue_row_count = (
        work_queue.get("blocked_row_count") if isinstance(work_queue, Mapping) else 0
    )
    return {
        "schema": "frontier_rate_attack_autonomous_receiver_closure_summary.v1",
        "context_closure_plan_count": len(plans),
        "targeted_chain_work_queue_row_count": work_queue_row_count,
        "targeted_chain_executable_work_queue_row_count": executable_work_queue_row_count,
        "targeted_chain_blocked_work_queue_row_count": blocked_work_queue_row_count,
        "missing_context_field_count": len(_unique_strings(missing_fields)),
        "missing_context_fields": _unique_strings(missing_fields),
        "provided_context_fields": _unique_strings(provided_fields),
        "receiver_proof_request_targets": _unique_strings(proof_targets),
        "parser_only_proofs_rejected": True,
        "single_composed_runtime_consumption_proof_required": bool(plans),
        "component_replay_required_before_budget_spend": bool(plans),
        "exact_auth_eval_required_before_score_claim": True,
        **FALSE_AUTHORITY,
    }


def _autonomous_chain_bridge_summary(
    operation_materializer_bridge: Mapping[str, Any],
) -> dict[str, Any]:
    work_queue = operation_materializer_bridge.get("materializer_work_queue")
    contexts = operation_materializer_bridge.get("materializer_contexts")
    return {
        "schema": "frontier_rate_attack_autonomous_bridge_summary.v1",
        "bridge_row_count": operation_materializer_bridge.get("bridge_row_count", 0),
        "materializer_backlog_row_count": operation_materializer_bridge.get(
            "materializer_backlog_row_count",
            0,
        ),
        "context_row_count": (
            contexts.get("row_count") if isinstance(contexts, Mapping) else 0
        ),
        "blocked_context_count": (
            contexts.get("blocked_context_count")
            if isinstance(contexts, Mapping)
            else 0
        ),
        "work_queue_row_count": (
            work_queue.get("row_count") if isinstance(work_queue, Mapping) else 0
        ),
        "executable_work_row_count": (
            work_queue.get("executable_row_count")
            if isinstance(work_queue, Mapping)
            else 0
        ),
        "blocked_work_row_count": (
            work_queue.get("blocked_row_count") if isinstance(work_queue, Mapping) else 0
        ),
        **FALSE_AUTHORITY,
    }


def _autonomous_chain_scheduler_actions(
    *,
    target_classes: Sequence[str],
    registered_chain_targets: Sequence[str],
    unregistered_targets: Sequence[str],
    receiver_closure: Mapping[str, Any],
    bridge_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if bridge_summary.get("work_queue_row_count"):
        executable_bridge_rows = int(bridge_summary.get("executable_work_row_count") or 0)
        bridge_action: dict[str, Any] = {
            "id": "run_operation_materializer_context_closure_queue",
            "purpose": "bind portfolio materializer contexts across packet_archive_tensor_targets",
            "bounded_local_execution": True,
            "requires_exact_auth_before_score_claim": True,
            **FALSE_AUTHORITY,
        }
        if executable_bridge_rows > 0:
            bridge_action.update(
                {
                    "queue_artifact_key": "operation_materializer_execution_queue",
                    "advisory_only": False,
                }
            )
        else:
            bridge_action.update(
                {
                    "source_artifact_key": "operation_materializer_work_queue",
                    "advisory_only": True,
                    "advisory_reason": (
                        "portfolio materializer work queue has no executable rows "
                        "until context closure artifacts are supplied"
                    ),
                }
            )
        actions.append(bridge_action)
    if registered_chain_targets:
        executable_targeted_rows = int(
            receiver_closure.get("targeted_chain_executable_work_queue_row_count") or 0
        )
        targeted_action: dict[str, Any] = {
            "id": "bind_targeted_chain_materializer_contexts",
            "purpose": "close_receiver_consumed_context_for_many_registered_targets",
            "target_count": len(registered_chain_targets),
            "bounded_local_execution": True,
            "requires_exact_auth_before_score_claim": True,
            **FALSE_AUTHORITY,
        }
        if executable_targeted_rows > 0:
            targeted_action.update(
                {
                    "queue_artifact_key": (
                        "targeted_component_correction_chain_materializer_execution_queue"
                    ),
                    "advisory_only": False,
                }
            )
        else:
            targeted_action.update(
                {
                    "source_artifact_key": (
                        "targeted_component_correction_chain_materializer_work_queue"
                    ),
                    "advisory_only": True,
                    "advisory_reason": (
                        "registered chain materializer rows remain blocked until "
                        "receiver repair fills required runtime-consumption contexts"
                    ),
                }
            )
        actions.append(targeted_action)
    if receiver_closure.get("missing_context_field_count"):
        actions.append(
            {
                "id": "fill_receiver_runtime_proof_requests",
                "queue_artifact_key": "receiver_repair_queue",
                "purpose": "turn parser_or_planner_signal_into_runtime_consumed_proof",
                "missing_context_field_count": (
                    receiver_closure.get("missing_context_field_count")
                ),
                "bounded_local_execution": True,
                "advisory_only": False,
                "requires_exact_auth_before_score_claim": True,
                **FALSE_AUTHORITY,
            }
        )
    if "component_response" in target_classes or any(
        target.endswith("_component_response") for target in unregistered_targets
    ):
        actions.append(
            {
                "id": "run_component_guarded_targeted_drop_many_queue",
                "queue_artifact_key": (
                    "targeted_component_correction_operation_chain_queue"
                ),
                "purpose": "search_pair_frame_batch_drop_many_under_segnet_posenet_guard",
                "bounded_local_execution": True,
                "advisory_only": False,
                "requires_exact_auth_before_score_claim": True,
                **FALSE_AUTHORITY,
            }
        )
    actions.append(
        {
            "id": "fit_segnet_posenet_repair_waterfill_policy",
            "queue_artifact_key": "repair_budget_waterfill_queue",
            "purpose": (
                "estimate_where_drop_many_distortion_debt_should_be_repaired_"
                "from_receiver_closed_rate_budget"
            ),
            "bounded_local_execution": True,
            "advisory_only": False,
            "requires_exact_auth_before_score_claim": True,
            **FALSE_AUTHORITY,
        }
    )
    actions.append(
        {
            "id": "replay_component_response_and_exact_readiness_bridge",
            "source_artifact_key": "materializer_chain_exact_readiness_bridge",
            "purpose": "convert_local_many_op_candidate_into_exact_axis_request_only",
            "bounded_local_execution": False,
            "advisory_only": True,
            "advisory_reason": (
                "bridge intent only until a concrete many-op candidate archive "
                "and exact-readiness artifact exist"
            ),
            "requires_exact_auth_before_score_claim": True,
            **FALSE_AUTHORITY,
        }
    )
    return actions


def _autonomous_chain_repair_waterfill_plan(
    *,
    operation_levels: Sequence[str],
    target_kinds: Sequence[str],
    correction_families: Sequence[str],
    receiver_closure: Mapping[str, Any],
    bridge_summary: Mapping[str, Any],
    rate_budget_preservation_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preservation = (
        rate_budget_preservation_plan
        if isinstance(rate_budget_preservation_plan, Mapping)
        else {}
    )
    operator_action_ledger = (
        preservation.get("operator_action_ledger")
        if isinstance(preservation.get("operator_action_ledger"), Mapping)
        else {}
    )
    repair_dimensions = [
        level
        for level in _OPERATION_LEVELS
        if level
        in {
            "pixel",
            "region",
            "boundary",
            "frame",
            "pair",
            "batch",
            "full_video",
            "tensor_channel",
            "scorer_axis",
        }
        and level in set(operation_levels)
    ]
    rate_credit_targets = [
        target
        for target in target_kinds
        if _autonomous_chain_target_class(target)
        in {
            "packet_member",
            "archive_section",
            "byte_range",
            "tensor",
            "receiver_runtime_payload",
        }
    ]
    distortion_debt_sources = _unique_strings(
        [
            family
            for family in correction_families
            if "drop" in family
            or "geometry" in family
            or "inverse_scorer" in family
        ]
    )
    if not distortion_debt_sources and {
        "frame",
        "pair",
        "batch",
    }.intersection(operation_levels):
        distortion_debt_sources.append("drop_many_or_pair_frame_batch_candidate")
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_plan.v1",
        "allocator": (
            "measured_component_marginal_waterfill_over_segnet_posenet_rate_budget"
        ),
        "component_axes": ["segnet", "posenet"],
        "repair_dimensions": repair_dimensions,
        "rate_credit_targets": _unique_strings(rate_credit_targets),
        "distortion_debt_sources": distortion_debt_sources,
        "allocation_variables": [
            "bytes_freed_by_receiver_closed_materializers",
            "segnet_delta_from_drop_many_or_geometry_edit",
            "posenet_delta_from_drop_many_or_geometry_edit",
            "marginal_repair_score_per_byte_by_dimension",
            "interaction_synergy_or_antagonism_by_chain_stage",
        ],
        "rate_only_preservation": {
            "schema": "frontier_rate_attack_rate_only_preservation_reference.v1",
            "plan_schema": preservation.get("schema"),
            "active": preservation.get("active") is True,
            "rate_only_candidate_count": preservation.get("rate_only_candidate_count"),
            "rate_only_saved_bytes_total": preservation.get(
                "rate_only_saved_bytes_total"
            ),
            "rate_positive_distortion_regression_count": preservation.get(
                "rate_positive_distortion_regression_count"
            ),
            "preserve_before_budget_spend": True,
            "spent_budget_candidate_is_child_of_rate_only_floor": True,
            **FALSE_AUTHORITY,
        },
        "cumulative_rate_attack": dict(
            preservation.get("cumulative_rate_attack")
            if isinstance(preservation.get("cumulative_rate_attack"), Mapping)
            else {}
        ),
        "operator_action_ledger": dict(operator_action_ledger),
        "waterfill_policy": {
            "schema": "frontier_rate_attack_repair_budget_waterfill_policy.v1",
            "objective": (
                "allocate_receiver_closed_rate_credit_to_measured_segnet_posenet_"
                "repair_terms_without_crossing_parent_rate_only_floor"
            ),
            "rate_operator_ledger_schema": operator_action_ledger.get("schema"),
            "rate_operator_action_term_count": operator_action_ledger.get("term_count"),
            "repair_allocation_action_term_schema": (
                REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
            ),
            "hard_constraints": [
                "rate_only_parent_archive_must_materialize_first",
                "spent_budget_child_archive_cannot_replace_parent_without_exact_eval",
                "receiver_decode_only_adapter_must_consume_final_packet",
            ],
            "budget_spend_allowed": False,
            **FALSE_AUTHORITY,
        },
        "required_measurements": [
            "component_response_replay_for_drop_many_candidate",
            "segnet_posenet_repair_probe_by_region_boundary_frame_pair_batch",
            "receiver_consumed_candidate_archive_byte_delta",
            "exact_axis_component_response_before_budget_spend",
        ],
        "priority_hint": (
            "prefer_chains_that_bank_rate_credit_then_spend_only_where_"
            "component_marginal_repair_score_per_byte_is_highest"
        ),
        "bridge_executable_work_row_count": bridge_summary.get(
            "executable_work_row_count"
        ),
        "missing_receiver_context_field_count": receiver_closure.get(
            "missing_context_field_count"
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "exact_auth_eval_required_before_score_claim": True,
        "allowed_use": "repair_budget_allocator_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _autonomous_chain_build_row(
    *,
    chain_id: str,
    chain_family: str,
    objective: str,
    operation_levels: Sequence[str],
    target_kinds: Sequence[str],
    registered_chain_targets: Sequence[str],
    unregistered_targets: Sequence[str],
    correction_families: Sequence[str],
    receiver_closure: Mapping[str, Any],
    bridge_summary: Mapping[str, Any],
    rate_budget_preservation_plan: Mapping[str, Any] | None,
    priority_base: float,
    source_artifact_keys: Sequence[str],
    next_queue_edges: Sequence[str],
) -> dict[str, Any]:
    target_classes = _unique_strings(
        _autonomous_chain_target_class(target) for target in target_kinds
    )
    blockers = [
        "exact_auth_eval_required_before_score_or_promotion_claim",
        "component_replay_required_before_budget_spend",
    ]
    blockers.extend(
        f"missing_receiver_context:{field}"
        for field in _string_list(receiver_closure.get("missing_context_fields"))
    )
    if int(receiver_closure.get("targeted_chain_blocked_work_queue_row_count") or 0) > 0:
        blockers.append("targeted_chain_materializer_work_queue_has_blocked_rows")
    if (
        int(receiver_closure.get("targeted_chain_work_queue_row_count") or 0) > 0
        and int(receiver_closure.get("targeted_chain_executable_work_queue_row_count") or 0)
        == 0
    ):
        blockers.append("targeted_chain_materializer_execution_queue_not_yet_buildable")
    blockers.extend(
        f"unregistered_chain_target:{target}" for target in unregistered_targets
    )
    if not target_kinds:
        blockers.append("no_materializer_targets_bound")
    if not correction_families and {
        "pixel",
        "region",
        "boundary",
        "frame",
        "pair",
        "batch",
    }.intersection(operation_levels):
        blockers.append("component_correction_family_not_yet_bound")
    scheduler_actions = _autonomous_chain_scheduler_actions(
        target_classes=target_classes,
        registered_chain_targets=registered_chain_targets,
        unregistered_targets=unregistered_targets,
        receiver_closure=receiver_closure,
        bridge_summary=bridge_summary,
    )
    repair_waterfill_plan = _autonomous_chain_repair_waterfill_plan(
        operation_levels=operation_levels,
        target_kinds=target_kinds,
        correction_families=correction_families,
        receiver_closure=receiver_closure,
        bridge_summary=bridge_summary,
        rate_budget_preservation_plan=rate_budget_preservation_plan,
    )
    coverage_score = (
        float(len(_unique_strings(operation_levels))) * 4.0
        + float(len(target_classes)) * 8.0
        + float(len(_unique_strings(target_kinds))) * 2.0
        + float(len(_unique_strings(correction_families))) * 3.0
    )
    return {
        "schema": AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA,
        "chain_id": chain_id,
        "chain_family": chain_family,
        "optimization_objective": objective,
        "priority_score": priority_base + coverage_score,
        "operation_levels": _unique_strings(operation_levels),
        "covered_operation_level_count": len(_unique_strings(operation_levels)),
        "target_kinds": _unique_strings(target_kinds),
        "target_classes": target_classes,
        "target_class_count": len(target_classes),
        "materializer_target_count": len(_unique_strings(target_kinds)),
        "registered_chain_targets": _unique_strings(registered_chain_targets),
        "registered_chain_target_count": len(_unique_strings(registered_chain_targets)),
        "unregistered_targets": _unique_strings(unregistered_targets),
        "correction_families": _unique_strings(correction_families),
        "correction_family_count": len(_unique_strings(correction_families)),
        "receiver_closure": dict(receiver_closure),
        "bridge_summary": dict(bridge_summary),
        "rate_budget_preservation_plan": dict(rate_budget_preservation_plan or {}),
        "repair_budget_waterfill_plan": repair_waterfill_plan,
        "scheduler_actions": scheduler_actions,
        "scheduler_action_count": len(scheduler_actions),
        "source_artifact_keys": _unique_strings(source_artifact_keys),
        "next_queue_edges": _unique_strings(next_queue_edges),
        "local_queue_execution_plan_present": any(
            not bool(action.get("advisory_only")) for action in scheduler_actions
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_materializer_execution": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": _unique_strings(blockers),
        "allowed_use": "autonomous_many_operator_queue_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def build_frontier_autonomous_chain_optimization(
    *,
    operation_portfolio: Mapping[str, Any],
    operation_materializer_bridge: Mapping[str, Any] | None = None,
    targeted_component_correction_chain_materializer_handoff: Mapping[str, Any] | None = None,
    chain_limit: int = 4,
) -> dict[str, Any]:
    """Lift portfolio/bridge/handoff signal into broad queue-owned campaigns."""

    if chain_limit < 1:
        raise FrontierRateAttackFeedbackError("chain_limit must be >= 1")
    require_no_truthy_authority_fields(
        operation_portfolio,
        context="autonomous_chain_optimization_operation_portfolio",
    )
    bridge = dict(operation_materializer_bridge or {})
    handoff = dict(targeted_component_correction_chain_materializer_handoff or {})
    if bridge:
        require_no_truthy_authority_fields(
            bridge,
            context="autonomous_chain_optimization_operation_materializer_bridge",
        )
    if handoff:
        require_no_truthy_authority_fields(
            handoff,
            context="autonomous_chain_optimization_targeted_chain_handoff",
        )
    bridge_targets = _autonomous_chain_targets_from_bridge(bridge)
    registered_targets, unregistered_targets = _autonomous_chain_targets_from_handoff(
        handoff
    )
    target_kinds = _unique_strings([*bridge_targets, *registered_targets])
    all_unregistered_targets = _unique_strings(unregistered_targets)
    operation_levels = _autonomous_chain_levels_from_payloads(
        operation_portfolio=operation_portfolio,
        operation_materializer_bridge=bridge,
        targeted_component_correction_chain_materializer_handoff=handoff,
    )
    correction_families = _autonomous_chain_correction_families(handoff)
    receiver_closure = _autonomous_chain_receiver_closure_summary(handoff)
    bridge_summary = _autonomous_chain_bridge_summary(bridge)
    rate_budget_preservation_plan = (
        operation_portfolio.get("rate_budget_preservation_plan")
        if isinstance(operation_portfolio.get("rate_budget_preservation_plan"), Mapping)
        else {}
    )
    source_artifact_keys = [
        "operation_portfolio",
        "operation_materializer_bridge",
        "rate_budget_preservation_plan",
    ]
    if handoff:
        source_artifact_keys.append(
            "targeted_component_correction_chain_materializer_handoff"
        )
    edges = [
        "operation_portfolio_to_materializer_backlog_context_work_queue",
        "targeted_component_operation_chain_to_materializer_handoff",
        "targeted_operation_chain_queue_to_targeted_drop_many_child_queue",
        "autonomous_chain_optimization_to_queue_owned_many_op_plan",
        "many_op_plan_to_component_replay_and_exact_readiness_bridge",
    ]
    rows = [
        _autonomous_chain_build_row(
            chain_id="global_many_op_rate_distortion_receiver_campaign",
            chain_family="rate_distortion_receiver_closed_many_op_campaign",
            objective=(
                "minimize_rate_plus_segnet_plus_posenet_by_composing_packet_archive_"
                "tensor_frame_pair_batch_and_receiver_operations"
            ),
            operation_levels=operation_levels,
            target_kinds=target_kinds,
            registered_chain_targets=registered_targets,
            unregistered_targets=all_unregistered_targets,
            correction_families=correction_families,
            receiver_closure=receiver_closure,
            bridge_summary=bridge_summary,
            rate_budget_preservation_plan=rate_budget_preservation_plan,
            priority_base=120.0,
            source_artifact_keys=source_artifact_keys,
            next_queue_edges=edges,
        )
    ]
    if registered_targets or correction_families:
        rows.append(
            _autonomous_chain_build_row(
                chain_id="receiver_closed_budget_reinvestment_campaign",
                chain_family="rate_win_to_targeted_correction_budget_campaign",
                objective=(
                    "turn_receiver_closed_byte_savings_into_component_guarded_"
                    "segnet_posenet_repair_budget"
                ),
                operation_levels=operation_levels,
                target_kinds=registered_targets,
                registered_chain_targets=registered_targets,
                unregistered_targets=all_unregistered_targets,
                correction_families=correction_families,
                receiver_closure=receiver_closure,
                bridge_summary=bridge_summary,
                rate_budget_preservation_plan=rate_budget_preservation_plan,
                priority_base=96.0,
                source_artifact_keys=source_artifact_keys,
                next_queue_edges=edges,
            )
        )
    if bridge_targets:
        rows.append(
            _autonomous_chain_build_row(
                chain_id="portfolio_materializer_context_closure_campaign",
                chain_family="portfolio_to_materializer_context_closure_campaign",
                objective=(
                    "close_many_registered_materializer_contexts_before_leaf_execution"
                ),
                operation_levels=operation_levels,
                target_kinds=bridge_targets,
                registered_chain_targets=(),
                unregistered_targets=(),
                correction_families=correction_families,
                receiver_closure=receiver_closure,
                bridge_summary=bridge_summary,
                rate_budget_preservation_plan=rate_budget_preservation_plan,
                priority_base=84.0,
                source_artifact_keys=source_artifact_keys,
                next_queue_edges=edges,
            )
        )
    scorer_levels = {
        "pixel",
        "region",
        "boundary",
        "frame",
        "pair",
        "batch",
        "full_video",
        "scorer_axis",
    }
    if scorer_levels.intersection(operation_levels):
        rows.append(
            _autonomous_chain_build_row(
                chain_id="segnet_posenet_geometry_drop_many_campaign",
                chain_family="component_response_geometry_pair_batch_campaign",
                objective=(
                    "search_drop_many_and_geometry_edits_at_pair_frame_batch_scale_"
                    "under_segnet_posenet_component_marginals"
                ),
                operation_levels=[
                    level for level in operation_levels if level in scorer_levels
                ],
                target_kinds=[
                    target
                    for target in target_kinds
                    if _autonomous_chain_target_class(target)
                    in {"component_response", "inverse_scorer", "tensor"}
                ],
                registered_chain_targets=[
                    target
                    for target in registered_targets
                    if _autonomous_chain_target_class(target)
                    in {"component_response", "inverse_scorer", "tensor"}
                ],
                unregistered_targets=[
                    target
                    for target in all_unregistered_targets
                    if target.endswith("_component_response")
                ],
                correction_families=correction_families,
                receiver_closure=receiver_closure,
                bridge_summary=bridge_summary,
                rate_budget_preservation_plan=rate_budget_preservation_plan,
                priority_base=80.0,
                source_artifact_keys=source_artifact_keys,
                next_queue_edges=edges,
            )
        )
    rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get("priority_score") or 0.0),
            str(row.get("chain_id") or ""),
        ),
    )[:chain_limit]
    target_class_counts: dict[str, int] = {}
    for row in rows:
        for target_class in _string_list(row.get("target_classes")):
            target_class_counts[target_class] = target_class_counts.get(target_class, 0) + 1
    level_counts: dict[str, int] = {}
    for row in rows:
        for level in _string_list(row.get("operation_levels")):
            level_counts[level] = level_counts.get(level, 0) + 1
    payload = {
        "schema": AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_operation_portfolio_schema": operation_portfolio.get("schema"),
        "source_operation_materializer_bridge_schema": bridge.get("schema"),
        "source_targeted_chain_materializer_handoff_schema": handoff.get("schema"),
        "chain_count": len(rows),
        "top_chain_ids": [str(row.get("chain_id") or "") for row in rows[:8]],
        "target_kinds": target_kinds,
        "target_classes": sorted(target_class_counts),
        "target_class_counts": dict(sorted(target_class_counts.items())),
        "operation_level_counts": dict(sorted(level_counts.items())),
        "registered_target_count": len(_unique_strings(registered_targets)),
        "unregistered_target_count": len(all_unregistered_targets),
        "rate_only_candidate_count": rate_budget_preservation_plan.get(
            "rate_only_candidate_count"
        ),
        "rate_only_saved_bytes_total": rate_budget_preservation_plan.get(
            "rate_only_saved_bytes_total"
        ),
        "rate_positive_distortion_regression_count": (
            rate_budget_preservation_plan.get(
                "rate_positive_distortion_regression_count"
            )
        ),
        "correction_families": correction_families,
        "receiver_closure_summary": receiver_closure,
        "bridge_summary": bridge_summary,
        "rate_budget_preservation_plan": dict(rate_budget_preservation_plan),
        "rows": rows,
        "allowed_use": "autonomous_many_operator_queue_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="frontier_autonomous_chain_optimization",
    )
    return payload


def _autonomous_chain_optimization_queue_metadata(
    autonomous_chain_optimization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "frontier_rate_attack_autonomous_chain_optimization_queue_metadata.v1",
        "chain_count": autonomous_chain_optimization.get("chain_count"),
        "top_chain_ids": list(
            autonomous_chain_optimization.get("top_chain_ids") or []
        ),
        "target_classes": list(
            autonomous_chain_optimization.get("target_classes") or []
        ),
        "registered_target_count": autonomous_chain_optimization.get(
            "registered_target_count"
        ),
        "unregistered_target_count": autonomous_chain_optimization.get(
            "unregistered_target_count"
        ),
        "rate_only_candidate_count": autonomous_chain_optimization.get(
            "rate_only_candidate_count"
        ),
        "rate_only_saved_bytes_total": autonomous_chain_optimization.get(
            "rate_only_saved_bytes_total"
        ),
        "rate_positive_distortion_regression_count": (
            autonomous_chain_optimization.get(
                "rate_positive_distortion_regression_count"
            )
        ),
        "allowed_use": "queue_metadata_for_many_op_autonomous_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def attach_frontier_autonomous_chain_optimization(
    report: dict[str, Any],
    *,
    targeted_component_correction_chain_materializer_handoff: Mapping[str, Any]
    | None = None,
    update_queue_metadata: bool = True,
) -> dict[str, Any] | None:
    """Attach final autonomous-chain payload and keep queue metadata in sync."""

    operation_portfolio = report.get("operation_portfolio")
    operation_materializer_bridge = report.get("operation_materializer_bridge")
    if not isinstance(operation_portfolio, Mapping) or not isinstance(
        operation_materializer_bridge,
        Mapping,
    ):
        return None
    handoff: Mapping[str, Any] | None
    if targeted_component_correction_chain_materializer_handoff is not None:
        handoff = targeted_component_correction_chain_materializer_handoff
    else:
        maybe_handoff = report.get(
            "targeted_component_correction_chain_materializer_handoff"
        )
        handoff = maybe_handoff if isinstance(maybe_handoff, Mapping) else None
    autonomous_chain_optimization = build_frontier_autonomous_chain_optimization(
        operation_portfolio=operation_portfolio,
        operation_materializer_bridge=operation_materializer_bridge,
        targeted_component_correction_chain_materializer_handoff=handoff,
        chain_limit=int(report.get("candidate_limit") or 4),
    )
    report["autonomous_chain_optimization"] = autonomous_chain_optimization
    if update_queue_metadata:
        queue = report.get("queue")
        if isinstance(queue, dict):
            metadata_payload = _autonomous_chain_optimization_queue_metadata(
                autonomous_chain_optimization
            )
            for experiment in queue.get("experiments") or []:
                if not isinstance(experiment, dict):
                    continue
                metadata = experiment.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["frontier_autonomous_chain_optimization"] = (
                        metadata_payload
                    )
    return autonomous_chain_optimization


def build_frontier_materializer_execution_queue_if_available(
    *,
    repo_root: str | Path,
    materializer_work_queue: Mapping[str, Any],
    materializer_work_queue_path: str | Path,
    queue_id: str,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    candidate_limit: int = 4,
) -> dict[str, Any] | None:
    """Build a local execution queue only when materializer rows are runnable."""

    require_no_truthy_authority_fields(
        materializer_work_queue,
        context=f"{queue_id}_materializer_work_queue",
    )
    executable_count = int(materializer_work_queue.get("executable_row_count") or 0)
    if executable_count <= 0:
        return None
    limit = max(1, min(candidate_limit, executable_count))
    queue = build_materializer_execution_queue(
        materializer_work_queue,
        queue_id=queue_id,
        repo_root=repo_root,
        source_work_queue_path=materializer_work_queue_path,
        scheduler_results_root=str(results_root),
        step_timeout_seconds=900,
        limit=limit,
        include_exact_readiness_followup=True,
        require_renderer_payload_dfl1_parity_followup=True,
    )
    queue.update(
        {
            "allowed_use": "bounded_local_materializer_execution_queue_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        }
    )
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, dict):
            continue
        metadata = experiment.get("metadata")
        if not isinstance(metadata, dict):
            continue
        metadata.update(
            {
                "storage_preflight_status": "not_wired_smoke_bounded",
                "storage_preflight_blocker": (
                    "autonomous_child_materializer_execution_queue_omits_"
                    "scheduler_storage_preflight_parameters"
                ),
                "bounded_smoke_candidate_limit": limit,
                "allowed_use": "bounded_local_materializer_smoke_only",
                **FALSE_AUTHORITY,
            }
        )
        require_no_truthy_authority_fields(
            metadata,
            context=f"{queue_id}_materializer_execution_metadata",
        )
    require_no_truthy_authority_fields(
        queue,
        context=f"{queue_id}_materializer_execution_queue",
    )
    return queue


def _autonomous_chain_row_for_id(
    autonomous_chain_optimization: Mapping[str, Any],
    chain_id: str,
) -> dict[str, Any]:
    require_no_truthy_authority_fields(
        autonomous_chain_optimization,
        context="autonomous_chain_work_order_source",
    )
    for row in autonomous_chain_optimization.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("chain_id") or "") == chain_id:
            require_no_truthy_authority_fields(
                row,
                context=f"autonomous_chain_work_order_row:{chain_id}",
            )
            return dict(row)
    raise FrontierRateAttackFeedbackError(f"unknown autonomous chain id: {chain_id}")


def _autonomous_chain_pipeline_stages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    chain_id = str(row.get("chain_id") or "unknown_chain")
    return [
        {
            "id": "select_many_operator_chain",
            "pipeline_side": "encoder_planner",
            "owner": "queue_owned_autonomous_optimizer",
            "purpose": "choose packet/archive/tensor/frame/pair/batch operators jointly",
            "source_chain_id": chain_id,
            "writes_submission_runtime": False,
            **FALSE_AUTHORITY,
        },
        {
            "id": "apply_rate_attack_materializers",
            "pipeline_side": "encoder_materializer",
            "owner": "local_child_queues",
            "purpose": "execute receiver-closed byte-saving transforms before packaging",
            "source_chain_id": chain_id,
            "writes_submission_runtime": True,
            **FALSE_AUTHORITY,
        },
        {
            "id": "preserve_rate_only_floor_archive",
            "pipeline_side": "encoder_archive_builder",
            "owner": "rate_budget_preservation_plan",
            "purpose": (
                "emit immutable rate-only checkpoint before any distortion-side "
                "budget reinvestment"
            ),
            "source_chain_id": chain_id,
            "writes_submission_runtime": True,
            "budget_spend_allowed": False,
            **FALSE_AUTHORITY,
        },
        {
            "id": "allocate_repair_budget",
            "pipeline_side": "encoder_repair_allocator",
            "owner": "component_marginal_waterfill_policy",
            "purpose": "spend freed bytes only where SegNet/PoseNet marginal repair helps",
            "source_chain_id": chain_id,
            "writes_submission_runtime": True,
            "budget_spend_allowed": False,
            **FALSE_AUTHORITY,
        },
        {
            "id": "emit_byte_closed_archive",
            "pipeline_side": "encoder_archive_builder",
            "owner": "archive_packaging_queue",
            "purpose": "package transformed payload and deterministic receiver runtime",
            "source_chain_id": chain_id,
            "writes_submission_runtime": True,
            **FALSE_AUTHORITY,
        },
        {
            "id": "consume_transformed_packet",
            "pipeline_side": "receiver_decode_only",
            "owner": "inflate_runtime_adapter",
            "purpose": "deterministically decode encoder-authored transforms without scorer access",
            "source_chain_id": chain_id,
            "writes_submission_runtime": False,
            **FALSE_AUTHORITY,
        },
        {
            "id": "measure_exact_axis_and_feedback",
            "pipeline_side": "exact_eval_feedback",
            "owner": "contest_axis_eval_harvest",
            "purpose": "measure archive/runtime on contest CPU/CUDA axis before any score claim",
            "source_chain_id": chain_id,
            "writes_submission_runtime": False,
            **FALSE_AUTHORITY,
        },
    ]


def build_frontier_autonomous_chain_work_order(
    *,
    autonomous_chain_optimization: Mapping[str, Any],
    chain_id: str,
    child_queue_artifact_paths: Sequence[str | Path] = (),
    missing_queue_artifact_keys: Sequence[str] = (),
    queue_actuation_ready: bool | None = None,
    post_repair_refresh_planned: bool = False,
) -> dict[str, Any]:
    """Turn a selected many-op chain into a typed encoder/receiver work order."""

    row = _autonomous_chain_row_for_id(autonomous_chain_optimization, chain_id)
    scheduler_actions = [
        dict(action)
        for action in row.get("scheduler_actions") or []
        if isinstance(action, Mapping)
    ]
    local_queue_actions = [
        action for action in scheduler_actions if not bool(action.get("advisory_only"))
    ]
    advisory_actions = [
        action for action in scheduler_actions if bool(action.get("advisory_only"))
    ]
    pipeline_stages = _autonomous_chain_pipeline_stages(row)
    child_queue_paths = _unique_strings(
        [str(path) for path in child_queue_artifact_paths]
    )
    missing_queue_keys = _unique_strings(missing_queue_artifact_keys)
    if queue_actuation_ready is None:
        queue_actuation_ready = bool(local_queue_actions) and not missing_queue_keys
    payload = {
        "schema": AUTONOMOUS_CHAIN_WORK_ORDER_SCHEMA,
        "generated_at_utc": autonomous_chain_optimization.get("generated_at_utc"),
        "source_autonomous_chain_optimization_schema": (
            autonomous_chain_optimization.get("schema")
        ),
        "chain_id": chain_id,
        "chain_family": row.get("chain_family"),
        "optimization_objective": row.get("optimization_objective"),
        "pipeline_placement": {
            "rate_attack_owner": "encoder_materializer_and_archive_builder",
            "repair_owner": "encoder_repair_allocator_before_archive_packaging",
            "receiver_owner": "deterministic_inflate_runtime_adapter_only",
            "exact_eval_owner": "contest_axis_measurement_and_feedback",
            "answer_to_pipeline_question": (
                "repair and final rate attack are optimized/applied in the "
                "encoder-side queue before archive emission; receiver code only "
                "consumes the transformed representation deterministically"
            ),
            **FALSE_AUTHORITY,
        },
        "pipeline_stages": pipeline_stages,
        "local_queue_actions": local_queue_actions,
        "local_queue_action_count": len(local_queue_actions),
        "advisory_actions": advisory_actions,
        "advisory_action_count": len(advisory_actions),
        "child_queue_artifact_paths": child_queue_paths,
        "child_queue_artifact_count": len(child_queue_paths),
        "missing_queue_artifact_keys": missing_queue_keys,
        "queue_actuation_ready": bool(queue_actuation_ready),
        "post_repair_refresh_planned": bool(post_repair_refresh_planned),
        "repair_budget_waterfill_plan": dict(
            row.get("repair_budget_waterfill_plan")
            if isinstance(row.get("repair_budget_waterfill_plan"), Mapping)
            else {}
        ),
        "receiver_closure": dict(
            row.get("receiver_closure")
            if isinstance(row.get("receiver_closure"), Mapping)
            else {}
        ),
        "target_kinds": _string_list(row.get("target_kinds")),
        "operation_levels": _string_list(row.get("operation_levels")),
        "blockers": _unique_strings(
            [
                *(_string_list(row.get("blockers"))),
                "exact_auth_eval_required_before_score_or_promotion_claim",
            ]
        ),
        "ready_for_materializer_execution": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "encoder_side_many_op_work_order_for_local_queue_actuation",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context=f"frontier_autonomous_chain_work_order:{chain_id}",
    )
    return payload


def _repair_response_entropy_position(row: Mapping[str, Any]) -> tuple[str, str]:
    explicit = str(row.get("entropy_position") or "").strip()
    if explicit:
        return explicit, "explicit_response_entropy_position"

    target_kind_candidates = _unique_strings(
        [
            row.get("target_kind"),
            row.get("rate_target_kind"),
            row.get("rate_packet_target_kind"),
            row.get("materializer_target_kind"),
        ]
    )
    for target_kind in target_kind_candidates:
        position = _entropy_pipeline_position(target_kind)
        if position != "unknown_entropy_pipeline_position":
            return position, f"target_kind:{target_kind}"

    corpus = " ".join(
        _unique_strings(
            [
                row.get("correction_family"),
                *_targeted_component_response_operation_levels(row),
                *_targeted_component_response_targeted_dimensions(row),
            ]
        )
    ).lower()
    if any(token in corpus for token in ("huffman", "arithmetic", "range", "ans")):
        return "at_entropy_coder", "correction_family_or_level_entropy_codec_hint"
    if any(token in corpus for token in ("zip", "header", "container", "member")):
        return (
            "after_entropy_coder_container_or_zip_grammar",
            "correction_family_or_level_container_hint",
        )
    if any(
        token in corpus
        for token in (
            "segnet",
            "posenet",
            "frame",
            "pair",
            "region",
            "palette",
            "chroma",
            "luma",
            "rgb",
            "roll",
            "repair",
        )
    ):
        return (
            "before_entropy_coder_distribution_shaping",
            "scorer_or_frame_space_repair_hint",
        )
    return "unknown_entropy_pipeline_position", "insufficient_response_stage_signal"


def _repair_response_scope_values(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if value is None:
        metadata = row.get("source_metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(key)
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [str(value)]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _repair_response_interaction_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    pair_indices = _repair_response_scope_values(row, "selected_pair_indices")
    frame_indices = _repair_response_scope_values(row, "selected_frame_indices")
    region_ids = _repair_response_scope_values(row, "selected_region_ids")
    mode_ids = _repair_response_scope_values(row, "selected_mode_ids")
    if not mode_ids:
        mode_ids = _repair_response_scope_values(row, "compact_palette_mode_ids")
    return {
        "schema": "frontier_rate_attack_repair_response_interaction_scope.v1",
        "pair_indices": pair_indices,
        "pair_count": len(pair_indices) if pair_indices else row.get("selected_pair_count"),
        "frame_indices": frame_indices,
        "frame_count": len(frame_indices) if frame_indices else row.get("selected_frame_count"),
        "region_ids": region_ids,
        "region_count": len(region_ids) if region_ids else row.get("selected_region_count"),
        "mode_ids": mode_ids,
        "mode_count": len(mode_ids) if mode_ids else row.get("palette_size"),
        "operation_levels": _targeted_component_response_operation_levels(row),
        "targeted_dimensions": _targeted_component_response_targeted_dimensions(row),
        "interaction_status": "must_remeasure_stack_synergy_before_promotion",
        "allowed_use": "repair_stackability_feature_for_local_waterfill_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _repair_response_marginal_curve(
    *,
    requested_bytes: int,
    segnet_delta: float | None,
    posenet_delta: float | None,
    component_delta: float | None,
    rate_delta: float | None,
    objective_delta: float | None,
) -> dict[str, Any]:
    denominator = requested_bytes if requested_bytes > 0 else None

    def per_byte(value: float | None) -> float | None:
        return value / denominator if value is not None and denominator else None

    return {
        "schema": "frontier_rate_attack_repair_response_marginal_curves.v1",
        "requested_repair_bytes": requested_bytes,
        "segnet": {
            "delta_score_units": segnet_delta,
            "marginal_delta_per_byte": per_byte(segnet_delta),
        },
        "posenet": {
            "delta_score_units": posenet_delta,
            "marginal_delta_per_byte": per_byte(posenet_delta),
        },
        "component": {
            "delta_score_units": component_delta,
            "marginal_delta_per_byte": per_byte(component_delta),
        },
        "lambda_rate": {
            "delta_score_units": rate_delta,
            "marginal_delta_per_byte": per_byte(rate_delta),
        },
        "objective": {
            "delta_score_units": objective_delta,
            "marginal_delta_per_byte": per_byte(objective_delta),
            "improvement_score_units": (
                -objective_delta
                if objective_delta is not None and objective_delta < 0.0
                else 0.0
            ),
            "improvement_per_byte": (
                (-objective_delta / denominator)
                if objective_delta is not None
                and objective_delta < 0.0
                and denominator
                else 0.0
            ),
        },
        "allowed_use": "typed_marginal_curve_for_local_repair_waterfill",
        "forbidden_use": "score_claim_or_budget_spend_authority",
        **FALSE_AUTHORITY,
    }


def _repair_response_local_advisory_custody(row: Mapping[str, Any]) -> dict[str, Any]:
    """Record local advisory artifacts, or name the exact missing artifact keys."""

    response_path = row.get("response_artifact_path") or row.get("source_harvest_path")
    artifact_rows = [
        ("local_cpu_advisory_path", row.get("local_cpu_advisory_path"), True),
        (
            "reference_local_cpu_advisory_path",
            row.get("reference_local_cpu_advisory_path"),
            False,
        ),
        ("local_mlx_response_path", row.get("local_mlx_response_path"), True),
        (
            "reference_local_mlx_response_path",
            row.get("reference_local_mlx_response_path"),
            False,
        ),
        ("response_artifact_path", response_path, True),
    ]
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for artifact_key, raw_path, required in artifact_rows:
        path_text = str(raw_path or "").strip()
        present = bool(path_text)
        if required and not present:
            missing.append(artifact_key)
        artifacts.append(
            {
                "schema": "frontier_rate_attack_local_advisory_artifact_ref.v1",
                "artifact_key": artifact_key,
                "path": path_text or None,
                "required_for_local_execution": required,
                "declared": present,
                "allowed_use": "local_advisory_custody_pointer_only",
                "forbidden_use": "score_claim_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )

    mlx_axis = str(row.get("local_mlx_score_axis") or "").strip()
    reference_mlx_axis = str(row.get("reference_local_mlx_score_axis") or "").strip()
    mlx_path_present = any(
        item["artifact_key"] == "local_mlx_response_path" and item["declared"]
        for item in artifacts
    )
    reference_mlx_path_present = any(
        item["artifact_key"] == "reference_local_mlx_response_path"
        and item["declared"]
        for item in artifacts
    )
    axis_ok = mlx_axis == "[macOS-MLX research-signal]"
    reference_axis_ok = (
        not reference_mlx_path_present
        or reference_mlx_axis == "[macOS-MLX research-signal]"
    )
    blockers = [f"missing_local_advisory_artifact:{key}" for key in missing]
    if mlx_path_present and not axis_ok:
        blockers.append("local_mlx_response_axis_not_research_signal")
    if reference_mlx_path_present and not reference_axis_ok:
        blockers.append("reference_local_mlx_response_axis_not_research_signal")
    mlx_custody_present = mlx_path_present and axis_ok
    return {
        "schema": "frontier_rate_attack_repair_response_local_advisory_custody.v1",
        "artifact_refs": artifacts,
        "declared_artifact_keys": [
            item["artifact_key"] for item in artifacts if item["declared"]
        ],
        "missing_required_artifact_keys": missing,
        "missing_required_artifact_count": len(missing),
        "missing_artifact_blockers": _unique_strings(blockers),
        "local_mlx_score_axis": mlx_axis or None,
        "reference_local_mlx_score_axis": reference_mlx_axis or None,
        "mlx_advisory_custody_present": mlx_custody_present,
        "paired_mlx_advisory_custody_present": (
            mlx_custody_present and reference_mlx_path_present and reference_axis_ok
        ),
        "local_execution_signal": (
            "mlx_advisory_custody_present"
            if mlx_custody_present
            else "missing_mlx_advisory_custody_exact_artifact_named"
        ),
        "score_axis": "[macOS-MLX research-signal]" if mlx_custody_present else None,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "local_mlx_advisory_custody_for_repair_waterfill_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _repair_waterfill_added_bytes(row: Mapping[str, Any]) -> int | None:
    terms = row.get("local_cpu_component_terms")
    if isinstance(terms, Mapping):
        value = _finite_int_or_none(terms.get("correction_added_archive_bytes"))
        if value is not None:
            return max(0, value)
    paired = row.get("local_cpu_paired_reference_terms")
    if isinstance(paired, Mapping):
        value = _finite_int_or_none(paired.get("correction_added_archive_bytes"))
        if value is not None:
            return max(0, value)
    return None


def _repair_budget_typed_response_ledger(
    *,
    accepted_rows: Sequence[Mapping[str, Any]],
    available_rate_credit_bytes: int,
) -> dict[str, Any]:
    """Normalize measured repair responses into the waterfill action basis."""

    typed_rows: list[dict[str, Any]] = []
    entropy_histogram: dict[str, int] = {}
    lambda_rate_score_per_byte = rate_delta_for_archive_byte_delta(1)
    for rank, row in enumerate(
        sorted(accepted_rows, key=_targeted_component_response_sort_key),
        start=1,
    ):
        normalized = dict(row)
        added_bytes = _repair_waterfill_added_bytes(row)
        requested_bytes = added_bytes if added_bytes is not None else 0
        local_terms = (
            row.get("local_cpu_component_terms")
            if isinstance(row.get("local_cpu_component_terms"), Mapping)
            else {}
        )
        segnet_delta = _finite_float_or_none(
            local_terms.get("segnet_delta_score_units")
        )
        posenet_delta = _finite_float_or_none(
            local_terms.get("posenet_delta_score_units")
        )
        component_delta = _finite_float_or_none(
            row.get("measured_component_delta_score_units")
        )
        if component_delta is None and segnet_delta is not None and posenet_delta is not None:
            component_delta = segnet_delta + posenet_delta
        rate_delta = _finite_float_or_none(
            local_terms.get("correction_rate_delta_score_units")
        )
        if rate_delta is None:
            rate_delta = rate_delta_for_archive_byte_delta(requested_bytes)
        objective_delta = _finite_float_or_none(
            row.get("measured_lagrangian_delta_score_units")
        )
        if objective_delta is None and component_delta is not None:
            objective_delta = component_delta + rate_delta
        entropy_position, entropy_evidence = _repair_response_entropy_position(row)
        entropy_histogram[entropy_position] = (
            entropy_histogram.get(entropy_position, 0) + 1
        )
        typed_response_id = _bounded_content_key(
            "repair_budget_typed_response",
            (
                row.get("acquisition_id"),
                row.get("candidate_id"),
                row.get("correction_family"),
                rank,
                requested_bytes,
                objective_delta,
                entropy_position,
            ),
        )
        marginal_curves = _repair_response_marginal_curve(
            requested_bytes=requested_bytes,
            segnet_delta=segnet_delta,
            posenet_delta=posenet_delta,
            component_delta=component_delta,
            rate_delta=rate_delta,
            objective_delta=objective_delta,
        )
        interaction_scope = _repair_response_interaction_scope(row)
        local_advisory_custody = _repair_response_local_advisory_custody(row)
        normalized.update(
            {
                "schema": REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA,
                "rank": rank,
                "typed_response_id": typed_response_id,
                "source_response_schema": row.get("schema"),
                "source_response_artifact_path": row.get("response_artifact_path")
                or row.get("source_harvest_path"),
                "operation_levels": _targeted_component_response_operation_levels(row),
                "targeted_dimensions": _targeted_component_response_targeted_dimensions(
                    row
                ),
                "entropy_position_label": entropy_position,
                "entropy_position_evidence": entropy_evidence,
                "entropy_position_model": {
                    "schema": "frontier_rate_attack_entropy_position_model.v1",
                    "class": entropy_position,
                    "principle": (
                        "before_coder_shapes_symbol_distribution; at_coder_attacks_"
                        "integer_codeword_or_model_gap; after_coder_can_only_remove_"
                        "container_or_runtime_grammar_overhead"
                    ),
                    "receiver_role": "deterministic_decode_only_no_eval_time_adaptation",
                    "allowed_use": "local_action_functional_feature_only",
                    "forbidden_use": "score_claim_or_dispatch_authority",
                    **FALSE_AUTHORITY,
                },
                "requested_repair_bytes": requested_bytes,
                "segnet_delta_score_units": segnet_delta,
                "posenet_delta_score_units": posenet_delta,
                "component_delta_score_units": component_delta,
                "lambda_delta_bytes_score_units": rate_delta,
                "objective_delta_score_units": objective_delta,
                "marginal_response_curves": marginal_curves,
                "interaction_scope": interaction_scope,
                "local_advisory_custody": local_advisory_custody,
                "mlx_advisory_custody_present": local_advisory_custody[
                    "mlx_advisory_custody_present"
                ],
                "missing_local_advisory_artifacts": list(
                    local_advisory_custody["missing_required_artifact_keys"]
                ),
                "missing_local_advisory_artifact_blockers": list(
                    local_advisory_custody["missing_artifact_blockers"]
                ),
                "stacking_interaction_terms": {
                    "schema": (
                        "frontier_rate_attack_repair_response_stacking_"
                        "interaction_terms.v1"
                    ),
                    "status": "measured_stack_interactions_required_before_promotion",
                    "same_pair_region_mode_collision_possible": bool(
                        interaction_scope.get("pair_indices")
                        or interaction_scope.get("region_ids")
                        or interaction_scope.get("mode_ids")
                    ),
                    "synergy_or_antagonism_score_units": None,
                    "must_remeasure_with_parent_and_sibling_repairs": True,
                    "allowed_use": "queue_owned_stackability_prior_only",
                    "forbidden_use": "score_claim_or_dispatch_authority",
                    **FALSE_AUTHORITY,
                },
                "hard_constraints": [
                    "allocated_bytes_must_not_exceed_receiver_closed_rate_credit",
                    "parent_rate_only_archive_must_materialize_first",
                    "receiver_consumes_materialized_runtime_output",
                    "component_response_replayed_before_budget_spend",
                    "exact_auth_eval_required_before_score_or_promotion_claim",
                ],
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_exact_eval_dispatch": False,
                "allowed_use": "typed_response_row_for_repair_waterfill_only",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
        require_no_truthy_authority_fields(
            normalized,
            context=f"repair_budget_typed_response_row:{typed_response_id}",
        )
        typed_rows.append(normalized)

    objective_improvement_total = sum(
        float(
            (
                row.get("marginal_response_curves", {})
                if isinstance(row.get("marginal_response_curves"), Mapping)
                else {}
            )
            .get("objective", {})
            .get("improvement_score_units")
            or 0.0
        )
        for row in typed_rows
        if isinstance(row.get("marginal_response_curves"), Mapping)
        and isinstance(row["marginal_response_curves"].get("objective"), Mapping)
    )
    mlx_custody_count = sum(
        1 for row in typed_rows if row.get("mlx_advisory_custody_present") is True
    )
    missing_local_advisory_artifacts = _unique_strings(
        item
        for row in typed_rows
        for item in _string_list(row.get("missing_local_advisory_artifact_blockers"))
    )
    payload = {
        "schema": REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA,
        "row_schema": REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA,
        "row_count": len(typed_rows),
        "accepted_response_count": len(typed_rows),
        "available_receiver_closed_rate_credit_bytes": max(
            0,
            int(available_rate_credit_bytes),
        ),
        "lambda_rate_score_per_byte": lambda_rate_score_per_byte,
        "optimization_objective": (
            "minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes"
        ),
        "sort_key": (
            "measured_lagrangian_delta_score_units_ascending_then_acquisition_id"
        ),
        "entropy_position_histogram": dict(sorted(entropy_histogram.items())),
        "objective_improvement_score_units_total": objective_improvement_total,
        "mlx_advisory_custody_row_count": mlx_custody_count,
        "missing_mlx_advisory_custody_row_count": len(typed_rows) - mlx_custody_count,
        "missing_local_advisory_artifact_blockers": missing_local_advisory_artifacts,
        "hard_constraints": [
            "rate_only_floor_preserved_as_parent_candidate",
            "receiver_closed_rate_credit_is_a_hard_byte_budget",
            "local_mlx_or_cpu_response_is_planning_signal_only",
            "exact_cpu_or_cuda_auth_axis_required_before_dispatch_or_promotion",
        ],
        "rows": typed_rows,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "typed_cumulative_repair_response_ledger_for_queue_waterfill",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="repair_budget_typed_response_ledger",
    )
    return payload


def _repair_waterfill_allocation_action_term(
    *,
    row: Mapping[str, Any],
    rank: int,
    requested_bytes: int,
    proposed_bytes: int,
    remaining_after: int,
    lagrangian_delta: float | None,
    component_delta: float | None,
) -> dict[str, Any]:
    local_terms = (
        row.get("local_cpu_component_terms")
        if isinstance(row.get("local_cpu_component_terms"), Mapping)
        else {}
    )
    segnet_delta = _finite_float_or_none(local_terms.get("segnet_delta_score_units"))
    posenet_delta = _finite_float_or_none(local_terms.get("posenet_delta_score_units"))
    correction_rate_delta = _finite_float_or_none(
        local_terms.get("correction_rate_delta_score_units")
    )
    if correction_rate_delta is None:
        correction_rate_delta = rate_delta_for_archive_byte_delta(proposed_bytes)
    if component_delta is None and segnet_delta is not None and posenet_delta is not None:
        component_delta = segnet_delta + posenet_delta
    objective_delta = (
        lagrangian_delta
        if lagrangian_delta is not None
        else (
            (component_delta or 0.0) + correction_rate_delta
            if component_delta is not None
            else None
        )
    )
    action_id = _bounded_content_key(
        "repair_waterfill_allocation_action",
        (
            row.get("typed_response_id"),
            row.get("acquisition_id"),
            row.get("candidate_id"),
            row.get("correction_family"),
            requested_bytes,
            proposed_bytes,
            rank,
        ),
    )
    rate_packet_context = _targeted_rate_packet_context(row)
    return {
        "schema": REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA,
        "rank": rank,
        "operator_action_id": action_id,
        "acquisition_id": row.get("acquisition_id"),
        "candidate_id": row.get("candidate_id"),
        "T_i": {
            "operator_id": row.get("candidate_id") or action_id,
            "correction_family": row.get("correction_family"),
            "targeted_dimensions": list(row.get("targeted_dimensions") or []),
            "operation_levels": list(row.get("operation_levels") or []),
            "archive_byte_delta_vs_parent_rate_floor": proposed_bytes,
            "requested_repair_bytes": requested_bytes,
            "allocated_repair_bytes": proposed_bytes,
            "segnet_delta_score_units": segnet_delta,
            "posenet_delta_score_units": posenet_delta,
            "component_delta_score_units": component_delta,
            "lambda_delta_bytes_score_units": correction_rate_delta,
            "objective_delta_score_units": objective_delta,
            "typed_response_id": row.get("typed_response_id"),
            "entropy_position_label": row.get("entropy_position_label"),
            "entropy_position_model": dict(
                row.get("entropy_position_model")
                if isinstance(row.get("entropy_position_model"), Mapping)
                else {}
            ),
            "marginal_response_curves": dict(
                row.get("marginal_response_curves")
                if isinstance(row.get("marginal_response_curves"), Mapping)
                else {}
            ),
            "interaction_scope": dict(
                row.get("interaction_scope")
                if isinstance(row.get("interaction_scope"), Mapping)
                else {}
            ),
            "local_advisory_custody": dict(
                row.get("local_advisory_custody")
                if isinstance(row.get("local_advisory_custody"), Mapping)
                else {}
            ),
            "mlx_advisory_custody_present": (
                row.get("mlx_advisory_custody_present") is True
            ),
            "stacking_interaction_terms": dict(
                row.get("stacking_interaction_terms")
                if isinstance(row.get("stacking_interaction_terms"), Mapping)
                else {}
            ),
            **_targeted_rate_packet_context_fields(row),
            "receiver_closed_rate_packet_context": dict(rate_packet_context),
            **FALSE_AUTHORITY,
        },
        "R_i": {
            "receiver_proof_kind": "receiver_consumed_spent_budget_child_archive",
            "parent_rate_only_floor_required": True,
            "receiver_closed_rate_packet_context": dict(rate_packet_context),
            "component_response_replay_required": True,
            "parser_only_proof_rejected": True,
            "deterministic_adapter_only": True,
            "entropy_position_constraint": (
                "repair_and_rate_attack_are_encoder_side_only; receiver_must_not_"
                "optimize_or_inspect_scorer_state"
            ),
            **FALSE_AUTHORITY,
        },
        "interaction_terms": {
            "schema": "frontier_rate_attack_repair_allocation_interaction_terms.v1",
            "status": "unmeasured_until_spent_budget_child_materialization",
            "remaining_receiver_closed_rate_credit_bytes_after": remaining_after,
            "must_remeasure_parent_child_interaction_before_promotion": True,
            "synergy_or_antagonism_score_units": None,
            "typed_response_id": row.get("typed_response_id"),
            "source_interaction_scope": dict(
                row.get("interaction_scope")
                if isinstance(row.get("interaction_scope"), Mapping)
                else {}
            ),
            **FALSE_AUTHORITY,
        },
        "legal_runtime_constraints": [
            "parent_rate_only_archive_materialized_before_child",
            "receiver_consumes_materialized_runtime_output",
            "component_response_replayed_before_budget_spend",
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ],
        "budget_spend_allowed": False,
        "allowed_use": "typed_repair_waterfill_allocation_term_for_queue_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _repair_waterfill_allocation_rows(
    *,
    accepted_rows: Sequence[Mapping[str, Any]],
    available_rate_credit_bytes: int,
) -> list[dict[str, Any]]:
    remaining = max(0, int(available_rate_credit_bytes))
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted(accepted_rows, key=_targeted_component_response_sort_key), start=1):
        lagrangian_delta = _finite_float_or_none(
            row.get("measured_lagrangian_delta_score_units")
        )
        component_delta = _finite_float_or_none(
            row.get("measured_component_delta_score_units")
        )
        added_bytes = _repair_waterfill_added_bytes(row)
        requested_bytes = added_bytes if added_bytes is not None else 0
        proposed_bytes = min(remaining, requested_bytes) if requested_bytes > 0 else 0
        remaining -= proposed_bytes
        allocation_action_term = _repair_waterfill_allocation_action_term(
            row=row,
            rank=rank,
            requested_bytes=requested_bytes,
            proposed_bytes=proposed_bytes,
            remaining_after=remaining,
            lagrangian_delta=lagrangian_delta,
            component_delta=component_delta,
        )
        allocation_t = allocation_action_term.get("T_i", {})
        blockers = [
            "exact_axis_component_response_required_before_budget_spend",
            "receiver_runtime_materialization_required_before_budget_spend",
        ]
        if added_bytes is None:
            blockers.append("correction_added_archive_bytes_missing")
        if proposed_bytes <= 0:
            blockers.append("no_receiver_closed_rate_credit_bytes_allocated")
        blockers.extend(_string_list(row.get("missing_local_advisory_artifact_blockers")))
        rate_packet_context = _targeted_rate_packet_context(row)
        out.append(
            {
                "schema": "frontier_rate_attack_repair_budget_waterfill_allocation_row.v1",
                "rank": rank,
                "acquisition_id": row.get("acquisition_id"),
                "candidate_id": row.get("candidate_id"),
                "correction_family": row.get("correction_family"),
                "typed_response_id": row.get("typed_response_id"),
                "targeted_dimensions": list(row.get("targeted_dimensions") or []),
                "operation_levels": list(row.get("operation_levels") or []),
                "entropy_position_label": row.get("entropy_position_label"),
                "entropy_position_model": dict(
                    row.get("entropy_position_model")
                    if isinstance(row.get("entropy_position_model"), Mapping)
                    else {}
                ),
                "allocation_action_term": allocation_action_term,
                "measured_component_delta_score_units": component_delta,
                "measured_lagrangian_delta_score_units": lagrangian_delta,
                "segnet_delta_score_units": (
                    allocation_t.get("segnet_delta_score_units")
                    if isinstance(allocation_t, Mapping)
                    else None
                ),
                "posenet_delta_score_units": (
                    allocation_t.get("posenet_delta_score_units")
                    if isinstance(allocation_t, Mapping)
                    else None
                ),
                "correction_rate_delta_score_units": (
                    allocation_t.get("lambda_delta_bytes_score_units")
                    if isinstance(allocation_t, Mapping)
                    else None
                ),
                "local_lagrangian_improvement_score_units": (
                    -lagrangian_delta
                    if lagrangian_delta is not None and lagrangian_delta < 0.0
                    else 0.0
                ),
                "marginal_response_curves": dict(
                    row.get("marginal_response_curves")
                    if isinstance(row.get("marginal_response_curves"), Mapping)
                    else {}
                ),
                "interaction_scope": dict(
                    row.get("interaction_scope")
                    if isinstance(row.get("interaction_scope"), Mapping)
                    else {}
                ),
                "local_advisory_custody": dict(
                    row.get("local_advisory_custody")
                    if isinstance(row.get("local_advisory_custody"), Mapping)
                    else {}
                ),
                "mlx_advisory_custody_present": (
                    row.get("mlx_advisory_custody_present") is True
                ),
                "missing_local_advisory_artifacts": _string_list(
                    row.get("missing_local_advisory_artifacts")
                ),
                "missing_local_advisory_artifact_blockers": _string_list(
                    row.get("missing_local_advisory_artifact_blockers")
                ),
                "stacking_interaction_terms": dict(
                    row.get("stacking_interaction_terms")
                    if isinstance(row.get("stacking_interaction_terms"), Mapping)
                    else {}
                ),
                "estimated_receiver_closed_rate_credit_score_units": row.get(
                    "estimated_receiver_closed_rate_credit_score_units"
                ),
                **_targeted_rate_packet_context_fields(row),
                "receiver_closed_rate_packet_context": dict(rate_packet_context),
                "requested_repair_bytes": requested_bytes,
                "proposed_encoder_repair_bytes": proposed_bytes,
                "remaining_receiver_closed_rate_credit_bytes_after": remaining,
                "source_response_artifact_path": row.get("response_artifact_path")
                or row.get("source_harvest_path"),
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "budget_spend_blockers": _unique_strings(blockers),
                "allowed_use": "encoder_repair_allocator_local_waterfill_plan_only",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
    return out


def _repair_cascade_opportunity_rows() -> list[dict[str, Any]]:
    """Operator-surfaced structural repair cascades that still need measurement."""

    return [
        {
            "schema": REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA,
            "cascade_id": "cascade_c_posenet_null_segnet_region_selector_codec",
            "label": "Cascade C",
            "source_hint": (
                "P19 PoseNet-null bottom-decile + P18 SegNet-class-region "
                "waterfill + P11 per-region selector codec"
            ),
            "source_catalog_path": (
                ".omx/research/entropy_position_cascade_exploit_catalog_20260526.md"
            ),
            "source_relation": "PR110-OPT-5+7+10+12_UNTOUCHED",
            "estimated_score_delta_score_units": None,
            "estimated_archive_bytes_delta": None,
            "estimate_status": (
                "per_region_selector_codec_variant_empirically_falsified_"
                "scorer_repair_hypothesis_survives"
            ),
            "empirical_feedback": {
                "schema": "frontier_rate_attack_repair_cascade_empirical_feedback.v1",
                "source_artifact_paths": [
                    ".omx/research/cascade_c_artifacts_20260526/"
                    "cascade_c_per_region_codec_empirical.json",
                    ".omx/research/cascade_c_artifacts_20260526/"
                    "cascade_c_alternative_reducers_empirical.json",
                ],
                "axis_tag": "[macOS-CPU advisory]",
                "evidence_grade": "compress_time_encoding_statistics_only",
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "tested_variant": "per_region_selector_codec",
                "variant_verdict": "implementation_level_falsification",
                "best_observed_delta_vs_fec6_wire_bytes": 83,
                "best_observed_total_wire_bytes": 332,
                "baseline_fec6_wire_bytes": 249,
                "root_cause": (
                    "per_region_partition_preserves_joint_entropy_as_"
                    "H_region_plus_H_mode_given_region_and_adds_region_index_overhead"
                ),
                "surviving_hypotheses": [
                    "fold_posenet_null_signal_into_fec8_markov_transition_matrix",
                    "segnet_class_region_waterfill_as_future_frame1_selector_axis",
                    "posenet_bottom_decile_signal_for_trainer_curriculum_design",
                ],
                "blocked_variant_reuse": (
                    "do_not_rank_plain_per_region_selector_codec_as_unestimated_win"
                ),
                **FALSE_AUTHORITY,
            },
            "targeted_positions": [
                {
                    "position_id": "P19",
                    "position_name": "PoseNet entropy",
                    "repair_role": "select_bottom_decile_posenet_null_pairs",
                    "entropy_surface": "scorer_entropy",
                },
                {
                    "position_id": "P18",
                    "position_name": "SegNet entropy",
                    "repair_role": "class_region_waterfill_over_segnet_margin",
                    "entropy_surface": "scorer_entropy",
                },
                {
                    "position_id": "P11",
                    "position_name": "selector_stream",
                    "repair_role": "per_region_selector_codec_for_replay",
                    "entropy_surface": "selector_codec_entropy",
                },
            ],
            "pipeline_position": "scorer_entropy_repair_before_selector_codec",
            "canonical_mechanisms": [
                {
                    "mechanism_id": "uniward_textured_region_undetectability",
                    "source_lineage": "Fridrich_inverse_steganalysis",
                    "queue_obligation": "prefer_segnet_texture_or_class_regions_with_low_detector_margin",
                },
                {
                    "mechanism_id": "detector_informed_embedding",
                    "source_lineage": "Quantizr_TTO_scorer_informed_embedding",
                    "queue_obligation": "rank_repairs_by_measured_segnet_posenet_response_not_visual_proxy",
                },
                {
                    "mechanism_id": "square_root_law_capacity",
                    "source_lineage": "Yousfi_Fridirich_capacity_bound",
                    "queue_obligation": "model_per_pair_payload_capacity_as_sublinear_in_safe_region_count",
                },
                {
                    "mechanism_id": "cnn_blind_spot_texture_and_dct_statistics",
                    "source_lineage": "EfficientNet_stride2_stem_blind_spot_below_256x192",
                    "queue_obligation": "probe_segnet_stride2_subcell_regions_and_texture_statistics",
                },
            ],
            "required_probe_measurements": [
                "posenet_null_bottom_decile_pair_ids",
                "segnet_class_region_mask_ids",
                "segnet_logit_margin_or_detector_margin",
                "texture_region_capacity_proxy",
                "selector_payload_bits_per_region",
                "receiver_consumed_runtime_replay_proof",
            ],
            "optimization_implication": (
                "can move distortion budget where scorer response is lowest, then "
                "compress the chosen per-region decisions with a selector codec"
            ),
            "required_empirical_landing": (
                "MLX-local paired probe over PoseNet-null bottom-decile pairs, "
                "SegNet class-region masks, and per-region selector payload bytes"
            ),
            "next_queue_action": (
                "build_cascade_c_mlx_local_probe_queue_and_emit_component_response_rows"
            ),
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_materializer_execution": False,
            "ready_for_exact_eval_dispatch": False,
            "blockers": [
                "cascade_c_empirical_component_response_missing",
                "per_region_selector_codec_materializer_missing",
                "receiver_runtime_consumption_proof_missing",
                "exact_auth_eval_required_before_score_or_promotion_claim",
            ],
            "allowed_use": "structural_repair_opportunity_for_queue_prioritization",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
    ]


def build_frontier_repair_budget_waterfill_work_order(
    *,
    autonomous_chain_optimization: Mapping[str, Any],
    chain_id: str,
    targeted_component_correction_response_harvest: Mapping[str, Any],
    receiver_closed_correction_budget: Mapping[str, Any],
    autonomous_chain_optimization_path: str | Path | None = None,
    targeted_component_correction_response_harvest_path: str | Path | None = None,
    receiver_closed_correction_budget_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit an encoder-side repair waterfill plan from measured local responses."""

    row = _autonomous_chain_row_for_id(autonomous_chain_optimization, chain_id)
    if (
        targeted_component_correction_response_harvest.get("schema")
        != TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "targeted component correction response harvest has unexpected schema"
        )
    if (
        receiver_closed_correction_budget.get("schema")
        != RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "receiver closed correction budget has unexpected schema"
        )
    require_no_truthy_authority_fields(
        targeted_component_correction_response_harvest,
        context="repair_waterfill_response_harvest",
    )
    require_no_truthy_authority_fields(
        receiver_closed_correction_budget,
        context="repair_waterfill_receiver_closed_budget",
    )
    accepted_rows = _accepted_targeted_component_response_rows(
        targeted_component_correction_response_harvest
    )
    available_bytes = (
        _finite_int_or_none(
            receiver_closed_correction_budget.get("receiver_closed_saved_bytes_total")
        )
        or 0
    )
    rate_budget_preservation_plan = (
        row.get("rate_budget_preservation_plan")
        if isinstance(row.get("rate_budget_preservation_plan"), Mapping)
        else {}
    )
    operator_action_ledger = (
        rate_budget_preservation_plan.get("operator_action_ledger")
        if isinstance(rate_budget_preservation_plan.get("operator_action_ledger"), Mapping)
        else {}
    )
    typed_response_ledger = _repair_budget_typed_response_ledger(
        accepted_rows=accepted_rows,
        available_rate_credit_bytes=available_bytes,
    )
    allocation_rows = _repair_waterfill_allocation_rows(
        accepted_rows=[
            item
            for item in typed_response_ledger.get("rows") or []
            if isinstance(item, Mapping)
        ],
        available_rate_credit_bytes=available_bytes,
    )
    proposed_bytes_total = sum(
        int(item.get("proposed_encoder_repair_bytes") or 0)
        for item in allocation_rows
    )
    local_improvement_total = sum(
        float(item.get("local_lagrangian_improvement_score_units") or 0.0)
        for item in allocation_rows
    )
    blockers = [
        "exact_auth_eval_required_before_score_or_promotion_claim",
        "exact_axis_component_response_required_before_budget_spend",
        "receiver_runtime_materialization_required_before_budget_spend",
        "local_cpu_mlx_waterfill_is_not_budget_spend_authority",
    ]
    if not accepted_rows:
        blockers.append("no_negative_local_lagrangian_response_rows")
    if available_bytes <= 0:
        blockers.append("no_receiver_closed_rate_credit_bytes_available")
    if proposed_bytes_total <= 0:
        blockers.append("no_repair_bytes_allocated_by_local_waterfill")
    receiver_closed_credit_rows = _receiver_closed_rate_credit_rows(
        receiver_closed_correction_budget
    )
    cascade_opportunity_rows = _repair_cascade_opportunity_rows()
    payload = {
        "schema": REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA,
        "generated_at_utc": autonomous_chain_optimization.get("generated_at_utc"),
        "chain_id": chain_id,
        "chain_family": row.get("chain_family"),
        "optimization_objective": row.get("optimization_objective"),
        "pipeline_side": "encoder_repair_allocator",
        "pipeline_placement": {
            "owner": "encoder_repair_allocator_before_archive_packaging",
            "rate_credit_source": "receiver_closed_materializer_saved_bytes",
            "distortion_debt_source": "measured_segnet_posenet_component_response",
            "receiver_role": "decode_only_consume_final_packet",
            "exact_eval_role": "downstream_authority_gate_only",
            **FALSE_AUTHORITY,
        },
        "source_artifact_paths": _unique_strings(
            [
                str(autonomous_chain_optimization_path or ""),
                str(targeted_component_correction_response_harvest_path or ""),
                str(receiver_closed_correction_budget_path or ""),
            ]
        ),
        "repair_budget_waterfill_plan": dict(
            row.get("repair_budget_waterfill_plan")
            if isinstance(row.get("repair_budget_waterfill_plan"), Mapping)
            else {}
        ),
        "rate_budget_preservation_plan": dict(rate_budget_preservation_plan),
        "operator_action_ledger": dict(operator_action_ledger),
        "action_functional_lineage": {
            "schema": "frontier_rate_attack_repair_waterfill_action_functional_lineage.v1",
            "upstream_rate_budget_preservation_schema": (
                rate_budget_preservation_plan.get("schema")
            ),
            "upstream_operator_action_ledger_schema": operator_action_ledger.get("schema"),
            "upstream_operator_action_term_schema": OPERATOR_ACTION_TERM_SCHEMA,
            "repair_allocation_action_term_schema": (
                REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
            ),
            "typed_response_ledger_schema": REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA,
            "typed_response_row_schema": REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA,
            "upstream_waterfill_solver_schema": (
                rate_budget_preservation_plan.get("waterfill_solver", {}).get("schema")
                if isinstance(
                    rate_budget_preservation_plan.get("waterfill_solver"),
                    Mapping,
                )
                else None
            ),
            "upstream_action_functional_schema": (
                rate_budget_preservation_plan.get("action_functional", {}).get("schema")
                if isinstance(
                    rate_budget_preservation_plan.get("action_functional"),
                    Mapping,
                )
                else None
            ),
            "component_response_harvest_schema": (
                targeted_component_correction_response_harvest.get("schema")
            ),
            "receiver_closed_budget_schema": receiver_closed_correction_budget.get(
                "schema"
            ),
            "objective": (
                "minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes"
            ),
            "implemented_as": (
                "queue_child_of_existing_rate_budget_preservation_and_component_"
                "response_action_functional_surfaces"
            ),
            "new_parallel_action_functional_created": False,
            "budget_spend_allowed": False,
            **FALSE_AUTHORITY,
        },
        "preservation_contract": {
            "schema": "frontier_rate_attack_repair_waterfill_preservation_contract.v1",
            "emit_rate_only_floor_archive_before_repair_archive": True,
            "repair_archive_must_reference_parent_rate_only_preservation_id": True,
            "rate_only_candidate_remains_valid_even_if_repair_candidate_regresses": True,
            "rebrotli_default_after_rate_attack": True,
            "budget_spend_allowed": False,
            **FALSE_AUTHORITY,
        },
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": available_bytes,
            "receiver_closed_saved_bytes_max": receiver_closed_correction_budget.get(
                "receiver_closed_saved_bytes_max"
            ),
            "receiver_closed_candidate_count": receiver_closed_correction_budget.get(
                "receiver_closed_candidate_count"
            ),
            "source_targets": list(
                receiver_closed_correction_budget.get(
                    "receiver_closed_budget_source_targets"
                )
                or []
            ),
            "receiver_closed_rate_credit_rows": receiver_closed_credit_rows,
            "receiver_closed_rate_credit_row_count": len(receiver_closed_credit_rows),
            **FALSE_AUTHORITY,
        },
        "accepted_response_count": len(accepted_rows),
        "typed_response_ledger_schema": REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA,
        "typed_response_row_schema": REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA,
        "typed_response_row_count": typed_response_ledger.get("row_count"),
        "typed_response_ledger": typed_response_ledger,
        "allocation_row_count": len(allocation_rows),
        "repair_cascade_opportunity_count": len(cascade_opportunity_rows),
        "repair_cascade_opportunity_rows": cascade_opportunity_rows,
        "proposed_encoder_repair_bytes_total": proposed_bytes_total,
        "local_lagrangian_improvement_score_units_total": local_improvement_total,
        "allocation_rows": allocation_rows,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_materializer_execution": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": _unique_strings(blockers),
        "recommended_next_action": (
            "materialize_receiver_consumed_repair_candidates_for_exact_axis_component_replay"
            if allocation_rows and proposed_bytes_total > 0
            else "collect_component_response_rows_and_receiver_closed_rate_credit"
        ),
        "allowed_use": "queue_owned_encoder_repair_waterfill_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context=f"frontier_repair_budget_waterfill_work_order:{chain_id}",
    )
    return payload


def _receiver_closed_rate_credit_rows(
    receiver_closed_correction_budget: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in receiver_closed_correction_budget.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        submission_dir = str(row.get("submission_dir") or "").strip()
        archive_path = str(
            row.get("candidate_archive_path")
            or row.get("archive_path")
            or row.get("archive_zip_path")
            or ""
        ).strip()
        if not archive_path and submission_dir:
            archive_path = f"{submission_dir.rstrip('/')}/archive.zip"
        runtime_proof_path = str(
            row.get("runtime_consumption_proof_path")
            or row.get("receiver_runtime_consumption_proof_path")
            or ""
        ).strip()
        if not runtime_proof_path and submission_dir:
            runtime_proof_path = (
                f"{submission_dir.rstrip('/')}/runtime_consumption_proof.json"
            )
        rows.append(
            {
                "schema": "frontier_rate_attack_receiver_closed_rate_credit_row.v1",
                "candidate_id": row.get("candidate_id"),
                "target_kind": row.get("target_kind"),
                "saved_bytes": row.get("saved_bytes_at_risk")
                or row.get("receiver_closed_saved_bytes")
                or row.get("saved_bytes"),
                "archive_path": archive_path or None,
                "archive_sha256": row.get("archive_sha256")
                or row.get("candidate_archive_sha256"),
                "archive_bytes": row.get("archive_bytes")
                or row.get("candidate_archive_bytes"),
                "runtime_consumption_proof_path": runtime_proof_path or None,
                "receiver_closed": row.get("receiver_closed") is True,
                "closed_source_queue_path": row.get("closed_source_queue_path"),
                "closure_report_path": row.get("closure_report_path"),
                "submission_dir": row.get("submission_dir"),
                "source_archive_path": row.get("source_archive_path"),
                "source_archive_sha256": row.get("source_archive_sha256"),
                **_targeted_rate_packet_context_fields(row),
                "receiver_closed_rate_packet_context": _targeted_rate_packet_context(
                    row
                ),
                "allowed_use": "rate_only_parent_materialization_evidence",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _matching_receiver_closed_rate_credit_row(
    *,
    work_order: Mapping[str, Any],
    target_kinds: Sequence[str],
    source_candidate_ids: Sequence[str],
    rate_row_count: int,
) -> Mapping[str, Any]:
    if rate_row_count != 1:
        return {}
    credit = (
        work_order.get("receiver_closed_rate_credit")
        if isinstance(work_order.get("receiver_closed_rate_credit"), Mapping)
        else {}
    )
    rows = [
        row
        for row in credit.get("receiver_closed_rate_credit_rows") or []
        if isinstance(row, Mapping)
    ]
    target_set = set(target_kinds)
    candidate_set = set(source_candidate_ids)
    for row in rows:
        if str(row.get("target_kind") or "") in target_set:
            return row
        if str(row.get("candidate_id") or "") in candidate_set:
            return row
    return {}


def _entropy_pipeline_position(target_kind: str) -> str:
    if target_kind in {
        PACKET_MEMBER_MERGE_TARGET_KIND,
        PACKET_MEMBER_REORDER_TARGET_KIND,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND,
        ARCHIVE_SECTION_REORDER_TARGET_KIND,
    }:
        return "after_entropy_coder_container_or_zip_grammar"
    if target_kind in {
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
    }:
        return "at_entropy_coder"
    if target_kind in {
        RENDERER_PAYLOAD_DFL1_TARGET_KIND,
        TENSOR_FACTORIZE_TARGET_KIND,
        TENSOR_PRUNE_TARGET_KIND,
        TENSOR_QUANTIZE_TARGET_KIND,
        TENSOR_SHARED_CODEBOOK_TARGET_KIND,
        ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND,
    }:
        return "before_entropy_coder_distribution_shaping"
    return "unknown_entropy_pipeline_position"


def _entropy_pipeline_position_rows(target_kinds: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_kind in _unique_strings(target_kinds):
        position = _entropy_pipeline_position(target_kind)
        if position == "before_entropy_coder_distribution_shaping":
            opportunity = (
                "can_change_symbol_distribution_before_coding_and_stack_with_"
                "downstream_entropy_recode"
            )
        elif position == "at_entropy_coder":
            opportunity = (
                "can_attack_codeword_integer_boundary_and_model_mismatch_directly"
            )
        elif position == "after_entropy_coder_container_or_zip_grammar":
            opportunity = (
                "cannot_reduce_payload_entropy_unless_it_removes_container_or_runtime_"
                "grammar_overhead"
            )
        else:
            opportunity = "requires_empirical_stage_classification_before_ranking"
        rows.append(
            {
                "schema": "frontier_rate_attack_entropy_pipeline_position.v1",
                "target_kind": target_kind,
                "entropy_pipeline_position": position,
                "optimization_implication": opportunity,
                "composition_hint": (
                    "prefer_before_coder_distribution_shaping_then_at_coder_recode_"
                    "then_container_overhead_cleanup"
                ),
                "allowed_use": "local_action_functional_ranking_feature",
                "forbidden_use": "score_claim_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _repair_budget_rate_only_parent_row(
    *,
    chain_id: str,
    rate_budget_preservation_plan: Mapping[str, Any],
    work_order: Mapping[str, Any],
    repo_root: str | Path = _DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    rate_rows = [
        row
        for row in rate_budget_preservation_plan.get("rows") or []
        if isinstance(row, Mapping)
        and row.get("schema") == RATE_BUDGET_PRESERVATION_ROW_SCHEMA
        and row.get("preserve_as_rate_only_candidate") is True
    ]
    preservation_ids = _unique_strings(row.get("preservation_id") for row in rate_rows)
    source_candidate_ids = _unique_strings(row.get("candidate_id") for row in rate_rows)
    target_kinds = _unique_strings(row.get("target_kind") for row in rate_rows)
    entropy_positions = _entropy_pipeline_position_rows(target_kinds)
    cumulative = (
        rate_budget_preservation_plan.get("cumulative_rate_attack")
        if isinstance(rate_budget_preservation_plan.get("cumulative_rate_attack"), Mapping)
        else {}
    )
    operator_action_ledger = (
        rate_budget_preservation_plan.get("operator_action_ledger")
        if isinstance(rate_budget_preservation_plan.get("operator_action_ledger"), Mapping)
        else {}
    )
    saved_bytes_total = _finite_int_or_none(
        cumulative.get("saved_bytes_total")
    ) or sum(int(row.get("saved_bytes") or 0) for row in rate_rows)
    rate_credit_total = _finite_float_or_none(
        cumulative.get("rate_credit_score_units_total")
    ) or sum(float(row.get("rate_credit_score_units") or 0.0) for row in rate_rows)
    distortion_debt_total = sum(
        float(row.get("distortion_debt_score_units") or 0.0) for row in rate_rows
    )
    net_delta_total = sum(
        float(row.get("net_score_delta_score_units") or 0.0) for row in rate_rows
    )
    parent_candidate_chain_id = _bounded_content_key(
        "repair_rate_floor_parent",
        (chain_id, tuple(preservation_ids), saved_bytes_total, rate_credit_total),
    )
    receiver_credit = _matching_receiver_closed_rate_credit_row(
        work_order=work_order,
        target_kinds=target_kinds,
        source_candidate_ids=source_candidate_ids,
        rate_row_count=len(rate_rows),
    )
    archive_path = str(receiver_credit.get("archive_path") or "").strip()
    archive_sha = str(receiver_credit.get("archive_sha256") or "").strip()
    archive_bytes = receiver_credit.get("archive_bytes")
    runtime_proof_path = str(
        receiver_credit.get("runtime_consumption_proof_path") or ""
    ).strip()
    archive_revalidation = _candidate_archive_revalidation(
        archive_path_text=archive_path,
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        repo_root=repo_root,
        context="receiver_closed_rate_credit_archive",
    )
    proof_revalidation = _runtime_consumption_proof_revalidation(
        proof_path_text=runtime_proof_path,
        repo_root=repo_root,
        expected_candidate_archive_sha256=archive_sha,
        context="receiver_closed_rate_credit_runtime_consumption_proof",
    )
    archive_materialized = archive_revalidation["archive_valid"] is True
    runtime_proof_present = proof_revalidation["proof_valid"] is True
    receiver_consumed = bool(
        archive_materialized
        and runtime_proof_present
        and receiver_credit.get("receiver_closed") is True
        and proof_revalidation["receiver_contract_satisfied"] is True
    )
    blockers = [
        "full_frame_inflate_parity_required_before_exact_readiness",
        "exact_auth_eval_required_before_score_or_promotion_claim",
    ]
    blockers.extend(_string_list(archive_revalidation.get("blockers")))
    blockers.extend(_string_list(proof_revalidation.get("blockers")))
    if not archive_materialized:
        blockers.append("rate_only_candidate_archive_materialization_missing")
    if not runtime_proof_present or not receiver_consumed:
        blockers.append("receiver_runtime_consumption_proof_missing")
    if len(rate_rows) > 1:
        blockers.append("cumulative_rate_only_archive_composition_required")
    if not rate_rows:
        blockers.append("no_rate_only_preservation_rows_available")
    if saved_bytes_total <= 0:
        blockers.append("no_rate_only_saved_bytes_available")
    return {
        "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
        "candidate_kind": "rate_only_floor_parent",
        "candidate_chain_id": parent_candidate_chain_id,
        "chain_id": chain_id,
        "materialization_order": 1,
        "parent_candidate_chain_id": None,
        "preservation_scope": "cumulative_rate_only_archive_before_repair_spend",
        "preservation_ids": preservation_ids,
        "source_candidate_ids": source_candidate_ids,
        "target_kinds": target_kinds,
        "entropy_pipeline_positions": entropy_positions,
        "operator_action_ledger_schema": operator_action_ledger.get("schema"),
        "operator_action_term_schema": operator_action_ledger.get("term_schema"),
        "operator_action_term_count": operator_action_ledger.get("term_count"),
        "operator_action_terms": list(operator_action_ledger.get("terms") or []),
        "action_functional_objective": operator_action_ledger.get("objective"),
        "saved_bytes_total": saved_bytes_total,
        "rate_credit_score_units_total": rate_credit_total,
        "distortion_debt_score_units_total": distortion_debt_total,
        "net_score_delta_score_units_total": net_delta_total,
        "rate_only_candidate_count": len(rate_rows),
        "rate_only_candidate_remains_valid_even_if_child_regresses": True,
        "rebrotli_default_after_rate_attack": bool(
            work_order.get("preservation_contract", {}).get(
                "rebrotli_default_after_rate_attack"
            )
            if isinstance(work_order.get("preservation_contract"), Mapping)
            else False
        ),
        "receiver_closed_rate_credit_binding": dict(receiver_credit),
        "candidate_archive_materialized": archive_materialized,
        "candidate_archive_path": archive_path or None,
        "candidate_archive_sha256": archive_sha or None,
        "candidate_archive_bytes": archive_bytes,
        "candidate_archive_revalidation": archive_revalidation,
        "runtime_consumption_proof_path": runtime_proof_path or None,
        "runtime_consumption_proof_present": runtime_proof_present,
        "runtime_consumption_proof_revalidation": proof_revalidation,
        "receiver_consumed": receiver_consumed,
        "component_response_replayed": False,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_materializer_execution": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": _unique_strings(blockers),
        "allowed_use": "rate_only_floor_candidate_chain_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _repair_budget_spent_child_rows(
    *,
    chain_id: str,
    parent_candidate_chain_id: str,
    work_order: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, allocation in enumerate(work_order.get("allocation_rows") or [], start=2):
        if not isinstance(allocation, Mapping):
            continue
        proposed_bytes = _finite_int_or_none(
            allocation.get("proposed_encoder_repair_bytes")
        ) or 0
        requested_bytes = _finite_int_or_none(allocation.get("requested_repair_bytes")) or 0
        improvement = _finite_float_or_none(
            allocation.get("local_lagrangian_improvement_score_units")
        )
        candidate_chain_id = _bounded_content_key(
            "repair_budget_spent_child",
            (
                chain_id,
                parent_candidate_chain_id,
                allocation.get("rank"),
                allocation.get("candidate_id"),
                allocation.get("correction_family"),
                proposed_bytes,
            ),
        )
        blockers = [
            "parent_rate_only_archive_materialization_required",
            "repair_candidate_archive_materialization_missing",
            "receiver_runtime_consumption_proof_missing",
            "component_response_replay_required_before_budget_spend",
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ]
        blockers.extend(_string_list(allocation.get("budget_spend_blockers")))
        if proposed_bytes <= 0:
            blockers.append("no_encoder_repair_bytes_allocated")
        rows.append(
            {
                "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": candidate_chain_id,
                "chain_id": chain_id,
                "materialization_order": order,
                "parent_candidate_chain_id": parent_candidate_chain_id,
                "parent_must_be_preserved_before_child": True,
                "child_must_not_replace_parent_archive": True,
                "allocation_rank": allocation.get("rank"),
                "allocation_candidate_id": allocation.get("candidate_id"),
                "typed_response_id": allocation.get("typed_response_id"),
                "acquisition_id": allocation.get("acquisition_id"),
                "correction_family": allocation.get("correction_family"),
                "targeted_dimensions": list(allocation.get("targeted_dimensions") or []),
                "operation_levels": list(allocation.get("operation_levels") or []),
                "entropy_position_label": allocation.get("entropy_position_label"),
                "entropy_position_model": dict(
                    allocation.get("entropy_position_model")
                    if isinstance(allocation.get("entropy_position_model"), Mapping)
                    else {}
                ),
                **_targeted_rate_packet_context_fields(allocation),
                "receiver_closed_rate_packet_context": dict(
                    allocation.get("receiver_closed_rate_packet_context")
                    if isinstance(
                        allocation.get("receiver_closed_rate_packet_context"),
                        Mapping,
                    )
                    else _targeted_rate_packet_context(allocation)
                ),
                "allocation_action_term": dict(
                    allocation.get("allocation_action_term")
                    if isinstance(allocation.get("allocation_action_term"), Mapping)
                    else {}
                ),
                "requested_repair_bytes": requested_bytes,
                "proposed_encoder_repair_bytes": proposed_bytes,
                "remaining_receiver_closed_rate_credit_bytes_after": (
                    allocation.get("remaining_receiver_closed_rate_credit_bytes_after")
                ),
                "local_lagrangian_improvement_score_units": improvement,
                "measured_component_delta_score_units": (
                    allocation.get("measured_component_delta_score_units")
                ),
                "measured_lagrangian_delta_score_units": (
                    allocation.get("measured_lagrangian_delta_score_units")
                ),
                "marginal_response_curves": dict(
                    allocation.get("marginal_response_curves")
                    if isinstance(allocation.get("marginal_response_curves"), Mapping)
                    else {}
                ),
                "interaction_scope": dict(
                    allocation.get("interaction_scope")
                    if isinstance(allocation.get("interaction_scope"), Mapping)
                    else {}
                ),
                "local_advisory_custody": dict(
                    allocation.get("local_advisory_custody")
                    if isinstance(allocation.get("local_advisory_custody"), Mapping)
                    else {}
                ),
                "mlx_advisory_custody_present": (
                    allocation.get("mlx_advisory_custody_present") is True
                ),
                "missing_local_advisory_artifacts": _string_list(
                    allocation.get("missing_local_advisory_artifacts")
                ),
                "missing_local_advisory_artifact_blockers": _string_list(
                    allocation.get("missing_local_advisory_artifact_blockers")
                ),
                "stacking_interaction_terms": dict(
                    allocation.get("stacking_interaction_terms")
                    if isinstance(
                        allocation.get("stacking_interaction_terms"),
                        Mapping,
                    )
                    else {}
                ),
                "source_response_artifact_path": allocation.get(
                    "source_response_artifact_path"
                ),
                "acceptance_rule": (
                    "child_candidate_can_supersede_parent_only_after_receiver_consumed_"
                    "materialization_component_replay_and_exact_axis_confirmation"
                ),
                "candidate_archive_materialized": False,
                "candidate_archive_path": None,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": False,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_materializer_execution": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": _unique_strings(blockers),
                "allowed_use": "spent_budget_repair_candidate_chain_planning_only",
                "forbidden_use": (
                    "score_claim_or_budget_spend_or_promotion_or_dispatch_authority"
                ),
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _repair_budget_structural_cascade_child_rows(
    *,
    chain_id: str,
    parent_candidate_chain_id: str,
    work_order: Mapping[str, Any],
    start_order: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, cascade in enumerate(
        work_order.get("repair_cascade_opportunity_rows") or [],
        start=0,
    ):
        if not isinstance(cascade, Mapping):
            continue
        cascade_id = str(cascade.get("cascade_id") or "").strip()
        if not cascade_id:
            continue
        candidate_chain_id = _bounded_content_key(
            "repair_structural_cascade_child",
            (chain_id, parent_candidate_chain_id, cascade_id),
        )
        rows.append(
            {
                "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
                "candidate_kind": "structural_repair_cascade_probe",
                "candidate_chain_id": candidate_chain_id,
                "chain_id": chain_id,
                "materialization_order": start_order + offset,
                "parent_candidate_chain_id": parent_candidate_chain_id,
                "parent_must_be_preserved_before_child": True,
                "child_must_not_replace_parent_archive": True,
                "cascade_opportunity": dict(cascade),
                "cascade_id": cascade_id,
                "cascade_label": cascade.get("label"),
                "source_relation": cascade.get("source_relation"),
                "targeted_positions": list(cascade.get("targeted_positions") or []),
                "pipeline_position": cascade.get("pipeline_position"),
                "optimization_implication": cascade.get("optimization_implication"),
                "estimate_status": cascade.get("estimate_status"),
                "empirical_feedback": dict(
                    cascade.get("empirical_feedback")
                    if isinstance(cascade.get("empirical_feedback"), Mapping)
                    else {}
                ),
                "required_empirical_landing": cascade.get("required_empirical_landing"),
                "next_queue_action": cascade.get("next_queue_action"),
                "requested_repair_bytes": None,
                "proposed_encoder_repair_bytes": 0,
                "local_lagrangian_improvement_score_units": None,
                "measured_component_delta_score_units": None,
                "measured_lagrangian_delta_score_units": None,
                "candidate_archive_materialized": False,
                "candidate_archive_path": None,
                "candidate_archive_sha256": None,
                "candidate_archive_bytes": None,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": False,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_materializer_execution": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": _unique_strings(
                    [
                        "parent_rate_only_archive_materialization_required",
                        *(_string_list(cascade.get("blockers"))),
                    ]
                ),
                "allowed_use": "structural_repair_cascade_candidate_planning_only",
                "forbidden_use": (
                    "score_claim_or_budget_spend_or_promotion_or_dispatch_authority"
                ),
                **FALSE_AUTHORITY,
            }
        )
    return rows


def build_frontier_repair_budget_materialization_plan(
    *,
    repair_budget_waterfill_work_order: Mapping[str, Any],
    repair_budget_waterfill_work_order_path: str | Path | None = None,
    repo_root: str | Path = _DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    """Turn repair-waterfill allocations into parent/child candidate custody."""

    if (
        repair_budget_waterfill_work_order.get("schema")
        != REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "repair budget waterfill work order has unexpected schema"
        )
    require_no_truthy_authority_fields(
        repair_budget_waterfill_work_order,
        context="repair_budget_materialization_plan_work_order",
    )
    chain_id = str(
        repair_budget_waterfill_work_order.get("chain_id") or "unknown_chain"
    )
    rate_budget_preservation_plan = (
        repair_budget_waterfill_work_order.get("rate_budget_preservation_plan")
        if isinstance(
            repair_budget_waterfill_work_order.get("rate_budget_preservation_plan"),
            Mapping,
        )
        else {}
    )
    operator_action_ledger = (
        repair_budget_waterfill_work_order.get("operator_action_ledger")
        if isinstance(repair_budget_waterfill_work_order.get("operator_action_ledger"), Mapping)
        else {}
    )
    parent_row = _repair_budget_rate_only_parent_row(
        chain_id=chain_id,
        rate_budget_preservation_plan=rate_budget_preservation_plan,
        work_order=repair_budget_waterfill_work_order,
        repo_root=repo_root,
    )
    child_rows = _repair_budget_spent_child_rows(
        chain_id=chain_id,
        parent_candidate_chain_id=str(parent_row.get("candidate_chain_id") or ""),
        work_order=repair_budget_waterfill_work_order,
    )
    cascade_rows = _repair_budget_structural_cascade_child_rows(
        chain_id=chain_id,
        parent_candidate_chain_id=str(parent_row.get("candidate_chain_id") or ""),
        work_order=repair_budget_waterfill_work_order,
        start_order=2 + len(child_rows),
    )
    rows = [parent_row, *child_rows, *cascade_rows]
    proposed_repair_bytes_total = sum(
        int(row.get("proposed_encoder_repair_bytes") or 0)
        for row in child_rows
    )
    blockers = [
        "candidate_archives_not_materialized",
        "receiver_runtime_consumption_proof_missing",
        "full_frame_inflate_parity_required_before_exact_readiness",
        "exact_auth_eval_required_before_score_or_promotion_claim",
    ]
    if proposed_repair_bytes_total <= 0:
        blockers.append("no_spent_budget_child_bytes_allocated")
    if not child_rows:
        blockers.append("no_spent_budget_repair_child_rows")
    payload = {
        "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
        "generated_at_utc": _utc_now(),
        "chain_id": chain_id,
        "source_work_order_path": str(repair_budget_waterfill_work_order_path or ""),
        "source_work_order_schema": repair_budget_waterfill_work_order.get("schema"),
        "candidate_chain_row_count": len(rows),
        "rate_only_parent_candidate_count": 1,
        "spent_budget_child_candidate_count": len(child_rows),
        "structural_repair_cascade_candidate_count": len(cascade_rows),
        "parent_candidate_chain_id": parent_row.get("candidate_chain_id"),
        "operator_action_ledger_schema": operator_action_ledger.get("schema"),
        "operator_action_term_count": operator_action_ledger.get("term_count"),
        "repair_allocation_action_term_schema": (
            REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
        ),
        "typed_response_ledger_schema": repair_budget_waterfill_work_order.get(
            "typed_response_ledger_schema"
        ),
        "typed_response_row_schema": repair_budget_waterfill_work_order.get(
            "typed_response_row_schema"
        ),
        "typed_response_row_count": repair_budget_waterfill_work_order.get(
            "typed_response_row_count"
        ),
        "rate_only_floor_preserved_before_repair_spend": True,
        "spent_budget_candidates_are_children_of_rate_only_floor": True,
        "rate_only_candidate_remains_valid_even_if_child_regresses": True,
        "rebrotli_default_after_rate_attack": parent_row.get(
            "rebrotli_default_after_rate_attack"
        )
        is True,
        "receiver_closed_saved_bytes_total": repair_budget_waterfill_work_order.get(
            "receiver_closed_rate_credit", {}
        ).get("receiver_closed_saved_bytes_total")
        if isinstance(
            repair_budget_waterfill_work_order.get("receiver_closed_rate_credit"),
            Mapping,
        )
        else None,
        "proposed_encoder_repair_bytes_total": proposed_repair_bytes_total,
        "candidate_archive_materialized": False,
        "runtime_consumption_proof_present": False,
        "receiver_consumed": False,
        "component_response_replayed": False,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_materializer_execution": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_chain_rows": rows,
        "blockers": _unique_strings(blockers),
        "recommended_next_action": (
            "materialize_rate_only_parent_archive_then_spent_budget_child_archives_"
            "under_receiver_runtime_consumption_and_component_replay_gates"
            if child_rows and proposed_repair_bytes_total > 0
            else "collect_allocated_repair_bytes_before_candidate_archive_materialization"
        ),
        "allowed_use": "repair_budget_candidate_chain_materialization_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context=f"frontier_repair_budget_materialization_plan:{chain_id}",
    )
    return payload


def _repair_child_response_row(
    *,
    child_row: Mapping[str, Any],
    response_harvest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    rows = [
        row
        for row in response_harvest.get("rows") or []
        if isinstance(row, Mapping)
    ]
    candidate_id = str(child_row.get("allocation_candidate_id") or "").strip()
    correction_family = str(child_row.get("correction_family") or "").strip()
    for row in rows:
        if (
            candidate_id
            and str(row.get("candidate_id") or "").strip() != candidate_id
        ):
            continue
        if (
            correction_family
            and str(row.get("correction_family") or "").strip()
            != correction_family
        ):
            continue
        return row
    return rows[0] if rows else None


def _repair_child_component_replay_manifest(
    *,
    child_row: Mapping[str, Any],
    materialization_plan: Mapping[str, Any],
    materialization_plan_path: str | Path | None,
    response_harvest_path: str | Path,
    response_harvest: Mapping[str, Any],
    repo_root: str | Path = _DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    require_no_truthy_authority_fields(
        response_harvest,
        context=f"repair_child_component_replay_response:{response_harvest_path}",
    )
    response_row = _repair_child_response_row(
        child_row=child_row,
        response_harvest=response_harvest,
    )
    blockers = [
        "exact_auth_eval_required_before_score_or_promotion_claim",
        "component_response_replay_is_local_materialization_signal_only",
    ]
    if response_row is None:
        blockers.append("component_response_harvest_row_missing")
        response_row = {}
    response_blockers = _string_list(response_row.get("budget_spend_blockers"))
    local_mlx_response_path = str(
        response_row.get("local_mlx_response_path") or ""
    ).strip()
    reference_local_mlx_response_path = str(
        response_row.get("reference_local_mlx_response_path") or ""
    ).strip()
    axis_tag = str(response_row.get("local_mlx_score_axis") or "").strip()
    if not local_mlx_response_path:
        blockers.append("local_mlx_component_response_path_missing")
    if axis_tag != "[macOS-MLX research-signal]":
        blockers.append("local_mlx_component_response_axis_not_research_signal")
    archive_path = str(child_row.get("candidate_archive_path") or "").strip()
    archive_sha = str(child_row.get("candidate_archive_sha256") or "").strip()
    runtime_proof_path = str(
        child_row.get("runtime_consumption_proof_path") or ""
    ).strip()
    proof_revalidation = _runtime_consumption_proof_revalidation(
        proof_path_text=runtime_proof_path,
        repo_root=repo_root,
        expected_candidate_archive_sha256=archive_sha,
        context="child_runtime_consumption_proof",
    )
    receiver_proof_present = proof_revalidation["proof_valid"] is True
    receiver_consumed = bool(
        child_row.get("receiver_consumed") is True
        and receiver_proof_present
        and proof_revalidation["receiver_contract_satisfied"] is True
    )
    archive_materialized = (
        child_row.get("candidate_archive_materialized") is True
        and bool(archive_path)
        and bool(archive_sha)
    )
    if not archive_materialized:
        blockers.append("repair_candidate_archive_materialization_missing")
    blockers.extend(_string_list(proof_revalidation.get("blockers")))
    if not receiver_proof_present or not receiver_consumed:
        blockers.append("receiver_runtime_consumption_proof_missing")
    replayed = bool(local_mlx_response_path and axis_tag == "[macOS-MLX research-signal]")
    return {
        "schema": REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFEST_SCHEMA,
        "manifest_kind": "repair_budget_child_component_replay_evidence",
        "candidate_chain_id": child_row.get("candidate_chain_id"),
        "candidate_chain_ids": [child_row.get("candidate_chain_id")],
        "repair_budget_candidate_chain_id": child_row.get("candidate_chain_id"),
        "candidate_kind": child_row.get("candidate_kind"),
        "chain_id": child_row.get("chain_id"),
        "parent_candidate_chain_id": child_row.get("parent_candidate_chain_id"),
        "materialization_order": child_row.get("materialization_order"),
        "allocation_candidate_id": child_row.get("allocation_candidate_id"),
        "acquisition_id": child_row.get("acquisition_id"),
        "correction_family": child_row.get("correction_family"),
        "target_kind": (
            child_row.get("correction_family")
            or "repair_budget_child_component_replay"
        ),
        "source_materialization_plan_path": (
            None if materialization_plan_path is None else str(materialization_plan_path)
        ),
        "source_materialization_plan_schema": materialization_plan.get("schema"),
        "source_response_artifact_path": str(response_harvest_path),
        "source_response_artifact_schema": response_harvest.get("schema"),
        "source_response_row_schema": response_row.get("schema"),
        "byte_closed_candidate_emitted": archive_materialized,
        "candidate_archive": {
            "path": archive_path or None,
            "sha256": archive_sha or None,
            "bytes": child_row.get("candidate_archive_bytes"),
        },
        "runtime_consumption_proof_path": runtime_proof_path or None,
        "receiver_contract_satisfied": receiver_consumed,
        "receiver_verification": {
            "schema": "repair_budget_child_receiver_verification_from_plan.v1",
            "proof_path": runtime_proof_path or None,
            "proof_present": receiver_proof_present,
            "receiver_contract_satisfied": receiver_consumed,
            "runtime_consumption_proof_passed": receiver_consumed,
            "proof_revalidation": proof_revalidation,
            **FALSE_AUTHORITY,
        },
        "component_response_replayed": replayed,
        "component_response_replay": {
            "schema": "repair_budget_child_component_response_replay.v1",
            "replayed": replayed,
            "artifact_path": str(response_harvest_path),
            "local_mlx_response_path": local_mlx_response_path or None,
            "reference_local_mlx_response_path": (
                reference_local_mlx_response_path or None
            ),
            "axis_tag": axis_tag or None,
            "evidence_grade": "local_mlx_component_response_replay_only",
            "measured_component_delta_score_units": response_row.get(
                "measured_component_delta_score_units"
            ),
            "measured_lagrangian_delta_score_units": response_row.get(
                "measured_lagrangian_delta_score_units"
            ),
            "negative_measured_lagrangian_delta": (
                response_row.get("negative_measured_lagrangian_delta") is True
            ),
            "budget_spend_blockers": response_blockers,
            "blockers": _unique_strings(blockers),
            **FALSE_AUTHORITY,
        },
        "readiness_blockers": _unique_strings(
            [
                *blockers,
                *response_blockers,
            ]
        ),
        "allowed_use": (
            "repair_budget_child_component_replay_manifest_for_local_binding_only"
        ),
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def build_frontier_repair_budget_child_component_replay_manifests(
    *,
    repair_budget_materialization_plan: Mapping[str, Any],
    repair_budget_materialization_plan_path: str | Path | None = None,
    response_harvests_by_path: Mapping[str | Path, Mapping[str, Any]] | None = None,
    candidate_chain_ids: Sequence[str] = (),
    repo_root: str | Path = _DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    """Emit replay-evidence manifests for repair-budget child plan rows."""

    if (
        repair_budget_materialization_plan.get("schema")
        != REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "repair budget materialization plan has unexpected schema"
        )
    require_no_truthy_authority_fields(
        repair_budget_materialization_plan,
        context="repair_budget_child_component_replay_manifest_plan",
    )
    response_harvests_by_path = response_harvests_by_path or {}
    selected_ids = {str(value).strip() for value in candidate_chain_ids if str(value).strip()}
    manifests: list[dict[str, Any]] = []
    blockers = [
        "exact_auth_eval_required_before_score_or_promotion_claim",
        "child_archive_materialization_required_before_execution_ready",
    ]
    for row in repair_budget_materialization_plan.get("candidate_chain_rows") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("candidate_kind") != "spent_budget_repair_child":
            continue
        candidate_chain_id = str(row.get("candidate_chain_id") or "").strip()
        if selected_ids and candidate_chain_id not in selected_ids:
            continue
        response_path = str(row.get("source_response_artifact_path") or "").strip()
        if not response_path:
            blockers.append(f"source_response_artifact_path_missing:{candidate_chain_id}")
            continue
        response_harvest = response_harvests_by_path.get(response_path)
        if response_harvest is None:
            response_harvest = response_harvests_by_path.get(Path(response_path))
        if response_harvest is None:
            blockers.append(f"source_response_artifact_missing:{response_path}")
            continue
        manifests.append(
            _repair_child_component_replay_manifest(
                child_row=row,
                materialization_plan=repair_budget_materialization_plan,
                materialization_plan_path=repair_budget_materialization_plan_path,
                response_harvest_path=response_path,
                response_harvest=response_harvest,
                repo_root=repo_root,
            )
        )
    if not manifests:
        blockers.append("repair_child_component_replay_manifests_missing")
    payload = {
        "schema": REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFESTS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_materialization_plan_path": str(
            repair_budget_materialization_plan_path or ""
        ),
        "source_materialization_plan_schema": repair_budget_materialization_plan.get(
            "schema"
        ),
        "chain_id": repair_budget_materialization_plan.get("chain_id"),
        "parent_candidate_chain_id": repair_budget_materialization_plan.get(
            "parent_candidate_chain_id"
        ),
        "manifest_count": len(manifests),
        "component_response_replayed_count": sum(
            1
            for manifest in manifests
            if manifest.get("component_response_replayed") is True
        ),
        "byte_closed_candidate_emitted_count": sum(
            1
            for manifest in manifests
            if manifest.get("byte_closed_candidate_emitted") is True
        ),
        "candidate_chain_ids": _unique_strings(
            manifest.get("candidate_chain_id") for manifest in manifests
        ),
        "manifests": manifests,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": _unique_strings(
            [
                *blockers,
                *[
                    blocker
                    for manifest in manifests
                    for blocker in _string_list(manifest.get("readiness_blockers"))
                ],
            ]
        ),
        "allowed_use": (
            "repair_budget_child_component_replay_manifest_collection_for_binding_only"
        ),
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="frontier_repair_budget_child_component_replay_manifests",
    )
    return payload


def _materializer_manifest_candidate_chain_ids(
    manifest: Mapping[str, Any],
) -> list[str]:
    ids: list[str] = []
    for key in (
        "repair_budget_candidate_chain_id",
        "candidate_chain_id",
        "candidate_id",
    ):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            ids.append(value.strip())
    for key in (
        "repair_budget_candidate_chain_ids",
        "candidate_chain_ids",
        "candidate_ids",
    ):
        ids.extend(_string_list(manifest.get(key)))
    return _unique_strings(ids)


def _palette_modes_from_mapping(payload: Mapping[str, Any]) -> list[str]:
    modes: list[str] = []
    for key in (
        "selector_palette",
        "palette",
        "mode_palette",
        "palette_modes",
        "canonical_palette",
        "repair_palette_modes",
    ):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            modes.extend(_string_list(value))
    for key in (
        "archive_manifest",
        "packet_manifest",
        "selector_manifest",
        "repair_dynamics_prior",
        "selector",
        "fec6_selector",
    ):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            modes.extend(_palette_modes_from_mapping(nested))
    return _unique_strings(modes)


def _palette_mode_frame_index(mode: str) -> int | None:
    if not mode.startswith("frame"):
        return None
    suffix = mode[len("frame") :]
    number = suffix.split("_", 1)[0]
    return int(number) if number.isdigit() else None


def _palette_mode_family(mode: str) -> str:
    if mode == "none":
        return "identity"
    if "blue_chroma" in mode:
        return "blue_chroma"
    if "luma_bias" in mode:
        return "luma_bias"
    if "rgb_bias" in mode:
        return "rgb_bias"
    if "roll_" in mode:
        return "geometry_roll"
    return "other"


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _repair_dynamics_palette_prior(
    modes: Sequence[str],
    *,
    source: str,
) -> dict[str, Any]:
    palette_modes = _unique_strings(modes)
    if not palette_modes:
        return {}
    frame_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for mode in palette_modes:
        frame_index = _palette_mode_frame_index(mode)
        frame_key = f"frame{frame_index}" if frame_index is not None else "no_frame"
        _increment_count(frame_counts, frame_key)
        _increment_count(family_counts, _palette_mode_family(mode))
    total = len(palette_modes)
    identity_count = family_counts.get("identity", 0)
    non_identity_total = max(0, total - identity_count)
    frame0_count = frame_counts.get("frame0", 0)
    frame1_count = frame_counts.get("frame1", 0)
    color_family_count = sum(
        family_counts.get(key, 0)
        for key in ("blue_chroma", "luma_bias", "rgb_bias")
    )
    hints: list[str] = []
    if frame0_count:
        hints.append("frame0_palette_modes_are_first_class_repair_operators")
    if non_identity_total and frame0_count == non_identity_total:
        hints.append("empirical_non_identity_palette_is_all_frame0")
    if frame1_count == 0:
        hints.append("do_not_assume_frame1_direct_repair_mode_exists")
    if color_family_count:
        hints.append("prioritize_global_chroma_luma_rgb_bias_repair_before_pixel_leaf_search")
    if family_counts.get("geometry_roll", 0):
        hints.append("keep_frame0_roll_as_geometry_interaction_term")
    return {
        "schema": REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA,
        "source": source,
        "palette_modes": palette_modes,
        "mode_count": total,
        "identity_mode_count": identity_count,
        "non_identity_mode_count": non_identity_total,
        "frame_mode_counts": dict(sorted(frame_counts.items())),
        "mode_family_counts": dict(sorted(family_counts.items())),
        "frame0_mode_count": frame0_count,
        "frame1_mode_count": frame1_count,
        "frame0_mode_fraction": frame0_count / total,
        "frame0_non_identity_fraction": (
            frame0_count / non_identity_total if non_identity_total else 0.0
        ),
        "zero_frame1_modes": frame1_count == 0,
        "dominant_dynamics_interpretation": (
            "frame0_global_color_geometry_calibration_prior"
            if frame0_count and frame1_count == 0
            else "mixed_or_unclassified_palette_prior"
        ),
        "repair_waterfill_hints": _unique_strings(hints),
        "action_functional_implications": [
            "treat_frame0_palette_repairs_as_global_interaction_terms",
            "remeasure_parent_child_synergy_before_budget_spend",
            "do_not_rank_frame1_repairs_without_empirical_component_response",
        ],
        "budget_spend_allowed": False,
        "allowed_use": "repair_dynamics_prior_for_local_waterfill_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _aggregate_repair_dynamics_priors(
    priors: Sequence[Mapping[str, Any]],
    *,
    source: str = "repair_budget_materializer_binding_report_aggregate",
) -> dict[str, Any]:
    modes: list[str] = []
    sources: list[str] = []
    for prior in priors:
        if not isinstance(prior, Mapping):
            continue
        modes.extend(_string_list(prior.get("palette_modes")))
        source = str(prior.get("source") or "").strip()
        if source:
            sources.append(source)
    aggregate = _repair_dynamics_palette_prior(
        modes,
        source=source,
    )
    if aggregate:
        aggregate["source_prior_count"] = len(priors)
        aggregate["source_prior_refs"] = _unique_strings(sources)
    return aggregate


def _first_nonempty_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _component_response_replay_manifest_record(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    replay = (
        manifest.get("component_response_replay")
        if isinstance(manifest.get("component_response_replay"), Mapping)
        else {}
    )
    replayed_claim = (
        manifest.get("component_response_replayed") is True
        or replay.get("component_response_replayed") is True
        or replay.get("response_replayed") is True
        or replay.get("replayed") is True
    )
    replay_path = _first_nonempty_text(
        manifest.get("component_response_replay_path"),
        manifest.get("component_response_replay_artifact_path"),
        replay.get("path"),
        replay.get("artifact_path"),
        replay.get("report_path"),
        replay.get("proof_path"),
        replay.get("source_response_artifact_path"),
        replay.get("candidate_response_artifact_path"),
    )
    axis_tag = _first_nonempty_text(
        manifest.get("component_response_replay_axis_tag"),
        replay.get("axis_tag"),
        replay.get("evidence_axis"),
        replay.get("axis"),
    )
    evidence_grade = _first_nonempty_text(
        manifest.get("component_response_replay_evidence_grade"),
        replay.get("evidence_grade"),
        replay.get("authority_grade"),
    )
    blockers = [
        *_string_list(manifest.get("component_response_replay_blockers")),
        *_string_list(replay.get("blockers")),
    ]
    if replayed_claim and not replay_path:
        blockers.append("component_response_replay_artifact_path_missing")
    if replayed_claim and not axis_tag:
        blockers.append("component_response_replay_axis_tag_missing")
    return {
        "component_response_replayed": bool(replayed_claim and replay_path),
        "component_response_replay_path": replay_path or None,
        "component_response_replay_axis_tag": axis_tag or None,
        "component_response_replay_evidence_grade": evidence_grade or None,
        "component_response_replay_blockers": _unique_strings(blockers),
    }


def _materializer_manifest_record(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    require_no_truthy_authority_fields(
        manifest,
        context=f"repair_budget_materializer_manifest:{manifest_path or '<inline>'}",
    )
    candidate_archive = (
        manifest.get("candidate_archive")
        if isinstance(manifest.get("candidate_archive"), Mapping)
        else {}
    )
    source_archive = (
        manifest.get("source_archive")
        if isinstance(manifest.get("source_archive"), Mapping)
        else {}
    )
    receiver_verification = (
        manifest.get("receiver_verification")
        if isinstance(manifest.get("receiver_verification"), Mapping)
        else {}
    )
    candidate_archive_path = str(candidate_archive.get("path") or "").strip()
    candidate_archive_sha = str(candidate_archive.get("sha256") or "").strip()
    candidate_archive_bytes = candidate_archive.get("bytes")
    runtime_proof_path = str(
        manifest.get("runtime_consumption_proof_path")
        or receiver_verification.get("proof_path")
        or ""
    ).strip()
    receiver_contract_satisfied = (
        manifest.get("receiver_contract_satisfied") is True
        or receiver_verification.get("receiver_contract_satisfied") is True
    )
    byte_closed = manifest.get("byte_closed_candidate_emitted") is True
    selector_palette_modes = _palette_modes_from_mapping(manifest)
    repair_dynamics_prior = _repair_dynamics_palette_prior(
        selector_palette_modes,
        source=str(manifest_path or manifest.get("archive_sha256") or "inline_manifest"),
    )
    component_replay = _component_response_replay_manifest_record(manifest)
    readiness_blockers = _unique_strings(
        [
            *_string_list(manifest.get("readiness_blockers")),
            *_string_list(component_replay.get("component_response_replay_blockers")),
        ]
    )
    candidate_archive_verified = False
    proof_blockers: list[str] = []
    if byte_closed:
        if not candidate_archive_path:
            readiness_blockers.append("candidate_archive_path_missing")
        elif not candidate_archive_sha:
            readiness_blockers.append("candidate_archive_sha256_missing")
        else:
            archive_path = _resolve_path(candidate_archive_path, repo_root=repo_root)
            if not archive_path.is_file():
                readiness_blockers.append("candidate_archive_file_missing")
            else:
                actual_sha = _sha256_file(archive_path)
                if actual_sha != candidate_archive_sha:
                    readiness_blockers.append("candidate_archive_file_sha256_mismatch")
                elif (
                    isinstance(candidate_archive_bytes, int)
                    and not isinstance(candidate_archive_bytes, bool)
                    and archive_path.stat().st_size != candidate_archive_bytes
                ):
                    readiness_blockers.append("candidate_archive_file_bytes_mismatch")
                else:
                    candidate_archive_verified = True
    proof_revalidation = _runtime_consumption_proof_revalidation(
        proof_path_text=runtime_proof_path,
        repo_root=repo_root,
        expected_candidate_archive_sha256=candidate_archive_sha,
        context="runtime_consumption_proof",
    )
    proof_blockers.extend(_string_list(proof_revalidation.get("blockers")))
    receiver_blockers = [
        f"receiver_verification:{blocker}"
        for blocker in _string_list(receiver_verification.get("blockers"))
    ]
    readiness_blockers.extend(proof_blockers)
    readiness_blockers.extend(receiver_blockers)
    readiness_blockers = _unique_strings(readiness_blockers)
    runtime_proof_present = proof_revalidation["proof_valid"] is True
    receiver_consumed = bool(
        runtime_proof_present
        and receiver_contract_satisfied
        and proof_revalidation["receiver_contract_satisfied"] is True
        and not proof_blockers
        and not receiver_blockers
    )
    return {
        "schema": "frontier_rate_attack_materializer_manifest_binding_input.v1",
        "manifest_path": str(manifest_path or ""),
        "manifest_schema": manifest.get("schema"),
        "materializer_id": manifest.get("materializer_id"),
        "target_kind": manifest.get("target_kind"),
        "candidate_chain_ids": _materializer_manifest_candidate_chain_ids(manifest),
        "byte_closed_candidate_emitted": byte_closed,
        "candidate_archive_path": candidate_archive_path or None,
        "candidate_archive_sha256": candidate_archive_sha or None,
        "candidate_archive_bytes": candidate_archive_bytes,
        "source_archive_path": source_archive.get("path"),
        "source_archive_sha256": source_archive.get("sha256"),
        "runtime_consumption_proof_path": runtime_proof_path or None,
        "runtime_consumption_proof_present": runtime_proof_present,
        "runtime_consumption_proof_revalidation": proof_revalidation,
        "receiver_contract_kind": manifest.get("receiver_contract_kind"),
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "runtime_adapter_ready": manifest.get("runtime_adapter_ready") is True,
        "component_response_replayed": component_replay[
            "component_response_replayed"
        ],
        "component_response_replay_path": component_replay[
            "component_response_replay_path"
        ],
        "component_response_replay_axis_tag": component_replay[
            "component_response_replay_axis_tag"
        ],
        "component_response_replay_evidence_grade": component_replay[
            "component_response_replay_evidence_grade"
        ],
        "readiness_blockers": readiness_blockers,
        "selector_palette_modes": selector_palette_modes,
        "repair_dynamics_prior": repair_dynamics_prior,
        "candidate_archive_materialized": bool(byte_closed and candidate_archive_verified),
        "receiver_consumed": receiver_consumed,
        **FALSE_AUTHORITY,
    }


def _load_materializer_manifest_records(
    *,
    repo_root: str | Path,
    materializer_manifests: Sequence[Mapping[str, Any]] = (),
    materializer_manifest_paths: Sequence[str | Path] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    repo = Path(repo_root)
    records: list[dict[str, Any]] = []
    blockers: list[str] = []

    def append_manifest_records(
        manifest: Mapping[str, Any],
        *,
        manifest_path: str | Path | None,
    ) -> None:
        if manifest.get("schema") == REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFESTS_SCHEMA:
            require_no_truthy_authority_fields(
                manifest,
                context=(
                    "repair_budget_child_component_replay_manifest_collection:"
                    f"{manifest_path or '<inline>'}"
                ),
            )
            for index, child_manifest in enumerate(manifest.get("manifests") or []):
                if not isinstance(child_manifest, Mapping):
                    blockers.append(
                        f"repair_child_component_replay_manifest_not_object:{index}"
                    )
                    continue
                child_path = (
                    f"{manifest_path}#manifests/{index}"
                    if manifest_path is not None
                    else None
                )
                records.append(
                    _materializer_manifest_record(
                        manifest=child_manifest,
                        manifest_path=child_path,
                        repo_root=repo,
                    )
                )
            return
        records.append(
            _materializer_manifest_record(
                manifest=manifest,
                manifest_path=manifest_path,
                repo_root=repo,
            )
        )

    for manifest in materializer_manifests:
        append_manifest_records(manifest, manifest_path=None)
    for raw_path in materializer_manifest_paths:
        path = _resolve_path(raw_path, repo_root=repo)
        if not path.is_file():
            blockers.append(f"materializer_manifest_missing:{raw_path}")
            continue
        payload = _load_json(path)
        append_manifest_records(payload, manifest_path=_repo_rel(path, repo))
    return records, _unique_strings(blockers)


def _is_expected_materializer_manifest_path(path: str) -> bool:
    name = Path(path).name
    return name in {"manifest.json", "archive_manifest.json"} or name.endswith(
        "_manifest.json"
    )


def _expected_materializer_manifest_paths_from_queue(
    queue: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(queue, Mapping):
        return []
    paths: list[str] = []
    def append_command_manifest_paths(command: object) -> None:
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            return
        for index, item in enumerate(command[:-1]):
            if item == "--output-manifest":
                paths.append(str(command[index + 1]))

    def append_postcondition_manifest_path(condition: object) -> None:
        if not isinstance(condition, Mapping):
            return
        path = condition.get("path")
        if (
            isinstance(path, str)
            and path.strip()
            and _is_expected_materializer_manifest_path(path.strip())
        ):
            paths.append(path.strip())

    for row in queue.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        append_command_manifest_paths(row.get("command"))
        for condition in row.get("postconditions") or []:
            append_postcondition_manifest_path(condition)
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        for step in experiment.get("steps") or []:
            if not isinstance(step, Mapping):
                continue
            append_command_manifest_paths(step.get("command"))
            for condition in step.get("postconditions") or []:
                append_postcondition_manifest_path(condition)
    return _unique_strings(paths)


def _materializer_work_queue_summary(
    queue: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(queue, Mapping):
        return {
            "present": False,
            "schema": None,
            "row_count": 0,
            "executable_row_count": 0,
            "expected_manifest_paths": [],
            **FALSE_AUTHORITY,
        }
    return {
        "present": True,
        "schema": queue.get("schema"),
        "row_count": queue.get("row_count"),
        "executable_row_count": queue.get("executable_row_count"),
        "blocked_row_count": queue.get("blocked_row_count"),
        "expected_manifest_paths": _expected_materializer_manifest_paths_from_queue(queue),
        **FALSE_AUTHORITY,
    }


def _direct_manifest_ready(record: Mapping[str, Any]) -> bool:
    return (
        record.get("candidate_archive_materialized") is True
        and record.get("runtime_consumption_proof_present") is True
        and record.get("receiver_consumed") is True
    )


def _repair_budget_materializer_binding_row(
    *,
    plan_row: Mapping[str, Any],
    manifest_records: Sequence[Mapping[str, Any]],
    parent_candidate_chain_id: str,
    bound_parent_ids: set[str],
) -> dict[str, Any]:
    candidate_chain_id = str(plan_row.get("candidate_chain_id") or "")
    candidate_kind = str(plan_row.get("candidate_kind") or "unknown_candidate_kind")
    operator_terms = [
        term
        for term in plan_row.get("operator_action_terms") or []
        if isinstance(term, Mapping)
    ]
    required_term_ids = _unique_strings(
        term.get("operator_action_id") for term in operator_terms
    )
    required_target_kinds = _unique_strings(
        (
            term.get("target_kind")
            or (
                term.get("T_i", {}).get("target_kind")
                if isinstance(term.get("T_i"), Mapping)
                else None
            )
        )
        for term in operator_terms
    )
    direct_matches = [
        record
        for record in manifest_records
        if candidate_chain_id
        and candidate_chain_id in set(record.get("candidate_chain_ids") or [])
    ]
    required_target_kind_set = set(required_target_kinds)
    coverage_matches = [
        record
        for record in manifest_records
        if str(record.get("target_kind") or "") in required_target_kind_set
    ]
    direct_ready = [record for record in direct_matches if _direct_manifest_ready(record)]
    direct_component_replay = [
        record
        for record in direct_matches
        if record.get("component_response_replayed") is True
    ]
    plan_row_ready = (
        candidate_kind == "rate_only_floor_parent"
        and plan_row.get("candidate_archive_materialized") is True
        and plan_row.get("receiver_consumed") is True
    )
    selected = direct_ready[0] if direct_ready else {}
    replay_selected = (
        direct_component_replay[0]
        if direct_component_replay
        else selected
        if selected.get("component_response_replayed") is True
        else {}
    )
    if not selected and plan_row_ready:
        selected = {
            "schema": "frontier_rate_attack_plan_row_receiver_closed_binding.v1",
            "manifest_path": plan_row.get("source_work_order_path") or "",
            "candidate_archive_path": plan_row.get("candidate_archive_path"),
            "candidate_archive_sha256": plan_row.get("candidate_archive_sha256"),
            "candidate_archive_bytes": plan_row.get("candidate_archive_bytes"),
            "runtime_consumption_proof_path": plan_row.get(
                "runtime_consumption_proof_path"
            ),
            "runtime_consumption_proof_present": plan_row.get(
                "runtime_consumption_proof_present"
            )
            is True,
            "runtime_consumption_proof_revalidation": plan_row.get(
                "runtime_consumption_proof_revalidation"
            ),
            "receiver_consumed": True,
            "repair_dynamics_prior": {},
            **FALSE_AUTHORITY,
        }
    blockers: list[str] = []
    if not candidate_chain_id:
        blockers.append("candidate_chain_id_missing")
    if candidate_kind != "rate_only_floor_parent":
        if str(plan_row.get("parent_candidate_chain_id") or "") != parent_candidate_chain_id:
            blockers.append("child_parent_candidate_chain_id_mismatch")
        if parent_candidate_chain_id not in bound_parent_ids:
            blockers.append("parent_rate_only_archive_materialization_required")
    if not direct_ready and not plan_row_ready and direct_matches:
        blockers.append("direct_materializer_manifest_not_receiver_consumed")
    elif not direct_ready and not plan_row_ready and coverage_matches:
        blockers.append("individual_materializer_manifests_not_composed_single_candidate_archive")
    elif not direct_ready and not plan_row_ready:
        blockers.append("candidate_chain_materializer_manifest_missing")
    for record in direct_matches:
        blockers.extend(_string_list(record.get("readiness_blockers")))
    candidate_archive_materialized = bool(selected)
    component_response_replayed = (
        plan_row.get("component_response_replayed") is True
        or candidate_kind == "rate_only_floor_parent"
        or selected.get("component_response_replayed") is True
        or replay_selected.get("component_response_replayed") is True
    )
    runtime_proof_revalidation = (
        selected.get("runtime_consumption_proof_revalidation")
        or plan_row.get("runtime_consumption_proof_revalidation")
    )
    runtime_proof_present = bool(
        isinstance(runtime_proof_revalidation, Mapping)
        and runtime_proof_revalidation.get("proof_valid") is True
    )
    receiver_consumed = bool(
        selected.get("receiver_consumed") is True
        and runtime_proof_present
        and (
            not isinstance(runtime_proof_revalidation, Mapping)
            or runtime_proof_revalidation.get("receiver_contract_satisfied") is True
        )
    )
    if selected and not runtime_proof_present:
        blockers.append("runtime_consumption_proof_revalidation_missing_or_invalid")
    if (
        candidate_kind == "rate_only_floor_parent"
        and candidate_archive_materialized
        and receiver_consumed
    ):
        bound_parent_ids.add(candidate_chain_id)
    return {
        "schema": REPAIR_BUDGET_MATERIALIZER_BINDING_ROW_SCHEMA,
        "candidate_chain_id": candidate_chain_id,
        "candidate_kind": candidate_kind,
        "chain_id": plan_row.get("chain_id"),
        "materialization_order": plan_row.get("materialization_order"),
        "parent_candidate_chain_id": plan_row.get("parent_candidate_chain_id"),
        "direct_manifest_count": len(direct_matches),
        "plan_row_receiver_closed_binding": plan_row_ready,
        "coverage_manifest_count": len(coverage_matches),
        "required_operator_action_term_count": len(required_term_ids),
        "covered_operator_action_term_count": (
            sum(
                1
                for term in operator_terms
                if str(
                    term.get("target_kind")
                    or (
                        term.get("T_i", {}).get("target_kind")
                        if isinstance(term.get("T_i"), Mapping)
                        else ""
                    )
                )
                in {
                    str(record.get("target_kind") or "")
                    for record in coverage_matches
                }
            )
            if coverage_matches and not direct_ready
            else 0
        ),
        "required_operator_action_ids": required_term_ids,
        "required_target_kinds": required_target_kinds,
        "direct_manifest_paths": _unique_strings(
            record.get("manifest_path") for record in direct_matches
        ),
        "coverage_manifest_paths": _unique_strings(
            record.get("manifest_path") for record in coverage_matches
        ),
        "component_replay_manifest_paths": _unique_strings(
            record.get("manifest_path") for record in direct_component_replay
        ),
        "candidate_archive_path": selected.get("candidate_archive_path"),
        "candidate_archive_sha256": selected.get("candidate_archive_sha256"),
        "candidate_archive_bytes": selected.get("candidate_archive_bytes"),
        "runtime_consumption_proof_path": selected.get(
            "runtime_consumption_proof_path"
        ),
        "runtime_consumption_proof_revalidation": runtime_proof_revalidation,
        "repair_dynamics_prior": dict(
            selected.get("repair_dynamics_prior")
            if isinstance(selected.get("repair_dynamics_prior"), Mapping)
            else {}
        ),
        "runtime_consumption_proof_present": runtime_proof_present,
        "receiver_consumed": receiver_consumed,
        "component_response_replayed": component_response_replayed,
        "component_response_replay_path": (
            selected.get("component_response_replay_path")
            or replay_selected.get("component_response_replay_path")
            or plan_row.get("component_response_replay_path")
        ),
        "component_response_replay_axis_tag": (
            selected.get("component_response_replay_axis_tag")
            or replay_selected.get("component_response_replay_axis_tag")
            or plan_row.get("component_response_replay_axis_tag")
        ),
        "component_response_replay_evidence_grade": (
            selected.get("component_response_replay_evidence_grade")
            or replay_selected.get("component_response_replay_evidence_grade")
            or plan_row.get("component_response_replay_evidence_grade")
        ),
        "candidate_archive_materialized": candidate_archive_materialized,
        "ready_for_materialization_execution_audit": candidate_archive_materialized
        and receiver_consumed,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": _unique_strings(blockers),
        "allowed_use": "repair_budget_materializer_binding_row_for_execution_audit",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def build_frontier_repair_budget_materializer_binding_report(
    *,
    repo_root: str | Path,
    repair_budget_materialization_plan: Mapping[str, Any],
    repair_budget_materialization_plan_path: str | Path | None = None,
    materializer_work_queue: Mapping[str, Any] | None = None,
    materializer_work_queue_path: str | Path | None = None,
    materializer_execution_queue: Mapping[str, Any] | None = None,
    materializer_execution_queue_path: str | Path | None = None,
    materializer_manifests: Sequence[Mapping[str, Any]] = (),
    materializer_manifest_paths: Sequence[str | Path] = (),
    repair_palette_modes: Sequence[str] = (),
) -> dict[str, Any]:
    """Bind materializer outputs to repair-budget parent/child plan rows.

    A leaf materializer manifest is not enough to promote a cumulative
    rate-only parent. Parent/child rows become materialized only when a manifest
    explicitly binds to that candidate chain id and carries receiver proof.
    """

    if (
        repair_budget_materialization_plan.get("schema")
        != REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "repair budget materialization plan has unexpected schema"
        )
    require_no_truthy_authority_fields(
        repair_budget_materialization_plan,
        context="repair_budget_materializer_binding_plan",
    )
    repo = Path(repo_root)
    expected_paths = _unique_strings(
        [
            *_expected_materializer_manifest_paths_from_queue(materializer_work_queue),
            *_expected_materializer_manifest_paths_from_queue(
                materializer_execution_queue
            ),
        ]
    )
    manifest_records, manifest_blockers = _load_materializer_manifest_records(
        repo_root=repo,
        materializer_manifests=materializer_manifests,
        materializer_manifest_paths=(
            *tuple(materializer_manifest_paths),
            *tuple(expected_paths),
        ),
    )
    repair_dynamics_priors: list[Mapping[str, Any]] = []
    manual_prior = _repair_dynamics_palette_prior(
        repair_palette_modes,
        source="operator_supplied_repair_palette_modes",
    )
    if manual_prior:
        repair_dynamics_priors.append(manual_prior)
    for record in manifest_records:
        prior = record.get("repair_dynamics_prior")
        if isinstance(prior, Mapping) and prior:
            repair_dynamics_priors.append(prior)
    aggregate_repair_dynamics_prior = _aggregate_repair_dynamics_priors(
        repair_dynamics_priors
    )
    parent_candidate_chain_id = str(
        repair_budget_materialization_plan.get("parent_candidate_chain_id") or ""
    )
    sorted_rows = sorted(
        [
            row
            for row in repair_budget_materialization_plan.get("candidate_chain_rows") or []
            if isinstance(row, Mapping)
        ],
        key=lambda row: (
            _finite_int_or_none(row.get("materialization_order")) or 10**9,
            str(row.get("candidate_chain_id") or ""),
        ),
    )
    bound_parent_ids: set[str] = set()
    binding_rows = [
        _repair_budget_materializer_binding_row(
            plan_row=row,
            manifest_records=manifest_records,
            parent_candidate_chain_id=parent_candidate_chain_id,
            bound_parent_ids=bound_parent_ids,
        )
        for row in sorted_rows
    ]
    bound_count = sum(
        1 for row in binding_rows if row.get("candidate_archive_materialized") is True
    )
    blockers = [
        "exact_auth_eval_required_before_score_or_promotion_claim",
        *manifest_blockers,
        *[
            blocker
            for row in binding_rows
            for blocker in row.get("blockers") or []
        ],
    ]
    if not binding_rows:
        blockers.append("candidate_chain_rows_missing")
    if bound_count < len(binding_rows):
        blockers.append("not_all_candidate_chain_rows_bound_to_receiver_consumed_manifests")
    payload = {
        "schema": REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA,
        "generated_at_utc": _utc_now(),
        "chain_id": repair_budget_materialization_plan.get("chain_id"),
        "source_materialization_plan_path": str(
            repair_budget_materialization_plan_path or ""
        ),
        "source_materialization_plan_schema": repair_budget_materialization_plan.get(
            "schema"
        ),
        "materializer_work_queue_path": str(materializer_work_queue_path or ""),
        "materializer_execution_queue_path": str(
            materializer_execution_queue_path or ""
        ),
        "materializer_work_queue_summary": _materializer_work_queue_summary(
            materializer_work_queue
        ),
        "materializer_execution_queue_summary": _materializer_work_queue_summary(
            materializer_execution_queue
        ),
        "materializer_manifest_count": len(manifest_records),
        "materializer_manifest_paths": _unique_strings(
            record.get("manifest_path") for record in manifest_records
        ),
        "repair_dynamics_prior_count": len(repair_dynamics_priors),
        "repair_dynamics_palette_prior": aggregate_repair_dynamics_prior,
        "candidate_chain_row_count": len(binding_rows),
        "candidate_archive_materialized_count": bound_count,
        "candidate_archive_materialized": bool(binding_rows)
        and bound_count == len(binding_rows),
        "runtime_consumption_proof_present_count": sum(
            1
            for row in binding_rows
            if row.get("runtime_consumption_proof_present") is True
        ),
        "receiver_consumed_count": sum(
            1 for row in binding_rows if row.get("receiver_consumed") is True
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_materializer_execution": False,
        "ready_for_exact_eval_dispatch": False,
        "binding_rows": binding_rows,
        "blockers": _unique_strings(blockers),
        "recommended_next_action": (
            "run_or_compose_receiver_consumed_materializer_manifests_for_each_"
            "repair_budget_candidate_chain_row"
        ),
        "allowed_use": "repair_budget_materializer_binding_for_execution_audit_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="frontier_repair_budget_materializer_binding_report",
    )
    return payload


def _binding_rows_by_candidate_id(
    binding_report: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(binding_report, Mapping):
        return {}
    if binding_report.get("schema") != REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA:
        raise FrontierRateAttackFeedbackError(
            "repair budget materializer binding report has unexpected schema"
        )
    require_no_truthy_authority_fields(
        binding_report,
        context="repair_budget_materializer_binding_report",
    )
    rows: dict[str, Mapping[str, Any]] = {}
    for row in binding_report.get("binding_rows") or []:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row.get("candidate_chain_id") or "")
        if candidate_id:
            rows[candidate_id] = row
    return rows


_BINDING_CLEARS_PLAN_BLOCKERS = frozenset(
    {
        "rate_only_candidate_archive_materialization_missing",
        "repair_candidate_archive_materialization_missing",
        "candidate_archive_materialization_missing",
        "receiver_runtime_consumption_proof_missing",
        "parent_rate_only_archive_materialization_required",
        "component_response_replay_required_before_budget_spend",
    }
)


def _repair_budget_materialization_execution_row(
    *,
    row: Mapping[str, Any],
    parent_candidate_chain_id: str,
    materialized_parent_ids: set[str],
    materialization_order_seen: list[int],
    binding_rows_by_candidate_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_chain_id = str(row.get("candidate_chain_id") or "")
    candidate_kind = str(row.get("candidate_kind") or "unknown_candidate_kind")
    order = _finite_int_or_none(row.get("materialization_order")) or 0
    materialization_order_seen.append(order)
    binding_row = (
        binding_rows_by_candidate_id.get(candidate_chain_id)
        if isinstance(binding_rows_by_candidate_id, Mapping)
        else None
    )
    blockers = _string_list(row.get("blockers"))
    if isinstance(binding_row, Mapping):
        blockers = [
            blocker
            for blocker in blockers
            if blocker not in _BINDING_CLEARS_PLAN_BLOCKERS
        ]
        blockers.extend(_string_list(binding_row.get("blockers")))
    if not candidate_chain_id:
        blockers.append("candidate_chain_id_missing")
    if order <= 0:
        blockers.append("materialization_order_missing")
    if candidate_kind != "rate_only_floor_parent":
        if str(row.get("parent_candidate_chain_id") or "") != parent_candidate_chain_id:
            blockers.append("child_parent_candidate_chain_id_mismatch")
        if parent_candidate_chain_id not in materialized_parent_ids:
            blockers.append("parent_rate_only_archive_materialization_required")
    archive_path = str(row.get("candidate_archive_path") or "").strip()
    archive_sha = str(row.get("candidate_archive_sha256") or "").strip()
    archive_bytes = row.get("candidate_archive_bytes")
    archive_materialized = row.get("candidate_archive_materialized") is True
    runtime_proof_present = row.get("runtime_consumption_proof_present") is True
    receiver_consumed = row.get("receiver_consumed") is True
    component_replayed = row.get("component_response_replayed") is True
    component_replay_path = str(row.get("component_response_replay_path") or "").strip()
    component_replay_axis_tag = str(
        row.get("component_response_replay_axis_tag") or ""
    ).strip()
    component_replay_evidence_grade = str(
        row.get("component_response_replay_evidence_grade") or ""
    ).strip()
    if isinstance(binding_row, Mapping):
        archive_path = str(
            binding_row.get("candidate_archive_path") or archive_path or ""
        ).strip()
        archive_sha = str(
            binding_row.get("candidate_archive_sha256") or archive_sha or ""
        ).strip()
        archive_bytes = binding_row.get("candidate_archive_bytes") or archive_bytes
        archive_materialized = (
            archive_materialized
            or binding_row.get("candidate_archive_materialized") is True
        )
        runtime_proof_present = (
            binding_row.get("runtime_consumption_proof_present") is True
        )
        receiver_consumed = binding_row.get("receiver_consumed") is True
        component_replayed = (
            component_replayed
            or binding_row.get("component_response_replayed") is True
        )
        component_replay_path = str(
            binding_row.get("component_response_replay_path")
            or component_replay_path
            or ""
        ).strip()
        component_replay_axis_tag = str(
            binding_row.get("component_response_replay_axis_tag")
            or component_replay_axis_tag
            or ""
        ).strip()
        component_replay_evidence_grade = str(
            binding_row.get("component_response_replay_evidence_grade")
            or component_replay_evidence_grade
            or ""
        ).strip()
    if archive_materialized and candidate_kind == "rate_only_floor_parent":
        materialized_parent_ids.add(candidate_chain_id)
    if not archive_materialized:
        blockers.append("candidate_archive_materialized_false")
    if archive_materialized and not archive_path:
        blockers.append("candidate_archive_path_missing")
    if not runtime_proof_present:
        blockers.append("runtime_consumption_proof_present_false")
    if not receiver_consumed:
        blockers.append("receiver_consumed_false")
    if candidate_kind == "spent_budget_repair_child" and not component_replayed:
        blockers.append("component_response_replayed_false")
    if candidate_kind == "spent_budget_repair_child" and component_replayed:
        if not component_replay_path:
            blockers.append("component_response_replay_path_missing")
        if not component_replay_axis_tag:
            blockers.append("component_response_replay_axis_tag_missing")
    if row.get("budget_spend_allowed") is True:
        blockers.append("budget_spend_authority_forbidden")
    if row.get("ready_for_exact_eval_dispatch") is True:
        blockers.append("exact_dispatch_authority_forbidden")
    component_replay_ready = (
        component_replayed
        and (
            candidate_kind != "spent_budget_repair_child"
            or bool(component_replay_path)
        )
    )
    ready_for_local_materialization = (
        archive_materialized
        and runtime_proof_present
        and receiver_consumed
        and (
            candidate_kind == "rate_only_floor_parent"
            or component_replay_ready
        )
    )
    return {
        "schema": REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA,
        "candidate_chain_id": candidate_chain_id,
        "candidate_kind": candidate_kind,
        "chain_id": row.get("chain_id"),
        "materialization_order": order,
        "parent_candidate_chain_id": row.get("parent_candidate_chain_id"),
        "parent_must_be_preserved_before_child": (
            row.get("parent_must_be_preserved_before_child") is True
            or candidate_kind == "spent_budget_repair_child"
        ),
        "candidate_archive_path": archive_path or None,
        "candidate_archive_sha256": archive_sha or None,
        "candidate_archive_bytes": archive_bytes,
        "candidate_archive_materialized": archive_materialized,
        "runtime_consumption_proof_present": runtime_proof_present,
        "receiver_consumed": receiver_consumed,
        "component_response_replayed": component_replayed,
        "component_response_replay_path": component_replay_path or None,
        "component_response_replay_axis_tag": component_replay_axis_tag or None,
        "component_response_replay_evidence_grade": (
            component_replay_evidence_grade or None
        ),
        "ready_for_local_materialization": ready_for_local_materialization,
        "ready_for_materializer_execution": False,
        "ready_for_budget_spend": False,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "execution_status": (
            "ready_for_receiver_closed_candidate_replay"
            if ready_for_local_materialization
            else "blocked_missing_receiver_consumed_candidate_archive"
        ),
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "reason": (
                "repair_budget_materialization_requires_receiver_consumed_archive_"
                "proof_before_exact_axis_dispatch"
            ),
            "blockers": _unique_strings(blockers),
            **FALSE_AUTHORITY,
        },
        "blockers": _unique_strings(blockers),
        "allowed_use": "repair_budget_materialization_execution_readiness_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def build_frontier_repair_budget_materialization_execution_report(
    *,
    repair_budget_materialization_plan: Mapping[str, Any],
    repair_budget_materialization_plan_path: str | Path | None = None,
    materializer_binding_report: Mapping[str, Any] | None = None,
    materializer_binding_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit repair-budget candidate-chain materialization readiness."""

    if (
        repair_budget_materialization_plan.get("schema")
        != REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "repair budget materialization plan has unexpected schema"
        )
    require_no_truthy_authority_fields(
        repair_budget_materialization_plan,
        context="repair_budget_materialization_execution_plan",
    )
    plan_rows = [
        row
        for row in repair_budget_materialization_plan.get("candidate_chain_rows") or []
        if isinstance(row, Mapping)
        and row.get("schema") == REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA
    ]
    parent_candidate_chain_id = str(
        repair_budget_materialization_plan.get("parent_candidate_chain_id") or ""
    )
    binding_rows_by_candidate_id = _binding_rows_by_candidate_id(
        materializer_binding_report
    )
    sorted_rows = sorted(
        plan_rows,
        key=lambda row: (
            _finite_int_or_none(row.get("materialization_order")) or 10**9,
            str(row.get("candidate_chain_id") or ""),
        ),
    )
    materialized_parent_ids: set[str] = set()
    materialization_order_seen: list[int] = []
    execution_rows = [
        _repair_budget_materialization_execution_row(
            row=row,
            parent_candidate_chain_id=parent_candidate_chain_id,
            materialized_parent_ids=materialized_parent_ids,
            materialization_order_seen=materialization_order_seen,
            binding_rows_by_candidate_id=binding_rows_by_candidate_id,
        )
        for row in sorted_rows
    ]
    ready_count = sum(
        1 for row in execution_rows if row.get("ready_for_local_materialization") is True
    )
    archive_materialized_count = sum(
        1 for row in execution_rows if row.get("candidate_archive_materialized") is True
    )
    runtime_proof_count = sum(
        1 for row in execution_rows if row.get("runtime_consumption_proof_present") is True
    )
    blockers = ["exact_auth_eval_required_before_score_or_promotion_claim"]
    if archive_materialized_count < len(execution_rows) or not execution_rows:
        blockers.append("candidate_archives_not_materialized")
    if runtime_proof_count < len(execution_rows) or not execution_rows:
        blockers.append("receiver_runtime_consumption_proof_missing")
    if not plan_rows:
        blockers.append("candidate_chain_rows_missing")
    if not parent_candidate_chain_id:
        blockers.append("parent_candidate_chain_id_missing")
    if materialization_order_seen and materialization_order_seen != sorted(
        materialization_order_seen
    ):
        blockers.append("candidate_chain_rows_not_sorted_by_materialization_order")
    if (
        not execution_rows
        or execution_rows[0].get("candidate_kind") != "rate_only_floor_parent"
    ):
        blockers.append("rate_only_floor_parent_not_first")
    blockers.extend(
        blocker
        for row in execution_rows
        for blocker in row.get("blockers") or []
    )
    payload = {
        "schema": REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA,
        "generated_at_utc": _utc_now(),
        "chain_id": repair_budget_materialization_plan.get("chain_id"),
        "source_materialization_plan_path": str(
            repair_budget_materialization_plan_path or ""
        ),
        "source_materialization_plan_schema": repair_budget_materialization_plan.get(
            "schema"
        ),
        "source_materializer_binding_report_path": str(
            materializer_binding_report_path or ""
        ),
        "source_materializer_binding_report_schema": (
            materializer_binding_report.get("schema")
            if isinstance(materializer_binding_report, Mapping)
            else None
        ),
        "materializer_binding_row_count": len(binding_rows_by_candidate_id),
        "candidate_chain_row_count": len(execution_rows),
        "parent_candidate_chain_id": parent_candidate_chain_id or None,
        "rate_only_floor_parent_first": bool(
            execution_rows
            and execution_rows[0].get("candidate_kind") == "rate_only_floor_parent"
        ),
        "spent_budget_candidates_are_children_of_rate_only_floor": (
            repair_budget_materialization_plan.get(
                "spent_budget_candidates_are_children_of_rate_only_floor"
            )
            is True
        ),
        "rate_only_candidate_remains_valid_even_if_child_regresses": (
            repair_budget_materialization_plan.get(
                "rate_only_candidate_remains_valid_even_if_child_regresses"
            )
            is True
        ),
        "ready_for_local_materialization_count": ready_count,
        "candidate_archive_materialized_count": archive_materialized_count,
        "runtime_consumption_proof_present_count": runtime_proof_count,
        "receiver_consumed_count": sum(
            1 for row in execution_rows if row.get("receiver_consumed") is True
        ),
        "component_response_replayed_count": sum(
            1 for row in execution_rows if row.get("component_response_replayed") is True
        ),
        "candidate_archive_materialized": archive_materialized_count == len(execution_rows)
        and bool(execution_rows),
        "runtime_consumption_proof_present": runtime_proof_count == len(execution_rows)
        and bool(execution_rows),
        "receiver_consumed": all(
            row.get("receiver_consumed") is True for row in execution_rows
        )
        and bool(execution_rows),
        "component_response_replayed": all(
            row.get("component_response_replayed") is True
            or row.get("candidate_kind") == "rate_only_floor_parent"
            for row in execution_rows
        )
        and bool(execution_rows),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_materializer_execution": False,
        "ready_for_exact_eval_dispatch": False,
        "execution_rows": execution_rows,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "reason": (
                "repair_budget_candidate_chain_is_local_materialization_readiness_"
                "only_until_receiver_consumed_archives_and_exact_axis_payload_exist"
            ),
            "blockers": _unique_strings(blockers),
            **FALSE_AUTHORITY,
        },
        "blockers": _unique_strings(blockers),
        "recommended_next_action": (
            "bind_receiver_runtime_materializers_to_rate_only_parent_then_child_"
            "repair_candidates_and_rerun_this_execution_report"
        ),
        "allowed_use": "queue_owned_repair_budget_materialization_readiness_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="frontier_repair_budget_materialization_execution_report",
    )
    return payload


def build_frontier_repair_budget_waterfill_queue(
    *,
    repo_root: str | Path,
    autonomous_chain_optimization: Mapping[str, Any],
    autonomous_chain_optimization_path: str | Path,
    targeted_component_correction_response_harvest: Mapping[str, Any] | None = None,
    targeted_component_correction_response_harvest_path: str | Path | None = None,
    receiver_closed_correction_budget: Mapping[str, Any] | None = None,
    receiver_closed_correction_budget_path: str | Path | None = None,
    materializer_work_queue: Mapping[str, Any] | None = None,
    materializer_work_queue_path: str | Path | None = None,
    materializer_execution_queue: Mapping[str, Any] | None = None,
    materializer_execution_queue_path: str | Path | None = None,
    repair_dynamics_palette_prior: Mapping[str, Any] | None = None,
    target_optimization_profile_metadata: Mapping[str, Any] | None = None,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    queue_id: str = "frontier_repair_budget_waterfill_queue",
    chain_limit: int = 4,
) -> dict[str, Any] | None:
    """Build a local queue that emits encoder-side repair allocation work orders."""

    if chain_limit < 1:
        raise FrontierRateAttackFeedbackError("chain_limit must be >= 1")
    require_no_truthy_authority_fields(
        autonomous_chain_optimization,
        context="repair_budget_waterfill_queue_autonomous_chain",
    )
    if isinstance(repair_dynamics_palette_prior, Mapping) and repair_dynamics_palette_prior:
        require_no_truthy_authority_fields(
            repair_dynamics_palette_prior,
            context="repair_budget_waterfill_queue_repair_dynamics_palette_prior",
        )
    else:
        repair_dynamics_palette_prior = {}
    target_profile_metadata = (
        dict(target_optimization_profile_metadata)
        if isinstance(target_optimization_profile_metadata, Mapping)
        and target_optimization_profile_metadata
        else {}
    )
    if target_profile_metadata:
        require_no_truthy_authority_fields(
            target_profile_metadata,
            context="repair_budget_waterfill_queue_target_optimization_profile",
        )
    repair_palette_modes = _string_list(repair_dynamics_palette_prior.get("palette_modes"))
    rows = [
        row
        for row in autonomous_chain_optimization.get("rows") or []
        if isinstance(row, Mapping)
        and row.get("schema") == AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA
    ][:chain_limit]
    if not rows:
        return None

    repo = Path(repo_root)
    source_path = _resolve_path(autonomous_chain_optimization_path, repo_root=repo)
    materializer_work_queue_ref = (
        _repo_rel(_resolve_path(materializer_work_queue_path, repo_root=repo), repo)
        if materializer_work_queue_path is not None
        else ""
    )
    materializer_execution_queue_ref = (
        _repo_rel(
            _resolve_path(materializer_execution_queue_path, repo_root=repo),
            repo,
        )
        if materializer_execution_queue_path is not None
        else ""
    )
    missing_prerequisite_keys: list[str] = []
    if not isinstance(targeted_component_correction_response_harvest, Mapping):
        missing_prerequisite_keys.append("targeted_component_correction_response_harvest")
    if not isinstance(receiver_closed_correction_budget, Mapping):
        missing_prerequisite_keys.append("receiver_closed_correction_budget")
    if missing_prerequisite_keys:
        results_base = _resolve_path(str(results_root), repo_root=repo)
        experiments: list[dict[str, Any]] = []
        for priority, row in enumerate(rows, start=1):
            chain_id = str(row.get("chain_id") or f"chain_{priority}")
            rate_plan = (
                row.get("rate_budget_preservation_plan")
                if isinstance(row.get("rate_budget_preservation_plan"), Mapping)
                else {}
            )
            operator_action_ledger = (
                rate_plan.get("operator_action_ledger")
                if isinstance(rate_plan.get("operator_action_ledger"), Mapping)
                else {}
            )
            metadata = {
                "schema": REPAIR_BUDGET_WATERFILL_QUEUE_METADATA_SCHEMA,
                "chain_id": chain_id,
                "chain_family": row.get("chain_family"),
                "pipeline_side": "encoder_repair_allocator",
                "accepted_response_count": 0,
                "receiver_closed_saved_bytes_total": 0,
                "repair_dynamics_prior_active": bool(repair_palette_modes),
                "repair_dynamics_palette_prior": dict(repair_dynamics_palette_prior),
                "frontier_target_optimization_profile": dict(
                    target_profile_metadata
                ),
                "rate_only_candidate_count": (
                    rate_plan.get("rate_only_candidate_count")
                ),
                "rate_only_saved_bytes_total": (
                    rate_plan.get("rate_only_saved_bytes_total")
                ),
                "operator_action_ledger_schema": operator_action_ledger.get("schema"),
                "operator_action_term_count": operator_action_ledger.get("term_count"),
                "repair_allocation_action_term_schema": (
                    REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
                ),
                "typed_response_ledger_schema": (
                    REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA
                ),
                "typed_response_row_schema": REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA,
                "typed_response_row_count": 0,
                "materializer_work_queue_schema": (
                    materializer_work_queue.get("schema")
                    if isinstance(materializer_work_queue, Mapping)
                    else None
                ),
                "materializer_execution_queue_schema": (
                    materializer_execution_queue.get("schema")
                    if isinstance(materializer_execution_queue, Mapping)
                    else None
                ),
                "materializer_work_queue_path": materializer_work_queue_ref or None,
                "materializer_execution_queue_path": materializer_execution_queue_ref
                or None,
                "missing_prerequisite_artifact_keys": _unique_strings(
                    missing_prerequisite_keys
                ),
                "queue_actuation_ready": False,
                "queue_actuation_blockers": [
                    f"missing_prerequisite_artifact:{key}"
                    for key in _unique_strings(missing_prerequisite_keys)
                ],
                "preserve_rate_only_archive_before_budget_spend": True,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                "source_artifact_paths": _unique_strings(
                    [
                        _repo_rel(source_path, repo),
                        str(targeted_component_correction_response_harvest_path or ""),
                        str(receiver_closed_correction_budget_path or ""),
                    ]
                ),
                "allowed_use": "blocked_local_encoder_repair_waterfill_queue_only",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
            require_no_truthy_authority_fields(
                metadata,
                context=f"blocked_repair_budget_waterfill_queue_metadata:{chain_id}",
            )
            experiments.append(
                {
                    "id": f"repair_waterfill_{_slug_token(chain_id)}",
                    "priority": priority,
                    "status": "frozen",
                    "tags": [
                        "frontier-rate-attack",
                        "encoder-repair-allocator",
                        "segnet-posenet-waterfill",
                        "blocked-missing-prerequisite",
                        "no-score-authority",
                    ],
                    "metadata": metadata,
                    "steps": [
                        {
                            "id": "inspect_missing_repair_waterfill_prerequisites",
                            "kind": "command",
                            "command": [
                                ".venv/bin/python",
                                "-m",
                                "json.tool",
                                _repo_rel(source_path, repo),
                            ],
                            "resources": {"kind": "local_io_heavy"},
                            "timeout_seconds": 60,
                            "telemetry": {
                                "input_artifact_paths": [_repo_rel(source_path, repo)],
                            },
                        }
                    ],
                }
            )
        return normalize_queue_definition(
            {
                "schema": QUEUE_SCHEMA,
                "queue_id": queue_id,
                "controls": {
                    "mode": "running",
                    "local_first": True,
                    "max_concurrency": {"local_io_heavy": 1},
                },
                "metadata": {
                    "schema": "frontier_rate_attack_repair_budget_waterfill_queue_blocked_metadata.v1",
                    "frontier_target_optimization_profile": dict(
                        target_profile_metadata
                    ),
                    "queue_actuation_ready": False,
                    "queue_actuation_blockers": [
                        f"missing_prerequisite_artifact:{key}"
                        for key in _unique_strings(missing_prerequisite_keys)
                    ],
                    "missing_prerequisite_artifact_keys": _unique_strings(
                        missing_prerequisite_keys
                    ),
                    "materializer_work_queue_path": materializer_work_queue_ref
                    or None,
                    "materializer_execution_queue_path": (
                        materializer_execution_queue_ref or None
                    ),
                    "results_root": _repo_rel(results_base, repo),
                    **FALSE_AUTHORITY,
                },
                "experiments": experiments,
            }
        )
    harvest_path = _resolve_path(
        targeted_component_correction_response_harvest_path or "",
        repo_root=repo,
    )
    budget_path = _resolve_path(
        receiver_closed_correction_budget_path or "",
        repo_root=repo,
    )
    results_base = _resolve_path(str(results_root), repo_root=repo)
    queue_root = results_base / "frontier_repair_budget_waterfill" / _slug_token(queue_id)
    accepted_count = len(
        _accepted_targeted_component_response_rows(
            targeted_component_correction_response_harvest
        )
    )
    available_bytes = (
        _finite_int_or_none(
            receiver_closed_correction_budget.get("receiver_closed_saved_bytes_total")
        )
        or 0
    )
    waterfill_blockers: list[str] = []
    if accepted_count <= 0:
        waterfill_blockers.append("no_accepted_targeted_component_correction_responses")
        waterfill_blockers.extend(
            _string_list(targeted_component_correction_response_harvest.get("blockers"))
        )
    if available_bytes <= 0:
        waterfill_blockers.append("no_receiver_closed_saved_bytes_available")
        waterfill_blockers.extend(_string_list(receiver_closed_correction_budget.get("blockers")))
    waterfill_blockers = _unique_strings(waterfill_blockers)
    if waterfill_blockers:
        experiments: list[dict[str, Any]] = []
        for priority, row in enumerate(rows, start=1):
            chain_id = str(row.get("chain_id") or f"chain_{priority}")
            rate_plan = (
                row.get("rate_budget_preservation_plan")
                if isinstance(row.get("rate_budget_preservation_plan"), Mapping)
                else {}
            )
            operator_action_ledger = (
                rate_plan.get("operator_action_ledger")
                if isinstance(rate_plan.get("operator_action_ledger"), Mapping)
                else {}
            )
            metadata = {
                "schema": REPAIR_BUDGET_WATERFILL_QUEUE_METADATA_SCHEMA,
                "chain_id": chain_id,
                "chain_family": row.get("chain_family"),
                "pipeline_side": "encoder_repair_allocator",
                "accepted_response_count": accepted_count,
                "receiver_closed_saved_bytes_total": available_bytes,
                "repair_dynamics_prior_active": bool(repair_palette_modes),
                "repair_dynamics_palette_prior": dict(repair_dynamics_palette_prior),
                "frontier_target_optimization_profile": dict(
                    target_profile_metadata
                ),
                "rate_only_candidate_count": (
                    rate_plan.get("rate_only_candidate_count")
                ),
                "rate_only_saved_bytes_total": (
                    rate_plan.get("rate_only_saved_bytes_total")
                ),
                "operator_action_ledger_schema": operator_action_ledger.get("schema"),
                "operator_action_term_count": operator_action_ledger.get("term_count"),
                "repair_allocation_action_term_schema": (
                    REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
                ),
                "typed_response_ledger_schema": (
                    REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA
                ),
                "typed_response_row_schema": REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA,
                "typed_response_row_count": accepted_count,
                "materializer_work_queue_schema": (
                    materializer_work_queue.get("schema")
                    if isinstance(materializer_work_queue, Mapping)
                    else None
                ),
                "materializer_execution_queue_schema": (
                    materializer_execution_queue.get("schema")
                    if isinstance(materializer_execution_queue, Mapping)
                    else None
                ),
                "materializer_work_queue_path": materializer_work_queue_ref or None,
                "materializer_execution_queue_path": materializer_execution_queue_ref
                or None,
                "queue_actuation_ready": False,
                "queue_actuation_blockers": waterfill_blockers,
                "preserve_rate_only_archive_before_budget_spend": True,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                "source_artifact_paths": [
                    _repo_rel(source_path, repo),
                    _repo_rel(harvest_path, repo),
                    _repo_rel(budget_path, repo),
                ],
                "allowed_use": "blocked_local_encoder_repair_waterfill_queue_only",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
            require_no_truthy_authority_fields(
                metadata,
                context=(
                    "blocked_empty_response_repair_budget_waterfill_queue_metadata:"
                    f"{chain_id}"
                ),
            )
            experiments.append(
                {
                    "id": f"repair_waterfill_{_slug_token(chain_id)}",
                    "priority": priority,
                    "status": "frozen",
                    "tags": [
                        "frontier-rate-attack",
                        "encoder-repair-allocator",
                        "segnet-posenet-waterfill",
                        "blocked-no-actionable-response",
                        "no-score-authority",
                    ],
                    "metadata": metadata,
                    "steps": [
                        {
                            "id": "inspect_blocked_repair_waterfill_response_prerequisites",
                            "kind": "command",
                            "command": [
                                ".venv/bin/python",
                                "-m",
                                "json.tool",
                                _repo_rel(harvest_path, repo),
                            ],
                            "resources": {"kind": "local_io_heavy"},
                            "timeout_seconds": 60,
                            "telemetry": {
                                "input_artifact_paths": [
                                    _repo_rel(source_path, repo),
                                    _repo_rel(harvest_path, repo),
                                    _repo_rel(budget_path, repo),
                                ],
                                "include_postcondition_paths": True,
                            },
                        }
                    ],
                }
            )
        return normalize_queue_definition(
            {
                "schema": QUEUE_SCHEMA,
                "queue_id": queue_id,
                "controls": {
                    "mode": "running",
                    "local_first": True,
                    "max_concurrency": {"local_io_heavy": 1},
                },
                "metadata": {
                    "schema": (
                        "frontier_rate_attack_repair_budget_waterfill_queue_"
                        "blocked_metadata.v1"
                    ),
                    "frontier_target_optimization_profile": dict(
                        target_profile_metadata
                    ),
                    "queue_actuation_ready": False,
                    "queue_actuation_blockers": waterfill_blockers,
                    "accepted_response_count": accepted_count,
                    "receiver_closed_saved_bytes_total": available_bytes,
                    "materializer_work_queue_path": materializer_work_queue_ref
                    or None,
                    "materializer_execution_queue_path": (
                        materializer_execution_queue_ref or None
                    ),
                    "results_root": _repo_rel(results_base, repo),
                    **FALSE_AUTHORITY,
                },
                "experiments": experiments,
            }
        )
    experiments: list[dict[str, Any]] = []
    for priority, row in enumerate(rows, start=1):
        chain_id = str(row.get("chain_id") or f"chain_{priority}")
        rate_plan = (
            row.get("rate_budget_preservation_plan")
            if isinstance(row.get("rate_budget_preservation_plan"), Mapping)
            else {}
        )
        operator_action_ledger = (
            rate_plan.get("operator_action_ledger")
            if isinstance(rate_plan.get("operator_action_ledger"), Mapping)
            else {}
        )
        work_order_path = (
            queue_root / _slug_token(chain_id) / "repair_budget_waterfill_work_order.json"
        )
        materialization_plan_path = (
            queue_root
            / _slug_token(chain_id)
            / "repair_budget_materialization_plan.json"
        )
        materializer_binding_report_path = (
            queue_root
            / _slug_token(chain_id)
            / "repair_budget_materializer_binding_report.json"
        )
        child_component_replay_manifests_path = (
            queue_root
            / _slug_token(chain_id)
            / "repair_budget_child_component_replay_manifests.json"
        )
        execution_report_path = (
            queue_root
            / _slug_token(chain_id)
            / "repair_budget_materialization_execution_report.json"
        )
        work_order_ref = _repo_rel(work_order_path, repo)
        materialization_plan_ref = _repo_rel(materialization_plan_path, repo)
        materializer_binding_report_ref = _repo_rel(
            materializer_binding_report_path,
            repo,
        )
        child_component_replay_manifests_ref = _repo_rel(
            child_component_replay_manifests_path,
            repo,
        )
        execution_report_ref = _repo_rel(execution_report_path, repo)
        metadata = {
            "schema": REPAIR_BUDGET_WATERFILL_QUEUE_METADATA_SCHEMA,
            "chain_id": chain_id,
            "chain_family": row.get("chain_family"),
            "pipeline_side": "encoder_repair_allocator",
            "accepted_response_count": accepted_count,
            "receiver_closed_saved_bytes_total": available_bytes,
            "repair_dynamics_prior_active": bool(repair_palette_modes),
            "repair_dynamics_palette_prior": dict(repair_dynamics_palette_prior),
            "frontier_target_optimization_profile": dict(target_profile_metadata),
            "rate_only_candidate_count": (
                rate_plan.get("rate_only_candidate_count")
            ),
            "rate_only_saved_bytes_total": (
                rate_plan.get("rate_only_saved_bytes_total")
            ),
            "cumulative_rate_attack": dict(
                rate_plan.get("cumulative_rate_attack")
                if isinstance(rate_plan.get("cumulative_rate_attack"), Mapping)
                else {}
            ),
            "operator_action_ledger_schema": operator_action_ledger.get("schema"),
            "operator_action_term_count": operator_action_ledger.get("term_count"),
            "repair_allocation_action_term_schema": (
                REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
            ),
            "typed_response_ledger_schema": REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA,
            "typed_response_row_schema": REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA,
            "typed_response_row_count": accepted_count,
            "preserve_rate_only_archive_before_budget_spend": True,
            "candidate_chain_materialization_plan_path": materialization_plan_ref,
            "candidate_chain_component_replay_manifests_path": (
                child_component_replay_manifests_ref
            ),
            "candidate_chain_materializer_binding_report_path": (
                materializer_binding_report_ref
            ),
            "candidate_chain_execution_report_path": execution_report_ref,
            "candidate_archive_materialized": False,
            "materializer_work_queue_schema": (
                materializer_work_queue.get("schema")
                if isinstance(materializer_work_queue, Mapping)
                else None
            ),
            "materializer_execution_queue_schema": (
                materializer_execution_queue.get("schema")
                if isinstance(materializer_execution_queue, Mapping)
                else None
            ),
            "materializer_work_queue_path": materializer_work_queue_ref or None,
            "materializer_execution_queue_path": materializer_execution_queue_ref
            or None,
            "queue_actuation_ready": True,
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            "source_artifact_paths": [
                _repo_rel(source_path, repo),
                _repo_rel(harvest_path, repo),
                _repo_rel(budget_path, repo),
            ],
            "allowed_use": "local_encoder_repair_waterfill_work_order_queue_only",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            metadata,
            context=f"repair_budget_waterfill_queue_metadata:{chain_id}",
        )
        experiments.append(
            {
                "id": f"repair_waterfill_{_slug_token(chain_id)}",
                "priority": priority,
                "status": "queued",
                "tags": [
                    "frontier-rate-attack",
                    "encoder-repair-allocator",
                    "segnet-posenet-waterfill",
                    "no-score-authority",
                ],
                "metadata": metadata,
                "steps": [
                    {
                        "id": "emit_repair_budget_waterfill_work_order",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/build_frontier_repair_budget_waterfill_work_order.py",
                            "--autonomous-chain-optimization",
                            _repo_rel(source_path, repo),
                            "--chain-id",
                            chain_id,
                            "--targeted-component-correction-response-harvest",
                            _repo_rel(harvest_path, repo),
                            "--receiver-closed-correction-budget",
                            _repo_rel(budget_path, repo),
                            "--work-order-out",
                            work_order_ref,
                            "--overwrite",
                        ],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": work_order_ref,
                                "key": "schema",
                                "equals": REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": work_order_ref,
                            },
                            {
                                "type": "json_equals",
                                "path": work_order_ref,
                                "key": "typed_response_ledger_schema",
                                "equals": REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA,
                            },
                            {
                                "type": "json_equals",
                                "path": work_order_ref,
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                            {
                                "type": "json_equals",
                                "path": work_order_ref,
                                "key": "budget_spend_allowed",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [work_order_ref],
                            "input_artifact_paths": [
                                _repo_rel(source_path, repo),
                                _repo_rel(harvest_path, repo),
                                _repo_rel(budget_path, repo),
                            ],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "emit_repair_budget_materialization_plan",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/build_frontier_repair_budget_materialization_plan.py",
                            "--work-order",
                            work_order_ref,
                            "--materialization-plan-out",
                            materialization_plan_ref,
                            "--overwrite",
                        ],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": materialization_plan_ref,
                                "key": "schema",
                                "equals": REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": materialization_plan_ref,
                            },
                            {
                                "type": "json_equals",
                                "path": materialization_plan_ref,
                                "key": "candidate_archive_materialized",
                                "equals": False,
                            },
                            {
                                "type": "json_equals",
                                "path": materialization_plan_ref,
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                            {
                                "type": "json_equals",
                                "path": materialization_plan_ref,
                                "key": "budget_spend_allowed",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [materialization_plan_ref],
                            "input_artifact_paths": [work_order_ref],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "emit_repair_budget_child_component_replay_manifests",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/build_frontier_repair_budget_child_component_replay_manifests.py",
                            "--materialization-plan",
                            materialization_plan_ref,
                            "--output-manifest",
                            child_component_replay_manifests_ref,
                            "--overwrite",
                        ],
                        "requires": ["emit_repair_budget_materialization_plan"],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": child_component_replay_manifests_ref,
                                "key": "schema",
                                "equals": (
                                    REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFESTS_SCHEMA
                                ),
                            },
                            {
                                "type": "json_false_authority",
                                "path": child_component_replay_manifests_ref,
                            },
                            {
                                "type": "json_equals",
                                "path": child_component_replay_manifests_ref,
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                            {
                                "type": "json_equals",
                                "path": child_component_replay_manifests_ref,
                                "key": "budget_spend_allowed",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [child_component_replay_manifests_ref],
                            "input_artifact_paths": [materialization_plan_ref],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "bind_repair_budget_materializer_execution",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/build_frontier_repair_budget_materializer_binding_report.py",
                            "--materialization-plan",
                            materialization_plan_ref,
                            "--binding-report-out",
                            materializer_binding_report_ref,
                            "--materializer-manifest",
                            child_component_replay_manifests_ref,
                            *(
                                [
                                    "--materializer-work-queue",
                                    materializer_work_queue_ref,
                                ]
                                if materializer_work_queue_ref
                                else []
                            ),
                            *(
                                [
                                    "--materializer-execution-queue",
                                    materializer_execution_queue_ref,
                                ]
                                if materializer_execution_queue_ref
                                else []
                            ),
                            *[
                                item
                                for mode in repair_palette_modes
                                for item in ("--repair-palette-mode", mode)
                            ],
                            "--overwrite",
                        ],
                        "requires": [
                            "emit_repair_budget_child_component_replay_manifests"
                        ],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": materializer_binding_report_ref,
                                "key": "schema",
                                "equals": (
                                    REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA
                                ),
                            },
                            {
                                "type": "json_false_authority",
                                "path": materializer_binding_report_ref,
                            },
                            {
                                "type": "json_equals",
                                "path": materializer_binding_report_ref,
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                            {
                                "type": "json_equals",
                                "path": materializer_binding_report_ref,
                                "key": "budget_spend_allowed",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [materializer_binding_report_ref],
                            "input_artifact_paths": _unique_strings(
                                [
                                    materialization_plan_ref,
                                    child_component_replay_manifests_ref,
                                    materializer_work_queue_ref,
                                    materializer_execution_queue_ref,
                                ]
                            ),
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "audit_repair_budget_materialization_execution",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/build_frontier_repair_budget_materialization_execution_report.py",
                            "--materialization-plan",
                            materialization_plan_ref,
                            "--materializer-binding-report",
                            materializer_binding_report_ref,
                            "--execution-report-out",
                            execution_report_ref,
                            "--overwrite",
                        ],
                        "requires": ["bind_repair_budget_materializer_execution"],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": execution_report_ref,
                                "key": "schema",
                                "equals": (
                                    REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
                                ),
                            },
                            {
                                "type": "json_false_authority",
                                "path": execution_report_ref,
                            },
                            {
                                "type": "json_equals",
                                "path": execution_report_ref,
                                "key": "candidate_archive_materialized",
                                "equals": False,
                            },
                            {
                                "type": "json_equals",
                                "path": execution_report_ref,
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                            {
                                "type": "json_equals",
                                "path": execution_report_ref,
                                "key": "budget_spend_allowed",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [execution_report_ref],
                            "input_artifact_paths": [
                                materialization_plan_ref,
                                materializer_binding_report_ref,
                            ],
                            "include_postcondition_paths": True,
                        },
                    },
                ],
            }
        )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "metadata": {
                "schema": "frontier_rate_attack_repair_budget_waterfill_queue_metadata.v1",
                "frontier_target_optimization_profile": dict(
                    target_profile_metadata
                ),
                "queue_actuation_ready": True,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                "allowed_use": "repair_waterfill_queue_target_binding_metadata",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            },
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_io_heavy": 1},
            },
            "experiments": experiments,
        }
    )


def _autonomous_action_artifact_key(action: Mapping[str, Any]) -> str:
    return str(action.get("queue_artifact_key") or action.get("source_artifact_key") or "")


def _child_queue_health(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "queued_experiment_count": 0,
            "frozen_experiment_count": 0,
            "disabled_experiment_count": 0,
            "blockers": ["child_queue_artifact_missing"],
            **FALSE_AUTHORITY,
        }
    try:
        payload = _load_json(path)
        require_no_truthy_authority_fields(
            payload,
            context=f"autonomous_chain_child_queue:{path}",
        )
    except (FrontierRateAttackFeedbackError, ExperimentQueueError) as exc:
        return {
            "path": str(path),
            "present": True,
            "queued_experiment_count": 0,
            "frozen_experiment_count": 0,
            "disabled_experiment_count": 0,
            "blockers": [f"child_queue_artifact_invalid:{exc}"],
            **FALSE_AUTHORITY,
        }
    experiments = (
        payload.get("experiments") if isinstance(payload.get("experiments"), list) else []
    )
    queued = 0
    frozen = 0
    disabled = 0
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            continue
        status = str(experiment.get("status") or "")
        if status == "queued":
            queued += 1
        elif status == "frozen":
            frozen += 1
        elif status == "disabled":
            disabled += 1
    blockers: list[str] = []
    if queued <= 0:
        blockers.append("child_queue_has_no_queued_experiments")
    return {
        "path": str(path),
        "present": True,
        "queue_id": str(payload.get("queue_id") or ""),
        "queued_experiment_count": queued,
        "frozen_experiment_count": frozen,
        "disabled_experiment_count": disabled,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def build_frontier_autonomous_chain_optimization_queue(
    *,
    repo_root: str | Path,
    autonomous_chain_optimization: Mapping[str, Any],
    autonomous_chain_optimization_path: str | Path,
    artifact_paths_by_key: Mapping[str, str | Path],
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    queue_id: str = "frontier_autonomous_chain_optimization_queue",
    chain_limit: int = 4,
) -> dict[str, Any] | None:
    """Build the parent queue that actuates selected many-op chains locally."""

    if chain_limit < 1:
        raise FrontierRateAttackFeedbackError("chain_limit must be >= 1")
    require_no_truthy_authority_fields(
        autonomous_chain_optimization,
        context="autonomous_chain_optimization_queue_input",
    )
    rows = [
        row
        for row in autonomous_chain_optimization.get("rows") or []
        if isinstance(row, Mapping)
        and row.get("schema") == AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA
    ][:chain_limit]
    if not rows:
        return None

    repo = Path(repo_root)
    source_path = _resolve_path(autonomous_chain_optimization_path, repo_root=repo)
    results_base = _resolve_path(str(results_root), repo_root=repo)
    queue_root = (
        results_base / "frontier_autonomous_chain_optimization" / _slug_token(queue_id)
    )
    experiments: list[dict[str, Any]] = []
    used_resource_kinds = {"local_io_heavy"}
    for priority, row in enumerate(rows, start=1):
        chain_id = str(row.get("chain_id") or f"chain_{priority}")
        experiment_id = f"autonomous_chain_{_slug_token(chain_id)}"
        work_dir = queue_root / _slug_token(chain_id)
        work_order_path = work_dir / "autonomous_chain_work_order.json"
        local_actions = [
            dict(action)
            for action in row.get("scheduler_actions") or []
            if isinstance(action, Mapping) and not bool(action.get("advisory_only"))
        ]
        if (
            "pair_frame_5d_coverage_acquisition_queue" in artifact_paths_by_key
            and not any(
                action.get("queue_artifact_key")
                == "pair_frame_5d_coverage_acquisition_queue"
                for action in local_actions
            )
        ):
            local_actions.append(
                {
                    "id": "run_pair_frame_5d_coverage_acquisition_queue",
                    "queue_artifact_key": "pair_frame_5d_coverage_acquisition_queue",
                    "purpose": (
                        "consume_5d_coverage_work_orders_refresh_canvas_and_"
                        "refire_extended_operators"
                    ),
                    "bounded_local_execution": True,
                    "advisory_only": False,
                    "requires_exact_auth_before_score_claim": True,
                    "max_steps": 16,
                    "max_experiments": 6,
                    "max_parallel": 1,
                    **FALSE_AUTHORITY,
                }
            )
        if (
            "pair_frame_5d_extended_operator_queue" in artifact_paths_by_key
            and not any(
                action.get("queue_artifact_key")
                == "pair_frame_5d_extended_operator_queue"
                for action in local_actions
            )
        ):
            local_actions.append(
                {
                    "id": "run_pair_frame_5d_extended_operator_queue",
                    "queue_artifact_key": "pair_frame_5d_extended_operator_queue",
                    "purpose": (
                        "fire_all_8_pair_frame_5d_extended_operators_against_"
                        "populated_canvas"
                    ),
                    "bounded_local_execution": True,
                    "advisory_only": False,
                    "requires_exact_auth_before_score_claim": True,
                    **FALSE_AUTHORITY,
                }
            )
        if (
            "repair_campaign_score_queue" in artifact_paths_by_key
            and not any(
                action.get("queue_artifact_key") == "repair_campaign_score_queue"
                for action in local_actions
            )
        ):
            local_actions.append(
                {
                    "id": (
                        "score_repair_campaign_and_update_stackability_posterior"
                    ),
                    "queue_artifact_key": "repair_campaign_score_queue",
                    "purpose": (
                        "score_repair_waterfill_ledger_then_run_stackability_"
                        "replay_learning_and_posterior_update"
                    ),
                    "bounded_local_execution": True,
                    "advisory_only": False,
                    "requires_exact_auth_before_score_claim": True,
                    "max_steps": REPAIR_CAMPAIGN_SCORE_QUEUE_MAX_STEPS,
                    "max_experiments": REPAIR_CAMPAIGN_SCORE_QUEUE_MAX_EXPERIMENTS,
                    "max_parallel": REPAIR_CAMPAIGN_SCORE_QUEUE_MAX_PARALLEL,
                    **FALSE_AUTHORITY,
                }
            )
        advisory_actions = [
            dict(action)
            for action in row.get("scheduler_actions") or []
            if isinstance(action, Mapping) and bool(action.get("advisory_only"))
        ]
        steps: list[dict[str, Any]] = [
            {
                "id": "emit_autonomous_chain_work_order",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "tools/build_frontier_autonomous_chain_work_order.py",
                    "--autonomous-chain-optimization",
                    _repo_rel(source_path, repo),
                    "--chain-id",
                    chain_id,
                    "--work-order-out",
                    _repo_rel(work_order_path, repo),
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": _repo_rel(work_order_path, repo),
                        "key": "schema",
                        "equals": AUTONOMOUS_CHAIN_WORK_ORDER_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": _repo_rel(work_order_path, repo),
                    },
                    {
                        "type": "json_equals",
                        "path": _repo_rel(work_order_path, repo),
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [_repo_rel(work_order_path, repo)],
                    "input_artifact_paths": [_repo_rel(source_path, repo)],
                    "include_postcondition_paths": True,
                },
            }
        ]
        missing_queue_artifact_keys: list[str] = []
        child_queue_paths: list[str] = []
        child_queue_run_step_ids: list[str] = []
        receiver_repair_run_step_ids: list[str] = []
        planned_child_queue_run_step_id_by_key: dict[str, str] = {}
        child_queue_health_by_key: dict[str, dict[str, Any]] = {}
        blocked_child_queue_artifact_keys: list[str] = []
        for action_index, action in enumerate(local_actions, start=1):
            queue_key = str(action.get("queue_artifact_key") or "")
            if not queue_key or queue_key not in artifact_paths_by_key:
                continue
            slug = _slug_token(f"{action.get('id') or action_index}_{queue_key}")
            planned_child_queue_run_step_id_by_key.setdefault(
                queue_key,
                f"run_{slug}_bounded_local",
            )
        for action_index, action in enumerate(local_actions, start=1):
            queue_key = str(action.get("queue_artifact_key") or "")
            if not queue_key:
                missing_queue_artifact_keys.append(
                    f"{action.get('id') or action_index}:missing_queue_artifact_key"
                )
                continue
            raw_child_queue = artifact_paths_by_key.get(queue_key)
            if raw_child_queue is None:
                missing_queue_artifact_keys.append(queue_key)
                continue
            child_queue_path = _resolve_path(raw_child_queue, repo_root=repo)
            child_queue_ref = _repo_rel(child_queue_path, repo)
            child_queue_paths.append(child_queue_ref)
            child_health = _child_queue_health(child_queue_path)
            child_queue_health_by_key[queue_key] = {
                **child_health,
                "path": child_queue_ref,
            }
            if child_health.get("blockers"):
                blocked_child_queue_artifact_keys.append(queue_key)
            slug = _slug_token(f"{action.get('id') or action_index}_{queue_key}")
            validate_step_id = f"validate_{slug}"
            steps.append(
                {
                    "id": validate_step_id,
                    "kind": "command",
                    "command": [
                        ".venv/bin/python",
                        "tools/experiment_queue.py",
                        "--queue",
                        child_queue_ref,
                        "validate",
                    ],
                    "requires": ["emit_autonomous_chain_work_order"],
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 120,
                    "telemetry": {
                        "input_artifact_paths": [
                            child_queue_ref,
                            _repo_rel(work_order_path, repo),
                        ],
                    },
                }
            )
            run_step_id = f"run_{slug}_bounded_local"
            child_worker_result_path = work_dir / f"{slug}_worker_result.json"
            child_worker_result_ref = _repo_rel(child_worker_result_path, repo)
            child_max_steps = (
                _finite_int_or_none(action.get("max_steps"))
                or AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_STEPS
            )
            child_max_experiments = (
                _finite_int_or_none(action.get("max_experiments"))
                or AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_EXPERIMENTS
            )
            child_max_parallel = (
                _finite_int_or_none(action.get("max_parallel"))
                or AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_PARALLEL
            )
            if child_max_steps < 1:
                child_max_steps = AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_STEPS
            if child_max_experiments < 1:
                child_max_experiments = AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_EXPERIMENTS
            if child_max_parallel < 1:
                child_max_parallel = AUTONOMOUS_CHILD_QUEUE_DEFAULT_MAX_PARALLEL
            required_run_steps = [validate_step_id]
            if queue_key == "repair_campaign_score_queue":
                repair_budget_run_step_id = (
                    planned_child_queue_run_step_id_by_key.get(
                        "repair_budget_waterfill_queue"
                    )
                )
                if repair_budget_run_step_id and repair_budget_run_step_id != run_step_id:
                    required_run_steps.append(repair_budget_run_step_id)
            steps.append(
                {
                    "id": run_step_id,
                    "kind": "command",
                    "command": [
                        ".venv/bin/python",
                        "tools/experiment_queue.py",
                        "--queue",
                        child_queue_ref,
                        "run-worker",
                        "--execute",
                        "--max-steps",
                        str(child_max_steps),
                        "--max-experiments",
                        str(child_max_experiments),
                        "--max-parallel",
                        str(child_max_parallel),
                        "--output",
                        child_worker_result_ref,
                    ],
                    "requires": _unique_strings(required_run_steps),
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 900,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": child_worker_result_ref,
                            "key": "schema",
                            "equals": "experiment_queue_worker_result.v1",
                        },
                        {
                            "type": "json_equals",
                            "path": child_worker_result_ref,
                            "key": "failure_count",
                            "equals": 0,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [child_worker_result_ref],
                        "input_artifact_paths": [
                            child_queue_ref,
                            _repo_rel(work_order_path, repo),
                        ],
                        "include_postcondition_paths": True,
                    },
                }
            )
            child_queue_run_step_ids.append(run_step_id)
            if queue_key == "receiver_repair_queue":
                receiver_repair_run_step_ids.append(run_step_id)
        post_repair_refresh_planned = False
        post_repair_refresh_report_ref = ""
        if receiver_repair_run_step_ids and any(
            str(action.get("id") or "") == "bind_targeted_chain_materializer_contexts"
            for action in advisory_actions
        ):
            post_repair_refresh_dir = work_dir / "post_receiver_repair_refresh"
            post_repair_refresh_report = post_repair_refresh_dir / "feedback_refresh_report.json"
            post_repair_refresh_report_ref = _repo_rel(post_repair_refresh_report, repo)
            steps.append(
                {
                    "id": "refresh_after_receiver_repair_for_targeted_materializers",
                    "kind": "command",
                    "command": [
                        ".venv/bin/python",
                        "tools/build_frontier_rate_attack_feedback_refresh.py",
                        "--output-dir",
                        _repo_rel(post_repair_refresh_dir, repo),
                        "--results-root",
                        _repo_rel(results_base, repo),
                        "--frontier-artifact-root",
                        _repo_rel(results_base, repo),
                        "--candidate-limit",
                        str(chain_limit),
                        "--action-summary",
                        "latest",
                    ],
                    "requires": receiver_repair_run_step_ids,
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 600,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": post_repair_refresh_report_ref,
                            "key": "schema",
                            "equals": FEEDBACK_REFRESH_SCHEMA,
                        },
                        {
                            "type": "json_false_authority",
                            "path": post_repair_refresh_report_ref,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [post_repair_refresh_report_ref],
                        "input_artifact_paths": [
                            _repo_rel(work_order_path, repo),
                            *child_queue_paths,
                        ],
                        "recursive": True,
                        "include_postcondition_paths": True,
                    },
                }
            )
            post_repair_refresh_planned = True
        queue_actuation_blockers: list[str] = []
        if not local_actions:
            queue_actuation_blockers.append("no_non_advisory_child_queue_actions")
        queue_actuation_blockers.extend(
            f"missing_child_queue_artifact:{key}"
            for key in _unique_strings(missing_queue_artifact_keys)
        )
        queue_actuation_blockers.extend(
            f"child_queue_not_runnable:{key}"
            for key in _unique_strings(blocked_child_queue_artifact_keys)
        )
        if len(child_queue_paths) < len(local_actions):
            queue_actuation_blockers.append("not_all_local_actions_bound_to_child_queues")
        queue_actuation_ready = bool(local_actions) and not queue_actuation_blockers
        emit_command = steps[0]["command"]
        for child_queue_path in _unique_strings(child_queue_paths):
            emit_command.extend(["--child-queue-artifact-path", child_queue_path])
        for missing_key in _unique_strings(missing_queue_artifact_keys):
            emit_command.extend(["--missing-queue-artifact-key", missing_key])
        if queue_actuation_ready:
            emit_command.append("--queue-actuation-ready")
        if post_repair_refresh_planned:
            emit_command.append("--post-repair-refresh-planned")
        metadata = {
            "schema": AUTONOMOUS_CHAIN_QUEUE_METADATA_SCHEMA,
            "chain_id": chain_id,
            "chain_family": row.get("chain_family"),
            "priority_score": row.get("priority_score"),
            "pipeline_stages": _autonomous_chain_pipeline_stages(row),
            "local_queue_actions": local_actions,
            "local_queue_action_count": len(local_actions),
            "advisory_actions": advisory_actions,
            "advisory_action_count": len(advisory_actions),
            "child_queue_artifact_paths": child_queue_paths,
            "child_queue_health_by_key": child_queue_health_by_key,
            "missing_queue_artifact_keys": _unique_strings(missing_queue_artifact_keys),
            "blocked_child_queue_artifact_keys": _unique_strings(
                blocked_child_queue_artifact_keys
            ),
            "queue_actuation_ready": queue_actuation_ready,
            "queue_actuation_blockers": _unique_strings(queue_actuation_blockers),
            "post_repair_refresh_planned": post_repair_refresh_planned,
            "post_repair_refresh_report_path": post_repair_refresh_report_ref,
            "source_artifact_paths": [
                _repo_rel(source_path, repo),
                *[
                    _repo_rel(_resolve_path(path, repo_root=repo), repo)
                    for path in artifact_paths_by_key.values()
                ],
            ],
            "allowed_use": "local_many_op_chain_actuation_queue_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            metadata,
            context=f"frontier_autonomous_chain_queue_metadata:{chain_id}",
        )
        experiments.append(
            {
                "id": experiment_id,
                "priority": priority,
                "status": "queued" if queue_actuation_ready else "frozen",
                "tags": [
                    "frontier-rate-attack",
                    "autonomous-many-op-chain",
                    "encoder-side-actuation",
                    "no-score-authority",
                    *(
                        []
                        if queue_actuation_ready
                        else ["blocked-child-queue-artifact-binding"]
                    ),
                ],
                "metadata": metadata,
                "steps": steps,
            }
        )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            **FALSE_AUTHORITY,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": dict.fromkeys(sorted(used_resource_kinds), 1),
            },
            "experiments": experiments,
        }
    )


def _targeted_component_prior_status(
    *,
    seed: Mapping[str, Any],
    component_summary: Mapping[str, Any],
    master_gradient: Mapping[str, Any],
    receiver_closed_correction_budget: Mapping[str, Any],
    repair_dynamics_palette_prior: Mapping[str, Any],
) -> dict[str, Any]:
    available: list[str] = []
    missing: list[str] = []
    blockers: list[str] = []
    for prior in _string_list(seed.get("required_priors")):
        if prior == "component_marginal_rows":
            present = component_summary.get("active") is True
        elif prior in {
            "receiver_closed_rate_budget",
            "byte_closed_materializer_context",
        }:
            present = receiver_closed_correction_budget.get("active") is True
        elif prior == "master_gradient_or_inverse_scorer":
            present = master_gradient.get("active") is True
        elif prior == "repair_dynamics_palette_prior":
            present = bool(repair_dynamics_palette_prior)
        else:
            present = False
        if present:
            available.append(prior)
        else:
            missing.append(prior)
            blockers.append(f"requires_{prior}")
    return {
        "available_priors": _unique_strings(available),
        "missing_priors": _unique_strings(missing),
        "prior_blockers": _unique_strings(blockers),
        **FALSE_AUTHORITY,
    }


def _repair_dynamics_prior_from_correction_budget(
    *,
    receiver_closed_correction_budget: Mapping[str, Any],
    repair_dynamics_palette_prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    priors: list[Mapping[str, Any]] = []
    if isinstance(repair_dynamics_palette_prior, Mapping) and repair_dynamics_palette_prior:
        require_no_truthy_authority_fields(
            repair_dynamics_palette_prior,
            context="targeted_component_correction_manual_repair_dynamics_palette_prior",
        )
        priors.append(repair_dynamics_palette_prior)
    for key in ("repair_dynamics_palette_prior", "repair_dynamics_prior"):
        value = receiver_closed_correction_budget.get(key)
        if isinstance(value, Mapping) and value:
            require_no_truthy_authority_fields(
                value,
                context=f"targeted_component_correction_receiver_budget_{key}",
            )
            priors.append(value)
    for index, row in enumerate(receiver_closed_correction_budget.get("rows") or []):
        if not isinstance(row, Mapping):
            continue
        for key in ("repair_dynamics_palette_prior", "repair_dynamics_prior"):
            value = row.get(key)
            if isinstance(value, Mapping) and value:
                require_no_truthy_authority_fields(
                    value,
                    context=(
                        "targeted_component_correction_receiver_budget_row_"
                        f"{index}_{key}"
                    ),
                )
                priors.append(value)
    return _aggregate_repair_dynamics_priors(
        priors,
        source="targeted_component_correction_repair_dynamics_prior_aggregate",
    )


def _repair_dynamics_targeted_component_family_seeds(
    prior: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if not prior:
        return ()
    family_counts = (
        prior.get("mode_family_counts")
        if isinstance(prior.get("mode_family_counts"), Mapping)
        else {}
    )
    has_color_bias = any(
        int(family_counts.get(key, 0) or 0)
        for key in ("blue_chroma", "luma_bias", "rgb_bias")
    )
    seeds: list[dict[str, Any]] = [
        {
            "correction_family": "repair_dynamics_frame0_palette_interaction_waterfill",
            "operation_levels": [
                "pixel",
                "region",
                "boundary",
                "frame",
                "pair",
                "batch",
                "scorer_axis",
            ],
            "priority_base": 99.0,
            "recommended_next_action": (
                "probe_frame0_palette_color_geometry_interactions_as_grouped_repair_atoms"
            ),
            "targeted_dimensions": [
                "pixel",
                "region",
                "boundary",
                "frame",
                "pair",
                "batch",
            ],
            "required_priors": [
                "component_marginal_rows",
                "repair_dynamics_palette_prior",
            ],
        }
    ]
    if has_color_bias:
        seeds.append(
            {
                "correction_family": "repair_dynamics_chroma_luma_bias_basis_expansion",
                "operation_levels": [
                    "pixel",
                    "region",
                    "frame",
                    "pair",
                    "batch",
                    "scorer_axis",
                ],
                "priority_base": 97.0,
                "recommended_next_action": (
                    "expand_chroma_luma_rgb_bias_modes_before_more_pixel_leaf_search"
                ),
                "targeted_dimensions": [
                    "pixel",
                    "region",
                    "frame",
                    "pair",
                    "batch",
                ],
                "required_priors": [
                    "component_marginal_rows",
                    "repair_dynamics_palette_prior",
                ],
            }
        )
    if prior.get("zero_frame1_modes") is True:
        seeds.append(
            {
                "correction_family": "repair_dynamics_frame1_counterfactual_null_probe",
                "operation_levels": [
                    "frame",
                    "pair",
                    "batch",
                    "scorer_axis",
                ],
                "priority_base": 94.0,
                "recommended_next_action": (
                    "measure_frame1_counterfactuals_to_classify_null_space_vs_search_gap"
                ),
                "targeted_dimensions": ["frame", "pair", "batch"],
                "required_priors": [
                    "component_marginal_rows",
                    "repair_dynamics_palette_prior",
                ],
            }
        )
    return tuple(seeds)


def _repair_dynamics_palette_context(prior: Mapping[str, Any]) -> dict[str, Any]:
    if not prior:
        return {}
    return {
        "schema": "frontier_rate_attack_repair_dynamics_palette_context.v1",
        "source": prior.get("source"),
        "source_prior_refs": list(prior.get("source_prior_refs") or []),
        "mode_count": prior.get("mode_count"),
        "non_identity_mode_count": prior.get("non_identity_mode_count"),
        "frame_mode_counts": dict(
            prior.get("frame_mode_counts")
            if isinstance(prior.get("frame_mode_counts"), Mapping)
            else {}
        ),
        "mode_family_counts": dict(
            prior.get("mode_family_counts")
            if isinstance(prior.get("mode_family_counts"), Mapping)
            else {}
        ),
        "frame0_non_identity_fraction": prior.get("frame0_non_identity_fraction"),
        "zero_frame1_modes": prior.get("zero_frame1_modes") is True,
        "dominant_dynamics_interpretation": prior.get(
            "dominant_dynamics_interpretation"
        ),
        "repair_waterfill_hints": list(prior.get("repair_waterfill_hints") or []),
        "allowed_use": "compact_repair_dynamics_context_for_grouped_acquisition_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _repair_dynamics_priority_bonus(
    *,
    family: str,
    prior: Mapping[str, Any],
) -> float:
    if not prior or not family.startswith("repair_dynamics_"):
        return 0.0
    mode_count = _finite_int_or_none(prior.get("mode_count")) or 0
    bonus = min(10.0, float(mode_count) / 2.0)
    frame0_fraction = prior.get("frame0_non_identity_fraction")
    if isinstance(frame0_fraction, (int, float)) and math.isfinite(float(frame0_fraction)):
        bonus += 6.0 * float(frame0_fraction)
    if prior.get("zero_frame1_modes") is True:
        bonus += 3.0
    if "counterfactual_null_probe" in family:
        bonus -= 2.0
    return bonus


def _repair_operator_portfolio(
    *,
    rows: Sequence[Mapping[str, Any]],
    repair_dynamics_prior: Mapping[str, Any],
) -> dict[str, Any]:
    families = _unique_strings(row.get("correction_family") for row in rows)
    stacks: list[dict[str, Any]] = []
    if repair_dynamics_prior:
        stacks.extend(
            [
                {
                    "stack_id": "frame0_palette_then_region_boundary_waterfill",
                    "operator_families": [
                        "repair_dynamics_frame0_palette_interaction_waterfill",
                        "segnet_posenet_waterfill_region_repair",
                    ],
                    "rationale": (
                        "measure global color_geometry calibration before spending "
                        "bytes on localized SegNet/PoseNet region repair"
                    ),
                    "requires_same_axis_component_response": True,
                    **FALSE_AUTHORITY,
                },
                {
                    "stack_id": "motion_then_palette_interaction_probe",
                    "operator_families": [
                        "pose_stable_pair_frame_motion_correction",
                        "repair_dynamics_frame0_palette_interaction_waterfill",
                    ],
                    "rationale": (
                        "separate pair-frame motion repair from frame0 color "
                        "calibration before trusting either marginal alone"
                    ),
                    "requires_same_axis_component_response": True,
                    **FALSE_AUTHORITY,
                },
                {
                    "stack_id": "inverse_basis_then_palette_family_expansion",
                    "operator_families": [
                        "inverse_scorer_cell_basis_expansion",
                        "repair_dynamics_chroma_luma_bias_basis_expansion",
                    ],
                    "rationale": (
                        "let inverse-scorer cells propose basis directions, then "
                        "test nearby chroma/luma/RGB repair atoms under MLX"
                    ),
                    "requires_same_axis_component_response": True,
                    **FALSE_AUTHORITY,
                },
            ]
        )
    if "full_video_batch_residual_budget_reallocation" in families:
        stacks.append(
            {
                "stack_id": "full_video_residual_then_component_waterfill",
                "operator_families": [
                    "full_video_batch_residual_budget_reallocation",
                    "segnet_posenet_waterfill_region_repair",
                ],
                "rationale": (
                    "test whether full-video residual budget movement composes "
                    "with localized component repair instead of increasing distortion"
                ),
                "requires_same_axis_component_response": True,
                **FALSE_AUTHORITY,
            }
        )
    return {
        "schema": "frontier_rate_attack_repair_operator_portfolio.v1",
        "operator_family_count": len(families),
        "operator_families": families,
        "stack_candidate_count": len(stacks),
        "stack_candidates": stacks,
        "acceptance_rule": (
            "compose_only_if_measured_delta_segnet_plus_delta_posenet_plus_delta_rate_"
            "is_negative_under_same_axis_for_the_stack"
        ),
        "allowed_use": "component_response_guided_repair_stack_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def build_frontier_targeted_component_correction_acquisition(
    *,
    operation_portfolio: Mapping[str, Any],
    receiver_closed_correction_budget: Mapping[str, Any],
    repair_dynamics_palette_prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn receiver-closed rate wins into component-guarded correction requests."""

    require_no_truthy_authority_fields(
        operation_portfolio,
        context="targeted_component_correction_operation_portfolio",
    )
    require_no_truthy_authority_fields(
        receiver_closed_correction_budget,
        context="targeted_component_correction_receiver_closed_budget",
    )
    component_summary = (
        operation_portfolio.get("component_behavior_summary")
        if isinstance(operation_portfolio.get("component_behavior_summary"), Mapping)
        else {}
    )
    master_gradient = (
        operation_portfolio.get("master_gradient_summary")
        if isinstance(operation_portfolio.get("master_gradient_summary"), Mapping)
        else {}
    )
    targeted_budget = (
        operation_portfolio.get("targeted_correction_budget_summary")
        if isinstance(operation_portfolio.get("targeted_correction_budget_summary"), Mapping)
        else {}
    )
    receiver_rows = [
        row
        for row in receiver_closed_correction_budget.get("rows") or []
        if isinstance(row, Mapping) and row.get("receiver_closed") is True
    ]
    repair_dynamics_prior = _repair_dynamics_prior_from_correction_budget(
        receiver_closed_correction_budget=receiver_closed_correction_budget,
        repair_dynamics_palette_prior=repair_dynamics_palette_prior,
    )
    repair_dynamics_context = _repair_dynamics_palette_context(repair_dynamics_prior)
    family_seeds = (
        *_TARGETED_COMPONENT_CORRECTION_FAMILY_SEEDS,
        *_repair_dynamics_targeted_component_family_seeds(repair_dynamics_prior),
    )
    rows: list[dict[str, Any]] = []
    for budget_row in receiver_rows:
        saved_bytes = _finite_int_or_none(budget_row.get("saved_bytes_at_risk")) or 0
        if saved_bytes <= 0:
            continue
        candidate_id = str(budget_row.get("candidate_id") or "unknown_candidate")
        target_kind = str(budget_row.get("target_kind") or "unknown_target")
        submission_dir = str(budget_row.get("submission_dir") or "")
        candidate_archive_path = str(
            budget_row.get("candidate_archive_path")
            or budget_row.get("archive_path")
            or ""
        )
        candidate_inflate_sh_path = str(
            budget_row.get("candidate_inflate_sh_path")
            or budget_row.get("inflate_sh_path")
            or (f"{submission_dir}/inflate.sh" if submission_dir else "")
        )
        source_archive_path = str(budget_row.get("source_archive_path") or "")
        source_submission_dir = str(budget_row.get("source_submission_dir") or "")
        source_inflate_sh_path = str(budget_row.get("source_inflate_sh_path") or "")
        closure_report_path = str(budget_row.get("closure_report_path") or "")
        bridge_report_path = str(
            budget_row.get("paired_exact_readiness_bridge_report_path") or ""
        )
        rate_credit = _rate_credit_score_units_for_saved_bytes(saved_bytes)
        for seed in family_seeds:
            prior_status = _targeted_component_prior_status(
                seed=seed,
                component_summary=component_summary,
                master_gradient=master_gradient,
                receiver_closed_correction_budget=receiver_closed_correction_budget,
                repair_dynamics_palette_prior=repair_dynamics_prior,
            )
            budget_spend_blockers = [
                "candidate_specific_local_cpu_component_eval_required_before_budget_spend",
                "candidate_specific_mlx_or_exact_axis_component_response_required_before_spend",
                "exact_auth_eval_required_before_score_or_promotion_claim",
            ]
            if not submission_dir:
                budget_spend_blockers.append("submission_dir_missing_for_component_eval")
            if not candidate_archive_path:
                budget_spend_blockers.append(
                    "candidate_archive_path_missing_for_component_eval"
                )
            if not candidate_inflate_sh_path:
                budget_spend_blockers.append(
                    "candidate_inflate_sh_path_missing_for_component_eval"
                )
            if budget_row.get("active_rate_floor_blocked") is True:
                budget_spend_blockers.append(
                    "active_rate_floor_override_required_before_exact_dispatch"
                )
            if component_summary.get("active") is not True:
                budget_spend_blockers.append(
                    "segnet_posenet_component_behavior_rows_required_before_allocation"
                )
            family = str(seed.get("correction_family") or "unknown_correction_family")
            repair_dynamics_family = family.startswith("repair_dynamics_")
            acquisition_id = (
                "targeted_component_correction_"
                f"{_slug_token(candidate_id)}_{_slug_token(family)}"
            )
            rows.append(
                {
                    "schema": TARGETED_COMPONENT_CORRECTION_ACQUISITION_ROW_SCHEMA,
                    "acquisition_id": acquisition_id,
                    "candidate_id": candidate_id,
                    "target_kind": target_kind,
                    "correction_family": family,
                    "operation_levels": list(seed.get("operation_levels") or []),
                    "targeted_dimensions": list(seed.get("targeted_dimensions") or []),
                    "saved_bytes_budget": saved_bytes,
                    "receiver_closed_saved_bytes": saved_bytes,
                    "estimated_rate_credit_score_units": rate_credit,
                    "estimated_rate_credit_byte_delta": -saved_bytes,
                    "receiver_closed_budget_gate": budget_row.get(
                        "correction_budget_gate"
                    ),
                    "rate_packet_manifest_path": budget_row.get(
                        "rate_packet_manifest_path"
                    ),
                    "parent_rate_packet_manifest_path": budget_row.get(
                        "parent_rate_packet_manifest_path"
                    ),
                    "candidate_compact_selector_codec": budget_row.get(
                        "candidate_compact_selector_codec"
                    ),
                    "parent_compact_selector_codec": budget_row.get(
                        "parent_compact_selector_codec"
                    ),
                    "selector_policy_mode": budget_row.get("selector_policy_mode"),
                    "archive_byte_delta_vs_parent": budget_row.get(
                        "archive_byte_delta_vs_parent"
                    ),
                    "selector_payload_wire_bytes": budget_row.get(
                        "selector_payload_wire_bytes"
                    ),
                    "parent_selector_payload_wire_bytes": budget_row.get(
                        "parent_selector_payload_wire_bytes"
                    ),
                    "selector_payload_wire_delta_bytes": budget_row.get(
                        "selector_payload_wire_delta_bytes"
                    ),
                    "selector_code_bits_total": budget_row.get(
                        "selector_code_bits_total"
                    ),
                    "parent_selector_code_bits_total": budget_row.get(
                        "parent_selector_code_bits_total"
                    ),
                    "selector_avg_bits_per_pair": budget_row.get(
                        "selector_avg_bits_per_pair"
                    ),
                    "parent_selector_avg_bits_per_pair": budget_row.get(
                        "parent_selector_avg_bits_per_pair"
                    ),
                    "palette_size": budget_row.get("palette_size"),
                    "n_pairs": budget_row.get("n_pairs"),
                    "compact_palette_mode_ids": budget_row.get(
                        "compact_palette_mode_ids"
                    ),
                    "entropy_position": budget_row.get("entropy_position"),
                    "ready_for_budget_spend": False,
                    "budget_spend_allowed": False,
                    "submission_dir": submission_dir or None,
                    "archive_path": candidate_archive_path or None,
                    "archive_sha256": (
                        budget_row.get("archive_sha256")
                        or budget_row.get("candidate_archive_sha256")
                    ),
                    "archive_bytes": (
                        budget_row.get("archive_bytes")
                        or budget_row.get("candidate_archive_bytes")
                    ),
                    "candidate_archive_path": candidate_archive_path or None,
                    "candidate_archive_sha256": (
                        budget_row.get("candidate_archive_sha256")
                        or budget_row.get("archive_sha256")
                    ),
                    "candidate_archive_bytes": (
                        budget_row.get("candidate_archive_bytes")
                        or budget_row.get("archive_bytes")
                    ),
                    "inflate_sh_path": candidate_inflate_sh_path or None,
                    "source_archive_path": source_archive_path or None,
                    "source_archive_sha256": budget_row.get("source_archive_sha256"),
                    "source_archive_bytes": budget_row.get("source_archive_bytes"),
                    "source_submission_dir": source_submission_dir or None,
                    "source_inflate_sh_path": source_inflate_sh_path or None,
                    "reference_component_eval_context": dict(
                        budget_row.get("reference_component_eval_context")
                        if isinstance(
                            budget_row.get("reference_component_eval_context"),
                            Mapping,
                        )
                        else {}
                    ),
                    "closure_report_path": closure_report_path,
                    "paired_exact_readiness_bridge_report_path": bridge_report_path,
                    "component_behavior_active": component_summary.get("active") is True,
                    "best_component_behavior_candidate_id": component_summary.get(
                        "best_candidate_id"
                    ),
                    "best_component_deltas": component_summary.get(
                        "best_component_deltas"
                    ),
                    "component_marginal_status_counts": component_summary.get(
                        "component_marginal_status_counts"
                    ),
                    "component_behavior_context": {
                        "component_summary_schema": component_summary.get("schema"),
                        "best_candidate_id": component_summary.get(
                            "best_candidate_id"
                        ),
                        "best_score_delta_vs_baseline": component_summary.get(
                            "best_score_delta_vs_baseline"
                        ),
                        "best_component_deltas": component_summary.get(
                            "best_component_deltas"
                        ),
                        "best_selected_pair_indices": component_summary.get(
                            "best_selected_pair_indices"
                        ),
                        "component_marginal_status_counts": component_summary.get(
                            "component_marginal_status_counts"
                        ),
                        **FALSE_AUTHORITY,
                    },
                    "receiver_closed_budget_context": {
                        "receiver_closed_correction_budget_schema": (
                            receiver_closed_correction_budget.get("schema")
                        ),
                        "receiver_closed_saved_bytes_total": saved_bytes,
                        "receiver_closed_rate_credit_score_units_total": rate_credit,
                        "source_candidate_id": candidate_id,
                        "ready_for_budget_spend": False,
                        "budget_spend_allowed": False,
                        **FALSE_AUTHORITY,
                    },
                    "repair_dynamics_prior_active": bool(repair_dynamics_prior),
                    "repair_dynamics_palette_prior": (
                        dict(repair_dynamics_prior) if repair_dynamics_family else {}
                    ),
                    "repair_dynamics_context": (
                        dict(repair_dynamics_context) if repair_dynamics_family else {}
                    ),
                    "prior_status": prior_status,
                    "budget_spend_gate": {
                        "schema": (
                            "frontier_rate_attack_targeted_component_correction_"
                            "budget_spend_gate.v1"
                        ),
                        "ready_for_budget_spend": False,
                        "required_before_spend": _unique_strings(
                            [
                                *budget_spend_blockers,
                                *prior_status["prior_blockers"],
                            ]
                        ),
                        "acceptance_rule": (
                            "accept_only_if_measured_delta_segnet_plus_delta_posenet_"
                            "plus_delta_rate_is_negative_under_same_axis"
                        ),
                        "max_rate_credit_score_units": rate_credit,
                        "max_extra_archive_bytes_before_rate_credit_exhausted": (
                            saved_bytes
                        ),
                        "budget_spend_allowed": False,
                        **FALSE_AUTHORITY,
                    },
                    "queue_actionable": bool(
                        submission_dir
                        and candidate_archive_path
                        and candidate_inflate_sh_path
                    ),
                    "queue_consumer": "frontier_targeted_component_correction_queue",
                    "recommended_next_action": seed.get("recommended_next_action"),
                    "wire_in_hooks": _targeted_component_correction_wire_hooks(family),
                    "blockers": _unique_strings(
                        [*budget_spend_blockers, *prior_status["prior_blockers"]]
                    ),
                    "priority_score": (
                        float(seed.get("priority_base") or 0.0)
                        + float(saved_bytes) / 32.0
                        + rate_credit * 1_000_000.0
                        + _repair_dynamics_priority_bonus(
                            family=family,
                            prior=repair_dynamics_prior,
                        )
                        - float(len(prior_status["prior_blockers"])) * 4.0
                    ),
                    "allowed_use": (
                        "receiver_closed_rate_budget_component_correction_acquisition_only"
                    ),
                    "forbidden_use": (
                        "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                    ),
                    **FALSE_AUTHORITY,
                }
            )
    rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get("priority_score") or 0.0),
            str(row.get("acquisition_id") or ""),
        ),
    )
    unique_budget_candidates = _unique_strings(row.get("candidate_id") for row in rows)
    blockers: list[str] = []
    if not receiver_rows:
        blockers.append("no_receiver_closed_rate_budget_rows")
    if not rows:
        blockers.append("no_targeted_component_correction_rows")
    if component_summary.get("active") is not True:
        blockers.append("segnet_posenet_component_behavior_rows_required")
    if repair_dynamics_prior:
        blockers.append("repair_dynamics_palette_probe_required_before_budget_spend")
    if not any(row.get("queue_actionable") is True for row in rows):
        blockers.append("no_component_eval_queue_actionable_submission_dirs")
    blockers.append("component_eval_required_before_budget_spend")
    blockers.append("exact_auth_eval_required_before_score_or_promotion_claim")
    saved_by_candidate: dict[str, int] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        saved_by_candidate.setdefault(candidate_id, int(row.get("saved_bytes_budget") or 0))
    total_saved_bytes = sum(saved_by_candidate.values())
    return {
        "schema": TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA,
        "generated_at_utc": _utc_now(),
        "active": bool(rows),
        "operation_portfolio_schema": operation_portfolio.get("schema"),
        "receiver_closed_correction_budget_schema": (
            receiver_closed_correction_budget.get("schema")
        ),
        "targeted_correction_budget_summary_schema": targeted_budget.get("schema"),
        "receiver_closed_candidate_count": len(unique_budget_candidates),
        "component_correction_row_count": len(rows),
        "row_count": len(rows),
        "queue_actionable_row_count": sum(
            1 for row in rows if row.get("queue_actionable") is True
        ),
        "queue_actionable_acquisition_count": sum(
            1 for row in rows if row.get("queue_actionable") is True
        ),
        "ready_for_budget_spend_count": 0,
        "receiver_closed_saved_bytes_total": total_saved_bytes,
        "estimated_rate_credit_score_units_total": sum(
            _rate_credit_score_units_for_saved_bytes(saved)
            for saved in saved_by_candidate.values()
        ),
        "receiver_closed_rate_credit_score_units_total": sum(
            _rate_credit_score_units_for_saved_bytes(saved)
            for saved in saved_by_candidate.values()
        ),
        "component_behavior_active": component_summary.get("active") is True,
        "best_component_behavior_candidate_id": component_summary.get(
            "best_candidate_id"
        ),
        "best_component_behavior_score_delta_vs_baseline": component_summary.get(
            "best_score_delta_vs_baseline"
        ),
        "master_gradient_active": master_gradient.get("active") is True,
        "repair_dynamics_prior_active": bool(repair_dynamics_prior),
        "repair_dynamics_palette_prior": dict(repair_dynamics_prior),
        "repair_dynamics_palette_probe_count": sum(
            1
            for row in rows
            if str(row.get("correction_family") or "").startswith("repair_dynamics_")
        ),
        "repair_dynamics_repair_waterfill_hints": list(
            repair_dynamics_prior.get("repair_waterfill_hints") or []
        ),
        "repair_operator_portfolio": _repair_operator_portfolio(
            rows=rows,
            repair_dynamics_prior=repair_dynamics_prior,
        ),
        "targeted_dimensions": _unique_strings(
            [
                dimension
                for row in rows
                for dimension in _string_list(row.get("targeted_dimensions"))
            ]
        ),
        "top_acquisition_ids": [
            str(row.get("acquisition_id") or "") for row in rows[:8]
        ],
        "top_correction_families": _unique_strings(
            row.get("correction_family") for row in rows[:8]
        ),
        "blockers": _unique_strings(blockers),
        "recommended_next_action": (
            "run_component_eval_queue_for_receiver_closed_budget_candidates_then_"
            "materialize_only_corrections_with_negative_measured_lagrangian_delta"
            if rows
            else "close_receiver_static_runtime_budget_before_component_correction_acquisition"
        ),
        "rows": rows,
        "allowed_use": "queue_owned_targeted_component_correction_acquisition_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _targeted_component_correction_queue_metadata(
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": TARGETED_COMPONENT_CORRECTION_QUEUE_METADATA_SCHEMA,
        "targeted_component_correction_acquisition_schema": acquisition.get("schema"),
        "active": acquisition.get("active") is True,
        "row_count": acquisition.get("row_count")
        or acquisition.get("component_correction_row_count"),
        "queue_actionable_acquisition_count": (
            acquisition.get("queue_actionable_acquisition_count")
            or acquisition.get("queue_actionable_row_count")
        ),
        "receiver_closed_candidate_count": acquisition.get(
            "receiver_closed_candidate_count"
        ),
        "component_correction_row_count": acquisition.get(
            "component_correction_row_count"
        ),
        "queue_actionable_row_count": acquisition.get("queue_actionable_row_count"),
        "ready_for_budget_spend_count": acquisition.get("ready_for_budget_spend_count"),
        "receiver_closed_saved_bytes_total": acquisition.get(
            "receiver_closed_saved_bytes_total"
        ),
        "estimated_rate_credit_score_units_total": acquisition.get(
            "estimated_rate_credit_score_units_total"
        ),
        "component_behavior_active": acquisition.get("component_behavior_active") is True,
        "master_gradient_active": acquisition.get("master_gradient_active") is True,
        "repair_dynamics_prior_active": (
            acquisition.get("repair_dynamics_prior_active") is True
        ),
        "repair_dynamics_palette_probe_count": acquisition.get(
            "repair_dynamics_palette_probe_count"
        ),
        "repair_dynamics_repair_waterfill_hints": list(
            acquisition.get("repair_dynamics_repair_waterfill_hints") or []
        ),
        "top_acquisition_ids": list(acquisition.get("top_acquisition_ids") or []),
        "top_correction_families": list(acquisition.get("top_correction_families") or []),
        "blockers": list(acquisition.get("blockers") or []),
        "allowed_use": "queue_metadata_pointer_to_targeted_component_correction_acquisition",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _blocked_targeted_component_correction_queue(
    *,
    repo_root: Path,
    targeted_component_correction_acquisition: Mapping[str, Any],
    targeted_component_correction_acquisition_path: str | Path,
    results_root: str | Path,
    queue_id: str,
    candidate_limit: int,
    target_optimization_profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    acquisition_path = _resolve_path(
        targeted_component_correction_acquisition_path,
        repo_root=repo_root,
    )
    results_base = _resolve_path(str(results_root), repo_root=repo_root)
    acquisition_metadata = _targeted_component_correction_queue_metadata(
        targeted_component_correction_acquisition
    )
    target_profile_metadata = (
        dict(target_optimization_profile_metadata)
        if isinstance(target_optimization_profile_metadata, Mapping)
        and target_optimization_profile_metadata
        else {}
    )
    if target_profile_metadata:
        require_no_truthy_authority_fields(
            target_profile_metadata,
            context=(
                "blocked_targeted_component_correction_queue_"
                "target_optimization_profile"
            ),
        )
    blockers = _unique_strings(
        [
            "targeted_component_correction_selected_rows_empty",
            "no_queue_actionable_targeted_component_correction_rows",
            *list(targeted_component_correction_acquisition.get("blockers") or []),
        ]
    )
    queue_metadata = {
        "schema": (
            "frontier_rate_attack_targeted_component_correction_queue_blocked_"
            "metadata.v1"
        ),
        "targeted_component_correction_acquisition": acquisition_metadata,
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "queue_actuation_ready": False,
        "queue_actuation_blockers": blockers,
        "selected_row_count": 0,
        "candidate_limit": candidate_limit,
        "results_root": _repo_rel(results_base, repo_root),
        "source_artifact_paths": [_repo_rel(acquisition_path, repo_root)],
        "allowed_use": "blocked_targeted_component_correction_queue_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        queue_metadata,
        context="blocked_targeted_component_correction_queue_metadata",
    )
    experiment_metadata = {
        **queue_metadata,
        "schema": TARGETED_COMPONENT_CORRECTION_QUEUE_METADATA_SCHEMA,
        "pipeline_side": "targeted_component_correction_acquisition",
        "budget_spend_ready": False,
        "budget_spend_allowed": False,
        "component_response_harvest_available": False,
        "missing_response_reason": "no_selected_targeted_component_correction_rows",
        "allowed_use": "blocked_targeted_component_correction_queue_metadata_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        experiment_metadata,
        context="blocked_targeted_component_correction_queue_experiment_metadata",
    )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": 0,
                    "local_io_heavy": 1,
                    "local_mlx": 0,
                    "modal_cpu": 0,
                    "modal_gpu": 0,
                },
            },
            "metadata": queue_metadata,
            "experiments": [
                {
                    "id": "inspect_empty_targeted_component_correction_selection",
                    "priority": 1,
                    "status": "frozen",
                    "tags": [
                        "frontier-rate-attack",
                        "targeted-component-correction",
                        "blocked-empty-selection",
                        "no-score-authority",
                    ],
                    "metadata": experiment_metadata,
                    "steps": [
                        {
                            "id": "inspect_targeted_component_correction_acquisition_blockers",
                            "kind": "command",
                            "command": [
                                ".venv/bin/python",
                                "-m",
                                "json.tool",
                                _repo_rel(acquisition_path, repo_root),
                            ],
                            "resources": {"kind": "local_io_heavy"},
                            "timeout_seconds": 60,
                            "telemetry": {
                                "input_artifact_paths": [
                                    _repo_rel(acquisition_path, repo_root)
                                ],
                                "include_postcondition_paths": True,
                            },
                        }
                    ],
                }
            ],
            **FALSE_AUTHORITY,
        }
    )


def build_frontier_targeted_component_correction_work_order(
    *,
    targeted_component_correction_acquisition: Mapping[str, Any],
    acquisition_id: str,
    repo_root: str | Path | None = None,
    target_optimization_profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single receiver-budget component-correction work order."""

    require_no_truthy_authority_fields(
        targeted_component_correction_acquisition,
        context="targeted_component_correction_acquisition",
    )
    row = _targeted_component_correction_row_by_id(
        targeted_component_correction_acquisition,
        acquisition_id,
    )
    require_no_truthy_authority_fields(
        row,
        context=f"targeted_component_correction_row:{acquisition_id}",
    )
    candidate_id = str(row.get("candidate_id") or "")
    sibling_rows = [
        sibling
        for sibling in targeted_component_correction_acquisition.get("rows") or []
        if isinstance(sibling, Mapping)
        and str(sibling.get("candidate_id") or "") == candidate_id
    ]
    reference_context = (
        _targeted_component_reference_eval_context_for_queue(
            row,
            repo_root=Path(repo_root),
        )
        if repo_root is not None
        else dict(
            row.get("reference_component_eval_context")
            if isinstance(row.get("reference_component_eval_context"), Mapping)
            else {}
        )
    )
    source_submission_dir = (
        row.get("source_submission_dir") or reference_context.get("source_submission_dir")
    )
    source_inflate_sh_path = (
        row.get("source_inflate_sh_path")
        or reference_context.get("source_inflate_sh_path")
        or (
            f"{source_submission_dir}/inflate.sh"
            if source_submission_dir
            else None
        )
    )
    target_profile_metadata = _target_profile_metadata_from_payloads(
        target_optimization_profile_metadata,
        row,
        targeted_component_correction_acquisition,
        context=f"targeted_component_correction_work_order:{acquisition_id}",
    )
    work_order = {
        "schema": TARGETED_COMPONENT_CORRECTION_WORK_ORDER_SCHEMA,
        "generated_at_utc": str(
            targeted_component_correction_acquisition.get("generated_at_utc")
            or _utc_now()
        ),
        "acquisition_id": acquisition_id,
        "candidate_id": candidate_id,
        "target_kind": row.get("target_kind"),
        "correction_family": row.get("correction_family"),
        "operation_levels": list(row.get("operation_levels") or []),
        "targeted_dimensions": list(row.get("targeted_dimensions") or []),
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "sibling_correction_families": _unique_strings(
            sibling.get("correction_family") for sibling in sibling_rows
        ),
        "saved_bytes_budget": row.get("saved_bytes_budget"),
        "receiver_closed_saved_bytes": row.get("receiver_closed_saved_bytes"),
        "estimated_rate_credit_score_units": row.get(
            "estimated_rate_credit_score_units"
        ),
        **_targeted_rate_packet_context_fields(row),
        "receiver_closed_rate_packet_context": _targeted_rate_packet_context(row),
        "submission_dir": row.get("submission_dir"),
        "archive_path": row.get("archive_path"),
        "inflate_sh_path": row.get("inflate_sh_path"),
        "source_archive_path": (
            row.get("source_archive_path") or reference_context.get("source_archive_path")
        ),
        "source_archive_sha256": (
            row.get("source_archive_sha256")
            or reference_context.get("source_archive_sha256")
        ),
        "source_archive_bytes": (
            row.get("source_archive_bytes")
            or reference_context.get("source_archive_bytes")
        ),
        "source_submission_dir": source_submission_dir,
        "source_inflate_sh_path": source_inflate_sh_path,
        "reference_component_eval_context": dict(reference_context),
        "closure_report_path": row.get("closure_report_path"),
        "paired_exact_readiness_bridge_report_path": row.get(
            "paired_exact_readiness_bridge_report_path"
        ),
        "repair_dynamics_prior_active": row.get("repair_dynamics_prior_active") is True,
        "repair_dynamics_palette_prior": dict(
            row.get("repair_dynamics_palette_prior")
            if isinstance(row.get("repair_dynamics_palette_prior"), Mapping)
            else {}
        ),
        "repair_dynamics_context": dict(
            row.get("repair_dynamics_context")
            if isinstance(row.get("repair_dynamics_context"), Mapping)
            else {}
        ),
        "budget_spend_gate": row.get("budget_spend_gate"),
        "command_hints": _targeted_component_correction_command_hints(row),
        "lagrangian_acceptance_rule": {
            "schema": "frontier_rate_attack_component_correction_lagrangian_rule.v1",
            "objective": "minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes",
            "negative_delta_is_better": True,
            "receiver_closed_rate_budget_bytes": row.get("saved_bytes_budget"),
            "component_eval_required": True,
            "exact_auth_eval_required_before_score_claim": True,
            **FALSE_AUTHORITY,
        },
        "wire_in_hooks": dict(
            row.get("wire_in_hooks") if isinstance(row.get("wire_in_hooks"), Mapping) else {}
        ),
        "component_eval_plan": {
            "schema": "frontier_rate_attack_targeted_component_eval_plan.v1",
            "local_cpu_advisory_required": True,
            "reference_local_cpu_advisory_required": True,
            "reference_role": "receiver_closed_source_reference",
            "mlx_response_required_or_exact_axis_component_trace": True,
            "acceptance_rule": (
                "use local CPU/MLX as acquisition signal only; exact auth-axis "
                "evaluation remains required before score or promotion claims"
            ),
            **FALSE_AUTHORITY,
        },
        "response_harvest_contract": {
            "schema": (
                "frontier_rate_attack_targeted_component_correction_response_"
                "contract.v1"
            ),
            "required_delta_fields": [
                "segnet_delta",
                "posenet_delta",
                (
                    "archive_byte_delta_vs_receiver_closed_candidate "
                    "or rate_delta_vs_receiver_closed_candidate"
                ),
            ],
            "objective": (
                "measured_delta_segnet_plus_delta_posenet_plus_correction_rate_"
                "spend_minus_receiver_closed_rate_credit"
            ),
            "negative_delta_is_local_acquisition_candidate": True,
            "budget_spend_allowed": False,
            "exact_auth_eval_required_before_score_claim": True,
            **FALSE_AUTHORITY,
        },
        "candidate_family_rows": [
            {
                "acquisition_id": sibling.get("acquisition_id"),
                "correction_family": sibling.get("correction_family"),
                "operation_levels": list(sibling.get("operation_levels") or []),
                "targeted_dimensions": list(sibling.get("targeted_dimensions") or []),
                "priority_score": sibling.get("priority_score"),
                "blockers": list(sibling.get("blockers") or []),
                **FALSE_AUTHORITY,
            }
            for sibling in sibling_rows
        ],
        "allowed_use": "targeted_component_correction_work_order_for_local_acquisition_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        work_order,
        context=f"targeted_component_correction_work_order:{acquisition_id}",
    )
    return work_order


def _first_finite_float(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = _finite_float_or_none(payload.get(key))
        if value is not None:
            return value
    return None


def _first_finite_int(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> int | None:
    for key in keys:
        value = _finite_int_or_none(payload.get(key))
        if value is not None:
            return value
    return None


def _advisory_archive_size_bytes(payload: Mapping[str, Any]) -> int | None:
    for key in ("archive_size_bytes", "archive_bytes", "compressed_size", "bytes"):
        value = _finite_int_or_none(payload.get(key))
        if value is not None and value > 0:
            return value
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        value = _finite_int_or_none(provenance.get("archive_size_bytes"))
        if value is not None and value > 0:
            return value
    return None


def _advisory_segnet_score_units(payload: Mapping[str, Any]) -> float | None:
    raw = _first_finite_float(
        payload,
        ("avg_segnet_dist", "segnet_dist", "segnet_distortion"),
    )
    if raw is not None:
        return 100.0 * raw
    return _first_finite_float(
        payload,
        ("score_seg_contribution", "segnet_score_contribution"),
    )


def _advisory_posenet_score_units(payload: Mapping[str, Any]) -> float | None:
    raw = _first_finite_float(
        payload,
        ("avg_posenet_dist", "pose_dist", "pose_distortion", "posenet_dist"),
    )
    if raw is not None and raw >= 0.0:
        return math.sqrt(10.0 * raw)
    return _first_finite_float(
        payload,
        ("score_pose_contribution", "posenet_score_contribution"),
    )


def _advisory_rate_score_units(payload: Mapping[str, Any]) -> float | None:
    value = _first_finite_float(
        payload,
        ("score_rate_contribution", "rate_score_contribution", "rate"),
    )
    if value is not None:
        return value
    archive_bytes = _advisory_archive_size_bytes(payload)
    if archive_bytes is None:
        return None
    return rate_delta_for_archive_byte_delta(archive_bytes)


def _advisory_total_score_units(payload: Mapping[str, Any]) -> float | None:
    value = _first_finite_float(
        payload,
        ("canonical_score", "score_recomputed_from_components", "final_score"),
    )
    if value is not None:
        return value
    seg = _advisory_segnet_score_units(payload)
    pose = _advisory_posenet_score_units(payload)
    rate = _advisory_rate_score_units(payload)
    if seg is None or pose is None or rate is None:
        return None
    return seg + pose + rate


def _advisory_axis_score_terms(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "score_units": _advisory_total_score_units(payload),
        "segnet_score_units": _advisory_segnet_score_units(payload),
        "posenet_score_units": _advisory_posenet_score_units(payload),
        "rate_score_units": _advisory_rate_score_units(payload),
        "archive_size_bytes": _advisory_archive_size_bytes(payload),
    }


def _targeted_component_response_delta_source(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    for key in (
        "targeted_component_correction_response",
        "component_correction_response",
        "component_deltas",
        "component_axis_deltas",
        "measured_component_deltas",
        "lagrangian_response",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def _targeted_component_response_delta_terms(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    source = _targeted_component_response_delta_source(payload)
    segnet_delta = _first_finite_float(
        source,
        (
            "segnet_delta",
            "segnet_score_delta",
            "score_seg_delta",
            "delta_segnet",
            "delta_segnet_score_units",
        ),
    )
    posenet_delta = _first_finite_float(
        source,
        (
            "posenet_delta",
            "posenet_score_delta",
            "score_pose_delta",
            "delta_posenet",
            "delta_posenet_score_units",
        ),
    )
    correction_rate_delta = _first_finite_float(
        source,
        (
            "rate_delta_vs_receiver_closed_candidate",
            "correction_rate_spend_score_units",
            "rate_delta_score_units",
            "rate_delta",
            "score_rate_delta",
        ),
    )
    added_bytes = _first_finite_int(
        source,
        (
            "archive_byte_delta_vs_receiver_closed_candidate",
            "correction_added_archive_bytes",
            "added_archive_bytes",
            "archive_byte_delta",
        ),
    )
    if correction_rate_delta is None and added_bytes is not None:
        correction_rate_delta = rate_delta_for_archive_byte_delta(added_bytes)
    return {
        "segnet_delta_score_units": segnet_delta,
        "posenet_delta_score_units": posenet_delta,
        "correction_rate_delta_score_units": correction_rate_delta,
        "correction_added_archive_bytes": added_bytes,
        "delta_source_keys": sorted(str(key) for key in source),
    }


def _targeted_component_response_paired_delta_terms(
    *,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_role: str,
) -> dict[str, Any]:
    candidate_seg = _advisory_segnet_score_units(candidate)
    reference_seg = _advisory_segnet_score_units(reference)
    candidate_pose = _advisory_posenet_score_units(candidate)
    reference_pose = _advisory_posenet_score_units(reference)
    candidate_bytes = _advisory_archive_size_bytes(candidate)
    reference_bytes = _advisory_archive_size_bytes(reference)
    archive_delta = (
        candidate_bytes - reference_bytes
        if candidate_bytes is not None and reference_bytes is not None
        else None
    )
    receiver_closed_rate_delta = (
        rate_delta_for_archive_byte_delta(archive_delta)
        if archive_delta is not None
        else None
    )
    if reference_role == "correction_spend_reference":
        correction_added_bytes = archive_delta
        correction_rate_delta = receiver_closed_rate_delta
    else:
        correction_added_bytes = 0
        correction_rate_delta = 0.0
    return {
        "segnet_delta_score_units": (
            None
            if candidate_seg is None or reference_seg is None
            else candidate_seg - reference_seg
        ),
        "posenet_delta_score_units": (
            None
            if candidate_pose is None or reference_pose is None
            else candidate_pose - reference_pose
        ),
        "correction_rate_delta_score_units": correction_rate_delta,
        "correction_added_archive_bytes": correction_added_bytes,
        "paired_reference_role": reference_role,
        "candidate_segnet_score_units": candidate_seg,
        "reference_segnet_score_units": reference_seg,
        "candidate_posenet_score_units": candidate_pose,
        "reference_posenet_score_units": reference_pose,
        "candidate_archive_size_bytes": candidate_bytes,
        "reference_archive_size_bytes": reference_bytes,
        "receiver_closed_archive_byte_delta_vs_reference": archive_delta,
        "receiver_closed_rate_delta_score_units": receiver_closed_rate_delta,
        "delta_source_keys": sorted(
            [
                *(f"candidate.{key}" for key in candidate),
                *(f"reference.{key}" for key in reference),
            ]
        ),
    }


def _targeted_component_response_mlx_cpu_drift_terms(
    *,
    mlx: Mapping[str, Any],
    local_cpu: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure absolute MLX-vs-local-CPU scorer drift for the same artifact."""

    mlx_terms = _advisory_axis_score_terms(mlx)
    cpu_terms = _advisory_axis_score_terms(local_cpu)
    out: dict[str, Any] = {
        "schema": "targeted_component_mlx_cpu_absolute_drift_terms.v1",
        "mlx_axis_score_terms": mlx_terms,
        "local_cpu_axis_score_terms": cpu_terms,
        "delta_source_keys": sorted(
            [
                *(f"mlx.{key}" for key in mlx),
                *(f"local_cpu.{key}" for key in local_cpu),
            ]
        ),
    }
    component_abs_sum = 0.0
    component_count = 0
    for key in (
        "score_units",
        "segnet_score_units",
        "posenet_score_units",
        "rate_score_units",
    ):
        mlx_value = mlx_terms.get(key)
        cpu_value = cpu_terms.get(key)
        delta_key = f"mlx_minus_local_cpu_{key}"
        delta = (
            None
            if mlx_value is None or cpu_value is None
            else float(mlx_value) - float(cpu_value)
        )
        out[delta_key] = delta
        if key != "score_units" and delta is not None:
            component_abs_sum += abs(float(delta))
            component_count += 1
    out["component_abs_drift_sum_score_units"] = (
        component_abs_sum if component_count else None
    )
    return out


def _targeted_component_response_paired_mlx_cpu_delta_drift_terms(
    *,
    mlx_paired: Mapping[str, Any] | None,
    local_cpu_paired: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Measure whether MLX preserves the CPU candidate-vs-reference delta."""

    if mlx_paired is None or local_cpu_paired is None:
        return None
    out: dict[str, Any] = {
        "schema": "targeted_component_mlx_cpu_paired_delta_drift_terms.v1",
        "delta_source_keys": sorted(
            [
                *(f"mlx_paired.{key}" for key in mlx_paired),
                *(f"local_cpu_paired.{key}" for key in local_cpu_paired),
            ]
        ),
    }
    lagrangian_sum = 0.0
    available = True
    for key in (
        "segnet_delta_score_units",
        "posenet_delta_score_units",
        "receiver_closed_rate_delta_score_units",
    ):
        mlx_value = _finite_float_or_none(mlx_paired.get(key))
        cpu_value = _finite_float_or_none(local_cpu_paired.get(key))
        delta_key = f"mlx_minus_local_cpu_{key}"
        if mlx_value is None or cpu_value is None:
            out[delta_key] = None
            available = False
            continue
        delta = float(mlx_value) - float(cpu_value)
        out[delta_key] = delta
        lagrangian_sum += delta
    out["mlx_minus_local_cpu_paired_lagrangian_delta_score_units"] = (
        lagrangian_sum if available else None
    )
    out["paired_delta_abs_drift_sum_score_units"] = (
        sum(
            abs(float(out[f"mlx_minus_local_cpu_{key}"]))
            for key in (
                "segnet_delta_score_units",
                "posenet_delta_score_units",
                "receiver_closed_rate_delta_score_units",
            )
            if out.get(f"mlx_minus_local_cpu_{key}") is not None
        )
    )
    return out


def _targeted_component_response_score_delta_summary(
    *,
    terms: Mapping[str, Any] | None,
    paired_reference_terms: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Make candidate-vs-reference score deltas explicit for queue consumers."""

    if terms is None and paired_reference_terms is None:
        return None
    term_payload = terms if isinstance(terms, Mapping) else {}
    paired_payload = (
        paired_reference_terms if isinstance(paired_reference_terms, Mapping) else {}
    )
    segnet_delta = _finite_float_or_none(term_payload.get("segnet_delta_score_units"))
    posenet_delta = _finite_float_or_none(term_payload.get("posenet_delta_score_units"))
    correction_rate_delta = _finite_float_or_none(
        term_payload.get("correction_rate_delta_score_units")
    )
    receiver_closed_rate_delta = _finite_float_or_none(
        paired_payload.get("receiver_closed_rate_delta_score_units")
    )
    component_delta = (
        None
        if segnet_delta is None or posenet_delta is None
        else float(segnet_delta) + float(posenet_delta)
    )
    correction_spend_total = (
        None
        if component_delta is None or correction_rate_delta is None
        else float(component_delta) + float(correction_rate_delta)
    )
    receiver_closed_total = (
        None
        if component_delta is None or receiver_closed_rate_delta is None
        else float(component_delta) + float(receiver_closed_rate_delta)
    )
    return {
        "schema": "targeted_component_score_delta_summary.v1",
        "component_delta_score_units": component_delta,
        "segnet_delta_score_units": segnet_delta,
        "posenet_delta_score_units": posenet_delta,
        "correction_rate_delta_score_units": correction_rate_delta,
        "correction_spend_total_delta_score_units": correction_spend_total,
        "receiver_closed_rate_delta_score_units": receiver_closed_rate_delta,
        "receiver_closed_total_delta_score_units": receiver_closed_total,
        "receiver_closed_archive_byte_delta_vs_reference": _finite_int_or_none(
            paired_payload.get("receiver_closed_archive_byte_delta_vs_reference")
        ),
        "paired_reference_role": paired_payload.get("paired_reference_role")
        or term_payload.get("paired_reference_role"),
        "allowed_use": "targeted_component_score_delta_summary_local_signal_only",
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }


def build_frontier_targeted_component_correction_response_harvest_from_artifacts(
    *,
    work_order: Mapping[str, Any],
    local_cpu_advisory: Mapping[str, Any] | None = None,
    reference_local_cpu_advisory: Mapping[str, Any] | None = None,
    local_mlx_response: Mapping[str, Any] | None = None,
    reference_local_mlx_response: Mapping[str, Any] | None = None,
    work_order_path: str | Path | None = None,
    local_cpu_advisory_path: str | Path | None = None,
    reference_local_cpu_advisory_path: str | Path | None = None,
    local_mlx_response_path: str | Path | None = None,
    reference_local_mlx_response_path: str | Path | None = None,
    response_artifact_path: str | Path | None = None,
    reference_role: str = "receiver_closed_source_reference",
    target_optimization_profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Harvest one component-correction response into a false-authority row."""

    if reference_role not in {
        "receiver_closed_source_reference",
        "correction_spend_reference",
    }:
        raise FrontierRateAttackFeedbackError(
            "reference_role must be receiver_closed_source_reference or "
            "correction_spend_reference"
        )
    require_no_truthy_authority_fields(
        work_order,
        context="targeted_component_correction_response_work_order",
    )
    if local_cpu_advisory is not None:
        require_no_truthy_authority_fields(
            local_cpu_advisory,
            context="targeted_component_correction_local_cpu_advisory",
        )
    if reference_local_cpu_advisory is not None:
        require_no_truthy_authority_fields(
            reference_local_cpu_advisory,
            context="targeted_component_correction_reference_local_cpu_advisory",
        )
    if local_mlx_response is not None:
        require_no_truthy_authority_fields(
            local_mlx_response,
            context="targeted_component_correction_local_mlx_response",
        )
    if reference_local_mlx_response is not None:
        require_no_truthy_authority_fields(
            reference_local_mlx_response,
            context="targeted_component_correction_reference_local_mlx_response",
        )

    acquisition_id = str(work_order.get("acquisition_id") or "unknown_acquisition")
    candidate_id = str(work_order.get("candidate_id") or "unknown_candidate")
    correction_family = str(
        work_order.get("correction_family") or "unknown_correction_family"
    )
    target_profile_metadata = _target_profile_metadata_from_payloads(
        target_optimization_profile_metadata,
        work_order,
        context=f"targeted_component_correction_response_row:{acquisition_id}",
    )
    saved_bytes = _finite_int_or_none(work_order.get("saved_bytes_budget")) or 0
    rate_credit = _finite_float_or_none(
        work_order.get("estimated_rate_credit_score_units")
    )
    if rate_credit is None:
        rate_credit = _rate_credit_score_units_for_saved_bytes(saved_bytes)

    blockers = [
        "exact_auth_eval_required_before_score_or_promotion_claim",
        "component_response_harvest_is_local_acquisition_signal_only",
    ]
    local_axis = None
    local_terms: dict[str, Any] = {
        "segnet_delta_score_units": None,
        "posenet_delta_score_units": None,
        "correction_rate_delta_score_units": None,
        "correction_added_archive_bytes": None,
        "delta_source_keys": [],
    }
    local_paired_reference_terms: dict[str, Any] | None = None
    reference_local_axis = None
    if local_cpu_advisory is None:
        blockers.append("local_cpu_component_advisory_missing")
    else:
        local_axis = str(local_cpu_advisory.get("score_axis") or "")
        if local_axis != "cpu_advisory":
            blockers.append("local_cpu_component_advisory_axis_not_cpu_advisory")
        local_terms = _targeted_component_response_delta_terms(local_cpu_advisory)
        if reference_local_cpu_advisory is not None:
            reference_local_axis = str(
                reference_local_cpu_advisory.get("score_axis") or ""
            )
            if reference_local_axis != "cpu_advisory":
                blockers.append(
                    "reference_local_cpu_component_advisory_axis_not_cpu_advisory"
                )
            local_paired_reference_terms = (
                _targeted_component_response_paired_delta_terms(
                    candidate=local_cpu_advisory,
                    reference=reference_local_cpu_advisory,
                    reference_role=reference_role,
                )
            )
            if (
                local_terms["segnet_delta_score_units"] is None
                and local_paired_reference_terms["segnet_delta_score_units"] is not None
            ):
                local_terms["segnet_delta_score_units"] = (
                    local_paired_reference_terms["segnet_delta_score_units"]
                )
            if (
                local_terms["posenet_delta_score_units"] is None
                and local_paired_reference_terms["posenet_delta_score_units"] is not None
            ):
                local_terms["posenet_delta_score_units"] = (
                    local_paired_reference_terms["posenet_delta_score_units"]
                )
            if local_terms["correction_rate_delta_score_units"] is None:
                local_terms["correction_rate_delta_score_units"] = (
                    local_paired_reference_terms[
                        "correction_rate_delta_score_units"
                    ]
                )
            if local_terms["correction_added_archive_bytes"] is None:
                local_terms["correction_added_archive_bytes"] = (
                    local_paired_reference_terms["correction_added_archive_bytes"]
                )
            local_terms["paired_reference_role"] = reference_role
            local_terms["paired_reference_delta_source_keys"] = list(
                local_paired_reference_terms["delta_source_keys"]
            )
        elif (
            local_terms["segnet_delta_score_units"] is None
            or local_terms["posenet_delta_score_units"] is None
        ):
            blockers.append("paired_reference_local_cpu_advisory_required_for_component_delta")

    if local_terms["segnet_delta_score_units"] is None:
        blockers.append("local_cpu_segnet_delta_missing")
    if local_terms["posenet_delta_score_units"] is None:
        blockers.append("local_cpu_posenet_delta_missing")
    if local_terms["correction_rate_delta_score_units"] is None:
        blockers.append(
            "correction_added_byte_delta_missing_assumed_zero_for_acquisition"
        )
        local_terms["correction_rate_delta_score_units"] = 0.0

    mlx_axis = None
    mlx_terms: dict[str, Any] | None = None
    mlx_paired_reference_terms: dict[str, Any] | None = None
    reference_mlx_axis = None
    if local_mlx_response is None:
        blockers.append("local_mlx_component_response_missing_for_spend_filter")
    else:
        mlx_axis = str(local_mlx_response.get("score_axis") or "")
        if mlx_axis != "[macOS-MLX research-signal]":
            blockers.append("local_mlx_component_response_axis_not_research_signal")
        mlx_terms = _targeted_component_response_delta_terms(local_mlx_response)
        if reference_local_mlx_response is not None:
            reference_mlx_axis = str(reference_local_mlx_response.get("score_axis") or "")
            if reference_mlx_axis != "[macOS-MLX research-signal]":
                blockers.append(
                    "reference_local_mlx_component_response_axis_not_research_signal"
                )
            mlx_paired_reference_terms = (
                _targeted_component_response_paired_delta_terms(
                    candidate=local_mlx_response,
                    reference=reference_local_mlx_response,
                    reference_role=reference_role,
                )
            )
            if (
                mlx_terms["segnet_delta_score_units"] is None
                and mlx_paired_reference_terms["segnet_delta_score_units"] is not None
            ):
                mlx_terms["segnet_delta_score_units"] = (
                    mlx_paired_reference_terms["segnet_delta_score_units"]
                )
            if (
                mlx_terms["posenet_delta_score_units"] is None
                and mlx_paired_reference_terms["posenet_delta_score_units"] is not None
            ):
                mlx_terms["posenet_delta_score_units"] = (
                    mlx_paired_reference_terms["posenet_delta_score_units"]
                )
            if mlx_terms["correction_rate_delta_score_units"] is None:
                mlx_terms["correction_rate_delta_score_units"] = (
                    mlx_paired_reference_terms["correction_rate_delta_score_units"]
                )
            if mlx_terms["correction_added_archive_bytes"] is None:
                mlx_terms["correction_added_archive_bytes"] = (
                    mlx_paired_reference_terms["correction_added_archive_bytes"]
                )
            mlx_terms["paired_reference_role"] = reference_role
            mlx_terms["paired_reference_delta_source_keys"] = list(
                mlx_paired_reference_terms["delta_source_keys"]
            )
        if (
            mlx_terms["segnet_delta_score_units"] is None
            or mlx_terms["posenet_delta_score_units"] is None
        ):
            blockers.append("local_mlx_component_delta_missing")

    mlx_cpu_drift_terms = (
        _targeted_component_response_mlx_cpu_drift_terms(
            mlx=local_mlx_response,
            local_cpu=local_cpu_advisory,
        )
        if local_mlx_response is not None and local_cpu_advisory is not None
        else None
    )
    reference_mlx_cpu_drift_terms = (
        _targeted_component_response_mlx_cpu_drift_terms(
            mlx=reference_local_mlx_response,
            local_cpu=reference_local_cpu_advisory,
        )
        if reference_local_mlx_response is not None
        and reference_local_cpu_advisory is not None
        else None
    )
    paired_mlx_cpu_delta_drift_terms = (
        _targeted_component_response_paired_mlx_cpu_delta_drift_terms(
            mlx_paired=mlx_paired_reference_terms,
            local_cpu_paired=local_paired_reference_terms,
        )
    )
    local_cpu_score_delta_summary = _targeted_component_response_score_delta_summary(
        terms=local_terms,
        paired_reference_terms=local_paired_reference_terms,
    )
    local_mlx_score_delta_summary = _targeted_component_response_score_delta_summary(
        terms=mlx_terms,
        paired_reference_terms=mlx_paired_reference_terms,
    )

    measured_component_delta = None
    measured_lagrangian_delta = None
    budget_credit_remaining = None
    local_acquisition_recommended = False
    component_delta_available = (
        local_terms["segnet_delta_score_units"] is not None
        and local_terms["posenet_delta_score_units"] is not None
    )
    if component_delta_available:
        correction_rate_delta = float(
            local_terms["correction_rate_delta_score_units"] or 0.0
        )
        measured_component_delta = float(local_terms["segnet_delta_score_units"]) + float(
            local_terms["posenet_delta_score_units"]
        )
        budget_credit_remaining = float(rate_credit) - correction_rate_delta
        measured_lagrangian_delta = (
            measured_component_delta + correction_rate_delta - float(rate_credit)
        )
        local_acquisition_recommended = measured_lagrangian_delta < 0.0
    else:
        blockers.append("measured_component_delta_missing")

    verdict = "blocked_missing_component_response"
    if component_delta_available:
        verdict = (
            "local_acquisition_negative_lagrangian_candidate"
            if local_acquisition_recommended
            else "local_acquisition_reject_nonnegative_lagrangian"
        )

    row = {
        "schema": TARGETED_COMPONENT_CORRECTION_RESPONSE_ROW_SCHEMA,
        "generated_at_utc": _utc_now(),
        "acquisition_id": acquisition_id,
        "candidate_id": candidate_id,
        "correction_family": correction_family,
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "work_order_path": None if work_order_path is None else str(work_order_path),
        "local_cpu_advisory_path": (
            None if local_cpu_advisory_path is None else str(local_cpu_advisory_path)
        ),
        "reference_local_cpu_advisory_path": (
            None
            if reference_local_cpu_advisory_path is None
            else str(reference_local_cpu_advisory_path)
        ),
        "local_mlx_response_path": (
            None if local_mlx_response_path is None else str(local_mlx_response_path)
        ),
        "response_artifact_path": (
            None if response_artifact_path is None else str(response_artifact_path)
        ),
        "local_cpu_score_axis": local_axis,
        "reference_local_cpu_score_axis": reference_local_axis,
        "local_mlx_score_axis": mlx_axis,
        "reference_local_mlx_score_axis": reference_mlx_axis,
        "saved_bytes_budget": saved_bytes,
        "receiver_closed_saved_bytes": work_order.get("receiver_closed_saved_bytes"),
        "estimated_receiver_closed_rate_credit_score_units": rate_credit,
        **_targeted_rate_packet_context_fields(work_order),
        "receiver_closed_rate_packet_context": _targeted_rate_packet_context(
            work_order
        ),
        "candidate_archive_path": work_order.get("archive_path"),
        "candidate_inflate_sh_path": work_order.get("inflate_sh_path"),
        "candidate_submission_dir": work_order.get("submission_dir"),
        "source_archive_path": work_order.get("source_archive_path"),
        "source_archive_sha256": work_order.get("source_archive_sha256"),
        "source_archive_bytes": work_order.get("source_archive_bytes"),
        "source_inflate_sh_path": work_order.get("source_inflate_sh_path"),
        "source_submission_dir": work_order.get("source_submission_dir"),
        "reference_component_eval_context": dict(
            work_order.get("reference_component_eval_context")
            if isinstance(work_order.get("reference_component_eval_context"), Mapping)
            else {}
        ),
        "closure_report_path": work_order.get("closure_report_path"),
        "paired_exact_readiness_bridge_report_path": work_order.get(
            "paired_exact_readiness_bridge_report_path"
        ),
        "operation_levels": list(work_order.get("operation_levels") or []),
        "targeted_dimensions": list(work_order.get("targeted_dimensions") or []),
        "sibling_correction_families": list(
            work_order.get("sibling_correction_families") or []
        ),
        "local_cpu_component_terms": local_terms,
        "local_cpu_paired_reference_terms": local_paired_reference_terms,
        "local_cpu_score_delta_summary": local_cpu_score_delta_summary,
        "local_mlx_component_terms": mlx_terms,
        "local_mlx_paired_reference_terms": mlx_paired_reference_terms,
        "local_mlx_score_delta_summary": local_mlx_score_delta_summary,
        "local_mlx_vs_local_cpu_drift_terms": mlx_cpu_drift_terms,
        "reference_local_mlx_vs_local_cpu_drift_terms": (
            reference_mlx_cpu_drift_terms
        ),
        "local_mlx_vs_local_cpu_paired_delta_drift_terms": (
            paired_mlx_cpu_delta_drift_terms
        ),
        "reference_local_mlx_response_path": (
            None
            if reference_local_mlx_response_path is None
            else str(reference_local_mlx_response_path)
        ),
        "measured_component_delta_score_units": measured_component_delta,
        "measured_lagrangian_delta_score_units": measured_lagrangian_delta,
        "budget_credit_remaining_score_units": budget_credit_remaining,
        "negative_measured_lagrangian_delta": (
            measured_lagrangian_delta is not None and measured_lagrangian_delta < 0.0
        ),
        "local_acquisition_recommended": local_acquisition_recommended,
        "ready_for_budget_spend": False,
        "budget_spend_allowed": False,
        "budget_spend_blockers": _unique_strings(
            [
                *blockers,
                "exact_axis_component_response_required_before_budget_spend",
                "receiver_runtime_materialization_required_before_budget_spend",
            ]
        ),
        "verdict": verdict,
        "command_hints": dict(
            work_order.get("command_hints")
            if isinstance(work_order.get("command_hints"), Mapping)
            else {}
        ),
        "wire_in_hooks": dict(
            work_order.get("wire_in_hooks")
            if isinstance(work_order.get("wire_in_hooks"), Mapping)
            else {}
        ),
        "allowed_use": (
            "targeted_component_correction_response_harvest_local_acquisition_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context=f"targeted_component_correction_response_row:{acquisition_id}",
    )
    return row


def _targeted_component_response_rows_from_queue(
    *,
    repo_root: Path,
    queue: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    queue_target_profile_metadata = _target_profile_metadata_from_payloads(
        queue,
        context="targeted_component_correction_response_queue",
    )
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        metadata = experiment.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        raw_requests = metadata.get("correction_requests")
        request_metadata = (
            [request for request in raw_requests if isinstance(request, Mapping)]
            if isinstance(raw_requests, list)
            else [metadata]
        )
        for request in request_metadata:
            work_order_path_text = str(
                request.get("work_order_path") or metadata.get("work_order_path") or ""
            )
            local_cpu_path_text = str(
                request.get("local_cpu_advisory_path")
                or metadata.get("local_cpu_advisory_path")
                or ""
            )
            reference_local_cpu_path_text = str(
                request.get("reference_local_cpu_advisory_path")
                or metadata.get("reference_local_cpu_advisory_path")
                or ""
            )
            response_path_text = str(
                request.get("component_correction_response_harvest_path")
                or metadata.get("component_correction_response_harvest_path")
                or ""
            )
            local_mlx_path_text = str(
                request.get("local_mlx_response_path")
                or metadata.get("local_mlx_response_path")
                or ""
            )
            reference_local_mlx_path_text = str(
                request.get("reference_local_mlx_response_path")
                or metadata.get("reference_local_mlx_response_path")
                or ""
            )
            if not work_order_path_text:
                continue
            work_order_path = _resolve_path(work_order_path_text, repo_root=repo_root)
            local_cpu_path = (
                _resolve_path(local_cpu_path_text, repo_root=repo_root)
                if local_cpu_path_text
                else None
            )
            reference_local_cpu_path = (
                _resolve_path(reference_local_cpu_path_text, repo_root=repo_root)
                if reference_local_cpu_path_text
                else None
            )
            local_mlx_path = (
                _resolve_path(local_mlx_path_text, repo_root=repo_root)
                if local_mlx_path_text
                else None
            )
            reference_local_mlx_path = (
                _resolve_path(reference_local_mlx_path_text, repo_root=repo_root)
                if reference_local_mlx_path_text
                else None
            )
            response_path = (
                _resolve_path(response_path_text, repo_root=repo_root)
                if response_path_text
                else None
            )
            request_context_source = dict(metadata)
            request_context_source.update(dict(request))
            request_target_profile_metadata = _target_profile_metadata_from_payloads(
                request,
                metadata,
                queue,
                queue_target_profile_metadata,
                context="targeted_component_correction_response_queue_request",
            )
            missing = [
                _repo_rel(path, repo_root)
                for path in (work_order_path, local_cpu_path, reference_local_cpu_path)
                if path is not None and not path.exists()
            ]
            if missing:
                row = {
                    "schema": TARGETED_COMPONENT_CORRECTION_RESPONSE_ROW_SCHEMA,
                    "generated_at_utc": _utc_now(),
                    "acquisition_id": request.get("acquisition_id"),
                    "candidate_id": request.get("candidate_id")
                    or metadata.get("candidate_id"),
                    "correction_family": request.get("correction_family"),
                    "frontier_target_optimization_profile": dict(
                        request_target_profile_metadata
                    ),
                    "operation_levels": list(request.get("operation_levels") or []),
                    "targeted_dimensions": list(
                        request.get("targeted_dimensions") or []
                    ),
                    "work_order_path": work_order_path_text or None,
                    "local_cpu_advisory_path": local_cpu_path_text or None,
                    "reference_local_cpu_advisory_path": (
                        reference_local_cpu_path_text or None
                    ),
                    "local_mlx_response_path": local_mlx_path_text or None,
                    "reference_local_mlx_response_path": (
                        reference_local_mlx_path_text or None
                    ),
                    "response_artifact_path": response_path_text or None,
                    "saved_bytes_budget": request.get("saved_bytes_budget")
                    or metadata.get("saved_bytes_budget"),
                    "receiver_closed_saved_bytes": request.get(
                        "receiver_closed_saved_bytes"
                    )
                    or metadata.get("receiver_closed_saved_bytes"),
                    "estimated_receiver_closed_rate_credit_score_units": request.get(
                        "estimated_rate_credit_score_units"
                    )
                    or metadata.get("estimated_rate_credit_score_units"),
                    **_targeted_rate_packet_context_fields(request_context_source),
                    "receiver_closed_rate_packet_context": (
                        _targeted_rate_packet_context(request_context_source)
                    ),
                    "measured_component_delta_score_units": None,
                    "measured_lagrangian_delta_score_units": None,
                    "budget_credit_remaining_score_units": None,
                    "negative_measured_lagrangian_delta": False,
                    "local_acquisition_recommended": False,
                    "ready_for_budget_spend": False,
                    "budget_spend_allowed": False,
                    "budget_spend_blockers": _unique_strings(
                        [
                            "targeted_component_correction_response_artifacts_missing",
                            *(f"missing:{path}" for path in missing),
                            "exact_axis_component_response_required_before_budget_spend",
                        ]
                    ),
                    "verdict": "blocked_missing_component_response_artifact",
                    "allowed_use": (
                        "targeted_component_correction_response_harvest_local_acquisition_only"
                    ),
                    "forbidden_use": (
                        "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                    ),
                    **FALSE_AUTHORITY,
                }
                rows.append(row)
                continue
            work_order = _load_json(work_order_path)
            local_cpu = (
                _load_json(local_cpu_path) if local_cpu_path is not None else None
            )
            reference_local_cpu = (
                _load_json(reference_local_cpu_path)
                if reference_local_cpu_path is not None
                else None
            )
            local_mlx = (
                _load_json(local_mlx_path)
                if local_mlx_path is not None and local_mlx_path.exists()
                else None
            )
            reference_local_mlx = (
                _load_json(reference_local_mlx_path)
                if reference_local_mlx_path is not None
                and reference_local_mlx_path.exists()
                else None
            )
            rows.append(
                build_frontier_targeted_component_correction_response_harvest_from_artifacts(
                    work_order=work_order,
                    local_cpu_advisory=local_cpu,
                    reference_local_cpu_advisory=reference_local_cpu,
                    local_mlx_response=local_mlx,
                    reference_local_mlx_response=reference_local_mlx,
                    work_order_path=_repo_rel(work_order_path, repo_root),
                    local_cpu_advisory_path=(
                        None
                        if local_cpu_path is None
                        else _repo_rel(local_cpu_path, repo_root)
                    ),
                    reference_local_cpu_advisory_path=(
                        None
                        if reference_local_cpu_path is None
                        else _repo_rel(reference_local_cpu_path, repo_root)
                    ),
                    local_mlx_response_path=(
                        None
                        if local_mlx_path is None
                        else _repo_rel(local_mlx_path, repo_root)
                    ),
                    reference_local_mlx_response_path=(
                        None
                        if reference_local_mlx_path is None
                        else _repo_rel(reference_local_mlx_path, repo_root)
                    ),
                    response_artifact_path=(
                        None
                        if response_path is None
                        else _repo_rel(response_path, repo_root)
                    ),
                    target_optimization_profile_metadata=(
                        request_target_profile_metadata
                    ),
                )
            )
    return rows


def _targeted_component_response_rows_from_existing_harvests(
    *,
    repo_root: Path,
    results_root: str | Path,
) -> list[dict[str, Any]]:
    resolved = _resolve_path(results_root, repo_root=repo_root)
    if not resolved.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("component_correction_response_harvest.json")):
        try:
            payload = _load_json(path)
        except FrontierRateAttackFeedbackError:
            continue
        if payload.get("schema") != TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA:
            continue
        for row in payload.get("rows") or []:
            if isinstance(row, dict):
                copied = dict(row)
                copied.setdefault("source_harvest_path", _repo_rel(path, repo_root))
                rows.append(copied)
    return rows


def _targeted_component_correction_queue_blockers(
    queue: Mapping[str, Any],
) -> list[str]:
    blockers: list[Any] = []
    metadata = queue.get("metadata")
    if isinstance(metadata, Mapping):
        blockers.extend(_string_list(metadata.get("blockers")))
        blockers.extend(_string_list(metadata.get("queue_actuation_blockers")))
        acquisition = metadata.get("targeted_component_correction_acquisition")
        if isinstance(acquisition, Mapping):
            blockers.extend(_string_list(acquisition.get("blockers")))
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        if str(experiment.get("status") or "") == "frozen":
            blockers.append(
                f"targeted_component_correction_experiment_frozen:{experiment.get('id')}"
            )
        experiment_metadata = experiment.get("metadata")
        if isinstance(experiment_metadata, Mapping):
            blockers.extend(_string_list(experiment_metadata.get("blockers")))
            blockers.extend(_string_list(experiment_metadata.get("queue_actuation_blockers")))
    return _unique_strings(blockers)


def build_frontier_targeted_component_correction_response_harvest(
    *,
    repo_root: str | Path,
    targeted_component_correction_queue: Mapping[str, Any] | None = None,
    response_rows: Sequence[Mapping[str, Any]] = (),
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any]:
    """Aggregate measured correction responses without granting budget authority."""

    repo = Path(repo_root)
    rows: list[dict[str, Any]] = [dict(row) for row in response_rows]
    queue_blockers: list[str] = []
    if targeted_component_correction_queue is not None:
        require_no_truthy_authority_fields(
            targeted_component_correction_queue,
            context="targeted_component_correction_response_queue_input",
        )
        queue_blockers = _targeted_component_correction_queue_blockers(
            targeted_component_correction_queue
        )
        rows.extend(
            _targeted_component_response_rows_from_queue(
                repo_root=repo,
                queue=targeted_component_correction_queue,
            )
        )
    if not rows and not queue_blockers:
        rows.extend(
            _targeted_component_response_rows_from_existing_harvests(
                repo_root=repo,
                results_root=results_root,
            )
        )
    for index, row in enumerate(rows):
        require_no_truthy_authority_fields(
            row,
            context=f"targeted_component_correction_response_harvest_row:{index}",
        )
    accepted = [
        row
        for row in rows
        if row.get("negative_measured_lagrangian_delta") is True
        and row.get("local_acquisition_recommended") is True
    ]
    blocked = [
        row
        for row in rows
        if row.get("measured_lagrangian_delta_score_units") is None
        or row.get("budget_spend_blockers")
    ]
    candidate_ids = _unique_strings(row.get("candidate_id") for row in rows)
    families = _unique_strings(row.get("correction_family") for row in rows)
    target_profile_metadata = _target_profile_metadata_from_payloads(
        targeted_component_correction_queue,
        *rows,
        context="targeted_component_correction_response_harvest",
    )
    blockers = ["exact_auth_eval_required_before_score_or_promotion_claim"]
    if not rows:
        blockers.append("no_targeted_component_correction_response_rows")
    if queue_blockers:
        blockers.extend(queue_blockers)
    if blocked:
        blockers.append("response_rows_blocked_before_budget_spend")
    absolute_score_drifts: list[float] = []
    paired_delta_drifts: list[float] = []
    for row in rows:
        terms = row.get("local_mlx_vs_local_cpu_drift_terms")
        if isinstance(terms, Mapping):
            value = _finite_float_or_none(
                terms.get("mlx_minus_local_cpu_score_units")
            )
            if value is not None:
                absolute_score_drifts.append(abs(value))
        paired_terms = row.get("local_mlx_vs_local_cpu_paired_delta_drift_terms")
        if isinstance(paired_terms, Mapping):
            value = _finite_float_or_none(
                paired_terms.get(
                    "mlx_minus_local_cpu_paired_lagrangian_delta_score_units"
                )
            )
            if value is not None:
                paired_delta_drifts.append(abs(value))
    return {
        "schema": TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA,
        "generated_at_utc": _utc_now(),
        "active": bool(rows),
        "row_count": len(rows),
        "candidate_count": len(candidate_ids),
        "correction_family_count": len(families),
        "negative_measured_lagrangian_delta_count": len(accepted),
        "local_acquisition_recommended_count": len(accepted),
        "blocked_response_count": len(blocked),
        "ready_for_budget_spend_count": 0,
        "candidate_ids": candidate_ids,
        "correction_families": families,
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "source_targeted_component_correction_queue_blockers": queue_blockers,
        "mlx_cpu_drift_summary": {
            "schema": "targeted_component_mlx_cpu_drift_summary.v1",
            "absolute_score_drift_row_count": len(absolute_score_drifts),
            "absolute_score_drift_max_abs": (
                max(absolute_score_drifts) if absolute_score_drifts else None
            ),
            "paired_delta_drift_row_count": len(paired_delta_drifts),
            "paired_lagrangian_delta_drift_max_abs": (
                max(paired_delta_drifts) if paired_delta_drifts else None
            ),
            "interpretation": (
                "absolute MLX-vs-local-CPU offsets may be nonzero; paired "
                "candidate-vs-reference delta drift is the MLX acquisition "
                "quality signal for receiver-closed rate attacks"
            ),
            **FALSE_AUTHORITY,
        },
        "top_local_acquisition_ids": [
            str(row.get("acquisition_id") or "")
            for row in sorted(
                accepted,
                key=lambda item: float(
                    item.get("measured_lagrangian_delta_score_units") or 0.0
                ),
            )[:8]
        ],
        "blockers": _unique_strings(blockers),
        "recommended_next_action": (
            "materialize_exact_axis_receiver_consumed_correction_candidates_for_"
            "negative_local_lagrangian_rows"
            if accepted
            else (
                "resolve_targeted_component_correction_queue_blockers"
                if queue_blockers
                else "run_targeted_component_correction_queue_until_response_rows_exist"
            )
        ),
        "rows": rows,
        "allowed_use": (
            "queue_owned_targeted_component_correction_response_harvest_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }


def _targeted_component_family_seed(correction_family: Any) -> Mapping[str, Any]:
    family = str(correction_family or "")
    for seed in _TARGETED_COMPONENT_CORRECTION_FAMILY_SEEDS:
        if str(seed.get("correction_family") or "") == family:
            return seed
    return {}


def _targeted_component_response_operation_levels(
    row: Mapping[str, Any],
) -> list[str]:
    levels = _unique_strings(_string_list(row.get("operation_levels")))
    if levels:
        return levels
    seed = _targeted_component_family_seed(row.get("correction_family"))
    return _unique_strings(_string_list(seed.get("operation_levels")))


def _targeted_component_response_targeted_dimensions(
    row: Mapping[str, Any],
) -> list[str]:
    dimensions = _unique_strings(_string_list(row.get("targeted_dimensions")))
    if dimensions:
        return dimensions
    seed = _targeted_component_family_seed(row.get("correction_family"))
    return _unique_strings(_string_list(seed.get("targeted_dimensions")))


def _targeted_component_response_sort_key(row: Mapping[str, Any]) -> tuple[float, str]:
    delta = _finite_float_or_none(row.get("measured_lagrangian_delta_score_units"))
    return (
        float(delta) if delta is not None else 0.0,
        str(row.get("acquisition_id") or ""),
    )


_TARGETED_RATE_PACKET_CONTEXT_KEYS = (
    "receiver_closed_saved_bytes",
    "rate_packet_manifest_path",
    "parent_rate_packet_manifest_path",
    "candidate_compact_selector_codec",
    "parent_compact_selector_codec",
    "selector_policy_mode",
    "archive_byte_delta_vs_parent",
    "selector_payload_wire_bytes",
    "parent_selector_payload_wire_bytes",
    "selector_payload_wire_delta_bytes",
    "selector_code_bits_total",
    "parent_selector_code_bits_total",
    "selector_avg_bits_per_pair",
    "parent_selector_avg_bits_per_pair",
    "palette_size",
    "n_pairs",
    "compact_palette_mode_ids",
    "entropy_position",
)


def _targeted_rate_packet_context_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in _TARGETED_RATE_PACKET_CONTEXT_KEYS:
        value = payload.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            fields[key] = list(value)
        else:
            fields[key] = value
    return fields


def _targeted_rate_packet_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = _targeted_rate_packet_context_fields(payload)
    if not any(
        fields.get(key)
        for key in (
            "rate_packet_manifest_path",
            "candidate_compact_selector_codec",
            "entropy_position",
        )
    ):
        return {}
    return {
        "schema": "targeted_component_receiver_closed_rate_packet_context.v1",
        **fields,
        "context_status": "receiver_closed_rate_packet_entropy_context_available",
        "pipeline_side": "encoder_rate_packet_repair_budget_allocator",
        "receiver_role": "deterministic_decode_only_no_eval_time_adaptation",
        "allowed_use": (
            "receiver_closed_rate_packet_context_for_local_repair_chain_planning"
        ),
        "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
        **FALSE_AUTHORITY,
    }


def _target_profile_metadata_from_payloads(
    *payloads: Mapping[str, Any] | None,
    context: str,
) -> dict[str, Any]:
    """Return the first target-profile metadata payload carried by queue artifacts."""

    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        if payload.get("schema") in TARGET_OPTIMIZATION_PROFILE_METADATA_SCHEMAS:
            copied = dict(payload)
            require_no_truthy_authority_fields(
                copied,
                context=f"{context}.frontier_target_optimization_profile",
            )
            return copied
        candidates = (
            payload.get("frontier_target_optimization_profile"),
            payload.get("target_optimization_profile_metadata"),
        )
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            candidates = (
                *candidates,
                metadata.get("frontier_target_optimization_profile"),
                metadata.get("target_optimization_profile_metadata"),
            )
        for candidate in candidates:
            if isinstance(candidate, Mapping) and candidate:
                copied = dict(candidate)
                require_no_truthy_authority_fields(
                    copied,
                    context=f"{context}.frontier_target_optimization_profile",
                )
                return copied
    return {}


def _accepted_targeted_component_response_rows(
    targeted_component_correction_response_harvest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (
        targeted_component_correction_response_harvest.get("schema")
        != TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "targeted component correction response harvest has unexpected schema"
        )
    require_no_truthy_authority_fields(
        targeted_component_correction_response_harvest,
        context="targeted_component_correction_materialization_response_harvest",
    )
    accepted: list[dict[str, Any]] = []
    for index, row in enumerate(
        targeted_component_correction_response_harvest.get("rows") or []
    ):
        if not isinstance(row, Mapping):
            continue
        require_no_truthy_authority_fields(
            row,
            context=f"targeted_component_correction_materialization_response_row:{index}",
        )
        if (
            row.get("negative_measured_lagrangian_delta") is True
            and row.get("local_acquisition_recommended") is True
        ):
            accepted.append(dict(row))
    return sorted(accepted, key=_targeted_component_response_sort_key)


def _targeted_component_materializer_basis_entry(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    family = str(row.get("correction_family") or "unknown_correction_family")
    target_profile_metadata = _target_profile_metadata_from_payloads(
        row,
        context="targeted_component_materializer_basis_entry",
    )
    seed = _targeted_component_family_seed(family)
    command_hints = row.get("command_hints")
    normalized_command_hints: list[dict[str, Any]] | dict[str, Any]
    if isinstance(command_hints, Sequence) and not isinstance(
        command_hints,
        (str, bytes, bytearray),
    ):
        normalized_command_hints = [
            dict(hint) for hint in command_hints if isinstance(hint, Mapping)
        ]
    elif isinstance(command_hints, Mapping):
        normalized_command_hints = dict(command_hints)
    else:
        normalized_command_hints = []
    return {
        "schema": (
            "frontier_rate_attack_targeted_component_correction_materializer_"
            "basis_entry.v1"
        ),
        "source_acquisition_id": row.get("acquisition_id"),
        "correction_family": family,
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "operation_levels": _targeted_component_response_operation_levels(row),
        "targeted_dimensions": _targeted_component_response_targeted_dimensions(row),
        "recommended_next_action": (
            seed.get("recommended_next_action")
            or "compile_receiver_consumed_targeted_correction_materializer"
        ),
        "measured_lagrangian_delta_score_units": row.get(
            "measured_lagrangian_delta_score_units"
        ),
        "measured_component_delta_score_units": row.get(
            "measured_component_delta_score_units"
        ),
        "budget_credit_remaining_score_units": row.get(
            "budget_credit_remaining_score_units"
        ),
        "local_cpu_score_delta_summary": dict(
            row.get("local_cpu_score_delta_summary")
            if isinstance(row.get("local_cpu_score_delta_summary"), Mapping)
            else {}
        ),
        "local_mlx_score_delta_summary": dict(
            row.get("local_mlx_score_delta_summary")
            if isinstance(row.get("local_mlx_score_delta_summary"), Mapping)
            else {}
        ),
        "saved_bytes_budget": row.get("saved_bytes_budget"),
        "receiver_closed_saved_bytes": row.get("receiver_closed_saved_bytes"),
        **_targeted_rate_packet_context_fields(row),
        "receiver_closed_rate_packet_context": _targeted_rate_packet_context(row),
        "work_order_path": row.get("work_order_path"),
        "local_cpu_advisory_path": row.get("local_cpu_advisory_path"),
        "local_mlx_response_path": row.get("local_mlx_response_path"),
        "response_artifact_path": row.get("response_artifact_path"),
        "candidate_archive_path": row.get("candidate_archive_path"),
        "candidate_inflate_sh_path": row.get("candidate_inflate_sh_path"),
        "candidate_submission_dir": row.get("candidate_submission_dir"),
        "source_archive_path": row.get("source_archive_path"),
        "source_archive_sha256": row.get("source_archive_sha256"),
        "source_archive_bytes": row.get("source_archive_bytes"),
        "source_inflate_sh_path": row.get("source_inflate_sh_path"),
        "source_submission_dir": row.get("source_submission_dir"),
        "reference_component_eval_context": dict(
            row.get("reference_component_eval_context")
            if isinstance(row.get("reference_component_eval_context"), Mapping)
            else {}
        ),
        "closure_report_path": row.get("closure_report_path"),
        "paired_exact_readiness_bridge_report_path": row.get(
            "paired_exact_readiness_bridge_report_path"
        ),
        "command_hints": normalized_command_hints,
        "wire_in_hooks": dict(
            row.get("wire_in_hooks")
            if isinstance(row.get("wire_in_hooks"), Mapping)
            else {}
        ),
        "allowed_use": (
            "local_targeted_component_correction_materializer_basis_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }


def _first_basis_value(
    basis: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Any:
    for entry in basis:
        for key in keys:
            value = entry.get(key)
            if value not in (None, ""):
                return value
    return None


def _unique_basis_strings(
    basis: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> list[str]:
    return _unique_strings(
        value
        for entry in basis
        for key in keys
        for value in (entry.get(key),)
        if value not in (None, "")
    )


def _targeted_component_receiver_runtime_binding_context(
    basis: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_context = _first_basis_value(
        basis,
        ("reference_component_eval_context",),
    )
    if not isinstance(reference_context, Mapping):
        reference_context = {}
    candidate_archive_path = _first_basis_value(
        basis,
        ("candidate_archive_path", "archive_path"),
    )
    candidate_inflate_sh_path = _first_basis_value(
        basis,
        ("candidate_inflate_sh_path", "inflate_sh_path"),
    )
    candidate_submission_dir = _first_basis_value(
        basis,
        ("candidate_submission_dir", "submission_dir"),
    )
    source_archive_path = _first_basis_value(
        basis,
        ("source_archive_path",),
    ) or reference_context.get("source_archive_path")
    source_inflate_sh_path = _first_basis_value(
        basis,
        ("source_inflate_sh_path",),
    ) or reference_context.get("source_inflate_sh_path")
    source_submission_dir = _first_basis_value(
        basis,
        ("source_submission_dir",),
    ) or reference_context.get("source_submission_dir")
    return {
        "schema": (
            "frontier_rate_attack_targeted_component_receiver_runtime_binding.v1"
        ),
        "candidate_archive_path": candidate_archive_path,
        "candidate_inflate_sh_path": candidate_inflate_sh_path,
        "candidate_submission_dir": candidate_submission_dir,
        "source_archive_path": source_archive_path,
        "source_archive_sha256": _first_basis_value(
            basis,
            ("source_archive_sha256",),
        )
        or reference_context.get("source_archive_sha256"),
        "source_archive_bytes": _first_basis_value(
            basis,
            ("source_archive_bytes",),
        )
        or reference_context.get("source_archive_bytes"),
        "source_inflate_sh_path": source_inflate_sh_path,
        "source_submission_dir": source_submission_dir,
        "source_queue_path": reference_context.get("source_queue_path"),
        "work_order_paths": _unique_basis_strings(basis, ("work_order_path",)),
        "response_artifact_paths": _unique_basis_strings(
            basis,
            ("response_artifact_path",),
        ),
        "binding_complete_for_component_eval": bool(
            candidate_archive_path and candidate_inflate_sh_path
        ),
        "binding_complete_for_reference_eval": bool(
            source_archive_path and source_inflate_sh_path
        ),
        "allowed_use": (
            "targeted_component_correction_runtime_binding_for_queue_planning_only"
        ),
        "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
        **FALSE_AUTHORITY,
    }


def _build_targeted_component_materialization_request_row(
    *,
    candidate_id: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    request_rank: int,
) -> dict[str, Any]:
    rows = sorted(candidate_rows, key=_targeted_component_response_sort_key)
    basis = [_targeted_component_materializer_basis_entry(row) for row in rows]
    target_profile_metadata = _target_profile_metadata_from_payloads(
        *basis,
        *rows,
        context="targeted_component_materialization_request",
    )
    runtime_binding_context = (
        _targeted_component_receiver_runtime_binding_context(basis)
    )
    request_id = (
        "targeted_component_materialization_"
        f"{_slug_token(candidate_id)}_{request_rank:03d}"
    )
    operation_levels = _unique_strings(
        [level for entry in basis for level in entry.get("operation_levels") or []]
    )
    targeted_dimensions = _unique_strings(
        [
            dimension
            for entry in basis
            for dimension in entry.get("targeted_dimensions") or []
        ]
    )
    saved_bytes_budget = max(
        int(row.get("saved_bytes_budget") or 0) for row in rows
    )
    receiver_closed_saved_bytes = max(
        int(row.get("receiver_closed_saved_bytes") or row.get("saved_bytes_budget") or 0)
        for row in rows
    )
    rate_packet_contexts: list[dict[str, Any]] = []
    for entry in basis:
        context = entry.get("receiver_closed_rate_packet_context")
        if isinstance(context, Mapping) and context.get("schema"):
            rate_packet_contexts.append(dict(context))
            continue
        context = _targeted_rate_packet_context(entry)
        if context:
            rate_packet_contexts.append(context)
    rate_credit = max(
        float(row.get("estimated_receiver_closed_rate_credit_score_units") or 0.0)
        for row in rows
    )
    lagrangian_sum = sum(
        float(row.get("measured_lagrangian_delta_score_units") or 0.0)
        for row in rows
    )
    best_row = rows[0]
    blockers = _unique_strings(
        [
            "receiver_consumed_correction_materializer_missing",
            "full_frame_inflate_parity_required_before_budget_spend",
            "exact_axis_component_response_required_before_budget_spend",
            "exact_auth_eval_required_before_score_or_promotion_claim",
            "generated_request_is_local_materialization_only",
            *[
                blocker
                for row in rows
                for blocker in _string_list(row.get("budget_spend_blockers"))
            ],
        ]
    )
    row = {
        "schema": TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA,
        "generated_at_utc": _utc_now(),
        "materialization_request_id": request_id,
        "candidate_id": candidate_id,
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "source_acquisition_ids": _unique_strings(
            [row.get("acquisition_id") for row in rows]
        ),
        "accepted_correction_families": _unique_strings(
            [row.get("correction_family") for row in rows]
        ),
        "accepted_response_count": len(rows),
        "operation_levels": operation_levels,
        "targeted_dimensions": targeted_dimensions,
        "saved_bytes_budget": saved_bytes_budget,
        "receiver_closed_saved_bytes": receiver_closed_saved_bytes,
        "receiver_closed_rate_packet_contexts": rate_packet_contexts,
        "rate_packet_manifest_paths": _unique_basis_strings(
            basis,
            ("rate_packet_manifest_path",),
        ),
        "parent_rate_packet_manifest_paths": _unique_basis_strings(
            basis,
            ("parent_rate_packet_manifest_path",),
        ),
        "candidate_compact_selector_codecs": _unique_basis_strings(
            basis,
            ("candidate_compact_selector_codec",),
        ),
        "parent_compact_selector_codecs": _unique_basis_strings(
            basis,
            ("parent_compact_selector_codec",),
        ),
        "entropy_positions": _unique_basis_strings(basis, ("entropy_position",)),
        "estimated_receiver_closed_rate_credit_score_units": rate_credit,
        "measured_lagrangian_delta_score_units_sum": lagrangian_sum,
        "best_measured_lagrangian_delta_score_units": best_row.get(
            "measured_lagrangian_delta_score_units"
        ),
        "best_local_cpu_score_delta_summary": dict(
            best_row.get("local_cpu_score_delta_summary")
            if isinstance(best_row.get("local_cpu_score_delta_summary"), Mapping)
            else {}
        ),
        "best_local_mlx_score_delta_summary": dict(
            best_row.get("local_mlx_score_delta_summary")
            if isinstance(best_row.get("local_mlx_score_delta_summary"), Mapping)
            else {}
        ),
        "budget_credit_remaining_score_units_min": min(
            float(row.get("budget_credit_remaining_score_units") or 0.0)
            for row in rows
        ),
        "materializer_chain_basis": basis,
        "receiver_runtime_binding_context": runtime_binding_context,
        "receiver_materialization_contract": {
            "schema": (
                "frontier_rate_attack_receiver_consumed_targeted_correction_"
                "contract.v1"
            ),
            "receiver_consumed_transform_required": True,
            "source_runtime_adapter_proof_required": True,
            "full_frame_inflate_parity_required": True,
            "segnet_posenet_component_response_required": True,
            "exact_auth_eval_required_before_score_claim": True,
            "parser_only_or_planner_only_signal_is_insufficient": True,
            "allowed_use": (
                "receiver_consumed_targeted_correction_materialization_contract"
            ),
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        },
        "queue_consumer": (
            "frontier_targeted_component_correction_materialization_queue"
        ),
        "ready_for_materializer_execution": False,
        "ready_for_budget_spend": False,
        "budget_spend_allowed": False,
        "budget_spend_blockers": blockers,
        "recommended_next_action": (
            "compile_and_run_receiver_consumed_targeted_correction_chain_for_"
            "accepted_local_lagrangian_rows"
        ),
        "allowed_use": (
            "local_targeted_component_correction_materialization_request_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context=f"targeted_component_correction_materialization_request:{request_id}",
    )
    return row


def build_frontier_targeted_component_correction_materialization_requests(
    *,
    targeted_component_correction_response_harvest: Mapping[str, Any],
    candidate_limit: int = 4,
    family_limit_per_candidate: int = 8,
) -> dict[str, Any]:
    """Compile negative local correction responses into grouped request rows."""

    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    if family_limit_per_candidate < 1:
        raise FrontierRateAttackFeedbackError("family_limit_per_candidate must be >= 1")
    accepted_rows = _accepted_targeted_component_response_rows(
        targeted_component_correction_response_harvest
    )
    target_profile_metadata = _target_profile_metadata_from_payloads(
        targeted_component_correction_response_harvest,
        *accepted_rows,
        context="targeted_component_correction_materialization_requests",
    )
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in accepted_rows:
        candidate_id = str(row.get("candidate_id") or "unknown_candidate")
        rows_by_candidate.setdefault(candidate_id, []).append(row)
    candidate_order = sorted(
        rows_by_candidate,
        key=lambda candidate: (
            sum(
                float(row.get("measured_lagrangian_delta_score_units") or 0.0)
                for row in rows_by_candidate[candidate]
            ),
            candidate,
        ),
    )[:candidate_limit]
    request_rows = [
        _build_targeted_component_materialization_request_row(
            candidate_id=candidate,
            candidate_rows=rows_by_candidate[candidate][:family_limit_per_candidate],
            request_rank=index,
        )
        for index, candidate in enumerate(candidate_order, start=1)
    ]
    for index, row in enumerate(request_rows):
        require_no_truthy_authority_fields(
            row,
            context=f"targeted_component_correction_materialization_request_row:{index}",
        )
    blockers = ["exact_auth_eval_required_before_score_or_promotion_claim"]
    if not accepted_rows:
        blockers.append("no_negative_local_lagrangian_response_rows")
    else:
        blockers.extend(
            [
                "receiver_consumed_correction_materializer_missing",
                "exact_axis_component_response_required_before_budget_spend",
            ]
        )
    return {
        "schema": TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "active": bool(request_rows),
        "response_harvest_schema": (
            targeted_component_correction_response_harvest.get("schema")
        ),
        "accepted_response_count": len(accepted_rows),
        "row_count": len(request_rows),
        "candidate_count": len(candidate_order),
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "requested_correction_family_count": len(
            _unique_strings(
                [
                    family
                    for row in request_rows
                    for family in row.get("accepted_correction_families") or []
                ]
            )
        ),
        "ready_for_budget_spend_count": 0,
        "candidate_limit": candidate_limit,
        "family_limit_per_candidate": family_limit_per_candidate,
        "top_materialization_request_ids": [
            str(row.get("materialization_request_id") or "") for row in request_rows[:8]
        ],
        "blockers": _unique_strings(blockers),
        "rows": request_rows,
        "allowed_use": (
            "queue_owned_targeted_component_correction_materialization_requests_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }


def build_frontier_targeted_component_correction_materialization_request(
    *,
    targeted_component_correction_response_harvest: Mapping[str, Any],
    materialization_request_id: str,
    candidate_limit: int = 4,
    family_limit_per_candidate: int = 8,
) -> dict[str, Any]:
    """Return one grouped targeted-correction materialization request row."""

    requests = build_frontier_targeted_component_correction_materialization_requests(
        targeted_component_correction_response_harvest=(
            targeted_component_correction_response_harvest
        ),
        candidate_limit=candidate_limit,
        family_limit_per_candidate=family_limit_per_candidate,
    )
    for row in requests.get("rows") or []:
        if (
            isinstance(row, Mapping)
            and str(row.get("materialization_request_id") or "")
            == materialization_request_id
        ):
            selected = dict(row)
            require_no_truthy_authority_fields(
                selected,
                context=(
                    "targeted_component_correction_materialization_request:"
                    f"{materialization_request_id}"
                ),
            )
            return selected
    raise FrontierRateAttackFeedbackError(
        f"unknown targeted component correction materialization request: "
        f"{materialization_request_id}"
    )


def build_frontier_targeted_component_correction_materialization_queue(
    *,
    repo_root: str | Path,
    targeted_component_correction_response_harvest: Mapping[str, Any],
    targeted_component_correction_response_harvest_path: str | Path,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    queue_id: str = "frontier_targeted_component_correction_materialization_queue",
    candidate_limit: int = 4,
    family_limit_per_candidate: int = 8,
) -> dict[str, Any] | None:
    """Queue local request emission for accepted targeted correction responses."""

    repo = Path(repo_root)
    response_harvest_path = _resolve_path(
        targeted_component_correction_response_harvest_path,
        repo_root=repo,
    )
    requests = build_frontier_targeted_component_correction_materialization_requests(
        targeted_component_correction_response_harvest=(
            targeted_component_correction_response_harvest
        ),
        candidate_limit=candidate_limit,
        family_limit_per_candidate=family_limit_per_candidate,
    )
    request_rows = [
        row for row in requests.get("rows") or [] if isinstance(row, Mapping)
    ]
    if not request_rows:
        return None
    results_base = _resolve_path(str(results_root), repo_root=repo)
    queue_root = results_base / "frontier_targeted_component_correction_materialize" / (
        _slug_token(queue_id)
    )
    experiments: list[dict[str, Any]] = []
    for priority, row in enumerate(request_rows, start=1):
        request_id = str(row.get("materialization_request_id") or f"request_{priority}")
        candidate_id = str(row.get("candidate_id") or request_id)
        work_dir = queue_root / _slug_token(candidate_id) / _slug_token(request_id)
        request_path = work_dir / "materialization_request.json"
        steps = [
            {
                "id": "emit_targeted_component_correction_materialization_request",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "tools/build_frontier_targeted_component_correction_materialization_request.py",
                    "--targeted-component-correction-response-harvest",
                    _repo_rel(response_harvest_path, repo),
                    "--materialization-request-id",
                    request_id,
                    "--request-out",
                    _repo_rel(request_path, repo),
                    "--candidate-limit",
                    str(candidate_limit),
                    "--family-limit-per-candidate",
                    str(family_limit_per_candidate),
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": _repo_rel(request_path, repo),
                        "key": "schema",
                        "equals": (
                            TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA
                        ),
                    },
                    {
                        "type": "json_false_authority",
                        "path": _repo_rel(request_path, repo),
                    },
                    {
                        "type": "json_equals",
                        "path": _repo_rel(request_path, repo),
                        "key": "ready_for_budget_spend",
                        "equals": False,
                    },
                    {
                        "type": "json_equals",
                        "path": _repo_rel(request_path, repo),
                        "key": "ready_for_materializer_execution",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [_repo_rel(request_path, repo)],
                    "input_artifact_paths": [_repo_rel(response_harvest_path, repo)],
                    "include_postcondition_paths": True,
                },
            }
        ]
        experiments.append(
            {
                "id": _slug_token(request_id),
                "status": "queued",
                "priority": priority,
                "lane_id": "lane_frontier_targeted_component_materialization_20260526",
                "tags": [
                    "targeted_component_correction",
                    "materialization_request",
                    "receiver_consumed_budget",
                    *[
                        str(family)
                        for family in row.get("accepted_correction_families") or []
                    ],
                ],
                "metadata": {
                    "schema": (
                        TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_QUEUE_METADATA_SCHEMA
                    ),
                    "materialization_request_id": request_id,
                    "candidate_id": candidate_id,
                    "frontier_target_optimization_profile": dict(
                        _target_profile_metadata_from_payloads(
                            row,
                            requests,
                            context=(
                                "targeted_component_correction_materialization_"
                                "queue_metadata"
                            ),
                        )
                    ),
                    "accepted_correction_families": list(
                        row.get("accepted_correction_families") or []
                    ),
                    "operation_levels": list(row.get("operation_levels") or []),
                    "targeted_dimensions": list(row.get("targeted_dimensions") or []),
                    "response_harvest_path": _repo_rel(response_harvest_path, repo),
                    "materialization_request_path": _repo_rel(request_path, repo),
                    "ready_for_materializer_execution": False,
                    "ready_for_budget_spend": False,
                    "budget_spend_allowed": False,
                    "allowed_use": (
                        "targeted_component_correction_materialization_queue_"
                        "metadata_only"
                    ),
                    "forbidden_use": "score_claim_or_dispatch_authority",
                    **FALSE_AUTHORITY,
                },
                "steps": steps,
            }
        )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_io_heavy": 1,
                    "local_cpu": 1,
                    "local_mlx": 0,
                    "modal_cpu": 0,
                    "modal_gpu": 0,
                },
            },
            "experiments": experiments,
        }
    )
    queue["metadata"] = {
        "schema": (
            "frontier_rate_attack_targeted_component_correction_"
            "materialization_queue_root_metadata.v1"
        ),
        "frontier_target_optimization_profile": dict(
            _target_profile_metadata_from_payloads(
                requests,
                *request_rows,
                context="targeted_component_correction_materialization_queue",
            )
        ),
        "response_harvest_path": _repo_rel(response_harvest_path, repo),
        "request_row_count": len(request_rows),
        "ready_for_materializer_execution": False,
        "ready_for_budget_spend": False,
        "budget_spend_allowed": False,
        "allowed_use": (
            "targeted_component_correction_materialization_queue_metadata_only"
        ),
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        queue["metadata"],
        context="targeted_component_correction_materialization_queue_metadata",
    )
    queue["materialization_request_summary"] = {
        "schema": TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA,
        "row_count": requests.get("row_count"),
        "accepted_response_count": requests.get("accepted_response_count"),
        "frontier_target_optimization_profile": dict(
            _target_profile_metadata_from_payloads(
                requests,
                *request_rows,
                context=(
                    "targeted_component_correction_materialization_queue_summary"
                ),
            )
        ),
        "ready_for_budget_spend_count": 0,
        "allowed_use": (
            "targeted_component_correction_materialization_queue_summary_only"
        ),
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        queue["materialization_request_summary"],
        context="targeted_component_correction_materialization_queue_summary",
    )
    return queue


def _targeted_component_chain_rate_targets(row: Mapping[str, Any]) -> list[str]:
    dimensions = set(_string_list(row.get("targeted_dimensions")))
    levels = set(_string_list(row.get("operation_levels")))
    surfaces = dimensions | levels
    targets: list[str] = []
    if {"bit", "byte", "packet_member"} & surfaces or int(row.get("saved_bytes_budget") or 0) > 0:
        targets.extend(
            [
                RENDERER_PAYLOAD_DFL1_TARGET_KIND,
                PACKET_MEMBER_MERGE_TARGET_KIND,
                PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
                BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
                PACKET_MEMBER_REORDER_TARGET_KIND,
            ]
        )
    if {"archive_section", "bit", "byte"} & surfaces or int(row.get("saved_bytes_budget") or 0) > 0:
        targets.extend(
            [
                ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
                ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND,
                ARCHIVE_SECTION_REORDER_TARGET_KIND,
                ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND,
            ]
        )
    if {"tensor_channel", "training_substrate", "full_video"} & surfaces:
        targets.extend(
            [
                TENSOR_FACTORIZE_TARGET_KIND,
                TENSOR_QUANTIZE_TARGET_KIND,
                TENSOR_PRUNE_TARGET_KIND,
                TENSOR_SHARED_CODEBOOK_TARGET_KIND,
            ]
        )
    if {"pixel", "region", "boundary", "frame", "pair", "batch", "scorer_axis"} & surfaces:
        targets.append(INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND)
    return _unique_strings(targets)


def _targeted_component_chain_stage_plan(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    families = _unique_strings(row.get("accepted_correction_families") or [])
    rate_targets = _targeted_component_chain_rate_targets(row)
    component_response_targets = [
        "segnet_component_response",
        "posenet_component_response",
        "full_video_lagrangian_response",
    ]
    return [
        {
            "stage": "scorer_sensitive_operation_selection",
            "targets": families,
            "required_before_execution": [
                "paired_cpu_mlx_delta_model",
                "master_gradient_or_component_marginal_model",
                "chain_synergy_antagonism_model",
            ],
        },
        {
            "stage": "receiver_consumed_correction_synthesis",
            "targets": families,
            "required_before_execution": [
                "source_runtime_adapter_per_operation",
                "single_composed_receiver_runtime_consumption_proof",
                "parser_only_proof_rejected",
            ],
        },
        {
            "stage": "payload_grammar_and_entropy",
            "targets": rate_targets,
            "required_before_execution": [
                "rate_payload_pack_and_materialize",
                "materializer_context_binding",
                "byte_closed_archive_export",
                "full_frame_inflate_parity",
            ],
        },
        {
            "stage": "component_guarded_budget_replay",
            "targets": component_response_targets,
            "required_before_execution": [
                "segnet_posenet_component_eval",
                "total_lagrangian_improvement",
                "exact_readiness_bridge",
                "budget_spend_gate",
            ],
        },
    ]


def _targeted_component_chain_targets(row: Mapping[str, Any]) -> list[str]:
    targets: list[str] = []
    for stage in _targeted_component_chain_stage_plan(row):
        targets.extend(_string_list(stage.get("targets")))
    return _unique_strings(targets)


def _targeted_component_chain_work_order(
    row: Mapping[str, Any],
    *,
    rank: int,
) -> dict[str, Any]:
    request_id = str(row.get("materialization_request_id") or f"request_{rank}")
    target_profile_metadata = _target_profile_metadata_from_payloads(
        row,
        context=f"targeted_component_chain_work_order:{request_id}",
    )
    source_operation_id = f"targeted_component_chain_{_slug_token(request_id)}"
    runtime_binding_context = dict(
        row.get("receiver_runtime_binding_context")
        if isinstance(row.get("receiver_runtime_binding_context"), Mapping)
        else {}
    )
    materializer_chain_basis = [
        dict(entry)
        for entry in row.get("materializer_chain_basis") or []
        if isinstance(entry, Mapping)
    ]
    paired_delta_basis = [
        {
            "schema": "targeted_component_chain_paired_delta_basis.v1",
            "source_acquisition_id": entry.get("source_acquisition_id"),
            "correction_family": entry.get("correction_family"),
            "measured_lagrangian_delta_score_units": entry.get(
                "measured_lagrangian_delta_score_units"
            ),
            "measured_component_delta_score_units": entry.get(
                "measured_component_delta_score_units"
            ),
            "budget_credit_remaining_score_units": entry.get(
                "budget_credit_remaining_score_units"
            ),
            "saved_bytes_budget": entry.get("saved_bytes_budget"),
            "receiver_closed_saved_bytes": entry.get("receiver_closed_saved_bytes"),
            **_targeted_rate_packet_context_fields(entry),
            "receiver_closed_rate_packet_context": dict(
                entry.get("receiver_closed_rate_packet_context")
                if isinstance(entry.get("receiver_closed_rate_packet_context"), Mapping)
                else {}
            ),
            "local_cpu_score_delta_summary": dict(
                entry.get("local_cpu_score_delta_summary")
                if isinstance(entry.get("local_cpu_score_delta_summary"), Mapping)
                else {}
            ),
            "local_mlx_score_delta_summary": dict(
                entry.get("local_mlx_score_delta_summary")
                if isinstance(entry.get("local_mlx_score_delta_summary"), Mapping)
                else {}
            ),
            "allowed_use": "targeted_component_chain_paired_delta_basis_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            **FALSE_AUTHORITY,
        }
        for entry in materializer_chain_basis
    ]
    source_blockers = _unique_strings(
        [
            "targeted_component_chain_requires_receiver_runtime_materializer",
            "targeted_component_chain_requires_composed_inflate_parity",
            "targeted_component_chain_requires_exact_axis_component_response",
            "targeted_component_chain_requires_budget_spend_gate",
            *_string_list(row.get("budget_spend_blockers")),
        ]
    )
    work_order = {
        "schema": OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA,
        "source_operation_id": source_operation_id,
        "source_operation_family": (
            "targeted_component_correction_receiver_consumed_multi_op_chain"
        ),
        "source_materialization_request_id": request_id,
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "chain_targets": _targeted_component_chain_targets(row),
        "required_before_execution": [
            "per_stage_materializer_contexts",
            "single_composed_receiver_runtime_consumption_proof",
            "full_frame_inflate_parity",
            "segnet_posenet_component_eval",
            "chain_exact_readiness_bridge",
            "targeted_component_budget_spend_gate",
        ],
        "targeted_correction_budget": {
            "schema": "frontier_rate_attack_targeted_chain_budget.v1",
            "materialization_request_id": request_id,
            "candidate_id": row.get("candidate_id"),
            "frontier_target_optimization_profile": dict(target_profile_metadata),
            "source_acquisition_ids": list(row.get("source_acquisition_ids") or []),
            "accepted_correction_families": list(
                row.get("accepted_correction_families") or []
            ),
            "accepted_response_count": row.get("accepted_response_count"),
            "operation_levels": list(row.get("operation_levels") or []),
            "targeted_dimensions": list(row.get("targeted_dimensions") or []),
            "saved_bytes_budget": row.get("saved_bytes_budget"),
            "receiver_closed_saved_bytes": row.get("receiver_closed_saved_bytes"),
            "receiver_closed_rate_packet_contexts": [
                dict(context)
                for context in row.get("receiver_closed_rate_packet_contexts") or []
                if isinstance(context, Mapping)
            ],
            "rate_packet_manifest_paths": list(
                row.get("rate_packet_manifest_paths") or []
            ),
            "parent_rate_packet_manifest_paths": list(
                row.get("parent_rate_packet_manifest_paths") or []
            ),
            "candidate_compact_selector_codecs": list(
                row.get("candidate_compact_selector_codecs") or []
            ),
            "parent_compact_selector_codecs": list(
                row.get("parent_compact_selector_codecs") or []
            ),
            "entropy_positions": list(row.get("entropy_positions") or []),
            "estimated_receiver_closed_rate_credit_score_units": row.get(
                "estimated_receiver_closed_rate_credit_score_units"
            ),
            "measured_lagrangian_delta_score_units_sum": row.get(
                "measured_lagrangian_delta_score_units_sum"
            ),
            "best_measured_lagrangian_delta_score_units": row.get(
                "best_measured_lagrangian_delta_score_units"
            ),
            "best_local_cpu_score_delta_summary": dict(
                row.get("best_local_cpu_score_delta_summary")
                if isinstance(row.get("best_local_cpu_score_delta_summary"), Mapping)
                else {}
            ),
            "best_local_mlx_score_delta_summary": dict(
                row.get("best_local_mlx_score_delta_summary")
                if isinstance(row.get("best_local_mlx_score_delta_summary"), Mapping)
                else {}
            ),
            "paired_delta_basis": paired_delta_basis,
            "receiver_runtime_binding_context": runtime_binding_context,
            "candidate_archive_path": runtime_binding_context.get(
                "candidate_archive_path"
            ),
            "candidate_inflate_sh_path": runtime_binding_context.get(
                "candidate_inflate_sh_path"
            ),
            "candidate_submission_dir": runtime_binding_context.get(
                "candidate_submission_dir"
            ),
            "source_archive_path": runtime_binding_context.get(
                "source_archive_path"
            ),
            "source_inflate_sh_path": runtime_binding_context.get(
                "source_inflate_sh_path"
            ),
            "source_submission_dir": runtime_binding_context.get(
                "source_submission_dir"
            ),
            "budget_spend_allowed": False,
            "allowed_use": (
                "targeted_component_chain_budget_for_local_planning_only"
            ),
            "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
            **FALSE_AUTHORITY,
        },
        "stage_plan": _targeted_component_chain_stage_plan(row),
        "source_bridge_blockers": source_blockers,
        "allowed_use": (
            "targeted_component_correction_operation_chain_work_order_planning_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        work_order,
        context=f"targeted_component_chain_work_order:{source_operation_id}",
    )
    return work_order


def build_frontier_targeted_component_correction_chain_work_orders(
    *,
    targeted_component_correction_materialization_requests: Mapping[str, Any],
    request_limit: int = 4,
) -> dict[str, Any]:
    """Compile grouped correction requests into operation-chain work orders."""

    if request_limit < 1:
        raise FrontierRateAttackFeedbackError("request_limit must be >= 1")
    if (
        targeted_component_correction_materialization_requests.get("schema")
        != TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "targeted component correction materialization requests schema mismatch"
        )
    require_no_truthy_authority_fields(
        targeted_component_correction_materialization_requests,
        context="targeted_component_correction_chain_work_order_requests",
    )
    request_rows = [
        row
        for row in targeted_component_correction_materialization_requests.get("rows")
        or []
        if isinstance(row, Mapping)
    ][:request_limit]
    target_profile_metadata = _target_profile_metadata_from_payloads(
        targeted_component_correction_materialization_requests,
        *request_rows,
        context="targeted_component_correction_chain_work_orders",
    )
    work_orders: list[dict[str, Any]] = []
    for rank, row in enumerate(request_rows, start=1):
        require_no_truthy_authority_fields(
            row,
            context=f"targeted_component_chain_request_row:{rank}",
        )
        work_orders.append(_targeted_component_chain_work_order(row, rank=rank))
    blockers = ["exact_auth_eval_required_before_score_or_promotion_claim"]
    if not work_orders:
        blockers.append("no_targeted_component_materialization_request_rows")
    else:
        blockers.extend(
            [
                "receiver_consumed_correction_materializer_missing",
                "single_composed_runtime_consumption_proof_required",
                "exact_axis_component_response_required_before_budget_spend",
            ]
        )
    payload = {
        "schema": OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_schema": (
            targeted_component_correction_materialization_requests.get("schema")
        ),
        "source_family": "targeted_component_correction_materialization_requests",
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "active": bool(work_orders),
        "work_order_count": len(work_orders),
        "request_count": len(request_rows),
        "request_limit": request_limit,
        "top_source_operation_ids": [
            str(row.get("source_operation_id") or "") for row in work_orders[:8]
        ],
        "top_materialization_request_ids": [
            str(row.get("source_materialization_request_id") or "")
            for row in work_orders[:8]
        ],
        "blockers": _unique_strings(blockers),
        "work_orders": work_orders,
        "allowed_use": (
            "targeted_component_correction_chain_work_orders_for_queue_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="targeted_component_correction_chain_work_orders",
    )
    return payload


def _targeted_chain_required_context_fields(
    *,
    adapter: Mapping[str, Any],
    target_kind: str,
) -> list[str]:
    fields = list(adapter.get("required_context_fields") or [])
    if target_kind == BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND:
        fields.extend(
            [
                "archive_path",
                "schema_manifest",
                "beam_probe_reports",
                "source_runtime_dir",
                "output_dir",
                "runtime_consumption_proof",
            ]
        )
    elif target_kind == PACKET_MEMBER_REORDER_TARGET_KIND:
        fields.extend(
            [
                "archive_path",
                "member_order_contract",
                "runtime_consumption_proof",
            ]
        )
    elif target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        fields.extend(
            [
                "renderer_payload_dfl1_source_runtime_dir",
                "renderer_payload_dfl1_candidate_runtime_dir",
                "renderer_payload_dfl1_full_frame_file_list_or_entries",
                "renderer_payload_dfl1_expected_full_frame_file_list_sha256",
                "renderer_payload_dfl1_expected_full_frame_entry_count",
                "renderer_payload_dfl1_full_frame_file_list_source",
            ]
        )
    elif target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        fields.extend(["tensor_manifest", "factorization_contract_or_rank"])
    elif target_kind == TENSOR_QUANTIZE_TARGET_KIND:
        fields.extend(["tensor_manifest", "quantization_contract"])
    elif target_kind == TENSOR_PRUNE_TARGET_KIND:
        fields.extend(["tensor_manifest", "pruning_contract"])
    elif target_kind == TENSOR_SHARED_CODEBOOK_TARGET_KIND:
        fields.extend(["tensor_manifest", "codebook_contract"])
    elif target_kind == ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND:
        fields.extend(["section_manifest", "header_elision_contract"])
    elif target_kind == ARCHIVE_SECTION_REORDER_TARGET_KIND:
        fields.extend(["section_manifest", "section_order_contract"])
    elif target_kind == ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND:
        fields.extend(["section_manifest", "procedural_receiver_spec"])
    elif target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND:
        fields.extend(["section_manifest"])
    elif target_kind == INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND:
        fields.extend(
            [
                "candidate_family",
                "archive_grammar",
                "receiver_contract_kind",
                "operation_set_compiler",
                "runtime_consumption_proof",
            ]
        )
    return _unique_strings(fields)


def _targeted_chain_context_field_present(
    hints: Mapping[str, Any],
    field: str,
) -> bool:
    aliases = {
        "archive_member_name": ("archive_member_name", "member_name"),
        "archive_byte_range": ("archive_byte_range", "byte_range"),
        "factorization_contract_or_rank": ("factorization_contract", "rank"),
        "output_manifest": ("output_manifest", "manifest_out", "json_out"),
        "packet_member_merge_source_runtime_dir": (
            "packet_member_merge_source_runtime_dir",
            "source_runtime_dir",
            "inflate_runtime_dir",
        ),
        "renderer_payload_dfl1_full_frame_file_list_or_entries": (
            "renderer_payload_dfl1_full_frame_file_list",
            "full_frame_file_list",
            "renderer_payload_dfl1_full_frame_file_list_entries",
            "full_frame_file_list_entries",
        ),
    }
    keys = aliases.get(field, (field,))
    for key in keys:
        value = hints.get(key)
        if value in (None, ""):
            continue
        if (
            isinstance(value, Sequence)
            and not isinstance(
                value,
                (str, bytes, bytearray),
            )
            and len(value) == 0
        ):
            continue
        return True
    return False


def _targeted_chain_receiver_proof_request(
    *,
    target_kind: str,
    adapter: Mapping[str, Any],
    output_hint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "targeted_component_chain_receiver_proof_request.v1",
        "target_kind": target_kind,
        "receiver_contract_id": adapter.get("receiver_contract_id"),
        "receiver_contract_kind": adapter.get("receiver_contract_kind"),
        "runtime_consumption_proof": output_hint.get("runtime_consumption_proof"),
        "runtime_consumption_proof_out": output_hint.get(
            "runtime_consumption_proof_out"
        ),
        "source_runtime_dir": output_hint.get("source_runtime_dir"),
        "candidate_runtime_dir": output_hint.get("candidate_runtime_dir"),
        "parser_only_proof_rejected": True,
        "full_frame_inflate_parity_required": True,
        "component_replay_required_before_budget_spend": True,
        "exact_auth_eval_required_before_score_claim": True,
        "allowed_use": "targeted_chain_receiver_proof_request_only",
        "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
        **FALSE_AUTHORITY,
    }


def _targeted_chain_context_closure_plan(
    *,
    work_order: Mapping[str, Any],
    target_kind: str,
    adapter: Mapping[str, Any],
    output_hint: Mapping[str, Any],
) -> dict[str, Any]:
    required = _targeted_chain_required_context_fields(
        adapter=adapter,
        target_kind=target_kind,
    )
    missing = [
        field
        for field in required
        if not _targeted_chain_context_field_present(output_hint, field)
    ]
    context_blockers = _unique_strings(_string_list(output_hint.get("context_blockers")))
    runtime_binding = {}
    budget = work_order.get("targeted_correction_budget")
    if isinstance(budget, Mapping) and isinstance(
        budget.get("receiver_runtime_binding_context"),
        Mapping,
    ):
        runtime_binding = dict(budget["receiver_runtime_binding_context"])
    return {
        "schema": TARGETED_COMPONENT_CHAIN_MATERIALIZER_CONTEXT_CLOSURE_PLAN_SCHEMA,
        "source_operation_id": work_order.get("source_operation_id"),
        "source_materialization_request_id": work_order.get(
            "source_materialization_request_id"
        ),
        "target_kind": target_kind,
        "materializer_id": adapter.get("materializer_id"),
        "receiver_contract_id": adapter.get("receiver_contract_id"),
        "receiver_contract_kind": adapter.get("receiver_contract_kind"),
        "required_context_fields": required,
        "provided_context_fields": [
            field
            for field in required
            if _targeted_chain_context_field_present(output_hint, field)
        ],
        "missing_context_fields": missing,
        "context_blockers": context_blockers,
        "receiver_runtime_binding_context": runtime_binding,
        "receiver_proof_request": _targeted_chain_receiver_proof_request(
            target_kind=target_kind,
            adapter=adapter,
            output_hint=output_hint,
        ),
        "next_required_action": (
            "run_receiver_consumed_materializer"
            if not missing and not context_blockers
            else "fill_missing_context_fields_then_run_receiver_consumed_materializer"
        ),
        "ready_for_materializer_execution": not missing and not context_blockers,
        "ready_for_budget_spend": False,
        "budget_spend_allowed": False,
        "allowed_use": "targeted_chain_materializer_context_closure_only",
        "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
        **FALSE_AUTHORITY,
    }


def _bind_renderer_payload_dfl1_file_list_context(
    output_hint: dict[str, Any],
    *,
    repo_root: Path,
    full_frame_file_list: str | Path | None,
) -> None:
    if _targeted_chain_context_field_present(
        output_hint,
        "renderer_payload_dfl1_full_frame_file_list_or_entries",
    ):
        return
    file_list_path = (
        Path(full_frame_file_list)
        if full_frame_file_list is not None
        else DEFAULT_TARGETED_CHAIN_FULL_FRAME_FILE_LIST
    )
    resolved = _resolve_path(file_list_path, repo_root=repo_root)
    if not resolved.is_file():
        return
    entries = _normalized_file_list_entries(resolved)
    if not entries:
        return
    output_basenames = [Path(entry).stem + ".raw" for entry in entries]
    if len(set(output_basenames)) != len(output_basenames):
        return
    rel_path = _repo_rel(resolved, repo_root)
    output_hint.setdefault("renderer_payload_dfl1_full_frame_file_list", rel_path)
    output_hint.setdefault("full_frame_file_list", rel_path)
    output_hint.setdefault(
        "renderer_payload_dfl1_expected_full_frame_file_list_sha256",
        _normalized_file_list_sha256(entries),
    )
    output_hint.setdefault(
        "expected_full_frame_file_list_sha256",
        output_hint["renderer_payload_dfl1_expected_full_frame_file_list_sha256"],
    )
    output_hint.setdefault(
        "renderer_payload_dfl1_expected_full_frame_entry_count",
        len(entries),
    )
    output_hint.setdefault("expected_full_frame_entry_count", len(entries))
    output_hint.setdefault(
        "renderer_payload_dfl1_full_frame_file_list_source",
        rel_path,
    )
    output_hint.setdefault("full_frame_file_list_source", rel_path)


def _bind_renderer_payload_dfl1_archive_context(
    output_hint: dict[str, Any],
    *,
    budget: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    repo_root: Path,
) -> None:
    candidates = [
        (
            "candidate_archive_path",
            budget.get("candidate_archive_path")
            or runtime_binding.get("candidate_archive_path"),
            budget.get("candidate_submission_dir")
            or runtime_binding.get("candidate_submission_dir"),
        ),
        (
            "source_archive_path",
            budget.get("source_archive_path")
            or runtime_binding.get("source_archive_path"),
            budget.get("source_submission_dir")
            or runtime_binding.get("source_submission_dir"),
        ),
        (
            "bound_archive_path",
            output_hint.get("archive_path") or output_hint.get("source_archive"),
            output_hint.get("source_runtime_dir") or output_hint.get("inflate_runtime_dir"),
        ),
    ]
    for role, archive_path, runtime_dir in candidates:
        if not _zip_contains_all_members(
            archive_path,
            repo_root=repo_root,
            member_names=RENDERER_PAYLOAD_DFL1_REQUIRED_MEMBERS,
        ):
            continue
        output_hint["archive_path"] = archive_path
        output_hint["source_archive"] = archive_path
        if runtime_dir:
            output_hint["source_runtime_dir"] = runtime_dir
            output_hint["inflate_runtime_dir"] = runtime_dir
            output_hint["renderer_payload_dfl1_source_runtime_dir"] = runtime_dir
            output_hint["renderer_payload_dfl1_candidate_runtime_dir"] = runtime_dir
            output_hint["candidate_runtime_dir"] = runtime_dir
        output_hint["renderer_payload_dfl1_source_archive_role"] = role
        output_hint["renderer_payload_dfl1_required_members"] = list(
            RENDERER_PAYLOAD_DFL1_REQUIRED_MEMBERS
        )
        return
    output_hint.setdefault("context_blockers", []).append(
        "renderer_payload_dfl1_source_archive_missing_required_members:"
        + ",".join(RENDERER_PAYLOAD_DFL1_REQUIRED_MEMBERS)
    )


def _targeted_chain_materializer_portfolio_row(
    *,
    work_order: Mapping[str, Any],
    target_kind: str,
    adapter: Mapping[str, Any],
    rank: int,
    default_output_root: str | Path | None,
    repo_root: Path,
    full_frame_file_list: str | Path | None,
) -> dict[str, Any]:
    source_operation_id = str(work_order.get("source_operation_id") or f"chain_{rank}")
    budget = (
        work_order.get("targeted_correction_budget")
        if isinstance(work_order.get("targeted_correction_budget"), Mapping)
        else {}
    )
    target_profile_metadata = _target_profile_metadata_from_payloads(
        work_order,
        budget,
        context=f"targeted_chain_materializer_portfolio_row:{source_operation_id}",
    )
    saved_bytes = _finite_int_or_none(budget.get("saved_bytes_budget")) or 0
    lagrangian = _finite_float_or_none(
        budget.get("measured_lagrangian_delta_score_units_sum")
    )
    priority = abs(float(lagrangian or 0.0)) + float(max(saved_bytes, 0)) * 1e-6
    output_hint: dict[str, Any] = {}
    target_output_root: Path | None = None
    if default_output_root is not None:
        target_output_root = (
            Path(default_output_root)
            / "targeted_component_chain_materializers"
            / _slug_token(source_operation_id)
            / _slug_token(target_kind)
        )
        output_hint["output_dir"] = str(target_output_root)
        output_hint["output_archive"] = str(target_output_root / "candidate.zip")
        output_hint["output_manifest"] = str(target_output_root / "manifest.json")
        output_hint["runtime_consumption_proof_out"] = str(
            target_output_root / "runtime_consumption_proof.json"
        )
    runtime_binding = (
        budget.get("receiver_runtime_binding_context")
        if isinstance(budget.get("receiver_runtime_binding_context"), Mapping)
        else {}
    )
    archive_path = (
        budget.get("candidate_archive_path")
        or runtime_binding.get("candidate_archive_path")
        or budget.get("source_archive_path")
        or runtime_binding.get("source_archive_path")
    )
    runtime_dir = (
        budget.get("candidate_submission_dir")
        or runtime_binding.get("candidate_submission_dir")
        or budget.get("source_submission_dir")
        or runtime_binding.get("source_submission_dir")
    )
    if archive_path:
        output_hint["archive_path"] = archive_path
        output_hint["source_archive"] = archive_path
    output_hint["targeted_correction_budget"] = dict(budget)
    output_hint["frontier_target_optimization_profile"] = dict(
        target_profile_metadata
    )
    output_hint["receiver_closed_rate_packet_contexts"] = [
        dict(context)
        for context in budget.get("receiver_closed_rate_packet_contexts") or []
        if isinstance(context, Mapping)
    ]
    if output_hint["receiver_closed_rate_packet_contexts"]:
        output_hint["receiver_closed_rate_packet_context"] = dict(
            output_hint["receiver_closed_rate_packet_contexts"][0]
        )
    output_hint["rate_packet_manifest_paths"] = list(
        budget.get("rate_packet_manifest_paths") or []
    )
    output_hint["candidate_compact_selector_codecs"] = list(
        budget.get("candidate_compact_selector_codecs") or []
    )
    output_hint["entropy_positions"] = list(budget.get("entropy_positions") or [])
    output_hint["target_kind"] = target_kind
    output_hint["materializer_id"] = adapter.get("materializer_id")
    output_hint["receiver_contract_id"] = adapter.get("receiver_contract_id")
    output_hint["receiver_contract_kind"] = adapter.get("receiver_contract_kind")
    output_hint["candidate_family"] = (
        "targeted_component_correction_chain"
    )
    output_hint["accepted_correction_families"] = list(
        budget.get("accepted_correction_families") or []
    )
    if runtime_dir:
        output_hint["source_runtime_dir"] = runtime_dir
        output_hint["inflate_runtime_dir"] = runtime_dir
        output_hint["packet_member_merge_source_runtime_dir"] = runtime_dir
        output_hint["tensor_source_runtime_dir"] = runtime_dir
        output_hint["renderer_payload_dfl1_source_runtime_dir"] = runtime_dir
    candidate_runtime = (
        budget.get("candidate_submission_dir")
        or runtime_binding.get("candidate_submission_dir")
    )
    if candidate_runtime:
        output_hint["candidate_runtime_dir"] = candidate_runtime
        output_hint["renderer_payload_dfl1_candidate_runtime_dir"] = (
            candidate_runtime
        )
    if target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        _bind_renderer_payload_dfl1_archive_context(
            output_hint,
            budget=budget,
            runtime_binding=runtime_binding,
            repo_root=repo_root,
        )
        _bind_renderer_payload_dfl1_file_list_context(
            output_hint,
            repo_root=repo_root,
            full_frame_file_list=full_frame_file_list,
        )
    closure_plan = _targeted_chain_context_closure_plan(
        work_order=work_order,
        target_kind=target_kind,
        adapter=adapter,
        output_hint=output_hint,
    )
    output_hint["receiver_runtime_binding_context"] = dict(runtime_binding)
    output_hint["targeted_chain_context_closure_plan"] = closure_plan
    output_hint["targeted_chain_receiver_proof_request"] = dict(
        closure_plan["receiver_proof_request"]
    )
    return {
        "operation_id": (
            f"{source_operation_id}:{_slug_token(target_kind)}:{rank:03d}"
        ),
        "operation_family": (
            "targeted_component_correction_receiver_consumed_multi_op_chain"
        ),
        "operation_levels": list(budget.get("operation_levels") or []),
        "queue_consumer": "byte_shaving_campaign_queue",
        "recommended_next_action": (
            "bind_targeted_chain_materializer_context_and_receiver_proof"
        ),
        "priority_score": priority,
        "evidence_sources": [source_operation_id],
        "evidence_summary": {
            "schema": "targeted_component_chain_materializer_handoff_evidence.v1",
            "target_kind": target_kind,
            "source_operation_id": source_operation_id,
            "source_materialization_request_id": work_order.get(
                "source_materialization_request_id"
            ),
            "frontier_target_optimization_profile": dict(target_profile_metadata),
            "targeted_correction_budget": dict(budget),
            "chain_targets": list(work_order.get("chain_targets") or []),
            "best_context_hint": output_hint,
            "context_closure_plan": closure_plan,
            "candidate_saved_bytes": saved_bytes,
            **FALSE_AUTHORITY,
        },
        "blockers": _unique_strings(
            [
                "targeted_chain_materializer_context_binding_required",
                "targeted_chain_single_runtime_consumption_proof_required",
                "targeted_chain_component_replay_required_before_budget_spend",
                "targeted_chain_exact_readiness_bridge_required",
                *_string_list(work_order.get("source_bridge_blockers")),
            ]
        ),
        "queue_executable": False,
        "source_kind": "targeted_component_operation_chain",
        **FALSE_AUTHORITY,
    }


def build_frontier_targeted_component_correction_chain_materializer_handoff(
    *,
    repo_root: str | Path,
    targeted_component_correction_chain_work_orders: Mapping[str, Any],
    default_output_root: str | Path | None = None,
    target_limit: int | None = None,
    full_frame_file_list: str | Path | None = None,
) -> dict[str, Any]:
    """Bind targeted multi-op chains into typed materializer work surfaces."""

    repo = Path(repo_root)
    if target_limit is not None and target_limit < 1:
        raise FrontierRateAttackFeedbackError("target_limit must be >= 1")
    if (
        targeted_component_correction_chain_work_orders.get("schema")
        != OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA
    ):
        raise FrontierRateAttackFeedbackError(
            "targeted component correction chain work orders schema mismatch"
        )
    require_no_truthy_authority_fields(
        targeted_component_correction_chain_work_orders,
        context="targeted_component_chain_materializer_handoff_work_orders",
    )
    target_profile_metadata = _target_profile_metadata_from_payloads(
        targeted_component_correction_chain_work_orders,
        context="targeted_component_chain_materializer_handoff",
    )
    adapters_by_target = _materializer_registry_adapters_by_target()
    backlog_rows: list[dict[str, Any]] = []
    registered_targets: list[str] = []
    unregistered_targets: list[str] = []
    source_operation_ids: list[str] = []
    rank = 1
    for work_order in targeted_component_correction_chain_work_orders.get(
        "work_orders"
    ) or []:
        if not isinstance(work_order, Mapping):
            continue
        require_no_truthy_authority_fields(
            work_order,
            context=f"targeted_component_chain_materializer_handoff.row:{rank}",
        )
        source_operation_ids.append(str(work_order.get("source_operation_id") or ""))
        for target in _string_list(work_order.get("chain_targets")):
            adapter = adapters_by_target.get(target)
            if adapter is None:
                unregistered_targets.append(target)
                continue
            if target_limit is not None and len(backlog_rows) >= target_limit:
                continue
            portfolio_row = _targeted_chain_materializer_portfolio_row(
                work_order=work_order,
                target_kind=target,
                adapter=adapter,
                rank=rank,
                default_output_root=default_output_root,
                repo_root=repo,
                full_frame_file_list=full_frame_file_list,
            )
            backlog_rows.append(
                _portfolio_materializer_backlog_row(
                    row=portfolio_row,
                    adapter=adapter,
                    rank=rank,
                )
            )
            registered_targets.append(target)
            rank += 1
    materializer_backlog = {
        "schema": MATERIALIZER_BACKLOG_SCHEMA,
        "tool": "comma_lab.scheduler.frontier_rate_attack_feedback",
        "generated_at_utc": _utc_now(),
        "source_schema": targeted_component_correction_chain_work_orders.get(
            "schema"
        ),
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "backlog_row_count": len(backlog_rows),
        "registered_chain_target_count": len(_unique_strings(registered_targets)),
        "unregistered_chain_target_count": len(_unique_strings(unregistered_targets)),
        "rows": backlog_rows,
        "allowed_use": (
            "targeted_component_chain_to_materializer_backlog_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    materializer_contexts = build_final_byte_operation_contexts(
        materializer_backlog,
        artifact_map=None,
        repo_root=repo_root,
        default_output_root=default_output_root,
    )
    materializer_work_queue = build_materializer_work_queue(
        materializer_backlog,
        repo_root=repo_root,
        contexts=(
            materializer_contexts_from_payload(materializer_contexts)
            if backlog_rows
            else {}
        ),
        source_plan_path=None,
        limit=target_limit,
    )
    context_closure_plans = [
        dict(params["targeted_chain_context_closure_plan"])
        for row in backlog_rows
        for params in (row.get("operation_params"),)
        if isinstance(params, Mapping)
        and isinstance(params.get("targeted_chain_context_closure_plan"), Mapping)
    ]
    payload = {
        "schema": TARGETED_COMPONENT_CORRECTION_CHAIN_MATERIALIZER_HANDOFF_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_schema": targeted_component_correction_chain_work_orders.get(
            "schema"
        ),
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "source_operation_ids": _unique_strings(source_operation_ids),
        "registered_chain_targets": _unique_strings(registered_targets),
        "unregistered_chain_targets": _unique_strings(unregistered_targets),
        "registered_chain_target_count": len(_unique_strings(registered_targets)),
        "unregistered_chain_target_count": len(_unique_strings(unregistered_targets)),
        "materializer_backlog_row_count": materializer_backlog.get(
            "backlog_row_count"
        ),
        "context_row_count": materializer_contexts.get("row_count"),
        "blocked_context_count": materializer_contexts.get("blocked_context_count"),
        "work_queue_row_count": materializer_work_queue.get("row_count"),
        "executable_work_row_count": materializer_work_queue.get(
            "executable_row_count"
        ),
        "blocked_work_row_count": materializer_work_queue.get("blocked_row_count"),
        "context_closure_plan_count": len(context_closure_plans),
        "context_closure_plans": context_closure_plans,
        "materializer_backlog": materializer_backlog,
        "materializer_contexts": materializer_contexts,
        "materializer_work_queue": materializer_work_queue,
        "blockers": _unique_strings(
            [
                "targeted_chain_materializer_contexts_require_receiver_custody",
                "targeted_chain_component_replay_required_before_budget_spend",
                "exact_auth_eval_required_before_score_or_promotion_claim",
                *[
                    f"unregistered_chain_target:{target}"
                    for target in _unique_strings(unregistered_targets)
                ],
            ]
        ),
        "allowed_use": (
            "targeted_component_chain_materializer_handoff_for_queue_only"
        ),
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="targeted_component_chain_materializer_handoff",
    )
    return payload


def _operation_chain_work_order_by_id(
    operation_chain_compiler_work_orders: Mapping[str, Any],
    source_operation_id: str,
) -> Mapping[str, Any]:
    for row in operation_chain_compiler_work_orders.get("work_orders") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("source_operation_id") or "") == source_operation_id:
            return row
    raise FrontierRateAttackFeedbackError(
        f"operation chain compiler work order not found: {source_operation_id}"
    )


def _operation_chain_stage_rows(work_order: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_profile_metadata = _target_profile_metadata_from_payloads(
        work_order,
        work_order.get("targeted_correction_budget")
        if isinstance(work_order.get("targeted_correction_budget"), Mapping)
        else None,
        context="operation_chain_stage_rows",
    )
    for index, stage in enumerate(work_order.get("stage_plan") or [], start=1):
        if not isinstance(stage, Mapping):
            continue
        stage_id = str(stage.get("stage") or f"stage_{index:02d}")
        required = _string_list(stage.get("required_before_execution"))
        blockers = [
            f"{stage_id}_requires:{requirement}"
            for requirement in required
        ]
        rows.append(
            {
                "schema": "frontier_rate_attack_operation_chain_stage_row.v1",
                "stage_index": index,
                "stage_id": stage_id,
                "frontier_target_optimization_profile": dict(
                    target_profile_metadata
                ),
                "targets": _string_list(stage.get("targets")),
                "required_before_execution": required,
                "stage_ready_for_execution": False,
                "blockers": blockers,
                "allowed_use": "operation_chain_stage_planning_only",
                "forbidden_use": "score_claim_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
    return rows


def build_frontier_operation_chain_compiler_stage_plan(
    *,
    operation_chain_compiler_work_orders: Mapping[str, Any],
    source_operation_id: str,
) -> dict[str, Any]:
    """Build a staged queue-owned plan from one multisurface chain work order."""

    require_no_truthy_authority_fields(
        operation_chain_compiler_work_orders,
        context="operation_chain_compiler_work_orders",
    )
    work_order = _operation_chain_work_order_by_id(
        operation_chain_compiler_work_orders,
        source_operation_id,
    )
    require_no_truthy_authority_fields(
        work_order,
        context=f"operation_chain_compiler_work_order:{source_operation_id}",
    )
    stage_rows = _operation_chain_stage_rows(work_order)
    source_blockers = _string_list(work_order.get("source_bridge_blockers"))
    missing_contracts = [
        blocker.removeprefix("chain_missing_contract:")
        for blocker in source_blockers
        if blocker.startswith("chain_missing_contract:")
    ]
    target_kinds = _unique_strings(work_order.get("chain_targets") or [])
    target_profile_metadata = _target_profile_metadata_from_payloads(
        work_order,
        work_order.get("targeted_correction_budget")
        if isinstance(work_order.get("targeted_correction_budget"), Mapping)
        else None,
        context=f"operation_chain_compiler_stage_plan:{source_operation_id}",
    )
    stage_target_counts: dict[str, int] = {}
    for row in stage_rows:
        for target in _string_list(row.get("targets")):
            stage_target_counts[target] = stage_target_counts.get(target, 0) + 1
    uncovered_targets = [
        target for target in target_kinds if target not in stage_target_counts
    ]
    blockers = [
        "operation_chain_stage_plan_requires_materializer_context_binding",
        "operation_chain_stage_plan_requires_single_runtime_consumption_proof",
        "operation_chain_stage_plan_requires_exact_readiness_bridge",
        "operation_chain_stage_plan_requires_targeted_component_budget_spend_gate",
        *source_blockers,
    ]
    if uncovered_targets:
        blockers.extend(
            f"operation_chain_target_missing_stage:{target}"
            for target in uncovered_targets
        )
    plan = {
        "schema": OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_operation_id": source_operation_id,
        "source_operation_family": work_order.get("source_operation_family"),
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "chain_targets": target_kinds,
        "stage_count": len(stage_rows),
        "covered_target_count": len(stage_target_counts),
        "uncovered_chain_targets": uncovered_targets,
        "missing_contracts": _unique_strings(missing_contracts),
        "required_before_execution": _string_list(
            work_order.get("required_before_execution")
        ),
        "targeted_correction_budget": dict(
            work_order.get("targeted_correction_budget")
            if isinstance(work_order.get("targeted_correction_budget"), Mapping)
            else {}
        ),
        "stage_rows": stage_rows,
        "queue_handoffs": [
            {
                "queue_consumer": "byte_shaving_campaign_queue",
                "handoff_reason": "bind_per_stage_materializer_contexts",
                "frontier_target_optimization_profile": dict(
                    target_profile_metadata
                ),
                "target_kinds": target_kinds,
                **FALSE_AUTHORITY,
            },
            {
                "queue_consumer": "frontier_receiver_repair_queue",
                "handoff_reason": "prove_single_runtime_consumption_after_composition",
                "frontier_target_optimization_profile": dict(
                    target_profile_metadata
                ),
                "required_before_budget_spend": True,
                **FALSE_AUTHORITY,
            },
            {
                "queue_consumer": "frontier_targeted_component_correction_queue",
                "handoff_reason": "spend_receiver_closed_rate_budget_only_after_component_eval",
                "frontier_target_optimization_profile": dict(
                    target_profile_metadata
                ),
                "budget_spend_allowed": False,
                **FALSE_AUTHORITY,
            },
        ],
        "source_bridge_blockers": source_blockers,
        "blockers": _unique_strings(blockers),
        "execution_ready": False,
        "allowed_use": "queue_owned_operation_chain_stage_plan_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        plan,
        context=f"operation_chain_compiler_stage_plan:{source_operation_id}",
    )
    return plan


def _stage_row_by_id(
    operation_chain_stage_plan: Mapping[str, Any],
    stage_id: str,
) -> Mapping[str, Any] | None:
    for row in operation_chain_stage_plan.get("stage_rows") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("stage_id") or "") == stage_id:
            return row
    return None


def _first_existing_repo_file(
    repo: Path,
    candidates: Sequence[str | Path],
) -> Path | None:
    for candidate in candidates:
        path = _resolve_path(candidate, repo_root=repo)
        if path.is_file():
            return path
    return None


def _existing_repo_files(
    repo: Path,
    candidates: Sequence[str | Path],
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = _resolve_path(candidate, repo_root=repo)
        key = path.resolve(strict=False).as_posix()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _first_existing_byte_range_runtime_dir(
    repo: Path,
    candidates: Sequence[str | Path],
) -> Path | None:
    for candidate in candidates:
        path = _resolve_path(candidate, repo_root=repo)
        if (
            path.is_dir()
            and (path / "inflate.py").is_file()
            and (path / "inflate.sh").is_file()
        ):
            return path
    return None


def _byte_range_source_from_schema_manifest(
    schema_manifest: Path | None,
    *,
    repo_root: Path,
) -> tuple[Path | None, str]:
    if schema_manifest is None:
        return None, ""
    try:
        payload = _load_json(schema_manifest)
    except FrontierRateAttackFeedbackError:
        return None, ""
    source = payload.get("source_archive")
    if not isinstance(source, Mapping):
        return None, ""
    source_path = source.get("path")
    resolved = (
        _resolve_path(str(source_path), repo_root=repo_root)
        if isinstance(source_path, str) and source_path.strip()
        else None
    )
    if resolved is not None and not resolved.is_file():
        resolved = None
    return resolved, str(source.get("member_name") or "")


def _byte_range_member_name_from_single_member_archive(
    source_archive: Path | None,
    *,
    repo_root: Path,
) -> tuple[str, dict[str, Any]]:
    if source_archive is None:
        return "", {
            "status": "not_attempted",
            "reason": "source_archive_missing",
            **FALSE_AUTHORITY,
        }
    record: dict[str, Any] = {
        "source_archive": _repo_rel(source_archive, repo_root),
        **FALSE_AUTHORITY,
    }
    if not source_archive.is_file():
        return "", {
            **record,
            "status": "failed",
            "reason": "source_archive_not_file",
        }
    try:
        archive = read_strict_single_member_zip(source_archive)
    except HnervLowlevelPackError as exc:
        return "", {
            **record,
            "status": "failed",
            "reason": "not_strict_single_member_zip",
            "error": str(exc),
        }
    return archive.member_name, {
        **record,
        "status": "inferred",
        "inference_rule": "strict_single_member_zip",
        "member_name": archive.member_name,
        "archive_bytes": archive.archive_bytes,
        "archive_sha256": archive.archive_sha256,
        "member_bytes": archive.member_bytes,
    }


def _byte_range_path_records(
    paths: Sequence[Path],
    *,
    repo_root: Path,
    kind: str,
) -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "path": _repo_rel(path, repo_root),
            "exists": path.exists(),
            **FALSE_AUTHORITY,
        }
        for path in paths
    ]


def _byte_range_stage_command(
    *,
    schema_manifest: Path,
    beam_probe_reports: Sequence[Path],
    source_runtime_dir: Path,
    output_dir: Path,
    source_archive: Path | None,
    global_combo_report: Path | None,
    member_name: str,
    repo_root: Path,
) -> list[str]:
    command = [
        ".venv/bin/python",
        BYTE_RANGE_CHAIN_TOOL,
        "--schema-manifest",
        _repo_rel(schema_manifest, repo_root),
        "--source-runtime-dir",
        _repo_rel(source_runtime_dir, repo_root),
        "--output-dir",
        _repo_rel(output_dir, repo_root),
        "--overwrite",
    ]
    for report in beam_probe_reports:
        command.extend(["--beam-probe-report", _repo_rel(report, repo_root)])
    if source_archive is not None:
        command.extend(["--source-archive", _repo_rel(source_archive, repo_root)])
    if global_combo_report is not None:
        command.extend(["--global-combo-report", _repo_rel(global_combo_report, repo_root)])
    if member_name:
        command.extend(["--member-name", member_name])
    return command


def _targeted_component_byte_range_binding(
    operation_chain_stage_plan: Mapping[str, Any],
) -> dict[str, Any]:
    budget = (
        operation_chain_stage_plan.get("targeted_correction_budget")
        if isinstance(operation_chain_stage_plan.get("targeted_correction_budget"), Mapping)
        else {}
    )
    runtime_binding = (
        budget.get("receiver_runtime_binding_context")
        if isinstance(budget.get("receiver_runtime_binding_context"), Mapping)
        else {}
    )
    if not isinstance(runtime_binding, Mapping):
        runtime_binding = {}
    return {
        "source_operation_family": operation_chain_stage_plan.get(
            "source_operation_family"
        ),
        "candidate_archive_path": runtime_binding.get("candidate_archive_path")
        or budget.get("candidate_archive_path"),
        "candidate_inflate_sh_path": runtime_binding.get("candidate_inflate_sh_path")
        or budget.get("candidate_inflate_sh_path"),
        "candidate_submission_dir": runtime_binding.get("candidate_submission_dir")
        or budget.get("candidate_submission_dir"),
        "source_archive_path": runtime_binding.get("source_archive_path")
        or budget.get("source_archive_path"),
        "source_inflate_sh_path": runtime_binding.get("source_inflate_sh_path")
        or budget.get("source_inflate_sh_path"),
        "source_submission_dir": runtime_binding.get("source_submission_dir")
        or budget.get("source_submission_dir"),
        "receiver_runtime_binding_context": dict(runtime_binding),
    }


def _bound_path_or_none(
    value: Any,
    *,
    repo_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _resolve_path(value, repo_root=repo_root)


def build_frontier_byte_range_stage_inputs(
    *,
    repo_root: str | Path,
    operation_chain_stage_plan: Mapping[str, Any],
    stage_id: str = "payload_grammar_and_entropy",
    schema_manifest: str | Path | None = None,
    beam_probe_reports: Sequence[str | Path] = (),
    source_runtime_dir: str | Path | None = None,
    source_archive: str | Path | None = None,
    global_combo_report: str | Path | None = None,
    member_name: str | None = None,
    chain_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Bind byte-range entropy-recode stage inputs into a queue-consumable packet."""

    repo = Path(repo_root)
    require_no_truthy_authority_fields(
        operation_chain_stage_plan,
        context="operation_chain_stage_plan",
    )
    if operation_chain_stage_plan.get("schema") != OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA:
        raise FrontierRateAttackFeedbackError(
            "operation chain stage plan schema mismatch"
        )
    stage_row = _stage_row_by_id(operation_chain_stage_plan, stage_id)
    stage_targets = _string_list(stage_row.get("targets")) if stage_row else []
    target_profile_metadata = _target_profile_metadata_from_payloads(
        operation_chain_stage_plan,
        operation_chain_stage_plan.get("targeted_correction_budget")
        if isinstance(operation_chain_stage_plan.get("targeted_correction_budget"), Mapping)
        else None,
        context="byte_range_stage_inputs_target_profile",
    )
    target_present = "byte_range_entropy_recode_v1" in stage_targets
    targeted_binding = _targeted_component_byte_range_binding(
        operation_chain_stage_plan
    )
    source_operation_family = str(
        targeted_binding.get("source_operation_family") or ""
    )
    target_bound_chain = (
        source_operation_family
        == "targeted_component_correction_receiver_consumed_multi_op_chain"
    )
    bound_archive_path = _bound_path_or_none(
        targeted_binding.get("candidate_archive_path")
        or targeted_binding.get("source_archive_path"),
        repo_root=repo,
    )
    bound_runtime_dir = _bound_path_or_none(
        targeted_binding.get("candidate_submission_dir")
        or targeted_binding.get("source_submission_dir"),
        repo_root=repo,
    )
    if bound_runtime_dir is None:
        bound_inflate = _bound_path_or_none(
            targeted_binding.get("candidate_inflate_sh_path")
            or targeted_binding.get("source_inflate_sh_path"),
            repo_root=repo,
        )
        if bound_inflate is not None:
            bound_runtime_dir = bound_inflate.parent
    missing_contracts = _string_list(operation_chain_stage_plan.get("missing_contracts"))
    source_bridge_blockers = _string_list(
        operation_chain_stage_plan.get("source_bridge_blockers")
    )
    unbound_chain_requires_payload_grammar = (
        "payload_grammar_schema_manifest" in missing_contracts
        or "chain_missing_contract:payload_grammar_schema_manifest"
        in source_bridge_blockers
    )
    disable_default_byte_range_context = (
        target_bound_chain or unbound_chain_requires_payload_grammar
    )
    default_context_disable_reason = ""
    if target_bound_chain:
        default_context_disable_reason = "target_bound_chain"
    elif unbound_chain_requires_payload_grammar:
        default_context_disable_reason = "unbound_chain_missing_payload_grammar_contract"

    schema_path = (
        _resolve_path(schema_manifest, repo_root=repo)
        if schema_manifest is not None
        else (
            None
            if disable_default_byte_range_context
            else _first_existing_repo_file(repo, _DEFAULT_BYTE_RANGE_SCHEMA_MANIFEST_PATHS)
        )
    )
    beam_paths = (
        [_resolve_path(path, repo_root=repo) for path in beam_probe_reports]
        if beam_probe_reports
        else (
            []
            if disable_default_byte_range_context
            else _existing_repo_files(repo, _DEFAULT_BYTE_RANGE_BEAM_PROBE_REPORT_PATHS)
        )
    )
    runtime_path = (
        _resolve_path(source_runtime_dir, repo_root=repo)
        if source_runtime_dir is not None
        else (
            bound_runtime_dir
            if bound_runtime_dir is not None
            else (
                None
                if disable_default_byte_range_context
                else _first_existing_byte_range_runtime_dir(
                    repo,
                    _DEFAULT_BYTE_RANGE_SOURCE_RUNTIME_DIR_PATHS,
                )
            )
        )
    )
    default_source_archive, default_member_name = _byte_range_source_from_schema_manifest(
        schema_path if schema_path is not None and schema_path.is_file() else None,
        repo_root=repo,
    )
    source_archive_path = (
        _resolve_path(source_archive, repo_root=repo)
        if source_archive is not None
        else (
            bound_archive_path
            if bound_archive_path is not None
            else (
                None
                if disable_default_byte_range_context
                else default_source_archive
            )
        )
    )
    combo_path = (
        _resolve_path(global_combo_report, repo_root=repo)
        if global_combo_report is not None
        else (
            None
            if disable_default_byte_range_context
            else _first_existing_repo_file(repo, _DEFAULT_BYTE_RANGE_GLOBAL_COMBO_REPORT_PATHS)
        )
    )
    output_dir = (
        _resolve_path(chain_output_dir, repo_root=repo)
        if chain_output_dir is not None
        else _resolve_path(
            "experiments/results/frontier_operation_chain_compiler/"
            f"{_slug_token(stage_id)}_byte_range_entropy_recode_chain",
            repo_root=repo,
        )
    )
    inferred_member_name, member_name_inference = (
        _byte_range_member_name_from_single_member_archive(
            source_archive_path,
            repo_root=repo,
        )
    )
    selected_member_name = str(
        member_name
        or ("" if disable_default_byte_range_context else default_member_name)
        or inferred_member_name
        or ""
    )

    context_blockers: list[str] = []
    if stage_row is None:
        context_blockers.append(f"operation_chain_stage_missing:{stage_id}")
    if not target_present:
        context_blockers.append("byte_range_entropy_recode_target_missing_from_stage")
    if target_bound_chain:
        context_blockers.append(
            "byte_range_stage_default_pr103_context_disabled_for_target_bound_chain"
        )
    elif (
        unbound_chain_requires_payload_grammar
        and schema_manifest is None
        and not beam_probe_reports
        and source_runtime_dir is None
        and source_archive is None
    ):
        context_blockers.append(
            "byte_range_stage_default_pr103_context_disabled_for_unbound_chain"
        )
    if schema_path is None or not schema_path.is_file():
        context_blockers.append("byte_range_stage_missing:schema_manifest")
    if not beam_paths or any(not path.is_file() for path in beam_paths):
        context_blockers.append("byte_range_stage_missing:beam_probe_reports")
    if runtime_path is None or not runtime_path.is_dir():
        context_blockers.append("byte_range_stage_missing:source_runtime_dir")
    elif disable_default_byte_range_context:
        if not (runtime_path / "inflate.sh").is_file():
            context_blockers.append(
                "byte_range_stage_target_runtime_missing_inflate_sh"
            )
    elif (
        not (runtime_path / "inflate.py").is_file()
        or not (runtime_path / "inflate.sh").is_file()
    ):
        context_blockers.append("byte_range_stage_runtime_missing_inflate_entrypoint")
    if source_archive_path is None or not source_archive_path.is_file():
        context_blockers.append("byte_range_stage_missing:source_archive")
    if not selected_member_name:
        context_blockers.append("byte_range_stage_missing:member_name")
    local_chain_queueable = not context_blockers
    chain_command: list[str] = []
    if (
        local_chain_queueable
        and schema_path is not None
        and runtime_path is not None
        and source_archive_path is not None
    ):
        chain_command = _byte_range_stage_command(
            schema_manifest=schema_path,
            beam_probe_reports=beam_paths,
            source_runtime_dir=runtime_path,
            output_dir=output_dir,
            source_archive=source_archive_path,
            global_combo_report=combo_path,
            member_name=selected_member_name,
            repo_root=repo,
        )

    materializer_context = {
        "schema_manifest": _repo_rel(schema_path, repo) if schema_path else "",
        "beam_probe_reports": [_repo_rel(path, repo) for path in beam_paths],
        "source_runtime_dir": _repo_rel(runtime_path, repo) if runtime_path else "",
        "source_archive": (
            _repo_rel(source_archive_path, repo) if source_archive_path else ""
        ),
        "global_combo_report": _repo_rel(combo_path, repo) if combo_path else "",
        "member_name": selected_member_name,
        "member_name_inference": member_name_inference,
        "output_dir": _repo_rel(output_dir, repo),
        "chain_output_dir": _repo_rel(output_dir, repo),
        "fail_if_receiver_blocked": False,
        "context_blockers": _unique_strings(context_blockers),
        "default_pr103_context_disabled": disable_default_byte_range_context,
        "default_pr103_context_disable_reason": default_context_disable_reason,
        "targeted_component_runtime_binding": targeted_binding,
        **FALSE_AUTHORITY,
    }
    payload = {
        "schema": BYTE_RANGE_STAGE_INPUTS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_operation_id": operation_chain_stage_plan.get("source_operation_id"),
        "stage_id": stage_id,
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "stage_targets": stage_targets,
        "target_present": target_present,
        "required_before_execution": (
            _string_list(stage_row.get("required_before_execution"))
            if stage_row
            else []
        ),
        "materializer_target_kind": "byte_range_entropy_recode_v1",
        "materializer_context": materializer_context,
        "input_artifacts": [
            *_byte_range_path_records(
                [schema_path] if schema_path is not None else [],
                repo_root=repo,
                kind="schema_manifest",
            ),
            *_byte_range_path_records(
                beam_paths,
                repo_root=repo,
                kind="beam_probe_report",
            ),
            *_byte_range_path_records(
                [combo_path] if combo_path is not None else [],
                repo_root=repo,
                kind="global_combo_report",
            ),
            *_byte_range_path_records(
                [source_archive_path] if source_archive_path is not None else [],
                repo_root=repo,
                kind="source_archive",
            ),
            *_byte_range_path_records(
                [runtime_path] if runtime_path is not None else [],
                repo_root=repo,
                kind="source_runtime_dir",
            ),
        ],
        "local_chain_queueable": local_chain_queueable,
        "local_chain_command": chain_command,
        "chain_manifest_path": _repo_rel(
            output_dir / "byte_range_entropy_recode_chain_manifest.json",
            repo,
        ),
        "rate_budget_policy": {
            "schema": "frontier_rate_attack_stage_rate_budget_policy.v1",
            "freed_bytes_destination": "targeted_component_correction_acquisition",
            "freed_bytes_can_fund": [
                "segnet_boundary_repair",
                "posenet_geometry_repair",
                "full_video_residual_reallocation",
            ],
            "spend_requires": [
                "receiver_contract_satisfied",
                "exact_readiness_bridge",
                "component_eval_under_segnet_posenet_guard",
                "total_lagrangian_improvement",
            ],
            "budget_spend_allowed": False,
            "allowed_use": (
                "local_byte_range_chain_rate_budget_planning_after_receiver_proof_only"
            ),
            "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
            **FALSE_AUTHORITY,
        },
        "targeted_correction_budget": dict(
            operation_chain_stage_plan.get("targeted_correction_budget")
            if isinstance(
                operation_chain_stage_plan.get("targeted_correction_budget"),
                Mapping,
            )
            else {}
        ),
        "blockers": _unique_strings(
            [
                *context_blockers,
                "byte_range_stage_requires_receiver_proof_after_local_chain",
                "byte_range_stage_requires_exact_readiness_bridge_after_local_chain",
                "byte_range_stage_rate_budget_requires_component_spend_gate",
            ]
        ),
        "exact_execution_ready": False,
        "budget_spend_allowed": False,
        "allowed_use": "queue_owned_byte_range_stage_input_binding_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context=f"byte_range_stage_inputs:{stage_id}",
    )
    return payload


def _targeted_drop_many_stage_command(
    *,
    selector_pareto: Path,
    pair_frame_geometry_lattice: Path,
    output_path: Path,
    drop_many_counts: str,
    max_drop_many: int,
    repo_root: Path,
) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/plan_decoder_q_pairset_acquisition.py",
        "--selector-pareto",
        _repo_rel(selector_pareto, repo_root),
        "--drop-many-counts",
        drop_many_counts,
        "--max-drop-many",
        str(max_drop_many),
        "--pair-frame-geometry-lattice-json",
        _repo_rel(pair_frame_geometry_lattice, repo_root),
        "--json-out",
        _repo_rel(output_path, repo_root),
    ]


def _stage_targets_any(
    operation_chain_stage_plan: Mapping[str, Any],
    *,
    stage_id: str,
) -> list[str]:
    stage_row = _stage_row_by_id(operation_chain_stage_plan, stage_id)
    return _string_list(stage_row.get("targets")) if stage_row else []


def build_frontier_targeted_drop_many_stage_inputs(
    *,
    repo_root: str | Path,
    operation_chain_stage_plan: Mapping[str, Any],
    stage_id: str = "scorer_sensitive_operation_selection",
    selector_pareto: str | Path | None = None,
    pair_frame_geometry_lattice: str | Path | None = None,
    output_dir: str | Path | None = None,
    drop_many_counts: str = "3,4,6,8,12,16",
    max_drop_many: int = 96,
) -> dict[str, Any]:
    """Bind high-level targeted correction families into a drop-many plan step."""

    repo = Path(repo_root)
    require_no_truthy_authority_fields(
        operation_chain_stage_plan,
        context="targeted_drop_many_stage_plan",
    )
    if operation_chain_stage_plan.get("schema") != OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA:
        raise FrontierRateAttackFeedbackError(
            "operation chain stage plan schema mismatch"
        )
    if max_drop_many < 0:
        raise FrontierRateAttackFeedbackError("max_drop_many must be >= 0")
    if not drop_many_counts.strip():
        raise FrontierRateAttackFeedbackError("drop_many_counts must be non-empty")

    stage_row = _stage_row_by_id(operation_chain_stage_plan, stage_id)
    stage_targets = _stage_targets_any(
        operation_chain_stage_plan,
        stage_id=stage_id,
    )
    target_profile_metadata = _target_profile_metadata_from_payloads(
        operation_chain_stage_plan,
        operation_chain_stage_plan.get("targeted_correction_budget")
        if isinstance(operation_chain_stage_plan.get("targeted_correction_budget"), Mapping)
        else None,
        context="targeted_drop_many_stage_inputs_target_profile",
    )
    drop_many_family_targets = [
        "drop_within_selected_set_masked_boundary",
        "inverse_scorer_cell_basis_expansion",
        "pose_stable_pair_frame_motion_correction",
        "full_video_batch_residual_budget_reallocation",
    ]
    selected_family_targets = [
        target for target in drop_many_family_targets if target in stage_targets
    ]
    target_present = bool(selected_family_targets)
    selector_path = (
        _resolve_path(selector_pareto, repo_root=repo)
        if selector_pareto is not None
        else _first_existing_repo_file(repo, _DEFAULT_SELECTOR_PARETO_PATHS)
    )
    lattice_path = (
        _resolve_path(pair_frame_geometry_lattice, repo_root=repo)
        if pair_frame_geometry_lattice is not None
        else _first_existing_repo_file(repo, _DEFAULT_PAIR_FRAME_GEOMETRY_LATTICE_PATHS)
    )
    plan_output_dir = (
        _resolve_path(output_dir, repo_root=repo)
        if output_dir is not None
        else _resolve_path(
            "experiments/results/frontier_operation_chain_compiler/"
            f"{_slug_token(stage_id)}_targeted_drop_many",
            repo_root=repo,
        )
    )
    plan_output_path = plan_output_dir / "targeted_drop_many_pairset_acquisition.json"

    context_blockers: list[str] = []
    selector_summary: dict[str, Any] = {}
    lattice_summary: dict[str, Any] = {}
    if stage_row is None:
        context_blockers.append(f"operation_chain_stage_missing:{stage_id}")
    if not target_present:
        context_blockers.append("targeted_drop_many_family_missing_from_stage")
    if selector_path is None or not selector_path.is_file():
        context_blockers.append("targeted_drop_many_stage_missing:selector_pareto")
    else:
        selector_payload = _load_json(selector_path)
        require_no_truthy_authority_fields(
            selector_payload,
            context=f"targeted_drop_many_selector_pareto:{selector_path}",
        )
        if selector_payload.get("schema") != "decoder_q_selective_selector_pareto.v1":
            context_blockers.append("targeted_drop_many_stage_bad_selector_schema")
        selector_summary = {
            "schema": selector_payload.get("schema"),
            "candidate_count": (
                selector_payload.get("summary", {}).get("candidate_count")
                if isinstance(selector_payload.get("summary"), Mapping)
                else None
            ),
            "recommended_selector_id": (
                selector_payload.get("summary", {}).get("recommended_selector_id")
                if isinstance(selector_payload.get("summary"), Mapping)
                else None
            ),
            **FALSE_AUTHORITY,
        }
    if lattice_path is None or not lattice_path.is_file():
        context_blockers.append(
            "targeted_drop_many_stage_missing:pair_frame_geometry_lattice"
        )
    else:
        lattice_payload = _load_json(lattice_path)
        require_no_truthy_authority_fields(
            lattice_payload,
            context=f"targeted_drop_many_pair_frame_geometry_lattice:{lattice_path}",
        )
        if lattice_payload.get("schema") != PAIR_FRAME_GEOMETRY_LATTICE_SCHEMA:
            context_blockers.append(
                "targeted_drop_many_stage_bad_pair_frame_geometry_lattice_schema"
            )
        lattice_summary = {
            "schema": lattice_payload.get("schema"),
            "row_count": (
                lattice_payload.get("summary", {}).get("row_count")
                if isinstance(lattice_payload.get("summary"), Mapping)
                else None
            ),
            "queue_executable_request_count": (
                lattice_payload.get("summary", {}).get(
                    "queue_executable_request_count"
                )
                if isinstance(lattice_payload.get("summary"), Mapping)
                else None
            ),
            "drop_counts": (
                lattice_payload.get("summary", {}).get("drop_counts")
                if isinstance(lattice_payload.get("summary"), Mapping)
                else []
            ),
            "geometry_coverage": (
                lattice_payload.get("coverage", {}).get("geometry_coverage")
                if isinstance(lattice_payload.get("coverage"), Mapping)
                else None
            ),
            **FALSE_AUTHORITY,
        }

    local_plan_queueable = not context_blockers
    plan_command: list[str] = []
    if local_plan_queueable and selector_path is not None and lattice_path is not None:
        plan_command = _targeted_drop_many_stage_command(
            selector_pareto=selector_path,
            pair_frame_geometry_lattice=lattice_path,
            output_path=plan_output_path,
            drop_many_counts=drop_many_counts,
            max_drop_many=max_drop_many,
            repo_root=repo,
        )

    payload = {
        "schema": TARGETED_DROP_MANY_STAGE_INPUTS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_operation_id": operation_chain_stage_plan.get("source_operation_id"),
        "stage_id": stage_id,
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "stage_targets": stage_targets,
        "selected_family_targets": selected_family_targets,
        "target_present": target_present,
        "required_before_execution": (
            _string_list(stage_row.get("required_before_execution"))
            if stage_row
            else []
        ),
        "selector_pareto_path": _repo_rel(selector_path, repo) if selector_path else "",
        "pair_frame_geometry_lattice_path": (
            _repo_rel(lattice_path, repo) if lattice_path else ""
        ),
        "selector_pareto_summary": selector_summary,
        "pair_frame_geometry_lattice_summary": lattice_summary,
        "drop_many_counts": drop_many_counts,
        "max_drop_many": max_drop_many,
        "local_plan_queueable": local_plan_queueable,
        "local_plan_command": plan_command,
        "pairset_acquisition_path": _repo_rel(plan_output_path, repo),
        "rate_budget_policy": {
            "schema": "frontier_rate_attack_targeted_drop_many_budget_policy.v1",
            "freed_bytes_destination": "targeted_component_correction_acquisition",
            "correction_budget_role": (
                "use receiver-closed byte savings to explore multi-pair/"
                "geometry-safe correction candidates before exact spend"
            ),
            "spend_requires": [
                "receiver_contract_satisfied",
                "paired_candidate_reference_component_response",
                "total_lagrangian_improvement",
                "exact_axis_auth_eval_before_score_claim",
            ],
            "budget_spend_allowed": False,
            "allowed_use": "drop_many_stage_rate_budget_planning_only",
            "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
            **FALSE_AUTHORITY,
        },
        "targeted_correction_budget": dict(
            operation_chain_stage_plan.get("targeted_correction_budget")
            if isinstance(
                operation_chain_stage_plan.get("targeted_correction_budget"),
                Mapping,
            )
            else {}
        ),
        "blockers": _unique_strings(
            [
                *context_blockers,
                "targeted_drop_many_requires_component_replay_before_budget_spend",
                "targeted_drop_many_requires_exact_readiness_before_dispatch",
            ]
        ),
        "exact_execution_ready": False,
        "budget_spend_allowed": False,
        "allowed_use": "queue_owned_targeted_drop_many_stage_input_binding_only",
        "forbidden_use": (
            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        ),
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context=f"targeted_drop_many_stage_inputs:{stage_id}",
    )
    return payload


def build_frontier_operation_chain_compiler_queue(
    *,
    repo_root: str | Path,
    operation_chain_compiler_work_orders: Mapping[str, Any],
    operation_chain_compiler_work_orders_path: str | Path,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    queue_id: str = "frontier_operation_chain_compiler_queue",
    candidate_limit: int = 4,
    dqs1_observation_source_paths: Sequence[str | Path] = (),
) -> dict[str, Any] | None:
    """Compile multisurface chain work orders into local staged-plan queue rows."""

    repo = Path(repo_root)
    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    require_no_truthy_authority_fields(
        operation_chain_compiler_work_orders,
        context="operation_chain_compiler_queue_input",
    )
    target_profile_metadata = _target_profile_metadata_from_payloads(
        operation_chain_compiler_work_orders,
        context="operation_chain_compiler_queue",
    )
    work_order_rows = [
        row
        for row in operation_chain_compiler_work_orders.get("work_orders") or []
        if isinstance(row, Mapping)
        and row.get("schema") == OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA
    ]
    if not work_order_rows:
        return None
    work_order_rows = work_order_rows[:candidate_limit]
    work_orders_path = _resolve_path(
        operation_chain_compiler_work_orders_path,
        repo_root=repo,
    )
    results_base = _resolve_path(str(results_root), repo_root=repo)
    queue_root = results_base / "frontier_operation_chain_compiler" / _slug_token(queue_id)
    experiments: list[dict[str, Any]] = []
    for priority, row in enumerate(work_order_rows, start=1):
        source_operation_id = str(row.get("source_operation_id") or f"chain_{priority}")
        work_dir = queue_root / _slug_token(source_operation_id)
        stage_plan_path = work_dir / "stage_plan.json"
        byte_range_stage_inputs_path = work_dir / "byte_range_stage_inputs.json"
        targeted_drop_many_stage_inputs_path = (
            work_dir / "targeted_drop_many_stage_inputs.json"
        )
        targeted_drop_many_output_dir = work_dir / "targeted_drop_many_pairset"
        targeted_drop_many_pairset_path = (
            targeted_drop_many_output_dir / "targeted_drop_many_pairset_acquisition.json"
        )
        targeted_drop_many_dqs1_queue_path = (
            targeted_drop_many_output_dir / "targeted_drop_many_dqs1_followup_queue.json"
        )
        targeted_drop_many_dqs1_selected_pairset_path = (
            targeted_drop_many_output_dir / "dqs1_selected_pairset_acquisition.json"
        )
        targeted_drop_many_dqs1_feedback_bridge_path = (
            targeted_drop_many_output_dir / "dqs1_materializer_feedback_bridge.json"
        )
        targeted_drop_many_dqs1_results_root = (
            targeted_drop_many_output_dir / "dqs1_local_first_results"
        )
        byte_range_chain_output_dir = work_dir / "byte_range_entropy_recode_chain"
        byte_range_handoff_dir = byte_range_chain_output_dir / "exact_eval_handoff"
        byte_range_harvest_source_queue_path = byte_range_handoff_dir / "source_queue.json"
        byte_range_harvest_report_path = byte_range_handoff_dir / "harvest_report.json"
        byte_range_submission_closure_dir = byte_range_handoff_dir / "submission_closure"
        byte_range_submission_dir = byte_range_submission_closure_dir / "submission"
        byte_range_closed_source_queue_path = (
            byte_range_submission_closure_dir / "closed_source_queue.json"
        )
        byte_range_closure_report_path = (
            byte_range_submission_closure_dir / "submission_closure_report.json"
        )
        byte_range_readiness_dir = byte_range_handoff_dir / "exact_readiness"
        byte_range_bridge_report_path = (
            byte_range_handoff_dir / "exact_readiness_bridge_report.json"
        )
        target_kinds = _unique_strings(row.get("chain_targets") or [])
        stage_plan_preview = build_frontier_operation_chain_compiler_stage_plan(
            operation_chain_compiler_work_orders={
                "schema": OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA,
                "work_orders": [row],
                **FALSE_AUTHORITY,
            },
            source_operation_id=source_operation_id,
        )
        byte_range_inputs_preview = build_frontier_byte_range_stage_inputs(
            repo_root=repo,
            operation_chain_stage_plan=stage_plan_preview,
            chain_output_dir=byte_range_chain_output_dir,
        )
        targeted_drop_many_inputs_preview = (
            build_frontier_targeted_drop_many_stage_inputs(
                repo_root=repo,
                operation_chain_stage_plan=stage_plan_preview,
                output_dir=targeted_drop_many_output_dir,
            )
        )
        steps: list[dict[str, Any]] = [
            {
                "id": "emit_operation_chain_stage_plan",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "tools/build_frontier_operation_chain_stage_plan.py",
                    "--operation-chain-compiler-work-orders",
                    _repo_rel(work_orders_path, repo),
                    "--source-operation-id",
                    source_operation_id,
                    "--stage-plan-out",
                    _repo_rel(stage_plan_path, repo),
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": _repo_rel(stage_plan_path, repo),
                        "key": "schema",
                        "equals": OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": _repo_rel(stage_plan_path, repo),
                    },
                    {
                        "type": "json_equals",
                        "path": _repo_rel(stage_plan_path, repo),
                        "key": "execution_ready",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [_repo_rel(stage_plan_path, repo)],
                    "input_artifact_paths": [_repo_rel(work_orders_path, repo)],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "emit_byte_range_stage_inputs",
                "kind": "command",
                "requires": ["emit_operation_chain_stage_plan"],
                "command": [
                    ".venv/bin/python",
                    BYTE_RANGE_STAGE_INPUTS_TOOL,
                    "--operation-chain-stage-plan",
                    _repo_rel(stage_plan_path, repo),
                    "--stage-inputs-out",
                    _repo_rel(byte_range_stage_inputs_path, repo),
                    "--chain-output-dir",
                    _repo_rel(byte_range_chain_output_dir, repo),
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": _repo_rel(byte_range_stage_inputs_path, repo),
                        "key": "schema",
                        "equals": BYTE_RANGE_STAGE_INPUTS_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": _repo_rel(byte_range_stage_inputs_path, repo),
                    },
                    {
                        "type": "json_equals",
                        "path": _repo_rel(byte_range_stage_inputs_path, repo),
                        "key": "exact_execution_ready",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [_repo_rel(byte_range_stage_inputs_path, repo)],
                    "input_artifact_paths": [_repo_rel(stage_plan_path, repo)],
                    "include_postcondition_paths": True,
                },
            },
        ]
        if targeted_drop_many_inputs_preview.get("target_present") is True:
            steps.append(
                {
                    "id": "emit_targeted_drop_many_stage_inputs",
                    "kind": "command",
                    "requires": ["emit_operation_chain_stage_plan"],
                    "command": [
                        ".venv/bin/python",
                        TARGETED_DROP_MANY_STAGE_INPUTS_TOOL,
                        "--operation-chain-stage-plan",
                        _repo_rel(stage_plan_path, repo),
                        "--stage-inputs-out",
                        _repo_rel(targeted_drop_many_stage_inputs_path, repo),
                        "--output-dir",
                        _repo_rel(targeted_drop_many_output_dir, repo),
                        "--overwrite",
                    ],
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": _repo_rel(
                                targeted_drop_many_stage_inputs_path,
                                repo,
                            ),
                            "key": "schema",
                            "equals": TARGETED_DROP_MANY_STAGE_INPUTS_SCHEMA,
                        },
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(
                                targeted_drop_many_stage_inputs_path,
                                repo,
                            ),
                        },
                        {
                            "type": "json_equals",
                            "path": _repo_rel(
                                targeted_drop_many_stage_inputs_path,
                                repo,
                            ),
                            "key": "exact_execution_ready",
                            "equals": False,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [
                            _repo_rel(targeted_drop_many_stage_inputs_path, repo)
                        ],
                        "input_artifact_paths": [_repo_rel(stage_plan_path, repo)],
                        "include_postcondition_paths": True,
                    },
                }
            )
        if targeted_drop_many_inputs_preview.get("local_plan_queueable") is True:
            steps.append(
                {
                    "id": "run_targeted_drop_many_pairset_acquisition",
                    "kind": "command",
                    "command": list(
                        targeted_drop_many_inputs_preview["local_plan_command"]
                    ),
                    "requires": ["emit_targeted_drop_many_stage_inputs"],
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 180,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": _repo_rel(targeted_drop_many_pairset_path, repo),
                            "key": "schema",
                            "equals": "decoder_q_pairset_acquisition.v1",
                        },
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(targeted_drop_many_pairset_path, repo),
                        },
                        {
                            "type": "json_equals",
                            "path": _repo_rel(targeted_drop_many_pairset_path, repo),
                            "key": "ready_for_exact_eval_dispatch",
                            "equals": False,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [
                            _repo_rel(targeted_drop_many_pairset_path, repo)
                        ],
                        "input_artifact_paths": [
                            _repo_rel(targeted_drop_many_stage_inputs_path, repo),
                        ],
                        "include_postcondition_paths": True,
                    },
                }
            )
            steps.extend(
                [
                    {
                        "id": "build_targeted_drop_many_dqs1_followup_queue",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            DQS1_LOCAL_FIRST_QUEUE_TOOL,
                            "--pairset-acquisition",
                            _repo_rel(targeted_drop_many_pairset_path, repo),
                            "--selector-kind",
                            "drop_many_beam_pairwise_interaction_waterfill",
                            "--selector-kind",
                            "pair_frame_geometry_low_impact_drop_many",
                            "--output",
                            _repo_rel(targeted_drop_many_dqs1_queue_path, repo),
                            "--queue-id",
                            (
                                f"{_slug_token(source_operation_id)}_"
                                "targeted_drop_many_dqs1_followup"
                            ),
                            "--eureka-run-id",
                            (
                                f"{_slug_token(source_operation_id)}_"
                                "targeted_drop_many_dqs1_followup"
                            ),
                            "--results-root",
                            _repo_rel(targeted_drop_many_dqs1_results_root, repo),
                            "--candidate-limit",
                            str(candidate_limit),
                            "--selected-pairset-acquisition-out",
                            _repo_rel(
                                targeted_drop_many_dqs1_selected_pairset_path,
                                repo,
                            ),
                            "--materializer-feedback-bridge-out",
                            _repo_rel(
                                targeted_drop_many_dqs1_feedback_bridge_path,
                                repo,
                            ),
                            *[
                                item
                                for observation_path in dqs1_observation_source_paths
                                for item in (
                                    "--dqs1-observation-jsonl",
                                    _repo_rel(
                                        _resolve_path(observation_path, repo_root=repo),
                                        repo,
                                    ),
                                )
                            ],
                            "--write",
                        ],
                        "requires": ["run_targeted_drop_many_pairset_acquisition"],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 180,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(
                                    targeted_drop_many_dqs1_queue_path,
                                    repo,
                                ),
                                "key": "schema",
                                "equals": QUEUE_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(
                                    targeted_drop_many_dqs1_queue_path,
                                    repo,
                                ),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(
                                    targeted_drop_many_dqs1_selected_pairset_path,
                                    repo,
                                ),
                                "key": "schema",
                                "equals": "dqs1_selected_pairset_acquisition.v1",
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(
                                    targeted_drop_many_dqs1_selected_pairset_path,
                                    repo,
                                ),
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(
                                    targeted_drop_many_dqs1_feedback_bridge_path,
                                    repo,
                                ),
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(targeted_drop_many_dqs1_queue_path, repo),
                                _repo_rel(
                                    targeted_drop_many_dqs1_selected_pairset_path,
                                    repo,
                                ),
                                _repo_rel(
                                    targeted_drop_many_dqs1_feedback_bridge_path,
                                    repo,
                                ),
                            ],
                            "input_artifact_paths": [
                                _repo_rel(targeted_drop_many_pairset_path, repo),
                            ],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "validate_targeted_drop_many_dqs1_followup_queue",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/experiment_queue.py",
                            "--queue",
                            _repo_rel(targeted_drop_many_dqs1_queue_path, repo),
                            "validate",
                        ],
                        "requires": [
                            "build_targeted_drop_many_dqs1_followup_queue"
                        ],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 120,
                        "telemetry": {
                            "input_artifact_paths": [
                                _repo_rel(targeted_drop_many_dqs1_queue_path, repo)
                            ],
                        },
                    },
                ]
            )
        if byte_range_inputs_preview.get("local_chain_queueable") is True:
            chain_manifest_path = str(byte_range_inputs_preview["chain_manifest_path"])
            steps.extend(
                [
                    {
                        "id": "run_byte_range_entropy_recode_chain",
                        "kind": "command",
                        "command": list(byte_range_inputs_preview["local_chain_command"]),
                        "requires": ["emit_byte_range_stage_inputs"],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 600,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": chain_manifest_path,
                                "key": "schema",
                                "equals": "byte_range_entropy_recode_chain_v1",
                            },
                            {
                                "type": "json_false_authority",
                                "path": chain_manifest_path,
                            },
                            {
                                "type": "json_equals",
                                "path": chain_manifest_path,
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [chain_manifest_path],
                            "input_artifact_paths": [
                                _repo_rel(byte_range_stage_inputs_path, repo),
                            ],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "harvest_byte_range_entropy_recode_chain",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            MATERIALIZER_CHAIN_HARVEST_TOOL,
                            "--repo-root",
                            repo.as_posix(),
                            "--chain-manifest",
                            chain_manifest_path,
                            "--source-queue-out",
                            _repo_rel(byte_range_harvest_source_queue_path, repo),
                            "--report-out",
                            _repo_rel(byte_range_harvest_report_path, repo),
                            "--require-accepted",
                            "--overwrite",
                        ],
                        "requires": ["run_byte_range_entropy_recode_chain"],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 180,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(
                                    byte_range_harvest_source_queue_path,
                                    repo,
                                ),
                                "key": "schema",
                                "equals": "optimizer_candidate_queue_v1",
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(
                                    byte_range_harvest_source_queue_path,
                                    repo,
                                ),
                                "false_or_missing": list(
                                    QUEUE_FALSE_AUTHORITY_FALSE_OR_MISSING_FIELDS
                                ),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(byte_range_harvest_report_path, repo),
                                "key": "schema",
                                "equals": MATERIALIZER_CHAIN_HARVEST_REPORT_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(byte_range_harvest_report_path, repo),
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(byte_range_harvest_source_queue_path, repo),
                                _repo_rel(byte_range_harvest_report_path, repo),
                            ],
                            "input_artifact_paths": [chain_manifest_path],
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "build_byte_range_submission_closure",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            MATERIALIZER_SUBMISSION_CLOSURE_TOOL,
                            "--source-queue",
                            _repo_rel(byte_range_harvest_source_queue_path, repo),
                            "--submission-dir-out",
                            _repo_rel(byte_range_submission_dir, repo),
                            "--closed-source-queue-out",
                            _repo_rel(byte_range_closed_source_queue_path, repo),
                            "--closure-report-out",
                            _repo_rel(byte_range_closure_report_path, repo),
                            "--overwrite",
                        ],
                        "requires": ["harvest_byte_range_entropy_recode_chain"],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 180,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(
                                    byte_range_closed_source_queue_path,
                                    repo,
                                ),
                                "key": "schema",
                                "equals": "optimizer_candidate_queue_v1",
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(
                                    byte_range_closed_source_queue_path,
                                    repo,
                                ),
                                "false_or_missing": list(
                                    QUEUE_FALSE_AUTHORITY_FALSE_OR_MISSING_FIELDS
                                ),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(byte_range_closure_report_path, repo),
                                "key": "schema",
                                "equals": MATERIALIZER_SUBMISSION_CLOSURE_REPORT_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(byte_range_closure_report_path, repo),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(byte_range_closure_report_path, repo),
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(byte_range_submission_dir, repo),
                                _repo_rel(byte_range_closed_source_queue_path, repo),
                                _repo_rel(byte_range_closure_report_path, repo),
                            ],
                            "input_artifact_paths": [
                                _repo_rel(byte_range_harvest_source_queue_path, repo),
                                chain_manifest_path,
                            ],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "run_byte_range_exact_readiness_bridge",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            MATERIALIZER_EXACT_READINESS_BRIDGE_TOOL,
                            "--source-queue",
                            _repo_rel(byte_range_closed_source_queue_path, repo),
                            "--exact-readiness-out-dir",
                            _repo_rel(byte_range_readiness_dir, repo),
                            "--bridge-report-out",
                            _repo_rel(byte_range_bridge_report_path, repo),
                            "--overwrite",
                            "--force-recompute",
                        ],
                        "requires": ["build_byte_range_submission_closure"],
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 180,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(byte_range_bridge_report_path, repo),
                                "key": "schema",
                                "equals": MATERIALIZER_EXACT_READINESS_BRIDGE_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(byte_range_bridge_report_path, repo),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(byte_range_bridge_report_path, repo),
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(byte_range_readiness_dir, repo),
                                _repo_rel(byte_range_bridge_report_path, repo),
                            ],
                            "input_artifact_paths": [
                                _repo_rel(byte_range_closed_source_queue_path, repo),
                                _repo_rel(byte_range_closure_report_path, repo),
                            ],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    },
                ]
            )
        experiments.append(
            {
                "id": _slug_token(source_operation_id),
                "status": "queued",
                "priority": priority,
                "lane_id": "lane_frontier_operation_chain_compiler_20260526",
                "tags": [
                    "operation_chain_compiler",
                    "multisurface_materializer_chain",
                    "receiver_runtime",
                    "targeted_correction_budget",
                ],
                "metadata": {
                    "schema": OPERATION_CHAIN_COMPILER_QUEUE_METADATA_SCHEMA,
                    "source_operation_id": source_operation_id,
                    "source_operation_family": row.get("source_operation_family"),
                    "frontier_target_optimization_profile": dict(
                        _target_profile_metadata_from_payloads(
                            row,
                            target_profile_metadata,
                            context="operation_chain_compiler_queue_experiment",
                        )
                    ),
                    "chain_target_count": len(target_kinds),
                    "chain_targets": target_kinds,
                    "stage_plan_path": _repo_rel(stage_plan_path, repo),
                    "byte_range_stage_inputs_path": _repo_rel(
                        byte_range_stage_inputs_path,
                        repo,
                    ),
                    "byte_range_chain_manifest_path": (
                        byte_range_inputs_preview.get("chain_manifest_path")
                    ),
                    "byte_range_local_chain_queueable": (
                        byte_range_inputs_preview.get("local_chain_queueable") is True
                    ),
                    "byte_range_harvest_source_queue_path": _repo_rel(
                        byte_range_harvest_source_queue_path,
                        repo,
                    ),
                    "byte_range_harvest_report_path": _repo_rel(
                        byte_range_harvest_report_path,
                        repo,
                    ),
                    "byte_range_submission_closure_report_path": _repo_rel(
                        byte_range_closure_report_path,
                        repo,
                    ),
                    "byte_range_closed_source_queue_path": _repo_rel(
                        byte_range_closed_source_queue_path,
                        repo,
                    ),
                    "byte_range_exact_readiness_bridge_report_path": _repo_rel(
                        byte_range_bridge_report_path,
                        repo,
                    ),
                    "byte_range_exact_readiness_handoff_enabled": (
                        byte_range_inputs_preview.get("local_chain_queueable") is True
                    ),
                    "byte_range_rate_budget_policy": dict(
                        byte_range_inputs_preview.get("rate_budget_policy")
                        if isinstance(
                            byte_range_inputs_preview.get("rate_budget_policy"),
                            Mapping,
                        )
                        else {}
                    ),
                    "targeted_drop_many_stage_inputs_path": _repo_rel(
                        targeted_drop_many_stage_inputs_path,
                        repo,
                    ),
                    "targeted_drop_many_pairset_acquisition_path": _repo_rel(
                        targeted_drop_many_pairset_path,
                        repo,
                    ),
                    "targeted_drop_many_dqs1_followup_queue_path": _repo_rel(
                        targeted_drop_many_dqs1_queue_path,
                        repo,
                    ),
                    "targeted_drop_many_dqs1_selected_pairset_acquisition_path": (
                        _repo_rel(targeted_drop_many_dqs1_selected_pairset_path, repo)
                    ),
                    "targeted_drop_many_dqs1_materializer_feedback_bridge_path": (
                        _repo_rel(targeted_drop_many_dqs1_feedback_bridge_path, repo)
                    ),
                    "targeted_drop_many_dqs1_followup_queue_enabled": (
                        targeted_drop_many_inputs_preview.get("local_plan_queueable")
                        is True
                    ),
                    "targeted_drop_many_dqs1_results_root": _repo_rel(
                        targeted_drop_many_dqs1_results_root,
                        repo,
                    ),
                    "targeted_drop_many_dqs1_selector_kind_allowlist": [
                        "drop_many_beam_pairwise_interaction_waterfill",
                        "pair_frame_geometry_low_impact_drop_many",
                    ],
                    "targeted_drop_many_dqs1_observation_source_paths": [
                        _repo_rel(_resolve_path(path, repo_root=repo), repo)
                        for path in dqs1_observation_source_paths
                    ],
                    "targeted_drop_many_dqs1_candidate_limit": candidate_limit,
                    "targeted_drop_many_local_plan_queueable": (
                        targeted_drop_many_inputs_preview.get("local_plan_queueable")
                        is True
                    ),
                    "targeted_drop_many_selected_family_targets": list(
                        targeted_drop_many_inputs_preview.get(
                            "selected_family_targets"
                        )
                        or []
                    ),
                    "targeted_drop_many_rate_budget_policy": dict(
                        targeted_drop_many_inputs_preview.get("rate_budget_policy")
                        if isinstance(
                            targeted_drop_many_inputs_preview.get(
                                "rate_budget_policy"
                            ),
                            Mapping,
                        )
                        else {}
                    ),
                    "targeted_correction_budget": dict(
                        row.get("targeted_correction_budget")
                        if isinstance(row.get("targeted_correction_budget"), Mapping)
                        else {}
                    ),
                    "execution_ready": False,
                    "budget_spend_allowed": False,
                    "allowed_use": "operation_chain_compiler_queue_metadata_only",
                    "forbidden_use": "score_claim_or_dispatch_authority",
                    **FALSE_AUTHORITY,
                },
                "steps": steps,
            }
        )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": 1,
                    "local_io_heavy": 1,
                    "local_mlx": 0,
                    "modal_cpu": 0,
                    "modal_gpu": 0,
                },
            },
            "metadata": {
                "schema": OPERATION_CHAIN_COMPILER_QUEUE_METADATA_SCHEMA,
                "frontier_target_optimization_profile": dict(
                    target_profile_metadata
                ),
                "work_order_count": len(work_order_rows),
                "candidate_limit": candidate_limit,
                "allowed_use": "operation_chain_compiler_queue_metadata_only",
                "forbidden_use": "score_claim_or_dispatch_authority",
                **FALSE_AUTHORITY,
            },
            "experiments": experiments,
            "allowed_use": "queue_owned_operation_chain_compiler_stage_plans_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
            **FALSE_AUTHORITY,
        }
    )


def _selected_targeted_component_correction_rows(
    acquisition: Mapping[str, Any],
    *,
    candidate_limit: int,
) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in acquisition.get("rows") or []
        if isinstance(row, Mapping) and row.get("queue_actionable") is True
    ]
    ranked_rows = sorted(
        rows,
        key=lambda item: (
            -float(item.get("priority_score") or 0.0),
            str(item.get("acquisition_id") or ""),
        ),
    )
    rows_by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for row in ranked_rows:
        candidate_key = str(row.get("candidate_id") or row.get("submission_dir") or "")
        rows_by_candidate.setdefault(candidate_key, []).append(row)

    candidate_order = sorted(
        rows_by_candidate,
        key=lambda candidate: (
            -float(rows_by_candidate[candidate][0].get("priority_score") or 0.0),
            candidate,
        ),
    )
    candidate_offsets = dict.fromkeys(candidate_order, 0)
    selected: list[Mapping[str, Any]] = []
    seen_candidate_families: set[tuple[str, str]] = set()
    while len(selected) < candidate_limit:
        progressed = False
        for candidate in candidate_order:
            candidate_rows = rows_by_candidate[candidate]
            offset = candidate_offsets[candidate]
            while offset < len(candidate_rows):
                row = candidate_rows[offset]
                offset += 1
                family = str(row.get("correction_family") or "")
                key = (candidate, family)
                if key in seen_candidate_families:
                    continue
                seen_candidate_families.add(key)
                selected.append(row)
                progressed = True
                break
            candidate_offsets[candidate] = offset
            if len(selected) >= candidate_limit:
                break
        if not progressed:
            break
    return selected


def _targeted_component_correction_queue_selection_policy(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    candidate_limit: int,
) -> dict[str, Any]:
    selected_candidates = _unique_strings(
        row.get("candidate_id") or row.get("submission_dir") for row in selected_rows
    )
    selected_families = _unique_strings(
        row.get("correction_family") for row in selected_rows
    )
    return {
        "schema": "frontier_rate_attack_targeted_component_queue_selection_policy.v1",
        "policy": "bounded_candidate_family_round_robin",
        "candidate_limit": candidate_limit,
        "selected_row_count": len(selected_rows),
        "selected_candidate_count": len(selected_candidates),
        "selected_correction_family_count": len(selected_families),
        "selected_candidate_ids": selected_candidates,
        "selected_correction_families": selected_families,
        "rationale": (
            "fan out receiver-closed rate credit across multiple correction "
            "operators instead of collapsing each candidate to one leaf probe"
        ),
        "budget_spend_allowed": False,
        "allowed_use": "local_component_correction_acquisition_selection_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _component_response_cache_source(
    *,
    repo_root: Path,
    cache_roots: Sequence[str | Path],
    cache_key: str,
    role: str,
) -> dict[str, Any] | None:
    if not cache_key:
        return None
    if role == "reference":
        advisory_name = "reference_local_cpu_advisory.json"
        hashes_name = "reference_scorer_input_cache_hashes.json"
        mlx_cache_name = "reference_mlx_scorer_input_cache"
        mlx_audit_name = "reference_mlx_scorer_input_cache_audit.json"
        shared_name = "shared_reference_component_response"
    else:
        advisory_name = "local_cpu_advisory.json"
        hashes_name = "scorer_input_cache_hashes.json"
        mlx_cache_name = "mlx_scorer_input_cache"
        mlx_audit_name = "mlx_scorer_input_cache_audit.json"
        shared_name = "shared_candidate_component_response"
    candidate_dirs: list[Path] = []
    for root in cache_roots:
        root_path = _resolve_path(root, repo_root=repo_root)
        candidate_dirs.extend(
            [
                root_path,
                root_path / cache_key,
                root_path / shared_name / cache_key,
                root_path
                / "frontier_targeted_component_correction"
                / shared_name
                / cache_key,
            ]
        )
    seen: set[Path] = set()
    for source_dir in candidate_dirs:
        resolved = source_dir.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        advisory = resolved / advisory_name
        hashes = resolved / hashes_name
        if advisory.is_file() and hashes.is_file():
            source: dict[str, str | bool] = {
                "role": role,
                "source_dir": _repo_rel(resolved, repo_root),
                "source_local_cpu_advisory": _repo_rel(advisory, repo_root),
                "source_scorer_input_cache_hashes": _repo_rel(hashes, repo_root),
                "reuse_mode": "import_false_authority_component_response_cache",
                **FALSE_AUTHORITY,
            }
            mlx_cache_dir = resolved / mlx_cache_name
            mlx_cache_audit = resolved / mlx_audit_name
            if (mlx_cache_dir / "manifest.json").is_file() and mlx_cache_audit.is_file():
                source["source_mlx_scorer_input_cache"] = _repo_rel(
                    mlx_cache_dir,
                    repo_root,
                )
                source["source_mlx_scorer_input_cache_audit"] = _repo_rel(
                    mlx_cache_audit,
                    repo_root,
                )
                source["mlx_cache_reuse_mode"] = (
                    "reuse_false_authority_mlx_scorer_input_cache"
                )
            return source
    return None


def _targeted_component_reference_eval_context_for_queue(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Recover receiver-closed source-reference context for component deltas."""

    raw_context = row.get("reference_component_eval_context")
    context: dict[str, Any] = (
        dict(raw_context) if isinstance(raw_context, Mapping) else {}
    )
    require_no_truthy_authority_fields(
        context,
        context="targeted_component_reference_eval_context_for_queue",
    )

    closure_report_text = _path_text_from_value(row.get("closure_report_path"))
    if not closure_report_text:
        return context
    missing_reference_context = not (
        _path_text_from_value(row.get("source_archive_path"))
        or _path_text_from_value(context.get("source_archive_path"))
    ) or not (
        _path_text_from_value(row.get("source_inflate_sh_path"))
        or _path_text_from_value(context.get("source_inflate_sh_path"))
        or _path_text_from_value(row.get("source_submission_dir"))
        or _path_text_from_value(context.get("source_submission_dir"))
    )
    if not missing_reference_context:
        return context

    closure_report_path = _resolve_path(closure_report_text, repo_root=repo_root)
    if not closure_report_path.is_file():
        return context
    try:
        closure = _load_json(closure_report_path)
        require_no_truthy_authority_fields(
            closure,
            context=f"{closure_report_path} targeted component reference recovery",
        )
    except (FrontierRateAttackFeedbackError, ValueError):
        return context
    if closure.get("schema") != MATERIALIZER_SUBMISSION_CLOSURE_REPORT_SCHEMA:
        return context
    recovered = _submission_closure_reference_eval_context(
        closure,
        closure_report_path,
        repo_root=repo_root,
        candidate_id=str(row.get("candidate_id") or closure.get("candidate_id") or ""),
    )
    require_no_truthy_authority_fields(
        recovered,
        context="targeted_component_recovered_reference_eval_context",
    )
    merged = {
        **recovered,
        **{key: value for key, value in context.items() if value not in (None, "")},
        "reference_context_recovery_mode": (
            "receiver_closure_source_reference_context"
        ),
        "reference_context_recovery_source": _repo_rel(closure_report_path, repo_root),
    }
    return merged


def build_frontier_targeted_component_correction_queue(
    *,
    repo_root: str | Path,
    targeted_component_correction_acquisition: Mapping[str, Any],
    targeted_component_correction_acquisition_path: str | Path,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    queue_id: str = "frontier_targeted_component_correction_queue",
    candidate_limit: int = 4,
    local_cpu_concurrency: int = 1,
    local_mlx_concurrency: int = 1,
    upstream_dir: str | Path = "upstream",
    video_names_file: str | Path = "upstream/public_test_video_names.txt",
    mlx_reference_cache_dir: str | Path = DEFAULT_MLX_REFERENCE_CACHE_DIR,
    mlx_device: str = "gpu",
    include_mlx_response: bool = True,
    component_response_cache_roots: Sequence[str | Path] = (),
    target_optimization_profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compile receiver-closed budget rows into local CPU/MLX component probes."""

    repo = Path(repo_root)
    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    if local_cpu_concurrency < 1:
        raise FrontierRateAttackFeedbackError("local_cpu_concurrency must be >= 1")
    if local_mlx_concurrency < 0:
        raise FrontierRateAttackFeedbackError("local_mlx_concurrency must be >= 0")
    if mlx_device not in {"cpu", "gpu"}:
        raise FrontierRateAttackFeedbackError("mlx_device must be 'cpu' or 'gpu'")
    require_no_truthy_authority_fields(
        targeted_component_correction_acquisition,
        context="targeted_component_correction_queue_input",
    )
    target_profile_metadata = (
        dict(target_optimization_profile_metadata)
        if isinstance(target_optimization_profile_metadata, Mapping)
        and target_optimization_profile_metadata
        else {}
    )
    if target_profile_metadata:
        require_no_truthy_authority_fields(
            target_profile_metadata,
            context="targeted_component_correction_queue_target_optimization_profile",
        )
    selected_rows = _selected_targeted_component_correction_rows(
        targeted_component_correction_acquisition,
        candidate_limit=candidate_limit,
    )
    if not selected_rows:
        return _blocked_targeted_component_correction_queue(
            repo_root=repo,
            targeted_component_correction_acquisition=(
                targeted_component_correction_acquisition
            ),
            targeted_component_correction_acquisition_path=(
                targeted_component_correction_acquisition_path
            ),
            results_root=results_root,
            queue_id=queue_id,
            candidate_limit=candidate_limit,
            target_optimization_profile_metadata=target_profile_metadata,
        )
    selection_policy = _targeted_component_correction_queue_selection_policy(
        selected_rows=selected_rows,
        candidate_limit=candidate_limit,
    )
    repair_operator_portfolio = (
        targeted_component_correction_acquisition.get("repair_operator_portfolio")
        if isinstance(
            targeted_component_correction_acquisition.get("repair_operator_portfolio"),
            Mapping,
        )
        else {}
    )
    acquisition_path = _resolve_path(
        targeted_component_correction_acquisition_path,
        repo_root=repo,
    )
    results_base = _resolve_path(str(results_root), repo_root=repo)
    queue_root = results_base / "frontier_targeted_component_correction" / _slug_token(
        queue_id
    )
    rows_by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected_rows:
        candidate_id = str(row.get("candidate_id") or row.get("acquisition_id") or "")
        rows_by_candidate.setdefault(candidate_id, []).append(row)
    candidate_order = _unique_strings(
        row.get("candidate_id") or row.get("acquisition_id") for row in selected_rows
    )
    experiments: list[dict[str, Any]] = []
    for priority, candidate_id in enumerate(candidate_order, start=1):
        candidate_rows = rows_by_candidate[candidate_id]
        primary_row = candidate_rows[0]
        candidate_dir = queue_root / _slug_token(candidate_id)
        archive_path = str(
            primary_row.get("archive_path")
            or primary_row.get("candidate_archive_path")
            or ""
        )
        inflate_sh_path = str(
            primary_row.get("inflate_sh_path")
            or primary_row.get("candidate_inflate_sh_path")
            or ""
        )
        if not archive_path:
            raise FrontierRateAttackFeedbackError(
                f"{candidate_id}: candidate_archive_path_missing_for_component_queue"
            )
        if not inflate_sh_path:
            raise FrontierRateAttackFeedbackError(
                f"{candidate_id}: candidate_inflate_sh_path_missing_for_component_queue"
            )
        candidate_cache_key = _bounded_content_key(
            "candidate_component_response",
            (
                primary_row.get("archive_sha256"),
                archive_path,
                inflate_sh_path,
            ),
        )
        shared_dir = (
            results_base
            / "frontier_targeted_component_correction"
            / "shared_candidate_component_response"
            / candidate_cache_key
        )
        local_cpu_advisory = shared_dir / "local_cpu_advisory.json"
        local_cpu_work_dir = shared_dir / "local_cpu_advisory_work"
        scorer_hashes = shared_dir / "scorer_input_cache_hashes.json"
        component_cache_source = _component_response_cache_source(
            repo_root=repo,
            cache_roots=component_response_cache_roots,
            cache_key=candidate_cache_key,
            role="candidate",
        )
        component_advisory_step_id = (
            "import_local_cpu_component_advisory_cache"
            if component_cache_source is not None
            else "local_cpu_component_advisory"
        )
        reference_context = _targeted_component_reference_eval_context_for_queue(
            primary_row,
            repo_root=repo,
        )
        reference_archive_path = str(
            primary_row.get("source_archive_path")
            or reference_context.get("source_archive_path")
            or ""
        )
        reference_source_submission_dir = str(
            primary_row.get("source_submission_dir")
            or reference_context.get("source_submission_dir")
            or ""
        )
        reference_inflate_sh_path = str(
            primary_row.get("source_inflate_sh_path")
            or reference_context.get("source_inflate_sh_path")
            or (
                f"{reference_source_submission_dir}/inflate.sh"
                if reference_source_submission_dir
                else ""
            )
        )
        reference_eval_available = bool(
            reference_archive_path and reference_inflate_sh_path
        )
        reference_cache_key = _bounded_content_key(
            "reference_component_response",
            (
                primary_row.get("source_archive_sha256")
                or reference_context.get("source_archive_sha256"),
                reference_archive_path,
                reference_inflate_sh_path,
            ),
        )
        reference_shared_dir = (
            results_base
            / "frontier_targeted_component_correction"
            / "shared_reference_component_response"
            / reference_cache_key
        )
        reference_local_cpu_advisory = (
            reference_shared_dir / "reference_local_cpu_advisory.json"
        )
        reference_local_cpu_work_dir = (
            reference_shared_dir / "reference_local_cpu_advisory_work"
        )
        reference_scorer_hashes = (
            reference_shared_dir / "reference_scorer_input_cache_hashes.json"
        )
        reference_cache_source = (
            _component_response_cache_source(
                repo_root=repo,
                cache_roots=component_response_cache_roots,
                cache_key=reference_cache_key,
                role="reference",
            )
            if reference_eval_available
            else None
        )
        reference_advisory_step_id = (
            "import_local_cpu_reference_advisory_cache"
            if reference_cache_source is not None
            else "local_cpu_reference_advisory"
        )
        local_mlx_response: Path | None = None
        reference_local_mlx_response: Path | None = None
        request_metadata: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        request_ready_step_ids: list[str] = []
        for row_index, row in enumerate(candidate_rows, start=1):
            acquisition_id = str(
                row.get("acquisition_id")
                or f"targeted_correction_{priority}_{row_index}"
            )
            request_dir = candidate_dir / _slug_token(acquisition_id)
            work_order_path = request_dir / "work_order.json"
            repair_dynamics_probe_matrix_path = (
                request_dir / "repair_dynamics_palette_probe_matrix.json"
            )
            repair_dynamics_probe_output_dir = request_dir / "repair_dynamics_palette_probe"
            response_harvest_path = (
                request_dir / "component_correction_response_harvest.json"
            )
            work_order_step_id = f"emit_targeted_component_correction_work_order_{row_index:02d}"
            repair_dynamics_probe_step_id = (
                f"emit_repair_dynamics_palette_probe_matrix_{row_index:02d}"
            )
            request_ready_step_ids.append(work_order_step_id)
            row_correction_family = str(row.get("correction_family") or "")
            repair_dynamics_prior_active = row.get("repair_dynamics_prior_active") is True
            repair_dynamics_probe_required = (
                repair_dynamics_prior_active
                and row_correction_family.startswith("repair_dynamics_")
            )
            work_order_command = [
                ".venv/bin/python",
                "tools/build_frontier_targeted_component_correction_work_order.py",
                "--targeted-component-correction-acquisition",
                _repo_rel(acquisition_path, repo),
                "--acquisition-id",
                acquisition_id,
                "--work-order-out",
                _repo_rel(work_order_path, repo),
                "--overwrite",
            ]
            if target_profile_metadata:
                work_order_command.extend(
                    [
                        "--target-optimization-profile-metadata-json",
                        json.dumps(
                            target_profile_metadata,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ]
                )
            request_metadata.append(
                {
                    "schema": (
                        "frontier_rate_attack_targeted_component_correction_"
                        "queue_request_metadata.v1"
                    ),
                    "acquisition_id": acquisition_id,
                    "candidate_id": candidate_id,
                    "target_kind": row.get("target_kind"),
                    "correction_family": row.get("correction_family"),
                    "operation_levels": list(row.get("operation_levels") or []),
                    "targeted_dimensions": list(row.get("targeted_dimensions") or []),
                    "saved_bytes_budget": row.get("saved_bytes_budget"),
                    "estimated_rate_credit_score_units": row.get(
                        "estimated_rate_credit_score_units"
                    ),
                    "work_dir": _repo_rel(request_dir, repo),
                    "work_order_path": _repo_rel(work_order_path, repo),
                    "repair_dynamics_prior_active": repair_dynamics_prior_active,
                    "repair_dynamics_probe_required": repair_dynamics_probe_required,
                    "repair_dynamics_palette_probe_matrix_path": (
                        _repo_rel(repair_dynamics_probe_matrix_path, repo)
                        if repair_dynamics_probe_required
                        else None
                    ),
                    "component_correction_response_harvest_path": _repo_rel(
                        response_harvest_path,
                        repo,
                    ),
                    "frontier_target_optimization_profile": dict(
                        target_profile_metadata
                    ),
                    "local_cpu_advisory_path": _repo_rel(local_cpu_advisory, repo),
                    "local_cpu_advisory_reuse_mode": (
                        component_cache_source.get("reuse_mode")
                        if isinstance(component_cache_source, Mapping)
                        else "queue_owned_local_cpu_component_advisory"
                    ),
                    "local_cpu_advisory_source_path": (
                        component_cache_source.get("source_local_cpu_advisory")
                        if isinstance(component_cache_source, Mapping)
                        else None
                    ),
                    "candidate_cache_key": candidate_cache_key,
                    "shared_component_response_dir": _repo_rel(shared_dir, repo),
                    "reference_local_cpu_advisory_path": (
                        _repo_rel(reference_local_cpu_advisory, repo)
                        if reference_eval_available
                        else None
                    ),
                    "reference_local_cpu_advisory_reuse_mode": (
                        reference_cache_source.get("reuse_mode")
                        if isinstance(reference_cache_source, Mapping)
                        else (
                            "queue_owned_local_cpu_component_advisory"
                            if reference_eval_available
                            else None
                        )
                    ),
                    "reference_local_cpu_advisory_source_path": (
                        reference_cache_source.get("source_local_cpu_advisory")
                        if isinstance(reference_cache_source, Mapping)
                        else None
                    ),
                    "reference_archive_path": reference_archive_path or None,
                    "reference_inflate_sh_path": reference_inflate_sh_path or None,
                    "reference_component_eval_context": dict(
                        _targeted_component_reference_eval_context_for_queue(
                            row,
                            repo_root=repo,
                        )
                    ),
                    "reference_cache_key": reference_cache_key,
                    "reference_shared_component_response_dir": (
                        _repo_rel(reference_shared_dir, repo)
                        if reference_eval_available
                        else None
                    ),
                    "local_mlx_response_path": (
                        None
                        if not include_mlx_response
                        else _repo_rel(shared_dir / "mlx_scorer_response.json", repo)
                    ),
                    "reference_local_mlx_response_path": (
                        _repo_rel(
                            reference_shared_dir
                            / "reference_mlx_scorer_response.json",
                            repo,
                        )
                        if include_mlx_response and reference_eval_available
                        else None
                    ),
                    "budget_spend_ready": False,
                    "budget_spend_gate": dict(
                        row.get("budget_spend_gate")
                        if isinstance(row.get("budget_spend_gate"), Mapping)
                        else {}
                    ),
                    "allowed_use": (
                        "targeted_component_correction_queue_request_metadata_only"
                    ),
                    "forbidden_use": "score_claim_or_dispatch_authority",
                    **FALSE_AUTHORITY,
                }
            )
            steps.append(
                {
                    "id": work_order_step_id,
                    "kind": "command",
                    "command": work_order_command,
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": _repo_rel(work_order_path, repo),
                            "key": "schema",
                            "equals": TARGETED_COMPONENT_CORRECTION_WORK_ORDER_SCHEMA,
                        },
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(work_order_path, repo),
                        },
                        {
                            "type": "json_equals",
                            "path": _repo_rel(work_order_path, repo),
                            "key": "budget_spend_gate.ready_for_budget_spend",
                            "equals": False,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [_repo_rel(work_order_path, repo)],
                        "input_artifact_paths": _unique_strings(
                            [
                                _repo_rel(acquisition_path, repo),
                                row.get("closure_report_path"),
                                row.get("paired_exact_readiness_bridge_report_path"),
                            ]
                        ),
                        "include_postcondition_paths": True,
                    },
                }
            )
            if repair_dynamics_probe_required:
                request_ready_step_ids.append(repair_dynamics_probe_step_id)
                steps.append(
                    {
                        "id": repair_dynamics_probe_step_id,
                        "kind": "command",
                        "requires": [work_order_step_id],
                        "command": [
                            ".venv/bin/python",
                            "tools/build_repair_dynamics_palette_probe_matrix.py",
                            "--work-order",
                            _repo_rel(work_order_path, repo),
                            "--matrix-out",
                            _repo_rel(repair_dynamics_probe_matrix_path, repo),
                            "--probe-output-dir",
                            _repo_rel(repair_dynamics_probe_output_dir, repo),
                            "--device",
                            "mlx",
                            "--n-pairs",
                            "48",
                            "--overwrite",
                        ],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(
                                    repair_dynamics_probe_matrix_path,
                                    repo,
                                ),
                                "key": "schema",
                                "equals": REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(
                                    repair_dynamics_probe_matrix_path,
                                    repo,
                                ),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(
                                    repair_dynamics_probe_matrix_path,
                                    repo,
                                ),
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(repair_dynamics_probe_matrix_path, repo)
                            ],
                            "input_artifact_paths": [_repo_rel(work_order_path, repo)],
                            "include_postcondition_paths": True,
                        },
                    }
                )
        steps.append(
            {
                "id": "emit_targeted_component_correction_work_order",
                "kind": "command",
                "requires": request_ready_step_ids,
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json; "
                        "print(json.dumps({'schema': "
                        "'frontier_rate_attack_targeted_component_correction_"
                        "grouped_work_order_barrier.v1', 'ok': True, "
                        "'score_claim': False, 'promotion_eligible': False, "
                        "'rank_or_kill_eligible': False, "
                        "'ready_for_exact_eval_dispatch': False}))"
                    ),
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
            },
        )
        if reference_eval_available and reference_cache_source is not None:
            steps.append(
                {
                    "id": reference_advisory_step_id,
                    "kind": "command",
                    "requires": ["emit_targeted_component_correction_work_order"],
                    "command": [
                        ".venv/bin/python",
                        "tools/import_frontier_component_response_cache.py",
                        "--source-local-cpu-advisory",
                        str(reference_cache_source["source_local_cpu_advisory"]),
                        "--source-scorer-input-cache-hashes",
                        str(reference_cache_source["source_scorer_input_cache_hashes"]),
                        "--local-cpu-advisory-out",
                        _repo_rel(reference_local_cpu_advisory, repo),
                        "--scorer-input-cache-hashes-out",
                        _repo_rel(reference_scorer_hashes, repo),
                        "--role",
                        "reference",
                        "--overwrite",
                    ],
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(reference_local_cpu_advisory, repo),
                            "axis_key": "score_axis",
                            "axis_equals": "cpu_advisory",
                        },
                        {
                            "type": "json_equals",
                            "path": _repo_rel(reference_scorer_hashes, repo),
                            "key": "schema_version",
                            "equals": "mlx_scorer_input_cache_hashes.v1",
                        },
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(reference_scorer_hashes, repo),
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [
                            _repo_rel(reference_local_cpu_advisory, repo),
                            _repo_rel(reference_scorer_hashes, repo),
                        ],
                        "input_artifact_paths": [
                            str(reference_cache_source["source_local_cpu_advisory"]),
                            str(reference_cache_source["source_scorer_input_cache_hashes"]),
                        ],
                        "include_postcondition_paths": True,
                    },
                }
            )
        elif reference_eval_available:
            steps.append(
                {
                    "id": "local_cpu_reference_advisory",
                    "kind": "command",
                    "requires": ["emit_targeted_component_correction_work_order"],
                    "command": [
                        ".venv/bin/python",
                        "experiments/contest_auth_eval.py",
                        "--archive",
                        reference_archive_path,
                        "--inflate-sh",
                        reference_inflate_sh_path,
                        "--upstream-dir",
                        str(upstream_dir),
                        "--video-names-file",
                        str(video_names_file),
                        "--device",
                        "cpu",
                        "--work-dir",
                        _repo_rel(reference_local_cpu_work_dir, repo),
                        "--json-out",
                        _repo_rel(reference_local_cpu_advisory, repo),
                        "--reuse-valid-json-out",
                        "--inflate-timeout",
                        "1800",
                        "--evaluate-timeout",
                        "1800",
                        "--keep-work-dir",
                        "--scorer-input-cache-hashes-out",
                        _repo_rel(reference_scorer_hashes, repo),
                        "--allow-scorer-input-cache-artifact-output-outside-work-dir",
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 3900,
                    "postconditions": [
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(reference_local_cpu_advisory, repo),
                            "axis_key": "score_axis",
                            "axis_equals": "cpu_advisory",
                        },
                        {
                            "type": "json_equals",
                            "path": _repo_rel(reference_scorer_hashes, repo),
                            "key": "schema_version",
                            "equals": "mlx_scorer_input_cache_hashes.v1",
                        },
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(reference_scorer_hashes, repo),
                        }
                    ],
                    "telemetry": {
                        "artifact_paths": [
                            _repo_rel(reference_local_cpu_advisory, repo),
                            _repo_rel(reference_scorer_hashes, repo),
                        ],
                        "input_artifact_paths": [
                            reference_archive_path,
                            reference_inflate_sh_path,
                        ],
                        "include_postcondition_paths": True,
                    },
                }
            )
        if component_cache_source is not None:
            steps.append(
                {
                    "id": component_advisory_step_id,
                    "kind": "command",
                    "requires": ["emit_targeted_component_correction_work_order"],
                    "command": [
                        ".venv/bin/python",
                        "tools/import_frontier_component_response_cache.py",
                        "--source-local-cpu-advisory",
                        str(component_cache_source["source_local_cpu_advisory"]),
                        "--source-scorer-input-cache-hashes",
                        str(component_cache_source["source_scorer_input_cache_hashes"]),
                        "--local-cpu-advisory-out",
                        _repo_rel(local_cpu_advisory, repo),
                        "--scorer-input-cache-hashes-out",
                        _repo_rel(scorer_hashes, repo),
                        "--role",
                        "candidate",
                        "--overwrite",
                    ],
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(local_cpu_advisory, repo),
                            "axis_key": "score_axis",
                            "axis_equals": "cpu_advisory",
                        },
                        {
                            "type": "json_equals",
                            "path": _repo_rel(scorer_hashes, repo),
                            "key": "schema_version",
                            "equals": "mlx_scorer_input_cache_hashes.v1",
                        },
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(scorer_hashes, repo),
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [
                            _repo_rel(local_cpu_advisory, repo),
                            _repo_rel(scorer_hashes, repo),
                        ],
                        "input_artifact_paths": [
                            str(component_cache_source["source_local_cpu_advisory"]),
                            str(component_cache_source["source_scorer_input_cache_hashes"]),
                            *[
                                str(request.get("work_order_path") or "")
                                for request in request_metadata
                            ],
                        ],
                        "include_postcondition_paths": True,
                    },
                },
            )
        else:
            steps.append(
                {
                    "id": "local_cpu_component_advisory",
                    "kind": "command",
                    "requires": ["emit_targeted_component_correction_work_order"],
                    "command": [
                        ".venv/bin/python",
                        "experiments/contest_auth_eval.py",
                        "--archive",
                        archive_path,
                        "--inflate-sh",
                        inflate_sh_path,
                        "--upstream-dir",
                        str(upstream_dir),
                        "--video-names-file",
                        str(video_names_file),
                        "--device",
                        "cpu",
                        "--work-dir",
                        _repo_rel(local_cpu_work_dir, repo),
                        "--json-out",
                        _repo_rel(local_cpu_advisory, repo),
                        "--reuse-valid-json-out",
                        "--inflate-timeout",
                        "1800",
                        "--evaluate-timeout",
                        "1800",
                        "--keep-work-dir",
                        "--scorer-input-cache-hashes-out",
                        _repo_rel(scorer_hashes, repo),
                        "--allow-scorer-input-cache-artifact-output-outside-work-dir",
                    ],
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 3900,
                    "postconditions": [
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(local_cpu_advisory, repo),
                            "axis_key": "score_axis",
                            "axis_equals": "cpu_advisory",
                        },
                        {
                            "type": "json_equals",
                            "path": _repo_rel(scorer_hashes, repo),
                            "key": "schema_version",
                            "equals": "mlx_scorer_input_cache_hashes.v1",
                        },
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(scorer_hashes, repo),
                        }
                    ],
                    "telemetry": {
                        "artifact_paths": [
                            _repo_rel(local_cpu_advisory, repo),
                            _repo_rel(scorer_hashes, repo),
                        ],
                        "input_artifact_paths": [
                            archive_path,
                            inflate_sh_path,
                            *[
                                str(request.get("work_order_path") or "")
                                for request in request_metadata
                            ],
                        ],
                        "include_postcondition_paths": True,
                    },
                },
            )
        if include_mlx_response:
            mlx_cache_dir = shared_dir / "mlx_scorer_input_cache"
            mlx_cache_audit = shared_dir / "mlx_scorer_input_cache_audit.json"
            mlx_response = shared_dir / "mlx_scorer_response.json"
            local_mlx_response = mlx_response
            reused_mlx_cache_path = (
                str(component_cache_source.get("source_mlx_scorer_input_cache") or "")
                if isinstance(component_cache_source, Mapping)
                else ""
            )
            reused_mlx_cache_audit_path = (
                str(
                    component_cache_source.get("source_mlx_scorer_input_cache_audit")
                    or ""
                )
                if isinstance(component_cache_source, Mapping)
                else ""
            )
            mlx_cache_input_path = reused_mlx_cache_path or _repo_rel(mlx_cache_dir, repo)
            mlx_cache_audit_input_path = (
                reused_mlx_cache_audit_path or _repo_rel(mlx_cache_audit, repo)
            )
            mlx_cache_step_id = (
                "reuse_mlx_component_cache"
                if reused_mlx_cache_path and reused_mlx_cache_audit_path
                else "build_mlx_component_cache"
            )
            reference_mlx_cache_dir = (
                reference_shared_dir / "reference_mlx_scorer_input_cache"
            )
            reference_mlx_cache_audit = (
                reference_shared_dir / "reference_mlx_scorer_input_cache_audit.json"
            )
            reference_mlx_response = (
                reference_shared_dir / "reference_mlx_scorer_response.json"
            )
            reference_local_mlx_response = (
                reference_mlx_response if reference_eval_available else None
            )
            reused_reference_mlx_cache_path = (
                str(reference_cache_source.get("source_mlx_scorer_input_cache") or "")
                if isinstance(reference_cache_source, Mapping)
                else ""
            )
            reused_reference_mlx_cache_audit_path = (
                str(
                    reference_cache_source.get("source_mlx_scorer_input_cache_audit")
                    or ""
                )
                if isinstance(reference_cache_source, Mapping)
                else ""
            )
            reference_mlx_cache_input_path = (
                reused_reference_mlx_cache_path
                or _repo_rel(reference_mlx_cache_dir, repo)
            )
            reference_mlx_cache_audit_input_path = (
                reused_reference_mlx_cache_audit_path
                or _repo_rel(reference_mlx_cache_audit, repo)
            )
            reference_mlx_cache_step_id = (
                "reuse_reference_mlx_component_cache"
                if reused_reference_mlx_cache_path
                and reused_reference_mlx_cache_audit_path
                else "build_reference_mlx_component_cache"
            )
            for request in request_metadata:
                request["local_mlx_response_path"] = _repo_rel(mlx_response, repo)
                request["local_mlx_cache_path"] = mlx_cache_input_path
                request["local_mlx_cache_reuse_mode"] = (
                    component_cache_source.get("mlx_cache_reuse_mode")
                    if isinstance(component_cache_source, Mapping)
                    else None
                )
                if reference_eval_available:
                    request["reference_local_mlx_response_path"] = _repo_rel(
                        reference_mlx_response,
                        repo,
                    )
                    request["reference_local_mlx_cache_path"] = (
                        reference_mlx_cache_input_path
                    )
                    request["reference_local_mlx_cache_reuse_mode"] = (
                        reference_cache_source.get("mlx_cache_reuse_mode")
                        if isinstance(reference_cache_source, Mapping)
                        else None
                    )
            mlx_components_dir = shared_dir / "mlx_components"
            reference_mlx_components_dir = (
                reference_shared_dir / "reference_mlx_components"
            )
            build_cache_command = [
                ".venv/bin/python",
                "tools/build_mlx_scorer_input_cache_from_local_advisory.py",
                "--local-cpu-advisory",
                _repo_rel(local_cpu_advisory, repo),
                "--output-cache-dir",
                _repo_rel(mlx_cache_dir, repo),
                "--audit-output",
                _repo_rel(mlx_cache_audit, repo),
                "--expected-pair-count",
                "600",
                "--batch-pairs",
                "1",
                "--allow-large-tensor-cache",
                "--stamp-cache-manifest-on-pass",
                "--reuse-valid-cache",
            ]
            mlx_response_command = [
                ".venv/bin/python",
                "tools/run_mlx_scorer_response_from_local_advisory.py",
                "--local-cpu-advisory",
                _repo_rel(local_cpu_advisory, repo),
                "--reference-cache-dir",
                str(mlx_reference_cache_dir),
                "--candidate-cache-dir",
                mlx_cache_input_path,
                "--output",
                _repo_rel(mlx_response, repo),
                "--repo-root",
                ".",
                "--batch-pairs",
                "1",
                "--device",
                mlx_device,
                "--allow-local-cpu-advisory-cache-identity",
                "--components-dir",
                _repo_rel(mlx_components_dir, repo),
                "--response-family",
                "targeted_component_correction_receiver_closed_budget",
            ]
            if mlx_device == "gpu":
                mlx_response_command.append("--allow-gpu-research-signal")
            reference_mlx_steps: list[dict[str, Any]] = []
            if reference_eval_available:
                reference_build_cache_command = [
                    ".venv/bin/python",
                    "tools/build_mlx_scorer_input_cache_from_local_advisory.py",
                    "--local-cpu-advisory",
                    _repo_rel(reference_local_cpu_advisory, repo),
                    "--output-cache-dir",
                    _repo_rel(reference_mlx_cache_dir, repo),
                    "--audit-output",
                    _repo_rel(reference_mlx_cache_audit, repo),
                    "--expected-pair-count",
                    "600",
                    "--batch-pairs",
                    "1",
                    "--allow-large-tensor-cache",
                    "--stamp-cache-manifest-on-pass",
                    "--reuse-valid-cache",
                ]
                reference_mlx_response_command = [
                    ".venv/bin/python",
                    "tools/run_mlx_scorer_response_from_local_advisory.py",
                    "--local-cpu-advisory",
                    _repo_rel(reference_local_cpu_advisory, repo),
                    "--reference-cache-dir",
                    str(mlx_reference_cache_dir),
                    "--candidate-cache-dir",
                    reference_mlx_cache_input_path,
                    "--output",
                    _repo_rel(reference_mlx_response, repo),
                    "--repo-root",
                    ".",
                    "--batch-pairs",
                    "1",
                    "--device",
                    mlx_device,
                    "--allow-local-cpu-advisory-cache-identity",
                    "--components-dir",
                    _repo_rel(reference_mlx_components_dir, repo),
                    "--response-family",
                    "targeted_component_correction_receiver_closed_reference",
                ]
                if mlx_device == "gpu":
                    reference_mlx_response_command.append("--allow-gpu-research-signal")
                reference_mlx_steps = [
                    {
                        "id": reference_mlx_cache_step_id,
                        "kind": "command",
                        "requires": [reference_advisory_step_id],
                        "command": (
                            [
                                ".venv/bin/python",
                                "-c",
                                (
                                    "import json; print(json.dumps({'schema': "
                                    "'frontier_rate_attack_reused_mlx_component_cache.v1', "
                                    "'role': 'reference', 'score_claim': False, "
                                    "'promotion_eligible': False, "
                                    "'rank_or_kill_eligible': False, "
                                    "'ready_for_exact_eval_dispatch': False}))"
                                ),
                            ]
                            if reused_reference_mlx_cache_path
                            and reused_reference_mlx_cache_audit_path
                            else reference_build_cache_command
                        ),
                        "resources": {
                            "kind": (
                                "local_io_heavy"
                                if reused_reference_mlx_cache_path
                                and reused_reference_mlx_cache_audit_path
                                else "local_cpu"
                            )
                        },
                        "timeout_seconds": (
                            120
                            if reused_reference_mlx_cache_path
                            and reused_reference_mlx_cache_audit_path
                            else 1800
                        ),
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": reference_mlx_cache_audit_input_path,
                                "key": "passed",
                                "equals": True,
                            },
                            {
                                "type": "json_false_authority",
                                "path": reference_mlx_cache_audit_input_path,
                            },
                            {
                                "type": "json_false_authority",
                                "path": f"{reference_mlx_cache_input_path}/manifest.json",
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                reference_mlx_cache_input_path,
                                reference_mlx_cache_audit_input_path,
                            ],
                            "input_artifact_paths": [
                                _repo_rel(reference_local_cpu_advisory, repo)
                            ],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "reference_local_mlx_component_response",
                        "kind": "command",
                        "requires": [reference_mlx_cache_step_id],
                        "command": reference_mlx_response_command,
                        "resources": {
                            "kind": "local_mlx" if mlx_device == "gpu" else "local_cpu"
                        },
                        "timeout_seconds": 900,
                        "postconditions": [
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(reference_mlx_response, repo),
                                "axis_key": "score_axis",
                                "axis_equals": "[macOS-MLX research-signal]",
                            }
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(reference_mlx_response, repo),
                                _repo_rel(reference_mlx_components_dir, repo),
                            ],
                            "input_artifact_paths": [
                                _repo_rel(reference_local_cpu_advisory, repo),
                                str(mlx_reference_cache_dir),
                                reference_mlx_cache_input_path,
                            ],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    },
                ]
            steps.extend(
                [
                    {
                        "id": mlx_cache_step_id,
                        "kind": "command",
                        "requires": [component_advisory_step_id],
                        "command": (
                            [
                                ".venv/bin/python",
                                "-c",
                                (
                                    "import json; print(json.dumps({'schema': "
                                    "'frontier_rate_attack_reused_mlx_component_cache.v1', "
                                    "'role': 'candidate', 'score_claim': False, "
                                    "'promotion_eligible': False, "
                                    "'rank_or_kill_eligible': False, "
                                    "'ready_for_exact_eval_dispatch': False}))"
                                ),
                            ]
                            if reused_mlx_cache_path and reused_mlx_cache_audit_path
                            else build_cache_command
                        ),
                        "resources": {
                            "kind": (
                                "local_io_heavy"
                                if reused_mlx_cache_path and reused_mlx_cache_audit_path
                                else "local_cpu"
                            )
                        },
                        "timeout_seconds": (
                            120
                            if reused_mlx_cache_path and reused_mlx_cache_audit_path
                            else 1800
                        ),
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": mlx_cache_audit_input_path,
                                "key": "passed",
                                "equals": True,
                            },
                            {
                                "type": "json_false_authority",
                                "path": mlx_cache_audit_input_path,
                            },
                            {
                                "type": "json_false_authority",
                                "path": f"{mlx_cache_input_path}/manifest.json",
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                mlx_cache_input_path,
                                mlx_cache_audit_input_path,
                            ],
                            "input_artifact_paths": [_repo_rel(local_cpu_advisory, repo)],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "local_mlx_component_response",
                        "kind": "command",
                        "requires": [mlx_cache_step_id],
                        "command": mlx_response_command,
                        "resources": {
                            "kind": "local_mlx" if mlx_device == "gpu" else "local_cpu"
                        },
                        "timeout_seconds": 900,
                        "postconditions": [
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(mlx_response, repo),
                                "axis_key": "score_axis",
                                "axis_equals": "[macOS-MLX research-signal]",
                            }
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(mlx_response, repo),
                                _repo_rel(mlx_components_dir, repo),
                            ],
                            "input_artifact_paths": [
                                _repo_rel(local_cpu_advisory, repo),
                                str(mlx_reference_cache_dir),
                                mlx_cache_input_path,
                            ],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    },
                    *reference_mlx_steps,
                ]
            )
        for row_index, request in enumerate(request_metadata, start=1):
            work_order_path_text = str(request.get("work_order_path") or "")
            response_harvest_path_text = str(
                request.get("component_correction_response_harvest_path") or ""
            )
            harvest_command = [
                ".venv/bin/python",
                "tools/harvest_frontier_targeted_component_correction_response.py",
                "--work-order",
                work_order_path_text,
                "--local-cpu-advisory",
                _repo_rel(local_cpu_advisory, repo),
                "--output",
                response_harvest_path_text,
                "--repo-root",
                ".",
            ]
            harvest_requires = [component_advisory_step_id]
            harvest_input_paths = [
                work_order_path_text,
                _repo_rel(local_cpu_advisory, repo),
            ]
            if reference_eval_available:
                harvest_command.extend(
                    [
                        "--reference-local-cpu-advisory",
                        _repo_rel(reference_local_cpu_advisory, repo),
                    ]
                )
                harvest_requires.append(reference_advisory_step_id)
                harvest_input_paths.append(_repo_rel(reference_local_cpu_advisory, repo))
            if local_mlx_response is not None:
                harvest_command.extend(
                    ["--local-mlx-response", _repo_rel(local_mlx_response, repo)]
                )
                if reference_local_mlx_response is not None:
                    harvest_command.extend(
                        [
                            "--reference-local-mlx-response",
                            _repo_rel(reference_local_mlx_response, repo),
                        ]
                    )
                harvest_requires = [
                    "local_mlx_component_response",
                    *(
                        [reference_advisory_step_id]
                        if reference_eval_available
                        else []
                    ),
                    *(
                        ["reference_local_mlx_component_response"]
                        if reference_local_mlx_response is not None
                        else []
                    ),
                ]
                harvest_input_paths.append(_repo_rel(local_mlx_response, repo))
                if reference_local_mlx_response is not None:
                    harvest_input_paths.append(
                        _repo_rel(reference_local_mlx_response, repo)
                    )
            steps.append(
                {
                    "id": f"harvest_targeted_component_correction_response_{row_index:02d}",
                    "kind": "command",
                    "requires": [
                        f"emit_targeted_component_correction_work_order_{row_index:02d}",
                        *harvest_requires,
                    ],
                    "command": harvest_command,
                    "resources": {"kind": "local_io_heavy"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": response_harvest_path_text,
                            "key": "schema",
                            "equals": TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA,
                        },
                        {
                            "type": "json_false_authority",
                            "path": response_harvest_path_text,
                        },
                        {
                            "type": "json_equals",
                            "path": response_harvest_path_text,
                            "key": "ready_for_budget_spend_count",
                            "equals": 0,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [response_harvest_path_text],
                        "input_artifact_paths": harvest_input_paths,
                        "include_postcondition_paths": True,
                    },
                }
            )
        experiments.append(
            {
                "id": _slug_token(candidate_id),
                "status": "queued",
                "priority": priority,
                "lane_id": "lane_frontier_targeted_component_correction_20260526",
                "tags": [
                    "targeted_component_correction",
                    "candidate_shared_component_response",
                    *[
                        str(row.get("correction_family") or "unknown_family")
                        for row in candidate_rows
                    ],
                    *[
                        str(row.get("target_kind") or "unknown_target")
                        for row in candidate_rows
                    ],
                    "receiver_closed_budget",
                ],
                "metadata": {
                    "schema": TARGETED_COMPONENT_CORRECTION_QUEUE_METADATA_SCHEMA,
                    "candidate_id": candidate_id,
                    "frontier_target_optimization_profile": dict(
                        target_profile_metadata
                    ),
                    "selected_acquisition_count": len(candidate_rows),
                    "selected_correction_families": _unique_strings(
                        row.get("correction_family") for row in candidate_rows
                    ),
                    "selected_target_kinds": _unique_strings(
                        row.get("target_kind") for row in candidate_rows
                    ),
                    "correction_requests": request_metadata,
                    "submission_dir": primary_row.get("submission_dir"),
                    "work_dir": _repo_rel(candidate_dir, repo),
                    "local_cpu_advisory_path": _repo_rel(local_cpu_advisory, repo),
                    "local_cpu_advisory_step_id": component_advisory_step_id,
                    "local_cpu_advisory_reuse_mode": (
                        component_cache_source.get("reuse_mode")
                        if isinstance(component_cache_source, Mapping)
                        else "queue_owned_local_cpu_component_advisory"
                    ),
                    "local_cpu_advisory_source_path": (
                        component_cache_source.get("source_local_cpu_advisory")
                        if isinstance(component_cache_source, Mapping)
                        else None
                    ),
                    "candidate_cache_key": candidate_cache_key,
                    "shared_component_response_dir": _repo_rel(shared_dir, repo),
                    "reference_local_cpu_advisory_path": (
                        _repo_rel(reference_local_cpu_advisory, repo)
                        if reference_eval_available
                        else None
                    ),
                    "reference_local_cpu_advisory_step_id": (
                        reference_advisory_step_id if reference_eval_available else None
                    ),
                    "reference_local_cpu_advisory_reuse_mode": (
                        reference_cache_source.get("reuse_mode")
                        if isinstance(reference_cache_source, Mapping)
                        else (
                            "queue_owned_local_cpu_component_advisory"
                            if reference_eval_available
                            else None
                        )
                    ),
                    "reference_local_cpu_advisory_source_path": (
                        reference_cache_source.get("source_local_cpu_advisory")
                        if isinstance(reference_cache_source, Mapping)
                        else None
                    ),
                    "reference_archive_path": reference_archive_path or None,
                    "reference_inflate_sh_path": reference_inflate_sh_path or None,
                    "reference_component_eval_available": reference_eval_available,
                    "reference_cache_key": reference_cache_key,
                    "reference_shared_component_response_dir": (
                        _repo_rel(reference_shared_dir, repo)
                        if reference_eval_available
                        else None
                    ),
                    "local_mlx_response_path": (
                        None
                        if local_mlx_response is None
                        else _repo_rel(local_mlx_response, repo)
                    ),
                    "local_mlx_cache_path": (
                        None if local_mlx_response is None else mlx_cache_input_path
                    ),
                    "local_mlx_cache_reuse_mode": (
                        component_cache_source.get("mlx_cache_reuse_mode")
                        if isinstance(component_cache_source, Mapping)
                        else None
                    ),
                    "reference_local_mlx_response_path": (
                        None
                        if reference_local_mlx_response is None
                        else _repo_rel(reference_local_mlx_response, repo)
                    ),
                    "reference_local_mlx_cache_path": (
                        reference_mlx_cache_input_path
                        if reference_local_mlx_response is not None
                        else None
                    ),
                    "reference_local_mlx_cache_reuse_mode": (
                        reference_cache_source.get("mlx_cache_reuse_mode")
                        if isinstance(reference_cache_source, Mapping)
                        else None
                    ),
                    "local_mlx_response_enabled": include_mlx_response,
                    "mlx_device": mlx_device,
                    "shared_component_response_reuse": True,
                    "deduped_full_local_cpu_eval_count": max(0, len(candidate_rows) - 1),
                    "budget_spend_ready": False,
                    "selection_policy": dict(selection_policy),
                    "repair_operator_portfolio": dict(repair_operator_portfolio),
                    "allowed_use": (
                        "targeted_component_correction_queue_metadata_only"
                    ),
                    "forbidden_use": "score_claim_or_dispatch_authority",
                    **FALSE_AUTHORITY,
                },
                "steps": steps,
            }
        )
    queue_metadata = {
        "schema": (
            "frontier_rate_attack_targeted_component_correction_queue_root_"
            "metadata.v1"
        ),
        "targeted_component_correction_acquisition": (
            _targeted_component_correction_queue_metadata(
                targeted_component_correction_acquisition
            )
        ),
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "selection_policy": dict(selection_policy),
        "selected_candidate_count": len(candidate_order),
        "selected_row_count": len(selected_rows),
        "candidate_limit": candidate_limit,
        "results_root": _repo_rel(results_base, repo),
        "source_artifact_paths": [_repo_rel(acquisition_path, repo)],
        "allowed_use": "targeted_component_correction_queue_metadata_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        queue_metadata,
        context="targeted_component_correction_queue_root_metadata",
    )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": local_cpu_concurrency,
                    "local_io_heavy": 1,
                    "local_mlx": local_mlx_concurrency if include_mlx_response else 0,
                    "modal_cpu": 0,
                    "modal_gpu": 0,
                },
            },
            "metadata": queue_metadata,
            "experiments": experiments,
        }
    )
    queue["selection_policy"] = selection_policy
    return queue


def _receiver_repair_classification(blocker: str) -> dict[str, Any]:
    text = str(blocker or "")
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "full_frame_inflate_parity",
            "inflate_output_parity",
            "strict_full_frame",
        )
    ):
        family = "full_frame_inflate_parity_repair"
        action = "run_source_and_candidate_inflate_then_compare_full_frame_outputs"
        consumer = "frontier_exact_readiness_handoff"
        queue_actionable = True
        priority_base = 95.0
    elif any(
        token in lowered
        for token in (
            "runtime_consumption_proof",
            "consumed_by_runtime",
            "no_op_detector",
        )
    ):
        family = "runtime_consumption_proof_repair"
        action = "replace_parser_only_signal_with_runtime_adapter_consumption_proof"
        consumer = "frontier_final_rate_attack_materializer_queue"
        queue_actionable = True
        priority_base = 90.0
    elif any(
        token in lowered
        for token in (
            "receiver_contract",
            "runtime_adapter",
            "native_unpacker",
            "shadow_archive_reconstruction",
            "receiver_runtime",
            "receiver_binding",
            "materializer_or_receiver",
            "queue_context",
        )
    ):
        family = "receiver_runtime_contract_repair"
        action = "generate_or_harden_receiver_runtime_adapter_contract"
        consumer = "frontier_final_rate_attack_materializer_queue"
        queue_actionable = True
        priority_base = 88.0
    elif any(
        token in lowered
        for token in (
            "inflate_sh_missing",
            "report_txt_missing",
            "archive_manifest_missing",
            "runtime_tree_sha256_missing",
            "runtime_content_tree_sha256_missing",
            "runtime_manifest",
        )
    ):
        family = "submission_runtime_manifest_closure"
        action = "materialize_submission_runtime_manifest_and_runtime_tree_hashes"
        consumer = "frontier_exact_readiness_handoff"
        queue_actionable = True
        priority_base = 82.0
    elif "above_active_floor_archive_bytes" in lowered:
        family = "rate_floor_scope_control"
        action = "recompute_active_floor_byte_delta_or_require_operator_override"
        consumer = "frontier_rate_attack_feedback_refresh"
        queue_actionable = False
        priority_base = 35.0
    elif any(
        token in lowered
        for token in (
            "dispatch_authority",
            "exact_eval_readiness_gate",
            "lane_dispatch_claim",
            "non_proxy_score_evidence",
            "optimizer_candidate_queue_is_planning_only",
            "planning_only",
            "promotion",
        )
    ):
        family = "authority_gate"
        action = "preserve_false_authority_and_wait_for_exact_readiness_payload"
        consumer = "false_authority_guard"
        queue_actionable = False
        priority_base = 10.0
    else:
        family = "unclassified_receiver_exact_readiness_repair"
        action = "classify_blocker_then_bind_to_receiver_runtime_or_exact_readiness_step"
        consumer = "frontier_rate_attack_feedback_refresh"
        queue_actionable = False
        priority_base = 45.0
    return {
        "repair_family": family,
        "recommended_next_action": action,
        "queue_consumer": consumer,
        "queue_actionable": queue_actionable,
        "priority_base": priority_base,
    }


def _bridge_blocker_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw_counts = summary.get("blocker_counts")
    if isinstance(raw_counts, Mapping):
        for blocker, count in raw_counts.items():
            text = str(blocker or "").strip()
            if not text:
                continue
            counts[text] = counts.get(text, 0) + (_finite_int_or_none(count) or 1)
    for blocker in _string_list(summary.get("top_blockers")):
        counts.setdefault(blocker, 1)
    for report in summary.get("reports") or []:
        if not isinstance(report, Mapping):
            continue
        for blocker in _string_list(report.get("row_blockers_sample")):
            counts.setdefault(blocker, 1)
    return counts


def _bridge_summaries_from_operation_row(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    evidence = row.get("evidence_summary")
    if not isinstance(evidence, Mapping):
        return []
    summaries: list[Mapping[str, Any]] = []
    for key in ("exact_readiness_bridge", "exact_readiness_bridge_summary"):
        value = evidence.get(key)
        if isinstance(value, Mapping):
            summaries.append(value)
    return summaries


def _bridge_report_rows(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        report
        for report in summary.get("reports") or []
        if isinstance(report, Mapping)
    ]


def _bridge_report_paths(summary: Mapping[str, Any]) -> list[str]:
    return _unique_strings(
        report.get("path") for report in _bridge_report_rows(summary)
    )


def _bridge_candidate_ids(summary: Mapping[str, Any]) -> list[str]:
    candidate_ids: list[Any] = []
    for report in _bridge_report_rows(summary):
        candidate_ids.extend(_string_list(report.get("candidate_ids")))
    return _unique_strings(candidate_ids)


def _bridge_row_blocker_samples(summary: Mapping[str, Any]) -> list[str]:
    blockers: list[Any] = []
    for report in _bridge_report_rows(summary):
        blockers.extend(_string_list(report.get("row_blockers_sample")))
    return _unique_strings(blockers)[:16]


def _saved_bytes_at_risk(row: Mapping[str, Any]) -> int:
    evidence = row.get("evidence_summary")
    if not isinstance(evidence, Mapping):
        return 0
    for key in (
        "total_positive_saved_bytes_from_observed_parts",
        "total_positive_saved_bytes",
        "max_saved_bytes",
    ):
        saved = _finite_int_or_none(evidence.get(key))
        if saved is not None and saved > 0:
            return saved
    return 0


def _source_blocker_receiver_repair_candidate(blocker: str) -> bool:
    lowered = str(blocker or "").lower()
    if lowered.startswith("exact_readiness_bridge:"):
        return False
    return any(
        token in lowered
        for token in (
            "receiver",
            "runtime",
            "proof",
            "parity",
            "exact_readiness",
            "queue_context",
            "materializer_or_receiver",
            "payload_grammar",
        )
    )


def _receiver_repair_priority(
    *,
    priority_base: float,
    blocker_count: int,
    saved_bytes_at_risk: int,
    source_priority: float | None,
) -> float:
    priority = priority_base + float(max(blocker_count, 1)) * 3.0
    priority += float(max(saved_bytes_at_risk, 0)) / 16.0
    if source_priority is not None and source_priority > 0:
        priority += min(float(source_priority), 100.0) / 10.0
    return priority


def build_frontier_receiver_repair_backlog(
    operation_portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile exact-readiness blockers into receiver-runtime repair work orders."""

    rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    queue_actionable_count = 0
    targeted_budget = (
        operation_portfolio.get("targeted_correction_budget_summary")
        if isinstance(
            operation_portfolio.get("targeted_correction_budget_summary"),
            Mapping,
        )
        else {}
    )
    materializer_budget_total = _finite_int_or_none(
        targeted_budget.get("materializer_rate_positive_saved_bytes_total")
        if isinstance(targeted_budget, Mapping)
        else None
    ) or 0
    local_budget_total = _finite_int_or_none(
        targeted_budget.get("local_drop_saved_bytes_total")
        if isinstance(targeted_budget, Mapping)
        else None
    ) or 0
    for source_row in operation_portfolio.get("rows") or []:
        if not isinstance(source_row, Mapping):
            continue
        source_operation_id = str(source_row.get("operation_id") or "unknown_operation")
        bridge_summaries = _bridge_summaries_from_operation_row(source_row)
        bridge_blockers: set[str] = set()
        source_priority = _finite_float_or_none(source_row.get("priority_score"))
        saved_bytes = _saved_bytes_at_risk(source_row)
        source_queue_consumer = str(
            source_row.get("queue_consumer")
            or "frontier_final_rate_attack_materializer_queue"
        )
        evidence = source_row.get("evidence_summary")
        target_kind = (
            str(evidence.get("target_kind") or source_operation_id)
            if isinstance(evidence, Mapping)
            else source_operation_id
        )
        for summary_index, bridge_summary in enumerate(bridge_summaries):
            bridge_blockers.update(_bridge_blocker_counts(bridge_summary))
            candidate_count = _finite_int_or_none(
                bridge_summary.get("candidate_count")
            ) or 0
            ready_count = _finite_int_or_none(
                bridge_summary.get("ready_candidate_count")
            ) or 0
            blocked_count = _finite_int_or_none(
                bridge_summary.get("blocked_candidate_count")
            ) or 0
            bridge_report_paths = _bridge_report_paths(bridge_summary)
            bridge_candidate_ids = _bridge_candidate_ids(bridge_summary)
            bridge_row_blockers = _bridge_row_blocker_samples(bridge_summary)
            for blocker, count in sorted(_bridge_blocker_counts(bridge_summary).items()):
                classification = _receiver_repair_classification(blocker)
                family = str(classification["repair_family"])
                family_counts[family] = family_counts.get(family, 0) + 1
                queue_actionable = bool(classification["queue_actionable"])
                if queue_actionable:
                    queue_actionable_count += 1
                blocker_count = int(count)
                repair_id = (
                    "receiver_repair_"
                    f"{_slug_token(source_operation_id)}_"
                    f"{_slug_token(family)}_"
                    f"{_slug_token(blocker)}"
                )
                if len(bridge_summaries) > 1:
                    repair_id = f"{repair_id}_bridge{summary_index}"
                rows.append(
                    {
                        "schema": RECEIVER_REPAIR_ROW_SCHEMA,
                        "repair_id": repair_id,
                        "source_operation_id": source_operation_id,
                        "target_kind": target_kind,
                        "repair_family": family,
                        "blocker": blocker,
                        "blocker_count": blocker_count,
                        "candidate_count": candidate_count,
                        "ready_candidate_count": ready_count,
                        "blocked_candidate_count": blocked_count,
                        "bridge_report_paths": bridge_report_paths,
                        "candidate_ids": bridge_candidate_ids,
                        "row_blocker_samples": bridge_row_blockers,
                        "saved_bytes_at_risk": saved_bytes,
                        "priority_score": _receiver_repair_priority(
                            priority_base=float(classification["priority_base"]),
                            blocker_count=blocker_count,
                            saved_bytes_at_risk=saved_bytes,
                            source_priority=source_priority,
                        ),
                        "queue_consumer": classification["queue_consumer"],
                        "source_queue_consumer": source_queue_consumer,
                        "queue_actionable": queue_actionable,
                        "recommended_next_action": classification[
                            "recommended_next_action"
                        ],
                        "correction_budget_context": {
                            "materializer_rate_positive_saved_bytes_total": (
                                materializer_budget_total
                            ),
                            "local_drop_saved_bytes_total": local_budget_total,
                            "saved_bytes_at_risk_from_source_operation": saved_bytes,
                            "spend_policy": (
                                "receiver_repair_must_prove_runtime_consumption_before_"
                                "freed_rate_budget_can_fund_segnet_posenet_corrections"
                            ),
                            **FALSE_AUTHORITY,
                        },
                        "allowed_use": "receiver_runtime_repair_planning_only",
                        "forbidden_use": (
                            "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                        ),
                        **FALSE_AUTHORITY,
                    }
                )
        for blocker in _string_list(source_row.get("blockers")):
            if blocker in bridge_blockers:
                continue
            if not _source_blocker_receiver_repair_candidate(blocker):
                continue
            classification = _receiver_repair_classification(blocker)
            family = str(classification["repair_family"])
            family_counts[family] = family_counts.get(family, 0) + 1
            queue_actionable = bool(classification["queue_actionable"])
            if queue_actionable:
                queue_actionable_count += 1
            repair_id = (
                "receiver_repair_"
                f"{_slug_token(source_operation_id)}_"
                f"{_slug_token(family)}_"
                f"{_slug_token(blocker)}"
            )
            rows.append(
                {
                    "schema": RECEIVER_REPAIR_ROW_SCHEMA,
                    "repair_id": repair_id,
                    "source_operation_id": source_operation_id,
                    "target_kind": target_kind,
                    "repair_family": family,
                    "blocker": blocker,
                    "blocker_count": 1,
                    "candidate_count": 0,
                    "ready_candidate_count": 0,
                    "blocked_candidate_count": 0,
                    "bridge_report_paths": [],
                    "candidate_ids": [],
                    "row_blocker_samples": [],
                    "saved_bytes_at_risk": saved_bytes,
                    "priority_score": _receiver_repair_priority(
                        priority_base=float(classification["priority_base"]),
                        blocker_count=1,
                        saved_bytes_at_risk=saved_bytes,
                        source_priority=source_priority,
                    ),
                    "queue_consumer": classification["queue_consumer"],
                    "source_queue_consumer": source_queue_consumer,
                    "queue_actionable": queue_actionable,
                    "recommended_next_action": classification[
                        "recommended_next_action"
                    ],
                    "correction_budget_context": {
                        "materializer_rate_positive_saved_bytes_total": (
                            materializer_budget_total
                        ),
                        "local_drop_saved_bytes_total": local_budget_total,
                        "saved_bytes_at_risk_from_source_operation": saved_bytes,
                        "spend_policy": (
                            "receiver_repair_must_prove_runtime_consumption_before_"
                            "freed_rate_budget_can_fund_segnet_posenet_corrections"
                        ),
                        **FALSE_AUTHORITY,
                    },
                    "allowed_use": "receiver_runtime_repair_planning_only",
                    "forbidden_use": (
                        "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                    ),
                    **FALSE_AUTHORITY,
                }
            )
    rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get("priority_score") or 0.0),
            str(row.get("repair_id") or ""),
        ),
    )
    top_rows = rows[:8]
    return {
        "schema": RECEIVER_REPAIR_BACKLOG_SCHEMA,
        "generated_at_utc": _utc_now(),
        "operation_portfolio_schema": operation_portfolio.get("schema"),
        "row_count": len(rows),
        "queue_actionable_repair_count": queue_actionable_count,
        "authority_gate_repair_count": family_counts.get("authority_gate", 0),
        "repair_family_counts": dict(sorted(family_counts.items())),
        "top_repair_ids": [str(row.get("repair_id") or "") for row in top_rows],
        "top_repair_families": _unique_strings(
            [row.get("repair_family") for row in top_rows]
        ),
        "top_source_operation_ids": _unique_strings(
            [row.get("source_operation_id") for row in top_rows]
        ),
        "materializer_rate_positive_saved_bytes_total": materializer_budget_total,
        "local_drop_saved_bytes_total": local_budget_total,
        "targeted_correction_budget_active": (
            targeted_budget.get("active") is True
            if isinstance(targeted_budget, Mapping)
            else False
        ),
        "recommended_next_action": (
            "repair_top_receiver_runtime_and_exact_readiness_blockers_before_spending_"
            "freed_rate_budget_on_segnet_posenet_targeted_corrections"
        ),
        "rows": rows,
        "allowed_use": "queue_owned_receiver_repair_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _receiver_repair_backlog_queue_metadata(backlog: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "frontier_rate_attack_receiver_repair_backlog_queue_metadata.v1",
        "receiver_repair_backlog_schema": backlog.get("schema"),
        "row_count": backlog.get("row_count"),
        "queue_actionable_repair_count": backlog.get("queue_actionable_repair_count"),
        "authority_gate_repair_count": backlog.get("authority_gate_repair_count"),
        "top_repair_ids": list(backlog.get("top_repair_ids") or []),
        "top_repair_families": list(backlog.get("top_repair_families") or []),
        "top_source_operation_ids": list(backlog.get("top_source_operation_ids") or []),
        "materializer_rate_positive_saved_bytes_total": backlog.get(
            "materializer_rate_positive_saved_bytes_total"
        ),
        "local_drop_saved_bytes_total": backlog.get("local_drop_saved_bytes_total"),
        "targeted_correction_budget_active": (
            backlog.get("targeted_correction_budget_active") is True
        ),
        "allowed_use": "queue_metadata_pointer_to_receiver_repair_backlog_artifact",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _receiver_repair_row_by_id(
    backlog: Mapping[str, Any],
    repair_id: str,
) -> Mapping[str, Any]:
    for row in backlog.get("rows") or []:
        if isinstance(row, Mapping) and row.get("repair_id") == repair_id:
            return row
    raise FrontierRateAttackFeedbackError(f"unknown receiver repair id: {repair_id}")


def _bridge_report_details(
    *,
    bridge_report_paths: Sequence[str],
    candidate_ids: Sequence[str],
    repo_root: Path,
) -> dict[str, Any]:
    candidate_filter = {str(candidate) for candidate in candidate_ids if candidate}
    reports: list[dict[str, Any]] = []
    source_queue_paths: list[str] = []
    exact_readiness_report_paths: list[str] = []
    exact_ready_queue_paths: list[str] = []
    source_manifest_paths: list[str] = []
    candidate_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for rel_path in bridge_report_paths:
        path = _resolve_path(rel_path, repo_root=repo_root)
        if not path.is_file():
            blockers.append(f"bridge_report_missing:{rel_path}")
            continue
        payload = _load_json(path)
        try:
            require_no_truthy_authority_fields(
                payload,
                context=f"receiver_repair_bridge_report:{rel_path}",
            )
        except ValueError as exc:
            blockers.append(f"bridge_report_authority_leak:{rel_path}:{exc}")
            continue
        source_queue = payload.get("source_queue_path")
        if isinstance(source_queue, str) and source_queue.strip():
            source_queue_paths.append(source_queue)
        for bridge_row in payload.get("rows") or []:
            if not isinstance(bridge_row, Mapping):
                continue
            candidate_id = str(bridge_row.get("candidate_id") or "")
            if candidate_filter and candidate_id not in candidate_filter:
                continue
            exact_report = bridge_row.get("exact_readiness_report_path")
            if isinstance(exact_report, str) and exact_report.strip():
                exact_readiness_report_paths.append(exact_report)
            exact_queue = bridge_row.get("exact_ready_queue_path")
            if isinstance(exact_queue, str) and exact_queue.strip():
                exact_ready_queue_paths.append(exact_queue)
        reports.append(
            {
                "path": _repo_rel(path, repo_root),
                "schema": payload.get("schema"),
                "candidate_count": payload.get("candidate_count"),
                "ready_candidate_count": payload.get("ready_candidate_count"),
                "blocked_candidate_count": payload.get("blocked_candidate_count"),
                "source_queue_path": source_queue,
                "dispatch_blockers": _string_list(payload.get("dispatch_blockers")),
                **FALSE_AUTHORITY,
            }
        )

    for source_queue in _unique_strings(source_queue_paths):
        queue_path = _resolve_path(source_queue, repo_root=repo_root)
        if not queue_path.is_file():
            blockers.append(f"source_queue_missing:{source_queue}")
            continue
        payload = _load_json(queue_path)
        rows: list[Mapping[str, Any]] = []
        for key in ("top_k", "dispatch_ready"):
            for row in payload.get(key) or []:
                if isinstance(row, Mapping):
                    rows.append(row)
        seen_rows: set[str] = set()
        for source_row in rows:
            candidate_id = str(source_row.get("candidate_id") or "")
            if candidate_filter and candidate_id not in candidate_filter:
                continue
            stable_key = candidate_id or json.dumps(source_row, sort_keys=True)
            if stable_key in seen_rows:
                continue
            seen_rows.add(stable_key)
            source_manifest = source_row.get("source_manifest_path")
            if isinstance(source_manifest, str) and source_manifest.strip():
                source_manifest_paths.append(source_manifest)
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target_kind": source_row.get("target_kind"),
                    "candidate_archive_path": source_row.get("candidate_archive_path")
                    or source_row.get("archive_path"),
                    "candidate_archive_sha256": source_row.get(
                        "candidate_archive_sha256"
                    )
                    or source_row.get("archive_sha256"),
                    "source_archive_path": source_row.get("source_archive_path"),
                    "source_archive_sha256": source_row.get("source_archive_sha256"),
                    "source_manifest_path": source_manifest,
                    "runtime_consumption_proof_path": source_row.get(
                        "runtime_consumption_proof_path"
                    ),
                    "receiver_contract_kind": source_row.get(
                        "receiver_contract_kind"
                    ),
                    "receiver_contract_satisfied": source_row.get(
                        "receiver_contract_satisfied"
                    )
                    is True,
                    "runtime_adapter_ready": source_row.get("runtime_adapter_ready")
                    is True,
                    "readiness_blockers": _string_list(
                        source_row.get("readiness_blockers")
                    ),
                    **FALSE_AUTHORITY,
                }
            )
    return {
        "bridge_reports": reports,
        "source_queue_paths": _unique_strings(source_queue_paths),
        "exact_readiness_report_paths": _unique_strings(exact_readiness_report_paths),
        "exact_ready_queue_paths": _unique_strings(exact_ready_queue_paths),
        "source_manifest_paths": _unique_strings(source_manifest_paths),
        "candidate_rows": candidate_rows,
        "blockers": _unique_strings(blockers),
    }


def _receiver_repair_command_hints(
    *,
    row: Mapping[str, Any],
    bridge_details: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bridge_paths = _string_list(row.get("bridge_report_paths"))
    target_kind = str(row.get("target_kind") or "")
    repair_family = str(row.get("repair_family") or "")
    hints: list[dict[str, Any]] = []
    if bridge_paths:
        hints.append(
            {
                "action_id": "rebuild_exact_eval_consumer_after_receiver_repair",
                "when": "after_exact_ready_queue_written_true",
                "command_template": [
                    ".venv/bin/python",
                    "tools/build_materializer_exact_eval_consumer.py",
                    *[
                        item
                        for bridge_path in bridge_paths
                        for item in ("--bridge-report", bridge_path)
                    ],
                    "--consumer-report-out",
                    "<receiver_repair_dir>/exact_eval_consumer_report.json",
                    "--experiment-queue-out",
                    "<receiver_repair_dir>/exact_eval_consumer_queue.json",
                    "--overwrite",
                ],
                "blocked_until": [
                    "receiver_runtime_contract_satisfied",
                    "runtime_consumption_proof_passed",
                    "exact_ready_queue_written_true",
                ],
                **FALSE_AUTHORITY,
            }
        )
    if repair_family == "submission_runtime_manifest_closure":
        hints.append(
            {
                "action_id": "build_materializer_submission_runtime_closure",
                "target_kind": target_kind,
                "queue_consumer": row.get("queue_consumer"),
                "source_queue_paths": list(bridge_details.get("source_queue_paths") or []),
                "candidate_ids": list(row.get("candidate_ids") or []),
                "command_template": [
                    ".venv/bin/python",
                    MATERIALIZER_SUBMISSION_CLOSURE_TOOL,
                    "--source-queue",
                    "<source_queue_path>",
                    "--candidate-id",
                    "<candidate_id>",
                    "--submission-dir-out",
                    "<receiver_repair_dir>/submission_closure/submission",
                    "--closed-source-queue-out",
                    "<receiver_repair_dir>/submission_closure/closed_source_queue.json",
                    "--closure-report-out",
                    "<receiver_repair_dir>/submission_closure/submission_closure_report.json",
                    "--overwrite",
                ],
                "blocked_until": [
                    "receiver_contract_satisfied",
                    "runtime_consumption_proof_present",
                    "source_runtime_dir_with_inflate_sh",
                ],
                **FALSE_AUTHORITY,
            }
        )
    if repair_family in {
        "runtime_consumption_proof_repair",
        "receiver_runtime_contract_repair",
    }:
        required_context = [
            "source_runtime_dir",
            "packet_or_section_materializer_context",
        ]
        if target_kind == "packet_member_merge_v1":
            required_context.append("packet_member_merge_source_runtime_dir")
        if target_kind == "renderer_payload_dfl1_v1":
            required_context.append("full_frame_inflate_parity_proof")
        hints.append(
            {
                "action_id": "rerun_materializer_with_receiver_runtime_context",
                "target_kind": target_kind,
                "queue_consumer": row.get("queue_consumer"),
                "source_manifest_paths": list(
                    bridge_details.get("source_manifest_paths") or []
                ),
                "required_context": _unique_strings(required_context),
                "command_template": [
                    ".venv/bin/python",
                    "tools/run_family_agnostic_materializer.py",
                    "--target-kind",
                    target_kind or "<target_kind>",
                    "--archive-path",
                    "<source_archive_path>",
                    "--output-archive",
                    "<receiver_repair_dir>/candidate.zip",
                    "--output-manifest",
                    "<receiver_repair_dir>/candidate.json",
                    "--runtime-consumption-proof-out",
                    "<receiver_repair_dir>/candidate.runtime_consumption_proof.json",
                    "--allow-overwrite",
                ],
                "blocked_until": required_context,
                **FALSE_AUTHORITY,
            }
        )
    if repair_family == "full_frame_inflate_parity_repair":
        hints.append(
            {
                "action_id": "prove_same_runtime_full_frame_inflate_parity",
                "target_kind": target_kind,
                "command_template": [
                    ".venv/bin/python",
                    "tools/prove_shell_inflate_parity.py",
                    "--left-archive",
                    "<source_archive_path>",
                    "--left-submission-dir",
                    "<source_runtime_dir>",
                    "--right-archive",
                    "<candidate_archive_path>",
                    "--right-submission-dir",
                    "<candidate_runtime_dir>",
                    "--full-frame-file-list-claim",
                    "--output-dir",
                    "<receiver_repair_dir>/shell_inflate_parity",
                ],
                "blocked_until": [
                    "source_runtime_dir",
                    "candidate_runtime_dir",
                    "full_frame_file_list",
                ],
                **FALSE_AUTHORITY,
            }
        )
    return hints


def build_frontier_receiver_repair_work_order(
    *,
    repo_root: str | Path,
    receiver_repair_backlog: Mapping[str, Any],
    repair_id: str,
) -> dict[str, Any]:
    """Build a machine-readable handoff for one receiver/exact-readiness repair."""

    repo = Path(repo_root)
    require_no_truthy_authority_fields(
        receiver_repair_backlog,
        context="receiver_repair_backlog",
    )
    row = _receiver_repair_row_by_id(receiver_repair_backlog, repair_id)
    require_no_truthy_authority_fields(row, context=f"receiver_repair_row:{repair_id}")
    bridge_details = _bridge_report_details(
        bridge_report_paths=_string_list(row.get("bridge_report_paths")),
        candidate_ids=_string_list(row.get("candidate_ids")),
        repo_root=repo,
    )
    source_backlog_generated_at = str(
        receiver_repair_backlog.get("generated_at_utc")
        or "unknown_source_backlog_generation_time"
    )
    work_order = {
        "schema": RECEIVER_REPAIR_WORK_ORDER_SCHEMA,
        "generated_at_utc": source_backlog_generated_at,
        "source_backlog_generated_at_utc": source_backlog_generated_at,
        "repair_id": repair_id,
        "repair_family": row.get("repair_family"),
        "source_operation_id": row.get("source_operation_id"),
        "target_kind": row.get("target_kind"),
        "queue_consumer": row.get("queue_consumer"),
        "source_queue_consumer": row.get("source_queue_consumer"),
        "recommended_next_action": row.get("recommended_next_action"),
        "blocker": row.get("blocker"),
        "blocker_count": row.get("blocker_count"),
        "saved_bytes_at_risk": row.get("saved_bytes_at_risk"),
        "candidate_ids": list(row.get("candidate_ids") or []),
        "bridge_report_paths": list(row.get("bridge_report_paths") or []),
        "bridge_details": bridge_details,
        "command_hints": _receiver_repair_command_hints(
            row=row,
            bridge_details=bridge_details,
        ),
        "correction_budget_context": dict(
            row.get("correction_budget_context")
            if isinstance(row.get("correction_budget_context"), Mapping)
            else {}
        ),
        "budget_spend_gate": {
            "schema": "frontier_rate_attack_receiver_repair_budget_spend_gate.v1",
            "ready_for_targeted_correction_budget_spend": False,
            "required_before_spend": [
                "runtime_consumption_proof_passed",
                "receiver_contract_satisfied",
                "component_guarded_segnet_posenet_repair_candidate_selected",
            ],
            "allowed_budget_sources": [
                "materializer_rate_positive_saved_bytes_total_after_receiver_proof",
                "local_drop_saved_bytes_total_component_measured",
            ],
            "forbidden_use": "score_claim_or_dispatch_authority",
            **FALSE_AUTHORITY,
        },
        "allowed_use": "receiver_repair_work_order_for_queue_owned_runtime_repair_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        work_order,
        context=f"receiver_repair_work_order:{repair_id}",
    )
    return work_order


def _selected_receiver_repair_rows(
    backlog: Mapping[str, Any],
    *,
    candidate_limit: int,
) -> list[Mapping[str, Any]]:
    actionable_rows = [
        row
        for row in backlog.get("rows") or []
        if isinstance(row, Mapping) and row.get("queue_actionable") is True
    ]
    selected: list[Mapping[str, Any]] = []
    selected_repair_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_groups: set[tuple[str, str, str]] = set()

    def add_row(row: Mapping[str, Any]) -> None:
        repair_id = str(row.get("repair_id") or json.dumps(row, sort_keys=True))
        if repair_id in selected_repair_ids:
            return
        selected_repair_ids.add(repair_id)
        selected.append(row)
        seen_sources.add(str(row.get("source_operation_id") or ""))
        seen_groups.add(
            (
                str(row.get("source_operation_id") or ""),
                str(row.get("repair_family") or ""),
                str(row.get("queue_consumer") or ""),
            )
        )

    bridge_backed_rows = [
        row
        for row in actionable_rows
        if _string_list(row.get("bridge_report_paths"))
        and _string_list(row.get("candidate_ids"))
    ]
    work_order_only_rows = [
        row
        for row in actionable_rows
        if row not in bridge_backed_rows
    ]

    for row in bridge_backed_rows:
        if str(row.get("repair_family") or "") != "submission_runtime_manifest_closure":
            continue
        add_row(row)
        break
    if len(selected) >= candidate_limit:
        return selected

    for row in bridge_backed_rows:
        source = str(row.get("source_operation_id") or "")
        if source in seen_sources:
            continue
        add_row(row)
        if len(selected) >= candidate_limit:
            return selected

    for row in bridge_backed_rows:
        group = (
            str(row.get("source_operation_id") or ""),
            str(row.get("repair_family") or ""),
            str(row.get("queue_consumer") or ""),
        )
        if group in seen_groups:
            continue
        add_row(row)
        if len(selected) >= candidate_limit:
            return selected

    for row in work_order_only_rows:
        source = str(row.get("source_operation_id") or "")
        if source in seen_sources:
            continue
        add_row(row)
        if len(selected) >= candidate_limit:
            return selected

    for row in work_order_only_rows:
        group = (
            str(row.get("source_operation_id") or ""),
            str(row.get("repair_family") or ""),
            str(row.get("queue_consumer") or ""),
        )
        if group in seen_groups:
            continue
        add_row(row)
        if len(selected) >= candidate_limit:
            break
    return selected


def build_frontier_receiver_repair_queue(
    *,
    repo_root: str | Path,
    receiver_repair_backlog: Mapping[str, Any],
    receiver_repair_backlog_path: str | Path,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    queue_id: str = "frontier_receiver_repair_queue",
    candidate_limit: int = 4,
    local_io_concurrency: int = 1,
) -> dict[str, Any] | None:
    """Compile top receiver repair rows into a bounded local work-order queue."""

    repo = Path(repo_root)
    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    if local_io_concurrency < 1:
        raise FrontierRateAttackFeedbackError("local_io_concurrency must be >= 1")
    require_no_truthy_authority_fields(
        receiver_repair_backlog,
        context="receiver_repair_backlog_queue_input",
    )
    selected_rows = _selected_receiver_repair_rows(
        receiver_repair_backlog,
        candidate_limit=candidate_limit,
    )
    if not selected_rows:
        return None
    backlog_path = _resolve_path(receiver_repair_backlog_path, repo_root=repo)
    results_base = _resolve_path(str(results_root), repo_root=repo)
    queue_root = results_base / "frontier_receiver_repair" / _slug_token(queue_id)
    experiments: list[dict[str, Any]] = []
    for priority, row in enumerate(selected_rows, start=1):
        repair_id = str(row.get("repair_id") or f"receiver_repair_{priority}")
        repair_dir = queue_root / _slug_token(repair_id)
        work_order_path = repair_dir / "work_order.json"
        bridge_details = _bridge_report_details(
            bridge_report_paths=_string_list(row.get("bridge_report_paths")),
            candidate_ids=_string_list(row.get("candidate_ids")),
            repo_root=repo,
        )
        receiver_runtime_steps = _receiver_repair_exact_readiness_bridge_steps(
            row=row,
            bridge_details=bridge_details,
            repair_dir=repair_dir,
            repo_root=repo,
        )
        exact_readiness_bridge_step_count = sum(
            1
            for step in receiver_runtime_steps
            if str(step.get("id") or "").startswith("run_exact_readiness_bridge")
        )
        submission_closure_step_count = sum(
            1
            for step in receiver_runtime_steps
            if str(step.get("id") or "").startswith("build_submission_runtime_closure")
        )
        experiments.append(
            {
                "id": _slug_token(repair_id),
                "status": "queued",
                "priority": priority,
                "lane_id": "lane_frontier_receiver_repair_queue_20260526",
                "tags": [
                    "receiver_repair",
                    str(row.get("repair_family") or "unknown_family"),
                    str(row.get("target_kind") or "unknown_target"),
                ],
                "metadata": {
                    "schema": RECEIVER_REPAIR_QUEUE_METADATA_SCHEMA,
                    "repair_id": repair_id,
                    "repair_family": row.get("repair_family"),
                    "source_operation_id": row.get("source_operation_id"),
                    "target_kind": row.get("target_kind"),
                    "queue_consumer": row.get("queue_consumer"),
                    "source_queue_consumer": row.get("source_queue_consumer"),
                    "candidate_ids": list(row.get("candidate_ids") or []),
                    "bridge_report_paths": list(row.get("bridge_report_paths") or []),
                    "source_queue_paths": list(
                        bridge_details.get("source_queue_paths") or []
                    ),
                    "submission_closure_step_count": submission_closure_step_count,
                    "exact_readiness_bridge_step_count": (
                        exact_readiness_bridge_step_count
                    ),
                    "saved_bytes_at_risk": row.get("saved_bytes_at_risk"),
                    "correction_budget_context": dict(
                        row.get("correction_budget_context")
                        if isinstance(row.get("correction_budget_context"), Mapping)
                        else {}
                    ),
                    "receiver_repair_required_before_budget_spend": True,
                    "allowed_use": "receiver_repair_queue_metadata_only",
                    "forbidden_use": "score_claim_or_dispatch_authority",
                    **FALSE_AUTHORITY,
                },
                "steps": [
                    {
                        "id": "emit_receiver_repair_work_order",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/build_frontier_receiver_repair_work_order.py",
                            "--receiver-repair-backlog",
                            _repo_rel(backlog_path, repo),
                            "--repair-id",
                            repair_id,
                            "--work-order-out",
                            _repo_rel(work_order_path, repo),
                            "--overwrite",
                        ],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(work_order_path, repo),
                                "key": "schema",
                                "equals": RECEIVER_REPAIR_WORK_ORDER_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(work_order_path, repo),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(work_order_path, repo),
                                "key": "budget_spend_gate.ready_for_targeted_correction_budget_spend",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [_repo_rel(work_order_path, repo)],
                            "input_artifact_paths": [
                                _repo_rel(backlog_path, repo),
                                *list(row.get("bridge_report_paths") or []),
                            ],
                            "include_postcondition_paths": True,
                        },
                    },
                    *receiver_runtime_steps,
                ],
            }
        )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": 1,
                    "local_io_heavy": local_io_concurrency,
                    "local_mlx": 0,
                    "modal_cpu": 0,
                    "modal_gpu": 0,
                },
            },
            "experiments": experiments,
            "allowed_use": "queue_owned_receiver_repair_work_orders_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
            **FALSE_AUTHORITY,
        }
    )


def _submission_closure_candidate_ids_for_source_queue(
    source_queue_path: str,
    *,
    candidate_ids: Sequence[str],
    repo_root: Path,
) -> list[str | None]:
    """Return candidates that can be statically closed from a source queue."""

    queue_path = _resolve_path(source_queue_path, repo_root=repo_root)
    if not queue_path.is_file():
        fallback = [candidate_id for candidate_id in candidate_ids if candidate_id]
        return fallback or [None]
    payload = _load_json(queue_path)
    allowed = {str(candidate_id) for candidate_id in candidate_ids if str(candidate_id)}
    out: list[str | None] = []
    seen: set[str] = set()
    for key in ("top_k", "dispatch_ready"):
        for row in payload.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            candidate_id = str(row.get("candidate_id") or "")
            if allowed and candidate_id not in allowed:
                continue
            if row.get("receiver_contract_satisfied") is not True:
                continue
            stable = candidate_id or "<single-implicit-candidate>"
            if stable in seen:
                continue
            seen.add(stable)
            out.append(candidate_id or None)
    return out


def _receiver_repair_exact_readiness_bridge_steps(
    *,
    row: Mapping[str, Any],
    bridge_details: Mapping[str, Any],
    repair_dir: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    source_queue_paths = _string_list(bridge_details.get("source_queue_paths"))
    if not source_queue_paths:
        return []
    candidate_ids = _string_list(row.get("candidate_ids"))
    requires_submission_closure = (
        str(row.get("repair_family") or "") == "submission_runtime_manifest_closure"
    )
    steps: list[dict[str, Any]] = []
    for index, source_queue_path in enumerate(source_queue_paths, start=1):
        if requires_submission_closure:
            closure_candidate_ids = _submission_closure_candidate_ids_for_source_queue(
                source_queue_path,
                candidate_ids=candidate_ids,
                repo_root=repo_root,
            )
            if not closure_candidate_ids:
                continue
        elif len(source_queue_paths) == 1 and len(candidate_ids) == 1:
            closure_candidate_ids = [candidate_ids[0]]
        else:
            closure_candidate_ids = [None]
        for candidate_index, closure_candidate_id in enumerate(
            closure_candidate_ids,
            start=1,
        ):
            multi_source_suffix = "" if len(source_queue_paths) == 1 else f"_{index}"
            multi_candidate_suffix = (
                ""
                if len(closure_candidate_ids) == 1
                else f"_{candidate_index}_{_slug_token(closure_candidate_id)}"
            )
            suffix = f"{multi_source_suffix}{multi_candidate_suffix}"
            bridge_source_queue_path = source_queue_path
            step_requires = ["emit_receiver_repair_work_order"]
            if requires_submission_closure:
                closure_dir = repair_dir / f"submission_closure{suffix}"
                closure_submission_dir = closure_dir / "submission"
                closed_source_queue_path = closure_dir / "closed_source_queue.json"
                closure_report_path = closure_dir / "submission_closure_report.json"
                closure_command = [
                    ".venv/bin/python",
                    MATERIALIZER_SUBMISSION_CLOSURE_TOOL,
                    "--source-queue",
                    source_queue_path,
                    "--submission-dir-out",
                    _repo_rel(closure_submission_dir, repo_root),
                    "--closed-source-queue-out",
                    _repo_rel(closed_source_queue_path, repo_root),
                    "--closure-report-out",
                    _repo_rel(closure_report_path, repo_root),
                    "--overwrite",
                ]
                if closure_candidate_id:
                    closure_command.extend(["--candidate-id", closure_candidate_id])
                closure_step_id = f"build_submission_runtime_closure{suffix}"
                steps.append(
                    {
                        "id": closure_step_id,
                        "kind": "command",
                        "command": closure_command,
                        "requires": ["emit_receiver_repair_work_order"],
                        "resources": {"kind": "local_io_heavy"},
                        "timeout_seconds": 180,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(closure_report_path, repo_root),
                                "key": "schema",
                                "equals": MATERIALIZER_SUBMISSION_CLOSURE_REPORT_SCHEMA,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(closure_report_path, repo_root),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(closed_source_queue_path, repo_root),
                                "key": "schema",
                                "equals": "optimizer_candidate_queue_v1",
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(closed_source_queue_path, repo_root),
                                "false_or_missing": list(
                                    QUEUE_FALSE_AUTHORITY_FALSE_OR_MISSING_FIELDS
                                ),
                            },
                            {
                                "type": "json_equals",
                                "path": _repo_rel(closure_report_path, repo_root),
                                "key": "ready_for_exact_eval_dispatch",
                                "equals": False,
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(closure_report_path, repo_root),
                                _repo_rel(closed_source_queue_path, repo_root),
                                _repo_rel(closure_submission_dir, repo_root),
                            ],
                            "input_artifact_paths": [
                                source_queue_path,
                                *list(row.get("bridge_report_paths") or []),
                            ],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    }
                )
                bridge_source_queue_path = _repo_rel(closed_source_queue_path, repo_root)
                step_requires = ["emit_receiver_repair_work_order", closure_step_id]
            bridge_dir = repair_dir / f"exact_readiness_bridge{suffix}"
            bridge_report_path = bridge_dir / "exact_readiness_bridge_report.json"
            readiness_dir = bridge_dir / "exact_readiness"
            command = [
                ".venv/bin/python",
                MATERIALIZER_EXACT_READINESS_BRIDGE_TOOL,
                "--source-queue",
                bridge_source_queue_path,
                "--exact-readiness-out-dir",
                _repo_rel(readiness_dir, repo_root),
                "--bridge-report-out",
                _repo_rel(bridge_report_path, repo_root),
                "--overwrite",
                "--force-recompute",
            ]
            bridge_candidate_ids = (
                [closure_candidate_id]
                if requires_submission_closure and closure_candidate_id
                else ([] if requires_submission_closure else candidate_ids)
            )
            for candidate_id in bridge_candidate_ids:
                if candidate_id:
                    command.extend(["--candidate-id", candidate_id])
            steps.append(
                {
                    "id": f"run_exact_readiness_bridge{suffix}",
                    "kind": "command",
                    "command": command,
                    "requires": step_requires,
                    "resources": {"kind": "local_cpu"},
                    "timeout_seconds": 120,
                    "postconditions": [
                        {
                            "type": "json_equals",
                            "path": _repo_rel(bridge_report_path, repo_root),
                            "key": "schema",
                            "equals": MATERIALIZER_EXACT_READINESS_BRIDGE_SCHEMA,
                        },
                        {
                            "type": "json_false_authority",
                            "path": _repo_rel(bridge_report_path, repo_root),
                        },
                        {
                            "type": "json_equals",
                            "path": _repo_rel(bridge_report_path, repo_root),
                            "key": "ready_for_exact_eval_dispatch",
                            "equals": False,
                        },
                    ],
                    "telemetry": {
                        "artifact_paths": [
                            _repo_rel(bridge_report_path, repo_root),
                            _repo_rel(readiness_dir, repo_root),
                        ],
                        "input_artifact_paths": [
                            bridge_source_queue_path,
                            *list(row.get("bridge_report_paths") or []),
                        ],
                        "recursive": True,
                        "include_postcondition_paths": True,
                    },
                }
            )
    return steps


def _queue_summary(queue: Mapping[str, Any]) -> dict[str, Any]:
    experiments = queue.get("experiments")
    experiment_rows = experiments if isinstance(experiments, list) else []
    return {
        "queue_id": queue.get("queue_id"),
        "experiment_count": len(experiment_rows),
        "step_count": sum(
            len(exp.get("steps", []))
            for exp in experiment_rows
            if isinstance(exp, Mapping)
        ),
        "selected_candidate_ids": [
            str(exp.get("id"))
            for exp in experiment_rows
            if isinstance(exp, Mapping) and exp.get("id")
        ],
        **FALSE_AUTHORITY,
    }


def build_frontier_rate_attack_feedback_refresh(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    local_cpu_eureka_roots: Sequence[str | Path] = (),
    materializer_feedback_paths: Sequence[str | Path] = (),
    pair_frame_geometry_paths: Sequence[str | Path] = (),
    dqs1_observation_paths: Sequence[str | Path] = (),
    max_files_per_root: int = 4096,
    action_summary_path: str | Path | None = None,
    results_root: str = DEFAULT_RESULTS_ROOT,
    queue_id: str = DEFAULT_QUEUE_ID,
    candidate_limit: int = 4,
    skip_observed_dqs1_candidates: bool = True,
    local_cpu_concurrency: int = 1,
    local_io_concurrency: int = 1,
    include_raw_retention_plan: bool = True,
    raw_retention_execute: bool = False,
    raw_retention_action: str = "move",
    raw_retention_cold_store_roots: Sequence[str] = (),
    raw_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    include_mlx_retention_plan: bool = True,
    mlx_retention_execute: bool = False,
    mlx_retention_action: str = "move",
    mlx_retention_cold_store_roots: Sequence[str] = (),
    mlx_retention_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
    repair_palette_modes: Sequence[str] = (),
    repair_dynamics_palette_priors: Sequence[Mapping[str, Any]] = (),
    repair_dynamics_prior_source_paths: Sequence[str] = (),
    component_response_cache_roots: Sequence[str | Path] = (),
    receiver_closed_rate_packet_paths: Sequence[str | Path] = (),
    receiver_closed_rate_parent_paths: Sequence[str | Path] = (),
    target_profile_id: str = "contest_video_0",
    target_mode: str = "contest_video_overfit",
    target_video_paths: Sequence[str | Path] = (),
    target_corpus_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a forest-level feedback refresh and optional DQS1 follow-up queue."""

    repo = Path(repo_root)
    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    if max_files_per_root < 1:
        raise FrontierRateAttackFeedbackError("max_files_per_root must be >= 1")
    target_optimization_profile = build_frontier_target_optimization_profile(
        repo_root=repo,
        target_profile_id=target_profile_id,
        target_mode=target_mode,
        target_video_paths=target_video_paths,
        target_corpus_manifest_path=target_corpus_manifest_path,
    )
    target_profile_metadata = target_optimization_profile_queue_metadata(
        target_optimization_profile
    )
    payloads, source_paths, discovery = discover_materializer_feedback_payloads(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
        materializer_feedback_paths=materializer_feedback_paths,
        max_files_per_root=max_files_per_root,
    )
    pair_frame_requests, pair_frame_source_paths, pair_frame_discovery = (
        discover_pair_frame_geometry_queue_requests(
            repo_root=repo,
            frontier_artifact_roots=frontier_artifact_roots,
            pair_frame_geometry_paths=pair_frame_geometry_paths,
            max_files_per_root=max_files_per_root,
        )
    )
    default_eureka_root = repo / ".omx" / "research"
    resolved_eureka_roots: tuple[str | Path, ...] = tuple(
        local_cpu_eureka_roots
        or frontier_artifact_roots
        or ((default_eureka_root,) if default_eureka_root.exists() else ())
    )
    strict_eureka_authority = bool(local_cpu_eureka_roots) and all(
        _resolve_path(root, repo_root=repo).is_file() for root in local_cpu_eureka_roots
    )
    eureka_planning = discover_local_cpu_eureka_planning_signals(
        repo_root=repo,
        frontier_artifact_roots=resolved_eureka_roots,
        max_files_per_root=max_files_per_root,
        strict_authority=strict_eureka_authority,
    )
    dqs1_observation_discovery = discover_dqs1_observation_jsonl_paths(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
    )
    discovered_dqs1_observation_paths = tuple(
        str(path)
        for path in dqs1_observation_discovery.get(
            "discovered_observation_jsonl_paths",
            (),
        )
    )
    dqs1_observations, dqs1_source_paths = load_dqs1_observations(
        repo_root=repo,
        observation_paths=(
            *dqs1_observation_paths,
            *discovered_dqs1_observation_paths,
        ),
    )
    receiver_closed_correction_budget = build_receiver_closed_correction_budget(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
        results_root=results_root,
        receiver_closed_rate_packet_paths=receiver_closed_rate_packet_paths,
        receiver_closed_rate_parent_paths=receiver_closed_rate_parent_paths,
    )
    manual_repair_dynamics_palette_prior = _repair_dynamics_palette_prior(
        repair_palette_modes,
        source="frontier_rate_attack_feedback_refresh_repair_palette_modes",
    )
    supplied_repair_dynamics_priors: list[Mapping[str, Any]] = [
        prior
        for prior in repair_dynamics_palette_priors
        if isinstance(prior, Mapping) and prior
    ]
    for index, prior in enumerate(supplied_repair_dynamics_priors):
        require_no_truthy_authority_fields(
            prior,
            context=f"frontier_feedback_refresh_repair_dynamics_palette_prior:{index}",
        )
    if manual_repair_dynamics_palette_prior:
        supplied_repair_dynamics_priors.append(manual_repair_dynamics_palette_prior)
    repair_dynamics_palette_prior = _aggregate_repair_dynamics_priors(
        supplied_repair_dynamics_priors,
        source="frontier_rate_attack_feedback_refresh_repair_dynamics_prior_aggregate",
    )
    operation_portfolio = build_frontier_operation_portfolio(
        repo_root=repo,
        materializer_feedback_payloads=payloads,
        materializer_feedback_source_paths=source_paths,
        dqs1_observations=dqs1_observations,
        eureka_planning=eureka_planning,
        pair_frame_requests=pair_frame_requests,
        pair_frame_source_paths=pair_frame_source_paths,
        receiver_closed_correction_budget=receiver_closed_correction_budget,
    )
    receiver_repair_backlog = build_frontier_receiver_repair_backlog(
        operation_portfolio
    )
    operation_materializer_bridge = build_frontier_operation_materializer_bridge(
        repo_root=repo,
        operation_portfolio=operation_portfolio,
        default_output_root=(
            Path(results_root)
            / "frontier_operation_materializer"
            / _slug_token(queue_id)
        ),
        candidate_limit=candidate_limit,
    )
    targeted_component_correction_acquisition = (
        build_frontier_targeted_component_correction_acquisition(
            operation_portfolio=operation_portfolio,
            receiver_closed_correction_budget=receiver_closed_correction_budget,
            repair_dynamics_palette_prior=repair_dynamics_palette_prior,
        )
    )
    queue_payload: dict[str, Any] | None = None
    bridge: dict[str, Any] | None = None
    selected_pairset_acquisition: dict[str, Any] | None = None
    selected_candidate_ids: list[str] = []
    if action_summary_path is not None:
        result = build_queue_from_action_summary(
            _resolve_path(action_summary_path, repo_root=repo),
            repo_root=repo,
            results_root=results_root,
            queue_id=queue_id,
            candidate_limit=candidate_limit,
            materializer_feedback_payloads=payloads,
            materializer_feedback_source_paths=source_paths,
            dqs1_observations=dqs1_observations,
            dqs1_observation_source_paths=dqs1_source_paths,
            skip_observed_dqs1_candidates=skip_observed_dqs1_candidates,
            additional_queue_requests=pair_frame_requests,
            additional_queue_request_source_paths=pair_frame_source_paths,
            local_cpu_concurrency=local_cpu_concurrency,
            local_io_concurrency=local_io_concurrency,
            include_raw_retention_plan=include_raw_retention_plan,
            raw_retention_execute=raw_retention_execute,
            raw_retention_action=raw_retention_action,
            raw_retention_cold_store_roots=tuple(raw_retention_cold_store_roots),
            raw_retention_cold_store_reserve_gb=raw_retention_cold_store_reserve_gb,
            include_mlx_retention_plan=include_mlx_retention_plan,
            mlx_retention_execute=mlx_retention_execute,
            mlx_retention_action=mlx_retention_action,
            mlx_retention_cold_store_roots=tuple(mlx_retention_cold_store_roots),
            mlx_retention_cold_store_reserve_gb=mlx_retention_cold_store_reserve_gb,
        )
        queue_payload = normalize_queue_definition(result.queue)
        if eureka_planning.get("active") is True:
            for experiment in queue_payload.get("experiments", []):
                if not isinstance(experiment, dict):
                    continue
                metadata = experiment.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["frontier_feedback_eureka_planning"] = eureka_planning
        for experiment in queue_payload.get("experiments", []):
            if not isinstance(experiment, dict):
                continue
            metadata = experiment.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["frontier_target_optimization_profile"] = (
                    target_profile_metadata
                )
        if operation_portfolio.get("operation_count"):
            portfolio_metadata = _operation_portfolio_queue_metadata(
                operation_portfolio
            )
            receiver_repair_metadata = _receiver_repair_backlog_queue_metadata(
                receiver_repair_backlog
            )
            receiver_closed_metadata = _receiver_closed_correction_budget_queue_metadata(
                receiver_closed_correction_budget
            )
            targeted_component_metadata = (
                _targeted_component_correction_queue_metadata(
                    targeted_component_correction_acquisition
                )
            )
            for experiment in queue_payload.get("experiments", []):
                if not isinstance(experiment, dict):
                    continue
                metadata = experiment.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["frontier_operation_portfolio"] = portfolio_metadata
                    metadata["frontier_receiver_repair_backlog"] = (
                        receiver_repair_metadata
                    )
                    metadata["frontier_receiver_closed_correction_budget"] = (
                        receiver_closed_metadata
                    )
                    metadata["frontier_targeted_component_correction_acquisition"] = (
                        targeted_component_metadata
                    )
        bridge = result.materializer_feedback_bridge
        selected_pairset_acquisition = result.selected_pairset_acquisition
        selected_candidate_ids = [selection.candidate_id for selection in result.selections]
    else:
        try:
            bridge = build_dqs1_materializer_feedback_bridge(
                materializer_feedback_payloads=payloads,
                materializer_feedback_source_paths=source_paths,
                candidate_limit=candidate_limit,
                dqs1_observations=dqs1_observations,
                dqs1_observation_source_paths=dqs1_source_paths,
            )
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(str(exc)) from exc

    report = {
        "schema": FEEDBACK_REFRESH_SCHEMA,
        "generated_at_utc": _utc_now(),
        "candidate_limit": candidate_limit,
        "target_optimization_profile": target_optimization_profile,
        "target_optimization_profile_metadata": target_profile_metadata,
        "discovery": discovery,
        "pair_frame_geometry_discovery": pair_frame_discovery,
        "pair_frame_geometry_request_source_paths": list(pair_frame_source_paths),
        "pair_frame_geometry_queue_request_count": len(pair_frame_requests),
        "local_cpu_eureka_planning": eureka_planning,
        "local_cpu_eureka_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in resolved_eureka_roots
        ],
        "dqs1_observation_discovery": dqs1_observation_discovery,
        "operation_portfolio": operation_portfolio,
        "rate_budget_preservation_plan": operation_portfolio.get(
            "rate_budget_preservation_plan"
        ),
        "operation_materializer_bridge": operation_materializer_bridge,
        "receiver_repair_backlog": receiver_repair_backlog,
        "receiver_closed_correction_budget": receiver_closed_correction_budget,
        "receiver_closed_rate_packet_manifest_paths": [
            _repo_rel(_resolve_path(path, repo_root=repo), repo)
            for path in receiver_closed_rate_packet_paths
        ],
        "receiver_closed_rate_parent_manifest_paths": [
            _repo_rel(_resolve_path(path, repo_root=repo), repo)
            for path in receiver_closed_rate_parent_paths
        ],
        "repair_dynamics_palette_prior": repair_dynamics_palette_prior,
        "repair_dynamics_prior_source_paths": list(repair_dynamics_prior_source_paths),
        "manual_repair_dynamics_palette_prior": manual_repair_dynamics_palette_prior,
        "component_response_cache_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in component_response_cache_roots
        ],
        "targeted_component_correction_acquisition": (
            targeted_component_correction_acquisition
        ),
        "materializer_feedback_source_paths": list(source_paths),
        "materializer_feedback_payload_count": len(payloads),
        "dqs1_observation_source_paths": list(dqs1_source_paths),
        "dqs1_observation_count": len(dqs1_observations),
        "action_summary_path": (
            None
            if action_summary_path is None
            else _repo_rel(_resolve_path(action_summary_path, repo_root=repo), repo)
        ),
        "queue_id": queue_id,
        "results_root": results_root,
        "selected_candidate_ids": selected_candidate_ids,
        "selected_pairset_acquisition": selected_pairset_acquisition,
        "materializer_feedback_bridge": bridge,
        "queue_summary": None if queue_payload is None else _queue_summary(queue_payload),
        "queue": queue_payload,
        "retention_policy": {
            "schema": "frontier_rate_attack_feedback_retention_policy.v1",
            "raw_retention_plan_included": include_raw_retention_plan,
            "raw_retention_execute": raw_retention_execute,
            "raw_retention_action": raw_retention_action,
            "raw_retention_cold_store_roots": list(raw_retention_cold_store_roots),
            "raw_retention_cold_store_reserve_gb": raw_retention_cold_store_reserve_gb,
            "mlx_retention_plan_included": include_mlx_retention_plan,
            "mlx_retention_execute": mlx_retention_execute,
            "mlx_retention_action": mlx_retention_action,
            "mlx_retention_cold_store_roots": list(mlx_retention_cold_store_roots),
            "mlx_retention_cold_store_reserve_gb": mlx_retention_cold_store_reserve_gb,
            **FALSE_AUTHORITY,
        },
        "allowed_use": "queue_owned_frontier_feedback_replanning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    attach_frontier_autonomous_chain_optimization(report, update_queue_metadata=True)
    return report


__all__ = [
    "AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA",
    "AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA",
    "AUTONOMOUS_CHAIN_QUEUE_METADATA_SCHEMA",
    "AUTONOMOUS_CHAIN_WORK_ORDER_SCHEMA",
    "DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA",
    "DQS1_OBSERVATION_DISCOVERY_SCHEMA",
    "FEEDBACK_REFRESH_SCHEMA",
    "FRONTIER_RATE_ATTACK_FEEDBACK_REFRESH_SCHEMA",
    "LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA",
    "LOCAL_CPU_EUREKA_PAIRSET_PROFILE_SCHEMA",
    "LOCAL_CPU_EUREKA_PLANNER_HINT_SCHEMA",
    "MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA",
    "OPERATION_MATERIALIZER_BRIDGE_SCHEMA",
    "OPERATION_PORTFOLIO_SCHEMA",
    "OPERATION_PORTFOLIO_TAXONOMY_SCHEMA",
    "OPERATOR_ACTION_LEDGER_SCHEMA",
    "OPERATOR_ACTION_TERM_SCHEMA",
    "PAIR_FRAME_GEOMETRY_DISCOVERY_SCHEMA",
    "RATE_BUDGET_PRESERVATION_PLAN_SCHEMA",
    "RATE_BUDGET_PRESERVATION_ROW_SCHEMA",
    "RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA",
    "RECEIVER_CLOSED_RATE_PACKET_SIGNAL_SCHEMA",
    "RECEIVER_REPAIR_BACKLOG_SCHEMA",
    "RECEIVER_REPAIR_QUEUE_METADATA_SCHEMA",
    "RECEIVER_REPAIR_ROW_SCHEMA",
    "RECEIVER_REPAIR_WORK_ORDER_SCHEMA",
    "REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFESTS_SCHEMA",
    "REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFEST_SCHEMA",
    "REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA",
    "REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA",
    "REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA",
    "REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA",
    "REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA",
    "REPAIR_BUDGET_MATERIALIZER_BINDING_ROW_SCHEMA",
    "REPAIR_BUDGET_TYPED_RESPONSE_LEDGER_SCHEMA",
    "REPAIR_BUDGET_TYPED_RESPONSE_ROW_SCHEMA",
    "REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA",
    "REPAIR_BUDGET_WATERFILL_QUEUE_METADATA_SCHEMA",
    "REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA",
    "REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_CHAIN_MATERIALIZER_HANDOFF_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_QUEUE_METADATA_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_QUEUE_METADATA_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_RESPONSE_ROW_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_WORK_ORDER_SCHEMA",
    "TARGETED_DROP_MANY_STAGE_INPUTS_SCHEMA",
    "TARGET_OPTIMIZATION_PROFILE_METADATA_SCHEMA",
    "TARGET_OPTIMIZATION_PROFILE_METADATA_SCHEMAS",
    "TARGET_OPTIMIZATION_PROFILE_QUEUE_METADATA_SCHEMA",
    "TARGET_OPTIMIZATION_PROFILE_SCHEMA",
    "FrontierRateAttackFeedbackError",
    "attach_frontier_autonomous_chain_optimization",
    "build_frontier_autonomous_chain_optimization",
    "build_frontier_autonomous_chain_optimization_queue",
    "build_frontier_autonomous_chain_work_order",
    "build_frontier_byte_range_stage_inputs",
    "build_frontier_materializer_execution_queue_if_available",
    "build_frontier_operation_materializer_bridge",
    "build_frontier_operation_portfolio",
    "build_frontier_rate_attack_feedback_refresh",
    "build_frontier_rate_budget_preservation_plan",
    "build_frontier_receiver_repair_backlog",
    "build_frontier_receiver_repair_queue",
    "build_frontier_receiver_repair_work_order",
    "build_frontier_repair_budget_child_component_replay_manifests",
    "build_frontier_repair_budget_materialization_execution_report",
    "build_frontier_repair_budget_materialization_plan",
    "build_frontier_repair_budget_materializer_binding_report",
    "build_frontier_repair_budget_waterfill_queue",
    "build_frontier_repair_budget_waterfill_work_order",
    "build_frontier_target_optimization_profile",
    "build_frontier_targeted_component_correction_acquisition",
    "build_frontier_targeted_component_correction_chain_materializer_handoff",
    "build_frontier_targeted_component_correction_chain_work_orders",
    "build_frontier_targeted_component_correction_materialization_queue",
    "build_frontier_targeted_component_correction_materialization_request",
    "build_frontier_targeted_component_correction_materialization_requests",
    "build_frontier_targeted_component_correction_queue",
    "build_frontier_targeted_component_correction_response_harvest",
    "build_frontier_targeted_component_correction_response_harvest_from_artifacts",
    "build_frontier_targeted_component_correction_work_order",
    "build_frontier_targeted_drop_many_stage_inputs",
    "build_receiver_closed_correction_budget",
    "discover_dqs1_observation_jsonl_paths",
    "discover_local_cpu_eureka_planning_signals",
    "discover_materializer_feedback_payloads",
    "discover_pair_frame_geometry_queue_requests",
    "load_dqs1_observations",
]
