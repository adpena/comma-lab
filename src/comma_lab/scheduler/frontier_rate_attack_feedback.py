# SPDX-License-Identifier: MIT
"""Compile frontier materializer feedback into queue-owned follow-up surfaces."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.storage_tiers import DEFAULT_RESERVE_FREE_GB
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

from .byte_shaving_campaign_queue import (
    MATERIALIZER_BACKLOG_SCHEMA,
    build_materializer_work_queue,
    materializer_contexts_from_payload,
)
from .byte_shaving_materializer_registry import registry_manifest
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
MATERIALIZER_EXACT_READINESS_BRIDGE_SCHEMA = (
    "materializer_chain_exact_readiness_bridge_report.v1"
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
BYTE_RANGE_CHAIN_TOOL = "tools/run_byte_range_entropy_recode_chain.py"
MATERIALIZER_CHAIN_HARVEST_REPORT_SCHEMA = "materializer_chain_harvest_report.v1"
MATERIALIZER_SUBMISSION_CLOSURE_REPORT_SCHEMA = (
    "materializer_submission_runtime_closure_report.v1"
)
RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA = (
    "frontier_rate_attack_receiver_closed_correction_budget.v1"
)
RECEIVER_CLOSED_CORRECTION_BUDGET_QUEUE_METADATA_SCHEMA = (
    "frontier_rate_attack_receiver_closed_correction_budget_queue_metadata.v1"
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
        for row in payload.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            for blocker in _string_list(row.get("blockers")):
                row_blockers.append(blocker)
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        for blocker in _string_list(payload.get("dispatch_blockers")):
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
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
                "candidate_ids": [
                    str(row.get("candidate_id") or "")
                    for row in payload.get("rows") or []
                    if isinstance(row, Mapping) and row.get("candidate_id")
                ],
                "row_blockers_sample": _unique_strings(row_blockers)[:12],
                **FALSE_AUTHORITY,
            }
        )
    ready_count = sum(int(report["ready_candidate_count"]) for report in reports)
    candidate_count = sum(int(report["candidate_count"]) for report in reports)
    blocked_count = sum(int(report["blocked_candidate_count"]) for report in reports)
    return {
        "schema": "frontier_rate_attack_materializer_exact_readiness_bridge_summary.v1",
        "bridge_report_count": len(reports),
        "candidate_count": candidate_count,
        "ready_candidate_count": ready_count,
        "blocked_candidate_count": blocked_count,
        "missing_bridge_report_paths": missing_paths,
        "invalid_bridge_report_paths": invalid_paths,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "top_blockers": [
            blocker
            for blocker, _count in sorted(
                blocker_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:16]
        ],
        "reports": reports,
        "ready_for_chain_exact_readiness": bool(candidate_count and ready_count == candidate_count),
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
    return {
        "schema": "frontier_rate_attack_receiver_closed_correction_budget_row.v1",
        "candidate_id": candidate_id,
        "target_kind": target_kind,
        "archive_sha256": closure.get("archive_sha256"),
        "archive_bytes": closure.get("archive_bytes"),
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


def build_receiver_closed_correction_budget(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any]:
    """Harvest receiver-closed materializer bytes into repair-budget signal."""

    repo = Path(repo_root)
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

    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            str(row.get("target_kind") or ""),
            str(row.get("candidate_id") or ""),
            str(row.get("closure_report_path") or ""),
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
        "deduped_closure_report_count": len(rows),
        "duplicate_closure_report_count": duplicate_count,
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
        if exact_readiness_bridge["bridge_report_count"]:
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
    if exact_bridge_summaries and exact_ready_candidates < exact_candidate_count:
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
                    "ready_candidate_count": exact_ready_candidates,
                    "blocked_candidate_count": sum(
                        int(summary.get("blocked_candidate_count") or 0)
                        for summary in exact_bridge_summaries
                    ),
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
    return hints


def _slug_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in out.split("_") if part) or "unknown"


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


def _targeted_component_prior_status(
    *,
    seed: Mapping[str, Any],
    component_summary: Mapping[str, Any],
    master_gradient: Mapping[str, Any],
    receiver_closed_correction_budget: Mapping[str, Any],
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


def build_frontier_targeted_component_correction_acquisition(
    *,
    operation_portfolio: Mapping[str, Any],
    receiver_closed_correction_budget: Mapping[str, Any],
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
    rows: list[dict[str, Any]] = []
    for budget_row in receiver_rows:
        saved_bytes = _finite_int_or_none(budget_row.get("saved_bytes_at_risk")) or 0
        if saved_bytes <= 0:
            continue
        candidate_id = str(budget_row.get("candidate_id") or "unknown_candidate")
        target_kind = str(budget_row.get("target_kind") or "unknown_target")
        submission_dir = str(budget_row.get("submission_dir") or "")
        closure_report_path = str(budget_row.get("closure_report_path") or "")
        bridge_report_path = str(
            budget_row.get("paired_exact_readiness_bridge_report_path") or ""
        )
        rate_credit = _rate_credit_score_units_for_saved_bytes(saved_bytes)
        for seed in _TARGETED_COMPONENT_CORRECTION_FAMILY_SEEDS:
            prior_status = _targeted_component_prior_status(
                seed=seed,
                component_summary=component_summary,
                master_gradient=master_gradient,
                receiver_closed_correction_budget=receiver_closed_correction_budget,
            )
            budget_spend_blockers = [
                "candidate_specific_local_cpu_component_eval_required_before_budget_spend",
                "candidate_specific_mlx_or_exact_axis_component_response_required_before_spend",
                "exact_auth_eval_required_before_score_or_promotion_claim",
            ]
            if not submission_dir:
                budget_spend_blockers.append("submission_dir_missing_for_component_eval")
            if budget_row.get("active_rate_floor_blocked") is True:
                budget_spend_blockers.append(
                    "active_rate_floor_override_required_before_exact_dispatch"
                )
            if component_summary.get("active") is not True:
                budget_spend_blockers.append(
                    "segnet_posenet_component_behavior_rows_required_before_allocation"
                )
            family = str(seed.get("correction_family") or "unknown_correction_family")
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
                    "estimated_rate_credit_score_units": rate_credit,
                    "estimated_rate_credit_byte_delta": -saved_bytes,
                    "receiver_closed_budget_gate": budget_row.get(
                        "correction_budget_gate"
                    ),
                    "ready_for_budget_spend": False,
                    "budget_spend_allowed": False,
                    "submission_dir": submission_dir or None,
                    "archive_path": f"{submission_dir}/archive.zip" if submission_dir else None,
                    "inflate_sh_path": f"{submission_dir}/inflate.sh" if submission_dir else None,
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
                    "queue_actionable": bool(submission_dir),
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
        "top_acquisition_ids": list(acquisition.get("top_acquisition_ids") or []),
        "top_correction_families": list(acquisition.get("top_correction_families") or []),
        "blockers": list(acquisition.get("blockers") or []),
        "allowed_use": "queue_metadata_pointer_to_targeted_component_correction_acquisition",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }

def build_frontier_targeted_component_correction_work_order(
    *,
    targeted_component_correction_acquisition: Mapping[str, Any],
    acquisition_id: str,
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
        "sibling_correction_families": _unique_strings(
            sibling.get("correction_family") for sibling in sibling_rows
        ),
        "saved_bytes_budget": row.get("saved_bytes_budget"),
        "estimated_rate_credit_score_units": row.get(
            "estimated_rate_credit_score_units"
        ),
        "submission_dir": row.get("submission_dir"),
        "archive_path": row.get("archive_path"),
        "inflate_sh_path": row.get("inflate_sh_path"),
        "closure_report_path": row.get("closure_report_path"),
        "paired_exact_readiness_bridge_report_path": row.get(
            "paired_exact_readiness_bridge_report_path"
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


def build_frontier_targeted_component_correction_response_harvest_from_artifacts(
    *,
    work_order: Mapping[str, Any],
    local_cpu_advisory: Mapping[str, Any] | None = None,
    local_mlx_response: Mapping[str, Any] | None = None,
    work_order_path: str | Path | None = None,
    local_cpu_advisory_path: str | Path | None = None,
    local_mlx_response_path: str | Path | None = None,
    response_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Harvest one component-correction response into a false-authority row."""

    require_no_truthy_authority_fields(
        work_order,
        context="targeted_component_correction_response_work_order",
    )
    if local_cpu_advisory is not None:
        require_no_truthy_authority_fields(
            local_cpu_advisory,
            context="targeted_component_correction_local_cpu_advisory",
        )
    if local_mlx_response is not None:
        require_no_truthy_authority_fields(
            local_mlx_response,
            context="targeted_component_correction_local_mlx_response",
        )

    acquisition_id = str(work_order.get("acquisition_id") or "unknown_acquisition")
    candidate_id = str(work_order.get("candidate_id") or "unknown_candidate")
    correction_family = str(
        work_order.get("correction_family") or "unknown_correction_family"
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
    if local_cpu_advisory is None:
        blockers.append("local_cpu_component_advisory_missing")
    else:
        local_axis = str(local_cpu_advisory.get("score_axis") or "")
        if local_axis != "cpu_advisory":
            blockers.append("local_cpu_component_advisory_axis_not_cpu_advisory")
        local_terms = _targeted_component_response_delta_terms(local_cpu_advisory)

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
    if local_mlx_response is None:
        blockers.append("local_mlx_component_response_missing_for_spend_filter")
    else:
        mlx_axis = str(local_mlx_response.get("score_axis") or "")
        if mlx_axis != "[macOS-MLX research-signal]":
            blockers.append("local_mlx_component_response_axis_not_research_signal")
        mlx_terms = _targeted_component_response_delta_terms(local_mlx_response)
        if (
            mlx_terms["segnet_delta_score_units"] is None
            or mlx_terms["posenet_delta_score_units"] is None
        ):
            blockers.append("local_mlx_component_delta_missing")

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
        "work_order_path": None if work_order_path is None else str(work_order_path),
        "local_cpu_advisory_path": (
            None if local_cpu_advisory_path is None else str(local_cpu_advisory_path)
        ),
        "local_mlx_response_path": (
            None if local_mlx_response_path is None else str(local_mlx_response_path)
        ),
        "response_artifact_path": (
            None if response_artifact_path is None else str(response_artifact_path)
        ),
        "local_cpu_score_axis": local_axis,
        "local_mlx_score_axis": mlx_axis,
        "saved_bytes_budget": saved_bytes,
        "estimated_receiver_closed_rate_credit_score_units": rate_credit,
        "operation_levels": list(work_order.get("operation_levels") or []),
        "targeted_dimensions": list(work_order.get("targeted_dimensions") or []),
        "sibling_correction_families": list(
            work_order.get("sibling_correction_families") or []
        ),
        "local_cpu_component_terms": local_terms,
        "local_mlx_component_terms": mlx_terms,
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
            if not work_order_path_text:
                continue
            work_order_path = _resolve_path(work_order_path_text, repo_root=repo_root)
            local_cpu_path = (
                _resolve_path(local_cpu_path_text, repo_root=repo_root)
                if local_cpu_path_text
                else None
            )
            local_mlx_path = (
                _resolve_path(local_mlx_path_text, repo_root=repo_root)
                if local_mlx_path_text
                else None
            )
            response_path = (
                _resolve_path(response_path_text, repo_root=repo_root)
                if response_path_text
                else None
            )
            missing = [
                _repo_rel(path, repo_root)
                for path in (work_order_path, local_cpu_path)
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
                    "operation_levels": list(request.get("operation_levels") or []),
                    "targeted_dimensions": list(
                        request.get("targeted_dimensions") or []
                    ),
                    "work_order_path": work_order_path_text or None,
                    "local_cpu_advisory_path": local_cpu_path_text or None,
                    "local_mlx_response_path": local_mlx_path_text or None,
                    "response_artifact_path": response_path_text or None,
                    "saved_bytes_budget": request.get("saved_bytes_budget")
                    or metadata.get("saved_bytes_budget"),
                    "estimated_receiver_closed_rate_credit_score_units": request.get(
                        "estimated_rate_credit_score_units"
                    )
                    or metadata.get("estimated_rate_credit_score_units"),
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
            local_mlx = (
                _load_json(local_mlx_path)
                if local_mlx_path is not None and local_mlx_path.exists()
                else None
            )
            rows.append(
                build_frontier_targeted_component_correction_response_harvest_from_artifacts(
                    work_order=work_order,
                    local_cpu_advisory=local_cpu,
                    local_mlx_response=local_mlx,
                    work_order_path=_repo_rel(work_order_path, repo_root),
                    local_cpu_advisory_path=(
                        None
                        if local_cpu_path is None
                        else _repo_rel(local_cpu_path, repo_root)
                    ),
                    local_mlx_response_path=(
                        None
                        if local_mlx_path is None
                        else _repo_rel(local_mlx_path, repo_root)
                    ),
                    response_artifact_path=(
                        None
                        if response_path is None
                        else _repo_rel(response_path, repo_root)
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
    if targeted_component_correction_queue is not None:
        require_no_truthy_authority_fields(
            targeted_component_correction_queue,
            context="targeted_component_correction_response_queue_input",
        )
        rows.extend(
            _targeted_component_response_rows_from_queue(
                repo_root=repo,
                queue=targeted_component_correction_queue,
            )
        )
    if not rows:
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
    blockers = ["exact_auth_eval_required_before_score_or_promotion_claim"]
    if not rows:
        blockers.append("no_targeted_component_correction_response_rows")
    if blocked:
        blockers.append("response_rows_blocked_before_budget_spend")
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
            else "run_targeted_component_correction_queue_until_response_rows_exist"
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
    seed = _targeted_component_family_seed(family)
    return {
        "schema": (
            "frontier_rate_attack_targeted_component_correction_materializer_"
            "basis_entry.v1"
        ),
        "source_acquisition_id": row.get("acquisition_id"),
        "correction_family": family,
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
        "saved_bytes_budget": row.get("saved_bytes_budget"),
        "work_order_path": row.get("work_order_path"),
        "local_cpu_advisory_path": row.get("local_cpu_advisory_path"),
        "local_mlx_response_path": row.get("local_mlx_response_path"),
        "response_artifact_path": row.get("response_artifact_path"),
        "command_hints": dict(
            row.get("command_hints")
            if isinstance(row.get("command_hints"), Mapping)
            else {}
        ),
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


def _build_targeted_component_materialization_request_row(
    *,
    candidate_id: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    request_rank: int,
) -> dict[str, Any]:
    rows = sorted(candidate_rows, key=_targeted_component_response_sort_key)
    basis = [_targeted_component_materializer_basis_entry(row) for row in rows]
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
        "estimated_receiver_closed_rate_credit_score_units": rate_credit,
        "measured_lagrangian_delta_score_units_sum": lagrangian_sum,
        "best_measured_lagrangian_delta_score_units": best_row.get(
            "measured_lagrangian_delta_score_units"
        ),
        "budget_credit_remaining_score_units_min": min(
            float(row.get("budget_credit_remaining_score_units") or 0.0)
            for row in rows
        ),
        "materializer_chain_basis": basis,
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
    queue["materialization_request_summary"] = {
        "schema": TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA,
        "row_count": requests.get("row_count"),
        "accepted_response_count": requests.get("accepted_response_count"),
        "ready_for_budget_spend_count": 0,
        "allowed_use": (
            "targeted_component_correction_materialization_queue_summary_only"
        ),
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    return queue


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
                "target_kinds": target_kinds,
                **FALSE_AUTHORITY,
            },
            {
                "queue_consumer": "frontier_receiver_repair_queue",
                "handoff_reason": "prove_single_runtime_consumption_after_composition",
                "required_before_budget_spend": True,
                **FALSE_AUTHORITY,
            },
            {
                "queue_consumer": "frontier_targeted_component_correction_queue",
                "handoff_reason": "spend_receiver_closed_rate_budget_only_after_component_eval",
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
    target_present = "byte_range_entropy_recode_v1" in stage_targets

    schema_path = (
        _resolve_path(schema_manifest, repo_root=repo)
        if schema_manifest is not None
        else _first_existing_repo_file(repo, _DEFAULT_BYTE_RANGE_SCHEMA_MANIFEST_PATHS)
    )
    beam_paths = (
        [_resolve_path(path, repo_root=repo) for path in beam_probe_reports]
        if beam_probe_reports
        else _existing_repo_files(repo, _DEFAULT_BYTE_RANGE_BEAM_PROBE_REPORT_PATHS)
    )
    runtime_path = (
        _resolve_path(source_runtime_dir, repo_root=repo)
        if source_runtime_dir is not None
        else _first_existing_byte_range_runtime_dir(
            repo,
            _DEFAULT_BYTE_RANGE_SOURCE_RUNTIME_DIR_PATHS,
        )
    )
    default_source_archive, default_member_name = _byte_range_source_from_schema_manifest(
        schema_path if schema_path is not None and schema_path.is_file() else None,
        repo_root=repo,
    )
    source_archive_path = (
        _resolve_path(source_archive, repo_root=repo)
        if source_archive is not None
        else default_source_archive
    )
    combo_path = (
        _resolve_path(global_combo_report, repo_root=repo)
        if global_combo_report is not None
        else _first_existing_repo_file(repo, _DEFAULT_BYTE_RANGE_GLOBAL_COMBO_REPORT_PATHS)
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
    selected_member_name = str(member_name or default_member_name or "")

    context_blockers: list[str] = []
    if stage_row is None:
        context_blockers.append(f"operation_chain_stage_missing:{stage_id}")
    if not target_present:
        context_blockers.append("byte_range_entropy_recode_target_missing_from_stage")
    if schema_path is None or not schema_path.is_file():
        context_blockers.append("byte_range_stage_missing:schema_manifest")
    if not beam_paths or any(not path.is_file() for path in beam_paths):
        context_blockers.append("byte_range_stage_missing:beam_probe_reports")
    if runtime_path is None or not runtime_path.is_dir():
        context_blockers.append("byte_range_stage_missing:source_runtime_dir")
    elif not (runtime_path / "inflate.py").is_file() or not (runtime_path / "inflate.sh").is_file():
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
        "output_dir": _repo_rel(output_dir, repo),
        "chain_output_dir": _repo_rel(output_dir, repo),
        "fail_if_receiver_blocked": False,
        "context_blockers": _unique_strings(context_blockers),
        **FALSE_AUTHORITY,
    }
    payload = {
        "schema": BYTE_RANGE_STAGE_INPUTS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_operation_id": operation_chain_stage_plan.get("source_operation_id"),
        "stage_id": stage_id,
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


def build_frontier_operation_chain_compiler_queue(
    *,
    repo_root: str | Path,
    operation_chain_compiler_work_orders: Mapping[str, Any],
    operation_chain_compiler_work_orders_path: str | Path,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    queue_id: str = "frontier_operation_chain_compiler_queue",
    candidate_limit: int = 4,
) -> dict[str, Any] | None:
    """Compile multisurface chain work orders into local staged-plan queue rows."""

    repo = Path(repo_root)
    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    require_no_truthy_authority_fields(
        operation_chain_compiler_work_orders,
        context="operation_chain_compiler_queue_input",
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
        if byte_range_inputs_preview.get("local_chain_queueable") is True:
            chain_manifest_path = str(byte_range_inputs_preview["chain_manifest_path"])
            steps.extend(
                [
                    {
                        "id": "run_byte_range_entropy_recode_chain",
                        "kind": "command",
                        "command": list(byte_range_inputs_preview["local_chain_command"]),
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
    selected_rows = _selected_targeted_component_correction_rows(
        targeted_component_correction_acquisition,
        candidate_limit=candidate_limit,
    )
    if not selected_rows:
        return None
    selection_policy = _targeted_component_correction_queue_selection_policy(
        selected_rows=selected_rows,
        candidate_limit=candidate_limit,
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
        shared_dir = candidate_dir / "shared_component_response"
        local_cpu_advisory = shared_dir / "local_cpu_advisory.json"
        local_cpu_work_dir = shared_dir / "local_cpu_advisory_work"
        scorer_hashes = shared_dir / "scorer_input_cache_hashes.json"
        submission_dir = str(primary_row.get("submission_dir") or "")
        archive_path = str(
            primary_row.get("archive_path") or f"{submission_dir}/archive.zip"
        )
        inflate_sh_path = str(
            primary_row.get("inflate_sh_path") or f"{submission_dir}/inflate.sh"
        )
        local_mlx_response: Path | None = None
        request_metadata: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        work_order_step_ids: list[str] = []
        for row_index, row in enumerate(candidate_rows, start=1):
            acquisition_id = str(
                row.get("acquisition_id")
                or f"targeted_correction_{priority}_{row_index}"
            )
            request_dir = candidate_dir / _slug_token(acquisition_id)
            work_order_path = request_dir / "work_order.json"
            response_harvest_path = (
                request_dir / "component_correction_response_harvest.json"
            )
            work_order_step_id = f"emit_targeted_component_correction_work_order_{row_index:02d}"
            work_order_step_ids.append(work_order_step_id)
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
                    "component_correction_response_harvest_path": _repo_rel(
                        response_harvest_path,
                        repo,
                    ),
                    "local_cpu_advisory_path": _repo_rel(local_cpu_advisory, repo),
                    "local_mlx_response_path": (
                        None
                        if not include_mlx_response
                        else _repo_rel(shared_dir / "mlx_scorer_response.json", repo)
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
                    "command": [
                        ".venv/bin/python",
                        "tools/build_frontier_targeted_component_correction_work_order.py",
                        "--targeted-component-correction-acquisition",
                        _repo_rel(acquisition_path, repo),
                        "--acquisition-id",
                        acquisition_id,
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
                        "input_artifact_paths": [
                            _repo_rel(acquisition_path, repo),
                            str(row.get("closure_report_path") or ""),
                            str(row.get("paired_exact_readiness_bridge_report_path") or ""),
                        ],
                        "include_postcondition_paths": True,
                    },
                }
            )
        steps.append(
            {
                "id": "emit_targeted_component_correction_work_order",
                "kind": "command",
                "requires": work_order_step_ids,
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
                    "--inflate-timeout",
                    "1800",
                    "--evaluate-timeout",
                    "1800",
                    "--keep-work-dir",
                    "--scorer-input-cache-hashes-out",
                    _repo_rel(scorer_hashes, repo),
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 3900,
                "postconditions": [
                    {
                        "type": "json_false_authority",
                        "path": _repo_rel(local_cpu_advisory, repo),
                        "axis_key": "score_axis",
                        "axis_equals": "cpu_advisory",
                    }
                ],
                "telemetry": {
                    "artifact_paths": [
                        _repo_rel(local_cpu_advisory, repo),
                        _repo_rel(local_cpu_work_dir, repo),
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
                    "recursive": True,
                    "include_postcondition_paths": True,
                },
            },
        )
        if include_mlx_response:
            mlx_cache_dir = shared_dir / "mlx_scorer_input_cache"
            mlx_cache_audit = shared_dir / "mlx_scorer_input_cache_audit.json"
            mlx_response = shared_dir / "mlx_scorer_response.json"
            local_mlx_response = mlx_response
            for request in request_metadata:
                request["local_mlx_response_path"] = _repo_rel(mlx_response, repo)
            mlx_components_dir = shared_dir / "mlx_components"
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
            ]
            mlx_response_command = [
                ".venv/bin/python",
                "tools/run_mlx_scorer_response_from_local_advisory.py",
                "--local-cpu-advisory",
                _repo_rel(local_cpu_advisory, repo),
                "--reference-cache-dir",
                str(mlx_reference_cache_dir),
                "--candidate-cache-dir",
                _repo_rel(mlx_cache_dir, repo),
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
            steps.extend(
                [
                    {
                        "id": "build_mlx_component_cache",
                        "kind": "command",
                        "requires": ["local_cpu_component_advisory"],
                        "command": build_cache_command,
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 1800,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": _repo_rel(mlx_cache_audit, repo),
                                "key": "passed",
                                "equals": True,
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(mlx_cache_audit, repo),
                            },
                            {
                                "type": "json_false_authority",
                                "path": _repo_rel(mlx_cache_dir / "manifest.json", repo),
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                _repo_rel(mlx_cache_dir, repo),
                                _repo_rel(mlx_cache_audit, repo),
                            ],
                            "input_artifact_paths": [_repo_rel(local_cpu_advisory, repo)],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    },
                    {
                        "id": "local_mlx_component_response",
                        "kind": "command",
                        "requires": ["build_mlx_component_cache"],
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
                                _repo_rel(mlx_cache_dir, repo),
                            ],
                            "recursive": True,
                            "include_postcondition_paths": True,
                        },
                    },
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
            harvest_requires = ["local_cpu_component_advisory"]
            harvest_input_paths = [
                work_order_path_text,
                _repo_rel(local_cpu_advisory, repo),
            ]
            if local_mlx_response is not None:
                harvest_command.extend(
                    ["--local-mlx-response", _repo_rel(local_mlx_response, repo)]
                )
                harvest_requires = ["local_mlx_component_response"]
                harvest_input_paths.append(_repo_rel(local_mlx_response, repo))
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
                    "local_mlx_response_path": (
                        None
                        if local_mlx_response is None
                        else _repo_rel(local_mlx_response, repo)
                    ),
                    "local_mlx_response_enabled": include_mlx_response,
                    "mlx_device": mlx_device,
                    "shared_component_response_reuse": True,
                    "deduped_full_local_cpu_eval_count": max(0, len(candidate_rows) - 1),
                    "budget_spend_ready": False,
                    "selection_policy": dict(selection_policy),
                    "allowed_use": (
                        "targeted_component_correction_queue_metadata_only"
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
                    "local_cpu": local_cpu_concurrency,
                    "local_io_heavy": 1,
                    "local_mlx": local_mlx_concurrency if include_mlx_response else 0,
                    "modal_cpu": 0,
                    "modal_gpu": 0,
                },
            },
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
) -> dict[str, Any]:
    """Build a forest-level feedback refresh and optional DQS1 follow-up queue."""

    repo = Path(repo_root)
    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    payloads, source_paths, discovery = discover_materializer_feedback_payloads(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
        materializer_feedback_paths=materializer_feedback_paths,
    )
    pair_frame_requests, pair_frame_source_paths, pair_frame_discovery = (
        discover_pair_frame_geometry_queue_requests(
            repo_root=repo,
            frontier_artifact_roots=frontier_artifact_roots,
            pair_frame_geometry_paths=pair_frame_geometry_paths,
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

    return {
        "schema": FEEDBACK_REFRESH_SCHEMA,
        "generated_at_utc": _utc_now(),
        "candidate_limit": candidate_limit,
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
        "operation_materializer_bridge": operation_materializer_bridge,
        "receiver_repair_backlog": receiver_repair_backlog,
        "receiver_closed_correction_budget": receiver_closed_correction_budget,
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
        "queue_id": queue_id if queue_payload is not None else None,
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


__all__ = [
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
    "PAIR_FRAME_GEOMETRY_DISCOVERY_SCHEMA",
    "RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA",
    "RECEIVER_REPAIR_BACKLOG_SCHEMA",
    "RECEIVER_REPAIR_QUEUE_METADATA_SCHEMA",
    "RECEIVER_REPAIR_ROW_SCHEMA",
    "RECEIVER_REPAIR_WORK_ORDER_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_QUEUE_METADATA_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_QUEUE_METADATA_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_RESPONSE_ROW_SCHEMA",
    "TARGETED_COMPONENT_CORRECTION_WORK_ORDER_SCHEMA",
    "FrontierRateAttackFeedbackError",
    "build_frontier_byte_range_stage_inputs",
    "build_frontier_operation_materializer_bridge",
    "build_frontier_operation_portfolio",
    "build_frontier_rate_attack_feedback_refresh",
    "build_frontier_receiver_repair_backlog",
    "build_frontier_receiver_repair_queue",
    "build_frontier_receiver_repair_work_order",
    "build_frontier_targeted_component_correction_acquisition",
    "build_frontier_targeted_component_correction_materialization_queue",
    "build_frontier_targeted_component_correction_materialization_request",
    "build_frontier_targeted_component_correction_materialization_requests",
    "build_frontier_targeted_component_correction_queue",
    "build_frontier_targeted_component_correction_response_harvest",
    "build_frontier_targeted_component_correction_response_harvest_from_artifacts",
    "build_frontier_targeted_component_correction_work_order",
    "build_receiver_closed_correction_budget",
    "discover_dqs1_observation_jsonl_paths",
    "discover_local_cpu_eureka_planning_signals",
    "discover_materializer_feedback_payloads",
    "discover_pair_frame_geometry_queue_requests",
    "load_dqs1_observations",
]
