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
from tac.optimization.pairset_component_marginal import component_marginal_status
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .dqs1_local_first_queue import (
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
MATERIALIZER_SUBMISSION_CLOSURE_TOOL = (
    "tools/build_materializer_submission_closure.py"
)
MATERIALIZER_SUBMISSION_CLOSURE_REPORT_SCHEMA = (
    "materializer_submission_runtime_closure_report.v1"
)
QUEUE_FALSE_AUTHORITY_FALSE_OR_MISSING_FIELDS = tuple(
    field
    for field in DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS
    if field != "dispatch_ready"
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
    scanned_files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        scanned_files += 1
        if scanned_files > max_files:
            raise FrontierRateAttackFeedbackError(
                f"{root}: materializer feedback discovery exceeded max_files={max_files}"
            )
        if path.name in {"sweep.json", "observations.jsonl"} or (path.suffix in {".json", ".jsonl"} and "materializer" in path.as_posix()):
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
    paths: list[Path] = []
    seen_paths: set[str] = set()
    for value in materializer_feedback_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            paths.append(path)
    for value in frontier_artifact_roots:
        root = _resolve_path(value, repo_root=repo)
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
            for root in frontier_artifact_roots
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


def _targeted_correction_budget_summary(
    *,
    component_summary: Mapping[str, Any],
    materializer_rows: Sequence[Mapping[str, Any]],
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

    active = bool(local_byte_savings or materializer_saved_bytes)
    blockers: list[str] = []
    if not local_rate_credits:
        blockers.append("requires_component_measured_rate_credit_for_repair_budget")
    if materializer_saved_bytes:
        blockers.append("materializer_saved_bytes_require_receiver_runtime_proof_before_spend")
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
        "recommended_next_action": (
            "compose_rate_positive_ops_with_targeted_segnet_posenet_repairs_and_accept_"
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
    )
    rows = [
        *materializer_rows,
        *_materializer_chain_operation_rows(materializer_rows),
        *_dqs1_component_operation_rows(dqs1_observations, component_summary),
        *_eureka_operation_rows(eureka_planning),
        *_pair_frame_operation_rows(pair_frame_requests, pair_frame_source_paths),
        *_seed_operation_rows(
            component_summary=component_summary,
            master_gradient=master_gradient,
        ),
        *_backlog_operation_rows(
            component_summary=component_summary,
            master_gradient=master_gradient,
        ),
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
        "allowed_use": "queue_metadata_pointer_to_operation_portfolio_artifact",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _slug_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in out.split("_") if part) or "unknown"


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
    operation_portfolio = build_frontier_operation_portfolio(
        repo_root=repo,
        materializer_feedback_payloads=payloads,
        materializer_feedback_source_paths=source_paths,
        dqs1_observations=dqs1_observations,
        eureka_planning=eureka_planning,
        pair_frame_requests=pair_frame_requests,
        pair_frame_source_paths=pair_frame_source_paths,
    )
    receiver_repair_backlog = build_frontier_receiver_repair_backlog(
        operation_portfolio
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
            for experiment in queue_payload.get("experiments", []):
                if not isinstance(experiment, dict):
                    continue
                metadata = experiment.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["frontier_operation_portfolio"] = portfolio_metadata
                    metadata["frontier_receiver_repair_backlog"] = (
                        receiver_repair_metadata
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
        "receiver_repair_backlog": receiver_repair_backlog,
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
    "OPERATION_PORTFOLIO_SCHEMA",
    "OPERATION_PORTFOLIO_TAXONOMY_SCHEMA",
    "PAIR_FRAME_GEOMETRY_DISCOVERY_SCHEMA",
    "RECEIVER_REPAIR_BACKLOG_SCHEMA",
    "RECEIVER_REPAIR_QUEUE_METADATA_SCHEMA",
    "RECEIVER_REPAIR_ROW_SCHEMA",
    "RECEIVER_REPAIR_WORK_ORDER_SCHEMA",
    "FrontierRateAttackFeedbackError",
    "build_frontier_operation_portfolio",
    "build_frontier_rate_attack_feedback_refresh",
    "build_frontier_receiver_repair_backlog",
    "build_frontier_receiver_repair_queue",
    "build_frontier_receiver_repair_work_order",
    "discover_dqs1_observation_jsonl_paths",
    "discover_local_cpu_eureka_planning_signals",
    "discover_materializer_feedback_payloads",
    "discover_pair_frame_geometry_queue_requests",
    "load_dqs1_observations",
]
