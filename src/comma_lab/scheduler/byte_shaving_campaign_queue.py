# SPDX-License-Identifier: MIT
"""Compile byte-shaving campaign plans into executable local-first queues.

The executable surface is deliberately narrower than the planner. Materializer
resolution happens through ``byte_shaving_materializer_registry`` so DQS1
``drop_pair`` selections can become queue actions while frame, tensor, byte,
archive, scorer-response, and future substrate operations fail closed with
typed missing-materializer blockers.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    read_strict_single_member_zip,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME,
    CHAIN_SCHEMA,
)
from tac.optimization.byte_shaving_campaign import (
    COUPLED_OPERATION_SET_SCHEMA,
    FALSE_AUTHORITY,
    PLAN_SCHEMA,
)
from tac.optimization.decoder_q_constants import FEC6_PAIR_COUNT
from tac.optimization.family_agnostic_materializers import (
    ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
    PACKET_MEMBER_MERGE_SCHEMA,
    PACKET_MEMBER_RECOMPRESS_SCHEMA,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA,
    RENDERER_PAYLOAD_DFL1_SCHEMA,
    TENSOR_FACTORIZE_SCHEMA,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_MANIFEST_NAME as INVERSE_SCORER_CELL_CHAIN_MANIFEST,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_SCHEMA as INVERSE_SCORER_CELL_CHAIN_SCHEMA,
)
from tac.optimization.inverse_steganalysis_operation_set_compiler import (
    packet_ir_operation_set_from_compiler_hint,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.packet_compiler.deterministic_compiler import (
    PACKET_IR_OPERATION_SET_SCHEMA,
    packetir_operation_set_bridge_contract,
)

from .byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    DQS1_PAIRSET_TARGET_KIND,
    INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
    INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
    INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
    INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER,
    INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND,
    INVERSE_SCORER_CELL_TARGET_KIND,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    REGISTRY_SCHEMA,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
    known_materializer_target_kinds,
    registry_manifest,
    resolve_materializer,
    suggest_materializer_adapters,
)
from .dqs1_local_first_queue import SAFE_OPERATOR_ACTION, candidate_slug
from .experiment_queue import (
    QUEUE_SCHEMA,
    ExperimentQueueError,
    default_state_path,
    normalize_queue_definition,
)
from .storage_preflight import (
    build_scheduler_storage_preflight_experiment,
    validate_scheduler_storage_preflight_config,
)

MATERIALIZATION_SCHEMA = "byte_shaving_campaign_materialization.v1"
MATERIALIZER_BACKLOG_SCHEMA = "byte_shaving_materializer_backlog.v1"
MATERIALIZER_CONTEXTS_SCHEMA = "byte_shaving_materializer_contexts.v1"
MATERIALIZER_WORK_QUEUE_SCHEMA = "byte_shaving_materializer_work_queue.v1"
MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA = "byte_shaving_materializer_execution_experiment_metadata.v1"
GROUPED_ARCHIVE_STATE_MATERIALIZER_REQUEST_SCHEMA = "grouped_archive_state_materializer_request.v1"
GROUPED_ARCHIVE_STATE_MATERIALIZER_CHAIN_SCHEMA = "grouped_archive_state_materializer_chain.v1"
MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID = "materializer_scheduler_preflight"
ACTION_SUMMARY_SCHEMA = "cross_family_candidate_portfolio_action_summary.v1"
PORTFOLIO_SCHEMA = "byte_shaving_campaign_dqs1_operator_portfolio.v1"
TOOL_NAME = "comma_lab.scheduler.byte_shaving_campaign_queue"
BYTE_RANGE_CHAIN_TOOL = "tools/run_byte_range_entropy_recode_chain.py"
FAMILY_AGNOSTIC_MATERIALIZER_TOOL = "tools/run_family_agnostic_materializer.py"
FAMILY_AGNOSTIC_MATERIALIZER_SWEEP_TOOL = "tools/run_family_agnostic_materializer_sweep.py"
GROUPED_ARCHIVE_STATE_MATERIALIZER_TOOL = "tools/run_grouped_family_agnostic_materializer.py"
SHELL_INFLATE_PARITY_TOOL = "tools/prove_shell_inflate_parity.py"
INVERSE_ACTION_FUNCTIONAL_TOOL = "tools/build_inverse_steganalysis_action_functional.py"
INVERSE_SCORER_CELL_TOOL = "tools/materialize_inverse_scorer_cell_candidate.py"
INVERSE_SCORER_CELL_CHAIN_TOOL = "tools/run_inverse_scorer_cell_candidate_chain.py"
HARVEST_MATERIALIZER_TOOL = "tools/harvest_materializer_chain_candidates.py"
MATERIALIZER_DISPATCH_PLAN_TOOL = "tools/build_materializer_exact_eval_dispatch_plan.py"
BYTE_RANGE_CHAIN_MANIFEST = CHAIN_MANIFEST_NAME
INVERSE_ACTION_FUNCTIONAL_SCHEMA = "inverse_steganalysis_discrete_action_functional.v1"
INVERSE_SCORER_CELL_CANDIDATE_SCHEMA = "inverse_scorer_cell_candidate_v1"
MATERIALIZER_EXECUTION_STEP_ID = "materialize_local_proof_chain"
MATERIALIZER_DFL1_PARITY_STEP_ID = "prove_renderer_payload_dfl1_shell_parity"
MATERIALIZER_HARVEST_STEP_ID = "harvest_materializer_chains"
MATERIALIZER_DISPATCH_PLAN_STEP_ID = "build_exact_eval_dispatch_plan"
MATERIALIZER_HARVEST_REPORT_SCHEMA = "materializer_chain_harvest_report.v1"
MATERIALIZER_EXACT_READINESS_BRIDGE_SCHEMA = "materializer_chain_exact_readiness_bridge_report.v1"
MATERIALIZER_EXACT_EVAL_DISPATCH_PLAN_SCHEMA = "materializer_exact_eval_dispatch_plan.v1"
FAMILY_AGNOSTIC_MATERIALIZER_SWEEP_SCHEMA = "family_agnostic_materializer_empirical_sweep.v1"
FAMILY_AGNOSTIC_MATERIALIZER_SWEEP_OBSERVATION_SCHEMA = "family_agnostic_materializer_empirical_observation.v1"
OPTIMIZER_CANDIDATE_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
HARVESTABLE_MATERIALIZER_MANIFEST_SCHEMAS = frozenset(
    {
        CHAIN_SCHEMA,
        INVERSE_SCORER_CELL_CHAIN_SCHEMA,
        ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
        PACKET_MEMBER_MERGE_SCHEMA,
        PACKET_MEMBER_RECOMPRESS_SCHEMA,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA,
        RENDERER_PAYLOAD_DFL1_SCHEMA,
        TENSOR_FACTORIZE_SCHEMA,
    }
)
GROUPED_ARCHIVE_STATE_SUPPORTED_TARGET_KINDS = frozenset(
    {
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    }
)
OPERATION_SET_CLEARABLE_SOURCE_BLOCKERS = frozenset(
    {
        "operation_set_requires_atomic_materializer_or_explicit_partial_set_split",
        "operation_set_order_requires_materialization_probe",
    }
)
OPERATION_SET_ENFORCED_SOURCE_BLOCKERS = OPERATION_SET_CLEARABLE_SOURCE_BLOCKERS | {
    "operation_set_sequence_not_permutation_of_selected_operations"
}
DQS1_LOCAL_MATERIALIZATION_CLEARABLE_BLOCKERS = frozenset(
    {
        "dqs1_pairset_acquisition_unit_is_planning_only",
        "dqs1_pairset_acquisition_signal_is_planning_only",
        "requires_local_dqs1_materialization_and_locality_controls",
        "requires_receiver_runtime_consumption_proof",
        "requires_exact_auth_eval_before_score_claim",
        "packetir_operation_not_byte_closed:pair",
    }
)
OPERATION_PARAM_HINT_KEYS = (
    "archive_section",
    "archive_path",
    "source_archive",
    "output_archive",
    "output_manifest",
    "json_out",
    "manifest_out",
    "section_name",
    "target_section",
    "target_sections",
    "packet_member",
    "packet_member_manifest",
    "member_name",
    "member_names",
    "payload_member_name",
    "renderer_payload_dfl1_source_runtime_dir",
    "renderer_payload_dfl1_inflate_runtime_dir",
    "renderer_payload_dfl1_candidate_runtime_dir",
    "renderer_payload_dfl1_full_frame_file_list",
    "renderer_payload_dfl1_full_frame_file_list_entries",
    "renderer_payload_dfl1_expected_full_frame_file_list_sha256",
    "renderer_payload_dfl1_expected_full_frame_entry_count",
    "renderer_payload_dfl1_full_frame_file_list_source",
    "renderer_payload_dfl1_inflate_parity_output_dir",
    "full_frame_file_list",
    "full_frame_file_list_entries",
    "expected_full_frame_file_list_sha256",
    "expected_full_frame_entry_count",
    "full_frame_file_list_source",
    "tensor_name",
    "tensor_path",
    "tensor_manifest",
    "rank",
    "byte_range",
    "archive_byte_range",
    "archive_member_name",
    "frame_range",
    "pair_indices",
    "region_bbox",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _repo_rel_no_resolve(path: Path, repo_root: Path) -> str:
    try:
        return path.absolute().relative_to(repo_root.absolute()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_output_path(path_value: Any, *, repo_root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _expected_materializer_workload_root(
    *,
    results_root: str,
    expected_workload_root: str | None,
) -> Path | None:
    if expected_workload_root is not None:
        return Path(expected_workload_root).expanduser().resolve(strict=False)
    path = Path(results_root).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return None


def _path_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _operation_params(
    operation: Mapping[str, Any],
    unit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for source in (unit, operation):
        if not isinstance(source, Mapping):
            continue
        params.update(dict(_as_mapping(source.get("operation_params"))))
        params.update(dict(_as_mapping(source.get("params"))))
        for key in OPERATION_PARAM_HINT_KEYS:
            value = source.get(key)
            if value is not None and key not in params:
                params[key] = value
    return params


def _operation_sequence_key(operation: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(operation.get("unit_id") or ""),
        str(operation.get("operation_id") or ""),
    )


def _finite_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _validate_pair_indices(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    if not values:
        raise ExperimentQueueError(f"{label}: pair index list must be non-empty")
    pairs = tuple(int(value) for value in values)
    if len(set(pairs)) != len(pairs):
        raise ExperimentQueueError(f"{label}: pair indices contain duplicates")
    if tuple(sorted(pairs)) != pairs:
        raise ExperimentQueueError(f"{label}: pair indices must be sorted ascending")
    bad = [pair for pair in pairs if not 0 <= pair < FEC6_PAIR_COUNT]
    if bad:
        raise ExperimentQueueError(f"{label}: pair indices out of range 0..{FEC6_PAIR_COUNT - 1}: {bad}")
    return pairs


def _pair_index_list_from_params(
    params: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> tuple[int, ...] | None:
    raw = params.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ExperimentQueueError(f"{label}: expected list of integer pair indices")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise ExperimentQueueError(f"{label}: expected integer pair indices")
    return _validate_pair_indices([int(item) for item in raw], label=label)


def _is_dqs1_pairset_selector_operation(operation: Mapping[str, Any]) -> bool:
    params = _operation_params(operation)
    return (
        operation.get("target_kind") == DQS1_PAIRSET_TARGET_KIND
        and isinstance(params.get("dropped_pair_indices"), list)
        and isinstance(params.get("selected_pair_indices"), list)
    )


def _is_dqs1_pairset_selector_unit(unit: Mapping[str, Any]) -> bool:
    if any(
        _is_dqs1_pairset_selector_operation(operation)
        for operation in _as_list(unit.get("operations"))
        if isinstance(operation, Mapping)
    ):
        return True
    params = _as_mapping(unit.get("recommended_operation_params"))
    return (
        unit.get("recommended_operation_target_kind") == DQS1_PAIRSET_TARGET_KIND
        and isinstance(params.get("dropped_pair_indices"), list)
        and isinstance(params.get("selected_pair_indices"), list)
    )


def _ranked_unit_selected_operations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    operations = [
        dict(item) for item in _as_list(row.get("operations")) if isinstance(item, Mapping)
    ]
    if operations:
        return operations
    operation_family = str(row.get("recommended_operation_family") or "").strip()
    target_kind = str(row.get("recommended_operation_target_kind") or "").strip()
    materializer = str(row.get("recommended_operation_materializer") or "").strip()
    if not (operation_family or target_kind or materializer):
        return []
    metadata = row.get("recommended_operation_metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    return [
        {
            "unit_id": row.get("unit_id"),
            "operation_id": row.get("recommended_operation_id")
            or operation_family
            or "ranked_unit_operation",
            "operation_family": operation_family,
            "target_kind": target_kind,
            "materializer": materializer,
            "params": dict(_as_mapping(row.get("recommended_operation_params"))),
            "candidate_saved_bytes": row.get("candidate_saved_bytes"),
            **{
                key: metadata_map[key]
                for key in (
                    "receiver_contract_kind",
                    "materializer_executable",
                    "materializer_execution_status",
                )
                if key in metadata_map and metadata_map[key] is not None
            },
        }
    ]


def _local_dqs1_materialization_blockers(
    blockers: Sequence[Any],
    *,
    clear_planning_pairset_blockers: bool,
) -> list[str]:
    out: list[str] = []
    for blocker in blockers:
        text = str(blocker)
        if clear_planning_pairset_blockers and text in DQS1_LOCAL_MATERIALIZATION_CLEARABLE_BLOCKERS:
            continue
        out.append(text)
    return ordered_unique(out)


def _base_pair_indices(
    payload: Mapping[str, Any],
    explicit: Sequence[int] | None,
) -> tuple[int, ...] | None:
    if explicit is not None:
        return _validate_pair_indices([int(value) for value in explicit], label="base_pair_indices")
    for key in ("dqs1_base_pair_indices", "base_pair_indices", "selected_pair_indices"):
        raw = payload.get(key)
        if isinstance(raw, list) and raw:
            if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
                raise ExperimentQueueError(f"{key}: expected integer pair indices")
            return _validate_pair_indices([int(item) for item in raw], label=key)
    return None


def _pair_index_from_operation(operation: Mapping[str, Any]) -> int | None:
    params = operation.get("params")
    param_map = params if isinstance(params, Mapping) else {}
    for source in (operation, param_map):
        for key in (
            "pair_index",
            "dropped_pair_index",
            "drop_pair_index",
            "pair",
            "pair_id",
        ):
            parsed = _finite_int(source.get(key))
            if parsed is not None:
                return parsed
    for value in (operation.get("unit_id"), operation.get("operation_id")):
        if not isinstance(value, str):
            continue
        match = re.search(r"(?:^|[_-])pair0*(\d{1,4})(?:$|[_-])", value)
        if match is None:
            match = re.search(r"pair0*(\d{1,4})", value)
        if match is not None:
            return int(match.group(1))
    return None


def _candidate_id(kind: str, selection_id: str, dropped_pairs: Sequence[int]) -> str:
    safe_kind = re.sub(r"[^a-z0-9_]+", "_", kind.lower()).strip("_") or "row"
    safe_selection = re.sub(r"[^a-z0-9_]+", "_", selection_id.lower()).strip("_") or "selection"
    suffix = "_".join(f"p{pair:04d}" for pair in sorted(dropped_pairs)[:8])
    if len(dropped_pairs) > 8:
        suffix = f"{suffix}_plus{len(dropped_pairs) - 8:02d}"
    return f"pairset_byte_shave_{safe_kind}_{safe_selection}_{suffix}"


def _selection_id(kind: str, row: Mapping[str, Any]) -> str:
    value = row.get("operation_set_id") or row.get("combo_id") or row.get("sweep_id") or row.get("unit_id") or row.get("selection_id")
    if isinstance(value, str) and value.strip():
        return value
    return kind


def _iter_plan_rows(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    operation_sets = [row for row in _as_list(payload.get("operation_set_ladder")) if isinstance(row, Mapping)]
    if operation_sets:
        rows.extend(("operation_set", row) for row in operation_sets)
    else:
        rows.extend(("combo", row) for row in _as_list(payload.get("combination_ladder")) if isinstance(row, Mapping))
    rows.extend(
        ("ranked_unit", row)
        for row in _as_list(payload.get("ranked_units"))
        if isinstance(row, Mapping) and _is_dqs1_pairset_selector_unit(row)
    )
    for kind, key in (("prefix", "sweep_ladder"),):
        for row in _as_list(payload.get(key)):
            if isinstance(row, Mapping):
                rows.append((kind, row))
    return rows


def _unit_blockers_by_id(source_units: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for unit in source_units:
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        out[unit_id] = ordered_unique(str(item) for item in _as_list(unit.get("blockers")))
    return out


def _units_by_id(source_units: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(unit.get("unit_id") or ""): unit for unit in source_units if str(unit.get("unit_id") or "")}


def _packet_ir_operation_set_for_row(
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    operation_set_id = str(row.get("operation_set_id") or "")
    if not operation_set_id:
        return None
    for packet_ir_set in _as_list(payload.get("packet_ir_operation_sets")):
        if not isinstance(packet_ir_set, Mapping):
            continue
        if str(packet_ir_set.get("source_operation_set_id") or "") == operation_set_id:
            return packet_ir_set
    return None


def _packet_ir_operation_set_blockers(
    row: Mapping[str, Any],
    packet_ir: Mapping[str, Any] | None,
) -> list[str]:
    if packet_ir is None:
        return ["operation_set_packet_ir_operation_set_missing"]

    blockers: list[str] = []
    contract = packetir_operation_set_bridge_contract()
    operation_set_id = str(row.get("operation_set_id") or "")
    if packet_ir.get("schema") != PACKET_IR_OPERATION_SET_SCHEMA:
        blockers.append("operation_set_packet_ir_schema_mismatch")
    if str(packet_ir.get("source_operation_set_id") or "") != operation_set_id:
        blockers.append("operation_set_packet_ir_source_id_mismatch")
    if packet_ir.get("chosen_operation_sequence_sha256") != row.get("chosen_operation_sequence_sha256"):
        blockers.append("operation_set_packet_ir_sequence_hash_mismatch")
    if packet_ir.get("chosen_operation_sequence_is_permutation") is not True:
        blockers.append("operation_set_packet_ir_sequence_not_permutation")
    if _as_list(packet_ir.get("required_order")) != list(contract["required_order"]):
        blockers.append("operation_set_packet_ir_required_order_mismatch")
    if _as_list(packet_ir.get("required_proofs")) != list(contract["required_proofs"]):
        blockers.append("operation_set_packet_ir_required_proofs_mismatch")

    compiler_contract = packet_ir.get("compiler_contract")
    if not isinstance(compiler_contract, Mapping):
        blockers.append("operation_set_packet_ir_compiler_contract_missing")
    else:
        for key in (
            "schema",
            "canonical_packet_compiler_module",
            "canonical_packet_compiler_schema",
            "recommended_ir_schema",
        ):
            if compiler_contract.get(key) != contract.get(key):
                blockers.append(f"operation_set_packet_ir_compiler_contract_mismatch:{key}")
        if _as_list(compiler_contract.get("required_order")) != list(contract["required_order"]):
            blockers.append("operation_set_packet_ir_compiler_contract_mismatch:required_order")
        if _as_list(compiler_contract.get("required_proofs")) != list(contract["required_proofs"]):
            blockers.append("operation_set_packet_ir_compiler_contract_mismatch:required_proofs")

    operations = [operation for operation in _as_list(packet_ir.get("operations")) if isinstance(operation, Mapping)]
    if not operations:
        blockers.append("operation_set_packet_ir_operations_missing")
    try:
        require_no_truthy_authority_fields(
            packet_ir,
            context="packet_ir_operation_set",
        )
    except ValueError as exc:
        blockers.append(f"operation_set_packet_ir_authority_violation:{exc}")
    return ordered_unique(blockers)


def _resolution_gap_class(resolution: Mapping[str, Any], blockers: Sequence[str]) -> str:
    joined = "\n".join(blockers)
    if "non_dqs1_target_requires_materializer_work_queue:" in joined:
        return "materializer_work_queue_required"
    if "planning_only_materializer_not_candidate_archive:" in joined:
        return "materializer_work_queue_required"
    if "materializer_target_kind_required:" in joined:
        return "target_kind_required"
    if "materializer_not_registered:" in joined:
        return "adapter_not_registered"
    if "materializer_unit_kind_mismatch:" in joined:
        return "adapter_unit_kind_mismatch"
    if "materializer_operation_family_mismatch:" in joined:
        return "adapter_operation_family_mismatch"
    if "materializer_target_kind_mismatch:" in joined:
        return "adapter_target_kind_mismatch"
    if "materializer_not_executable:" in joined:
        return "adapter_not_executable"
    if "operation_family_missing:" in joined:
        return "operation_family_missing"
    if "unknown_operation_family:" in joined:
        return "unknown_operation_family"
    if any(str(item) for item in _as_list(resolution.get("selected_operation_blockers"))):
        return "selected_operation_blocked"
    return "source_unit_blocked"


def _backlog_key(resolution: Mapping[str, Any], gap_class: str) -> str:
    unit_kind = str(resolution.get("unit_kind") or "<missing>")
    family = str(resolution.get("operation_family") or "<missing>")
    target = str(resolution.get("target_kind") or "<target_tbd>")
    materializer = str(resolution.get("materializer_id") or "<materializer_tbd>")
    return f"{gap_class}:{target}:{unit_kind}:{family}:{materializer}"


def _suggested_materializer_rows(
    resolution: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "materializer_id": adapter.materializer_id,
            "target_kind": adapter.target_kind,
            "executable": adapter.executable,
            "emits_candidate_archive": adapter.emits_candidate_archive,
            "planning_only": adapter.planning_only,
            "receiver_contract_id": adapter.receiver_contract_id,
            "receiver_contract_kind": adapter.receiver_contract_kind,
            "cooperative_receiver_required": adapter.cooperative_receiver_required,
            "materialization_resource_kind": adapter.materialization_resource_kind,
            "required_context_fields": list(adapter.required_context_fields),
            "implementation_module": adapter.implementation_module,
            "plan_function": adapter.plan_function,
            "materialize_function": adapter.materialize_function,
            "receiver_proof_function": adapter.receiver_proof_function,
            "receiver_verify_function": adapter.receiver_verify_function,
            "description": adapter.description,
        }
        for adapter in suggest_materializer_adapters(
            unit_kind=str(resolution.get("unit_kind") or ""),
            operation_family=str(resolution.get("operation_family") or ""),
        )
    ]


def _receiver_contract_status(resolution: Mapping[str, Any], gap_class: str) -> str:
    if resolution.get("receiver_contract_id") and gap_class == "source_unit_blocked":
        return "receiver_contract_registered_but_source_blocked"
    if resolution.get("receiver_contract_id") and gap_class == "selected_operation_blocked":
        return "receiver_contract_registered_but_operation_blocked"
    if gap_class == "target_kind_required":
        return "receiver_target_contract_required"
    if gap_class == "adapter_not_registered":
        return "receiver_adapter_contract_missing"
    if resolution.get("receiver_contract_id") and gap_class == "adapter_not_executable":
        return "receiver_contract_registered_but_adapter_not_executable"
    if gap_class in {
        "adapter_unit_kind_mismatch",
        "adapter_operation_family_mismatch",
        "adapter_target_kind_mismatch",
    }:
        return "receiver_contract_mismatch"
    if gap_class == "materializer_work_queue_required":
        return "receiver_contract_registered_for_materializer_work_queue"
    if gap_class in {"operation_family_missing", "unknown_operation_family"}:
        return "receiver_operation_contract_invalid"
    return "receiver_contract_blocked"


def _packet_ir_compiled_row(packet_ir: Mapping[str, Any]) -> dict[str, Any]:
    operations = [operation for operation in _as_list(packet_ir.get("operations")) if isinstance(operation, Mapping)]
    if not operations:
        raise ExperimentQueueError("packet_ir_operation_set operations[] missing")
    if packet_ir.get("chosen_operation_sequence_is_permutation") is False:
        raise ExperimentQueueError("packet_ir_operation_set chosen sequence is not a permutation")
    materializer_resolutions: list[dict[str, Any]] = []
    source_units: list[dict[str, Any]] = []
    known_target_kinds = known_materializer_target_kinds()
    for operation_index, operation in enumerate(operations, start=1):
        unit_id = str(operation.get("unit_id") or "")
        operation_params = _operation_params(operation)
        unit = {
            "unit_id": unit_id,
            "unit_kind": operation.get("unit_kind"),
            "candidate_saved_bytes": operation.get("candidate_saved_bytes"),
            "packet_ir_operation_index": operation_index,
            "operation_params": operation_params,
            "blockers": [],
        }
        resolution = resolve_materializer(operation=operation, unit=unit)
        packet_ir_context_blockers = (
            []
            if resolution.target_kind == DQS1_PAIRSET_TARGET_KIND and resolution.executable and not resolution.blockers
            else ["packetir_operation_set_requires_materializer_contexts"]
        )
        resolution_blockers = ordered_unique(
            [
                *[str(item) for item in resolution.blockers],
                *[
                    f"selected_operation_blocker:{unit_id or '<missing>'}:{item}"
                    for item in _as_list(operation.get("blockers"))
                ],
                *(
                    [f"non_dqs1_target_requires_materializer_work_queue:{resolution.target_kind}"]
                    if resolution.target_kind
                    and resolution.target_kind != DQS1_PAIRSET_TARGET_KIND
                    and resolution.executable
                    else []
                ),
                *(
                    [f"unsupported_materializer_target:{resolution.target_kind}"]
                    if resolution.target_kind and resolution.target_kind not in known_target_kinds
                    else []
                ),
                *packet_ir_context_blockers,
            ]
        )
        materializer_resolutions.append(
            {
                "unit_id": resolution.unit_id,
                "unit_kind": resolution.unit_kind,
                "operation_id": resolution.operation_id,
                "operation_family": resolution.operation_family,
                "explicit_materializer": resolution.explicit_materializer,
                "materializer_id": resolution.materializer_id,
                "target_kind": resolution.target_kind,
                "receiver_contract_id": resolution.receiver_contract_id,
                "receiver_contract_kind": resolution.receiver_contract_kind,
                "cooperative_receiver_required": resolution.cooperative_receiver_required,
                "materialization_resource_kind": resolution.materialization_resource_kind,
                "executable": resolution.executable,
                "packet_ir_operation_index": operation_index,
                "blockers": resolution_blockers,
                "selected_operation_blockers": _as_list(operation.get("blockers")),
            }
        )
        source_units.append(unit)
    return {
        "schema": "packet_ir_operation_set_materializer_backlog_source_row.v1",
        "candidate_id": packet_ir.get("operation_set_id"),
        "selection_id": packet_ir.get("operation_set_id"),
        "selection_kind": "packet_ir_operation_set",
        "operation_set_id": packet_ir.get("source_operation_set_id"),
        "packet_ir_operation_set": dict(packet_ir),
        "candidate_saved_bytes": packet_ir.get("candidate_saved_bytes"),
        "expected_delta_score": packet_ir.get("expected_delta_score"),
        "expected_score_gain": packet_ir.get("expected_score_gain"),
        "source_units": source_units,
        "materializer_resolutions": materializer_resolutions,
        "executable": False,
        "materialization_blockers": [
            "packetir_operation_set_requires_materializer_contexts",
            "packetir_operation_set_requires_runtime_consumption_proof",
            "packetir_operation_set_requires_exact_readiness_handoff",
        ],
        **FALSE_AUTHORITY,
    }


def lower_packetir_operation_set_to_backlog_rows(
    packet_ir: Mapping[str, Any],
    source_backlog_row: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Lower one PacketIR operation set into scheduler materializer backlog rows.

    PacketIR remains a byte-grammar handoff, not execution authority. This
    helper resolves each concrete operation through the scheduler materializer
    registry, then returns the same backlog-row schema consumed by the existing
    context/work-queue path.
    """

    if packet_ir.get("schema") != PACKET_IR_OPERATION_SET_SCHEMA:
        raise ExperimentQueueError(f"expected schema {PACKET_IR_OPERATION_SET_SCHEMA}")
    try:
        require_no_truthy_authority_fields(
            packet_ir,
            context="packet_ir_operation_set",
        )
        if source_backlog_row is not None:
            require_no_truthy_authority_fields(
                source_backlog_row,
                context="packet_ir_source_backlog_row",
            )
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    backlog = build_materializer_backlog([_packet_ir_compiled_row(packet_ir)])
    rows = [dict(row) for row in _as_list(backlog.get("rows")) if isinstance(row, Mapping)]
    for row in rows:
        row["source_packet_ir_schema"] = packet_ir.get("schema")
        row["source_packet_ir_operation_set_id"] = packet_ir.get("operation_set_id")
        row["source_packet_ir_source_operation_set_id"] = packet_ir.get("source_operation_set_id")
        if source_backlog_row is not None:
            row["source_backlog_key"] = source_backlog_row.get("backlog_key")
        row.update(FALSE_AUTHORITY)
    return rows


def _counter_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, count in value.items():
        parsed = _finite_int(count)
        if parsed is None:
            continue
        out[str(key)] = parsed
    return out


def _merge_packet_ir_materializer_backlog_rows(
    materializer_backlog: Mapping[str, Any],
    packet_ir_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge PacketIR-lowered rows into the authoritative materializer backlog."""

    rows_by_key: dict[str, dict[str, Any]] = {}
    packet_ir_keys: set[str] = set()
    for row in _as_list(materializer_backlog.get("rows")):
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("backlog_key") or "")
        if not key:
            continue
        rows_by_key[key] = dict(row)

    for packet_row in packet_ir_rows:
        if not isinstance(packet_row, Mapping):
            continue
        key = str(packet_row.get("backlog_key") or "")
        if not key:
            continue
        packet_ir_keys.add(key)
        current = rows_by_key.get(key)
        if current is None:
            current = dict(packet_row)
            rows_by_key[key] = current
        current["packet_ir_lowered_row_count"] = int(current.get("packet_ir_lowered_row_count") or 0) + 1
        current["source_packet_ir_schemas"] = ordered_unique(
            [
                *_as_list(current.get("source_packet_ir_schemas")),
                str(packet_row.get("source_packet_ir_schema") or ""),
            ]
        )
        current["source_packet_ir_operation_set_ids"] = ordered_unique(
            [
                *_as_list(current.get("source_packet_ir_operation_set_ids")),
                str(packet_row.get("source_packet_ir_operation_set_id") or ""),
            ]
        )
        current["source_packet_ir_source_operation_set_ids"] = ordered_unique(
            [
                *_as_list(current.get("source_packet_ir_source_operation_set_ids")),
                str(packet_row.get("source_packet_ir_source_operation_set_id") or ""),
            ]
        )
        current["source_unit_ids"] = ordered_unique(
            [
                *_as_list(current.get("source_unit_ids")),
                *_as_list(packet_row.get("source_unit_ids")),
            ]
        )
        current["source_packet_ir_operation_indices"] = sorted(
            {
                int(index)
                for index in [
                    *_as_list(current.get("source_packet_ir_operation_indices")),
                    *_as_list(packet_row.get("source_packet_ir_operation_indices")),
                ]
                if _finite_int(index) is not None
            }
        )
        by_unit = dict(_as_mapping(current.get("source_packet_ir_operation_indices_by_unit")))
        by_unit.update(
            dict(_as_mapping(packet_row.get("source_packet_ir_operation_indices_by_unit")))
        )
        if by_unit:
            current["source_packet_ir_operation_indices_by_unit"] = by_unit
        current["source_selection_ids"] = ordered_unique(
            [
                *_as_list(current.get("source_selection_ids")),
                *_as_list(packet_row.get("source_selection_ids")),
            ]
        )
        packet_blocker_counts = Counter(_counter_dict(current.get("packet_ir_blocker_counts")))
        packet_blocker_counts.update(_counter_dict(packet_row.get("blocker_counts")))
        current["packet_ir_blocker_counts"] = dict(sorted(packet_blocker_counts.items()))
        current.update(FALSE_AUTHORITY)

    ranked_rows = sorted(rows_by_key.values(), key=_backlog_row_sort_key)
    for rank, row in enumerate(ranked_rows, start=1):
        row["backlog_rank"] = rank
        row["implementation_priority_score"] = (
            float(row.get("expected_score_gain_sum") or 0.0)
            + float(row.get("candidate_saved_bytes_sum") or 0) * 1e-9
            + float(row.get("blocked_row_count") or 0) * 1e-6
        )
        row.update(FALSE_AUTHORITY)

    return apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZER_BACKLOG_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "backlog_row_count": len(ranked_rows),
            "packet_ir_lowered_row_count": len(packet_ir_rows),
            "packet_ir_lowered_backlog_key_count": len(packet_ir_keys),
            "rows": ranked_rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            "materializer_backlog_is_planning_only",
            "requires_adapter_implementation_before_queue_dispatch",
        ),
    )


def _high_level_context_packet_ir_rows(
    materializer_backlog: Mapping[str, Any],
    contexts: Mapping[str, Mapping[str, Any]] | None,
    *,
    source_plan_path: str | None,
) -> list[dict[str, Any]]:
    if not contexts:
        return []
    out: list[dict[str, Any]] = []
    for row in _as_list(materializer_backlog.get("rows")):
        if not isinstance(row, Mapping):
            continue
        if (
            row.get("unit_kind") != "scorer_inverse_surface_cell"
            or row.get("operation_family") != INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY
            or row.get("target_kind") != INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND
        ):
            continue
        matches = _context_matches_for_backlog_row(
            contexts,
            row,
            extra_keys=(
                INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
                INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
            ),
        )
        if len(matches) != 1:
            continue
        context = matches[0][1]
        packet_ir = _context_mapping_value(context, "packet_ir_operation_set")
        if packet_ir is None:
            compiler = _context_mapping_value(context, "operation_set_compiler")
            if compiler is None:
                continue
            source_selection_ids = _as_list(row.get("source_selection_ids"))
            candidate_id = (
                str(source_selection_ids[0])
                if source_selection_ids
                else str(row.get("backlog_key") or "")
            )
            try:
                packet_ir = packet_ir_operation_set_from_compiler_hint(
                    compiler,
                    source_backlog_key=str(row.get("backlog_key") or ""),
                    source_unit_ids=[str(item) for item in _as_list(row.get("source_unit_ids"))],
                    candidate_id=candidate_id,
                    source_paths=[source_plan_path] if source_plan_path else [],
                )
            except ValueError as exc:
                raise ExperimentQueueError(
                    f"inverse_action_operation_set_compiler_invalid:{exc}"
                ) from exc
        out.extend(lower_packetir_operation_set_to_backlog_rows(packet_ir, source_backlog_row=row))
    return out


def _backlog_row_sort_key(row: Mapping[str, Any]) -> tuple[float, int, int, str]:
    return (
        -float(row.get("expected_score_gain_sum") or 0.0),
        -int(row.get("candidate_saved_bytes_sum") or 0),
        -int(row.get("blocked_row_count") or 0),
        str(row.get("backlog_key") or ""),
    )


def _merge_operation_params(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value in (None, ""):
            continue
        prior = merged.get(str(key))
        if prior in (None, ""):
            merged[str(key)] = value
            continue
        if prior == value:
            continue
        values = list(prior) if isinstance(prior, list) else [prior]
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
        merged[str(key)] = ordered_unique(str(item) for item in values)
    return merged


def build_materializer_backlog(compiled_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate blocked materialization rows into ranked adapter work orders."""

    rows: dict[str, dict[str, Any]] = {}
    seen_selection_by_key: dict[str, set[str]] = {}
    seen_unit_by_key: dict[str, set[str]] = {}
    for row in compiled_rows:
        if row.get("executable") is True:
            continue
        selection_id = str(row.get("selection_id") or row.get("candidate_id") or "")
        row_saved_bytes = _finite_int(row.get("candidate_saved_bytes")) or 0
        row_expected_gain = _finite_float(row.get("expected_score_gain")) or 0.0
        expected_delta = _finite_float(row.get("expected_delta_score"))
        source_units = [unit for unit in _as_list(row.get("source_units")) if isinstance(unit, Mapping)]
        unit_blockers = _unit_blockers_by_id(source_units)
        source_units_by_id = _units_by_id(source_units)
        resolutions = [item for item in _as_list(row.get("materializer_resolutions")) if isinstance(item, Mapping)]
        gain_share = row_expected_gain / float(max(1, len(resolutions)))
        saved_share = row_saved_bytes // max(1, len(resolutions))
        for resolution in resolutions:
            unit_id = str(resolution.get("unit_id") or "")
            unit_saved_bytes = _finite_int(source_units_by_id.get(unit_id, {}).get("candidate_saved_bytes"))
            saved_bytes = unit_saved_bytes if unit_saved_bytes is not None else saved_share
            resolution_blockers = ordered_unique(
                [
                    *[str(item) for item in _as_list(resolution.get("blockers"))],
                    *[
                        f"selected_operation_blocker:{resolution.get('unit_id') or '<missing>'}:{item}"
                        for item in _as_list(resolution.get("selected_operation_blockers"))
                    ],
                    *[
                        f"selected_unit_blocker:{resolution.get('unit_id') or '<missing>'}:{item}"
                        for item in unit_blockers.get(str(resolution.get("unit_id") or ""), [])
                    ],
                ]
            )
            if not resolution_blockers:
                continue
            gap_class = _resolution_gap_class(resolution, resolution_blockers)
            receiver_contract_status = _receiver_contract_status(resolution, gap_class)
            suggested_materializers = _suggested_materializer_rows(resolution)
            key = _backlog_key(resolution, gap_class)
            current = rows.get(key)
            if current is None:
                current = {
                    "schema": "byte_shaving_materializer_backlog_row.v1",
                    "backlog_key": key,
                    "gap_class": gap_class,
                    "target_kind": resolution.get("target_kind"),
                    "materializer_id": resolution.get("materializer_id"),
                    "receiver_contract_id": resolution.get("receiver_contract_id"),
                    "receiver_contract_kind": resolution.get("receiver_contract_kind"),
                    "receiver_contract_status": receiver_contract_status,
                    "cooperative_receiver_required": bool(resolution.get("cooperative_receiver_required")),
                    "materialization_resource_kind": resolution.get("materialization_resource_kind"),
                    "suggested_materializer_count": len(suggested_materializers),
                    "suggested_materializers": suggested_materializers,
                    "unit_kind": resolution.get("unit_kind"),
                    "operation_family": resolution.get("operation_family"),
                    "blocked_row_count": 0,
                    "blocked_resolution_count": 0,
                    "selected_operation_count": 0,
                    "affected_unit_count": 0,
                    "candidate_saved_bytes_sum": 0,
                    "expected_score_gain_sum": 0.0,
                    "best_expected_score_gain": None,
                    "best_expected_delta_score": None,
                    "best_candidate_saved_bytes": 0,
                    "blocker_counts": {},
                    "source_unit_ids": [],
                    "source_selection_ids": [],
                    "source_selection_samples": [],
                    "source_packet_ir_operation_indices": [],
                    "source_packet_ir_operation_indices_by_unit": {},
                    "operation_params": {},
                    "source_operation_params_by_unit": {},
                    **FALSE_AUTHORITY,
                }
                rows[key] = current
                seen_selection_by_key[key] = set()
                seen_unit_by_key[key] = set()
            current["blocked_resolution_count"] = int(current["blocked_resolution_count"]) + 1
            current["selected_operation_count"] = int(current["selected_operation_count"]) + 1
            current["candidate_saved_bytes_sum"] = int(current["candidate_saved_bytes_sum"]) + saved_bytes
            current["expected_score_gain_sum"] = float(current["expected_score_gain_sum"]) + gain_share
            best_gain = _finite_float(current.get("best_expected_score_gain"))
            if best_gain is None or gain_share > best_gain:
                current["best_expected_score_gain"] = gain_share
                current["best_expected_delta_score"] = expected_delta
                current["best_candidate_saved_bytes"] = saved_bytes
            blocker_counts = Counter(current["blocker_counts"])
            blocker_counts.update(resolution_blockers)
            current["blocker_counts"] = dict(sorted(blocker_counts.items()))
            if unit_id and unit_id not in seen_unit_by_key[key]:
                seen_unit_by_key[key].add(unit_id)
                current["source_unit_ids"].append(unit_id)
                current["affected_unit_count"] = len(seen_unit_by_key[key])
            operation_index = _finite_int(resolution.get("packet_ir_operation_index"))
            if operation_index is not None:
                indices = list(_as_list(current.get("source_packet_ir_operation_indices")))
                indices.append(operation_index)
                current["source_packet_ir_operation_indices"] = sorted(
                    {
                        int(index)
                        for index in indices
                        if _finite_int(index) is not None
                    }
                )
                if unit_id:
                    by_unit = dict(_as_mapping(current.get("source_packet_ir_operation_indices_by_unit")))
                    by_unit[unit_id] = operation_index
                    current["source_packet_ir_operation_indices_by_unit"] = by_unit
            source_unit = source_units_by_id.get(unit_id, {})
            operation_params = _as_mapping(source_unit.get("operation_params"))
            if operation_params:
                current["source_operation_params_by_unit"][unit_id] = dict(operation_params)
                current["operation_params"] = _merge_operation_params(
                    _as_mapping(current.get("operation_params")),
                    operation_params,
                )
            if selection_id and selection_id not in seen_selection_by_key[key]:
                seen_selection_by_key[key].add(selection_id)
                current["source_selection_ids"].append(selection_id)
                current["blocked_row_count"] = len(seen_selection_by_key[key])
            samples = current["source_selection_samples"]
            if len(samples) < 8:
                samples.append(
                    {
                        "selection_id": selection_id,
                        "selection_kind": row.get("selection_kind"),
                        "unit_id": unit_id,
                        "candidate_saved_bytes": saved_bytes,
                        "expected_score_gain": gain_share,
                        "expected_delta_score": expected_delta,
                    }
                )

    ranked_rows = sorted(rows.values(), key=_backlog_row_sort_key)
    for rank, backlog_row in enumerate(ranked_rows, start=1):
        backlog_row["backlog_rank"] = rank
        backlog_row["implementation_priority_score"] = (
            float(backlog_row["expected_score_gain_sum"])
            + float(backlog_row["candidate_saved_bytes_sum"]) * 1e-9
            + float(backlog_row["blocked_row_count"]) * 1e-6
        )
    return apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZER_BACKLOG_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "backlog_row_count": len(ranked_rows),
            "rows": ranked_rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            "materializer_backlog_is_planning_only",
            "requires_adapter_implementation_before_queue_dispatch",
        ),
    )


def summarize_materializer_backlog(backlog: Mapping[str, Any], *, limit: int = 8) -> dict[str, Any]:
    rows = [item for item in _as_list(backlog.get("rows")) if isinstance(item, Mapping)]
    top_rows = rows[: max(0, limit)]
    return apply_proxy_evidence_boundary(
        {
            "schema": "byte_shaving_materializer_backlog_summary.v1",
            "source_schema": backlog.get("schema"),
            "backlog_row_count": len(rows),
            "top_backlog_rows": [
                {
                    "backlog_rank": row.get("backlog_rank"),
                    "backlog_key": row.get("backlog_key"),
                    "gap_class": row.get("gap_class"),
                    "unit_kind": row.get("unit_kind"),
                    "operation_family": row.get("operation_family"),
                    "target_kind": row.get("target_kind"),
                    "materializer_id": row.get("materializer_id"),
                    "receiver_contract_id": row.get("receiver_contract_id"),
                    "receiver_contract_kind": row.get("receiver_contract_kind"),
                    "receiver_contract_status": row.get("receiver_contract_status"),
                    "cooperative_receiver_required": row.get("cooperative_receiver_required"),
                    "materialization_resource_kind": row.get("materialization_resource_kind"),
                    "suggested_materializer_count": row.get("suggested_materializer_count"),
                    "suggested_materializers": row.get("suggested_materializers"),
                    "blocked_row_count": row.get("blocked_row_count"),
                    "blocked_resolution_count": row.get("blocked_resolution_count"),
                    "selected_operation_count": row.get("selected_operation_count"),
                    "affected_unit_count": row.get("affected_unit_count"),
                    "packet_ir_lowered_row_count": row.get("packet_ir_lowered_row_count"),
                    "source_packet_ir_operation_set_ids": row.get("source_packet_ir_operation_set_ids"),
                    "source_packet_ir_source_operation_set_ids": row.get("source_packet_ir_source_operation_set_ids"),
                    "packet_ir_blocker_counts": row.get("packet_ir_blocker_counts"),
                    "candidate_saved_bytes_sum": row.get("candidate_saved_bytes_sum"),
                    "expected_score_gain_sum": row.get("expected_score_gain_sum"),
                    "implementation_priority_score": row.get("implementation_priority_score"),
                }
                for row in top_rows
            ],
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=("materializer_backlog_summary_is_planning_only",),
    )


def _materializer_work_id(backlog_key: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", backlog_key.lower()).strip("_")
    return f"materializer_work_{safe or 'row'}"


def _grouped_archive_state_request_id(packet_ir_operation_set_id: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", packet_ir_operation_set_id.lower()).strip("_")
    return f"grouped_archive_state_{safe or 'packetir'}"


def _work_row_packet_ir_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    indices = [
        parsed
        for parsed in (_finite_int(item) for item in _as_list(row.get("source_packet_ir_operation_indices")))
        if parsed is not None
    ]
    first_index = min(indices) if indices else 1_000_000
    work_rank = _finite_int(row.get("work_rank")) or 1_000_000
    return first_index, work_rank, str(row.get("work_id") or row.get("backlog_key") or "")


def _grouped_archive_state_materializer_requests(
    work_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_packet_ir: dict[str, list[Mapping[str, Any]]] = {}
    for row in work_rows:
        for packet_ir_id in _as_list(row.get("source_packet_ir_operation_set_ids")):
            packet_ir_text = str(packet_ir_id or "").strip()
            if packet_ir_text:
                by_packet_ir.setdefault(packet_ir_text, []).append(row)

    requests: list[dict[str, Any]] = []
    for packet_ir_id in sorted(by_packet_ir):
        rows = sorted(by_packet_ir[packet_ir_id], key=_work_row_packet_ir_sort_key)
        if len(rows) < 2:
            continue
        blockers: list[str] = []
        ordered_work_ids: list[str] = []
        operation_indices: list[int] = []
        operation_rows: list[dict[str, Any]] = []
        source_packet_ir_source_ids: list[str] = []
        for row in rows:
            work_id = str(row.get("work_id") or "").strip()
            if not work_id:
                blockers.append("grouped_archive_state_member_work_id_missing")
                work_id = str(row.get("backlog_key") or "<missing>")
            ordered_work_ids.append(work_id)
            target_kind = str(row.get("target_kind") or "")
            if target_kind not in GROUPED_ARCHIVE_STATE_SUPPORTED_TARGET_KINDS:
                blockers.append(f"grouped_archive_state_target_not_supported:{target_kind or '<missing>'}")
            if row.get("executable") is not True:
                blockers.append(f"grouped_archive_state_member_not_executable:{work_id}")
            command = row.get("command")
            if not isinstance(command, list) or not command:
                blockers.append(f"grouped_archive_state_member_command_missing:{work_id}")
            indices = [
                parsed
                for parsed in (
                    _finite_int(item)
                    for item in _as_list(row.get("source_packet_ir_operation_indices"))
                )
                if parsed is not None
            ]
            if not indices:
                blockers.append(f"grouped_archive_state_member_order_missing:{work_id}")
            operation_indices.extend(indices)
            source_packet_ir_source_ids.extend(
                str(item)
                for item in _as_list(row.get("source_packet_ir_source_operation_set_ids"))
                if str(item).strip()
            )
            operation_rows.append(
                {
                    "work_id": work_id,
                    "work_rank": row.get("work_rank"),
                    "target_kind": target_kind or None,
                    "materializer_id": row.get("materializer_id"),
                    "unit_kind": row.get("unit_kind"),
                    "operation_family": row.get("operation_family"),
                    "source_unit_ids": _as_list(row.get("source_unit_ids")),
                    "source_packet_ir_operation_indices": indices,
                    "tool": row.get("tool"),
                    "resource_kind": row.get("resource_kind") or "local_cpu",
                    **FALSE_AUTHORITY,
                }
            )

        ordered_indices = sorted(set(operation_indices))
        if ordered_indices and ordered_indices != list(range(min(ordered_indices), max(ordered_indices) + 1)):
            blockers.append("grouped_archive_state_operation_order_has_gap")
        executable = not blockers
        requests.append(
            apply_proxy_evidence_boundary(
                {
                    "schema": GROUPED_ARCHIVE_STATE_MATERIALIZER_REQUEST_SCHEMA,
                    "request_id": _grouped_archive_state_request_id(packet_ir_id),
                    "source_packet_ir_operation_set_id": packet_ir_id,
                    "source_packet_ir_source_operation_set_ids": ordered_unique(
                        source_packet_ir_source_ids
                    ),
                    "operation_count": len(rows),
                    "ordered_work_ids": ordered_work_ids,
                    "source_packet_ir_operation_indices": ordered_indices,
                    "target_kinds": ordered_unique(
                        str(row.get("target_kind") or "") for row in rows
                    ),
                    "operation_rows": operation_rows,
                    "tool": GROUPED_ARCHIVE_STATE_MATERIALIZER_TOOL,
                    "resource_kind": "local_cpu",
                    "executable": executable,
                    "grouped_execution_ready": executable,
                    "grouped_execution_blockers": ordered_unique(blockers),
                    **FALSE_AUTHORITY,
                },
                dispatch_blockers=(
                    (
                        "grouped_archive_state_materializer_local_proof_chain_only",
                        "exact_auth_eval_required_before_score_claim",
                    )
                    if executable
                    else ordered_unique(blockers)
                ),
            )
        )
    return requests


def _first_suggested_materializer(row: Mapping[str, Any]) -> Mapping[str, Any]:
    suggestions = [item for item in _as_list(row.get("suggested_materializers")) if isinstance(item, Mapping)]
    return suggestions[0] if suggestions else {}


def _context_for_backlog_row(
    contexts: Mapping[str, Mapping[str, Any]],
    row: Mapping[str, Any],
    *,
    extra_keys: Sequence[str] = (),
) -> Mapping[str, Any]:
    keys = [
        str(row.get("backlog_key") or ""),
        str(row.get("materializer_id") or ""),
        str(row.get("target_kind") or ""),
    ]
    keys.extend(str(item) for item in _as_list(row.get("source_unit_ids")))
    keys.extend(str(key) for key in extra_keys if str(key))
    for key in ordered_unique(keys):
        context = contexts.get(key)
        if isinstance(context, Mapping):
            return context
    return {}


def _context_matches_for_backlog_row(
    contexts: Mapping[str, Mapping[str, Any]],
    row: Mapping[str, Any],
    *,
    extra_keys: Sequence[str] = (),
) -> list[tuple[str, Mapping[str, Any]]]:
    keys = [
        str(row.get("backlog_key") or ""),
        str(row.get("materializer_id") or ""),
        str(row.get("target_kind") or ""),
    ]
    keys.extend(str(item) for item in _as_list(row.get("source_unit_ids")))
    keys.extend(str(key) for key in extra_keys if str(key))
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for key in ordered_unique(keys):
        context = contexts.get(key)
        if not isinstance(context, Mapping):
            continue
        context_dict = dict(context)
        if any(dict(existing) == context_dict for _existing_key, existing in matches):
            continue
        matches.append((key, context))
    return matches


def _context_value_present(context: Mapping[str, Any], key: str) -> bool:
    value = context.get(key)
    if isinstance(value, Path):
        return bool(value.as_posix())
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return bool(value)
    return value is not None


def _merged_context_missing_blocker_resolved(
    context: Mapping[str, Any],
    blocker: str,
) -> bool:
    prefix = "materializer_context_missing:"
    if not blocker.startswith(prefix):
        return False
    key = blocker[len(prefix) :]
    if key == "factorization_contract_or_rank":
        return _path_context_value(context, "factorization_contract") is not None or _finite_int(
            context.get("rank")
        ) is not None
    aliases = {
        "archive_path": ("archive_path", "source_archive"),
        "output_manifest": ("output_manifest", "manifest_out", "json_out"),
    }
    return any(_context_value_present(context, alias) for alias in aliases.get(key, (key,)))


def _merge_context_matches(
    matches: Sequence[tuple[str, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    """Compose generic context hints with row-specific materializer paths.

    Context lookup intentionally accepts broad keys such as materializer id and
    target kind, while operators often add the executable archive paths under a
    row-specific backlog key.  The row-specific match appears first in
    ``matches``; apply broader hints first, then let specific rows override
    them.
    """

    merged: dict[str, Any] = {}
    blockers: list[str] = []
    for _key, context in reversed(matches):
        blockers.extend(_string_list_context_value(context, "context_blockers"))
        for field, value in context.items():
            if field == "context_blockers":
                continue
            merged[field] = value
    blockers = ordered_unique(
        blocker
        for blocker in blockers
        if not _merged_context_missing_blocker_resolved(merged, blocker)
    )
    if blockers:
        merged["context_blockers"] = blockers
    else:
        merged.pop("context_blockers", None)
    return merged


def _context_matches_are_composable(
    matches: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    row: Mapping[str, Any],
    materializer_id: str,
    target_kind: str,
) -> bool:
    if len(matches) < 2:
        return False
    keys = [key for key, _context in matches]
    backlog_key = str(row.get("backlog_key") or "")
    if backlog_key not in keys:
        return False
    composable_keys = {
        backlog_key,
        str(row.get("materializer_id") or ""),
        str(row.get("target_kind") or ""),
        materializer_id,
        target_kind,
    }
    composable_keys.discard("")
    return all(key in composable_keys for key in keys)


def _path_context_value(context: Mapping[str, Any], key: str) -> str | None:
    value = context.get(key)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str) and value.strip():
        return value
    return None


def _path_list_context_value(context: Mapping[str, Any], key: str) -> list[str]:
    value = context.get(key)
    if isinstance(value, (str, Path)):
        item = _path_context_value(context, key)
        return [] if item is None else [item]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: list[str] = []
        for item in value:
            if isinstance(item, Path):
                out.append(item.as_posix())
            elif isinstance(item, str) and item.strip():
                out.append(item)
        return out
    return []


def _string_list_context_value(context: Mapping[str, Any], key: str) -> list[str]:
    value = context.get(key)
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return ordered_unique(str(item).strip() for item in value if str(item).strip())
    return []


def _context_string_any(
    context: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _context_mapping_value(context: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = context.get(key)
    return value if isinstance(value, Mapping) else None


def _renderer_payload_dfl1_parity_context(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    source_runtime = _context_string_any(
        context,
        (
            "renderer_payload_dfl1_source_runtime_dir",
            "renderer_payload_dfl1_inflate_runtime_dir",
            "source_runtime_dir",
            "inflate_runtime_dir",
        ),
    )
    candidate_runtime = _context_string_any(
        context,
        (
            "renderer_payload_dfl1_candidate_runtime_dir",
            "candidate_runtime_dir",
        ),
    )
    file_list = _context_string_any(
        context,
        (
            "renderer_payload_dfl1_full_frame_file_list",
            "full_frame_file_list",
            "inflate_file_list",
            "file_list",
        ),
    )
    file_list_entries: list[str] = []
    for key in (
        "renderer_payload_dfl1_full_frame_file_list_entries",
        "full_frame_file_list_entries",
        "file_list_entries",
        "file_list_entry",
    ):
        file_list_entries.extend(_string_list_context_value(context, key))
    output_dir = _context_string_any(
        context,
        (
            "renderer_payload_dfl1_inflate_parity_output_dir",
            "full_frame_inflate_parity_output_dir",
            "inflate_parity_output_dir",
        ),
    )
    expected_file_list_sha = _context_string_any(
        context,
        (
            "renderer_payload_dfl1_expected_full_frame_file_list_sha256",
            "expected_full_frame_file_list_sha256",
        ),
    )
    expected_entry_count = _finite_int(
        context.get("renderer_payload_dfl1_expected_full_frame_entry_count")
    )
    if expected_entry_count is None:
        expected_entry_count = _finite_int(
            context.get("expected_full_frame_entry_count")
        )
    file_list_source = _context_string_any(
        context,
        (
            "renderer_payload_dfl1_full_frame_file_list_source",
            "full_frame_file_list_source",
        ),
    )
    archive_path = _path_context_value(context, "archive_path")
    output_archive = _path_context_value(context, "output_archive")
    out: dict[str, Any] = {}
    for key, value in (
        ("source_archive", archive_path),
        ("candidate_archive", output_archive),
        ("source_runtime_dir", source_runtime),
        ("candidate_runtime_dir", candidate_runtime),
        ("file_list", file_list),
        ("output_dir", output_dir),
        ("expected_full_frame_file_list_sha256", expected_file_list_sha),
        ("full_frame_file_list_source", file_list_source),
    ):
        if value is not None:
            out[key] = value
    if expected_entry_count is not None:
        out["expected_full_frame_entry_count"] = expected_entry_count
    if file_list_entries:
        out["file_list_entries"] = ordered_unique(file_list_entries)
    return out


def _renderer_payload_dfl1_parity_followup_blockers(
    work_row: Mapping[str, Any],
) -> list[str]:
    if work_row.get("target_kind") != RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        return []
    context = work_row.get("renderer_payload_dfl1_parity_context")
    if not isinstance(context, Mapping):
        return ["renderer_payload_dfl1_parity_context_missing"]
    blockers: list[str] = []
    for key in (
        "source_archive",
        "candidate_archive",
        "source_runtime_dir",
        "candidate_runtime_dir",
    ):
        if _context_string_any(context, (key,)) is None:
            blockers.append(f"renderer_payload_dfl1_parity_context_missing:{key}")
    file_list = _context_string_any(context, ("file_list",))
    file_list_entries = _string_list_context_value(context, "file_list_entries")
    if file_list is None and not file_list_entries:
        blockers.append(
            "renderer_payload_dfl1_parity_context_missing:file_list_or_entries"
        )
    expected_file_list_sha = _context_string_any(
        context,
        ("expected_full_frame_file_list_sha256",),
    )
    if (
        expected_file_list_sha is None
        or len(expected_file_list_sha) != 64
        or any(char not in "0123456789abcdef" for char in expected_file_list_sha)
    ):
        blockers.append(
            "renderer_payload_dfl1_parity_context_missing:expected_full_frame_file_list_sha256"
        )
    expected_entry_count = _finite_int(context.get("expected_full_frame_entry_count"))
    if expected_entry_count is None or expected_entry_count < 1:
        blockers.append(
            "renderer_payload_dfl1_parity_context_missing:expected_full_frame_entry_count"
        )
    if _context_string_any(context, ("full_frame_file_list_source",)) is None:
        blockers.append(
            "renderer_payload_dfl1_parity_context_missing:full_frame_file_list_source"
        )
    return ordered_unique(blockers)


def _command_flag_values(command: Sequence[str], flags: set[str]) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(command) - 1:
        if command[index] in flags:
            values.append(command[index + 1])
            index += 2
        else:
            index += 1
    return ordered_unique(values)


def _context_keys_from_row(row: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    raw_keys = row.get("context_keys")
    if isinstance(raw_keys, str):
        keys.append(raw_keys)
    elif isinstance(raw_keys, Sequence) and not isinstance(raw_keys, (bytes, bytearray)):
        keys.extend(str(item) for item in raw_keys if str(item))
    for key in ("backlog_key", "materializer_id", "target_kind", "source_unit_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            keys.append(value)
    source_unit_ids = row.get("source_unit_ids")
    if isinstance(source_unit_ids, str):
        keys.append(source_unit_ids)
    elif isinstance(source_unit_ids, Sequence) and not isinstance(source_unit_ids, (bytes, bytearray)):
        keys.extend(str(item) for item in source_unit_ids if str(item))
    return ordered_unique(keys)


def materializer_contexts_from_payload(
    payload: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Parse a durable materializer-context file into lookup keys.

    Context rows are keyed by backlog key, materializer id, target kind, or
    source unit id. This keeps operator-authored context files small while still
    allowing the work-queue compiler to match future backlog rows without
    hand-written Python glue.
    """

    if payload.get("schema") != MATERIALIZER_CONTEXTS_SCHEMA:
        raise ExperimentQueueError(f"expected schema {MATERIALIZER_CONTEXTS_SCHEMA}")
    contexts: dict[str, Mapping[str, Any]] = {}
    mapping_contexts = payload.get("contexts")
    if isinstance(mapping_contexts, Mapping):
        for key, context in mapping_contexts.items():
            if not isinstance(context, Mapping):
                raise ExperimentQueueError(f"context {key!r} must be an object")
            try:
                require_no_truthy_authority_fields(
                    context,
                    context=f"materializer_contexts.contexts.{key}",
                )
            except ValueError as exc:
                raise ExperimentQueueError(str(exc)) from exc
            contexts[str(key)] = dict(context)

    for index, row in enumerate(_as_list(payload.get("rows"))):
        if not isinstance(row, Mapping):
            raise ExperimentQueueError(f"materializer context row {index} must be an object")
        context = row.get("context")
        if not isinstance(context, Mapping):
            raise ExperimentQueueError(f"materializer context row {index} must include object field 'context'")
        try:
            require_no_truthy_authority_fields(
                context,
                context=f"materializer_contexts.rows.{index}.context",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        keys = _context_keys_from_row(row)
        if not keys:
            raise ExperimentQueueError(f"materializer context row {index} must declare at least one key")
        for key in keys:
            contexts[key] = dict(context)

    if not contexts:
        raise ExperimentQueueError("materializer context payload contains no contexts")
    return contexts


def _byte_range_chain_command(context: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    schema_manifest = _path_context_value(context, "schema_manifest")
    if schema_manifest is None:
        blockers.append("materializer_context_missing:schema_manifest")
    beam_probe_reports = _path_list_context_value(context, "beam_probe_reports")
    if not beam_probe_reports:
        blockers.append("materializer_context_missing:beam_probe_reports")
    source_runtime_dir = _path_context_value(context, "source_runtime_dir")
    if source_runtime_dir is None:
        blockers.append("materializer_context_missing:source_runtime_dir")
    output_dir = _path_context_value(context, "output_dir")
    if output_dir is None:
        blockers.append("materializer_context_missing:output_dir")
    if blockers:
        return [], blockers

    assert schema_manifest is not None
    assert source_runtime_dir is not None
    assert output_dir is not None
    command = [
        ".venv/bin/python",
        BYTE_RANGE_CHAIN_TOOL,
        "--schema-manifest",
        schema_manifest,
        "--source-runtime-dir",
        source_runtime_dir,
        "--output-dir",
        output_dir,
    ]
    for report in beam_probe_reports:
        command.extend(["--beam-probe-report", report])
    optional_path_flags = (
        ("global_combo_report", "--global-combo-report"),
        ("source_archive", "--source-archive"),
    )
    for key, flag in optional_path_flags:
        value = _path_context_value(context, key)
        if value is not None:
            command.extend([flag, value])
    member_name = context.get("member_name")
    if isinstance(member_name, str) and member_name.strip():
        command.extend(["--member-name", member_name])
    retune_section = context.get("retune_brotli_section")
    if isinstance(retune_section, str) and retune_section.strip():
        command.extend(["--retune-brotli-section", retune_section])
    min_free_bytes = _finite_int(context.get("min_free_bytes"))
    if min_free_bytes is not None:
        command.extend(["--min-free-bytes", str(min_free_bytes)])
    if context.get("fail_if_receiver_blocked") is True:
        command.append("--fail-if-receiver-blocked")
    return command, []


def _inverse_scorer_action_functional_command(
    context: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    blockers: list[str] = []
    output = _path_context_value(context, "output")
    if output is None:
        blockers.append("materializer_context_missing:output")
    scorer_responses = _path_list_context_value(context, "scorer_response")
    scorer_responses.extend(_path_list_context_value(context, "scorer_responses"))
    inverse_surfaces = _path_list_context_value(context, "inverse_scorer_surface")
    inverse_surfaces.extend(_path_list_context_value(context, "inverse_scorer_surfaces"))
    byte_shaving_surfaces = _path_list_context_value(
        context,
        "byte_shaving_signal_surface",
    )
    byte_shaving_surfaces.extend(_path_list_context_value(context, "byte_shaving_signal_surfaces"))
    byte_shaving_plans = _path_list_context_value(
        context,
        "byte_shaving_campaign_plan",
    )
    byte_shaving_plans.extend(_path_list_context_value(context, "byte_shaving_campaign_plans"))
    if not scorer_responses and not inverse_surfaces and not byte_shaving_surfaces and not byte_shaving_plans:
        blockers.append("materializer_context_missing:inverse_action_source_surface")
    if blockers:
        return [], blockers, {}

    assert output is not None
    command = [
        ".venv/bin/python",
        INVERSE_ACTION_FUNCTIONAL_TOOL,
        "--output",
        output,
    ]
    for path in scorer_responses:
        command.extend(["--scorer-response", path])
    for path in inverse_surfaces:
        command.extend(["--inverse-scorer-surface", path])
    for path in byte_shaving_surfaces:
        command.extend(["--byte-shaving-signal-surface", path])
    for path in byte_shaving_plans:
        command.extend(["--byte-shaving-campaign-plan", path])

    input_paths = [
        *scorer_responses,
        *inverse_surfaces,
        *byte_shaving_surfaces,
        *byte_shaving_plans,
    ]
    optional_path_flags = (
        ("md_out", "--md-out"),
        ("queue_performance_runtime_identity", "--queue-performance-runtime-identity"),
        ("queue_performance_cache_identity", "--queue-performance-cache-identity"),
        ("queue_performance_candidate_map", "--queue-performance-candidate-map"),
    )
    for key, flag in optional_path_flags:
        value = _path_context_value(context, key)
        if value is not None:
            command.extend([flag, value])
            if key != "md_out":
                input_paths.append(value)
    for path in _path_list_context_value(context, "queue_performance_summary"):
        command.extend(["--queue-performance-summary", path])
        input_paths.append(path)

    optional_text_flags = (
        ("candidate_id", "--candidate-id"),
        ("resource_kind", "--resource-kind"),
        ("queue_performance_axis", "--queue-performance-axis"),
    )
    for key, flag in optional_text_flags:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            command.extend([flag, value])
    optional_int_flags = (
        ("artifact_bytes", "--artifact-bytes"),
        ("total_byte_budget", "--total-byte-budget"),
        ("inverse_scorer_max_units", "--inverse-scorer-max-units"),
    )
    for key, flag in optional_int_flags:
        value = _finite_int(context.get(key))
        if value is not None:
            command.extend([flag, str(value)])
    optional_float_flags = (
        ("elapsed_seconds", "--elapsed-seconds"),
        ("lambda_rate", "--lambda-rate"),
        ("inverse_scorer_null_delta_epsilon", "--inverse-scorer-null-delta-epsilon"),
        (
            "inverse_scorer_fragile_delta_threshold",
            "--inverse-scorer-fragile-delta-threshold",
        ),
    )
    for key, flag in optional_float_flags:
        value = _finite_float(context.get(key))
        if value is not None:
            command.extend([flag, str(value)])
    if context.get("inverse_scorer_allow_native_mlx_window_objective") is True:
        command.append("--inverse-scorer-allow-native-mlx-window-objective")

    telemetry_paths = [output]
    md_out = _path_context_value(context, "md_out")
    if md_out is not None:
        telemetry_paths.append(md_out)
    return (
        command,
        [],
        {
            "artifact_paths": telemetry_paths,
            "input_artifact_paths": input_paths,
            "pullback_artifact_paths": telemetry_paths,
            "include_postcondition_paths": True,
        },
    )


def _inverse_scorer_cell_candidate_command(
    context: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[list[str], list[str], dict[str, Any]]:
    blockers: list[str] = []
    template = _path_context_value(context, "candidate_archive_template")
    if template is None:
        blockers.append("materializer_context_missing:candidate_archive_template")
    else:
        blockers.extend(_strict_single_member_zip_blockers(template, repo=repo_root))
    action_functional = _path_context_value(context, "inverse_action_functional")
    if action_functional is None:
        blockers.append("materializer_context_missing:inverse_action_functional")
    raw_digest = context.get("raw_contest_video_digest")
    if not isinstance(raw_digest, str) or not raw_digest.strip():
        blockers.append("materializer_context_missing:raw_contest_video_digest")
    output_dir = _path_context_value(context, "chain_output_dir")
    if output_dir is None:
        output_dir = _path_context_value(context, "output_dir")
    output_archive = _path_context_value(context, "output_archive")
    if output_dir is None and output_archive is None:
        blockers.append("materializer_context_missing:output_archive")
    manifest_out = _path_context_value(context, "manifest_out")
    if output_dir is None and manifest_out is None:
        blockers.append("materializer_context_missing:manifest_out")
    source_inflate_output_dir = _path_context_value(context, "source_inflate_output_dir")
    candidate_inflate_output_dir = _path_context_value(
        context,
        "candidate_inflate_output_dir",
    )
    inflate_runtime_dir = _path_context_value(context, "inflate_runtime_dir")
    has_precomputed_parity_context = source_inflate_output_dir is not None and candidate_inflate_output_dir is not None
    has_partial_precomputed_parity_context = (source_inflate_output_dir is None) != (
        candidate_inflate_output_dir is None
    )
    has_runtime_parity_context = inflate_runtime_dir is not None
    exact_chain_requires_inflate_parity = output_dir is not None and context.get("descriptor_probe_only") is not True
    explicit_fail_if_parity_blocked = context.get("fail_if_inflate_parity_blocked") is True
    if output_dir is not None and has_partial_precomputed_parity_context:
        blockers.append("inverse_scorer_cell_inflate_parity_requires_source_and_candidate_output_dirs")
    if (exact_chain_requires_inflate_parity or explicit_fail_if_parity_blocked) and not (
        has_precomputed_parity_context or has_runtime_parity_context
    ):
        blockers.append("inverse_scorer_cell_exact_chain_requires_inflate_parity_context")
    if blockers:
        return [], blockers, {}

    assert template is not None
    assert action_functional is not None
    assert isinstance(raw_digest, str)
    input_paths = [template, action_functional]
    if output_dir is not None:
        command = [
            ".venv/bin/python",
            INVERSE_SCORER_CELL_CHAIN_TOOL,
            "--candidate-archive-template",
            template,
            "--inverse-action-functional",
            action_functional,
            "--raw-contest-video-digest",
            raw_digest,
            "--output-dir",
            output_dir,
        ]
        min_free_bytes = _finite_int(context.get("min_free_bytes"))
        if min_free_bytes is not None:
            command.extend(["--min-free-bytes", str(min_free_bytes)])
        if source_inflate_output_dir is not None:
            command.extend(["--source-inflate-output-dir", source_inflate_output_dir])
            input_paths.append(source_inflate_output_dir)
        if candidate_inflate_output_dir is not None:
            command.extend(["--candidate-inflate-output-dir", candidate_inflate_output_dir])
            input_paths.append(candidate_inflate_output_dir)
        if inflate_runtime_dir is not None:
            command.extend(["--inflate-runtime-dir", inflate_runtime_dir])
            input_paths.append(inflate_runtime_dir)
        source_archive_for_parity = _path_context_value(context, "source_archive_for_parity")
        if source_archive_for_parity is not None:
            command.extend(["--source-archive-for-parity", source_archive_for_parity])
            input_paths.append(source_archive_for_parity)
        inflate_timeout_seconds = _finite_int(context.get("inflate_timeout_seconds"))
        if inflate_timeout_seconds is not None:
            command.extend(["--inflate-timeout-seconds", str(inflate_timeout_seconds)])
        inflate_work_dir = _path_context_value(context, "inflate_work_dir")
        if inflate_work_dir is not None:
            command.extend(["--inflate-work-dir", inflate_work_dir])
        if context.get("keep_inflate_work_dir") is True:
            command.append("--keep-inflate-work-dir")
        if context.get("fail_if_receiver_blocked") is True:
            command.append("--fail-if-receiver-blocked")
        if exact_chain_requires_inflate_parity or explicit_fail_if_parity_blocked:
            command.append("--fail-if-inflate-parity-blocked")
    else:
        assert output_archive is not None
        assert manifest_out is not None
        command = [
            ".venv/bin/python",
            INVERSE_SCORER_CELL_TOOL,
            "--candidate-archive-template",
            template,
            "--inverse-action-functional",
            action_functional,
            "--raw-contest-video-digest",
            raw_digest,
            "--output-archive",
            output_archive,
            "--manifest-out",
            manifest_out,
        ]
        runtime_proof = _path_context_value(context, "runtime_consumption_proof")
        if runtime_proof is not None:
            command.extend(["--runtime-consumption-proof", runtime_proof])
            input_paths.append(runtime_proof)
    for atom_id in _path_list_context_value(context, "atom_id"):
        command.extend(["--atom-id", atom_id])
    for atom_id in _path_list_context_value(context, "atom_ids"):
        command.extend(["--atom-id", atom_id])
    selected_limit = _finite_int(context.get("selected_limit"))
    if selected_limit is not None:
        command.extend(["--selected-limit", str(selected_limit)])
    if output_dir is None and context.get("allow_overwrite") is True:
        command.append("--allow-overwrite")
        expected_output_sha = context.get("expected_output_sha256")
        if isinstance(expected_output_sha, str) and expected_output_sha.strip():
            command.extend(["--expected-output-sha256", expected_output_sha])
        expected_manifest_sha = context.get("expected_manifest_sha256")
        if isinstance(expected_manifest_sha, str) and expected_manifest_sha.strip():
            command.extend(["--expected-manifest-sha256", expected_manifest_sha])
    if output_dir is not None:
        artifact_paths = [output_dir]
        for optional_path in (
            _path_context_value(context, "source_inflate_output_dir"),
            _path_context_value(context, "candidate_inflate_output_dir"),
            _path_context_value(context, "inflate_runtime_dir"),
            _path_context_value(context, "inflate_work_dir"),
        ):
            if optional_path is not None:
                artifact_paths.append(optional_path)
        return (
            command,
            [],
            {
                "artifact_paths": artifact_paths,
                "input_artifact_paths": input_paths,
                "pullback_artifact_paths": [output_dir],
                "pullback_recursive": True,
                "pullback_max_recursive_entries": 512,
                "recursive": True,
                "max_recursive_entries": 512,
                "include_postcondition_paths": True,
                "parity_probe_required": (
                    exact_chain_requires_inflate_parity
                    or explicit_fail_if_parity_blocked
                    or has_precomputed_parity_context
                    or has_runtime_parity_context
                ),
            },
        )
    return (
        command,
        [],
        {
            "artifact_paths": [output_archive, manifest_out],
            "input_artifact_paths": input_paths,
            "pullback_artifact_paths": [output_archive, manifest_out],
            "include_postcondition_paths": True,
        },
    )


def _strict_single_member_zip_blockers(path: str, *, repo: Path) -> list[str]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo / candidate
    if not candidate.exists():
        return ["candidate_archive_template_missing"]
    if candidate.is_symlink():
        return ["candidate_archive_template_is_symlink"]
    try:
        read_strict_single_member_zip(candidate)
    except (HnervLowlevelPackError, OSError) as exc:
        return [f"candidate_archive_template_invalid_strict_single_member_zip:{exc}"]
    return []


def _family_agnostic_materializer_command(
    context: Mapping[str, Any],
    *,
    target_kind: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    blockers: list[str] = _string_list_context_value(context, "context_blockers")
    archive_path = _path_context_value(context, "archive_path")
    if archive_path is None:
        archive_path = _path_context_value(context, "source_archive")
    if archive_path is None:
        blockers.append("materializer_context_missing:archive_path")
    output_archive = _path_context_value(context, "output_archive")
    if output_archive is None:
        blockers.append("materializer_context_missing:output_archive")
    output_manifest = _path_context_value(context, "output_manifest")
    if output_manifest is None:
        output_manifest = _path_context_value(context, "manifest_out")
    if output_manifest is None:
        output_manifest = _path_context_value(context, "json_out")
    if output_manifest is None and output_archive is not None:
        output_manifest = Path(output_archive).with_suffix(".json").as_posix()

    input_paths: list[str] = []
    if archive_path is not None:
        input_paths.append(archive_path)
    if target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND:
        section_manifest = _path_context_value(context, "section_manifest")
        if section_manifest is None:
            blockers.append("materializer_context_missing:section_manifest")
        else:
            input_paths.append(section_manifest)
    elif target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            input_paths.append(packet_member_manifest)
    elif target_kind == PACKET_MEMBER_MERGE_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            input_paths.append(packet_member_manifest)
        merge_contract = _path_context_value(context, "merge_contract")
        if merge_contract is None:
            merge_contract = _path_context_value(context, "member_merge_contract")
        if merge_contract is None:
            blockers.append("materializer_context_missing:merge_contract")
        else:
            input_paths.append(merge_contract)
        source_runtime = _path_context_value(
            context,
            "packet_member_merge_source_runtime_dir",
        )
        if source_runtime is None:
            source_runtime = _path_context_value(context, "source_runtime_dir")
        if source_runtime is None:
            source_runtime = _path_context_value(context, "inflate_runtime_dir")
        if source_runtime is None:
            blockers.append(
                "materializer_context_missing:packet_member_merge_source_runtime_dir"
            )
        else:
            input_paths.append(source_runtime)
    elif target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            input_paths.append(packet_member_manifest)
        parity_proof = _path_context_value(context, "full_frame_inflate_parity_proof")
        if parity_proof is None:
            parity_proof = _path_context_value(
                context,
                "renderer_payload_dfl1_inflate_parity_proof",
            )
        if parity_proof is None:
            parity_proof = _path_context_value(context, "inflate_parity_proof")
        if parity_proof is not None:
            input_paths.append(parity_proof)
    elif target_kind == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            input_paths.append(packet_member_manifest)
        header_elision_contract = _path_context_value(context, "header_elision_contract")
        if header_elision_contract is not None:
            input_paths.append(header_elision_contract)
    elif target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        tensor_manifest = _path_context_value(context, "tensor_manifest")
        if tensor_manifest is None:
            blockers.append("materializer_context_missing:tensor_manifest")
        else:
            input_paths.append(tensor_manifest)
        factorization_contract = _path_context_value(context, "factorization_contract")
        if factorization_contract is None and _finite_int(context.get("rank")) is None:
            blockers.append("materializer_context_missing:factorization_contract_or_rank")
        elif factorization_contract is not None:
            input_paths.append(factorization_contract)
    else:
        blockers.append(f"family_agnostic_materializer_target_unknown:{target_kind}")
    if blockers:
        return [], blockers, {}

    assert archive_path is not None
    assert output_archive is not None
    assert output_manifest is not None
    command = [
        ".venv/bin/python",
        FAMILY_AGNOSTIC_MATERIALIZER_TOOL,
        "--target-kind",
        target_kind,
        "--archive-path",
        archive_path,
        "--output-archive",
        output_archive,
        "--output-manifest",
        output_manifest,
    ]
    runtime_proof = _path_context_value(context, "runtime_consumption_proof")
    if runtime_proof is not None:
        command.extend(["--runtime-consumption-proof", runtime_proof])
        input_paths.append(runtime_proof)
    runtime_proof_out: str | None = None
    if runtime_proof is None and target_kind in {
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        PACKET_MEMBER_MERGE_TARGET_KIND,
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        RENDERER_PAYLOAD_DFL1_TARGET_KIND,
        TENSOR_FACTORIZE_TARGET_KIND,
    }:
        runtime_proof_out = _path_context_value(context, "runtime_consumption_proof_out")
        if runtime_proof_out is None:
            runtime_proof_out = _path_context_value(
                context,
                "runtime_consumption_proof_output",
            )
        if runtime_proof_out is None:
            runtime_proof_out = Path(output_manifest).with_name(
                f"{Path(output_manifest).stem}.runtime_consumption_proof.json"
            ).as_posix()
        command.extend(["--runtime-consumption-proof-out", runtime_proof_out])
    min_free_bytes = _finite_int(context.get("min_free_bytes"))
    if min_free_bytes is not None:
        command.extend(["--min-free-bytes", str(min_free_bytes)])
    if context.get("allow_size_regression") is True or context.get("allow_rate_regression") is True:
        command.append("--allow-size-regression")
    if context.get("allow_overwrite") is True:
        command.append("--allow-overwrite")
        expected_output_sha = _context_string_any(
            context,
            ("expected_existing_output_sha256", "expected_output_sha256"),
        )
        if expected_output_sha is not None:
            command.extend(["--expected-existing-output-sha256", expected_output_sha])
        expected_manifest_sha = _context_string_any(
            context,
            ("expected_existing_manifest_sha256", "expected_manifest_sha256"),
        )
        if expected_manifest_sha is not None:
            command.extend(["--expected-existing-manifest-sha256", expected_manifest_sha])
        expected_runtime_proof_sha = _context_string_any(
            context,
            (
                "expected_existing_runtime_consumption_proof_sha256",
                "expected_runtime_consumption_proof_sha256",
            ),
        )
        if expected_runtime_proof_sha is not None:
            command.extend(
                [
                    "--expected-existing-runtime-consumption-proof-sha256",
                    expected_runtime_proof_sha,
                ]
            )

    if target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND:
        section_manifest = _path_context_value(context, "section_manifest")
        assert section_manifest is not None
        command.extend(["--section-manifest", section_manifest])
        section_names = _string_list_context_value(context, "target_sections")
        section_names.extend(_string_list_context_value(context, "target_section"))
        section_names.extend(_string_list_context_value(context, "section_names"))
        section_names.extend(_string_list_context_value(context, "section_name"))
        for section in ordered_unique(section_names):
            command.extend(["--section-name", section])
        qualities = _string_list_context_value(context, "brotli_quality")
        qualities.extend(_string_list_context_value(context, "brotli_qualities"))
        qualities.extend(_string_list_context_value(context, "quality"))
        qualities.extend(_string_list_context_value(context, "qualities"))
        for quality in ordered_unique(qualities):
            command.extend(["--brotli-quality", quality])
    elif target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            command.extend(["--packet-member-manifest", packet_member_manifest])
        member_name = _context_string_any(
            context,
            ("member_name", "archive_member_name", "packet_member_name"),
        )
        if member_name is not None:
            command.extend(["--member-name", member_name])
        methods = _string_list_context_value(context, "zip_compression_method")
        methods.extend(_string_list_context_value(context, "zip_compression_methods"))
        methods.extend(_string_list_context_value(context, "compression_method"))
        for method in ordered_unique(methods):
            command.extend(["--zip-compression-method", method])
        levels = _string_list_context_value(context, "zip_compresslevel")
        levels.extend(_string_list_context_value(context, "zip_compresslevels"))
        levels.extend(_string_list_context_value(context, "compresslevel"))
        for level in ordered_unique(levels):
            command.extend(["--zip-compresslevel", level])
    elif target_kind == PACKET_MEMBER_MERGE_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            command.extend(["--packet-member-manifest", packet_member_manifest])
        merge_contract = _path_context_value(context, "merge_contract")
        if merge_contract is None:
            merge_contract = _path_context_value(context, "member_merge_contract")
        if merge_contract is not None:
            command.extend(["--merge-contract", merge_contract])
        source_runtime = _path_context_value(
            context,
            "packet_member_merge_source_runtime_dir",
        )
        if source_runtime is None:
            source_runtime = _path_context_value(context, "source_runtime_dir")
        if source_runtime is None:
            source_runtime = _path_context_value(context, "inflate_runtime_dir")
        if source_runtime is not None:
            command.extend(["--packet-member-merge-source-runtime-dir", source_runtime])
        runtime_dir_out = _path_context_value(
            context,
            "packet_member_merge_runtime_dir_out",
        )
        if runtime_dir_out is None:
            runtime_dir_out = _path_context_value(context, "runtime_dir_out")
        if runtime_dir_out is None:
            runtime_dir_out = Path(output_manifest).with_name(
                f"{Path(output_manifest).stem}.runtime"
            ).as_posix()
        command.extend(["--packet-member-merge-runtime-dir-out", runtime_dir_out])
        runtime_manifest_out = _path_context_value(
            context,
            "packet_member_merge_runtime_manifest_out",
        )
        if runtime_manifest_out is None:
            runtime_manifest_out = _path_context_value(context, "runtime_manifest_out")
        if runtime_manifest_out is None:
            runtime_manifest_out = Path(output_manifest).with_name(
                f"{Path(output_manifest).stem}.runtime_adapter.json"
            ).as_posix()
        command.extend(
            ["--packet-member-merge-runtime-manifest-out", runtime_manifest_out]
        )
        if context.get("allow_packet_member_merge_runtime_sidecars") is True:
            command.append("--allow-packet-member-merge-runtime-sidecars")
        member_selection = _context_string_any(
            context,
            ("member_selection", "zip_member_selection", "packet_member_selection"),
        )
        if context.get("all_members") is True or member_selection in {
            "all",
            "*",
            "all_members",
        }:
            command.append("--all-members")
        member_name = _context_string_any(
            context,
            ("member_name", "archive_member_name", "packet_member_name"),
        )
        if member_name is not None:
            command.extend(["--member-name", member_name])
        member_names = _string_list_context_value(context, "member_names")
        member_names.extend(_string_list_context_value(context, "archive_member_names"))
        member_names.extend(_string_list_context_value(context, "packet_member_names"))
        for selected_member_name in ordered_unique(member_names):
            command.extend(["--member-names", selected_member_name])
        merged_member_name = _context_string_any(
            context,
            ("merged_member_name", "candidate_member_name", "output_member_name"),
        )
        if merged_member_name is not None:
            command.extend(["--merged-member-name", merged_member_name])
    elif target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            command.extend(["--packet-member-manifest", packet_member_manifest])
        parity_proof = _path_context_value(context, "full_frame_inflate_parity_proof")
        if parity_proof is None:
            parity_proof = _path_context_value(
                context,
                "renderer_payload_dfl1_inflate_parity_proof",
            )
        if parity_proof is None:
            parity_proof = _path_context_value(context, "inflate_parity_proof")
        if parity_proof is not None:
            command.extend(["--full-frame-inflate-parity-proof", parity_proof])
        member_names = _string_list_context_value(context, "member_names")
        member_names.extend(_string_list_context_value(context, "archive_member_names"))
        member_names.extend(_string_list_context_value(context, "packet_member_names"))
        for selected_member_name in ordered_unique(member_names):
            command.extend(["--member-names", selected_member_name])
        payload_member_name = _context_string_any(
            context,
            ("payload_member_name", "renderer_payload_member_name"),
        )
        if payload_member_name is not None:
            command.extend(["--payload-member-name", payload_member_name])
    elif target_kind == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            command.extend(["--packet-member-manifest", packet_member_manifest])
        header_elision_contract = _path_context_value(context, "header_elision_contract")
        if header_elision_contract is not None:
            command.extend(["--header-elision-contract", header_elision_contract])
        member_selection = _context_string_any(
            context,
            ("member_selection", "zip_member_selection", "packet_member_selection"),
        )
        if context.get("all_members") is True or member_selection in {
            "all",
            "*",
            "all_members",
        }:
            command.append("--all-members")
        member_name = _context_string_any(
            context,
            ("member_name", "archive_member_name", "packet_member_name"),
        )
        if member_name is not None:
            command.extend(["--member-name", member_name])
        member_names = _string_list_context_value(context, "member_names")
        member_names.extend(_string_list_context_value(context, "archive_member_names"))
        member_names.extend(_string_list_context_value(context, "packet_member_names"))
        for selected_member_name in ordered_unique(member_names):
            command.extend(["--member-names", selected_member_name])
    elif target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        tensor_manifest = _path_context_value(context, "tensor_manifest")
        assert tensor_manifest is not None
        command.extend(["--tensor-manifest", tensor_manifest])
        factorization_contract = _path_context_value(context, "factorization_contract")
        if factorization_contract is not None:
            command.extend(["--factorization-contract", factorization_contract])
        rank = _finite_int(context.get("rank"))
        if rank is not None:
            command.extend(["--rank", str(rank)])

    artifact_paths = [output_archive, output_manifest]
    if runtime_proof_out is not None:
        artifact_paths.append(runtime_proof_out)
    if target_kind == PACKET_MEMBER_MERGE_TARGET_KIND:
        runtime_dir_out = _path_context_value(
            context,
            "packet_member_merge_runtime_dir_out",
        )
        if runtime_dir_out is None:
            runtime_dir_out = _path_context_value(context, "runtime_dir_out")
        if runtime_dir_out is None:
            runtime_dir_out = Path(output_manifest).with_name(
                f"{Path(output_manifest).stem}.runtime"
            ).as_posix()
        runtime_manifest_out = _path_context_value(
            context,
            "packet_member_merge_runtime_manifest_out",
        )
        if runtime_manifest_out is None:
            runtime_manifest_out = _path_context_value(context, "runtime_manifest_out")
        if runtime_manifest_out is None:
            runtime_manifest_out = Path(output_manifest).with_name(
                f"{Path(output_manifest).stem}.runtime_adapter.json"
            ).as_posix()
        artifact_paths.extend([runtime_dir_out, runtime_manifest_out])

    return (
        command,
        [],
        {
            "artifact_paths": artifact_paths,
            "input_artifact_paths": ordered_unique(input_paths),
            "pullback_artifact_paths": artifact_paths,
            "include_postcondition_paths": True,
            "family_agnostic_materializer_contract": {
                "schema": "family_agnostic_materializer_command_contract.v1",
                "target_kind": target_kind,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        },
    )


def _sweep_archive_specs_context_value(context: Mapping[str, Any]) -> list[str]:
    specs: list[str] = []
    for key in (
        "sweep_archives",
        "sweep_archive_specs",
        "materializer_sweep_archives",
        "materializer_sweep_archive_specs",
    ):
        value = context.get(key)
        if isinstance(value, Mapping):
            label = _context_string_any(value, ("label", "archive_label"))
            path = _context_string_any(value, ("path", "archive_path", "source_archive"))
            if path is not None:
                specs.append(f"{label}={path}" if label else path)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            for item in value:
                if isinstance(item, Mapping):
                    label = _context_string_any(item, ("label", "archive_label"))
                    path = _context_string_any(
                        item,
                        ("path", "archive_path", "source_archive"),
                    )
                    if path is not None:
                        specs.append(f"{label}={path}" if label else path)
                elif isinstance(item, (str, Path)) and str(item).strip():
                    specs.append(str(item).strip())
            continue
        if isinstance(value, (str, Path)) and str(value).strip():
            specs.append(str(value).strip())
    return ordered_unique(specs)


def _sweep_input_path_from_archive_spec(spec: str) -> str:
    return spec.split("=", 1)[1].strip() if "=" in spec else spec.strip()


def _context_has_family_materializer_sweep(context: Mapping[str, Any]) -> bool:
    return bool(_sweep_archive_specs_context_value(context))


def _family_agnostic_materializer_sweep_command(
    context: Mapping[str, Any],
    *,
    target_kind: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    blockers: list[str] = _string_list_context_value(context, "context_blockers")
    archive_specs = _sweep_archive_specs_context_value(context)
    if not archive_specs:
        blockers.append("materializer_sweep_context_missing:archives")
    output_dir = _context_string_any(
        context,
        ("sweep_output_dir", "output_dir", "materializer_sweep_output_dir"),
    )
    if output_dir is None:
        blockers.append("materializer_sweep_context_missing:output_dir")
    output_json = _context_string_any(
        context,
        ("sweep_output_json", "output_json", "json_out"),
    )
    observation_jsonl = _context_string_any(
        context,
        ("sweep_observation_jsonl", "observation_jsonl", "observations_jsonl"),
    )
    if output_dir is not None:
        output_root = Path(output_dir)
        if output_json is None:
            output_json = (output_root / "sweep.json").as_posix()
        if observation_jsonl is None:
            observation_jsonl = (output_root / "observations.jsonl").as_posix()

    supported_targets = {
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        TENSOR_FACTORIZE_TARGET_KIND,
    }
    if target_kind not in supported_targets:
        blockers.append(f"family_agnostic_materializer_sweep_target_unknown:{target_kind}")
    input_paths = [_sweep_input_path_from_archive_spec(spec) for spec in archive_specs]
    command: list[str] = []
    if not blockers:
        assert output_dir is not None
        assert output_json is not None
        assert observation_jsonl is not None
        command = [
            ".venv/bin/python",
            FAMILY_AGNOSTIC_MATERIALIZER_SWEEP_TOOL,
            "--target-kind",
            target_kind,
            "--output-dir",
            output_dir,
            "--output-json",
            output_json,
            "--observation-jsonl",
            observation_jsonl,
        ]
        for spec in archive_specs:
            command.extend(["--archive", spec])

    if target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND:
        section_manifest = _path_context_value(context, "section_manifest")
        if section_manifest is None:
            blockers.append("materializer_sweep_context_missing:section_manifest")
        else:
            input_paths.append(section_manifest)
            if command:
                command.extend(["--section-manifest", section_manifest])
        section_names = _string_list_context_value(context, "section_name")
        section_names.extend(_string_list_context_value(context, "section_names"))
        section_names.extend(_string_list_context_value(context, "target_sections"))
        for section_name in ordered_unique(section_names):
            if command:
                command.extend(["--section-name", section_name])
        qualities = _string_list_context_value(context, "brotli_quality")
        qualities.extend(_string_list_context_value(context, "brotli_qualities"))
        qualities.extend(_string_list_context_value(context, "quality"))
        qualities.extend(_string_list_context_value(context, "qualities"))
        for quality in ordered_unique(qualities):
            if command:
                command.extend(["--brotli-quality", quality])
    elif target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            input_paths.append(packet_member_manifest)
            if command:
                command.extend(["--packet-member-manifest", packet_member_manifest])
        member_name = _context_string_any(
            context,
            ("member_name", "archive_member_name", "packet_member_name"),
        )
        if member_name is not None and command:
            command.extend(["--member-name", member_name])
        methods = _string_list_context_value(context, "zip_compression_method")
        methods.extend(_string_list_context_value(context, "zip_compression_methods"))
        methods.extend(_string_list_context_value(context, "compression_method"))
        for method in ordered_unique(methods):
            if command:
                command.extend(["--zip-compression-method", method])
        levels = _string_list_context_value(context, "zip_compresslevel")
        levels.extend(_string_list_context_value(context, "zip_compresslevels"))
        levels.extend(_string_list_context_value(context, "compresslevel"))
        for level in ordered_unique(levels):
            if command:
                command.extend(["--zip-compresslevel", level])
    elif target_kind == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND:
        packet_member_manifest = _path_context_value(context, "packet_member_manifest")
        if packet_member_manifest is not None:
            input_paths.append(packet_member_manifest)
            if command:
                command.extend(["--packet-member-manifest", packet_member_manifest])
        header_elision_contract = _path_context_value(context, "header_elision_contract")
        if header_elision_contract is not None:
            input_paths.append(header_elision_contract)
            if command:
                command.extend(["--header-elision-contract", header_elision_contract])
        member_selection = _context_string_any(
            context,
            ("member_selection", "zip_member_selection", "packet_member_selection"),
        )
        if command and (
            context.get("all_members") is True
            or member_selection in {"all", "*", "all_members"}
        ):
            command.append("--all-members")
        member_name = _context_string_any(
            context,
            ("member_name", "archive_member_name", "packet_member_name"),
        )
        if member_name is not None and command:
            command.extend(["--member-name", member_name])
        member_names = _string_list_context_value(context, "member_names")
        member_names.extend(_string_list_context_value(context, "archive_member_names"))
        member_names.extend(_string_list_context_value(context, "packet_member_names"))
        for selected_member_name in ordered_unique(member_names):
            if command:
                command.extend(["--member-names", selected_member_name])
    elif target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        tensor_manifest = _path_context_value(context, "tensor_manifest")
        if tensor_manifest is None:
            blockers.append("materializer_sweep_context_missing:tensor_manifest")
        else:
            input_paths.append(tensor_manifest)
            if command:
                command.extend(["--tensor-manifest", tensor_manifest])
        factorization_contract = _path_context_value(context, "factorization_contract")
        rank = _finite_int(context.get("rank"))
        if factorization_contract is None and rank is None:
            blockers.append("materializer_sweep_context_missing:factorization_contract_or_rank")
        elif factorization_contract is not None:
            input_paths.append(factorization_contract)
            if command:
                command.extend(["--factorization-contract", factorization_contract])
        if rank is not None and command:
            command.extend(["--rank", str(rank)])

    min_free_bytes = _finite_int(context.get("min_free_bytes"))
    if min_free_bytes is not None and command:
        command.extend(["--min-free-bytes", str(min_free_bytes)])
    if context.get("allow_overwrite") is True and command:
        command.append("--allow-overwrite")

    if blockers:
        return [], ordered_unique(blockers), {}
    assert output_dir is not None
    assert output_json is not None
    assert observation_jsonl is not None
    artifact_paths = [output_dir, output_json, observation_jsonl]
    return (
        command,
        [],
        {
            "artifact_paths": artifact_paths,
            "input_artifact_paths": ordered_unique(input_paths),
            "pullback_artifact_paths": artifact_paths,
            "include_postcondition_paths": True,
            "family_agnostic_materializer_sweep_contract": {
                "schema": "family_agnostic_materializer_sweep_command_contract.v1",
                "target_kind": target_kind,
                "output_json": output_json,
                "observation_jsonl": observation_jsonl,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        },
    )


def _materializer_sweep_postconditions(
    *,
    output_json: str,
    observation_jsonl: str,
    target_kind: str,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "json_equals",
            "path": output_json,
            "key": "schema",
            "equals": FAMILY_AGNOSTIC_MATERIALIZER_SWEEP_SCHEMA,
        },
        {
            "type": "json_completion_contract",
            "path": output_json,
            "required_equals": {
                "schema": FAMILY_AGNOSTIC_MATERIALIZER_SWEEP_SCHEMA,
                "target_kind": target_kind,
            },
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
            ],
            "false_or_missing": [
                "ready_for_exact_eval_dispatch",
                "dispatch_attempted",
                "gpu_launched",
            ],
            "required_positive_int": ["observation_count"],
            "required_nonempty": ["observations", "planner_feedback"],
            "forbidden_statuses": ["failed"],
        },
        {
            "type": "jsonl_false_authority",
            "path": observation_jsonl,
            "schema_equals": FAMILY_AGNOSTIC_MATERIALIZER_SWEEP_OBSERVATION_SCHEMA,
            "require_nonempty": True,
        },
    ]


def _family_agnostic_materializer_queue_adapter(
    context: Mapping[str, Any],
    *,
    target_kind: str,
    schema: str,
) -> tuple[list[str], list[str], dict[str, Any], list[dict[str, Any]]]:
    if _context_has_family_materializer_sweep(context):
        command, blockers, telemetry = _family_agnostic_materializer_sweep_command(
            context,
            target_kind=target_kind,
        )
        if blockers:
            return command, blockers, telemetry, []
        contract = telemetry.get("family_agnostic_materializer_sweep_contract")
        if not isinstance(contract, Mapping):
            return command, ["materializer_sweep_contract_missing"], telemetry, []
        output_json = _context_string_any(contract, ("output_json",))
        observation_jsonl = _context_string_any(contract, ("observation_jsonl",))
        if output_json is None or observation_jsonl is None:
            return command, ["materializer_sweep_output_paths_missing"], telemetry, []
        return (
            command,
            [],
            telemetry,
            _materializer_sweep_postconditions(
                output_json=output_json,
                observation_jsonl=observation_jsonl,
                target_kind=target_kind,
            ),
        )

    command, blockers, telemetry = _family_agnostic_materializer_command(
        context,
        target_kind=target_kind,
    )
    postconditions: list[dict[str, Any]] = []
    manifest_out = _path_context_value(context, "output_manifest")
    if manifest_out is None:
        manifest_out = _path_context_value(context, "manifest_out")
    if manifest_out is None:
        manifest_out = _path_context_value(context, "json_out")
    output_archive = _path_context_value(context, "output_archive")
    if manifest_out is None and output_archive is not None:
        manifest_out = Path(output_archive).with_suffix(".json").as_posix()
    if command and manifest_out is not None:
        postconditions = _materializer_candidate_postconditions(
            manifest_path=manifest_out,
            schema=schema,
            target_kind=target_kind,
        )
    return command, blockers, telemetry, postconditions


def _materializer_work_dispatch_blockers(target_kind: str) -> tuple[str, ...]:
    blockers = ["materializer_work_queue_local_proof_chain_only"]
    if target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND:
        blockers.extend(
            [
                "archive_section_entropy_recode_requires_archive_preflight",
                "archive_section_entropy_recode_requires_same_runtime_inflate_parity",
            ]
        )
    if target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND:
        blockers.extend(
            [
                "packet_member_recompress_requires_archive_preflight",
                "packet_member_recompress_requires_runtime_consumption_proof",
            ]
        )
    if target_kind == PACKET_MEMBER_MERGE_TARGET_KIND:
        blockers.extend(
            [
                "packet_member_merge_requires_archive_preflight",
                "packet_member_merge_requires_cooperative_receiver_runtime_adapter",
                "packet_member_merge_requires_runtime_consumption_proof",
            ]
        )
    if target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        blockers.extend(
            [
                "renderer_payload_dfl1_requires_archive_preflight",
                "renderer_payload_dfl1_requires_source_runtime_unpack_proof",
                "renderer_payload_dfl1_requires_same_runtime_full_frame_parity",
            ]
        )
    if target_kind == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND:
        blockers.extend(
            [
                "packet_member_zip_header_elide_requires_archive_preflight",
                "packet_member_zip_header_elide_requires_runtime_consumption_proof",
            ]
        )
    if target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        blockers.extend(
            [
                "tensor_factorize_requires_cooperative_receiver",
                "tensor_factorize_requires_runtime_consumption_proof",
            ]
        )
    if target_kind == INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND:
        blockers.extend(
            [
                "local_inverse_action_functional_proof_chain_only",
                "inverse_action_functional_is_not_candidate_archive",
            ]
        )
    if target_kind == INVERSE_SCORER_CELL_TARGET_KIND:
        blockers.extend(
            [
                "inverse_scorer_cell_candidate_requires_receiver_proof",
                "inverse_scorer_cell_candidate_requires_inflate_parity",
            ]
        )
    if target_kind == INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND:
        blockers.extend(
            [
                "inverse_action_operation_set_compiler_handoff_is_planning_only",
                "packet_ir_lowering_required_before_materializer_contexts",
                "runtime_consumption_proof_required_before_exact_eval",
            ]
        )
    blockers.append("exact_auth_eval_required_before_score_claim")
    return tuple(blockers)


def _materializer_chain_postconditions(
    *,
    manifest_path: str,
    schema: str,
    require_serialized_archive_saving: bool = False,
    require_inflate_parity: bool = False,
) -> list[dict[str, Any]]:
    required_positive_int = ["candidate_archive_bytes"]
    required_less_than: list[dict[str, str]] = []
    required_equals: dict[str, Any] = {"schema": schema}
    required_true = [
        "byte_closed_candidate_emitted",
        "runtime_adapter_ready",
        "receiver_proof_ready",
        "receiver_contract_satisfied",
        "candidate_runtime_adapter_blocker_cleared",
    ]
    if require_serialized_archive_saving:
        required_positive_int.append("source_archive_bytes")
        required_less_than.append({"left": "candidate_archive_bytes", "right": "source_archive_bytes"})
        required_equals["serialized_archive_delta.status"] = "realized_saving"
    if require_inflate_parity:
        required_true.append("inflate_parity_satisfied")
    chain_contract: dict[str, Any] = {
        "type": "materializer_chain_complete",
        "path": manifest_path,
        "schema": schema,
    }
    if require_serialized_archive_saving:
        chain_contract["required_serialized_archive_saving"] = True
    if require_inflate_parity:
        chain_contract["required_inflate_parity"] = True
    return [
        {
            "type": "json_equals",
            "path": manifest_path,
            "key": "schema",
            "equals": schema,
        },
        {
            "type": "json_completion_contract",
            "path": manifest_path,
            "required_equals": required_equals,
            "required_true": required_true,
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
            ],
            "false_or_missing": [
                "ready_for_exact_eval_dispatch",
                "dispatch_attempted",
                "gpu_launched",
            ],
            "required_sha256": ["candidate_archive_sha256"],
            "required_positive_int": required_positive_int,
            "required_artifact_records": ["candidate_archive"],
            "required_less_than": required_less_than,
            "forbidden_statuses": ["failed"],
        },
        chain_contract,
    ]


def _grouped_archive_state_materializer_postconditions(
    *,
    manifest_path: str,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "json_equals",
            "path": manifest_path,
            "key": "schema",
            "equals": GROUPED_ARCHIVE_STATE_MATERIALIZER_CHAIN_SCHEMA,
        },
        {
            "type": "json_completion_contract",
            "path": manifest_path,
            "required_equals": {
                "schema": GROUPED_ARCHIVE_STATE_MATERIALIZER_CHAIN_SCHEMA,
            },
            "required_true": [
                "byte_closed_candidate_emitted",
                "runtime_adapter_ready",
                "receiver_proof_ready",
                "receiver_contract_satisfied",
                "candidate_runtime_adapter_blocker_cleared",
            ],
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
            ],
            "false_or_missing": [
                "ready_for_exact_eval_dispatch",
                "dispatch_attempted",
                "gpu_launched",
            ],
            "required_sha256": ["candidate_archive_sha256"],
            "required_positive_int": [
                "operation_count",
                "candidate_archive_bytes",
                "source_archive_bytes",
            ],
            "required_artifact_records": ["candidate_archive", "final_candidate_archive"],
            "forbidden_statuses": ["failed"],
        },
    ]


def _materializer_candidate_postconditions(
    *,
    manifest_path: str,
    schema: str,
    target_kind: str | None = None,
) -> list[dict[str, Any]]:
    required_equals: dict[str, Any] = {"schema": schema}
    required_sha256 = ["candidate_archive.sha256"]
    required_positive_int = ["candidate_archive.bytes"]
    required_nonempty: list[str] = []
    required_nonempty_unless_true: list[dict[str, str]] = []
    if target_kind in {
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        PACKET_MEMBER_MERGE_TARGET_KIND,
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        RENDERER_PAYLOAD_DFL1_TARGET_KIND,
        TENSOR_FACTORIZE_TARGET_KIND,
    }:
        required_equals["receiver_verification.schema"] = "family_agnostic_runtime_consumption_proof_verification.v1"
        required_equals["serialized_archive_delta.schema"] = (
            "serialized_archive_delta_contract.v1"
        )
        required_sha256.append("candidate_member.sha256")
        required_positive_int.extend(
            [
                "candidate_member.bytes",
                "serialized_archive_delta.source_archive_bytes",
                "serialized_archive_delta.candidate_archive_bytes",
            ]
        )
        required_nonempty.append("runtime_consumption_proof_path")
    if target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        required_equals["receiver_contract_kind"] = "family_agnostic_tensor_factorize"
        required_sha256.append("receiver_verification.runtime_adapter_sha256")
        required_positive_int.append("factorization.factor_payload_bytes")
    if target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        required_equals["receiver_contract_kind"] = (
            "source_runtime_native_renderer_payload_dfl1"
        )
        required_positive_int.append("selected_payload.payload_bytes")
    if target_kind == PACKET_MEMBER_MERGE_TARGET_KIND:
        required_nonempty.extend(
            [
                "packet_member_merge_receiver_runtime.runtime_dir",
                "packet_member_merge_receiver_runtime.runtime_tree_sha256",
            ]
        )
    return [
        {
            "type": "json_equals",
            "path": manifest_path,
            "key": "schema",
            "equals": schema,
        },
        {
            "type": "json_completion_contract",
            "path": manifest_path,
            "required_equals": required_equals,
            "required_true": [
                "byte_closed_candidate_emitted",
                *(
                    [
                        "receiver_contract_satisfied",
                        "receiver_verification.receiver_contract_satisfied",
                    ]
                    if target_kind
                    in {
                        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
                        PACKET_MEMBER_MERGE_TARGET_KIND,
                        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
                        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
                        RENDERER_PAYLOAD_DFL1_TARGET_KIND,
                        TENSOR_FACTORIZE_TARGET_KIND,
                    }
                    else []
                ),
                *(
                    [
                        "runtime_adapter_ready",
                        "receiver_verification.runtime_adapter_ready",
                    ]
                    if target_kind == TENSOR_FACTORIZE_TARGET_KIND
                    else []
                ),
                *(
                    ["packet_member_merge_receiver_runtime.runtime_adapter_ready"]
                    if target_kind == PACKET_MEMBER_MERGE_TARGET_KIND
                    else []
                ),
                *(
                    [
                        "runtime_adapter_ready",
                        "receiver_verification.runtime_adapter_ready",
                        "full_frame_inflate_parity_proven",
                        "full_frame_inflate_parity_verification.full_frame_inflate_parity_satisfied",
                        "renderer_payload_dfl1_inflate_parity_satisfied",
                    ]
                    if target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND
                    else []
                ),
            ],
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
            ],
            "false_or_missing": [
                "ready_for_exact_eval_dispatch",
                "dispatch_attempted",
                "gpu_launched",
            ],
            "required_nonempty": required_nonempty,
            "required_nonempty_unless_true": required_nonempty_unless_true,
            "required_sha256": required_sha256,
            "required_positive_int": required_positive_int,
            "required_artifact_records": ["candidate_archive"],
            "forbidden_statuses": ["failed"],
        },
    ]


def _planning_artifact_postconditions(
    *,
    manifest_path: str,
    schema: str,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "json_equals",
            "path": manifest_path,
            "key": "schema",
            "equals": schema,
        },
        {
            "type": "json_completion_contract",
            "path": manifest_path,
            "required_equals": {"schema": schema},
            "required_true": ["planning_only", "candidate_generation_only"],
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
            ],
            "false_or_missing": [
                "ready_for_exact_eval_dispatch",
                "dispatch_attempted",
                "gpu_launched",
            ],
            "required_positive_int": ["integral_totals.cell_count"],
            "required_nonempty": ["water_bucket"],
            "forbidden_statuses": ["failed"],
        },
    ]


def build_materializer_work_queue(
    backlog: Mapping[str, Any],
    *,
    repo_root: str | Path,
    contexts: Mapping[str, Mapping[str, Any]] | None = None,
    source_plan_path: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert materializer backlog rows into fail-closed local proof-chain work."""

    if backlog.get("schema") != MATERIALIZER_BACKLOG_SCHEMA:
        raise ExperimentQueueError(f"expected schema {MATERIALIZER_BACKLOG_SCHEMA}")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ExperimentQueueError("limit must be >= 1 when provided")
    repo = Path(repo_root)
    context_map = contexts or {}
    rows = [item for item in _as_list(backlog.get("rows")) if isinstance(item, Mapping)]
    if limit is not None:
        rows = rows[:limit]
    existing_backlog_keys = {str(row.get("backlog_key") or "") for row in rows}
    rows.extend(
        row
        for row in _high_level_context_packet_ir_rows(
            {"schema": MATERIALIZER_BACKLOG_SCHEMA, "rows": rows},
            context_map,
            source_plan_path=source_plan_path,
        )
        if str(row.get("backlog_key") or "") not in existing_backlog_keys
    )

    work_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        suggestion = _first_suggested_materializer(row)
        materializer_id = str(row.get("materializer_id") or suggestion.get("materializer_id") or "")
        target_kind = str(row.get("target_kind") or suggestion.get("target_kind") or "")
        unit_kind = str(row.get("unit_kind") or "")
        operation_family = str(row.get("operation_family") or "")
        backlog_key = str(row.get("backlog_key") or f"{unit_kind}:{operation_family}:{rank}")
        context_matches = _context_matches_for_backlog_row(
            context_map,
            row,
            extra_keys=(
                str(suggestion.get("materializer_id") or ""),
                str(suggestion.get("target_kind") or ""),
                materializer_id,
                target_kind,
            ),
        )
        blockers: list[str] = []
        if len(context_matches) > 1:
            if _context_matches_are_composable(
                context_matches,
                row=row,
                materializer_id=materializer_id,
                target_kind=target_kind,
            ):
                context = _merge_context_matches(context_matches)
            else:
                blockers.append(
                    "materializer_context_ambiguous:"
                    + ",".join(key for key, _context in context_matches)
                )
                context = {}
        else:
            context = context_matches[0][1] if context_matches else {}
        blockers.extend(_string_list_context_value(context, "context_blockers"))
        command: list[str] = []
        postconditions: list[dict[str, Any]] = []
        telemetry: dict[str, Any] = {}
        if (
            unit_kind == "byte_range"
            and operation_family == "entropy_recode"
            and target_kind == "byte_range_entropy_recode_v1"
        ):
            command, command_blockers = _byte_range_chain_command(context)
            blockers.extend(command_blockers)
            if command:
                input_paths = _command_flag_values(
                    command,
                    {
                        "--beam-probe-report",
                        "--global-combo-report",
                        "--schema-manifest",
                        "--source-archive",
                        "--source-runtime-dir",
                    },
                )
                postconditions = _materializer_chain_postconditions(
                    manifest_path=str(Path(context["output_dir"]) / BYTE_RANGE_CHAIN_MANIFEST),
                    schema=CHAIN_SCHEMA,
                    require_serialized_archive_saving=True,
                )
                telemetry = {
                    "artifact_paths": [str(context["output_dir"])],
                    "input_artifact_paths": input_paths,
                    "pullback_artifact_paths": [str(context["output_dir"])],
                    "pullback_recursive": True,
                    "pullback_max_recursive_entries": 512,
                    "recursive": True,
                    "max_recursive_entries": 512,
                    "include_postcondition_paths": True,
                }
        elif (
            unit_kind == "archive_section"
            and operation_family == "section_entropy_recode"
            and target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
        ):
            command, command_blockers, telemetry, postconditions = (
                _family_agnostic_materializer_queue_adapter(
                context,
                target_kind=ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
                schema=ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
                )
            )
            blockers.extend(command_blockers)
        elif (
            unit_kind == "packet_member"
            and operation_family == "member_recompress"
            and target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND
        ):
            command, command_blockers, telemetry, postconditions = (
                _family_agnostic_materializer_queue_adapter(
                context,
                target_kind=PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
                schema=PACKET_MEMBER_RECOMPRESS_SCHEMA,
                )
            )
            blockers.extend(command_blockers)
        elif (
            unit_kind == "packet_member"
            and operation_family == "member_merge"
            and target_kind == PACKET_MEMBER_MERGE_TARGET_KIND
        ):
            command, command_blockers, telemetry, postconditions = (
                _family_agnostic_materializer_queue_adapter(
                context,
                target_kind=PACKET_MEMBER_MERGE_TARGET_KIND,
                schema=PACKET_MEMBER_MERGE_SCHEMA,
                )
            )
            blockers.extend(command_blockers)
        elif (
            unit_kind == "packet_member"
            and operation_family == "native_renderer_payload"
            and target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND
        ):
            command, command_blockers, telemetry, postconditions = (
                _family_agnostic_materializer_queue_adapter(
                context,
                target_kind=RENDERER_PAYLOAD_DFL1_TARGET_KIND,
                schema=RENDERER_PAYLOAD_DFL1_SCHEMA,
                )
            )
            blockers.extend(command_blockers)
        elif (
            unit_kind == "packet_member"
            and operation_family == "zip_header_elide"
            and target_kind == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND
        ):
            command, command_blockers, telemetry, postconditions = (
                _family_agnostic_materializer_queue_adapter(
                context,
                target_kind=PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
                schema=PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA,
                )
            )
            blockers.extend(command_blockers)
        elif (
            unit_kind == "tensor"
            and operation_family == "factorize_tensor"
            and target_kind == TENSOR_FACTORIZE_TARGET_KIND
        ):
            command, command_blockers, telemetry, postconditions = (
                _family_agnostic_materializer_queue_adapter(
                context,
                target_kind=TENSOR_FACTORIZE_TARGET_KIND,
                schema=TENSOR_FACTORIZE_SCHEMA,
                )
            )
            blockers.extend(command_blockers)
        elif (
            unit_kind == "scorer_inverse_surface_cell"
            and operation_family == "probe_inverse_scorer_surface_cell"
            and target_kind == INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND
        ):
            command, command_blockers, telemetry = _inverse_scorer_action_functional_command(context)
            blockers.extend(command_blockers)
            output = _path_context_value(context, "output")
            if command and output is not None:
                postconditions = _planning_artifact_postconditions(
                    manifest_path=output,
                    schema=INVERSE_ACTION_FUNCTIONAL_SCHEMA,
                )
        elif (
            unit_kind == "scorer_inverse_surface_cell"
            and operation_family == "materialize_inverse_scorer_cell_candidate"
            and target_kind == INVERSE_SCORER_CELL_TARGET_KIND
        ):
            command, command_blockers, telemetry = _inverse_scorer_cell_candidate_command(
                context,
                repo_root=repo,
            )
            blockers.extend(command_blockers)
            output_dir = _path_context_value(context, "chain_output_dir")
            if output_dir is None:
                output_dir = _path_context_value(context, "output_dir")
            manifest_out = _path_context_value(context, "manifest_out")
            if command and output_dir is not None:
                require_inflate_parity = (
                    context.get("descriptor_probe_only") is not True
                    or _path_context_value(context, "source_inflate_output_dir") is not None
                    or _path_context_value(context, "candidate_inflate_output_dir") is not None
                    or _path_context_value(context, "inflate_runtime_dir") is not None
                    or context.get("fail_if_inflate_parity_blocked") is True
                )
                postconditions = _materializer_chain_postconditions(
                    manifest_path=str(Path(output_dir) / INVERSE_SCORER_CELL_CHAIN_MANIFEST),
                    schema=INVERSE_SCORER_CELL_CHAIN_SCHEMA,
                    require_inflate_parity=require_inflate_parity,
                )
            elif command and manifest_out is not None:
                postconditions = _materializer_candidate_postconditions(
                    manifest_path=manifest_out,
                    schema=INVERSE_SCORER_CELL_CANDIDATE_SCHEMA,
                )
        elif (
            unit_kind == "scorer_inverse_surface_cell"
            and operation_family == INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY
            and target_kind == INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND
        ):
            if _context_mapping_value(
                context,
                "packet_ir_operation_set",
            ) or _context_mapping_value(context, "operation_set_compiler"):
                blockers.append(
                    "inverse_action_high_level_context_lowered_to_packet_ir_materializer_rows"
                )
            else:
                blockers.append(
                    "inverse_action_high_level_context_requires_operation_set_compiler"
                )
        else:
            blockers.append(
                f"materializer_work_queue_adapter_missing:{unit_kind}:{operation_family}:{target_kind or '<target_tbd>'}"
            )
        if not context:
            blockers.append(f"materializer_context_missing:{backlog_key}")
        blockers = ordered_unique(blockers)
        executable = not blockers
        dfl1_parity_context = (
            _renderer_payload_dfl1_parity_context(context)
            if target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND
            else {}
        )
        work_rows.append(
            apply_proxy_evidence_boundary(
                {
                    "schema": "byte_shaving_materializer_work_row.v1",
                    "work_id": _materializer_work_id(backlog_key),
                    "work_rank": rank,
                    "backlog_key": backlog_key,
                    "backlog_rank": row.get("backlog_rank"),
                    "source_backlog_key": row.get("source_backlog_key"),
                    "source_plan_path": source_plan_path,
                    "repo_root": _repo_rel(repo, repo),
                    "unit_kind": unit_kind,
                    "operation_family": operation_family,
                    "target_kind": target_kind or None,
                    "materializer_id": materializer_id or None,
                    "receiver_contract_id": (row.get("receiver_contract_id") or suggestion.get("receiver_contract_id")),
                    "receiver_contract_kind": (
                        row.get("receiver_contract_kind") or suggestion.get("receiver_contract_kind")
                    ),
                    "tool": command[1] if command else None,
                    "command": command,
                    "postconditions": postconditions,
                    "telemetry": telemetry,
                    "renderer_payload_dfl1_parity_context": (
                        dfl1_parity_context or None
                    ),
                    "resource_kind": row.get("materialization_resource_kind")
                    or suggestion.get("materialization_resource_kind")
                    or "local_cpu",
                    "source_unit_ids": _as_list(row.get("source_unit_ids")),
                    "source_selection_ids": _as_list(row.get("source_selection_ids")),
                    "packet_ir_lowered_row_count": row.get("packet_ir_lowered_row_count")
                    or (1 if row.get("source_packet_ir_operation_set_id") else None),
                    "source_packet_ir_operation_indices": _as_list(
                        row.get("source_packet_ir_operation_indices")
                    ),
                    "source_packet_ir_operation_indices_by_unit": _as_mapping(
                        row.get("source_packet_ir_operation_indices_by_unit")
                    ),
                    "source_packet_ir_operation_set_ids": ordered_unique(
                        [
                            *_as_list(row.get("source_packet_ir_operation_set_ids")),
                            str(row.get("source_packet_ir_operation_set_id") or ""),
                        ]
                    ),
                    "source_packet_ir_source_operation_set_ids": _as_list(
                        row.get("source_packet_ir_source_operation_set_ids")
                    )
                    or (
                        [str(row.get("source_packet_ir_source_operation_set_id"))]
                        if row.get("source_packet_ir_source_operation_set_id")
                        else []
                    ),
                    "packet_ir_blocker_counts": _counter_dict(
                        row.get("packet_ir_blocker_counts")
                    )
                    or _counter_dict(row.get("blocker_counts")),
                    "candidate_saved_bytes_sum": row.get("candidate_saved_bytes_sum"),
                    "expected_score_gain_sum": row.get("expected_score_gain_sum"),
                    "executable": executable,
                    "materialization_blockers": blockers,
                    **FALSE_AUTHORITY,
                },
                dispatch_blockers=(_materializer_work_dispatch_blockers(target_kind) if executable else blockers),
            )
        )
    executable_rows = [row for row in work_rows if row["executable"] is True]
    blocked_rows = [row for row in work_rows if row["executable"] is not True]
    grouped_requests = _grouped_archive_state_materializer_requests(work_rows)
    grouped_executable_requests = [
        request for request in grouped_requests if request["executable"] is True
    ]
    return apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZER_WORK_QUEUE_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "source_backlog_schema": backlog.get("schema"),
            "source_plan_path": source_plan_path,
            "row_count": len(work_rows),
            "executable_row_count": len(executable_rows),
            "blocked_row_count": len(blocked_rows),
            "grouped_archive_state_request_count": len(grouped_requests),
            "grouped_archive_state_executable_request_count": len(
                grouped_executable_requests
            ),
            "grouped_archive_state_materializer_requests": grouped_requests,
            "rows": work_rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=([] if executable_rows else ("materializer_work_queue_has_no_executable_rows",)),
    )


def _normalize_materializer_queue_postconditions(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    postconditions: list[dict[str, Any]] = []
    for index, raw_condition in enumerate(value):
        if not isinstance(raw_condition, Mapping):
            raise ExperimentQueueError(f"materializer work row postconditions[{index}] must be an object")
        condition = dict(raw_condition)
        condition_type = str(condition.get("type") or "")
        if condition_type == "json_file_key_equals":
            if "value" not in condition:
                raise ExperimentQueueError("json_file_key_equals postcondition must include value")
            condition = {
                "type": "json_equals",
                "path": condition.get("path"),
                "key": condition.get("key"),
                "equals": condition.get("value"),
            }
        postconditions.append(condition)
    return postconditions


def _materializer_execution_priority(row: Mapping[str, Any], fallback: int) -> int:
    rank = _finite_int(row.get("work_rank"))
    if rank is None or rank < 1:
        return fallback
    return rank


def _materializer_execution_experiment_id(
    row: Mapping[str, Any],
    rank: int,
    seen: set[str],
) -> str:
    raw = str(row.get("work_id") or row.get("backlog_key") or f"row_{rank}")
    safe = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_") or f"row_{rank}"
    experiment_id = safe
    if experiment_id not in seen:
        seen.add(experiment_id)
        return experiment_id
    experiment_id = f"{safe}_r{rank:04d}"
    if experiment_id in seen:
        raise ExperimentQueueError(f"duplicate materializer execution experiment id: {safe}")
    seen.add(experiment_id)
    return experiment_id


def _resolve_repo_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _resolve_repo_path_no_resolve(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.absolute()


def _materializer_chain_manifest_path(
    postconditions: Sequence[Mapping[str, Any]],
    *,
    row_id: str,
) -> str:
    path, kind, _saw_manifest_postcondition = (
        _harvestable_materializer_manifest_reference(postconditions)
    )
    if kind == "sweep_manifest":
        path = None
    if path is not None:
        return path
    raise ExperimentQueueError(
        f"include_exact_readiness_followup requires a harvestable materializer manifest postcondition for {row_id}"
    )


def _harvestable_materializer_manifest_path(
    postconditions: Sequence[Mapping[str, Any]],
) -> tuple[str | None, bool]:
    path, _kind, saw_manifest_postcondition = (
        _harvestable_materializer_manifest_reference(postconditions)
    )
    return path, saw_manifest_postcondition


def _harvestable_materializer_manifest_reference(
    postconditions: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None, bool]:
    saw_manifest_postcondition = False
    for condition in postconditions:
        if condition.get("type") != "materializer_chain_complete":
            continue
        path = condition.get("path")
        if isinstance(path, str) and path.strip():
            return path, "chain_manifest", True
    for condition in postconditions:
        if condition.get("type") != "json_completion_contract":
            continue
        path = condition.get("path")
        if isinstance(path, str) and path.strip():
            saw_manifest_postcondition = True
        required_equals = condition.get("required_equals")
        schema = required_equals.get("schema") if isinstance(required_equals, Mapping) else condition.get("schema")
        if isinstance(path, str) and path.strip():
            if schema in HARVESTABLE_MATERIALIZER_MANIFEST_SCHEMAS:
                return path, "chain_manifest", saw_manifest_postcondition
            if schema == FAMILY_AGNOSTIC_MATERIALIZER_SWEEP_SCHEMA:
                return path, "sweep_manifest", saw_manifest_postcondition
    return None, None, saw_manifest_postcondition


def _planning_only_exact_readiness_skip_reason(row: Mapping[str, Any]) -> str | None:
    target_kind = str(row.get("target_kind") or "")
    materializer_id = str(row.get("materializer_id") or "")
    if (
        target_kind == INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND
        or materializer_id == INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER
    ):
        return "planning_only_inverse_action_functional_not_candidate_archive"
    return None


def _materializer_exact_readiness_skip_reason(
    row: Mapping[str, Any],
    postconditions: Sequence[Mapping[str, Any]],
) -> str | None:
    planning_reason = _planning_only_exact_readiness_skip_reason(row)
    if planning_reason is not None:
        return planning_reason
    path, _kind, saw_manifest_postcondition = (
        _harvestable_materializer_manifest_reference(postconditions)
    )
    if path is None and saw_manifest_postcondition:
        return "materializer_manifest_not_harvestable_for_exact_readiness"
    return None


def _materializer_exact_readiness_followup_steps(
    *,
    queue_id: str,
    repo_root: Path,
    row_id: str,
    source_work_queue_path: Path,
    source_state_path: Path,
    work_row: Mapping[str, Any],
    postconditions: Sequence[Mapping[str, Any]],
    handoff_dir: Path,
    materializer_step_id: str,
    step_timeout_seconds: int,
    require_ready: bool,
    dispatch_require_authorized: bool,
    dispatch_provider: str,
    dispatch_label_prefix: str,
    dispatch_max_total_cost: float,
    dispatch_estimated_cost_per_dispatch: float,
) -> list[dict[str, Any]]:
    manifest_ref_path, manifest_ref_kind, _saw_manifest_postcondition = (
        _harvestable_materializer_manifest_reference(postconditions)
    )
    if manifest_ref_path is None or manifest_ref_kind is None:
        raise ExperimentQueueError(
            "include_exact_readiness_followup requires a harvestable "
            f"materializer manifest postcondition for {row_id}"
        )
    manifest_path = _resolve_repo_path(manifest_ref_path, repo_root=repo_root)
    source_queue_path = handoff_dir / "source_queue.json"
    harvest_report_path = handoff_dir / "harvest_report.json"
    readiness_dir = handoff_dir / "exact_readiness"
    bridge_report_path = handoff_dir / "exact_readiness_bridge_report.json"
    dispatch_plan_path = handoff_dir / "dispatch_plan.json"
    dispatch_queue_path = handoff_dir / "dispatch_queue.json"
    dfl1_parity_step = _renderer_payload_dfl1_parity_followup_step(
        work_row,
        repo_root=repo_root,
        handoff_dir=handoff_dir,
        materializer_step_id=materializer_step_id,
        step_timeout_seconds=step_timeout_seconds,
    )
    harvest_requires = [materializer_step_id]
    dfl1_parity_proof_path: Path | None = None
    if dfl1_parity_step is not None:
        harvest_requires = [MATERIALIZER_DFL1_PARITY_STEP_ID]
        dfl1_parity_proof_path = _resolve_repo_path(
            str(dfl1_parity_step["postconditions"][0]["path"]),
            repo_root=repo_root,
        )

    harvest_command = [
        ".venv/bin/python",
        HARVEST_MATERIALIZER_TOOL,
        "--repo-root",
        repo_root.as_posix(),
        "--work-queue",
        _repo_rel(source_work_queue_path, repo_root),
        "--state",
        _repo_rel(source_state_path, repo_root),
        "--queue-id",
        queue_id,
        "--source-queue-out",
        _repo_rel(source_queue_path, repo_root),
        "--report-out",
        _repo_rel(harvest_report_path, repo_root),
        "--exact-readiness-out-dir",
        _repo_rel(readiness_dir, repo_root),
        "--exact-readiness-bridge-report-out",
        _repo_rel(bridge_report_path, repo_root),
        "--require-accepted",
    ]
    if manifest_ref_kind == "sweep_manifest":
        harvest_command.extend(
            [
                "--sweep-manifest",
                f"{row_id}={_repo_rel(manifest_path, repo_root)}",
            ]
        )
    else:
        harvest_command.extend(
            [
                "--chain-manifest",
                _repo_rel(manifest_path, repo_root),
            ]
        )
    if dfl1_parity_proof_path is not None:
        harvest_command.extend(
            [
                "--renderer-payload-dfl1-inflate-parity-proof",
                _repo_rel(dfl1_parity_proof_path, repo_root),
                "--allowed-artifact-root",
                _repo_rel(dfl1_parity_proof_path.parent, repo_root),
            ]
        )
    if require_ready:
        harvest_command.append("--exact-readiness-require-ready")

    dispatch_plan_command = [
        ".venv/bin/python",
        MATERIALIZER_DISPATCH_PLAN_TOOL,
        "--repo-root",
        repo_root.as_posix(),
        "--bridge-report",
        _repo_rel(bridge_report_path, repo_root),
        "--dispatch-plan-out",
        _repo_rel(dispatch_plan_path, repo_root),
        "--experiment-queue-out",
        _repo_rel(dispatch_queue_path, repo_root),
        "--experiment-queue-id",
        f"{queue_id}_{row_id}_exact_eval_dispatch",
        "--provider",
        dispatch_provider,
        "--label-prefix",
        dispatch_label_prefix,
        "--estimated-cost-per-dispatch",
        f"{dispatch_estimated_cost_per_dispatch:.8g}",
        "--max-total-cost",
        f"{dispatch_max_total_cost:.8g}",
    ]
    if dispatch_require_authorized:
        dispatch_plan_command.append("--require-authorized")

    steps = []
    if dfl1_parity_step is not None:
        steps.append(dfl1_parity_step)
    steps.extend(
        [
        {
            "id": MATERIALIZER_HARVEST_STEP_ID,
            "kind": "command",
            "command": harvest_command,
            "requires": harvest_requires,
            "resources": {"kind": "local_cpu"},
            "timeout_seconds": step_timeout_seconds,
            "postconditions": [
                {
                    "type": "json_equals",
                    "path": _repo_rel(source_queue_path, repo_root),
                    "key": "schema",
                    "equals": OPTIMIZER_CANDIDATE_QUEUE_SCHEMA,
                },
                {
                    "type": "json_equals",
                    "path": _repo_rel(harvest_report_path, repo_root),
                    "key": "schema",
                    "equals": MATERIALIZER_HARVEST_REPORT_SCHEMA,
                },
                {
                    "type": "json_equals",
                    "path": _repo_rel(bridge_report_path, repo_root),
                    "key": "schema",
                    "equals": MATERIALIZER_EXACT_READINESS_BRIDGE_SCHEMA,
                },
            ],
            "telemetry": {
                "artifact_paths": [
                    _repo_rel(source_queue_path, repo_root),
                    _repo_rel(harvest_report_path, repo_root),
                    _repo_rel(readiness_dir, repo_root),
                    _repo_rel(bridge_report_path, repo_root),
                ],
                "recursive": True,
            },
        },
        {
            "id": MATERIALIZER_DISPATCH_PLAN_STEP_ID,
            "kind": "command",
            "command": dispatch_plan_command,
            "requires": [MATERIALIZER_HARVEST_STEP_ID],
            "resources": {"kind": "local_cpu"},
            "timeout_seconds": step_timeout_seconds,
            "postconditions": [
                {
                    "type": "json_equals",
                    "path": _repo_rel(dispatch_plan_path, repo_root),
                    "key": "schema",
                    "equals": MATERIALIZER_EXACT_EVAL_DISPATCH_PLAN_SCHEMA,
                },
                {
                    "type": "json_equals",
                    "path": _repo_rel(dispatch_queue_path, repo_root),
                    "key": "schema",
                    "equals": QUEUE_SCHEMA,
                },
            ],
            "telemetry": {
                "artifact_paths": [
                    _repo_rel(dispatch_plan_path, repo_root),
                    _repo_rel(dispatch_queue_path, repo_root),
                ],
                "recursive": False,
            },
        },
        ]
    )
    return steps


def _grouped_archive_state_execution_experiments(
    *,
    work_queue: Mapping[str, Any],
    queue_id: str,
    repo_root: Path,
    source_work_queue_path: Path | None,
    preflight_dependency: str | None,
    step_timeout_seconds: int,
    lane_id: str | None,
    seen_experiments: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    requests = [
        request
        for request in _as_list(work_queue.get("grouped_archive_state_materializer_requests"))
        if isinstance(request, Mapping) and request.get("executable") is True
    ]
    if not requests:
        return [], set()
    if source_work_queue_path is None:
        raise ExperimentQueueError(
            "source_work_queue_path is required for grouped archive-state materializer execution"
        )

    experiments: list[dict[str, Any]] = []
    used_resource_kinds = {"local_cpu"}
    source_queue_path = _resolve_repo_path(source_work_queue_path, repo_root=repo_root)
    grouped_root = source_queue_path.with_suffix("").parent / "grouped_archive_state"
    for rank, request in enumerate(requests, start=1):
        request_id = str(request.get("request_id") or "")
        if not request_id:
            raise ExperimentQueueError("grouped archive-state request_id missing")
        output_dir = grouped_root / request_id
        manifest_path = output_dir / "grouped_archive_state_materializer_chain.json"
        command = [
            ".venv/bin/python",
            GROUPED_ARCHIVE_STATE_MATERIALIZER_TOOL,
            "--repo-root",
            repo_root.as_posix(),
            "--work-queue",
            _repo_rel(source_queue_path, repo_root),
            "--request-id",
            request_id,
            "--output-dir",
            _repo_rel_no_resolve(output_dir, repo_root),
            "--output-manifest",
            _repo_rel_no_resolve(manifest_path, repo_root),
        ]
        experiment_id = _materializer_execution_experiment_id(
            {
                "work_id": request_id,
                "work_rank": rank,
            },
            rank,
            seen_experiments,
        )
        steps = [
            {
                "id": "materialize_grouped_archive_state_chain",
                "kind": "command",
                "command": command,
                "requires": [] if preflight_dependency is None else [preflight_dependency],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": step_timeout_seconds,
                "postconditions": _grouped_archive_state_materializer_postconditions(
                    manifest_path=_repo_rel_no_resolve(manifest_path, repo_root)
                ),
                "telemetry": {
                    "artifact_paths": [_repo_rel_no_resolve(output_dir, repo_root)],
                    "input_artifact_paths": [_repo_rel(source_queue_path, repo_root)],
                    "pullback_artifact_paths": [_repo_rel_no_resolve(output_dir, repo_root)],
                    "include_postcondition_paths": True,
                    "recursive": True,
                    "pullback_recursive": True,
                    "pullback_max_recursive_entries": 512,
                },
            }
        ]
        metadata = apply_proxy_evidence_boundary(
            {
                "schema": MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA,
                "source_work_queue_schema": work_queue.get("schema"),
                "source_work_queue_path": _repo_rel(source_queue_path, repo_root),
                "work_id": request_id,
                "work_rank": rank,
                "grouped_archive_state_request": request,
                "source_packet_ir_operation_set_id": request.get(
                    "source_packet_ir_operation_set_id"
                ),
                "ordered_work_ids": _as_list(request.get("ordered_work_ids")),
                "target_kinds": _as_list(request.get("target_kinds")),
                "allowed_use": "local_grouped_archive_state_materializer_proof_chain_only",
                **FALSE_AUTHORITY,
            },
            dispatch_blockers=(
                "grouped_archive_state_materializer_local_proof_chain_only",
                "grouped_archive_state_requires_runtime_consumption_proof",
                "grouped_archive_state_requires_same_runtime_full_frame_parity_or_rate_only_control",
                "grouped_archive_state_requires_exact_auth_eval_axis_payload",
                "exact_auth_eval_required_before_score_claim",
            ),
        )
        experiments.append(
            {
                "id": experiment_id,
                "lane_id": lane_id,
                "priority": rank,
                "status": "queued",
                "tags": [
                    "byte-shaving",
                    "materializer",
                    "grouped-archive-state",
                    "local-proof-chain",
                    "no-score-authority",
                ],
                "metadata": metadata,
                "steps": steps,
            }
        )
    return experiments, used_resource_kinds


def _renderer_payload_dfl1_parity_followup_step(
    work_row: Mapping[str, Any],
    *,
    repo_root: Path,
    handoff_dir: Path,
    materializer_step_id: str,
    step_timeout_seconds: int,
) -> dict[str, Any] | None:
    if work_row.get("target_kind") != RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        return None
    if _renderer_payload_dfl1_parity_followup_blockers(work_row):
        return None
    context = work_row.get("renderer_payload_dfl1_parity_context")
    if not isinstance(context, Mapping):
        return None
    source_archive = _context_string_any(context, ("source_archive",))
    candidate_archive = _context_string_any(context, ("candidate_archive",))
    source_runtime = _context_string_any(context, ("source_runtime_dir",))
    candidate_runtime = _context_string_any(context, ("candidate_runtime_dir",))
    file_list = _context_string_any(context, ("file_list",))
    file_list_entries = _string_list_context_value(context, "file_list_entries")
    expected_file_list_sha = _context_string_any(
        context,
        ("expected_full_frame_file_list_sha256",),
    )
    expected_entry_count = _finite_int(context.get("expected_full_frame_entry_count"))
    file_list_source = _context_string_any(context, ("full_frame_file_list_source",))
    if (
        source_archive is None
        or candidate_archive is None
        or source_runtime is None
        or candidate_runtime is None
        or (file_list is None and not file_list_entries)
        or expected_file_list_sha is None
        or expected_entry_count is None
        or file_list_source is None
    ):
        return None
    output_dir = _context_string_any(context, ("output_dir",))
    parity_dir = (
        _resolve_repo_path_no_resolve(output_dir, repo_root=repo_root)
        if output_dir is not None
        else handoff_dir / "renderer_payload_dfl1_shell_parity"
    )
    proof_path = parity_dir / "shell_inflate_parity.json"
    command = [
        ".venv/bin/python",
        SHELL_INFLATE_PARITY_TOOL,
        "--left-archive",
        source_archive,
        "--left-submission-dir",
        source_runtime,
        "--right-archive",
        candidate_archive,
        "--right-submission-dir",
        candidate_runtime,
        "--full-frame-file-list-claim",
        "--expected-full-frame-file-list-sha256",
        expected_file_list_sha,
        "--expected-full-frame-entry-count",
        str(expected_entry_count),
        "--full-frame-file-list-source",
        file_list_source,
        "--output-dir",
        _repo_rel_no_resolve(parity_dir, repo_root),
    ]
    if file_list is not None:
        command.extend(["--file-list", file_list])
    else:
        for entry in file_list_entries:
            command.extend(["--file-list-entry", entry])
    input_paths = [source_archive, candidate_archive, source_runtime, candidate_runtime]
    if file_list is not None:
        input_paths.append(file_list)
    return {
        "id": MATERIALIZER_DFL1_PARITY_STEP_ID,
        "kind": "command",
        "command": command,
        "requires": [materializer_step_id],
        "resources": {"kind": "local_io_heavy"},
        "timeout_seconds": step_timeout_seconds,
        "postconditions": [
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "schema",
                "equals": "shell_inflate_parity_proof_v2",
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "full_frame_file_list_claim",
                "equals": True,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "full_frame_file_list_sha256_match",
                "equals": True,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "full_frame_entry_count_match",
                "equals": True,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "expected_full_frame_file_list_sha256",
                "equals": expected_file_list_sha,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "expected_full_frame_entry_count",
                "equals": expected_entry_count,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "full_frame_file_list_source",
                "equals": file_list_source,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "parity_scope_kind",
                "equals": "declared_file_list",
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "contest_full_sample_claim",
                "equals": False,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "contest_full_sample_parity_claim",
                "equals": False,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "full_frame_inflate_output_parity_claim",
                "equals": True,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "output_bytes_match",
                "equals": True,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "output_sha256_match",
                "equals": True,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "output_manifest_sha256_match",
                "equals": True,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "cmp_equal",
                "equals": True,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "blockers",
                "equals": [],
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "score_claim",
                "equals": False,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "ready_for_exact_eval_dispatch",
                "equals": False,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "rank_or_kill_eligible",
                "equals": False,
            },
            {
                "type": "json_equals",
                "path": _repo_rel_no_resolve(proof_path, repo_root),
                "key": "promotable",
                "equals": False,
            },
        ],
        "telemetry": {
            "artifact_paths": [_repo_rel_no_resolve(parity_dir, repo_root)],
            "input_artifact_paths": input_paths,
            "recursive": True,
        },
    }


def build_materializer_execution_queue(
    work_queue: Mapping[str, Any],
    *,
    queue_id: str,
    repo_root: str | Path,
    lane_id: str | None = None,
    source_work_queue_path: str | Path | None = None,
    source_state_path: str | Path | None = None,
    local_cpu_concurrency: int = 1,
    resource_concurrency: Mapping[str, int] | None = None,
    step_timeout_seconds: int = 0,
    limit: int | None = None,
    include_scheduler_preflight: bool = False,
    scheduler_results_root: str = "experiments/results",
    scheduler_storage_tiers: Sequence[str] = (),
    scheduler_storage_workload_subdir: str | None = None,
    scheduler_storage_expected_workload_root: str | None = None,
    scheduler_storage_reserve_free_gb: float = 40.0,
    scheduler_storage_expected_bytes: int = 0,
    scheduler_proactive_cleanup_roots: Sequence[str] = (),
    scheduler_proactive_cleanup_execute: bool = False,
    scheduler_proactive_cleanup_action: str = "move",
    scheduler_proactive_cleanup_min_bytes: str = "1",
    scheduler_proactive_cleanup_cold_store_roots: Sequence[str] = (),
    scheduler_proactive_cleanup_cold_store_reserve_gb: float = 40.0,
    include_exact_readiness_followup: bool = False,
    require_renderer_payload_dfl1_parity_followup: bool = False,
    exact_readiness_followup_require_ready: bool = False,
    exact_eval_dispatch_require_authorized: bool = False,
    exact_eval_dispatch_provider: str = "lightning",
    exact_eval_dispatch_label_prefix: str = "materializer_exact_eval",
    exact_eval_dispatch_max_total_cost: float = 5.0,
    exact_eval_dispatch_estimated_cost_per_dispatch: float = 0.30,
) -> dict[str, Any]:
    """Compile executable materializer rows into ``experiment_queue.v1``."""

    if work_queue.get("schema") != MATERIALIZER_WORK_QUEUE_SCHEMA:
        raise ExperimentQueueError(f"expected schema {MATERIALIZER_WORK_QUEUE_SCHEMA}")
    if isinstance(local_cpu_concurrency, bool) or local_cpu_concurrency < 1:
        raise ExperimentQueueError("local_cpu_concurrency must be >= 1")
    if isinstance(step_timeout_seconds, bool) or step_timeout_seconds < 0:
        raise ExperimentQueueError("step_timeout_seconds must be non-negative")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ExperimentQueueError("limit must be >= 1 when provided")
    if isinstance(scheduler_storage_expected_bytes, bool) or scheduler_storage_expected_bytes < 0:
        raise ExperimentQueueError("scheduler_storage_expected_bytes must be non-negative")
    if scheduler_proactive_cleanup_action not in {"move", "delete"}:
        raise ExperimentQueueError("scheduler_proactive_cleanup_action must be move or delete")
    if include_scheduler_preflight:
        try:
            validate_scheduler_storage_preflight_config(
                proactive_cleanup_execute=scheduler_proactive_cleanup_execute,
                proactive_cleanup_action=scheduler_proactive_cleanup_action,
                proactive_cleanup_cold_store_roots=tuple(scheduler_proactive_cleanup_cold_store_roots),
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
    if exact_eval_dispatch_provider not in {"lightning", "vastai"}:
        raise ExperimentQueueError("exact_eval_dispatch_provider must be lightning or vastai")
    if exact_eval_dispatch_max_total_cost <= 0:
        raise ExperimentQueueError("exact_eval_dispatch_max_total_cost must be > 0")
    if exact_eval_dispatch_estimated_cost_per_dispatch <= 0:
        raise ExperimentQueueError("exact_eval_dispatch_estimated_cost_per_dispatch must be > 0")
    if include_scheduler_preflight and not scheduler_proactive_cleanup_execute:
        raise ExperimentQueueError(
            "scheduler_proactive_cleanup_execute must be true when scheduler preflight gates materializer execution"
        )
    if require_renderer_payload_dfl1_parity_followup and not include_exact_readiness_followup:
        raise ExperimentQueueError(
            "require_renderer_payload_dfl1_parity_followup requires "
            "include_exact_readiness_followup"
        )

    queue_id = str(queue_id or "").strip()
    if not queue_id:
        raise ExperimentQueueError("queue_id must be a non-empty string")
    repo = Path(repo_root)
    work_queue_ref = (
        _repo_rel(_resolve_repo_path(source_work_queue_path, repo_root=repo), repo)
        if source_work_queue_path is not None
        else None
    )
    state_ref = (
        _repo_rel(_resolve_repo_path(source_state_path, repo_root=repo), repo)
        if source_state_path is not None
        else _repo_rel(default_state_path(repo, queue_id), repo)
    )

    source_rows = [item for item in _as_list(work_queue.get("rows")) if isinstance(item, Mapping)]
    executable_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(source_rows):
        try:
            require_no_truthy_authority_fields(
                row,
                context=f"materializer_work_queue.rows.{index}",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        if row.get("executable") is True:
            executable_rows.append(row)
    if limit is not None:
        executable_rows = executable_rows[:limit]
    if not executable_rows:
        raise ExperimentQueueError("no executable materializer work rows")
    expected_output_root = (
        _expected_materializer_workload_root(
            results_root=scheduler_results_root,
            expected_workload_root=scheduler_storage_expected_workload_root,
        )
        if include_scheduler_preflight
        else None
    )
    if include_scheduler_preflight and expected_output_root is None:
        raise ExperimentQueueError(
            "scheduler_storage_expected_workload_root is required when "
            "scheduler_results_root is relative and scheduler preflight gates "
            "materializer execution"
        )
    if expected_output_root is not None:
        for index, row in enumerate(executable_rows, start=1):
            telemetry = row.get("telemetry")
            artifact_paths = telemetry.get("artifact_paths") if isinstance(telemetry, Mapping) else None
            if not isinstance(artifact_paths, list) or not artifact_paths:
                raise ExperimentQueueError(
                    f"materializer work row {index} has no telemetry artifact_paths for storage preflight"
                )
            for raw_path in artifact_paths:
                path = _resolve_output_path(raw_path, repo_root=repo)
                if path is None or not _path_under_root(path, expected_output_root):
                    raise ExperimentQueueError(
                        "materializer work row "
                        f"{index} artifact path outside scheduler workload root: "
                        f"{raw_path!r} not under {expected_output_root}"
                    )
    if include_exact_readiness_followup and expected_output_root is not None:
        for index, row in enumerate(executable_rows, start=1):
            postconditions = _normalize_materializer_queue_postconditions(row.get("postconditions"))
            if _materializer_exact_readiness_skip_reason(row, postconditions) is not None:
                continue
            manifest_ref_path, _manifest_ref_kind, _saw_manifest_postcondition = (
                _harvestable_materializer_manifest_reference(postconditions)
            )
            if manifest_ref_path is None:
                raise ExperimentQueueError(
                    "include_exact_readiness_followup requires a harvestable "
                    f"materializer manifest postcondition for {row.get('work_id') or index}"
                )
            materializer_manifest_path = _resolve_repo_path(
                manifest_ref_path,
                repo_root=repo,
            )
            handoff_dir = materializer_manifest_path.parent / "exact_eval_handoff"
            if not _path_under_root(handoff_dir, expected_output_root):
                raise ExperimentQueueError(
                    "materializer exact-readiness follow-up path outside "
                    f"scheduler workload root: {handoff_dir} not under {expected_output_root}"
                )
            if (
                row.get("target_kind") == RENDERER_PAYLOAD_DFL1_TARGET_KIND
                and not _renderer_payload_dfl1_parity_followup_blockers(row)
            ):
                dfl1_context = row.get("renderer_payload_dfl1_parity_context")
                assert isinstance(dfl1_context, Mapping)
                raw_parity_output_dir = _context_string_any(dfl1_context, ("output_dir",))
                parity_dir = (
                    _resolve_repo_path(raw_parity_output_dir, repo_root=repo)
                    if raw_parity_output_dir is not None
                    else handoff_dir / "renderer_payload_dfl1_shell_parity"
                )
                if not _path_under_root(parity_dir, expected_output_root):
                    raise ExperimentQueueError(
                        "materializer DFL1 parity artifact path outside scheduler "
                        f"workload root: {parity_dir} not under {expected_output_root}"
                    )

    resource_limits: dict[str, int] = {}
    for key, value in (resource_concurrency or {}).items():
        parsed_limit = _finite_int(value)
        if parsed_limit is None or parsed_limit < 1:
            raise ExperimentQueueError(f"resource_concurrency[{key!r}] must be >= 1")
        resource_limits[str(key)] = parsed_limit

    used_resource_kinds: set[str] = set()
    experiments: list[dict[str, Any]] = []
    seen_experiments: set[str] = set()
    preflight_dependency = (
        f"{MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID}.proactive_cleanup" if include_scheduler_preflight else None
    )
    for rank, row in enumerate(executable_rows, start=1):
        command = row.get("command")
        if not isinstance(command, list) or not command:
            raise ExperimentQueueError(f"materializer work row {rank} command must be non-empty")
        command_items = [str(item) for item in command]
        resource_kind = str(row.get("resource_kind") or "local_cpu")
        if not resource_kind.startswith("local"):
            raise ExperimentQueueError(f"materializer work row {rank} uses non-local resource {resource_kind!r}")
        used_resource_kinds.add(resource_kind)
        experiment_id = _materializer_execution_experiment_id(
            row,
            rank,
            seen_experiments,
        )
        postconditions = _normalize_materializer_queue_postconditions(row.get("postconditions"))
        exact_readiness_skip_reason = (
            _materializer_exact_readiness_skip_reason(row, postconditions) if include_exact_readiness_followup else None
        )
        exact_readiness_followup_enabled = include_exact_readiness_followup and exact_readiness_skip_reason is None
        if exact_readiness_followup_enabled:
            if work_queue_ref is None:
                raise ExperimentQueueError(
                    "source_work_queue_path is required for generated exact-readiness "
                    f"follow-up harvest step {experiment_id}"
                )
            used_resource_kinds.add("local_cpu")
        dfl1_parity_followup_requested = (
            exact_readiness_followup_enabled
            and row.get("target_kind") == RENDERER_PAYLOAD_DFL1_TARGET_KIND
        )
        dfl1_parity_followup_blockers = (
            _renderer_payload_dfl1_parity_followup_blockers(row)
            if dfl1_parity_followup_requested
            else []
        )
        if (
            require_renderer_payload_dfl1_parity_followup
            and dfl1_parity_followup_requested
            and dfl1_parity_followup_blockers
        ):
            raise ExperimentQueueError(
                "renderer_payload_dfl1 parity follow-up is required but blocked for "
                f"{row.get('work_id') or row.get('backlog_key') or experiment_id}: "
                + ", ".join(dfl1_parity_followup_blockers)
            )
        steps = [
            {
                "id": MATERIALIZER_EXECUTION_STEP_ID,
                "kind": "command",
                "command": command_items,
                "requires": [] if preflight_dependency is None else [preflight_dependency],
                "resources": {"kind": resource_kind},
                "postconditions": postconditions,
                "telemetry": dict(row.get("telemetry") or {}),
                "timeout_seconds": step_timeout_seconds,
            }
        ]
        if exact_readiness_followup_enabled:
            manifest_ref_path, _manifest_ref_kind, _saw_manifest_postcondition = (
                _harvestable_materializer_manifest_reference(postconditions)
            )
            if manifest_ref_path is None:
                raise ExperimentQueueError(
                    "include_exact_readiness_followup requires a harvestable "
                    f"materializer manifest postcondition for {row.get('work_id') or experiment_id}"
                )
            materializer_manifest_path = _resolve_repo_path(
                manifest_ref_path,
                repo_root=repo,
            )
            steps.extend(
                _materializer_exact_readiness_followup_steps(
                    queue_id=queue_id,
                    repo_root=repo,
                    row_id=experiment_id,
                    source_work_queue_path=_resolve_repo_path(work_queue_ref, repo_root=repo),
                    source_state_path=_resolve_repo_path(state_ref, repo_root=repo),
                    work_row=row,
                    postconditions=postconditions,
                    handoff_dir=materializer_manifest_path.parent
                    / "exact_eval_handoff",
                    materializer_step_id=MATERIALIZER_EXECUTION_STEP_ID,
                    step_timeout_seconds=step_timeout_seconds,
                    require_ready=exact_readiness_followup_require_ready,
                    dispatch_require_authorized=exact_eval_dispatch_require_authorized,
                    dispatch_provider=exact_eval_dispatch_provider,
                    dispatch_label_prefix=exact_eval_dispatch_label_prefix,
                    dispatch_max_total_cost=exact_eval_dispatch_max_total_cost,
                    dispatch_estimated_cost_per_dispatch=(exact_eval_dispatch_estimated_cost_per_dispatch),
                )
            )
        for step in steps:
            resources = step.get("resources")
            if isinstance(resources, Mapping):
                kind = resources.get("kind")
                if isinstance(kind, str) and kind:
                    used_resource_kinds.add(kind)
        metadata = apply_proxy_evidence_boundary(
            {
                "schema": MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA,
                "source_work_queue_schema": work_queue.get("schema"),
                "source_work_queue_path": work_queue_ref,
                "source_state_path": state_ref,
                "source_plan_path": row.get("source_plan_path"),
                "work_id": row.get("work_id"),
                "work_rank": row.get("work_rank"),
                "backlog_key": row.get("backlog_key"),
                "backlog_rank": row.get("backlog_rank"),
                "unit_kind": row.get("unit_kind"),
                "operation_family": row.get("operation_family"),
                "target_kind": row.get("target_kind"),
                "materializer_id": row.get("materializer_id"),
                "receiver_contract_id": row.get("receiver_contract_id"),
                "receiver_contract_kind": row.get("receiver_contract_kind"),
                "source_unit_ids": _as_list(row.get("source_unit_ids")),
                "source_selection_ids": _as_list(row.get("source_selection_ids")),
                "exact_readiness_followup_requested": bool(include_exact_readiness_followup),
                "exact_readiness_followup_enabled": bool(exact_readiness_followup_enabled),
                "exact_readiness_followup_skipped_reason": exact_readiness_skip_reason,
                "renderer_payload_dfl1_parity_followup_requested": bool(
                    dfl1_parity_followup_requested
                ),
                "renderer_payload_dfl1_parity_followup_enabled": bool(
                    dfl1_parity_followup_requested
                    and not dfl1_parity_followup_blockers
                ),
                "renderer_payload_dfl1_parity_followup_blockers": (
                    dfl1_parity_followup_blockers
                ),
                "candidate_saved_bytes_sum": row.get("candidate_saved_bytes_sum"),
                "expected_score_gain_sum": row.get("expected_score_gain_sum"),
                "allowed_use": "local_materializer_proof_chain_only",
                **FALSE_AUTHORITY,
            },
            dispatch_blockers=(
                "materializer_execution_queue_local_proof_chain_only",
                "exact_auth_eval_required_before_score_claim",
            ),
        )
        experiments.append(
            {
                "id": experiment_id,
                "lane_id": lane_id or row.get("lane_id"),
                "priority": _materializer_execution_priority(row, rank),
                "status": "queued",
                "tags": [
                    "byte-shaving",
                    "materializer",
                    "local-proof-chain",
                    "no-score-authority",
                ],
                "metadata": metadata,
                "steps": steps,
            }
        )
    grouped_experiments, grouped_resource_kinds = (
        _grouped_archive_state_execution_experiments(
            work_queue=work_queue,
            queue_id=queue_id,
            repo_root=repo,
            source_work_queue_path=(
                Path(source_work_queue_path)
                if source_work_queue_path is not None
                else None
            ),
            preflight_dependency=preflight_dependency,
            step_timeout_seconds=step_timeout_seconds,
            lane_id=lane_id,
            seen_experiments=seen_experiments,
        )
    )
    experiments.extend(grouped_experiments)
    used_resource_kinds.update(grouped_resource_kinds)
    for kind in sorted(used_resource_kinds):
        resource_limits.setdefault(
            kind,
            local_cpu_concurrency if kind == "local_cpu" else 1,
        )
    if include_scheduler_preflight:
        stamp = _utc_stamp()
        resource_limits.setdefault("local_cpu", local_cpu_concurrency)
        resource_limits.setdefault("local_io_heavy", 1)
        experiments = [
            build_scheduler_storage_preflight_experiment(
                experiment_id=MATERIALIZER_SCHEDULER_PREFLIGHT_EXPERIMENT_ID,
                lane_id=f"lane_materializer_scheduler_preflight_{stamp}",
                tags=[
                    "byte-shaving",
                    "materializer",
                    "scheduler-preflight",
                    "storage",
                    "cleanup",
                    "no-score-authority",
                ],
                artifact_prefix="byte_shaving_materializer",
                date=stamp,
                results_root=scheduler_results_root,
                storage_tiers=tuple(scheduler_storage_tiers),
                storage_workload_subdir=scheduler_storage_workload_subdir,
                storage_expected_workload_root=scheduler_storage_expected_workload_root,
                storage_reserve_free_gb=scheduler_storage_reserve_free_gb,
                storage_expected_bytes=scheduler_storage_expected_bytes,
                proactive_cleanup_roots=tuple(scheduler_proactive_cleanup_roots),
                proactive_cleanup_execute=scheduler_proactive_cleanup_execute,
                proactive_cleanup_action=scheduler_proactive_cleanup_action,
                proactive_cleanup_min_bytes=scheduler_proactive_cleanup_min_bytes,
                proactive_cleanup_cold_store_roots=tuple(scheduler_proactive_cleanup_cold_store_roots),
                proactive_cleanup_cold_store_reserve_gb=(scheduler_proactive_cleanup_cold_store_reserve_gb),
            ),
            *experiments,
        ]
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": resource_limits,
            },
            "experiments": experiments,
        }
    )


def _materialize_row(
    *,
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    kind: str,
    base_pairs: tuple[int, ...] | None,
    units_by_id: Mapping[str, Mapping[str, Any]],
    rank: int,
) -> dict[str, Any]:
    selection_id = _selection_id(kind, row)
    try:
        require_no_truthy_authority_fields(row, context=f"byte_shaving_campaign.{kind}.{selection_id}")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc

    selected_operations = [item for item in _as_list(row.get("selected_operations")) if isinstance(item, Mapping)]
    if kind == "ranked_unit" and not selected_operations:
        selected_operations = _ranked_unit_selected_operations(row)
    chosen_operation_sequence = [
        item for item in _as_list(row.get("chosen_operation_sequence")) if isinstance(item, Mapping)
    ]
    source_dispatch_blockers = [str(item) for item in _as_list(row.get("dispatch_blockers")) if str(item)]
    blockers: list[str] = []
    if not selected_operations:
        blockers.append("selected_operations_missing")
    if kind == "operation_set" and row.get("schema") != COUPLED_OPERATION_SET_SCHEMA:
        blockers.append("operation_set_schema_mismatch")
    operations_for_materialization = selected_operations
    operation_set_sequence_valid: bool | None = None
    if kind == "operation_set":
        if not chosen_operation_sequence:
            blockers.append("operation_set_chosen_sequence_missing")
        else:
            selected_keys = [_operation_sequence_key(operation) for operation in selected_operations]
            chosen_keys = [_operation_sequence_key(operation) for operation in chosen_operation_sequence]
            operation_set_sequence_valid = Counter(selected_keys) == Counter(chosen_keys)
            if not operation_set_sequence_valid:
                blockers.append("operation_set_sequence_not_permutation_of_selected_operations")
            else:
                selected_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
                for operation in selected_operations:
                    selected_by_key.setdefault(_operation_sequence_key(operation), []).append(operation)
                ordered_operations: list[Mapping[str, Any]] = []
                for key in chosen_keys:
                    ordered_operations.append(selected_by_key[key].pop(0))
                operations_for_materialization = ordered_operations

    dropped_pairs: list[int] = []
    dqs1_operation_count = 0
    materializer_resolutions: list[dict[str, Any]] = []
    source_units: list[dict[str, Any]] = []
    pairset_selector_selected_pairs: tuple[int, ...] | None = None
    for operation in operations_for_materialization:
        unit_id = str(operation.get("unit_id") or "")
        unit = units_by_id.get(unit_id)
        operation_params = _operation_params(operation, unit)
        resolution = resolve_materializer(operation=operation, unit=unit)
        is_dqs1_pairset_selector = (
            resolution.target_kind == DQS1_PAIRSET_TARGET_KIND
            and isinstance(operation_params.get("dropped_pair_indices"), list)
            and isinstance(operation_params.get("selected_pair_indices"), list)
        )
        operation_blockers = _local_dqs1_materialization_blockers(
            _as_list(operation.get("blockers")),
            clear_planning_pairset_blockers=is_dqs1_pairset_selector,
        )
        blockers.extend(
            f"selected_operation_blocker:{unit_id or '<missing>'}:{blocker}" for blocker in operation_blockers
        )
        resolution_blockers = list(resolution.blockers)
        if resolution.target_kind and resolution.target_kind != DQS1_PAIRSET_TARGET_KIND and resolution.executable:
            resolution_blockers.append(f"non_dqs1_target_requires_materializer_work_queue:{resolution.target_kind}")
        blockers.extend(resolution_blockers)
        materializer_resolutions.append(
            {
                "unit_id": resolution.unit_id,
                "unit_kind": resolution.unit_kind,
                "operation_id": resolution.operation_id,
                "operation_family": resolution.operation_family,
                "explicit_materializer": resolution.explicit_materializer,
                "materializer_id": resolution.materializer_id,
                "target_kind": resolution.target_kind,
                "receiver_contract_id": resolution.receiver_contract_id,
                "receiver_contract_kind": resolution.receiver_contract_kind,
                "cooperative_receiver_required": resolution.cooperative_receiver_required,
                "materialization_resource_kind": resolution.materialization_resource_kind,
                "executable": resolution.executable,
                "blockers": resolution_blockers,
                "selected_operation_blockers": operation_blockers,
            }
        )
        if unit is None:
            pass
        else:
            unit_kind = str(unit.get("unit_kind") or "")
            unit_blockers = ordered_unique(
                [
                    *[str(item) for item in _as_list(unit.get("blockers"))],
                    *[str(item) for item in _as_list(unit.get("materialization_blockers"))],
                    *[str(item) for item in _as_list(unit.get("candidate_trust_region_blockers"))],
                ]
            )
            unit_blockers = _local_dqs1_materialization_blockers(
                unit_blockers,
                clear_planning_pairset_blockers=is_dqs1_pairset_selector,
            )
            blockers.extend(f"selected_unit_blocker:{unit_id}:{blocker}" for blocker in unit_blockers)
            source_units.append(
                {
                    "unit_id": unit_id,
                    "unit_kind": unit_kind,
                    "candidate_saved_bytes": unit.get("candidate_saved_bytes"),
                    "score_axis": unit.get("score_axis"),
                    "source_paths": unit.get("source_paths"),
                    "source_candidate_id": unit.get("source_candidate_id"),
                    "candidate_archive_sha256": unit.get("candidate_archive_sha256"),
                    "candidate_archive_bytes": unit.get("candidate_archive_bytes"),
                    "operation_params": operation_params,
                    "candidate_trust_region_blockers": unit.get("candidate_trust_region_blockers"),
                    "blockers": unit_blockers,
                }
            )
        if resolution.target_kind != DQS1_PAIRSET_TARGET_KIND:
            continue
        dqs1_operation_count += 1
        if is_dqs1_pairset_selector:
            op_dropped_pairs = _pair_index_list_from_params(
                operation_params,
                "dropped_pair_indices",
                label=f"{unit_id or operation.get('operation_id')}.dropped_pair_indices",
            )
            op_selected_pairs = _pair_index_list_from_params(
                operation_params,
                "selected_pair_indices",
                label=f"{unit_id or operation.get('operation_id')}.selected_pair_indices",
            )
            if op_dropped_pairs is None:
                blockers.append(f"dropped_pair_indices_missing:{unit_id or operation.get('operation_id')}")
                continue
            if op_selected_pairs is None:
                blockers.append(f"selected_pair_indices_missing:{unit_id or operation.get('operation_id')}")
                continue
            dropped_pairs.extend(op_dropped_pairs)
            if pairset_selector_selected_pairs is None:
                pairset_selector_selected_pairs = op_selected_pairs
            elif pairset_selector_selected_pairs != op_selected_pairs:
                blockers.append("dqs1_pairset_selector_selected_pairs_mismatch")
            continue
        pair_index = _pair_index_from_operation(operation)
        if pair_index is None:
            blockers.append(f"pair_index_missing:{operation.get('unit_id') or operation.get('operation_id')}")
            continue
        if not 0 <= pair_index < FEC6_PAIR_COUNT:
            blockers.append(f"pair_index_out_of_range:{pair_index}")
            continue
        dropped_pairs.append(pair_index)
    target_kinds = ordered_unique(
        resolution["target_kind"] for resolution in materializer_resolutions if resolution["target_kind"]
    )
    known_target_kinds = known_materializer_target_kinds()
    unsupported_targets = [target for target in target_kinds if target not in known_target_kinds]
    if unsupported_targets:
        blockers.append("unsupported_materializer_target:" + ",".join(unsupported_targets))
    operation_set_materialization_mode: str | None = None
    if kind == "operation_set":
        can_clear_operation_set_source_blockers = (
            operation_set_sequence_valid is True
            and bool(materializer_resolutions)
            and all(
                resolution["target_kind"] == DQS1_PAIRSET_TARGET_KIND and resolution["executable"] is True
                for resolution in materializer_resolutions
            )
        )
        if can_clear_operation_set_source_blockers:
            operation_set_materialization_mode = "ordered_dqs1_pairset_sequence"
            enforced_source_blockers = [
                blocker
                for blocker in source_dispatch_blockers
                if blocker in OPERATION_SET_ENFORCED_SOURCE_BLOCKERS
                and blocker not in OPERATION_SET_CLEARABLE_SOURCE_BLOCKERS
            ]
        else:
            operation_set_materialization_mode = "blocked_or_requires_atomic_materializer"
            enforced_source_blockers = [
                blocker for blocker in source_dispatch_blockers if blocker in OPERATION_SET_ENFORCED_SOURCE_BLOCKERS
            ]
        blockers.extend(f"source_dispatch_blocker:{blocker}" for blocker in enforced_source_blockers)
    if dqs1_operation_count and base_pairs is None:
        blockers.append("dqs1_base_pair_indices_required")
    dropped_pairs = sorted(set(dropped_pairs))
    if (
        dqs1_operation_count
        and pairset_selector_selected_pairs is None
        and len(dropped_pairs) != dqs1_operation_count
    ):
        blockers.append("dropped_pair_indices_do_not_match_selected_operations")
    if base_pairs is not None and dqs1_operation_count:
        missing = [pair for pair in dropped_pairs if pair not in set(base_pairs)]
        if missing:
            blockers.append("dropped_pair_not_in_base:" + ",".join(str(pair) for pair in missing))
        selected_pairs = tuple(pair for pair in base_pairs if pair not in set(dropped_pairs))
        if not selected_pairs:
            blockers.append("selected_pair_indices_empty_after_drop")
        if pairset_selector_selected_pairs is not None and selected_pairs != pairset_selector_selected_pairs:
            blockers.append("dqs1_pairset_selector_selected_pairs_not_base_minus_drop")
    else:
        selected_pairs = pairset_selector_selected_pairs or ()

    conflict_violations = _as_list(row.get("conflict_violations"))
    if conflict_violations:
        blockers.append("conflict_violations_present")
    packet_ir_operation_set = _packet_ir_operation_set_for_row(payload, row)
    if kind == "operation_set":
        blockers.extend(_packet_ir_operation_set_blockers(row, packet_ir_operation_set))
    blockers = ordered_unique(blockers)
    executable = not blockers
    candidate_id = (
        _candidate_id(kind, selection_id, dropped_pairs) if dropped_pairs else f"pairset_byte_shave_{kind}_{rank:04d}"
    )
    try:
        slug = candidate_slug(candidate_id)
    except ExperimentQueueError:
        blockers = ordered_unique([*blockers, f"unsupported_candidate_id:{candidate_id}"])
        executable = False
        slug = candidate_id.removeprefix("pairset_")

    base = {
        "schema": "byte_shaving_campaign_materialization_row.v1",
        "candidate_id": candidate_id,
        "candidate_slug": slug,
        "campaign_id": payload.get("campaign_id"),
        "lane_id": payload.get("lane_id"),
        "selection_kind": kind,
        "selection_id": selection_id,
        "operator_action_rank": rank,
        "source_plan_schema": payload.get("schema"),
        "selected_operations": selected_operations,
        "operation_set_id": row.get("operation_set_id"),
        "packet_ir_operation_set": (dict(packet_ir_operation_set) if packet_ir_operation_set is not None else None),
        "chosen_operation_sequence": chosen_operation_sequence,
        "chosen_operation_sequence_sha256": row.get("chosen_operation_sequence_sha256"),
        "chosen_operation_sequence_source": row.get("chosen_operation_sequence_source"),
        "chosen_operation_sequence_is_permutation": operation_set_sequence_valid,
        "operation_set_materialization_mode": operation_set_materialization_mode,
        "active_interactions": _as_list(row.get("active_interactions")),
        "selected_unit_ids": _as_list(row.get("selected_unit_ids")),
        "operation_families": _as_list(row.get("operation_families")),
        "source_dispatch_blockers": source_dispatch_blockers,
        "materializer_registry_schema": REGISTRY_SCHEMA,
        "materializer_resolutions": materializer_resolutions,
        "materializer_target_kinds": target_kinds,
        "source_units": source_units,
        "base_pair_indices": list(base_pairs or []),
        "dropped_pair_indices": dropped_pairs,
        "selected_pair_indices": list(selected_pairs),
        "selected_pair_count": len(selected_pairs),
        "expected_delta_score": row.get("expected_delta_score"),
        "expected_score_gain": row.get("expected_score_gain"),
        "candidate_saved_bytes": row.get("candidate_saved_bytes"),
        "source_row": {
            "selection_kind": kind,
            "selection_id": selection_id,
            "operation_set_id": row.get("operation_set_id"),
            "packet_ir_operation_set": (dict(packet_ir_operation_set) if packet_ir_operation_set is not None else None),
            "combo_id": row.get("combo_id"),
            "sweep_id": row.get("sweep_id"),
            "chosen_operation_sequence": chosen_operation_sequence,
            "chosen_operation_sequence_sha256": row.get("chosen_operation_sequence_sha256"),
            "chosen_operation_sequence_source": row.get("chosen_operation_sequence_source"),
            "chosen_operation_sequence_is_permutation": operation_set_sequence_valid,
            "operation_set_materialization_mode": operation_set_materialization_mode,
            "active_interactions": _as_list(row.get("active_interactions")),
        },
        "executable": executable,
        "materialization_blockers": blockers,
        **FALSE_AUTHORITY,
    }
    return apply_proxy_evidence_boundary(
        base,
        dispatch_blockers=[] if executable else blockers,
    )


def compile_dqs1_byte_shaving_campaign(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
    plan_path: str | Path | None = None,
    base_pair_indices: Sequence[int] | None = None,
    candidate_limit: int | None = None,
    portfolio_json: str | None = None,
    allow_partial_materialization: bool = False,
    partial_materialization_rationale: str | None = None,
    materializer_contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile supported DQS1 pair drops into queue-builder action surfaces."""

    if payload.get("schema") != PLAN_SCHEMA:
        raise ExperimentQueueError(f"expected schema {PLAN_SCHEMA}")
    try:
        require_no_truthy_authority_fields(payload, context="byte_shaving_campaign_plan")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    if candidate_limit is not None and (isinstance(candidate_limit, bool) or candidate_limit < 1):
        raise ExperimentQueueError("candidate_limit must be >= 1 when provided")
    rationale = str(partial_materialization_rationale or "").strip()
    if allow_partial_materialization and not rationale:
        raise ExperimentQueueError(
            "partial_materialization_rationale is required when partial materialization is allowed"
        )
    repo = Path(repo_root)
    base_pairs = _base_pair_indices(payload, base_pair_indices)
    units_by_id = {
        str(unit.get("unit_id") or ""): unit
        for unit in _as_list(payload.get("ranked_units"))
        if isinstance(unit, Mapping) and str(unit.get("unit_id") or "")
    }
    plan_ref = _repo_rel(Path(plan_path), repo) if plan_path is not None else None
    compiled_rows: list[dict[str, Any]] = []
    for rank, (kind, row) in enumerate(_iter_plan_rows(payload), start=1):
        compiled_rows.append(
            _materialize_row(
                payload=payload,
                row=row,
                kind=kind,
                base_pairs=base_pairs,
                units_by_id=units_by_id,
                rank=rank,
            )
        )

    executable_rows = [row for row in compiled_rows if row["executable"] is True]
    if candidate_limit is not None:
        executable_rows = executable_rows[:candidate_limit]
    blocked_rows = [row for row in compiled_rows if row["executable"] is not True]
    base_materializer_backlog = build_materializer_backlog(compiled_rows)
    packet_ir_materializer_backlog_rows = [
        backlog_row
        for packet_ir in _as_list(payload.get("packet_ir_operation_sets"))
        if isinstance(packet_ir, Mapping)
        for backlog_row in lower_packetir_operation_set_to_backlog_rows(packet_ir)
    ]
    packet_ir_materializer_backlog_rows.extend(
        _high_level_context_packet_ir_rows(
            base_materializer_backlog,
            materializer_contexts,
            source_plan_path=plan_ref,
        )
    )
    materializer_backlog = _merge_packet_ir_materializer_backlog_rows(
        base_materializer_backlog,
        packet_ir_materializer_backlog_rows,
    )
    materializer_backlog_summary = summarize_materializer_backlog(materializer_backlog)
    materializer_work_queue = build_materializer_work_queue(
        materializer_backlog,
        repo_root=repo,
        contexts=materializer_contexts,
        source_plan_path=plan_ref,
    )
    partial_materialization_blockers: list[str] = []
    if blocked_rows and executable_rows and not allow_partial_materialization:
        partial_materialization_blockers.append("partial_materialization_requires_explicit_allow")
    queueable_rows = executable_rows if allow_partial_materialization or not partial_materialization_blockers else []

    portfolio_rows = []
    top_actions = []
    for rank, row in enumerate(queueable_rows, start=1):
        metadata = {
            "schema": "byte_shaving_campaign_dqs1_source_metadata.v1",
            "selector_kind": "byte_shaving_campaign_drop_pair",
            "selected_pair_count": row["selected_pair_count"],
            "selected_pair_indices": row["selected_pair_indices"],
            "base_pair_indices": row["base_pair_indices"],
            "dropped_pair_indices": row["dropped_pair_indices"],
            "selected_operations": row["selected_operations"],
            "operation_set_id": row.get("operation_set_id"),
            "packet_ir_operation_set": row.get("packet_ir_operation_set"),
            "chosen_operation_sequence": row.get("chosen_operation_sequence"),
            "chosen_operation_sequence_sha256": row.get("chosen_operation_sequence_sha256"),
            "chosen_operation_sequence_source": row.get("chosen_operation_sequence_source"),
            "chosen_operation_sequence_is_permutation": row.get("chosen_operation_sequence_is_permutation"),
            "operation_set_materialization_mode": row.get("operation_set_materialization_mode"),
            "active_interactions": row.get("active_interactions"),
            "materializer_resolutions": row["materializer_resolutions"],
            "source_units": row["source_units"],
            "selection_kind": row["selection_kind"],
            "selection_id": row["selection_id"],
            "source_plan_path": plan_ref,
            "source_plan_schema": payload.get("schema"),
            "materializer_registry_schema": REGISTRY_SCHEMA,
            "receiver_contracts": ordered_unique(
                resolution["receiver_contract_id"]
                for resolution in row["materializer_resolutions"]
                if resolution.get("receiver_contract_id")
            ),
            "cooperative_receiver_required": any(
                bool(resolution.get("cooperative_receiver_required")) for resolution in row["materializer_resolutions"]
            ),
            "allowed_use": "dqs1_local_first_materialization_only",
            **FALSE_AUTHORITY,
        }
        portfolio_rows.append(
            apply_proxy_evidence_boundary(
                {
                    "candidate_id": row["candidate_id"],
                    "operator_next_action": SAFE_OPERATOR_ACTION,
                    "operator_action_rank": rank,
                    "source_kind": "byte_shaving_campaign_dqs1_materialization",
                    "source_metadata": metadata,
                    "local_materialization_ready": True,
                    "expected_delta_score": row.get("expected_delta_score"),
                    "expected_score_gain": row.get("expected_score_gain"),
                    "candidate_saved_bytes": row.get("candidate_saved_bytes"),
                    **FALSE_AUTHORITY,
                },
                dispatch_blockers=(
                    "dqs1_local_first_materialization_only",
                    "exact_auth_eval_required_before_score_claim",
                ),
            )
        )
        top_actions.append(
            apply_proxy_evidence_boundary(
                {
                    "candidate_id": row["candidate_id"],
                    "operator_next_action": SAFE_OPERATOR_ACTION,
                    "operator_action_rank": rank,
                    "source_kind": "byte_shaving_campaign_dqs1_materialization",
                    "local_materialization_ready": True,
                    **FALSE_AUTHORITY,
                },
                dispatch_blockers=(
                    "dqs1_local_first_materialization_only",
                    "exact_auth_eval_required_before_score_claim",
                ),
            )
        )

    portfolio = apply_proxy_evidence_boundary(
        {
            "schema": PORTFOLIO_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "source_plan_path": plan_ref,
            "materializer_registry": registry_manifest(),
            "operator_action_rows": portfolio_rows,
            "blocked_rows": blocked_rows,
            "queueable_row_count": len(queueable_rows),
            "partial_materialization_allowed": bool(allow_partial_materialization),
            "partial_materialization_rationale": rationale or None,
            "partial_materialization_blockers": partial_materialization_blockers,
            "materializer_backlog_summary": materializer_backlog_summary,
            "materializer_work_queue": materializer_work_queue,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(partial_materialization_blockers or ["exact_auth_eval_required_before_score_claim"]),
    )
    action_summary = apply_proxy_evidence_boundary(
        {
            "schema": ACTION_SUMMARY_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "campaign_id": payload.get("campaign_id"),
            "lane_id": payload.get("lane_id"),
            "portfolio_json": portfolio_json,
            "materializer_registry": registry_manifest(),
            "top_operator_actions": top_actions,
            "blocked_row_count": len(blocked_rows),
            "executable_row_count": len(executable_rows),
            "queueable_row_count": len(queueable_rows),
            "partial_materialization_allowed": bool(allow_partial_materialization),
            "partial_materialization_rationale": rationale or None,
            "partial_materialization_blockers": partial_materialization_blockers,
            "materializer_backlog_summary": materializer_backlog_summary,
            "materializer_work_queue": materializer_work_queue,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(partial_materialization_blockers or ["exact_auth_eval_required_before_score_claim"]),
    )
    return apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZATION_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "source_plan_path": plan_ref,
            "candidate_limit": candidate_limit,
            "base_pair_indices": list(base_pairs or []),
            "materializer_registry": registry_manifest(),
            "compiled_row_count": len(compiled_rows),
            "executable_row_count": len(executable_rows),
            "blocked_row_count": len(blocked_rows),
            "queueable_row_count": len(queueable_rows),
            "partial_materialization_allowed": bool(allow_partial_materialization),
            "partial_materialization_rationale": rationale or None,
            "partial_materialization_blockers": partial_materialization_blockers,
            "materializer_backlog": materializer_backlog,
            "packet_ir_materializer_backlog_row_count": len(packet_ir_materializer_backlog_rows),
            "packet_ir_materializer_backlog_rows": packet_ir_materializer_backlog_rows,
            "materializer_backlog_summary": materializer_backlog_summary,
            "materializer_work_queue": materializer_work_queue,
            "executable_rows": executable_rows,
            "blocked_rows": blocked_rows,
            "portfolio": portfolio,
            "action_summary": action_summary,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(partial_materialization_blockers or ["exact_auth_eval_required_before_score_claim"]),
    )


__all__ = [
    "ACTION_SUMMARY_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "MATERIALIZER_BACKLOG_SCHEMA",
    "MATERIALIZER_CONTEXTS_SCHEMA",
    "MATERIALIZER_DISPATCH_PLAN_STEP_ID",
    "MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA",
    "MATERIALIZER_EXECUTION_STEP_ID",
    "MATERIALIZER_HARVEST_STEP_ID",
    "MATERIALIZER_WORK_QUEUE_SCHEMA",
    "PORTFOLIO_SCHEMA",
    "build_materializer_backlog",
    "build_materializer_execution_queue",
    "build_materializer_work_queue",
    "compile_dqs1_byte_shaving_campaign",
    "lower_packetir_operation_set_to_backlog_rows",
    "materializer_contexts_from_payload",
    "summarize_materializer_backlog",
]
