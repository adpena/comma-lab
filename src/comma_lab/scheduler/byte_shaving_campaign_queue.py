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

from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME,
    CHAIN_SCHEMA,
)
from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY, PLAN_SCHEMA
from tac.optimization.decoder_q_selective_runtime_packet import FEC6_PAIR_COUNT
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_MANIFEST_NAME as INVERSE_SCORER_CELL_CHAIN_MANIFEST,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_SCHEMA as INVERSE_SCORER_CELL_CHAIN_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)

from .byte_shaving_materializer_registry import (
    DQS1_PAIRSET_TARGET_KIND,
    INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND,
    INVERSE_SCORER_CELL_TARGET_KIND,
    REGISTRY_SCHEMA,
    known_materializer_target_kinds,
    registry_manifest,
    resolve_materializer,
    suggest_materializer_adapters,
)
from .dqs1_local_first_queue import SAFE_OPERATOR_ACTION, candidate_slug
from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

MATERIALIZATION_SCHEMA = "byte_shaving_campaign_materialization.v1"
MATERIALIZER_BACKLOG_SCHEMA = "byte_shaving_materializer_backlog.v1"
MATERIALIZER_CONTEXTS_SCHEMA = "byte_shaving_materializer_contexts.v1"
MATERIALIZER_WORK_QUEUE_SCHEMA = "byte_shaving_materializer_work_queue.v1"
MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA = (
    "byte_shaving_materializer_execution_experiment_metadata.v1"
)
ACTION_SUMMARY_SCHEMA = "cross_family_candidate_portfolio_action_summary.v1"
PORTFOLIO_SCHEMA = "byte_shaving_campaign_dqs1_operator_portfolio.v1"
TOOL_NAME = "comma_lab.scheduler.byte_shaving_campaign_queue"
BYTE_RANGE_CHAIN_TOOL = "tools/run_byte_range_entropy_recode_chain.py"
INVERSE_ACTION_FUNCTIONAL_TOOL = "tools/build_inverse_steganalysis_action_functional.py"
INVERSE_SCORER_CELL_TOOL = "tools/materialize_inverse_scorer_cell_candidate.py"
INVERSE_SCORER_CELL_CHAIN_TOOL = "tools/run_inverse_scorer_cell_candidate_chain.py"
BYTE_RANGE_CHAIN_MANIFEST = CHAIN_MANIFEST_NAME
INVERSE_ACTION_FUNCTIONAL_SCHEMA = "inverse_steganalysis_discrete_action_functional.v1"
INVERSE_SCORER_CELL_CANDIDATE_SCHEMA = "inverse_scorer_cell_candidate_v1"
MATERIALIZER_EXECUTION_STEP_ID = "materialize_local_proof_chain"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
        raise ExperimentQueueError(
            f"{label}: pair indices out of range 0..{FEC6_PAIR_COUNT - 1}: {bad}"
        )
    return pairs


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
    value = row.get("combo_id") or row.get("sweep_id") or row.get("selection_id")
    if isinstance(value, str) and value.strip():
        return value
    return kind


def _iter_plan_rows(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for kind, key in (("combo", "combination_ladder"), ("prefix", "sweep_ladder")):
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
    return {
        str(unit.get("unit_id") or ""): unit
        for unit in source_units
        if str(unit.get("unit_id") or "")
    }


def _resolution_gap_class(resolution: Mapping[str, Any], blockers: Sequence[str]) -> str:
    joined = "\n".join(blockers)
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
    if gap_class in {"operation_family_missing", "unknown_operation_family"}:
        return "receiver_operation_contract_invalid"
    return "receiver_contract_blocked"


def _backlog_row_sort_key(row: Mapping[str, Any]) -> tuple[float, int, int, str]:
    return (
        -float(row.get("expected_score_gain_sum") or 0.0),
        -int(row.get("candidate_saved_bytes_sum") or 0),
        -int(row.get("blocked_row_count") or 0),
        str(row.get("backlog_key") or ""),
    )


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
        source_units = [
            unit for unit in _as_list(row.get("source_units")) if isinstance(unit, Mapping)
        ]
        unit_blockers = _unit_blockers_by_id(source_units)
        source_units_by_id = _units_by_id(source_units)
        resolutions = [
            item for item in _as_list(row.get("materializer_resolutions")) if isinstance(item, Mapping)
        ]
        gain_share = row_expected_gain / float(max(1, len(resolutions)))
        saved_share = row_saved_bytes // max(1, len(resolutions))
        for resolution in resolutions:
            unit_id = str(resolution.get("unit_id") or "")
            unit_saved_bytes = _finite_int(
                source_units_by_id.get(unit_id, {}).get("candidate_saved_bytes")
            )
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
                    "cooperative_receiver_required": bool(
                        resolution.get("cooperative_receiver_required")
                    ),
                    "materialization_resource_kind": resolution.get(
                        "materialization_resource_kind"
                    ),
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
                    **FALSE_AUTHORITY,
                }
                rows[key] = current
                seen_selection_by_key[key] = set()
                seen_unit_by_key[key] = set()
            current["blocked_resolution_count"] = (
                int(current["blocked_resolution_count"]) + 1
            )
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
                    "cooperative_receiver_required": row.get(
                        "cooperative_receiver_required"
                    ),
                    "materialization_resource_kind": row.get(
                        "materialization_resource_kind"
                    ),
                    "suggested_materializer_count": row.get(
                        "suggested_materializer_count"
                    ),
                    "suggested_materializers": row.get("suggested_materializers"),
                    "blocked_row_count": row.get("blocked_row_count"),
                    "blocked_resolution_count": row.get("blocked_resolution_count"),
                    "selected_operation_count": row.get("selected_operation_count"),
                    "affected_unit_count": row.get("affected_unit_count"),
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


def _first_suggested_materializer(row: Mapping[str, Any]) -> Mapping[str, Any]:
    suggestions = [
        item for item in _as_list(row.get("suggested_materializers")) if isinstance(item, Mapping)
    ]
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
            raise ExperimentQueueError(
                f"materializer context row {index} must include object field 'context'"
            )
        try:
            require_no_truthy_authority_fields(
                context,
                context=f"materializer_contexts.rows.{index}.context",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        keys = _context_keys_from_row(row)
        if not keys:
            raise ExperimentQueueError(
                f"materializer context row {index} must declare at least one key"
            )
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
    if not scorer_responses and not inverse_surfaces:
        blockers.append(
            "materializer_context_missing:scorer_response_or_inverse_scorer_surface"
        )
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
    for path in _path_list_context_value(context, "queue_performance_summary"):
        command.extend(["--queue-performance-summary", path])

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
    return command, [], {
        "artifact_paths": telemetry_paths,
        "include_postcondition_paths": True,
    }


def _inverse_scorer_cell_candidate_command(
    context: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    blockers: list[str] = []
    template = _path_context_value(context, "candidate_archive_template")
    if template is None:
        blockers.append("materializer_context_missing:candidate_archive_template")
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
    if blockers:
        return [], blockers, {}

    assert template is not None
    assert action_functional is not None
    assert isinstance(raw_digest, str)
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
        source_inflate_output_dir = _path_context_value(context, "source_inflate_output_dir")
        if source_inflate_output_dir is not None:
            command.extend(["--source-inflate-output-dir", source_inflate_output_dir])
        candidate_inflate_output_dir = _path_context_value(
            context,
            "candidate_inflate_output_dir",
        )
        if candidate_inflate_output_dir is not None:
            command.extend(["--candidate-inflate-output-dir", candidate_inflate_output_dir])
        inflate_runtime_dir = _path_context_value(context, "inflate_runtime_dir")
        if inflate_runtime_dir is not None:
            command.extend(["--inflate-runtime-dir", inflate_runtime_dir])
        source_archive_for_parity = _path_context_value(context, "source_archive_for_parity")
        if source_archive_for_parity is not None:
            command.extend(["--source-archive-for-parity", source_archive_for_parity])
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
        if context.get("fail_if_inflate_parity_blocked") is True:
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
        return command, [], {
            "artifact_paths": artifact_paths,
            "recursive": True,
            "max_recursive_entries": 512,
            "include_postcondition_paths": True,
            "parity_probe_required": (
                _path_context_value(context, "source_inflate_output_dir") is not None
                or _path_context_value(context, "candidate_inflate_output_dir") is not None
                or _path_context_value(context, "inflate_runtime_dir") is not None
            ),
        }
    return command, [], {
        "artifact_paths": [output_archive, manifest_out],
        "include_postcondition_paths": True,
    }


def _materializer_work_dispatch_blockers(target_kind: str) -> tuple[str, ...]:
    blockers = ["materializer_work_queue_local_proof_chain_only"]
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
    blockers.append("exact_auth_eval_required_before_score_claim")
    return tuple(blockers)


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
            blockers.append(
                "materializer_context_ambiguous:"
                + ",".join(key for key, _context in context_matches)
            )
            context: Mapping[str, Any] = {}
        else:
            context = context_matches[0][1] if context_matches else {}
        command: list[str] = []
        postcondition: dict[str, Any] | None = None
        telemetry: dict[str, Any] = {}
        if (
            unit_kind == "byte_range"
            and operation_family == "entropy_recode"
            and target_kind == "byte_range_entropy_recode_v1"
        ):
            command, command_blockers = _byte_range_chain_command(context)
            blockers.extend(command_blockers)
            if command:
                postcondition = {
                    "type": "json_equals",
                    "path": str(
                        Path(context["output_dir"]) / BYTE_RANGE_CHAIN_MANIFEST
                    ),
                    "key": "schema",
                    "equals": CHAIN_SCHEMA,
                }
                telemetry = {
                    "artifact_paths": [str(context["output_dir"])],
                    "recursive": True,
                    "max_recursive_entries": 512,
                    "include_postcondition_paths": True,
                }
        elif (
            unit_kind == "scorer_inverse_surface_cell"
            and operation_family == "probe_inverse_scorer_surface_cell"
            and target_kind == INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND
        ):
            command, command_blockers, telemetry = (
                _inverse_scorer_action_functional_command(context)
            )
            blockers.extend(command_blockers)
            output = _path_context_value(context, "output")
            if command and output is not None:
                postcondition = {
                    "type": "json_equals",
                    "path": output,
                    "key": "schema",
                    "equals": INVERSE_ACTION_FUNCTIONAL_SCHEMA,
                }
        elif (
            unit_kind == "scorer_inverse_surface_cell"
            and operation_family == "materialize_inverse_scorer_cell_candidate"
            and target_kind == INVERSE_SCORER_CELL_TARGET_KIND
        ):
            command, command_blockers, telemetry = _inverse_scorer_cell_candidate_command(
                context
            )
            blockers.extend(command_blockers)
            output_dir = _path_context_value(context, "chain_output_dir")
            if output_dir is None:
                output_dir = _path_context_value(context, "output_dir")
            manifest_out = _path_context_value(context, "manifest_out")
            if command and output_dir is not None:
                postcondition = {
                    "type": "json_equals",
                    "path": str(Path(output_dir) / INVERSE_SCORER_CELL_CHAIN_MANIFEST),
                    "key": "schema",
                    "equals": INVERSE_SCORER_CELL_CHAIN_SCHEMA,
                }
            elif command and manifest_out is not None:
                postcondition = {
                    "type": "json_equals",
                    "path": manifest_out,
                    "key": "schema",
                    "equals": INVERSE_SCORER_CELL_CANDIDATE_SCHEMA,
                }
        else:
            blockers.append(
                f"materializer_work_queue_adapter_missing:{unit_kind}:{operation_family}:{target_kind or '<target_tbd>'}"
            )
        if not context:
            blockers.append(f"materializer_context_missing:{backlog_key}")
        blockers = ordered_unique(blockers)
        executable = not blockers
        work_rows.append(
            apply_proxy_evidence_boundary(
                {
                    "schema": "byte_shaving_materializer_work_row.v1",
                    "work_id": _materializer_work_id(backlog_key),
                    "work_rank": rank,
                    "backlog_key": backlog_key,
                    "backlog_rank": row.get("backlog_rank"),
                    "source_plan_path": source_plan_path,
                    "repo_root": _repo_rel(repo, repo),
                    "unit_kind": unit_kind,
                    "operation_family": operation_family,
                    "target_kind": target_kind or None,
                    "materializer_id": materializer_id or None,
                    "receiver_contract_id": (
                        row.get("receiver_contract_id")
                        or suggestion.get("receiver_contract_id")
                    ),
                    "receiver_contract_kind": (
                        row.get("receiver_contract_kind")
                        or suggestion.get("receiver_contract_kind")
                    ),
                    "tool": command[1] if command else None,
                    "command": command,
                    "postconditions": [] if postcondition is None else [postcondition],
                    "telemetry": telemetry,
                    "resource_kind": row.get("materialization_resource_kind")
                    or suggestion.get("materialization_resource_kind")
                    or "local_cpu",
                    "source_unit_ids": _as_list(row.get("source_unit_ids")),
                    "source_selection_ids": _as_list(row.get("source_selection_ids")),
                    "candidate_saved_bytes_sum": row.get("candidate_saved_bytes_sum"),
                    "expected_score_gain_sum": row.get("expected_score_gain_sum"),
                    "executable": executable,
                    "materialization_blockers": blockers,
                    **FALSE_AUTHORITY,
                },
                dispatch_blockers=(
                    _materializer_work_dispatch_blockers(target_kind)
                    if executable
                    else blockers
                ),
            )
        )
    executable_rows = [row for row in work_rows if row["executable"] is True]
    blocked_rows = [row for row in work_rows if row["executable"] is not True]
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
            "rows": work_rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            []
            if executable_rows
            else ("materializer_work_queue_has_no_executable_rows",)
        ),
    )


def _normalize_materializer_queue_postconditions(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    postconditions: list[dict[str, Any]] = []
    for index, raw_condition in enumerate(value):
        if not isinstance(raw_condition, Mapping):
            raise ExperimentQueueError(
                f"materializer work row postconditions[{index}] must be an object"
            )
        condition = dict(raw_condition)
        condition_type = str(condition.get("type") or "")
        if condition_type == "json_file_key_equals":
            if "value" not in condition:
                raise ExperimentQueueError(
                    "json_file_key_equals postcondition must include value"
                )
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


def build_materializer_execution_queue(
    work_queue: Mapping[str, Any],
    *,
    queue_id: str,
    repo_root: str | Path,
    lane_id: str | None = None,
    source_work_queue_path: str | Path | None = None,
    local_cpu_concurrency: int = 1,
    resource_concurrency: Mapping[str, int] | None = None,
    step_timeout_seconds: int = 0,
    limit: int | None = None,
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

    queue_id = str(queue_id or "").strip()
    if not queue_id:
        raise ExperimentQueueError("queue_id must be a non-empty string")
    repo = Path(repo_root)
    work_queue_ref = (
        _repo_rel(Path(source_work_queue_path), repo)
        if source_work_queue_path is not None
        else None
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

    resource_limits: dict[str, int] = {}
    for key, value in (resource_concurrency or {}).items():
        parsed_limit = _finite_int(value)
        if parsed_limit is None or parsed_limit < 1:
            raise ExperimentQueueError(f"resource_concurrency[{key!r}] must be >= 1")
        resource_limits[str(key)] = parsed_limit

    used_resource_kinds: set[str] = set()
    experiments: list[dict[str, Any]] = []
    seen_experiments: set[str] = set()
    for rank, row in enumerate(executable_rows, start=1):
        command = row.get("command")
        if not isinstance(command, list) or not command:
            raise ExperimentQueueError(f"materializer work row {rank} command must be non-empty")
        command_items = [str(item) for item in command]
        resource_kind = str(row.get("resource_kind") or "local_cpu")
        if not resource_kind.startswith("local"):
            raise ExperimentQueueError(
                f"materializer work row {rank} uses non-local resource {resource_kind!r}"
            )
        used_resource_kinds.add(resource_kind)
        experiment_id = _materializer_execution_experiment_id(
            row,
            rank,
            seen_experiments,
        )
        metadata = apply_proxy_evidence_boundary(
            {
                "schema": MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA,
                "source_work_queue_schema": work_queue.get("schema"),
                "source_work_queue_path": work_queue_ref,
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
                "steps": [
                    {
                        "id": MATERIALIZER_EXECUTION_STEP_ID,
                        "kind": "command",
                        "command": command_items,
                        "requires": [],
                        "resources": {"kind": resource_kind},
                        "postconditions": _normalize_materializer_queue_postconditions(
                            row.get("postconditions")
                        ),
                        "telemetry": dict(row.get("telemetry") or {}),
                        "timeout_seconds": step_timeout_seconds,
                    }
                ],
            }
        )

    for kind in sorted(used_resource_kinds):
        resource_limits.setdefault(
            kind,
            local_cpu_concurrency if kind == "local_cpu" else 1,
        )
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

    selected_operations = [
        item for item in _as_list(row.get("selected_operations")) if isinstance(item, Mapping)
    ]
    source_dispatch_blockers = [
        str(item) for item in _as_list(row.get("dispatch_blockers")) if str(item)
    ]
    blockers: list[str] = []
    if not selected_operations:
        blockers.append("selected_operations_missing")

    dropped_pairs: list[int] = []
    dqs1_operation_count = 0
    materializer_resolutions: list[dict[str, Any]] = []
    source_units: list[dict[str, Any]] = []
    for operation in selected_operations:
        unit_id = str(operation.get("unit_id") or "")
        unit = units_by_id.get(unit_id)
        operation_blockers = ordered_unique(
            str(item) for item in _as_list(operation.get("blockers"))
        )
        blockers.extend(
            f"selected_operation_blocker:{unit_id or '<missing>'}:{blocker}"
            for blocker in operation_blockers
        )
        resolution = resolve_materializer(operation=operation, unit=unit)
        blockers.extend(resolution.blockers)
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
                "blockers": list(resolution.blockers),
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
                    *[
                        str(item)
                        for item in _as_list(unit.get("materialization_blockers"))
                    ],
                    *[
                        str(item)
                        for item in _as_list(unit.get("candidate_trust_region_blockers"))
                    ],
                ]
            )
            blockers.extend(
                f"selected_unit_blocker:{unit_id}:{blocker}"
                for blocker in unit_blockers
            )
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
                    "candidate_trust_region_blockers": unit.get(
                        "candidate_trust_region_blockers"
                    ),
                    "blockers": unit_blockers,
                }
            )
        if resolution.target_kind != DQS1_PAIRSET_TARGET_KIND:
            continue
        dqs1_operation_count += 1
        pair_index = _pair_index_from_operation(operation)
        if pair_index is None:
            blockers.append(f"pair_index_missing:{operation.get('unit_id') or operation.get('operation_id')}")
            continue
        if not 0 <= pair_index < FEC6_PAIR_COUNT:
            blockers.append(f"pair_index_out_of_range:{pair_index}")
            continue
        dropped_pairs.append(pair_index)
    target_kinds = ordered_unique(
        resolution["target_kind"]
        for resolution in materializer_resolutions
        if resolution["target_kind"]
    )
    known_target_kinds = known_materializer_target_kinds()
    unsupported_targets = [
        target for target in target_kinds if target not in known_target_kinds
    ]
    if unsupported_targets:
        blockers.append("unsupported_materializer_target:" + ",".join(unsupported_targets))
    if dqs1_operation_count and base_pairs is None:
        blockers.append("dqs1_base_pair_indices_required")
    dropped_pairs = sorted(set(dropped_pairs))
    if dqs1_operation_count and len(dropped_pairs) != dqs1_operation_count:
        blockers.append("dropped_pair_indices_do_not_match_selected_operations")
    if base_pairs is not None and dqs1_operation_count:
        missing = [pair for pair in dropped_pairs if pair not in set(base_pairs)]
        if missing:
            blockers.append("dropped_pair_not_in_base:" + ",".join(str(pair) for pair in missing))
        selected_pairs = tuple(pair for pair in base_pairs if pair not in set(dropped_pairs))
        if not selected_pairs:
            blockers.append("selected_pair_indices_empty_after_drop")
    else:
        selected_pairs = ()

    conflict_violations = _as_list(row.get("conflict_violations"))
    if conflict_violations:
        blockers.append("conflict_violations_present")
    blockers = ordered_unique(blockers)
    executable = not blockers
    candidate_id = (
        _candidate_id(kind, selection_id, dropped_pairs)
        if dropped_pairs
        else f"pairset_byte_shave_{kind}_{rank:04d}"
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
            "combo_id": row.get("combo_id"),
            "sweep_id": row.get("sweep_id"),
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
    if candidate_limit is not None and (
        isinstance(candidate_limit, bool) or candidate_limit < 1
    ):
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
    plan_ref = (
        _repo_rel(Path(plan_path), repo)
        if plan_path is not None
        else None
    )
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
    materializer_backlog = build_materializer_backlog(compiled_rows)
    materializer_backlog_summary = summarize_materializer_backlog(materializer_backlog)
    materializer_work_queue = build_materializer_work_queue(
        materializer_backlog,
        repo_root=repo,
        contexts=materializer_contexts,
        source_plan_path=plan_ref,
    )
    partial_materialization_blockers: list[str] = []
    if blocked_rows and executable_rows and not allow_partial_materialization:
        partial_materialization_blockers.append(
            "partial_materialization_requires_explicit_allow"
        )
    queueable_rows = (
        executable_rows
        if allow_partial_materialization or not partial_materialization_blockers
        else []
    )

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
                bool(resolution.get("cooperative_receiver_required"))
                for resolution in row["materializer_resolutions"]
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
        dispatch_blockers=(
            partial_materialization_blockers
            or ["exact_auth_eval_required_before_score_claim"]
        ),
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
        dispatch_blockers=(
            partial_materialization_blockers
            or ["exact_auth_eval_required_before_score_claim"]
        ),
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
            "materializer_backlog_summary": materializer_backlog_summary,
            "materializer_work_queue": materializer_work_queue,
            "executable_rows": executable_rows,
            "blocked_rows": blocked_rows,
            "portfolio": portfolio,
            "action_summary": action_summary,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            partial_materialization_blockers
            or ["exact_auth_eval_required_before_score_claim"]
        ),
    )


__all__ = [
    "ACTION_SUMMARY_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "MATERIALIZER_BACKLOG_SCHEMA",
    "MATERIALIZER_CONTEXTS_SCHEMA",
    "MATERIALIZER_EXECUTION_EXPERIMENT_METADATA_SCHEMA",
    "MATERIALIZER_EXECUTION_STEP_ID",
    "MATERIALIZER_WORK_QUEUE_SCHEMA",
    "PORTFOLIO_SCHEMA",
    "build_materializer_backlog",
    "build_materializer_execution_queue",
    "build_materializer_work_queue",
    "compile_dqs1_byte_shaving_campaign",
    "materializer_contexts_from_payload",
    "summarize_materializer_backlog",
]
