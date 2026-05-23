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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY, PLAN_SCHEMA
from tac.optimization.decoder_q_selective_runtime_packet import FEC6_PAIR_COUNT
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)

from .byte_shaving_materializer_registry import (
    DQS1_PAIRSET_TARGET_KIND,
    REGISTRY_SCHEMA,
    registry_manifest,
    resolve_materializer,
)
from .dqs1_local_first_queue import SAFE_OPERATOR_ACTION, candidate_slug
from .experiment_queue import ExperimentQueueError

MATERIALIZATION_SCHEMA = "byte_shaving_campaign_materialization.v1"
ACTION_SUMMARY_SCHEMA = "cross_family_candidate_portfolio_action_summary.v1"
PORTFOLIO_SCHEMA = "byte_shaving_campaign_dqs1_operator_portfolio.v1"
TOOL_NAME = "comma_lab.scheduler.byte_shaving_campaign_queue"


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
    unsupported_targets = [
        target for target in target_kinds if target != DQS1_PAIRSET_TARGET_KIND
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
    if allow_partial_materialization and not str(
        partial_materialization_rationale or ""
    ).strip():
        raise ExperimentQueueError(
            "partial_materialization_rationale is required when partial materialization is allowed"
        )
    rationale = str(partial_materialization_rationale or "").strip()
    if allow_partial_materialization and not rationale:
        raise ExperimentQueueError(
            "allow_partial_materialization requires a non-empty "
            "partial_materialization_rationale"
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
    "PORTFOLIO_SCHEMA",
    "compile_dqs1_byte_shaving_campaign",
]
