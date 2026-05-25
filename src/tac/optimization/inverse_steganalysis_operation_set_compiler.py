# SPDX-License-Identifier: MIT
"""Planning-only compiler handoff for inverse-steganalysis operation sets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY, INVERSE_ACTION_COMPILER_TARGET_DEFAULTS
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.packet_compiler.deterministic_compiler import (
    PACKET_IR_OPERATION_SET_SCHEMA,
    packetir_operation_set_bridge_contract,
)

HANDOFF_SCHEMA = "inverse_steganalysis_operation_set_compiler_handoff.v1"
OPERATION_SET_COMPILER_HINT_SCHEMA = "inverse_action_operation_set_compiler_hint.v1"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _target_defaults(raw_operation: Mapping[str, Any], compiler: Mapping[str, Any]) -> Mapping[str, Any]:
    target_kind = str(
        raw_operation.get("target_kind") or compiler.get("target_kind") or ""
    ).strip()
    if target_kind in INVERSE_ACTION_COMPILER_TARGET_DEFAULTS:
        return INVERSE_ACTION_COMPILER_TARGET_DEFAULTS[target_kind]
    operation_family = str(
        raw_operation.get("operation_family")
        or compiler.get("operation_family")
        or ""
    ).strip()
    unit_kind = str(
        raw_operation.get("unit_kind") or compiler.get("unit_kind") or ""
    ).strip()
    for defaults in INVERSE_ACTION_COMPILER_TARGET_DEFAULTS.values():
        if operation_family and operation_family == defaults.get("operation_family"):
            return defaults
        if unit_kind and unit_kind == defaults.get("unit_kind"):
            return defaults
    return {}


def _operation_params(raw_operation: Mapping[str, Any], compiler: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for source in (compiler, raw_operation):
        value = source.get("params")
        if isinstance(value, Mapping):
            params.update(dict(value))
        for key, value in source.items():
            if key in {
                "schema",
                "selected_operations",
                "operation_set_id",
                "operation_family",
                "unit_kind",
                "target_kind",
                "materializer",
                "operation_id",
                "unit_id",
                "candidate_saved_bytes",
                "predicted_quality_score_delta",
                "blockers",
            }:
                continue
            if key in FALSE_AUTHORITY:
                continue
            if value is not None and key not in params:
                params[str(key)] = value
    return params


def _sequence_hash(sequence: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "unit_id": str(operation.get("unit_id") or ""),
            "operation_id": str(operation.get("operation_id") or ""),
            "operation_family": str(operation.get("operation_family") or ""),
            "unit_kind": str(operation.get("unit_kind") or ""),
            "materializer": operation.get("materializer"),
            "target_kind": operation.get("target_kind"),
        }
        for operation in sequence
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def packet_ir_operation_set_from_compiler_hint(
    compiler: Mapping[str, Any],
    *,
    source_backlog_key: str,
    source_unit_ids: Sequence[str] = (),
    candidate_id: str | None = None,
    lane_id: str | None = None,
    source_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Lower an inverse-action compiler hint into deterministic PacketIR."""

    require_no_truthy_authority_fields(compiler, context="operation_set_compiler")
    if compiler.get("schema") != OPERATION_SET_COMPILER_HINT_SCHEMA:
        raise ValueError(f"expected schema {OPERATION_SET_COMPILER_HINT_SCHEMA}")
    raw_operations = [
        item for item in _as_list(compiler.get("selected_operations")) if isinstance(item, Mapping)
    ]
    if not raw_operations:
        raise ValueError("operation_set_compiler selected_operations[] missing")

    operations: list[dict[str, Any]] = []
    for index, raw_operation in enumerate(raw_operations):
        defaults = _target_defaults(raw_operation, compiler)
        target_kind = str(
            raw_operation.get("target_kind")
            or compiler.get("target_kind")
            or defaults.get("target_kind")
            or ""
        ).strip()
        if target_kind in INVERSE_ACTION_COMPILER_TARGET_DEFAULTS:
            defaults = INVERSE_ACTION_COMPILER_TARGET_DEFAULTS[target_kind]
        unit_kind = str(
            raw_operation.get("unit_kind")
            or compiler.get("unit_kind")
            or defaults.get("unit_kind")
            or ""
        ).strip()
        operation_family = str(
            raw_operation.get("operation_family")
            or compiler.get("operation_family")
            or defaults.get("operation_family")
            or ""
        ).strip()
        materializer = str(
            raw_operation.get("materializer")
            or compiler.get("materializer")
            or defaults.get("materializer")
            or ""
        ).strip()
        unit_id = str(
            raw_operation.get("unit_id")
            or compiler.get("unit_id")
            or f"{compiler.get('operation_set_id') or 'compiled'}_op{index:04d}"
        )
        operation_id = str(
            raw_operation.get("operation_id")
            or f"{operation_family or 'compile_operation'}_{unit_id}"
        )
        operations.append(
            {
                "unit_id": unit_id,
                "unit_kind": unit_kind,
                "operation_id": operation_id,
                "operation_family": operation_family,
                "target_kind": target_kind or None,
                "materializer": materializer or None,
                "params": _operation_params(raw_operation, compiler),
                "candidate_saved_bytes": raw_operation.get("candidate_saved_bytes")
                or compiler.get("candidate_saved_bytes")
                or 0,
                "predicted_quality_score_delta": raw_operation.get(
                    "predicted_quality_score_delta"
                )
                or compiler.get("predicted_quality_score_delta"),
                "blockers": ordered_unique(
                    str(item)
                    for item in [
                        *_as_list(raw_operation.get("blockers")),
                        *_as_list(compiler.get("blockers")),
                    ]
                    if str(item)
                ),
                **FALSE_AUTHORITY,
            }
        )

    operation_set_id = str(
        compiler.get("operation_set_id")
        or f"compiled_high_level_{_sha256_text(source_backlog_key)[:12]}"
    )
    chosen_sequence = [
        {
            "unit_id": operation["unit_id"],
            "operation_id": operation["operation_id"],
            "operation_family": operation["operation_family"],
            "unit_kind": operation["unit_kind"],
            "materializer": operation["materializer"],
            "target_kind": operation["target_kind"],
        }
        for operation in operations
    ]
    return {
        "schema": PACKET_IR_OPERATION_SET_SCHEMA,
        "source_schema": HANDOFF_SCHEMA,
        "source_paths": list(source_paths),
        "source_backlog_key": source_backlog_key,
        "candidate_id": candidate_id or compiler.get("candidate_id"),
        "lane_id": lane_id or compiler.get("lane_id"),
        "source_operation_set_id": operation_set_id,
        "operation_set_id": f"packetir_{operation_set_id}",
        "compiler_contract": packetir_operation_set_bridge_contract(),
        "selected_unit_ids": ordered_unique(str(item) for item in source_unit_ids),
        "selected_operations": operations,
        "chosen_operation_sequence": chosen_sequence,
        "chosen_operation_sequence_sha256": _sequence_hash(chosen_sequence),
        "chosen_operation_sequence_source": "inverse_action_operation_set_compiler_hint_order",
        "chosen_operation_sequence_is_permutation": True,
        "requires_atomic_materialization": True,
        "partial_materialization_allowed": False,
        "operation_count": len(operations),
        "byte_closed_operation_count": len(operations),
        "operations": operations,
        "blockers": [
            "packetir_operation_set_requires_materializer_contexts",
            "packetir_operation_set_requires_runtime_consumption_proof",
            "packetir_operation_set_requires_exact_readiness_handoff",
        ],
        **FALSE_AUTHORITY,
    }


__all__ = [
    "HANDOFF_SCHEMA",
    "OPERATION_SET_COMPILER_HINT_SCHEMA",
    "packet_ir_operation_set_from_compiler_hint",
]
