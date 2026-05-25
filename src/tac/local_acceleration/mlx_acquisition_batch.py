# SPDX-License-Identifier: MIT
"""Group local MLX scorer/training evidence into acquisition operation sets."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from tac.local_acceleration import (
    EVIDENCE_GRADE_MLX,
    EVIDENCE_TAG_MLX,
    MLX_ACQUISITION_BATCH_OPERATION_SET_SCHEMA,
    MLX_ACQUISITION_BATCH_SCHEMA,
    MLX_ACQUISITION_REPRESENTATION_CONTRACT_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)

SCHEMA_VERSION = MLX_ACQUISITION_BATCH_SCHEMA
OPERATION_SET_SCHEMA = MLX_ACQUISITION_BATCH_OPERATION_SET_SCHEMA
OPERATION_SET_INTERACTION_SCHEMA = "mlx_acquisition_operation_set_interaction.v1"
MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA = (
    "mlx_effective_spend_triage_candidate_selection.v1"
)
MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA = (
    "mlx_effective_spend_triage_candidate_row.v1"
)
TOOL_NAME = "tac.local_acceleration.mlx_acquisition_batch"
REPRESENTATION_CONTRACT_SCHEMA = MLX_ACQUISITION_REPRESENTATION_CONTRACT_SCHEMA
OPERATION_SET_COMPILER_HINT_SCHEMA = "inverse_action_operation_set_compiler_hint.v1"
COMPILER_HINT_KEYS = (
    "operation_set_compiler",
    "operation_set_compiler_hint",
    "compiler_hint",
)
COMPILER_TARGET_KEYS = (
    "target_kind",
    "archive_section",
    "section_name",
    "target_section",
    "target_sections",
    "packet_member",
    "member_name",
    "tensor_name",
    "tensor_path",
    "byte_range",
    "frame_range",
    "pair_indices",
    "region_bbox",
)
COMPILER_OPERATION_METADATA_KEYS = (
    "unit_id",
    "unit_kind",
    "operation_id",
    "operation_family",
    "target_kind",
    "materializer",
    "candidate_saved_bytes",
    "predicted_quality_score_delta",
    "receiver_contract_kind",
    "representation_family",
    "representation_family_class",
    "representation_contract",
    "bolt_on_families",
    "params",
    "blockers",
)
REPRESENTATION_FAMILY_CLASSES = frozenset(
    {
        "hnerv_variant",
        "boostnerv_bolton",
        "nerv_family",
        "non_nerv",
        "unknown",
    }
)
REPRESENTATION_CLASS_ALIASES = {
    "hnerv": "hnerv_variant",
    "hnerv_variant": "hnerv_variant",
    "hnerv_variants": "hnerv_variant",
    "boostnerv": "boostnerv_bolton",
    "boost_nerv": "boostnerv_bolton",
    "boostnerv_bolton": "boostnerv_bolton",
    "boostnerv_boltons": "boostnerv_bolton",
    "nerv": "nerv_family",
    "nerv_family": "nerv_family",
    "nerv_family_variant": "nerv_family",
    "non_nerv": "non_nerv",
    "non_nerv_family": "non_nerv",
    "generic": "non_nerv",
    "unknown": "unknown",
}
HNERV_TOKENS = frozenset(
    {
        "hnerv",
        "hnerv_ft_microcodec",
        "hnerv_lc",
        "hnerv_lc_ac",
        "hnerv_lc_v2",
        "ff_packed_brotli_hnerv",
    }
)
BOOSTNERV_TOKENS = frozenset({"boostnerv", "boost_nerv", "boost-nerv"})
NON_NERV_TOKENS = frozenset(
    {
        "non_nerv",
        "non-nerv",
        "non_nerv_family",
        "non-nerv-family",
        "nonneural",
        "generic",
        "decoder_q",
        "packetir",
        "packet_ir",
        "wavelet",
        "cool_chic",
        "c3",
        "siren",
        "wire",
        "bacon",
        "compressai",
        "dcvc",
        "raft",
        "stc",
        "symbolic",
        "pose_codec",
        "lfv1",
    }
)


class MLXAcquisitionBatchError(ValueError):
    """Raised when local MLX acquisition batch input is malformed."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _false_authority(row: Mapping[str, Any], *blockers: str) -> dict[str, Any]:
    return apply_proxy_evidence_boundary(
        {
            **dict(row),
            **PROXY_FALSE_AUTHORITY_FIELDS,
            "candidate_generation_only": True,
            "planning_only": True,
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "score_axis": EVIDENCE_TAG_MLX,
        },
        dispatch_blockers=blockers,
    )


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MLXAcquisitionBatchError(f"{label} must be an object")
    return dict(value)


def _optional_mapping(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _as_mapping(value, label=label)


def _list_rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise MLXAcquisitionBatchError(f"{label} must be a non-empty list")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        out.append(_as_mapping(item, label=f"{label}[{index}]"))
    return out


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MLXAcquisitionBatchError(f"{label} is required")
    return text


def _float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise MLXAcquisitionBatchError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MLXAcquisitionBatchError(f"{label} must be numeric") from exc
    if minimum is not None and parsed < minimum:
        raise MLXAcquisitionBatchError(f"{label} must be >= {minimum}")
    return parsed


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise MLXAcquisitionBatchError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MLXAcquisitionBatchError(f"{label} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise MLXAcquisitionBatchError(f"{label} must be >= {minimum}")
    return parsed


def _int_list(value: Any, *, label: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise MLXAcquisitionBatchError(f"{label} must be a list")
    return [_int(item, f"{label}[{index}]", minimum=0) for index, item in enumerate(value)]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "row"


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise MLXAcquisitionBatchError(f"{label} must be a string or list of strings")
    out: list[str] = []
    for index, item in enumerate(value):
        text = str(item or "").strip()
        if not text:
            raise MLXAcquisitionBatchError(f"{label}[{index}] must be non-empty")
        out.append(text)
    return out


def _operation_mappings(value: Any, *, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    return _list_rows(value, label=label)


def _row_compiler_hint(row: Mapping[str, Any], *, label: str) -> dict[str, Any] | None:
    for key in COMPILER_HINT_KEYS:
        hint = _optional_mapping(row.get(key), label=f"{label}.{key}")
        if hint is not None:
            return hint
    row_operations = _operation_mappings(
        row.get("selected_operations"),
        label=f"{label}.selected_operations",
    )
    if row_operations:
        return {
            "schema": OPERATION_SET_COMPILER_HINT_SCHEMA,
            "selected_operations": row_operations,
        }
    if row.get("target_kind") is not None:
        return {
            "schema": OPERATION_SET_COMPILER_HINT_SCHEMA,
            **{
                key: row.get(key)
                for key in (*COMPILER_OPERATION_METADATA_KEYS, *COMPILER_TARGET_KEYS)
                if row.get(key) is not None
            },
        }
    return None


def _compiler_raw_operations(
    row: Mapping[str, Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    hint = _row_compiler_hint(row, label=label)
    if hint is None:
        return []
    operations = _operation_mappings(
        hint.get("selected_operations"),
        label=f"{label}.operation_set_compiler.selected_operations",
    )
    if operations:
        return operations
    return [hint]


def _compiler_operations_from_selection_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    row_id: str,
    candidate_id: str,
    family: str,
    representation_contract: Mapping[str, Any],
    added_bytes: int,
    row_gain_value: float,
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    raw_operations = _compiler_raw_operations(
        row,
        label=f"selected_rows[{row_index}]",
    )
    operation_count = len(raw_operations)
    saved_bytes = max(0, -added_bytes)
    for op_index, raw_operation in enumerate(raw_operations):
        operation_label = (
            f"selected_rows[{row_index}].operation_set_compiler."
            f"selected_operations[{op_index}]"
        )
        require_no_truthy_authority_fields(
            raw_operation,
            context=operation_label,
        )
        target_kind = _optional_string(raw_operation.get("target_kind"))
        if target_kind is None:
            raise MLXAcquisitionBatchError(f"{operation_label}.target_kind is required")
        params = dict(
            _optional_mapping(raw_operation.get("params"), label=f"{operation_label}.params")
            or {}
        )
        for key in COMPILER_TARGET_KEYS:
            value = raw_operation.get(key, row.get(key))
            if value is not None and key not in params:
                params[key] = value
        operation_saved = raw_operation.get("candidate_saved_bytes")
        if operation_saved is None:
            operation_saved = saved_bytes
        operation = {
            "unit_id": raw_operation.get("unit_id")
            or f"mlx_compiler_{_slug(row_id)}_{op_index:04d}",
            "operation_id": raw_operation.get("operation_id")
            or f"mlx_compile_{_slug(row_id)}_{op_index:04d}",
            "target_kind": target_kind,
            "candidate_id": candidate_id,
            "source_row_id": row_id,
            "source_family": family,
            "representation_family": representation_contract["representation_family"],
            "representation_family_class": representation_contract[
                "representation_family_class"
            ],
            "representation_contract": dict(representation_contract),
            "bolt_on_families": list(representation_contract["bolt_on_families"]),
            "receiver_contract_kind": raw_operation.get("receiver_contract_kind")
            or representation_contract["receiver_contract_kind"],
            "materializer_contract_kind": representation_contract[
                "materializer_contract_kind"
            ],
            "operation_portability": "family_agnostic",
            "candidate_saved_bytes": operation_saved,
            "predicted_quality_score_delta": raw_operation.get(
                "predicted_quality_score_delta",
                -row_gain_value / float(max(1, operation_count)),
            ),
            "params": {
                **params,
                "source_row_id": row_id,
                "candidate_id": candidate_id,
                "source_path": row.get("source_path"),
                "archive_sha256": row.get("archive_sha256"),
                "raw_sha256": row.get("raw_sha256"),
            },
            "blockers": ordered_unique(
                [
                    *[
                        str(item)
                        for item in _string_list(
                            raw_operation.get("blockers"),
                            label=f"{operation_label}.blockers",
                        )
                    ],
                    "compiled_from_mlx_acquisition_operation_set_compiler",
                    "requires_materializer_contexts",
                    "requires_runtime_consumption_proof_before_exact_eval",
                    "requires_exact_auth_eval_before_score_claim",
                ]
            ),
        }
        for key in (
            "unit_kind",
            "operation_family",
            "materializer",
        ):
            if raw_operation.get(key) is not None:
                operation[key] = raw_operation[key]
        operations.append(operation)
    return operations


def _operation_unit_id(operation: Mapping[str, Any]) -> str:
    unit_id = _optional_string(operation.get("unit_id"))
    if unit_id:
        return unit_id
    operation_id = _optional_string(operation.get("operation_id")) or "operation"
    return f"unit_{_slug(operation_id)}"


def _operation_family(operation: Mapping[str, Any]) -> str:
    return _optional_string(operation.get("operation_family")) or "unknown_operation"


def _operation_params_mapping(operation: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_optional_mapping(operation.get("params"), label="operation.params") or {})


def _param_text(operation: Mapping[str, Any], *keys: str) -> str | None:
    params = _operation_params_mapping(operation)
    for key in keys:
        value = operation.get(key)
        if value is None:
            value = params.get(key)
        if value is not None:
            text = _optional_string(value)
            if text:
                return text
    return None


def _dynamic_sparse_gate_param(
    operation: Mapping[str, Any],
    key: str,
) -> str | None:
    nested = _operation_params_mapping(operation).get("dynamic_sparse_channel_gate")
    if not isinstance(nested, Mapping):
        return None
    return _optional_string(nested.get(key))


def _operation_pair_indices(operation: Mapping[str, Any]) -> list[int]:
    params = _operation_params_mapping(operation)
    return ordered_unique(
        str(pair)
        for pair in [
            *_int_list(operation.get("pair_indices"), label="operation.pair_indices"),
            *_int_list(params.get("pair_indices"), label="operation.params.pair_indices"),
        ]
    )


def _interaction_row(
    *,
    interaction_kind: str,
    key: str,
    operations: Sequence[Mapping[str, Any]],
    rationale: str,
) -> dict[str, Any]:
    unit_ids = ordered_unique(_operation_unit_id(operation) for operation in operations)
    operation_ids = ordered_unique(
        _optional_string(operation.get("operation_id")) or _operation_unit_id(operation)
        for operation in operations
    )
    return {
        "schema": OPERATION_SET_INTERACTION_SCHEMA,
        "interaction_id": f"mlx_interaction:{interaction_kind}:{_slug(key)}",
        "interaction_kind": interaction_kind,
        "interaction_key": key,
        "unit_ids": unit_ids,
        "operation_ids": operation_ids,
        "operation_families": ordered_unique(_operation_family(operation) for operation in operations),
        "operation_count": len(unit_ids),
        "delta_score": 0.0,
        "quality_cost_delta_score": 0.0,
        "extra_saved_bytes": 0,
        "shared_overhead_bytes": 0,
        "interaction_model": "structural_overlap_unmeasured_zero_delta_prior",
        "measurement_state": "requires_queue_observation_or_exact_calibration",
        "rationale": rationale,
        "dispatch_blockers": [
            "mlx_acquisition_interaction_is_planning_only",
            "requires_empirical_interaction_calibration_before_score_claim",
            "requires_byte_closed_materialization_before_dispatch",
        ],
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _append_group_interactions(
    rows: list[dict[str, Any]],
    *,
    operations: Sequence[Mapping[str, Any]],
    interaction_kind: str,
    value_label: str,
    rationale: str,
) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for operation in operations:
        value = _param_text(operation, value_label)
        if value:
            groups.setdefault(value, []).append(operation)
    for key, grouped in sorted(groups.items()):
        if len(grouped) >= 2:
            rows.append(
                _interaction_row(
                    interaction_kind=interaction_kind,
                    key=key,
                    operations=grouped,
                    rationale=rationale,
                )
            )


def _active_structural_interactions(
    *,
    selected_operations: Sequence[Mapping[str, Any]],
    compiler_operations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic unmeasured interaction priors for grouped search."""

    rows: list[dict[str, Any]] = []
    operations = [*selected_operations, *compiler_operations]
    if len(operations) < 2:
        return rows

    grouped_values: tuple[tuple[str, str, str], ...] = (
        (
            "shared_target_kind",
            "target_kind",
            "operations target the same materializer family and may share setup or conflict",
        ),
        (
            "shared_operation_family",
            "operation_family",
            "operations share a transformation family and need grouped calibration",
        ),
        (
            "shared_packet_member",
            "member_name",
            "operations touch the same packet member and may have non-additive byte effects",
        ),
        (
            "shared_tensor",
            "tensor_name",
            "operations touch the same tensor and may have non-additive distortion effects",
        ),
        (
            "shared_archive_section",
            "section_name",
            "operations touch the same archive section and may share coding overhead",
        ),
    )
    for interaction_kind, value_label, rationale in grouped_values:
        _append_group_interactions(
            rows,
            operations=operations,
            interaction_kind=interaction_kind,
            value_label=value_label,
            rationale=rationale,
        )

    pair_groups: dict[str, list[Mapping[str, Any]]] = {}
    for operation in operations:
        for pair in _operation_pair_indices(operation):
            pair_groups.setdefault(pair, []).append(operation)
    for pair, grouped in sorted(pair_groups.items(), key=lambda item: int(item[0])):
        if len(grouped) >= 2:
            rows.append(
                _interaction_row(
                    interaction_kind="shared_pair_index",
                    key=pair,
                    operations=grouped,
                    rationale="operations share a contest pair index and may interact through batching or pose geometry",
                )
            )

    for nested_key, interaction_kind, rationale in (
        (
            "source_id",
            "dynamic_sparse_same_source",
            "dynamic sparse gate operations share a source axis and may compete for the same latent budget",
        ),
        (
            "channel_id",
            "dynamic_sparse_same_channel",
            "dynamic sparse gate operations share a channel axis and may expose source/channel synergy",
        ),
    ):
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for operation in compiler_operations:
            value = _dynamic_sparse_gate_param(operation, nested_key)
            if value:
                groups.setdefault(value, []).append(operation)
        for key, grouped in sorted(groups.items()):
            if len(grouped) >= 2:
                rows.append(
                    _interaction_row(
                        interaction_kind=interaction_kind,
                        key=key,
                        operations=grouped,
                        rationale=rationale,
                    )
                )

    return sorted(
        rows,
        key=lambda row: (
            str(row["interaction_kind"]),
            str(row["interaction_key"]),
            str(row["interaction_id"]),
        ),
    )


def _family_tokens(row: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in (
        "representation_family",
        "substrate_family",
        "architecture_family",
        "architecture_class",
        "codec_family",
        "family",
        "candidate_family",
        "materializer_family",
    ):
        text = _optional_string(row.get(key))
        if text:
            tokens.append(text)
    for key in ("bolt_on_families", "addon_families", "family_tags"):
        tokens.extend(_string_list(row.get(key), label=key))
    return ordered_unique(tokens)


def _normalize_family_class(value: Any) -> str | None:
    text = _optional_string(value)
    if not text:
        return None
    key = _slug(text)
    family_class = REPRESENTATION_CLASS_ALIASES.get(key)
    if family_class is None:
        raise MLXAcquisitionBatchError(
            "representation_family_class must be one of "
            f"{sorted(REPRESENTATION_FAMILY_CLASSES)}"
        )
    return family_class


def _infer_family_class(tokens: Sequence[str]) -> str:
    normalized = {_slug(token) for token in tokens}
    joined = " ".join(normalized)
    if normalized & {_slug(token) for token in BOOSTNERV_TOKENS}:
        return "boostnerv_bolton"
    if normalized & {_slug(token) for token in HNERV_TOKENS}:
        return "hnerv_variant"
    if normalized & {_slug(token) for token in NON_NERV_TOKENS}:
        return "non_nerv"
    if any(_slug(token) in joined for token in NON_NERV_TOKENS):
        return "non_nerv"
    if "boostnerv" in joined or "boost_nerv" in joined:
        return "boostnerv_bolton"
    if "hnerv" in joined:
        return "hnerv_variant"
    if "nerv" in joined:
        return "nerv_family"
    return "unknown"


def _representation_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    source_family = _optional_string(row.get("family")) or "unknown"
    tokens = _family_tokens(row)
    family_class = _normalize_family_class(
        row.get("representation_family_class") or row.get("family_class")
    ) or _infer_family_class(tokens or [source_family])
    representation_family = (
        _optional_string(row.get("representation_family"))
        or _optional_string(row.get("architecture_family"))
        or source_family
    )
    bolt_ons = _string_list(row.get("bolt_on_families"), label="bolt_on_families")
    bolt_ons.extend(_string_list(row.get("addon_families"), label="addon_families"))
    if family_class == "boostnerv_bolton" and "boostnerv" not in {
        _slug(item) for item in bolt_ons
    }:
        bolt_ons.append("boostnerv")
    return {
        "schema": REPRESENTATION_CONTRACT_SCHEMA,
        "source_family": source_family,
        "representation_family": representation_family,
        "representation_family_class": family_class,
        "representation_variant": _optional_string(
            row.get("representation_variant")
            or row.get("architecture_variant")
            or row.get("codec_variant")
        ),
        "substrate_family": _optional_string(row.get("substrate_family")),
        "family_tokens": sorted({_slug(token) for token in tokens if _slug(token)}),
        "bolt_on_families": ordered_unique(bolt_ons),
        "receiver_contract_kind": (
            _optional_string(row.get("receiver_contract_kind"))
            or f"family_agnostic_{family_class}_mlx_candidate_receiver"
        ),
        "materializer_contract_kind": (
            _optional_string(row.get("materializer_contract_kind"))
            or f"family_agnostic_{family_class}_candidate_materializer"
        ),
        "runtime_contract_kind": (
            _optional_string(row.get("runtime_contract_kind"))
            or "runtime_consumption_proof_required"
        ),
        "operation_portability": "family_agnostic",
    }


def _selection_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    require_no_truthy_authority_fields(selection, context="mlx_acquisition_selection")
    if selection.get("schema") != MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA:
        raise MLXAcquisitionBatchError(
            "selection schema must be "
            f"{MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA}"
        )
    if selection.get("candidate_generation_only") is not True:
        raise MLXAcquisitionBatchError("selection candidate_generation_only must be true")
    if selection.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise MLXAcquisitionBatchError(
            f"selection evidence_grade must be {EVIDENCE_GRADE_MLX}"
        )
    if selection.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise MLXAcquisitionBatchError(
            f"selection evidence_tag must be {EVIDENCE_TAG_MLX}"
        )
    rows = _list_rows(selection.get("selected_rows"), label="selection.selected_rows")
    for index, row in enumerate(rows):
        require_no_truthy_authority_fields(
            row,
            context=f"mlx_acquisition_selection.selected_rows[{index}]",
        )
        if row.get("schema") != MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA:
            raise MLXAcquisitionBatchError(f"selection row {index} schema mismatch")
        if row.get("candidate_generation_only") is not True:
            raise MLXAcquisitionBatchError(
                f"selection row {index} candidate_generation_only must be true"
            )
    return rows


def _row_gain(row: Mapping[str, Any], *, index: int) -> float:
    gain = _float(
        row.get("normalized_full_video_scorer_gain_vs_baseline"),
        f"selected_rows[{index}].normalized_full_video_scorer_gain_vs_baseline",
        minimum=0.0,
    )
    if gain <= 0.0:
        raise MLXAcquisitionBatchError(
            f"selected_rows[{index}] normalized_full_video_scorer_gain_vs_baseline must be positive"
        )
    return gain


def _operation_set_from_selection_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    set_index: int,
    source_path: str | None,
) -> dict[str, Any]:
    if not rows:
        raise MLXAcquisitionBatchError("operation set rows must not be empty")
    selected_operations: list[dict[str, Any]] = []
    selected_unit_ids: list[str] = []
    pair_indices: list[int] = []
    expected_score_gain = 0.0
    candidate_saved_bytes = 0
    families: list[str] = []
    family_classes: list[str] = []
    receiver_contract_kinds: list[str] = []
    materializer_contract_kinds: list[str] = []
    representation_contracts: list[dict[str, Any]] = []
    row_refs: list[dict[str, Any]] = []
    compiler_operations: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        row_index = set_index + offset
        row_id = _text(row.get("row_id") or row.get("candidate_id"), "selection row_id")
        candidate_id = _text(row.get("candidate_id"), "selection candidate_id")
        family = str(row.get("family") or "mlx_scorer_response")
        families.append(family)
        representation_contract = _representation_contract(row)
        representation_contracts.append(representation_contract)
        family_classes.append(str(representation_contract["representation_family_class"]))
        receiver_contract_kinds.append(str(representation_contract["receiver_contract_kind"]))
        materializer_contract_kinds.append(
            str(representation_contract["materializer_contract_kind"])
        )
        unit_id = f"mlx_row_{_slug(row_id)}"
        selected_unit_ids.append(unit_id)
        row_pairs = _int_list(
            row.get("pair_indices") or row.get("source_pair_window"),
            label=f"selected_rows[{row_index}].pair_indices",
        )
        pair_indices.extend(row_pairs)
        row_gain_value = _row_gain(row, index=row_index)
        expected_score_gain += row_gain_value
        added_bytes = int(row.get("added_archive_bytes") or 0)
        if added_bytes < 0:
            candidate_saved_bytes += abs(added_bytes)
        compiler_operations.extend(
            _compiler_operations_from_selection_row(
                row,
                row_index=row_index,
                row_id=row_id,
                candidate_id=candidate_id,
                family=family,
                representation_contract=representation_contract,
                added_bytes=added_bytes,
                row_gain_value=row_gain_value,
            )
        )
        selected_operations.append(
            {
                "operation_id": f"materialize_mlx_response_{_slug(row_id)}",
                "operation_family": "materialize_scorer_response_candidate",
                "unit_id": unit_id,
                "unit_kind": "scorer_response_row",
                "target_kind": "mlx_scorer_response_candidate_v1",
                "candidate_id": candidate_id,
                "source_family": family,
                "representation_family": representation_contract[
                    "representation_family"
                ],
                "representation_family_class": representation_contract[
                    "representation_family_class"
                ],
                "representation_contract": representation_contract,
                "bolt_on_families": representation_contract["bolt_on_families"],
                "receiver_contract_kind": representation_contract[
                    "receiver_contract_kind"
                ],
                "materializer_contract_kind": representation_contract[
                    "materializer_contract_kind"
                ],
                "candidate_saved_bytes": max(0, -added_bytes),
                "predicted_quality_score_delta": -row_gain_value,
                "pair_indices": row_pairs,
                "params": {
                    "source_row_id": row_id,
                    "source_path": row.get("source_path"),
                    "archive_sha256": row.get("archive_sha256"),
                    "raw_sha256": row.get("raw_sha256"),
                },
                "blockers": [
                    "mlx_acquisition_operation_requires_candidate_materializer",
                    "requires_exact_auth_eval_before_score_claim",
                ],
                **PROXY_FALSE_AUTHORITY_FIELDS,
            }
        )
        row_refs.append(
            {
                "row_id": row_id,
                "candidate_id": candidate_id,
                "family": family,
                "representation_family": representation_contract[
                    "representation_family"
                ],
                "representation_family_class": representation_contract[
                    "representation_family_class"
                ],
                "receiver_contract_kind": representation_contract[
                    "receiver_contract_kind"
                ],
                "source_path": row.get("source_path"),
            }
        )
    first = rows[0]
    source_candidate_id = _text(first.get("candidate_id"), "selection candidate_id")
    operation_set_id = f"mlx_opset_{set_index:04d}_{_slug(source_candidate_id)}"
    active_interactions = _active_structural_interactions(
        selected_operations=selected_operations,
        compiler_operations=compiler_operations,
    )
    operation_set_compiler = None
    if compiler_operations:
        operation_set_compiler = {
            "schema": OPERATION_SET_COMPILER_HINT_SCHEMA,
            "operation_set_id": f"{operation_set_id}_compiled",
            "candidate_id": source_candidate_id,
            "candidate_saved_bytes": sum(
                max(0, int(operation.get("candidate_saved_bytes") or 0))
                for operation in compiler_operations
            ),
            "operation_portability": "family_agnostic",
            "source_kind": "mlx_acquisition_batch_selection_row_compiler_hints",
            "selected_operations": compiler_operations,
            **PROXY_FALSE_AUTHORITY_FIELDS,
        }
        require_no_truthy_authority_fields(
            operation_set_compiler,
            context=f"mlx_acquisition_batch.operation_sets[{set_index}].operation_set_compiler",
        )
    return _false_authority(
        {
            "schema": OPERATION_SET_SCHEMA,
            "operation_set_id": operation_set_id,
            "candidate_id": source_candidate_id,
            "operation_set_rank": set_index + 1,
            "resource_kind": "local_mlx",
            "component": "scorer",
            "source_path": source_path,
            "source_schema": MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA,
            "operation_families": ["materialize_scorer_response_candidate"],
            "source_families": sorted(set(families)),
            "source_family_classes": sorted(set(family_classes)),
            "representation_contracts": representation_contracts,
            "receiver_contract_kinds": sorted(set(receiver_contract_kinds)),
            "materializer_contract_kinds": sorted(set(materializer_contract_kinds)),
            "operation_portability": "family_agnostic",
            **(
                {"operation_set_compiler": operation_set_compiler}
                if operation_set_compiler is not None
                else {}
            ),
            "selected_unit_ids": selected_unit_ids,
            "selected_operations": selected_operations,
            "chosen_operation_sequence": [dict(item) for item in selected_operations],
            "chosen_operation_sequence_source": "mlx_acquisition_batch_order",
            "active_interactions": active_interactions,
            "interaction_delta_score": 0.0,
            "interaction_extra_saved_bytes": 0,
            "interaction_shared_overhead_bytes": 0,
            "interaction_model": "structural_overlap_unmeasured_zero_delta_prior",
            "pair_indices": sorted(set(pair_indices)),
            "candidate_saved_bytes": candidate_saved_bytes,
            "expected_score_gain": expected_score_gain,
            "expected_delta_score": -expected_score_gain,
            "quality_cost_score": -expected_score_gain,
            "row_refs": row_refs,
        },
        "mlx_acquisition_batch_is_planning_only",
        "requires_byte_closed_materialization_before_dispatch",
        "requires_exact_auth_eval_before_score_claim",
    )


def build_mlx_acquisition_batch_from_selection(
    selection: Mapping[str, Any],
    *,
    source_path: str | None = None,
    set_size: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return grouped MLX acquisition operation sets from strict selection rows."""

    if isinstance(set_size, bool) or set_size < 1:
        raise MLXAcquisitionBatchError("set_size must be >= 1")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise MLXAcquisitionBatchError("limit must be >= 1 when provided")
    rows = _selection_rows(selection)
    selected = rows[:limit] if limit is not None else rows
    operation_sets = [
        _operation_set_from_selection_rows(
            selected[index : index + set_size],
            set_index=index // set_size,
            source_path=source_path,
        )
        for index in range(0, len(selected), set_size)
    ]
    return _false_authority(
        {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "source_schema": selection.get("schema"),
            "source_path": source_path,
            "set_size": set_size,
            "source_row_count": len(rows),
            "operation_set_count": len(operation_sets),
            "operation_sets": operation_sets,
            "summary": {
                "operation_set_count": len(operation_sets),
                "selected_operation_count": sum(
                    len(item["selected_operations"]) for item in operation_sets
                ),
                "expected_score_gain_sum": sum(
                    float(item["expected_score_gain"]) for item in operation_sets
                ),
                "candidate_saved_bytes_sum": sum(
                    int(item["candidate_saved_bytes"]) for item in operation_sets
                ),
                "resource_kinds": sorted(
                    {str(item.get("resource_kind") or "") for item in operation_sets}
                ),
                "source_families": sorted(
                    {
                        family
                        for item in operation_sets
                        for family in item.get("source_families", [])
                    }
                ),
                "source_family_classes": sorted(
                    {
                        family_class
                        for item in operation_sets
                        for family_class in item.get("source_family_classes", [])
                    }
                ),
                "operation_portability": "family_agnostic",
                "operation_set_interaction_count_sum": sum(
                    len(item.get("active_interactions") or []) for item in operation_sets
                ),
            },
        },
        "mlx_acquisition_batch_is_planning_only",
        "requires_byte_closed_materialization_before_dispatch",
        "requires_exact_auth_eval_before_score_claim",
    )


def validate_mlx_acquisition_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a MLX acquisition batch."""

    require_no_truthy_authority_fields(batch, context="mlx_acquisition_batch")
    if batch.get("schema") != SCHEMA_VERSION:
        raise MLXAcquisitionBatchError(f"batch schema must be {SCHEMA_VERSION}")
    if batch.get("candidate_generation_only") is not True:
        raise MLXAcquisitionBatchError("batch candidate_generation_only must be true")
    if batch.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise MLXAcquisitionBatchError(f"batch evidence_grade must be {EVIDENCE_GRADE_MLX}")
    if batch.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise MLXAcquisitionBatchError(f"batch evidence_tag must be {EVIDENCE_TAG_MLX}")
    operation_sets = _list_rows(batch.get("operation_sets"), label="batch.operation_sets")
    normalized_sets: list[dict[str, Any]] = []
    for index, operation_set in enumerate(operation_sets):
        require_no_truthy_authority_fields(
            operation_set,
            context=f"mlx_acquisition_batch.operation_sets[{index}]",
        )
        if operation_set.get("schema") != OPERATION_SET_SCHEMA:
            raise MLXAcquisitionBatchError(f"operation set {index} schema mismatch")
        if operation_set.get("candidate_generation_only") is not True:
            raise MLXAcquisitionBatchError(
                f"operation set {index} candidate_generation_only must be true"
            )
        if not _list_rows(
            operation_set.get("selected_operations"),
            label=f"operation_sets[{index}].selected_operations",
        ):
            raise MLXAcquisitionBatchError(
                f"operation_sets[{index}].selected_operations must not be empty"
            )
        compiler = operation_set.get("operation_set_compiler")
        if compiler is not None:
            _as_mapping(
                compiler,
                label=f"operation_sets[{index}].operation_set_compiler",
            )
        normalized_sets.append(dict(operation_set))
    return {**dict(batch), "operation_sets": normalized_sets}


__all__ = [
    "OPERATION_SET_INTERACTION_SCHEMA",
    "OPERATION_SET_SCHEMA",
    "REPRESENTATION_CONTRACT_SCHEMA",
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "MLXAcquisitionBatchError",
    "build_mlx_acquisition_batch_from_selection",
    "validate_mlx_acquisition_batch",
]
