# SPDX-License-Identifier: MIT
"""Shared materializer feedback extraction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    require_no_truthy_authority_fields,
)
from tac.optimization.serialized_archive_economics import SERIALIZED_ARCHIVE_DELTA_SCHEMA
from tac.score_composition import CANONICAL_RATE_DENOM_BYTES, CANONICAL_RATE_MULTIPLIER

FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA = (
    "family_agnostic_materializer_empirical_observation.v1"
)
FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA = (
    "family_agnostic_materializer_empirical_sweep.v1"
)
DEFAULT_LOCAL_MATERIALIZER_AXIS = "[local-materializer advisory]"
DEFAULT_MATERIALIZER_RATE_SCORE_PER_BYTE = CANONICAL_RATE_MULTIPLIER / float(
    CANONICAL_RATE_DENOM_BYTES
)
MATERIALIZER_FALSE_AUTHORITY: dict[str, bool] = {
    **PROXY_FALSE_AUTHORITY_FIELDS,
    "dispatch_attempted": False,
    "gpu_launched": False,
}
MATERIALIZER_DELTA_KEYS: tuple[str, ...] = (
    "section_recode",
    "selected_compression",
    "selected_merge",
    "selected_payload",
    "selected_elision",
    "factorization",
)
SERIALIZED_ARCHIVE_DELTA_KEY = "serialized_archive_delta"


def _slug(value: Any, *, fallback: str = "materializer_feedback") -> str:
    text = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip()
    ).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or fallback


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _string_sequence(value: Any) -> list[str]:
    return [str(item) for item in _as_sequence(value) if str(item)]


def _recommended_planner_action(
    *,
    target_kind: str,
    rate_positive: bool,
    receiver_contract_satisfied: bool,
) -> str:
    if rate_positive and receiver_contract_satisfied:
        return "keep_rate_positive_candidate_for_inflate_parity_gate"
    if rate_positive:
        return "repair_receiver_contract_before_exact_readiness"
    suffix_by_target = {
        "archive_section_entropy_recode_v1": "archive_section_entropy_recode",
        "packet_member_recompress_v1": "member_recompress",
        "packet_member_zip_header_elide_v1": "header_elide",
        "tensor_factorize_v1": "tensor_factorize",
    }
    return (
        "demote_matching_archive_class_for_"
        f"{suffix_by_target.get(target_kind, target_kind or 'materializer')}"
    )


def optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def selected_materializer_delta(
    payload: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    for key in MATERIALIZER_DELTA_KEYS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return key, value
    return "", {}


def _serialized_archive_delta(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    delta = payload.get(SERIALIZED_ARCHIVE_DELTA_KEY)
    if not isinstance(delta, Mapping):
        return {}
    require_no_truthy_authority_fields(
        delta,
        context="family_agnostic_materializer_feedback.serialized_archive_delta",
    )
    schema = delta.get("schema")
    if schema not in (None, SERIALIZED_ARCHIVE_DELTA_SCHEMA):
        return {}
    return delta


def _archive_delta_status(saved_bytes: int) -> str:
    if saved_bytes > 0:
        return "realized_saving"
    if saved_bytes < 0:
        return "realized_cost"
    return "no_realized_delta"


def materializer_archive_delta(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    selected_key, selected = selected_materializer_delta(payload)
    if not selected_key:
        selected_key = SERIALIZED_ARCHIVE_DELTA_KEY
        selected = _serialized_archive_delta(payload)
    if not selected:
        return None
    source_archive = (
        payload.get("source_archive")
        if isinstance(payload.get("source_archive"), Mapping)
        else {}
    )
    candidate_archive = (
        payload.get("candidate_archive")
        if isinstance(payload.get("candidate_archive"), Mapping)
        else {}
    )
    source_bytes = optional_int(
        selected.get("source_archive_bytes") or source_archive.get("bytes")
    )
    candidate_bytes = optional_int(
        selected.get("candidate_archive_bytes") or candidate_archive.get("bytes")
    )
    saved_bytes = optional_int(
        selected.get("saved_bytes") if selected_key != SERIALIZED_ARCHIVE_DELTA_KEY else None
    )
    if saved_bytes is None:
        saved_bytes = optional_int(selected.get("realized_saved_bytes"))
    if saved_bytes is None:
        archive_delta_bytes = optional_int(selected.get("archive_delta_bytes"))
        if archive_delta_bytes is not None:
            saved_bytes = -archive_delta_bytes
    if saved_bytes is None and source_bytes is not None and candidate_bytes is not None:
        saved_bytes = source_bytes - candidate_bytes
    if saved_bytes is None and source_bytes is None and candidate_bytes is None:
        return None
    if saved_bytes is None:
        saved_bytes = 0
    status = str(selected.get("status") or _archive_delta_status(saved_bytes))
    savings_realized = (
        selected.get("savings_realized")
        if isinstance(selected.get("savings_realized"), bool)
        else saved_bytes > 0
    )
    return {
        "selected_materialization_key": selected_key or None,
        "realized_saved_bytes": saved_bytes,
        "source_archive_bytes": source_bytes,
        "candidate_archive_bytes": candidate_bytes,
        "savings_realized": savings_realized,
        "status": status,
    }


def materializer_observation_from_manifest(
    result: Mapping[str, Any],
    *,
    row_slug: str | None = None,
    manifest_path: str | None = None,
    label: str | None = None,
    axis: str = DEFAULT_LOCAL_MATERIALIZER_AXIS,
    scorer_version: str = FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
    rate_score_per_byte: float = DEFAULT_MATERIALIZER_RATE_SCORE_PER_BYTE,
) -> dict[str, Any] | None:
    """Build one canonical empirical-observation row from a materializer manifest."""

    require_no_truthy_authority_fields(
        result,
        context="family_agnostic_materializer_feedback.manifest",
    )
    archive_delta = materializer_archive_delta(result)
    if archive_delta is None:
        return None
    target_kind = str(result.get("target_kind") or "").strip()
    if not target_kind:
        return None

    saved_bytes = int(archive_delta.get("realized_saved_bytes") or 0)
    selected_key, selected = selected_materializer_delta(result)
    if not selected_key:
        selected_key = str(archive_delta.get("selected_materialization_key") or "")
        selected = _as_mapping(result.get(selected_key))
    readiness_blockers = _string_sequence(result.get("readiness_blockers"))
    source_archive = _as_mapping(result.get("source_archive"))
    candidate_archive = _as_mapping(result.get("candidate_archive"))
    proof_write = _as_mapping(result.get("runtime_consumption_proof_write"))
    receiver_verification = _as_mapping(result.get("receiver_verification"))
    source_sha = str(source_archive.get("sha256") or "").strip()
    candidate_sha = str(candidate_archive.get("sha256") or "").strip()
    stable_tail = candidate_sha[:12] or source_sha[:12] or _slug(row_slug or target_kind)
    observation_id = str(row_slug or result.get("observation_id") or "").strip()
    if not observation_id:
        observation_id = f"materializer_feedback_{_slug(target_kind)}_{stable_tail}"
    candidate_id = str(result.get("candidate_id") or observation_id).strip()
    receiver_contract_satisfied = result.get("receiver_contract_satisfied") is True
    inflate_parity_satisfied = result.get("inflate_parity_satisfied") is True
    savings_realized = bool(archive_delta.get("savings_realized"))
    rate_positive = (
        savings_realized
        and saved_bytes > 0
        and "candidate_not_rate_positive" not in readiness_blockers
    )
    observed_rate_gain = (
        float(rate_score_per_byte) * float(saved_bytes) if rate_positive else 0.0
    )
    observed_score_gain = (
        observed_rate_gain
        if receiver_contract_satisfied or inflate_parity_satisfied
        else 0.0
    )
    selected_member_names = _string_sequence(result.get("selected_member_names"))
    selected_member_name = str(result.get("selected_member_name") or "").strip()
    if selected_member_name and selected_member_name not in selected_member_names:
        selected_member_names.insert(0, selected_member_name)
    source_unit_ids = _string_sequence(result.get("source_unit_ids"))
    if not source_unit_ids:
        source_unit_ids = _string_sequence(result.get("backlog_keys"))
    if not source_unit_ids:
        source_unit_ids = selected_member_names[:1]
    if not source_unit_ids:
        source_unit_ids = [candidate_id]

    return {
        "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
        "observation_kind": "family_agnostic_materializer_empirical_observation",
        "observation_id": observation_id,
        "candidate_id": candidate_id,
        "axis": axis,
        "resource_kind": result.get("resource_kind") or "local_cpu",
        "runtime_identity": {
            "runtime_contract_sha256": proof_write.get("sha256"),
            "scorer_version": scorer_version,
        },
        "cache_identity": {
            "cache_sha256": candidate_sha or source_sha or None,
            "source_archive_sha256": source_sha or None,
        },
        "archive_label": label,
        "target_kind": target_kind,
        "materializer_id": result.get("materializer_id"),
        "portability_contract": result.get("portability_contract"),
        "receiver_contract_kind": result.get("receiver_contract_kind"),
        "source_archive_path": source_archive.get("path"),
        "source_archive_sha256": source_sha or None,
        "source_archive_bytes": source_archive.get("bytes"),
        "candidate_archive_path": candidate_archive.get("path"),
        "candidate_archive_sha256": candidate_sha or None,
        "candidate_archive_bytes": candidate_archive.get("bytes"),
        "artifact_bytes": candidate_archive.get("bytes") or source_archive.get("bytes") or 0,
        "saved_bytes": saved_bytes,
        "archive_delta_status": archive_delta.get("status"),
        "observed_rate_gain": observed_rate_gain,
        "observed_score_gain": observed_score_gain,
        "rate_positive": rate_positive,
        "savings_realized": savings_realized,
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "inflate_parity_satisfied": inflate_parity_satisfied,
        "receiver_verification_blockers": receiver_verification.get("blockers") or [],
        "readiness_blockers": readiness_blockers,
        "runtime_consumption_proof_path": result.get("runtime_consumption_proof_path"),
        "manifest_path": manifest_path,
        "source_path": manifest_path,
        "selected_member_name": selected_member_name or None,
        "selected_member_names": selected_member_names,
        "selection_scope": result.get("selection_scope"),
        "selected_materialization_key": selected_key or None,
        "selected_materialization": dict(selected),
        "serialized_archive_delta": dict(selected)
        if selected_key == SERIALIZED_ARCHIVE_DELTA_KEY
        else {},
        "selected_elision": dict(selected) if selected_key == "selected_elision" else {},
        "section_recode": dict(selected) if selected_key == "section_recode" else {},
        "selected_compression": dict(selected)
        if selected_key == "selected_compression"
        else {},
        "selected_merge": dict(selected) if selected_key == "selected_merge" else {},
        "selected_payload": dict(selected) if selected_key == "selected_payload" else {},
        "factorization": dict(selected) if selected_key == "factorization" else {},
        "recommended_planner_action": _recommended_planner_action(
            target_kind=target_kind,
            rate_positive=rate_positive,
            receiver_contract_satisfied=receiver_contract_satisfied,
        ),
        "source_unit_ids": source_unit_ids,
        "source_selection_ids": _string_sequence(result.get("source_selection_ids")),
        "work_ids": _string_sequence(result.get("work_ids")),
        "backlog_keys": _string_sequence(result.get("backlog_keys")),
        "observation_feedback_is_not_score_authority": True,
        **MATERIALIZER_FALSE_AUTHORITY,
    }


def _false_authority_observation(row: Mapping[str, Any], *, source_path: str | None) -> dict[str, Any]:
    require_no_truthy_authority_fields(
        row,
        context="family_agnostic_materializer_feedback.observation",
    )
    out = dict(row)
    out.setdefault("schema", FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA)
    out.setdefault(
        "observation_kind",
        "family_agnostic_materializer_empirical_observation",
    )
    if source_path and not out.get("source_path"):
        out["source_path"] = source_path
    out["observation_feedback_is_not_score_authority"] = True
    for key, value in MATERIALIZER_FALSE_AUTHORITY.items():
        out[key] = value
    return out


def materializer_observation_feedback_rows(
    payloads: Mapping[str, Any] | list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    source_path: str | None = None,
    default_axis: str = DEFAULT_LOCAL_MATERIALIZER_AXIS,
    rate_score_per_byte: float = DEFAULT_MATERIALIZER_RATE_SCORE_PER_BYTE,
) -> list[dict[str, Any]]:
    """Normalize manifests, sweeps, or observation rows into planner feedback rows."""

    raw_payloads: list[Any] = [payloads] if isinstance(payloads, Mapping) else list(payloads)
    rows: list[dict[str, Any]] = []
    for payload in raw_payloads:
        if not isinstance(payload, Mapping):
            continue
        schema = str(payload.get("schema") or "")
        observation_kind = str(payload.get("observation_kind") or "")
        observations = payload.get("observations")
        if observations is None:
            observations = payload.get("rows")
        if schema == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA or isinstance(
            observations,
            list,
        ):
            require_no_truthy_authority_fields(
                payload,
                context="family_agnostic_materializer_feedback.sweep",
            )
            for observation in observations if isinstance(observations, list) else []:
                if isinstance(observation, Mapping):
                    rows.append(
                        _false_authority_observation(
                            observation,
                            source_path=source_path,
                        )
                    )
            continue
        if (
            schema == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA
            or observation_kind == "family_agnostic_materializer_empirical_observation"
        ):
            rows.append(_false_authority_observation(payload, source_path=source_path))
            continue
        row = materializer_observation_from_manifest(
            payload,
            row_slug=str(payload.get("row_slug") or ""),
            manifest_path=source_path,
            label=payload.get("archive_label"),
            axis=default_axis,
            rate_score_per_byte=rate_score_per_byte,
        )
        if row is not None:
            rows.append(row)
    return rows


__all__ = [
    "DEFAULT_LOCAL_MATERIALIZER_AXIS",
    "DEFAULT_MATERIALIZER_RATE_SCORE_PER_BYTE",
    "FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA",
    "FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA",
    "MATERIALIZER_DELTA_KEYS",
    "MATERIALIZER_FALSE_AUTHORITY",
    "SERIALIZED_ARCHIVE_DELTA_KEY",
    "materializer_archive_delta",
    "materializer_observation_feedback_rows",
    "materializer_observation_from_manifest",
    "optional_int",
    "selected_materializer_delta",
]
