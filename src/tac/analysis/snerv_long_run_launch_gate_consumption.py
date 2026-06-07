# SPDX-License-Identifier: MIT
"""Shared SNeRV long-run launch-gate consumption checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.analysis.nerv_long_run_launch_gate import NERV_LONG_RUN_LAUNCH_GATE_SCHEMA
from tac.analysis.snerv_source_forward_proof import (
    SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import (
    require_no_truthy_authority_fields,
)

SNERV_LONG_RUN_LAUNCH_GATE_CONSUMPTION_SCHEMA = "snerv_long_run_launch_gate_consumption.v1"
SNERV_LF_HF_REPLACEMENT_CANDIDATE_ROW_SCHEMA = "snerv_lf_hf_replacement_candidate_row.v1"
SNERV_LF_HF_BOUNDED_STATUSES = frozenset(
    {
        "local_bounded_smoke_ready_no_authority",
        "blocked_until_prerequisite_evidence",
    }
)

_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "promotable",
    "ready_for_exact_eval_dispatch",
    "ready_for_provider_dispatch",
    "dispatch_attempted",
    "gpu_launched",
    "exact_cuda_auth_eval",
    "contest_cuda_auth_eval",
    "contest_cpu_auth_eval",
)


def snerv_long_run_launch_gate_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the launch-gate consumption packet carried by a row, if any."""

    direct = row.get("snerv_long_run_launch_gate")
    if isinstance(direct, Mapping):
        return direct
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("snerv_long_run_launch_gate")
        if isinstance(nested, Mapping):
            return nested
        source_row = metadata.get("source_selected_row")
        if isinstance(source_row, Mapping):
            nested = snerv_long_run_launch_gate_from_row(source_row)
            if nested:
                return nested
    launch = row.get("launch_authority_contract")
    if isinstance(launch, Mapping):
        nested = launch.get("snerv_long_run_launch_gate")
        if isinstance(nested, Mapping):
            return nested
    return {}


def snerv_row_is_bounded_proof_not_long_training(row: Mapping[str, Any]) -> bool:
    """Return true for SNeRV diagnostic/bounded rows allowed to create evidence."""

    for carrier in (row, row.get("metadata"), row.get("launch_authority_contract")):
        if isinstance(carrier, Mapping) and carrier.get(
            "current_command_is_bounded_proof_not_long_training"
        ) is True:
            return True

    status = str(row.get("status") or "").strip().lower()
    if (
        row.get("schema") == SNERV_LF_HF_REPLACEMENT_CANDIDATE_ROW_SCHEMA
        and status in SNERV_LF_HF_BOUNDED_STATUSES
    ):
        return True
    return (
        str(row.get("candidate_class") or "") == "learned_lf_hf_replacement"
        and isinstance(row.get("bounded_training_binding_contract"), Mapping)
    )


def snerv_long_run_launch_gate_blockers(
    gate: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    label_prefix: str = "selected_row_snerv_long_run_launch_gate",
) -> list[str]:
    """Return fail-closed blockers for SNeRV long-run launch-gate consumption."""

    if _normalize_family(row.get("family")) != "snerv":
        return []
    if snerv_row_is_bounded_proof_not_long_training(row):
        return []
    if not gate:
        return [f"{label_prefix}_missing"]

    blockers: list[str] = []
    blockers.extend(_authority_blockers(gate, label=label_prefix))
    if gate.get("schema") != SNERV_LONG_RUN_LAUNCH_GATE_CONSUMPTION_SCHEMA:
        blockers.append(f"{label_prefix}_schema_mismatch")
    if gate.get("required") is not True:
        blockers.append(f"{label_prefix}_not_required")
    if gate.get("approved") is not True:
        blockers.append(f"{label_prefix}_not_approved")
    if gate.get("verdict_schema") != NERV_LONG_RUN_LAUNCH_GATE_SCHEMA:
        blockers.append(f"{label_prefix}_verdict_schema_mismatch")
    if gate.get("gate_highest_level") != "L4":
        blockers.append(f"{label_prefix}_not_l4")
    if gate.get("source_forward_action_effect_indexed") is not True:
        blockers.append(f"{label_prefix}_source_forward_action_effect_missing")
    for item in _string_list(gate.get("blockers")):
        blockers.append(f"{label_prefix}_blocker:{item}")
    indexed_schemas = _string_list(gate.get("indexed_evidence_schemas"))
    if indexed_schemas and SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA not in indexed_schemas:
        blockers.append(f"{label_prefix}_index_missing_source_forward_schema")
    return _dedupe(blockers)


def snerv_long_run_launch_gate_guard(
    row: Mapping[str, Any],
    *,
    label_prefix: str = "selected_row_snerv_long_run_launch_gate",
) -> dict[str, Any]:
    gate = snerv_long_run_launch_gate_from_row(row)
    blockers = snerv_long_run_launch_gate_blockers(
        gate,
        row=row,
        label_prefix=label_prefix,
    )
    return {
        "schema": "snerv_long_run_launch_gate_consumption_guard.v1",
        "family": _normalize_family(row.get("family")),
        "bounded_proof_not_long_training": snerv_row_is_bounded_proof_not_long_training(row),
        "gate_present": bool(gate),
        "gate_approved": bool(gate.get("approved") is True) if gate else False,
        "passed": not blockers,
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _authority_blockers(payload: Mapping[str, Any], *, label: str) -> list[str]:
    blockers: list[str] = []
    for field in _AUTHORITY_FIELDS:
        if payload.get(field) is True:
            blockers.append(_safe_blocker_text(f"{label}:{field}=truthy"))
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        blockers.append(_safe_blocker_text(str(exc)))
    return blockers


def _normalize_family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"hinerv", "hi_nerv"}:
        return "hi_nerv"
    if text == "snerv":
        return "snerv"
    return text or "unknown"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _safe_blocker_text(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )[:240]
