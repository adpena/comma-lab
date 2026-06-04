# SPDX-License-Identifier: MIT
"""Fail-closed HiNeRV official control-parity guard.

Local HiNeRV pruning, QuantNoise, decoder-waterfill, and receiver codec probes
are real controls, but they are not official HiNeRV bitstream/QAT parity unless
the official torchac encode/decode path is bound by numeric replay and byte
stream custody.  This guard makes that authority split machine-readable.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "hinerv_official_control_parity_guard.v1"
AUTHORITY = "false_authority_control_guard_no_score_claim"
FAMILY = "hi_nerv"

OFFICIAL_SOURCE_AUDIT_SCHEMA = "hinerv_official_source_parity_audit.v1"
OFFICIAL_FORWARD_PARITY_SCHEMA = "hinerv_official_forward_parity.v1"
LOCAL_BITSTREAM_ROUNDTRIP_SCHEMA = "hi_nerv_bitstream_roundtrip_measurement.v1"
LOCAL_OFFICIAL_ENTROPY_SCHEMA = "hi_nerv_official_entropy_receiver_consumption.v1"

PRUNE_QUANT_TORCHAC_CONTROL_ID = "hi_nerv_prune_quant_torchac_bitstream"
PRUNE_QUANT_COMPONENT_ID = "prune_quant_codec"
OFFICIAL_QUANT_PRUNE_GROUP_ID = "official_quant_prune_torchac_bitstream"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
    "ready_for_exact_eval_dispatch": False,
}


class HiNervOfficialControlParityGuardError(ValueError):
    """Raised when a caller requires official control parity but it is blocked."""


def build_hinerv_official_control_parity_guard(
    *,
    source_audit_report: Mapping[str, Any] | None = None,
    source_audit_report_path: str | Path | None = None,
    forward_parity_artifact: Mapping[str, Any] | None = None,
    forward_parity_artifact_path: str | Path | None = None,
    local_bitstream_report: Mapping[str, Any] | None = None,
    local_bitstream_report_path: str | Path | None = None,
    claimed_control_ids: Iterable[str] = (PRUNE_QUANT_TORCHAC_CONTROL_ID,),
) -> dict[str, Any]:
    """Return a fail-closed control-parity verdict for requested HiNeRV controls."""

    source_audit = _load_optional_mapping(
        source_audit_report,
        source_audit_report_path,
    )
    forward_artifact = _load_optional_mapping(
        forward_parity_artifact,
        forward_parity_artifact_path,
    )
    local_report = _load_optional_mapping(
        local_bitstream_report,
        local_bitstream_report_path,
    )
    control_ids = _claimed_control_ids(claimed_control_ids)
    rows = [
        _build_prune_quant_torchac_row(
            source_audit=source_audit,
            forward_artifact=forward_artifact,
            local_report=local_report,
        )
        for control_id in control_ids
        if control_id == PRUNE_QUANT_TORCHAC_CONTROL_ID
    ]
    unknown = [
        control_id
        for control_id in control_ids
        if control_id != PRUNE_QUANT_TORCHAC_CONTROL_ID
    ]
    unknown_rows = [
        {
            "schema": "hinerv_official_control_parity_row.v1",
            "control_id": control_id,
            "family": FAMILY,
            "official_control_parity_proven": False,
            "status": "blocked_unknown_control",
            "blockers": [f"hinerv_official_control_guard_unknown:{control_id}"],
            **FALSE_AUTHORITY,
        }
        for control_id in unknown
    ]
    rows.extend(unknown_rows)
    blockers = _ordered_unique(
        blocker
        for row in rows
        for blocker in row.get("blockers", ())
    )
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "family": FAMILY,
        "claimed_control_ids": control_ids,
        "control_rows": rows,
        "official_control_parity_proven": bool(rows) and not blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def require_hinerv_official_control_parity(
    *,
    source_audit_report: Mapping[str, Any] | None = None,
    source_audit_report_path: str | Path | None = None,
    forward_parity_artifact: Mapping[str, Any] | None = None,
    forward_parity_artifact_path: str | Path | None = None,
    local_bitstream_report: Mapping[str, Any] | None = None,
    local_bitstream_report_path: str | Path | None = None,
    claimed_control_ids: Iterable[str] = (PRUNE_QUANT_TORCHAC_CONTROL_ID,),
) -> dict[str, Any]:
    """Return the guard report or raise with exact blocker semantics."""

    report = build_hinerv_official_control_parity_guard(
        source_audit_report=source_audit_report,
        source_audit_report_path=source_audit_report_path,
        forward_parity_artifact=forward_parity_artifact,
        forward_parity_artifact_path=forward_parity_artifact_path,
        local_bitstream_report=local_bitstream_report,
        local_bitstream_report_path=local_bitstream_report_path,
        claimed_control_ids=claimed_control_ids,
    )
    if report["blockers"]:
        raise HiNervOfficialControlParityGuardError(
            "HiNeRV official control parity blocked: "
            + ", ".join(str(blocker) for blocker in report["blockers"])
        )
    return report


def _build_prune_quant_torchac_row(
    *,
    source_audit: Mapping[str, Any] | None,
    forward_artifact: Mapping[str, Any] | None,
    local_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    audit_component = _component_row(source_audit, PRUNE_QUANT_COMPONENT_ID)
    artifact_component = _component_row(forward_artifact, PRUNE_QUANT_COMPONENT_ID)
    if not artifact_component and audit_component:
        artifact_component = _mapping_or_empty(
            audit_component.get("forward_parity_artifact_component")
        )
    control_proof = _control_proof_mapping(artifact_component)
    local_entropy = _local_entropy_mapping(local_report)
    source_markers_present = _official_group_present(
        source_audit,
        OFFICIAL_QUANT_PRUNE_GROUP_ID,
    )
    component_source_forward = bool(
        audit_component.get("source_forward_parity_proven") is True
        or artifact_component.get("source_forward_parity_proven") is True
    )
    component_numeric_blockers = _numeric_component_blockers(artifact_component)
    schema_blockers = _schema_blockers(
        source_audit=source_audit,
        forward_artifact=forward_artifact,
    )
    source_false_authority_blockers = _false_authority_blockers(
        "hinerv_official_control_source_audit",
        source_audit,
    )
    artifact_false_authority_blockers = _false_authority_blockers(
        "hinerv_official_control_forward_artifact",
        forward_artifact,
    )
    torchac_bound = _bool_from_any(
        control_proof,
        (
            "official_torchac_encode_decode_bound",
            "torchac_encode_decode_bound",
            "torchac_roundtrip_bound",
        ),
    )
    torchac_streams_present = _bool_from_any(
        control_proof,
        (
            "official_torchac_byte_streams_present",
            "torchac_byte_streams_present",
            "torchac_byte_stream_present",
        ),
    )
    torchac_stream_sha256 = _sha_from_any(
        control_proof,
        (
            "official_torchac_stream_sha256",
            "official_torchac_byte_stream_sha256",
            "official_codec_bytes_sha256",
            "official_bitstream_sha256",
            "torchac_stream_sha256",
            "bitstream_sha256",
        ),
    )
    roundtrip_error = _float_from_any(
        control_proof,
        (
            "torchac_roundtrip_max_abs_error",
            "torchac_max_abs_error",
            "roundtrip_error",
            "max_abs_error",
            "max_abs_delta",
        ),
    )
    roundtrip_tolerance = _float_from_any(
        control_proof,
        (
            "torchac_roundtrip_tolerance",
            "tolerance",
            "max_abs_tolerance",
        ),
    )
    local_receiver_real = _local_receiver_control_real(local_report)
    blockers = _ordered_unique(
        [
            "hinerv_official_source_parity_audit_missing"
            if source_audit is None
            else "",
            "hinerv_official_quant_prune_torchac_source_markers_missing"
            if not source_markers_present
            else "",
            "hinerv_prune_quant_codec_source_forward_replay_missing"
            if not component_source_forward
            else "",
            *[
                f"{blocker}:{PRUNE_QUANT_COMPONENT_ID}"
                for blocker in component_numeric_blockers
            ],
            *schema_blockers,
            *source_false_authority_blockers,
            *artifact_false_authority_blockers,
            "hinerv_official_torchac_encode_decode_not_bound"
            if not torchac_bound
            else "",
            "hinerv_official_torchac_byte_streams_not_present"
            if not torchac_streams_present
            else "",
            "hinerv_official_torchac_stream_sha256_missing"
            if torchac_stream_sha256 is None
            else "",
            "hinerv_official_torchac_roundtrip_error_missing"
            if roundtrip_error is None
            else "",
            "hinerv_official_torchac_roundtrip_tolerance_missing"
            if roundtrip_tolerance is None or roundtrip_tolerance < 0.0
            else "",
            (
                "hinerv_official_torchac_roundtrip_error_exceeds_tolerance"
                if (
                    roundtrip_error is not None
                    and roundtrip_tolerance is not None
                    and roundtrip_error > roundtrip_tolerance
                )
                else ""
            ),
        ]
    )
    return {
        "schema": "hinerv_official_control_parity_row.v1",
        "control_id": PRUNE_QUANT_TORCHAC_CONTROL_ID,
        "component_id": PRUNE_QUANT_COMPONENT_ID,
        "official_group_id": OFFICIAL_QUANT_PRUNE_GROUP_ID,
        "family": FAMILY,
        "official_source_markers_present": source_markers_present,
        "local_receiver_control_real": local_receiver_real,
        "local_report_schema": None if local_report is None else local_report.get("schema"),
        "local_entropy_schema": local_entropy.get("schema"),
        "local_entropy_torchac_encode_decode_bound": bool(
            local_entropy.get("torchac_encode_decode_bound") is True
        ),
        "component_source_forward_parity_proven": component_source_forward,
        "official_torchac_encode_decode_bound": torchac_bound,
        "official_torchac_byte_streams_present": torchac_streams_present,
        "official_torchac_stream_sha256": torchac_stream_sha256,
        "official_torchac_roundtrip_max_abs_error": roundtrip_error,
        "official_torchac_roundtrip_tolerance": roundtrip_tolerance,
        "official_control_parity_proven": not blockers,
        "status": "official_control_parity_proven"
        if not blockers
        else "blocked_receiver_visible_or_weak_control_only",
        "blockers": blockers,
        "blocker_semantics": {
            "hinerv_prune_quant_codec_source_forward_replay_missing": (
                "The official prune/QuantNoise/torchac component lacks "
                "source-forward replay authority."
            ),
            "hinerv_official_torchac_encode_decode_not_bound": (
                "Local receiver entropy estimates or tensor codecs do not bind "
                "the official torchac encode/decode CDF streams."
            ),
            "hinerv_official_torchac_stream_sha256_missing": (
                "The proof does not identify the official encoded byte stream."
            ),
        },
        **FALSE_AUTHORITY,
    }


def _load_optional_mapping(
    payload: Mapping[str, Any] | None,
    path: str | Path | None,
) -> Mapping[str, Any] | None:
    if payload is not None:
        return payload
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise HiNervOfficialControlParityGuardError(
            f"expected JSON object in {path}"
        )
    return data


def _claimed_control_ids(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        control_id = str(value).strip()
        if not control_id or control_id in seen:
            continue
        seen.add(control_id)
        out.append(control_id)
    if not out:
        out.append(PRUNE_QUANT_TORCHAC_CONTROL_ID)
    return out


def _component_row(
    payload: Mapping[str, Any] | None,
    component_id: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    for key in ("component_state_rows", "component_rows", "control_rows"):
        rows = payload.get(key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if (
                isinstance(row, Mapping)
                and str(row.get("component_id") or "") == component_id
            ):
                return row
    forward_row = _mapping_or_empty(payload.get("official_forward_parity_artifact_row"))
    return _component_row(forward_row, component_id)


def _control_proof_mapping(component_row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in (
        "official_torchac_replay",
        "official_bitstream_replay",
        "official_control_replay",
        "official_entropy_receiver_consumption",
    ):
        nested = _mapping_or_empty(component_row.get(key))
        if nested:
            return nested
    return component_row


def _local_entropy_mapping(
    local_report: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(local_report, Mapping):
        return {}
    if local_report.get("schema") == LOCAL_OFFICIAL_ENTROPY_SCHEMA:
        return local_report
    return _mapping_or_empty(local_report.get("official_entropy_receiver_consumption"))


def _local_receiver_control_real(local_report: Mapping[str, Any] | None) -> bool:
    if not isinstance(local_report, Mapping):
        return False
    if local_report.get("schema") == LOCAL_BITSTREAM_ROUNDTRIP_SCHEMA:
        rows = local_report.get("rows")
        return (
            isinstance(rows, Sequence)
            and not isinstance(rows, (str, bytes))
            and any(isinstance(row, Mapping) for row in rows)
        )
    preparation = _mapping_or_empty(local_report.get("preparation"))
    pruning = _mapping_or_empty(preparation.get("pruning"))
    quant_noise = _mapping_or_empty(preparation.get("quant_noise"))
    return bool(
        _int_or_none(pruning.get("actual_new_zero_values"))
        or _int_or_none(quant_noise.get("actual_changed_value_count"))
    )


def _official_group_present(
    source_audit: Mapping[str, Any] | None,
    group_id: str,
) -> bool:
    if not isinstance(source_audit, Mapping):
        return False
    rows = source_audit.get("official_marker_group_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    return any(
        isinstance(row, Mapping)
        and str(row.get("group_id") or "") == group_id
        and row.get("all_markers_present") is True
        for row in rows
    )


def _numeric_component_blockers(row: Mapping[str, Any]) -> list[str]:
    if not row:
        return ["component_missing"]
    blockers: list[str] = []
    if row.get("source_forward_parity_proven") is not True:
        blockers.append("component_not_proven")
    tolerance = _float_from_any(row, ("tolerance", "max_abs_tolerance"))
    max_abs_error = _float_from_any(
        row,
        ("max_abs_error", "max_error", "max_abs_delta"),
    )
    if tolerance is None or tolerance < 0.0:
        blockers.append("numeric_tolerance_missing")
    if max_abs_error is None:
        blockers.append("numeric_max_abs_error_missing")
    elif tolerance is not None and max_abs_error > tolerance:
        blockers.append("numeric_max_abs_error_exceeds_tolerance")
    for field in (
        "input_sha256",
        "official_output_sha256",
        "portable_output_sha256",
    ):
        if not _is_sha256_hex(row.get(field)):
            blockers.append(f"{field}_missing")
    if row.get("official_output_sha256") != row.get("portable_output_sha256"):
        blockers.append("official_portable_output_sha256_mismatch")
    if not (
        _is_sha256_hex(row.get("official_weight_sha256"))
        or _is_sha256_hex(row.get("official_source_sha256"))
    ):
        blockers.append("official_weight_identity_missing")
    return blockers


def _schema_blockers(
    *,
    source_audit: Mapping[str, Any] | None,
    forward_artifact: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if (
        isinstance(source_audit, Mapping)
        and source_audit.get("schema") != OFFICIAL_SOURCE_AUDIT_SCHEMA
    ):
        blockers.append("hinerv_official_source_parity_audit_schema_invalid")
    if (
        isinstance(forward_artifact, Mapping)
        and forward_artifact.get("schema") != OFFICIAL_FORWARD_PARITY_SCHEMA
    ):
        blockers.append("hinerv_official_forward_parity_artifact_schema_invalid")
    return blockers


def _false_authority_blockers(
    prefix: str,
    payload: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    blockers: list[str] = []
    for field in (
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if field in payload and payload.get(field) is not False:
            blockers.append(f"{prefix}_{field}_not_false")
    return blockers


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_from_any(payload: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(payload.get(key) is True for key in keys)


def _sha_from_any(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if _is_sha256_hex(value):
            return str(value)
    return None


def _float_from_any(payload: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for operator/preflight use."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit-report", type=Path)
    parser.add_argument("--forward-parity-artifact", type=Path)
    parser.add_argument("--local-bitstream-report", type=Path)
    parser.add_argument(
        "--claimed-control-id",
        action="append",
        dest="claimed_control_ids",
        default=[],
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--require",
        action="store_true",
        help="Return nonzero when requested official control parity is blocked.",
    )
    args = parser.parse_args(argv)
    report = build_hinerv_official_control_parity_guard(
        source_audit_report_path=args.source_audit_report,
        forward_parity_artifact_path=args.forward_parity_artifact,
        local_bitstream_report_path=args.local_bitstream_report,
        claimed_control_ids=(
            tuple(args.claimed_control_ids)
            if args.claimed_control_ids
            else (PRUNE_QUANT_TORCHAC_CONTROL_ID,)
        ),
    )
    blob = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob, encoding="utf-8")
    else:
        print(blob, end="")
    if args.require and report["blockers"]:
        return 1
    return 0


__all__ = [
    "AUTHORITY",
    "FAMILY",
    "PRUNE_QUANT_COMPONENT_ID",
    "PRUNE_QUANT_TORCHAC_CONTROL_ID",
    "SCHEMA",
    "HiNervOfficialControlParityGuardError",
    "build_hinerv_official_control_parity_guard",
    "main",
    "require_hinerv_official_control_parity",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
