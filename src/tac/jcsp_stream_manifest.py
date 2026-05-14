# SPDX-License-Identifier: MIT
"""Typed manifest rows for byte-bearing JCSP archive members.

The JCSP builder already produces byte-closed ``jcsp.bin`` archive members and
the robust-current runtime bridge already probes those members fail-closed.
This adapter normalizes that evidence into one deterministic row that planning
and runtime orchestration can consume without treating loader parity as a score
claim or dispatch unlock.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.joint_codec_stack_orchestrator import (
    JCSP_ARCHIVE_MEMBER_NAME,
    JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA,
    JCSP_STREAM_ARCHIVE_BYTE_RECONCILIATION_SCHEMA,
    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
    JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER,
    load_jcsp_archive_member_for_runtime,
    unpack_jcsp_container,
)

JCSP_STREAM_MANIFEST_ROW_SCHEMA = "jcsp_byte_bearing_stream_manifest_row_v1"
JCSP_STREAM_MANIFEST_SOURCE_TOOL = "tac.jcsp_stream_manifest"
JCSP_DEFAULT_CANDIDATE_ID = "joint_admm_balle_arithmetic_stack"
JCSP_DEFAULT_RUNTIME_CONSUMER_BLOCKER = "submission_runtime_stream_consumer_missing"


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid SHA-256 hex string") from exc
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _read_single_jcsp_member_payload(
    *,
    archive_bytes: bytes,
    member_name: str,
) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
            return zf.read(member_name)
    except KeyError as exc:
        raise ValueError(f"JCSP member {member_name!r} not found") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError("archive_bytes are not a valid ZIP archive") from exc


def _stream_payload_rows(container_bytes: bytes) -> list[dict[str, Any]]:
    parsed = unpack_jcsp_container(container_bytes)
    rows: list[dict[str, Any]] = []
    for index, stream in enumerate(parsed["streams"]):
        payload = bytes(stream["payload"])
        payload_magic = stream["payload_magic"]
        if isinstance(payload_magic, bytes):
            payload_magic_text = payload_magic.decode("ascii", errors="replace")
        else:
            payload_magic_text = str(payload_magic)
        rows.append(
            {
                "index": index,
                "name": str(stream["name"]),
                "codec_kind": int(stream["codec_kind"]),
                "admm_bytes_target": int(stream["admm_bytes_target"]),
                "actual_bytes": _require_positive_int(
                    int(stream["actual_bytes"]),
                    field=f"streams[{index}].actual_bytes",
                ),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_magic": payload_magic_text,
                "score_delta": float(stream["score_delta"]),
                "marginal": float(stream["marginal"]),
                "runtime_dispatch_checked": True,
            }
        )
    return rows


def build_jcsp_stream_manifest_row(
    *,
    archive_bytes: bytes | bytearray | memoryview,
    archive_path: str | Path | None = None,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    candidate_id: str = JCSP_DEFAULT_CANDIDATE_ID,
    source_manifest_path: str | Path | None = None,
    source_manifest_sha256: str | None = None,
    source_tool: str = JCSP_STREAM_MANIFEST_SOURCE_TOOL,
) -> dict[str, Any]:
    """Return a deterministic typed row for a real byte-bearing JCSP member.

    The adapter fails closed unless the archive bytes validate through the
    existing JCSP runtime-loader contract and the runtime-consumption contract
    records the required fail-closed consumer fields. The emitted row remains
    non-dispatchable until submission runtime consumption, raw-output parity,
    a lane claim, and exact CUDA auth eval exist.
    """

    if not isinstance(archive_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(
            "archive_bytes must be bytes-like, got "
            f"{type(archive_bytes).__name__}"
        )
    source_manifest_sha = (
        _require_sha256(
            source_manifest_sha256,
            field="source_manifest_sha256",
        )
        if source_manifest_sha256 is not None
        else ""
    )
    archive_blob = bytes(archive_bytes)
    if not archive_blob:
        raise ValueError("archive_bytes must be non-empty")

    contract = load_jcsp_archive_member_for_runtime(
        archive_bytes=archive_blob,
        member_name=member_name,
        require_single_member=True,
    )
    archive_size = _require_positive_int(
        contract.get("archive_bytes"),
        field="archive_bytes",
    )
    member_size = _require_positive_int(
        contract.get("member_bytes"),
        field="member_bytes",
    )
    archive_sha256 = _require_sha256(
        contract.get("archive_sha256"),
        field="archive_sha256",
    )
    member_sha256 = _require_sha256(
        contract.get("member_sha256"),
        field="member_sha256",
    )

    container_bytes = _read_single_jcsp_member_payload(
        archive_bytes=archive_blob,
        member_name=member_name,
    )
    streams = _stream_payload_rows(container_bytes)
    if not streams:
        raise ValueError("JCSP manifest row requires at least one stream payload")
    stream_payload_bytes = sum(int(row["actual_bytes"]) for row in streams)
    container_overhead_bytes = member_size - stream_payload_bytes
    if container_overhead_bytes < 0:
        raise ValueError("member_bytes are smaller than summed JCSP stream bytes")
    compressed_member_bytes = _require_positive_int(
        contract.get("member_compress_size"),
        field="member_compress_size",
    )
    archive_wrapper_bytes = archive_size - compressed_member_bytes
    if archive_wrapper_bytes < 0:
        raise ValueError("archive_bytes are smaller than compressed member bytes")

    runtime_contract = contract.get("runtime_consumption_contract")
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("runtime_consumption_contract is required")
    output_contract = runtime_contract.get("contest_output_contract")
    if not isinstance(output_contract, Mapping):
        raise ValueError("contest_output_contract is required")
    parity_contract = output_contract.get("raw_output_parity_contract")
    if not isinstance(parity_contract, Mapping):
        raise ValueError("raw_output_parity_contract is required")
    if parity_contract.get("schema") != JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA:
        raise ValueError("raw_output_parity_contract has the wrong schema")

    runtime_blockers = _ordered_unique(
        runtime_contract.get("dispatch_blockers", ())
        if isinstance(runtime_contract.get("dispatch_blockers"), list)
        else ()
    )
    if JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER not in runtime_blockers:
        raise ValueError("runtime consumption blocker is missing")
    if JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER not in runtime_blockers:
        raise ValueError("runtime output-parity blocker is missing")

    contract_blockers = _ordered_unique(
        contract.get("dispatch_blockers", ())
        if isinstance(contract.get("dispatch_blockers"), list)
        else ()
    )
    dispatch_blockers = _ordered_unique(
        [
            *contract_blockers,
            *runtime_blockers,
            JCSP_DEFAULT_RUNTIME_CONSUMER_BLOCKER,
            "no_lane_dispatch_claim",
            "exact_cuda_auth_eval_missing",
        ]
    )

    row: dict[str, Any] = {
        "schema": JCSP_STREAM_MANIFEST_ROW_SCHEMA,
        "source_tool": source_tool,
        "source_path": str(source_manifest_path or ""),
        "source_manifest_sha256": source_manifest_sha,
        "key": candidate_id,
        "candidate_id": candidate_id,
        "component": "jcsp_archive_member",
        "typed_output_contract": "JCSP container bytes inside deterministic archive.zip member",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_runtime_loader": contract.get("ready_for_runtime_loader") is True,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_closed_archive_member": True,
        "single_member_no_sidecars": contract.get("archive_members") == [member_name],
        "archive_path": str(archive_path or ""),
        "archive_bytes": archive_size,
        "archive_sha256": archive_sha256,
        "member_name": member_name,
        "member_bytes": member_size,
        "member_sha256": member_sha256,
        "member_compress_size": compressed_member_bytes,
        "member_compress_type": int(contract["member_compress_type"]),
        "stream_count": len(streams),
        "stream_payload_bytes": stream_payload_bytes,
        "member_container_overhead_bytes": container_overhead_bytes,
        "archive_wrapper_overhead_bytes": archive_wrapper_bytes,
        "streams": streams,
        "runtime_consumer": {
            "schema": runtime_contract.get("schema"),
            "required_submission_runtime": runtime_contract.get(
                "required_submission_runtime"
            ),
            "runtime_bridge_path": runtime_contract.get("runtime_bridge_path"),
            "required_member_name": runtime_contract.get("required_member_name"),
            "detects_required_member": runtime_contract.get(
                "detects_required_member"
            )
            is True,
            "consumes_required_member": runtime_contract.get(
                "consumes_required_member"
            )
            is True,
            "ready_for_runtime_loader": contract.get("ready_for_runtime_loader")
            is True,
            "ready_for_submission_runtime_consumption": False,
            "ready_for_exact_eval_dispatch": False,
            "output_contract_schema": output_contract.get("schema"),
            "raw_output_parity_contract_schema": parity_contract.get("schema"),
            "dispatch_blockers": runtime_blockers,
        },
        "byte_reconciliation": {
            "schema": JCSP_STREAM_ARCHIVE_BYTE_RECONCILIATION_SCHEMA,
            "stream_payload_bytes": stream_payload_bytes,
            "member_payload_bytes": member_size,
            "member_container_overhead_bytes": container_overhead_bytes,
            "charged_archive_bytes": archive_size,
            "compressed_member_bytes": compressed_member_bytes,
            "archive_wrapper_overhead_bytes": archive_wrapper_bytes,
            "stream_payloads_inside_member": True,
        },
        "dispatch_blockers": dispatch_blockers,
        "fail_closed_criteria": [
            "refuse_if_archive_sha256_missing",
            "refuse_if_member_sha256_missing",
            "refuse_if_runtime_consumption_contract_missing",
            "refuse_if_runtime_output_parity_contract_missing",
            "refuse_if_submission_runtime_consumption_blocker_missing",
            "refuse_if_exact_cuda_auth_eval_missing",
        ],
        "next_required_proof": [
            "submission_runtime_consumes_jcsp_member",
            "raw_output_parity_artifact",
            "lane_dispatch_claim",
            "exact_cuda_auth_eval",
        ],
    }
    row["manifest_row_sha256"] = _canonical_json_sha256(row)
    return row


__all__ = [
    "JCSP_DEFAULT_CANDIDATE_ID",
    "JCSP_STREAM_MANIFEST_ROW_SCHEMA",
    "build_jcsp_stream_manifest_row",
]
