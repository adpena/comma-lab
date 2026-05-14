# SPDX-License-Identifier: MIT
"""Deterministic CodecOp bitstream materialization.

``codec_op_admm_adapter`` and the CodecOp search tools intentionally emit
planning rows, not contest archives. This module is the next byte-custody
step: it takes a reviewed CodecOp encode/result manifest plus the actual
payload bytes and writes a deterministic charged-byte envelope with a golden
vector manifest. It never claims score or dispatch readiness.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.repo_io import json_text, sha256_bytes, sha256_file

SCHEMA_VERSION = "codec_op_bitstream_materializer.v1"
HEADER_SCHEMA_VERSION = "codec_op_bitstream_header.v1"
BITSTREAM_MAGIC = b"COBM1"
BITSTREAM_MAGIC_TEXT = BITSTREAM_MAGIC.decode("ascii")
HEADER_LENGTH_FORMAT = ">I"
HEADER_LENGTH_BYTES = struct.calcsize(HEADER_LENGTH_FORMAT)
HEADER_PREFIX_BYTES = len(BITSTREAM_MAGIC) + HEADER_LENGTH_BYTES
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

DEFAULT_SCORE_BLOCKERS = (
    "archive_substitution_required_before_score_use",
    "exact_cuda_auth_eval_required_before_score_or_dispatch",
    "pre_submission_compliance_check_required_before_release_or_promotion",
)


class CodecOpBitstreamMaterializerError(ValueError):
    """Raised when materialization cannot produce trustworthy byte custody."""


def materialize_codec_op_bitstream(
    source: Mapping[str, Any],
    *,
    output_blob: Path | str,
    manifest_output: Path | str | None = None,
    source_manifest_path: Path | str | None = None,
    payload_path: Path | str | None = None,
    candidate_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write a deterministic CodecOp bitstream envelope and custody manifest.

    Hard failures:

    * no actual payload bytes are supplied;
    * supplied bytes disagree with expected byte/SHA custody;
    * the source does not carry an explicit full roundtrip/decode proof.

    CPU-only evidence and absent archive identity are recorded as fail-closed
    blockers in the manifest. The blob is still useful as a planning/golden
    vector, but the manifest remains non-dispatchable and non-promotable.
    """

    if not isinstance(source, Mapping):
        raise CodecOpBitstreamMaterializerError("source must be a mapping")
    blob_path = Path(output_blob)
    manifest_path = Path(manifest_output) if manifest_output is not None else None
    _ensure_write_target(blob_path, force=force, label="output blob")
    if manifest_path is not None:
        _ensure_write_target(manifest_path, force=force, label="manifest output")

    source_path = Path(source_manifest_path) if source_manifest_path is not None else None
    payload, payload_source = _extract_payload_bytes(
        source,
        source_manifest_path=source_path,
        payload_path=Path(payload_path) if payload_path is not None else None,
    )
    payload_sha = sha256_bytes(payload)
    _verify_expected_payload_custody(source, payload=payload, payload_sha256=payload_sha)

    roundtrip = _roundtrip_status(source)
    if not roundtrip["decode_reconstructs"]:
        raise CodecOpBitstreamMaterializerError(
            "CodecOp decode/roundtrip did not reconstruct the source tensors: "
            f"status={roundtrip['status']}"
        )

    deterministic_params = _deterministic_params(
        source,
        candidate_id=candidate_id,
        source_manifest_path=source_path,
    )
    archive_identity = _archive_identity(source)
    evidence = _evidence_summary(source)
    blockers = _fail_closed_blockers(
        archive_identity=archive_identity,
        evidence=evidence,
        source=source,
    )
    archive_substitution_blockers = _archive_substitution_blockers(
        archive_identity=archive_identity,
        evidence=evidence,
    )

    header = {
        "schema_version": HEADER_SCHEMA_VERSION,
        "codec_magic": BITSTREAM_MAGIC_TEXT,
        "candidate_id": deterministic_params["candidate_id"],
        "payload_bytes": len(payload),
        "payload_sha256": payload_sha,
        "payload_source": payload_source,
        "deterministic_params": deterministic_params,
        "roundtrip_status": roundtrip["status"],
    }
    header_bytes = _canonical_json_bytes(header)
    materialized = (
        BITSTREAM_MAGIC
        + struct.pack(HEADER_LENGTH_FORMAT, len(header_bytes))
        + header_bytes
        + payload
    )
    parsed = parse_materialized_codec_op_bitstream(materialized)
    if parsed["payload"] != payload or parsed["header"] != header:
        raise CodecOpBitstreamMaterializerError(
            "internal materialized bitstream roundtrip check failed"
        )

    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(materialized)
    materialized_sha = sha256_bytes(materialized)
    header_sha = sha256_bytes(header_bytes)
    payload_offset = HEADER_PREFIX_BYTES + len(header_bytes)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.codec_op_bitstream_materializer.materialize_codec_op_bitstream",
        "candidate_id": deterministic_params["candidate_id"],
        "score_claim": False,
        "promotion_eligible": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "materialized_charged_byte_artifact": True,
        "ready_for_archive_substitution": not archive_substitution_blockers,
        "evidence_grade": "byte_materialization_planning",
        "evidence_semantics": "codec_op_materialized_bitstream_golden_vector",
        "codec_magic": BITSTREAM_MAGIC_TEXT,
        "bitstream_format": {
            "magic": BITSTREAM_MAGIC_TEXT,
            "header_schema_version": HEADER_SCHEMA_VERSION,
            "header_length_format": HEADER_LENGTH_FORMAT,
            "header_length_bytes": HEADER_LENGTH_BYTES,
            "payload_offset": payload_offset,
            "canonical_header_json": True,
        },
        "charged_byte_blob": {
            "path": blob_path.as_posix(),
            "bytes": len(materialized),
            "sha256": materialized_sha,
            "header_bytes": len(header_bytes),
            "header_sha256": header_sha,
            "payload_offset": payload_offset,
            "payload_bytes": len(payload),
            "payload_sha256": payload_sha,
        },
        "source_payload": {
            "source": payload_source,
            "bytes": len(payload),
            "sha256": payload_sha,
        },
        "deterministic_params": deterministic_params,
        "roundtrip": roundtrip,
        "archive_identity": archive_identity,
        "source_evidence": evidence,
        "golden_vector": {
            "codec_magic": BITSTREAM_MAGIC_TEXT,
            "blob_bytes": len(materialized),
            "blob_sha256": materialized_sha,
            "header_bytes": len(header_bytes),
            "header_sha256": header_sha,
            "payload_offset": payload_offset,
            "payload_bytes": len(payload),
            "payload_sha256": payload_sha,
            "roundtrip_status": roundtrip["status"],
        },
        "archive_substitution_blockers": archive_substitution_blockers,
        "blockers": blockers,
        "remaining_exact_eval_blockers": sorted(
            {
                *blockers,
                "archive_substitution_candidate_not_built_from_materialized_blob",
                "exact_cuda_auth_eval_not_run_for_materialized_blob",
                "auth_eval_adjudication_not_supplied",
            }
        ),
        "notes": (
            "Materialized CodecOp payload bytes into a deterministic golden-vector "
            "envelope. This is byte custody and planning evidence only until the "
            "blob is consumed by a reviewed archive substitution and exact CUDA "
            "auth eval validates the resulting archive."
        ),
    }
    if source_path is not None:
        manifest["source_manifest"] = {
            "path": source_path.as_posix(),
            "bytes": source_path.stat().st_size if source_path.is_file() else None,
            "sha256": sha256_file(source_path) if source_path.is_file() else None,
        }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json_text(manifest), encoding="utf-8")
    return manifest


def parse_materialized_codec_op_bitstream(blob: bytes | bytearray) -> dict[str, Any]:
    """Parse and verify a materialized CodecOp bitstream envelope."""

    data = bytes(blob)
    if len(data) < HEADER_PREFIX_BYTES:
        raise CodecOpBitstreamMaterializerError("bitstream is shorter than header prefix")
    if data[: len(BITSTREAM_MAGIC)] != BITSTREAM_MAGIC:
        raise CodecOpBitstreamMaterializerError(
            f"bad codec magic: {data[: len(BITSTREAM_MAGIC)]!r}"
        )
    header_len = struct.unpack_from(
        HEADER_LENGTH_FORMAT,
        data,
        len(BITSTREAM_MAGIC),
    )[0]
    header_start = HEADER_PREFIX_BYTES
    header_end = header_start + int(header_len)
    if header_len <= 0 or header_end > len(data):
        raise CodecOpBitstreamMaterializerError("header length is out of range")
    try:
        header = json.loads(data[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodecOpBitstreamMaterializerError(f"invalid header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise CodecOpBitstreamMaterializerError("header JSON must be an object")
    if header.get("schema_version") != HEADER_SCHEMA_VERSION:
        raise CodecOpBitstreamMaterializerError(
            f"unsupported header schema: {header.get('schema_version')!r}"
        )
    if header.get("codec_magic") != BITSTREAM_MAGIC_TEXT:
        raise CodecOpBitstreamMaterializerError("header codec magic mismatch")
    payload = data[header_end:]
    expected_bytes = _optional_nonnegative_int(header.get("payload_bytes"))
    if expected_bytes is None or expected_bytes != len(payload):
        raise CodecOpBitstreamMaterializerError(
            "payload byte count mismatch in materialized bitstream"
        )
    expected_sha = _optional_sha256(header.get("payload_sha256"), "header payload_sha256")
    if expected_sha is None or expected_sha != sha256_bytes(payload):
        raise CodecOpBitstreamMaterializerError("payload SHA-256 mismatch in bitstream")
    return {
        "header": header,
        "payload": payload,
        "payload_offset": header_end,
        "header_bytes": int(header_len),
        "blob_bytes": len(data),
        "blob_sha256": sha256_bytes(data),
    }


def read_materialized_codec_op_bitstream(path: Path | str) -> dict[str, Any]:
    """Read and parse a materialized CodecOp bitstream file."""

    return parse_materialized_codec_op_bitstream(Path(path).read_bytes())


def _ensure_write_target(path: Path, *, force: bool, label: str) -> None:
    if path.exists() and not force:
        raise CodecOpBitstreamMaterializerError(
            f"{label} already exists; pass force=True to overwrite: {path}"
        )


def _extract_payload_bytes(
    source: Mapping[str, Any],
    *,
    source_manifest_path: Path | None,
    payload_path: Path | None,
) -> tuple[bytes, dict[str, Any]]:
    if payload_path is not None:
        return _read_payload_path(payload_path), {
            "kind": "explicit_payload_path",
            "path": payload_path.as_posix(),
        }

    for key in ("payload", "blob", "replacement_substream"):
        value = source.get(key)
        if isinstance(value, bytes | bytearray):
            return bytes(value), {"kind": f"inline_bytes:{key}"}

    for key in ("payload_base64", "blob_base64", "candidate_substream_base64"):
        value = source.get(key)
        if isinstance(value, str):
            try:
                return base64.b64decode(value.encode("ascii"), validate=True), {
                    "kind": f"inline_base64:{key}"
                }
            except (binascii.Error, UnicodeEncodeError) as exc:
                raise CodecOpBitstreamMaterializerError(
                    f"{key} is not valid base64 payload bytes"
                ) from exc

    for key in ("payload_hex", "blob_hex", "candidate_substream_hex"):
        value = source.get(key)
        if isinstance(value, str):
            try:
                return bytes.fromhex(value), {"kind": f"inline_hex:{key}"}
            except ValueError as exc:
                raise CodecOpBitstreamMaterializerError(
                    f"{key} is not valid hex payload bytes"
                ) from exc

    base_dir = source_manifest_path.parent if source_manifest_path is not None else Path.cwd()
    for key in (
        "materialized_payload_path",
        "replacement_payload_path",
        "payload_path",
        "blob_path",
        "candidate_substream_path",
        "replacement_substream_path",
        "replacement_substream",
    ):
        value = source.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                path = base_dir / path
            return _read_payload_path(path), {
                "kind": f"manifest_payload_path:{key}",
                "path": path.as_posix(),
            }

    raise CodecOpBitstreamMaterializerError(
        "payload bytes are missing; provide payload/blob bytes, base64/hex, "
        "a payload path in the manifest, or --payload"
    )


def _read_payload_path(path: Path) -> bytes:
    if not path.is_file():
        raise CodecOpBitstreamMaterializerError(f"payload path does not exist: {path}")
    payload = path.read_bytes()
    if not payload:
        raise CodecOpBitstreamMaterializerError(f"payload path is empty: {path}")
    return payload


def _verify_expected_payload_custody(
    source: Mapping[str, Any],
    *,
    payload: bytes,
    payload_sha256: str,
) -> None:
    expected_bytes = _first_nonnegative_int(
        source,
        (
            "payload_bytes",
            "blob_bytes",
            "bytes_out",
            "materialized_payload_bytes",
            "replacement_payload_bytes",
            "candidate_substream_bytes",
            "replacement_bytes",
            "expected_payload_bytes",
        ),
    )
    if expected_bytes is not None and expected_bytes != len(payload):
        raise CodecOpBitstreamMaterializerError(
            f"payload byte count mismatch: expected={expected_bytes} actual={len(payload)}"
        )
    expected_sha = _first_sha256(
        source,
        (
            "payload_sha256",
            "blob_sha256",
            "materialized_payload_sha256",
            "replacement_payload_sha256",
            "candidate_substream_sha256",
            "replacement_sha256",
            "expected_payload_sha256",
        ),
    )
    if expected_sha is not None and expected_sha != payload_sha256:
        raise CodecOpBitstreamMaterializerError(
            f"payload SHA-256 mismatch: expected={expected_sha} actual={payload_sha256}"
        )


def _roundtrip_status(source: Mapping[str, Any]) -> dict[str, Any]:
    raw_status = source.get("roundtrip_status")
    if isinstance(raw_status, str):
        normalized = raw_status.strip().lower()
        return {
            "status": normalized,
            "decode_reconstructs": normalized in {"passed", "pass", "ok", "full"},
            "source": "roundtrip_status",
        }
    for key in ("roundtrip_ok", "decode_reconstructs", "full_decode_ok"):
        value = source.get(key)
        if isinstance(value, bool):
            return {
                "status": "passed" if value else "failed",
                "decode_reconstructs": value,
                "source": key,
            }

    decode_validation = source.get("decode_validation")
    if isinstance(decode_validation, Mapping):
        status = str(
            decode_validation.get("decode_coverage_status")
            or source.get("decode_coverage_status")
            or "unknown"
        ).lower()
        expected = _optional_nonnegative_int(decode_validation.get("expected_tensor_count"))
        matched = _optional_nonnegative_int(decode_validation.get("matched_tensor_count"))
        failure_lists = (
            "missing_tensor_keys",
            "non_tensor_decoded_keys",
            "shape_mismatch_tensor_keys",
            "dtype_mismatch_tensor_keys",
        )
        clean = all(not decode_validation.get(key) for key in failure_lists)
        passed = bool(status == "full" and expected and matched == expected and clean)
        return {
            "status": "passed" if passed else status,
            "decode_reconstructs": passed,
            "source": "decode_validation",
            "expected_tensor_count": expected,
            "matched_tensor_count": matched,
            "failure_keys": {
                key: list(decode_validation.get(key) or [])
                for key in failure_lists
            },
        }

    status = str(source.get("decode_coverage_status") or "").lower()
    expected = _optional_nonnegative_int(source.get("expected_tensor_count"))
    matched = _optional_nonnegative_int(source.get("matched_tensor_count"))
    if status or expected is not None or matched is not None:
        clean = all(
            not source.get(key)
            for key in (
                "missing_tensor_keys",
                "non_tensor_decoded_keys",
                "shape_mismatch_tensor_keys",
                "dtype_mismatch_tensor_keys",
            )
        )
        passed = bool(status == "full" and expected and matched == expected and clean)
        return {
            "status": "passed" if passed else status or "failed",
            "decode_reconstructs": passed,
            "source": "decode_coverage_fields",
            "expected_tensor_count": expected,
            "matched_tensor_count": matched,
        }

    return {
        "status": "absent",
        "decode_reconstructs": False,
        "source": "missing_roundtrip_proof",
    }


def _deterministic_params(
    source: Mapping[str, Any],
    *,
    candidate_id: str | None,
    source_manifest_path: Path | None,
) -> dict[str, Any]:
    resolved_candidate_id = candidate_id or _string_or_none(source.get("candidate_id"))
    if not resolved_candidate_id:
        op_name = _string_or_none(source.get("op_name")) or _string_or_none(source.get("op_class"))
        resolved_candidate_id = op_name or "codec_op_materialized_bitstream"
    params = {
        "candidate_id": resolved_candidate_id,
        "op_module": _string_or_none(source.get("op_module")),
        "op_class": _string_or_none(source.get("op_class")),
        "op_name": _string_or_none(source.get("op_name")),
        "stream_name": _string_or_none(source.get("stream_name")),
        "source_label": _string_or_none(source.get("source_label")),
        "op_params": _jsonable(source.get("op_params", {})),
        "context_keys": _jsonable(source.get("context_keys", [])),
        "tensor_contract_sha256": _string_or_none(source.get("tensor_contract_sha256")),
        "op_state_sha256": _string_or_none(source.get("op_state_sha256")),
        "source_row_sha256": _sha256_json(_scrub_source_for_hash(source)),
    }
    if source_manifest_path is not None:
        params["source_manifest_path"] = source_manifest_path.as_posix()
        if source_manifest_path.is_file():
            params["source_manifest_sha256"] = sha256_file(source_manifest_path)
    return _jsonable(params)


def _archive_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    archive = source.get("archive")
    nested = archive if isinstance(archive, Mapping) else {}
    sha = (
        _optional_sha256(source.get("archive_sha256"), "archive_sha256")
        or _optional_sha256(nested.get("sha256"), "archive.sha256")
        or _optional_sha256(nested.get("old_archive_sha256"), "archive.old_archive_sha256")
        or _optional_sha256(nested.get("source_sha256"), "archive.source_sha256")
    )
    bytes_len = (
        _optional_nonnegative_int(source.get("archive_bytes"))
        or _optional_nonnegative_int(nested.get("bytes"))
        or _optional_nonnegative_int(nested.get("old_archive_bytes"))
        or _optional_nonnegative_int(nested.get("source_bytes"))
    )
    path = (
        _string_or_none(source.get("archive_path"))
        or _string_or_none(nested.get("path"))
        or _string_or_none(nested.get("source_path"))
    )
    present = sha is not None and bytes_len is not None
    return {
        "present": present,
        "path": path,
        "sha256": sha,
        "bytes": bytes_len,
        "status": "present" if present else "absent",
    }


def _evidence_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    evidence_strings = _evidence_strings(source)
    cpu_only = any("cpu" in item.lower() for item in evidence_strings)
    exact_cuda = bool(source.get("exact_cuda_auth_eval") is True)
    return {
        "evidence_semantics": _jsonable(source.get("evidence_semantics")),
        "evidence_grade": _jsonable(source.get("evidence_grade")),
        "target_modes": _jsonable(source.get("target_modes", [])),
        "deployment_target": _string_or_none(source.get("deployment_target")),
        "exact_cuda_auth_eval": exact_cuda,
        "cpu_only": cpu_only,
        "source_score_claim": bool(source.get("score_claim") is True),
        "source_ready_for_exact_eval_dispatch": bool(
            source.get("ready_for_exact_eval_dispatch") is True
        ),
    }


def _fail_closed_blockers(
    *,
    archive_identity: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source: Mapping[str, Any],
) -> list[str]:
    blockers: set[str] = set(DEFAULT_SCORE_BLOCKERS)
    if evidence.get("cpu_only") is True:
        blockers.add("cpu_only_evidence_not_score_or_dispatch_evidence")
    if evidence.get("exact_cuda_auth_eval") is not True:
        blockers.add("missing_exact_cuda_auth_eval")
    if archive_identity.get("present") is not True:
        blockers.add("archive_identity_absent")
        blockers.add("source_archive_sha256_and_bytes_required_before_substitution")
    if source.get("score_claim") is True:
        blockers.add("source_score_claim_ignored_by_materializer")
    if source.get("ready_for_exact_eval_dispatch") is True:
        blockers.add("source_dispatch_readiness_revalidated_by_materializer")
    return sorted(blockers)


def _archive_substitution_blockers(
    *,
    archive_identity: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    blockers: set[str] = set()
    if evidence.get("cpu_only") is True:
        blockers.add("cpu_only_evidence_not_archive_substitution_ready")
    if archive_identity.get("present") is not True:
        blockers.add("source_archive_sha256_and_bytes_required_before_substitution")
    return sorted(blockers)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _jsonable(payload),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _scrub_source_for_hash(source: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(source, key=str):
        value = source[key]
        if isinstance(value, bytes | bytearray):
            payload = bytes(value)
            out[str(key)] = {
                "kind": "bytes",
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        else:
            out[str(key)] = _jsonable(value)
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CodecOpBitstreamMaterializerError(
                f"non-finite float is not JSON custody-safe: {value!r}"
            )
        return value
    if isinstance(value, bytes | bytearray):
        payload = bytes(value)
        return {"bytes": len(payload), "sha256": sha256_bytes(payload)}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            if key in out:
                raise CodecOpBitstreamMaterializerError(
                    f"mapping has duplicate JSON key after stringification: {key!r}"
                )
            out[key] = _jsonable(value[raw_key])
        return out
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except Exception as exc:
            raise CodecOpBitstreamMaterializerError(
                f"value of type {type(value).__name__} is not JSON custody-safe"
            ) from exc
    raise CodecOpBitstreamMaterializerError(
        f"value of type {type(value).__name__} is not JSON custody-safe"
    )


def _evidence_strings(source: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in (
        "evidence_semantics",
        "evidence_grade",
        "deployment_target",
        "dispatch_blockers",
        "blockers",
        "target_modes",
    ):
        out.extend(_flatten_strings(source.get(key)))
    return out


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return []


def _first_nonnegative_int(source: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _optional_nonnegative_int(source.get(key))
        if value is not None:
            return value
    return None


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _first_sha256(source: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _optional_sha256(source.get(key), key)
        if value is not None:
            return value
    return None


def _optional_sha256(value: Any, label: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if not SHA256_RE.match(value):
        raise CodecOpBitstreamMaterializerError(f"{label} must be 64 hex chars")
    return value.lower()


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


__all__ = [
    "BITSTREAM_MAGIC",
    "BITSTREAM_MAGIC_TEXT",
    "SCHEMA_VERSION",
    "CodecOpBitstreamMaterializerError",
    "materialize_codec_op_bitstream",
    "parse_materialized_codec_op_bitstream",
    "read_materialized_codec_op_bitstream",
]
