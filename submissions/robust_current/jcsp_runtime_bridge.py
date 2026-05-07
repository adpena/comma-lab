#!/usr/bin/env python3
"""Fail-closed JCSP runtime probe for ``submissions/robust_current``.

This module is intentionally stdlib-only.  It runs from ``inflate.sh`` before
any rendering branch so an archive carrying ``jcsp.bin`` cannot silently fall
through to an unrelated runtime path.  The bridge currently probes and proves
the member shape; it does not decode streams or emit frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

JCSP_RUNTIME_BRIDGE_PROBE_SCHEMA = "jcsp_submission_runtime_bridge_probe_v1"
JCSP_ARCHIVE_MEMBER_NAME = "jcsp.bin"
JCSP_REQUIRED_SUBMISSION_RUNTIME = "submissions/robust_current"
JCSP_RUNTIME_BRIDGE_PATH = "submissions/robust_current/jcsp_runtime_bridge.py"
JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER = (
    "submissions_robust_current_jcsp_bin_consumption_missing"
)
JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER = (
    "jcsp_local_skeleton_not_submission_runtime_container"
)
JCSP_RUNTIME_OUTPUT_CONTRACT_SCHEMA = "jcsp_submission_runtime_output_contract_v1"
JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER = "jcsp_runtime_raw_output_parity_missing"
JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA = (
    "jcsp_runtime_raw_output_parity_contract_v1"
)
JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA = (
    "jcsp_runtime_raw_output_parity_proof_v1"
)
JCSP_MAGIC = b"JCSP"
JCSK_MAGIC = b"JCSK"
JCSP_VERSION = 1
JCSK_VERSION = 1
KIND_ARITHMETIC_STATIC = 0
KIND_BALLE_HYPERPRIOR = 1
KIND_RAW_PASSTHROUGH = 2
EXIT_JCSP_MEMBER_REFUSED = 44

_PAYLOAD_MAGICS_BY_CODEC_KIND: dict[int, tuple[bytes, ...]] = {
    KIND_ARITHMETIC_STATIC: (b"AQv1", b"AQc1"),
    KIND_BALLE_HYPERPRIOR: (b"BHv1",),
}


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_manifest_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("manifest_sha256", None)
    out["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(out))
    return out


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(manifest) + b"\n")


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items))


def _raw_output_for_video_name(video_name: str) -> str:
    text = str(video_name).strip()
    if not text:
        raise ValueError("empty video name")
    if "\x00" in text or text.startswith("/"):
        raise ValueError(f"unsafe video name {text!r}")
    parts = Path(text).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe video name {text!r}")
    stem = text.rsplit(".", 1)[0] if "." in text else text
    return f"{stem}.raw"


def _validate_raw_output_rel_path(raw_output: str) -> str:
    text = str(raw_output).strip()
    if not text:
        raise ValueError("empty raw output path")
    if "\x00" in text or text.startswith("/"):
        raise ValueError(f"unsafe raw output path {text!r}")
    parts = Path(text).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe raw output path {text!r}")
    if not text.endswith(".raw"):
        raise ValueError(f"raw output path must end with .raw: {text!r}")
    return text


def _expected_raw_outputs_from_names_file(
    video_names_file: str | Path | None,
) -> tuple[list[str], str | None]:
    if video_names_file is None:
        return [], None
    try:
        text = Path(video_names_file).read_text(encoding="utf-8")
        outputs = [
            _raw_output_for_video_name(line)
            for line in text.splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [], str(exc)
    return _dedupe(outputs), None


def _observe_existing_raw_outputs(
    inflated_dir: str | Path | None,
    expected_raw_outputs: list[str],
) -> list[dict[str, Any]]:
    if inflated_dir is None:
        return []
    root = Path(inflated_dir)
    rows: list[dict[str, Any]] = []
    for rel in expected_raw_outputs:
        rel = _validate_raw_output_rel_path(rel)
        path = root / rel
        exists = path.exists()
        row: dict[str, Any] = {
            "path": rel,
            "exists": exists,
            "is_file": path.is_file(),
            "bytes": None,
            "sha256": None,
            "sha256_status": "not_hashed_pre_dispatch_probe",
            "parity_proof_source": "preexisting_raw_output_unproven",
        }
        if path.is_file():
            try:
                row["bytes"] = int(path.stat().st_size)
            except OSError:
                row["bytes"] = None
        rows.append(row)
    return rows


def _raw_output_parity_contract(
    *,
    expected_raw_outputs: list[str],
    existing_raw_output_count: int,
    names_error: str | None,
) -> dict[str, Any]:
    expected_known = names_error is None and bool(expected_raw_outputs)
    return {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA,
        "required_proof_schema": JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "expected_raw_outputs_known": expected_known,
        "expected_raw_output_count": len(expected_raw_outputs),
        "expected_raw_outputs_sha256": _sha256_bytes(
            _canonical_json_bytes({"expected_raw_outputs": expected_raw_outputs})
        ),
        "required_candidate_output_source": "jcsp_runtime_bridge_emitted_rawvideo",
        "required_reference_output_source": (
            "contest_reference_runtime_or_byte_custody_baseline"
        ),
        "preexisting_raw_outputs_are_not_parity_proof": True,
        "existing_raw_output_count_at_probe": int(existing_raw_output_count),
        "required_per_output_fields": [
            "path",
            "candidate_exists",
            "candidate_bytes",
            "candidate_sha256",
            "reference_exists",
            "reference_bytes",
            "reference_sha256",
            "byte_exact_match",
        ],
        "acceptance_conditions": [
            "jcsp_stream_consumer_decodes_jcsp_streams",
            "bridge_emits_exactly_expected_raw_outputs",
            "candidate_outputs_are_from_current_bridge_run",
            "reference_outputs_are_from_contest_runtime_or_custody_baseline",
            "all_candidate_sha256_values_match_reference_sha256_values",
            "parity_proof_manifest_uses_required_schema",
        ],
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "dispatch_blocker": JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
    }


def _raw_output_file_identity(
    root: str | Path,
    rel: str,
    *,
    role: str,
) -> dict[str, Any]:
    path = Path(root) / rel
    exists = path.exists()
    is_file = path.is_file()
    row: dict[str, Any] = {
        f"{role}_exists": exists,
        f"{role}_is_file": is_file,
        f"{role}_bytes": None,
        f"{role}_sha256": None,
    }
    if is_file:
        row[f"{role}_bytes"] = int(path.stat().st_size)
        row[f"{role}_sha256"] = _sha256_file(path)
    return row


def prove_jcsp_runtime_raw_output_parity(
    expected_raw_outputs: Iterable[str],
    *,
    candidate_raw_dir: str | Path,
    reference_raw_dir: str | Path,
    candidate_outputs_emitted_by_bridge: bool,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic byte-exact raw-output parity proof.

    This helper is intentionally independent from the current inflate probe.
    The current bridge does not emit raw outputs, so ``probe_jcsp_runtime_bridge``
    still refuses every present ``jcsp.bin``.  The future JCSP stream consumer
    can call this only after it writes the contest ``.raw`` files for the
    current run.
    """

    expected = _dedupe(
        [_validate_raw_output_rel_path(str(item)) for item in expected_raw_outputs]
    )
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for rel in expected:
        row = {"path": rel}
        row.update(_raw_output_file_identity(candidate_raw_dir, rel, role="candidate"))
        row.update(_raw_output_file_identity(reference_raw_dir, rel, role="reference"))
        row["byte_exact_match"] = bool(
            row["candidate_is_file"]
            and row["reference_is_file"]
            and row["candidate_bytes"] == row["reference_bytes"]
            and row["candidate_sha256"] == row["reference_sha256"]
        )
        if not row["candidate_is_file"]:
            blockers.append("jcsp_candidate_raw_output_missing")
        if not row["reference_is_file"]:
            blockers.append("jcsp_reference_raw_output_missing")
        if row["candidate_is_file"] and row["reference_is_file"] and not row[
            "byte_exact_match"
        ]:
            blockers.append("jcsp_raw_output_sha256_mismatch")
        rows.append(row)

    if not expected:
        blockers.append("jcsp_expected_raw_outputs_missing")
    if not candidate_outputs_emitted_by_bridge:
        blockers.append("jcsp_candidate_outputs_not_emitted_by_bridge")

    all_candidate_present = bool(expected) and all(
        row["candidate_is_file"] for row in rows
    )
    all_reference_present = bool(expected) and all(
        row["reference_is_file"] for row in rows
    )
    byte_exact = bool(expected) and all(row["byte_exact_match"] for row in rows)
    ready_for_output_parity = bool(
        candidate_outputs_emitted_by_bridge
        and all_candidate_present
        and all_reference_present
        and byte_exact
    )
    manifest: dict[str, Any] = {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_outputs_emitted_by_bridge": bool(
            candidate_outputs_emitted_by_bridge
        ),
        "candidate_output_source": (
            "jcsp_runtime_bridge_emitted_rawvideo"
            if candidate_outputs_emitted_by_bridge
            else "preexisting_or_unproven_raw_files"
        ),
        "reference_output_source": "contest_reference_runtime_or_byte_custody_baseline",
        "candidate_raw_dir": str(Path(candidate_raw_dir)),
        "reference_raw_dir": str(Path(reference_raw_dir)),
        "expected_raw_outputs": expected,
        "expected_raw_output_count": len(expected),
        "output_count": len(rows),
        "all_candidate_outputs_present": all_candidate_present,
        "all_reference_outputs_present": all_reference_present,
        "byte_exact_raw_output_parity": byte_exact,
        "ready_for_output_parity": ready_for_output_parity,
        "ready_for_submission_runtime_consumption": ready_for_output_parity,
        "ready_for_exact_eval_dispatch": False,
        "outputs": rows,
        "dispatch_blockers": (
            []
            if ready_for_output_parity
            else _dedupe([JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER, *blockers])
        ),
    }
    manifest = _with_manifest_sha256(manifest)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def _contest_output_contract(
    *,
    member_present: bool,
    inflated_dir: str | Path | None,
    video_names_file: str | Path | None,
) -> dict[str, Any]:
    expected_raw_outputs, names_error = _expected_raw_outputs_from_names_file(
        video_names_file,
    )
    observed = _observe_existing_raw_outputs(inflated_dir, expected_raw_outputs)
    existing_count = sum(1 for row in observed if row["exists"])
    parity_contract = _raw_output_parity_contract(
        expected_raw_outputs=expected_raw_outputs,
        existing_raw_output_count=existing_count,
        names_error=names_error,
    )
    blockers: list[str] = []
    if member_present:
        blockers.extend(
            [
                JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
                "jcsp_stream_decode_emit_frames_missing",
                "jcsp_raw_output_emission_missing",
            ]
        )
        if names_error is not None:
            blockers.append("jcsp_video_names_file_parse_failed")
        if existing_count:
            blockers.append("jcsp_existing_raw_outputs_unproven")
    return {
        "schema": JCSP_RUNTIME_OUTPUT_CONTRACT_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "required_when_jcsp_member_present": member_present,
        "contest_output_format": "one uint8 RGB rawvideo .raw per names_file entry",
        "video_names_file_observed": video_names_file is not None,
        "video_names_file_parse_error": names_error,
        "inflated_dir_observed": inflated_dir is not None,
        "expected_raw_outputs": expected_raw_outputs,
        "expected_raw_output_count": len(expected_raw_outputs),
        "existing_raw_outputs_at_probe": observed,
        "existing_raw_output_count": existing_count,
        "raw_output_parity_contract": parity_contract,
        "required_raw_output_parity_proof_schema": (
            JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA
        ),
        "bridge_emits_contest_raw_outputs": False,
        "raw_output_emission_attempted": False,
        "output_parity_checked": False,
        "output_parity_artifact": None,
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "dispatch_blocker": JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER,
        "dispatch_blockers": _dedupe(blockers),
    }


def _require_available(blob: bytes, cursor: int, n_bytes: int, context: str) -> None:
    if n_bytes < 0 or cursor < 0 or cursor + n_bytes > len(blob):
        raise ValueError(
            f"truncated {context} at offset {cursor}; need {n_bytes} bytes, "
            f"blob len={len(blob)}"
        )


def _payload_magic_for_kind(
    *,
    codec_kind: int,
    payload: bytes,
    stream_name: str,
) -> str:
    if codec_kind == KIND_RAW_PASSTHROUGH:
        if not payload:
            raise ValueError(f"stream {stream_name!r} raw payload is empty")
        return payload[:4].decode("ascii", errors="replace")
    allowed = _PAYLOAD_MAGICS_BY_CODEC_KIND.get(int(codec_kind))
    if allowed is None:
        raise ValueError(f"stream {stream_name!r} has invalid codec_kind {codec_kind}")
    if len(payload) < 4:
        raise ValueError(
            f"stream {stream_name!r} payload is too small for codec magic"
        )
    magic = payload[:4]
    if magic not in allowed:
        allowed_text = ", ".join(repr(item) for item in allowed)
        raise ValueError(
            f"stream {stream_name!r} payload magic {magic!r} is incompatible "
            f"with codec_kind {codec_kind}; expected one of {allowed_text}"
        )
    return magic.decode("ascii", errors="replace")


def _parse_real_jcsp_container(blob: bytes) -> dict[str, Any]:
    _require_available(blob, 0, 7, "JCSP header")
    if blob[:4] != JCSP_MAGIC:
        raise ValueError(f"bad JCSP magic {blob[:4]!r}")
    cursor = 4
    (version,) = struct.unpack_from("<H", blob, cursor)
    cursor += 2
    if version != JCSP_VERSION:
        raise ValueError(f"unsupported JCSP version {version}")
    (stream_count,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1

    streams: list[dict[str, Any]] = []
    names: list[str] = []
    for index in range(int(stream_count)):
        _require_available(blob, cursor, 1, f"stream {index} name length")
        (name_len,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        if name_len <= 0:
            raise ValueError(f"stream {index} has empty name")
        _require_available(blob, cursor, name_len, f"stream {index} name")
        name = blob[cursor : cursor + name_len].decode("utf-8", errors="strict")
        cursor += name_len
        if "\x00" in name:
            raise ValueError(f"stream {index} name contains NUL")
        if name in names:
            raise ValueError(f"duplicate stream name {name!r}")
        names.append(name)

        _require_available(blob, cursor, 1, f"stream {name!r} codec kind")
        (codec_kind,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        if codec_kind not in (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
        ):
            raise ValueError(f"stream {name!r} has invalid codec_kind {codec_kind}")

        _require_available(blob, cursor, 4, f"stream {name!r} ADMM target")
        (admm_bytes_target,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} actual bytes")
        (actual_bytes,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} score delta")
        (score_delta_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} marginal")
        (marginal_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        _require_available(blob, cursor, 4, f"stream {name!r} payload length")
        (payload_len,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        if int(actual_bytes) != int(payload_len):
            raise ValueError(
                f"stream {name!r} actual_bytes={actual_bytes} does not match "
                f"payload_len={payload_len}"
            )
        _require_available(blob, cursor, payload_len, f"stream {name!r} payload")
        payload = blob[cursor : cursor + payload_len]
        cursor += payload_len
        payload_magic = _payload_magic_for_kind(
            codec_kind=int(codec_kind),
            payload=payload,
            stream_name=name,
        )
        streams.append(
            {
                "index": index,
                "name": name,
                "codec_kind": int(codec_kind),
                "admm_bytes_target": int(admm_bytes_target),
                "actual_bytes": int(actual_bytes),
                "score_delta_milli": int(score_delta_milli),
                "marginal_milli": int(marginal_milli),
                "payload_magic": payload_magic,
                "payload_sha256": _sha256_bytes(payload),
            }
        )

    _require_available(blob, cursor, 4, "JCSP KKT residual")
    (kkt_residual_milli,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    _require_available(blob, cursor, 4, "JCSP iteration count")
    (iters,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    _require_available(blob, cursor, 1, "JCSP converged flag")
    (converged_raw,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if converged_raw not in (0, 1):
        raise ValueError(f"invalid JCSP converged flag {converged_raw}")
    if cursor != len(blob):
        raise ValueError(
            f"trailing bytes after JCSP container: cursor={cursor}, len={len(blob)}"
        )
    return {
        "container_magic": "JCSP",
        "container_version": int(version),
        "stream_count": int(stream_count),
        "streams": streams,
        "waterline_kkt_residual_milli": int(kkt_residual_milli),
        "iters": int(iters),
        "converged": bool(converged_raw),
        "noop_fixture": int(stream_count) == 0,
    }


def _probe_local_skeleton_container(blob: bytes) -> dict[str, Any]:
    details: dict[str, Any] = {
        "container_magic": "JCSK",
        "refused_preview_member": True,
    }
    if len(blob) < 10:
        details["preview_parse_error"] = "truncated JCSK header"
        return details
    (version,) = struct.unpack_from("<H", blob, 4)
    (body_len,) = struct.unpack_from("<I", blob, 6)
    body_start = 10
    body_end = body_start + int(body_len)
    details["container_version"] = int(version)
    details["declared_body_bytes"] = int(body_len)
    if version != JCSK_VERSION:
        details["preview_parse_error"] = f"unsupported JCSK version {version}"
        return details
    if body_end != len(blob):
        details["preview_parse_error"] = (
            f"JCSK body length mismatch declared={body_len} "
            f"actual={len(blob) - body_start}"
        )
        return details
    try:
        manifest = json.loads(
            blob[body_start:body_end].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_pairs,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        details["preview_parse_error"] = str(exc)
        return details
    if not isinstance(manifest, Mapping):
        details["preview_parse_error"] = "JCSK manifest is not a mapping"
        return details
    details["preview_manifest_schema"] = str(manifest.get("schema", ""))
    details["preview_manifest_sha256"] = str(manifest.get("manifest_sha256", ""))
    try:
        details["preview_stream_count"] = int(manifest.get("stream_count", -1))
    except (TypeError, ValueError):
        details["preview_parse_error"] = "JCSK manifest stream_count is invalid"
    return details


def probe_jcsp_runtime_bridge(
    archive_dir: str | Path,
    *,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    inflated_dir: str | Path | None = None,
    video_names_file: str | Path | None = None,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    """Probe ``archive_dir/member_name`` and return a deterministic contract.

    A present JCSP member is never treated as dispatch-ready by this tranche.
    Real ``JCSP`` bytes are parsed and then refused because no stream consumer
    is wired.  Local ``JCSK`` preview bytes are refused before runtime-loader
    readiness.
    """

    archive_root = Path(archive_dir)
    member_path = archive_root / member_name
    member_exists = member_path.exists()
    output_contract = _contest_output_contract(
        member_present=member_exists,
        inflated_dir=inflated_dir,
        video_names_file=video_names_file,
    )
    base: dict[str, Any] = {
        "schema": JCSP_RUNTIME_BRIDGE_PROBE_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "required_submission_runtime": JCSP_REQUIRED_SUBMISSION_RUNTIME,
        "runtime_bridge_path": JCSP_RUNTIME_BRIDGE_PATH,
        "member_name": member_name,
        "member_present": member_exists,
        "detects_required_member": member_exists,
        "detected_real_jcsp_member": False,
        "refused_preview_member": False,
        "ready_for_runtime_probe": True,
        "ready_for_runtime_loader": False,
        "consumes_required_member": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_action": "no_jcsp_member_present",
        "contest_output_contract": output_contract,
        "dispatch_blockers": [],
    }
    if not member_path.exists():
        manifest = _with_manifest_sha256(base)
        if manifest_json is not None:
            _write_manifest(Path(manifest_json), manifest)
        return manifest
    if not member_path.is_file():
        base.update(
            {
                "runtime_action": "refuse_non_file_jcsp_member_path",
                "refusal_reason": "jcsp member path is not a regular file",
                "dispatch_blockers": [
                    "jcsp_member_path_not_regular_file",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    *output_contract["dispatch_blockers"],
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
        manifest = _with_manifest_sha256(base)
        if manifest_json is not None:
            _write_manifest(Path(manifest_json), manifest)
        return manifest

    blob = member_path.read_bytes()
    base.update(
        {
            "member_bytes": len(blob),
            "member_sha256": _sha256_bytes(blob),
            "member_prefix_hex": blob[:16].hex(),
        }
    )
    if len(blob) < 4:
        base.update(
            {
                "runtime_action": "refuse_invalid_jcsp_member",
                "refusal_reason": "jcsp member is too small for magic",
                "dispatch_blockers": [
                    "jcsp_member_too_small_for_magic",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    *output_contract["dispatch_blockers"],
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
    elif blob[:4] == JCSK_MAGIC:
        details = _probe_local_skeleton_container(blob)
        base.update(details)
        base.update(
            {
                "runtime_action": "refuse_jcsk_preview_member",
                "refusal_reason": (
                    "jcsp.bin contains local JCSK preview bytes, not the "
                    "runtime JCSP container"
                ),
                "dispatch_blockers": [
                    JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    *output_contract["dispatch_blockers"],
                    "strict_preflight_proof_missing",
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )
    elif blob[:4] == JCSP_MAGIC:
        try:
            parsed = _parse_real_jcsp_container(blob)
        except ValueError as exc:
            base.update(
                {
                    "container_magic": "JCSP",
                    "runtime_action": "refuse_invalid_jcsp_container",
                    "refusal_reason": str(exc),
                    "dispatch_blockers": [
                        "jcsp_runtime_probe_parse_failed",
                        JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                        *output_contract["dispatch_blockers"],
                        "exact_cuda_auth_eval_missing",
                    ],
                }
            )
        else:
            base.update(parsed)
            base.update(
                {
                    "detected_real_jcsp_member": True,
                    "ready_for_runtime_loader": True,
                    "runtime_action": "refuse_until_jcsp_stream_consumer_implemented",
                    "refusal_reason": (
                        "real JCSP container parsed, but robust_current does "
                        "not decode JCSP streams or emit frames from them yet"
                    ),
                    "dispatch_blockers": [
                        JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                        "not_integrated_into_submission_inflate_path",
                        *output_contract["dispatch_blockers"],
                        "jcsp_stream_decode_emit_frames_missing",
                        "exact_cuda_auth_eval_missing",
                    ],
                }
            )
    else:
        base.update(
            {
                "container_magic": blob[:4].decode("ascii", errors="replace"),
                "runtime_action": "refuse_unknown_jcsp_member_magic",
                "refusal_reason": (
                    f"unknown jcsp.bin magic {blob[:4]!r}; expected "
                    f"{JCSP_MAGIC!r} or refused preview {JCSK_MAGIC!r}"
                ),
                "dispatch_blockers": [
                    "jcsp_unknown_member_magic",
                    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                    *output_contract["dispatch_blockers"],
                    "exact_cuda_auth_eval_missing",
                ],
            }
        )

    base["dispatch_blockers"] = _dedupe(base["dispatch_blockers"])
    manifest = _with_manifest_sha256(base)
    if manifest_json is not None:
        _write_manifest(Path(manifest_json), manifest)
    return manifest


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe robust_current JCSP runtime bridge state"
    )
    parser.add_argument("archive_dir", help="inflater archive directory")
    parser.add_argument(
        "--member-name",
        default=JCSP_ARCHIVE_MEMBER_NAME,
        help="JCSP member filename inside archive_dir",
    )
    parser.add_argument(
        "--manifest-json",
        required=True,
        help="path to write deterministic probe manifest JSON",
    )
    parser.add_argument(
        "--inflated-dir",
        default=None,
        help="inflated output directory used to inspect pre-existing raw outputs",
    )
    parser.add_argument(
        "--video-names-file",
        default=None,
        help="contest video names file used to derive required .raw outputs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest = probe_jcsp_runtime_bridge(
        args.archive_dir,
        member_name=args.member_name,
        inflated_dir=args.inflated_dir,
        video_names_file=args.video_names_file,
        manifest_json=args.manifest_json,
    )
    if not manifest["member_present"]:
        return 0
    print(
        "[jcsp-runtime-bridge] wrote deterministic probe manifest: "
        f"{args.manifest_json}",
        file=sys.stderr,
    )
    print(
        "[jcsp-runtime-bridge] FATAL: "
        f"{manifest.get('refusal_reason', 'jcsp member refused')}",
        file=sys.stderr,
    )
    return EXIT_JCSP_MEMBER_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
