#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build PR85 QMA9-native grammar byte candidates or a fail-closed proof.

This is a local-only builder for runtime-supported QMA9 reductions.  It does
not modify runtime files, run a scorer, dispatch GPU work, or make a score
claim.  Candidate archives are written only after the decoded token tensor hash
matches the PR85 token source and the deterministic archive is byte-positive
against the PR85 anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import pack_pr85_bundle, parse_pr85_bundle  # noqa: E402
from tac.qma9_range_mask_contract import (  # noqa: E402
    QMA9_HEADER_BYTES,
    QMA9_MAGIC,
    QMA9ContractError,
    decode_qma9_mask,
    parse_qma9_header,
)


TOOL = "experiments/build_pr85_qma9_native_grammar_candidates.py"
SCHEMA = "pr85_qma9_native_grammar_candidates_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_TOKEN_SOURCE = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/"
    "pr85_qma9_tokens_u8_storage_order.bin"
)
DEFAULT_CPP_DECODER = REPO_ROOT / "submissions/robust_current/range_mask_codec.cpp"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_qma9_native_grammar_candidates_20260504"
DEFAULT_RUN_GRAMMAR_SUMMARY = (
    REPO_ROOT / "experiments/results/pr85_qma9_run_grammar_candidates_20260504_worker/candidate_summary.json"
)
DEFAULT_ALT_GRAMMAR_SUMMARY = (
    REPO_ROOT / "experiments/results/pr85_qma9_alt_grammar_candidates_20260504/candidate_summary.json"
)
DEFAULT_MODE_SWEEP_SUMMARY = (
    REPO_ROOT / "experiments/results/pr85_qma9_mode_sweep_20260504_codex/pr85_qma9_cpp_mode_sweep_summary.json"
)
DEFAULT_MACRO_PRIOR_DIR = REPO_ROOT / "experiments/results/pr85_qma9_macro_prior_screen_20260504_codex"

EXPECTED_PR85_ARCHIVE_BYTES = 236_328
EXPECTED_PR85_ARCHIVE_SHA256 = (
    "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
)
EXPECTED_PR85_TOKEN_SHA256 = (
    "c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a"
)
EXPECTED_PR85_TOKEN_BYTES = 117_964_800
ORIGINAL_VIDEO_BYTES = 37_545_489
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class NativeGrammarCandidateError(RuntimeError):
    """Raised when the native grammar screen cannot proceed safely."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise NativeGrammarCandidateError(f"expected JSON object in {path}")
    return payload


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_single_x_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise NativeGrammarCandidateError(f"source PR85 archive is missing: {path}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise NativeGrammarCandidateError(
                f"PR85 source archive must contain exactly one file member 'x'; got {names!r}"
            )
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise NativeGrammarCandidateError("PR85 source archive member 'x' must be ZIP_STORED")
        raw = zf.read(info)
    archive_bytes = path.stat().st_size
    return (
        {
            "path": _repo_rel(path),
            "archive_bytes": archive_bytes,
            "archive_sha256": _sha256_file(path),
            "member_name": "x",
            "member_bytes": len(raw),
            "member_sha256": _sha256(raw),
            "zip_overhead_bytes": archive_bytes - len(raw),
        },
        raw,
    )


def _validate_pr85_anchor(
    source_meta: Mapping[str, Any],
    *,
    expected_archive_sha256: str | None,
    expected_archive_bytes: int | None,
) -> None:
    if expected_archive_bytes is not None and int(source_meta["archive_bytes"]) != int(expected_archive_bytes):
        raise NativeGrammarCandidateError(
            f"source archive bytes {source_meta['archive_bytes']} != expected {expected_archive_bytes}"
        )
    if expected_archive_sha256 is not None and str(source_meta["archive_sha256"]) != str(expected_archive_sha256):
        raise NativeGrammarCandidateError(
            f"source archive sha256 {source_meta['archive_sha256']} != expected {expected_archive_sha256}"
        )


def _validate_token_source(
    token_source: Path,
    *,
    expected_token_sha256: str,
    expected_token_bytes: int | None,
) -> dict[str, Any]:
    if not token_source.is_file():
        raise NativeGrammarCandidateError(f"decoded token source is missing: {token_source}")
    token_bytes = token_source.stat().st_size
    token_sha = _sha256_file(token_source)
    if expected_token_bytes is not None and token_bytes != int(expected_token_bytes):
        raise NativeGrammarCandidateError(
            f"decoded token source bytes {token_bytes} != expected {expected_token_bytes}"
        )
    if token_sha != expected_token_sha256:
        raise NativeGrammarCandidateError(
            f"decoded token source sha256 {token_sha} != expected {expected_token_sha256}"
        )
    return {
        "path": _repo_rel(token_source),
        "bytes": token_bytes,
        "sha256": token_sha,
        "expected_sha256": expected_token_sha256,
        "expected_bytes": expected_token_bytes,
    }


def _compile_cpp_decoder(source: Path, work_dir: Path) -> tuple[Path, list[str]]:
    if not source.is_file():
        raise NativeGrammarCandidateError(f"QMA9 runtime decoder source is missing: {source}")
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise NativeGrammarCandidateError("no C++ compiler found on PATH for QMA9 runtime decode parity")
    binary = work_dir / "qma9_runtime_decoder"
    cmd = [compiler, "-O3", "-std=c++17", str(source), "-o", str(binary)]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise NativeGrammarCandidateError(f"QMA9 runtime decoder compile failed: {proc.stderr[-4000:]}")
    return binary, cmd


def _decode_payload_sha256(
    payload: bytes,
    *,
    implementation: str,
    cpp_decoder_source: Path,
    work_dir: Path,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    header = parse_qma9_header(payload)
    if implementation == "python":
        decoded = decode_qma9_mask(payload)
        return decoded.sha256, {
            "implementation": "src/tac.qma9_range_mask_contract.decode_qma9_mask",
            "storage_order": decoded.storage_order,
            "header": asdict(decoded.header),
        }
    if implementation != "cpp":
        raise NativeGrammarCandidateError(f"unknown decode implementation: {implementation!r}")

    binary, compile_cmd = _compile_cpp_decoder(cpp_decoder_source, work_dir)
    qma9_path = work_dir / "candidate.qma9"
    raw_path = work_dir / "candidate_tokens.bin"
    qma9_path.write_bytes(payload)
    run_cmd = [str(binary), "decode", str(qma9_path), str(raw_path)]
    proc = subprocess.run(
        run_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        raise NativeGrammarCandidateError(f"QMA9 runtime decoder failed: {proc.stderr[-4000:]}")
    expected_bytes = header.frame_count * header.width * header.height
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise NativeGrammarCandidateError(
            f"QMA9 runtime decoder wrote {actual_bytes} token bytes, expected {expected_bytes}"
        )
    return _sha256_file(raw_path), {
        "implementation": "submissions/robust_current/range_mask_codec.cpp",
        "source": _repo_rel(cpp_decoder_source),
        "compile_command": [_repo_rel(Path(compile_cmd[0])) or compile_cmd[0], *compile_cmd[1:]],
        "run_returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "storage_order": "frame_major_header_width_by_header_height",
        "header": asdict(header),
    }


def _zip_member_bytes(member_name: str, payload: bytes) -> bytes:
    if member_name != "x":
        raise NativeGrammarCandidateError(f"PR85 candidate ZIP member must be 'x', got {member_name!r}")
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(member_name)
    info.date_time = FIXED_ZIP_TIMESTAMP
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(info, payload)
    return buffer.getvalue()


def _write_candidate_archive(path: Path, x_payload: bytes) -> dict[str, Any]:
    first = _zip_member_bytes("x", x_payload)
    second = _zip_member_bytes("x", x_payload)
    if first != second:
        raise NativeGrammarCandidateError("deterministic ZIP construction produced non-identical bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(first)
    with zipfile.ZipFile(path, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if names != ["x"]:
            raise NativeGrammarCandidateError(f"candidate archive must contain exactly ['x']; got {names!r}")
        stored = zf.read("x")
    if stored != x_payload:
        raise NativeGrammarCandidateError("candidate archive member roundtrip mismatch")
    return {
        "path": _repo_rel(path),
        "bytes": len(first),
        "sha256": _sha256(first),
        "member_name": "x",
        "member_bytes": len(x_payload),
        "member_sha256": _sha256(x_payload),
        "zip_overhead_bytes": len(first) - len(x_payload),
        "zip_storage": "stored",
        "deterministic_rewrite_identical": True,
    }


def _pack_with_candidate_mask(bundle_raw: bytes, candidate_mask: bytes) -> tuple[bytes, str]:
    bundle = parse_pr85_bundle(bundle_raw)
    segments = dict(bundle.segments)
    segments["mask"] = candidate_mask
    header_mode = "explicit_30" if bundle.format == "pr85_explicit_30byte_lengths" else "v5"
    return pack_pr85_bundle(segments, header_mode=header_mode), header_mode


def _qma9_with_bitstream_length(source_mask: bytes, bitstream_bytes: int) -> bytes:
    header = parse_qma9_header(source_mask)
    bitstream_bytes = int(bitstream_bytes)
    if bitstream_bytes <= 0 or bitstream_bytes > header.bitstream_bytes:
        raise NativeGrammarCandidateError(f"invalid QMA9 bitstream trim length: {bitstream_bytes}")
    bitstream = source_mask[QMA9_HEADER_BYTES : QMA9_HEADER_BYTES + bitstream_bytes]
    return struct.pack(
        "<4sIIII",
        QMA9_MAGIC,
        header.frame_count,
        header.width,
        header.height,
        bitstream_bytes,
    ) + bitstream


def _screen_records(source_mask: bytes, *, max_prefix_trim_bytes: int) -> list[dict[str, Any]]:
    header = parse_qma9_header(source_mask)
    bitstream = source_mask[QMA9_HEADER_BYTES : header.packed_bytes]
    trailing_zero_bytes = len(bitstream) - len(bitstream.rstrip(b"\x00"))
    max_prefix_trim_bytes = max(0, int(max_prefix_trim_bytes))
    records: list[dict[str, Any]] = []
    if header.packed_bytes < len(source_mask):
        records.append(
            {
                "candidate_id": "qma9_declared_payload_prefix",
                "screen": "trim_bytes_after_declared_qma9_bitstream",
                "candidate_mask": source_mask[: header.packed_bytes],
                "proposed_mask_delta_bytes": header.packed_bytes - len(source_mask),
                "planning_basis": "QMA9 runtime reads exactly the declared bitstream byte count and ignores no suffix.",
            }
        )
    else:
        records.append(
            {
                "candidate_id": "qma9_declared_payload_prefix",
                "screen": "trim_bytes_after_declared_qma9_bitstream",
                "candidate_mask": None,
                "rejection_reasons": ["source_qma9_segment_has_no_bytes_after_declared_bitstream"],
                "observed_declared_packed_bytes": header.packed_bytes,
                "observed_segment_bytes": len(source_mask),
            }
        )

    if trailing_zero_bytes > 0:
        for trim_bytes in range(trailing_zero_bytes, 0, -1):
            records.append(
                {
                    "candidate_id": f"qma9_declared_bitstream_trim{trim_bytes}",
                    "screen": "trim_trailing_zero_bytes_from_declared_qma9_bitstream",
                    "candidate_mask": _qma9_with_bitstream_length(
                        source_mask,
                        header.bitstream_bytes - trim_bytes,
                    ),
                    "proposed_mask_delta_bytes": -trim_bytes,
                    "trimmed_trailing_zero_bytes": trim_bytes,
                    "planning_basis": "Runtime bit reader pads exhausted input with zero bits; full token parity is still mandatory.",
                }
            )
    else:
        records.append(
            {
                "candidate_id": "qma9_declared_bitstream_trim",
                "screen": "trim_trailing_zero_bytes_from_declared_qma9_bitstream",
                "candidate_mask": None,
                "rejection_reasons": ["source_qma9_declared_bitstream_has_no_trailing_zero_bytes"],
                "observed_trailing_zero_bytes": 0,
            }
        )

    max_decode_proven_trim = min(max_prefix_trim_bytes, header.bitstream_bytes - 1)
    start_trim = trailing_zero_bytes + 1
    if max_decode_proven_trim >= start_trim:
        for trim_bytes in range(start_trim, max_decode_proven_trim + 1):
            records.append(
                {
                    "candidate_id": f"qma9_decode_proven_prefix_trim{trim_bytes}",
                    "screen": "decode_proven_nonzero_suffix_byte_trim",
                    "candidate_mask": _qma9_with_bitstream_length(
                        source_mask,
                        header.bitstream_bytes - trim_bytes,
                    ),
                    "proposed_mask_delta_bytes": -trim_bytes,
                    "trimmed_suffix_bytes": trim_bytes,
                    "planning_basis": (
                        "The runtime bit reader pads exhausted input with zero bits, so every "
                        "nonzero suffix trim must prove full decoded-token SHA parity."
                    ),
                }
            )
    elif max_prefix_trim_bytes > 0:
        records.append(
            {
                "candidate_id": "qma9_decode_proven_prefix_trim",
                "screen": "decode_proven_nonzero_suffix_byte_trim",
                "candidate_mask": None,
                "rejection_reasons": ["source_qma9_bitstream_too_short_for_prefix_trim_screen"],
                "max_prefix_trim_bytes": max_prefix_trim_bytes,
                "observed_bitstream_bytes": header.bitstream_bytes,
            }
        )
    return records


def _summarize_run_grammar_screen(path: Path) -> dict[str, Any]:
    payload = _read_json_if_exists(path)
    if payload is None:
        return {"path": _repo_rel(path), "available": False}
    best = payload.get("best_bytes_vs_pr85_qma9_159011B", {})
    if not isinstance(best, dict):
        best = {}
    best_payload = payload.get("best_payload_candidate", {})
    if not isinstance(best_payload, dict):
        best_payload = {}
    byte_positive = _as_int(payload.get("byte_positive_candidate_count")) or 0
    runtime_positive = _as_int(payload.get("runtime_supported_byte_positive_candidate_count")) or 0
    return {
        "path": _repo_rel(path),
        "available": True,
        "schema": payload.get("schema"),
        "byte_positive_candidate_count": byte_positive,
        "runtime_supported_byte_positive_candidate_count": runtime_positive,
        "best_mode": best.get("best_mode") or best_payload.get("mode"),
        "best_payload_bytes": _as_int(best.get("best_payload_bytes") or best_payload.get("payload_bytes")),
        "best_delta_bytes_vs_pr85_qma9": _as_int(
            best.get("best_delta_bytes") or best_payload.get("delta_bytes_vs_pr85_qma9_159011B")
        ),
        "blockers": list(payload.get("blockers", [])) if isinstance(payload.get("blockers"), list) else [],
        "status": (
            "byte_positive_runtime_supported"
            if runtime_positive > 0
            else "byte_positive_runtime_locked"
            if byte_positive > 0
            else "no_byte_positive_qrg1_row_run_candidate"
        ),
    }


def _summarize_alt_grammar_screen(path: Path) -> dict[str, Any]:
    payload = _read_json_if_exists(path)
    if payload is None:
        return {"path": _repo_rel(path), "available": False}
    fail_closed = payload.get("fail_closed", {})
    if not isinstance(fail_closed, dict):
        fail_closed = {}
    best_alt = payload.get("best_alt_candidate", {})
    if not isinstance(best_alt, dict):
        best_alt = {}
    byte_positive = _as_int(payload.get("byte_positive_candidate_count")) or 0
    runtime_positive = _as_int(payload.get("runtime_supported_byte_positive_candidate_count")) or 0
    return {
        "path": _repo_rel(path),
        "available": True,
        "schema": payload.get("schema"),
        "byte_positive_candidate_count": byte_positive,
        "runtime_supported_byte_positive_candidate_count": runtime_positive,
        "best_mode": fail_closed.get("best_alt_mode") or best_alt.get("mode"),
        "best_payload_bytes": _as_int(fail_closed.get("best_alt_payload_bytes") or best_alt.get("payload_bytes")),
        "best_delta_bytes_vs_source_qma9": _as_int(
            fail_closed.get("best_alt_delta_bytes_vs_source_qma9")
            or best_alt.get("delta_bytes_vs_source_qma9")
        ),
        "fail_closed_reason": fail_closed.get("reason"),
        "status": (
            "byte_positive_runtime_supported"
            if runtime_positive > 0
            else "byte_positive_runtime_locked"
            if byte_positive > 0
            else "no_byte_positive_alt_table_candidate"
        ),
    }


def _summarize_mode_sweep(path: Path) -> dict[str, Any]:
    payload = _read_json_if_exists(path)
    if payload is None:
        return {"path": _repo_rel(path), "available": False}
    modes = payload.get("modes", [])
    if not isinstance(modes, list):
        modes = []
    deltas = [
        _as_int(mode.get("delta_bytes_vs_pr85_qma9"))
        for mode in modes
        if isinstance(mode, dict) and _as_int(mode.get("delta_bytes_vs_pr85_qma9")) is not None
    ]
    decision = payload.get("decision", {})
    if not isinstance(decision, dict):
        decision = {}
    return {
        "path": _repo_rel(path),
        "available": True,
        "schema": payload.get("schema"),
        "mode_count": len(modes),
        "accepted_candidate_count": _as_int(decision.get("accepted_candidate_count")) or 0,
        "best_mode_by_bytes": decision.get("best_mode_by_bytes"),
        "best_runtime_compatible_nonbaseline": decision.get("best_runtime_compatible_nonbaseline"),
        "min_delta_bytes_vs_pr85_qma9": min(deltas) if deltas else None,
        "status": (
            "no_exposed_cpp_mode_beats_adaptive9bin"
            if (_as_int(decision.get("accepted_candidate_count")) or 0) == 0
            else "accepted_cpp_mode_present"
        ),
        "summary": decision.get("summary"),
    }


def _summarize_macro_prior_dir(path: Path, *, source_mask_bytes: int) -> dict[str, Any]:
    if not path.is_dir():
        return {"path": _repo_rel(path), "available": False}
    rows: list[dict[str, Any]] = []
    for payload_path in sorted(path.glob("*.qma9")):
        size = payload_path.stat().st_size
        rows.append(
            {
                "path": _repo_rel(payload_path),
                "bytes": size,
                "sha256": _sha256_file(payload_path),
                "delta_bytes_vs_source_qma9": size - int(source_mask_bytes),
            }
        )
    best = min(rows, key=lambda row: int(row["bytes"])) if rows else None
    return {
        "path": _repo_rel(path),
        "available": True,
        "payload_count": len(rows),
        "best_payload": best,
        "byte_positive_payload_count": sum(
            1 for row in rows if int(row["delta_bytes_vs_source_qma9"]) < 0
        ),
        "status": (
            "no_macro_prior_payload_beats_source_qma9"
            if not any(int(row["delta_bytes_vs_source_qma9"]) < 0 for row in rows)
            else "macro_prior_byte_positive_payload_present_requires_token_parity"
        ),
    }


def _family_resolution(
    *,
    candidates: list[dict[str, Any]],
    source_mask_bytes: int,
    expected_token_sha256: str,
    related_screens: Mapping[str, Any],
    max_prefix_trim_bytes: int,
) -> dict[str, Any]:
    candidate_archives = [
        row.get("archive", {}).get("path")
        for row in candidates
        if isinstance(row.get("archive"), dict) and row.get("archive", {}).get("path")
    ]
    missing = [
        (
            "A full-stream QMA9-compatible encoder that emits fewer than "
            f"{source_mask_bytes} charged mask bytes while the submitted "
            f"range_mask_codec.cpp decoder reproduces token SHA {expected_token_sha256}."
        ),
        (
            "Or, if the winning payload uses a new magic/table/run grammar, a charged "
            "robust_current runtime decoder plus raw-token SHA parity and runtime output "
            "parity before any later exact CUDA eval."
        ),
        (
            "A deterministic archive builder that writes exactly one stored member 'x' "
            "and records archive bytes/SHA after the payload change."
        ),
    ]
    if candidates:
        status = "candidate_archives_built_local_only"
        blocker = None
    else:
        status = "fail_closed_no_byte_positive_runtime_supported_or_screened_run_table_candidate"
        blocker = "no_deterministic_byte_closed_pr85_qma9_native_run_or_table_candidate"
    return {
        "family_id": "qma9_native_run_grammar_or_table_reduction",
        "top_matrix_blocker_converted": True,
        "status": status,
        "blocker": blocker,
        "source_mask_bytes": source_mask_bytes,
        "max_decode_proven_prefix_trim_bytes_screened": int(max_prefix_trim_bytes),
        "candidate_archive_paths": candidate_archives,
        "candidate_count": len(candidates),
        "safe_for_remote_dispatch": False,
        "score_claim": False,
        "dispatch_performed": False,
        "related_screen_status": {
            key: value.get("status")
            for key, value in related_screens.items()
            if isinstance(value, Mapping)
        },
        "minimal_missing_implementation": [] if candidates else missing,
    }


def _reject_record(record: Mapping[str, Any], reasons: list[str]) -> dict[str, Any]:
    out = {key: value for key, value in record.items() if key != "candidate_mask"}
    existing = list(out.get("rejection_reasons", []))
    out["rejection_reasons"] = existing + reasons
    out["archive_written"] = False
    out["safe_for_remote_dispatch"] = False
    return out


def _evaluate_candidate(
    *,
    record: Mapping[str, Any],
    candidate_mask: bytes,
    source_mask: bytes,
    source_bundle_raw: bytes,
    source_meta: Mapping[str, Any],
    expected_token_sha256: str,
    out_dir: Path,
    decode_implementation: str,
    cpp_decoder: Path,
    decode_timeout_seconds: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if candidate_mask == source_mask:
        reasons.append("no_op_or_source_preserving_transform")
    if len(candidate_mask) >= len(source_mask):
        reasons.append("mask_segment_not_byte_positive")
    try:
        header = parse_qma9_header(candidate_mask)
    except QMA9ContractError as exc:
        return _reject_record(record, [f"candidate_qma9_header_invalid:{exc}"])
    if header.packed_bytes != len(candidate_mask):
        reasons.append("candidate_qma9_payload_not_tight_to_declared_bitstream")
    if reasons:
        rejected = _reject_record(record, reasons)
        rejected["candidate_mask_bytes"] = len(candidate_mask)
        rejected["source_mask_bytes"] = len(source_mask)
        return rejected

    with tempfile.TemporaryDirectory(prefix="pr85-qma9-decode-") as tmp:
        decoded_sha, decode_meta = _decode_payload_sha256(
            candidate_mask,
            implementation=decode_implementation,
            cpp_decoder_source=cpp_decoder,
            work_dir=Path(tmp),
            timeout_seconds=decode_timeout_seconds,
        )
    if decoded_sha != expected_token_sha256:
        rejected = _reject_record(record, ["decoded_token_sha_mismatch"])
        rejected["candidate_mask_bytes"] = len(candidate_mask)
        rejected["source_mask_bytes"] = len(source_mask)
        rejected["mask_delta_bytes_vs_source"] = len(candidate_mask) - len(source_mask)
        rejected["decoded_token_sha256"] = decoded_sha
        rejected["expected_token_sha256"] = expected_token_sha256
        rejected["runtime_compatibility"] = {
            "decode_attempted": True,
            "decode_implementation": decode_meta["implementation"],
            "decoded_token_sha_matches_source": False,
        }
        return rejected

    candidate_x, header_mode = _pack_with_candidate_mask(source_bundle_raw, candidate_mask)
    archive_bytes = _zip_member_bytes("x", candidate_x)
    archive_delta = len(archive_bytes) - int(source_meta["archive_bytes"])
    if archive_delta >= 0:
        rejected = _reject_record(record, ["candidate_archive_not_byte_positive_vs_pr85"])
        rejected["candidate_archive_predicted_bytes"] = len(archive_bytes)
        rejected["archive_delta_bytes_vs_pr85"] = archive_delta
        rejected["decoded_token_sha256"] = decoded_sha
        return rejected

    candidate_dir = out_dir / str(record["candidate_id"])
    archive_path = candidate_dir / "archive.zip"
    archive_meta = _write_candidate_archive(archive_path, candidate_x)
    if int(archive_meta["bytes"]) >= int(source_meta["archive_bytes"]):
        archive_path.unlink(missing_ok=True)
        return _reject_record(record, ["candidate_archive_not_byte_positive_vs_pr85_after_write"])

    manifest = {
        "schema": "pr85_qma9_native_grammar_candidate_v1",
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "candidate_id": record["candidate_id"],
        "source_archive": dict(source_meta),
        "archive": archive_meta,
        "archive_delta_bytes_vs_pr85": int(archive_meta["bytes"]) - int(source_meta["archive_bytes"]),
        "rate_score_delta_if_components_identical_formula_only": (
            (int(archive_meta["bytes"]) - int(source_meta["archive_bytes"])) * 25.0 / ORIGINAL_VIDEO_BYTES
        ),
        "qma9": {
            "source_mask_bytes": len(source_mask),
            "candidate_mask_bytes": len(candidate_mask),
            "mask_delta_bytes": len(candidate_mask) - len(source_mask),
            "candidate_header": asdict(header),
            "candidate_mask_sha256": _sha256(candidate_mask),
            "decoded_token_sha256": decoded_sha,
            "decoded_token_sha_matches_pr85_source": True,
            "runtime_decode": decode_meta,
        },
        "pr85_bundle": {
            "header_mode": header_mode,
            "member_bytes": len(candidate_x),
            "member_sha256": _sha256(candidate_x),
        },
        "dispatch_gate": {
            "safe_for_remote_dispatch": False,
            "exact_eval_dispatch_allowed": False,
            "reason": "Task is local-only and explicitly forbids GPU/remote dispatch or score claims.",
        },
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    accepted = {key: value for key, value in record.items() if key != "candidate_mask"}
    accepted.update(
        {
            "archive_written": True,
            "archive": archive_meta,
            "archive_delta_bytes_vs_pr85": int(archive_meta["bytes"]) - int(source_meta["archive_bytes"]),
            "mask_delta_bytes_vs_source": len(candidate_mask) - len(source_mask),
            "candidate_mask_bytes": len(candidate_mask),
            "candidate_mask_sha256": _sha256(candidate_mask),
            "decoded_token_sha256": decoded_sha,
            "runtime_compatibility": {
                "decode_attempted": True,
                "decode_implementation": decode_meta["implementation"],
                "decoded_token_sha_matches_source": True,
                "runtime_supported_qma9": True,
            },
            "safe_for_remote_dispatch": False,
            "dispatch_blocker": "Local-only task; exact eval dispatch is explicitly forbidden.",
        }
    )
    return accepted


def build_native_grammar_candidates(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    token_source: Path = DEFAULT_TOKEN_SOURCE,
    out_dir: Path = DEFAULT_OUT_DIR,
    expected_archive_sha256: str | None = EXPECTED_PR85_ARCHIVE_SHA256,
    expected_archive_bytes: int | None = EXPECTED_PR85_ARCHIVE_BYTES,
    expected_token_sha256: str = EXPECTED_PR85_TOKEN_SHA256,
    expected_token_bytes: int | None = EXPECTED_PR85_TOKEN_BYTES,
    decode_implementation: str = "cpp",
    cpp_decoder: Path = DEFAULT_CPP_DECODER,
    decode_timeout_seconds: int = 300,
    max_prefix_trim_bytes: int = 16,
    run_grammar_summary: Path | None = DEFAULT_RUN_GRAMMAR_SUMMARY,
    alt_grammar_summary: Path | None = DEFAULT_ALT_GRAMMAR_SUMMARY,
    mode_sweep_summary: Path | None = DEFAULT_MODE_SWEEP_SUMMARY,
    macro_prior_dir: Path | None = DEFAULT_MACRO_PRIOR_DIR,
) -> dict[str, Any]:
    """Run the finite QMA9-native screen and write a local summary."""

    source_meta, source_bundle_raw = _read_single_x_archive(archive)
    _validate_pr85_anchor(
        source_meta,
        expected_archive_sha256=expected_archive_sha256,
        expected_archive_bytes=expected_archive_bytes,
    )
    token_meta = _validate_token_source(
        token_source,
        expected_token_sha256=expected_token_sha256,
        expected_token_bytes=expected_token_bytes,
    )
    bundle = parse_pr85_bundle(source_bundle_raw)
    source_mask = bytes(bundle.segments["mask"])
    source_header = parse_qma9_header(source_mask)

    with tempfile.TemporaryDirectory(prefix="pr85-qma9-source-decode-") as tmp:
        source_decoded_sha, source_decode_meta = _decode_payload_sha256(
            source_mask,
            implementation=decode_implementation,
            cpp_decoder_source=cpp_decoder,
            work_dir=Path(tmp),
            timeout_seconds=decode_timeout_seconds,
        )
    if source_decoded_sha != expected_token_sha256:
        raise NativeGrammarCandidateError(
            f"source QMA9 decodes to token sha256 {source_decoded_sha}, expected {expected_token_sha256}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    screen_results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for record in _screen_records(source_mask, max_prefix_trim_bytes=max_prefix_trim_bytes):
        candidate_mask = record.get("candidate_mask")
        if candidate_mask is None:
            result = _reject_record(record, [])
        else:
            result = _evaluate_candidate(
                record=record,
                candidate_mask=bytes(candidate_mask),
                source_mask=source_mask,
                source_bundle_raw=source_bundle_raw,
                source_meta=source_meta,
                expected_token_sha256=expected_token_sha256,
                out_dir=out_dir,
                decode_implementation=decode_implementation,
                cpp_decoder=cpp_decoder,
                decode_timeout_seconds=decode_timeout_seconds,
            )
        screen_results.append(result)
        if result.get("archive_written"):
            candidates.append(result)

    best_candidate = (
        min(candidates, key=lambda row: int(row["archive_delta_bytes_vs_pr85"]))
        if candidates
        else None
    )
    blockers: list[str] = []
    for result in screen_results:
        blockers.extend(str(reason) for reason in result.get("rejection_reasons", []))
    if not candidates:
        blockers.append("no_byte_positive_runtime_supported_qma9_native_grammar_candidate")
    blockers.append("alternate_qma9_grammar_magics_require_runtime_edits_and_are_planning_only")

    related_screens = {
        "qrg1_row_run_grammar": _summarize_run_grammar_screen(run_grammar_summary)
        if run_grammar_summary is not None
        else {"available": False, "reason": "not_requested"},
        "alternate_table_grammar": _summarize_alt_grammar_screen(alt_grammar_summary)
        if alt_grammar_summary is not None
        else {"available": False, "reason": "not_requested"},
        "cpp_mode_sweep": _summarize_mode_sweep(mode_sweep_summary)
        if mode_sweep_summary is not None
        else {"available": False, "reason": "not_requested"},
        "macro_prior_screen": _summarize_macro_prior_dir(macro_prior_dir, source_mask_bytes=len(source_mask))
        if macro_prior_dir is not None
        else {"available": False, "reason": "not_requested"},
    }
    resolution = _family_resolution(
        candidates=candidates,
        source_mask_bytes=len(source_mask),
        expected_token_sha256=expected_token_sha256,
        related_screens=related_screens,
        max_prefix_trim_bytes=max_prefix_trim_bytes,
    )
    if resolution["blocker"]:
        blockers.append(str(resolution["blocker"]))

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "source_archive": source_meta,
        "token_source": token_meta,
        "source_qma9": {
            "segment_bytes": len(source_mask),
            "segment_sha256": _sha256(source_mask),
            "header": asdict(source_header),
            "declared_packed_bytes_equals_segment_bytes": source_header.packed_bytes == len(source_mask),
            "declared_bitstream_trailing_zero_bytes": (
                len(source_mask[QMA9_HEADER_BYTES:source_header.packed_bytes])
                - len(source_mask[QMA9_HEADER_BYTES:source_header.packed_bytes].rstrip(b"\x00"))
            ),
            "source_decode": source_decode_meta,
            "source_decoded_token_sha256": source_decoded_sha,
            "source_decoded_token_sha_matches_expected": True,
        },
        "runtime_supported_grammar": {
            "accepted_magic": "QMA9",
            "header_bytes": QMA9_HEADER_BYTES,
            "charged_model_or_table_bytes_in_qma9_payload": 0,
            "unsupported_without_runtime_edit": [
                "QMB1 vertical block escape",
                "QMF1 first-row specialization",
                "QMH1 horizontal run escape",
                "external context tables",
                "alternate adaptive-model initialization tables",
            ],
        },
        "related_full_stream_screens": related_screens,
        "family_resolution": resolution,
        "screened_variant_count": len(screen_results),
        "candidate_count": len(candidates),
        "best_candidate": best_candidate,
        "best_byte_delta": (
            int(best_candidate["archive_delta_bytes_vs_pr85"]) if best_candidate is not None else None
        ),
        "candidates": candidates,
        "screen_results": screen_results,
        "blockers": sorted(set(blockers)),
        "dispatch_gate": {
            "candidate_archive_exists": bool(candidates),
            "safe_for_remote_dispatch": False,
            "exact_eval_dispatch_allowed": False,
            "reason": "No GPU/remote dispatch or score claim is allowed by this task.",
        },
    }
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--token-source", type=Path, default=DEFAULT_TOKEN_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--expected-archive-sha256", default=EXPECTED_PR85_ARCHIVE_SHA256)
    parser.add_argument("--expected-archive-bytes", type=int, default=EXPECTED_PR85_ARCHIVE_BYTES)
    parser.add_argument("--expected-token-sha256", default=EXPECTED_PR85_TOKEN_SHA256)
    parser.add_argument("--expected-token-bytes", type=int, default=EXPECTED_PR85_TOKEN_BYTES)
    parser.add_argument("--decode-implementation", choices=("cpp", "python"), default="cpp")
    parser.add_argument("--cpp-decoder", type=Path, default=DEFAULT_CPP_DECODER)
    parser.add_argument("--decode-timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--max-prefix-trim-bytes",
        type=int,
        default=16,
        help="Maximum nonzero QMA9 bitstream suffix bytes to remove and prove by full-token decode parity.",
    )
    parser.add_argument("--run-grammar-summary", type=Path, default=DEFAULT_RUN_GRAMMAR_SUMMARY)
    parser.add_argument("--alt-grammar-summary", type=Path, default=DEFAULT_ALT_GRAMMAR_SUMMARY)
    parser.add_argument("--mode-sweep-summary", type=Path, default=DEFAULT_MODE_SWEEP_SUMMARY)
    parser.add_argument("--macro-prior-dir", type=Path, default=DEFAULT_MACRO_PRIOR_DIR)
    parser.add_argument(
        "--allow-source-archive-mismatch",
        action="store_true",
        help="Disable PR85 archive byte/SHA anchor checks for synthetic local tests only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_native_grammar_candidates(
        archive=args.archive,
        token_source=args.token_source,
        out_dir=args.out_dir,
        expected_archive_sha256=None
        if args.allow_source_archive_mismatch
        else args.expected_archive_sha256,
        expected_archive_bytes=None
        if args.allow_source_archive_mismatch
        else args.expected_archive_bytes,
        expected_token_sha256=args.expected_token_sha256,
        expected_token_bytes=args.expected_token_bytes,
        decode_implementation=args.decode_implementation,
        cpp_decoder=args.cpp_decoder,
        decode_timeout_seconds=args.decode_timeout_seconds,
        max_prefix_trim_bytes=args.max_prefix_trim_bytes,
        run_grammar_summary=args.run_grammar_summary,
        alt_grammar_summary=args.alt_grammar_summary,
        mode_sweep_summary=args.mode_sweep_summary,
        macro_prior_dir=args.macro_prior_dir,
    )
    print(f"wrote {args.out_dir / 'candidate_summary.json'}")
    print(
        "score_claim=false dispatch_performed=false "
        f"candidate_count={summary['candidate_count']} "
        f"best_byte_delta={summary['best_byte_delta']} "
        f"family_status={summary['family_resolution']['status']} "
        f"safe_for_remote_dispatch={str(summary['dispatch_gate']['safe_for_remote_dispatch']).lower()}"
    )
    if not summary["candidate_count"]:
        print("blockers=" + ";".join(summary["blockers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
