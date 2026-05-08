#!/usr/bin/env python3
"""Analyze the active PR103-on-PR106 floor archive without making score claims."""

from __future__ import annotations

import argparse
import io
import math
import struct
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_bytes, write_json  # noqa: E402

CONTEST_ORIGINAL_BYTES = 37_545_489
TOOL_NAME = "tools.analyze_active_pr103_pr106_floor"
DEFAULT_ACTIVE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip"
)
DEFAULT_FALLBACK_ARCHIVE = REPO_ROOT / "experiments/results/pr103_repack_pr106_standalone_20260507/archive.zip"
DEFAULT_AUTH_EVAL = (
    REPO_ROOT
    / "experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z"
    / "contest_auth_eval.adjudicated.json"
)
DEFAULT_RUNTIME_CLOSURE = (
    REPO_ROOT / "experiments/results/pr103_repack_pr106_standalone_20260507/runtime_closure.json"
)
DEFAULT_RUNTIME_PACKET_PROOF = (
    REPO_ROOT
    / "experiments/results/pr103_repack_pr106_standalone_20260507/final_runtime_packet_proof.json"
)
DEFAULT_PRE_SUBMISSION = (
    REPO_ROOT
    / "experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.contest_final.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports/active_pr103_pr106_floor_anatomy_20260507.json"
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/active_pr103_pr106_floor_anatomy_20260507_worker_p.md"
DEFAULT_PUBLIC_INTAKE_ROOT = REPO_ROOT / "experiments/results/public_pr_intake_full"
DEFAULT_MANIFESTS = (
    REPO_ROOT / "experiments/results/pr103_repack_pr106_standalone_20260507/manifest.json",
    REPO_ROOT / "experiments/results/pr103_repack_pr106_composed_op1_op2_20260507/manifest.json",
)

PR103_PR106_DECODER_SECTION_ORDER = ("br", "hists", "merged_ac", "hi_hist", "ac_fallback")
PR103_AC_FIXED_SCALES_BYTES = 56
PR103_LC_AC_SECTIONS = (
    ("scales_fp16", 56),
    ("non_ac_weights_brotli", 7_097),
    ("ac_histograms_brotli", 895),
    ("merged_range_coded_weights_and_hi_latents", 153_856),
    ("latent_min_scale_fp16", 112),
    ("latent_low_bytes_brotli", 15_537),
    ("latent_hi_histogram_brotli", 15),
)
PR101_MICROCODEC_SECTIONS = (
    ("decoder_compact_brotli_streams", 162_164),
    ("latents_raw_lzma_delta_u8", 15_387),
)
SCORE_FIELD_NAMES = {
    "avg_posenet_dist",
    "avg_segnet_dist",
    "canonical_score",
    "final_score",
    "rate_unscaled",
    "reported_final_score_display_rounded",
    "score_pose_contribution",
    "score_rate_contribution",
    "score_recomputed_from_components",
    "score_reported_rounded_differs_from_canonical",
    "score_rounding_abs_delta",
    "score_seg_contribution",
}


def entropy_bits_per_byte(data: bytes) -> float:
    """Return byte entropy rounded for deterministic JSON output."""

    if not data:
        return 0.0
    total = len(data)
    counts = Counter(data)
    value = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round(value, 6)


def _display_path(path: Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    return repo_relative(path, repo_root)


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _unsafe_member_reason(name: str) -> str | None:
    path = Path(name)
    if name.startswith("/") or "\\" in name:
        return "absolute_or_backslash_member_name"
    if ".." in path.parts:
        return "zip_slip_parent_reference"
    if path.name.startswith(".") or "__MACOSX" in path.parts:
        return "hidden_or_resource_member"
    return None


def _find_eocd_offset(blob: bytes) -> int | None:
    offset = blob.rfind(b"PK\x05\x06")
    return offset if offset >= 0 else None


def _local_header_record(blob: bytes, info: zipfile.ZipInfo) -> dict[str, Any]:
    offset = int(info.header_offset)
    if offset < 0 or offset + 30 > len(blob):
        return {
            "local_header_offset": offset,
            "local_header_parse_error": "offset_out_of_range",
        }
    header = blob[offset : offset + 30]
    try:
        (
            signature,
            version_needed,
            flag_bits,
            compress_type,
            mod_time,
            mod_date,
            crc32,
            compressed_size,
            file_size,
            filename_len,
            extra_len,
        ) = struct.unpack("<IHHHHHIIIHH", header)
    except struct.error as exc:  # pragma: no cover - guarded by length check
        return {
            "local_header_offset": offset,
            "local_header_parse_error": str(exc),
        }
    name_start = offset + 30
    name_end = name_start + filename_len
    extra_end = name_end + extra_len
    local_name_bytes = blob[name_start:name_end]
    local_name = local_name_bytes.decode("utf-8", errors="replace")
    data_start = extra_end
    data_end = data_start + int(info.compress_size)
    return {
        "local_header_offset": offset,
        "local_header_signature": f"0x{signature:08x}",
        "local_header_signature_ok": signature == 0x04034B50,
        "local_header_bytes": 30 + filename_len + extra_len,
        "local_header_version_needed": version_needed,
        "local_header_flag_bits": flag_bits,
        "local_header_compress_type": compress_type,
        "local_header_mod_time_raw": mod_time,
        "local_header_mod_date_raw": mod_date,
        "local_header_crc32": f"{crc32:08x}",
        "local_header_compressed_size": compressed_size,
        "local_header_file_size": file_size,
        "local_header_filename_len": filename_len,
        "local_header_extra_len": extra_len,
        "local_header_name": local_name,
        "local_header_name_matches_central": local_name == info.filename,
        "compressed_data_start": data_start,
        "compressed_data_end": min(data_end, len(blob)),
        "compressed_data_size_from_central": int(info.compress_size),
        "uses_data_descriptor": bool(flag_bits & 0x08),
    }


def _zip_member_table(blob: bytes) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bytes]]:
    payloads: dict[str, bytes] = {}
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        start_dir = getattr(zf, "start_dir", None)
        eocd_offset = _find_eocd_offset(blob)
        for index, info in enumerate(zf.infolist()):
            local = _local_header_record(blob, info)
            payload: bytes | None = None
            payload_sha: str | None = None
            if not info.is_dir():
                payload = zf.read(info.filename)
                payload_sha = sha256_bytes(payload)
                payloads[info.filename] = payload
            members.append(
                {
                    "index": index,
                    "name": info.filename,
                    "is_dir": info.is_dir(),
                    "unsafe_reason": _unsafe_member_reason(info.filename),
                    "central_file_size": int(info.file_size),
                    "central_compress_size": int(info.compress_size),
                    "central_compress_type": int(info.compress_type),
                    "central_crc32": f"{info.CRC:08x}",
                    "central_flag_bits": int(info.flag_bits),
                    "central_create_system": int(info.create_system),
                    "central_create_version": int(info.create_version),
                    "central_extract_version": int(info.extract_version),
                    "central_external_attr": int(info.external_attr),
                    "central_date_time": list(info.date_time),
                    "payload_sha256": payload_sha,
                    "stored_payload_matches_compressed_bytes": (
                        payload is not None
                        and info.compress_type == zipfile.ZIP_STORED
                        and local.get("compressed_data_end") is not None
                        and blob[
                            int(local["compressed_data_start"]) : int(local["compressed_data_end"])
                        ]
                        == payload
                    ),
                    **local,
                }
            )
    summary = {
        "file_member_count": len([item for item in members if not item["is_dir"]]),
        "directory_member_count": len([item for item in members if item["is_dir"]]),
        "duplicate_file_names": duplicate_names,
        "unsafe_members": [
            {"name": item["name"], "unsafe_reason": item["unsafe_reason"]}
            for item in members
            if item["unsafe_reason"] is not None
        ],
        "central_directory_offset": start_dir,
        "central_directory_bytes": (
            eocd_offset - start_dir if isinstance(start_dir, int) and eocd_offset is not None else None
        ),
        "eocd_offset": eocd_offset,
        "eocd_bytes": len(blob) - eocd_offset if eocd_offset is not None else None,
    }
    return members, summary, payloads


def _section_record(
    *,
    name: str,
    payload: bytes,
    start: int,
    end: int,
    payload_archive_offset: int | None,
) -> dict[str, Any]:
    section = payload[start:end]
    return {
        "name": name,
        "payload_start": start,
        "payload_end": end,
        "archive_start": payload_archive_offset + start if payload_archive_offset is not None else None,
        "archive_end": payload_archive_offset + end if payload_archive_offset is not None else None,
        "bytes": len(section),
        "sha256": sha256_bytes(section),
        "entropy_bits_per_byte": entropy_bits_per_byte(section),
    }


def _runtime_closure_payload(runtime_closure_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if runtime_closure_record is None:
        return None
    closure = runtime_closure_record.get("runtime_closure", runtime_closure_record)
    return closure if isinstance(closure, dict) else None


def _profile_pr103_pr106_packed_payload(
    payload: bytes,
    payload_archive_offset: int | None,
    runtime_closure_record: dict[str, Any] | None,
) -> dict[str, Any]:
    decoder_len = int.from_bytes(payload[1:4], "little") if len(payload) >= 4 else -1
    record: dict[str, Any] = {
        "kind": "pr103_ac_decoder_inside_pr106_ff_packed_v1",
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "packed_header_decoder_len": decoder_len,
        "packed_header_valid": len(payload) >= 4 and payload[:1] == b"\xff" and 0 < decoder_len < len(payload) - 4,
        "sections": [],
        "nested_decoder_sections": [],
        "runtime_closure_match": None,
        "runtime_closure_blockers": [],
    }
    if not record["packed_header_valid"]:
        record["runtime_closure_blockers"].append("invalid_ff_len24_packed_header")
        return record

    decoder_start = 4
    decoder_end = 4 + decoder_len
    record["sections"] = [
        _section_record(
            name="packed_header_ff_len24",
            payload=payload,
            start=0,
            end=4,
            payload_archive_offset=payload_archive_offset,
        ),
        _section_record(
            name="decoder_pr103_ac_bytes",
            payload=payload,
            start=decoder_start,
            end=decoder_end,
            payload_archive_offset=payload_archive_offset,
        ),
        _section_record(
            name="latents_pr106_fixed_brotli",
            payload=payload,
            start=decoder_end,
            end=len(payload),
            payload_archive_offset=payload_archive_offset,
        ),
    ]

    closure = _runtime_closure_payload(runtime_closure_record)
    if closure is None:
        record["runtime_closure_blockers"].append("runtime_closure_missing")
        return record

    section_lengths = closure.get("section_lengths")
    if not isinstance(section_lengths, dict):
        record["runtime_closure_blockers"].append("runtime_closure_section_lengths_missing")
        return record

    cursor = decoder_start
    nested: list[dict[str, Any]] = [
        _section_record(
            name="decoder.scales_fp16",
            payload=payload,
            start=cursor,
            end=cursor + PR103_AC_FIXED_SCALES_BYTES,
            payload_archive_offset=payload_archive_offset,
        )
    ]
    cursor += PR103_AC_FIXED_SCALES_BYTES
    for name in PR103_PR106_DECODER_SECTION_ORDER:
        value = section_lengths.get(name)
        if not isinstance(value, int) or value < 0:
            record["runtime_closure_blockers"].append(f"runtime_closure_bad_section_length:{name}")
            continue
        nested.append(
            _section_record(
                name=f"decoder.{name}",
                payload=payload,
                start=cursor,
                end=cursor + value,
                payload_archive_offset=payload_archive_offset,
            )
        )
        cursor += value
    decoder_sha = sha256_bytes(payload[decoder_start:decoder_end])
    latents_sha = sha256_bytes(payload[decoder_end:])
    record["nested_decoder_sections"] = nested
    record["runtime_closure_match"] = {
        "section_lengths_sum_matches_decoder": cursor == decoder_end,
        "decoder_section_bytes_matches": closure.get("decoder_section_bytes") == decoder_len,
        "decoder_section_sha256_matches": closure.get("decoder_section_sha256") == decoder_sha,
        "latents_section_bytes_matches": closure.get("latents_section_bytes") == len(payload) - decoder_end,
        "latents_section_sha256_matches": closure.get("latents_section_sha256") == latents_sha,
        "ac_fallback_set": closure.get("ac_fallback_set", []),
        "format": closure.get("format"),
    }
    for key, value in record["runtime_closure_match"].items():
        if key.endswith("_matches") and value is not True:
            record["runtime_closure_blockers"].append(key)
    return record


def _profile_fixed_sections(
    *,
    kind: str,
    payload: bytes,
    specs: tuple[tuple[str, int], ...],
    payload_archive_offset: int | None,
) -> dict[str, Any]:
    cursor = 0
    sections: list[dict[str, Any]] = []
    for name, size in specs:
        sections.append(
            _section_record(
                name=name,
                payload=payload,
                start=cursor,
                end=min(cursor + size, len(payload)),
                payload_archive_offset=payload_archive_offset,
            )
        )
        cursor += size
    if cursor < len(payload):
        sections.append(
            _section_record(
                name="tail_or_sidecar",
                payload=payload,
                start=cursor,
                end=len(payload),
                payload_archive_offset=payload_archive_offset,
            )
        )
    return {
        "kind": kind,
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "fixed_section_bytes": cursor,
        "sections": sections,
        "fixed_layout_fits_payload": cursor <= len(payload),
    }


def _payload_archive_offset_for_member(member: dict[str, Any]) -> int | None:
    if member.get("central_compress_type") != zipfile.ZIP_STORED:
        return None
    if member.get("stored_payload_matches_compressed_bytes") is not True:
        return None
    value = member.get("compressed_data_start")
    return int(value) if isinstance(value, int) else None


def _classify_payload(
    *,
    archive_label: str,
    member: dict[str, Any],
    payload: bytes,
    runtime_closure_record: dict[str, Any] | None,
) -> dict[str, Any]:
    lower = archive_label.lower()
    payload_archive_offset = _payload_archive_offset_for_member(member)
    if len(payload) >= 4 and payload[:1] == b"\xff":
        return _profile_pr103_pr106_packed_payload(payload, payload_archive_offset, runtime_closure_record)
    if "pr101" in lower or "microcodec" in lower:
        return _profile_fixed_sections(
            kind="pr101_microcodec_or_compatible_fixed_layout",
            payload=payload,
            specs=PR101_MICROCODEC_SECTIONS,
            payload_archive_offset=payload_archive_offset,
        )
    if "pr103" in lower or "lc_ac" in lower:
        return _profile_fixed_sections(
            kind="pr103_lc_ac_or_compatible_fixed_layout",
            payload=payload,
            specs=PR103_LC_AC_SECTIONS,
            payload_archive_offset=payload_archive_offset,
        )
    return {
        "kind": "opaque_single_payload",
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "sections": [
            _section_record(
                name="opaque_single_payload",
                payload=payload,
                start=0,
                end=len(payload),
                payload_archive_offset=payload_archive_offset,
            )
        ],
    }


def inspect_archive_blob(
    *,
    archive_blob: bytes,
    archive_path: Path | None,
    archive_label: str,
    repo_root: Path,
    runtime_closure_record: dict[str, Any] | None = None,
    synthetic_fixture: bool = False,
) -> dict[str, Any]:
    members, zip_summary, payloads = _zip_member_table(archive_blob)
    file_members = [member for member in members if not member["is_dir"]]
    packed_payload_candidates: list[dict[str, Any]] = []
    for member in file_members:
        payload = payloads.get(str(member["name"]))
        if payload is None:
            continue
        packed_payload_candidates.append(
            {
                "member_name": member["name"],
                "member_index": member["index"],
                **_classify_payload(
                    archive_label=archive_label,
                    member=member,
                    payload=payload,
                    runtime_closure_record=runtime_closure_record,
                ),
            }
        )
    return {
        "label": archive_label,
        "path": _display_path(archive_path, repo_root),
        "synthetic_fixture": synthetic_fixture,
        "exists_on_disk": archive_path.is_file() if archive_path is not None else False,
        "bytes": len(archive_blob),
        "sha256": sha256_bytes(archive_blob),
        "zip_summary": zip_summary,
        "zip_member_table": members,
        "zip_overhead_vs_compressed_member_data_bytes": len(archive_blob)
        - sum(int(member["central_compress_size"]) for member in file_members),
        "zip_overhead_vs_uncompressed_member_data_bytes": len(archive_blob)
        - sum(int(member["central_file_size"]) for member in file_members),
        "packed_payload_candidates": packed_payload_candidates,
    }


def _synthetic_payload_and_closure() -> tuple[bytes, dict[str, Any]]:
    decoder_sections = {
        "scales_fp16": b"s" * PR103_AC_FIXED_SCALES_BYTES,
        "br": b"br" * 5,
        "hists": b"hist",
        "merged_ac": b"range-coded-symbols",
        "hi_hist": b"",
        "ac_fallback": b"",
    }
    decoder = decoder_sections["scales_fp16"] + b"".join(
        decoder_sections[name] for name in PR103_PR106_DECODER_SECTION_ORDER
    )
    latents = b"latent-brotli-placeholder"
    payload = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + latents
    closure = {
        "runtime_closure": {
            "schema_version": 1,
            "format": "synthetic_pr103_ac_decoder_inside_pr106_ff_packed_v1",
            "section_lengths": {
                name: len(value)
                for name, value in decoder_sections.items()
                if name in PR103_PR106_DECODER_SECTION_ORDER
            },
            "ac_fallback_set": [],
            "n_latent_hi_symbols": 0,
            "decoder_section_bytes": len(decoder),
            "decoder_section_sha256": sha256_bytes(decoder),
            "latents_section_bytes": len(latents),
            "latents_section_sha256": sha256_bytes(latents),
            "brotli_quality": 11,
            "adaptive_lgwin": True,
            "ac_auto_fallback": True,
        }
    }
    return payload, closure


def synthetic_fixture_zip_blob() -> tuple[bytes, dict[str, Any]]:
    """Return a deterministic tiny PR103/PR106-shaped ZIP fixture and closure."""

    payload, closure = _synthetic_payload_and_closure()
    buffer = io.BytesIO()
    info = zipfile.ZipInfo("0.bin", date_time=(2026, 5, 7, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(info, payload)
    return buffer.getvalue(), closure


def _resolve_default_archive() -> Path | None:
    for candidate in (DEFAULT_ACTIVE_ARCHIVE, DEFAULT_FALLBACK_ARCHIVE):
        if candidate.is_file():
            return candidate
    return None


def _manifest_summary(manifest: dict[str, Any], path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo_root),
        "tool": manifest.get("tool"),
        "score_claim": manifest.get("score_claim"),
        "score_evidence_grade": manifest.get("score_evidence_grade"),
        "source_archive_path": repo_relative(str(manifest.get("source_archive_path")), repo_root)
        if isinstance(manifest.get("source_archive_path"), str)
        else None,
        "source_archive_sha256": manifest.get("source_archive_sha256"),
        "source_archive_bytes": manifest.get("source_archive_bytes"),
        "source_payload_bytes": manifest.get("source_payload_bytes"),
        "source_member_name": manifest.get("source_member_name"),
        "source_decoder_section_bytes": manifest.get("source_decoder_section_bytes"),
        "source_latents_section_bytes": manifest.get("source_latents_section_bytes"),
        "output_archive_path": repo_relative(str(manifest.get("output_archive_path")), repo_root)
        if isinstance(manifest.get("output_archive_path"), str)
        else None,
        "output_archive_sha256": manifest.get("output_archive_sha256"),
        "output_archive_bytes": manifest.get("output_archive_bytes"),
        "output_decoder_section_bytes": manifest.get("output_decoder_section_bytes"),
        "output_latents_section_bytes": manifest.get("output_latents_section_bytes"),
        "decoder_delta_bytes": manifest.get("decoder_delta_bytes"),
        "archive_delta_bytes": manifest.get("archive_delta_bytes"),
        "runtime_adapter_required": manifest.get("runtime_adapter_required"),
        "runtime_adapter_blockers": manifest.get("runtime_adapter_blockers", []),
        "compose_with_pr101": manifest.get("compose_with_pr101"),
        "ac_audit": manifest.get("ac_audit", {}),
    }


def _load_candidate_manifests(paths: tuple[Path, ...], repo_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_optional_json(path)
        if payload is not None:
            out.append(_manifest_summary(payload, path, repo_root))
    return out


def _runtime_packet_summary(
    proof: dict[str, Any] | None,
    pre_submission: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "runtime_packet_proof_present": proof is not None,
        "pre_submission_compliance_present": pre_submission is not None,
    }
    if proof is not None:
        summary.update(
            {
                "runtime_packet_proof_passed": proof.get("passed"),
                "runtime_dir": proof.get("runtime_dir"),
                "runtime_files": proof.get("runtime_files", {}),
                "runtime_static_scan": proof.get("runtime_static_scan", {}),
                "dependency_custody": proof.get("dependency_custody", {}),
                "decode_proof": proof.get("decode_proof", {}),
                "runtime_packet_score_claim": proof.get("score_claim"),
            }
        )
    if pre_submission is not None:
        archive = pre_submission.get("archive") if isinstance(pre_submission.get("archive"), dict) else {}
        auth_eval = pre_submission.get("auth_eval") if isinstance(pre_submission.get("auth_eval"), dict) else {}
        summary.update(
            {
                "pre_submission_passed": pre_submission.get("passed"),
                "pre_submission_archive": {
                    "path": archive.get("path"),
                    "bytes": archive.get("bytes"),
                    "sha256": archive.get("sha256"),
                    "members": archive.get("members", []),
                },
                "pre_submission_auth_eval_path": auth_eval.get("path"),
                "pre_submission_runtime_tree_candidates": auth_eval.get("runtime_tree_candidates", {}),
            }
        )
    return summary


def _auth_eval_custody(
    auth_eval: dict[str, Any] | None,
    auth_eval_path: Path | None,
    archive_record: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if auth_eval is None:
        return {
            "path": _display_path(auth_eval_path, repo_root),
            "exists": False,
            "score_claim": False,
            "score_fields_suppressed": [],
            "logs_parsed_for_score": False,
            "exact_eval_blockers": ["auth_eval_artifact_missing"],
        }
    provenance = auth_eval.get("provenance") if isinstance(auth_eval.get("provenance"), dict) else {}
    runtime_manifest = (
        provenance.get("inflate_runtime_manifest")
        if isinstance(provenance.get("inflate_runtime_manifest"), dict)
        else {}
    )
    suppressed = sorted(key for key in SCORE_FIELD_NAMES if key in auth_eval)
    archive_sha = provenance.get("archive_sha256") or auth_eval.get("archive_sha256")
    archive_bytes = provenance.get("archive_size_bytes") or auth_eval.get("archive_size_bytes")
    blockers: list[str] = []
    if archive_sha != archive_record["sha256"]:
        blockers.append("auth_eval_archive_sha256_mismatch")
    if archive_bytes != archive_record["bytes"]:
        blockers.append("auth_eval_archive_bytes_mismatch")
    if provenance.get("device") != "cuda":
        blockers.append("auth_eval_not_cuda")
    if provenance.get("gpu_t4_match") is not True:
        blockers.append("auth_eval_not_t4_equivalent")
    if auth_eval.get("n_samples") != 600:
        blockers.append("auth_eval_not_full_600_samples")
    if not runtime_manifest.get("runtime_tree_sha256"):
        blockers.append("auth_eval_runtime_tree_sha256_missing")
    custody_caveats: list[str] = []
    if isinstance(provenance.get("pact_commit"), str) and provenance["pact_commit"].startswith("<error:"):
        custody_caveats.append("auth_eval_provenance_pact_commit_unavailable")
    return {
        "path": _display_path(auth_eval_path, repo_root),
        "exists": True,
        "score_claim": False,
        "score_values_suppressed": True,
        "score_fields_suppressed": suppressed,
        "logs_parsed_for_score": False,
        "archive_identity": {
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "matches_active_archive": not any(
                item in blockers
                for item in ("auth_eval_archive_sha256_mismatch", "auth_eval_archive_bytes_mismatch")
            ),
        },
        "cuda_custody": {
            "device": provenance.get("device"),
            "cuda_available": provenance.get("cuda_available"),
            "cuda_device_count": provenance.get("cuda_device_count"),
            "cuda_version": provenance.get("cuda_version"),
            "gpu_model": provenance.get("gpu_model"),
            "gpu_t4_match": provenance.get("gpu_t4_match"),
            "n_samples": auth_eval.get("n_samples"),
            "inflate_elapsed_seconds": auth_eval.get("inflate_elapsed_seconds"),
            "evaluate_elapsed_seconds": auth_eval.get("evaluate_elapsed_seconds"),
            "contest_auth_eval_elapsed_seconds": auth_eval.get("contest_auth_eval_elapsed_seconds"),
        },
        "runtime_tree_custody": {
            "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
            "runtime_file_count": runtime_manifest.get("runtime_file_count"),
            "runtime_files": runtime_manifest.get("files", []),
            "repo_local_tac_import_manifest": runtime_manifest.get("repo_local_tac_import_manifest"),
            "upstream_evaluate_py": runtime_manifest.get("upstream_evaluate_py"),
            "inflate_script_sha256": provenance.get("inflate_script_sha256"),
        },
        "custody_caveats": custody_caveats,
        "exact_eval_blockers": blockers,
        "exact_eval_status": "present_identity_matched_cuda_t4_600_samples" if not blockers else "blocked",
    }


def _public_pr_inventory(public_intake_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for pr_number in range(100, 108):
        pr_dir = public_intake_root / f"public_pr{pr_number}_intake_20260505_auto"
        archive = pr_dir / "archive.zip"
        metadata = _load_optional_json(pr_dir / "pr_metadata.json") or {}
        provenance = _load_optional_json(pr_dir / "archive_provenance.json") or {}
        record: dict[str, Any] = {
            "pr_number": pr_number,
            "intake_dir": repo_relative(pr_dir, repo_root),
            "archive_present": archive.is_file(),
            "metadata_present": bool(metadata),
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "head_repo": metadata.get("head_repo"),
            "head_sha": metadata.get("head_sha"),
            "created_at": metadata.get("created_at"),
            "updated_or_closed_at": metadata.get("updated_at") or metadata.get("closed_at"),
            "leaderboard_name": metadata.get("leaderboard_name"),
            "leaderboard_score_external_only": metadata.get("leaderboard_score"),
            "url": f"https://github.com/commaai/comma_video_compression_challenge/pull/{pr_number}",
            "archive_provenance": {
                "archive_size_bytes": provenance.get("archive_size_bytes"),
                "archive_sha256": provenance.get("archive_sha256"),
                "status": provenance.get("status"),
                "source_url": provenance.get("source_url"),
                "fetched_at_utc": provenance.get("fetched_at_utc"),
            },
        }
        if archive.is_file():
            try:
                archive_blob = archive.read_bytes()
                inspected = inspect_archive_blob(
                    archive_blob=archive_blob,
                    archive_path=archive,
                    archive_label=f"public_pr{pr_number}",
                    repo_root=repo_root,
                    runtime_closure_record=None,
                )
                record["archive_bytes"] = inspected["bytes"]
                record["archive_sha256"] = inspected["sha256"]
                record["zip_summary"] = inspected["zip_summary"]
                record["zip_member_table"] = inspected["zip_member_table"]
                record["packed_payload_candidates"] = inspected["packed_payload_candidates"]
            except (OSError, zipfile.BadZipFile, ValueError) as exc:
                record["archive_inspection_error"] = str(exc)
        inventory.append(record)
    return inventory


def _active_lineage(
    *,
    archive_record: dict[str, Any],
    candidate_manifests: list[dict[str, Any]],
    public_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    inventory_by_pr = {item["pr_number"]: item for item in public_inventory}
    matching_manifests = [
        manifest for manifest in candidate_manifests if manifest.get("output_archive_sha256") == archive_record["sha256"]
    ]
    return {
        "label": "PR103-on-PR106 active floor anatomy",
        "active_archive_sha256": archive_record["sha256"],
        "active_archive_bytes": archive_record["bytes"],
        "source_pr_roles": [
            {
                "role": "arithmetic_schema_source",
                "pr_number": 103,
                "archive_sha256": inventory_by_pr.get(103, {}).get("archive_sha256"),
                "archive_bytes": inventory_by_pr.get(103, {}).get("archive_bytes"),
                "title": inventory_by_pr.get(103, {}).get("title"),
                "author": inventory_by_pr.get(103, {}).get("author"),
            },
            {
                "role": "packed_pr106_envelope_and_latents_source",
                "pr_number": 106,
                "archive_sha256": inventory_by_pr.get(106, {}).get("archive_sha256"),
                "archive_bytes": inventory_by_pr.get(106, {}).get("archive_bytes"),
                "title": inventory_by_pr.get(106, {}).get("title"),
                "author": inventory_by_pr.get(106, {}).get("author"),
            },
        ],
        "matching_candidate_manifests": matching_manifests,
        "lineage_score_claim": False,
    }


def _charged_byte_proof(
    archive_record: dict[str, Any],
    auth_eval_custody: dict[str, Any],
    candidate_manifests: list[dict[str, Any]],
    runtime_closure_record: dict[str, Any] | None,
) -> dict[str, Any]:
    identity_sources: list[dict[str, Any]] = [
        {
            "source": "active_archive",
            "archive_bytes": archive_record["bytes"],
            "archive_sha256": archive_record["sha256"],
            "matches_active": True,
        }
    ]
    if auth_eval_custody.get("exists"):
        identity = auth_eval_custody.get("archive_identity", {})
        if isinstance(identity, dict):
            identity_sources.append(
                {
                    "source": "auth_eval_adjudicated_json_identity_only",
                    "archive_bytes": identity.get("archive_bytes"),
                    "archive_sha256": identity.get("archive_sha256"),
                    "matches_active": identity.get("matches_active_archive"),
                }
            )
    for manifest in candidate_manifests:
        identity_sources.append(
            {
                "source": manifest.get("path"),
                "archive_bytes": manifest.get("output_archive_bytes"),
                "archive_sha256": manifest.get("output_archive_sha256"),
                "matches_active": manifest.get("output_archive_sha256") == archive_record["sha256"]
                and manifest.get("output_archive_bytes") == archive_record["bytes"],
            }
        )
    closure = _runtime_closure_payload(runtime_closure_record)
    if closure is not None:
        identity_sources.append(
            {
                "source": "runtime_closure_candidate_archive",
                "archive_bytes": (runtime_closure_record or {}).get("candidate_archive", {}).get("archive_bytes"),
                "archive_sha256": (runtime_closure_record or {}).get("candidate_archive", {}).get("archive_sha256"),
                "matches_active": (runtime_closure_record or {}).get("candidate_archive", {}).get("archive_sha256")
                == archive_record["sha256"],
            }
        )
    return {
        "charged_archive_bytes": archive_record["bytes"],
        "charged_archive_sha256": archive_record["sha256"],
        "contest_original_bytes": CONTEST_ORIGINAL_BYTES,
        "rate_term_multiplier": 25,
        "score_recompute_intentionally_omitted": True,
        "score_claim": False,
        "identity_sources": identity_sources,
        "all_available_identity_sources_match": all(
            source.get("matches_active") is True
            for source in identity_sources
            if source.get("archive_sha256") is not None or source.get("archive_bytes") is not None
        ),
    }


def build_anatomy_report(
    *,
    archive_path: Path | None = None,
    auth_eval_path: Path | None = DEFAULT_AUTH_EVAL,
    runtime_closure_path: Path | None = DEFAULT_RUNTIME_CLOSURE,
    runtime_packet_proof_path: Path | None = DEFAULT_RUNTIME_PACKET_PROOF,
    pre_submission_path: Path | None = DEFAULT_PRE_SUBMISSION,
    public_intake_root: Path = DEFAULT_PUBLIC_INTAKE_ROOT,
    candidate_manifest_paths: tuple[Path, ...] = DEFAULT_MANIFESTS,
    repo_root: Path = REPO_ROOT,
    allow_synthetic_fixture: bool = True,
) -> dict[str, Any]:
    resolved_archive = archive_path if archive_path is not None else _resolve_default_archive()
    runtime_closure_record = _load_optional_json(runtime_closure_path)
    synthetic_fixture = False
    if resolved_archive is not None and resolved_archive.is_file():
        archive_blob = resolved_archive.read_bytes()
    elif allow_synthetic_fixture:
        archive_blob, synthetic_closure = synthetic_fixture_zip_blob()
        runtime_closure_record = runtime_closure_record or synthetic_closure
        resolved_archive = None
        synthetic_fixture = True
    else:
        raise FileNotFoundError("active PR103-on-PR106 archive not found and synthetic fixture disabled")

    archive_record = inspect_archive_blob(
        archive_blob=archive_blob,
        archive_path=resolved_archive,
        archive_label="active_pr103_pr106_floor",
        repo_root=repo_root,
        runtime_closure_record=runtime_closure_record,
        synthetic_fixture=synthetic_fixture,
    )
    auth_eval = _load_optional_json(auth_eval_path)
    candidate_manifests = _load_candidate_manifests(candidate_manifest_paths, repo_root)
    runtime_packet_proof = _load_optional_json(runtime_packet_proof_path)
    pre_submission = _load_optional_json(pre_submission_path)
    auth_custody = _auth_eval_custody(auth_eval, auth_eval_path, archive_record, repo_root)
    public_inventory = _public_pr_inventory(public_intake_root, repo_root)
    report = {
        "schema_version": 1,
        "tool": TOOL_NAME,
        "score_claim": False,
        "score_values_suppressed": True,
        "logs_parsed_for_score": False,
        "active_archive": archive_record,
        "charged_byte_proof": _charged_byte_proof(
            archive_record,
            auth_custody,
            candidate_manifests,
            runtime_closure_record,
        ),
        "auth_eval_custody": auth_custody,
        "runtime_and_inflate_tree_custody_hints": _runtime_packet_summary(runtime_packet_proof, pre_submission),
        "candidate_manifests": candidate_manifests,
        "active_floor_lineage": _active_lineage(
            archive_record=archive_record,
            candidate_manifests=candidate_manifests,
            public_inventory=public_inventory,
        ),
        "public_pr100_107_inventory": public_inventory,
        "exact_eval_blockers": auth_custody.get("exact_eval_blockers", []),
        "blocker_policy": {
            "score_from_logs_forbidden": True,
            "score_from_adjudication_suppressed_in_this_anatomy": True,
            "missing_or_mismatched_exact_eval_identity_is_a_blocker": True,
        },
    }
    return report


def render_ledger(report: dict[str, Any]) -> str:
    archive = report["active_archive"]
    auth = report["auth_eval_custody"]
    charged = report["charged_byte_proof"]
    exact_blockers = report.get("exact_eval_blockers", [])
    candidates = archive.get("packed_payload_candidates", [])
    first_candidate = candidates[0] if candidates else {}
    sections = first_candidate.get("sections", []) if isinstance(first_candidate, dict) else []
    nested = first_candidate.get("nested_decoder_sections", []) if isinstance(first_candidate, dict) else []
    lines = [
        "# Active PR103-on-PR106 Floor Anatomy - Worker P",
        "",
        "This ledger records byte/custody anatomy only. It does not claim score from logs or eval JSON.",
        "",
        "## Active Archive",
        "",
        f"- path: `{archive.get('path') or 'synthetic fixture'}`",
        f"- bytes: `{archive['bytes']}`",
        f"- sha256: `{archive['sha256']}`",
        f"- zip file members: `{archive['zip_summary']['file_member_count']}`",
        f"- zip overhead vs stored payload bytes: `{archive['zip_overhead_vs_uncompressed_member_data_bytes']}`",
        "",
        "## Charged Byte Proof",
        "",
        f"- charged_archive_bytes: `{charged['charged_archive_bytes']}`",
        f"- charged_archive_sha256: `{charged['charged_archive_sha256']}`",
        f"- identity_sources_match: `{charged['all_available_identity_sources_match']}`",
        f"- score_claim: `{charged['score_claim']}`",
        "",
        "## Packed Payload",
        "",
    ]
    for section in sections:
        lines.append(
            f"- `{section['name']}` payload[{section['payload_start']}:{section['payload_end']}] "
            f"{section['bytes']}B sha256 `{section['sha256']}`"
        )
    if nested:
        lines.extend(["", "## Nested Decoder Sections", ""])
        for section in nested:
            lines.append(
                f"- `{section['name']}` payload[{section['payload_start']}:{section['payload_end']}] "
                f"{section['bytes']}B sha256 `{section['sha256']}`"
            )
    lines.extend(
        [
            "",
            "## Exact Eval Identity",
            "",
            f"- auth eval artifact: `{auth.get('path')}`",
            f"- exact_eval_blockers: `{exact_blockers}`",
            f"- score_fields_suppressed: `{auth.get('score_fields_suppressed', [])}`",
            f"- runtime_tree_sha256: `{auth.get('runtime_tree_custody', {}).get('runtime_tree_sha256')}`",
            "",
            "## PR100-107 Intake Coverage",
            "",
        ]
    )
    for item in report["public_pr100_107_inventory"]:
        lines.append(
            f"- PR{item['pr_number']}: archive_present=`{item['archive_present']}` "
            f"bytes=`{item.get('archive_bytes')}` sha256=`{item.get('archive_sha256')}` "
            f"title=`{item.get('title')}`"
        )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Archive to analyze; defaults to active exact floor if present.")
    parser.add_argument("--auth-eval", type=Path, default=DEFAULT_AUTH_EVAL)
    parser.add_argument("--runtime-closure", type=Path, default=DEFAULT_RUNTIME_CLOSURE)
    parser.add_argument("--runtime-packet-proof", type=Path, default=DEFAULT_RUNTIME_PACKET_PROOF)
    parser.add_argument("--pre-submission", type=Path, default=DEFAULT_PRE_SUBMISSION)
    parser.add_argument("--public-intake-root", type=Path, default=DEFAULT_PUBLIC_INTAKE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger-out", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-synthetic-fixture", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if exact-eval blockers remain.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_anatomy_report(
        archive_path=args.archive,
        auth_eval_path=args.auth_eval,
        runtime_closure_path=args.runtime_closure,
        runtime_packet_proof_path=args.runtime_packet_proof,
        pre_submission_path=args.pre_submission,
        public_intake_root=args.public_intake_root,
        allow_synthetic_fixture=not args.no_synthetic_fixture,
    )
    write_json(args.output, report)
    args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
    args.ledger_out.write_text(render_ledger(report), encoding="utf-8")
    print(f"[active-pr103-pr106-floor] wrote {args.output}")
    print(f"[active-pr103-pr106-floor] wrote {args.ledger_out}")
    blockers = report.get("exact_eval_blockers", [])
    print(f"[active-pr103-pr106-floor] exact_eval_blockers={blockers}")
    if args.strict and blockers:
        return 1
    print(json_text({"archive_bytes": report["active_archive"]["bytes"], "score_claim": False}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
