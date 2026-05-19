#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a strict runtime-consumption proof for a monolithic packet candidate.

This is a CPU-only, non-score proof path. It does not run the scorer, dispatch
GPU work, or touch OMX/provider state. For PR106-style monolithic HNeRV packets
it independently consumes the single ZIP member with the runtime wire grammar,
verifies the Brotli sections decode, writes a deterministic runtime log, and
emits ``tac_runtime_consumption_proof_v1`` for the monolithic closure gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import brotli

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

SCHEMA = "tac_runtime_consumption_proof_v1"
PROOF_KIND = "tac_monolithic_runtime_consumption_probe_v1"
MONOLITHIC_MANIFEST_SCHEMA = "tac_monolithic_packet_candidate_v1"
SUPPORTED_GRAMMARS = frozenset(
    {
        "pr106_ff_packed_hnerv",
        "pr101_fixed_offset_hnerv_microcodec",
    }
)
PR106_HEADER_LEN = 4
PR106_HEADER_MAGIC = 0xFF
PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
ZIP_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
ZIP_LOCAL_HEADER_SIGNATURE = 0x04034B50
EXPECTED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
EXPECTED_EXTERNAL_ATTR = 0o100644 << 16


class MonolithicRuntimeConsumptionProofError(ValueError):
    """Raised when runtime-consumption proof inputs are malformed."""


def build_monolithic_runtime_consumption_proof(
    *,
    candidate_manifest_path: Path,
    runtime_log_out: Path | None = None,
    command_text: str | None = None,
) -> dict[str, Any]:
    """Return a strict no-score runtime-consumption proof for one candidate."""

    candidate_manifest_path = Path(candidate_manifest_path)
    manifest = _load_json_object(candidate_manifest_path)
    manifest_sha256 = _sha256_file(candidate_manifest_path)
    blockers: list[str] = []
    transcript: list[str] = [
        "proof_kind=tac_monolithic_runtime_consumption_probe_v1",
        "score_claim=false",
        f"candidate_manifest_path={candidate_manifest_path}",
        f"candidate_manifest_sha256={manifest_sha256}",
    ]

    if manifest.get("schema") != MONOLITHIC_MANIFEST_SCHEMA:
        blockers.append("candidate_manifest_schema_mismatch")
    if manifest.get("score_claim") is not False:
        blockers.append("candidate_manifest_score_claim_not_false")

    archive_info = _require_mapping(manifest.get("candidate_archive"), "candidate_archive")
    layout = _require_mapping(manifest.get("monolithic_layout"), "monolithic_layout")
    replacements = _require_list(manifest.get("replacements"), "replacements")
    archive_path = _resolve_manifest_path(
        archive_info.get("path"),
        base_dir=candidate_manifest_path.parent,
        label="candidate_archive.path",
    )
    expected_archive_bytes = _optional_int(archive_info.get("bytes"))
    expected_archive_sha = _optional_sha(archive_info.get("sha256"))
    expected_member_name = layout.get("member_name")
    expected_member_sha = _optional_sha(layout.get("new_member_sha256"))
    grammar = layout.get("grammar") if isinstance(layout.get("grammar"), str) else ""
    if grammar not in SUPPORTED_GRAMMARS:
        blockers.append(f"unsupported_monolithic_runtime_grammar:{grammar or '<missing>'}")

    archive_bytes = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes)
    transcript.extend(
        [
            f"candidate_archive_path={archive_path}",
            f"candidate_archive_bytes={len(archive_bytes)}",
            f"candidate_archive_sha256={archive_sha}",
        ]
    )
    if expected_archive_bytes is not None and len(archive_bytes) != expected_archive_bytes:
        blockers.append("candidate_archive_bytes_mismatch")
    if expected_archive_sha is None or archive_sha != expected_archive_sha:
        blockers.append("candidate_archive_sha256_mismatch")

    extracted = _extract_single_member(
        candidate_archive=archive_path,
        archive_bytes=archive_bytes,
    )
    blockers.extend(extracted["blockers"])
    member_name = extracted["member_name"]
    member_bytes = extracted["member_bytes"]
    member_sha = _sha256_bytes(member_bytes) if member_bytes else ""
    transcript.extend(
        [
            f"member_name={member_name}",
            f"member_bytes={len(member_bytes)}",
            f"rebuilt_member_sha256={member_sha}",
            f"new_member_sha256={member_sha}",
        ]
    )
    if isinstance(expected_member_name, str) and expected_member_name and member_name != expected_member_name:
        blockers.append("monolithic_member_name_mismatch")
    if expected_member_sha is None or member_sha != expected_member_sha:
        blockers.append("rebuilt_member_sha256_mismatch")

    parsed_sections: list[dict[str, Any]] = []
    if member_bytes and grammar == "pr106_ff_packed_hnerv":
        try:
            parsed_sections = _standalone_runtime_parse_pr106(member_bytes)
            transcript.append("standalone_runtime_parse=pr106_ff_packed_hnerv")
        except MonolithicRuntimeConsumptionProofError as exc:
            blockers.append(f"standalone_runtime_parse_failed:{exc}")
            transcript.append(f"standalone_runtime_parse_failed={exc}")
    elif member_bytes and grammar == "pr101_fixed_offset_hnerv_microcodec":
        try:
            parsed_sections = _standalone_runtime_parse_pr101(member_bytes)
            transcript.append("standalone_runtime_parse=pr101_fixed_offset_hnerv_microcodec")
        except MonolithicRuntimeConsumptionProofError as exc:
            blockers.append(f"standalone_runtime_parse_failed:{exc}")
            transcript.append(f"standalone_runtime_parse_failed={exc}")
    elif not member_bytes:
        blockers.append("member_bytes_missing")

    layout_sections = _manifest_section_map(layout.get("sections"))
    section_comparison = _compare_manifest_sections(
        expected_sections=layout_sections,
        runtime_sections=parsed_sections,
    )
    blockers.extend(section_comparison["blockers"])
    transcript.extend(section_comparison["transcript"])

    changed_sections, changed_blockers, changed_transcript = _changed_sections(
        replacements=replacements,
        runtime_sections=parsed_sections,
    )
    blockers.extend(changed_blockers)
    transcript.extend(changed_transcript)

    command = command_text or (
        "tools/prove_monolithic_runtime_consumption.py "
        f"--candidate-manifest {candidate_manifest_path}"
    )
    transcript_text = dumps_runtime_log(transcript)
    log_path = ""
    log_sha256 = _sha256_text(transcript_text)
    if runtime_log_out is None:
        blockers.append("runtime_log_output_path_missing")
    else:
        runtime_log_out = Path(runtime_log_out)
        runtime_log_out.parent.mkdir(parents=True, exist_ok=True)
        runtime_log_out.write_text(transcript_text, encoding="utf-8")
        log_path = str(runtime_log_out)
        log_sha256 = _sha256_file(runtime_log_out)

    ready = not blockers
    return {
        "schema": SCHEMA,
        "proof_kind": PROOF_KIND,
        "candidate_id": manifest.get("candidate_id", ""),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": manifest_sha256,
        "candidate_archive_path": str(archive_path),
        "candidate_archive_bytes": len(archive_bytes),
        "candidate_archive_sha256": archive_sha,
        "member_name": member_name,
        "member_bytes": len(member_bytes),
        "member_sha256": member_sha,
        "rebuilt_member_sha256": member_sha,
        "new_member_sha256": member_sha,
        "runtime_grammar": grammar,
        "consumed_sections": parsed_sections,
        "changed_sections": changed_sections,
        "command_sha256": _sha256_text(command),
        "command_source": "inline" if command_text is not None else "generated",
        "log_path": log_path,
        "log_sha256": log_sha256,
        "runtime_log_sha256": log_sha256,
        "ready_for_exact_eval_runtime": ready,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        "dispatch_blockers": [
            "runtime_consumption_proof_only_no_lane_claim",
            "closure_gate_still_required_before_exact_eval_dispatch",
            "contest_cuda_auth_eval_not_run",
        ],
        "promotion_blockers": [
            "contest_cuda_auth_eval_missing",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "omx_state_touched": False,
    }


def dumps_json(payload: Mapping[str, Any]) -> str:
    """Return stable pretty JSON for the proof payload."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def dumps_runtime_log(lines: list[str]) -> str:
    """Return the deterministic runtime log text hashed into the proof."""

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--runtime-log-out", type=Path)
    parser.add_argument("--command-text")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = build_monolithic_runtime_consumption_proof(
            candidate_manifest_path=args.candidate_manifest,
            runtime_log_out=args.runtime_log_out,
            command_text=args.command_text,
        )
    except (MonolithicRuntimeConsumptionProofError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"monolithic runtime consumption proof failed: {exc}") from None

    text = dumps_json(payload)
    if args.json_out is None:
        print(text, end="")
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    if args.fail_if_not_ready and payload["ready_for_exact_eval_runtime"] is not True:
        return 1
    return 0


def _extract_single_member(
    *,
    candidate_archive: Path,
    archive_bytes: bytes,
) -> dict[str, Any]:
    blockers: list[str] = []
    with zipfile.ZipFile(candidate_archive) as zf:
        infos = zf.infolist()
        if zf.comment:
            blockers.append("zip_comment_present")
        if len(infos) != 1:
            raise MonolithicRuntimeConsumptionProofError(
                f"expected one ZIP member, got {len(infos)}"
            )
        info = infos[0]
        _validate_member_name(info.filename)
        member_bytes = zf.read(info.filename)

    local = _inspect_local_zip_header(
        archive_bytes=archive_bytes,
        header_offset=info.header_offset,
    )
    blockers.extend(_zip_member_blockers(info=info, local=local))
    if info.file_size != len(member_bytes):
        blockers.append("zip_member_file_size_mismatch")
    return {
        "member_name": info.filename,
        "member_bytes": member_bytes,
        "blockers": blockers,
    }


def _validate_member_name(member_name: str) -> None:
    if not member_name:
        raise MonolithicRuntimeConsumptionProofError("ZIP member name is empty")
    if member_name.startswith(("/", "\\")) or "\\" in member_name:
        raise MonolithicRuntimeConsumptionProofError(
            f"unsafe ZIP member name: {member_name!r}"
        )
    parts = member_name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise MonolithicRuntimeConsumptionProofError(
            f"unsafe ZIP member name: {member_name!r}"
        )
    if any(part.startswith(".") for part in parts):
        raise MonolithicRuntimeConsumptionProofError(
            f"hidden ZIP member name rejected: {member_name!r}"
        )


def _inspect_local_zip_header(
    *,
    archive_bytes: bytes,
    header_offset: int,
) -> dict[str, Any]:
    if header_offset < 0 or header_offset + ZIP_LOCAL_HEADER.size > len(archive_bytes):
        raise MonolithicRuntimeConsumptionProofError("local ZIP header offset is invalid")
    (
        signature,
        version_needed,
        flag_bits,
        compression,
        mod_time,
        mod_date,
        crc32,
        compressed_size,
        uncompressed_size,
        filename_len,
        extra_len,
    ) = ZIP_LOCAL_HEADER.unpack_from(archive_bytes, header_offset)
    if signature != ZIP_LOCAL_HEADER_SIGNATURE:
        raise MonolithicRuntimeConsumptionProofError("bad local ZIP header signature")
    name_start = header_offset + ZIP_LOCAL_HEADER.size
    name_end = name_start + filename_len
    extra_end = name_end + extra_len
    if extra_end > len(archive_bytes):
        raise MonolithicRuntimeConsumptionProofError("truncated local ZIP header")
    filename = archive_bytes[name_start:name_end].decode("utf-8")
    return {
        "header_offset": header_offset,
        "version_needed": version_needed,
        "flag_bits": flag_bits,
        "compression": compression,
        "mod_time": mod_time,
        "mod_date": mod_date,
        "crc32": f"{crc32:08x}",
        "compressed_size": compressed_size,
        "uncompressed_size": uncompressed_size,
        "filename": filename,
        "filename_len": filename_len,
        "extra_len": extra_len,
        "data_offset": extra_end,
    }


def _zip_member_blockers(info: zipfile.ZipInfo, local: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if info.compress_type != zipfile.ZIP_STORED:
        blockers.append("zip_member_not_stored")
    if info.date_time != EXPECTED_ZIP_DATE_TIME:
        blockers.append("zip_member_non_deterministic_timestamp")
    if info.create_system != 3:
        blockers.append("zip_member_create_system_not_unix")
    if info.external_attr != EXPECTED_EXTERNAL_ATTR:
        blockers.append("zip_member_external_attr_mismatch")
    if info.flag_bits != 0:
        blockers.append("zip_member_flag_bits_nonzero")
    if info.comment:
        blockers.append("zip_member_comment_present")
    if info.extra:
        blockers.append("zip_member_extra_present")
    if local.get("filename") != info.filename:
        blockers.append("zip_local_central_filename_mismatch")
    if local.get("compression") != info.compress_type:
        blockers.append("zip_local_central_compression_mismatch")
    if local.get("crc32") != f"{info.CRC:08x}":
        blockers.append("zip_local_central_crc32_mismatch")
    if local.get("compressed_size") != info.compress_size:
        blockers.append("zip_local_central_compressed_size_mismatch")
    if local.get("uncompressed_size") != info.file_size:
        blockers.append("zip_local_central_uncompressed_size_mismatch")
    if local.get("extra_len") != 0:
        blockers.append("zip_local_extra_present")
    return blockers


def _standalone_runtime_parse_pr106(member: bytes) -> list[dict[str, Any]]:
    if len(member) < PR106_HEADER_LEN:
        raise MonolithicRuntimeConsumptionProofError("PR106 member too short for header")
    if member[0] != PR106_HEADER_MAGIC:
        raise MonolithicRuntimeConsumptionProofError(
            f"bad PR106 magic: 0x{member[0]:02x}"
        )
    decoder_len = int.from_bytes(member[1:4], "little")
    decoder_start = PR106_HEADER_LEN
    decoder_end = decoder_start + decoder_len
    if decoder_len <= 0:
        raise MonolithicRuntimeConsumptionProofError("PR106 decoder length must be positive")
    if decoder_end > len(member):
        raise MonolithicRuntimeConsumptionProofError(
            f"PR106 decoder section ends at {decoder_end}, member has {len(member)} bytes"
        )
    tail = member[decoder_end:]
    if not tail:
        raise MonolithicRuntimeConsumptionProofError("PR106 latent/sidecar tail is empty")
    header = member[:PR106_HEADER_LEN]
    decoder = member[decoder_start:decoder_end]
    decoder_raw = _brotli_decompress(decoder, section_name="decoder_packed_brotli")
    tail_raw = _brotli_decompress(tail, section_name="latents_and_sidecar_brotli")
    return [
        _section_row("ff_header", offset=0, data=header, role="internal_length_header"),
        _section_row(
            "decoder_packed_brotli",
            offset=decoder_start,
            data=decoder,
            role="renderer_decoder_weights",
            decompressed=decoder_raw,
        ),
        _section_row(
            "latents_and_sidecar_brotli",
            offset=decoder_end,
            data=tail,
            role="latent_sidecar_not_separate_pose_or_mask_member",
            decompressed=tail_raw,
        ),
    ]


def _standalone_runtime_parse_pr101(member: bytes) -> list[dict[str, Any]]:
    minimum = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    if len(member) < minimum:
        raise MonolithicRuntimeConsumptionProofError(
            f"PR101 member too short: {len(member)} < {minimum}"
        )
    decoder = member[:PR101_DECODER_BLOB_LEN]
    latent_start = PR101_DECODER_BLOB_LEN
    sidecar_start = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    latent = member[latent_start:sidecar_start]
    sidecar = member[sidecar_start:]
    if not sidecar:
        raise MonolithicRuntimeConsumptionProofError("PR101 sidecar section is empty")
    decoder_raw, stream_count = _decompress_concatenated_brotli(
        decoder,
        section_name="decoder_blob",
    )
    return [
        _section_row(
            "decoder_blob",
            offset=0,
            data=decoder,
            role="renderer_decoder_weights",
            decompressed=decoder_raw,
            extra={"split_brotli_stream_count": stream_count},
        ),
        _section_row(
            "latent_blob",
            offset=latent_start,
            data=latent,
            role="latent_motion_or_frame_conditioning",
        ),
        _section_row(
            "sidecar_blob",
            offset=sidecar_start,
            data=sidecar,
            role="latent_sidecar_not_separate_pose_or_mask_member",
        ),
    ]


def _brotli_decompress(data: bytes, *, section_name: str) -> bytes:
    try:
        return brotli.decompress(data)
    except brotli.error as exc:
        raise MonolithicRuntimeConsumptionProofError(
            f"section {section_name} does not Brotli-decompress"
        ) from exc


def _decompress_concatenated_brotli(data: bytes, *, section_name: str) -> tuple[bytes, int]:
    cursor = 0
    raw_parts: list[bytes] = []
    stream_count = 0
    while cursor < len(data):
        decoder = brotli.Decompressor()
        while cursor < len(data):
            try:
                raw_parts.append(decoder.process(data[cursor: cursor + 1]))
            except brotli.error as exc:
                raise MonolithicRuntimeConsumptionProofError(
                    f"section {section_name} split-Brotli decode failed at byte {cursor}"
                ) from exc
            cursor += 1
            if decoder.is_finished():
                stream_count += 1
                break
        if not decoder.is_finished():
            raise MonolithicRuntimeConsumptionProofError(
                f"section {section_name} has an unterminated split-Brotli stream"
            )
    if stream_count == 0:
        raise MonolithicRuntimeConsumptionProofError(
            f"section {section_name} contains no split-Brotli streams"
        )
    return b"".join(raw_parts), stream_count


def _section_row(
    name: str,
    *,
    offset: int,
    data: bytes,
    role: str,
    decompressed: bytes | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "section_name": name,
        "name": name,
        "role": role,
        "offset": offset,
        "length": len(data),
        "len": len(data),
        "sha256": _sha256_bytes(data),
        "new_sha256": _sha256_bytes(data),
        "runtime_consumed": True,
    }
    if decompressed is not None:
        row.update(
            {
                "brotli_decompressed": True,
                "decompressed_bytes": len(decompressed),
                "decompressed_sha256": _sha256_bytes(decompressed),
            }
        )
    if extra is not None:
        row.update(dict(extra))
    return row


def _manifest_section_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    sections: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        sections[name] = {
            "offset": item.get("new_offset", item.get("offset")),
            "length": item.get("new_len", item.get("len")),
            "sha256": item.get("new_sha256", item.get("sha256")),
        }
    return sections


def _compare_manifest_sections(
    *,
    expected_sections: dict[str, dict[str, Any]],
    runtime_sections: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    transcript: list[str] = []
    if not expected_sections:
        blockers.append("candidate_manifest_logical_sections_missing")
    runtime_by_name = {section["section_name"]: section for section in runtime_sections}
    for name, expected in expected_sections.items():
        runtime = runtime_by_name.get(name)
        if runtime is None:
            blockers.append(f"runtime_section_missing:{name}")
            continue
        if expected.get("offset") != runtime["offset"]:
            blockers.append(f"runtime_section_offset_mismatch:{name}")
        if expected.get("length") != runtime["length"]:
            blockers.append(f"runtime_section_length_mismatch:{name}")
        if not _sha_matches(runtime["sha256"], expected.get("sha256")):
            blockers.append(f"runtime_section_sha256_mismatch:{name}")
        transcript.append(
            "section "
            f"{name} offset={runtime['offset']} length={runtime['length']} "
            f"sha256={runtime['sha256']} runtime_consumed=true"
        )
    return {"blockers": blockers, "transcript": transcript}


def _changed_sections(
    *,
    replacements: list[Any],
    runtime_sections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    blockers: list[str] = []
    transcript: list[str] = []
    changed: list[dict[str, Any]] = []
    runtime_by_name = {section["section_name"]: section for section in runtime_sections}
    for item in replacements:
        if not isinstance(item, Mapping):
            blockers.append("replacement_entry_not_object")
            continue
        name = item.get("section_name")
        expected_sha = item.get("new_sha256", item.get("expected_new_sha256"))
        if not isinstance(name, str) or not name:
            blockers.append("replacement_section_name_missing")
            continue
        if not _is_sha256(expected_sha):
            blockers.append(f"replacement_new_sha256_invalid:{name}")
            continue
        runtime = runtime_by_name.get(name)
        if runtime is None:
            blockers.append(f"runtime_changed_section_missing:{name}")
            continue
        matched = _sha_matches(runtime["sha256"], expected_sha)
        if not matched:
            blockers.append(f"runtime_changed_section_sha256_mismatch:{name}")
        row = {
            "section_name": name,
            "new_sha256": str(expected_sha).lower(),
            "runtime_reported_sha256": runtime["sha256"],
            "runtime_consumed": matched and runtime.get("runtime_consumed") is True,
            "runtime_log_token_found": matched,
            "offset": runtime["offset"],
            "length": runtime["length"],
        }
        if "decompressed_sha256" in runtime:
            row["decompressed_sha256"] = runtime["decompressed_sha256"]
            row["decompressed_bytes"] = runtime["decompressed_bytes"]
        changed.append(row)
        transcript.append(
            "changed_section "
            f"section_name={name} new_sha256={expected_sha} "
            f"runtime_consumed={row['runtime_consumed']}"
        )
    if not changed:
        blockers.append("changed_sections_missing")
    return changed, blockers, transcript


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MonolithicRuntimeConsumptionProofError(f"{path} must contain a JSON object")
    return payload


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MonolithicRuntimeConsumptionProofError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise MonolithicRuntimeConsumptionProofError(f"{name} must be a list")
    return value


def _resolve_manifest_path(value: Any, *, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MonolithicRuntimeConsumptionProofError(f"{label} is required")
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return base_dir / path


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_sha(value: Any) -> str | None:
    return value.lower() if _is_sha256(value) else None


def _sha_matches(value: Any, expected: Any) -> bool:
    return _is_sha256(value) and _is_sha256(expected) and str(value).lower() == str(expected).lower()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
