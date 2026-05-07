"""PR101-style schema-driven packing for PR106 HNeRV decoder sections.

PR101's decoder win is not generic Brotli recompression: it stores tensors in a
fixed schema order, applies per-tensor byte maps/permutations, and concatenates
multiple Brotli streams. This module ports that structure into a deterministic
PR106 raw-equivalence fixture. The first promoted variant keeps PR106 f32 scale
bytes so raw decoder equality is exact; fp16-scale variants must be treated as
runtime-parity experiments, not byte-equivalence claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import brotli
import numpy as np

from tac.hnerv_decoder_recode import (
    FIXED_STATE_SCHEMA,
    PACKED_STATE_SCHEMA,
    PackedDecoderRaw,
    PackedDecoderRecord,
    parse_packed_decoder_brotli,
)
from tac.hnerv_lowlevel_packer import (
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.repo_io import repo_relative, sha256_file, write_json

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_pr101_schema_packer.build_pr101_schema_archive_candidate"
VARIANT_NAME = "pr101_schema_split_brotli_f32_scale_raw_equal"

DECODER_STORAGE_ORDER = (
    14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
    18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
)
DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)
CONV4_STORAGE_PERMS = {
    2: (3, 0, 2, 1),
    4: (3, 0, 2, 1),
    6: (0, 1, 2, 3),
    8: (3, 0, 1, 2),
    10: (3, 0, 2, 1),
    12: (3, 0, 1, 2),
    14: (1, 0, 2, 3),
    16: (3, 0, 2, 1),
    18: (1, 0, 2, 3),
    20: (0, 3, 2, 1),
    22: (0, 3, 2, 1),
    24: (0, 2, 3, 1),
    26: (0, 1, 3, 2),
}
CONV4_INVERSE_PERMS = {
    idx: tuple(int(value) for value in np.argsort(perm))
    for idx, perm in CONV4_STORAGE_PERMS.items()
}
DECODER_BYTE_MAPS = {
    9: "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
}


class HnervPr101SchemaPackerError(ValueError):
    """Raised when a PR101 schema-packing input is invalid."""


def encode_pr101_schema_split_fixture(
    parsed: PackedDecoderRaw,
    *,
    quality: int = 11,
) -> tuple[bytes, dict[str, Any]]:
    """Encode PR106 raw decoder records in PR101-style split streams."""

    _validate_schema_constants()
    parts = _schema_record_payloads(parsed)
    streams: list[bytes] = []
    stream_rows: list[dict[str, Any]] = []
    start = 0
    for stream_index, end in enumerate(DECODER_STREAM_ENDS):
        raw = b"".join(parts[start:end])
        compressed = brotli.compress(raw, quality=quality)
        streams.append(compressed)
        stream_rows.append(
            {
                "stream_index": stream_index,
                "record_start": start,
                "record_end": end,
                "record_count": end - start,
                "raw_bytes": len(raw),
                "brotli_bytes": len(compressed),
                "sha256": sha256_bytes(compressed),
            }
        )
        start = end
    payload = b"".join(streams)
    restored = decode_pr101_schema_split_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervPr101SchemaPackerError("PR101 schema split fixture failed raw roundtrip")
    return payload, {
        "schema_version": SCHEMA_VERSION,
        "variant": VARIANT_NAME,
        "brotli_quality": quality,
        "stream_count": len(streams),
        "record_count": len(DECODER_STORAGE_ORDER),
        "scale_bytes_per_record": 4,
        "raw_bytes": sum(len(part) for part in parts),
        "payload_bytes": len(payload),
        "stream_rows": stream_rows,
        "decoder_storage_order": list(DECODER_STORAGE_ORDER),
        "decoder_stream_ends": list(DECODER_STREAM_ENDS),
        "decoder_byte_maps": {str(key): value for key, value in DECODER_BYTE_MAPS.items()},
    }


def decode_pr101_schema_split_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode a PR101-style f32-scale split-stream fixture to PR106 raw records."""

    raw = _decompress_concatenated_brotli_streams(payload, len(DECODER_STREAM_ENDS))
    cursor = 0
    records_by_name: dict[str, PackedDecoderRecord] = {}
    for fixed_index in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[fixed_index]
        value_count = _prod(shape)
        mapped = np.frombuffer(_read_exact(raw, cursor, value_count, f"{name}:q"), dtype=np.uint8)
        cursor += value_count
        scale_f32 = _read_exact(raw, cursor, 4, f"{name}:scale_f32")
        cursor += 4
        q_storage = _decode_mapped_u8(mapped, DECODER_BYTE_MAPS.get(fixed_index, "zig"))
        if len(shape) == 4:
            storage_perm = CONV4_STORAGE_PERMS[fixed_index]
            inverse_perm = CONV4_INVERSE_PERMS[fixed_index]
            stored_shape = tuple(shape[index] for index in storage_perm)
            q_original = np.transpose(q_storage.reshape(stored_shape), inverse_perm).copy()
        else:
            q_original = q_storage.reshape(shape)
        records_by_name[name] = PackedDecoderRecord(
            name=name,
            shape=shape,
            q_zz_u8=_zigzag_encode_i8(q_original.reshape(-1)).tobytes(),
            scale_f32=scale_f32,
        )
    if cursor != len(raw):
        raise HnervPr101SchemaPackerError("PR101 schema fixture has trailing raw bytes")
    records = []
    for name, shape in PACKED_STATE_SCHEMA:
        record = records_by_name.get(name)
        if record is None:
            raise HnervPr101SchemaPackerError(f"missing decoded schema record: {name}")
        if record.shape != shape:
            raise HnervPr101SchemaPackerError(f"decoded schema shape mismatch for {name}")
        records.append(record)
    return PackedDecoderRaw(records=tuple(records))


def build_pr101_schema_archive_candidate(
    *,
    source_archive: str | Path,
    output_dir: str | Path,
    source_label: str,
    allow_rate_regression: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed PR101-style decoder-section archive candidate."""

    source_path = Path(source_archive)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()

    source = read_strict_single_member_zip(source_path)
    packed = parse_ff_packed_brotli_hnerv(source.payload)
    parsed = parse_packed_decoder_brotli(packed.decoder_packed_brotli)
    source_raw = brotli.decompress(packed.decoder_packed_brotli)
    stream, stats = encode_pr101_schema_split_fixture(parsed)
    restored = decode_pr101_schema_split_fixture(stream)
    raw_equal = restored.to_raw() == source_raw
    byte_delta = len(stream) - len(packed.decoder_packed_brotli)
    rate_positive = byte_delta < 0
    blockers: list[str] = []
    if not raw_equal:
        blockers.append("pr101_schema_raw_decoder_mismatch")
    if not rate_positive and not allow_rate_regression:
        blockers.append("pr101_schema_decoder_section_not_rate_positive")

    candidate_archive_path: Path | None = None
    candidate_payload = b""
    candidate_archive_sha = ""
    candidate_archive_bytes: int | None = None
    ready_for_archive_preflight = False
    if not blockers:
        candidate_payload = PackedHnervPayload(
            header=packed.header,
            decoder_packed_brotli=stream,
            latents_and_sidecar_brotli=packed.latents_and_sidecar_brotli,
        ).to_bytes()
        candidate_archive_path = output_root / f"{_slug(source_label)}_pr101_schema_candidate.zip"
        write_stored_single_member_zip(
            candidate_archive_path,
            member_name=source.member_name,
            payload=candidate_payload,
        )
        candidate_archive_sha = sha256_file(candidate_archive_path)
        candidate_archive_bytes = candidate_archive_path.stat().st_size
        checked = parse_ff_packed_brotli_hnerv(candidate_payload)
        if checked.decoder_packed_brotli != stream:
            blockers.append("candidate_decoder_section_not_pr101_schema_stream")
        if checked.latents_and_sidecar_brotli != packed.latents_and_sidecar_brotli:
            blockers.append("candidate_latents_section_changed")
        if candidate_archive_sha == source.archive_sha256:
            blockers.append("candidate_archive_sha256_unchanged")
        ready_for_archive_preflight = not blockers

    runtime_adapter = _runtime_adapter_blocker_report()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_archive_preflight": ready_for_archive_preflight,
        "archive_build_gate": ready_for_archive_preflight,
        "source_label": source_label,
        "source_archive_path": repo_relative(source_path, repo),
        "source_archive_sha256": source.archive_sha256,
        "source_archive_bytes": source.archive_bytes,
        "source_member_name": source.member_name,
        "source_payload_sha256": sha256_bytes(source.payload),
        "source_payload_bytes": len(source.payload),
        "source_decoder_section_sha256": sha256_bytes(packed.decoder_packed_brotli),
        "source_decoder_section_bytes": len(packed.decoder_packed_brotli),
        "source_decoder_raw_sha256": sha256_bytes(source_raw),
        "source_decoder_raw_bytes": len(source_raw),
        "candidate_variant": VARIANT_NAME,
        "candidate_decoder_section_sha256": sha256_bytes(stream),
        "candidate_decoder_section_bytes": len(stream),
        "candidate_decoder_section_byte_delta": byte_delta,
        "candidate_rate_positive": rate_positive,
        "candidate_rate_score_delta_if_runtime_supported_and_components_equal": round(
            byte_delta * (25 / 37_545_489),
            12,
        ),
        "candidate_archive_path": (
            repo_relative(candidate_archive_path, repo) if candidate_archive_path is not None else ""
        ),
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_payload_sha256": sha256_bytes(candidate_payload) if candidate_payload else "",
        "candidate_payload_bytes": len(candidate_payload) if candidate_payload else None,
        "schema_split_stats": stats,
        "decoder_raw_equivalence": {
            "contract": "pr101_schema_decoder_raw_equivalence_v1",
            "source_decoder_raw_sha256": sha256_bytes(source_raw),
            "restored_decoder_raw_sha256": sha256_bytes(restored.to_raw()),
            "raw_equal": raw_equal,
        },
        "runtime_adapter_proof": runtime_adapter,
        "archive_build_blockers": blockers,
        "dispatch_blockers": [
            *blockers,
            *runtime_adapter["dispatch_blockers"],
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }
    write_json(output_root / "pr101_schema_archive_candidate_manifest.json", manifest)
    return manifest


def _schema_record_payloads(parsed: PackedDecoderRaw) -> list[bytes]:
    records_by_name = {record.name: record for record in parsed.records}
    parts: list[bytes] = []
    for fixed_index in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[fixed_index]
        record = records_by_name[name]
        q = _zigzag_decode_u8(np.frombuffer(record.q_zz_u8, dtype=np.uint8)).reshape(shape)
        if len(shape) == 4:
            q = np.transpose(q, CONV4_STORAGE_PERMS[fixed_index]).copy()
        mapped = _encode_mapped_u8(q.reshape(-1), DECODER_BYTE_MAPS.get(fixed_index, "zig"))
        parts.append(mapped.tobytes() + record.scale_f32)
    return parts


def _decompress_concatenated_brotli_streams(payload: bytes, n_streams: int) -> bytes:
    outputs: list[bytes] = []
    cursor = 0
    for _ in range(n_streams):
        decoder = brotli.Decompressor()
        chunks: list[bytes] = []
        while cursor < len(payload) and not decoder.is_finished():
            chunks.append(decoder.process(payload[cursor : cursor + 1]))
            cursor += 1
        if not decoder.is_finished():
            raise HnervPr101SchemaPackerError("truncated PR101 schema Brotli stream")
        outputs.append(b"".join(chunks))
    if cursor != len(payload):
        raise HnervPr101SchemaPackerError("trailing PR101 schema Brotli stream bytes")
    return b"".join(outputs)


def _zigzag_decode_u8(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.int16)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def _zigzag_encode_i8(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.int16)
    return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)


def _encode_mapped_u8(values: np.ndarray, byte_map: str) -> np.ndarray:
    q = values.astype(np.int8, copy=False)
    if byte_map == "zig":
        return _zigzag_encode_i8(q)
    if byte_map == "negzig":
        return _zigzag_encode_i8((-q.astype(np.int16)).astype(np.int8))
    if byte_map == "off":
        return (q.astype(np.int16) + 128).astype(np.uint8)
    if byte_map == "twos":
        return q.view(np.uint8)
    raise HnervPr101SchemaPackerError(f"unknown decoder byte map: {byte_map}")


def _decode_mapped_u8(values: np.ndarray, byte_map: str) -> np.ndarray:
    if byte_map == "zig":
        return _zigzag_decode_u8(values)
    if byte_map == "negzig":
        return (-_zigzag_decode_u8(values).astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (values.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return values.view(np.int8)
    raise HnervPr101SchemaPackerError(f"unknown decoder byte map: {byte_map}")


def _validate_schema_constants() -> None:
    if sorted(DECODER_STORAGE_ORDER) != list(range(len(FIXED_STATE_SCHEMA))):
        raise HnervPr101SchemaPackerError("DECODER_STORAGE_ORDER is not a schema permutation")
    if DECODER_STREAM_ENDS[-1] != len(DECODER_STORAGE_ORDER):
        raise HnervPr101SchemaPackerError("DECODER_STREAM_ENDS does not end at record count")
    if any(left >= right for left, right in zip((0, *DECODER_STREAM_ENDS[:-1]), DECODER_STREAM_ENDS, strict=True)):
        raise HnervPr101SchemaPackerError("DECODER_STREAM_ENDS must be strictly increasing")


def _runtime_adapter_blocker_report() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "pr101_schema_runtime_adapter_proof_v1",
        "content_detection_required": True,
        "legacy_brotli_fallback_required": True,
        "submission_runtime_integrated": False,
        "runtime_tree_parity_manifest_present": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "pr101_schema_submission_runtime_adapter_not_integrated",
            "pr101_schema_runtime_tree_parity_manifest_missing",
            "pr101_schema_inflate_output_parity_missing",
        ],
    }


def _read_exact(payload: bytes, cursor: int, size: int, label: str) -> bytes:
    end = cursor + size
    if end > len(payload):
        raise HnervPr101SchemaPackerError(f"truncated PR101 schema payload at {label}")
    return payload[cursor:end]


def _prod(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return value


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "hnerv"


__all__ = [
    "DECODER_BYTE_MAPS",
    "DECODER_STORAGE_ORDER",
    "DECODER_STREAM_ENDS",
    "VARIANT_NAME",
    "HnervPr101SchemaPackerError",
    "build_pr101_schema_archive_candidate",
    "decode_pr101_schema_split_fixture",
    "encode_pr101_schema_split_fixture",
]
