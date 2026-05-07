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
FP16_SCALE_PROBE_VARIANT_NAME = "pr101_schema_split_brotli_fp16_scale_runtime_probe"
LEGACY_BROTLI_QUALITY = 10
RUNTIME_DECODER_ADAPTER_CONTRACT = "pr101_schema_runtime_decoder_section_adapter_v1"

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


def decode_pr101_schema_or_legacy_decoder_raw(
    decoder_section: bytes,
    *,
    require_pr101_schema: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Return PR106 packed-decoder raw bytes from a supported decoder section.

    This is the runtime-facing adapter contract. It content-detects the
    decoder section as either legacy PR106 single-Brotli bytes or PR101
    schema-split concatenated Brotli bytes. Unknown, ambiguous, truncated, or
    length-invalid sections raise :class:`HnervPr101SchemaPackerError` before
    any score-affecting decode can proceed.
    """

    if not decoder_section:
        raise HnervPr101SchemaPackerError("empty HNeRV decoder section")

    if not require_pr101_schema:
        try:
            raw_decoder = brotli.decompress(decoder_section)
        except brotli.error as legacy_exc:
            legacy_error = str(legacy_exc) or legacy_exc.__class__.__name__
        else:
            _validate_packed_decoder_raw(raw_decoder, label="legacy PR106 Brotli decoder")
            return raw_decoder, _runtime_decoder_section_adapter_proof(
                mode="legacy_pr106_brotli",
                input_decoder=decoder_section,
                output_raw_decoder=raw_decoder,
                legacy_decode_without_adapter=True,
                require_pr101_schema=require_pr101_schema,
            )
    else:
        legacy_error = "skipped because require_pr101_schema=True"

    try:
        restored = decode_pr101_schema_split_fixture(decoder_section)
    except HnervPr101SchemaPackerError as schema_exc:
        if require_pr101_schema:
            raise HnervPr101SchemaPackerError(
                f"decoder section is not PR101 schema-split: {schema_exc}"
            ) from schema_exc
        raise HnervPr101SchemaPackerError(
            "decoder section is neither valid legacy PR106 Brotli nor PR101 "
            f"schema-split; legacy_error={legacy_error}; schema_error={schema_exc}"
        ) from schema_exc

    raw_decoder = restored.to_raw()
    _validate_packed_decoder_raw(raw_decoder, label="PR101 schema-split decoder")
    return raw_decoder, _runtime_decoder_section_adapter_proof(
        mode="pr101_schema_split_brotli",
        input_decoder=decoder_section,
        output_raw_decoder=raw_decoder,
        legacy_decode_without_adapter=False,
        require_pr101_schema=require_pr101_schema,
    )


def restore_pr101_schema_payload_to_legacy_brotli(
    payload: bytes,
    *,
    brotli_quality: int = LEGACY_BROTLI_QUALITY,
    require_pr101_schema: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Return a PR106-compatible payload plus a runtime-adapter proof.

    Legacy PR106 decoder sections pass through unchanged unless
    ``require_pr101_schema`` is true. PR101 schema-split decoder sections are
    decoded to the fixed raw decoder record stream and rewrapped as a legacy
    Brotli decoder section so the public PR106 runtime can consume them.
    Unknown decoder sections fail closed.
    """

    packed = parse_ff_packed_brotli_hnerv(payload)
    decoder = packed.decoder_packed_brotli
    input_decoder_sha = sha256_bytes(decoder)
    input_latents_sha = sha256_bytes(packed.latents_and_sidecar_brotli)

    raw_decoder = b""
    decoder_adapter_proof: dict[str, Any] | None = None
    if not require_pr101_schema:
        try:
            raw_decoder, decoder_adapter_proof = decode_pr101_schema_or_legacy_decoder_raw(
                decoder
            )
        except HnervPr101SchemaPackerError:
            decoder_adapter_proof = None
        else:
            if decoder_adapter_proof["mode"] == "legacy_pr106_brotli":
                proof = _runtime_adapter_payload_proof(
                    mode="legacy_brotli_passthrough",
                    input_payload=payload,
                    output_payload=payload,
                    input_decoder_sha=input_decoder_sha,
                    output_decoder_sha=input_decoder_sha,
                    input_decoder_bytes=len(decoder),
                    output_decoder_bytes=len(decoder),
                    raw_decoder=raw_decoder,
                    restored_raw_decoder=raw_decoder,
                    input_latents_sha=input_latents_sha,
                    output_latents_sha=input_latents_sha,
                    brotli_quality=None,
                    legacy_decode_without_adapter=True,
                    decoder_section_adapter_proof=decoder_adapter_proof,
                )
                return payload, proof

    if decoder_adapter_proof is None or decoder_adapter_proof["mode"] != "pr101_schema_split_brotli":
        try:
            raw_decoder, decoder_adapter_proof = decode_pr101_schema_or_legacy_decoder_raw(
                decoder,
                require_pr101_schema=True,
            )
        except HnervPr101SchemaPackerError as exc:
            if require_pr101_schema:
                raise HnervPr101SchemaPackerError(
                    f"payload decoder section is not PR101 schema-split: {exc}"
                ) from exc
            raise HnervPr101SchemaPackerError(
                "decoder section is neither legacy Brotli nor PR101 schema-split"
            ) from exc

    restored_decoder = brotli.compress(raw_decoder, quality=brotli_quality)
    try:
        restored_raw = brotli.decompress(restored_decoder)
    except brotli.error as exc:  # pragma: no cover - brotli self-check guard
        raise HnervPr101SchemaPackerError(
            "restored legacy PR101 schema Brotli is not decompressible"
        ) from exc
    if restored_raw != raw_decoder:
        raise HnervPr101SchemaPackerError("restored legacy PR101 schema raw mismatch")

    restored_payload = PackedHnervPayload(
        header=packed.header,
        decoder_packed_brotli=restored_decoder,
        latents_and_sidecar_brotli=packed.latents_and_sidecar_brotli,
    ).to_bytes()
    proof = _runtime_adapter_payload_proof(
        mode="pr101_schema_restored_to_legacy_brotli",
        input_payload=payload,
        output_payload=restored_payload,
        input_decoder_sha=input_decoder_sha,
        output_decoder_sha=sha256_bytes(restored_decoder),
        input_decoder_bytes=len(decoder),
        output_decoder_bytes=len(restored_decoder),
        raw_decoder=raw_decoder,
        restored_raw_decoder=restored_raw,
        input_latents_sha=input_latents_sha,
        output_latents_sha=sha256_bytes(packed.latents_and_sidecar_brotli),
        brotli_quality=brotli_quality,
        legacy_decode_without_adapter=False,
        decoder_section_adapter_proof=decoder_adapter_proof,
    )
    return restored_payload, proof


def encode_pr101_schema_split_fixture(
    parsed: PackedDecoderRaw,
    *,
    quality: int = 11,
) -> tuple[bytes, dict[str, Any]]:
    """Encode PR106 raw decoder records in PR101-style split streams."""

    _validate_schema_constants()
    parts = _schema_record_payloads(parsed)
    payload, stats = _compress_schema_parts(
        parts,
        variant=VARIANT_NAME,
        quality=quality,
        scale_bytes_per_record=4,
    )
    restored = decode_pr101_schema_split_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervPr101SchemaPackerError("PR101 schema split fixture failed raw roundtrip")
    stats["decoder_storage_order"] = list(DECODER_STORAGE_ORDER)
    stats["decoder_stream_ends"] = list(DECODER_STREAM_ENDS)
    stats["decoder_byte_maps"] = {str(key): value for key, value in DECODER_BYTE_MAPS.items()}
    return payload, stats


def encode_pr101_schema_split_fp16_scale_probe(
    parsed: PackedDecoderRaw,
    *,
    quality: int = 11,
) -> tuple[bytes, dict[str, Any]]:
    """Encode a PR101-native fp16-scale probe without raw-equivalence claims.

    PR101 stores one fp16 scale per tensor. Porting that to PR106 changes the
    reconstructed decoder weights, so this probe is a numerical/runtime parity
    target, not a byte-equivalent replacement.
    """

    _validate_schema_constants()
    parts, scale_rows = _schema_record_payloads_fp16_scale_probe(parsed)
    payload, stats = _compress_schema_parts(
        parts,
        variant=FP16_SCALE_PROBE_VARIANT_NAME,
        quality=quality,
        scale_bytes_per_record=2,
    )
    restored = decode_pr101_schema_split_fp16_scale_probe(payload)
    q_equal = restored.q_stream == parsed.q_stream
    if not q_equal:
        raise HnervPr101SchemaPackerError("PR101 fp16-scale probe failed q roundtrip")
    stats["q_roundtrip_equal"] = q_equal
    stats["scale_raw_equal"] = restored.scale_stream == parsed.scale_stream
    stats["scale_rows"] = scale_rows
    stats["max_abs_scale_error"] = max(row["abs_scale_error"] for row in scale_rows)
    stats["max_rel_scale_error"] = max(row["rel_scale_error"] for row in scale_rows)
    stats["max_abs_weight_error_bound"] = max(
        row["max_abs_weight_error_bound"] for row in scale_rows
    )
    stats["decoder_storage_order"] = list(DECODER_STORAGE_ORDER)
    stats["decoder_stream_ends"] = list(DECODER_STREAM_ENDS)
    stats["decoder_byte_maps"] = {str(key): value for key, value in DECODER_BYTE_MAPS.items()}
    result = {
        **stats,
        "runtime_parity_required": True,
        "raw_equivalence_claim": False,
        "dispatch_blockers": [
            "pr101_fp16_scale_runtime_adapter_not_integrated",
            "pr101_fp16_scale_inflate_output_parity_missing",
            "pr101_fp16_scale_exact_cuda_auth_eval_missing",
        ],
    }
    result["hidden_gem_classification"] = _fp16_probe_hidden_gem_classification(result)
    return payload, result


def decode_pr101_schema_split_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode a PR101-style f32-scale split-stream fixture to PR106 raw records."""

    return _decode_pr101_schema_split(payload, scale_bytes_per_record=4)


def decode_pr101_schema_split_fp16_scale_probe(payload: bytes) -> PackedDecoderRaw:
    """Decode a PR101-native fp16-scale probe to PR106-shaped records."""

    return _decode_pr101_schema_split(payload, scale_bytes_per_record=2)


def _decode_pr101_schema_split(payload: bytes, *, scale_bytes_per_record: int) -> PackedDecoderRaw:
    raw = _decompress_concatenated_brotli_streams(payload, len(DECODER_STREAM_ENDS))
    cursor = 0
    records_by_name: dict[str, PackedDecoderRecord] = {}
    for fixed_index in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[fixed_index]
        value_count = _prod(shape)
        mapped = np.frombuffer(_read_exact(raw, cursor, value_count, f"{name}:q"), dtype=np.uint8)
        cursor += value_count
        scale_raw = _read_exact(raw, cursor, scale_bytes_per_record, f"{name}:scale")
        cursor += scale_bytes_per_record
        if scale_bytes_per_record == 4:
            scale_f32 = scale_raw
        elif scale_bytes_per_record == 2:
            scale_f32 = _scale_fp16_bytes_to_f32_bytes(scale_raw)
        else:
            raise HnervPr101SchemaPackerError(
                f"unsupported PR101 schema scale byte width: {scale_bytes_per_record}"
            )
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
    emit_fp16_probe_archive: bool = False,
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
    fp16_probe_stream, fp16_probe_stats = encode_pr101_schema_split_fp16_scale_probe(parsed)
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

    fp16_probe_archive_path: Path | None = None
    fp16_probe_archive_sha = ""
    fp16_probe_archive_bytes: int | None = None
    fp16_probe_payload = b""
    if emit_fp16_probe_archive:
        fp16_probe_payload = PackedHnervPayload(
            header=packed.header,
            decoder_packed_brotli=fp16_probe_stream,
            latents_and_sidecar_brotli=packed.latents_and_sidecar_brotli,
        ).to_bytes()
        fp16_probe_archive_path = (
            output_root / f"{_slug(source_label)}_pr101_schema_fp16_scale_probe.zip"
        )
        write_stored_single_member_zip(
            fp16_probe_archive_path,
            member_name=source.member_name,
            payload=fp16_probe_payload,
        )
        fp16_probe_archive_sha = sha256_file(fp16_probe_archive_path)
        fp16_probe_archive_bytes = fp16_probe_archive_path.stat().st_size

    runtime_adapter = _runtime_adapter_blocker_report(
        source_payload=source.payload,
        candidate_payload=candidate_payload,
        candidate_decoder_raw_equal=raw_equal,
        candidate_latents_preserved=(
            bool(candidate_payload)
            and parse_ff_packed_brotli_hnerv(candidate_payload).latents_and_sidecar_brotli
            == packed.latents_and_sidecar_brotli
        ),
    )
    runtime_decoder_adapter = _runtime_decoder_section_adapter_report(
        source_decoder_section=packed.decoder_packed_brotli,
        candidate_decoder_section=stream,
        candidate_decoder_raw_equal=raw_equal,
    )
    fp16_probe_delta = len(fp16_probe_stream) - len(packed.decoder_packed_brotli)
    dispatch_blockers = list(
        dict.fromkeys(
            [
                *blockers,
                *runtime_adapter["dispatch_blockers"],
                *runtime_decoder_adapter["integration_blockers"],
                "strict_pre_submission_compliance_json_missing",
                "lane_dispatch_claim_missing",
                "exact_cuda_auth_eval_missing",
            ]
        )
    )
    hidden_gem_classification = _f32_hidden_gem_classification(
        byte_delta=byte_delta,
        raw_equal=raw_equal,
        runtime_adapter=runtime_adapter,
        dispatch_blockers=dispatch_blockers,
    )
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
        "fp16_scale_probe": {
            "variant": FP16_SCALE_PROBE_VARIANT_NAME,
            "score_claim": False,
            "ready_for_archive_preflight": False,
            "ready_for_exact_eval_dispatch": False,
            "source_decoder_section_bytes": len(packed.decoder_packed_brotli),
            "probe_decoder_section_sha256": sha256_bytes(fp16_probe_stream),
            "probe_decoder_section_bytes": len(fp16_probe_stream),
            "probe_decoder_section_byte_delta_vs_source": fp16_probe_delta,
            "probe_decoder_section_byte_delta_vs_f32_schema": len(fp16_probe_stream) - len(stream),
            "probe_archive_bytes_if_materialized": (
                source.archive_bytes + fp16_probe_delta if not emit_fp16_probe_archive else fp16_probe_archive_bytes
            ),
            "probe_archive_path": (
                repo_relative(fp16_probe_archive_path, repo)
                if fp16_probe_archive_path is not None
                else ""
            ),
            "probe_archive_sha256": fp16_probe_archive_sha,
            "probe_payload_sha256": sha256_bytes(fp16_probe_payload) if fp16_probe_payload else "",
            "probe_payload_bytes": len(fp16_probe_payload) if fp16_probe_payload else None,
            "rate_score_delta_if_runtime_supported_and_components_equal": round(
                fp16_probe_delta * (25 / 37_545_489),
                12,
            ),
            "q_roundtrip_equal": fp16_probe_stats["q_roundtrip_equal"],
            "scale_raw_equal": fp16_probe_stats["scale_raw_equal"],
            "max_abs_scale_error": fp16_probe_stats["max_abs_scale_error"],
            "max_rel_scale_error": fp16_probe_stats["max_rel_scale_error"],
            "max_abs_weight_error_bound": fp16_probe_stats["max_abs_weight_error_bound"],
            "stats": fp16_probe_stats,
            "dispatch_blockers": fp16_probe_stats["dispatch_blockers"],
            "hidden_gem_classification": fp16_probe_stats["hidden_gem_classification"],
        },
        "decoder_raw_equivalence": {
            "contract": "pr101_schema_decoder_raw_equivalence_v1",
            "source_decoder_raw_sha256": sha256_bytes(source_raw),
            "restored_decoder_raw_sha256": sha256_bytes(restored.to_raw()),
            "raw_equal": raw_equal,
        },
        "runtime_adapter_proof": runtime_adapter,
        "runtime_decoder_section_adapter": runtime_decoder_adapter,
        "hidden_gem_classification": hidden_gem_classification,
        "archive_build_blockers": blockers,
        "dispatch_blockers": dispatch_blockers,
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


def _schema_record_payloads_fp16_scale_probe(
    parsed: PackedDecoderRaw,
) -> tuple[list[bytes], list[dict[str, Any]]]:
    records_by_name = {record.name: record for record in parsed.records}
    parts: list[bytes] = []
    rows: list[dict[str, Any]] = []
    for fixed_index in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[fixed_index]
        record = records_by_name[name]
        q = _zigzag_decode_u8(np.frombuffer(record.q_zz_u8, dtype=np.uint8)).reshape(shape)
        if len(shape) == 4:
            q = np.transpose(q, CONV4_STORAGE_PERMS[fixed_index]).copy()
        mapped = _encode_mapped_u8(q.reshape(-1), DECODER_BYTE_MAPS.get(fixed_index, "zig"))
        scale_fp16 = _scale_f32_bytes_to_fp16_bytes(record.scale_f32)
        source_scale = _scale_f32_bytes_to_float(record.scale_f32)
        restored_scale = _scale_f32_bytes_to_float(_scale_fp16_bytes_to_f32_bytes(scale_fp16))
        abs_scale_error = abs(restored_scale - source_scale)
        rel_scale_error = abs_scale_error / abs(source_scale) if source_scale else abs_scale_error
        max_abs_q = int(np.max(np.abs(q.astype(np.int16)))) if q.size else 0
        rows.append(
            {
                "fixed_schema_index": fixed_index,
                "name": name,
                "shape": list(shape),
                "q_value_count": int(q.size),
                "max_abs_q": max_abs_q,
                "source_scale_f32": source_scale,
                "restored_scale_from_fp16": restored_scale,
                "abs_scale_error": abs_scale_error,
                "rel_scale_error": rel_scale_error,
                "max_abs_weight_error_bound": abs_scale_error * max_abs_q,
            }
        )
        parts.append(mapped.tobytes() + scale_fp16)
    return parts, rows


def _compress_schema_parts(
    parts: list[bytes],
    *,
    variant: str,
    quality: int,
    scale_bytes_per_record: int,
) -> tuple[bytes, dict[str, Any]]:
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
    return payload, {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "brotli_quality": quality,
        "stream_count": len(streams),
        "record_count": len(DECODER_STORAGE_ORDER),
        "scale_bytes_per_record": scale_bytes_per_record,
        "raw_bytes": sum(len(part) for part in parts),
        "payload_bytes": len(payload),
        "stream_rows": stream_rows,
    }


def _decompress_concatenated_brotli_streams(payload: bytes, n_streams: int) -> bytes:
    outputs: list[bytes] = []
    cursor = 0
    for _ in range(n_streams):
        decoder = brotli.Decompressor()
        chunks: list[bytes] = []
        try:
            while cursor < len(payload) and not decoder.is_finished():
                chunks.append(decoder.process(payload[cursor : cursor + 1]))
                cursor += 1
        except brotli.error as exc:
            raise HnervPr101SchemaPackerError(
                "invalid PR101 schema Brotli stream"
            ) from exc
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


def _scale_f32_bytes_to_float(scale_f32: bytes) -> float:
    if len(scale_f32) != 4:
        raise HnervPr101SchemaPackerError("f32 scale must be exactly 4 bytes")
    value = float(np.frombuffer(scale_f32, dtype="<f4", count=1)[0])
    if not np.isfinite(value):
        raise HnervPr101SchemaPackerError("non-finite f32 scale")
    return value


def _scale_f32_bytes_to_fp16_bytes(scale_f32: bytes) -> bytes:
    value = _scale_f32_bytes_to_float(scale_f32)
    return np.asarray([np.float16(value)], dtype="<f2").tobytes()


def _scale_fp16_bytes_to_f32_bytes(scale_fp16: bytes) -> bytes:
    if len(scale_fp16) != 2:
        raise HnervPr101SchemaPackerError("fp16 scale must be exactly 2 bytes")
    value = float(np.frombuffer(scale_fp16, dtype="<f2", count=1)[0])
    if not np.isfinite(value):
        raise HnervPr101SchemaPackerError("non-finite fp16 scale")
    return np.asarray([np.float32(value)], dtype="<f4").tobytes()


def _validate_schema_constants() -> None:
    if sorted(DECODER_STORAGE_ORDER) != list(range(len(FIXED_STATE_SCHEMA))):
        raise HnervPr101SchemaPackerError("DECODER_STORAGE_ORDER is not a schema permutation")
    if DECODER_STREAM_ENDS[-1] != len(DECODER_STORAGE_ORDER):
        raise HnervPr101SchemaPackerError("DECODER_STREAM_ENDS does not end at record count")
    if any(left >= right for left, right in zip((0, *DECODER_STREAM_ENDS[:-1]), DECODER_STREAM_ENDS, strict=True)):
        raise HnervPr101SchemaPackerError("DECODER_STREAM_ENDS must be strictly increasing")


def _runtime_adapter_blocker_report(
    *,
    source_payload: bytes,
    candidate_payload: bytes,
    candidate_decoder_raw_equal: bool,
    candidate_latents_preserved: bool,
) -> dict[str, Any]:
    source_restored, source_proof = restore_pr101_schema_payload_to_legacy_brotli(source_payload)
    candidate_proof: dict[str, Any] = {
        "provided": False,
        "ready_for_public_runtime_inflate": False,
        "remaining_dispatch_blockers": ["candidate_archive_not_materialized"],
    }
    candidate_restored = b""
    if candidate_payload:
        candidate_restored, candidate_proof = restore_pr101_schema_payload_to_legacy_brotli(
            candidate_payload,
            require_pr101_schema=True,
        )
        candidate_proof = {"provided": True, **candidate_proof}
    adapter_parity_closed = (
        bool(candidate_payload)
        and candidate_decoder_raw_equal
        and candidate_latents_preserved
        and source_proof["decoder_raw_sha256"] == candidate_proof.get("decoder_raw_sha256")
        and source_proof["latents_and_sidecar_input_sha256"]
        == candidate_proof.get("latents_and_sidecar_input_sha256")
        and candidate_proof.get("ready_for_public_runtime_inflate") is True
    )
    dispatch_blockers = [
        "pr101_schema_submission_runtime_adapter_not_integrated",
        "pr101_schema_runtime_tree_parity_manifest_missing",
        "pr101_schema_inflate_output_parity_missing",
    ]
    if not candidate_payload:
        dispatch_blockers.insert(0, "pr101_schema_candidate_archive_not_materialized")
    if bool(candidate_payload) and not adapter_parity_closed:
        dispatch_blockers.insert(0, "pr101_schema_payload_adapter_parity_not_closed")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "pr101_schema_runtime_adapter_proof_v1",
        "content_detection_required": True,
        "legacy_brotli_fallback_required": True,
        "legacy_brotli_quality": LEGACY_BROTLI_QUALITY,
        "source_legacy_payload_proof": source_proof,
        "candidate_schema_payload_proof": candidate_proof,
        "source_restored_payload_sha256": sha256_bytes(source_restored),
        "candidate_restored_payload_sha256": (
            sha256_bytes(candidate_restored) if candidate_restored else ""
        ),
        "candidate_decoder_raw_equal_to_source": candidate_decoder_raw_equal,
        "candidate_latents_preserved": candidate_latents_preserved,
        "candidate_legacy_decode_without_adapter": bool(
            candidate_proof.get("legacy_decode_without_adapter")
        ),
        "schema_payload_adapter_parity_closed": adapter_parity_closed,
        "ready_for_public_runtime_inflate": adapter_parity_closed,
        "submission_runtime_integrated": False,
        "runtime_tree_parity_manifest_present": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": dispatch_blockers,
    }


def _runtime_decoder_section_adapter_report(
    *,
    source_decoder_section: bytes,
    candidate_decoder_section: bytes,
    candidate_decoder_raw_equal: bool,
) -> dict[str, Any]:
    source_raw = b""
    candidate_raw = b""
    blockers: list[str] = []
    try:
        source_raw, source_proof = decode_pr101_schema_or_legacy_decoder_raw(
            source_decoder_section
        )
    except HnervPr101SchemaPackerError as exc:
        source_proof = {
            "provided": False,
            "error": str(exc),
            "runtime_consumable_decoder_raw": False,
        }
        blockers.append("pr101_schema_source_decoder_adapter_rejected")
    try:
        candidate_raw, candidate_proof = decode_pr101_schema_or_legacy_decoder_raw(
            candidate_decoder_section,
            require_pr101_schema=True,
        )
    except HnervPr101SchemaPackerError as exc:
        candidate_proof = {
            "provided": False,
            "error": str(exc),
            "runtime_consumable_decoder_raw": False,
        }
        blockers.append("pr101_schema_candidate_decoder_adapter_rejected")

    candidate_raw_equal_to_source = (
        bool(source_raw)
        and bool(candidate_raw)
        and sha256_bytes(source_raw) == sha256_bytes(candidate_raw)
        and candidate_decoder_raw_equal
    )
    adapter_code_path_ready = (
        candidate_raw_equal_to_source
        and source_proof.get("runtime_consumable_decoder_raw") is True
        and candidate_proof.get("runtime_consumable_decoder_raw") is True
    )
    if not adapter_code_path_ready:
        blockers.append("pr101_schema_runtime_decoder_adapter_not_closed")
    integration_blockers = [
        *blockers,
        "pr101_schema_submission_runtime_adapter_not_integrated",
        "pr101_schema_submission_runtime_callsite_not_patched",
        "pr101_schema_runtime_tree_parity_manifest_missing",
        "pr101_schema_inflate_output_parity_missing",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RUNTIME_DECODER_ADAPTER_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "content_detection_order": [
            "legacy_pr106_single_brotli",
            "pr101_schema_split_concatenated_brotli",
        ],
        "fail_closed_unknown_decoder_section": True,
        "source_decoder_adapter_proof": source_proof,
        "candidate_decoder_adapter_proof": candidate_proof,
        "source_decoder_raw_sha256": sha256_bytes(source_raw) if source_raw else "",
        "candidate_decoder_raw_sha256": (
            sha256_bytes(candidate_raw) if candidate_raw else ""
        ),
        "candidate_decoder_raw_equal_to_source": candidate_raw_equal_to_source,
        "adapter_code_path_ready": adapter_code_path_ready,
        "ready_for_runtime_callsite_patch": adapter_code_path_ready,
        "runtime_patch_owned_by_this_turn": False,
        "submission_runtime_patch_applied": False,
        "runtime_tree_parity_manifest_present": False,
        "inflate_output_parity_present": False,
        "ready_for_exact_eval_dispatch": False,
        "integration_targets": [
            "submissions/pr106_latent_sidecar/src/codec.py::decode_packed_decoder",
            "submissions/apogee_v2/src/codec.py::decode_packed_decoder",
        ],
        "integration_blockers": list(dict.fromkeys(integration_blockers)),
    }


def _runtime_adapter_payload_proof(
    *,
    mode: str,
    input_payload: bytes,
    output_payload: bytes,
    input_decoder_sha: str,
    output_decoder_sha: str,
    input_decoder_bytes: int,
    output_decoder_bytes: int,
    raw_decoder: bytes,
    restored_raw_decoder: bytes,
    input_latents_sha: str,
    output_latents_sha: str,
    brotli_quality: int | None,
    legacy_decode_without_adapter: bool,
    decoder_section_adapter_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_sha = sha256_bytes(raw_decoder)
    restored_raw_sha = sha256_bytes(restored_raw_decoder)
    raw_equal = raw_sha == restored_raw_sha
    latents_preserved = input_latents_sha == output_latents_sha
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "pr101_schema_payload_runtime_adapter_parity_v1",
        "mode": mode,
        "score_claim": False,
        "dispatch_attempted": False,
        "legacy_decode_without_adapter": legacy_decode_without_adapter,
        "input_payload_sha256": sha256_bytes(input_payload),
        "input_payload_bytes": len(input_payload),
        "output_payload_sha256": sha256_bytes(output_payload),
        "output_payload_bytes": len(output_payload),
        "payload_changed": output_payload != input_payload,
        "input_decoder_section_sha256": input_decoder_sha,
        "input_decoder_section_bytes": input_decoder_bytes,
        "output_decoder_section_sha256": output_decoder_sha,
        "output_decoder_section_bytes": output_decoder_bytes,
        "decoder_section_byte_delta_runtime_only": output_decoder_bytes - input_decoder_bytes,
        "decoder_raw_sha256": raw_sha,
        "restored_decoder_raw_sha256": restored_raw_sha,
        "decoder_raw_equal": raw_equal,
        "latents_and_sidecar_input_sha256": input_latents_sha,
        "latents_and_sidecar_output_sha256": output_latents_sha,
        "latents_and_sidecar_preserved": latents_preserved,
        "legacy_brotli_quality": brotli_quality,
        "decoder_section_adapter_proof": decoder_section_adapter_proof or {},
        "ready_for_public_runtime_inflate": raw_equal and latents_preserved,
        "ready_for_exact_eval_dispatch": False,
        "remaining_dispatch_blockers": [
            "exact_inflate_output_parity_missing",
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }


def _f32_hidden_gem_classification(
    *,
    byte_delta: int,
    raw_equal: bool,
    runtime_adapter: dict[str, Any],
    dispatch_blockers: list[str],
) -> dict[str, Any]:
    stack_candidate = byte_delta < 0 and raw_equal
    return {
        "candidate_id": "PR101-schema-f32-on-PR106x",
        "classification": "stack_candidate" if stack_candidate else "blocked_schema_probe",
        "classification_evidence_grade": "empirical",
        "evidence_semantics": "byte_equivalence_only_raw_decoder_equal_no_score_claim",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "stack_candidate": stack_candidate,
        "substitute_candidate": False,
        "runtime_adapter_payload_parity_closed": bool(
            runtime_adapter.get("schema_payload_adapter_parity_closed")
        ),
        "rationale": (
            "Raw decoder bytes are restored exactly while charged bytes shrink; "
            "this is a pack/layout component that can stack after runtime-adapter "
            "integration and exact CUDA auth eval."
            if stack_candidate
            else "The schema stream did not close the raw-equal, rate-positive stack contract."
        ),
        "next_actions": [
            "integrate fail-closed runtime content detection for the PR101 schema stream",
            "record runtime-tree parity manifest and inflate-output parity",
            "run strict pre-submission compliance before any lane claim or CUDA dispatch",
        ],
        "blockers": list(dict.fromkeys(dispatch_blockers)),
    }


def _fp16_probe_hidden_gem_classification(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": "PR101-fp16-scale-on-PR106x",
        "classification": "substitute_candidate_probe",
        "classification_evidence_grade": "empirical",
        "evidence_semantics": "q_roundtrip_equal_but_scale_raw_not_equal_no_score_claim",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "stack_candidate": False,
        "substitute_candidate": True,
        "conditional_stack_candidate_after_exact_cuda": True,
        "rationale": (
            "The q-stream roundtrips but fp16 scales change restored weights, so this "
            "is a scorer-facing substitute probe rather than a lossless pack-only row."
        ),
        "next_actions": [
            "prove inflate-output parity or classify the scale quantization delta before dispatch",
            "integrate the runtime adapter with legacy fallback",
            "only compare score after exact CUDA auth eval on exact archive bytes",
        ],
        "blockers": list(stats.get("dispatch_blockers") or []),
    }


def _read_exact(payload: bytes, cursor: int, size: int, label: str) -> bytes:
    end = cursor + size
    if end > len(payload):
        raise HnervPr101SchemaPackerError(f"truncated PR101 schema payload at {label}")
    return payload[cursor:end]


def _validate_packed_decoder_raw(raw_decoder: bytes, *, label: str) -> None:
    q_bytes = sum(_prod(shape) for _name, shape in PACKED_STATE_SCHEMA)
    expected = q_bytes + 4 * len(PACKED_STATE_SCHEMA)
    if len(raw_decoder) != expected:
        raise HnervPr101SchemaPackerError(
            f"{label} raw length mismatch: expected {expected}, got {len(raw_decoder)}"
        )
    scales = np.frombuffer(raw_decoder[q_bytes:], dtype="<f4")
    if scales.size != len(PACKED_STATE_SCHEMA):
        raise HnervPr101SchemaPackerError(
            f"{label} scale count mismatch: expected {len(PACKED_STATE_SCHEMA)}, got {scales.size}"
        )
    if not np.all(np.isfinite(scales)):
        raise HnervPr101SchemaPackerError(f"{label} contains non-finite decoder scales")


def _runtime_decoder_section_adapter_proof(
    *,
    mode: str,
    input_decoder: bytes,
    output_raw_decoder: bytes,
    legacy_decode_without_adapter: bool,
    require_pr101_schema: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RUNTIME_DECODER_ADAPTER_CONTRACT,
        "mode": mode,
        "score_claim": False,
        "dispatch_attempted": False,
        "require_pr101_schema": require_pr101_schema,
        "legacy_decode_without_adapter": legacy_decode_without_adapter,
        "input_decoder_section_sha256": sha256_bytes(input_decoder),
        "input_decoder_section_bytes": len(input_decoder),
        "output_decoder_raw_sha256": sha256_bytes(output_raw_decoder),
        "output_decoder_raw_bytes": len(output_raw_decoder),
        "expected_decoder_raw_bytes": sum(
            _prod(shape) for _name, shape in PACKED_STATE_SCHEMA
        )
        + 4 * len(PACKED_STATE_SCHEMA),
        "runtime_consumable_decoder_raw": True,
        "fail_closed_unknown_decoder_section": True,
        "adapter_code_path_ready": True,
        "ready_for_exact_eval_dispatch": False,
        "remaining_integration_blockers": [
            "submission_runtime_decode_packed_decoder_callsite_not_patched",
            "runtime_tree_parity_manifest_missing",
            "inflate_output_parity_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }


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
    "FP16_SCALE_PROBE_VARIANT_NAME",
    "VARIANT_NAME",
    "HnervPr101SchemaPackerError",
    "build_pr101_schema_archive_candidate",
    "decode_pr101_schema_or_legacy_decoder_raw",
    "decode_pr101_schema_split_fixture",
    "decode_pr101_schema_split_fp16_scale_probe",
    "encode_pr101_schema_split_fixture",
    "encode_pr101_schema_split_fp16_scale_probe",
    "restore_pr101_schema_payload_to_legacy_brotli",
]
