#!/usr/bin/env python3
"""Probe pre-coarsening entropy coders for PR101/PR106 HNeRV int8 weights.

This is a no-score, CPU-only byte accounting tool. It reads exact public
frontier archives already present in the repo, parses the renderer decoder
weight streams before any lossy coarsening, and compares:

* exact/source decoder section bytes;
* Brotli q11 on the source logical raw layout;
* Brotli q11 on a fixed-schema canonical int8 stream;
* a static-table AC/rANS/FSE proxy with charged model headers; and
* a practical constriction range-coder packet when constriction is available.

The output is a reactivation screen only. It does not write candidate archives,
does not load scorers, does not use CUDA, and does not claim score movement.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.hnerv_lowlevel_packer import parse_ff_packed_brotli_hnerv  # noqa: E402
from tac.pr101_split_brotli_codec import (  # noqa: E402
    CONV4_STORAGE_PERMS,
    DECODER_BLOB_LEN,
    DECODER_BYTE_MAPS,
    DECODER_STORAGE_ORDER,
    DECODER_STREAM_ENDS,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    decode_mapped_u8,
    decompress_brotli_streams,
)

TOOL = "tools/probe_precoarsening_entropy_coders.py"
SCHEMA = "precoarsening_entropy_coder_probe_v1"
N_CATEGORIES = 255
STATIC_MODEL_SMOOTHING = 1
FIXED_PACKET_HEADER_BYTES = 8
STREAM_LENGTH_BYTES = 4
FREQ_ENTRY_BYTES = 2

DEFAULT_PR101 = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
DEFAULT_PR106 = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)


@dataclasses.dataclass(frozen=True)
class TensorStream:
    schema_index: int
    name: str
    shape: tuple[int, ...]
    q_i8: np.ndarray
    scale_bytes: bytes
    source_symbol_bytes: bytes

    @property
    def symbol_count(self) -> int:
        return int(self.q_i8.size)

    @property
    def scale_bytes_len(self) -> int:
        return len(self.scale_bytes)


@dataclasses.dataclass(frozen=True)
class ParsedDecoder:
    source_format: str
    decoder_section: bytes
    source_layout_raw: bytes
    source_layout_brotli_q11: bytes
    canonical_records: tuple[TensorStream, ...]
    logical_sections: tuple[dict[str, Any], ...]
    source_layout_note: str

    @property
    def q_symbol_count(self) -> int:
        return sum(record.symbol_count for record in self.canonical_records)

    @property
    def scale_bytes(self) -> int:
        return sum(record.scale_bytes_len for record in self.canonical_records)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_single_member_archive(path: Path) -> dict[str, Any]:
    archive_bytes = path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise SystemExit(f"{path} has {len(infos)} non-directory members; expected one")
        info = infos[0]
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"{path} failed ZIP CRC validation at member {bad!r}")
        payload = zf.read(info.filename)
    return {
        "archive_path": _repo_rel(path),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": sha256_bytes(archive_bytes),
        "member_name": info.filename,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "zip_overhead_bytes": len(archive_bytes) - len(payload),
        "payload": payload,
    }


def _zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    arr = arr_i8.astype(np.int16)
    return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)


def _zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    arr = arr_u8.astype(np.int16)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def _symbols_0_254(q_i8: np.ndarray) -> np.ndarray:
    symbols = q_i8.astype(np.int16) + 127
    if symbols.size and (int(symbols.min()) < 0 or int(symbols.max()) > 254):
        raise ValueError(
            f"int8 symbols outside signed-7-bit range: "
            f"min={int(q_i8.min())} max={int(q_i8.max())}"
        )
    return symbols.astype(np.uint8)


def _canonical_precoarsening_bytes(records: tuple[TensorStream, ...]) -> bytes:
    parts: list[bytes] = []
    for record in sorted(records, key=lambda item: item.schema_index):
        parts.append(_symbols_0_254(record.q_i8).tobytes())
        parts.append(record.scale_bytes)
    return b"".join(parts)


def _packed_state_schema() -> tuple[tuple[int, str, tuple[int, ...]], ...]:
    name_to_index = {name: idx for idx, (name, _shape) in enumerate(FIXED_STATE_SCHEMA)}
    rows = sorted(FIXED_STATE_SCHEMA, key=lambda item: -math.prod(item[1]))
    return tuple((name_to_index[name], name, shape) for name, shape in rows)


def _parse_pr101_decoder(payload: bytes) -> ParsedDecoder:
    if len(payload) < DECODER_BLOB_LEN + LATENT_BLOB_LEN:
        raise SystemExit("PR101 payload is shorter than fixed decoder+latent offsets")
    decoder = payload[:DECODER_BLOB_LEN]
    latent = payload[DECODER_BLOB_LEN : DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar = payload[DECODER_BLOB_LEN + LATENT_BLOB_LEN :]

    raw = decompress_brotli_streams(decoder, len(DECODER_STREAM_ENDS))
    cursor = 0
    records_by_index: dict[int, TensorStream] = {}
    source_parts_by_storage: list[bytes] = []
    for schema_index in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[schema_index]
        count = math.prod(shape)
        mapped = np.frombuffer(raw, dtype=np.uint8, count=count, offset=cursor).copy()
        cursor += count
        scale_bytes = raw[cursor : cursor + 2]
        cursor += 2

        q_storage = decode_mapped_u8(mapped, DECODER_BYTE_MAPS.get(schema_index, "zig"))
        if len(shape) == 4 and schema_index in CONV4_STORAGE_PERMS:
            storage_perm = CONV4_STORAGE_PERMS[schema_index]
            stored_shape = tuple(shape[i] for i in storage_perm)
            inverse_perm = tuple(int(v) for v in np.argsort(storage_perm))
            q = q_storage.reshape(stored_shape)
            q = np.transpose(q, inverse_perm).copy()
        else:
            q = q_storage.reshape(shape).copy()
        source_payload = mapped.tobytes() + scale_bytes
        source_parts_by_storage.append(source_payload)
        records_by_index[schema_index] = TensorStream(
            schema_index=schema_index,
            name=name,
            shape=shape,
            q_i8=q.reshape(-1).astype(np.int8, copy=True),
            scale_bytes=scale_bytes,
            source_symbol_bytes=mapped.tobytes(),
        )

    if cursor != len(raw):
        raise SystemExit(f"PR101 compact decoder raw parse left trailing bytes: {len(raw) - cursor}")

    streams: list[bytes] = []
    start = 0
    for end in DECODER_STREAM_ENDS:
        window = b"".join(source_parts_by_storage[start:end])
        streams.append(brotli.compress(window, quality=11))
        start = end

    return ParsedDecoder(
        source_format="pr101_split_brotli_fixed_offsets",
        decoder_section=decoder,
        source_layout_raw=raw,
        source_layout_brotli_q11=b"".join(streams),
        canonical_records=tuple(records_by_index[idx] for idx in range(len(FIXED_STATE_SCHEMA))),
        logical_sections=(
            {
                "name": "decoder_blob",
                "offset": 0,
                "bytes": len(decoder),
                "sha256": sha256_bytes(decoder),
                "role": "renderer_decoder_weights",
            },
            {
                "name": "latent_blob",
                "offset": DECODER_BLOB_LEN,
                "bytes": len(latent),
                "sha256": sha256_bytes(latent),
                "role": "latents",
            },
            {
                "name": "sidecar_blob",
                "offset": DECODER_BLOB_LEN + LATENT_BLOB_LEN,
                "bytes": len(sidecar),
                "sha256": sha256_bytes(sidecar),
                "role": "sidecar",
            },
        ),
        source_layout_note=(
            "PR101 source layout is seven concatenated Brotli q11 streams over "
            "PR101 storage-order mapped uint8 codes plus fp16 scales."
        ),
    )


def _parse_pr106_decoder(payload: bytes) -> ParsedDecoder:
    packed = parse_ff_packed_brotli_hnerv(payload)
    raw = brotli.decompress(packed.decoder_packed_brotli)
    cursor = 0
    q_parts: list[tuple[int, str, tuple[int, ...], bytes]] = []
    for schema_index, name, shape in _packed_state_schema():
        count = math.prod(shape)
        end = cursor + count
        if end > len(raw):
            raise SystemExit("PR106 packed decoder q stream is truncated")
        q_parts.append((schema_index, name, shape, raw[cursor:end]))
        cursor = end
    scales_start = cursor
    expected = scales_start + 4 * len(q_parts)
    if expected != len(raw):
        raise SystemExit(f"bad PR106 packed decoder raw bytes: expected {expected}, got {len(raw)}")

    records_by_index: dict[int, TensorStream] = {}
    for order_index, (schema_index, name, shape, q_zz_bytes) in enumerate(q_parts):
        scale_start = scales_start + 4 * order_index
        scale_bytes = raw[scale_start : scale_start + 4]
        q_zz = np.frombuffer(q_zz_bytes, dtype=np.uint8)
        q_i8 = _zigzag_decode_u8(q_zz).reshape(shape).copy()
        records_by_index[schema_index] = TensorStream(
            schema_index=schema_index,
            name=name,
            shape=shape,
            q_i8=q_i8.reshape(-1).astype(np.int8, copy=True),
            scale_bytes=scale_bytes,
            source_symbol_bytes=q_zz_bytes,
        )

    recompressed = brotli.compress(raw, quality=11)
    return ParsedDecoder(
        source_format="pr106_ff_packed_hnerv",
        decoder_section=packed.decoder_packed_brotli,
        source_layout_raw=raw,
        source_layout_brotli_q11=recompressed,
        canonical_records=tuple(records_by_index[idx] for idx in range(len(FIXED_STATE_SCHEMA))),
        logical_sections=(
            {
                "name": "ff_header",
                "offset": 0,
                "bytes": len(packed.header),
                "sha256": sha256_bytes(packed.header),
                "role": "internal_length_header",
            },
            {
                "name": "decoder_packed_brotli",
                "offset": len(packed.header),
                "bytes": len(packed.decoder_packed_brotli),
                "sha256": sha256_bytes(packed.decoder_packed_brotli),
                "role": "renderer_decoder_weights",
            },
            {
                "name": "latents_and_sidecar_brotli",
                "offset": len(packed.header) + len(packed.decoder_packed_brotli),
                "bytes": len(packed.latents_and_sidecar_brotli),
                "sha256": sha256_bytes(packed.latents_and_sidecar_brotli),
                "role": "latents_and_sidecar",
            },
        ),
        source_layout_note=(
            "PR106 source layout is one Brotli q11 stream over packed fixed-schema "
            "zigzag uint8 codes followed by fp32 scales."
        ),
    )


def coder_packet_overhead_bytes(
    *,
    n_tensors: int,
    scale_bytes: int,
    n_categories: int = N_CATEGORIES,
    freq_entry_bytes: int = FREQ_ENTRY_BYTES,
    stream_length_bytes: int = STREAM_LENGTH_BYTES,
    fixed_packet_header_bytes: int = FIXED_PACKET_HEADER_BYTES,
) -> dict[str, int]:
    """Return conservative static-model packet overhead for separate tensor streams."""

    model_header_bytes = n_tensors * n_categories * freq_entry_bytes
    stream_length_table_bytes = n_tensors * stream_length_bytes
    total = fixed_packet_header_bytes + model_header_bytes + stream_length_table_bytes + scale_bytes
    return {
        "fixed_packet_header_bytes": fixed_packet_header_bytes,
        "model_header_bytes": model_header_bytes,
        "stream_length_table_bytes": stream_length_table_bytes,
        "scale_bytes": scale_bytes,
        "total_packet_overhead_bytes": total,
    }


def _entropy_bits_for_smoothed_counts(symbols: np.ndarray) -> tuple[float, np.ndarray]:
    counts = np.bincount(symbols.astype(np.int32), minlength=N_CATEGORIES).astype(np.int64)
    smoothed = counts + STATIC_MODEL_SMOOTHING
    denom = float(smoothed.sum())
    bits = 0.0
    for count, model_count in zip(counts, smoothed, strict=False):
        if count:
            bits += float(count) * -math.log2(float(model_count) / denom)
    return bits, smoothed


def build_static_model_proxy(records: tuple[TensorStream, ...]) -> dict[str, Any]:
    """Build a fully-headered static AC/rANS/FSE byte proxy for fixed schema streams."""

    payload_floor_bits = 0.0
    payload_floor_bytes = 0
    max_header_value = 0
    per_tensor: list[dict[str, Any]] = []
    for record in records:
        symbols = _symbols_0_254(record.q_i8)
        bits, smoothed = _entropy_bits_for_smoothed_counts(symbols)
        max_header_value = max(max_header_value, int(smoothed.max(initial=0)))
        per_tensor.append(
            {
                "schema_index": record.schema_index,
                "name": record.name,
                "symbols": record.symbol_count,
                "unique_symbols": int(np.count_nonzero(np.bincount(symbols, minlength=N_CATEGORIES))),
                "smoothed_entropy_bits": round(bits, 6),
                "smoothed_entropy_payload_bytes": math.ceil(bits / 8),
            }
        )
        payload_floor_bits += bits
        payload_floor_bytes += math.ceil(bits / 8)

    overhead = coder_packet_overhead_bytes(
        n_tensors=len(records),
        scale_bytes=sum(record.scale_bytes_len for record in records),
        freq_entry_bytes=2 if max_header_value <= 65535 else 4,
    )
    total = payload_floor_bytes + overhead["total_packet_overhead_bytes"]
    return {
        "variant": "static_ac_rans_fse_proxy_per_tensor_smoothed_counts",
        "codec_family": "AC/rANS/FSE proxy",
        "evidence_label": "CPU/proxy_no_bitstream",
        "score_claim": False,
        "payload_floor_bits": round(payload_floor_bits, 6),
        "payload_floor_bytes": payload_floor_bytes,
        "payload_floor_byte_alignment": "sum_per_tensor_ceil_bits_div_8",
        **overhead,
        "total_estimated_bytes": total,
        "n_categories": N_CATEGORIES,
        "smoothing": STATIC_MODEL_SMOOTHING,
        "frequency_table_storage": (
            "uint16 counts_plus_one"
            if overhead["model_header_bytes"] == len(records) * N_CATEGORIES * 2
            else "uint32 counts_plus_one"
        ),
        "per_tensor": per_tensor,
        "limitations": [
            "proxy_floor_not_an_actual_bitstream",
            "assumes fixed runtime schema so tensor names and shapes are not charged",
            "does_not_include_submission_decoder_code_bytes",
        ],
    }


def build_constriction_range_packet(records: tuple[TensorStream, ...]) -> dict[str, Any]:
    """Encode tensors with constriction range coding plus conservative headers."""

    try:
        import constriction
    except ImportError as exc:
        return {
            "variant": "constriction_range_per_tensor_smoothed_counts",
            "codec_family": "constriction range coder",
            "evidence_label": "CPU/proxy_missing_dependency",
            "available": False,
            "score_claim": False,
            "error": str(exc),
        }

    model_mod = constriction.stream.model
    encoder_cls = constriction.stream.queue.RangeEncoder
    decoder_cls = constriction.stream.queue.RangeDecoder
    version = _package_version("constriction")
    total_encoded_bytes = 0
    total_num_bits = 0
    max_header_value = 0
    per_tensor: list[dict[str, Any]] = []
    t0 = time.time()

    for record in records:
        symbols_u8 = _symbols_0_254(record.q_i8)
        symbols = symbols_u8.astype(np.int32)
        counts = np.bincount(symbols, minlength=N_CATEGORIES).astype(np.int64)
        smoothed = counts + STATIC_MODEL_SMOOTHING
        max_header_value = max(max_header_value, int(smoothed.max(initial=0)))
        probabilities = smoothed.astype(np.float64) / float(smoothed.sum())
        model = model_mod.Categorical(probabilities=probabilities, perfect=False)
        encoder = encoder_cls()
        encoder.encode(symbols, model)
        words = encoder.get_compressed()
        encoded = words.tobytes()
        decoder = decoder_cls(words)
        restored = decoder.decode(model, len(symbols)).astype(np.int32)
        roundtrip_equal = bool(np.array_equal(restored, symbols))
        if not roundtrip_equal:
            raise SystemExit(f"constriction roundtrip failed for {record.name}")
        total_encoded_bytes += len(encoded)
        total_num_bits += int(encoder.num_bits())
        per_tensor.append(
            {
                "schema_index": record.schema_index,
                "name": record.name,
                "symbols": record.symbol_count,
                "unique_symbols": int(np.count_nonzero(counts)),
                "encoded_bytes": len(encoded),
                "encoder_num_bits": int(encoder.num_bits()),
                "encoded_sha256": sha256_bytes(encoded),
                "roundtrip_equal": roundtrip_equal,
            }
        )

    overhead = coder_packet_overhead_bytes(
        n_tensors=len(records),
        scale_bytes=sum(record.scale_bytes_len for record in records),
        freq_entry_bytes=2 if max_header_value <= 65535 else 4,
    )
    total = total_encoded_bytes + overhead["total_packet_overhead_bytes"]
    return {
        "variant": "constriction_range_per_tensor_smoothed_counts",
        "codec_family": "constriction range coder",
        "evidence_label": "CPU/practical_actual_bitstream_no_runtime_adapter",
        "available": True,
        "score_claim": False,
        "constriction_version": version,
        "encoded_payload_bytes": total_encoded_bytes,
        "encoder_num_bits_total": total_num_bits,
        **overhead,
        "total_estimated_bytes": total,
        "n_categories": N_CATEGORIES,
        "smoothing": STATIC_MODEL_SMOOTHING,
        "frequency_table_storage": (
            "uint16 counts_plus_one"
            if overhead["model_header_bytes"] == len(records) * N_CATEGORIES * 2
            else "uint32 counts_plus_one"
        ),
        "elapsed_seconds": round(time.time() - t0, 6),
        "per_tensor": per_tensor,
        "limitations": [
            "actual_constriction_payload_but_no_submission_runtime_decoder",
            "separate_tensor_streams_require_charged_length_table",
            "assumes fixed runtime schema so tensor names and shapes are not charged",
            "does_not_include_submission_decoder_code_bytes",
        ],
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def reactivation_review(
    *,
    source_decoder_section_bytes: int,
    brotli_q11_source_layout_bytes: int,
    brotli_q11_canonical_bytes: int,
    proxy_total_bytes: int,
    constriction_total_bytes: int | None,
) -> dict[str, Any]:
    reference = min(source_decoder_section_bytes, brotli_q11_source_layout_bytes)
    practical_delta = (
        None if constriction_total_bytes is None else constriction_total_bytes - reference
    )
    proxy_delta = proxy_total_bytes - reference
    canonical_delta = brotli_q11_canonical_bytes - reference
    if practical_delta is not None and practical_delta < 0:
        verdict = "reactivate_runtime_prototype_only_no_score_claim"
        rationale = "actual constriction packet beats current Brotli reference after model headers"
    elif proxy_delta < 0:
        verdict = "reactivate_model_design_only_proxy_not_enough_for_dispatch"
        rationale = "proxy floor beats current Brotli reference but no practical packet cleared"
    else:
        verdict = "measured_precoarsening_static_config_retired"
        rationale = (
            "this zero-order pre-coarsening static configuration does not beat "
            "the current Brotli reference after charged model headers"
        )
    return {
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "reference_decoder_bytes": reference,
        "source_decoder_section_bytes": source_decoder_section_bytes,
        "brotli_q11_source_layout_bytes": brotli_q11_source_layout_bytes,
        "brotli_q11_canonical_i8_bytes": brotli_q11_canonical_bytes,
        "canonical_i8_delta_vs_reference_bytes": canonical_delta,
        "static_proxy_total_delta_vs_reference_bytes": proxy_delta,
        "constriction_total_delta_vs_reference_bytes": practical_delta,
        "verdict": verdict,
        "rationale": rationale,
        "reactivation_criteria": [
            "reactivate only for runtime-prototype work if actual practical coder total, including model headers, scale bytes, and stream lengths, beats current Brotli reference",
            "proxy-only wins may justify model design but not archive building or exact-eval dispatch",
            "rerun after coarsening, context mixing, rANS/FSE table sharing, or another distribution-changing transform before drawing broader conclusions",
            "any score-affecting path must later pay decoder code bytes, emit a candidate archive, pass strict preflight, and land exact CUDA auth eval",
        ],
    }


def _histogram_summary(records: tuple[TensorStream, ...]) -> dict[str, Any]:
    all_symbols = b"".join(_symbols_0_254(record.q_i8).tobytes() for record in records)
    counts = Counter(all_symbols)
    total = len(all_symbols)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return {
        "symbols": total,
        "unique_symbols": len(counts),
        "entropy_bits_per_symbol": round(entropy, 12),
        "entropy_floor_bytes_without_headers": math.ceil(entropy * total / 8),
        "top_symbols": [
            {"symbol_0_254": int(symbol), "signed_i8": int(symbol) - 127, "count": int(count)}
            for symbol, count in counts.most_common(16)
        ],
    }


def _build_target_record(label: str, archive_path: Path, parser: str) -> dict[str, Any]:
    archive = _read_single_member_archive(archive_path)
    payload = archive.pop("payload")
    if parser == "pr101":
        parsed = _parse_pr101_decoder(payload)
    elif parser == "pr106":
        parsed = _parse_pr106_decoder(payload)
    else:
        raise SystemExit(f"unknown parser: {parser}")

    canonical = _canonical_precoarsening_bytes(parsed.canonical_records)
    brotli_canonical = brotli.compress(canonical, quality=11)
    proxy = build_static_model_proxy(parsed.canonical_records)
    constriction = build_constriction_range_packet(parsed.canonical_records)
    constriction_total = (
        int(constriction["total_estimated_bytes"]) if constriction.get("available") else None
    )
    review = reactivation_review(
        source_decoder_section_bytes=len(parsed.decoder_section),
        brotli_q11_source_layout_bytes=len(parsed.source_layout_brotli_q11),
        brotli_q11_canonical_bytes=len(brotli_canonical),
        proxy_total_bytes=int(proxy["total_estimated_bytes"]),
        constriction_total_bytes=constriction_total,
    )
    return {
        "label": label,
        "parser": parser,
        "score_claim": False,
        "evidence_label": "CPU/proxy_byte_accounting_no_scorer_no_cuda",
        **archive,
        "source_format": parsed.source_format,
        "logical_sections": list(parsed.logical_sections),
        "decoder_section_bytes": len(parsed.decoder_section),
        "decoder_section_sha256": sha256_bytes(parsed.decoder_section),
        "source_layout_raw_bytes": len(parsed.source_layout_raw),
        "source_layout_raw_sha256": sha256_bytes(parsed.source_layout_raw),
        "source_layout_note": parsed.source_layout_note,
        "q_symbol_count": parsed.q_symbol_count,
        "scale_bytes": parsed.scale_bytes,
        "canonical_precoarsening_i8_stream": {
            "description": (
                "fixed-schema canonical stream: signed int8 weight codes shifted to "
                "0..254 per tensor in schema order, followed by source scale bytes"
            ),
            "bytes": len(canonical),
            "sha256": sha256_bytes(canonical),
            "brotli_q11_bytes": len(brotli_canonical),
            "brotli_q11_sha256": sha256_bytes(brotli_canonical),
        },
        "source_layout_brotli_q11": {
            "bytes": len(parsed.source_layout_brotli_q11),
            "sha256": sha256_bytes(parsed.source_layout_brotli_q11),
            "matches_source_decoder_section": parsed.source_layout_brotli_q11 == parsed.decoder_section,
            "delta_vs_source_decoder_section_bytes": len(parsed.source_layout_brotli_q11)
            - len(parsed.decoder_section),
        },
        "global_symbol_histogram": _histogram_summary(parsed.canonical_records),
        "static_model_proxy": proxy,
        "constriction_range_packet": constriction,
        "reactivation_review": review,
        "dispatch_blockers": [
            "probe_only_no_candidate_archive",
            "no_runtime_decoder_adapter",
            "decoder_code_bytes_not_charged",
            "strict_pre_submission_compliance_not_run",
            "exact_cuda_auth_eval_missing",
        ],
    }


def build_probe_manifest(pr101_archive: Path, pr106_archive: Path) -> dict[str, Any]:
    targets = [
        _build_target_record("PR101 hnerv_ft_microcodec", pr101_archive, "pr101"),
        _build_target_record("PR106 belt_and_suspenders", pr106_archive, "pr106"),
    ]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "brotli_version": _package_version("brotli"),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_label": "CPU/proxy_precoarsening_entropy_accounting",
        "inputs": {
            "pr101_archive": _repo_rel(pr101_archive),
            "pr106_archive": _repo_rel(pr106_archive),
        },
        "target_count": len(targets),
        "targets": targets,
        "hstack_vstack_review": hstack_vstack_review(targets),
        "global_dispatch_blockers": [
            "no_score_claim",
            "no_candidate_archive_emitted",
            "no_runtime_decoder_adapter",
            "decoder_code_bytes_not_charged",
            "exact_cuda_auth_eval_missing",
        ],
    }


def hstack_vstack_review(targets: list[dict[str, Any]]) -> dict[str, Any]:
    any_practical_win = any(
        target["reactivation_review"]["verdict"]
        == "reactivate_runtime_prototype_only_no_score_claim"
        for target in targets
    )
    return {
        "score_claim": False,
        "HStack": {
            "status": "decoder_stream_only_horizontal_candidate",
            "synergy": [
                "Touches parser-proven renderer decoder bytes only, so it can be evaluated independently from latent/sidecar HStack lanes once a runtime adapter exists.",
                "Could HStack with low-level latent/sidecar repacks because those operate on disjoint logical sections.",
            ],
            "antagonism": [
                "Competes with PR101 split-Brotli, HDM-style decoder recodes, and any decoder replacement that owns the same renderer weight stream.",
                "Any custom entropy runtime adds code bytes and dependency risk that are not charged in this probe.",
            ],
        },
        "VStack": {
            "status": "terminal_entropy_stage_after_representation_quantization",
            "synergy": [
                "Best positioned after byte-map/order derivation and after any lossy coarsening, because those transforms change the symbol distribution being coded.",
                "A post-coarsening rerun could shrink frequency tables and improve practical static-model economics.",
            ],
            "antagonism": [
                "Pre-coarsening static coding is intentionally a stricter screen; it may understate a later coarsened-stack coder but cannot justify dispatch by itself.",
                "Zero-order per-tensor models ignore sequential context that Brotli already exploits, so source-layout Brotli remains the reference to beat.",
            ],
        },
        "synergy_antagonism_verdict": (
            "practical_runtime_prototype_allowed_but_still_no_score_claim"
            if any_practical_win
            else "do_not_build_archive_from_precoarsening_static_entropy_probe"
        ),
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Pre-Coarsening Entropy-Coder Probe: PR101/PR106",
        "",
        f"- schema: `{manifest['schema']}`",
        f"- evidence: `{manifest['evidence_label']}`",
        f"- score_claim: `{str(manifest['score_claim']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(manifest['ready_for_exact_eval_dispatch']).lower()}`",
        "",
        "## Target Summary",
        "",
        "| target | archive bytes | decoder bytes | source q11 | canonical q11 | proxy total | constriction total | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for target in manifest["targets"]:
        constriction = target["constriction_range_packet"]
        constriction_total = (
            str(constriction["total_estimated_bytes"]) if constriction.get("available") else "n/a"
        )
        lines.append(
            "| {label} | {archive_bytes} | {decoder} | {source_q11} | {canon_q11} | {proxy} | {constriction} | `{verdict}` |".format(
                label=target["label"],
                archive_bytes=target["archive_bytes"],
                decoder=target["decoder_section_bytes"],
                source_q11=target["source_layout_brotli_q11"]["bytes"],
                canon_q11=target["canonical_precoarsening_i8_stream"]["brotli_q11_bytes"],
                proxy=target["static_model_proxy"]["total_estimated_bytes"],
                constriction=constriction_total,
                verdict=target["reactivation_review"]["verdict"],
            )
        )
    lines.extend(["", "## HStack/VStack Review", ""])
    review = manifest["hstack_vstack_review"]
    lines.append(f"- HStack status: `{review['HStack']['status']}`")
    lines.append(f"- VStack status: `{review['VStack']['status']}`")
    lines.append(f"- verdict: `{review['synergy_antagonism_verdict']}`")
    lines.extend(["", "### HStack Synergy", ""])
    lines.extend(f"- {item}" for item in review["HStack"]["synergy"])
    lines.extend(["", "### HStack Antagonism", ""])
    lines.extend(f"- {item}" for item in review["HStack"]["antagonism"])
    lines.extend(["", "### VStack Synergy", ""])
    lines.extend(f"- {item}" for item in review["VStack"]["synergy"])
    lines.extend(["", "### VStack Antagonism", ""])
    lines.extend(f"- {item}" for item in review["VStack"]["antagonism"])
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{item}`" for item in manifest["global_dispatch_blockers"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr101-archive", type=Path, default=DEFAULT_PR101)
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args(argv)

    manifest = build_probe_manifest(args.pr101_archive, args.pr106_archive)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(_json_text(manifest), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(manifest), encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(_json_text(manifest), end="")
    else:
        for target in manifest["targets"]:
            constriction = target["constriction_range_packet"]
            constriction_total = (
                constriction["total_estimated_bytes"] if constriction.get("available") else "n/a"
            )
            print(
                f"{target['label']}: decoder={target['decoder_section_bytes']} "
                f"source_q11={target['source_layout_brotli_q11']['bytes']} "
                f"canonical_q11={target['canonical_precoarsening_i8_stream']['brotli_q11_bytes']} "
                f"proxy={target['static_model_proxy']['total_estimated_bytes']} "
                f"constriction={constriction_total} "
                f"verdict={target['reactivation_review']['verdict']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
