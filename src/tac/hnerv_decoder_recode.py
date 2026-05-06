"""Planning-only structural recode probes for PR106-style HNeRV decoders."""

from __future__ import annotations

import dataclasses
import io
import math
import struct
from collections import Counter
from typing import Any

import brotli
import numpy as np

from tac.arithmetic_qint_codec import decode_qints_arithmetic, encode_qints_arithmetic
from tac.hnerv_lowlevel_packer import PackedHnervPayload, sha256_bytes
from tac.lossless.frequency_coder import (
    decode_uint16_frequency_stream,
    encode_uint16_frequency_stream,
)

FIXED_STATE_SCHEMA: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("stem.weight", (1728, 28)),
    ("stem.bias", (1728,)),
    ("blocks.0.weight", (144, 36, 3, 3)),
    ("blocks.0.bias", (144,)),
    ("blocks.1.weight", (144, 36, 3, 3)),
    ("blocks.1.bias", (144,)),
    ("blocks.2.weight", (108, 36, 3, 3)),
    ("blocks.2.bias", (108,)),
    ("blocks.3.weight", (80, 27, 3, 3)),
    ("blocks.3.bias", (80,)),
    ("blocks.4.weight", (72, 20, 3, 3)),
    ("blocks.4.bias", (72,)),
    ("blocks.5.weight", (72, 18, 3, 3)),
    ("blocks.5.bias", (72,)),
    ("skips.2.weight", (27, 36, 1, 1)),
    ("skips.2.bias", (27,)),
    ("skips.3.weight", (20, 27, 1, 1)),
    ("skips.3.bias", (20,)),
    ("skips.4.weight", (18, 20, 1, 1)),
    ("skips.4.bias", (18,)),
    ("refine.0.weight", (9, 18, 3, 3)),
    ("refine.0.bias", (9,)),
    ("refine.1.weight", (18, 9, 3, 3)),
    ("refine.1.bias", (18,)),
    ("rgb_0.weight", (3, 18, 3, 3)),
    ("rgb_0.bias", (3,)),
    ("rgb_1.weight", (3, 18, 3, 3)),
    ("rgb_1.bias", (3,)),
)

PACKED_STATE_SCHEMA: tuple[tuple[str, tuple[int, ...]], ...] = tuple(
    sorted(FIXED_STATE_SCHEMA, key=lambda item: -math.prod(item[1]))
)
SCHEMA_VERSION = 1


class HnervDecoderRecodeError(ValueError):
    """Raised when an HNeRV decoder structural recode probe is invalid."""


@dataclasses.dataclass(frozen=True)
class PackedDecoderRecord:
    name: str
    shape: tuple[int, ...]
    q_zz_u8: bytes
    scale_f32: bytes

    @property
    def value_count(self) -> int:
        return math.prod(self.shape)


@dataclasses.dataclass(frozen=True)
class PackedDecoderRaw:
    records: tuple[PackedDecoderRecord, ...]

    def to_raw(self) -> bytes:
        q = b"".join(record.q_zz_u8 for record in self.records)
        scales = b"".join(record.scale_f32 for record in self.records)
        return q + scales

    @property
    def q_stream(self) -> bytes:
        return b"".join(record.q_zz_u8 for record in self.records)

    @property
    def scale_stream(self) -> bytes:
        return b"".join(record.scale_f32 for record in self.records)


def parse_packed_decoder_brotli(decoder_brotli: bytes) -> PackedDecoderRaw:
    """Parse PR106 packed decoder brotli bytes into schema records."""

    raw = brotli.decompress(decoder_brotli)
    cursor = 0
    q_parts: list[bytes] = []
    for _name, shape in PACKED_STATE_SCHEMA:
        count = math.prod(shape)
        end = cursor + count
        if end > len(raw):
            raise HnervDecoderRecodeError("packed decoder q stream is truncated")
        q_parts.append(raw[cursor:end])
        cursor = end
    scales_start = cursor
    expected = scales_start + 4 * len(PACKED_STATE_SCHEMA)
    if expected != len(raw):
        raise HnervDecoderRecodeError(
            f"bad packed decoder raw length: expected {expected}, got {len(raw)}"
        )
    records = []
    for index, (name, shape) in enumerate(PACKED_STATE_SCHEMA):
        scale_start = scales_start + 4 * index
        records.append(
            PackedDecoderRecord(
                name=name,
                shape=shape,
                q_zz_u8=q_parts[index],
                scale_f32=raw[scale_start : scale_start + 4],
            )
        )
    parsed = PackedDecoderRaw(records=tuple(records))
    if parsed.to_raw() != raw:
        raise HnervDecoderRecodeError("packed decoder did not raw-roundtrip")
    return parsed


def build_structural_recode_profile(
    packed: PackedHnervPayload,
    *,
    source_label: str,
    source_archive_sha256: str,
) -> dict[str, Any]:
    """Profile lossless structural recodes for a PR106-style decoder section."""

    parsed = parse_packed_decoder_brotli(packed.decoder_packed_brotli)
    source_raw = parsed.to_raw()
    source_brotli = packed.decoder_packed_brotli
    entropy_summary = _entropy_summary(parsed, source_section_bytes=len(source_brotli))
    variants = [
        _variant_brotli("brotli_q11_current_raw", source_raw, quality=11),
        _variant_brotli("brotli_q10_current_raw", source_raw, quality=10),
        _variant_brotli("brotli_q9_current_raw", source_raw, quality=9),
        _variant_aq_global(parsed),
        _variant_aq_per_tensor(parsed),
        _variant_huffman_global(parsed),
    ]
    for row in variants:
        row["byte_delta_vs_source_section"] = int(row["bytes"]) - len(source_brotli)
        row["rate_score_delta_if_runtime_supported_and_components_equal"] = round(
            row["byte_delta_vs_source_section"] * (25 / 37_545_489), 12
        )
        if str(row["variant"]).endswith("_q_stream_plus_raw_scales"):
            row["byte_gap_vs_global_q_entropy_floor_plus_raw_scales"] = int(row["bytes"]) - int(
                entropy_summary["global_q_entropy_floor_plus_raw_scales_bytes"]
            )
    best = min(variants, key=lambda row: (int(row["bytes"]), str(row["variant"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_decoder_recode.build_structural_recode_profile",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_label": source_label,
        "source_archive_sha256": source_archive_sha256,
        "source_decoder_section_sha256": sha256_bytes(source_brotli),
        "source_decoder_section_bytes": len(source_brotli),
        "source_decoder_raw_sha256": sha256_bytes(source_raw),
        "source_decoder_raw_bytes": len(source_raw),
        "record_count": len(parsed.records),
        "q_stream_bytes": len(parsed.q_stream),
        "scale_stream_bytes": len(parsed.scale_stream),
        "entropy_summary": entropy_summary,
        "variants": variants,
        "best_variant": best,
        "dispatch_blockers": [
            "planning_only_structural_recode_profile",
            "requires_runtime_decoder_implementation",
            "requires_archive_manifest_preflight",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _entropy_summary(parsed: PackedDecoderRaw, *, source_section_bytes: int) -> dict[str, Any]:
    global_q = _symbol_entropy_summary(parsed.q_stream)
    per_tensor_floor = sum(
        _symbol_entropy_summary(record.q_zz_u8)["entropy_floor_bytes"]
        for record in parsed.records
    )
    global_floor_plus_scales = int(global_q["entropy_floor_bytes"]) + len(parsed.scale_stream)
    per_tensor_floor_plus_scales = per_tensor_floor + len(parsed.scale_stream)
    return {
        "scope": "zero_order_symbol_entropy_floor_for_current_q_streams",
        "score_claim": False,
        "q_stream_symbols": len(parsed.q_stream),
        "q_stream_unique_symbols": global_q["unique_symbols"],
        "q_stream_entropy_bits_per_symbol": global_q["entropy_bits_per_symbol"],
        "global_q_entropy_floor_bytes": global_q["entropy_floor_bytes"],
        "global_q_entropy_floor_plus_raw_scales_bytes": global_floor_plus_scales,
        "global_q_entropy_floor_delta_vs_source_section_bytes": global_floor_plus_scales
        - source_section_bytes,
        "per_tensor_q_entropy_floor_bytes": per_tensor_floor,
        "per_tensor_q_entropy_floor_plus_raw_scales_bytes": per_tensor_floor_plus_scales,
        "per_tensor_q_entropy_floor_delta_vs_source_section_bytes": per_tensor_floor_plus_scales
        - source_section_bytes,
        "top_global_q_symbols": global_q["top_symbols"],
        "current_static_model_interpretation": (
            "zero_order_q_symbol_floor_loses_to_current_brotli"
            if global_floor_plus_scales >= source_section_bytes
            else "zero_order_q_symbol_floor_has_byte_headroom"
        ),
        "model_limitations": [
            "zero_order_symbol_floor_only",
            "does_not_model_contexts_run_lengths_deltas_tensor_shapes_or_brotli_transforms",
            "does_not_include_decoder_runtime_cost",
            "does_not_rank_or_kill_aq_huffman_family_without_exact_archive_eval",
        ],
    }


def _symbol_entropy_summary(payload: bytes) -> dict[str, Any]:
    if not payload:
        return {
            "symbols": 0,
            "unique_symbols": 0,
            "entropy_bits_per_symbol": 0.0,
            "entropy_floor_bytes": 0,
            "top_symbols": [],
        }
    counts = Counter(payload)
    total = len(payload)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return {
        "symbols": total,
        "unique_symbols": len(counts),
        "entropy_bits_per_symbol": round(float(entropy), 12),
        "entropy_floor_bytes": int(math.ceil(entropy * total / 8)),
        "top_symbols": [
            {
                "symbol": int(symbol),
                "count": int(count),
                "frequency": round(float(count / total), 12),
            }
            for symbol, count in counts.most_common(16)
        ],
    }


def _variant_brotli(label: str, raw: bytes, *, quality: int) -> dict[str, Any]:
    payload = brotli.compress(raw, quality=quality)
    return {
        "variant": label,
        "codec": "brotli",
        "quality": quality,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "raw_equal": brotli.decompress(payload) == raw,
    }


def _variant_aq_global(parsed: PackedDecoderRaw) -> dict[str, Any]:
    q = np.frombuffer(parsed.q_stream, dtype=np.uint8)
    q_blob = encode_qints_arithmetic(q, num_symbols=256, offset=0)
    payload = _container(
        b"HDA1",
        [
            (b"q", q_blob),
            (b"s", parsed.scale_stream),
        ],
    )
    decoded_q = decode_qints_arithmetic(q_blob, expected_dtype=np.uint8).tobytes()
    return {
        "variant": "aq_global_q_stream_plus_raw_scales",
        "codec": "AQv1_static_global_uint8",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": decoded_q == parsed.q_stream,
        "scale_roundtrip_equal": True,
        "raw_equal": decoded_q + parsed.scale_stream == parsed.to_raw(),
    }


def _variant_aq_per_tensor(parsed: PackedDecoderRaw) -> dict[str, Any]:
    records = []
    decoded_q_parts = []
    for record in parsed.records:
        q = np.frombuffer(record.q_zz_u8, dtype=np.uint8)
        q_blob = encode_qints_arithmetic(q, num_symbols=256, offset=0)
        records.append((record.name.encode("utf-8"), q_blob))
        decoded_q_parts.append(decode_qints_arithmetic(q_blob, expected_dtype=np.uint8).tobytes())
    records.append((b"scales", parsed.scale_stream))
    payload = _container(b"HDA2", records)
    decoded_q = b"".join(decoded_q_parts)
    return {
        "variant": "aq_per_tensor_q_streams_plus_raw_scales",
        "codec": "AQv1_static_per_tensor_uint8",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": decoded_q == parsed.q_stream,
        "scale_roundtrip_equal": True,
        "raw_equal": decoded_q + parsed.scale_stream == parsed.to_raw(),
    }


def _variant_huffman_global(parsed: PackedDecoderRaw) -> dict[str, Any]:
    q = np.frombuffer(parsed.q_stream, dtype=np.uint8).astype(np.uint16)
    encoded = encode_uint16_frequency_stream(q)
    decoded = decode_uint16_frequency_stream(encoded.encoded_bytes).astype(np.uint8).tobytes()
    payload = _container(
        b"HDH1",
        [
            (b"q", encoded.encoded_bytes),
            (b"s", parsed.scale_stream),
        ],
    )
    return {
        "variant": "canonical_huffman_global_q_stream_plus_raw_scales",
        "codec": "TFC1_static_global_uint8",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": decoded == parsed.q_stream,
        "scale_roundtrip_equal": True,
        "raw_equal": decoded + parsed.scale_stream == parsed.to_raw(),
        "max_code_bits": encoded.max_code_bits,
        "unique_symbols": encoded.unique_symbols,
    }


def _container(magic: bytes, records: list[tuple[bytes, bytes]]) -> bytes:
    if len(magic) != 4:
        raise HnervDecoderRecodeError("container magic must be 4 bytes")
    out = io.BytesIO()
    out.write(magic)
    out.write(struct.pack("<H", len(records)))
    for name, payload in records:
        if len(name) > 0xFFFF:
            raise HnervDecoderRecodeError("record name too long")
        out.write(struct.pack("<H", len(name)))
        out.write(name)
        out.write(struct.pack("<I", len(payload)))
        out.write(payload)
    return out.getvalue()


__all__ = [
    "FIXED_STATE_SCHEMA",
    "PACKED_STATE_SCHEMA",
    "SCHEMA_VERSION",
    "HnervDecoderRecodeError",
    "PackedDecoderRaw",
    "PackedDecoderRecord",
    "build_structural_recode_profile",
    "parse_packed_decoder_brotli",
]
