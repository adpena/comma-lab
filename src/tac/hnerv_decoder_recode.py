"""Planning-only structural recode probes for PR106-style HNeRV decoders."""

from __future__ import annotations

import dataclasses
import io
import math
import struct
from collections import Counter
from itertools import pairwise
from typing import Any

import brotli
import numpy as np

from tac.arithmetic_qint_codec import decode_qints_arithmetic, encode_qints_arithmetic
from tac.hnerv_lowlevel_packer import PackedHnervPayload, sha256_bytes
from tac.lossless.frequency_coder import (
    decode_uint16_frequency_stream,
    encode_uint16_frequency_stream,
)
from tac.lossless.range_coder import decode_static_symbols, encode_static_symbols

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
        _variant_context_range_per_tensor(parsed),
        _variant_context_range_global(parsed),
        _variant_mixed_context_global(parsed),
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
        if row["variant"] == "range_prev_symbol_per_tensor_q_streams_plus_raw_scales":
            row["byte_gap_vs_per_tensor_q_entropy_floor_plus_raw_scales"] = int(row["bytes"]) - int(
                entropy_summary["per_tensor_q_entropy_floor_plus_raw_scales_bytes"]
            )
            row["byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales"] = int(
                row["bytes"]
            ) - int(entropy_summary["per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes"])
        if row["variant"] == "range_prev_symbol_global_q_streams_plus_raw_scales":
            row["byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales"] = int(
                row["bytes"]
            ) - int(entropy_summary["per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes"])
        if (
            row["variant"]
            == "mixed_range_raw_global_prev_symbol_schema_indexed_q_streams_plus_raw_scales"
        ):
            row["byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales"] = int(
                row["bytes"]
            ) - int(entropy_summary["per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes"])
    best = min(variants, key=lambda row: (int(row["bytes"]), str(row["variant"])))
    context_overhead_plan = _context_overhead_plan(
        entropy_summary,
        variants,
        source_section_bytes=len(source_brotli),
    )
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
        "context_overhead_plan": context_overhead_plan,
        "variants": variants,
        "best_variant": best,
        "dispatch_blockers": [
            "planning_only_structural_recode_profile",
            "requires_runtime_decoder_implementation",
            "requires_archive_manifest_preflight",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _context_overhead_plan(
    entropy_summary: dict[str, Any],
    variants: list[dict[str, Any]],
    *,
    source_section_bytes: int,
) -> dict[str, Any]:
    hdc1 = _variant_by_name(variants, "range_prev_symbol_per_tensor_q_streams_plus_raw_scales")
    hdc2 = _variant_by_name(variants, "range_prev_symbol_global_q_streams_plus_raw_scales")
    hdc2_mixed = _variant_by_name(
        variants,
        "mixed_range_raw_global_prev_symbol_schema_indexed_q_streams_plus_raw_scales",
    )
    floor_bytes = int(entropy_summary["per_tensor_prev_symbol_entropy_floor_bytes"])
    floor_plus_scales = int(
        entropy_summary["per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes"]
    )
    source_floor_headroom = source_section_bytes - floor_plus_scales
    if hdc1 is None or hdc2 is None:
        return {
            "score_claim": False,
            "planning_only": True,
            "ready_for_exact_eval_dispatch": False,
            "largest_accounted_gap": None,
            "largest_remaining_safe_target": None,
            "dispatch_blockers": [
                "missing_hdc1_or_hdc2_context_fixture",
                "requires_runtime_decoder_implementation",
                "requires_exact_cuda_auth_eval",
            ],
        }

    hdc1_header = int(hdc1["header_bytes"])
    hdc2_header = int(hdc2["header_bytes"])
    hdc1_payload = int(hdc1["range_payload_bytes"])
    hdc2_payload = int(hdc2["range_payload_bytes"])
    hdc2_delta_vs_source = int(hdc2["byte_delta_vs_source_section"])
    break_even_reduction = max(0, hdc2_delta_vs_source + 1)
    realized_header_savings = hdc1_header - hdc2_header
    hdc2_payload_gap = hdc2_payload - floor_bytes
    hdc2_gap_to_floor = int(hdc2["bytes"]) - floor_plus_scales
    minimum_payload_reduction_after_zero_header = max(
        0,
        break_even_reduction - hdc2_header,
    )
    remaining_targets = [
        {
            "gap_id": "hdc2_self_describing_context_header",
            "bytes": hdc2_header,
            "status": "remaining_safe_accounting_target",
            "target_action": "replace_or_codebook_share_context_table_metadata",
        },
        {
            "gap_id": "hdc2_range_payload_above_prev_symbol_entropy_floor",
            "bytes": hdc2_payload_gap,
            "status": "remaining_entropy_gap_target",
            "target_action": "improve_context_partition_or_range_payload_coding",
        },
    ]
    remaining_targets.sort(key=lambda row: (-int(row["bytes"]), str(row["gap_id"])))
    mixed_candidate = None
    if hdc2_mixed is not None:
        mixed_bytes = int(hdc2_mixed["bytes"])
        mixed_header = int(hdc2_mixed["header_bytes"])
        mixed_payload = int(hdc2_mixed["mixed_payload_bytes"])
        mixed_delta_vs_source = mixed_bytes - source_section_bytes
        mixed_candidate = {
            "variant": hdc2_mixed["variant"],
            "codec": hdc2_mixed["codec"],
            "score_claim": False,
            "planning_only": True,
            "archive_ready": False,
            "ready_for_exact_eval_dispatch": False,
            "bytes": mixed_bytes,
            "byte_delta_vs_hdc2_bytes": mixed_bytes - int(hdc2["bytes"]),
            "byte_reduction_vs_hdc2_bytes": int(hdc2["bytes"]) - mixed_bytes,
            "byte_delta_vs_source_section": mixed_delta_vs_source,
            "remaining_reduction_to_beat_source_section_bytes": max(
                0,
                mixed_delta_vs_source + 1,
            ),
            "header_bytes": mixed_header,
            "static_context_header_reduction_vs_hdc2_bytes": hdc2_header - mixed_header,
            "mixed_payload_bytes": mixed_payload,
            "payload_delta_vs_hdc2_bytes": mixed_payload - hdc2_payload,
            "raw_context_count": int(hdc2_mixed["raw_context_count"]),
            "range_context_count": int(hdc2_mixed["range_context_count"]),
            "raw_context_payload_bytes": int(hdc2_mixed["raw_payload_bytes"]),
            "range_context_payload_bytes": int(hdc2_mixed["range_payload_bytes"]),
            "schema_metadata_elided_vs_hdc2_bytes": int(
                hdc2_mixed["schema_metadata_elided_vs_hdc2_bytes"]
            ),
        }
    return {
        "score_claim": False,
        "planning_only": True,
        "ready_for_exact_eval_dispatch": False,
        "source_section_bytes": source_section_bytes,
        "prev_symbol_entropy_floor_plus_raw_scales_bytes": floor_plus_scales,
        "perfect_prev_symbol_floor_delta_vs_source_bytes": floor_plus_scales
        - source_section_bytes,
        "floor_headroom_before_runtime_overhead_bytes": source_floor_headroom,
        "hdc1_per_tensor_context_count": int(hdc1["context_count"]),
        "hdc2_global_context_count": int(hdc2["context_count"]),
        "hdc1_header_bytes": hdc1_header,
        "hdc2_header_bytes": hdc2_header,
        "hdc1_range_payload_bytes": hdc1_payload,
        "hdc2_range_payload_bytes": hdc2_payload,
        "hdc1_to_hdc2_header_savings_bytes": realized_header_savings,
        "hdc2_range_payload_penalty_vs_hdc1_bytes": hdc2_payload - hdc1_payload,
        "hdc2_net_savings_vs_hdc1_bytes": int(hdc1["bytes"]) - int(hdc2["bytes"]),
        "hdc2_gap_to_prev_symbol_entropy_floor_plus_raw_scales_bytes": hdc2_gap_to_floor,
        "hdc2_payload_gap_to_prev_symbol_entropy_floor_bytes": hdc2_payload_gap,
        "break_even_reduction_vs_source_from_hdc2_bytes": break_even_reduction,
        "minimum_payload_gap_reduction_after_zero_header_bytes": (
            minimum_payload_reduction_after_zero_header
        ),
        "largest_accounted_gap": {
            "gap_id": "hdc1_to_hdc2_context_header_amortization",
            "bytes": realized_header_savings,
            "status": "already_realized_by_hdc2_fixture",
        },
        "largest_remaining_safe_target": remaining_targets[0],
        "remaining_target_ranking": remaining_targets,
        "bounded_hdc2_mixed_context_candidate": mixed_candidate,
        "planner_verdict": (
            "target_hdc2_context_header_first_but_payload_gap_must_also_shrink"
            if minimum_payload_reduction_after_zero_header
            else "target_hdc2_context_header_first"
        ),
        "dispatch_blockers": [
            "planning_only_context_overhead_accounting",
            "requires_submission_runtime_decoder",
            "requires_archive_builder_and_payload_diff",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _variant_by_name(
    variants: list[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    for row in variants:
        if row.get("variant") == name:
            return row
    return None


def _entropy_summary(parsed: PackedDecoderRaw, *, source_section_bytes: int) -> dict[str, Any]:
    global_q = _symbol_entropy_summary(parsed.q_stream)
    per_tensor_floor = sum(
        _symbol_entropy_summary(record.q_zz_u8)["entropy_floor_bytes"]
        for record in parsed.records
    )
    per_tensor_prev_symbol = _prev_symbol_entropy_summary(parsed.records)
    global_floor_plus_scales = int(global_q["entropy_floor_bytes"]) + len(parsed.scale_stream)
    per_tensor_floor_plus_scales = per_tensor_floor + len(parsed.scale_stream)
    per_tensor_prev_symbol_floor_plus_scales = (
        int(per_tensor_prev_symbol["entropy_floor_bytes"]) + len(parsed.scale_stream)
    )
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
        "per_tensor_prev_symbol_contexts": per_tensor_prev_symbol["context_count"],
        "per_tensor_prev_symbol_tokens": per_tensor_prev_symbol["tokens"],
        "per_tensor_prev_symbol_entropy_bits_per_token": per_tensor_prev_symbol[
            "entropy_bits_per_token"
        ],
        "per_tensor_prev_symbol_entropy_floor_bytes": per_tensor_prev_symbol["entropy_floor_bytes"],
        "per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes": (
            per_tensor_prev_symbol_floor_plus_scales
        ),
        "per_tensor_prev_symbol_entropy_floor_delta_vs_source_section_bytes": (
            per_tensor_prev_symbol_floor_plus_scales - source_section_bytes
        ),
        "top_global_q_symbols": global_q["top_symbols"],
        "current_static_model_interpretation": (
            "zero_order_q_symbol_floor_loses_to_current_brotli"
            if global_floor_plus_scales >= source_section_bytes
            else "zero_order_q_symbol_floor_has_byte_headroom"
        ),
        "model_limitations": [
            "zero_order_symbol_floor_only",
            "prev_symbol_context_floor_models_only_one_local_context_family",
            "does_not_model_run_lengths_deltas_tensor_shapes_or_brotli_transforms",
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
        "entropy_floor_bytes": math.ceil(entropy * total / 8),
        "top_symbols": [
            {
                "symbol": int(symbol),
                "count": int(count),
                "frequency": round(float(count / total), 12),
            }
            for symbol, count in counts.most_common(16)
        ],
    }


def _prev_symbol_entropy_summary(records: tuple[PackedDecoderRecord, ...]) -> dict[str, Any]:
    context_count = 0
    token_count = 0
    entropy_bits = 0.0
    for record in records:
        contexts = _prev_symbol_contexts(record.q_zz_u8)
        context_count += len(contexts)
        for stream in contexts.values():
            summary = _symbol_entropy_summary(bytes(stream))
            token_count += len(stream)
            entropy_bits += float(summary["entropy_bits_per_symbol"]) * len(stream)
    entropy_floor = math.ceil(entropy_bits / 8)
    return {
        "context_count": context_count,
        "tokens": token_count,
        "entropy_bits_per_token": round(float(entropy_bits / token_count), 12)
        if token_count
        else 0.0,
        "entropy_floor_bytes": entropy_floor,
    }


def _prev_symbol_contexts(payload: bytes) -> dict[int, list[int]]:
    contexts: dict[int, list[int]] = {}
    for previous, current in pairwise(payload):
        contexts.setdefault(int(previous), []).append(int(current))
    return contexts


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


def _variant_context_range_per_tensor(parsed: PackedDecoderRaw) -> dict[str, Any]:
    payload, stats = encode_prev_symbol_context_range_fixture(parsed)
    restored = decode_prev_symbol_context_range_fixture(payload)
    decoded_q = restored.q_stream
    return {
        "variant": "range_prev_symbol_per_tensor_q_streams_plus_raw_scales",
        "codec": "HDC1_prev_symbol_per_tensor_range_uint8",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": decoded_q == parsed.q_stream,
        "scale_roundtrip_equal": restored.scale_stream == parsed.scale_stream,
        "raw_equal": restored.to_raw() == parsed.to_raw(),
        "tensor_count": len(parsed.records),
        "context_count": stats["context_count"],
        "context_token_count": stats["context_token_count"],
        "header_bytes": stats["header_bytes"],
        "range_payload_bytes": stats["range_payload_bytes"],
        "raw_scale_bytes": len(parsed.scale_stream),
        "parity_fixture": True,
        "archive_ready": False,
        "dispatch_blockers": [
            "parity_fixture_only",
            "requires_submission_runtime_decoder",
            "requires_archive_builder_and_payload_diff",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _variant_context_range_global(parsed: PackedDecoderRaw) -> dict[str, Any]:
    payload, stats = encode_global_prev_symbol_context_range_fixture(parsed)
    restored = decode_global_prev_symbol_context_range_fixture(payload)
    return {
        "variant": "range_prev_symbol_global_q_streams_plus_raw_scales",
        "codec": "HDC2_global_prev_symbol_range_uint8",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": restored.q_stream == parsed.q_stream,
        "scale_roundtrip_equal": restored.scale_stream == parsed.scale_stream,
        "raw_equal": restored.to_raw() == parsed.to_raw(),
        "tensor_count": len(parsed.records),
        "context_count": stats["context_count"],
        "context_token_count": stats["context_token_count"],
        "header_bytes": stats["header_bytes"],
        "range_payload_bytes": stats["range_payload_bytes"],
        "raw_scale_bytes": len(parsed.scale_stream),
        "parity_fixture": True,
        "archive_ready": False,
        "dispatch_blockers": [
            "parity_fixture_only",
            "requires_submission_runtime_decoder",
            "requires_archive_builder_and_payload_diff",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _variant_mixed_context_global(parsed: PackedDecoderRaw) -> dict[str, Any]:
    payload, stats = encode_global_prev_symbol_mixed_context_fixture(parsed)
    restored = decode_global_prev_symbol_mixed_context_fixture(payload)
    return {
        "variant": "mixed_range_raw_global_prev_symbol_schema_indexed_q_streams_plus_raw_scales",
        "codec": "HDM2_global_prev_symbol_mixed_range_raw_schema_indexed_uint8",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": restored.q_stream == parsed.q_stream,
        "scale_roundtrip_equal": restored.scale_stream == parsed.scale_stream,
        "raw_equal": restored.to_raw() == parsed.to_raw(),
        "tensor_count": len(parsed.records),
        "context_count": stats["context_count"],
        "context_token_count": stats["context_token_count"],
        "range_context_count": stats["range_context_count"],
        "raw_context_count": stats["raw_context_count"],
        "header_bytes": stats["header_bytes"],
        "range_payload_bytes": stats["range_payload_bytes"],
        "raw_payload_bytes": stats["raw_payload_bytes"],
        "mixed_payload_bytes": stats["mixed_payload_bytes"],
        "raw_scale_bytes": len(parsed.scale_stream),
        "schema_indexed": True,
        "schema_metadata_elided_vs_hdc2_bytes": stats["schema_metadata_elided_vs_hdc2_bytes"],
        "parity_fixture": True,
        "archive_ready": False,
        "dispatch_blockers": [
            "parity_fixture_only",
            "requires_submission_runtime_decoder",
            "requires_archive_builder_and_payload_diff",
            "requires_exact_cuda_auth_eval",
        ],
    }


def encode_prev_symbol_context_range_fixture(
    parsed: PackedDecoderRaw,
) -> tuple[bytes, dict[str, int]]:
    """Encode q streams with a per-tensor previous-symbol range fixture.

    The container is intentionally self-describing and decoder-first. It is not
    an archive format until an inflate-runtime decoder and section builder exist.
    """

    out = io.BytesIO()
    out.write(b"HDC1")
    out.write(_varint_encode(1))
    out.write(_varint_encode(len(parsed.records)))
    header_bytes = 4 + 1 + len(_varint_encode(len(parsed.records)))
    range_payload_bytes = 0
    context_count = 0
    context_token_count = 0

    for record in parsed.records:
        encoded_name = record.name.encode("utf-8")
        out.write(_varint_encode(len(encoded_name)))
        out.write(encoded_name)
        out.write(_varint_encode(record.value_count))
        if not record.q_zz_u8:
            raise HnervDecoderRecodeError(f"empty q stream for record {record.name}")
        out.write(bytes([record.q_zz_u8[0]]))
        contexts = _prev_symbol_contexts(record.q_zz_u8)
        out.write(_varint_encode(len(contexts)))
        header_bytes += (
            len(_varint_encode(len(encoded_name)))
            + len(encoded_name)
            + len(_varint_encode(record.value_count))
            + 1
            + len(_varint_encode(len(contexts)))
        )
        for previous_symbol in sorted(contexts):
            stream = contexts[previous_symbol]
            encoded, local_header_bytes = _encode_context_symbol_stream(stream)
            out.write(bytes([previous_symbol]))
            out.write(_varint_encode(len(stream)))
            out.write(_varint_encode(len(encoded["symbols"])))
            out.write(_varint_encode(len(encoded["payload"])))
            previous_symbol_for_delta = 0
            for index, symbol in enumerate(encoded["symbols"]):
                delta = symbol if index == 0 else symbol - previous_symbol_for_delta
                out.write(_varint_encode(delta))
                out.write(_varint_encode(encoded["frequencies"][index]))
                previous_symbol_for_delta = symbol
            out.write(encoded["payload"])
            context_count += 1
            context_token_count += len(stream)
            range_payload_bytes += len(encoded["payload"])
            header_bytes += (
                1
                + len(_varint_encode(len(stream)))
                + len(_varint_encode(len(encoded["symbols"])))
                + len(_varint_encode(len(encoded["payload"])))
                + local_header_bytes
            )

    out.write(_varint_encode(len(parsed.scale_stream)))
    out.write(parsed.scale_stream)
    header_bytes += len(_varint_encode(len(parsed.scale_stream)))
    payload = out.getvalue()
    restored = decode_prev_symbol_context_range_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervDecoderRecodeError("HDC1 context range fixture failed raw roundtrip")
    return payload, {
        "header_bytes": header_bytes,
        "range_payload_bytes": range_payload_bytes,
        "context_count": context_count,
        "context_token_count": context_token_count,
    }


def decode_prev_symbol_context_range_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode an HDC1 parity fixture emitted by encode_prev_symbol_context_range_fixture."""

    cursor = 0
    if payload[:4] != b"HDC1":
        raise HnervDecoderRecodeError("invalid HDC1 context range fixture magic")
    cursor = 4
    version, cursor = _varint_decode(payload, cursor, "version")
    if version != 1:
        raise HnervDecoderRecodeError(f"unsupported HDC1 version: {version}")
    record_count, cursor = _varint_decode(payload, cursor, "record_count")
    records: list[PackedDecoderRecord] = []
    for _ in range(record_count):
        name_len, cursor = _varint_decode(payload, cursor, "record_name_len")
        name_bytes = _read_payload_exact(payload, cursor, name_len, "record_name")
        cursor += name_len
        name = name_bytes.decode("utf-8")
        shape = _shape_for_record_name(name)
        value_count, cursor = _varint_decode(payload, cursor, "value_count")
        if value_count != math.prod(shape):
            raise HnervDecoderRecodeError(f"HDC1 value_count mismatch for {name}")
        first = _read_payload_exact(payload, cursor, 1, "first_symbol")[0]
        cursor += 1
        context_count, cursor = _varint_decode(payload, cursor, "context_count")
        decoded_contexts: dict[int, list[int]] = {}
        for _context_index in range(context_count):
            previous_symbol = _read_payload_exact(payload, cursor, 1, "previous_symbol")[0]
            cursor += 1
            if previous_symbol in decoded_contexts:
                raise HnervDecoderRecodeError("HDC1 duplicate previous-symbol context")
            token_count, cursor = _varint_decode(payload, cursor, "context_token_count")
            unique_count, cursor = _varint_decode(payload, cursor, "unique_count")
            payload_len, cursor = _varint_decode(payload, cursor, "context_payload_len")
            symbols: list[int] = []
            frequencies: list[int] = []
            previous_symbol_for_delta = 0
            for index in range(unique_count):
                delta, cursor = _varint_decode(payload, cursor, "context_symbol_delta")
                frequency, cursor = _varint_decode(payload, cursor, "context_frequency")
                symbol = delta if index == 0 else previous_symbol_for_delta + delta
                if symbol > 0xFF:
                    raise HnervDecoderRecodeError("HDC1 context symbol exceeds uint8 range")
                symbols.append(symbol)
                frequencies.append(frequency)
                previous_symbol_for_delta = symbol
            encoded = _read_payload_exact(payload, cursor, payload_len, "context_payload")
            cursor += payload_len
            indices = decode_static_symbols(encoded, count=token_count, frequencies=frequencies)
            decoded_contexts[previous_symbol] = [symbols[int(index)] for index in indices]
        q = _restore_prev_symbol_stream(first, decoded_contexts, value_count=value_count)
        records.append(
            PackedDecoderRecord(
                name=name,
                shape=shape,
                q_zz_u8=bytes(q),
                scale_f32=b"",
            )
        )
    scale_len, cursor = _varint_decode(payload, cursor, "scale_len")
    scale_stream = _read_payload_exact(payload, cursor, scale_len, "scale_stream")
    cursor += scale_len
    if cursor != len(payload):
        raise HnervDecoderRecodeError("HDC1 fixture has trailing bytes")
    if scale_len != 4 * record_count:
        raise HnervDecoderRecodeError("HDC1 scale stream length does not match record count")
    restored_records = []
    for index, record in enumerate(records):
        restored_records.append(
            dataclasses.replace(record, scale_f32=scale_stream[index * 4 : index * 4 + 4])
        )
    return PackedDecoderRaw(records=tuple(restored_records))


def encode_global_prev_symbol_context_range_fixture(
    parsed: PackedDecoderRaw,
) -> tuple[bytes, dict[str, int]]:
    """Encode q streams with one global previous-symbol context table."""

    if not parsed.records:
        raise HnervDecoderRecodeError("HDC2 requires at least one record")
    global_contexts: dict[int, list[int]] = {}
    for record in parsed.records:
        if not record.q_zz_u8:
            raise HnervDecoderRecodeError(f"empty q stream for record {record.name}")
        for previous, current in pairwise(record.q_zz_u8):
            global_contexts.setdefault(int(previous), []).append(int(current))

    out = io.BytesIO()
    out.write(b"HDC2")
    out.write(_varint_encode(1))
    out.write(_varint_encode(len(parsed.records)))
    header_bytes = 4 + 1 + len(_varint_encode(len(parsed.records)))

    for record in parsed.records:
        encoded_name = record.name.encode("utf-8")
        out.write(_varint_encode(len(encoded_name)))
        out.write(encoded_name)
        out.write(_varint_encode(record.value_count))
        out.write(bytes([record.q_zz_u8[0]]))
        header_bytes += (
            len(_varint_encode(len(encoded_name)))
            + len(encoded_name)
            + len(_varint_encode(record.value_count))
            + 1
        )

    out.write(_varint_encode(len(global_contexts)))
    header_bytes += len(_varint_encode(len(global_contexts)))
    range_payload_bytes = 0
    context_token_count = 0
    for previous_symbol in sorted(global_contexts):
        stream = global_contexts[previous_symbol]
        encoded, local_header_bytes = _encode_context_symbol_stream(stream)
        out.write(bytes([previous_symbol]))
        out.write(_varint_encode(len(stream)))
        out.write(_varint_encode(len(encoded["symbols"])))
        out.write(_varint_encode(len(encoded["payload"])))
        previous_symbol_for_delta = 0
        for index, symbol in enumerate(encoded["symbols"]):
            delta = symbol if index == 0 else symbol - previous_symbol_for_delta
            out.write(_varint_encode(delta))
            out.write(_varint_encode(encoded["frequencies"][index]))
            previous_symbol_for_delta = symbol
        out.write(encoded["payload"])
        range_payload_bytes += len(encoded["payload"])
        context_token_count += len(stream)
        header_bytes += (
            1
            + len(_varint_encode(len(stream)))
            + len(_varint_encode(len(encoded["symbols"])))
            + len(_varint_encode(len(encoded["payload"])))
            + local_header_bytes
        )

    out.write(_varint_encode(len(parsed.scale_stream)))
    out.write(parsed.scale_stream)
    header_bytes += len(_varint_encode(len(parsed.scale_stream)))
    payload = out.getvalue()
    restored = decode_global_prev_symbol_context_range_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervDecoderRecodeError("HDC2 global context fixture failed raw roundtrip")
    return payload, {
        "header_bytes": header_bytes,
        "range_payload_bytes": range_payload_bytes,
        "context_count": len(global_contexts),
        "context_token_count": context_token_count,
    }


def encode_global_prev_symbol_mixed_context_fixture(
    parsed: PackedDecoderRaw,
) -> tuple[bytes, dict[str, int]]:
    """Encode q streams with a fixed-schema global context table and raw escapes.

    HDM2 is a planning-only HDC2-family fixture. It keeps the same global
    previous-symbol context semantics as HDC2, drops self-describing tensor
    names/counts by relying on PACKED_STATE_SCHEMA order, and stores a context
    as raw bytes when the static range model is locally more expensive.
    """

    if not parsed.records:
        raise HnervDecoderRecodeError("HDM2 requires at least one record")
    if len(parsed.records) != len(PACKED_STATE_SCHEMA):
        raise HnervDecoderRecodeError("HDM2 requires the fixed packed decoder schema")

    global_contexts: dict[int, list[int]] = {}
    for record in parsed.records:
        if not record.q_zz_u8:
            raise HnervDecoderRecodeError(f"empty q stream for record {record.name}")
        for previous, current in pairwise(record.q_zz_u8):
            global_contexts.setdefault(int(previous), []).append(int(current))

    out = io.BytesIO()
    out.write(b"HDM2")
    out.write(_varint_encode(1))
    out.write(_varint_encode(len(parsed.records)))
    header_bytes = 4 + 1 + len(_varint_encode(len(parsed.records)))
    self_describing_record_metadata_bytes = 0

    for record in parsed.records:
        expected_shape = _shape_for_record_name(record.name)
        if record.shape != expected_shape:
            raise HnervDecoderRecodeError(f"HDM2 fixed schema mismatch for {record.name}")
        encoded_name = record.name.encode("utf-8")
        self_describing_record_metadata_bytes += (
            len(_varint_encode(len(encoded_name)))
            + len(encoded_name)
            + len(_varint_encode(record.value_count))
            + 1
        )
        out.write(bytes([record.q_zz_u8[0]]))
        header_bytes += 1

    out.write(_varint_encode(len(global_contexts)))
    header_bytes += len(_varint_encode(len(global_contexts)))
    range_payload_bytes = 0
    raw_payload_bytes = 0
    range_context_count = 0
    raw_context_count = 0
    context_token_count = 0
    for previous_symbol in sorted(global_contexts):
        stream = global_contexts[previous_symbol]
        encoded, local_header_bytes = _encode_context_symbol_stream(stream)
        token_count_bytes = _varint_encode(len(stream))
        unique_count_bytes = _varint_encode(len(encoded["symbols"]))
        payload_len_bytes = _varint_encode(len(encoded["payload"]))
        range_header_bytes = (
            1
            + 1
            + len(token_count_bytes)
            + len(unique_count_bytes)
            + len(payload_len_bytes)
            + local_header_bytes
        )
        raw_header_bytes = 1 + 1 + len(token_count_bytes)
        range_total_bytes = range_header_bytes + len(encoded["payload"])
        raw_total_bytes = raw_header_bytes + len(stream)

        out.write(bytes([previous_symbol]))
        context_token_count += len(stream)
        if raw_total_bytes < range_total_bytes:
            out.write(bytes([1]))
            out.write(token_count_bytes)
            out.write(bytes(stream))
            raw_context_count += 1
            raw_payload_bytes += len(stream)
            header_bytes += raw_header_bytes
            continue

        out.write(bytes([0]))
        out.write(token_count_bytes)
        out.write(unique_count_bytes)
        out.write(payload_len_bytes)
        previous_symbol_for_delta = 0
        for index, symbol in enumerate(encoded["symbols"]):
            delta = symbol if index == 0 else symbol - previous_symbol_for_delta
            out.write(_varint_encode(delta))
            out.write(_varint_encode(encoded["frequencies"][index]))
            previous_symbol_for_delta = symbol
        out.write(encoded["payload"])
        range_context_count += 1
        range_payload_bytes += len(encoded["payload"])
        header_bytes += range_header_bytes

    out.write(_varint_encode(len(parsed.scale_stream)))
    out.write(parsed.scale_stream)
    header_bytes += len(_varint_encode(len(parsed.scale_stream)))
    payload = out.getvalue()
    restored = decode_global_prev_symbol_mixed_context_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervDecoderRecodeError("HDM2 mixed context fixture failed raw roundtrip")
    return payload, {
        "header_bytes": header_bytes,
        "range_payload_bytes": range_payload_bytes,
        "raw_payload_bytes": raw_payload_bytes,
        "mixed_payload_bytes": range_payload_bytes + raw_payload_bytes,
        "context_count": len(global_contexts),
        "range_context_count": range_context_count,
        "raw_context_count": raw_context_count,
        "context_token_count": context_token_count,
        "schema_metadata_elided_vs_hdc2_bytes": (
            self_describing_record_metadata_bytes - len(parsed.records)
        ),
    }


def decode_global_prev_symbol_context_range_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode an HDC2 global previous-symbol context fixture."""

    if payload[:4] != b"HDC2":
        raise HnervDecoderRecodeError("invalid HDC2 global context range fixture magic")
    cursor = 4
    version, cursor = _varint_decode(payload, cursor, "version")
    if version != 1:
        raise HnervDecoderRecodeError(f"unsupported HDC2 version: {version}")
    record_count, cursor = _varint_decode(payload, cursor, "record_count")
    records: list[PackedDecoderRecord] = []
    first_symbols: list[int] = []
    for _ in range(record_count):
        name_len, cursor = _varint_decode(payload, cursor, "record_name_len")
        name_bytes = _read_payload_exact(payload, cursor, name_len, "record_name")
        cursor += name_len
        name = name_bytes.decode("utf-8")
        shape = _shape_for_record_name(name)
        value_count, cursor = _varint_decode(payload, cursor, "value_count")
        if value_count != math.prod(shape):
            raise HnervDecoderRecodeError(f"HDC2 value_count mismatch for {name}")
        first = _read_payload_exact(payload, cursor, 1, "first_symbol")[0]
        cursor += 1
        records.append(PackedDecoderRecord(name=name, shape=shape, q_zz_u8=b"", scale_f32=b""))
        first_symbols.append(first)

    context_count, cursor = _varint_decode(payload, cursor, "context_count")
    decoded_contexts: dict[int, list[int]] = {}
    for _context_index in range(context_count):
        previous_symbol = _read_payload_exact(payload, cursor, 1, "previous_symbol")[0]
        cursor += 1
        if previous_symbol in decoded_contexts:
            raise HnervDecoderRecodeError("HDC2 duplicate previous-symbol context")
        token_count, cursor = _varint_decode(payload, cursor, "context_token_count")
        unique_count, cursor = _varint_decode(payload, cursor, "unique_count")
        payload_len, cursor = _varint_decode(payload, cursor, "context_payload_len")
        symbols: list[int] = []
        frequencies: list[int] = []
        previous_symbol_for_delta = 0
        for index in range(unique_count):
            delta, cursor = _varint_decode(payload, cursor, "context_symbol_delta")
            frequency, cursor = _varint_decode(payload, cursor, "context_frequency")
            symbol = delta if index == 0 else previous_symbol_for_delta + delta
            if symbol > 0xFF:
                raise HnervDecoderRecodeError("HDC2 context symbol exceeds uint8 range")
            symbols.append(symbol)
            frequencies.append(frequency)
            previous_symbol_for_delta = symbol
        encoded = _read_payload_exact(payload, cursor, payload_len, "context_payload")
        cursor += payload_len
        indices = decode_static_symbols(encoded, count=token_count, frequencies=frequencies)
        decoded_contexts[previous_symbol] = [symbols[int(index)] for index in indices]

    restored_records = []
    context_offsets = dict.fromkeys(decoded_contexts, 0)
    for record, first in zip(records, first_symbols, strict=True):
        q, context_offsets = _restore_prev_symbol_stream_with_offsets(
            first,
            decoded_contexts,
            context_offsets,
            value_count=record.value_count,
        )
        restored_records.append(dataclasses.replace(record, q_zz_u8=q))
    for symbol, values in decoded_contexts.items():
        if context_offsets[symbol] != len(values):
            raise HnervDecoderRecodeError("HDC2 previous-symbol context has trailing values")

    scale_len, cursor = _varint_decode(payload, cursor, "scale_len")
    scale_stream = _read_payload_exact(payload, cursor, scale_len, "scale_stream")
    cursor += scale_len
    if cursor != len(payload):
        raise HnervDecoderRecodeError("HDC2 fixture has trailing bytes")
    if scale_len != 4 * record_count:
        raise HnervDecoderRecodeError("HDC2 scale stream length does not match record count")
    with_scales = []
    for index, record in enumerate(restored_records):
        with_scales.append(
            dataclasses.replace(record, scale_f32=scale_stream[index * 4 : index * 4 + 4])
        )
    return PackedDecoderRaw(records=tuple(with_scales))


def decode_global_prev_symbol_mixed_context_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode an HDM2 global previous-symbol mixed range/raw fixture."""

    if payload[:4] != b"HDM2":
        raise HnervDecoderRecodeError("invalid HDM2 mixed context fixture magic")
    cursor = 4
    version, cursor = _varint_decode(payload, cursor, "version")
    if version != 1:
        raise HnervDecoderRecodeError(f"unsupported HDM2 version: {version}")
    record_count, cursor = _varint_decode(payload, cursor, "record_count")
    if record_count != len(PACKED_STATE_SCHEMA):
        raise HnervDecoderRecodeError("HDM2 fixed schema record count mismatch")
    records = [
        PackedDecoderRecord(name=name, shape=shape, q_zz_u8=b"", scale_f32=b"")
        for name, shape in PACKED_STATE_SCHEMA
    ]
    first_symbols = []
    for _ in range(record_count):
        first_symbols.append(_read_payload_exact(payload, cursor, 1, "first_symbol")[0])
        cursor += 1

    context_count, cursor = _varint_decode(payload, cursor, "context_count")
    decoded_contexts: dict[int, list[int]] = {}
    for _context_index in range(context_count):
        previous_symbol = _read_payload_exact(payload, cursor, 1, "previous_symbol")[0]
        cursor += 1
        if previous_symbol in decoded_contexts:
            raise HnervDecoderRecodeError("HDM2 duplicate previous-symbol context")
        mode = _read_payload_exact(payload, cursor, 1, "context_mode")[0]
        cursor += 1
        token_count, cursor = _varint_decode(payload, cursor, "context_token_count")
        if mode == 1:
            raw = _read_payload_exact(payload, cursor, token_count, "raw_context_payload")
            cursor += token_count
            decoded_contexts[previous_symbol] = list(raw)
            continue
        if mode != 0:
            raise HnervDecoderRecodeError(f"unsupported HDM2 context mode: {mode}")
        unique_count, cursor = _varint_decode(payload, cursor, "unique_count")
        payload_len, cursor = _varint_decode(payload, cursor, "context_payload_len")
        symbols: list[int] = []
        frequencies: list[int] = []
        previous_symbol_for_delta = 0
        for index in range(unique_count):
            delta, cursor = _varint_decode(payload, cursor, "context_symbol_delta")
            frequency, cursor = _varint_decode(payload, cursor, "context_frequency")
            symbol = delta if index == 0 else previous_symbol_for_delta + delta
            if symbol > 0xFF:
                raise HnervDecoderRecodeError("HDM2 context symbol exceeds uint8 range")
            symbols.append(symbol)
            frequencies.append(frequency)
            previous_symbol_for_delta = symbol
        encoded = _read_payload_exact(payload, cursor, payload_len, "context_payload")
        cursor += payload_len
        indices = decode_static_symbols(encoded, count=token_count, frequencies=frequencies)
        decoded_contexts[previous_symbol] = [symbols[int(index)] for index in indices]

    restored_records = []
    context_offsets = dict.fromkeys(decoded_contexts, 0)
    for record, first in zip(records, first_symbols, strict=True):
        q, context_offsets = _restore_prev_symbol_stream_with_offsets(
            first,
            decoded_contexts,
            context_offsets,
            value_count=record.value_count,
        )
        restored_records.append(dataclasses.replace(record, q_zz_u8=q))
    for symbol, values in decoded_contexts.items():
        if context_offsets[symbol] != len(values):
            raise HnervDecoderRecodeError("HDM2 previous-symbol context has trailing values")

    scale_len, cursor = _varint_decode(payload, cursor, "scale_len")
    scale_stream = _read_payload_exact(payload, cursor, scale_len, "scale_stream")
    cursor += scale_len
    if cursor != len(payload):
        raise HnervDecoderRecodeError("HDM2 fixture has trailing bytes")
    if scale_len != 4 * record_count:
        raise HnervDecoderRecodeError("HDM2 scale stream length does not match record count")
    with_scales = []
    for index, record in enumerate(restored_records):
        with_scales.append(
            dataclasses.replace(record, scale_f32=scale_stream[index * 4 : index * 4 + 4])
        )
    return PackedDecoderRaw(records=tuple(with_scales))


def _encode_context_symbol_stream(stream: list[int]) -> tuple[dict[str, Any], int]:
    counts = Counter(stream)
    symbols = sorted(counts)
    symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
    indices = [symbol_to_index[symbol] for symbol in stream]
    frequencies = [int(counts[symbol]) for symbol in symbols]
    payload = encode_static_symbols(indices, frequencies=frequencies)
    restored = decode_static_symbols(payload, count=len(indices), frequencies=frequencies)
    if [symbols[int(index)] for index in restored] != stream:
        raise HnervDecoderRecodeError("context range stream failed local roundtrip")
    header_bytes = 0
    previous_symbol = 0
    for index, symbol in enumerate(symbols):
        delta = symbol if index == 0 else symbol - previous_symbol
        header_bytes += len(_varint_encode(delta)) + len(_varint_encode(counts[symbol]))
        previous_symbol = symbol
    return {
        "symbols": symbols,
        "frequencies": frequencies,
        "payload": payload,
    }, header_bytes


def _restore_prev_symbol_stream(
    first_symbol: int,
    decoded_contexts: dict[int, list[int]],
    *,
    value_count: int,
) -> bytes:
    if value_count <= 0:
        raise HnervDecoderRecodeError("HDC1 value_count must be positive")
    out = bytearray([first_symbol])
    offsets = dict.fromkeys(decoded_contexts, 0)
    previous = first_symbol
    while len(out) < value_count:
        values = decoded_contexts.get(previous)
        if values is None:
            raise HnervDecoderRecodeError("HDC1 missing previous-symbol context")
        offset = offsets[previous]
        if offset >= len(values):
            raise HnervDecoderRecodeError("HDC1 previous-symbol context exhausted early")
        current = int(values[offset])
        offsets[previous] = offset + 1
        out.append(current)
        previous = current
    for symbol, values in decoded_contexts.items():
        if offsets[symbol] != len(values):
            raise HnervDecoderRecodeError("HDC1 previous-symbol context has trailing values")
    return bytes(out)


def _restore_prev_symbol_stream_with_offsets(
    first_symbol: int,
    decoded_contexts: dict[int, list[int]],
    offsets: dict[int, int],
    *,
    value_count: int,
) -> tuple[bytes, dict[int, int]]:
    if value_count <= 0:
        raise HnervDecoderRecodeError("HDC2 value_count must be positive")
    out = bytearray([first_symbol])
    previous = first_symbol
    while len(out) < value_count:
        values = decoded_contexts.get(previous)
        if values is None:
            raise HnervDecoderRecodeError("HDC2 missing previous-symbol context")
        offset = offsets[previous]
        if offset >= len(values):
            raise HnervDecoderRecodeError("HDC2 previous-symbol context exhausted early")
        current = int(values[offset])
        offsets[previous] = offset + 1
        out.append(current)
        previous = current
    return bytes(out), offsets


def _shape_for_record_name(name: str) -> tuple[int, ...]:
    for schema_name, shape in PACKED_STATE_SCHEMA:
        if schema_name == name:
            return shape
    raise HnervDecoderRecodeError(f"unknown HDC1 record name: {name}")


def _varint_encode(value: int) -> bytes:
    if value < 0:
        raise HnervDecoderRecodeError("varint value must be non-negative")
    out = bytearray()
    remaining = int(value)
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _varint_decode(payload: bytes, offset: int, label: str) -> tuple[int, int]:
    value = 0
    shift = 0
    cursor = offset
    while True:
        if cursor >= len(payload):
            raise HnervDecoderRecodeError(f"HDC1 truncated while reading {label}")
        byte = payload[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            raise HnervDecoderRecodeError(f"HDC1 {label} exceeds supported bounds")


def _read_payload_exact(payload: bytes, offset: int, length: int, label: str) -> bytes:
    end = offset + length
    if length < 0 or end > len(payload):
        raise HnervDecoderRecodeError(f"HDC1 truncated while reading {label}")
    return payload[offset:end]


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
    "decode_global_prev_symbol_context_range_fixture",
    "decode_global_prev_symbol_mixed_context_fixture",
    "decode_prev_symbol_context_range_fixture",
    "encode_global_prev_symbol_context_range_fixture",
    "encode_global_prev_symbol_mixed_context_fixture",
    "encode_prev_symbol_context_range_fixture",
    "parse_packed_decoder_brotli",
]
