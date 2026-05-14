"""Planning-only structural recode probes for PR106-style HNeRV decoders."""

from __future__ import annotations

import dataclasses
import io
import math
import os
import struct
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
HDM4_RECIPE_ID = 1
HDM4_RECIPE_NAME = "conv3x3_then_1x1_then_bias_tail_dp4"
HDM4_SPLIT_POINTS = (6, 9, 26, 28)
HDM6_RECIPE_ID = 1
HDM6_RECIPE_NAME = "hdm4_order_dp4_with_fixed_lgwin_tuning"
HDM6_CHUNK_BROTLI_PARAMS = (
    {"quality": 11, "lgwin": 18, "mode": brotli.MODE_GENERIC},
    {"quality": 11, "lgwin": 16, "mode": brotli.MODE_GENERIC},
    {"quality": 11, "lgwin": 16, "mode": brotli.MODE_GENERIC},
    {"quality": 10, "lgwin": 16, "mode": brotli.MODE_GENERIC},
)
HDM5_SCHEMA_VERSION = 1


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


@dataclasses.dataclass(frozen=True)
class Hdm5OrderFamily:
    family_id: int
    name: str
    records: tuple[PackedDecoderRecord, ...]
    rationale: str


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


def parse_decoder_section_for_recode(decoder_section: bytes) -> tuple[PackedDecoderRaw, bytes, str]:
    """Parse a legacy/HDM decoder section for lossless recode planning.

    This keeps profiler/search tools apples-to-apples with the current archive
    surface: an HDM4 release packet must be compared from its consumed HDM4
    bytes, not accidentally rejected or converted to a nearby legacy Brotli
    substrate before profiling.
    """

    if decoder_section.startswith(b"HDM3"):
        parsed = decode_hdm3_q_brotli_split_fixture(decoder_section)
        return parsed, parsed.to_raw(), "hdm3_q_brotli_split"
    if decoder_section.startswith(b"HDM4"):
        parsed = decode_hdm4_q_brotli_split_fixture(decoder_section)
        return parsed, parsed.to_raw(), "hdm4_q_brotli_split"
    if decoder_section.startswith(b"HDM6"):
        parsed = decode_hdm6_q_brotli_tuned_fixture(decoder_section)
        return parsed, parsed.to_raw(), "hdm6_q_brotli_tuned_split"
    parsed = parse_packed_decoder_brotli(decoder_section)
    return parsed, parsed.to_raw(), "legacy_brotli_packed_decoder"


def build_structural_recode_profile(
    packed: PackedHnervPayload,
    *,
    source_label: str,
    source_archive_sha256: str,
    include_hdm5_search: bool = False,
    hdm5_max_parts: int = 8,
    hdm5_workers: int | None = None,
    hdm5_top_k: int = 16,
) -> dict[str, Any]:
    """Profile lossless structural recodes for a PR106-style decoder section."""

    parsed, source_raw, source_decoder_codec = parse_decoder_section_for_recode(
        packed.decoder_packed_brotli
    )
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
        _variant_hdm3_q_brotli_split(parsed),
        _variant_hdm4_q_brotli_split_dp(parsed),
        _variant_hdm6_q_brotli_tuned(parsed),
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
        if (
            row["variant"]
            == "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales"
        ):
            row["byte_gap_vs_per_tensor_prev_symbol_entropy_floor_plus_raw_scales"] = int(
                row["bytes"]
            ) - int(entropy_summary["per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes"])
        if (
            row["variant"]
            == "hdm6_q_brotli_split_fixed_recipe_tuned_lgwin_plus_raw_scales"
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
    profile = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_decoder_recode.build_structural_recode_profile",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_label": source_label,
        "source_archive_sha256": source_archive_sha256,
        "source_decoder_section_codec": source_decoder_codec,
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
    if include_hdm5_search:
        profile["hdm5_search"] = search_hdm5_q_brotli_split_recipes(
            parsed,
            baseline_section_bytes=len(source_brotli),
            hdm4_section_bytes=len(encode_hdm4_q_brotli_split_fixture(parsed)[0]),
            quality=11,
            max_parts=hdm5_max_parts,
            workers=hdm5_workers,
            top_k=hdm5_top_k,
        )
    return profile


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


def _variant_hdm3_q_brotli_split(parsed: PackedDecoderRaw) -> dict[str, Any]:
    payload, stats = encode_hdm3_q_brotli_split_fixture(parsed)
    restored = decode_hdm3_q_brotli_split_fixture(payload)
    return {
        "variant": "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales",
        "codec": "HDM3_fixed_schema_q_brotli_raw_scales",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": restored.q_stream == parsed.q_stream,
        "scale_roundtrip_equal": restored.scale_stream == parsed.scale_stream,
        "raw_equal": restored.to_raw() == parsed.to_raw(),
        "tensor_count": len(parsed.records),
        "q_brotli_bytes": stats["q_brotli_bytes"],
        "q_stream_bytes": stats["q_stream_bytes"],
        "raw_scale_bytes": stats["raw_scale_bytes"],
        "header_bytes": stats["header_bytes"],
        "brotli_quality": stats["brotli_quality"],
        "parity_fixture": True,
        "archive_ready": False,
        "dispatch_blockers": [
            "parity_fixture_only",
            "requires_submission_runtime_decoder",
            "requires_archive_builder_and_payload_diff",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _variant_hdm4_q_brotli_split_dp(parsed: PackedDecoderRaw) -> dict[str, Any]:
    payload, stats = encode_hdm4_q_brotli_split_fixture(parsed)
    restored = decode_hdm4_q_brotli_split_fixture(payload)
    return {
        "variant": "hdm4_q_brotli_split_fixed_recipe_dp4_plus_raw_scales",
        "codec": "HDM4_fixed_recipe_dp4_q_brotli_raw_scales",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": restored.q_stream == parsed.q_stream,
        "scale_roundtrip_equal": restored.scale_stream == parsed.scale_stream,
        "raw_equal": restored.to_raw() == parsed.to_raw(),
        "tensor_count": len(parsed.records),
        "recipe_id": stats["recipe_id"],
        "recipe_name": stats["recipe_name"],
        "split_points": stats["split_points"],
        "q_brotli_bytes": stats["q_brotli_bytes"],
        "q_stream_bytes": stats["q_stream_bytes"],
        "raw_scale_bytes": stats["raw_scale_bytes"],
        "header_bytes": stats["header_bytes"],
        "brotli_quality": stats["brotli_quality"],
        "non_arbitrary_selection": stats["non_arbitrary_selection"],
        "parity_fixture": True,
        "archive_ready": False,
        "dispatch_blockers": [
            "parity_fixture_only",
            "requires_submission_runtime_decoder",
            "requires_archive_builder_and_payload_diff",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _variant_hdm6_q_brotli_tuned(parsed: PackedDecoderRaw) -> dict[str, Any]:
    payload, stats = encode_hdm6_q_brotli_tuned_fixture(parsed)
    restored = decode_hdm6_q_brotli_tuned_fixture(payload)
    return {
        "variant": "hdm6_q_brotli_split_fixed_recipe_tuned_lgwin_plus_raw_scales",
        "codec": "HDM6_fixed_recipe_tuned_q_brotli_raw_scales",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "q_roundtrip_equal": restored.q_stream == parsed.q_stream,
        "scale_roundtrip_equal": restored.scale_stream == parsed.scale_stream,
        "raw_equal": restored.to_raw() == parsed.to_raw(),
        "tensor_count": len(parsed.records),
        "recipe_id": stats["recipe_id"],
        "recipe_name": stats["recipe_name"],
        "split_points": stats["split_points"],
        "q_brotli_bytes": stats["q_brotli_bytes"],
        "q_stream_bytes": stats["q_stream_bytes"],
        "raw_scale_bytes": stats["raw_scale_bytes"],
        "header_bytes": stats["header_bytes"],
        "brotli_params_by_chunk": stats["brotli_params_by_chunk"],
        "non_arbitrary_selection": stats["non_arbitrary_selection"],
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


def encode_hdm3_q_brotli_split_fixture(
    parsed: PackedDecoderRaw,
    *,
    quality: int = 11,
) -> tuple[bytes, dict[str, Any]]:
    """Encode fixed-schema q bytes as Brotli plus raw fp32 scales.

    HDM3 is a deterministic planning/runtime fixture, not an archive claim. It
    removes all tensor-name/count metadata by relying on PACKED_STATE_SCHEMA,
    stores Brotli(q_stream, quality=11), and appends the raw scale stream.
    """

    q_stream = parsed.q_stream
    scale_stream = parsed.scale_stream
    compressed = brotli.compress(q_stream, quality=quality)
    if len(compressed) > 0xFFFFFF:
        raise HnervDecoderRecodeError("HDM3 q Brotli payload exceeds len24")
    out = io.BytesIO()
    out.write(b"HDM3")
    out.write(len(compressed).to_bytes(3, "little"))
    out.write(compressed)
    out.write(scale_stream)
    payload = out.getvalue()
    restored = decode_hdm3_q_brotli_split_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervDecoderRecodeError("HDM3 q Brotli split fixture failed raw roundtrip")
    return payload, {
        "header_bytes": 7,
        "brotli_quality": quality,
        "q_brotli_bytes": len(compressed),
        "q_stream_bytes": len(q_stream),
        "raw_scale_bytes": len(scale_stream),
    }


def encode_hdm4_q_brotli_split_fixture(
    parsed: PackedDecoderRaw,
    *,
    quality: int = 11,
    recipe_id: int = HDM4_RECIPE_ID,
) -> tuple[bytes, dict[str, Any]]:
    """Encode fixed-schema q bytes as a deterministic DP-selected Brotli split.

    HDM4 is the first post-HDM3 deterministic byte-only candidate. The stored
    recipe is intentionally tiny: a one-byte recipe id selects a fixed record
    order and contiguous split points, followed by len24-prefixed Brotli chunks
    and the raw fp32 scale stream. Recipe 1 is the best exact-lossless row found
    by exhaustive DP over the declared order family and k<=8 contiguous splits
    on the PR106-R2 current frontier; it is a reproducible selection rule, not
    an arbitrary constant.
    """

    ordered_schema = _hdm4_ordered_schema(recipe_id)
    record_by_name = {record.name: record for record in parsed.records}
    missing = [name for name, _shape in ordered_schema if name not in record_by_name]
    if missing:
        raise HnervDecoderRecodeError("HDM4 fixed recipe missing records: " + ", ".join(missing))
    if len(parsed.records) != len(ordered_schema):
        raise HnervDecoderRecodeError("HDM4 fixed recipe requires the packed decoder schema")

    ordered_records = []
    for name, shape in ordered_schema:
        record = record_by_name[name]
        if record.shape != shape:
            raise HnervDecoderRecodeError(f"HDM4 fixed recipe shape mismatch for {name}")
        ordered_records.append(record)

    previous = 0
    compressed_parts: list[bytes] = []
    for split in HDM4_SPLIT_POINTS:
        q_stream = b"".join(record.q_zz_u8 for record in ordered_records[previous:split])
        compressed = brotli.compress(q_stream, quality=quality)
        if len(compressed) > 0xFFFFFF:
            raise HnervDecoderRecodeError("HDM4 q Brotli chunk exceeds len24")
        compressed_parts.append(compressed)
        previous = split
    scale_stream = parsed.scale_stream
    out = io.BytesIO()
    out.write(b"HDM4")
    out.write(bytes([recipe_id]))
    for compressed in compressed_parts:
        out.write(len(compressed).to_bytes(3, "little"))
    for compressed in compressed_parts:
        out.write(compressed)
    out.write(scale_stream)
    payload = out.getvalue()
    restored = decode_hdm4_q_brotli_split_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervDecoderRecodeError("HDM4 q Brotli split fixture failed raw roundtrip")
    return payload, {
        "header_bytes": 4 + 1 + 3 * len(compressed_parts),
        "brotli_quality": quality,
        "recipe_id": recipe_id,
        "recipe_name": HDM4_RECIPE_NAME,
        "split_points": list(HDM4_SPLIT_POINTS),
        "q_brotli_bytes": sum(len(part) for part in compressed_parts),
        "q_chunk_bytes": [len(part) for part in compressed_parts],
        "q_stream_bytes": sum(len(record.q_zz_u8) for record in ordered_records),
        "raw_scale_bytes": len(scale_stream),
        "ordered_record_names": [record.name for record in ordered_records],
        "non_arbitrary_selection": {
            "search_family": "fixed_schema_record_orderings_x_contiguous_brotli_partitions",
            "objective": "minimize total decoder section bytes including fixed runtime header",
            "quality": quality,
            "candidate_order_family_size": 7,
            "max_partitions_considered": 8,
            "selected_by": "exhaustive_dynamic_programming_probe_20260513",
        },
    }


def encode_hdm6_q_brotli_tuned_fixture(
    parsed: PackedDecoderRaw,
    *,
    recipe_id: int = HDM6_RECIPE_ID,
) -> tuple[bytes, dict[str, Any]]:
    """Encode fixed-schema q bytes using the HDM4 order with tuned Brotli params.

    HDM6 is intentionally a tiny runtime-specialized step beyond HDM4: it
    preserves HDM4's proven order and split recipe, but fixes per-chunk Brotli
    encoder parameters from an exhaustive local grid over quality/lgwin on the
    PR106-R2 HDM4+HLM2 decoder section. The decoder needs no parameter
    metadata because Brotli streams are self-describing; the recipe id is only
    a fail-closed contract for the ordering and split points.
    """

    if recipe_id != HDM6_RECIPE_ID:
        raise HnervDecoderRecodeError(f"unsupported HDM6 recipe id: {recipe_id}")
    ordered_schema = _hdm4_ordered_schema(HDM4_RECIPE_ID)
    record_by_name = {record.name: record for record in parsed.records}
    missing = [name for name, _shape in ordered_schema if name not in record_by_name]
    if missing:
        raise HnervDecoderRecodeError("HDM6 fixed recipe missing records: " + ", ".join(missing))
    if len(parsed.records) != len(ordered_schema):
        raise HnervDecoderRecodeError("HDM6 fixed recipe requires the packed decoder schema")

    ordered_records = []
    for name, shape in ordered_schema:
        record = record_by_name[name]
        if record.shape != shape:
            raise HnervDecoderRecodeError(f"HDM6 fixed recipe shape mismatch for {name}")
        ordered_records.append(record)

    if len(HDM6_CHUNK_BROTLI_PARAMS) != len(HDM4_SPLIT_POINTS):
        raise HnervDecoderRecodeError("HDM6 chunk parameter count does not match split recipe")
    previous = 0
    compressed_parts: list[bytes] = []
    params_by_chunk: list[dict[str, int]] = []
    for split, params in zip(HDM4_SPLIT_POINTS, HDM6_CHUNK_BROTLI_PARAMS, strict=True):
        q_stream = b"".join(record.q_zz_u8 for record in ordered_records[previous:split])
        quality = int(params["quality"])
        lgwin = int(params["lgwin"])
        mode = int(params.get("mode", brotli.MODE_GENERIC))
        compressed = brotli.compress(q_stream, quality=quality, lgwin=lgwin, mode=mode)
        if len(compressed) > 0xFFFFFF:
            raise HnervDecoderRecodeError("HDM6 q Brotli chunk exceeds len24")
        compressed_parts.append(compressed)
        params_by_chunk.append({"quality": quality, "lgwin": lgwin, "mode": mode})
        previous = split
    scale_stream = parsed.scale_stream
    out = io.BytesIO()
    out.write(b"HDM6")
    out.write(bytes([recipe_id]))
    for compressed in compressed_parts:
        out.write(len(compressed).to_bytes(3, "little"))
    for compressed in compressed_parts:
        out.write(compressed)
    out.write(scale_stream)
    payload = out.getvalue()
    restored = decode_hdm6_q_brotli_tuned_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervDecoderRecodeError("HDM6 q Brotli tuned fixture failed raw roundtrip")
    return payload, {
        "header_bytes": 4 + 1 + 3 * len(compressed_parts),
        "recipe_id": recipe_id,
        "recipe_name": HDM6_RECIPE_NAME,
        "split_points": list(HDM4_SPLIT_POINTS),
        "q_brotli_bytes": sum(len(part) for part in compressed_parts),
        "q_chunk_bytes": [len(part) for part in compressed_parts],
        "q_stream_bytes": sum(len(record.q_zz_u8) for record in ordered_records),
        "raw_scale_bytes": len(scale_stream),
        "ordered_record_names": [record.name for record in ordered_records],
        "brotli_params_by_chunk": params_by_chunk,
        "non_arbitrary_selection": {
            "search_family": "hdm4_fixed_order_split_x_brotli_quality_lgwin_grid",
            "objective": "minimize decoder section bytes including fixed runtime header",
            "quality_values_considered": list(range(12)),
            "lgwin_values_considered": list(range(10, 25)),
            "mode_values_considered": [
                {"name": "generic", "value": brotli.MODE_GENERIC},
                {"name": "text", "value": brotli.MODE_TEXT},
                {"name": "font", "value": brotli.MODE_FONT},
            ],
            "selected_mode_by_chunk": ["generic"] * len(params_by_chunk),
            "selected_by": "exhaustive_param_grid_probe_20260514",
            "baseline": "HDM4 recipe 1",
        },
    }


def encode_hdm5_q_brotli_split_planning_fixture(
    parsed: PackedDecoderRaw,
    *,
    ordered_record_names: tuple[str, ...],
    split_points: tuple[int, ...],
    quality: int = 11,
) -> tuple[bytes, dict[str, Any]]:
    """Encode a planning-only self-describing HDM5 q-Brotli split fixture.

    HDM5 is deliberately conservative: the payload stores the exact record
    permutation plus split endpoints, so byte accounting does not depend on a
    future runtime hard-coding a recipe ID. If this self-describing fixture
    cannot beat HDM4 bytes, a runtime-specialized HDM5 packet should not be
    promoted without new evidence.
    """

    ordered_records = _ordered_records_from_names(parsed, ordered_record_names)
    split_points = _validate_split_points(split_points, record_count=len(ordered_records))
    compressed_parts: list[bytes] = []
    previous = 0
    for split in split_points:
        q_stream = b"".join(record.q_zz_u8 for record in ordered_records[previous:split])
        compressed = brotli.compress(q_stream, quality=quality)
        if len(compressed) > 0xFFFFFF:
            raise HnervDecoderRecodeError("HDM5 q Brotli chunk exceeds len24")
        compressed_parts.append(compressed)
        previous = split

    schema_index_by_name = {name: index for index, (name, _shape) in enumerate(PACKED_STATE_SCHEMA)}
    order_indices = bytes(schema_index_by_name[record.name] for record in ordered_records)
    scale_stream = parsed.scale_stream
    out = io.BytesIO()
    out.write(b"HDM5")
    out.write(bytes([HDM5_SCHEMA_VERSION]))
    out.write(bytes([len(ordered_records)]))
    out.write(bytes([len(split_points)]))
    out.write(order_indices)
    out.write(bytes(split_points))
    for compressed in compressed_parts:
        out.write(len(compressed).to_bytes(3, "little"))
    for compressed in compressed_parts:
        out.write(compressed)
    out.write(scale_stream)
    payload = out.getvalue()
    restored = decode_hdm5_q_brotli_split_planning_fixture(payload)
    if restored.to_raw() != parsed.to_raw():
        raise HnervDecoderRecodeError("HDM5 q Brotli split fixture failed raw roundtrip")

    header_bytes = 4 + 1 + 1 + 1 + len(order_indices) + len(split_points) + 3 * len(split_points)
    return payload, {
        "header_bytes": header_bytes,
        "brotli_quality": quality,
        "split_points": list(split_points),
        "part_count": len(split_points),
        "q_brotli_bytes": sum(len(part) for part in compressed_parts),
        "q_chunk_bytes": [len(part) for part in compressed_parts],
        "q_stream_bytes": sum(len(record.q_zz_u8) for record in ordered_records),
        "raw_scale_bytes": len(scale_stream),
        "record_order_metadata_bytes": len(order_indices),
        "split_metadata_bytes": len(split_points),
        "length_prefix_bytes": 3 * len(split_points),
        "ordered_record_names": [record.name for record in ordered_records],
        "self_describing_order": True,
        "planning_only": True,
    }


def search_hdm5_q_brotli_split_recipes(
    parsed: PackedDecoderRaw,
    *,
    baseline_section_bytes: int | None = None,
    hdm4_section_bytes: int | None = None,
    quality: int = 11,
    max_parts: int = 8,
    workers: int | None = None,
    top_k: int = 16,
) -> dict[str, Any]:
    """Search deterministic HDM5 record orders and contiguous Brotli partitions.

    The search is exact for each declared order family: every contiguous segment
    cost is measured with Brotli, and dynamic programming chooses the minimum
    byte partition for each part count. It emits planning evidence only; a
    runtime, archive build, no-op proof, and exact eval remain mandatory.
    """

    if max_parts < 1:
        raise HnervDecoderRecodeError("HDM5 max_parts must be positive")
    families = _hdm5_order_families(parsed)
    hdm4_bytes = (
        int(hdm4_section_bytes)
        if hdm4_section_bytes is not None
        else len(encode_hdm4_q_brotli_split_fixture(parsed, quality=quality)[0])
    )
    baseline_bytes = int(baseline_section_bytes) if baseline_section_bytes is not None else hdm4_bytes
    effective_workers = _resolve_hdm5_workers(workers)
    family_rows: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for family in families:
        rows = _search_hdm5_family(
            parsed,
            family,
            quality=quality,
            max_parts=max_parts,
            workers=effective_workers,
            baseline_section_bytes=baseline_bytes,
            hdm4_section_bytes=hdm4_bytes,
        )
        family_best = min(rows, key=lambda row: (int(row["bytes"]), int(row["part_count"])))
        family_rows.append(
            {
                "family_id": family.family_id,
                "family_name": family.name,
                "rationale": family.rationale,
                "best_candidate": family_best,
                "candidate_count": len(rows),
            }
        )
        all_candidates.extend(rows)

    all_candidates.sort(
        key=lambda row: (
            int(row["bytes"]),
            int(row["part_count"]),
            int(row["family_id"]),
            str(row["family_name"]),
        )
    )
    best = all_candidates[0]
    best_fixed_recipe_projection = min(
        all_candidates,
        key=lambda row: (
            int(row["fixed_recipe_projected_bytes"]),
            int(row["part_count"]),
            int(row["family_id"]),
            str(row["family_name"]),
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_decoder_recode.search_hdm5_q_brotli_split_recipes",
        "score_claim": False,
        "dispatch_attempted": False,
        "archive_ready": False,
        "ready_for_exact_eval_dispatch": False,
        "planning_only": True,
        "codec": "HDM5_planning_self_describing_order_split_q_brotli_raw_scales",
        "quality": quality,
        "max_parts": max_parts,
        "workers": effective_workers,
        "order_family_count": len(families),
        "candidate_count": len(all_candidates),
        "baseline_section_bytes": baseline_bytes,
        "hdm4_section_bytes": hdm4_bytes,
        "best_candidate": best,
        "best_fixed_recipe_projection": best_fixed_recipe_projection,
        "top_candidates": all_candidates[: max(1, int(top_k))],
        "family_summaries": family_rows,
        "verdict": (
            "hdm5_self_describing_candidate_beats_hdm4_plan_runtime_next"
            if int(best["byte_delta_vs_hdm4_section"]) < 0
            else "hdm5_fixed_recipe_candidate_beats_hdm4_plan_runtime_next"
            if int(best_fixed_recipe_projection["fixed_recipe_projected_byte_delta_vs_hdm4_section"]) < 0
            else "hdm5_self_describing_search_does_not_beat_hdm4"
        ),
        "dispatch_blockers": [
            "planning_only_hdm5_search",
            "requires_runtime_decoder_implementation_if_positive",
            "requires_archive_builder_and_payload_diff_if_positive",
            "requires_exact_cuda_auth_eval_if_positive",
        ],
    }


def decode_hdm5_q_brotli_split_planning_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode a self-describing HDM5 planning fixture."""

    if payload[:4] != b"HDM5":
        raise HnervDecoderRecodeError("invalid HDM5 q Brotli split fixture magic")
    cursor = 4
    version = _read_payload_exact(payload, cursor, 1, "HDM5 version")[0]
    cursor += 1
    if version != HDM5_SCHEMA_VERSION:
        raise HnervDecoderRecodeError(f"unsupported HDM5 version: {version}")
    record_count = _read_payload_exact(payload, cursor, 1, "HDM5 record_count")[0]
    cursor += 1
    if record_count != len(PACKED_STATE_SCHEMA):
        raise HnervDecoderRecodeError("HDM5 record count mismatch")
    part_count = _read_payload_exact(payload, cursor, 1, "HDM5 part_count")[0]
    cursor += 1
    if part_count < 1 or part_count > record_count:
        raise HnervDecoderRecodeError("HDM5 part count out of range")
    order_indices = _read_payload_exact(payload, cursor, record_count, "HDM5 order_indices")
    cursor += record_count
    if sorted(order_indices) != list(range(record_count)):
        raise HnervDecoderRecodeError("HDM5 order indices are not a schema permutation")
    split_points = tuple(
        _read_payload_exact(payload, cursor, part_count, "HDM5 split_points")
    )
    cursor += part_count
    split_points = _validate_split_points(split_points, record_count=record_count)
    lengths = []
    for index in range(part_count):
        lengths.append(
            int.from_bytes(
                _read_payload_exact(payload, cursor, 3, f"HDM5 q_brotli_len24_{index}"),
                "little",
            )
        )
        cursor += 3

    ordered_schema = tuple(PACKED_STATE_SCHEMA[index] for index in order_indices)
    ordered_records: list[PackedDecoderRecord] = []
    split_start = 0
    for index, (length, split_end) in enumerate(zip(lengths, split_points, strict=True)):
        compressed = _read_payload_exact(payload, cursor, length, f"HDM5 q_brotli_{index}")
        cursor += length
        try:
            q_chunk = brotli.decompress(compressed)
        except brotli.error as exc:
            raise HnervDecoderRecodeError(
                f"HDM5 q stream chunk {index} brotli decode failed: {exc}"
            ) from exc
        schema_slice = ordered_schema[split_start:split_end]
        expected = sum(math.prod(shape) for _name, shape in schema_slice)
        if len(q_chunk) != expected:
            raise HnervDecoderRecodeError("HDM5 q chunk length mismatch")
        q_cursor = 0
        for name, shape in schema_slice:
            value_count = math.prod(shape)
            ordered_records.append(
                PackedDecoderRecord(
                    name=name,
                    shape=shape,
                    q_zz_u8=q_chunk[q_cursor : q_cursor + value_count],
                    scale_f32=b"",
                )
            )
            q_cursor += value_count
        split_start = split_end
    scale_len = 4 * len(PACKED_STATE_SCHEMA)
    scale_stream = _read_payload_exact(payload, cursor, scale_len, "HDM5 scale_stream")
    cursor += scale_len
    if cursor != len(payload):
        raise HnervDecoderRecodeError("HDM5 fixture has trailing bytes")

    by_name = {record.name: record for record in ordered_records}
    records = []
    for index, (name, shape) in enumerate(PACKED_STATE_SCHEMA):
        record = by_name.get(name)
        if record is None or record.shape != shape:
            raise HnervDecoderRecodeError(f"HDM5 decoded schema mismatch for {name}")
        records.append(
            PackedDecoderRecord(
                name=name,
                shape=shape,
                q_zz_u8=record.q_zz_u8,
                scale_f32=scale_stream[index * 4 : index * 4 + 4],
            )
        )
    return PackedDecoderRaw(records=tuple(records))


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


def decode_hdm3_q_brotli_split_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode an HDM3 fixed-schema q-Brotli/raw-scale fixture."""

    if payload[:4] != b"HDM3":
        raise HnervDecoderRecodeError("invalid HDM3 q Brotli split fixture magic")
    cursor = 4
    compressed_len = int.from_bytes(
        _read_payload_exact(payload, cursor, 3, "q_brotli_len24"),
        "little",
    )
    cursor += 3
    compressed = _read_payload_exact(payload, cursor, compressed_len, "compressed_hdm3_payload")
    cursor += compressed_len
    scale_len = 4 * len(PACKED_STATE_SCHEMA)
    scale_stream = _read_payload_exact(payload, cursor, scale_len, "scale_stream")
    cursor += scale_len
    if cursor != len(payload):
        raise HnervDecoderRecodeError("HDM3 fixture has trailing bytes")
    try:
        q_stream = brotli.decompress(compressed)
    except brotli.error as exc:
        raise HnervDecoderRecodeError(f"HDM3 brotli decompression failed: {exc}") from exc
    expected_q_len = sum(math.prod(shape) for _name, shape in PACKED_STATE_SCHEMA)
    if len(q_stream) != expected_q_len:
        raise HnervDecoderRecodeError("HDM3 q stream length mismatch")
    records = []
    q_cursor = 0
    for index, (name, shape) in enumerate(PACKED_STATE_SCHEMA):
        value_count = math.prod(shape)
        records.append(
            PackedDecoderRecord(
                name=name,
                shape=shape,
                q_zz_u8=q_stream[q_cursor : q_cursor + value_count],
                scale_f32=scale_stream[index * 4 : index * 4 + 4],
            )
        )
        q_cursor += value_count
    return PackedDecoderRaw(records=tuple(records))


def decode_hdm4_q_brotli_split_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode an HDM4 fixed-recipe q-Brotli/raw-scale fixture."""

    if payload[:4] != b"HDM4":
        raise HnervDecoderRecodeError("invalid HDM4 q Brotli split fixture magic")
    cursor = 4
    recipe_id = _read_payload_exact(payload, cursor, 1, "HDM4 recipe_id")[0]
    cursor += 1
    ordered_schema = _hdm4_ordered_schema(recipe_id)
    chunk_count = len(HDM4_SPLIT_POINTS)
    lengths: list[int] = []
    for index in range(chunk_count):
        lengths.append(
            int.from_bytes(
                _read_payload_exact(payload, cursor, 3, f"HDM4 q_brotli_len24_{index}"),
                "little",
            )
        )
        cursor += 3

    q_chunks: list[bytes] = []
    for index, length in enumerate(lengths):
        compressed = _read_payload_exact(payload, cursor, length, f"HDM4 q_brotli_{index}")
        cursor += length
        try:
            q_chunks.append(brotli.decompress(compressed))
        except brotli.error as exc:
            raise HnervDecoderRecodeError(
                f"HDM4 q stream chunk {index} brotli decode failed: {exc}"
            ) from exc
    scale_len = 4 * len(PACKED_STATE_SCHEMA)
    scale_stream = _read_payload_exact(payload, cursor, scale_len, "scale_stream")
    cursor += scale_len
    if cursor != len(payload):
        raise HnervDecoderRecodeError("HDM4 fixture has trailing bytes")

    ordered_records: list[PackedDecoderRecord] = []
    split_start = 0
    for chunk, split_end in zip(q_chunks, HDM4_SPLIT_POINTS, strict=True):
        schema_slice = ordered_schema[split_start:split_end]
        expected = sum(math.prod(shape) for _name, shape in schema_slice)
        if len(chunk) != expected:
            raise HnervDecoderRecodeError("HDM4 q chunk length mismatch")
        q_cursor = 0
        for name, shape in schema_slice:
            value_count = math.prod(shape)
            ordered_records.append(
                PackedDecoderRecord(
                    name=name,
                    shape=shape,
                    q_zz_u8=chunk[q_cursor : q_cursor + value_count],
                    scale_f32=b"",
                )
            )
            q_cursor += value_count
        split_start = split_end
    if len(ordered_records) != len(PACKED_STATE_SCHEMA):
        raise HnervDecoderRecodeError("HDM4 decoded record count mismatch")
    by_name = {record.name: record for record in ordered_records}
    records = []
    for index, (name, shape) in enumerate(PACKED_STATE_SCHEMA):
        record = by_name.get(name)
        if record is None or record.shape != shape:
            raise HnervDecoderRecodeError(f"HDM4 decoded schema mismatch for {name}")
        records.append(
            PackedDecoderRecord(
                name=name,
                shape=shape,
                q_zz_u8=record.q_zz_u8,
                scale_f32=scale_stream[index * 4 : index * 4 + 4],
            )
        )
    return PackedDecoderRaw(records=tuple(records))


def decode_hdm6_q_brotli_tuned_fixture(payload: bytes) -> PackedDecoderRaw:
    """Decode an HDM6 fixed-recipe tuned q-Brotli/raw-scale fixture."""

    if payload[:4] != b"HDM6":
        raise HnervDecoderRecodeError("invalid HDM6 q Brotli tuned fixture magic")
    cursor = 4
    recipe_id = _read_payload_exact(payload, cursor, 1, "HDM6 recipe_id")[0]
    cursor += 1
    if recipe_id != HDM6_RECIPE_ID:
        raise HnervDecoderRecodeError(f"unsupported HDM6 recipe id: {recipe_id}")
    ordered_schema = _hdm4_ordered_schema(HDM4_RECIPE_ID)
    chunk_count = len(HDM4_SPLIT_POINTS)
    lengths: list[int] = []
    for index in range(chunk_count):
        lengths.append(
            int.from_bytes(
                _read_payload_exact(payload, cursor, 3, f"HDM6 q_brotli_len24_{index}"),
                "little",
            )
        )
        cursor += 3

    q_chunks: list[bytes] = []
    for index, length in enumerate(lengths):
        compressed = _read_payload_exact(payload, cursor, length, f"HDM6 q_brotli_{index}")
        cursor += length
        try:
            q_chunks.append(brotli.decompress(compressed))
        except brotli.error as exc:
            raise HnervDecoderRecodeError(
                f"HDM6 q stream chunk {index} brotli decode failed: {exc}"
            ) from exc
    scale_len = 4 * len(PACKED_STATE_SCHEMA)
    scale_stream = _read_payload_exact(payload, cursor, scale_len, "HDM6 scale_stream")
    cursor += scale_len
    if cursor != len(payload):
        raise HnervDecoderRecodeError("HDM6 fixture has trailing bytes")

    ordered_records: list[PackedDecoderRecord] = []
    split_start = 0
    for chunk, split_end in zip(q_chunks, HDM4_SPLIT_POINTS, strict=True):
        schema_slice = ordered_schema[split_start:split_end]
        expected = sum(math.prod(shape) for _name, shape in schema_slice)
        if len(chunk) != expected:
            raise HnervDecoderRecodeError("HDM6 q chunk length mismatch")
        q_cursor = 0
        for name, shape in schema_slice:
            value_count = math.prod(shape)
            ordered_records.append(
                PackedDecoderRecord(
                    name=name,
                    shape=shape,
                    q_zz_u8=chunk[q_cursor : q_cursor + value_count],
                    scale_f32=b"",
                )
            )
            q_cursor += value_count
        split_start = split_end
    if len(ordered_records) != len(PACKED_STATE_SCHEMA):
        raise HnervDecoderRecodeError("HDM6 decoded record count mismatch")
    by_name = {record.name: record for record in ordered_records}
    records = []
    for index, (name, shape) in enumerate(PACKED_STATE_SCHEMA):
        record = by_name.get(name)
        if record is None or record.shape != shape:
            raise HnervDecoderRecodeError(f"HDM6 decoded schema mismatch for {name}")
        records.append(
            PackedDecoderRecord(
                name=name,
                shape=shape,
                q_zz_u8=record.q_zz_u8,
                scale_f32=scale_stream[index * 4 : index * 4 + 4],
            )
        )
    return PackedDecoderRaw(records=tuple(records))


def _hdm4_ordered_schema(recipe_id: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if recipe_id != HDM4_RECIPE_ID:
        raise HnervDecoderRecodeError(f"unsupported HDM4 recipe id: {recipe_id}")
    return tuple(
        sorted(
            PACKED_STATE_SCHEMA,
            key=lambda item: (
                0
                if len(item[1]) == 4 and item[1][2:] == (3, 3)
                else 1
                if len(item[1]) == 4
                else 2
                if item[0].endswith(".bias")
                else 3,
                -math.prod(item[1]),
                item[0],
            ),
        )
    )


def _resolve_hdm5_workers(workers: int | None) -> int:
    if workers is not None:
        return max(1, int(workers))
    return max(1, min(8, (os.cpu_count() or 1)))


def _ordered_records_from_names(
    parsed: PackedDecoderRaw,
    ordered_record_names: tuple[str, ...],
) -> tuple[PackedDecoderRecord, ...]:
    if len(ordered_record_names) != len(PACKED_STATE_SCHEMA):
        raise HnervDecoderRecodeError("HDM5 ordered names length does not match schema")
    schema_names = [name for name, _shape in PACKED_STATE_SCHEMA]
    if sorted(ordered_record_names) != sorted(schema_names):
        raise HnervDecoderRecodeError("HDM5 ordered names are not a schema permutation")
    record_by_name = {record.name: record for record in parsed.records}
    ordered = []
    for name in ordered_record_names:
        record = record_by_name.get(name)
        if record is None:
            raise HnervDecoderRecodeError(f"HDM5 missing record {name}")
        expected_shape = _shape_for_record_name(name)
        if record.shape != expected_shape:
            raise HnervDecoderRecodeError(f"HDM5 shape mismatch for {name}")
        ordered.append(record)
    return tuple(ordered)


def _validate_split_points(
    split_points: tuple[int, ...],
    *,
    record_count: int,
) -> tuple[int, ...]:
    if not split_points:
        raise HnervDecoderRecodeError("HDM5 split points must not be empty")
    previous = 0
    for point in split_points:
        if point <= previous or point > record_count:
            raise HnervDecoderRecodeError("HDM5 split points must be strictly increasing")
        previous = point
    if split_points[-1] != record_count:
        raise HnervDecoderRecodeError("HDM5 final split point must equal record count")
    return tuple(int(point) for point in split_points)


def _hdm5_order_families(parsed: PackedDecoderRaw) -> tuple[Hdm5OrderFamily, ...]:
    record_by_name = {record.name: record for record in parsed.records}

    def order_from_schema(schema: tuple[tuple[str, tuple[int, ...]], ...]) -> tuple[PackedDecoderRecord, ...]:
        return tuple(record_by_name[name] for name, _shape in schema)

    def entropy(record: PackedDecoderRecord) -> float:
        return float(_symbol_entropy_summary(record.q_zz_u8)["entropy_bits_per_symbol"])

    def mean_symbol(record: PackedDecoderRecord) -> float:
        if not record.q_zz_u8:
            return 0.0
        return float(sum(record.q_zz_u8) / len(record.q_zz_u8))

    candidates: list[tuple[str, str, tuple[PackedDecoderRecord, ...]]] = [
        (
            "packed_size_desc",
            "source packed schema order, largest tensors first",
            tuple(parsed.records),
        ),
        (
            "architecture_forward",
            "model declaration order from FIXED_STATE_SCHEMA",
            order_from_schema(FIXED_STATE_SCHEMA),
        ),
        (
            "hdm4_role_order",
            "current HDM4 role order: 3x3 conv, 1x1 conv, bias tail",
            order_from_schema(_hdm4_ordered_schema(HDM4_RECIPE_ID)),
        ),
        (
            "architecture_reverse",
            "reverse model declaration order tests tail-to-head locality",
            tuple(reversed(order_from_schema(FIXED_STATE_SCHEMA))),
        ),
        (
            "q_entropy_ascending",
            "data-driven grouping from low-entropy tensors to high-entropy tensors",
            tuple(sorted(parsed.records, key=lambda record: (entropy(record), record.name))),
        ),
        (
            "q_entropy_descending",
            "data-driven grouping from high-entropy tensors to low-entropy tensors",
            tuple(sorted(parsed.records, key=lambda record: (-entropy(record), record.name))),
        ),
        (
            "q_mean_symbol_ascending",
            "data-driven grouping by average quantized symbol value",
            tuple(sorted(parsed.records, key=lambda record: (mean_symbol(record), record.name))),
        ),
    ]
    seen: set[tuple[str, ...]] = set()
    families: list[Hdm5OrderFamily] = []
    for name, rationale, records in candidates:
        key = tuple(record.name for record in records)
        if key in seen:
            continue
        seen.add(key)
        families.append(
            Hdm5OrderFamily(
                family_id=len(families),
                name=name,
                records=records,
                rationale=rationale,
            )
        )
    return tuple(families)


def _search_hdm5_family(
    parsed: PackedDecoderRaw,
    family: Hdm5OrderFamily,
    *,
    quality: int,
    max_parts: int,
    workers: int,
    baseline_section_bytes: int,
    hdm4_section_bytes: int,
) -> list[dict[str, Any]]:
    record_count = len(family.records)
    max_parts = min(max_parts, record_count)
    q_stream = b"".join(record.q_zz_u8 for record in family.records)
    offsets = [0]
    cursor = 0
    for record in family.records:
        cursor += len(record.q_zz_u8)
        offsets.append(cursor)

    segment_costs = _hdm5_segment_costs(
        q_stream,
        offsets=tuple(offsets),
        quality=quality,
        workers=workers,
    )
    dp: list[list[tuple[int, tuple[int, ...]] | None]] = [
        [None for _ in range(record_count + 1)] for _ in range(max_parts + 1)
    ]
    for end in range(1, record_count + 1):
        dp[1][end] = (segment_costs[(0, end)], (end,))
    for part_count in range(2, max_parts + 1):
        for end in range(part_count, record_count + 1):
            best: tuple[int, tuple[int, ...]] | None = None
            for previous in range(part_count - 1, end):
                previous_row = dp[part_count - 1][previous]
                if previous_row is None:
                    continue
                candidate_cost = previous_row[0] + segment_costs[(previous, end)]
                candidate_splits = previous_row[1] + (end,)
                if best is None or (candidate_cost, candidate_splits) < best:
                    best = (candidate_cost, candidate_splits)
            dp[part_count][end] = best

    rows: list[dict[str, Any]] = []
    for part_count in range(1, max_parts + 1):
        best = dp[part_count][record_count]
        if best is None:
            continue
        q_brotli_bytes, split_points = best
        payload, stats = encode_hdm5_q_brotli_split_planning_fixture(
            parsed,
            ordered_record_names=tuple(record.name for record in family.records),
            split_points=split_points,
            quality=quality,
        )
        if len(payload) != stats["header_bytes"] + q_brotli_bytes + len(parsed.scale_stream):
            raise HnervDecoderRecodeError("HDM5 search byte accounting mismatch")
        fixed_recipe_header_bytes = 5 + stats["length_prefix_bytes"]
        fixed_recipe_projected_bytes = (
            q_brotli_bytes + stats["raw_scale_bytes"] + fixed_recipe_header_bytes
        )
        rows.append(
            {
                "variant": "hdm5_q_brotli_split_search_self_describing",
                "codec": "HDM5_planning_self_describing_order_split_q_brotli_raw_scales",
                "score_claim": False,
                "archive_ready": False,
                "ready_for_exact_eval_dispatch": False,
                "raw_equal": decode_hdm5_q_brotli_split_planning_fixture(payload).to_raw()
                == parsed.to_raw(),
                "family_id": family.family_id,
                "family_name": family.name,
                "family_rationale": family.rationale,
                "part_count": part_count,
                "split_points": list(split_points),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "byte_delta_vs_baseline_section": len(payload) - baseline_section_bytes,
                "byte_delta_vs_hdm4_section": len(payload) - hdm4_section_bytes,
                "q_brotli_bytes": q_brotli_bytes,
                "header_bytes": stats["header_bytes"],
                "fixed_recipe_header_bytes": fixed_recipe_header_bytes,
                "fixed_recipe_projected_bytes": fixed_recipe_projected_bytes,
                "fixed_recipe_projected_byte_delta_vs_hdm4_section": (
                    fixed_recipe_projected_bytes - hdm4_section_bytes
                ),
                "fixed_recipe_projected_byte_delta_vs_baseline_section": (
                    fixed_recipe_projected_bytes - baseline_section_bytes
                ),
                "fixed_recipe_projection_contract": (
                    "planning_only_assumes_runtime_hardcodes_order_and_split_recipe"
                ),
                "record_order_metadata_bytes": stats["record_order_metadata_bytes"],
                "split_metadata_bytes": stats["split_metadata_bytes"],
                "length_prefix_bytes": stats["length_prefix_bytes"],
                "raw_scale_bytes": stats["raw_scale_bytes"],
                "q_chunk_bytes": stats["q_chunk_bytes"],
                "ordered_record_names": stats["ordered_record_names"],
                "planning_only": True,
            }
        )
    return rows


def _hdm5_segment_costs(
    q_stream: bytes,
    *,
    offsets: tuple[int, ...],
    quality: int,
    workers: int,
) -> dict[tuple[int, int], int]:
    record_count = len(offsets) - 1
    segments = [(start, end) for start in range(record_count) for end in range(start + 1, record_count + 1)]

    def measure(segment: tuple[int, int]) -> tuple[tuple[int, int], int]:
        start, end = segment
        payload = q_stream[offsets[start] : offsets[end]]
        return segment, len(brotli.compress(payload, quality=quality))

    if workers <= 1:
        return dict(measure(segment) for segment in segments)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(measure, segments))


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
    "decode_hdm3_q_brotli_split_fixture",
    "decode_hdm4_q_brotli_split_fixture",
    "decode_hdm5_q_brotli_split_planning_fixture",
    "decode_hdm6_q_brotli_tuned_fixture",
    "decode_prev_symbol_context_range_fixture",
    "encode_global_prev_symbol_context_range_fixture",
    "encode_global_prev_symbol_mixed_context_fixture",
    "encode_hdm3_q_brotli_split_fixture",
    "encode_hdm4_q_brotli_split_fixture",
    "encode_hdm5_q_brotli_split_planning_fixture",
    "encode_hdm6_q_brotli_tuned_fixture",
    "encode_prev_symbol_context_range_fixture",
    "parse_decoder_section_for_recode",
    "parse_packed_decoder_brotli",
    "search_hdm5_q_brotli_split_recipes",
]
