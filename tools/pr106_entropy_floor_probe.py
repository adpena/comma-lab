#!/usr/bin/env python3
"""Entropy-floor probe for PR106/frontier HNeRV payloads.

This is a planning and adversarial-validation tool. It reads a PR106-style
HNeRV archive or payload, extracts the real decoder and latent byte streams
that the inflate path consumes, and reports oracle IID/Markov/transform floors
with SHA custody. It does not emit a candidate archive and does not claim a
score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import brotli
import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_decoder_recode import (  # noqa: E402
    PACKED_STATE_SCHEMA,
    decode_hdm3_q_brotli_split_fixture,
    decode_hdm4_q_brotli_split_fixture,
    parse_packed_decoder_brotli,
)
from tac.hnerv_lowlevel_packer import (  # noqa: E402
    HnervLowlevelPackError,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
)
from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_SIDECAR_MAGIC,
    parse_pr106_sidecar_packet,
)
from tac.packet_compiler.pr106_fixed_latent_recode import (  # noqa: E402
    HLM1_MAGIC,
    HLM2_MAGIC,
    decode_pr106_fixed_latent_raw,
)

TOOL = "tools/pr106_entropy_floor_probe.py"
SCHEMA_VERSION = 1
PR106_LATENT_N = 600
PR106_LATENT_D = 28
PR106_FIXED_LATENT_RAW_BYTES = PR106_LATENT_N * PR106_LATENT_D * 2 + PR106_LATENT_D * 4


@dataclass(frozen=True)
class SymbolStream:
    """A finite categorical symbol stream used by the floor estimators."""

    name: str
    symbols: np.ndarray
    n_categories: int
    semantic: str

    def __post_init__(self) -> None:
        arr = np.asarray(self.symbols, dtype=np.int64).reshape(-1)
        if self.n_categories <= 0:
            raise ValueError(f"{self.name}: n_categories must be positive")
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= self.n_categories):
            raise ValueError(
                f"{self.name}: symbol outside [0, {self.n_categories}) "
                f"(min={int(arr.min())}, max={int(arr.max())})"
            )
        object.__setattr__(self, "symbols", arr)


def entropy_bits_from_counts(counts: Counter[int] | np.ndarray) -> float:
    """Return Shannon entropy in bits for a count table."""

    if isinstance(counts, Counter):
        total = sum(counts.values())
        values = counts.values()
    else:
        arr = np.asarray(counts, dtype=np.float64)
        total = float(arr.sum())
        values = arr[arr > 0]
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in values:
        if count <= 0:
            continue
        p = float(count) / float(total)
        entropy -= p * math.log2(p)
    return float(entropy) if entropy > 0.0 else 0.0


def iid_bits(streams: list[SymbolStream]) -> float:
    bits = 0.0
    for stream in streams:
        counts = np.bincount(stream.symbols, minlength=stream.n_categories)
        bits += stream.symbols.size * entropy_bits_from_counts(counts)
    return bits


def markov1_bits(streams: list[SymbolStream]) -> float:
    """Oracle order-1 empirical code length in bits.

    The first symbol in each stream is costed with the stream's marginal
    distribution; all later symbols use empirical ``p(x_t | x_{t-1})``.
    """

    bits = 0.0
    for stream in streams:
        syms = [int(value) for value in stream.symbols]
        if not syms:
            continue
        bits += entropy_bits_from_counts(Counter(syms))
        if len(syms) == 1:
            continue
        context_counts: dict[int, Counter[int]] = {}
        for previous, current in pairwise(syms):
            context_counts.setdefault(previous, Counter())[current] += 1
        for conditional in context_counts.values():
            bits += sum(conditional.values()) * entropy_bits_from_counts(conditional)
    return float(bits)


def markov2_bits(streams: list[SymbolStream]) -> float:
    """Oracle order-2 empirical code length in bits."""

    bits = 0.0
    for stream in streams:
        syms = [int(value) for value in stream.symbols]
        if not syms:
            continue
        bits += entropy_bits_from_counts(Counter(syms))
        if len(syms) == 1:
            continue
        pair_counts = Counter(pairwise(syms))
        prev_counts = Counter(syms[:-1])
        first_context = syms[0]
        second_symbol = syms[1]
        p_second = pair_counts[(first_context, second_symbol)] / prev_counts[first_context]
        bits += -math.log2(p_second)
        if len(syms) == 2:
            continue
        context_counts: dict[tuple[int, int], Counter[int]] = {}
        for index in range(len(syms) - 2):
            context = (syms[index], syms[index + 1])
            context_counts.setdefault(context, Counter())[syms[index + 2]] += 1
        for conditional in context_counts.values():
            bits += sum(conditional.values()) * entropy_bits_from_counts(conditional)
    return float(bits)


def ceil_bits_to_bytes(bits: float) -> int:
    if bits <= 0.0:
        return 0
    return math.ceil(bits / 8.0)


def source_stream_manifest(streams: list[SymbolStream]) -> list[dict[str, Any]]:
    return [
        {
            "name": stream.name,
            "semantic": stream.semantic,
            "symbols": int(stream.symbols.size),
            "n_categories": int(stream.n_categories),
            "sha256_u8_symbols": _u8_symbol_sha256(stream),
            "unique_symbols": len({int(value) for value in stream.symbols}),
        }
        for stream in streams
    ]


def _u8_symbol_sha256(stream: SymbolStream) -> str | None:
    if stream.n_categories > 256:
        return None
    return sha256_bytes(stream.symbols.astype(np.uint8).tobytes())


def transform_identity(streams: list[SymbolStream]) -> list[SymbolStream]:
    return streams


def transform_delta_mod(streams: list[SymbolStream]) -> list[SymbolStream]:
    out: list[SymbolStream] = []
    for stream in streams:
        syms = stream.symbols
        if syms.size == 0:
            transformed = syms.copy()
        else:
            transformed = np.empty_like(syms)
            transformed[0] = syms[0]
            transformed[1:] = (syms[1:] - syms[:-1]) % stream.n_categories
        out.append(
            SymbolStream(
                f"{stream.name}:delta_mod{stream.n_categories}",
                transformed,
                stream.n_categories,
                f"first_symbol_then_mod_{stream.n_categories}_delta",
            )
        )
    return out


def transform_xor_prev(streams: list[SymbolStream]) -> list[SymbolStream]:
    out: list[SymbolStream] = []
    for stream in streams:
        if stream.n_categories != 256:
            continue
        syms = stream.symbols.astype(np.uint8)
        if syms.size == 0:
            transformed = syms.astype(np.int64)
        else:
            transformed_u8 = np.empty_like(syms)
            transformed_u8[0] = syms[0]
            transformed_u8[1:] = np.bitwise_xor(syms[1:], syms[:-1])
            transformed = transformed_u8.astype(np.int64)
        out.append(SymbolStream(f"{stream.name}:xor_prev", transformed, 256, "xor_previous_byte"))
    return out


def transform_nibbles(streams: list[SymbolStream]) -> list[SymbolStream]:
    out: list[SymbolStream] = []
    for stream in streams:
        if stream.n_categories != 256:
            continue
        syms = stream.symbols
        out.append(SymbolStream(f"{stream.name}:hi4", syms >> 4, 16, "high_nibble"))
        out.append(SymbolStream(f"{stream.name}:lo4", syms & 15, 16, "low_nibble"))
    return out


def transform_bitplanes(streams: list[SymbolStream]) -> list[SymbolStream]:
    out: list[SymbolStream] = []
    for stream in streams:
        if stream.n_categories != 256:
            continue
        syms = stream.symbols
        for bit in range(8):
            out.append(
                SymbolStream(
                    f"{stream.name}:bit{bit}",
                    (syms >> bit) & 1,
                    2,
                    "bitplane",
                )
            )
    return out


def transform_zero0_nonzero_value(streams: list[SymbolStream]) -> list[SymbolStream]:
    out: list[SymbolStream] = []
    for stream in streams:
        syms = stream.symbols
        out.append(SymbolStream(f"{stream.name}:is_zero0", (syms == 0).astype(np.int64), 2, "zero0_mask"))
        nonzero = syms[syms != 0] - 1
        out.append(
            SymbolStream(
                f"{stream.name}:nonzero_value_minus1",
                nonzero,
                max(1, stream.n_categories - 1),
                "nonzero_value_minus1",
            )
        )
    return out


TRANSFORMS = {
    "identity": transform_identity,
    "delta_mod": transform_delta_mod,
    "xor_prev_byte": transform_xor_prev,
    "nibble_split": transform_nibbles,
    "bitplanes": transform_bitplanes,
    "zero0_mask_nonzero_value": transform_zero0_nonzero_value,
}


def floor_rows_for_group(
    streams: list[SymbolStream],
    *,
    current_storage_bytes: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transform_name, transform in TRANSFORMS.items():
        transformed = transform(streams)
        if not transformed:
            continue
        bits_iid = iid_bits(transformed)
        bits_m1 = markov1_bits(transformed)
        bits_m2 = markov2_bits(transformed)
        row: dict[str, Any] = {
            "transform": transform_name,
            "invertible_fixed_transform": True,
            "metadata_bytes_charged": 0,
            "stream_count": len(transformed),
            "symbol_count": int(sum(stream.symbols.size for stream in transformed)),
            "model_complexity": context_model_complexity(transformed),
            "iid_floor_bytes": ceil_bits_to_bytes(bits_iid),
            "markov1_floor_bytes": ceil_bits_to_bytes(bits_m1),
            "markov2_floor_bytes": ceil_bits_to_bytes(bits_m2),
            "iid_bits_per_symbol": _bits_per_symbol(bits_iid, transformed),
            "markov1_bits_per_symbol": _bits_per_symbol(bits_m1, transformed),
            "markov2_bits_per_symbol": _bits_per_symbol(bits_m2, transformed),
        }
        if current_storage_bytes is not None:
            row["iid_floor_delta_vs_current_storage_bytes"] = (
                row["iid_floor_bytes"] - current_storage_bytes
            )
            row["markov1_floor_delta_vs_current_storage_bytes"] = (
                row["markov1_floor_bytes"] - current_storage_bytes
            )
            row["markov2_floor_delta_vs_current_storage_bytes"] = (
                row["markov2_floor_bytes"] - current_storage_bytes
            )
        rows.append(row)
    rows.sort(key=lambda item: (int(item["markov2_floor_bytes"]), str(item["transform"])))
    return rows


def context_model_complexity(streams: list[SymbolStream]) -> dict[str, int]:
    """Count empirical contexts/edges whose coding cost is not charged."""

    markov1_contexts = 0
    markov1_edges = 0
    markov2_contexts = 0
    markov2_edges = 0
    for stream in streams:
        syms = [int(value) for value in stream.symbols]
        if len(syms) >= 2:
            pairs = set(pairwise(syms))
            markov1_edges += len(pairs)
            markov1_contexts += len({previous for previous, _current in pairs})
        if len(syms) >= 3:
            triples = {
                (syms[index], syms[index + 1], syms[index + 2])
                for index in range(len(syms) - 2)
            }
            markov2_edges += len(triples)
            markov2_contexts += len({(a, b) for a, b, _c in triples})
    return {
        "markov1_contexts_unpriced": int(markov1_contexts),
        "markov1_edges_unpriced": int(markov1_edges),
        "markov2_contexts_unpriced": int(markov2_contexts),
        "markov2_edges_unpriced": int(markov2_edges),
    }


def _bits_per_symbol(bits: float, streams: list[SymbolStream]) -> float:
    symbols = sum(stream.symbols.size for stream in streams)
    if symbols == 0:
        return 0.0
    return round(float(bits / symbols), 12)


def build_group(
    name: str,
    streams: list[SymbolStream],
    *,
    current_storage_bytes: int | None,
    current_storage_label: str | None,
) -> dict[str, Any]:
    rows = floor_rows_for_group(streams, current_storage_bytes=current_storage_bytes)
    best = rows[0] if rows else None
    return {
        "group": name,
        "current_storage_bytes": current_storage_bytes,
        "current_storage_label": current_storage_label,
        "source_stream_count": len(streams),
        "source_symbol_count": int(sum(stream.symbols.size for stream in streams)),
        "source_streams": source_stream_manifest(streams),
        "floors": rows,
        "best_markov2_transform": best["transform"] if best else None,
        "best_markov2_floor_bytes": best["markov2_floor_bytes"] if best else None,
        "best_markov2_delta_vs_current_storage_bytes": (
            best.get("markov2_floor_delta_vs_current_storage_bytes") if best else None
        ),
        "floor_limitations": [
            "oracle_entropy_floor_only",
            "model_table_and_transform_metadata_not_charged",
            "no_actual_coder_bitstream_emitted",
            "no_runtime_decoder_implementation",
            "no_archive_manifest_or_exact_cuda_auth_eval",
        ],
    }


def build_report_from_archive(
    archive: Path,
    *,
    pr101_reference_archive_bytes: int | None = None,
    active_floor_archive_bytes: int | None = None,
    active_floor_label: str | None = None,
) -> dict[str, Any]:
    source = read_strict_single_member_zip(archive)
    return build_report_from_payload(
        source.payload,
        source={
            "mode": "archive",
            "path": str(archive),
            "archive_bytes": source.archive_bytes,
            "archive_sha256": source.archive_sha256,
            "member_name": source.member_name,
            "member_bytes": source.member_bytes,
            "member_sha256": sha256_bytes(source.payload),
            "zip_overhead_bytes": source.archive_bytes - source.member_bytes,
        },
        pr101_reference_archive_bytes=pr101_reference_archive_bytes,
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_label=active_floor_label,
    )


def build_report_from_payload(
    payload: bytes,
    *,
    source: dict[str, Any],
    pr101_reference_archive_bytes: int | None = None,
    active_floor_archive_bytes: int | None = None,
    active_floor_label: str | None = None,
) -> dict[str, Any]:
    payload, source = _unwrap_pr106_sidecar_if_present(payload, source)
    try:
        packed = parse_ff_packed_brotli_hnerv(payload)
    except HnervLowlevelPackError as exc:
        raise ValueError(f"payload is not PR106 ff-packed HNeRV: {exc}") from exc
    parsed_decoder, decoder_raw, decoder_section_codec = _decode_decoder_section(
        packed.decoder_packed_brotli
    )
    latents_raw = decode_pr106_fixed_latent_raw(packed.latents_and_sidecar_brotli)
    latents_section_codec = (
        "hlm1_sparse_hi_delta_positions"
        if packed.latents_and_sidecar_brotli.startswith(HLM1_MAGIC)
        else (
            "hlm2_sparse_hi_delta_positions"
            if packed.latents_and_sidecar_brotli.startswith(HLM2_MAGIC)
            else "brotli_fixed_latents_raw"
        )
    )

    decoder_streams = [
        SymbolStream(
            f"decoder_q_zz:{record.name}",
            np.frombuffer(record.q_zz_u8, dtype=np.uint8),
            256,
            "packed_decoder_q_zz_u8",
        )
        for record in parsed_decoder.records
    ]
    decoder_streams.append(
        SymbolStream(
            "decoder_scale_f32_bytes",
            np.frombuffer(parsed_decoder.scale_stream, dtype=np.uint8),
            256,
            "packed_decoder_f32_scales",
        )
    )
    latent_streams = fixed_latent_raw_streams(latents_raw)

    groups = [
        build_group(
            "decoder_q_zz_plus_f32_scales",
            decoder_streams,
            current_storage_bytes=len(packed.decoder_packed_brotli),
            current_storage_label="decoder_section_encoded",
        ),
        build_group(
            "fixed_latents_delta_zz_plus_fp16_meta",
            latent_streams,
            current_storage_bytes=len(packed.latents_and_sidecar_brotli),
            current_storage_label="latents_and_sidecar_brotli",
        ),
        build_group(
            "decoded_payload_sections_without_ff_header",
            decoder_streams + latent_streams,
            current_storage_bytes=len(packed.decoder_packed_brotli)
            + len(packed.latents_and_sidecar_brotli),
            current_storage_label="decoder_packed_brotli+latents_and_sidecar_brotli",
        ),
    ]
    source = {
        **source,
        "payload_magic": "ff_packed_hnerv",
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "header_bytes": len(packed.header),
        "header_sha256": sha256_bytes(packed.header),
        "decoder_section_bytes": len(packed.decoder_packed_brotli),
        "decoder_section_sha256": sha256_bytes(packed.decoder_packed_brotli),
        "decoder_section_codec": decoder_section_codec,
        "decoder_raw_bytes": len(decoder_raw),
        "decoder_raw_sha256": sha256_bytes(decoder_raw),
        "latents_section_bytes": len(packed.latents_and_sidecar_brotli),
        "latents_section_sha256": sha256_bytes(packed.latents_and_sidecar_brotli),
        "latents_section_codec": latents_section_codec,
        "latents_raw_bytes": len(latents_raw),
        "latents_raw_sha256": sha256_bytes(latents_raw),
    }
    return build_report(
        source=source,
        groups=groups,
        pr101_reference_archive_bytes=pr101_reference_archive_bytes,
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_label=active_floor_label,
    )


def _decode_decoder_section(decoder_section: bytes) -> tuple[Any, bytes, str]:
    """Decode current frontier decoder-section variants into raw records."""

    if decoder_section.startswith(b"HDM3"):
        parsed = decode_hdm3_q_brotli_split_fixture(decoder_section)
        return parsed, parsed.to_raw(), "hdm3_q_brotli_split"
    if decoder_section.startswith(b"HDM4"):
        parsed = decode_hdm4_q_brotli_split_fixture(decoder_section)
        return parsed, parsed.to_raw(), "hdm4_q_brotli_split"
    parsed = parse_packed_decoder_brotli(decoder_section)
    return parsed, parsed.to_raw(), "brotli_packed_decoder_raw"


def _unwrap_pr106_sidecar_if_present(
    payload: bytes,
    source: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Preserve PR106 wrapper custody while modeling inner HNeRV streams."""

    if not payload or payload[0] != PR106_SIDECAR_MAGIC:
        return payload, source

    packet = parse_pr106_sidecar_packet(payload)
    framing_meta = packet.framing_meta or b""
    return packet.pr106_bytes, {
        **source,
        "outer_payload_magic": "pr106_sidecar_wrapper",
        "outer_payload_bytes": len(payload),
        "outer_payload_sha256": sha256_bytes(payload),
        "sidecar_format_id": packet.format_id,
        "sidecar_kind": packet.sidecar_kind,
        "sidecar_payload_bytes": len(packet.sidecar_payload),
        "sidecar_payload_sha256": sha256_bytes(packet.sidecar_payload),
        "framing_meta_bytes": len(framing_meta),
        "framing_meta_sha256": sha256_bytes(framing_meta) if framing_meta else None,
        "pr106_inner_payload_bytes": len(packet.pr106_bytes),
        "pr106_inner_payload_sha256": sha256_bytes(packet.pr106_bytes),
        "wrapper_unwrapped_for_entropy_model": True,
    }


def build_report_from_state_dict(
    state_dict_path: Path,
    *,
    pr101_reference_archive_bytes: int | None = None,
    active_floor_archive_bytes: int | None = None,
    active_floor_label: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ValueError("torch is required for --state-dict input") from exc

    blob = state_dict_path.read_bytes()
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"{state_dict_path} did not load as a state_dict dict")
    decoder_streams = quantized_state_dict_streams(state_dict)
    groups = [
        build_group(
            "state_dict_requantized_decoder_q_zz_plus_f32_scales",
            decoder_streams,
            current_storage_bytes=None,
            current_storage_label=None,
        )
    ]
    return build_report(
        source={
            "mode": "state_dict_requantized_proxy",
            "path": str(state_dict_path),
            "input_bytes": len(blob),
            "input_sha256": hashlib.sha256(blob).hexdigest(),
            "state_dict_tensor_count": len(state_dict),
            "schema_label": "FIXED_STATE_SCHEMA_requantized_not_charged_payload",
        },
        groups=groups,
        pr101_reference_archive_bytes=pr101_reference_archive_bytes,
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_label=active_floor_label,
    )


def build_report(
    *,
    source: dict[str, Any],
    groups: list[dict[str, Any]],
    pr101_reference_archive_bytes: int | None,
    active_floor_archive_bytes: int | None,
    active_floor_label: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "created_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "empirical_planning",
        "source": source,
        "groups": groups,
        "adversarial_claim_check": adversarial_claim_check(
            source,
            groups,
            pr101_reference_archive_bytes=pr101_reference_archive_bytes,
            active_floor_archive_bytes=active_floor_archive_bytes,
            active_floor_label=active_floor_label,
        ),
        "dispatch_blockers": [
            "entropy_floor_probe_only",
            "metadata_and_model_tables_not_charged_in_floor_rows",
            "no_candidate_archive_emitted",
            "no_runtime_decoder_adapter",
            "missing_exact_cuda_auth_eval_for_any_floor_as_codec",
        ],
    }


def fixed_latent_raw_streams(raw: bytes) -> list[SymbolStream]:
    if len(raw) != PR106_FIXED_LATENT_RAW_BYTES:
        raise ValueError(
            f"bad PR106 fixed latent raw length: {len(raw)} "
            f"expected {PR106_FIXED_LATENT_RAW_BYTES}"
        )
    total = PR106_LATENT_N * PR106_LATENT_D
    mins_start = total
    scales_start = mins_start + PR106_LATENT_D * 2
    hi_start = total + PR106_LATENT_D * 4
    return [
        SymbolStream(
            "latents_delta_zz_lo_byte",
            np.frombuffer(raw[:total], dtype=np.uint8),
            256,
            "fixed_latent_delta_zigzag_low_byte",
        ),
        SymbolStream(
            "latents_min_fp16_bytes",
            np.frombuffer(raw[mins_start:scales_start], dtype=np.uint8),
            256,
            "fixed_latent_min_fp16_meta",
        ),
        SymbolStream(
            "latents_scale_fp16_bytes",
            np.frombuffer(raw[scales_start:hi_start], dtype=np.uint8),
            256,
            "fixed_latent_scale_fp16_meta",
        ),
        SymbolStream(
            "latents_delta_zz_hi_byte",
            np.frombuffer(raw[hi_start : hi_start + total], dtype=np.uint8),
            256,
            "fixed_latent_delta_zigzag_high_byte",
        ),
    ]


def quantized_state_dict_streams(state_dict: dict[str, Any]) -> list[SymbolStream]:
    streams: list[SymbolStream] = []
    scale_bytes = bytearray()
    for name, shape in PACKED_STATE_SCHEMA:
        if name not in state_dict:
            raise ValueError(f"state_dict missing tensor {name!r}")
        tensor = state_dict[name].detach().cpu().float()
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"state_dict tensor {name!r} has shape {tuple(tensor.shape)}, expected {shape}"
            )
        max_abs = float(tensor.abs().max().item())
        scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
        q = (tensor / scale).round().clamp(-127, 127).numpy().astype(np.int16)
        q_zz = np.where(q >= 0, 2 * q, -2 * q - 1).astype(np.uint8).reshape(-1)
        streams.append(SymbolStream(f"state_dict_q_zz:{name}", q_zz, 256, "requantized_q_zz_u8"))
        scale_bytes.extend(struct.pack("<f", scale))
    streams.append(
        SymbolStream(
            "state_dict_requantized_scale_f32_bytes",
            np.frombuffer(bytes(scale_bytes), dtype=np.uint8),
            256,
            "requantized_f32_scales",
        )
    )
    return streams


def adversarial_claim_check(
    source: dict[str, Any],
    groups: list[dict[str, Any]],
    *,
    pr101_reference_archive_bytes: int | None,
    active_floor_archive_bytes: int | None,
    active_floor_label: str | None,
) -> dict[str, Any]:
    archive_bytes = source.get("archive_bytes")
    payload_bytes = source.get("payload_bytes")
    decoded_payload = next(
        (
            group
            for group in groups
            if group["group"] == "decoded_payload_sections_without_ff_header"
        ),
        None,
    )
    best_floor = None
    if decoded_payload is not None:
        best_floor = decoded_payload.get("best_markov2_floor_bytes")

    evidence: list[str] = []
    if pr101_reference_archive_bytes is not None and archive_bytes is not None:
        evidence.append(
            "current_source_archive_exceeds_pr101_reference_by_"
            f"{int(archive_bytes) - int(pr101_reference_archive_bytes)}_bytes"
        )
    if pr101_reference_archive_bytes is not None and active_floor_archive_bytes is not None:
        evidence.append(
            "active_pr106_rate_floor_exceeds_pr101_reference_by_"
            f"{int(active_floor_archive_bytes) - int(pr101_reference_archive_bytes)}_bytes"
        )
    if best_floor is not None and payload_bytes is not None:
        evidence.append(
            "best_oracle_markov2_payload_floor_delta_vs_current_payload_"
            f"{int(best_floor) - int(payload_bytes)}_bytes_before_model_overhead"
        )

    verdict = "insufficient_cross_archive_reference"
    if pr101_reference_archive_bytes is not None and archive_bytes is not None:
        verdict = "pr101_only_not_transferable_to_pr106_without_pr106_specific_codec_and_exact_eval"
    if source.get("mode") == "state_dict_requantized_proxy":
        verdict = "state_dict_proxy_cannot_validate_transfer_to_charged_pr106_payload"

    return {
        "claim": "encoder_side_bounded_at_about_178kb_without_ml",
        "verdict": verdict,
        "pr101_reference_archive_bytes": pr101_reference_archive_bytes,
        "active_floor_archive_bytes": active_floor_archive_bytes,
        "active_floor_label": active_floor_label,
        "source_archive_bytes": archive_bytes,
        "source_payload_bytes": payload_bytes,
        "best_decoded_payload_markov2_floor_bytes_before_overhead": best_floor,
        "evidence": evidence,
        "interpretation": (
            "The floor rows are oracle model-class bounds, not charged archive "
            "candidates. Transfer from PR101 to PR106 requires a PR106-specific "
            "bitstream/runtime that pays model tables, metadata, packet overhead, "
            "and exact CUDA replay on the resulting archive."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PR106 Entropy Floor Probe",
        "",
        f"- score_claim: `{str(report['score_claim']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(report['ready_for_exact_eval_dispatch']).lower()}`",
        f"- source_mode: `{report['source'].get('mode')}`",
    ]
    if report["source"].get("path"):
        lines.append(f"- source_path: `{report['source']['path']}`")
    for key in (
        "archive_bytes",
        "payload_bytes",
        "decoder_section_bytes",
        "latents_section_bytes",
    ):
        if report["source"].get(key) is not None:
            lines.append(f"- {key}: `{report['source'][key]}`")
    lines.extend(["", "## Group Floors", ""])
    lines.append(
        "| group | current bytes | best transform | best markov2 floor | delta vs current |"
    )
    lines.append("|---|---:|---|---:|---:|")
    for group in report["groups"]:
        delta = group.get("best_markov2_delta_vs_current_storage_bytes")
        lines.append(
            f"| `{group['group']}` | {group.get('current_storage_bytes')} | "
            f"`{group.get('best_markov2_transform')}` | "
            f"{group.get('best_markov2_floor_bytes')} | {delta} |"
        )
    claim = report["adversarial_claim_check"]
    lines.extend(
        [
            "",
            "## Adversarial Claim Check",
            "",
            f"- claim: `{claim['claim']}`",
            f"- verdict: `{claim['verdict']}`",
            f"- pr101_reference_archive_bytes: `{claim.get('pr101_reference_archive_bytes')}`",
            f"- active_floor_archive_bytes: `{claim.get('active_floor_archive_bytes')}`",
            f"- best_decoded_payload_markov2_floor_bytes_before_overhead: "
            f"`{claim.get('best_decoded_payload_markov2_floor_bytes_before_overhead')}`",
            "",
            claim["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Single-member PR106/frontier archive.zip")
    source.add_argument("--payload-bin", type=Path, help="Raw ff-packed HNeRV payload")
    source.add_argument("--state-dict", type=Path, help="Decoded state_dict.pt proxy input")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--pr101-reference-archive-bytes", type=int)
    parser.add_argument("--active-floor-archive-bytes", type=int)
    parser.add_argument("--active-floor-label")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.archive:
        report = build_report_from_archive(
            args.archive,
            pr101_reference_archive_bytes=args.pr101_reference_archive_bytes,
            active_floor_archive_bytes=args.active_floor_archive_bytes,
            active_floor_label=args.active_floor_label,
        )
    elif args.payload_bin:
        payload = args.payload_bin.read_bytes()
        report = build_report_from_payload(
            payload,
            source={
                "mode": "payload_bin",
                "path": str(args.payload_bin),
                "input_bytes": len(payload),
                "input_sha256": sha256_bytes(payload),
            },
            pr101_reference_archive_bytes=args.pr101_reference_archive_bytes,
            active_floor_archive_bytes=args.active_floor_archive_bytes,
            active_floor_label=args.active_floor_label,
        )
    else:
        report = build_report_from_state_dict(
            args.state_dict,
            pr101_reference_archive_bytes=args.pr101_reference_archive_bytes,
            active_floor_archive_bytes=args.active_floor_archive_bytes,
            active_floor_label=args.active_floor_label,
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(report), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.json_out}")
    if args.md_out:
        print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
