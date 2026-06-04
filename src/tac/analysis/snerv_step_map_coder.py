# SPDX-License-Identifier: MIT
"""Compact receiver-visible SNeRV L-infinity step-map coder.

SNeRV's current blocker is that per-coefficient L-infinity step maps are real
receiver state. If they are stored as fp32 and LZMA-compressed, they can eat the
rate win. This module provides a deterministic, self-contained packet grammar
for those maps:

* shared log-domain quantizer across every map in the packet;
* uint8 codes compressed with LZMA;
* header carries shapes and quantizer parameters;
* decode reconstructs positive step maps without scorer access.

This is a packet coder, not a score claim. It is meant to replace the current
conservative fp32 step-map accounting once wired into the SNeRV advisory/runtime.
"""

from __future__ import annotations

import json
import lzma
import math
import struct
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SCHEMA = "snerv_step_map_coder.v1"
ADAPTIVE_SCHEMA = "snerv_step_map_coder.adaptive.v1"
MAGIC = b"SNSM1"
ADAPTIVE_MAGIC = b"SNSA1"
HEADER_LEN_FMT = "<I"
COMPACT_CONSTANT_SHARED_SHAPE_MIN_MAPS = 8


class SnervStepMapCoderError(ValueError):
    """Raised when step-map packet invariants fail."""


@dataclass(frozen=True)
class StepMapPacket:
    """Encoded step-map packet with enough metadata for receiver decode."""

    packet: bytes
    schema: str
    map_count: int
    bins: int
    bits_per_code: int
    code_storage: str
    code_count: int
    packed_code_bytes: int
    payload_bytes: int
    header_bytes: int
    total_bytes: int
    per_map_packet_baseline_bytes: int
    bundle_savings_bytes: int
    bundle_savings_ratio: float
    unique_code_count: int
    code_entropy_bits_per_symbol: float
    code_entropy_ideal_bytes: int
    fp32_lzma_baseline_bytes: int
    max_relative_error: float
    mean_relative_error: float
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["packet"] = {
            "bytes": len(self.packet),
            "sha256_uncomputed": True,
        }
        return d


@dataclass(frozen=True)
class AdaptiveStepMapPacket:
    """Receiver-visible packet with per-map precision groups."""

    packet: bytes
    schema: str
    map_count: int
    groups: tuple[dict[str, Any], ...]
    payload_bytes: int
    header_bytes: int
    total_bytes: int
    fp32_lzma_baseline_bytes: int
    max_relative_error: float
    mean_relative_error: float
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["packet"] = {
            "bytes": len(self.packet),
            "sha256_uncomputed": True,
        }
        return d


def encode_step_maps(
    step_maps: list[np.ndarray],
    *,
    bins: int = 128,
) -> StepMapPacket:
    """Encode positive step maps into a compact receiver-visible packet."""

    arrays = [_validate_step_map(a) for a in step_maps]
    if not arrays:
        raise SnervStepMapCoderError("at least one step map is required")
    if bins < 2 or bins > 256:
        raise SnervStepMapCoderError("bins must be in [2, 256]")
    flat = np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)
    logs = np.log2(flat)
    log_min = float(logs.min())
    log_max = float(logs.max())
    if log_max <= log_min:
        log_step = 1.0
        codes_flat = np.zeros_like(logs, dtype=np.uint8)
    else:
        log_step = (log_max - log_min) / float(bins - 1)
        q = np.rint((logs - log_min) / log_step).clip(0, bins - 1)
        codes_flat = q.astype(np.uint8)

    bits_per_code = _bits_per_code(bins)
    packed_codes, code_storage = _pack_codes(codes_flat, bits_per_code)
    compressed_codes = lzma.compress(
        packed_codes,
        format=lzma.FORMAT_XZ,
        preset=9 | lzma.PRESET_EXTREME,
    )
    header = {
        "schema": SCHEMA,
        "bins": bins,
        "bits_per_code": bits_per_code,
        "code_storage": code_storage,
        "log_min": log_min,
        "log_step": log_step,
        "dtype": "uint8_codes_log2_steps",
        "shapes": [list(a.shape) for a in arrays],
        "code_count": int(codes_flat.size),
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    packet = (
        MAGIC
        + struct.pack(HEADER_LEN_FMT, len(header_bytes))
        + header_bytes
        + compressed_codes
    )
    decoded = decode_step_maps(packet)
    rel_errors = []
    for ref, got in zip(arrays, decoded, strict=True):
        rel_errors.append(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12))
    rel = np.concatenate([e.reshape(-1) for e in rel_errors])
    baseline = _fp32_lzma_baseline(arrays)
    per_map_baseline = sum(
        _standard_packet_total_bytes([a], bins=bins) for a in arrays
    )
    bundle_savings = per_map_baseline - len(packet)
    unique_codes, counts = np.unique(codes_flat, return_counts=True)
    probs = counts.astype(np.float64) / float(counts.sum())
    entropy_bits = float(-(probs * np.log2(probs)).sum())
    ideal_entropy_bytes = math.ceil(entropy_bits * int(codes_flat.size) / 8.0)
    return StepMapPacket(
        packet=packet,
        schema=SCHEMA,
        map_count=len(arrays),
        bins=bins,
        bits_per_code=bits_per_code,
        code_storage=code_storage,
        code_count=int(codes_flat.size),
        packed_code_bytes=len(packed_codes),
        payload_bytes=len(compressed_codes),
        header_bytes=len(header_bytes) + len(MAGIC) + struct.calcsize(HEADER_LEN_FMT),
        total_bytes=len(packet),
        per_map_packet_baseline_bytes=per_map_baseline,
        bundle_savings_bytes=bundle_savings,
        bundle_savings_ratio=(
            float(bundle_savings) / float(per_map_baseline)
            if per_map_baseline
            else 0.0
        ),
        unique_code_count=int(unique_codes.size),
        code_entropy_bits_per_symbol=entropy_bits,
        code_entropy_ideal_bytes=ideal_entropy_bytes,
        fp32_lzma_baseline_bytes=baseline,
        max_relative_error=float(rel.max()),
        mean_relative_error=float(rel.mean()),
    )


def encode_step_maps_adaptive(
    step_maps: list[np.ndarray],
    *,
    map_importance: list[float] | np.ndarray,
    bin_choices: tuple[int, ...] = (128, 16, 4),
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
    constant_importance_quantile: float | None = None,
) -> AdaptiveStepMapPacket:
    """Encode maps with per-map precision selected by saliency importance.

    This is a receiver-visible mixed int8/int4/int2 grammar. The scorer/saliency
    may choose the precision during compression, but the archive packet carries
    the selected groups and the receiver only decodes codes. When
    ``constant_importance_quantile`` is supplied, the least-important maps are
    encoded as header-only constant log2 fills: a receiver-visible run-length
    group with zero per-coefficient code bits.
    """

    arrays = [_validate_step_map(a) for a in step_maps]
    if not arrays:
        raise SnervStepMapCoderError("at least one step map is required")
    importance = np.asarray(map_importance, dtype=np.float64).reshape(-1)
    if importance.size != len(arrays):
        raise SnervStepMapCoderError(
            f"map_importance has {importance.size} entries for {len(arrays)} maps"
        )
    if not np.all(np.isfinite(importance)):
        raise SnervStepMapCoderError("map_importance must be finite")
    choices = tuple(sorted({int(v) for v in bin_choices}, reverse=True))
    if not choices or any(v < 2 or v > 256 for v in choices):
        raise SnervStepMapCoderError("bin choices must be in [2, 256]")
    if len(choices) == 1:
        assigned = np.full(len(arrays), choices[0], dtype=np.int64)
    else:
        hi = float(np.quantile(importance, high_quantile))
        lo = float(np.quantile(importance, low_quantile))
        high_bins = choices[0]
        low_bins = choices[-1]
        mid_bins = choices[min(1, len(choices) - 1)]
        assigned = np.full(len(arrays), mid_bins, dtype=np.int64)
        assigned[importance >= hi] = high_bins
        assigned[importance <= lo] = low_bins
    constant_indices: set[int] = set()
    if constant_importance_quantile is not None:
        if not 0.0 <= constant_importance_quantile <= 1.0:
            raise SnervStepMapCoderError("constant_importance_quantile must be in [0, 1]")
        constant_count = math.ceil(float(importance.size) * constant_importance_quantile)
        constant_count = max(0, min(int(importance.size), constant_count))
        ranked = sorted(
            range(int(importance.size)),
            key=lambda idx: (float(importance[idx]), int(idx)),
        )
        constant_indices = {
            int(idx) for idx in ranked[:constant_count]
        }

    groups: list[dict[str, Any]] = []
    payload = bytearray()
    if constant_indices:
        indices = sorted(constant_indices)
        groups.append(
            {
                "kind": "constant_log2_fill",
                "bins": 0,
                "bits_per_code": 0,
                "code_storage": "run_length_constant_log2_f32",
                "map_indices": indices,
                "payload_offset": 0,
                "payload_bytes": 0,
                "packed_code_bytes": 0,
                "log2_values": [
                    float(np.mean(np.log2(arrays[idx].astype(np.float64))))
                    for idx in indices
                ],
                "shapes": [list(arrays[idx].shape) for idx in indices],
                "code_count": 0,
            }
        )
    for bins in choices:
        indices = [
            idx
            for idx, value in enumerate(assigned.tolist())
            if value == bins and idx not in constant_indices
        ]
        if not indices:
            continue
        group_arrays = [arrays[idx] for idx in indices]
        group_meta, compressed_codes = _encode_group_payload(group_arrays, bins=bins)
        offset = len(payload)
        payload.extend(compressed_codes)
        groups.append(
            {
                "bins": bins,
                "bits_per_code": group_meta["bits_per_code"],
                "code_storage": group_meta["code_storage"],
                "map_indices": indices,
                "payload_offset": offset,
                "payload_bytes": len(compressed_codes),
                "packed_code_bytes": group_meta["packed_code_bytes"],
                "log_min": group_meta["log_min"],
                "log_step": group_meta["log_step"],
                "shapes": group_meta["shapes"],
                "code_count": group_meta["code_count"],
            }
        )

    header = {
        "schema": ADAPTIVE_SCHEMA,
        "map_count": len(arrays),
        "groups": groups,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    packet_bytes = (
        ADAPTIVE_MAGIC
        + struct.pack(HEADER_LEN_FMT, len(header_bytes))
        + header_bytes
        + bytes(payload)
    )
    decoded = decode_step_maps(packet_bytes)
    rel_errors = []
    for ref, got in zip(arrays, decoded, strict=True):
        rel_errors.append(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12))
    rel = np.concatenate([e.reshape(-1) for e in rel_errors])
    return AdaptiveStepMapPacket(
        packet=packet_bytes,
        schema=ADAPTIVE_SCHEMA,
        map_count=len(arrays),
        groups=tuple(groups),
        payload_bytes=len(payload),
        header_bytes=len(header_bytes)
        + len(ADAPTIVE_MAGIC)
        + struct.calcsize(HEADER_LEN_FMT),
        total_bytes=len(packet_bytes),
        fp32_lzma_baseline_bytes=_fp32_lzma_baseline(arrays),
        max_relative_error=float(rel.max()),
        mean_relative_error=float(rel.mean()),
    )


def encode_step_maps_waterfill(
    step_maps: list[np.ndarray],
    *,
    map_importance: list[float] | np.ndarray,
    target_bits_per_coeff: float = 4.0,
    precision_ladder: tuple[Any, ...] = ("constant", 4, 16, 256, "fp16"),
) -> AdaptiveStepMapPacket:
    """Encode maps with a score-aware reverse-waterfill precision policy.

    The archive carries the chosen precision groups. Compression-time scorer
    saliency may decide which maps deserve fp16/int8/int4/int2/constant
    treatment, but the receiver only consumes bytes in this packet.
    """

    arrays = [_validate_step_map(a) for a in step_maps]
    if not arrays:
        raise SnervStepMapCoderError("at least one step map is required")
    importance = np.asarray(map_importance, dtype=np.float64).reshape(-1)
    if importance.size != len(arrays):
        raise SnervStepMapCoderError(
            f"map_importance has {importance.size} entries for {len(arrays)} maps"
        )
    if not np.all(np.isfinite(importance)):
        raise SnervStepMapCoderError("map_importance must be finite")
    if target_bits_per_coeff < 0:
        raise SnervStepMapCoderError("target_bits_per_coeff must be non-negative")
    levels = _precision_levels(precision_ladder)
    assignments = _waterfill_assignments(
        importance=importance,
        coeff_counts=np.asarray([a.size for a in arrays], dtype=np.float64),
        target_bits_per_coeff=float(target_bits_per_coeff),
        levels=levels,
    )
    packet_bytes, groups, payload_bytes = _pack_adaptive_groups(
        arrays,
        assignments=assignments,
        policy={
            "name": "score_aware_reverse_waterfill",
            "target_bits_per_coeff": float(target_bits_per_coeff),
            "precision_ladder": [level["label"] for level in levels],
        },
    )
    decoded = decode_step_maps(packet_bytes)
    rel_errors = []
    for ref, got in zip(arrays, decoded, strict=True):
        rel_errors.append(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12))
    rel = np.concatenate([e.reshape(-1) for e in rel_errors])
    return AdaptiveStepMapPacket(
        packet=packet_bytes,
        schema=ADAPTIVE_SCHEMA,
        map_count=len(arrays),
        groups=tuple(groups),
        payload_bytes=payload_bytes,
        header_bytes=len(packet_bytes) - payload_bytes,
        total_bytes=len(packet_bytes),
        fp32_lzma_baseline_bytes=_fp32_lzma_baseline(arrays),
        max_relative_error=float(rel.max()),
        mean_relative_error=float(rel.mean()),
    )


def decode_step_maps(packet: bytes) -> list[np.ndarray]:
    """Decode a packet produced by :func:`encode_step_maps`."""

    if packet.startswith(ADAPTIVE_MAGIC):
        return _decode_adaptive_step_maps(packet)
    if not packet.startswith(MAGIC):
        raise SnervStepMapCoderError("bad SNeRV step-map packet magic")
    offset = len(MAGIC)
    if len(packet) < offset + struct.calcsize(HEADER_LEN_FMT):
        raise SnervStepMapCoderError("truncated SNeRV step-map packet header")
    (header_len,) = struct.unpack(
        HEADER_LEN_FMT, packet[offset : offset + struct.calcsize(HEADER_LEN_FMT)]
    )
    offset += struct.calcsize(HEADER_LEN_FMT)
    header_end = offset + header_len
    if header_end > len(packet):
        raise SnervStepMapCoderError("declared header length exceeds packet size")
    header = json.loads(packet[offset:header_end].decode("utf-8"))
    if header.get("schema") != SCHEMA:
        raise SnervStepMapCoderError(f"unsupported schema: {header.get('schema')!r}")
    bins = int(header["bins"])
    if bins < 2 or bins > 256:
        raise SnervStepMapCoderError("invalid bins in packet")
    bits_per_code = int(header.get("bits_per_code", 8))
    if bits_per_code < 1 or bits_per_code > 8:
        raise SnervStepMapCoderError("invalid bits_per_code in packet")
    shapes = [tuple(int(v) for v in shape) for shape in header["shapes"]]
    code_count = int(header["code_count"])
    raw_codes = lzma.decompress(packet[header_end:])
    codes = _unpack_codes(raw_codes, code_count, bits_per_code)
    if codes.size != code_count:
        raise SnervStepMapCoderError(
            f"code count {codes.size} != packet header {code_count}"
        )
    values = np.exp2(float(header["log_min"]) + codes.astype(np.float64) * float(header["log_step"]))
    out = []
    cursor = 0
    for shape in shapes:
        n = int(np.prod(shape))
        out.append(values[cursor : cursor + n].reshape(shape).astype(np.float32))
        cursor += n
    if cursor != values.size:
        raise SnervStepMapCoderError("unused codes after decoding step maps")
    return out


def _validate_step_map(step_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(step_map, dtype=np.float32)
    if arr.size == 0:
        raise SnervStepMapCoderError("step maps must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise SnervStepMapCoderError("step maps must be finite")
    if not np.all(arr > 0):
        raise SnervStepMapCoderError("step maps must be strictly positive")
    return arr


def _fp32_lzma_baseline(arrays: list[np.ndarray]) -> int:
    raw = b"".join(np.asarray(a, dtype=np.float32).astype("<f4").tobytes() for a in arrays)
    return len(
        lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    )


def _decode_adaptive_step_maps(packet: bytes) -> list[np.ndarray]:
    offset = len(ADAPTIVE_MAGIC)
    if len(packet) < offset + struct.calcsize(HEADER_LEN_FMT):
        raise SnervStepMapCoderError("truncated adaptive SNeRV step-map header")
    (header_len,) = struct.unpack(
        HEADER_LEN_FMT, packet[offset : offset + struct.calcsize(HEADER_LEN_FMT)]
    )
    offset += struct.calcsize(HEADER_LEN_FMT)
    header_end = offset + header_len
    if header_end > len(packet):
        raise SnervStepMapCoderError("declared adaptive header exceeds packet size")
    header = json.loads(packet[offset:header_end].decode("utf-8"))
    if header.get("schema") != ADAPTIVE_SCHEMA:
        raise SnervStepMapCoderError(
            f"unsupported adaptive schema: {header.get('schema')!r}"
        )
    map_count = int(header["map_count"])
    if map_count <= 0:
        raise SnervStepMapCoderError("adaptive packet map_count must be positive")
    out: list[np.ndarray | None] = [None] * map_count
    payload = packet[header_end:]
    seen_indices: set[int] = set()
    payload_ranges: list[tuple[int, int, int]] = []
    for group_index, group in enumerate(header["groups"]):
        indices = [int(idx) for idx in group["map_indices"]]
        if not indices:
            raise SnervStepMapCoderError("adaptive group has no map indices")
        for idx in indices:
            if idx < 0 or idx >= map_count:
                raise SnervStepMapCoderError(
                    f"adaptive map index {idx} outside map_count {map_count}"
                )
            if idx in seen_indices:
                raise SnervStepMapCoderError(f"duplicate adaptive map index {idx}")
            seen_indices.add(idx)
        if group.get("kind") == "constant_log2_shared_shape":
            if int(group.get("payload_bytes", 0)) != 0:
                raise SnervStepMapCoderError(
                    "adaptive shared-shape constant group must not carry payload bytes"
                )
            shape = tuple(int(v) for v in group["shape"])
            log2_value = float(group["log2_value"])
            for idx in indices:
                out[idx] = np.full(shape, np.exp2(log2_value), dtype=np.float32)
            continue
        if group.get("kind") == "constant_log2_shared_shape_f64_lzma":
            shape = tuple(int(v) for v in group["shape"])
            start = int(group["payload_offset"])
            end = start + int(group["payload_bytes"])
            if start < 0 or end > len(payload) or end < start:
                raise SnervStepMapCoderError(
                    "adaptive shared-shape constant payload bounds invalid"
                )
            if end == start:
                raise SnervStepMapCoderError(
                    "adaptive shared-shape constant payload must be non-empty"
                )
            payload_ranges.append((start, end, group_index))
            raw = lzma.decompress(payload[start:end])
            expected = len(indices) * np.dtype("<f8").itemsize
            if len(raw) != expected:
                raise SnervStepMapCoderError(
                    f"adaptive shared-shape constant raw bytes {len(raw)} != expected {expected}"
                )
            values = np.frombuffer(raw, dtype="<f8")
            if values.size != len(indices):
                raise SnervStepMapCoderError(
                    "adaptive shared-shape constant map count mismatch"
                )
            for idx, log2_value in zip(indices, values, strict=True):
                out[idx] = np.full(
                    shape,
                    np.exp2(float(log2_value)),
                    dtype=np.float32,
                )
            continue
        if group.get("kind") == "constant_log2_fill":
            if int(group.get("payload_bytes", 0)) != 0:
                raise SnervStepMapCoderError(
                    "adaptive constant group must not carry payload bytes"
                )
            shapes = [tuple(int(v) for v in shape) for shape in group["shapes"]]
            log2_values = [float(v) for v in group["log2_values"]]
            if len(shapes) != len(indices) or len(log2_values) != len(indices):
                raise SnervStepMapCoderError("adaptive constant group map count mismatch")
            for idx, shape, log2_value in zip(
                indices,
                shapes,
                log2_values,
                strict=True,
            ):
                out[idx] = np.full(shape, np.exp2(log2_value), dtype=np.float32)
            continue
        if group.get("kind") == "fp16_steps_lzma":
            shapes = [tuple(int(v) for v in shape) for shape in group["shapes"]]
            start = int(group["payload_offset"])
            end = start + int(group["payload_bytes"])
            if start < 0 or end > len(payload) or end < start:
                raise SnervStepMapCoderError("adaptive fp16 payload bounds invalid")
            if end == start:
                raise SnervStepMapCoderError("adaptive fp16 payload must be non-empty")
            payload_ranges.append((start, end, group_index))
            raw = lzma.decompress(payload[start:end])
            expected = int(group["raw_bytes"])
            if len(raw) != expected:
                raise SnervStepMapCoderError(
                    f"adaptive fp16 raw bytes {len(raw)} != expected {expected}"
                )
            cursor = 0
            decoded = []
            for shape in shapes:
                n = int(np.prod(shape))
                nbytes = n * np.dtype("<f2").itemsize
                view = np.frombuffer(raw[cursor : cursor + nbytes], dtype="<f2")
                if view.size != n:
                    raise SnervStepMapCoderError("truncated adaptive fp16 group")
                decoded.append(view.reshape(shape).astype(np.float32))
                cursor += nbytes
            if cursor != len(raw):
                raise SnervStepMapCoderError("unused adaptive fp16 bytes")
            if len(decoded) != len(indices):
                raise SnervStepMapCoderError("adaptive fp16 group map count mismatch")
            for idx, arr in zip(indices, decoded, strict=True):
                out[idx] = arr
            continue
        start = int(group["payload_offset"])
        end = start + int(group["payload_bytes"])
        if start < 0 or end > len(payload) or end < start:
            raise SnervStepMapCoderError("adaptive code payload bounds invalid")
        if end == start:
            raise SnervStepMapCoderError("adaptive code payload must be non-empty")
        payload_ranges.append((start, end, group_index))
        raw_codes = lzma.decompress(payload[start:end])
        codes = _unpack_codes(
            raw_codes,
            int(group["code_count"]),
            int(group["bits_per_code"]),
        )
        values = np.exp2(
            float(group["log_min"])
            + codes.astype(np.float64) * float(group["log_step"])
        )
        decoded = []
        cursor = 0
        for shape in group["shapes"]:
            out_shape = tuple(int(v) for v in shape)
            n = int(np.prod(out_shape))
            decoded.append(
                values[cursor : cursor + n].reshape(out_shape).astype(np.float32)
            )
            cursor += n
        if cursor != values.size:
            raise SnervStepMapCoderError("unused adaptive codes after group decode")
        if len(decoded) != len(indices):
            raise SnervStepMapCoderError("adaptive group map count mismatch")
        for idx, arr in zip(indices, decoded, strict=True):
            out[idx] = arr
    _validate_adaptive_payload_coverage(payload_ranges, payload_len=len(payload))
    if any(arr is None for arr in out):
        raise SnervStepMapCoderError("adaptive packet left maps undecoded")
    return [arr for arr in out if arr is not None]


def _validate_adaptive_payload_coverage(
    ranges: list[tuple[int, int, int]],
    *,
    payload_len: int,
) -> None:
    """Reject adaptive packets with unowned, overlapping, or trailing payload bytes."""

    cursor = 0
    for start, end, group_index in sorted(ranges, key=lambda row: (row[0], row[1])):
        if start != cursor:
            raise SnervStepMapCoderError(
                "adaptive payload ranges are not contiguous at group "
                f"{group_index}: offset {start} != expected {cursor}"
            )
        if end <= start:
            raise SnervStepMapCoderError(
                f"adaptive payload range for group {group_index} is empty"
            )
        cursor = end
    if cursor != int(payload_len):
        raise SnervStepMapCoderError(
            f"unused adaptive payload bytes: consumed {cursor} of {payload_len}"
        )


def _precision_levels(precision_ladder: tuple[Any, ...]) -> list[dict[str, Any]]:
    if not precision_ladder:
        raise SnervStepMapCoderError("precision_ladder must be non-empty")

    levels: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in precision_ladder:
        if isinstance(item, str):
            token = item.strip().lower()
            if token in {"constant", "zero", "rle", "run_length"}:
                level = {
                    "label": "constant",
                    "kind": "constant_log2_fill",
                    "bins": 0,
                    "bits": 0,
                }
            elif token in {"fp16", "float16", "half"}:
                level = {
                    "label": "fp16",
                    "kind": "fp16_steps_lzma",
                    "bins": -1,
                    "bits": 16,
                }
            else:
                raise SnervStepMapCoderError(
                    f"unsupported precision ladder token: {item!r}"
                )
        else:
            bins = int(item)
            if bins < 2 or bins > 256:
                raise SnervStepMapCoderError("precision ladder bins must be in [2, 256]")
            bits = _bits_per_code(bins)
            level = {
                "label": f"int{bits}_bins{bins}",
                "kind": "log2_quantized_codes",
                "bins": bins,
                "bits": bits,
            }
        key = (str(level["kind"]), int(level["bits"]))
        if key not in seen:
            seen.add(key)
            levels.append(level)

    levels.sort(key=lambda level: (int(level["bits"]), str(level["label"])))
    if not levels:
        raise SnervStepMapCoderError("precision_ladder produced no levels")
    return levels


def _waterfill_assignments(
    *,
    importance: np.ndarray,
    coeff_counts: np.ndarray,
    target_bits_per_coeff: float,
    levels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if coeff_counts.size != importance.size:
        raise SnervStepMapCoderError("importance and coeff_counts length mismatch")
    if not np.all(coeff_counts > 0):
        raise SnervStepMapCoderError("coeff_counts must be positive")

    total_coeffs = float(np.sum(coeff_counts))
    bits = np.asarray([float(level["bits"]) for level in levels], dtype=np.float64)
    min_bits = float(bits[0])
    max_bits = float(bits[-1])
    min_total = min_bits * total_coeffs
    max_total = max_bits * total_coeffs
    target_total = float(target_bits_per_coeff) * total_coeffs
    target_total = min(max(target_total, min_total), max_total)
    if target_total <= min_total + 1e-9:
        return [levels[0] for _ in range(int(importance.size))]
    if target_total >= max_total - 1e-9:
        return [levels[-1] for _ in range(int(importance.size))]

    signal = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
    signal = (
        np.ones_like(signal)
        if not np.any(signal > 0.0)
        else signal + float(np.max(signal)) * 1e-12
    )

    def weighted_total(lambda_value: float) -> float:
        continuous = np.log2(signal / max(lambda_value, np.finfo(np.float64).tiny))
        continuous = np.clip(continuous, min_bits, max_bits)
        return float(np.sum(continuous * coeff_counts))

    lo = np.finfo(np.float64).tiny
    hi = float(np.max(signal) * (2.0 ** (-min_bits)) * 2.0)
    hi = max(hi, lo * 2.0)
    while weighted_total(hi) > target_total and hi < 1e300:
        hi *= 2.0
    for _ in range(96):
        mid = math.sqrt(lo * hi) if lo > 0.0 else hi / 2.0
        if weighted_total(mid) > target_total:
            lo = mid
        else:
            hi = mid

    continuous = np.log2(signal / max(hi, np.finfo(np.float64).tiny))
    continuous = np.clip(continuous, min_bits, max_bits)
    assignment_indices: list[int] = []
    for value in continuous:
        idx = 0
        for level_idx, level_bits in enumerate(bits):
            if level_bits <= float(value) + 1e-9:
                idx = level_idx
        assignment_indices.append(idx)

    spent = float(
        sum(bits[idx] * float(count) for idx, count in zip(assignment_indices, coeff_counts, strict=True))
    )
    leftover = target_total - spent
    for _ in range(len(assignment_indices) * max(len(levels) - 1, 1)):
        best: tuple[float, float, int] | None = None
        for idx, level_idx in enumerate(assignment_indices):
            if level_idx >= len(levels) - 1:
                continue
            upgrade_bits = float(bits[level_idx + 1] - bits[level_idx])
            cost = upgrade_bits * float(coeff_counts[idx])
            if cost <= 0.0 or cost > leftover + 1e-9:
                continue
            value_per_bit = float(signal[idx]) / cost
            candidate = (value_per_bit, float(signal[idx]), -idx)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        idx = -int(best[2])
        old_level = assignment_indices[idx]
        assignment_indices[idx] = old_level + 1
        leftover -= float(bits[old_level + 1] - bits[old_level]) * float(
            coeff_counts[idx]
        )

    return [levels[idx] for idx in assignment_indices]


def _pack_adaptive_groups(
    arrays: list[np.ndarray],
    *,
    assignments: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[bytes, list[dict[str, Any]], int]:
    if len(assignments) != len(arrays):
        raise SnervStepMapCoderError("assignment count must match step-map count")

    payload = bytearray()
    groups: list[dict[str, Any]] = []
    labels = sorted(
        {str(level["label"]) for level in assignments},
        key=lambda label: (
            int(next(level["bits"] for level in assignments if level["label"] == label)),
            label,
        ),
    )
    for label in labels:
        indices = [
            idx
            for idx, level in enumerate(assignments)
            if str(level["label"]) == label
        ]
        if not indices:
            continue
        level = assignments[indices[0]]
        kind = str(level["kind"])
        group_arrays = [arrays[idx] for idx in indices]
        if kind == "constant_log2_fill":
            log2_values = [
                float(np.mean(np.log2(arr.astype(np.float64))))
                for arr in group_arrays
            ]
            if _should_use_shared_shape_constant_payload(group_arrays):
                raw = np.asarray(log2_values, dtype="<f8").tobytes()
                compressed = lzma.compress(
                    raw,
                    format=lzma.FORMAT_XZ,
                    preset=9 | lzma.PRESET_EXTREME,
                )
                offset = len(payload)
                payload.extend(compressed)
                groups.append(
                    {
                        "kind": "constant_log2_shared_shape_f64_lzma",
                        "precision_label": label,
                        "bins": 0,
                        "bits_per_code": 0,
                        "code_storage": (
                            "run_length_constant_log2_shared_shape_f64_lzma"
                        ),
                        "map_indices": indices,
                        "payload_offset": offset,
                        "payload_bytes": len(compressed),
                        "packed_code_bytes": len(raw),
                        "raw_bytes": len(raw),
                        "log2_dtype": "float64_le",
                        "shape": list(group_arrays[0].shape),
                        "code_count": 0,
                    }
                )
                continue
            groups.append(
                {
                    "kind": "constant_log2_fill",
                    "precision_label": label,
                    "bins": 0,
                    "bits_per_code": 0,
                    "code_storage": "run_length_constant_log2_f32",
                    "map_indices": indices,
                    "payload_offset": 0,
                    "payload_bytes": 0,
                    "packed_code_bytes": 0,
                    "log2_values": log2_values,
                    "shapes": [list(arr.shape) for arr in group_arrays],
                    "code_count": 0,
                }
            )
            continue
        if kind == "fp16_steps_lzma":
            raw = b"".join(
                np.asarray(arr, dtype="<f2").ravel().tobytes()
                for arr in group_arrays
            )
            compressed = lzma.compress(
                raw,
                format=lzma.FORMAT_XZ,
                preset=9 | lzma.PRESET_EXTREME,
            )
            offset = len(payload)
            payload.extend(compressed)
            groups.append(
                {
                    "kind": "fp16_steps_lzma",
                    "precision_label": label,
                    "bins": -1,
                    "bits_per_code": 16,
                    "code_storage": "lzma_float16_steps",
                    "map_indices": indices,
                    "payload_offset": offset,
                    "payload_bytes": len(compressed),
                    "packed_code_bytes": len(raw),
                    "raw_bytes": len(raw),
                    "shapes": [list(arr.shape) for arr in group_arrays],
                    "code_count": int(sum(arr.size for arr in group_arrays)),
                }
            )
            continue

        bins = int(level["bins"])
        group_meta, compressed_codes = _encode_group_payload(group_arrays, bins=bins)
        offset = len(payload)
        payload.extend(compressed_codes)
        groups.append(
            {
                "kind": "log2_quantized_codes",
                "precision_label": label,
                "bins": bins,
                "bits_per_code": group_meta["bits_per_code"],
                "code_storage": group_meta["code_storage"],
                "map_indices": indices,
                "payload_offset": offset,
                "payload_bytes": len(compressed_codes),
                "packed_code_bytes": group_meta["packed_code_bytes"],
                "log_min": group_meta["log_min"],
                "log_step": group_meta["log_step"],
                "shapes": group_meta["shapes"],
                "code_count": group_meta["code_count"],
            }
        )

    header = {
        "schema": ADAPTIVE_SCHEMA,
        "map_count": len(arrays),
        "policy": policy,
        "groups": groups,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    packet = (
        ADAPTIVE_MAGIC
        + struct.pack(HEADER_LEN_FMT, len(header_bytes))
        + header_bytes
        + bytes(payload)
    )
    return packet, groups, len(payload)


def _should_use_shared_shape_constant_payload(arrays: list[np.ndarray]) -> bool:
    if len(arrays) < COMPACT_CONSTANT_SHARED_SHAPE_MIN_MAPS:
        return False
    first_shape = tuple(int(v) for v in arrays[0].shape)
    return all(tuple(int(v) for v in arr.shape) == first_shape for arr in arrays)


def _bits_per_code(bins: int) -> int:
    return max(1, int(bins - 1).bit_length())


def _encode_group_payload(
    arrays: list[np.ndarray],
    *,
    bins: int,
) -> tuple[dict[str, Any], bytes]:
    flat = np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)
    logs = np.log2(flat)
    log_min = float(logs.min())
    log_max = float(logs.max())
    if log_max <= log_min:
        log_step = 1.0
        codes_flat = np.zeros_like(logs, dtype=np.uint8)
    else:
        log_step = (log_max - log_min) / float(bins - 1)
        q = np.rint((logs - log_min) / log_step).clip(0, bins - 1)
        codes_flat = q.astype(np.uint8)
    bits_per_code = _bits_per_code(bins)
    packed_codes, code_storage = _pack_codes(codes_flat, bits_per_code)
    compressed_codes = lzma.compress(
        packed_codes,
        format=lzma.FORMAT_XZ,
        preset=9 | lzma.PRESET_EXTREME,
    )
    return (
        {
            "bits_per_code": bits_per_code,
            "code_storage": code_storage,
            "code_count": int(codes_flat.size),
            "packed_code_bytes": len(packed_codes),
            "log_min": log_min,
            "log_step": log_step,
            "shapes": [list(a.shape) for a in arrays],
        },
        compressed_codes,
    )


def _standard_packet_total_bytes(
    arrays: list[np.ndarray],
    *,
    bins: int,
) -> int:
    group_meta, compressed_codes = _encode_group_payload(arrays, bins=bins)
    header = {
        "schema": SCHEMA,
        "bins": bins,
        "bits_per_code": group_meta["bits_per_code"],
        "code_storage": group_meta["code_storage"],
        "log_min": group_meta["log_min"],
        "log_step": group_meta["log_step"],
        "dtype": "uint8_codes_log2_steps",
        "shapes": group_meta["shapes"],
        "code_count": group_meta["code_count"],
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return (
        len(MAGIC)
        + struct.calcsize(HEADER_LEN_FMT)
        + len(header_bytes)
        + len(compressed_codes)
    )


def _pack_codes(codes: np.ndarray, bits_per_code: int) -> tuple[bytes, str]:
    """Pack fixed-width integer codes, least-significant bits first."""

    if bits_per_code >= 8:
        return np.asarray(codes, dtype=np.uint8).tobytes(), "uint8"
    acc = 0
    nbits = 0
    out = bytearray()
    mask = (1 << bits_per_code) - 1
    for value in np.asarray(codes, dtype=np.uint8).reshape(-1):
        acc |= (int(value) & mask) << nbits
        nbits += bits_per_code
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out), "packed_bits_lsb"


def _unpack_codes(raw: bytes, code_count: int, bits_per_code: int) -> np.ndarray:
    """Unpack fixed-width integer codes produced by :func:`_pack_codes`."""

    if bits_per_code >= 8:
        return np.frombuffer(raw, dtype=np.uint8)[:code_count]
    out = np.empty(code_count, dtype=np.uint8)
    mask = (1 << bits_per_code) - 1
    acc = 0
    nbits = 0
    cursor = 0
    for byte in raw:
        acc |= int(byte) << nbits
        nbits += 8
        while nbits >= bits_per_code and cursor < code_count:
            out[cursor] = acc & mask
            cursor += 1
            acc >>= bits_per_code
            nbits -= bits_per_code
    if cursor != code_count:
        raise SnervStepMapCoderError(
            f"unpacked {cursor} codes, expected {code_count}"
        )
    return out
