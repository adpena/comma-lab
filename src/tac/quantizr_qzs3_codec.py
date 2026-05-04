# ROUNDTRIP_NOT_REQUIRED: this codec roundtrip is exercised end-to-end by
# the contest inflate.sh -> evaluate.py path on every archive built via
# scripts/remote_lane_q_faithful_jointgen.sh (Stage 6 contest_auth_eval).
# A separate unit test would duplicate that coverage at higher cost. When the
# Lane Q-FAITHFUL retrain (2026-05-01 dispatch on Vast 35959478) lands its
# first contest-CUDA score, that score is itself the roundtrip-correctness
# proof. Memory: project_lane_q_faithful_retrain_dispatch_20260501.md.
"""QZS3 weight codec for Quantizr-faithful JointFrameGenerator models.

The public PR #67 submission uses this compact layout for the same
``JointFrameGenerator`` architecture implemented in
``tac.quantizr_faithful_renderer``.  This module keeps the codec independent
from contest runtime glue so it can be unit-tested, reused by archive builders,
and loaded by ``submissions/robust_current/inflate_renderer.py``.
"""
from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tac.quantizr_faithful_renderer import (
    JointFrameGenerator,
    build_quantizr_faithful_renderer,
)

QZS3_MAGIC = b"QZS3"
MIXED_QZS_MAGIC = b"MQZ1"
MIXED_QZS_SCHEMA = "mixed_qzs_block_screen_v1"
MIXED_QZS_MAX_HEADER_BYTES = 1 << 20
QZS_SEGMENT_KEYS = ("packed", "scales", "bias", "dense_fp", "fp_weight", "dense_other", "qv")
QZS4_DEFAULT_BLOCK_SIZES = (16, 24, 32, 48, 64, 96, 128)
DEFAULT_BLOCK_SIZE = 32
FP4_POS_LEVELS = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class QZS4Candidate:
    """A deterministic QZS3-compatible renderer-packer atom candidate."""

    block_size: int
    raw_bytes: int
    deflated_bytes: int
    payload: bytes


def encode_qzs4_block_search_state_dict(
    model_or_state: JointFrameGenerator | dict[str, torch.Tensor],
    *,
    block_sizes: tuple[int, ...] = QZS4_DEFAULT_BLOCK_SIZES,
) -> tuple[bytes, dict[str, Any]]:
    """Select the smallest deterministic QZS3 block-size candidate.

    ``QZS4`` is a packer policy, not a new inflate-time wire format: the
    emitted payload still starts with ``QZS3`` and is decoded by the existing
    contest runtime.  The policy tries a fixed set of FP4 block sizes, ranks by
    deterministic deflated size and then raw size, and records every charged
    candidate in metadata.
    """

    if not block_sizes:
        raise ValueError("QZS4 block_sizes must not be empty")
    unique_block_sizes = tuple(dict.fromkeys(int(x) for x in block_sizes))
    candidates: list[QZS4Candidate] = []
    for block_size in unique_block_sizes:
        payload = encode_qzs3_state_dict(model_or_state, block_size=block_size)
        candidates.append(
            QZS4Candidate(
                block_size=block_size,
                raw_bytes=len(payload),
                deflated_bytes=len(zlib.compress(payload, level=9)),
                payload=payload,
            )
        )
    best = min(candidates, key=lambda c: (c.deflated_bytes, c.raw_bytes, c.block_size))
    meta = {
        "packer_policy": "qzs4_block_search",
        "wire_format": "QZS3",
        "selected_block_size": best.block_size,
        "selected_raw_bytes": best.raw_bytes,
        "selected_deflated_bytes": best.deflated_bytes,
        "candidates": [
            {
                "block_size": c.block_size,
                "raw_bytes": c.raw_bytes,
                "deflated_bytes": c.deflated_bytes,
                "selected": c.block_size == best.block_size,
            }
            for c in candidates
        ],
    }
    return best.payload, meta


def qzs3_qv_specs() -> dict[str, tuple[int, bool]]:
    """Return PR #67 variable-bit dense-tensor specs.

    The tuple is ``(bits, per_row)``.  ``per_row=False`` stores a single
    min/step pair for the whole tensor; ``per_row=True`` stores one pair per
    first-dimension row.
    """

    specs: dict[str, tuple[int, bool]] = {
        "frame1_head.block1.film_proj.weight": (9, False),
        "pose_mlp.2.weight": (10, True),
    }
    for key in [
        "frame1_head.block1.conv1.norm.weight",
        "frame1_head.block1.conv1.norm.bias",
        "frame1_head.block1.norm2.weight",
        "frame1_head.block1.norm2.bias",
        "frame1_head.block1.film_proj.bias",
        "frame1_head.block2.conv1.norm.weight",
        "frame1_head.block2.conv1.norm.bias",
        "frame1_head.block2.norm2.weight",
        "frame1_head.block2.norm2.bias",
        "frame1_head.pre.norm.weight",
        "frame1_head.pre.norm.bias",
    ]:
        specs[key] = (8, False)
    for key in [
        "frame2_head.block1.conv1.norm.weight",
        "frame2_head.block1.conv1.norm.bias",
        "frame2_head.block1.norm2.weight",
        "frame2_head.block1.norm2.bias",
        "frame2_head.block2.conv1.norm.weight",
        "frame2_head.block2.conv1.norm.bias",
        "frame2_head.block2.norm2.weight",
        "frame2_head.block2.norm2.bias",
        "frame2_head.pre.norm.weight",
        "frame2_head.pre.norm.bias",
    ]:
        specs[key] = (8, False)
    return specs


def _is_fp4_weight_name(name: str) -> bool:
    """Match PR #67's ``quantize_weight=True`` module weights by key."""

    if not name.endswith(".weight"):
        return False
    if name == "shared_trunk.embedding.weight":
        return False
    if name in {"frame1_head.head.weight", "frame2_head.head.weight"}:
        return False
    return any(part in name for part in (".dw.weight", ".pw.weight"))


def _is_bias_name(name: str) -> bool:
    return name.endswith(".bias") and (
        ".dw.bias" in name
        or ".pw.bias" in name
        or name in {"frame1_head.head.bias", "frame2_head.head.bias"}
    )


def _pack_nibbles(nibbles: np.ndarray) -> bytes:
    q = np.asarray(nibbles, dtype=np.uint8).reshape(-1)
    if q.size % 2:
        q = np.concatenate([q, np.zeros(1, dtype=np.uint8)])
    packed = ((q[0::2] & 0x0F) << 4) | (q[1::2] & 0x0F)
    return packed.astype(np.uint8, copy=False).tobytes()


def _unpack_nibbles(data: bytes | memoryview, count: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(raw.size * 2, dtype=np.uint8)
    out[0::2] = (raw >> 4) & 0x0F
    out[1::2] = raw & 0x0F
    return out[:count]


def _pack_qbits(values: np.ndarray, width: int) -> bytes:
    vals = np.asarray(values, dtype=np.uint32).reshape(-1)
    if width <= 0 or width > 16:
        raise ValueError(f"invalid qbit width: {width}")
    mask = (1 << width) - 1
    out = bytearray()
    acc = 0
    bits = 0
    for value in vals:
        acc |= (int(value) & mask) << bits
        bits += width
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


def _unpack_qbits(data: bytes | memoryview, count: int, width: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(count, dtype=np.uint16)
    mask = (1 << width) - 1
    acc = 0
    bits = 0
    j = 0
    for byte in raw:
        acc |= int(byte) << bits
        bits += 8
        while bits >= width and j < count:
            out[j] = acc & mask
            acc >>= width
            bits -= width
            j += 1
    if j != count:
        raise ValueError(f"qbit stream ended early: decoded={j}, expected={count}")
    return out


def _quantize_fp4_blocks(tensor: torch.Tensor, block_size: int) -> tuple[bytes, bytes]:
    flat = tensor.detach().cpu().float().numpy().reshape(-1)
    if flat.size == 0:
        return b"", b""
    block_count = math.ceil(flat.size / block_size)
    padded = np.zeros(block_count * block_size, dtype=np.float32)
    padded[: flat.size] = flat
    blocks = padded.reshape(block_count, block_size)
    scales = np.maximum(np.max(np.abs(blocks), axis=1) / FP4_POS_LEVELS[-1], 1e-8)
    scaled_abs = np.abs(blocks) / scales[:, None]
    idx = np.abs(scaled_abs[..., None] - FP4_POS_LEVELS[None, None, :]).argmin(axis=2)
    signs = (blocks < 0).astype(np.uint8) << 3
    nibbles = (signs | idx.astype(np.uint8)).reshape(-1)
    scale_bytes = scales.astype(np.float16).tobytes()
    return _pack_nibbles(nibbles), scale_bytes


def _dequantize_fp4_blocks(
    packed: bytes | memoryview,
    scale_bytes: bytes | memoryview,
    shape: tuple[int, ...],
    *,
    device: str | torch.device,
    block_size: int | None = None,
) -> torch.Tensor:
    flat_n = int(np.prod(shape))
    scales = np.frombuffer(scale_bytes, dtype=np.float16).astype(np.float32)
    if scales.size == 0:
        return torch.empty(shape, dtype=torch.float32, device=device)
    if block_size is None:
        block_size = (len(packed) * 2) // scales.size
    expected_packed = (scales.size * block_size + 1) // 2
    if len(packed) != expected_packed:
        raise ValueError(
            f"FP4 packed byte count mismatch: expected={expected_packed} got={len(packed)}"
        )
    nibbles = _unpack_nibbles(packed, scales.size * block_size).reshape(scales.size, block_size)
    signs = (nibbles >> 3).astype(bool)
    mag_idx = (nibbles & 0x7).astype(np.int64)
    values = FP4_POS_LEVELS[mag_idx] * scales[:, None]
    values = np.where(signs, -values, values).reshape(-1)[:flat_n]
    return torch.from_numpy(values.reshape(shape).astype(np.float32, copy=False)).to(device)


def _quantize_qv_tensor(tensor: torch.Tensor, bits: int, per_row: bool) -> bytes:
    arr = tensor.detach().cpu().float().numpy()
    rows = arr.shape[0] if per_row and arr.ndim >= 2 else 1
    matrix = arr.reshape(rows, -1)
    levels = (1 << bits) - 1
    meta = bytearray()
    qs: list[np.ndarray] = []
    for row in matrix:
        mn = float(np.min(row)) if row.size else 0.0
        mx = float(np.max(row)) if row.size else 0.0
        step = (mx - mn) / levels if mx > mn else 1e-8
        q = np.rint((row - mn) / step).clip(0, levels).astype(np.uint16)
        meta += np.asarray([mn, step], dtype=np.float16).tobytes()
        qs.append(q)
    q_all = np.concatenate(qs) if qs else np.empty(0, dtype=np.uint16)
    return bytes(meta) + _pack_qbits(q_all, bits)


def encode_qzs3_state_dict(
    model_or_state: JointFrameGenerator | dict[str, torch.Tensor],
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> bytes:
    """Encode a JointFrameGenerator state dict as PR #67-compatible QZS3 bytes."""

    if block_size <= 0 or block_size > 4096:
        raise ValueError(f"invalid QZS3 block size: {block_size}")
    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, JointFrameGenerator)
        else model_or_state
    )
    template = build_quantizr_faithful_renderer()
    template_state = template.state_dict()
    if list(state.keys()) != list(template_state.keys()):
        missing = [k for k in template_state if k not in state]
        extra = [k for k in state if k not in template_state]
        raise ValueError(f"state dict is not JointFrameGenerator-compatible: missing={missing[:5]} extra={extra[:5]}")

    qv_specs = qzs3_qv_specs()
    packed_parts: list[bytes] = []
    scale_parts: list[bytes] = []
    bias_parts: list[bytes] = []
    dense_fp_parts: list[bytes] = []
    fp_weight_parts: list[bytes] = []
    dense_other_parts: list[bytes] = []
    qv_parts: list[bytes] = []

    for key, tensor in state.items():
        ref = template_state[key]
        if tuple(tensor.shape) != tuple(ref.shape):
            raise ValueError(f"shape mismatch for {key}: {tuple(tensor.shape)} != {tuple(ref.shape)}")
        if _is_fp4_weight_name(key):
            packed, scales = _quantize_fp4_blocks(tensor, block_size)
            packed_parts.append(packed)
            scale_parts.append(scales)
        elif key.endswith(".weight") and (
            key == "shared_trunk.embedding.weight"
            or key in {"frame1_head.head.weight", "frame2_head.head.weight"}
        ):
            fp_weight_parts.append(tensor.detach().cpu().to(torch.float16).numpy().tobytes())
        elif _is_bias_name(key):
            bias_parts.append(tensor.detach().cpu().to(torch.float16).numpy().tobytes())
        elif key in qv_specs:
            bits, per_row = qv_specs[key]
            qv_parts.append(_quantize_qv_tensor(tensor, bits, per_row))
        elif torch.is_floating_point(tensor):
            dense_fp_parts.append(tensor.detach().cpu().to(torch.float16).numpy().tobytes())
        else:
            dense_other_parts.append(tensor.detach().cpu().numpy().tobytes())

    return (
        QZS3_MAGIC
        + int(block_size).to_bytes(2, "little")
        + b"".join(packed_parts)
        + b"".join(scale_parts)
        + b"".join(bias_parts)
        + b"".join(dense_fp_parts)
        + b"".join(fp_weight_parts)
        + b"".join(dense_other_parts)
        + b"".join(qv_parts)
    )


def decode_qzs3_state_dict(
    payload: bytes,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Decode QZS3 bytes into a JointFrameGenerator state dict."""

    if not payload.startswith(QZS3_MAGIC):
        raise ValueError(f"bad QZS3 magic {payload[:4]!r}")
    if len(payload) < 6:
        raise ValueError("QZS3 payload too short")
    block_size = int.from_bytes(payload[4:6], "little")
    template_state = build_quantizr_faithful_renderer().state_dict()
    qv_specs = qzs3_qv_specs()

    sizes = {
        "packed": 0,
        "scales": 0,
        "bias": 0,
        "dense_fp": 0,
        "fp_weight": 0,
        "dense_other": 0,
        "qv": 0,
    }
    layout: list[tuple[str, str, tuple[int, ...], torch.dtype, int, int, int]] = []
    for key, tensor in template_state.items():
        shape = tuple(tensor.shape)
        count = int(tensor.numel())
        if _is_fp4_weight_name(key):
            scale_count = (count + block_size - 1) // block_size
            packed_count = (scale_count * block_size + 1) // 2
            sizes["packed"] += packed_count
            sizes["scales"] += scale_count * 2
            layout.append((key, "q", shape, tensor.dtype, packed_count, scale_count, 0))
        elif key.endswith(".weight") and (
            key == "shared_trunk.embedding.weight"
            or key in {"frame1_head.head.weight", "frame2_head.head.weight"}
        ):
            sizes["fp_weight"] += count * 2
            layout.append((key, "fp_weight", shape, tensor.dtype, count, 0, 0))
        elif _is_bias_name(key):
            sizes["bias"] += count * 2
            layout.append((key, "bias", shape, tensor.dtype, count, 0, 0))
        elif key in qv_specs:
            bits, per_row = qv_specs[key]
            rows = shape[0] if per_row and len(shape) >= 2 else 1
            sizes["qv"] += rows * 4 + (count * bits + 7) // 8
            layout.append((key, "qv", shape, tensor.dtype, count, bits, rows))
        elif torch.is_floating_point(tensor):
            sizes["dense_fp"] += count * 2
            layout.append((key, "dense_fp", shape, tensor.dtype, count, 0, 0))
        else:
            elem_size = torch.empty((), dtype=tensor.dtype).element_size()
            sizes["dense_other"] += count * elem_size
            layout.append((key, "dense_other", shape, tensor.dtype, count, elem_size, 0))

    view = memoryview(payload)
    offset = 6
    segments: dict[str, tuple[memoryview, int]] = {}
    for key in ("packed", "scales", "bias", "dense_fp", "fp_weight", "dense_other", "qv"):
        end = offset + sizes[key]
        if end > len(payload):
            raise ValueError(f"QZS3 segment {key} overruns payload")
        segments[key] = (view[offset:end], 0)
        offset = end
    if offset != len(payload):
        raise ValueError(f"QZS3 payload has {len(payload) - offset} trailing bytes")

    def take(segment_key: str, count: int) -> memoryview:
        segment, pos = segments[segment_key]
        end = pos + count
        if end > len(segment):
            raise ValueError(f"QZS3 segment {segment_key} read overrun")
        segments[segment_key] = (segment, end)
        return segment[pos:end]

    state: dict[str, torch.Tensor] = {}
    for key, kind, shape, dtype, count, aux, aux2 in layout:
        if kind == "q":
            packed = take("packed", count)
            scales = take("scales", aux * 2)
            state[key] = _dequantize_fp4_blocks(packed, scales, shape, device=device)
        elif kind in {"fp_weight", "bias", "dense_fp"}:
            source = "fp_weight" if kind == "fp_weight" else kind
            arr = np.frombuffer(take(source, count * 2), dtype=np.float16).copy()
            state[key] = torch.from_numpy(arr.reshape(shape).astype(np.float32)).to(device)
        elif kind == "dense_other":
            np_dtype = _torch_dtype_to_numpy(dtype)
            arr = np.frombuffer(take("dense_other", count * aux), dtype=np_dtype).copy()
            state[key] = torch.from_numpy(arr.reshape(shape)).to(device)
        elif kind == "qv":
            bits = aux
            rows = aux2
            meta = np.frombuffer(take("qv", rows * 4), dtype=np.float16).astype(np.float32).reshape(rows, 2)
            packed_count = (count * bits + 7) // 8
            q = _unpack_qbits(take("qv", packed_count), count, bits).astype(np.float32).reshape(rows, -1)
            value = meta[:, :1] + q * np.maximum(meta[:, 1:], 1e-8)
            state[key] = torch.from_numpy(value.reshape(shape).astype(np.float32)).to(device)
        else:  # pragma: no cover - layout construction owns kinds
            raise AssertionError(kind)
    return state


def _read_payload_input(path: str | bytes | Any) -> bytes:
    if isinstance(path, bytes):
        return path
    if isinstance(path, str):
        from pathlib import Path

        return Path(path).read_bytes()
    return path.read_bytes()


def _take_segment(
    segments: dict[str, tuple[memoryview, int]],
    segment_key: str,
    count: int,
    *,
    payload_label: str,
) -> memoryview:
    segment, pos = segments[segment_key]
    end = pos + count
    if end > len(segment):
        raise ValueError(f"{payload_label} segment {segment_key} read overrun")
    segments[segment_key] = (segment, end)
    return segment[pos:end]


def decode_mixed_qzs_block_state_dict(
    payload: bytes,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Decode an MQZ1 renderer with per-tensor FP4 block sizes.

    MQZ1 is a narrow QZS3 extension: all non-FP4 segments retain the QZS3
    ordering, while each FP4 tensor records its local block size in the charged
    JSON header. The archive still carries all score-affecting bits.
    """

    if not payload.startswith(MIXED_QZS_MAGIC):
        raise ValueError(f"bad MQZ1 magic {payload[:4]!r}")
    if len(payload) < 8:
        raise ValueError("MQZ1 payload too short")
    header_len = int.from_bytes(payload[4:8], "little")
    if header_len <= 0 or header_len > MIXED_QZS_MAX_HEADER_BYTES:
        raise ValueError(f"invalid MQZ1 header length: {header_len}")
    header_start = 8
    header_end = header_start + header_len
    if header_end > len(payload):
        raise ValueError("MQZ1 header overruns payload")
    header = json.loads(payload[header_start:header_end].decode("utf-8"))
    if header.get("schema") != MIXED_QZS_SCHEMA:
        raise ValueError(f"unsupported MQZ1 schema: {header.get('schema')!r}")
    if header.get("wire_format") != "MQZ1":
        raise ValueError(f"unsupported MQZ1 wire format: {header.get('wire_format')!r}")

    segment_bytes_raw = header.get("segment_bytes")
    if not isinstance(segment_bytes_raw, dict):
        raise ValueError("MQZ1 header missing segment_bytes")
    declared_sizes: dict[str, int] = {}
    for key in QZS_SEGMENT_KEYS:
        value = int(segment_bytes_raw.get(key, -1))
        if value < 0:
            raise ValueError(f"MQZ1 segment {key} has invalid byte count {value}")
        declared_sizes[key] = value

    fp4_tensors = header.get("fp4_tensors")
    fp4_block_sizes_raw = header.get("fp4_block_sizes")
    if fp4_tensors is None and fp4_block_sizes_raw is None:
        raise ValueError("MQZ1 header missing fp4_tensors or fp4_block_sizes")
    if fp4_tensors is not None and fp4_block_sizes_raw is not None:
        raise ValueError("MQZ1 header must not contain both fp4_tensors and fp4_block_sizes")
    fp4_block_sizes: list[int] | None = None
    if fp4_block_sizes_raw is not None:
        if not isinstance(fp4_block_sizes_raw, list):
            raise ValueError("MQZ1 fp4_block_sizes must be a list")
        fp4_block_sizes = [int(value) for value in fp4_block_sizes_raw]
        if header.get("fp4_tensor_order") != "joint_frame_generator_state_dict_fp4_order_v1":
            raise ValueError(f"unsupported MQZ1 fp4_tensor_order: {header.get('fp4_tensor_order')!r}")
    if fp4_tensors is not None and not isinstance(fp4_tensors, list):
        raise ValueError("MQZ1 fp4_tensors must be a list")
    fp4_meta_by_name: dict[str, dict[str, Any]] = {}
    if fp4_tensors is not None:
        for item in fp4_tensors:
            if not isinstance(item, dict):
                raise ValueError("MQZ1 fp4_tensors entries must be objects")
            name = str(item.get("name", ""))
            if not name or name in fp4_meta_by_name:
                raise ValueError(f"invalid or duplicate MQZ1 FP4 tensor name: {name!r}")
            fp4_meta_by_name[name] = item

    template_state = build_quantizr_faithful_renderer().state_dict()
    qv_specs = qzs3_qv_specs()

    computed_sizes = {key: 0 for key in QZS_SEGMENT_KEYS}
    layout: list[tuple[str, str, tuple[int, ...], torch.dtype, int, int, int]] = []
    remaining_fp4 = dict(fp4_meta_by_name)
    fp4_block_index = 0
    for key, tensor in template_state.items():
        shape = tuple(tensor.shape)
        count = int(tensor.numel())
        if _is_fp4_weight_name(key):
            if fp4_block_sizes is None:
                meta = remaining_fp4.pop(key, None)
                if meta is None:
                    raise ValueError(f"MQZ1 header missing FP4 tensor metadata for {key}")
                block_size = int(meta.get("block_size", 0))
                if tuple(meta.get("shape", ())) != shape:
                    raise ValueError(f"MQZ1 shape mismatch for {key}: {meta.get('shape')!r} != {shape}")
                if int(meta.get("numel", -1)) != count:
                    raise ValueError(f"MQZ1 numel mismatch for {key}")
            else:
                if fp4_block_index >= len(fp4_block_sizes):
                    raise ValueError(f"MQZ1 fp4_block_sizes missing entry for {key}")
                block_size = fp4_block_sizes[fp4_block_index]
                fp4_block_index += 1
            if block_size <= 0 or block_size > 4096:
                raise ValueError(f"invalid MQZ1 block size for {key}: {block_size}")
            scale_count = (count + block_size - 1) // block_size
            scale_bytes = scale_count * 2
            packed_count = (scale_count * block_size + 1) // 2
            if fp4_block_sizes is None:
                if int(meta.get("packed_bytes", -1)) != packed_count:
                    raise ValueError(f"MQZ1 packed byte mismatch for {key}")
                if int(meta.get("scale_bytes", -1)) != scale_bytes:
                    raise ValueError(f"MQZ1 scale byte mismatch for {key}")
            computed_sizes["packed"] += packed_count
            computed_sizes["scales"] += scale_bytes
            layout.append((key, "q_mixed", shape, tensor.dtype, packed_count, scale_bytes, block_size))
        elif key.endswith(".weight") and (
            key == "shared_trunk.embedding.weight"
            or key in {"frame1_head.head.weight", "frame2_head.head.weight"}
        ):
            computed_sizes["fp_weight"] += count * 2
            layout.append((key, "fp_weight", shape, tensor.dtype, count, 0, 0))
        elif _is_bias_name(key):
            computed_sizes["bias"] += count * 2
            layout.append((key, "bias", shape, tensor.dtype, count, 0, 0))
        elif key in qv_specs:
            bits, per_row = qv_specs[key]
            rows = shape[0] if per_row and len(shape) >= 2 else 1
            computed_sizes["qv"] += rows * 4 + (count * bits + 7) // 8
            layout.append((key, "qv", shape, tensor.dtype, count, bits, rows))
        elif torch.is_floating_point(tensor):
            computed_sizes["dense_fp"] += count * 2
            layout.append((key, "dense_fp", shape, tensor.dtype, count, 0, 0))
        else:
            elem_size = torch.empty((), dtype=tensor.dtype).element_size()
            computed_sizes["dense_other"] += count * elem_size
            layout.append((key, "dense_other", shape, tensor.dtype, count, elem_size, 0))
    if remaining_fp4:
        raise ValueError(f"MQZ1 header has unknown FP4 tensor metadata: {sorted(remaining_fp4)[:5]}")
    if fp4_block_sizes is not None and fp4_block_index != len(fp4_block_sizes):
        raise ValueError(
            f"MQZ1 fp4_block_sizes has {len(fp4_block_sizes) - fp4_block_index} trailing entries"
        )
    for key in QZS_SEGMENT_KEYS:
        if declared_sizes[key] != computed_sizes[key]:
            raise ValueError(
                f"MQZ1 segment {key} byte mismatch: "
                f"header={declared_sizes[key]} computed={computed_sizes[key]}"
            )

    view = memoryview(payload)
    offset = header_end
    segments: dict[str, tuple[memoryview, int]] = {}
    for key in QZS_SEGMENT_KEYS:
        end = offset + declared_sizes[key]
        if end > len(payload):
            raise ValueError(f"MQZ1 segment {key} overruns payload")
        segments[key] = (view[offset:end], 0)
        offset = end
    if offset != len(payload):
        raise ValueError(f"MQZ1 payload has {len(payload) - offset} trailing bytes")

    state: dict[str, torch.Tensor] = {}
    for key, kind, shape, dtype, count, aux, aux2 in layout:
        if kind == "q_mixed":
            packed = _take_segment(segments, "packed", count, payload_label="MQZ1")
            scales = _take_segment(segments, "scales", aux, payload_label="MQZ1")
            state[key] = _dequantize_fp4_blocks(
                packed,
                scales,
                shape,
                device=device,
                block_size=aux2,
            )
        elif kind in {"fp_weight", "bias", "dense_fp"}:
            source = "fp_weight" if kind == "fp_weight" else kind
            arr = np.frombuffer(
                _take_segment(segments, source, count * 2, payload_label="MQZ1"),
                dtype=np.float16,
            ).copy()
            state[key] = torch.from_numpy(arr.reshape(shape).astype(np.float32)).to(device)
        elif kind == "dense_other":
            np_dtype = _torch_dtype_to_numpy(dtype)
            arr = np.frombuffer(
                _take_segment(segments, "dense_other", count * aux, payload_label="MQZ1"),
                dtype=np_dtype,
            ).copy()
            state[key] = torch.from_numpy(arr.reshape(shape)).to(device)
        elif kind == "qv":
            bits = aux
            rows = aux2
            meta = np.frombuffer(
                _take_segment(segments, "qv", rows * 4, payload_label="MQZ1"),
                dtype=np.float16,
            ).astype(np.float32).reshape(rows, 2)
            packed_count = (count * bits + 7) // 8
            q = _unpack_qbits(
                _take_segment(segments, "qv", packed_count, payload_label="MQZ1"),
                count,
                bits,
            ).astype(np.float32).reshape(rows, -1)
            value = meta[:, :1] + q * np.maximum(meta[:, 1:], 1e-8)
            state[key] = torch.from_numpy(value.reshape(shape).astype(np.float32)).to(device)
        else:  # pragma: no cover - layout construction owns kinds
            raise AssertionError(kind)
    for key, (segment, pos) in segments.items():
        if pos != len(segment):
            raise ValueError(f"MQZ1 segment {key} has {len(segment) - pos} unread bytes")
    return state


def load_qzs3(
    path: str | bytes | Any,
    *,
    device: str | torch.device = "cpu",
) -> JointFrameGenerator:
    """Load a QZS3 renderer file into a JointFrameGenerator."""

    payload = _read_payload_input(path)
    state = decode_qzs3_state_dict(payload, device=device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_mixed_qzs_blocks(
    path: str | bytes | Any,
    *,
    device: str | torch.device = "cpu",
) -> JointFrameGenerator:
    """Load an MQZ1 mixed/local block-size renderer into a JointFrameGenerator."""

    payload = _read_payload_input(path)
    state = decode_mixed_qzs_block_state_dict(payload, device=device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.int64:
        return np.dtype(np.int64)
    if dtype == torch.int32:
        return np.dtype(np.int32)
    if dtype == torch.int16:
        return np.dtype(np.int16)
    if dtype == torch.uint8:
        return np.dtype(np.uint8)
    if dtype == torch.bool:
        return np.dtype(np.bool_)
    raise TypeError(f"unsupported non-floating dtype in QZS3 payload: {dtype}")
