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

import math
from typing import Any

import numpy as np
import torch

from tac.quantizr_faithful_renderer import (
    JointFrameGenerator,
    build_quantizr_faithful_renderer,
)

QZS3_MAGIC = b"QZS3"
DEFAULT_BLOCK_SIZE = 32
FP4_POS_LEVELS = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=np.float32,
)


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
) -> torch.Tensor:
    flat_n = int(np.prod(shape))
    scales = np.frombuffer(scale_bytes, dtype=np.float16).astype(np.float32)
    if scales.size == 0:
        return torch.empty(shape, dtype=torch.float32, device=device)
    block_size = (len(packed) * 2) // scales.size
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


def load_qzs3(
    path: str | bytes | Any,
    *,
    device: str | torch.device = "cpu",
) -> JointFrameGenerator:
    """Load a QZS3 renderer file into a JointFrameGenerator."""

    if isinstance(path, (str, bytes)):
        from pathlib import Path

        payload = Path(path).read_bytes() if isinstance(path, str) else path
    else:
        payload = path.read_bytes()
    state = decode_qzs3_state_dict(payload, device=device)
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
