#!/usr/bin/env python3
"""Fail-closed receiver for the counted DDM-SM4 low-rank grid format.

``SM4R`` stores the five fixed PR130 pointwise matrices as SVD factors.  The
rank, factor precision, and optional row-centering flag are counted header
fields; factor scales, codes, and optional row means all remain in the counted
semantic payload.  Exact magic absence returns ``None`` so the caller can use
an older receiver.  A present but malformed field always raises.
"""

from __future__ import annotations

import struct
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch

MAGIC = b"SM4R"
VERSION = 1
FLAG_CENTERED = 1
LOWRANK_NAMES = frozenset(
    {
        "coord_mix.weight",
        "blocks.0.pw.weight",
        "blocks.1.pw.weight",
        "blocks.2.pw.weight",
        "blocks.3.pw.weight",
    }
)


class SM4RFormatError(ValueError):
    """The tagged semantic field cannot be decoded without guessing."""


def _require(remaining: memoryview, count: int, label: str) -> memoryview:
    if count < 0 or len(remaining) < count:
        raise SM4RFormatError(f"truncated {label}")
    return remaining[:count]


def _unpack_signed_bits(
    blob: memoryview,
    count: int,
    bits: int,
) -> tuple[torch.Tensor, memoryview]:
    if count < 0 or not 2 <= bits <= 8:
        raise SM4RFormatError("invalid signed bit-stream geometry")
    byte_count = (count * bits + 7) // 8
    packed = np.frombuffer(
        _require(blob, byte_count, "signed code stream"),
        dtype=np.uint8,
    )
    stream = np.unpackbits(packed, bitorder="little")[: count * bits]
    stream = stream.reshape(count, bits).astype(np.int16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.int16))[None]
    unsigned = (stream * shifts).sum(axis=1, dtype=np.int16)
    sign = 1 << (bits - 1)
    values = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    return torch.from_numpy(values.astype(np.int8, copy=False)), blob[byte_count:]


def _scale_count(name: str, value: torch.Tensor) -> int:
    return int(value.shape[-1] if name.endswith("embed.weight") else value.shape[0])


def _decode_quantized(
    name: str,
    template: torch.Tensor,
    blob: memoryview,
    bits: int,
) -> tuple[torch.Tensor, memoryview]:
    count = _scale_count(name, template)
    scale_bytes = count * np.dtype("<f2").itemsize
    scales = np.frombuffer(
        _require(blob, scale_bytes, f"fp16 scales for {name}"),
        dtype="<f2",
    ).copy()
    remaining = blob[scale_bytes:]
    codes, remaining = _unpack_signed_bits(remaining, template.numel(), bits)
    scale_shape = [1] * template.ndim
    scale_shape[-1 if name.endswith("embed.weight") else 0] = count
    restored = codes.reshape(template.shape).float()
    restored *= torch.from_numpy(scales).float().reshape(scale_shape)
    return restored, remaining


def _quantized_names(template: Mapping[str, torch.Tensor]) -> list[str]:
    return [name for name, value in template.items() if value.ndim >= 2]


def _selection_mask(names: list[str]) -> int:
    if not LOWRANK_NAMES.issubset(names):
        raise SM4RFormatError("low-rank tensor set differs from the receiver template")
    return sum(1 << index for index, name in enumerate(names) if name in LOWRANK_NAMES)


def unpack_sm4r_or_none(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor] | None:
    """Decode one ``SM4R`` field, or return ``None`` only when magic is absent."""

    if not blob.startswith(MAGIC):
        return None
    if len(blob) < 10:
        raise SM4RFormatError("truncated SM4R header")
    version, rank, bits, flags = blob[4:8]
    if version != VERSION:
        raise SM4RFormatError(f"unsupported SM4R version {version}")
    if not 1 <= rank < 96 or not 4 <= bits <= 8:
        raise SM4RFormatError("invalid SM4R rank or factor precision")
    if flags & ~FLAG_CENTERED:
        raise SM4RFormatError("unknown SM4R flags")
    centered = bool(flags & FLAG_CENTERED)
    remaining = memoryview(blob)[8:]
    mask = struct.unpack_from("<H", _require(remaining, 2, "selection mask"))[0]
    remaining = remaining[2:]
    if mask != _selection_mask(_quantized_names(template)):
        raise SM4RFormatError("SM4R low-rank selection mask differs")

    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        if value.ndim < 2:
            byte_count = value.numel() * np.dtype("<f2").itemsize
            array = np.frombuffer(
                _require(remaining, byte_count, f"fp16 tensor {name}"),
                dtype="<f2",
            ).copy()
            remaining = remaining[byte_count:]
            restored[name] = torch.from_numpy(array.reshape(value.shape)).float()
        elif name not in LOWRANK_NAMES:
            restored[name], remaining = _decode_quantized(name, value, remaining, 4)
        else:
            rows = int(value.shape[0])
            columns = value.numel() // rows
            if rank >= min(rows, columns):
                raise SM4RFormatError(f"SM4R rank does not reduce tensor {name}")
            row_mean = None
            if centered:
                mean_bytes = rows * np.dtype("<f2").itemsize
                row_mean = np.frombuffer(
                    _require(remaining, mean_bytes, f"row means for {name}"),
                    dtype="<f2",
                ).copy()
                remaining = remaining[mean_bytes:]
            left_template = torch.empty((rows, rank), dtype=torch.float32)
            right_template = torch.empty((rank, columns), dtype=torch.float32)
            left, remaining = _decode_quantized("factor.left", left_template, remaining, bits)
            right, remaining = _decode_quantized("factor.right", right_template, remaining, bits)
            matrix = left @ right
            if row_mean is not None:
                matrix += torch.from_numpy(row_mean).float()[:, None]
            restored[name] = matrix.reshape(value.shape)

    if remaining:
        raise SM4RFormatError(f"SM4R payload has {len(remaining)} trailing bytes")
    return restored
