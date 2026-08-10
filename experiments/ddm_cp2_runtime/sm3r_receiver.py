#!/usr/bin/env python3
"""Fail-closed receiver for the counted DDM-SM3 semantic formats.

The dispatch has exactly three cases:

* no ``SM3R`` field: the caller must use the unchanged legacy loader;
* a known version-1 VQ or low-rank field: decode it here;
* any other ``SM3R`` field: refuse it without guessing.

Only generic receiver code lives here.  Every codebook, index stream, factor,
scale, and quantized value remains counted inside ``archive.zip``.
"""

from __future__ import annotations

import math
import struct
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch

MAGIC = b"SM3R"
VERSION = 1
MODE_VECTOR_VQ = 1
MODE_SCALE_VQ = 2
MODE_BOTH_VQ = 3
MODE_LOWRANK = 4
SUPPORTED_MODES = frozenset({MODE_VECTOR_VQ, MODE_SCALE_VQ, MODE_BOTH_VQ, MODE_LOWRANK})

LOWRANK_NAMES = frozenset(
    {
        "coord_mix.weight",
        "blocks.0.pw.weight",
        "blocks.1.pw.weight",
        "blocks.2.pw.weight",
        "blocks.3.pw.weight",
    }
)


class SM3RFormatError(ValueError):
    """The tagged semantic field cannot be decoded without guessing."""


def _require(remaining: memoryview, count: int, label: str) -> memoryview:
    if count < 0 or len(remaining) < count:
        raise SM3RFormatError(f"truncated {label}")
    return remaining[:count]


def _unpack_unsigned_bits(
    blob: memoryview,
    count: int,
    bits: int,
) -> tuple[np.ndarray, memoryview]:
    if count < 0 or not 1 <= bits <= 8:
        raise SM3RFormatError("invalid unsigned bit-stream geometry")
    byte_count = (count * bits + 7) // 8
    packed_view = _require(blob, byte_count, "unsigned bit stream")
    packed = np.frombuffer(packed_view, dtype=np.uint8)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits]
    if count:
        stream = stream.reshape(count, bits).astype(np.uint16, copy=False)
        shifts = (1 << np.arange(bits, dtype=np.uint16))[None]
        values = (stream * shifts).sum(axis=1, dtype=np.uint16)
    else:
        values = np.empty(0, dtype=np.uint16)
    return values, blob[byte_count:]


def _unpack_signed_bits(
    blob: memoryview,
    count: int,
    bits: int,
) -> tuple[torch.Tensor, memoryview]:
    if count < 0 or not 2 <= bits <= 8:
        raise SM3RFormatError("invalid signed bit-stream geometry")
    byte_count = (count * bits + 7) // 8
    packed_view = _require(blob, byte_count, "signed code stream")
    packed = np.frombuffer(packed_view, dtype=np.uint8)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits]
    stream = stream.reshape(count, bits).astype(np.int16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.int16))[None]
    unsigned = (stream * shifts).sum(axis=1, dtype=np.int16)
    sign = 1 << (bits - 1)
    values = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    return (
        torch.from_numpy(values.astype(np.int8, copy=False)),
        blob[byte_count:],
    )


def _scale_count(name: str, value: torch.Tensor) -> int:
    return int(value.shape[-1] if name.endswith("embed.weight") else value.shape[0])


def _decode_standard_q4(
    name: str,
    template: torch.Tensor,
    blob: memoryview,
) -> tuple[torch.Tensor, memoryview]:
    count = _scale_count(name, template)
    scale_bytes = count * np.dtype("<f2").itemsize
    scales = np.frombuffer(
        _require(blob, scale_bytes, f"q4 scales for {name}"),
        dtype="<f2",
    ).copy()
    remaining = blob[scale_bytes:]
    codes, remaining = _unpack_signed_bits(remaining, template.numel(), 4)
    scale_shape = [1] * template.ndim
    scale_shape[-1 if name.endswith("embed.weight") else 0] = count
    restored = codes.reshape(template.shape).float()
    restored *= torch.from_numpy(scales).float().reshape(scale_shape)
    return restored, remaining


def _quantized_names(template: Mapping[str, torch.Tensor]) -> list[str]:
    return [name for name, value in template.items() if value.ndim >= 2]


def _selection_mask(names: list[str], selected: frozenset[str]) -> int:
    if not selected.issubset(names):
        raise SM3RFormatError("low-rank tensor set differs from the receiver template")
    return sum(1 << index for index, name in enumerate(names) if name in selected)


def _decode_vq(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    _, mode, codebook_size, reserved = blob[4:8]
    if mode not in {MODE_VECTOR_VQ, MODE_SCALE_VQ, MODE_BOTH_VQ}:
        raise SM3RFormatError("unsupported SM3R VQ mode")
    if reserved != 0:
        raise SM3RFormatError("SM3R VQ reserved byte is nonzero")
    if not 2 <= codebook_size <= 256 or codebook_size & (codebook_size - 1):
        raise SM3RFormatError("SM3R VQ codebook size is not a supported power of two")
    bits = int(math.log2(codebook_size))
    remaining = memoryview(blob)[8:]
    vector_names = [name for name, value in template.items() if value.ndim < 2]
    quantized_names = _quantized_names(template)
    vector_count = sum(template[name].numel() for name in vector_names)
    scale_counts = {name: _scale_count(name, template[name]) for name in quantized_names}
    scale_count = sum(scale_counts.values())

    vector_codebook = vector_indices = None
    scale_codebook = scale_indices = None
    codebook_bytes = codebook_size * np.dtype("<f2").itemsize
    if mode in {MODE_VECTOR_VQ, MODE_BOTH_VQ}:
        vector_codebook = np.frombuffer(
            _require(remaining, codebook_bytes, "vector VQ codebook"),
            dtype="<f2",
        ).copy()
        remaining = remaining[codebook_bytes:]
        vector_indices, remaining = _unpack_unsigned_bits(
            remaining,
            vector_count,
            bits,
        )
    if mode in {MODE_SCALE_VQ, MODE_BOTH_VQ}:
        scale_codebook = np.frombuffer(
            _require(remaining, codebook_bytes, "scale VQ codebook"),
            dtype="<f2",
        ).copy()
        remaining = remaining[codebook_bytes:]
        scale_indices, remaining = _unpack_unsigned_bits(
            remaining,
            scale_count,
            bits,
        )

    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    vector_offset = 0
    scale_offset = 0
    for name, value in template.items():
        if value.ndim < 2:
            count = value.numel()
            if vector_codebook is None or vector_indices is None:
                byte_count = count * np.dtype("<f2").itemsize
                array = np.frombuffer(
                    _require(remaining, byte_count, f"fp16 tensor {name}"),
                    dtype="<f2",
                ).copy()
                remaining = remaining[byte_count:]
            else:
                selected = vector_indices[vector_offset : vector_offset + count]
                if len(selected) != count:
                    raise SM3RFormatError(f"vector VQ indices end inside {name}")
                array = vector_codebook[selected].copy()
                vector_offset += count
            restored[name] = torch.from_numpy(array.reshape(value.shape)).float()
            continue

        count = scale_counts[name]
        if scale_codebook is None or scale_indices is None:
            byte_count = count * np.dtype("<f2").itemsize
            scales = np.frombuffer(
                _require(remaining, byte_count, f"fp16 scales for {name}"),
                dtype="<f2",
            ).copy()
            remaining = remaining[byte_count:]
        else:
            selected = scale_indices[scale_offset : scale_offset + count]
            if len(selected) != count:
                raise SM3RFormatError(f"scale VQ indices end inside {name}")
            scales = scale_codebook[selected].copy()
            scale_offset += count
        codes, remaining = _unpack_signed_bits(remaining, value.numel(), 4)
        scale_shape = [1] * value.ndim
        scale_shape[-1 if name.endswith("embed.weight") else 0] = count
        restored[name] = codes.reshape(value.shape).float()
        restored[name] *= torch.from_numpy(scales).float().reshape(scale_shape)

    if remaining:
        raise SM3RFormatError(f"SM3R VQ payload has {len(remaining)} trailing bytes")
    return restored


def _decode_lowrank(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    _, mode, rank, reserved = blob[4:8]
    if mode != MODE_LOWRANK:
        raise SM3RFormatError("unsupported SM3R low-rank mode")
    if reserved != 0 or rank == 0:
        raise SM3RFormatError("invalid SM3R low-rank header")
    remaining = memoryview(blob)[8:]
    mask_view = _require(remaining, 2, "low-rank selection mask")
    mask = struct.unpack_from("<H", mask_view)[0]
    remaining = remaining[2:]
    names = _quantized_names(template)
    if mask != _selection_mask(names, LOWRANK_NAMES):
        raise SM3RFormatError("SM3R low-rank selection mask differs")

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
            restored[name], remaining = _decode_standard_q4(name, value, remaining)
        else:
            rows = int(value.shape[0])
            columns = value.numel() // rows
            if rank >= min(rows, columns):
                raise SM3RFormatError(f"SM3R rank does not reduce tensor {name}")
            left_template = torch.empty((rows, rank), dtype=torch.float32)
            right_template = torch.empty((rank, columns), dtype=torch.float32)
            left, remaining = _decode_standard_q4(
                "factor.left",
                left_template,
                remaining,
            )
            right, remaining = _decode_standard_q4(
                "factor.right",
                right_template,
                remaining,
            )
            restored[name] = (left @ right).reshape(value.shape)

    if remaining:
        raise SM3RFormatError(f"SM3R low-rank payload has {len(remaining)} trailing bytes")
    return restored


def unpack_sm3r_or_none(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor] | None:
    """Decode one known ``SM3R`` field, or return ``None`` when absent.

    Returning ``None`` is reserved exclusively for exact magic absence.  A
    present-but-malformed, unknown-version, or unknown-mode field always raises.
    """

    if not blob.startswith(MAGIC):
        return None
    if len(blob) < 8:
        raise SM3RFormatError("truncated SM3R header")
    version, mode = blob[4], blob[5]
    if version != VERSION:
        raise SM3RFormatError(f"unsupported SM3R version {version}")
    if mode not in SUPPORTED_MODES:
        raise SM3RFormatError(f"unsupported SM3R mode {mode}")
    if mode == MODE_LOWRANK:
        return _decode_lowrank(blob, template)
    return _decode_vq(blob, template)
