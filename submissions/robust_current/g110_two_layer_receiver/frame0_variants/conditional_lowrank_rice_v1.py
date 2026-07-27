# SPDX-License-Identifier: MIT
"""Standalone conditional-Y0 decoder for the G110 two-layer packet."""

from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Final

import numpy as np

VARIANT_ID: Final = "tac.semantic_root_y0.conditional_lowrank_rice.v1"
MAGIC: Final = b"G110TL01"
VERSION: Final = 1
PAIR_COUNT: Final = 600
H: Final = 384
W: Final = 512
C: Final = 3
MAX_RANK: Final = 64
MAX_GRID_SIDE: Final = 64
MAX_PACKET_BYTES: Final = 2_100_000
MAX_SEMANTIC_PACKET_BYTES: Final = 2_000_000
CODEC_RICE_DELTA: Final = 0
FINAL_Y1_DOMAIN: Final = b"G110_FINAL_Y1_N600_V1\x00"

_HEADER = struct.Struct(">8sBBHHHBBHHBBIIII32sI")
_F32_BE = np.dtype(">f4")


class ConditionalVariantError(ValueError):
    """The counted conditional packet or render failed closed."""


def accepts_packet(packet: bytes) -> bool:
    return type(packet) is bytes and packet.startswith(MAGIC)


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.byte = 0
        self.used = 0

    def bit(self, value: int) -> None:
        self.byte = (self.byte << 1) | (value & 1)
        self.used += 1
        if self.used == 8:
            self.data.append(self.byte)
            self.byte = 0
            self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.data.append(self.byte << (8 - self.used))
        return bytes(self.data)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def bit(self) -> int:
        if self.offset >= len(self.payload) * 8:
            raise ConditionalVariantError("conditional Rice stream is truncated")
        value = (self.payload[self.offset // 8] >> (7 - self.offset % 8)) & 1
        self.offset += 1
        return value

    def zero_padding(self) -> None:
        while self.offset < len(self.payload) * 8:
            if self.bit():
                raise ConditionalVariantError("conditional Rice padding is nonzero")


def _unsigned_deltas(coefficients: np.ndarray) -> tuple[int, ...]:
    previous = np.zeros(coefficients.shape[1], dtype=np.int64)
    values: list[int] = []
    for row in coefficients.astype(np.int64):
        delta = row - previous
        previous = row
        values.extend(
            int(2 * value if value >= 0 else -2 * value - 1)
            for value in delta
        )
    return tuple(values)


def _rice_encode(coefficients: np.ndarray) -> tuple[int, bytes]:
    if coefficients.shape[1] == 0:
        return 0, b""
    unsigned = _unsigned_deltas(coefficients)
    rice_k = min(
        range(16),
        key=lambda k: (sum((value >> k) + 1 + k for value in unsigned), k),
    )
    writer = _BitWriter()
    mask = (1 << rice_k) - 1
    for value in unsigned:
        quotient = value >> rice_k
        for _ in range(quotient):
            writer.bit(1)
        writer.bit(0)
        remainder = value & mask
        for shift in range(rice_k - 1, -1, -1):
            writer.bit((remainder >> shift) & 1)
    return rice_k, writer.finish()


def _rice_decode(payload: bytes, *, rice_k: int, rank: int) -> np.ndarray:
    if rank == 0:
        if payload or rice_k:
            raise ConditionalVariantError("rank-zero stream must be empty Rice-0")
        return np.empty((PAIR_COUNT, 0), dtype=np.int16)
    if not 0 <= rice_k <= 15 or not payload:
        raise ConditionalVariantError("conditional Rice header is invalid")
    reader = _BitReader(payload)
    result = np.empty((PAIR_COUNT, rank), dtype=np.int16)
    previous = np.zeros(rank, dtype=np.int64)
    for pair_id in range(PAIR_COUNT):
        for column in range(rank):
            quotient = 0
            while reader.bit():
                quotient += 1
                if quotient > 262_143:
                    raise ConditionalVariantError("conditional Rice quotient exceeds bound")
            remainder = 0
            for _ in range(rice_k):
                remainder = (remainder << 1) | reader.bit()
            unsigned = (quotient << rice_k) | remainder
            delta = unsigned // 2 if not unsigned & 1 else -(unsigned // 2) - 1
            value = int(previous[column]) + delta
            if not -32_768 <= value <= 32_767:
                raise ConditionalVariantError("conditional delta leaves int16 range")
            result[pair_id, column] = value
            previous[column] = value
    reader.zero_padding()
    canonical_k, canonical = _rice_encode(result)
    if (canonical_k, canonical) != (rice_k, payload):
        raise ConditionalVariantError("conditional Rice stream is not canonical/minimal-k")
    return np.ascontiguousarray(result)


def parse_packet(packet: bytes) -> dict[str, object]:
    if type(packet) is not bytes or not _HEADER.size <= len(packet) <= MAX_PACKET_BYTES:
        raise ConditionalVariantError("two-layer packet must be bounded exact bytes")
    (
        magic,
        version,
        flags,
        pairs,
        height,
        width,
        channels,
        rank,
        grid_h,
        grid_w,
        codec,
        rice_k,
        semantic_length,
        basis_length,
        scale_length,
        coefficient_length,
        binding,
        expected_crc,
    ) = _HEADER.unpack_from(packet)
    if (
        magic != MAGIC
        or version != VERSION
        or flags != 0
        or (pairs, height, width, channels) != (PAIR_COUNT, H, W, C)
        or codec != CODEC_RICE_DELTA
        or not 0 <= rank <= MAX_RANK
        or not 0 < semantic_length <= MAX_SEMANTIC_PACKET_BYTES
    ):
        raise ConditionalVariantError("two-layer header changes the closed n600 ABI")
    if rank == 0:
        if (grid_h, grid_w, rice_k) != (0, 0, 0):
            raise ConditionalVariantError("rank-zero conditional header is noncanonical")
        expected_lengths = (semantic_length, 0, 0, 0)
    else:
        if not 1 <= grid_h <= MAX_GRID_SIDE or not 1 <= grid_w <= MAX_GRID_SIDE:
            raise ConditionalVariantError("conditional grid side is outside [1,64]")
        expected_lengths = (
            semantic_length,
            rank * grid_h * grid_w * C,
            rank * 4,
            coefficient_length,
        )
    lengths = (semantic_length, basis_length, scale_length, coefficient_length)
    if lengths != expected_lengths or _HEADER.size + sum(lengths) != len(packet):
        raise ConditionalVariantError("typed section lengths or exact EOF disagree")
    body = packet[_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != expected_crc:
        raise ConditionalVariantError("two-layer body CRC32 mismatch")
    cursor = 0
    sections = []
    for length in lengths:
        sections.append(body[cursor : cursor + length])
        cursor += length
    basis = np.frombuffer(sections[1], dtype=np.int8).reshape(rank, grid_h, grid_w, C).copy()
    scales = np.frombuffer(sections[2], dtype=_F32_BE).astype(np.float32)
    coefficients = _rice_decode(sections[3], rice_k=rice_k, rank=rank)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ConditionalVariantError("combined scales must be finite positive float32")
    if rank:
        dead_rank = np.all(basis == 0, axis=(1, 2, 3)) | np.all(coefficients == 0, axis=0)
        if np.any(dead_rank):
            raise ConditionalVariantError("unused conditional ranks were not removed")
    return {
        "semantic_packet": sections[0],
        "binding": bytes(binding),
        "basis": np.ascontiguousarray(basis),
        "scales": np.ascontiguousarray(scales),
        "coefficients": coefficients,
    }


def semantic_packet(state: dict[str, object]) -> bytes:
    packet = state.get("semantic_packet")
    if type(packet) is not bytes:
        raise ConditionalVariantError("parsed state lost semantic packet custody")
    return packet


def verify_final_y1_population(
    state: dict[str, object],
    population_digest: bytes,
) -> None:
    if type(population_digest) is not bytes or len(population_digest) != 32:
        raise ConditionalVariantError("final-Y1 population digest must be SHA-256 bytes")
    semantic = semantic_packet(state)
    observed = hashlib.sha256(
        FINAL_Y1_DOMAIN + hashlib.sha256(semantic).digest() + population_digest
    ).digest()
    binding = state.get("binding")
    if type(binding) is not bytes or observed != binding:
        raise ConditionalVariantError("rendered final-Y1 n600 population binding differs")


def _bilinear_resize(image: np.ndarray) -> np.ndarray:
    input_h, input_w, _ = image.shape
    ys = (np.arange(H, dtype=np.float64) + 0.5) * input_h / H - 0.5
    xs = (np.arange(W, dtype=np.float64) + 0.5) * input_w / W - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    y0 = np.clip(y0, 0, input_h - 1)
    x0 = np.clip(x0, 0, input_w - 1)
    y1 = np.clip(y0 + 1, 0, input_h - 1)
    x1 = np.clip(x0 + 1, 0, input_w - 1)
    top = image[y0[:, None], x0[None, :]] * (1.0 - wx[None, :, None])
    top += image[y0[:, None], x1[None, :]] * wx[None, :, None]
    bottom = image[y1[:, None], x0[None, :]] * (1.0 - wx[None, :, None])
    bottom += image[y1[:, None], x1[None, :]] * wx[None, :, None]
    return top * (1.0 - wy[:, None, None]) + bottom * wy[:, None, None]


def render_scorer_y0(
    state: dict[str, object],
    pair_id: int,
    scorer_y1: np.ndarray,
) -> np.ndarray:
    if type(pair_id) is not int or not 0 <= pair_id < PAIR_COUNT:
        raise ConditionalVariantError("pair_id is outside exact n600")
    y1 = np.asarray(scorer_y1)
    if y1.dtype != np.uint8 or y1.shape != (H, W, C):
        raise ConditionalVariantError("conditional Y0 requires uint8 scorer Y1")
    basis = state["basis"]
    scales = state["scales"]
    coefficients = state["coefficients"]
    if (
        type(basis) is not np.ndarray
        or type(scales) is not np.ndarray
        or type(coefficients) is not np.ndarray
    ):
        raise ConditionalVariantError("parsed conditional operands lost ndarray types")
    if basis.shape[0] == 0 or not np.any(coefficients[pair_id]):
        return np.ascontiguousarray(y1)
    weights = coefficients[pair_id].astype(np.float32) * scales
    grid = np.einsum(
        "r,rhwc->hwc",
        weights,
        basis.astype(np.float32),
        optimize=True,
        dtype=np.float32,
    )
    residual = _bilinear_resize(np.ascontiguousarray(grid, dtype=np.float32))
    return np.ascontiguousarray(
        np.clip(np.rint(y1.astype(np.float32) + residual), 0, 255).astype(np.uint8)
    )
