#!/usr/bin/env python3
"""Counted categorical Cool-Chic decoder: packet grammar and integer receiver.

GDC1 retained a K=8 ordered-scanline teacher render of the move-43 token field
and wrote the governed spec for the surviving construction: distil that teacher
into a *counted* coordinate decoder whose whole source object is one packet.

This module owns the receiver side of that construction and nothing else.  It
holds the fixed architecture, the deterministic training tile schedule, the
packet grammar, and a NumPy integer receiver that is the verdict authority.  The
MLX trainer is research signal only and lives in the experiment driver.

Every number the receiver consumes is inside the packet: both int8 latent grids,
every int8 convolution weight, every int32 bias, and every per-layer requant
shift.  The receiver is given no target field, no teacher render, no previous
token plane and no scorer output.  The interpolation lattice, the convolution
algebra, the requant arithmetic and the argmax are generic decoder code.

Arithmetic contract (this is what makes MLX/NumPy parity checkable):
every intermediate value the receiver computes is an integer whose magnitude
stays below 2**24, so float32 and int32 evaluation of the same expression agree
exactly and the argmax is order independent.
"""

from __future__ import annotations

import hashlib
import struct
import warnings
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
CLASSES: Final = 5
SEED: Final = 20260910

# Fixed architecture (GDC1 governed spec, verbatim).
Z0_SHAPE: Final = (75, 24, 32, 4)
Z1_SHAPE: Final = (150, 12, 16, 2)
FEATURES: Final = Z0_SHAPE[3] + Z1_SHAPE[3]
WIDTH_CH: Final = 24
BLOCKS: Final = 5
KERNEL: Final = 3

# Fixed-point contract.  Both latent grids are resampled into a common Q16
# lattice whose weights are exact dyadic rationals because every upsampling
# factor is a power of two, then requantised once into the stem input domain.
FEATURE_Q: Final = 16
INPUT_SHIFT: Final = 14          # stem input keeps 2 fractional latent bits
SHIFT_STEM: Final = 5
SHIFT_DW: Final = 6
SHIFT_PW: Final = 7
ACT_MAX: Final = 127
WEIGHT_MAX: Final = 127
LATENT_MAX: Final = 127
HEAD_LOSS_DIVISOR: Final = 512.0  # fixed tau for the training surrogate

PACKET_MAGIC: Final = b"GDC2CC1!"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct("<8sBBH")
PACKET_SECTION: Final = struct.Struct("<BBII")
SECTION_NAMES: Final = ("z0", "z1", "params")
CODERS: Final = ("brotli_q11", "zlib_9", "lzma2_extreme")
CODER_IDS: Final = {name: index + 1 for index, name in enumerate(CODERS)}
ID_CODERS: Final = {value: key for key, value in CODER_IDS.items()}


class GDC2Error(RuntimeError):
    """A GDC2 packet, receiver, or arithmetic invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# deterministic resampling lattice
# --------------------------------------------------------------------------


def axis_lattice(source: int, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Half-pixel-centred linear resampling lattice with exact dyadic weights.

    Returns ``(index0, index1, weight1, denominator)`` where the companion
    weight is ``denominator - weight1``.  ``target`` must be an integral
    power-of-two multiple of ``source`` so the weights are exact.
    """
    if target % source:
        raise GDC2Error(f"non-integral resample {source} -> {target}")
    scale = target // source
    if scale & (scale - 1):
        raise GDC2Error(f"resample factor {scale} is not a power of two")
    denominator = 2 * scale
    destination = np.arange(target, dtype=np.int64)
    numerator = 2 * destination + 1 - scale
    index0 = np.floor_divide(numerator, denominator)
    fraction = numerator - index0 * denominator
    index1 = index0 + 1
    np.clip(index0, 0, source - 1, out=index0)
    np.clip(index1, 0, source - 1, out=index1)
    return index0, index1, fraction, denominator


@dataclass(frozen=True)
class Lattice:
    """The full resampling lattice for one latent grid."""

    t0: np.ndarray
    t1: np.ndarray
    tw: np.ndarray
    td: int
    y0: np.ndarray
    y1: np.ndarray
    yw: np.ndarray
    yd: int
    x0: np.ndarray
    x1: np.ndarray
    xw: np.ndarray
    xd: int

    @property
    def denominator(self) -> int:
        return self.td * self.yd * self.xd


def build_lattice(shape: tuple[int, int, int, int]) -> Lattice:
    t0, t1, tw, td = axis_lattice(shape[0], N_PAIRS)
    y0, y1, yw, yd = axis_lattice(shape[1], HEIGHT)
    x0, x1, xw, xd = axis_lattice(shape[2], WIDTH)
    return Lattice(t0, t1, tw, td, y0, y1, yw, yd, x0, x1, xw, xd)


LATTICE_Z0: Final = build_lattice(Z0_SHAPE)
LATTICE_Z1: Final = build_lattice(Z1_SHAPE)
if (1 << FEATURE_Q) % LATTICE_Z0.denominator or (1 << FEATURE_Q) % LATTICE_Z1.denominator:
    raise GDC2Error("latent lattices do not share the Q16 feature domain")
Z0_GAIN: Final = (1 << FEATURE_Q) // LATTICE_Z0.denominator
Z1_GAIN: Final = (1 << FEATURE_Q) // LATTICE_Z1.denominator


def resample_window(
    latent: np.ndarray,
    lattice: Lattice,
    gain: int,
    pairs: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Exact integer trilinear resample of one latent grid onto a window.

    ``latent`` is ``(T, H, W, C)``; the result is ``(len(pairs), len(rows),
    len(columns), C)`` in the shared Q16 feature domain.
    """
    work = latent.astype(np.int64, copy=False)
    a = work[lattice.t0[pairs]]
    b = work[lattice.t1[pairs]]
    w1 = lattice.tw[pairs].reshape(-1, 1, 1, 1)
    stage = a * (lattice.td - w1) + b * w1
    a = stage[:, lattice.y0[rows]]
    b = stage[:, lattice.y1[rows]]
    w1 = lattice.yw[rows].reshape(1, -1, 1, 1)
    stage = a * (lattice.yd - w1) + b * w1
    a = stage[:, :, lattice.x0[columns]]
    b = stage[:, :, lattice.x1[columns]]
    w1 = lattice.xw[columns].reshape(1, 1, -1, 1)
    stage = a * (lattice.xd - w1) + b * w1
    return stage * gain


def round_shift(value: np.ndarray, shift: int) -> np.ndarray:
    """Round-half-up arithmetic right shift; identical to floor((v+h)/2**s)."""
    if shift <= 0:
        return value
    return (value + (1 << (shift - 1))) >> shift


def feature_window(
    z0: np.ndarray,
    z1: np.ndarray,
    pairs: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Stem input for a window: ``(len(pairs), len(rows), len(columns), 6)``."""
    part0 = resample_window(z0, LATTICE_Z0, Z0_GAIN, pairs, rows, columns)
    part1 = resample_window(z1, LATTICE_Z1, Z1_GAIN, pairs, rows, columns)
    stacked = np.concatenate((part0, part1), axis=3)
    return round_shift(stacked, INPUT_SHIFT).astype(np.int64)


# --------------------------------------------------------------------------
# counted parameters
# --------------------------------------------------------------------------


@dataclass
class Parameters:
    """Every counted decoder parameter except the two latent grids."""

    stem_w: np.ndarray          # int8 (WIDTH_CH, FEATURES)
    stem_b: np.ndarray          # int32 (WIDTH_CH,)
    dw_w: np.ndarray            # int8 (BLOCKS, WIDTH_CH, KERNEL, KERNEL)
    dw_b: np.ndarray            # int32 (BLOCKS, WIDTH_CH)
    pw_w: np.ndarray            # int8 (BLOCKS, WIDTH_CH, WIDTH_CH)
    pw_b: np.ndarray            # int32 (BLOCKS, WIDTH_CH)
    head_w: np.ndarray          # int8 (CLASSES, WIDTH_CH)
    head_b: np.ndarray          # int32 (CLASSES,)
    shifts: np.ndarray          # uint8 (3,) stem, depthwise, pointwise

    def to_bytes(self) -> bytes:
        blobs = [
            self.stem_w.astype(np.int8).tobytes(),
            self.stem_b.astype("<i4").tobytes(),
            self.dw_w.astype(np.int8).tobytes(),
            self.dw_b.astype("<i4").tobytes(),
            self.pw_w.astype(np.int8).tobytes(),
            self.pw_b.astype("<i4").tobytes(),
            self.head_w.astype(np.int8).tobytes(),
            self.head_b.astype("<i4").tobytes(),
            self.shifts.astype(np.uint8).tobytes(),
        ]
        return b"".join(blobs)

    @staticmethod
    def from_bytes(payload: bytes) -> Parameters:
        cursor = 0

        def take(count: int, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal cursor
            width = count * np.dtype(dtype).itemsize
            block = np.frombuffer(payload, dtype=dtype, count=count, offset=cursor)
            cursor += width
            return block.reshape(shape).astype(np.int64)

        stem_w = take(WIDTH_CH * FEATURES, np.int8, (WIDTH_CH, FEATURES))
        stem_b = take(WIDTH_CH, "<i4", (WIDTH_CH,))
        dw_w = take(BLOCKS * WIDTH_CH * KERNEL * KERNEL, np.int8,
                    (BLOCKS, WIDTH_CH, KERNEL, KERNEL))
        dw_b = take(BLOCKS * WIDTH_CH, "<i4", (BLOCKS, WIDTH_CH))
        pw_w = take(BLOCKS * WIDTH_CH * WIDTH_CH, np.int8, (BLOCKS, WIDTH_CH, WIDTH_CH))
        pw_b = take(BLOCKS * WIDTH_CH, "<i4", (BLOCKS, WIDTH_CH))
        head_w = take(CLASSES * WIDTH_CH, np.int8, (CLASSES, WIDTH_CH))
        head_b = take(CLASSES, "<i4", (CLASSES,))
        shifts = take(3, np.uint8, (3,))
        if cursor != len(payload):
            raise GDC2Error("parameter section length mismatch")
        return Parameters(stem_w, stem_b, dw_w, dw_b, pw_w, pw_b, head_w, head_b, shifts)

    def raw_bytes(self) -> int:
        return len(self.to_bytes())


def default_shifts() -> np.ndarray:
    return np.array([SHIFT_STEM, SHIFT_DW, SHIFT_PW], dtype=np.uint8)


# --------------------------------------------------------------------------
# integer receiver
# --------------------------------------------------------------------------


EXACT_FLOAT32_LIMIT: Final = 1 << 24


def _matmul_exact(flat: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Integer matmul carried in float32, guarded to stay bit-exact.

    Every operand and every partial sum is an integer of magnitude below 2**24,
    so float32 BLAS reproduces the int64 product exactly and is order
    independent.  The guard is a real check, not a comment: it recomputes in
    int64 whenever the derived bound could be violated.
    """
    bound = int(np.abs(flat).max(initial=0)) * int(np.abs(kernel).max(initial=0)) * kernel.shape[0]
    if bound >= EXACT_FLOAT32_LIMIT:
        return flat.astype(np.int64) @ kernel.astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        product = flat.astype(np.float32) @ kernel.astype(np.float32)
    return product.astype(np.int64)


def _pointwise(activation: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               shift: int, relu: bool) -> np.ndarray:
    """``activation`` is (H, W, Cin) int64; weight is (Cout, Cin)."""
    height, width, _ = activation.shape
    flat = activation.reshape(height * width, -1)
    kernel = weight.T
    accumulated = _matmul_exact(flat, kernel)
    out = accumulated.reshape(height, width, -1) + bias.reshape(1, 1, -1)
    if shift:
        out = round_shift(out, shift)
    if relu:
        np.clip(out, 0, ACT_MAX, out=out)
    return out


def _depthwise(activation: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               shift: int) -> np.ndarray:
    """3x3 depthwise convolution with zero padding; (H, W, C) int64 in and out."""
    height, width, channels = activation.shape
    padded = np.zeros((height + 2, width + 2, channels), dtype=np.int64)
    padded[1:-1, 1:-1] = activation
    out = np.zeros((height, width, channels), dtype=np.int64)
    for row in range(KERNEL):
        for column in range(KERNEL):
            taps = weight[:, row, column].reshape(1, 1, channels)
            out += padded[row:row + height, column:column + width] * taps
    out += bias.reshape(1, 1, channels)
    out = round_shift(out, shift)
    np.clip(out, 0, ACT_MAX, out=out)
    return out


def receiver_logits(features: np.ndarray, params: Parameters) -> np.ndarray:
    """Integer logits for one frame: (H, W, 6) int64 in, (H, W, 5) int64 out."""
    shift_stem, shift_dw, shift_pw = (int(value) for value in params.shifts)
    activation = _pointwise(features, params.stem_w, params.stem_b, shift_stem, True)
    for block in range(BLOCKS):
        activation = _depthwise(activation, params.dw_w[block], params.dw_b[block], shift_dw)
        activation = _pointwise(activation, params.pw_w[block], params.pw_b[block],
                                shift_pw, True)
    return _pointwise(activation, params.head_w, params.head_b, 0, False)


def receiver_frame(z0: np.ndarray, z1: np.ndarray, params: Parameters, pair: int) -> np.ndarray:
    rows = np.arange(HEIGHT)
    columns = np.arange(WIDTH)
    features = feature_window(z0, z1, np.array([pair]), rows, columns)[0]
    logits = receiver_logits(features, params)
    return np.argmax(logits, axis=2).astype(np.uint8)


def receiver_render(z0: np.ndarray, z1: np.ndarray, params: Parameters,
                    output: np.ndarray) -> None:
    """Render every pair into ``output`` (N_PAIRS, HEIGHT, WIDTH) uint8."""
    if output.shape != (N_PAIRS, HEIGHT, WIDTH):
        raise GDC2Error("render target has the wrong shape")
    for pair in range(N_PAIRS):
        output[pair] = receiver_frame(z0, z1, params, pair)


# --------------------------------------------------------------------------
# packet grammar
# --------------------------------------------------------------------------


def compress_payload(raw: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        import brotli

        return brotli.compress(raw, quality=11)
    if coder == "zlib_9":
        import zlib

        return zlib.compress(raw, 9)
    if coder == "lzma2_extreme":
        import lzma

        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
        return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filters)
    raise GDC2Error(f"unknown coder: {coder}")


def decompress_payload(coded: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        import brotli

        return brotli.decompress(coded)
    if coder == "zlib_9":
        import zlib

        return zlib.decompress(coded)
    if coder == "lzma2_extreme":
        import lzma

        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
        return lzma.decompress(coded, format=lzma.FORMAT_RAW, filters=filters)
    raise GDC2Error(f"unknown coder: {coder}")


def race_section(raw: bytes) -> dict[str, Any]:
    """Race the three physical coders and prove a deterministic exact repeat."""
    rows: dict[str, Any] = {}
    for coder in CODERS:
        coded = compress_payload(raw, coder)
        repeated = compress_payload(raw, coder)
        if coded != repeated:
            raise GDC2Error(f"{coder} is not deterministic on this payload")
        if decompress_payload(coded, coder) != raw:
            raise GDC2Error(f"{coder} did not parse back exactly")
        rows[coder] = {
            "coder": coder,
            "bytes": len(coded),
            "sha256": sha256_bytes(coded),
            "deterministic_repeat_equal": True,
            "raw_parseback_equal": True,
        }
    winner = min(CODERS, key=lambda name: (rows[name]["bytes"], CODERS.index(name)))
    return {"coders": rows, "winner": winner, "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw)}


def build_packet(z0: np.ndarray, z1: np.ndarray, params: Parameters) -> tuple[bytes, dict[str, Any]]:
    """The packet is the complete source object; every stored byte is counted."""
    sections = {
        "z0": z0.astype(np.int8).tobytes(),
        "z1": z1.astype(np.int8).tobytes(),
        "params": params.to_bytes(),
    }
    races = {name: race_section(sections[name]) for name in SECTION_NAMES}
    rows = bytearray()
    bodies = bytearray()
    for name in SECTION_NAMES:
        winner = races[name]["winner"]
        coded = compress_payload(sections[name], winner)
        rows.extend(PACKET_SECTION.pack(CODER_IDS[winner], 0, len(sections[name]), len(coded)))
        bodies.extend(coded)
    packet = PACKET_HEADER.pack(PACKET_MAGIC, PACKET_VERSION, len(SECTION_NAMES), 0)
    packet = packet + bytes(rows) + bytes(bodies)
    facts = {
        "sections": races,
        "raw_total_bytes": sum(len(sections[name]) for name in SECTION_NAMES),
        "header_overhead_bytes": PACKET_HEADER.size + PACKET_SECTION.size * len(SECTION_NAMES),
        "packet_bytes": len(packet),
        "packet_sha256": sha256_bytes(packet),
    }
    return packet, facts


def parse_packet(packet: bytes) -> tuple[np.ndarray, np.ndarray, Parameters]:
    if len(packet) < PACKET_HEADER.size:
        raise GDC2Error("packet truncated")
    magic, version, count, reserved = PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC or version != PACKET_VERSION or count != len(SECTION_NAMES):
        raise GDC2Error("packet header mismatch")
    if reserved:
        raise GDC2Error("packet reserved field is not zero")
    cursor = PACKET_HEADER.size
    descriptors = []
    for _ in range(count):
        coder_id, pad, raw_length, coded_length = PACKET_SECTION.unpack_from(packet, cursor)
        cursor += PACKET_SECTION.size
        if pad or coder_id not in ID_CODERS:
            raise GDC2Error("packet section descriptor invalid")
        descriptors.append((ID_CODERS[coder_id], raw_length, coded_length))
    payloads: list[bytes] = []
    for coder, raw_length, coded_length in descriptors:
        body = packet[cursor:cursor + coded_length]
        cursor += coded_length
        raw = decompress_payload(body, coder)
        if len(raw) != raw_length:
            raise GDC2Error("packet section length mismatch")
        payloads.append(raw)
    if cursor != len(packet):
        raise GDC2Error("packet has trailing bytes")
    z0 = np.frombuffer(payloads[0], dtype=np.int8).reshape(Z0_SHAPE).astype(np.int64)
    z1 = np.frombuffer(payloads[1], dtype=np.int8).reshape(Z1_SHAPE).astype(np.int64)
    return z0, z1, Parameters.from_bytes(payloads[2])


# --------------------------------------------------------------------------
# deterministic training tile schedule
# --------------------------------------------------------------------------


def build_tile_schedule(steps: int, blocks: int, pair_span: int, tile: int) -> np.ndarray:
    """One deterministic schedule for the whole campaign; generated once."""
    generator = np.random.default_rng(SEED)
    schedule = np.empty((steps, blocks, 3), dtype=np.int32)
    schedule[:, :, 0] = generator.integers(0, N_PAIRS - pair_span + 1, size=(steps, blocks))
    schedule[:, :, 1] = generator.integers(0, HEIGHT - tile + 1, size=(steps, blocks))
    schedule[:, :, 2] = generator.integers(0, WIDTH - tile + 1, size=(steps, blocks))
    return schedule
