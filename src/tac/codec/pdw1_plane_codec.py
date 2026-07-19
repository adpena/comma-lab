# SPDX-License-Identifier: MIT
"""PDW1P — the minimal PDW1-partition → scorer-plane receiver grammar (v1).

Payload → labels → uint8 scorer plane; the plane then rides the PROVEN
factor-2 lattice realization (``tac.optimization.uint8_lattice_feasibility``)
to an exact uint8 camera frame — the C1 spine.  This codec deliberately owns
ONLY the section between counted bytes and the scorer plane; it forks no
decode downstream of the plane.

Wire format ``PDW1PLN1`` (little-endian, strict, fail-closed):

    magic   8s   b"PDW1PLN1"
    version u16  = 1
    n_pairs u16  >= 1
    height  u16  (scorer plane H)
    width   u16  (scorer plane W)
    K       u8   number of classes (= 5 for the frozen head)
    then per pair, in ascending canonical pair order:
        fills   K*3 bytes  uint8 RGB fill per class id 0..K-1
        len     u32        compressed label-field byte length
        labels  <len>      Brotli(q11) of the row-major uint8 label field

Decode refuses: bad magic/version, zero pairs/geometry, label values >= K,
compressed streams that do not decode to exactly H*W bytes, truncation, and
every trailing byte.  Encode→fresh decode→encode is byte-identical (Brotli
q11 is deterministic for fixed input).

Honesty boundary (measured in the 2026-07-19 realization memo): the PDW1
power-diagram COEFFICIENTS are an encoder-side certificate — with no channel
feature field at decode time no receiver can expand coefficients alone into a
spatial partition, so this grammar ships the realized partition (the
contract-fp32 PDW1 cell labels) explicitly and ships ONLY bytes the receiver
consumes.  Coefficient bytes are therefore NOT in the payload (dead bytes
would violate the consumption discipline); they are recorded in the receipt.

Research-only: no score, promotion, or archive authority.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import brotli
import numpy as np

MAGIC = b"PDW1PLN1"
VERSION = 1
_HEADER = struct.Struct("<8sHHHHB")
_U32 = struct.Struct("<I")
BROTLI_QUALITY = 11
MAX_PAIRS = 600
MAX_DIM = 2048
MAX_CLASSES = 16


class Pdw1PlaneCodecError(ValueError):
    """Strict PDW1P parse/build failure."""


@dataclass(frozen=True)
class Pdw1PlanePayload:
    """Decoded PDW1P payload: per-pair label fields + class fills."""

    labels: np.ndarray  # (n_pairs, H, W) uint8, values < n_classes
    fills: np.ndarray  # (n_pairs, n_classes, 3) uint8

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        fills = np.asarray(self.fills)
        if labels.dtype != np.uint8 or labels.ndim != 3:
            raise Pdw1PlaneCodecError("labels must be (n_pairs,H,W) uint8")
        if fills.dtype != np.uint8 or fills.ndim != 3 or fills.shape[2] != 3:
            raise Pdw1PlaneCodecError("fills must be (n_pairs,K,3) uint8")
        n_pairs, height, width = labels.shape
        if not (1 <= n_pairs <= MAX_PAIRS):
            raise Pdw1PlaneCodecError(f"n_pairs {n_pairs} outside [1,{MAX_PAIRS}]")
        if not (1 <= height <= MAX_DIM and 1 <= width <= MAX_DIM):
            raise Pdw1PlaneCodecError(f"plane geometry {height}x{width} outside bounds")
        if fills.shape[0] != n_pairs:
            raise Pdw1PlaneCodecError("fills pair count must match labels")
        n_classes = fills.shape[1]
        if not (2 <= n_classes <= MAX_CLASSES):
            raise Pdw1PlaneCodecError(f"n_classes {n_classes} outside [2,{MAX_CLASSES}]")
        if int(labels.max(initial=0)) >= n_classes:
            raise Pdw1PlaneCodecError("label value >= n_classes")

    @property
    def n_pairs(self) -> int:
        return int(np.asarray(self.labels).shape[0])

    @property
    def n_classes(self) -> int:
        return int(np.asarray(self.fills).shape[1])


def encode_pdw1p(payload: Pdw1PlanePayload) -> bytes:
    """Serialize to strict PDW1P bytes (labels Brotli-q11 per pair)."""

    labels = np.ascontiguousarray(payload.labels)
    fills = np.ascontiguousarray(payload.fills)
    n_pairs, height, width = labels.shape
    out = bytearray(
        _HEADER.pack(MAGIC, VERSION, n_pairs, height, width, payload.n_classes)
    )
    for pair in range(n_pairs):
        out += fills[pair].tobytes()
        stream = brotli.compress(labels[pair].tobytes(), quality=BROTLI_QUALITY)
        out += _U32.pack(len(stream))
        out += stream
    return bytes(out)


def decode_pdw1p(blob: bytes | bytearray | memoryview) -> Pdw1PlanePayload:
    """Strict parse; refuses malformed geometry, short/overlong streams, trailers."""

    data = bytes(blob)
    if len(data) < _HEADER.size:
        raise Pdw1PlaneCodecError("payload shorter than header")
    magic, version, n_pairs, height, width, n_classes = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise Pdw1PlaneCodecError("bad magic")
    if version != VERSION:
        raise Pdw1PlaneCodecError(f"unsupported version {version}")
    if not (1 <= n_pairs <= MAX_PAIRS):
        raise Pdw1PlaneCodecError(f"n_pairs {n_pairs} outside [1,{MAX_PAIRS}]")
    if not (1 <= height <= MAX_DIM and 1 <= width <= MAX_DIM):
        raise Pdw1PlaneCodecError(f"plane geometry {height}x{width} outside bounds")
    if not (2 <= n_classes <= MAX_CLASSES):
        raise Pdw1PlaneCodecError(f"n_classes {n_classes} outside [2,{MAX_CLASSES}]")
    offset = _HEADER.size
    plane_bytes = height * width
    labels = np.empty((n_pairs, height, width), dtype=np.uint8)
    fills = np.empty((n_pairs, n_classes, 3), dtype=np.uint8)
    for pair in range(n_pairs):
        fill_len = n_classes * 3
        if offset + fill_len + _U32.size > len(data):
            raise Pdw1PlaneCodecError(f"truncated fills/length at pair {pair}")
        fills[pair] = np.frombuffer(
            data, dtype=np.uint8, count=fill_len, offset=offset
        ).reshape(n_classes, 3)
        offset += fill_len
        (stream_len,) = _U32.unpack_from(data, offset)
        offset += _U32.size
        if offset + stream_len > len(data):
            raise Pdw1PlaneCodecError(f"truncated label stream at pair {pair}")
        try:
            raw = brotli.decompress(data[offset : offset + stream_len])
        except brotli.error as exc:  # pragma: no cover - brotli detail text varies
            raise Pdw1PlaneCodecError(f"label stream {pair} failed to decode") from exc
        offset += stream_len
        if len(raw) != plane_bytes:
            raise Pdw1PlaneCodecError(
                f"label stream {pair} decoded {len(raw)} bytes, expected {plane_bytes}"
            )
        plane = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        if int(plane.max(initial=0)) >= n_classes:
            raise Pdw1PlaneCodecError(f"label value >= n_classes in pair {pair}")
        labels[pair] = plane
    if offset != len(data):
        raise Pdw1PlaneCodecError(f"{len(data) - offset} trailing bytes")
    return Pdw1PlanePayload(labels=labels, fills=fills)


def expand_scorer_plane(payload: Pdw1PlanePayload, pair: int) -> np.ndarray:
    """Expand one pair to its uint8 RGB scorer plane: ``plane[y,x] = fill[label[y,x]]``.

    The returned (H,W,3) uint8 plane is EXACTLY what the proven factor-2
    lattice realization consumes (``realize_factor2_uint8_scorer_plane``);
    ``A(camera_frame) = plane`` then holds to integer-numerator equality.
    """

    if not (0 <= pair < payload.n_pairs):
        raise Pdw1PlaneCodecError(f"pair {pair} out of range")
    labels = np.asarray(payload.labels)[pair]
    fills = np.asarray(payload.fills)[pair]
    return fills[labels]
