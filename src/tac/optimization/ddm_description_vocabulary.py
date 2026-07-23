# SPDX-License-Identifier: MIT
"""Description-space primitives for persistent ground-boundary worldsheets.

The primitives in this module are semantic, receiver-portable derivation
strings.  They deliberately stop at class-cell proposals: they neither load a
scorer nor claim that a semantic proposal survives RGB realization.

Three complementary coordinate systems are implemented:

``PersistentLevelSetV1``
    a time-amortized class partition ``argmax_c count_t[class_t(x)=c]``;
``BoundaryWorldsheetSplineV1``
    the Road separatrix ``y = gamma(t, x)`` sampled on sparse temporal and
    horizontal knots, then bilinearly reconstructed;
``TurningAngleCurveV1``
    closed Road curves represented by arc lengths and quantized tangent
    angles, rather than per-pixel corrections or raw polygon vertices.

Every wire string is encoded by the smallest measured member of the actual
raw/Brotli/LZMA/zlib coder family and is strictly parsed before it may be used.
"""

from __future__ import annotations

import lzma
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Final

import brotli
import cv2
import numpy as np

HEIGHT: Final = 384
WIDTH: Final = 512
N_CLASSES: Final = 5
ROAD_ID: Final = 0
UNDRIVABLE_ID: Final = 2

_ENVELOPE_MAGIC: Final = b"DV1E1"
_ENVELOPE_HEADER: Final = struct.Struct(">5sBBII")
_CODEC_IDS: Final = {"raw": 0, "brotli11": 1, "lzma1_raw": 2, "zlib9": 3}
_CODEC_NAMES: Final = {value: key for key, value in _CODEC_IDS.items()}
_KIND_IDS: Final = {
    "persistent_level_set": 1,
    "boundary_worldsheet_spline": 2,
    "turning_angle_curve": 3,
    "joint_ground_vocabulary": 4,
}
_KIND_NAMES: Final = {value: key for key, value in _KIND_IDS.items()}

_LEVEL_SET_HEADER: Final = struct.Struct(">5sBHH")
_SPLINE_HEADER: Final = struct.Struct(">5sBHHHHHH")
_CURVE_HEADER: Final = struct.Struct(">5sBHHIB")


class DescriptionVocabularyError(RuntimeError):
    """Raised when a derivation string or geometric invariant is invalid."""


@dataclass(frozen=True, slots=True)
class CodedDerivation:
    kind: str
    codec: str
    raw_bytes: int
    counted_bytes: int
    envelope: bytes


@dataclass(frozen=True, slots=True)
class BoundarySplineMetadata:
    pair_count: int
    temporal_stride: int
    horizontal_stride: int
    temporal_knots: int
    horizontal_knots: int


@dataclass(frozen=True, slots=True)
class TurningCurveMetadata:
    pair_count: int
    heading_bins: int
    epsilon_pixels: float
    contour_count: int
    segment_count: int


def _put_uleb(output: bytearray, value: int) -> None:
    if value < 0:
        raise DescriptionVocabularyError("ULEB value must be nonnegative")
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return


def _get_uleb(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise DescriptionVocabularyError("truncated or oversized ULEB")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _zigzag(value: int) -> int:
    return (int(value) << 1) ^ (int(value) >> 63)


def _unzigzag(value: int) -> int:
    return (int(value) >> 1) ^ -(int(value) & 1)


def _codec_variants(raw: bytes) -> dict[str, bytes]:
    return {
        "raw": raw,
        "brotli11": brotli.compress(raw, quality=11),
        "lzma1_raw": lzma.compress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=[
                {
                    "id": lzma.FILTER_LZMA1,
                    "preset": 1,
                    "dict_size": 1 << 20,
                    "lc": 3,
                    "lp": 0,
                    "pb": 2,
                }
            ],
        ),
        "zlib9": zlib.compress(raw, 9),
    }


def encode_derivation(kind: str, raw: bytes) -> CodedDerivation:
    """Encode one typed derivation with measured real-coder selection."""

    if kind not in _KIND_IDS:
        raise DescriptionVocabularyError(f"unknown derivation kind: {kind!r}")
    variants = _codec_variants(raw)
    codec = min(
        variants,
        key=lambda name: (
            _ENVELOPE_HEADER.size + len(variants[name]),
            _CODEC_IDS[name],
        ),
    )
    coded = variants[codec]
    envelope = _ENVELOPE_HEADER.pack(
        _ENVELOPE_MAGIC,
        _KIND_IDS[kind],
        _CODEC_IDS[codec],
        len(raw),
        len(coded),
    ) + coded
    decoded_kind, decoded_raw = decode_derivation(envelope)
    if decoded_kind != kind or decoded_raw != raw:
        raise DescriptionVocabularyError("real-coder parse-back mismatch")
    return CodedDerivation(
        kind=kind,
        codec=codec,
        raw_bytes=len(raw),
        counted_bytes=len(envelope),
        envelope=envelope,
    )


def decode_derivation(envelope: bytes) -> tuple[str, bytes]:
    if len(envelope) < _ENVELOPE_HEADER.size:
        raise DescriptionVocabularyError("truncated derivation envelope")
    magic, kind_id, codec_id, raw_size, coded_size = _ENVELOPE_HEADER.unpack_from(envelope)
    if magic != _ENVELOPE_MAGIC:
        raise DescriptionVocabularyError("derivation magic mismatch")
    if kind_id not in _KIND_NAMES or codec_id not in _CODEC_NAMES:
        raise DescriptionVocabularyError("derivation kind or codec is unknown")
    coded = envelope[_ENVELOPE_HEADER.size :]
    if len(coded) != coded_size:
        raise DescriptionVocabularyError("derivation coded size mismatch")
    codec = _CODEC_NAMES[codec_id]
    try:
        if codec == "raw":
            raw = coded
        elif codec == "brotli11":
            raw = brotli.decompress(coded)
        elif codec == "lzma1_raw":
            raw = lzma.decompress(
                coded,
                format=lzma.FORMAT_RAW,
                filters=[
                    {
                        "id": lzma.FILTER_LZMA1,
                        "preset": 1,
                        "dict_size": 1 << 20,
                        "lc": 3,
                        "lp": 0,
                        "pb": 2,
                    }
                ],
            )
        else:
            raw = zlib.decompress(coded)
    except (brotli.error, lzma.LZMAError, zlib.error) as exc:
        raise DescriptionVocabularyError("derivation decompression failed") from exc
    if len(raw) != raw_size:
        raise DescriptionVocabularyError("derivation raw size mismatch")
    return _KIND_NAMES[kind_id], raw


def inspect_coded_derivation(envelope: bytes) -> CodedDerivation:
    """Return validated envelope metadata without trusting external receipts."""

    kind, raw = decode_derivation(envelope)
    _magic, _kind_id, codec_id, _raw_size, _coded_size = _ENVELOPE_HEADER.unpack_from(
        envelope
    )
    return CodedDerivation(
        kind=kind,
        codec=_CODEC_NAMES[codec_id],
        raw_bytes=len(raw),
        counted_bytes=len(envelope),
        envelope=envelope,
    )


def fit_persistent_level_set(labels: np.ndarray) -> np.ndarray:
    """Fit the time-amortized five-class level-set partition."""

    if labels.ndim != 3 or labels.shape[1:] != (HEIGHT, WIDTH):
        raise DescriptionVocabularyError(f"unexpected label shape: {labels.shape}")
    counts = np.zeros((N_CLASSES, HEIGHT, WIDTH), dtype=np.uint16)
    for pair_index in range(labels.shape[0]):
        plane = np.asarray(labels[pair_index], dtype=np.uint8)
        if np.any(plane >= N_CLASSES):
            raise DescriptionVocabularyError("label id outside the five-class vocabulary")
        for class_id in range(N_CLASSES):
            counts[class_id] += plane == class_id
    return np.argmax(counts, axis=0).astype(np.uint8)


def encode_persistent_level_set(field: np.ndarray) -> CodedDerivation:
    values = np.ascontiguousarray(field, dtype=np.uint8)
    if values.shape != (HEIGHT, WIDTH) or np.any(values >= N_CLASSES):
        raise DescriptionVocabularyError("persistent level-set field is invalid")
    raw = _LEVEL_SET_HEADER.pack(b"DVLS1", 1, HEIGHT, WIDTH) + values.tobytes()
    return encode_derivation("persistent_level_set", raw)


def decode_persistent_level_set(envelope: bytes) -> np.ndarray:
    kind, raw = decode_derivation(envelope)
    if kind != "persistent_level_set" or len(raw) < _LEVEL_SET_HEADER.size:
        raise DescriptionVocabularyError("not a persistent level-set derivation")
    magic, version, height, width = _LEVEL_SET_HEADER.unpack_from(raw)
    if magic != b"DVLS1" or version != 1 or (height, width) != (HEIGHT, WIDTH):
        raise DescriptionVocabularyError("persistent level-set header mismatch")
    body = raw[_LEVEL_SET_HEADER.size :]
    if len(body) != HEIGHT * WIDTH:
        raise DescriptionVocabularyError("persistent level-set body size mismatch")
    field = np.frombuffer(body, dtype=np.uint8).reshape(HEIGHT, WIDTH)
    if np.any(field >= N_CLASSES):
        raise DescriptionVocabularyError("persistent level-set class id is invalid")
    return np.array(field, copy=True)


def _road_top_at_columns(mask: np.ndarray, columns: np.ndarray) -> np.ndarray:
    values = np.full(columns.size, HEIGHT, dtype=np.int16)
    for index, column in enumerate(columns):
        rows = np.flatnonzero(mask[96:, int(column)])
        if rows.size:
            values[index] = int(rows[0]) + 96
    return values


def fit_boundary_worldsheet_spline(
    labels: np.ndarray,
    *,
    temporal_stride: int = 8,
    horizontal_stride: int = 16,
) -> tuple[CodedDerivation, np.ndarray, BoundarySplineMetadata]:
    """Fit ``gamma(t,x)`` for the upper Road separatrix."""

    if labels.ndim != 3 or labels.shape[1:] != (HEIGHT, WIDTH):
        raise DescriptionVocabularyError(f"unexpected label shape: {labels.shape}")
    if temporal_stride < 1 or horizontal_stride < 1:
        raise DescriptionVocabularyError("spline strides must be positive")
    pair_count = int(labels.shape[0])
    temporal = np.unique(
        np.append(np.arange(0, pair_count, temporal_stride, dtype=np.int32), pair_count - 1)
    )
    horizontal = np.unique(
        np.append(np.arange(0, WIDTH, horizontal_stride, dtype=np.int32), WIDTH - 1)
    )
    knot_values = np.empty((temporal.size, horizontal.size), dtype=np.int16)
    for row_index, pair_index in enumerate(temporal):
        knot_values[row_index] = _road_top_at_columns(
            np.asarray(labels[int(pair_index)]) == ROAD_ID,
            horizontal,
        )

    body = bytearray()
    previous = np.zeros(horizontal.size, dtype=np.int32)
    for row_index, values in enumerate(knot_values.astype(np.int32)):
        delta = values if row_index == 0 else values - previous
        for value in delta:
            _put_uleb(body, _zigzag(int(value)))
        previous = values
    raw = _SPLINE_HEADER.pack(
        b"DVBS1",
        1,
        pair_count,
        HEIGHT,
        WIDTH,
        temporal_stride,
        horizontal_stride,
        temporal.size,
    ) + bytes(body)
    derivation = encode_derivation("boundary_worldsheet_spline", raw)
    rendered, metadata = decode_boundary_worldsheet_spline(derivation.envelope)
    return derivation, rendered, metadata


def decode_boundary_worldsheet_spline(
    envelope: bytes,
) -> tuple[np.ndarray, BoundarySplineMetadata]:
    kind, raw = decode_derivation(envelope)
    if kind != "boundary_worldsheet_spline" or len(raw) < _SPLINE_HEADER.size:
        raise DescriptionVocabularyError("not a boundary-worldsheet spline")
    (
        magic,
        version,
        pair_count,
        height,
        width,
        temporal_stride,
        horizontal_stride,
        temporal_count,
    ) = _SPLINE_HEADER.unpack_from(raw)
    if magic != b"DVBS1" or version != 1 or (height, width) != (HEIGHT, WIDTH):
        raise DescriptionVocabularyError("boundary-worldsheet spline header mismatch")
    temporal = np.unique(
        np.append(np.arange(0, pair_count, temporal_stride, dtype=np.int32), pair_count - 1)
    )
    horizontal = np.unique(
        np.append(np.arange(0, WIDTH, horizontal_stride, dtype=np.int32), WIDTH - 1)
    )
    if temporal.size != temporal_count:
        raise DescriptionVocabularyError("boundary-worldsheet temporal knot count mismatch")
    knots = np.empty((temporal.size, horizontal.size), dtype=np.int32)
    offset = _SPLINE_HEADER.size
    previous = np.zeros(horizontal.size, dtype=np.int32)
    for row_index in range(temporal.size):
        values = np.empty(horizontal.size, dtype=np.int32)
        for column_index in range(horizontal.size):
            coded, offset = _get_uleb(raw, offset)
            values[column_index] = _unzigzag(coded)
        if row_index:
            values += previous
        knots[row_index] = values
        previous = values
    if offset != len(raw):
        raise DescriptionVocabularyError("boundary-worldsheet spline has trailing bytes")

    pair_axis = np.arange(pair_count)
    column_axis = np.arange(WIDTH)
    dense_temporal = np.empty((pair_count, horizontal.size), dtype=np.float64)
    for column_index in range(horizontal.size):
        dense_temporal[:, column_index] = np.interp(
            pair_axis,
            temporal,
            knots[:, column_index],
        )
    curves = np.empty((pair_count, WIDTH), dtype=np.int16)
    for pair_index in range(pair_count):
        curves[pair_index] = np.rint(
            np.interp(column_axis, horizontal, dense_temporal[pair_index])
        ).clip(0, HEIGHT)
    rows = np.arange(HEIGHT, dtype=np.int16)[:, None]
    rendered = rows[None, :, :] >= curves[:, None, :]
    metadata = BoundarySplineMetadata(
        pair_count=pair_count,
        temporal_stride=temporal_stride,
        horizontal_stride=horizontal_stride,
        temporal_knots=int(temporal.size),
        horizontal_knots=int(horizontal.size),
    )
    return np.ascontiguousarray(rendered), metadata


def fit_turning_angle_curves(
    labels: np.ndarray,
    *,
    epsilon_pixels: float = 3.0,
    heading_bins: int = 32,
    minimum_area: float = 24.0,
) -> tuple[CodedDerivation, np.ndarray, TurningCurveMetadata]:
    """Fit closed Road curves as ``(arc length, tangent angle)`` segments."""

    if labels.ndim != 3 or labels.shape[1:] != (HEIGHT, WIDTH):
        raise DescriptionVocabularyError(f"unexpected label shape: {labels.shape}")
    if epsilon_pixels <= 0 or heading_bins < 8 or heading_bins > 255:
        raise DescriptionVocabularyError("turning-curve quantization is invalid")
    pair_count = int(labels.shape[0])
    body = bytearray()
    contour_count = 0
    segment_count = 0
    for pair_index in range(pair_count):
        mask = np.asarray(labels[pair_index] == ROAD_ID, dtype=np.uint8)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[np.ndarray] = []
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                continue
            polygon = cv2.approxPolyDP(contour, epsilon_pixels, True)[:, 0, :].astype(np.int32)
            if polygon.shape[0] >= 3:
                polygons.append(polygon)
        polygons.sort(key=lambda row: tuple(int(value) for value in cv2.boundingRect(row[:, None, :])[:2]))
        _put_uleb(body, len(polygons))
        for polygon in polygons:
            _put_uleb(body, int(polygon[0, 0]))
            _put_uleb(body, int(polygon[0, 1]))
            _put_uleb(body, int(polygon.shape[0]))
            closed = np.vstack((polygon, polygon[0]))
            for delta in np.diff(closed, axis=0):
                dx, dy = int(delta[0]), int(delta[1])
                length = max(1, round(math.hypot(dx, dy)))
                angle = math.atan2(dy, dx) % (2.0 * math.pi)
                heading = round(angle * heading_bins / (2.0 * math.pi)) % heading_bins
                body.append(heading)
                _put_uleb(body, length)
                segment_count += 1
            contour_count += 1
    epsilon_q8 = round(epsilon_pixels * 8.0)
    raw = _CURVE_HEADER.pack(
        b"DVTC1",
        1,
        pair_count,
        epsilon_q8,
        contour_count,
        heading_bins,
    ) + bytes(body)
    derivation = encode_derivation("turning_angle_curve", raw)
    rendered, metadata = decode_turning_angle_curves(derivation.envelope)
    if metadata.segment_count != segment_count:
        raise DescriptionVocabularyError("turning-curve segment accounting mismatch")
    return derivation, rendered, metadata


def decode_turning_angle_curves(
    envelope: bytes,
) -> tuple[np.ndarray, TurningCurveMetadata]:
    kind, raw = decode_derivation(envelope)
    if kind != "turning_angle_curve" or len(raw) < _CURVE_HEADER.size:
        raise DescriptionVocabularyError("not a turning-angle curve derivation")
    magic, version, pair_count, epsilon_q8, expected_contours, heading_bins = (
        _CURVE_HEADER.unpack_from(raw)
    )
    if magic != b"DVTC1" or version != 1:
        raise DescriptionVocabularyError("turning-angle curve header mismatch")
    output = np.zeros((pair_count, HEIGHT, WIDTH), dtype=np.uint8)
    offset = _CURVE_HEADER.size
    contour_count = 0
    segment_count = 0
    angle_step = 2.0 * math.pi / heading_bins
    for pair_index in range(pair_count):
        pair_contours, offset = _get_uleb(raw, offset)
        for _ in range(pair_contours):
            start_x, offset = _get_uleb(raw, offset)
            start_y, offset = _get_uleb(raw, offset)
            edge_count, offset = _get_uleb(raw, offset)
            if edge_count < 3:
                raise DescriptionVocabularyError("turning curve has fewer than three edges")
            points = [(float(start_x), float(start_y))]
            for _edge in range(edge_count):
                if offset >= len(raw):
                    raise DescriptionVocabularyError("truncated turning-curve heading")
                heading = raw[offset]
                offset += 1
                if heading >= heading_bins:
                    raise DescriptionVocabularyError("turning-curve heading outside codebook")
                length, offset = _get_uleb(raw, offset)
                x_value = points[-1][0] + length * math.cos(heading * angle_step)
                y_value = points[-1][1] + length * math.sin(heading * angle_step)
                points.append((x_value, y_value))
                segment_count += 1
            polygon = np.rint(np.asarray(points[:-1])).astype(np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, WIDTH - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, HEIGHT - 1)
            cv2.fillPoly(output[pair_index], [polygon[:, None, :]], 1)
            contour_count += 1
    if offset != len(raw) or contour_count != expected_contours:
        raise DescriptionVocabularyError("turning-curve count or trailing-byte mismatch")
    metadata = TurningCurveMetadata(
        pair_count=pair_count,
        heading_bins=heading_bins,
        epsilon_pixels=epsilon_q8 / 8.0,
        contour_count=contour_count,
        segment_count=segment_count,
    )
    return output.astype(bool), metadata


def encode_joint_ground_vocabulary(sections: list[CodedDerivation]) -> CodedDerivation:
    """Build one jointly coded typed stream; sections are not byte-estimated."""

    if not sections:
        raise DescriptionVocabularyError("joint ground vocabulary cannot be empty")
    raw = bytearray(b"DVJG1")
    raw.append(1)
    _put_uleb(raw, len(sections))
    for section in sections:
        kind_id = _KIND_IDS[section.kind]
        raw.append(kind_id)
        _put_uleb(raw, len(section.envelope))
        raw.extend(section.envelope)
    derivation = encode_derivation("joint_ground_vocabulary", bytes(raw))
    decoded = decode_joint_ground_vocabulary(derivation.envelope)
    if tuple(section.envelope for section in decoded) != tuple(
        section.envelope for section in sections
    ):
        raise DescriptionVocabularyError("joint ground vocabulary parse-back mismatch")
    return derivation


def decode_joint_ground_vocabulary(envelope: bytes) -> tuple[CodedDerivation, ...]:
    """Strictly parse and validate every nested typed derivation."""

    kind, raw = decode_derivation(envelope)
    if kind != "joint_ground_vocabulary" or len(raw) < 6:
        raise DescriptionVocabularyError("not a joint ground vocabulary")
    if raw[:5] != b"DVJG1" or raw[5] != 1:
        raise DescriptionVocabularyError("joint ground vocabulary header mismatch")
    count, offset = _get_uleb(raw, 6)
    if count < 1:
        raise DescriptionVocabularyError("joint ground vocabulary cannot be empty")
    sections: list[CodedDerivation] = []
    for _index in range(count):
        if offset >= len(raw):
            raise DescriptionVocabularyError("truncated joint ground vocabulary kind")
        expected_kind_id = raw[offset]
        offset += 1
        size, offset = _get_uleb(raw, offset)
        stop = offset + size
        if stop > len(raw):
            raise DescriptionVocabularyError("truncated joint ground vocabulary section")
        section = inspect_coded_derivation(raw[offset:stop])
        if _KIND_IDS[section.kind] != expected_kind_id or section.kind == "joint_ground_vocabulary":
            raise DescriptionVocabularyError("joint ground vocabulary section kind mismatch")
        sections.append(section)
        offset = stop
    if offset != len(raw):
        raise DescriptionVocabularyError("joint ground vocabulary has trailing bytes")
    return tuple(sections)


__all__ = [
    "HEIGHT",
    "N_CLASSES",
    "ROAD_ID",
    "UNDRIVABLE_ID",
    "WIDTH",
    "BoundarySplineMetadata",
    "CodedDerivation",
    "DescriptionVocabularyError",
    "TurningCurveMetadata",
    "decode_boundary_worldsheet_spline",
    "decode_derivation",
    "decode_joint_ground_vocabulary",
    "decode_persistent_level_set",
    "decode_turning_angle_curves",
    "encode_derivation",
    "encode_joint_ground_vocabulary",
    "encode_persistent_level_set",
    "fit_boundary_worldsheet_spline",
    "fit_persistent_level_set",
    "fit_turning_angle_curves",
    "inspect_coded_derivation",
]
