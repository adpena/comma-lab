# SPDX-License-Identifier: MIT
"""Deterministic two-plane predictor/residual codec for the V10 receiver.

The payload is deliberately self-describing and fully charged.  Every record
contains Brotli-Q11 streams for its frame-0 bootstrap and exact little-endian
int16 residual, plus any raw video-derived predictor descriptor.  Decode needs
only this module, NumPy, and the generic Brotli runtime; it never consults a
scorer, source frame, cache, or learned parameter.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from itertools import pairwise
from typing import Any, cast

import numpy as np

MAGIC = b"TACV10PR"
VERSION = 1
Q12_ONE = 1 << 12

# magic, version, closed content-codec tag, pair_count, height, width, channels
PREFIX = struct.Struct("<8sHB3xIIIH")
# pair_id, mode_id, bootstrap_len, descriptor_len, residual_len,
# bootstrap_sha, descriptor_sha, residual_sha, reconstructed_y1_sha
PAIR_PREFIX = struct.Struct("<IB3xIII32s32s32s32s")
AFFINE6 = struct.Struct("<6i")

CODEC_ID = "predictor-residual-u8.v1"
CONTENT_CODEC_ID = "brotli-q11.v1"
CONTENT_CODEC_TAG = 1
PREVIOUS_PLANE_COPY_ID = "previous-plane-copy.v1"
AFFINE6_Q12_ID = "affine6-q12.v1"
SPATIAL_SMOOTH_121_ID = "spatial-smooth-121.v1"

MAX_PAIRS = 10_000
MAX_HEIGHT = 2048
MAX_WIDTH = 2048
MAX_PAYLOAD_BYTES = 1 << 30
MAX_DECODED_WORKING_SET_BYTES = 2 << 30
_INT64_MAX = (1 << 63) - 1
_AFFINE_GRADIENT_NUMERATOR_ABS_MAX = 255
_AFFINE_RESPONSE_NUMERATOR_ABS_MAX = 510


class PredictorResidualError(ValueError):
    """Fail-closed malformed predictor descriptor, payload, or plane."""


class PredictorMode(IntEnum):
    PREVIOUS_PLANE_COPY = 0
    AFFINE6_Q12 = 1
    SPATIAL_SMOOTH_121 = 2


MODE_IDS = {
    PredictorMode.PREVIOUS_PLANE_COPY: PREVIOUS_PLANE_COPY_ID,
    PredictorMode.AFFINE6_Q12: AFFINE6_Q12_ID,
    PredictorMode.SPATIAL_SMOOTH_121: SPATIAL_SMOOTH_121_ID,
}
MODE_BY_ID = {value: key for key, value in MODE_IDS.items()}


@dataclass(frozen=True)
class PredictorResidualAccounting:
    payload_bytes: int
    framing_bytes: int
    bootstrap_bytes: int
    descriptor_bytes: int
    residual_bytes: int
    decoded_bootstrap_bytes: int
    decoded_residual_bytes: int

    @property
    def conditional_bytes(self) -> int:
        """Descriptor plus residual bytes, excluding the paid bootstrap."""

        return self.descriptor_bytes + self.residual_bytes


@dataclass(frozen=True)
class DecodedPredictorResidual:
    frame0: np.ndarray
    frame1: np.ndarray
    pair_ids: tuple[int, ...]
    modes: tuple[str, ...]
    descriptors: tuple[bytes, ...]
    accounting: PredictorResidualAccounting


def _sha256(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _brotli() -> Any:
    try:
        import brotli  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PredictorResidualError("brotli-q11 predictor content codec is unavailable") from exc
    return brotli


def _compress_content(payload: bytes) -> bytes:
    try:
        return bytes(_brotli().compress(payload, quality=11))
    except Exception as exc:
        raise PredictorResidualError("brotli-q11 predictor compression failed") from exc


def _decompress_content(payload: bytes, *, expected_bytes: int, label: str) -> bytes:
    try:
        decoded = bytes(_brotli().decompress(payload))
    except Exception as exc:
        raise PredictorResidualError(f"brotli-q11 {label} decompression failed") from exc
    if len(decoded) != expected_bytes:
        raise PredictorResidualError(f"brotli-q11 {label} decoded length/geometry drift")
    return decoded


def _exact_uint8_plane(value: np.ndarray, label: str) -> np.ndarray:
    plane = np.asarray(value)
    if plane.dtype != np.uint8 or plane.ndim != 3 or plane.shape[-1] != 3:
        raise PredictorResidualError(f"{label} must be exact uint8 [H,W,3]")
    height, width, channels = map(int, plane.shape)
    if not 1 <= height <= MAX_HEIGHT or not 1 <= width <= MAX_WIDTH or channels != 3:
        raise PredictorResidualError(f"{label} geometry is outside the admitted bounds")
    return np.ascontiguousarray(plane)


def _exact_uint8_planes(value: np.ndarray, label: str) -> np.ndarray:
    planes = np.asarray(value)
    if planes.dtype != np.uint8 or planes.ndim != 4 or planes.shape[-1] != 3:
        raise PredictorResidualError(f"{label} must be exact uint8 [pairs,H,W,3]")
    pair_count, height, width, channels = map(int, planes.shape)
    if (
        not 1 <= pair_count <= MAX_PAIRS
        or not 1 <= height <= MAX_HEIGHT
        or not 1 <= width <= MAX_WIDTH
        or channels != 3
    ):
        raise PredictorResidualError(f"{label} geometry is outside the admitted bounds")
    return np.ascontiguousarray(planes)


def _coerce_mode(value: PredictorMode | str | int) -> PredictorMode:
    if isinstance(value, str):
        try:
            return MODE_BY_ID[value]
        except KeyError as exc:
            raise PredictorResidualError(f"unknown predictor mode {value!r}") from exc
    try:
        mode = PredictorMode(value)
    except (TypeError, ValueError) as exc:
        raise PredictorResidualError(f"unknown predictor mode {value!r}") from exc
    if mode not in MODE_IDS:  # defensive if the enum grows without codec support
        raise PredictorResidualError(f"unimplemented predictor mode {mode!r}")
    return mode


def _descriptor_length(mode: PredictorMode) -> int:
    return AFFINE6.size if mode is PredictorMode.AFFINE6_Q12 else 0


def _decoded_working_set_bytes(pair_count: int, height: int, width: int, channels: int) -> int:
    """Validate the aggregate decoded y0+y1+int16-residual working set."""

    working_bytes = pair_count * height * width * channels * (1 + 1 + 2)
    if working_bytes > MAX_DECODED_WORKING_SET_BYTES:
        raise PredictorResidualError("predictor decoded aggregate working set exceeds its safety cap")
    return working_bytes


def _affine_sample_stride(height: int, width: int, channels: int) -> int:
    """Return a canonical stride that proves every int64 accumulation safe."""

    sample_count = height * width * channels
    max_coordinate = max(height - 1, width - 1, 1)
    max_design = _AFFINE_GRADIENT_NUMERATOR_ABS_MAX * max_coordinate
    max_term = max(
        max_design * max_design,
        max_design * _AFFINE_RESPONSE_NUMERATOR_ABS_MAX,
        1,
    )
    safe_samples = _INT64_MAX // max_term
    if safe_samples < 1:
        raise PredictorResidualError("affine integer normal equations have no safe sample")
    return max(1, (sample_count + safe_samples - 1) // safe_samples)


def _exact_normal_equation_solution(normal: Sequence[Sequence[int]], rhs: Sequence[int]) -> tuple[Fraction, ...]:
    """Solve a 6x6 rational system by canonical RREF, free variables zero."""

    if len(normal) != 6 or any(len(row) != 6 for row in normal) or len(rhs) != 6:
        raise PredictorResidualError("affine normal equations must be exactly 6x6")
    matrix = [
        [Fraction(int(normal[row][column])) for column in range(6)] + [Fraction(int(rhs[row]))] for row in range(6)
    ]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(6):
        selected = next(
            (row for row in range(pivot_row, 6) if matrix[row][column] != 0),
            None,
        )
        if selected is None:
            continue
        if selected != pivot_row:
            matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        pivot = matrix[pivot_row][column]
        matrix[pivot_row] = [value / pivot for value in matrix[pivot_row]]
        for row in range(6):
            if row == pivot_row or matrix[row][column] == 0:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                value - factor * pivot_value for value, pivot_value in zip(matrix[row], matrix[pivot_row], strict=True)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == 6:
            break
    for row in matrix:
        if all(value == 0 for value in row[:6]) and row[6] != 0:
            raise PredictorResidualError("affine exact normal equations are inconsistent")
    solution = [Fraction(0) for _ in range(6)]
    for row, column in enumerate(pivot_columns):
        solution[column] = matrix[row][6]
    return tuple(solution)


def _round_fraction_ties_to_even(value: Fraction) -> int:
    """Round an exact rational to the nearest integer with ties to even."""

    sign = -1 if value < 0 else 1
    numerator = abs(value.numerator)
    quotient, remainder = divmod(numerator, value.denominator)
    doubled = remainder * 2
    if doubled > value.denominator or (doubled == value.denominator and quotient % 2 == 1):
        quotient += 1
    return sign * quotient


def fit_affine6_q12_descriptor(frame0: np.ndarray, frame1: np.ndarray) -> bytes:
    """Fit a linearized six-parameter output-to-input affine displacement.

    Edge-replicated centered differences are kept as integer numerators:
    ``2*(y1-y0) = gx2*dx + gy2*dy``.  Canonical C-order full-pixel samples are
    used whenever the explicit int64 sum bound permits; otherwise a geometry-
    derived flat stride is used.  Integer normal equations are reduced by exact
    rational RREF with first-nonzero pivots and free variables fixed to zero.
    Q12 conversion uses exact ties-to-even rounding.  The path has no LAPACK,
    floating-point solve, or host-dependent pivot tolerance.

    Coefficients are ``dx=a0+a1*x+a2*y`` and ``dy=b0+b1*x+b2*y`` in pixels.
    """

    y0 = _exact_uint8_plane(frame0, "frame0")
    y1 = _exact_uint8_plane(frame1, "frame1")
    if y0.shape != y1.shape:
        raise PredictorResidualError("affine fit requires equal frame geometry")
    height, width, _ = y0.shape
    source = y0.astype(np.int64)
    target = y1.astype(np.int64)

    # Twice the centered difference avoids every fractional input to the solve.
    padded_x = np.pad(source, ((0, 0), (1, 1), (0, 0)), mode="edge")
    padded_y = np.pad(source, ((1, 1), (0, 0), (0, 0)), mode="edge")
    grad_x2 = padded_x[:, 2:, :] - padded_x[:, :-2, :]
    grad_y2 = padded_y[2:, :, :] - padded_y[:-2, :, :]
    yy, xx = np.indices((height, width), dtype=np.int64)
    xx = np.broadcast_to(xx[..., None], source.shape)
    yy = np.broadcast_to(yy[..., None], source.shape)
    design = np.stack(
        (
            grad_x2,
            grad_x2 * xx,
            grad_x2 * yy,
            grad_y2,
            grad_y2 * xx,
            grad_y2 * yy,
        ),
        axis=-1,
    ).reshape(-1, 6)
    response = (2 * (target - source)).reshape(-1)
    stride = _affine_sample_stride(height, width, 3)
    sampled_design = design[::stride]
    sampled_response = response[::stride]
    sample_count = len(sampled_response)
    actual_design_max = int(np.max(np.abs(sampled_design), initial=0))
    actual_response_max = int(np.max(np.abs(sampled_response), initial=0))
    if (
        sample_count
        * max(
            actual_design_max * actual_design_max,
            actual_design_max * actual_response_max,
            1,
        )
        > _INT64_MAX
    ):
        raise PredictorResidualError("affine integer accumulation bound was violated")
    normal = [[0 for _ in range(6)] for _ in range(6)]
    rhs = [0 for _ in range(6)]
    for row in range(6):
        rhs[row] = int(np.sum(sampled_design[:, row] * sampled_response, dtype=np.int64))
        for column in range(row, 6):
            value = int(
                np.sum(
                    sampled_design[:, row] * sampled_design[:, column],
                    dtype=np.int64,
                )
            )
            normal[row][column] = value
            normal[column][row] = value
    coefficients = _exact_normal_equation_solution(normal, rhs)
    scaled = tuple(_round_fraction_ties_to_even(coefficient * Q12_ONE) for coefficient in coefficients)
    int32 = np.iinfo(np.int32)
    if any(value < int32.min or value > int32.max for value in scaled):
        raise PredictorResidualError("affine exact Q12 coefficient exceeds int32")
    return AFFINE6.pack(*scaled)


def _affine6_q12_predict(frame0: np.ndarray, descriptor: bytes) -> np.ndarray:
    if not isinstance(descriptor, bytes) or len(descriptor) != AFFINE6.size:
        raise PredictorResidualError("affine6-q12 descriptor must be exactly 24 bytes")
    a0, a1, a2, b0, b1, b2 = AFFINE6.unpack(descriptor)
    source = _exact_uint8_plane(frame0, "frame0")
    height, width, _ = source.shape
    yy, xx = np.indices((height, width), dtype=np.int64)
    xq = (xx << 12) + np.int64(a0) + np.int64(a1) * xx + np.int64(a2) * yy
    yq = (yy << 12) + np.int64(b0) + np.int64(b1) * xx + np.int64(b2) * yy
    xq = np.clip(xq, 0, (width - 1) << 12)
    yq = np.clip(yq, 0, (height - 1) << 12)
    x0 = xq >> 12
    y0 = yq >> 12
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = xq & (Q12_ONE - 1)
    fy = yq & (Q12_ONE - 1)
    wx0 = Q12_ONE - fx
    wy0 = Q12_ONE - fy
    source64 = source.astype(np.int64)
    numerator = (
        source64[y0, x0] * (wx0 * wy0)[..., None]
        + source64[y0, x1] * (fx * wy0)[..., None]
        + source64[y1, x0] * (wx0 * fy)[..., None]
        + source64[y1, x1] * (fx * fy)[..., None]
    )
    return np.ascontiguousarray(((numerator + (1 << 23)) >> 24).astype(np.uint8))


def _smooth_121_predict(frame0: np.ndarray) -> np.ndarray:
    source = _exact_uint8_plane(frame0, "frame0").astype(np.uint16)
    padded_x = np.pad(source, ((0, 0), (1, 1), (0, 0)), mode="edge")
    horizontal = (padded_x[:, :-2] + 2 * padded_x[:, 1:-1] + padded_x[:, 2:] + 2) // 4
    padded_y = np.pad(horizontal, ((1, 1), (0, 0), (0, 0)), mode="edge")
    vertical = (padded_y[:-2] + 2 * padded_y[1:-1] + padded_y[2:] + 2) // 4
    return np.ascontiguousarray(vertical.astype(np.uint8))


def predict_plane(
    frame0: np.ndarray,
    mode: PredictorMode | str | int,
    descriptor: bytes = b"",
) -> np.ndarray:
    """Apply one closed-registry deterministic predictor."""

    selected = _coerce_mode(mode)
    source = _exact_uint8_plane(frame0, "frame0")
    if not isinstance(descriptor, bytes) or len(descriptor) != _descriptor_length(selected):
        raise PredictorResidualError("predictor descriptor length disagrees with its mode")
    if selected is PredictorMode.PREVIOUS_PLANE_COPY:
        return source.copy()
    if selected is PredictorMode.AFFINE6_Q12:
        return _affine6_q12_predict(source, descriptor)
    if selected is PredictorMode.SPATIAL_SMOOTH_121:
        return _smooth_121_predict(source)
    raise PredictorResidualError("predictor mode is outside the closed registry")


def _mode_sequence(
    modes: PredictorMode | str | int | Sequence[PredictorMode | str | int],
    pair_count: int,
) -> tuple[PredictorMode, ...]:
    if isinstance(modes, Sequence) and not isinstance(modes, (str, bytes, bytearray)):
        selected = tuple(_coerce_mode(cast("PredictorMode | str | int", mode)) for mode in modes)
        if len(selected) != pair_count:
            raise PredictorResidualError("predictor mode count must equal pair count")
        return selected
    return (_coerce_mode(cast("PredictorMode | str | int", modes)),) * pair_count


def encode_predictor_residual(
    frame0: np.ndarray,
    frame1: np.ndarray,
    *,
    modes: PredictorMode | str | int | Sequence[PredictorMode | str | int] = PredictorMode.PREVIOUS_PLANE_COPY,
    descriptors: Sequence[bytes] | None = None,
    pair_ids: Sequence[int] | None = None,
) -> bytes:
    """Encode exact two-plane uint8 arrays into a fully consumed payload."""

    y0 = _exact_uint8_planes(frame0, "frame0")
    y1 = _exact_uint8_planes(frame1, "frame1")
    if y0.shape != y1.shape:
        raise PredictorResidualError("frame0 and frame1 arrays must have identical geometry")
    pair_count, height, width, channels = map(int, y0.shape)
    _decoded_working_set_bytes(pair_count, height, width, channels)
    selected_modes = _mode_sequence(modes, pair_count)
    if descriptors is not None and len(descriptors) != pair_count:
        raise PredictorResidualError("descriptor count must equal pair count")
    selected_pair_ids = tuple(range(pair_count)) if pair_ids is None else tuple(pair_ids)
    if len(selected_pair_ids) != pair_count:
        raise PredictorResidualError("pair id count must equal pair count")
    if any(type(pair_id) is not int or pair_id < 0 or pair_id > 0xFFFFFFFF for pair_id in selected_pair_ids):
        raise PredictorResidualError("pair ids must be exact uint32 integers")
    if any(right <= left for left, right in pairwise(selected_pair_ids)):
        raise PredictorResidualError("pair ids must be unique and strictly increasing")

    payload = bytearray(PREFIX.pack(MAGIC, VERSION, CONTENT_CODEC_TAG, pair_count, height, width, channels))
    plane_bytes = height * width * channels
    residual_bytes = plane_bytes * 2
    for pair_index, (pair_id, mode) in enumerate(zip(selected_pair_ids, selected_modes, strict=True)):
        bootstrap_raw = y0[pair_index].tobytes(order="C")
        bootstrap = _compress_content(bootstrap_raw)
        if descriptors is None:
            descriptor = (
                fit_affine6_q12_descriptor(y0[pair_index], y1[pair_index]) if mode is PredictorMode.AFFINE6_Q12 else b""
            )
        else:
            descriptor = descriptors[pair_index]
        if not isinstance(descriptor, bytes) or len(descriptor) != _descriptor_length(mode):
            raise PredictorResidualError("descriptor bytes disagree with predictor mode")
        predictor = predict_plane(y0[pair_index], mode, descriptor)
        residual = (y1[pair_index].astype(np.int16) - predictor.astype(np.int16)).astype("<i2", copy=False)
        residual_decoded = residual.tobytes(order="C")
        residual_raw = _compress_content(residual_decoded)
        if len(bootstrap_raw) != plane_bytes or len(residual_decoded) != residual_bytes:
            raise PredictorResidualError("predictor payload geometry accounting drift")
        payload.extend(
            PAIR_PREFIX.pack(
                pair_id,
                int(mode),
                len(bootstrap),
                len(descriptor),
                len(residual_raw),
                _sha256(bootstrap),
                _sha256(descriptor),
                _sha256(residual_raw),
                _sha256(y1[pair_index].tobytes(order="C")),
            )
        )
        payload.extend(bootstrap)
        payload.extend(descriptor)
        payload.extend(residual_raw)
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise PredictorResidualError("predictor payload exceeds its byte cap")
    return bytes(payload)


def decode_predictor_residual(payload: bytes) -> DecodedPredictorResidual:
    """Parse a payload exactly and reconstruct both planes byte-identically."""

    if not isinstance(payload, bytes):
        raise PredictorResidualError("predictor payload must be immutable bytes")
    if len(payload) < PREFIX.size or len(payload) > MAX_PAYLOAD_BYTES:
        raise PredictorResidualError("predictor payload is truncated or exceeds its byte cap")
    magic, version, content_codec_tag, pair_count, height, width, channels = PREFIX.unpack_from(payload)
    if magic != MAGIC or version != VERSION or content_codec_tag != CONTENT_CODEC_TAG:
        raise PredictorResidualError("predictor payload magic/version/content-codec mismatch")
    if (
        not 1 <= pair_count <= MAX_PAIRS
        or not 1 <= height <= MAX_HEIGHT
        or not 1 <= width <= MAX_WIDTH
        or channels != 3
    ):
        raise PredictorResidualError("predictor payload geometry is outside admitted bounds")
    # This aggregate check precedes pair-header parsing, Brotli invocation, and
    # every decoded-array allocation, so a tiny hostile header cannot inflate a
    # geometry-sized working set.
    _decoded_working_set_bytes(pair_count, height, width, channels)
    plane_bytes = height * width * channels
    residual_bytes_per_pair = plane_bytes * 2
    cursor = PREFIX.size
    frame0_rows: list[np.ndarray] = []
    frame1_rows: list[np.ndarray] = []
    mode_rows: list[str] = []
    pair_ids: list[int] = []
    descriptors: list[bytes] = []
    bootstrap_total = 0
    descriptor_total = 0
    residual_total = 0
    previous_pair_id = -1
    for _pair_index in range(pair_count):
        if cursor + PAIR_PREFIX.size > len(payload):
            raise PredictorResidualError("predictor payload has a truncated pair header")
        (
            pair_id,
            mode_value,
            bootstrap_length,
            descriptor_length,
            residual_length,
            bootstrap_sha,
            descriptor_sha,
            residual_sha,
            reconstructed_sha,
        ) = PAIR_PREFIX.unpack_from(payload, cursor)
        cursor += PAIR_PREFIX.size
        if pair_id <= previous_pair_id:
            raise PredictorResidualError("predictor pair ids are not unique and strictly increasing")
        previous_pair_id = pair_id
        mode = _coerce_mode(mode_value)
        if (
            bootstrap_length < 1
            or descriptor_length != _descriptor_length(mode)
            or residual_length < 1
            or bootstrap_length > MAX_PAYLOAD_BYTES
            or residual_length > MAX_PAYLOAD_BYTES
        ):
            raise PredictorResidualError("predictor pair length/geometry declaration drift")
        end = cursor + bootstrap_length + descriptor_length + residual_length
        if end > len(payload):
            raise PredictorResidualError("predictor payload has a truncated pair body")
        bootstrap = payload[cursor : cursor + bootstrap_length]
        cursor += bootstrap_length
        descriptor = payload[cursor : cursor + descriptor_length]
        cursor += descriptor_length
        residual_raw = payload[cursor : cursor + residual_length]
        cursor += residual_length
        if (
            _sha256(bootstrap) != bootstrap_sha
            or _sha256(descriptor) != descriptor_sha
            or _sha256(residual_raw) != residual_sha
        ):
            raise PredictorResidualError("predictor pair component hash custody failure")
        bootstrap_decoded = _decompress_content(bootstrap, expected_bytes=plane_bytes, label="frame0 bootstrap")
        residual_decoded = _decompress_content(residual_raw, expected_bytes=residual_bytes_per_pair, label="residual")
        y0 = np.frombuffer(bootstrap_decoded, dtype=np.uint8).reshape(height, width, channels).copy()
        residual = np.frombuffer(residual_decoded, dtype="<i2").reshape(height, width, channels)
        predictor = predict_plane(y0, mode, descriptor)
        reconstructed = predictor.astype(np.int32) + residual.astype(np.int32)
        if np.any((reconstructed < 0) | (reconstructed > 255)):
            raise PredictorResidualError("predictor residual reconstructs outside uint8")
        y1 = np.ascontiguousarray(reconstructed.astype(np.uint8))
        if _sha256(y1.tobytes(order="C")) != reconstructed_sha:
            raise PredictorResidualError("predictor reconstructed frame1 hash custody failure")
        frame0_rows.append(y0)
        frame1_rows.append(y1)
        mode_rows.append(MODE_IDS[mode])
        pair_ids.append(pair_id)
        descriptors.append(descriptor)
        bootstrap_total += bootstrap_length
        descriptor_total += descriptor_length
        residual_total += residual_length
    if cursor != len(payload):
        raise PredictorResidualError("predictor payload has trailing data")
    framing = PREFIX.size + pair_count * PAIR_PREFIX.size
    if framing + bootstrap_total + descriptor_total + residual_total != len(payload):
        raise PredictorResidualError("predictor payload internal byte accounting drift")
    return DecodedPredictorResidual(
        frame0=np.stack(frame0_rows),
        frame1=np.stack(frame1_rows),
        pair_ids=tuple(pair_ids),
        modes=tuple(mode_rows),
        descriptors=tuple(descriptors),
        accounting=PredictorResidualAccounting(
            payload_bytes=len(payload),
            framing_bytes=framing,
            bootstrap_bytes=bootstrap_total,
            descriptor_bytes=descriptor_total,
            residual_bytes=residual_total,
            decoded_bootstrap_bytes=pair_count * plane_bytes,
            decoded_residual_bytes=pair_count * residual_bytes_per_pair,
        ),
    )


__all__ = [
    "AFFINE6",
    "AFFINE6_Q12_ID",
    "CODEC_ID",
    "CONTENT_CODEC_ID",
    "CONTENT_CODEC_TAG",
    "MAGIC",
    "MAX_DECODED_WORKING_SET_BYTES",
    "MODE_BY_ID",
    "MODE_IDS",
    "PAIR_PREFIX",
    "PREFIX",
    "PREVIOUS_PLANE_COPY_ID",
    "Q12_ONE",
    "SPATIAL_SMOOTH_121_ID",
    "VERSION",
    "DecodedPredictorResidual",
    "PredictorMode",
    "PredictorResidualAccounting",
    "PredictorResidualError",
    "decode_predictor_residual",
    "encode_predictor_residual",
    "fit_affine6_q12_descriptor",
    "predict_plane",
]
