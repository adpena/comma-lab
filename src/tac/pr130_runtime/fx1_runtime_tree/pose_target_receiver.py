#!/usr/bin/env python3
"""Counted PZ2-target receiver for the reproduced PR130 pose carrier.

PZ3R keeps PR130's deployed basis, consumes the six quantized PZ2 target-code
streams, and reconstructs the twelve deployed coefficient-code streams from a
counted fixed-point predictor plus an exact counted residual.  The public
receiver never loads PoseNet or any ground-truth artifact.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import brotli
import carrier_codec
import numpy as np

MAGIC = b"PZ3R"
VERSION = 1
N = 600
TARGET_DIMS = 6
CARRIER_DIMS = 12
BASIS_SHAPE = (CARRIER_DIMS, 3, 24, 32)
BASIS_COUNT = int(np.prod(BASIS_SHAPE))

PZ2_MAGIC = b"PZ2TGT1\0"
PZ2_HEADER = struct.Struct("<8sBBHB")
PZ2_STREAM_HEADER = struct.Struct("<BffI")
PZ2_DIRECT = 0

FEATURE_TARGET = 1
FEATURE_TARGET_QUADRATIC = 2
FEATURE_TARGET_PREVIOUS = 3
FEATURE_TARGET_QUADRATIC_PREVIOUS = 4
FEATURE_COUNTS = {
    FEATURE_TARGET: 6,
    FEATURE_TARGET_QUADRATIC: 27,
    FEATURE_TARGET_PREVIOUS: 18,
    FEATURE_TARGET_QUADRATIC_PREVIOUS: 39,
}

HEADER = struct.Struct("<4sBBBBIIII32s")
MODEL_HEADER = struct.Struct("<BBH")
RESIDUAL_HEADER = struct.Struct("<I")


@dataclass(frozen=True)
class Predictor:
    """Integer fixed-point target-to-coefficient predictor."""

    feature_mode: int
    shift: int
    feature_offsets: np.ndarray
    output_offsets: np.ndarray
    weights: np.ndarray


def _sha256_array(value: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest()


def _signed_mod12(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    return (((array + 2048) & 0xFFF) - 2048).astype(np.int32)


def _round_shift_away(value: np.ndarray, shift: int) -> np.ndarray:
    if not 1 <= shift <= 30:
        raise ValueError("fixed-point shift must be in 1..30")
    value = np.asarray(value, dtype=np.int64)
    half = 1 << (shift - 1)
    magnitude = (np.abs(value) + half) >> shift
    return np.where(value < 0, -magnitude, magnitude).astype(np.int64)


def _deserialize_direct_codes(raw: bytes, count: int) -> np.ndarray:
    expected = count * np.dtype("<u2").itemsize
    if len(raw) != expected:
        raise ValueError("PZ2 direct stream length mismatch")
    return np.frombuffer(raw, dtype="<u2").astype(np.int32)


def decode_pz2_packet(payload: bytes | memoryview) -> tuple[np.ndarray, np.ndarray]:
    """Return PZ2 integer target codes and their decoded float values."""
    payload = bytes(payload)
    if len(payload) < PZ2_HEADER.size:
        raise ValueError("truncated PZ2 packet")
    magic, version, method, count, dimensions = PZ2_HEADER.unpack_from(payload)
    if (
        magic != PZ2_MAGIC
        or version != 1
        or method != PZ2_DIRECT
        or count != N
        or dimensions != TARGET_DIMS
    ):
        raise ValueError("unsupported PZ2 target packet")
    cursor = PZ2_HEADER.size
    code_columns: list[np.ndarray] = []
    value_columns: list[np.ndarray] = []
    for _ in range(TARGET_DIMS):
        if cursor + PZ2_STREAM_HEADER.size > len(payload):
            raise ValueError("truncated PZ2 stream header")
        bits, low, step, compressed_bytes = PZ2_STREAM_HEADER.unpack_from(
            payload, cursor
        )
        cursor += PZ2_STREAM_HEADER.size
        if not 0 <= bits <= 16 or compressed_bytes <= 0:
            raise ValueError("invalid PZ2 stream declaration")
        end = cursor + compressed_bytes
        if end > len(payload):
            raise ValueError("truncated PZ2 compressed stream")
        raw = brotli.decompress(payload[cursor:end])
        cursor = end
        codes = _deserialize_direct_codes(raw, N)
        maximum = (1 << bits) - 1 if bits else 0
        if np.any(codes < 0) or np.any(codes > maximum):
            raise ValueError("PZ2 target code exceeds declared precision")
        code_columns.append(codes)
        value_columns.append(
            np.float64(low) + codes.astype(np.float64) * np.float64(step)
        )
    if cursor != len(payload):
        raise ValueError("PZ2 packet has trailing bytes")
    return np.stack(code_columns, axis=1), np.stack(value_columns, axis=1)


def _quadratic_features(target_codes: np.ndarray) -> np.ndarray:
    target_codes = np.asarray(target_codes, dtype=np.int64)
    columns = [target_codes]
    products = [
        target_codes[:, left] * target_codes[:, right]
        for left in range(TARGET_DIMS)
        for right in range(left, TARGET_DIMS)
    ]
    columns.append(np.stack(products, axis=1))
    return np.concatenate(columns, axis=1)


def feature_matrix(
    target_codes: np.ndarray,
    feature_mode: int,
    previous_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the declared integer feature matrix."""
    target_codes = np.asarray(target_codes, dtype=np.int64)
    if target_codes.shape != (N, TARGET_DIMS):
        raise ValueError("target codes must have shape [600, 6]")
    if feature_mode in (FEATURE_TARGET, FEATURE_TARGET_PREVIOUS):
        features = target_codes
    elif feature_mode in (
        FEATURE_TARGET_QUADRATIC,
        FEATURE_TARGET_QUADRATIC_PREVIOUS,
    ):
        features = _quadratic_features(target_codes)
    else:
        raise ValueError("unsupported PZ3R feature mode")
    if feature_mode in (FEATURE_TARGET_PREVIOUS, FEATURE_TARGET_QUADRATIC_PREVIOUS):
        if previous_coefficients is None:
            raise ValueError("previous-coefficient features are required")
        previous_coefficients = np.asarray(previous_coefficients, dtype=np.int64)
        if previous_coefficients.shape != (N, CARRIER_DIMS):
            raise ValueError("previous coefficients must have shape [600, 12]")
        previous = np.zeros_like(previous_coefficients)
        previous[1:] = previous_coefficients[:-1]
        features = np.concatenate([features, previous], axis=1)
    expected = FEATURE_COUNTS.get(feature_mode)
    if expected is None or features.shape != (N, expected):
        raise RuntimeError("PZ3R feature construction mismatch")
    return features.astype(np.int64, copy=False)


def predict_coefficients(
    target_codes: np.ndarray,
    predictor: Predictor,
    previous_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Run the deterministic counted predictor."""
    features = feature_matrix(
        target_codes,
        predictor.feature_mode,
        previous_coefficients,
    )
    feature_count = FEATURE_COUNTS[predictor.feature_mode]
    offsets = np.asarray(predictor.feature_offsets, dtype=np.int64)
    outputs = np.asarray(predictor.output_offsets, dtype=np.int64)
    weights = np.asarray(predictor.weights, dtype=np.int64)
    if offsets.shape != (feature_count,):
        raise ValueError("predictor feature-offset shape mismatch")
    if outputs.shape != (CARRIER_DIMS,):
        raise ValueError("predictor output-offset shape mismatch")
    if weights.shape != (feature_count, CARRIER_DIMS):
        raise ValueError("predictor weight shape mismatch")
    accumulators = (features - offsets[None]) @ weights
    predicted = outputs[None] + _round_shift_away(
        accumulators, predictor.shift
    )
    return _signed_mod12(predicted)


def serialize_predictor(predictor: Predictor, coefficient_scales: np.ndarray) -> bytes:
    feature_count = FEATURE_COUNTS.get(predictor.feature_mode)
    if feature_count is None:
        raise ValueError("unsupported PZ3R feature mode")
    scales = np.asarray(coefficient_scales, dtype="<f4").reshape(-1)
    offsets = np.asarray(predictor.feature_offsets, dtype="<i4").reshape(-1)
    outputs = np.asarray(predictor.output_offsets, dtype="<i4").reshape(-1)
    weights = np.asarray(predictor.weights, dtype="<i4")
    if scales.shape != (CARRIER_DIMS,):
        raise ValueError("coefficient scales must have shape [12]")
    if offsets.shape != (feature_count,) or outputs.shape != (CARRIER_DIMS,):
        raise ValueError("predictor offset shape mismatch")
    if weights.shape != (feature_count, CARRIER_DIMS):
        raise ValueError("predictor weight shape mismatch")
    return b"".join(
        [
            MODEL_HEADER.pack(predictor.feature_mode, predictor.shift, feature_count),
            scales.tobytes(),
            offsets.tobytes(),
            outputs.tobytes(),
            weights.tobytes(),
        ]
    )


def deserialize_predictor(payload: bytes | memoryview) -> tuple[Predictor, np.ndarray]:
    payload = memoryview(payload)
    if len(payload) < MODEL_HEADER.size:
        raise ValueError("truncated PZ3R predictor")
    feature_mode, shift, feature_count = MODEL_HEADER.unpack(payload[: MODEL_HEADER.size])
    if FEATURE_COUNTS.get(feature_mode) != feature_count or not 1 <= shift <= 30:
        raise ValueError("invalid PZ3R predictor declaration")
    expected = MODEL_HEADER.size + 4 * (
        CARRIER_DIMS + feature_count + CARRIER_DIMS + feature_count * CARRIER_DIMS
    )
    if len(payload) != expected:
        raise ValueError("PZ3R predictor length mismatch")
    cursor = MODEL_HEADER.size
    scales_bytes = CARRIER_DIMS * 4
    coefficient_scales = np.frombuffer(
        payload[cursor : cursor + scales_bytes], dtype="<f4"
    ).copy()
    cursor += scales_bytes
    offset_bytes = feature_count * 4
    feature_offsets = np.frombuffer(
        payload[cursor : cursor + offset_bytes], dtype="<i4"
    ).copy()
    cursor += offset_bytes
    output_bytes = CARRIER_DIMS * 4
    output_offsets = np.frombuffer(
        payload[cursor : cursor + output_bytes], dtype="<i4"
    ).copy()
    cursor += output_bytes
    weights = np.frombuffer(payload[cursor:], dtype="<i4").copy().reshape(
        feature_count, CARRIER_DIMS
    )
    return (
        Predictor(feature_mode, shift, feature_offsets, output_offsets, weights),
        coefficient_scales,
    )


def encode_residual(residual: np.ndarray) -> bytes:
    residual = _signed_mod12(residual)
    unsigned = carrier_codec._zigzag_signed(residual, carrier_codec.COEFF_BITS)
    ks, payload, bit_count = carrier_codec._encode_rice(unsigned)
    return RESIDUAL_HEADER.pack(bit_count) + ks.tobytes() + payload


def decode_residual(payload: bytes | memoryview) -> np.ndarray:
    payload = memoryview(payload)
    prefix = RESIDUAL_HEADER.size + CARRIER_DIMS
    if len(payload) <= prefix:
        raise ValueError("truncated PZ3R residual")
    bit_count = RESIDUAL_HEADER.unpack(payload[: RESIDUAL_HEADER.size])[0]
    ks = np.frombuffer(
        payload[RESIDUAL_HEADER.size:prefix], dtype=np.uint8
    ).copy()
    unsigned = carrier_codec._decode_rice(
        ks,
        payload[prefix:],
        bit_count,
        N,
        CARRIER_DIMS,
    )
    return carrier_codec._unzigzag_unsigned(
        unsigned, carrier_codec.COEFF_BITS
    ).reshape(N, CARRIER_DIMS)


def encode_pose_target_carrier(
    *,
    basis_component: bytes,
    target_packet: bytes,
    predictor: Predictor,
    coefficient_scales: np.ndarray,
    absolute_coefficients: np.ndarray,
) -> bytes:
    """Encode a counted PZ3R packet that exactly reconstructs PR130 arrays."""
    target_codes, _ = decode_pz2_packet(target_packet)
    absolute = np.asarray(absolute_coefficients, dtype=np.int32)
    if absolute.shape != (N, CARRIER_DIMS):
        raise ValueError("absolute coefficients must have shape [600, 12]")
    predicted = predict_coefficients(target_codes, predictor, absolute)
    residual = _signed_mod12(absolute.astype(np.int64) - predicted.astype(np.int64))
    model = serialize_predictor(predictor, coefficient_scales)
    residual_payload = encode_residual(residual)
    return b"".join(
        [
            HEADER.pack(
                MAGIC,
                VERSION,
                predictor.feature_mode,
                predictor.shift,
                0,
                len(basis_component),
                len(target_packet),
                len(model),
                len(residual_payload),
                _sha256_array(absolute.astype("<i4")),
            ),
            basis_component,
            target_packet,
            model,
            residual_payload,
        ]
    )


def _decode_basis_component(component: bytes | memoryview) -> tuple[np.ndarray, np.ndarray]:
    component = memoryview(component)
    minimum = 4 + CARRIER_DIMS * 4 + carrier_codec.ALPHABET_SIZE
    if len(component) <= minimum:
        raise ValueError("truncated PZ3R basis component")
    bit_count = struct.unpack_from("<I", component)[0]
    cursor = 4
    scale_bytes = CARRIER_DIMS * 4
    scales = np.frombuffer(component[cursor : cursor + scale_bytes], dtype="<f4").copy()
    cursor += scale_bytes
    lengths = np.frombuffer(
        component[cursor : cursor + carrier_codec.ALPHABET_SIZE], dtype=np.uint8
    ).copy()
    cursor += carrier_codec.ALPHABET_SIZE
    unsigned = carrier_codec._decode_huffman(
        lengths,
        component[cursor:],
        bit_count,
        BASIS_COUNT,
    )
    codes = carrier_codec._unzigzag_unsigned(unsigned, carrier_codec.BASIS_BITS)
    basis = codes.reshape(BASIS_SHAPE).astype(np.float32)
    basis *= scales[:, None, None, None]
    return basis, codes.reshape(BASIS_SHAPE)


def decode_pose_target_carrier(
    blob: bytes | memoryview,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode PZ3R to PR130 basis and scaled coefficient arrays."""
    blob = memoryview(blob)
    if len(blob) < HEADER.size:
        raise ValueError("truncated PZ3R carrier")
    (
        magic,
        version,
        feature_mode,
        shift,
        reserved,
        basis_bytes,
        target_bytes,
        model_bytes,
        residual_bytes,
        expected_coefficients_sha256,
    ) = HEADER.unpack(blob[: HEADER.size])
    if (
        magic != MAGIC
        or version != VERSION
        or reserved
        or FEATURE_COUNTS.get(feature_mode) is None
        or not 1 <= shift <= 30
    ):
        raise ValueError("unsupported PZ3R carrier header")
    expected = HEADER.size + basis_bytes + target_bytes + model_bytes + residual_bytes
    if len(blob) != expected:
        raise ValueError("PZ3R carrier length mismatch")
    cursor = HEADER.size
    basis_component = blob[cursor : cursor + basis_bytes]
    cursor += basis_bytes
    target_packet = blob[cursor : cursor + target_bytes]
    cursor += target_bytes
    model_payload = blob[cursor : cursor + model_bytes]
    cursor += model_bytes
    residual_payload = blob[cursor : cursor + residual_bytes]

    basis, _ = _decode_basis_component(basis_component)
    target_codes, _ = decode_pz2_packet(target_packet)
    predictor, coefficient_scales = deserialize_predictor(model_payload)
    if predictor.feature_mode != feature_mode or predictor.shift != shift:
        raise ValueError("PZ3R header/predictor mismatch")
    residual = decode_residual(residual_payload)

    # Previous-coefficient feature modes are causal: each row consumes only the
    # already reconstructed prior row.  Target-only modes decode in one pass.
    if feature_mode in (FEATURE_TARGET_PREVIOUS, FEATURE_TARGET_QUADRATIC_PREVIOUS):
        absolute = np.zeros((N, CARRIER_DIMS), dtype=np.int32)
        for frame in range(N):
            predicted = predict_coefficients(
                target_codes, predictor, absolute
            )[frame]
            absolute[frame] = _signed_mod12(
                predicted.astype(np.int64) + residual[frame].astype(np.int64)
            )
    else:
        predicted = predict_coefficients(target_codes, predictor)
        absolute = _signed_mod12(
            predicted.astype(np.int64) + residual.astype(np.int64)
        )
    if _sha256_array(absolute.astype("<i4")) != expected_coefficients_sha256:
        raise ValueError("PZ3R reconstructed coefficient hash mismatch")
    coefficients = absolute.astype(np.float32) * coefficient_scales[None]
    return basis.astype(np.float32), coefficients.astype(np.float32)
