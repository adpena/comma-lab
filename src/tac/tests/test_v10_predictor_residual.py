"""Behavioral tests for the V10 two-plane predictor/residual codec."""

from __future__ import annotations

import hashlib
from fractions import Fraction

import brotli
import numpy as np
import pytest

from tac.codec.v10_predictor_residual import (
    AFFINE6,
    CONTENT_CODEC_ID,
    CONTENT_CODEC_TAG,
    MAGIC,
    MAX_DECODED_WORKING_SET_BYTES,
    MAX_HEIGHT,
    MAX_PAIRS,
    MAX_WIDTH,
    PAIR_PREFIX,
    PREFIX,
    VERSION,
    PredictorMode,
    PredictorResidualError,
    _round_fraction_ties_to_even,
    decode_predictor_residual,
    encode_predictor_residual,
    fit_affine6_q12_descriptor,
    predict_plane,
)


def _planes(pair_count: int = 3, height: int = 5, width: int = 7) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(pair_count * height * width * 3, dtype=np.uint16)
    frame0 = ((values * 29 + 17) % 256).astype(np.uint8).reshape(pair_count, height, width, 3)
    frame1 = np.roll(frame0, shift=-1, axis=2)
    frame1[:, :, -1] = frame0[:, :, -1]
    return frame0, frame1


def test_copy_and_smooth_predictors_are_exact_integer_operations() -> None:
    frame0, _ = _planes(pair_count=1, height=3, width=4)
    source = frame0[0]
    np.testing.assert_array_equal(
        predict_plane(source, "previous-plane-copy.v1"),
        source,
    )

    padded_x = np.pad(source.astype(np.uint16), ((0, 0), (1, 1), (0, 0)), mode="edge")
    horizontal = (padded_x[:, :-2] + 2 * padded_x[:, 1:-1] + padded_x[:, 2:] + 2) // 4
    padded_y = np.pad(horizontal, ((1, 1), (0, 0), (0, 0)), mode="edge")
    expected = ((padded_y[:-2] + 2 * padded_y[1:-1] + padded_y[2:] + 2) // 4).astype(np.uint8)
    np.testing.assert_array_equal(
        predict_plane(source, "spatial-smooth-121.v1"),
        expected,
    )


def test_affine6_fixed_point_bilinear_warp_applies_descriptor() -> None:
    frame0, _ = _planes(pair_count=1, height=4, width=6)
    source = frame0[0]
    # input_x = output_x + 1; clamping repeats the final source column.
    descriptor = AFFINE6.pack(1 << 12, 0, 0, 0, 0, 0)
    predicted = predict_plane(source, PredictorMode.AFFINE6_Q12, descriptor)
    expected = np.concatenate((source[:, 1:], source[:, -1:]), axis=1)
    np.testing.assert_array_equal(predicted, expected)


def test_affine6_fit_is_six_q12_values_and_repeated_byte_deterministic() -> None:
    frame0, frame1 = _planes(pair_count=1, height=8, width=9)
    first = fit_affine6_q12_descriptor(frame0[0], frame1[0])
    second = fit_affine6_q12_descriptor(frame0[0], frame1[0])
    assert len(first) == AFFINE6.size == 24
    assert first == second
    assert AFFINE6.unpack(first) != (0, 0, 0, 0, 0, 0)
    assert hashlib.sha256(first).hexdigest() == "c049baee67da8c7e91d65581c7ecabe3a0fae1a186b551683531a1b229d6a46f"


def test_affine6_fit_has_no_lapack_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    frame0, frame1 = _planes(pair_count=1, height=8, width=9)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("LAPACK path must not be called")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    descriptor = fit_affine6_q12_descriptor(frame0[0], frame1[0])
    assert hashlib.sha256(descriptor).hexdigest() == "c049baee67da8c7e91d65581c7ecabe3a0fae1a186b551683531a1b229d6a46f"


def test_affine6_singular_system_uses_canonical_zero_free_variables() -> None:
    frame0 = np.full((4, 6, 3), 17, dtype=np.uint8)
    frame1 = np.full((4, 6, 3), 29, dtype=np.uint8)

    # A constant source has a rank-zero design matrix, so every variable is free.
    assert AFFINE6.unpack(fit_affine6_q12_descriptor(frame0, frame1)) == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Fraction(1, 2), 0),
        (Fraction(3, 2), 2),
        (Fraction(5, 2), 2),
        (Fraction(-1, 2), 0),
        (Fraction(-3, 2), -2),
    ],
)
def test_exact_q12_rounding_is_ties_to_even(value: Fraction, expected: int) -> None:
    assert _round_fraction_ties_to_even(value) == expected


@pytest.mark.parametrize(
    "mode",
    [
        PredictorMode.PREVIOUS_PLANE_COPY,
        PredictorMode.AFFINE6_Q12,
        PredictorMode.SPATIAL_SMOOTH_121,
    ],
)
def test_payload_roundtrips_both_planes_and_accounts_every_byte(mode: PredictorMode) -> None:
    frame0, frame1 = _planes()
    first = encode_predictor_residual(frame0, frame1, modes=mode)
    second = encode_predictor_residual(frame0, frame1, modes=mode)
    assert first == second
    decoded = decode_predictor_residual(first)
    np.testing.assert_array_equal(decoded.frame0, frame0)
    np.testing.assert_array_equal(decoded.frame1, frame1)
    assert decoded.accounting.payload_bytes == len(first)
    assert decoded.accounting.framing_bytes == PREFIX.size + len(frame0) * PAIR_PREFIX.size
    assert decoded.accounting.decoded_bootstrap_bytes == frame0.nbytes
    assert decoded.accounting.decoded_residual_bytes == frame1.nbytes * 2
    expected_descriptor_bytes = len(frame0) * AFFINE6.size if mode is PredictorMode.AFFINE6_Q12 else 0
    assert decoded.accounting.descriptor_bytes == expected_descriptor_bytes
    assert decoded.accounting.conditional_bytes == expected_descriptor_bytes + decoded.accounting.residual_bytes
    assert (
        decoded.accounting.framing_bytes
        + decoded.accounting.bootstrap_bytes
        + decoded.accounting.descriptor_bytes
        + decoded.accounting.residual_bytes
        == len(first)
    )


def test_payload_supports_per_pair_closed_modes() -> None:
    frame0, frame1 = _planes()
    modes = [
        "previous-plane-copy.v1",
        "affine6-q12.v1",
        "spatial-smooth-121.v1",
    ]
    decoded = decode_predictor_residual(encode_predictor_residual(frame0, frame1, modes=modes, pair_ids=[12, 17, 99]))
    assert decoded.pair_ids == (12, 17, 99)
    assert decoded.modes == tuple(modes)
    assert tuple(map(len, decoded.descriptors)) == (0, 24, 0)


def test_payload_explicitly_tags_and_content_prices_brotli_q11_streams() -> None:
    frame0 = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    frame1 = np.full_like(frame0, 1)
    payload = encode_predictor_residual(frame0, frame1)
    assert PREFIX.unpack_from(payload)[2] == CONTENT_CODEC_TAG
    accounting = decode_predictor_residual(payload).accounting
    assert accounting.bootstrap_bytes < accounting.decoded_bootstrap_bytes
    assert accounting.residual_bytes < accounting.decoded_residual_bytes


def test_header_bomb_refuses_before_decompression_and_n48_geometry_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("decompression must not run during aggregate header validation")

    monkeypatch.setattr("tac.codec.v10_predictor_residual._decompress_content", forbidden)
    bomb = PREFIX.pack(
        MAGIC,
        VERSION,
        CONTENT_CODEC_TAG,
        MAX_PAIRS,
        MAX_HEIGHT,
        MAX_WIDTH,
        3,
    )
    with pytest.raises(PredictorResidualError, match="aggregate working set"):
        decode_predictor_residual(bomb)

    n48_working_bytes = 48 * 384 * 512 * 3 * 4
    assert n48_working_bytes < MAX_DECODED_WORKING_SET_BYTES
    n48_header = PREFIX.pack(MAGIC, VERSION, CONTENT_CODEC_TAG, 48, 384, 512, 3)
    with pytest.raises(PredictorResidualError, match="truncated pair header"):
        decode_predictor_residual(n48_header)
    assert CONTENT_CODEC_ID == "brotli-q11.v1"


def test_parser_refuses_unknown_mode_length_hash_truncation_and_trailing_data() -> None:
    frame0, frame1 = _planes(pair_count=1)
    payload = encode_predictor_residual(frame0, frame1)

    unknown_mode = bytearray(payload)
    unknown_mode[PREFIX.size + 4] = 255
    with pytest.raises(PredictorResidualError, match="unknown predictor mode"):
        decode_predictor_residual(bytes(unknown_mode))

    wrong_descriptor_length = bytearray(payload)
    fields = list(PAIR_PREFIX.unpack_from(wrong_descriptor_length, PREFIX.size))
    fields[3] = 1
    PAIR_PREFIX.pack_into(wrong_descriptor_length, PREFIX.size, *fields)
    with pytest.raises(PredictorResidualError, match="length/geometry"):
        decode_predictor_residual(bytes(wrong_descriptor_length))

    corrupt_bootstrap = bytearray(payload)
    corrupt_bootstrap[PREFIX.size + PAIR_PREFIX.size] ^= 1
    with pytest.raises(PredictorResidualError, match="component hash"):
        decode_predictor_residual(bytes(corrupt_bootstrap))
    with pytest.raises(PredictorResidualError, match="truncated"):
        decode_predictor_residual(payload[:-1])
    with pytest.raises(PredictorResidualError, match="trailing"):
        decode_predictor_residual(payload + b"x")


def test_parser_refuses_residual_that_reconstructs_outside_uint8() -> None:
    frame0 = np.full((1, 1, 1, 3), 255, dtype=np.uint8)
    original = encode_predictor_residual(frame0, frame0)
    fields = list(PAIR_PREFIX.unpack_from(original, PREFIX.size))
    body_offset = PREFIX.size + PAIR_PREFIX.size
    bootstrap = original[body_offset : body_offset + fields[2]]
    descriptor = original[body_offset + fields[2] : body_offset + fields[2] + fields[3]]
    residual_offset = body_offset + fields[2] + fields[3]
    residual = bytearray(brotli.decompress(original[residual_offset:]))
    residual[0:2] = (1).to_bytes(2, "little", signed=True)
    compressed_residual = bytes(brotli.compress(bytes(residual), quality=11))
    fields[4] = len(compressed_residual)
    fields[7] = hashlib.sha256(compressed_residual).digest()
    payload = original[: PREFIX.size] + PAIR_PREFIX.pack(*fields) + bootstrap + descriptor + compressed_residual
    with pytest.raises(PredictorResidualError, match="outside uint8"):
        decode_predictor_residual(payload)


def test_encoder_refuses_dtype_geometry_mode_and_descriptor_drift() -> None:
    frame0, frame1 = _planes(pair_count=1)
    with pytest.raises(PredictorResidualError, match="exact uint8"):
        encode_predictor_residual(frame0.astype(np.int16), frame1)
    with pytest.raises(PredictorResidualError, match="identical geometry"):
        encode_predictor_residual(frame0, frame1[:, :, :-1])
    with pytest.raises(PredictorResidualError, match="unknown predictor mode"):
        encode_predictor_residual(frame0, frame1, modes="not-a-mode")
    with pytest.raises(PredictorResidualError, match="exact uint32"):
        encode_predictor_residual(frame0, frame1, pair_ids=[True])
    two_frame0, two_frame1 = _planes(pair_count=2)
    with pytest.raises(PredictorResidualError, match="strictly increasing"):
        encode_predictor_residual(two_frame0, two_frame1, pair_ids=[2, 2])
    with pytest.raises(PredictorResidualError, match="descriptor bytes"):
        encode_predictor_residual(
            frame0,
            frame1,
            modes=PredictorMode.AFFINE6_Q12,
            descriptors=[b""],
        )
    with pytest.raises(PredictorResidualError, match="descriptor length"):
        predict_plane(frame0[0], PredictorMode.AFFINE6_Q12, b"short")
