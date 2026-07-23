from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.direct_description_preuint8_channel import (
    _BAYER8,
    PreUint8Q8ProgramV1,
    SparseQ8CorrectionV1,
    TemplateQ8CorrectionV1,
    _resize_null_sigma_delta_round_q8,
    _scorer_resize_operator,
    decode_preuint8_q8_program,
    encode_preuint8_q8_program,
)


def test_bayer8_is_a_complete_deterministic_threshold_permutation() -> None:
    assert _BAYER8.shape == (8, 8)
    assert np.array_equal(np.sort(_BAYER8.reshape(-1)), np.arange(64))


def test_q8_ordered_rounding_preserves_integer_quanta() -> None:
    values = np.asarray([-4, -1, 0, 1, 4], dtype=np.int32) * 256
    thresholds = (_BAYER8.reshape(-1).astype(np.int32) * 4 + 2)[:, None]
    rounded = np.floor_divide(values[None, :] + thresholds, 256)
    assert np.all(rounded == values[None, :] // 256)


def test_preuint8_program_is_canonical_and_byte_identical_on_parseback() -> None:
    program = PreUint8Q8ProgramV1(
        templates=(TemplateQ8CorrectionV1(53, 2, (-128, 0, 128)),),
        sparse=(SparseQ8CorrectionV1(53, 1, 10, 11, (32, -64, 96)),),
        dither_mode="bayer8",
        dither_seed=123,
    )
    payload = encode_preuint8_q8_program(program)
    assert decode_preuint8_q8_program(payload) == program
    assert encode_preuint8_q8_program(decode_preuint8_q8_program(payload)) == payload


def test_preuint8_program_refuses_trailing_or_noncanonical_records() -> None:
    program = PreUint8Q8ProgramV1(
        sparse=(
            SparseQ8CorrectionV1(53, 0, 0, 0, (1, 0, 0)),
            SparseQ8CorrectionV1(53, 0, 0, 1, (0, 1, 0)),
        )
    )
    with pytest.raises(DirectDescriptionError, match="trailing"):
        decode_preuint8_q8_program(encode_preuint8_q8_program(program) + b"\x00")
    with pytest.raises(DirectDescriptionError, match="canonical-order"):
        PreUint8Q8ProgramV1(sparse=tuple(reversed(program.sparse)))


def test_resize_null_sigma_delta_mode_roundtrips_without_extra_payload() -> None:
    program = PreUint8Q8ProgramV1(dither_mode="resize_null_sigma_delta")
    payload = encode_preuint8_q8_program(program)
    assert len(payload) == len(
        encode_preuint8_q8_program(PreUint8Q8ProgramV1(dither_mode="bayer8"))
    )
    assert decode_preuint8_q8_program(payload) == program


def test_resize_null_sigma_delta_reduces_exact_resize_numerator_error() -> None:
    operator = _scorer_resize_operator()
    q8 = np.zeros((1, operator.camera_h, operator.camera_w, 1), dtype=np.int32)
    rows = operator.row_supports[0].indices
    cols = operator.col_supports[0].indices
    q8[np.ix_((0,), rows, cols, (0,))] = 128
    shaped = _resize_null_sigma_delta_round_q8(q8)[0]
    uniform = np.floor_divide(q8[0] + 128, 256).astype(np.uint8)
    continuous = operator.apply(q8[0].astype(np.float64) / 256.0)
    shaped_error = abs(float(operator.apply(shaped)[0, 0, 0] - continuous[0, 0, 0]))
    uniform_error = abs(float(operator.apply(uniform)[0, 0, 0] - continuous[0, 0, 0]))
    assert shaped_error < uniform_error
    assert np.count_nonzero(shaped) < np.count_nonzero(uniform)


def test_resize_null_sigma_delta_preserves_integer_q8_values() -> None:
    operator = _scorer_resize_operator()
    q8 = np.zeros((1, operator.camera_h, operator.camera_w, 1), dtype=np.int32)
    q8[:, 0, 0, 0] = 7 * 256
    shaped = _resize_null_sigma_delta_round_q8(q8)
    assert shaped[0, 0, 0, 0] == 7
    assert np.count_nonzero(shaped) == 1
