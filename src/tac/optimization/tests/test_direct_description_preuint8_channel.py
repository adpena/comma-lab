from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.direct_description_preuint8_channel import (
    _BAYER8,
    PreUint8Q8ProgramV1,
    SparseQ8CorrectionV1,
    TemplateQ8CorrectionV1,
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
