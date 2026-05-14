# SPDX-License-Identifier: MIT
from __future__ import annotations

import struct

import numpy as np
import pytest

from tac.arithmetic_terminal import decode_hyperprior, encode_hyperprior


def _atbh_blob() -> bytes:
    qints = np.array([-1, 0, 1, 0, -1, 1, 0], dtype=np.int8)
    sigma = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    return encode_hyperprior(
        qints=qints,
        num_symbols=3,
        offset=1,
        sigma_per_block=sigma,
        block_size=3,
    )


def test_hyperprior_roundtrip() -> None:
    qints = np.array([-1, 0, 1, 0, -1, 1, 0], dtype=np.int8)
    blob = _atbh_blob()

    decoded = decode_hyperprior(blob=blob, expected_dtype=np.int8)

    assert np.array_equal(decoded, qints)


def test_hyperprior_rejects_trailing_bytes() -> None:
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_hyperprior(blob=_atbh_blob() + b"JUNK", expected_dtype=np.int8)


def test_hyperprior_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError, match="payload truncated|truncated sigma"):
        decode_hyperprior(blob=_atbh_blob()[:-1], expected_dtype=np.int8)


def test_hyperprior_rejects_bad_block_size_and_block_count() -> None:
    blob = bytearray(_atbh_blob())
    struct.pack_into("<I", blob, 20, 0)
    with pytest.raises(ValueError, match="block_size"):
        decode_hyperprior(blob=bytes(blob), expected_dtype=np.int8)

    blob = bytearray(_atbh_blob())
    struct.pack_into("<I", blob, 24, 999)
    with pytest.raises(ValueError, match="n_blocks"):
        decode_hyperprior(blob=bytes(blob), expected_dtype=np.int8)
