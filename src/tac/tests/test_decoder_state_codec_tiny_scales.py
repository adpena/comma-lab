# SPDX-License-Identifier: MIT
"""Regression tests for tiny decoder-weight quantization scales."""

from __future__ import annotations

import warnings

import numpy as np
import torch

from tac.substrates._shared import decoder_state_codec as codec


def _assert_no_divide_warning(caught: list[warnings.WarningMessage]) -> None:
    messages = [str(item.message) for item in caught]
    assert not any("divide by zero" in message for message in messages)


def test_int8_tiny_nonzero_tensor_uses_positive_fp16_scale() -> None:
    tiny = torch.full((2, 3), 1.0e-10, dtype=torch.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        record = codec._encode_int8_record(tiny)

    _assert_no_divide_warning(caught)
    scale = np.asarray(record["scale"], dtype=np.float16).astype(np.float32)
    assert np.all(scale > 0.0)
    decoded = codec._decode_int8_record(record)
    assert torch.isfinite(decoded).all()


def test_int8_tiny_vector_tensor_uses_positive_fp16_scale() -> None:
    tiny = torch.full((3,), 1.0e-10, dtype=torch.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        record = codec._encode_int8_record(tiny)

    _assert_no_divide_warning(caught)
    scale = np.asarray(record["scale"], dtype=np.float16).astype(np.float32)
    assert float(scale) > 0.0
    decoded = codec._decode_int8_record(record)
    assert torch.isfinite(decoded).all()


def test_nbit_tiny_nonzero_tensor_uses_positive_fp16_scale() -> None:
    tiny = torch.full((2, 3), 1.0e-10, dtype=torch.float32)

    for bits in (2, 4):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            record = codec._encode_nbit_record(tiny, bits=bits)

        _assert_no_divide_warning(caught)
        scale = np.asarray(record["scale"], dtype=np.float16).astype(np.float32)
        assert np.all(scale > 0.0)
        decoded = codec._decode_nbit_record(record, bits=bits)
        assert torch.isfinite(decoded).all()

