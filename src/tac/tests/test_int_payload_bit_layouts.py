# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.packet_compiler.int_payload_bit_layouts import (
    DEFAULT_INT_PAYLOAD_LAYOUTS,
    decode_int_payload_layout,
    encode_int_payload_layout,
)


@pytest.mark.parametrize("raw_len", [0, 1, 2, 3, 17, 256, 257])
@pytest.mark.parametrize("layout", DEFAULT_INT_PAYLOAD_LAYOUTS)
def test_int_payload_bit_layouts_roundtrip_exact(raw_len: int, layout: str) -> None:
    payload = bytes((idx * 37 + 11) % 256 for idx in range(raw_len))

    encoded = encode_int_payload_layout(payload, layout)  # type: ignore[arg-type]
    decoded = decode_int_payload_layout(
        encoded,
        layout=layout,  # type: ignore[arg-type]
        raw_len=raw_len,
    )

    assert decoded == payload


def test_int_payload_bit_layouts_reject_wrong_length() -> None:
    encoded = encode_int_payload_layout(b"abc", "bitplanes_lsb")

    with pytest.raises(ValueError, match="does not match"):
        decode_int_payload_layout(encoded + b"x", layout="bitplanes_lsb", raw_len=3)


def test_int_payload_bit_layouts_reject_unknown_layout() -> None:
    with pytest.raises(ValueError, match="unknown"):
        encode_int_payload_layout(b"abc", "bad")  # type: ignore[arg-type]
