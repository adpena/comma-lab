# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from experiments.ddm_r7_token_coder import (
    CODEC_IDS,
    HEADER,
    DDMR7CoderError,
    decode_token_codes,
    encode_token_codes,
    factor_mode_delta,
    frame_accounting,
    reconstruct_mode_delta,
)


def _fixture() -> np.ndarray:
    return ((np.arange(4 * 3 * 4 * 4, dtype=np.uint16) * 7 + 3) % 16).astype(np.uint8).reshape(4, 3, 4, 4)


@pytest.mark.parametrize("codec", sorted(CODEC_IDS))
def test_every_physical_codec_is_deterministic_and_exact(codec: str) -> None:
    source = _fixture()
    first = encode_token_codes(source, codec=codec)
    second = encode_token_codes(source, codec=codec)
    assert first == second
    assert np.array_equal(decode_token_codes(first), source)
    accounting = frame_accounting(first)
    assert accounting.codec == codec
    assert accounting.framed_bytes == len(first)
    assert accounting.header_bytes + accounting.base_bytes + accounting.delta_bytes == len(first)


def test_default_exact_outer_selector_has_counted_codec_tag_and_golden_bytes() -> None:
    source = _fixture()
    frame = encode_token_codes(source)
    assert frame_accounting(frame).codec == "smevr"
    assert hashlib.sha256(frame).hexdigest() == ("50c0d676e3b138c689eebf61e607590af44c875f55ee88c7952c3d78a5039e94")
    assert np.array_equal(decode_token_codes(frame), source)


def test_mode_factorization_is_exact_and_uses_lowest_mode_tie() -> None:
    source = np.array(
        [
            [[[0, 1]]],
            [[[1, 1]]],
            [[[0, 2]]],
            [[[1, 2]]],
        ],
        dtype=np.uint8,
    )
    base, delta = factor_mode_delta(source, 16)
    assert base.tolist() == [[[0, 1]]]
    assert np.array_equal(reconstruct_mode_delta(base, delta, 16), source)


@pytest.mark.parametrize(
    "codec",
    ("smevr", "kt_prev1", "cae_inspired_identity_inter", "huffman_nibble"),
)
def test_all_lattice_extremes_roundtrip(codec: str) -> None:
    source = np.tile(np.arange(16, dtype=np.uint8), 16).reshape(4, 4, 4, 4)
    assert np.array_equal(decode_token_codes(encode_token_codes(source, codec=codec)), source)


@pytest.mark.parametrize("mutation", ("truncate", "trailer", "bitflip"))
def test_frame_corruption_and_inert_bytes_are_refused(mutation: str) -> None:
    frame = bytearray(encode_token_codes(_fixture(), codec="smevr"))
    if mutation == "truncate":
        changed = bytes(frame[:-1])
    elif mutation == "trailer":
        changed = bytes(frame) + b"\0"
    else:
        frame[-1] ^= 1
        changed = bytes(frame)
    with pytest.raises(DDMR7CoderError):
        decode_token_codes(changed)


def test_malformed_shape_levels_and_codec_are_refused() -> None:
    frame = bytearray(encode_token_codes(_fixture(), codec="kt_prev1"))
    fields = list(HEADER.unpack_from(frame))
    for field_index, replacement in ((2, 255), (3, 1), (4, 3), (5, 0)):
        changed_fields = fields.copy()
        changed_fields[field_index] = replacement
        changed = HEADER.pack(*changed_fields) + bytes(frame[HEADER.size :])
        with pytest.raises(DDMR7CoderError):
            decode_token_codes(changed)


def test_input_contract_rejects_wrong_dtype_rank_and_levels() -> None:
    source = _fixture()
    with pytest.raises(DDMR7CoderError):
        encode_token_codes(source.astype(np.int16))
    with pytest.raises(DDMR7CoderError):
        encode_token_codes(source[0])
    with pytest.raises(DDMR7CoderError):
        encode_token_codes(source, levels=17)


@pytest.mark.parametrize("codec", ("smevr", "rans_o0", "lzma1"))
def test_odd_token_and_mode_counts_roundtrip_with_zero_nibble_padding(codec: str) -> None:
    source = np.array([0, 1, 15], dtype=np.uint8).reshape(3, 1, 1, 1)
    frame = encode_token_codes(source, codec=codec)
    assert np.array_equal(decode_token_codes(frame), source)


def test_adjacent_innovation_rans_honors_non_power_of_two_levels() -> None:
    source = (np.arange(5 * 2 * 3 * 1, dtype=np.uint8) % 7).reshape(5, 2, 3, 1)
    frame = encode_token_codes(
        source,
        levels=7,
        codec="rans_o0_on_adjacent_innovation",
    )
    assert np.array_equal(decode_token_codes(frame), source)


def test_semantic_header_shape_and_codec_are_authenticated() -> None:
    source = _fixture()
    frame = bytearray(encode_token_codes(source, codec="rans_o0"))
    fields = list(HEADER.unpack_from(frame))
    height_width_swapped = fields.copy()
    height_width_swapped[6], height_width_swapped[7] = (
        height_width_swapped[7],
        height_width_swapped[6],
    )
    changed_shape = HEADER.pack(*height_width_swapped) + bytes(frame[HEADER.size :])
    with pytest.raises(DDMR7CoderError, match="semantic SHA-256"):
        decode_token_codes(changed_shape)

    changed_codec_fields = fields.copy()
    changed_codec_fields[2] = CODEC_IDS["rans_o0_on_adjacent_innovation"]
    changed_codec = HEADER.pack(*changed_codec_fields) + bytes(frame[HEADER.size :])
    with pytest.raises(DDMR7CoderError, match="semantic SHA-256"):
        decode_token_codes(changed_codec)


def test_header_layout_is_fixed_little_endian_and_compact() -> None:
    assert HEADER.size == 56
    assert struct.calcsize("<4sBBBB4HII32s") == HEADER.size
