# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from experiments import ddm_r7_token_coder as coder_module
from experiments.ddm_r7_token_coder import (
    CODEC_IDS,
    HEADER,
    VERIFY_CANONICAL,
    VERIFY_DIGEST,
    DDMR7CoderError,
    _decode_smevr,
    _decode_smevr_reference,
    _encode_smevr,
    _encode_smevr_reference,
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


def _frame_with_inert_trailing_bytes() -> tuple[bytes, np.ndarray]:
    """A frame that decodes correctly but is NOT the canonical encoding.

    Four zero bytes are appended to the SMEVR value stream and the header's
    declared delta length is grown to match, so the frame still closes exactly
    on length.  The arithmetic decoder never reads them (past its own
    termination it reads zero padding regardless), so the decoded lattice -- and
    therefore the semantic SHA-256 -- is unchanged.  Only the canonical rung can
    see the difference.  This is the concrete slack-byte smuggling shape the
    canonical re-encode exists to refuse.
    """

    source = _fixture()
    frame = encode_token_codes(source, codec="smevr")
    fields = list(HEADER.unpack_from(frame))
    inert = b"\x00\x00\x00\x00"
    fields[10] = fields[10] + len(inert)
    return HEADER.pack(*fields) + frame[HEADER.size :] + inert, source


def test_verify_ladder_defaults_to_canonical_and_refuses_inert_slack_bytes() -> None:
    changed, _source = _frame_with_inert_trailing_bytes()
    with pytest.raises(DDMR7CoderError, match="noncanonical or has inert bytes"):
        decode_token_codes(changed)
    with pytest.raises(DDMR7CoderError, match="noncanonical or has inert bytes"):
        decode_token_codes(changed, verify=VERIFY_CANONICAL)


def test_verify_digest_rung_is_a_real_but_bounded_weakening() -> None:
    changed, source = _frame_with_inert_trailing_bytes()
    # The opt-out accepts the slack bytes the canonical rung refuses ...
    assert np.array_equal(decode_token_codes(changed, verify=VERIFY_DIGEST), source)
    # ... but keeps every content guarantee: a frame whose payload no longer
    # reconstructs the digested lattice is still refused.
    corrupted = bytearray(changed)
    corrupted[HEADER.size + 1] ^= 0xFF
    with pytest.raises(DDMR7CoderError):
        decode_token_codes(bytes(corrupted), verify=VERIFY_DIGEST)


def test_verify_digest_rung_still_refuses_structural_and_lattice_violations() -> None:
    frame = encode_token_codes(_fixture(), codec="smevr")
    for changed in (frame[:-1], frame + b"\0"):
        with pytest.raises(DDMR7CoderError):
            decode_token_codes(changed, verify=VERIFY_DIGEST)
    fields = list(HEADER.unpack_from(frame))
    fields[3] = 1  # levels outside [2,16]
    with pytest.raises(DDMR7CoderError):
        decode_token_codes(HEADER.pack(*fields) + frame[HEADER.size :], verify=VERIFY_DIGEST)


@pytest.mark.parametrize("mode", ("", "none", "off", "CANONICAL", True, None))
def test_unknown_verify_mode_fails_closed(mode: object) -> None:
    frame = encode_token_codes(_fixture(), codec="smevr")
    with pytest.raises(DDMR7CoderError, match="unsupported token verify mode"):
        decode_token_codes(frame, verify=mode)  # type: ignore[arg-type]


def _smevr_parity_corpus() -> list[np.ndarray]:
    rng = np.random.default_rng(20260801)
    sparse = np.zeros((7, 4, 5, 2), dtype=np.uint8)
    sparse[3, 1, 2, 0] = 9
    sparse[4, 1, 2, 0] = 9
    sparse[5, 0, 0, 1] = 1
    return [
        _fixture(),
        rng.integers(0, 16, size=(6, 5, 7, 3), dtype=np.uint8),
        sparse,
        np.tile(np.arange(16, dtype=np.uint8), 16).reshape(4, 4, 4, 4),
        # single-column and single-row lattices exercise the missing-neighbour
        # branches the cyclic per-cell masks replaced.
        rng.integers(0, 16, size=(5, 6, 1, 1), dtype=np.uint8),
        rng.integers(0, 16, size=(5, 1, 6, 1), dtype=np.uint8),
    ]


@pytest.mark.parametrize("rescale_at", (8, 64, 32768))
def test_smevr_fast_paths_reproduce_the_reference_oracle_exactly(
    monkeypatch: pytest.MonkeyPatch,
    rescale_at: int,
) -> None:
    """The optimised SMEVR paths are byte/array identical to the reference.

    ``_RESCALE_AT`` is swept because the bounded-count rescale is a branch inside
    both rewritten inner loops.  MEASURED on this corpus: the branch fires 7972
    times at 8 and 12 times at 64, but ZERO times at the shipped 32768 -- so
    without the sweep this test would leave that branch entirely uncovered and
    the real fields would be its only witness.
    """

    monkeypatch.setattr(coder_module, "_RESCALE_AT", rescale_at)
    for codes in _smevr_parity_corpus():
        levels = 16
        base, delta = factor_mode_delta(codes, levels)
        reference_stream = _encode_smevr_reference(base, delta, levels)
        assert _encode_smevr(base, delta, levels) == reference_stream
        shape = tuple(int(value) for value in codes.shape)
        reference_delta = _decode_smevr_reference(reference_stream, base, shape, levels)
        fast_delta = _decode_smevr(reference_stream, base, shape, levels)
        assert fast_delta.dtype == reference_delta.dtype
        assert np.array_equal(fast_delta, reference_delta)
        assert np.array_equal(reconstruct_mode_delta(base, fast_delta, levels), codes)
