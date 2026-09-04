"""Falsifiers for the frame-0 selector encoder (ddm_fs1).

Each test is a falsifier of a claim the byte-close rests on, not a "does it run"
check.  The headline is ``test_encoder_reproduces_the_shipped_blob``: the encoder
must rebuild the 14 bytes the afr1 archive actually carries, from the choices the
SHIPPED decoder reads out of them.  If that fails, every byte delta this arm
reports is meaningless.

Every round trip goes through the SHIPPED ``decode_selector``.  A round trip
against a locally re-implemented decoder would only prove the encoder agrees with
itself, which is the "agreeing-with-the-test" failure the operating manual names.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tac.semantic_pipeline.frame0_selector_codec import (
    FRAME_COUNT,
    MODE_COUNT,
    STORED_PREFIX,
    Frame0SelectorEncodeError,
    combination_rank,
    encode_selector,
    load_shipped_decoder,
    pack_labels,
    selector_blob_length,
    stored_tail,
)

REPO = Path(__file__).resolve().parents[3]
SHIPPED_DECODER = (
    REPO / "submissions" / "semantic_joint_ctxmix" / "runtime" / "frame0_selector.py"
)

#: The exact selector blob the afr1 frontier archive carries, lifted from
#: ``archive.zip`` (sha cbb8d928...) through the receiver's own parse: 5 active
#: pairs (60, 85, 116, 241, 373) in modes (4, 3, 4, 7, 4).
AFR1_SELECTOR_BLOB = bytes.fromhex("463045310105000dab3567db69e6")
AFR1_ACTIVE_PAIRS = (60, 85, 116, 241, 373)
AFR1_ACTIVE_MODES = (4, 3, 4, 7, 4)


@pytest.fixture(scope="module")
def decoder():
    if not SHIPPED_DECODER.is_file():
        pytest.skip(f"shipped decoder absent: {SHIPPED_DECODER}")
    return load_shipped_decoder()


def _choices(pairs, modes) -> np.ndarray:
    out = np.zeros(FRAME_COUNT, dtype=np.uint8)
    out[list(pairs)] = list(modes)
    return out


def test_shipped_blob_decodes_to_the_documented_selector(decoder):
    """The fixture is what the archive carries, not what this file believes."""
    _modes, choices = decoder.decode_selector(AFR1_SELECTOR_BLOB)
    assert np.flatnonzero(choices).tolist() == list(AFR1_ACTIVE_PAIRS)
    assert choices[list(AFR1_ACTIVE_PAIRS)].tolist() == list(AFR1_ACTIVE_MODES)


def test_encoder_reproduces_the_shipped_blob(decoder):
    """THE control: encode(decode(shipped)) must be the shipped bytes, exactly."""
    _modes, choices = decoder.decode_selector(AFR1_SELECTOR_BLOB)
    assert encode_selector(choices) == AFR1_SELECTOR_BLOB


def test_blob_length_formula_matches_the_shipped_body():
    """At k=5 the receiver's closed formula must return the 14 B the archive carries."""
    assert selector_blob_length(5) == len(AFR1_SELECTOR_BLOB) == 14


@pytest.mark.parametrize("count", [1, 2, 3, 5, 17, 24, 42, 52, 128, 300, 599, 600])
def test_encoded_length_equals_the_closed_formula(decoder, count):
    rng = np.random.default_rng(1000 + count)
    positions = rng.choice(FRAME_COUNT, size=count, replace=False)
    modes = rng.integers(1, MODE_COUNT, size=count)
    choices = _choices(positions, modes)
    blob = encode_selector(choices)
    assert len(blob) == selector_blob_length(count)
    _m, parsed = decoder.decode_selector(blob)
    assert np.array_equal(np.asarray(parsed, dtype=np.uint8), choices)


def test_fuzz_roundtrip_through_the_shipped_decoder(decoder):
    """300 random selectors must survive encode -> SHIPPED decode unchanged."""
    rng = np.random.default_rng(20260904)
    for _ in range(300):
        count = int(rng.integers(1, 90))
        positions = rng.choice(FRAME_COUNT, size=count, replace=False)
        modes = rng.integers(1, MODE_COUNT, size=count)
        choices = _choices(positions, modes)
        blob = encode_selector(choices)
        _m, parsed = decoder.decode_selector(blob)
        assert np.array_equal(np.asarray(parsed, dtype=np.uint8), choices)


def test_all_identity_selector_is_refused_not_invented():
    """``decode_selector`` refuses count == 0, so the encoder must too."""
    with pytest.raises(Frame0SelectorEncodeError, match="all-identity"):
        encode_selector(np.zeros(FRAME_COUNT, dtype=np.uint8))


def test_out_of_range_mode_is_refused():
    choices = np.zeros(FRAME_COUNT, dtype=np.int64)
    choices[3] = MODE_COUNT  # one past the last catalog entry
    with pytest.raises(Frame0SelectorEncodeError, match="0\\.\\.7"):
        encode_selector(choices)


def test_wrong_length_choice_vector_is_refused():
    with pytest.raises(Frame0SelectorEncodeError, match="vector"):
        encode_selector(np.ones(FRAME_COUNT - 1, dtype=np.uint8))


def test_combination_rank_is_the_decoder_inverse(decoder):
    """Ranking then un-ranking must return the same positions, on the shipped code."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        count = int(rng.integers(1, 40))
        positions = np.sort(rng.choice(FRAME_COUNT, size=count, replace=False))
        rank = combination_rank(positions.tolist())
        assert 0 <= rank < math.comb(FRAME_COUNT, count)
        recovered = decoder._combination_unrank(rank, count, FRAME_COUNT)
        assert np.array_equal(np.asarray(recovered), positions)


def test_combination_rank_refuses_unsorted_duplicate_and_out_of_range():
    with pytest.raises(Frame0SelectorEncodeError, match="ascending"):
        combination_rank([5, 3])
    with pytest.raises(Frame0SelectorEncodeError, match="ascending"):
        combination_rank([5, 5])
    with pytest.raises(Frame0SelectorEncodeError, match="out of range"):
        combination_rank([FRAME_COUNT])
    with pytest.raises(Frame0SelectorEncodeError, match="empty"):
        combination_rank([])


def test_pack_labels_is_msb_first_with_zero_padding():
    """The shipped labels (4,3,4,7,4) minus one are 3,2,3,6,3 -> 0x69 0xE6."""
    assert pack_labels([3, 2, 3, 6, 3]) == bytes((0x69, 0xE6))
    # 3 labels = 9 bits -> 2 bytes, and the 7 padding bits must be zero.
    packed = pack_labels([0, 0, 1])
    assert len(packed) == 2
    assert packed[-1] & 0x7F == 0


def test_pack_labels_refuses_a_label_the_decoder_would_reject():
    with pytest.raises(Frame0SelectorEncodeError, match="0\\.\\.6"):
        pack_labels([7])


def test_stored_tail_strips_exactly_the_container_prefix():
    assert stored_tail(AFR1_SELECTOR_BLOB) == AFR1_SELECTOR_BLOB[len(STORED_PREFIX) :]
    assert len(stored_tail(AFR1_SELECTOR_BLOB)) == 9
    with pytest.raises(Frame0SelectorEncodeError, match="F0E1"):
        stored_tail(b"XXXX\x01\x00")


def test_selector_blob_length_refuses_impossible_counts():
    with pytest.raises(Frame0SelectorEncodeError):
        selector_blob_length(0)
    with pytest.raises(Frame0SelectorEncodeError):
        selector_blob_length(FRAME_COUNT + 1)


def test_verification_is_on_by_default_and_catches_a_corrupted_encoder(monkeypatch):
    """Poison the label packer; the shipped-decoder check must refuse the bytes."""
    import tac.semantic_pipeline.frame0_selector_codec as codec

    choices = _choices((10, 20, 30), (1, 2, 3))
    monkeypatch.setattr(codec, "pack_labels", lambda labels: pack_labels([6, 6, 6]))
    with pytest.raises(Frame0SelectorEncodeError, match="differing choices"):
        codec.encode_selector(choices)


def test_missing_shipped_decoder_refuses_rather_than_falling_back(tmp_path):
    with pytest.raises(Frame0SelectorEncodeError, match="refusing to encode"):
        encode_selector(
            _choices((1,), (1,)), decoder_path=tmp_path / "no_such_decoder.py"
        )
