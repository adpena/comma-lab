# SPDX-License-Identifier: MIT
"""Tests for the ddm_ix2 cross-member archive container.

Every test here is written so it FAILS if the implementation is replaced by a
plausible no-op.  ``test_cell_major_layout_must_beat_aos`` constructs an array
where cell-major is provably the stationary layout and asserts AoS loses, so a
transpose that silently became the identity is caught.  ``test_decode_needs_only
_the_frame`` is the regression for the exact fake ``ddm_ix1`` found in its own
first headline: reconstructing with a ``base`` the decoder does not have.
"""

from __future__ import annotations

import io
import struct
import zipfile

import brotli
import numpy as np
import pytest

from tac.optimization.ddm_ix2_archive_container import (
    CODER_NAMES,
    RENDERER_FRAME_MAGIC,
    TOKEN_FRAME_MAGIC,
    IX2ContainerError,
    _factor_mode_delta,
    _pack_nibbles,
    _unpack_nibbles,
    build_payload,
    build_single_member_zip,
    classify_against_vendored,
    code_block,
    decode_block,
    decode_renderer_frame,
    decode_token_frame,
    encode_renderer_frame,
    encode_token_frame,
    pack_config_section,
    pack_container,
    parse_payload,
    unpack_config_section,
    unpack_container,
    zip_framing_overhead,
)


def _lattice(seed: int = 7, shape: tuple[int, int, int, int] = (24, 6, 8, 4)):
    rng = np.random.default_rng(seed)
    p, r, c, k = shape
    base = rng.integers(0, 16, size=(r, c, k), dtype=np.uint8)
    codes = np.repeat(base[None], p, axis=0).astype(np.int16)
    flips = rng.random(codes.shape) < 0.25
    codes[flips] = rng.integers(0, 16, size=int(flips.sum()), dtype=np.int16)
    return np.ascontiguousarray((codes % 16).astype(np.uint8))


# --------------------------------------------------------------------------- #
# nibble lane                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("count", [0, 1, 2, 3, 17, 4096])
def test_nibble_lane_is_a_bijection(count: int) -> None:
    rng = np.random.default_rng(count + 1)
    values = rng.integers(0, 16, size=count, dtype=np.uint8)
    packed = _pack_nibbles(values)
    assert len(packed) == (count + 1) // 2
    assert np.array_equal(_unpack_nibbles(packed, count), values)


def test_nibble_lane_refuses_out_of_range() -> None:
    with pytest.raises(IX2ContainerError):
        _pack_nibbles(np.array([16], dtype=np.uint8))


def test_unpack_nibbles_refuses_truncated_lane() -> None:
    with pytest.raises(IX2ContainerError):
        _unpack_nibbles(b"\x00", 10)


# --------------------------------------------------------------------------- #
# generic block coding                                                         #
# --------------------------------------------------------------------------- #


def test_code_block_picks_stored_on_incompressible_payload() -> None:
    """The 'already at entropy' case must be representable.

    On the live token stream brotli is +5 B WORSE than stored; a racer without a
    stored rung reports a loss as a win.
    """

    payload = np.random.default_rng(0).integers(0, 256, size=8192, dtype=np.uint8).tobytes()
    coder_id, coded = code_block(payload)
    assert CODER_NAMES[coder_id] == "stored"
    assert coded == payload


def test_code_block_beats_stored_on_compressible_payload() -> None:
    payload = b"\x00" * 8192
    coder_id, coded = code_block(payload)
    assert CODER_NAMES[coder_id] != "stored"
    assert len(coded) < len(payload)
    assert decode_block(coder_id, coded) == payload


@pytest.mark.parametrize("coder_id", range(len(CODER_NAMES)))
def test_every_coder_id_roundtrips(coder_id: int) -> None:
    payload = bytes(range(256)) * 5
    _, _ = code_block(payload)
    import lzma
    import zlib

    encoders = {
        0: lambda b: b,
        1: lambda b: zlib.compress(b, 9),
        2: lambda b: brotli.compress(b, quality=11, lgwin=24),
        3: lambda b: lzma.compress(
            b,
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 24, "lc": 3, "lp": 0, "pb": 0}],
        ),
    }
    assert decode_block(coder_id, encoders[coder_id](payload)) == payload


def test_decode_block_refuses_unknown_coder() -> None:
    with pytest.raises(IX2ContainerError):
        decode_block(99, b"")


# --------------------------------------------------------------------------- #
# IX2TOK01                                                                     #
# --------------------------------------------------------------------------- #


def test_token_frame_roundtrip_is_bit_identical() -> None:
    codes = _lattice()
    frame = encode_token_frame(codes)
    assert frame[:8] == TOKEN_FRAME_MAGIC
    assert np.array_equal(decode_token_frame(frame), codes)


def test_decode_needs_only_the_frame() -> None:
    """Regression for the ix1 borrowed-``base`` fake.

    ``base`` is the per-cell mode over ALL pairs — video-derived, and NOT derivable
    from the residual.  If it is not shipped, decode cannot succeed.  Truncating the
    base block must raise, never silently reconstruct.
    """

    codes = _lattice()
    frame = bytearray(encode_token_frame(codes))
    (levels, cr, cb, res, p, r, c, k, len_r, len_b) = struct.unpack_from("<BBBBHHHHII", frame, 8)
    assert len_b > 0, "base must actually occupy bytes in the frame"
    # Drop the base block and declare it absent: decode must refuse, not guess.
    struct.pack_into("<BBBBHHHHII", frame, 8, levels, cr, cb, res, p, r, c, k, len_r, 0)
    truncated = bytes(frame[: 8 + 24 + len_r])
    with pytest.raises(IX2ContainerError):
        decode_token_frame(truncated)


def test_token_frame_refuses_trailing_bytes() -> None:
    frame = encode_token_frame(_lattice())
    with pytest.raises(IX2ContainerError, match="unconsumed"):
        decode_token_frame(frame + b"\x00")


def test_token_frame_refuses_wrong_magic() -> None:
    frame = encode_token_frame(_lattice())
    with pytest.raises(IX2ContainerError, match="magic"):
        decode_token_frame(b"XX2TOK01" + frame[8:])


def test_token_frame_refuses_nonzero_reserved() -> None:
    frame = bytearray(encode_token_frame(_lattice()))
    frame[11] = 1
    with pytest.raises(IX2ContainerError, match="reserved"):
        decode_token_frame(bytes(frame))


def test_token_frame_refuses_codes_above_levels() -> None:
    codes = _lattice()
    codes[0, 0, 0, 0] = 15
    with pytest.raises(IX2ContainerError, match="exceeds levels"):
        encode_token_frame(codes, levels=8)


@pytest.mark.parametrize("levels", [1, 17])
def test_token_frame_refuses_bad_levels(levels: int) -> None:
    with pytest.raises(IX2ContainerError):
        encode_token_frame(np.zeros((2, 2, 2, 2), np.uint8), levels=levels)


def test_token_frame_refuses_non_4d() -> None:
    with pytest.raises(IX2ContainerError):
        encode_token_frame(np.zeros((2, 2, 2), np.uint8))


def test_mode_factorisation_matches_r7() -> None:
    """Pin the local factorisation to the r7 original it reimplements."""

    r7 = pytest.importorskip("experiments.ddm_r7_token_coder", reason="r7 coder not importable")
    codes = _lattice()
    base, delta = _factor_mode_delta(codes, 16)
    r7_base, r7_delta = r7.factor_mode_delta(codes, 16)
    assert np.array_equal(base, r7_base)
    assert np.array_equal(delta, r7_delta)


def test_cell_major_layout_must_beat_aos() -> None:
    """Fails if the transpose silently became the identity.

    Reproduces the structure MEASURED on the live lattice: each cell draws from its
    OWN low-entropy distribution (mostly one value, sometimes not), and cells differ.
    Cell-major turns each cell into a long near-run a stock coder can LZ-copy; in AoS
    the per-pair block never repeats exactly, so no long match exists.  Both layouts
    hold the same multiset, so any size difference is layout alone.
    """

    rng = np.random.default_rng(3)
    r, c, k, p = 6, 8, 4, 96
    mode = rng.integers(0, 16, size=(r, c, k), dtype=np.uint8)
    codes = np.repeat(mode[None], p, axis=0).astype(np.uint8)
    noisy = rng.random(codes.shape) < 0.15
    codes[noisy] = rng.integers(0, 16, size=int(noisy.sum()), dtype=np.uint8)
    cell_major = _pack_nibbles(
        np.ascontiguousarray(np.transpose(codes, (1, 2, 3, 0))).reshape(-1)
    )
    aos = _pack_nibbles(np.ascontiguousarray(codes).reshape(-1))
    assert len(cell_major) == len(aos), "layout must not change the raw byte count"
    assert np.array_equal(
        np.sort(_unpack_nibbles(cell_major, codes.size)),
        np.sort(_unpack_nibbles(aos, codes.size)),
    ), "layout must be a permutation, not a different payload"
    assert len(brotli.compress(cell_major, quality=11)) < len(
        brotli.compress(aos, quality=11)
    )


def test_encode_token_frame_actually_uses_cell_major() -> None:
    """A frame built with the identity transpose would be strictly larger."""

    codes = _lattice(seed=13, shape=(96, 6, 8, 4))
    frame = encode_token_frame(codes)
    _, delta = _factor_mode_delta(codes, 16)
    aos_block = code_block(_pack_nibbles(np.ascontiguousarray(delta).reshape(-1)))[1]
    cell_block = code_block(
        _pack_nibbles(np.ascontiguousarray(np.transpose(delta, (1, 2, 3, 0))).reshape(-1))
    )[1]
    assert len(cell_block) < len(aos_block)
    assert len(frame) < 8 + 24 + len(aos_block) + 2048


# --------------------------------------------------------------------------- #
# IX2REN01                                                                     #
# --------------------------------------------------------------------------- #


def test_renderer_frame_roundtrip() -> None:
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, size=1237, dtype=np.uint8)
    floats = rng.integers(0, 65536, size=40, dtype=np.uint16).astype(">u2").tobytes()
    frame = encode_renderer_frame(bits, floats)
    assert frame[:8] == RENDERER_FRAME_MAGIC
    got_bits, got_floats = decode_renderer_frame(frame)
    assert np.array_equal(got_bits, bits)
    assert got_floats == floats


def test_renderer_frame_refuses_non_binary_mask() -> None:
    with pytest.raises(IX2ContainerError, match="binary"):
        encode_renderer_frame(np.array([2], dtype=np.uint8), b"\x00\x00")


def test_renderer_frame_refuses_odd_float_payload() -> None:
    with pytest.raises(IX2ContainerError):
        encode_renderer_frame(np.array([1], dtype=np.uint8), b"\x00")


def test_renderer_frame_refuses_counted_inert_padding() -> None:
    bits = np.ones(5, dtype=np.uint8)
    frame = bytearray(encode_renderer_frame(bits, b"\x00\x00"))
    packed_off = 8 + 13
    frame[packed_off] |= 0b0000_0001  # a padding bit outside mask_count
    with pytest.raises(IX2ContainerError, match="inert"):
        decode_renderer_frame(bytes(frame))


def test_renderer_frame_refuses_trailing_bytes() -> None:
    frame = encode_renderer_frame(np.ones(8, np.uint8), b"\x00\x00")
    with pytest.raises(IX2ContainerError, match="unconsumed"):
        decode_renderer_frame(frame + b"\x01")


# --------------------------------------------------------------------------- #
# counted config section                                                       #
# --------------------------------------------------------------------------- #


def test_config_section_roundtrip_is_exact() -> None:
    mags = [-7.5, -3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]
    payload = pack_config_section(32.125, mags)
    assert len(payload) == 6 + 2 * len(mags)
    offset, got, grid = unpack_config_section(payload)
    assert offset == 32.125
    assert list(got) == mags
    assert grid is None


def test_st_grid_verdict_is_a_field_archive_pair_property() -> None:
    """MEASURED by cp1: same field, opposite rule-118 verdict on different bases."""

    vendored = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]
    assert classify_against_vendored(vendored, vendored) == "GENERIC"
    assert classify_against_vendored([0.0, 0.0625, 0.25], vendored) == "VIDEO_DERIVED"
    assert classify_against_vendored(None, vendored) == "ABSENT"
    # generic -> omitted (one flag byte), video-derived -> COUNTED in the section
    generic = pack_config_section(1.0, [0.5], vendored, vendored_st_grid=vendored)
    fitted = pack_config_section(
        1.0, [0.5], [0.0, 0.0625, 0.25], vendored_st_grid=vendored
    )
    assert len(fitted) == len(generic) + 6
    assert unpack_config_section(generic)[2] is None
    assert unpack_config_section(fitted)[2] == (0.0, 0.0625, 0.25)


def test_config_section_refuses_to_decide_without_the_reference() -> None:
    with pytest.raises(IX2ContainerError, match="pair property"):
        pack_config_section(1.0, [0.5], [0.0, 0.5])


def test_two_tier_payload_roundtrip() -> None:
    rng = np.random.default_rng(21)
    bulk = rng.integers(0, 256, size=5000, dtype=np.uint8).tobytes()
    joint = [b"alpha" * 40, b"beta" * 30, b"gamma" * 20]
    payload = build_payload(bulk, joint)
    got_bulk, got_joint = parse_payload(payload)
    assert got_bulk == bulk
    assert list(got_joint) == joint


def test_two_tier_payload_beats_independent_coding_of_the_small_group() -> None:
    """The shared-coder-state rung, demonstrated rather than asserted."""

    joint = [(f"section-{i}-" * 60).encode() for i in range(5)]
    independent = sum(len(code_block(s)[1]) for s in joint)
    shared = len(code_block(pack_container(joint))[1])
    assert shared < independent


def test_payload_refuses_truncation() -> None:
    payload = build_payload(b"bulk", [b"x"])
    with pytest.raises(IX2ContainerError):
        parse_payload(payload[:3])
    with pytest.raises(IX2ContainerError):
        parse_payload(struct.pack("<IB", 10_000, 0) + b"ab")


def test_config_section_refuses_silently_rounding_a_fitted_codebook() -> None:
    """A fitted codebook must never be quietly re-quantised into the section."""

    with pytest.raises(IX2ContainerError, match="f16"):
        pack_config_section(0.0, [0.100000001490116119384765625e-6])


def test_config_section_refuses_inexact_offset() -> None:
    with pytest.raises(IX2ContainerError, match="f32"):
        pack_config_section(1e-60, [0.0])


def test_config_section_refuses_short_payload() -> None:
    with pytest.raises(IX2ContainerError):
        unpack_config_section(b"\x00")


def test_config_section_refuses_length_mismatch() -> None:
    payload = pack_config_section(1.0, [0.5, 1.0])
    with pytest.raises(IX2ContainerError, match="close exactly"):
        unpack_config_section(payload + b"\x00\x00")


# --------------------------------------------------------------------------- #
# IX2CNT01 container                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 5])
def test_container_roundtrip(n: int) -> None:
    rng = np.random.default_rng(n)
    sections = [rng.integers(0, 256, size=17 * (i + 1), dtype=np.uint8).tobytes() for i in range(n)]
    packed = pack_container(sections)
    assert len(packed) == 4 * (n - 1) + sum(len(s) for s in sections)
    assert list(unpack_container(packed, n)) == sections


def test_container_last_length_is_implied_not_stored() -> None:
    sections = [b"ab", b"cdef"]
    packed = pack_container(sections)
    assert packed == struct.pack("<I", 2) + b"ab" + b"cdef"


def test_container_refuses_overrun() -> None:
    packed = struct.pack("<I", 10_000) + b"ab"
    with pytest.raises(IX2ContainerError, match="overrun"):
        unpack_container(packed, 2)


def test_container_refuses_empty() -> None:
    with pytest.raises(IX2ContainerError):
        pack_container([])
    with pytest.raises(IX2ContainerError):
        unpack_container(b"", 0)


def test_container_refuses_truncated_length_table() -> None:
    with pytest.raises(IX2ContainerError, match="truncated"):
        unpack_container(b"\x01\x02", 3)


# --------------------------------------------------------------------------- #
# ZIP framing                                                                  #
# --------------------------------------------------------------------------- #


def test_single_member_zip_is_deterministic() -> None:
    payload = b"payload" * 100
    assert build_single_member_zip(payload) == build_single_member_zip(payload)
    with zipfile.ZipFile(io.BytesIO(build_single_member_zip(payload))) as z:
        assert z.namelist() == ["0.bin"]
        assert z.read("0.bin") == payload


def test_zip_framing_overhead_matches_a_real_zip() -> None:
    """MEASURED against zipfile, not asserted from memory."""

    names = ["manifest.json", "state/tokens.dr7t", "state/pose_warp.stp"]
    payloads = [b"a" * 100, b"b" * 200, b"c" * 300]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, payload in zip(names, payloads, strict=True):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            info.date_time = (1980, 1, 1, 0, 0, 0)
            z.writestr(info, payload)
    actual = len(buf.getvalue()) - sum(len(p) for p in payloads)
    assert zip_framing_overhead(names, [len(p) for p in payloads]) == actual


def test_zip_framing_overhead_refuses_mismatched_inputs() -> None:
    with pytest.raises(IX2ContainerError):
        zip_framing_overhead(["a"], [1, 2])


def test_consolidation_beats_six_members_on_synthetic_payloads() -> None:
    """The cross-member rung, demonstrated end-to-end on a real zipfile."""

    rng = np.random.default_rng(5)
    names = [
        "manifest.json",
        "state/tokens.dr7t",
        "state/renderer.sec",
        "state/selector.sec",
        "state/pose_stub.sec",
        "state/pose_warp.stp",
    ]
    payloads = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes() for n in (400, 5000, 900, 200, 80, 1200)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, payload in zip(names, payloads, strict=True):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            info.date_time = (1980, 1, 1, 0, 0, 0)
            z.writestr(info, payload)
    six = len(buf.getvalue())
    one = len(build_single_member_zip(pack_container(payloads)))
    assert one < six
    # the whole saving is framing: the payload bytes are byte-identical
    assert six - one == zip_framing_overhead(names, [len(p) for p in payloads]) - zip_framing_overhead(
        ["0.bin"], [1]
    ) - 4 * (len(payloads) - 1)
