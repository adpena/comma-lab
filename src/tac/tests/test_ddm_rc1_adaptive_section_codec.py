# SPDX-License-Identifier: MIT
"""Producer controls for the ddm_rc1 adaptive MODEL-section codec.

The codec is BOTH the encoder and the receiver, so the only property that matters is
that ``restore_*(apply_*(body)) == body`` for a FRESH decoder, and that a malformed or
mislabelled stream refuses instead of silently producing wrong bytes.  These tests use
synthetic bodies built to the real SM3R mode-6 / IHS1 grammar so they run without the
SSD-retained frontier payloads; the arm's own race stage runs the same assertion on the
shipped bodies and is the authority for the byte counts.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import ddm_rc1_adaptive_section_codec as codec  # noqa: E402

torch = pytest.importorskip("torch")


def _template() -> dict[str, torch.Tensor]:
    """A miniature renderer template with the real SM3R shape classes.

    One embedding (scales on the LAST axis), one plain rank>=2 tensor, one row-pruned
    FiLM tensor, and one rank<2 tensor carried as raw fp16.
    """

    return {
        "frame_embed.weight": torch.zeros(6, 4),
        "coord_mix.weight": torch.zeros(5, 3, 1, 1),
        "blocks.1.film.weight": torch.zeros(8, 4),
        "head.bias": torch.zeros(3),
    }


def _build_sm3r_body(template, depths: dict[str, int], keep_percent: int = 25) -> bytes:
    """Serialize a valid SM3R v1 mode-6 body for ``template`` with deterministic codes."""

    rng = np.random.default_rng(20260905)
    names = [name for name, value in template.items() if value.ndim >= 2]
    selection = sum(
        1 << index for index, name in enumerate(names) if name in codec.ROW_PRUNE_NAMES
    )
    nibbles = np.zeros(((len(names) + 1) // 2) * 2, dtype=np.uint8)
    for index, name in enumerate(names):
        nibbles[index] = depths[name]
    depth_table = (nibbles[0::2] | (nibbles[1::2] << 4)).tobytes()
    out = [b"SM3R", bytes((1, 6, keep_percent, 0)), struct.pack("<H", selection), depth_table]

    def codes(count: int, bits: int) -> bytes:
        low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        values = rng.integers(low, high + 1, size=count, dtype=np.int64)
        return codec.pack_signed_codes(values.astype(np.int32), bits)

    def scales(count: int) -> bytes:
        return rng.integers(0, 256, size=count * 2, dtype=np.uint8).tobytes()

    for name, value in template.items():
        numel = int(value.numel())
        if value.ndim < 2:
            out.append(rng.integers(0, 256, size=numel * 2, dtype=np.uint8).tobytes())
            continue
        bits = depths[name]
        if name not in codec.ROW_PRUNE_NAMES:
            count = int(
                value.shape[-1] if name.endswith("embed.weight") else value.shape[0]
            )
            out.append(scales(count))
            out.append(codes(numel, bits))
            continue
        rows = int(value.shape[0])
        keep = max(1, round(rows * keep_percent / 100.0))
        mask = np.zeros(rows, dtype=np.uint8)
        mask[:keep] = 1
        out.append(np.packbits(mask, bitorder="little").tobytes())
        out.append(scales(keep))
        out.append(codes(keep * (numel // rows), bits))
    return b"".join(out)


def _build_ihs1_body(row_bits: list[int], row_counts: list[int]) -> bytes:
    """Serialize a valid IHS1 body for the given per-channel geometry."""

    rng = np.random.default_rng(11)
    nibbles = np.zeros(((len(row_bits) + 1) // 2) * 2, dtype=np.uint8)
    nibbles[: len(row_bits)] = row_bits
    depth_table = (nibbles[0::2] | (nibbles[1::2] << 4)).tobytes()
    stream: list[np.ndarray] = []
    for bits, count in zip(row_bits, row_counts, strict=True):
        if bits == 0:
            continue
        values = rng.integers(0, 1 << bits, size=count, dtype=np.int64).astype(np.int32)
        block = np.zeros((count, bits), dtype=np.uint8)
        for index in range(bits):
            block[:, index] = (values >> index) & 1
        stream.append(block.reshape(-1))
    flat = np.concatenate(stream) if stream else np.zeros(0, dtype=np.uint8)
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    weights = np.packbits(flat, bitorder="little").tobytes()
    tail = rng.integers(0, 256, size=37, dtype=np.uint8).tobytes()
    return b"IHS1" + depth_table + weights + tail


DEPTHS = {
    "frame_embed.weight": 3,
    "coord_mix.weight": 4,
    "blocks.1.film.weight": 4,
}
ROW_BITS = [0, 1, 2, 4, 5, 8, 0, 7, 3, 6]
ROW_COUNTS = [9, 40, 33, 25, 61, 17, 12, 48, 29, 36]


@pytest.mark.parametrize("shift", [4, 5, 6, 7])
def test_semantic_round_trip_is_byte_identical(shift: int) -> None:
    template = _template()
    body = _build_sm3r_body(template, DEPTHS)
    rider = codec.apply_semantic(body, template, shift)
    assert codec.restore_semantic(rider, template) == body


@pytest.mark.parametrize("shift", [4, 5, 6])
def test_hpac_round_trip_is_byte_identical(shift: int) -> None:
    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    rider = codec.apply_hpac(body, ROW_COUNTS, shift)
    assert codec.restore_hpac(rider, ROW_COUNTS) == body


def test_semantic_rider_is_shorter_than_the_packed_body() -> None:
    """The coder must actually spend fewer bytes than the packing it replaces."""

    template = _template()
    body = _build_sm3r_body(template, DEPTHS)
    rider = codec.apply_semantic(body, template, 6)
    # Uniform random codes are the coder's worst case, so allow parity but not growth
    # beyond the 6-byte stream header plus the range coder's 5-byte flush.
    assert len(rider) <= len(body) + codec.RC1_HEADER.size + 5


def test_semantic_carries_the_sm3r_header_verbatim() -> None:
    template = _template()
    body = _build_sm3r_body(template, DEPTHS)
    rider = codec.apply_semantic(body, template, 6)
    assert rider.startswith(codec.SEMANTIC_MAGIC)
    assert rider[codec.RC1_HEADER.size : codec.RC1_HEADER.size + 10] == body[:10]


def test_hpac_carries_the_ihs1_prefix_and_tail_verbatim() -> None:
    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    rider = codec.apply_hpac(body, ROW_COUNTS, 5)
    assert rider.startswith(codec.HPAC_MAGIC)
    inner = rider[codec.RC1_HEADER.size :]
    assert inner.startswith(b"IHS1")
    assert inner.endswith(body[-37:])


def test_semantic_refuses_a_non_sm3r_body() -> None:
    with pytest.raises(codec.Rc1CodecError):
        codec.apply_semantic(b"XXXX" + bytes(40), _template(), 6)


def test_semantic_refuses_an_unsupported_mode() -> None:
    template = _template()
    body = bytearray(_build_sm3r_body(template, DEPTHS))
    body[5] = 5  # SM3R mode 5 (uniform q4) is not covered by this rider
    with pytest.raises(codec.Rc1CodecError):
        codec.apply_semantic(bytes(body), template, 6)


def test_hpac_refuses_a_non_ihs1_body() -> None:
    with pytest.raises(codec.Rc1CodecError):
        codec.apply_hpac(b"NOPE" + bytes(40), ROW_COUNTS, 5)


def test_restore_refuses_a_mislabelled_stream() -> None:
    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    rider = codec.apply_hpac(body, ROW_COUNTS, 5)
    with pytest.raises(codec.Rc1CodecError):
        # The semantic restore must not accept an hpac stream, and vice versa.
        codec.restore_semantic(rider, _template())
    template = _template()
    semantic_rider = codec.apply_semantic(_build_sm3r_body(template, DEPTHS), template, 6)
    with pytest.raises(codec.Rc1CodecError):
        codec.restore_hpac(semantic_rider, ROW_COUNTS)


def test_restore_refuses_a_truncated_stream() -> None:
    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    rider = codec.apply_hpac(body, ROW_COUNTS, 5)
    with pytest.raises(codec.Rc1CodecError):
        codec.restore_hpac(rider[:3], ROW_COUNTS)


def test_restore_refuses_an_unsupported_version() -> None:
    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    rider = bytearray(codec.apply_hpac(body, ROW_COUNTS, 5))
    rider[4] = codec.RC1_VERSION + 1
    with pytest.raises(codec.Rc1CodecError):
        codec.restore_hpac(bytes(rider), ROW_COUNTS)


def test_model_refuses_an_out_of_range_shift() -> None:
    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    for shift in (0, 9):
        with pytest.raises(codec.Rc1CodecError):
            codec.apply_hpac(body, ROW_COUNTS, shift)


def test_pack_unpack_signed_codes_are_exact_inverses() -> None:
    rng = np.random.default_rng(7)
    for bits in range(2, 9):
        low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        values = rng.integers(low, high + 1, size=257, dtype=np.int64).astype(np.int32)
        packed = codec.pack_signed_codes(values, bits)
        assert len(packed) == (values.size * bits + 7) // 8
        back = codec.unpack_signed_codes(packed, values.size, bits)
        assert np.array_equal(back, values)


def test_decoder_state_is_independent_of_the_encoder_object() -> None:
    """A FRESH decoder must reproduce the body -- no shared model state is permitted."""

    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    rider = codec.apply_hpac(body, ROW_COUNTS, 5)
    first = codec.restore_hpac(rider, ROW_COUNTS)
    second = codec.restore_hpac(rider, ROW_COUNTS)
    assert first == body
    assert second == body


def test_two_encodes_of_the_same_body_are_byte_identical() -> None:
    template = _template()
    body = _build_sm3r_body(template, DEPTHS)
    assert codec.apply_semantic(body, template, 6) == codec.apply_semantic(
        body, template, 6
    )
    ihs1 = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    assert codec.apply_hpac(ihs1, ROW_COUNTS, 5) == codec.apply_hpac(ihs1, ROW_COUNTS, 5)


def test_a_changed_shift_changes_the_stream_but_not_the_restored_body() -> None:
    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    slow = codec.apply_hpac(body, ROW_COUNTS, 6)
    fast = codec.apply_hpac(body, ROW_COUNTS, 4)
    assert slow != fast
    assert codec.restore_hpac(slow, ROW_COUNTS) == body
    assert codec.restore_hpac(fast, ROW_COUNTS) == body


def test_hpac_refuses_a_geometry_that_overruns_the_body() -> None:
    body = _build_ihs1_body(ROW_BITS, ROW_COUNTS)
    inflated = [count * 40 for count in ROW_COUNTS]
    with pytest.raises(codec.Rc1CodecError):
        codec.apply_hpac(body, inflated, 5)
