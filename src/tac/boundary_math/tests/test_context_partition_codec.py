# SPDX-License-Identifier: MIT
"""NO-FAKE behavior tests for the context-adaptive partition codec.

These verify BEHAVIOR not constants:
  - encode -> decode reconstructs the label stack BIT-EXACT (a no-op decode FAILS);
  - the reported total_bytes is the REAL emitted payload length (len(payload)),
    not an estimate;
  - the temporal coder ACTUALLY beats the spatial coder on a temporally-redundant
    stack (it never increases bytes on identical-frame stacks);
  - the coder handles all-one-class, single-frame, and full-5-class partitions;
  - out-of-alphabet / shape-mismatch inputs fail closed.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.context_partition_codec import (
    PartitionStackCode,
    decode_partition_stack,
    encode_partition_stack,
    partition_stack_bytes_per_frame,
)


def _smooth_partition(h: int, w: int, seed: int) -> np.ndarray:
    """A block-structured 5-class partition (large smooth regions + a little wiggle)."""
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w), dtype=np.int64)
    a[: h // 3] = 1
    a[h // 3 : 2 * h // 3] = 0
    a[2 * h // 3 :] = 3
    a[:, : w // 4] = 4
    # sprinkle a few class-2 speckles (boundary wiggle)
    for _ in range(max(1, (h * w) // 200)):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        a[r, c] = 2
    return a


# ---------------------------------------------------------------------------
# Core NO-FAKE: bit-exact reversibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", ["spatial", "temporal"])
def test_roundtrip_bit_exact_multiframe(template):
    frames = [_smooth_partition(24, 30, s) for s in range(4)]
    code = encode_partition_stack(frames, template=template)
    decoded = decode_partition_stack(code.payload)
    assert len(decoded) == len(frames)
    for orig, dec in zip(frames, decoded, strict=True):
        assert np.array_equal(orig, dec), f"{template} not bit-exact"


@pytest.mark.parametrize("template", ["spatial", "temporal"])
def test_roundtrip_single_frame(template):
    frames = [_smooth_partition(16, 20, 7)]
    code = encode_partition_stack(frames, template=template)
    decoded = decode_partition_stack(code.payload)
    assert np.array_equal(frames[0], decoded[0])


@pytest.mark.parametrize("template", ["spatial", "temporal"])
def test_roundtrip_all_one_class(template):
    frames = [np.full((12, 14), 3, dtype=np.int64) for _ in range(3)]
    code = encode_partition_stack(frames, template=template)
    decoded = decode_partition_stack(code.payload)
    for orig, dec in zip(frames, decoded, strict=True):
        assert np.array_equal(orig, dec)
    # A constant stack should be extremely cheap (well under 1 B/frame amortized
    # for the stream; model + header are the floor).
    assert code.stream_bytes <= 64


@pytest.mark.parametrize("template", ["spatial", "temporal"])
def test_roundtrip_uses_all_five_classes(template):
    rng = np.random.default_rng(123)
    frames = [rng.integers(0, 5, (20, 25)).astype(np.int64) for _ in range(3)]
    code = encode_partition_stack(frames, template=template)
    decoded = decode_partition_stack(code.payload)
    for orig, dec in zip(frames, decoded, strict=True):
        assert np.array_equal(orig, dec)


# ---------------------------------------------------------------------------
# NO-FAKE: reported bytes are the REAL payload length, not an estimate
# ---------------------------------------------------------------------------

def test_total_bytes_equals_payload_length():
    frames = [_smooth_partition(20, 24, s) for s in range(3)]
    code = encode_partition_stack(frames, template="temporal")
    assert code.total_bytes == len(code.payload)
    # the three sections + header must sum to the payload length
    header_len = code.total_bytes - code.model_bytes - code.stream_bytes
    assert header_len > 0  # there is a real header
    assert code.model_bytes >= 0 and code.stream_bytes > 0
    assert code.bytes_per_frame == pytest.approx(code.total_bytes / code.n_frames)


def test_decode_is_not_a_noop_passthrough():
    """If decode just returned a constant/zero stack, this would fail."""
    frames = [_smooth_partition(18, 22, 5)]
    code = encode_partition_stack(frames, template="spatial")
    decoded = decode_partition_stack(code.payload)
    assert not np.array_equal(decoded[0], np.zeros_like(decoded[0]))
    assert np.array_equal(decoded[0], frames[0])


def test_convenience_bpf_matches_full_encode():
    frames = [_smooth_partition(16, 20, s) for s in range(3)]
    bpf = partition_stack_bytes_per_frame(frames, template="temporal")
    code = encode_partition_stack(frames, template="temporal")
    assert bpf == pytest.approx(code.bytes_per_frame)


# ---------------------------------------------------------------------------
# NO-FAKE: the temporal lever actually exploits temporal redundancy
# ---------------------------------------------------------------------------

def test_temporal_beats_spatial_on_redundant_stack():
    """An identical-frame stack is pure temporal redundancy: temporal stream must
    be no larger than spatial (and strictly smaller once >1 identical frame)."""
    base = _smooth_partition(40, 50, 9)
    frames = [base.copy() for _ in range(6)]
    sp = encode_partition_stack(frames, template="spatial")
    tp = encode_partition_stack(frames, template="temporal")
    # temporal predictor makes each repeat ~free; spatial re-pays boundary each frame.
    assert tp.stream_bytes <= sp.stream_bytes
    assert tp.stream_bytes < sp.stream_bytes  # strict on a 6x-repeated stack


def test_context_coder_beats_iid_lower_bound_sanity():
    """The coded stream must be << the raw label byte count (1 byte/pixel)."""
    frames = [_smooth_partition(40, 50, s) for s in range(4)]
    code = encode_partition_stack(frames, template="temporal")
    raw_bytes = sum(f.size for f in frames)  # 1 byte/label upper bound
    assert code.stream_bytes < raw_bytes // 4  # smooth partition compresses a lot


# ---------------------------------------------------------------------------
# Fail-closed input validation
# ---------------------------------------------------------------------------

def test_out_of_alphabet_fails_closed():
    bad = [np.array([[0, 1], [2, 9]], dtype=np.int64)]  # 9 >= n_classes
    with pytest.raises(ValueError):
        encode_partition_stack(bad, n_classes=5)


def test_shape_mismatch_fails_closed():
    frames = [np.zeros((4, 4), dtype=np.int64), np.zeros((4, 5), dtype=np.int64)]
    with pytest.raises(ValueError):
        encode_partition_stack(frames)


def test_empty_stack_fails_closed():
    with pytest.raises(ValueError):
        encode_partition_stack([])


def test_bad_magic_fails_closed():
    with pytest.raises(ValueError):
        decode_partition_stack(b"NOPEnotacodecpayload________")


def test_returned_type_is_dataclass():
    frames = [_smooth_partition(10, 10, 1)]
    code = encode_partition_stack(frames)
    assert isinstance(code, PartitionStackCode)
    assert code.template in ("spatial", "temporal")
