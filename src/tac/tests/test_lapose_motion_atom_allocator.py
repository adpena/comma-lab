# SPDX-License-Identifier: MIT
"""Tests for ``tac.lapose_motion_atom_allocator`` — LAPose inverse-dynamics
motion atom allocator sidecar.

Per CLAUDE.md HNeRV parity discipline 13 lessons + handoff caveat
"LAPose-inspired until a paper-faithful inverse-dynamics encoder and pose
head exist; add class/openpilot manifests, calibrate confidence, and require
a charged archive consumer before dispatch." These tests pin the byte
budget, the per-atom-class param contract, and the allocator's confidence
+ sign-of-Δscore filtering.
"""
from __future__ import annotations

import pytest

from tac.lapose_motion_atom_allocator import (
    ATOM_CLASS_PARAM_BYTES,
    LAPOSE_MOTION_ATOM_FORMAT_ID,
    MAX_ENCODED_BYTES,
    PR106_RESIDUAL_MAGIC,
    AtomCandidate,
    AtomClass,
    LaposeAtomAllocator,
    LaposeAtomStream,
    MotionAtom,
    compute_lapose_atom_stream_bytes,
    decode_class_token_params,
    decode_foveation_pull_params,
    decode_motion_atom_stream,
    decode_yaw_rate_params,
    encode_class_token_params,
    encode_foveation_pull_params,
    encode_motion_atom_stream,
    encode_yaw_rate_params,
    is_no_op,
)


def _make_yaw_atom(frame_idx: int = 0, value: float = 0.05) -> MotionAtom:
    return MotionAtom(
        atom_class=AtomClass.YAW_RATE,
        frame_idx=frame_idx,
        params=encode_yaw_rate_params(value),
    )


def _make_class_token_atom(frame_idx: int = 0, token: int = 7) -> MotionAtom:
    return MotionAtom(
        atom_class=AtomClass.CLASS_TOKEN,
        frame_idx=frame_idx,
        params=encode_class_token_params(token),
    )


def _make_foveation_pull_atom(frame_idx: int = 0) -> MotionAtom:
    return MotionAtom(
        atom_class=AtomClass.FOVEATION_PULL,
        frame_idx=frame_idx,
        params=encode_foveation_pull_params(0.5, 0.6, 0.02),
    )


def test_yaw_rate_params_round_trip():
    raw = encode_yaw_rate_params(0.0123)
    assert len(raw) == 2
    decoded = decode_yaw_rate_params(raw)
    assert abs(decoded - 0.0123) < 1e-3


def test_yaw_rate_params_extremes_clamp():
    raw = encode_yaw_rate_params(1e9)  # huge value clamps to int16 max
    decoded = decode_yaw_rate_params(raw)
    assert decoded == pytest.approx(32767 * 1e-4)


def test_class_token_params_round_trip():
    raw = encode_class_token_params(42)
    assert len(raw) == 1
    assert decode_class_token_params(raw) == 42


def test_class_token_params_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        encode_class_token_params(256)
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        encode_class_token_params(-1)


def test_foveation_pull_params_round_trip():
    raw = encode_foveation_pull_params(0.3, 0.7, 0.05)
    assert len(raw) == 6
    cx, cy, amp = decode_foveation_pull_params(raw)
    assert abs(cx - 0.3) < 1e-3
    assert abs(cy - 0.7) < 1e-3
    assert abs(amp - 0.05) < 1e-3


def test_motion_atom_validate_param_length_must_match_class():
    bad = MotionAtom(
        atom_class=AtomClass.YAW_RATE,
        frame_idx=0,
        params=b"\x00\x00\x00",  # 3 bytes but yaw needs 2
    )
    with pytest.raises(ValueError, match="param bytes"):
        bad.validate()


def test_motion_atom_byte_cost_includes_header():
    atom = _make_yaw_atom()
    # Header 6 bytes (atom_class + frame_idx + param_bytes_len) + 2 yaw params.
    assert atom.byte_cost == 8


def test_atom_stream_validate_rejects_out_of_range_frame_idx():
    bad = LaposeAtomStream(n_frames=10, atoms=[_make_yaw_atom(frame_idx=20)])
    with pytest.raises(ValueError, match=r"frame_idx 20 >= n_frames 10"):
        bad.validate()


def test_atom_stream_validate_rejects_n_frames_zero():
    bad = LaposeAtomStream(n_frames=0)
    with pytest.raises(ValueError, match="n_frames"):
        bad.validate()


def test_encode_decode_round_trip_full_stream():
    """All 4 atom classes round-trip cleanly through the wire format."""
    stream = LaposeAtomStream(
        n_frames=100,
        atoms=[
            _make_yaw_atom(frame_idx=5, value=0.03),
            _make_class_token_atom(frame_idx=10, token=3),
            _make_foveation_pull_atom(frame_idx=50),
            MotionAtom(
                atom_class=AtomClass.PITCH_RATE,
                frame_idx=99,
                params=encode_yaw_rate_params(-0.01),  # same encoder
            ),
        ],
    )
    blob = encode_motion_atom_stream(stream)
    decoded = decode_motion_atom_stream(blob)
    assert decoded.n_frames == 100
    assert len(decoded.atoms) == 4
    for orig, dec in zip(stream.atoms, decoded.atoms):
        assert orig.atom_class == dec.atom_class
        assert orig.frame_idx == dec.frame_idx
        assert orig.params == dec.params


def test_encode_blob_starts_with_correct_magic_and_format_id():
    stream = LaposeAtomStream(n_frames=10, atoms=[_make_yaw_atom()])
    blob = encode_motion_atom_stream(stream)
    assert blob[0] == PR106_RESIDUAL_MAGIC
    assert blob[1] == LAPOSE_MOTION_ATOM_FORMAT_ID


def test_encode_empty_atom_stream_round_trips():
    stream = LaposeAtomStream(n_frames=10, atoms=[])
    blob = encode_motion_atom_stream(stream)
    decoded = decode_motion_atom_stream(blob)
    assert decoded.n_frames == 10
    assert decoded.atoms == []


def test_encode_enforces_budget():
    """Packing more atoms than fit in 1 KB raises."""
    atoms = [_make_yaw_atom(frame_idx=i % 200, value=0.01 * i) for i in range(200)]
    stream = LaposeAtomStream(n_frames=200, atoms=atoms)
    with pytest.raises(ValueError, match=r"> budget"):
        encode_motion_atom_stream(stream, enforce_budget=True)


def test_decode_rejects_bad_magic():
    stream = LaposeAtomStream(n_frames=10, atoms=[_make_yaw_atom()])
    blob = bytearray(encode_motion_atom_stream(stream))
    blob[0] = 0x00
    with pytest.raises(ValueError, match="magic mismatch"):
        decode_motion_atom_stream(bytes(blob))


def test_decode_rejects_bad_format_id():
    stream = LaposeAtomStream(n_frames=10, atoms=[_make_yaw_atom()])
    blob = bytearray(encode_motion_atom_stream(stream))
    blob[1] = 0x99
    with pytest.raises(ValueError, match="format_id mismatch"):
        decode_motion_atom_stream(bytes(blob))


def test_decode_rejects_unknown_atom_class():
    stream = LaposeAtomStream(n_frames=10, atoms=[_make_yaw_atom()])
    blob = bytearray(encode_motion_atom_stream(stream))
    # First atom starts at offset 2 + 4 + 2 = 8. atom_class byte at offset 8.
    blob[8] = 0xFF
    with pytest.raises(ValueError, match="unknown atom_class"):
        decode_motion_atom_stream(bytes(blob))


def test_decode_rejects_trailing_bytes():
    stream = LaposeAtomStream(n_frames=10, atoms=[_make_yaw_atom()])
    blob = encode_motion_atom_stream(stream) + b"\xde\xad"
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_motion_atom_stream(blob)


def test_is_no_op_detects_empty_stream():
    stream = LaposeAtomStream(n_frames=10, atoms=[])
    assert is_no_op(stream)


def test_is_no_op_returns_false_when_stream_has_atoms():
    stream = LaposeAtomStream(n_frames=10, atoms=[_make_yaw_atom()])
    assert not is_no_op(stream)


def test_allocator_rejects_low_confidence_atoms():
    """Per handoff caveat: low-confidence atoms must NOT consume bytes."""
    allocator = LaposeAtomAllocator(max_total_bytes=MAX_ENCODED_BYTES, min_confidence=0.5)
    candidates = [
        AtomCandidate(atom=_make_yaw_atom(0), predicted_delta_score=-0.001, confidence=0.05),
        AtomCandidate(atom=_make_yaw_atom(1), predicted_delta_score=-0.001, confidence=0.9),
    ]
    stream = allocator.allocate(candidates, n_frames=10)
    assert len(stream.atoms) == 1
    assert stream.atoms[0].frame_idx == 1  # the high-confidence one survived


def test_allocator_rejects_positive_delta_score_candidates():
    """Positive Δscore means the contest score would go UP (worse); refuse."""
    allocator = LaposeAtomAllocator(max_total_bytes=MAX_ENCODED_BYTES)
    candidates = [
        AtomCandidate(atom=_make_yaw_atom(0), predicted_delta_score=+0.001, confidence=0.9),
    ]
    stream = allocator.allocate(candidates, n_frames=10)
    assert stream.atoms == []


def test_allocator_packs_highest_value_per_byte_first():
    """Greedy: high value_per_byte beats low."""
    allocator = LaposeAtomAllocator(max_total_bytes=20)
    # Yaw atom: 8 bytes. value/byte = 0.005/8 ~ 6.25e-4
    # Class token: 7 bytes. value/byte = 0.001/7 ~ 1.43e-4
    # With 12 bytes available after header, only ONE fits: should pick yaw.
    candidates = [
        AtomCandidate(
            atom=_make_yaw_atom(0), predicted_delta_score=-0.005, confidence=0.9
        ),
        AtomCandidate(
            atom=_make_class_token_atom(1), predicted_delta_score=-0.001, confidence=0.9
        ),
    ]
    stream = allocator.allocate(candidates, n_frames=10)
    # 12 byte budget vs 8 yaw + 7 class = 15 → only yaw fits (higher value/byte).
    assert len(stream.atoms) == 1
    assert stream.atoms[0].atom_class == AtomClass.YAW_RATE


def test_allocator_packs_multiple_atoms_when_budget_allows():
    allocator = LaposeAtomAllocator(max_total_bytes=200)
    candidates = [
        AtomCandidate(
            atom=_make_yaw_atom(i), predicted_delta_score=-0.001 * (10 - i), confidence=0.9
        )
        for i in range(10)
    ]
    stream = allocator.allocate(candidates, n_frames=20)
    assert len(stream.atoms) >= 5  # several should fit
    # Highest-Δscore atom (i=0) should be first.
    assert stream.atoms[0].frame_idx == 0


def test_allocator_rejects_bad_min_confidence():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LaposeAtomAllocator(max_total_bytes=512, min_confidence=1.5)


def test_allocator_rejects_too_small_budget():
    with pytest.raises(ValueError, match="max_total_bytes"):
        LaposeAtomAllocator(max_total_bytes=4)


def test_compute_bytes_matches_encode_length():
    stream = LaposeAtomStream(n_frames=10, atoms=[_make_yaw_atom(0), _make_class_token_atom(5)])
    assert compute_lapose_atom_stream_bytes(stream) == len(encode_motion_atom_stream(stream))


def test_atom_class_param_bytes_table_complete():
    """Every AtomClass enum entry has a registered byte size."""
    for member in AtomClass:
        assert member in ATOM_CLASS_PARAM_BYTES


def test_byte_budget_under_1kb_for_typical_stream():
    """Typical stream of ~30 atoms must fit within 1 KB."""
    atoms = [_make_yaw_atom(i, value=0.01 * i) for i in range(30)]
    stream = LaposeAtomStream(n_frames=100, atoms=atoms)
    blob = encode_motion_atom_stream(stream)
    assert len(blob) <= MAX_ENCODED_BYTES
