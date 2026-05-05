"""Tests for pr106_yshift_sidechannel — wire format + roundtrip + builder.

Covers:
  - SC01 YSHIFT payload encode/decode roundtrip (numerics)
  - Outer 0xFC dispatch byte + uint24 PR106 length wrapper
  - Brotli round-trip on zero-init payload (CPU smoke / wire-format proof)
  - Magic-byte anti-corruption guards
  - shift_rgb_uint8 numerical correctness
  - apply_yshift composes (Y_offset clip, then shift) correctly

Mirrors the test_apogee_v2_parser_roundtrip.py + sister sidechannel patterns.
CPU-only — does not load CUDA scorers (per CLAUDE.md strict-scorer-rule, scorers
are NEVER loaded at inflate time).

Per CLAUDE.md MPS-noise rule: any score-producing assertion would need
[contest-CUDA] tag — this file ONLY tests bytewise wire format + numerics,
which are CUDA-independent.
"""
from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_ARCHIVE = REPO_ROOT / (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
INFLATE_DIR = REPO_ROOT / "submissions/pr106_yshift_sidechannel"
APOGEE_SRC = REPO_ROOT / "submissions/apogee_intN/src"


def _load_inflate():
    sys.modules.pop("inflate", None)  # avoid sister-module collision
    sys.path.insert(0, str(INFLATE_DIR))
    sys.path.insert(0, str(APOGEE_SRC))
    import inflate  # type: ignore[import-not-found]
    return inflate


def _load_builder():
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import build_pr106_yshift_sidechannel as builder  # type: ignore[import-not-found]
    return builder


def test_sc01_constants():
    """Magic bytes + struct shape match the codex_metric_yshift SC01 mode-7 wire."""
    inflate = _load_inflate()
    assert inflate.SC01_MAGIC == b"SC01"
    assert inflate.SIDECHANNEL_MODE_Y_SHIFT == 7
    assert inflate.SC01_HEADER.size == 14  # 4-byte magic + uint8 mode + uint8 ch + uint32 n + float32 step
    assert inflate.YSHIFT_MAGIC_BYTE == 0xFC
    assert inflate.SIDECHANNEL_VERSION == 1


def test_sc01_encode_decode_roundtrip_zero():
    """All-zero corrections roundtrip exactly through SC01 + brotli."""
    builder = _load_builder()
    inflate = _load_inflate()
    n_frames = 1200
    values = np.zeros((n_frames, 3), dtype=np.int8)
    raw = builder._encode_sc01_yshift(values, step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    decoded = inflate.decode_sidechannel_blob(blob)
    assert decoded["mode_id"] == 7
    assert decoded["channels"] == 3
    assert decoded["step"] == 1.0
    assert decoded["raw"].shape == (n_frames, 3)
    assert np.array_equal(decoded["raw"], values)


def test_sc01_encode_decode_roundtrip_random():
    """Random int8 corrections roundtrip exactly through SC01 + brotli."""
    builder = _load_builder()
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=42)
    n_frames = 1200
    values = rng.integers(-127, 128, size=(n_frames, 3), dtype=np.int8)
    raw = builder._encode_sc01_yshift(values, step=0.5)
    import brotli
    blob = brotli.compress(raw, quality=11)
    decoded = inflate.decode_sidechannel_blob(blob)
    assert decoded["raw"].dtype == np.int8
    assert decoded["raw"].shape == (n_frames, 3)
    assert np.array_equal(decoded["raw"], values)
    assert abs(decoded["step"] - 0.5) < 1e-6


def test_sc01_decode_rejects_bad_magic():
    """SC01 parser raises on wrong magic (anti-corruption)."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBIf", b"BAD!", 7, 3, 1200, 1.0) + (b"\x00" * 3600)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="bad SC01 magic"):
        inflate.decode_sidechannel_blob(blob)


def test_sc01_decode_rejects_wrong_mode():
    """SC01 parser raises on non-7 mode (we only support YSHIFT here)."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBIf", b"SC01", 6, 2, 1200, 1.0) + (b"\x00" * 2400)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="unsupported sidechannel mode_id"):
        inflate.decode_sidechannel_blob(blob)


def test_sc01_decode_rejects_wrong_channel_count():
    """SC01 parser raises if YSHIFT mode has != 3 channels."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBIf", b"SC01", 7, 2, 1200, 1.0) + (b"\x00" * 2400)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="YSHIFT expects 3 channels"):
        inflate.decode_sidechannel_blob(blob)


def test_shift_rgb_uint8_zero():
    """shift_rgb_uint8(0, 0) returns the input frame unmodified."""
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=7)
    frame = rng.integers(0, 256, size=(384, 512, 3), dtype=np.uint8)
    out = inflate.shift_rgb_uint8(frame, 0, 0)
    assert np.array_equal(out, frame)


def test_shift_rgb_uint8_translation():
    """A (dy=2, dx=3) translation moves pixels by (2 rows down, 3 cols right).

    Codex shift_rgb pattern: out = frame.copy(), then out[dst_slice] = frame[src_slice].
    So (10, 20) gets overwritten with frame[8, 17] = 0; (12, 23) becomes the moved pixel.
    """
    inflate = _load_inflate()
    h, w = 32, 48
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[10, 20] = [123, 45, 67]  # set one pixel
    out = inflate.shift_rgb_uint8(frame, 2, 3)
    # Pixel (10, 20) → (12, 23) after dy=2, dx=3 shift
    assert tuple(out[12, 23]) == (123, 45, 67)
    # Edge column 0..2 retains zero (source [src_x0=0:src_x1=w-3] starts at column 0,
    # so column 0..2 of the output is the un-shifted fallback = zero in this test).
    assert tuple(out[12, 0]) == (0, 0, 0)
    # Original (10, 20) is overwritten by the shifted source = frame[8, 17] = zero.
    assert tuple(out[10, 20]) == (0, 0, 0)


def test_shift_rgb_uint8_negative_translation():
    """A (dy=-1, dx=-1) translation moves pixels up-left."""
    inflate = _load_inflate()
    h, w = 16, 16
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[5, 5] = [200, 100, 50]
    out = inflate.shift_rgb_uint8(frame, -1, -1)
    # Pixel (5, 5) → (4, 4)
    assert tuple(out[4, 4]) == (200, 100, 50)


def test_apply_yshift_zero():
    """apply_yshift with all-zero correction is identity."""
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=11)
    frame = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    sc_row = np.zeros(3, dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert np.array_equal(out, frame)


def test_apply_yshift_y_offset_only():
    """y_off=10, step=1.0 shifts all channels by +10 with clipping at 255."""
    inflate = _load_inflate()
    frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    sc_row = np.array([10, 0, 0], dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert (out == 110).all(), f"expected uniform 110, got min={out.min()}, max={out.max()}"


def test_apply_yshift_y_offset_clips_high():
    """y_off pushing values >255 saturates to 255."""
    inflate = _load_inflate()
    frame = np.full((4, 4, 3), 250, dtype=np.uint8)
    sc_row = np.array([20, 0, 0], dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert (out == 255).all()


def test_apply_yshift_y_offset_clips_low():
    """y_off pushing values <0 saturates to 0."""
    inflate = _load_inflate()
    frame = np.full((4, 4, 3), 5, dtype=np.uint8)
    sc_row = np.array([-20, 0, 0], dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    assert (out == 0).all()


def test_outer_archive_layout_zero_search():
    """Build with --search-mode zero produces a parseable archive +44 bytes vs PR106."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    n_frames = 1200
    values = builder._zero_search(n_frames)
    raw = builder._encode_sc01_yshift(values, step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = builder._build_yshift_archive_bytes(pr106_bytes, blob)

    # Outer wrapper structure
    assert new_bin[0] == 0xFC
    pr106_len = int.from_bytes(new_bin[1:4], "little")
    assert pr106_len == len(pr106_bytes)
    assert new_bin[4 + pr106_len] == 1  # SIDECHANNEL_VERSION

    # Roundtrip
    sd, lat, meta, sc = inflate.parse_yshift_archive(new_bin)
    assert sc is not None
    assert sc["mode_id"] == 7
    assert sc["raw"].shape == (n_frames, 3)
    assert (sc["raw"] == 0).all()
    assert len(sd) == 28
    assert tuple(lat.shape) == (600, 28)
    assert meta == {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}


def test_outer_archive_layout_no_sidechannel():
    """Build with sc01_blob=None embeds PR106 only — parses with sidechannel=None."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder._build_yshift_archive_bytes(pr106_bytes, sc01_blob=None)
    sd, lat, meta, sc = inflate.parse_yshift_archive(new_bin)
    assert sc is None
    assert len(sd) == 28


def test_outer_archive_rejects_bad_magic():
    """Outer parser raises if first byte != 0xFC."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    new_bin = bytearray(builder._build_yshift_archive_bytes(pr106_bytes, sc01_blob=None))
    new_bin[0] = 0xA5  # apogee_int5's magic — should be rejected
    with pytest.raises(ValueError, match="pr106_yshift magic mismatch"):
        inflate.parse_yshift_archive(bytes(new_bin))


def test_outer_archive_rejects_wrong_sidechannel_version():
    """Outer parser raises on unknown sidechannel version byte."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    n_frames = 1200
    raw = builder._encode_sc01_yshift(np.zeros((n_frames, 3), dtype=np.int8), step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = bytearray(builder._build_yshift_archive_bytes(pr106_bytes, blob))
    pr106_len = int.from_bytes(new_bin[1:4], "little")
    new_bin[4 + pr106_len] = 99  # bogus version
    with pytest.raises(ValueError, match="sidechannel version mismatch"):
        inflate.parse_yshift_archive(bytes(new_bin))


def test_search_mode_gradient_raises_without_cuda():
    """Gradient mode is a stub that raises NotImplementedError — must be invoked via CUDA dispatch."""
    builder = _load_builder()
    with pytest.raises(NotImplementedError, match="gradient search mode requires CUDA"):
        builder._gradient_search_stub(1200)


def test_search_mode_brute_force_raises_without_cuda():
    """Brute-force mode is a stub that raises NotImplementedError — must be invoked via CUDA dispatch."""
    builder = _load_builder()
    with pytest.raises(NotImplementedError, match="brute_force search mode requires CUDA"):
        builder._brute_force_search_stub(1200)


def test_archive_size_overhead_under_2kb():
    """Zero-init brotli'd SC01 + 6-byte outer wrapper adds < 2KB to PR106 archive."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    n_frames = 1200
    values = builder._zero_search(n_frames)
    raw = builder._encode_sc01_yshift(values, step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = builder._build_yshift_archive_bytes(pr106_bytes, blob)
    overhead = len(new_bin) - len(pr106_bytes)
    # 6-byte outer wrapper + ~37 byte brotli'd zero payload = ~43 bytes
    assert overhead < 2048, f"overhead {overhead} bytes exceeds 2KB ceiling"
    assert overhead < 100, f"zero-init overhead {overhead} should be tiny (~50 bytes)"


def test_apply_yshift_pure_shift():
    """Pure (dy, dx) translation with y_off=0 applies shift without DC change."""
    inflate = _load_inflate()
    h, w = 16, 16
    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    frame[8, 8] = [200, 50, 30]
    sc_row = np.array([0, 1, 1], dtype=np.int8)
    out = inflate.apply_yshift(frame, sc_row, step=1.0)
    # Pixel (8, 8) → (9, 9)
    assert tuple(out[9, 9]) == (200, 50, 30)
    # No DC shift applied (y_off=0)
    # Original (8, 8) is overwritten by the shifted-source fallback pattern;
    # at minimum no clipping happens elsewhere
    assert out.dtype == np.uint8
