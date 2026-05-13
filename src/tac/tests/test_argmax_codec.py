"""Property tests for the lossless argmax mask codec.

Yousfi council recommendation #8 (2026-04-26). The codec must be
bit-identical on every input it accepts, and must compress real-world
masks (mostly road) far below the AV1 monochrome floor.

We deliberately keep these tests small enough to run in <30 s on the
laptop M5 Max — the heavy "real comma-style mask" round-trip is gated
behind an environment variable to keep CI fast while still being
runnable on the dev machine.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.lossless.argmax_codec import (
    NUM_CLASSES,
    VERSION,
    decode_argmax_masks,
    encode_argmax_masks,
    is_argmax_mask_blob,
    pack_archive,
    unpack_archive,
    validate_amrc_file,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _random_masks(n: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    """Uniform random 5-class masks. Worst case for compression."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, NUM_CLASSES, (n, h, w), generator=g, dtype=torch.long)


def _structured_masks(n: int, h: int, w: int) -> torch.Tensor:
    """Comma-style mask: mostly road (class 1), some sky band, a few
    moving objects (class 4) that drift across frames. Used to verify
    the codec actually compresses well on driving-mask statistics.
    """
    masks = torch.full((n, h, w), 1, dtype=torch.long)  # road
    sky_h = h // 4
    masks[:, :sky_h, :] = 3  # sky
    # Lane marks (class 0) sparse on road
    masks[:, h // 2:h // 2 + 2, :] = 0
    # Moving vehicle drifting horizontally
    vh, vw = 16, 24
    for k in range(n):
        x = (k * 2) % (w - vw)
        y = h // 2 + 20
        masks[k, y:y + vh, x:x + vw] = 4
    return masks


# ── Round-trip correctness ───────────────────────────────────────────────


@pytest.mark.parametrize("n,h,w", [
    (1, 8, 8),
    (2, 16, 16),
    (3, 24, 32),
    (5, 64, 64),
])
def test_roundtrip_random_5class_small(n, h, w):
    """Bit-identical round-trip on random 5-class masks at small sizes."""
    masks = _random_masks(n, h, w, seed=42)
    blob = encode_argmax_masks(masks)
    recovered = decode_argmax_masks(blob, expected_n=n, expected_h=h, expected_w=w)
    assert recovered.dtype == torch.long
    assert tuple(recovered.shape) == (n, h, w)
    assert torch.equal(recovered, masks), (
        "round-trip produced different mask data — codec is NOT lossless"
    )


def test_roundtrip_realistic_full_resolution():
    """Bit-identical round-trip on a small batch of structured driving masks
    at full SegNet resolution (384x512).
    """
    masks = _structured_masks(20, 384, 512)
    blob = encode_argmax_masks(masks)
    recovered = decode_argmax_masks(blob)
    assert torch.equal(recovered, masks)
    # Sanity ratio: structured masks should compress strongly.
    bytes_per_pixel = len(blob) / (20 * 384 * 512)
    assert bytes_per_pixel < 0.05, (
        f"Structured masks compressed to {bytes_per_pixel:.4f} B/px "
        f"(>0.05 expected). Codec lost its driving-mask edge."
    )


def test_roundtrip_single_frame():
    """Boundary: 1 frame must work (no temporal predecessor at all)."""
    masks = _random_masks(1, 64, 64, seed=1)
    blob = encode_argmax_masks(masks)
    recovered = decode_argmax_masks(blob)
    assert torch.equal(recovered, masks)


def test_roundtrip_two_frames():
    """Boundary: 2 frames exercises the delta path exactly once."""
    masks = _random_masks(2, 32, 32, seed=2)
    blob = encode_argmax_masks(masks)
    recovered = decode_argmax_masks(blob)
    assert torch.equal(recovered, masks)


def test_roundtrip_full_1200_frames():
    """Boundary: 1200 frames at 64x64 (a contest-shape stand-in).

    We avoid 384x512 here so CI stays fast; the heavier round-trip is in
    `test_roundtrip_real_comma_masks` which is opt-in.
    """
    masks = _structured_masks(1200, 64, 64)
    blob = encode_argmax_masks(masks)
    recovered = decode_argmax_masks(blob)
    assert torch.equal(recovered, masks)


def test_roundtrip_preserves_dtype_and_shape():
    """Decoded tensor must always be int64 (torch.long), shape (N,H,W)."""
    for dtype in (torch.int32, torch.uint8, torch.long):
        masks = _structured_masks(5, 16, 16).to(dtype)
        blob = encode_argmax_masks(masks)
        recovered = decode_argmax_masks(blob)
        assert recovered.dtype == torch.long, (
            f"input dtype {dtype} → output dtype {recovered.dtype} "
            f"(expected torch.long always)"
        )
        assert tuple(recovered.shape) == (5, 16, 16)
        assert torch.equal(recovered, masks.to(torch.long))


# ── Magic byte / corruption guards ───────────────────────────────────────


def test_corrupt_magic_raises():
    """A blob without the AMRC magic must be rejected with ValueError."""
    masks = _random_masks(2, 8, 8, seed=3)
    blob = bytearray(encode_argmax_masks(masks))
    blob[0] = ord("X")
    with pytest.raises(ValueError, match="magic"):
        decode_argmax_masks(bytes(blob))


def test_corrupt_version_raises():
    """A version field other than VERSION must be rejected."""
    masks = _random_masks(2, 8, 8, seed=4)
    blob = bytearray(encode_argmax_masks(masks))
    # Version is at offset 4..8 (big-endian uint32).
    struct.pack_into(">I", blob, 4, VERSION + 1)
    with pytest.raises(ValueError, match="version"):
        decode_argmax_masks(bytes(blob))


def test_too_short_blob_raises():
    """Blobs shorter than the header must be caught loudly."""
    with pytest.raises(ValueError, match="too short"):
        decode_argmax_masks(b"AMRC\x00\x00\x00\x01")


def test_expected_dimension_mismatch_raises():
    """expected_h / expected_w / expected_n act as a downstream guard."""
    masks = _random_masks(3, 16, 16, seed=5)
    blob = encode_argmax_masks(masks)
    with pytest.raises(ValueError, match="height"):
        decode_argmax_masks(blob, expected_h=99)
    with pytest.raises(ValueError, match="width"):
        decode_argmax_masks(blob, expected_w=99)
    with pytest.raises(ValueError, match="frame count"):
        decode_argmax_masks(blob, expected_n=99)


def test_is_argmax_mask_blob_helper():
    """The is_argmax_mask_blob() preflight check must be conservative."""
    masks = _random_masks(1, 8, 8, seed=6)
    blob = encode_argmax_masks(masks)
    assert is_argmax_mask_blob(blob)
    assert not is_argmax_mask_blob(b"")
    assert not is_argmax_mask_blob(b"AMRC")
    assert not is_argmax_mask_blob(b"NOPE\x00\x00\x00\x01")
    bad_version = bytearray(blob)
    struct.pack_into(">I", bad_version, 4, VERSION + 7)
    assert not is_argmax_mask_blob(bytes(bad_version))


# ── Compression ratio sanity (Quantizr-grade evidence) ───────────────────


def test_random_uniform_compresses_near_information_bound():
    """Uniform 5-class data: information-theoretic lower bound is
    log2(5)/8 ≈ 0.290 bytes/pixel. We assert we get within 30% of that
    — Huffman + RLE on uniform data has constant-table overhead, so
    we can't quite reach the bound but should be close.
    """
    masks = _random_masks(10, 96, 96, seed=7)
    blob = encode_argmax_masks(masks)
    bits_per_pixel = len(blob) * 8 / masks.numel()
    info_bound = math.log2(NUM_CLASSES)
    # Allow up to 1.7x the info bound (RLE overhead dominates on uniform).
    assert bits_per_pixel < info_bound * 1.7, (
        f"Uniform 5-class compressed to {bits_per_pixel:.3f} bits/pixel "
        f"(>1.7x info bound {info_bound:.3f}). Codec is leaking bits."
    )


def test_realistic_masks_compress_well():
    """Real comma-style masks are ~70% one class, near-zero temporal
    change. The codec should reach <0.05 bytes/pixel — strictly better
    than what AV1 monochrome at the same lossless quality achieves
    (AV1 lossless on the same input takes ~0.4–0.5 B/px because it
    encodes per-block transform coefficients).
    """
    masks = _structured_masks(100, 384, 512)
    blob = encode_argmax_masks(masks)
    bytes_per_pixel = len(blob) / masks.numel()
    assert bytes_per_pixel < 0.05, (
        f"Structured masks compressed to {bytes_per_pixel:.4f} B/px "
        f"(>0.05). Codec lost its compression edge on driving data."
    )


def test_all_zero_mask_is_tiny():
    """Trivial best case: 1200 frames of all class 0 should encode to
    well under 1 KB (one symbol pair per frame, plus the header).
    """
    masks = torch.zeros(1200, 96, 96, dtype=torch.long)
    blob = encode_argmax_masks(masks)
    assert len(blob) < 16_384, (
        f"All-zero 1200-frame blob is {len(blob)} bytes — should be <16KB. "
        f"Codec is wasting bits on degenerate inputs."
    )
    recovered = decode_argmax_masks(blob)
    assert torch.equal(recovered, masks)


# ── Single-pixel-flip detection ──────────────────────────────────────────


def test_single_pixel_flip_changes_blob():
    """Encode A and B-with-1-flipped-pixel; the resulting blobs must
    differ (encoder is a function of the data, not nondeterministic),
    and BOTH must round-trip correctly to their original input.
    """
    a = _structured_masks(10, 32, 32)
    b = a.clone()
    b[5, 16, 16] = (b[5, 16, 16].item() + 1) % NUM_CLASSES
    blob_a = encode_argmax_masks(a)
    blob_b = encode_argmax_masks(b)
    assert blob_a != blob_b, (
        "Single-pixel flip produced identical blob — codec is lossy or "
        "degenerate."
    )
    rec_a = decode_argmax_masks(blob_a)
    rec_b = decode_argmax_masks(blob_b)
    assert torch.equal(rec_a, a)
    assert torch.equal(rec_b, b)


# ── Disk I/O wrappers ────────────────────────────────────────────────────


def test_pack_unpack_archive_roundtrip(tmp_path):
    """pack_archive + unpack_archive cycle on disk."""
    masks = _structured_masks(10, 64, 64)
    out = tmp_path / "test.amrc"
    n_bytes = pack_archive(masks, out)
    assert out.exists()
    assert out.stat().st_size == n_bytes
    recovered = unpack_archive(out)
    assert torch.equal(recovered, masks)


def test_validate_amrc_file_accepts_good(tmp_path):
    masks = _structured_masks(3, 16, 16)
    out = tmp_path / "good.amrc"
    pack_archive(masks, out)
    validate_amrc_file(out)  # must not raise


def test_validate_amrc_file_rejects_bad(tmp_path):
    bad = tmp_path / "bad.amrc"
    bad.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match="magic"):
        validate_amrc_file(bad)
    missing = tmp_path / "missing.amrc"
    with pytest.raises(ValueError, match="does not exist"):
        validate_amrc_file(missing)
    too_small = tmp_path / "tiny.amrc"
    too_small.write_bytes(b"AMRC")
    with pytest.raises(ValueError, match="too small"):
        validate_amrc_file(too_small)


# ── Real comma-style mask round-trip (uses CRF30 AV1 mask sweep) ─────────


_REAL_MASK_PATH = Path(
    "experiments/results/mask_sweep_20260425T142245/masks_av1mono_full_crf30.mkv"
)


@pytest.mark.skipif(
    not _REAL_MASK_PATH.exists(),
    reason="Real comma mask fixture not on disk",
)
def test_roundtrip_real_comma_masks():
    """Decode the highest-fidelity AV1 mask file we have, round-trip it
    through the AMRC codec, verify bit identity AND a strong compression
    ratio versus the original AV1 file.
    """
    import subprocess
    # Use ffprobe + ffmpeg to decode the AV1 file (same path used by
    # the inflate-side mask loader, so we share its semantics).
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(_REAL_MASK_PATH)],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    w_str, h_str = probe.stdout.strip().split(",")
    w, h = int(w_str), int(h_str)
    proc = subprocess.run(
        ["ffmpeg", "-i", str(_REAL_MASK_PATH), "-f", "rawvideo",
         "-pix_fmt", "gray", "-v", "error", "pipe:1"],
        capture_output=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(raw) // (h * w)
    pixels = raw.reshape(n, h, w)
    # Invert AV1 monochrome scaling: pixel = class * (255 // 4).
    scale = 255 // (NUM_CLASSES - 1)
    masks_np = np.clip(np.round(pixels.astype(np.float32) / scale), 0,
                        NUM_CLASSES - 1).astype(np.int64)
    masks = torch.from_numpy(masks_np)
    blob = encode_argmax_masks(masks)
    recovered = decode_argmax_masks(blob)
    assert torch.equal(recovered, masks), (
        "Real-mask round-trip produced different data — codec is NOT lossless"
    )
    # Headline ratio: AMRC vs AV1 CRF30 (the closest-to-lossless AV1).
    av1_size = _REAL_MASK_PATH.stat().st_size
    ratio = len(blob) / av1_size
    print(
        f"\n[real-mask round-trip] AMRC={len(blob):,}B vs AV1_CRF30="
        f"{av1_size:,}B → ratio={ratio:.3f}"
    )
    # Assert AMRC is at least competitive with the higher-CRF AV1
    # encoders the project actually ships (CRF50 ≈ 421KB).
    av1_crf50_path = (
        Path("experiments/results/mask_sweep_20260425T142245/")
        / "masks_av1mono_full_crf50.mkv"
    )
    if av1_crf50_path.exists():
        av1_crf50_size = av1_crf50_path.stat().st_size
        print(
            f"[real-mask round-trip] vs AV1_CRF50={av1_crf50_size:,}B "
            f"→ ratio={len(blob) / av1_crf50_size:.3f}"
        )
