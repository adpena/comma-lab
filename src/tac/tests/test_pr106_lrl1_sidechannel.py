"""Tests for pr106_lrl1_sidechannel — wire format + roundtrip + builder + apply.

Covers:
  - LR01 LRL1 payload encode/decode roundtrip (numerics, basis + coeffs)
  - Outer 0xFB dispatch byte + uint24 PR106 length wrapper
  - Brotli round-trip on zero-init payload (CPU smoke / wire-format proof)
  - Magic-byte + version anti-corruption guards
  - upsample_basis bilinear correctness (resolution + scale)
  - apply_lrl1_to_frame numerical correctness (zero, single-component, clipping)
  - Builder _zero_search produces consistent (basis, coeffs) shapes
  - Search-mode stubs (gradient, brute_force) raise NotImplementedError without CUDA

Mirrors the test_pr106_yshift_sidechannel.py + test_pr106_latent_sidecar.py patterns.
CPU-only — does not load CUDA scorers (per CLAUDE.md strict-scorer-rule, scorers
are NEVER loaded at inflate time).

Per CLAUDE.md MPS-noise rule: any score-producing assertion would need
[contest-CUDA] tag — this file ONLY tests bytewise wire format + numerics,
which are CUDA-independent.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.repo_io import read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
PR106_ARCHIVE = REPO_ROOT / (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
INFLATE_DIR = REPO_ROOT / "submissions/pr106_lrl1_sidechannel"
APOGEE_SRC = REPO_ROOT / "submissions/apogee_intN/src"


def _load_inflate():
    sys.modules.pop("inflate", None)  # avoid sister-module collision
    sys.path.insert(0, str(INFLATE_DIR))
    sys.path.insert(0, str(APOGEE_SRC))
    import inflate  # type: ignore[import-not-found]
    return inflate


def _load_builder():
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import build_pr106_lrl1_sidechannel as builder  # type: ignore[import-not-found]
    return builder


def test_lr01_constants():
    """Magic bytes + struct shape match the codex_metric LRL1 mode-8 wire."""
    inflate = _load_inflate()
    assert inflate.LR01_MAGIC == b"LR01"
    assert inflate.SIDECHANNEL_MODE_LRL1 == 8
    # 4-byte magic + uint8 mode + uint8 K + uint16 low_h + uint16 low_w
    # + uint32 n_frames + float32 coeff_step + float32 basis_step = 22 bytes
    assert inflate.LR01_HEADER.size == 22
    assert inflate.LRL1_MAGIC_BYTE == 0xFB
    assert inflate.SIDECHANNEL_VERSION == 1


def test_lr01_encode_decode_roundtrip_zero():
    """All-zero basis+coeffs roundtrip exactly through LR01 + brotli."""
    builder = _load_builder()
    inflate = _load_inflate()
    K, low_h, low_w, n_frames = 4, 48, 64, 1200
    basis = np.zeros((K, low_h, low_w), dtype=np.int8)
    coeffs = np.zeros((n_frames, K), dtype=np.int8)
    raw = builder._encode_lr01_lrl1(basis, coeffs, basis_step=1.0, coeff_step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    decoded = inflate.decode_sidechannel_blob(blob)
    assert decoded["mode_id"] == 8
    assert decoded["K"] == K
    assert decoded["low_h"] == low_h
    assert decoded["low_w"] == low_w
    assert decoded["n_frames"] == n_frames
    assert decoded["coeff_step"] == 1.0
    assert decoded["basis_step"] == 1.0
    assert decoded["basis"].shape == (K, low_h, low_w)
    assert decoded["coeffs"].shape == (n_frames, K)
    assert np.array_equal(decoded["basis"], basis)
    assert np.array_equal(decoded["coeffs"], coeffs)


def test_lr01_encode_decode_roundtrip_random():
    """Random int8 basis+coeffs roundtrip exactly through LR01 + brotli."""
    builder = _load_builder()
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=42)
    K, low_h, low_w, n_frames = 2, 24, 32, 1200
    basis = rng.integers(-127, 128, size=(K, low_h, low_w), dtype=np.int8)
    coeffs = rng.integers(-127, 128, size=(n_frames, K), dtype=np.int8)
    raw = builder._encode_lr01_lrl1(basis, coeffs, basis_step=0.5, coeff_step=0.25)
    import brotli
    blob = brotli.compress(raw, quality=11)
    decoded = inflate.decode_sidechannel_blob(blob)
    assert decoded["basis"].dtype == np.int8
    assert decoded["coeffs"].dtype == np.int8
    assert np.array_equal(decoded["basis"], basis)
    assert np.array_equal(decoded["coeffs"], coeffs)
    assert abs(decoded["basis_step"] - 0.5) < 1e-6
    assert abs(decoded["coeff_step"] - 0.25) < 1e-6


def test_lr01_decode_rejects_bad_magic():
    """LR01 parser raises on wrong magic (anti-corruption)."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBHHIff", b"BAD!", 8, 4, 48, 64, 1200, 1.0, 1.0)
    bad += b"\x00" * (4 * 48 * 64 + 1200 * 4)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="bad LR01 magic"):
        inflate.decode_sidechannel_blob(blob)


def test_lr01_decode_rejects_wrong_mode():
    """LR01 parser raises on non-8 mode (we only support LRL1 here)."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBHHIff", b"LR01", 7, 4, 48, 64, 1200, 1.0, 1.0)
    bad += b"\x00" * (4 * 48 * 64 + 1200 * 4)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="unsupported sidechannel mode_id"):
        inflate.decode_sidechannel_blob(blob)


def test_lr01_decode_rejects_zero_K():
    """LR01 parser raises if K=0 (would be empty payload)."""
    inflate = _load_inflate()
    import brotli
    bad = struct.pack("<4sBBHHIff", b"LR01", 8, 0, 48, 64, 1200, 1.0, 1.0)
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="LRL1 expects K >= 1"):
        inflate.decode_sidechannel_blob(blob)


def test_lr01_decode_rejects_bad_length():
    """LR01 parser raises if payload length doesn't match (K, low_h, low_w, n_frames)."""
    inflate = _load_inflate()
    import brotli
    # Header says K=4, basis=48x64, n_frames=1200 → expected 22 + 12288 + 4800 = 17110
    # But payload has wrong length
    bad = struct.pack("<4sBBHHIff", b"LR01", 8, 4, 48, 64, 1200, 1.0, 1.0)
    bad += b"\x00" * 100  # Wrong length
    blob = brotli.compress(bad, quality=11)
    with pytest.raises(ValueError, match="bad LR01 length"):
        inflate.decode_sidechannel_blob(blob)


def test_upsample_basis_zero():
    """upsample_basis on zero input returns zero of target shape."""
    inflate = _load_inflate()
    K, low_h, low_w = 4, 48, 64
    basis = np.zeros((K, low_h, low_w), dtype=np.int8)
    up = inflate.upsample_basis(basis, basis_step=1.0, target_h=874, target_w=1164)
    assert up.shape == (K, 874, 1164)
    assert torch.all(up == 0)


def test_upsample_basis_constant():
    """upsample_basis on constant int8 input returns constant float (× basis_step)."""
    inflate = _load_inflate()
    K, low_h, low_w = 2, 16, 16
    basis = np.full((K, low_h, low_w), 10, dtype=np.int8)
    up = inflate.upsample_basis(basis, basis_step=0.5, target_h=64, target_w=64)
    assert up.shape == (K, 64, 64)
    # Bilinear of constant is constant; with step=0.5 → 5.0
    assert torch.allclose(up, torch.full((K, 64, 64), 5.0))


def test_upsample_basis_shape():
    """upsample_basis correctly resizes (K, low_h, low_w) → (K, target_h, target_w)."""
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=7)
    K, low_h, low_w = 8, 32, 48
    basis = rng.integers(-50, 51, size=(K, low_h, low_w), dtype=np.int8)
    up = inflate.upsample_basis(basis, basis_step=1.0, target_h=384, target_w=512)
    assert up.shape == (K, 384, 512)
    assert up.dtype == torch.float32


def test_apply_lrl1_zero_correction():
    """apply_lrl1_to_frame with zero coeffs is identity."""
    inflate = _load_inflate()
    rng = np.random.default_rng(seed=11)
    h, w = 64, 64
    K = 4
    frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    basis = torch.randn(K, h, w)  # Already upsampled
    coeffs = np.zeros(K, dtype=np.int8)
    out = inflate.apply_lrl1_to_frame(frame, basis, coeffs, coeff_step=1.0)
    assert np.array_equal(out, frame)


def test_apply_lrl1_single_component():
    """K=1 with constant basis=1.0 + coeff=10, step=1.0 adds 10 uniformly."""
    inflate = _load_inflate()
    h, w = 16, 16
    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    basis = torch.ones(1, h, w)  # Already upsampled
    coeffs = np.array([10], dtype=np.int8)
    out = inflate.apply_lrl1_to_frame(frame, basis, coeffs, coeff_step=1.0)
    assert (out == 110).all()


def test_apply_lrl1_clips_high():
    """LRL1 correction pushing values >255 saturates to 255."""
    inflate = _load_inflate()
    h, w = 8, 8
    frame = np.full((h, w, 3), 250, dtype=np.uint8)
    basis = torch.ones(1, h, w)
    coeffs = np.array([20], dtype=np.int8)
    out = inflate.apply_lrl1_to_frame(frame, basis, coeffs, coeff_step=1.0)
    assert (out == 255).all()


def test_apply_lrl1_clips_low():
    """LRL1 correction pushing values <0 saturates to 0."""
    inflate = _load_inflate()
    h, w = 8, 8
    frame = np.full((h, w, 3), 5, dtype=np.uint8)
    basis = torch.ones(1, h, w)
    coeffs = np.array([-20], dtype=np.int8)
    out = inflate.apply_lrl1_to_frame(frame, basis, coeffs, coeff_step=1.0)
    assert (out == 0).all()


def test_apply_lrl1_broadcasts_across_rgb():
    """LRL1 correction is identical on R, G, B channels (luma broadcast)."""
    inflate = _load_inflate()
    h, w = 16, 16
    rng = np.random.default_rng(seed=99)
    frame = rng.integers(50, 200, size=(h, w, 3), dtype=np.uint8).astype(np.uint8)
    K = 2
    basis = torch.randn(K, h, w)
    coeffs = np.array([5, -3], dtype=np.int8)
    out = inflate.apply_lrl1_to_frame(frame, basis, coeffs, coeff_step=0.5)
    # The delta from frame should be IDENTICAL across R, G, B channels
    delta_r = out[..., 0].astype(np.int16) - frame[..., 0].astype(np.int16)
    delta_g = out[..., 1].astype(np.int16) - frame[..., 1].astype(np.int16)
    delta_b = out[..., 2].astype(np.int16) - frame[..., 2].astype(np.int16)
    # Where no clipping happened, delta should match across channels
    no_clip_mask = (out > 0).all(axis=-1) & (out < 255).all(axis=-1)
    assert np.array_equal(delta_r[no_clip_mask], delta_g[no_clip_mask])
    assert np.array_equal(delta_g[no_clip_mask], delta_b[no_clip_mask])


def test_outer_archive_layout_zero_search():
    """Build with --search-mode zero produces a parseable archive with small overhead."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    K, low_h, low_w, n_frames = 4, 48, 64, 1200
    basis, coeffs = builder._zero_search(K, low_h, low_w, n_frames)
    raw = builder._encode_lr01_lrl1(basis, coeffs, basis_step=1.0, coeff_step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = builder._build_lrl1_archive_bytes(pr106_bytes, blob)

    # Outer wrapper structure
    assert new_bin[0] == 0xFB
    pr106_len = int.from_bytes(new_bin[1:4], "little")
    assert pr106_len == len(pr106_bytes)
    assert new_bin[4 + pr106_len] == 1  # SIDECHANNEL_VERSION

    # Roundtrip
    sd, lat, meta, sc = inflate.parse_lrl1_archive(new_bin)
    assert sc is not None
    assert sc["mode_id"] == 8
    assert sc["K"] == K
    assert sc["low_h"] == low_h
    assert sc["low_w"] == low_w
    assert sc["basis"].shape == (K, low_h, low_w)
    assert sc["coeffs"].shape == (n_frames, K)
    assert (sc["basis"] == 0).all()
    assert (sc["coeffs"] == 0).all()
    assert len(sd) == 28
    assert tuple(lat.shape) == (600, 28)
    assert meta == {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}


def test_outer_archive_layout_no_sidechannel():
    """Build with lr01_blob=None embeds PR106 only — parses with sidechannel=None."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    new_bin = builder._build_lrl1_archive_bytes(pr106_bytes, lr01_blob=None)
    sd, lat, meta, sc = inflate.parse_lrl1_archive(new_bin)
    assert sc is None
    assert len(sd) == 28


def test_outer_archive_rejects_bad_magic():
    """Outer parser raises if first byte != 0xFB."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    new_bin = bytearray(builder._build_lrl1_archive_bytes(pr106_bytes, lr01_blob=None))
    new_bin[0] = 0xFC  # yshift's magic — should be rejected
    with pytest.raises(ValueError, match="pr106_lrl1 magic mismatch"):
        inflate.parse_lrl1_archive(bytes(new_bin))


def test_outer_archive_rejects_wrong_sidechannel_version():
    """Outer parser raises on unknown sidechannel version byte."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()
    inflate = _load_inflate()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    K, low_h, low_w, n_frames = 2, 16, 24, 1200
    basis, coeffs = builder._zero_search(K, low_h, low_w, n_frames)
    raw = builder._encode_lr01_lrl1(basis, coeffs, basis_step=1.0, coeff_step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = bytearray(builder._build_lrl1_archive_bytes(pr106_bytes, blob))
    pr106_len = int.from_bytes(new_bin[1:4], "little")
    new_bin[4 + pr106_len] = 99  # bogus version
    with pytest.raises(ValueError, match="sidechannel version mismatch"):
        inflate.parse_lrl1_archive(bytes(new_bin))


def test_search_mode_gradient_raises_without_cuda():
    """Gradient mode is a stub that raises NotImplementedError — must be invoked via CUDA dispatch."""
    builder = _load_builder()
    with pytest.raises(NotImplementedError, match="gradient search mode requires CUDA"):
        builder._gradient_search_stub(4, 48, 64, 1200)


def test_search_mode_brute_force_raises_without_cuda():
    """Brute-force mode is a stub that raises NotImplementedError — must be invoked via CUDA dispatch."""
    builder = _load_builder()
    with pytest.raises(NotImplementedError, match="brute_force search mode requires CUDA"):
        builder._brute_force_search_stub(4, 48, 64, 1200)


def test_archive_size_overhead_under_10kb_for_K4():
    """Zero-init brotli'd LR01 (K=4, 48x64 basis) + 6-byte outer wrapper adds < 10KB."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    builder = _load_builder()

    pr106_bytes = builder._read_pr106_bytes(PR106_ARCHIVE)
    K, low_h, low_w, n_frames = 4, 48, 64, 1200
    basis, coeffs = builder._zero_search(K, low_h, low_w, n_frames)
    raw = builder._encode_lr01_lrl1(basis, coeffs, basis_step=1.0, coeff_step=1.0)
    import brotli
    blob = brotli.compress(raw, quality=11)
    new_bin = builder._build_lrl1_archive_bytes(pr106_bytes, blob)
    overhead = len(new_bin) - len(pr106_bytes)
    # Zero payload is highly compressible; expect ~30-100 bytes total
    assert overhead < 10240, f"overhead {overhead} bytes exceeds 10KB ceiling"
    assert overhead < 200, f"zero-init overhead {overhead} should be tiny (~50 bytes)"


def test_builder_zero_mode_metadata_is_fail_closed(tmp_path):
    """Zero-mode LRL1 build records deterministic custody but remains non-dispatchable."""
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"PR106 anchor not present at {PR106_ARCHIVE}")
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/build_pr106_lrl1_sidechannel.py"),
            "--pr106-archive",
            str(PR106_ARCHIVE),
            "--out-dir",
            str(out_dir),
            "--search-mode",
            "zero",
            "--K",
            "2",
            "--low-h",
            "8",
            "--low-w",
            "8",
            "--n-pairs",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    archive_path = out_dir / "pr106_lrl1_sidechannel_archive.zip"
    metadata = read_json(out_dir / "build_metadata.json")
    assert metadata["manifest_schema"] == "pr106_lrl1_sidechannel_build_metadata_v2"
    assert metadata["score_claim"] is False
    assert metadata["dispatch_attempted"] is False
    assert metadata["remote_jobs_dispatched"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["archive_sha256"] == sha256_file(archive_path)
    assert metadata["source_archive_sha256"] == sha256_file(PR106_ARCHIVE)
    assert "requires_real_cuda_lrl1_search" in metadata["dispatch_blockers"]
    assert "requires_exact_cuda_auth_eval_on_built_archive" in metadata["dispatch_blockers"]


def test_zero_search_shapes():
    """_zero_search returns correctly shaped int8 arrays."""
    builder = _load_builder()
    K, low_h, low_w, n_frames = 4, 48, 64, 1200
    basis, coeffs = builder._zero_search(K, low_h, low_w, n_frames)
    assert basis.dtype == np.int8
    assert coeffs.dtype == np.int8
    assert basis.shape == (K, low_h, low_w)
    assert coeffs.shape == (n_frames, K)
    assert (basis == 0).all()
    assert (coeffs == 0).all()


def test_encode_rejects_K_mismatch():
    """_encode_lr01_lrl1 raises if basis K != coeffs K."""
    builder = _load_builder()
    basis = np.zeros((4, 48, 64), dtype=np.int8)
    coeffs = np.zeros((1200, 8), dtype=np.int8)  # K=8 mismatch
    with pytest.raises(ValueError, match="basis K=4 doesn't match coeffs K=8"):
        builder._encode_lr01_lrl1(basis, coeffs, basis_step=1.0, coeff_step=1.0)


def test_encode_rejects_wrong_dtype():
    """_encode_lr01_lrl1 raises if basis/coeffs aren't int8."""
    builder = _load_builder()
    basis_f = np.zeros((4, 48, 64), dtype=np.float32)
    coeffs = np.zeros((1200, 4), dtype=np.int8)
    with pytest.raises(TypeError, match="basis must be int8"):
        builder._encode_lr01_lrl1(basis_f, coeffs, basis_step=1.0, coeff_step=1.0)
