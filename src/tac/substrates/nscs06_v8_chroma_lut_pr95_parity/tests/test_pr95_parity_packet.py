# SPDX-License-Identifier: MIT
"""Test the NSCS06 v8 chroma_lut + cls_stream PR-95-parity packet.

Per HNeRV parity discipline lessons L11 (no-op detector via byte-mutation smoke)
+ L9 (runtime closure tested in clean env BEFORE dispatch) + Catalog #287
(empirical-claim-without-evidence-tag) + Catalog #229 (premise-verification).

13 inviolable lesson empirical coverage matrix in test functions below.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.nscs06_v8_chroma_lut_pr95_parity import (
    CH09_HEADER_SIZE,
    CH09_MAGIC,
    CH09_SCHEMA_VERSION_PR95_PARITY,
    CHROMA_LUT_BYTES_DEFAULT,
    DECODER_BLOB_LEN,
    GRAYSCALE_LEVELS_DEFAULT,
    L4_LOC_WAIVER,
    LATENT_BLOB_LEN,
    NUM_SEGNET_CLASSES,
    Nscs06V8Pr95ParityArchive,
    Nscs06V8Pr95ParityConfig,
    POSE_DIMS,
    POSE_QUANT_SCALE_DEFAULT,
    PROCEDURAL_SEED_SIZE_BYTES,
    SCORE_AWARE_LAGRANGIAN_ALPHA,
    SCORE_AWARE_LAGRANGIAN_BETA,
    SCORE_AWARE_LAGRANGIAN_GAMMA,
    affine_warp_frame1_from_frame0,
    build_chroma_lut_from_ground_truth,
    derive_chroma_lut_bytes_from_seed,
    inflate_one_video,
    lookup_rgb_via_chroma_lut,
    main_cli,
    pack_archive,
    parse_archive,
    score_aware_lagrangian_loss,
    select_inflate_device,
)


# Fixtures
def _make_synthetic_archive(N: int = 4, H_low: int = 8, W_low: int = 8,
                            H_out: int = 24, W_out: int = 32, seed: int = 42) -> bytes:
    """Build a deterministic synthetic CH09 archive for testing."""
    rng = np.random.RandomState(seed)
    chroma_seed = bytes(range(32))
    pose_u8 = rng.randint(0, 256, size=(N, POSE_DIMS), dtype=np.uint8)
    gray_lowres = rng.randint(0, 256, size=(N, H_low, W_low), dtype=np.uint8)
    cls_lowres = rng.randint(0, 5, size=(N, H_low, W_low), dtype=np.uint8)
    return pack_archive(
        chroma_seed=chroma_seed,
        pose_quantized_u8=pose_u8,
        grayscale_lowres_u8=gray_lowres,
        cls_lowres_u8=cls_lowres,
        output_height=H_out,
        output_width=W_out,
    )


# L2 export-first design
def test_l2_canonical_constants_declared() -> None:
    """L2: archive grammar fixed offsets declared in source."""
    assert CH09_MAGIC == b"CH09"
    assert CH09_SCHEMA_VERSION_PR95_PARITY == 2
    assert CH09_HEADER_SIZE == 37
    assert DECODER_BLOB_LEN == PROCEDURAL_SEED_SIZE_BYTES == 32
    assert LATENT_BLOB_LEN == 0  # No learned latents (closed-form LUT)


def test_l2_config_canonical() -> None:
    cfg = Nscs06V8Pr95ParityConfig()
    assert cfg.grayscale_levels == 16
    assert cfg.num_segnet_classes == 5
    assert cfg.chroma_lut_bytes == 4096
    assert cfg.seed_size_bytes == 32
    assert cfg.chroma_lut_shape == (16, 5, 3)


def test_l2_config_invariants() -> None:
    with pytest.raises(ValueError, match="grayscale_levels"):
        Nscs06V8Pr95ParityConfig(grayscale_levels=0)
    with pytest.raises(ValueError, match="seed_size_bytes"):
        Nscs06V8Pr95ParityConfig(seed_size_bytes=16)
    with pytest.raises(ValueError, match="generator_kind"):
        Nscs06V8Pr95ParityConfig(generator_kind="xoshiro256")
    with pytest.raises(ValueError, match="chroma_lut_bytes.*minimum required"):
        Nscs06V8Pr95ParityConfig(chroma_lut_bytes=10)


# L3 monolithic single-file 0.bin
def test_l3_pack_parse_roundtrip() -> None:
    archive_bytes = _make_synthetic_archive()
    parsed = parse_archive(archive_bytes)
    assert parsed.schema_version == CH09_SCHEMA_VERSION_PR95_PARITY
    assert parsed.num_pairs == 4
    assert parsed.grayscale_h == 8
    assert parsed.grayscale_w == 8
    assert parsed.chroma_seed == bytes(range(32))


def test_l3_pack_deterministic_byte_stable() -> None:
    """L3: byte-stable archive (deterministic struct.pack)."""
    archive_a = _make_synthetic_archive(seed=42)
    archive_b = _make_synthetic_archive(seed=42)
    assert archive_a == archive_b, "Archive NOT byte-deterministic"


def test_l3_parse_rejects_bad_magic() -> None:
    archive_bytes = _make_synthetic_archive()
    mutated = bytearray(archive_bytes)
    mutated[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(bytes(mutated))


# L4 inflate <= 200 LOC (waived from 100) + <= 2 ext deps + CUDA-or-CPU agnostic
def test_l4_loc_waiver_declared() -> None:
    assert "L4" in L4_LOC_WAIVER
    assert "waiver" in L4_LOC_WAIVER.lower()
    assert "numpy" in L4_LOC_WAIVER.lower() and "pillow" in L4_LOC_WAIVER.lower()


def test_l4_select_inflate_device_canonical() -> None:
    """L4 + Catalog #205: canonical select_inflate_device honors PACT_INFLATE_DEVICE."""
    import os
    # Default = auto
    os.environ.pop("PACT_INFLATE_DEVICE", None)
    assert select_inflate_device() == "auto"
    # Valid values
    os.environ["PACT_INFLATE_DEVICE"] = "cpu"
    assert select_inflate_device() == "cpu"
    os.environ["PACT_INFLATE_DEVICE"] = "cuda"
    assert select_inflate_device() == "cuda"
    # mps REFUSED per CLAUDE.md
    os.environ["PACT_INFLATE_DEVICE"] = "mps"
    with pytest.raises(RuntimeError, match="MPS auth eval is NOISE"):
        select_inflate_device()
    # Invalid REFUSED
    os.environ["PACT_INFLATE_DEVICE"] = "xpu"
    with pytest.raises(RuntimeError, match="not in"):
        select_inflate_device()
    # cleanup
    os.environ.pop("PACT_INFLATE_DEVICE", None)


def test_l4_ext_deps_only_numpy_pillow() -> None:
    """L4: inflate runtime closure = numpy + Pillow only (NO torch, NO scorer)."""
    # Read inflate.py source + assert no forbidden imports
    from tac.substrates.nscs06_v8_chroma_lut_pr95_parity import inflate as m
    src = Path(m.__file__).read_text()
    forbidden = ["import torch", "from torch", "import smp", "from torch.nn",
                 "EfficientNet", "FastViT", "from upstream.modules", "import upstream.modules"]
    for tok in forbidden:
        assert tok not in src, f"FORBIDDEN import {tok!r} found in inflate.py (L4 + strict-scorer-rule)"


# L5 FULL RGB renderer
def test_l5_inflate_produces_rgb_frames() -> None:
    N, H_out, W_out = 4, 24, 32
    archive_bytes = _make_synthetic_archive(N=N, H_out=H_out, W_out=W_out)
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = inflate_one_video(archive_bytes, Path(tmpdir) / "test")
        raw_bytes = raw_path.read_bytes()
        # 2 frames per pair x RGB
        expected = N * 2 * H_out * W_out * 3
        assert len(raw_bytes) == expected, f"WRONG-SIZE: {len(raw_bytes)} != {expected}"


def test_l5_lookup_rgb_returns_full_rgb() -> None:
    chroma_lut = np.random.RandomState(0).randint(0, 256, size=(16, 5, 3), dtype=np.uint8)
    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    cls = np.array([[0, 2, 4]], dtype=np.uint8)
    out = lookup_rgb_via_chroma_lut(gray, cls, chroma_lut)
    assert out.shape == (1, 3, 3)  # (H, W, 3) per pixel RGB
    assert out.dtype == np.uint8


# L6 score-domain Lagrangian
def test_l6_lagrangian_canonical_constants() -> None:
    assert SCORE_AWARE_LAGRANGIAN_ALPHA == 25.0
    assert SCORE_AWARE_LAGRANGIAN_BETA == 100.0
    assert SCORE_AWARE_LAGRANGIAN_GAMMA == 1.0


def test_l6_lagrangian_canonical_formula_numpy() -> None:
    """L6: score = 25*archive_bytes/37545489 + 100*d_seg + sqrt(10*d_pose)."""
    components = {"d_seg": 0.002, "d_pose": 0.0005}
    archive_bytes = 178517
    loss = score_aware_lagrangian_loss(components=components, archive_bytes=archive_bytes)
    expected = 25.0 * archive_bytes / 37_545_489 + 100.0 * 0.002 + (10.0 * 0.0005) ** 0.5
    assert abs(loss - expected) < 1e-9


def test_l6_lagrangian_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="d_seg.*d_pose"):
        score_aware_lagrangian_loss(components={}, archive_bytes=1000)


# L9 runtime closure (numpy + Pillow only; canonical select_inflate_device)
def test_l9_runtime_closure_clean_env() -> None:
    """L9: inflate runs end-to-end with numpy + Pillow + canonical helper."""
    archive_bytes = _make_synthetic_archive()
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = inflate_one_video(archive_bytes, Path(tmpdir) / "video_0")
        assert raw_path.exists()
        assert raw_path.suffix == ".raw"


# L10 mask/pose coupling
def test_l10_affine_warp_consumes_pose_deltas() -> None:
    """L10: pose deltas drive frame_1 affine warp from frame_0."""
    frame_0 = np.random.RandomState(0).randint(0, 256, size=(24, 32, 3), dtype=np.uint8)
    pose_zero = np.zeros(POSE_DIMS, dtype=np.float32)
    pose_nonzero = np.array([1.0, 0.5, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    f1_zero = affine_warp_frame1_from_frame0(frame_0, pose_zero)
    f1_nonzero = affine_warp_frame1_from_frame0(frame_0, pose_nonzero)
    assert f1_zero.shape == frame_0.shape
    assert f1_nonzero.shape == frame_0.shape
    # Zero pose -> approximately identity (within bilinear resampling)
    # Non-zero pose -> measurably different frame
    n_diff = int(np.sum(f1_zero != f1_nonzero))
    assert n_diff > 0, "pose deltas had no effect on frame_1"


def test_l10_affine_warp_validates_inputs() -> None:
    with pytest.raises(ValueError, match="frame_0 must be"):
        affine_warp_frame1_from_frame0(np.zeros((10, 10), dtype=np.uint8), np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="pose must be"):
        affine_warp_frame1_from_frame0(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros(5, dtype=np.float32))


# L11 no-op detector via byte-mutation smoke
def test_l11_byte_mutation_in_chroma_seed_changes_output() -> None:
    """L11: chroma_seed byte mutation MUST change output frames (no-op detector PASS)."""
    archive_bytes = _make_synthetic_archive(N=2)
    mutated = bytearray(archive_bytes)
    # Mutate first chroma_seed byte (offset = CH09_HEADER_SIZE = 37)
    mutated[CH09_HEADER_SIZE] = (mutated[CH09_HEADER_SIZE] + 13) & 0xFF
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_a = inflate_one_video(archive_bytes, Path(tmpdir) / "a").read_bytes()
        raw_b = inflate_one_video(bytes(mutated), Path(tmpdir) / "b").read_bytes()
        n_diff = sum(1 for x, y in zip(raw_a, raw_b) if x != y)
        assert n_diff > 0, "No-op detector FAIL: chroma_seed bytes consumed but no frame change"


def test_l11_byte_mutation_in_pose_changes_output() -> None:
    """L11: pose byte mutation MUST change frame_1 (no-op detector PASS)."""
    archive_bytes = _make_synthetic_archive(N=2)
    parsed = parse_archive(archive_bytes)
    pose_offset = CH09_HEADER_SIZE + PROCEDURAL_SEED_SIZE_BYTES
    mutated = bytearray(archive_bytes)
    mutated[pose_offset] = (mutated[pose_offset] + 50) & 0xFF
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_a = inflate_one_video(archive_bytes, Path(tmpdir) / "a").read_bytes()
        raw_b = inflate_one_video(bytes(mutated), Path(tmpdir) / "b").read_bytes()
        n_diff = sum(1 for x, y in zip(raw_a, raw_b) if x != y)
        assert n_diff > 0, "No-op detector FAIL: pose bytes consumed but no frame change"


def test_l11_byte_mutation_in_cls_stream_changes_output() -> None:
    """L11: cls_stream byte mutation MUST change frame (Wave N+22 wire-in canonical).

    Mutate ENTIRE cls_stream from one class to a different class to guarantee
    chroma_lut[level, c_new] differs from chroma_lut[level, c_old] for some pixel.
    """
    archive_bytes = _make_synthetic_archive(N=2)
    parsed = parse_archive(archive_bytes)
    cls_offset = (CH09_HEADER_SIZE + PROCEDURAL_SEED_SIZE_BYTES
                  + len(parsed.pose_bytes) + len(parsed.grayscale_bytes))
    cls_len = len(parsed.cls_bytes)
    # Force ALL cls bytes to class 0 in archive_a, class 4 in archive_b
    a_buf = bytearray(archive_bytes)
    b_buf = bytearray(archive_bytes)
    for i in range(cls_offset, cls_offset + cls_len):
        a_buf[i] = 0
        b_buf[i] = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_a = inflate_one_video(bytes(a_buf), Path(tmpdir) / "a").read_bytes()
        raw_b = inflate_one_video(bytes(b_buf), Path(tmpdir) / "b").read_bytes()
        n_diff = sum(1 for x, y in zip(raw_a, raw_b) if x != y)
        assert n_diff > 0, "No-op detector FAIL: cls_stream bytes consumed but no frame change"


# Canonical PROCEDURAL VARIANT pattern (Catalog #26 equation IN-DOMAIN context)
def test_derive_chroma_lut_from_seed_deterministic() -> None:
    seed = bytes(range(32))
    lut_a = derive_chroma_lut_bytes_from_seed(seed)
    lut_b = derive_chroma_lut_bytes_from_seed(seed)
    assert np.array_equal(lut_a, lut_b), "LUT derivation NOT deterministic"
    assert lut_a.shape == (16, 5, 3)
    assert lut_a.dtype == np.uint8


def test_derive_chroma_lut_seed_validates_length() -> None:
    with pytest.raises(ValueError, match="seed_bytes len"):
        derive_chroma_lut_bytes_from_seed(b"too_short")


# Compress-side LUT derivation (HARD-EARNED case)
def test_build_chroma_lut_from_ground_truth() -> None:
    rng = np.random.RandomState(0)
    rgb = rng.randint(0, 256, size=(2, 3, 8, 8), dtype=np.uint8)
    cls = rng.randint(0, 5, size=(2, 8, 8), dtype=np.uint8)
    lut = build_chroma_lut_from_ground_truth(rgb, cls)
    assert lut.shape == (16, 5, 3)
    assert lut.dtype == np.uint8


# CLI smoke
def test_main_cli_smoke() -> None:
    """CLI: inflate.py <archive_dir> <output_dir> <file_list> per Catalog #146."""
    import sys
    archive_bytes = _make_synthetic_archive(N=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_dir = Path(tmpdir) / "archive"
        archive_dir.mkdir()
        (archive_dir / "0.bin").write_bytes(archive_bytes)
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        file_list = Path(tmpdir) / "files.txt"
        file_list.write_text("0.mkv\n", encoding="utf-8")
        old_argv = sys.argv[:]
        try:
            sys.argv = ["inflate.py", str(archive_dir), str(output_dir), str(file_list)]
            rc = main_cli()
        finally:
            sys.argv = old_argv
        assert rc == 0
        assert (output_dir / "0.raw").exists()


def test_main_cli_usage_error() -> None:
    import sys
    old_argv = sys.argv[:]
    try:
        sys.argv = ["inflate.py"]
        rc = main_cli()
    finally:
        sys.argv = old_argv
    assert rc == 2
