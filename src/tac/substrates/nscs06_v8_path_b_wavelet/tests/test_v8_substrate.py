# SPDX-License-Identifier: MIT
"""Tests for NSCS06 v8 Path B wavelet residual substrate.

Coverage:
  1. DB4 depth-2 DWT/IDWT roundtrip lossless (≤1e-9 max abs err)
  2. DB4 filter coefficient lengths (Daubechies 1992 invariant)
  3. NUM_SUBBANDS == 7 invariant
  4. Per-subband quant + dequant roundtrip exact for integer-aligned coeffs
  5. Per-class Laplacian CDF: monotone-non-decreasing + CDF[0]=0 + CDF[-1]=CDF_MAX
  6. encode_subband_arith / decode_subband_arith byte-stable roundtrip
  7. Wyner-Ziv residual: f1 - f0 + f0 -> f1 (symbol-space exact, modulo clamp)
  8. WLV2 archive: pack -> parse roundtrip preserves all fields
  9. WLV2 header size invariant: == 61 bytes (Catalog #124 byte-level invariant)
 10. Empty-stream inflate degenerate path (smoke gate): zero subbands -> raw written
 11. Byte-mutation Catalog #220 smoke: mutating wavelet stream bytes DOES change output
 12. parse_archive rejects bad magic
 13. parse_archive rejects wrong version
 14. parse_archive rejects trailing bytes
 15. quant_steps tuple length invariant
 16. priors shape invariant (NUM_SUBBANDS, NUM_SEGNET_CLASSES)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.nscs06_v8_path_b_wavelet import (
    NUM_SUBBANDS,
    PER_SUBBAND_QUANT_STEPS,
    SUBBAND_LABELS,
    WLV2_HEADER_FMT,
    WLV2_HEADER_SIZE,
    WLV2_MAGIC,
    WLV2_SCHEMA_VERSION,
    WaveletResidualArchive,
    compute_wyner_ziv_residual,
    dequantize_subband,
    dwt2_db4_depth2,
    idwt2_db4_depth2,
    inflate_one_video,
    laplacian_cdf_uint16,
    pack_archive,
    parse_archive,
    quantize_subband,
    reconstruct_frame1_from_frame0_and_residual,
)
from tac.substrates.nscs06_v8_path_b_wavelet.wavelet_codec import (
    DB4_DECOMP_HI,
    DB4_DECOMP_LO,
    DB4_RECON_HI,
    DB4_RECON_LO,
    DWT_LEVEL,
    PerSubbandLaplacianPriors,
    QUANT_CLAMP_T,
    QUANT_LEVELS,
    decode_subband_arith,
    encode_subband_arith,
)


def _make_priors(scale: float = 4.0) -> PerSubbandLaplacianPriors:
    """Helper: uniform-scale priors for test fixtures."""
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
        NUM_SEGNET_CLASSES,
    )

    scales = np.full((NUM_SUBBANDS, NUM_SEGNET_CLASSES), scale, dtype=np.float32)
    return PerSubbandLaplacianPriors(scales=scales)


# ---------------------------------------------------------------------------
# 1. DB4 depth-2 DWT/IDWT roundtrip lossless
# ---------------------------------------------------------------------------
def test_dwt_idwt_roundtrip_lossless():
    rng = np.random.default_rng(20260516)
    img = rng.uniform(0, 255, size=(96, 128)).astype(np.float64)
    subbands = dwt2_db4_depth2(img)
    rec = idwt2_db4_depth2(subbands, (96, 128))
    err = float(np.max(np.abs(img - rec)))
    assert err < 1e-9, f"DWT roundtrip not lossless: max abs err {err}"


# ---------------------------------------------------------------------------
# 2. DB4 filter coefficient invariants (Daubechies 1992 Table 6.1)
# ---------------------------------------------------------------------------
def test_db4_filter_lengths():
    assert len(DB4_DECOMP_LO) == 8, "DB4 decomp_lo must be length 8"
    assert len(DB4_DECOMP_HI) == 8, "DB4 decomp_hi must be length 8"
    assert len(DB4_RECON_LO) == 8, "DB4 recon_lo must be length 8"
    assert len(DB4_RECON_HI) == 8, "DB4 recon_hi must be length 8"


def test_dwt_level_invariant():
    assert DWT_LEVEL == 2


# ---------------------------------------------------------------------------
# 3. NUM_SUBBANDS == 7
# ---------------------------------------------------------------------------
def test_num_subbands_invariant():
    assert NUM_SUBBANDS == 7
    assert len(SUBBAND_LABELS) == 7
    assert SUBBAND_LABELS == ("LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1")
    assert len(PER_SUBBAND_QUANT_STEPS) == 7


# ---------------------------------------------------------------------------
# 4. Per-subband quant + dequant roundtrip
# ---------------------------------------------------------------------------
def test_quantize_dequantize_roundtrip():
    # Build coefficients that are integer-multiples of step so roundtrip is exact
    step = 4
    coeff = np.array([[0, 4, 8, -4, -8, 16, -16]], dtype=np.float64)
    q = quantize_subband(coeff, step)
    deq = dequantize_subband(q, step)
    assert np.allclose(coeff, deq), f"quantize roundtrip mismatch: {coeff} vs {deq}"


def test_quantize_clamps_to_T():
    # Values beyond ±T*step should clip
    step = 1
    coeff = np.array([[100.0, -100.0]], dtype=np.float64)
    q = quantize_subband(coeff, step)
    assert int(q.max()) == QUANT_CLAMP_T
    assert int(q.min()) == -QUANT_CLAMP_T


def test_quantize_rejects_negative_step():
    with pytest.raises(ValueError, match="step must be > 0"):
        quantize_subband(np.zeros((4, 4)), 0)


# ---------------------------------------------------------------------------
# 5. Laplacian CDF validity invariants
# ---------------------------------------------------------------------------
def test_laplacian_cdf_monotone_and_endpoints():
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import CDF_MAX

    for scale in [0.5, 1.0, 4.0, 20.0]:
        cdf = laplacian_cdf_uint16(scale)
        assert cdf.dtype == np.uint16
        assert len(cdf) == QUANT_LEVELS + 1
        assert int(cdf[0]) == 0, f"CDF[0] must be 0 for scale={scale}"
        assert int(cdf[-1]) == CDF_MAX, f"CDF[-1] must be CDF_MAX for scale={scale}"
        diffs = np.diff(cdf.astype(np.int32))
        assert (diffs >= 0).all(), f"CDF must be non-decreasing for scale={scale}"
        # Strict monotone except possibly at the very end clamp
        assert (diffs[:-1] > 0).all() or (diffs > 0).sum() >= QUANT_LEVELS - 1


def test_laplacian_cdf_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="scale must be > 0"):
        laplacian_cdf_uint16(0.0)
    with pytest.raises(ValueError, match="scale must be > 0"):
        laplacian_cdf_uint16(-1.0)


# ---------------------------------------------------------------------------
# 6. encode/decode subband arith roundtrip
# ---------------------------------------------------------------------------
def test_encode_decode_subband_arith_roundtrip():
    rng = np.random.default_rng(20260516)
    # Synthetic quantized subband
    q = rng.integers(-5, 5, size=(8, 12), dtype=np.int32).astype(np.int8)
    cls = rng.integers(0, 5, size=(8, 12), dtype=np.uint8)
    priors = _make_priors(scale=4.0)
    encoded = encode_subband_arith(q, cls, priors, subband_index=0)
    assert len(encoded) > 0
    decoded = decode_subband_arith(encoded, cls, priors, subband_index=0)
    assert np.array_equal(q, decoded), "arith encode/decode must roundtrip exact"


# ---------------------------------------------------------------------------
# 7. Wyner-Ziv residual + reconstruct
# ---------------------------------------------------------------------------
def test_wyner_ziv_reconstruct_in_symbol_space():
    # Build two sets of quantized subbands; residual + f0 -> f1
    rng = np.random.default_rng(20260516)
    f0_q = [rng.integers(-5, 5, size=(4, 4), dtype=np.int8) for _ in range(NUM_SUBBANDS)]
    f1_q = [rng.integers(-5, 5, size=(4, 4), dtype=np.int8) for _ in range(NUM_SUBBANDS)]
    # Residual in symbol space
    residual_q = [
        (f1_q[s].astype(np.int32) - f0_q[s].astype(np.int32)).astype(np.int8)
        for s in range(NUM_SUBBANDS)
    ]
    rec_f1 = reconstruct_frame1_from_frame0_and_residual(f0_q, residual_q)
    for s in range(NUM_SUBBANDS):
        # In this test the residual values are bounded; reconstruction is exact
        assert np.array_equal(rec_f1[s], f1_q[s]), (
            f"Wyner-Ziv reconstruct mismatch at subband {s}"
        )


def test_compute_wyner_ziv_residual_subtraction():
    # In float space, residual = f1 - f0
    f0 = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])] + [np.zeros((2, 2)) for _ in range(5)]
    f1 = [np.array([[5.0, 6.0]]), np.array([[7.0, 8.0]])] + [np.zeros((2, 2)) for _ in range(5)]
    residual = compute_wyner_ziv_residual(f0, f1)
    assert np.allclose(residual[0], [[4.0, 4.0]])
    assert np.allclose(residual[1], [[4.0, 4.0]])


# ---------------------------------------------------------------------------
# 8. WLV2 archive pack/parse roundtrip
# ---------------------------------------------------------------------------
def test_pack_parse_archive_roundtrip():
    priors = _make_priors(scale=4.0)
    n_pairs = 4
    per_pair_offsets = np.zeros((n_pairs, 7), dtype=np.uint32)
    meta = {"grayscale_downsample": 4, "test_marker": "v8_smoke"}
    archive_bytes = pack_archive(
        priors=priors,
        per_pair_offsets=per_pair_offsets,
        gray_f0_bytes=b"\x00\x01\x02",
        gray_f1res_bytes=b"\x03\x04",
        cb_f0_bytes=b"\x05",
        cb_f1res_bytes=b"\x06\x07",
        cr_f0_bytes=b"",
        cr_f1res_bytes=b"\x08",
        cls_bytes=b"\x09\x0A",
        meta=meta,
        quant_steps=PER_SUBBAND_QUANT_STEPS,
        num_pairs=n_pairs,
        eval_height=96,
        eval_width=128,
        output_height=874,
        output_width=1164,
    )
    arc = parse_archive(archive_bytes)
    assert arc.schema_version == WLV2_SCHEMA_VERSION
    assert arc.num_pairs == n_pairs
    assert arc.eval_height == 96
    assert arc.eval_width == 128
    assert arc.output_height == 874
    assert arc.output_width == 1164
    assert arc.dwt_level == 2
    assert arc.num_subbands == NUM_SUBBANDS
    assert arc.gray_f0_bytes == b"\x00\x01\x02"
    assert arc.gray_f1res_bytes == b"\x03\x04"
    assert arc.cb_f0_bytes == b"\x05"
    assert arc.cls_bytes == b"\x09\x0A"
    assert arc.meta["test_marker"] == "v8_smoke"
    assert arc.quant_steps == PER_SUBBAND_QUANT_STEPS
    assert np.allclose(arc.priors.scales, 4.0)


# ---------------------------------------------------------------------------
# 9. WLV2 header size invariant
# ---------------------------------------------------------------------------
def test_wlv2_header_size_invariant():
    assert WLV2_HEADER_SIZE == 61, f"WLV2 header size must be 61; got {WLV2_HEADER_SIZE}"
    # Sanity: WLV2_HEADER_FMT parses to WLV2_HEADER_SIZE
    assert struct.calcsize(WLV2_HEADER_FMT) == WLV2_HEADER_SIZE


# ---------------------------------------------------------------------------
# 10. Empty-stream inflate (smoke gate degenerate path)
# ---------------------------------------------------------------------------
def test_inflate_empty_streams_writes_raw(tmp_path: Path):
    """v1 SCAFFOLD: empty streams produce zero-subband decode + valid raw bytes."""
    priors = _make_priors(scale=4.0)
    n_pairs = 2
    per_pair_offsets = np.zeros((n_pairs, 7), dtype=np.uint32)
    bin_bytes = pack_archive(
        priors=priors,
        per_pair_offsets=per_pair_offsets,
        gray_f0_bytes=b"",
        gray_f1res_bytes=b"",
        cb_f0_bytes=b"",
        cb_f1res_bytes=b"",
        cr_f0_bytes=b"",
        cr_f1res_bytes=b"",
        cls_bytes=b"",
        meta={},
        quant_steps=PER_SUBBAND_QUANT_STEPS,
        num_pairs=n_pairs,
        eval_height=24,
        eval_width=32,
        output_height=48,
        output_width=64,
    )
    out_stem = tmp_path / "smoke" / "0"
    raw_path = inflate_one_video(bin_bytes, out_stem)
    assert raw_path.is_file()
    expected_bytes = 48 * 64 * 3 * 2 * n_pairs  # H*W*RGB*2_frames*N_pairs
    actual_bytes = raw_path.stat().st_size
    assert actual_bytes == expected_bytes, (
        f"raw bytes {actual_bytes} != expected {expected_bytes}"
    )


# ---------------------------------------------------------------------------
# 11. Catalog #220 byte-mutation smoke: distinguishing-feature contract
# ---------------------------------------------------------------------------
def test_byte_mutation_changes_output(tmp_path: Path):
    """Mutating a wavelet stream byte MUST change rendered frame pixels.

    This is the Catalog #272 distinguishing-feature contract surface: the
    wavelet streams ARE the distinguishing bytes. A no-op detector test
    (Catalog #220) confirms compress-side bytes are operationally consumed.
    """
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
        NUM_SEGNET_CLASSES,
    )

    rng = np.random.default_rng(20260516)
    eval_h, eval_w = 24, 32
    n_pairs = 1
    # Build a real (non-empty) wavelet stream for one channel by quantizing a
    # synthetic subband signal then arith-encoding it. The class labels for
    # arith decode must match what the inflate runtime sees per subband shape.

    # Generate class labels per the depth-2 periodization shape rule
    from tac.substrates.nscs06_v8_path_b_wavelet.inflate import (
        _depth2_subband_shapes,
    )

    expected_shapes = _depth2_subband_shapes(eval_h, eval_w)
    # Build the full per-channel stream: concatenate arith across all 7 subbands
    # using uniform class=0 + Laplacian priors.
    priors = _make_priors(scale=4.0)
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
        ArithmeticCoder,
    )

    # Build cls subband stream FIRST (uniform CDF) — inflate decodes this first
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import CDF_MAX
    uniform_cdf = np.linspace(0, CDF_MAX, NUM_SEGNET_CLASSES + 1, dtype=np.int64)
    uniform_cdf[-1] = CDF_MAX
    uniform_cdf = uniform_cdf.astype(np.uint16)
    cls_coder = ArithmeticCoder()
    cls_subband_arrays: list[np.ndarray] = []
    for s in range(NUM_SUBBANDS):
        sb_h, sb_w = expected_shapes[s]
        cls_sb = np.zeros((sb_h, sb_w), dtype=np.uint8)  # all class 0
        cls_subband_arrays.append(cls_sb)
        for c in cls_sb.ravel():
            cls_coder.encode_symbol(int(c), uniform_cdf)
    cls_bytes = cls_coder.finish_encoding()

    # Build gray_f0 channel stream
    from tac.substrates.nscs06_v8_path_b_wavelet.wavelet_codec import (
        laplacian_cdf_uint16,
        QUANT_ZERO_INDEX,
    )

    gray_coder = ArithmeticCoder()
    for s in range(NUM_SUBBANDS):
        sb_h, sb_w = expected_shapes[s]
        # Use a non-zero distinct value at known position so mutation can be observed
        q_sb = np.full((sb_h, sb_w), 1, dtype=np.int8)  # all symbol = +1
        cdf_class0 = laplacian_cdf_uint16(float(priors.scales[s, 0]))
        for q in q_sb.ravel():
            gray_coder.encode_symbol(int(q) + QUANT_ZERO_INDEX, cdf_class0)
    gray_f0_bytes = gray_coder.finish_encoding()

    per_pair_offsets = np.zeros((n_pairs, 7), dtype=np.uint32)
    bin_bytes = pack_archive(
        priors=priors,
        per_pair_offsets=per_pair_offsets,
        gray_f0_bytes=gray_f0_bytes,
        gray_f1res_bytes=b"",
        cb_f0_bytes=b"",
        cb_f1res_bytes=b"",
        cr_f0_bytes=b"",
        cr_f1res_bytes=b"",
        cls_bytes=cls_bytes,
        meta={},
        quant_steps=PER_SUBBAND_QUANT_STEPS,
        num_pairs=n_pairs,
        eval_height=eval_h,
        eval_width=eval_w,
        output_height=48,
        output_width=64,
    )

    # Render original
    raw_orig = inflate_one_video(bin_bytes, tmp_path / "orig" / "0").read_bytes()
    # Mutate first byte of gray_f0_bytes (find its offset in the archive)
    # Layout: HEADER + LAPLACIAN_BLOB + OFFSETS_BLOB + GRAY_F0_BLOB + ...
    laplacian_len = NUM_SUBBANDS * NUM_SEGNET_CLASSES * 4
    offsets_len = n_pairs * 7 * 4
    gray_f0_start = WLV2_HEADER_SIZE + laplacian_len + offsets_len
    # Flip the first byte of the gray_f0 stream
    mutated = bytearray(bin_bytes)
    mutated[gray_f0_start] ^= 0xFF
    raw_mut = inflate_one_video(bytes(mutated), tmp_path / "mut" / "0").read_bytes()

    assert raw_orig != raw_mut, (
        "Catalog #220 byte-mutation FAIL: mutating gray_f0 byte did NOT change output"
    )


# ---------------------------------------------------------------------------
# 12. parse_archive rejects bad magic
# ---------------------------------------------------------------------------
def test_parse_rejects_bad_magic():
    fake = b"\x00" * WLV2_HEADER_SIZE
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(fake)


# ---------------------------------------------------------------------------
# 13. parse_archive rejects wrong version
# ---------------------------------------------------------------------------
def test_parse_rejects_wrong_version():
    """Hand-craft a header with version=99 and verify rejection."""
    priors = _make_priors(scale=4.0)
    per_pair_offsets = np.zeros((1, 7), dtype=np.uint32)
    archive_bytes = pack_archive(
        priors=priors,
        per_pair_offsets=per_pair_offsets,
        gray_f0_bytes=b"",
        gray_f1res_bytes=b"",
        cb_f0_bytes=b"",
        cb_f1res_bytes=b"",
        cr_f0_bytes=b"",
        cr_f1res_bytes=b"",
        cls_bytes=b"",
        meta={},
        quant_steps=PER_SUBBAND_QUANT_STEPS,
        num_pairs=1,
        eval_height=4,
        eval_width=4,
        output_height=4,
        output_width=4,
    )
    # Mutate version byte (offset 4 in WLV2 header)
    bad = bytearray(archive_bytes)
    bad[4] = 99
    with pytest.raises(ValueError, match="unsupported WLV2 version"):
        parse_archive(bytes(bad))


# ---------------------------------------------------------------------------
# 14. parse_archive rejects trailing bytes
# ---------------------------------------------------------------------------
def test_parse_rejects_trailing_bytes():
    priors = _make_priors(scale=4.0)
    per_pair_offsets = np.zeros((1, 7), dtype=np.uint32)
    archive_bytes = pack_archive(
        priors=priors,
        per_pair_offsets=per_pair_offsets,
        gray_f0_bytes=b"",
        gray_f1res_bytes=b"",
        cb_f0_bytes=b"",
        cb_f1res_bytes=b"",
        cr_f0_bytes=b"",
        cr_f1res_bytes=b"",
        cls_bytes=b"",
        meta={},
        quant_steps=PER_SUBBAND_QUANT_STEPS,
        num_pairs=1,
        eval_height=4,
        eval_width=4,
        output_height=4,
        output_width=4,
    )
    with pytest.raises(ValueError, match="trailing bytes"):
        parse_archive(archive_bytes + b"\x00\x00")


# ---------------------------------------------------------------------------
# 15. quant_steps tuple length invariant
# ---------------------------------------------------------------------------
def test_pack_archive_rejects_bad_quant_steps():
    priors = _make_priors(scale=4.0)
    per_pair_offsets = np.zeros((1, 7), dtype=np.uint32)
    # Too short
    with pytest.raises(ValueError, match="quant_steps must have"):
        pack_archive(
            priors=priors,
            per_pair_offsets=per_pair_offsets,
            gray_f0_bytes=b"",
            gray_f1res_bytes=b"",
            cb_f0_bytes=b"",
            cb_f1res_bytes=b"",
            cr_f0_bytes=b"",
            cr_f1res_bytes=b"",
            cls_bytes=b"",
            meta={},
            quant_steps=(1, 4, 4),  # too few
            num_pairs=1,
            eval_height=4,
            eval_width=4,
            output_height=4,
            output_width=4,
        )
    # Non-positive
    with pytest.raises(ValueError, match="must all be > 0"):
        pack_archive(
            priors=priors,
            per_pair_offsets=per_pair_offsets,
            gray_f0_bytes=b"",
            gray_f1res_bytes=b"",
            cb_f0_bytes=b"",
            cb_f1res_bytes=b"",
            cr_f0_bytes=b"",
            cr_f1res_bytes=b"",
            cls_bytes=b"",
            meta={},
            quant_steps=(1, 4, 4, 8, 4, 4, 0),  # zero step
            num_pairs=1,
            eval_height=4,
            eval_width=4,
            output_height=4,
            output_width=4,
        )


# ---------------------------------------------------------------------------
# 16. priors shape invariant
# ---------------------------------------------------------------------------
def test_priors_shape_invariant():
    # Bad shape
    bad_scales = np.ones((NUM_SUBBANDS - 1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must be"):
        PerSubbandLaplacianPriors(scales=bad_scales)
    # Bad dtype
    bad_scales2 = np.ones((NUM_SUBBANDS, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="must be float32"):
        PerSubbandLaplacianPriors(scales=bad_scales2)
    # Non-positive
    bad_scales3 = np.zeros((NUM_SUBBANDS, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="all scales must be positive"):
        PerSubbandLaplacianPriors(scales=bad_scales3)
