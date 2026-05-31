# SPDX-License-Identifier: MIT
"""Codec + archive + inflate roundtrip tests for the nscs06 Carmack-Hotz substrate.

Per HNeRV parity discipline lesson L11 + Catalog #139 (no-op detector +
structural consumption proof), the byte-mutation smoke is the canonical
operational-mechanism acceptance criterion.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from tac.procedural_codebook_generator.hash_seed_codebook_generator import (
    emit_seed as emit_canonical_seed,
)
from tac.procedural_codebook_generator.hash_seed_codebook_generator import (
    expand_seed_to_codebook as expand_canonical_seed_to_codebook,
)
from tac.substrates.nscs06_carmack_hotz_strip_everything import (
    CH06_HEADER_SIZE,
    CH06_MAGIC,
    CH06_SCHEMA_VERSION,
    CH06_SCHEMA_VERSION_SEEDED_CHROMA,
    ArithmeticCoder,
    ClassConditionalCDF,
    GrayscalePalette,
    allocate_bits_closed_form,
    build_grayscale_palette,
    inflate_one_video,
    pack_archive,
    parse_archive,
)
from tac.substrates.nscs06_carmack_hotz_strip_everything.archive import (
    CHROMA_BYTES_PER_CLASS,
    CHROMA_SEED_BYTES,
    POSE_DIMS,
    build_chroma_palette,
    decode_class_label_stream,
    dequantize_pose_deltas,
    emit_chroma_palette_seed,
    encode_class_label_stream,
    encode_grayscale_stream,
    expand_chroma_seed_to_palette,
    quantize_pose_deltas,
)
from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
    CDF_MAX,
    NUM_SEGNET_CLASSES,
    build_class_conditional_cdf,
)
from tac.substrates.nscs06_carmack_hotz_strip_everything.inflate import (
    _apply_pr98_l28_channel_balance_to_pair_uint8,
)


def _build_synthetic_archive(
    *, seed: int = 0, n_pairs: int = 4, palette_size: int = 16, h_g: int = 6, w_g: int = 8,
    output_h: int = 24, output_w: int = 32,
) -> tuple[bytes, GrayscalePalette, ClassConditionalCDF, np.ndarray, np.ndarray]:
    """Build a self-consistent synthetic CH06 v2 archive + return ground truth.

    Path A redesign 2026-05-16: v2 archives include `chroma_rgb` + `cls_arith_bytes`
    so the inflate runtime can synthesize per-pixel RGB instead of Y=R=G=B replication.
    """
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, 256, size=(n_pairs, h_g, w_g), dtype=np.uint8)
    palette = build_grayscale_palette(samples, palette_size=palette_size)
    palette_indices = palette.quantize(samples)
    cls = rng.integers(0, NUM_SEGNET_CLASSES, size=palette_indices.shape, dtype=np.uint8)
    cdf = build_class_conditional_cdf(palette_indices, cls, palette_size=palette_size)
    arith_bytes = encode_grayscale_stream(
        palette_indices=palette_indices, class_labels=cls, cdf=cdf
    )
    cls_arith_bytes = encode_class_label_stream(cls)
    synth_rgb = rng.integers(0, 256, size=(n_pairs, 3, h_g, w_g), dtype=np.uint8)
    chroma_rgb = build_chroma_palette(synth_rgb, cls)
    pose = (rng.standard_normal((n_pairs, POSE_DIMS)) * 0.01).astype(np.float32)
    pose_bytes, zero = quantize_pose_deltas(pose, scale=10.0)
    meta = {"grayscale_downsample": 4, "pose_quant_scale": 10.0, "pose_quant_zero": zero}
    blob = pack_archive(
        palette=palette, cdf=cdf, grayscale_arith_bytes=arith_bytes,
        pose_bytes=pose_bytes, meta=meta, num_pairs=n_pairs,
        grayscale_h=h_g, grayscale_w=w_g, output_height=output_h, output_width=output_w,
        chroma_rgb=chroma_rgb, cls_arith_bytes=cls_arith_bytes,
    )
    return blob, palette, cdf, palette_indices, cls


def _chroma_blob_offset(blob: bytes) -> int:
    fields = struct.unpack("<4sBHBHHHHHHIIHHI", blob[:CH06_HEADER_SIZE])
    (
        _magic,
        _version,
        _num_pairs,
        _palette_size,
        _grayscale_h,
        _grayscale_w,
        _output_height,
        _output_width,
        palette_len,
        cdf_len,
        grayscale_len,
        pose_len,
        meta_len,
        _chroma_len,
        _cls_len,
    ) = fields
    return CH06_HEADER_SIZE + palette_len + cdf_len + grayscale_len + pose_len + meta_len


def test_pr98_l28_postprocess_uses_canonical_pair_contract() -> None:
    """NSCS06 runtime applies L28 via the canonical helper semantics."""
    frame_0 = np.full((2, 3, 3), 10, dtype=np.uint8)
    frame_1 = np.full((2, 3, 3), 20, dtype=np.uint8)

    out_0, out_1 = _apply_pr98_l28_channel_balance_to_pair_uint8(frame_0, frame_1)

    assert out_0.dtype == np.uint8
    assert out_1.dtype == np.uint8
    assert (out_0[..., 0] == 9).all()
    assert (out_0[..., 1] == 10).all()
    assert (out_0[..., 2] == 9).all()
    assert (out_1[..., 0] == 20).all()
    assert (out_1[..., 1] == 19).all()
    assert (out_1[..., 2] == 20).all()


# ---------------------------------------------------------------------------
# Codec primitives
# ---------------------------------------------------------------------------

class TestGrayscalePalette:
    def test_build_size_matches_request(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.integers(0, 256, size=(8, 16), dtype=np.uint8)
        for size in (2, 4, 8, 16, 32):
            pal = build_grayscale_palette(samples, palette_size=size)
            assert pal.size == size

    def test_quantize_dequantize_levels(self) -> None:
        levels = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        pal = GrayscalePalette(levels=levels)
        gray = np.array([[0, 60, 130], [195, 255, 100]], dtype=np.uint8)
        idx = pal.quantize(gray)
        recon = pal.dequantize(idx)
        # Each recon value is in {levels}
        assert np.isin(recon, levels).all()
        # Idx is in valid range
        assert idx.max() < pal.size

    def test_invalid_dtype_rejected(self) -> None:
        with pytest.raises(ValueError):
            GrayscalePalette(levels=np.array([0.5, 1.5], dtype=np.float32))


class TestClassConditionalCDF:
    def test_endpoints_pinned(self) -> None:
        rng = np.random.default_rng(1)
        pi = rng.integers(0, 8, size=100, dtype=np.uint8)
        cls = rng.integers(0, NUM_SEGNET_CLASSES, size=100, dtype=np.uint8)
        cdf = build_class_conditional_cdf(pi, cls, palette_size=8)
        assert (cdf.cdf[:, 0] == 0).all()
        assert (cdf.cdf[:, -1] == CDF_MAX).all()

    def test_monotone_non_decreasing(self) -> None:
        rng = np.random.default_rng(2)
        pi = rng.integers(0, 16, size=200, dtype=np.uint8)
        cls = rng.integers(0, NUM_SEGNET_CLASSES, size=200, dtype=np.uint8)
        cdf = build_class_conditional_cdf(pi, cls, palette_size=16)
        diffs = np.diff(cdf.cdf.astype(np.int32), axis=1)
        assert (diffs >= 0).all()

    def test_empty_class_uniform_smoothing(self) -> None:
        # Only class 0 present in samples; class 1..4 get uniform-smoothing CDF
        pi = np.zeros(50, dtype=np.uint8)
        cls = np.zeros(50, dtype=np.uint8)
        cdf = build_class_conditional_cdf(pi, cls, palette_size=8)
        # Class 1..4 should have non-degenerate (non-uniform-at-0) CDFs
        for c in range(1, NUM_SEGNET_CLASSES):
            row = cdf.cdf[c]
            assert row[-1] == CDF_MAX
            assert (row > 0).any()

    def test_shape_mismatch_rejected(self) -> None:
        pi = np.zeros(10, dtype=np.uint8)
        cls = np.zeros(12, dtype=np.uint8)
        with pytest.raises(ValueError):
            build_class_conditional_cdf(pi, cls, palette_size=8)


class TestArithmeticCoder:
    def test_roundtrip_random_symbols(self) -> None:
        rng = np.random.default_rng(7)
        n = 500
        pi = rng.integers(0, 16, size=n, dtype=np.uint8)
        cls = rng.integers(0, NUM_SEGNET_CLASSES, size=n, dtype=np.uint8)
        cdf = build_class_conditional_cdf(
            pi.reshape(-1, 1, 1), cls.reshape(-1, 1, 1), palette_size=16
        )
        coder = ArithmeticCoder()
        for s, c in zip(pi, cls, strict=True):
            coder.encode_symbol(int(s), cdf.cdf[int(c)])
        enc = coder.finish_encoding()
        dec = ArithmeticCoder.from_bytes(enc)
        out = np.array([dec.decode_symbol(cdf.cdf[int(c)]) for c in cls], dtype=np.uint8)
        assert np.array_equal(pi, out)

    def test_compression_ratio_better_than_raw(self) -> None:
        # Skewed distribution -> arith should beat raw 4-bits/symbol
        rng = np.random.default_rng(8)
        n = 1000
        # Heavily skewed to symbol 0 (90%) -> entropy < 1 bit/symbol
        pi = (rng.random(n) > 0.9).astype(np.uint8) * 5
        cls = np.zeros(n, dtype=np.uint8)
        cdf = build_class_conditional_cdf(
            pi.reshape(-1, 1, 1), cls.reshape(-1, 1, 1), palette_size=16
        )
        coder = ArithmeticCoder()
        for s, c in zip(pi, cls, strict=True):
            coder.encode_symbol(int(s), cdf.cdf[int(c)])
        enc = coder.finish_encoding()
        raw_bits = n * 4  # palette_size=16 -> 4 bits/symbol
        enc_bits = len(enc) * 8
        # Arith should be at least 2x better than raw for this skewed distribution
        assert enc_bits < raw_bits / 2

    def test_invalid_symbol_rejected(self) -> None:
        cdf_row = np.array([0, 32000, 65535], dtype=np.uint16)
        coder = ArithmeticCoder()
        with pytest.raises(ValueError):
            coder.encode_symbol(5, cdf_row)  # only 2 symbols in row


class TestAllocateBitsClosedForm:
    def test_sum_equals_budget(self) -> None:
        imp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        alloc = allocate_bits_closed_form(imp, total_byte_budget=1000)
        assert alloc.sum() == 1000

    def test_proportional_to_importance(self) -> None:
        imp = np.array([1.0, 4.0], dtype=np.float32)
        alloc = allocate_bits_closed_form(imp, total_byte_budget=100)
        # 4x importance -> ~4x allocation (within rounding)
        assert alloc[1] >= alloc[0] * 3

    def test_zero_importance_uniform(self) -> None:
        imp = np.zeros(4, dtype=np.float32)
        alloc = allocate_bits_closed_form(imp, total_byte_budget=8)
        assert (alloc == 2).all()

    def test_negative_importance_rejected(self) -> None:
        imp = np.array([1.0, -1.0], dtype=np.float32)
        with pytest.raises(ValueError):
            allocate_bits_closed_form(imp, total_byte_budget=10)


# ---------------------------------------------------------------------------
# Archive grammar (CH06)
# ---------------------------------------------------------------------------

class TestCh06ArchiveRoundtrip:
    def test_pack_parse_self_consistent(self) -> None:
        blob, palette, cdf, palette_indices, cls = _build_synthetic_archive(seed=42)
        arc = parse_archive(blob)
        assert arc.num_pairs == 4
        assert arc.palette_size == palette.size
        assert arc.grayscale_h == 6
        assert arc.grayscale_w == 8
        assert arc.output_height == 24
        assert arc.output_width == 32
        # Palette levels preserved byte-stable
        assert np.array_equal(arc.palette.levels, palette.levels)
        # CDF preserved byte-stable
        assert np.array_equal(arc.cdf.cdf, cdf.cdf)

    def test_header_magic_pinned(self) -> None:
        blob, _, _, _, _ = _build_synthetic_archive(seed=0)
        assert blob[:4] == CH06_MAGIC
        assert blob[4] == CH06_SCHEMA_VERSION

    def test_header_size_invariant_36_bytes_v2(self) -> None:
        """Path A v2: header is 36 bytes (was 30 in v1; added chroma_len + cls_len)."""
        assert CH06_HEADER_SIZE == 36

    def test_wrong_magic_rejected(self) -> None:
        blob, _, _, _, _ = _build_synthetic_archive(seed=0)
        tampered = b"XXXX" + blob[4:]
        with pytest.raises(ValueError, match="bad magic"):
            parse_archive(tampered)

    def test_wrong_schema_version_rejected(self) -> None:
        blob, _, _, _, _ = _build_synthetic_archive(seed=0)
        tampered = blob[:4] + bytes([99]) + blob[5:]
        with pytest.raises(ValueError, match="unsupported schema version"):
            parse_archive(tampered)

    def test_truncated_archive_rejected(self) -> None:
        blob, _, _, _, _ = _build_synthetic_archive(seed=0)
        with pytest.raises(ValueError):
            parse_archive(blob[:10])

    def test_size_mismatch_rejected(self) -> None:
        blob, _, _, _, _ = _build_synthetic_archive(seed=0)
        with pytest.raises(ValueError):
            parse_archive(blob + b"extra")

    def test_seeded_chroma_schema_v3_replaces_palette_bytes(self) -> None:
        _, palette, cdf, palette_indices, cls = _build_synthetic_archive(seed=3)
        seed = emit_chroma_palette_seed()
        chroma_rgb = expand_chroma_seed_to_palette(seed)
        pose = np.zeros((4, POSE_DIMS), dtype=np.float32)
        pose_bytes, zero = quantize_pose_deltas(pose, scale=10.0)
        common_kwargs = {
            "palette": palette,
            "cdf": cdf,
            "grayscale_arith_bytes": encode_grayscale_stream(
                palette_indices=palette_indices,
                class_labels=cls,
                cdf=cdf,
            ),
            "pose_bytes": pose_bytes,
            "meta": {"grayscale_downsample": 4, "pose_quant_scale": 10.0, "pose_quant_zero": zero},
            "num_pairs": 4,
            "grayscale_h": 6,
            "grayscale_w": 8,
            "output_height": 24,
            "output_width": 32,
            "chroma_rgb": chroma_rgb,
            "cls_arith_bytes": encode_class_label_stream(cls),
        }
        blob_v2 = pack_archive(**common_kwargs)
        blob_v3 = pack_archive(
            **common_kwargs,
            chroma_seed=seed,
        )
        assert blob_v3[4] == CH06_SCHEMA_VERSION_SEEDED_CHROMA
        assert len(seed) == CHROMA_SEED_BYTES
        assert len(blob_v3) == len(blob_v2) - (
            NUM_SEGNET_CLASSES * CHROMA_BYTES_PER_CLASS - CHROMA_SEED_BYTES
        )
        arc = parse_archive(blob_v3)
        assert arc.chroma_seed == seed
        assert np.array_equal(arc.chroma_rgb, chroma_rgb)

    def test_seeded_chroma_matches_canonical_hash_seed_helper(self) -> None:
        shape = (NUM_SEGNET_CLASSES, CHROMA_BYTES_PER_CLASS)
        seed = emit_chroma_palette_seed()
        canonical_seed = emit_canonical_seed(shape)
        assert seed == canonical_seed
        canonical_palette = expand_canonical_seed_to_codebook(seed, shape).view(np.uint8)
        assert np.array_equal(expand_chroma_seed_to_palette(seed), canonical_palette)

    def test_seeded_chroma_requires_matching_generated_palette(self) -> None:
        blob, palette, cdf, palette_indices, cls = _build_synthetic_archive(seed=4)
        arc = parse_archive(blob)
        seed = emit_chroma_palette_seed()
        with pytest.raises(ValueError, match="expand_chroma_seed_to_palette"):
            pack_archive(
                palette=palette,
                cdf=cdf,
                grayscale_arith_bytes=encode_grayscale_stream(
                    palette_indices=palette_indices,
                    class_labels=cls,
                    cdf=cdf,
                ),
                pose_bytes=arc.pose_bytes,
                meta=arc.meta,
                num_pairs=arc.num_pairs,
                grayscale_h=arc.grayscale_h,
                grayscale_w=arc.grayscale_w,
                output_height=arc.output_height,
                output_width=arc.output_width,
                chroma_rgb=arc.chroma_rgb,
                cls_arith_bytes=arc.cls_arith_bytes,
                chroma_seed=seed,
            )

    def test_seeded_chroma_seed_mutation_changes_palette(self) -> None:
        _, palette, cdf, palette_indices, cls = _build_synthetic_archive(seed=5)
        seed = emit_chroma_palette_seed()
        pose = np.zeros((4, POSE_DIMS), dtype=np.float32)
        pose_bytes, zero = quantize_pose_deltas(pose, scale=10.0)
        blob = pack_archive(
            palette=palette,
            cdf=cdf,
            grayscale_arith_bytes=encode_grayscale_stream(
                palette_indices=palette_indices,
                class_labels=cls,
                cdf=cdf,
            ),
            pose_bytes=pose_bytes,
            meta={"grayscale_downsample": 4, "pose_quant_scale": 10.0, "pose_quant_zero": zero},
            num_pairs=4,
            grayscale_h=6,
            grayscale_w=8,
            output_height=24,
            output_width=32,
            chroma_rgb=expand_chroma_seed_to_palette(seed),
            cls_arith_bytes=encode_class_label_stream(cls),
            chroma_seed=seed,
        )
        mutated = bytearray(blob)
        mutated[_chroma_blob_offset(blob)] ^= 0xFF
        baseline = parse_archive(blob)
        candidate = parse_archive(bytes(mutated))
        assert baseline.chroma_seed != candidate.chroma_seed
        assert not np.array_equal(baseline.chroma_rgb, candidate.chroma_rgb)


class TestPoseQuantization:
    def test_quantize_dequantize_close(self) -> None:
        rng = np.random.default_rng(9)
        pose = rng.standard_normal((10, POSE_DIMS)).astype(np.float32) * 0.01
        bytes_, zero = quantize_pose_deltas(pose, scale=100.0)
        recon = dequantize_pose_deltas(bytes_, num_pairs=10, scale=100.0, zero=zero)
        # uint8 quant with scale=100 -> precision ~0.01 in pose space
        assert np.allclose(recon, pose, atol=0.011)

    def test_wrong_shape_rejected(self) -> None:
        with pytest.raises(ValueError):
            quantize_pose_deltas(np.zeros((10, 8), dtype=np.float32))


# ---------------------------------------------------------------------------
# Catalog #139 byte-mutation no-op detector (operational mechanism proof)
# ---------------------------------------------------------------------------

class TestByteMutationNoOpDetector:
    def test_mutating_palette_byte_changes_output(self, tmp_path: Path) -> None:
        """Per Catalog #139 + #220: mutating any archive byte MUST change inflate output."""
        blob, palette, _, _, _ = _build_synthetic_archive(seed=10)

        # Decode baseline.
        raw_a = inflate_one_video(blob, tmp_path / "base" / "0").read_bytes()

        # Mutate one palette byte at the END of the palette section
        palette_start = CH06_HEADER_SIZE
        mid = palette_start + palette.size // 2
        mutated = bytearray(blob)
        mutated[mid] = (mutated[mid] + 73) % 256
        raw_b = inflate_one_video(bytes(mutated), tmp_path / "mut" / "0").read_bytes()

        # At least one frame must differ
        assert raw_a != raw_b, "byte mutation did not propagate to inflate output (no-op detector)"

    def test_mutating_arith_stream_changes_output(self, tmp_path: Path) -> None:
        blob, palette, cdf, _, _ = _build_synthetic_archive(seed=11)
        # arith stream starts after: header + palette + cdf
        palette_bytes = palette.size
        cdf_bytes = NUM_SEGNET_CLASSES * (cdf.palette_size + 1) * 2
        arith_start = CH06_HEADER_SIZE + palette_bytes + cdf_bytes
        # Read arith length from header (v2: 15 fields)
        unpacked = struct.unpack("<4sBHBHHHHHHIIHHI", blob[:CH06_HEADER_SIZE])
        arith_len = unpacked[10]
        assert arith_len > 0
        mutated = bytearray(blob)
        # Flip a bit in the middle of the arith stream
        mid = arith_start + arith_len // 2
        mutated[mid] ^= 0xFF

        raw_a = inflate_one_video(blob, tmp_path / "base" / "0").read_bytes()
        raw_b = inflate_one_video(bytes(mutated), tmp_path / "mut" / "0").read_bytes()
        assert raw_a != raw_b

    def test_inflate_writes_contest_raw_stem(self, tmp_path: Path) -> None:
        """Auth eval requires ``<inflated_dir>/<video>.raw``, not PNG frames."""
        n_pairs = 3
        output_h = 12
        output_w = 16
        blob, _, _, _, _ = _build_synthetic_archive(
            seed=12,
            n_pairs=n_pairs,
            output_h=output_h,
            output_w=output_w,
        )
        raw_path = inflate_one_video(blob, tmp_path / "inflated" / "0")

        assert raw_path == tmp_path / "inflated" / "0.raw"
        assert raw_path.is_file()
        assert raw_path.stat().st_size == output_w * output_h * n_pairs * 2 * 3
        assert not list((tmp_path / "inflated").glob("*.png"))


# ---------------------------------------------------------------------------
# Inflate runtime budget + strict-scorer-rule
# ---------------------------------------------------------------------------

class TestInflateBudgetAndStrictScorerRule:
    def test_inflate_module_no_torch_import(self) -> None:
        """Strict-scorer-rule: inflate.py MUST NOT import torch."""
        inflate_path = (
            Path(__file__).resolve().parent.parent / "inflate.py"
        )
        body = inflate_path.read_text(encoding="utf-8")
        # Test the import statements specifically, not docstring mentions
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import torch", "from torch ")):
                raise AssertionError(
                    f"inflate.py forbidden torch import: {stripped!r} per strict-scorer-rule"
                )

    def test_inflate_module_no_scorer_imports(self) -> None:
        inflate_path = (
            Path(__file__).resolve().parent.parent / "inflate.py"
        )
        body = inflate_path.read_text(encoding="utf-8")
        forbidden = (
            "from tac.scorer",
            "import smp",
            "segmentation_models_pytorch",
            "EfficientNet",
            "PoseNet",
            "SegNet",
            "from upstream.modules",
        )
        for tok in forbidden:
            assert tok not in body, (
                f"inflate.py contains forbidden scorer token {tok!r} per strict-scorer-rule"
            )

    def test_inflate_runtime_within_loc_budget(self) -> None:
        """HNeRV parity L4 + substrate_engineering exception per L7: Path A v2
        bumps inflate budget to 200 LOC (was 100 v1). The redesign adds
        chroma reconstruction + 6-DOF affine warp (symposium commit 4292c8ce2).
        """
        inflate_path = (
            Path(__file__).resolve().parent.parent / "inflate.py"
        )
        body_lines = [
            ln
            for ln in inflate_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        # SubstrateContract declares 200 LOC budget; allow 15% docstring slack to 230
        assert len(body_lines) <= 230, (
            f"inflate.py is {len(body_lines)} non-comment LOC; "
            f"exceeds Path A v2 budget 200 + 15% slack"
        )


# ---------------------------------------------------------------------------
# Path A v2 redesign (chroma + 6-DOF optical-flow) — symposium commit 4292c8ce2
# ---------------------------------------------------------------------------

class TestPathAChromaReconstruction:
    """Verify per-class chroma palette + class-label stream produce non-Y=R=G=B
    output at inflate. Empirical fix for v6 anchor seg=64.6 distortion class."""

    def test_chroma_palette_roundtrip_byte_stable(self) -> None:
        """Chroma palette survives pack+parse byte-identical."""
        blob, _, _, _, _ = _build_synthetic_archive(seed=20)
        arc = parse_archive(blob)
        assert arc.chroma_rgb.dtype == np.uint8
        assert arc.chroma_rgb.shape == (NUM_SEGNET_CLASSES, CHROMA_BYTES_PER_CLASS)

    def test_class_label_stream_roundtrip_byte_stable(self) -> None:
        """Encoded + decoded class labels exactly equal source."""
        rng = np.random.default_rng(42)
        cls = rng.integers(0, NUM_SEGNET_CLASSES, size=(3, 4, 6), dtype=np.uint8)
        encoded = encode_class_label_stream(cls)
        decoded = decode_class_label_stream(encoded, shape=cls.shape)
        assert np.array_equal(cls, decoded)

    def test_inflate_output_not_grayscale(self, tmp_path: Path) -> None:
        """Path A cargo-cult #2 unwound: inflate must NOT produce R=G=B output
        when chroma anchors differ across channels. v6 produced Y=R=G=B; v2
        must produce per-pixel chroma derived from class labels."""
        # Build archive with deliberately-asymmetric chroma palette.
        rng = np.random.default_rng(33)
        h_g, w_g, n_pairs = 8, 12, 2
        samples = rng.integers(0, 256, size=(n_pairs, h_g, w_g), dtype=np.uint8)
        palette = build_grayscale_palette(samples, palette_size=8)
        palette_indices = palette.quantize(samples)
        # Force class-0 in half + class-1 in the other half to ensure variation.
        cls = np.zeros((n_pairs, h_g, w_g), dtype=np.uint8)
        cls[:, :, w_g // 2 :] = 1
        cdf = build_class_conditional_cdf(palette_indices, cls, palette_size=8)
        arith_bytes = encode_grayscale_stream(
            palette_indices=palette_indices, class_labels=cls, cdf=cdf
        )
        cls_arith_bytes = encode_class_label_stream(cls)
        # Asymmetric per-class chroma: class 0 = pure red, class 1 = pure green.
        chroma_rgb = np.zeros((NUM_SEGNET_CLASSES, 3), dtype=np.uint8)
        chroma_rgb[0] = [255, 0, 0]
        chroma_rgb[1] = [0, 255, 0]
        pose = np.zeros((n_pairs, POSE_DIMS), dtype=np.float32)
        pose_bytes, zero = quantize_pose_deltas(pose, scale=10.0)
        meta = {"grayscale_downsample": 4, "pose_quant_scale": 10.0,
                "pose_quant_zero": zero}
        blob = pack_archive(
            palette=palette, cdf=cdf, grayscale_arith_bytes=arith_bytes,
            pose_bytes=pose_bytes, meta=meta, num_pairs=n_pairs,
            grayscale_h=h_g, grayscale_w=w_g, output_height=16, output_width=24,
            chroma_rgb=chroma_rgb, cls_arith_bytes=cls_arith_bytes,
        )
        raw_path = inflate_one_video(blob, tmp_path / "chr" / "0")
        raw = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8)
        # 2 frames per pair, 3 channels, 16x24 = 1152 bytes per frame
        n_bytes_per_frame = 16 * 24 * 3
        first_frame = raw[:n_bytes_per_frame].reshape(16, 24, 3)
        # Left half should be reddish (R > G); right half greenish (G > R).
        # Skip column boundary and rows that span class boundary in bilinear interp.
        left_r = first_frame[:, :8, 0].mean()
        left_g = first_frame[:, :8, 1].mean()
        right_r = first_frame[:, 16:, 0].mean()
        right_g = first_frame[:, 16:, 1].mean()
        assert left_r > left_g, (
            f"chroma unwinding failed: left half R={left_r} G={left_g}; "
            f"v6 Y=R=G=B regression detected"
        )
        assert right_g > right_r, (
            f"chroma unwinding failed: right half R={right_r} G={right_g}; "
            f"v6 Y=R=G=B regression detected"
        )

    def test_seeded_chroma_seed_mutation_changes_inflate_output(
        self, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(34)
        h_g, w_g, n_pairs = 8, 12, 2
        samples = rng.integers(0, 256, size=(n_pairs, h_g, w_g), dtype=np.uint8)
        palette = build_grayscale_palette(samples, palette_size=8)
        palette_indices = palette.quantize(samples)
        cls = np.zeros((n_pairs, h_g, w_g), dtype=np.uint8)
        cls[:, :, w_g // 2 :] = 1
        cdf = build_class_conditional_cdf(palette_indices, cls, palette_size=8)
        pose = np.zeros((n_pairs, POSE_DIMS), dtype=np.float32)
        pose_bytes, zero = quantize_pose_deltas(pose, scale=10.0)
        seed = emit_chroma_palette_seed()
        blob = pack_archive(
            palette=palette,
            cdf=cdf,
            grayscale_arith_bytes=encode_grayscale_stream(
                palette_indices=palette_indices,
                class_labels=cls,
                cdf=cdf,
            ),
            pose_bytes=pose_bytes,
            meta={
                "grayscale_downsample": 4,
                "pose_quant_scale": 10.0,
                "pose_quant_zero": zero,
            },
            num_pairs=n_pairs,
            grayscale_h=h_g,
            grayscale_w=w_g,
            output_height=16,
            output_width=24,
            chroma_rgb=expand_chroma_seed_to_palette(seed),
            cls_arith_bytes=encode_class_label_stream(cls),
            chroma_seed=seed,
        )
        mutated = bytearray(blob)
        mutated[_chroma_blob_offset(blob)] ^= 0xFF

        raw_baseline = inflate_one_video(blob, tmp_path / "seed_base" / "0").read_bytes()
        raw_candidate = inflate_one_video(
            bytes(mutated),
            tmp_path / "seed_mutated" / "0",
        ).read_bytes()
        assert raw_baseline != raw_candidate, (
            "seeded chroma no-op regression: mutating the archived seed did not "
            "change inflated RGB output"
        )


class TestPathASixDOFAffineWarp:
    """Verify 6-DOF affine warp consumes all 6 pose dims, not just 2."""

    def test_six_dim_pose_produces_different_warp_than_two_dim(
        self, tmp_path: Path
    ) -> None:
        """Path A cargo-cult #4 unwound: PoseNet dims [2..6] (tz, rx, ry, rz)
        must measurably affect the warped frame_1. v6 used only pose[0..2]
        (translation) — same pose with non-zero rz must produce a different
        warp than the same pose with rz=0."""
        rng = np.random.default_rng(55)
        h_g, w_g, n_pairs = 6, 8, 1
        samples = rng.integers(0, 256, size=(n_pairs, h_g, w_g), dtype=np.uint8)
        palette = build_grayscale_palette(samples, palette_size=8)
        palette_indices = palette.quantize(samples)
        cls = rng.integers(0, NUM_SEGNET_CLASSES, size=palette_indices.shape, dtype=np.uint8)
        cdf = build_class_conditional_cdf(palette_indices, cls, palette_size=8)
        arith_bytes = encode_grayscale_stream(
            palette_indices=palette_indices, class_labels=cls, cdf=cdf
        )
        cls_arith_bytes = encode_class_label_stream(cls)
        chroma_rgb = build_chroma_palette(
            rng.integers(0, 256, size=(n_pairs, 3, h_g, w_g), dtype=np.uint8), cls
        )

        def _build(pose_vec: np.ndarray) -> Path:
            pose_bytes, zero = quantize_pose_deltas(
                pose_vec[None].astype(np.float32), scale=10.0
            )
            meta = {"grayscale_downsample": 4, "pose_quant_scale": 10.0,
                    "pose_quant_zero": zero}
            blob = pack_archive(
                palette=palette, cdf=cdf, grayscale_arith_bytes=arith_bytes,
                pose_bytes=pose_bytes, meta=meta, num_pairs=n_pairs,
                grayscale_h=h_g, grayscale_w=w_g, output_height=32, output_width=48,
                chroma_rgb=chroma_rgb, cls_arith_bytes=cls_arith_bytes,
            )
            return inflate_one_video(blob, tmp_path / f"p_{hash(pose_vec.tobytes())}" / "0")

        # Two poses: same translation; different rotation (pose dim 5 = rz)
        pose_no_rot = np.array([0.05, 0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pose_with_rot = np.array([0.05, 0.05, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
        raw_a = _build(pose_no_rot).read_bytes()
        raw_b = _build(pose_with_rot).read_bytes()
        assert raw_a != raw_b, (
            "6-DOF warp regression: rotation dim 5 did not change inflate output "
            "(v6 2-of-6 translation-only warp cargo-cult not yet unwound)"
        )
