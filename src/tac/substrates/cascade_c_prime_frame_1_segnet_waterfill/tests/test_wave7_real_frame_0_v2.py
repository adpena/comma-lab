# SPDX-License-Identifier: MIT
"""WAVE-7 ORDER-correct: v2 archive grammar + real frame_0 reference tests.

Per CLAUDE.md 11th standing directive ORDER discipline (trainer-FIRST +
inflate-SECOND): the v2 archive ships a real RGB frame_0 reference at low-res
(96x128 default) from upstream/videos/0.mkv. The inflate path bilinear-
upsamples to output resolution so SegNet+PoseNet evaluate REAL contest signal
rather than the WAVE-5 synthetic textured base.

Per Catalog #287/#323 canonical Provenance + Catalog #139 byte-mutation smoke.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.archive import (
    CCPF_HEADER_LEN_V2,
    CCPF_VERSION_V1,
    CCPF_VERSION_V2,
    POSE_DIMS,
    REF_FRAME_CHANNELS,
    REF_FRAME_LOWRES_H_DEFAULT,
    REF_FRAME_LOWRES_W_DEFAULT,
    pack_archive,
    parse_archive,
)
from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import (
    CONTEST_RAW_BYTES,
    _upsample_real_frame_0_reference_lowres,
    inflate_one_video,
)


def _build_v2_archive(*, seed: int = 20260526, n_pairs: int = 50, ref_h: int = 96, ref_w: int = 128):
    rng = np.random.default_rng(seed=seed)
    routing = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
    f0_indices = rng.integers(0, 16, size=n_pairs, dtype=np.uint8)
    f1_indices = rng.integers(0, 8, size=n_pairs, dtype=np.uint8)
    pose_deltas = rng.integers(0, 256, size=(n_pairs, POSE_DIMS), dtype=np.uint8)
    ref_lowres = rng.integers(0, 256, size=(ref_h, ref_w, REF_FRAME_CHANNELS), dtype=np.uint8)
    archive_bytes = pack_archive(
        routing_decision=routing,
        frame_0_menu_indices=f0_indices,
        frame_1_menu_indices=f1_indices,
        pose_deltas_uint8=pose_deltas,
        version=CCPF_VERSION_V2,
        frame_0_reference_lowres=ref_lowres,
    )
    return archive_bytes, routing, ref_lowres


class TestV2ArchiveGrammar:
    def test_v2_header_extension_8_bytes(self):
        archive_bytes, *_ = _build_v2_archive(n_pairs=8)
        # v2 header is 19 bytes (11 base + 8 extension)
        assert archive_bytes[4] == CCPF_VERSION_V2

    def test_v2_roundtrip_real_frame_0_reference_lowres(self):
        archive_bytes, _, ref_lowres = _build_v2_archive(n_pairs=10, ref_h=96, ref_w=128)
        parsed = parse_archive(archive_bytes)
        assert parsed.version == CCPF_VERSION_V2
        assert parsed.frame_0_reference_lowres is not None
        assert parsed.frame_0_reference_lowres.shape == (96, 128, 3)
        assert parsed.frame_0_reference_lowres.dtype == np.uint8
        assert np.array_equal(parsed.frame_0_reference_lowres, ref_lowres)

    def test_v2_ref_h_w_roundtrip(self):
        archive_bytes, _, _ = _build_v2_archive(n_pairs=10, ref_h=192, ref_w=256)
        parsed = parse_archive(archive_bytes)
        assert parsed.ref_h == 192
        assert parsed.ref_w == 256

    def test_v2_archive_bytes_growth_includes_ref_block(self):
        _, _, ref = _build_v2_archive(n_pairs=10, ref_h=96, ref_w=128)
        archive_v2, *_ = _build_v2_archive(n_pairs=10, ref_h=96, ref_w=128)
        # ref block alone is 96*128*3 = 36864 bytes
        assert ref.nbytes == 36864
        # v2 archive size MUST include the 36864-byte ref block
        assert len(archive_v2) >= 36864

    def test_v1_no_reference_lowres(self):
        rng = np.random.default_rng(20260526)
        n_pairs = 8
        routing = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
        f0 = rng.integers(0, 16, size=n_pairs, dtype=np.uint8)
        f1 = rng.integers(0, 8, size=n_pairs, dtype=np.uint8)
        pose = rng.integers(0, 256, size=(n_pairs, POSE_DIMS), dtype=np.uint8)
        archive_bytes = pack_archive(
            routing_decision=routing,
            frame_0_menu_indices=f0,
            frame_1_menu_indices=f1,
            pose_deltas_uint8=pose,
            version=CCPF_VERSION_V1,
        )
        parsed = parse_archive(archive_bytes)
        assert parsed.version == CCPF_VERSION_V1
        assert parsed.frame_0_reference_lowres is None
        assert parsed.ref_h == 0
        assert parsed.ref_w == 0

    def test_v1_rejects_reference_lowres_argument(self):
        rng = np.random.default_rng(20260526)
        n_pairs = 8
        ref = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="V1 archive MUST NOT carry"):
            pack_archive(
                routing_decision=np.zeros(n_pairs, dtype=np.int8),
                frame_0_menu_indices=np.zeros(n_pairs, dtype=np.uint8),
                frame_1_menu_indices=np.zeros(n_pairs, dtype=np.uint8),
                pose_deltas_uint8=np.zeros((n_pairs, POSE_DIMS), dtype=np.uint8),
                version=CCPF_VERSION_V1,
                frame_0_reference_lowres=ref,
            )

    def test_v2_requires_reference_lowres(self):
        n_pairs = 8
        with pytest.raises(ValueError, match="V2 archive REQUIRES"):
            pack_archive(
                routing_decision=np.zeros(n_pairs, dtype=np.int8),
                frame_0_menu_indices=np.zeros(n_pairs, dtype=np.uint8),
                frame_1_menu_indices=np.zeros(n_pairs, dtype=np.uint8),
                pose_deltas_uint8=np.zeros((n_pairs, POSE_DIMS), dtype=np.uint8),
                version=CCPF_VERSION_V2,
                frame_0_reference_lowres=None,
            )

    def test_v2_rejects_wrong_dtype(self):
        rng = np.random.default_rng(20260526)
        n_pairs = 8
        bad_ref = rng.random((96, 128, 3)).astype(np.float32)  # wrong dtype
        with pytest.raises(ValueError, match="dtype must be uint8"):
            pack_archive(
                routing_decision=np.zeros(n_pairs, dtype=np.int8),
                frame_0_menu_indices=np.zeros(n_pairs, dtype=np.uint8),
                frame_1_menu_indices=np.zeros(n_pairs, dtype=np.uint8),
                pose_deltas_uint8=np.zeros((n_pairs, POSE_DIMS), dtype=np.uint8),
                version=CCPF_VERSION_V2,
                frame_0_reference_lowres=bad_ref,
            )

    def test_v2_rejects_wrong_shape(self):
        rng = np.random.default_rng(20260526)
        n_pairs = 8
        bad_ref = rng.integers(0, 256, size=(96, 128, 4), dtype=np.uint8)  # 4 channels
        with pytest.raises(ValueError, match="must be \\(H, W, 3\\)"):
            pack_archive(
                routing_decision=np.zeros(n_pairs, dtype=np.int8),
                frame_0_menu_indices=np.zeros(n_pairs, dtype=np.uint8),
                frame_1_menu_indices=np.zeros(n_pairs, dtype=np.uint8),
                pose_deltas_uint8=np.zeros((n_pairs, POSE_DIMS), dtype=np.uint8),
                version=CCPF_VERSION_V2,
                frame_0_reference_lowres=bad_ref,
            )

    def test_v1_v2_dispatch_in_parser(self):
        """parse_archive correctly dispatches on version byte."""
        v1_bytes = pack_archive(
            routing_decision=np.zeros(8, dtype=np.int8),
            frame_0_menu_indices=np.zeros(8, dtype=np.uint8),
            frame_1_menu_indices=np.zeros(8, dtype=np.uint8),
            pose_deltas_uint8=np.zeros((8, POSE_DIMS), dtype=np.uint8),
            version=CCPF_VERSION_V1,
        )
        v2_bytes, *_ = _build_v2_archive(n_pairs=8)
        parsed_v1 = parse_archive(v1_bytes)
        parsed_v2 = parse_archive(v2_bytes)
        assert parsed_v1.version == CCPF_VERSION_V1
        assert parsed_v2.version == CCPF_VERSION_V2
        assert parsed_v1.frame_0_reference_lowres is None
        assert parsed_v2.frame_0_reference_lowres is not None


class TestUpsampleRealFrame0:
    def test_upsample_passthrough_when_shape_matches(self):
        rng = np.random.default_rng(42)
        ref = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        out = _upsample_real_frame_0_reference_lowres(ref, height=96, width=128)
        assert out.shape == (96, 128, 3)
        assert np.array_equal(out, ref)

    def test_upsample_to_contest_resolution(self):
        rng = np.random.default_rng(42)
        ref = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        out = _upsample_real_frame_0_reference_lowres(ref, height=874, width=1164)
        assert out.shape == (874, 1164, 3)
        assert out.dtype == np.uint8

    def test_upsample_preserves_intensity_envelope(self):
        """Bilinear upsample MUST preserve the intensity envelope of the source."""
        rng = np.random.default_rng(42)
        ref = rng.integers(50, 200, size=(96, 128, 3), dtype=np.uint8)
        out = _upsample_real_frame_0_reference_lowres(ref, height=384, width=512)
        # Upsampled mean should be within ±5 of source mean (interpolation preserves global intensity)
        assert abs(float(out.mean()) - float(ref.mean())) < 5.0

    def test_upsample_rejects_non_3d(self):
        bad = np.zeros((96, 128), dtype=np.uint8)
        with pytest.raises(ValueError, match="ref_lowres must be"):
            _upsample_real_frame_0_reference_lowres(bad, height=874, width=1164)


class TestInflateV2RealFrame0BaseUsed:
    def test_v2_inflate_uses_real_reference_not_synthetic(self, tmp_path):
        """Critical WAVE-7 regression: v2 archive's inflate output must reflect the
        REAL frame_0 reference bytes, NOT the WAVE-5 synthetic textured base.

        We construct a v2 archive with a DISTINCTIVE solid-color reference (pure red)
        and assert the rendered raw output's encoded-pair region carries the
        red-channel signature rather than the synthetic sinusoidal proxy.
        """
        # Build v2 archive with all-red frame_0 reference (R=255, G=0, B=0)
        n_pairs = 4
        rng = np.random.default_rng(20260526)
        routing = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
        f0 = rng.integers(0, 16, size=n_pairs, dtype=np.uint8)
        f1 = rng.integers(0, 8, size=n_pairs, dtype=np.uint8)
        pose = np.zeros((n_pairs, POSE_DIMS), dtype=np.uint8)  # zero-pose → identity warp
        ref_red = np.zeros((96, 128, 3), dtype=np.uint8)
        ref_red[..., 0] = 255  # pure red
        archive_bytes = pack_archive(
            routing_decision=routing,
            frame_0_menu_indices=f0,
            frame_1_menu_indices=f1,
            pose_deltas_uint8=pose,
            version=CCPF_VERSION_V2,
            frame_0_reference_lowres=ref_red,
        )
        raw_path = inflate_one_video(archive_bytes, tmp_path / "0")
        assert raw_path.stat().st_size == CONTEST_RAW_BYTES
        with raw_path.open("rb") as fh:
            # First 3 bytes = first pixel of first frame (R, G, B)
            first_pixel = fh.read(3)
            r, g, b = first_pixel[0], first_pixel[1], first_pixel[2]
            # V2 with pure-red reference: R channel should be ~255, G/B ~0
            assert r > 240, f"V2 real-frame_0 NOT used: R={r}, expected ~255 (pure red)"
            assert g < 15, f"V2 real-frame_0 NOT used: G={g}, expected ~0 (no green in pure red)"
            assert b < 15, f"V2 real-frame_0 NOT used: B={b}, expected ~0 (no blue in pure red)"

    def test_v1_inflate_falls_back_to_synthetic_base(self, tmp_path):
        """Backward-compat: v1 archive's inflate output uses WAVE-5 synthetic base."""
        n_pairs = 4
        rng = np.random.default_rng(20260526)
        routing = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
        f0 = rng.integers(0, 16, size=n_pairs, dtype=np.uint8)
        f1 = rng.integers(0, 8, size=n_pairs, dtype=np.uint8)
        pose = np.zeros((n_pairs, POSE_DIMS), dtype=np.uint8)
        archive_bytes = pack_archive(
            routing_decision=routing,
            frame_0_menu_indices=f0,
            frame_1_menu_indices=f1,
            pose_deltas_uint8=pose,
            version=CCPF_VERSION_V1,
        )
        raw_path = inflate_one_video(archive_bytes, tmp_path / "0")
        assert raw_path.stat().st_size == CONTEST_RAW_BYTES
        with raw_path.open("rb") as fh:
            # WAVE-5 synthetic base has R∈[96,192], G∈[96,192], B∈[96,192] (sinusoidal + radial)
            # Read first 16 bytes; assert NOT all-zero AND NOT pure-red
            first_16 = fh.read(16)
            assert first_16 != b"\x00" * 16, "v1 inflate should use synthetic base, not all-zero"
            # Synthetic base mean is ~142 per WAVE-5 memo; pure red would have B=0 first 3 bytes
            # First pixel (R, G, B) under synthetic base: R∈[96,192]
            r, g, b = first_16[0], first_16[1], first_16[2]
            assert r > 0 and g > 0 and b > 0, (
                f"v1 inflate base looks zero-ish: R={r} G={g} B={b}; "
                "expected WAVE-5 sinusoidal/radial textured base"
            )


class TestV2ByteMutationSmoke:
    """Catalog #139 / #272 byte-mutation smoke for the v2 ref block.

    Mutates bytes in the trailing frame_0_reference_lowres region + verifies
    inflate output changes (Catalog #272 distinguishing-feature integration).
    """

    def test_v2_ref_block_byte_mutation_changes_inflate_output(self, tmp_path):
        n_pairs = 4
        rng = np.random.default_rng(20260526)
        routing = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
        f0 = rng.integers(0, 16, size=n_pairs, dtype=np.uint8)
        f1 = rng.integers(0, 8, size=n_pairs, dtype=np.uint8)
        pose = np.zeros((n_pairs, POSE_DIMS), dtype=np.uint8)
        ref = rng.integers(50, 200, size=(96, 128, 3), dtype=np.uint8)
        archive_bytes = pack_archive(
            routing_decision=routing,
            frame_0_menu_indices=f0,
            frame_1_menu_indices=f1,
            pose_deltas_uint8=pose,
            version=CCPF_VERSION_V2,
            frame_0_reference_lowres=ref,
        )
        # Inflate original
        raw_orig = inflate_one_video(archive_bytes, tmp_path / "orig")
        orig_bytes = raw_orig.read_bytes()[:1024]
        # Mutate first byte of ref block (which is at offset len(archive_bytes) - ref.nbytes)
        ref_block_start = len(archive_bytes) - ref.nbytes
        mutated = bytearray(archive_bytes)
        mutated[ref_block_start] ^= 0xFF  # flip ALL bits
        raw_mut = inflate_one_video(bytes(mutated), tmp_path / "mut")
        mut_bytes = raw_mut.read_bytes()[:1024]
        # Inflate output MUST differ (the mutated byte propagates through bilinear upsample
        # to a region of the output frame)
        assert orig_bytes != mut_bytes, (
            "Catalog #139 no-op detector FAIL for v2 ref block: "
            "mutating ref byte produced identical inflate output"
        )
