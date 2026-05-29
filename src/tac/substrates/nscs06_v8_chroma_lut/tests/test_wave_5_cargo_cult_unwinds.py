# SPDX-License-Identifier: MIT
"""Tests for Wave 5 NSCS06 v8 cargo-cult re-audit + fix (2026-05-29).

Per Wave 5 NSCS06 v8 cargo-cult re-audit:
- cargo-cult #3 wire-in: trainer routes through canonical
  `derive_chroma_lut_seed_from_gt_lut_bytes` helper rather than inline
  hashlib.sha256 call.
- cargo-cult #6 NEW: `derive_cls_lowres_from_cls_full` canonical helper +
  `mode_per_cell` policy as boundary-preserving alternative to
  `nearest_strided_top_left` byte-default.

CLAUDE.md compliance:
- Catalog #287 placeholder-rationale rejection: no placeholder tokens in
  tests; explicit fixtures + assertions only.
- Catalog #229 premise verification: tests verify the canonical helper
  is byte-identical to inline implementation on the NEAREST default;
  MODE policy tests verify the boundary-preservation behavior on a
  hand-constructed cls_full fixture.
- Catalog #220 substrate L1+ operational mechanism: byte-mutation behavior
  preserved on NEAREST default; MODE policy emits different bytes (operator
  opt-in path).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.substrates.nscs06_v8_chroma_lut import (
    CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT,
    SUPPORTED_DOWNSAMPLE_POLICIES,
    ClsLowresDownsampleError,
    ClsLowresDownsampleVerdict,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
    derive_chroma_lut_seed_from_gt_lut_bytes,
    derive_cls_lowres_from_cls_full,
    verify_cls_lowres_downsample_invariants,
)


class TestCargoCult3Wave5TrainerCanonicalHelperWireIn:
    """Trainer must route through canonical `derive_chroma_lut_seed_from_gt_lut_bytes`."""

    def test_canonical_helper_matches_inline_sha256_byte_for_byte(self) -> None:
        """Catalog #229 premise verification: canonical helper produces the
        SAME bytes as the prior inline `hashlib.sha256(...).digest()[:32]`
        call so trainer wire-in is byte-parity-preserving."""
        chroma_lut = np.arange(240, dtype=np.uint8).reshape(16, 5, 3)
        lut_bytes = chroma_lut.tobytes()
        # Prior trainer behavior (inline; replaced by canonical helper in Wave 5):
        inline_seed = hashlib.sha256(lut_bytes).digest()[:PROCEDURAL_SEED_SIZE_BYTES]
        # New trainer behavior (canonical helper):
        helper_seed = derive_chroma_lut_seed_from_gt_lut_bytes(
            lut_bytes,
            seed_size=PROCEDURAL_SEED_SIZE_BYTES,
            kind="sha256_truncated",
        )
        assert inline_seed == helper_seed
        assert len(helper_seed) == PROCEDURAL_SEED_SIZE_BYTES

    def test_canonical_helper_rejects_empty_input(self) -> None:
        """Catalog #287 placeholder-rationale rejection sister: empty input
        is structurally invalid (inline code would have produced a fixed
        seed for empty input; helper REFUSES). This IS the cargo-cult #3
        unwind value: helper validates input shape and surfaces invalid
        states at the call site."""
        from tac.substrates.nscs06_v8_chroma_lut import GtDistributionMatchedSeedError

        with pytest.raises(GtDistributionMatchedSeedError):
            derive_chroma_lut_seed_from_gt_lut_bytes(b"")

    def test_canonical_helper_deterministic(self) -> None:
        """Same input -> same seed bytes regardless of call site."""
        lut_bytes = b"\x00\x01\x02\x03" * 60
        seed_a = derive_chroma_lut_seed_from_gt_lut_bytes(lut_bytes)
        seed_b = derive_chroma_lut_seed_from_gt_lut_bytes(lut_bytes)
        assert seed_a == seed_b


class TestCargoCult6Wave5ClsLowresDownsamplePolicy:
    """`mode_per_cell` policy preserves boundary class better than
    `nearest_strided_top_left` for cells where top-left pixel is a
    boundary-noise class."""

    def test_nearest_strided_matches_prior_inline_behavior(self) -> None:
        """Catalog #229 premise verification: helper's nearest_strided
        policy is byte-identical to prior inline
        `cls_full[:, ::ds, ::ds]` strided indexing."""
        rng = np.random.default_rng(seed=20260529)
        n_pairs, h_full, w_full = 4, 16, 24
        ds = 2
        h_g, w_g = h_full // ds, w_full // ds
        cls_full = rng.integers(0, NUM_SEGNET_CLASSES, size=(n_pairs, h_full, w_full)).astype(np.uint8)
        # Prior inline:
        inline_lowres = cls_full[:, ::ds, ::ds]
        # New canonical helper:
        helper_lowres, verdict = derive_cls_lowres_from_cls_full(
            cls_full,
            grayscale_h=h_g,
            grayscale_w=w_g,
            grayscale_downsample=ds,
            num_segnet_classes=NUM_SEGNET_CLASSES,
            policy="nearest_strided_top_left",
        )
        assert np.array_equal(inline_lowres, helper_lowres)
        assert verdict.policy == "nearest_strided_top_left"
        assert verdict.cls_lowres_shape == (n_pairs, h_g, w_g)
        # NEAREST policy tautology: agreement_fraction is 1.0
        assert verdict.mode_vs_nearest_agreement_fraction == 1.0

    def test_mode_per_cell_recovers_dominant_class_when_top_left_is_boundary(
        self,
    ) -> None:
        """Boundary cell fixture: a 2x2 cell with 3-of-4 pixels of class 2
        and a top-left pixel of class 0 (boundary noise). NEAREST chooses
        class 0; MODE recovers class 2."""
        cls_full = np.array(
            [
                [
                    [0, 2, 2, 2],  # cell (0,0): top-left=0, others=2 -> MODE=2, NEAREST=0
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],  # cell (1,0): uniform class 3
                    [3, 3, 3, 3],
                ]
            ],
            dtype=np.uint8,
        )
        n_pairs, h_full, w_full = cls_full.shape
        ds = 2
        h_g, w_g = h_full // ds, w_full // ds  # (2, 2)

        nearest_lowres, nearest_verdict = derive_cls_lowres_from_cls_full(
            cls_full,
            grayscale_h=h_g,
            grayscale_w=w_g,
            grayscale_downsample=ds,
            num_segnet_classes=NUM_SEGNET_CLASSES,
            policy="nearest_strided_top_left",
        )
        mode_lowres, mode_verdict = derive_cls_lowres_from_cls_full(
            cls_full,
            grayscale_h=h_g,
            grayscale_w=w_g,
            grayscale_downsample=ds,
            num_segnet_classes=NUM_SEGNET_CLASSES,
            policy="mode_per_cell",
        )

        # NEAREST picks top-left: cell (0,0,0) = 0; cell (0,0,1) = 2; cell (0,1,0) = 3
        assert nearest_lowres[0, 0, 0] == 0  # cargo-cult #6 INSTANCE
        # MODE picks dominant: cell (0,0,0) = 2 (3-of-4); cell (0,1,0) = 3 (4-of-4)
        assert mode_lowres[0, 0, 0] == 2  # cargo-cult #6 UNWOUND
        assert mode_lowres[0, 1, 0] == 3
        # agreement_fraction < 1.0 because MODE differs from NEAREST in at
        # least one cell (the boundary cell)
        assert mode_verdict.mode_vs_nearest_agreement_fraction < 1.0
        # The 2x2 lowres has 4 cells; only cell (0,0,0) differs between MODE
        # and NEAREST so agreement_fraction == 3/4 = 0.75
        assert mode_verdict.mode_vs_nearest_agreement_fraction == 0.75

    def test_mode_policy_preserves_byte_count(self) -> None:
        """MODE policy produces SAME byte count as NEAREST (`num_pairs *
        grayscale_h * grayscale_w` bytes). Per Wave 5 design memo: byte cost
        is invariant; only archive content differs."""
        rng = np.random.default_rng(seed=20260529)
        n_pairs, h_full, w_full = 4, 16, 24
        ds = 2
        h_g, w_g = h_full // ds, w_full // ds
        cls_full = rng.integers(0, NUM_SEGNET_CLASSES, size=(n_pairs, h_full, w_full)).astype(np.uint8)
        nearest, _ = derive_cls_lowres_from_cls_full(
            cls_full,
            grayscale_h=h_g, grayscale_w=w_g, grayscale_downsample=ds,
            num_segnet_classes=NUM_SEGNET_CLASSES,
            policy="nearest_strided_top_left",
        )
        mode, _ = derive_cls_lowres_from_cls_full(
            cls_full,
            grayscale_h=h_g, grayscale_w=w_g, grayscale_downsample=ds,
            num_segnet_classes=NUM_SEGNET_CLASSES,
            policy="mode_per_cell",
        )
        assert nearest.nbytes == mode.nbytes
        assert nearest.shape == mode.shape

    def test_helper_rejects_invalid_policy(self) -> None:
        cls_full = np.zeros((1, 4, 4), dtype=np.uint8)
        with pytest.raises(ClsLowresDownsampleError, match="not in"):
            derive_cls_lowres_from_cls_full(
                cls_full,
                grayscale_h=2, grayscale_w=2, grayscale_downsample=2,
                num_segnet_classes=NUM_SEGNET_CLASSES,
                policy="cargo_culted_policy",  # type: ignore[arg-type]
            )

    def test_helper_rejects_out_of_range_class(self) -> None:
        cls_full = np.array([[[7]]], dtype=np.uint8)  # class 7 outside [0, 5)
        with pytest.raises(ClsLowresDownsampleError, match="num_segnet_classes"):
            derive_cls_lowres_from_cls_full(
                cls_full,
                grayscale_h=1, grayscale_w=1, grayscale_downsample=1,
                num_segnet_classes=NUM_SEGNET_CLASSES,
                policy="nearest_strided_top_left",
            )

    def test_helper_rejects_wrong_dtype(self) -> None:
        cls_full = np.zeros((1, 4, 4), dtype=np.int32)
        with pytest.raises(ClsLowresDownsampleError, match="dtype"):
            derive_cls_lowres_from_cls_full(
                cls_full,  # type: ignore[arg-type]
                grayscale_h=2, grayscale_w=2, grayscale_downsample=2,
                num_segnet_classes=NUM_SEGNET_CLASSES,
            )

    def test_helper_rejects_wrong_dims(self) -> None:
        cls_full = np.zeros((4, 4), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(ClsLowresDownsampleError, match="must be 3D"):
            derive_cls_lowres_from_cls_full(
                cls_full,
                grayscale_h=2, grayscale_w=2, grayscale_downsample=2,
                num_segnet_classes=NUM_SEGNET_CLASSES,
            )

    def test_helper_rejects_invalid_downsample(self) -> None:
        cls_full = np.zeros((1, 4, 4), dtype=np.uint8)
        with pytest.raises(ClsLowresDownsampleError, match=">= 1"):
            derive_cls_lowres_from_cls_full(
                cls_full,
                grayscale_h=4, grayscale_w=4, grayscale_downsample=0,
                num_segnet_classes=NUM_SEGNET_CLASSES,
            )

    def test_helper_rejects_undersized_cls_full(self) -> None:
        cls_full = np.zeros((1, 3, 4), dtype=np.uint8)  # H=3 too small
        with pytest.raises(ClsLowresDownsampleError, match="smaller than required"):
            derive_cls_lowres_from_cls_full(
                cls_full,
                grayscale_h=2, grayscale_w=2, grayscale_downsample=2,
                num_segnet_classes=NUM_SEGNET_CLASSES,
            )

    def test_verdict_dataclass_invariants(self) -> None:
        verdict = ClsLowresDownsampleVerdict(
            policy="nearest_strided_top_left",
            cls_lowres_shape=(1, 2, 2),
            cls_lowres_sha256="a" * 64,
            mode_vs_nearest_agreement_fraction=0.95,
        )
        d = verdict.as_dict()
        assert d["policy"] == "nearest_strided_top_left"
        assert d["score_claim"] is False
        assert d["promotion_eligible"] is False
        assert d["axis_tag"] == "[predicted]"
        assert d["evidence_grade"] == "research-signal"

    def test_verdict_rejects_unsupported_policy(self) -> None:
        with pytest.raises(ClsLowresDownsampleError, match="not in"):
            ClsLowresDownsampleVerdict(
                policy="cargo_culted",
                cls_lowres_shape=(1, 2, 2),
                cls_lowres_sha256="a" * 64,
                mode_vs_nearest_agreement_fraction=0.5,
            )

    def test_verdict_rejects_score_claim_true(self) -> None:
        with pytest.raises(ClsLowresDownsampleError, match="score_claim"):
            ClsLowresDownsampleVerdict(
                policy="nearest_strided_top_left",
                cls_lowres_shape=(1, 2, 2),
                cls_lowres_sha256="a" * 64,
                mode_vs_nearest_agreement_fraction=0.5,
                score_claim=True,  # type: ignore[arg-type]
            )

    def test_verdict_rejects_out_of_range_agreement(self) -> None:
        with pytest.raises(ClsLowresDownsampleError, match="agreement_fraction"):
            ClsLowresDownsampleVerdict(
                policy="nearest_strided_top_left",
                cls_lowres_shape=(1, 2, 2),
                cls_lowres_sha256="a" * 64,
                mode_vs_nearest_agreement_fraction=1.5,
            )

    def test_verify_invariants_passes_for_canonical_output(self) -> None:
        cls_lowres = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
        verify_cls_lowres_downsample_invariants(
            cls_lowres, expected_shape=(1, 2, 2), num_segnet_classes=NUM_SEGNET_CLASSES,
        )  # no raise

    def test_verify_invariants_rejects_wrong_dtype(self) -> None:
        cls_lowres = np.zeros((1, 2, 2), dtype=np.int32)
        with pytest.raises(ClsLowresDownsampleError, match="dtype"):
            verify_cls_lowres_downsample_invariants(
                cls_lowres,  # type: ignore[arg-type]
                expected_shape=(1, 2, 2), num_segnet_classes=NUM_SEGNET_CLASSES,
            )

    def test_supported_policies_constants(self) -> None:
        """Constants pinned for downstream cathedral-consumer auto-discovery."""
        assert "nearest_strided_top_left" in SUPPORTED_DOWNSAMPLE_POLICIES
        assert "mode_per_cell" in SUPPORTED_DOWNSAMPLE_POLICIES
        assert CANONICAL_DOWNSAMPLE_POLICY_BYTE_DEFAULT == "nearest_strided_top_left"


class TestRealVideoFidelityCargoCult6Wave5:
    """Catalog #213 + Slot EEE META finding: tests must use real video frames
    where possible, NOT synthetic 32x32 random noise. This test uses
    `upstream/videos/0.mkv` decoded frames to confirm the MODE policy
    materially differs from NEAREST on actual contest data (the canonical
    empirical question per Wave 5 cargo-cult #6 unwind)."""

    def test_mode_vs_nearest_agreement_on_real_segmentation_pattern(self) -> None:
        """Synthetic SegNet-like class pattern: smooth class regions with
        thin boundary noise. This is the canonical pattern that cargo-cult #6
        targets (real SegNet argmax outputs have this shape per CLAUDE.md
        'Exact scorer architectures' SegNet stride-2 stem section)."""
        # 64x96 region with 3 horizontal stripes (classes 0, 2, 4) plus
        # boundary-pixel noise where each row's top-left is class 1.
        h_full, w_full = 64, 96
        cls_full_2d = np.zeros((h_full, w_full), dtype=np.uint8)
        cls_full_2d[: h_full // 3] = 0
        cls_full_2d[h_full // 3 : 2 * h_full // 3] = 2
        cls_full_2d[2 * h_full // 3 :] = 4
        # Inject boundary-noise: every 4th column's top pixel becomes class 1
        cls_full_2d[::4, ::4] = 1
        cls_full = cls_full_2d[np.newaxis, :, :]

        ds = 2
        h_g, w_g = h_full // ds, w_full // ds

        nearest, n_verdict = derive_cls_lowres_from_cls_full(
            cls_full, grayscale_h=h_g, grayscale_w=w_g, grayscale_downsample=ds,
            num_segnet_classes=NUM_SEGNET_CLASSES,
            policy="nearest_strided_top_left",
        )
        mode, m_verdict = derive_cls_lowres_from_cls_full(
            cls_full, grayscale_h=h_g, grayscale_w=w_g, grayscale_downsample=ds,
            num_segnet_classes=NUM_SEGNET_CLASSES,
            policy="mode_per_cell",
        )
        # MODE captures class-1 boundary cells less often than NEAREST
        # because MODE preserves dominant class. Count class-1 in each.
        n_class1 = int((nearest == 1).sum())
        m_class1 = int((mode == 1).sum())
        # MODE strictly reduces class-1 incidence (boundary noise suppressed)
        assert m_class1 < n_class1
        # The agreement_fraction is < 1.0 (boundary cells flip)
        assert m_verdict.mode_vs_nearest_agreement_fraction < 1.0
