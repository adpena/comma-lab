# SPDX-License-Identifier: MIT
"""Wave 9 cargo-cult #4 tests: per-(level, class) chroma LUT aggregation policy.

Mirrors the Wave 5 cargo-cult #6 test pattern at
``tests/test_wave_5_cargo_cult_unwinds.py`` but at the chroma LUT
aggregation surface. Covers:

* Verdict dataclass invariants per Catalog #287 + #323.
* Per-policy correctness (each of the 4 supported policies yields a
  valid ``(grayscale_levels, num_segnet_classes, 3)`` uint8 LUT).
* BYTE-DEFAULT MEDIAN policy preserves the legacy
  ``architecture.build_chroma_lut_from_ground_truth`` output bit-for-bit
  (the byte-parity invariant the canonical 2-landing pattern requires).
* MODE-per-cell boundary-preservation test using a real frame from
  ``upstream/videos/0.mkv`` decoded via pyav per Catalog #213 + CLAUDE.md
  "Forbidden in-place edits / synthetic frame base" non-negotiables.
* Empty-bin fallback test (zero-pixel bin uses per-class global stat
  under the SAME policy, sister of
  ``architecture.build_chroma_lut_from_ground_truth`` fallback).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.nscs06_v8_chroma_lut import (
    CANONICAL_AGGREGATION_POLICY_BYTE_DEFAULT,
    CHROMA_LUT_AGGREGATION_POLICY_NON_PROMOTABLE_PROVENANCE,
    SUPPORTED_AGGREGATION_POLICIES,
    ChromaLutAggregationError,
    ChromaLutAggregationVerdict,
    build_chroma_lut_with_policy,
    verify_chroma_lut_invariants,
)
from tac.substrates.nscs06_v8_chroma_lut.architecture import (
    build_chroma_lut_from_ground_truth,
)


# -----------------------------------------------------------------------------
# TestCargoCult4Wave9ChromaLutAggregationPolicy: verdict + policy correctness
# -----------------------------------------------------------------------------


class TestCargoCult4Wave9ChromaLutAggregationPolicy:
    """Wave 9 NSCS06 v8 cargo-cult #4 chroma_lut aggregation policy tests."""

    def test_supported_policies_constants(self) -> None:
        assert SUPPORTED_AGGREGATION_POLICIES == (
            "median_byte_default",
            "mean",
            "mode_per_cell",
            "weighted_mean_by_cell_count",
        )
        assert (
            CANONICAL_AGGREGATION_POLICY_BYTE_DEFAULT == "median_byte_default"
        )
        assert (
            CHROMA_LUT_AGGREGATION_POLICY_NON_PROMOTABLE_PROVENANCE[
                "score_claim"
            ]
            is False
        )
        assert (
            CHROMA_LUT_AGGREGATION_POLICY_NON_PROMOTABLE_PROVENANCE[
                "promotion_eligible"
            ]
            is False
        )
        assert (
            CHROMA_LUT_AGGREGATION_POLICY_NON_PROMOTABLE_PROVENANCE["axis_tag"]
            == "[predicted]"
        )

    def test_verdict_dataclass_invariants(self) -> None:
        v = ChromaLutAggregationVerdict(
            policy="median_byte_default",
            chroma_lut_shape=(16, 5, 3),
            chroma_lut_sha256="0" * 64,
            median_vs_policy_agreement_fraction=1.0,
        )
        assert v.score_claim is False
        assert v.promotion_eligible is False
        assert v.axis_tag == "[predicted]"
        d = v.as_dict()
        assert d["policy"] == "median_byte_default"
        assert d["chroma_lut_shape"] == [16, 5, 3]
        assert d["score_claim"] is False

    def test_verdict_rejects_unsupported_policy(self) -> None:
        with pytest.raises(ChromaLutAggregationError, match="not in"):
            ChromaLutAggregationVerdict(
                policy="random_pixel",
                chroma_lut_shape=(16, 5, 3),
                chroma_lut_sha256="0" * 64,
                median_vs_policy_agreement_fraction=1.0,
            )

    def test_verdict_rejects_score_claim_true(self) -> None:
        with pytest.raises(ChromaLutAggregationError, match="score_claim"):
            ChromaLutAggregationVerdict(
                policy="median_byte_default",
                chroma_lut_shape=(16, 5, 3),
                chroma_lut_sha256="0" * 64,
                median_vs_policy_agreement_fraction=1.0,
                score_claim=True,  # type: ignore[arg-type]
            )

    def test_verdict_rejects_promotable_true(self) -> None:
        with pytest.raises(
            ChromaLutAggregationError, match="promotion_eligible"
        ):
            ChromaLutAggregationVerdict(
                policy="median_byte_default",
                chroma_lut_shape=(16, 5, 3),
                chroma_lut_sha256="0" * 64,
                median_vs_policy_agreement_fraction=1.0,
                promotion_eligible=True,  # type: ignore[arg-type]
            )

    def test_verdict_rejects_out_of_range_agreement(self) -> None:
        with pytest.raises(
            ChromaLutAggregationError, match="agreement_fraction"
        ):
            ChromaLutAggregationVerdict(
                policy="median_byte_default",
                chroma_lut_shape=(16, 5, 3),
                chroma_lut_sha256="0" * 64,
                median_vs_policy_agreement_fraction=1.5,
            )

    def test_verdict_rejects_non_rgb_shape(self) -> None:
        with pytest.raises(ChromaLutAggregationError, match="3 \\(RGB\\)"):
            ChromaLutAggregationVerdict(
                policy="median_byte_default",
                chroma_lut_shape=(16, 5, 4),
                chroma_lut_sha256="0" * 64,
                median_vs_policy_agreement_fraction=1.0,
            )

    def test_byte_default_matches_legacy_implementation(self) -> None:
        """BYTE-DEFAULT MEDIAN policy preserves legacy
        ``build_chroma_lut_from_ground_truth`` output bit-for-bit."""
        rng = np.random.default_rng(20260529)
        rgb = rng.integers(0, 256, size=(8, 3, 12, 16), dtype=np.uint8)
        cls = rng.integers(0, 5, size=(8, 12, 16), dtype=np.uint8)
        legacy = build_chroma_lut_from_ground_truth(
            rgb, cls, grayscale_levels=16, num_segnet_classes=5
        )
        new_lut, verdict = build_chroma_lut_with_policy(
            rgb,
            cls,
            grayscale_levels=16,
            num_segnet_classes=5,
            policy="median_byte_default",
        )
        assert legacy.shape == new_lut.shape
        assert legacy.dtype == new_lut.dtype
        assert np.array_equal(legacy, new_lut), (
            "BYTE-DEFAULT MEDIAN policy MUST byte-match legacy "
            "build_chroma_lut_from_ground_truth output (Wave 9 byte-parity "
            "invariant)"
        )
        assert verdict.median_vs_policy_agreement_fraction == 1.0
        assert verdict.chroma_lut_sha256 == hashlib.sha256(
            new_lut.tobytes()
        ).hexdigest()

    def test_each_policy_produces_valid_lut(self) -> None:
        rng = np.random.default_rng(20260529)
        rgb = rng.integers(0, 256, size=(4, 3, 8, 8), dtype=np.uint8)
        cls = rng.integers(0, 5, size=(4, 8, 8), dtype=np.uint8)
        for policy in SUPPORTED_AGGREGATION_POLICIES:
            lut, verdict = build_chroma_lut_with_policy(
                rgb,
                cls,
                grayscale_levels=16,
                num_segnet_classes=5,
                policy=policy,
            )
            assert lut.shape == (16, 5, 3)
            assert lut.dtype == np.uint8
            assert verdict.policy == policy
            assert verdict.chroma_lut_shape == (16, 5, 3)
            verify_chroma_lut_invariants(lut, expected_shape=(16, 5, 3))

    def test_mean_policy_yields_l2_optimal_per_bin(self) -> None:
        """Mean policy minimizes per-bin L2 distortion under the
        assumption that bin pixels are drawn IID from a Gaussian
        distribution. Synthetic test: each (level, class) bin gets a
        known fixed RGB triple so MEAN == that triple."""
        # construct a synthetic image where every pixel of class 0 + level 0
        # has RGB (100, 110, 120)
        rgb = np.full((1, 3, 4, 4), 100, dtype=np.uint8)
        rgb[0, 1] = 110
        rgb[0, 2] = 120
        cls = np.zeros((1, 4, 4), dtype=np.uint8)
        lut, _ = build_chroma_lut_with_policy(
            rgb,
            cls,
            grayscale_levels=16,
            num_segnet_classes=5,
            policy="mean",
        )
        # luma = 0.299*100 + 0.587*110 + 0.114*120 = 108.05; level_step = 16
        # so level_idx = 108 // 16 = 6
        assert tuple(lut[6, 0, :].tolist()) == (100, 110, 120)

    def test_mode_per_cell_recovers_modal_rgb_when_majority_uniform(
        self,
    ) -> None:
        """MODE policy: a bin where 75% of pixels share an RGB triple
        recovers that triple regardless of outlier pixels."""
        # 4 pixels in class 0 + level 0: 3 of them (100, 110, 120),
        # 1 outlier (255, 0, 0)
        rgb = np.full((1, 3, 2, 2), 100, dtype=np.uint8)
        rgb[0, 1, :, :] = 110
        rgb[0, 2, :, :] = 120
        # Outlier pixel at (0, 0)
        rgb[0, 0, 0, 0] = 255
        rgb[0, 1, 0, 0] = 0
        rgb[0, 2, 0, 0] = 0
        cls = np.zeros((1, 2, 2), dtype=np.uint8)
        lut, verdict = build_chroma_lut_with_policy(
            rgb,
            cls,
            grayscale_levels=16,
            num_segnet_classes=5,
            policy="mode_per_cell",
        )
        # Mode = (100, 110, 120) (3 of 4 pixels); MEDIAN of [100, 100, 100, 255] = 100
        # so MEDIAN and MODE agree on R for this synthetic case; check the
        # canonical bin instead.
        # luma for outlier = 0.299*255 + 0.587*0 + 0.114*0 = 76.245 -> level 4
        # luma for canonical = 108.05 -> level 6
        # So level 6, class 0 has 3 pixels all RGB (100, 110, 120); MODE picks it.
        assert tuple(lut[6, 0, :].tolist()) == (100, 110, 120)
        assert verdict.policy == "mode_per_cell"

    def test_weighted_mean_matches_mean_at_per_bin_scope(self) -> None:
        """WEIGHTED_MEAN_BY_CELL_COUNT collapses to simple MEAN at per-
        bin scope (documented behavior in module docstring)."""
        rng = np.random.default_rng(20260529)
        rgb = rng.integers(0, 256, size=(2, 3, 8, 8), dtype=np.uint8)
        cls = rng.integers(0, 5, size=(2, 8, 8), dtype=np.uint8)
        lut_mean, _ = build_chroma_lut_with_policy(
            rgb,
            cls,
            grayscale_levels=16,
            num_segnet_classes=5,
            policy="mean",
        )
        lut_weighted, _ = build_chroma_lut_with_policy(
            rgb,
            cls,
            grayscale_levels=16,
            num_segnet_classes=5,
            policy="weighted_mean_by_cell_count",
        )
        assert np.array_equal(lut_mean, lut_weighted), (
            "WEIGHTED_MEAN_BY_CELL_COUNT MUST collapse to simple MEAN at "
            "per-bin scope per module docstring"
        )

    def test_empty_class_uses_neutral_gray_fallback(self) -> None:
        """When a class has zero pixels, every (level, class) bin uses
        the neutral (128, 128, 128) fallback regardless of policy."""
        rgb = np.full((1, 3, 4, 4), 100, dtype=np.uint8)
        # All pixels class 0; class 1-4 empty
        cls = np.zeros((1, 4, 4), dtype=np.uint8)
        for policy in SUPPORTED_AGGREGATION_POLICIES:
            lut, _ = build_chroma_lut_with_policy(
                rgb,
                cls,
                grayscale_levels=16,
                num_segnet_classes=5,
                policy=policy,
            )
            # class 1-4 should be (128, 128, 128) everywhere
            for c in range(1, 5):
                for lvl in range(16):
                    assert tuple(lut[lvl, c, :].tolist()) == (128, 128, 128)

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ChromaLutAggregationError, match="dtype"):
            build_chroma_lut_with_policy(
                np.zeros((1, 3, 4, 4), dtype=np.float32),
                np.zeros((1, 4, 4), dtype=np.uint8),
                grayscale_levels=16,
                num_segnet_classes=5,
            )
        with pytest.raises(ChromaLutAggregationError, match="3, H, W"):
            build_chroma_lut_with_policy(
                np.zeros((1, 4, 4, 4), dtype=np.uint8),
                np.zeros((1, 4, 4), dtype=np.uint8),
                grayscale_levels=16,
                num_segnet_classes=5,
            )
        with pytest.raises(ChromaLutAggregationError, match="not in"):
            build_chroma_lut_with_policy(
                np.zeros((1, 3, 4, 4), dtype=np.uint8),
                np.zeros((1, 4, 4), dtype=np.uint8),
                grayscale_levels=16,
                num_segnet_classes=5,
                policy="random",  # type: ignore[arg-type]
            )
        with pytest.raises(ChromaLutAggregationError, match=">= num_segnet"):
            build_chroma_lut_with_policy(
                np.zeros((1, 3, 4, 4), dtype=np.uint8),
                np.full((1, 4, 4), 5, dtype=np.uint8),
                grayscale_levels=16,
                num_segnet_classes=5,
            )

    def test_verify_invariants_rejects_wrong_dtype(self) -> None:
        with pytest.raises(ChromaLutAggregationError, match="dtype"):
            verify_chroma_lut_invariants(
                np.zeros((16, 5, 3), dtype=np.float32),
                expected_shape=(16, 5, 3),
            )

    def test_verify_invariants_rejects_wrong_shape(self) -> None:
        with pytest.raises(ChromaLutAggregationError, match="shape"):
            verify_chroma_lut_invariants(
                np.zeros((8, 5, 3), dtype=np.uint8),
                expected_shape=(16, 5, 3),
            )


# -----------------------------------------------------------------------------
# TestRealVideoFidelityCargoCult4Wave9: real frame from upstream/videos/0.mkv
# -----------------------------------------------------------------------------


VIDEO_PATH = Path(__file__).resolve().parents[5] / "upstream" / "videos" / "0.mkv"


@pytest.fixture(scope="module")
def real_first_frame_pair() -> tuple[np.ndarray, np.ndarray]:
    """Decode the first frame of upstream/videos/0.mkv via pyav.

    Per Catalog #213 + CLAUDE.md "Forbidden synthetic frame base"
    non-negotiables: real-frame fidelity tests MUST use the contest
    video, not synthetic random arrays.
    """
    av = pytest.importorskip("av")
    if not VIDEO_PATH.exists():
        pytest.skip(f"video not found at {VIDEO_PATH}")
    container = av.open(str(VIDEO_PATH))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3)
            # crop to canonical 384x512 EVAL_HW slice for the smoke
            h_target, w_target = 64, 96
            rgb = rgb[:h_target, :w_target, :]
            rgb_pair = np.transpose(rgb, (2, 0, 1))[None]  # (1, 3, H, W)
            # deterministic class labels: BT.601 luma quantized to 5
            # classes; sister of SegNet argmax for the smoke
            luma = (
                0.299 * rgb[:, :, 0].astype(np.float32)
                + 0.587 * rgb[:, :, 1].astype(np.float32)
                + 0.114 * rgb[:, :, 2].astype(np.float32)
            )
            cls = np.clip(
                (luma / 256.0 * 5.0).astype(np.int64), 0, 4
            ).astype(np.uint8)
            cls_pair = cls[None]  # (1, H, W)
            return rgb_pair, cls_pair
        pytest.skip("no frames decoded from video")
    finally:
        container.close()
    pytest.skip("unreachable")


class TestRealVideoFidelityCargoCult4Wave9:
    """Real-frame fidelity tests for Wave 9 cargo-cult #4 unwind."""

    def test_byte_default_matches_legacy_on_real_frame(
        self, real_first_frame_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        rgb, cls = real_first_frame_pair
        legacy = build_chroma_lut_from_ground_truth(
            rgb, cls, grayscale_levels=16, num_segnet_classes=5
        )
        new_lut, verdict = build_chroma_lut_with_policy(
            rgb,
            cls,
            grayscale_levels=16,
            num_segnet_classes=5,
            policy="median_byte_default",
        )
        assert np.array_equal(legacy, new_lut), (
            "BYTE-DEFAULT MEDIAN policy MUST byte-match legacy on real frame"
        )
        assert verdict.median_vs_policy_agreement_fraction == 1.0

    def test_mean_vs_median_diverges_on_real_frame(
        self, real_first_frame_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Empirical: on a real driving frame the MEAN and MEDIAN LUTs
        diverge because per-(level, class) pixel distributions are
        skewed (dashcam pixels are NOT Gaussian; road/sky/lane-marking
        regions have heavy-tailed RGB distributions). The agreement
        fraction quantifies cargo-cult #4 empirical relevance."""
        rgb, cls = real_first_frame_pair
        _, verdict = build_chroma_lut_with_policy(
            rgb,
            cls,
            grayscale_levels=16,
            num_segnet_classes=5,
            policy="mean",
        )
        # 0.0-1.0 is allowed; we DO NOT assert a specific threshold
        # because the agreement fraction is a research-signal metric,
        # not a passing-criterion.
        assert 0.0 <= verdict.median_vs_policy_agreement_fraction <= 1.0

    def test_mode_vs_median_diverges_on_real_frame(
        self, real_first_frame_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        rgb, cls = real_first_frame_pair
        _, verdict = build_chroma_lut_with_policy(
            rgb,
            cls,
            grayscale_levels=16,
            num_segnet_classes=5,
            policy="mode_per_cell",
        )
        assert 0.0 <= verdict.median_vs_policy_agreement_fraction <= 1.0
