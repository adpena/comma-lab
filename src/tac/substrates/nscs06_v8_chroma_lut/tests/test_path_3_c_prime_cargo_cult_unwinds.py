# SPDX-License-Identifier: MIT
"""Tests for Path 3 C' Phase 3 cargo-cult-unwind extensions.

Per Path 3 C' Phase 1 audit (commit ``a6e2a06e3``) + Phase 2 design memo
(commit ``bac0ec05d``). Covers the 4 critical cargo-cult unwind paths:
- cargo-cult #3 via :mod:`gt_distribution_matched_seed`
- cargo-cult #5 via :mod:`distinguishing_feature_smoke`
- cargo-cult #8 via :mod:`mlx_iteration.measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline`
- cargo-cult #12 via :mod:`predicted_band_axis_attribution`

Plus regression guards verifying predecessor + sister tests still pass.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.nscs06_v8_chroma_lut import (
    CHROMA_LUT_BYTES_DEFAULT,
    DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME,
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    POSE_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN,
    POSE_DIMS,
    PREDICTED_BAND_VALIDATION_STATUS_PENDING_TOKEN,
    PREDICTED_BAND_VALIDATION_STATUS_VALIDATED_TOKEN,
    PROCEDURAL_SEED_SIZE_BYTES,
    SEG_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN,
    SUPPORTED_SEED_DERIVATION_KINDS,
    TOTAL_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN,
    DistinguishingFeatureSmokeError,
    GtDistributionMatchedSeedError,
    GtDistributionMatchedSeedVerdict,
    PerClassChromaDistinguishingFeatureVerdict,
    SegNetArgmaxDisplacementVerdict,
    axis_attribution_to_dict_for_metadata_json,
    compute_lut_byte_offset_for_class,
    derive_chroma_lut_seed_from_gt_lut_bytes,
    expand_gt_matched_seed_to_lut,
    is_predicted_band_validated_post_training,
    mutate_class_anchor_bytes_in_archive,
    pack_archive,
    predicted_delta_s_with_axis_attribution,
    verify_per_class_chroma_anchors_consumed_at_inflate,
    verify_seed_encodes_gt_fingerprint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_synthetic_v1_archive_with_distinct_class_anchors(
    *, seed: int = 12345
) -> bytes:
    """Build a tiny CH08 v1 inline-LUT archive with DISTINCT per-class anchors.

    Distinct per-class anchors are required for the per-class byte-mutation
    smoke to be meaningful (if all classes had identical anchors, mutating
    any class would produce identical frames regardless of inflate-side
    consumption).
    """
    rng = np.random.RandomState(seed)
    num_pairs = 2
    gh, gw = 4, 6
    out_h, out_w = 16, 24
    # Build LUT with DISTINCT per-class anchors (each class gets a unique
    # base RGB triplet across all levels).
    lut = np.zeros(
        (GRAYSCALE_LEVELS_DEFAULT, NUM_SEGNET_CLASSES, 3), dtype=np.uint8
    )
    for c in range(NUM_SEGNET_CLASSES):
        base_r = (50 * c) % 256
        base_g = (80 * c) % 256
        base_b = (110 * c) % 256
        for lvl in range(GRAYSCALE_LEVELS_DEFAULT):
            lut[lvl, c, 0] = np.uint8((base_r + lvl * 5) % 256)
            lut[lvl, c, 1] = np.uint8((base_g + lvl * 5) % 256)
            lut[lvl, c, 2] = np.uint8((base_b + lvl * 5) % 256)
    pose_bytes = rng.randint(
        0, 256, size=num_pairs * POSE_DIMS, dtype=np.uint8
    ).tobytes()
    grayscale_bytes = rng.randint(
        0, 256, size=num_pairs * gh * gw, dtype=np.uint8
    ).tobytes()
    blob = pack_archive(
        num_pairs=num_pairs,
        grayscale_h=gh,
        grayscale_w=gw,
        output_height=out_h,
        output_width=out_w,
        pose_bytes=pose_bytes,
        grayscale_bytes=grayscale_bytes,
        chroma_lut=lut,
    )
    return blob


# ---------------------------------------------------------------------------
# Cargo-cult #5 unwind: per-class chroma byte-mutation distinguishing-feature smoke
# ---------------------------------------------------------------------------


class TestPerClassChromaDistinguishingFeatureSmoke:
    """Tests for :func:`verify_per_class_chroma_anchors_consumed_at_inflate`."""

    def test_compute_lut_byte_offset_for_class_validates_bounds(self) -> None:
        with pytest.raises(DistinguishingFeatureSmokeError, match="outside"):
            compute_lut_byte_offset_for_class(
                -1, grayscale_levels=16, num_segnet_classes=5
            )
        with pytest.raises(DistinguishingFeatureSmokeError, match="outside"):
            compute_lut_byte_offset_for_class(
                5, grayscale_levels=16, num_segnet_classes=5
            )

    def test_compute_lut_byte_offset_for_class_0_spans_correctly(self) -> None:
        start, span = compute_lut_byte_offset_for_class(
            0, grayscale_levels=16, num_segnet_classes=5
        )
        # class=0 starts at byte 0; encompasses from (lvl=0, c=0, ch=0)=0 to
        # (lvl=15, c=0, ch=2) = (15*5+0)*3+2 = 227; span = 228.
        assert start == 0
        assert span == 228

    def test_compute_lut_byte_offset_for_class_4_starts_at_class_offset(
        self,
    ) -> None:
        # class=4 starts at byte (0*5+4)*3 = 12.
        start, span = compute_lut_byte_offset_for_class(
            4, grayscale_levels=16, num_segnet_classes=5
        )
        assert start == 12
        # Last index = (15*5+4)*3+2 = 79*3+2 = 239; span = 239-12+1 = 228.
        assert span == 228

    def test_mutate_class_anchor_bytes_v1_archive_preserves_length(self) -> None:
        archive = _build_synthetic_v1_archive_with_distinct_class_anchors()
        mutated = mutate_class_anchor_bytes_in_archive(archive, 1)
        assert len(mutated) == len(archive)

    def test_mutate_class_anchor_bytes_v1_archive_only_changes_target_class(
        self,
    ) -> None:
        archive = _build_synthetic_v1_archive_with_distinct_class_anchors()
        mutated = mutate_class_anchor_bytes_in_archive(archive, 2)
        assert mutated != archive  # SOMETHING changed
        # Verify other-class bytes are byte-identical.
        from tac.substrates.nscs06_v8_chroma_lut import CH08_HEADER_SIZE
        from tac.substrates.nscs06_v8_chroma_lut.archive import parse_archive

        arc_orig = parse_archive(archive)
        arc_mutated = parse_archive(mutated)
        # All bytes except the LUT dense bytes should be byte-identical.
        # In particular: pose_bytes + grayscale_bytes must match exactly.
        assert arc_orig.pose_bytes == arc_mutated.pose_bytes
        assert arc_orig.grayscale_bytes == arc_mutated.grayscale_bytes
        # LUT[:, c=0, :] should be unchanged; LUT[:, c=2, :] should differ.
        assert np.array_equal(arc_orig.chroma_lut[:, 0, :], arc_mutated.chroma_lut[:, 0, :])
        assert np.array_equal(arc_orig.chroma_lut[:, 1, :], arc_mutated.chroma_lut[:, 1, :])
        assert not np.array_equal(arc_orig.chroma_lut[:, 2, :], arc_mutated.chroma_lut[:, 2, :])
        assert np.array_equal(arc_orig.chroma_lut[:, 3, :], arc_mutated.chroma_lut[:, 3, :])
        assert np.array_equal(arc_orig.chroma_lut[:, 4, :], arc_mutated.chroma_lut[:, 4, :])

    def test_mutate_class_anchor_bytes_rejects_v2_archive(self) -> None:
        """v2 procedural-seed has no inline LUT bytes to mutate."""
        rng = np.random.RandomState(99)
        num_pairs = 2
        gh, gw = 4, 6
        out_h, out_w = 16, 24
        pose_bytes = rng.randint(
            0, 256, size=num_pairs * POSE_DIMS, dtype=np.uint8
        ).tobytes()
        grayscale_bytes = rng.randint(
            0, 256, size=num_pairs * gh * gw, dtype=np.uint8
        ).tobytes()
        seed = rng.randint(
            0, 256, size=PROCEDURAL_SEED_SIZE_BYTES, dtype=np.uint8
        ).tobytes()
        v2_archive = pack_archive(
            num_pairs=num_pairs,
            grayscale_h=gh,
            grayscale_w=gw,
            output_height=out_h,
            output_width=out_w,
            pose_bytes=pose_bytes,
            grayscale_bytes=grayscale_bytes,
            chroma_seed=seed,
        )
        with pytest.raises(DistinguishingFeatureSmokeError, match="v1 inline-LUT"):
            mutate_class_anchor_bytes_in_archive(v2_archive, 1)

    def test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5(
        self, tmp_path: Path
    ) -> None:
        """THE CANONICAL EMPIRICAL CONFIRMATION of Path 3 C' Phase 1 cargo-cult #5.

        L0 SCAFFOLD inflate uses cls=0 uniformly. Mutating LUT[:, 1, :] should
        produce IDENTICAL frame-1 bytes because class=1 anchor is never consumed.
        The verdict at L0 is FAIL_AT_CLASS_1 (or FAIL_AT_CLASS_<smallest c>).

        This test EMPIRICALLY CONFIRMS the structural test-invalidity finding
        from Phase 1 audit. The L1 promotion blocker is to wire cls_stream
        consumption at inflate so this verdict becomes PASS_PER_CLASS.
        """
        archive = _build_synthetic_v1_archive_with_distinct_class_anchors()
        verdict = verify_per_class_chroma_anchors_consumed_at_inflate(
            archive, tmp_path, classes_to_mutate=(1, 2, 3, 4)
        )
        # At L0 SCAFFOLD (current inflate.py:185 cls=0 uniform), mutating
        # any class>=1 anchor produces NO frame changes.
        assert verdict.verdict_kind.startswith("FAIL_AT_CLASS_"), (
            f"Expected FAIL_AT_CLASS_* at L0 SCAFFOLD per cargo-cult #5 audit; "
            f"got verdict_kind={verdict.verdict_kind!r}"
        )
        # ALL mutated classes (1,2,3,4) should be in classes_without_frame_changes
        # because cls=0 uniformly is consumed; none of them affect frames.
        assert verdict.classes_without_frame_changes == (1, 2, 3, 4)
        assert verdict.classes_with_frame_changes == ()
        # Non-promotable contract.
        assert verdict.score_claim is False
        assert verdict.promotion_eligible is False
        assert verdict.axis_tag == "[structural-verifier]"

    def test_verify_per_class_chroma_at_L0_class_0_baseline_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Sanity: mutating class=0 anchor (consumed at L0) DOES change frames."""
        archive = _build_synthetic_v1_archive_with_distinct_class_anchors()
        verdict = verify_per_class_chroma_anchors_consumed_at_inflate(
            archive, tmp_path, classes_to_mutate=(0,)
        )
        # Class=0 IS the only consumed class at L0; mutating it should change frames.
        assert verdict.verdict_kind == "PASS_PER_CLASS"
        assert verdict.classes_with_frame_changes == (0,)
        assert verdict.classes_without_frame_changes == ()

    def test_per_class_verdict_rejects_promotable_construction(self) -> None:
        with pytest.raises(DistinguishingFeatureSmokeError, match="score_claim"):
            PerClassChromaDistinguishingFeatureVerdict(
                verdict_kind="PASS_PER_CLASS",
                classes_mutated=(1, 2),
                classes_with_frame_changes=(1, 2),
                classes_without_frame_changes=(),
                baseline_frame1_sha256="abc",
                mutated_frame1_sha256_per_class={1: "x", 2: "y"},
                score_claim=True,  # type: ignore[arg-type]  # noqa: PIE807
            )

    def test_per_class_verdict_rejects_invalid_verdict_kind(self) -> None:
        with pytest.raises(DistinguishingFeatureSmokeError, match="verdict_kind"):
            PerClassChromaDistinguishingFeatureVerdict(
                verdict_kind="UNKNOWN_KIND",
                classes_mutated=(1,),
                classes_with_frame_changes=(1,),
                classes_without_frame_changes=(),
                baseline_frame1_sha256="x",
                mutated_frame1_sha256_per_class={1: "y"},
            )

    def test_per_class_verdict_rejects_disjoint_partition_violation(self) -> None:
        # classes_mutated = (1,2,3); classes_with_changes = (1,2); without = ()
        # Union {1,2} != {1,2,3} → reject.
        with pytest.raises(DistinguishingFeatureSmokeError, match="classes_mutated"):
            PerClassChromaDistinguishingFeatureVerdict(
                verdict_kind="PASS_PER_CLASS",
                classes_mutated=(1, 2, 3),
                classes_with_frame_changes=(1, 2),
                classes_without_frame_changes=(),
                baseline_frame1_sha256="x",
                mutated_frame1_sha256_per_class={1: "a", 2: "b", 3: "c"},
            )


# ---------------------------------------------------------------------------
# Cargo-cult #3 unwind: GT-distribution-matched seed derivation
# ---------------------------------------------------------------------------


class TestGtDistributionMatchedSeed:
    """Tests for :func:`derive_chroma_lut_seed_from_gt_lut_bytes`."""

    def test_derive_seed_is_deterministic_sha256_truncated(self) -> None:
        gt_bytes = b"\x12\x34\x56" * 100
        seed_a = derive_chroma_lut_seed_from_gt_lut_bytes(
            gt_bytes, kind="sha256_truncated"
        )
        seed_b = derive_chroma_lut_seed_from_gt_lut_bytes(
            gt_bytes, kind="sha256_truncated"
        )
        assert seed_a == seed_b
        assert len(seed_a) == PROCEDURAL_SEED_SIZE_BYTES

    def test_derive_seed_is_deterministic_blake2b_truncated(self) -> None:
        gt_bytes = b"\xab\xcd\xef" * 50
        seed_a = derive_chroma_lut_seed_from_gt_lut_bytes(
            gt_bytes, kind="blake2b_truncated"
        )
        seed_b = derive_chroma_lut_seed_from_gt_lut_bytes(
            gt_bytes, kind="blake2b_truncated"
        )
        assert seed_a == seed_b
        assert len(seed_a) == PROCEDURAL_SEED_SIZE_BYTES

    def test_derive_seed_distinct_gt_inputs_produce_distinct_seeds(self) -> None:
        seed_a = derive_chroma_lut_seed_from_gt_lut_bytes(b"input_a" * 10)
        seed_b = derive_chroma_lut_seed_from_gt_lut_bytes(b"input_b" * 10)
        assert seed_a != seed_b

    def test_derive_seed_rejects_empty(self) -> None:
        with pytest.raises(GtDistributionMatchedSeedError, match="non-empty"):
            derive_chroma_lut_seed_from_gt_lut_bytes(b"")

    def test_derive_seed_rejects_non_bytes(self) -> None:
        with pytest.raises(GtDistributionMatchedSeedError, match="bytes-like"):
            derive_chroma_lut_seed_from_gt_lut_bytes("not_bytes")  # type: ignore[arg-type]

    def test_derive_seed_rejects_invalid_size(self) -> None:
        with pytest.raises(GtDistributionMatchedSeedError, match="outside"):
            derive_chroma_lut_seed_from_gt_lut_bytes(b"x" * 10, seed_size=7)
        with pytest.raises(GtDistributionMatchedSeedError, match="outside"):
            derive_chroma_lut_seed_from_gt_lut_bytes(b"x" * 10, seed_size=300)

    def test_derive_seed_rejects_unsupported_kind(self) -> None:
        with pytest.raises(GtDistributionMatchedSeedError, match="kind="):
            derive_chroma_lut_seed_from_gt_lut_bytes(
                b"x" * 10, kind="unknown_kind"  # type: ignore[arg-type]
            )

    def test_derive_seed_supports_custom_seed_size(self) -> None:
        for size in (8, 16, 32, 64, 128):
            seed = derive_chroma_lut_seed_from_gt_lut_bytes(
                b"gt_lut_bytes" * 10, seed_size=size
            )
            assert len(seed) == size

    def test_expand_seed_to_lut_returns_canonical_shape(self) -> None:
        seed = derive_chroma_lut_seed_from_gt_lut_bytes(b"gt" * 100)
        lut = expand_gt_matched_seed_to_lut(seed)
        assert lut.shape == (
            GRAYSCALE_LEVELS_DEFAULT,
            NUM_SEGNET_CLASSES,
            3,
        )
        assert lut.dtype == np.uint8

    def test_expand_seed_is_deterministic(self) -> None:
        seed = derive_chroma_lut_seed_from_gt_lut_bytes(b"gt" * 50)
        lut_a = expand_gt_matched_seed_to_lut(seed)
        lut_b = expand_gt_matched_seed_to_lut(seed)
        assert np.array_equal(lut_a, lut_b)

    def test_verify_seed_encodes_gt_fingerprint_distinct_inputs_distinct_seeds(
        self,
    ) -> None:
        assert verify_seed_encodes_gt_fingerprint(b"gt_a" * 50, b"gt_b" * 50)

    def test_verify_seed_encodes_gt_fingerprint_identical_inputs_returns_false(
        self,
    ) -> None:
        assert not verify_seed_encodes_gt_fingerprint(b"gt" * 50, b"gt" * 50)

    def test_gt_matched_seed_verdict_carries_non_promotable_contract(self) -> None:
        seed = derive_chroma_lut_seed_from_gt_lut_bytes(b"gt" * 50)
        verdict = GtDistributionMatchedSeedVerdict(
            seed_kind="sha256_truncated",
            seed_bytes=seed,
            seed_sha256=hashlib.sha256(seed).hexdigest(),
            derived_lut_bytes=bytes(240),
            derived_lut_sha256=hashlib.sha256(bytes(240)).hexdigest(),
            gt_lut_input_sha256=hashlib.sha256(b"gt" * 50).hexdigest(),
        )
        assert verdict.score_claim is False
        assert verdict.promotion_eligible is False
        assert verdict.evidence_grade == "research-signal"
        as_dict = verdict.as_dict()
        assert as_dict["score_claim"] is False
        assert as_dict["axis_tag"] == "[macOS-MLX research-signal]"

    def test_gt_matched_seed_verdict_rejects_wrong_size(self) -> None:
        with pytest.raises(GtDistributionMatchedSeedError, match="seed_bytes length"):
            GtDistributionMatchedSeedVerdict(
                seed_kind="sha256_truncated",
                seed_bytes=b"too_short",  # not 32 bytes
                seed_sha256="x",
                derived_lut_bytes=bytes(240),
                derived_lut_sha256="y",
                gt_lut_input_sha256="z",
            )

    def test_supported_seed_kinds_canonical_set(self) -> None:
        assert "sha256_truncated" in SUPPORTED_SEED_DERIVATION_KINDS
        assert "blake2b_truncated" in SUPPORTED_SEED_DERIVATION_KINDS


# ---------------------------------------------------------------------------
# Cargo-cult #12 unwind: Catalog #324-grade predicted-band axis attribution
# ---------------------------------------------------------------------------


class TestPredictedBandAxisAttribution:
    """Tests for :func:`predicted_delta_s_with_axis_attribution`."""

    def test_predicted_delta_s_with_axis_attribution_returns_canonical_dict(
        self,
    ) -> None:
        result = predicted_delta_s_with_axis_attribution()
        assert isinstance(result, dict)
        # rate_axis is canonical equation #26 closed-form float
        assert isinstance(result["rate_axis"], float)
        assert result["rate_axis"] < 0  # negative ΔS = score improvement
        # seg + pose + total axes carry the canonical UNKNOWN token
        assert result["seg_axis"] == SEG_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN
        assert result["pose_axis"] == POSE_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN
        assert result["total_axis"] == TOTAL_AXIS_UNKNOWN_PENDING_PAIRED_SMOKE_TOKEN
        assert result["canonical_equation_in_domain_context"] == "nscs06_v8_chroma_lut"
        assert (
            result["validation_status"]
            == PREDICTED_BAND_VALIDATION_STATUS_PENDING_TOKEN
        )

    def test_axis_attribution_to_metadata_json_dict_carries_provenance(self) -> None:
        result = axis_attribution_to_dict_for_metadata_json()
        # All canonical-Provenance contract fields present
        assert result["score_claim"] is False
        assert result["promotion_eligible"] is False
        assert result["ready_for_exact_eval_dispatch"] is False
        assert result["evidence_grade"] == "predicted"
        assert "axis_tag" in result
        assert "rate-axis-only" in result["axis_tag"]
        assert "seg+pose-pending-per-catalog-324" in result["axis_tag"]
        assert "reactivation_criteria" in result
        assert (
            result["predicted_band_validation_status"]
            == PREDICTED_BAND_VALIDATION_STATUS_PENDING_TOKEN
        )

    def test_is_predicted_band_validated_post_training_default_false(self) -> None:
        default_dict = predicted_delta_s_with_axis_attribution()
        assert is_predicted_band_validated_post_training(default_dict) is False

    def test_is_predicted_band_validated_post_training_true_when_all_axes_float(
        self,
    ) -> None:
        validated = {
            "rate_axis": -0.002706,
            "seg_axis": -0.005,
            "pose_axis": -0.001,
            "validation_status": PREDICTED_BAND_VALIDATION_STATUS_VALIDATED_TOKEN,
        }
        assert is_predicted_band_validated_post_training(validated) is True

    def test_is_predicted_band_validated_post_training_false_when_axes_nan(
        self,
    ) -> None:
        invalid = {
            "rate_axis": float("nan"),
            "seg_axis": 0.0,
            "pose_axis": 0.0,
            "validation_status": PREDICTED_BAND_VALIDATION_STATUS_VALIDATED_TOKEN,
        }
        assert is_predicted_band_validated_post_training(invalid) is False

    def test_is_predicted_band_validated_post_training_false_when_axes_inf(
        self,
    ) -> None:
        invalid = {
            "rate_axis": float("inf"),
            "seg_axis": 0.0,
            "pose_axis": 0.0,
            "validation_status": PREDICTED_BAND_VALIDATION_STATUS_VALIDATED_TOKEN,
        }
        assert is_predicted_band_validated_post_training(invalid) is False

    def test_is_predicted_band_validated_post_training_false_when_status_pending(
        self,
    ) -> None:
        pending = {
            "rate_axis": -0.002706,
            "seg_axis": -0.005,
            "pose_axis": -0.001,
            "validation_status": PREDICTED_BAND_VALIDATION_STATUS_PENDING_TOKEN,
        }
        assert is_predicted_band_validated_post_training(pending) is False

    def test_is_predicted_band_validated_post_training_false_when_non_dict(
        self,
    ) -> None:
        assert is_predicted_band_validated_post_training("not a dict") is False  # type: ignore[arg-type]
        assert is_predicted_band_validated_post_training(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cargo-cult #8 unwind: SegNet noise-floor probe (Catalog #305 observability)
# ---------------------------------------------------------------------------


class TestSegNetArgmaxDisplacementVerdict:
    """Tests for the noise-floor verdict dataclass (no MLX required for these)."""

    def test_displacement_verdict_constructable_with_canonical_pass(self) -> None:
        verdict = SegNetArgmaxDisplacementVerdict(
            num_pairs_probed=10,
            argmax_displacement_fraction=0.05,  # above 1e-3 threshold
            noise_floor_threshold=DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME,
            in_noise_floor=False,
            recommended_proceed=True,
            max_per_pair_displacement=0.08,
            mean_per_pair_displacement=0.05,
        )
        assert verdict.score_claim is False
        assert verdict.recommended_proceed is True

    def test_displacement_verdict_constructable_with_canonical_fail(self) -> None:
        verdict = SegNetArgmaxDisplacementVerdict(
            num_pairs_probed=10,
            argmax_displacement_fraction=1e-5,  # below threshold
            noise_floor_threshold=DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME,
            in_noise_floor=True,
            recommended_proceed=False,
            max_per_pair_displacement=1e-5,
            mean_per_pair_displacement=1e-5,
        )
        assert verdict.in_noise_floor is True
        assert verdict.recommended_proceed is False

    def test_displacement_verdict_rejects_inconsistent_in_noise_floor(self) -> None:
        # displacement (0.5) > threshold (1e-3) → expected in_noise_floor=False
        # Pass in_noise_floor=True → reject.
        from tac.substrates.nscs06_v8_chroma_lut import MLXIterationError

        with pytest.raises(MLXIterationError, match="in_noise_floor inconsistent"):
            SegNetArgmaxDisplacementVerdict(
                num_pairs_probed=10,
                argmax_displacement_fraction=0.5,
                noise_floor_threshold=DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME,
                in_noise_floor=True,  # wrong; should be False
                recommended_proceed=False,
                max_per_pair_displacement=0.6,
                mean_per_pair_displacement=0.5,
            )

    def test_displacement_verdict_rejects_inconsistent_recommended_proceed(
        self,
    ) -> None:
        from tac.substrates.nscs06_v8_chroma_lut import MLXIterationError

        with pytest.raises(MLXIterationError, match="recommended_proceed"):
            SegNetArgmaxDisplacementVerdict(
                num_pairs_probed=10,
                argmax_displacement_fraction=0.05,
                noise_floor_threshold=DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME,
                in_noise_floor=False,
                recommended_proceed=False,  # should be True (recommended_proceed = NOT in_noise_floor)
                max_per_pair_displacement=0.08,
                mean_per_pair_displacement=0.05,
            )

    def test_displacement_verdict_rejects_promotable(self) -> None:
        from tac.substrates.nscs06_v8_chroma_lut import MLXIterationError

        with pytest.raises(MLXIterationError, match="score_claim"):
            SegNetArgmaxDisplacementVerdict(
                num_pairs_probed=10,
                argmax_displacement_fraction=0.05,
                noise_floor_threshold=DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME,
                in_noise_floor=False,
                recommended_proceed=True,
                max_per_pair_displacement=0.08,
                mean_per_pair_displacement=0.05,
                score_claim=True,
            )

    def test_displacement_verdict_as_dict_carries_canonical_fields(self) -> None:
        verdict = SegNetArgmaxDisplacementVerdict(
            num_pairs_probed=5,
            argmax_displacement_fraction=0.05,
            noise_floor_threshold=DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME,
            in_noise_floor=False,
            recommended_proceed=True,
            max_per_pair_displacement=0.08,
            mean_per_pair_displacement=0.05,
        )
        as_dict = verdict.as_dict()
        assert as_dict["score_claim"] is False
        assert as_dict["promotion_eligible"] is False
        assert as_dict["axis_tag"] == "[macOS-MLX research-signal]"
        assert as_dict["evidence_grade"] == "research-signal"
        assert "blockers" in as_dict


# ---------------------------------------------------------------------------
# Regression guards: predecessor + sister tests still pass
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    """Verify Path 3 C' Phase 3 additions did not break sister + predecessor APIs."""

    def test_predecessor_mlx_iteration_arm_canonical_baseline_preserved(self) -> None:
        from tac.substrates.nscs06_v8_chroma_lut import (
            enumerate_cargo_cult_unwind_arms,
        )

        arms = enumerate_cargo_cult_unwind_arms()
        # Predecessor's canonical baseline arm + 4 unwind arms = 5 total.
        assert len(arms) == 5
        assert arms[0].arm_label == "baseline_4bit_per_class"

    def test_sister_canonical_constants_preserved(self) -> None:
        assert GRAYSCALE_LEVELS_DEFAULT == 16
        assert NUM_SEGNET_CLASSES == 5
        assert CHROMA_LUT_BYTES_DEFAULT == 4096
        assert PROCEDURAL_SEED_SIZE_BYTES == 32

    def test_sister_predicted_delta_s_canonical_equation_26_preserved(self) -> None:
        from tac.substrates.nscs06_v8_chroma_lut.procedural_variant import (
            predicted_delta_s,
        )

        ds = predicted_delta_s()
        # Canonical equation #26 closed form: -25 * (4096 - 32) / 37_545_489 ≈ -0.002706
        assert isinstance(ds, float)
        assert -0.0028 < ds < -0.0026

    def test_path_3_c_prime_new_exports_visible_from_package(self) -> None:
        """Verify all NEW Phase 3 symbols are exported via package __init__."""
        from tac.substrates import nscs06_v8_chroma_lut as pkg

        # Cargo-cult #5 surface
        assert hasattr(pkg, "verify_per_class_chroma_anchors_consumed_at_inflate")
        assert hasattr(pkg, "PerClassChromaDistinguishingFeatureVerdict")
        # Cargo-cult #3 surface
        assert hasattr(pkg, "derive_chroma_lut_seed_from_gt_lut_bytes")
        assert hasattr(pkg, "GtDistributionMatchedSeedVerdict")
        # Cargo-cult #8 surface
        assert hasattr(pkg, "measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline")
        assert hasattr(pkg, "SegNetArgmaxDisplacementVerdict")
        # Cargo-cult #12 surface
        assert hasattr(pkg, "predicted_delta_s_with_axis_attribution")
        assert hasattr(pkg, "axis_attribution_to_dict_for_metadata_json")
        assert hasattr(pkg, "is_predicted_band_validated_post_training")
