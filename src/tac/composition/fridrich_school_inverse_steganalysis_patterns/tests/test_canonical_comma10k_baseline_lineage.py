# SPDX-License-Identifier: MIT
"""Tests for canonical comma10k-baseline lineage."""

from __future__ import annotations

import pytest

from tac.composition.fridrich_school_inverse_steganalysis_patterns import (
    COMMA10K_BASELINE_BACKBONE,
    CONTEST_NATIVE_RESOLUTION,
    CONTEST_SEGNET_BACKBONE,
    Comma10kBaselineStage,
    Comma10kCurriculumConfig,
    Comma10kLineageError,
    Comma10kTrainingStrategy,
    compute_resolution_for_stage,
)


def test_contest_native_resolution_canonical_874x1164() -> None:
    """Per Catalog #367 + CLAUDE.md: contest native = (874, 1164)."""
    assert CONTEST_NATIVE_RESOLUTION == (874, 1164)
    # Verify the canonical raw byte count: 1164 * 874 * 1200 * 3 = 3,662,409,600
    h, w = CONTEST_NATIVE_RESOLUTION
    expected_raw_bytes = w * h * 1200 * 3
    assert expected_raw_bytes == 3_662_409_600


def test_comma10k_baseline_backbone_b4() -> None:
    """Yousfi's baseline uses EfficientNet-B4 backbone."""
    assert COMMA10K_BASELINE_BACKBONE == "tu-efficientnet_b4"


def test_contest_segnet_backbone_b2() -> None:
    """Contest SegNet uses B2 (downgrade from baseline B4 per CLAUDE.md)."""
    assert CONTEST_SEGNET_BACKBONE == "tu-efficientnet_b2"
    # B2 is smaller + faster than B4 (canonical engineering tradeoff).
    assert COMMA10K_BASELINE_BACKBONE != CONTEST_SEGNET_BACKBONE


def test_compute_resolution_for_stage_1_canonical() -> None:
    """Per Yousfi README verbatim: Stage 1 = 437x582 100 epochs."""
    h, w = compute_resolution_for_stage(Comma10kBaselineStage.STAGE_1_437x582_100_EPOCHS)
    assert (h, w) == (437, 582)


def test_compute_resolution_for_stage_2_canonical() -> None:
    """Per Yousfi README verbatim: Stage 2 = 874x1164 30 epochs."""
    h, w = compute_resolution_for_stage(Comma10kBaselineStage.STAGE_2_874x1164_30_EPOCHS)
    assert (h, w) == (874, 1164)
    # Stage 2 = contest native resolution.
    assert (h, w) == CONTEST_NATIVE_RESOLUTION


def test_stage_resolutions_substantively_distinct() -> None:
    """Slot EEE: Stage 1 and Stage 2 have substantively different resolution."""
    s1 = compute_resolution_for_stage(Comma10kBaselineStage.STAGE_1_437x582_100_EPOCHS)
    s2 = compute_resolution_for_stage(Comma10kBaselineStage.STAGE_2_874x1164_30_EPOCHS)
    assert s1 != s2
    # Stage 2 is exactly 2x Stage 1 (canonical resolution doubling).
    assert s2[0] == 2 * s1[0]
    assert s2[1] == 2 * s1[1]


def test_curriculum_strategies_substantively_distinct() -> None:
    """4 canonical curriculum strategies; each is a structurally distinct training approach."""
    expected = {
        "canonical_2_stage_yousfi",
        "single_stage_full_res_baseline",
        "full_res_only_fine_tune_from_imagenet",
        "comma_pair_latent_adaptation",
    }
    actual = {s.value for s in Comma10kTrainingStrategy}
    assert actual == expected


def test_curriculum_config_default_canonical_2_stage() -> None:
    """Default strategy is Yousfi's canonical 2-stage curriculum."""
    cfg = Comma10kCurriculumConfig()
    assert cfg.strategy == Comma10kTrainingStrategy.CANONICAL_2_STAGE_YOUSFI
    assert cfg.backbone == CONTEST_SEGNET_BACKBONE
    assert cfg.target_resolution == CONTEST_NATIVE_RESOLUTION


def test_curriculum_config_invalid_strategy_raises() -> None:
    """Wrong type for strategy raises."""
    with pytest.raises(Comma10kLineageError, match="must be Comma10kTrainingStrategy"):
        Comma10kCurriculumConfig(strategy="bogus")  # type: ignore[arg-type]


def test_curriculum_config_invalid_backbone_raises() -> None:
    """Empty backbone raises."""
    with pytest.raises(Comma10kLineageError, match="backbone empty"):
        Comma10kCurriculumConfig(backbone="")


def test_curriculum_config_invalid_resolution_raises() -> None:
    """Non-positive resolution dims raise."""
    with pytest.raises(Comma10kLineageError, match="target_resolution"):
        Comma10kCurriculumConfig(target_resolution=(0, 100))


def test_compute_resolution_unhandled_stage_raises() -> None:
    """Unknown stage raises Comma10kLineageError."""
    with pytest.raises((Comma10kLineageError, AttributeError)):
        compute_resolution_for_stage("bogus")  # type: ignore[arg-type]
