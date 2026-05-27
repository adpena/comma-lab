# SPDX-License-Identifier: MIT
"""Unit tests for the RendererBundle contract (no MLX arrays needed)."""
from __future__ import annotations

import pytest

from tac.substrates._shared.mlx_score_aware.bundle import (
    FORWARD_CONVENTIONS,
    RendererBundle,
)
from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
)


def test_forward_conventions_are_the_canonical_pair() -> None:
    assert {"reconstruct_pair_nchw01", "call_b2chw_255"} == FORWARD_CONVENTIONS


def test_rejects_bad_convention() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="forward_convention"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            forward_convention="not_real",
        )


def test_rejects_zero_pairs() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="num_pairs"):
        RendererBundle(
            model=object(), target_rgb_0=None, target_rgb_1=None, num_pairs=0
        )


def test_rejects_negative_distillation_weight() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="distillation_weight"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            distillation_weight=-0.1,
        )


def test_rejects_nonpositive_temperature() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="distillation_temperature"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            distillation_temperature=0.0,
        )


def test_rejects_bad_num_classes() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="distillation_num_classes"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            distillation_num_classes=0,
        )


def test_rejects_bad_segnet_teacher_frame_index() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="segnet_teacher_frame_index"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            segnet_teacher_frame_index=2,
        )


def test_rejects_negative_pose_distillation_weight() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="pose_distillation_weight"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            pose_distillation_weight=-0.1,
        )


def test_rejects_bad_pose_dims() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="pose_dims"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            pose_dims=0,
        )


def test_rejects_pose_distill_without_pose_teacher() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="pose_scorer_teacher"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            pose_distillation_weight=0.5,
        )


def test_rejects_pose_teacher_without_pose_head() -> None:
    class _PoseTeacher:
        pose_dims = 6

        def teacher_pose_for_indices(self, idx):
            return idx

    with pytest.raises(MlxScoreAwareHarnessError, match="learnable_pose_student_head"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            pose_distillation_weight=0.5,
            pose_scorer_teacher=_PoseTeacher(),
        )


def test_rejects_real_segnet_binding_without_pose_unless_research_opted_in() -> None:
    class _SegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, idx):
            return idx

    class _SegHead:
        pass

    with pytest.raises(MlxScoreAwareHarnessError, match="binds the REAL SegNet"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            distillation_weight=0.5,
            scorer_teacher=_SegTeacher(),
            learnable_student_head=_SegHead(),
        )

    bundle = RendererBundle(
        model=object(),
        target_rgb_0=None,
        target_rgb_1=None,
        num_pairs=4,
        distillation_weight=0.5,
        scorer_teacher=_SegTeacher(),
        learnable_student_head=_SegHead(),
        allow_segnet_only_research=True,
    )
    assert bundle.allow_segnet_only_research is True


def test_accepts_canonical_conventions_with_defaults() -> None:
    for conv in ("reconstruct_pair_nchw01", "call_b2chw_255"):
        b = RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            forward_convention=conv,
        )
        assert b.forward_convention == conv
        assert b.distillation_weight == 0.0
        assert b.distillation_temperature == 2.0
        assert b.distillation_num_classes == 5
        assert b.segnet_teacher_frame_index == 1
        assert b.pose_distillation_weight == 0.0
        assert b.pose_dims == 6
