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


def test_source_pair_indices_must_match_local_target_rows() -> None:
    bundle = RendererBundle(
        model=object(),
        target_rgb_0=None,
        target_rgb_1=None,
        num_pairs=2,
        source_pair_indices=[417, 22],  # type: ignore[arg-type]
    )
    assert bundle.source_pair_indices == (417, 22)

    with pytest.raises(MlxScoreAwareHarnessError, match="length must equal num_pairs"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=2,
            source_pair_indices=(7,),
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="must not contain duplicates"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=2,
            source_pair_indices=(7, 7),
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="non-negative"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=1,
            source_pair_indices=(-1,),
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="integer source pair ids"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=1,
            source_pair_indices=("bad",),  # type: ignore[arg-type]
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


def test_rejects_bad_segnet_distillation_objective_and_tau() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="segnet_distillation_objective"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            segnet_distillation_objective="not_a_real_objective",
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="segnet_tau_boundary"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            segnet_tau_boundary=0.0,
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="segnet_hinge_margin"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            segnet_hinge_margin=0.0,
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


def test_rejects_bad_pose_distillation_loss_config() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="pose_distillation_loss"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            pose_distillation_loss="not_real",
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="pose_distillation_huber_delta"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            pose_distillation_loss="huber",
            pose_distillation_huber_delta=0.0,
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


def test_rejects_bad_eval_roundtrip_camera_hw() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="eval_roundtrip_camera_hw"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            eval_roundtrip_camera_hw=(0, 1164),
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="eval_roundtrip_camera_hw"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            eval_roundtrip_camera_hw=(874, 1164, 3),  # type: ignore[arg-type]
        )


def test_rejects_bad_pose_student_input_preprocess() -> None:
    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="pose_student_input_preprocess",
    ):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            pose_student_input_preprocess="not_real",
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
        assert b.segnet_distillation_objective == "kl_t2"
        assert b.segnet_tau_boundary == 1.0
        assert b.segnet_hinge_margin == 1.0
        assert b.distillation_num_classes == 5
        assert b.segnet_teacher_frame_index == 1
        assert b.pose_distillation_weight == 0.0
        assert b.pose_dims == 6
        assert b.eval_roundtrip_ste_enabled is False
        assert b.eval_roundtrip_camera_hw == (874, 1164)
        assert b.pose_student_input_preprocess == "rgb"


def test_substrate_artifact_metadata_accepts_non_authority_lineage() -> None:
    bundle = RendererBundle(
        model=object(),
        target_rgb_0=None,
        target_rgb_1=None,
        num_pairs=4,
        substrate_artifact_metadata={
            "schema": "mlx_substrate_backend_lineage.v1",
            "backend_lineage": "reference_s6_mlx",
            "backend_claim_blockers": ["canonical_ssd_mlx_backend_not_wired"],
        },
    )

    assert bundle.substrate_artifact_metadata["backend_lineage"] == "reference_s6_mlx"


def test_substrate_artifact_metadata_rejects_duplicate_authority_keys() -> None:
    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="canonical authority/readiness key",
    ):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            substrate_artifact_metadata={"ready_for_exact_eval_dispatch": False},
        )

    with pytest.raises(
        MlxScoreAwareHarnessError,
        match="canonical authority/readiness key",
    ):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            substrate_artifact_metadata={
                "nested": {"ready_for_exact_eval_dispatch": False}
            },
        )
