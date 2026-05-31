# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.joint_p18_p19_waterfill import (
    FULL_VIDEO_AUTHORITY_SCOPE,
    JOINT_P18_P19_RATE_ATTACK_ROLE,
    JOINT_P18_P19_WEIGHT_FORMULA,
    JointP18P19WaterfillAuthorityError,
    JointP18P19WaterfillConfig,
    build_joint_p18_p19_waterfill_surface,
    mahalanobis_pose_jacobian_norm,
    require_full_video_joint_surface,
)


def test_joint_p18_p19_weight_uses_segnet_and_pose_mahalanobis_terms() -> None:
    cfg = JointP18P19WaterfillConfig(
        d_pose=3.4e-5,
        pose_inverse_variance=(1.0, 4.0),
        pose_null_threshold=0.05,
    )
    seg = np.array([0.01, 0.01, 0.0], dtype=np.float64)
    pose_j = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.2],
        ],
        dtype=np.float64,
    )

    surface = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=pose_j,
        config=cfg,
    )

    expected_pose_norm = np.array([0.0, 0.1, 0.4])
    np.testing.assert_allclose(
        surface["pose_mahalanobis_norm"],
        expected_pose_norm,
    )
    np.testing.assert_allclose(surface["segnet_term"], np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(
        surface["joint_weight"],
        surface["segnet_term"] + surface["pose_term"],
    )
    assert surface["formula"] == JOINT_P18_P19_WEIGHT_FORMULA
    assert surface["rate_axis_attack_role"] == JOINT_P18_P19_RATE_ATTACK_ROLE
    assert surface["pose_null_mask"].tolist() == [True, False, False]
    assert surface["rate_attack_deadzone_mask"].tolist() == [True, False, False]
    assert surface["distortion_protect_mask"].tolist() == [False, True, True]
    assert surface["safe_rate_spend_mask"].tolist() == [True, False, False]
    assert surface["full_video_authority"] is False
    assert "joint_p18_p19_surface_not_full_video_scope" in surface["full_video_authority_blockers"]
    assert surface["ready_for_budget_spend"] is False
    assert surface["score_claim"] is False
    assert surface["ready_for_exact_eval_dispatch"] is False


def test_joint_p18_p19_requires_full_video_authority_for_budget_spend() -> None:
    seg = np.array([0.01, 0.02], dtype=np.float64)
    pose_j = np.zeros((2, 1), dtype=np.float64)
    proposal = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=pose_j,
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
        ),
    )

    with pytest.raises(
        JointP18P19WaterfillAuthorityError,
        match="not_full_video_scope",
    ):
        require_full_video_joint_surface(proposal)

    authoritative = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=pose_j,
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE,
            full_video_atom_count=seg.size,
        ),
    )

    require_full_video_joint_surface(authoritative)
    assert authoritative["full_video_authority"] is True
    assert authoritative["full_video_authority_blockers"] == []
    assert authoritative["ready_for_budget_spend"] is True


def test_joint_p18_p19_rejects_partial_full_video_coverage() -> None:
    seg = np.array([0.01, 0.02], dtype=np.float64)
    pose_j = np.zeros((2, 1), dtype=np.float64)
    surface = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=pose_j,
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE,
            full_video_atom_count=4,
        ),
    )

    assert surface["full_video_authority"] is False
    assert surface["full_video_coverage_fraction"] == 0.5
    with pytest.raises(
        JointP18P19WaterfillAuthorityError,
        match="coverage_incomplete",
    ):
        require_full_video_joint_surface(surface)


def test_pose_ail_gain_increases_as_pose_distortion_approaches_zero() -> None:
    loose = JointP18P19WaterfillConfig(
        d_pose=1e-2,
        pose_inverse_variance=(1.0,),
    )
    frontier = JointP18P19WaterfillConfig(
        d_pose=3.4e-5,
        pose_inverse_variance=(1.0,),
    )

    assert frontier.pose_ail_gain > loose.pose_ail_gain


def test_mahalanobis_pose_norm_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        mahalanobis_pose_jacobian_norm(
            np.zeros((2, 3)),
            np.ones((2,)),
        )
