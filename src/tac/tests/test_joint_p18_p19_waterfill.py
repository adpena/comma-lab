# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.joint_p18_p19_waterfill import (
    CONTEST_VIDEO_OVERFIT_MODE,
    FULL_VIDEO_AUTHORITY_SCOPE,
    FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
    JOINT_P18_P19_IMPLICIT_ALLOCATOR_SCHEMA,
    JOINT_P18_P19_RATE_ATTACK_ROLE,
    JOINT_P18_P19_WEIGHT_FORMULA,
    JOINT_P18_SEGNET_CLASS_BOUNDARY_SCHEMA,
    JOINT_P18_SEGNET_DEEPFOOL_MARGIN_SCHEMA,
    JointP18P19WaterfillAuthorityError,
    JointP18P19WaterfillConfig,
    build_joint_p18_p19_waterfill_surface,
    build_segnet_deepfool_margin_class_region_weight,
    contest_fixed_rate_price_per_archive_byte,
    mahalanobis_pose_jacobian_norm,
    require_exact_full_video_gradient_reduction,
    require_fresh_joint_surface,
    require_full_video_joint_surface,
    solve_joint_p18_p19_implicit_kkt_dykstra_allocator,
    surface_is_stale_for_archive,
)
from tac.optimization.target_modes import CORPUS_GENERALIZATION_MODE

ARCHIVE_A_SHA = "a" * 64
ARCHIVE_B_SHA = "b" * 64
ARCHIVE_D_SHA = "d" * 64


def _full_video_surface(
    *,
    linearization_archive_sha: str | None = None,
) -> dict:
    """Helper: a full-video-COVERAGE-authoritative 2-atom surface."""

    if linearization_archive_sha is None:
        sha = None
    else:
        aliases = {"archiveA": ARCHIVE_A_SHA, "archiveB": ARCHIVE_B_SHA}
        sha = aliases.get(linearization_archive_sha, linearization_archive_sha)
    seg = np.array([0.01, 0.02], dtype=np.float64)
    pose_j = np.zeros((2, 1), dtype=np.float64)
    return build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=pose_j,
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE,
            full_video_atom_count=seg.size,
            linearization_archive_sha=sha,
            gradient_reduction_semantics=FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        ),
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
    assert surface["contest_rate_price_per_archive_byte"] == pytest.approx(
        25.0 / 37_545_489
    )
    assert surface["target_mode"] == CONTEST_VIDEO_OVERFIT_MODE
    assert surface["declared_overfit_allowed"] is True
    assert surface["corpus_manifest_required"] is False
    assert surface["pose_null_mask"].tolist() == [True, False, False]
    assert surface["rate_attack_deadzone_mask"].tolist() == [True, False, False]
    assert surface["distortion_protect_mask"].tolist() == [False, True, True]
    assert surface["safe_rate_spend_mask"].tolist() == [True, False, False]
    assert surface["rate_attack_allocation"][0] > 0.0
    assert surface["implicit_kkt_dykstra_allocator"]["schema"] == JOINT_P18_P19_IMPLICIT_ALLOCATOR_SCHEMA
    assert surface["implicit_kkt_dykstra_allocator"]["solver_converged"] is True
    assert surface["implicit_allocator_authority"] is False
    assert "joint_p18_p19_allocator_budget_spend_authority_missing" in surface["implicit_allocator_blockers"]
    assert surface["full_video_authority"] is False
    assert "joint_p18_p19_surface_not_full_video_scope" in surface["full_video_authority_blockers"]
    assert surface["gradient_reduction_authority"] is False
    assert (
        "joint_p18_p19_gradient_reduction_not_exact_full_video_accumulation" in surface["gradient_reduction_blockers"]
    )
    assert surface["optimizer_update_semantics"] == "no_update_probe_only"
    assert surface["ready_for_budget_spend"] is False
    assert surface["score_claim"] is False
    assert surface["ready_for_exact_eval_dispatch"] is False


def test_segnet_deepfool_margin_weight_prioritizes_low_margin_high_gradient_atoms() -> None:
    margin = np.array([0.01, 1.0, 0.25, 0.02], dtype=np.float64)
    grad_sq = np.array([1.0, 1.0, 0.0, 0.25], dtype=np.float64)

    report = build_segnet_deepfool_margin_class_region_weight(
        top2_margin=margin,
        margin_gradient_norm_sq=grad_sq,
        boundary_margin_quantile=0.25,
    )

    weight = report["class_region_weight"]
    assert report["schema"] == JOINT_P18_SEGNET_DEEPFOOL_MARGIN_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["blockers"] == []
    assert weight[0] > weight[3] > weight[1] > weight[2]
    assert report["boundary_protect_mask"].tolist() == [True, False, False, False]


def test_segnet_deepfool_margin_weight_feeds_existing_allocator_without_static_mask() -> None:
    margin_report = build_segnet_deepfool_margin_class_region_weight(
        top2_margin=np.array([0.02, 1.0, 0.5], dtype=np.float64),
        margin_gradient_norm_sq=np.array([1.0, 1.0, 0.0], dtype=np.float64),
    )
    seg = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    surface = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=np.zeros((3, 1), dtype=np.float64),
        segnet_class_region_weight=margin_report["class_region_weight"],
        segnet_boundary_protect_mask=margin_report["boundary_protect_mask"],
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE,
            full_video_atom_count=seg.size,
            linearization_archive_sha=ARCHIVE_A_SHA,
            gradient_reduction_semantics=FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        ),
    )

    assert surface["segnet_class_boundary_report"]["class_region_weight_applied"] is True
    assert surface["segnet_class_boundary_report"]["boundary_protect_mask_applied"] is True
    assert surface["ready_for_budget_spend"] is True
    assert surface["safe_rate_spend_mask"][0].item() is False


def test_fixed_contest_rate_price_is_source_grounded_water_level() -> None:
    assert contest_fixed_rate_price_per_archive_byte() == pytest.approx(25.0 / 37_545_489)


def test_segnet_deepfool_margin_weight_rejects_fake_negative_margin() -> None:
    with pytest.raises(ValueError, match="top2_margin"):
        build_segnet_deepfool_margin_class_region_weight(
            top2_margin=np.array([0.1, -0.1], dtype=np.float64),
            margin_gradient_norm_sq=np.ones((2,), dtype=np.float64),
        )


def test_joint_p18_p19_class_boundary_modifier_blocks_boundary_coarsening() -> None:
    seg = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    pose_j = np.zeros((4, 1), dtype=np.float64)
    class_weight = np.array([1.0, 4.0, 1.0, 1.0], dtype=np.float64)
    boundary_mask = np.array([False, True, False, False])
    class_ids = np.array([0, 1, 0, 2], dtype=np.int64)

    surface = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=pose_j,
        segnet_class_region_weight=class_weight,
        segnet_boundary_protect_mask=boundary_mask,
        segnet_class_ids=class_ids,
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE,
            full_video_atom_count=seg.size,
            linearization_archive_sha=ARCHIVE_A_SHA,
            gradient_reduction_semantics=FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        ),
    )

    report = surface["segnet_class_boundary_report"]
    assert report["schema"] == JOINT_P18_SEGNET_CLASS_BOUNDARY_SCHEMA
    assert report["class_region_weight_applied"] is True
    assert report["boundary_protect_mask_applied"] is True
    assert report["boundary_protect_atom_count"] == 1
    assert report["safe_rate_spend_atom_count"] == 3
    assert report["class_histogram"] == {"0": 2, "1": 1, "2": 1}
    assert surface["safe_rate_spend_mask"].tolist() == [True, False, True, True]
    assert surface["rate_attack_deadzone_mask"][1].item() is False
    assert surface["distortion_protect_mask"][1].item() is True
    np.testing.assert_allclose(surface["segnet_term"], 100.0 * seg * class_weight)
    assert surface["ready_for_budget_spend"] is True
    assert surface["implicit_allocator_authority"] is True


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
    assert authoritative["ready_for_budget_spend"] is False
    assert "joint_p18_p19_linearization_archive_sha_missing" in authoritative["fresh_surface_blockers"]
    assert authoritative["gradient_reduction_authority"] is False


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


def test_surface_carries_local_tangent_plane_markers_and_pinned_sha() -> None:
    sha = ARCHIVE_D_SHA
    surface = _full_video_surface(linearization_archive_sha=sha)
    assert surface["surface_is_local_tangent_plane"] is True
    assert surface["recompute_required_after_archive_mutation"] is True
    assert surface["linearization_archive_sha"] == sha
    assert surface["gradient_reduction_semantics"] == FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION
    assert surface["pair_chunk_updates_forbidden"] is True
    assert surface["optimizer_update_semantics"] == "single_update_after_all_pair_shards_reduce"


def test_surface_is_stale_for_archive_detects_accepted_mutation() -> None:
    archive_a = ARCHIVE_A_SHA
    archive_b = ARCHIVE_B_SHA
    surface = _full_video_surface(linearization_archive_sha=archive_a)
    # The decision was about to mutate the archive it was linearized at -> fresh.
    assert surface_is_stale_for_archive(surface, archive_a) is False
    # After an accepted mutation produces a new archive sha -> stale tangent plane.
    assert surface_is_stale_for_archive(surface, archive_b) is True


def test_surface_is_stale_for_archive_unpinned_is_stale() -> None:
    surface = _full_video_surface(linearization_archive_sha=None)
    # No pinned linearization point means the tangent plane cannot prove freshness.
    assert surface_is_stale_for_archive(surface, ARCHIVE_A_SHA) is True


def test_require_fresh_joint_surface_refuses_stale_tangent_plane() -> None:
    archive_a = ARCHIVE_A_SHA
    archive_b = ARCHIVE_B_SHA
    surface = _full_video_surface(linearization_archive_sha=archive_a)
    require_fresh_joint_surface(surface, current_archive_sha=archive_a)  # fresh: ok
    with pytest.raises(
        JointP18P19WaterfillAuthorityError,
        match="stale_tangent_plane",
    ):
        require_fresh_joint_surface(surface, current_archive_sha=archive_b)


def test_require_fresh_joint_surface_refuses_unpinned_surface() -> None:
    surface = _full_video_surface(linearization_archive_sha=None)
    with pytest.raises(
        JointP18P19WaterfillAuthorityError,
        match="linearization_archive_sha_missing",
    ):
        require_fresh_joint_surface(surface, current_archive_sha=ARCHIVE_A_SHA)


def test_config_rejects_blank_linearization_sha() -> None:
    with pytest.raises(ValueError, match="linearization_archive_sha"):
        JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            linearization_archive_sha="   ",
        )


def test_joint_surface_target_mode_toggle_records_production_contract() -> None:
    seg = np.array([0.01, 0.02], dtype=np.float64)
    surface = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=np.zeros((2, 1), dtype=np.float64),
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            target_mode=CORPUS_GENERALIZATION_MODE,
            evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE,
            full_video_atom_count=seg.size,
            linearization_archive_sha=ARCHIVE_A_SHA,
            gradient_reduction_semantics=FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        ),
    )

    assert surface["target_mode"] == CORPUS_GENERALIZATION_MODE
    assert surface["declared_overfit_allowed"] is False
    assert surface["corpus_manifest_required"] is True
    assert surface["ready_for_budget_spend"] is True
    assert surface["objective_mode_contract"]["no_hidden_video_or_corpus_binding"] is True


def test_joint_surface_requires_exact_full_video_reduction_for_budget_spend() -> None:
    seg = np.array([0.01, 0.02], dtype=np.float64)
    sampled = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=np.zeros((2, 1), dtype=np.float64),
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE,
            full_video_atom_count=seg.size,
            linearization_archive_sha=ARCHIVE_A_SHA,
        ),
    )

    require_full_video_joint_surface(sampled)
    require_fresh_joint_surface(sampled, current_archive_sha=ARCHIVE_A_SHA)
    assert sampled["ready_for_budget_spend"] is False
    with pytest.raises(JointP18P19WaterfillAuthorityError, match="not_exact_full_video_accumulation"):
        require_exact_full_video_gradient_reduction(sampled)

    exact = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=np.zeros((2, 1), dtype=np.float64),
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            evidence_scope=FULL_VIDEO_AUTHORITY_SCOPE,
            full_video_atom_count=seg.size,
            linearization_archive_sha=ARCHIVE_A_SHA,
            gradient_reduction_semantics=FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        ),
    )

    require_exact_full_video_gradient_reduction(exact)
    assert exact["ready_for_budget_spend"] is True
    assert exact["implicit_allocator_authority"] is True


def test_implicit_kkt_dykstra_allocator_solves_box_budget_projection() -> None:
    priority = np.array([3.0, 2.0, 1.0, 9.0], dtype=np.float64)
    safe = np.array([True, True, True, False])

    result = solve_joint_p18_p19_implicit_kkt_dykstra_allocator(
        coarsening_priority=priority,
        safe_rate_spend_mask=safe,
        budget=1.5,
        atom_capacity=1.0,
    )

    assert result["schema"] == JOINT_P18_P19_IMPLICIT_ALLOCATOR_SCHEMA
    assert result["solver_converged"] is True
    assert result["budget_binding"] is True
    np.testing.assert_allclose(result["allocation"], np.array([1.0, 0.5, 0.0, 0.0]), atol=1e-7)
    assert float(np.sum(result["allocation"])) <= result["effective_budget"] + 1e-9
    assert result["allocation"][3] == 0.0
    assert result["kkt_residuals"]["dykstra_projection_linf"] <= 1e-7
    assert result["implicit_budget_jacobian"][1] == pytest.approx(1.0)


def test_implicit_kkt_dykstra_allocator_rejects_nonmatching_mask() -> None:
    with pytest.raises(ValueError, match="shape"):
        solve_joint_p18_p19_implicit_kkt_dykstra_allocator(
            coarsening_priority=np.zeros((2,)),
            safe_rate_spend_mask=np.zeros((3,), dtype=bool),
            budget=1.0,
        )


def test_joint_surface_rejects_unknown_target_mode() -> None:
    with pytest.raises(ValueError, match="target_mode"):
        JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            target_mode="secret_single_video_hack",
        )


def test_coverage_and_freshness_are_orthogonal_and_both_required() -> None:
    """Full-video COVERAGE (codex) and FRESH tangent plane (fresh-surface) are
    orthogonal budget-spend gates: spend is authoritative only when BOTH pass.

    This is the headline NO-FAKE guard for the full-video joint-backprop
    authority: a surface can pass one gate and still be refused by the other.
    """

    # (1) Full-video coverage authoritative BUT stale: coverage gate passes,
    #     freshness gate REFUSES (linearized at archiveA, deciding on archiveB).
    archive_a = ARCHIVE_A_SHA
    archive_b = ARCHIVE_B_SHA
    stale_full = _full_video_surface(linearization_archive_sha=archive_a)
    require_full_video_joint_surface(stale_full)  # coverage: ok
    with pytest.raises(JointP18P19WaterfillAuthorityError, match="stale_tangent_plane"):
        require_fresh_joint_surface(stale_full, current_archive_sha=archive_b)

    # (2) Fresh tangent plane BUT crop/window-only coverage: freshness gate
    #     passes, coverage gate REFUSES (proposal_only scope, no full coverage).
    seg = np.array([0.01, 0.02], dtype=np.float64)
    crop_fresh = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg,
        pose_jacobian=np.zeros((2, 1), dtype=np.float64),
        config=JointP18P19WaterfillConfig(
            d_pose=1e-4,
            pose_inverse_variance=(1.0,),
            linearization_archive_sha=archive_a,
            gradient_reduction_semantics=FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        ),
    )
    require_fresh_joint_surface(crop_fresh, current_archive_sha=archive_a)  # fresh: ok
    with pytest.raises(JointP18P19WaterfillAuthorityError, match="not_full_video_scope"):
        require_full_video_joint_surface(crop_fresh)

    # (3) Full-video coverage AND fresh: BOTH gates pass -> budget-spend ready.
    spendable = _full_video_surface(linearization_archive_sha=archive_a)
    require_full_video_joint_surface(spendable)
    require_fresh_joint_surface(spendable, current_archive_sha=archive_a)
    assert spendable["ready_for_budget_spend"] is True
