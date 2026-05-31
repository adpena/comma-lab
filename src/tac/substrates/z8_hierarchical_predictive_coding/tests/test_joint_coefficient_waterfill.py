# SPDX-License-Identifier: MIT
"""Tests for executable Z8 joint P18/P19 coefficient water-fill materializer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_archive
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
    parse_pair_blobs_from_wavelet_blob,
)
from tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill import (
    FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
    SINGLE_UPDATE_AFTER_FULL_REDUCTION,
    Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE,
    Z8_JOINT_COEFFICIENT_RELINEARIZED_SEARCH_SCHEMA,
    Z8_JOINT_COEFFICIENT_VARIANT_MANIFEST_SCHEMA,
    Z8JointCoefficientRelinearizationSearchConfig,
    Z8JointCoefficientWaterfillConfig,
    apply_joint_p18_p19_deadzone_to_z8_archive,
    load_joint_p18_p19_surface_file,
    materialize_joint_p18_p19_deadzone_candidate,
    materialize_joint_p18_p19_relinearized_deadzone_search,
    run_joint_p18_p19_relinearized_deadzone_search,
)
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    Z8HierarchicalConfig,
)


def _cfg(*, num_pairs: int = 1) -> Z8HierarchicalConfig:
    return Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=num_pairs,
        deterministic_state_dim=16,
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(16, 16),
    )


def _archive_bytes(*, num_pairs: int = 1) -> bytes:
    rng = np.random.RandomState(19)
    f0 = rng.uniform(0, 1, size=(num_pairs, 16, 16, 3)).astype(np.float32)
    f1 = rng.uniform(0, 1, size=(num_pairs, 16, 16, 3)).astype(np.float32)
    binding = build_canonical_quadruple_binding_from_z8_config(_cfg(num_pairs=num_pairs))
    return build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)


def _surface_for_archive(
    archive_bytes: bytes,
    joint_weight: np.ndarray,
    pose_null_mask: np.ndarray | None,
) -> dict[str, Any]:
    return {
        "joint_weight": joint_weight,
        "rate_attack_deadzone_mask": pose_null_mask,
        "linearization_archive_sha": hashlib.sha256(archive_bytes).hexdigest(),
        "evidence_scope": "full_video",
        "target_mode": "contest_video_overfit",
        "gradient_reduction_semantics": FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        "gradient_reduction_authority": True,
        "optimizer_update_authority": True,
        "optimizer_update_semantics": SINGLE_UPDATE_AFTER_FULL_REDUCTION,
        "full_video_reduction_complete": True,
        "budget_spend_authority": True,
    }


def test_joint_p18_p19_deadzone_mutates_wavelet_details_and_reduces_rate() -> None:
    archive_bytes = _archive_bytes()
    original = parse_archive(archive_bytes)
    original_pyramids = parse_pair_blobs_from_wavelet_blob(original.wavelet_coeffs_blob)
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    result = apply_joint_p18_p19_deadzone_to_z8_archive(
        archive_bytes,
        joint_weight=_surface_for_archive(archive_bytes, joint_weight, pose_null_mask),
        config=Z8JointCoefficientWaterfillConfig(
            joint_weight_quantile=1.0,
            coefficient_deadzone_quantile=1.0,
            quantization_step=0.25,
        ),
    )

    mutated_archive = result["mutated_archive_bytes"]
    mutated = parse_archive(mutated_archive)
    mutated_pyramids = parse_pair_blobs_from_wavelet_blob(mutated.wavelet_coeffs_blob)
    assert result["role"] == Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE
    assert result["coefficient_report"]["dead_zoned_coefficients"] > 0
    assert result["rate_report"]["after_wavelet_blob_bytes"] < result["rate_report"]["before_wavelet_blob_bytes"]
    assert result["distortion_report"]["small_receiver_distortion_measured"] is True
    assert result["distortion_report"]["max_abs_delta"] > 0.0
    assert result["surface_freshness_report"]["fresh_for_current_archive"] is True
    assert result["surface_gradient_reduction_report"]["exact_full_video_gradient_reduction"] is True
    np.testing.assert_allclose(
        mutated_pyramids[0]["frame_0_top_ll"],
        original_pyramids[0]["frame_0_top_ll"],
    )
    assert not np.allclose(
        mutated_pyramids[0]["frame_0_details"][0].lh,
        original_pyramids[0]["frame_0_details"][0].lh,
    )
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_joint_p18_p19_deadzone_materializer_emits_byte_closed_archive(
    tmp_path: Path,
) -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    manifest = materialize_joint_p18_p19_deadzone_candidate(
        archive_bytes,
        tmp_path,
        joint_weight=_surface_for_archive(archive_bytes, joint_weight, pose_null_mask),
        config=Z8JointCoefficientWaterfillConfig(
            joint_weight_quantile=1.0,
            coefficient_deadzone_quantile=1.0,
            quantization_step=0.25,
            emit_archive_zip=True,
            emit_receiver_proof=False,
        ),
    )

    assert manifest["schema"] == Z8_JOINT_COEFFICIENT_VARIANT_MANIFEST_SCHEMA
    assert Path(manifest["candidate_bin_path"]).is_file()
    assert Path(manifest["archive_zip_path"]).is_file()
    assert Path(manifest["manifest_path"]).is_file()
    assert manifest["receiver_proof_executed"] is False
    assert manifest["exact_axis_blocker"] == ("receiver_proof_and_contest_cpu_cuda_eval_not_executed")
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_joint_p18_p19_deadzone_rejects_pair_broadcast_surface_for_full_video() -> None:
    archive_bytes = _archive_bytes(num_pairs=2)
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    with pytest.raises(ValueError, match="full_archive_pair_grid"):
        apply_joint_p18_p19_deadzone_to_z8_archive(
            archive_bytes,
            joint_weight=_surface_for_archive(archive_bytes, joint_weight, pose_null_mask),
            config=Z8JointCoefficientWaterfillConfig(
                joint_weight_quantile=1.0,
                coefficient_deadzone_quantile=1.0,
                quantization_step=0.25,
                require_full_video_surface_coverage=True,
            ),
        )


def test_relinearized_search_requires_fresh_surfaces() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    try:
        run_joint_p18_p19_relinearized_deadzone_search(
            archive_bytes,
            surfaces=[
                _surface_for_archive(archive_bytes, joint_weight, pose_null_mask),
                _surface_for_archive(archive_bytes, joint_weight.copy(), pose_null_mask.copy()),
            ],
            config=Z8JointCoefficientRelinearizationSearchConfig(
                joint_weight_quantiles=(1.0,),
                coefficient_deadzone_quantiles=(1.0,),
                quantization_steps=(0.25,),
                max_iterations=2,
                require_fresh_surface_per_iteration=True,
            ),
        )
    except ValueError as exc:
        assert "fresh surface required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate iterative surface was accepted")


def test_relinearized_search_accepts_fresh_surface_and_writes_final_candidate(
    tmp_path: Path,
) -> None:
    archive_bytes = _archive_bytes()
    joint_weight_0 = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    joint_weight_1 = np.ones((1, 2, 16, 16, 3), dtype=np.float32) * 0.1
    joint_weight_1[..., :8, :8, :] = 0.0
    pose_null_mask = np.ones_like(joint_weight_0, dtype=bool)

    def surface_provider(_iteration_index: int, current_archive: bytes) -> dict[str, Any]:
        current_sha = hashlib.sha256(current_archive).hexdigest()
        joint_weight = joint_weight_0 if current_sha == hashlib.sha256(archive_bytes).hexdigest() else joint_weight_1
        return _surface_for_archive(current_archive, joint_weight, pose_null_mask)

    manifest = materialize_joint_p18_p19_relinearized_deadzone_search(
        archive_bytes,
        tmp_path,
        surface_provider=surface_provider,
        config=Z8JointCoefficientRelinearizationSearchConfig(
            joint_weight_quantiles=(1.0,),
            coefficient_deadzone_quantiles=(0.5, 1.0),
            quantization_steps=(0.25,),
            max_iterations=2,
            max_cumulative_mse=1.0,
            emit_archive_zip=True,
            emit_receiver_proof=False,
        ),
    )

    assert manifest["schema"] == Z8_JOINT_COEFFICIENT_RELINEARIZED_SEARCH_SCHEMA
    assert manifest["iterations_accepted"] == 2
    assert manifest["candidate_count"] == 4
    assert all(row["surface_freshness_report"]["fresh_for_current_archive"] for row in manifest["accepted_candidates"])
    assert all(
        row["surface_gradient_reduction_report"]["exact_full_video_gradient_reduction"]
        for row in manifest["accepted_candidates"]
    )
    assert Path(manifest["candidate_bin_path"]).is_file()
    assert Path(manifest["archive_zip_path"]).is_file()
    assert Path(manifest["manifest_path"]).is_file()
    assert (
        manifest["cumulative_rate_report"]["final_archive_bytes"]
        <= manifest["cumulative_rate_report"]["original_archive_bytes"]
    )
    assert manifest["final_distortion_report"]["small_receiver_distortion_measured"] is True
    assert manifest["receiver_proof_executed"] is False
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_relinearized_search_uses_full_video_local_replay_for_accept_reject() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)
    replay_calls: list[dict[str, Any]] = []

    def local_replay_evaluator(**kwargs: Any) -> dict[str, Any]:
        meta = dict(kwargs["candidate_metadata"])
        replay_calls.append(meta)
        # Force the search to prefer the less aggressive deadzone even when the
        # receiver-MSE/rate proxy would otherwise be free to choose either.
        objective = 1.0 + float(meta["coefficient_deadzone_quantile"])
        return {
            "schema": "unit_full_video_local_replay.v1",
            "full_video_local_replay_executed": True,
            "full_video_local_replay_scope": "full_video",
            "replay_ok": True,
            "contest_action_proxy": objective,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }

    result = run_joint_p18_p19_relinearized_deadzone_search(
        archive_bytes,
        surfaces=[_surface_for_archive(archive_bytes, joint_weight, pose_null_mask)],
        local_replay_evaluator=local_replay_evaluator,
        config=Z8JointCoefficientRelinearizationSearchConfig(
            joint_weight_quantiles=(1.0,),
            coefficient_deadzone_quantiles=(0.25, 1.0),
            quantization_steps=(0.25,),
            max_iterations=1,
            max_cumulative_mse=1.0,
        ),
    )

    assert len(replay_calls) == 2
    assert result["iterations_accepted"] == 1
    accepted = result["accepted_candidates"][0]
    assert accepted["objective_source"] == "full_video_local_replay"
    assert accepted["full_video_local_replay_required"] is True
    assert accepted["hard_archive_projection_replay_executed"] is True
    assert accepted["coefficient_deadzone_quantile"] == 0.25
    assert accepted["full_video_local_replay_report"]["contest_action_proxy"] == 1.25
    assert accepted["score_claim"] is False


def test_joint_p18_p19_deadzone_rejects_stale_surface() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    with pytest.raises(ValueError, match="stale_tangent_plane"):
        apply_joint_p18_p19_deadzone_to_z8_archive(
            archive_bytes,
            joint_weight={
                "joint_weight": joint_weight,
                "rate_attack_deadzone_mask": pose_null_mask,
                "linearization_archive_sha": "b" * 64,
                "evidence_scope": "full_video",
            },
            config=Z8JointCoefficientWaterfillConfig(
                joint_weight_quantile=1.0,
                coefficient_deadzone_quantile=1.0,
                quantization_step=0.25,
            ),
        )


def test_joint_p18_p19_deadzone_rejects_non_exact_reduced_surface() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)
    surface = _surface_for_archive(archive_bytes, joint_weight, pose_null_mask)
    surface["gradient_reduction_semantics"] = "minibatch_probe"
    surface["gradient_reduction_authority"] = False
    surface["optimizer_update_authority"] = False
    surface["optimizer_update_semantics"] = "optimizer_update_per_pair_chunk"
    surface["full_video_reduction_complete"] = False
    surface["budget_spend_authority"] = False

    with pytest.raises(ValueError, match="gradient_reduction_not_exact_full_video_accumulation"):
        apply_joint_p18_p19_deadzone_to_z8_archive(
            archive_bytes,
            joint_weight=surface,
            config=Z8JointCoefficientWaterfillConfig(
                joint_weight_quantile=1.0,
                coefficient_deadzone_quantile=1.0,
                quantization_step=0.25,
            ),
        )


def test_joint_p18_p19_npz_surface_loader_preserves_archive_freshness(
    tmp_path: Path,
) -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)
    surface = _surface_for_archive(archive_bytes, joint_weight, pose_null_mask)
    surface_path = tmp_path / "surface.npz"
    np.savez_compressed(
        surface_path,
        joint_weight=joint_weight,
        rate_attack_deadzone_mask=pose_null_mask,
        linearization_archive_sha=np.asarray(surface["linearization_archive_sha"]),
        evidence_scope=np.asarray(surface["evidence_scope"]),
        target_mode=np.asarray(surface["target_mode"]),
        gradient_reduction_semantics=np.asarray(surface["gradient_reduction_semantics"]),
        gradient_reduction_authority=np.asarray(surface["gradient_reduction_authority"]),
        optimizer_update_authority=np.asarray(surface["optimizer_update_authority"]),
        optimizer_update_semantics=np.asarray(surface["optimizer_update_semantics"]),
        full_video_reduction_complete=np.asarray(surface["full_video_reduction_complete"]),
        budget_spend_authority=np.asarray(surface["budget_spend_authority"]),
    )

    loaded = load_joint_p18_p19_surface_file(surface_path)
    result = apply_joint_p18_p19_deadzone_to_z8_archive(
        archive_bytes,
        joint_weight=loaded,
        config=Z8JointCoefficientWaterfillConfig(
            joint_weight_quantile=1.0,
            coefficient_deadzone_quantile=1.0,
            quantization_step=0.25,
        ),
    )

    assert result["surface_freshness_report"]["fresh_for_current_archive"] is True
    assert result["surface_gradient_reduction_report"]["exact_full_video_gradient_reduction"] is True
