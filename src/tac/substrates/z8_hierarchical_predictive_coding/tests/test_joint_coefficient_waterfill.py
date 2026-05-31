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
    _DETAIL_CODEC_QI16_CONSTRICTION_RANGE,
    _decode_quantized_detail_payload,
    _encode_qi16_constriction_range,
    build_canonical_quadruple_binding_from_z8_config,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
    pack_pair_pyramids_to_wavelet_blob,
    parse_pair_blobs_from_wavelet_blob,
    summarize_wavelet_blob_detail_codecs,
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
        "pose_surface_kind": "per_axis_posenet_jacobian_mahalanobis_v1",
        "pose_surface_authority": True,
        "pose_axis_count": 6,
        "pose_inverse_variance": [1.0] * 6,
        "pose_surface_blockers": [],
        "segnet_class_boundary_authority": True,
        "segnet_class_boundary_blockers": [],
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
    assert result["surface_true_p19_report"]["true_p19_pose_surface"] is True
    assert result["surface_p18_class_boundary_report"]["p18_class_boundary_surface"] is True
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
    assert manifest["inflate_runtime_benchmark_executed"] is False
    assert manifest["inflate_runtime_benchmark_work_order"]["schema"] == (
        "z8_inflate_runtime_benchmark_work_order.v1"
    )
    assert manifest["inflate_runtime_benchmark_work_order"]["command"][1] == (
        "tools/benchmark_z8_submission_inflate_runtime.py"
    )
    assert manifest["exact_axis_blocker"] == ("receiver_proof_and_contest_cpu_cuda_eval_not_executed")
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_joint_materializer_rejects_legacy_non_true_p19_surface() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)
    surface = _surface_for_archive(archive_bytes, joint_weight, pose_null_mask)
    surface["pose_surface_kind"] = "scalar_first6_pose_mse_vjp_proxy_v1"
    surface["pose_surface_authority"] = False
    surface["pose_axis_count"] = 1
    surface["pose_inverse_variance"] = [1.0]
    surface["pose_surface_blockers"] = ["p19_pose_surface_not_true_per_axis_jacobian"]

    with pytest.raises(ValueError, match="z8_joint_surface_pose_kind_not_true_p19"):
        apply_joint_p18_p19_deadzone_to_z8_archive(
            archive_bytes,
            joint_weight=surface,
            config=Z8JointCoefficientWaterfillConfig(
                joint_weight_quantile=1.0,
                coefficient_deadzone_quantile=1.0,
                quantization_step=0.25,
            ),
        )


def test_quantized_detail_entropy_codec_roundtrips_pair_pyramids() -> None:
    archive_bytes = _archive_bytes()
    original = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(original.wavelet_coeffs_blob)

    quantized_blob = pack_pair_pyramids_to_wavelet_blob(
        pyramids,
        detail_quantization_step=0.25,
    )
    decoded = parse_pair_blobs_from_wavelet_blob(quantized_blob)
    summary = summarize_wavelet_blob_detail_codecs(quantized_blob)

    assert len(decoded) == len(pyramids)
    assert quantized_blob != original.wavelet_coeffs_blob
    assert decoded[0]["frame_0_top_ll"].shape == pyramids[0]["frame_0_top_ll"].shape
    assert decoded[0]["frame_0_details"][0].lh.shape == pyramids[0]["frame_0_details"][0].lh.shape
    assert decoded[0]["frame_1_details"][0].hh.dtype == np.float32
    assert summary["float32_detail_subband_count"] == 0
    assert summary["quantized_or_preconditioned_detail_subband_count"] > 0


def _all_detail_step_keys(pair_pyramid: dict[str, Any], step: float) -> dict[str, float]:
    steps: dict[str, float] = {}
    for details_key in ("frame_0_details", "frame_1_details"):
        for level_idx, _detail in enumerate(pair_pyramid[details_key]):
            for subband in ("lh", "hl", "hh"):
                steps[f"{details_key}:{level_idx}:{subband}"] = float(step)
    return steps


def test_per_subband_detail_quantization_steps_do_not_fall_back_to_float32() -> None:
    archive_bytes = _archive_bytes()
    original = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(original.wavelet_coeffs_blob)
    step_map = {
        (details_key, int(level_idx), subband): step
        for raw_key, step in _all_detail_step_keys(pyramids[0], 0.25).items()
        for details_key, level_idx, subband in [raw_key.split(":")]
    }

    quantized_blob = pack_pair_pyramids_to_wavelet_blob(
        pyramids,
        detail_quantization_steps=step_map,
    )
    summary = summarize_wavelet_blob_detail_codecs(quantized_blob)

    assert summary["float32_detail_subband_count"] == 0
    assert summary["quantized_or_preconditioned_detail_subband_count"] == (
        summary["total_detail_subbands"]
    )


def test_native_constriction_range_detail_codec_roundtrips_i16_symbols() -> None:
    values = np.array(
        [-17, 0, 0, 2, 2, 2, 9, -17, 300, -301, 0, 2],
        dtype=np.int16,
    ).reshape(2, 2, 3)

    payload = _encode_qi16_constriction_range(values)
    decoded = _decode_quantized_detail_payload(
        method=_DETAIL_CODEC_QI16_CONSTRICTION_RANGE,
        payload=payload,
        shape=values.shape,
        quantization_step=0.25,
    )

    np.testing.assert_array_equal(decoded, values.astype(np.float32) * 0.25)


def test_quantized_detail_selector_roundtrips_sparse_streams() -> None:
    coeff = np.zeros((64, 64, 1), dtype=np.float32)
    coeff.reshape(-1)[::97] = 0.25

    from tac.substrates.z8_hierarchical_predictive_coding import (
        canonical_quadruple_binding as cqb,
    )

    method, payload = cqb._encode_quantized_detail_payload(
        coeff,
        quantization_step=0.25,
    )

    assert method in cqb._DETAIL_CODEC_NAMES
    decoded = _decode_quantized_detail_payload(
        method=method,
        payload=payload,
        shape=coeff.shape,
        quantization_step=0.25,
    )
    np.testing.assert_array_equal(decoded, coeff)


def test_lossless_brotli_preconditioned_detail_codec_roundtrips_exactly() -> None:
    archive_bytes = _archive_bytes()
    original = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(original.wavelet_coeffs_blob)

    preconditioned_blob = pack_pair_pyramids_to_wavelet_blob(
        pyramids,
        detail_lossless_preconditioner=True,
    )
    decoded = parse_pair_blobs_from_wavelet_blob(preconditioned_blob)
    summary = summarize_wavelet_blob_detail_codecs(preconditioned_blob)

    assert preconditioned_blob != original.wavelet_coeffs_blob
    assert summary["detail_codec_method_counts"] == {
        "f32_byte_shuffle": summary["total_detail_subbands"]
    }
    np.testing.assert_array_equal(decoded[0]["frame_0_top_ll"], pyramids[0]["frame_0_top_ll"])
    np.testing.assert_array_equal(decoded[0]["frame_1_top_ll"], pyramids[0]["frame_1_top_ll"])
    for details_key in ("frame_0_details", "frame_1_details"):
        for got, expected in zip(decoded[0][details_key], pyramids[0][details_key], strict=True):
            np.testing.assert_array_equal(got.lh, expected.lh)
            np.testing.assert_array_equal(got.hl, expected.hl)
            np.testing.assert_array_equal(got.hh, expected.hh)


def test_joint_p18_p19_entropy_codes_quantized_surviving_details() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    result = apply_joint_p18_p19_deadzone_to_z8_archive(
        archive_bytes,
        joint_weight=_surface_for_archive(archive_bytes, joint_weight, pose_null_mask),
        config=Z8JointCoefficientWaterfillConfig(
            joint_weight_quantile=1.0,
            coefficient_deadzone_quantile=1.0,
            quantization_step=0.25,
            entropy_code_quantized_details=True,
            entropy_detail_quantization_step=0.25,
        ),
    )

    assert result["rate_report"]["entropy_code_quantized_details"] is True
    assert result["rate_report"]["entropy_detail_quantization_step"] == 0.25
    assert result["rate_report"]["after_detail_codec_summary"]["float32_detail_subband_count"] == 0
    assert result["coefficient_report"]["dead_zoned_coefficients"] > 0
    assert result["rate_report"]["after_wavelet_blob_bytes"] < result["rate_report"]["before_wavelet_blob_bytes"]
    parsed = parse_archive(result["mutated_archive_bytes"])
    decoded = parse_pair_blobs_from_wavelet_blob(parsed.wavelet_coeffs_blob)
    assert decoded[0]["frame_0_details"][0].lh.shape == (8, 8, 3)


def test_joint_p18_p19_accepts_per_subband_entropy_steps_without_global_step() -> None:
    archive_bytes = _archive_bytes()
    original = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(original.wavelet_coeffs_blob)
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    result = apply_joint_p18_p19_deadzone_to_z8_archive(
        archive_bytes,
        joint_weight=_surface_for_archive(archive_bytes, joint_weight, pose_null_mask),
        config=Z8JointCoefficientWaterfillConfig(
            joint_weight_quantile=1.0,
            coefficient_deadzone_quantile=1.0,
            quantization_step=0.25,
            entropy_code_quantized_details=True,
            entropy_detail_quantization_steps=_all_detail_step_keys(pyramids[0], 0.25),
        ),
    )

    assert result["rate_report"]["entropy_detail_quantization_step"] is None
    assert result["rate_report"]["entropy_detail_quantization_steps"]
    assert result["rate_report"]["after_detail_codec_summary"]["float32_detail_subband_count"] == 0


def test_relinearized_search_propagates_per_subband_entropy_steps() -> None:
    archive_bytes = _archive_bytes()
    original = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(original.wavelet_coeffs_blob)
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    result = run_joint_p18_p19_relinearized_deadzone_search(
        archive_bytes,
        surfaces=[_surface_for_archive(archive_bytes, joint_weight, pose_null_mask)],
        config=Z8JointCoefficientRelinearizationSearchConfig(
            joint_weight_quantiles=(1.0,),
            coefficient_deadzone_quantiles=(1.0,),
            quantization_steps=(0.25,),
            max_iterations=1,
            entropy_code_quantized_details=True,
            entropy_detail_quantization_steps=_all_detail_step_keys(pyramids[0], 0.25),
        ),
    )

    assert result["iterations_accepted"] == 1
    accepted = result["accepted_candidates"][0]
    assert accepted["rate_report"]["entropy_detail_quantization_step"] is None
    assert accepted["rate_report"]["entropy_detail_quantization_steps"]
    assert accepted["rate_report"]["after_detail_codec_summary"]["float32_detail_subband_count"] == 0


def test_joint_p18_p19_can_emit_lossless_brotli_preconditioned_details() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.ones((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.zeros_like(joint_weight, dtype=bool)

    result = apply_joint_p18_p19_deadzone_to_z8_archive(
        archive_bytes,
        joint_weight=_surface_for_archive(archive_bytes, joint_weight, pose_null_mask),
        config=Z8JointCoefficientWaterfillConfig(
            joint_weight_quantile=0.0,
            coefficient_deadzone_quantile=0.0,
            quantization_step=0.0,
            mutate_coefficients=False,
            lossless_brotli_precondition_details=True,
        ),
    )

    assert result["coefficient_report"]["mutated_pair_count"] == 0
    assert result["coefficient_report"]["dead_zoned_coefficients"] == 0
    assert result["rate_report"]["mutate_coefficients"] is False
    assert result["rate_report"]["lossless_brotli_precondition_details"] is True
    assert result["rate_report"]["entropy_code_quantized_details"] is False
    assert result["rate_report"]["after_detail_codec_summary"]["detail_codec_method_counts"] == {
        "f32_byte_shuffle": result["rate_report"]["after_detail_codec_summary"]["total_detail_subbands"]
    }
    assert result["distortion_report"]["max_abs_delta"] == 0.0
    parsed = parse_archive(result["mutated_archive_bytes"])
    decoded = parse_pair_blobs_from_wavelet_blob(parsed.wavelet_coeffs_blob)
    assert decoded[0]["frame_1_details"][0].hh.dtype == np.float32


def test_storage_only_entropy_transcode_does_not_require_surface() -> None:
    archive_bytes = _archive_bytes()
    original = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(original.wavelet_coeffs_blob)

    result = apply_joint_p18_p19_deadzone_to_z8_archive(
        archive_bytes,
        joint_weight=None,
        config=Z8JointCoefficientWaterfillConfig(
            mutate_coefficients=False,
            entropy_code_quantized_details=True,
            entropy_detail_quantization_steps=_all_detail_step_keys(pyramids[0], 0.25),
        ),
    )

    assert result["coefficient_report"]["mutated_pair_count"] == 0
    assert result["full_video_surface_coverage_report"]["storage_only_no_surface_required"] is True
    assert result["surface_freshness_report"]["fresh_for_current_archive"] is True
    assert result["surface_gradient_reduction_report"]["exact_full_video_gradient_reduction"] is True
    assert result["surface_true_p19_report"]["true_p19_pose_surface"] is True
    assert result["rate_report"]["after_detail_codec_summary"]["float32_detail_subband_count"] == 0


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
    assert all(
        row["surface_true_p19_report"]["true_p19_pose_surface"]
        for row in manifest["accepted_candidates"]
    )
    assert Path(manifest["candidate_bin_path"]).is_file()
    assert Path(manifest["archive_zip_path"]).is_file()
    assert Path(manifest["manifest_path"]).is_file()
    assert manifest["inflate_runtime_benchmark_executed"] is False
    assert manifest["inflate_runtime_benchmark_work_order"]["schema"] == (
        "z8_inflate_runtime_benchmark_work_order.v1"
    )
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


def test_relinearized_search_can_prefilter_before_expensive_full_video_replay() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)
    replay_calls: list[dict[str, Any]] = []

    def local_replay_evaluator(**kwargs: Any) -> dict[str, Any]:
        replay_calls.append(dict(kwargs["candidate_metadata"]))
        return {
            "schema": "unit_full_video_local_replay.v1",
            "full_video_local_replay_executed": True,
            "full_video_local_replay_scope": "full_video",
            "replay_ok": True,
            "contest_action_proxy": 0.5,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }

    result = run_joint_p18_p19_relinearized_deadzone_search(
        archive_bytes,
        surfaces=[_surface_for_archive(archive_bytes, joint_weight, pose_null_mask)],
        local_replay_evaluator=local_replay_evaluator,
        config=Z8JointCoefficientRelinearizationSearchConfig(
            joint_weight_quantiles=(1.0,),
            coefficient_deadzone_quantiles=(0.25, 0.5, 1.0),
            quantization_steps=(0.25,),
            max_iterations=1,
            local_replay_prefilter_top_k=1,
        ),
    )

    assert len(replay_calls) == 1
    assert result["candidate_count"] == 3
    replayed = [row for row in result["candidate_grid"] if row["hard_archive_projection_replay_executed"]]
    skipped = [row for row in result["candidate_grid"] if not row["hard_archive_projection_replay_executed"]]
    assert len(replayed) == 1
    assert len(skipped) == 2
    assert replayed[0]["objective_source"] == "full_video_local_replay"
    assert replayed[0]["receiver_mse_proxy_measured"] is False
    assert all(row["guard_ok"] is False for row in skipped)
    assert all(
        row["full_video_local_replay_blockers"] == ["full_video_local_replay_skipped_by_prefilter"]
        for row in skipped
    )
    assert result["iterations_accepted"] == 1


def test_relinearized_search_blocks_mutating_rate_only_entropy_headroom_probe() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    result = run_joint_p18_p19_relinearized_deadzone_search(
        archive_bytes,
        surfaces=[_surface_for_archive(archive_bytes, joint_weight, pose_null_mask)],
        config=Z8JointCoefficientRelinearizationSearchConfig(
            joint_weight_quantiles=(1.0,),
            coefficient_deadzone_quantiles=(1.0,),
            quantization_steps=(0.25,),
            max_iterations=1,
            measure_receiver_mse_proxy=False,
            entropy_code_quantized_details=True,
            entropy_detail_quantization_step=0.25,
        ),
    )

    assert result["iterations_accepted"] == 0
    row = result["candidate_grid"][0]
    assert row["guard_ok"] is False
    assert row["receiver_mse_proxy_measured"] is False
    assert row["objective_source"] == "rate_only_mutation_probe_blocked"
    assert row["acceptance_blockers"] == [
        "mutating_candidate_requires_receiver_proxy_or_full_video_replay"
    ]
    assert row["incremental_distortion_report"]["blocker"] == (
        "receiver_mse_proxy_skipped_by_full_video_local_replay"
    )
    assert row["rate_report"]["entropy_code_quantized_details"] is True


def test_relinearized_search_allows_storage_only_rate_probe_without_receiver_proxy() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    result = run_joint_p18_p19_relinearized_deadzone_search(
        archive_bytes,
        surfaces=[_surface_for_archive(archive_bytes, joint_weight, pose_null_mask)],
        config=Z8JointCoefficientRelinearizationSearchConfig(
            joint_weight_quantiles=(1.0,),
            coefficient_deadzone_quantiles=(1.0,),
            quantization_steps=(0.0,),
            max_iterations=1,
            measure_receiver_mse_proxy=False,
            mutate_coefficients=False,
            lossless_brotli_precondition_details=True,
        ),
    )

    assert result["iterations_accepted"] == 1
    row = result["accepted_candidates"][0]
    assert row["guard_ok"] is True
    assert row["objective_source"] == "rate_only_storage_probe"
    assert row["acceptance_blockers"] == ["rate_only_storage_probe_no_receiver_proxy"]
    assert row["coefficient_summary"]["dead_zoned_coefficients"] == 0


def test_joint_p18_p19_deadzone_rejects_nonfinite_surface() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    joint_weight.reshape(-1)[0] = np.nan
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    with pytest.raises(ValueError, match="joint_weight contains non-finite"):
        apply_joint_p18_p19_deadzone_to_z8_archive(
            archive_bytes,
            joint_weight=_surface_for_archive(archive_bytes, joint_weight, pose_null_mask),
            config=Z8JointCoefficientWaterfillConfig(
                joint_weight_quantile=1.0,
                coefficient_deadzone_quantile=1.0,
                quantization_step=0.25,
            ),
        )


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


def test_joint_p18_p19_deadzone_rejects_stale_pre_class_boundary_surface() -> None:
    archive_bytes = _archive_bytes()
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)
    surface = _surface_for_archive(archive_bytes, joint_weight, pose_null_mask)
    surface.pop("segnet_class_boundary_authority")
    surface.pop("segnet_class_boundary_blockers")

    with pytest.raises(ValueError, match="p18_class_boundary_authority_missing"):
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
        pose_surface_kind=np.asarray(surface["pose_surface_kind"]),
        pose_surface_authority=np.asarray(surface["pose_surface_authority"]),
        pose_axis_count=np.asarray(surface["pose_axis_count"]),
        pose_inverse_variance=np.asarray(surface["pose_inverse_variance"], dtype=np.float64),
        segnet_class_boundary_authority=np.asarray(surface["segnet_class_boundary_authority"]),
        segnet_class_boundary_blockers_json=np.asarray("[]"),
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
    assert result["surface_true_p19_report"]["true_p19_pose_surface"] is True
    assert result["surface_p18_class_boundary_report"]["p18_class_boundary_surface"] is True
