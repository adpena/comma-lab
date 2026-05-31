# SPDX-License-Identifier: MIT
"""Tests for executable Z8 joint P18/P19 coefficient water-fill materializer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_archive
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
    parse_pair_blobs_from_wavelet_blob,
)
from tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill import (
    Z8_JOINT_COEFFICIENT_RATE_ATTACK_ROLE,
    Z8_JOINT_COEFFICIENT_VARIANT_MANIFEST_SCHEMA,
    Z8JointCoefficientWaterfillConfig,
    apply_joint_p18_p19_deadzone_to_z8_archive,
    materialize_joint_p18_p19_deadzone_candidate,
)
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    Z8HierarchicalConfig,
)


def _cfg() -> Z8HierarchicalConfig:
    return Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=1,
        deterministic_state_dim=16,
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(16, 16),
    )


def _archive_bytes() -> bytes:
    rng = np.random.RandomState(19)
    f0 = rng.uniform(0, 1, size=(1, 16, 16, 3)).astype(np.float32)
    f1 = rng.uniform(0, 1, size=(1, 16, 16, 3)).astype(np.float32)
    binding = build_canonical_quadruple_binding_from_z8_config(_cfg())
    return build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)


def test_joint_p18_p19_deadzone_mutates_wavelet_details_and_reduces_rate() -> None:
    archive_bytes = _archive_bytes()
    original = parse_archive(archive_bytes)
    original_pyramids = parse_pair_blobs_from_wavelet_blob(original.wavelet_coeffs_blob)
    joint_weight = np.zeros((1, 2, 16, 16, 3), dtype=np.float32)
    pose_null_mask = np.ones_like(joint_weight, dtype=bool)

    result = apply_joint_p18_p19_deadzone_to_z8_archive(
        archive_bytes,
        joint_weight=joint_weight,
        rate_attack_deadzone_mask=pose_null_mask,
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
    assert result["rate_report"]["after_wavelet_blob_bytes"] < result["rate_report"][
        "before_wavelet_blob_bytes"
    ]
    assert result["distortion_report"]["small_receiver_distortion_measured"] is True
    assert result["distortion_report"]["max_abs_delta"] > 0.0
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
        joint_weight=joint_weight,
        rate_attack_deadzone_mask=pose_null_mask,
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
    assert manifest["exact_axis_blocker"] == (
        "receiver_proof_and_contest_cpu_cuda_eval_not_executed"
    )
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
