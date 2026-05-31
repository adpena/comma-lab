# SPDX-License-Identifier: MIT
"""Tests for Z8 full-video VJP acquisition contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.optimization.target_modes import CORPUS_GENERALIZATION_MODE
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
)
from tac.substrates.z8_hierarchical_predictive_coding.full_video_vjp_acquisition import (
    FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
    SINGLE_UPDATE_AFTER_FULL_REDUCTION,
    Z8_FULL_VIDEO_VJP_MLX_SHARD_BACKEND,
    Z8FullVideoMlxVjpShardConfig,
    Z8FullVideoVjpAcquisitionConfig,
    assemble_z8_full_video_vjp_surface_bundle,
    build_z8_full_video_mlx_surface_provider,
    build_z8_full_video_mlx_vjp_surface_shard,
    build_z8_full_video_vjp_acquisition_contract,
    build_z8_full_video_vjp_acquisition_plan,
    load_z8_full_video_vjp_surface_shard_file,
    reconstruct_z8_archive_pairs_rgb255,
    write_z8_full_video_vjp_acquisition_plan,
    write_z8_full_video_vjp_surface_bundle,
    write_z8_full_video_vjp_surface_shard,
)
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    Z8HierarchicalConfig,
)


def _archive_bytes(*, num_pairs: int = 5) -> bytes:
    rng = np.random.RandomState(31)
    f0 = rng.uniform(0, 1, size=(num_pairs, 16, 16, 3)).astype(np.float32)
    f1 = rng.uniform(0, 1, size=(num_pairs, 16, 16, 3)).astype(np.float32)
    cfg = Z8HierarchicalConfig(
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
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    return build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)


def _surface_shard(
    *,
    start: int,
    end: int,
    archive_sha: str,
    value: float,
    shard_index: int,
) -> dict:
    shape = (end - start, 2, 16, 16, 3)
    return {
        "shard_index": shard_index,
        "pair_start": start,
        "pair_end": end,
        "linearization_archive_sha": archive_sha,
        "joint_weight": np.full(shape, value, dtype=np.float32),
        "rate_attack_deadzone_mask": np.ones(shape, dtype=bool),
    }


class _ToyMlxScorer:
    def __init__(self, mx_module):
        self.mx = mx_module

    def segnet(self, x_nhwc):
        signal = self.mx.mean(x_nhwc, axis=-1, keepdims=True) * (0.1 / 255.0)
        return self.mx.concatenate([signal, -signal], axis=-1)

    def posenet(self, x_nhwc):
        spatial = self.mx.mean(self.mx.mean(x_nhwc, axis=2), axis=1)
        return {"pose": spatial}


def test_full_video_vjp_plan_shards_archive_pairs_and_marks_probes_non_authority() -> None:
    archive = _archive_bytes(num_pairs=5)

    plan = build_z8_full_video_vjp_acquisition_plan(
        archive,
        config=Z8FullVideoVjpAcquisitionConfig(pair_chunk_size=2, parallel_workers=3),
    )

    assert plan["full_video_vjp_is_first_class_acquisition_lane"] is True
    assert plan["archive_num_pairs"] == 5
    assert plan["shard_count"] == 3
    assert [(row["pair_start"], row["pair_end"]) for row in plan["pair_shards"]] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert plan["parallel_workers"] == 3
    assert plan["pair_chunk_updates_forbidden"] is True
    assert plan["gradient_reduction_semantics"] == FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION
    assert plan["optimizer_update_semantics"] == SINGLE_UPDATE_AFTER_FULL_REDUCTION
    assert plan["minibatch_window_gradients_budget_spend_authority"] is False
    assert plan["score_claim"] is False


def test_full_video_vjp_surface_bundle_requires_archive_pinned_complete_shards() -> None:
    archive = _archive_bytes(num_pairs=4)
    plan = build_z8_full_video_vjp_acquisition_plan(
        archive,
        config=Z8FullVideoVjpAcquisitionConfig(pair_chunk_size=2),
    )
    sha = plan["archive_sha256"]

    bundle = assemble_z8_full_video_vjp_surface_bundle(
        archive,
        shard_surfaces=[
            _surface_shard(start=0, end=2, archive_sha=sha, value=0.0, shard_index=0),
            _surface_shard(start=2, end=4, archive_sha=sha, value=0.1, shard_index=1),
        ],
        config=Z8FullVideoVjpAcquisitionConfig(pair_chunk_size=2),
    )

    assert bundle["linearization_archive_sha"] == sha
    assert bundle["full_video_surface_coverage"] is True
    assert bundle["full_video_reduction_complete"] is True
    assert bundle["gradient_reduction_semantics"] == FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION
    assert bundle["gradient_reduction_authority"] is True
    assert bundle["budget_spend_authority"] is True
    assert bundle["optimizer_update_authority"] is True
    assert bundle["optimizer_update_semantics"] == SINGLE_UPDATE_AFTER_FULL_REDUCTION
    assert bundle["joint_weight"].shape == (4, 2, 16, 16, 3)
    assert bundle["rate_attack_deadzone_mask"].shape == (4, 2, 16, 16, 3)


def test_full_video_vjp_surface_bundle_rejects_partial_or_stale_shards() -> None:
    archive = _archive_bytes(num_pairs=4)
    sha = build_z8_full_video_vjp_acquisition_plan(archive)["archive_sha256"]

    with pytest.raises(ValueError, match="does not cover archive pair grid"):
        assemble_z8_full_video_vjp_surface_bundle(
            archive,
            shard_surfaces=[
                _surface_shard(start=0, end=2, archive_sha=sha, value=0.0, shard_index=0),
            ],
        )

    with pytest.raises(ValueError, match="archive mismatch"):
        assemble_z8_full_video_vjp_surface_bundle(
            archive,
            shard_surfaces=[
                _surface_shard(start=0, end=2, archive_sha="b" * 64, value=0.0, shard_index=0),
                _surface_shard(start=2, end=4, archive_sha=sha, value=0.1, shard_index=1),
            ],
        )

    bad_shard = _surface_shard(start=0, end=4, archive_sha=sha, value=0.0, shard_index=0)
    bad_shard["optimizer_update_applied"] = True
    with pytest.raises(ValueError, match="cannot carry optimizer update authority"):
        assemble_z8_full_video_vjp_surface_bundle(
            archive,
            shard_surfaces=[bad_shard],
        )

    bad_authority_shard = _surface_shard(start=0, end=4, archive_sha=sha, value=0.0, shard_index=0)
    bad_authority_shard["gradient_reduction_authority"] = True
    with pytest.raises(ValueError, match="cannot carry optimizer update authority"):
        assemble_z8_full_video_vjp_surface_bundle(
            archive,
            shard_surfaces=[bad_authority_shard],
        )


def test_full_video_vjp_production_mode_requires_corpus_manifest() -> None:
    with pytest.raises(ValueError, match="corpus_manifest_path"):
        Z8FullVideoVjpAcquisitionConfig(target_mode=CORPUS_GENERALIZATION_MODE)

    archive = _archive_bytes(num_pairs=2)
    plan = build_z8_full_video_vjp_acquisition_plan(
        archive,
        config=Z8FullVideoVjpAcquisitionConfig(
            target_mode=CORPUS_GENERALIZATION_MODE,
            corpus_manifest_path="fleet/corpus.json",
        ),
    )

    assert plan["target_mode"] == CORPUS_GENERALIZATION_MODE
    assert plan["declared_overfit_allowed"] is False
    assert plan["corpus_manifest_required"] is True
    assert plan["production_mode_requires_declared_corpus_manifest"] is True


def test_full_video_vjp_surface_bundle_writes_materializer_ready_npz(tmp_path: Path) -> None:
    archive = _archive_bytes(num_pairs=2)
    sha = build_z8_full_video_vjp_acquisition_plan(archive)["archive_sha256"]
    bundle = assemble_z8_full_video_vjp_surface_bundle(
        archive,
        shard_surfaces=[
            _surface_shard(start=0, end=2, archive_sha=sha, value=0.2, shard_index=0),
        ],
        config=Z8FullVideoVjpAcquisitionConfig(pair_chunk_size=2),
    )

    manifest = write_z8_full_video_vjp_surface_bundle(bundle, tmp_path)

    assert Path(manifest["surface_path"]).is_file()
    assert Path(manifest["manifest_path"]).is_file()
    assert manifest["linearization_archive_sha"] == sha
    assert manifest["gradient_reduction_semantics"] == FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION
    assert manifest["gradient_reduction_authority"] is True
    assert manifest["budget_spend_authority"] is True
    assert manifest["optimizer_update_authority"] is True
    assert manifest["optimizer_update_semantics"] == SINGLE_UPDATE_AFTER_FULL_REDUCTION
    payload = np.load(manifest["surface_path"])
    assert payload["joint_weight"].shape == (2, 2, 16, 16, 3)
    assert payload["rate_attack_deadzone_mask"].shape == (2, 2, 16, 16, 3)
    assert str(payload["gradient_reduction_semantics"]) == FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION
    assert bool(payload["gradient_reduction_authority"]) is True
    assert bool(payload["optimizer_update_authority"]) is True
    assert str(payload["optimizer_update_semantics"]) == SINGLE_UPDATE_AFTER_FULL_REDUCTION


def test_full_video_vjp_plan_and_shard_file_loader_are_queue_ready(tmp_path: Path) -> None:
    archive = _archive_bytes(num_pairs=2)
    plan = write_z8_full_video_vjp_acquisition_plan(
        archive,
        tmp_path / "plan",
        config=Z8FullVideoVjpAcquisitionConfig(pair_chunk_size=1),
    )
    sha = plan["archive_sha256"]
    shard_path = tmp_path / "shard0.npz"
    np.savez_compressed(
        shard_path,
        shard_index=np.asarray(0),
        pair_start=np.asarray(0),
        pair_end=np.asarray(2),
        linearization_archive_sha=np.asarray(sha),
        joint_weight=np.zeros((2, 2, 16, 16, 3), dtype=np.float32),
        rate_attack_deadzone_mask=np.ones((2, 2, 16, 16, 3), dtype=bool),
    )

    loaded = load_z8_full_video_vjp_surface_shard_file(shard_path)
    bundle = assemble_z8_full_video_vjp_surface_bundle(
        archive,
        shard_surfaces=[loaded],
    )

    assert Path(plan["plan_path"]).is_file()
    assert loaded["linearization_archive_sha"] == sha
    assert loaded["gradient_reduction_authority"] is False
    assert bundle["gradient_reduction_semantics"] == FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION
    assert bundle["gradient_reduction_authority"] is True
    assert bundle["budget_spend_authority"] is True
    assert bundle["optimizer_update_semantics"] == SINGLE_UPDATE_AFTER_FULL_REDUCTION


def test_full_video_vjp_contract_keeps_contest_and_production_modes_explicit() -> None:
    contract = build_z8_full_video_vjp_acquisition_contract()

    assert "full_video_pair_grid_coverage" in contract["contest_budget_spend_requires"]
    assert "single_optimizer_update_after_full_shard_reduction" in contract["contest_budget_spend_requires"]
    assert "declared_corpus_manifest" in contract["production_budget_spend_requires"]
    assert contract["minibatch_window_gradients_role"] == "ranking_probe_only_between_full_video_passes"
    assert contract["score_claim"] is False


def test_mlx_vjp_shard_producer_emits_non_authority_exact_reduction_surface(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")

    archive = _archive_bytes(num_pairs=2)
    rng = np.random.default_rng(316)
    reference = rng.uniform(16, 220, size=(2, 2, 8, 10, 3)).astype(np.float32)
    candidate = np.clip(
        reference + rng.normal(0, 2.0, size=reference.shape),
        0,
        255,
    ).astype(np.float32)

    shard = build_z8_full_video_mlx_vjp_surface_shard(
        archive,
        reference_pairs_rgb=reference,
        candidate_pairs_rgb=candidate,
        mlx_scorer=_ToyMlxScorer(mx),
        config=Z8FullVideoMlxVjpShardConfig(
            shard_index=0,
            pair_start=0,
            pair_end=2,
            full_video_pair_count=2,
            full_video_d_pose=0.01,
            scorer_hw=(8, 10),
            seg_margin_delta=1.0,
        ),
    )

    assert shard["surface_generation_backend"] == Z8_FULL_VIDEO_VJP_MLX_SHARD_BACKEND
    assert shard["gradient_reduction_semantics"] == FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION
    assert shard["optimizer_update_applied"] is False
    assert shard["budget_spend_authority"] is False
    assert shard["joint_weight"].shape == (2, 2, 8, 10, 3)
    assert shard["rate_attack_deadzone_mask"].shape == (2, 2, 8, 10, 3)
    assert shard["segnet_vjp_abs_max"] > 0.0
    assert shard["pose_vjp_abs_max"] > 0.0
    assert shard["score_claim"] is False

    manifest = write_z8_full_video_vjp_surface_shard(shard, tmp_path)
    loaded = load_z8_full_video_vjp_surface_shard_file(manifest["shard_path"])
    bundle = assemble_z8_full_video_vjp_surface_bundle(
        archive,
        shard_surfaces=[loaded],
        config=Z8FullVideoVjpAcquisitionConfig(pair_chunk_size=2),
    )

    assert loaded["surface_generation_backend"] == Z8_FULL_VIDEO_VJP_MLX_SHARD_BACKEND
    assert loaded["budget_spend_authority"] is False
    assert bundle["full_video_reduction_complete"] is True
    assert bundle["gradient_reduction_authority"] is True
    assert bundle["budget_spend_authority"] is True


def test_mlx_surface_provider_reconstructs_archive_and_reduces_fresh_bundle(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    archive = _archive_bytes(num_pairs=2)
    reference_pairs = reconstruct_z8_archive_pairs_rgb255(archive)

    provider = build_z8_full_video_mlx_surface_provider(
        reference_pairs_rgb=reference_pairs,
        mlx_scorer=_ToyMlxScorer(mx),
        acquisition_config=Z8FullVideoVjpAcquisitionConfig(pair_chunk_size=1),
        scorer_hw=(16, 16),
        artifact_dir=tmp_path / "provider_artifacts",
    )
    bundle = provider(0, archive)

    assert bundle["surface_provider_backend"] == "z8_full_video_mlx_archive_fresh_surface_provider.v1"
    assert bundle["linearization_archive_sha"] == build_z8_full_video_vjp_acquisition_plan(archive)["archive_sha256"]
    assert bundle["full_video_surface_coverage"] is True
    assert bundle["full_video_reduction_complete"] is True
    assert bundle["gradient_reduction_authority"] is True
    assert bundle["budget_spend_authority"] is True
    assert bundle["joint_weight"].shape == reference_pairs.shape
    assert Path(bundle["surface_provider_bundle_manifest"]["manifest_path"]).is_file()
