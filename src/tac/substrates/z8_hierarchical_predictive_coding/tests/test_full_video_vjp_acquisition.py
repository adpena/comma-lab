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
    Z8FullVideoVjpAcquisitionConfig,
    assemble_z8_full_video_vjp_surface_bundle,
    build_z8_full_video_vjp_acquisition_contract,
    build_z8_full_video_vjp_acquisition_plan,
    write_z8_full_video_vjp_surface_bundle,
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
    assert plan["optimizer_update_semantics"] == "single_update_after_all_pair_shards_reduce"
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
    assert bundle["budget_spend_authority"] is True
    assert bundle["optimizer_update_authority"] is True
    assert bundle["optimizer_update_semantics"] == "single_update_after_all_pair_shards_reduce"
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
    assert manifest["budget_spend_authority"] is True
    assert manifest["optimizer_update_authority"] is True
    assert manifest["optimizer_update_semantics"] == "single_update_after_all_pair_shards_reduce"
    payload = np.load(manifest["surface_path"])
    assert payload["joint_weight"].shape == (2, 2, 16, 16, 3)
    assert payload["rate_attack_deadzone_mask"].shape == (2, 2, 16, 16, 3)


def test_full_video_vjp_contract_keeps_contest_and_production_modes_explicit() -> None:
    contract = build_z8_full_video_vjp_acquisition_contract()

    assert "full_video_pair_grid_coverage" in contract["contest_budget_spend_requires"]
    assert "single_optimizer_update_after_full_shard_reduction" in contract["contest_budget_spend_requires"]
    assert "declared_corpus_manifest" in contract["production_budget_spend_requires"]
    assert contract["minibatch_window_gradients_role"] == "ranking_probe_only_between_full_video_passes"
    assert contract["score_claim"] is False
