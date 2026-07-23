# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tools.measure_ddm_v15_scorer_solved_templates import (
    REPRESENTATIVE_ISLANDS,
    DDMV15ScorerSolvedTemplateConfigV1,
    _bank_with,
    _deterministic_storage_receipt,
    _role_templates,
)


def _config(**updates: object) -> DDMV15ScorerSolvedTemplateConfigV1:
    values: dict[str, object] = {
        "run_id": "fixture_ddm_v15_n64",
        "pair_start": 448,
        "pair_count": 64,
        "v14_receipt_path": "v14.json",
        "v14_receipt_sha256": "1" * 64,
        "v14_archive_path": "v14.zip",
        "v14_archive_sha256": "2" * 64,
        "solve_archive_path": "solve.zip",
        "solve_archive_sha256": "3" * 64,
        "target_cache_path": "gt_n600.npz",
        "target_cache_bytes": 5_078_017_610,
        "target_cache_sha256": "4" * 64,
        "upstream_root": "/absolute/upstream",
    }
    values.update(updates)
    return DDMV15ScorerSolvedTemplateConfigV1(**values)


def test_v15_config_seals_preregistered_development_set_and_false_authority() -> None:
    config = _config()
    assert config.representative_source_pair_ids == REPRESENTATIVE_ISLANDS
    assert config.row_band_edges == (0, 128, 256, 384)
    assert config.max_candidate_stages_per_invocation == 1
    assert config.archive_box_bytes == 160_000
    assert config.score_claim is False
    assert config.execution_allowed is False
    with pytest.raises(ValueError, match="preregistered eight"):
        _config(representative_source_pair_ids=tuple(reversed(REPRESENTATIVE_ISLANDS)))


def test_v15_full_n600_requires_sha_bound_n64_template_source() -> None:
    with pytest.raises(ValueError, match="full n600 verdict"):
        _config(run_id="fixture_ddm_v15_n600", pair_start=0, pair_count=600)
    config = _config(
        run_id="fixture_ddm_v15_n600",
        pair_start=0,
        pair_count=600,
        template_source_path="solved.ddst",
        template_source_sha256="5" * 64,
    )
    assert config.template_source_path == "solved.ddst"


def test_v15_row_band_template_builder_is_canonical_and_counted() -> None:
    movable = _role_templates("Movable", "fill", (0, 128, 256, 384), np.array((107, 0, 114)))
    lane = _role_templates("Lane", "inner_boundary", (0, 192, 384), np.array((11, 3, 9)))
    bank = _bank_with(movable, lane)
    assert [row.role for row in bank.templates] == ["Lane", "Lane", "Movable", "Movable", "Movable"]
    assert all(row.patch_height == row.patch_width == 1 for row in bank.templates)
    assert sum(len(row.rgb_u8) for row in bank.templates) == 15


def test_v15_storage_receipt_excludes_live_capacity_and_worktree_path() -> None:
    receipt = _deterministic_storage_receipt(
        {
            "output_tier": "/volatile/worktree/.omx/research",
            "required_free_bytes": 128 * 1024 * 1024,
            "observed_free_bytes": 987_654_321,
            "free_space_gate_satisfied": True,
            "bulk_target_tier": "/Volumes/VertigoDataTier/pact",
            "bulk_target_read_only": True,
            "status": "PASS",
        }
    )
    assert receipt["output_tier"] == "local_small_receipt"
    assert receipt["observed_free_bytes_recorded"] is False
    assert "observed_free_bytes" not in receipt
    assert "/volatile/worktree" not in str(receipt)
