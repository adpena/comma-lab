# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from comma_lab.storage_tiers import (
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)


def test_storage_waterfall_selects_first_writable_tier(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    fast = tmp_path / "fast"
    slow = tmp_path / "slow"
    repo.mkdir()
    fast.mkdir()
    slow.mkdir()
    tiers = parse_storage_tier_specs(
        [f"fast={fast}", f"slow={slow}"],
        repo_root=repo,
        reserve_free_gb=0,
        allow_local_disk=True,
    )

    plan = plan_experiment_storage(
        tiers,
        workload_subdir="experiments/results/post_training",
        create=True,
    )

    assert plan.selected_tier == "fast"
    assert require_selected_storage(plan) == fast / "experiments/results/post_training"
    payload = plan.to_dict()
    assert payload["score_claim"] is False
    assert payload["tiers"][0]["eligible"] is True
    assert payload["tiers"][0]["workload_root_exists"] is True


def test_storage_waterfall_rejects_absolute_workload_subdir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    fast = tmp_path / "fast"
    repo.mkdir()
    fast.mkdir()
    tiers = parse_storage_tier_specs(
        [f"fast={fast}"],
        repo_root=repo,
        reserve_free_gb=0,
        allow_local_disk=True,
    )

    with pytest.raises(StorageTierError, match="workload_subdir must be relative"):
        plan_experiment_storage(tiers, workload_subdir=str(tmp_path / "abs"))


def test_storage_waterfall_keeps_local_disk_opt_in(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    tiers = parse_storage_tier_specs(
        [f"local={repo}"],
        repo_root=repo,
        reserve_free_gb=0,
        allow_local_disk=False,
    )

    plan = plan_experiment_storage(tiers, workload_subdir="experiments/results")

    assert plan.selected_tier is None
    assert "local_disk_tier_disabled" in plan.tiers[0].blockers


def test_storage_waterfall_does_not_select_missing_workload_root_without_create(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    fast = tmp_path / "fast"
    repo.mkdir()
    fast.mkdir()
    tiers = parse_storage_tier_specs(
        [f"fast={fast}"],
        repo_root=repo,
        reserve_free_gb=0,
        allow_local_disk=True,
    )

    plan = plan_experiment_storage(
        tiers,
        workload_subdir="experiments/results/post_training",
        create=False,
    )

    assert plan.selected_tier is None
    assert plan.tiers[0].workload_root_exists is False
    assert "workload_root_missing" in plan.tiers[0].blockers
