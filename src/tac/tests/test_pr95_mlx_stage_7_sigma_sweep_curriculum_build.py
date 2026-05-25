# SPDX-License-Identifier: MIT
"""Tests for the PR95 Stage 7 sigma_sweep MLX curriculum bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration.pr95_hnerv_mlx import (
    PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS,
    PR95_STAGE_MODULES,
    pr95_default_optimizer_descriptor_id,
    run_pr95_mlx_synthetic_timing_smoke,
    stage_smoke_config,
)
from tac.optimization.optimizer_scheduler_registry import (
    FALSE_AUTHORITY_FIELDS,
    default_optimizer_scheduler_registry,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate


def test_pr95_stage_modules_contains_stage_7_sigma_sweep() -> None:
    assert PR95_STAGE_MODULES[7] == "stage7_sigma_sweep"
    assert PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[7] == (
        "pr95_stage7_adamw_sigma_sweep_mlx"
    )
    assert pr95_default_optimizer_descriptor_id(7) == (
        "pr95_stage7_adamw_sigma_sweep_mlx"
    )


def test_pr95_stage7_descriptor_matches_recovered_curriculum() -> None:
    registry = default_optimizer_scheduler_registry()
    row = registry.get("pr95_stage7_adamw_sigma_sweep_mlx").to_planner_candidate()

    assert row["optimizer_config"]["use_muon"] is False
    assert row["optimizer_config"]["adamw_lr"] == 3e-5
    assert row["optimizer_config"]["latent_lr_mult"] == 10.0

    training = row["training_config"]
    assert training["pr95_stage_indices"] == [7]
    assert training["stage_modules"] == ["stage7_sigma_sweep"]
    assert training["stage_epochs"] == 3000
    assert training["stage_loss_family"] == "l7_softplus_seg_loss"
    assert training["stage_cat_lambda"] == 0.02
    assert training["stage_cat_sigma"] == 0.1
    assert training["stage_uses_qat"] is True
    assert training["stage_uses_muon"] is False
    assert training["score_claim"] is False
    assert training["promotion_eligible"] is False
    assert training["rank_or_kill_eligible"] is False
    assert training["ready_for_exact_eval_dispatch"] is False

    assert validate_proxy_candidate(row) == []
    for key, expected in FALSE_AUTHORITY_FIELDS.items():
        assert row[key] is expected


def test_pr95_stage7_sigma_sweep_is_the_distinguishing_parameter() -> None:
    registry = default_optimizer_scheduler_registry()
    stage6 = registry.get("pr95_stage6_adamw_lambda_sweep_mlx").to_planner_candidate()
    stage7 = registry.get("pr95_stage7_adamw_sigma_sweep_mlx").to_planner_candidate()

    assert stage6["optimizer_config"]["adamw_lr"] == stage7["optimizer_config"][
        "adamw_lr"
    ] == 3e-5
    assert stage6["training_config"]["stage_loss_family"] == stage7[
        "training_config"
    ]["stage_loss_family"] == "l7_softplus_seg_loss"
    assert stage6["training_config"]["stage_cat_lambda"] == stage7[
        "training_config"
    ]["stage_cat_lambda"] == 0.02
    assert stage6["training_config"]["stage_uses_qat"] is True
    assert stage7["training_config"]["stage_uses_qat"] is True
    assert stage6["training_config"]["stage_uses_muon"] is False
    assert stage7["training_config"]["stage_uses_muon"] is False

    assert stage6["training_config"]["stage_cat_sigma"] == 0.2
    assert stage7["training_config"]["stage_cat_sigma"] == 0.1


def test_stage_smoke_config_stage_7_dispatches_canonical_module() -> None:
    cfg = stage_smoke_config(7)
    assert cfg.stage_index == 7
    assert cfg.stage_module == "stage7_sigma_sweep"
    assert cfg.optimizer_descriptor_id == "pr95_stage7_adamw_sigma_sweep_mlx"
    assert cfg.optimizer.use_muon is False
    assert cfg.optimizer.adamw_lr == 3e-5
    assert cfg.optimizer_backend_status == "implemented_mlx_local_timing_proxy"


@pytest.mark.timeout(60)
def test_stage_7_synthetic_timing_smoke_runs_end_to_end() -> None:
    result = run_pr95_mlx_synthetic_timing_smoke(
        stage_index=7,
        steps=3,
        batch_size=1,
        synthetic_pairs=1,
        seed=20260525,
        base_channels=36,
        latent_dim=28,
    )

    assert result["stage_index"] == 7
    assert result["stage_module"] == "stage7_sigma_sweep"
    assert "stage7_pr95_stage7_adamw_sigma_sweep_mlx" in result["candidate_id"]
    assert isinstance(result["last_loss"], float)
    assert np.isfinite(result["last_loss"])
    assert result["runtime_profile"]["optimizer_descriptor_id"] == (
        "pr95_stage7_adamw_sigma_sweep_mlx"
    )
    assert result["runtime_profile"]["training_fidelity"] == "synthetic_timing_only"
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["rank_or_kill_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_catalog_313_probe_outcomes_row_registered_for_stage7() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    ledger_path = repo_root / ".omx" / "state" / "probe_outcomes.jsonl"
    probe_id = (
        "pr95_mlx_stage_7_sigma_sweep_curriculum_build_synthetic_timing_smoke_3step"
    )

    found = False
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("probe_id") == probe_id:
            found = True
            assert row.get("verdict") == "PROCEED"
            assert row.get("blocker_status") == "advisory"
            break
    assert found, f"missing Catalog #313 probe outcome row for {probe_id}"
