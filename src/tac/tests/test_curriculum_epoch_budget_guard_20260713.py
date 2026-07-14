# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.witness_dsl.curriculum_dsl import schedule_epoch_budget_violations
from tac.witness_dsl.spec_throughput_component_timer_20260713 import (
    compile_throughput_component_timer_ticket,
)
from tools.launch_witness_run import derive_named_config

GT_N24 = "experiments/results/mlx_fleet_gt_cache/gt_n24.npz"


def test_enabled_short_budget_reports_every_effective_stage_cap() -> None:
    violations = schedule_epoch_budget_violations({
        "--curriculum": True,
        "--epochs": 4,
        "--muon-start-epoch": 726,
        "--pose-finish-start-epoch": 726,
    })
    assert len(violations) == 1
    message = violations[0]
    assert "epochs=4" in message
    assert "--muon-start-epoch=726" in message
    assert "--pose-finish-start-epoch=726" in message
    # Real argparse defaults are part of the effective config, not hidden.
    assert "--tau-softplus-start-epoch=300" in message
    assert "--l7-start-epoch=800" in message


def test_stage_at_budget_boundary_passes() -> None:
    assert schedule_epoch_budget_violations({
        "--curriculum": True,
        "--epochs": 800,
        "--muon-start-epoch": 800,
    }) == []


def test_disabled_curriculum_is_vacuous_even_with_legacy_caps() -> None:
    assert schedule_epoch_budget_violations({
        "--curriculum": False,
        "--epochs": 4,
        "--muon-start-epoch": 726,
    }) == []
    assert schedule_epoch_budget_violations([
        ("--no-curriculum", None),
        ("--epochs", "4"),
        ("--muon-start-epoch", "726"),
    ]) == []


@pytest.mark.parametrize(
    "config",
    [
        "proven_base",
        "all_levers",
        "sealed_205",
        "store_nothing_205",
        "crucible_v6",
        "crucible_v7",
        "crucible_v752",
        "crucible_v753",
        "v9_cgauge_432",
        "v9_cgauge_truly_optimal_core",
        "v9_cgauge_ideal_mod19",
        "v9_cgauge_ideal_mod32",
        "next_launch_all_levers_20260713",
        "next_launch_all_levers_trimmed_20260713",
        "throughput_component_timer_async_20260713",
        "throughput_component_timer_solo_20260713",
    ],
)
def test_feasible_named_configs_pass_at_sealed_budget(config: str) -> None:
    cfg = derive_named_config(
        config, GT_N24, num_pairs=24, epochs=None, overfit=True
    )
    assert cfg is not None


def test_fresh_seeded_latent_l7_window_refuses_at_sealed_budget() -> None:
    with pytest.raises(ValueError, match=r"--l7-start-epoch=1001"):
        derive_named_config(
            "fresh_seeded", GT_N24, num_pairs=24, epochs=None, overfit=True
        )


@pytest.mark.parametrize(
    "config",
    [
        "proven_base",
        "fresh_seeded",
        "crucible_v6",
        "crucible_v7",
        "crucible_v752",
        "crucible_v753",
        "v9_cgauge_432",
        "v9_cgauge_truly_optimal_core",
        "v9_cgauge_ideal_mod19",
        "v9_cgauge_ideal_mod32",
        "next_launch_all_levers_20260713",
        "next_launch_all_levers_trimmed_20260713",
    ],
)
def test_short_epoch_non_timer_configs_refuse_at_construction(config: str) -> None:
    with pytest.raises(ValueError, match="EPOCH-BUDGET FEASIBILITY"):
        derive_named_config(
            config, GT_N24, num_pairs=24, epochs=4, overfit=True
        )


@pytest.mark.parametrize("config", ["all_levers", "sealed_205", "store_nothing_205"])
def test_short_epoch_configs_with_scaled_schedule_pass(config: str) -> None:
    cfg = derive_named_config(
        config, GT_N24, num_pairs=24, epochs=4, overfit=True
    )
    assert schedule_epoch_budget_violations(cfg.to_trainer_flags("OUT")) == []


@pytest.mark.parametrize("variant", ["async_current", "solo_control"])
def test_timer_passes_generic_guard(variant: str) -> None:
    cfg = compile_throughput_component_timer_ticket(variant=variant)
    assert schedule_epoch_budget_violations(cfg.to_trainer_flags("OUT")) == []
