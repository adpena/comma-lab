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
        "v9_cgauge_ideal_mod19_sR",
        "v9_cgauge_ideal_mod32",
        "v9_cgauge_432_taper_off",
        "v9_cgauge_432_horizon_iso",
        "v9_cgauge_432_step_iso",
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


def test_fresh_seeded_l7_epochs_plus_one_parking_is_legal() -> None:
    # (c2 adversarial review 2026-07-16) l7 parked at exactly epochs+1 (1001/epochs=1000) is the
    # CANONICAL "TRUE never" form — the trainer's loop is range(start, epochs+1) INCLUSIVE, so
    # ==epochs would run l7 on the final epoch. fresh_seeded's own docstring anticipated this
    # relax ("a follow-up (the C2/L1 wave) is expected to admit the '>= epochs == never' form").
    cfg = derive_named_config(
        "fresh_seeded", GT_N24, num_pairs=24, epochs=None, overfit=True
    )
    assert cfg is not None


def test_l7_past_epochs_but_not_plus_one_still_refuses() -> None:
    # Dead-stage protection intact: only the exact epochs+1 parking is exempt; an arbitrary
    # past-budget l7 value is still a config bug.
    violations = schedule_epoch_budget_violations({
        "--curriculum": True,
        "--epochs": 1000,
        "--tau-softplus-start-epoch": 300,
        "--l7-start-epoch": 1005,
    })
    assert violations and "--l7-start-epoch=1005" in violations[0]


def test_l7_exactly_epochs_plus_one_is_exempt_pure() -> None:
    violations = schedule_epoch_budget_violations({
        "--curriculum": True,
        "--epochs": 1000,
        "--tau-softplus-start-epoch": 300,
        "--l7-start-epoch": 1001,
    })
    assert all("--l7-start-epoch" not in v for v in violations)


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
        "v9_cgauge_ideal_mod19_sR",
        "v9_cgauge_ideal_mod32",
        "v9_cgauge_432_taper_off",
        "v9_cgauge_432_horizon_iso",
        "v9_cgauge_432_step_iso",
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
