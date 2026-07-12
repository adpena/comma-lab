"""Regression tests for CLI Lever composition on requirement-V typed configs."""

from __future__ import annotations

from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config
from tac.witness_dsl.typed_config import verify_launch_manifest


def _base():
    return compile_v9_cgauge_432_launch_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n24.npz",
        num_pairs=8,
        epochs=5,
        out_dir="experiments/results/test_typed_launcher_dsl_composition",
    )


def test_typed_v9_composes_fresh_factory_and_regenerates_manifest() -> None:
    base = _base()
    composed = base.with_dsl_lever_factories("FreshFrequencyShift")
    flags = composed.typed.to_program().flag_dict()

    assert base.dsl_levers[-1] != "fresh_frequency_shift_init"
    assert composed.dsl_levers == (*base.dsl_levers, "fresh_frequency_shift_init")
    assert flags["--fresh-init"] is True
    assert flags["--seed-islands"] is False
    assert flags["--seed-island-eased"] is False
    assert flags["--witness-alone-island-loss"] is False
    assert composed.dsl_program_manifest["expected_active_levers"] == list(
        composed.dsl_levers
    )
    assert composed.dsl_program_manifest["cli_appended_lever_factories"] == [
        "FreshFrequencyShift"
    ]
    emitted = [flag for flag, _ in composed.to_trainer_flags("unused")]
    assert verify_launch_manifest(composed.dsl_program_manifest, emitted)[0] is True


def test_typed_v9_control_and_treatment_share_matched_basis() -> None:
    control = _base().with_dsl_lever_factories("FreShInitControl")
    treatment = _base().with_dsl_lever_factories("FreshFrequencyShift")
    control_flags = control.typed.to_program().flag_dict()
    treatment_flags = treatment.typed.to_program().flag_dict()

    for flag in (
        "--activation",
        "--siren-init",
        "--self-orient",
        "--n-dir-freqs",
        "--freq-across",
        "--freq-along",
        "--seed-islands",
        "--seed-island-eased",
        "--witness-alone-island-loss",
    ):
        assert control_flags[flag] == treatment_flags[flag]
    assert control_flags["--fresh-init-control"] is True
    assert "--fresh-init" not in control_flags
    assert treatment_flags["--fresh-init"] is True
    assert "--fresh-init-control" not in treatment_flags


def test_typed_purpose_rebind_updates_hash_without_changing_argv() -> None:
    base = _base().with_dsl_lever_factories("FreshFrequencyShift")
    rebound = base.with_purpose("FreSh matched slice")

    assert rebound.purpose == "FreSh matched slice"
    assert rebound.to_trainer_flags("same") == base.to_trainer_flags("same")
    assert (
        rebound.dsl_program_manifest["typed_config_hash"]
        != base.dsl_program_manifest["typed_config_hash"]
    )
    emitted = [flag for flag, _ in rebound.to_trainer_flags("unused")]
    assert verify_launch_manifest(rebound.dsl_program_manifest, emitted)[0] is True


def test_fixed_quality_slice_composes_identically_on_both_init_arms() -> None:
    control = _base().with_dsl_lever_factories(
        "FreShInitControl", "FreShFixedQualitySlice"
    )
    treatment = _base().with_dsl_lever_factories(
        "FreshFrequencyShift", "FreShFixedQualitySlice"
    )
    for compiled in (control, treatment):
        flags = compiled.typed.to_program().flag_dict()
        assert flags["--eval-every"] == 1
        assert flags["--ckpt-every"] == 1
        assert flags["--stage-checkpoints"] is True
