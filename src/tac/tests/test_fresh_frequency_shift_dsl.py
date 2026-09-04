"""Typed-DSL regression coverage for the default-off FreSh init arm."""

from __future__ import annotations

import pytest

from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl.lever_registry import (
    lever_factories,
    name_composable_levers,
    resolve_composable_lever,
)

_CONTROL_FLAGS = {
    "--activation": "hosc",
    "--siren-init": True,
    "--self-orient": True,
    "--n-dir-freqs": 4,
    "--freq-across": 32.0,
    "--freq-along": 8.0,
    "--seed-islands": False,
    "--seed-island-eased": False,
    "--witness-alone-island-loss": False,
    "--fresh-init-control": True,
}


def test_control_and_treatment_share_exact_directional_basis() -> None:
    control = cd.FreShInitControl()
    treatment = cd.FreshFrequencyShift()
    assert control.overrides == _CONTROL_FLAGS
    shared = {flag: value for flag, value in _CONTROL_FLAGS.items() if flag != "--fresh-init-control"}
    assert {flag: treatment.overrides[flag] for flag in shared} == shared
    assert "--fresh-init-control" not in treatment.overrides
    assert treatment.overrides["--fresh-init"] is True
    assert set(treatment.overrides).difference(control.overrides) == {
        "--fresh-init",
        "--fresh-spectrum-size",
        "--fresh-sample-pairs",
        "--fresh-reference-freq-along",
        "--fresh-tangent-deficit",
        "--fresh-bias-k-min",
        "--fresh-bias-k-max",
        "--fresh-bias-k-step",
    } - {"--fresh-init-control"}


def test_fresh_factories_are_auto_registered_and_bare_name_composable() -> None:
    factories = lever_factories()
    composable = name_composable_levers()
    for name in (
        "FreShInitControl",
        "FreShFixedQualitySlice",
        "FreshFrequencyShift",
    ):
        assert name in factories
        assert name in composable
        assert isinstance(resolve_composable_lever(name), cd.Lever)
    assert factories["FreshFrequencyShift"] >= set(
        cd.FreshFrequencyShift().overrides
    )


def test_fresh_treatment_compiles_through_real_trainer_parser() -> None:
    lever = cd.FreshFrequencyShift()
    parser = cd.build_real_trainer_parser()
    argv: list[str] = ["--out-dir", "experiments/results/test_fresh_parser"]
    for flag, value in lever.overrides.items():
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{flag.removeprefix('--')}")
        else:
            argv.append(flag)
            argv.append(str(value))
    parsed = parser.parse_args(argv)
    assert parsed.fresh_init is True
    assert parsed.fresh_init_control is False
    assert parsed.self_orient is True
    assert parsed.fresh_spectrum_size == 64
    assert parsed.fresh_sample_pairs == 10
    assert parsed.fresh_tangent_deficit == pytest.approx(3.2)


def test_factory_validation_fails_before_compile() -> None:
    with pytest.raises(ValueError, match="n_dir_freqs"):
        cd.FreShInitControl(n_dir_freqs=0)
    with pytest.raises(ValueError, match="freq_along"):
        cd.FreShInitControl(freq_along=0.0)
    with pytest.raises(ValueError, match="spectrum_size"):
        cd.FreshFrequencyShift(spectrum_size=0)
    with pytest.raises(ValueError, match="bias"):
        cd.FreshFrequencyShift(bias_k_min=0.1)
    with pytest.raises(ValueError, match="divide"):
        cd.FreshFrequencyShift(bias_k_step=0.07)
    with pytest.raises(ValueError, match="eval_every"):
        cd.FreShFixedQualitySlice(eval_every=0)
    with pytest.raises(ValueError, match="ckpt_every"):
        cd.FreShFixedQualitySlice(ckpt_every=0)


@pytest.mark.parametrize(
    ("factory", "kwargs", "match"),
    [
        (cd.FreShInitControl, {"n_dir_freqs": 4.9}, "positive integer"),
        (cd.FreShInitControl, {"n_dir_freqs": True}, "positive integer"),
        (cd.FreshFrequencyShift, {"spectrum_size": 64.9}, "positive integer"),
        (cd.FreshFrequencyShift, {"sample_pairs": 10.9}, "positive integer"),
        (cd.FreShFixedQualitySlice, {"eval_every": 1.5}, "exactly 1"),
        (cd.FreShFixedQualitySlice, {"eval_every": 2}, "exactly 1"),
        (cd.FreShFixedQualitySlice, {"ckpt_every": 1.5}, "exactly 1"),
        (cd.FreShFixedQualitySlice, {"ckpt_every": 26}, "exactly 1"),
    ],
)
def test_fresh_integer_inputs_fail_closed_without_coercion(
    factory: object,
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)  # type: ignore[operator]


def test_fresh_candidate_ceiling_is_identical_in_dsl_and_runtime() -> None:
    # Three de-duplicated frequency candidates x 91 bias widths = 273.
    with pytest.raises(ValueError, match="273 candidates; maximum is 256"):
        cd.FreshFrequencyShift(bias_k_max=9.0, bias_k_step=0.1)


def test_fixed_quality_slice_changes_only_measurement_and_checkpoint_cadence() -> None:
    assert cd.FreShFixedQualitySlice().overrides == {
        "--eval-every": 1,
        "--ckpt-every": 1,
        "--stage-checkpoints": True,
    }


def test_default_v9_program_does_not_silently_enable_fresh() -> None:
    from tac.witness_dsl.spec_v9_cgauge import derive_v9_cgauge_432_config

    # epochs must satisfy the CURRICULUM EPOCH-BUDGET FEASIBILITY gate
    # (witness_autoconfig.py:2755; latest stage start is --l7-start-epoch=800):
    # a 2-epoch program is refused at compile since that gate landed, and this
    # test only inspects the compiled flag set, so use the v9 default budget.
    typed = derive_v9_cgauge_432_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n24.npz",
        num_pairs=8,
        epochs=3000,
        out_dir="experiments/results/test_fresh_default_off",
    )
    flags = typed.to_program().flag_dict()
    assert "--fresh-init" not in flags
