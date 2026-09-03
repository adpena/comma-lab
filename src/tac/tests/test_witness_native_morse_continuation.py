from __future__ import annotations

import math

import pytest

from tac.canonical_equations.v9_hosc_beta_endpoint_20260715 import (
    resolve_v9_hosc_beta_endpoint,
    v9_hosc_beta_endpoint,
)
from tac.canonical_equations.witness_native_morse_continuation_20260715 import (
    derive_morse_continuation_controls,
)
from tac.witness_dsl.curriculum_dsl import (
    WitnessNativeMorseContinuationSchedule,
    build_real_trainer_parser,
    schedule_primitive_kinds,
)


def test_v9_hosc_endpoint_is_single_derived_dyadic_path() -> None:
    resolved = resolve_v9_hosc_beta_endpoint()
    assert resolved.equation_id == "v9_hosc_beta_endpoint_v1"
    assert resolved.value == 8.0
    assert resolved.fallback_used is False
    assert v9_hosc_beta_endpoint(1.0, 3) == 8.0
    with pytest.raises(ValueError, match="positive integer"):
        v9_hosc_beta_endpoint(1.0, 2.5)


def test_morse_continuation_derives_all_three_legacy_scalars() -> None:
    controls = derive_morse_continuation_controls(
        trust_region_kl=2.0e-4,
        fisher_curvature_upper=25.0,
        m_safe=0.039180326461791926,
    )
    assert controls.muon_lr == pytest.approx(math.sqrt(4.0e-4 / 25.0))
    assert controls.l7_mult == 0.0
    assert controls.l7_threshold == pytest.approx(0.039180326461791926)


def test_schedule_is_dsl_native_and_real_argparse_compatible() -> None:
    schedule = WitnessNativeMorseContinuationSchedule(
        trust_region_kl=2.0e-4,
        fisher_curvature_upper=25.0,
    )
    assert not schedule.validate()
    assert "MorseContinuationSchedule" in schedule_primitive_kinds()
    assert schedule.flags()["--l7-mult"] == 0.0
    assert schedule.flags()["--l7-threshold"] == pytest.approx(
        0.04376363754272461
    )
    assert schedule.canonical_manifest()["equation_id"] == (
        "witness_native_morse_continuation_v1"
    )

    argv: list[str] = ["--out-dir", "unused-test-output"]
    for flag, value in schedule.flags().items():
        argv.extend((flag, str(value)))
    parsed = build_real_trainer_parser().parse_args(argv)
    assert parsed.muon_lr == pytest.approx(schedule.muon_lr)
    assert parsed.l7_mult == 0.0
    assert parsed.l7_threshold == pytest.approx(schedule.m_safe)


@pytest.mark.parametrize(
    ("delta", "curvature", "match"),
    [
        (0.0, 1.0, "trust_region_kl"),
        (1.0, 0.0, "fisher_curvature_upper"),
        (float("nan"), 1.0, "trust_region_kl"),
    ],
)
def test_schedule_refuses_missing_or_invalid_checkpoint_geometry(
    delta: float, curvature: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        WitnessNativeMorseContinuationSchedule(
            trust_region_kl=delta,
            fisher_curvature_upper=curvature,
        )
