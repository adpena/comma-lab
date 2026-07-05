# SPDX-License-Identifier: MIT
"""Tests for the #302 curriculum-derivation symposium landings (2026-07-05):
the four canonical-equation builders/callables in
``tac.canonical_equations.curriculum_derivation_laws_20260705`` and the ``CurriculumGauge``
append in ``tac.witness_dsl.gauge``. All $0/local; no trainer launch."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.canonical_equations.curriculum_derivation_laws_20260705 import (
    HANDOFF_MIN_STAGE_EP,
    HANDOFF_PLATEAU_REL_EPS,
    build_curriculum_handoff_critical_nucleus_v1,
    build_ema_window_pi_group_v1,
    build_muon_switch_conditioning_criterion_v1,
    build_rewarmup_beta2_memory_window_v1,
    ema_window_steps,
    handoff_ready,
    min_rewarmup_epochs,
    muon_switch_ready,
    pi_ema,
)
from tac.witness_dsl.gauge import (
    COMPONENT_GAUGES,
    CURRICULUM_TRAINER_FLAGS,
    CurriculumGauge,
    GaugeComponent,
    component_of,
    curriculum_trainer_flags,
)

_TRAINER = Path(__file__).resolve().parents[3] / (
    "experiments/train_levelset_witness_realized_through_R_mlx.py")


# --------------------------- law 1: handoff_ready ---------------------------

def test_handoff_fires_when_all_conditions_met():
    assert handoff_ready(-5e-5, 300, [True, True, True, True, True]) is True


def test_handoff_refuses_mid_descent_slope():
    # the C1 anchor: rel slope -8.22e-4 at ep151 must NOT fire at the recalibrated eps
    assert handoff_ready(-8.22e-4, 300, [True] * 5) is False


def test_handoff_refuses_below_min_stage():
    assert handoff_ready(-5e-5, HANDOFF_MIN_STAGE_EP - 1, [True] * 5) is False


def test_handoff_refuses_sub_nucleus_class():
    # the #205 anchor: lane part_frac 0 => never fire regardless of plateau
    assert handoff_ready(-1e-6, 999, [True, False, True, True, True]) is False


def test_handoff_eps_boundary_inclusive():
    assert handoff_ready(HANDOFF_PLATEAU_REL_EPS, HANDOFF_MIN_STAGE_EP, [True]) is True


# --------------------------- law 2: EMA window ------------------------------

def test_ema_window_steps_at_house_value():
    assert ema_window_steps(0.997) == pytest.approx(333.333, rel=1e-3)


def test_ema_window_steps_rejects_out_of_range():
    with pytest.raises(ValueError):
        ema_window_steps(1.0)
    with pytest.raises(ValueError):
        ema_window_steps(0.0)


def test_pi_ema_run2_finisher_is_tiny():
    # 0.997 over a 274-ep x 75-step finisher: ~1.6% coverage (the memo's number)
    assert pi_ema(0.997, 274 * 75) == pytest.approx(0.0162, rel=0.02)


def test_pi_ema_finisher_value_lands_in_polyak_band():
    # rho=0.9995 => 2000-step window; over 20550 steps => ~0.097 (~the 0.1-0.3 band edge)
    assert 0.05 < pi_ema(0.9995, 274 * 75) < 0.3


def test_pi_ema_rejects_nonpositive_stage():
    with pytest.raises(ValueError):
        pi_ema(0.997, 0)


# --------------------------- law 3: muon switch -----------------------------

def test_muon_switch_requires_both():
    assert muon_switch_ready(True, [True, True]) is True
    assert muon_switch_ready(False, [True, True]) is False
    assert muon_switch_ready(True, [True, False]) is False


# --------------------------- law 4: rewarmup window -------------------------

def test_min_rewarmup_epochs_run2_config():
    # beta2 0.999, 75 steps/ep => 1000-step memory => 14 epochs; run-2's 20 satisfies it
    assert min_rewarmup_epochs(0.999, 75) == 14
    assert min_rewarmup_epochs(0.999, 75) <= 20
    # the cert's earlier 8-ep value did NOT
    assert min_rewarmup_epochs(0.999, 75) > 8


def test_min_rewarmup_epochs_rejects_bad_inputs():
    with pytest.raises(ValueError):
        min_rewarmup_epochs(1.0, 75)
    with pytest.raises(ValueError):
        min_rewarmup_epochs(0.999, 0)


# --------------------------- builders --------------------------------------

@pytest.mark.parametrize("build,eq_id", [
    (build_curriculum_handoff_critical_nucleus_v1, "curriculum_handoff_critical_nucleus_v1"),
    (build_ema_window_pi_group_v1, "ema_window_pi_group_v1"),
    (build_muon_switch_conditioning_criterion_v1, "muon_switch_conditioning_criterion_v1"),
    (build_rewarmup_beta2_memory_window_v1, "rewarmup_beta2_memory_window_v1"),
])
def test_builders_construct_valid_equations(build, eq_id):
    eq = build()
    assert eq.equation_id == eq_id
    assert eq.empirical_anchors, "every law carries at least one anchor"
    assert eq.canonical_consumers and eq.canonical_producers


def test_rewarmup_law_is_marked_provisional():
    eq = build_rewarmup_beta2_memory_window_v1()
    assert "PROVISIONAL" in str(eq.domain_of_validity.get("note", ""))


# --------------------------- CurriculumGauge -------------------------------

def test_pr95_echo_is_byte_identical():
    assert curriculum_trainer_flags(CurriculumGauge.PR95_ECHO) == ()


def test_derived_native_emits_expected_flags():
    assert curriculum_trainer_flags(CurriculumGauge.DERIVED_NATIVE) == (
        "--seed-anneal-epochs", "275",
        "--persistence-warmup-epochs", "275",
        "--ema-decay-finisher", "0.9995",
    )


def test_derived_native_flags_exist_in_trainer_argparse():
    # never-invent-flags: every emitted flag name appears as an add_argument in the live trainer
    src = _TRAINER.read_text(encoding="utf-8", errors="replace")
    for flag in ("--seed-anneal-epochs", "--persistence-warmup-epochs", "--ema-decay-finisher",
                 # (#302 build) HANDOFF_NUCLEUS chart flags
                 "--curriculum-event-triggered", "--curriculum-plateau-rel-eps",
                 "--curriculum-plateau-windows", "--curriculum-min-stage-epochs",
                 "--curriculum-nucleus-guard", "--curriculum-reanchor-levers",
                 "--handoff-readiness-telemetry"):
        assert f'"{flag}"' in src, f"{flag} missing from trainer argparse"


def test_unified_energy_is_design_stage_fail_closed():
    with pytest.raises(NotImplementedError):
        curriculum_trainer_flags(CurriculumGauge.UNIFIED_ENERGY)


def test_component_registration():
    assert COMPONENT_GAUGES[GaugeComponent.CURRICULUM] is CurriculumGauge
    assert component_of(CurriculumGauge.PR95_ECHO) is GaugeComponent.CURRICULUM


def test_trainer_flags_dict_covers_non_design_charts():
    assert set(CURRICULUM_TRAINER_FLAGS) == {
        CurriculumGauge.PR95_ECHO, CurriculumGauge.DERIVED_NATIVE,
        CurriculumGauge.HANDOFF_NUCLEUS}   # (#302 build) the completed CE->tau hand-off chart
