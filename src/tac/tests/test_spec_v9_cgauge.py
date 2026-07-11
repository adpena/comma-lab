# SPDX-License-Identifier: MIT
"""SPEC_v9_cgauge behavior tests: builds, validates fail-closed, compiles, parses
clean on the REAL trainer parser (never-invent-flags), provenance manifest complete."""
from __future__ import annotations

import pytest

from tac.witness_dsl.spec_v9_cgauge import (
    V9_CGAUGE_PROVENANCE,
    compile_v9_cgauge_config,
)

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


@pytest.fixture(scope="module")
def compiled():
    return compile_v9_cgauge_config(_GT, num_pairs=600, epochs=3000)


def test_builds_named_v9_cgauge_and_validates(compiled) -> None:
    typed, argv = compiled
    assert typed.name == "v9_cgauge"
    assert typed.validate_program() == []  # fail-closed gate re-check


def test_t1_phase_term_is_on_at_derived_values(compiled) -> None:
    _, argv = compiled
    a = list(argv)
    assert a[a.index("--seg-phase-advect-weight") + 1] == "0.4"
    assert a[a.index("--seg-phase-advect-start-epoch") + 1] == "726"


def test_argv_parses_on_real_trainer_parser(compiled) -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    _, argv = compiled
    ns = build_real_trainer_parser().parse_args(list(argv[2:]))
    assert ns.seg_phase_advect_weight == pytest.approx(0.4)
    assert ns.mod_dim == 32  # SAFE anchor; derived-19 gated on #299/harvest
    assert ns.adam_beta2 == pytest.approx(0.999)


def test_inherits_v752_self_orient_off_trunk(compiled) -> None:
    _, argv = compiled
    a = list(argv)
    assert "--self-orient" not in a  # owed-16 refuted front-end stays dropped
    assert "--length-sigma-matrix" not in a  # W-1: sigma_cc' stays rung-1b
    assert a[a.index("--max-bank-freq") + 1] == "64"
    assert "--lane-band-dash-comb" in a  # the C2-violation cure term


def test_provenance_manifest_rows_are_ladder_and_form_complete() -> None:
    from tac.canonical_equations.cgauge_master_action_20260711 import VALUE_FORMS

    rungs = {"derived_live", "derived_at_config", "measured_anchor", "hardcoded_waiver"}
    assert len(V9_CGAUGE_PROVENANCE) >= 10
    for flag, row in V9_CGAUGE_PROVENANCE.items():
        assert row["rung"] in rungs, flag
        assert row["form"] in VALUE_FORMS, flag
        assert row["law"].endswith("_v1"), flag
        assert row["note"], flag


def test_purpose_declares_gates_and_containment(compiled) -> None:
    typed, _ = compiled
    assert "SEAL + n600 A/B" in typed.purpose
    assert "operator-GO" in typed.purpose
    assert "0.19108282" in typed.purpose


def test_lawref_evaluators_executable_for_223_laws() -> None:
    from tac.canonical_equations.evaluators import (
        populate_lawref_evaluators,
        resolve_equation_value,
    )

    populate_lawref_evaluators()
    assert resolve_equation_value(
        "cgauge_whitney_moddim_v1", {"intrinsic_dim": 8, "gauge_margin": 2}) == 19
    assert resolve_equation_value(
        "cgauge_curvelet_parabolic_bank_v1", {"nu_across": 64}) == pytest.approx(8.0)
    lo, hi = resolve_equation_value("cgauge_beta2_window_v1", {"steps_per_epoch": 75})
    assert lo < 0.999 < hi
