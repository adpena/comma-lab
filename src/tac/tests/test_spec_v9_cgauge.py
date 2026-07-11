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


# ───────────────────────── task #432 coherent-schedule arm ─────────────────────────
@pytest.fixture(scope="module")
def launch_432():
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config

    return compile_v9_cgauge_432_launch_config(_GT, num_pairs=600, epochs=3000)


def test_432_builds_validates_and_parses(launch_432) -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    assert launch_432.name == "v9_cgauge_432"
    assert launch_432.typed.validate_program() == []
    argv = list(launch_432.typed.to_program().compile_trainer_argv())
    ns = build_real_trainer_parser().parse_args(argv[2:])
    assert ns.mod_dim == 19  # cgauge_whitney_moddim_v1 (17 + 2 gauge margin)
    assert ns.seg_phase_advect_weight == pytest.approx(0.4)
    assert ns.seg_phase_advect_start_epoch == 726


def test_432_delta_vs_v9_base_is_exactly_the_intended_set(launch_432, compiled) -> None:
    """A/B cleanliness: vs the v9_cgauge base, the 432 arm changes ONLY mod-dim +
    the T1 lever's explicit companion flags + the amber stability values."""
    _, base_argv = compiled

    def pairs(argv):
        d, toks = {}, list(argv)
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.startswith("--"):
                if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                    d[t] = toks[i + 1]
                    i += 2
                else:
                    d[t] = True
                    i += 1
            else:
                i += 1
        return d

    new = pairs(launch_432.typed.to_program().compile_trainer_argv())
    old = pairs(base_argv)
    skip = {"--out-dir"}
    added = {k for k in new if k not in old and k not in skip}
    dropped = {k for k in old if k not in new and k not in skip}
    changed = {k for k in new if k in old and k not in skip and new[k] != old[k]}
    # T1 companion flags become explicit when the lever composes (values = trainer
    # defaults, verified by the parse test above).
    assert added <= {"--seg-phase-advect-classes", "--seg-phase-advect-band",
                     "--seg-phase-advect-gap-xi", "--seg-phase-advect-ref",
                     "--pose-finish-engage-on", "--grad-normalize",
                     "--pose-grad-coeff-max"}
    assert dropped == set()
    assert changed <= {"--mod-dim", "--grad-clip", "--pose-grad-coeff-max"}
    assert new["--mod-dim"] == "19"


def test_432_expected_levers_include_t1_and_pose_gate(launch_432) -> None:
    from tac.witness_dsl.spec_v9_cgauge import V9_CGAUGE_432_EXPECTED_LEVERS

    got = tuple(launch_432.dsl_levers)
    assert sorted(got) == sorted(V9_CGAUGE_432_EXPECTED_LEVERS)
    assert "phase_advection_consistency" in got
    assert "pose_finish_conditioning_gate" in got


def test_432_cascade_realization_mirrors_430_stages() -> None:
    from tac.witness_control.schedule_backtest import CASCADE_STAGES
    from tac.witness_dsl.spec_v9_cgauge import V9_CGAUGE_432_CASCADE_REALIZATION

    stage_names = {s["name"] for s in CASCADE_STAGES}
    realized = {k for k in V9_CGAUGE_432_CASCADE_REALIZATION if not k.startswith("_")}
    assert realized == stage_names  # every #430 stage dispositioned, none invented
    for s in CASCADE_STAGES:
        real = V9_CGAUGE_432_CASCADE_REALIZATION[s["name"]]
        assert set(map(str, real.get("bundle", {}))) >= set(s["bundle"]) or s["name"] in (
            "boundary_form", "finish"), s["name"]


def test_432_t1_start_epoch_is_lawref_derived_not_naked(launch_432) -> None:
    cm = launch_432.constants_manifest
    row = cm["seg_phase_advect_start_epoch"]
    assert row["value"] == 726
    assert row["equation_id"] == (
        "gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1")
    assert "N7 BUILD-OWED" in row["note"]  # the static-approximation honesty marker


def test_432_purpose_declares_fresh_start_containment_and_control(launch_432) -> None:
    p = launch_432.purpose
    assert "FRESH start" in p
    assert "operator-GO" in p
    assert "#205 banked mod-32 baseline" in p
    assert "0.19108282" in p


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
