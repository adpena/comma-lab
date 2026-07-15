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


def test_active_structured_and_pose_companions_are_typed_not_parser_defaults(compiled) -> None:
    _, argv = compiled
    a = list(argv)
    expected = {
        "--structured-init-thresh": "0.5",
        "--structured-init-steps": "600",
        "--structured-init-lr": "0.005",
        "--structured-init-subsample": "8192",
        "--structured-init-sdf-clip": "20.0",
        "--pose-carrier-residual-scale": "1.0",
        "--pose-carrier-s-t": "0.044",
        "--pose-carrier-s-r": "0.0",
        "--pose-carrier-pitch": "0.0",
    }
    assert "--structured-init-include-lane" in a
    for flag, value in expected.items():
        assert a[a.index(flag) + 1] == value


def test_active_controller_sensor_companions_are_typed_not_parser_defaults(compiled) -> None:
    _, argv = compiled
    a = list(argv)
    expected = {
        "--annulus-band": "2.0",
        "--annulus-bottom-k": "0.05",
        "--annulus-plateau-rel-eps": "0.0001",
        "--annulus-plateau-dwell-windows": "4",
        "--annulus-plateau-min-epochs": "150",
        "--curriculum-nucleus-within-flip": "0.5",
        "--curriculum-nucleus-min-part-frac": "0.0",
        "--jacobian-basin-k-pairs": "32",
        "--jacobian-basin-every": "4",
        "--jacobian-basin-sigma-floor": "0.0001",
        "--jacobian-basin-f-basin": "1.0",
        "--jacobian-basin-quorum-q": "0.8",
        "--seed-anneal-epochs": "0",
        "--seed-anneal-shape": "linear",
        "--seed-blend": "1.0",
        "--seed-lr": "0.02",
        "--containment-mode": "shield",
        "--containment-damp": "0.1",
        "--muon-adamw-lr": "0.0001",
    }
    affirmative = {
        "--annulus-telemetry",
        "--jacobian-basin-telemetry",
        "--jacobian-basin-t0",
        "--jacobian-basin-stratify-t",
    }
    assert affirmative.issubset(a)
    assert "--no-tail-live-mq" in a
    for flag, value in expected.items():
        assert a[a.index(flag) + 1] == value


def test_argv_parses_on_real_trainer_parser(compiled) -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    _, argv = compiled
    ns = build_real_trainer_parser().parse_args(list(argv[2:]))
    assert ns.seg_phase_advect_weight == pytest.approx(0.4)
    assert ns.mod_dim == 32  # SAFE anchor; derived-19 gated on #299/harvest
    assert ns.adam_beta2 == pytest.approx(0.999)
    assert ns.annulus_telemetry is True
    assert ns.jacobian_basin_telemetry is True
    assert ns.jacobian_basin_k_pairs == 32
    assert ns.jacobian_basin_every == 4
    assert ns.tail_live_mq is False


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


# ───────────────────────── 2026-07-13 event-native ideal + family A/B ─────────────────────────
@pytest.fixture(scope="module")
def ideal_ab():
    from tac.witness_dsl.spec_v9_cgauge import (
        compile_v9_cgauge_ideal_mod19_launch_config,
        compile_v9_cgauge_ideal_mod32_launch_config,
    )

    return (
        compile_v9_cgauge_ideal_mod19_launch_config(gt_cache_path=_GT),
        compile_v9_cgauge_ideal_mod32_launch_config(gt_cache_path=_GT),
    )


@pytest.fixture(scope="module")
def ideal_sr_ab():
    from tac.witness_dsl.spec_v9_cgauge import (
        compile_v9_cgauge_ideal_mod19_launch_config,
        compile_v9_cgauge_ideal_mod19_sR_launch_config,
    )

    return (
        compile_v9_cgauge_ideal_mod19_launch_config(gt_cache_path=_GT),
        compile_v9_cgauge_ideal_mod19_sR_launch_config(gt_cache_path=_GT),
    )


def _argv_pairs(argv):
    from tac.witness_autoconfig import _crucible_v7_argv_pairs

    return dict(_crucible_v7_argv_pairs(argv))


def test_ideal_programs_validate_parse_and_actuate(ideal_ab) -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    for cfg, dim in zip(ideal_ab, (19, 32), strict=True):
        assert cfg.typed.validate_program() == []
        argv = tuple(cfg.typed.to_program().compile_trainer_argv())
        ns = build_real_trainer_parser().parse_args(list(argv[2:]))
        assert ns.mod_dim == dim
        assert ns.seg_form_unify_tau is True
        assert ns.tau_advance_mode == "event"
        assert ns.eikonal_weight == pytest.approx(0.01)
        assert ns.eikonal_weight_end == pytest.approx(0.05)
        assert ns.stage_transition_rewarmup_epochs == 14
        assert ns.stage_transition_reset_moments is True
        assert ns.length_sigma_matrix == "fitted-20260707"
        assert ns.seg_subpix_edge_weight_source == "pa_flipmass"
        assert ns.closed_loop_control is True
        assert ns.stage_checkpoints is True
        assert ns.verdict_pairs == 0
        assert ns.micro_batch_pairs == 1
        assert ns.margin_saliency_weight == pytest.approx(1.0)
        assert ns.margin_saliency_start_epoch == 0


def test_ideal_mod19_vs_mod32_scientific_argv_diff_is_only_mod_dim(ideal_ab) -> None:
    p19 = _argv_pairs(ideal_ab[0].typed.to_program().compile_trainer_argv())
    p32 = _argv_pairs(ideal_ab[1].typed.to_program().compile_trainer_argv())
    # Custody paths differ by necessity; after removing them the family dimension is
    # the sole scientific treatment delta.
    p19.pop("--out-dir")
    p32.pop("--out-dir")
    changed = {k for k in set(p19) | set(p32) if p19.get(k) != p32.get(k)}
    assert changed == {"--mod-dim"}
    assert p19["--mod-dim"] == "19"
    assert p32["--mod-dim"] == "32"


def test_ideal_manifest_parity_extincts_hosc_duplicate_owner(ideal_ab) -> None:
    for cfg in ideal_ab:
        emitted = _argv_pairs(cfg.typed.to_program().compile_trainer_argv())
        row = cfg.constants_manifest["hosc_beta_end"]
        assert row["single_value_owner"] == "compiled_dsl_argv"
        assert float(row["value"]) == float(emitted["--hosc-beta-end"]) == pytest.approx(3.177)
        assert float(row["inherited_manifest_value_replaced"]) == pytest.approx(10.0)


def test_ideal_includes_safe_set_and_composed_margin(ideal_ab) -> None:
    expected = {
        "unified_tau_eikonal_hold",
        "n292_closed_loop_eikonal_control",
        "R7_beta2_window_rewarmup",
        "FEED_08a_length_sigma",
        "tie_locus_displacement",
        # margin_band_satisficing composed 2026-07-13 after its provenance fix landed (a79f5d68cd):
        # ON in core + both A/B arms (identical → mod-dim FAMILY A/B stays unconfounded).
        "margin_band_satisficing",
        # C1 shared w-only control; the S_R treatment composes one additional
        # source-selector Lever over this exact program.
        "margin_saliency",
    }
    for cfg in ideal_ab:
        names = set(cfg.dsl_levers)
        assert expected <= names
        assert "margin_band_satisficing" in names
        manifest = cfg.dsl_program_manifest
        assert manifest["ab_decision_rule"]["threshold"] == pytest.approx(0.02)
        assert manifest["held"] is True and manifest["operator_go_required"] is True


def test_ideal_sr_treatment_is_exactly_one_dsl_flag_over_matched_control(ideal_sr_ab) -> None:
    control, treatment = ideal_sr_ab
    p0 = _argv_pairs(control.typed.to_program().compile_trainer_argv())
    p1 = _argv_pairs(treatment.typed.to_program().compile_trainer_argv())
    p0.pop("--out-dir")
    p1.pop("--out-dir")
    changed = {
        k for k in set(p0) | set(p1)
        if (k in p0, p0.get(k)) != (k in p1, p1.get(k))
    }
    assert changed == {"--margin-saliency-reachability"}
    assert "--margin-saliency-reachability" not in p0
    assert p1["--margin-saliency-reachability"] is None
    assert p0["--micro-batch-pairs"] == p1["--micro-batch-pairs"] == "1"


def test_ideal_sr_manifest_validate_parser_and_lawref(ideal_sr_ab) -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser
    from tac.witness_dsl.spec_v9_cgauge import (
        V9_CGAUGE_IDEAL_SR_EQUATION_ID,
        V9_CGAUGE_IDEAL_SR_EXPECTED_ADDITION,
    )

    control, treatment = ideal_sr_ab
    assert control.typed.validate_program() == treatment.typed.validate_program() == []
    ns0 = build_real_trainer_parser().parse_args(
        list(control.typed.to_program().compile_trainer_argv()[2:]))
    ns1 = build_real_trainer_parser().parse_args(
        list(treatment.typed.to_program().compile_trainer_argv()[2:]))
    assert ns0.margin_saliency_reachability is False
    assert ns1.margin_saliency_reachability is True
    assert ns0.margin_saliency_weight == ns1.margin_saliency_weight == pytest.approx(1.0)
    manifest = treatment.dsl_program_manifest
    assert V9_CGAUGE_IDEAL_SR_EXPECTED_ADDITION in manifest["expected_active_levers"]
    assert V9_CGAUGE_IDEAL_SR_EXPECTED_ADDITION not in manifest["excluded_levers"]
    assert manifest["sr_ab_contract"]["equation_id"] == V9_CGAUGE_IDEAL_SR_EQUATION_ID
    assert treatment.constants_manifest["margin_saliency_reachability"]["equation_id"] == (
        V9_CGAUGE_IDEAL_SR_EQUATION_ID)


def test_sr_factory_is_mapped_and_activation_duty_is_preserved(tmp_path) -> None:
    from tac.witness_dsl.activation_ledger import duty_to_measure
    from tac.witness_dsl.lever_registry import completeness, lever_factories

    report = completeness()
    factories = lever_factories()
    assert factories["MarginSaliencyReachability"] == frozenset(
        {"--margin-saliency-reachability"})
    assert "--margin-saliency-reachability" in report.mapped
    assert "--margin-saliency-reachability" not in report.unmapped
    assert "MarginSaliencyReachability" in duty_to_measure(
        path=tmp_path / "empty-activation-ledger.jsonl")


def test_ideal_sr_has_zero_naked_schedule_epochs(ideal_sr_ab) -> None:
    from pathlib import Path

    from tools import schedule_provenance_gate as gate

    treatment = ideal_sr_ab[1]
    trainer = Path("experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    verdicts = gate.classify_launch(
        list(treatment.to_trainer_flags("OUT")),
        registry=gate.schedule_when_flags(trainer),
        manifest_keys=set(treatment.constants_manifest),
        governance=treatment.schedule_governance,
        event_registry=gate.event_start_flags(trainer),
    )
    ok, violations, table = gate.gate_report(verdicts)
    assert ok, table
    assert violations == []


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
