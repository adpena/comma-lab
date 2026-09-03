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
        # F10 satisfiability law: (dwell-1)*eval_every >= min_epochs => 7 at 25/150
        "--annulus-plateau-dwell-windows": "7",
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
    # 2026-07-17 #332 backfill: the launch config additionally composes the
    # value-neutral v9_flag_custody_rollup AFTER the expected-lever gate ran.
    assert sorted(got) == sorted(
        (*V9_CGAUGE_432_EXPECTED_LEVERS, "v9_flag_custody_rollup"))
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
        # Reconciled 2026-07-15 (merge of the derive-solver V9-constants branch): the emitted
        # DSL argv is the single scalar-value OWNER (anti-drift: the hosc 10.0-vs-3.177 incident),
        # and the value is the LawRef-derived 3.177 via hosc_beta_fireband_pin_v1 — matching the
        # sealed live launches (H4 confound hunt verified byte-identity). The prior expectation
        # (owner=v9_hosc_beta_endpoint_v1, value=8.0) is superseded.
        assert row["single_value_owner"] == "compiled_dsl_argv"
        assert row["equation_id"] == "hosc_beta_fireband_pin_v1"
        assert float(row["value"]) == float(emitted["--hosc-beta-end"]) == pytest.approx(3.177)
        assert float(row["inherited_manifest_value_replaced"]) == pytest.approx(10.0)


def test_ideal_manifest_parity_extincts_margin_msafe_duplicate_owner(ideal_ab) -> None:
    for cfg in ideal_ab:
        emitted = _argv_pairs(cfg.typed.to_program().compile_trainer_argv())
        row = cfg.constants_manifest["seg_margin_satisfice_msafe"]
        # Reconciled 2026-07-15: lever-owned constants are stamped single_value_owner =
        # "dsl_lever:<name>" by _merge_lever_constant_manifests (the #332 one-DSL-Lever-owner
        # contract); the LawRef equation stays custodied via equation_id. Prior expectation
        # (owner == the equation id) is superseded.
        assert row["single_value_owner"] == "dsl_lever:margin_band_satisficing"
        assert row["equation_id"] == "margin_band_satisficing_threshold_v1"
        assert float(row["value"]) == float(
            emitted["--seg-margin-satisfice-msafe"]
        ) == pytest.approx(0.04376363754272461)


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


# ───────────────────────── 2026-07-15 top-3 one-delta ISO configs ─────────────────────────
@pytest.fixture(scope="module")
def iso_configs():
    from tac.witness_dsl.spec_v9_cgauge import (
        compile_v9_cgauge_432_horizon_iso_launch_config,
        compile_v9_cgauge_432_step_iso_launch_config,
        compile_v9_cgauge_432_taper_off_launch_config,
        compile_v9_cgauge_ideal_mod19_launch_config,
    )

    return {
        "control": compile_v9_cgauge_ideal_mod19_launch_config(gt_cache_path=_GT),
        "taper": compile_v9_cgauge_432_taper_off_launch_config(gt_cache_path=_GT),
        "horizon": compile_v9_cgauge_432_horizon_iso_launch_config(gt_cache_path=_GT),
        "step": compile_v9_cgauge_432_step_iso_launch_config(gt_cache_path=_GT),
    }


def test_iso_configs_validate_parse_and_name_their_duty(iso_configs) -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser
    from tac.witness_dsl.spec_v9_cgauge import V9_CGAUGE_ISO_CONFIG_IDS

    expected_duty = {"taper": 78.9, "horizon": 47.3, "step": 34.2}
    assert tuple(iso_configs[k].name for k in ("taper", "horizon", "step")) == (
        V9_CGAUGE_ISO_CONFIG_IDS)
    for key, duty in expected_duty.items():
        cfg = iso_configs[key]
        assert cfg.typed.validate_program() == []
        ns = build_real_trainer_parser().parse_args(
            list(cfg.typed.to_program().compile_trainer_argv()[2:]))
        assert ns.mod_dim == 19
        assert cfg.dsl_program_manifest["iso_contract"]["duty_to_measure_percent"] == duty
        assert cfg.dsl_program_manifest["iso_contract"]["one_lever_delta"] is True


@pytest.mark.parametrize("config_id", [
    "v9_cgauge_432_taper_off",
    "v9_cgauge_432_horizon_iso",
    "v9_cgauge_432_step_iso",
])
def test_launcher_resolves_each_iso_config_id(config_id: str) -> None:
    from tools.launch_witness_run import config_family, derive_named_config

    cfg = derive_named_config(config_id, _GT, num_pairs=600, epochs=None, overfit=True)
    assert cfg.name == config_id
    assert config_family(cfg) == config_id


def test_taper_off_drops_whole_lever_and_all_four_lawrefs(iso_configs) -> None:
    control = iso_configs["control"].typed.to_program()
    treatment = iso_configs["taper"].typed.to_program()
    owned = [lever for lever in control.levers if lever.name == "dseg_aware_taper"]
    assert len(owned) == 1
    assert set(owned[0].lawrefs) == {
        "--dseg-aware-taper", "--dseg-aware-taper-strength",
        "--dseg-aware-taper-scale", "--dseg-aware-taper-floor",
    }
    # The ideal control owns taper through exactly one typed Lever; none of its
    # flags may remain as anonymous base argv after the ownership refactor.
    assert not set(owned[0].overrides) & set(iso_configs["control"].typed.base)
    assert not any(lever.name == "dseg_aware_taper" for lever in treatment.levers)
    assert not set(owned[0].overrides) & set(treatment.flag_dict())
    diff = iso_configs["taper"].dsl_program_manifest["iso_contract"]["argv_diff"]
    assert set(diff) == set(owned[0].overrides)
    assert all(after == "<ABSENT>" for _, after in diff.values())
    assert "dseg_aware_taper" not in iso_configs["taper"].dsl_program_manifest[
        "expected_active_levers"]


def test_horizon_iso_has_seven_lawrefs_receipts_and_derived_weight_consumer(iso_configs) -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    cfg = iso_configs["horizon"]
    levers = [lever for lever in cfg.typed.to_program().levers
              if lever.name == "horizon_weighted_margin"]
    assert len(levers) == 1
    lever = levers[0]
    assert len(lever.lawrefs) == len(lever.constant_manifest) == 7
    assert lever.constant_manifest["--seg-horizon-margin-weight"]["ladder_class"] == "derived_live"
    assert set(lever.runtime_receipt_schemas) == set(lever.overrides)
    assert set(lever.runtime_receipt_schemas.values()) == {"hwm_v9_stage_share_boundary.v1"}
    ns = build_real_trainer_parser().parse_args(
        list(cfg.typed.to_program().compile_trainer_argv()[2:]))
    assert ns.seg_horizon_margin_derived_live is True
    assert ns.seg_horizon_margin_weight == pytest.approx(0.15)
    assert ns.seg_horizon_margin_start_epoch == 726
    assert "horizon_weighted_margin" not in cfg.dsl_program_manifest["excluded_levers"]


def test_horizon_iso_missing_consumer_refuses() -> None:
    from pathlib import Path

    from tac.v9_provenance_gates import _trainer_consumers
    from tac.witness_dsl.curriculum_dsl import HorizonWeightedMargin
    from tac.witness_dsl.spec_v9_cgauge import _assert_iso_lever_custody

    trainer = Path("experiments/train_levelset_witness_realized_through_R_mlx.py")
    consumers = _trainer_consumers(trainer.resolve(), Path.cwd())
    consumers.pop("seg_horizon_margin_weight", None)
    lever = HorizonWeightedMargin(
        weight=0.15, start_epoch=726, window=0,
        stage_share_derived_live=True, scientific_declaration=True)
    with pytest.raises(ValueError, match=r"ISO provenance/consumer REFUSE.*weight"):
        _assert_iso_lever_custody(
            lever, trainer_path=trainer.resolve(), consumer_locations=consumers)


def test_step_iso_is_one_activation_delta_with_distinct_beta_lawref(iso_configs) -> None:
    cfg = iso_configs["step"]
    diff = cfg.dsl_program_manifest["iso_contract"]["argv_diff"]
    assert diff == {"--hosc-beta-end": ["3.177", "8.0"]}
    levers = [lever for lever in cfg.typed.to_program().levers
              if lever.name == "FEED_07b_step_native_activation"]
    assert len(levers) == 1
    beta = levers[0].constant_manifest["--hosc-beta-end"]
    assert beta["value"] == pytest.approx(8.0)
    assert beta["equation_id"] == "step_native_activation_edge_optimality_v1"
    assert cfg.constants_manifest["hosc_beta_end"]["single_value_owner"] == (
        "dsl_lever:FEED_07b_step_native_activation")
    assert "step_native_endpoint" not in cfg.dsl_program_manifest["excluded_levers"]


def test_iso_lever_factories_are_registry_mapped() -> None:
    from tac.witness_dsl.lever_registry import completeness, lever_factories

    factories = lever_factories()
    assert set(factories["DsegAwareTaper"]) == {
        "--dseg-aware-taper", "--dseg-aware-taper-strength",
        "--dseg-aware-taper-scale", "--dseg-aware-taper-floor",
    }
    assert "--seg-horizon-margin-weight" in factories["HorizonWeightedMargin"]
    assert "--seg-horizon-margin-derived-live" in factories["HorizonWeightedMargin"]
    assert "--hosc-beta-end" in factories["StepNativeActivation"]
    report = completeness()
    for flag in (
        "--dseg-aware-taper", "--seg-horizon-margin-weight",
        "--seg-horizon-margin-derived-live", "--hosc-beta-end",
    ):
        assert flag in report.mapped
        assert flag not in report.unmapped


# ───────────────────────── task #438 smoke-regime cost arm ─────────────────────────
@pytest.fixture(scope="module")
def smoke_regime_432():
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_smoke_regime_config

    return compile_v9_cgauge_432_smoke_regime_config(_GT, num_pairs=600, epochs=3000)


def test_smoke_regime_factory_actually_forces_post_event_backstops(
    smoke_regime_432, launch_432
) -> None:
    """The factory must be INVOKED and the forced values must reach the emitted
    argv — this is the test the 2026-07-15 recovery landing lacked (the flag
    ``--seg-temporal-screw-start-epoch`` is Lever-resident, not base-resident,
    so a base-only override raised at invocation)."""
    from tac.witness_dsl.spec_v9_cgauge import (
        V9_CGAUGE_432_SMOKE_REGIME_FORCED_STARTS,
    )

    def pairs(argv):
        out, toks = {}, list(argv)
        i = 0
        while i < len(toks):
            if toks[i].startswith("--"):
                if i + 1 < len(toks) and not str(toks[i + 1]).startswith("--"):
                    out[toks[i]] = str(toks[i + 1])
                    i += 2
                    continue
                out[toks[i]] = True
            i += 1
        return out

    smoke = pairs(smoke_regime_432.typed.to_program().compile_trainer_argv())
    launch = pairs(launch_432.typed.to_program().compile_trainer_argv())
    for flag, forced in V9_CGAUGE_432_SMOKE_REGIME_FORCED_STARTS.items():
        assert smoke[flag] == str(forced), flag
        assert launch[flag] != str(forced), flag
    changed = {key for key in smoke if smoke.get(key) != launch.get(key)}
    assert changed == set(V9_CGAUGE_432_SMOKE_REGIME_FORCED_STARTS)


def test_smoke_regime_is_typed_hash_distinct_validated_and_parseable(
    smoke_regime_432, launch_432
) -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    assert smoke_regime_432.typed.validate_program() == []
    assert (
        smoke_regime_432.typed.typed_config_hash()
        != launch_432.typed.typed_config_hash()
    )
    argv = list(smoke_regime_432.typed.to_program().compile_trainer_argv())
    ns = build_real_trainer_parser().parse_args(argv[2:])
    assert ns.lane_band_start_epoch == 1
    assert ns.seg_chroma_boundary_start_epoch == 1
    assert ns.seg_temporal_screw_start_epoch == 1


def test_smoke_regime_manifests_declare_non_promotable_forcing(smoke_regime_432) -> None:
    manifest = smoke_regime_432.dsl_program_manifest
    assert manifest["program_name"] == "v9_cgauge_432_smoke_regime"
    assert manifest["smoke_regime_forced_starts"] == {
        "--lane-band-start-epoch": 1,
        "--seg-chroma-boundary-start-epoch": 1,
        "--seg-temporal-screw-start-epoch": 1,
    }
    constants_row = smoke_regime_432.constants_manifest["smoke_regime_forced_starts"]
    assert constants_row["ladder_class"] == "measurement_apparatus_forced"
    assert "NON-PROMOTABLE" in constants_row["note"]
