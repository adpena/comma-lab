"""Tests for the §14 schedule-design DSL primitives (task #339; operator directive
2026-07-07 "design the SCHEDULE, not just the lever set" + amendments "Schedule
should be DSL too" + "consumers must track DSL evolution").

Covers:
  * LevelPath (levels-as-paths λ(t)) round-trips through the trainer's REAL argparse
    per quantity; unsupported shapes/segments fire typed TrainerSupportGaps with a
    conservative nearest-real compile (never an invented flag);
  * StageSpec {repeat_until, priming, exit_event} — compilable halves emit real
    flags, un-compilable halves surface named gaps;
  * OperationalSchedule (verdict/telemetry/reorient cadences) compiles; per-stage
    verdict overrides fire the gap and compile the DENSEST cadence conservatively;
  * duplicate-emitter refusal (compose, don't duplicate);
  * the AUTO-DERIVED schedule_primitive_kinds() registry + the uniform
    describe()/to_display_dict() consumer surface (the class test that keeps future
    primitives honest);
  * THE COUNCIL-SHAPED EXECUTABLE SPEC: CE → tau(event exit) → Muon(warm-start
    priming, LR geometric anneal) → repeat-finishing-block-until-dry, WITH an
    operational-schedule block (verdict sparse in CE/tau, dense in Muon) —
    constructs, validates, compiles through the real argparse, and reports its
    named gaps. This test IS the spec the council reads.

Authority: MEANS only. The frontier pointer (contest-CPU 0.19110) is UNMOVED by
anything here.
"""
from __future__ import annotations

import json

import pytest

from tac.witness_dsl import curriculum_dsl as cd

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


# ---------------------------------------------------------------------------
# helpers: parse an emitted flag dict through the trainer's REAL argparse
# ---------------------------------------------------------------------------
def _argv_from_flags(flags: dict) -> list[str]:
    """Render a flag dict exactly the way WitnessProgram.compile_trainer_argv does."""
    argv: list[str] = []
    for flag, val in flags.items():
        if val is True:
            argv.append(flag)
        elif val is False:
            argv.append(flag.replace("--", "--no-", 1))
        else:
            argv.extend([flag, str(val)])
    return argv


def _parse_real(flags: dict):
    """Round-trip: the emitted flags MUST parse through the trainer's real argparse.

    --out-dir is the trainer's only required arg; supply a placeholder."""
    ap = cd.build_real_trainer_parser()
    return ap.parse_args(["--out-dir", "_"] + _argv_from_flags(flags))


# ---------------------------------------------------------------------------
# LevelPath — per-quantity round-trips through the REAL argparse
# ---------------------------------------------------------------------------
def test_levelpath_softmax_temp_geometric_round_trips():
    lp = cd.LevelPath("softmax_temp", (cd.PathSegment("geometric", 1.0, 0.05),),
                      anneal_epochs=1500)
    assert lp.validate() == []
    ns = _parse_real(lp.flags())
    assert ns.softmax_temp_start == 1.0
    assert ns.softmax_temp_end == 0.05
    assert ns.tau_anneal_shape == "geometric"
    assert ns.anneal_epochs == 1500
    assert lp.support_gaps() == ()  # fully trainer-supported → NO gap


def test_levelpath_softmax_temp_cosine_hold_emits_hold_frac():
    lp = cd.LevelPath("softmax_temp",
                      (cd.PathSegment("cosine_hold", 1.0, 0.05, hold_frac=0.7),))
    assert lp.validate() == []
    ns = _parse_real(lp.flags())
    assert ns.tau_anneal_shape == "cosine_hold"
    assert ns.tau_hold_frac == 0.7


def test_levelpath_hosc_beta_cosine_round_trips():
    lp = cd.LevelPath("hosc_beta", (cd.PathSegment("cosine", 1.0, 4.0),))
    ns = _parse_real(lp.flags())
    assert ns.hosc_beta == 1.0 and ns.hosc_beta_end == 4.0
    assert ns.hosc_beta_anneal == "cosine"
    assert lp.support_gaps() == ()


def test_levelpath_constant_is_a_decision_not_a_default():
    # §14 axis 2: constancy must be a DECISION — a constant path emits start==end.
    lp = cd.LevelPath("hosc_beta", (cd.PathSegment("constant", 4.0),))
    f = lp.flags()
    assert f["--hosc-beta"] == 4.0 and f["--hosc-beta-end"] == 4.0
    _parse_real(f)


def test_levelpath_lr_constant_disables_schedule():
    lp = cd.LevelPath("lr", (cd.PathSegment("constant", 1e-3),))
    f = lp.flags()
    assert f["--lr-schedule"] is False  # BooleanOptionalAction → --no-lr-schedule
    ns = _parse_real(f)
    assert ns.lr == 1e-3 and ns.lr_schedule is False


def test_levelpath_ema_decay_two_segment_piecewise_round_trips():
    # the §14 "EMA per stage" π-group: .997 bulk → .9995 finisher from ep726.
    lp = cd.LevelPath("ema_decay", (cd.PathSegment("constant", 0.997, epochs=726),
                                    cd.PathSegment("constant", 0.9995)))
    ns = _parse_real(lp.flags())
    assert ns.ema_decay == 0.997
    assert ns.ema_decay_finisher == 0.9995
    assert ns.ema_decay_finisher_start_epoch == 726
    assert lp.support_gaps() == ()  # the ONE multi-segment form with trainer support


def test_levelpath_eikonal_step_round_trips():
    lp = cd.LevelPath("eikonal_weight", (cd.PathSegment("step", 0.05, 0.10),),
                      stage="tau_softplus")
    ns = _parse_real(lp.flags())
    assert ns.eikonal_weight == 0.05 and ns.eikonal_weight_end == 0.10
    assert lp.support_gaps() == ()  # eikonal_weight is stage-anchored → stage= is genuine


def test_levelpath_muon_lr_frac_round_trips():
    lp = cd.LevelPath("muon_lr_frac", (cd.PathSegment("cosine", 1.0, 0.1),), stage="muon")
    ns = _parse_real(lp.flags())
    assert ns.muon_lr_final_frac == 0.1
    assert lp.support_gaps() == ()


# ---------------------------------------------------------------------------
# LevelPath — TrainerSupportGap fires for deliberately-unsupported paths
# ---------------------------------------------------------------------------
def test_levelpath_geometric_lr_fires_gap_and_compiles_nearest_cosine():
    lp = cd.LevelPath("lr", (cd.PathSegment("geometric", 1e-3, 1e-4),))
    f = lp.flags()
    # conservative nearest-real compile: cosine with the SAME endpoints, real flags only
    assert f == {"--lr": 1e-3, "--lr-end": 1e-4, "--lr-schedule": True}
    _parse_real(f)
    (gap,) = lp.support_gaps()
    assert gap.axis == "levels_as_paths"
    assert "geometric" in gap.requirement
    assert "cosine" in gap.nearest_real_compilation
    # never-invent-flags: the proposal is text, NOT an emitted flag
    assert not any(tok.startswith("--") and tok in gap.flag_proposal for tok in f)


def test_levelpath_three_segments_fires_gap_compiles_first():
    lp = cd.LevelPath("softmax_temp", (cd.PathSegment("cosine", 1.0, 0.3),
                                       cd.PathSegment("constant", 0.3),
                                       cd.PathSegment("cosine", 0.3, 0.05)))
    f = lp.flags()
    assert f["--softmax-temp-start"] == 1.0 and f["--softmax-temp-end"] == 0.3
    gaps = lp.support_gaps()
    assert any("3-segment" in g.requirement for g in gaps)


def test_levelpath_stage_scoping_on_global_quantity_fires_gap():
    lp = cd.LevelPath("softmax_temp", (cd.PathSegment("cosine", 1.0, 0.05),),
                      stage="muon")
    assert any("scoped to stage" in g.requirement for g in lp.support_gaps())


def test_levelpath_unknown_quantity_fails_closed():
    lp = cd.LevelPath("per_class_lambda", (cd.PathSegment("cosine", 0.0, 1.0),))
    assert any("unknown quantity" in p for p in lp.validate())
    with pytest.raises(ValueError):
        lp.flags()


def test_levelpath_geometric_needs_positive_endpoints():
    lp = cd.LevelPath("softmax_temp", (cd.PathSegment("geometric", 1.0, 0.0),))
    assert any("positive" in p for p in lp.validate())


# ---------------------------------------------------------------------------
# StageSpec — priming / exit events / repetition
# ---------------------------------------------------------------------------
def test_stagespec_muon_priming_round_trips():
    sp = cd.StageSpec("muon", "--muon-start-epoch", 726,
                      priming=cd.Priming(warm_start_momentum=True))
    ns = _parse_real(sp.flags())
    assert ns.muon_start_epoch == 726
    assert ns.muon_warm_start_momentum is True
    assert sp.validate() == [] and sp.support_gaps() == ()


def test_stagespec_warm_start_on_non_muon_stage_is_a_violation():
    sp = cd.StageSpec("tau_softplus", "--tau-softplus-start-epoch", 300,
                      priming=cd.Priming(warm_start_momentum=True))
    assert any("AdamW->Muon switch" in p for p in sp.validate())


def test_stagespec_entry_init_priming_round_trips_on_entry_stage():
    sp = cd.StageSpec("CE", None, None,
                      priming=cd.Priming(structured_init=True, lane_prior_phi1=True,
                                         siren_init=True, finer_bias_k=10.0))
    ns = _parse_real(sp.flags())
    assert ns.structured_init is True and ns.lane_prior_phi1 is True
    assert ns.siren_init is True
    assert ns.finer_bias_init is True and ns.finer_bias_k == 10.0
    assert sp.support_gaps() == ()  # entry stage → init priming is genuine


def test_stagespec_mid_run_init_priming_fires_gap():
    sp = cd.StageSpec("muon", "--muon-start-epoch", 726,
                      priming=cd.Priming(finer_bias_k=10.0))
    assert any(g.axis == "priming" and "MID-RUN" in g.requirement
               for g in sp.support_gaps())


def test_stagespec_exit_event_nucleus_plateau_round_trips():
    sp = cd.StageSpec("tau_softplus", "--tau-softplus-start-epoch", 300,
                      exit_event=cd.ExitEvent("nucleus_guarded_plateau",
                                              min_stage_epochs=150, rel_eps=1e-4,
                                              windows=4, within_flip=0.5))
    ns = _parse_real(sp.flags())
    assert ns.curriculum_event_triggered is True
    assert ns.curriculum_nucleus_guard is True
    assert ns.curriculum_min_stage_epochs == 150
    assert ns.curriculum_plateau_rel_eps == 1e-4
    assert ns.curriculum_plateau_windows == 4
    assert ns.curriculum_nucleus_within_flip == 0.5
    assert sp.support_gaps() == ()  # the #315 controller consumes this — no gap


def test_stagespec_marginal_dseg_floor_exit_fires_gap_and_emits_no_flags():
    ev = cd.ExitEvent("marginal_dseg_floor", floor=1e-5, cap_epoch=1000)
    sp = cd.StageSpec("muon", "--muon-start-epoch", 726, exit_event=ev)
    assert ev.flags() == {}  # conservative compile: the fixed boundary IS the exit
    (gap,) = sp.support_gaps()
    assert gap.axis == "exit_events"
    assert "marginal_dseg_floor" in gap.requirement
    assert "cap_epoch=1000" in gap.nearest_real_compilation


def test_exit_event_marginal_kind_requires_floor():
    assert any("floor" in p for p in cd.ExitEvent("lever_exhaustion").validate())


def test_repeat_until_fires_gap_with_deterministic_bound():
    ru = cd.RepeatUntil("marginal_dseg_floor", ("muon", "muon_leap"),
                        block_epochs=100, max_repeats=4, floor=1e-5)
    assert ru.validate() == []
    assert ru.conservative_epoch_bound() == 400
    sp = cd.StageSpec("muon", "--muon-start-epoch", 726, repeat_until=ru)
    gaps = [g for g in sp.support_gaps() if g.axis == "stage_repetition"]
    assert len(gaps) == 1
    assert "400 epochs" in gaps[0].nearest_real_compilation
    assert "deterministic" in gaps[0].notes


def test_repeat_until_unbounded_is_refused():
    # deterministic-reproducibility spine: repetition MUST carry an explicit bound.
    ru = cd.RepeatUntil("exhaustion", ("muon",), block_epochs=100, max_repeats=0)
    assert any("max_repeats" in p for p in ru.validate())


# ---------------------------------------------------------------------------
# OperationalSchedule (operator amendment: "Schedule should be DSL too")
# ---------------------------------------------------------------------------
def test_operational_schedule_round_trips():
    op = cd.OperationalSchedule(
        verdict=cd.VerdictCadence(eval_every=25, verdict_pairs=0, verdict_batch=32,
                                  async_verdict=True),
        telemetry=cd.TelemetryCadence(annulus=True, loss_term_log_every=0,
                                      handoff_readiness=True, dm1=True),
        reorient_every=50)
    ns = _parse_real(op.flags())
    assert ns.eval_every == 25 and ns.verdict_pairs == 0 and ns.verdict_batch == 32
    assert ns.async_verdict is True
    assert ns.annulus_telemetry is True and ns.annulus_band == 2.0
    assert ns.handoff_readiness_telemetry is True and ns.dm1_telemetry is True
    assert ns.reorient_every == 50
    assert op.validate() == [] and op.support_gaps() == ()


def test_operational_per_stage_verdict_fires_gap_and_compiles_densest():
    # council intent: verdict sparse in CE/tau (50), dense in the Muon finisher (10).
    op = cd.OperationalSchedule(verdict=cd.VerdictCadence(eval_every=50),
                                per_stage_verdict={"muon": 10})
    f = op.flags()
    assert f["--eval-every"] == 10  # densest requested cadence, globally (conservative)
    _parse_real(f)
    (gap,) = op.support_gaps()
    assert gap.axis == "operational_schedule"
    assert "per-stage verdict cadence" in gap.requirement
    assert "--eval-every-per-stage" in gap.flag_proposal  # proposal ONLY, never emitted
    assert "--eval-every-per-stage" not in f


def test_operational_validate_fails_closed():
    assert cd.OperationalSchedule(verdict=cd.VerdictCadence(eval_every=0)).validate()
    assert cd.OperationalSchedule(per_stage_verdict={"muon": -1}).validate()
    assert cd.OperationalSchedule(reorient_every=0).validate()


# ---------------------------------------------------------------------------
# compose, don't duplicate — the duplicate-emitter refusal
# ---------------------------------------------------------------------------
def _mini_curriculum(**kw):
    base = dict(stages=(cd.Stage("CE", None, None),
                        cd.Stage("tau_softplus", "--tau-softplus-start-epoch", 300)),
                temp=cd.Anneal(1.0, 0.05))
    base.update(kw)
    return cd.Curriculum(**base)


def test_duplicate_emitter_refused_on_unequal_values():
    # a softmax_temp LevelPath DISAGREEING with the temp Anneal endpoints is ambiguous.
    cur = _mini_curriculum(level_paths=(
        cd.LevelPath("softmax_temp", (cd.PathSegment("cosine", 1.0, 0.10),)),))
    assert any("DUPLICATE EMITTER" in p and "--softmax-temp-end" in p
               for p in cur.validate())


def test_duplicate_emitter_allows_agreeing_values():
    # same endpoints → the path only ADDS the shape flag; legal composition.
    cur = _mini_curriculum(level_paths=(
        cd.LevelPath("softmax_temp", (cd.PathSegment("geometric", 1.0, 0.05),)),))
    assert cur.validate() == []
    assert cur.flags()["--tau-anneal-shape"] == "geometric"


def test_hosc_levelpath_must_agree_with_hosc_schedule():
    cur = _mini_curriculum(
        hosc=cd.HoscSchedule(1.0, 4.0, "linear", 1.0),
        level_paths=(cd.LevelPath("hosc_beta", (cd.PathSegment("cosine", 1.0, 8.0),)),))
    assert any("DUPLICATE EMITTER" in p and "--hosc-beta" in p for p in cur.validate())


# ---------------------------------------------------------------------------
# consumer-introspection surface (operator amendment: dashboard/costate track
# the DSL live) — the class test that keeps FUTURE primitives honest
# ---------------------------------------------------------------------------
# one representative instance per registered kind; a NEW primitive must be added
# here (the executable spec forces the sample) — the structural assertions below
# cover it automatically via schedule_primitive_kinds().
_SAMPLES = {
    "Anneal": lambda: cd.Anneal(1.0, 0.05),
    "Curriculum": lambda: _mini_curriculum(),
    "ExitEvent": lambda: cd.ExitEvent("plateau"),
    "HoscSchedule": lambda: cd.HoscSchedule(),
    "LevelPath": lambda: cd.LevelPath("lr", (cd.PathSegment("cosine", 1e-3, 1e-4),)),
    "OperationalSchedule": lambda: cd.OperationalSchedule(),
    "PathSegment": lambda: cd.PathSegment("cosine", 1.0, 0.05),
    "Preserve": lambda: cd.Preserve(),
    "Priming": lambda: cd.Priming(warm_start_momentum=True),
    "Regularizer": lambda: cd.Regularizer("--eikonal-weight", 0.01),
    "RepeatUntil": lambda: cd.RepeatUntil("plateau", ("muon",), 100, 3),
    "Stage": lambda: cd.Stage("CE", None, None),
    "StageSpec": lambda: cd.StageSpec("muon", "--muon-start-epoch", 726),
    "TelemetryCadence": lambda: cd.TelemetryCadence(),
    "TrainerSupportGap": lambda: cd.TrainerSupportGap("a", "r", "n", "p"),
    "Transition": lambda: cd.Transition(),
    "VerdictCadence": lambda: cd.VerdictCadence(),
    "WitnessProgram": lambda: cd.BASELINE,
}


def test_every_schedule_primitive_exposes_the_display_surface():
    kinds = cd.schedule_primitive_kinds()
    assert kinds, "registry must not be empty"
    # a future primitive that forgets the surface (or the sample) fails HERE.
    missing_samples = sorted(set(kinds) - set(_SAMPLES))
    assert not missing_samples, (
        f"new schedule primitive(s) {missing_samples} need a _SAMPLES entry "
        "(the executable display spec)")
    for name, cls in sorted(kinds.items()):
        assert callable(getattr(cls, "to_display_dict", None)), name
        assert callable(getattr(cls, "describe", None)), name
        inst = _SAMPLES[name]()
        d = inst.to_display_dict()
        assert d["kind"] == name
        json.dumps(d)  # plain data — a generic renderer needs NO type knowledge
        assert isinstance(inst.describe(), str) and inst.describe()


def test_registry_is_auto_derived_not_hand_typed():
    kinds = cd.schedule_primitive_kinds()
    # every §14 primitive auto-registered; Lever/Contain/Authority are NOT schedule
    # primitives and must stay out.
    for expected in ("LevelPath", "StageSpec", "OperationalSchedule", "RepeatUntil",
                     "Priming", "ExitEvent", "TrainerSupportGap", "Curriculum"):
        assert expected in kinds, expected
    for excluded in ("Lever", "Contain", "Authority"):
        assert excluded not in kinds, excluded


def test_display_dict_includes_compiled_flags_and_gaps():
    lp = cd.LevelPath("lr", (cd.PathSegment("geometric", 1e-3, 1e-4),))
    d = lp.to_display_dict()
    assert d["flags"]["--lr"] == 1e-3
    assert d["gaps"] and d["gaps"][0]["kind"] == "TrainerSupportGap"


def test_display_is_fail_open_for_invalid_objects():
    # the display surface feeds a load-bearing dashboard/costate daemon: an INVALID
    # in-flight object renders its error instead of crashing the tick (validate()
    # remains the fail-CLOSED gate).
    lp = cd.LevelPath("no_such_quantity", (cd.PathSegment("cosine", 0.0, 1.0),))
    d = lp.to_display_dict()
    json.dumps(d)
    assert "error" in d["flags"] and "ValueError" in d["flags"]["error"]
    assert lp.validate()  # fail-closed channel still reports it


# ---------------------------------------------------------------------------
# THE COUNCIL-SHAPED EXECUTABLE SPEC (§14): CE → tau(event exit) → Muon(warm-start
# priming, LR geometric anneal) → repeat-finishing-block-until-dry, plus the
# operational-schedule block (verdict sparse in CE/tau, dense in Muon finishing).
# ---------------------------------------------------------------------------
def _council_curriculum() -> cd.Curriculum:
    return cd.Curriculum(
        stages=(
            # CE: entry stage, structured/seeded init priming (openpilot lane SDF seed).
            cd.StageSpec("CE", None, None,
                         priming=cd.Priming(structured_init=True, lane_prior_phi1=True,
                                            siren_init=True)),
            # tau: EVENT exit — the #315 nucleus-guarded plateau hand-off, not ep-fixed.
            cd.StageSpec("tau_softplus", "--tau-softplus-start-epoch", 300,
                         exit_event=cd.ExitEvent("nucleus_guarded_plateau",
                                                 min_stage_epochs=150, rel_eps=1e-4,
                                                 windows=4)),
            # l7 parked (measured defect — a TRUE-never tail).
            cd.Stage("l7_softplus", "--l7-start-epoch", 1401),
            # Muon finisher: warm-start momentum priming + repeat-until-dry block.
            cd.StageSpec("muon", "--muon-start-epoch", 726,
                         priming=cd.Priming(warm_start_momentum=True),
                         repeat_until=cd.RepeatUntil(
                             "marginal_dseg_floor", ("muon", "muon_leap"),
                             block_epochs=100, max_repeats=4, floor=1e-5),
                         exit_event=cd.ExitEvent("lever_exhaustion", floor=1e-5,
                                                 cap_epoch=1400)),
        ),
        temp=cd.Anneal(1.0, 0.05),
        regularizers=(cd.Regularizer("--length-weight", 0.001),),
        hosc=cd.HoscSchedule(1.0, 4.0, "linear", 1.0),
        tau=0.3,
        transition=cd.Transition(rewarmup_epochs=8, rewarmup_floor=0.1,
                                 rewarmup_shape="linear", reset_moments=True),
        handoff="event",
        level_paths=(
            # §14 axis 2: levels as PATHS, each a declared decision.
            cd.LevelPath("softmax_temp", (cd.PathSegment("geometric", 1.0, 0.05),),
                         anneal_epochs=1400),
            cd.LevelPath("lr", (cd.PathSegment("geometric", 1e-3, 1e-4),)),   # → GAP
            cd.LevelPath("ema_decay", (cd.PathSegment("constant", 0.997, epochs=726),
                                       cd.PathSegment("constant", 0.9995))),
            cd.LevelPath("eikonal_weight", (cd.PathSegment("step", 0.05, 0.10),),
                         stage="tau_softplus"),
            cd.LevelPath("muon_lr_frac", (cd.PathSegment("cosine", 1.0, 0.1),),
                         stage="muon"),
        ),
        operational=cd.OperationalSchedule(
            verdict=cd.VerdictCadence(eval_every=50, verdict_pairs=0,
                                      verdict_batch=32, async_verdict=True),
            telemetry=cd.TelemetryCadence(annulus=True, handoff_readiness=True),
            reorient_every=50,
            per_stage_verdict={"muon": 10},                                    # → GAP
        ),
    )


def test_council_example_constructs_validates_and_compiles():
    cur = _council_curriculum()
    assert cur.validate() == []  # gaps are NON-blocking; the conservative compile is legal
    # the whole schedule round-trips through the trainer's REAL argparse
    ns = _parse_real(cur.flags())
    assert ns.curriculum is True
    assert ns.tau_softplus_start_epoch == 300 and ns.muon_start_epoch == 726
    assert ns.l7_start_epoch == 1401                       # parked = TRUE never
    assert ns.curriculum_event_triggered is True           # tau event exit
    assert ns.curriculum_nucleus_guard is True
    assert ns.muon_warm_start_momentum is True             # Muon priming
    assert ns.tau_anneal_shape == "geometric"              # τ path
    assert ns.lr_schedule is True and ns.lr_end == 1e-4    # LR nearest-cosine compile
    assert ns.ema_decay_finisher == 0.9995                 # EMA π-group path
    assert ns.eikonal_weight_end == 0.10                   # eikonal step at tau onset
    assert ns.muon_lr_final_frac == 0.1                    # Muon LR anneal
    assert ns.eval_every == 10                             # densest cadence (conservative)
    assert ns.async_verdict is True and ns.verdict_pairs == 0


def test_council_example_reports_exactly_its_named_gaps():
    cur = _council_curriculum()
    gaps = cur.support_gaps()
    by_axis = sorted(g.axis for g in gaps)
    # exactly these four §14 gaps: LR-geometric shape, repeat-until-dry,
    # lever-exhaustion exit, per-stage verdict cadence.
    assert by_axis == ["exit_events", "levels_as_paths",
                       "operational_schedule", "stage_repetition"]
    # each gap names its trainer build; NONE of the proposals is an emitted flag.
    emitted = set(cur.flags())
    for g in gaps:
        assert g.flag_proposal
        assert not any(tok in emitted for tok in g.flag_proposal.split()
                       if tok.startswith("--")), g.flag_proposal
    # validate(surface_gaps=True) surfaces the same gaps as NAMED lines
    surfaced = cur.validate(surface_gaps=True)
    assert sum("TRAINER-SUPPORT GAP" in p for p in surfaced) == len(gaps)


def test_council_example_as_full_program_compiles_and_validates():
    from dataclasses import replace
    cur = _council_curriculum()
    prog = replace(
        cd.BASELINE,
        out_dir="experiments/results/_sec14_council_example",
        gt_cache=_GT,
        num_pairs=600,
        epochs=1500,
        curriculum=cur,
        resume_from=None,
        base={k: v for k, v in cd.BASELINE.base.items()
              # schedule/operational flags now owned by the curriculum object
              if k not in cur.flags()},
    )
    problems = prog.validate()
    assert problems == [], problems
    argv = prog.compile_trainer_argv()
    ap = cd.build_real_trainer_parser()
    ap.parse_args(argv[2:])  # the REAL argparse accepts the whole program
    assert prog.support_gaps() == cur.support_gaps()


def test_council_example_double_emitter_with_base_is_refused():
    from dataclasses import replace
    cur = _council_curriculum()
    prog = replace(cd.BASELINE, curriculum=cur,
                   base=dict(cd.BASELINE.base, **{"--eval-every": 25}))  # conflicts w/ 10
    assert any("DOUBLE EMITTER" in p and "--eval-every" in p for p in prog.validate())
