"""Tests for the witness curriculum/behavior DSL (task #189, Layer-0 of the bridge).

Locks: structural never-invent-flags, the BASELINE round-trip against the completed
run, the enforced preserve/contain/authority clauses, and lever composition.
"""
from dataclasses import replace

import pytest

from tac.witness_dsl import (
    BASELINE,
    Anneal,
    Authority,
    CacheGtSkeleton,
    CodeSpectralEntropy,
    Contain,
    DirectionalBasis,
    DM1Minimal,
    FiLMFix,
    Freeze,
    LanePrior,
    MicroBatch,
    Muon,
    PoseDecouple,
    Preserve,
    SoftBoundary,
    StageTransitionSoftVelocityBlend,
    StiefelW,
    TauFrozen,
    real_store_true_flags,
    real_trainer_flags,
)

# the exact flags the completed CE->tau->l7 run was launched with (grounded from the log)
_LAUNCHED = {
    "--resume-from", "--out-dir", "--gt-cache", "--num-pairs", "--epochs", "--render-h",
    "--render-w", "--hidden-dim", "--mod-dim", "--activation", "--siren-init",
    "--softmax-temp-start", "--softmax-temp-end", "--curriculum",
    "--tau-softplus-start-epoch", "--l7-start-epoch", "--palette-anchor", "--self-orient",
    "--reorient-every", "--freq-across", "--n-dir-freqs", "--freq-along", "--max-bank-freq",
    "--chroma", "--lane-edge-weight", "--lane-edge-class", "--lane-margin-target",
    "--lane-edge-start-epoch", "--w-seg", "--w-pose", "--eikonal-weight", "--length-weight",
    "--ema-decay", "--accum-pairs", "--grad-clip", "--verdict-pairs", "--eval-every",
    "--ckpt-every", "--async-verdict", "--mlx-device",
}


def test_real_trainer_flags_nonempty_and_known():
    flags = real_trainer_flags()
    assert len(flags) > 30
    for known in ("--epochs", "--muon-start-epoch", "--w-pose", "--resume-from"):
        assert known in flags


def test_baseline_validates_clean():
    assert BASELINE.validate() == []


def test_baseline_roundtrips_launched_flags():
    compiled = set(BASELINE.flag_dict())
    # every launched flag must be reproduced
    assert _LAUNCHED - compiled == set()
    # the only addition is the enforced PRESERVE clause
    assert compiled - _LAUNCHED == {"--stage-checkpoints"}


def test_invented_flag_is_refused():
    bad = replace(BASELINE, base={**BASELINE.base, "--totally-made-up": 1})
    probs = bad.validate()
    assert any("INVENTED FLAG" in p and "--totally-made-up" in p for p in probs)


def test_preserve_ckpt_cadence_binding():
    bad = replace(BASELINE, preserve=Preserve(ckpt_every=50))
    assert any("PRESERVE" in p and "ckpt-every" in p for p in bad.validate())
    bad0 = replace(BASELINE, preserve=Preserve(ckpt_every=0))
    assert any("PRESERVE" in p for p in bad0.validate())


def test_preserve_stage_boundaries_required():
    bad = replace(BASELINE, preserve=Preserve(stage_boundaries=False))
    assert any("stage-boundary" in p for p in bad.validate())


def test_contain_10gb_floor_binding():
    bad = replace(BASELINE, contain=Contain(min_free_gb=5.0))
    assert any("CONTAIN" in p and "10GB" in p for p in bad.validate())


def test_authority_realized_through_R_required():
    bad = replace(BASELINE, authority=Authority(realized_through_R=False))
    assert any("AUTHORITY" in p for p in bad.validate())


def test_freeze_is_constant_anneal():
    f = Freeze(0.05)
    assert isinstance(f, Anneal) and f.start == f.end == 0.05


def test_pose_decouple_sets_w_pose_zero():
    a5 = BASELINE.with_lever(PoseDecouple())
    assert a5.flag_dict()["--w-pose"] == 0.0
    assert a5.validate() == []


def test_muon_lever_extends_epochs_and_freezes_tau():
    a4 = BASELINE.with_lever(Muon(start_epoch=1500, window=100))
    fd = a4.flag_dict()
    assert a4.epochs == 1600
    assert fd["--muon-start-epoch"] == 1500
    assert fd["--softmax-temp-start"] == 0.05 and fd["--softmax-temp-end"] == 0.05
    assert fd["--stage-transition-reset-moments"] is True
    assert a4.validate() == []


def test_lever_composition_merges_overrides():
    combo = BASELINE.with_lever(PoseDecouple(), DirectionalBasis(weight=0.5))
    fd = combo.flag_dict()
    assert fd["--w-pose"] == 0.0          # from A5
    assert fd["--lane-edge-weight"] == 0.5  # from directional (baseline had 0)
    assert combo.validate() == []


def test_with_lever_does_not_mutate_baseline():
    _ = BASELINE.with_lever(Muon(start_epoch=1500, window=100))
    assert BASELINE.epochs == 1500
    assert BASELINE.flag_dict()["--w-pose"] == 1.0
    assert "--muon-start-epoch" not in BASELINE.flag_dict()


def test_compile_trainer_argv_booleans_bare():
    argv = BASELINE.compile_trainer_argv()
    assert "--siren-init" in argv
    # the bare boolean has no value token following it
    i = argv.index("--siren-init")
    assert i == len(argv) - 1 or argv[i + 1].startswith("--")


def test_compile_trainer_argv_false_boolean_emits_no_variant():
    prog = replace(BASELINE, preserve=Preserve(stage_boundaries=False))
    argv = prog.compile_trainer_argv()
    assert "--no-stage-checkpoints" in argv


def test_compile_daemon_argv_wraps_with_containment():
    argv = BASELINE.compile_daemon_argv(label="t", log="/x.log")
    assert "tools/spawn_durable_daemon.py" in argv
    assert "--min-free-gb" in argv and "10.0" in argv
    assert "--" in argv
    # the trainer command follows the --
    tail = argv[argv.index("--") + 1:]
    assert any("train_levelset_witness" in t for t in tail)


def test_tau_frozen_lever_isolates():
    arm = BASELINE.with_lever(TauFrozen(0.05))
    fd = arm.flag_dict()
    assert fd["--softmax-temp-start"] == 0.05 and fd["--softmax-temp-end"] == 0.05


# --- DSL adversarial-review regression guards (2026-06-28) ---
def test_review_C1_tau_frozen_extends_epochs_not_dead_arm():
    # C1: TauFrozen must carry a window or it runs zero steps when warm-started.
    arm = BASELINE.with_lever(TauFrozen())
    assert arm.epochs > BASELINE.epochs, "TauFrozen warm-start arm must run new steps"


def test_review_C1_dead_arm_guard_catches_zero_window(tmp_path):
    # C1 self-protection: a zero-window lever resumed from an end-of-run ckpt is flagged.
    import numpy as np
    ck = tmp_path / "resume_ep1500.npz"
    np.savez(ck, epoch=np.asarray(1500))
    from tac.witness_dsl import Lever
    dead = BASELINE.with_lever(Lever("zerowin", {"--softmax-temp-start": 0.05}),
                               resume_from=str(ck))
    assert any("DEAD ARM" in p for p in dead.validate())


def test_review_C2_validate_refuses_false_on_store_true():
    from dataclasses import replace
    bad = replace(BASELINE, base={**BASELINE.base, "--stage-transition-reset-moments": False})
    assert any("store_true" in p for p in bad.validate())


def test_type_compat_refuses_bool_override_on_value_flag():
    # TYPE-COMPAT (review 2026-07-06, the EikonalViscosity-class static guard): a True on a
    # type=float flag compiles to a BARE token → trainer argparse "expected one argument"
    # crash AFTER every launcher gate. validate() must refuse it statically.
    from dataclasses import replace
    bad = replace(BASELINE, base={**BASELINE.base, "--eikonal-viscosity": True})
    assert any("TYPE-INCOMPATIBLE" in p and "--eikonal-viscosity" in p for p in bad.validate())


def test_type_compat_refuses_value_override_on_boolean_flag():
    # the mirror direction: a numeric value on a store_true / BooleanOptionalAction flag
    # compiles to '--flag 1.0' → "unrecognized arguments" at launch.
    from dataclasses import replace
    bad = replace(BASELINE, base={**BASELINE.base, "--film-stiefel": 1.0})
    assert any("TYPE-INCOMPATIBLE" in p and "--film-stiefel" in p for p in bad.validate())


def test_type_compat_bools_on_boolean_flags_stay_clean():
    # True on boolean-action flags (BASELINE's --curriculum/--chroma/... + a False on a
    # BooleanOptionalAction flag) is legal — no TYPE-INCOMPATIBLE noise on valid programs.
    from dataclasses import replace
    ok = replace(BASELINE, base={**BASELINE.base, "--async-verdict": False})  # BooleanOptionalAction
    assert not any("TYPE-INCOMPATIBLE" in p for p in ok.validate())


def test_review_M2_with_lever_can_clear_resume_from():
    fresh = BASELINE.with_lever(SoftBoundary(), resume_from=None)
    assert fresh.resume_from is None
    inherited = BASELINE.with_lever(SoftBoundary())  # default = inherit
    assert inherited.resume_from == BASELINE.resume_from


def test_soft_boundary_replaces_beta_steplim():
    arm = BASELINE.with_lever(SoftBoundary(2.0))
    assert arm.flag_dict()["--hosc-beta"] == 2.0 and arm.epochs > BASELINE.epochs
    assert arm.validate() == []


def test_stage_transition_soft_velocity_blend_default_off_is_arg_inert():
    lever = StageTransitionSoftVelocityBlend(beta2=0.999, c=2.0)
    arm = BASELINE.with_lever(lever)
    assert lever.overrides == {}
    assert arm.flag_dict() == BASELINE.flag_dict()
    assert arm.validate() == []
    assert "m_new=(1-a(t))*m_mapped+a(t)*m_fresh" in lever.notes
    assert "2000 optimizer steps" in lever.notes


def test_stage_transition_soft_velocity_blend_enabled_refuses_until_consumer_exists():
    with pytest.raises(ValueError, match="no levelset trainer consumer"):
        StageTransitionSoftVelocityBlend(enabled=True)


# --- LEVER-A (FiLM-rank-fix) + LEVER-B (thin-lane prior) DSL levers (task: film-rank-fix + lane-prior) ---
def test_film_fix_default_turns_on_per_layer_and_concat():
    ov = FiLMFix().overrides
    assert ov["--film-per-layer"] is True
    assert ov["--film-concat-code"] is True
    # default rank-floor weight is 0 => NOT emitted (off)
    assert "--film-rank-floor-weight" not in ov


def test_film_fix_lever_validates_clean_and_real_flags():
    arm = BASELINE.with_lever(FiLMFix())
    assert arm.validate() == []
    real = real_trainer_flags()
    for f in arm.levers[-1].overrides:
        assert f in real, f"FiLMFix emitted a non-real flag: {f}"


def test_film_fix_rank_floor_emits_weight_and_target():
    ov = FiLMFix(rank_floor_weight=0.05, rank_floor_target=6.0).overrides
    assert ov["--film-rank-floor-weight"] == 0.05
    assert ov["--film-rank-floor-target"] == 6.0


def test_film_fix_never_emits_false_on_store_true():
    # review C2: store_true flags must never be emitted False (=> --no-X crash). With per_layer off
    # the flag is simply ABSENT (trainer default off), never False.
    ov = FiLMFix(per_layer=False, concat_code=False).overrides
    assert "--film-per-layer" not in ov and "--film-concat-code" not in ov
    st = real_store_true_flags()
    assert "--film-per-layer" in st and "--film-concat-code" in st  # they ARE store_true
    # and a composed program with film-fix off-routes still validates (no False store_true emitted)
    assert BASELINE.with_lever(FiLMFix(per_layer=True, concat_code=False)).validate() == []


def test_film_fix_extends_epochs_window():
    arm = BASELINE.with_lever(FiLMFix(window=120))
    assert arm.epochs == BASELINE.epochs + 120  # warm-start window (no dead-arm)


def test_lane_prior_lever_validates_clean_and_real_flags():
    arm = BASELINE.with_lever(LanePrior())
    assert arm.validate() == []
    real = real_trainer_flags()
    for f in arm.levers[-1].overrides:
        assert f in real, f"LanePrior emitted a non-real flag: {f}"


def test_lane_prior_distinct_from_lane_prior_phi1():
    ov = LanePrior().overrides
    # LEVER-B is the --lane-thin-* realized-margin prior, NOT the --lane-prior-phi1 structured-init flag
    assert "--lane-prior-phi1" not in ov
    assert ov["--lane-thin-weight"] == 1.0
    assert ov["--lane-thin-class"] == 1


def test_film_fix_and_lane_prior_compose_clean():
    combo = BASELINE.with_lever(FiLMFix(rank_floor_weight=0.1), LanePrior(weight=2.0))
    fd = combo.flag_dict()
    assert fd["--film-per-layer"] is True
    assert fd["--film-rank-floor-weight"] == 0.1
    assert fd["--lane-thin-weight"] == 2.0
    assert combo.validate() == []
    # baseline is unmutated (composition is pure)
    assert "--film-per-layer" not in BASELINE.flag_dict()
    assert "--lane-thin-weight" not in BASELINE.flag_dict()


# --- DM1 minimal cure DSL levers (Stiefel-W + code spectral-entropy) ---
def test_stiefel_w_lever_emits_real_store_true_flag():
    ov = StiefelW().overrides
    assert ov["--film-stiefel"] is True
    real = real_trainer_flags()
    assert "--film-stiefel" in real
    st = real_store_true_flags()
    assert "--film-stiefel" in st  # it IS store_true


def test_stiefel_w_lever_validates_clean_and_extends_window():
    arm = BASELINE.with_lever(StiefelW(window=80))
    assert arm.validate() == []                  # never emits False on a store_true
    assert arm.epochs == BASELINE.epochs + 80    # warm-start window (no dead-arm, review C1)
    assert arm.flag_dict()["--film-stiefel"] is True


def test_code_spectral_entropy_lever_emits_weight_and_off_is_absent():
    ov = CodeSpectralEntropy(beta=0.02).overrides
    assert ov["--code-spectral-entropy-weight"] == 0.02
    assert "--code-spectral-entropy-weight" in real_trainer_flags()
    # beta<=0 => OFF => flag absent (never emitted as a no-op)
    assert CodeSpectralEntropy(beta=0.0).overrides == {}


def test_code_spectral_entropy_validates_clean():
    arm = BASELINE.with_lever(CodeSpectralEntropy(beta=0.01, window=60))
    assert arm.validate() == []
    assert arm.flag_dict()["--code-spectral-entropy-weight"] == 0.01
    assert arm.epochs == BASELINE.epochs + 60


def test_dm1_minimal_composes_both_with_single_window():
    s, c = DM1Minimal(beta=0.01, window=80)
    # the entropy half carries NO window (the Stiefel half carries the single warm-start window)
    assert s.epochs_delta == 80 and c.epochs_delta == 0
    arm = BASELINE.with_lever(*DM1Minimal(beta=0.01, window=80))
    fd = arm.flag_dict()
    assert fd["--film-stiefel"] is True
    assert fd["--code-spectral-entropy-weight"] == 0.01
    assert arm.epochs == BASELINE.epochs + 80   # window counted ONCE, not 2x
    assert arm.validate() == []


def test_dm1_levers_default_off_in_baseline_byte_identity_guard():
    # the two DM1 flags must be ABSENT from the baseline flag set (default-off => byte-identical)
    fd = BASELINE.flag_dict()
    assert "--film-stiefel" not in fd
    assert "--code-spectral-entropy-weight" not in fd


# ===========================================================================
# OPENPILOT-SEEDED OPENING + ADAPTIVE STACKING (DAG FEED-ln, 2026-06-29)
# ===========================================================================
from tac.witness_dsl import (  # noqa: E402
    StagePolicy,
    advance_to_l7,
    advance_to_muon,
    decide_next_stage,
    extend_stage,
    openpilot_seeded_opening,
    plan_adaptive_step,
    stack_next_program,
    stage_trajectory,
)

_GT = "experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz"


def _opening():
    return openpilot_seeded_opening(
        out_dir="experiments/results/openpilot_seeded_opening_TEST",
        gt_cache=_GT, num_pairs=200, ce_to=300, tau_window=300)


def test_opening_validates_clean():
    assert _opening().validate() == []


def test_opening_is_from_scratch_seeded_not_resumed():
    op = _opening()
    assert op.resume_from is None          # FROM SCRATCH (structured-init seed, not a ckpt)
    fd = op.flag_dict()
    assert fd["--structured-init"] is True
    assert fd["--lane-prior-phi1"] is True
    assert fd["--lane-prior-phi1-mode"] == "replace"


def test_opening_is_d_seg_only_pose_on_sidecar():
    # the witness's sole controllable job is d_seg; pose rides the stored Quantizr sidecar
    assert _opening().flag_dict()["--w-pose"] == 0.0


def test_opening_tau_is_reachability_floor():
    # tau=0.3 == the anneal-memo reachability floor Delta_min (margin resonance)
    assert _opening().flag_dict()["--tau-softplus-tau"] == 0.3


def test_opening_reheat_on_every_transition():
    fd = _opening().flag_dict()
    assert fd["--stage-transition-rewarmup-epochs"] == 8
    assert fd["--stage-transition-rewarmup-floor"] == 0.1
    assert fd["--stage-transition-reset-moments"] is True  # store_true, emitted True only


def test_opening_l7_parked_as_noop_tail():
    # the opening is EXACTLY ce->tau: l7 boundary parked AT epochs (no-op tail) so l7+Muon
    # are stacked adaptively, and the trainer's curriculum ordering assert still holds.
    fd = _opening().flag_dict()
    assert fd["--l7-start-epoch"] == fd["--epochs"] == 600
    assert fd["--tau-softplus-start-epoch"] == 300


def test_opening_records_single_seed_determinism():
    fd = _opening().flag_dict()
    assert fd["--seed"] == 0
    # deterministic: same args -> identical flag_dict (pure construction)
    assert _opening().flag_dict() == _opening().flag_dict()


def test_opening_skips_smooth_and_rate_stages_structurally():
    # smooth (RAISES d_seg) + lambda/sigma rate stages do not exist in the trainer curriculum;
    # the opening's stages are exactly CE/tau/l7 -> the skip is structural, not a flag.
    names = [s.name for s in _opening().stages]
    assert names == ["CE", "tau_softplus", "l7_softplus"]
    assert "smooth" not in names


def test_curriculum_ordering_violation_is_refused():
    # a doomed config (tau stage never runs: tau_start >= l7_start) is refused at DSL-validate
    # time (the trainer's REAL assert surfaced: 0 < tau_start < l7_start).
    from dataclasses import replace

    from tac.witness_dsl import Stage
    bad = replace(_opening(), stages=(
        _opening().stages[0],
        Stage("tau_softplus", "--tau-softplus-start-epoch", 500),
        Stage("l7_softplus", "--l7-start-epoch", 300),  # tau >= l7 → tau never forms the partition
    ))
    assert any("CURRICULUM ORDERING" in p for p in bad.validate())


def test_l7_start_beyond_epochs_is_the_legitimate_parked_form():
    # ALIGNED TO THE TRAINER (L1 SEAL-review relax, 4bf533cab) + Curriculum.validate():
    # l7_start > epochs is the LEGITIMATE "l7 NEVER runs" form (l7 is a measured defect demoted
    # from the default curriculum; fresh_seeded parks l7 at epochs+1). The prior WitnessProgram
    # "<= epochs" clause was stale and refused this form — it must validate CLEAN now.
    from dataclasses import replace

    from tac.witness_dsl import Stage
    parked = replace(_opening(), stages=(
        _opening().stages[0],
        Stage("tau_softplus", "--tau-softplus-start-epoch", 300),
        Stage("l7_softplus", "--l7-start-epoch", _opening().epochs + 1),  # parked: never runs
    ))
    probs = parked.validate()
    assert not any("CURRICULUM ORDERING" in p for p in probs), probs


# --- the deterministic reactive decision policy ---
_POLICY = StagePolicy()
_DESCENDING = ((300, 0.0064), (325, 0.0058), (350, 0.0052), (375, 0.0046), (400, 0.0040))
_PLATEAU = ((300, 0.003960), (325, 0.003958), (350, 0.003957), (375, 0.003957), (400, 0.003956))
_RISING = ((300, 0.0040), (325, 0.0042), (350, 0.0045), (375, 0.0048), (400, 0.0050))


def test_decide_extend_when_still_descending():
    d = decide_next_stage(_DESCENDING, policy=_POLICY, final_ckpt="F.npz", best_ckpt="B.npz")
    assert d.action == "EXTEND"
    assert d.resume_from == "F.npz"        # resume from the final (continue the stage)
    assert d.slope is not None and d.slope < 0


def test_decide_advance_when_plateau():
    d = decide_next_stage(_PLATEAU, policy=_POLICY, final_ckpt="F.npz", best_ckpt="B.npz")
    assert d.action == "ADVANCE"
    assert d.resume_from == "F.npz"
    assert abs(d.slope) < _POLICY.plateau_abs_slope


def test_decide_rollback_when_stage_raises_d_seg():
    d = decide_next_stage(_RISING, policy=_POLICY, final_ckpt="F.npz", best_ckpt="B.npz")
    assert d.action == "ROLLBACK_BRANCH"
    assert d.resume_from == "B.npz"        # roll back to the BEST checkpoint
    assert d.best_epoch == 300             # the min-d_seg verdict


def test_decide_empty_trajectory_is_conservative_extend():
    d = decide_next_stage((), policy=_POLICY, final_ckpt="F.npz", best_ckpt="B.npz")
    assert d.action == "EXTEND" and d.n_verdicts == 0


def test_decision_is_deterministic_pure_function():
    a = decide_next_stage(_PLATEAU, policy=_POLICY, final_ckpt="F.npz", best_ckpt="B.npz")
    b = decide_next_stage(_PLATEAU, policy=_POLICY, final_ckpt="F.npz", best_ckpt="B.npz")
    assert a == b                          # same trajectory + policy -> identical decision


# --- program transforms (the stacked continuations validate + are launch-valid) ---
def test_extend_stage_pushes_l7_out_and_extends_epochs():
    op = _opening()
    ext = extend_stage(op, resume_from="tau.npz", out_dir="OUT_ext", window=300)
    fd = ext.flag_dict()
    assert ext.epochs == op.epochs + 300
    assert fd["--l7-start-epoch"] == ext.epochs   # still a no-op tail (extend tau, not advance)
    assert ext.resume_from == "tau.npz"
    assert ext.validate() == []


def test_advance_to_l7_engages_at_resume_epoch():
    op = _opening()
    a = advance_to_l7(op, resume_from="tau.npz", out_dir="OUT_l7", window=300)
    fd = a.flag_dict()
    assert fd["--l7-start-epoch"] == op.epochs          # fires immediately at resume
    assert a.epochs == op.epochs + 300                  # l7 window
    assert fd["--l7-start-epoch"] < fd["--epochs"]      # l7 actually runs
    assert a.validate() == []


def test_advance_to_muon_uses_feed_fi_lr_not_unwired_recall():
    op = _opening()
    a7 = advance_to_l7(op, resume_from="tau.npz", out_dir="OUT_l7", window=300)
    m = advance_to_muon(a7, resume_from="l7.npz", out_dir="OUT_muon", window=100)
    fd = m.flag_dict()
    # FEED-fi conservative measured band (1e-3..2e-3), NOT the unwired-recall 0.03
    assert fd["--muon-lr"] == 2e-3
    assert fd["--muon-start-epoch"] == a7.epochs
    assert fd["--stage-transition-reset-moments"] is True
    # Muon freezes tau + render softmax-temp at the l7-end value (clean A/B)
    assert fd["--softmax-temp-start"] == 0.05 and fd["--softmax-temp-end"] == 0.05
    assert m.validate() == []


def test_stack_next_program_dispatches_on_action():
    op = _opening()
    adv = stack_next_program(
        op, decide_next_stage(_PLATEAU, policy=_POLICY, final_ckpt="F.npz", best_ckpt="B.npz"),
        advance_to="l7", out_dir="OUT", policy=_POLICY)
    assert adv.flag_dict()["--l7-start-epoch"] == op.epochs   # ADVANCE engaged l7
    ext = stack_next_program(
        op, decide_next_stage(_DESCENDING, policy=_POLICY, final_ckpt="F.npz", best_ckpt="B.npz"),
        advance_to="l7", out_dir="OUT", policy=_POLICY)
    assert ext.flag_dict()["--l7-start-epoch"] == ext.epochs  # EXTEND kept l7 parked


def test_stage_trajectory_parses_verdict_rows(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        '{"stage": "verdict", "epoch": 305, "d_seg": 0.0061, "seg_form": "tau_softplus"}\n'
        '{"stage": "other", "epoch": 306, "d_seg": 9.9}\n'
        '{"stage": "verdict", "epoch": 330, "d_seg": 0.0050, "seg_form": "tau_softplus"}\n'
        '{"stage": "verdict", "epoch": 355, "d_seg": 0.0070, "seg_form": "ce"}\n')
    rows = stage_trajectory(log, seg_form="tau_softplus")
    assert rows == ((305, 0.0061), (330, 0.0050))   # ce row filtered out


def test_plan_adaptive_step_emit_only_roundtrip(tmp_path):
    import numpy as np
    out = tmp_path / "tau_run"
    out.mkdir()
    (out / "levelset_resume_state.npz").write_bytes(b"x")          # the final-state ckpt
    np.savez(out / "levelset_resume_stageTau_ep600.npz", epoch=np.asarray(600))  # a preserved ckpt
    log = tmp_path / "tau.log"
    # a PLATEAU trajectory -> ADVANCE to l7
    log.write_text("".join(
        f'{{"stage": "verdict", "epoch": {e}, "d_seg": {s}, "seg_form": "tau_softplus"}}\n'
        for e, s in _PLATEAU))
    op = _opening()
    step = plan_adaptive_step(
        op, prev_out_dir=str(out), log_path=str(log),
        advance_to="l7", out_dir="experiments/results/l7_cont_TEST",
        next_log="/x/l7.log", seg_form="tau_softplus")
    assert step["valid"] is True and step["violations"] == []
    assert step["decision"]["action"] == "ADVANCE"
    # the emitted continuation warm-starts from the prior final-state ckpt
    assert step["next_resume_from"] == str(out / "levelset_resume_state.npz")
    # CONTAINMENT: the step only EMITS a daemon launch command (never fires)
    assert any("train_levelset_witness" in t for t in step["daemon_argv"])
    assert any("--resume-from" in t for t in step["daemon_argv"])


def test_plan_adaptive_step_rollback_resumes_from_best_preserved_ckpt(tmp_path):
    import numpy as np
    out = tmp_path / "smooth_run"
    out.mkdir()
    (out / "levelset_resume_state.npz").write_bytes(b"x")
    np.savez(out / "levelset_resume_stageTau_ep300.npz", epoch=np.asarray(300))  # best-region ckpt
    log = tmp_path / "s.log"
    log.write_text("".join(
        f'{{"stage": "verdict", "epoch": {e}, "d_seg": {s}, "seg_form": "tau_softplus"}}\n'
        for e, s in _RISING))   # rising -> ROLLBACK to best (ep300)
    step = plan_adaptive_step(
        _opening(), prev_out_dir=str(out), log_path=str(log),
        advance_to="l7", out_dir="experiments/results/branch_TEST",
        next_log="/x/b.log", seg_form="tau_softplus")
    assert step["decision"]["action"] == "ROLLBACK_BRANCH"
    # rolled back to the preserved ckpt at/<= the best epoch (300), not the final state
    assert step["next_resume_from"].endswith("levelset_resume_stageTau_ep300.npz")
    assert step["valid"] is True


# ===========================================================================
# SEARCH ENRICHMENT (stage × config × pass × scale) — operator 2026-06-29 (FEED-lo)
# ===========================================================================
from tac.witness_dsl import (  # noqa: E402
    ArmResult,
    Cycle,
    MarginSaliency,
    PrimingContext,
    ScalePass,
    UniWARD,
    curvelet_scale_passes,
    expand_cycles,
    measure_synergy,
    priming_chain,
    polar_fourier_scale_passes,
    rerun_stage_new_config,
    scale_progression,
    select_synergistic_combos,
    synergy_map,
)


# --- B. UNIWARD + margin-saliency levers (Fridrich inverse-steganalysis; BUILT) ---
def test_uniward_lever_validates_and_emits_real_flags():
    arm = BASELINE.with_lever(UniWARD())
    assert arm.validate() == []
    real = real_trainer_flags()
    for f in arm.levers[-1].overrides:
        assert f in real, f"UniWARD emitted a non-real flag: {f}"


def test_uniward_store_true_emitted_true_only():
    ov = UniWARD(beta=6.0, start_epoch=900).overrides
    assert ov["--margin-saliency-uniward"] is True  # store_true -> True only (never False)
    assert ov["--margin-saliency-uniward-beta"] == 6.0
    assert ov["--margin-saliency-start-epoch"] == 900  # late-stage (l7/Muon) arm
    assert "--margin-saliency-uniward" in real_store_true_flags()


def test_margin_saliency_lever_validates():
    arm = BASELINE.with_lever(MarginSaliency(weight=2.0))
    assert arm.validate() == []
    assert arm.flag_dict()["--margin-saliency-weight"] == 2.0


# --- A1. MULTI-PASS / RERUN_NEW_CONFIG (the config axis) ---
_POLICY_RERUN = StagePolicy(rerun_floor=0.001)   # plateau above 0.001 -> re-run sharper


def test_decide_rerun_when_plateau_above_floor():
    d = decide_next_stage(_PLATEAU, policy=_POLICY_RERUN, final_ckpt="F.npz", best_ckpt="B.npz")
    assert d.action == "RERUN_NEW_CONFIG"   # best ~0.00396 > rerun_floor 0.001
    assert d.resume_from == "F.npz"


def test_decide_advance_when_plateau_at_or_below_floor():
    d = decide_next_stage(_PLATEAU, policy=StagePolicy(rerun_floor=0.005),
                          final_ckpt="F.npz", best_ckpt="B.npz")
    assert d.action == "ADVANCE"            # best ~0.00396 <= rerun_floor 0.005 -> exhausted -> advance


def test_decide_rerun_off_by_default_back_compat():
    # rerun_floor None (default) -> plateau ADVANCES (the 3-action back-compat path)
    d = decide_next_stage(_PLATEAU, policy=StagePolicy(), final_ckpt="F.npz", best_ckpt="B.npz")
    assert d.action == "ADVANCE"


def test_rerun_stage_new_config_sharper_tau_validates():
    op = _opening()
    r = rerun_stage_new_config(op, resume_from="tau.npz", out_dir="OUT_tau2",
                               window=300, config_overrides={"--tau-softplus-tau": 0.2})
    fd = r.flag_dict()
    assert fd["--tau-softplus-tau"] == 0.2          # sharper 2nd tau pass
    assert fd["--l7-start-epoch"] == r.epochs       # l7 still parked (same stage)
    assert r.epochs == op.epochs + 300
    assert r.validate() == []


def test_rerun_stage_refuses_shape_changing_override():
    with pytest.raises(ValueError, match="shape-changing"):
        rerun_stage_new_config(_opening(), resume_from="x.npz", out_dir="O",
                               window=100, config_overrides={"--hidden-dim": 128})


def test_stack_next_program_rerun_dispatch():
    op = _opening()
    dec = decide_next_stage(_PLATEAU, policy=_POLICY_RERUN, final_ckpt="F.npz", best_ckpt="B.npz")
    nxt = stack_next_program(op, dec, advance_to="l7", out_dir="OUT", policy=_POLICY_RERUN,
                             rerun_config={"--tau-softplus-tau": 0.2})
    assert nxt.flag_dict()["--tau-softplus-tau"] == 0.2
    with pytest.raises(ValueError, match="requires a rerun_config"):
        stack_next_program(op, dec, advance_to="l7", out_dir="OUT", policy=_POLICY_RERUN)


# --- A1'. Cycle multi-pass-with-varying-config + fresh + l7 auto-park ---
def test_expand_cycles_applies_per_pass_config_and_parks_l7():
    op = _opening()
    cycles = [Cycle("tau_sharp", window=200, config={"--tau-softplus-tau": 0.2})]
    progs = expand_cycles(op, cycles, start_resume_from="seed.npz", start_epoch=600,
                          out_dir_prefix="OUT")
    p = progs[0]
    fd = p.flag_dict()
    assert fd["--tau-softplus-tau"] == 0.2
    assert fd["--l7-start-epoch"] == p.epochs == 800   # l7 auto-parked at the new end
    assert p.resume_from == "seed.npz"                 # warm (value-only config)
    assert p.validate() == []


def test_expand_cycles_shape_changing_config_forces_fresh():
    progs = expand_cycles(_opening(), [Cycle("grow", window=200, config={"--hidden-dim": 128})],
                          start_resume_from="seed.npz", start_epoch=600, out_dir_prefix="OUT")
    assert progs[0].resume_from is None   # shape change -> FRESH arm (no warm-start crash)


# --- C. PRIMING (first-class) ---
def test_priming_recorded_and_primer_conditioned_floor():
    prime = PrimingContext(primer_stage="tau_softplus", resume_ckpt="tau.npz",
                           primer_final_d_seg=0.00396)
    pol = StagePolicy(rerun_floor_by_primer={"tau_softplus": 0.001})
    d = decide_next_stage(_PLATEAU, policy=pol, final_ckpt="F.npz", best_ckpt="B.npz", priming=prime)
    assert d.primed_by == "tau_softplus"       # priming recorded
    assert d.action == "RERUN_NEW_CONFIG"      # primer-conditioned floor 0.001 -> best above -> rerun
    assert d.to_record()["primed_by"] == "tau_softplus"


def test_priming_chain_extracts_warm_start_edges():
    op = _opening()
    progs = expand_cycles(op, [Cycle("c0", 200), Cycle("c1", 200)],
                          start_resume_from="seed.npz", start_epoch=600, out_dir_prefix="OUT")
    chain = priming_chain([op, *progs])
    assert chain[0]["from_scratch"] is True            # the opening is from-scratch (seeded)
    assert chain[1]["primed_by_ckpt"] == "seed.npz"    # c0 primed by the seed
    assert chain[2]["primed_by_ckpt"].endswith("levelset_resume_state.npz")  # c1 primed by c0


# --- B'. SYNERGIES ---
def _arm(label, delta):
    return ArmResult(label, "l7_softplus", 5, None, None, None, None, None, delta, f"{label}.log")


def test_measure_synergy_superadditive_is_compound():
    ind = {"A": _arm("A", -0.0010), "B": _arm("B", -0.0005)}
    combo = _arm("AB", -0.0020)   # better than the -0.0015 additive null
    syn = measure_synergy(ind, combo, ("A", "B"))
    assert syn.superadditive is True
    assert abs(syn.synergy - (-0.0005)) < 1e-12


def test_measure_synergy_subadditive_is_interference():
    ind = {"A": _arm("A", -0.0010), "B": _arm("B", -0.0005)}
    combo = _arm("AB", -0.0010)   # worse than the -0.0015 additive null -> interference
    syn = measure_synergy(ind, combo, ("A", "B"))
    assert syn.superadditive is False and syn.synergy > 0


def test_measure_synergy_missing_delta_is_none():
    ind = {"A": _arm("A", -0.001)}   # B missing
    syn = measure_synergy(ind, _arm("AB", -0.002), ("A", "B"))
    assert syn.synergy is None


def test_synergy_map_and_select_compounding_combos():
    from tac.witness_dsl import DirectionalBasis
    ind = {"dir": _arm("dir", -0.0010), "uni": _arm("uni", -0.0005)}
    combos = {("dir", "uni"): _arm("dir+uni", -0.0020)}   # compound
    smap = synergy_map(ind, combos)
    levers_by_label = {"dir": DirectionalBasis(), "uni": UniWARD()}
    winners = select_synergistic_combos(smap, levers_by_label)
    assert len(winners) == 1 and len(winners[0]) == 2   # the compounding (dir,uni) combo selected


# --- D. SCALING = the curvelet coarse->fine multi-scale band climb ---
def test_curvelet_scale_passes_coarse_to_fine_warm():
    passes = curvelet_scale_passes((16.0, 32.0, 64.0), window=200, freq_across=48.0)
    assert len(passes) == 3
    assert passes[0].overrides["--max-bank-freq"] == 16.0    # coarse
    assert passes[2].overrides["--max-bank-freq"] == 64.0    # fine
    assert passes[0].overrides["--freq-across"] == 48.0      # anisotropy ratio (directional)
    assert all(not p.is_fresh for p in passes)               # value-only -> warm-start safe


def test_truthful_scale_api_preserves_legacy_resume_names() -> None:
    current = polar_fourier_scale_passes((16.0, 32.0), window=10)
    legacy = curvelet_scale_passes((16.0, 32.0), window=10)
    assert [p.overrides for p in current] == [p.overrides for p in legacy]
    assert [p.name for p in current] == ["polar_fourier_band_16", "polar_fourier_band_32"]
    assert [p.name for p in legacy] == ["curvelet_band_16", "curvelet_band_32"]


def test_scale_progression_warm_chain_validates_and_parks_l7():
    op = _opening()
    passes = curvelet_scale_passes((32.0, 64.0), window=200)
    progs = scale_progression(op, passes, start_resume_from="seed.npz", start_epoch=600,
                              out_dir_prefix="experiments/results/curvelet")
    assert [p.epochs for p in progs] == [800, 1000]
    for p in progs:
        assert p.flag_dict()["--l7-start-epoch"] == p.epochs   # l7 parked (finer-scale re-run)
        assert p.validate() == []
    assert progs[0].resume_from == "seed.npz"
    assert progs[1].resume_from.endswith("levelset_resume_state.npz")  # warm chain


def test_scale_progression_band_count_change_is_fresh():
    # adding BANDS (--bank-n-scales) changes the feature count -> shape mismatch -> FRESH arm
    fresh_pass = ScalePass("more_bands", window=200, overrides={"--bank-n-scales": 6})
    assert fresh_pass.is_fresh is True
    progs = scale_progression(_opening(), [fresh_pass], start_resume_from="seed.npz",
                              start_epoch=600, out_dir_prefix="OUT")
    assert progs[0].resume_from is None   # fresh (no warm-start shape crash)


def test_max_bank_freq_is_warm_but_bank_n_scales_is_fresh():
    assert ScalePass("a", 100, {"--max-bank-freq": 64.0}).is_fresh is False
    assert ScalePass("b", 100, {"--bank-n-scales": 8}).is_fresh is True


# --- SPEED levers (#260 cache-gt-skeleton, #261 micro-batch) — the gauge's non-curriculum
#     compute config that compiles to trainer argv (re-syncs triality leg 2) ---
def test_cache_gt_skeleton_lever_emits_store_true_only():
    # store_true flag: emitted True ONLY (never False), so validate() never trips the C2 --no- guard.
    arm = BASELINE.with_lever(CacheGtSkeleton())
    fd = arm.flag_dict()
    assert fd["--cache-gt-skeleton"] is True
    assert arm.validate() == []
    # it is a store_true trainer flag, so a False would be a C2 violation — the lever never emits one
    assert "--cache-gt-skeleton" in real_store_true_flags()


def test_cache_gt_skeleton_compiles_as_bare_flag():
    # bare boolean: no value token follows in the compiled argv (matches --siren-init etc.)
    argv = BASELINE.with_lever(CacheGtSkeleton()).compile_trainer_argv()
    assert "--cache-gt-skeleton" in argv
    i = argv.index("--cache-gt-skeleton")
    assert i == len(argv) - 1 or argv[i + 1].startswith("--")


def test_cache_gt_skeleton_is_global_config_not_an_epoch_extension():
    # SPEED lever: no warm-start window (it is global config, not an A/B stage) -> epochs unchanged.
    arm = BASELINE.with_lever(CacheGtSkeleton())
    assert arm.epochs == BASELINE.epochs


def test_micro_batch_lever_emits_int_value():
    arm = BASELINE.with_lever(MicroBatch(4))
    fd = arm.flag_dict()
    assert fd["--micro-batch-pairs"] == 4
    assert arm.validate() == []
    # value flag (not store_true) — the trainer declares it type=int
    assert "--micro-batch-pairs" not in real_store_true_flags()


def test_micro_batch_default_is_four_and_value_flag_in_argv():
    argv = BASELINE.with_lever(MicroBatch()).compile_trainer_argv()  # default pairs=4
    j = argv.index("--micro-batch-pairs")
    assert argv[j + 1] == "4"


def test_micro_batch_one_emits_serial_baseline_explicitly():
    # B=1 is the byte-identical serial path -> MicroBatch(1) emits it for an apples-to-apples A/B.
    arm = BASELINE.with_lever(MicroBatch(1))
    assert arm.flag_dict()["--micro-batch-pairs"] == 1
    assert arm.validate() == []


def test_micro_batch_is_global_config_not_an_epoch_extension():
    arm = BASELINE.with_lever(MicroBatch(4))
    assert arm.epochs == BASELINE.epochs


@pytest.mark.parametrize("pairs", [0, -1, -8])
def test_micro_batch_rejects_nonpositive_pair_count(pairs):
    with pytest.raises(ValueError, match=r"pairs must be >= 1"):
        MicroBatch(pairs)


def test_micro_batch_composes_with_seed_islands_after_training_override():
    """The canonical V9 arm uses seed islands; dual batched co-grad is a supported composition."""
    from tac.witness_dsl.curriculum_dsl import SeedIslandBirth

    combo = BASELINE.with_lever(SeedIslandBirth(), MicroBatch(2))
    fd = combo.flag_dict()
    assert fd["--seed-islands"] is True
    assert fd["--micro-batch-pairs"] == 2
    assert combo.validate() == []


def test_micro_batch_notes_stamp_training_only_override_and_no_score_authority():
    lever = MicroBatch(2)
    note = lever.notes.lower()
    assert "training-only" in note
    assert "operator-waived" in note
    assert "no score authority" in note


def test_real_v9_program_with_micro_batch_two_compiles_and_parses_all_routed_legs():
    """Typed DSL -> WitnessProgram -> real trainer parser, not a hand-built argv surrogate."""
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config

    compiled = compile_v9_cgauge_432_launch_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600, epochs=3000, out_dir="experiments/results/__v9_mb2_parse_test__")
    program = compiled.typed.to_program().with_lever(MicroBatch(2))
    assert program.validate() == []
    argv = program.compile_trainer_argv()
    ns = build_real_trainer_parser().parse_args(argv[2:])
    assert ns.micro_batch_pairs == 2
    # Complete inventory of the trainer's six real micro-batch refusal predicates as they
    # existed before this landing. The typed V9 argv activates exactly phase + chroma; the
    # four still-unrouted predicates remain dormant. Pinning both sides prevents a future V9
    # DSL edit from silently activating a scattered guard.
    assert ns.margin_saliency_reachability is False
    assert ns.seg_spike_reweight is False
    assert ns.seg_subpix_boundary_weight == 0.0
    assert ns.seg_chroma_boundary_weight > 0.0
    assert ns.seg_phase_advect_weight > 0.0
    assert ns.eikonal_steik_normalized is False
    # V9-active semantics that had no explicit refusal but were previously omitted or share
    # the render composition must also remain in the compatibility receipt.
    assert ns.seg_temporal_screw_weight > 0.0
    assert ns.seg_temporal_screw_xi_source == "ground_gt"
    assert ns.seg_temporal_screw_sky_rotation_only is False
    assert ns.lane_band_weight > 0.0 and ns.lane_render_band is True
    assert ns.area_constraint_birth is True
    assert ns.birth_completion_event is True and ns.birth_completion_ramp is True
    assert ns.logit_adjust_loss_tau > 0.0
    assert ns.amplify_weight > 0.0 and ns.persistence_loss_weight > 0.0
    assert ns.cache_gt_skeleton is True
    assert ns.seed_islands is True and ns.witness_alone_island_loss is True
    # Live loss/render semantics omitted by the first inventory pass.  Keep these parser-backed:
    # pose-carrier changes the realized pair scored by the batched loss, unify-tau selects its base
    # seg form, and the entropy term is added once per model by the batched twin.  The carrier would
    # be semantically inert without the compiled terminal pose weight, so pin that coupling too.
    assert ns.pose_carrier is True and ns.pose_carrier_source == "generated"
    assert ns.w_pose == 1.0
    assert ns.seg_form_unify_tau is True
    assert ns.logit_adjust_classes == "3"
    assert ns.weight_entropy_penalty_lambda == 15.0
    # The two phase alternatives are intentionally fail-loud/spec-only. Canonical V9 must remain on
    # the fully implemented pair-local provider mode or the dry-start could still fail after parsing.
    assert ns.seg_phase_advect_gap_xi == "interp"
    assert ns.seg_phase_advect_ref == "gt_advected"


def test_v9_micro_batch_triality_notes_name_each_routed_or_shared_surface():
    """Each lever's DSL authority records its batched disposition; no stale fail-close prose."""
    from tac.witness_dsl.curriculum_dsl import (
        AnalyticLaneRenderBand,
        AreaConstraintBirth,
        BirthCompletionEvent,
        LogitAdjust,
        PhaseAdvectionConsistency,
        SegChromaBoundary,
        TemporalScrewConsistency,
    )

    notes = {
        "lane": AnalyticLaneRenderBand().notes.lower(),
        "area": AreaConstraintBirth().notes.lower(),
        "birth": BirthCompletionEvent(ramp_apply=True).notes.lower(),
        "logit": LogitAdjust(window=0).notes.lower(),
        "phase": PhaseAdvectionConsistency().notes.lower(),
        "chroma": SegChromaBoundary().notes.lower(),
        "temporal": TemporalScrewConsistency().notes.lower(),
    }
    assert "microbatch" in notes["lane"] or "micro-batch" in notes["lane"]
    for lever in ("area", "birth", "logit", "phase", "chroma", "temporal"):
        assert "micro-batch" in notes[lever], (lever, notes[lever])
    assert "fused metal" in notes["phase"]
    assert "fused metal" in notes["chroma"]
    assert "fused metal" in notes["temporal"]


def test_speed_levers_compose_and_validate_clean():
    combo = BASELINE.with_lever(CacheGtSkeleton(), MicroBatch(4))
    fd = combo.flag_dict()
    assert fd["--cache-gt-skeleton"] is True
    assert fd["--micro-batch-pairs"] == 4
    assert combo.validate() == []


def test_speed_levers_do_not_mutate_baseline():
    _ = BASELINE.with_lever(CacheGtSkeleton(), MicroBatch(4))
    assert "--cache-gt-skeleton" not in BASELINE.flag_dict()
    assert "--micro-batch-pairs" not in BASELINE.flag_dict()
