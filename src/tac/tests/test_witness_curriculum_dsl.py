"""Tests for the witness curriculum/behavior DSL (task #189, Layer-0 of the bridge).

Locks: structural never-invent-flags, the BASELINE round-trip against the completed
run, the enforced preserve/contain/authority clauses, and lever composition.
"""
from dataclasses import replace

import pytest

from tac.witness_dsl import (
    Anneal,
    Authority,
    Contain,
    Freeze,
    Preserve,
    WitnessProgram,
    real_trainer_flags,
    real_store_true_flags,
    BASELINE,
    PoseDecouple,
    Muon,
    DirectionalBasis,
    TauFrozen,
    SoftBoundary,
    FiLMFix,
    LanePrior,
    StiefelW,
    CodeSpectralEntropy,
    DM1Minimal,
)

# the exact flags the completed CE->tau->l7 run was launched with (grounded from the log)
_LAUNCHED = set("""--resume-from --out-dir --gt-cache --num-pairs --epochs --render-h --render-w
--hidden-dim --mod-dim --activation --siren-init --softmax-temp-start --softmax-temp-end
--curriculum --tau-softplus-start-epoch --l7-start-epoch --palette-anchor --self-orient
--reorient-every --freq-across --n-dir-freqs --freq-along --max-bank-freq --chroma
--lane-edge-weight --lane-edge-class --lane-margin-target --lane-edge-start-epoch --w-seg
--w-pose --eikonal-weight --length-weight --ema-decay --accum-pairs --grad-clip
--verdict-pairs --eval-every --ckpt-every --async-verdict --mlx-device""".split())


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


def test_review_M2_with_lever_can_clear_resume_from():
    fresh = BASELINE.with_lever(SoftBoundary(), resume_from=None)
    assert fresh.resume_from is None
    inherited = BASELINE.with_lever(SoftBoundary())  # default = inherit
    assert inherited.resume_from == BASELINE.resume_from


def test_soft_boundary_replaces_beta_steplim():
    arm = BASELINE.with_lever(SoftBoundary(2.0))
    assert arm.flag_dict()["--hosc-beta"] == 2.0 and arm.epochs > BASELINE.epochs
    assert arm.validate() == []


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
    openpilot_seeded_opening,
    StageDecision,
    StagePolicy,
    advance_to_l7,
    advance_to_muon,
    decide_next_stage,
    extend_stage,
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
    # a doomed config (l7_start > epochs) is refused at DSL-validate time (trainer assert surfaced)
    from dataclasses import replace
    from tac.witness_dsl import Stage
    bad = replace(_opening(), stages=(
        _opening().stages[0],
        Stage("tau_softplus", "--tau-softplus-start-epoch", 300),
        Stage("l7_softplus", "--l7-start-epoch", 9999),  # > epochs(600)
    ))
    assert any("CURRICULUM ORDERING" in p for p in bad.validate())


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
