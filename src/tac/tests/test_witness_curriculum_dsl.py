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
