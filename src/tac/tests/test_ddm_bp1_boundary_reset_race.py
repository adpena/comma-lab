"""ddm_bp1 (#824) — tests for the BOUNDARY RESET RACE build (arm A = B, arm B' = Bprime).

Discipline note (CLAUDE.md NO-FAKE #2, "tests-verify-constants-not-behavior"): the load-bearing
tests here are BEHAVIOURAL against the real ``mlx.optimizers.Adam`` — if the trainer stopped
passing ``bias_correction`` through, or MLX changed its default, ``test_mlx_adam_default_is_arm_b``
and ``test_bias_corrected_step_ratio_is_one_over_eta`` fail. They would NOT pass against a
``return canonical_markers`` stub. The pure helpers are tested on their real inputs (a real parent
checkpoint's ``telemetry_tail`` shape), and the DSL tests compile against the trainer's ACTUAL
argparse (never-invent-flags), not against a hand-written flag list.

No scorer claim is made anywhere here. Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

# OPT_STATE_DROP_OK: test FIXTURE checkpoints written to exercise save/load shape; no live
# resume ever reads them, so the #824 arm-B boundary impulse this gate prices cannot apply
# here (ddm_op2 OP2-1).

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[3]
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.train_tr1_partition_renderer_mlx import (  # noqa: E402
    RESET_ADAM_BETAS,
    TR1Config,
    boundary_jump_row,
    build_argparser,
    derive_jd1_lr_tail_schedule,
    derive_jd1_realized_hold_margin,
    derive_jd1_stage_ema_decay,
    derive_ema_decay,
    gate_interval_fields,
    jd1_ema_gate_basis_label,
    jd1_forced_resume_start_epoch,
    jd1_lr_at_epoch,
    jd1_should_reanchor_stage_ema,
    load_checkpoint,
    parent_boundary_ema_decay_fields,
    reset_arm_for,
    resume_ema_decay_fields,
    save_checkpoint,
    validate_jd1_pose_finish_args,
)
from tac.optimization.reset_operator import (  # noqa: E402
    ARM_B_ZERO_RESET,
    ARM_BPRIME_BIAS_CORRECTED,
    effective_lr_multiplier,
    resolve_arm_name,
)
from tac.witness_dsl.spec_tr1_renderer_20260728 import (  # noqa: E402
    TR1RendererProgramV1,
    bp1_boundary_reset_race_program,
    lever_boundary_probe,
    lever_reset_operator,
    trainer_declared_flags,
)

mx = pytest.importorskip("mlx.core")


def _cfg(**kw) -> TR1Config:
    base = {
        "variant": "plain", "num_pairs": 4, "grid_downsample": 16, "code_width": 2,
        "renderer_width": 8, "token_quant_levels": 16, "seed": 3, "lotto_seed": 118,
        "lotto_mask_density_init": 0.5, "seg_form_start": "ce", "w_seg": 100.0, "lr": 1e-3,
        "batch_pairs": 2, "epochs": 2, "gate_every": 1, "ema_decay": 0.95,
        "ema_decay_provenance": "test", "token_temporal_mode": "shared_base",
        "token_ste": "round", "class_weight_lane": 1.0, "margin_target": 1.0}
    base.update(kw)
    return TR1Config(**base)


# ----------------------------------------------------- the optimizer contract ----
def _adam_first_steps(bias_correction, n=4, lr=1e-2):
    """Max |delta w| per step under a constant unit gradient — the honest first-order kick model."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(0)
    m = nn.Linear(4, 3)
    mx.eval(m.parameters())
    opt = (optim.Adam(learning_rate=lr) if bias_correction is None
           else optim.Adam(learning_rate=lr, bias_correction=bias_correction))
    g = {"weight": mx.ones_like(m.weight), "bias": mx.ones_like(m.bias)}
    prev, out = mx.array(m.weight), []
    for _ in range(n):
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        out.append(float(mx.max(mx.abs(m.weight - prev))))
        prev = mx.array(m.weight)
    return out


def test_mlx_adam_default_is_arm_b():
    """Arm A byte-identity: ``Adam(lr)`` and ``Adam(lr, bias_correction=False)`` must be the SAME
    optimizer. If MLX ever changed that default, the control arm would silently stop being the
    incumbent and every pre-#824 comparison would break."""
    assert _adam_first_steps(None) == _adam_first_steps(False)
    assert _adam_first_steps(None) != _adam_first_steps(True)


def test_bias_corrected_step_ratio_is_one_over_eta():
    """Cross-validate reset_operator's CLOSED FORM against the real optimizer (behaviour, not a
    recalled constant): corrected/uncorrected step ratio == 1/eta(t), eta=(1-b1^t)/sqrt(1-b2^t)."""
    unc, cor = _adam_first_steps(False), _adam_first_steps(True)
    for t, (u, c) in enumerate(zip(unc, cor, strict=True), start=1):
        assert c / u == pytest.approx(1.0 / effective_lr_multiplier(t, RESET_ADAM_BETAS), rel=1e-4)


def test_bias_corrected_first_step_is_lr_sign_g():
    """The corrected update at a constant gradient is exactly ``lr * sign(g)`` — the property that
    makes arm B' the 'no free displacement' arm."""
    for s in _adam_first_steps(True, n=3, lr=1e-2):
        assert s == pytest.approx(1e-2, rel=2e-5)


def test_reset_adam_betas_match_the_installed_mlx_default():
    """The betas constant is not a guess: read it back off the real ``Adam`` signature."""
    import inspect

    import mlx.optimizers as optim

    default = inspect.signature(optim.Adam.__init__).parameters["betas"].default
    assert tuple(float(b) for b in default) == RESET_ADAM_BETAS


def test_adam_bias_correction_gate_passes_through_at_tr1_beta2():
    """The trainer REUSES the levelset gate rather than reimplementing it; at MLX's default
    beta2 the gate must be the identity on ``reference_semantics`` (else the arm selector lies)."""
    from experiments.train_levelset_witness_realized_through_R_mlx import (
        _adam_bias_correction_for,
    )

    assert _adam_bias_correction_for(RESET_ADAM_BETAS[1], reference_semantics=False) is False
    assert _adam_bias_correction_for(RESET_ADAM_BETAS[1], reference_semantics=True) is True


# ------------------------------------------------------------- arm selection ----
def test_default_config_is_the_incumbent_arm_b():
    cfg = _cfg()
    assert cfg.adam_bias_correction is False
    assert reset_arm_for(cfg) is ARM_B_ZERO_RESET
    assert ARM_B_ZERO_RESET.is_incumbent


def test_bias_correction_on_selects_arm_bprime():
    arm = reset_arm_for(_cfg(adam_bias_correction=True))
    assert arm is ARM_BPRIME_BIAS_CORRECTED
    assert resolve_arm_name(arm) == "Bprime"
    assert not arm.is_incumbent


def test_both_reachable_arms_need_no_optimizer_state_persistence():
    """The whole reason A and C are out of scope: they would need ``opt_state_flat`` plumbing that
    this trainer does not have. B and B' must not."""
    for arm in (ARM_B_ZERO_RESET, ARM_BPRIME_BIAS_CORRECTED):
        assert arm.requires_persistence is False


def test_arm_selector_changes_the_config_identity():
    """A knob that changes TRAINING must change the config hash (it is not observability)."""
    assert _cfg().config_hash() != _cfg(adam_bias_correction=True).config_hash()


# ------------------------------------------------------- checkpoint compatibility ----
def test_pre_824_checkpoint_cfg_still_constructs(tmp_path):
    """RESUME COMPAT: the arms resume from checkpoints written BEFORE this field existed. A cfg
    dict lacking the key must still build a TR1Config (defaulting to the incumbent arm)."""
    d = dict(_cfg(adam_bias_correction=True).__dict__)
    d.pop("adam_bias_correction")
    assert TR1Config(**d).adam_bias_correction is False


def test_config_roundtrips_through_a_checkpoint(tmp_path):
    import mlx.nn as nn

    from experiments.train_tr1_partition_renderer_mlx import build_module

    cfg = _cfg(adam_bias_correction=True)
    model = build_module(cfg)
    mx.eval(model.parameters())
    p = tmp_path / "ck.npz"
    save_checkpoint(p, model=model, ema={}, opt_state_flat={}, epoch=7, stage="s",
                    cfg=cfg, telemetry_tail=[])
    st = load_checkpoint(p, build_module(cfg))
    assert st["meta"]["cfg"]["adam_bias_correction"] is True
    assert isinstance(model, nn.Module)


# ------------------------------------------------------- the boundary instrument ----
_TAIL = [
    {"event": "a1_gate", "epoch": 940, "realized_gate_dseg_mean": 0.00430,
     "gate_params": "ema_shadow"},
    {"event": "lane_guard", "epoch": 944},                       # no gate reading -> ignored
    {"event": "a1_gate", "epoch": 945, "realized_gate_dseg_mean": 0.004201253255208332,
     "gate_params": "ema_shadow"},
]


def _gate(epoch=949, dseg=0.00405):
    return {"event": "a1_gate", "epoch": epoch, "realized_gate_dseg_mean": dseg,
            "gate_params": "ema_shadow"}


def test_boundary_jump_picks_the_latest_real_anchor_and_prices_the_interval():
    row = boundary_jump_row(_TAIL, 0.99994362, 0.99994362, 947, _gate(), "Bprime")
    assert row["parent_gate_epoch"] == 945           # NOT the epoch-944 row (no gate reading)
    assert row["boundary_span_epochs"] == 4
    assert row["boundary_dseg_delta"] == pytest.approx(0.00405 - 0.004201253255208332)
    assert row["boundary_dseg_per_epoch"] == pytest.approx(row["boundary_dseg_delta"] / 4)
    assert row["ema_basis_held"] is True
    assert row["score_claim"] is False and "ADVISORY" in row["caveat"]
    assert row["arm"] == "Bprime"


def test_boundary_jump_flags_ema_basis_drift():
    """The confound that corrupted the burn: the gate reads the EMA shadow, so a parent/child
    decay mismatch makes the two readings incommensurable. It must be VISIBLE on the row."""
    held = boundary_jump_row(_TAIL, 0.99994362, 0.99994362, 947, _gate(), "B")
    drift = boundary_jump_row(_TAIL, 0.99993383, 0.99994362, 947, _gate(), "B")
    assert held["ema_basis_held"] is True
    assert drift["ema_basis_held"] is False
    assert boundary_jump_row(_TAIL, None, 0.99994362, 947, _gate(), "B")["ema_basis_held"] is False


def test_boundary_jump_uses_parent_active_jd1_decay_not_cfg_decay():
    meta = {
        "cfg": {"ema_decay": 0.999960019990005},
        "jd1_pose_finish": {
            "active_ema_decay": 0.9997777777777778,
            "active_ema_decay_provenance": "JD1 stage-scoped U=18000",
        },
    }
    fields = parent_boundary_ema_decay_fields(meta)
    assert fields["parent_cfg_ema_decay"] == pytest.approx(0.999960019990005)
    assert fields["parent_ema_decay"] == pytest.approx(0.9997777777777778)
    row = boundary_jump_row(
        _TAIL,
        fields["parent_boundary_ema_decay"],
        0.9997777777777778,
        947,
        _gate(),
        "B",
        parent_cfg_ema_decay=fields["parent_cfg_ema_decay"],
    )
    assert row["ema_decay_held"] is True
    assert row["ema_basis_held"] is True
    assert row["parent_cfg_ema_decay"] == pytest.approx(0.999960019990005)


def test_resume_event_fields_carry_post_restore_active_decay():
    parent = parent_boundary_ema_decay_fields({
        "cfg": {"ema_decay": 0.999960019990005},
        "jd1_pose_finish": {"active_ema_decay": 0.9997777777777778},
    })
    row = resume_ema_decay_fields(
        parent,
        child_cfg_ema_decay=0.999960019990005,
        active_ema_decay=0.9997777777777778,
        active_ema_decay_provenance="restored from checkpoint jd1 state",
    )
    assert row["child_cfg_ema_decay"] == pytest.approx(0.999960019990005)
    assert row["post_restore_active_ema_decay"] == pytest.approx(0.9997777777777778)
    assert row["child_ema_decay"] == pytest.approx(0.9997777777777778)
    assert row["ema_decay_held"] is True


def test_boundary_jump_returns_none_without_an_anchor():
    assert boundary_jump_row([], 0.9, 0.9, 1, _gate(), "B") is None
    assert boundary_jump_row([{"epoch": 3}], 0.9, 0.9, 1, _gate(), "B") is None
    assert boundary_jump_row(_TAIL, 0.9, 0.9, 1, {"epoch": 949}, "B") is None


def test_boundary_jump_reads_a_real_parent_checkpoint_shape(tmp_path):
    """Runs against the REAL producer: a checkpoint written by save_checkpoint, whose meta carries
    the same telemetry_tail shape a live parent has (not a hand-invented dict)."""
    from experiments.train_tr1_partition_renderer_mlx import build_module

    cfg = _cfg(ema_decay=0.99994362)
    model = build_module(cfg)
    mx.eval(model.parameters())
    p = tmp_path / "parent.npz"
    save_checkpoint(p, model=model, ema={}, opt_state_flat={}, epoch=946, stage="seg_trunk_tau",
                    cfg=cfg, telemetry_tail=list(_TAIL))
    meta = json.loads(bytes(np.load(p, allow_pickle=False)["meta::json"]).decode())
    row = boundary_jump_row(meta["telemetry_tail"], float(meta["cfg"]["ema_decay"]),
                            cfg.ema_decay, 947, _gate(), "B")
    assert row is not None and row["ema_basis_held"] is True
    assert row["parent_gate_epoch"] == 945


def test_gate_interval_fields():
    assert gate_interval_fields(None, _gate())["interval_epochs"] is None
    f = gate_interval_fields(_gate(944, 0.0043), _gate(949, 0.0041))
    assert f["interval_epochs"] == 5
    assert f["interval_dseg_delta"] == pytest.approx(-0.0002)
    assert f["interval_dseg_per_epoch"] == pytest.approx(-0.00004)
    # malformed / same-epoch inputs degrade to None, never to a divide-by-zero or a wrong number
    assert gate_interval_fields({"epoch": 1}, _gate())["interval_dseg_delta"] is None
    assert gate_interval_fields(_gate(949, 0.1), _gate(949, 0.2))["interval_dseg_per_epoch"] is None


def test_boundary_positive_control_tolerance_is_sub_pixel():
    """The re-gate tolerance is DERIVED as half a single-pixel quantum, so it accepts only a
    bit-exact reproduction. Guard the derivation, not a literal."""
    from experiments.train_tr1_partition_renderer_mlx import SEG_H, SEG_W, resolve_gate_ids

    n = len(resolve_gate_ids(600))
    tol = 0.5 / float(n * SEG_H * SEG_W)
    assert tol < 1.0 / float(n * SEG_H * SEG_W)   # strictly below one flipped pixel
    assert n == 36


# ------------------------------------------------------------------ the DSL ----
def test_levers_are_declared_by_the_real_trainer_argparse():
    """never-invent-flags, checked against the trainer's ACTUAL argparse (AST), not a list."""
    declared = trainer_declared_flags()
    assert "--adam-bias-correction" in declared
    assert "--boundary-probe" in declared


def test_arm_selector_lever_values():
    assert lever_reset_operator("B").overrides == {"--adam-bias-correction": "off"}
    assert lever_reset_operator("Bprime").overrides == {"--adam-bias-correction": "on"}
    for bad in ("A", "C", "Dplus", ""):
        with pytest.raises(ValueError, match="reachable"):
            lever_reset_operator(bad)
    with pytest.raises(ValueError, match=r"off\|on"):
        lever_boundary_probe("yes")


def test_levers_compile_to_valued_flags_never_a_stray_bool():
    """The store_true seal break: a bool override stringifies to 'True' and argparse would eat the
    next token. Both new flags must compile as VALUED tokens."""
    prog = TR1RendererProgramV1(
        levers=(lever_reset_operator("Bprime"), lever_boundary_probe("on")),
        num_pairs=4, out_dir="/tmp/x", seed=0)
    argv = prog.compile_trainer_argv()
    assert "True" not in argv and "False" not in argv
    assert argv[argv.index("--adam-bias-correction") + 1] == "on"
    assert argv[argv.index("--boundary-probe") + 1] == "on"


def test_compiled_flags_parse_through_the_real_argparser():
    """Execute the contract, do not trace it: the compiled argv must actually parse."""
    prog = TR1RendererProgramV1(
        levers=(lever_reset_operator("Bprime"), lever_boundary_probe("on")),
        num_pairs=4, out_dir="/tmp/x", seed=0)
    argv = [*prog.compile_trainer_argv()[1:], "--variant", "plain"]
    ns = build_argparser().parse_args(argv)
    assert ns.adam_bias_correction == "on" and ns.boundary_probe == "on"
    assert build_argparser().parse_args(["--variant", "plain", "--out-dir", "/tmp/x"]
                                        ).adam_bias_correction == "off"


def _race_argvs(**kw):
    parent = (lever_reset_operator("B"),)  # a stale arm lever the program must supersede
    common: dict = {"resume_from": "/p/ck.npz", "ema_decay": 0.99994362, "epochs": 987,
                    "max_wall_minutes": 44.6, "parent_levers": parent, "gt_cache": "/p/gt.npz"}
    common.update(kw)
    a = bp1_boundary_reset_race_program("B", out_dir="/o/a", **common).compile_trainer_argv()
    b = bp1_boundary_reset_race_program("Bprime", out_dir="/o/b", **common).compile_trainer_argv()
    return a, b


def test_race_arms_differ_in_exactly_one_token():
    a, b = _race_argvs()
    strip = lambda v: [t for i, t in enumerate(v)                      # noqa: E731
                       if t != "--out-dir" and v[i - 1] != "--out-dir"]
    a2, b2 = strip(a), strip(b)
    assert len(a2) == len(b2)
    pos = [i for i, (x, y) in enumerate(zip(a2, b2, strict=True)) if x != y]
    assert len(pos) == 1
    assert a2[pos[0] - 1] == "--adam-bias-correction"
    assert (a2[pos[0]], b2[pos[0]]) == ("off", "on")


def test_race_program_pins_the_ema_basis_and_supersedes_stale_levers():
    a, _ = _race_argvs()
    assert a[a.index("--ema-decay") + 1] == repr(0.99994362)
    assert a[a.index("--epochs") + 1] == "987"
    assert a[a.index("--boundary-probe") + 1] == "on"
    assert a.count("--adam-bias-correction") == 1   # the stale parent arm lever was superseded
    assert a.count("--epochs") == 1


# ---------------------------------------------------------- the ticket builder ----
def test_window_length_is_derived_from_the_impulse():
    from tools.build_ddm_bp1_arm_tickets import derive_window_epochs

    g = derive_window_epochs(75, 5)
    assert g["impulse_epochs"] == pytest.approx(16.1676, abs=1e-3)
    assert g["window_epochs"] == 40                      # 2x16.17 -> 35 on the lattice, +1 gate
    assert g["window_epochs"] >= 2 * g["impulse_epochs"]
    assert g["window_epochs"] % 5 == 0
    # the requirement, not the number, is what must hold when the geometry changes:
    g2 = derive_window_epochs(150, 10)
    assert g2["window_epochs"] >= 2 * g2["impulse_epochs"] and g2["window_epochs"] % 10 == 0


def test_argv_diff_is_positional_not_membership():
    """Regression for a real bug found during this build: a set-membership diff reported ZERO
    difference because the tokens 'on'/'off' also appear as OTHER levers' values."""
    from tools.build_ddm_bp1_arm_tickets import argv_diff

    a = ["--out-dir", "/a", "--telemetry-v9-port", "on", "--adam-bias-correction", "off"]
    b = ["--out-dir", "/b", "--telemetry-v9-port", "on", "--adam-bias-correction", "on"]
    d = argv_diff(a, b)
    assert d["identical_except"] is True
    assert d["differing_flag"] == ["--adam-bias-correction"]
    assert (d["only_in_arm_A"], d["only_in_arm_Bprime"]) == (["off"], ["on"])
    assert d["out_dir_A"] == "/a" and d["out_dir_Bprime"] == "/b"
    # two real differences must NOT be reported as a clean one-flag A/B
    c = list(b)
    c[3] = "off"
    assert argv_diff(a, c)["identical_except"] is False


# ------------------------------------------- round-2 corrections (MAIN, seal-blocking) ----
def test_ema_shadow_is_continuous_across_a_resume_not_re_anchored():
    """Round-2 correction, VERIFIED_VIA_SOURCE_INSPECTION: the boundary resets are TWO (Adam
    moments, EMA decay VALUE), not three. The shadow is LOADED from the checkpoint, so it is
    continuous. Any doc that claims a shadow re-anchor is wrong; guard the source fact."""
    src = (WORKTREE / "experiments/train_tr1_partition_renderer_mlx.py").read_text()
    assert 'ema = st["ema"]' in src, "the resume must LOAD the shadow (continuous across boundaries)"


def test_gate_basis_differs_between_fresh_and_resumed_runs():
    """The unnamed fourth reset — a MEASUREMENT-BASIS reset. A resumed run starts at
    global_step = ema_warmup_updates so its first gate reads ``ema_shadow``; a fresh run reads
    ``live_ema_warmup`` for its first U/2 updates. Mixing a fresh arm with a resumed arm compares
    two different instruments. Re-derived from source, since it is the seal-blocking invariant."""
    src = (WORKTREE / "experiments/train_tr1_partition_renderer_mlx.py").read_text()
    assert "global_step = 0 if args.resume_from is None else ema_warmup_updates" in src
    assert "jd1_ema_gate_basis_label(" in src
    assert jd1_ema_gate_basis_label(
        global_step=0,
        ema_warmup_updates=1,
        state={},
    ) == "live_ema_warmup"
    assert jd1_ema_gate_basis_label(
        global_step=1,
        ema_warmup_updates=1,
        state={},
    ) == "ema_shadow"


def test_builder_refuses_arms_that_are_not_on_the_same_gate_basis():
    """The invariant is enforced where it CAN be: only the builder sees both arms. Behavioural."""
    from tools.build_ddm_bp1_arm_tickets import check_same_gate_basis

    same = {"A": ["--resume-from", "/p/ck.npz"], "Bprime": ["--resume-from", "/p/ck.npz"]}
    ok, why, res = check_same_gate_basis(same)
    assert ok and res["A"] == "/p/ck.npz" and "same gate basis" in why

    mixed = {"A": ["--resume-from", "/p/ck.npz"], "Bprime": ["--epochs", "40"]}
    ok, why, _ = check_same_gate_basis(mixed)
    assert not ok and "different gate BASIS" in why

    diff_ck = {"A": ["--resume-from", "/p/a.npz"], "Bprime": ["--resume-from", "/p/b.npz"]}
    assert check_same_gate_basis(diff_ck)[0] is False

    fresh = {"A": ["--epochs", "40"], "Bprime": ["--epochs", "40"]}
    ok, why, _ = check_same_gate_basis(fresh)
    assert not ok and "FRESH" in why


def test_a1_alarm_summary_is_first_class():
    """The corroborating channel MAIN found unread: 6 firings, none at a final gate, so every
    decision record missed all six. It must now be summarized, with its epochs."""
    from experiments.train_tr1_partition_renderer_mlx import a1_alarm_summary

    rows = [
        {"epoch": 649, "a1_alarm": True, "a1_classification": "A1_REALIZATION_GAP_ALARM"},
        {"epoch": 654, "a1_alarm": False, "a1_classification": "COUPLED_DESCENT"},
        {"epoch": 659, "a1_alarm": True, "a1_classification": "A1_REALIZATION_GAP_ALARM"},
    ]
    s = a1_alarm_summary(rows)
    assert s["a1_alarm_count"] == 2
    assert s["a1_alarm_epochs"] == [649, 659]
    assert s["gates_seen"] == 3
    assert "COUPLED_DESCENT" in s["a1_classifications"]
    assert a1_alarm_summary([])["a1_alarm_count"] == 0


def test_window_receipt_and_boundary_row_carry_the_new_channels():
    """Structural: the receipt must propagate the arm, the alarm summary and the basis mode —
    the three things whose ABSENCE from the burn's decision records hid the finding."""
    src = (WORKTREE / "experiments/train_tr1_partition_renderer_mlx.py").read_text()
    receipt = src.split('"schema": "ddm_tb1_tr1_window_receipt.v1"', 1)[1][:1600]
    for key in ('"reset_arm"', '"a1_alarms"', '"gate_basis_mode"', '"ema_basis_held"'):
        assert key in receipt, f"window receipt does not propagate {key}"
    # The CURRENT gate must be included: telemetry_tail is appended AFTER the boundary block, so
    # at the first post-resume gate the tail is empty and a bare `telemetry_tail` would report
    # 0 alarms even when THIS gate alarmed.
    assert '_bj["a1_alarms"] = a1_alarm_summary([*telemetry_tail, gate_row])' in src
    assert '_bj["gate_basis_mode"] = "resumed_warm_shadow"' in src
    # ...and the append really does come after (the reason the splice is needed).
    # ddm_bs3 (#909): located STRUCTURALLY (the parsed call node), not by matching the
    # append's exact argument text. The old locator was
    # ``src.index("telemetry_tail.append(dict(gate_row.items()))")`` -- a substring that
    # pinned the ARGUMENT EXPRESSION while the invariant under test is ORDERING, so a
    # semantically identical rewrite of the argument broke it (it did). Same
    # wrong-projection shape this suite exists to catch, one level up.
    import ast

    tree = ast.parse(src)
    gate_appends = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "append"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "telemetry_tail"
        and "gate_row" in ast.dump(n)
    ]
    assert gate_appends, "no telemetry_tail.append(... gate_row ...) found -- VACUOUS, not a pass"
    boundary_lineno = src[: src.index('_bj["a1_alarms"]')].count("\n") + 1
    assert min(gate_appends) > boundary_lineno, (
        f"gate-row append at line {min(gate_appends)} must come AFTER the boundary block "
        f"at line {boundary_lineno}")


def test_trainer_actually_wires_the_arm_into_its_optimizer():
    """The ONE gap the behavioural optimizer tests above cannot close: they exercise MLX, not this
    trainer's ``main()`` (which needs the scorer + GT cache and is out of scope for a unit test).

    Labelled honestly: VERIFIED_VIA_SOURCE_INSPECTION, not behavioural. Its backstop is the
    runtime assertion in ``main()`` that refuses to launch if the gate's return disagrees with
    ``cfg.adam_bias_correction``. Together they mean a broken wiring fails either this test or the
    launch — never silently trains the wrong arm.
    """
    import ast

    src = (WORKTREE / "experiments/train_tr1_partition_renderer_mlx.py").read_text()
    tree = ast.parse(src)
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def enclosing_function(node):
        cur = node
        while cur in parent:
            cur = parent[cur]
            if isinstance(cur, ast.FunctionDef):
                return cur.name
        return None

    adam_calls = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "Adam"]
    boundary_adam_calls = [
        n for n in adam_calls
        if enclosing_function(n) != "build_tr1_jd1_muon_finisher_optimizer"
    ]
    assert len(boundary_adam_calls) == 1, (
        "more than one non-finisher optimizer construction => an unwired arm is possible")
    kw = {k.arg for k in boundary_adam_calls[0].keywords}
    assert "bias_correction" in kw, "the arm selector is not passed to the optimizer"

    gate_calls = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "_adam_bias_correction_for"]
    assert len(gate_calls) == 1
    ref = {k.arg: k.value for k in gate_calls[0].keywords}["reference_semantics"]
    assert isinstance(ref, ast.Attribute) and ref.attr == "adam_bias_correction", (
        "the gate must be driven by cfg.adam_bias_correction, not a literal")

    lr_assigns = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and any(isinstance(t, ast.Attribute)
                and t.attr == "learning_rate"
                and isinstance(t.value, ast.Name)
                and t.value.id == "optimizer"
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
    ]
    assert lr_assigns, "JD1 LR anneal must reach optimizer.learning_rate in main()"
    assert any("jd1_lr_current" in ast.dump(n.value) for n in lr_assigns), (
        "optimizer.learning_rate must be driven by the derived JD1 schedule value")
    lr_schedule_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "jd1_lr_at_epoch"
    ]
    assert lr_schedule_calls, "main() never consumes jd1_lr_at_epoch()"


# ------------------------------------------ launcher G4 slot gate (bug found in this build) ----
_PS = """  PID COMMAND
 6443 /bin/zsh -c source /Users/x/.claude/shell-snapshots/snap.sh && export A=1 train_tr1_partition_renderer_mlx
 6445 rtk grep -rln train_tr1_partition_renderer_mlx\\|tr1_config\\|TR1Config experiments/ tools/ src/
 6447 /usr/bin/grep -rln train_tr1_partition_renderer_mlx.py experiments/
 7001 /repo/.venv/bin/python tools/launch_tr1_run.py --ticket /t/a.json --dry-run
 7002 /repo/.venv/bin/python -m pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py
 9999 /repo/.venv/bin/python experiments/train_tr1_partition_renderer_mlx.py --num-pairs 600
"""


def test_slot_gate_counts_only_real_python_trainer_processes():
    """REGRESSION for the false REFUSE found during this build's own dry-run: a background
    ``rtk grep -rln train_tr1_partition_renderer_mlx|...`` held the slot and blocked the launch.
    A guard that fires when nothing is wrong is a guard that gets routed around."""
    from tools.launch_tr1_run import slot_holders

    holders, mentions = slot_holders(_PS, self_pid=7002)
    assert len(holders) == 1 and " 9999 " in holders[0]
    # the zsh, both greps mention a trainer but hold nothing — and are REPORTED, not swallowed
    assert len(mentions) == 3
    assert all("9999" not in m for m in mentions)


def test_slot_gate_still_catches_the_real_launch_forms():
    from tools.launch_tr1_run import slot_holders

    for line in (
        " 1 /repo/.venv/bin/python experiments/train_tr1_partition_renderer_mlx.py --variant lotto",
        " 2 /usr/bin/python3 /abs/experiments/train_levelset_witness_realized_through_R_mlx.py -x",
        " 3 python3.13 experiments/train_witness_realized_through_R_mlx.py",
    ):
        holders, _ = slot_holders("  PID COMMAND\n" + line, self_pid=0)
        assert len(holders) == 1, f"missed a real trainer: {line}"


def test_slot_gate_ignores_itself_and_the_launcher():
    from tools.launch_tr1_run import slot_holders

    assert slot_holders(_PS, self_pid=9999)[0] == []          # never blocks on its own pid
    only_launcher = ("  PID COMMAND\n 7001 /repo/.venv/bin/python tools/launch_tr1_run.py "
                     "-- /repo/.venv/bin/python experiments/train_tr1_partition_renderer_mlx.py")
    assert slot_holders(only_launcher, self_pid=0)[0] == []


def test_eta_impulse_is_a_converged_sum_not_a_window_artifact():
    from tac.optimization.reset_operator import cumulative_excess_sign_steps

    assert cumulative_excess_sign_steps(20_000) == pytest.approx(
        cumulative_excess_sign_steps(40_000), rel=1e-6)
    assert math.isclose(cumulative_excess_sign_steps(20_000), 1212.57, abs_tol=0.05)


# --------------------------------------------------------------- jd3 reroute ----
def test_jd3_realized_hold_margin_derives_from_first_gate_uncertainty():
    args = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/jd3-test"),
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-pose-finish-engage-on", "start_epoch",
        "--jd1-pose-finish-start-epoch", "1",
        "--jd1-w-pose", "1.0",
        "--jd1-seg-hold-weight", "0.25",
        "--jd1-seg-hold-floor-source", "last_pre_pose_epoch_loss",
        "--jd1-seg-hold-space", "realized",
        "--jd1-live-gate-telemetry", "on",
    ])
    gate_row = {
        "realized_gate_pair_ids": list(range(36)),
        "realized_gate_dseg_per_pair_sd": 0.00072,
    }
    margin, prov = derive_jd1_realized_hold_margin(args, gate_row)
    assert margin == pytest.approx(0.00072 / 6.0)
    assert "sqrt(n_gate=36)" in prov


def test_jd3_stage_ema_uses_remaining_window_not_parent_chain():
    decay, prov = derive_jd1_stage_ema_decay(remaining_epochs=8, steps_per_epoch=150)
    expected, _ = derive_ema_decay(8 * 150)
    parent_chain, _ = derive_ema_decay(1336 * 75)
    assert decay == expected
    assert decay != parent_chain
    assert "stage-scoped window" in prov


def test_jd4_force_ema_reanchor_flag_fails_closed_when_inert():
    off_args = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/jd4-test"),
        "--jd1-force-ema-reanchor-on-resume",
    ])
    with pytest.raises(SystemExit, match="JD1 value flags set"):
        validate_jd1_pose_finish_args(off_args)

    no_resume = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/jd4-test"),
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-pose-finish-engage-on", "start_epoch",
        "--jd1-pose-finish-start-epoch", "1",
        "--jd1-w-pose", "1.0",
        "--jd1-seg-hold-weight", "0.25",
        "--jd1-seg-hold-floor-source", "last_pre_pose_epoch_loss",
        "--jd1-ema-stage-scope", "window",
        "--jd1-force-ema-reanchor-on-resume",
    ])
    with pytest.raises(SystemExit, match="requires --resume-from"):
        validate_jd1_pose_finish_args(no_resume)

    wrong_scope = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/jd4-test"),
        "--resume-from", "parent.npz",
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-pose-finish-engage-on", "start_epoch",
        "--jd1-pose-finish-start-epoch", "1",
        "--jd1-w-pose", "1.0",
        "--jd1-seg-hold-weight", "0.25",
        "--jd1-seg-hold-floor-source", "last_pre_pose_epoch_loss",
        "--jd1-force-ema-reanchor-on-resume",
    ])
    with pytest.raises(SystemExit, match="requires --jd1-ema-stage-scope window"):
        validate_jd1_pose_finish_args(wrong_scope)


def test_jd4_force_ema_reanchor_ignores_carried_latch_only_on_resume():
    base = [
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/jd4-test"),
        "--resume-from", "parent.npz",
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-pose-finish-engage-on", "start_epoch",
        "--jd1-pose-finish-start-epoch", "1",
        "--jd1-w-pose", "1.0",
        "--jd1-seg-hold-weight", "0.25",
        "--jd1-seg-hold-floor-source", "last_pre_pose_epoch_loss",
        "--jd1-ema-stage-scope", "window",
    ]
    legacy = build_argparser().parse_args(base)
    forced = build_argparser().parse_args([*base, "--jd1-force-ema-reanchor-on-resume"])
    state = {"engaged": True, "stage_ema_reanchored": True}

    validate_jd1_pose_finish_args(legacy)
    validate_jd1_pose_finish_args(forced)
    assert not jd1_should_reanchor_stage_ema(
        legacy, state, reason="resume_inside_joint_pose_finish")
    assert jd1_should_reanchor_stage_ema(
        forced, state, reason="resume_inside_joint_pose_finish")
    assert not jd1_should_reanchor_stage_ema(
        forced, state, reason="fresh_joint_pose_engagement")


def test_jd4_forced_resume_start_epoch_uses_terminal_tail_geometry():
    start, row = jd1_forced_resume_start_epoch(
        saved_epoch=1406,
        checkpoint_tail=[{"epoch": 1402}, {"epoch": 1405}],
        force_reanchor_on_resume=True,
    )
    assert start == 1406
    assert row is not None
    assert row["legacy_start_epoch"] == 1407
    assert row["forced_start_epoch"] == 1406

    legacy_start, legacy_row = jd1_forced_resume_start_epoch(
        saved_epoch=1406,
        checkpoint_tail=[{"epoch": 1405}],
        force_reanchor_on_resume=False,
    )
    assert legacy_start == 1407
    assert legacy_row is None

    intra_start, intra_row = jd1_forced_resume_start_epoch(
        saved_epoch=1404,
        checkpoint_tail=[{"epoch": 1404}],
        force_reanchor_on_resume=True,
    )
    assert intra_start == 1405
    assert intra_row is None


def test_la1_jd1_lr_anneal_flags_fail_closed_when_inert_or_unresumable():
    defaults = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/la1-test"),
    ])
    validate_jd1_pose_finish_args(defaults)
    assert defaults.jd1_lr_anneal == "off"
    assert defaults.jd1_lr_final_frac == 0.0

    inert = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/la1-test"),
        "--jd1-lr-anneal", "derived_tail",
    ])
    with pytest.raises(SystemExit, match="JD1 value flags set"):
        validate_jd1_pose_finish_args(inert)

    no_resume = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/la1-test"),
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-pose-finish-engage-on", "start_epoch",
        "--jd1-pose-finish-start-epoch", "1",
        "--jd1-w-pose", "1.0",
        "--jd1-lr-anneal", "derived_tail",
    ])
    with pytest.raises(SystemExit, match="requires --resume-from"):
        validate_jd1_pose_finish_args(no_resume)

    dangling_frac = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/la1-test"),
        "--resume-from", "parent.npz",
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-pose-finish-engage-on", "start_epoch",
        "--jd1-pose-finish-start-epoch", "1",
        "--jd1-w-pose", "1.0",
        "--jd1-lr-final-frac", "0.5",
    ])
    with pytest.raises(SystemExit, match="requires --jd1-lr-anneal"):
        validate_jd1_pose_finish_args(dangling_frac)


def test_la1_jd1_lr_anneal_derives_tail_from_parent_telemetry(tmp_path):
    run_dir = tmp_path / "parent"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    resume_from = ckpt_dir / "stage_joint_pose_finish_final.npz"
    resume_from.write_bytes(b"fixture")
    values = [0.91 + (0.014 if i % 2 else -0.010) + i * 1e-4 for i in range(70)]
    with (run_dir / "telemetry.jsonl").open("w") as fh:
        for i, value in enumerate(values):
            fh.write(json.dumps({
                "event": "epoch",
                "epoch": 1500 + i,
                "stage": "joint_pose_finish",
                "jd1_pose_finish_active": True,
                "ep_loss": value,
            }) + "\n")

    sched = derive_jd1_lr_tail_schedule(
        base_lr=2e-3,
        start_epoch=1526,
        end_epoch=1646,
        steps_per_epoch=150,
        beta2=0.999,
        active_ema_decay=0.9997777777777778,
        resume_from=resume_from,
    )
    tail = np.asarray(values[-60:], dtype=np.float64)
    half_range = float((tail.max() - tail.min()) / 2.0)
    sd = float(tail.std())
    assert sched["tail_epochs"] == 60
    assert sched["onset_epoch"] == 1586
    assert sched["beta2_memory_epochs_c2"] == 14
    assert sched["active_ema_memory_epochs_c2"] == 60
    assert sched["signal_source"] == "epoch.ep_loss[jd1_pose_finish_active]"
    assert sched["final_frac"] == pytest.approx(sd / (sd + half_range))
    assert jd1_lr_at_epoch(1585, sched) == pytest.approx(2e-3)
    assert jd1_lr_at_epoch(1645, sched) == pytest.approx(sched["final_lr"])


def test_jd3_realized_hold_flags_fail_closed_when_declared_but_unread():
    args = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/jd3-test"),
        "--jd1-seg-hold-space", "realized",
        "--jd1-live-gate-telemetry", "on",
    ])
    with pytest.raises(SystemExit, match="JD1 value flags set"):
        validate_jd1_pose_finish_args(args)


def test_jd3_realized_hold_requires_live_basis_telemetry():
    args = build_argparser().parse_args([
        "--variant", "plain",
        "--out-dir", str(WORKTREE / "scratchpad/jd3-test"),
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-pose-finish-engage-on", "start_epoch",
        "--jd1-pose-finish-start-epoch", "1",
        "--jd1-w-pose", "1.0",
        "--jd1-seg-hold-weight", "0.25",
        "--jd1-seg-hold-floor-source", "last_pre_pose_epoch_loss",
        "--jd1-seg-hold-space", "realized",
    ])
    with pytest.raises(SystemExit, match="requires --jd1-live-gate-telemetry on"):
        validate_jd1_pose_finish_args(args)
