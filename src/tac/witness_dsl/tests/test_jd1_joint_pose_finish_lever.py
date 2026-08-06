# SPDX-License-Identifier: MIT
"""JD1 TR1 joint pose-finish DSL wiring tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.train_tr1_partition_renderer_mlx import (
    build_argparser,
    validate_jd1_pose_finish_args,
)
from tac.witness_dsl.spec_tr1_renderer_20260728 import (
    TRAINER_RELPATH,
    TR1RendererProgramV1,
    default_t1_smoke_program,
    lever_jd1_joint_pose_finish,
    lever_jd1_lr_anneal,
    lever_jd1_muon_finisher,
    lever_jd1_plateau_tail_average_ema,
    lever_seg_grad_q3_project,
    trainer_declared_flags,
)

_REPO = Path(__file__).resolve().parents[4]


def test_jd1_lever_emits_only_real_tr1_flags_and_parses():
    lv = lever_jd1_joint_pose_finish(
        w_pose=1.0,
        start_epoch=0,
        engage_on="post_knee",
        seg_hold_weight=0.25,
        seg_hold_floor_source="checkpoint_tail_ep_loss",
        seg_hold_margin=0.1,
    )
    base = default_t1_smoke_program("plain", "/unused")
    prog = TR1RendererProgramV1(
        levers=base.levers + (lv,),
        num_pairs=24,
        out_dir="/unused",
        resume_from="parent.npz",
    )
    argv = prog.compile_trainer_argv()
    assert argv[0] == TRAINER_RELPATH
    for flag in lv.overrides:
        assert flag in argv
    ns = build_argparser().parse_args(argv[1:])
    validate_jd1_pose_finish_args(ns)
    assert ns.jd1_pose_finish_mode == "joint_loss"
    assert ns.jd1_seg_hold_floor_source == "checkpoint_tail_ep_loss"


def test_jd1_lever_flags_are_declared_by_the_live_trainer():
    lv = lever_jd1_joint_pose_finish(w_pose=1.0)
    missing = sorted(set(lv.overrides) - trainer_declared_flags())
    assert not missing
    assert (_REPO / TRAINER_RELPATH).is_file()


def test_jd1_tail_average_ema_lever_composes_with_joint_pose_finish():
    joint = lever_jd1_joint_pose_finish(w_pose=1.0, start_epoch=5, engage_on="start_epoch")
    tail = lever_jd1_plateau_tail_average_ema(anchor_epoch=1424)
    base = default_t1_smoke_program("plain", "/unused")
    prog = TR1RendererProgramV1(
        levers=base.levers + (joint, tail),
        num_pairs=24,
        out_dir="/unused",
        resume_from="parent.npz",
    )
    argv = prog.compile_trainer_argv()
    ns = build_argparser().parse_args(argv[1:])
    validate_jd1_pose_finish_args(ns)
    assert ns.jd1_pose_finish_mode == "joint_loss"
    assert ns.jd1_ema_stage_scope == "window"
    assert ns.jd1_ema_mode == "plateau_tail_average"
    assert ns.jd1_ema_tail_anchor_epoch == 1424


def test_jd1_tail_average_ema_lever_flags_are_declared_by_the_live_trainer():
    lv = lever_jd1_plateau_tail_average_ema(anchor_epoch=1424)
    missing = sorted(set(lv.overrides) - trainer_declared_flags())
    assert not missing


def test_jd1_tail_average_ema_factory_refuses_negative_anchor():
    with pytest.raises(ValueError, match="anchor_epoch"):
        lever_jd1_plateau_tail_average_ema(anchor_epoch=-1)


def test_jd1_lr_anneal_lever_composes_with_joint_pose_finish_and_defaults_derive():
    joint = lever_jd1_joint_pose_finish(w_pose=1.0, start_epoch=5, engage_on="start_epoch")
    lr = lever_jd1_lr_anneal()
    base = default_t1_smoke_program("plain", "/unused")
    prog = TR1RendererProgramV1(
        levers=base.levers + (joint, lr),
        num_pairs=24,
        out_dir="/unused",
        resume_from="parent.npz",
    )
    argv = prog.compile_trainer_argv()
    ns = build_argparser().parse_args(argv[1:])
    validate_jd1_pose_finish_args(ns)
    assert ns.jd1_pose_finish_mode == "joint_loss"
    assert ns.jd1_lr_anneal == "derived_tail"
    assert ns.jd1_lr_final_frac == 0.0


def test_jd1_lr_anneal_lever_flags_are_declared_by_the_live_trainer():
    lv = lever_jd1_lr_anneal()
    missing = sorted(set(lv.overrides) - trainer_declared_flags())
    assert not missing


def test_jd1_lr_anneal_factory_refuses_inert_or_invalid_shapes():
    with pytest.raises(ValueError, match="final_frac"):
        lever_jd1_lr_anneal(final_frac=0.0)
    with pytest.raises(ValueError, match="final_frac"):
        lever_jd1_lr_anneal(final_frac=1.0)


def test_jd1_muon_finisher_lever_composes_with_case_b_start_boundary():
    joint = lever_jd1_joint_pose_finish(w_pose=1.0, start_epoch=1646, engage_on="start_epoch")
    muon = lever_jd1_muon_finisher()
    base = default_t1_smoke_program("plain", "/unused")
    prog = TR1RendererProgramV1(
        levers=base.levers + (joint, muon),
        num_pairs=24,
        out_dir="/unused",
        resume_from="parent.npz",
    )
    argv = prog.compile_trainer_argv()
    ns = build_argparser().parse_args(argv[1:])
    validate_jd1_pose_finish_args(ns)
    assert ns.jd1_pose_finish_mode == "joint_loss"
    assert ns.jd1_pose_finish_engage_on == "start_epoch"
    assert ns.jd1_finisher == "muon"


def test_jd1_muon_finisher_lever_flags_are_declared_by_the_live_trainer():
    lv = lever_jd1_muon_finisher()
    missing = sorted(set(lv.overrides) - trainer_declared_flags())
    assert not missing


def test_jd1_muon_finisher_validation_refuses_mid_window_or_unresumed_shapes():
    base_args = [
        "--variant", "plain",
        "--out-dir", "/unused",
        "--jd1-pose-finish-mode", "joint_loss",
        "--jd1-w-pose", "1.0",
        "--jd1-finisher", "muon",
    ]
    ns = build_argparser().parse_args(base_args)
    with pytest.raises(SystemExit, match="requires --resume-from"):
        validate_jd1_pose_finish_args(ns)

    ns = build_argparser().parse_args([*base_args, "--resume-from", "parent.npz"])
    with pytest.raises(SystemExit, match="engage-on start_epoch"):
        validate_jd1_pose_finish_args(ns)


def test_jd1_factory_refuses_inert_or_invalid_shapes():
    with pytest.raises(ValueError, match="w_pose"):
        lever_jd1_joint_pose_finish(w_pose=0.0)
    with pytest.raises(ValueError, match="positive start_epoch"):
        lever_jd1_joint_pose_finish(w_pose=1.0, engage_on="start_epoch", start_epoch=0)
    with pytest.raises(ValueError, match="non-off floor source"):
        lever_jd1_joint_pose_finish(w_pose=1.0, seg_hold_weight=0.1)
    with pytest.raises(ValueError, match="explicit floor"):
        lever_jd1_joint_pose_finish(
            w_pose=1.0,
            seg_hold_weight=0.1,
            seg_hold_floor_source="explicit",
        )


def test_pg1_q3_lever_emits_real_tr1_flag_and_parses():
    lv = lever_seg_grad_q3_project()
    base = default_t1_smoke_program("plain", "/unused")
    prog = TR1RendererProgramV1(levers=base.levers + (lv,), num_pairs=24, out_dir="/unused")
    argv = prog.compile_trainer_argv()
    assert argv[0] == TRAINER_RELPATH
    assert "--seg-grad-q3-project" in argv
    ns = build_argparser().parse_args(argv[1:])
    assert ns.seg_grad_q3_project == "on"


def test_pg1_q3_lever_flags_are_declared_by_the_live_trainer():
    lv = lever_seg_grad_q3_project()
    missing = sorted(set(lv.overrides) - trainer_declared_flags())
    assert not missing
    assert (_REPO / TRAINER_RELPATH).is_file()


def test_pg1_q3_factory_refuses_inert_off_state():
    with pytest.raises(ValueError, match="omit the lever"):
        lever_seg_grad_q3_project("off")
