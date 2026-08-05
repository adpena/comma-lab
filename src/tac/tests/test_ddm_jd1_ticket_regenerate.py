"""Tests for the JD1/JD3/JD4 ticket regenerator factory.

These tests use synthetic ticket/checkpoint metadata only. They verify ticket
custody surfaces and refuse paths; no trainer, scorer, Metal, or launch is run.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[3]
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments import ddm_jd1_ticket_regenerate as R  # noqa: E402
from experiments.train_tr1_partition_renderer_mlx import derive_ema_decay  # noqa: E402


def _write_ckpt(path: Path, *, epoch: int, tail_epoch: int | None, stage_ema_u: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    jd1 = {
        "schema": "ddm_jd1_tr1_joint_pose_finish_runtime.v1",
        "engaged": True,
        "stage_ema_reanchored": True,
        "active_ema_decay": 0.9966666666666667,
        "active_ema_decay_provenance": (
            "JD3 stage-scoped window EMA: DERIVED ema_decay_run_geometry_v1 "
            f"decay_from_warmup_fraction phi=0.5 U={stage_ema_u} -> 0.996667"
            if stage_ema_u is not None else ""
        ),
    }
    meta = {
        "stage": "joint_pose_finish",
        "cfg": {"epochs": epoch, "num_pairs": 600, "batch_pairs": 4, "ema_decay": 0.999},
        "jd1_pose_finish": jd1,
        "telemetry_tail": ([] if tail_epoch is None else [{"event": "a1_gate", "epoch": tail_epoch}]),
    }
    payload = {
        "meta::epoch": np.array([epoch], dtype=np.int64),
        "meta::json": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    }
    np.savez(path, **payload)


def _base_ticket(old_resume: Path) -> dict:
    sealed_decay, _ = derive_ema_decay(10 * 75)
    return {
        "argv": [
            R.TRAINER_SCRIPT,
            "--num-pairs", "600",
            "--batch-pairs", "4",
            "--epochs", "1406",
            "--ema-decay", repr(sealed_decay),
            "--out-dir", "/ancestor/out",
            "--max-wall-minutes", "130",
            "--resume-from", str(old_resume),
            "--jd1-pose-finish-mode", "joint_loss",
            "--jd1-pose-finish-engage-on", "post_knee",
            "--jd1-pose-finish-start-epoch", "0",
            "--jd1-seg-hold-floor-source", "checkpoint_tail_ep_loss",
            "--jd1-seg-hold-weight", "0.25",
            "--jd1-w-pose", "1.0",
            "--jd1-ema-stage-scope", "window",
            "--jd1-live-gate-telemetry", "on",
            "--lane-guard",
        ],
        "child_out_dir": "/ancestor/out",
        "child_resume_from": str(old_resume),
        "levers": [
            {
                "name": "window",
                "overrides": {
                    "--epochs": "1076",
                    "--max-wall-minutes": "130.0",
                },
            },
            {
                "name": "ema",
                "overrides": {"--ema-decay": "0.9999436222692036"},
            },
            {
                "name": "jd1",
                "overrides": {
                    "--jd1-seg-hold-floor-source": "checkpoint_tail_ep_loss",
                    "--jd1-w-pose": "1.0",
                },
            },
            {
                "name": "bool",
                "overrides": {"--lane-guard": True},
            },
        ],
        "recursive_encode_pass_loop": {
            "continue_policy": {
                "next_resume_from_template": "/ancestor/out/checkpoints/stage_joint_pose_finish_final.npz",
            },
        },
        "ticket_hash": "old",
    }


def test_jd4_continuation_emission_repairs_all_debt_surfaces(tmp_path, monkeypatch):
    old_resume = tmp_path / "old" / "stage_seg_trunk_tau_final.npz"
    endpoint = tmp_path / "endpoint" / "stage_joint_pose_finish_final.npz"
    _write_ckpt(old_resume, epoch=10, tail_epoch=9, stage_ema_u=750)
    _write_ckpt(endpoint, epoch=1406, tail_epoch=1405, stage_ema_u=1200)
    base_ticket = tmp_path / "base_ticket.json"
    base_ticket.write_text(json.dumps(_base_ticket(old_resume)) + "\n")
    monkeypatch.setattr(R, "JD4_ROOT", tmp_path / "jd4")

    out_ticket = tmp_path / "jd4_ticket.json"
    regen = R.emit_jd4_continuation(Namespace(
        base_ticket=base_ticket,
        winner_ckpt=endpoint,
        out_ticket=out_ticket,
        window_epochs=120,
        epochs=1526,
    ))

    assert out_ticket.exists()
    ticket = json.loads(out_ticket.read_text())
    argv = ticket["argv"]
    values = R.argv_value_map(argv)
    assert argv[0] == R.VENV_PYTHON
    assert argv[1] == R.TRAINER_SCRIPT
    assert values["--resume-from"] == str(endpoint)
    assert values["--out-dir"] == str(tmp_path / "jd4" / "tr1_jd4_cont_ep1406")
    assert values["--epochs"] == "1526"
    assert values["--max-wall-minutes"] == "165"
    assert values["--jd1-force-ema-reanchor-on-resume"] is True
    assert ticket["regenerated_from"]["parent_stage_ema_u"] == 1200
    assert ticket["regenerated_from"]["new_window_u"] == 18000
    assert ticket["regenerated_from"]["force_ema_reanchor_on_resume"] is True
    assert ticket["recursive_encode_pass_loop"]["continue_policy"][
        "next_resume_from_template"
    ].startswith(str(tmp_path / "jd4" / "tr1_jd4_cont_ep1406"))
    for lever in ticket["levers"]:
        for flag, declared in (lever.get("overrides") or {}).items():
            assert declared == values[flag]
    assert regen["ticket_hash"] == ticket["ticket_hash"]


def test_child_out_dir_checkpoint_reuse_refuses(tmp_path):
    child = tmp_path / "existing_child"
    _write_ckpt(child / "checkpoints" / "stage_joint_pose_finish_final.npz",
                epoch=5, tail_epoch=4, stage_ema_u=600)
    with pytest.raises(SystemExit, match="already contains checkpoints"):
        R.refuse_child_out_dir_checkpoint_reuse(child)


def test_recursive_template_must_stay_under_child_or_declared_new_dir(tmp_path):
    child = tmp_path / "child"
    ticket = {
        "recursive_encode_pass_loop": {
            "continue_policy": {
                "next_resume_from_template": str(
                    tmp_path / "ancestor" / "checkpoints" / "stage_joint_pose_finish_final.npz"
                ),
            },
        },
    }
    with pytest.raises(SystemExit, match="next_resume_from_template"):
        R.validate_recursive_resume_template(ticket, child_out_dir=child)


def test_declared_lever_override_mismatch_refuses():
    ticket = {
        "argv": [R.VENV_PYTHON, R.TRAINER_SCRIPT, "--epochs", "1526"],
        "levers": [{"name": "window", "overrides": {"--epochs": "1076"}}],
    }
    with pytest.raises(SystemExit, match="declared-vs-argv mismatch"):
        R.validate_lever_overrides_match_argv(ticket)


def test_missing_declared_lever_override_refuses_during_rebuild():
    with pytest.raises(SystemExit, match="final argv lacks"):
        R.rebuild_lever_overrides_from_argv(
            [{"name": "missing", "overrides": {"--not-present": "1"}}],
            [R.VENV_PYTHON, R.TRAINER_SCRIPT, "--epochs", "1526"],
        )
