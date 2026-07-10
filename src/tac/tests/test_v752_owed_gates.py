"""Tests for the v7.5.2 owed-gate build (task #384): the launcher dry-start helpers + the owed-14
observer-replay harness + the owed-4 speed-stack audit + the owed-15 isolation-arm builder. Pure-function
level (no trainer spawn, no GPU) — the heavy launcher/trainer integration is exercised by the live
dry-start run itself, not here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "tools"), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- dry-start helpers --
def test_parse_dry_start_run_metrics(tmp_path):
    import launch_witness_run as L
    rl = tmp_path / "run.log"
    rl.write_text("\n".join([
        '{"stage": "gt", "n_pairs": 600, "secs": 1.5}',
        '{"stage": "loss_terms", "ep": 0, "total": 1.0}',
        '{"stage": "loss_terms", "ep": 1, "total": 0.9}',
        '{"stage": "checkpoint", "kind": "intra_stage", "epoch": 1, "resume_latest": "levelset_resume_state.npz"}',
        'NOT JSON — should be skipped',
        '{"stage": "verdict", "epoch": 2, "d_seg": 0.03}',
    ]))
    m = L.parse_dry_start_run_metrics(rl)
    assert m["epochs_completed"] == 2
    assert m["gt_secs"] == 1.5
    assert m["checkpoint_written"] is True
    assert m["resume_model_source"] is False  # no resume row in pass-1 log
    assert m["resume_start_epoch"] is None


def test_parse_dry_start_run_metrics_resume(tmp_path):
    import launch_witness_run as L
    rl = tmp_path / "run.log"
    rl.write_text("\n".join([
        '{"stage": "resume_model_source", "resume_model_from": "x/levelset_resume_state.npz"}',
        '{"resume_start_epoch": 3}',
        '{"stage": "loss_terms", "ep": 4, "total": 0.8}',
        '{"stage": "checkpoint", "epoch": 4, "resume_latest": "levelset_resume_state.npz"}',
    ]))
    m = L.parse_dry_start_run_metrics(rl)
    assert m["resume_model_source"] is True
    assert m["resume_start_epoch"] == 3
    assert m["epochs_completed"] == 4


def test_parse_dry_start_run_metrics_missing_file(tmp_path):
    import launch_witness_run as L
    m = L.parse_dry_start_run_metrics(tmp_path / "nope.log")
    assert m["epochs_completed"] == -1 and m["gt_secs"] is None


def test_dry_start_sec_per_ep():
    import launch_witness_run as L
    gross, marginal = L.dry_start_sec_per_ep(wall_s=300.0, gt_secs=1.5, epochs_completed=3)
    assert gross == 100.0
    assert marginal == round((300.0 - 1.5) / 3.0, 2)
    assert L.dry_start_sec_per_ep(300.0, None, 0) == (None, None)  # no epochs -> no rate


def test_dry_start_boot_and_resume_ok():
    import launch_witness_run as L
    good1 = {"epochs_completed": 3, "checkpoint_written": True, "peak_rss_gib": 70.0}
    assert L.dry_start_boot_ok(good1) is True
    assert L.dry_start_boot_ok({**good1, "checkpoint_written": False}) is False
    assert L.dry_start_boot_ok({**good1, "epochs_completed": 0}) is False
    good2 = {"resume_model_source": True, "resume_start_epoch": 3, "epochs_completed": 4}
    assert L.dry_start_resume_ok(good2) is True
    assert L.dry_start_resume_ok({**good2, "resume_model_source": False}) is False
    assert L.dry_start_resume_ok({**good2, "epochs_completed": 2}) is False  # did not step past resume


def test_inject_extra_flag():
    import launch_witness_run as L
    assert L._inject_extra_flag([], "--ckpt-every", "1") == ["--ckpt-every", "1"]
    assert L._inject_extra_flag(["--ckpt-every", "25"], "--ckpt-every", "1") == ["--ckpt-every", "1"]
    assert L._inject_extra_flag(["--x", "y"], "--ckpt-every", "1") == ["--x", "y", "--ckpt-every", "1"]


# --------------------------------------------------------------------------- observer replay --
def test_replay_pose_gate_negative_control_rising():
    import witness_observer_replay as R
    # a rising / oscillating σ_min series (like the stopped run) must NOT fire the plateau detector
    eps = [float(4 * i) for i in range(1, 32)]
    smins = [0.084, 0.064, 0.056, 0.083, 0.095, 0.07, 0.09, 0.11, 0.08, 0.10] * 3 + [0.106]
    leg = R.replay_pose_gate(eps, smins)
    assert leg["fired"] is False
    assert leg["pass"] is True  # negative control passes iff not fired


def test_replay_pose_gate_fires_on_clean_plateau():
    import witness_observer_replay as R
    # a synthetic clean flat plateau SHOULD fire (positive control) -> so the negative-control PASS is
    # not a vacuous "never fires" detector
    eps = [float(i) for i in range(40)]
    smins = [0.5 - 0.4 * (2.718 ** (-i / 3.0)) for i in range(30)] + [0.5] * 10
    leg = R.replay_pose_gate(eps, smins)
    assert leg["fired"] is True
    assert leg["pass"] is False  # a firing detector is NOT a valid negative control on a plateau curve


def test_replay_parsers(tmp_path):
    import witness_observer_replay as R
    rl = tmp_path / "run.log"
    rl.write_text("\n".join([
        '{"stage": "jacobian_basin", "epoch": 4, "median_sigma_min": 0.08}',
        '{"stage": "jacobian_basin", "epoch": 8, "median_sigma_min": 0.09}',
        '{"stage": "verdict", "epoch": 0, "d_seg": 0.2, "d_pose": 7.0}',
        '{"stage": "verdict", "epoch": 25, "d_seg": 0.03, "d_pose": 19.0}',
        '{"stage": "verdict", "epoch": 50, "d_seg": 0.034, "d_pose": 22.0}',
    ]))
    rows = R.load_jsonl_rows(rl)
    eps, smins = R.sigma_min_series(rows)
    assert eps == [4.0, 8.0] and smins == [0.08, 0.09]
    assert len(R.verdict_rows(rows)) == 3


def test_replay_costate_shadow():
    import witness_observer_replay as R
    rows = [{"actuation": "NONE", "costates": [{"name": "lambda_d_seg"}, {"name": "lambda_bytes"}]}]
    leg = R.replay_costate_shadow(rows)
    assert leg["pass"] is True
    assert "lambda_d_seg" in leg["costate_names"]
    assert R.replay_costate_shadow([])["pass"] is False


def test_replay_run_on_stopped_run():
    """End-to-end on the real stopped run (READ-ONLY) — all legs must pass (the SYNTHESIS §A.4
    negative control + parse smokes). Skips if the fixture run dir is absent."""
    import witness_observer_replay as R
    run_dir = _REPO / R._DEFAULT_RUN
    if not (run_dir / "run.log").exists():
        import pytest
        pytest.skip("stopped-run fixture absent")
    rep = R.replay_run(run_dir)
    assert rep["all_pass"] is True
    pose = next(leg for leg in rep["legs"] if leg["leg"] == "pose_gate_negative_control")
    assert pose["fired"] is False  # the load-bearing negative control


# --------------------------------------------------------------------------- speed-stack audit --
_LAUNCH_SAMPLE = (
    "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 TAC_MLX_CUSTOM_PERSISTENCE_POOL=1 .venv/bin/python trainer.py "
    "--async-verdict --verdict-batch 32 --fused-r-kernel --safe-compile-regions hosc_activation "
    "--self-orient --render-aa ipe --num-pairs 600"
)


def test_audit_speed_levers_all_on():
    import witness_speed_stack_audit as S
    levers = {v["lever"]: v for v in S.audit_speed_levers(_LAUNCH_SAMPLE)}
    assert levers["fused_r_kernel"]["ok"] and levers["fused_r_kernel"]["status"] == "ON"
    assert levers["grouped_backward"]["ok"]
    assert levers["async_verdict"]["ok"]
    assert levers["safe_compile_regions"]["status"] == "ON"
    # micro-batch-pairs absent -> EXCLUDED_WITH_REASON and ok
    assert levers["micro_batch_pairs"]["status"] == "EXCLUDED_WITH_REASON"
    assert levers["micro_batch_pairs"]["ok"]


def test_audit_speed_levers_missing_and_nondefault():
    import witness_speed_stack_audit as S
    text = "python trainer.py --micro-batch-pairs 4 --safe-compile-regions none"
    levers = {v["lever"]: v for v in S.audit_speed_levers(text)}
    assert levers["fused_r_kernel"]["status"] == "MISSING" and not levers["fused_r_kernel"]["ok"]
    assert levers["grouped_backward"]["status"] == "MISSING"
    assert levers["safe_compile_regions"]["status"] == "BYTE_IDENTICAL_OFF"
    assert levers["micro_batch_pairs"]["status"] == "PRESENT_NONDEFAULT"
    assert not levers["micro_batch_pairs"]["ok"]  # a non-default micro-batch is NOT a neutral lever


def test_wall_clock_budget_scales_with_sec_per_ep():
    import witness_speed_stack_audit as S
    b = S.wall_clock_budget(42.0)
    assert b["sec_per_ep"] == 42.0
    # 3 main stages nominal 250 + pose 100 = 850 epochs at 42 s/ep = ~9.9 h + head-solve
    assert b["total_train_epochs"]["nominal"] == 3 * 250 + 100
    assert 9.5 < b["total_wall_h"]["nominal"] < 10.5
    assert b["total_wall_h"]["lo"] < b["total_wall_h"]["hi"]


def test_sec_per_ep_from_report(tmp_path):
    import witness_speed_stack_audit as S
    rp = tmp_path / "dry_start_report.json"
    rp.write_text(json.dumps({"sec_per_ep_marginal": 44.0, "sec_per_ep_gross": 60.0,
                              "config": "crucible_v7", "num_pairs": 600}))
    sec, prov = S.sec_per_ep_from_report(rp)
    assert sec == 44.0 and "marginal" in prov
    assert S.sec_per_ep_from_report(tmp_path / "nope.json")[0] is None


# --------------------------------------------------------------------------- isolation arms --
def test_isolation_flag_validator():
    import build_v752_isolation_arms as B
    real = frozenset({"--self-orient", "--render-aa", "--dseg-aware-taper"})
    assert B._validate_flags("--self-orient --render-aa none", real) == []
    assert B._validate_flags("--self-orient --totally-invented", real) == ["--totally-invented"]
