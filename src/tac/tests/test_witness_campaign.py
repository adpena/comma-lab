"""Tests for the witness A/B campaign engine (task #189): cyclic recursion,
harvest measure-loop, winner selection, theta* composition, dry-run planning."""
import json

import pytest

from tac.witness_dsl import (
    BASELINE,
    Arm,
    Cycle,
    Lever,
    Muon,
    PoseDecouple,
    TauFrozen,
    compose_theta_star,
    expand_cycles,
    harvest_arm,
    plan_campaign,
    select_winners,
)

_L7 = "experiments/results/levelset_l7_preserved_snapshots/levelset_resume_stageL7_ep1500.npz"


# --- cyclic recursion -------------------------------------------------------
def test_expand_cycles_chains_warm_starts():
    cycles = [
        Cycle("muon_prime", window=50, lever_fn=lambda ep: (Muon(start_epoch=ep, window=50),)),
        Cycle("l7_refine", window=50, lever_fn=lambda ep: ()),
        Cycle("muon2", window=50, lever_fn=lambda ep: (Muon(start_epoch=ep, window=50),)),
    ]
    progs = expand_cycles(BASELINE, cycles, start_resume_from=_L7, start_epoch=1500,
                          out_dir_prefix="experiments/results/cyc")
    assert len(progs) == 3
    # epochs accumulate by window
    assert [p.epochs for p in progs] == [1550, 1600, 1650]
    # warm-start chaining: cycle i resumes from cycle i-1's out_dir
    assert progs[0].resume_from == _L7
    assert progs[1].resume_from == progs[0].out_dir + "/levelset_resume_state.npz"
    assert progs[2].resume_from == progs[1].out_dir + "/levelset_resume_state.npz"
    # the muon cycles carry the correct per-cycle muon-start-epoch
    assert progs[0].flag_dict()["--muon-start-epoch"] == 1500
    assert progs[2].flag_dict()["--muon-start-epoch"] == 1600
    # every expanded program is valid
    for p in progs:
        assert p.validate() == []


def test_expand_cycles_empty_lever_cycle_is_plain_continuation():
    cycles = [Cycle("l7_more", window=100, lever_fn=lambda ep: ())]
    progs = expand_cycles(BASELINE, cycles, start_resume_from=_L7, start_epoch=1500,
                          out_dir_prefix="experiments/results/cyc")
    assert progs[0].epochs == 1600
    assert "--muon-start-epoch" not in progs[0].flag_dict()


# --- harvest ----------------------------------------------------------------
def _write_log(tmp_path, rows):
    p = tmp_path / "arm.log"
    with p.open("w") as f:
        for ep, dseg, dpose, bb in rows:
            f.write(json.dumps({"stage": "verdict", "epoch": ep, "seg_form": "l7_softplus",
                                "d_seg": dseg, "d_pose": dpose, "blob_bytes": bb}) + "\n")
    return p


def test_harvest_computes_best_final_and_delta(tmp_path):
    log = _write_log(tmp_path, [(900, 0.0040, 0.001, 73000),
                                (950, 0.0031, 0.001, 73010),   # best
                                (1000, 0.0033, 0.001, 73020)])
    res = harvest_arm(log, label="t", seg_form="l7_softplus", baseline_d_seg=0.0035)
    assert res.n_verdicts == 3
    assert res.d_seg_best == 0.0031 and res.d_seg_best_epoch == 950
    assert res.d_seg_final == 0.0033
    assert abs(res.delta_vs_baseline - (0.0031 - 0.0035)) < 1e-12
    assert res.improved is True


def test_harvest_no_baseline_no_delta(tmp_path):
    log = _write_log(tmp_path, [(900, 0.004, 0.001, 73000)])
    res = harvest_arm(log, seg_form="l7_softplus")
    assert res.delta_vs_baseline is None and res.improved is False


def test_harvest_missing_log_is_empty_result(tmp_path):
    res = harvest_arm(tmp_path / "nope.log", label="x")
    assert res.n_verdicts == 0 and res.d_seg_best is None


def test_harvest_seg_form_filter(tmp_path):
    p = tmp_path / "m.log"
    with p.open("w") as f:
        f.write(json.dumps({"stage": "verdict", "epoch": 1, "seg_form": "ce", "d_seg": 0.5}) + "\n")
        f.write(json.dumps({"stage": "verdict", "epoch": 2, "seg_form": "l7_softplus", "d_seg": 0.003}) + "\n")
    res = harvest_arm(p, seg_form="l7_softplus")
    assert res.n_verdicts == 1 and res.d_seg_best == 0.003


# --- winner selection + compose --------------------------------------------
def test_select_winners_drops_band_noise():
    a5 = PoseDecouple()
    a4 = Muon(start_epoch=1500, window=100)
    results = {
        "A5": _arm("A5", -3e-4),   # real improvement
        "A4": _arm("A4", -2e-6),   # band noise (above threshold)
        "Ax": _arm("Ax", +5e-4),   # worse
    }
    winners = select_winners(results, {"A5": a5, "A4": a4}, threshold=-1e-5)
    names = [w.name for w in winners]
    assert "A5_pose_decouple" in names
    assert "A4_muon" not in names  # band noise dropped
    assert len(winners) == 1


def test_compose_theta_star_binds_winners():
    theta = compose_theta_star(BASELINE, [PoseDecouple(), TauFrozen(0.05)],
                               resume_from=_L7, out_dir="experiments/results/theta_star",
                               epochs=2000)
    fd = theta.flag_dict()
    assert fd["--w-pose"] == 0.0          # A5
    assert fd["--softmax-temp-start"] == 0.05  # tau frozen
    assert theta.epochs == 2000
    assert theta.resume_from == _L7
    assert theta.validate() == []


# --- dry-run planning (containment) ----------------------------------------
def test_plan_campaign_emits_without_running():
    arms = [
        Arm("A5", BASELINE.with_lever(PoseDecouple(), resume_from=_L7,
            out_dir="experiments/results/ab_A5"), ".omx/tmp/A5.log"),
        Arm("A4", BASELINE.with_lever(Muon(1500, 100), resume_from=_L7,
            out_dir="experiments/results/ab_A4"), ".omx/tmp/A4.log"),
    ]
    plan = plan_campaign(arms)
    assert len(plan) == 2
    assert all(row["valid"] for row in plan)
    # the emitted command is a spawn_durable_daemon wrapper (compiled, NOT executed)
    for row in plan:
        assert "tools/spawn_durable_daemon.py" in row["daemon_argv"]
        assert "--min-free-gb" in row["daemon_argv"]


def test_plan_campaign_flags_invalid_arm():
    from dataclasses import replace
    bad = replace(BASELINE, base={**BASELINE.base, "--bogus-flag": 1})
    plan = plan_campaign([Arm("bad", bad, "/x.log")])
    assert plan[0]["valid"] is False
    assert any("INVENTED FLAG" in v for v in plan[0]["violations"])


def _arm(label, delta):
    from tac.witness_dsl import ArmResult
    return ArmResult(label=label, seg_form="l7_softplus", n_verdicts=10,
                     d_seg_best=0.003 + delta, d_seg_best_epoch=1100,
                     d_seg_final=0.003 + delta, d_pose_final=0.001,
                     blob_bytes_final=73000, delta_vs_baseline=delta, log_path="x")
