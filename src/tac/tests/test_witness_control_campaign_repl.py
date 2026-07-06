"""Tests for the campaign-REPL surface (task #324).

Governance-critical: propose_argv MUST never dispatch (go_required, not dispatched);
world-model + action-efficiency compute correctly; trace rows append without mutating.
"""
import json

from tac.witness_control.campaign_repl import (
    ProposedAction,
    action_efficiency,
    campaign_world_model,
    propose_argv,
    write_world_model_row,
)


def test_propose_argv_is_governed_never_fires():
    a = propose_argv(
        lever="islands-on",
        base_argv=["python", "train.py", "--epochs", "1000"],
        overrides={"--island-amplify": 0.1, "--per-group-grad-clip": None},
        rationale="birth movable via SDF-dilation homotopy",
    )
    assert isinstance(a, ProposedAction)
    assert a.go_required is True          # NON-NEGOTIABLE: always needs operator GO
    assert a.dispatched is False          # NON-NEGOTIABLE: this module cannot fire
    assert a.argv[:4] == ["python", "train.py", "--epochs", "1000"]
    assert "--island-amplify" in a.argv and "0.1" in a.argv
    assert "--per-group-grad-clip" in a.argv       # bare boolean flag, no value appended
    assert a.argv.count("--per-group-grad-clip") == 1


def test_propose_argv_normalizes_flag_prefix():
    a = propose_argv("l", ["x"], {"mod-dim": 32}, "r")
    assert "--mod-dim" in a.argv and "32" in a.argv


def test_action_efficiency_maximization_metric():
    # frontier fell 0.20 → 0.19 → 0.17 over two dispatches
    e = action_efficiency([0.20, 0.19, 0.17])
    assert e["dispatches"] == 2
    assert abs(e["total_improvement"] - 0.03) < 1e-9
    assert abs(e["mean_abs_dS_per_dispatch"] - 0.015) < 1e-9


def test_action_efficiency_degenerate():
    assert action_efficiency([])["mean_abs_dS_per_dispatch"] is None
    assert action_efficiency([0.19])["dispatches"] == 0


def test_world_model_from_synthetic_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    log = run / "run.log"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"stage": "verdict", "epoch": 25, "d_seg": 0.0093, "ts": "2026-07-06T10:00:00Z"},
        {"stage": "verdict", "epoch": 50, "d_seg": 0.0072, "ts": "2026-07-06T10:30:00Z"},
        {"stage": "loss_terms", "ep": 60, "terms": {"seg": 6.0}},
    ]))
    (run / "launch.sh").write_text("python train.py --mod-dim 32 --w-seg 100\n")
    wm = campaign_world_model(run)
    assert wm.best_d_seg is not None and abs(wm.best_d_seg - 0.0072) < 1e-9
    assert wm.last_delta_d_seg is not None and wm.last_delta_d_seg < 0   # descending
    assert wm.epoch_latest is not None
    # trace-row append is non-mutating + reloadable
    p = write_world_model_row(run, wm)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert rows[-1]["best_d_seg"] == wm.best_d_seg


def test_world_model_carries_ruled_out_verbatim():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        run = Path(d) / "r"
        run.mkdir()
        (run / "run.log").write_text(
            json.dumps({"stage": "verdict", "epoch": 25, "d_seg": 0.01,
                        "ts": "2026-07-06T10:00:00Z"}))
        wm = campaign_world_model(run, ruled_out=["paint-seed (starves d_seg #300)"])
        assert wm.ruled_out == ["paint-seed (starves d_seg #300)"]
