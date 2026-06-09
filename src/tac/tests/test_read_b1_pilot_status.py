# SPDX-License-Identifier: MIT
"""NO-FAKE tests for tools/read_b1_pilot_status.py (read-only B1 status reader)."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = REPO_ROOT / "tools" / "read_b1_pilot_status.py"


def _load_tool():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("read_b1_pilot_status", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["read_b1_pilot_status"] = mod
    spec.loader.exec_module(mod)
    return mod


RS = _load_tool()


def _write_telemetry(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_heartbeat(path: Path, *, age_seconds: float, train_exit: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age_seconds))
    body = f"{ts} pid=42 run=test\n"
    if train_exit:
        body += f"{ts} TRAIN_EXIT rc=0 run=test\n"
    path.write_text(body, encoding="utf-8")


def _write_checkpoint_meta(d: Path, *, epoch, role="periodic", loss=1.0, metric=None):
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": "long_training_canonical_checkpoint.v1",
        "global_epoch": epoch,
        "loss": loss,
        "checkpoint_role": role,
        "is_final": role == "final",
        "checkpoint_selection_metric_key": "total",
        "checkpoint_selection_metric_value": metric if metric is not None else loss,
        "checkpoint_selection_metric_mode": "min",
        "captured_at_utc": "2026-06-09T00:00:00Z",
    }
    (d / f"epoch{epoch:06d}.meta.json").write_text(json.dumps(meta))


def test_heartbeat_status_alive(tmp_path: Path):
    hb = tmp_path / "hb.log"
    _write_heartbeat(hb, age_seconds=20)
    st = RS.heartbeat_status(hb, stale_seconds=7 * 60)
    assert st["alive"] is True
    assert 10 <= st["age_seconds"] <= 60
    assert st["last_pid"] == "42"


def test_heartbeat_status_stale(tmp_path: Path):
    hb = tmp_path / "hb.log"
    _write_heartbeat(hb, age_seconds=10 * 60)
    st = RS.heartbeat_status(hb, stale_seconds=7 * 60)
    assert st["alive"] is False
    assert st["age_seconds"] >= 7 * 60


def test_heartbeat_status_train_exit_not_alive(tmp_path: Path):
    hb = tmp_path / "hb.log"
    _write_heartbeat(hb, age_seconds=20, train_exit=True)
    st = RS.heartbeat_status(hb)
    assert st["train_exited"] is True
    assert st["alive"] is False  # TRAIN_EXIT => not "alive" (finished)


def test_telemetry_status_extracts_epoch_loss_and_sec_per_epoch(tmp_path: Path):
    tele = tmp_path / "telemetry.jsonl"
    _write_telemetry(
        tele,
        [
            {"epoch": 0, "loss": 385.0, "stage_name": "s", "wall_clock_seconds": 0.0, "learning_rate": 1e-3},
            {"epoch": 100, "loss": 30.0, "stage_name": "s", "wall_clock_seconds": 2000.0, "learning_rate": 1e-3},
        ],
    )
    st = RS.telemetry_status(tele)
    assert st["latest_epoch"] == 100
    assert st["latest_loss"] == 30.0
    # 2000s over 100 epochs => 20 s/epoch.
    assert st["seconds_per_epoch_estimate"] == pytest.approx(20.0, abs=0.1)


def test_telemetry_status_empty(tmp_path: Path):
    st = RS.telemetry_status(tmp_path / "nope.jsonl")
    assert st["rows"] == 0


def test_checkpoint_status_latest_and_best(tmp_path: Path):
    d = tmp_path / "checkpoints"
    _write_checkpoint_meta(d, epoch=250, loss=2.0, metric=2.0)
    _write_checkpoint_meta(d, epoch=500, loss=1.0, metric=1.0)
    st = RS.checkpoint_status(d)
    assert st["count"] == 2
    assert st["epochs"] == [250, 500]
    assert st["latest"]["global_epoch"] == 500
    # best = min metric => ep500 (metric 1.0).
    assert st["best"]["global_epoch"] == 500


def test_checkpoint_status_empty(tmp_path: Path):
    st = RS.checkpoint_status(tmp_path / "checkpoints")
    assert st["count"] == 0 and st["latest"] is None


def test_manifest_status_reads_gate_fields(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "b1_launch_manifest_run.json").write_text(
        json.dumps(
            {
                "param_count": 228903,
                "parity_target": 228958,
                "sidecar_export_enabled": False,
                "pay_rent_gate_active": True,
                "stage8_muon_status": "WIRED_AND_VALIDATED",
                "measurement_axis": "[macOS-MLX research-signal]",
                "checkpoint_cadence_epochs": 250,
            }
        )
    )
    st = RS.manifest_status(run_dir)
    assert st["param_count"] == 228903
    assert st["sidecar_export_enabled"] is False
    assert st["pay_rent_gate_active"] is True
    assert st["stage8_muon_status"] == "WIRED_AND_VALIDATED"


def test_build_status_computes_eta_to_next_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_heartbeat(run_dir / "hb.log", age_seconds=20)
    _write_telemetry(
        run_dir / "telemetry.jsonl",
        [
            {"epoch": 0, "loss": 385.0, "wall_clock_seconds": 0.0},
            {"epoch": 165, "loss": 30.0, "wall_clock_seconds": 3300.0},  # 20 s/epoch
        ],
    )
    (run_dir / "b1_launch_manifest_run.json").write_text(
        json.dumps({"checkpoint_cadence_epochs": 250, "param_count": 228903})
    )
    st = RS.build_status(run_dir, heartbeat_path=run_dir / "hb.log")
    eta = st["eta"]
    assert eta["next_checkpoint_epoch"] == 250
    # (250 - 165) * 20 = 1700s.
    assert eta["eta_to_next_checkpoint_seconds"] == pytest.approx(1700.0, abs=5)


def test_build_status_reports_harvest_result_when_present(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_heartbeat(run_dir / "hb.log", age_seconds=20)
    _write_telemetry(run_dir / "telemetry.jsonl", [{"epoch": 250, "loss": 1.0, "wall_clock_seconds": 5000.0}])
    (run_dir / "hi_nerv_backend_only_ep250_exact_eval.json").write_text(
        json.dumps(
            {
                "first_exact_score_advisory": 0.20,
                "evidence_grade": "[macOS-CPU advisory]",
                "target_epoch": 250,
            }
        )
    )
    st = RS.build_status(run_dir, heartbeat_path=run_dir / "hb.log")
    res = st["harvest"]["harvest_result_files"]
    assert len(res) == 1
    assert res[0]["first_exact_score_advisory"] == 0.20


def test_render_human_includes_key_signals(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_heartbeat(run_dir / "hb.log", age_seconds=20)
    _write_telemetry(run_dir / "telemetry.jsonl", [{"epoch": 165, "loss": 30.0, "wall_clock_seconds": 3300.0}])
    (run_dir / "b1_launch_manifest_run.json").write_text(
        json.dumps({"param_count": 228903, "parity_target": 228958, "pay_rent_gate_active": True})
    )
    st = RS.build_status(run_dir, heartbeat_path=run_dir / "hb.log")
    text = RS.render_human(st)
    assert "B1 pilot status" in text
    assert "epoch=165" in text
    assert "228903" in text
    assert "pay_rent_gate=True" in text


def test_scaled_curriculum_status_reaches_muon_at_reduced_total():
    """At total=3000 the scaled curriculum must reach stage-8 Muon (~ep2494), NOT ep24650."""
    from tools.read_b1_pilot_status import scaled_curriculum_status

    s_early = scaled_curriculum_status(3000, 250)
    s_qat = scaled_curriculum_status(3000, 1050)
    s_muon = scaled_curriculum_status(3000, 2600)
    assert s_early["scaled_curriculum_available"] is True
    assert s_early["current_stage_index"] == 1 and s_early["muon_active_now"] is False
    assert s_qat["current_stage_index"] == 4  # QAT stage in the scaled schedule
    assert s_muon["current_stage_index"] == 8 and s_muon["muon_active_now"] is True
    assert s_early["muon_starts_epoch"] < 3000  # scaled, not canonical 24650


def test_research_total_from_launch_script_parses_reduced_total(tmp_path):
    """The reader must read the ACTUAL research total from the launch script, not the
    manifest's canonical 29650."""
    from tools.read_b1_pilot_status import _research_total_from_launch_script

    run = tmp_path
    (run / "launch_b1_pilot.sh").write_text(
        "#!/bin/bash\n.venv/bin/python x.py --full \\\n  --research-curriculum-total-epochs 3000 \\\n  --foo\n"
    )
    assert _research_total_from_launch_script(run) == 3000
    assert _research_total_from_launch_script(tmp_path / "nope") is None
