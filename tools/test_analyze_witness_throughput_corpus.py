# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("analyze_witness_throughput_corpus.py")


def _load():
    spec = importlib.util.spec_from_file_location("analyze_witness_throughput_corpus", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_async_only_log_measures_cadence_but_refuses_contention(tmp_path: Path) -> None:
    module = _load()
    log = tmp_path / "run.log"
    _write(log, [
        {"stage": "gt", "n_pairs": 600},
        {"stage": "verdict", "epoch": 25, "ts": "2026-07-13T00:00:00Z"},
        {"stage": "verdict_async_done", "epoch": 25, "secs": 40.0},
        {"stage": "verdict", "epoch": 50, "ts": "2026-07-13T00:02:00Z"},
    ])
    row = module.analyze_log(log)
    assert row["n_pairs"] == 600
    assert row["verdict_skip_count"] == 0
    assert row["async_identifiability"]["no_cadence_miss_measured"] is True
    assert row["contention_penalty_fraction"] is None
    assert row["completion_interval_s_per_epoch_median"] == pytest.approx(4.8)
    assert row["service_over_completion_interval_median"] == pytest.approx(1 / 3)


def test_component_rows_derive_incremental_vjp_without_summing(tmp_path: Path) -> None:
    module = _load()
    log = tmp_path / "component.jsonl"
    row = {
        "stage": "witness_component_wallclock",
        "complete": True,
        "measurement_scope": "probe",
        "teacher_forward_s": 0.6,
        "teacher_backward_s": 0.9,
        "witness_forward_s": 0.1,
        "witness_backward_s": 0.2,
        "realized_R_s": 0.01,
        "verdict_s": 0.0,
        "checkpoint_io_s": 0.0,
        "epoch_total_s": 8.0,
    }
    _write(log, [row])
    out = module.analyze_log(log)["component_timing"]
    assert out["teacher_incremental_vjp_median_s"] == pytest.approx(0.3)
    assert out["additivity_refused"] is True


def test_output_inside_input_run_is_refused(tmp_path: Path) -> None:
    module = _load()
    run = tmp_path / "run"
    run.mkdir()
    log = run / "run.log"
    _write(log, [])
    with pytest.raises(SystemExit):
        module.main([str(log), "--output", str(run / "analysis.json")])


def test_historical_microbatch_summary_reports_full_step_not_bench_claim(tmp_path: Path) -> None:
    module = _load()
    paths = []
    for batch, step, epoch in ((1, 10.0, 15.0), (2, 9.0, 14.0)):
        arm = tmp_path / f"B{batch}"
        arm.mkdir()
        log = arm / "run.log"
        _write(log, [
            {"stage": "provenance", "git_sha": str(batch) * 40},
            {"stage": "profile_timing", "epoch": 3,
             "t_epoch_s": epoch, "t_step_fwd_bwd_opt_ema_s": step, "t_verdict_s": 5.0},
            {"stage": "profile_timing", "epoch": 6,
             "t_epoch_s": epoch, "t_step_fwd_bwd_opt_ema_s": step, "t_verdict_s": 5.0},
        ])
        paths.append(log)
    summary = module.build_report(paths)["microbatch_comparison"]
    assert summary["current_v9_in_loop_transfer_established"] is False
    assert summary["arms"][1]["epoch_speedup_vs_B1"] == pytest.approx(15 / 14)
    assert summary["arms"][1]["step_speedup_vs_B1"] == pytest.approx(10 / 9)
