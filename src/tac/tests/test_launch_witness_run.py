# SPDX-License-Identifier: MIT
"""Tests for tools/launch_witness_run.py — the canonical ONE-COMMAND witness
launcher (operator 2026-06-30 "automatically in the future" + the build-the-
automated-value-generator guiding principle).

Covers: never-invent-a-flag validation, the script-based launch.sh (no word-split
fragility), perf-env verification parsing, and the --dry-run safe path. NO real
spawn, NO GPU — --dry-run + unit helpers only (the live n600 run is untouched)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(_REPO / rel))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


lw = _load("launch_witness_run_under_test", "tools/launch_witness_run.py")
rld = pytest.importorskip("render_levelset_dashboard")
from tac import witness_autoconfig as wac  # noqa: E402

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _cfg():
    return wac.derive_config(_GT, num_pairs=600, overfit=True, epochs=1000)


# ───────────────────────── never-invent-a-flag ─────────────────────────
def test_real_trainer_flags_nonempty_and_has_known():
    flags = lw.real_trainer_flags()
    assert "--out-dir" in flags and "--curriculum" in flags and "--muon-start-epoch" in flags
    assert len(flags) > 30


def test_validate_emitted_flags_all_pass():
    ok, results = lw.validate_emitted_flags(_cfg(), "out")
    assert ok is True
    assert all(passed for _, passed in results)
    assert len(results) > 40


def test_validate_detects_invented_flag(monkeypatch):
    # simulate the trainer argparse missing every flag -> all emitted are "invented"
    monkeypatch.setattr(lw, "real_trainer_flags", lambda: frozenset())
    ok, results = lw.validate_emitted_flags(_cfg(), "out")
    assert ok is False
    assert any(not passed for _, passed in results)


# ───────────────────────── launch.sh (no word-split fragility) ─────────────────────────
def test_build_launch_sh_structure():
    body = lw.build_launch_sh(_cfg(), "out", repo_root=Path("/repo"))
    assert body.startswith("#!/bin/bash\nset -euo pipefail\ncd /repo\n")
    assert "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in body  # perf-env prefix present
    assert "train_levelset_witness_realized_through_R_mlx.py" in body
    assert "--curriculum" in body and "--stage-checkpoints" in body  # resumable


def test_write_launch_sh_roundtrips_through_schedule_parser(tmp_path):
    launch = lw.write_launch_sh(_cfg(), tmp_path)
    assert launch.exists() and os.access(launch, os.X_OK)
    # the parser (dashboard side) must recover the curriculum schedule from it
    cfg = rld.parse_run_config(tmp_path)
    assert cfg["source"] == "launch.sh"
    sched = cfg["schedule"]
    assert sched["tau_start"] is not None and sched["l7_start"] is not None
    assert sched["muon_start"] is not None and sched["epochs"] == 1000


# ───────────────────────── perf-env verification ─────────────────────────
def test_verify_perf_env_active(tmp_path):
    log = tmp_path / "run.log"
    log.write_text('{"stage": "custom_grouped_backward", "active": true, "note": "x"}\n')
    status, line = lw.verify_perf_env(log, timeout_s=1.0)
    assert status == "active" and line is not None


def test_verify_perf_env_inactive(tmp_path):
    log = tmp_path / "run.log"
    log.write_text('{"stage": "custom_grouped_backward", "active": false}\n')
    status, _ = lw.verify_perf_env(log, timeout_s=1.0)
    assert status == "inactive"


def test_verify_perf_env_not_seen(tmp_path):
    log = tmp_path / "run.log"
    log.write_text('{"stage": "gt", "n_pairs": 600}\n')
    status, line = lw.verify_perf_env(log, timeout_s=0.5, poll_s=0.1)
    assert status == "not_seen" and line is None


# ───────────────────────── dashboard ensure (down path; hermetic) ─────────────────────────
def test_ensure_dashboard_down_returns_false(capsys):
    # nothing is listening on this port -> False + an actionable warning
    assert lw.ensure_dashboard(59997) is False
    err = capsys.readouterr().err
    assert "NOT serving" in err and "dashboard_reload.py" in err


# ───────────────────────── main --dry-run (safe; no spawn) ─────────────────────────
def test_main_dry_run_writes_launch_sh_no_spawn(tmp_path, capsys):
    out = tmp_path / "run1"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600", "--epochs", "1000",
                  "--out-dir", str(out), "--dry-run", "--no-dashboard"])
    assert rc == 0
    assert (out / "launch.sh").exists()
    assert not (out / "run.log").exists()  # DRY-RUN must NOT spawn
    out_txt = capsys.readouterr().out
    assert "DRY-RUN" in out_txt and "55/55" in out_txt or "flags exist" in out_txt


def test_main_refuses_invented_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lw, "real_trainer_flags", lambda: frozenset())
    out = tmp_path / "run2"
    rc = lw.main(["--gt-cache", _GT, "--num-pairs", "600",
                  "--out-dir", str(out), "--dry-run", "--no-dashboard"])
    assert rc == 2
    assert not (out / "launch.sh").exists()  # refused BEFORE writing
    err = capsys.readouterr().err
    assert "invented flag" in err
