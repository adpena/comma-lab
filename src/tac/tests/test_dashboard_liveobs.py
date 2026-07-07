# SPDX-License-Identifier: MIT
"""Tests for the dashboard LIVE-RUN observability fixes (operator 2026-06-30
"full observability" + "automatically in the future"):

  1. ``render_levelset_dashboard._resolve_run_log`` surfaces the NEWEST-mtime
     WITNESS RUN (verdict-bearing OR still warming up in structured_init), so a
     freshly-launched 0-verdict run is visible IMMEDIATELY instead of at ~ep25.
  2. ``LiveState.refresh`` follows that warming-up run for meta + liveness (its
     own empty trajectory), and parses the run's CONFIG + CURRICULUM SCHEDULE from
     its OWN launch.sh / run.log — generalizable to any future run.
  3. ``_slim`` recomputes the DISPLAYED implied_S with the DEPLOY stored-pose
     sidecar (telemetry accuracy — the witness trains d_seg only), preserving the
     raw monitoring value.

Pure/unit; no network, no GPU, no uvicorn, no live-run interference (all tmp dirs).
"""
from __future__ import annotations

import json
import os
import sys
import time

import pytest

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, _TOOLS)

rld = pytest.importorskip("render_levelset_dashboard")
ds = pytest.importorskip("dashboard_server")


# ───────────────────────── helpers ─────────────────────────
def _gt_line(n=600):
    return json.dumps({"stage": "gt", "n_pairs": n, "secs": 3.8})


def _verdict_line(ep, dseg, blob=58435, dpose=163.3):
    return json.dumps({"stage": "verdict", "epoch": ep, "d_seg": dseg, "d_pose": dpose,
                       "blob_bytes": blob, "implied_S": 68.99, "ts": "2026-06-30T19:49:25Z"})


_LAUNCH_SH = """#!/bin/bash
set -euo pipefail
cd /repo
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \\
  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \\
  --out-dir experiments/results/levelset_x \\
  --num-pairs 600 \\
  --epochs 1000 \\
  --eval-every 25 \\
  --curriculum \\
  --tau-softplus-start-epoch 300 \\
  --l7-start-epoch 600 \\
  --muon-start-epoch 726 \\
  --w-seg 100 \\
  --w-pose 0 \\
  --score-domain-loss \\
  --mod-dim 26 \\
  --hidden-dim 96 \\
  --activation hosc \\
  --self-orient \\
  --chroma \\
  --ema-decay 0.997 \\
  --structured-init
"""


# ───────────────────────── _is_run_log / _resolve_run_log ─────────────────────────
def test_is_run_log_gt_only(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(_gt_line() + "\n")  # warming up, NO verdict yet
    assert rld._is_run_log(p) is True
    assert rld._has_verdict(p) is False  # confirms the warming distinction


def test_is_run_log_verdict(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(_verdict_line(0, 0.3) + "\n")
    assert rld._is_run_log(p) is True


def test_is_run_log_rejects_non_run_log(tmp_path):
    p = tmp_path / "dash_server.log"
    p.write_text('{"stage": "dashboard_server", "started": true}\n[noise]\n')
    assert rld._is_run_log(p) is False


def test_resolve_run_log_prefers_newest_warming_over_older_verdict(tmp_path):
    """The core bug: a newer warming-up run (gt, 0 verdicts) must be resolved over
    an older verdict-bearing run — so it is visible immediately, not at ~ep25."""
    old = tmp_path / "runA"; old.mkdir()
    (old / "run.log").write_text(_verdict_line(0, 0.3) + "\n" + _verdict_line(25, 0.2) + "\n")
    new = tmp_path / "runB"; new.mkdir()
    (new / "run.log").write_text(_gt_line() + "\n")  # warming up
    # make runB strictly newer
    os.utime(old / "run.log", (time.time() - 100, time.time() - 100))
    os.utime(new / "run.log", (time.time(), time.time()))
    glob = str(tmp_path / "*" / "run.log")
    run_latest = rld._resolve_run_log(None, glob)
    verdict_latest = rld._resolve_watched_log(None, glob)
    assert run_latest == new / "run.log"           # warming run wins for run-identity
    assert verdict_latest == old / "run.log"        # only the old run has verdicts


# ───────────────────────── launch.sh flag + schedule parsing ─────────────────────────
def test_parse_launch_sh_flags_value_bool_and_env_prefix():
    flags = rld._parse_launch_sh_flags(_LAUNCH_SH)
    assert flags["num-pairs"] == "600"
    assert flags["epochs"] == "1000"
    assert flags["curriculum"] is True            # boolean flag
    assert flags["score-domain-loss"] is True
    assert flags["w-pose"] == "0"
    assert "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" not in flags  # env prefix ignored


def test_parse_launch_sh_flags_equals_form():
    flags = rld._parse_launch_sh_flags("python x.py --foo=bar --baz 7\n")
    assert flags["foo"] == "bar"
    assert flags["baz"] == "7"


def test_schedule_from_flags_four_stages():
    flags = rld._parse_launch_sh_flags(_LAUNCH_SH)
    sched = rld._schedule_from_flags(flags)
    assert sched["epochs"] == 1000 and sched["eval_every"] == 25
    assert sched["tau_start"] == 300 and sched["l7_start"] == 600 and sched["muon_start"] == 726
    names = [(s["name"], s["start"], s["end"]) for s in sched["stages"]]
    assert names == [("CE", 0, 300), ("tau", 300, 600), ("l7", 600, 726), ("Muon", 726, 1000)]


def test_schedule_from_flags_demoted_l7_never_emits_no_l7_segment():
    """(L7-F1 fix) ``--l7-start-epoch 1001`` @ epochs=1000 (the demoted-"never" form the
    fresh_seeded config emits) must NOT produce the inverted [1001, 726) l7 segment nor a tau
    stage overlapping Muon: the chain skips l7 and Muon follows tau directly."""
    flags = {"epochs": "1000", "eval-every": "25", "tau-softplus-start-epoch": "300",
             "l7-start-epoch": "1001", "muon-start-epoch": "726"}
    sched = rld._schedule_from_flags(flags)
    names = [(s["name"], s["start"], s["end"]) for s in sched["stages"]]
    assert names == [("CE", 0, 300), ("tau", 300, 726), ("Muon", 726, 1000)]
    # no inverted segment anywhere (end can be None for an open tail; never end < start)
    assert all(s["end"] is None or s["start"] < s["end"] for s in sched["stages"])
    # the raw flag value is still surfaced for clients that display it
    assert sched["l7_start"] == 1001


def test_schedule_from_flags_l7_past_muon_skipped_too():
    """(L7-F1 fix, second form) l7_start > muon_start but < epochs is unreachable under Muon
    precedence — also skipped (no overlap, no inversion)."""
    flags = {"epochs": "1000", "tau-softplus-start-epoch": "300",
             "l7-start-epoch": "900", "muon-start-epoch": "726"}
    sched = rld._schedule_from_flags(flags)
    assert [(s["name"], s["start"], s["end"]) for s in sched["stages"]] == \
        [("CE", 0, 300), ("tau", 300, 726), ("Muon", 726, 1000)]


def test_parse_run_config_from_launch_sh(tmp_path):
    (tmp_path / "launch.sh").write_text(_LAUNCH_SH)
    (tmp_path / "run.log").write_text(_gt_line() + "\n")
    cfg = rld.parse_run_config(tmp_path)
    assert cfg["source"] == "launch.sh"
    assert "architecture" in cfg["groups"] and "loss" in cfg["groups"]
    assert ["w-pose", "0"] in cfg["groups"]["loss"]
    assert cfg["schedule"]["muon_start"] == 726


def test_parse_run_config_fallback_to_run_log(tmp_path):
    # no launch.sh -> fall back to the run.log stage emissions
    (tmp_path / "run.log").write_text(
        _gt_line() + "\n" +
        json.dumps({"stage": "front_end", "curvelet_cols": 40, "self_orient": True}) + "\n" +
        json.dumps({"stage": "structured_init", "steps": 600}) + "\n"
    )
    cfg = rld.parse_run_config(tmp_path)
    assert cfg["source"] == "run.log"
    stage_names = {s[0] for s in cfg.get("stages", [])}
    assert "gt" in stage_names and "front_end" in stage_names


def test_parse_run_config_none_dir():
    out = rld.parse_run_config(None)
    assert out["source"] == "none" and out["flags"] == {}


# ───────────────────────── implied_S: measured-only (no borrowed sidecar) ─────────────────────────
def test_ancestor_deploy_sidecar_surface_removed():
    """The deploy-override implied_S (ancestor RGB-vehicle sidecar d_pose 3.4e-5 — a
    BORROWED number, never witness-validated) was removed 2026-07-03: the displayed
    implied_S is the run's OWN measured composite, never a substituted constant.
    Structural guard against reintroducing the borrowed-constant class."""
    assert not hasattr(ds, "_implied_s_deploy")
    assert not hasattr(ds, "DEPLOY_SIDECAR_D_POSE")


def test_slim_keeps_measured_implied_s_and_aliases_monitoring():
    row = {"stage": "verdict", "epoch": 0, "d_seg": 0.285409, "d_pose": 163.3,
           "blob_bytes": 58435, "implied_S": 68.99, "ts": "t", "secret": "x"}
    out = ds._slim(row)
    assert set(out.keys()) == set(ds._TRAJ_KEYS)
    assert "secret" not in out
    # the run's OWN measured composite, passed through unchanged; monitoring aliases it
    assert out["implied_S"] == 68.99
    assert out["implied_S_monitoring"] == 68.99


# ───────────────────────── refresh(): warming-up + normal ─────────────────────────
def _state_for(tmp_path, glob):
    cfg = ds.Config(log_glob=glob, cadence_state=str(tmp_path / "cad.json"))
    return ds.LiveState(cfg)


def test_refresh_follows_warming_run_with_zero_verdicts(tmp_path):
    old = tmp_path / "runA"; old.mkdir()
    (old / "run.log").write_text(_verdict_line(0, 0.3) + "\n")
    new = tmp_path / "runB"; new.mkdir()
    (new / "launch.sh").write_text(_LAUNCH_SH)
    (new / "run.log").write_text(_gt_line() + "\n")  # warming up, 0 verdicts
    os.utime(old / "run.log", (time.time() - 100, time.time() - 100))
    os.utime(new / "run.log", (time.time(), time.time()))
    st = _state_for(tmp_path, str(tmp_path / "*" / "*.log"))
    new_points = st.refresh()
    m = st.snapshot()["meta"]
    assert m["warming_up"] is True
    assert m["run_dir"].endswith("runB")             # follows the live warming run
    assert new_points == [] and len(st.trajectory) == 0   # its own empty trajectory
    # config/schedule parsed from the warming run's launch.sh (full observability)
    assert m["config"]["source"] == "launch.sh"
    assert m["schedule"]["muon_start"] == 726
    assert m["tau"] == 300 and m["l7"] == 600 and m["muon_start"] == 726


def test_refresh_normal_run_with_verdicts_parses_schedule(tmp_path):
    rundir = tmp_path / "run1"; rundir.mkdir()
    (rundir / "launch.sh").write_text(_LAUNCH_SH)
    (rundir / "run.log").write_text(
        _gt_line() + "\n" + _verdict_line(0, 0.285409) + "\n" + _verdict_line(25, 0.2) + "\n"
    )
    st = _state_for(tmp_path, str(tmp_path / "*" / "*.log"))
    st.refresh()
    snap = st.snapshot()
    m = snap["meta"]
    assert m["warming_up"] is False
    assert m["run_dir"].endswith("run1")
    assert len(snap["trajectory"]) == 2
    assert m["muon_start"] == 726 and m["schedule"]["epochs"] == 1000
    # the ancestor deploy-sidecar surface is REMOVED (borrowed 3.4e-5, never
    # witness-validated): implied_S is the run's own measured composite.
    assert "deploy_sidecar_d_pose" not in m
    assert snap["trajectory"][0]["implied_S"] == 68.99
    assert snap["trajectory"][0]["implied_S_monitoring"] == 68.99


# ─────────────────── #205 run-info strip surfaced by the LIVE server ───────────────────
def test_meta_surfaces_run_info_strip_when_run_present(tmp_path):
    """The LIVE server must surface the SAME #205 run-info strip the standalone HTML does:
    refresh() composes rld._collect_run_info + pre-renders rld._run_info_html, and meta()
    ships it as ``run_info_html``. Proves the strip is wired into the WS/JSON page, not only
    the standalone renderer's _write_html output."""
    rundir = tmp_path / "run1"; rundir.mkdir()
    (rundir / "launch.sh").write_text(_LAUNCH_SH)
    (rundir / "run.log").write_text(
        _gt_line() + "\n" + _verdict_line(0, 0.285409) + "\n" + _verdict_line(25, 0.2) + "\n"
    )
    st = _state_for(tmp_path, str(tmp_path / "*" / "*.log"))
    st.refresh()
    m = st.snapshot()["meta"]
    # the pre-rendered strip HTML is present, non-empty, and carries the real card grid.
    html = m["run_info_html"]
    assert isinstance(html, str) and html
    assert 'class="grid"' in html and 'class="card"' in html
    assert "curriculum stage" in html            # stage-progress card
    assert "throughput" in html.lower()          # throughput+ETA card
    # the underlying run_info dict is the REAL composed telemetry (not a stub).
    ri = st.run_info
    for key in ("stage_prog", "best", "checkpoints", "resume", "throughput", "perf", "config"):
        assert key in ri
    # config provenance rides through the strip's data path too.
    assert ri["config"]["source"] == "launch.sh"


def test_run_info_strip_empty_when_no_run_resolved(tmp_path):
    """Back-compat: no run resolved -> run_info_html == '' so the strip (:empty) is hidden
    and the page is unchanged."""
    st = _state_for(tmp_path, str(tmp_path / "no_such" / "*.log"))
    st.refresh()
    m = st.snapshot()["meta"]
    assert m["run_info_html"] == ""
    assert st.run_info == {}


def test_page_template_wires_run_info_strip():
    """The served page must (a) carry the strip target div, (b) scope the strip CSS under
    .rinfo so it never restyles the chart .grid/.card/.fill, and (c) render META.run_info_html
    into it. Guards the client half of the wire-in."""
    page = ds._page_html(ds.Config())
    assert 'id="runinfostrip"' in page               # target div
    assert ".rinfo .grid{" in page and ".rinfo .card{" in page  # scoped CSS (no chart collision)
    assert "function renderRunInfo()" in page        # client renderer
    assert "META.run_info_html" in page              # consumes the server-shipped strip
    assert "renderRunInfo();" in page                # called from render()
