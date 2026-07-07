"""Tests for the canonical DSL schedule read-back (tac.witness_dsl.schedule_readback).

Covers (operator directive + amendment 2026-07-07):
  * the REAL live run dir (mod32cap): CE@0, tau@300, Muon@726, NO l7 (disabled at 1001)
    — the exact mislabel incident class the read-back extincts;
  * a synthetic launch.sh fixture (backslash continuations + env prefix + export/cd);
  * the fail-open fallback path (no launch.sh / no run dir);
  * event-gated stages (#315/#334): pending (trigger from the DSL Curriculum object,
    cap as hard ceiling) -> fired via a trainer log row AND via a fake per-stage
    checkpoint file — labels correct before and after the fired epoch;
  * stage_at labeling: ep 500 -> tau, ep 800 -> Muon, NEVER l7 on the live map.

Authority: observability only; the frontier pointer (0.19110) is unmoved by tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.witness_dsl.schedule_readback import (
    ScheduleReadback,
    event_trigger_description,
    read_schedule,
    resolve_run_dir_for_log,
    trainer_argv_from_launch_sh,
)

REPO = Path(__file__).resolve().parents[3]
LIVE_RUN = REPO / "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z"

_FIXTURE_LAUNCH = """#!/bin/bash
set -euo pipefail
cd /nowhere/pact
export TAC_GOVERNED_ADMISSION=1  # resume protection
SOME_ENV=1 \\
  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \\
  --out-dir /tmp/x \\
  --epochs 1000 \\
  --eval-every 25 \\
  --curriculum \\
  --tau-softplus-start-epoch 300 \\
  --l7-start-epoch 1001 \\
  --muon-start-epoch 726 \\
  --seed 0
"""


def _write_fixture(tmp_path: Path, extra_flags: str = "") -> Path:
    text = _FIXTURE_LAUNCH
    if extra_flags:
        text = text.replace("  --seed 0\n", f"  {extra_flags} \\\n  --seed 0\n")
    (tmp_path / "launch.sh").write_text(text)
    return tmp_path


# ── argv extraction (the line-continuation / env-prefix attack surface) ──────────
def test_trainer_argv_from_launch_sh_handles_continuations_and_env_prefix():
    argv = trainer_argv_from_launch_sh(_FIXTURE_LAUNCH)
    assert argv is not None
    assert argv[:2] == ["--out-dir", "/tmp/x"]
    assert "--curriculum" in argv
    assert "SOME_ENV=1" not in argv and ".venv/bin/python" not in argv


def test_trainer_argv_none_when_no_trainer_line():
    assert trainer_argv_from_launch_sh("#!/bin/bash\necho hello\n") is None


# ── the REAL live run (the incident's own run dir) ───────────────────────────────
@pytest.mark.skipif(not LIVE_RUN.is_dir(), reason="live run dir absent on this host")
def test_live_run_derived_map_has_no_l7():
    rb = read_schedule(LIVE_RUN)
    assert rb.ok and rb.source == "launch.sh"
    names = [s.name for s in rb.stages]
    assert names == ["CE", "tau", "Muon"]          # l7 (disabled at 1001) OMITTED
    starts = {s.name: s.start for s in rb.stages}
    assert starts == {"CE": 0, "tau": 300, "Muon": 726}
    assert rb.epochs == 1000 and rb.eval_every == 25
    sd = rb.as_schedule_dict()
    assert sd["tau_start"] == 300 and sd["l7_start"] is None and sd["muon_start"] == 726


@pytest.mark.skipif(not LIVE_RUN.is_dir(), reason="live run dir absent on this host")
def test_live_run_labels_ep500_tau_ep800_muon_never_l7():
    rb = read_schedule(LIVE_RUN)
    assert rb.stage_at(500) == "tau"               # the OLD --l7 600 bug said "l7" here
    assert rb.stage_at(600) == "tau"
    assert rb.stage_at(800) == "Muon"
    assert all(rb.stage_at(ep) != "l7" for ep in range(0, 1000, 25))


@pytest.mark.skipif(not LIVE_RUN.is_dir(), reason="live run dir absent on this host")
def test_live_run_actual_evidence_muon_from_run_dir_artifacts():
    # the stageMuonStart ckpt lives in the run dir -> Muon carries actual provenance
    rb = read_schedule(LIVE_RUN)
    muon = next(s for s in rb.stages if s.name == "Muon")
    assert muon.start == 726 and muon.source in ("checkpoint", "log")


# ── synthetic fixture: fixed-epoch schedule ──────────────────────────────────────
def test_fixture_fixed_schedule_l7_disabled_omitted(tmp_path):
    rb = read_schedule(_write_fixture(tmp_path))
    assert rb.ok
    assert [s.name for s in rb.stages] == ["CE", "tau", "Muon"]
    assert rb.stage_at(150) == "CE"
    assert rb.stage_at(500) == "tau"
    assert rb.stage_at(800) == "Muon"


def test_fixture_l7_enabled_renders_l7(tmp_path):
    _write_fixture(tmp_path)
    text = (tmp_path / "launch.sh").read_text().replace(
        "--l7-start-epoch 1001", "--l7-start-epoch 600")
    (tmp_path / "launch.sh").write_text(text)
    rb = read_schedule(tmp_path)
    assert [s.name for s in rb.stages] == ["CE", "tau", "l7", "Muon"]
    assert rb.stage_at(650) == "l7" and rb.stage_at(750) == "Muon"


# ── fail-open fallback ───────────────────────────────────────────────────────────
def test_fallback_no_launch_sh(tmp_path):
    rb = read_schedule(tmp_path)
    assert not rb.ok and "launch.sh" in rb.reason
    assert rb.stage_at(500) == "CE"                # degenerate map, never raises


def test_fallback_none_run_dir():
    rb = read_schedule(None)
    assert not rb.ok and rb.reason == "no run_dir"


def test_fallback_malformed_launch_sh(tmp_path):
    (tmp_path / "launch.sh").write_text("python train_x.py --epochs notanint\n")
    rb = read_schedule(tmp_path)
    assert isinstance(rb, ScheduleReadback) and not rb.ok


# ── run-dir resolution (the .omx/tmp tee-log gap that forced hand-fed constants) ─
def test_resolve_run_dir_from_launcher_echo_in_tee_log(tmp_path):
    run_dir = tmp_path / "runA"
    run_dir.mkdir()
    _write_fixture(run_dir)
    tee = tmp_path / "tmplogs" / "levelset_x.log"
    tee.parent.mkdir()
    tee.write_text(f"[durable-daemon] launching: bash {run_dir}/launch.sh\n"
                   '{"stage": "gt", "n_pairs": 600}\n')
    assert resolve_run_dir_for_log(tee) == run_dir
    rb = read_schedule(resolve_run_dir_for_log(tee), log_paths=[str(tee)])
    assert rb.ok and rb.stage_at(500) == "tau"


def test_resolve_run_dir_prefers_own_parent_when_it_has_launch_sh(tmp_path):
    _write_fixture(tmp_path)
    log = tmp_path / "run.log"
    log.write_text("{}\n")
    assert resolve_run_dir_for_log(log) == tmp_path


@pytest.mark.skipif(not LIVE_RUN.is_dir(), reason="live run dir absent on this host")
def test_resolve_run_dir_live_tee_log():
    tee = REPO / ".omx/tmp/levelset_mod32cap_20260706T115614Z.log"
    if not tee.is_file():
        pytest.skip("live tee log absent")
    assert resolve_run_dir_for_log(tee) == LIVE_RUN


# ── event-gated stages (#315 / #334 Curriculum(handoff='event')) ─────────────────
_EVT_FLAGS = "--curriculum-event-triggered --curriculum-nucleus-guard"


def test_event_gated_pending_uses_dsl_trigger_and_cap(tmp_path):
    rb = read_schedule(_write_fixture(tmp_path, _EVT_FLAGS))
    assert rb.ok and rb.event_triggered
    tau = next(s for s in rb.stages if s.name == "tau")
    assert tau.mode == "event" and tau.status == "pending"
    assert tau.start is None and tau.cap == 300 and tau.fired_epoch is None
    # trigger description is DERIVED from the DSL Curriculum object's emitted flags
    assert "--curriculum-event-triggered" in (tau.trigger or "")
    assert "--curriculum-nucleus-guard" in (tau.trigger or "")
    # labels: pre-cap CE (unfired); >= cap provably tau (cap is a hard ceiling)
    assert rb.stage_at(200) == "CE"
    assert rb.stage_at(350) == "tau"
    # Muon remains fixed (no Muon event-trigger built)
    muon = next(s for s in rb.stages if s.name == "Muon")
    assert muon.mode == "fixed" and muon.start == 726


def test_event_gated_fired_via_log_row(tmp_path):
    _write_fixture(tmp_path, _EVT_FLAGS)
    (tmp_path / "run.log").write_text(
        '{"stage": "curriculum_transition_fired", "from": "ce", "to": "tau_softplus",'
        ' "epoch": 213, "trigger": "loss_plateau", "nucleus_gated": true,'
        ' "nucleus_ready": true}\n')
    rb = read_schedule(tmp_path)
    tau = next(s for s in rb.stages if s.name == "tau")
    assert tau.status == "fired" and tau.fired_epoch == 213 and tau.start == 213
    assert tau.source == "log"
    assert rb.stage_at(212) == "CE" and rb.stage_at(213) == "tau"
    assert rb.stage_at(500) == "tau" and rb.stage_at(800) == "Muon"


def test_event_gated_fired_via_fake_stage_checkpoint_file(tmp_path):
    # amendment fixture: ONE fired event-transition evidenced by a stage ckpt file
    # (event mode saves the NEW stage's tag AT the fired epoch) + Muon still pending
    # in evidence (planned-fixed). Labels correct before and after the fired epoch.
    _write_fixture(tmp_path, _EVT_FLAGS)
    (tmp_path / "levelset_ckpt_stageTau_ep213.npz").write_bytes(b"fake")
    rb = read_schedule(tmp_path)
    tau = next(s for s in rb.stages if s.name == "tau")
    assert tau.status == "fired" and tau.fired_epoch == 213 and tau.source == "checkpoint"
    assert rb.stage_at(100) == "CE" and rb.stage_at(300) == "tau"


def test_fixed_mode_completed_stage_ckpt_corroborates_next_boundary(tmp_path):
    # fixed mode: stageCE_ep299 = CE completed at 299 -> tau ACTUALLY started at 300
    _write_fixture(tmp_path)
    (tmp_path / "levelset_ckpt_stageCE_ep299.npz").write_bytes(b"fake")
    rb = read_schedule(tmp_path)
    tau = next(s for s in rb.stages if s.name == "tau")
    assert tau.start == 300 and tau.source == "checkpoint"


def test_event_trigger_description_reads_dsl_flags():
    d = event_trigger_description(True)
    assert "--curriculum-event-triggered" in d and "--curriculum-nucleus-guard" in d
    d2 = event_trigger_description(False)
    assert "--curriculum-nucleus-guard" not in d2


# ── dashboard label semantics against the derived map (dtm-compatible dict) ─────
def test_as_schedule_dict_feeds_stage_at_epoch_consistently(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dtm", REPO / "tools/dashboard_trajectory_model.py")
    dtm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dtm)
    rb = read_schedule(_write_fixture(tmp_path))
    sd = {**rb.as_schedule_dict()}
    assert dtm.stage_at_epoch(500, sd) == "tau"    # never "l7": l7_start is None
    assert dtm.stage_at_epoch(800, sd) == "Muon"
    assert all(dtm.stage_at_epoch(ep, sd) != "l7" for ep in range(0, 1000, 25))
