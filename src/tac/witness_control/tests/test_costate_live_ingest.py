# SPDX-License-Identifier: MIT
"""Tests for the live factorized-feature ingest tool (organ upgrade C).

NO-FAKE: parsing tests run against synthetic run.log content in tmp dirs built from the
REAL live row shapes (including the measured classify_line wart: full verdict rows carry
``"frozen_epoch": false`` and classify as ``confound_alarm`` — the parser must still bind
them); the organ-ledger non-clobber test appends through the REAL
``append_trajectory_record`` and reconstructs with the REAL ``load_organ_memory``.
The heavy per-verdict feature computation is exercised for real in its own modules'
tests + the live integration run; HERE the orchestration (idempotency, state, routing)
is under test with a patched compute."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import costate_live_ingest as cli  # noqa: E402

from tac.witness_control.continual_costate import (  # noqa: E402
    append_trajectory_record,
    load_organ_memory,
)

# a REAL-shaped full verdict row (trimmed from the live run.log; note frozen_epoch:false)
_VERDICT = ('{"stage": "verdict", "epoch": %d, "seg_form": "tau_softplus", "d_seg": %f, '
            '"d_pose": null, "blob_bytes": 83183, "ep_loss": 33.97, "verdict_device": "cpu", '
            '"d_seg_by_class": [0.006, 0.223, 0.001, 0.034, 0.0007], '
            '"flip_share_by_class": [0.39, 0.33, 0.12, 0.10, 0.04], '
            '"accepted_frac": 1.0, "weights_stepped": true, "frozen_epoch": false, "async": true}')
_ASYNC_DONE = '{"stage": "verdict_async_done", "epoch": %d, "secs": 2947.7}'
_WALLCLOCK = ('{"stage": "witness_component_wallclock", "epoch": 903, "errors": [], '
              '"epoch_total_s": 122.8}')


def _write_log(tmp_path: Path, epochs=(850, 875)) -> Path:
    run = tmp_path / "levelset_n600_witness_test"
    run.mkdir()
    lines = [_WALLCLOCK]
    for e in epochs:
        lines.append(_VERDICT % (e, 0.004))
        lines.append(_ASYNC_DONE % e)
    lines.append("not json at all")
    lines.append('{"stage": "verdict", "epoch": 999}')  # thin/malformed: no d_seg -> skip
    (run / "run.log").write_text("\n".join(lines) + "\n")
    return run


def test_read_verdict_rows_binds_full_rows_with_monitor_classifier_drift(tmp_path):
    """The parser owns verdict-row binding; monitor category churn is not authority."""
    from tac.witness_run_monitor import classify_line

    assert classify_line(_VERDICT % (850, 0.004)) in {
        "confound_alarm",
        "verdict",
    }
    run = _write_log(tmp_path)
    rows = cli.read_verdict_rows(run / "run.log")
    assert [r["epoch"] for r in rows] == [850, 875]
    assert all(isinstance(r["d_seg"], float) for r in rows)


def test_read_verdict_rows_skips_async_done_and_malformed(tmp_path):
    run = _write_log(tmp_path)
    rows = cli.read_verdict_rows(run / "run.log")
    assert all(r["stage"] == "verdict" for r in rows)
    assert 999 not in [r["epoch"] for r in rows]  # d_seg-less thin row skipped


def test_read_verdict_rows_missing_log_is_empty(tmp_path):
    assert cli.read_verdict_rows(tmp_path / "absent.log") == []


def test_ingested_epochs_helper():
    st = {"runA": [850, 875]}
    assert cli.ingested_epochs(st, "runA") == {850, 875}
    assert cli.ingested_epochs(st, "runB") == set()


def _patched_compute(monkeypatch, calls):
    def fake_compute(run_dir, verdict, *, gt_cache, n_pairs_sample=12, energy_split=True,
                     segnet_cpu=None):
        ep = int(verdict["epoch"])
        calls.append(ep)
        payload = {"run_ref": f"{run_dir.name}#factorized-ep{ep}", "generated_at": f"t{ep}",
                   "kind": "factorized_features_v1", "n_intervals": 0, "prototypes": [],
                   "score_claim": False}
        row = {"schema": "witness_factorized_snapshot.v1", "run_ref": run_dir.name,
               "verdict_epoch": ep, "d_seg_sample": 0.004, "n_flips": 10,
               "visible_blind_energy": {"residual_energy_visible_frac_mean": 0.8}}
        return payload, row
    monkeypatch.setattr(cli, "compute_factorized_features", fake_compute)
    monkeypatch.setattr(
        "tac.witness_control.factorized_features.load_frozen_segnet_cpu", lambda **k: object())


def test_ingest_idempotent_and_only_latest(tmp_path, monkeypatch):
    calls: list[int] = []
    _patched_compute(monkeypatch, calls)
    run = _write_log(tmp_path, epochs=(850, 875))
    ledger = tmp_path / "ledger.md"
    snap = tmp_path / "snap.jsonl"
    state: dict = {}
    done = cli.ingest_new_verdicts(run, gt_cache=tmp_path / "gt.npz", only_latest=True,
                                   ledger_path=ledger, snapshot_jsonl=snap,
                                   state_override=state)
    assert done == [875] and calls == [875]  # newest unseen only
    # ep875 is now seen -> never re-ingested; the next call drains the older unseen 850
    done2 = cli.ingest_new_verdicts(run, gt_cache=tmp_path / "gt.npz", only_latest=True,
                                    ledger_path=ledger, snapshot_jsonl=snap,
                                    state_override=state)
    assert done2 == [850] and calls == [875, 850]
    # all seen -> fully idempotent no-op
    done3 = cli.ingest_new_verdicts(run, gt_cache=tmp_path / "gt.npz", only_latest=True,
                                    ledger_path=ledger, snapshot_jsonl=snap,
                                    state_override=state)
    assert done3 == [] and calls == [875, 850]


def test_ingest_all_unseen_processes_every_epoch_in_order(tmp_path, monkeypatch):
    calls: list[int] = []
    _patched_compute(monkeypatch, calls)
    run = _write_log(tmp_path, epochs=(800, 825, 850))
    state: dict = {}
    done = cli.ingest_new_verdicts(run, gt_cache=tmp_path / "gt.npz", only_latest=False,
                                   ledger_path=tmp_path / "l.md",
                                   snapshot_jsonl=tmp_path / "s.jsonl",
                                   state_override=state)
    assert done == [800, 825, 850]
    assert sorted(state[run.name]) == [800, 825, 850]


def test_ingest_writes_snapshot_rows_and_ledger_blocks(tmp_path, monkeypatch):
    calls: list[int] = []
    _patched_compute(monkeypatch, calls)
    run = _write_log(tmp_path, epochs=(875,))
    ledger = tmp_path / "ledger.md"
    snap = tmp_path / "snap.jsonl"
    cli.ingest_new_verdicts(run, gt_cache=tmp_path / "gt.npz", ledger_path=ledger,
                            snapshot_jsonl=snap, state_override={})
    rows = [json.loads(x) for x in snap.read_text().splitlines() if x.strip()]
    assert rows[0]["verdict_epoch"] == 875
    text = ledger.read_text()
    assert "#factorized-ep875" in text and "FEED-426-organ" in text


def test_run_ref_suffix_never_clobbers_organ_tournament_records(tmp_path):
    """REAL ledger semantics: load_organ_memory dedups per run_ref — the factorized
    record's ``#factorized-ep`` suffix keeps the SAME run's organ (tournament) record
    alive.  This is the clobber class the suffix exists to prevent."""
    ledger = tmp_path / "organ.md"
    organ_payload = {
        "run_ref": "runX", "generated_at": "20260717T000000Z", "n_verdicts": 5,
        "n_intervals": 4, "arch_reports": {"A_ridge_solve": {
            "loo_mae": 0.1, "loo_heur": 0.2, "wf_mae": 0.1, "wf_heur": 0.2,
            "perclass_loo": 0.1, "passed": True, "passed_walkforward": True}},
        "winner_walkforward": "A_ridge_solve", "prototypes": [], "duty_queue_top": [],
        "score_claim": False,
    }
    append_trajectory_record(organ_payload, ledger_path=ledger)
    fact_payload = {"run_ref": "runX#factorized-ep875", "generated_at": "20260717T010000Z",
                    "kind": "factorized_features_v1", "n_intervals": 0, "prototypes": [],
                    "score_claim": False}
    append_trajectory_record(fact_payload, ledger_path=ledger)
    mem = load_organ_memory(ledger)
    refs = {r.get("run_ref") for r in mem.records}
    assert refs == {"runX", "runX#factorized-ep875"}  # both alive — no clobber
    # re-appending the SAME factorized epoch supersedes itself (idempotent backstop)
    append_trajectory_record(dict(fact_payload, generated_at="20260717T020000Z"),
                             ledger_path=ledger)
    mem2 = load_organ_memory(ledger)
    assert sum(1 for r in mem2.records if r.get("run_ref") == "runX#factorized-ep875") == 1


def test_state_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "STATE_PATH", tmp_path / "state.json")
    cli._save_state({"runA": [1, 2]})
    assert cli._load_state() == {"runA": [1, 2]}
    monkeypatch.setattr(cli, "STATE_PATH", tmp_path / "missing.json")
    assert cli._load_state() == {}


@pytest.mark.skipif(
    not (_REPO.name != "" and Path("/Users/adpena/Projects/pact/upstream/models/segnet.safetensors").is_file()
         and Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz").is_file()),
    reason="real artifacts unavailable (worktree/CI)")
def test_integration_real_snapshot_one_pair(monkeypatch, tmp_path):
    """Gated integration: one REAL pair through decode -> frozen SegNet -> margins.
    Skipped where the big artifacts are absent; on the workstation it proves the ingest
    compute path end-to-end on real bytes."""
    monkeypatch.setenv("TAC_UPSTREAM_DIR", "/Users/adpena/Projects/pact/upstream")
    from tac.witness_control.factorized_features import snapshot_witness_margins

    run = Path("/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z")
    ckpt = run / "levelset_witness_ema_BEST.npz"
    if not ckpt.is_file():
        pytest.skip("live checkpoint absent")
    snap = snapshot_witness_margins(
        ckpt, "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz", [0])
    assert snap.n_flips > 0 and snap.flip_margin.min() >= 0.0
    assert 0.0 < snap.d_seg_sample < 0.1
    assert np.all(snap.flip_wrong != snap.flip_gt)
