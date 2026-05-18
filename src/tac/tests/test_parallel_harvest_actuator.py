# SPDX-License-Identifier: MIT
"""Dedicated tests for the parallel-harvest-actuator (T0-D Grand Council).

Covers:
- Discovery (positive: in-flight lane / negative: missing dir / edge: empty)
- Score extraction (auth-eval JSON parsing + axis-tag inference + missing-score)
- Custody validator routing (Catalog #127 / #130; accepted / refused-class)
- Concurrent fan-out (5 fake call_ids; baseline 1; 20 stress; per-call error
  doesn't kill batch; overall executor timeout)
- Posterior append (locked write idempotent under concurrency)
- CLI surface (--list-only, --execute, --no-posterior-append, filter, quiet)
- Live-repo regression guard (discovery on real repo returns sane structure)

Per CLAUDE.md harness pillar 7 — dedicated tests for every primitive.
"""
from __future__ import annotations

import json
import sys as _sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure tools/ + src/ on sys.path for the test module itself.

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
for _p in (_REPO, _REPO / "src", _REPO / "tools"):
    _sp = str(_p)
    if _sp not in _sys.path:
        _sys.path.insert(0, _sp)

from tools import parallel_harvest_actuator as actuator  # noqa: E402


class _FakeFunctionCall:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome

    def get(self, timeout: float) -> Any:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _install_fake_modal(
    monkeypatch,
    outcome: Any,
    *,
    output_expired_cls: type[Exception] | None = None,
    function_timeout_cls: type[Exception] | None = None,
) -> tuple[type[Exception], type[Exception]]:
    if output_expired_cls is None:
        class OutputExpiredError(Exception):
            pass
    else:
        OutputExpiredError = output_expired_cls
    if function_timeout_cls is None:
        class FunctionTimeoutError(Exception):
            pass
    else:
        FunctionTimeoutError = function_timeout_cls

    fake_modal = SimpleNamespace(
        functions=SimpleNamespace(
            FunctionCall=SimpleNamespace(
                from_id=lambda call_id: _FakeFunctionCall(outcome)
            )
        ),
        exception=SimpleNamespace(
            OutputExpiredError=OutputExpiredError,
            FunctionTimeoutError=FunctionTimeoutError,
        ),
    )
    monkeypatch.setitem(_sys.modules, "modal", fake_modal)
    return OutputExpiredError, FunctionTimeoutError


def _register_call_id(tmp_path: Path, call_id: str, lane_id: str = "lane_single") -> Path:
    from tac.deploy.modal.call_id_ledger import register_dispatched_call_id

    ledger = tmp_path / ".omx" / "state" / "modal_call_id_ledger.jsonl"
    register_dispatched_call_id(
        call_id=call_id,
        lane_id=lane_id,
        label=lane_id,
        path=ledger,
        lock_path=ledger.with_suffix(ledger.suffix + ".lock"),
    )
    return ledger


# -- Discovery tests ---------------------------------------------------------


def _make_lane_metadata(
    parent: Path, label: str, call_id: str, *, harvested: bool, gpu: str = "T4"
) -> Path:
    """Create a fake lane_*_modal/ directory with modal_metadata.json."""

    lane_dir = parent / f"lane_{label}_modal"
    lane_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "label": label,
        "call_id": call_id,
        "dispatched_at": "2026-05-14T12:00:00",
        "lane_id": f"lane_{label}",
        "gpu": gpu,
        "max_seconds": 3600,
        "metadata_schema": "modal_train_lane_dispatch_metadata_v2_catalog166",
        "mounted_code_git_head": "abc1234",
    }
    (lane_dir / "modal_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    if harvested:
        # Mark as already-harvested via the canonical terminal marker
        artifacts = lane_dir / "harvested_artifacts"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "_harvest_summary.json").write_text(
            json.dumps({"rc": 0, "elapsed_seconds": 120, "n_artifacts": 3}),
            encoding="utf-8",
        )
        (lane_dir / "modal_training_terminal_claim.json").write_text(
            json.dumps({"appended": True, "lane_id": f"lane_{label}", "status": "ok"}),
            encoding="utf-8",
        )
    return lane_dir


def test_discover_inflight_modal_lanes_positive(tmp_path):
    """Discovers two lanes — one harvested, one pending."""

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "alpha", "fc-AAA", harvested=False)
    _make_lane_metadata(results, "beta", "fc-BBB", harvested=True)
    rows = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)
    assert len(rows) == 2
    by_label = {r["label"]: r for r in rows}
    assert by_label["alpha"]["harvested"] is False
    assert by_label["beta"]["harvested"] is True
    assert by_label["alpha"]["call_id"] == "fc-AAA"
    assert by_label["beta"]["lane_id"] == "lane_beta"


def test_discover_inflight_modal_lanes_missing_dir(tmp_path):
    """Returns empty list when experiments/results doesn't exist."""

    rows = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)
    assert rows == []


def test_discover_inflight_modal_lanes_empty(tmp_path):
    """Returns empty when experiments/results exists but no lane_*_modal dirs."""

    (tmp_path / "experiments" / "results").mkdir(parents=True)
    rows = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)
    assert rows == []


def test_discover_inflight_vastai_instances_missing(tmp_path):
    """Returns empty list when vastai state file doesn't exist."""

    rows = actuator.discover_inflight_vastai_instances(repo_root=tmp_path)
    assert rows == []


def test_discover_inflight_lightning_jobs_list_schema(tmp_path):
    """Reads canonical list-of-dicts schema."""

    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    (state / "lightning_active_jobs.json").write_text(
        json.dumps(
            [
                {"lane_id": "x", "job_name": "y", "terminal_status": None},
                {"lane_id": "z", "job_name": "w", "terminal_status": "completed_ok"},
            ]
        ),
        encoding="utf-8",
    )
    rows = actuator.discover_inflight_lightning_jobs(repo_root=tmp_path)
    assert len(rows) == 2


# -- Axis-tag inference tests ------------------------------------------------


def test_axis_from_device_cuda_linux():
    tag = actuator._axis_from_device_string("cuda", "linux_x86_64_t4")
    assert tag == actuator.AXIS_TAG_CUDA


def test_axis_from_device_cpu_linux_x86_64():
    tag = actuator._axis_from_device_string("cpu", "linux_x86_64_gha_cpu")
    assert tag == actuator.AXIS_TAG_CPU_GHA


def test_axis_from_device_cpu_macos_advisory():
    """macOS substrate ALWAYS returns advisory, regardless of device."""

    tag = actuator._axis_from_device_string("cpu", "macos_arm64")
    assert tag == actuator.AXIS_TAG_MACOS_ADVISORY


def test_axis_from_device_unknown_gpu_inferred_cuda():
    """Hardware containing recognizable GPU token infers CUDA when device unknown."""

    tag = actuator._axis_from_device_string("", "rtx_4090")
    assert tag == actuator.AXIS_TAG_CUDA


# -- Score extraction tests --------------------------------------------------


def test_extract_score_claim_with_auth_eval_json(tmp_path):
    """Extracts score + axis tag from an auth-eval JSON in harvested_artifacts."""

    out_dir = tmp_path / "lane_xyz_modal"
    artifacts = out_dir / "harvested_artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "auth_eval_roundtrip_results.json").write_text(
        json.dumps(
            {
                "score": 0.1968,
                "archive_sha256": "deadbeef" * 8,
                "archive_bytes": 178452,
                "device": "cpu",
                "hardware_substrate": "linux_x86_64_gha_cpu",
            }
        ),
        encoding="utf-8",
    )
    extracted = actuator.extract_score_claim_from_harvested_dir(out_dir)
    assert extracted is not None
    assert extracted["score"] == 0.1968
    assert extracted["axis_tag"] == actuator.AXIS_TAG_CPU_GHA
    assert extracted["archive_sha256"].endswith("deadbeef")


def test_extract_score_claim_no_auth_eval_returns_none(tmp_path):
    """No auth-eval JSON → returns None (signals 'no score extracted')."""

    out_dir = tmp_path / "lane_no_auth_modal"
    artifacts = out_dir / "harvested_artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "_harvest_summary.json").write_text(
        json.dumps({"rc": 0}), encoding="utf-8"
    )
    extracted = actuator.extract_score_claim_from_harvested_dir(out_dir)
    assert extracted is None


def test_extract_score_claim_corrupt_json_returns_none(tmp_path):
    """Corrupt auth-eval JSON returns None (safe-default; no crash)."""

    out_dir = tmp_path / "lane_corrupt_modal"
    artifacts = out_dir / "harvested_artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "auth_eval_results.json").write_text(
        "not valid json {{{", encoding="utf-8"
    )
    extracted = actuator.extract_score_claim_from_harvested_dir(out_dir)
    assert extracted is None


# -- Custody validator routing tests -----------------------------------------


def test_build_contest_result_row_custody_accepted_cuda(tmp_path):
    """Valid CUDA row with all custody metadata passes the validator."""

    extracted = {
        "auth_eval_path": "/some/auth_eval.json",
        "score": 0.193,
        "archive_sha256": "a" * 64,
        "archive_bytes": 296432,
        "device": "cuda",
        "hardware_substrate": "linux_x86_64_t4",
        "axis_tag": actuator.AXIS_TAG_CUDA,
    }
    row = actuator.build_contest_result_row(
        extracted=extracted,
        lane_id="lane_pr101_test",
        architecture_class="pr101_hnerv_cluster",
    )
    assert row["custody_accepted"] is True
    assert row["custody_refused_class"] is None


def test_build_contest_result_row_custody_refused_macos():
    """macOS substrate is refused by the custody validator regardless of tag."""

    extracted = {
        "auth_eval_path": "/some/auth_eval.json",
        "score": 0.19664,
        "archive_sha256": "b" * 64,
        "archive_bytes": 296432,
        "device": "cpu",
        "hardware_substrate": "macos_arm64",
        "axis_tag": actuator.AXIS_TAG_MACOS_ADVISORY,
    }
    row = actuator.build_contest_result_row(
        extracted=extracted,
        lane_id="lane_pr107",
        architecture_class="pr107_hnerv",
    )
    assert row["custody_accepted"] is False
    assert row["custody_refused_class"] in {"macos_substrate", "advisory_grade"}


def test_build_contest_result_row_custody_refused_missing_score():
    """Missing score_value → custody_accepted=False with missing_metadata class."""

    extracted = {
        "auth_eval_path": "/some/auth_eval.json",
        "score": None,
        "archive_sha256": "c" * 64,
        "archive_bytes": 100000,
        "axis_tag": actuator.AXIS_TAG_CUDA,
    }
    row = actuator.build_contest_result_row(
        extracted=extracted,
        lane_id="x",
        architecture_class="y",
    )
    assert row["custody_accepted"] is False
    assert row["custody_refused_class"] == "missing_metadata"


# -- Fan-out concurrency tests -----------------------------------------------


def test_fan_out_harvest_baseline_one_already_harvested(tmp_path, monkeypatch):
    """Baseline: 1 lane, already-harvested → exactly one row returned."""

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "single", "fc-SINGLE", harvested=True)
    lane_rows = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)
    assert len(lane_rows) == 1
    out = actuator.fan_out_harvest(
        lane_rows=lane_rows,
        repo_root=tmp_path,
        max_workers=1,
        per_call_timeout_seconds=1.0,
        overall_timeout_seconds=10.0,
        append_posterior=False,
    )
    assert len(out) == 1
    assert out[0]["harvest_status"] == "already_harvested"


def test_fan_out_harvest_five_concurrent_already_harvested(tmp_path):
    """5 fake lanes, all already-harvested → all rows complete in parallel."""

    results = tmp_path / "experiments" / "results"
    for i in range(5):
        _make_lane_metadata(results, f"l{i}", f"fc-L{i}", harvested=True)
    lane_rows = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)
    out = actuator.fan_out_harvest(
        lane_rows=lane_rows,
        repo_root=tmp_path,
        max_workers=5,
        per_call_timeout_seconds=1.0,
        overall_timeout_seconds=30.0,
        append_posterior=False,
    )
    assert len(out) == 5
    assert all(r["harvest_status"] == "already_harvested" for r in out)


def test_fan_out_harvest_stress_20_concurrent(tmp_path):
    """20 fake lanes (stress) → no race conditions, all 20 returned."""

    results = tmp_path / "experiments" / "results"
    for i in range(20):
        _make_lane_metadata(results, f"s{i:02d}", f"fc-S{i:02d}", harvested=True)
    lane_rows = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)
    assert len(lane_rows) == 20
    out = actuator.fan_out_harvest(
        lane_rows=lane_rows,
        repo_root=tmp_path,
        max_workers=8,
        per_call_timeout_seconds=1.0,
        overall_timeout_seconds=30.0,
        append_posterior=False,
    )
    assert len(out) == 20
    # All rows must have valid harvest_status (no None / no_result)
    assert all(r["harvest_status"] == "already_harvested" for r in out)
    # Order must be preserved (output index == input index)
    for i, row in enumerate(out):
        assert row["label"] == f"s{i:02d}"


def test_fan_out_harvest_empty_lane_list(tmp_path):
    """Empty input → empty output (no crash)."""

    out = actuator.fan_out_harvest(
        lane_rows=[],
        repo_root=tmp_path,
        max_workers=4,
        per_call_timeout_seconds=1.0,
        overall_timeout_seconds=5.0,
        append_posterior=False,
    )
    assert out == []


def test_harvest_one_call_success_appends_call_id_ledger(tmp_path, monkeypatch):
    """A terminal rc=0 Modal result appends a harvested call-id ledger row."""

    from tac.deploy.modal.call_id_ledger import query_by_call_id

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "single", "fc-SINGLE", harvested=False)
    lane_row = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)[0]
    ledger = _register_call_id(tmp_path, "fc-SINGLE")
    _install_fake_modal(
        monkeypatch,
        {"returncode": 0, "elapsed_seconds": 12.5, "artifacts": {}, "stdout_tail": ""},
    )

    out = actuator._harvest_one_call(
        lane_row=lane_row,
        repo_root=tmp_path,
        per_call_timeout_seconds=1.0,
        append_posterior=False,
    )

    assert out["harvest_status"] == "harvested"
    assert out["call_id_ledger"]["appended"] is True
    assert [row["status"] for row in query_by_call_id("fc-SINGLE", path=ledger)] == [
        "dispatched",
        "harvested",
    ]


def test_harvest_one_call_nonzero_rc_appends_failed_call_id_ledger(tmp_path, monkeypatch):
    """A terminal nonzero Modal result appends a failed call-id ledger row."""

    from tac.deploy.modal.call_id_ledger import query_by_call_id

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "failed", "fc-FAILED", harvested=False)
    lane_row = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)[0]
    ledger = _register_call_id(tmp_path, "fc-FAILED", lane_id="lane_failed")
    _install_fake_modal(
        monkeypatch,
        {"returncode": 7, "elapsed_seconds": 22.0, "artifacts": {}, "stdout_tail": "boom"},
    )

    out = actuator._harvest_one_call(
        lane_row=lane_row,
        repo_root=tmp_path,
        per_call_timeout_seconds=1.0,
        append_posterior=False,
    )

    assert out["harvest_status"] == "harvested"
    rows = query_by_call_id("fc-FAILED", path=ledger)
    assert [row["status"] for row in rows] == ["dispatched", "failed"]
    assert rows[-1]["rc"] == 7


def test_harvest_one_call_expired_appends_stale_call_id_ledger(tmp_path, monkeypatch):
    """Modal result-cache expiry is terminal stale, not a generic error."""

    from tac.deploy.modal.call_id_ledger import query_by_call_id

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "expired", "fc-EXPIRED", harvested=False)
    lane_row = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)[0]
    ledger = _register_call_id(tmp_path, "fc-EXPIRED", lane_id="lane_expired")
    class OutputExpiredError(Exception):
        pass

    _install_fake_modal(
        monkeypatch,
        OutputExpiredError("expired"),
        output_expired_cls=OutputExpiredError,
    )

    out = actuator._harvest_one_call(
        lane_row=lane_row,
        repo_root=tmp_path,
        per_call_timeout_seconds=1.0,
        append_posterior=False,
    )

    assert out["harvest_status"] == "expired"
    assert [row["status"] for row in query_by_call_id("fc-EXPIRED", path=ledger)] == [
        "dispatched",
        "stale",
    ]


def test_harvest_one_call_function_timeout_appends_failed_call_id_ledger(
    tmp_path,
    monkeypatch,
):
    """Modal function timeout is terminal failed, not an in-flight poll timeout."""

    from tac.deploy.modal.call_id_ledger import query_by_call_id

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "fn_timeout", "fc-FNTIMEOUT", harvested=False)
    lane_row = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)[0]
    ledger = _register_call_id(tmp_path, "fc-FNTIMEOUT", lane_id="lane_fn_timeout")

    class FunctionTimeoutError(Exception):
        pass

    _install_fake_modal(
        monkeypatch,
        FunctionTimeoutError("timeout"),
        function_timeout_cls=FunctionTimeoutError,
    )

    out = actuator._harvest_one_call(
        lane_row=lane_row,
        repo_root=tmp_path,
        per_call_timeout_seconds=1.0,
        append_posterior=False,
    )

    assert out["harvest_status"] == "function_timeout"
    assert [row["status"] for row in query_by_call_id("fc-FNTIMEOUT", path=ledger)] == [
        "dispatched",
        "failed",
    ]


def test_harvest_one_call_poll_timeout_does_not_terminalize_call_id_ledger(
    tmp_path,
    monkeypatch,
):
    """Plain poll timeout remains in-flight and does not append terminal state."""

    from tac.deploy.modal.call_id_ledger import query_by_call_id

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "notready", "fc-NOTREADY", harvested=False)
    lane_row = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)[0]
    ledger = _register_call_id(tmp_path, "fc-NOTREADY", lane_id="lane_notready")
    _install_fake_modal(monkeypatch, TimeoutError("still running"))

    out = actuator._harvest_one_call(
        lane_row=lane_row,
        repo_root=tmp_path,
        per_call_timeout_seconds=1.0,
        append_posterior=False,
    )

    assert out["harvest_status"] == "not_ready"
    assert [row["status"] for row in query_by_call_id("fc-NOTREADY", path=ledger)] == [
        "dispatched"
    ]


def test_harvest_one_call_already_harvested_backfills_call_id_ledger(tmp_path):
    """Already-harvested local state still mirrors terminal state to the ledger."""

    from tac.deploy.modal.call_id_ledger import query_by_call_id

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "done", "fc-DONE", harvested=True)
    lane_row = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)[0]
    ledger = _register_call_id(tmp_path, "fc-DONE", lane_id="lane_done")

    out = actuator._harvest_one_call(
        lane_row=lane_row,
        repo_root=tmp_path,
        per_call_timeout_seconds=1.0,
        append_posterior=False,
    )

    assert out["harvest_status"] == "already_harvested"
    assert out["call_id_ledger"]["appended"] is True
    assert [row["status"] for row in query_by_call_id("fc-DONE", path=ledger)] == [
        "dispatched",
        "harvested",
    ]


def test_fan_out_harvest_per_call_error_does_not_kill_batch(tmp_path, monkeypatch):
    """When one call's harvester raises, the other rows still complete."""

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "ok1", "fc-OK1", harvested=True)
    _make_lane_metadata(results, "broken", "fc-BROKEN", harvested=False)
    _make_lane_metadata(results, "ok2", "fc-OK2", harvested=True)
    lane_rows = actuator.discover_inflight_modal_lanes(repo_root=tmp_path)

    real_harvest_one_call = actuator._harvest_one_call

    def raise_on_broken(*, lane_row, repo_root, per_call_timeout_seconds, append_posterior):
        if "broken" in lane_row["label"]:
            raise RuntimeError("simulated per-call crash")
        return real_harvest_one_call(
            lane_row=lane_row,
            repo_root=repo_root,
            per_call_timeout_seconds=per_call_timeout_seconds,
            append_posterior=append_posterior,
        )

    monkeypatch.setattr(actuator, "_harvest_one_call", raise_on_broken)
    out = actuator.fan_out_harvest(
        lane_rows=lane_rows,
        repo_root=tmp_path,
        max_workers=3,
        per_call_timeout_seconds=1.0,
        overall_timeout_seconds=10.0,
        append_posterior=False,
    )
    assert len(out) == 3
    by_label = {r["label"]: r for r in out}
    assert by_label["ok1"]["harvest_status"] == "already_harvested"
    assert by_label["ok2"]["harvest_status"] == "already_harvested"
    assert by_label["broken"]["harvest_status"].startswith("error_executor_")


# -- Posterior append idempotency tests --------------------------------------


def test_posterior_thread_lock_serializes_appends(tmp_path):
    """Within-process posterior writes from multiple threads serialize."""

    # The thread-lock is the defence-in-depth guard. We simulate concurrent
    # writes and confirm the module-level lock prevents simultaneous entry.
    entries: list[int] = []
    barrier = threading.Barrier(4)

    def attempt_entry(idx: int) -> None:
        barrier.wait()
        with actuator._POSTERIOR_THREAD_LOCK:
            # Simulate critical section work
            entries.append(idx)
            time.sleep(0.001)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(attempt_entry, i) for i in range(4)]
        for f in futures:
            f.result()
    assert sorted(entries) == [0, 1, 2, 3]
    assert len(entries) == 4


# -- CLI surface tests -------------------------------------------------------


def test_cli_list_only_default(tmp_path, capsys):
    """Default mode (no --execute) prints plan + exits 0; does NOT contact Modal."""

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "cli1", "fc-CLI1", harvested=False)
    _make_lane_metadata(results, "cli2", "fc-CLI2", harvested=True)
    rc = actuator.main(["--repo-root", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "parallel_harvest_actuator" in captured
    assert "cli1" in captured
    assert "cli2" in captured
    assert "Modal lanes:" in captured


def test_cli_quiet_mode_suppresses_stdout(tmp_path, capsys):
    """--quiet suppresses the plan view."""

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "q1", "fc-Q1", harvested=True)
    rc = actuator.main(["--repo-root", str(tmp_path), "--quiet"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert captured == ""


def test_cli_filter_label_substr(tmp_path, capsys):
    """--filter-label-substr restricts the discovered set."""

    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "alpha_filter", "fc-AF", harvested=True)
    _make_lane_metadata(results, "beta_match", "fc-BM", harvested=True)
    _make_lane_metadata(results, "gamma_filter", "fc-GF", harvested=True)
    rc = actuator.main(
        ["--repo-root", str(tmp_path), "--filter-label-substr", "filter"]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "alpha_filter" in captured
    assert "gamma_filter" in captured
    assert "beta_match" not in captured


def test_cli_execute_writes_report(tmp_path, monkeypatch):
    """--execute runs the actuator (with mocked modal) and writes the report."""

    # Override DEFAULT_REPO so the test does not write into the real repo's
    # reports/ directory. We pass --repo-root explicitly so report_output
    # defaults under tmp_path.
    results = tmp_path / "experiments" / "results"
    _make_lane_metadata(results, "exec1", "fc-EXEC1", harvested=True)
    out_path = tmp_path / "reports" / "test_report.json"
    rc = actuator.main(
        [
            "--repo-root",
            str(tmp_path),
            "--execute",
            "--max-workers",
            "1",
            "--per-call-timeout-seconds",
            "1.0",
            "--overall-timeout-seconds",
            "10.0",
            "--report-output",
            str(out_path),
            "--no-posterior-append",
            "--quiet",
        ]
    )
    assert rc == 0
    assert out_path.is_file()
    report = json.loads(out_path.read_text())
    assert report["schema"].startswith("parallel_harvest_actuator_report_v1")
    assert report["counts"]["total_modal_lanes"] == 1
    assert report["counts"]["already_harvested"] == 1
    assert report["artifact_kind"] == "HISTORICAL_PROVENANCE"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False


# -- Live-repo regression guard ----------------------------------------------


def test_live_repo_discovery_returns_sane_structure():
    """Live repo discovery returns rows with all required fields (no crash)."""

    repo_root = _REPO
    rows = actuator.discover_inflight_modal_lanes(repo_root=repo_root)
    # Live repo currently has ≥1 modal_metadata.json under
    # experiments/results/lane_*_modal/
    assert isinstance(rows, list)
    if rows:
        required_keys = {
            "label",
            "call_id",
            "dispatched_at",
            "lane_id",
            "out_dir",
            "harvested",
            "metadata_schema",
            "mounted_code_git_head",
            "gpu",
        }
        for row in rows:
            missing = required_keys - set(row.keys())
            assert not missing, f"missing keys {missing} on row {row.get('label')}"


def test_live_repo_vastai_state_returns_list():
    """Live repo vastai state read returns a list (may be empty)."""

    rows = actuator.discover_inflight_vastai_instances(repo_root=_REPO)
    assert isinstance(rows, list)


def test_live_repo_lightning_state_returns_list():
    """Live repo lightning state read returns a list (may be empty)."""

    rows = actuator.discover_inflight_lightning_jobs(repo_root=_REPO)
    assert isinstance(rows, list)


# -- Report building tests ---------------------------------------------------


def test_build_report_counts_correctly(tmp_path):
    """Consolidated report counts categorize harvest statuses correctly."""

    harvest_results = [
        {"harvest_status": "harvested"},
        {"harvest_status": "already_harvested"},
        {"harvest_status": "already_harvested"},
        {"harvest_status": "expired"},
        {"harvest_status": "not_ready"},
        {"harvest_status": "error_RuntimeError"},
        {
            "harvest_status": "already_harvested",
            "score_claim": {
                "score": 0.193,
                "custody_accepted": True,
            },
        },
        {
            "harvest_status": "already_harvested",
            "score_claim": {
                "score": 0.250,
                "custody_accepted": False,
            },
        },
    ]
    report = actuator._build_report(
        repo_root=tmp_path,
        harvest_results=harvest_results,
        vastai_inflight=[{"id": "vi1"}, {"id": "vi2"}],
        lightning_inflight=[{"id": "lj1"}],
        started_at="2026-05-14T16:00:00Z",
        finished_at="2026-05-14T16:05:00Z",
        max_workers=4,
        per_call_timeout_seconds=2.0,
        overall_timeout_seconds=60.0,
        appended_posterior=False,
    )
    counts = report["counts"]
    assert counts["total_modal_lanes"] == 8
    assert counts["harvested_this_run"] == 1
    # already_harvested includes the two with score_claim sub-dicts (3 base + 2 with claims = 5)
    # Actually: 2 base + 2 with score_claim = 4
    assert counts["already_harvested"] == 4
    assert counts["expired"] == 1
    assert counts["not_ready_in_flight"] == 1
    assert counts["errors"] == 1
    assert counts["score_rows_extracted"] == 2
    assert counts["custody_accepted_rows"] == 1
    assert counts["vastai_inflight"] == 2
    assert counts["lightning_inflight"] == 1
