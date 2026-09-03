from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "tools" / "launch_detached_process.py"
LIVENESS = REPO / "tools" / "run_liveness_watcher.py"
QUALITY = REPO / "tools" / "run_quality_poller.py"
QUEUE = REPO / "tools" / "codex_arm_queue.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launch_mod = _load("watched_launch_launcher_test", LAUNCHER)
watch_mod = _load("watched_launch_monitor_test", REPO / "tools" / "codex_arm_watch.py")
liveness_mod = _load("watched_launch_liveness_test", LIVENESS)
quality_mod = _load("watched_launch_quality_test", QUALITY)


def _quality_config(tmp_path: Path, log: Path, pid_file: Path) -> Path:
    path = tmp_path / "quality-config.json"
    path.write_text(
        json.dumps(
            {
                "schema": "pact.run_quality_poller.config.v1",
                "log_path": str(log),
                "pid_file": str(pid_file),
                "telemetry_path": str(tmp_path / "quality-telemetry.jsonl"),
                "alert_path": str(tmp_path / "quality-alert.json"),
                "poll_s": 1,
                "eval_period_s": 10,
                "stale_periods": 3,
                "startup_grace_s": 0,
                "json_marker": "estimated_joint_bytes",
                "fields": {
                    "epoch": "epoch",
                    "value": "estimated_joint_bytes",
                    "phase": "phase",
                    "finite": ["bpp", "top1_error"],
                },
                "bar_value": 131220,
                "bar_start_epoch": 481,
                "phase_knee": {
                    "epoch": 481,
                    "window_epochs": 3,
                    "shock_multiplier": 1.25,
                    "continuous_phase": "continuous",
                },
                "best_not_latest": {
                    "phase": "discrete_qat",
                    "min_rows": 4,
                    "lag_epochs": 6,
                },
                "alert_conditions": {
                    "joint_regression": False,
                    "qat_knee_shock": False,
                    "nan_or_garbage": True,
                    "stale_telemetry": False,
                },
            }
        )
    )
    return path


def _run_launcher(tmp_path: Path, *args: str, timeout: float = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LAUNCHER), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_stale_root_refuses_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    root.mkdir()
    sentinel = root / "checkpoints"
    sentinel.mkdir()
    result = _run_launcher(
        tmp_path,
        "--fresh-root",
        str(root),
        "--output-dir",
        str(root / "detached"),
        "--cwd",
        str(tmp_path),
        "--dry-run",
        "--",
        "/usr/bin/true",
    )
    assert result.returncode == 7
    payload = json.loads(result.stderr)
    assert payload["fresh_root"] == str(root)
    assert sentinel.is_dir()
    assert not (root / "detached").exists()


def test_fresh_root_suffix_mints_and_rewrites_real_argv(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    root.mkdir()
    (root / "prior.txt").write_text("keep", encoding="utf-8")
    target = root / "actual.txt"
    result = _run_launcher(
        tmp_path,
        "--fresh-root",
        str(root),
        "--fresh-root-suffix",
        "--output-dir",
        str(root / "detached"),
        "--cwd",
        str(tmp_path),
        "--verify-alive-secs",
        "0.2",
        "--",
        sys.executable,
        "-c",
        "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('real'); time.sleep(5)",
        str(target),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    pid = int(payload["pid"])
    try:
        effective = Path(payload["fresh_roots"][0]["effective_path"])
        assert effective != root
        deadline = time.time() + 3
        while time.time() < deadline and not (effective / "actual.txt").exists():
            time.sleep(0.05)
        assert (effective / "actual.txt").read_text() == "real"
        assert (root / "prior.txt").read_text() == "keep"
        assert not target.exists()
        manifest = json.loads((effective / "detached" / "launch_manifest.json").read_text())
        assert str(effective / "actual.txt") in manifest["argv"]
    finally:
        _kill_group(pid)


def test_unconsumed_receipt_refuses_then_explicit_supersede_tombstones(tmp_path: Path) -> None:
    first = _run_launcher(
        tmp_path,
        "--output-dir",
        str(tmp_path / "first"),
        "--cwd",
        str(tmp_path),
        "--done-receipt",
        "same_name",
        "--verify-alive-secs",
        "0",
        "--",
        sys.executable,
        "-c",
        "pass",
    )
    assert first.returncode == 0, first.stderr
    done = tmp_path / ".omx" / "tmp" / "codex_runs" / "same_name.done"
    deadline = time.time() + 3
    while time.time() < deadline and not done.exists():
        time.sleep(0.05)
    receipt = json.loads(done.read_text())
    assert set(receipt["launch_id"]) == {"manifest_path", "pid", "monotonic_launch_counter"}

    refused = _run_launcher(
        tmp_path,
        "--output-dir",
        str(tmp_path / "second"),
        "--cwd",
        str(tmp_path),
        "--done-receipt",
        "same_name",
        "--dry-run",
        "--",
        sys.executable,
        "-c",
        "pass",
    )
    assert refused.returncode == 6
    assert "unconsumed" in json.loads(refused.stderr)["error"]

    preview = _run_launcher(
        tmp_path,
        "--output-dir",
        str(tmp_path / "preview"),
        "--cwd",
        str(tmp_path),
        "--done-receipt",
        "same_name",
        "--receipt-supersede",
        "--dry-run",
        "--",
        sys.executable,
        "-c",
        "pass",
    )
    assert preview.returncode == 0, preview.stderr
    assert done.exists(), "dry-run must not tombstone the live receipt"
    assert not list(done.parent.glob("same_name.done.superseded.*"))

    superseded = _run_launcher(
        tmp_path,
        "--output-dir",
        str(tmp_path / "third"),
        "--cwd",
        str(tmp_path),
        "--done-receipt",
        "same_name",
        "--receipt-supersede",
        "--verify-alive-secs",
        "0.1",
        "--",
        "/bin/sleep",
        "5",
    )
    assert superseded.returncode == 0, superseded.stderr
    payload = json.loads(superseded.stdout)
    try:
        tombstones = list(done.parent.glob("same_name.done.superseded.*.tombstone.json"))
        assert len(tombstones) == 1
        tombstone = json.loads(tombstones[0].read_text())
        assert tombstone["receipt_sha256"]
        assert tombstone["reason"] == "explicit_unconsumed_supersede"
        active_refusal = _run_launcher(
            tmp_path,
            "--output-dir",
            str(tmp_path / "fourth"),
            "--cwd",
            str(tmp_path),
            "--done-receipt",
            "same_name",
            "--receipt-supersede",
            "--dry-run",
            "--",
            "/usr/bin/true",
        )
        assert active_refusal.returncode == 6
        assert "active launch reservation" in json.loads(active_refusal.stderr)["error"]
    finally:
        _kill_group(int(payload["pid"]))


def test_verify_alive_failure_receipt_is_suppressed_by_real_monitor_reader(tmp_path: Path) -> None:
    result = _run_launcher(
        tmp_path,
        "--output-dir",
        str(tmp_path / "dead"),
        "--cwd",
        str(tmp_path),
        "--done-receipt",
        "adjudicated",
        "--verify-alive-secs",
        "0.3",
        "--",
        sys.executable,
        "-c",
        "import sys; print('measured failure'); sys.exit(17)",
    )
    assert result.returncode == 17
    runs = tmp_path / ".omx" / "tmp" / "codex_runs"
    receipt = json.loads((runs / "adjudicated.done").read_text())
    assert receipt["adjudicated_at_launch"] is True
    assert receipt["rc"] == 17
    assert watch_mod.format_events(runs, {}, watch_mod._snapshot(runs)) == []


def test_nice_application_is_verified_and_mismatch_fails_closed(monkeypatch) -> None:
    calls: list[tuple[int, int, int]] = []

    def setpriority(which: int, pid: int, value: int) -> None:
        calls.append((which, pid, value))

    monkeypatch.setattr(launch_mod.os, "setpriority", setpriority)
    monkeypatch.setattr(launch_mod.os, "getpriority", lambda _which, _pid: 10)
    assert launch_mod._apply_and_verify_nice(12345, 10) == 10
    assert calls == [(os.PRIO_PROCESS, 12345, 10)]
    monkeypatch.setattr(launch_mod.os, "getpriority", lambda _which, _pid: 0)
    with pytest.raises(launch_mod.LaunchRefusal) as exc:
        launch_mod._apply_and_verify_nice(12345, 10)
    assert exc.value.rc == 8


def test_resource_budgets_derive_from_measured_need_and_governor_ceiling(
    tmp_path: Path,
) -> None:
    out = tmp_path / "resource_plan"
    result = _run_launcher(
        tmp_path,
        "--output-dir",
        str(out),
        "--cwd",
        str(REPO),
        "--derive-resource-budgets",
        "--measured-peak-rss-gib",
        "12.5",
        "--measured-thread-need",
        "2",
        "--walltime-cap-s",
        "7200",
        "--dry-run",
        "--",
        "/usr/bin/true",
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "launch_manifest.json").read_text())
    budget = manifest["resource_budget"]
    assert budget["mode"] == "derived_and_enforced"
    assert budget["measured_peak_rss_gib"] == 12.5
    assert budget["operator_ceiling_gib"] == 116.0
    assert budget["rss_cap_mib"] == 116 * 1024
    assert budget["thread_budget"] == 2
    assert "tools/safe_run.py" in " ".join(manifest["effective_argv"])
    assert "16384" not in " ".join(manifest["effective_argv"])


def test_liveness_watcher_dead_pid_round_trip(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    pid_file.write_text("99999999\n")
    receipt = tmp_path / "status.json"
    receipt.write_text("{}\n")
    alert = tmp_path / "liveness.alert"
    manifest = tmp_path / "launch_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "launch_id": {
                    "manifest_path": str(manifest),
                    "pid": 12345,
                    "monotonic_launch_counter": 9,
                }
            }
        )
    )
    event_receipt = tmp_path / "runs" / "watch_liveness.done"
    config = tmp_path / "liveness.json"
    config.write_text(
        json.dumps(
            {
                "schema": "pact.run_liveness_watcher.config.v1",
                "pid_file": str(pid_file),
                "alert_path": str(alert),
                "poll_s": 0.01,
                "warmup_s": 0,
                "receipt_checks": [
                    {"label": "status", "path": str(receipt), "max_age_s": 60, "grace_s": 0}
                ],
                "heartbeat_checks": [],
                "artifact_checks": [],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(LIVENESS),
            "--config",
            str(config),
            "--once",
            "--event-receipt",
            str(event_receipt),
            "--launch-manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    payload = json.loads(alert.read_text())
    assert payload["reason"] == "child_dead"
    assert payload["pid"] == 99999999
    event = json.loads(event_receipt.read_text())
    assert event["launch_id"]["monotonic_launch_counter"] == 9
    lines = watch_mod.format_events(event_receipt.parent, {}, watch_mod._snapshot(event_receipt.parent))
    assert len(lines) == 1 and "ALERT rc=1" in lines[0]


@pytest.mark.parametrize(
    "writer",
    [liveness_mod.atomic_write_once, quality_mod._atomic_write_once],
)
def test_watcher_alert_publish_does_not_require_hard_links(
    tmp_path: Path, monkeypatch, writer
) -> None:
    def hard_link_is_unsupported(*_args, **_kwargs):
        raise OSError(45, "Operation not supported")

    monkeypatch.setattr(os, "link", hard_link_is_unsupported)
    alert = tmp_path / "watcher.alert.json"
    assert writer(alert, {"sequence": 1}) is True
    assert writer(alert, {"sequence": 2}) is False
    assert json.loads(alert.read_text()) == {"sequence": 1}


def test_rx2_quality_semantics_are_config_driven(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    log = tmp_path / "run.log"
    log.write_text(
        json.dumps(
            {
                "epoch": 33,
                "phase": "discrete_qat",
                "estimated_joint_bytes": 186074,
                "bpp": 1.0,
            }
        )
        + "\n"
    )
    alert = tmp_path / "quality.alert"
    config = tmp_path / "quality.json"
    config.write_text(
        json.dumps(
            {
                "schema": "pact.run_quality_poller.config.v1",
                "log_path": str(log),
                "pid_file": str(pid_file),
                "telemetry_path": str(tmp_path / "telemetry.jsonl"),
                "alert_path": str(alert),
                "poll_s": 1,
                "eval_period_s": 2880,
                "stale_periods": 3,
                "startup_grace_s": 0,
                "json_marker": "\"estimated_joint_bytes\"",
                "fields": {
                    "epoch": "epoch",
                    "value": "estimated_joint_bytes",
                    "phase": "phase",
                    "finite": ["bpp"],
                },
                "bar_value": 186073,
                "bar_start_epoch": 33,
                "phase_knee": {
                    "epoch": 31,
                    "window_epochs": 3,
                    "shock_multiplier": 1.25,
                    "continuous_phase": "continuous",
                },
                "best_not_latest": {"phase": "discrete_qat", "min_rows": 4, "lag_epochs": 6},
                "alert_conditions": {
                    "joint_regression": True,
                    "qat_knee_shock": True,
                    "nan_or_garbage": True,
                    "stale_telemetry": True,
                },
            }
        )
    )
    result = subprocess.run(
        [sys.executable, str(QUALITY), "--config", str(config), "--once"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    payload = json.loads(alert.read_text())
    assert payload["reason"] == "joint_regression"
    assert payload["bar"] == 186073


def test_quality_regression_band_alerts_on_top1_without_joint_regression(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    log = tmp_path / "run.log"
    log.write_text(
        json.dumps(
            {
                "epoch": 482,
                "phase": "discrete_qat",
                "estimated_joint_bytes": 130000,
                "bpp": 1.0,
                "top1_error": 0.0021,
            }
        )
        + "\n"
    )
    alert = tmp_path / "quality.alert"
    config = tmp_path / "quality.json"
    config.write_text(
        json.dumps(
            {
                "schema": "pact.run_quality_poller.config.v1",
                "log_path": str(log),
                "pid_file": str(pid_file),
                "telemetry_path": str(tmp_path / "telemetry.jsonl"),
                "alert_path": str(alert),
                "poll_s": 1,
                "eval_period_s": 2880,
                "stale_periods": 3,
                "startup_grace_s": 0,
                "json_marker": "estimated_joint_bytes",
                "fields": {
                    "epoch": "epoch",
                    "value": "estimated_joint_bytes",
                    "phase": "phase",
                    "finite": ["bpp", "top1_error"],
                },
                "bar_value": 131220,
                "bar_start_epoch": 481,
                "regression_bands": [
                    {
                        "label": "top1_error_parent_discrete_qat_max",
                        "field": "top1_error",
                        "upper": 0.002,
                        "start_epoch": 481,
                    }
                ],
                "phase_knee": {
                    "epoch": 481,
                    "window_epochs": 3,
                    "shock_multiplier": 1.25,
                    "continuous_phase": "continuous",
                },
                "best_not_latest": {
                    "phase": "discrete_qat",
                    "min_rows": 4,
                    "lag_epochs": 6,
                },
                "alert_conditions": {
                    "joint_regression": False,
                    "qat_knee_shock": False,
                    "nan_or_garbage": True,
                    "stale_telemetry": False,
                },
            }
        )
    )
    result = subprocess.run(
        [sys.executable, str(QUALITY), "--config", str(config), "--once"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    payload = json.loads(alert.read_text())
    assert payload["reason"] == "regression_band"
    assert payload["field"] == "top1_error"
    assert payload["upper"] == 0.002


def test_quality_regression_band_fails_closed_when_field_is_missing(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    log = tmp_path / "run.log"
    log.write_text(
        json.dumps(
            {
                "epoch": 482,
                "phase": "discrete_qat",
                "estimated_joint_bytes": 130000,
                "bpp": 1.0,
            }
        )
        + "\n"
    )
    config = quality_mod.load_config(_quality_config(tmp_path, log, pid_file))
    config["conditions"]["nan_or_garbage"] = False
    config["regression_bands"] = [
        {
            "label": "top1_error_parent_band",
            "field": "top1_error",
            "upper": 0.002,
            "start_epoch": 481,
        }
    ]
    alert, _ = quality_mod.poll_once(config, quality_mod.PollState(), now=1.0)
    assert alert == {
        "reason": "regression_band_unreadable",
        "label": "top1_error_parent_band",
        "field": "top1_error",
        "epoch": 482,
    }


def test_arm_watchers_spawns_both_canonical_tools_before_job(tmp_path: Path) -> None:
    out = tmp_path / "watched"
    status = tmp_path / "status.json"
    status.write_text("{}\n")
    liveness = tmp_path / "liveness.json"
    liveness.write_text(
        json.dumps(
            {
                "schema": "pact.run_liveness_watcher.config.v1",
                "pid_file": str(out / "run.pid"),
                "alert_path": str(tmp_path / "live.alert"),
                "poll_s": 1,
                "warmup_s": 5,
                "receipt_checks": [
                    {"path": str(status), "max_age_s": 60, "grace_s": 0, "label": "status"}
                ],
                "heartbeat_checks": [],
                "artifact_checks": [],
            }
        )
    )
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "schema": "pact.run_quality_poller.config.v1",
                "log_path": str(out / "run.log"),
                "pid_file": str(out / "run.pid"),
                "telemetry_path": str(tmp_path / "telemetry.jsonl"),
                "alert_path": str(tmp_path / "quality.alert"),
                "poll_s": 1,
                "eval_period_s": 60,
                "stale_periods": 3,
                "startup_grace_s": 5,
                "json_marker": "\"metric\"",
                "fields": {"epoch": "epoch", "value": "metric", "phase": "phase", "finite": []},
                "bar_value": 10,
                "bar_start_epoch": 10,
                "phase_knee": {
                    "epoch": 5,
                    "window_epochs": 2,
                    "shock_multiplier": 2,
                    "continuous_phase": "continuous",
                },
                "best_not_latest": {"phase": "qat", "min_rows": 4, "lag_epochs": 6},
                "alert_conditions": {
                    "joint_regression": False,
                    "qat_knee_shock": False,
                    "nan_or_garbage": True,
                    "stale_telemetry": False,
                },
            }
        )
    )
    result = _run_launcher(
        tmp_path,
        "--output-dir",
        str(out),
        "--cwd",
        str(tmp_path),
        "--arm-watchers",
        "--liveness-config",
        str(liveness),
        "--quality-config",
        str(quality),
        "--verify-alive-secs",
        "0.2",
        "--",
        "/bin/sleep",
        "5",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    try:
        assert {row["kind"] for row in payload["watchers"]} == {"liveness", "quality"}
        manifest = json.loads((out / "launch_manifest.json").read_text())
        assert len(manifest["watchers"]) == 2
        assert all(row["pid"] > 1 for row in manifest["watchers"])
    finally:
        _kill_group(int(payload["pid"]))
        for row in payload["watchers"]:
            try:
                os.kill(int(row["pid"]), signal.SIGKILL)
            except ProcessLookupError:
                pass

    failed_out = tmp_path / "watched_early_failure"
    failed = _run_launcher(
        tmp_path,
        "--output-dir",
        str(failed_out),
        "--cwd",
        str(tmp_path),
        "--done-receipt",
        "watched_early_failure",
        "--arm-watchers",
        "--liveness-config",
        str(liveness),
        "--quality-config",
        str(quality),
        "--verify-alive-secs",
        "0.3",
        "--",
        sys.executable,
        "-c",
        "import sys; sys.exit(17)",
    )
    assert failed.returncode == 17
    failed_manifest = json.loads((failed_out / "launch_manifest.json").read_text())
    assert failed_manifest["adjudicated_at_launch"] is True
    assert all(row["stopped_at_launch"] is True for row in failed_manifest["watchers"])
    runs = tmp_path / ".omx" / "tmp" / "codex_runs"
    receipt = json.loads((runs / "watched_early_failure.done").read_text())
    assert receipt["adjudicated_at_launch"] is True
    assert not list(runs.glob("watched_launch_2_*.done"))
    assert watch_mod.format_events(runs, {}, watch_mod._snapshot(runs)) == []


def test_capability_lint_warns_and_names_main_handoff(tmp_path: Path) -> None:
    charter = tmp_path / "charter.md"
    charter.write_text(
        "# fixture\n\nOPTIMAL_FORM_NA: advisory fixture only, no implementation\n"
        "The arm must run nice -n 10 and prove the child priority.\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(QUEUE),
            "lint",
            "--name",
            "nice_fixture",
            "--prompt",
            str(charter),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "charter-lint WARN [nice_fixture]" in result.stdout
    assert "process_priority_control" in result.stdout
    assert "MAIN-handoff" in result.stdout


def test_nice_best_effort_degrades_instead_of_refusing(monkeypatch):
    """--nice-best-effort: a sandbox that refuses setpriority must not refuse the LAUNCH
    (measured 2026-09-03: rc=8 pushed an arm into in-session compute at default priority)."""
    import tools.launch_detached_process as launch_mod

    def _deny(*_a, **_k):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(launch_mod.os, "setpriority", _deny)
    assert launch_mod._apply_and_verify_nice(12345, 10, best_effort=True) is None
    with pytest.raises(launch_mod.LaunchRefusal):
        launch_mod._apply_and_verify_nice(12345, 10)
