# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.audit_contract import AuditReport

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location("all_lanes_preflight_timing_test", ALL_LANES)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(module: Any, section: str, number: int, name: str, elapsed_s: float, passed: bool):
    step = module.PreflightStep(
        section,
        number,
        name,
        lambda: (passed, name),
        f"{name} passed",
        f"{name} failed",
        forensic_only=(name == "forensic"),
        local_smoke_only=(name == "smoke"),
    )
    return module.PreflightResult(step, passed, name, elapsed_s)


def test_timing_profile_payload_is_deterministic_and_hot_sorted() -> None:
    module = _load_all_lanes_module()
    results = [
        _result(module, "LANE", 1, "smoke", 0.25, True),
        _result(module, "GATE", 2, "slow gate", 3.5, False),
        _result(module, "GATE", 1, "fast gate", 0.125, True),
    ]

    payload = module._build_timing_profile(results, max_workers=4, wall_elapsed_s=3.75)

    assert payload["schema"] == module.TIMING_PROFILE_SCHEMA
    assert payload["max_workers"] == 4
    assert payload["step_count"] == 3
    assert payload["passed_count"] == 2
    assert payload["failed_count"] == 1
    assert payload["wall_elapsed_s"] == 3.75
    assert payload["serial_elapsed_s"] == 3.875
    assert payload["parallel_speedup_estimate"] == 1.033333
    assert payload["slow_step_threshold_s"] == module.SLOW_STEP_THRESHOLD_S
    assert payload["slow_step_count"] == 1
    assert [row["name"] for row in payload["steps"]] == ["fast gate", "slow gate", "smoke"]
    assert [row["name"] for row in payload["slow_steps"]] == ["slow gate"]
    assert [row["name"] for row in payload["hot_steps"]] == ["slow gate", "smoke", "fast gate"]
    assert payload["steps"][2]["local_smoke_only"] is True


def test_print_timing_summary_reports_parallelism_and_slow_steps(capsys) -> None:
    module = _load_all_lanes_module()
    results = [
        _result(module, "GATE", 1, "fast gate", 0.125, True),
        _result(module, "GATE", 2, "slow gate", 3.5, True),
        _result(module, "LANE", 1, "smoke", 0.25, True),
    ]

    module._print_timing_summary(results, max_workers=4, wall_elapsed_s=3.75)

    out = capsys.readouterr().out
    assert "wall=3.75s" in out
    assert "serial_sum=3.88s" in out
    assert "workers=4" in out
    assert "estimated_speedup=1.03x" in out
    assert "slow_steps=1/3" in out
    assert "Slow steps (>= 0.50s)" in out
    assert "GATE #2: slow gate" in out
    assert "Remaining steps:" in out


def test_all_lanes_preflight_timeout_default_is_thirty_seconds() -> None:
    module = _load_all_lanes_module()

    assert module.DEFAULT_ALL_LANES_PREFLIGHT_TIMEOUT_S == 30.0
    assert (
        module._all_lanes_preflight_timeout_seconds(
            timeout_s=module.DEFAULT_ALL_LANES_PREFLIGHT_TIMEOUT_S,
            allow_slow_preflight=False,
        )
        == 30.0
    )


def test_all_lanes_preflight_slow_override_is_explicit() -> None:
    module = _load_all_lanes_module()

    assert (
        module._all_lanes_preflight_timeout_seconds(
            timeout_s=30.0,
            allow_slow_preflight=True,
        )
        is None
    )


def test_all_lanes_preflight_timeout_rejects_non_positive_budget() -> None:
    module = _load_all_lanes_module()

    try:
        module._all_lanes_preflight_timeout_seconds(
            timeout_s=0.0,
            allow_slow_preflight=False,
        )
    except ValueError as exc:
        assert "--timeout-s" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected ValueError")


def test_hard_watchdog_can_be_disabled_for_diagnostic_runs(monkeypatch) -> None:
    module = _load_all_lanes_module()

    monkeypatch.setenv("PACT_ALL_LANES_PREFLIGHT_HARD_WATCHDOG", "0")

    assert module._hard_watchdog_enabled() is False
    assert module._start_hard_wall_clock_watchdog(30.0) is None


def test_hard_watchdog_fires_exit_func_after_budget_plus_grace(monkeypatch, capsys) -> None:
    module = _load_all_lanes_module()
    fired = module.threading.Event()
    exit_codes: list[int] = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        fired.set()

    monkeypatch.setenv("PACT_ALL_LANES_PREFLIGHT_HARD_WATCHDOG", "1")
    monkeypatch.setenv("PACT_ALL_LANES_PREFLIGHT_HARD_WATCHDOG_GRACE_S", "0")

    timer = module._start_hard_wall_clock_watchdog(0.01, exit_func=fake_exit)
    assert timer is not None
    assert fired.wait(timeout=1.0)
    timer.cancel()

    assert exit_codes == [124]
    assert "hard watchdog fired" in capsys.readouterr().err


def test_hard_watchdog_message_names_noncooperative_gate_class() -> None:
    module = _load_all_lanes_module()

    message = module._format_hard_watchdog_message(timeout_s=30.0, grace_s=2.0)

    assert "30.00s budget + 2.00s grace" in message
    assert "did not return cooperatively" in message
    assert "DO NOT DISPATCH" in message


def test_all_lanes_budget_failure_reports_hot_steps() -> None:
    module = _load_all_lanes_module()
    results = [
        _result(module, "GATE", 1, "fast gate", 0.125, True),
        _result(module, "GATE", 2, "slow gate", 3.5, True),
    ]

    message = module._format_wall_clock_budget_failure(
        results,
        wall_elapsed_s=30.25,
        timeout_s=30.0,
    )

    assert "all-lanes preflight exceeded 30.00s wall-clock DX budget" in message
    assert "DO NOT DISPATCH" in message
    assert "GATE #2: slow gate" in message
    assert "GATE #1: fast gate" in message
    assert "--allow-slow-preflight" in message


def test_execute_step_marks_subprocess_budget_timeout(monkeypatch) -> None:
    module = _load_all_lanes_module()

    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(["slow-tool"], kwargs.get("timeout", 0.01))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    step = module.PreflightStep(
        "GATE",
        1,
        "slow subprocess",
        lambda: (
            False,
            module._run_subprocess(["slow-tool"], capture_output=True, text=True).stderr,
        ),
        "passed",
        "failed",
    )
    context = module.PreflightRunContext(
        started_s=time.perf_counter(),
        deadline_s=time.perf_counter() + 1.0,
        cancel_event=module.threading.Event(),
    )

    result = module._execute_step(step, context)

    assert result.passed is False
    assert result.status == "timeout"
    assert "TIMEOUT: all-lanes preflight wall-clock budget exhausted" in result.output


def test_active_dispatch_claims_gate_passes_clean_summary(monkeypatch) -> None:
    module = _load_all_lanes_module()

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        assert cmd[:2] == [sys.executable, str(module.CLAIM_LANE_DISPATCH)]
        assert cmd[2:5] == ["summary", "--format", "json"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "active_count": 0,
                    "stale_nonterminal_count": 0,
                    "terminal_latest_count": 7,
                    "unparsable_timestamp_count": 0,
                    "invalid_lane_id_count": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module, "_run_subprocess", fake_run)

    passed, output = module._run_active_dispatch_claims_gate()

    assert passed is True
    assert "active dispatch claims: PASS" in output


def test_active_dispatch_claims_gate_fails_on_active_or_stale_claims(monkeypatch) -> None:
    module = _load_all_lanes_module()

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        assert cmd[:5] == [
            sys.executable,
            str(module.CLAIM_LANE_DISPATCH),
            "summary",
            "--format",
            "json",
        ]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "active_count": 1,
                    "stale_nonterminal_count": 2,
                    "terminal_latest_count": 7,
                    "unparsable_timestamp_count": 0,
                    "invalid_lane_id_count": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module, "_run_subprocess", fake_run)

    passed, output = module._run_active_dispatch_claims_gate()

    assert passed is False
    assert "active=1" in output
    assert "stale_nonterminal=2" in output


def test_run_steps_with_budget_cancels_unstarted_serial_work() -> None:
    module = _load_all_lanes_module()
    ran: list[str] = []

    def first_runner() -> tuple[bool, str]:
        ran.append("first")
        time.sleep(0.03)
        return True, "first done"

    def second_runner() -> tuple[bool, str]:
        ran.append("second")
        return True, "second done"

    steps = [
        module.PreflightStep("GATE", 1, "first", first_runner, "first passed", "first failed"),
        module.PreflightStep("GATE", 2, "second", second_runner, "second passed", "second failed"),
    ]

    results = module._run_steps_with_budget(
        steps,
        max_workers=1,
        wall_clock_budget_s=0.01,
        run_started=time.perf_counter(),
    )

    assert ran == ["first"]
    assert [result.status for result in results] == ["passed", "cancelled"]
    assert "CANCELLED" in results[1].output


def test_run_steps_with_budget_parallel_preserves_full_coverage_and_order() -> None:
    module = _load_all_lanes_module()
    ran: list[str] = []

    def runner(name: str, passed: bool, delay_s: float) -> tuple[bool, str]:
        time.sleep(delay_s)
        ran.append(name)
        return passed, f"{name} done"

    steps = [
        module.PreflightStep(
            "LANE",
            2,
            "lane two",
            lambda: runner("lane two", True, 0.005),
            "lane two passed",
            "lane two failed",
        ),
        module.PreflightStep(
            "GATE",
            1,
            "gate one",
            lambda: runner("gate one", True, 0.015),
            "gate one passed",
            "gate one failed",
        ),
        module.PreflightStep(
            "GATE",
            2,
            "gate two",
            lambda: runner("gate two", False, 0.001),
            "gate two passed",
            "gate two failed",
        ),
    ]

    results = module._run_steps_with_budget(
        steps,
        max_workers=3,
        wall_clock_budget_s=5.0,
        run_started=time.perf_counter(),
    )

    assert set(ran) == {"gate one", "gate two", "lane two"}
    assert [(r.step.section, r.step.number, r.step.name) for r in results] == [
        ("GATE", 1, "gate one"),
        ("GATE", 2, "gate two"),
        ("LANE", 2, "lane two"),
    ]
    assert [r.passed for r in results] == [True, False, True]
    assert [r.status for r in results] == ["passed", "failed", "passed"]


def test_write_timing_profile_creates_parent_and_uses_repo_json_style(tmp_path: Path) -> None:
    module = _load_all_lanes_module()
    path = tmp_path / "profiles" / "all_lanes_timing.json"
    payload = {
        "schema": module.TIMING_PROFILE_SCHEMA,
        "steps": [{"section": "GATE", "number": 1, "elapsed_s": 0.1}],
    }

    module._write_timing_profile(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert path.read_text(encoding="utf-8").endswith("\n")


@dataclass(frozen=True)
class _SemanticResult:
    ok: bool = True
    contract_ok: bool = True
    blocking_findings: tuple[object, ...] = ()
    advisory_findings: tuple[object, ...] = ()

    @property
    def findings(self) -> tuple[object, ...]:
        return self.blocking_findings + self.advisory_findings


def test_scoped_fast_gates_run_in_process_without_subprocess(monkeypatch) -> None:
    module = _load_all_lanes_module()

    from tools import audit_semantic_label_contract, audit_tooling_consolidation, check_dispatch_cli_shell_hazards

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("scoped fast gates must not shell out")

    source_index = object()

    def fake_scan_paths(*_args: object, **kwargs: object) -> list[object]:
        assert kwargs["source_index"] is source_index
        return []

    def fake_semantic_audit(**kwargs: object) -> _SemanticResult:
        assert kwargs["source_index"] is source_index
        return _SemanticResult()

    def fake_tooling_audit(*_args: object, **kwargs: object) -> AuditReport:
        assert kwargs["source_index"] is source_index
        return AuditReport(
            audit="tooling_consolidation_inventory",
            readiness_key="ready_for_incremental_consolidation",
            ready=True,
            summary={
                "file_count": 2,
                "pattern_counts": {
                    "local_sha256_helper": 0,
                    "local_json_dump": 0,
                },
            },
        )

    monkeypatch.setattr(module.subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(
        check_dispatch_cli_shell_hazards,
        "scan_paths",
        fake_scan_paths,
    )
    monkeypatch.setattr(
        audit_semantic_label_contract,
        "audit_semantic_label_contract",
        fake_semantic_audit,
    )
    monkeypatch.setattr(
        audit_tooling_consolidation,
        "audit_tooling",
        fake_tooling_audit,
    )

    assert module._run_dispatch_cli_shell_hazards_gate(source_index=source_index) == (
        True,
        "dispatch CLI/shell hazards: PASS",
    )
    semantic_ok, semantic_output = module._run_semantic_label_contract_gate(
        source_index=source_index
    )
    tooling_ok, tooling_output = module._run_tooling_consolidation_gate(
        source_index=source_index
    )

    assert semantic_ok is True
    assert "semantic-label contract: PASS" in semantic_output
    assert tooling_ok is True
    assert "2 files scanned" in tooling_output


def test_shared_source_index_gate_serializes_cache_heavy_scans(monkeypatch) -> None:
    module = _load_all_lanes_module()
    events: list[str] = []
    inside = module.threading.Event()
    release = module.threading.Event()

    def first_runner() -> tuple[bool, str]:
        events.append("first-enter")
        inside.set()
        assert release.wait(timeout=1.0)
        events.append("first-exit")
        return True, "first"

    def second_runner() -> tuple[bool, str]:
        events.append("second-enter")
        events.append("second-exit")
        return True, "second"

    with module.concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(module._run_source_index_gate, first_runner)
        assert inside.wait(timeout=1.0)
        second = pool.submit(module._run_source_index_gate, second_runner)
        module.time.sleep(0.01)
        assert events == ["first-enter"]
        release.set()

    assert first.result(timeout=1.0) == (True, "first")
    assert second.result(timeout=1.0) == (True, "second")
    assert events == ["first-enter", "first-exit", "second-enter", "second-exit"]
