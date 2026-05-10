from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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
