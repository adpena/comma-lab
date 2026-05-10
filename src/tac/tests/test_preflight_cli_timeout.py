from __future__ import annotations

import json
import time

import pytest

import tac.preflight as preflight
from tac.preflight import (
    DEFAULT_PREFLIGHT_CLI_TIMEOUT_S,
    PREFLIGHT_CLI_TIMING_SCHEMA,
    PreflightTimeoutError,
    _PreflightCliTimingRecorder,
    _build_preflight_cli_timing_payload,
    _preflight_cli_timeout_seconds,
    _preflight_timeout_after,
    _write_preflight_cli_timing_json,
)


def test_cli_timeout_default_is_thirty_seconds() -> None:
    assert DEFAULT_PREFLIGHT_CLI_TIMEOUT_S == 30.0
    assert (
        _preflight_cli_timeout_seconds(
            timeout_s=DEFAULT_PREFLIGHT_CLI_TIMEOUT_S,
            allow_slow_preflight=False,
        )
        == 30.0
    )


def test_cli_slow_override_must_be_explicit() -> None:
    assert (
        _preflight_cli_timeout_seconds(
            timeout_s=30.0,
            allow_slow_preflight=True,
        )
        is None
    )


def test_cli_timeout_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="--timeout-s"):
        _preflight_cli_timeout_seconds(
            timeout_s=0.0,
            allow_slow_preflight=False,
        )


def test_cli_timeout_context_raises_quickly() -> None:
    with pytest.raises(PreflightTimeoutError, match="preflight exceeded"):
        with _preflight_timeout_after(0.01):
            time.sleep(0.05)


def test_cli_timing_recorder_wraps_scope_checks(monkeypatch) -> None:
    events: list[tuple[bool, bool]] = []

    def check_fake_cli_timing(*, strict: bool, verbose: bool) -> None:
        events.append((strict, verbose))

    def fake_runner() -> None:
        preflight.check_fake_cli_timing(strict=True, verbose=False)

    monkeypatch.setattr(
        preflight,
        "check_fake_cli_timing",
        check_fake_cli_timing,
        raising=False,
    )
    monkeypatch.setattr(
        preflight.inspect,
        "getsource",
        lambda func: (
            "def fake_runner():\n"
            "    check_fake_cli_timing(strict=True, verbose=False)\n"
        ),
    )

    with _PreflightCliTimingRecorder(scope="dev", runner=fake_runner) as recorder:
        fake_runner()

    rows = recorder.rows()
    assert events == [(True, False)]
    assert len(rows) == 1
    assert rows[0]["name"] == "check_fake_cli_timing"
    assert rows[0]["status"] == "passed"
    assert rows[0]["elapsed_s"] >= 0
    assert rows[0]["sequence"] == 1


def test_cli_timing_payload_is_hot_sorted_and_records_budget() -> None:
    rows = [
        {"sequence": 1, "name": "fast", "status": "passed", "elapsed_s": 0.1},
        {"sequence": 2, "name": "slow", "status": "passed", "elapsed_s": 0.75},
    ]

    payload = _build_preflight_cli_timing_payload(
        scope="dev",
        status="passed",
        wall_elapsed_s=0.9,
        timeout_s=30.0,
        allow_slow_preflight=False,
        check_codebase=True,
        step_rows=rows,
    )

    assert payload["schema"] == PREFLIGHT_CLI_TIMING_SCHEMA
    assert payload["scope"] == "dev"
    assert payload["timeout_s"] == 30.0
    assert payload["wall_elapsed_s"] == 0.9
    assert payload["serial_elapsed_s"] == 0.85
    assert payload["slow_step_count"] == 1
    assert [row["name"] for row in payload["steps"]] == ["fast", "slow"]
    assert [row["name"] for row in payload["hot_steps"]] == ["slow", "fast"]


def test_write_cli_timing_json_creates_parent(tmp_path) -> None:
    path = tmp_path / "profiles" / "preflight.json"
    payload = {"schema": PREFLIGHT_CLI_TIMING_SCHEMA, "steps": []}

    _write_preflight_cli_timing_json(path, payload)

    assert json.loads(path.read_text()) == payload
    assert path.read_text().endswith("\n")
