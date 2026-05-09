from __future__ import annotations

import time

import pytest

from tac.preflight import (
    DEFAULT_PREFLIGHT_CLI_TIMEOUT_S,
    PreflightTimeoutError,
    _preflight_cli_timeout_seconds,
    _preflight_timeout_after,
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
