# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess

import tools.preflight_hook as preflight_hook


def test_preflight_hook_defaults_to_no_codebase(monkeypatch) -> None:
    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)

    assert preflight_hook._preflight_command() == [
        ".venv/bin/python",
        "-m",
        "tac.preflight",
        "--no-codebase",
    ]
    assert preflight_hook._preflight_timeout_seconds() == 30


def test_preflight_hook_full_mode_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("PREFLIGHT_FULL", "1")
    monkeypatch.delenv("PREFLIGHT_ALLOW_SLOW", raising=False)

    assert preflight_hook._preflight_command() == [
        ".venv/bin/python",
        "-m",
        "tac.preflight",
        "--scope",
        "all",
    ]
    assert preflight_hook._preflight_timeout_seconds() == 30


def test_preflight_hook_slow_release_mode_requires_separate_env(monkeypatch) -> None:
    monkeypatch.setenv("PREFLIGHT_FULL", "1")
    monkeypatch.setenv("PREFLIGHT_ALLOW_SLOW", "1")

    assert preflight_hook._preflight_command() == [
        ".venv/bin/python",
        "-m",
        "tac.preflight",
        "--scope",
        "all",
        "--allow-slow-preflight",
    ]
    assert preflight_hook._preflight_timeout_seconds() == 600


def test_preflight_hook_timeout_env_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("PREFLIGHT_TIMEOUT_SECONDS", "12")
    assert preflight_hook._preflight_timeout_seconds() == 12

    monkeypatch.setenv("PREFLIGHT_TIMEOUT_SECONDS", "not-an-int")
    assert preflight_hook._preflight_timeout_seconds() == 30


def test_run_preflight_reports_timeout(monkeypatch, capsys) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or args[0],
            timeout=kwargs["timeout"],
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.delenv("PREFLIGHT_FULL", raising=False)
    monkeypatch.setattr(preflight_hook.subprocess, "run", fake_run)

    assert preflight_hook.run_preflight() == 1
    captured = capsys.readouterr()
    assert "preflight timed out" in captured.err
    assert "tac.preflight --no-codebase" in captured.err
