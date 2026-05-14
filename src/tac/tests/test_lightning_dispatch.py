# SPDX-License-Identifier: MIT
"""Smoke tests for tac.deploy.lightning.LightningDispatcher + scripts/launch_lane_lightning.py.

We do NOT invoke real SSH in these tests. We verify:
  * The module imports cleanly.
  * GPU-tier normalization handles all common Lightning device names.
  * Dispatch builds correct tmux + ssh commands.
  * State file round-trips cleanly.
  * Launcher CLI subcommands are wired and the parser rejects missing args
    (e.g., --predicted-band is required per CLAUDE.md).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = REPO_ROOT / "scripts" / "launch_lane_lightning.py"


def _load_launcher_module():
    spec = importlib.util.spec_from_file_location(
        "launch_lane_lightning_mod", str(LAUNCHER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["launch_lane_lightning_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dispatcher_module_imports_clean():
    from tac.deploy.lightning import LightningDispatcher, DispatchResult  # noqa: F401


def test_dispatcher_gpu_tier_normalize():
    from tac.deploy.lightning import LightningDispatcher

    cases = [
        ("Tesla T4", "T4"),
        ("NVIDIA A100-SXM4-40GB", "A100"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA L40S", "L40S"),
        ("NVIDIA L4", "L4"),
        ("NVIDIA A10G", "A10G"),
        ("Tesla V100-SXM2-16GB", "V100"),
        ("Some-future-card", "Some-future-card"),  # passthrough
    ]
    for name, expected in cases:
        assert LightningDispatcher._gpu_tier_normalize(name) == expected, (
            f"normalize({name!r}) != {expected!r}"
        )


def test_dispatcher_state_round_trip(tmp_path, monkeypatch):
    """The state file is persisted and read back correctly."""
    import tac.deploy.lightning.lightning_dispatch as mod
    from tac.deploy.lightning import LightningDispatcher

    state_path = tmp_path / "lightning_active_sessions.json"
    monkeypatch.setattr(mod, "LIGHTNING_STATE", state_path)

    LightningDispatcher.register_session(
        {"session_id": "abc", "label": "test", "status": "running"}
    )
    LightningDispatcher.register_session(
        {"session_id": "def", "label": "test2", "status": "running"}
    )

    assert state_path.exists()
    sessions = LightningDispatcher.list_sessions()
    assert len(sessions) == 2
    assert {s["session_id"] for s in sessions} == {"abc", "def"}

    removed = LightningDispatcher.remove_session("abc")
    assert removed is True

    sessions_after = LightningDispatcher.list_sessions()
    assert len(sessions_after) == 1
    assert sessions_after[0]["session_id"] == "def"


def test_dispatcher_state_round_trip_missing_file(tmp_path, monkeypatch):
    """Loading a non-existent state file returns []."""
    import tac.deploy.lightning.lightning_dispatch as mod
    from tac.deploy.lightning import LightningDispatcher

    state_path = tmp_path / "missing.json"
    monkeypatch.setattr(mod, "LIGHTNING_STATE", state_path)
    assert LightningDispatcher.list_sessions() == []


def test_dispatcher_ssh_args_no_key():
    from tac.deploy.lightning import LightningDispatcher

    d = LightningDispatcher(
        ssh_user="user_x", ssh_host="host_y", ssh_key=None
    )
    args = d._ssh_args()
    assert args[0] == "ssh"
    assert "user_x@host_y" in args
    assert "-i" not in args
    assert "BatchMode=yes" in args
    assert "PasswordAuthentication=no" in args
    assert "KbdInteractiveAuthentication=no" in args
    assert "ServerAliveInterval=15" in args
    assert "ServerAliveCountMax=4" in args
    assert "ConnectionAttempts=3" in args
    assert "TCPKeepAlive=yes" in args


def test_dispatcher_ssh_args_with_alias_target():
    from tac.deploy.lightning import LightningDispatcher

    d = LightningDispatcher(ssh_target="lightning-pact", ssh_key=None)
    args = d._ssh_args()

    assert args[0] == "ssh"
    assert "lightning-pact" in args
    assert not any(item.endswith("@ssh.lightning.ai") for item in args)


def test_dispatcher_rejects_bare_lightning_host():
    from tac.deploy.lightning import LightningDispatcher

    with pytest.raises(ValueError, match="bare ssh.lightning.ai"):
        LightningDispatcher(ssh_target="ssh.lightning.ai")


def test_dispatcher_ssh_args_with_key():
    from tac.deploy.lightning import LightningDispatcher

    d = LightningDispatcher(
        ssh_user="user_x", ssh_host="host_y", ssh_key="/path/to/key"
    )
    args = d._ssh_args()
    idx = args.index("-i")
    assert args[idx + 1] == "/path/to/key"


def test_dispatcher_scp_args():
    from tac.deploy.lightning import LightningDispatcher

    d = LightningDispatcher(
        ssh_user="user_x", ssh_host="host_y", ssh_key=None
    )
    args = d._scp_args("/local/path", "user_x@host_y:/remote/path")
    assert args[0] == "scp"
    assert "-r" in args
    assert "/local/path" in args
    assert "user_x@host_y:/remote/path" in args
    assert "BatchMode=yes" in args
    assert "ServerAliveInterval=15" in args
    assert "ServerAliveCountMax=4" in args
    assert "ConnectionAttempts=3" in args


def test_harvest_prefers_contest_auth_eval_json_over_log_regex(tmp_path, monkeypatch):
    import tac.deploy.lightning.lightning_dispatch as mod
    from tac.deploy.lightning import LightningDispatcher

    state_path = tmp_path / "lightning_active_sessions.json"
    monkeypatch.setattr(mod, "LIGHTNING_STATE", state_path)
    LightningDispatcher.register_session(
        {
            "session_id": "sess",
            "label": "lane_x",
            "remote_workspace": "/remote/pact",
            "remote_log_path": "/remote/pact/lane_x.log",
        }
    )

    dispatcher = LightningDispatcher(ssh_user="u", ssh_host="h")
    monkeypatch.setattr(dispatcher, "_run_ssh", lambda *a, **k: (0, "OK", ""))

    def fake_run(args, capture_output, text, timeout):
        copied = tmp_path / "local" / "remote_results"
        copied.mkdir(parents=True)
        (copied / "auth_eval.log").write_text(
            'RESULT_JSON: {"score_recomputed_from_components": 99.0}\n'
        )
        (copied / "contest_auth_eval.json").write_text(
            json.dumps(
                {
                    "final_score": 1.24,
                    "score_recomputed_from_components": 1.2345,
                    "n_samples": 600,
                }
            )
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = dispatcher.harvest(
        "sess",
        local_dir=tmp_path / "local",
        remote_subdir="remote_results",
    )

    assert result["auth_eval_score"] == 1.2345
    assert result["contest_auth_eval_json"] == "remote_results/contest_auth_eval.json"
    assert result["contest_auth_eval"]["n_samples"] == 600


# -----------------------------------------------------------------------------
# Launcher CLI tests (via subprocess so the parser is exercised end-to-end).
# -----------------------------------------------------------------------------


def test_launcher_cli_help():
    """The launcher --help should exit 0 and list subcommands."""
    result = subprocess.run(
        [sys.executable, str(LAUNCHER_PATH), "--help"],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "dispatch" in result.stdout
    assert "harvest" in result.stdout
    assert "status" in result.stdout
    assert "teardown" in result.stdout


def test_launcher_cli_dispatch_requires_predicted_band():
    """CLAUDE.md non-negotiable: every dispatch MUST document predicted band.

    The launcher MUST refuse to dispatch without --predicted-band.
    """
    result = subprocess.run(
        [
            sys.executable, str(LAUNCHER_PATH), "dispatch",
            "--lane-script", "scripts/remote_lane_x.sh",
            "--label", "foo",
        ],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "predicted-band" in combined.lower()


def test_launcher_cli_unknown_subcommand():
    result = subprocess.run(
        [sys.executable, str(LAUNCHER_PATH), "nonexistent-cmd"],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode != 0


def test_launcher_module_loads():
    mod = _load_launcher_module()
    assert hasattr(mod, "build_parser")
    assert hasattr(mod, "cmd_dispatch")
    assert hasattr(mod, "cmd_status")
    assert hasattr(mod, "cmd_harvest")
    assert hasattr(mod, "cmd_teardown")
    assert hasattr(mod, "cmd_list")
    assert hasattr(mod, "cmd_probe")


def test_launcher_env_kv_parsing():
    mod = _load_launcher_module()
    parsed = mod._parse_env_kv(["A=1", "B=hello world", "C=path/with/slashes"])
    assert parsed == {"A": "1", "B": "hello world", "C": "path/with/slashes"}


def test_launcher_env_kv_rejects_bare_key():
    mod = _load_launcher_module()
    with pytest.raises(SystemExit):
        mod._parse_env_kv(["BAD_NO_EQUALS"])
