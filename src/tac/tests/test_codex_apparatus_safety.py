"""Negate-tested apparatus guards for delegation sandbox/cap and drain liveness."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from tac import confound_gates as cg
from tac.preflight import PreflightError

_TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import codex_delegate  # noqa: E402
import codex_drain_detector as drain  # noqa: E402
import codex_status  # noqa: E402


def _policy(**overrides):
    kwargs = {
        "label": "arm",
        "requested_sandbox": "danger-full-access",
        "isolate": True,
        "prior_sandboxes": [],
        "live_nonisolated_writers": 0,
        "nonisolated_writer_cap": 1,
    }
    kwargs.update(overrides)
    return codex_delegate._launch_policy_refusal(**kwargs)


# Bug 1: retry/relaunch sandbox custody.
def test_retry_refuses_full_to_workspace_downgrade():
    refusal = _policy(
        requested_sandbox="workspace-write", prior_sandboxes=["danger-full-access"]
    )
    assert refusal and refusal[0] == 5 and "sandbox downgrade" in refusal[1]


def test_retry_refuses_full_to_readonly_downgrade():
    assert _policy(
        requested_sandbox="read-only", prior_sandboxes=["danger-full-access"]
    )[0] == 5


def test_retry_allows_same_sandbox():
    assert _policy(prior_sandboxes=["danger-full-access"]) is None


def test_retry_allows_authority_upgrade():
    assert _policy(prior_sandboxes=["workspace-write"]) is None


def test_retry_uses_strongest_prior_authority_not_latest_weaker_row():
    assert _policy(
        requested_sandbox="workspace-write",
        prior_sandboxes=["danger-full-access", "workspace-write"],
    )[0] == 5


def test_generated_transient_retry_uses_one_immutable_sandbox_variable(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_delegate, "RUNS", tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("work", encoding="utf-8")
    body = codex_delegate._write_launcher(
        "arm", "stamp", prompt, "gpt-5.6-sol", "high", "danger-full-access",
        tmp_path / "run.log", tmp_path / "last.txt", tmp_path / "done", False, tmp_path,
    ).read_text(encoding="utf-8")
    assert "ORIGINAL_SANDBOX=danger-full-access" in body
    assert body.count('--sandbox "$ORIGINAL_SANDBOX"') == 2
    assert "--sandbox workspace-write" not in body


def test_force_cannot_bypass_sandbox_downgrade(monkeypatch, tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("work", encoding="utf-8")
    monkeypatch.setattr(codex_delegate, "_prior_sandboxes", lambda label: ["danger-full-access"])
    monkeypatch.setattr(codex_delegate, "_live_nonisolated_writer_count", lambda: 0)
    rc = codex_delegate.main([
        "--label", "arm", "--prompt-file", str(prompt), "--sandbox", "workspace-write",
        "--no-launch", "--force",
    ])
    assert rc == 5
    assert "--force cannot bypass" in capsys.readouterr().out


# Bug 2: shared-tree writer cap.
@pytest.mark.parametrize("sandbox", ["workspace-write", "danger-full-access"])
def test_nonisolated_writer_at_cap_is_refused(sandbox):
    refusal = _policy(
        requested_sandbox=sandbox, isolate=False, live_nonisolated_writers=1
    )
    assert refusal and refusal[0] == 6 and "cap=1" in refusal[1]


def test_isolated_writer_is_exempt_from_shared_tree_cap():
    assert _policy(isolate=True, live_nonisolated_writers=99) is None


def test_nonisolated_readonly_arm_is_exempt_from_writer_cap():
    assert _policy(
        requested_sandbox="read-only", isolate=False, live_nonisolated_writers=99
    ) is None


def test_first_nonisolated_writer_is_allowed_below_cap():
    assert _policy(
        requested_sandbox="workspace-write", isolate=False, live_nonisolated_writers=0
    ) is None


def test_isolation_setup_failure_has_no_shared_tree_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_delegate, "RUNS", tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("work", encoding="utf-8")
    body = codex_delegate._write_launcher(
        "arm", "stamp", prompt, "gpt-5.6-sol", "high", "danger-full-access",
        tmp_path / "run.log", tmp_path / "last.txt", tmp_path / "done", True,
        tmp_path / "worktree",
    ).read_text(encoding="utf-8")
    assert "no shared-tree writer fallback" in body
    assert "shared-tree fallback (NO isolation)" not in body
    assert "rc=$RC" in body and "RC=12" in body


def test_force_cannot_bypass_nonisolated_writer_cap(monkeypatch, tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("work", encoding="utf-8")
    monkeypatch.setattr(codex_delegate, "_prior_sandboxes", lambda label: [])
    monkeypatch.setattr(codex_delegate, "_live_nonisolated_writer_count", lambda: 1)
    rc = codex_delegate.main([
        "--label", "arm", "--prompt-file", str(prompt), "--sandbox", "workspace-write",
        "--no-isolate", "--no-launch", "--force",
    ])
    assert rc == 6
    assert "--force cannot bypass" in capsys.readouterr().out


def test_live_nonisolated_writer_count_exempts_isolated_and_readonly(monkeypatch):
    rows = [
        {"label": "writer", "stamp": "1", "sandbox": "workspace-write", "isolate": False},
        {"label": "isolated", "stamp": "2", "sandbox": "danger-full-access", "isolate": True},
        {"label": "reader", "stamp": "3", "sandbox": "read-only", "isolate": False},
    ]
    monkeypatch.setattr(codex_delegate, "_ledger_rows", lambda: rows)

    class Result:
        returncode = 0
        stdout = "123\n"

    monkeypatch.setattr(codex_delegate.subprocess, "run", lambda *a, **k: Result())
    assert codex_delegate._live_nonisolated_writer_count() == 1


def test_pre_cfl_live_writer_is_retroactively_strand_doomed():
    legacy = {"sandbox": "workspace-write"}  # pre-CFL rows have no isolate field
    assert codex_status._is_strand_doomed(legacy) is True
    assert codex_status._bucket(
        True, None, None, 0.1, strand_doomed=codex_status._is_strand_doomed(legacy)
    ) == "STRAND_DOOMED"


def test_isolated_or_readonly_live_arm_is_not_strand_doomed():
    assert codex_status._is_strand_doomed(
        {"sandbox": "workspace-write", "isolate": True}
    ) is False
    assert codex_status._is_strand_doomed(
        {"sandbox": "read-only", "isolate": False}
    ) is False


# Bug 3: timeout classification uses activity, not elapsed wall clock alone.
def _obs(*, mtime: float, cursor: int) -> dict[str, dict]:
    return {"arm_s": {"label": "arm", "stamp": "s", "log_mtime": mtime,
                      "progress_cursor": cursor}}


def test_healthy_recent_log_past_timeout_is_not_wedged():
    status, details = drain.classify_timeout(
        _obs(mtime=100, cursor=7), _obs(mtime=995, cursor=7),
        now=1000, liveness_window_seconds=60,
    )
    assert status == drain.HEALTHY_BUT_SLOW
    assert details[0]["log_recent"] is True


def test_healthy_advancing_progress_past_timeout_is_not_wedged():
    status, details = drain.classify_timeout(
        _obs(mtime=100, cursor=7), _obs(mtime=100, cursor=8),
        now=1000, liveness_window_seconds=60,
    )
    assert status == drain.HEALTHY_BUT_SLOW
    assert details[0]["progress_advanced"] is True


def test_genuinely_stale_arm_is_wedged():
    status, details = drain.classify_timeout(
        _obs(mtime=100, cursor=7), _obs(mtime=100, cursor=7),
        now=1000, liveness_window_seconds=60,
    )
    assert status == drain.WEDGED and details[0]["health"] == "wedged"


def test_mixed_fleet_alarms_when_any_arm_is_wedged():
    baseline = {
        **_obs(mtime=100, cursor=7),
        "other_s": {"label": "other", "stamp": "s", "log_mtime": 100, "progress_cursor": 2},
    }
    current = {
        **_obs(mtime=995, cursor=7),
        "other_s": {"label": "other", "stamp": "s", "log_mtime": 100, "progress_cursor": 2},
    }
    assert drain.classify_timeout(
        baseline, current, now=1000, liveness_window_seconds=60
    )[0] == drain.WEDGED


def test_empty_fleet_is_drained():
    assert drain.classify_timeout({}, {}, now=1000)[0] == drain.DRAINED


def test_timeout_and_wedged_are_nonzero_but_drained_is_success():
    assert drain.exit_code_for_status(drain.DRAINED) == 0
    assert drain.exit_code_for_status(drain.TIMED_OUT) == 2
    assert drain.exit_code_for_status(drain.WEDGED) == 3


def test_progress_cursor_reads_nested_index(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"progress": {"index": 41}}), encoding="utf-8")
    assert drain._cursor_from_json(path) == 41


def test_cli_healthy_but_slow_returns_timeout_nonzero(monkeypatch, tmp_path, capsys):
    log = tmp_path / "arm.log"
    log.write_text("still working", encoding="utf-8")
    monkeypatch.setattr(
        drain.codex_status, "status_rows",
        lambda: [{"label": "arm", "stamp": "s", "status": "RUNNING", "log": str(log)}],
    )
    assert drain.main(["--timeout-seconds", "0", "--liveness-window-seconds", "60"]) == 2
    assert "TIMEOUT" in capsys.readouterr().out


def test_strand_doomed_arm_is_wedged_even_with_recent_log(monkeypatch, tmp_path, capsys):
    log = tmp_path / "arm.log"
    log.write_text("still writing shared tree", encoding="utf-8")
    monkeypatch.setattr(
        drain.codex_status,
        "status_rows",
        lambda: [
            {"label": "arm", "stamp": "s", "status": "STRAND_DOOMED", "log": str(log)}
        ],
    )
    assert drain.main(["--timeout-seconds", "0", "--liveness-window-seconds", "60"]) == 3
    assert "WEDGED" in capsys.readouterr().out


def test_cli_stale_arm_returns_three(monkeypatch, tmp_path, capsys):
    log = tmp_path / "arm.log"
    log.write_text("stale", encoding="utf-8")
    old = time.time() - 3600
    log.touch()
    import os
    os.utime(log, (old, old))
    monkeypatch.setattr(
        drain.codex_status, "status_rows",
        lambda: [{"label": "arm", "stamp": "s", "status": "RUNNING", "log": str(log)}],
    )
    assert drain.main(["--timeout-seconds", "0", "--liveness-window-seconds", "60"]) == 3
    assert "WEDGED" in capsys.readouterr().out


# STRICT behavior probes: each negate fixture restores the pre-fix anti-pattern
# and must be refused. These are the second landing, not tests of constants.
def _delegate_fixture(root: Path, policy_body: str) -> None:
    path = root / "tools" / "codex_delegate.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "_MAX_NONISOLATED_WRITERS = 1\n"
        "def _launch_policy_refusal(**kwargs):\n"
        f"    {policy_body}\n"
        "def main():\n"
        "    return _launch_policy_refusal()\n",
        encoding="utf-8",
    )


def test_strict_retry_gate_catches_behaviorally_broken_policy(tmp_path):
    _delegate_fixture(tmp_path, "return None")
    with pytest.raises(PreflightError, match="sandbox"):
        cg.check_codex_retry_preserves_original_sandbox_authority(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_strict_retry_gate_catches_missing_checkpoint_custody(tmp_path):
    path = tmp_path / "tools" / "codex_delegate.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "_MAX_CAPACITY_RETRIES=8\n"
        "def _launch_policy_refusal(**kwargs): return None\n"
        "def main(): return _launch_policy_refusal()\n",
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match=r"retry|checkpoint"):
        cg.check_codex_retry_preserves_original_sandbox_authority(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_strict_writer_cap_gate_catches_behaviorally_broken_policy(tmp_path):
    _delegate_fixture(tmp_path, "return None")
    with pytest.raises(PreflightError, match="writer"):
        cg.check_codex_nonisolated_writer_cap(repo_root=tmp_path, strict=True, verbose=False)


def test_strict_writer_cap_gate_catches_missing_pre_cfl_retrofit(tmp_path):
    _delegate_fixture(
        tmp_path,
        "return (6, 'writer refused') if not kwargs.get('isolate', True) "
        "and kwargs.get('requested_sandbox') != 'read-only' else None",
    )
    with pytest.raises(PreflightError, match=r"codex_status|strand"):
        cg.check_codex_nonisolated_writer_cap(repo_root=tmp_path, strict=True, verbose=False)


def test_strict_drain_gate_catches_wallclock_only_classifier(tmp_path):
    path = tmp_path / "tools" / "codex_drain_detector.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "HEALTHY_BUT_SLOW='HEALTHY_BUT_SLOW'\nWEDGED='WEDGED'\n"
        "def classify_timeout(*args, **kwargs): return WEDGED, []\n"
        "def main(): return classify_timeout({}, {}, now=0)\n",
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="recent-log"):
        cg.check_codex_drain_timeout_uses_liveness(
            repo_root=tmp_path, strict=True, verbose=False
        )


@pytest.mark.parametrize(
    "gate",
    [
        cg.check_codex_retry_preserves_original_sandbox_authority,
        cg.check_codex_nonisolated_writer_cap,
        cg.check_codex_drain_timeout_uses_liveness,
    ],
)
def test_apparatus_strict_gate_passes_live_repo(gate):
    assert gate(strict=True, verbose=False) == []
