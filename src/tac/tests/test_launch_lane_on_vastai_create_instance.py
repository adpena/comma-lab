from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "scripts" / "launch_lane_on_vastai.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("launch_lane_on_vastai_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vast_create_api_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()

    def fake_run(_cmd, timeout=60, capture=True):
        return 0, '{"error": true, "status_code": 400, "msg": "Your account lacks credit"}', ""

    monkeypatch.setattr(launcher, "run", fake_run)
    with pytest.raises(RuntimeError, match="API error"):
        launcher.create_instance(123, "unit-test", disk_gb=80)


def test_vast_create_empty_output_reports_unknown_state(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()

    def fake_run(_cmd, timeout=60, capture=True):
        return 0, "", ""

    monkeypatch.setattr(launcher, "run", fake_run)
    with pytest.raises(RuntimeError, match="empty stdout/stderr"):
        launcher.create_instance(123, "unit-test", disk_gb=80)


def test_vast_env_overrides_parse_shell_safe_keys() -> None:
    launcher = _load_launcher()

    parsed = launcher.parse_env_overrides(["FOO=bar", "SPACED=a b"])

    assert parsed == {"FOO": "bar", "SPACED": "a b"}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        launcher.parse_env_overrides(["NOPE"])
    with pytest.raises(ValueError, match="shell-safe"):
        launcher.parse_env_overrides(["BAD-KEY=value"])


def test_vast_lane_wrapper_exports_claim_context(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    calls: list[list[str]] = []

    def fake_run(cmd, timeout=60, capture=True):
        calls.append(cmd)
        return 0, "launched", ""

    monkeypatch.setattr(launcher, "run", fake_run)

    assert launcher.execute_lane_in_tmux(
        "host.example",
        2222,
        "scripts/remote_lane_unit.sh",
        instance_id=12345,
        env_overrides={"FOO": "bar baz"},
    )

    write_cmd = next(cmd for cmd in calls if cmd[:2] == ["bash", "-c"])
    wrapper = write_cmd[2]
    assert "export INSTANCE_JOB_ID=12345" in wrapper
    assert "export DISPATCH_INSTANCE_JOB_ID=12345" in wrapper
    assert "export DISPATCH_CLAIMS_PATH=.omx/state/active_lane_dispatch_claims.md" in wrapper
    assert "export FOO='bar baz'" in wrapper


def test_vast_claim_uses_provider_instance_as_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    calls: list[list[str]] = []

    def fake_run(cmd, timeout=60, capture=True):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(launcher, "run", fake_run)

    rc = launcher.claim_vastai_lane_dispatch(
        SimpleNamespace(label="lane_unit_test"),
        instance_id=987,
    )

    assert rc == 0
    cmd = calls[0]
    assert "tools/claim_lane_dispatch.py" in " ".join(cmd)
    assert cmd[cmd.index("--lane-id") + 1] == "lane_unit_test"
    assert cmd[cmd.index("--platform") + 1] == "vastai"
    assert cmd[cmd.index("--instance-job-id") + 1] == "987"


def test_vast_terminal_claim_uses_force_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    calls: list[list[str]] = []

    def fake_run(cmd, timeout=60, capture=True):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(launcher, "run", fake_run)

    launcher.close_vastai_lane_dispatch(
        lane_id="lane_unit_test",
        instance_id=654,
        status="failed_cuda_probe",
        notes="unit test terminal row",
    )

    cmd = calls[0]
    assert cmd[cmd.index("--lane-id") + 1] == "lane_unit_test"
    assert cmd[cmd.index("--instance-job-id") + 1] == "654"
    assert cmd[cmd.index("--status") + 1] == "failed_cuda_probe"
    assert "--force" in cmd
