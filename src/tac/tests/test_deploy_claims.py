from __future__ import annotations

from pathlib import Path

import pytest

from tac.deploy.claims import (
    DispatchClaimSpec,
    dispatch_claim_command,
    terminal_dispatch_claim,
)


def test_dispatch_claim_command_is_provider_neutral() -> None:
    cmd = dispatch_claim_command(
        spec=DispatchClaimSpec(
            lane_id="lane_unit",
            platform="kaggle",
            instance_job_id="kaggle:unit/kernel",
            agent="codex:test",
            predicted_eta_utc="2026-05-11T00:00:00Z",
            notes="proxy only",
        ),
        status="active_proxy_dispatch",
        python_executable="python",
        claim_tool=Path("tools/claim_lane_dispatch.py"),
    )

    assert cmd[:3] == ["python", "tools/claim_lane_dispatch.py", "claim"]
    assert cmd[cmd.index("--lane-id") + 1] == "lane_unit"
    assert cmd[cmd.index("--platform") + 1] == "kaggle"
    assert cmd[cmd.index("--instance-job-id") + 1] == "kaggle:unit/kernel"
    assert cmd[cmd.index("--status") + 1] == "active_proxy_dispatch"
    assert "--force" not in cmd


def test_dispatch_claim_command_force_and_validation() -> None:
    cmd = dispatch_claim_command(
        spec=DispatchClaimSpec(
            lane_id="lane_unit",
            platform="vastai",
            instance_job_id="123",
            agent="codex:test",
            force=True,
        ),
        status="failed_cuda_probe",
    )

    assert "--force" in cmd
    with pytest.raises(ValueError, match="lane_id"):
        dispatch_claim_command(
            spec=DispatchClaimSpec(
                lane_id="",
                platform="vastai",
                instance_job_id="123",
                agent="codex:test",
            ),
            status="active_dispatching",
        )


def test_terminal_dispatch_claim_uses_force_and_terminal_notes(monkeypatch, tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    def fake_record_dispatch_claim(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr("tac.deploy.claims.record_dispatch_claim", fake_record_dispatch_claim)

    terminal_dispatch_claim(
        repo_root=tmp_path,
        spec=DispatchClaimSpec(
            lane_id="lane_unit",
            platform="modal",
            instance_job_id="fc-123",
            agent="codex:test",
        ),
        status="completed_modal_auth_eval_recovered",
        notes="result recovered",
    )

    assert len(captured) == 1
    spec = captured[0]["spec"]
    assert isinstance(spec, DispatchClaimSpec)
    assert spec.force is True
    assert spec.notes == "result recovered"
    assert captured[0]["status"] == "completed_modal_auth_eval_recovered"
