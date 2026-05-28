# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.optimization.proxy_candidate_contract import truthy_authority_field_violations
from tools import check_exact_dispatch_provider_preclaim as preclaim


def _clear_lightning_env(monkeypatch) -> None:
    for key in (
        "LIGHTNING_STUDIO",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_SDK_USER",
        "LIGHTNING_ORG",
        "LIGHTNING_SSH_TARGET",
    ):
        monkeypatch.delenv(key, raising=False)


def test_lightning_preclaim_blocks_missing_route_before_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_lightning_env(monkeypatch)
    out = tmp_path / "preclaim.json"

    rc = preclaim.main(
        [
            "--provider",
            "lightning",
            "--job-id",
            "fixture_exact_job",
            "--output",
            str(out),
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert payload["schema"] == preclaim.SCHEMA
    assert payload["preclaim_ready"] is False
    assert payload["blockers"] == [
        "lightning_studio_missing",
        "lightning_teamspace_missing",
        "lightning_owner_missing",
        "lightning_ssh_target_missing",
    ]
    assert truthy_authority_field_violations(payload) == []


def test_lightning_preclaim_writes_redacted_ready_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_lightning_env(monkeypatch)
    monkeypatch.setenv("LIGHTNING_STUDIO", "private-studio")
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "private-teamspace")
    monkeypatch.setenv("LIGHTNING_SDK_USER", "user@example.test")
    monkeypatch.setenv("LIGHTNING_SSH_TARGET", "private-ssh-target")
    out = tmp_path / "preclaim.json"

    rc = preclaim.main(
        [
            "--provider",
            "lightning",
            "--job-id",
            "fixture_exact_job",
            "--output",
            str(out),
        ]
    )

    text = out.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert rc == 0
    assert payload["preclaim_ready"] is True
    assert payload["blockers"] == []
    assert payload["env_status"]["LIGHTNING_STUDIO"] == "present"
    assert payload["env_status"]["LIGHTNING_ORG"] == "missing"
    assert "private-studio" not in text
    assert "private-teamspace" not in text
    assert "user@example.test" not in text
    assert "private-ssh-target" not in text
    assert truthy_authority_field_violations(payload) == []
