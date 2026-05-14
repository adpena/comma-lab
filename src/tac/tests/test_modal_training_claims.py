# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import shutil
from pathlib import Path

from tac.deploy.modal.training_claims import (
    append_modal_training_terminal_claim,
    modal_training_terminal_status,
)


def test_modal_training_terminal_status_is_score_claim_safe() -> None:
    assert (
        modal_training_terminal_status({"returncode": 0, "timed_out": False})
        == "completed_modal_training_recovered_no_score_claim"
    )
    assert (
        modal_training_terminal_status({"returncode": 7, "timed_out": False})
        == "failed_modal_training_rc_7"
    )
    assert (
        modal_training_terminal_status({"returncode": 0, "timed_out": True})
        == "failed_modal_training_timeout"
    )


def test_modal_training_terminal_status_accepts_legacy_rc_key() -> None:
    assert (
        modal_training_terminal_status({"rc": 1, "timed_out": False})
        == "failed_modal_training_rc_1"
    )


def test_append_modal_training_terminal_claim_idempotent(tmp_path: Path) -> None:
    repo = tmp_path
    repo_tools = repo / "tools"
    repo_tools.mkdir()
    shutil.copy(
        Path(__file__).resolve().parents[3] / "tools" / "claim_lane_dispatch.py",
        repo_tools / "claim_lane_dispatch.py",
    )
    claims = repo / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "# Active lane dispatch claims — cross-agent coordination ledger\n\n"
        "## Claims (newest first)\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )
    out_dir = repo / "experiments/results/lane_unit_modal"
    metadata = {
        "lane_id": "unit_lane",
        "label": "unit_job",
        "platform": "modal",
    }
    result = {
        "returncode": 0,
        "timed_out": False,
        "elapsed_seconds": 123.0,
    }

    first = append_modal_training_terminal_claim(
        repo_root=repo,
        out_dir=out_dir,
        metadata=metadata,
        result=result,
        agent="codex:test",
    )
    second = append_modal_training_terminal_claim(
        repo_root=repo,
        out_dir=out_dir,
        metadata=metadata,
        result=result,
        agent="codex:test",
    )

    assert first["appended"] is True
    assert first["status"] == "completed_modal_training_recovered_no_score_claim"
    assert first["score_claim"] is False
    assert first["promotion_eligible"] is False
    assert second["already_appended"] is True
    marker = json.loads((out_dir / "modal_training_terminal_claim.json").read_text())
    assert marker["lane_id"] == "unit_lane"
    ledger = claims.read_text()
    assert ledger.count("completed_modal_training_recovered_no_score_claim") == 1


def test_append_modal_training_terminal_claim_skips_without_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    manifest = append_modal_training_terminal_claim(
        repo_root=tmp_path,
        out_dir=out_dir,
        metadata={"label": "missing_lane"},
        result={"returncode": 0},
    )
    assert manifest["appended"] is False
    assert manifest["reason"] == "metadata_missing_lane_id_or_instance_job_id"
    marker = json.loads((out_dir / "modal_training_terminal_claim.json").read_text())
    assert marker["appended"] is False
    assert marker["metadata_label"] == "missing_lane"
    assert marker["score_claim"] is False
    assert marker["promotion_eligible"] is False
