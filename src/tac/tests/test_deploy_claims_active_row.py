from __future__ import annotations

from pathlib import Path

import pytest

from tac.deploy.claims import active_claim_row, is_terminal_status


def test_active_claim_row_returns_newest_nonterminal_match(tmp_path: Path) -> None:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| 2026-05-11T02:00:00Z | codex | lane_x | kaggle | job_1 | 2026-05-11T03:00:00Z | active_dispatching | newest |\n"
        "| 2026-05-11T01:00:00Z | codex | lane_x | kaggle | job_1 | 2026-05-11T02:00:00Z | active_dispatching | older |\n",
        encoding="utf-8",
    )

    row = active_claim_row(claims, lane_id="lane_x", instance_job_id="job_1")

    assert row["timestamp_utc"] == "2026-05-11T02:00:00Z"
    assert row["notes"] == "newest"


def test_active_claim_row_rejects_newest_terminal_match(tmp_path: Path) -> None:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| 2026-05-11T02:00:00Z | codex | lane_x | kaggle | job_1 | 2026-05-11T03:00:00Z | completed_success | done |\n"
        "| 2026-05-11T01:00:00Z | codex | lane_x | kaggle | job_1 | 2026-05-11T02:00:00Z | active_dispatching | older |\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="terminal"):
        active_claim_row(claims, lane_id="lane_x", instance_job_id="job_1")


def test_is_terminal_status_covers_dispatch_close_prefixes() -> None:
    assert is_terminal_status("failed_stage")
    assert is_terminal_status("stopped_operator")
    assert is_terminal_status("refused_dispatch_duplicate")
    assert not is_terminal_status("active_dispatching")
