from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HELPER_PATH = Path(__file__).resolve().parents[3] / "tools" / "export_active_lane_claim_json.py"
spec = importlib.util.spec_from_file_location("export_active_lane_claim_json", _HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
sys.modules["export_active_lane_claim_json"] = helper
spec.loader.exec_module(helper)


def test_export_active_lane_claim_json_selects_newest_nonterminal(tmp_path: Path) -> None:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                "| 2026-05-08T00:05:00Z | a | L | lightning | j2 |  | eval | latest |",
                "| 2026-05-08T00:04:00Z | a | L | lightning | j1 |  | completed_score_0.1 | done |",
                "| 2026-05-08T00:03:00Z | a | L | lightning | j1 |  | eval | stale by terminal |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = helper.build_active_lane_claim_json(
        claims_path=claims,
        lane_id="L",
        now_utc="2026-05-08T00:06:00Z",
    )

    assert payload["schema"] == "tac_active_lane_claim_json_v1"
    assert payload["active"] is True
    assert payload["instance_job_id"] == "j2"
    assert payload["claim_status"] == "eval"
    assert len(payload["claim_row_sha256"]) == 64
    assert payload["blockers"] == []


def test_export_active_lane_claim_json_fails_closed_when_missing(tmp_path: Path) -> None:
    payload = helper.build_active_lane_claim_json(
        claims_path=tmp_path / "missing.md",
        lane_id="absent",
        now_utc="2026-05-08T00:06:00Z",
    )

    assert payload["active"] is False
    assert payload["blockers"] == ["active_lane_claim_not_found"]
