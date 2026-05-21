from __future__ import annotations

import json
from pathlib import Path

import tools.reconcile_dispatch_claims_with_modal_ledger as reconciler


def _write_claims(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Active lane dispatch claims -- test",
                "",
                "## Claims (newest first)",
                "",
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                "| 2026-05-21T18:11:26Z | claude | lane_eval | modal | eval_label |  | active_modal_auth_eval_spawned | call_id=fc-TERM123 |",
                "| 2026-05-21T18:12:00Z | claude | lane_label | modal | label_job |  | active_dispatch | no explicit call id |",
                "| 2026-05-21T18:13:00Z | claude | lane_live | modal | live_job |  | active_dispatch | call_id=fc-LIVE123 |",
                "| 2026-05-21T18:14:00Z | claude | lane_manual | modal | pending-spawn |  | active_dispatch | no modal evidence |",
                "| 2026-05-21T18:15:00Z | claude | lane_closed | modal | closed_job |  | completed_old | already terminal |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "call_id": "fc-TERM123",
            "status": "harvested",
            "event_type": "harvested",
            "rc": 0,
            "evidence_grade": "contest-CUDA",
            "score_axis": "contest_cuda",
            "score_recomputed_from_components": 0.2,
        },
        {
            "call_id": "fc-LABEL123",
            "label": "label_job",
            "lane_id": "lane_label",
            "status": "dispatched",
            "event_type": "dispatched",
        },
        {
            "call_id": "fc-LABEL123",
            "status": "failed",
            "event_type": "failed",
            "rc": 1,
            "failure_class": "unit_test_failure",
        },
        {
            "call_id": "fc-LIVE123",
            "status": "dispatched",
            "event_type": "dispatched",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_reconcile_high_confidence_terminal_and_live_rows(tmp_path: Path) -> None:
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    ledger = tmp_path / ".omx/state/modal_call_id_ledger.jsonl"
    _write_claims(claims)
    _write_ledger(ledger)

    rows = reconciler.reconcile(
        repo_root=tmp_path,
        claims_path=claims,
        modal_ledger=ledger,
    )

    by_job = {row.claim.instance_job_id: row for row in rows}
    assert by_job["eval_label"].confidence == "high"
    assert by_job["eval_label"].suggested_status == "completed_modal_auth_eval_recovered"
    assert by_job["label_job"].confidence == "high"
    assert by_job["label_job"].call_id == "fc-LABEL123"
    assert by_job["label_job"].suggested_status == "failed_modal_call_recovered_rc_1"
    assert by_job["live_job"].confidence == "active_or_unharvested"
    assert by_job["pending-spawn"].confidence == "manual_review"
    assert "closed_job" not in by_job
