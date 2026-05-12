"""Tests for tools/build_autopilot_dry_run_summary.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make tools/ importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import build_autopilot_dry_run_summary as bdrs  # noqa: E402


def _autopilot_report_payload() -> dict:
    """Synthetic autopilot report with 3 dispatch halt events."""
    return {
        "schema": "tac_cathedral_autopilot_autonomous_loop_v1",
        "iterations_run": 1,
        "operator_authorized_mode": {
            "enabled": False,
            "env_authorized": False,
        },
        "substrate_composition_ranking": {
            "n_dropped": 2,
        },
        "reports": [
            {
                "ended_at_utc": "2026-05-12T04:39:49Z",
                "halt_events": [
                    {
                        "event_class": "dispatch",
                        "candidate_id": "cand_a",
                        "estimated_cost_usd": 0.50,
                        "predicted_score_delta": -0.001,
                        "blockers": [],
                        "requires_approval": True,
                        "autopilot_authorized": False,
                        "decision": "defer",
                        "halt_at_utc": "2026-05-12T04:39:49Z",
                    },
                    {
                        "event_class": "dispatch",
                        "candidate_id": "cand_b",
                        "estimated_cost_usd": 3.00,
                        "predicted_score_delta": -0.005,
                        "blockers": ["operator_authorization_required_for_dispatch"],
                        "requires_approval": True,
                        "autopilot_authorized": False,
                        "decision": "defer",
                        "halt_at_utc": "2026-05-12T04:39:49Z",
                    },
                    {
                        "event_class": "dispatch",
                        "candidate_id": "cand_huge_cost",
                        "estimated_cost_usd": 80.00,  # exceeds $5/individual cap
                        "predicted_score_delta": -0.030,
                        "blockers": [],
                        "requires_approval": True,
                        "autopilot_authorized": False,
                        "decision": "defer",
                        "halt_at_utc": "2026-05-12T04:39:49Z",
                    },
                    {
                        "event_class": "kill",
                        "candidate_id": "cand_kill",
                        "estimated_cost_usd": 0.0,
                        "predicted_score_delta": 0.0,
                        "blockers": [],
                        "requires_approval": True,
                        "autopilot_authorized": False,
                        "decision": "defer",
                        "halt_at_utc": "2026-05-12T04:39:49Z",
                    },
                ],
            }
        ],
    }


def test_extract_only_dispatch_events():
    rows = bdrs.extract_would_dispatch_rows(_autopilot_report_payload())
    # KILL event must be excluded.
    assert len(rows) == 3
    assert all(r["candidate_id"] != "cand_kill" for r in rows)


def test_extract_sorts_by_cost_then_delta():
    rows = bdrs.extract_would_dispatch_rows(_autopilot_report_payload())
    assert rows[0]["candidate_id"] == "cand_a"  # $0.50 cheapest
    assert rows[1]["candidate_id"] == "cand_b"  # $3.00 next
    assert rows[2]["candidate_id"] == "cand_huge_cost"


def test_extract_carries_blockers_and_approval_flag():
    rows = bdrs.extract_would_dispatch_rows(_autopilot_report_payload())
    cand_b = next(r for r in rows if r["candidate_id"] == "cand_b")
    assert "operator_authorization_required_for_dispatch" in cand_b["blockers"]
    assert cand_b["requires_approval"] is True
    assert cand_b["autopilot_authorized"] is False


def test_extract_tags_each_row_predicted():
    rows = bdrs.extract_would_dispatch_rows(_autopilot_report_payload())
    for r in rows:
        assert r["evidence_tag"] == "[predicted; cathedral autopilot ranking]"
        assert r["promotion_eligible"] is False
        assert r["score_claim"] is False
        assert r["ready_for_exact_eval_dispatch"] is False


def test_cumulative_cost_band_excludes_huge_cost():
    rows = bdrs.extract_would_dispatch_rows(_autopilot_report_payload())
    cum = bdrs.cumulative_cost_band(rows)
    assert cum["total_dispatch_count"] == 3
    assert cum["total_cumulative_cost_usd_if_all_dispatched"] == 83.50
    # cand_huge_cost @ $80 exceeds $5/individual cap; should be excluded.
    assert cum["pareto_subset_within_envelope_count"] == 2
    assert cum["pareto_subset_cumulative_cost_usd"] == 3.50
    assert "cand_huge_cost" not in cum["pareto_subset_candidate_ids"]
    assert "cand_a" in cum["pareto_subset_candidate_ids"]
    assert "cand_b" in cum["pareto_subset_candidate_ids"]


def test_cumulative_cost_band_respects_cumulative_envelope():
    """Synthetic: 30 dispatches @ $1 each — should stop at $20 envelope."""
    rows = []
    for i in range(30):
        rows.append(
            {
                "candidate_id": f"cand_{i:02d}",
                "estimated_cost_usd": 1.00,
                "predicted_score_delta": -0.0001 * (i + 1),
                "blockers": [],
                "requires_approval": True,
                "autopilot_authorized": False,
                "decision": "defer",
                "halt_at_utc": "2026-05-12T04:39:49Z",
                "evidence_tag": "[predicted; cathedral autopilot ranking]",
                "promotion_eligible": False,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    cum = bdrs.cumulative_cost_band(rows)
    assert cum["total_cumulative_cost_usd_if_all_dispatched"] == 30.00
    assert cum["pareto_subset_within_envelope_count"] == 20
    assert cum["pareto_subset_cumulative_cost_usd"] == 20.00


def test_render_would_dispatch_table_has_header():
    rows = bdrs.extract_would_dispatch_rows(_autopilot_report_payload())
    table = bdrs.render_would_dispatch_table(rows)
    assert "candidate_id" in table
    assert "cost ($)" in table
    assert "predicted Δ" in table
    assert "cand_a" in table


def test_render_would_dispatch_table_handles_empty():
    table = bdrs.render_would_dispatch_table([])
    assert "no dispatch halt events" in table


def test_main_writes_both_artifacts(tmp_path: Path):
    report_path = tmp_path / "autopilot_loop_report.json"
    report_path.write_text(json.dumps(_autopilot_report_payload()), encoding="utf-8")
    out_dir = tmp_path / "out"

    rc = bdrs.main(
        [
            "--autopilot-report",
            str(report_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "would_dispatch.json").exists()
    assert (out_dir / "dry_run_report.md").exists()

    payload = json.loads((out_dir / "would_dispatch.json").read_text())
    assert payload["schema"] == "tac_autopilot_dry_run_summary_v1"
    assert payload["promotion_eligible"] is False
    assert payload["score_claim"] is False
    assert len(payload["would_dispatch_rows"]) == 3


def test_main_refuses_tmp_output_dir(tmp_path: Path, capsys):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_autopilot_report_payload()), encoding="utf-8")

    rc = bdrs.main(
        [
            "--autopilot-report",
            str(report_path),
            "--output-dir",
            "/tmp/dry_run",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "/tmp" in captured.err


def test_main_refuses_missing_report(tmp_path: Path, capsys):
    rc = bdrs.main(
        [
            "--autopilot-report",
            str(tmp_path / "nonexistent.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_md_report_contains_pareto_section(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_autopilot_report_payload()), encoding="utf-8")
    out_dir = tmp_path / "out"
    bdrs.main(
        [
            "--autopilot-report",
            str(report_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    md = (out_dir / "dry_run_report.md").read_text()
    assert "Pareto subset" in md
    assert "Cumulative cost projection" in md
    assert "operator_authorize_autopilot_le_5_dollar_mode.sh" in md
    assert "[predicted; cathedral autopilot ranking]" in md
    assert "promotion_eligible: false" in md.lower()


def test_md_report_omits_dollar_huge_cost_from_pareto(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_autopilot_report_payload()), encoding="utf-8")
    out_dir = tmp_path / "out"
    bdrs.main(
        [
            "--autopilot-report",
            str(report_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    md = (out_dir / "dry_run_report.md").read_text()
    # The huge-cost candidate must NOT appear in the Pareto subset section
    # (bounded by the next H2 section header so we don't bleed into the
    # full would-dispatch table which legitimately lists every candidate).
    pareto_block_start = md.find("Recommended subset to authorize first")
    next_section = md.find("\n## ", pareto_block_start)
    pareto_block = md[pareto_block_start:next_section if next_section > 0 else None]
    assert "cand_huge_cost" not in pareto_block


def test_main_includes_composition_constraints_section(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_autopilot_report_payload()), encoding="utf-8")
    out_dir = tmp_path / "out"
    bdrs.main(
        [
            "--autopilot-report",
            str(report_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    md = (out_dir / "dry_run_report.md").read_text()
    assert "n_dropped" in md
