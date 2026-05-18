# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_duckdb import (
    CANONICAL_TABLES,
    audit_table_provenance,
    push_to_hf,
    refresh_all_tables,
    run_canonical_query,
)


def _seed_repo(root: Path) -> None:
    state = root / ".omx" / "state"
    research = root / ".omx" / "research"
    state.mkdir(parents=True)
    research.mkdir(parents=True)
    (state / "lane_registry.json").write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "id": "lane_test",
                        "name": "Test Lane",
                        "phase": 2,
                        "level": 0,
                        "gates": {},
                        "notes": "",
                    }
                ]
            }
        )
    )
    (state / "events.jsonl").write_text('{"event": 1}\n{"event": 2}\n')
    (state / "sample.json").write_text('{"ok": true}\n')
    (research / "test_memo.md").write_text(
        "---\n"
        "title: Test Memo\n"
        "date_utc: 2026-05-18T00:00:00Z\n"
        "lane_id: lane_test\n"
        "research_only: true\n"
        "score_claim: false\n"
        "promotion_eligible: false\n"
        "---\n"
        "\n"
        "# Test Memo\n"
    )


def test_canonical_duckdb_refreshes_source_backed_tables(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"

    result = refresh_all_tables(
        tmp_path,
        db_path=db_path,
        tables=("lanes", "research_memos", "state_json_files", "state_jsonl_files"),
    )

    assert set(CANONICAL_TABLES) >= {"lanes", "research_memos"}
    assert result["lanes"]["row_count"] == 1
    assert result["research_memos"]["row_count"] == 1
    assert result["state_json_files"]["row_count"] >= 1
    assert result["state_jsonl_files"]["row_count"] == 1

    provenance = audit_table_provenance("research_memos", db_path=db_path)
    assert provenance["duckdb_is_source_of_truth"] is False
    assert provenance["row_count"] == 1

    rows = run_canonical_query("lanes_with_research_memos", db_path=db_path)
    assert rows[0]["lane_id"] == "lane_test"
    assert rows[0]["memo_count"] == 1


def test_canonical_duckdb_hf_push_requires_operator_approval(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"
    refresh_all_tables(tmp_path, db_path=db_path, tables=("lanes",))

    with pytest.raises(PermissionError, match="operator_approved"):
        push_to_hf(
            "lanes",
            "adpena/comma-canonical-test",
            db_path=db_path,
            export_dir=tmp_path / "export",
            operator_approved=False,
        )
