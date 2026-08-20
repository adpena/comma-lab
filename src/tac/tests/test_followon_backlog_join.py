# SPDX-License-Identifier: MIT
"""Tests for the registry-first follow-on backlog join."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from tac.followon_backlog_join import (
    build_followon_backlog_join,
    render_markdown,
    task_refs,
)
from tac.followon_ledger import ExecutionCorpus, SuccessorIndex


def _memo(root: Path, name: str, body: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(body, encoding="utf-8")


def _task(task_id: str, *, status: str = "pending", owner: str = "codex-owner") -> dict:
    return {
        "task_id": task_id,
        "title": f"task {task_id}",
        "status": status,
        "owner": owner,
        "event_notes": "",
    }


def _corpus(*names: str) -> ExecutionCorpus:
    return ExecutionCorpus(
        produced_names=frozenset({"wr1_kneeA_realized_gate_receipt.json", *names}),
        produced_paths={},
    )


def _index() -> SuccessorIndex:
    return SuccessorIndex(
        touched={"src/tac/followon_ledger.py": date(2026, 8, 1)},
        available=True,
        since=date(2026, 7, 18),
        reason="test index",
        tracked=frozenset({"src/tac/followon_ledger.py", "src/tac/x.py"}),
    )


def test_task_refs_are_exact_and_deduped() -> None:
    assert task_refs("#840 and #840 plus #18 and #18510") == ("840", "18")


def test_unknown_memo_row_gets_repo_task_owner_and_fire_order(tmp_path: Path) -> None:
    _memo(
        tmp_path,
        "ddm_q_20260801.md",
        "A $0 follow-on is owed for #840 but it names no file artifact yet.\n",
    )
    report = build_followon_backlog_join(
        memo_root=tmp_path,
        since=date(2026, 8, 1),
        today=date(2026, 8, 4),
        task_rows=[_task("840", owner="ddm_cf1")],
        corpus=_corpus(),
        successor_index=_index(),
    )
    row = next(r for r in report["dispositions"] if r["source"] == "memo_followon")
    assert row["verdict"] == "UNKNOWN"
    assert row["disposition"] == "QUEUED-WITH-FIRE-ORDER"
    assert row["owner"] == "ddm_cf1"
    assert row["task_matches"][0]["task_id"] == "840"
    assert report["summaries"]["unowned_queued_rows"] == 0


def test_present_task_output_folds_task_row(tmp_path: Path) -> None:
    report = build_followon_backlog_join(
        memo_root=tmp_path,
        since=date(2026, 8, 1),
        today=date(2026, 8, 4),
        task_rows=[
            {
                **_task("10", status="pending", owner="ddm_x"),
                "title": "land `x_receipt.json`",
            }
        ],
        corpus=_corpus("x_receipt.json"),
        successor_index=_index(),
    )
    task_row = next(r for r in report["dispositions"] if r["source"] == "canonical_task")
    assert task_row["verdict"] == "EXECUTED"
    assert task_row["disposition"] == "FOLDED"


def test_placeholder_task_owner_is_reassigned_to_qj1(tmp_path: Path) -> None:
    report = build_followon_backlog_join(
        memo_root=tmp_path,
        since=date(2026, 8, 1),
        today=date(2026, 8, 4),
        task_rows=[_task("10", status="pending", owner="MAIN / burn owner")],
        corpus=_corpus(),
        successor_index=_index(),
    )
    task_row = next(r for r in report["dispositions"] if r["source"] == "canonical_task")
    assert task_row["disposition"] == "QUEUED-WITH-FIRE-ORDER"
    assert task_row["owner"] == "codex-qj1-followon-drain"


def test_handoff_task_target_uses_closed_task_ids(tmp_path: Path) -> None:
    _memo(
        tmp_path,
        "ddm_h_20260801.md",
        "## NEXT-IF-RESUMED\n\nHand this off to #840 as the natural donor for the cure.\n",
    )
    report = build_followon_backlog_join(
        memo_root=tmp_path,
        since=date(2026, 8, 1),
        today=date(2026, 8, 4),
        task_rows=[_task("840", status="completed", owner="ddm_cf1")],
        corpus=_corpus(),
        successor_index=_index(),
    )
    handoff = next(r for r in report["dispositions"] if r["source"] == "handoff")
    assert handoff["verdict"] == "ADVANCED"
    assert handoff["disposition"] == "FOLDED"
    assert handoff["owner"] == "ddm_cf1"


def test_render_markdown_contains_boundaries(tmp_path: Path) -> None:
    report = build_followon_backlog_join(
        memo_root=tmp_path,
        since=date(2026, 8, 1),
        today=date(2026, 8, 4),
        task_rows=[_task("1")],
        corpus=_corpus(),
        successor_index=_index(),
    )
    text = render_markdown(report)
    assert "repo-visible join" in text
    assert "score_claim=false" in text


def test_ranked_head_rows_are_wired_with_declared_denominator(tmp_path: Path) -> None:
    _memo(
        tmp_path,
        "ddm_p1a_followon_unknown_adjudication_20260801.md",
        """# p1a

## §3 The 29 open items, ranked by COST-TO-FALSIFY

### T0 -- a read

| # | item | rows | evidence / next measurement |
|---|---|---|---|
| 1 | phi reducer gate | gc15:390 | $0 read before D+/- |

## §4 QA52
""",
    )
    _memo(
        tmp_path,
        "ddm_p2a_task_backlog_drain_20260801.md",
        """# p2a

## §4 THE ONE WORKING SIGNAL

| # | status | upd | created | subject |
|---|---|---:|---|---|
| **375** | pending | 0 | 07-09 | Auto-push Stop hook |
| 450 | pending | 0 | 07-12 | Lens Engine |

## §5 THE ADJUDICATION

| # | verdict | evidence (hand-verified) | cost-to-falsify |
|---|---|---|---|
| **375** | **ALREADY-CLOSED** | content verified | ~0 |

## §6 THE CURE
""",
    )
    report = build_followon_backlog_join(
        memo_root=tmp_path,
        since=date(2026, 8, 1),
        today=date(2026, 8, 4),
        task_rows=[
            _task("375", owner="MAIN"),
            _task("450", owner="ddm_lens"),
        ],
        corpus=_corpus(),
        successor_index=_index(),
    )

    assert report["schema"] == "tac.followon_backlog_join.v2"
    assert report["scopes"]["ranked_head"]["declared"] == 47
    assert report["summaries"]["ranked_head_rows"] == 3
    assert report["summaries"]["ranked_head_parse_coverage"] == 3 / 47
    assert report["summaries"]["ranked_head_dispositions"] == {
        "QUEUED-WITH-FIRE-ORDER": 2,
        "FOLDED": 1,
    }
    assert report["dispositions"][0]["source"] == "ranked_followon_head"
    assert report["dispositions"][0]["rank"] == 1
    assert "registered phi reducer" in report["dispositions"][0]["fire_order"]
    folded = next(
        row
        for row in report["dispositions"]
        if row["source_id"].endswith("p2a-never-named-375")
    )
    assert folded["disposition"] == "FOLDED"
    assert report["summaries"]["unowned_queued_rows"] == 0
