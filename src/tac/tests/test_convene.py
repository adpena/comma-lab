"""Tests for tools/convene.py — the 20-store grounding-packet assembler (#346 stage 2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import convene  # noqa: E402
import corpus_query as cq  # noqa: E402


def test_slugify() -> None:
    assert convene._slugify("Pose Carrier: byte-close!") == "pose_carrier_byte_close"
    assert convene._slugify("***") == "topic"
    assert len(convene._slugify("x" * 200)) <= 60


def test_checklist_has_20_stores_with_unique_numbers() -> None:
    nums = [s["num"] for s in convene.STANDING_CHECKLIST]
    assert nums == list(range(1, 21))
    for store in convene.STANDING_CHECKLIST:
        assert store["kind"] in {"bucket", "inline"}
        if store["kind"] == "inline":
            assert store["num"] in convene._INLINE_GENERATORS


def test_every_bucket_key_is_reachable() -> None:
    bucket_keys = {s["bucket"] for s in convene.STANDING_CHECKLIST if s["kind"] == "bucket"}
    reachable = {"research_general", "equations", "dag", "council", "memory", "tasks", "docs"}
    reachable |= {name for name, _ in convene._RESEARCH_SUBBUCKETS}
    assert bucket_keys <= reachable


@pytest.mark.parametrize(
    "ref,store,expected",
    [
        ("x/litsweep_training_dynamics_20260705.md", "research", "litsweep"),
        ("x/group_theory_deepmath_review_20260707.md", "research", "deepmath"),
        ("docs/triality_dag_dsl_equations_deepmath.md", "research", "deepmath"),
        ("x/openpilot_cross_surface_audit.md", "research", "openpilot"),
        ("x/fresh_run_master_lever_ledger_20260704.md", "research", "levers"),
        ("x/sweep_C_task_research_orphan_lever_ledger.md", "research", "orphans"),
        ("x/council_symposium_clean_config.md", "research", "symposia"),
        ("x/random_memo_20260701.md", "research", "research_general"),
        ("eq_id_v1", "equations", "equations"),
        ("dag.md :: ## FEED-08c", "dag", "dag"),
        ("delib_id", "council", "council"),
        ("MEMORY.md", "memory", "memory"),
        ("task_id", "tasks", "tasks"),
        ("docs/manual.md", "docs", "docs"),
    ],
)
def test_bucket_for_hit(ref: str, store: str, expected: str) -> None:
    assert convene._bucket_for_hit({"store": store, "ref": ref}) == expected


def test_orphan_pattern_wins_over_lever_pattern_first_match() -> None:
    # sweep_C is an ORPHAN ledger (compendium store 10) even though its name also
    # contains "lever_ledger" — the orphans pattern is deliberately checked first.
    got = convene._bucket_for_hit(
        {"store": "research", "ref": "x/sweep_C_task_research_orphan_lever_ledger.md"}
    )
    assert got == "orphans"


@pytest.fixture()
def fake_corpus(tmp_path, monkeypatch):
    research = tmp_path / "research"
    research.mkdir()
    (research / "memo_pose_carrier_20260707.md").write_text(
        "pose carrier byte close details\npose pose carrier\n"
    )
    (research / "litsweep_pose_20260705.md").write_text("pose carrier literature\n")
    (research / "sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md").write_text(
        "## FEED-x pose carrier feed\nbyte close\n"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("pose carrier doc\n")
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("pose carrier memory line\n")
    monkeypatch.setattr(cq, "_RESEARCH_DIR", research)
    monkeypatch.setattr(cq, "_DOCS_DIR", docs)
    monkeypatch.setattr(cq, "_memory_dir", lambda: memory)
    monkeypatch.setattr(cq, "_EQUATIONS_JSONL", tmp_path / "absent_eq.jsonl")
    monkeypatch.setattr(cq, "_COUNCIL_JSONL", tmp_path / "absent_council.jsonl")
    monkeypatch.setattr(cq, "_TASKS_JSONL", tmp_path / "absent_tasks.jsonl")
    return tmp_path


def test_build_packet_structure(fake_corpus) -> None:
    packet = convene.build_packet("pose carrier byte close")
    assert packet.startswith("# GROUNDING PACKET — pose carrier byte close")
    assert "## MUST-READ (in order)" in packet
    assert "## SEAT CONTRACT (binding)" in packet
    for store in convene.STANDING_CHECKLIST:
        assert f"### Store {store['num']} — {store['name']}" in packet
    assert "STORES CONSULTED:" in packet


def test_build_packet_buckets_hits(fake_corpus) -> None:
    packet = convene.build_packet("pose carrier byte close")
    # research memo lands in store 1; litsweep memo in store 7; dag FEED in store 3.
    store1 = packet.split("### Store 1 —")[1].split("### Store 2 —")[0]
    assert "memo_pose_carrier_20260707.md" in store1
    store7 = packet.split("### Store 7 —")[1].split("### Store 8 —")[0]
    assert "litsweep_pose_20260705.md" in store7
    store3 = packet.split("### Store 3 —")[1].split("### Store 4 —")[0]
    assert "FEED-x" in store3


def test_build_packet_empty_store_notes_no_hits(fake_corpus) -> None:
    packet = convene.build_packet("pose carrier byte close")
    store4 = packet.split("### Store 4 —")[1].split("### Store 5 —")[0]
    assert "no corpus hits" in store4


def test_build_packet_seat_contract_carries_key_disciplines(fake_corpus) -> None:
    packet = convene.build_packet("pose carrier")
    assert "ANTI-ANCHORING" in packet
    assert "STORES CONSULTED:" in packet  # retrieval-first clause in contract + trailer
    assert "base-content-sha256" in packet  # the serializer-absorption fix
    assert "FOREGROUND only" in packet


def test_inline_durable_state_flags_missing_and_stale() -> None:
    lines = convene._inline_durable_state([])
    assert lines, "durable-state section must emit at least one line"
    joined = "\n".join(lines)
    # These files are known-stale (compendium store 16) or live; either way tagged.
    assert ("STALE" in joined) or ("live" in joined) or ("MISSING" in joined)


def test_inline_git_log_filters_by_terms() -> None:
    lines = convene._inline_git_log(["zzzznomatchzzzz"])
    assert lines == ["- (no matching commit messages in the last 300)"] or "unavailable" in lines[0]


def test_inline_claude_md_matches_or_fallback() -> None:
    lines = convene._inline_claude_md(["witness", "capstone"])
    assert lines and all(isinstance(line, str) for line in lines)
    assert any("CLAUDE.md" in line for line in lines)


def test_inline_generators_never_raise() -> None:
    for num, gen in convene._INLINE_GENERATORS.items():
        out = gen(["pose", "carrier"])
        assert isinstance(out, list) and out, f"inline generator {num} must emit lines"


def test_main_writes_packet_to_out(fake_corpus, tmp_path, capsys) -> None:
    out = tmp_path / "packets" / "packet.md"
    rc = convene.main(["pose carrier byte close", "--out", str(out), "--top", "20"])
    assert rc == 0
    assert out.is_file()
    assert "GROUNDING PACKET" in out.read_text()
    assert str(out) in capsys.readouterr().out
