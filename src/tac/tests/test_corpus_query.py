"""Tests for tools/corpus_query.py — the #346 retrieval-first one-query surface.

All end-to-end tests run against a synthetic temp corpus (monkeypatched store paths)
so they are hermetic; the real-corpus smoke lives in the tool's own CLI usage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import corpus_query as cq  # noqa: E402


# ─────────────────────────── tokenize / dates / scoring ───────────────────────────


def test_tokenize_drops_stopwords_and_dedupes() -> None:
    terms = cq.tokenize_query("the dash comb OF the dash registration a x")
    assert terms == ["dash", "comb", "registration"]


def test_tokenize_keeps_underscore_and_digit_terms() -> None:
    assert cq.tokenize_query("freq_along 08l") == ["freq_along", "08l"]


def test_tokenize_empty_query() -> None:
    assert cq.tokenize_query("the of a") == []


def test_derive_date_compact_and_dashed() -> None:
    assert cq._derive_date("memo_20260707.md", None) == "2026-07-07"
    assert cq._derive_date("row 2026-06-30 text", None) == "2026-06-30"


def test_derive_date_rejects_invalid_month_falls_back_to_mtime() -> None:
    # 20261399 has month 13 → invalid; falls back to the mtime date.
    out = cq._derive_date("memo_20261399.md", 1_600_000_000.0)
    assert out == "2020-09-13"


def test_derive_date_none_when_nothing() -> None:
    assert cq._derive_date("no date here", None) is None


def test_score_requires_two_distinct_terms_for_long_queries() -> None:
    terms = ["alpha", "beta", "gamma"]
    score_one, _ = cq._score_unit("alpha alpha alpha", terms, "2026-07-07")
    assert score_one == 0.0
    score_two, _ = cq._score_unit("alpha beta", terms, "2026-07-07")
    assert score_two > 0.0


def test_score_single_term_query_matches_on_one() -> None:
    score, _ = cq._score_unit("muon everywhere", ["muon"], None)
    assert score > 0.0


def test_score_recency_bonus_prefers_fresh() -> None:
    terms = ["muon"]
    fresh, _ = cq._score_unit("muon", terms, cq._dt.datetime.now().strftime("%Y-%m-%d"))
    stale, _ = cq._score_unit("muon", terms, "2026-01-01")
    assert fresh > stale


def test_matching_lines_capped_and_truncated() -> None:
    text = "\n".join(["muon " + "x" * 500] * 10)
    lines = cq._matching_lines(text, ["muon"])
    assert len(lines) == 3
    assert all(len(ln) <= 200 for ln in lines)


# ─────────────────────────── synthetic corpus end-to-end ───────────────────────────


@pytest.fixture()
def fake_corpus(tmp_path, monkeypatch):
    research = tmp_path / "research"
    research.mkdir()
    (research / "memo_dash_comb_20260707.md").write_text(
        "# dash comb\nthe dash comb registration audit is owed\ncomb comb dash\n"
    )
    (research / "memo_unrelated_20260101.md").write_text("nothing to see here\n")
    sub = research / "subdir"
    sub.mkdir()
    (sub / "nested_dash_20260706.md").write_text("dash comb nested memo\n")
    # A DAG file (picked up by the dag store, excluded from research).
    (research / "sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md").write_text(
        "preamble\n## FEED-08c dash comb in-training A/B\ncomb registration detail\n"
        "## FEED-zz unrelated feed\nno terms\n"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "manual.md").write_text("dash comb appears in docs\n")
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("- [dash comb owed audit](L65)\n")

    eq = tmp_path / "equations.jsonl"
    eq.write_text(
        json.dumps({"equation_id": "dash_erasure_v1", "equation_payload": {"x": "dash comb"},
                    "written_at_utc": "2026-07-05T00:00:00Z"}) + "\n"
        + json.dumps({"equation_id": "dash_erasure_v1",
                      "equation_payload": {"x": "dash comb dash comb refined"},
                      "written_at_utc": "2026-07-06T00:00:00Z"}) + "\n"
    )
    council = tmp_path / "council.jsonl"
    council.write_text(
        json.dumps({"deliberation_id": "dash_comb_symposium_20260705",
                    "topic": "dash comb", "council_verdict": "PROCEED"}) + "\n"
        + "not json\n"
    )

    monkeypatch.setattr(cq, "_RESEARCH_DIR", research)
    monkeypatch.setattr(cq, "_DOCS_DIR", docs)
    monkeypatch.setattr(cq, "_memory_dir", lambda: memory)
    monkeypatch.setattr(cq, "_EQUATIONS_JSONL", eq)
    monkeypatch.setattr(cq, "_COUNCIL_JSONL", council)
    monkeypatch.setattr(cq, "_TASKS_JSONL", tmp_path / "absent_tasks.jsonl")
    return tmp_path


def test_run_query_finds_hits_across_stores(fake_corpus) -> None:
    result = cq.run_query("dash comb", top=20)
    stores_hit = {h["store"] for h in result["hits"]}
    assert {"research", "dag", "docs", "memory", "equations", "council"} <= stores_hit
    refs = [h["ref"] for h in result["hits"]]
    assert any("memo_dash_comb_20260707.md" in r for r in refs)
    assert any("nested_dash_20260706.md" in r for r in refs), "recursive subdir memo missing"


def test_run_query_excludes_dag_file_from_research_store(fake_corpus) -> None:
    result = cq.run_query("dash comb", top=50)
    research_refs = [h["ref"] for h in result["hits"] if h["store"] == "research"]
    assert not any("sub015_DAG" in r for r in research_refs)


def test_run_query_dag_store_yields_only_matching_feed_block(fake_corpus) -> None:
    result = cq.run_query("dash comb", stores=["dag"], top=10)
    assert len(result["hits"]) == 1
    assert "FEED-08c" in result["hits"][0]["ref"]


def test_run_query_dedupes_repeated_equation_ids(fake_corpus) -> None:
    result = cq.run_query("dash comb", stores=["equations"], top=10)
    assert len(result["hits"]) == 1
    assert result["hits"][0]["ref"] == "dash_erasure_v1"


def test_run_query_absent_tasks_store_skipped_gracefully(fake_corpus) -> None:
    result = cq.run_query("dash comb", stores=["tasks"], top=10)
    assert result["hits"] == []
    assert result["stores_consulted"]["tasks"] == 0


def test_run_query_unparseable_jsonl_line_skipped(fake_corpus) -> None:
    result = cq.run_query("dash comb", stores=["council"], top=10)
    assert len(result["hits"]) == 1
    assert result["hits"][0]["ref"] == "dash_comb_symposium_20260705"


def test_run_query_ranking_is_deterministic(fake_corpus) -> None:
    a = cq.run_query("dash comb", top=20)
    b = cq.run_query("dash comb", top=20)
    assert [h["ref"] for h in a["hits"]] == [h["ref"] for h in b["hits"]]


def test_run_query_store_filter(fake_corpus) -> None:
    result = cq.run_query("dash comb", stores=["docs"], top=10)
    assert all(h["store"] == "docs" for h in result["hits"])
    assert set(result["stores_consulted"]) == {"docs"}


def test_run_query_time_budget_truncates_without_crash(fake_corpus) -> None:
    result = cq.run_query("dash comb", max_seconds=0.0)
    assert result["truncated"] is True


def test_run_query_no_terms_returns_empty(fake_corpus) -> None:
    result = cq.run_query("the of a")
    assert result["hits"] == [] and result["terms"] == []


def test_run_query_hit_shape_and_date(fake_corpus) -> None:
    result = cq.run_query("dash comb registration audit", top=5)
    hit = result["hits"][0]
    assert set(hit) == {"store", "ref", "date", "score", "lines"}
    assert hit["date"] and hit["date"].startswith("20")
    assert hit["lines"] and all(isinstance(ln, str) for ln in hit["lines"])


def test_format_human_contains_stores_consulted(fake_corpus) -> None:
    out = cq.format_human(cq.run_query("dash comb", top=3))
    assert "STORES CONSULTED:" in out
    assert "[research]" in out


def test_cli_json_and_unknown_store(fake_corpus, capsys) -> None:
    rc = cq.main(["dash comb", "--json", "--top", "3", "--stores", "research"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"]
    assert cq.main(["dash comb", "--stores", "notastore"]) == 2
