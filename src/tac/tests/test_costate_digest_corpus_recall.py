"""Tests for tools/costate_digest.py section_corpus_recall — the #346 recall-push.

The digest surfaces top corpus hits for an ACTIVE convening's next phase (push, not
pull). Fail-open, ≤5 lines, time-bounded — a broken recall must never break a session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import corpus_query as cq  # noqa: E402
import costate_digest as cd  # noqa: E402


@pytest.fixture()
def fake_corpus(tmp_path, monkeypatch):
    research = tmp_path / "research"
    research.mkdir()
    (research / "memo_debate_round_20260707.md").write_text(
        "debate round designer defends the draft\nred-team debate round detail\n"
    )
    monkeypatch.setattr(cq, "_RESEARCH_DIR", research)
    monkeypatch.setattr(cq, "_DOCS_DIR", tmp_path / "absent_docs")
    monkeypatch.setattr(cq, "_memory_dir", lambda: tmp_path / "absent_memory")
    monkeypatch.setattr(cq, "_EQUATIONS_JSONL", tmp_path / "absent.jsonl")
    monkeypatch.setattr(cq, "_COUNCIL_JSONL", tmp_path / "absent.jsonl")
    monkeypatch.setattr(cq, "_TASKS_JSONL", tmp_path / "absent.jsonl")
    return tmp_path


def test_no_convening_yields_nothing() -> None:
    assert cd.section_corpus_recall(None) == ([], None)
    assert cd.section_corpus_recall({}) == ([], None)
    assert cd.section_corpus_recall({"next_phase": ""}) == ([], None)


def test_active_convening_surfaces_top_hits(fake_corpus) -> None:
    lines, data = cd.section_corpus_recall({"next_phase": "debate round designer"})
    assert lines and lines[0].startswith("the corpus knows")
    assert len(lines) <= 5
    assert any("memo_debate_round_20260707.md" in ln for ln in lines)
    assert data and data["hits"]


def test_no_hits_yields_nothing(fake_corpus) -> None:
    lines, data = cd.section_corpus_recall({"next_phase": "zzznothingmatcheszzz qqq"})
    assert lines == [] and data is None


def test_fail_open_on_query_crash(monkeypatch) -> None:
    def boom(*_a, **_k):  # pragma: no cover - trivial
        raise RuntimeError("store exploded")

    monkeypatch.setattr(cq, "run_query", boom)
    lines, data = cd.section_corpus_recall({"next_phase": "anything"})
    assert data is None
    assert len(lines) == 1 and "unavailable" in lines[0]


def test_build_digest_never_raises_with_recall_wired() -> None:
    lines, data = cd.build_digest()
    assert lines and "corpus_recall" in data
