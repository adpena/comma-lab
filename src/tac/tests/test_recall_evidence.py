# SPDX-License-Identifier: MIT
"""Tests for tac.recall_evidence (task #569, P0-2 fused RecallEvidence ranking).

Pins: (1) the corpus + graph adapters normalize into typed RecallEvidence rows,
(2) reciprocal-rank fusion merges same-artifact hits across surfaces and drops
NOTHING (exact-identifier preservation), (3) cross-surface ordering, and the
key-normalization + guardrails.
"""
from __future__ import annotations

import pytest

from tac.graph_memory.model import Edge, Graph, Node
from tac.graph_memory.recall import reconstruct
from tac.recall_evidence import (
    RecallEvidence,
    _norm_key,
    evidence_from_corpus,
    evidence_from_reconstruction,
    fuse_recall,
    reciprocal_rank_fuse,
)


# ---------------------------------------------------------------- adapters ---
def test_evidence_from_corpus_ranks_and_hooks() -> None:
    result = {
        "hits": [
            {"store": "dag", "ref": "a.md :: FEED-x", "date": "2026-07-01",
             "score": 9.0, "lines": ["first hook", "second"]},
            {"store": "research", "ref": "b.md", "date": None, "score": 4.0, "lines": []},
        ],
    }
    rows = evidence_from_corpus(result)
    assert [r.rank for r in rows] == [1, 2]
    assert rows[0].source_surface == "corpus"
    assert rows[0].store == "dag"
    assert rows[0].hook_line == "first hook"
    assert rows[1].hook_line == ""  # no lines -> empty hook


def test_evidence_from_corpus_empty() -> None:
    assert evidence_from_corpus({"hits": []}) == []
    assert evidence_from_corpus({}) == []


def _lane_graph() -> Graph:
    g = Graph()
    g.add_node(Node(id="memory:lane", ntype="memory", title="lane d_seg finding",
                    summary="lane islands d_seg", source="/x/lane.md"))
    g.add_node(Node(id="feed:lane", ntype="finding", title="lane block",
                    summary="lane d_seg confirmed", source="/x/dag.md#L1-L2"))
    g.add_edge(Edge("memory:lane", "feed:lane", "references", "/x/lane.md"))
    return g


def test_evidence_from_reconstruction_positional_score_monotone() -> None:
    g = _lane_graph()
    recon = reconstruct(g, "lane d_seg")
    rows = evidence_from_reconstruction(recon, g)
    assert rows, "reconstruction should find lane seeds"
    assert rows[0].source_surface == "graph"
    # positional score must be strictly non-increasing with rank.
    scores = [r.score for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in rows] == list(range(1, len(rows) + 1))


# ---------------------------------------------------------------- fusion -----
def _ev(surface: str, ref: str, rank: int, *, store: str = "s", path: str | None = None) -> RecallEvidence:
    return RecallEvidence(
        source_surface=surface, store=store, ref=ref, path=path if path is not None else ref,
        score=1.0, hook_line="h", rank=rank,
    )


def test_fuse_merges_same_key_across_surfaces() -> None:
    corpus = [_ev("corpus", ".omx/research/x.md", 1)]
    graph = [_ev("graph", ".omx/research/x.md", 2, path=".omx/research/x.md#L3-L9")]
    fused = reciprocal_rank_fuse([corpus, graph])
    assert len(fused) == 1  # same artifact -> one merged row
    row = fused[0]
    assert row.contributing_surfaces == ("corpus", "graph")
    # rrf = 1/(60+1) + 1/(60+2), rounded to 6 decimals by the fuser.
    assert row.rrf_score == pytest.approx(1 / 61 + 1 / 62, abs=1e-6)
    # representative kept is the better (lower) native rank -> corpus.
    assert row.source_surface == "corpus"


def test_fuse_preserves_all_distinct_keys_no_drop() -> None:
    corpus = [_ev("corpus", "a", 1), _ev("corpus", "b", 2)]
    graph = [_ev("graph", "c", 1)]
    fused = reciprocal_rank_fuse([corpus, graph])
    assert {r.ref for r in fused} == {"a", "b", "c"}  # nothing dropped


def test_fuse_orders_shared_hit_above_single_surface() -> None:
    # 'shared' is rank-2 in both surfaces; 'solo' is rank-1 in one surface only.
    corpus = [_ev("corpus", "solo", 1), _ev("corpus", "shared", 2)]
    graph = [_ev("graph", "shared", 2)]
    fused = reciprocal_rank_fuse([corpus, graph])
    order = [r.ref for r in fused]
    # shared: 1/62 + 1/62 = 0.03226 ; solo: 1/61 = 0.01639 -> shared wins.
    assert order[0] == "shared"


def test_norm_key_strips_anchors_prefixes_and_dag_header() -> None:
    assert _norm_key(_ev("graph", "file:tools/x.py", 1, path="file:tools/x.py")) == "tools/x.py"
    assert _norm_key(_ev("corpus", "a.md :: FEED-x", 1, path="a.md :: FEED-x")) == "a.md"
    assert _norm_key(_ev("graph", "n", 1, path="/x/dag.md#L10-L20")) == "/x/dag.md"


def test_fuse_rejects_negative_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fuse([[_ev("corpus", "a", 1)]], k=-1)


def test_fuse_recall_convenience_single_surface() -> None:
    result = {"hits": [{"store": "dag", "ref": "a", "score": 5.0, "lines": ["hk"]}]}
    fused = fuse_recall(corpus_result=result)
    assert len(fused) == 1
    assert fused[0].contributing_surfaces == ("corpus",)


def test_fuse_recall_both_surfaces() -> None:
    g = _lane_graph()
    recon = reconstruct(g, "lane")
    result = {"hits": [{"store": "research", "ref": "/x/lane.md", "score": 5.0, "lines": ["hk"]}]}
    fused = fuse_recall(corpus_result=result, reconstruction=recon, graph=g)
    assert fused
    # /x/lane.md appears in corpus AND as the graph memory node source -> merged.
    merged = [r for r in fused if r.contributing_surfaces == ("corpus", "graph")]
    assert merged, "expected a cross-surface merge on the shared lane.md path"
