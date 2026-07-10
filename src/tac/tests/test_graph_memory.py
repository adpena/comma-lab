# SPDX-License-Identifier: MIT
"""Tests for tac.graph_memory (task #411).

Two layers:
  (1) deterministic fixture tests — a tiny synthetic corpus with known nodes/edges,
      asserting the parsers + reconstruction behave exactly (no dependence on the
      live corpus, which changes).
  (2) a real-corpus smoke — the parsers run on the actual repo corpus and produce a
      structurally-valid graph (proves the parsers work on real data, robust to
      count drift).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.graph_memory import build_graph, format_human, reconstruct
from tac.graph_memory.build import (
    parse_dag_feeds,
    parse_deferrals,
    parse_equations,
    parse_memory_files,
    parse_tasks,
)
from tac.graph_memory.model import Edge, Graph, Node
from tac.graph_memory.recall import tokenize


# ---------------------------------------------------------------- fixtures ---
@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> dict:
    """A small synthetic corpus mirroring the real store formats."""
    memdir = tmp_path / "memory"
    memdir.mkdir()
    (memdir / "alpha.md").write_text(
        '---\nname: alpha-lane-dseg\ntype: feedback\n'
        'description: "Alpha finding about lane d_seg islands and #205 OOM."\n'
        '---\nBody links to [[beta-pose-solved]] and sister: [[gamma-note]].\n'
        'It references src/tac/boundary_math/lever_b_generator.py and #247.\n',
        encoding="utf-8",
    )
    (memdir / "beta.md").write_text(
        '---\nname: beta-pose-solved\ntype: project\n'
        'description: "Beta pose carrier VERDICT CONFIRMED via [[alpha-lane-dseg]]."\n'
        '---\nSee eq analytic_lane_band_dseg_recon_floor_v1.\n',
        encoding="utf-8",
    )
    dag = tmp_path / "sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
    dag.write_text(
        "# DAG\n\n"
        "## FEED-oom (2026-07-02): #205 OOM verdict-batch NO-GO\n"
        "The #205 n600 OOM measured; see [[alpha-lane-dseg]] and "
        "src/tac/boundary_math/lever_b_generator.py; Shannon reviewed.\n\n"
        "### FEED-lane (2026-07-03) — lane d_seg CONFIRMED\n"
        "Lane islands finding, references #247 and eq "
        "analytic_lane_band_dseg_recon_floor_v1.\n",
        encoding="utf-8",
    )
    eqs = tmp_path / "canonical_equations_registry.jsonl"
    eqs.write_text(
        json.dumps({
            "equation_id": "analytic_lane_band_dseg_recon_floor_v1",
            "equation_payload": {
                "equation_id": "analytic_lane_band_dseg_recon_floor_v1",
                "name": "Analytic lane-band d_seg floor",
                "one_line_summary": "analytic poly-band lane d_seg 0.00087",
                "canonical_producers": ["tools/lane_band.py"],
                "canonical_consumers": ["tac.boundary_math.lane_sdf_component"],
            },
        }) + "\n",
        encoding="utf-8",
    )
    tasks = tmp_path / "canonical_task_status.jsonl"
    tasks.write_text(
        json.dumps({"task_id": "t_lane", "title": "Lane d_seg task",
                    "status": "open", "blockers": ["#205"], "event_notes": "n"}) + "\n",
        encoding="utf-8",
    )
    deferral = tmp_path / "deferral_ledger.md"
    deferral.write_text(
        "| # | Deferral | Trigger | Owner | Status |\n|---|---|---|---|---|\n"
        "| D1 | GPU verdict probe | #205 pre-launch | #355 | ARMED |\n",
        encoding="utf-8",
    )
    return {
        "memdir": memdir, "dag": dag, "eqs": eqs, "tasks": tasks, "deferral": deferral,
        "tmp": tmp_path,
    }


def _build(tiny_corpus: dict) -> Graph:
    return build_graph(
        memory_dir=tiny_corpus["memdir"],
        dag_path=tiny_corpus["dag"],
        equations_path=tiny_corpus["eqs"],
        tasks_path=tiny_corpus["tasks"],
        deferral_path=tiny_corpus["deferral"],
    )


# --------------------------------------------------------------- unit tests --
def test_model_add_and_traverse():
    g = Graph()
    g.add_node(Node("a", "memory", "A"))
    g.add_node(Node("b", "memory", "B"))
    g.add_edge(Edge("a", "b", "links"))
    assert g.degree("a") == 1 and g.degree("b") == 1
    assert [e.dst for e in g.out_edges("a")] == ["b"]
    assert [e.src for e in g.in_edges("b")] == ["a"]  # backlink present
    nbrs = list(g.neighbors("b"))
    assert nbrs and nbrs[0][1] == "in" and nbrs[0][2] == "a"


def test_model_no_self_loops_and_dedup():
    g = Graph()
    g.add_edge(Edge("a", "a", "links"))  # self-loop dropped
    g.add_edge(Edge("a", "b", "links"))
    g.add_edge(Edge("a", "b", "links"))  # dup dropped
    assert len(g.edges) == 1


def test_node_merge_upgrades_entity_stub():
    g = Graph()
    g.ensure_stub("memory:x", ntype="memory", title="x")
    g.add_node(Node("memory:x", "memory", "X full", summary="rich summary here"))
    assert g.nodes["memory:x"].summary == "rich summary here"


def test_parse_memory_frontmatter_and_links(tiny_corpus):
    g = Graph()
    n = parse_memory_files(g, tiny_corpus["memdir"])
    assert n == 2
    assert "memory:alpha-lane-dseg" in g.nodes
    assert g.nodes["memory:alpha-lane-dseg"].ntype == "memory"
    # [[wikilink]] edge
    assert any(e.etype == "links" and e.dst == "memory:beta-pose-solved"
               for e in g.out_edges("memory:alpha-lane-dseg"))
    # sister edge
    assert any(e.etype == "sister" and e.dst == "memory:gamma-note"
               for e in g.out_edges("memory:alpha-lane-dseg"))
    # tag -> topic
    assert any(e.etype == "tagged" and e.dst == "topic:feedback"
               for e in g.out_edges("memory:alpha-lane-dseg"))
    # #247 ref entity
    assert any(e.dst == "ref:#247" for e in g.out_edges("memory:alpha-lane-dseg"))


def test_parse_equations_producers_consumers(tiny_corpus):
    g = Graph()
    ids = parse_equations(g, tiny_corpus["eqs"])
    assert "analytic_lane_band_dseg_recon_floor_v1" in ids
    eq = "eq:analytic_lane_band_dseg_recon_floor_v1"
    assert g.nodes[eq].ntype == "equation"
    assert any(e.etype == "produces" and e.dst == eq for e in g.in_edges(eq))
    assert any(e.etype == "consumes" and e.dst == eq for e in g.in_edges(eq))


def test_parse_dag_feeds_verdict_becomes_decision(tiny_corpus):
    g = Graph()
    ids = parse_equations(g, tiny_corpus["eqs"])
    parse_dag_feeds(g, tiny_corpus["dag"], ids)
    assert "feed:oom" in g.nodes and "feed:lane" in g.nodes
    assert g.nodes["feed:oom"].ntype == "decision"  # NO-GO verdict
    assert g.nodes["feed:lane"].ntype == "decision"  # CONFIRMED verdict
    # FEED references a memory + a file + a person + the equation
    outs = {(e.dst, e.etype) for e in g.out_edges("feed:oom")}
    assert ("memory:alpha-lane-dseg", "references") in outs
    assert ("person:Shannon", "references") in outs
    assert ("ref:#205", "references") in outs
    # eq ref only when the id is a known equation
    assert any(e.dst == "eq:analytic_lane_band_dseg_recon_floor_v1"
               for e in g.out_edges("feed:lane"))


def test_parse_dag_duplicate_slug_disambiguated(tmp_path):
    dag = tmp_path / "sub015_DAG_topaiml_reopen_and_pursuit_plan_x.md"
    dag.write_text(
        "## FEED-gf first block\nbody one\n\n"
        "### FEED-gf CORRECTION second block\nbody two\n",
        encoding="utf-8",
    )
    g = Graph()
    parse_dag_feeds(g, dag, set())
    assert "feed:gf" in g.nodes and "feed:gf#2" in g.nodes  # both preserved


def test_parse_tasks_and_deferrals(tiny_corpus):
    g = Graph()
    parse_tasks(g, tiny_corpus["tasks"])
    parse_deferrals(g, tiny_corpus["deferral"])
    assert g.nodes["task:t_lane"].ntype == "task"
    assert any(e.etype == "blocks" and e.dst == "task:t_lane"
               for e in g.out_edges("ref:#205"))
    assert "deferral:D1" in g.nodes and g.nodes["deferral:D1"].ntype == "deferral"


# -------------------------------------------------------- reconstruction ----
def test_reconstruct_finds_seed_and_traverses(tiny_corpus):
    g = _build(tiny_corpus)
    recon = reconstruct(g, "lane d_seg islands", max_seeds=3, max_nodes=10)
    seed_ids = {s for s, _ in recon.seeds}
    # a lane-d_seg node must be a seed
    assert any("lane" in s or "dseg" in s or "lane" in g.nodes[s].title.lower()
               for s in seed_ids)
    # reconstruction assembles connected nodes (more than just the seeds)
    assert len(recon.nodes) >= len(recon.seeds)
    # every reconstructed non-seed node has a traversal path back to a seed
    for nid in recon.nodes:
        if nid not in seed_ids:
            assert nid in recon.paths


def test_reconstruct_entity_anchor(tiny_corpus):
    g = _build(tiny_corpus)
    recon = reconstruct(g, "#205 OOM", max_seeds=4, max_nodes=12)
    # the #205 OOM FEED (a decision) should reconstruct
    assert any(nid == "feed:oom" for nid in recon.nodes)


def test_reconstruct_deterministic(tiny_corpus):
    g = _build(tiny_corpus)
    r1 = reconstruct(g, "lane d_seg", max_nodes=8)
    r2 = reconstruct(g, "lane d_seg", max_nodes=8)
    assert r1.nodes == r2.nodes and r1.seeds == r2.seeds


def test_reconstruct_no_match_is_graceful(tiny_corpus):
    g = _build(tiny_corpus)
    recon = reconstruct(g, "zzznonexistentqqq")
    assert recon.seeds == [] and recon.nodes == []
    out = format_human(recon, g)
    assert "no matching seed" in out


def test_format_human_shows_paths(tiny_corpus):
    g = _build(tiny_corpus)
    recon = reconstruct(g, "lane d_seg")
    out = format_human(recon, g)
    assert "RECONSTRUCTED CONTEXT" in out and "SEEDS" in out


def test_tokenize_drops_stopwords():
    toks = tokenize("what do we know about the lane d_seg")
    assert "the" not in toks and "what" not in toks
    assert "lane" in toks


# ---------------------------------------------------------- persistence -----
def test_save_load_roundtrip(tiny_corpus, tmp_path):
    g = _build(tiny_corpus)
    npath, epath = tmp_path / "n.jsonl", tmp_path / "e.jsonl"
    g.save(npath, epath)
    g2 = Graph.load(npath, epath)
    assert set(g.nodes) == set(g2.nodes)
    assert set(g.edges) == set(g2.edges)


def test_save_is_deterministic(tiny_corpus, tmp_path):
    g = _build(tiny_corpus)
    g.save(tmp_path / "n1.jsonl", tmp_path / "e1.jsonl")
    g.save(tmp_path / "n2.jsonl", tmp_path / "e2.jsonl")
    assert (tmp_path / "n1.jsonl").read_text() == (tmp_path / "n2.jsonl").read_text()
    assert (tmp_path / "e1.jsonl").read_text() == (tmp_path / "e2.jsonl").read_text()


# ---------------------------------------------------- real-corpus smoke -----
def test_real_corpus_builds_structurally_valid():
    """The parsers run on the ACTUAL repo corpus and produce a valid graph."""
    g = build_graph()
    assert len(g.nodes) > 100, "real corpus should yield a substantial graph"
    assert len(g.edges) > 100
    types = g.counts_by_type()
    # the corpus must contribute all the major node kinds
    for expected in ("memory", "finding", "equation"):
        assert types.get(expected, 0) > 0, f"missing {expected} nodes"
    etypes = g.edge_counts_by_type()
    assert etypes.get("links", 0) > 0, "wikilink edges must be present"


def test_real_corpus_reconstructs_known_topic():
    """A real query reconstructs a non-empty, path-connected context."""
    g = build_graph()
    recon = reconstruct(g, "lane d_seg", max_nodes=6)
    assert recon.seeds, "known topic should have seeds in the real corpus"
    assert recon.nodes
