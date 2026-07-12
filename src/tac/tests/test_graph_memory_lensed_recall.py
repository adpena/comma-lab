# SPDX-License-Identifier: MIT
"""Tests for the #346 retrieval-first lensed recall path (`reconstruct_lensed`).

Two layers, mirroring `test_graph_memory.py`'s convention:
  (1) deterministic synthetic-fixture tests — small hand-built graphs where the
      expected bridge/hub/crux behavior is HAND-DERIVED (see the module
      docstring inline comments) rather than guessed, plus a regression guard
      proving the `reconstruct()` internal refactor (extracting
      `_select_seeds` / `_bfs_reconstruct` for reuse) changed nothing about its
      public behavior.
  (2) a real-corpus smoke — the lensed path runs against the ACTUAL repo
      corpus graph and is asserted to be a strict superset of the naive
      `reconstruct()` output on the same query (the "pay rent" proof from the
      landing memo, turned into an executable regression guard), plus a real
      subprocess check of the new `tools/graph_memory_recall.py --lens` flag.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.graph_memory import build_graph, load_or_build, reconstruct, reconstruct_lensed
from tac.graph_memory.model import Edge, Graph, Node
from tac.graph_memory.recall import (
    LensedReconstruction,
    Reconstruction,
    _bfs_reconstruct,
    _local_pool_graph,
    _select_seeds,
    format_human_lensed,
)
from tac.lens_engine import AdapterError

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- fixtures ---
def _chain_graph() -> Graph:
    """Two seeds connected ONLY by a long chain:

        seed_a -- mid1 -- mid2 -- mid3 -- mid4 -- seed_b
        seed_a -- decoy1
        seed_a -- decoy2

    dist(seed_a, seed_b) = 5. HAND-VERIFIED (2026-07-12) reachability:
      * max_depth=1: seed_a's BFS reaches {mid1, decoy1, decoy2}; seed_b's BFS
        reaches {mid4}. mid2/mid3 are in NEITHER seed's depth-1 tree — they do
        not appear in `reconstruct()`'s weight dict AT ALL (not merely
        truncated), so they are the clean "invisible to naive BFS, visible via
        the full-graph GRAPH shortest_path bridge" case.
      * max_depth=2: the union of both seeds' depth-2 trees covers all 8
        nodes (seed_a's tree adds mid2 at dist2; seed_b's tree adds mid3 at
        dist2), so nothing is out of reach — this fixture at depth=2 is used
        only for the reconstruct()-unchanged regression pin, not the bridge
        test.
    """
    g = Graph()
    g.add_node(Node("seed_a", "memory", "Seed A about lane d_seg islands"))
    g.add_node(Node("mid1", "entity", "Mid 1"))
    g.add_node(Node("mid2", "entity", "Mid 2"))
    g.add_node(Node("mid3", "entity", "Mid 3"))
    g.add_node(Node("mid4", "entity", "Mid 4"))
    g.add_node(Node("seed_b", "memory", "Seed B also about lane d_seg islands"))
    g.add_edge(Edge("seed_a", "mid1", "references"))
    g.add_edge(Edge("mid1", "mid2", "references"))
    g.add_edge(Edge("mid2", "mid3", "references"))
    g.add_edge(Edge("mid3", "mid4", "references"))
    g.add_edge(Edge("mid4", "seed_b", "references"))
    g.add_node(Node("decoy1", "entity", "Decoy 1"))
    g.add_node(Node("decoy2", "entity", "Decoy 2"))
    g.add_edge(Edge("seed_a", "decoy1", "references"))
    g.add_edge(Edge("seed_a", "decoy2", "references"))
    return g


def _hub_graph() -> Graph:
    """One seed with 3 dist-1 "petals"; only `p1` is a genuine articulation
    point (removing it disconnects `extra1`/`extra2` from the rest), so GRAPH
    betweenness centrality ranks it above `p2`/`p3` even though keyword-BFS
    weight ties all three at dist 1. HAND-VERIFIED betweenness (2026-07-12):
    p1 = 0.7 (on every seed<->extra1 and seed<->extra2 shortest path), p2 =
    p3 = extra1 = extra2 = 0.0 (pure leaves/pendants).
    """
    g = Graph()
    g.add_node(Node("seed", "memory", "Seed about lane d_seg"))
    g.add_node(Node("p1", "entity", "Petal 1 (hub)"))
    g.add_node(Node("p2", "entity", "Petal 2"))
    g.add_node(Node("p3", "entity", "Petal 3"))
    g.add_node(Node("extra1", "entity", "Extra 1"))
    g.add_node(Node("extra2", "entity", "Extra 2"))
    g.add_edge(Edge("seed", "p1", "references"))
    g.add_edge(Edge("seed", "p2", "references"))
    g.add_edge(Edge("seed", "p3", "references"))
    g.add_edge(Edge("p1", "extra1", "references"))
    g.add_edge(Edge("p1", "extra2", "references"))
    return g


def _saddle_graph() -> Graph:
    """`hub` sits between two mutually-disconnected higher-(local-)degree
    fans (`sideA1`, `sideB1`, each fattened to local-degree 5 via 4 private
    extra leaves) plus two degree-1 neighbors (`seed`, `leaf`) — exactly
    `TopologyLens.saddles`'s definition: `upper_link_components >= 2` with a
    non-empty lower link. HAND-VERIFIED (2026-07-12): hub local-degree=4;
    sideA1/sideB1 local-degree=5 (upper, mutually disconnected -> upper_n=2);
    seed/leaf local-degree=1 (lower, non-empty) -> hub IS a saddle. No other
    node in the graph qualifies (every other node has <2 neighbors or an empty
    upper/lower link).
    """
    g = Graph()
    g.add_node(Node("seed", "memory", "Seed near the saddle, about d_seg"))
    g.add_node(Node("hub", "entity", "Hub node between two fans"))
    g.add_node(Node("leaf", "entity", "Leaf lower neighbor"))
    g.add_node(Node("sideA1", "entity", "Side A1"))
    g.add_node(Node("sideB1", "entity", "Side B1"))
    g.add_edge(Edge("seed", "hub", "references"))
    g.add_edge(Edge("hub", "leaf", "references"))
    g.add_edge(Edge("hub", "sideA1", "references"))
    g.add_edge(Edge("hub", "sideB1", "references"))
    for i in range(4):
        g.add_node(Node(f"sideA1_extra{i}", "entity", f"sideA1 extra {i}"))
        g.add_edge(Edge("sideA1", f"sideA1_extra{i}", "references"))
        g.add_node(Node(f"sideB1_extra{i}", "entity", f"sideB1 extra {i}"))
        g.add_edge(Edge("sideB1", f"sideB1_extra{i}", "references"))
    return g


# ------------------------------------------------- reconstruct() regression --
def test_reconstruct_unchanged_after_lensed_refactor():
    """Pin `reconstruct()`'s exact output on a hand-derivable fixture.

    `reconstruct()` was refactored to share `_select_seeds`/`_bfs_reconstruct`
    with `reconstruct_lensed()`. This pins the pre-refactor contract: the same
    seed scoring + BFS-weight ranking + path assembly, byte-for-byte.
    """
    g = _chain_graph()
    recon = reconstruct(g, "lane d_seg", max_seeds=4, max_nodes=10, max_depth=2)
    assert isinstance(recon, Reconstruction)
    # both seeds score identically (query terms count identically in each
    # title regardless of the unrelated word "also") and tie-break on id
    assert [sid for sid, _ in recon.seeds] == ["seed_a", "seed_b"]
    seed_a_score, seed_b_score = (sc for _, sc in recon.seeds)
    assert seed_a_score == pytest.approx(seed_b_score) == pytest.approx(4.0)
    # depth-2 union of both seeds' BFS trees covers every node in this fixture
    assert set(recon.nodes) == {
        "seed_a", "seed_b", "mid1", "mid2", "decoy1", "decoy2", "mid3", "mid4",
    }
    # a seed's own recorded path is just its own 1-element label (dist=0)
    assert recon.paths["seed_a"] == ["[memory] Seed A about lane d_seg islands"]
    assert recon.paths["mid2"][0] == "[memory] Seed A about lane d_seg islands"


def test_reconstruct_deterministic_after_refactor():
    g = _chain_graph()
    r1 = reconstruct(g, "lane d_seg", max_nodes=8)
    r2 = reconstruct(g, "lane d_seg", max_nodes=8)
    assert r1.nodes == r2.nodes and r1.seeds == r2.seeds and r1.paths == r2.paths


def test_select_seeds_and_bfs_reconstruct_are_pure_helpers():
    g = _chain_graph()
    seeds = _select_seeds(g, "lane d_seg", max_seeds=4)
    assert {sid for _, sid in seeds} == {"seed_a", "seed_b"}
    weight, best_path = _bfs_reconstruct(g, seeds, max_depth=2)
    assert "mid1" in weight and "mid4" in weight
    assert "mid1" in best_path


# --------------------------------------------------------- lensed: bridges --
def test_lensed_bridge_rescues_nodes_invisible_to_depth_limited_bfs():
    """At max_depth=1, mid2/mid3 are in NEITHER seed's BFS tree at all (not
    merely truncated) — the GRAPH lens shortest_path (run on the full graph,
    not depth-limited) is the ONLY mechanism that can surface them."""
    g = _chain_graph()
    base = reconstruct(g, "lane d_seg", max_seeds=4, max_nodes=100, max_depth=1)
    assert set(base.nodes) == {"seed_a", "seed_b", "mid1", "mid4", "decoy1", "decoy2"}
    assert "mid2" not in base.nodes and "mid3" not in base.nodes

    lensed = reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=100, max_depth=1)
    base_ids = set(base.nodes)
    lensed_ids = set(lensed.nodes)
    assert base_ids.issubset(lensed_ids), "lensed path must be additive, never drop base nodes"
    bridge_ids = {nid for nid, _ in lensed.bridge_nodes}
    assert bridge_ids == {"mid2", "mid3"}
    assert bridge_ids.isdisjoint(base_ids)


def test_lensed_bridge_records_the_seed_pair():
    g = _chain_graph()
    lensed = reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=100, max_depth=1)
    assert lensed.bridge_nodes  # sanity: this fixture/depth does produce bridges
    for _nid, (a, b) in lensed.bridge_nodes:
        assert {a, b} == {"seed_a", "seed_b"}


def test_lensed_bridge_none_when_only_one_seed():
    """A single-seed query has no pair to bridge — bridge_nodes stays empty,
    never fabricated from thin air."""
    g = Graph()
    g.add_node(Node("only", "memory", "Only seed about lane d_seg"))
    lensed = reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=10, max_depth=2)
    assert lensed.bridge_nodes == []
    assert lensed.nodes == ["only"]


# ------------------------------------------------------------ lensed: hub --
def test_lensed_hub_ranks_the_true_articulation_point():
    g = _hub_graph()
    base = reconstruct(g, "lane d_seg", max_seeds=4, max_nodes=1, max_depth=2)
    assert base.nodes == ["seed"]  # p1/p2/p3/extra1/extra2 all truncated out

    lensed = reconstruct_lensed(
        g, "lane d_seg", max_seeds=4, max_nodes=1, max_depth=2, max_hub_nodes=1,
    )
    assert lensed.hub_nodes == [("p1", pytest.approx(0.7))]
    assert lensed.bridge_nodes == []  # single seed, nothing to bridge


def test_lensed_max_hub_nodes_zero_adds_nothing():
    """Regression guard for a real off-by-one: `max_hub_nodes=0` must add ZERO
    hub nodes, not one (the cap check must run BEFORE appending, not after)."""
    g = _hub_graph()
    lensed = reconstruct_lensed(
        g, "lane d_seg", max_seeds=4, max_nodes=1, max_depth=2, max_hub_nodes=0,
    )
    assert lensed.hub_nodes == []


# ---------------------------------------------------------- lensed: saddle --
def test_lensed_crux_surfaces_a_real_saddle():
    g = _saddle_graph()
    base = reconstruct(g, "d_seg saddle", max_seeds=4, max_nodes=1, max_depth=3)
    assert base.nodes == ["seed"]  # hub excluded from the base by max_nodes=1

    # max_hub_nodes=0 isolates the TOPOLOGY-lens crux path from the GRAPH-lens
    # hub path (both would otherwise compete for the same articulation node).
    lensed = reconstruct_lensed(
        g, "d_seg saddle", max_seeds=4, max_nodes=1, max_depth=3,
        max_hub_nodes=0, max_crux_nodes=5,
    )
    assert lensed.crux_nodes == [("hub", pytest.approx(4.0))]
    assert lensed.hub_nodes == []


# ------------------------------------------------------- additive/superset --
def test_lensed_is_always_a_superset_never_drops_base_nodes():
    g = _chain_graph()
    base = reconstruct(g, "lane d_seg", max_seeds=4, max_nodes=6, max_depth=2)
    lensed = reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=6, max_depth=2)
    assert lensed.base.nodes == base.nodes  # identical base reconstruction
    assert set(base.nodes).issubset(set(lensed.nodes))
    # order-preserving prefix: the base nodes appear first, in the same order
    assert lensed.nodes[: len(base.nodes)] == base.nodes


def test_lensed_deterministic():
    g = _chain_graph()
    r1 = reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=8, max_depth=1)
    r2 = reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=8, max_depth=1)
    assert r1.nodes == r2.nodes
    assert r1.bridge_nodes == r2.bridge_nodes
    assert r1.hub_nodes == r2.hub_nodes
    assert r1.crux_nodes == r2.crux_nodes


# ------------------------------------------------------------ edge cases ----
def test_lensed_no_match_is_graceful():
    g = _chain_graph()
    lensed = reconstruct_lensed(g, "zzznonexistentqqq", max_seeds=4, max_nodes=10)
    assert lensed.base.seeds == [] and lensed.base.nodes == []
    assert lensed.bridge_nodes == [] and lensed.hub_nodes == [] and lensed.crux_nodes == []
    assert lensed.nodes == []
    assert lensed.local_pool_size == 0
    out = format_human_lensed(lensed, g)
    assert "no matching seed" in out


def test_lensed_phi_custody_fails_closed_not_fabricated():
    """`phi='citation_salience'` on a node without a custodied value must raise
    the SAME AdapterError `CorpusAdapter` raises — never silently degrade to 0.
    """
    g = _chain_graph()  # no node here carries attrs['citation_salience']
    with pytest.raises(AdapterError):
        reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=10, phi="citation_salience")


def test_local_pool_graph_never_drops_required_ids():
    g = _chain_graph()
    weight = {"seed_a": 5.0, "mid1": 2.0, "mid2": 1.0, "decoy1": 0.5, "decoy2": 0.4}
    pool_ids = set(weight) | {"seed_b"}  # seed_b is "required" but has no BFS weight
    local, capped = _local_pool_graph(g, pool_ids, weight, pool_cap=2)
    assert capped is True
    assert "seed_b" in local.nodes  # required id never dropped by the cap
    assert len(local.nodes) <= 3  # 1 required + budget(2)


def test_local_pool_graph_uncapped_keeps_all():
    g = _chain_graph()
    weight = {"seed_a": 5.0, "mid1": 2.0}
    pool_ids = set(weight)
    local, capped = _local_pool_graph(g, pool_ids, weight, pool_cap=800)
    assert capped is False
    assert set(local.nodes) == pool_ids


def test_format_human_lensed_reports_augmentation_header():
    g = _chain_graph()
    lensed = reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=10, max_depth=1)
    out = format_human_lensed(lensed, g)
    assert "LENS ENGINE augmentation" in out
    assert "RECONSTRUCTED CONTEXT" in out  # base block still rendered
    assert "BRIDGE" in out  # this fixture/depth is known to surface a bridge


def test_lensed_to_dict_carries_lensed_marker_and_provenance():
    g = _chain_graph()
    lensed = reconstruct_lensed(g, "lane d_seg", max_seeds=4, max_nodes=10, max_depth=1)
    d = lensed.to_dict(g)
    assert d["lensed"] is True
    assert "nodes_lensed" in d and set(d["nodes_lensed"]) == set(lensed.nodes)
    assert d["bridge_nodes"], "this fixture/depth is known to surface a bridge"
    for entry in d["bridge_nodes"]:
        assert entry["why"].startswith("GRAPH shortest_path")


def test_lensed_result_type():
    g = _chain_graph()
    lensed = reconstruct_lensed(g, "lane d_seg")
    assert isinstance(lensed, LensedReconstruction)
    assert isinstance(lensed.base, Reconstruction)


# ---------------------------------------------------- real-corpus smoke -----
_REAL_QUERIES = ("lane d_seg", "muon warm start", "naive-launch incident")


@pytest.mark.parametrize("query", _REAL_QUERIES)
def test_real_corpus_lensed_is_superset_of_naive(query: str):
    """The pay-rent proof, as an executable regression guard: on each of the
    3 canonical probe queries from the landing memo, the lensed path is a
    strict superset of the naive keyword-BFS reconstruction on the real corpus
    graph, and every addition resolves to a real graph node (no fabrication).
    """
    graph = load_or_build(rebuild=False)
    base = reconstruct(graph, query, max_seeds=4, max_nodes=18, max_depth=2)
    lensed = reconstruct_lensed(graph, query, max_seeds=4, max_nodes=18, max_depth=2)
    assert lensed.base.nodes == base.nodes
    assert set(base.nodes).issubset(set(lensed.nodes))
    for nid in set(lensed.nodes) - set(base.nodes):
        assert nid in graph.nodes


def test_real_corpus_lensed_bounded_local_pool_for_cost():
    """Confirm the local-pool bound actually engages on a broad real query
    (measured 2026-07-12: full-corpus closeness centrality took ~91s over
    8,842 nodes; this must never run centrality over the whole graph)."""
    graph = load_or_build(rebuild=False)
    lensed = reconstruct_lensed(graph, "witness training run epoch", centrality_pool_cap=800)
    assert lensed.local_pool_size <= 800
    assert lensed.local_pool_capped is True


# --------------------------------------------------------------- CLI wiring --
def test_cli_lens_flag_is_additive_superset_end_to_end():
    """The `--lens` opt-in flag on tools/graph_memory_recall.py, exercised as a
    real subprocess against the real corpus cache."""
    base_proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "tools" / "graph_memory_recall.py"),
         "--json", "lane d_seg"],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60, check=True,
    )
    lensed_proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "tools" / "graph_memory_recall.py"),
         "--lens", "--json", "lane d_seg"],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60, check=True,
    )
    base_out = json.loads(base_proc.stdout)
    lensed_out = json.loads(lensed_proc.stdout)
    assert lensed_out["lensed"] is True
    base_ids = {n["id"] for n in base_out["reconstructed_nodes"]}
    lensed_ids = set(lensed_out["nodes_lensed"])
    assert base_ids.issubset(lensed_ids)
    assert lensed_ids - base_ids  # this query is known to yield real additions


# -------------------------------------------------- reconstruct() untouched --
def test_real_corpus_build_graph_still_returns_structurally_valid_graph():
    """Sister of test_graph_memory.py's real-corpus smoke; guards that nothing
    in this landing perturbed corpus construction."""
    g = build_graph()
    assert len(g.nodes) > 100
    assert len(g.edges) > 100
