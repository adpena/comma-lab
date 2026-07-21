# SPDX-License-Identifier: MIT
"""Link-hygiene regression tests for graph-memory task #594."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from tac.graph_memory.build import (
    parse_memory_files,
    parse_memory_indexes,
    parse_research_memos,
    parse_section_documents,
    wikilink_target,
)
from tac.graph_memory.link_hygiene import measure_link_hygiene
from tac.graph_memory.model import Graph
from tac.recall_evidence import RecallEvidence


def test_wikilink_filter_rejects_pure_numeric() -> None:
    assert wikilink_target("910") is None


def test_wikilink_filter_rejects_numeric_comma_tuple() -> None:
    assert wikilink_target("910,0,582") is None
    assert wikilink_target("(1.0, 0.0, -1.0, 0.5)") is None


def test_wikilink_filter_rejects_named_coordinate_tuple() -> None:
    assert wikilink_target("x=910,y=0,w=582") is None


def test_wikilink_filter_preserves_semantic_numeric_slug() -> None:
    assert wikilink_target("L85") == "L85"
    assert wikilink_target("task-594|display") == "task-594"


def _note(path: Path, name: str, body: str = "") -> None:
    path.write_text(f"---\nname: {name}\ntype: feedback\n---\n{body}\n", encoding="utf-8")


def test_memory_index_links_create_typed_discoverability_edges(tmp_path: Path) -> None:
    mdir = tmp_path / "memory"
    mdir.mkdir()
    _note(mdir / "alpha.md", "alpha")
    _note(mdir / "beta.md", "beta")
    (mdir / "MEMORY.md").write_text(
        "# Index\n- [Alpha](alpha.md)\n- [Beta](beta.md)\n", encoding="utf-8"
    )
    (mdir / "MEMORY_topic_cluster_2026.md").write_text(
        "# Cluster\n[Alpha](alpha.md)\n", encoding="utf-8"
    )
    graph = Graph()
    parse_memory_files(graph, mdir)
    assert parse_memory_indexes(graph, mdir) == 3
    assert any(e.etype == "indexed_by" and e.dst == "index:MEMORY.md"
               for e in graph.out_edges("memory:alpha"))
    assert any(e.etype == "indexed_by" and e.dst == "index:MEMORY_topic_cluster_2026.md"
               for e in graph.out_edges("memory:alpha"))


def test_memory_filename_alias_resolves_prefix_and_separator_variants(tmp_path: Path) -> None:
    mdir = tmp_path / "memory"
    mdir.mkdir()
    _note(mdir / "feedback_alpha_lane_dseg.md", "alpha-lane-dseg")
    graph = Graph()
    parse_memory_files(graph, mdir)
    for alias in ("feedback_alpha_lane_dseg", "alpha-lane-dseg"):
        alias_id = "memory:" + alias
        if alias_id != "memory:alpha-lane-dseg":
            assert any(e.etype == "aliases" and e.dst == "memory:alpha-lane-dseg"
                       for e in graph.out_edges(alias_id))


def test_doctrine_heading_and_declared_anchor_resolve_as_aliases(tmp_path: Path) -> None:
    doctrine = tmp_path / "CLAUDE.md"
    doctrine.write_text(
        "## MLX portable-local-substrate authority — NON-NEGOTIABLE\n"
        "Rule body. Anchors: `proactive-recall-consult-own-research-before-concluding`.\n",
        encoding="utf-8",
    )
    graph = Graph()
    assert parse_section_documents(graph, [doctrine]) == 1
    for slug in (
        "mlx-portable-local-substrate-authority",
        "proactive-recall-consult-own-research-before-concluding",
    ):
        alias_id = "memory:" + slug
        assert any(e.etype == "aliases" for e in graph.out_edges(alias_id))


def test_research_memo_entity_families_are_typed(tmp_path: Path) -> None:
    rdir = tmp_path / "research"
    rdir.mkdir()
    (rdir / "b.md").write_text("# B memo\n", encoding="utf-8")
    (rdir / "a.md").write_text(
        "# A memo\n[B](b.md) analytic_example_law_v1 #594 FEED-alpha "
        "lane_link_hygiene Catalog #332 [[semantic-memory]]\n",
        encoding="utf-8",
    )
    graph = Graph()
    assert parse_research_memos(graph, {"analytic_example_law_v1"}, rdir) == 2
    etypes = {e.etype for e in graph.out_edges("research:a.md")}
    assert {"memo_link", "equation_ref", "task_ref", "feed_ref", "lane_ref", "catalog_ref"} <= etypes
    assert graph.nodes["research:b.md"].source.endswith("b.md")


def test_hygiene_measurement_counts_indexed_notes_and_filtered_fp(tmp_path: Path) -> None:
    mdir = tmp_path / "memory"
    rdir = tmp_path / "research"
    mdir.mkdir()
    rdir.mkdir()
    _note(mdir / "alpha.md", "alpha")
    _note(mdir / "beta.md", "beta")
    (mdir / "MEMORY.md").write_text("[Alpha](alpha.md)\n[Beta](beta.md)\n", encoding="utf-8")
    (rdir / "memo.md").write_text("[[alpha]] [[910,0,582]] [[missing]]\n", encoding="utf-8")
    graph = Graph()
    parse_memory_files(graph, mdir)
    parse_memory_indexes(graph, mdir)
    payload = measure_link_hygiene(graph, memory_dir=mdir, research_dir=rdir)
    assert payload["files_scanned"] == 4
    assert payload["raw_wikilinks"] == 3
    assert payload["false_positive_filtered"] == 1
    assert payload["truly_unwritten"] == 1
    assert payload["unreachable_orphans"] == 0


def test_unreferenced_filename_alias_does_not_fake_note_reachability(tmp_path: Path) -> None:
    mdir = tmp_path / "memory"
    rdir = tmp_path / "research"
    mdir.mkdir()
    rdir.mkdir()
    _note(mdir / "feedback_unreferenced_note.md", "unreferenced-note")
    graph = Graph()
    parse_memory_files(graph, mdir)
    payload = measure_link_hygiene(graph, memory_dir=mdir, research_dir=rdir)
    assert payload["unreachable_orphans"] == 1


def _load_suggester_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "suggest_sister_links.py"
    spec = importlib.util.spec_from_file_location("suggest_sister_links", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sister_candidate_filter_is_advisory_unique_and_nonself(tmp_path: Path) -> None:
    mdir = tmp_path / "memory"
    mdir.mkdir()
    note = mdir / "alpha.md"
    _note(note, "alpha", "Already cites [[beta]].")
    _note(mdir / "beta.md", "beta")
    _note(mdir / "gamma.md", "gamma")
    (mdir / "MEMORY_archive_full_2026.md").write_text("# Archive\n", encoding="utf-8")
    rows = [
        RecallEvidence(
            "corpus",
            "memory",
            "index",
            str(mdir / "MEMORY_archive_full_2026.md"),
            4,
            "",
            1,
            0.4,
            ("corpus",),
        ),
        RecallEvidence("graph", "memory", "memory:alpha", "", 3, "", 1, 0.3, ("graph",)),
        RecallEvidence("graph", "memory", "memory:beta", "", 2, "", 2, 0.2, ("graph",)),
        RecallEvidence("graph", "memory", "memory:gamma", "", 1, "", 3, 0.1, ("graph",)),
    ]
    module = _load_suggester_module()
    before = note.read_bytes()
    got = module.rank_sister_candidates(note, rows, top_k=5, memory_dir=mdir)
    assert [row["slug"] for row in got] == ["gamma"]
    assert note.read_bytes() == before
