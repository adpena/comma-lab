# SPDX-License-Identifier: MIT
"""Tests for tac.graph_memory.manifest (task #569, P0-1 source/completeness manifest).

The manifest turns a silent required-source collapse (the 07-19 worktree bug:
9,704/32,156 -> 3,157/4,856 when the memory source vanished) into a loud diff.
These tests pin: (1) per-source node attribution + required flags, (2) the
warn-only completeness comparison, (3) publish/load roundtrip + the wire-in.
"""
from __future__ import annotations

import json
from pathlib import Path

from tac.graph_memory.manifest import (
    CompletenessWarning,
    SourceManifest,
    SourceSnapshot,
    build_source_manifest,
    check_completeness,
    load_latest_manifest,
    publish_and_check,
    publish_manifest,
)
from tac.graph_memory.model import Edge, Graph, Node


# ---------------------------------------------------------------- fixtures ---
def _tiny_corpus(tmp_path: Path) -> dict:
    memdir = tmp_path / "memory"
    memdir.mkdir()
    m1 = memdir / "alpha.md"
    m1.write_text("---\nname: alpha\n---\nbody [[beta]]\n", encoding="utf-8")
    m2 = memdir / "beta.md"
    m2.write_text("---\nname: beta\n---\nbody\n", encoding="utf-8")
    (memdir / "MEMORY.md").write_text("# index\n", encoding="utf-8")  # excluded
    dag = tmp_path / "sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
    dag.write_text("## FEED-x (2026-07-02)\nbody\n", encoding="utf-8")
    eqs = tmp_path / "equations.jsonl"
    eqs.write_text(
        json.dumps({"equation_id": "e_v1", "equation_payload": {"equation_id": "e_v1"}}) + "\n",
        encoding="utf-8",
    )
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"task_id": "#569", "title": "t"}) + "\n", encoding="utf-8")
    deferral = tmp_path / "deferral.md"
    deferral.write_text("| D1 | owner | trigger | open |\n", encoding="utf-8")
    return {
        "memory_dir": memdir, "m1": m1, "m2": m2,
        "dag": dag, "eqs": eqs, "tasks": tasks, "deferral": deferral,
    }


def _graph_over(c: dict) -> Graph:
    """A graph whose node sources point at the tiny-corpus files (for attribution)."""
    g = Graph()
    g.add_node(Node(id="memory:alpha", ntype="memory", title="alpha", source=str(c["m1"])))
    g.add_node(Node(id="memory:beta", ntype="memory", title="beta", source=str(c["m2"])))
    g.add_node(Node(id="feed:x", ntype="finding", title="x",
                    source=f"{c['dag']}#L1-L2"))
    g.add_node(Node(id="eq:e_v1", ntype="equation", title="e_v1", source=str(c["eqs"])))
    g.add_node(Node(id="task:#569", ntype="task", title="t", source=str(c["tasks"])))
    g.add_node(Node(id="deferral:D1", ntype="deferral", title="D1", source=str(c["deferral"])))
    # a stub with no source — must NOT be attributed to any source.
    g.ensure_stub("ref:#999", ntype="entity", title="#999")
    g.add_edge(Edge("memory:alpha", "memory:beta", "links", str(c["m1"])))
    return g


def _manifest(c: dict, g: Graph) -> SourceManifest:
    return build_source_manifest(
        g, memory_dir=c["memory_dir"], dag_path=c["dag"], equations_path=c["eqs"],
        tasks_path=c["tasks"], deferral_path=c["deferral"], organ_path=c["dag"].parent / "absent.md",
    )


# ---------------------------------------------------------------- build ------
def test_build_manifest_attributes_nodes_per_source(tmp_path: Path) -> None:
    c = _tiny_corpus(tmp_path)
    m = _manifest(c, _graph_over(c))
    by = {s.name: s for s in m.sources}
    assert by["memory"].nodes_emitted == 2
    assert by["dag"].nodes_emitted == 1  # line anchor stripped for match
    assert by["equations"].nodes_emitted == 1
    assert by["tasks"].nodes_emitted == 1
    assert by["deferral"].nodes_emitted == 1
    # graph totals reflect the whole graph, including the unattributed stub.
    assert m.node_count == 7
    assert m.edge_count == 1


def test_build_manifest_required_flags_and_units(tmp_path: Path) -> None:
    c = _tiny_corpus(tmp_path)
    m = _manifest(c, _graph_over(c))
    by = {s.name: s for s in m.sources}
    assert by["memory"].required and by["dag"].required and by["equations"].required
    assert not by["tasks"].required and not by["deferral"].required
    assert not by["costate_organ"].required
    # memory dir scans 2 md files (MEMORY.md excluded); equations jsonl has 1 line.
    assert by["memory"].units_present == 2
    assert by["equations"].units_present == 1
    assert by["memory"].exists is True
    # the deliberately-absent organ ledger is reported not-existing with a reason.
    assert by["costate_organ"].exists is False
    assert by["costate_organ"].skip_reason == "file_absent"


def test_manifest_roundtrip_jsonl(tmp_path: Path) -> None:
    c = _tiny_corpus(tmp_path)
    m = _manifest(c, _graph_over(c))
    path = tmp_path / "manifest.jsonl"
    publish_manifest(m, path)
    loaded = load_latest_manifest(path)
    assert loaded is not None
    assert loaded.node_count == m.node_count
    assert {s.name for s in loaded.sources} == {s.name for s in m.sources}
    assert loaded.source("memory").nodes_emitted == 2


# ---------------------------------------------------------------- completeness
def _snap(name: str, *, required: bool, nodes: int, exists: bool = True) -> SourceSnapshot:
    return SourceSnapshot(
        name=name, canonical_root=f"/x/{name}", required=required, exists=exists,
        units_present=nodes, bytes_present=0, latest_source_mtime=0.0, nodes_emitted=nodes,
    )


def _mk(node_count: int, edge_count: int, snaps: list[SourceSnapshot]) -> SourceManifest:
    return SourceManifest(
        generated_at="t", git_rev="g", repo_root="/r", canonical_root="/r",
        node_count=node_count, edge_count=edge_count, sources=tuple(snaps),
    )


def test_completeness_no_prev_is_silent() -> None:
    curr = _mk(100, 200, [_snap("memory", required=True, nodes=50)])
    assert check_completeness(None, curr) == []


def test_completeness_node_and_edge_drop_warns() -> None:
    prev = _mk(100, 200, [_snap("memory", required=True, nodes=50)])
    curr = _mk(70, 100, [_snap("memory", required=True, nodes=50)])
    kinds = {w.kind for w in check_completeness(prev, curr)}
    assert "node_count_drop" in kinds  # 30% > 20%
    assert "edge_count_drop" in kinds  # 50% > 20%


def test_completeness_small_drop_below_threshold_silent() -> None:
    prev = _mk(100, 100, [_snap("memory", required=True, nodes=50)])
    curr = _mk(90, 95, [_snap("memory", required=True, nodes=45)])  # 10% / 5%
    assert check_completeness(prev, curr) == []


def test_completeness_required_source_collapse_warns() -> None:
    prev = _mk(100, 100, [_snap("memory", required=True, nodes=50)])
    curr = _mk(95, 100, [_snap("memory", required=True, nodes=0)])
    findings = check_completeness(prev, curr)
    assert any(
        w.kind == "required_source_collapsed" and w.source == "memory" for w in findings
    )


def test_completeness_optional_source_collapse_is_silent() -> None:
    prev = _mk(100, 100, [_snap("tasks", required=False, nodes=30)])
    curr = _mk(95, 100, [_snap("tasks", required=False, nodes=0)])
    # optional source dropping to zero (fresh worktree) is NOT a warning.
    assert [w for w in check_completeness(prev, curr) if w.source == "tasks"] == []


def test_completeness_required_source_absent_warns() -> None:
    prev = _mk(100, 100, [_snap("dag", required=True, nodes=40, exists=True)])
    curr = _mk(95, 100, [_snap("dag", required=True, nodes=40, exists=False)])
    findings = check_completeness(prev, curr)
    assert any(w.kind == "required_source_absent" and w.source == "dag" for w in findings)


# ---------------------------------------------------------------- wire-in -----
def test_publish_and_check_appends_and_warns_on_collapse(tmp_path: Path, capsys) -> None:
    path = tmp_path / "m.jsonl"
    big = Graph()
    for i in range(100):
        big.add_node(Node(id=f"n{i}", ntype="memory", title=str(i)))
    # first publish: no prev -> silent, one row appended.
    publish_and_check(big, path=path)
    assert len(path.read_text().splitlines()) == 1
    capsys.readouterr()

    small = Graph()
    for i in range(40):  # 60% node drop
        small.add_node(Node(id=f"n{i}", ntype="memory", title=str(i)))
    _, findings = publish_and_check(small, path=path)
    assert any(f.kind == "node_count_drop" for f in findings)
    err = capsys.readouterr().err
    assert "completeness WARN" in err
    assert len(path.read_text().splitlines()) == 2  # both rows persisted


def test_completeness_warning_to_dict_is_serializable() -> None:
    w = CompletenessWarning(kind="node_count_drop", source="", detail="d")
    assert json.loads(json.dumps(w.to_dict()))["kind"] == "node_count_drop"
