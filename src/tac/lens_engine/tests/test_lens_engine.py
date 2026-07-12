# SPDX-License-Identifier: MIT
"""Increment-1 contract tests for the unified Lens Engine."""

from __future__ import annotations

import operator
import subprocess
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from tac.graph_memory import Edge, Graph, Node
from tac.lens_engine import (
    TOPOLOGY,
    AdapterError,
    ComplexAdapter,
    ComplexEdge,
    ComplexElement,
    ComplexValidationError,
    CorpusAdapter,
    GraphLens,
    Lens,
    LensOperationError,
    QueryError,
    SpatialGeometry,
    T,
    TopologyLens,
    TypedRelation,
    WitnessAdapter,
    query,
)


def _force_setattr(value: object, name: str, replacement: Any) -> None:
    setattr(value, name, replacement)


def _branch_complex() -> T:
    return T(
        elements=(
            ComplexElement("a", "finding", phi=3.0),
            ComplexElement("s", "finding", phi=1.0),
            ComplexElement("b", "finding", phi=3.0),
            ComplexElement("l", "finding", phi=0.0),
        ),
        edges=(
            ComplexEdge("s", "a", "links", directed=False),
            ComplexEdge("s", "b", "links", directed=False),
            ComplexEdge("s", "l", "links", directed=False),
        ),
        metadata={"adapter": "fixture"},
    )


def _directed_complex() -> T:
    return T(
        elements=tuple(ComplexElement(value, "node") for value in "abcd"),
        edges=(
            ComplexEdge("a", "b", "link", weight=1.0),
            ComplexEdge("b", "c", "link", weight=1.0),
            ComplexEdge("a", "c", "link", weight=5.0),
        ),
    )


def _corpus_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node("a", "finding", "high a", attrs={"phi": 3.0, "date": "2026-07-10"}))
    graph.add_node(Node("s", "decision", "crux", attrs={"phi": 1.0}))
    graph.add_node(Node("b", "finding", "high b", attrs={"phi": 3.0}))
    graph.add_node(Node("l", "memory", "low", attrs={"phi": 0.0}))
    graph.add_edge(Edge("s", "a", "references", "memo.md:1"))
    graph.add_edge(Edge("s", "b", "references", "memo.md:2"))
    graph.add_edge(Edge("s", "l", "supersedes", "memo.md:3"))
    return graph


def _witness_arrays() -> tuple[np.ndarray, np.ndarray]:
    field = np.array(
        [
            [3.0, 2.0, 4.0],
            [2.0, 1.0, 3.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [2, 2, 1],
        ],
        dtype=np.int64,
    )
    return field, labels


# ---------------------------------------------------------------------------
# Core T + protocols
# ---------------------------------------------------------------------------


def test_t_is_frozen_and_exposes_named_maps() -> None:
    complex_ = T(
        elements=(
            ComplexElement(
                "n",
                "finding",
                phi=2.0,
                vec=(1.0, 2.0),
                scopes=frozenset({"formulation"}),
                spatial=SpatialGeometry("point", ((3.0, 4.0),)),
            ),
        ),
    )
    assert complex_.E[0].id == "n"
    assert complex_.Phi["n"] == 2.0
    assert complex_.S["n"] == frozenset({"formulation"})
    spatial = complex_.X["n"]
    assert spatial is not None
    assert spatial.coordinates == ((3.0, 4.0),)
    with pytest.raises(TypeError):
        operator.setitem(cast("MutableMapping[str, float]", complex_.Phi), "n", 9.0)
    with pytest.raises(FrozenInstanceError):
        _force_setattr(complex_, "elements", ())


def test_t_recursively_freezes_nested_mappings_and_array_backing() -> None:
    source: dict[str, Any] = {
        "nested": {"values": [1, 2]},
        "field": np.array([[1.0, 2.0]]),
    }
    complex_ = T(elements=(), metadata=source)
    source["nested"]["values"].append(3)
    source["field"][0, 0] = 9.0
    nested = complex_.metadata["nested"]
    field = complex_.metadata["field"]
    assert isinstance(nested, Mapping)
    assert isinstance(field, np.ndarray)
    assert nested["values"] == (1, 2)
    assert field[0, 0] == 1.0
    with pytest.raises(TypeError):
        operator.setitem(cast("MutableMapping[str, Any]", nested), "new", 1)
    with pytest.raises(ValueError):
        field.setflags(write=True)


@pytest.mark.parametrize(
    "factory,match",
    [
        (
            lambda: T(
                elements=(ComplexElement("x", "n"), ComplexElement("x", "n")),
            ),
            "duplicate element",
        ),
        (
            lambda: T(
                elements=(ComplexElement("x", "n"),),
                edges=(ComplexEdge("x", "missing", "link"),),
            ),
            "unknown endpoint",
        ),
        (
            lambda: T(
                elements=(ComplexElement("x", "n"),),
                relations=(TypedRelation("scope", ("missing",)),),
            ),
            "unknown member",
        ),
        (lambda: T(elements=(ComplexElement("x", "n", phi=float("nan")),)), "finite"),
    ],
)
def test_t_rejects_invalid_structures(factory, match: str) -> None:
    with pytest.raises(ComplexValidationError, match=match):
        factory()


def test_protocols_are_runtime_checkable() -> None:
    adapter = CorpusAdapter(Graph())
    assert isinstance(adapter, ComplexAdapter)
    assert isinstance(TopologyLens(), Lens)


def test_base_package_import_does_not_eagerly_require_scipy(tmp_path: Path) -> None:
    code = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'scipy' or name.startswith('scipy.'):
        raise ImportError('simulated base install')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import tac.lens_engine
assert sorted(tac.lens_engine.LENSES) == ['graph', 'spatial', 'statistics', 'topology']
empty = tac.lens_engine.T(elements=())
assert tac.lens_engine.query(empty, 'graph', 'components').value.components == ()
try:
    tac.lens_engine.query(empty, 'statistics', 'kde_density', values=(0.0, 1.0))
except tac.lens_engine.LensOperationError as exc:
    assert 'tac[analysis]' in str(exc)
else:
    raise AssertionError('SciPy-backed operation did not fail with an install hint')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Corpus adapter
# ---------------------------------------------------------------------------


def test_corpus_adapter_maps_graph_field_lineage_and_relations() -> None:
    adapter = CorpusAdapter(_corpus_graph(), phi="phi")
    complex_ = adapter.to_complex()
    assert len(complex_) == 4
    assert complex_.Phi["s"] == 1.0
    assert {edge.kind for edge in complex_.G} == {"references", "supersedes"}
    assert tuple(edge.kind for edge in complex_.L) == ("supersedes",)
    assert {relation.kind for relation in complex_.R} == {"references", "supersedes"}


def test_corpus_adapter_preserves_absent_embeddings_and_scopes() -> None:
    complex_ = CorpusAdapter(_corpus_graph(), phi="phi").to_complex()
    assert all(element.vec == () for element in complex_)
    assert all(element.scopes == frozenset() for element in complex_)
    assert complex_.metadata["missing_embedding_count"] == 4
    assert complex_.metadata["missing_verdict_scope_count"] == 4


def test_corpus_adapter_accepts_custodied_embeddings_and_scopes() -> None:
    adapter = CorpusAdapter(
        _corpus_graph(),
        phi="phi",
        embeddings_by_id={"s": (0.1, 0.2)},
        scopes_by_id={"s": ("formulation", "instance")},
    )
    element = adapter.to_complex().element("s")
    assert element.vec == (0.1, 0.2)
    assert element.scopes == frozenset({"formulation", "instance"})


def test_corpus_adapter_degree_and_recency_are_explicit_modes() -> None:
    graph = _corpus_graph()
    assert CorpusAdapter(graph).to_complex().Phi["s"] == 3.0
    with pytest.raises(AdapterError, match="no custodied recency"):
        CorpusAdapter(graph, phi="recency").to_complex()
    with pytest.raises(AdapterError, match="citation_salience"):
        CorpusAdapter(graph, phi="citation_salience").to_complex()
    dated = Graph()
    dated.add_node(Node("d", "finding", "dated", attrs={"date": "2026-07-12"}))
    assert CorpusAdapter(dated, phi="recency").to_complex().Phi["d"] > 0.0


def test_corpus_adapter_rejects_partial_phi_mapping() -> None:
    with pytest.raises(AdapterError, match="no value"):
        CorpusAdapter(_corpus_graph(), phi={"a": 1.0}).to_complex()


def test_corpus_adapter_rejects_nonnumeric_custodied_field() -> None:
    graph = Graph()
    graph.add_node(Node("a", "finding", "a", attrs={"citation_salience": "unknown"}))
    with pytest.raises(AdapterError, match="must be numeric"):
        CorpusAdapter(graph, phi="citation_salience").to_complex()


def test_corpus_adapter_rejects_dangling_graph_edge() -> None:
    graph = Graph()
    graph.add_node(Node("a", "finding", "a"))
    graph.add_edge(Edge("a", "missing", "references"))
    with pytest.raises(AdapterError, match="dangling"):
        CorpusAdapter(graph).to_complex()


def test_corpus_adapter_loads_existing_cache_without_rebuild(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes.jsonl"
    edges = tmp_path / "edges.jsonl"
    graph = _corpus_graph()
    graph.save(nodes, edges)
    adapter = CorpusAdapter.from_cache(nodes_path=nodes, edges_path=edges, phi="phi")
    assert len(adapter.to_complex()) == 4


def test_corpus_adapter_empty_graph_is_a_valid_edge_case() -> None:
    complex_ = CorpusAdapter(Graph()).to_complex()
    assert len(complex_) == 0
    assert complex_.G == ()


# ---------------------------------------------------------------------------
# Witness adapter
# ---------------------------------------------------------------------------


def test_witness_cell_adapter_reuses_connected_regions_and_rag() -> None:
    field, labels = _witness_arrays()
    complex_ = WitnessAdapter(field, labels, n_classes=3).to_complex()
    assert len(complex_) == 3
    assert all(element.kind == "cell" for element in complex_)
    assert {element.scopes for element in complex_} == {
        frozenset({"class:0"}),
        frozenset({"class:1"}),
        frozenset({"class:2"}),
    }
    assert all(edge.kind == "region_adjacency" and not edge.directed for edge in complex_.G)


def test_witness_pixel_adapter_builds_four_neighbour_grid() -> None:
    field = np.array([[0.0, 1.0], [2.0, 3.0]])
    labels = np.array([[0, 0], [1, 1]])
    complex_ = WitnessAdapter(field, labels, mode="pixels", n_classes=2).to_complex()
    assert len(complex_) == 4
    assert len(complex_.G) == 4
    assert complex_.element("pixel:1:1").phi == 3.0
    spatial = complex_.element("pixel:1:1").spatial
    assert spatial is not None
    assert spatial.axes == ("row", "column")


def test_witness_adapter_copies_inputs_and_exposes_read_only_metadata() -> None:
    field, labels = _witness_arrays()
    adapter = WitnessAdapter(field, labels, n_classes=3)
    field[:] = -10
    labels[:] = 0
    complex_ = adapter.to_complex()
    assert float(complex_.metadata["field"].max()) == 4.0
    assert int(complex_.metadata["labels"].max()) == 2
    with pytest.raises(ValueError):
        complex_.metadata["field"][0, 0] = 99
    with pytest.raises(ValueError):
        complex_.metadata["field"].setflags(write=True)


def test_witness_pixel_adapter_requires_opt_in_above_materialization_limit() -> None:
    field = np.zeros((257, 257), dtype=np.float32)
    labels = np.zeros((257, 257), dtype=np.int64)
    with pytest.raises(AdapterError, match="allow_large_pixels=True"):
        WitnessAdapter(field, labels, mode="pixels")


@pytest.mark.parametrize(
    "field,labels,match",
    [
        (np.zeros((2, 2, 1)), np.zeros((2, 2)), r"must be \(H,W\)"),
        (np.zeros((2, 2)), np.zeros((3, 2)), "equal non-empty"),
        (np.array([[np.nan]]), np.array([[0]]), "finite"),
        (np.zeros((1, 1)), np.array([[4]]), "class labels"),
    ],
)
def test_witness_adapter_rejects_invalid_arrays(field, labels, match: str) -> None:
    with pytest.raises(AdapterError, match=match):
        WitnessAdapter(field, labels, n_classes=3)


def test_witness_npz_adapter_selects_one_pair(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    fields = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    labels = np.stack([np.zeros((3, 3), dtype=np.int64), np.ones((3, 3), dtype=np.int64)])
    np.savez(path, margins=fields, lstars=labels)
    adapter = WitnessAdapter.from_npz(path, pair_index=1, n_classes=2)
    assert adapter.to_complex().metadata["pair_index"] == 1
    assert np.array_equal(adapter.to_complex().metadata["field"], fields[1])


def test_witness_npz_pair_loader_does_not_materialize_whole_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cache.npz"
    fields = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    labels = np.zeros((2, 3, 3), dtype=np.int64)
    np.savez(path, margins=fields, lstars=labels)

    def forbidden_numpy_load(*args, **kwargs):
        raise AssertionError("from_npz must use bounded member reads, not numpy.load")

    monkeypatch.setattr(np, "load", forbidden_numpy_load)
    adapter = WitnessAdapter.from_npz(path, pair_index=1)
    assert np.array_equal(adapter.to_complex().metadata["field"], fields[1])


def test_witness_npz_adapter_rejects_missing_key_and_bad_pair(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    np.savez(missing, margins=np.zeros((1, 2, 2)))
    with pytest.raises(AdapterError, match="missing key"):
        WitnessAdapter.from_npz(missing, pair_index=0)
    full = tmp_path / "full.npz"
    np.savez(full, margins=np.zeros((1, 2, 2)), lstars=np.zeros((1, 2, 2), dtype=int))
    with pytest.raises(AdapterError, match="outside margins range"):
        WitnessAdapter.from_npz(full, pair_index=2)


# ---------------------------------------------------------------------------
# Topology lens + required cross-surface worked queries
# ---------------------------------------------------------------------------


def test_worked_corpus_query_finds_branching_saddle() -> None:
    result = query(CorpusAdapter(_corpus_graph(), phi="phi"), TOPOLOGY, "saddles")
    assert result.adapter == "corpus"
    assert tuple(point.element_id for point in result.value) == ("s",)
    assert result.value[0].upper_link_components == 2


def test_worked_witness_query_finds_stable_margin_basin_with_same_lens() -> None:
    field, labels = _witness_arrays()
    adapter = WitnessAdapter(field, labels, n_classes=3)
    result = query(adapter, TOPOLOGY, "basins")
    assert result.adapter == "witness"
    assert result.lens == "topology"
    assert sum(len(basin.member_ids) for basin in result.value) == len(adapter.to_complex())
    assert all(basin.mode == "max" for basin in result.value)


def test_topology_peaks_strictly_distinguishes_plateaus() -> None:
    complex_ = T(
        elements=(ComplexElement("a", "n", phi=1), ComplexElement("b", "n", phi=1)),
        edges=(ComplexEdge("a", "b", "link", directed=False),),
    )
    assert len(query(complex_, "topology", "peaks").value) == 2
    assert query(complex_, "topology", "peaks", strict=True).value == ()


def test_topology_watershed_has_two_basins_and_a_separatrix() -> None:
    result = query(_branch_complex(), "topology", "watershed")
    assert len(result.value.basins) == 2
    assert len(result.value.separatrices) == 1
    assert result.value.separatrices[0].kind == "separatrix"


def test_topology_routes_terminate_at_basins() -> None:
    result = query(_branch_complex(), "topology", "routes", starts=("l", "b"))
    assert result.value[0].element_ids[0] == "l"
    assert result.value[0].element_ids[-1] in {"a", "b"}
    assert result.value[1].element_ids == ("b",)


def test_topology_persistence_returns_finite_and_essential_pairs() -> None:
    result = query(_branch_complex(), "topology", "persistence")
    pairs = result.value
    assert any(pair.essential and np.isinf(pair.persistence) for pair in pairs)
    assert any(not pair.essential and pair.persistence >= 0 for pair in pairs)
    assert set(result.element_ids) <= {"a", "s", "b", "l"}
    assert result.element_ids


def test_topology_empty_complex_is_an_explicit_edge_case() -> None:
    result = query(T(elements=()), "topology", "basins")
    assert result.value == ()
    assert result.metadata["empty"] is True
    watershed = query(T(elements=()), "topology", "watershed")
    assert watershed.value.basins == ()
    assert watershed.value.separatrices == ()
    with pytest.raises(LensOperationError, match="unexpected topology"):
        query(T(elements=()), "topology", "basins", invented=True)


def test_query_rejects_unknown_lens_operation_and_mode() -> None:
    with pytest.raises(QueryError, match="unknown lens"):
        query(_branch_complex(), "vector", "knn")
    with pytest.raises(QueryError, match="no operation"):
        query(_branch_complex(), "topology", "unknown")
    with pytest.raises(LensOperationError, match="mode"):
        query(_branch_complex(), "topology", "basins", mode="sideways")


# ---------------------------------------------------------------------------
# Graph lens
# ---------------------------------------------------------------------------


def test_graph_bfs_honours_direction_and_depth() -> None:
    complex_ = _directed_complex()
    assert query(complex_, "graph", "bfs", start="c", directed=True).value.order == ("c",)
    assert query(complex_, "graph", "bfs", start="c").value.order == ("c", "a", "b")
    assert query(complex_, "graph", "bfs", start="a", max_depth=1).value.order == (
        "a",
        "b",
        "c",
    )


def test_graph_dfs_is_cycle_safe_and_deterministic() -> None:
    complex_ = T(
        elements=tuple(ComplexElement(value, "n") for value in "abc"),
        edges=(
            ComplexEdge("a", "b", "link"),
            ComplexEdge("b", "c", "link"),
            ComplexEdge("c", "a", "link"),
        ),
    )
    assert query(complex_, GraphLens(), "dfs", start="a", directed=True).value.order == (
        "a",
        "b",
        "c",
    )


def test_graph_shortest_path_uses_edge_weights_and_reports_unreachable() -> None:
    result = query(_directed_complex(), "graph", "shortest_path", start="a", target="c", directed=True)
    assert result.value.nodes == ("a", "b", "c")
    assert result.value.distance == 2.0
    unreachable = query(
        _directed_complex(),
        "graph",
        "shortest_path",
        start="a",
        target="d",
        directed=True,
    )
    assert not unreachable.value.reachable
    assert np.isinf(unreachable.value.distance)


def test_graph_shortest_path_zero_weight_cycle_remains_acyclic() -> None:
    complex_ = T(
        elements=tuple(ComplexElement(value, "n") for value in "abc"),
        edges=(
            ComplexEdge("a", "b", "link", directed=False, weight=0.0),
            ComplexEdge("b", "c", "link", directed=False, weight=0.0),
            ComplexEdge("c", "a", "link", directed=False, weight=0.0),
        ),
    )
    result = query(complex_, "graph", "shortest_path", start="a", target="c")
    assert result.value.reachable
    assert result.value.nodes[0] == "a"
    assert result.value.nodes[-1] == "c"
    assert len(result.value.nodes) <= len(complex_)


@pytest.mark.parametrize("method", ["degree", "closeness", "betweenness"])
def test_graph_centrality_methods_rank_the_chain_hub(method: str) -> None:
    result = query(_directed_complex(), "graph", "centrality", method=method)
    scores = dict(result.value.scores)
    assert scores["b"] >= scores["d"]


def test_graph_centrality_weight_mode_is_explicit() -> None:
    unweighted = query(
        _directed_complex(),
        "graph",
        "centrality",
        method="degree",
        weighted=False,
    ).value
    weighted = query(
        _directed_complex(),
        "graph",
        "centrality",
        method="degree",
        weighted=True,
    ).value
    assert dict(weighted.scores)["a"] > dict(unweighted.scores)["a"]
    assert weighted.weighted
    with pytest.raises(LensOperationError, match="weighted betweenness"):
        query(
            _directed_complex(),
            "graph",
            "centrality",
            method="betweenness",
            weighted=True,
        )


def test_graph_components_cover_empty_singleton_and_disconnected() -> None:
    assert query(T(elements=()), "graph", "components").value.components == ()
    singleton = T(elements=(ComplexElement("x", "n"),))
    assert query(singleton, "graph", "components").value.components == (("x",),)
    assert query(_directed_complex(), "graph", "components").value.components == (
        ("a", "b", "c"),
        ("d",),
    )
    with pytest.raises(LensOperationError, match="requires directed=False"):
        query(_directed_complex(), "graph", "components", directed=True)


def test_graph_community_is_deterministic_and_keeps_isolate() -> None:
    first = query(_directed_complex(), "graph", "community").value
    second = query(_directed_complex(), "graph", "community").value
    assert first == second
    assert ("d",) in first.communities


def test_graph_rejects_bad_start_method_and_directed_community() -> None:
    with pytest.raises(LensOperationError, match="unknown BFS"):
        query(_directed_complex(), "graph", "bfs", start="missing")
    with pytest.raises(LensOperationError, match="centrality method"):
        query(_directed_complex(), "graph", "centrality", method="pagerank")
    with pytest.raises(LensOperationError, match="directed=False"):
        query(_directed_complex(), "graph", "community", directed=True)


# ---------------------------------------------------------------------------
# Spatial lens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "point,inside,on_boundary",
    [((1.0, 1.0), True, False), ((3.0, 1.0), False, False), ((0.0, 1.0), True, True)],
)
def test_spatial_point_in_polygon(point, inside: bool, on_boundary: bool) -> None:
    polygon = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    result = query(T(elements=()), "spatial", "point_in_polygon", point=point, polygon=polygon)
    assert result.value.inside is inside
    assert result.value.on_boundary is on_boundary


def test_spatial_polygon_boundary_can_be_excluded() -> None:
    polygon = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    result = query(
        T(elements=()),
        "spatial",
        "point_in_polygon",
        point=(0.0, 1.0),
        polygon=polygon,
        include_boundary=False,
    )
    assert not result.value.inside
    assert result.value.on_boundary


def test_spatial_iou_handles_masks_and_empty_union() -> None:
    left = np.array([[True, False], [True, False]])
    right = np.array([[True, True], [False, False]])
    result = query(T(elements=()), "spatial", "iou_overlap", left=left, right=right)
    assert result.value.intersection == 1
    assert result.value.union == 3
    assert result.value.iou == pytest.approx(1 / 3)
    empty = np.zeros((2, 2), dtype=bool)
    assert query(T(elements=()), "spatial", "iou_overlap", left=empty, right=empty).value.iou == 1.0


def test_spatial_iou_rejects_shape_mismatch() -> None:
    with pytest.raises(LensOperationError, match="equal shape"):
        query(
            T(elements=()),
            "spatial",
            "iou_overlap",
            left=np.zeros((2, 2)),
            right=np.zeros((3, 2)),
        )


def test_spatial_distance_uses_explicit_coordinates_and_element_geometry() -> None:
    complex_ = T(
        elements=(
            ComplexElement("a", "point", spatial=SpatialGeometry("point", ((0.0, 0.0),))),
            ComplexElement("b", "point", spatial=SpatialGeometry("point", ((3.0, 4.0),))),
        ),
    )
    assert query(complex_, "spatial", "distance", left_id="a", right_id="b").value.value == 5.0
    assert query(
        complex_,
        "spatial",
        "distance",
        left=(0, 0),
        right=(3, 4),
        metric="manhattan",
    ).value.value == 7.0


def test_spatial_distance_rejects_dimension_mismatch() -> None:
    with pytest.raises(LensOperationError, match="equal dimensions"):
        query(T(elements=()), "spatial", "distance", left=(0, 0), right=(1, 2, 3))


def test_spatial_distance_rejects_incompatible_coordinate_axes() -> None:
    complex_ = T(
        elements=(
            ComplexElement(
                "grid",
                "point",
                spatial=SpatialGeometry(
                    "point",
                    ((0.0, 0.0),),
                    axes=("row", "column"),
                ),
            ),
            ComplexElement(
                "cartesian",
                "point",
                spatial=SpatialGeometry("point", ((3.0, 4.0),), axes=("x", "y")),
            ),
        ),
    )
    with pytest.raises(LensOperationError, match="explicit coordinate transform"):
        query(complex_, "spatial", "distance", left_id="grid", right_id="cartesian")


def test_spatial_laguerre_reuses_power_assignment_and_weights_move_cells() -> None:
    base = query(
        T(elements=()),
        "spatial",
        "laguerre_cells",
        sites=((0, 0), (0, 2)),
        weights=(0, 0),
        classes=(4, 7),
        shape=(1, 3),
    ).value
    shifted = query(
        T(elements=()),
        "spatial",
        "laguerre_cells",
        sites=((0, 0), (0, 2)),
        weights=(0, 5),
        classes=(4, 7),
        shape=(1, 3),
    ).value
    assert base.class_labels.tolist() == [[4, 4, 7]]
    assert shifted.class_labels.tolist() == [[7, 7, 7]]
    assert not shifted.class_labels.flags.writeable


def test_spatial_laguerre_rejects_empty_diagram() -> None:
    with pytest.raises(LensOperationError, match="Laguerre"):
        query(
            T(elements=()),
            "spatial",
            "laguerre_cells",
            sites=np.empty((0, 2)),
            shape=(2, 2),
        )


def test_spatial_laguerre_rejects_malformed_sites_with_typed_error() -> None:
    with pytest.raises(LensOperationError, match=r"shape \(K,2\)"):
        query(
            T(elements=()),
            "spatial",
            "laguerre_cells",
            sites=(1, 2, 3),
            shape=(2, 2),
        )


# ---------------------------------------------------------------------------
# Statistics lens
# ---------------------------------------------------------------------------


def test_statistics_kde_density_is_finite_and_positive() -> None:
    result = query(
        _branch_complex(),
        "statistics",
        "kde_density",
        points=(0.0, 1.0, 2.0),
    )
    assert len(result.value.density) == 3
    assert all(np.isfinite(result.value.density))
    assert all(value > 0 for value in result.value.density)


def test_statistics_kde_rejects_constant_or_singleton_samples() -> None:
    with pytest.raises(LensOperationError, match="distinct"):
        query(T(elements=()), "statistics", "kde_density", values=(1.0, 1.0))
    with pytest.raises(LensOperationError, match="at least 2"):
        query(T(elements=()), "statistics", "kde_density", values=(1.0,))


def test_statistics_distribution_drift_distinguishes_shift() -> None:
    identical = query(
        T(elements=()),
        "statistics",
        "distribution_drift",
        baseline=(0, 1, 2),
        current=(0, 1, 2),
    ).value
    shifted = query(
        T(elements=()),
        "statistics",
        "distribution_drift",
        baseline=(0, 1, 2),
        current=(3, 4, 5),
    ).value
    assert identical.wasserstein == 0.0
    assert identical.ks_statistic == 0.0
    assert shifted.wasserstein > 0.0
    assert shifted.mean_shift == 3.0


def test_statistics_distribution_drift_rejects_empty_series() -> None:
    with pytest.raises(LensOperationError, match="at least 1"):
        query(
            T(elements=()),
            "statistics",
            "distribution_drift",
            baseline=(),
            current=(1,),
        )


def test_statistics_anisotropy_reuses_structure_tensor_and_is_read_only() -> None:
    field = np.tile(np.arange(8, dtype=np.float64), (8, 1))
    complex_ = T(elements=(), metadata={"field": field})
    result = query(complex_, "statistics", "anisotropy", sigma=1.0)
    assert result.value.dH.shape == field.shape
    assert np.isfinite(result.value.dH).all()
    assert not result.value.dH.flags.writeable
    with pytest.raises(ValueError):
        result.value.dH.setflags(write=True)
    assert "structure_tensor_dH" in result.provenance[0]


def test_statistics_anisotropy_rejects_missing_or_nonfinite_field() -> None:
    with pytest.raises(LensOperationError, match="requires field"):
        query(T(elements=()), "statistics", "anisotropy")
    with pytest.raises(LensOperationError, match="finite 2-D"):
        query(
            T(elements=()),
            "statistics",
            "anisotropy",
            field=np.array([[np.nan]]),
        )


def test_statistics_persistence_delegates_to_topology() -> None:
    topology = query(_branch_complex(), "topology", "persistence")
    statistics = query(_branch_complex(), "statistics", "persistence")
    assert statistics.value == topology.value
    assert "delegates" in statistics.provenance[-1]


def test_statistics_change_point_finds_clean_mean_shift_and_uses_ema() -> None:
    result = query(
        T(elements=()),
        "statistics",
        "change_point",
        values=(0, 0, 0, 0, 10, 10, 10, 10),
        min_score=5,
    ).value
    assert result.index == 4
    assert result.detected
    assert result.left_mean == 0.0
    assert result.right_mean == 10.0


def test_statistics_change_point_does_not_flag_constant_series_at_zero_threshold() -> None:
    result = query(
        T(elements=()),
        "statistics",
        "change_point",
        values=(2, 2, 2, 2, 2, 2),
    ).value
    assert result.score == 0.0
    assert not result.detected


def test_statistics_change_point_rejects_insufficient_series() -> None:
    with pytest.raises(LensOperationError, match="at least 4"):
        query(T(elements=()), "statistics", "change_point", values=(1, 2, 3))
