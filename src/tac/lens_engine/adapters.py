# SPDX-License-Identifier: MIT
"""Authority-preserving corpus and witness adapters for the Lens Engine."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast
from zipfile import BadZipFile, ZipFile

import numpy as np

from tac.graph_memory import Graph, Node, cache_paths

from .core import (
    AdapterError,
    ComplexEdge,
    ComplexElement,
    SpatialGeometry,
    T,
    TimeInterval,
    TypedRelation,
)

CorpusField = str | Mapping[str, float] | Callable[[Node], float]


def _class_masks_from_argmax(labels: np.ndarray, n_classes: int) -> np.ndarray:
    try:
        from tac.boundary_math.bitmask_dseg import class_masks_from_argmax
    except ImportError as exc:
        raise AdapterError(
            "WitnessAdapter requires the optional analysis dependencies; "
            "install tac[analysis]"
        ) from exc
    return class_masks_from_argmax(labels, n_classes)


def _build_region_adjacency_graph(labels: np.ndarray, n_classes: int) -> Any:
    try:
        from tac.boundary_math.partition import build_region_adjacency_graph
    except ImportError as exc:
        raise AdapterError(
            "WitnessAdapter cell mode requires the optional analysis dependencies; "
            "install tac[analysis]"
        ) from exc
    return build_region_adjacency_graph(labels, n_classes)


def _read_npz_pair(path: Path, key: str, pair_index: int) -> np.ndarray:
    """Read one C-contiguous pair directly from an NPY member inside an NPZ.

    ``numpy.load(npz)[key][pair]`` first materializes the complete member.  The
    canonical 600-pair witness cache stores ordinary C-order NPY members, so the
    selected leading-axis slice is one contiguous byte range.  Reading that
    range through :class:`zipfile.ZipExtFile` keeps peak memory proportional to
    one pair even when the member itself is nearly a gigabyte.  Compressed NPZ
    members remain bounded-memory (``ZipExtFile.seek`` may decompress preceding
    bytes, but it does not materialize the full array).
    """

    member = key if key.endswith(".npy") else f"{key}.npy"
    try:
        with ZipFile(path) as archive:
            try:
                handle = archive.open(member)
            except KeyError as exc:
                raise AdapterError(f"witness NPZ missing key: {key!r}") from exc
            with handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise AdapterError(
                        f"witness NPZ key {key!r} uses unsupported NPY version {version}"
                    )
                dtype = np.dtype(dtype)
                if dtype.hasobject:
                    raise AdapterError(f"witness NPZ key {key!r} has forbidden object dtype")
                if fortran_order:
                    raise AdapterError(
                        f"witness NPZ key {key!r} is Fortran-order; pass a bounded array explicitly"
                    )
                if len(shape) == 2:
                    if pair_index != 0:
                        raise AdapterError("2-D witness NPZ accepts pair_index=0 only")
                    pair_shape = tuple(int(value) for value in shape)
                    byte_offset = handle.tell()
                elif len(shape) == 3:
                    if pair_index < 0 or pair_index >= int(shape[0]):
                        raise AdapterError(
                            f"pair_index {pair_index} outside {key} range [0,{shape[0]})"
                        )
                    pair_shape = (int(shape[1]), int(shape[2]))
                    pair_bytes = math.prod(pair_shape) * dtype.itemsize
                    byte_offset = handle.tell() + pair_index * pair_bytes
                else:
                    raise AdapterError(
                        f"witness NPZ key {key!r} must be (H,W) or (N,H,W); got {shape}"
                    )
                pair_bytes = math.prod(pair_shape) * dtype.itemsize
                handle.seek(byte_offset)
                payload = handle.read(pair_bytes)
                if len(payload) != pair_bytes:
                    raise AdapterError(
                        f"witness NPZ key {key!r} truncated: read {len(payload)}/{pair_bytes} bytes"
                    )
                return np.frombuffer(payload, dtype=dtype).copy().reshape(pair_shape)
    except AdapterError:
        raise
    except (BadZipFile, OSError, ValueError) as exc:
        raise AdapterError(f"failed to read witness NPZ {path}: {exc}") from exc


def _vector(value: Any, *, owner: str) -> tuple[float, ...]:
    if value is None:
        return ()
    try:
        out = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{owner} embedding must be a finite numeric vector") from exc
    if not all(math.isfinite(item) for item in out):
        raise AdapterError(f"{owner} embedding must be a finite numeric vector")
    return out


def _scopes(value: Any) -> frozenset[str]:
    if value is None or value == "":
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    try:
        return frozenset(str(item) for item in value)
    except TypeError as exc:
        raise AdapterError("verdict scopes must be a string or iterable of strings") from exc


def _explicit_interval(attrs: Mapping[str, Any]) -> TimeInterval | None:
    start = attrs.get("timestamp") or attrs.get("date") or attrs.get("created_at")
    if not start:
        return None
    end = attrs.get("end_timestamp") or attrs.get("end_date")
    return TimeInterval(str(start), str(end) if end else None)


class CorpusAdapter:
    """Expose a :class:`tac.graph_memory.Graph` as ``T``.

    The current canonical cache contains no embeddings and no structured
    ``verdict_scope`` field.  Missing source fields therefore remain empty.
    Callers may inject separately-custodied embeddings/scopes; the adapter never
    hashes prose or invents vectors as a fallback.
    """

    name = "corpus"

    def __init__(
        self,
        graph: Graph,
        *,
        phi: CorpusField = "degree",
        embeddings_by_id: Mapping[str, Sequence[float]] | None = None,
        scopes_by_id: Mapping[str, str | Sequence[str]] | None = None,
        source: str = "injected tac.graph_memory.Graph",
    ) -> None:
        self._graph = graph
        self._phi = phi
        self._embeddings = dict(embeddings_by_id or {})
        self._scopes = dict(scopes_by_id or {})
        self._source = source
        self._complex: T | None = None

    @classmethod
    def from_cache(
        cls,
        *,
        nodes_path: str | Path | None = None,
        edges_path: str | Path | None = None,
        **kwargs: Any,
    ) -> CorpusAdapter:
        """Load the existing deterministic cache without rebuilding or writing it."""

        default_nodes, default_edges = cache_paths()
        nodes = Path(nodes_path) if nodes_path is not None else default_nodes
        edges = Path(edges_path) if edges_path is not None else default_edges
        if not nodes.is_file() or not edges.is_file():
            raise AdapterError(
                f"graph-memory cache missing: nodes={nodes.is_file()} edges={edges.is_file()}"
            )
        try:
            graph = Graph.load(nodes, edges)
        except (OSError, ValueError, KeyError) as exc:
            raise AdapterError(f"failed to load graph-memory cache: {exc}") from exc
        return cls(graph, source=f"Graph.load({nodes}, {edges})", **kwargs)

    def _field_value(self, node: Node) -> float:
        field = self._phi
        if isinstance(field, str):
            if field == "degree":
                value = float(self._graph.degree(node.id))
            elif field == "citation_salience":
                if field not in node.attrs:
                    raise AdapterError(
                        f"node {node.id!r} has no custodied citation_salience value"
                    )
                raw = node.attrs[field]
                try:
                    value = float(raw)
                except (TypeError, ValueError) as exc:
                    raise AdapterError(
                        f"node {node.id!r} citation_salience must be numeric; got {raw!r}"
                    ) from exc
            elif field == "recency":
                date = node.attrs.get("date") or node.attrs.get("timestamp")
                if not date:
                    raise AdapterError(f"node {node.id!r} has no custodied recency value")
                else:
                    try:
                        value = float(dt.date.fromisoformat(str(date)[:10]).toordinal())
                    except ValueError as exc:
                        raise AdapterError(
                            f"node {node.id!r} has invalid ISO recency field: {date!r}"
                        ) from exc
            elif field in {"relevance", "phi"}:
                if field not in node.attrs:
                    raise AdapterError(f"node {node.id!r} has no custodied {field} value")
                raw = node.attrs[field]
                try:
                    value = float(raw)
                except (TypeError, ValueError) as exc:
                    raise AdapterError(
                        f"node {node.id!r} {field} must be numeric; got {raw!r}"
                    ) from exc
            else:
                raise AdapterError(
                    "corpus phi must be 'citation_salience', 'degree', 'recency', "
                    "'relevance', 'phi', a mapping, or a callable"
                )
        elif isinstance(field, Mapping):
            field_map = cast("Mapping[str, float]", field)
            if node.id not in field_map:
                raise AdapterError(f"corpus phi mapping has no value for node {node.id!r}")
            try:
                value = float(field_map[node.id])
            except (TypeError, ValueError) as exc:
                raise AdapterError(
                    f"corpus phi mapping value for node {node.id!r} must be numeric"
                ) from exc
        elif callable(field):
            try:
                value = float(field(node))
            except (TypeError, ValueError) as exc:
                raise AdapterError(f"corpus phi callable failed for node {node.id!r}") from exc
        else:
            raise AdapterError("invalid corpus phi specification")
        if not math.isfinite(value):
            raise AdapterError(f"node {node.id!r} phi must be finite")
        return value

    def to_complex(self) -> T:
        if self._complex is not None:
            return self._complex
        known = frozenset(self._graph.nodes)
        elements: list[ComplexElement] = []
        absent_embeddings = 0
        absent_scopes = 0
        for node_id in sorted(self._graph.nodes):
            node = self._graph.nodes[node_id]
            vector_source = self._embeddings.get(node_id, node.attrs.get("embedding"))
            vector = _vector(vector_source, owner=f"node {node_id!r}")
            scope_source = self._scopes.get(node_id, node.attrs.get("verdict_scope"))
            scopes = _scopes(scope_source)
            absent_embeddings += not bool(vector)
            absent_scopes += not bool(scopes)
            elements.append(
                ComplexElement(
                    id=node.id,
                    kind=node.ntype,
                    phi=self._field_value(node),
                    vec=vector,
                    scopes=scopes,
                    interval=_explicit_interval(node.attrs),
                    attrs={
                        "title": node.title,
                        "summary": node.summary,
                        "source": node.source,
                        "graph_attrs": dict(node.attrs),
                    },
                )
            )
        edges: list[ComplexEdge] = []
        lineage: list[ComplexEdge] = []
        relations: list[TypedRelation] = []
        for key in sorted(self._graph.edges):
            source_edge = self._graph.edges[key]
            missing = {source_edge.src, source_edge.dst} - known
            if missing:
                raise AdapterError(
                    f"graph-memory edge {source_edge.etype!r} has dangling endpoint(s): "
                    f"{sorted(missing)}"
                )
            edge = ComplexEdge(
                source=source_edge.src,
                target=source_edge.dst,
                kind=source_edge.etype,
                directed=True,
                attrs={"source": source_edge.source},
            )
            edges.append(edge)
            relations.append(
                TypedRelation(
                    kind=source_edge.etype,
                    members=(source_edge.src, source_edge.dst),
                    attrs={"source": source_edge.source},
                )
            )
            if source_edge.etype == "supersedes":
                lineage.append(edge)
        phi_mode = self._phi if isinstance(self._phi, str) else type(self._phi).__name__
        self._complex = T(
            elements=tuple(elements),
            edges=tuple(edges),
            lineage=tuple(lineage),
            relations=tuple(relations),
            metadata={
                "adapter": self.name,
                "source": self._source,
                "phi_source": str(phi_mode),
                "missing_embedding_count": absent_embeddings,
                "missing_verdict_scope_count": absent_scopes,
                "embedding_absence_is_explicit": True,
                "scope_absence_is_explicit": True,
            },
        )
        return self._complex


class WitnessAdapter:
    """Expose one cached witness margin/loss field and class partition as ``T``."""

    name = "witness"

    def __init__(
        self,
        field: np.ndarray,
        labels: np.ndarray,
        *,
        pair_index: int = 0,
        mode: Literal["cells", "pixels"] = "cells",
        n_classes: int = 5,
        source: str = "injected witness arrays",
        telemetry: Mapping[str, Any] | None = None,
        pixel_materialization_limit: int = 65_536,
        allow_large_pixels: bool = False,
    ) -> None:
        try:
            field_array = np.asarray(field, dtype=np.float64)
            labels_array = np.asarray(labels, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise AdapterError("witness field and labels must be numeric arrays") from exc
        if field_array.ndim != 2 or labels_array.ndim != 2:
            raise AdapterError(
                f"witness field/labels must be (H,W); got {field_array.shape}/{labels_array.shape}"
            )
        if field_array.shape != labels_array.shape or not field_array.size:
            raise AdapterError("witness field and labels must have equal non-empty shape")
        if not np.isfinite(field_array).all():
            raise AdapterError("witness field must contain only finite values")
        if mode not in {"cells", "pixels"}:
            raise AdapterError("witness mode must be 'cells' or 'pixels'")
        if int(pair_index) < 0:
            raise AdapterError("pair_index must be non-negative")
        if int(n_classes) <= 0:
            raise AdapterError("n_classes must be positive")
        if int(pixel_materialization_limit) <= 0:
            raise AdapterError("pixel_materialization_limit must be positive")
        if (
            mode == "pixels"
            and field_array.size > int(pixel_materialization_limit)
            and not allow_large_pixels
        ):
            raise AdapterError(
                f"pixel mode would materialize {field_array.size} elements, above limit "
                f"{int(pixel_materialization_limit)}; use cells mode or set "
                "allow_large_pixels=True explicitly"
            )
        try:
            class_masks = _class_masks_from_argmax(labels_array, int(n_classes))
        except ValueError as exc:
            raise AdapterError(f"invalid witness class labels: {exc}") from exc
        self._field = field_array.copy()
        self._labels = labels_array.copy()
        self._class_masks = class_masks
        self._field.setflags(write=False)
        self._labels.setflags(write=False)
        self._class_masks.setflags(write=False)
        self._pair_index = int(pair_index)
        self._mode = mode
        self._n_classes = int(n_classes)
        self._source = source
        self._telemetry = dict(telemetry or {})
        self._pixel_materialization_limit = int(pixel_materialization_limit)
        self._allow_large_pixels = bool(allow_large_pixels)
        self._complex: T | None = None

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        pair_index: int,
        field_key: str = "margins",
        labels_key: str = "lstars",
        **kwargs: Any,
    ) -> WitnessAdapter:
        """Select one pair from an NPZ cache without modifying the archive.

        NPZ members are not ordinary memory-mapped arrays.  This method is
        intentionally explicit about pair selection; callers handling the live
        multi-gigabyte cache may instead pass already-bounded read-only arrays to
        the primary constructor to control peak memory.
        """

        source_path = Path(path)
        if not source_path.is_file():
            raise AdapterError(f"witness NPZ does not exist: {source_path}")
        field = _read_npz_pair(source_path, field_key, pair_index)
        labels = _read_npz_pair(source_path, labels_key, pair_index)
        return cls(
            field,
            labels,
            pair_index=pair_index,
            source=f"read-only NPZ pair {pair_index}: {source_path}",
            **kwargs,
        )

    def _cell_complex(self) -> T:
        rag = _build_region_adjacency_graph(self._labels, self._n_classes)
        region_of = np.array(rag.region_of, copy=True)
        region_of.setflags(write=False)
        elements: list[ComplexElement] = []
        relations: list[TypedRelation] = []
        for region_id in sorted(rag.regions):
            region = rag.regions[region_id]
            rows = region.coords[0]
            cols = region.coords[1]
            centroid = (float(rows.mean()), float(cols.mean()))
            element_id = f"cell:{region_id}"
            elements.append(
                ComplexElement(
                    id=element_id,
                    kind="cell",
                    phi=float(self._field[rows, cols].mean()),
                    vec=centroid,
                    scopes=frozenset({f"class:{region.label}"}),
                    spatial=SpatialGeometry(
                        kind="region_cell",
                        coordinates=(centroid,),
                        axes=("row", "column"),
                        bounds=(
                            float(rows.min()),
                            float(cols.min()),
                            float(rows.max()),
                            float(cols.max()),
                        ),
                        attrs={"support": "metadata.region_of"},
                    ),
                    interval=TimeInterval(f"pair:{self._pair_index}"),
                    attrs={"class_id": region.label, "pixels": region.pixels},
                )
            )
            relations.append(
                TypedRelation(
                    kind="class_membership",
                    members=(element_id,),
                    attrs={"class_id": region.label},
                )
            )
        edges: list[ComplexEdge] = []
        for region_id in sorted(rag.adjacency):
            for neighbour in sorted(rag.adjacency[region_id]):
                if neighbour <= region_id:
                    continue
                edges.append(
                    ComplexEdge(
                        source=f"cell:{region_id}",
                        target=f"cell:{neighbour}",
                        kind="region_adjacency",
                        directed=False,
                    )
                )
        return T(
            elements=tuple(elements),
            edges=tuple(edges),
            relations=tuple(relations),
            metadata=self._metadata(region_of=region_of),
        )

    def _pixel_complex(self) -> T:
        height, width = self._field.shape
        elements: list[ComplexElement] = []
        relations: list[TypedRelation] = []
        edges: list[ComplexEdge] = []
        for row in range(height):
            for column in range(width):
                element_id = f"pixel:{row}:{column}"
                label = int(self._labels[row, column])
                elements.append(
                    ComplexElement(
                        id=element_id,
                        kind="pixel",
                        phi=float(self._field[row, column]),
                        vec=(float(row), float(column)),
                        scopes=frozenset({f"class:{label}"}),
                        spatial=SpatialGeometry(
                            kind="grid_pixel",
                            coordinates=((float(row), float(column)),),
                            axes=("row", "column"),
                        ),
                        interval=TimeInterval(f"pair:{self._pair_index}"),
                        attrs={"class_id": label},
                    )
                )
                relations.append(
                    TypedRelation(
                        kind="class_membership",
                        members=(element_id,),
                        attrs={"class_id": label},
                    )
                )
                if column > 0:
                    edges.append(
                        ComplexEdge(
                            source=f"pixel:{row}:{column - 1}",
                            target=element_id,
                            kind="grid_adjacency",
                            directed=False,
                        )
                    )
                if row > 0:
                    edges.append(
                        ComplexEdge(
                            source=f"pixel:{row - 1}:{column}",
                            target=element_id,
                            kind="grid_adjacency",
                            directed=False,
                        )
                    )
        region_of = np.arange(height * width, dtype=np.int64).reshape(height, width)
        region_of.setflags(write=False)
        return T(
            elements=tuple(elements),
            edges=tuple(edges),
            relations=tuple(relations),
            metadata=self._metadata(region_of=region_of),
        )

    def _metadata(self, *, region_of: np.ndarray) -> Mapping[str, Any]:
        return {
            "adapter": self.name,
            "source": self._source,
            "pair_index": self._pair_index,
            "mode": self._mode,
            "field": self._field,
            "labels": self._labels,
            "class_masks": self._class_masks,
            "region_of": region_of,
            "telemetry": dict(self._telemetry),
            "field_authority": "cached margin/loss field; analysis only",
            "pixel_materialization_limit": self._pixel_materialization_limit,
            "allow_large_pixels": self._allow_large_pixels,
        }

    def to_complex(self) -> T:
        if self._complex is None:
            self._complex = self._cell_complex() if self._mode == "cells" else self._pixel_complex()
        return self._complex
