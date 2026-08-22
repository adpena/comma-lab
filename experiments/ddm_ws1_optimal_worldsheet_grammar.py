#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Price the full-partition persistent worldsheet grammar required by DDM-WS1.

This is a scorer-free, CPU-only n600 coder experiment.  The sender represents
the complete five-class partition as its labelled dual-grid curves, carries
persistent curve identities and explicit split/merge parents, and predicts
those curves with the already-carried G1 Pose6/xi ground homography.  Four
registered specialized grammars are raced against the same generic receiver:
the #234 parametric Road/Lane polynomial, a Road/Undrivable horizon polyline,
a shared Undrivable/Movable silhouette template, and a shared static hood
curve.  Every real coder payload is retained before its byte count is used.

The exact receiver below reconstructs the class field from one seed label and
the labelled boundary graph.  It does not read source labels, scorer state, or
ordinal boundary correspondences.  The G1 cross-pair motion remains the
documented nearest-target-pair proxy; this script does not upgrade that proxy
to an exact trajectory claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

_REPO: Final = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "src", _REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_bd1_class_field_receiver as bd1
import ddm_ws0_worldsheet_grammar_price as ws0

from tac.boundary_math.analytic_lane_render_band import (
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    deserialize_lane_band_any,
    rasterize_lane_coverage_range_dependent,
    serialize_lane_band_rd_tracked,
)
from tac.boundary_math.lane_sdf_component import cluster_lane_lines, fit_lane_line, image_to_ground
from tac.boundary_math.lane_track_and_smooth import coherent_slot_pack
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom, homography_from_xi_numpy
from tac.optimization.predictor_upgrade_xi_chart import load_g1_worldsheet_motion, relative_adjacent_xi

SCHEMA: Final = "ddm_ws1_optimal_worldsheet_grammar.v1"
AXIS: Final = "[macOS-CPU advisory, scorer-free n600 coder]"
MAGIC: Final = b"WSO1"
VERSION: Final = 1
N_PAIRS: Final = ws0.N_PAIRS
HEIGHT: Final = ws0.HEIGHT
WIDTH: Final = ws0.WIDTH
N_CLASSES: Final = ws0.N_CLASSES
TOTAL_CELLS: Final = ws0.TOTAL_CELLS
TOLERANCE_DSEG: Final = 0.00116
TOLERANCE_CELLS: Final = math.floor(TOLERANCE_DSEG * TOTAL_CELLS)
EDGE_PAIRS: Final = ws0.EDGE_PAIRS
EDGE_TO_ID: Final = ws0.EDGE_TO_ID
CLASS_NAMES: Final = ws0.CLASS_NAMES
EDGE_ID_MATRIX: Final = np.full((N_CLASSES, N_CLASSES), -1, dtype=np.int8)
for _edge_id, (_left, _right) in enumerate(EDGE_PAIRS):
    EDGE_ID_MATRIX[_left, _right] = _edge_id
    EDGE_ID_MATRIX[_right, _left] = _edge_id
V_SITES: Final = HEIGHT * (WIDTH - 1)
H_SITES: Final = (HEIGHT - 1) * WIDTH
N_SITES: Final = V_SITES + H_SITES
EXPECTED_CACHE_SHA256: Final = ws0.EXPECTED_CACHE_SHA256
EXPECTED_WS0_SHA256: Final = "275bf0c9a7a390de40f8313e7595f01ca68ddf23ac9ef3ac658921b62b8d39de"
EXPECTED_ES1_SHA256: Final = ws0.EXPECTED_ES1_SHA256
WS0_ORDINAL_TEMPORAL_BYTES: Final = 318_885
PP1_BYTES: Final = 173_616
LOSSLESS_FALSIFIER: Final = 130_000
TOLERANCE_FALSIFIER: Final = 110_000
DEFAULT_CACHE: Final = ws0.DEFAULT_CACHE
DEFAULT_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_ws1_optimal_worldsheet_grammar/retained")
DEFAULT_WS0_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_ws0_worldsheet_grammar_price/retained")
DEFAULT_WS0_MEMO: Final = _REPO / ".omx/research/ddm_ws0_worldsheet_grammar_price_20260821.md"
DEFAULT_ES1: Final = ws0.DEFAULT_ES1
CODECS: Final = ("brotli-q11", "lzma1-raw", "smevr-r7-nibble")
SPECIAL_STRATA: Final = {
    "Road<->Lane": "road_lane_polynomial",
    "Road<->Undrivable": "road_undrivable_horizon",
    "Undrivable<->Movable": "movable_silhouette_template",
    "Road<->MyCar": "shared_static_hood",
    "Movable<->MyCar": "shared_static_hood",
}


class WorldsheetOptimalError(RuntimeError):
    """A receiver, provenance, storage, or byte-accounting check failed closed."""


@dataclass(frozen=True, slots=True)
class Curve:
    identity: int
    sites: np.ndarray


@dataclass(frozen=True, slots=True)
class FrameLineage:
    curves: tuple[Curve, ...]
    parents: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class StreamRace:
    name: str
    records_sha256: str
    canonical_raw_bytes: int
    payloads: dict[str, bytes]
    winner: str


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    semantics: str
    records: dict[str, tuple[bytes, ...]]
    envelope: bytes
    races: dict[str, StreamRace]
    diagnostics: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n").encode())


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def edge_name(edge_id: int) -> str:
    left, right = EDGE_PAIRS[edge_id]
    return f"{CLASS_NAMES[left]}<->{CLASS_NAMES[right]}"


def put_uleb(output: bytearray, value: int) -> None:
    ws0.put_uleb(output, int(value))


def get_uleb(payload: bytes, offset: int) -> tuple[int, int]:
    return ws0.get_uleb(payload, offset)


def encode_sorted(values: Iterable[int]) -> bytes:
    ordered = sorted({int(value) for value in values})
    output = bytearray()
    put_uleb(output, len(ordered))
    prior = -1
    for value in ordered:
        if not prior < value < N_SITES:
            raise WorldsheetOptimalError("site sequence is not canonical")
        put_uleb(output, value - prior - 1)
        prior = value
    return bytes(output)


def decode_sorted(payload: bytes, offset: int = 0) -> tuple[np.ndarray, int]:
    count, offset = get_uleb(payload, offset)
    output = np.empty(count, dtype=np.uint32)
    prior = -1
    for index in range(count):
        gap, offset = get_uleb(payload, offset)
        value = prior + int(gap) + 1
        if not prior < value < N_SITES:
            raise WorldsheetOptimalError("decoded site escapes the registered grid")
        output[index] = value
        prior = value
    return output, offset


def xor_sites(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(sorted(set(map(int, left)) ^ set(map(int, right))), dtype=np.uint32)


def frame_boundary_sites(labels: np.ndarray) -> tuple[np.ndarray, ...]:
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.shape != (HEIGHT, WIDTH) or np.any(labels >= N_CLASSES):
        raise WorldsheetOptimalError(f"unexpected label field {labels.shape}")
    buckets: list[list[np.ndarray]] = [[] for _ in EDGE_PAIRS]
    for lhs, rhs, base, stride in (
        (labels[:, :-1], labels[:, 1:], 0, WIDTH - 1),
        (labels[:-1, :], labels[1:, :], V_SITES, WIDTH),
    ):
        yy, xx = np.nonzero(lhs != rhs)
        ids = EDGE_ID_MATRIX[lhs[yy, xx], rhs[yy, xx]]
        sites = (base + yy.astype(np.int64) * stride + xx.astype(np.int64)).astype(np.uint32)
        for edge_id in range(len(EDGE_PAIRS)):
            buckets[edge_id].append(sites[ids == edge_id])
    return tuple(
        np.sort(np.concatenate(parts)).astype(np.uint32) if any(part.size for part in parts)
        else np.empty(0, dtype=np.uint32)
        for parts in buckets
    )


def site_endpoints(site: int) -> tuple[int, int]:
    if site < V_SITES:
        y, x0 = divmod(site, WIDTH - 1)
        x = x0 + 1
        return y * (WIDTH + 1) + x, (y + 1) * (WIDTH + 1) + x
    y, x = divmod(site - V_SITES, WIDTH)
    row = y + 1
    return row * (WIDTH + 1) + x, row * (WIDTH + 1) + x + 1


def decompose_sites(sites: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split an edge stratum into deterministic dual-grid connected curves."""

    ordered = np.asarray(sites, dtype=np.uint32)
    if ordered.size == 0:
        return ()
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    endpoints: dict[int, int] = {}
    for index, raw_site in enumerate(ordered.tolist()):
        for endpoint in site_endpoints(int(raw_site)):
            prior = endpoints.get(endpoint)
            if prior is None:
                endpoints[endpoint] = index
                continue
            left, right = find(index), find(prior)
            if left != right:
                parent[max(left, right)] = min(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, raw_site in enumerate(ordered.tolist()):
        groups[find(index)].append(int(raw_site))
    return tuple(
        np.asarray(values, dtype=np.uint32)
        for values in sorted(groups.values(), key=lambda values: (values[0], len(values)))
    )


def warp_sites(sites: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Advect dual-grid segments by G1 H, preserving their local orientation."""

    output: set[int] = set()
    for raw_site in np.asarray(sites, dtype=np.uint32).tolist():
        site = int(raw_site)
        if site < V_SITES:
            y, x0 = divmod(site, WIDTH - 1)
            point = np.array([x0 + 1.0, y + 0.5, 1.0], dtype=np.float64)
            projected = homography @ point
            if projected[2] > 1e-9:
                x = round(projected[0] / projected[2])
                yy = round(projected[1] / projected[2] - 0.5)
                if 1 <= x < WIDTH and 0 <= yy < HEIGHT:
                    output.add(yy * (WIDTH - 1) + x - 1)
                    continue
        else:
            y0, x = divmod(site - V_SITES, WIDTH)
            point = np.array([x + 0.5, y0 + 1.0, 1.0], dtype=np.float64)
            projected = homography @ point
            if projected[2] > 1e-9:
                xx = round(projected[0] / projected[2] - 0.5)
                y = round(projected[1] / projected[2])
                if 0 <= xx < WIDTH and 1 <= y < HEIGHT:
                    output.add(V_SITES + (y - 1) * WIDTH + xx)
                    continue
        output.add(site)  # registered persist fallback for invalid projection
    return np.asarray(sorted(output), dtype=np.uint32)


def match_lineage(
    current: tuple[np.ndarray, ...],
    previous: tuple[Curve, ...],
    homography: np.ndarray | None,
    next_identity: int,
) -> tuple[FrameLineage, int]:
    predicted = {
        curve.identity: warp_sites(curve.sites, homography) if homography is not None else curve.sites
        for curve in previous
    }
    inverted: dict[int, set[int]] = defaultdict(set)
    for identity, sites in predicted.items():
        for site in sites.tolist():
            inverted[int(site)].add(identity)
    used: set[int] = set()
    curves: list[Curve] = []
    all_parents: list[tuple[int, ...]] = []
    sizes = {curve.identity: int(curve.sites.size) for curve in previous}
    for sites in current:
        overlap: dict[int, int] = defaultdict(int)
        for site in sites.tolist():
            for identity in inverted.get(int(site), ()):
                overlap[identity] += 1
        parents = tuple(
            sorted(
                identity for identity, count in overlap.items()
                if count >= max(2, math.ceil(0.08 * min(int(sites.size), sizes[identity])))
            )
        )
        reusable = next((identity for identity in parents if identity not in used), None)
        if reusable is None:
            reusable = next_identity
            next_identity += 1
        used.add(reusable)
        curves.append(Curve(reusable, sites))
        all_parents.append(parents)
    return FrameLineage(tuple(curves), tuple(all_parents)), next_identity


def encode_lineage(frame: FrameLineage) -> bytes:
    output = bytearray()
    put_uleb(output, len(frame.curves))
    prior_identity = 0
    for curve, parents in zip(frame.curves, frame.parents, strict=True):
        put_uleb(output, ws0.zigzag(curve.identity - prior_identity))
        prior_identity = curve.identity
        put_uleb(output, len(parents))
        prior_parent = 0
        for parent in parents:
            put_uleb(output, ws0.zigzag(parent - prior_parent))
            prior_parent = parent
    return bytes(output)


def decode_lineage(payload: bytes) -> tuple[tuple[int, tuple[int, ...]], ...]:
    count, offset = get_uleb(payload, 0)
    output: list[tuple[int, tuple[int, ...]]] = []
    prior_identity = 0
    for _ in range(count):
        delta, offset = get_uleb(payload, offset)
        identity = prior_identity + ws0.unzigzag(delta)
        prior_identity = identity
        nparents, offset = get_uleb(payload, offset)
        parents: list[int] = []
        prior_parent = 0
        for _ in range(nparents):
            raw, offset = get_uleb(payload, offset)
            prior_parent += ws0.unzigzag(raw)
            parents.append(prior_parent)
        output.append((identity, tuple(parents)))
    if offset != len(payload) or len({row[0] for row in output}) != len(output):
        raise WorldsheetOptimalError("lineage record is noncanonical")
    return tuple(output)


def predictor_from_lineage(
    previous: dict[int, np.ndarray],
    entries: tuple[tuple[int, tuple[int, ...]], ...],
    homography: np.ndarray | None,
    *,
    use_identity: bool,
) -> np.ndarray:
    identities = set(previous) if not use_identity else {parent for _, parents in entries for parent in parents}
    parts = [warp_sites(previous[identity], homography) if homography is not None else previous[identity]
             for identity in sorted(identities) if identity in previous]
    return np.asarray(sorted(set().union(*(set(map(int, part)) for part in parts))), dtype=np.uint32)


def seed_record(labels: np.ndarray) -> bytes:
    return bytes([int(np.asarray(labels)[0, 0])])


def reconstruct_labels(seed: int, strata: Sequence[np.ndarray]) -> np.ndarray:
    vertical = np.full((HEIGHT, WIDTH - 1), -1, dtype=np.int8)
    horizontal = np.full((HEIGHT - 1, WIDTH), -1, dtype=np.int8)
    for edge_id, sites in enumerate(strata):
        sites = np.asarray(sites, dtype=np.uint32)
        v = sites[sites < V_SITES].astype(np.int64)
        h = (sites[sites >= V_SITES] - V_SITES).astype(np.int64)
        vy, vx = np.divmod(v, WIDTH - 1)
        hy, hx = np.divmod(h, WIDTH)
        if np.any(vertical[vy, vx] >= 0) or np.any(horizontal[hy, hx] >= 0):
            raise WorldsheetOptimalError("multiple class pairs claim one dual-grid edge")
        vertical[vy, vx] = edge_id
        horizontal[hy, hx] = edge_id
    output = np.empty((HEIGHT, WIDTH), dtype=np.uint8)
    first = int(seed)
    for y in range(HEIGHT):
        if y:
            edge_id = int(horizontal[y - 1, 0])
            if edge_id >= 0:
                edge = EDGE_PAIRS[edge_id]
                if first not in edge:
                    raise WorldsheetOptimalError("horizontal seed edge is inconsistent")
                first = edge[1] if first == edge[0] else edge[0]
        edge_positions = np.flatnonzero(vertical[y] >= 0)
        left_x = 0
        label = first
        for x0 in edge_positions.tolist():
            output[y, left_x:x0 + 1] = label
            edge = EDGE_PAIRS[int(vertical[y, x0])]
            if label not in edge:
                raise WorldsheetOptimalError("vertical edge is inconsistent with the receiver state")
            label = edge[1] if label == edge[0] else edge[0]
            left_x = x0 + 1
        output[y, left_x:] = label
    low = np.minimum(output[:-1], output[1:])
    high = np.maximum(output[:-1], output[1:])
    expected = np.full_like(horizontal, -1)
    for edge_id, edge in enumerate(EDGE_PAIRS):
        expected[(low == edge[0]) & (high == edge[1])] = edge_id
    if not np.array_equal(horizontal, expected):
        raise WorldsheetOptimalError("decoded horizontal curves do not close the partition")
    return output


def decode_frame_sites(payload: bytes) -> tuple[np.ndarray, ...]:
    records = ws0.unpack_records(payload)
    if len(records) != len(EDGE_PAIRS):
        raise WorldsheetOptimalError("boundary-site checkpoint has the wrong stratum count")
    output = []
    for record in records:
        sites, offset = decode_sorted(record)
        if offset != len(record):
            raise WorldsheetOptimalError("boundary-site checkpoint record has trailing bytes")
        output.append(sites)
    return tuple(output)


def load_or_build_site_frames(
    labels: Sequence[np.ndarray], stage_dir: Path
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Resume the full dual-grid extraction from one atomic frame checkpoint."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for pair, frame in enumerate(labels):
        path = stage_dir / f"frame_{pair:04d}.records"
        try:
            sites = decode_frame_sites(path.read_bytes())
        except (OSError, ValueError, WorldsheetOptimalError):
            sites = frame_boundary_sites(frame)
            atomic_bytes(path, ws0.pack_records(tuple(encode_sorted(row) for row in sites)))
        output.append(sites)
    manifest = {
        "schema": "ddm_ws1_boundary_site_stage.v1", "frames": N_PAIRS,
        "frame_files": N_PAIRS,
        "aggregate_bytes": sum((stage_dir / f"frame_{pair:04d}.records").stat().st_size for pair in range(N_PAIRS)),
    }
    atomic_json(stage_dir / "STAGE_COMPLETE.json", manifest)
    return tuple(output)


def make_lineages(
    frame_sites: Sequence[tuple[np.ndarray, ...]], homographies: Sequence[np.ndarray], stage_dir: Path
) -> tuple[tuple[FrameLineage, ...], ...]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    by_edge: list[list[FrameLineage]] = [[] for _ in EDGE_PAIRS]
    for edge_id in range(len(EDGE_PAIRS)):
        path = stage_dir / f"edge_{edge_id:02d}.records"
        try:
            event_rows = ws0.unpack_records(path.read_bytes())
            if len(event_rows) != N_PAIRS:
                raise WorldsheetOptimalError("lineage checkpoint frame count mismatch")
            restored = []
            for pair, record in enumerate(event_rows):
                entries = decode_lineage(record)
                components = decompose_sites(frame_sites[pair][edge_id])
                if len(entries) != len(components):
                    raise WorldsheetOptimalError("lineage checkpoint component count mismatch")
                restored.append(FrameLineage(
                    tuple(Curve(identity, sites) for (identity, _parents), sites in zip(entries, components, strict=True)),
                    tuple(parents for _identity, parents in entries),
                ))
            by_edge[edge_id] = restored
            continue
        except (OSError, ValueError, WorldsheetOptimalError):
            pass
        previous: tuple[Curve, ...] = ()
        next_identity = 1
        for pair in range(N_PAIRS):
            components = decompose_sites(frame_sites[pair][edge_id])
            homography = homographies[pair] if pair else None
            lineage, next_identity = match_lineage(components, previous, homography, next_identity)
            by_edge[edge_id].append(lineage)
            previous = lineage.curves
        atomic_bytes(path, ws0.pack_records(tuple(encode_lineage(frame) for frame in by_edge[edge_id])))
    atomic_json(stage_dir / "STAGE_COMPLETE.json", {
        "schema": "ddm_ws1_lineage_stage.v1", "strata": len(EDGE_PAIRS), "frames": N_PAIRS,
        "explicit_split_merge_lineage": True,
    })
    return tuple(tuple(rows) for rows in by_edge)


def generic_records(
    frame_sites: Sequence[tuple[np.ndarray, ...]],
    seeds: Sequence[bytes],
    homographies: Sequence[np.ndarray],
    lineages: Sequence[Sequence[FrameLineage]],
    *,
    use_identity: bool,
    use_advection: bool,
) -> tuple[dict[str, tuple[bytes, ...]], dict[str, Any]]:
    records: dict[str, tuple[bytes, ...]] = {"shared_topology_seed": tuple(seeds)}
    diagnostics: dict[str, Any] = {"strata": {}}
    for edge_id in range(len(EDGE_PAIRS)):
        name = edge_name(edge_id)
        event_rows: list[bytes] = []
        residual_rows: list[bytes] = []
        previous: dict[int, np.ndarray] = {}
        events = {"birth": 0, "persist": 0, "split": 0, "merge": 0, "death": 0}
        events_by_split = {
            "pairs_0_399": {"birth": 0, "persist": 0, "split": 0, "merge": 0, "death": 0},
            "pairs_400_599": {"birth": 0, "persist": 0, "split": 0, "merge": 0, "death": 0},
        }
        for pair in range(N_PAIRS):
            lineage = lineages[edge_id][pair]
            event_rows.append(encode_lineage(lineage) if use_identity else b"")
            entries = tuple((curve.identity, parents) for curve, parents in zip(lineage.curves, lineage.parents, strict=True))
            homography = homographies[pair] if use_advection and pair else None
            prediction = predictor_from_lineage(previous, entries, homography, use_identity=use_identity)
            target = frame_sites[pair][edge_id]
            residual_rows.append(encode_sorted(xor_sites(target, prediction)))
            parent_use: dict[int, int] = defaultdict(int)
            split_name = "pairs_0_399" if pair < 400 else "pairs_400_599"
            for _identity, parents in entries:
                for parent in parents:
                    parent_use[parent] += 1
                if not parents:
                    events["birth"] += 1
                    events_by_split[split_name]["birth"] += 1
                elif len(parents) > 1:
                    events["merge"] += 1
                    events_by_split[split_name]["merge"] += 1
                else:
                    events["persist"] += 1
                    events_by_split[split_name]["persist"] += 1
            splits = sum(max(0, count - 1) for count in parent_use.values())
            deaths = sum(identity not in parent_use for identity in previous)
            events["split"] += splits
            events["death"] += deaths
            events_by_split[split_name]["split"] += splits
            events_by_split[split_name]["death"] += deaths
            previous = {curve.identity: curve.sites for curve in lineage.curves}
        if use_identity:
            records[f"{name}__lifetime_events"] = tuple(event_rows)
        records[f"{name}__curve_innovation"] = tuple(residual_rows)
        diagnostics["strata"][name] = {"events": events, "corpus_split_identity_events": events_by_split}
    return records, diagnostics


def bool_mask_sites(mask: np.ndarray) -> np.ndarray:
    labels = np.where(np.asarray(mask, dtype=bool), 1, 0).astype(np.uint8)
    return frame_boundary_sites(labels)[EDGE_TO_ID[(0, 1)]]


def horizon_model(target: np.ndarray, knot_step: int) -> tuple[bytes, np.ndarray]:
    points: list[tuple[float, float]] = []
    for raw in target.tolist():
        site = int(raw)
        if site < V_SITES:
            y, x0 = divmod(site, WIDTH - 1)
            points.append((x0 + 1.0, y + 0.5))
        else:
            y0, x = divmod(site - V_SITES, WIDTH)
            points.append((x + 0.5, y0 + 1.0))
    knots_x = np.arange(0, WIDTH, knot_step, dtype=np.int64)
    knots_y = np.zeros_like(knots_x)
    if points:
        array = np.asarray(points)
        for index, x in enumerate(knots_x):
            nearby = array[np.abs(array[:, 0] - x) <= knot_step]
            knots_y[index] = round(np.median(nearby[:, 1] if len(nearby) else array[:, 1]))
    record = bytearray([knot_step])
    for value in knots_y.tolist():
        put_uleb(record, int(value))
    predicted = decode_horizon(bytes(record))
    return bytes(record), predicted


def decode_horizon(record: bytes) -> np.ndarray:
    if not record:
        raise WorldsheetOptimalError("empty horizon model")
    step = record[0]
    xs = np.arange(0, WIDTH, step, dtype=np.int64)
    offset = 1
    ys: list[int] = []
    for _ in xs:
        value, offset = get_uleb(record, offset)
        ys.append(min(HEIGHT - 1, int(value)))
    if offset != len(record):
        raise WorldsheetOptimalError("horizon model has trailing bytes")
    mask = np.indices((HEIGHT, WIDTH))[0] >= np.interp(np.arange(WIDTH), xs, ys)[None, :]
    return bool_mask_sites(mask)


def horizon_spatial_holdout(targets: Sequence[np.ndarray], knot_step: int) -> dict[str, Any]:
    """Fit alternating horizontal knot bands and test the interleaved bands."""

    errors = []
    for target in targets:
        points = []
        for raw in target.tolist():
            site = int(raw)
            if site < V_SITES:
                y, x0 = divmod(site, WIDTH - 1)
                points.append((x0 + 1.0, y + 0.5))
            else:
                y0, x = divmod(site - V_SITES, WIDTH)
                points.append((x + 0.5, y0 + 1.0))
        if not points:
            continue
        array = np.asarray(points)
        train = array[(array[:, 0] // knot_step).astype(np.int64) % 2 == 0]
        test = array[(array[:, 0] // knot_step).astype(np.int64) % 2 == 1]
        if len(train) < 2 or not len(test):
            continue
        train_bins = np.unique((train[:, 0] // knot_step).astype(np.int64))
        xs = np.asarray([np.median(train[(train[:, 0] // knot_step).astype(np.int64) == bin_id, 0]) for bin_id in train_bins])
        ys = np.asarray([np.median(train[(train[:, 0] // knot_step).astype(np.int64) == bin_id, 1]) for bin_id in train_bins])
        order = np.argsort(xs)
        errors.extend(np.abs(test[:, 1] - np.interp(test[:, 0], xs[order], ys[order])).tolist())
    values = np.asarray(errors, dtype=np.float64)
    return {
        "split": "alternating horizontal knot bands fit/test within every n600 frame",
        "heldout_boundary_points": int(values.size),
        "median_abs_y_px": float(np.median(values)) if values.size else None,
        "p95_abs_y_px": float(np.percentile(values, 95)) if values.size else None,
    }


def component_template_coordinates(component: np.ndarray, grid: int) -> tuple[tuple[int, int, int], ...]:
    coords: list[tuple[int, int, int]] = []
    xy: list[tuple[float, float, int]] = []
    for raw in component.tolist():
        site = int(raw)
        if site < V_SITES:
            y, x0 = divmod(site, WIDTH - 1)
            xy.append((x0 + 1.0, y + 0.5, 0))
        else:
            y0, x = divmod(site - V_SITES, WIDTH)
            xy.append((x + 0.5, y0 + 1.0, 1))
    if not xy:
        return ()
    array = np.asarray([(x, y) for x, y, _ in xy])
    lo = array.min(axis=0)
    span = np.maximum(array.max(axis=0) - lo, 1.0)
    for x, y, orientation in xy:
        qx = round((x - lo[0]) * (grid - 1) / span[0])
        qy = round((y - lo[1]) * (grid - 1) / span[1])
        coords.append((orientation, qy, qx))
    return tuple(sorted(set(coords)))


def learn_silhouette_template(
    all_sites: Sequence[np.ndarray], grid: int, threshold: float
) -> tuple[tuple[int, int, int], ...]:
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    ncomponents = 0
    for sites in all_sites[:400]:
        for component in decompose_sites(sites):
            ncomponents += 1
            for coord in component_template_coordinates(component, grid):
                counts[coord] += 1
    cutoff = max(1, math.ceil(threshold * max(1, ncomponents)))
    return tuple(sorted(coord for coord, count in counts.items() if count >= cutoff))


def encode_template(template: Sequence[tuple[int, int, int]], grid: int) -> bytes:
    output = bytearray([grid])
    put_uleb(output, len(template))
    for orientation, qy, qx in template:
        output.extend((orientation, qy, qx))
    return bytes(output)


def encode_placements(target: np.ndarray) -> bytes:
    output = bytearray()
    components = decompose_sites(target)
    put_uleb(output, len(components))
    for component in components:
        endpoints = [site_endpoints(int(site)) for site in component.tolist()]
        points = [divmod(endpoint, WIDTH + 1) for pair in endpoints for endpoint in pair]
        ys = [point[0] for point in points]
        xs = [point[1] for point in points]
        for value in (min(xs), min(ys), max(xs), max(ys)):
            put_uleb(output, value)
    return bytes(output)


def render_template(template_record: bytes, placement_record: bytes) -> np.ndarray:
    grid = template_record[0]
    count, offset = get_uleb(template_record, 1)
    template = []
    for _ in range(count):
        if offset + 3 > len(template_record):
            raise WorldsheetOptimalError("template record is truncated")
        template.append(tuple(template_record[offset:offset + 3]))
        offset += 3
    if offset != len(template_record):
        raise WorldsheetOptimalError("template record has trailing bytes")
    count, offset = get_uleb(placement_record, 0)
    boxes = []
    for _ in range(count):
        box = []
        for _ in range(4):
            value, offset = get_uleb(placement_record, offset)
            box.append(value)
        boxes.append(box)
    if offset != len(placement_record):
        raise WorldsheetOptimalError("placement record has trailing bytes")
    sites: set[int] = set()
    for x0, y0, x1, y1 in boxes:
        for orientation, qy, qx in template:
            x = round(x0 + qx * max(1, x1 - x0) / max(1, grid - 1))
            y = round(y0 + qy * max(1, y1 - y0) / max(1, grid - 1))
            if orientation == 0 and 1 <= x < WIDTH and 0 <= y < HEIGHT:
                sites.add(y * (WIDTH - 1) + x - 1)
            elif orientation == 1 and 0 <= x < WIDTH and 1 <= y < HEIGHT:
                sites.add(V_SITES + (y - 1) * WIDTH + x)
    return np.asarray(sorted(sites), dtype=np.uint32)


def static_template(all_sites: Sequence[np.ndarray], threshold: float) -> np.ndarray:
    counts: dict[int, int] = defaultdict(int)
    for sites in all_sites[:400]:
        for site in sites.tolist():
            counts[int(site)] += 1
    cutoff = max(1, math.ceil(threshold * min(400, len(all_sites))))
    return np.asarray(sorted(site for site, count in counts.items() if count >= cutoff), dtype=np.uint32)


def lane_identity_spatial_holdout(labels: Sequence[np.ndarray]) -> dict[str, Any]:
    """Fit each lane on even image rows and measure odd rows by coherent slot."""

    fitted_by_pair: list[list[Any]] = []
    heldout_by_pair: list[list[np.ndarray]] = []
    for frame in labels:
        fitted = []
        heldout = []
        for cluster in cluster_lane_lines(np.asarray(frame, dtype=np.uint8), lane_cls=1):
            train = cluster[cluster[:, 0] % 2 == 0]
            test = cluster[cluster[:, 0] % 2 == 1]
            line = fit_lane_line(train, centerline_deg=3, fit_dash=False)
            if line is not None and len(test) >= 12:
                fitted.append(line)
                heldout.append(test)
        fitted_by_pair.append(fitted)
        heldout_by_pair.append(heldout)
    assignment = coherent_slot_pack(fitted_by_pair)
    residuals: dict[int, list[np.ndarray]] = defaultdict(list)
    for pair, (lines, tests) in enumerate(zip(fitted_by_pair, heldout_by_pair, strict=True)):
        if not lines:
            continue
        matrix = assignment.M[pair].reshape(assignment.K, -1)
        active = np.flatnonzero(assignment.presence[pair])
        unused = set(active.tolist())
        for line, test in zip(lines, tests, strict=True):
            reference = float(np.polyval(line.centerline_coeffs, 12.0))
            slot = min(unused, key=lambda index: (abs(float(np.polyval(matrix[index, :4], 12.0)) - reference), index))
            unused.remove(slot)
            forward, lateral = image_to_ground(test[:, 1], test[:, 0])
            valid = np.isfinite(forward) & np.isfinite(lateral)
            if np.any(valid):
                residuals[slot].append(np.abs(lateral[valid] - line.lateral_of_forward(forward[valid])))
    rows = []
    all_values = []
    for slot in sorted(residuals):
        values = np.concatenate(residuals[slot])
        all_values.append(values)
        rows.append({
            "persistent_slot": slot,
            "heldout_pixels": int(values.size),
            "median_abs_lateral_m": float(np.median(values)),
            "p95_abs_lateral_m": float(np.percentile(values, 95)),
        })
    combined = np.concatenate(all_values) if all_values else np.empty(0)
    return {
        "split": "even scorer rows fit; odd scorer rows held out within every n600 frame",
        "persistent_identity_assignment": "bounded-K coherent slot (#234)",
        "slots": rows,
        "heldout_pixels": int(combined.size),
        "median_abs_lateral_m": float(np.median(combined)) if combined.size else None,
        "p95_abs_lateral_m": float(np.percentile(combined, 95)) if combined.size else None,
    }


def lane_polynomial_model(labels: Sequence[np.ndarray]) -> tuple[bytes, tuple[np.ndarray, ...], dict[str, Any]]:
    cfg = LaneBandRenderConfig()
    lines, fit = build_lane_band_pairs_from_lstars(np.asarray(labels, dtype=np.uint8), cfg)
    blob, meta = serialize_lane_band_rd_tracked(lines, cfg, pack_mode="coherent_slot", smooth="none")
    decoded, header = deserialize_lane_band_any(blob)
    decoded_cfg = LaneBandRenderConfig(
        softness=float(header["softness"]),
        dash_gate=bool(header["dash_gate"]),
        dash_forward_max_m=float(header["dash_forward_max_m"]),
        v_h=float(header["v_h"]),
        cx=None if header.get("cx") is None else float(header["cx"]),
    )
    predictions = []
    heldout_errors: list[float] = []
    for pair, pair_lines in enumerate(decoded):
        coverage = rasterize_lane_coverage_range_dependent(
            pair_lines, h=HEIGHT, w=WIDTH, softness=decoded_cfg.softness,
            dash_gate=decoded_cfg.dash_gate,
            dash_forward_max_m=decoded_cfg.dash_forward_max_m,
            v_h=decoded_cfg.v_h, cx=decoded_cfg.cx,
        )
        mask = coverage >= 0.5
        predictions.append(bool_mask_sites(mask))
        if pair >= 400:
            lane = np.asarray(labels[pair]) == 1
            heldout_errors.append(float(np.mean(mask != lane)))
    return blob, tuple(predictions), {
        "fit": fit,
        "serializer": meta,
        "last_200_realized_lane_mask_disagreement_mean": float(np.mean(heldout_errors)),
        "per_identity_heldout_residual": lane_identity_spatial_holdout(labels),
        "diagnostic_note": "heldout residual fits even rows only and evaluates odd rows; last-200 mask disagreement is descriptive, not called heldout",
    }


def specialize_stratum(
    name: str,
    targets: Sequence[np.ndarray],
    labels: Sequence[np.ndarray] | None,
) -> list[tuple[str, tuple[bytes, ...], tuple[np.ndarray, ...], dict[str, Any]]]:
    rows = []
    if name == "Road<->Lane":
        if labels is None:
            raise WorldsheetOptimalError("lane specialization requires the source label fields")
        blob, predictions, diagnostics = lane_polynomial_model(labels)
        model = tuple([blob] + [b""] * (N_PAIRS - 1))
        rows.append(("lbnd2_coherent_slot", model, predictions, diagnostics))
    elif name == "Road<->Undrivable":
        for step in (16, 32, 64):
            models, predictions = zip(*(horizon_model(target, step) for target in targets), strict=True)
            rows.append((f"horizon_step{step}", tuple(models), tuple(predictions), {
                "knots_per_frame": math.ceil(WIDTH / step),
                "spatial_holdout": horizon_spatial_holdout(targets, step),
            }))
    elif name == "Undrivable<->Movable":
        for grid, threshold in ((8, 0.01), (16, 0.01), (16, 0.025)):
            template = learn_silhouette_template(targets, grid, threshold)
            template_record = encode_template(template, grid)
            placements = tuple(encode_placements(target) for target in targets)
            model = tuple(
                struct.pack("<H", len(template_record)) + template_record + placement
                for placement in placements
            )
            predictions = tuple(render_template(template_record, placement) for placement in placements)
            rows.append((f"template_g{grid}_p{threshold:g}", model, predictions, {
                "template_sites": len(template), "fit_pairs": 400, "heldout_pairs": 200,
                "heldout_note": "template learned on pairs 0:400; pair-local heldout boxes are counted",
            }))
    elif name in {"Road<->MyCar", "Movable<->MyCar"}:
        for threshold in (0.25, 0.5, 0.75):
            template = static_template(targets, threshold)
            model = tuple([encode_sorted(template)] + [b""] * (N_PAIRS - 1))
            predictions = tuple(template.copy() for _ in targets)
            rows.append((f"static_frequency_{threshold:g}", model, predictions, {
                "template_sites": int(template.size), "fit_pairs": 400, "heldout_pairs": 200,
                "heldout_note": "static curve learned on pairs 0:400 only",
            }))
    else:
        raise WorldsheetOptimalError(f"no specialized grammar registered for {name}")
    diagnosed = []
    for form, model, predictions, diagnostics in rows:
        fit_innovation = sum(
            int(xor_sites(target, prediction).size)
            for target, prediction in zip(targets[:400], predictions[:400], strict=True)
        )
        heldout_innovation = sum(
            int(xor_sites(target, prediction).size)
            for target, prediction in zip(targets[400:], predictions[400:], strict=True)
        )
        if name == "Undrivable<->Movable":
            diagnostics = {
                **diagnostics,
                "parameter_stability": {
                    "fit_0_399_innovation_sites": fit_innovation,
                    "heldout_400_599_innovation_sites": heldout_innovation,
                    "heldout_per_frame": heldout_innovation / 200,
                    "fit_per_frame": fit_innovation / 400,
                },
            }
        diagnosed.append((form, model, predictions, diagnostics))
    return diagnosed


def codec_decode(codec: str, payload: bytes, raw_size: int) -> tuple[bytes, ...]:
    if codec == "brotli-q11":
        raw = brotli.decompress(payload)
        if len(raw) != raw_size:
            raise WorldsheetOptimalError("Brotli raw-size mismatch")
        return ws0.unpack_records(raw)
    if codec == "lzma1-raw":
        return ws0.unpack_records(bd1.unlzma1_raw(payload, raw_size))
    if codec == "smevr-r7-nibble":
        return tuple(bd1.unsmevr_records(payload))
    raise WorldsheetOptimalError(f"unknown codec {codec}")


def race_records(name: str, records: tuple[bytes, ...]) -> StreamRace:
    canonical = ws0.pack_records(records)
    payloads = {
        "brotli-q11": bytes(brotli.compress(canonical, quality=11)),
        "lzma1-raw": bd1.lzma1_raw(canonical),
        "smevr-r7-nibble": bd1.smevr_records(list(records)),
    }
    for codec, payload in payloads.items():
        if codec_decode(codec, payload, len(canonical)) != records:
            raise WorldsheetOptimalError(f"{name}: {codec} parse-back failed")
    winner = min(payloads, key=lambda codec: (len(payloads[codec]), CODECS.index(codec)))
    return StreamRace(name, sha256_bytes(canonical), len(canonical), payloads, winner)


def retain_race(root: Path, candidate: str, race: StreamRace) -> dict[str, Any]:
    stream_dir = root / "coder_races" / candidate / race.name.replace("/", "_")
    canonical_path = stream_dir / "canonical.records"
    # The canonical record bytes are retained too; coder sizes are never scalar-only.
    atomic_bytes(canonical_path, ws0.pack_records(codec_decode(race.winner, race.payloads[race.winner], race.canonical_raw_bytes)))
    artifacts = {
        "canonical": {"path": str(canonical_path), "bytes": canonical_path.stat().st_size,
                      "sha256": sha256_file(canonical_path)}
    }
    for codec, payload in race.payloads.items():
        path = stream_dir / f"{codec}.bin"
        atomic_bytes(path, payload)
        artifacts[codec] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"winner": race.winner, "records_sha256": race.records_sha256,
            "canonical_raw_bytes": race.canonical_raw_bytes, "artifacts": artifacts}


def load_retained_race_cache(root: Path) -> dict[str, StreamRace]:
    """Resume real coder races from their retained canonical bytes and payloads."""

    cache: dict[str, StreamRace] = {}
    manifest_rows = []
    manifest_path = root / "stages" / "stage_04_coder_cache_verified.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            for row in manifest["streams"]:
                canonical_path = Path(row["canonical_path"])
                canonical = canonical_path.read_bytes()
                if sha256_bytes(canonical) != row["records_sha256"]:
                    raise WorldsheetOptimalError("resumed canonical stream SHA mismatch")
                payloads = {}
                for codec in CODECS:
                    payload_path = Path(row["payloads"][codec]["path"])
                    payload = payload_path.read_bytes()
                    if sha256_bytes(payload) != row["payloads"][codec]["sha256"]:
                        raise WorldsheetOptimalError("resumed coder payload SHA mismatch")
                    payloads[codec] = payload
                key = row["records_sha256"]
                cache[key] = StreamRace(
                    canonical_path.parent.name, key, len(canonical), payloads, row["winner"]
                )
                manifest_rows.append(row)
        except (KeyError, OSError, ValueError, WorldsheetOptimalError):
            cache.clear()
            manifest_rows.clear()
    for canonical_path in sorted((root / "coder_races").glob("**/canonical.records")):
        try:
            canonical = canonical_path.read_bytes()
            key = sha256_bytes(canonical)
            if key in cache:
                continue
            records = ws0.unpack_records(canonical)
            payloads = {
                codec: (canonical_path.parent / f"{codec}.bin").read_bytes()
                for codec in CODECS
            }
            for codec, payload in payloads.items():
                if codec_decode(codec, payload, len(canonical)) != records:
                    raise WorldsheetOptimalError("retained coder payload does not parse back")
            winner = min(payloads, key=lambda codec: (len(payloads[codec]), CODECS.index(codec)))
            cache[key] = StreamRace(canonical_path.parent.name, key, len(canonical), payloads, winner)
            manifest_rows.append({
                "canonical_path": str(canonical_path), "records_sha256": key, "winner": winner,
                "payloads": {
                    codec: {
                        "path": str(canonical_path.parent / f"{codec}.bin"),
                        "sha256": sha256_bytes(payload),
                    }
                    for codec, payload in payloads.items()
                },
            })
        except (OSError, ValueError, WorldsheetOptimalError):
            # A partial interrupted stream is ignored and deterministically rebuilt.
            continue
    atomic_json(manifest_path, {
        "schema": "ddm_ws1_verified_coder_cache.v1", "created_utc": utc_now(),
        "verification": "all three codecs parsed to the retained canonical record pack",
        "streams": manifest_rows,
    })
    return cache


def build_envelope(candidate: str, semantics: str, races: dict[str, StreamRace]) -> bytes:
    roster = []
    bodies = bytearray()
    for name in sorted(races):
        race = races[name]
        coded = race.payloads[race.winner]
        roster.append({"name": name, "codec": race.winner, "raw_bytes": race.canonical_raw_bytes,
                       "coded_bytes": len(coded), "records_sha256": race.records_sha256})
        bodies.extend(struct.pack("<I", len(coded)))
        bodies.extend(coded)
    metadata = json.dumps({"schema": SCHEMA, "candidate": candidate, "semantics": semantics,
                           "n_pairs": N_PAIRS, "height": HEIGHT, "width": WIDTH,
                           "streams": roster}, sort_keys=True, separators=(",", ":")).encode()
    metadata_coded = brotli.compress(metadata, quality=11)
    return MAGIC + struct.pack("<BII", VERSION, len(metadata), len(metadata_coded)) + metadata_coded + bodies


def parse_envelope(payload: bytes) -> tuple[dict[str, Any], dict[str, tuple[bytes, ...]]]:
    header_size = len(MAGIC) + struct.calcsize("<BII")
    if len(payload) < header_size or payload[:4] != MAGIC:
        raise WorldsheetOptimalError("bad worldsheet envelope magic")
    version, raw_size, coded_size = struct.unpack_from("<BII", payload, 4)
    if version != VERSION:
        raise WorldsheetOptimalError("unsupported worldsheet envelope version")
    offset = header_size
    metadata = json.loads(brotli.decompress(payload[offset:offset + coded_size]))
    if len(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()) != raw_size:
        raise WorldsheetOptimalError("metadata raw size mismatch")
    offset += coded_size
    records = {}
    for row in metadata["streams"]:
        if offset + 4 > len(payload):
            raise WorldsheetOptimalError("truncated stream length")
        (length,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        coded = payload[offset:offset + length]
        offset += length
        if len(coded) != length or length != int(row["coded_bytes"]):
            raise WorldsheetOptimalError("truncated stream payload")
        decoded = codec_decode(row["codec"], coded, int(row["raw_bytes"]))
        if sha256_bytes(ws0.pack_records(decoded)) != row["records_sha256"]:
            raise WorldsheetOptimalError("stream semantic digest mismatch")
        records[row["name"]] = decoded
    if offset != len(payload) or len(records) != len(metadata["streams"]):
        raise WorldsheetOptimalError("envelope roster mismatch or trailing bytes")
    return metadata, records


def decode_candidate(
    payload: bytes, homographies: Sequence[np.ndarray]
) -> Iterator[np.ndarray]:
    metadata, records = parse_envelope(payload)
    semantics = metadata["semantics"]
    previous = [{} for _ in EDGE_PAIRS]
    selected_forms = metadata.get("selected_forms", {})
    if selected_forms:
        raise WorldsheetOptimalError("selected forms must be encoded in stream names, not hidden metadata")
    specialized_predictions: dict[str, tuple[np.ndarray, ...]] = {}
    for stream, rows in records.items():
        if "__special_model__" not in stream:
            continue
        stratum, form = stream.split("__special_model__", 1)
        if stratum == "Road<->Lane":
            lines, header = deserialize_lane_band_any(rows[0])
            cfg = LaneBandRenderConfig(
                softness=float(header["softness"]), dash_gate=bool(header["dash_gate"]),
                dash_forward_max_m=float(header["dash_forward_max_m"]), v_h=float(header["v_h"]),
                cx=None if header.get("cx") is None else float(header["cx"]),
            )
            specialized_predictions[stratum] = tuple(
                bool_mask_sites(rasterize_lane_coverage_range_dependent(
                    pair_lines, h=HEIGHT, w=WIDTH, softness=cfg.softness,
                    dash_gate=cfg.dash_gate, dash_forward_max_m=cfg.dash_forward_max_m,
                    v_h=cfg.v_h, cx=cfg.cx,
                ) >= 0.5)
                for pair_lines in lines
            )
        elif stratum == "Road<->Undrivable":
            specialized_predictions[stratum] = tuple(decode_horizon(row) for row in rows)
        elif stratum == "Undrivable<->Movable":
            predictions = []
            for row in rows:
                if len(row) < 2:
                    raise WorldsheetOptimalError("silhouette model record is truncated")
                (template_size,) = struct.unpack_from("<H", row)
                template = row[2:2 + template_size]
                placement = row[2 + template_size:]
                predictions.append(render_template(template, placement))
            specialized_predictions[stratum] = tuple(predictions)
        elif stratum in {"Road<->MyCar", "Movable<->MyCar"}:
            template, offset = decode_sorted(rows[0])
            if offset != len(rows[0]):
                raise WorldsheetOptimalError("static hood model has trailing bytes")
            specialized_predictions[stratum] = tuple(template for _ in range(N_PAIRS))
        elif stratum == "SharedHood":
            template, offset = decode_sorted(rows[0])
            if offset != len(rows[0]):
                raise WorldsheetOptimalError("shared hood model has trailing bytes")
            predictions = tuple(template for _ in range(N_PAIRS))
            specialized_predictions["Road<->MyCar"] = predictions
            specialized_predictions["Movable<->MyCar"] = predictions
        else:
            raise WorldsheetOptimalError(f"unregistered specialized model {stratum}:{form}")
    for pair in range(N_PAIRS):
        strata = []
        for edge_id in range(len(EDGE_PAIRS)):
            name = edge_name(edge_id)
            event_key = f"{name}__lifetime_events"
            entries = decode_lineage(records[event_key][pair]) if event_key in records else ()
            homography = homographies[pair] if pair and "xi" in semantics else None
            use_identity = event_key in records
            prediction = specialized_predictions[name][pair] if name in specialized_predictions else predictor_from_lineage(
                previous[edge_id], entries, homography, use_identity=use_identity
            )
            residual_key = f"{name}__curve_innovation"
            residual, end = decode_sorted(records[residual_key][pair])
            if end != len(records[residual_key][pair]):
                raise WorldsheetOptimalError("curve innovation has trailing bytes")
            sites = xor_sites(prediction, residual)
            components = decompose_sites(sites)
            if use_identity:
                if len(components) != len(entries):
                    raise WorldsheetOptimalError("lineage/component count mismatch")
                previous[edge_id] = {identity: component for (identity, _), component in zip(entries, components, strict=True)}
            else:
                previous[edge_id] = dict(enumerate(components, 1))
            strata.append(sites)
        seed = records["shared_topology_seed"][pair]
        if len(seed) != 1 or seed[0] >= N_CLASSES:
            raise WorldsheetOptimalError("invalid seed record")
        yield reconstruct_labels(seed[0], strata)


def replace_specialized_records(
    generic: dict[str, tuple[bytes, ...]],
    frame_sites: Sequence[tuple[np.ndarray, ...]],
    labels: Sequence[np.ndarray],
    race_cache: dict[str, StreamRace],
    output_root: Path,
    leg: str,
) -> tuple[dict[str, tuple[bytes, ...]], dict[str, Any]]:
    selected = dict(generic)
    report: dict[str, Any] = {}
    hood_names = ("Road<->MyCar", "Movable<->MyCar")
    for edge_id in range(len(EDGE_PAIRS)):
        name = edge_name(edge_id)
        if name not in SPECIAL_STRATA or name in hood_names:
            continue
        targets = tuple(frame[edge_id] for frame in frame_sites)
        generic_key = f"{name}__curve_innovation"
        generic_race = race_cache[sha256_bytes(ws0.pack_records(selected[generic_key]))]
        generic_bytes = len(generic_race.payloads[generic_race.winner])
        rows = []
        best: tuple[int, str, tuple[bytes, ...], tuple[bytes, ...], dict[str, Any]] | None = None
        for form, model_records, predictions, diagnostics in specialize_stratum(name, targets, labels):
            residual_records = tuple(encode_sorted(xor_sites(target, prediction)) for target, prediction in zip(targets, predictions, strict=True))
            model_name = f"{name}__special_model__{form}"
            residual_name = f"{name}__special_innovation__{form}"
            model_race = cached_race(model_name, model_records, race_cache)
            residual_race = cached_race(residual_name, residual_records, race_cache)
            retain_race(output_root, f"{leg}_special_race_{name}_{form}", model_race)
            retain_race(output_root, f"{leg}_special_race_{name}_{form}", residual_race)
            coded = len(model_race.payloads[model_race.winner]) + len(residual_race.payloads[residual_race.winner])
            row = {"form": form, "coded_stream_bytes": coded, "model_bytes": len(model_race.payloads[model_race.winner]),
                   "innovation_bytes": len(residual_race.payloads[residual_race.winner]), "diagnostics": diagnostics}
            rows.append(row)
            choice = (coded, form, model_records, residual_records, diagnostics)
            if best is None or choice[:2] < best[:2]:
                best = choice
        if best is None:
            raise WorldsheetOptimalError(f"specialization race had no rows for {name}")
        adopted = best[0] < generic_bytes
        if adopted:
            del selected[generic_key]
            selected[f"{name}__special_model__{best[1]}"] = best[2]
            selected[f"{name}__curve_innovation"] = best[3]
        report[name] = {"generic_bytes": generic_bytes, "specialized_rows": rows,
                        "best_specialized_form": best[1], "best_specialized_bytes": best[0],
                        "selected": best[1] if adopted else "generic_xi_persistent", "adopted": adopted}
    hood_ids = tuple(EDGE_TO_ID[tuple(CLASS_NAMES.index(part) for part in name.split("<->"))] for name in hood_names)
    hood_targets = {
        name: tuple(frame[edge_id] for frame in frame_sites)
        for name, edge_id in zip(hood_names, hood_ids, strict=True)
    }
    combined = tuple(
        np.asarray(sorted(set(map(int, left)) | set(map(int, right))), dtype=np.uint32)
        for left, right in zip(hood_targets[hood_names[0]], hood_targets[hood_names[1]], strict=True)
    )
    hood_generic_bytes = 0
    for name in hood_names:
        key = f"{name}__curve_innovation"
        race = race_cache[sha256_bytes(ws0.pack_records(selected[key]))]
        hood_generic_bytes += len(race.payloads[race.winner])
    hood_rows = []
    hood_best: tuple[int, str, tuple[bytes, ...], dict[str, tuple[bytes, ...]]] | None = None
    for threshold in (0.25, 0.5, 0.75):
        template = static_template(combined, threshold)
        form = f"static_frequency_{threshold:g}"
        if template.size == 0:
            hood_rows.append({
                "form": form,
                "eligible": False,
                "reason": "empty template is not the chartered one-shared-curve mechanism",
                "template_sites": 0,
            })
            continue
        model_records = tuple([encode_sorted(template)] + [b""] * (N_PAIRS - 1))
        residual_by_name = {
            name: tuple(encode_sorted(xor_sites(target, template)) for target in hood_targets[name])
            for name in hood_names
        }
        model_race = cached_race(f"SharedHood__special_model__{form}", model_records, race_cache)
        residual_races = {
            name: cached_race(f"{name}__special_innovation__{form}", rows, race_cache)
            for name, rows in residual_by_name.items()
        }
        retain_race(output_root, f"{leg}_special_race_SharedHood_{form}", model_race)
        for race in residual_races.values():
            retain_race(output_root, f"{leg}_special_race_SharedHood_{form}", race)
        coded = len(model_race.payloads[model_race.winner]) + sum(
            len(race.payloads[race.winner]) for race in residual_races.values()
        )
        heldout = {
            name: sum(int(xor_sites(target, template).size) for target in hood_targets[name][400:]) / 200
            for name in hood_names
        }
        hood_rows.append({"form": form, "eligible": True, "coded_stream_bytes": coded, "template_sites": int(template.size),
                          "fit_pairs": 400, "heldout_pairs": 200,
                          "heldout_innovation_sites_per_frame": heldout})
        choice = (coded, form, model_records, residual_by_name)
        if hood_best is None or choice[:2] < hood_best[:2]:
            hood_best = choice
    if hood_best is None:
        raise WorldsheetOptimalError("shared hood specialization race had no rows")
    hood_adopted = hood_best[0] < hood_generic_bytes
    if hood_adopted:
        for name in hood_names:
            selected[f"{name}__curve_innovation"] = hood_best[3][name]
        selected[f"SharedHood__special_model__{hood_best[1]}"] = hood_best[2]
    hood_report = {"generic_joint_bytes": hood_generic_bytes, "specialized_rows": hood_rows,
                   "best_specialized_form": hood_best[1], "best_specialized_joint_bytes": hood_best[0],
                   "selected": hood_best[1] if hood_adopted else "generic_xi_persistent",
                   "adopted": hood_adopted,
                   "mechanism": "one shared static curve plus separately coded class-pair microdeltas"}
    for name in hood_names:
        report[name] = hood_report
    return selected, report


def cached_race(name: str, records: tuple[bytes, ...], cache: dict[str, StreamRace]) -> StreamRace:
    key = sha256_bytes(ws0.pack_records(records))
    if key not in cache:
        cache[key] = race_records(name, records)
    return dataclasses.replace(cache[key], name=name)


def materialize_candidate(
    name: str,
    semantics: str,
    records: dict[str, tuple[bytes, ...]],
    diagnostics: dict[str, Any],
    race_cache: dict[str, StreamRace],
    output_root: Path,
) -> Candidate:
    races = {stream: cached_race(stream, rows, race_cache) for stream, rows in sorted(records.items())}
    custody = {stream: retain_race(output_root, name, race) for stream, race in races.items()}
    envelope = build_envelope(name, semantics, races)
    path = output_root / "candidates" / name / "worldsheet.wso"
    repeat_path = output_root / "candidates" / name / "worldsheet.repeat.wso"
    if path.exists():
        prior = path.read_bytes()
        atomic_bytes(repeat_path, prior)
        if prior != envelope:
            raise WorldsheetOptimalError(
                f"{name}: deterministic repeat differs from the previously retained envelope"
            )
    atomic_bytes(path, envelope)
    atomic_json(output_root / "candidates" / name / "custody.json", {
        "candidate": name, "semantics": semantics, "streams": custody,
        "envelope": {"path": str(path), "bytes": len(envelope), "sha256": sha256_bytes(envelope)},
        "determinism_repeat": (
            {"path": str(repeat_path), "bytes": repeat_path.stat().st_size, "sha256": sha256_file(repeat_path),
             "byte_identical": True}
            if repeat_path.exists() else None
        ),
    })
    return Candidate(name, semantics, records, envelope, races, diagnostics)


def verify_candidate(
    candidate: Candidate,
    homographies: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
    originals: np.memmap,
) -> dict[str, Any]:
    decoded_digest = hashlib.sha256()
    expected_digest = hashlib.sha256()
    mismatch = 0
    frames = 0
    for pair, decoded in enumerate(decode_candidate(candidate.envelope, homographies)):
        target = np.asarray(expected[pair], dtype=np.uint8)
        if not np.array_equal(decoded, target):
            raise WorldsheetOptimalError(f"{candidate.name}: receiver differs from semantics at pair {pair}")
        decoded_digest.update(decoded.tobytes())
        expected_digest.update(target.tobytes())
        mismatch += int(np.count_nonzero(decoded != np.asarray(originals[pair], dtype=np.uint8)))
        frames += 1
    if frames != N_PAIRS or decoded_digest.digest() != expected_digest.digest():
        raise WorldsheetOptimalError(f"{candidate.name}: incomplete receiver pass")
    return {"receiver_semantic_roundtrip": True, "pairs": frames,
            "decoded_u8_sha256": decoded_digest.hexdigest(), "mismatches_vs_original": mismatch,
            "dseg_equivalent_mass": mismatch / TOTAL_CELLS}


def pin_inputs(cache: Path, ws0_memo: Path, es1: Path, output: Path) -> dict[str, Any]:
    if not output.as_posix().startswith("/Volumes/VertigoDataTier/pact/"):
        raise WorldsheetOptimalError("payload custody must use the first-priority SSD tier")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise WorldsheetOptimalError("fewer than 2 GiB free on retained-payload tier")
    pins = {
        "cache": (cache, EXPECTED_CACHE_SHA256),
        "ws0_memo": (ws0_memo, EXPECTED_WS0_SHA256),
        "es1": (es1, EXPECTED_ES1_SHA256),
    }
    receipt = {}
    for name, (path, expected) in pins.items():
        actual = sha256_file(path)
        if actual != expected:
            raise WorldsheetOptimalError(f"{name} SHA mismatch: {actual}")
        receipt[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    return receipt


def load_or_build_labels(
    originals: np.memmap, output: Path
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict[str, Any]]:
    exact = tuple(np.asarray(originals[pair], dtype=np.uint8) for pair in range(N_PAIRS))
    stage = output / "stages" / "stage_01_q2_labels.npy"
    final_path = output / "FINAL_RESULT.json"
    if stage.exists() and final_path.exists():
        prior = json.loads(final_path.read_text())
        tolerance_meta = prior.get("tolerance", {})
        expected = tolerance_meta.get("stage_checkpoint", {})
        mapped = np.load(stage, mmap_mode="r", allow_pickle=False)
        if (
            mapped.shape == (N_PAIRS, HEIGHT, WIDTH)
            and mapped.dtype == np.uint8
            and expected.get("sha256") == sha256_file(stage)
        ):
            tolerant = tuple(np.asarray(mapped[pair], dtype=np.uint8) for pair in range(N_PAIRS))
            return exact, tolerant, {**tolerance_meta, "resumed_from_disk": True}
    ws0_extract = DEFAULT_WS0_OUTPUT / "stage_01_extracted"
    frames = tuple(ws0.frame_from_npz(ws0._frame_path(ws0_extract, pair)) for pair in range(N_PAIRS))
    quantized = ws0.quantize_frames(frames, q_step=2, error_cap=TOLERANCE_CELLS)
    tolerant = tuple(ws0.render_frame(frame) for frame in quantized.frames)
    # A durable, resumable stage checkpoint keeps the complete q2 semantics.
    if not stage.exists():
        temporary = stage.with_name(f".{stage.name}.tmp-{os.getpid()}")
        stage.parent.mkdir(parents=True, exist_ok=True)
        mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH))
        for pair, labels in enumerate(tolerant):
            mapped[pair] = labels
        mapped.flush()
        del mapped
        os.replace(temporary, stage)
    return exact, tolerant, {"q_step": 2, "error_cap": TOLERANCE_CELLS,
        "selected_shift_upper_bound": quantized.selected_shift_upper_bound,
        "selected_boundaries": quantized.selected_boundaries,
        "stage_checkpoint": {"path": str(stage), "bytes": stage.stat().st_size, "sha256": sha256_file(stage)}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pins = pin_inputs(args.cache, args.ws0_memo, args.es1, args.output)
    originals = open_stored_npy_memmap(args.cache, "lstars")
    poses = open_stored_npy_memmap(args.cache, "gt_poses")
    exact_labels, tolerant_labels, tolerance_meta = load_or_build_labels(originals, args.output)
    s_t, s_r, pitch, motion_custody = load_g1_worldsheet_motion(_REPO)
    relative_xi, relative_custody = relative_adjacent_xi(poses, s_t=s_t, s_r=s_r, pitch_rad=pitch)
    xi_path = args.output / "stages" / "stage_00_relative_xi.npy"
    if xi_path.exists():
        retained_xi = np.load(xi_path, allow_pickle=False)
        if not np.array_equal(retained_xi, relative_xi):
            raise WorldsheetOptimalError("retained relative-xi stage differs from the derived G1 proxy")
    else:
        buffer = bytearray()
        stream = __import__("io").BytesIO()
        np.save(stream, relative_xi, allow_pickle=False)
        buffer.extend(stream.getvalue())
        atomic_bytes(xi_path, bytes(buffer))
    geom = GroundHomographyGeom.eon(native_hw=(HEIGHT, WIDTH), pitch=pitch)
    homographies = tuple(homography_from_xi_numpy(relative_xi[pair], geom) for pair in range(N_PAIRS))
    atomic_json(args.output / "stages" / "stage_00_motion_complete.json", {
        "relative_xi": {"path": str(xi_path), "bytes": xi_path.stat().st_size, "sha256": sha256_file(xi_path)},
        "calibration": {"s_t": s_t, "s_r": s_r, "pitch_rad": pitch},
        "motion_custody": motion_custody, "relative_custody": relative_custody,
    })

    race_cache = load_retained_race_cache(args.resume_from)
    results: dict[str, Any] = {}
    candidates: dict[str, Candidate] = {}
    for leg, labels in (("lossless", exact_labels), ("tolerance_q2", tolerant_labels)):
        site_frames = load_or_build_site_frames(labels, args.output / "stages" / f"stage_02_{leg}_sites")
        seeds = tuple(seed_record(frame) for frame in labels)
        lineages = make_lineages(site_frames, homographies, args.output / "stages" / f"stage_03_{leg}_lineage")
        controls = (
            ("no_identity_xi", False, True),
            ("persistent_static", True, False),
            ("persistent_xi", True, True),
        )
        leg_candidates = {}
        for suffix, identity, advection in controls:
            records, diagnostics = generic_records(site_frames, seeds, homographies, lineages,
                                                    use_identity=identity, use_advection=advection)
            name = f"{leg}_{suffix}"
            candidate = materialize_candidate(name, suffix, records, diagnostics, race_cache, args.output)
            verification = verify_candidate(candidate, homographies, labels, originals)
            candidates[name] = candidate
            leg_candidates[suffix] = {"bytes": len(candidate.envelope), "sha256": sha256_bytes(candidate.envelope),
                                      "verification": verification, "diagnostics": diagnostics}
            if suffix == "persistent_xi":
                specialized_records, specialized_report = replace_specialized_records(
                    records, site_frames, labels, race_cache, args.output, leg)
                specialized = materialize_candidate(f"{leg}_persistent_xi_specialized", "persistent_xi",
                                                    specialized_records, {"specialized": specialized_report},
                                                    race_cache, args.output)
                specialized_verify = verify_candidate(specialized, homographies, labels, originals)
                candidates[specialized.name] = specialized
                leg_candidates["persistent_xi_specialized"] = {
                    "bytes": len(specialized.envelope), "sha256": sha256_bytes(specialized.envelope),
                    "verification": specialized_verify, "specialized": specialized_report,
                }
        final_key = min(("persistent_xi", "persistent_xi_specialized"),
                        key=lambda key: (leg_candidates[key]["bytes"], key))
        leg_candidates["selected"] = final_key
        leg_candidates["identity_buy_bytes"] = leg_candidates["no_identity_xi"]["bytes"] - leg_candidates["persistent_xi"]["bytes"]
        leg_candidates["advection_buy_bytes"] = leg_candidates["persistent_static"]["bytes"] - leg_candidates["persistent_xi"]["bytes"]
        leg_candidates["vs_ws0_ordinal_bytes"] = leg_candidates[final_key]["bytes"] - WS0_ORDINAL_TEMPORAL_BYTES
        results[leg] = leg_candidates
        atomic_json(args.output / "stages" / f"stage_02_{leg}_complete.json", leg_candidates)

    lossless_bytes = results["lossless"][results["lossless"]["selected"]]["bytes"]
    tolerance_bytes = results["tolerance_q2"][results["tolerance_q2"]["selected"]]["bytes"]
    if lossless_bytes >= LOSSLESS_FALSIFIER and tolerance_bytes >= TOLERANCE_FALSIFIER:
        verdict = "FAMILY_NO_GO"
    elif tolerance_bytes <= TOLERANCE_FALSIFIER:
        verdict = "PARTIAL_REHABILITATION_OF_90K_ASSUMPTION"
    elif min(lossless_bytes, tolerance_bytes) < PP1_BYTES:
        verdict = "REOPENED_BY_BEATING_PP1"
    else:
        verdict = "INCONCLUSIVE_BETWEEN_REGISTERED_GATES"
    final = {
        "schema": SCHEMA, "created_utc": utc_now(), "axis": AXIS,
        "authority": "scorer-free coder measurement; not an upstream/evaluate.py score",
        "inputs": pins, "motion": {"relative_xi_path": str(xi_path), "custody": relative_custody,
            "limitation": "G1 nearest-target-pair Pose6 proxy; no exact cross-pair trajectory is claimed"},
        "tolerance": tolerance_meta, "results": results,
        "thresholds": {"lossless_family_no_go_bytes": LOSSLESS_FALSIFIER,
                       "tolerance_family_no_go_bytes": TOLERANCE_FALSIFIER,
                       "pp1_bytes": PP1_BYTES, "ws0_ordinal_temporal_bytes": WS0_ORDINAL_TEMPORAL_BYTES},
        "verdict": verdict,
        "custody": {"root": str(args.output), "all_real_coder_payloads_retained": True},
    }
    atomic_json(args.output / "FINAL_RESULT.json", final)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ws0-memo", type=Path, default=DEFAULT_WS0_MEMO)
    parser.add_argument("--es1", type=Path, default=DEFAULT_ES1)
    parser.add_argument(
        "--resume-from", type=Path, default=DEFAULT_OUTPUT,
        help="retained root whose verified stage/coder payloads may be resumed",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
