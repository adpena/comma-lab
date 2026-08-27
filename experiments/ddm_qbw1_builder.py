"""Resumable QBW1 first-rung builder; this arm runs stages 00--02 only.

The script consumes the pinned 600-pair semantic field, selects the preregistered n32 population,
fits and retains every frozen-schema grammar candidate, then performs real reset-record
encode/decode, repeat, mutation, and observability closure.  It does not load a scorer or launch
training.  Stages 03--05 are represented by the sealed MAIN fire order emitted after stage 02.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ddm_qbw1_packet as packet
import numpy as np
import psutil
from scipy import ndimage
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
ARM = "ddm_qbw1_builder_first_rung"
STORE = Path("/Volumes/APDataStore/pact/ddm_qbw1_boundary_event_quotient")
SOURCE_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/retained/fields/"
    "decoded_tokens_instrumented.u8"
)
SOURCE_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
SCHEMA_DOC = REPO / ".omx/research/SPEC_ddm_qbw1_packet_schema_v1_20260827.md"
N, H, W = 600, packet.HEIGHT, packet.WIDTH
SEED = 20260827
RESERVE_BYTES = 8 * 1024**3
REQUIRED_WORK_BYTES = 4 * 1024**3
FIXED_ENVELOPE_BYTES = 53_076
QUOTIENT_ALLOWANCE_BYTES = 84_910
COMPLETE_CAP_BYTES = 137_986
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
GB1_ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/retained/"
    "candidate_gb1_groupbin8_surprise.zip"
)
GB1_SHA256 = "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4"
GB1_BYTES = 180_215
GB1_SCORE = 0.14811799921260607


class QBW1BuildError(RuntimeError):
    """Fail-closed builder error."""


@dataclass(frozen=True, slots=True)
class CanonicalEdge:
    start_rank: int
    end_rank: int
    left_label: int
    right_label: int


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes_once(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise QBW1BuildError(f"resume payload drift: {path}")
        return file_fact(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_json_once(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes_once(path, canonical_json_bytes(value))


def atomic_npz_once(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    payload = io.BytesIO()
    with zipfile.ZipFile(
        payload, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True
    ) as archive:
        for name in sorted(arrays):
            array_payload = io.BytesIO()
            np.lib.format.write_array(array_payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, array_payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return atomic_bytes_once(path, payload.getvalue())


def load_checkpoint(path: Path, schema: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("schema") != schema or value.get("complete") is not True:
        raise QBW1BuildError(f"invalid or incomplete checkpoint: {path}")
    return value


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def upstream_identity() -> dict[str, Any]:
    upstream = REPO / "upstream"
    return {
        "path": str(upstream),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
        ).strip(),
        "evaluate_sha256": sha256_file(upstream / "evaluate.py"),
    }


def storage_preflight() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in (
        Path("/Volumes/APDataStore/pact"),
        Path("/Volumes/VertigoDataTier/pact"),
        REPO,
    ):
        usage = shutil.disk_usage(path)
        rows.append(
            {
                "path": str(path),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )
    ap = rows[0]
    required = RESERVE_BYTES + REQUIRED_WORK_BYTES
    if ap["free_bytes"] < required:
        raise QBW1BuildError(
            f"APDataStore preflight refused: free={ap['free_bytes']} required={required}"
        )
    STORE.mkdir(parents=True, exist_ok=True)
    probe = STORE / f".write_probe.{os.getpid()}"
    try:
        with probe.open("xb") as handle:
            handle.write(b"qbw1-write-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "schema": "ddm_qbw1_storage_preflight.v1",
        "waterfall": rows,
        "selected_root": str(STORE),
        "reserve_bytes": RESERVE_BYTES,
        "required_work_bytes": REQUIRED_WORK_BYTES,
        "required_free_bytes": required,
        "pass": True,
    }


def require_sources() -> dict[str, Any]:
    if SOURCE_FIELD.stat().st_size != N * H * W:
        raise QBW1BuildError("source field byte count drifted")
    if sha256_file(SOURCE_FIELD) != SOURCE_SHA256:
        raise QBW1BuildError("source field SHA-256 drifted")
    if sha256_file(GB1_ARCHIVE) != GB1_SHA256 or GB1_ARCHIVE.stat().st_size != GB1_BYTES:
        raise QBW1BuildError("matched GB1 archive custody drifted")
    return {
        "source_field": file_fact(SOURCE_FIELD),
        "schema_doc": file_fact(SCHEMA_DOC),
        "gb1_archive": file_fact(GB1_ARCHIVE),
        "gb1_score_axis": "[contest-CUDA T4 n600]",
        "gb1_score": GB1_SCORE,
    }


def source_memmap() -> np.memmap:
    return np.memmap(SOURCE_FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))


def road_lane_crack_count(field: np.ndarray) -> int:
    left, right = field[:, :-1], field[:, 1:]
    top, bottom = field[:-1, :], field[1:, :]
    horizontal = ((left == 0) & (right == 1)) | ((left == 1) & (right == 0))
    vertical = ((top == 0) & (bottom == 1)) | ((top == 1) & (bottom == 0))
    return int(horizontal.sum(dtype=np.int64) + vertical.sum(dtype=np.int64))


def all_interface_counts(field: np.ndarray) -> tuple[int, int, int]:
    right_diff = field[:, 1:] != field[:, :-1]
    down_diff = field[1:, :] != field[:-1, :]
    total = int(right_diff.sum(dtype=np.int64) + down_diff.sum(dtype=np.int64))
    road_lane = road_lane_crack_count(field)
    return total, road_lane, total - road_lane


def selection_rows(field: np.memmap) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    crack_counts = [road_lane_crack_count(np.asarray(field[pair])) for pair in range(N)]
    rng = np.random.Generator(np.random.PCG64(SEED))
    strata: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for block in range(10):
        members = list(range(block * 60, (block + 1) * 60))
        ordered = sorted(members, key=lambda pair: (crack_counts[pair], pair))
        halves = (("low", ordered[:30]), ("high", ordered[30:]))
        take = 2 if block <= 5 else 1
        for half, population in halves:
            chosen = sorted(int(value) for value in rng.choice(population, size=take, replace=False))
            stratum_id = f"block_{block:02d}_{half}"
            strata.append(
                {
                    "stratum_id": stratum_id,
                    "temporal_block": block,
                    "crack_half": half,
                    "population_size": len(population),
                    "sample_size": take,
                    "inclusion_probability": take / len(population),
                    "population_pair_ids": population,
                    "population_crack_count_min": min(crack_counts[pair] for pair in population),
                    "population_crack_count_max": max(crack_counts[pair] for pair in population),
                    "selected_pair_ids": chosen,
                }
            )
            for pair in chosen:
                selected.append(
                    {
                        "pair_id": pair,
                        "stratum_id": stratum_id,
                        "population_size": len(population),
                        "sample_size": take,
                        "inclusion_probability": take / len(population),
                        "road_lane_crack_count": crack_counts[pair],
                    }
                )
    selected.sort(key=lambda row: row["pair_id"])
    if len(selected) != 32 or len({row["pair_id"] for row in selected}) != 32:
        raise QBW1BuildError("preregistered selection did not produce 32 unique pairs")
    return strata, selected


def stage_00() -> dict[str, Any]:
    checkpoint = STORE / "stage_00_selection" / "STAGE_00_CHECKPOINT.json"
    storage_preflight()
    require_sources()
    resumed = load_checkpoint(checkpoint, "ddm_qbw1_stage_00_selection.v1")
    if resumed is not None:
        return resumed
    started = time.time()
    storage = storage_preflight()
    sources = require_sources()
    field = source_memmap()
    strata, selected = selection_rows(field)
    config = {
        "schema": "ddm_qbw1_stage_config.v1",
        "arm": ARM,
        "seed": SEED,
        "numpy_bit_generator": "PCG64",
        "shape": [N, H, W],
        "dictionary_capacities": list(packet.DICTIONARY_CAPACITIES),
        "fixed_envelope_bytes_projection": FIXED_ENVELOPE_BYTES,
        "quotient_allowance_bytes": QUOTIENT_ALLOWANCE_BYTES,
        "complete_cap_bytes": COMPLETE_CAP_BYTES,
        "counterfactual_hooks": {
            "section_bit_mutation": True,
            "grammar_branch_ablation": True,
            "lane_native_vs_separate_control": True,
            "joint_render_vs_fixed_paint_control": True,
            "boundary_branch_ablation": True,
            "interior_branch_ablation": True,
        },
    }
    config_bytes = canonical_json_bytes(config)
    config_fact = atomic_bytes_once(STORE / "CONFIG.json", config_bytes)
    receipt = {
        "schema": "ddm_qbw1_stage_00_selection.v1",
        "complete": True,
        "stage": "stage_00_selection",
        "arm": ARM,
        "axis": "[macOS-CPU scorer-free advisory]",
        "score_claim": False,
        "git_head": git_head(),
        "upstream": upstream_identity(),
        "storage_preflight": storage,
        "sources": sources,
        "config": config_fact,
        "seed": SEED,
        "numpy_bit_generator": "PCG64",
        "strata": strata,
        "selected": selected,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_once(checkpoint, receipt)
    return receipt


def canonical_edges(base: np.ndarray) -> tuple[CanonicalEdge, ...]:
    edges: list[CanonicalEdge] = []
    ys, xs = np.nonzero(base[:, 1:] != base[:, :-1])
    for y, x_left in zip(ys.tolist(), xs.tolist(), strict=True):
        x = x_left + 1
        # Canonical top->bottom edge.  Its geometric left is the east/right pixel.
        edges.append(
            CanonicalEdge(
                packet.vertex_rank(y, x),
                packet.vertex_rank(y + 1, x),
                int(base[y, x]),
                int(base[y, x - 1]),
            )
        )
    ys, xs = np.nonzero(base[1:, :] != base[:-1, :])
    for y_top, x in zip(ys.tolist(), xs.tolist(), strict=True):
        y = y_top + 1
        # Canonical left->right edge.  Its geometric left is the north/top pixel.
        edges.append(
            CanonicalEdge(
                packet.vertex_rank(y, x),
                packet.vertex_rank(y, x + 1),
                int(base[y - 1, x]),
                int(base[y, x]),
            )
        )
    edges.sort(key=lambda edge: (edge.start_rank, edge.end_rank))
    return tuple(edges)


def _step_from_edge(edge: CanonicalEdge, start: int) -> tuple[packet.CrackStep, int]:
    if start == edge.start_rank:
        end = edge.end_rank
        left, right = edge.left_label, edge.right_label
    elif start == edge.end_rank:
        end = edge.start_rank
        left, right = edge.right_label, edge.left_label
    else:
        raise QBW1BuildError("chain traversal entered a nonincident edge")
    y0, x0 = packet.vertex_coords(start)
    y1, x1 = packet.vertex_coords(end)
    direction = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}[(y1 - y0, x1 - x0)]
    return packet.CrackStep(direction, left, right), end


def decompose_chains(edges: tuple[CanonicalEdge, ...]) -> tuple[packet.CrackChain, ...]:
    adjacency: dict[int, list[int]] = {}
    for index, edge in enumerate(edges):
        adjacency.setdefault(edge.start_rank, []).append(index)
        adjacency.setdefault(edge.end_rank, []).append(index)
    for vertex, incident in adjacency.items():
        incident.sort(
            key=lambda index: (
                edges[index].end_rank if edges[index].start_rank == vertex else edges[index].start_rank
            )
        )
    unvisited = set(range(len(edges)))
    chains: list[packet.CrackChain] = []

    def walk(start_vertex: int, first_edge: int) -> packet.CrackChain:
        current = start_vertex
        edge_index = first_edge
        steps: list[packet.CrackStep] = []
        while edge_index in unvisited:
            unvisited.remove(edge_index)
            step, endpoint = _step_from_edge(edges[edge_index], current)
            steps.append(step)
            current = endpoint
            candidates = [index for index in adjacency[current] if index in unvisited]
            if len(adjacency[current]) != 2 or not candidates:
                break
            edge_index = candidates[0]
        return packet.CrackChain(start_vertex, tuple(steps))

    for vertex in sorted(adjacency):
        if len(adjacency[vertex]) == 2:
            continue
        for edge_index in adjacency[vertex]:
            if edge_index in unvisited:
                chains.append(walk(vertex, edge_index))
    while unvisited:
        first_edge = min(unvisited, key=lambda index: (edges[index].start_rank, edges[index].end_rank))
        start = min(edges[first_edge].start_rank, edges[first_edge].end_rank)
        chains.append(walk(start, first_edge))
    chains.sort(
        key=lambda chain: (
            chain.birth_rank,
            len(chain.steps),
            tuple((step.direction, step.left_label, step.right_label) for step in chain.steps),
        )
    )
    return tuple(chains)


def seed_labels(base: np.ndarray, edges: tuple[packet.EdgeFact, ...]) -> tuple[tuple[int, ...], np.ndarray]:
    cells, count = packet.integrate_cells(edges)
    first = np.full(count, H * W, dtype=np.int64)
    flat_cells = cells.ravel()
    np.minimum.at(first, flat_cells, np.arange(H * W, dtype=np.int64))
    labels = tuple(int(base.ravel()[index]) for index in first.tolist())
    decoded = packet.assign_seed_labels(cells, labels)
    if not np.array_equal(decoded, base):
        raise QBW1BuildError("base cells do not reconstruct exactly")
    packet.verify_closed_chain_consistency(edges, decoded)
    return labels, cells


def lane_dash_events(source: np.ndarray, edges: tuple[packet.EdgeFact, ...]) -> tuple[packet.LaneDashEvent, ...]:
    lane = source == 1
    component_map, count = ndimage.label(lane, structure=ndimage.generate_binary_structure(2, 1))
    midpoints, _road_tangents, _road_normals = packet.road_boundary_basis(edges)
    if count and not len(midpoints):
        raise QBW1BuildError("Lane exists but the base object has no Road boundary graph")
    tree = cKDTree(midpoints) if len(midpoints) else None
    absolute: list[tuple[int, int, int, int, int, int]] = []
    for component_id in range(1, count + 1):
        coordinates = np.argwhere(component_map == component_id).astype(np.float64)
        if not len(coordinates):
            continue
        center = coordinates.mean(axis=0)
        assert tree is not None
        _distance, anchor = tree.query(center, k=1)
        anchor = int(anchor)
        tangent = _road_tangents[anchor].astype(np.float64)
        normal = _road_normals[anchor].astype(np.float64)
        delta = center - midpoints[anchor]
        tangent_q4 = round(float(delta @ tangent) * 4.0)
        normal_q4 = round(float(delta @ normal) * 4.0)
        if len(coordinates) >= 2:
            centered = coordinates - center
            covariance = centered.T @ centered / len(centered)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
            projections = centered @ major_axis
            orthogonal = centered @ np.asarray((-major_axis[1], major_axis[0]))
            major_half = max(0.5, (float(projections.max()) - float(projections.min()) + 1.0) / 2.0)
            minor_half = max(0.5, (float(orthogonal.max()) - float(orthogonal.min()) + 1.0) / 2.0)
        else:
            major_axis = np.asarray((0.0, 1.0))
            major_half = minor_half = 0.5
        angle = round(
            math.atan2(float(major_axis[0]), float(major_axis[1])) * 256.0 / (2 * math.pi)
        ) % 256
        absolute.append(
            (
                anchor,
                tangent_q4,
                normal_q4,
                max(1, round(major_half * 4.0)),
                max(1, round(minor_half * 4.0)),
                angle,
            )
        )
    absolute.sort()
    events: list[packet.LaneDashEvent] = []
    previous_anchor = 0
    for anchor, tangent, normal, major, minor, angle in absolute:
        events.append(
            packet.LaneDashEvent(
                anchor - previous_anchor,
                tangent,
                normal,
                major,
                minor,
                angle,
            )
        )
        previous_anchor = anchor
    return tuple(events)


def extract_object(source: np.ndarray) -> dict[str, Any]:
    if source.shape != (H, W) or source.dtype != np.uint8:
        raise QBW1BuildError("source pair field geometry/dtype mismatch")
    if not set(np.unique(source).tolist()).issubset({0, 1, 2, 3, 4}):
        raise QBW1BuildError("source pair contains a noncanonical class")
    base = source.copy()
    base[base == 1] = 0
    canonical = canonical_edges(base)
    chains = decompose_chains(canonical)
    expanded = packet.expand_edges(chains)
    labels, cells = seed_labels(base, expanded)
    lane_events = lane_dash_events(source, expanded)
    return {
        "chains": chains,
        "edges": expanded,
        "seed_labels": labels,
        "lane_events": lane_events,
        "base_field": base,
        "cells": cells,
    }


def raw_sections(obj: dict[str, Any]) -> tuple[tuple[int, bytes], ...]:
    return (
        (packet.SECTION_BASE_CRACK_CHAINS, packet.encode_chains(obj["chains"])),
        (packet.SECTION_REGION_SEEDS, packet.encode_seed_labels(obj["seed_labels"])),
        (packet.SECTION_LANE_DASH_EVENTS, packet.encode_lane_events(obj["lane_events"])),
    )


def stage_01() -> dict[str, Any]:
    stage0 = stage_00()
    checkpoint = STORE / "stage_01_grammar_fit" / "STAGE_01_CHECKPOINT.json"
    storage_preflight()
    require_sources()
    resumed = load_checkpoint(checkpoint, "ddm_qbw1_stage_01_grammar_fit.v1")
    if resumed is not None:
        return resumed
    started = time.time()
    storage_preflight()
    field = source_memmap()
    selected = stage0["selected"]
    raw_by_pair: dict[int, tuple[tuple[int, bytes], ...]] = {}
    object_rows: list[dict[str, Any]] = []
    raw_root = STORE / "stage_01_grammar_fit" / "declared_objects"
    for selection in selected:
        pair_id = int(selection["pair_id"])
        source = np.asarray(field[pair_id]).copy()
        obj = extract_object(source)
        sections = raw_sections(obj)
        raw_by_pair[pair_id] = sections
        section_facts = []
        for section_id, raw in sections:
            section_facts.append(
                atomic_bytes_once(raw_root / f"pair_{pair_id:04d}" / f"section_{section_id}.raw", raw)
            )
        encoder_layers = atomic_npz_once(
            raw_root / f"pair_{pair_id:04d}" / "encoder_layers.npz",
            source_field_u8=source,
            encoder_base_u8=np.asarray(obj["base_field"], dtype=np.uint8),
            encoder_cells_i32=np.asarray(obj["cells"], dtype=np.int32),
            encoder_lane_raster_u8=packet.rasterize_lane_events(
                obj["lane_events"], obj["edges"]
            ).astype(np.uint8),
        )
        total_interfaces, road_lane, non_lane = all_interface_counts(source)
        object_rows.append(
            {
                "pair_id": pair_id,
                "stratum_id": selection["stratum_id"],
                "chain_count": len(obj["chains"]),
                "crack_edge_count": len(obj["edges"]),
                "cell_count": len(obj["seed_labels"]),
                "lane_event_count": len(obj["lane_events"]),
                "source_interface_length": total_interfaces,
                "road_lane_interface_length": road_lane,
                "non_lane_interface_length": non_lane,
                "raw_sections": section_facts,
                "encoder_layers": encoder_layers,
            }
        )
    dictionary_source = b"".join(
        raw
        for pair_id in sorted(raw_by_pair)
        for _section_id, raw in sorted(raw_by_pair[pair_id])
    )
    dictionary_source_fact = atomic_bytes_once(
        STORE / "stage_01_grammar_fit" / "dictionary_fit_source.bin", dictionary_source
    )
    candidates: list[dict[str, Any]] = []
    for capacity in packet.DICTIONARY_CAPACITIES:
        dictionary = dictionary_source[-capacity:] if capacity else b""
        model = packet.QBW1Model(dictionary=dictionary)
        candidate_root = STORE / "stage_01_grammar_fit" / "candidates" / f"dict_{capacity:05d}"
        model_fact = atomic_bytes_once(candidate_root / "model.qbm", model.to_bytes())
        records: list[dict[str, Any]] = []
        for pair_id in sorted(raw_by_pair):
            sections = dict(raw_by_pair[pair_id])
            record = packet.encode_record(
                pair_id,
                model,
                packet.decode_chains(sections[packet.SECTION_BASE_CRACK_CHAINS]),
                packet.decode_seed_labels(sections[packet.SECTION_REGION_SEEDS]),
                packet.decode_lane_events(sections[packet.SECTION_LANE_DASH_EVENTS]),
            )
            records.append(atomic_bytes_once(candidate_root / "records" / f"pair_{pair_id:04d}.qbr", record))
        total = model_fact["bytes"] + sum(row["bytes"] for row in records)
        candidate = {
            "capacity": capacity,
            "model": model_fact,
            "records": records,
            "n32_exact_model_plus_records_bytes": total,
        }
        atomic_json_once(candidate_root / "CANDIDATE.json", candidate)
        candidates.append(candidate)
    winner = min(candidates, key=lambda row: (row["n32_exact_model_plus_records_bytes"], row["capacity"]))
    result = {
        "schema": "ddm_qbw1_stage_01_grammar_fit.v1",
        "complete": True,
        "stage": "stage_01_grammar_fit",
        "arm": ARM,
        "axis": "[macOS-CPU scorer-free advisory]",
        "score_claim": False,
        "git_head": git_head(),
        "source_field_sha256": SOURCE_SHA256,
        "schema_doc_sha256": sha256_file(SCHEMA_DOC),
        "object_rows": object_rows,
        "dictionary_fit_source": dictionary_source_fact,
        "candidates": candidates,
        "winner_capacity": winner["capacity"],
        "winner_model": winner["model"],
        "loser_payloads_retained": True,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_once(checkpoint, result)
    return result


def boundary_inputs(base: np.ndarray) -> dict[str, np.ndarray]:
    boundary = np.zeros((H, W), dtype=np.bool_)
    diff = base[:, 1:] != base[:, :-1]
    boundary[:, 1:] |= diff
    boundary[:, :-1] |= diff
    diff = base[1:, :] != base[:-1, :]
    boundary[1:, :] |= diff
    boundary[:-1, :] |= diff
    distance = ndimage.distance_transform_edt(~boundary).astype(np.float32)
    gradient_y, gradient_x = np.gradient(distance)
    norm = np.sqrt(gradient_y * gradient_y + gradient_x * gradient_x) + 1e-6
    tangent_y = (-gradient_x / norm).astype(np.float16)
    tangent_x = (gradient_y / norm).astype(np.float16)
    onehot = np.stack([base == label for label in packet.BASE_LABELS]).astype(np.uint8)
    return {
        "base_onehot_u8": onehot,
        "boundary_u8": boundary.astype(np.uint8),
        "boundary_distance_f16": distance.astype(np.float16),
        "boundary_tangent_y_f16": tangent_y,
        "boundary_tangent_x_f16": tangent_x,
    }


def flip_one_bit(payload: bytes, start: int, end: int) -> bytes:
    if not 0 <= start < end <= len(payload):
        raise QBW1BuildError("mutation span is empty or outside payload")
    mutated = bytearray(payload)
    mutated[start] ^= 0x01
    return bytes(mutated)


def mutation_outcome(mutated: bytes, model: packet.QBW1Model, original: packet.DecodedRecord) -> str:
    try:
        decoded = packet.decode_record(mutated, model)
    except packet.QBW1FormatError:
        return "REFUSED"
    if decoded == original:
        raise QBW1BuildError("one-bit mutation was accepted without changing the declared object")
    return "DECLARED_OBJECT_CHANGED"


def _stratum_lookup(stage0: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["stratum_id"]: row for row in stage0["strata"]}


def stage_02() -> dict[str, Any]:
    stage0 = stage_00()
    stage1 = stage_01()
    checkpoint = STORE / "stage_02_encode_decode" / "STAGE_02_CHECKPOINT.json"
    storage_preflight()
    require_sources()
    resumed = load_checkpoint(checkpoint, "ddm_qbw1_stage_02_encode_decode.v1")
    if resumed is not None:
        return resumed
    started = time.time()
    storage_preflight()
    winner = int(stage1["winner_capacity"])
    candidate_root = STORE / "stage_01_grammar_fit" / "candidates" / f"dict_{winner:05d}"
    model_bytes = (candidate_root / "model.qbm").read_bytes()
    model = packet.QBW1Model.from_bytes(model_bytes)
    model_fact = atomic_bytes_once(
        STORE / "stage_02_encode_decode" / "primary" / "model.qbm", model_bytes
    )
    repeat_model_fact = atomic_bytes_once(
        STORE / "stage_02_encode_decode" / "repeat" / "model.qbm", model.to_bytes()
    )
    if model_fact["sha256"] != repeat_model_fact["sha256"]:
        raise QBW1BuildError("shared model repeat is not byte-identical")
    mutated_model = flip_one_bit(model_bytes, _model_mutation_offset(model_bytes), len(model_bytes))
    mutated_model_fact = atomic_bytes_once(
        STORE / "stage_02_encode_decode" / "mutations" / "model_one_bit.qbm", mutated_model
    )
    try:
        packet.QBW1Model.from_bytes(mutated_model).to_bytes()
    except packet.QBW1FormatError:
        model_mutation_outcome = "REFUSED"
    else:
        if mutated_model == model_bytes:
            raise QBW1BuildError("model mutation did not change bytes")
        model_mutation_outcome = "DECLARED_MODEL_CHANGED"

    field = source_memmap()
    rows: list[dict[str, Any]] = []
    all_mutations: list[dict[str, Any]] = [
        {"section": "shared_model", "payload": mutated_model_fact, "outcome": model_mutation_outcome}
    ]
    for selection in stage0["selected"]:
        pair_id = int(selection["pair_id"])
        declared_dir = STORE / "stage_01_grammar_fit" / "declared_objects" / f"pair_{pair_id:04d}"
        raw = {
            section_id: (declared_dir / f"section_{section_id}.raw").read_bytes()
            for section_id in packet.SECTION_IDS
        }
        chains = packet.decode_chains(raw[packet.SECTION_BASE_CRACK_CHAINS])
        seeds = packet.decode_seed_labels(raw[packet.SECTION_REGION_SEEDS])
        lane_events = packet.decode_lane_events(raw[packet.SECTION_LANE_DASH_EVENTS])
        primary_bytes = packet.encode_record(pair_id, model, chains, seeds, lane_events)
        repeat_bytes = packet.encode_record(pair_id, model, chains, seeds, lane_events)
        primary = atomic_bytes_once(
            STORE / "stage_02_encode_decode" / "primary" / "records" / f"pair_{pair_id:04d}.qbr",
            primary_bytes,
        )
        repeat = atomic_bytes_once(
            STORE / "stage_02_encode_decode" / "repeat" / "records" / f"pair_{pair_id:04d}.qbr",
            repeat_bytes,
        )
        if primary["sha256"] != repeat["sha256"]:
            raise QBW1BuildError(f"pair {pair_id} repeat is not byte-identical")
        stage1_path = candidate_root / "records" / f"pair_{pair_id:04d}.qbr"
        if primary["sha256"] != sha256_file(stage1_path):
            raise QBW1BuildError(f"pair {pair_id} stage01/stage02 encoding drift")
        decoded = packet.decode_record(primary_bytes, model)
        if tuple(decoded.raw_sections) != tuple((section_id, raw[section_id]) for section_id in packet.SECTION_IDS):
            raise QBW1BuildError(f"pair {pair_id} parseback raw sections differ")
        receiver = packet.decode_receiver(primary_bytes, model)
        source = np.asarray(field[pair_id]).copy()
        expected_base = source.copy()
        expected_base[expected_base == 1] = 0
        if not np.array_equal(receiver["base_field"], expected_base):
            raise QBW1BuildError(f"pair {pair_id} integrated base differs from encoder declaration")
        inputs = boundary_inputs(expected_base)
        observability = atomic_npz_once(
            STORE / "stage_02_encode_decode" / "observability" / f"pair_{pair_id:04d}.npz",
            source_field_u8=source,
            integrated_cells_i32=np.asarray(receiver["cells"], dtype=np.int32),
            decoded_base_u8=np.asarray(receiver["base_field"], dtype=np.uint8),
            decoded_lane_mask_u8=np.asarray(receiver["lane_mask"], dtype=np.uint8),
            decoded_categorical_u8=np.asarray(receiver["categorical_field"], dtype=np.uint8),
            **inputs,
        )
        mutations: list[dict[str, Any]] = []
        for section_name, start, end in packet.section_spans(primary_bytes):
            mutated = flip_one_bit(primary_bytes, start, end)
            mutation_fact = atomic_bytes_once(
                STORE
                / "stage_02_encode_decode"
                / "mutations"
                / f"pair_{pair_id:04d}_{section_name}_one_bit.qbr",
                mutated,
            )
            outcome = mutation_outcome(mutated, model, decoded)
            mutation = {"section": section_name, "payload": mutation_fact, "outcome": outcome}
            mutations.append(mutation)
            all_mutations.append({"pair_id": pair_id, **mutation})
        total_interfaces, road_lane, non_lane = all_interface_counts(source)
        row = {
            "pair_id": pair_id,
            "stratum_id": selection["stratum_id"],
            "population_size": selection["population_size"],
            "sample_size": selection["sample_size"],
            "ht_weight": selection["population_size"] / selection["sample_size"],
            "record_bytes": primary["bytes"],
            "source_interface_length": total_interfaces,
            "road_lane_interface_length": road_lane,
            "non_lane_interface_length": non_lane,
            "bytes_per_source_interface": primary["bytes"] / total_interfaces,
            "primary": primary,
            "repeat": repeat,
            "observability": observability,
            "parseback_exact": True,
            "repeat_byte_identical": True,
            "mutation_rows": mutations,
        }
        atomic_json_once(
            STORE / "stage_02_encode_decode" / "pair_rows" / f"pair_{pair_id:04d}.json", row
        )
        rows.append(row)

    variable_hat = sum(row["ht_weight"] * row["record_bytes"] for row in rows)
    interface_hat = sum(row["ht_weight"] * row["source_interface_length"] for row in rows)
    road_lane_hat = sum(row["ht_weight"] * row["road_lane_interface_length"] for row in rows)
    non_lane_hat = sum(row["ht_weight"] * row["non_lane_interface_length"] for row in rows)
    quotient_hat = model_fact["bytes"] + math.ceil(variable_hat)
    complete_projection = FIXED_ENVELOPE_BYTES + quotient_hat
    strata = _stratum_lookup(stage0)
    stratum_rows = []
    for stratum_id, stratum in strata.items():
        members = [row for row in rows if row["stratum_id"] == stratum_id]
        stratum_rows.append(
            {
                "stratum_id": stratum_id,
                "population_size": stratum["population_size"],
                "sample_size": stratum["sample_size"],
                "selected_pair_ids": [row["pair_id"] for row in members],
                "selected_record_bytes": sum(row["record_bytes"] for row in members),
                "projected_record_bytes": sum(row["record_bytes"] * row["ht_weight"] for row in members),
                "selected_interface_length": sum(row["source_interface_length"] for row in members),
                "projected_interface_length": sum(
                    row["source_interface_length"] * row["ht_weight"] for row in members
                ),
            }
        )
    result = {
        "schema": "ddm_qbw1_stage_02_encode_decode.v1",
        "complete": True,
        "stage": "stage_02_encode_decode",
        "arm": ARM,
        "axis": "[macOS-CPU scorer-free advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "contest_eval_run": False,
        "git_head": git_head(),
        "source_field_sha256": SOURCE_SHA256,
        "config_sha256": sha256_file(STORE / "CONFIG.json"),
        "schema_doc_sha256": sha256_file(SCHEMA_DOC),
        "shared_model": model_fact,
        "shared_model_repeat": repeat_model_fact,
        "winner_dictionary_capacity": winner,
        "B_shared": model_fact["bytes"],
        "B_var_hat": variable_hat,
        "B_hat_quotient": quotient_hat,
        "fixed_envelope_projection_bytes": FIXED_ENVELOPE_BYTES,
        "complete_archive_projection_bytes": complete_projection,
        "quotient_allowance_bytes": QUOTIENT_ALLOWANCE_BYTES,
        "complete_cap_bytes": COMPLETE_CAP_BYTES,
        "rate_leg_pass": complete_projection <= COMPLETE_CAP_BYTES,
        "interface_length_hat": interface_hat,
        "road_lane_interface_length_hat": road_lane_hat,
        "non_lane_interface_length_hat": non_lane_hat,
        "serialized_variable_bytes_per_interface": variable_hat / interface_hat,
        "serialized_quotient_bytes_per_interface_including_shared": quotient_hat / interface_hat,
        "parseback_exact_all": all(row["parseback_exact"] for row in rows),
        "repeat_byte_identical_all": all(row["repeat_byte_identical"] for row in rows),
        "mutation_contract_all": all(
            row["outcome"] in {"REFUSED", "DECLARED_OBJECT_CHANGED", "DECLARED_MODEL_CHANGED"}
            for row in all_mutations
        ),
        "pair_rows": rows,
        "stratum_rows": stratum_rows,
        "mutation_rows": all_mutations,
        "distortion_not_measured": True,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_once(STORE / "RESULT_STAGE02.json", result)
    atomic_json_once(checkpoint, result)
    write_jsonl_surfaces(result)
    return result


def _model_mutation_offset(model_bytes: bytes) -> int:
    # Dictionary when present; otherwise its protected CRC field.
    return 16 if len(model_bytes) > 16 else 12


def write_jsonl_surfaces(result: dict[str, Any]) -> None:
    pair_lines = b"".join(canonical_json_bytes(row) for row in result["pair_rows"])
    stratum_lines = b"".join(canonical_json_bytes(row) for row in result["stratum_rows"])
    mutation_lines = b"".join(canonical_json_bytes(row) for row in result["mutation_rows"])
    root = STORE / "stage_02_encode_decode" / "query"
    atomic_bytes_once(root / "pair_facts.jsonl", pair_lines)
    atomic_bytes_once(root / "stratum_facts.jsonl", stratum_lines)
    atomic_bytes_once(root / "mutation_facts.jsonl", mutation_lines)


def memory_preflight_and_schedule() -> dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    memory = int(virtual_memory.total)
    available = int(virtual_memory.available)
    # WD3 observed 107.52 GiB at chunk 60; chunk 30 is the proven cure.  n32 executes 30+2.
    projected_chunk_gib = 107.52 * 30 / 60
    conservative_peak_gib = min(96.0, projected_chunk_gib + 32.0)
    schedule_seconds = 7_372 * (32 / 60)
    return {
        "schema": "ddm_qbw1_stage03_memory_schedule_preflight.v1",
        "real_config": {
            "n_pairs": 32,
            "frame_shape": [2, 3, 874, 1164],
            "full_autograd_chunk_pairs": 30,
            "chunk_partition": [30, 2],
            "epochs": 65,
            "periodic_checkpoint_epochs": 5,
            "stage_boundaries": ["interior_birth", "joint_ce", "pose_finish", "quantize_export"],
            "ema_shadow_required": True,
        },
        "operator_memory_ceiling_gib": 116.0,
        "wd3_chunk60_measured_watermark_gib": 107.52,
        "chunk30_linear_live_set_projection_gib": projected_chunk_gib,
        "conservative_peak_projection_gib": conservative_peak_gib,
        "projection_below_operator_ceiling": conservative_peak_gib <= 116.0,
        "host_total_bytes": memory,
        "host_available_bytes_at_seal": available,
        "schedule_projection_seconds": schedule_seconds,
        "schedule_projection_basis": "W96B 65 epochs n60 measured 7372 s, scaled linearly to n32",
        "fire_requires_live_recheck": True,
    }


def sealed_fire_order(stage2: dict[str, Any]) -> dict[str, Any]:
    order_path = STORE / "sealed_main_fire_order" / "FIRE_ORDER.json"
    if order_path.exists():
        sealed = json.loads(order_path.read_text())
        if sealed.get("schema") != "ddm_qbw1_sealed_main_fire_order.v1" or sealed.get("sealed") is not True:
            raise QBW1BuildError("existing MAIN fire order is malformed or unsealed")
        return sealed
    memory = memory_preflight_and_schedule()
    config = {
        "schema": "ddm_qbw1_main_stage03_05_config.v1",
        "arm": ARM,
        "seed": SEED,
        "input_stage02_checkpoint": file_fact(
            STORE / "stage_02_encode_decode" / "STAGE_02_CHECKPOINT.json"
        ),
        "selected_pair_ids": [row["pair_id"] for row in stage2["pair_rows"]],
        "device": "mps",
        "axis": "[Darwin-MPS training / macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "chunk_pairs": 30,
        "epochs": 65,
        "checkpoint_every_epochs": 5,
        "ema_decay": 0.999,
        "boundary_interior_split_required": True,
        "interior_owns_luma_photometry": True,
        "boundary_owns_seg_action": True,
        "fixed_palette_forbidden": True,
        "gt_decode": "upstream/frame_utils.py:yuv420_to_rgb",
        "matched_control": {
            "archive": file_fact(GB1_ARCHIVE),
            "score": GB1_SCORE,
            "axis": "[contest-CUDA T4 n600] source anchor; control is remeasured n32 advisory",
        },
        "stages": {
            "stage_03_joint_realizer": {
                "resume_from": str(STORE / "stage_03_joint_realizer"),
                "distinct_stage_checkpoints": True,
                "ema_shadow": True,
                "retain_every_model_and_optimizer_state": True,
            },
            "stage_04_render_score": {
                "retain_frames_before_scorer": True,
                "retain_logits_argmax_pose6_per_pair": True,
                "scorer_device": "cpu",
                "matched_gb1_same_pairs": True,
            },
            "stage_05_gate": {
                "conditions": [
                    "parseback_exact",
                    "repeat_byte_identical",
                    "one_bit_mutation_contract",
                    "B_hat<=137986",
                    "dpose_hat<=1.25e-4",
                    "S_hat<0.12",
                    "complete_action_beats_matched_GB1_n32",
                ],
                "score_formula": "100*dseg_hat+sqrt(10*dpose_hat)+25*B_hat/37545489",
            },
        },
        "counterfactual_hooks": json.loads((STORE / "CONFIG.json").read_text())["counterfactual_hooks"],
    }
    config_fact = atomic_json_once(STORE / "sealed_main_fire_order" / "STAGE03_05_CONFIG.json", config)
    trigger_pass = bool(
        stage2["parseback_exact_all"]
        and stage2["repeat_byte_identical_all"]
        and stage2["mutation_contract_all"]
        and stage2["rate_leg_pass"]
        and memory["projection_below_operator_ceiling"]
    )
    disposition = "QUEUED-WITH-A-FIRE-ORDER" if trigger_pass else "FOLDED_BY_SCORER_FREE_GATE"
    order = {
        "schema": "ddm_qbw1_sealed_main_fire_order.v1",
        "sealed": True,
        "disposition": disposition,
        "owner": "MAIN quotient-body joint-realizer operator",
        "consumer_store": str(STORE),
        "config": config_fact,
        "memory_preflight": memory,
        "fire_trigger": {
            "no_duplicate_active_lane": True,
            "live_storage_preflight_required": True,
            "stage02_rate_leg_pass": stage2["rate_leg_pass"],
            "stage02_receiver_contract_pass": (
                stage2["parseback_exact_all"]
                and stage2["repeat_byte_identical_all"]
                and stage2["mutation_contract_all"]
            ),
            "memory_projection_pass": memory["projection_below_operator_ceiling"],
            "main_operator_must_claim_metal_and_scorer_lanes": True,
        },
        "arm_must_not_launch": True,
        "training_launched": False,
        "scorer_gate_run": False,
    }
    atomic_json_once(order_path, order)
    return order


def custody_manifest() -> dict[str, Any]:
    manifest_path = STORE / "CUSTODY_MANIFEST.json"
    if manifest_path.exists():
        retained = json.loads(manifest_path.read_text())
        if retained.get("schema") != "ddm_qbw1_payload_custody_manifest.v1":
            raise QBW1BuildError("existing custody manifest has the wrong schema")
        for fact in retained.get("files", []):
            path = Path(fact["path"])
            if not path.is_file() or file_fact(path) != fact:
                raise QBW1BuildError(f"custody manifest payload drift: {path}")
        return retained
    files = []
    for path in sorted(STORE.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith("._")
            or ".part" in path.name
            or path.name == "CUSTODY_MANIFEST.json"
        ):
            continue
        files.append(file_fact(path))
    manifest = {
        "schema": "ddm_qbw1_payload_custody_manifest.v1",
        "root": str(STORE),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "all_payloads_retained": True,
        "cleanup_policy": "success-only atomic .part files are removed; material payloads are never deleted",
        "command": ".venv/bin/python experiments/ddm_qbw1_builder.py run-00-02",
        "git_head": git_head(),
        "upstream": upstream_identity(),
        "platform": platform.platform(),
    }
    atomic_json_once(manifest_path, manifest)
    return manifest


def run_00_02() -> dict[str, Any]:
    stage_00()
    stage_01()
    stage2 = stage_02()
    order = sealed_fire_order(stage2)
    summary = {
        "schema": "ddm_qbw1_arm_result.v1",
        "complete": True,
        "stages_run": ["stage_00_selection", "stage_01_grammar_fit", "stage_02_encode_decode"],
        "stages_not_run": ["stage_03_joint_realizer", "stage_04_render_score", "stage_05_gate"],
        "stage0_checkpoint": file_fact(
            STORE / "stage_00_selection" / "STAGE_00_CHECKPOINT.json"
        ),
        "stage1_checkpoint": file_fact(
            STORE / "stage_01_grammar_fit" / "STAGE_01_CHECKPOINT.json"
        ),
        "stage2_checkpoint": file_fact(
            STORE / "stage_02_encode_decode" / "STAGE_02_CHECKPOINT.json"
        ),
        "fire_order": file_fact(STORE / "sealed_main_fire_order" / "FIRE_ORDER.json"),
        "custody_manifest_path": str(STORE / "CUSTODY_MANIFEST.json"),
        "B_hat_quotient": stage2["B_hat_quotient"],
        "complete_archive_projection_bytes": stage2["complete_archive_projection_bytes"],
        "rate_leg_pass": stage2["rate_leg_pass"],
        "fire_order_disposition": order["disposition"],
        "score_claim": False,
        "pointer_moved": False,
    }
    atomic_json_once(STORE / "RESULT.json", summary)
    custody_manifest()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("stage-00", "stage-01", "stage-02", "seal-fire-order", "run-00-02", "audit"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "stage-00":
        result = stage_00()
    elif args.action == "stage-01":
        result = stage_01()
    elif args.action == "stage-02":
        result = stage_02()
    elif args.action == "seal-fire-order":
        result = sealed_fire_order(stage_02())
    elif args.action == "run-00-02":
        result = run_00_02()
    else:
        result = custody_manifest()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
