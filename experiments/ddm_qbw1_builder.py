"""Resumable QBW1 first-rung builder; this arm runs stages 00--02 only.

The script consumes the pinned 600-pair semantic field, selects the preregistered n32 population,
fits and retains every frozen-schema grammar candidate, then performs real reset-record
encode/decode, repeat, mutation, and observability closure.  It does not load a scorer or launch
training.  Stages 03--05 are represented by the sealed MAIN fire order emitted after stage 02.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import lzma
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
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
QBW2_ROOT = STORE / "qbw2_temporal_bound"
QBW2_CLEARING_BOUND_BYTES = 68_000
GB1_RUNTIME = Path(
    "/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/"
    "runtime_groupbin8_surprise"
)

_BOUND_STREAM_HEADER = struct.Struct(">4sBHHH")
_QBW_OBJECT_PAIR_HEADER = struct.Struct(">HIII")
_INNOVATION_PAIR_HEADER = struct.Struct(">HI")
_ROAD_TOPOLOGY_PAIR_HEADER = struct.Struct(">HI")
_MASK_BYTES = (H * W + 7) // 8
_CODER_NAMES = ("brotli_q11", "lzma9e", "zlib9")


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


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    """Pack a flat unsigned integer array in little-bit order."""
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if not 1 <= bits <= 8:
        raise QBW1BuildError("packed symbol width is outside 1..8")
    if flat.size and int(flat.max()) >= 1 << bits:
        raise QBW1BuildError("packed symbol exceeds declared width")
    bit_rows = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return np.packbits(bit_rows, bitorder="little").tobytes()


def unpack_unsigned(payload: bytes, count: int, bits: int) -> np.ndarray:
    """Inverse of :func:`pack_unsigned`, with canonical padding checks."""
    expected = (count * bits + 7) // 8
    if len(payload) != expected:
        raise QBW1BuildError("packed symbol payload length differs")
    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if np.any(unpacked[count * bits :]):
        raise QBW1BuildError("packed symbol payload has non-zero padding")
    rows = unpacked[: count * bits].reshape(count, bits).astype(np.uint16)
    weights = (1 << np.arange(bits, dtype=np.uint16))[None, :]
    return np.sum(rows * weights, axis=1, dtype=np.uint16).astype(np.uint8)


def encode_qbw_object_pair(pair_id: int, sections: tuple[tuple[int, bytes], ...]) -> bytes:
    by_id = dict(sections)
    if tuple(sorted(by_id)) != packet.SECTION_IDS:
        raise QBW1BuildError("QBW object pair lacks a frozen v1 section")
    ordered = [by_id[section_id] for section_id in packet.SECTION_IDS]
    return _QBW_OBJECT_PAIR_HEADER.pack(pair_id, *(len(raw) for raw in ordered)) + b"".join(ordered)


def decode_qbw_object_pair(payload: bytes) -> tuple[int, tuple[tuple[int, bytes], ...]]:
    if len(payload) < _QBW_OBJECT_PAIR_HEADER.size:
        raise QBW1BuildError("truncated QBW object pair")
    pair_id, *lengths = _QBW_OBJECT_PAIR_HEADER.unpack_from(payload)
    offset = _QBW_OBJECT_PAIR_HEADER.size
    sections = []
    for section_id, length in zip(packet.SECTION_IDS, lengths, strict=True):
        end = offset + length
        if end > len(payload):
            raise QBW1BuildError("truncated QBW object section")
        raw = payload[offset:end]
        if section_id == packet.SECTION_BASE_CRACK_CHAINS:
            packet.decode_chains(raw)
        elif section_id == packet.SECTION_REGION_SEEDS:
            packet.decode_seed_labels(raw)
        else:
            packet.decode_lane_events(raw)
        sections.append((section_id, raw))
        offset = end
    if offset != len(payload):
        raise QBW1BuildError("QBW object pair has trailing bytes")
    return pair_id, tuple(sections)


def encode_bound_stream(magic: bytes, pair_blobs: list[bytes]) -> bytes:
    if len(magic) != 4 or len(pair_blobs) > 0xFFFF:
        raise QBW1BuildError("invalid bound stream header")
    return _BOUND_STREAM_HEADER.pack(magic, 1, len(pair_blobs), H, W) + b"".join(pair_blobs)


def _compress_bound_payload(raw: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return brotli.compress(raw, mode=brotli.MODE_GENERIC, quality=11)
    if coder == "lzma9e":
        return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    if coder == "zlib9":
        return zlib.compress(raw, level=9)
    raise QBW1BuildError(f"unknown bound coder: {coder}")


def _decompress_bound_payload(payload: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return brotli.decompress(payload)
    if coder == "lzma9e":
        return lzma.decompress(payload, format=lzma.FORMAT_XZ)
    if coder == "zlib9":
        return zlib.decompress(payload)
    raise QBW1BuildError(f"unknown bound coder: {coder}")


def retain_coder_race(root: Path, raw: bytes) -> dict[str, Any]:
    """Persist one real input and every real-coder primary/repeat payload."""
    input_fact = atomic_bytes_once(root / "input.bin", raw)
    rows = []
    for coder in _CODER_NAMES:
        encoded = _compress_bound_payload(raw, coder)
        repeat = _compress_bound_payload(raw, coder)
        if encoded != repeat:
            raise QBW1BuildError(f"{coder} repeat differs")
        if _decompress_bound_payload(encoded, coder) != raw:
            raise QBW1BuildError(f"{coder} decode differs")
        primary_fact = atomic_bytes_once(root / f"payload.{coder}", encoded)
        repeat_fact = atomic_bytes_once(root / f"payload.repeat.{coder}", repeat)
        rows.append(
            {
                "coder": coder,
                "payload": primary_fact,
                "repeat": repeat_fact,
                "decode_exact": True,
                "repeat_byte_identical": True,
            }
        )
    winner = min(rows, key=lambda row: (row["payload"]["bytes"], row["coder"]))
    result = {
        "schema": "ddm_qbw2_real_coder_race.v1",
        "input": input_fact,
        "rows": rows,
        "winner": winner,
    }
    atomic_json_once(root / "RACE.json", result)
    return result


def shift_with_colocated_fallback(field: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a field while retaining the co-located prior outside overlap."""
    if field.ndim != 2:
        raise QBW1BuildError("translation expects a two-dimensional field")
    height, width = field.shape
    result = field.copy()
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    if src_y0 < src_y1 and src_x0 < src_x1:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = field[src_y0:src_y1, src_x0:src_x1]
    return result


def estimate_carrier_shift(previous: np.ndarray, current: np.ndarray, radius: int = 4) -> tuple[int, int]:
    """Find the deterministic integer translation minimizing carried-state RGB error."""
    if previous.shape != current.shape or previous.ndim != 3:
        raise QBW1BuildError("GB1 carried-state pair geometry differs")
    height, width, _channels = previous.shape
    best: tuple[int, int, int, int] | None = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            src_y0, src_y1 = max(0, -dy), min(height, height - dy)
            src_x0, src_x1 = max(0, -dx), min(width, width - dx)
            dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
            dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
            source = previous[src_y0:src_y1, src_x0:src_x1].astype(np.int32)
            target = current[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.int32)
            squared_error = int(np.square(source - target, dtype=np.int64).sum(dtype=np.int64))
            count = int(source.size)
            candidate = (squared_error, count, dy, dx)
            if best is None:
                best = candidate
                continue
            if squared_error * best[1] < best[0] * count or (
                squared_error * best[1] == best[0] * count and (dy, dx) < (best[2], best[3])
            ):
                best = candidate
    if best is None:
        raise QBW1BuildError("GB1 carried-state translation search was empty")
    return best[2], best[3]


def load_gb1_carried_state() -> tuple[np.ndarray, dict[str, Any]]:
    """Decode the actual GB1 12-D pose-carrier state; never substitute GT Pose6."""
    if sha256_file(GB1_ARCHIVE) != GB1_SHA256:
        raise QBW1BuildError("GB1 archive drift before carried-state decode")
    runtime_path = str(GB1_RUNTIME)
    cpr1_path = str(GB1_RUNTIME / "cpr1")
    for entry in (runtime_path, cpr1_path):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    inflate = importlib.import_module("runtime.f26_inflate")
    carrier_repack = importlib.import_module("runtime.carrier_repack")
    parts = inflate.read_residual_archive(GB1_ARCHIVE)
    renderer = inflate._load_renderer(GB1_RUNTIME / "cpr1")
    carrier_blob, _selector = carrier_repack.split_frame0_selector_carrier(parts.carrier_blob)
    canonical = carrier_repack.materialize_cpr1(carrier_blob, renderer)
    semantic_pose = (
        struct.pack("<II", len(parts.semantic_blob), len(canonical))
        + parts.semantic_blob
        + canonical
    )
    _semantic, basis_t, coefficients_t = renderer.unpack_semantic_pose(semantic_pose)
    basis = basis_t.detach().cpu().numpy().astype(np.float32, copy=False)
    coefficients = coefficients_t.detach().cpu().numpy().astype(np.float32, copy=False)
    if basis.shape != (12, 3, 24, 32) or coefficients.shape != (N, 12):
        raise QBW1BuildError("GB1 carried pose-carrier tensors have unexpected geometry")
    lowres = np.einsum("nd,dchw->nchw", coefficients, basis, optimize=False) / math.sqrt(12.0)
    lowres = np.clip(np.rint(127.5 + 64.0 * lowres), 0, 255).astype(np.uint8)
    lowres = np.transpose(lowres, (0, 2, 3, 1)).copy()
    retained = atomic_npz_once(
        QBW2_ROOT / "inputs" / "gb1_carried_pose_state.npz",
        carrier_lowres_u8=lowres,
        coefficients_f32=coefficients,
        basis_f32=basis,
    )
    return lowres, {
        "archive": file_fact(GB1_ARCHIVE),
        "runtime_root": str(GB1_RUNTIME),
        "retained_decoded_state": retained,
        "source_kind": "GB1 decoded 12-D pose-carrier state",
        "geometric_pose6_available_in_gb1": False,
        "gt_pose_substituted": False,
    }


def carried_state_shifts(lowres: np.ndarray) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    shifts = [(0, 0)]
    rows = [{"pair_id": 0, "carrier_dy": 0, "carrier_dx": 0, "field_dy": 0, "field_dx": 0}]
    scale_y, scale_x = H // lowres.shape[1], W // lowres.shape[2]
    if (scale_y, scale_x) != (16, 16):
        raise QBW1BuildError("GB1 carried-state grid does not scale exactly to quotient field")
    for pair_id in range(1, N):
        dy, dx = estimate_carrier_shift(lowres[pair_id - 1], lowres[pair_id])
        field_shift = (dy * scale_y, dx * scale_x)
        shifts.append(field_shift)
        rows.append(
            {
                "pair_id": pair_id,
                "carrier_dy": dy,
                "carrier_dx": dx,
                "field_dy": field_shift[0],
                "field_dx": field_shift[1],
            }
        )
    atomic_bytes_once(
        QBW2_ROOT / "inputs" / "gb1_carried_pose_shifts.jsonl",
        b"".join(canonical_json_bytes(row) for row in rows),
    )
    return shifts, rows


def innovation_pair_blob(pair_id: int, current: np.ndarray, prediction: np.ndarray) -> bytes:
    mismatch = np.asarray(current != prediction, dtype=np.uint8).reshape(-1)
    mask = pack_unsigned(mismatch, 1)
    labels = np.asarray(current, dtype=np.uint8).reshape(-1)[mismatch.astype(bool)]
    return _INNOVATION_PAIR_HEADER.pack(pair_id, labels.size) + mask + pack_unsigned(labels, 3)


def decode_innovation_pair(payload: bytes, prediction: np.ndarray) -> tuple[int, np.ndarray]:
    if len(payload) < _INNOVATION_PAIR_HEADER.size + _MASK_BYTES:
        raise QBW1BuildError("truncated temporal innovation pair")
    pair_id, mismatch_count = _INNOVATION_PAIR_HEADER.unpack_from(payload)
    offset = _INNOVATION_PAIR_HEADER.size
    mismatch = unpack_unsigned(payload[offset : offset + _MASK_BYTES], H * W, 1).astype(bool)
    if int(mismatch.sum()) != mismatch_count:
        raise QBW1BuildError("temporal innovation mismatch count differs")
    offset += _MASK_BYTES
    labels = unpack_unsigned(payload[offset:], mismatch_count, 3)
    if labels.size and int(labels.max()) > 4:
        raise QBW1BuildError("temporal innovation contains a noncanonical label")
    restored = np.asarray(prediction, dtype=np.uint8).copy().reshape(-1)
    restored[mismatch] = labels
    return pair_id, restored.reshape(H, W)


def road_topology_pair_blob(pair_id: int, current: np.ndarray) -> bytes:
    flat = np.asarray(current, dtype=np.uint8).reshape(-1)
    road = flat == 0
    exceptions = flat[~road]
    if exceptions.size and not set(np.unique(exceptions).tolist()).issubset({1, 2, 3, 4}):
        raise QBW1BuildError("Road topology exception contains a noncanonical label")
    return (
        _ROAD_TOPOLOGY_PAIR_HEADER.pack(pair_id, exceptions.size)
        + pack_unsigned(road.astype(np.uint8), 1)
        + pack_unsigned(exceptions - 1, 2)
    )


def decode_road_topology_pair(payload: bytes) -> tuple[int, np.ndarray]:
    if len(payload) < _ROAD_TOPOLOGY_PAIR_HEADER.size + _MASK_BYTES:
        raise QBW1BuildError("truncated Road topology pair")
    pair_id, exception_count = _ROAD_TOPOLOGY_PAIR_HEADER.unpack_from(payload)
    offset = _ROAD_TOPOLOGY_PAIR_HEADER.size
    road = unpack_unsigned(payload[offset : offset + _MASK_BYTES], H * W, 1).astype(bool)
    if int((~road).sum()) != exception_count:
        raise QBW1BuildError("Road topology exception count differs")
    offset += _MASK_BYTES
    exceptions = unpack_unsigned(payload[offset:], exception_count, 2) + 1
    restored = np.zeros(H * W, dtype=np.uint8)
    restored[~road] = exceptions
    return pair_id, restored.reshape(H, W)


def road_interface_facts(field: np.ndarray) -> tuple[int, int]:
    left, right = field[:, :-1], field[:, 1:]
    top, bottom = field[:-1, :], field[1:, :]
    right_diff = left != right
    down_diff = top != bottom
    total = int(right_diff.sum(dtype=np.int64) + down_diff.sum(dtype=np.int64))
    road_touch = int(
        (right_diff & ((left == 0) | (right == 0))).sum(dtype=np.int64)
        + (down_diff & ((top == 0) | (bottom == 0))).sum(dtype=np.int64)
    )
    return total, road_touch


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


def _qbw2_pair_blobs(
    pair_ids: list[int],
    field: np.memmap,
    shifts: list[tuple[int, int]],
) -> tuple[dict[str, list[bytes]], list[dict[str, Any]]]:
    blobs: dict[str, list[bytes]] = {"joint": [], "conditional": [], "road_topology": []}
    rows = []
    for pair_id in pair_ids:
        current = np.asarray(field[pair_id]).copy()
        obj = extract_object(current)
        sections = raw_sections(obj)
        object_blob = encode_qbw_object_pair(pair_id, sections)
        restored_pair, restored_sections = decode_qbw_object_pair(object_blob)
        if restored_pair != pair_id or restored_sections != sections:
            raise QBW1BuildError(f"pair {pair_id} QBW logical stream differs")

        if pair_id == 0:
            prediction = np.zeros((H, W), dtype=np.uint8)
        else:
            previous = np.asarray(field[pair_id - 1]).copy()
            prediction = shift_with_colocated_fallback(previous, *shifts[pair_id])
        conditional_blob = innovation_pair_blob(pair_id, current, prediction)
        decoded_pair, conditional_restored = decode_innovation_pair(conditional_blob, prediction)
        if decoded_pair != pair_id or not np.array_equal(conditional_restored, current):
            raise QBW1BuildError(f"pair {pair_id} temporal innovation decode differs")

        topology_blob = road_topology_pair_blob(pair_id, current)
        topology_pair, topology_restored = decode_road_topology_pair(topology_blob)
        if topology_pair != pair_id or not np.array_equal(topology_restored, current):
            raise QBW1BuildError(f"pair {pair_id} Road topology decode differs")
        total_interfaces, road_touch = road_interface_facts(current)
        mismatch_count = int(np.count_nonzero(current != prediction))
        blobs["joint"].append(object_blob)
        blobs["conditional"].append(conditional_blob)
        blobs["road_topology"].append(topology_blob)
        rows.append(
            {
                "pair_id": pair_id,
                "joint_raw_bytes": len(object_blob),
                "conditional_raw_bytes": len(conditional_blob),
                "road_topology_raw_bytes": len(topology_blob),
                "conditional_mismatch_count": mismatch_count,
                "conditional_mismatch_fraction": mismatch_count / (H * W),
                "field_shift_dy": shifts[pair_id][0],
                "field_shift_dx": shifts[pair_id][1],
                "source_interface_count": total_interfaces,
                "road_touch_interface_count": road_touch,
                "road_topology_determinism_fraction": road_touch / total_interfaces,
                "conditional_decode_exact": True,
                "road_topology_decode_exact": True,
            }
        )
    return blobs, rows


def _retain_n32_bound(
    blobs: dict[str, list[bytes]],
    pair_rows: list[dict[str, Any]],
    selection: list[dict[str, Any]],
) -> dict[str, Any]:
    selection_by_pair = {int(row["pair_id"]): row for row in selection}
    pair_row_by_pair = {int(row["pair_id"]): row for row in pair_rows}
    leg_results: dict[str, Any] = {}
    leg_magic = {"joint": b"QBJ2", "conditional": b"QBC2", "road_topology": b"QBT2"}
    for leg, leg_blobs in blobs.items():
        races = []
        for blob in leg_blobs:
            if leg == "joint":
                pair_id, _sections = decode_qbw_object_pair(blob)
            else:
                pair_id = struct.unpack_from(">H", blob)[0]
            race = retain_coder_race(
                QBW2_ROOT / "n32" / "per_pair" / f"pair_{pair_id:04d}" / leg,
                blob,
            )
            winner_bytes = int(race["winner"]["payload"]["bytes"])
            selection_row = selection_by_pair[pair_id]
            pair_row_by_pair[pair_id][f"{leg}_independent_winner_coder"] = race["winner"]["coder"]
            pair_row_by_pair[pair_id][f"{leg}_independent_winner_bytes"] = winner_bytes
            pair_row_by_pair[pair_id]["stratum_id"] = selection_row["stratum_id"]
            pair_row_by_pair[pair_id]["ht_weight"] = (
                selection_row["population_size"] / selection_row["sample_size"]
            )
            races.append((pair_id, winner_bytes, race))
        joint_stream = encode_bound_stream(leg_magic[leg], leg_blobs)
        joint_race = retain_coder_race(QBW2_ROOT / "n32" / "joint_races" / leg, joint_stream)
        selected_independent = sum(row[1] for row in races)
        ht_independent = sum(
            pair_row_by_pair[pair_id]["ht_weight"] * winner_bytes
            for pair_id, winner_bytes, _race in races
        )
        context_ratio = joint_race["winner"]["payload"]["bytes"] / selected_independent
        leg_results[leg] = {
            "selected_independent_winner_bytes": selected_independent,
            "ht_independent_projection_bytes": ht_independent,
            "joint_context_winner": joint_race["winner"],
            "joint_context_ratio_vs_independent": context_ratio,
            "n600_ratio_ht_projection_bytes": math.ceil(context_ratio * ht_independent),
            "projection_method": (
                "HT independent per-pair real-coder bytes multiplied by the n32 joint-context/"
                "independent ratio; ratio estimator, not a codec or theorem"
            ),
        }
    pair_rows.sort(key=lambda row: row["pair_id"])
    atomic_bytes_once(
        QBW2_ROOT / "n32" / "pair_rows.jsonl",
        b"".join(canonical_json_bytes(row) for row in pair_rows),
    )
    result = {
        "schema": "ddm_qbw2_n32_bound.v1",
        "axis": "[macOS-CPU scorer-free advisory, seeded-stratified random n32]",
        "score_claim": False,
        "pair_rows": pair_rows,
        "legs": leg_results,
    }
    atomic_json_once(QBW2_ROOT / "n32" / "RESULT.json", result)
    return result


def _retain_n600_bound(
    blobs: dict[str, list[bytes]], pair_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    leg_magic = {"joint": b"QBJ2", "conditional": b"QBC2", "road_topology": b"QBT2"}
    legs = {}
    for leg, leg_blobs in blobs.items():
        stream = encode_bound_stream(leg_magic[leg], leg_blobs)
        race = retain_coder_race(QBW2_ROOT / "n600" / "joint_races" / leg, stream)
        legs[leg] = {
            "input_bytes": len(stream),
            "winner": race["winner"],
            "n600_quotient_bound_bytes": int(race["winner"]["payload"]["bytes"]),
            "all_coder_rows": race["rows"],
            "decode_exact": True,
        }
    interface_total = sum(row["source_interface_count"] for row in pair_rows)
    road_touch_total = sum(row["road_touch_interface_count"] for row in pair_rows)
    mismatch_total = sum(row["conditional_mismatch_count"] for row in pair_rows)
    atomic_bytes_once(
        QBW2_ROOT / "n600" / "pair_rows.jsonl",
        b"".join(canonical_json_bytes(row) for row in pair_rows),
    )
    result = {
        "schema": "ddm_qbw2_n600_bound.v1",
        "axis": "[macOS-CPU scorer-free advisory, full n600]",
        "score_claim": False,
        "legs": legs,
        "pair_count": len(pair_rows),
        "conditional_mismatch_count": mismatch_total,
        "conditional_mismatch_fraction": mismatch_total / (N * H * W),
        "source_interface_count": interface_total,
        "road_touch_interface_count": road_touch_total,
        "road_topology_determinism_fraction": road_touch_total / interface_total,
        "pair_rows_path": str(QBW2_ROOT / "n600" / "pair_rows.jsonl"),
    }
    atomic_json_once(QBW2_ROOT / "n600" / "RESULT.json", result)
    return result


def qbw2_custody_manifest() -> dict[str, Any]:
    manifest_path = QBW2_ROOT / "QBW2_CUSTODY_MANIFEST.json"
    if manifest_path.exists():
        retained = json.loads(manifest_path.read_text())
        if retained.get("schema") != "ddm_qbw2_payload_custody_manifest.v1":
            raise QBW1BuildError("existing QBW2 custody manifest has the wrong schema")
        for fact in retained.get("files", []):
            path = Path(fact["path"])
            if not path.is_file() or file_fact(path) != fact:
                raise QBW1BuildError(f"QBW2 custody payload drift: {path}")
        return retained
    files = []
    for path in sorted(QBW2_ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith("._")
            or ".part" in path.name
            or path.name == manifest_path.name
        ):
            continue
        files.append(file_fact(path))
    manifest = {
        "schema": "ddm_qbw2_payload_custody_manifest.v1",
        "root": str(QBW2_ROOT),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "all_payloads_retained": True,
        "cleanup_policy": "success-only atomic .part files are removed; material payloads are never deleted",
        "command": ".venv/bin/python experiments/ddm_qbw1_builder.py run-qbw2-bound",
        "git_head": git_head(),
        "upstream": upstream_identity(),
        "platform": platform.platform(),
    }
    atomic_json_once(manifest_path, manifest)
    return manifest


def run_qbw2_bound() -> dict[str, Any]:
    checkpoint = QBW2_ROOT / "QBW2_BOUND_CHECKPOINT.json"
    storage = storage_preflight()
    require_sources()
    resumed = load_checkpoint(checkpoint, "ddm_qbw2_temporal_bound.v1")
    if resumed is not None:
        qbw2_custody_manifest()
        return resumed
    stage0 = stage_00()
    stage_01()
    stage_02()
    started = time.time()
    lowres, carried_state = load_gb1_carried_state()
    shifts, shift_rows = carried_state_shifts(lowres)
    field = source_memmap()
    selected_ids = [int(row["pair_id"]) for row in stage0["selected"]]
    n32_blobs, n32_rows = _qbw2_pair_blobs(selected_ids, field, shifts)
    n32_result = _retain_n32_bound(n32_blobs, n32_rows, stage0["selected"])
    n600_blobs, n600_rows = _qbw2_pair_blobs(list(range(N)), field, shifts)
    n600_result = _retain_n600_bound(n600_blobs, n600_rows)
    bound_rows = []
    names = {
        "joint": "joint context compression of time-ordered QBW1 logical sections",
        "conditional": (
            "exact innovations after GB1 carried pose-carrier translation proxy; "
            "not a QA39 ground-homography warp"
        ),
        "road_topology": "exact Road topology plus categorical exception stream",
    }
    for leg in ("joint", "conditional", "road_topology"):
        qa39_contract_pass = leg != "conditional"
        bound_rows.append(
            {
                "leg": leg,
                "mechanism": names[leg],
                "n32_ratio_ht_projection_bytes": n32_result["legs"][leg][
                    "n600_ratio_ht_projection_bytes"
                ],
                "n600_measured_bound_bytes": n600_result["legs"][leg][
                    "n600_quotient_bound_bytes"
                ],
                "allowance_bytes": QUOTIENT_ALLOWANCE_BYTES,
                "clearing_margin_bar_bytes": QBW2_CLEARING_BOUND_BYTES,
                "clears_allowance": n600_result["legs"][leg]["n600_quotient_bound_bytes"]
                <= QUOTIENT_ALLOWANCE_BYTES,
                "clears_v2_fire_bar": n600_result["legs"][leg]["n600_quotient_bound_bytes"]
                <= QBW2_CLEARING_BOUND_BYTES,
                "gate_eligible": qa39_contract_pass,
                "qa39_contract_pass": qa39_contract_pass,
            }
        )
    eligible_rows = [row for row in bound_rows if row["gate_eligible"]]
    best = min(eligible_rows, key=lambda row: (row["n600_measured_bound_bytes"], row["leg"]))
    result = {
        "schema": "ddm_qbw2_temporal_bound.v1",
        "complete": True,
        "axis": "[macOS-CPU scorer-free advisory, full n600 plus seeded-stratified random n32]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "git_head": git_head(),
        "source_field": file_fact(SOURCE_FIELD),
        "storage_preflight_at_write": storage,
        "qbw1_commit_pin": "55aa812a3ce794f8a9eadd5d81d3cfd580fbb8d6",
        "carried_state": carried_state,
        "carried_state_contract": {
            "source": "actual decoded GB1 12-D pose-carrier state",
            "warp": "deterministic low-resolution carrier RGB translation with co-located fallback",
            "qa39_reuse": "receiver-side previous-field warp precedent",
            "qa39_ground_homography_not_claimed": True,
            "reason": "GB1 carries no geometric Pose6; the older 7.2KB Pose6 plane was not substituted",
            "marginal_pose_bytes": 0,
            "charter_leg_b_contract_pass": False,
            "gate_treatment": "proxy retained and reported, but excluded from the schema-v2 fire gate",
        },
        "shift_rows": shift_rows,
        "bound_rows": bound_rows,
        "best_bound": best,
        "quotient_allowance_bytes": QUOTIENT_ALLOWANCE_BYTES,
        "v2_fire_bar_bytes": QBW2_CLEARING_BOUND_BYTES,
        "step_2_fires": best["n600_measured_bound_bytes"] <= QBW2_CLEARING_BOUND_BYTES,
        "step_2_outcome": (
            "SCHEMA_V2_REQUIRED"
            if best["n600_measured_bound_bytes"] <= QBW2_CLEARING_BOUND_BYTES
            else "CURRENT_GB1_QUOTIENT_FAMILY_CLOSURE_AT_MEASURED_SCOPE"
        ),
        "n32_result": file_fact(QBW2_ROOT / "n32" / "RESULT.json"),
        "n600_result": file_fact(QBW2_ROOT / "n600" / "RESULT.json"),
        "distortion_not_measured": True,
        "training_launched": False,
        "scorer_run": False,
        "modal_run": False,
        "metal_run": False,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_once(QBW2_ROOT / "RESULT.json", result)
    atomic_json_once(checkpoint, result)
    qbw2_custody_manifest()
    return result


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
        choices=(
            "stage-00",
            "stage-01",
            "stage-02",
            "seal-fire-order",
            "run-00-02",
            "run-qbw2-bound",
            "audit",
            "audit-qbw2",
        ),
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
    elif args.action == "run-qbw2-bound":
        result = run_qbw2_bound()
    elif args.action == "audit-qbw2":
        result = qbw2_custody_manifest()
    else:
        result = custody_manifest()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
