#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_cr1 scorer-free GDL1 coder races on real cached argmax payloads.

This script executes the two GDL1 follow-on races without scorers, Metal, or
synthetic fixtures:

* P1: Road<->Lane support coded flat vs explicit stride-2 phase cosets.
* P2: top class-pair edge support coded pooled vs edge-conditioned.

Every byte row is a real coder output with decode equality back to the cached
n600 label support it claims to describe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

_REPO: Final = Path(__file__).resolve().parents[1]
for _path in (_REPO / "experiments", _REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_bd1_class_field_receiver as bd1

SEG_H: Final = 384
SEG_W: Final = 512
N_PAIRS: Final = 600
WATERLINE_B_PER_FLIP: Final = 1.2731082153320312
RL1_SETTLED_ROAD_LANE_FLIPS: Final = 235_148

CLASS_NAMES: Final = ("Road", "Lane", "Undriv", "Movable", "MyCar")
ROAD: Final = 0
LANE: Final = 1
TOP_EDGE_PAIRS: Final = ((0, 1), (0, 2), (0, 4), (2, 3), (0, 3))

DEFAULT_GT_ARGMAX: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy"
)
DEFAULT_CURRENT_ARGMAX: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy"
)
DEFAULT_RESEARCH_DIR: Final = _REPO / ".omx/research/ddm_cr1_20260808"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_cr1_20260808")


class CR1Error(ValueError):
    """CR1 payload, codec, or decode-equality validation failed closed."""


@dataclass(frozen=True, slots=True)
class CoderRow:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str | None = None


@dataclass(frozen=True, slots=True)
class RaceResult:
    race_id: str
    baseline_id: str
    treatment_id: str
    baseline_rows: tuple[CoderRow, ...]
    treatment_rows: tuple[CoderRow, ...]
    baseline_best: CoderRow
    treatment_best: CoderRow
    delta_bytes: int
    delta_pct: float
    verdict: str


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, CoderRow):
        return {
            "codec": value.codec,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "artifact_path": value.artifact_path,
        }
    if isinstance(value, RaceResult):
        return {
            "race_id": value.race_id,
            "baseline_id": value.baseline_id,
            "treatment_id": value.treatment_id,
            "baseline_rows": list(value.baseline_rows),
            "treatment_rows": list(value.treatment_rows),
            "baseline_best": value.baseline_best,
            "treatment_best": value.treatment_best,
            "delta_bytes": value.delta_bytes,
            "delta_pct": value.delta_pct,
            "verdict": value.verdict,
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_varint(value: int) -> bytes:
    if value < 0:
        raise CR1Error(f"negative varint {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(payload):
            raise CR1Error("varint truncated")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7
        if shift > 63:
            raise CR1Error("varint too large")


def encode_sorted_deltas(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype=np.uint32).reshape(-1)
    if arr.size and np.any(arr[1:] < arr[:-1]):
        raise CR1Error("delta input must be sorted ascending")
    out = bytearray(write_varint(int(arr.size)))
    prev = 0
    for idx, raw in enumerate(arr):
        value = int(raw)
        delta = value if idx == 0 else value - prev
        out += write_varint(delta)
        prev = value
    return bytes(out)


def decode_sorted_deltas(payload: bytes, offset: int) -> tuple[np.ndarray, int]:
    count, offset = read_varint(payload, offset)
    values = np.empty(count, dtype=np.uint32)
    prev = 0
    for idx in range(count):
        delta, offset = read_varint(payload, offset)
        value = delta if idx == 0 else prev + delta
        values[idx] = value
        prev = value
    return values, offset


def pack_record_stream(magic: bytes, records: tuple[bytes, ...]) -> bytes:
    if len(magic) != 8:
        raise CR1Error("stream magic must be 8 bytes")
    out = bytearray(magic + struct.pack("<I", len(records)))
    for record in records:
        out += struct.pack("<I", len(record))
        out += record
    return bytes(out)


def edge_name(edge: tuple[int, int]) -> str:
    return f"{CLASS_NAMES[edge[0]]}<->{CLASS_NAMES[edge[1]]}"


def edge_band(labels: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    a, b = edge
    lab = np.asarray(labels, dtype=np.uint8)
    out = np.zeros(lab.shape, dtype=bool)
    vertical = ((lab[:-1, :] == a) & (lab[1:, :] == b)) | ((lab[:-1, :] == b) & (lab[1:, :] == a))
    out[:-1, :] |= vertical
    out[1:, :] |= vertical
    horizontal = ((lab[:, :-1] == a) & (lab[:, 1:] == b)) | (
        (lab[:, :-1] == b) & (lab[:, 1:] == a)
    )
    out[:, :-1] |= horizontal
    out[:, 1:] |= horizontal
    return out


def edge_direct_flip(labels: np.ndarray, current: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    a, b = edge
    return ((labels == a) & (current == b)) | ((labels == b) & (current == a))


def compact_header(schema: str, extra: dict[str, Any]) -> bytes:
    return json.dumps({"schema": schema, **extra}, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_phase_flat_record(flats: np.ndarray) -> bytes:
    return b"U" + encode_sorted_deltas(np.asarray(flats, dtype=np.uint32))


def decode_phase_flat_record(record: bytes) -> np.ndarray:
    if not record.startswith(b"U"):
        raise CR1Error("bad phase-flat record magic")
    values, offset = decode_sorted_deltas(record, 1)
    if offset != len(record):
        raise CR1Error("phase-flat record has trailing bytes")
    return values


def encode_phase_coset_record(flats: np.ndarray) -> bytes:
    arr = np.asarray(flats, dtype=np.uint32).reshape(-1)
    ys = arr // SEG_W
    xs = arr % SEG_W
    out = bytearray(b"P")
    for y_mod in (0, 1):
        for x_mod in (0, 1):
            mask = ((ys & 1) == y_mod) & ((xs & 1) == x_mod)
            q = ((ys[mask] >> 1) * (SEG_W // 2) + (xs[mask] >> 1)).astype(np.uint32)
            q.sort()
            out += encode_sorted_deltas(q)
    return bytes(out)


def decode_phase_coset_record(record: bytes) -> np.ndarray:
    if not record.startswith(b"P"):
        raise CR1Error("bad phase-coset record magic")
    offset = 1
    flats: list[np.ndarray] = []
    for y_mod in (0, 1):
        for x_mod in (0, 1):
            q, offset = decode_sorted_deltas(record, offset)
            y = (q // (SEG_W // 2)) * 2 + y_mod
            x = (q % (SEG_W // 2)) * 2 + x_mod
            flats.append((y * SEG_W + x).astype(np.uint32))
    if offset != len(record):
        raise CR1Error("phase-coset record has trailing bytes")
    out = np.concatenate(flats) if flats else np.empty(0, dtype=np.uint32)
    out.sort()
    return out


def encode_edge_pooled_record(entries: list[tuple[int, int]]) -> bytes:
    entries.sort()
    out = bytearray(b"B" + write_varint(len(entries)))
    prev_flat = 0
    for idx, (flat, edge_idx) in enumerate(entries):
        delta = flat if idx == 0 else flat - prev_flat
        if delta < 0:
            raise CR1Error("edge pooled entries must be sorted")
        out += write_varint(delta)
        out += write_varint(edge_idx)
        prev_flat = flat
    return bytes(out)


def decode_edge_pooled_record(record: bytes) -> list[tuple[int, int]]:
    if not record.startswith(b"B"):
        raise CR1Error("bad pooled-edge record magic")
    count, offset = read_varint(record, 1)
    entries: list[tuple[int, int]] = []
    prev_flat = 0
    for idx in range(count):
        delta, offset = read_varint(record, offset)
        edge_idx, offset = read_varint(record, offset)
        flat = delta if idx == 0 else prev_flat + delta
        entries.append((flat, edge_idx))
        prev_flat = flat
    if offset != len(record):
        raise CR1Error("pooled-edge record has trailing bytes")
    return entries


def encode_edge_conditioned_record(flats: np.ndarray) -> bytes:
    return b"E" + encode_sorted_deltas(np.asarray(flats, dtype=np.uint32))


def decode_edge_conditioned_record(record: bytes) -> np.ndarray:
    if not record.startswith(b"E"):
        raise CR1Error("bad conditioned-edge record magic")
    values, offset = decode_sorted_deltas(record, 1)
    if offset != len(record):
        raise CR1Error("conditioned-edge record has trailing bytes")
    return values


def race_coders(
    *,
    surface_id: str,
    raw: bytes,
    records: tuple[bytes, ...],
    artifact_dir: Path,
) -> tuple[tuple[CoderRow, ...], CoderRow]:
    encoded = {
        "zlib-9": zlib.compress(raw, level=9),
        "brotli-q11": bytes(brotli.compress(raw, quality=11)),
        "lzma1-raw": bd1.lzma1_raw(raw),
        "smevr-r7-nibble": bd1.smevr_records(list(records)),
    }
    if zlib.decompress(encoded["zlib-9"]) != raw:
        raise CR1Error(f"{surface_id}: zlib roundtrip failed")
    if brotli.decompress(encoded["brotli-q11"]) != raw:
        raise CR1Error(f"{surface_id}: Brotli roundtrip failed")
    if bd1.unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise CR1Error(f"{surface_id}: LZMA1 roundtrip failed")
    if tuple(bd1.unsmevr_records(encoded["smevr-r7-nibble"])) != records:
        raise CR1Error(f"{surface_id}: SMEVR record roundtrip failed")

    best_codec = min(encoded, key=lambda name: len(encoded[name]))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows: list[CoderRow] = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        artifact_path = None
        if codec == best_codec:
            safe_surface = surface_id.replace("/", "_").replace(":", "_")
            path = artifact_dir / f"{safe_surface}.{codec}.bin"
            path.write_bytes(payload)
            artifact_path = str(path)
        rows.append(CoderRow(codec, len(payload), sha256_bytes(payload), artifact_path))
    best = next(row for row in rows if row.codec == best_codec)
    return tuple(rows), best


def compare_race(
    *,
    race_id: str,
    baseline_id: str,
    treatment_id: str,
    baseline_rows: tuple[CoderRow, ...],
    treatment_rows: tuple[CoderRow, ...],
) -> RaceResult:
    baseline_best = min(baseline_rows, key=lambda row: row.bytes)
    treatment_best = min(treatment_rows, key=lambda row: row.bytes)
    delta = treatment_best.bytes - baseline_best.bytes
    delta_pct = delta / baseline_best.bytes if baseline_best.bytes else 0.0
    verdict = "WIN-w/-bytes" if delta < 0 else "LOSS-w/-bytes"
    return RaceResult(
        race_id=race_id,
        baseline_id=baseline_id,
        treatment_id=treatment_id,
        baseline_rows=baseline_rows,
        treatment_rows=treatment_rows,
        baseline_best=baseline_best,
        treatment_best=treatment_best,
        delta_bytes=delta,
        delta_pct=delta_pct,
        verdict=verdict,
    )


def load_argmax(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if tuple(arr.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise CR1Error(f"{path} has shape {arr.shape}, expected {(N_PAIRS, SEG_H, SEG_W)}")
    return arr


def build_supports(
    gt_argmax: np.ndarray,
    current_argmax: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> tuple[dict[tuple[int, int], list[np.ndarray]], dict[str, Any]]:
    supports = {edge: [] for edge in edges}
    stats = {
        edge: {"support_pixels": 0, "direct_flip_mass_cx1": 0}
        for edge in edges
    }
    for pair in range(N_PAIRS):
        labels = np.asarray(gt_argmax[pair], dtype=np.uint8)
        current = np.asarray(current_argmax[pair], dtype=np.uint8)
        for edge in edges:
            flats = np.flatnonzero(edge_band(labels, edge)).astype(np.uint32)
            supports[edge].append(flats)
            stats[edge]["support_pixels"] += int(flats.size)
            stats[edge]["direct_flip_mass_cx1"] += int(edge_direct_flip(labels, current, edge).sum())
    total_support = sum(item["support_pixels"] for item in stats.values())
    total_flip_mass = sum(item["direct_flip_mass_cx1"] for item in stats.values())
    ranked = [
        {
            "edge": edge_name(edge),
            "class_ids": list(edge),
            "support_pixels": values["support_pixels"],
            "support_fraction_within_selected_edges": values["support_pixels"] / max(1, total_support),
            "direct_flip_mass_cx1": values["direct_flip_mass_cx1"],
            "flip_fraction_within_selected_edges": values["direct_flip_mass_cx1"] / max(1, total_flip_mass),
        }
        for edge, values in stats.items()
    ]
    ranked.sort(key=lambda row: (-int(row["support_pixels"]), row["edge"]))
    return supports, {
        "selection_mode": "n600_all_pairs_no_prefix",
        "edges": ranked,
        "selected_edge_count": len(edges),
        "selected_support_pixels": total_support,
        "selected_direct_flip_mass_cx1": total_flip_mass,
    }


def run_phase_race(
    *,
    support: list[np.ndarray],
    support_stats: dict[str, Any],
    artifact_dir: Path,
) -> tuple[RaceResult, dict[str, Any]]:
    header_common = {
        "edge": "Road<->Lane",
        "height": SEG_H,
        "width": SEG_W,
        "pairs": N_PAIRS,
        "selection": "n600_all_pairs_no_prefix",
    }
    flat_records = (
        compact_header("ddm_cr1_phase_flat_records.v1", header_common),
        *(encode_phase_flat_record(flats) for flats in support),
    )
    phase_records = (
        compact_header("ddm_cr1_phase_coset_records.v1", header_common),
        *(encode_phase_coset_record(flats) for flats in support),
    )
    for pair, flats in enumerate(support):
        flat_decoded = decode_phase_flat_record(flat_records[pair + 1])
        phase_decoded = decode_phase_coset_record(phase_records[pair + 1])
        if not np.array_equal(flat_decoded, flats):
            raise CR1Error(f"phase-flat decode mismatch at pair {pair}")
        if not np.array_equal(phase_decoded, flats):
            raise CR1Error(f"phase-coset decode mismatch at pair {pair}")

    flat_raw = pack_record_stream(b"CR1P1U1!", flat_records)
    phase_raw = pack_record_stream(b"CR1P1P1!", phase_records)
    flat_rows, _flat_best = race_coders(
        surface_id="p1_flat_road_lane_support",
        raw=flat_raw,
        records=flat_records,
        artifact_dir=artifact_dir,
    )
    phase_rows, _phase_best = race_coders(
        surface_id="p1_phase_coset_road_lane_support",
        raw=phase_raw,
        records=phase_records,
        artifact_dir=artifact_dir,
    )
    result = compare_race(
        race_id="gdl1_phase_coset_stride2",
        baseline_id="flat_delta_road_lane_support",
        treatment_id="phase_coset_qflat_road_lane_support",
        baseline_rows=flat_rows,
        treatment_rows=phase_rows,
    )
    support_pixels = int(support_stats["support_pixels"])
    direct_flips = int(support_stats["direct_flip_mass_cx1"])
    meta = {
        "raw_bytes": {
            "flat": len(flat_raw),
            "phase_coset": len(phase_raw),
        },
        "road_lane_support_pixels": support_pixels,
        "road_lane_direct_flip_mass_cx1": direct_flips,
        "settled_road_lane_flips_from_rl1": RL1_SETTLED_ROAD_LANE_FLIPS,
        "best_bytes_per_support_pixel": result.treatment_best.bytes / max(1, support_pixels),
        "best_bytes_per_cx1_direct_flip": result.treatment_best.bytes / max(1, direct_flips),
        "best_bytes_per_rl1_settled_flip": result.treatment_best.bytes / RL1_SETTLED_ROAD_LANE_FLIPS,
        "waterline_b_per_flip": WATERLINE_B_PER_FLIP,
        "beats_waterline_on_rl1_settled_denominator": (
            result.treatment_best.bytes / RL1_SETTLED_ROAD_LANE_FLIPS < WATERLINE_B_PER_FLIP
        ),
        "decode_equality": "flat and phase-coset records both decode exactly to the same Road<->Lane n600 support arrays",
    }
    return result, meta


def run_edge_race(
    *,
    supports: dict[tuple[int, int], list[np.ndarray]],
    extraction: dict[str, Any],
    artifact_dir: Path,
) -> tuple[RaceResult, dict[str, Any]]:
    edge_list = list(TOP_EDGE_PAIRS)
    header_common = {
        "edges": [edge_name(edge) for edge in edge_list],
        "class_ids": [list(edge) for edge in edge_list],
        "height": SEG_H,
        "width": SEG_W,
        "pairs": N_PAIRS,
        "selection": "n600_all_pairs_no_prefix",
    }
    pooled_records: list[bytes] = [
        compact_header("ddm_cr1_edge_pooled_records.v1", header_common),
    ]
    for pair in range(N_PAIRS):
        entries: list[tuple[int, int]] = []
        for edge_idx, edge in enumerate(edge_list):
            entries.extend((int(flat), edge_idx) for flat in supports[edge][pair])
        pooled_records.append(encode_edge_pooled_record(entries))

    conditioned_records: list[bytes] = [
        compact_header("ddm_cr1_edge_conditioned_records.v1", header_common),
    ]
    for edge in edge_list:
        for pair in range(N_PAIRS):
            conditioned_records.append(encode_edge_conditioned_record(supports[edge][pair]))

    source_by_edge = {edge: supports[edge] for edge in edge_list}
    decoded_by_edge = {edge: [np.empty(0, dtype=np.uint32) for _ in range(N_PAIRS)] for edge in edge_list}
    for pair, record in enumerate(pooled_records[1:]):
        entries = decode_edge_pooled_record(record)
        groups: dict[int, list[int]] = {idx: [] for idx in range(len(edge_list))}
        for flat, edge_idx in entries:
            groups[edge_idx].append(flat)
        for edge_idx, edge in enumerate(edge_list):
            arr = np.asarray(groups[edge_idx], dtype=np.uint32)
            arr.sort()
            decoded_by_edge[edge][pair] = arr
            if not np.array_equal(arr, source_by_edge[edge][pair]):
                raise CR1Error(f"pooled edge decode mismatch edge {edge_name(edge)} pair {pair}")

    offset = 1
    for edge in edge_list:
        for pair in range(N_PAIRS):
            arr = decode_edge_conditioned_record(conditioned_records[offset])
            offset += 1
            if not np.array_equal(arr, source_by_edge[edge][pair]):
                raise CR1Error(f"conditioned edge decode mismatch edge {edge_name(edge)} pair {pair}")
    if offset != len(conditioned_records):
        raise CR1Error("conditioned record accounting mismatch")

    pooled_tuple = tuple(pooled_records)
    conditioned_tuple = tuple(conditioned_records)
    pooled_raw = pack_record_stream(b"CR1P2B1!", pooled_tuple)
    conditioned_raw = pack_record_stream(b"CR1P2E1!", conditioned_tuple)
    pooled_rows, _pooled_best = race_coders(
        surface_id="p2_pooled_edge_blind_support",
        raw=pooled_raw,
        records=pooled_tuple,
        artifact_dir=artifact_dir,
    )
    conditioned_rows, _conditioned_best = race_coders(
        surface_id="p2_edge_conditioned_support",
        raw=conditioned_raw,
        records=conditioned_tuple,
        artifact_dir=artifact_dir,
    )
    result = compare_race(
        race_id="gdl1_edge_graph_conditional_carrier",
        baseline_id="pooled_edge_blind_support",
        treatment_id="edge_conditioned_support",
        baseline_rows=pooled_rows,
        treatment_rows=conditioned_rows,
    )
    support_pixels = int(extraction["selected_support_pixels"])
    direct_flips = int(extraction["selected_direct_flip_mass_cx1"])
    meta = {
        "raw_bytes": {
            "pooled_edge_blind": len(pooled_raw),
            "edge_conditioned": len(conditioned_raw),
        },
        "selected_support_pixels": support_pixels,
        "selected_direct_flip_mass_cx1": direct_flips,
        "best_bytes_per_support_pixel": result.treatment_best.bytes / max(1, support_pixels),
        "best_bytes_per_cx1_direct_flip": result.treatment_best.bytes / max(1, direct_flips),
        "measured_n600_bytes": result.treatment_best.bytes,
        "decode_equality": "pooled and edge-conditioned records both decode exactly to the same edge-labeled n600 support arrays",
        "selected_edges": extraction["edges"],
    }
    return result, meta


def typed_row(
    *,
    race: RaceResult,
    meta: dict[str, Any],
    input_paths: dict[str, Any],
    verdict_scope: str,
    falsifier_status: str,
) -> dict[str, Any]:
    return {
        "schema": "ddm_cr1_coder_race_row.v1",
        "created_utc": now_utc(),
        "race_id": race.race_id,
        "axis": "[byte-only scorer-free]",
        "selection_mode": "n600_all_pairs_no_prefix",
        "input_paths": input_paths,
        "baseline_id": race.baseline_id,
        "treatment_id": race.treatment_id,
        "coders": ["zlib-9", "brotli-q11", "lzma1-raw", "smevr-r7-nibble"],
        "baseline_rows": race.baseline_rows,
        "treatment_rows": race.treatment_rows,
        "baseline_best": race.baseline_best,
        "treatment_best": race.treatment_best,
        "delta_bytes": race.delta_bytes,
        "delta_pct": race.delta_pct,
        "verdict": race.verdict,
        "verdict_scope": verdict_scope,
        "falsifier_status": falsifier_status,
        "meta": meta,
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    p1 = receipt["races"]["gdl1_phase_coset_stride2"]
    p2 = receipt["races"]["gdl1_edge_graph_conditional_carrier"]
    lines = [
        "# CR1 GDL1 coder races - 2026-08-08",
        "",
        "Tags: [no-triality] [p0-ledger-ok]",
        "",
        "## Answer First",
        "",
        "No scorer, no evaluator, no Metal, no GPU, no paid job, and no archive promotion ran.",
        "Both races use cached real n600 argmax payloads and real coder round-trips.",
        "",
        "| race | baseline best | treatment best | delta | verdict |",
        "|---|---:|---:|---:|---|",
        (
            f"| phase-coset stride-2 | {p1['baseline_best']['bytes']} B "
            f"({p1['baseline_best']['codec']}) | {p1['treatment_best']['bytes']} B "
            f"({p1['treatment_best']['codec']}) | {p1['delta_bytes']} B "
            f"({p1['delta_pct']:.3%}) | {p1['verdict']} |"
        ),
        (
            f"| edge-graph conditional carrier | {p2['baseline_best']['bytes']} B "
            f"({p2['baseline_best']['codec']}) | {p2['treatment_best']['bytes']} B "
            f"({p2['treatment_best']['codec']}) | {p2['delta_bytes']} B "
            f"({p2['delta_pct']:.3%}) | {p2['verdict']} |"
        ),
        "",
        "## Inputs",
        "",
        f"- GT argmax: `{receipt['inputs']['gt_argmax_path']}` "
        f"({receipt['inputs']['gt_argmax_sha256']})",
        f"- Current argmax: `{receipt['inputs']['current_argmax_path']}` "
        f"({receipt['inputs']['current_argmax_sha256']})",
        f"- Axis: `{receipt['axis']}`.",
        f"- Selection: `{receipt['selection_mode']}`.",
        "",
        "## Recall Evidence",
        "",
        "| source or query | result | impact |",
        "|---|---|---|",
    ]
    for item in receipt["recall_evidence"]:
        lines.append(f"| {item['source']} | {item['result']} | {item['impact']} |")
    lines.extend(
        [
            "",
            "## Race 1 - Phase-Coset Stride-2",
            "",
            (
                f"Road<->Lane support pixels: `{p1['meta']['road_lane_support_pixels']}`; "
                f"cx1 direct Road<->Lane flips: `{p1['meta']['road_lane_direct_flip_mass_cx1']}`; "
                f"RL1 settled flips: `{p1['meta']['settled_road_lane_flips_from_rl1']}`."
            ),
            (
                f"Treatment best bytes per RL1 settled flip: "
                f"`{p1['meta']['best_bytes_per_rl1_settled_flip']:.6f}` vs W "
                f"`{p1['meta']['waterline_b_per_flip']:.6f}`."
            ),
            f"Decode equality: {p1['meta']['decode_equality']}.",
            "",
            "Coder rows:",
            "",
            "| side | codec | bytes | sha256 | artifact |",
            "|---|---|---:|---|---|",
        ]
    )
    for side, rows in (("baseline", p1["baseline_rows"]), ("phase", p1["treatment_rows"])):
        for row in rows:
            lines.append(
                f"| {side} | {row['codec']} | {row['bytes']} | `{row['sha256']}` | "
                f"{row['artifact_path'] or ''} |"
            )
    lines.extend(
        [
            "",
            "## Race 2 - Edge-Graph Conditional Carrier",
            "",
            (
                f"Selected support pixels: `{p2['meta']['selected_support_pixels']}`; "
                f"selected cx1 direct flips: `{p2['meta']['selected_direct_flip_mass_cx1']}`."
            ),
            f"Decode equality: {p2['meta']['decode_equality']}.",
            "",
            "Selected edge denominators:",
            "",
            "| edge | support px | support share | cx1 direct flips | flip share |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for edge in p2["meta"]["selected_edges"]:
        lines.append(
            f"| {edge['edge']} | {edge['support_pixels']} | "
            f"{edge['support_fraction_within_selected_edges']:.6f} | "
            f"{edge['direct_flip_mass_cx1']} | {edge['flip_fraction_within_selected_edges']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Coder rows:",
            "",
            "| side | codec | bytes | sha256 | artifact |",
            "|---|---|---:|---|---|",
        ]
    )
    for side, rows in (("pooled", p2["baseline_rows"]), ("edge-conditioned", p2["treatment_rows"])):
        for row in rows:
            lines.append(
                f"| {side} | {row['codec']} | {row['bytes']} | `{row['sha256']}` | "
                f"{row['artifact_path'] or ''} |"
            )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- These are byte-only scorer-free measurements over cached argmax labels.",
            "- Support denominators are 4-neighbor endpoint pixels, not separatrix crack lengths.",
            "- No RGB receiver, archive parse-back, or n600 scorer survival is claimed.",
            "- Negative verdicts are formulation-scoped only, exactly as pre-registered.",
            "- Follow-ons named here exit this run as FIRED by these CR1 measurements; any scorer consumer remains queued behind a future owner.",
            "",
            "## Follow-On Disposition",
            "",
            "| item | disposition | fire order |",
            "|---|---|---|",
        ]
    )
    for item in receipt["follow_on_disposition"]:
        lines.append(f"| {item['id']} | {item['disposition']} | {item['fire_order']} |")
    lines.extend(
        [
            "",
            "## Frontier Honesty",
            "",
            (
                "Own-vehicle frontier remains "
                "`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; "
                "contest pointer remains borrowed/unmoved at `0.1910828242 [contest-CPU]`."
            ),
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    gt = load_argmax(args.gt_argmax)
    current = load_argmax(args.current_argmax)
    supports, extraction = build_supports(gt, current, TOP_EDGE_PAIRS)
    artifact_dir = args.ssd_dir / "payloads"

    p1_result, p1_meta = run_phase_race(
        support=supports[(ROAD, LANE)],
        support_stats={
            "support_pixels": next(
                row["support_pixels"] for row in extraction["edges"] if row["edge"] == "Road<->Lane"
            ),
            "direct_flip_mass_cx1": next(
                row["direct_flip_mass_cx1"] for row in extraction["edges"] if row["edge"] == "Road<->Lane"
            ),
        },
        artifact_dir=artifact_dir,
    )
    p2_result, p2_meta = run_edge_race(
        supports=supports,
        extraction=extraction,
        artifact_dir=artifact_dir,
    )

    input_paths = {
        "gt_argmax_path": args.gt_argmax,
        "current_argmax_path": args.current_argmax,
        "gdl1_crosswalk": ".omx/research/ddm_gdl1_20260807/GDL1_CROSSWALK.md",
        "gdl1_rows": ".omx/research/ddm_gdl1_20260807/GDL1_CROSSWALK_ROWS.jsonl",
    }
    p1_row = typed_row(
        race=p1_result,
        meta=p1_meta,
        input_paths=input_paths,
        verdict_scope="FORMULATION: phase-coset coordinate coding of Road<->Lane GT support",
        falsifier_status=(
            "phase representation did not beat the flat baseline"
            if p1_result.delta_bytes >= 0
            else "phase representation beat the flat baseline; receiver/scorer survival still unclaimed"
        ),
    )
    p2_row = typed_row(
        race=p2_result,
        meta=p2_meta,
        input_paths=input_paths,
        verdict_scope="FORMULATION: edge-conditioned coding of top edge-labeled GT support",
        falsifier_status=(
            "edge-conditioned representation did not beat pooled edge-blind baseline"
            if p2_result.delta_bytes >= 0
            else "edge-conditioned representation beat pooled baseline; receiver/scorer survival still unclaimed"
        ),
    )

    return {
        "schema": "ddm_cr1_gdl1_coder_races_receipt.v1",
        "created_utc": now_utc(),
        "axis": "[byte-only scorer-free]",
        "selection_mode": "n600_all_pairs_no_prefix",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "wall_seconds": time.perf_counter() - start,
        "inputs": {
            "gt_argmax_path": str(args.gt_argmax),
            "gt_argmax_sha256": sha256_file(args.gt_argmax),
            "current_argmax_path": str(args.current_argmax),
            "current_argmax_sha256": sha256_file(args.current_argmax),
            "class_names": CLASS_NAMES,
            "selected_edges": [edge_name(edge) for edge in TOP_EDGE_PAIRS],
        },
        "recall_evidence": [
            {
                "source": ".omx/research/ddm_gdl1_20260807/GDL1_CROSSWALK.md rows 1-2",
                "result": "Found the two queued ADOPT-CLASS probes and their falsifiers.",
                "impact": "Executed the registered phase-coset and edge-conditioned byte races instead of redesigning them.",
            },
            {
                "source": ".omx/research/ddm_rl1_roadlane_interface_price_20260803.md",
                "result": "RL1 supplies the non-prefix n32 Road<->Lane price line and W=1.273108 B/flip denominator.",
                "impact": "Phase race reports bytes per settled Road<->Lane flip and waterline status.",
            },
            {
                "source": ".omx/research/ddm_sx1_separatrix_carrier_20260803.md",
                "result": "SX1 supplies full-population separatrix edge denominators and Road<->Lane hub evidence.",
                "impact": "Edge race uses n600 edge-labeled support instead of a prefix or class-row proxy.",
            },
            {
                "source": "experiments/ddm_bd1_class_field_receiver.py and experiments/ddm_pe1_per_edge_partition_race.py",
                "result": "Existing BD1/PE1 coder primitives already provide Brotli q11, raw LZMA1, and SMEVR with decode checks.",
                "impact": "Reused landed coder surfaces; did not introduce a synthetic or unvalidated codec.",
            },
            {
                "source": "content search: phase_coset, edge_graph, Road<->Lane, SMEVR, #920, #984 over .omx/research and experiments",
                "result": "Found PE1/ST1/SM2/BD1 precedents beyond the charter seeds; no finished CR1 race artifact existed.",
                "impact": "Built a narrow new CR1 measurement and kept scorer consumers queued.",
            },
            {
                "source": "tools/list_canonical_equations.py --json",
                "result": "Canonical registry was consulted for relevant byte/coder equations; no CR1-specific equation superseded the charter.",
                "impact": "Receipt remains a scorer-free measured byte race, not a score or equation promotion.",
            },
        ],
        "support_extraction": extraction,
        "races": {
            p1_result.race_id: jsonable(p1_row),
            p2_result.race_id: jsonable(p2_row),
        },
        "typed_rows": [jsonable(p1_row), jsonable(p2_row)],
        "follow_on_disposition": [
            {
                "id": "GDL1-P1",
                "disposition": "FIRED",
                "fire_order": "This receipt is the scorer-free phase-coset byte race; any receiver/scorer consumer must claim a future lane separately.",
            },
            {
                "id": "GDL1-P2",
                "disposition": "FIRED",
                "fire_order": "This receipt is the scorer-free edge-conditioned byte race; #984/ty1 may consume only the measured rows and must not infer RGB survival.",
            },
        ],
        "boundaries": [
            "No scorer, no evaluator, no archive promotion, no Metal/GPU, no paid job.",
            "Payloads are cached real n600 argmax supports from GT and cx1 argmax arrays.",
            "Support denominators are 4-neighbor endpoint pixels, not separatrix crack lengths.",
            "All coder rows are real zlib/Brotli/LZMA/SMEVR outputs with round-trip checks.",
            "Verdicts are formulation-scoped byte-race outcomes only.",
        ],
        "frontier": {
            "own_vehicle": "S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]",
            "contest_pointer": "borrowed/unmoved 0.1910828242 [contest-CPU]",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-argmax", type=Path, default=DEFAULT_GT_ARGMAX)
    parser.add_argument("--current-argmax", type=Path, default=DEFAULT_CURRENT_ARGMAX)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_receipt(args)
    args.research_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.research_dir / "CR1_RECEIPT.json"
    rows_path = args.research_dir / "CR1_ROWS.jsonl"
    findings_path = args.research_dir / "CR1_FINDINGS.md"
    atomic_write_json(receipt_path, receipt)
    write_jsonl(rows_path, receipt["typed_rows"])
    write_markdown(findings_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "rows": str(rows_path),
                "findings": str(findings_path),
                "p1_verdict": receipt["races"]["gdl1_phase_coset_stride2"]["verdict"],
                "p1_delta_bytes": receipt["races"]["gdl1_phase_coset_stride2"]["delta_bytes"],
                "p2_verdict": receipt["races"]["gdl1_edge_graph_conditional_carrier"]["verdict"],
                "p2_delta_bytes": receipt["races"]["gdl1_edge_graph_conditional_carrier"]["delta_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
