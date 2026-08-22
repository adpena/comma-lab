#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Price an exact temporal partition-worldsheet grammar on frozen n600 labels.

This is a scorer-free, CPU-only measurement.  It turns every label row into an
ordered boundary curve sentence, predicts each boundary coordinate from the
previous row and/or the same row in the previous scored pair, and separates the
grammar/event stream from the coordinate-innovation stream.  The receiver
parses the selected real-coder payloads and reconstructs every label map.

Every Brotli-Q11, raw-LZMA1, and SMEVR candidate stream is retained.  The
lossless leg must reproduce the input partition exactly.  The tolerance leg
quantizes a frame-stratified set of boundary coordinates while keeping the
measured label-disagreement mass at or below the requested n600 allowance.
Nothing in this file runs SegNet, PoseNet, upstream/evaluate.py, or Modal, and
no result from it is a contest score.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
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

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap

SCHEMA: Final = "ddm_ws0_worldsheet_grammar_price.v1"
AXIS: Final = "[macOS-CPU advisory, scorer-free n600 coder]"
MAGIC: Final = b"WSG1"
VERSION: Final = 1
N_CLASSES: Final = 5
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
TOTAL_CELLS: Final = N_PAIRS * HEIGHT * WIDTH
TOLERANCE_DSEG_DEFAULT: Final = 0.00116
EXPECTED_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_ES1_SHA256: Final = "789b00f237bac0a8d1bdb3f00ae0a3b83be7ab75edfea472baaf64dbf0f05e18"
MIN_FREE_BYTES: Final = 2 * 1024**3
MAX_WORKERS: Final = 8
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
EDGE_PAIRS: Final = tuple((left, right) for left in range(N_CLASSES) for right in range(left + 1, N_CLASSES))
EDGE_TO_ID: Final = {edge: index for index, edge in enumerate(EDGE_PAIRS)}
ID_TO_EDGE: Final = {index: edge for edge, index in EDGE_TO_ID.items()}
SELECTION_MODES: Final = ("minabs", "spatial", "temporal")
CODEC_IDS: Final = {
    "brotli-q11": bd1.BD1_BROTLI_Q11,
    "lzma1-raw": bd1.BD1_LZMA1_RAW,
    "smevr-r7-nibble": bd1.BD1_SMEVR_R7_NIBBLE,
}
ID_CODECS: Final = {value: key for key, value in CODEC_IDS.items()}
TOPOLOGY_STREAM_ID: Final = 0
HEADER_STRUCT: Final = struct.Struct("<4sBHHHB")
STREAM_HEADER: Final = struct.Struct("<BBII")
DEFAULT_CACHE: Final = _REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_ws0_worldsheet_grammar_price/retained")
DEFAULT_ES1: Final = _REPO / ".omx/research/ddm_es1_end_state_characterization_20260821.md"


class WorldsheetError(RuntimeError):
    """The grammar, receiver, provenance, or custody contract failed closed."""


@dataclass(frozen=True, slots=True)
class FrameGrammar:
    """One frame's exact horizontal-boundary sentence."""

    initials: np.ndarray
    counts: np.ndarray
    rights: np.ndarray
    xs: np.ndarray

    def __post_init__(self) -> None:
        initials = np.ascontiguousarray(self.initials, dtype=np.uint8)
        counts = np.ascontiguousarray(self.counts, dtype=np.uint16)
        rights = np.ascontiguousarray(self.rights, dtype=np.uint8)
        xs = np.ascontiguousarray(self.xs, dtype=np.uint16)
        if initials.ndim != 1 or counts.shape != initials.shape:
            raise WorldsheetError("frame initials/counts shape mismatch")
        if rights.ndim != 1 or xs.shape != rights.shape:
            raise WorldsheetError("frame rights/xs shape mismatch")
        if int(counts.sum()) != int(rights.size):
            raise WorldsheetError("frame boundary counts do not cover rights")
        if initials.size == 0 or np.any(initials >= N_CLASSES) or np.any(rights >= N_CLASSES):
            raise WorldsheetError("frame labels escape the registered class alphabet")
        object.__setattr__(self, "initials", initials)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "rights", rights)
        object.__setattr__(self, "xs", xs)
        offset = 0
        for initial, count in zip(initials.tolist(), counts.tolist(), strict=True):
            stop = offset + int(count)
            row_x = xs[offset:stop]
            row_r = rights[offset:stop]
            if row_x.size and (
                int(row_x[0]) < 1
                or int(row_x[-1]) >= WIDTH
                or np.any(row_x[1:] <= row_x[:-1])
            ):
                raise WorldsheetError("row boundary coordinates are not strictly ordered")
            left = int(initial)
            for right in row_r.tolist():
                if int(right) == left:
                    raise WorldsheetError("boundary transition keeps the same label")
                left = int(right)
            offset = stop

    @property
    def height(self) -> int:
        return int(self.initials.size)

    def row(self, y: int) -> tuple[int, np.ndarray, np.ndarray]:
        start = int(np.sum(self.counts[:y], dtype=np.int64))
        stop = start + int(self.counts[y])
        return int(self.initials[y]), self.rights[start:stop], self.xs[start:stop]


@dataclass(frozen=True, slots=True)
class StreamRace:
    name: str
    records_sha256: str
    canonical_raw_bytes: int
    payloads: dict[str, bytes]
    winner: str


@dataclass(frozen=True, slots=True)
class CandidateSemantics:
    stream_records: dict[str, tuple[bytes, ...]]
    source_counts: dict[str, dict[str, int]]
    temporal_events: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class QuantizationResult:
    frames: tuple[FrameGrammar, ...]
    q_step: int
    requested_error_cap: int
    selected_shift_upper_bound: int
    selected_boundaries: int
    selected_by_edge: dict[str, dict[str, int]]
    unused_frame_quota: int


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


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


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


def put_uleb(output: bytearray, value: int) -> None:
    if value < 0:
        raise WorldsheetError("ULEB cannot encode a negative value")
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return


def get_uleb(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise WorldsheetError("truncated or overlong ULEB")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def zigzag(value: int) -> int:
    return (int(value) << 1) ^ (int(value) >> 63)


def unzigzag(value: int) -> int:
    return (int(value) >> 1) ^ -(int(value) & 1)


def pack_two_bit(values: Sequence[int]) -> bytes:
    output = bytearray((len(values) + 3) // 4)
    for index, value in enumerate(values):
        if not 0 <= int(value) <= 3:
            raise WorldsheetError("two-bit value outside [0,3]")
        output[index // 4] |= int(value) << (2 * (index % 4))
    return bytes(output)


def unpack_two_bit(payload: bytes, count: int) -> list[int]:
    expected = (count + 3) // 4
    if len(payload) != expected:
        raise WorldsheetError("two-bit payload length mismatch")
    values = [(payload[index // 4] >> (2 * (index % 4))) & 3 for index in range(count)]
    if count % 4 and payload[-1] >> (2 * (count % 4)):
        raise WorldsheetError("two-bit payload has nonzero padding")
    return values


def pack_records(records: Sequence[bytes]) -> bytes:
    if len(records) > 0xFFFF:
        raise WorldsheetError("too many frame records")
    output = bytearray(struct.pack("<H", len(records)))
    for record in records:
        output.extend(struct.pack("<I", len(record)))
    for record in records:
        output.extend(record)
    return bytes(output)


def unpack_records(payload: bytes) -> tuple[bytes, ...]:
    if len(payload) < 2:
        raise WorldsheetError("record pack is truncated")
    (count,) = struct.unpack_from("<H", payload)
    header = 2 + 4 * count
    if len(payload) < header:
        raise WorldsheetError("record length table is truncated")
    lengths = struct.unpack_from(f"<{count}I", payload, 2) if count else ()
    offset = header
    records: list[bytes] = []
    for length in lengths:
        record = payload[offset : offset + length]
        if len(record) != length:
            raise WorldsheetError("record body is truncated")
        records.append(record)
        offset += length
    if offset != len(payload):
        raise WorldsheetError("record pack has trailing bytes")
    return tuple(records)


def frame_from_labels(labels: np.ndarray) -> FrameGrammar:
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 2 or labels.shape[1] != WIDTH:
        raise WorldsheetError(f"unexpected frame shape {labels.shape}")
    if np.any(labels >= N_CLASSES):
        raise WorldsheetError("label frame escapes the class alphabet")
    initials = labels[:, 0].copy()
    counts = np.empty(labels.shape[0], dtype=np.uint16)
    rights: list[int] = []
    xs: list[int] = []
    for y, row in enumerate(labels):
        positions = np.flatnonzero(row[1:] != row[:-1]) + 1
        counts[y] = positions.size
        xs.extend(int(value) for value in positions)
        rights.extend(int(row[value]) for value in positions)
    return FrameGrammar(
        initials=initials,
        counts=counts,
        rights=np.asarray(rights, dtype=np.uint8),
        xs=np.asarray(xs, dtype=np.uint16),
    )


def render_frame(frame: FrameGrammar) -> np.ndarray:
    output = np.empty((frame.height, WIDTH), dtype=np.uint8)
    offset = 0
    for y, (initial, count) in enumerate(zip(frame.initials.tolist(), frame.counts.tolist(), strict=True)):
        stop = offset + int(count)
        positions = frame.xs[offset:stop]
        rights = frame.rights[offset:stop]
        left_x = 0
        label = int(initial)
        for x, right in zip(positions.tolist(), rights.tolist(), strict=True):
            output[y, left_x:int(x)] = label
            left_x = int(x)
            label = int(right)
        output[y, left_x:] = label
        offset = stop
    return output


def frame_to_npz_bytes(frame: FrameGrammar) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        initials=frame.initials,
        counts=frame.counts,
        rights=frame.rights,
        xs=frame.xs,
    )
    return output.getvalue()


def frame_from_npz(path: Path) -> FrameGrammar:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"initials", "counts", "rights", "xs"}:
            raise WorldsheetError(f"unexpected extracted-frame members in {path}")
        return FrameGrammar(
            initials=archive["initials"],
            counts=archive["counts"],
            rights=archive["rights"],
            xs=archive["xs"],
        )


_WORKER_LABELS: np.memmap | None = None


def _extract_worker_init(cache: str) -> None:
    global _WORKER_LABELS
    _WORKER_LABELS = open_stored_npy_memmap(cache, "lstars")


def _extract_worker(index: int) -> tuple[int, bytes]:
    if _WORKER_LABELS is None:
        raise WorldsheetError("extraction worker was not initialized")
    frame = frame_from_labels(np.asarray(_WORKER_LABELS[index], dtype=np.uint8))
    return index, frame_to_npz_bytes(frame)


def _frame_path(extract_dir: Path, index: int) -> Path:
    return extract_dir / f"frame_{index:04d}.npz"


def extract_or_resume_frames(
    *,
    cache: Path,
    cache_sha256: str,
    extract_dir: Path,
    pool: mp.pool.Pool,
) -> tuple[tuple[FrameGrammar, ...], dict[str, Any]]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    prior_manifest_path = extract_dir / "STAGE_COMPLETE.json"
    if prior_manifest_path.exists():
        try:
            prior_manifest = json.loads(prior_manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise WorldsheetError("extraction stage manifest is corrupt") from exc
        if prior_manifest.get("cache_sha256") != cache_sha256:
            raise WorldsheetError("extraction stage belongs to a different input; use a new output root")
    missing: list[int] = []
    for index in range(N_PAIRS):
        path = _frame_path(extract_dir, index)
        try:
            frame = frame_from_npz(path)
            if frame.height != HEIGHT:
                raise WorldsheetError("extracted frame height mismatch")
        except (FileNotFoundError, OSError, ValueError, WorldsheetError):
            missing.append(index)
    for index, payload in pool.imap_unordered(_extract_worker, missing, chunksize=2):
        atomic_bytes(_frame_path(extract_dir, index), payload)
    frames = tuple(frame_from_npz(_frame_path(extract_dir, index)) for index in range(N_PAIRS))
    manifest = {
        "schema": "ddm_ws0_extracted_frames.v1",
        "created_utc": utc_now(),
        "cache": str(cache),
        "cache_sha256": cache_sha256,
        "frames": N_PAIRS,
        "height": HEIGHT,
        "width": WIDTH,
        "resumed_frames": N_PAIRS - len(missing),
        "new_frames": len(missing),
        "artifacts": {
            _frame_path(extract_dir, index).name: {
                "bytes": _frame_path(extract_dir, index).stat().st_size,
                "sha256": sha256_file(_frame_path(extract_dir, index)),
            }
            for index in range(N_PAIRS)
        },
    }
    atomic_json(extract_dir / "STAGE_COMPLETE.json", manifest)
    return frames, manifest


def induced_rank_table(frames: Sequence[FrameGrammar]) -> tuple[tuple[int, ...], ...]:
    counts = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for frame in frames:
        offset = 0
        for initial, count in zip(frame.initials.tolist(), frame.counts.tolist(), strict=True):
            left = int(initial)
            stop = offset + int(count)
            for right in frame.rights[offset:stop].tolist():
                counts[left, int(right)] += 1
                left = int(right)
            offset = stop
    table: list[tuple[int, ...]] = []
    for left in range(N_CLASSES):
        rights = [right for right in range(N_CLASSES) if right != left]
        rights.sort(key=lambda right: (-int(counts[left, right]), right))
        table.append(tuple(rights))
    return tuple(table)


def rank_table_bytes(table: Sequence[Sequence[int]]) -> bytes:
    if len(table) != N_CLASSES or any(len(row) != N_CLASSES - 1 for row in table):
        raise WorldsheetError("rank table shape mismatch")
    payload = bytes(int(value) for row in table for value in row)
    for left, row in enumerate(table):
        if sorted(int(value) for value in row) != [value for value in range(N_CLASSES) if value != left]:
            raise WorldsheetError("rank table is not a per-left permutation")
    return payload


def rank_table_from_bytes(payload: bytes) -> tuple[tuple[int, ...], ...]:
    if len(payload) != N_CLASSES * (N_CLASSES - 1):
        raise WorldsheetError("rank table byte length mismatch")
    rows = tuple(
        tuple(payload[left * (N_CLASSES - 1) : (left + 1) * (N_CLASSES - 1)])
        for left in range(N_CLASSES)
    )
    rank_table_bytes(rows)
    return rows


def topology_record(frame: FrameGrammar, table: Sequence[Sequence[int]]) -> bytes:
    output = bytearray()
    offset = 0
    for initial, count in zip(frame.initials.tolist(), frame.counts.tolist(), strict=True):
        output.append(int(initial))
        put_uleb(output, int(count))
        ranks: list[int] = []
        left = int(initial)
        stop = offset + int(count)
        for right in frame.rights[offset:stop].tolist():
            try:
                rank = tuple(table[left]).index(int(right))
            except ValueError as exc:
                raise WorldsheetError("transition is absent from its induced rank table") from exc
            ranks.append(rank)
            left = int(right)
        output.extend(pack_two_bit(ranks))
        offset = stop
    return bytes(output)


def decode_topology_record(
    payload: bytes,
    table: Sequence[Sequence[int]],
    *,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initials = np.empty(height, dtype=np.uint8)
    counts = np.empty(height, dtype=np.uint16)
    rights: list[int] = []
    offset = 0
    for y in range(height):
        if offset >= len(payload):
            raise WorldsheetError("topology record is truncated")
        initial = int(payload[offset])
        offset += 1
        if initial >= N_CLASSES:
            raise WorldsheetError("topology initial label is invalid")
        count, offset = get_uleb(payload, offset)
        packed_size = (count + 3) // 4
        ranks = unpack_two_bit(payload[offset : offset + packed_size], count)
        offset += packed_size
        initials[y] = initial
        counts[y] = count
        left = initial
        for rank in ranks:
            if rank >= len(table[left]):
                raise WorldsheetError("topology transition rank is invalid")
            right = int(table[left][rank])
            rights.append(right)
            left = right
    if offset != len(payload):
        raise WorldsheetError("topology record has trailing bytes")
    return initials, counts, np.asarray(rights, dtype=np.uint8)


def row_ordered_positions(frame: FrameGrammar, y: int) -> dict[tuple[int, int], list[int]]:
    initial, rights, xs = frame.row(y)
    output: dict[tuple[int, int], list[int]] = defaultdict(list)
    left = initial
    for right, x in zip(rights.tolist(), xs.tolist(), strict=True):
        output[(left, int(right))].append(int(x))
        left = int(right)
    return output


def choose_source(
    *,
    mode: str,
    x: int,
    spatial: int | None,
    temporal: int | None,
) -> tuple[int, int]:
    if mode not in SELECTION_MODES:
        raise WorldsheetError(f"unknown selection mode {mode!r}")
    if spatial is None and temporal is None:
        return 0, x
    if mode == "spatial":
        return (1, x - spatial) if spatial is not None else (2, x - int(temporal))
    if mode == "temporal":
        return (2, x - temporal) if temporal is not None else (1, x - int(spatial))
    candidates: list[tuple[int, int, int]] = []
    if spatial is not None:
        candidates.append((abs(x - spatial), 1, x - spatial))
    if temporal is not None:
        candidates.append((abs(x - temporal), 2, x - temporal))
    _, source, residual = min(candidates)
    return source, residual


def temporal_event_counts(frames: Sequence[FrameGrammar]) -> dict[str, dict[str, int]]:
    output = {
        edge_name(edge): {"births": 0, "persists": 0, "deaths": 0}
        for edge in EDGE_PAIRS
    }
    for pair in range(1, len(frames)):
        previous = frames[pair - 1]
        current = frames[pair]
        for y in range(current.height):
            prev_counts = Counter(
                tuple(sorted(ordered)) for ordered, values in row_ordered_positions(previous, y).items() for _ in values
            )
            cur_counts = Counter(
                tuple(sorted(ordered)) for ordered, values in row_ordered_positions(current, y).items() for _ in values
            )
            for edge in EDGE_PAIRS:
                prev_n = int(prev_counts[edge])
                cur_n = int(cur_counts[edge])
                row = output[edge_name(edge)]
                row["persists"] += min(prev_n, cur_n)
                row["births"] += max(0, cur_n - prev_n)
                row["deaths"] += max(0, prev_n - cur_n)
    return output


def edge_name(edge: tuple[int, int]) -> str:
    return f"{CLASS_NAMES[edge[0]]}<->{CLASS_NAMES[edge[1]]}"


def source_stream_name(edge_id: int) -> str:
    return f"edge_{edge_id:02d}_{edge_name(ID_TO_EDGE[edge_id]).replace('<->', '_')}_events"


def coord_stream_name(edge_id: int) -> str:
    return f"edge_{edge_id:02d}_{edge_name(ID_TO_EDGE[edge_id]).replace('<->', '_')}_coords"


def stream_id(name: str) -> int:
    if name == "topology":
        return TOPOLOGY_STREAM_ID
    for edge_id in range(len(EDGE_PAIRS)):
        if name == source_stream_name(edge_id):
            return 1 + edge_id * 2
        if name == coord_stream_name(edge_id):
            return 2 + edge_id * 2
    raise WorldsheetError(f"unknown stream name {name!r}")


def stream_name(identifier: int) -> str:
    if identifier == TOPOLOGY_STREAM_ID:
        return "topology"
    edge_id, coordinate = divmod(identifier - 1, 2)
    if edge_id not in ID_TO_EDGE:
        raise WorldsheetError(f"unknown stream id {identifier}")
    return coord_stream_name(edge_id) if coordinate else source_stream_name(edge_id)


def build_candidate_semantics(
    frames: Sequence[FrameGrammar],
    table: Sequence[Sequence[int]],
    *,
    mode: str,
) -> CandidateSemantics:
    records: dict[str, list[bytes]] = {"topology": []}
    source_counts = {
        edge_name(edge): {"absolute_birth": 0, "spatial_curve": 0, "temporal_worldsheet": 0}
        for edge in EDGE_PAIRS
    }
    for edge_id in range(len(EDGE_PAIRS)):
        records[source_stream_name(edge_id)] = []
        records[coord_stream_name(edge_id)] = []
    for pair, frame in enumerate(frames):
        records["topology"].append(topology_record(frame, table))
        frame_sources: dict[int, list[int]] = {edge_id: [] for edge_id in range(len(EDGE_PAIRS))}
        frame_coords: dict[int, bytearray] = {edge_id: bytearray() for edge_id in range(len(EDGE_PAIRS))}
        for y in range(frame.height):
            initial, rights, xs = frame.row(y)
            spatial_positions = row_ordered_positions(frame, y - 1) if y else {}
            temporal_positions = row_ordered_positions(frames[pair - 1], y) if pair else {}
            ordinals: Counter[tuple[int, int]] = Counter()
            left = initial
            for right, x_value in zip(rights.tolist(), xs.tolist(), strict=True):
                right = int(right)
                x = int(x_value)
                ordered = (left, right)
                ordinal = ordinals[ordered]
                ordinals[ordered] += 1
                spatial_values = spatial_positions.get(ordered, ())
                temporal_values = temporal_positions.get(ordered, ())
                spatial = int(spatial_values[ordinal]) if ordinal < len(spatial_values) else None
                temporal = int(temporal_values[ordinal]) if ordinal < len(temporal_values) else None
                source, value = choose_source(
                    mode=mode,
                    x=x,
                    spatial=spatial,
                    temporal=temporal,
                )
                edge = tuple(sorted(ordered))
                edge_id = EDGE_TO_ID[edge]
                frame_sources[edge_id].append(source)
                put_uleb(frame_coords[edge_id], x if source == 0 else zigzag(value))
                key = ("absolute_birth", "spatial_curve", "temporal_worldsheet")[source]
                source_counts[edge_name(edge)][key] += 1
                left = right
        for edge_id in range(len(EDGE_PAIRS)):
            records[source_stream_name(edge_id)].append(pack_two_bit(frame_sources[edge_id]))
            records[coord_stream_name(edge_id)].append(bytes(frame_coords[edge_id]))
    return CandidateSemantics(
        stream_records={name: tuple(value) for name, value in records.items()},
        source_counts=source_counts,
        temporal_events=temporal_event_counts(frames),
    )


def _race_records_task(task: tuple[str, tuple[bytes, ...]]) -> StreamRace:
    name, records = task
    canonical = pack_records(records)
    payloads = {
        "brotli-q11": bytes(brotli.compress(canonical, quality=11)),
        "lzma1-raw": bd1.lzma1_raw(canonical),
        "smevr-r7-nibble": bd1.smevr_records(list(records)),
    }
    if unpack_records(brotli.decompress(payloads["brotli-q11"])) != records:
        raise WorldsheetError(f"{name}: Brotli record round-trip failed")
    if unpack_records(bd1.unlzma1_raw(payloads["lzma1-raw"], len(canonical))) != records:
        raise WorldsheetError(f"{name}: LZMA1 record round-trip failed")
    if tuple(bd1.unsmevr_records(payloads["smevr-r7-nibble"])) != records:
        raise WorldsheetError(f"{name}: SMEVR record round-trip failed")
    winner = min(payloads, key=lambda codec: (len(payloads[codec]), CODEC_IDS[codec]))
    return StreamRace(
        name=name,
        records_sha256=sha256_bytes(canonical),
        canonical_raw_bytes=len(canonical),
        payloads=payloads,
        winner=winner,
    )


def race_streams(
    records: dict[str, tuple[bytes, ...]],
    *,
    pool: mp.pool.Pool,
    cache: dict[str, StreamRace],
) -> dict[str, StreamRace]:
    tasks = sorted(records.items(), key=lambda item: stream_id(item[0]))
    resolved: dict[str, StreamRace] = {}
    pending: list[tuple[str, tuple[bytes, ...]]] = []
    pending_keys: dict[str, str] = {}
    for name, frame_records in tasks:
        key = sha256_bytes(pack_records(frame_records))
        if key in cache:
            resolved[name] = dataclasses.replace(cache[key], name=name)
        else:
            pending.append((name, frame_records))
            pending_keys[name] = key
    for race in pool.map(_race_records_task, pending, chunksize=1):
        key = pending_keys[race.name]
        cache[key] = race
        resolved[race.name] = race
    return resolved


def build_envelope(
    *,
    table: Sequence[Sequence[int]],
    races: dict[str, StreamRace],
) -> bytes:
    ordered = sorted(races.values(), key=lambda race: stream_id(race.name))
    if not ordered:
        raise WorldsheetError("cannot build an envelope without streams")
    output = bytearray(HEADER_STRUCT.pack(MAGIC, VERSION, N_PAIRS, HEIGHT, WIDTH, len(ordered)))
    output.extend(rank_table_bytes(table))
    for race in ordered:
        coded = race.payloads[race.winner]
        output.extend(
            STREAM_HEADER.pack(
                stream_id(race.name),
                CODEC_IDS[race.winner],
                race.canonical_raw_bytes,
                len(coded),
            )
        )
        output.extend(coded)
    return bytes(output)


def decode_stream_payload(codec: str, payload: bytes, raw_size: int) -> tuple[bytes, ...]:
    if codec == "brotli-q11":
        raw = brotli.decompress(payload)
        if len(raw) != raw_size:
            raise WorldsheetError("Brotli canonical raw size mismatch")
        return unpack_records(raw)
    if codec == "lzma1-raw":
        return unpack_records(bd1.unlzma1_raw(payload, raw_size))
    if codec == "smevr-r7-nibble":
        records = tuple(bd1.unsmevr_records(payload))
        if len(pack_records(records)) != raw_size:
            raise WorldsheetError("SMEVR canonical raw size mismatch")
        return records
    raise WorldsheetError(f"unknown codec {codec!r}")


def parse_envelope(payload: bytes) -> tuple[tuple[tuple[int, ...], ...], dict[str, tuple[bytes, ...]]]:
    if len(payload) < HEADER_STRUCT.size + N_CLASSES * (N_CLASSES - 1):
        raise WorldsheetError("worldsheet envelope is truncated")
    magic, version, pairs, height, width, stream_count = HEADER_STRUCT.unpack_from(payload)
    if (magic, version, pairs, height, width) != (MAGIC, VERSION, N_PAIRS, HEIGHT, WIDTH):
        raise WorldsheetError("worldsheet envelope header mismatch")
    offset = HEADER_STRUCT.size
    table_size = N_CLASSES * (N_CLASSES - 1)
    table = rank_table_from_bytes(payload[offset : offset + table_size])
    offset += table_size
    streams: dict[str, tuple[bytes, ...]] = {}
    previous_id = -1
    for _ in range(stream_count):
        if offset + STREAM_HEADER.size > len(payload):
            raise WorldsheetError("worldsheet stream header is truncated")
        identifier, codec_id, raw_size, coded_size = STREAM_HEADER.unpack_from(payload, offset)
        offset += STREAM_HEADER.size
        if identifier <= previous_id:
            raise WorldsheetError("worldsheet stream ids are not strictly ordered")
        previous_id = identifier
        coded = payload[offset : offset + coded_size]
        offset += coded_size
        if len(coded) != coded_size or codec_id not in ID_CODECS:
            raise WorldsheetError("worldsheet stream is truncated or has an invalid codec")
        name = stream_name(identifier)
        records = decode_stream_payload(ID_CODECS[codec_id], coded, raw_size)
        if len(records) != N_PAIRS:
            raise WorldsheetError("worldsheet stream does not contain n600 records")
        streams[name] = records
    if offset != len(payload):
        raise WorldsheetError("worldsheet envelope has trailing bytes")
    expected_names = {"topology"}
    for edge_id in range(len(EDGE_PAIRS)):
        expected_names.add(source_stream_name(edge_id))
        expected_names.add(coord_stream_name(edge_id))
    if set(streams) != expected_names:
        raise WorldsheetError("worldsheet envelope stream roster is incomplete")
    return table, streams


def iter_decode_frames(payload: bytes) -> Iterator[np.ndarray]:
    table, streams = parse_envelope(payload)
    decoded_frames: list[FrameGrammar] = []
    for pair in range(N_PAIRS):
        initials, counts, rights = decode_topology_record(streams["topology"][pair], table, height=HEIGHT)
        edge_occurrences = Counter()
        offset = 0
        for initial, count in zip(initials.tolist(), counts.tolist(), strict=True):
            left = int(initial)
            stop = offset + int(count)
            for right in rights[offset:stop].tolist():
                edge_occurrences[EDGE_TO_ID[tuple(sorted((left, int(right))))]] += 1
                left = int(right)
            offset = stop
        edge_sources: dict[int, list[int]] = {}
        edge_coords: dict[int, list[int]] = {}
        for edge_id in range(len(EDGE_PAIRS)):
            count = int(edge_occurrences[edge_id])
            edge_sources[edge_id] = unpack_two_bit(streams[source_stream_name(edge_id)][pair], count)
            coordinate_payload = streams[coord_stream_name(edge_id)][pair]
            coordinate_values: list[int] = []
            coordinate_offset = 0
            for _ in range(count):
                value, coordinate_offset = get_uleb(coordinate_payload, coordinate_offset)
                coordinate_values.append(value)
            if coordinate_offset != len(coordinate_payload):
                raise WorldsheetError("coordinate frame record has trailing bytes")
            edge_coords[edge_id] = coordinate_values
        edge_offsets = Counter()
        xs: list[int] = []
        topology_offset = 0
        provisional_rows: list[tuple[int, list[int], list[int]]] = []
        for y, (initial, count) in enumerate(zip(initials.tolist(), counts.tolist(), strict=True)):
            stop = topology_offset + int(count)
            row_rights = rights[topology_offset:stop]
            spatial_positions = (
                _row_positions_from_lists(*provisional_rows[y - 1]) if y else {}
            )
            temporal_positions = row_ordered_positions(decoded_frames[pair - 1], y) if pair else {}
            ordinals: Counter[tuple[int, int]] = Counter()
            row_xs: list[int] = []
            left = int(initial)
            for right_value in row_rights.tolist():
                right = int(right_value)
                ordered = (left, right)
                ordinal = ordinals[ordered]
                ordinals[ordered] += 1
                edge_id = EDGE_TO_ID[tuple(sorted(ordered))]
                edge_offset = edge_offsets[edge_id]
                edge_offsets[edge_id] += 1
                source = edge_sources[edge_id][edge_offset]
                encoded = edge_coords[edge_id][edge_offset]
                if source == 0:
                    x = encoded
                else:
                    reference = spatial_positions if source == 1 else temporal_positions
                    values = reference.get(ordered, ())
                    if ordinal >= len(values):
                        raise WorldsheetError("coordinate source references a missing curve ordinal")
                    x = int(values[ordinal]) + unzigzag(encoded)
                if not 1 <= x < WIDTH or (row_xs and x <= row_xs[-1]):
                    raise WorldsheetError("decoded boundary coordinate is invalid")
                row_xs.append(int(x))
                xs.append(int(x))
                left = right
            provisional_rows.append((int(initial), row_rights.tolist(), row_xs))
            topology_offset = stop
        if any(edge_offsets[edge_id] != edge_occurrences[edge_id] for edge_id in range(len(EDGE_PAIRS))):
            raise WorldsheetError("receiver did not consume every edge occurrence")
        frame = FrameGrammar(
            initials=initials,
            counts=counts,
            rights=rights,
            xs=np.asarray(xs, dtype=np.uint16),
        )
        decoded_frames.append(frame)
        yield render_frame(frame)


def _row_positions_from_lists(
    initial: int,
    rights: Sequence[int],
    xs: Sequence[int],
) -> dict[tuple[int, int], list[int]]:
    output: dict[tuple[int, int], list[int]] = defaultdict(list)
    left = int(initial)
    for right, x in zip(rights, xs, strict=True):
        output[(left, int(right))].append(int(x))
        left = int(right)
    return output


def frame_quantization_candidates(frame: FrameGrammar, q_step: int, frame_index: int) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    offset = 0
    for y, count in enumerate(frame.counts.tolist()):
        stop = offset + int(count)
        row_xs = frame.xs[offset:stop]
        for local, x_value in enumerate(row_xs.tolist()):
            x = int(x_value)
            quantized = min(WIDTH - 1, max(1, ((x + q_step // 2) // q_step) * q_step))
            shift = abs(quantized - x)
            if not shift:
                continue
            key_payload = struct.pack("<HHHHH", frame_index, y, local, q_step, x)
            tie = int.from_bytes(hashlib.blake2b(key_payload, digest_size=8).digest(), "little")
            candidates.append((shift, tie, offset + local, quantized))
        offset = stop
    return candidates


def _largest_remainder_quotas(weights: Sequence[int], total: int) -> list[int]:
    weight_sum = sum(weights)
    if weight_sum <= 0 or total <= 0:
        return [0 for _ in weights]
    numerators = [total * weight for weight in weights]
    quotas = [value // weight_sum for value in numerators]
    remainder = total - sum(quotas)
    order = sorted(range(len(weights)), key=lambda index: (-(numerators[index] % weight_sum), index))
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas


def quantize_frames(
    frames: Sequence[FrameGrammar],
    *,
    q_step: int,
    error_cap: int,
) -> QuantizationResult:
    if q_step < 2 or q_step > 64:
        raise WorldsheetError("quantization step must be in [2,64]")
    candidate_counts = [len(frame_quantization_candidates(frame, q_step, index)) for index, frame in enumerate(frames)]
    quotas = _largest_remainder_quotas(candidate_counts, error_cap)
    quantized_frames: list[FrameGrammar] = []
    selected_shift = 0
    selected_boundaries = 0
    unused_quota = 0
    selected_by_edge = {
        edge_name(edge): {"boundaries": 0, "shift_upper_bound": 0}
        for edge in EDGE_PAIRS
    }
    for frame_index, (frame, quota) in enumerate(zip(frames, quotas, strict=True)):
        mutable_xs = frame.xs.astype(np.int64).copy()
        spent = 0
        candidates = frame_quantization_candidates(frame, q_step, frame_index)
        candidates.sort(key=lambda row: (row[0], row[1]))
        for shift, _tie, flat_index, quantized in candidates:
            if spent + shift > quota:
                continue
            row = int(np.searchsorted(np.cumsum(frame.counts, dtype=np.int64), flat_index, side="right"))
            row_start = int(np.sum(frame.counts[:row], dtype=np.int64))
            local = flat_index - row_start
            row_count = int(frame.counts[row])
            lower = int(mutable_xs[flat_index - 1]) if local else 0
            upper = int(mutable_xs[flat_index + 1]) if local + 1 < row_count else WIDTH
            if not lower < quantized < upper:
                continue
            mutable_xs[flat_index] = quantized
            spent += shift
            selected_shift += shift
            selected_boundaries += 1
            initial, rights, _ = frame.row(row)
            left = initial
            for right in rights[:local].tolist():
                left = int(right)
            right = int(rights[local])
            edge = tuple(sorted((left, right)))
            selected_by_edge[edge_name(edge)]["boundaries"] += 1
            selected_by_edge[edge_name(edge)]["shift_upper_bound"] += shift
        unused_quota += quota - spent
        quantized_frames.append(
            FrameGrammar(
                initials=frame.initials,
                counts=frame.counts,
                rights=frame.rights,
                xs=mutable_xs.astype(np.uint16),
            )
        )
    if selected_shift > error_cap:
        raise WorldsheetError("quantized boundary-shift upper bound exceeds the tolerance cap")
    return QuantizationResult(
        frames=tuple(quantized_frames),
        q_step=q_step,
        requested_error_cap=error_cap,
        selected_shift_upper_bound=selected_shift,
        selected_boundaries=selected_boundaries,
        selected_by_edge=selected_by_edge,
        unused_frame_quota=unused_quota,
    )


def verify_candidate(
    payload: bytes,
    *,
    original_labels: np.memmap | np.ndarray,
    expected_frames: Sequence[FrameGrammar],
) -> dict[str, Any]:
    decoded_hash = hashlib.sha256()
    expected_hash = hashlib.sha256()
    original_hash = hashlib.sha256()
    mismatches = 0
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    frame_count = 0
    for pair, decoded in enumerate(iter_decode_frames(payload)):
        expected = render_frame(expected_frames[pair])
        original = np.asarray(original_labels[pair], dtype=np.uint8)
        if not np.array_equal(decoded, expected):
            raise WorldsheetError(f"receiver parse-back differs from candidate semantics at pair {pair}")
        decoded_hash.update(decoded.tobytes())
        expected_hash.update(expected.tobytes())
        original_hash.update(original.tobytes())
        changed = decoded != original
        mismatches += int(changed.sum())
        if np.any(changed):
            np.add.at(confusion, (original[changed], decoded[changed]), 1)
        frame_count += 1
    if frame_count != N_PAIRS or decoded_hash.hexdigest() != expected_hash.hexdigest():
        raise WorldsheetError("receiver did not decode the complete candidate semantics")
    return {
        "pairs": frame_count,
        "receiver_semantic_roundtrip": True,
        "decoded_u8_sha256": decoded_hash.hexdigest(),
        "original_u8_sha256": original_hash.hexdigest(),
        "mismatches_vs_original": mismatches,
        "dseg_equivalent_mass": mismatches / TOTAL_CELLS,
        "confusion_original_to_decoded": confusion.tolist(),
    }


def quantization_metadata(quantization: QuantizationResult) -> dict[str, Any]:
    return {
        "q_step": quantization.q_step,
        "requested_error_cap": quantization.requested_error_cap,
        "selected_shift_upper_bound": quantization.selected_shift_upper_bound,
        "selected_boundaries": quantization.selected_boundaries,
        "selected_by_edge": quantization.selected_by_edge,
        "unused_frame_quota": quantization.unused_frame_quota,
        "selection": "all-n600 frame-stratified largest-remainder quotas; within-frame shift then BLAKE2 tie order",
    }


def candidate_artifacts_valid(
    candidate_dir: Path,
    *,
    candidate_id: str,
    mode: str,
    source_sha256: str,
    original_labels: np.memmap,
    expected_frames: Sequence[FrameGrammar],
) -> dict[str, Any] | None:
    receipt_path = candidate_dir / "receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text())
        if receipt["candidate_id"] != candidate_id or receipt["selection_mode"] != mode:
            raise WorldsheetError("candidate receipt identity differs from the requested candidate")
        if receipt["source"]["script_sha256"] != source_sha256:
            raise WorldsheetError("candidate receipt was built by different source; use a new output root")
        artifacts = receipt["retained_artifacts"]
        for relative, metadata in artifacts.items():
            path = candidate_dir / relative
            if path.stat().st_size != metadata["bytes"] or sha256_file(path) != metadata["sha256"]:
                raise WorldsheetError(f"retained candidate artifact drifted: {path}")
        verification = verify_candidate(
            (candidate_dir / "worldsheet.wsg").read_bytes(),
            original_labels=original_labels,
            expected_frames=expected_frames,
        )
        if verification != receipt["verification"]:
            raise WorldsheetError("resumed candidate verification differs from its receipt")
        return receipt
    except FileNotFoundError:
        return None
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise WorldsheetError(f"candidate checkpoint is corrupt: {candidate_dir}") from exc


def price_candidate(
    *,
    candidate_id: str,
    frames: Sequence[FrameGrammar],
    original_labels: np.memmap,
    table: Sequence[Sequence[int]],
    mode: str,
    candidate_dir: Path,
    pool: mp.pool.Pool,
    race_cache: dict[str, StreamRace],
    quantization: QuantizationResult | None,
) -> dict[str, Any]:
    source_sha256 = sha256_file(Path(__file__))
    resumed = candidate_artifacts_valid(
        candidate_dir,
        candidate_id=candidate_id,
        mode=mode,
        source_sha256=source_sha256,
        original_labels=original_labels,
        expected_frames=frames,
    )
    if resumed is not None:
        return resumed
    semantics = build_candidate_semantics(frames, table, mode=mode)
    races = race_streams(semantics.stream_records, pool=pool, cache=race_cache)
    retained: dict[str, dict[str, Any]] = {}
    stream_rows: dict[str, Any] = {}
    for name, race in sorted(races.items(), key=lambda item: stream_id(item[0])):
        row = {
            "canonical_raw_bytes": race.canonical_raw_bytes,
            "records_sha256": race.records_sha256,
            "winner": race.winner,
            "coders": {},
        }
        for codec, payload in sorted(race.payloads.items()):
            relative = Path("streams") / f"{name}.{codec}.bin"
            path = candidate_dir / relative
            atomic_bytes(path, payload)
            metadata = {"bytes": len(payload), "sha256": sha256_bytes(payload)}
            retained[str(relative)] = metadata
            row["coders"][codec] = {**metadata, "path": str(path)}
        stream_rows[name] = row
    envelope = build_envelope(table=table, races=races)
    envelope_path = candidate_dir / "worldsheet.wsg"
    atomic_bytes(envelope_path, envelope)
    retained["worldsheet.wsg"] = {"bytes": len(envelope), "sha256": sha256_bytes(envelope)}
    verification = verify_candidate(
        envelope,
        original_labels=original_labels,
        expected_frames=frames,
    )
    if quantization is None and verification["mismatches_vs_original"] != 0:
        raise WorldsheetError("lossless candidate does not reproduce the original partition")
    header_bytes = HEADER_STRUCT.size + N_CLASSES * (N_CLASSES - 1)
    topology_race = races["topology"]
    grammar_bytes = header_bytes + STREAM_HEADER.size + len(topology_race.payloads[topology_race.winner])
    coordinate_bytes = 0
    per_stratum: dict[str, Any] = {}
    for edge_id, edge in enumerate(EDGE_PAIRS):
        source_race = races[source_stream_name(edge_id)]
        coord_race = races[coord_stream_name(edge_id)]
        source_bytes = STREAM_HEADER.size + len(source_race.payloads[source_race.winner])
        coord_bytes = STREAM_HEADER.size + len(coord_race.payloads[coord_race.winner])
        grammar_bytes += source_bytes
        coordinate_bytes += coord_bytes
        per_stratum[edge_name(edge)] = {
            "event_stream_bytes": source_bytes,
            "coordinate_stream_bytes": coord_bytes,
            "total_bytes_excluding_shared_topology": source_bytes + coord_bytes,
            "event_winner": source_race.winner,
            "coordinate_winner": coord_race.winner,
            "coordinate_sources": semantics.source_counts[edge_name(edge)],
            "temporal_birth_death": semantics.temporal_events[edge_name(edge)],
        }
    if grammar_bytes + coordinate_bytes != len(envelope):
        raise WorldsheetError("two-part accounting does not close to the receiver envelope")
    receipt = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "scorer_run": False,
        "candidate_id": candidate_id,
        "selection_mode": mode,
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": source_sha256,
        },
        "lossless_leg": quantization is None,
        "tolerance_leg": quantization is not None,
        "two_part_code_length": {
            "grammar_and_event_bytes": grammar_bytes,
            "coordinate_innovation_bytes": coordinate_bytes,
            "measured_total_bytes": len(envelope),
        },
        "shared_topology": {
            "header_and_induced_rank_table_bytes": header_bytes,
            "stream_header_bytes": STREAM_HEADER.size,
            "coded_bytes": len(topology_race.payloads[topology_race.winner]),
            "winner": topology_race.winner,
        },
        "per_stratum": per_stratum,
        "quantization": quantization_metadata(quantization) if quantization is not None else None,
        "verification": verification,
        "streams": stream_rows,
        "retained_artifacts": retained,
    }
    atomic_json(candidate_dir / "receipt.json", receipt)
    return receipt


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < MIN_FREE_BYTES:
        raise WorldsheetError(
            f"SSD preflight requires {MIN_FREE_BYTES} free bytes; only {usage.free} available at {output}"
        )
    probe = output / f".write_probe_{os.getpid()}"
    try:
        atomic_bytes(probe, b"ddm_ws0")
        if probe.read_bytes() != b"ddm_ws0":
            raise WorldsheetError("SSD write probe did not round-trip")
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "path": str(output),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "write_probe": "passed",
    }


def select_best(receipts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return min(
        receipts,
        key=lambda receipt: (
            receipt["two_part_code_length"]["measured_total_bytes"],
            receipt["candidate_id"],
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache = Path(args.gt_cache).resolve()
    output = Path(args.output_dir).resolve()
    resume_from = Path(args.resume_from).resolve() if args.resume_from else output
    if resume_from != output:
        raise WorldsheetError("--resume-from must identify the same durable output root as --output-dir")
    if not 1 <= args.workers <= MAX_WORKERS:
        raise WorldsheetError(f"--workers must be in [1,{MAX_WORKERS}]")
    if not 0.0 < float(args.tolerance_dseg) <= 0.01:
        raise WorldsheetError("--tolerance-dseg must be in (0,0.01]")
    quantization_steps = tuple(dict.fromkeys(int(value) for value in args.quantization_steps))
    if not quantization_steps or any(value < 2 or value > 64 for value in quantization_steps):
        raise WorldsheetError("--quantization-steps must be unique integers in [2,64]")
    storage = storage_preflight(output)
    if cache.stat().st_size != 5_078_017_610:
        raise WorldsheetError("GT cache byte size differs from the pinned n600 object")
    cache_sha = sha256_file(cache)
    if cache_sha != EXPECTED_CACHE_SHA256:
        raise WorldsheetError("GT cache SHA-256 differs from the pinned n600 object")
    es1 = Path(args.es1_memo).resolve()
    es1_sha = sha256_file(es1)
    if es1_sha != EXPECTED_ES1_SHA256:
        raise WorldsheetError("es1 provenance memo SHA-256 differs from the charter pin")
    labels = open_stored_npy_memmap(cache, "lstars")
    if tuple(labels.shape) != (N_PAIRS, HEIGHT, WIDTH) or labels.dtype != np.dtype("int64"):
        raise WorldsheetError(f"unexpected lstars object {labels.shape} {labels.dtype}")
    context = mp.get_context("spawn")
    with context.Pool(
        processes=args.workers,
        initializer=_extract_worker_init,
        initargs=(str(cache),),
    ) as pool:
        frames, extract_manifest = extract_or_resume_frames(
            cache=cache,
            cache_sha256=cache_sha,
            extract_dir=output / "stage_01_extracted",
            pool=pool,
        )
        table = induced_rank_table(frames)
        race_cache: dict[str, StreamRace] = {}
        exact_receipts: list[dict[str, Any]] = []
        for mode in SELECTION_MODES:
            candidate_id = f"lossless_{mode}"
            exact_receipts.append(
                price_candidate(
                    candidate_id=candidate_id,
                    frames=frames,
                    original_labels=labels,
                    table=table,
                    mode=mode,
                    candidate_dir=output / candidate_id,
                    pool=pool,
                    race_cache=race_cache,
                    quantization=None,
                )
            )
        tolerance_cap = int(np.floor(float(args.tolerance_dseg) * TOTAL_CELLS))
        tolerance_receipts: list[dict[str, Any]] = []
        for q_step in quantization_steps:
            quantization = quantize_frames(frames, q_step=q_step, error_cap=tolerance_cap)
            for mode in SELECTION_MODES:
                candidate_id = f"tolerance_q{q_step}_{mode}"
                receipt = price_candidate(
                    candidate_id=candidate_id,
                    frames=quantization.frames,
                    original_labels=labels,
                    table=table,
                    mode=mode,
                    candidate_dir=output / candidate_id,
                    pool=pool,
                    race_cache=race_cache,
                    quantization=quantization,
                )
                if receipt["verification"]["mismatches_vs_original"] > tolerance_cap:
                    raise WorldsheetError(f"{candidate_id} escaped the tolerance allowance")
                tolerance_receipts.append(receipt)
    best_lossless = select_best(exact_receipts)
    best_tolerance = select_best(tolerance_receipts)
    result = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "scorer_run": False,
        "modal_run": False,
        "input": {
            "path": str(cache),
            "bytes": cache.stat().st_size,
            "sha256": cache_sha,
            "member": "lstars.npy",
            "shape": list(labels.shape),
            "dtype": str(labels.dtype),
        },
        "provenance_pins": {str(es1): es1_sha},
        "storage_preflight": storage,
        "workers": args.workers,
        "process_visibility": "sandbox_blocked_ps_and_top; workers hard-capped at 8 so charter jo1 288% + ws0 <= 1088% nominal",
        "stage_01_extract": extract_manifest,
        "induced_transition_rank_table": [list(row) for row in table],
        "lossless_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "measured_total_bytes": row["two_part_code_length"]["measured_total_bytes"],
                "mismatches": row["verification"]["mismatches_vs_original"],
            }
            for row in exact_receipts
        ],
        "tolerance_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "measured_total_bytes": row["two_part_code_length"]["measured_total_bytes"],
                "mismatches": row["verification"]["mismatches_vs_original"],
                "dseg_equivalent_mass": row["verification"]["dseg_equivalent_mass"],
            }
            for row in tolerance_receipts
        ],
        "best_lossless": {
            "candidate_id": best_lossless["candidate_id"],
            "receipt": str(output / best_lossless["candidate_id"] / "receipt.json"),
            "measured_total_bytes": best_lossless["two_part_code_length"]["measured_total_bytes"],
            "two_part_code_length": best_lossless["two_part_code_length"],
            "per_stratum": best_lossless["per_stratum"],
        },
        "best_tolerance": {
            "candidate_id": best_tolerance["candidate_id"],
            "receipt": str(output / best_tolerance["candidate_id"] / "receipt.json"),
            "measured_total_bytes": best_tolerance["two_part_code_length"]["measured_total_bytes"],
            "two_part_code_length": best_tolerance["two_part_code_length"],
            "verification": best_tolerance["verification"],
            "per_stratum": best_tolerance["per_stratum"],
        },
        "controls": {
            "worldsheet_conjecture_bytes": 90_000,
            "pp1_direct_partition_ceiling_bytes": 173_616,
            "sp1_explicit_support_best_bytes": 421_366,
            "lossless_ge_130000": best_lossless["two_part_code_length"]["measured_total_bytes"] >= 130_000,
            "tolerance_ge_110000": best_tolerance["two_part_code_length"]["measured_total_bytes"] >= 110_000,
        },
    }
    result["falsifier"] = {
        "triggered": bool(
            result["controls"]["lossless_ge_130000"] and result["controls"]["tolerance_ge_110000"]
        ),
        "scope": "FORMULATION: horizontal row-boundary temporal worldsheet with induced transition ranks, ordinal spatial/temporal curve prediction, and boundary-coordinate quantization",
    }
    atomic_json(output / "FINAL_RESULT.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--resume-from",
        default=str(DEFAULT_OUTPUT),
        help="Durable stage root; must equal --output-dir so every launch is crash-resumable.",
    )
    parser.add_argument("--es1-memo", default=str(DEFAULT_ES1))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--tolerance-dseg", type=float, default=TOLERANCE_DSEG_DEFAULT)
    parser.add_argument("--quantization-steps", type=int, nargs="+", default=[2, 4, 8])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
