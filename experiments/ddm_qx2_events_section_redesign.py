#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""QX2 scorer-free exact redesign of QX1's event/exception section.

The experiment consumes the retained S2 event object used by QX1 and the exact
C1 baseline implied by the frozen target labels plus those events.  It builds
two receiver-conditional forms that never transmit ``(pair,row,col)``:

* ``boundary_field`` regenerates a boundary-dilation candidate field from the
  decoded baseline, codes candidate occupancy and target cells, and uses ranks
  in the same decoded field for the remaining interior events.
* ``distance_rank`` orders every decoded-baseline site by distance to its
  nearest class boundary and codes event ranks plus class transitions.

Every raw form, real-coder output, deterministic repeat, complete QXE packet,
and ZIP_STORED envelope is retained.  Decode must reproduce all 17,926 source
events exactly.  This is a rate/parse-back result, not a QX1 receiver or score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import shutil
import struct
import sys
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from scipy import ndimage

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.optimization.s2_partition_seed import (
    PartitionEvent,
    decode_partition_seed,
)

STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx2")
QX1_STORE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qx1")
SOURCE_PACKET: Final = (
    QX1_STORE / "retained/sections/08_events_exceptions_explicit_address_control/raw.bin"
)
SOURCE_PACKET_SHA256: Final = "df4c0534537a9919681509a0b44a392d7d4b46c812d7570c534e6b823adae7fc"
QX1_CENSUS: Final = QX1_STORE / "SECTION_CENSUS.json"
QX1_CENSUS_SHA256: Final = "d552955b1e7be08f03c77e3508756ad3e1dead9be759bf562f3cfc9ec8296db6"
QX1_CORE_PACKET: Final = (
    QX1_STORE / "retained/envelopes/core_without_events_exceptions/envelope.qxe"
)
QX1_CORE_PACKET_SHA256: Final = "4e6a2f6669c590258fc6c5d194ae6cb30951f5881e2055761de0bff753bdfb95"
QX1_CORE_ARCHIVE_BYTES: Final = 113_844
GT_CACHE: Final = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
GT_CACHE_LSTARS_SEMANTIC_SHA256: Final = (
    "f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557"
)
S2_TARGET_SEMANTIC_SHA256: Final = (
    "36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68"
)
KNOWN_CACHE_TARGET_MISMATCHES: Final = ((11, 286, 399, 0, 4, 0),)
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
EVENT_COUNT: Final = 17_926
GATE_BYTES_EXCLUSIVE: Final = 137_986
MINIMUM_FREE_BYTES: Final = 1_000_000_000
AXIS: Final = "[scorer-free exact rate and receiver-conditional parse-back measurement]"

QXE_HEADER: Final = struct.Struct(">4sBBH")
QXE_SECTION: Final = struct.Struct(">BBHII32sI")
BOUNDARY_HEADER: Final = struct.Struct(">4sBBHIIII32s")
RANK_HEADER: Final = struct.Struct(">4sBBHII32s")
ENUM_HEADER: Final = struct.Struct(">4sBBHIIII32s")
CODECS: Final = {"brotli_q11": 1, "lzma9e": 2, "zlib9": 3}
CODEC_NAMES: Final = {value: key for key, value in CODECS.items()}


class QX2Error(RuntimeError):
    """A custody, exactness, or framing gate failed closed."""


class BitWriter:
    """Canonical little-endian bit writer for enumerative ranks."""

    def __init__(self) -> None:
        self.output = bytearray()
        self.accumulator = 0
        self.available = 0
        self.bits = 0

    def write(self, value: int, width: int) -> None:
        if width < 0 or value < 0 or value >= 1 << width:
            raise QX2Error("enumerative rank does not fit its canonical width")
        self.accumulator |= value << self.available
        self.available += width
        self.bits += width
        while self.available >= 8:
            self.output.append(self.accumulator & 0xFF)
            self.accumulator >>= 8
            self.available -= 8

    def finish(self) -> bytes:
        if self.available:
            self.output.append(self.accumulator)
        return bytes(self.output)


class BitReader:
    """Canonical little-endian bit reader paired with :class:`BitWriter`."""

    def __init__(self, payload: bytes, bits: int) -> None:
        if len(payload) != (bits + 7) // 8:
            raise QX2Error("enumerative rank stream length is noncanonical")
        self.payload = payload
        self.total_bits = bits
        self.offset = 0

    def read(self, width: int) -> int:
        if width < 0 or self.offset + width > self.total_bits:
            raise QX2Error("enumerative rank stream is truncated")
        value = 0
        for shift in range(width):
            byte = self.payload[(self.offset + shift) // 8]
            value |= ((byte >> ((self.offset + shift) % 8)) & 1) << shift
        self.offset += width
        return value

    def finish(self) -> None:
        if self.offset != self.total_bits:
            raise QX2Error("enumerative rank stream has unconsumed bits")
        if self.total_bits % 8 and self.payload[-1] >> (self.total_bits % 8):
            raise QX2Error("enumerative rank stream padding is noncanonical")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def require_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise QX2Error(f"{label} custody drifted: {path}")
    return fact(path)


def write_uleb(value: int, output: bytearray) -> None:
    if value < 0:
        raise QX2Error("ULEB value cannot be negative")
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return


def read_uleb(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise QX2Error("truncated ULEB")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise QX2Error("ULEB exceeds uint64")


def pack_three_bit(values: list[int]) -> bytes:
    output = bytearray()
    accumulator = 0
    available = 0
    for value in values:
        if not 0 <= value < 5:
            raise QX2Error("target class is outside the five-class alphabet")
        accumulator |= value << available
        available += 3
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator)
    return bytes(output)


def unpack_three_bit(payload: bytes, count: int) -> list[int]:
    values: list[int] = []
    accumulator = 0
    available = 0
    offset = 0
    while len(values) < count:
        while available < 3:
            if offset >= len(payload):
                raise QX2Error("truncated three-bit target stream")
            accumulator |= payload[offset] << available
            available += 8
            offset += 1
        value = accumulator & 0x07
        accumulator >>= 3
        available -= 3
        if value >= 5:
            raise QX2Error("three-bit target stream contains a reserved class")
        values.append(value)
    if offset != len(payload) or accumulator:
        raise QX2Error("three-bit target stream has noncanonical padding")
    return values


def combination_rank(selected: list[int]) -> int:
    """Return the colex rank and its exact combination size."""

    return sum(math.comb(position, index + 1) for index, position in enumerate(selected))


def combination_unrank(rank: int, n: int, k: int) -> list[int]:
    """Invert the colex rank for one ``k``-subset of ``range(n)``."""

    if not 0 <= k <= n or not 0 <= rank < math.comb(n, k):
        raise QX2Error("enumerative subset identity is invalid")
    selected = [0] * k
    upper = n
    remainder = rank
    for order in range(k, 0, -1):
        low = order - 1
        high = upper - 1
        while low < high:
            midpoint = (low + high + 1) // 2
            if math.comb(midpoint, order) <= remainder:
                low = midpoint
            else:
                high = midpoint - 1
        selected[order - 1] = low
        remainder -= math.comb(low, order)
        upper = low
    if remainder:
        raise QX2Error("enumerative subset rank did not close")
    return selected


def transition_groups() -> tuple[tuple[int, int], ...]:
    return tuple((base, target) for base in range(5) for target in range(5) if target != base)


def boundary_and_distance(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    distance = ndimage.distance_transform_cdt(~boundary, metric="chessboard").astype(np.int16)
    return boundary, distance


def distance_order(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_distance = distance.reshape(-1)
    order = np.lexsort((np.arange(flat_distance.size, dtype=np.int32), flat_distance))
    inverse = np.empty(flat_distance.size, dtype=np.int32)
    inverse[order] = np.arange(flat_distance.size, dtype=np.int32)
    return order.astype(np.int32, copy=False), inverse


def event_key(event: PartitionEvent) -> tuple[int, int, int, int, int]:
    return (
        event.pair,
        event.row,
        event.col,
        event.target_class,
        event.baseline_class,
    )


def load_inputs(store: Path) -> tuple[Any, list[list[PartitionEvent]], np.memmap, dict[str, Any]]:
    resolved = store.resolve()
    if resolved != STORE.resolve():
        raise QX2Error(f"custody is pinned to {STORE}, not {resolved}")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise QX2Error(f"storage preflight failed: {free} < {MINIMUM_FREE_BYTES}")
    source_fact = require_file(SOURCE_PACKET, SOURCE_PACKET_SHA256, "QX1 event packet")
    census_fact = require_file(QX1_CENSUS, QX1_CENSUS_SHA256, "QX1 census")
    core_fact = require_file(QX1_CORE_PACKET, QX1_CORE_PACKET_SHA256, "QX1 core packet")
    gt_fact = require_file(GT_CACHE, GT_CACHE_SHA256, "GT cache")
    seed = decode_partition_seed(SOURCE_PACKET.read_bytes())
    if (seed.n_pairs, seed.height, seed.width, len(seed.events)) != (
        N_PAIRS,
        HEIGHT,
        WIDTH,
        EVENT_COUNT,
    ):
        raise QX2Error("source event geometry drifted")
    by_pair: list[list[PartitionEvent]] = [[] for _ in range(N_PAIRS)]
    for event in seed.events:
        by_pair[event.pair].append(event)
    targets = open_stored_npy_memmap(GT_CACHE, "lstars")
    if targets.shape != (N_PAIRS, HEIGHT, WIDTH) or not np.issubdtype(targets.dtype, np.integer):
        raise QX2Error("GT label member geometry or dtype drifted")
    preflight = {
        "schema": "ddm_qx2_preflight.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "storage": {"path": str(store), "observed_free_bytes": free},
        "inputs": {
            "source_event_packet": source_fact,
            "qx1_census": census_fact,
            "qx1_core_packet": core_fact,
            "gt_cache": gt_fact,
        },
        "denominators": {"pairs": N_PAIRS, "events": EVENT_COUNT, "sites": N_PAIRS * HEIGHT * WIDTH},
    }
    atomic_json(store / "checkpoints/STAGE0_PREFLIGHT.json", preflight)
    return seed, by_pair, targets, preflight


def build_baseline_and_stats(
    store: Path, by_pair: list[list[PartitionEvent]], targets: np.memmap
) -> tuple[Path, dict[str, Any]]:
    baseline_path = store / "retained/baseline/c1_baseline_labels.u8"
    stage_path = store / "checkpoints/STAGE0_CHARACTERIZATION.json"
    if stage_path.is_file() and baseline_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if fact(baseline_path) != stage["baseline"]:
            raise QX2Error("retained baseline drifted")
        return baseline_path, stage
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = baseline_path.with_suffix(".u8.partial")
    counts = np.array([len(events) for events in by_pair], dtype=np.int64)
    transition_counts: Counter[str] = Counter()
    distance_counts: Counter[int] = Counter()
    horizontal_runs = 0
    components = 0
    singleton_components = 0
    same_site_transition = 0
    previous_keys: set[tuple[int, int, int, int]] = set()
    baseline_hash = hashlib.sha256()
    cache_semantic_hash = hashlib.sha256()
    target_semantic_hash = hashlib.sha256()
    cache_target_mismatches: list[tuple[int, int, int, int, int, int]] = []
    with temporary.open("wb") as handle:
        for pair, events in enumerate(by_pair):
            cached_target = np.asarray(targets[pair], dtype=np.uint8)
            cache_semantic_hash.update(cached_target.tobytes(order="C"))
            target = cached_target.copy()
            baseline = cached_target.copy()
            for event in events:
                cached_class = int(cached_target[event.row, event.col])
                if cached_class != event.target_class:
                    cache_target_mismatches.append(
                        (
                            pair,
                            event.row,
                            event.col,
                            event.baseline_class,
                            event.target_class,
                            cached_class,
                        )
                    )
                target[event.row, event.col] = event.target_class
                baseline[event.row, event.col] = event.baseline_class
                transition_counts[f"{event.baseline_class}->{event.target_class}"] += 1
            target_semantic_hash.update(target.tobytes(order="C"))
            _, distance = boundary_and_distance(baseline)
            for event in events:
                distance_counts[int(distance[event.row, event.col])] += 1
            groups: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
            for event in events:
                groups[(event.baseline_class, event.target_class)].add((event.row, event.col))
            for points in groups.values():
                horizontal_runs += sum(1 for row, col in points if (row, col - 1) not in points)
                unseen = set(points)
                while unseen:
                    components += 1
                    stack = [unseen.pop()]
                    size = 0
                    while stack:
                        row, col = stack.pop()
                        size += 1
                        for drow in (-1, 0, 1):
                            for dcol in (-1, 0, 1):
                                if not (drow or dcol):
                                    continue
                                neighbor = (row + drow, col + dcol)
                                if neighbor in unseen:
                                    unseen.remove(neighbor)
                                    stack.append(neighbor)
                    singleton_components += int(size == 1)
            current_keys = {
                (event.row, event.col, event.baseline_class, event.target_class) for event in events
            }
            if pair:
                same_site_transition += len(current_keys & previous_keys)
            previous_keys = current_keys
            raw = baseline.tobytes(order="C")
            baseline_hash.update(raw)
            handle.write(raw)
    os.replace(temporary, baseline_path)
    if tuple(cache_target_mismatches) != KNOWN_CACHE_TARGET_MISMATCHES:
        raise QX2Error("GT cache versus retained S2 target mismatch roster drifted")
    if cache_semantic_hash.hexdigest() != GT_CACHE_LSTARS_SEMANTIC_SHA256:
        raise QX2Error("GT-cache lstars semantic SHA drifted")
    if target_semantic_hash.hexdigest() != S2_TARGET_SEMANTIC_SHA256:
        raise QX2Error("S2-completed target semantic SHA drifted")
    stage = {
        "schema": "ddm_qx2_characterization.v1",
        "complete": True,
        "axis": AXIS,
        "selection_mode": "full n600 retained S2/C1 event population",
        "events": EVENT_COUNT,
        "pairs": N_PAIRS,
        "events_per_pair": {
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "zero_event_pairs": int(np.count_nonzero(counts == 0)),
        },
        "transition_counts": dict(sorted(transition_counts.items())),
        "distance_to_decoded_baseline_boundary": {
            str(key): value for key, value in sorted(distance_counts.items())
        },
        "same_site_same_transition_from_previous_pair": same_site_transition,
        "same_site_same_transition_fraction": same_site_transition / EVENT_COUNT,
        "horizontal_runs": horizontal_runs,
        "events_per_horizontal_run": EVENT_COUNT / horizontal_runs,
        "eight_connected_components_by_transition": components,
        "singleton_components": singleton_components,
        "events_per_component": EVENT_COUNT / components,
        "baseline": fact(baseline_path),
        "gt_cache_lstars_semantic_sha256": cache_semantic_hash.hexdigest(),
        "s2_completed_target_semantic_sha256": target_semantic_hash.hexdigest(),
        "known_cache_target_mismatches": [list(row) for row in cache_target_mismatches],
        "interpretation": (
            "The population is spatially singleton-dominated and temporally nonpersistent, but "
            "98.6891% of events lie exactly on a boundary generated by the decoded C1 baseline."
        ),
    }
    atomic_json(stage_path, stage)
    return baseline_path, stage


def baseline_frame(path: Path, pair: int) -> np.ndarray:
    offset = pair * HEIGHT * WIDTH
    return np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(HEIGHT, WIDTH))


def encode_boundary_field(
    baseline_path: Path, by_pair: list[list[PartitionEvent]], radius: int, baseline_sha: str
) -> bytes:
    occupancy_chunks: list[np.ndarray] = []
    near_targets: list[int] = []
    residual = bytearray()
    candidate_total = 0
    residual_count = 0
    for pair, events in enumerate(by_pair):
        baseline = np.asarray(baseline_frame(baseline_path, pair))
        _, distance = boundary_and_distance(baseline)
        candidate = distance <= radius
        candidate_indices = np.flatnonzero(candidate.reshape(-1))
        candidate_total += int(candidate_indices.size)
        event_map = {event.row * WIDTH + event.col: event for event in events}
        selected = np.fromiter(
            (int(index) in event_map for index in candidate_indices),
            dtype=np.uint8,
            count=candidate_indices.size,
        )
        occupancy_chunks.append(selected)
        for index in candidate_indices[selected != 0].tolist():
            near_targets.append(event_map[int(index)].target_class)
        _, inverse = distance_order(distance)
        far = [event for event in events if not candidate[event.row, event.col]]
        write_uleb(len(far), residual)
        previous_rank = -1
        ranked = sorted((int(inverse[event.row * WIDTH + event.col]), event) for event in far)
        for rank, event in ranked:
            write_uleb(rank - previous_rank - 1, residual)
            residual.append(event.target_class | (event.baseline_class << 3))
            previous_rank = rank
            residual_count += 1
    occupancy = np.concatenate(occupancy_chunks) if occupancy_chunks else np.empty(0, dtype=np.uint8)
    packed_occupancy = np.packbits(occupancy, bitorder="little").tobytes()
    packed_targets = pack_three_bit(near_targets)
    header = BOUNDARY_HEADER.pack(
        b"QXB1",
        1,
        radius,
        N_PAIRS,
        EVENT_COUNT,
        candidate_total,
        len(near_targets),
        residual_count,
        bytes.fromhex(baseline_sha),
    )
    lengths = struct.pack(">III", len(packed_occupancy), len(packed_targets), len(residual))
    return header + lengths + packed_occupancy + packed_targets + bytes(residual)


def decode_boundary_field(payload: bytes, baseline_path: Path) -> tuple[PartitionEvent, ...]:
    if len(payload) < BOUNDARY_HEADER.size + 12:
        raise QX2Error("boundary-field payload is truncated")
    magic, version, radius, n_pairs, event_count, candidate_total, near_count, residual_count, baseline_sha = (
        BOUNDARY_HEADER.unpack_from(payload)
    )
    if (magic, version, n_pairs, event_count) != (b"QXB1", 1, N_PAIRS, EVENT_COUNT):
        raise QX2Error("boundary-field identity drifted")
    if baseline_sha.hex() != sha256_file(baseline_path):
        raise QX2Error("boundary-field baseline identity drifted")
    occ_len, target_len, residual_len = struct.unpack_from(">III", payload, BOUNDARY_HEADER.size)
    offset = BOUNDARY_HEADER.size + 12
    if offset + occ_len + target_len + residual_len != len(payload):
        raise QX2Error("boundary-field section lengths do not close")
    occ_payload = payload[offset : offset + occ_len]
    offset += occ_len
    target_payload = payload[offset : offset + target_len]
    offset += target_len
    residual_payload = payload[offset:]
    occupancy = np.unpackbits(np.frombuffer(occ_payload, dtype=np.uint8), bitorder="little")
    if occupancy.size < candidate_total or np.any(occupancy[candidate_total:]):
        raise QX2Error("boundary-field occupancy padding is noncanonical")
    occupancy = occupancy[:candidate_total]
    targets = unpack_three_bit(target_payload, near_count)
    events: list[PartitionEvent] = []
    occupancy_offset = 0
    target_offset = 0
    residual_offset = 0
    decoded_residual_count = 0
    for pair in range(N_PAIRS):
        baseline = np.asarray(baseline_frame(baseline_path, pair))
        _, distance = boundary_and_distance(baseline)
        candidate_indices = np.flatnonzero((distance <= radius).reshape(-1))
        selected = occupancy[occupancy_offset : occupancy_offset + candidate_indices.size]
        occupancy_offset += int(candidate_indices.size)
        for index in candidate_indices[selected != 0].tolist():
            target = targets[target_offset]
            target_offset += 1
            row, col = divmod(int(index), WIDTH)
            base = int(baseline[row, col])
            if target == base:
                raise QX2Error("boundary-field target equals decoded baseline")
            events.append(PartitionEvent(pair, row, col, target, base))
        count, residual_offset = read_uleb(residual_payload, residual_offset)
        order, _ = distance_order(distance)
        previous_rank = -1
        for _ in range(count):
            delta, residual_offset = read_uleb(residual_payload, residual_offset)
            if residual_offset >= len(residual_payload):
                raise QX2Error("boundary-field residual transition is truncated")
            rank = previous_rank + delta + 1
            packed = residual_payload[residual_offset]
            residual_offset += 1
            if rank >= order.size or packed & 0xC0:
                raise QX2Error("boundary-field residual rank/transition is invalid")
            index = int(order[rank])
            row, col = divmod(index, WIDTH)
            target = packed & 0x07
            base = (packed >> 3) & 0x07
            if int(baseline[row, col]) != base or target == base or distance[row, col] <= radius:
                raise QX2Error("boundary-field residual does not match decoded state")
            events.append(PartitionEvent(pair, row, col, target, base))
            previous_rank = rank
            decoded_residual_count += 1
    if (
        occupancy_offset != candidate_total
        or target_offset != near_count
        or residual_offset != len(residual_payload)
        or decoded_residual_count != residual_count
    ):
        raise QX2Error("boundary-field receiver did not consume all streams")
    return tuple(sorted(events))


def encode_boundary_enumerative(
    baseline_path: Path, by_pair: list[list[PartitionEvent]], baseline_sha: str
) -> bytes:
    groups = transition_groups()
    counts = bytearray()
    ranks = BitWriter()
    residual = bytearray()
    near_count = 0
    residual_count = 0
    for pair, events in enumerate(by_pair):
        baseline = np.asarray(baseline_frame(baseline_path, pair))
        boundary, distance = boundary_and_distance(baseline)
        by_transition: dict[tuple[int, int], list[int]] = defaultdict(list)
        far: list[PartitionEvent] = []
        for event in events:
            if boundary[event.row, event.col]:
                by_transition[(event.baseline_class, event.target_class)].append(
                    event.row * WIDTH + event.col
                )
                near_count += 1
            else:
                far.append(event)
        mask = sum(1 << index for index, group in enumerate(groups) if by_transition[group])
        counts.extend(mask.to_bytes(3, "little"))
        for group in groups:
            if by_transition[group]:
                write_uleb(len(by_transition[group]), counts)
        for base in range(5):
            available = np.flatnonzero(boundary.reshape(-1) & (baseline.reshape(-1) == base)).tolist()
            for target in range(5):
                if target == base:
                    continue
                selected_indices = sorted(by_transition[(base, target)])
                if not selected_indices:
                    continue
                positions = {index: position for position, index in enumerate(available)}
                try:
                    selected = [positions[index] for index in selected_indices]
                except KeyError as error:
                    raise QX2Error("enumerative event is absent from its decoded candidate field") from error
                total = math.comb(len(available), len(selected))
                ranks.write(combination_rank(selected), (total - 1).bit_length())
                selected_set = set(selected_indices)
                available = [index for index in available if index not in selected_set]
        write_uleb(len(far), residual)
        _, inverse = distance_order(distance)
        previous_rank = -1
        for rank, event in sorted(
            (int(inverse[event.row * WIDTH + event.col]), event) for event in far
        ):
            write_uleb(rank - previous_rank - 1, residual)
            residual.append(event.target_class | (event.baseline_class << 3))
            previous_rank = rank
            residual_count += 1
    rank_payload = ranks.finish()
    header = ENUM_HEADER.pack(
        b"QXC1",
        1,
        0,
        N_PAIRS,
        EVENT_COUNT,
        near_count,
        residual_count,
        ranks.bits,
        bytes.fromhex(baseline_sha),
    )
    lengths = struct.pack(">III", len(counts), len(rank_payload), len(residual))
    return header + lengths + bytes(counts) + rank_payload + bytes(residual)


def decode_boundary_enumerative(
    payload: bytes, baseline_path: Path
) -> tuple[PartitionEvent, ...]:
    if len(payload) < ENUM_HEADER.size + 12:
        raise QX2Error("enumerative boundary payload is truncated")
    magic, version, flags, n_pairs, event_count, near_count, residual_count, rank_bits, baseline_sha = (
        ENUM_HEADER.unpack_from(payload)
    )
    if (magic, version, flags, n_pairs, event_count) != (b"QXC1", 1, 0, N_PAIRS, EVENT_COUNT):
        raise QX2Error("enumerative boundary identity drifted")
    if baseline_sha.hex() != sha256_file(baseline_path):
        raise QX2Error("enumerative boundary baseline identity drifted")
    counts_len, ranks_len, residual_len = struct.unpack_from(">III", payload, ENUM_HEADER.size)
    offset = ENUM_HEADER.size + 12
    if offset + counts_len + ranks_len + residual_len != len(payload):
        raise QX2Error("enumerative boundary section lengths do not close")
    counts_payload = payload[offset : offset + counts_len]
    offset += counts_len
    rank_reader = BitReader(payload[offset : offset + ranks_len], rank_bits)
    offset += ranks_len
    residual_payload = payload[offset:]
    groups = transition_groups()
    counts_offset = 0
    residual_offset = 0
    decoded_near_count = 0
    decoded_residual_count = 0
    events: list[PartitionEvent] = []
    for pair in range(N_PAIRS):
        baseline = np.asarray(baseline_frame(baseline_path, pair))
        boundary, distance = boundary_and_distance(baseline)
        if counts_offset + 3 > len(counts_payload):
            raise QX2Error("enumerative transition mask is truncated")
        mask = int.from_bytes(counts_payload[counts_offset : counts_offset + 3], "little")
        counts_offset += 3
        if mask >> len(groups):
            raise QX2Error("enumerative transition mask has reserved bits")
        group_counts: dict[tuple[int, int], int] = {}
        for index, group in enumerate(groups):
            if mask & (1 << index):
                count, counts_offset = read_uleb(counts_payload, counts_offset)
                if not count:
                    raise QX2Error("enumerative transition mask names an empty group")
                group_counts[group] = count
            else:
                group_counts[group] = 0
        for base in range(5):
            available = np.flatnonzero(boundary.reshape(-1) & (baseline.reshape(-1) == base)).tolist()
            for target in range(5):
                if target == base:
                    continue
                count = group_counts[(base, target)]
                if not count:
                    continue
                if count > len(available):
                    raise QX2Error("enumerative transition count exceeds its candidate set")
                total = math.comb(len(available), count)
                rank = rank_reader.read((total - 1).bit_length())
                selected = combination_unrank(rank, len(available), count)
                selected_indices = [available[position] for position in selected]
                for index in selected_indices:
                    row, col = divmod(index, WIDTH)
                    events.append(PartitionEvent(pair, row, col, target, base))
                    decoded_near_count += 1
                selected_set = set(selected_indices)
                available = [index for index in available if index not in selected_set]
        count, residual_offset = read_uleb(residual_payload, residual_offset)
        order, _ = distance_order(distance)
        previous_rank = -1
        for _ in range(count):
            delta, residual_offset = read_uleb(residual_payload, residual_offset)
            if residual_offset >= len(residual_payload):
                raise QX2Error("enumerative residual transition is truncated")
            rank = previous_rank + delta + 1
            packed = residual_payload[residual_offset]
            residual_offset += 1
            if rank >= order.size or packed & 0xC0:
                raise QX2Error("enumerative residual rank/transition is invalid")
            index = int(order[rank])
            row, col = divmod(index, WIDTH)
            target = packed & 0x07
            base = (packed >> 3) & 0x07
            if int(baseline[row, col]) != base or target == base or boundary[row, col]:
                raise QX2Error("enumerative residual disagrees with decoded state")
            events.append(PartitionEvent(pair, row, col, target, base))
            previous_rank = rank
            decoded_residual_count += 1
    rank_reader.finish()
    if (
        counts_offset != len(counts_payload)
        or residual_offset != len(residual_payload)
        or decoded_near_count != near_count
        or decoded_residual_count != residual_count
    ):
        raise QX2Error("enumerative boundary receiver did not consume all streams")
    return tuple(sorted(events))


def encode_distance_rank(
    baseline_path: Path, by_pair: list[list[PartitionEvent]], baseline_sha: str
) -> bytes:
    body = bytearray()
    for pair, events in enumerate(by_pair):
        baseline = np.asarray(baseline_frame(baseline_path, pair))
        _, distance = boundary_and_distance(baseline)
        _, inverse = distance_order(distance)
        ranked = sorted((int(inverse[event.row * WIDTH + event.col]), event) for event in events)
        write_uleb(len(ranked), body)
        previous_rank = -1
        for rank, event in ranked:
            write_uleb(rank - previous_rank - 1, body)
            body.append(event.target_class | (event.baseline_class << 3))
            previous_rank = rank
    return RANK_HEADER.pack(
        b"QXR1", 1, 0, N_PAIRS, EVENT_COUNT, len(body), bytes.fromhex(baseline_sha)
    ) + bytes(body)


def decode_distance_rank(payload: bytes, baseline_path: Path) -> tuple[PartitionEvent, ...]:
    if len(payload) < RANK_HEADER.size:
        raise QX2Error("distance-rank payload is truncated")
    magic, version, flags, n_pairs, event_count, body_len, baseline_sha = RANK_HEADER.unpack_from(payload)
    if (magic, version, flags, n_pairs, event_count) != (b"QXR1", 1, 0, N_PAIRS, EVENT_COUNT):
        raise QX2Error("distance-rank identity drifted")
    if baseline_sha.hex() != sha256_file(baseline_path) or RANK_HEADER.size + body_len != len(payload):
        raise QX2Error("distance-rank baseline or length drifted")
    body = payload[RANK_HEADER.size :]
    offset = 0
    events: list[PartitionEvent] = []
    for pair in range(N_PAIRS):
        baseline = np.asarray(baseline_frame(baseline_path, pair))
        _, distance = boundary_and_distance(baseline)
        order, _ = distance_order(distance)
        count, offset = read_uleb(body, offset)
        previous_rank = -1
        for _ in range(count):
            delta, offset = read_uleb(body, offset)
            if offset >= len(body):
                raise QX2Error("distance-rank transition is truncated")
            rank = previous_rank + delta + 1
            packed = body[offset]
            offset += 1
            if rank >= order.size or packed & 0xC0:
                raise QX2Error("distance-rank value is invalid")
            index = int(order[rank])
            row, col = divmod(index, WIDTH)
            target = packed & 0x07
            base = (packed >> 3) & 0x07
            if int(baseline[row, col]) != base or target == base:
                raise QX2Error("distance-rank transition disagrees with decoded baseline")
            events.append(PartitionEvent(pair, row, col, target, base))
            previous_rank = rank
    if offset != len(body):
        raise QX2Error("distance-rank stream has trailing bytes")
    return tuple(sorted(events))


def compress(codec: str, raw: bytes) -> bytes:
    if codec == "brotli_q11":
        return brotli.compress(raw, quality=11)
    if codec == "lzma9e":
        return lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    if codec == "zlib9":
        return zlib.compress(raw, level=9)
    raise QX2Error(f"unknown codec: {codec}")


def decompress(codec: str, coded: bytes) -> bytes:
    if codec == "brotli_q11":
        return brotli.decompress(coded)
    if codec == "lzma9e":
        return lzma.decompress(coded)
    if codec == "zlib9":
        return zlib.decompress(coded)
    raise QX2Error(f"unknown codec: {codec}")


def split_qxe_records(packet: bytes) -> tuple[int, int, list[bytes]]:
    if len(packet) < QXE_HEADER.size:
        raise QX2Error("QXE core packet is truncated")
    magic, version, flags, count = QXE_HEADER.unpack_from(packet)
    if (magic, version, flags, count) != (b"QXE1", 1, 0, 7):
        raise QX2Error("QXE core identity drifted")
    offset = QXE_HEADER.size
    records: list[bytes] = []
    for expected_id in range(1, 8):
        start = offset
        if offset + QXE_SECTION.size > len(packet):
            raise QX2Error("QXE core section header is truncated")
        section_id, codec_id, reserved, raw_len, coded_len, raw_sha, crc = QXE_SECTION.unpack_from(packet, offset)
        offset += QXE_SECTION.size
        end = offset + coded_len
        if section_id != expected_id or codec_id not in CODEC_NAMES or reserved or end > len(packet):
            raise QX2Error("QXE core section envelope drifted")
        coded = packet[offset:end]
        raw = decompress(CODEC_NAMES[codec_id], coded)
        if len(raw) != raw_len or sha256_bytes(raw) != raw_sha.hex() or zlib.crc32(coded) & 0xFFFFFFFF != crc:
            raise QX2Error("QXE core section integrity failed")
        records.append(packet[start:end])
        offset = end
    if offset != len(packet):
        raise QX2Error("QXE core packet has trailing bytes")
    return version, flags, records


def build_complete_envelope(core: bytes, raw: bytes, coded: bytes, codec: str) -> bytes:
    version, flags, records = split_qxe_records(core)
    section = QXE_SECTION.pack(
        8,
        CODECS[codec],
        0,
        len(raw),
        len(coded),
        bytes.fromhex(sha256_bytes(raw)),
        zlib.crc32(coded) & 0xFFFFFFFF,
    ) + coded
    return QXE_HEADER.pack(b"QXE1", version, flags, 8) + b"".join(records) + section


def verify_complete_envelope(packet: bytes, expected_raw: bytes) -> None:
    if len(packet) < QXE_HEADER.size or QXE_HEADER.unpack_from(packet) != (b"QXE1", 1, 0, 8):
        raise QX2Error("complete QXE identity failed")
    offset = QXE_HEADER.size
    decoded_last: bytes | None = None
    for expected_id in range(1, 9):
        section_id, codec_id, reserved, raw_len, coded_len, raw_sha, crc = QXE_SECTION.unpack_from(packet, offset)
        offset += QXE_SECTION.size
        coded = packet[offset : offset + coded_len]
        offset += coded_len
        if section_id != expected_id or reserved or codec_id not in CODEC_NAMES:
            raise QX2Error("complete QXE section roster failed")
        raw = decompress(CODEC_NAMES[codec_id], coded)
        if len(raw) != raw_len or sha256_bytes(raw) != raw_sha.hex() or zlib.crc32(coded) & 0xFFFFFFFF != crc:
            raise QX2Error("complete QXE section integrity failed")
        decoded_last = raw
    if offset != len(packet) or decoded_last != expected_raw:
        raise QX2Error("complete QXE parse-back did not consume the exact candidate")


def build_zip(packet: bytes) -> bytes:
    from io import BytesIO

    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        info = zipfile.ZipInfo("state/qx1.qxe", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, packet)
    return output.getvalue()


def retain_form(
    store: Path,
    name: str,
    raw: bytes,
    decoder: Any,
    baseline_path: Path,
    expected_events: tuple[PartitionEvent, ...],
    core: bytes,
) -> dict[str, Any]:
    root = store / "retained/candidates" / name
    raw_path = root / "raw.bin"
    atomic_bytes(raw_path, raw)
    if tuple(decoder(raw, baseline_path)) != expected_events:
        raise QX2Error(f"{name}: raw receiver did not reconstruct the source event object")
    coder_rows = []
    for codec in CODECS:
        coded = compress(codec, raw)
        repeat = compress(codec, raw)
        if coded != repeat or decompress(codec, coded) != raw:
            raise QX2Error(f"{name}/{codec}: coder repeat or parse-back failed")
        coded_path = root / f"candidate.{codec}.bin"
        repeat_path = root / f"candidate.{codec}.repeat.bin"
        atomic_bytes(coded_path, coded)
        atomic_bytes(repeat_path, repeat)
        coder_rows.append(
            {
                "codec": codec,
                "payload": fact(coded_path),
                "repeat": fact(repeat_path),
                "deterministic_repeat": True,
                "parseback_exact": True,
            }
        )
    winner = min(coder_rows, key=lambda row: (row["payload"]["bytes"], row["codec"]))
    winner_coded = Path(winner["payload"]["path"]).read_bytes()
    envelope = build_complete_envelope(core, raw, winner_coded, winner["codec"])
    envelope_repeat = build_complete_envelope(core, raw, winner_coded, winner["codec"])
    verify_complete_envelope(envelope, raw)
    archive = build_zip(envelope)
    archive_repeat = build_zip(envelope_repeat)
    if envelope != envelope_repeat or archive != archive_repeat:
        raise QX2Error(f"{name}: envelope repeat is not byte-identical")
    envelope_path = root / "complete.qxe"
    envelope_repeat_path = root / "complete.repeat.qxe"
    archive_path = root / "archive.zip"
    archive_repeat_path = root / "archive.repeat.zip"
    atomic_bytes(envelope_path, envelope)
    atomic_bytes(envelope_repeat_path, envelope_repeat)
    atomic_bytes(archive_path, archive)
    atomic_bytes(archive_repeat_path, archive_repeat)
    if len(archive) != QX1_CORE_ARCHIVE_BYTES + QXE_SECTION.size + len(winner_coded):
        raise QX2Error(f"{name}: exact archive increment is not section-header plus payload")
    return {
        "candidate_id": name,
        "raw": fact(raw_path),
        "coders": coder_rows,
        "winner_codec": winner["codec"],
        "winner_payload": winner["payload"],
        "complete_packet": fact(envelope_path),
        "complete_packet_repeat": fact(envelope_repeat_path),
        "archive": fact(archive_path),
        "archive_repeat": fact(archive_repeat_path),
        "exact_event_reconstruction": True,
        "event_count": EVENT_COUNT,
        "explicit_pair_row_col_tuples_in_payload": False,
        "core_margin_bytes": GATE_BYTES_EXCLUSIVE - QX1_CORE_ARCHIVE_BYTES,
        "incremental_archive_bytes": len(archive) - QX1_CORE_ARCHIVE_BYTES,
        "delta_bytes_vs_strict_gate": len(archive) - (GATE_BYTES_EXCLUSIVE - 1),
        "archive_clears_strict_gate": len(archive) < GATE_BYTES_EXCLUSIVE,
    }


def run(store: Path) -> dict[str, Any]:
    started = time.perf_counter()
    seed, by_pair, targets, preflight = load_inputs(store)
    baseline_path, characterization = build_baseline_and_stats(store, by_pair, targets)
    expected_events = tuple(seed.events)
    baseline_sha = sha256_file(baseline_path)
    core = QX1_CORE_PACKET.read_bytes()
    candidates = []
    for radius in (0, 1, 2, 4):
        name = f"boundary_field_r{radius}"
        raw = encode_boundary_field(baseline_path, by_pair, radius, baseline_sha)
        candidates.append(
            retain_form(
                store,
                name,
                raw,
                decode_boundary_field,
                baseline_path,
                expected_events,
                core,
            )
        )
        atomic_json(
            store / "checkpoints/STAGE2_BUILD_PROGRESS.json",
            {"schema": "ddm_qx2_build_progress.v1", "complete_candidate_ids": [row["candidate_id"] for row in candidates]},
        )
    enumerative_raw = encode_boundary_enumerative(baseline_path, by_pair, baseline_sha)
    enumerative = retain_form(
        store,
        "boundary_enumerative_r0",
        enumerative_raw,
        decode_boundary_enumerative,
        baseline_path,
        expected_events,
        core,
    )
    (
        _,
        _,
        _,
        _,
        _,
        near_count,
        residual_count,
        rank_bits,
        _,
    ) = ENUM_HEADER.unpack_from(enumerative_raw)
    counts_len, ranks_len, residual_len = struct.unpack_from(">III", enumerative_raw, ENUM_HEADER.size)
    enumerative["representation_anatomy"] = {
        "near_events": near_count,
        "far_residual_events": residual_count,
        "enumerative_rank_bits": rank_bits,
        "enumerative_rank_bytes": ranks_len,
        "transition_count_stream_bytes": counts_len,
        "far_residual_stream_bytes": residual_len,
        "outer_header_and_lengths_bytes": ENUM_HEADER.size + 12,
    }
    candidates.append(enumerative)
    rank_raw = encode_distance_rank(baseline_path, by_pair, baseline_sha)
    candidates.append(
        retain_form(
            store,
            "distance_rank",
            rank_raw,
            decode_distance_rank,
            baseline_path,
            expected_events,
            core,
        )
    )
    best_boundary = min(
        (row for row in candidates if row["candidate_id"].startswith("boundary_")),
        key=lambda row: (row["archive"]["bytes"], row["candidate_id"]),
    )
    rank = next(row for row in candidates if row["candidate_id"] == "distance_rank")
    priced_forms = [best_boundary, rank]
    best = min(priced_forms, key=lambda row: (row["archive"]["bytes"], row["candidate_id"]))
    qx1_joint_receiver_exists = False
    verdict = (
        "ENVELOPE-CLEARED"
        if best["archive_clears_strict_gate"] and qx1_joint_receiver_exists
        else "OVER"
    )
    result = {
        "schema": "ddm_qx2_events_section_redesign.v1",
        "complete": True,
        "verdict": verdict,
        "verdict_scope": (
            "FORMULATION: retained S2/C1 event population conditioned on its exact decoded C1 baseline; "
            "not a QX1 receiver, distortion, score, or family theorem"
        ),
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "preflight": preflight,
        "characterization": characterization,
        "candidate_configurations": candidates,
        "priced_forms": [row["candidate_id"] for row in priced_forms],
        "best": best,
        "conditional_byte_gate_cleared": best["archive_clears_strict_gate"],
        "gate": {
            "strict_archive_bytes_lt": GATE_BYTES_EXCLUSIVE,
            "core_archive_bytes": QX1_CORE_ARCHIVE_BYTES,
            "core_margin_bytes": GATE_BYTES_EXCLUSIVE - QX1_CORE_ARCHIVE_BYTES,
            "section_header_bytes": QXE_SECTION.size,
            "maximum_section_payload_bytes": GATE_BYTES_EXCLUSIVE - 1 - QX1_CORE_ARCHIVE_BYTES - QXE_SECTION.size,
        },
        "authority_boundaries": {
            "scorers_loaded": 0,
            "contest_eval_invocations": 0,
            "metal_invocations": 0,
            "modal_invocations": 0,
            "qx1_joint_receiver_exists": qx1_joint_receiver_exists,
            "qx1_pose_cap_measured": False,
            "conditional_baseline_is_receiver_custody_not_counted_in_section": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "RESULT.json", result)
    manifest = {
        "schema": "ddm_qx2_run_manifest.v1",
        "complete": True,
        "result": fact(store / "RESULT.json"),
        "baseline": fact(baseline_path),
        "retention": "all raw forms, coder candidates, repeats, QXE packets, and ZIP envelopes retained",
        "cleanup": "none fired",
        "command": f"{sys.executable} {Path(__file__).resolve()} --resume-from {store}",
        "source": fact(Path(__file__).resolve()),
    }
    atomic_json(store / "RUN_MANIFEST.json", manifest)
    atomic_json(
        store / "checkpoints/STAGE3_COMPLETE.json",
        {"schema": "ddm_qx2_complete.v1", "complete": True, "verdict": verdict, "result": fact(store / "RESULT.json")},
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, default=STORE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.resume_from)
    print(json.dumps({"verdict": result["verdict"], "best": result["best"]}, sort_keys=True))


if __name__ == "__main__":
    main()
