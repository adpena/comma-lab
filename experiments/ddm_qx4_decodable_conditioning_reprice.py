#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reprice QX2's exact event object on QX3's receiver-produced field.

QX2's winning enumerative section was conditioned on an encoder-only C1
partition.  QX4 preserves every original ``PartitionEvent`` tuple but derives
all candidate sets and rank orders from the approximate QBT field produced by
the counted QX1 core.  The distinction matters: an event's original baseline
class is content, while the QBT class is receiver context.  They disagree at
9,619 of the 17,926 event sites, and 9,177 events are no-ops relative to QBT.

The primary form is QX2's boundary-transition enumerative subset.  If its best
real-coder payload exceeds the 24,093-byte section cap, QX4 also prices QX2's
four boundary-bitmap radii and decoded-distance-rank sibling.  Every raw form,
real-coder payload, deterministic repeat, complete packet/archive, decoded
event object, and receiver-applied partition field is retained under AP
custody.  This is scorer-free rate/receiver evidence, never a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import sys
import time
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tac.optimization.s2_partition_seed import PartitionEvent, decode_partition_seed

STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx4")
QX1_STORE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qx1")
QX2_STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx2")
QX3_STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx3")
QX1_CORE: Final = QX1_STORE / "retained/envelopes/core_without_events_exceptions/envelope.qxe"
EVENT_SOURCE: Final = (
    QX1_STORE / "retained/sections/08_events_exceptions_explicit_address_control/raw.bin"
)
QX2_RESULT: Final = QX2_STORE / "RESULT.json"
QX3_RESULT: Final = QX3_STORE / "RESULT.json"
QX3_FIELD: Final = QX3_STORE / "retained/derived/qx1_decoder_baseline.u8"
QX2_RUNNER: Final = REPO / "experiments/ddm_qx2_events_section_redesign.py"
QX3_RUNNER: Final = REPO / "experiments/ddm_qx3_receiver_closure.py"
QX2_MEMO: Final = REPO / ".omx/research/ddm_qx2_events_section_redesign_20260831.md"
QX3_MEMO: Final = REPO / ".omx/research/ddm_qx3_receiver_closure_20260831.md"

PINS: Final = {
    QX1_CORE: "4e6a2f6669c590258fc6c5d194ae6cb30951f5881e2055761de0bff753bdfb95",
    EVENT_SOURCE: "df4c0534537a9919681509a0b44a392d7d4b46c812d7570c534e6b823adae7fc",
    QX2_RESULT: "b3a63070260ca4d8d6ea23ec7395bb3156b2cbdae91c1a27bca2e0d82b63e234",
    QX3_RESULT: "f9a71967ec01aa8905aeb31806f29ebb40a8c9729d0db9c58d36a02a540d7867",
    QX3_FIELD: "afeb8c94d5181b03992aefad1daef49ee7aaf1f768d11aa5964dacbfa1e22dbd",
    QX2_RUNNER: "88457037f5cbc272b494306a1613f8c6e2abe3499fdf83164274e3db76b1311c",
    QX3_RUNNER: "238560265a040d942429c7e63e5fe78617a1719d67308e6fada0fbc994a6e272",
    QX2_MEMO: "3bcd01d2c46a5b68bfa931ddd4be81e2250a06b36ebbdf6f50ba671f3b9a3022",
    QX3_MEMO: "ac893d741fc34f70c74bf6f0fbe936b0d01c8e9d5e4484720f1df7a143369136",
}

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
SITES: Final = N_PAIRS * HEIGHT * WIDTH
EVENTS: Final = 17_926
SECTION_CAP_BYTES: Final = 24_093
ARCHIVE_GATE_EXCLUSIVE: Final = 137_986
QX1_CORE_ARCHIVE_BYTES: Final = 113_844
QXE_SECTION_BYTES: Final = 48
MINIMUM_FREE_BYTES: Final = 1_000_000_000
AXIS: Final = "[scorer-free exact rate and receiver-conditioned parse-back measurement]"

COMMON_HEADER: Final = struct.Struct(">4sBBHII32s32s")
ENUM_META: Final = struct.Struct(">IIIIIII")
BITMAP_META: Final = struct.Struct(">B3xIIIIIII")
RANK_META: Final = struct.Struct(">I")
FORM_ORDER: Final = (
    "boundary_enumerative_r0",
    "boundary_field_r0",
    "boundary_field_r1",
    "boundary_field_r2",
    "boundary_field_r4",
    "distance_rank",
)
FORM_IDS: Final = {name: index + 1 for index, name in enumerate(FORM_ORDER)}
FORM_NAMES: Final = {value: key for key, value in FORM_IDS.items()}
MAGICS: Final = {
    "boundary_enumerative_r0": b"Q4E1",
    "boundary_field_r0": b"Q4B1",
    "boundary_field_r1": b"Q4B1",
    "boundary_field_r2": b"Q4B1",
    "boundary_field_r4": b"Q4B1",
    "distance_rank": b"Q4R1",
}


class QX4Error(RuntimeError):
    """A custody, exactness, framing, or retention gate failed closed."""


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
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def require_fact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise QX4Error(f"{label} custody drifted: {path}")
    return fact(path)


def verify_recorded_fact(row: dict[str, Any], label: str) -> None:
    path = Path(row["path"])
    if not path.is_file() or fact(path) != row:
        raise QX4Error(f"{label} retained fact drifted: {path}")


def common_header(form: str, conditioning_sha: str) -> bytes:
    return COMMON_HEADER.pack(
        MAGICS[form],
        1,
        FORM_IDS[form],
        0,
        N_PAIRS,
        EVENTS,
        bytes.fromhex(conditioning_sha),
        bytes.fromhex(PINS[EVENT_SOURCE]),
    )


def parse_common(payload: bytes, conditioning_path: Path) -> tuple[str, int]:
    if len(payload) < COMMON_HEADER.size:
        raise QX4Error("QX4 event section is truncated")
    magic, version, form_id, flags, pairs, events, conditioning_sha, source_sha = (
        COMMON_HEADER.unpack_from(payload)
    )
    if version != 1 or flags or pairs != N_PAIRS or events != EVENTS or form_id not in FORM_NAMES:
        raise QX4Error("QX4 event section identity drifted")
    form = FORM_NAMES[form_id]
    if magic != MAGICS[form]:
        raise QX4Error("QX4 event section magic/form mismatch")
    if conditioning_sha.hex() != sha256_file(conditioning_path):
        raise QX4Error("QX4 conditioning field identity drifted")
    if source_sha.hex() != PINS[EVENT_SOURCE]:
        raise QX4Error("QX4 source-event identity drifted")
    return form, COMMON_HEADER.size


def full_transition_groups() -> tuple[tuple[int, int], ...]:
    """All QBT-context/target pairs, including QBT-relative no-ops."""

    return tuple((context, target) for context in range(5) for target in range(5))


def field_frame(path: Path, pair: int) -> np.ndarray:
    return np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=pair * HEIGHT * WIDTH,
        shape=(HEIGHT, WIDTH),
    )


def split_events(events: Sequence[PartitionEvent]) -> list[list[PartitionEvent]]:
    by_pair: list[list[PartitionEvent]] = [[] for _ in range(N_PAIRS)]
    for event in events:
        by_pair[event.pair].append(event)
    return by_pair


def encode_enumerative(
    qx2: Any,
    conditioning_path: Path,
    by_pair: list[list[PartitionEvent]],
    conditioning_sha: str,
) -> tuple[bytes, dict[str, Any]]:
    groups = full_transition_groups()
    counts = bytearray()
    ranks = qx2.BitWriter()
    near_bases: list[int] = []
    residual = bytearray()
    near_count = 0
    residual_count = 0
    for pair, events in enumerate(by_pair):
        conditioning = np.asarray(field_frame(conditioning_path, pair))
        boundary, distance = qx2.boundary_and_distance(conditioning)
        event_at = {event.row * WIDTH + event.col: event for event in events}
        by_transition: dict[tuple[int, int], list[int]] = defaultdict(list)
        far: list[PartitionEvent] = []
        for event in events:
            index = event.row * WIDTH + event.col
            if boundary[event.row, event.col]:
                context = int(conditioning[event.row, event.col])
                by_transition[(context, event.target_class)].append(index)
                near_count += 1
            else:
                far.append(event)
        mask = sum(1 << index for index, group in enumerate(groups) if by_transition[group])
        counts.extend(mask.to_bytes(4, "little"))
        for group in groups:
            if by_transition[group]:
                qx2.write_uleb(len(by_transition[group]), counts)
        flat_boundary = boundary.reshape(-1)
        flat_conditioning = conditioning.reshape(-1)
        for context in range(5):
            available = np.flatnonzero(flat_boundary & (flat_conditioning == context)).tolist()
            for target in range(5):
                selected_indices = sorted(by_transition[(context, target)])
                if not selected_indices:
                    continue
                positions = {index: position for position, index in enumerate(available)}
                try:
                    selected = [positions[index] for index in selected_indices]
                except KeyError as error:
                    raise QX4Error("event is absent from its QBT-derived candidate field") from error
                total = math.comb(len(available), len(selected))
                ranks.write(qx2.combination_rank(selected), (total - 1).bit_length())
                near_bases.extend(event_at[index].baseline_class for index in selected_indices)
                selected_set = set(selected_indices)
                available = [index for index in available if index not in selected_set]
        qx2.write_uleb(len(far), residual)
        _, inverse = qx2.distance_order(distance)
        previous_rank = -1
        for rank, event in sorted(
            (int(inverse[event.row * WIDTH + event.col]), event) for event in far
        ):
            qx2.write_uleb(rank - previous_rank - 1, residual)
            residual.append(event.target_class | (event.baseline_class << 3))
            previous_rank = rank
            residual_count += 1
    rank_payload = ranks.finish()
    base_payload = qx2.pack_three_bit(near_bases)
    meta = ENUM_META.pack(
        near_count,
        residual_count,
        ranks.bits,
        len(counts),
        len(rank_payload),
        len(base_payload),
        len(residual),
    )
    raw = (
        common_header("boundary_enumerative_r0", conditioning_sha)
        + meta
        + bytes(counts)
        + rank_payload
        + base_payload
        + bytes(residual)
    )
    anatomy = {
        "near_events": near_count,
        "far_residual_events": residual_count,
        "enumerative_rank_bits": ranks.bits,
        "enumerative_rank_bytes": len(rank_payload),
        "transition_count_stream_bytes": len(counts),
        "near_original_baseline_stream_bytes": len(base_payload),
        "far_residual_stream_bytes": len(residual),
        "outer_header_and_meta_bytes": COMMON_HEADER.size + ENUM_META.size,
    }
    return raw, anatomy


def decode_enumerative(qx2: Any, payload: bytes, conditioning_path: Path) -> tuple[PartitionEvent, ...]:
    form, offset = parse_common(payload, conditioning_path)
    if form != "boundary_enumerative_r0" or offset + ENUM_META.size > len(payload):
        raise QX4Error("enumerative QX4 section identity/length drifted")
    near_count, residual_count, rank_bits, counts_len, ranks_len, bases_len, residual_len = (
        ENUM_META.unpack_from(payload, offset)
    )
    offset += ENUM_META.size
    if offset + counts_len + ranks_len + bases_len + residual_len != len(payload):
        raise QX4Error("enumerative QX4 section lengths do not close")
    counts_payload = payload[offset : offset + counts_len]
    offset += counts_len
    rank_reader = qx2.BitReader(payload[offset : offset + ranks_len], rank_bits)
    offset += ranks_len
    bases = qx2.unpack_three_bit(payload[offset : offset + bases_len], near_count)
    offset += bases_len
    residual_payload = payload[offset:]
    groups = full_transition_groups()
    counts_offset = 0
    base_offset = 0
    residual_offset = 0
    decoded_residual = 0
    events: list[PartitionEvent] = []
    for pair in range(N_PAIRS):
        conditioning = np.asarray(field_frame(conditioning_path, pair))
        boundary, distance = qx2.boundary_and_distance(conditioning)
        if counts_offset + 4 > len(counts_payload):
            raise QX4Error("enumerative transition mask is truncated")
        mask = int.from_bytes(counts_payload[counts_offset : counts_offset + 4], "little")
        counts_offset += 4
        if mask >> len(groups):
            raise QX4Error("enumerative transition mask has reserved bits")
        group_counts: dict[tuple[int, int], int] = {}
        for index, group in enumerate(groups):
            if mask & (1 << index):
                count, counts_offset = qx2.read_uleb(counts_payload, counts_offset)
                if not count:
                    raise QX4Error("enumerative transition mask names an empty group")
                group_counts[group] = count
            else:
                group_counts[group] = 0
        flat_boundary = boundary.reshape(-1)
        flat_conditioning = conditioning.reshape(-1)
        for context in range(5):
            available = np.flatnonzero(flat_boundary & (flat_conditioning == context)).tolist()
            for target in range(5):
                count = group_counts[(context, target)]
                if not count:
                    continue
                if count > len(available):
                    raise QX4Error("enumerative count exceeds its QBT candidate set")
                total = math.comb(len(available), count)
                rank = rank_reader.read((total - 1).bit_length())
                selected = qx2.combination_unrank(rank, len(available), count)
                selected_indices = [available[position] for position in selected]
                for index in selected_indices:
                    if base_offset >= len(bases):
                        raise QX4Error("enumerative original-baseline stream is truncated")
                    source_base = bases[base_offset]
                    base_offset += 1
                    if source_base == target:
                        raise QX4Error("decoded source event is a no-op in its original semantics")
                    row, col = divmod(index, WIDTH)
                    events.append(PartitionEvent(pair, row, col, target, source_base))
                selected_set = set(selected_indices)
                available = [index for index in available if index not in selected_set]
        count, residual_offset = qx2.read_uleb(residual_payload, residual_offset)
        order, _ = qx2.distance_order(distance)
        previous_rank = -1
        for _ in range(count):
            delta, residual_offset = qx2.read_uleb(residual_payload, residual_offset)
            if residual_offset >= len(residual_payload):
                raise QX4Error("enumerative residual transition is truncated")
            rank = previous_rank + delta + 1
            packed = residual_payload[residual_offset]
            residual_offset += 1
            if rank >= order.size or packed & 0xC0:
                raise QX4Error("enumerative residual value is invalid")
            index = int(order[rank])
            row, col = divmod(index, WIDTH)
            target = packed & 0x07
            source_base = (packed >> 3) & 0x07
            if target >= 5 or source_base >= 5 or target == source_base or boundary[row, col]:
                raise QX4Error("enumerative residual violates source-event or rank semantics")
            events.append(PartitionEvent(pair, row, col, target, source_base))
            previous_rank = rank
            decoded_residual += 1
    rank_reader.finish()
    if (
        counts_offset != len(counts_payload)
        or base_offset != near_count
        or residual_offset != len(residual_payload)
        or decoded_residual != residual_count
    ):
        raise QX4Error("enumerative QX4 receiver did not consume all streams")
    return tuple(sorted(events))


def encode_bitmap(
    qx2: Any,
    form: str,
    radius: int,
    conditioning_path: Path,
    by_pair: list[list[PartitionEvent]],
    conditioning_sha: str,
) -> tuple[bytes, dict[str, Any]]:
    occupancy_chunks: list[np.ndarray] = []
    near_targets: list[int] = []
    near_bases: list[int] = []
    residual = bytearray()
    candidate_total = 0
    residual_count = 0
    for pair, events in enumerate(by_pair):
        conditioning = np.asarray(field_frame(conditioning_path, pair))
        _, distance = qx2.boundary_and_distance(conditioning)
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
            event = event_map[int(index)]
            near_targets.append(event.target_class)
            near_bases.append(event.baseline_class)
        _, inverse = qx2.distance_order(distance)
        far = [event for event in events if not candidate[event.row, event.col]]
        qx2.write_uleb(len(far), residual)
        previous_rank = -1
        for rank, event in sorted(
            (int(inverse[event.row * WIDTH + event.col]), event) for event in far
        ):
            qx2.write_uleb(rank - previous_rank - 1, residual)
            residual.append(event.target_class | (event.baseline_class << 3))
            previous_rank = rank
            residual_count += 1
    occupancy = np.concatenate(occupancy_chunks) if occupancy_chunks else np.empty(0, dtype=np.uint8)
    occupancy_payload = np.packbits(occupancy, bitorder="little").tobytes()
    target_payload = qx2.pack_three_bit(near_targets)
    base_payload = qx2.pack_three_bit(near_bases)
    meta = BITMAP_META.pack(
        radius,
        candidate_total,
        len(near_targets),
        residual_count,
        len(occupancy_payload),
        len(target_payload),
        len(base_payload),
        len(residual),
    )
    raw = (
        common_header(form, conditioning_sha)
        + meta
        + occupancy_payload
        + target_payload
        + base_payload
        + bytes(residual)
    )
    anatomy = {
        "radius": radius,
        "candidate_sites": candidate_total,
        "near_events": len(near_targets),
        "far_residual_events": residual_count,
        "occupancy_bytes": len(occupancy_payload),
        "near_target_bytes": len(target_payload),
        "near_original_baseline_bytes": len(base_payload),
        "far_residual_stream_bytes": len(residual),
        "outer_header_and_meta_bytes": COMMON_HEADER.size + BITMAP_META.size,
    }
    return raw, anatomy


def decode_bitmap(qx2: Any, payload: bytes, conditioning_path: Path) -> tuple[PartitionEvent, ...]:
    form, offset = parse_common(payload, conditioning_path)
    if not form.startswith("boundary_field_r") or offset + BITMAP_META.size > len(payload):
        raise QX4Error("boundary-bitmap QX4 identity/length drifted")
    radius, candidate_total, near_count, residual_count, occ_len, target_len, base_len, residual_len = (
        BITMAP_META.unpack_from(payload, offset)
    )
    if form != f"boundary_field_r{radius}":
        raise QX4Error("boundary-bitmap radius/form mismatch")
    offset += BITMAP_META.size
    if offset + occ_len + target_len + base_len + residual_len != len(payload):
        raise QX4Error("boundary-bitmap QX4 section lengths do not close")
    occ_payload = payload[offset : offset + occ_len]
    offset += occ_len
    targets = qx2.unpack_three_bit(payload[offset : offset + target_len], near_count)
    offset += target_len
    bases = qx2.unpack_three_bit(payload[offset : offset + base_len], near_count)
    offset += base_len
    residual_payload = payload[offset:]
    occupancy = np.unpackbits(np.frombuffer(occ_payload, dtype=np.uint8), bitorder="little")
    if occupancy.size < candidate_total or np.any(occupancy[candidate_total:]):
        raise QX4Error("boundary-bitmap occupancy padding is noncanonical")
    occupancy = occupancy[:candidate_total]
    occupancy_offset = 0
    near_offset = 0
    residual_offset = 0
    decoded_residual = 0
    events: list[PartitionEvent] = []
    for pair in range(N_PAIRS):
        conditioning = np.asarray(field_frame(conditioning_path, pair))
        _, distance = qx2.boundary_and_distance(conditioning)
        candidate_indices = np.flatnonzero((distance <= radius).reshape(-1))
        selected = occupancy[occupancy_offset : occupancy_offset + candidate_indices.size]
        occupancy_offset += int(candidate_indices.size)
        for index in candidate_indices[selected != 0].tolist():
            if near_offset >= near_count:
                raise QX4Error("boundary-bitmap near streams are truncated")
            target = targets[near_offset]
            source_base = bases[near_offset]
            near_offset += 1
            if target == source_base:
                raise QX4Error("decoded source event is a no-op in its original semantics")
            row, col = divmod(int(index), WIDTH)
            events.append(PartitionEvent(pair, row, col, target, source_base))
        count, residual_offset = qx2.read_uleb(residual_payload, residual_offset)
        order, _ = qx2.distance_order(distance)
        previous_rank = -1
        for _ in range(count):
            delta, residual_offset = qx2.read_uleb(residual_payload, residual_offset)
            if residual_offset >= len(residual_payload):
                raise QX4Error("boundary-bitmap residual is truncated")
            rank = previous_rank + delta + 1
            packed = residual_payload[residual_offset]
            residual_offset += 1
            if rank >= order.size or packed & 0xC0:
                raise QX4Error("boundary-bitmap residual value is invalid")
            index = int(order[rank])
            row, col = divmod(index, WIDTH)
            target = packed & 0x07
            source_base = (packed >> 3) & 0x07
            if (
                target >= 5
                or source_base >= 5
                or target == source_base
                or distance[row, col] <= radius
            ):
                raise QX4Error("boundary-bitmap residual violates source-event or rank semantics")
            events.append(PartitionEvent(pair, row, col, target, source_base))
            previous_rank = rank
            decoded_residual += 1
    if (
        occupancy_offset != candidate_total
        or near_offset != near_count
        or residual_offset != len(residual_payload)
        or decoded_residual != residual_count
    ):
        raise QX4Error("boundary-bitmap QX4 receiver did not consume all streams")
    return tuple(sorted(events))


def encode_rank(
    qx2: Any,
    conditioning_path: Path,
    by_pair: list[list[PartitionEvent]],
    conditioning_sha: str,
) -> tuple[bytes, dict[str, Any]]:
    body = bytearray()
    for pair, events in enumerate(by_pair):
        conditioning = np.asarray(field_frame(conditioning_path, pair))
        _, distance = qx2.boundary_and_distance(conditioning)
        _, inverse = qx2.distance_order(distance)
        ranked = sorted((int(inverse[event.row * WIDTH + event.col]), event) for event in events)
        qx2.write_uleb(len(ranked), body)
        previous_rank = -1
        for rank, event in ranked:
            qx2.write_uleb(rank - previous_rank - 1, body)
            body.append(event.target_class | (event.baseline_class << 3))
            previous_rank = rank
    raw = common_header("distance_rank", conditioning_sha) + RANK_META.pack(len(body)) + bytes(body)
    return raw, {
        "rank_stream_bytes": len(body),
        "outer_header_and_meta_bytes": COMMON_HEADER.size + RANK_META.size,
    }


def decode_rank(qx2: Any, payload: bytes, conditioning_path: Path) -> tuple[PartitionEvent, ...]:
    form, offset = parse_common(payload, conditioning_path)
    if form != "distance_rank" or offset + RANK_META.size > len(payload):
        raise QX4Error("distance-rank QX4 identity/length drifted")
    (body_len,) = RANK_META.unpack_from(payload, offset)
    offset += RANK_META.size
    if offset + body_len != len(payload):
        raise QX4Error("distance-rank QX4 body length drifted")
    body = payload[offset:]
    body_offset = 0
    events: list[PartitionEvent] = []
    for pair in range(N_PAIRS):
        conditioning = np.asarray(field_frame(conditioning_path, pair))
        _, distance = qx2.boundary_and_distance(conditioning)
        order, _ = qx2.distance_order(distance)
        count, body_offset = qx2.read_uleb(body, body_offset)
        previous_rank = -1
        for _ in range(count):
            delta, body_offset = qx2.read_uleb(body, body_offset)
            if body_offset >= len(body):
                raise QX4Error("distance-rank transition is truncated")
            rank = previous_rank + delta + 1
            packed = body[body_offset]
            body_offset += 1
            if rank >= order.size or packed & 0xC0:
                raise QX4Error("distance-rank value is invalid")
            index = int(order[rank])
            row, col = divmod(index, WIDTH)
            target = packed & 0x07
            source_base = (packed >> 3) & 0x07
            if target >= 5 or source_base >= 5 or target == source_base:
                raise QX4Error("distance-rank source-event semantics drifted")
            events.append(PartitionEvent(pair, row, col, target, source_base))
            previous_rank = rank
    if body_offset != len(body):
        raise QX4Error("distance-rank QX4 stream has trailing bytes")
    return tuple(sorted(events))


def encode_form(
    qx2: Any,
    form: str,
    conditioning_path: Path,
    by_pair: list[list[PartitionEvent]],
    conditioning_sha: str,
) -> tuple[bytes, dict[str, Any]]:
    if form == "boundary_enumerative_r0":
        return encode_enumerative(qx2, conditioning_path, by_pair, conditioning_sha)
    if form.startswith("boundary_field_r"):
        return encode_bitmap(
            qx2,
            form,
            int(form.rsplit("r", 1)[1]),
            conditioning_path,
            by_pair,
            conditioning_sha,
        )
    if form == "distance_rank":
        return encode_rank(qx2, conditioning_path, by_pair, conditioning_sha)
    raise QX4Error(f"unknown QX4 form: {form}")


def decode_form(qx2: Any, payload: bytes, conditioning_path: Path) -> tuple[PartitionEvent, ...]:
    form, _ = parse_common(payload, conditioning_path)
    if form == "boundary_enumerative_r0":
        return decode_enumerative(qx2, payload, conditioning_path)
    if form.startswith("boundary_field_r"):
        return decode_bitmap(qx2, payload, conditioning_path)
    if form == "distance_rank":
        return decode_rank(qx2, payload, conditioning_path)
    raise QX4Error(f"unknown QX4 form: {form}")


def validate_complete_receipt(
    receipt: dict[str, Any], qx2: Any, conditioning_path: Path, expected: tuple[PartitionEvent, ...]
) -> None:
    verify_recorded_fact(receipt["raw"], "candidate raw")
    verify_recorded_fact(receipt["conditioning_field"], "candidate conditioning field")
    raw = Path(receipt["raw"]["path"]).read_bytes()
    for row in receipt["coders"]:
        verify_recorded_fact(row["payload"], "candidate coded payload")
        verify_recorded_fact(row["repeat"], "candidate coded repeat")
        coded = Path(row["payload"]["path"]).read_bytes()
        repeat = Path(row["repeat"]["path"]).read_bytes()
        if coded != repeat or qx2.decompress(row["codec"], coded) != raw:
            raise QX4Error(f"{receipt['candidate_id']}/{row['codec']}: retained coder drifted")
    for key in ("complete_packet", "complete_packet_repeat", "archive", "archive_repeat"):
        verify_recorded_fact(receipt[key], key)
    packet = Path(receipt["complete_packet"]["path"]).read_bytes()
    packet_repeat = Path(receipt["complete_packet_repeat"]["path"]).read_bytes()
    archive = Path(receipt["archive"]["path"]).read_bytes()
    archive_repeat = Path(receipt["archive_repeat"]["path"]).read_bytes()
    if packet != packet_repeat or archive != archive_repeat:
        raise QX4Error(f"{receipt['candidate_id']}: retained packet/archive repeat drifted")
    qx2.verify_complete_envelope(packet, raw)
    with zipfile.ZipFile(BytesIO(archive), "r") as parsed:
        if parsed.namelist() != ["state/qx1.qxe"] or parsed.read("state/qx1.qxe") != packet:
            raise QX4Error(f"{receipt['candidate_id']}: retained archive parse-back drifted")
    if decode_form(qx2, raw, conditioning_path) != expected:
        raise QX4Error(f"{receipt['candidate_id']}: resumed raw no longer decodes exactly")


def retain_form(
    store: Path,
    qx2: Any,
    form: str,
    conditioning_path: Path,
    by_pair: list[list[PartitionEvent]],
    expected: tuple[PartitionEvent, ...],
    core: bytes,
) -> dict[str, Any]:
    root = store / "retained/candidates" / form
    receipt_path = root / "RECEIPT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_complete_receipt(receipt, qx2, conditioning_path, expected)
        return receipt
    raw, anatomy = encode_form(qx2, form, conditioning_path, by_pair, sha256_file(conditioning_path))
    raw_path = root / "raw.bin"
    atomic_bytes(raw_path, raw)
    if decode_form(qx2, raw, conditioning_path) != expected:
        raise QX4Error(f"{form}: raw receiver did not reconstruct all source events exactly")
    coder_rows: list[dict[str, Any]] = []
    for codec in qx2.CODECS:
        coded = qx2.compress(codec, raw)
        repeat = qx2.compress(codec, raw)
        if coded != repeat or qx2.decompress(codec, coded) != raw:
            raise QX4Error(f"{form}/{codec}: real-coder repeat or parse-back failed")
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
    packet = qx2.build_complete_envelope(core, raw, winner_coded, winner["codec"])
    packet_repeat = qx2.build_complete_envelope(core, raw, winner_coded, winner["codec"])
    qx2.verify_complete_envelope(packet, raw)
    archive = qx2.build_zip(packet)
    archive_repeat = qx2.build_zip(packet_repeat)
    if packet != packet_repeat or archive != archive_repeat:
        raise QX4Error(f"{form}: packet/archive determinism repeat failed")
    packet_path = root / "complete.qxe"
    packet_repeat_path = root / "complete.repeat.qxe"
    archive_path = root / "archive.zip"
    archive_repeat_path = root / "archive.repeat.zip"
    atomic_bytes(packet_path, packet)
    atomic_bytes(packet_repeat_path, packet_repeat)
    atomic_bytes(archive_path, archive)
    atomic_bytes(archive_repeat_path, archive_repeat)
    if len(archive) != QX1_CORE_ARCHIVE_BYTES + QXE_SECTION_BYTES + len(winner_coded):
        raise QX4Error(f"{form}: archive increment is not header plus selected payload")
    receipt = {
        "schema": "ddm_qx4_candidate_receipt.v1",
        "candidate_id": form,
        "raw": fact(raw_path),
        "representation_anatomy": anatomy,
        "coders": coder_rows,
        "winner_codec": winner["codec"],
        "winner_payload": winner["payload"],
        "complete_packet": fact(packet_path),
        "complete_packet_repeat": fact(packet_repeat_path),
        "archive": fact(archive_path),
        "archive_repeat": fact(archive_repeat_path),
        "event_count": EVENTS,
        "exact_original_event_tuple_reconstruction": True,
        "conditioning_field": fact(conditioning_path),
        "section_cap_bytes": SECTION_CAP_BYTES,
        "delta_bytes_vs_section_cap": winner["payload"]["bytes"] - SECTION_CAP_BYTES,
        "archive_clears_strict_gate": len(archive) < ARCHIVE_GATE_EXCLUSIVE,
        "delta_bytes_vs_strict_archive_gate": len(archive) - (ARCHIVE_GATE_EXCLUSIVE - 1),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def extract_raw_from_archive(qx3: Any, archive_bytes: bytes) -> tuple[bytes, bytes]:
    with zipfile.ZipFile(BytesIO(archive_bytes), "r") as archive:
        if archive.namelist() != ["state/qx1.qxe"]:
            raise QX4Error("QX4 archive member roster drifted")
        packet = archive.read("state/qx1.qxe")
    records, sections, _codecs = qx3.parse_qxe(packet, 8)
    core = qx3.QXE_HEADER.pack(b"QXE1", 1, 0, 7) + b"".join(records[:7])
    if sha256_bytes(core) != PINS[QX1_CORE]:
        raise QX4Error("QX4 archive core differs from the pinned QX1 core")
    return core, sections[8]


def apply_events(
    conditioning_path: Path,
    events: Sequence[PartitionEvent],
    output_path: Path,
) -> dict[str, Any]:
    by_pair = split_events(events)
    conditioning = np.memmap(
        conditioning_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.truncate(SITES)
    output = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(N_PAIRS, HEIGHT, WIDTH))
    context_equals_source = 0
    context_equals_target = 0
    context_is_third_class = 0
    changed_sites = 0
    for pair, pair_events in enumerate(by_pair):
        output[pair] = conditioning[pair]
        for event in pair_events:
            context = int(output[pair, event.row, event.col])
            context_equals_source += int(context == event.baseline_class)
            context_equals_target += int(context == event.target_class)
            context_is_third_class += int(context not in (event.baseline_class, event.target_class))
            changed_sites += int(context != event.target_class)
            output[pair, event.row, event.col] = event.target_class
        if (pair + 1) % 30 == 0:
            output.flush()
    output.flush()
    del output, conditioning
    os.replace(partial, output_path)
    return {
        "output": fact(output_path),
        "events_applied": len(events),
        "changed_sites": changed_sites,
        "context_equals_original_baseline": context_equals_source,
        "context_equals_target_noop": context_equals_target,
        "context_is_third_class": context_is_third_class,
        "application_rule": "overwrite each decoded event site with its retained target class",
    }


def receiver_proof(
    store: Path,
    qx2: Any,
    qx3: Any,
    selected: dict[str, Any],
    conditioning_path: Path,
    expected: tuple[PartitionEvent, ...],
) -> dict[str, Any]:
    stage_path = store / "checkpoints/STAGE2_RECEIVER_PROOF.json"
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        for row in (stage["primary"]["applied_field"]["output"], stage["repeat"]["applied_field"]["output"]):
            verify_recorded_fact(row, "receiver-applied field")
        return stage
    archive = Path(selected["archive"]["path"]).read_bytes()
    archive_repeat = Path(selected["archive_repeat"]["path"]).read_bytes()
    if archive != archive_repeat:
        raise QX4Error("selected candidate archive repeat drifted")
    passes: list[dict[str, Any]] = []
    for label, payload in (("primary", archive), ("repeat", archive_repeat)):
        core, raw = extract_raw_from_archive(qx3, payload)
        fresh_conditioning, derive_stage = qx3.derive_decoder_baseline(store, core)
        if fresh_conditioning != conditioning_path or sha256_file(fresh_conditioning) != PINS[QX3_FIELD]:
            raise QX4Error("archive receiver did not reproduce the bit-pinned QX3 conditioning field")
        decoded = decode_form(qx2, raw, fresh_conditioning)
        if decoded != expected:
            raise QX4Error("archive receiver did not reconstruct all original event tuples exactly")
        applied = apply_events(
            fresh_conditioning,
            decoded,
            store / f"retained/receiver/event_applied_partition.{label}.u8",
        )
        passes.append(
            {
                "label": label,
                "events_decoded": len(decoded),
                "exact_original_event_tuple_identity": True,
                "conditioning": fact(fresh_conditioning),
                "conditioning_derivation": derive_stage,
                "applied_field": applied,
                "receiver_inputs": [
                    "archive.zip",
                    "generic QX1/QBT decoder",
                    "generic QX4 event decoder",
                ],
                "encoder_only_inputs_used_by_receiver": [],
            }
        )
    if passes[0]["applied_field"]["output"]["sha256"] != passes[1]["applied_field"]["output"]["sha256"]:
        raise QX4Error("receiver-applied partition repeat is not bit-identical")
    stage = {
        "schema": "ddm_qx4_receiver_proof.v1",
        "complete": True,
        "primary": passes[0],
        "repeat": passes[1],
        "archive_repeat_byte_identical": True,
        "receiver_output_repeat_bit_identical": True,
    }
    atomic_json(stage_path, stage)
    return stage


def preflight(store: Path) -> tuple[dict[str, Any], Any, Any, tuple[PartitionEvent, ...]]:
    if store.resolve() != STORE.resolve():
        raise QX4Error(f"custody is pinned to {STORE}, not {store.resolve()}")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise QX4Error(f"storage preflight failed: {free} < {MINIMUM_FREE_BYTES}")
    inputs = {str(path): require_fact(path, digest, path.name) for path, digest in PINS.items()}
    qx2 = importlib.import_module("experiments.ddm_qx2_events_section_redesign")
    qx3 = importlib.import_module("experiments.ddm_qx3_receiver_closure")
    seed = decode_partition_seed(EVENT_SOURCE.read_bytes())
    expected = tuple(sorted(seed.events))
    if (seed.n_pairs, seed.height, seed.width, len(expected)) != (
        N_PAIRS,
        HEIGHT,
        WIDTH,
        EVENTS,
    ):
        raise QX4Error("retained source event geometry drifted")
    qx3_result = json.loads(QX3_RESULT.read_text(encoding="utf-8"))
    if (
        qx3_result["derived_baseline"]["derived_baseline"]["sha256"] != PINS[QX3_FIELD]
        or qx3_result["derived_baseline"]["fresh_decode_vs_retained_native_mismatches"] != 0
    ):
        raise QX4Error("QX3 receiver-produced conditioning proof drifted")
    stage = {
        "schema": "ddm_qx4_preflight.v1",
        "complete": True,
        "axis": AXIS,
        "selection_mode": "full n600",
        "storage": {"path": str(store), "observed_free_bytes": free},
        "inputs": inputs,
        "denominators": {"pairs": N_PAIRS, "sites": SITES, "events": EVENTS},
        "conditioning_contract": {
            "only_conditioning_source": fact(QX3_FIELD),
            "fresh_qx1_decode_vs_qbz1_native_field_mismatches": 0,
            "qx2_encoder_only_c1_baseline_consulted": False,
        },
        "scorers_loaded": 0,
        "contest_eval_invocations": 0,
        "modal_invocations": 0,
        "metal_invocations": 0,
    }
    atomic_json(store / "checkpoints/STAGE0_PREFLIGHT.json", stage)
    return stage, qx2, qx3, expected


def run(store: Path) -> dict[str, Any]:
    started = time.perf_counter()
    stage0, qx2, qx3, expected = preflight(store)
    core = QX1_CORE.read_bytes()
    conditioning_path, conditioning_stage = qx3.derive_decoder_baseline(store, core)
    if sha256_file(conditioning_path) != PINS[QX3_FIELD]:
        raise QX4Error("fresh receiver-produced conditioning field differs from QX3's bit pin")
    atomic_json(
        store / "checkpoints/STAGE0_CONDITIONING.json",
        {
            "schema": "ddm_qx4_conditioning.v1",
            "complete": True,
            "conditioning": fact(conditioning_path),
            "qx3_proven_decode": conditioning_stage,
        },
    )
    by_pair = split_events(expected)
    candidates = [
        retain_form(
            store,
            qx2,
            "boundary_enumerative_r0",
            conditioning_path,
            by_pair,
            expected,
            core,
        )
    ]
    atomic_json(
        store / "checkpoints/STAGE1_PRIMARY.json",
        {
            "schema": "ddm_qx4_primary.v1",
            "complete": True,
            "candidate": candidates[0],
        },
    )
    primary_over = candidates[0]["winner_payload"]["bytes"] > SECTION_CAP_BYTES
    if primary_over:
        for form in FORM_ORDER[1:]:
            candidates.append(
                retain_form(store, qx2, form, conditioning_path, by_pair, expected, core)
            )
            atomic_json(
                store / "checkpoints/STAGE1_SIBLING_PROGRESS.json",
                {
                    "schema": "ddm_qx4_sibling_progress.v1",
                    "complete_candidate_ids": [row["candidate_id"] for row in candidates],
                },
            )
    selected = min(
        candidates,
        key=lambda row: (row["winner_payload"]["bytes"], FORM_IDS[row["candidate_id"]]),
    )
    if primary_over and len(candidates) != len(FORM_ORDER):
        raise QX4Error("primary overprice did not trigger every decodable-conditioned sibling")
    receiver = receiver_proof(store, qx2, qx3, selected, conditioning_path, expected)
    section_cleared = selected["winner_payload"]["bytes"] <= SECTION_CAP_BYTES
    verdict = "SECTION-CLEARED" if section_cleared else "FORMULATION-CLOSED"
    result = {
        "schema": "ddm_qx4_decodable_conditioning_reprice.v1",
        "complete": True,
        "verdict": verdict,
        "verdict_scope": (
            "INSTANCE: selected decodable-conditioned form" if section_cleared else
            "FORMULATION: all six QX2 forms reconditioned on the bit-pinned QX3 field"
        ),
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "selection_mode": "full n600",
        "denominators": {"pairs": N_PAIRS, "sites": SITES, "events": EVENTS},
        "stage0": stage0,
        "conditioning": conditioning_stage,
        "candidate_configurations": candidates,
        "primary_over_cap": primary_over,
        "all_siblings_required_and_priced": primary_over,
        "selected": selected,
        "section_gate": {
            "maximum_payload_bytes": SECTION_CAP_BYTES,
            "observed_payload_bytes": selected["winner_payload"]["bytes"],
            "cleared": section_cleared,
            "delta_bytes_vs_cap": selected["winner_payload"]["bytes"] - SECTION_CAP_BYTES,
            "core_archive_bytes": QX1_CORE_ARCHIVE_BYTES,
            "section_header_bytes": QXE_SECTION_BYTES,
            "complete_archive_bytes": selected["archive"]["bytes"],
            "strict_archive_bytes_lt": ARCHIVE_GATE_EXCLUSIVE,
        },
        "receiver_proof": receiver,
        "scorer_realization_disposition": (
            {
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN-assigned scorer-realization arm",
                "consumer_store": str(store / "RESULT.json"),
                "fire_trigger": (
                    "MAIN claims the sole n600 scorer slot and binds this exact archive/receiver "
                    "output to QX1's two-plane realization without changing the selected section bytes"
                ),
            }
            if section_cleared
            else {"disposition": "FOLDED", "reason": "all decodable-conditioned QX2 forms exceed cap"}
        ),
        "authority_boundaries": {
            "scorers_loaded": 0,
            "contest_eval_invocations": 0,
            "modal_invocations": 0,
            "metal_invocations": 0,
            "distortion_measured": False,
            "contest_score_measured": False,
            "rate_and_receiver_exactness_measured": True,
            "receiver_applied_partition_is_not_a_scorer_result": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "RESULT.json", result)
    manifest = {
        "schema": "ddm_qx4_run_manifest.v1",
        "complete": True,
        "result": fact(store / "RESULT.json"),
        "command": f"{sys.executable} {Path(__file__).resolve()} --resume-from {store}",
        "source": fact(Path(__file__).resolve()),
        "retention": (
            "all raw forms, all real-coder candidates/repeats, packets/archives, fresh conditioning, "
            "and receiver-applied partitions retained"
        ),
        "cleanup": "none fired; all QX4 experiment payloads remain under AP custody",
    }
    atomic_json(store / "RUN_MANIFEST.json", manifest)
    atomic_json(
        store / "checkpoints/STAGE3_COMPLETE.json",
        {
            "schema": "ddm_qx4_complete.v1",
            "complete": True,
            "verdict": verdict,
            "result": fact(store / "RESULT.json"),
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, default=STORE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.resume_from)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "selected": result["selected"]["candidate_id"],
                "payload_bytes": result["selected"]["winner_payload"]["bytes"],
                "archive_bytes": result["selected"]["archive"]["bytes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
