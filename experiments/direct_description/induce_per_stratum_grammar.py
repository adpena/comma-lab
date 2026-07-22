#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Induce and measure per-stratum description grammars on frozen n600 labels.

This is a local, research-only corpus measurement.  It never emits a contest
archive and never invokes a scorer.  The generic decoder/vocabulary is free;
every video-derived derivation stream, codec tag, frame, and parameter is
counted in a small binary envelope.  Each stage is atomic and resumable.

The measured objects are:

* Movable: persistent island slots with BIRTH/PERSIST/DIE presence, centroid
  motion, and polygon shape productions;
* Lane: coherent slots with separate centerline, width, dash, and visibility
  productions, plus a persist-dash grammar;
* Boundary: exact transition support and lossy polygonal arc descriptions.

Lossless rows are verified by decoding their emitted bytes.  Lossy rows report
label-mask fidelity only; they are not through-R or score claims.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import lzma
import os
import shutil
import struct
import sys
import time
import zipfile
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import brotli
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.boundary_math.analytic_lane_render_band import (
    LaneBandRDTolerance,
    LaneBandRenderConfig,
    _unpack_matrix_to_pairs,
    build_lane_band_pairs_from_lstars,
    derive_rd_base_steps,
    rasterize_lane_coverage_range_dependent,
)
from tac.boundary_math.lane_track_and_smooth import coherent_slot_pack

SCHEMA = "ddm_g1_per_stratum_grammar_induction.v1"
AXIS = "[macOS-CPU frozen-scorer advisory]"
EXPECTED_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_SHAPE = (600, 384, 512)
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
TOTAL_CELLS = int(np.prod(EXPECTED_SHAPE))
RATE_BREAK_EVEN = 25.0 / 37_545_489.0
ENVELOPE_MAGIC = b"G1S1"
CODEC_IDS = {"brotli_q11": 1, "lzma1_raw_1m": 2, "zlib9": 3}
CODEC_NAMES = {value: key for key, value in CODEC_IDS.items()}
PRODUCTION_IDS = {
    "MASK_RUN": 1,
    "EVENT": 2,
    "CENTROID": 3,
    "SHAPE": 4,
    "LANE_CENTER": 5,
    "LANE_WIDTH": 6,
    "LANE_DASH": 7,
    "LANE_RANGE": 8,
    "EXCEPT_XOR": 9,
    "ARC_EVENT": 10,
    "ARC_VERTEX": 11,
}
PRODUCTION_NAMES = {value: key for key, value in PRODUCTION_IDS.items()}


class GrammarError(RuntimeError):
    """The measurement cannot support the requested claim."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def stored_npy_memmap(path: Path, key: str) -> np.memmap:
    """Memory-map one ZIP_STORED NPY member without copying its siblings."""
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED or info.file_size != info.compress_size:
            raise GrammarError(f"{path}:{member} is not ZIP_STORED")
        local_header = int(info.header_offset)
    with path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise GrammarError(f"bad ZIP local header for {member}")
        handle.seek(local_header + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def put_uleb(output: bytearray, value: int) -> None:
    if value < 0:
        raise GrammarError("ULEB value must be nonnegative")
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return


def get_uleb(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise GrammarError("corrupt ULEB stream")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def zigzag(value: int) -> int:
    return (int(value) << 1) ^ (int(value) >> 63)


def unzigzag(value: int) -> int:
    return (int(value) >> 1) ^ -(int(value) & 1)


def encode_signed_values(values: Iterable[int]) -> bytes:
    output = bytearray()
    for value in values:
        put_uleb(output, zigzag(int(value)))
    return bytes(output)


def decode_signed_values(data: bytes, count: int) -> np.ndarray:
    output = np.empty(count, dtype=np.int64)
    offset = 0
    for index in range(count):
        value, offset = get_uleb(data, offset)
        output[index] = unzigzag(value)
    if offset != len(data):
        raise GrammarError("signed stream has trailing bytes")
    return output


def compress_stream(raw: bytes) -> tuple[str, bytes, dict[str, int]]:
    candidates = {
        "brotli_q11": brotli.compress(raw, quality=11),
        "lzma1_raw_1m": lzma.compress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "preset": 1, "dict_size": 1 << 20}],
        ),
        "zlib9": zlib.compress(raw, 9),
    }
    winner = min(candidates, key=lambda name: (len(candidates[name]), CODEC_IDS[name]))
    return winner, candidates[winner], {name: len(value) for name, value in candidates.items()}


def decompress_stream(codec: str, payload: bytes) -> bytes:
    if codec == "brotli_q11":
        return brotli.decompress(payload)
    if codec == "lzma1_raw_1m":
        return lzma.decompress(
            payload,
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "preset": 1, "dict_size": 1 << 20}],
        )
    if codec == "zlib9":
        return zlib.decompress(payload)
    raise GrammarError(f"unknown codec: {codec}")


def encode_envelope(streams: Sequence[tuple[str, bytes]]) -> tuple[bytes, list[dict[str, Any]]]:
    """Encode complete production streams; every tag and frame byte is counted."""
    if len(streams) > 255:
        raise GrammarError("too many production streams")
    output = bytearray(ENVELOPE_MAGIC)
    output.append(len(streams))
    rows: list[dict[str, Any]] = []
    for production, raw in streams:
        codec, coded, candidates = compress_stream(raw)
        output.extend(
            struct.pack(
                "<BBII",
                PRODUCTION_IDS[production],
                CODEC_IDS[codec],
                len(raw),
                len(coded),
            )
        )
        output.extend(coded)
        rows.append(
            {
                "production": production,
                "logical_bytes": len(raw),
                "candidate_coded_bytes": candidates,
                "winner": codec,
                "winner_coded_bytes": len(coded),
                "frame_bytes": 10,
                "counted_bytes": len(coded) + 10,
            }
        )
    return bytes(output), rows


def decode_envelope(payload: bytes) -> list[tuple[str, bytes]]:
    if payload[:4] != ENVELOPE_MAGIC or len(payload) < 5:
        raise GrammarError("bad grammar envelope")
    count = payload[4]
    offset = 5
    streams: list[tuple[str, bytes]] = []
    for _ in range(count):
        if offset + 10 > len(payload):
            raise GrammarError("truncated grammar envelope")
        production_id, codec_id, raw_size, coded_size = struct.unpack_from("<BBII", payload, offset)
        offset += 10
        coded = payload[offset : offset + coded_size]
        offset += coded_size
        if len(coded) != coded_size:
            raise GrammarError("truncated coded production")
        raw = decompress_stream(CODEC_NAMES[codec_id], coded)
        if len(raw) != raw_size:
            raise GrammarError("production logical length mismatch")
        streams.append((PRODUCTION_NAMES[production_id], raw))
    if offset != len(payload):
        raise GrammarError("grammar envelope has trailing bytes")
    return streams


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    output = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    output[:, 1:] |= horizontal
    output[:, :-1] |= horizontal
    vertical = labels[1:, :] != labels[:-1, :]
    output[1:, :] |= vertical
    output[:-1, :] |= vertical
    return output


def encode_row_runs(mask_stack: np.ndarray) -> bytes:
    """Exact per-frame/per-row binary mask grammar."""
    mask_stack = np.asarray(mask_stack, dtype=bool)
    output = bytearray()
    put_uleb(output, mask_stack.shape[0])
    put_uleb(output, mask_stack.shape[1])
    put_uleb(output, mask_stack.shape[2])
    for mask in mask_stack:
        for row in mask:
            changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
            starts = np.flatnonzero(changes == 1)
            stops = np.flatnonzero(changes == -1)
            put_uleb(output, int(starts.size))
            previous_stop = 0
            for start, stop in zip(starts, stops, strict=True):
                put_uleb(output, int(start) - previous_stop)
                put_uleb(output, int(stop - start))
                previous_stop = int(stop)
    return bytes(output)


def decode_row_runs(payload: bytes) -> np.ndarray:
    offset = 0
    pairs, offset = get_uleb(payload, offset)
    height, offset = get_uleb(payload, offset)
    width, offset = get_uleb(payload, offset)
    output = np.zeros((pairs, height, width), dtype=bool)
    for pair in range(pairs):
        for row in range(height):
            count, offset = get_uleb(payload, offset)
            previous_stop = 0
            for _ in range(count):
                gap, offset = get_uleb(payload, offset)
                length, offset = get_uleb(payload, offset)
                start = previous_stop + gap
                stop = start + length
                if stop > width:
                    raise GrammarError("row run exceeds raster width")
                output[pair, row, start:stop] = True
                previous_stop = stop
    if offset != len(payload):
        raise GrammarError("row-run payload has trailing bytes")
    return output


def mask_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    union = true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "errors": false_positive + false_negative,
        "dseg_oracle_clean_rest": (false_positive + false_negative) / TOTAL_CELLS,
        "iou": true_positive / union if union else 1.0,
        "precision": true_positive / (true_positive + false_positive) if predicted.any() else 1.0,
        "recall": true_positive / (true_positive + false_negative) if target.any() else 1.0,
    }


def persist_payload(
    output_directory: Path,
    stratum: str,
    candidate: str,
    streams: Sequence[tuple[str, bytes]],
    *,
    fidelity: dict[str, Any],
    exact: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload, production_rows = encode_envelope(streams)
    decoded = decode_envelope(payload)
    if decoded != list(streams):
        raise GrammarError(f"{candidate}: envelope parse-back mismatch")
    path = output_directory / "payloads" / stratum.lower() / f"{candidate}.g1s"
    atomic_bytes(path, payload)
    return {
        "candidate": candidate,
        "stratum": stratum,
        "exact": bool(exact),
        "counted_bytes": len(payload),
        "payload_path": str(path),
        "payload_sha256": sha256_file(path),
        "production_bytes": production_rows,
        "fidelity": fidelity,
        "metadata": metadata or {},
    }


@dataclasses.dataclass
class PolygonTrackCorpus:
    presence: np.ndarray
    polygons: list[dict[int, np.ndarray]]
    births: int
    persists: int
    deaths: int
    max_slots: int


def extract_polygon_tracks(labels: np.memmap, epsilon: float) -> PolygonTrackCorpus:
    per_frame: list[list[np.ndarray]] = []
    max_count = 0
    for pair in range(EXPECTED_SHAPE[0]):
        target = np.asarray(labels[pair] == 3, dtype=np.uint8)
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [cv2.approxPolyDP(contour, epsilon, True)[:, 0, :].astype(np.int32) for contour in contours]
        polygons = [polygon for polygon in polygons if polygon.shape[0] >= 1]
        polygons.sort(key=lambda p: (int(np.mean(p[:, 1])), int(np.mean(p[:, 0])), -p.shape[0]))
        per_frame.append(polygons)
        max_count = max(max_count, len(polygons))

    presence = np.zeros((EXPECTED_SHAPE[0], max_count), dtype=bool)
    assignments: list[dict[int, np.ndarray]] = []
    previous_centers: dict[int, np.ndarray] = {}
    births = persists = deaths = 0
    for pair, polygons in enumerate(per_frame):
        current: dict[int, np.ndarray] = {}
        centers = np.asarray(
            [[float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))] for p in polygons], dtype=np.float64
        ).reshape(-1, 2)
        slots = sorted(previous_centers)
        used_polygons: set[int] = set()
        used_slots: set[int] = set()
        if slots and len(polygons):
            previous = np.stack([previous_centers[slot] for slot in slots])
            cost = np.sqrt(((previous[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
                if float(cost[row, col]) <= 48.0:
                    slot = slots[row]
                    current[slot] = polygons[col]
                    used_polygons.add(col)
                    used_slots.add(slot)
                    persists += 1
        free_slots = [slot for slot in range(max_count) if slot not in used_slots]
        for polygon_index, polygon in enumerate(polygons):
            if polygon_index in used_polygons:
                continue
            if not free_slots:
                raise GrammarError("polygon slot allocation overflow")
            slot = free_slots.pop(0)
            current[slot] = polygon
            births += 1
        deaths += len(set(previous_centers) - set(current))
        for slot in current:
            presence[pair, slot] = True
        assignments.append(current)
        previous_centers = {
            slot: np.asarray([np.mean(polygon[:, 0]), np.mean(polygon[:, 1])], dtype=np.float64)
            for slot, polygon in current.items()
        }
    return PolygonTrackCorpus(presence, assignments, births, persists, deaths, max_count)


def encode_polygon_tracks(
    corpus: PolygonTrackCorpus,
    *,
    morph_delta: bool,
) -> tuple[list[tuple[str, bytes]], np.ndarray, dict[str, Any]]:
    event = bytearray()
    put_uleb(event, corpus.presence.shape[0])
    put_uleb(event, corpus.max_slots)
    event.extend(np.packbits(corpus.presence.reshape(-1), bitorder="little").tobytes())
    centroids: list[int] = []
    shape = bytearray()
    previous_centroids: dict[int, tuple[int, int]] = {}
    previous_shapes: dict[int, np.ndarray] = {}
    rendered = np.zeros((corpus.presence.shape[0], EXPECTED_SHAPE[1], EXPECTED_SHAPE[2]), dtype=bool)
    vertices = 0
    morph_delta_events = 0
    for pair, assigned in enumerate(corpus.polygons):
        for slot in range(corpus.max_slots):
            polygon = assigned.get(slot)
            if polygon is None:
                previous_centroids.pop(slot, None)
                previous_shapes.pop(slot, None)
                continue
            center_x = int(np.rint(np.mean(polygon[:, 0])))
            center_y = int(np.rint(np.mean(polygon[:, 1])))
            old = previous_centroids.get(slot)
            centroids.extend(
                (center_x if old is None else center_x - old[0], center_y if old is None else center_y - old[1])
            )
            previous_centroids[slot] = (center_x, center_y)
            relative = polygon - np.asarray([center_x, center_y], dtype=np.int32)
            old_shape = previous_shapes.get(slot)
            use_delta = bool(morph_delta and old_shape is not None and old_shape.shape == relative.shape)
            put_uleb(shape, (polygon.shape[0] << 1) | int(use_delta))
            values = relative - old_shape if use_delta else relative
            for x_value, y_value in values:
                put_uleb(shape, zigzag(int(x_value)))
                put_uleb(shape, zigzag(int(y_value)))
            previous_shapes[slot] = relative
            morph_delta_events += int(use_delta)
            vertices += int(polygon.shape[0])
            cv2.fillPoly(rendered[pair].view(np.uint8), [polygon.reshape(-1, 1, 2)], 1)
    streams = [("EVENT", bytes(event)), ("CENTROID", encode_signed_values(centroids)), ("SHAPE", bytes(shape))]
    metadata = {
        "births": corpus.births,
        "persists": corpus.persists,
        "deaths": corpus.deaths,
        "max_slots": corpus.max_slots,
        "vertices": vertices,
        "morph_delta_events": morph_delta_events,
        "shape_mode": "same-order morph delta, otherwise absolute" if morph_delta else "absolute relative shape",
        "grammar": "BIRTH(abs_centroid,shape); PERSIST(delta_centroid,morph-or-shape); DIE; absent defaults to persist-nothing",
    }
    return streams, rendered, metadata


def decode_polygon_tracks(streams: Sequence[tuple[str, bytes]]) -> np.ndarray:
    by_name = dict(streams)
    event = by_name["EVENT"]
    event_offset = 0
    pairs, event_offset = get_uleb(event, event_offset)
    slots, event_offset = get_uleb(event, event_offset)
    presence_size = (pairs * slots + 7) // 8
    if len(event) - event_offset != presence_size:
        raise GrammarError("Movable EVENT stream length mismatch")
    presence = (
        np.unpackbits(np.frombuffer(event[event_offset:], dtype=np.uint8), bitorder="little")[: pairs * slots]
        .reshape(pairs, slots)
        .astype(bool)
    )
    centroid_values = decode_signed_values(by_name["CENTROID"], int(presence.sum()) * 2)
    centroid_offset = 0
    shape = by_name["SHAPE"]
    shape_offset = 0
    previous_centroids: dict[int, np.ndarray] = {}
    previous_shapes: dict[int, np.ndarray] = {}
    rendered = np.zeros((pairs, EXPECTED_SHAPE[1], EXPECTED_SHAPE[2]), dtype=bool)
    for pair in range(pairs):
        for slot in range(slots):
            if not presence[pair, slot]:
                previous_centroids.pop(slot, None)
                previous_shapes.pop(slot, None)
                continue
            delta_centroid = centroid_values[centroid_offset : centroid_offset + 2]
            centroid_offset += 2
            old_centroid = previous_centroids.get(slot)
            centroid = delta_centroid if old_centroid is None else old_centroid + delta_centroid
            previous_centroids[slot] = centroid
            header, shape_offset = get_uleb(shape, shape_offset)
            vertex_count = header >> 1
            use_delta = bool(header & 1)
            relative = np.empty((vertex_count, 2), dtype=np.int32)
            for vertex in range(vertex_count):
                x_value, shape_offset = get_uleb(shape, shape_offset)
                y_value, shape_offset = get_uleb(shape, shape_offset)
                relative[vertex] = (unzigzag(x_value), unzigzag(y_value))
            if use_delta:
                old_shape = previous_shapes.get(slot)
                if old_shape is None or old_shape.shape != relative.shape:
                    raise GrammarError("Movable morph delta lacks same-order predecessor")
                relative += old_shape
            previous_shapes[slot] = relative.copy()
            polygon = relative + centroid.astype(np.int32)
            cv2.fillPoly(rendered[pair].view(np.uint8), [polygon.reshape(-1, 1, 2)], 1)
    if centroid_offset != centroid_values.size or shape_offset != len(shape):
        raise GrammarError("Movable semantic stream has trailing values")
    return rendered


def measure_movable(labels: np.memmap, output_directory: Path) -> list[dict[str, Any]]:
    target = np.asarray(labels == 3, dtype=bool)
    exact_raw = encode_row_runs(target)
    if not np.array_equal(decode_row_runs(exact_raw), target):
        raise GrammarError("Movable exact row-run parse-back failed")
    rows = [
        persist_payload(
            output_directory,
            "Movable",
            "movable_lossless_row_runs",
            [("MASK_RUN", exact_raw)],
            fidelity=mask_metrics(target, target),
            exact=True,
            metadata={"grammar": "FRAME ROW RUN(gap,length)"},
        )
    ]
    for epsilon in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        corpus = extract_polygon_tracks(labels, epsilon)
        for morph_delta in (False, True):
            streams, rendered, metadata = encode_polygon_tracks(corpus, morph_delta=morph_delta)
            decoded = decode_polygon_tracks(streams)
            if not np.array_equal(decoded, rendered):
                raise GrammarError("Movable semantic parse-back mismatch")
            metadata["epsilon_px"] = epsilon
            mode = "morph_delta" if morph_delta else "shape_abs"
            rows.append(
                persist_payload(
                    output_directory,
                    "Movable",
                    f"movable_track_{mode}_eps{str(epsilon).replace('.', 'p')}",
                    streams,
                    fidelity=mask_metrics(decoded, target),
                    exact=False,
                    metadata=metadata,
                )
            )
    return rows


LANE_DIMENSIONS = {
    "LANE_CENTER": (0, 1, 2, 3),
    "LANE_WIDTH": (4, 5),
    "LANE_DASH": (6, 7, 8),
    "LANE_RANGE": (9, 10),
}


def lane_streams(
    matrix: np.ndarray,
    presence: np.ndarray,
    steps: np.ndarray,
    *,
    persist_dash: bool,
) -> tuple[list[tuple[str, bytes]], np.ndarray]:
    pairs, width = matrix.shape
    slots = presence.shape[1]
    dims = width // slots if slots else 11
    if dims != 11:
        raise GrammarError("lane slot width drift")
    full_steps = np.tile(steps, slots) if slots else np.zeros(0, dtype=np.float64)
    quantized = np.rint(matrix / full_steps).astype(np.int64) if width else np.zeros_like(matrix, dtype=np.int64)
    if persist_dash and slots:
        reshaped = quantized.reshape(pairs, slots, 11)
        for slot in range(slots):
            active = presence[:, slot]
            if active.any():
                median = np.rint(np.median(reshaped[active, slot, 6:9], axis=0)).astype(np.int64)
                reshaped[active, slot, 6:9] = median
        quantized = reshaped.reshape(pairs, width)
    event = bytearray()
    put_uleb(event, pairs)
    put_uleb(event, slots)
    event.extend(np.packbits(presence.reshape(-1), bitorder="little").tobytes())
    streams: list[tuple[str, bytes]] = [("EVENT", bytes(event))]
    quantized3 = quantized.reshape(pairs, slots, 11) if slots else np.zeros((pairs, 0, 11), dtype=np.int64)
    for production, indices in LANE_DIMENSIONS.items():
        emitted: list[int] = []
        previous = np.zeros((slots, len(indices)), dtype=np.int64)
        seen = np.zeros(slots, dtype=bool)
        for pair in range(pairs):
            for slot in range(slots):
                if not presence[pair, slot]:
                    continue
                current = quantized3[pair, slot, indices]
                values = current - previous[slot] if seen[slot] else current
                emitted.extend(int(value) for value in values)
                previous[slot] = current
                seen[slot] = True
        streams.append((production, encode_signed_values(emitted)))
    return streams, quantized


def decode_lane_streams(
    streams: Sequence[tuple[str, bytes]],
    steps: np.ndarray,
) -> tuple[list[list[Any]], np.ndarray]:
    by_name = dict(streams)
    event = by_name["EVENT"]
    offset = 0
    pairs, offset = get_uleb(event, offset)
    slots, offset = get_uleb(event, offset)
    presence_size = (pairs * slots + 7) // 8
    if len(event) - offset != presence_size:
        raise GrammarError("Lane EVENT stream length mismatch")
    presence = (
        np.unpackbits(np.frombuffer(event[offset:], dtype=np.uint8), bitorder="little")[: pairs * slots]
        .reshape(pairs, slots)
        .astype(bool)
    )
    pairs, slots = presence.shape
    quantized = np.zeros((pairs, slots, 11), dtype=np.int64)
    active_count = int(np.count_nonzero(presence))
    for production, indices in LANE_DIMENSIONS.items():
        values = decode_signed_values(by_name[production], active_count * len(indices))
        offset = 0
        previous = np.zeros((slots, len(indices)), dtype=np.int64)
        seen = np.zeros(slots, dtype=bool)
        for pair in range(pairs):
            for slot in range(slots):
                if not presence[pair, slot]:
                    continue
                delta = values[offset : offset + len(indices)]
                offset += len(indices)
                current = previous[slot] + delta if seen[slot] else delta
                quantized[pair, slot, indices] = current
                previous[slot] = current
                seen[slot] = True
        if offset != values.size:
            raise GrammarError(f"Lane {production} active-value count mismatch")
    matrix = quantized.reshape(pairs, slots * 11).astype(np.float64) * np.tile(steps, slots)
    return _unpack_matrix_to_pairs(matrix, presence, slots), presence


def render_lane_masks(lines: Sequence[Sequence[Any]]) -> np.ndarray:
    output = np.zeros(EXPECTED_SHAPE, dtype=bool)
    for pair, pair_lines in enumerate(lines):
        output[pair] = rasterize_lane_coverage_range_dependent(list(pair_lines)) >= 0.5
    return output


def measure_lane(labels: np.memmap, output_directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = np.asarray(labels == 1, dtype=bool)
    cfg = LaneBandRenderConfig()
    fitted_lines, fit_stats = build_lane_band_pairs_from_lstars(labels, cfg)
    assignment = coherent_slot_pack(fitted_lines)
    rows: list[dict[str, Any]] = []
    exact_raw = encode_row_runs(target)
    if not np.array_equal(decode_row_runs(exact_raw), target):
        raise GrammarError("Lane exact row-run parse-back failed")
    rows.append(
        persist_payload(
            output_directory,
            "Lane",
            "lane_lossless_row_runs",
            [("MASK_RUN", exact_raw)],
            fidelity=mask_metrics(target, target),
            exact=True,
            metadata={"grammar": "FRAME ROW RUN(gap,length)"},
        )
    )
    best_lossy: tuple[int, list[tuple[str, bytes]], np.ndarray, dict[str, Any], str] | None = None
    for multiplier in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        tolerance = LaneBandRDTolerance(
            lat_tol_m=0.02 * multiplier,
            hw_tol_px=0.1 * multiplier,
            dash_period_tol_m=0.1 * multiplier,
            dash_phase_tol_m=0.1 * multiplier,
            dash_duty_tol=0.02 * multiplier,
            forward_range_tol_m=0.5 * multiplier,
        )
        steps = derive_rd_base_steps(tolerance)
        for persist_dash in (False, True):
            suffix = "persist_dash" if persist_dash else "delta_dash"
            candidate = f"lane_slots_{suffix}_tolx{str(multiplier).replace('.', 'p')}"
            streams, _quantized = lane_streams(assignment.M, assignment.presence, steps, persist_dash=persist_dash)
            decoded_lines, decoded_presence = decode_lane_streams(streams, steps)
            if not np.array_equal(decoded_presence, assignment.presence):
                raise GrammarError(f"{candidate}: Lane EVENT parse-back mismatch")
            rendered = render_lane_masks(decoded_lines)
            row = persist_payload(
                output_directory,
                "Lane",
                candidate,
                streams,
                fidelity=mask_metrics(rendered, target),
                exact=False,
                metadata={
                    "tolerance_multiplier": multiplier,
                    "persist_dash": persist_dash,
                    "slots": assignment.K,
                    "births": assignment.n_births,
                    "deaths": assignment.n_deaths,
                    "base_steps": steps.tolist(),
                    "grammar": "EVENT; birth-absolute; active-persist delta; absent costs no value; CENTER/WIDTH/DASH/RANGE",
                },
            )
            rows.append(row)
            if best_lossy is None or (row["fidelity"]["errors"], row["counted_bytes"]) < (
                best_lossy[3]["errors"],
                best_lossy[0],
            ):
                best_lossy = (row["counted_bytes"], streams, rendered, row["fidelity"], candidate)

    if best_lossy is None:
        raise GrammarError("no Lane grammar candidates")
    # Two-part MDL: add an exact XOR exception stream to the highest-fidelity grammar.
    _, base_streams, base_rendered, _, base_name = best_lossy
    residual_raw = encode_row_runs(base_rendered ^ target)
    residual_decoded = decode_row_runs(residual_raw)
    exact_rendered = base_rendered ^ residual_decoded
    if not np.array_equal(exact_rendered, target):
        raise GrammarError("Lane grammar+residual parse-back failed")
    rows.append(
        persist_payload(
            output_directory,
            "Lane",
            "lane_lossless_two_part_grammar_plus_xor",
            [*base_streams, ("EXCEPT_XOR", residual_raw)],
            fidelity=mask_metrics(exact_rendered, target),
            exact=True,
            metadata={"base_candidate": base_name, "grammar": "L(G derivation)+L(exceptions|G)"},
        )
    )
    return rows, fit_stats


def encode_arc_streams(mask_stack: np.ndarray, epsilon: float) -> tuple[list[tuple[str, bytes]], np.ndarray, int]:
    events = bytearray()
    vertices = bytearray()
    rendered = np.zeros(mask_stack.shape, dtype=bool)
    vertex_count = 0
    put_uleb(events, mask_stack.shape[0])
    for pair, mask in enumerate(mask_stack):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(contour, epsilon, False) for contour in contours]
        contours = [contour for contour in contours if contour.shape[0] >= 2]
        contours.sort(key=lambda contour: tuple(int(value) for value in cv2.boundingRect(contour)[:2]))
        put_uleb(events, len(contours))
        for contour in contours:
            points = contour[:, 0, :].astype(np.int32)
            put_uleb(vertices, points.shape[0])
            previous_x = previous_y = 0
            for index, (x_value, y_value) in enumerate(points):
                dx = int(x_value) if index == 0 else int(x_value) - previous_x
                dy = int(y_value) if index == 0 else int(y_value) - previous_y
                put_uleb(vertices, zigzag(dx))
                put_uleb(vertices, zigzag(dy))
                previous_x, previous_y = int(x_value), int(y_value)
            vertex_count += int(points.shape[0])
            cv2.polylines(rendered[pair].view(np.uint8), [points.reshape(-1, 1, 2)], False, 1, 2)
    return [("ARC_EVENT", bytes(events)), ("ARC_VERTEX", bytes(vertices))], rendered, vertex_count


def decode_arc_streams(streams: Sequence[tuple[str, bytes]]) -> np.ndarray:
    by_name = dict(streams)
    events = by_name["ARC_EVENT"]
    vertices = by_name["ARC_VERTEX"]
    event_offset = vertex_offset = 0
    pairs, event_offset = get_uleb(events, event_offset)
    rendered = np.zeros((pairs, EXPECTED_SHAPE[1], EXPECTED_SHAPE[2]), dtype=bool)
    for pair in range(pairs):
        contour_count, event_offset = get_uleb(events, event_offset)
        for _ in range(contour_count):
            point_count, vertex_offset = get_uleb(vertices, vertex_offset)
            points = np.empty((point_count, 2), dtype=np.int32)
            previous_x = previous_y = 0
            for index in range(point_count):
                dx, vertex_offset = get_uleb(vertices, vertex_offset)
                dy, vertex_offset = get_uleb(vertices, vertex_offset)
                x_value = unzigzag(dx) + (previous_x if index else 0)
                y_value = unzigzag(dy) + (previous_y if index else 0)
                points[index] = (x_value, y_value)
                previous_x, previous_y = x_value, y_value
            cv2.polylines(rendered[pair].view(np.uint8), [points.reshape(-1, 1, 2)], False, 1, 2)
    if event_offset != len(events) or vertex_offset != len(vertices):
        raise GrammarError("Boundary semantic stream has trailing values")
    return rendered


def measure_boundary(labels: np.memmap, output_directory: Path) -> list[dict[str, Any]]:
    target = np.stack([boundary_mask(np.asarray(labels[pair])) for pair in range(EXPECTED_SHAPE[0])])
    exact_raw = encode_row_runs(target)
    if not np.array_equal(decode_row_runs(exact_raw), target):
        raise GrammarError("Boundary exact row-run parse-back failed")
    rows = [
        persist_payload(
            output_directory,
            "Boundary",
            "boundary_lossless_row_runs",
            [("MASK_RUN", exact_raw)],
            fidelity=mask_metrics(target, target),
            exact=True,
            metadata={"grammar": "transition-support ROW RUN"},
        )
    ]
    for epsilon in (0.5, 1.0, 2.0, 4.0):
        streams, rendered, vertices = encode_arc_streams(target, epsilon)
        decoded = decode_arc_streams(streams)
        if not np.array_equal(decoded, rendered):
            raise GrammarError("Boundary semantic parse-back mismatch")
        rows.append(
            persist_payload(
                output_directory,
                "Boundary",
                f"boundary_arc_eps{str(epsilon).replace('.', 'p')}",
                streams,
                fidelity=mask_metrics(decoded, target),
                exact=False,
                metadata={"epsilon_px": epsilon, "vertices": vertices, "grammar": "ARC(start,Freeman-like deltas)"},
            )
        )
    return rows


def rank_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            0 if row["exact"] else 1,
            row["counted_bytes"] if row["exact"] else row["fidelity"]["errors"],
            row["counted_bytes"],
            row["candidate"],
        ),
    )


def combined_projection(lane_rows: Sequence[dict[str, Any]], movable_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    combinations: list[dict[str, Any]] = []
    for lane in lane_rows:
        if lane["exact"]:
            continue
        for movable in movable_rows:
            if movable["exact"]:
                continue
            total_bytes = int(lane["counted_bytes"] + movable["counted_bytes"])
            errors_upper = int(lane["fidelity"]["errors"] + movable["fidelity"]["errors"])
            combinations.append(
                {
                    "lane": lane["candidate"],
                    "movable": movable["candidate"],
                    "counted_bytes": total_bytes,
                    "oracle_clean_rest_dseg_union_upper": errors_upper / TOTAL_CELLS,
                    "errors_union_upper": errors_upper,
                    "under_60000_bytes": total_bytes <= 60_000,
                    "under_0p005_union_upper": errors_upper / TOTAL_CELLS <= 0.005,
                }
            )
    combinations.sort(
        key=lambda row: (
            0 if row["under_60000_bytes"] else 1,
            row["oracle_clean_rest_dseg_union_upper"],
            row["counted_bytes"],
        )
    )
    feasible = [row for row in combinations if row["under_60000_bytes"]]
    best = feasible[0] if feasible else None
    return {
        "method": "DERIVED union upper bound from independently decoded Lane and Movable masks; Road/Undrivable/MyCar assumed exact and cross-stratum overwrite interactions omitted",
        "score_claim": False,
        "receiver_closed": False,
        "budget_bytes": 60_000,
        "target_dseg": 0.005,
        "best_under_budget": best,
        "joint_gate_passed": bool(best and best["under_0p005_union_upper"]),
        "ranked_combinations": combinations[:20],
    }


def stage_path(output_directory: Path, stage: str) -> Path:
    return output_directory / "stage_checkpoints" / f"{stage}.json"


def run_stage(
    stage: str,
    output_directory: Path,
    resume: bool,
    function: Any,
) -> dict[str, Any]:
    checkpoint = stage_path(output_directory, stage)
    if resume and checkpoint.is_file():
        payload = json.loads(checkpoint.read_text())
        if payload.get("stage") != stage or payload.get("status") != "complete":
            raise GrammarError(f"invalid resume checkpoint: {checkpoint}")
        for row in payload["rows"]:
            path = Path(row["payload_path"])
            if sha256_file(path) != row["payload_sha256"]:
                raise GrammarError(f"resume payload hash drift: {path}")
        return payload
    started = time.time()
    result = function()
    rows, extra = result if isinstance(result, tuple) else (result, {})
    payload = {
        "schema": "ddm_g1_grammar_stage.v1",
        "stage": stage,
        "status": "complete",
        "seconds": time.time() - started,
        "rows": rows,
        "extra": extra,
    }
    atomic_json(checkpoint, payload)
    return payload


def build_receipt(
    cache: Path,
    output_directory: Path,
    semantic_argv: list[str],
    *,
    resume: bool,
) -> dict[str, Any]:
    if sha256_file(cache) != EXPECTED_CACHE_SHA256:
        raise GrammarError("frozen n600 cache hash mismatch")
    labels = stored_npy_memmap(cache, "lstars")
    if tuple(labels.shape) != EXPECTED_SHAPE or labels.dtype != np.int64:
        raise GrammarError(f"lstars schema mismatch: {labels.shape} {labels.dtype}")
    usage = shutil.disk_usage(output_directory.parent)
    if usage.free < 64 << 20:
        raise GrammarError("storage preflight: less than 64 MiB free")

    movable = run_stage(
        "01_movable",
        output_directory,
        resume,
        lambda: measure_movable(labels, output_directory),
    )
    lane = run_stage(
        "02_lane",
        output_directory,
        resume,
        lambda: measure_lane(labels, output_directory),
    )
    boundary = run_stage(
        "03_boundary",
        output_directory,
        resume,
        lambda: measure_boundary(labels, output_directory),
    )
    ranked = {
        "Movable": rank_rows(movable["rows"]),
        "Lane": rank_rows(lane["rows"]),
        "Boundary": rank_rows(boundary["rows"]),
    }
    projection = combined_projection(lane["rows"], movable["rows"])
    receipt = {
        "schema": SCHEMA,
        "axis": AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "candidate_archive": False,
        "pointer": "0.1910828242 [contest-CPU] UNCHANGED",
        "verdict_scope": "REFERENCE_LABEL_CORPUS_AND_THE_EMITTED_GRAMMAR_FAMILIES_ONLY",
        "cache": {"path": str(cache), "bytes": cache.stat().st_size, "sha256": EXPECTED_CACHE_SHA256},
        "semantic_argv": semantic_argv,
        "storage_preflight": {
            "tier": str(output_directory),
            "observed_free_bytes": usage.free,
            "required_free_bytes": 64 << 20,
            "status": "PASS",
        },
        "coder_contract": {
            "families": ["Brotli quality 11", "raw LZMA1 preset 1 dict 1MiB", "zlib level 9"],
            "selection": "actual smallest complete production stream; codec id and framing counted",
            "entropy_estimates_used_for_ranking": False,
        },
        "rate_break_even_score_units_per_byte": RATE_BREAK_EVEN,
        "strata": list(CLASS_NAMES),
        "ranked_by_stratum": ranked,
        "coverage_projection": projection,
        "fit_stats": {"Lane": lane.get("extra", {})},
        "stage_checkpoints": [
            str(stage_path(output_directory, name)) for name in ("01_movable", "02_lane", "03_boundary")
        ],
        "blocker_delta_vs_603": "The v12 correction inventory is no longer the open measurement. This receipt prices PREDICT-native island/Lane/arc syntax; receiver-visible RGB/Pose realization and cross-stratum overwrite composition remain owed.",
        "stores_consulted": [
            "#596 rep_mine_solved_binary",
            "#610 wrong_levels per-stratum ops grammar",
            "v12 obligation drain residual decomposition",
            "truly_optimal_coder_survey_603_613_20260722",
            "frozen gt_n600.lstars",
            "tac.boundary_math analytic Lane/coherent-slot primitives",
        ],
        "main_landing_review_required": True,
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--execution-allowed", choices=("false",), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    semantic_argv = [
        "experiments/direct_description/induce_per_stratum_grammar.py",
        "--cache",
        str(args.cache),
        "--output-directory",
        str(args.output_directory),
        "--execution-allowed",
        "false",
        *(["--resume"] if args.resume else []),
    ]
    try:
        receipt = build_receipt(
            args.cache,
            args.output_directory,
            semantic_argv,
            resume=args.resume,
        )
        receipt_path = args.output_directory / "ddm_g1_grammar_induction_n600_receipt.json"
        atomic_json(receipt_path, receipt)
    except (GrammarError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"receipt": str(receipt_path), "sha256": sha256_file(receipt_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
