# SPDX-License-Identifier: MIT
"""Receiver-portable G1 Movable polygon-worldsheet derivation grammar.

This is the productionized, window-generic form of the measured G1 knee:
``EVENT`` presence, delta ``CENTROID``, and absolute relative ``SHAPE`` at
OpenCV contour epsilon 1.0.  Generic parsing/rasterization logic is free; the
returned G1S1 envelope is the complete counted derivation string.
"""

from __future__ import annotations

import hashlib
import lzma
import math
import struct
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import brotli
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from tac.optimization.direct_description_minimizer import DirectDescriptionError

MAGIC: Final = b"G1S1"
CODEC_IDS: Final = {"brotli_q11": 1, "lzma1_raw_1m": 2, "zlib9": 3}
CODEC_NAMES: Final = {value: key for key, value in CODEC_IDS.items()}
PRODUCTION_IDS: Final = {"EVENT": 2, "CENTROID": 3, "SHAPE": 4}
PRODUCTION_NAMES: Final = {value: key for key, value in PRODUCTION_IDS.items()}
HEIGHT: Final = 384
WIDTH: Final = 512
EPSILON_PIXELS: Final = 1.0
MATCH_RADIUS_PIXELS: Final = 48.0


@dataclass(frozen=True, slots=True)
class G1MovableWorldsheetMetadata:
    pair_count: int
    max_slots: int
    births: int
    persists: int
    deaths: int
    vertices: int
    production_counted_bytes: dict[str, int]
    payload_bytes: int
    payload_sha256: str
    decoded_mask_errors: int | None
    decoded_clean_rest_dseg: float | None


@dataclass(frozen=True, slots=True)
class G1ShapeTemplateV1:
    """One exact relative polygon owned by the counted G1 SHAPE stream.

    ``template_ref`` is content-addressed.  Keeping the integer vertices in the
    lift is intentional: they are already counted in G1 and are the lossless
    residual around which the low-dimensional aspect/rotation coordinates move.
    """

    template_ref: str
    relative_vertices_xy: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class G1WorldsheetKnotV1:
    """Explicit per-island knot recovered from one G1 observation."""

    object_id: int
    slot_id: int
    pair_index: int
    center_x: int
    center_y: int
    template_ref: str
    aspect_log: float
    rotation_radians: float


@dataclass(frozen=True, slots=True)
class G1WorldsheetTrackV1:
    """One contiguous birth/persist/death lifecycle in a reusable G1 slot."""

    object_id: int
    slot_id: int
    generation: int
    birth_pair: int
    death_pair_exclusive: int
    knot_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class G1WorldsheetParameterLiftV1:
    """Lossless typed parameter surface for a counted G1 derivation string.

    The lift does not add a second payload.  Re-emission rebuilds EVENT,
    CENTROID, and SHAPE from the typed records and passes them through the same
    canonical codec selection as :func:`encode_g1_movable_worldsheet`.
    """

    pair_count: int
    max_slots: int
    tracks: tuple[G1WorldsheetTrackV1, ...]
    knots: tuple[G1WorldsheetKnotV1, ...]
    templates: tuple[G1ShapeTemplateV1, ...]
    source_payload_sha256: str
    source_payload_bytes: int


def _shape_parameters(relative: np.ndarray) -> tuple[float, float]:
    """Return deterministic log-aspect and major-axis rotation for one polygon."""

    centered = np.asarray(relative, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered))
    values, vectors = np.linalg.eigh(covariance)
    major = max(float(values[-1]), 1.0e-12)
    minor = max(float(values[0]), 1.0e-12)
    aspect_log = 0.5 * math.log(major / minor)
    vector = vectors[:, -1]
    rotation = math.atan2(float(vector[1]), float(vector[0]))
    # Eigenvectors have a sign ambiguity.  Canonicalize to [-pi/2, pi/2).
    if rotation >= math.pi / 2:
        rotation -= math.pi
    elif rotation < -math.pi / 2:
        rotation += math.pi
    return aspect_log, rotation


def _template_ref(relative: np.ndarray) -> str:
    vertices = np.ascontiguousarray(relative, dtype="<i4")
    return hashlib.sha256(struct.pack("<I", len(vertices)) + vertices.tobytes()).hexdigest()


@dataclass(slots=True)
class _PolygonTrackCorpus:
    presence: np.ndarray
    polygons: list[dict[int, np.ndarray]]
    births: int
    persists: int
    deaths: int
    max_slots: int


def _put_uleb(output: bytearray, value: int) -> None:
    if value < 0:
        raise DirectDescriptionError("G1 ULEB value must be nonnegative")
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return


def _get_uleb(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise DirectDescriptionError("corrupt G1 ULEB stream")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _zigzag(value: int) -> int:
    return (int(value) << 1) ^ (int(value) >> 63)


def _unzigzag(value: int) -> int:
    return (int(value) >> 1) ^ -(int(value) & 1)


def _encode_signed(values: Iterable[int]) -> bytes:
    output = bytearray()
    for value in values:
        _put_uleb(output, _zigzag(int(value)))
    return bytes(output)


def _decode_signed(data: bytes, count: int) -> np.ndarray:
    output = np.empty(count, dtype=np.int64)
    offset = 0
    for index in range(count):
        value, offset = _get_uleb(data, offset)
        output[index] = _unzigzag(value)
    if offset != len(data):
        raise DirectDescriptionError("G1 signed stream has trailing bytes")
    return output


def _compress(raw: bytes) -> tuple[str, bytes]:
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
    return winner, candidates[winner]


def _decompress(codec: str, payload: bytes) -> bytes:
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
    raise DirectDescriptionError("unknown G1 derivation codec")


def _encode_envelope(streams: Sequence[tuple[str, bytes]]) -> tuple[bytes, dict[str, int]]:
    output = bytearray(MAGIC)
    output.append(len(streams))
    counted: dict[str, int] = {}
    for production, raw in streams:
        codec, coded = _compress(raw)
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
        counted[production] = len(coded) + 10
    counted["envelope_header"] = 5
    return bytes(output), counted


def _decode_envelope(payload: bytes) -> list[tuple[str, bytes]]:
    if len(payload) < 5 or payload[:4] != MAGIC or payload[4] != 3:
        raise DirectDescriptionError("G1 Movable envelope header is invalid")
    offset = 5
    streams: list[tuple[str, bytes]] = []
    for _ in range(payload[4]):
        if offset + 10 > len(payload):
            raise DirectDescriptionError("G1 Movable envelope is truncated")
        production_id, codec_id, raw_size, coded_size = struct.unpack_from("<BBII", payload, offset)
        offset += 10
        if production_id not in PRODUCTION_NAMES or codec_id not in CODEC_NAMES:
            raise DirectDescriptionError("G1 Movable envelope has an unknown tag")
        if coded_size > len(payload) - offset or raw_size > 64 << 20:
            raise DirectDescriptionError("G1 Movable envelope size is invalid")
        coded = payload[offset : offset + coded_size]
        offset += coded_size
        try:
            raw = _decompress(CODEC_NAMES[codec_id], coded)
        except (brotli.error, lzma.LZMAError, zlib.error) as exc:
            raise DirectDescriptionError("G1 Movable stream decompression failed") from exc
        if len(raw) != raw_size:
            raise DirectDescriptionError("G1 Movable raw-size custody failed")
        streams.append((PRODUCTION_NAMES[production_id], raw))
    if offset != len(payload) or [name for name, _ in streams] != ["EVENT", "CENTROID", "SHAPE"]:
        raise DirectDescriptionError("G1 Movable stream order/cardinality is invalid")
    canonical, _ = _encode_envelope(streams)
    if canonical != payload:
        raise DirectDescriptionError("G1 Movable derivation is not canonical on parse-back")
    return streams


def _extract_tracks(labels: np.ndarray) -> _PolygonTrackCorpus:
    if labels.ndim != 3 or labels.shape[1:] != (HEIGHT, WIDTH) or not 1 <= labels.shape[0] <= 600:
        raise DirectDescriptionError("G1 Movable source window must be [1,600]x384x512")
    per_frame: list[list[np.ndarray]] = []
    max_count = 0
    for pair in range(labels.shape[0]):
        target = np.asarray(labels[pair] == 3, dtype=np.uint8)
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [cv2.approxPolyDP(contour, EPSILON_PIXELS, True)[:, 0, :].astype(np.int32) for contour in contours]
        polygons = [polygon for polygon in polygons if polygon.shape[0] >= 1]
        polygons.sort(key=lambda polygon: (int(np.mean(polygon[:, 1])), int(np.mean(polygon[:, 0])), -polygon.shape[0]))
        per_frame.append(polygons)
        max_count = max(max_count, len(polygons))
    if max_count == 0 or max_count > 64:
        raise DirectDescriptionError("G1 Movable slot count is outside [1,64]")
    presence = np.zeros((labels.shape[0], max_count), dtype=bool)
    assignments: list[dict[int, np.ndarray]] = []
    previous_centers: dict[int, np.ndarray] = {}
    births = persists = deaths = 0
    for pair, polygons in enumerate(per_frame):
        current: dict[int, np.ndarray] = {}
        centers = np.asarray(
            [[float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1]))] for polygon in polygons],
            dtype=np.float64,
        ).reshape(-1, 2)
        slots = sorted(previous_centers)
        used_polygons: set[int] = set()
        used_slots: set[int] = set()
        if slots and polygons:
            previous = np.stack([previous_centers[slot] for slot in slots])
            cost = np.sqrt(np.square(previous[:, None, :] - centers[None, :, :]).sum(axis=2))
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                if float(cost[row, column]) <= MATCH_RADIUS_PIXELS:
                    slot = slots[row]
                    current[slot] = polygons[column]
                    used_polygons.add(column)
                    used_slots.add(slot)
                    persists += 1
        free_slots = [slot for slot in range(max_count) if slot not in used_slots]
        for polygon_index, polygon in enumerate(polygons):
            if polygon_index in used_polygons:
                continue
            if not free_slots:
                raise DirectDescriptionError("G1 Movable polygon slot allocation overflow")
            current[free_slots.pop(0)] = polygon
            births += 1
        deaths += len(set(previous_centers) - set(current))
        for slot in current:
            presence[pair, slot] = True
        assignments.append(current)
        previous_centers = {
            slot: np.asarray([np.mean(polygon[:, 0]), np.mean(polygon[:, 1])], dtype=np.float64)
            for slot, polygon in current.items()
        }
    return _PolygonTrackCorpus(presence, assignments, births, persists, deaths, max_count)


def encode_g1_movable_worldsheet(labels: np.ndarray) -> tuple[bytes, G1MovableWorldsheetMetadata]:
    """Encode the exact G1 eps1 absolute-shape production set for one window."""

    corpus = _extract_tracks(np.asarray(labels))
    event = bytearray()
    _put_uleb(event, corpus.presence.shape[0])
    _put_uleb(event, corpus.max_slots)
    event.extend(np.packbits(corpus.presence.reshape(-1), bitorder="little").tobytes())
    centroids: list[int] = []
    shape = bytearray()
    previous_centroids: dict[int, tuple[int, int]] = {}
    vertices = 0
    for assigned in corpus.polygons:
        for slot in range(corpus.max_slots):
            polygon = assigned.get(slot)
            if polygon is None:
                previous_centroids.pop(slot, None)
                continue
            center_x = int(np.rint(np.mean(polygon[:, 0])))
            center_y = int(np.rint(np.mean(polygon[:, 1])))
            old = previous_centroids.get(slot)
            centroids.extend(
                (center_x if old is None else center_x - old[0], center_y if old is None else center_y - old[1])
            )
            previous_centroids[slot] = (center_x, center_y)
            relative = polygon - np.asarray([center_x, center_y], dtype=np.int32)
            _put_uleb(shape, polygon.shape[0] << 1)  # absolute relative shape; G1 eps1 winner
            for x_value, y_value in relative:
                _put_uleb(shape, _zigzag(int(x_value)))
                _put_uleb(shape, _zigzag(int(y_value)))
            vertices += int(polygon.shape[0])
    payload, counted = _encode_envelope(
        [("EVENT", bytes(event)), ("CENTROID", _encode_signed(centroids)), ("SHAPE", bytes(shape))]
    )
    decoded, decoded_meta = decode_g1_movable_worldsheet(payload, expected_pairs=labels.shape[0])
    if decoded.shape != labels.shape or decoded_meta.vertices != vertices:
        raise DirectDescriptionError("G1 Movable semantic parse-back failed")
    target = np.asarray(labels == 3, dtype=bool)
    decoded_mask_errors = int(np.count_nonzero(decoded != target))
    metadata = G1MovableWorldsheetMetadata(
        pair_count=labels.shape[0],
        max_slots=corpus.max_slots,
        births=corpus.births,
        persists=corpus.persists,
        deaths=corpus.deaths,
        vertices=vertices,
        production_counted_bytes=counted,
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        decoded_mask_errors=decoded_mask_errors,
        decoded_clean_rest_dseg=decoded_mask_errors / float(labels.size),
    )
    return payload, metadata


def decode_g1_movable_worldsheet(
    payload: bytes,
    *,
    expected_pairs: int | None = None,
) -> tuple[np.ndarray, G1MovableWorldsheetMetadata]:
    """Strictly parse and semantic-decode one counted G1 derivation string."""

    streams = _decode_envelope(payload)
    by_name = dict(streams)
    event = by_name["EVENT"]
    event_offset = 0
    pairs, event_offset = _get_uleb(event, event_offset)
    slots, event_offset = _get_uleb(event, event_offset)
    if not 1 <= pairs <= 600 or not 1 <= slots <= 64 or (expected_pairs is not None and pairs != expected_pairs):
        raise DirectDescriptionError("G1 Movable pair/slot custody failed")
    presence_size = (pairs * slots + 7) // 8
    if len(event) - event_offset != presence_size:
        raise DirectDescriptionError("G1 Movable EVENT stream length mismatch")
    presence = (
        np.unpackbits(np.frombuffer(event[event_offset:], dtype=np.uint8), bitorder="little")[: pairs * slots]
        .reshape(pairs, slots)
        .astype(bool)
    )
    centroid_values = _decode_signed(by_name["CENTROID"], int(presence.sum()) * 2)
    centroid_offset = 0
    shape = by_name["SHAPE"]
    shape_offset = 0
    previous_centroids: dict[int, np.ndarray] = {}
    previous_shapes: dict[int, np.ndarray] = {}
    rendered = np.zeros((pairs, HEIGHT, WIDTH), dtype=bool)
    births = persists = deaths = vertices = 0
    previous_presence = np.zeros(slots, dtype=bool)
    for pair in range(pairs):
        births += int(np.count_nonzero(presence[pair] & ~previous_presence))
        persists += int(np.count_nonzero(presence[pair] & previous_presence))
        deaths += int(np.count_nonzero(~presence[pair] & previous_presence))
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
            header, shape_offset = _get_uleb(shape, shape_offset)
            vertex_count = header >> 1
            use_delta = bool(header & 1)
            if not 1 <= vertex_count <= 4096:
                raise DirectDescriptionError("G1 Movable vertex count is invalid")
            relative = np.empty((vertex_count, 2), dtype=np.int32)
            for vertex in range(vertex_count):
                x_value, shape_offset = _get_uleb(shape, shape_offset)
                y_value, shape_offset = _get_uleb(shape, shape_offset)
                relative[vertex] = (_unzigzag(x_value), _unzigzag(y_value))
            if use_delta:
                old_shape = previous_shapes.get(slot)
                if old_shape is None or old_shape.shape != relative.shape:
                    raise DirectDescriptionError("G1 Movable morph delta lacks same-order predecessor")
                relative += old_shape
            previous_shapes[slot] = relative.copy()
            polygon = relative + centroid.astype(np.int32)
            if (
                np.any(polygon[:, 0] < 0)
                or np.any(polygon[:, 0] >= WIDTH)
                or np.any(polygon[:, 1] < 0)
                or np.any(polygon[:, 1] >= HEIGHT)
            ):
                raise DirectDescriptionError("G1 Movable polygon escaped scorer geometry")
            cv2.fillPoly(rendered[pair].view(np.uint8), [polygon.reshape(-1, 1, 2)], 1)
            vertices += vertex_count
        previous_presence = presence[pair]
    if centroid_offset != centroid_values.size or shape_offset != len(shape):
        raise DirectDescriptionError("G1 Movable semantic stream has trailing values")
    _canonical, counted = _encode_envelope(streams)
    metadata = G1MovableWorldsheetMetadata(
        pair_count=pairs,
        max_slots=slots,
        births=births,
        persists=persists,
        deaths=deaths,
        vertices=vertices,
        production_counted_bytes=counted,
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        decoded_mask_errors=None,
        decoded_clean_rest_dseg=None,
    )
    return rendered, metadata


def lift_g1_movable_worldsheet(payload: bytes) -> G1WorldsheetParameterLiftV1:
    """Lift a canonical G1 payload into explicit lifecycle and knot records.

    This is a parser-to-parameters operation, not a mask re-extraction.  It
    consumes the exact EVENT/CENTROID/SHAPE symbols, so slot identity, birth and
    death, centroid deltas, polygon ordering, and codec-relevant integer values
    survive.  G1's production encoder emits absolute relative shapes; a future
    delta-shape revision is refused until it receives its own lossless lift.
    """

    streams = _decode_envelope(payload)
    by_name = dict(streams)
    event = by_name["EVENT"]
    event_offset = 0
    pairs, event_offset = _get_uleb(event, event_offset)
    slots, event_offset = _get_uleb(event, event_offset)
    presence_size = (pairs * slots + 7) // 8
    if len(event) - event_offset != presence_size:
        raise DirectDescriptionError("G1 lift EVENT stream length mismatch")
    presence = (
        np.unpackbits(np.frombuffer(event[event_offset:], dtype=np.uint8), bitorder="little")[: pairs * slots]
        .reshape(pairs, slots)
        .astype(bool)
    )
    centroid_values = _decode_signed(by_name["CENTROID"], int(presence.sum()) * 2)
    centroid_offset = 0
    shape = by_name["SHAPE"]
    shape_offset = 0
    previous_centroids: dict[int, np.ndarray] = {}
    active_object: dict[int, int] = {}
    generations = np.zeros(slots, dtype=np.int64)
    track_rows: dict[int, dict[str, object]] = {}
    knots: list[G1WorldsheetKnotV1] = []
    templates: dict[str, G1ShapeTemplateV1] = {}
    next_object_id = 0
    for pair_index in range(pairs):
        for slot_id in range(slots):
            if not presence[pair_index, slot_id]:
                previous_centroids.pop(slot_id, None)
                object_id = active_object.pop(slot_id, None)
                if object_id is not None:
                    track_rows[object_id]["death_pair_exclusive"] = pair_index
                continue
            if slot_id not in active_object:
                object_id = next_object_id
                next_object_id += 1
                active_object[slot_id] = object_id
                track_rows[object_id] = {
                    "object_id": object_id,
                    "slot_id": slot_id,
                    "generation": int(generations[slot_id]),
                    "birth_pair": pair_index,
                    "death_pair_exclusive": pairs,
                    "knot_indices": [],
                }
                generations[slot_id] += 1
            object_id = active_object[slot_id]
            delta_centroid = centroid_values[centroid_offset : centroid_offset + 2]
            centroid_offset += 2
            old_centroid = previous_centroids.get(slot_id)
            centroid = delta_centroid if old_centroid is None else old_centroid + delta_centroid
            previous_centroids[slot_id] = centroid
            header, shape_offset = _get_uleb(shape, shape_offset)
            vertex_count = header >> 1
            if header & 1:
                raise DirectDescriptionError(
                    "G1 lossless lift refuses delta-relative SHAPE; current v14/v15 payload must be absolute"
                )
            if not 1 <= vertex_count <= 4096:
                raise DirectDescriptionError("G1 lift vertex count is invalid")
            relative = np.empty((vertex_count, 2), dtype=np.int32)
            for vertex in range(vertex_count):
                x_value, shape_offset = _get_uleb(shape, shape_offset)
                y_value, shape_offset = _get_uleb(shape, shape_offset)
                relative[vertex] = (_unzigzag(x_value), _unzigzag(y_value))
            template_ref = _template_ref(relative)
            templates.setdefault(
                template_ref,
                G1ShapeTemplateV1(
                    template_ref=template_ref,
                    relative_vertices_xy=tuple((int(x), int(y)) for x, y in relative),
                ),
            )
            aspect_log, rotation = _shape_parameters(relative)
            knot_index = len(knots)
            knots.append(
                G1WorldsheetKnotV1(
                    object_id=object_id,
                    slot_id=slot_id,
                    pair_index=pair_index,
                    center_x=int(centroid[0]),
                    center_y=int(centroid[1]),
                    template_ref=template_ref,
                    aspect_log=aspect_log,
                    rotation_radians=rotation,
                )
            )
            cast_indices = track_rows[object_id]["knot_indices"]
            if not isinstance(cast_indices, list):  # defensive invariant for typed construction below
                raise DirectDescriptionError("G1 lift internal knot-index state is invalid")
            cast_indices.append(knot_index)
    if centroid_offset != centroid_values.size or shape_offset != len(shape):
        raise DirectDescriptionError("G1 lift streams have trailing values")
    tracks = tuple(
        G1WorldsheetTrackV1(
            object_id=int(row["object_id"]),
            slot_id=int(row["slot_id"]),
            generation=int(row["generation"]),
            birth_pair=int(row["birth_pair"]),
            death_pair_exclusive=int(row["death_pair_exclusive"]),
            knot_indices=tuple(int(value) for value in row["knot_indices"]),
        )
        for _, row in sorted(track_rows.items())
    )
    lift = G1WorldsheetParameterLiftV1(
        pair_count=pairs,
        max_slots=slots,
        tracks=tracks,
        knots=tuple(knots),
        templates=tuple(templates[key] for key in sorted(templates)),
        source_payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_payload_bytes=len(payload),
    )
    if encode_lifted_g1_movable_worldsheet(lift) != payload:
        raise DirectDescriptionError("G1 typed lift failed exact payload re-emission")
    return lift


def encode_lifted_g1_movable_worldsheet(lift: G1WorldsheetParameterLiftV1) -> bytes:
    """Re-emit EVENT/CENTROID/SHAPE solely from a typed lossless lift."""

    if not 1 <= lift.pair_count <= 600 or not 1 <= lift.max_slots <= 64:
        raise DirectDescriptionError("G1 lifted pair/slot bounds are invalid")
    template_by_ref = {row.template_ref: row for row in lift.templates}
    if len(template_by_ref) != len(lift.templates):
        raise DirectDescriptionError("G1 lifted template refs must be unique")
    by_pair_slot: dict[tuple[int, int], G1WorldsheetKnotV1] = {}
    track_by_object = {row.object_id: row for row in lift.tracks}
    if len(track_by_object) != len(lift.tracks):
        raise DirectDescriptionError("G1 lifted object IDs must be unique")
    for index, knot in enumerate(lift.knots):
        track = track_by_object.get(knot.object_id)
        if (
            track is None
            or knot.slot_id != track.slot_id
            or not track.birth_pair <= knot.pair_index < track.death_pair_exclusive
            or index not in track.knot_indices
        ):
            raise DirectDescriptionError("G1 lifted knot escaped its declared lifecycle")
        key = (knot.pair_index, knot.slot_id)
        if key in by_pair_slot:
            raise DirectDescriptionError("G1 lifted pair/slot owns multiple knots")
        if knot.template_ref not in template_by_ref:
            raise DirectDescriptionError("G1 lifted knot references an unknown template")
        by_pair_slot[key] = knot
    if sum(len(row.knot_indices) for row in lift.tracks) != len(lift.knots):
        raise DirectDescriptionError("G1 lifted knot ownership is not unique and complete")

    presence = np.zeros((lift.pair_count, lift.max_slots), dtype=bool)
    for pair_index, slot_id in by_pair_slot:
        if not (0 <= pair_index < lift.pair_count and 0 <= slot_id < lift.max_slots):
            raise DirectDescriptionError("G1 lifted knot address is out of bounds")
        presence[pair_index, slot_id] = True
    event = bytearray()
    _put_uleb(event, lift.pair_count)
    _put_uleb(event, lift.max_slots)
    event.extend(np.packbits(presence.reshape(-1), bitorder="little").tobytes())
    centroids: list[int] = []
    shape = bytearray()
    previous_centroids: dict[int, tuple[int, int]] = {}
    for pair_index in range(lift.pair_count):
        for slot_id in range(lift.max_slots):
            knot = by_pair_slot.get((pair_index, slot_id))
            if knot is None:
                previous_centroids.pop(slot_id, None)
                continue
            current = (knot.center_x, knot.center_y)
            old = previous_centroids.get(slot_id)
            centroids.extend(current if old is None else (current[0] - old[0], current[1] - old[1]))
            previous_centroids[slot_id] = current
            relative = template_by_ref[knot.template_ref].relative_vertices_xy
            _put_uleb(shape, len(relative) << 1)
            for x_value, y_value in relative:
                _put_uleb(shape, _zigzag(x_value))
                _put_uleb(shape, _zigzag(y_value))
    payload, _ = _encode_envelope(
        [("EVENT", bytes(event)), ("CENTROID", _encode_signed(centroids)), ("SHAPE", bytes(shape))]
    )
    return payload


__all__ = [
    "EPSILON_PIXELS",
    "G1MovableWorldsheetMetadata",
    "G1ShapeTemplateV1",
    "G1WorldsheetKnotV1",
    "G1WorldsheetParameterLiftV1",
    "G1WorldsheetTrackV1",
    "decode_g1_movable_worldsheet",
    "encode_g1_movable_worldsheet",
    "encode_lifted_g1_movable_worldsheet",
    "lift_g1_movable_worldsheet",
]
