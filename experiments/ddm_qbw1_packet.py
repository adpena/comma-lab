"""Frozen QBW1 v1 packet parser and generic quotient receiver.

This module implements the wire contract in
``.omx/research/SPEC_ddm_qbw1_packet_schema_v1_20260827.md``.  It contains no
video-derived constants or scorer state.  Encoder-side object extraction lives in the staged
builder; this file is deliberately receiver-complete and source-agnostic.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from collections import deque
from dataclasses import dataclass

import numpy as np

MODEL_MAGIC = b"QBM1"
RECORD_MAGIC = b"QBR1"
SCHEMA_VERSION = 1
CODER_RAW_DEFLATE = 1
RESET_SNAPSHOT = 1
HEIGHT = 384
WIDTH = 512
BASE_LABELS = (0, 2, 3, 4)
LABEL_TO_CODE = {label: code for code, label in enumerate(BASE_LABELS)}
SECTION_BASE_CRACK_CHAINS = 1
SECTION_REGION_SEEDS = 2
SECTION_LANE_DASH_EVENTS = 3
SECTION_IDS = (
    SECTION_BASE_CRACK_CHAINS,
    SECTION_REGION_SEEDS,
    SECTION_LANE_DASH_EVENTS,
)
DICTIONARY_CAPACITIES = (0, 4096, 16384, 32768)
MAX_INTERNAL_EDGES = HEIGHT * (WIDTH - 1) + (HEIGHT - 1) * WIDTH

_MODEL_HEADER = struct.Struct(">4sBBBBII")
_RECORD_HEADER = struct.Struct(">4sBBHHHHH32s")
_SECTION_HEADER = struct.Struct(">BBIII")


class QBW1FormatError(ValueError):
    """Raised when a QBW1 model, packet, or decoded object violates v1."""


@dataclass(frozen=True, slots=True)
class QBW1Model:
    dictionary: bytes
    level: int = 9

    def to_bytes(self) -> bytes:
        if self.level != 9:
            raise QBW1FormatError("QBW1 v1 fixes DEFLATE level at 9")
        if len(self.dictionary) not in DICTIONARY_CAPACITIES:
            raise QBW1FormatError("QBW1 v1 dictionary capacity is not registered")
        header = _MODEL_HEADER.pack(
            MODEL_MAGIC,
            SCHEMA_VERSION,
            CODER_RAW_DEFLATE,
            self.level,
            0,
            len(self.dictionary),
            zlib.crc32(self.dictionary) & 0xFFFFFFFF,
        )
        return header + self.dictionary

    @property
    def sha256(self) -> bytes:
        return hashlib.sha256(self.to_bytes()).digest()

    @classmethod
    def from_bytes(cls, payload: bytes) -> QBW1Model:
        if len(payload) < _MODEL_HEADER.size:
            raise QBW1FormatError("truncated QBM1 header")
        magic, version, coder, level, reserved, size, checksum = _MODEL_HEADER.unpack_from(payload)
        if magic != MODEL_MAGIC or version != SCHEMA_VERSION:
            raise QBW1FormatError("QBM1 magic/version mismatch")
        if coder != CODER_RAW_DEFLATE or level != 9 or reserved != 0:
            raise QBW1FormatError("unsupported QBM1 coder parameters")
        dictionary = payload[_MODEL_HEADER.size :]
        if len(dictionary) != size:
            raise QBW1FormatError("QBM1 dictionary length mismatch")
        if zlib.crc32(dictionary) & 0xFFFFFFFF != checksum:
            raise QBW1FormatError("QBM1 dictionary CRC mismatch")
        return cls(dictionary=dictionary, level=level)


@dataclass(frozen=True, slots=True)
class CrackStep:
    direction: int
    left_label: int
    right_label: int


@dataclass(frozen=True, slots=True)
class CrackChain:
    birth_rank: int
    steps: tuple[CrackStep, ...]


@dataclass(frozen=True, slots=True)
class LaneDashEvent:
    anchor_delta: int
    tangent_offset_q4: int
    normal_offset_q4: int
    major_half_extent_q4: int
    minor_half_extent_q4: int
    angle_q256: int


@dataclass(frozen=True, slots=True)
class DecodedRecord:
    pair_id: int
    chains: tuple[CrackChain, ...]
    seed_labels: tuple[int, ...]
    lane_events: tuple[LaneDashEvent, ...]
    raw_sections: tuple[tuple[int, bytes], ...]


@dataclass(frozen=True, slots=True)
class EdgeFact:
    start_rank: int
    end_rank: int
    left_label: int
    right_label: int

    @property
    def undirected_key(self) -> tuple[int, int]:
        return tuple(sorted((self.start_rank, self.end_rank)))


def encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise QBW1FormatError("unsigned varint cannot encode a negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def decode_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while offset < len(payload) and shift <= 63:
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if payload[start:offset] != encode_uvarint(value):
                raise QBW1FormatError("non-minimal unsigned varint")
            return value, offset
        shift += 7
    raise QBW1FormatError("truncated or overflowing unsigned varint")


def encode_svarint(value: int) -> bytes:
    zigzag = (value << 1) if value >= 0 else ((-value << 1) - 1)
    return encode_uvarint(zigzag)


def decode_svarint(payload: bytes, offset: int) -> tuple[int, int]:
    zigzag, offset = decode_uvarint(payload, offset)
    value = -(zigzag // 2) - 1 if zigzag & 1 else zigzag // 2
    return value, offset


def _compress(raw: bytes, model: QBW1Model) -> bytes:
    kwargs: dict[str, object] = {"level": model.level, "wbits": -15}
    if model.dictionary:
        kwargs["zdict"] = model.dictionary
    compressor = zlib.compressobj(**kwargs)
    return compressor.compress(raw) + compressor.flush(zlib.Z_FINISH)


def _decompress(coded: bytes, raw_len: int, model: QBW1Model) -> bytes:
    kwargs: dict[str, object] = {"wbits": -15}
    if model.dictionary:
        kwargs["zdict"] = model.dictionary
    try:
        decompressor = zlib.decompressobj(**kwargs)
        raw = decompressor.decompress(coded, raw_len + 1)
        raw += decompressor.flush()
    except zlib.error as exc:
        raise QBW1FormatError(f"DEFLATE refusal: {exc}") from exc
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise QBW1FormatError("DEFLATE stream termination mismatch")
    if len(raw) != raw_len:
        raise QBW1FormatError("decoded section length mismatch")
    return raw


def encode_chains(chains: tuple[CrackChain, ...]) -> bytes:
    out = bytearray(encode_uvarint(len(chains)))
    previous_birth = 0
    for chain in chains:
        if not chain.steps:
            raise QBW1FormatError("empty crack chain")
        out += encode_svarint(chain.birth_rank - previous_birth)
        previous_birth = chain.birth_rank
        out += encode_uvarint(len(chain.steps))
        for step in chain.steps:
            if step.direction not in range(4):
                raise QBW1FormatError("illegal crack direction")
            try:
                left = LABEL_TO_CODE[step.left_label]
                right = LABEL_TO_CODE[step.right_label]
            except KeyError as exc:
                raise QBW1FormatError("Lane is forbidden in base crack labels") from exc
            if left == right:
                raise QBW1FormatError("crack step cannot have identical side labels")
            out.append(step.direction | (left << 2) | (right << 4))
    return bytes(out)


def decode_chains(payload: bytes) -> tuple[CrackChain, ...]:
    count, offset = decode_uvarint(payload, 0)
    if count > MAX_INTERNAL_EDGES:
        raise QBW1FormatError("crack chain count exceeds the grid edge census")
    chains: list[CrackChain] = []
    previous_birth = 0
    total_steps = 0
    for _ in range(count):
        delta, offset = decode_svarint(payload, offset)
        birth = previous_birth + delta
        previous_birth = birth
        if birth < 0 or birth >= (HEIGHT + 1) * (WIDTH + 1):
            raise QBW1FormatError("crack birth outside lattice")
        step_count, offset = decode_uvarint(payload, offset)
        total_steps += step_count
        if (
            step_count == 0
            or total_steps > MAX_INTERNAL_EDGES
            or offset + step_count > len(payload)
        ):
            raise QBW1FormatError("invalid crack step count")
        steps: list[CrackStep] = []
        for byte in payload[offset : offset + step_count]:
            if byte & 0xC0:
                raise QBW1FormatError("reserved crack step bits are nonzero")
            direction = byte & 0x03
            left = BASE_LABELS[(byte >> 2) & 0x03]
            right = BASE_LABELS[(byte >> 4) & 0x03]
            if left == right:
                raise QBW1FormatError("decoded crack has identical side labels")
            steps.append(CrackStep(direction, left, right))
        offset += step_count
        chains.append(CrackChain(birth, tuple(steps)))
    if offset != len(payload):
        raise QBW1FormatError("trailing BASE_CRACK_CHAINS bytes")
    return tuple(chains)


def encode_seed_labels(labels: tuple[int, ...]) -> bytes:
    out = bytearray(encode_uvarint(len(labels)))
    packed = 0
    used = 0
    for label in labels:
        try:
            code = LABEL_TO_CODE[label]
        except KeyError as exc:
            raise QBW1FormatError("Lane is forbidden in base region seeds") from exc
        packed |= code << used
        used += 2
        if used == 8:
            out.append(packed)
            packed = 0
            used = 0
    if used:
        out.append(packed)
    return bytes(out)


def decode_seed_labels(payload: bytes) -> tuple[int, ...]:
    count, offset = decode_uvarint(payload, 0)
    if count > HEIGHT * WIDTH:
        raise QBW1FormatError("REGION_SEEDS count exceeds the pixel census")
    expected = (count + 3) // 4
    if len(payload) - offset != expected:
        raise QBW1FormatError("REGION_SEEDS packed length mismatch")
    labels: list[int] = []
    for byte in payload[offset:]:
        for shift in (0, 2, 4, 6):
            if len(labels) < count:
                labels.append(BASE_LABELS[(byte >> shift) & 0x03])
            elif byte >> shift:
                raise QBW1FormatError("nonzero REGION_SEEDS padding")
    return tuple(labels)


def encode_lane_events(events: tuple[LaneDashEvent, ...]) -> bytes:
    out = bytearray(encode_uvarint(len(events)))
    for event in events:
        if event.anchor_delta < 0:
            raise QBW1FormatError("Lane anchors must be monotone")
        if event.major_half_extent_q4 <= 0 or event.minor_half_extent_q4 <= 0:
            raise QBW1FormatError("Lane dash extents must be positive")
        if event.angle_q256 not in range(256):
            raise QBW1FormatError("Lane angle outside u8")
        out += encode_uvarint(event.anchor_delta)
        out += encode_svarint(event.tangent_offset_q4)
        out += encode_svarint(event.normal_offset_q4)
        out += encode_uvarint(event.major_half_extent_q4)
        out += encode_uvarint(event.minor_half_extent_q4)
        out.append(event.angle_q256)
    return bytes(out)


def decode_lane_events(payload: bytes) -> tuple[LaneDashEvent, ...]:
    count, offset = decode_uvarint(payload, 0)
    if count > HEIGHT * WIDTH:
        raise QBW1FormatError("Lane event count exceeds the pixel census")
    events: list[LaneDashEvent] = []
    for _ in range(count):
        anchor_delta, offset = decode_uvarint(payload, offset)
        tangent, offset = decode_svarint(payload, offset)
        normal, offset = decode_svarint(payload, offset)
        major, offset = decode_uvarint(payload, offset)
        minor, offset = decode_uvarint(payload, offset)
        if offset >= len(payload):
            raise QBW1FormatError("truncated Lane angle")
        angle = payload[offset]
        offset += 1
        event = LaneDashEvent(anchor_delta, tangent, normal, major, minor, angle)
        if major == 0 or minor == 0:
            raise QBW1FormatError("decoded Lane dash has zero extent")
        events.append(event)
    if offset != len(payload):
        raise QBW1FormatError("trailing LANE_DASH_EVENTS bytes")
    return tuple(events)


def encode_record(
    pair_id: int,
    model: QBW1Model,
    chains: tuple[CrackChain, ...],
    seed_labels: tuple[int, ...],
    lane_events: tuple[LaneDashEvent, ...],
) -> bytes:
    if pair_id not in range(600):
        raise QBW1FormatError("pair id outside 0..599")
    raw_sections = (
        (SECTION_BASE_CRACK_CHAINS, encode_chains(chains)),
        (SECTION_REGION_SEEDS, encode_seed_labels(seed_labels)),
        (SECTION_LANE_DASH_EVENTS, encode_lane_events(lane_events)),
    )
    out = bytearray(
        _RECORD_HEADER.pack(
            RECORD_MAGIC,
            SCHEMA_VERSION,
            RESET_SNAPSHOT,
            len(raw_sections),
            pair_id,
            HEIGHT,
            WIDTH,
            0,
            model.sha256,
        )
    )
    for section_id, raw in raw_sections:
        coded = _compress(raw, model)
        out += _SECTION_HEADER.pack(
            section_id,
            0,
            len(raw),
            len(coded),
            zlib.crc32(raw) & 0xFFFFFFFF,
        )
        out += coded
    return bytes(out)


def decode_record(payload: bytes, model: QBW1Model) -> DecodedRecord:
    if len(payload) < _RECORD_HEADER.size:
        raise QBW1FormatError("truncated QBR1 header")
    fields = _RECORD_HEADER.unpack_from(payload)
    magic, version, flags, count, pair_id, height, width, reserved, model_sha = fields
    if magic != RECORD_MAGIC or version != SCHEMA_VERSION or flags != RESET_SNAPSHOT:
        raise QBW1FormatError("QBR1 magic/version/flags mismatch")
    if count != len(SECTION_IDS) or pair_id not in range(600):
        raise QBW1FormatError("QBR1 section count or pair id mismatch")
    if height != HEIGHT or width != WIDTH or reserved != 0:
        raise QBW1FormatError("QBR1 geometry/reserved mismatch")
    if model_sha != model.sha256:
        raise QBW1FormatError("QBR1 model SHA mismatch")
    offset = _RECORD_HEADER.size
    sections: list[tuple[int, bytes]] = []
    for expected_id in SECTION_IDS:
        if offset + _SECTION_HEADER.size > len(payload):
            raise QBW1FormatError("truncated QBR1 section envelope")
        section_id, section_reserved, raw_len, coded_len, checksum = _SECTION_HEADER.unpack_from(
            payload, offset
        )
        offset += _SECTION_HEADER.size
        if section_id != expected_id or section_reserved != 0:
            raise QBW1FormatError("QBR1 section order/reserved mismatch")
        end = offset + coded_len
        if end > len(payload):
            raise QBW1FormatError("truncated QBR1 coded section")
        raw = _decompress(payload[offset:end], raw_len, model)
        if zlib.crc32(raw) & 0xFFFFFFFF != checksum:
            raise QBW1FormatError("QBR1 raw-section CRC mismatch")
        sections.append((section_id, raw))
        offset = end
    if offset != len(payload):
        raise QBW1FormatError("trailing QBR1 bytes")
    raw_by_id = dict(sections)
    return DecodedRecord(
        pair_id=pair_id,
        chains=decode_chains(raw_by_id[SECTION_BASE_CRACK_CHAINS]),
        seed_labels=decode_seed_labels(raw_by_id[SECTION_REGION_SEEDS]),
        lane_events=decode_lane_events(raw_by_id[SECTION_LANE_DASH_EVENTS]),
        raw_sections=tuple(sections),
    )


def vertex_rank(y: int, x: int) -> int:
    return y * (WIDTH + 1) + x


def vertex_coords(rank: int) -> tuple[int, int]:
    return divmod(rank, WIDTH + 1)


def step_endpoint(rank: int, direction: int) -> int:
    y, x = vertex_coords(rank)
    dy_dx = ((-1, 0), (0, 1), (1, 0), (0, -1))
    try:
        dy, dx = dy_dx[direction]
    except IndexError as exc:
        raise QBW1FormatError("invalid cardinal direction") from exc
    y2, x2 = y + dy, x + dx
    if not (0 <= y2 <= HEIGHT and 0 <= x2 <= WIDTH):
        raise QBW1FormatError("crack step leaves the lattice")
    return vertex_rank(y2, x2)


def expand_edges(chains: tuple[CrackChain, ...]) -> tuple[EdgeFact, ...]:
    edges: list[EdgeFact] = []
    seen: set[tuple[int, int]] = set()
    for chain in chains:
        current = chain.birth_rank
        for step in chain.steps:
            endpoint = step_endpoint(current, step.direction)
            edge = EdgeFact(current, endpoint, step.left_label, step.right_label)
            if edge.undirected_key in seen:
                raise QBW1FormatError("duplicate crack edge")
            seen.add(edge.undirected_key)
            edges.append(edge)
            current = endpoint
    return tuple(edges)


def _edge_barriers(edges: tuple[EdgeFact, ...]) -> tuple[np.ndarray, np.ndarray]:
    vertical = np.zeros((HEIGHT, WIDTH - 1), dtype=np.bool_)
    horizontal = np.zeros((HEIGHT - 1, WIDTH), dtype=np.bool_)
    for edge in edges:
        y0, x0 = vertex_coords(edge.start_rank)
        y1, x1 = vertex_coords(edge.end_rank)
        if x0 == x1 and abs(y1 - y0) == 1:
            y = min(y0, y1)
            if not 0 < x0 < WIDTH:
                raise QBW1FormatError("internal crack encoded on outer vertical frame")
            vertical[y, x0 - 1] = True
        elif y0 == y1 and abs(x1 - x0) == 1:
            x = min(x0, x1)
            if not 0 < y0 < HEIGHT:
                raise QBW1FormatError("internal crack encoded on outer horizontal frame")
            horizontal[y0 - 1, x] = True
        else:
            raise QBW1FormatError("non-cardinal crack edge")
    return vertical, horizontal


def integrate_cells(edges: tuple[EdgeFact, ...]) -> tuple[np.ndarray, int]:
    """Integrate the crack complement into canonical row-major cell ids."""

    vertical, horizontal = _edge_barriers(edges)
    cells = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    cell_id = 0
    for seed_y in range(HEIGHT):
        for seed_x in range(WIDTH):
            if cells[seed_y, seed_x] >= 0:
                continue
            cells[seed_y, seed_x] = cell_id
            queue: deque[tuple[int, int]] = deque(((seed_y, seed_x),))
            while queue:
                y, x = queue.popleft()
                if y > 0 and not horizontal[y - 1, x] and cells[y - 1, x] < 0:
                    cells[y - 1, x] = cell_id
                    queue.append((y - 1, x))
                if x + 1 < WIDTH and not vertical[y, x] and cells[y, x + 1] < 0:
                    cells[y, x + 1] = cell_id
                    queue.append((y, x + 1))
                if y + 1 < HEIGHT and not horizontal[y, x] and cells[y + 1, x] < 0:
                    cells[y + 1, x] = cell_id
                    queue.append((y + 1, x))
                if x > 0 and not vertical[y, x - 1] and cells[y, x - 1] < 0:
                    cells[y, x - 1] = cell_id
                    queue.append((y, x - 1))
            cell_id += 1
    return cells, cell_id


def assign_seed_labels(cells: np.ndarray, seed_labels: tuple[int, ...]) -> np.ndarray:
    count = int(cells.max()) + 1
    if count != len(seed_labels):
        raise QBW1FormatError("region seed count does not match integrated cells")
    labels = np.asarray(seed_labels, dtype=np.uint8)
    return labels[cells]


def derive_interface_keys(field: np.ndarray) -> set[tuple[int, int]]:
    if field.shape != (HEIGHT, WIDTH):
        raise QBW1FormatError("base field geometry mismatch")
    keys: set[tuple[int, int]] = set()
    ys, xs = np.nonzero(field[:, 1:] != field[:, :-1])
    for y, x_left in zip(ys.tolist(), xs.tolist(), strict=True):
        x = x_left + 1
        keys.add(tuple(sorted((vertex_rank(y, x), vertex_rank(y + 1, x)))))
    ys, xs = np.nonzero(field[1:, :] != field[:-1, :])
    for y_top, x in zip(ys.tolist(), xs.tolist(), strict=True):
        y = y_top + 1
        keys.add(tuple(sorted((vertex_rank(y, x), vertex_rank(y, x + 1)))))
    return keys


def verify_closed_chain_consistency(edges: tuple[EdgeFact, ...], field: np.ndarray) -> None:
    decoded = {edge.undirected_key for edge in edges}
    derived = derive_interface_keys(field)
    if decoded != derived:
        missing = len(decoded - derived)
        extra = len(derived - decoded)
        raise QBW1FormatError(f"closed-chain mismatch: missing={missing}, extra={extra}")
    for edge in edges:
        y0, x0 = vertex_coords(edge.start_rank)
        y1, x1 = vertex_coords(edge.end_rank)
        dy, dx = y1 - y0, x1 - x0
        midpoint_y = (y0 + y1) / 2.0
        midpoint_x = (x0 + x1) / 2.0
        left_y = math.floor(midpoint_y - 0.25 * dx)
        left_x = math.floor(midpoint_x + 0.25 * dy)
        right_y = math.floor(midpoint_y + 0.25 * dx)
        right_x = math.floor(midpoint_x - 0.25 * dy)
        if not (
            0 <= left_y < HEIGHT
            and 0 <= left_x < WIDTH
            and 0 <= right_y < HEIGHT
            and 0 <= right_x < WIDTH
        ):
            raise QBW1FormatError("crack side sample leaves the image")
        if (
            int(field[left_y, left_x]) != edge.left_label
            or int(field[right_y, right_x]) != edge.right_label
        ):
            raise QBW1FormatError("oriented crack side labels disagree with integrated cells")


def road_boundary_basis(edges: tuple[EdgeFact, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical Road-edge midpoints, tangents, and inward normals."""

    road_edges = sorted(
        (edge for edge in edges if 0 in (edge.left_label, edge.right_label)),
        key=lambda edge: edge.undirected_key,
    )
    if not road_edges:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    midpoints: list[tuple[float, float]] = []
    tangents: list[tuple[float, float]] = []
    normals: list[tuple[float, float]] = []
    for edge in road_edges:
        y0, x0 = vertex_coords(edge.start_rank)
        y1, x1 = vertex_coords(edge.end_rank)
        dy, dx = y1 - y0, x1 - x0
        length = math.hypot(dy, dx)
        ty, tx = dy / length, dx / length
        # Left normal in image (y, x) coordinates.  Flip if Road is on the right.
        ny, nx = -tx, ty
        if edge.right_label == 0:
            ny, nx = -ny, -nx
        midpoints.append(((y0 + y1) / 2.0, (x0 + x1) / 2.0))
        tangents.append((ty, tx))
        normals.append((ny, nx))
    return (
        np.asarray(midpoints, dtype=np.float32),
        np.asarray(tangents, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
    )


def rasterize_lane_events(
    events: tuple[LaneDashEvent, ...], edges: tuple[EdgeFact, ...]
) -> np.ndarray:
    midpoints, _tangents, _normals = road_boundary_basis(edges)
    lane = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    anchor = 0
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    for event in events:
        anchor += event.anchor_delta
        if anchor >= len(midpoints):
            raise QBW1FormatError("Lane event anchor exceeds Road graph")
        tangent = _tangents[anchor]
        normal = _normals[anchor]
        center = (
            midpoints[anchor]
            + tangent * (event.tangent_offset_q4 / 4.0)
            + normal * (event.normal_offset_q4 / 4.0)
        )
        theta = event.angle_q256 * (2.0 * math.pi / 256.0)
        axis_major = np.asarray((math.sin(theta), math.cos(theta)), dtype=np.float32)
        axis_minor = np.asarray((-axis_major[1], axis_major[0]), dtype=np.float32)
        dy = yy - float(center[0])
        dx = xx - float(center[1])
        major = (dy * axis_major[0] + dx * axis_major[1]) / (
            event.major_half_extent_q4 / 4.0
        )
        minor = (dy * axis_minor[0] + dx * axis_minor[1]) / (
            event.minor_half_extent_q4 / 4.0
        )
        lane |= major * major + minor * minor <= 1.0
    return lane


def decode_receiver(payload: bytes, model: QBW1Model) -> dict[str, object]:
    record = decode_record(payload, model)
    edges = expand_edges(record.chains)
    cells, _ = integrate_cells(edges)
    base_field = assign_seed_labels(cells, record.seed_labels)
    verify_closed_chain_consistency(edges, base_field)
    lane = rasterize_lane_events(record.lane_events, edges)
    categorical = base_field.copy()
    categorical[(base_field == 0) & lane] = 1
    return {
        "record": record,
        "edges": edges,
        "cells": cells,
        "base_field": base_field,
        "lane_mask": lane,
        "categorical_field": categorical,
    }


def section_spans(payload: bytes) -> tuple[tuple[str, int, int], ...]:
    """Locate counted mutation surfaces without interpreting their contents."""

    if len(payload) < _RECORD_HEADER.size:
        raise QBW1FormatError("truncated record")
    spans: list[tuple[str, int, int]] = [("record_framing", 0, _RECORD_HEADER.size)]
    offset = _RECORD_HEADER.size
    for expected_id in SECTION_IDS:
        if offset + _SECTION_HEADER.size > len(payload):
            raise QBW1FormatError("truncated section envelope")
        section_id, _, _, coded_len, _ = _SECTION_HEADER.unpack_from(payload, offset)
        if section_id != expected_id:
            raise QBW1FormatError("section order mismatch")
        start = offset + _SECTION_HEADER.size
        end = start + coded_len
        if end > len(payload):
            raise QBW1FormatError("truncated coded section")
        spans.append((f"section_{section_id}", start, end))
        offset = end
    if offset != len(payload):
        raise QBW1FormatError("trailing record bytes")
    return tuple(spans)
