#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LTG1 all-n600 Lane topology/event-generator arithmetic-floor measurement.

This is a scorer-free exact-rate experiment.  It consumes the retained D3 Lane
mask, reuses the #425 dash tracker, measures visibility conditioning with the
in-tree adaptive range coder, and races exact component-shape packets.  Every
materialized payload is persisted below ``--out-root`` and every shape packet
is decoded back to the exact 600-frame Lane mask before it can enter the result.

The shape grammar is component-native rather than a generic per-correction
price: each connected component is represented by an integer-pixel parametric
trunk (two- or three-knot piecewise-linear outer boundaries) plus an exact
adaptive contour-coded XOR residual.  Endpoint precision is selected by a real
q in {1,2,4,8,16} coder race.  The integer pixel lattice is the finest useful
precision; coarser candidates are admitted only if their retained exact packet
is smaller.

No score, scorer, Modal, MPS, or upstream mutation occurs here.  Evidence axis:
``[macOS-CPU scorer-free exact rate and receiver measurement]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import find_objects
from scipy.ndimage import label as cc_label

REPO = Path(__file__).resolve().parents[1]
for search_path in (REPO, REPO / "src", REPO / "tools"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from measure_contour_string_flip_coding import (
    AdaptiveStream,
    AdaptiveStreamDecoder,
    contour_decode_frames,
    contour_encode_frames,
)

from tac.boundary_math.dash_phase_carrier import (
    DashPhaseConfig,
    _advect_points_rc,
    _cross_xi,
    decode_dash_phase_carrier,
    encode_dash_phase_carrier,
)
from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    xi_from_pose_calibration,
)

AXIS = "[macOS-CPU scorer-free exact rate and receiver measurement]"
N_FRAMES = 600
HEIGHT = 384
WIDTH = 512
LANE_MASK_BYTES = N_FRAMES * HEIGHT * WIDTH // 8
LANE_MASK_SHA256 = "6ca82a7883411d0eb27addac7dcf662e84d2f9cc66404c299da2e15761c0e0cf"
SOURCE_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
QUOTIENT_FIELD_SHA256 = "deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07"
GF1_BYTES = 36_044
CARRIAGE_BAR_BYTES = 21_699
VISIBILITY_DEFICIT_BYTES = 8_259
RECALLED_DASH_SECTION_BYTES = 29_958
MIN_FREE_BYTES = 2 * 1024**3
SHAPE_MAGIC = b"LTGS1\x00"
EVENT_MAGIC = b"LTGE1\x00"
FLOOR_MAGIC = b"LTGF1\x00"
SHAPE_STREAM_NAMES = ("counts", "anchor", "chain", "cls")


class LTG1Error(RuntimeError):
    """Fail-closed LTG1 input, payload, or identity error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def record_is_intact(record: dict[str, Any]) -> bool:
    path = Path(record["path"])
    return path.is_file() and path.stat().st_size == int(record["bytes"]) and sha256_file(path) == record["sha256"]


def uvarint(value: int) -> bytes:
    if value < 0:
        raise LTG1Error(f"uvarint received negative value {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def read_uvarint(blob: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(blob) or shift > 63:
            raise LTG1Error("truncated or oversized uvarint")
        byte = blob[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def zigzag(value: int) -> int:
    return 2 * value if value >= 0 else -2 * value - 1


def adaptive_bytes_encode(raw: bytes) -> bytes:
    """Real adaptive range code for a canonical byte grammar."""
    stream = AdaptiveStream(256)
    previous_bucket = 16
    for position, byte in enumerate(raw):
        context = (position & 7) * 17 + previous_bucket
        stream.encode(byte, context)
        previous_bucket = byte >> 4
    return stream.finish()


def adaptive_bytes_decode(payload: bytes, raw_length: int) -> bytes:
    decoder = AdaptiveStreamDecoder(payload, 256)
    out = bytearray()
    previous_bucket = 16
    for position in range(raw_length):
        context = (position & 7) * 17 + previous_bucket
        byte = decoder.decode(context)
        out.append(byte)
        previous_bucket = byte >> 4
    return bytes(out)


@dataclass(frozen=True)
class Component:
    frame: int
    local_id: int
    rows: np.ndarray
    cols: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class TrunkDescriptor:
    orientation: int  # 0 row-parameterized, 1 col-parameterized
    axis_start: int
    axis_length: int
    knot_lr: tuple[tuple[int, int], ...]


def load_lane_mask(path: Path) -> np.ndarray:
    if path.stat().st_size != LANE_MASK_BYTES:
        raise LTG1Error(f"Lane packbits size {path.stat().st_size} != {LANE_MASK_BYTES}")
    digest = sha256_file(path)
    if digest != LANE_MASK_SHA256:
        raise LTG1Error(f"Lane packbits SHA {digest} != pinned {LANE_MASK_SHA256}")
    raw = np.fromfile(path, dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little").reshape(N_FRAMES, HEIGHT, WIDTH).astype(bool)


def materialize_source_receiver(
    *, lane_mask: np.ndarray, source_path: Path, quotient_path: Path, output_path: Path
) -> tuple[dict[str, Any], int]:
    expected_bytes = N_FRAMES * HEIGHT * WIDTH
    for label, path, expected_sha in (
        ("source", source_path, SOURCE_FIELD_SHA256),
        ("quotient", quotient_path, QUOTIENT_FIELD_SHA256),
    ):
        if path.stat().st_size != expected_bytes:
            raise LTG1Error(f"{label} field size {path.stat().st_size} != {expected_bytes}")
        digest = sha256_file(path)
        if digest != expected_sha:
            raise LTG1Error(f"{label} field SHA {digest} != pinned {expected_sha}")

    source = np.memmap(source_path, dtype=np.uint8, mode="r", shape=(expected_bytes,))
    quotient = np.memmap(quotient_path, dtype=np.uint8, mode="r", shape=(expected_bytes,))
    flat_mask = lane_mask.reshape(-1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
    mismatches = 0
    chunk_size = 4 * 1024 * 1024
    with partial.open("wb") as handle:
        for start in range(0, expected_bytes, chunk_size):
            stop = min(expected_bytes, start + chunk_size)
            restored = np.array(quotient[start:stop], copy=True)
            restored[flat_mask[start:stop]] = 1
            mismatches += int(np.count_nonzero(restored != source[start:stop]))
            handle.write(restored.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output_path)
    record = file_record(output_path)
    if mismatches or record["sha256"] != SOURCE_FIELD_SHA256:
        raise LTG1Error(f"Lane-over-quotient receiver mismatch={mismatches}, SHA={record['sha256']}")
    return record, mismatches


def extract_components(mask: np.ndarray) -> list[list[Component]]:
    structure = np.ones((3, 3), dtype=np.uint8)
    frames: list[list[Component]] = []
    for frame_index, frame in enumerate(mask):
        labels, count = cc_label(frame, structure=structure)
        objects = find_objects(labels, max_label=count)
        frame_components: list[Component] = []
        for label_index, slices in enumerate(objects, start=1):
            if slices is None:
                raise LTG1Error("connected-component label has no slice")
            local = labels[slices] == label_index
            local_rows, local_cols = np.nonzero(local)
            rows = local_rows.astype(np.int32) + int(slices[0].start)
            cols = local_cols.astype(np.int32) + int(slices[1].start)
            order = np.argsort(rows.astype(np.int64) * WIDTH + cols, kind="stable")
            rows = rows[order]
            cols = cols[order]
            frame_components.append(
                Component(
                    frame=frame_index,
                    local_id=label_index - 1,
                    rows=rows,
                    cols=cols,
                    area=int(rows.size),
                    bbox=(
                        int(rows.min()),
                        int(rows.max()) + 1,
                        int(cols.min()),
                        int(cols.max()) + 1,
                    ),
                )
            )
        frames.append(frame_components)
    return frames


def component_runs(component: Component, orientation: int) -> list[list[tuple[int, int]]]:
    primary = component.rows if orientation == 0 else component.cols
    secondary = component.cols if orientation == 0 else component.rows
    start = int(primary.min())
    stop = int(primary.max()) + 1
    runs_by_slice: list[list[tuple[int, int]]] = []
    for axis in range(start, stop):
        values = np.sort(secondary[primary == axis])
        if values.size == 0:
            raise LTG1Error("8-connected component has an empty interior axis slice")
        breaks = np.where(np.diff(values) > 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        stops = np.concatenate((breaks + 1, [values.size]))
        runs_by_slice.append(
            [(int(values[left]), int(values[right - 1])) for left, right in zip(starts, stops, strict=True)]
        )
    return runs_by_slice


def knot_indices(axis_length: int, knot_count: int) -> tuple[int, ...]:
    if knot_count == 2 or axis_length <= 2:
        return (0, max(0, axis_length - 1))
    if knot_count == 3:
        return (0, (axis_length - 1) // 2, axis_length - 1)
    raise LTG1Error(f"unsupported knot count {knot_count}")


def quantize_nonnegative(value: int, quantum: int) -> int:
    return ((int(value) + quantum // 2) // quantum) * quantum


def make_descriptor(component: Component, orientation: int, quantum: int, requested_knots: int) -> TrunkDescriptor:
    runs = component_runs(component, orientation)
    primary = component.rows if orientation == 0 else component.cols
    indices = knot_indices(len(runs), requested_knots)
    knot_lr = []
    for index in indices:
        left = min(run[0] for run in runs[index])
        right = max(run[1] for run in runs[index])
        left_q = quantize_nonnegative(left, quantum)
        right_q = quantize_nonnegative(right, quantum)
        knot_lr.append((min(left_q, right_q), max(left_q, right_q)))
    return TrunkDescriptor(
        orientation=orientation,
        axis_start=int(primary.min()),
        axis_length=len(runs),
        knot_lr=tuple(knot_lr),
    )


def interpolate_knots(values: Sequence[int], axis_length: int) -> np.ndarray:
    indices = knot_indices(axis_length, len(values))
    out = np.empty(axis_length, dtype=np.int32)
    for segment in range(len(indices) - 1):
        lo = indices[segment]
        hi = indices[segment + 1]
        denominator = max(1, hi - lo)
        for index in range(lo, hi + 1):
            offset = index - lo
            numerator = values[segment] * (denominator - offset) + values[segment + 1] * offset
            out[index] = int((numerator + denominator // 2) // denominator)
    return out


def descriptor_pixels(descriptor: TrunkDescriptor) -> tuple[np.ndarray, np.ndarray]:
    left = interpolate_knots([item[0] for item in descriptor.knot_lr], descriptor.axis_length)
    right = interpolate_knots([item[1] for item in descriptor.knot_lr], descriptor.axis_length)
    rows: list[int] = []
    cols: list[int] = []
    for offset, (left_value, right_value) in enumerate(zip(left.tolist(), right.tolist(), strict=True)):
        secondary_lo = max(0, min(left_value, right_value))
        secondary_hi = min((WIDTH if descriptor.orientation == 0 else HEIGHT) - 1, max(left_value, right_value))
        primary = descriptor.axis_start + offset
        if descriptor.orientation == 0:
            if not 0 <= primary < HEIGHT:
                continue
            for secondary in range(secondary_lo, secondary_hi + 1):
                rows.append(primary)
                cols.append(secondary)
        else:
            if not 0 <= primary < WIDTH:
                continue
            for secondary in range(secondary_lo, secondary_hi + 1):
                rows.append(secondary)
                cols.append(primary)
    return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)


def descriptor_bytes(descriptor: TrunkDescriptor) -> bytes:
    out = bytearray([descriptor.orientation])
    out += uvarint(descriptor.axis_start)
    out += uvarint(descriptor.axis_length)
    for left, right in descriptor.knot_lr:
        out += uvarint(left)
        out += uvarint(right)
    return bytes(out)


def descriptor_local_xor(component: Component, descriptor: TrunkDescriptor) -> int:
    row0, row1, col0, col1 = component.bbox
    height = row1 - row0
    width = col1 - col0
    exact = np.zeros((height, width), dtype=bool)
    exact[component.rows - row0, component.cols - col0] = True
    trunk = np.zeros_like(exact)
    rows, cols = descriptor_pixels(descriptor)
    inside = (rows >= row0) & (rows < row1) & (cols >= col0) & (cols < col1)
    trunk[rows[inside] - row0, cols[inside] - col0] = True
    outside = int(rows.size - int(inside.sum()))
    return int(np.logical_xor(exact, trunk).sum()) + outside


def select_descriptor(component: Component, quantum: int, requested_knots: int) -> tuple[TrunkDescriptor, int]:
    rows = make_descriptor(component, 0, quantum, requested_knots)
    cols = make_descriptor(component, 1, quantum, requested_knots)
    row_score = (descriptor_local_xor(component, rows), len(descriptor_bytes(rows)), 0)
    col_score = (descriptor_local_xor(component, cols), len(descriptor_bytes(cols)), 1)
    if row_score <= col_score:
        return rows, row_score[0]
    return cols, col_score[0]


def make_trunk(
    components: Sequence[Sequence[Component]], quantum: int, requested_knots: int
) -> tuple[bytes, list[np.ndarray], list[dict[str, Any]]]:
    raw = bytearray()
    trunk_frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for frame_index, frame_components in enumerate(components):
        raw += uvarint(len(frame_components))
        trunk = np.zeros((HEIGHT, WIDTH), dtype=bool)
        for component in frame_components:
            descriptor, local_xor = select_descriptor(component, quantum, requested_knots)
            record = descriptor_bytes(descriptor)
            raw += record
            rr, cc = descriptor_pixels(descriptor)
            inside = (rr >= 0) & (rr < HEIGHT) & (cc >= 0) & (cc < WIDTH)
            trunk[rr[inside], cc[inside]] = True
            rows.append(
                {
                    "frame": frame_index,
                    "component": component.local_id,
                    "area": component.area,
                    "bbox": list(component.bbox),
                    "orientation": "row" if descriptor.orientation == 0 else "col",
                    "axis_length": descriptor.axis_length,
                    "param_raw_bytes": len(record),
                    "local_xor_pixels": local_xor,
                }
            )
        trunk_frames.append(trunk)
    return bytes(raw), trunk_frames, rows


def decode_trunk(raw: bytes, knot_count: int) -> list[np.ndarray]:
    offset = 0
    frames: list[np.ndarray] = []
    for _frame in range(N_FRAMES):
        component_count, offset = read_uvarint(raw, offset)
        trunk = np.zeros((HEIGHT, WIDTH), dtype=bool)
        for _ in range(component_count):
            if offset >= len(raw):
                raise LTG1Error("truncated trunk descriptor")
            orientation = raw[offset]
            offset += 1
            axis_start, offset = read_uvarint(raw, offset)
            axis_length, offset = read_uvarint(raw, offset)
            actual_knots = len(knot_indices(axis_length, knot_count))
            pairs = []
            for _ in range(actual_knots):
                left, offset = read_uvarint(raw, offset)
                right, offset = read_uvarint(raw, offset)
                pairs.append((left, right))
            descriptor = TrunkDescriptor(orientation, axis_start, axis_length, tuple(pairs))
            rr, cc = descriptor_pixels(descriptor)
            inside = (rr >= 0) & (rr < HEIGHT) & (cc >= 0) & (cc < WIDTH)
            trunk[rr[inside], cc[inside]] = True
        frames.append(trunk)
    if offset != len(raw):
        raise LTG1Error(f"trailing trunk bytes: consumed {offset} of {len(raw)}")
    return frames


def pack_shape(
    *,
    quantum: int,
    knot_count: int,
    raw_params: bytes,
    params_payload: bytes,
    contour_streams: dict[str, bytes],
) -> bytes:
    header = {
        "schema": "ddm_ltg1.shape.v1",
        "n_frames": N_FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "quantum": quantum,
        "knot_count": knot_count,
        "raw_params_bytes": len(raw_params),
        "params_payload_bytes": len(params_payload),
        "stream_lengths": {name: len(contour_streams[name]) for name in SHAPE_STREAM_NAMES},
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    packet = bytearray(SHAPE_MAGIC)
    packet += struct.pack("<I", len(header_bytes))
    packet += header_bytes
    packet += params_payload
    for name in SHAPE_STREAM_NAMES:
        packet += contour_streams[name]
    return bytes(packet)


def unpack_shape(packet: bytes) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    if not packet.startswith(SHAPE_MAGIC):
        raise LTG1Error("bad shape packet magic")
    header_length = struct.unpack("<I", packet[len(SHAPE_MAGIC) : len(SHAPE_MAGIC) + 4])[0]
    offset = len(SHAPE_MAGIC) + 4
    header = json.loads(packet[offset : offset + header_length])
    offset += header_length
    params_length = int(header["params_payload_bytes"])
    params = packet[offset : offset + params_length]
    offset += params_length
    streams = {}
    for name in SHAPE_STREAM_NAMES:
        length = int(header["stream_lengths"][name])
        streams[name] = packet[offset : offset + length]
        offset += length
    if offset != len(packet):
        raise LTG1Error("shape packet has trailing bytes")
    return header, params, streams


def decode_shape(packet: bytes) -> np.ndarray:
    header, params_payload, streams = unpack_shape(packet)
    raw_params = adaptive_bytes_decode(params_payload, int(header["raw_params_bytes"]))
    if adaptive_bytes_encode(raw_params) != params_payload:
        raise LTG1Error("shape params are noncanonical under adaptive re-encode")
    trunks = decode_trunk(raw_params, int(header["knot_count"]))
    residuals, _classes = contour_decode_frames(streams, N_FRAMES, HEIGHT, WIDTH)
    return np.stack([np.logical_xor(trunk, residual) for trunk, residual in zip(trunks, residuals, strict=True)])


def build_shape_candidate(
    lane_mask: np.ndarray,
    components: Sequence[Sequence[Component]],
    quantum: int,
    knot_count: int,
) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    if quantum == 0:
        raw_params = b"".join(uvarint(0) for _ in range(N_FRAMES))
        trunks = [np.zeros((HEIGHT, WIDTH), dtype=bool) for _ in range(N_FRAMES)]
        component_rows: list[dict[str, Any]] = []
    else:
        raw_params, trunks, component_rows = make_trunk(components, quantum, knot_count)
    residuals = [np.logical_xor(lane_mask[index], trunks[index]) for index in range(N_FRAMES)]
    zero_classes = [np.zeros((HEIGHT, WIDTH), dtype=np.int64) for _ in range(N_FRAMES)]
    contour = contour_encode_frames(residuals, zero_classes)
    params_payload = adaptive_bytes_encode(raw_params)
    packet = pack_shape(
        quantum=quantum,
        knot_count=knot_count,
        raw_params=raw_params,
        params_payload=params_payload,
        contour_streams=contour["streams"],
    )
    decoded = decode_shape(packet)
    mismatch = int(np.count_nonzero(decoded != lane_mask))
    if mismatch:
        raise LTG1Error(f"shape candidate q={quantum} knots={knot_count} has {mismatch} mismatches")
    header, _params, _streams = unpack_shape(packet)
    row = {
        "candidate": "direct_contour" if quantum == 0 else f"trunk_q{quantum}_k{knot_count}",
        "quantum_px": quantum,
        "knot_count": knot_count,
        "packet_bytes": len(packet),
        "packet_sha256": sha256_bytes(packet),
        "container_overhead_bytes": len(packet)
        - len(params_payload)
        - sum(len(value) for value in contour["streams"].values()),
        "params_raw_bytes": len(raw_params),
        "params_coded_bytes": len(params_payload),
        "residual_pixels": int(sum(int(item.sum()) for item in residuals)),
        "residual_components": contour["n_components"],
        "residual_stream_bytes": contour["stream_bytes"],
        "identity_mismatches": mismatch,
        "identity": "EXACT",
        "header": header,
    }
    return packet, row, component_rows


def bucket_age(value: int) -> int:
    if value <= 1:
        return 0
    if value <= 3:
        return 1
    if value <= 7:
        return 2
    if value <= 15:
        return 3
    return 4


def bucket_area(value: int) -> int:
    if value <= 2:
        return 0
    if value <= 7:
        return 1
    if value <= 31:
        return 2
    if value <= 127:
        return 3
    return 4


def row_bucket(row: float) -> int:
    return min(7, max(0, int(float(row) * 8.0 / HEIGHT)))


def event_context(variant: str, *, age: int, area: int, row: float, motion_sign: int, dormant_gap: int = 0) -> int:
    if variant == "unconditioned":
        return 0
    age_bin = bucket_age(age)
    if variant == "lifetime":
        return age_bin
    row_bin = row_bucket(row)
    if variant == "horizon_lifetime":
        return row_bin * 5 + age_bin
    if variant == "ego_phase":
        return (row_bin * 3 + motion_sign) * 5 + age_bin
    if variant == "joint":
        return (((row_bin * 3 + motion_sign) * 5 + age_bin) * 5 + bucket_area(area)) * 5 + bucket_age(dormant_gap)
    raise LTG1Error(f"unknown event predictor {variant}")


def varint_symbols(value: int, base_context: int) -> list[tuple[int, int]]:
    return [(byte, base_context * 3 + min(index, 2)) for index, byte in enumerate(uvarint(value))]


def build_event_traces(
    decoded_frames: Sequence[Sequence[Any]],
    xi: np.ndarray,
    variant: str,
    geom: GroundHomographyGeom,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    traces: dict[str, list[tuple[int, int]]] = {
        "survival": [],
        "new_count": [],
        "new_kind": [],
        "track_ref": [],
        "topology": [],
    }
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    prior: dict[int, Any] = {}
    merges = splits = complex_events = births = rebirths = deaths = survives = 0

    for frame_index, frame in enumerate(decoded_frames):
        current = {int(item.track_id): item for item in frame}
        if frame_index == 0:
            first_seen.update(dict.fromkeys(current, 0))
            last_seen.update(dict.fromkeys(current, 0))
            prior = current
            traces["new_count"].extend(varint_symbols(len(current), 0))
            for _track_id in sorted(current):
                traces["new_kind"].append((0, 0))
                births += 1
            continue

        cross = _cross_xi(xi, frame_index, "interp")
        prior_ids = sorted(prior)
        prior_points = np.asarray([prior[track_id].centroid_rc for track_id in prior_ids], dtype=np.float64)
        predicted = _advect_points_rc(prior_points, cross, geom) if prior_ids else prior_points
        motion_sign_by_id: dict[int, int] = {}
        for index, track_id in enumerate(prior_ids):
            delta = float(predicted[index, 0] - prior[track_id].centroid_rc[0])
            motion_sign_by_id[track_id] = 0 if delta < -0.25 else 2 if delta > 0.25 else 1

        for track_id in prior_ids:
            item = prior[track_id]
            alive = int(track_id in current)
            context = event_context(
                variant,
                age=frame_index - first_seen.get(track_id, frame_index) + 1,
                area=int(item.area),
                row=float(item.centroid_rc[0]),
                motion_sign=motion_sign_by_id[track_id],
            )
            traces["survival"].append((alive, context))
            if alive:
                survives += 1
            else:
                deaths += 1

        new_ids = sorted(set(current) - set(prior))
        traces["new_count"].extend(varint_symbols(len(new_ids), min(7, len(prior))))
        previous_ref = 0
        for track_id in new_ids:
            item = current[track_id]
            is_rebirth = int(track_id in last_seen)
            # A new component's geometry is not known at its event decision,
            # so conditioning its kind or reference on that geometry would be
            # receiver-invalid.  Only survival events use the prior decoded
            # component's horizon/lifetime/ego-motion state.
            traces["new_kind"].append((is_rebirth, 0))
            if is_rebirth:
                traces["track_ref"].extend(varint_symbols(track_id - previous_ref, 0))
                previous_ref = track_id
                rebirths += 1
            else:
                births += 1

        current_items = list(current.values())
        current_points = np.asarray([item.centroid_rc for item in current_items], dtype=np.float64)
        if prior_ids and current_items:
            distances = np.linalg.norm(predicted[:, None, :] - current_points[None, :, :], axis=2)
            adjacency = distances <= 6.0
            prior_degrees = adjacency.sum(axis=1)
            current_degrees = adjacency.sum(axis=0)
            for current_index, _item in enumerate(current_items):
                incoming = int(current_degrees[current_index])
                parents = np.where(adjacency[:, current_index])[0]
                outgoing_max = int(prior_degrees[parents].max()) if parents.size else 0
                if incoming > 1 and outgoing_max > 1:
                    symbol = 3
                    complex_events += 1
                elif incoming > 1:
                    symbol = 1
                    merges += 1
                elif outgoing_max > 1:
                    symbol = 2
                    splits += 1
                else:
                    symbol = 0
                traces["topology"].append((symbol, 0))
        else:
            for _item in current_items:
                traces["topology"].append((0, 0))

        for track_id in current:
            first_seen.setdefault(track_id, frame_index)
            last_seen[track_id] = frame_index
        prior = current

    stats = {
        "survives": survives,
        "deaths": deaths,
        "births": births,
        "rebirths": rebirths,
        "merges": merges,
        "splits": splits,
        "complex_events": complex_events,
        "trace_symbols": {name: len(values) for name, values in traces.items()},
    }
    return traces, stats


def encode_trace(values: Sequence[tuple[int, int]], alphabet: int) -> bytes:
    stream = AdaptiveStream(alphabet)
    for symbol, context in values:
        stream.encode(symbol, context)
    payload = stream.finish()
    decoder = AdaptiveStreamDecoder(payload, alphabet)
    decoded = [decoder.decode(context) for _symbol, context in values]
    expected = [symbol for symbol, _context in values]
    if decoded != expected:
        raise LTG1Error("adaptive event stream failed symbol identity")
    return payload


def pack_event(variant: str, traces: dict[str, list[tuple[int, int]]]) -> tuple[bytes, dict[str, int]]:
    alphabets = {"survival": 2, "new_count": 256, "new_kind": 2, "track_ref": 256, "topology": 4}
    streams = {name: encode_trace(traces[name], alphabets[name]) for name in sorted(traces)}
    header = {
        "schema": "ddm_ltg1.event.v1",
        "variant_id": ["unconditioned", "lifetime", "horizon_lifetime", "ego_phase", "joint"].index(variant),
        "streams": {name: len(streams[name]) for name in sorted(streams)},
        "symbols": {name: len(traces[name]) for name in sorted(traces)},
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    packet = bytearray(EVENT_MAGIC)
    packet += struct.pack("<I", len(header_bytes))
    packet += header_bytes
    for name in sorted(streams):
        packet += streams[name]
    return bytes(packet), {name: len(streams[name]) for name in sorted(streams)}


def write_payload(path: Path, payload: bytes) -> dict[str, Any]:
    atomic_write(path, payload)
    return file_record(path)


def mask_sha(mask: np.ndarray) -> str:
    return sha256_bytes(np.packbits(mask, bitorder="little").tobytes())


def subset_components(components: Sequence[Sequence[Component]], predicate: Any) -> list[list[Component]]:
    return [[component for component in frame if predicate(component)] for frame in components]


def payload_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.name not in {"MANIFEST.json", "STAGE0_COMPLETE.json"}
        ):
            records.append(file_record(path))
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "ddm_ltg1.manifest.v1",
        "files": records,
        "file_count": len(records),
        "logical_bytes": sum(int(record["bytes"]) for record in records),
        "records_sha256": sha256_bytes(canonical),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = Path(args.out_root).resolve()
    retained = output_root / "retained"
    checkpoints = output_root / "checkpoints"
    retained.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    script_sha = sha256_file(Path(__file__))
    complete_checkpoint = checkpoints / "STAGE0_COMPLETE.json"
    if args.resume_from and complete_checkpoint.exists():
        checkpoint = json.loads(complete_checkpoint.read_text())
        result_path = Path(checkpoint["result"]["path"])
        if (
            checkpoint.get("script_sha256") == script_sha
            and result_path.exists()
            and sha256_file(result_path) == checkpoint["result"]["sha256"]
        ):
            result = json.loads(result_path.read_text())
            print(json.dumps(result, indent=2, sort_keys=True))
            return result

    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < MIN_FREE_BYTES:
        raise LTG1Error(f"APDataStore free space {free_bytes} < required {MIN_FREE_BYTES}")
    lane_path = Path(args.lane_mask).resolve()
    gt_cache_path = Path(args.gt_cache).resolve()
    source_path = Path(args.source_field).resolve()
    quotient_path = Path(args.quotient_field).resolve()
    preflight = {
        "schema": "ddm_ltg1.preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "script": {"path": str(Path(__file__).resolve()), "sha256": script_sha},
        "lane_mask": file_record(lane_path),
        "source_field": file_record(source_path),
        "quotient_field": file_record(quotient_path),
        "gt_cache": file_record(gt_cache_path),
        "storage": {
            "path": str(output_root),
            "observed_free_bytes": free_bytes,
            "required_free_bytes": MIN_FREE_BYTES,
            "status": "PASS",
        },
    }
    write_json(checkpoints / "STAGE0_PREFLIGHT.json", preflight)

    lane_mask = load_lane_mask(lane_path)
    with np.load(gt_cache_path, allow_pickle=False) as cache:
        gt_lstars = np.asarray(cache["lstars"])
        gt_poses = np.asarray(cache["gt_poses"])
    if gt_lstars.shape != (N_FRAMES, HEIGHT, WIDTH):
        raise LTG1Error(f"GT lstars shape {gt_lstars.shape} is not {(N_FRAMES, HEIGHT, WIDTH)}")
    if gt_poses.shape != (N_FRAMES, 6):
        raise LTG1Error(f"GT poses shape {gt_poses.shape} is not {(N_FRAMES, 6)}")
    gt_lane = gt_lstars == 1
    gt_class1_xor_pixels = int(np.count_nonzero(gt_lane != lane_mask))
    receiver_record, receiver_mismatches = materialize_source_receiver(
        lane_mask=lane_mask,
        source_path=source_path,
        quotient_path=quotient_path,
        output_path=retained / "receiver" / "source_tokens_stage0.u8",
    )
    binary_lstars = lane_mask.astype(np.uint8)
    xi = np.stack([xi_from_pose_calibration(pose, s_t=-0.00322, s_r=0.0, pitch=-0.01) for pose in gt_poses])
    geom = GroundHomographyGeom.eon(native_hw=(HEIGHT, WIDTH), pitch=-0.01)

    phase_checkpoint_path = checkpoints / "STAGE0_PHASE_REUSE_COMPLETE.json"
    phase_receipt: dict[str, Any] | None = None
    if args.resume_from and phase_checkpoint_path.exists():
        candidate_receipt = json.loads(phase_checkpoint_path.read_text())
        payload_records = [
            candidate_receipt[scope][kind] for scope in ("default", "all_components") for kind in ("payload", "repeat")
        ]
        if candidate_receipt.get("script_sha256") == script_sha and all(
            record_is_intact(record) for record in payload_records
        ):
            phase_receipt = candidate_receipt
            default_decoded = decode_dash_phase_carrier(
                Path(candidate_receipt["default"]["payload"]["path"]).read_bytes(),
                geom=geom,
            )
            all_decoded = decode_dash_phase_carrier(
                Path(candidate_receipt["all_components"]["payload"]["path"]).read_bytes(),
                geom=geom,
            )

    if phase_receipt is None:
        default_telemetry: list[dict[str, Any]] = []
        default_section, default_report, default_decoded = encode_dash_phase_carrier(
            binary_lstars,
            xi,
            DashPhaseConfig(include_xi=True),
            geom=geom,
            telemetry=default_telemetry,
        )
        default_repeat, _default_repeat_report, _ = encode_dash_phase_carrier(
            binary_lstars, xi, DashPhaseConfig(include_xi=True), geom=geom
        )
        if default_repeat != default_section:
            raise LTG1Error("default dash phase payload is nondeterministic")
        default_record = write_payload(retained / "phase" / "dash_phase_default.bin", default_section)
        default_repeat_record = write_payload(retained / "phase" / "dash_phase_default.repeat.bin", default_repeat)

        all_telemetry: list[dict[str, Any]] = []
        all_section, all_report, all_decoded = encode_dash_phase_carrier(
            binary_lstars,
            xi,
            DashPhaseConfig(min_area=1, border_px=0, include_xi=True),
            geom=geom,
            telemetry=all_telemetry,
        )
        all_repeat, _all_repeat_report, _ = encode_dash_phase_carrier(
            binary_lstars,
            xi,
            DashPhaseConfig(min_area=1, border_px=0, include_xi=True),
            geom=geom,
        )
        if all_repeat != all_section:
            raise LTG1Error("all-component dash phase payload is nondeterministic")
        all_record = write_payload(retained / "phase" / "dash_phase_all_components.bin", all_section)
        all_repeat_record = write_payload(retained / "phase" / "dash_phase_all_components.repeat.bin", all_repeat)
        phase_receipt = {
            "script_sha256": script_sha,
            "default": {
                "payload": default_record,
                "repeat": default_repeat_record,
                "section_bytes": default_report.section_bytes,
                "section_bytes_excl_xi": default_report.section_bytes_excl_xi,
                "recalled_section_bytes": RECALLED_DASH_SECTION_BYTES,
                "recalled_reproduced": (default_report.section_bytes_excl_xi == RECALLED_DASH_SECTION_BYTES),
                "tracks": default_report.n_tracks_total,
                "births": default_report.n_births,
                "rebirths": default_report.n_rebirths,
                "deaths": default_report.n_deaths,
            },
            "all_components": {
                "payload": all_record,
                "repeat": all_repeat_record,
                "section_bytes": all_report.section_bytes,
                "section_bytes_excl_xi": all_report.section_bytes_excl_xi,
                "tracks": all_report.n_tracks_total,
                "births": all_report.n_births,
                "rebirths": all_report.n_rebirths,
                "deaths": all_report.n_deaths,
            },
        }
        write_json(phase_checkpoint_path, phase_receipt)

    event_variants = ("unconditioned", "lifetime", "horizon_lifetime", "ego_phase", "joint")
    event_checkpoint_path = checkpoints / "STAGE0_EVENT_RACE_COMPLETE.json"
    event_checkpoint: dict[str, Any] | None = None
    if args.resume_from and event_checkpoint_path.exists():
        candidate_events = json.loads(event_checkpoint_path.read_text())
        event_records = [
            row["packet"] for key in ("default_rows", "all_component_rows") for row in candidate_events[key]
        ]
        if candidate_events.get("script_sha256") == script_sha and all(
            record_is_intact(record) for record in event_records
        ):
            event_checkpoint = candidate_events

    if event_checkpoint is None:
        default_event_rows = []
        all_event_rows = []
        default_baseline_payload_bytes = None
        for scope, decoded_frames, destination, rows in (
            (
                "default_dash",
                default_decoded,
                retained / "events" / "default",
                default_event_rows,
            ),
            (
                "all_components",
                all_decoded,
                retained / "events" / "all_components",
                all_event_rows,
            ),
        ):
            for variant in event_variants:
                traces, stats = build_event_traces(decoded_frames, xi, variant, geom)
                packet, stream_bytes = pack_event(variant, traces)
                record = write_payload(destination / f"{variant}.events", packet)
                payload_bytes = sum(stream_bytes.values())
                if scope == "default_dash" and variant == "unconditioned":
                    default_baseline_payload_bytes = payload_bytes
                rows.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "packet": record,
                        "payload_bytes": payload_bytes,
                        "container_overhead_bytes": len(packet) - payload_bytes,
                        "stream_bytes": stream_bytes,
                        "stats": stats,
                    }
                )
        if default_baseline_payload_bytes is None:
            raise LTG1Error("default visibility baseline was not measured")
        best_default_event = min(default_event_rows, key=lambda row: row["packet"]["bytes"])
        visibility_savings = default_baseline_payload_bytes - int(best_default_event["payload_bytes"])
        visibility = {
            "baseline_unconditioned_event_payload_bytes": default_baseline_payload_bytes,
            "best_predictor": best_default_event["variant"],
            "best_event_payload_bytes": best_default_event["payload_bytes"],
            "measured_savings_bytes": visibility_savings,
            "required_savings_bytes": VISIBILITY_DEFICIT_BYTES,
            "clears_deficit": visibility_savings > VISIBILITY_DEFICIT_BYTES,
            "recalled_section_bytes": RECALLED_DASH_SECTION_BYTES,
            "arithmetic_section_after_savings_bytes": (RECALLED_DASH_SECTION_BYTES - visibility_savings),
            "boundary": (
                "conditional receiver-valid survival-event payload savings applied "
                "to the retained 29,958 B section; shape/delta bytes held fixed"
            ),
        }
        event_checkpoint = {
            "script_sha256": script_sha,
            "default_rows": default_event_rows,
            "all_component_rows": all_event_rows,
            "visibility": visibility,
        }
        write_json(event_checkpoint_path, event_checkpoint)
    default_event_rows = event_checkpoint["default_rows"]
    all_event_rows = event_checkpoint["all_component_rows"]
    visibility = event_checkpoint["visibility"]

    components = extract_components(lane_mask)
    component_count = sum(len(frame) for frame in components)
    component_pixels = sum(component.area for frame in components for component in frame)
    if component_pixels != int(lane_mask.sum()):
        raise LTG1Error("component census does not partition the exact Lane mask")
    if any(len(frame) != len(decoded) for frame, decoded in zip(components, all_decoded, strict=True)):
        raise LTG1Error("all-component phase tracker and exact component census disagree on counts")

    shape_rows = []
    candidates = [(0, 2)] + [(quantum, knots) for knots in (2, 3) for quantum in (1, 2, 4, 8, 16)]
    for quantum, knots in candidates:
        name = "direct_contour" if quantum == 0 else f"trunk_q{quantum}_k{knots}"
        candidate_checkpoint_path = checkpoints / f"SHAPE_{name}_COMPLETE.json"
        row: dict[str, Any] | None = None
        if args.resume_from and candidate_checkpoint_path.exists():
            checkpoint_row = json.loads(candidate_checkpoint_path.read_text())
            if (
                checkpoint_row.get("script_sha256") == script_sha
                and record_is_intact(checkpoint_row["packet"])
                and np.array_equal(
                    decode_shape(Path(checkpoint_row["packet"]["path"]).read_bytes()),
                    lane_mask,
                )
            ):
                row = checkpoint_row
        if row is None:
            packet, row, _component_rows = build_shape_candidate(lane_mask, components, quantum, knots)
            record = write_payload(retained / "shape" / name / "shape.packet", packet)
            row["packet"] = record
            row["script_sha256"] = script_sha
            write_json(candidate_checkpoint_path, row)
        shape_rows.append(row)

    best_shape = min(shape_rows, key=lambda row: row["packet_bytes"])
    best_name = str(best_shape["candidate"])
    best_quantum = int(best_shape["quantum_px"])
    best_knots = int(best_shape["knot_count"])
    best_packet_path = Path(best_shape["packet"]["path"])
    best_repeat, repeat_row, repeat_components = build_shape_candidate(lane_mask, components, best_quantum, best_knots)
    if sha256_bytes(best_repeat) != best_shape["packet_sha256"]:
        raise LTG1Error("winning shape packet is nondeterministic")
    repeat_record = write_payload(retained / "shape" / best_name / "shape.repeat.packet", best_repeat)
    best_shape["repeat"] = repeat_record
    best_shape["repeat_row_identity"] = repeat_row["identity"]

    bucket_specs = (
        ("area_1_2", lambda component: component.area <= 2),
        ("area_3_7", lambda component: 3 <= component.area <= 7),
        ("area_8_31", lambda component: 8 <= component.area <= 31),
        ("area_32_plus", lambda component: component.area >= 32),
    )
    component_table = []
    for bucket_name, predicate in bucket_specs:
        bucket_components = subset_components(components, predicate)
        bucket_mask = np.zeros_like(lane_mask)
        for frame_index, frame in enumerate(bucket_components):
            for component in frame:
                bucket_mask[frame_index, component.rows, component.cols] = True
        packet, row, _ = build_shape_candidate(bucket_mask, bucket_components, best_quantum, best_knots)
        record = write_payload(retained / "shape" / "per_component" / f"{bucket_name}.packet", packet)
        component_table.append(
            {
                "bucket": bucket_name,
                "components": sum(len(frame) for frame in bucket_components),
                "source_pixels": int(bucket_mask.sum()),
                "standalone_packet_bytes": len(packet),
                "packet": record,
                "params_coded_bytes": row["params_coded_bytes"],
                "residual_pixels": row["residual_pixels"],
                "residual_stream_bytes": row["residual_stream_bytes"],
                "identity": row["identity"],
            }
        )

    ledger_path = retained / "component_ledger.jsonl"
    ledger_lines = []
    winner_rows = {(int(row["frame"]), int(row["component"])): row for row in repeat_components}
    for frame in components:
        for component in frame:
            census_row: dict[str, Any] = {
                "frame": component.frame,
                "component": component.local_id,
                "area": component.area,
                "bbox": list(component.bbox),
                "winner": best_name,
            }
            census_row.update(winner_rows.get((component.frame, component.local_id), {}))
            ledger_lines.append(json.dumps(census_row, sort_keys=True, separators=(",", ":")))
    atomic_write(ledger_path, (("\n".join(ledger_lines) + "\n") if ledger_lines else "").encode())

    best_all_event = min(all_event_rows, key=lambda row: row["packet"]["bytes"])
    event_packet = Path(best_all_event["packet"]["path"]).read_bytes()
    shape_packet = best_packet_path.read_bytes()
    floor_header = {
        "schema": "ddm_ltg1.floor.v1",
        "event_variant": best_all_event["variant"],
        "event_bytes": len(event_packet),
        "event_sha256": sha256_bytes(event_packet),
        "shape_candidate": best_name,
        "shape_bytes": len(shape_packet),
        "shape_sha256": sha256_bytes(shape_packet),
        "target_lane_mask_sha256": LANE_MASK_SHA256,
    }
    floor_header_bytes = json.dumps(floor_header, sort_keys=True, separators=(",", ":")).encode()
    floor_packet = (
        FLOOR_MAGIC + struct.pack("<I", len(floor_header_bytes)) + floor_header_bytes + event_packet + shape_packet
    )
    floor_record = write_payload(retained / "floor" / "ltg1_stage0_floor.packet", floor_packet)
    floor_overhead = len(floor_packet) - len(event_packet) - len(shape_packet)
    verdict = (
        "FAMILY-CLOSED"
        if len(floor_packet) >= GF1_BYTES
        else ("CARRIER-CLEARED" if len(floor_packet) < CARRIAGE_BAR_BYTES else "REOPENED")
    )

    result = {
        "schema": "ddm_ltg1.result.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "script_sha256": script_sha,
        "inputs": {
            "lane_mask": file_record(lane_path),
            "source_field": file_record(source_path),
            "quotient_field": file_record(quotient_path),
            "gt_cache": file_record(gt_cache_path),
            "lane_mask_bit_order": "little",
            "lane_mask_meaning": "1 restores source class 1 over quotient class 0",
            "gt_class1_xor_pixels": gt_class1_xor_pixels,
            "gt_class1_is_not_the_d3_difference_mask": True,
            "stage0_source_receiver": receiver_record,
            "stage0_source_receiver_mismatches": receiver_mismatches,
        },
        "denominator": {
            "frames": N_FRAMES,
            "shape": [N_FRAMES, HEIGHT, WIDTH],
            "positions": N_FRAMES * HEIGHT * WIDTH,
            "lane_pixels": int(lane_mask.sum()),
            "components": component_count,
            "scope_reduction": "NONE",
        },
        "phase_reuse": phase_receipt,
        "topology_events": {
            "default_rows": default_event_rows,
            "all_component_rows": all_event_rows,
            "visibility_law_race": visibility,
        },
        "shape_race": {
            "rows": shape_rows,
            "winner": best_shape,
            "per_component_table": component_table,
            "component_ledger": file_record(ledger_path),
        },
        "floor": {
            "topology_event_packet_bytes": len(event_packet),
            "shape_packet_bytes": len(shape_packet),
            "container_overhead_bytes": floor_overhead,
            "measured_floor_bytes": len(floor_packet),
            "packet": floor_record,
            "gf1_incumbent_bytes": GF1_BYTES,
            "delta_vs_gf1_bytes": len(floor_packet) - GF1_BYTES,
            "carriage_bar_bytes": CARRIAGE_BAR_BYTES,
            "delta_vs_carriage_bar_bytes": len(floor_packet) - CARRIAGE_BAR_BYTES,
            "shape_identity": "EXACT",
            "target_sha256": LANE_MASK_SHA256,
        },
        "verdict": verdict,
        "verdict_scope": "FORMULATION: joint adaptive topology/visibility events plus integer-knot component trunk and exact contour residual on the AFR1/D3 body",
        "stage1_authorized": verdict != "FAMILY-CLOSED",
        "stage1_fired": False,
        "stage2_authorized": verdict == "CARRIER-CLEARED",
        "stage2_fired": False,
        "not_measured": ["SegNet", "PoseNet", "R", "d_seg", "d_pose", "S", "archive.zip", "contest CPU/CUDA"],
        "elapsed_seconds": round(time.time() - started, 6),
    }
    result_path = output_root / "RESULT.json"
    write_json(result_path, result)
    manifest = payload_manifest(output_root)
    write_json(output_root / "MANIFEST.json", manifest)
    result_record = file_record(result_path)
    write_json(
        complete_checkpoint,
        {
            "schema": "ddm_ltg1.stage0_complete.v1",
            "script_sha256": script_sha,
            "verdict": verdict,
            "result": result_record,
            "manifest": file_record(output_root / "MANIFEST.json"),
            "next_if_resumed": (
                "STOP_STAGE0_AND_CLOSE_FORMULATION" if verdict == "FAMILY-CLOSED" else "RUN_STAGE1_REAL_CODER_STACK"
            ),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane-mask",
        default="/Volumes/APDataStore/pact/ddm_d3_alphabet_merge/retained/carriers/lane_mask_exact.packbits",
    )
    parser.add_argument(
        "--gt-cache",
        default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    )
    parser.add_argument(
        "--source-field",
        default=(
            "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/"
            "retained/fields/decoded_tokens_instrumented.u8"
        ),
    )
    parser.add_argument(
        "--quotient-field",
        default=("/Volumes/APDataStore/pact/ddm_d3_alphabet_merge/retained/fields/tokens_lane_to_road_canonical.u8"),
    )
    parser.add_argument("--out-root", default="/Volumes/APDataStore/pact/ddm_ltg1")
    parser.add_argument(
        "--resume-from",
        default="auto",
        help="Resume from verified stage checkpoints; use an empty string to force a full deterministic rerun.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (LTG1Error, OSError, ValueError) as exc:
        print(f"LTG1 REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
