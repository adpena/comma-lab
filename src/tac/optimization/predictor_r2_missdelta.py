# SPDX-License-Identifier: MIT
"""Task #578 round-2 predictor miss decomposition and finite delta coding.

All measurements are description-space ``[macOS-CPU advisory]`` rows.  The
module recomputes the merged round-1 predictor from its counted charts and the
same oracle-prior/proxy-motion custody, but never calls PROJECT realization or
an upstream scorer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.canonical_equations.day_consolidation_laws_20260720 import (
    RATE_PRICE_S_PER_BYTE,
    breakeven_bytes,
)
from tac.lossless.range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies
from tac.optimization.predict_project_schema import parse_constraint_seed
from tac.optimization.predictor_upgrade_xi_chart import (
    CLASS_NAMES,
    STRATA,
    classify_strata,
    load_g1_worldsheet_motion,
    load_lane_chart,
    parse_static_charts,
    predict_cell_field,
    relative_adjacent_xi,
    render_lane_mask,
    sha256_file,
)
from tac.optimization.seed_compose_b2 import GT_CACHE_SHA256

SCHEMA: Final = "predictor_r2_missdelta_task578.v1"
STAGE_SCHEMA: Final = "predictor_r2_missdelta_stage.v1"
CHUNK_SCHEMA: Final = "predictor_r2_missdelta_chunk.v1"
DELTA_MAGIC: Final = b"PBD1"
DELTA_VERSION: Final = 1
SHAPE_MAGIC: Final = b"PBS1"
KIND_NAMES: Final = ("boundary_delta_1_2px", "coherent_blob", "scattered_incoherent")
DISTANCE_BINS: Final = ("on_boundary", "1px", "2px", "3_4px", "5_8px", "gt8px")
RUN_BINS: Final = ("1", "2", "3_4", "5_8", "9_16", "gt16")
COMPONENT_BINS: Final = ("1", "2_3", "4_7", "8_15", "16_31", "ge32")
BAR_BITS_PER_MISS: Final = 0.365
BOX_BYTES: Final = 216_222
TOTAL_CELLS_N600: Final = 600 * 512 * 384
FLIP_QUANTUM_S: Final = 100.0 / TOTAL_CELLS_N600
_DIRS: Final = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_OFFSETS: Final = tuple(
    (dy, dx)
    for radius in (1, 2)
    for dy in range(-radius, radius + 1)
    for dx in range(-radius, radius + 1)
    if max(abs(dy), abs(dx)) == radius
)
_DELTA_HEADER: Final = struct.Struct("<4sBbbHHHIIIII")
_SHAPE_HEADER: Final = struct.Struct("<4sBHHHIIIIIII")


class PredictorR2Error(ValueError):
    """Fail-closed round-2 measurement or sidecar error."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_json(value) + b"\n")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    value = np.asarray(labels)
    out = np.zeros(value.shape, dtype=np.bool_)
    diff = value[:, 1:] != value[:, :-1]
    out[:, 1:] |= diff
    out[:, :-1] |= diff
    diff = value[1:, :] != value[:-1, :]
    out[1:, :] |= diff
    out[:-1, :] |= diff
    return out


def _target_side_anchor_mask(predicted: np.ndarray, baseline_class: int, target_class: int) -> np.ndarray:
    """Target-side predictor cells adjacent to ``baseline_class``.

    A corrected baseline-class site one/two digital cells from this mask has a
    signed digital displacement from a predictor-known contour, so the decoder
    can reconstruct the site without storing an absolute coordinate.
    """

    target = predicted == target_class
    baseline = predicted == baseline_class
    neighbor = np.zeros(predicted.shape, dtype=np.bool_)
    neighbor[:, 1:] |= baseline[:, :-1]
    neighbor[:, :-1] |= baseline[:, 1:]
    neighbor[1:, :] |= baseline[:-1, :]
    neighbor[:-1, :] |= baseline[1:, :]
    return target & neighbor


def _straight_order(last_dir: int | None) -> tuple[int, ...]:
    if last_dir is None:
        return tuple(range(8))
    direction = int(last_dir)
    return (
        direction,
        (direction + 1) % 8,
        (direction - 1) % 8,
        (direction + 2) % 8,
        (direction - 2) % 8,
        (direction + 3) % 8,
        (direction - 3) % 8,
        (direction + 4) % 8,
    )


def _turn_bin(previous: int | None, current: int | None) -> int:
    if previous is None or current is None:
        return 3
    delta = abs(int(current) - int(previous)) % 8
    delta = min(delta, 8 - delta)
    return 0 if delta == 0 else 1 if delta == 1 else 2


@dataclass(frozen=True)
class Anchor:
    flat_index: int
    phase_bin: int
    curvature_bin: int


def traverse_sparse_mask(mask: np.ndarray) -> list[Anchor]:
    """#307-compatible straightness-first deterministic 8-connected traversal."""

    value = np.asarray(mask, dtype=np.bool_)
    h, w = value.shape
    indices = np.flatnonzero(value.reshape(-1)).astype(np.int64)
    remaining = {int(index) for index in indices.tolist()}
    anchors: list[Anchor] = []
    for start in indices.tolist():
        start = int(start)
        if start not in remaining:
            continue
        remaining.remove(start)
        stack: list[tuple[int, int | None]] = [(start, None)]
        anchors.append(Anchor(start, len(anchors) & 7, 3))
        while stack:
            current, incoming = stack[-1]
            row, col = divmod(current, w)
            step: tuple[int, int] | None = None
            for direction in _straight_order(incoming):
                dy, dx = _DIRS[direction]
                yy, xx = row + dy, col + dx
                if 0 <= yy < h and 0 <= xx < w:
                    neighbor = yy * w + xx
                    if neighbor in remaining:
                        step = direction, neighbor
                        break
            if step is None:
                stack.pop()
                continue
            direction, neighbor = step
            remaining.remove(neighbor)
            anchors.append(
                Anchor(neighbor, len(anchors) & 7, _turn_bin(incoming, direction))
            )
            stack.append((neighbor, direction))
    if len(anchors) != len(indices):
        raise PredictorR2Error("contour traversal did not cover every anchor")
    return anchors


def _component_bin(size: int) -> int:
    return 0 if size == 1 else 1 if size <= 3 else 2 if size <= 7 else 3 if size <= 15 else 4 if size <= 31 else 5


def sparse_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return deterministic 8-connected components without optional SciPy."""

    value = np.asarray(mask, dtype=np.bool_)
    h, w = value.shape
    indices = np.flatnonzero(value.reshape(-1)).astype(np.int64)
    remaining = {int(index) for index in indices.tolist()}
    result: list[np.ndarray] = []
    for start_value in indices.tolist():
        start = int(start_value)
        if start not in remaining:
            continue
        remaining.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            row, col = divmod(current, w)
            for dy, dx in _DIRS:
                yy, xx = row + dy, col + dx
                if 0 <= yy < h and 0 <= xx < w:
                    neighbor = yy * w + xx
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        result.append(np.asarray(sorted(component), dtype=np.int64))
    return result


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=np.bool_).copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def boundary_distance_bins(misses: np.ndarray, predicted_boundary: np.ndarray) -> np.ndarray:
    """Exact Chebyshev distance bins through 8 px; all farther sites share the tail."""

    miss = np.asarray(misses, dtype=np.bool_)
    reached = np.asarray(predicted_boundary, dtype=np.bool_).copy()
    bins = np.full(miss.shape, 5, dtype=np.uint8)
    bins[miss & reached] = 0
    for radius in range(1, 9):
        previous = reached
        reached = _dilate_once(reached)
        newly = miss & reached & ~previous
        bins[newly] = 1 if radius == 1 else 2 if radius == 2 else 3 if radius <= 4 else 4
    return bins


def _run_histogram(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=np.bool_)
    padded = np.pad(value, ((0, 0), (1, 1)), constant_values=False)
    delta = np.diff(padded.astype(np.int8), axis=1)
    starts = np.argwhere(delta == 1)
    ends = np.argwhere(delta == -1)
    hist = np.zeros(6, dtype=np.int64)
    if len(starts) != len(ends):
        raise PredictorR2Error("horizontal run extraction is inconsistent")
    for (start_row, start_col), (end_row, end_col) in zip(starts, ends, strict=True):
        if start_row != end_row:
            raise PredictorR2Error("horizontal runs changed rows")
        length = int(end_col - start_col)
        bucket = 0 if length == 1 else 1 if length == 2 else 2 if length <= 4 else 3 if length <= 8 else 4 if length <= 16 else 5
        hist[bucket] += 1
    return hist


@dataclass(frozen=True)
class DeltaEvent:
    baseline_class: int
    target_class: int
    anchor_order: int
    offset_code: int
    site: int
    phase_bin: int
    curvature_bin: int
    stratum: int


def frame_delta_inventory(
    predicted: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray,
) -> tuple[np.ndarray, list[DeltaEvent], dict[tuple[int, int], list[Anchor]]]:
    """Find exactly replayable 1-2 px events and return their kind mask."""

    h, w = predicted.shape
    kind = np.full(predicted.shape, 255, dtype=np.uint8)
    events: list[DeltaEvent] = []
    anchor_groups: dict[tuple[int, int], list[Anchor]] = {}
    for baseline_class in range(5):
        for target_class in range(5):
            if baseline_class == target_class:
                continue
            anchors = traverse_sparse_mask(
                _target_side_anchor_mask(predicted, baseline_class, target_class)
            )
            anchor_groups[(baseline_class, target_class)] = anchors
            misses = (predicted == baseline_class) & (target == target_class)
            if not np.any(misses):
                continue
            by_site = {anchor.flat_index: order for order, anchor in enumerate(anchors)}
            if not by_site:
                continue
            for site_value in np.flatnonzero(misses.reshape(-1)).tolist():
                site = int(site_value)
                row, col = divmod(site, w)
                selected: tuple[int, int] | None = None
                for offset_code, (dy, dx) in enumerate(_OFFSETS):
                    yy, xx = row - dy, col - dx
                    if not (0 <= yy < h and 0 <= xx < w):
                        continue
                    anchor_order = by_site.get(yy * w + xx)
                    if anchor_order is not None:
                        selected = anchor_order, offset_code
                        break
                if selected is None:
                    continue
                anchor_order, offset_code = selected
                anchor = anchors[anchor_order]
                kind.reshape(-1)[site] = 0
                events.append(
                    DeltaEvent(
                        baseline_class,
                        target_class,
                        anchor_order,
                        offset_code,
                        site,
                        anchor.phase_bin,
                        anchor.curvature_bin,
                        int(strata.reshape(-1)[site]),
                    )
                )
    event_sites = [event.site for event in events]
    if len(event_sites) != len(set(event_sites)):
        raise PredictorR2Error("boundary-delta inventory assigned one site more than once")
    return kind, events, anchor_groups


def classify_remaining_misses(
    predicted: np.ndarray,
    target: np.ndarray,
    kind: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify non-delta misses as coherent >=4px blobs or scattered."""

    result = np.asarray(kind, dtype=np.uint8).copy()
    component_hist = np.zeros((5, 2, 6), dtype=np.int64)
    remaining = (predicted != target) & (result == 255)
    for class_id in range(5):
        for component in sparse_components(remaining & (target == class_id)):
            kind_id = 1 if len(component) >= 4 else 2
            result.reshape(-1)[component] = kind_id
            component_hist[class_id, kind_id - 1, _component_bin(len(component))] += 1
    if np.any((predicted != target) & (result == 255)) or np.any((predicted == target) & (result != 255)):
        raise PredictorR2Error("miss decomposition is not exclusive and exhaustive")
    return result, component_hist


def analyze_frame(predicted: np.ndarray, target: np.ndarray, strata: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    kind, events, _ = frame_delta_inventory(predicted, target, strata)
    kind, component_hist = classify_remaining_misses(predicted, target, kind)
    misses = predicted != target
    distance = boundary_distance_bins(misses, boundary_mask(predicted))
    counts = np.zeros((5, 4, 3), dtype=np.int64)
    distance_hist = np.zeros((5, 4, 6), dtype=np.int64)
    run_hist = np.zeros((5, 4, 3, 6), dtype=np.int64)
    adjacency = np.zeros((5, 5), dtype=np.int64)
    for class_id in range(5):
        for stratum_id in range(4):
            cell = (target == class_id) & (strata == stratum_id) & misses
            distance_hist[class_id, stratum_id] = np.bincount(
                distance[cell], minlength=6
            )[:6]
            for kind_id in range(3):
                selected = cell & (kind == kind_id)
                counts[class_id, stratum_id, kind_id] = int(np.count_nonzero(selected))
                run_hist[class_id, stratum_id, kind_id] = _run_histogram(selected)
    for event in events:
        adjacency[event.baseline_class, event.target_class] += 1
    return {
        "counts": counts,
        "distance_hist": distance_hist,
        "run_hist": run_hist,
        "component_hist": component_hist,
        "adjacency": adjacency,
        "event_count": len(events),
    }, kind


class AdaptiveStream:
    """Causal Laplace-smoothed context model over the in-tree range coder."""

    def __init__(self, alphabet: int) -> None:
        self.alphabet = int(alphabet)
        self.encoder = RangeEncoder()
        self.counts: dict[int, np.ndarray] = {}
        self.symbol_count = 0

    def encode(self, symbol: int, context: int) -> None:
        counts = self.counts.setdefault(int(context), np.ones(self.alphabet, dtype=np.int64))
        cumulative, total = cumulative_frequencies(counts.tolist())
        self.encoder.encode(symbol=int(symbol), cumulative=cumulative, total=total)
        counts[int(symbol)] += 1
        self.symbol_count += 1

    def finish(self) -> bytes:
        return self.encoder.finish() if self.symbol_count else b""


class AdaptiveStreamDecoder:
    def __init__(self, payload: bytes, alphabet: int) -> None:
        self.alphabet = int(alphabet)
        self.decoder = RangeDecoder(payload) if payload else None
        self.counts: dict[int, np.ndarray] = {}

    def decode(self, context: int) -> int:
        if self.decoder is None:
            raise PredictorR2Error("attempted to decode an empty adaptive stream")
        counts = self.counts.setdefault(int(context), np.ones(self.alphabet, dtype=np.int64))
        cumulative, total = cumulative_frequencies(counts.tolist())
        target = self.decoder.target(total)
        symbol = int(np.searchsorted(np.asarray(cumulative), target, side="right") - 1)
        self.decoder.update(
            low_count=cumulative[symbol], high_count=cumulative[symbol + 1], total=total
        )
        counts[symbol] += 1
        return symbol


def _activity_context(
    baseline_class: int,
    target_class: int,
    phase_bin: int,
    curvature_bin: int,
    previous_active: bool,
) -> int:
    pair_id = baseline_class * 5 + target_class
    return ((((pair_id * 8) + phase_bin) * 4 + curvature_bin) * 2) + int(previous_active)


def _offset_context(event: DeltaEvent, previous_offset: int) -> int:
    pair_id = event.baseline_class * 5 + event.target_class
    return ((((pair_id * 8) + event.phase_bin) * 4 + event.curvature_bin) * (len(_OFFSETS) + 1)) + previous_offset


def _selected_event(event: DeltaEvent, selection: tuple[int | None, int | None] | None) -> bool:
    if selection is None:
        return True
    class_id, stratum_id = selection
    return (class_id is None or event.target_class == class_id) and (
        stratum_id is None or event.stratum == stratum_id
    )


def encode_boundary_delta(
    predicted_frames: Sequence[np.ndarray],
    target_frames: Sequence[np.ndarray],
    strata_frames: Sequence[np.ndarray],
    *,
    selection: tuple[int | None, int | None] | None = None,
    inventories: Sequence[tuple[np.ndarray, list[DeltaEvent], dict[tuple[int, int], list[Anchor]]]] | None = None,
    verify: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Encode exact replayable class-(a) corrections against predicted contours."""

    if not predicted_frames:
        raise PredictorR2Error("boundary-delta coder needs at least one frame")
    h, w = np.asarray(predicted_frames[0]).shape
    activity = AdaptiveStream(25)
    offsets = AdaptiveStream(len(_OFFSETS))
    selected_events: list[DeltaEvent] = []
    anchor_symbols = 0
    if inventories is None:
        inventories = [
            frame_delta_inventory(predicted, target, strata)
            for predicted, target, strata in zip(predicted_frames, target_frames, strata_frames, strict=True)
        ]
    if len(inventories) != len(predicted_frames):
        raise PredictorR2Error("boundary-delta inventory count mismatch")
    for inventory in inventories:
        _, events, groups = inventory
        events = [event for event in events if _selected_event(event, selection)]
        selected_events.extend(events)
        by_group: dict[tuple[int, int], dict[int, list[DeltaEvent]]] = defaultdict(lambda: defaultdict(list))
        for event in events:
            by_group[(event.baseline_class, event.target_class)][event.anchor_order].append(event)
        for baseline_class in range(5):
            for target_class in range(5):
                if baseline_class == target_class:
                    continue
                if selection is not None and selection[0] is not None and selection[0] != target_class:
                    continue
                previous_active = False
                previous_offset = len(_OFFSETS)
                anchors = groups.get((baseline_class, target_class), ())
                for anchor_order, anchor in enumerate(anchors):
                    rows = sorted(
                        by_group[(baseline_class, target_class)].get(anchor_order, ()),
                        key=lambda event: (event.offset_code, event.site),
                    )
                    if len(rows) >= 25:
                        raise PredictorR2Error("one contour anchor has too many delta events")
                    context = _activity_context(
                        baseline_class,
                        target_class,
                        anchor.phase_bin,
                        anchor.curvature_bin,
                        previous_active,
                    )
                    activity.encode(len(rows), context)
                    anchor_symbols += 1
                    for event in rows:
                        offsets.encode(event.offset_code, _offset_context(event, previous_offset))
                        previous_offset = event.offset_code
                    previous_active = bool(rows)
    activity_payload = activity.finish()
    offset_payload = offsets.finish()
    event_count = len(selected_events)
    header = _DELTA_HEADER.pack(
        DELTA_MAGIC,
        DELTA_VERSION,
        -1 if selection is None or selection[0] is None else selection[0],
        -1 if selection is None or selection[1] is None else selection[1],
        len(predicted_frames),
        h,
        w,
        event_count,
        anchor_symbols,
        len(activity_payload),
        len(offset_payload),
        zlib.crc32(activity_payload + offset_payload) & 0xFFFFFFFF,
    )
    blob = header + activity_payload + offset_payload
    receipt = {
        "schema": "predictor_boundary_delta.v1",
        "event_count": event_count,
        "anchor_symbols": anchor_symbols,
        "activity_bytes": len(activity_payload),
        "offset_bytes": len(offset_payload),
        "header_bytes": len(header),
        "container_bytes": len(blob),
        "payload_bits_per_miss": 8.0 * (len(activity_payload) + len(offset_payload)) / max(event_count, 1),
        "container_bits_per_miss": 8.0 * len(blob) / max(event_count, 1),
        "bar_bits_per_miss": BAR_BITS_PER_MISS,
        "bar_met": 8.0 * len(blob) / max(event_count, 1) <= BAR_BITS_PER_MISS,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "selection": {"target_class": selection[0], "stratum": selection[1]} if selection else "all",
        "context_model": "#557 adaptive arithmetic: arc_phase8+class_pair+curvature4+prior_symbol",
        "digital_straightness": "#307 straightest-first contour traversal",
    }
    if verify:
        decoded = decode_boundary_delta(blob, predicted_frames)
        expected = [np.asarray(frame).copy() for frame in predicted_frames]
        expected_sites: list[tuple[int, int, int]] = []
        offset = 0
        for frame_index, inventory in enumerate(inventories):
            _, events, _ = inventory
            for event in events:
                if _selected_event(event, selection):
                    row, col = divmod(event.site, w)
                    expected[frame_index][row, col] = event.target_class
                    expected_sites.append((frame_index, row, col))
                    offset += 1
        if offset != event_count or any(not np.array_equal(a, b) for a, b in zip(decoded, expected, strict=True)):
            raise PredictorR2Error("boundary-delta decode did not reproduce selected corrections")
        if encode_boundary_delta(
            predicted_frames,
            target_frames,
            strata_frames,
            selection=selection,
            inventories=inventories,
            verify=False,
        )[0] != blob:
            raise PredictorR2Error("boundary-delta sidecar is not byte-identically re-encoded")
        receipt["decode_verified_exact"] = True
        receipt["reencode_verified_byte_identical"] = True
        receipt["corrected_site_tree_sha256"] = hashlib.sha256(canonical_json(expected_sites)).hexdigest()
    return blob, receipt


def decode_boundary_delta(blob: bytes, predicted_frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(blob) < _DELTA_HEADER.size:
        raise PredictorR2Error("boundary-delta sidecar is truncated")
    magic, version, selected_class, _selected_stratum, n_frames, h, w, event_count, anchor_symbols, activity_size, offset_size, checksum = _DELTA_HEADER.unpack_from(blob)
    if magic != DELTA_MAGIC or version != DELTA_VERSION:
        raise PredictorR2Error("boundary-delta magic/version mismatch")
    if n_frames != len(predicted_frames) or any(np.asarray(frame).shape != (h, w) for frame in predicted_frames):
        raise PredictorR2Error("boundary-delta baseline geometry mismatch")
    expected_size = _DELTA_HEADER.size + activity_size + offset_size
    if len(blob) != expected_size:
        raise PredictorR2Error("boundary-delta length mismatch or trailing bytes")
    activity_payload = blob[_DELTA_HEADER.size : _DELTA_HEADER.size + activity_size]
    offset_payload = blob[_DELTA_HEADER.size + activity_size :]
    if (zlib.crc32(activity_payload + offset_payload) & 0xFFFFFFFF) != checksum:
        raise PredictorR2Error("boundary-delta checksum mismatch")
    activity = AdaptiveStreamDecoder(activity_payload, 25)
    offsets = AdaptiveStreamDecoder(offset_payload, len(_OFFSETS))
    output = [np.asarray(frame, dtype=np.uint8).copy() for frame in predicted_frames]
    decoded_events = 0
    decoded_anchors = 0
    for frame_index, predicted in enumerate(predicted_frames):
        for baseline_class in range(5):
            for target_class in range(5):
                if baseline_class == target_class:
                    continue
                if selected_class >= 0 and selected_class != target_class:
                    continue
                anchors = traverse_sparse_mask(
                    _target_side_anchor_mask(predicted, baseline_class, target_class)
                )
                previous_active = False
                previous_offset = len(_OFFSETS)
                for anchor in anchors:
                    count = activity.decode(
                        _activity_context(
                            baseline_class,
                            target_class,
                            anchor.phase_bin,
                            anchor.curvature_bin,
                            previous_active,
                        )
                    )
                    decoded_anchors += 1
                    row, col = divmod(anchor.flat_index, w)
                    for _ in range(count):
                        stub = DeltaEvent(
                            baseline_class,
                            target_class,
                            0,
                            0,
                            0,
                            anchor.phase_bin,
                            anchor.curvature_bin,
                            0,
                        )
                        offset_code = offsets.decode(_offset_context(stub, previous_offset))
                        dy, dx = _OFFSETS[offset_code]
                        yy, xx = row + dy, col + dx
                        if not (0 <= yy < h and 0 <= xx < w):
                            raise PredictorR2Error("boundary-delta decoded site is out of bounds")
                        if int(predicted[yy, xx]) != baseline_class:
                            raise PredictorR2Error("boundary-delta decoded site baseline class mismatch")
                        if int(output[frame_index][yy, xx]) != baseline_class:
                            raise PredictorR2Error("boundary-delta decoded duplicate/conflicting site")
                        output[frame_index][yy, xx] = target_class
                        previous_offset = offset_code
                        decoded_events += 1
                    previous_active = count > 0
    if decoded_events != event_count or decoded_anchors != anchor_symbols:
        raise PredictorR2Error("boundary-delta declared counts do not match decoded stream")
    return output


def _varint_bytes(value: int) -> list[int]:
    if value < 0:
        raise PredictorR2Error("varint value must be nonnegative")
    result = []
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return result


def _write_varint(stream: AdaptiveStream, value: int) -> None:
    for index, byte in enumerate(_varint_bytes(value)):
        stream.encode(byte, min(index, 2))


def _read_varint(stream: AdaptiveStreamDecoder) -> int:
    value = 0
    shift = 0
    for index in range(10):
        byte = stream.decode(min(index, 2))
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
    raise PredictorR2Error("decoded varint exceeds uint64")


def _component_chain(component: np.ndarray, *, h: int, w: int) -> tuple[int, list[int]]:
    sites = {int(site) for site in np.asarray(component, dtype=np.int64).tolist()}
    if not sites:
        raise PredictorR2Error("shape component is empty")
    anchor = min(sites)
    sites.remove(anchor)
    stack: list[tuple[int, int | None]] = [(anchor, None)]
    symbols: list[int] = []
    while stack:
        current, incoming = stack[-1]
        row, col = divmod(current, w)
        selected: tuple[int, int] | None = None
        for direction in _straight_order(incoming):
            dy, dx = _DIRS[direction]
            yy, xx = row + dy, col + dx
            if 0 <= yy < h and 0 <= xx < w and yy * w + xx in sites:
                selected = direction, yy * w + xx
                break
        if selected is not None:
            direction, site = selected
            sites.remove(site)
            symbols.append(direction)
            stack.append((site, direction))
        elif len(stack) > 1:
            symbols.append(8)
            stack.pop()
        else:
            symbols.append(9)
            stack.pop()
    if sites:
        raise PredictorR2Error("shape chain left component sites unvisited")
    return anchor, symbols


def encode_shape_blobs(
    predicted_frames: Sequence[np.ndarray],
    target_frames: Sequence[np.ndarray],
    kind_frames: Sequence[np.ndarray],
    *,
    minimum_component_size: int = 4,
    verify: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Encode coherent class-(b) components with the reused #307 chain grammar."""

    if minimum_component_size < 4 or not predicted_frames:
        raise PredictorR2Error("shape coder needs frames and a component threshold >=4")
    h, w = np.asarray(predicted_frames[0]).shape
    counts = AdaptiveStream(256)
    anchors = AdaptiveStream(256)
    chains = AdaptiveStream(10)
    classes = AdaptiveStream(5)
    total_pixels = 0
    total_components = 0
    chain_symbols = 0
    component_hist = np.zeros(6, dtype=np.int64)
    for target, kind in zip(target_frames, kind_frames, strict=True):
        frame_components: list[tuple[int, int, np.ndarray, list[int]]] = []
        for class_id in range(5):
            for component in sparse_components((kind == 1) & (target == class_id)):
                if len(component) < minimum_component_size:
                    continue
                anchor, symbols = _component_chain(component, h=h, w=w)
                frame_components.append((anchor, class_id, component, symbols))
        frame_components.sort(key=lambda row: row[0])
        _write_varint(counts, len(frame_components))
        prior_anchor = 0
        for anchor, class_id, component, symbols in frame_components:
            _write_varint(anchors, anchor - prior_anchor)
            prior_anchor = anchor
            classes.encode(class_id, 5)
            previous_symbol = 10
            was_straight = False
            previous_move: int | None = None
            last_move: int | None = None
            for symbol in symbols:
                context = previous_symbol * 2 + int(was_straight)
                chains.encode(symbol, context)
                if symbol < 8:
                    previous_move, last_move = last_move, symbol
                    was_straight = previous_move is not None and previous_move == last_move
                else:
                    was_straight = False
                previous_symbol = symbol
            total_pixels += len(component)
            total_components += 1
            chain_symbols += len(symbols)
            component_hist[_component_bin(len(component))] += 1
    parts = (counts.finish(), anchors.finish(), chains.finish(), classes.finish())
    checksum = zlib.crc32(b"".join(parts)) & 0xFFFFFFFF
    header = _SHAPE_HEADER.pack(
        SHAPE_MAGIC,
        1,
        len(predicted_frames),
        h,
        w,
        total_pixels,
        total_components,
        len(parts[0]),
        len(parts[1]),
        len(parts[2]),
        len(parts[3]),
        checksum,
    )
    blob = header + b"".join(parts)
    receipt = {
        "schema": "predictor_shape_blob_sidecar.v1",
        "minimum_component_size": minimum_component_size,
        "pixel_count": total_pixels,
        "component_count": total_components,
        "component_histogram": dict(zip(COMPONENT_BINS, component_hist.tolist(), strict=True)),
        "chain_symbols": chain_symbols,
        "stream_bytes": {
            "counts": len(parts[0]),
            "anchors": len(parts[1]),
            "chain": len(parts[2]),
            "classes": len(parts[3]),
        },
        "container_bytes": len(blob),
        "container_bits_per_miss": 8.0 * len(blob) / max(total_pixels, 1),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "reuse": "#307 straightness-first contour strings over #557 in-tree range coder",
    }
    if verify:
        decoded = decode_shape_blobs(blob, predicted_frames)
        expected = [np.asarray(frame, dtype=np.uint8).copy() for frame in predicted_frames]
        for output, target, kind in zip(expected, target_frames, kind_frames, strict=True):
            for class_id in range(5):
                for component in sparse_components((kind == 1) & (target == class_id)):
                    if len(component) >= minimum_component_size:
                        output.reshape(-1)[component] = class_id
        if any(not np.array_equal(left, right) for left, right in zip(decoded, expected, strict=True)):
            raise PredictorR2Error("shape sidecar decode did not reproduce coherent blobs")
        if encode_shape_blobs(
            predicted_frames,
            target_frames,
            kind_frames,
            minimum_component_size=minimum_component_size,
            verify=False,
        )[0] != blob:
            raise PredictorR2Error("shape sidecar is not byte-identically re-encoded")
        receipt["decode_verified_exact"] = True
        receipt["reencode_verified_byte_identical"] = True
    return blob, receipt


def decode_shape_blobs(blob: bytes, predicted_frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(blob) < _SHAPE_HEADER.size:
        raise PredictorR2Error("shape sidecar is truncated")
    unpacked = _SHAPE_HEADER.unpack_from(blob)
    magic, version, n_frames, h, w, pixel_count, component_count, *tail = unpacked
    counts_size, anchor_size, chain_size, class_size, checksum = tail
    if magic != SHAPE_MAGIC or version != 1:
        raise PredictorR2Error("shape sidecar magic/version mismatch")
    if n_frames != len(predicted_frames) or any(np.asarray(frame).shape != (h, w) for frame in predicted_frames):
        raise PredictorR2Error("shape baseline geometry mismatch")
    sizes = (counts_size, anchor_size, chain_size, class_size)
    if len(blob) != _SHAPE_HEADER.size + sum(sizes):
        raise PredictorR2Error("shape sidecar length mismatch or trailing bytes")
    parts = []
    offset = _SHAPE_HEADER.size
    for size in sizes:
        parts.append(blob[offset : offset + size])
        offset += size
    if (zlib.crc32(b"".join(parts)) & 0xFFFFFFFF) != checksum:
        raise PredictorR2Error("shape sidecar checksum mismatch")
    counts = AdaptiveStreamDecoder(parts[0], 256)
    anchors = AdaptiveStreamDecoder(parts[1], 256)
    chains = AdaptiveStreamDecoder(parts[2], 10)
    classes = AdaptiveStreamDecoder(parts[3], 5)
    output = [np.asarray(frame, dtype=np.uint8).copy() for frame in predicted_frames]
    decoded_pixels = 0
    decoded_components = 0
    for frame_index in range(n_frames):
        n_components = _read_varint(counts)
        prior_anchor = 0
        occupied: set[int] = set()
        for _ in range(n_components):
            anchor = prior_anchor + _read_varint(anchors)
            prior_anchor = anchor
            if not 0 <= anchor < h * w:
                raise PredictorR2Error("shape anchor is out of bounds")
            target_class = classes.decode(5)
            stack = [anchor]
            component_sites = [anchor]
            occupied.add(anchor)
            previous_symbol = 10
            was_straight = False
            previous_move: int | None = None
            last_move: int | None = None
            while stack:
                symbol = chains.decode(previous_symbol * 2 + int(was_straight))
                if symbol < 8:
                    row, col = divmod(stack[-1], w)
                    dy, dx = _DIRS[symbol]
                    yy, xx = row + dy, col + dx
                    site = yy * w + xx
                    if not (0 <= yy < h and 0 <= xx < w) or site in occupied:
                        raise PredictorR2Error("shape chain decoded invalid/duplicate site")
                    stack.append(site)
                    occupied.add(site)
                    component_sites.append(site)
                    previous_move, last_move = last_move, symbol
                    was_straight = previous_move is not None and previous_move == last_move
                elif symbol == 8:
                    if len(stack) <= 1:
                        raise PredictorR2Error("shape backtrack underflow")
                    stack.pop()
                    was_straight = False
                elif symbol == 9:
                    if len(stack) != 1:
                        raise PredictorR2Error("shape component ended with open branches")
                    stack.pop()
                    was_straight = False
                else:
                    raise PredictorR2Error("shape chain symbol is invalid")
                previous_symbol = symbol
            for site in component_sites:
                if int(predicted_frames[frame_index].reshape(-1)[site]) == target_class:
                    raise PredictorR2Error("shape sidecar stores a non-miss site")
                output[frame_index].reshape(-1)[site] = target_class
            decoded_pixels += len(component_sites)
            decoded_components += 1
    if decoded_pixels != pixel_count or decoded_components != component_count:
        raise PredictorR2Error("shape sidecar declared counts mismatch decoded stream")
    return output


_POLICY_PREFIX: Final = struct.Struct("<4sBH")
_POLICY_ENTRY: Final = struct.Struct("<BBBBB")
_POLICY_CRC: Final = struct.Struct("<I")


@dataclass(frozen=True, order=True)
class RefinementEntry:
    target_class: int
    baseline_class: int
    phase_bin: int
    curvature_bin: int
    offset_code: int


def serialize_refinement_policy(entries: Sequence[RefinementEntry]) -> bytes:
    canonical = tuple(sorted(set(entries)))
    if tuple(entries) != canonical:
        raise PredictorR2Error("refinement entries must be sorted and unique")
    body = b"".join(
        _POLICY_ENTRY.pack(
            entry.target_class,
            entry.baseline_class,
            entry.phase_bin,
            entry.curvature_bin,
            entry.offset_code,
        )
        for entry in canonical
    )
    prefix = _POLICY_PREFIX.pack(b"PRF1", 1, len(canonical))
    return prefix + body + _POLICY_CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def parse_refinement_policy(payload: bytes) -> tuple[RefinementEntry, ...]:
    if len(payload) < _POLICY_PREFIX.size + _POLICY_CRC.size:
        raise PredictorR2Error("refinement policy is truncated")
    magic, version, count = _POLICY_PREFIX.unpack_from(payload)
    expected = _POLICY_PREFIX.size + count * _POLICY_ENTRY.size + _POLICY_CRC.size
    if magic != b"PRF1" or version != 1 or len(payload) != expected:
        raise PredictorR2Error("refinement policy header/length mismatch")
    body = payload[_POLICY_PREFIX.size : -_POLICY_CRC.size]
    (checksum,) = _POLICY_CRC.unpack(payload[-_POLICY_CRC.size :])
    if (zlib.crc32(body) & 0xFFFFFFFF) != checksum:
        raise PredictorR2Error("refinement policy checksum mismatch")
    entries = tuple(
        RefinementEntry(*_POLICY_ENTRY.unpack_from(body, index * _POLICY_ENTRY.size))
        for index in range(count)
    )
    if serialize_refinement_policy(entries) != payload:
        raise PredictorR2Error("refinement policy is noncanonical")
    return entries


def fit_refinement_policy(
    predicted_frames: Sequence[np.ndarray],
    target_frames: Sequence[np.ndarray],
    *,
    target_class: int,
) -> tuple[bytes, dict[str, Any]]:
    """Fit a finite contour-context correction table on the n64 prefix only."""

    candidate: dict[RefinementEntry, list[int]] = defaultdict(lambda: [0, 0, 0])
    h, w = np.asarray(predicted_frames[0]).shape
    for predicted, truth in zip(predicted_frames, target_frames, strict=True):
        for baseline_class in range(5):
            if baseline_class == target_class:
                continue
            anchors = traverse_sparse_mask(
                _target_side_anchor_mask(predicted, baseline_class, target_class)
            )
            for anchor in anchors:
                row, col = divmod(anchor.flat_index, w)
                for offset_code, (dy, dx) in enumerate(_OFFSETS):
                    yy, xx = row + dy, col + dx
                    if not (0 <= yy < h and 0 <= xx < w) or int(predicted[yy, xx]) != baseline_class:
                        continue
                    entry = RefinementEntry(
                        target_class,
                        baseline_class,
                        anchor.phase_bin,
                        anchor.curvature_bin,
                        offset_code,
                    )
                    before = int(predicted[yy, xx] != truth[yy, xx])
                    after = int(target_class != int(truth[yy, xx]))
                    stats = candidate[entry]
                    stats[0] += before - after
                    stats[1] += int(before == 1 and after == 0)
                    stats[2] += int(before == 0 and after == 1)
    by_context: dict[tuple[int, int, int, int], tuple[RefinementEntry, list[int]]] = {}
    for entry, stats in candidate.items():
        context = (entry.target_class, entry.baseline_class, entry.phase_bin, entry.curvature_bin)
        prior = by_context.get(context)
        if prior is None or (stats[0], -entry.offset_code) > (prior[1][0], -prior[0].offset_code):
            by_context[context] = entry, stats
    selected = sorted(row[0] for row in by_context.values() if row[1][0] > 0)
    payload = serialize_refinement_policy(selected)
    restored = parse_refinement_policy(payload)
    if restored != tuple(selected):
        raise PredictorR2Error("refinement policy parseback mismatch")
    training_gain = sum(by_context[(e.target_class, e.baseline_class, e.phase_bin, e.curvature_bin)][1][0] for e in selected)
    return payload, {
        "schema": "predictor_context_refinement_fit.v1",
        "target_class": target_class,
        "target_class_name": CLASS_NAMES[target_class],
        "fit_split": "n64 development prefix only",
        "candidate_entry_count": len(candidate),
        "selected_entry_count": len(selected),
        "training_net_miss_reduction_additive_before_conflict": training_gain,
        "policy_bytes": len(payload),
        "policy_sha256": hashlib.sha256(payload).hexdigest(),
    }


def apply_refinement_policy(predicted: np.ndarray, payload: bytes) -> np.ndarray:
    entries = parse_refinement_policy(payload)
    output = np.asarray(predicted, dtype=np.uint8).copy()
    h, w = output.shape
    for entry in entries:
        anchors = traverse_sparse_mask(
            _target_side_anchor_mask(predicted, entry.baseline_class, entry.target_class)
        )
        dy, dx = _OFFSETS[entry.offset_code]
        for anchor in anchors:
            if anchor.phase_bin != entry.phase_bin or anchor.curvature_bin != entry.curvature_bin:
                continue
            row, col = divmod(anchor.flat_index, w)
            yy, xx = row + dy, col + dx
            if 0 <= yy < h and 0 <= xx < w and int(output[yy, xx]) == entry.baseline_class:
                output[yy, xx] = entry.target_class
    return output


def _empty_aggregate() -> dict[str, np.ndarray]:
    return {
        "counts": np.zeros((5, 4, 3), dtype=np.int64),
        "distance_hist": np.zeros((5, 4, 6), dtype=np.int64),
        "run_hist": np.zeros((5, 4, 3, 6), dtype=np.int64),
        "component_hist": np.zeros((5, 2, 6), dtype=np.int64),
        "adjacency": np.zeros((5, 5), dtype=np.int64),
    }


def _add_summary(total: dict[str, np.ndarray], row: Mapping[str, Any]) -> None:
    for key in total:
        total[key] += np.asarray(row[key], dtype=np.int64)


def _decomposition_receipt(total: Mapping[str, np.ndarray]) -> dict[str, Any]:
    counts = np.asarray(total["counts"])
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for stratum_id, stratum in enumerate(STRATA):
            values = counts[class_id, stratum_id]
            miss_count = int(values.sum())
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "stratum": stratum,
                    "miss_count": miss_count,
                    "kinds": {
                        KIND_NAMES[kind_id]: {
                            "count": int(values[kind_id]),
                            "fraction_of_row_misses": int(values[kind_id]) / miss_count if miss_count else None,
                        }
                        for kind_id in range(3)
                    },
                    "distance_to_any_predicted_boundary_chebyshev": dict(
                        zip(
                            DISTANCE_BINS,
                            np.asarray(total["distance_hist"])[class_id, stratum_id].tolist(),
                            strict=True,
                        )
                    ),
                    "horizontal_run_count_histogram": {
                        KIND_NAMES[kind_id]: dict(
                            zip(
                                RUN_BINS,
                                np.asarray(total["run_hist"])[class_id, stratum_id, kind_id].tolist(),
                                strict=True,
                            )
                        )
                        for kind_id in range(3)
                    },
                }
            )
    adjacency = []
    matrix = np.asarray(total["adjacency"])
    for baseline_class in range(5):
        for target_class in range(5):
            if matrix[baseline_class, target_class]:
                adjacency.append(
                    {
                        "baseline_class": baseline_class,
                        "target_class": target_class,
                        "count": int(matrix[baseline_class, target_class]),
                    }
                )
    components = []
    component_hist = np.asarray(total["component_hist"])
    for class_id, class_name in enumerate(CLASS_NAMES):
        for kind_offset, kind_name in enumerate(KIND_NAMES[1:]):
            components.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "kind": kind_name,
                    "component_count_histogram": dict(
                        zip(COMPONENT_BINS, component_hist[class_id, kind_offset].tolist(), strict=True)
                    ),
                }
            )
    total_misses = int(counts.sum())
    by_kind = counts.sum(axis=(0, 1))
    return {
        "rows": rows,
        "overall": {
            "miss_count": total_misses,
            "by_kind": {
                KIND_NAMES[index]: {
                    "count": int(by_kind[index]),
                    "fraction": int(by_kind[index]) / total_misses if total_misses else None,
                }
                for index in range(3)
            },
        },
        "compatible_contour_adjacency": adjacency,
        "components": components,
    }


def _source_config_hash(
    *,
    repository_root: Path,
    cache: Path,
    round1_work_dir: Path,
    predecessor_seed_dir: Path,
    lane_chart: Path,
    n_pairs: int,
    chunk_size: int,
) -> str:
    import tac.optimization.predictor_upgrade_xi_chart as round1_module

    paths = {
        "round2_source": Path(__file__),
        "round1_source": Path(round1_module.__file__),
        "round1_chart": round1_work_dir / "charts" / "static_charts_n64.pxch",
        "round1_loose_seed": predecessor_seed_dir / "predictor_upgrade_loose.ppcs",
        "round1_tight_seed": predecessor_seed_dir / "predictor_upgrade_tight.ppcs",
        "lane_chart": lane_chart,
        "breakeven_law": repository_root
        / "src/tac/canonical_equations/day_consolidation_laws_20260720.py",
        "s2_interpreter": repository_root / "src/tac/optimization/s2_partition_seed.py",
        "contour_prior": repository_root / "tools/measure_contour_string_flip_coding.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise PredictorR2Error(f"round-2 source/config input is missing: {missing}")
    payload = {
        "schema": "predictor_r2_source_config.v1",
        "files": {name: sha256_file(path) for name, path in sorted(paths.items())},
        "cache": {"path": str(cache), "sha256": GT_CACHE_SHA256, "bytes": cache.stat().st_size},
        "n_pairs": n_pairs,
        "chunk_size": chunk_size,
        "classification": {
            "boundary_delta": "target-side compatible anchor at Chebyshev distance 1 or 2",
            "coherent_blob": "remaining same-target 8-connected component size >=4",
            "scattered": "remaining component size <=3",
        },
        "coder": "PBD1 #557 adaptive arithmetic + #307 straightness traversal",
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _storage_preflight(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(path)
    free_bytes = int(stats.f_bavail * stats.f_frsize)
    required = 5 * 1024**3
    if free_bytes < required:
        raise PredictorR2Error("SSD storage preflight has less than the 5 GiB safety floor")
    return {
        "status": "PASS",
        "tier": str(path),
        "free_bytes_before": free_bytes,
        "required_free_bytes": required,
        "cleanup": "preserve durable chunks/sidecars; certify or block before move/delete",
    }


def validate_resume_config(manifest: Mapping[str, Any], expected: str) -> None:
    if manifest.get("config_sha256") != expected:
        raise PredictorR2Error("round-2 resume refused source/config drift")


def _critical_sites(seed: Mapping[str, Any], n_pairs: int) -> dict[int, list[tuple[int, int]]]:
    result: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in seed["constraint_seeds"]:
        pair = int(row["time"])
        if pair < n_pairs and row["stratum"] == "critical_event":
            result[pair].append((int(row["y"]), int(row["x"])))
    return result


def run_measurement_stage(
    *,
    repository_root: Path,
    cache: Path,
    work_dir: Path,
    round1_work_dir: Path,
    predecessor_seed_dir: Path,
    lane_chart: Path,
    n_pairs: int,
    chunk_size: int = 16,
    resume: bool = True,
) -> dict[str, Any]:
    """Recompute and decompose one independently resumable n64/n600 stage."""

    if n_pairs not in (64, 600) or chunk_size <= 0:
        raise PredictorR2Error("only positive-chunk n64/n600 stages are admitted")
    storage = _storage_preflight(work_dir)
    if sha256_file(cache) != GT_CACHE_SHA256:
        raise PredictorR2Error("frozen n600 cache SHA-256 mismatch")
    config_sha256 = _source_config_hash(
        repository_root=repository_root,
        cache=cache,
        round1_work_dir=round1_work_dir,
        predecessor_seed_dir=predecessor_seed_dir,
        lane_chart=lane_chart,
        n_pairs=n_pairs,
        chunk_size=chunk_size,
    )
    stage_dir = work_dir / f"n{n_pairs}"
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        validate_resume_config(manifest, config_sha256)
        if not resume:
            raise PredictorR2Error("round-2 stage exists and --no-resume was requested")
    else:
        manifest = {
            "schema": "predictor_r2_stage_manifest.v1",
            "config_sha256": config_sha256,
            "n_pairs": n_pairs,
            "chunk_size": chunk_size,
            "cache": {"path": str(cache), "sha256": GT_CACHE_SHA256, "bytes": cache.stat().st_size},
            "storage_preflight": storage,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
            "main_review_required": True,
        }
        _atomic_json(manifest_path, manifest)
    chart_payload = (round1_work_dir / "charts" / "static_charts_n64.pxch").read_bytes()
    charts = parse_static_charts(chart_payload)
    lane_pairs, lane_config, lane_custody = load_lane_chart(lane_chart)
    s_t, s_r, pitch_rad, motion_custody = load_g1_worldsheet_motion(repository_root)
    loose = parse_constraint_seed((predecessor_seed_dir / "predictor_upgrade_loose.ppcs").read_bytes())
    tight = parse_constraint_seed((predecessor_seed_dir / "predictor_upgrade_tight.ppcs").read_bytes())
    movable_tracks = loose["movable_tracks"]
    critical = _critical_sites(tight, n_pairs)
    from tac.boundary_math import warp_real_luma_frame0 as g1_warp

    with np.load(cache, allow_pickle=False) as archive:
        labels = np.asarray(archive["lstars"][:n_pairs], dtype=np.uint8)
        relative, direct_motion_custody = relative_adjacent_xi(
            np.asarray(archive["gt_poses"][:n_pairs]), s_t=s_t, s_r=s_r, pitch_rad=pitch_rad
        )
        geom = g1_warp.GroundHomographyGeom.eon(native_hw=labels.shape[1:], pitch=pitch_rad)
        for start in range(0, n_pairs, chunk_size):
            stop = min(n_pairs, start + chunk_size)
            data_path = stage_dir / "chunks" / f"chunk_{start:04d}_{stop:04d}.npz"
            receipt_path = stage_dir / "chunks" / f"chunk_{start:04d}_{stop:04d}.json"
            if data_path.exists() and receipt_path.exists():
                row = json.loads(receipt_path.read_text())
                if row.get("config_sha256") != config_sha256:
                    raise PredictorR2Error("round-2 chunk resume refused config drift")
                continue
            predicted_rows = []
            kind_rows = []
            strata_rows = []
            lane_rows = []
            aggregate = _empty_aggregate()
            prediction_hashes = []
            for pair in range(start, stop):
                lane_mask = render_lane_mask(lane_pairs[pair], lane_config, h=labels.shape[1], w=labels.shape[2])
                predicted = predict_cell_field(
                    pair_index=pair,
                    prior_decoded_field=None if pair == 0 else labels[pair - 1],
                    charts=charts,
                    relative_xi=relative[pair],
                    worldsheet_geom=geom,
                    lane_mask=lane_mask,
                    movable_tracks=movable_tracks,
                )
                strata = classify_strata(labels[pair], critical.get(pair, ()))
                summary, kind = analyze_frame(predicted, labels[pair], strata)
                _add_summary(aggregate, summary)
                predicted_rows.append(predicted)
                kind_rows.append(kind)
                strata_rows.append(strata)
                lane_rows.append(lane_mask)
                prediction_hashes.append(hashlib.sha256(predicted.tobytes()).hexdigest())
            _atomic_npz(
                data_path,
                predicted=np.stack(predicted_rows),
                kind=np.stack(kind_rows),
                strata=np.stack(strata_rows),
                lane_mask=np.stack(lane_rows),
            )
            row = {
                "schema": CHUNK_SCHEMA,
                "config_sha256": config_sha256,
                "pair_range": [start, stop],
                "data_path": str(data_path),
                "data_sha256": sha256_file(data_path),
                "prediction_tree_sha256": hashlib.sha256(canonical_json(prediction_hashes)).hexdigest(),
                **{key: value.tolist() for key, value in aggregate.items()},
            }
            _atomic_json(receipt_path, row)
    aggregate = _empty_aggregate()
    chunks = []
    for receipt_path in sorted((stage_dir / "chunks").glob("chunk_*.json")):
        row = json.loads(receipt_path.read_text())
        if row["config_sha256"] != config_sha256:
            raise PredictorR2Error("round-2 aggregation refused chunk config drift")
        data_path = Path(row["data_path"])
        if not data_path.is_file() or sha256_file(data_path) != row["data_sha256"]:
            raise PredictorR2Error("round-2 chunk data custody mismatch")
        _add_summary(aggregate, row)
        chunks.append(
            {
                "pair_range": row["pair_range"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "data_path": str(data_path),
                "data_sha256": row["data_sha256"],
            }
        )
    if not chunks or chunks[0]["pair_range"][0] != 0 or chunks[-1]["pair_range"][1] != n_pairs:
        raise PredictorR2Error("round-2 stage is incomplete")
    round1_receipt = json.loads((round1_work_dir / f"n{n_pairs}" / "receipt.json").read_text())
    expected_misses = {
        int(row["class_id"]): int(row["total"] - row["correct"])
        for row in round1_receipt["satisfaction"]["per_class"]
    }
    measured_misses = {
        class_id: int(np.asarray(aggregate["counts"])[class_id].sum()) for class_id in range(5)
    }
    if measured_misses != expected_misses:
        raise PredictorR2Error(
            f"round-2 miss inventory does not reproduce round 1: {measured_misses} != {expected_misses}"
        )
    receipt = {
        "schema": STAGE_SCHEMA,
        "n_pairs": n_pairs,
        "measurement_label": "MEASURED_DEVELOPMENT_PREFIX" if n_pairs == 64 else "MEASURED [macOS-CPU advisory]",
        "config_sha256": config_sha256,
        "round1_reproduction": {
            "expected_misses_by_class": expected_misses,
            "measured_misses_by_class": measured_misses,
            "exact": True,
            "round1_receipt_sha256": sha256_file(round1_work_dir / f"n{n_pairs}" / "receipt.json"),
        },
        "D1_miss_structure": _decomposition_receipt(aggregate),
        "chunks": chunks,
        "motion_custody": {**motion_custody, **direct_motion_custody},
        "lane_chart_custody": lane_custody,
        "chart_custody": {
            "raw_bytes": len(chart_payload),
            "zlib9_bytes_diagnostic_only": len(zlib.compress(chart_payload, 9)),
            "sha256": hashlib.sha256(chart_payload).hexdigest(),
        },
        "storage_preflight": storage,
        "automatic_disk_hygiene": manifest["storage_preflight"]["cleanup"],
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "main_review_required": True,
    }
    _atomic_json(stage_dir / "receipt.json", receipt)
    return receipt


def _load_stage_frames(work_dir: Path, n_pairs: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    predicted: list[np.ndarray] = []
    kinds: list[np.ndarray] = []
    strata: list[np.ndarray] = []
    lane_masks: list[np.ndarray] = []
    for path in sorted((work_dir / f"n{n_pairs}" / "chunks").glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            predicted.extend(np.asarray(archive["predicted"], dtype=np.uint8))
            kinds.extend(np.asarray(archive["kind"], dtype=np.uint8))
            strata.extend(np.asarray(archive["strata"], dtype=np.uint8))
            lane_masks.extend(np.asarray(archive["lane_mask"], dtype=np.bool_))
    if len(predicted) != n_pairs:
        raise PredictorR2Error(f"round-2 stage frame count mismatch: {len(predicted)} != {n_pairs}")
    return predicted, kinds, strata, lane_masks


def _satisfaction(predicted: Sequence[np.ndarray], target: Sequence[np.ndarray]) -> dict[str, Any]:
    correct = np.zeros(5, dtype=np.int64)
    total = np.zeros(5, dtype=np.int64)
    for output, truth in zip(predicted, target, strict=True):
        for class_id in range(5):
            mask = truth == class_id
            total[class_id] += int(np.count_nonzero(mask))
            correct[class_id] += int(np.count_nonzero(mask & (output == truth)))
    return {
        "per_class": [
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "correct": int(correct[class_id]),
                "total": int(total[class_id]),
                "satisfaction": int(correct[class_id]) / int(total[class_id]),
            }
            for class_id in range(5)
        ],
        "overall": {
            "correct": int(correct.sum()),
            "total": int(total.sum()),
            "satisfaction": int(correct.sum()) / int(total.sum()),
            "miss_count": int(total.sum() - correct.sum()),
        },
    }


def _headroom_decomposition(
    predicted: Sequence[np.ndarray],
    target: Sequence[np.ndarray],
    kinds: Sequence[np.ndarray],
    lane_masks: Sequence[np.ndarray],
) -> dict[str, Any]:
    lane = Counter()
    road = Counter()
    for output, truth, kind, lane_mask in zip(predicted, target, kinds, lane_masks, strict=True):
        lane_fn = (truth == 1) & (output != 1)
        lane_fp = (truth != 1) & (output == 1)
        lane["false_negative"] += int(np.count_nonzero(lane_fn))
        lane["false_positive"] += int(np.count_nonzero(lane_fp))
        lane["fn_boundary_delta"] += int(np.count_nonzero(lane_fn & (kind == 0)))
        lane["fn_chart_visible_but_overwritten"] += int(np.count_nonzero(lane_fn & lane_mask))
        lane["fn_chart_not_visible"] += int(np.count_nonzero(lane_fn & ~lane_mask))
        road_miss = (truth == 0) & (output != 0)
        road["miss_count"] += int(np.count_nonzero(road_miss))
        road["boundary_delta"] += int(np.count_nonzero(road_miss & (kind == 0)))
        road["horizon_top_third"] += int(np.count_nonzero(road_miss[: truth.shape[0] // 3]))
        road["lower_two_thirds"] += int(np.count_nonzero(road_miss[truth.shape[0] // 3 :]))
        for predicted_class in range(1, 5):
            road[f"predicted_as_{CLASS_NAMES[predicted_class]}"] += int(
                np.count_nonzero(road_miss & (output == predicted_class))
            )
    return {
        "Lane": dict(lane),
        "Road": dict(road),
        "interpretation": {
            "Lane": "chart visibility and boundary alignment are mask-only diagnostics",
            "Road": "Movable prediction is the scoped movable-shadow proxy; horizon is top image third",
        },
    }


def _measure_delta_rows(
    *,
    label: str,
    sidecar_dir: Path,
    predicted: Sequence[np.ndarray],
    target: Sequence[np.ndarray],
    strata: Sequence[np.ndarray],
) -> tuple[dict[str, Any], list[tuple[np.ndarray, list[DeltaEvent], dict[tuple[int, int], list[Anchor]]]]]:
    inventories = [
        frame_delta_inventory(output, truth, stratum)
        for output, truth, stratum in zip(predicted, target, strata, strict=True)
    ]
    aggregate_blob, aggregate = encode_boundary_delta(
        predicted,
        target,
        strata,
        inventories=inventories,
    )
    aggregate_path = sidecar_dir / f"{label}_all.pbd1"
    _atomic_write(aggregate_path, aggregate_blob)
    aggregate["path"] = str(aggregate_path)
    rows = []
    event_counts = np.zeros((5, 4), dtype=np.int64)
    for _, events, _ in inventories:
        for event in events:
            event_counts[event.target_class, event.stratum] += 1
    for class_id in range(5):
        for stratum_id, stratum_name in enumerate(STRATA):
            count = int(event_counts[class_id, stratum_id])
            if count == 0:
                rows.append(
                    {
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "stratum": stratum_name,
                        "event_count": 0,
                        "payload_bits_per_miss": None,
                        "container_bits_per_miss": None,
                        "bar_met": None,
                        "container_bytes": 0,
                        "measurement_status": "MEASURED_EMPTY_ROW",
                    }
                )
                continue
            blob, row = encode_boundary_delta(
                predicted,
                target,
                strata,
                selection=(class_id, stratum_id),
                inventories=inventories,
            )
            path = sidecar_dir / f"{label}_class{class_id}_{stratum_name}.pbd1"
            _atomic_write(path, blob)
            rows.append(
                {
                    **row,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "stratum": stratum_name,
                    "path": str(path),
                    "measurement_status": "MEASURED [macOS-CPU advisory]" if len(predicted) == 600 else "MEASURED_DEVELOPMENT_PREFIX",
                }
            )
    if sum(row["event_count"] for row in rows) != aggregate["event_count"]:
        raise PredictorR2Error("per-class/stratum delta rows do not partition the aggregate events")
    return {
        "measurement_label": "MEASURED [macOS-CPU advisory]" if len(predicted) == 600 else "MEASURED_DEVELOPMENT_PREFIX",
        "aggregate": aggregate,
        "per_class_per_stratum": rows,
        "verdict": "BOUNDARY_DELTA_BAR_MET" if aggregate["bar_met"] else "BOUNDARY_DELTA_BAR_MISSED",
        "verdict_scope": "PBD1 predictor-known 1-2px compatible-contour events on this exact mask inventory only",
    }, inventories


def _apply_policy_frames(frames: Sequence[np.ndarray], payload: bytes) -> list[np.ndarray]:
    return [apply_refinement_policy(frame, payload) for frame in frames]


def _policy_candidate(
    *,
    name: str,
    target_class: int,
    n64_current: Sequence[np.ndarray],
    n600_current: Sequence[np.ndarray],
    target64: Sequence[np.ndarray],
    target600: Sequence[np.ndarray],
    d2_bits_per_miss_n64: float,
    work_dir: Path,
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    payload, fit = fit_refinement_policy(n64_current, target64, target_class=target_class)
    path = work_dir / "refinements" / f"{name}.prf1"
    _atomic_write(path, payload)
    next64 = _apply_policy_frames(n64_current, payload)
    next600 = _apply_policy_frames(n600_current, payload)
    before64 = _satisfaction(n64_current, target64)
    after64 = _satisfaction(next64, target64)
    before600 = _satisfaction(n600_current, target600)
    after600 = _satisfaction(next600, target600)
    gain64 = before64["overall"]["miss_count"] - after64["overall"]["miss_count"]
    gain600 = before600["overall"]["miss_count"] - after600["overall"]["miss_count"]
    saved_bits = gain64 * d2_bits_per_miss_n64
    policy_bits = len(payload) * 8
    admitted = gain64 > 0 and saved_bits > policy_bits
    row = {
        "name": name,
        "target_class": target_class,
        "target_class_name": CLASS_NAMES[target_class],
        "fit": fit,
        "path": str(path),
        "policy_bytes": len(payload),
        "policy_sha256": hashlib.sha256(payload).hexdigest(),
        "n64": {
            "before": before64,
            "after": after64,
            "net_miss_reduction": gain64,
            "measured_coder_savings_bits": saved_bits,
            "policy_bits": policy_bits,
        },
        "n600": {
            "before": before600,
            "after": after600,
            "net_miss_reduction": gain600,
        },
        "admitted": admitted,
        "stop_reason": None if admitted else "MARGINAL_GAIN_NOT_GREATER_THAN_MEASURED_D2_CODER_SAVINGS_GATE",
        "verdict_scope": "n64-fit contour-context table and its unchanged n600 application only",
    }
    return row, next64 if admitted else list(n64_current), next600 if admitted else list(n600_current)


def _nested_shape_curve(
    *,
    predicted: Sequence[np.ndarray],
    target: Sequence[np.ndarray],
    kinds: Sequence[np.ndarray],
    sidecar_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for threshold in (32, 16, 8, 4):
        blob, row = encode_shape_blobs(
            predicted,
            target,
            kinds,
            minimum_component_size=threshold,
        )
        path = sidecar_dir / f"n600_coherent_ge{threshold}.pbs1"
        _atomic_write(path, blob)
        rows.append({**row, "path": str(path)})
    prior_pixels = 0
    for row in rows:
        if row["pixel_count"] < prior_pixels:
            raise PredictorR2Error("shape curve is not nested by corrected pixels")
        prior_pixels = row["pixel_count"]
    return rows


def _compose_curve(
    *,
    round1_misses: int,
    refined_misses: int,
    policy_bytes: int,
    delta_rows: Sequence[Mapping[str, Any]],
    delta_aggregate: Mapping[str, Any],
    shape_rows: Sequence[Mapping[str, Any]],
    round1_declared_base_bytes: int,
) -> dict[str, Any]:
    exception_budget = math.floor(BAR_BITS_PER_MISS * round1_misses / 8.0)
    implied_base = BOX_BYTES - exception_budget
    candidates = []
    for row in delta_rows:
        if not row["event_count"]:
            continue
        benefit = int(row["event_count"]) * FLIP_QUANTUM_S
        candidates.append(
            {
                "name": f"delta:{row['class_name']}:{row['stratum']}",
                "kind": "boundary_delta",
                "corrected_misses": int(row["event_count"]),
                "bytes": int(row["container_bytes"]),
                "description_score_benefit": benefit,
                "description_score_per_byte": benefit / int(row["container_bytes"]),
            }
        )
    eligible_shapes = []
    for row in shape_rows:
        benefit = int(row["pixel_count"]) * FLIP_QUANTUM_S
        eligible_shapes.append(
            {
                "name": f"shape:components_ge{row['minimum_component_size']}",
                "kind": "coherent_shape",
                "corrected_misses": int(row["pixel_count"]),
                "bytes": int(row["container_bytes"]),
                "description_score_benefit": benefit,
                "description_score_per_byte": benefit / int(row["container_bytes"]),
            }
        )
    best_shape = max(
        eligible_shapes,
        key=lambda row: (row["description_score_per_byte"], row["corrected_misses"], -row["bytes"]),
    )
    candidates.append(best_shape)
    candidates.sort(
        key=lambda row: (-row["description_score_per_byte"], row["bytes"], row["name"])
    )
    variable_bytes = policy_bytes
    corrected = max(0, round1_misses - refined_misses)
    points = [
        {
            "name": "satisfaction_after_D3",
            "variable_bytes": variable_bytes,
            "corrected_misses_from_round1": corrected,
            "remaining_misses": refined_misses,
            "description_d_seg": refined_misses / TOTAL_CELLS_N600,
            "projected_total_bytes_implied_base": implied_base + variable_bytes,
            "round1_declared_total_bytes": round1_declared_base_bytes + variable_bytes,
        }
    ]
    admitted = []
    eaten = []
    for candidate in candidates:
        score_gate = candidate["description_score_per_byte"] >= RATE_PRICE_S_PER_BYTE
        box_gate = variable_bytes + candidate["bytes"] <= exception_budget
        if score_gate and box_gate:
            admitted.append(candidate)
            variable_bytes += candidate["bytes"]
            corrected += candidate["corrected_misses"]
            remaining = max(0, round1_misses - corrected)
            points.append(
                {
                    "name": candidate["name"],
                    "variable_bytes": variable_bytes,
                    "corrected_misses_from_round1": corrected,
                    "remaining_misses": remaining,
                    "description_d_seg": remaining / TOTAL_CELLS_N600,
                    "projected_total_bytes_implied_base": implied_base + variable_bytes,
                    "round1_declared_total_bytes": round1_declared_base_bytes + variable_bytes,
                }
            )
        else:
            eaten.append({**candidate, "score_gate": score_gate, "box_gate": box_gate})
    full_shape = next(row for row in shape_rows if row["minimum_component_size"] == 4)
    all_payload_bytes = policy_bytes + int(delta_aggregate["container_bytes"]) + int(full_shape["container_bytes"])
    all_corrected = (
        max(0, round1_misses - refined_misses)
        + int(delta_aggregate["event_count"])
        + int(full_shape["pixel_count"])
    )
    full_remaining = max(0, round1_misses - all_corrected)
    return {
        "lambda_star": {
            "equation_id": "realization_breakeven_bytes_v1",
            "resolved_rate_price_S_per_byte": RATE_PRICE_S_PER_BYTE,
            "description_flip_quantum_S": FLIP_QUANTUM_S,
            "breakeven_bytes_per_description_flip": breakeven_bytes(FLIP_QUANTUM_S),
            "authority_warning": "description-space diagnostic only; not realized hard-oracle recovery",
        },
        "box": {
            "target_bytes": BOX_BYTES,
            "bar_bits_per_round1_miss": BAR_BITS_PER_MISS,
            "derived_exception_budget_bytes": exception_budget,
            "derived_implied_base_bytes": implied_base,
            "round1_declared_static_plus_lane_base_bytes": round1_declared_base_bytes,
            "custody_conflict": round1_declared_base_bytes > BOX_BYTES,
            "note": "the implied-base projection follows the delegated 0.365-bit bar; the round-1 declared raw-PXCH accounting is shown separately and already exceeds the box",
        },
        "kkt_ranked_candidates": candidates,
        "admitted": admitted,
        "eaten": eaten,
        "curve": points,
        "knee": points[-1],
        "full_composition": {
            "variable_bytes": all_payload_bytes,
            "corrected_misses_from_round1": all_corrected,
            "remaining_misses": full_remaining,
            "description_d_seg": full_remaining / TOTAL_CELLS_N600,
            "projected_total_bytes_implied_base": implied_base + all_payload_bytes,
            "round1_declared_total_bytes": round1_declared_base_bytes + all_payload_bytes,
            "within_box_under_implied_base": implied_base + all_payload_bytes <= BOX_BYTES,
        },
        "eat_the_flip_first_class": {
            "round1_scattered_incoherent_and_unadmitted_misses_are_eaten": True,
            "knee_remaining_misses": points[-1]["remaining_misses"],
            "knee_description_d_seg": points[-1]["description_d_seg"],
        },
        "verdict_scope": "description-space exact mask corrections and counted sidecars only; no through-R score implication",
    }


def build_final_receipt(
    *,
    cache: Path,
    work_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Measure D2-D4 after both stage receipts exist and write the final receipt."""

    n64_receipt = json.loads((work_dir / "n64" / "receipt.json").read_text())
    n600_receipt = json.loads((work_dir / "n600" / "receipt.json").read_text())
    predicted64, kinds64, strata64, lane64 = _load_stage_frames(work_dir, 64)
    predicted600, kinds600, strata600, lane600 = _load_stage_frames(work_dir, 600)
    with np.load(cache, allow_pickle=False) as archive:
        target600_array = np.asarray(archive["lstars"][:600], dtype=np.uint8)
    target600 = list(target600_array)
    target64 = target600[:64]
    sidecar_dir = work_dir / "sidecars"
    d2_n64, _ = _measure_delta_rows(
        label="n64_base",
        sidecar_dir=sidecar_dir,
        predicted=predicted64,
        target=target64,
        strata=strata64,
    )
    d2_n600, _ = _measure_delta_rows(
        label="n600_base",
        sidecar_dir=sidecar_dir,
        predicted=predicted600,
        target=target600,
        strata=strata600,
    )
    d2_rate64 = float(d2_n64["aggregate"]["container_bits_per_miss"])
    current64 = list(predicted64)
    current600 = list(predicted600)
    refinements = []
    lane_row, current64, current600 = _policy_candidate(
        name="lane_arc_phase_jitter",
        target_class=1,
        n64_current=current64,
        n600_current=current600,
        target64=target64,
        target600=target600,
        d2_bits_per_miss_n64=d2_rate64,
        work_dir=work_dir,
    )
    refinements.append(lane_row)
    road_row, current64, current600 = _policy_candidate(
        name="road_contour_shadow_horizon",
        target_class=0,
        n64_current=current64,
        n600_current=current600,
        target64=target64,
        target600=target600,
        d2_bits_per_miss_n64=d2_rate64,
        work_dir=work_dir,
    )
    refinements.append(road_row)
    policy_bytes = sum(row["policy_bytes"] for row in refinements if row["admitted"])
    refined_kinds600 = []
    for output, truth, strata in zip(current600, target600, strata600, strict=True):
        kind, _, _ = frame_delta_inventory(output, truth, strata)
        kind, _ = classify_remaining_misses(output, truth, kind)
        refined_kinds600.append(kind)
    d2_refined, _ = _measure_delta_rows(
        label="n600_refined",
        sidecar_dir=sidecar_dir,
        predicted=current600,
        target=target600,
        strata=strata600,
    )
    shape_rows = _nested_shape_curve(
        predicted=current600,
        target=target600,
        kinds=refined_kinds600,
        sidecar_dir=sidecar_dir,
    )
    round1_misses = int(n600_receipt["D1_miss_structure"]["overall"]["miss_count"])
    refined_satisfaction = _satisfaction(current600, target600)
    refined_misses = int(refined_satisfaction["overall"]["miss_count"])
    round1_declared_base = (
        int(n600_receipt["chart_custody"]["raw_bytes"])
        + int(n600_receipt["lane_chart_custody"]["lane_chart_brotli_bytes"])
    )
    d4 = _compose_curve(
        round1_misses=round1_misses,
        refined_misses=refined_misses,
        policy_bytes=policy_bytes,
        delta_rows=d2_refined["per_class_per_stratum"],
        delta_aggregate=d2_refined["aggregate"],
        shape_rows=shape_rows,
        round1_declared_base_bytes=round1_declared_base,
    )
    receipt = {
        "schema": SCHEMA,
        "task": 578,
        "lane_id": "predictor_r2_missdelta",
        "research_only": True,
        "D1_miss_structure": {"n64": n64_receipt["D1_miss_structure"], "n600": n600_receipt["D1_miss_structure"]},
        "D2_boundary_delta": {"n64": d2_n64, "n600": d2_n600, "n600_after_D3": d2_refined},
        "D3_predictor_iteration": {
            "headroom_before": {
                "n64": _headroom_decomposition(predicted64, target64, kinds64, lane64),
                "n600": _headroom_decomposition(predicted600, target600, kinds600, lane600),
            },
            "refinements": refinements,
            "final_n600_satisfaction": refined_satisfaction,
            "admitted_policy_bytes": policy_bytes,
            "stop_rule": "admit only when n64 net misses saved * measured n64 D2 bits/miss exceeds counted table bits",
        },
        "D4_composed_curve": d4,
        "shape_curve": shape_rows,
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "receiver_closed": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
            "main_review_required": True,
        },
        "automatic_disk_hygiene": {
            "durable_root": str(work_dir),
            "bulk_policy": "preserve; certify or block before cold-store/delete",
            "scratch": "atomic same-directory temporary files removed after replace",
        },
        "verdict": d2_n600["verdict"],
        "verdict_scope": d2_n600["verdict_scope"],
    }
    _atomic_json(output_path, receipt)
    _atomic_json(work_dir / "receipt.json", receipt)
    return receipt


__all__ = [
    "BAR_BITS_PER_MISS",
    "BOX_BYTES",
    "KIND_NAMES",
    "PredictorR2Error",
    "RefinementEntry",
    "analyze_frame",
    "apply_refinement_policy",
    "boundary_distance_bins",
    "build_final_receipt",
    "classify_remaining_misses",
    "decode_boundary_delta",
    "decode_shape_blobs",
    "encode_boundary_delta",
    "encode_shape_blobs",
    "fit_refinement_policy",
    "frame_delta_inventory",
    "parse_refinement_policy",
    "run_measurement_stage",
    "serialize_refinement_policy",
    "sparse_components",
    "traverse_sparse_mask",
    "validate_resume_config",
]
