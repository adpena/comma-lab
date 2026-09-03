#!/usr/bin/env python3
"""Measure a real capacity curve for GF1's analytic generator form.

GC1 keeps GF1's four fitted semantic streams and adds one receiver-consumed,
all-class recursive dyadic block basis.  One retained atom paints one dyadic
rectangle with one canonical semantic class.  ``penalty`` is the explicit
capacity control: mismatch sites charged per retained atom.  For each penalty,
the fit uses bottom-up dynamic programming over the exact actions
``{leave GF1, paint one class, recurse to four children}`` on all 600 fields.

This runner is CPU-only and scorer-free.  It measures physical integer packet
bytes after real coding and prices every measured point's remaining mismatch
through HG1's actual correction grammar and coder/order race.  Every payload
materialized by a verdict row is retained under VertigoDataTier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for root in (REPO, REPO / "src", REPO / "experiments"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1

SCHEMA: Final = "ddm_gc1_generator_capacity_control.v1"
AXIS: Final = "[macOS-CPU scorer-free exact byte measurement]"
OUTPUT_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_gc1_generator_capacity_control")
JBP1_NULL_RECEIPT: Final = Path("/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/NULL_IDENTITY.json")
TARGET_FIELD: Final = Path("/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/retained/fields/sfp1_null_empty.u8")
TARGET_BYTES: Final = 117_964_800
TARGET_SHA256: Final = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
FIELD_SHAPE: Final = hg1.TOKEN_SHAPE
N_PAIRS, HEIGHT, WIDTH = FIELD_SHAPE
TOTAL_POSITIONS: Final = int(np.prod(FIELD_SHAPE))
CLASS_NAMES: Final = ("Road", "Lane", "Undriv", "Movable", "MyCar")

REFERENCE_PACKET_BYTES: Final = 47_603
PACKET_DIRECT_CAP: Final = 71_404.5
MISMATCH_DIRECT_CAP: Final = 46_804
REPLACEMENT_CAP: Final = 85_020
MAX_PACKET_RATIO: Final = 1.6
GENERIC_BYTES_PER_SITE: Final = 0.2909
MINIMUM_FREE_BYTES: Final = 2 * 1024**3
DEFAULT_PENALTIES: Final = (
    4096,
    2048,
    1024,
    512,
    256,
    192,
    128,
    96,
    64,
    48,
    32,
    24,
    16,
    12,
    8,
    6,
    4,
    2,
    1,
)

OVERLAY_MAGIC: Final = b"GC1O"
OVERLAY_VERSION: Final = 1
OVERLAY_HEADER: Final = struct.Struct("<4sBHHHBI")
GENERATOR_MAGIC: Final = b"GC1P"
GENERATOR_VERSION: Final = 2
GENERATOR_HEADER: Final = struct.Struct("<4sBBHIII")
CLOSED_MAGIC: Final = b"GC1C"
CLOSED_VERSION: Final = 1
CLOSED_HEADER: Final = struct.Struct("<4sBBBxIII32s32s32s")


class GC1Error(RuntimeError):
    """A GC1 source, fit, packet, receiver, or custody invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_fact(fact: Mapping[str, Any]) -> None:
    path = Path(str(fact["path"]))
    actual = file_fact(path)
    expected = {key: fact[key] for key in ("path", "bytes", "sha256")}
    if actual != expected:
        raise GC1Error(f"retained artifact identity changed: expected={expected} actual={actual}")


def atomic_bytes_once(path: Path, payload: bytes) -> dict[str, Any]:
    expected_sha = sha256_bytes(payload)
    if path.is_file():
        fact = file_fact(path)
        if fact["bytes"] != len(payload) or fact["sha256"] != expected_sha:
            raise GC1Error(f"refusing to overwrite different retained payload: {path}")
        return fact
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_json_once(path: Path, value: object) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n").encode()
    return atomic_bytes_once(path, payload)


def load_stage(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GC1Error(f"stage receipt is not an object: {path}")
    return value


def current_git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def put_uleb(output: bytearray, value: int) -> None:
    if value < 0:
        raise GC1Error("ULEB values must be non-negative")
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def get_uleb(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise GC1Error("truncated ULEB")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise GC1Error("overlong ULEB")


def level_offsets(max_depth: int) -> tuple[int, ...]:
    return tuple((4**depth - 1) // 3 for depth in range(max_depth + 1))


def nodes_per_frame(max_depth: int) -> int:
    return (4 ** (max_depth + 1) - 1) // 3


def morton_encode(row: int, col: int, depth: int) -> int:
    if not (0 <= row < 2**depth and 0 <= col < 2**depth):
        raise GC1Error("dyadic row/column is outside its depth")
    value = 0
    for bit in range(depth):
        value |= ((col >> bit) & 1) << (2 * bit)
        value |= ((row >> bit) & 1) << (2 * bit + 1)
    return value


def morton_decode(value: int, depth: int) -> tuple[int, int]:
    if value < 0 or value >= 4**depth:
        raise GC1Error("Morton value is outside its depth")
    row = 0
    col = 0
    for bit in range(depth):
        col |= ((value >> (2 * bit)) & 1) << bit
        row |= ((value >> (2 * bit + 1)) & 1) << bit
    return row, col


def node_id(
    pair: int,
    depth: int,
    row: int,
    col: int,
    *,
    max_depth: int,
) -> int:
    if max_depth < 0 or not (0 <= depth <= max_depth):
        raise GC1Error("dyadic depth is outside max_depth")
    if not (0 <= pair < N_PAIRS):
        raise GC1Error("pair is outside n600")
    return pair * nodes_per_frame(max_depth) + level_offsets(max_depth)[depth] + morton_encode(row, col, depth)


def decode_node_id(value: int, *, pairs: int, max_depth: int) -> tuple[int, int, int, int]:
    per_frame = nodes_per_frame(max_depth)
    pair, local = divmod(value, per_frame)
    if pair >= pairs:
        raise GC1Error("dyadic node address escaped the field")
    offsets = level_offsets(max_depth)
    depth = max_depth
    for candidate in range(max_depth + 1):
        stop = offsets[candidate] + 4**candidate
        if local < stop:
            depth = candidate
            break
    morton = local - offsets[depth]
    row, col = morton_decode(morton, depth)
    return pair, depth, row, col


def _validate_records(
    records: Sequence[tuple[int, int, int, int, int]],
    *,
    pairs: int,
    height: int,
    width: int,
    max_depth: int,
) -> list[tuple[int, int, int, int, int]]:
    if height % (2**max_depth) or width % (2**max_depth):
        raise GC1Error("field shape must be divisible by the finest dyadic grid")
    ordered = sorted(records, key=lambda row: node_id(*row[:4], max_depth=max_depth))
    if list(records) != ordered:
        raise GC1Error("dyadic records are not in canonical node order")
    occupied = np.zeros((pairs, 2**max_depth, 2**max_depth), dtype=bool)
    previous = -1
    for pair, depth, row, col, label in records:
        if pair >= pairs or depth > max_depth or label not in range(len(CLASS_NAMES)):
            raise GC1Error("dyadic record enum is invalid")
        address = node_id(pair, depth, row, col, max_depth=max_depth)
        if address <= previous:
            raise GC1Error("dyadic record address is duplicate or noncanonical")
        scale = 2 ** (max_depth - depth)
        view = occupied[pair, row * scale : (row + 1) * scale, col * scale : (col + 1) * scale]
        if np.any(view):
            raise GC1Error("dyadic records overlap")
        view[...] = True
        previous = address
    return ordered


def encode_overlay(
    records: Sequence[tuple[int, int, int, int, int]],
    *,
    shape: tuple[int, int, int] = FIELD_SHAPE,
    max_depth: int,
) -> bytes:
    pairs, height, width = shape
    canonical = _validate_records(
        records,
        pairs=pairs,
        height=height,
        width=width,
        max_depth=max_depth,
    )
    output = bytearray(
        OVERLAY_HEADER.pack(
            OVERLAY_MAGIC,
            OVERLAY_VERSION,
            pairs,
            height,
            width,
            max_depth,
            len(canonical),
        )
    )
    previous = -1
    for pair, depth, row, col, label in canonical:
        address = node_id(pair, depth, row, col, max_depth=max_depth)
        put_uleb(output, address - previous)
        output.append(label)
        previous = address
    return bytes(output)


def decode_overlay(payload: bytes) -> tuple[tuple[int, int, int], int, list[tuple[int, int, int, int, int]]]:
    if len(payload) < OVERLAY_HEADER.size:
        raise GC1Error("overlay is truncated")
    magic, version, pairs, height, width, max_depth, count = OVERLAY_HEADER.unpack_from(payload)
    if magic != OVERLAY_MAGIC or version != OVERLAY_VERSION:
        raise GC1Error("overlay magic/version mismatch")
    if pairs < 1 or max_depth > 15 or height % (2**max_depth) or width % (2**max_depth):
        raise GC1Error("overlay shape/depth is invalid")
    offset = OVERLAY_HEADER.size
    previous = -1
    records: list[tuple[int, int, int, int, int]] = []
    for _ in range(count):
        delta, offset = get_uleb(payload, offset)
        address = previous + delta
        if address <= previous or offset >= len(payload):
            raise GC1Error("overlay address or label is noncanonical/truncated")
        label = payload[offset]
        offset += 1
        pair, depth, row, col = decode_node_id(address, pairs=pairs, max_depth=max_depth)
        records.append((pair, depth, row, col, label))
        previous = address
    if offset != len(payload):
        raise GC1Error("overlay contains trailing bytes")
    _validate_records(
        records,
        pairs=pairs,
        height=height,
        width=width,
        max_depth=max_depth,
    )
    return (pairs, height, width), max_depth, records


def apply_overlay(payload: bytes, output: np.ndarray) -> int:
    shape, _max_depth, records = decode_overlay(payload)
    if tuple(output.shape) != shape:
        raise GC1Error(f"overlay/output shape mismatch: {shape} != {output.shape}")
    _pairs, height, width = shape
    for pair, depth, row, col, label in records:
        scale = 2**depth
        y0, y1 = row * height // scale, (row + 1) * height // scale
        x0, x1 = col * width // scale, (col + 1) * width // scale
        output[pair, y0:y1, x0:x1] = label
    return len(records)


def _coarsen(array: np.ndarray) -> np.ndarray:
    rows, cols = array.shape[:2]
    tail = array.shape[2:]
    return array.reshape(rows // 2, 2, cols // 2, 2, *tail).sum(axis=(1, 3))


def _lexicographically_better(
    objective: np.ndarray,
    distortion: np.ndarray,
    records: np.ndarray,
    best_objective: np.ndarray,
    best_distortion: np.ndarray,
    best_records: np.ndarray,
) -> np.ndarray:
    return (objective < best_objective) | (
        (objective == best_objective)
        & ((distortion < best_distortion) | ((distortion == best_distortion) & (records < best_records)))
    )


def _fit_frame(
    target: np.ndarray,
    baseline: np.ndarray,
    *,
    pair: int,
    max_depth: int,
    penalty: int,
) -> tuple[list[tuple[int, int, int, int, int]], int]:
    height, width = target.shape
    cells = 2**max_depth
    if height % cells or width % cells:
        raise GC1Error("fit shape is not divisible by the requested depth")
    block_h, block_w = height // cells, width // cells
    target_blocks = target.reshape(cells, block_h, cells, block_w).transpose(0, 2, 1, 3)
    mismatch_blocks = (target != baseline).reshape(cells, block_h, cells, block_w).transpose(0, 2, 1, 3)
    counts = np.stack(
        [np.count_nonzero(target_blocks == label, axis=(2, 3)) for label in range(len(CLASS_NAMES))],
        axis=-1,
    ).astype(np.int32)
    base_mismatch = np.count_nonzero(mismatch_blocks, axis=(2, 3)).astype(np.int32)

    counts_by_depth: list[np.ndarray] = [np.empty(0)] * (max_depth + 1)
    base_by_depth: list[np.ndarray] = [np.empty(0)] * (max_depth + 1)
    counts_by_depth[max_depth] = counts
    base_by_depth[max_depth] = base_mismatch
    for depth in range(max_depth - 1, -1, -1):
        counts_by_depth[depth] = _coarsen(counts_by_depth[depth + 1])
        base_by_depth[depth] = _coarsen(base_by_depth[depth + 1])

    actions: list[np.ndarray] = [np.empty(0, dtype=np.uint8)] * (max_depth + 1)
    labels: list[np.ndarray] = [np.empty(0, dtype=np.uint8)] * (max_depth + 1)
    child_objective: np.ndarray | None = None
    child_distortion: np.ndarray | None = None
    child_records: np.ndarray | None = None
    for depth in range(max_depth, -1, -1):
        level_counts = counts_by_depth[depth]
        level_base = base_by_depth[depth].astype(np.int64)
        area = (height // (2**depth)) * (width // (2**depth))
        level_labels = np.argmax(level_counts, axis=-1).astype(np.uint8)
        paint_distortion = (area - np.max(level_counts, axis=-1)).astype(np.int64)

        best_objective = level_base.copy()
        best_distortion = level_base.copy()
        best_records = np.zeros_like(level_base)
        action = np.zeros_like(level_base, dtype=np.uint8)

        paint_objective = paint_distortion + penalty
        paint_records = np.ones_like(level_base)
        use_paint = _lexicographically_better(
            paint_objective,
            paint_distortion,
            paint_records,
            best_objective,
            best_distortion,
            best_records,
        )
        best_objective[use_paint] = paint_objective[use_paint]
        best_distortion[use_paint] = paint_distortion[use_paint]
        best_records[use_paint] = 1
        action[use_paint] = 1

        if child_objective is not None and child_distortion is not None and child_records is not None:
            split_objective = _coarsen(child_objective[..., None])[..., 0]
            split_distortion = _coarsen(child_distortion[..., None])[..., 0]
            split_records = _coarsen(child_records[..., None])[..., 0]
            use_split = _lexicographically_better(
                split_objective,
                split_distortion,
                split_records,
                best_objective,
                best_distortion,
                best_records,
            )
            best_objective[use_split] = split_objective[use_split]
            best_distortion[use_split] = split_distortion[use_split]
            best_records[use_split] = split_records[use_split]
            action[use_split] = 2

        actions[depth] = action
        labels[depth] = level_labels
        child_objective = best_objective
        child_distortion = best_distortion
        child_records = best_records

    records_out: list[tuple[int, int, int, int, int]] = []
    stack = [(0, 0, 0)]
    while stack:
        depth, row, col = stack.pop()
        action = int(actions[depth][row, col])
        if action == 1:
            records_out.append((pair, depth, row, col, int(labels[depth][row, col])))
        elif action == 2:
            if depth == max_depth:
                raise GC1Error("fit attempted to split below max depth")
            for dr, dc in ((1, 1), (1, 0), (0, 1), (0, 0)):
                stack.append((depth + 1, row * 2 + dr, col * 2 + dc))
    records_out.sort(key=lambda row: node_id(*row[:4], max_depth=max_depth))
    root_distortion = int(child_distortion[0, 0])
    if len(records_out) != int(child_records[0, 0]):
        raise GC1Error("fit record count differs from dynamic-program root")
    return records_out, root_distortion


def fit_dyadic_overlay(
    target: np.ndarray,
    baseline: np.ndarray,
    *,
    max_depth: int,
    penalty: int,
) -> tuple[list[tuple[int, int, int, int, int]], dict[str, Any]]:
    if target.shape != baseline.shape or target.ndim != 3:
        raise GC1Error("target and baseline must be equal-shaped categorical fields")
    if penalty < 0:
        raise GC1Error("penalty must be non-negative")
    records: list[tuple[int, int, int, int, int]] = []
    distortion = 0
    started = time.monotonic()
    for pair in range(target.shape[0]):
        frame_records, frame_distortion = _fit_frame(
            np.asarray(target[pair]),
            np.asarray(baseline[pair]),
            pair=pair,
            max_depth=max_depth,
            penalty=penalty,
        )
        records.extend(frame_records)
        distortion += frame_distortion
    return records, {
        "penalty_mismatch_sites_per_atom": penalty,
        "atom_count": len(records),
        "dynamic_program_mismatches": distortion,
        "max_depth": max_depth,
        "finest_block_height": target.shape[1] // (2**max_depth),
        "finest_block_width": target.shape[2] // (2**max_depth),
        "fit_scope_pairs": target.shape[0],
        "fit_elapsed_seconds": time.monotonic() - started,
        "fit_form": "optimal bottom-up DP for leave/paint/recurse under uniform per-atom penalty",
    }


def run_coder_race(
    name: str,
    raw_path: Path,
    output_root: Path,
    *,
    storage_name: str | None = None,
) -> dict[str, Any]:
    raw = raw_path.read_bytes()
    rows: dict[str, Any] = {}
    for coder in hg1.CODERS:
        directory = output_root / "retained/coder_races" / (storage_name or name) / coder
        coded = hg1.et1.compress_payload(raw, coder)
        repeated = hg1.et1.compress_payload(raw, coder)
        coded_fact = atomic_bytes_once(directory / "payload.coded", coded)
        repeat_fact = atomic_bytes_once(directory / "payload.repeat.coded", repeated)
        if coded != repeated or hg1.et1.decompress_payload(coded, coder) != raw:
            raise GC1Error(f"{name}/{coder} failed deterministic exact coder race")
        rows[coder] = {
            "coder": coder,
            "coded": coded_fact,
            "repeat": repeat_fact,
            "deterministic_repeat_equal": True,
            "raw_parseback_equal": True,
        }
    winner = min(
        hg1.CODERS,
        key=lambda coder: (int(rows[coder]["coded"]["bytes"]), hg1.CODERS.index(coder)),
    )
    return {
        "name": name,
        "storage_name": storage_name or name,
        "raw": file_fact(raw_path),
        "coders": rows,
        "winner": winner,
    }


def parse_baseline_packet(packet: bytes) -> dict[str, bytes]:
    if len(packet) < hg1.PACKET_HEADER.size:
        raise GC1Error("baseline packet is truncated")
    magic, version, count, reserved = hg1.PACKET_HEADER.unpack_from(packet)
    if magic != hg1.PACKET_MAGIC or version != hg1.PACKET_VERSION or count != len(hg1.GENERATOR_STREAMS) or reserved:
        raise GC1Error("baseline packet header differs from the four-stream GF1 form")
    cursor = hg1.PACKET_HEADER.size
    rows = []
    for _ in range(count):
        if cursor + hg1.PACKET_ROW.size > len(packet):
            raise GC1Error("baseline packet roster is truncated")
        rows.append(hg1.PACKET_ROW.unpack_from(packet, cursor))
        cursor += hg1.PACKET_ROW.size
    streams: dict[str, bytes] = {}
    for stream_id, coder_id, raw_size, coded_size, raw_sha, coded_sha in rows:
        if stream_id not in hg1.ID_STREAMS or coder_id not in hg1.et1.CODER_NAMES:
            raise GC1Error("baseline packet enum is invalid")
        coded = packet[cursor : cursor + coded_size]
        cursor += coded_size
        if len(coded) != coded_size or sha256_bytes(coded) != coded_sha.hex():
            raise GC1Error("baseline coded stream identity mismatch")
        raw = hg1.et1.decompress_payload(coded, hg1.et1.CODER_NAMES[coder_id])
        if len(raw) != raw_size or sha256_bytes(raw) != raw_sha.hex():
            raise GC1Error("baseline raw stream identity mismatch")
        name = hg1.ID_STREAMS[stream_id]
        if name not in hg1.GENERATOR_STREAMS or name in streams:
            raise GC1Error("baseline packet stream roster is invalid")
        streams[name] = raw
    if cursor != len(packet) or set(streams) != set(hg1.GENERATOR_STREAMS):
        raise GC1Error("baseline packet roster/trailing bytes mismatch")
    return streams


def build_generator_packet(
    baseline_packet: bytes,
    overlay_race: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    winner = str(overlay_race["winner"])
    raw = Path(str(overlay_race["raw"]["path"])).read_bytes()
    coded = Path(str(overlay_race["coders"][winner]["coded"]["path"])).read_bytes()
    header = GENERATOR_HEADER.pack(
        GENERATOR_MAGIC,
        GENERATOR_VERSION,
        hg1.et1.CODER_IDS[winner],
        0,
        len(baseline_packet),
        len(raw),
        len(coded),
    )
    return atomic_bytes_once(output_path, header + baseline_packet + coded)


def parse_generator_packet(packet: bytes) -> tuple[dict[str, bytes], bytes | None]:
    if packet[:4] == hg1.PACKET_MAGIC:
        return parse_baseline_packet(packet), None
    if len(packet) < GENERATOR_HEADER.size:
        raise GC1Error("GC1 generator packet is truncated")
    (
        magic,
        version,
        coder_id,
        reserved,
        baseline_size,
        raw_size,
        coded_size,
    ) = GENERATOR_HEADER.unpack_from(packet)
    if magic != GENERATOR_MAGIC or version != GENERATOR_VERSION or coder_id not in hg1.et1.CODER_NAMES or reserved:
        raise GC1Error("GC1 generator packet header is invalid")
    cursor = GENERATOR_HEADER.size
    baseline = packet[cursor : cursor + baseline_size]
    cursor += baseline_size
    coded = packet[cursor : cursor + coded_size]
    cursor += coded_size
    if cursor != len(packet):
        raise GC1Error("GC1 generator packet contains trailing bytes")
    raw = hg1.et1.decompress_payload(coded, hg1.et1.CODER_NAMES[coder_id])
    if len(raw) != raw_size:
        raise GC1Error("GC1 generator overlay parse-back mismatch")
    decode_overlay(raw)
    return parse_baseline_packet(baseline), raw


def decode_generator_packet_to_file(packet: bytes, output_path: Path) -> dict[str, Any]:
    streams, overlay = parse_generator_packet(packet)
    hg1.render_generators(streams, output_path)
    atoms = 0
    if overlay is not None:
        output = np.memmap(output_path, mode="r+", dtype=np.uint8, shape=FIELD_SHAPE)
        atoms = apply_overlay(overlay, output)
        output.flush()
        del output
    return {**file_fact(output_path), "overlay_atom_count": atoms}


def build_closed_packet(
    generator_packet: bytes,
    residual_race: Mapping[str, Any],
    *,
    residual_order: str,
    output_path: Path,
) -> dict[str, Any]:
    winner = str(residual_race["winner"])
    raw = Path(str(residual_race["raw"]["path"])).read_bytes()
    coded = Path(str(residual_race["coders"][winner]["coded"]["path"])).read_bytes()
    header = CLOSED_HEADER.pack(
        CLOSED_MAGIC,
        CLOSED_VERSION,
        hg1.et1.CODER_IDS[winner],
        hg1.RESIDUAL_ORDER_IDS[residual_order],
        len(generator_packet),
        len(raw),
        len(coded),
        bytes.fromhex(sha256_bytes(generator_packet)),
        bytes.fromhex(sha256_bytes(raw)),
        bytes.fromhex(sha256_bytes(coded)),
    )
    return atomic_bytes_once(output_path, header + generator_packet + coded)


def parse_closed_packet(packet: bytes) -> tuple[bytes, bytes]:
    if len(packet) < CLOSED_HEADER.size:
        raise GC1Error("closed packet is truncated")
    (
        magic,
        version,
        coder_id,
        order_id,
        generator_size,
        raw_size,
        coded_size,
        generator_sha,
        raw_sha,
        coded_sha,
    ) = CLOSED_HEADER.unpack_from(packet)
    if (
        magic != CLOSED_MAGIC
        or version != CLOSED_VERSION
        or coder_id not in hg1.et1.CODER_NAMES
        or order_id not in hg1.ID_RESIDUAL_ORDERS
    ):
        raise GC1Error("closed packet enum/header is invalid")
    cursor = CLOSED_HEADER.size
    generator = packet[cursor : cursor + generator_size]
    cursor += generator_size
    coded = packet[cursor : cursor + coded_size]
    cursor += coded_size
    if cursor != len(packet):
        raise GC1Error("closed packet contains trailing bytes")
    if sha256_bytes(generator) != generator_sha.hex() or sha256_bytes(coded) != coded_sha.hex():
        raise GC1Error("closed packet section identity mismatch")
    raw = hg1.et1.decompress_payload(coded, hg1.et1.CODER_NAMES[coder_id])
    if len(raw) != raw_size or sha256_bytes(raw) != raw_sha.hex():
        raise GC1Error("closed packet residual parse-back mismatch")
    if len(raw) < hg1.RESIDUAL_HEADER.size:
        raise GC1Error("closed packet residual is truncated")
    if hg1.RESIDUAL_HEADER.unpack_from(raw)[2] != order_id:
        raise GC1Error("closed packet residual order header differs")
    parse_generator_packet(generator)
    return generator, raw


def decode_closed_packet_to_file(packet: bytes, output_path: Path) -> dict[str, Any]:
    generator, residual = parse_closed_packet(packet)
    decode_generator_packet_to_file(generator, output_path)
    output = np.memmap(output_path, mode="r+", dtype=np.uint8, shape=FIELD_SHAPE)
    corrections = hg1.apply_residual(residual, output)
    output.flush()
    del output
    return {**file_fact(output_path), "corrections": corrections}


def mismatch_facts(target: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    if target.shape != generated.shape:
        raise GC1Error("mismatch operands have different shapes")
    per_class = {}
    total = 0
    for label, name in enumerate(CLASS_NAMES):
        count = int(np.count_nonzero((target == label) & (generated != target)))
        per_class[name] = count
        total += count
    direct = int(np.count_nonzero(target != generated))
    if total != direct:
        raise GC1Error(f"per-class mismatch sum {total} != direct mismatch {direct}")
    return {
        "total": total,
        "fraction": total / int(target.size),
        "denominator_positions": int(target.size),
        "per_true_class": per_class,
        "class_order": list(CLASS_NAMES),
    }


def _validate_target_receipt() -> dict[str, Any]:
    if not JBP1_NULL_RECEIPT.is_file():
        raise GC1Error(f"JBP1 null receipt is absent: {JBP1_NULL_RECEIPT}")
    receipt = json.loads(JBP1_NULL_RECEIPT.read_text(encoding="utf-8"))
    source = receipt.get("source_field")
    if not isinstance(source, dict):
        raise GC1Error("JBP1 null receipt lacks source_field")
    expected = {
        "path": str(TARGET_FIELD),
        "bytes": TARGET_BYTES,
        "sha256": TARGET_SHA256,
    }
    observed = {key: source.get(key) for key in expected}
    if observed != expected or receipt.get("status") != "PASS":
        raise GC1Error(f"JBP1 null receipt identity/status differs: {observed}")
    actual = file_fact(TARGET_FIELD)
    if actual != expected:
        raise GC1Error(f"JBP1 target bytes drifted: {actual}")
    return {
        "receipt": file_fact(JBP1_NULL_RECEIPT),
        "source_field": actual,
        "null_status": receipt["status"],
        "archive_byte_identical": bool(receipt.get("byte_identical")),
    }


def stage_preflight(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_root).free
    if free_bytes < args.minimum_free_bytes:
        raise GC1Error(f"Vertigo storage preflight failed: {free_bytes} < {args.minimum_free_bytes}")
    target = _validate_target_receipt()
    source_paths = {
        "charter": REPO / ".omx/research/charters/ddm_gc1_generator_capacity_control_20260903.md",
        "common_contract": REPO / ".omx/tmp/codex_runs/_common_contract.md",
        "gf1_runner": REPO / "experiments/ddm_gf1_generator_form_on_lb1_field.py",
        "hg1_runner": Path(hg1.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "jbp1_null_receipt": JBP1_NULL_RECEIPT,
        "target_field": TARGET_FIELD,
    }
    sources = {name: file_fact(path) for name, path in source_paths.items()}
    payload = {
        "schema": "ddm_gc1_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "source_pins": sources,
        "target_custody": target,
        "storage": {
            "tier": "VertigoDataTier",
            "output_root": str(args.output_root.resolve()),
            "free_bytes_at_start": free_bytes,
            "minimum_free_bytes": args.minimum_free_bytes,
            "certify_or_block": True,
        },
        "provenance": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "git_head_before_serializer": current_git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "seed": "NONE_NO_RNG",
            "field_shape": list(FIELD_SHAPE),
            "max_depth": args.max_depth,
            "calibration_penalties": list(args.penalties),
            "measure_penalties": list(args.measure_penalties),
            "coders": list(hg1.CODERS),
            "residual_orders": list(hg1.RESIDUAL_ORDER_IDS),
        },
        "scope_reduction": (
            "same full n600 fit and max_depth for every capacity; CPU scorer-free target-field fit "
            "instead of realized d_seg/d_pose measurement"
        ),
        "mechanism_reduction": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "modal_calls": 0,
        "contest_evaluations": 0,
    }
    path = args.output_root / "PREFLIGHT.json"
    existing = load_stage(path)
    if existing is not None:
        old_sources = existing.get("source_pins", {})
        for name, fact in sources.items():
            if old_sources.get(name) == fact:
                continue
            if name != "runner":
                raise GC1Error(f"preflight source changed after capture: {name}")
            runner_fact = old_sources.get("runner")
            recoveries = sorted(args.output_root.glob("PREFLIGHT_SOURCE_RECOVERY_*.json"))
            for recovery_path in recoveries:
                recovery = load_stage(recovery_path)
                if recovery is None or recovery.get("superseded_runner") != runner_fact:
                    raise GC1Error(f"runner recovery chain is broken: {recovery_path}")
                runner_fact = recovery.get("current_runner")
            if runner_fact != fact:
                if (args.output_root / "RESULT.json").exists():
                    raise GC1Error("runner changed after a GC1 verdict was written")
                index = len(recoveries) + 1
                superseded_sha = runner_fact.get("sha256") if isinstance(runner_fact, dict) else None
                if superseded_sha == "a153c9844e28ae75d27308526f5c63e1985093597f6339d93386dfa999fc29e7":
                    trigger = "baseline packet build stopped with KeyError baseline_road_undrivable"
                elif superseded_sha == "159844f6f46eb46eba3de2a45e9ebf341c7d4ab8ed86f6aeee8beebe34d4f1ab":
                    trigger = "removed 96 redundant embedded hash bytes before verdict selection"
                else:
                    trigger = "repaired pre-verdict resumability/accounting without changing the measured mechanism"
                atomic_json_once(
                    args.output_root / f"PREFLIGHT_SOURCE_RECOVERY_{index:02d}.json",
                    {
                        "schema": "ddm_gc1_preflight_source_recovery.v1",
                        "status": "RECOVERED-PRE-VERDICT-CRASH",
                        "superseded_runner": runner_fact,
                        "current_runner": fact,
                        "trigger": trigger,
                        "retained_partial_artifacts_deleted": False,
                        "scope": "runner-only repair before any capacity or verdict point completed",
                    },
                )
        if existing.get("target_custody") != target:
            raise GC1Error("preflight target custody changed after capture")
        return existing
    atomic_json_once(path, payload)
    return payload


def _render_generators_resumable(streams: dict[str, bytes], output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fact = file_fact(output_path) if output_path.is_file() else hg1.render_generators(streams, output_path)
    repeat_path = output_path.with_name(f"{output_path.stem}.repeat{output_path.suffix}")
    repeat = file_fact(repeat_path) if repeat_path.is_file() else hg1.render_generators(streams, repeat_path)
    if fact["bytes"] != repeat["bytes"] or fact["sha256"] != repeat["sha256"]:
        raise GC1Error(f"resumed generator field differs from deterministic repeat: {output_path}")
    return fact, repeat


def _build_baseline_packet_resumable(
    races: Sequence[dict[str, Any]], output_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    fact = file_fact(output_path) if output_path.is_file() else hg1.build_packet(races, output_path)
    repeat_path = output_path.with_name(f"{output_path.stem}.repeat{output_path.suffix}")
    repeat = file_fact(repeat_path) if repeat_path.is_file() else hg1.build_packet(races, repeat_path)
    if fact["bytes"] != repeat["bytes"] or fact["sha256"] != repeat["sha256"]:
        raise GC1Error("resumed baseline packet differs from deterministic repeat")
    parse_baseline_packet(output_path.read_bytes())
    return fact, repeat


def _decode_generator_resumable(packet: bytes, output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fact = file_fact(output_path) if output_path.is_file() else decode_generator_packet_to_file(packet, output_path)
    repeat_path = output_path.with_name(f"{output_path.stem}.repeat{output_path.suffix}")
    repeat = file_fact(repeat_path) if repeat_path.is_file() else decode_generator_packet_to_file(packet, repeat_path)
    if fact["bytes"] != repeat["bytes"] or fact["sha256"] != repeat["sha256"]:
        raise GC1Error(f"resumed generator decode differs from deterministic repeat: {output_path}")
    return fact, repeat


def _encode_residual_resumable(
    target: np.ndarray,
    generated: np.ndarray,
    output_path: Path,
    order: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if output_path.is_file():
        raw = output_path.read_bytes()
        if len(raw) < hg1.RESIDUAL_HEADER.size:
            raise GC1Error(f"resumed residual is truncated: {output_path}")
        corrections = int(hg1.RESIDUAL_HEADER.unpack_from(raw)[-1])
        fact = {**file_fact(output_path), "corrections": corrections}
    else:
        fact = hg1.encode_residual(target, generated, output_path, None, order)
    repeat_path = output_path.with_name(f"{output_path.stem}.repeat{output_path.suffix}")
    if repeat_path.is_file():
        repeat_raw = repeat_path.read_bytes()
        if len(repeat_raw) < hg1.RESIDUAL_HEADER.size:
            raise GC1Error(f"resumed residual repeat is truncated: {repeat_path}")
        repeat = {
            **file_fact(repeat_path),
            "corrections": int(hg1.RESIDUAL_HEADER.unpack_from(repeat_raw)[-1]),
        }
    else:
        repeat = hg1.encode_residual(target, generated, repeat_path, None, order)
    if fact["bytes"] != repeat["bytes"] or fact["sha256"] != repeat["sha256"]:
        raise GC1Error(f"resumed residual differs from deterministic repeat: {output_path}")
    return fact, repeat


def _decode_closed_resumable(packet: bytes, output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fact = file_fact(output_path) if output_path.is_file() else decode_closed_packet_to_file(packet, output_path)
    repeat_path = output_path.with_name(f"{output_path.stem}.repeat{output_path.suffix}")
    repeat = file_fact(repeat_path) if repeat_path.is_file() else decode_closed_packet_to_file(packet, repeat_path)
    if fact["bytes"] != repeat["bytes"] or fact["sha256"] != repeat["sha256"]:
        raise GC1Error(f"resumed closed decode differs from deterministic repeat: {output_path}")
    return fact, repeat


def stage_baseline(args: argparse.Namespace) -> dict[str, Any]:
    stage_preflight(args)
    receipt_path = args.output_root / "stages/01_baseline.json"
    existing = load_stage(receipt_path)
    if existing is not None:
        for key in ("packet", "generated_field", "decoded_field"):
            require_fact(existing[key])
        for fact in existing["raw_streams"].values():
            require_fact(fact)
        return existing

    started = time.monotonic()
    target = np.memmap(TARGET_FIELD, mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
    root = args.output_root / "retained/baseline"
    generated_path = root / "generated.u8"
    decoded_path = root / "decoded_from_packet.u8"
    packet_path = root / "gf1_four_stream.packet"

    horizon = hg1.fit_horizon_payload(target)
    lane, lane_meta = hg1.fit_lane_payload(target)
    movable, movable_meta = hg1.fit_movable_payload(target)
    mycar, mycar_meta = hg1.fit_mycar_payload(target)
    streams = {
        "road_undrivable": horizon,
        "lane": lane,
        "movable": movable,
        "mycar": mycar,
    }
    raw_streams = {}
    races = []
    for name, payload in streams.items():
        raw_path = root / f"{name}.raw"
        raw_streams[name] = atomic_bytes_once(raw_path, payload)
        races.append(
            run_coder_race(
                name,
                raw_path,
                args.output_root,
                storage_name=f"baseline_{name}",
            )
        )
    generated_fact, generated_repeat = _render_generators_resumable(streams, generated_path)
    packet_fact, packet_repeat = _build_baseline_packet_resumable(races, packet_path)
    decoded_fact, decoded_repeat = _decode_generator_resumable(packet_path.read_bytes(), decoded_path)
    if generated_fact["sha256"] != decoded_fact["sha256"]:
        raise GC1Error("baseline packet receiver differs from direct four-stream render")
    generated = np.memmap(generated_path, mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
    mismatch = mismatch_facts(target, generated)
    coded_streams = {race["name"]: race["coders"][race["winner"]]["coded"] for race in races}
    payload = {
        "schema": "ddm_gc1_baseline.v1",
        "fit_scope_pairs": N_PAIRS,
        "fit_budget": "one full n600 exact GF1 analytic fit",
        "capacity_control": {"dyadic_atom_penalty": None, "atom_count": 0},
        "raw_streams": raw_streams,
        "coded_streams": coded_streams,
        "coder_races": races,
        "packet": packet_fact,
        "packet_repeat": packet_repeat,
        "packet_bytes": packet_fact["bytes"],
        "packet_ratio_to_47603": packet_fact["bytes"] / REFERENCE_PACKET_BYTES,
        "generated_field": generated_fact,
        "generated_field_repeat": generated_repeat,
        "decoded_field": decoded_fact,
        "decoded_field_repeat": decoded_repeat,
        "receiver_parseback_equal": True,
        "mismatch": mismatch,
        "lane_meta": hg1.json_safe(lane_meta),
        "movable_meta": hg1.json_safe(movable_meta),
        "mycar_meta": hg1.json_safe(mycar_meta),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_once(receipt_path, payload)
    del generated, target
    return payload


def calibration_label(penalty: int) -> str:
    return f"penalty_{penalty:06d}"


def stage_calibration_point(args: argparse.Namespace, penalty: int) -> dict[str, Any]:
    baseline = stage_baseline(args)
    label = calibration_label(penalty)
    receipt_path = args.output_root / f"stages/calibration/{label}.json"
    existing = load_stage(receipt_path)
    if existing is not None:
        for key in ("overlay_raw", "overlay_raw_repeat", "packet"):
            require_fact(existing[key])
        return existing

    target = np.memmap(TARGET_FIELD, mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
    generated = np.memmap(Path(baseline["generated_field"]["path"]), mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
    records, fit = fit_dyadic_overlay(
        target,
        generated,
        max_depth=args.max_depth,
        penalty=penalty,
    )
    raw = encode_overlay(records, max_depth=args.max_depth)
    root = args.output_root / "retained/calibration" / label
    raw_fact = atomic_bytes_once(root / "overlay.raw", raw)
    repeat_fact = atomic_bytes_once(root / "overlay.repeat.raw", encode_overlay(records, max_depth=args.max_depth))
    if raw_fact["sha256"] != repeat_fact["sha256"]:
        raise GC1Error("overlay repeat differs")
    race = run_coder_race(f"calibration_{label}", root / "overlay.raw", args.output_root)
    packet_path = root / "generator.packet"
    packet_fact = build_generator_packet(Path(baseline["packet"]["path"]).read_bytes(), race, packet_path)
    parse_generator_packet(packet_path.read_bytes())
    payload = {
        "schema": "ddm_gc1_calibration_point.v1",
        "penalty": penalty,
        "fit": fit,
        "overlay_raw": raw_fact,
        "overlay_raw_repeat": repeat_fact,
        "overlay_repeat_equal": True,
        "overlay_coder_race": race,
        "overlay_winner": race["winner"],
        "overlay_coded_bytes": race["coders"][race["winner"]]["coded"]["bytes"],
        "packet": packet_fact,
        "packet_bytes": packet_fact["bytes"],
        "packet_ratio_to_47603": packet_fact["bytes"] / REFERENCE_PACKET_BYTES,
        "calibration_only": True,
        "decoded_field_materialized": False,
        "residual_materialized": False,
        "note": (
            "full-n600 fit and physical packet size used only to select verdict points; "
            "mismatch is DP bookkeeping until receiver decode in measure phase"
        ),
    }
    atomic_json_once(receipt_path, payload)
    del generated, target
    return payload


def stage_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    baseline = stage_baseline(args)
    rows = [stage_calibration_point(args, penalty) for penalty in args.penalties]
    payload = {
        "schema": "ddm_gc1_calibration.v1",
        "axis": AXIS,
        "n_pairs": N_PAIRS,
        "baseline": {
            "penalty": None,
            "atom_count": 0,
            "packet_bytes": baseline["packet_bytes"],
            "packet_ratio_to_47603": baseline["packet_ratio_to_47603"],
            "mismatches": baseline["mismatch"]["total"],
        },
        "rows": rows,
        "selected_measure_penalties": list(args.measure_penalties),
        "selection_rule": (
            "MAIN selects at least three nonbaseline penalties which, with baseline, produce at "
            "least four distinct packets spanning through >=1.5x and never above 1.6x"
        ),
    }
    path = args.output_root / "CALIBRATION.json"
    existing = load_stage(path)
    if existing is not None:
        return existing
    atomic_json_once(path, payload)
    return payload


def measurement_label(penalty: int | None) -> str:
    return "baseline" if penalty is None else calibration_label(penalty)


def _copy_payload_once(source: Path, destination: Path) -> dict[str, Any]:
    return atomic_bytes_once(destination, source.read_bytes())


def stage_measure_point(args: argparse.Namespace, penalty: int | None) -> dict[str, Any]:
    baseline = stage_baseline(args)
    label = measurement_label(penalty)
    receipt_path = args.output_root / f"stages/measure/{label}.json"
    existing = load_stage(receipt_path)
    if existing is not None:
        for key in (
            "generator_packet",
            "decoded_generator_field",
            "closed_packet",
            "residual_closed_field",
        ):
            require_fact(existing[key])
        if existing["residual_closed_field"]["sha256"] != TARGET_SHA256:
            raise GC1Error(f"retained {label} residual-closed field differs from target")
        return existing

    root = args.output_root / "retained/measure" / label
    packet_path = root / "generator.packet"
    if penalty is None:
        fit = {
            "penalty_mismatch_sites_per_atom": None,
            "atom_count": 0,
            "dynamic_program_mismatches": baseline["mismatch"]["total"],
            "max_depth": args.max_depth,
            "fit_scope_pairs": N_PAIRS,
            "fit_elapsed_seconds": 0.0,
            "fit_form": "GF1 four-stream control",
        }
        packet_fact = _copy_payload_once(Path(baseline["packet"]["path"]), packet_path)
        overlay_repeat = None
    else:
        calibration = stage_calibration_point(args, penalty)
        target = np.memmap(TARGET_FIELD, mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
        generated = np.memmap(
            Path(baseline["generated_field"]["path"]),
            mode="r",
            dtype=np.uint8,
            shape=FIELD_SHAPE,
        )
        records, fit = fit_dyadic_overlay(
            target,
            generated,
            max_depth=args.max_depth,
            penalty=penalty,
        )
        repeated = encode_overlay(records, max_depth=args.max_depth)
        calibration_raw = Path(calibration["overlay_raw"]["path"]).read_bytes()
        if repeated != calibration_raw:
            raise GC1Error(f"measure refit changed calibration overlay for penalty {penalty}")
        overlay_repeat = atomic_bytes_once(root / "overlay.measure_repeat.raw", repeated)
        packet_fact = build_generator_packet(
            Path(baseline["packet"]["path"]).read_bytes(),
            calibration["overlay_coder_race"],
            packet_path,
        )
        expected_packet_bytes = (
            int(baseline["packet"]["bytes"]) + GENERATOR_HEADER.size + int(calibration["overlay_coded_bytes"])
        )
        if packet_fact["bytes"] != expected_packet_bytes:
            raise GC1Error("measurement packet differs from baseline + compact header + coded overlay")
        del generated, target

    decoded_path = root / "decoded_generator.u8"
    decoded_fact, decoded_repeat = _decode_generator_resumable(packet_path.read_bytes(), decoded_path)
    target = np.memmap(TARGET_FIELD, mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
    decoded = np.memmap(decoded_path, mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
    mismatch = mismatch_facts(target, decoded)
    if mismatch["total"] != fit["dynamic_program_mismatches"]:
        raise GC1Error(f"receiver mismatch {mismatch['total']} != DP mismatch {fit['dynamic_program_mismatches']}")

    residual_races: dict[str, Any] = {}
    residual_repeats: dict[str, Any] = {}
    for order in hg1.RESIDUAL_ORDER_IDS:
        raw_path = root / "residuals" / f"residual.{order}.raw"
        residual_fact, residual_repeat = _encode_residual_resumable(target, decoded, raw_path, order)
        if residual_fact["corrections"] != mismatch["total"]:
            raise GC1Error(f"residual correction count differs for {label}/{order}")
        residual_repeats[order] = residual_repeat
        residual_races[order] = run_coder_race(f"measure_{label}_residual_{order}", raw_path, args.output_root)
    best_order = min(
        hg1.RESIDUAL_ORDER_IDS,
        key=lambda order: (
            int(residual_races[order]["coders"][residual_races[order]["winner"]]["coded"]["bytes"]),
            tuple(hg1.RESIDUAL_ORDER_IDS).index(order),
        ),
    )
    best_race = residual_races[best_order]
    best_coder = best_race["winner"]
    residual_bytes = int(best_race["coders"][best_coder]["coded"]["bytes"])

    closed_path = root / "generator_plus_residual.packet"
    closed_fact = build_closed_packet(
        packet_path.read_bytes(),
        best_race,
        residual_order=best_order,
        output_path=closed_path,
    )
    closed_repeat = atomic_bytes_once(root / "generator_plus_residual.repeat.packet", closed_path.read_bytes())
    if closed_fact["sha256"] != closed_repeat["sha256"]:
        raise GC1Error("closed packet repeat differs")
    exact_path = root / "decoded_residual_closed.u8"
    exact_fact, exact_repeat = _decode_closed_resumable(closed_path.read_bytes(), exact_path)
    if exact_fact["sha256"] != TARGET_SHA256 or exact_fact["bytes"] != TARGET_BYTES:
        raise GC1Error(f"residual-closed receiver did not reproduce target for {label}")

    packet_bytes = int(packet_fact["bytes"])
    generic_residual = GENERIC_BYTES_PER_SITE * mismatch["total"]
    line_total = packet_bytes + residual_bytes
    payload = {
        "schema": "ddm_gc1_measure_point.v1",
        "axis": AXIS,
        "n_pairs": N_PAIRS,
        "penalty": penalty,
        "fit": fit,
        "overlay_measure_repeat": overlay_repeat,
        "generator_packet": packet_fact,
        "packet_bytes": packet_bytes,
        "packet_ratio_to_47603": packet_bytes / REFERENCE_PACKET_BYTES,
        "decoded_generator_field": decoded_fact,
        "decoded_generator_field_repeat": decoded_repeat,
        "mismatch": mismatch,
        "residual": {
            "generic_projection_bytes": generic_residual,
            "generic_projection_bytes_per_site": GENERIC_BYTES_PER_SITE,
            "domain_matched_orders": residual_races,
            "raw_deterministic_repeats": residual_repeats,
            "winner_order": best_order,
            "winner_coder": best_coder,
            "actual_coded_bytes": residual_bytes,
            "actual_bytes_per_mismatch": residual_bytes / max(mismatch["total"], 1),
            "actual_minus_generic_bytes": residual_bytes - generic_residual,
        },
        "line_accounting_packet_plus_residual_bytes": line_total,
        "closed_packet": closed_fact,
        "closed_packet_repeat": closed_repeat,
        "closed_packet_repeat_equal": True,
        "closed_packet_framing_bytes": closed_fact["bytes"] - line_total,
        "residual_closed_field": exact_fact,
        "residual_closed_field_repeat": exact_repeat,
        "residual_closed_equals_target": True,
        "thresholds": {
            "packet_le_71404_5": packet_bytes <= PACKET_DIRECT_CAP,
            "mismatches_le_46804": mismatch["total"] <= MISMATCH_DIRECT_CAP,
            "direct_joint_pass": packet_bytes <= PACKET_DIRECT_CAP and mismatch["total"] <= MISMATCH_DIRECT_CAP,
            "packet_plus_actual_residual_le_85020": line_total <= REPLACEMENT_CAP,
            "preregistered_candidate": (packet_bytes <= PACKET_DIRECT_CAP and mismatch["total"] <= MISMATCH_DIRECT_CAP)
            or line_total <= REPLACEMENT_CAP,
        },
        "score_claim": False,
        "promotable": False,
    }
    atomic_json_once(receipt_path, payload)
    del decoded, target
    return payload


def fit_power_law(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in points if int(row["packet_bytes"]) > 0 and int(row["mismatch"]["total"]) > 0]
    if len(positive) < 3:
        return {
            "status": "UNDEFINED_FEWER_THAN_3_POSITIVE_POINTS",
            "point_count": len(positive),
        }
    x = np.log(np.asarray([row["packet_bytes"] for row in positive], dtype=np.float64))
    y = np.log(np.asarray([row["mismatch"]["total"] for row in positive], dtype=np.float64))
    exponent, intercept = np.polyfit(x, y, 1)
    predicted = intercept + exponent * x
    residual = y - predicted
    total = y - np.mean(y)
    denominator = float(np.dot(total, total))
    r2 = 1.0 - float(np.dot(residual, residual)) / denominator if denominator else 1.0
    crossing = None
    if exponent < 0:
        crossing = math.exp((math.log(MISMATCH_DIRECT_CAP) - intercept) / exponent)
    return {
        "status": "FITTED_EXTRAPOLATION_NOT_MEASUREMENT",
        "model": "log(mismatches) = intercept + exponent * log(packet_bytes)",
        "point_count": len(positive),
        "intercept": float(intercept),
        "exponent": float(exponent),
        "r_squared_log_space": r2,
        "target_mismatches": MISMATCH_DIRECT_CAP,
        "extrapolated_packet_bytes_at_target": crossing,
        "crossing_vs_71404_5": None if crossing is None else crossing / PACKET_DIRECT_CAP,
        "scope": (
            "empirical fit over measured GF1+dyadic rows; extrapolation outside 1.0x-1.6x is "
            "not a measured packet or a family theorem"
        ),
    }


def stage_measure(args: argparse.Namespace) -> dict[str, Any]:
    if len(set(args.measure_penalties)) != len(args.measure_penalties):
        raise GC1Error("measure penalties must be unique")
    missing = sorted(set(args.measure_penalties) - set(args.penalties))
    if missing:
        raise GC1Error(f"measure penalties were not calibrated: {missing}")
    stage_calibrate(args)
    selection_path = args.output_root / "MEASURE_SELECTION.json"
    selection = {
        "schema": "ddm_gc1_measure_selection.v1",
        "axis": AXIS,
        "selection_mode": "calibrated full-n600 physical packet bytes",
        "measure_penalties": list(args.measure_penalties),
        "max_depth": args.max_depth,
        "argv_at_registration": sys.argv,
        "source_runner_at_registration": file_fact(Path(__file__).resolve()),
        "rule": (
            "baseline plus at least three nonbaseline points; distinct packet sizes; "
            "span through >=1.5x of 47,603 B; no point above 1.6x"
        ),
    }
    existing_selection = load_stage(selection_path)
    if existing_selection is None:
        atomic_json_once(selection_path, selection)
    else:
        for key in ("schema", "axis", "selection_mode", "measure_penalties", "max_depth", "rule"):
            if existing_selection.get(key) != selection[key]:
                raise GC1Error(f"measurement selection changed on resume: {key}")
    points = [stage_measure_point(args, None)] + [
        stage_measure_point(args, penalty) for penalty in args.measure_penalties
    ]
    packet_sizes = {int(row["packet_bytes"]) for row in points}
    ratios = [float(row["packet_ratio_to_47603"]) for row in points]
    if len(packet_sizes) < 4:
        raise GC1Error("measure roster has fewer than four distinct physical packet capacities")
    if min(ratios) > 1.02 or max(ratios) < 1.5:
        raise GC1Error(f"measure roster does not span 1.0x through >=1.5x: {ratios}")
    if max(ratios) > MAX_PACKET_RATIO:
        raise GC1Error(f"measure roster exceeds the 1.6x closed bound: {ratios}")

    power = fit_power_law(points)
    candidates = [row for row in points if row["thresholds"]["preregistered_candidate"]]
    best = min(
        points,
        key=lambda row: (
            int(row["line_accounting_packet_plus_residual_bytes"]),
            int(row["mismatch"]["total"]),
        ),
    )
    verdict = "CANDIDATE" if candidates else "CAPACITY-CLOSED"
    fire_order = None
    if candidates:
        winner = min(
            candidates,
            key=lambda row: int(row["line_accounting_packet_plus_residual_bytes"]),
        )
        fire_order = {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN scorer-lane owner",
            "consumer_store": str(args.output_root / "scorer_fire"),
            "candidate_packet": winner["generator_packet"],
            "candidate_decoded_field": winner["decoded_generator_field"],
            "fire_trigger": (
                "MAIN validates a complete archive using this exact generator packet and the shared "
                "30,856 B renderer plus 22,010 B pose lineage, claims the idle full-n600 scorer lane, "
                "then measures realized d_seg/d_pose in chunks <=120; no distortion is inherited"
            ),
            "bz2d_token_to_argmax_amplification_warning": 1.157,
        }
    payload = {
        "schema": SCHEMA,
        "axis": AXIS,
        "n_pairs": N_PAIRS,
        "selection_mode": "full n600; no subset",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "pointer_moved": False,
        "fit_reason": (
            "full-n600 categorical Hamming fit is the exact scorer-free capacity quantity named by "
            "the charter; realized Seg/Pose remains queued only if the byte/mismatch gate fires"
        ),
        "capacity_control": {
            "form": "all-class non-overlapping recursive dyadic block-paint basis after GF1",
            "control": "integer mismatch-site penalty per retained atom",
            "declared_dof": "retained dyadic atom count plus one canonical class per atom",
            "fit": "bottom-up dynamic program over leave/paint/recurse",
            "max_depth": args.max_depth,
            "finest_block": [HEIGHT // (2**args.max_depth), WIDTH // (2**args.max_depth)],
            "mechanism_reduction": False,
            "scope_reduction": "scorer-free target-field fit; identical full n600 budget per point",
        },
        "bars": {
            "reference_packet_bytes": REFERENCE_PACKET_BYTES,
            "packet_direct_cap": PACKET_DIRECT_CAP,
            "mismatch_direct_cap": MISMATCH_DIRECT_CAP,
            "replacement_cap": REPLACEMENT_CAP,
            "generic_bytes_per_site": GENERIC_BYTES_PER_SITE,
            "maximum_measured_packet_ratio": MAX_PACKET_RATIO,
        },
        "denominator": {
            "chartered_minimum_capacity_points": 4,
            "measured_capacity_points": len(points),
            "distinct_packet_capacities": len(packet_sizes),
            "missing_points": 0,
        },
        "points": points,
        "power_law": power,
        "typed_decision": {
            "verdict": verdict,
            "verdict_scope": (
                "FORMULATION: GF1 four-stream analytic generator plus this all-class recursive "
                "dyadic block-overlay basis at max_depth=" + str(args.max_depth)
            ),
            "candidate_count": len(candidates),
            "best_point": {
                "penalty": best["penalty"],
                "packet_bytes": best["packet_bytes"],
                "mismatches": best["mismatch"]["total"],
                "actual_residual_bytes": best["residual"]["actual_coded_bytes"],
                "packet_plus_actual_residual_bytes": best["line_accounting_packet_plus_residual_bytes"],
            },
            "reason": (
                "at least one physical point passes the preregistered byte/mismatch or exact residual gate"
                if candidates
                else "no physical point passes either preregistered gate"
            ),
        },
        "scorer_fire_order": fire_order,
        "boundaries": {
            "scorer_invocations": 0,
            "metal_invocations": 0,
            "modal_calls": 0,
            "contest_evaluations": 0,
            "d_seg_measured": False,
            "d_pose_measured": False,
            "score_measured": False,
            "upstream_modified": False,
        },
    }
    result_path = args.output_root / "RESULT.json"
    existing = load_stage(result_path)
    if existing is not None:
        return existing
    atomic_json_once(result_path, payload)
    return payload


def retained_inventory(output_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append(file_fact(path))
    return rows


def write_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.output_root / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for fact in manifest["files"]:
            require_fact(fact)
        return manifest
    files = retained_inventory(args.output_root, manifest_path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema": "ddm_gc1_manifest.v1",
        "output_root": str(args.output_root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(int(fact["bytes"]) for fact in files),
        "inventory_sha256": sha256_bytes(canonical),
        "files": files,
        "self_exclusion": "MANIFEST.json is self-referential and omitted",
        "destructive_actions": 0,
        "all_materialized_payloads_retained": True,
    }
    atomic_json_once(manifest_path, manifest)
    return manifest


def parse_integer_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("integer roster values must be non-negative")
    return parsed


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "measure", "all"), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument(
        "--penalties",
        type=parse_integer_list,
        default=DEFAULT_PENALTIES,
        help="comma-separated calibration penalties in mismatch-sites per dyadic atom",
    )
    parser.add_argument(
        "--measure-penalties",
        type=parse_integer_list,
        default=(),
        help="comma-separated calibrated penalties to promote to full residual-priced points",
    )
    parser.add_argument("--minimum-free-bytes", type=int, default=MINIMUM_FREE_BYTES)
    args = parser.parse_args(list(argv))
    args.output_root = args.output_root.resolve()
    if args.resume_from is not None and args.resume_from.resolve() != args.output_root:
        parser.error("--resume-from must equal --output-root")
    if not 1 <= args.max_depth <= 7:
        parser.error("--max-depth must be in [1,7] for the 384x512 field")
    if not args.penalties:
        parser.error("--penalties cannot be empty")
    if args.phase in ("measure", "all") and len(args.measure_penalties) < 3:
        parser.error("measure/all requires at least three nonbaseline --measure-penalties")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.phase in ("calibrate", "all"):
        calibration = stage_calibrate(args)
        print(
            json.dumps(
                {
                    "phase": "calibrate",
                    "baseline": calibration["baseline"],
                    "rows": [
                        {
                            "penalty": row["penalty"],
                            "atoms": row["fit"]["atom_count"],
                            "packet_bytes": row["packet_bytes"],
                            "packet_ratio": row["packet_ratio_to_47603"],
                            "dp_mismatches": row["fit"]["dynamic_program_mismatches"],
                        }
                        for row in calibration["rows"]
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
    if args.phase in ("measure", "all"):
        result = stage_measure(args)
        manifest = write_manifest(args)
        print(
            json.dumps(
                {
                    "phase": "measure",
                    "typed_decision": result["typed_decision"],
                    "power_law": result["power_law"],
                    "manifest": {key: manifest[key] for key in ("file_count", "total_bytes", "inventory_sha256")},
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
