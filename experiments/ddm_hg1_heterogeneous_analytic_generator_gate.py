#!/usr/bin/env python3
"""Price a heterogeneous analytic generator on the exact current DX2 field.

The receiver generates all five semantic classes from four heterogeneous
members: a Road/Undrivable horizon polyline, the existing coherent
polynomial-plus-dash Lane program, a temporally tracked Movable-box carrier,
and a static MyCar seed.  A single unique-home residual then carries either
all generator errors (the exact row) or only the BL1 concentration-protected
set (the byte-only bracket).

Every raw stream, every real-coder output, every deterministic repeat, every
packet, and every decoded field is retained.  The exact packet is checked
against the current 600x384x512 DX2 categorical field.  This tool is CPU-only
and scorer-free; it never claims that categorical disagreement is d_seg.
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
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import ddm_et1_edge_topology_container_gate as et1
import numpy as np
from scipy import ndimage

from tac.boundary_math.analytic_lane_render_band import (
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    deserialize_lane_band_any,
    rasterize_lane_coverage_range_dependent,
    serialize_lane_band_rd_tracked,
)

AXIS: Final = "[macOS-CPU advisory / scorer-free exact byte measurement]"
TOKEN_SHAPE: Final = et1.TOKEN_SHAPE
N_PAIRS, HEIGHT, WIDTH = TOKEN_SHAPE
TOTAL_POSITIONS: Final = int(np.prod(TOKEN_SHAPE))
CLASS_ROAD: Final = 0
CLASS_LANE: Final = 1
CLASS_UNDRIVABLE: Final = 2
CLASS_MOVABLE: Final = 3
CLASS_MYCAR: Final = 4

TOP1_BYTES: Final = 14_745_600
TOP1_SHA256: Final = "f48cd9d61c4580dda23dc1ff4c7504009612863760ad962c578c190114ce0bdf"
ERROR_BYTES: Final = 14_745_600
ERROR_SHA256: Final = "89d09fbf1dc6a0bf8d1117287e2fbc5473e1a6c218e9975604c8fac94a9a3127"
BL1_BITS_TOTAL: Final = 910_209.2806090603
BL1_TOP1_BITS: Final = 876_748.5484900061
BL1_ERROR_OUTSIDE_TOP1_BITS: Final = 47_927.05428740731 - 47_893.52019232952
BL1_TOP1_POSITIONS: Final = 1_179_648
MS9_ERROR_POSITIONS: Final = 23_757
MS9_TOP1_ERROR_POSITIONS: Final = 21_548

GENERATOR_STREAMS: Final = ("road_undrivable", "lane", "movable", "mycar")
STREAMS: Final = (*GENERATOR_STREAMS, "residual")
STREAM_IDS: Final = {name: index + 1 for index, name in enumerate(STREAMS)}
ID_STREAMS: Final = {value: key for key, value in STREAM_IDS.items()}
CODERS: Final = ("brotli_q11", "zlib_9", "lzma2_extreme")

PACKET_MAGIC: Final = b"HG1P"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct("<4sBBH")
PACKET_ROW: Final = struct.Struct("<BBII32s32s")
COMPLETE_MAGIC: Final = b"HG1C"
COMPLETE_VERSION: Final = 1
COMPLETE_HEADER: Final = struct.Struct("<4sBIIII")

HORIZON_MAGIC: Final = b"HGH1"
HORIZON_HEADER: Final = struct.Struct("<4sBHHHH")
HORIZON_STEP: Final = 32
HORIZON_X: Final = np.asarray(
    (*tuple(range(0, WIDTH, HORIZON_STEP)), WIDTH - 1), dtype=np.int32
)
LANE_MAGIC: Final = b"HGL1"
MOVABLE_MAGIC: Final = b"HGM1"
MOVABLE_HEADER: Final = struct.Struct("<4sBHHH")
MOVABLE_ROW: Final = struct.Struct("<HHHHHH")
MYCAR_MAGIC: Final = b"HGC1"
MYCAR_HEADER: Final = struct.Struct("<4sBHHI")
RESIDUAL_MAGIC: Final = b"HGR1"
RESIDUAL_HEADER: Final = struct.Struct("<4sBBHHHQ")
RESIDUAL_ORDER_IDS: Final = {
    "frame_raster": 1,
    "class_frame_raster": 2,
    "tile8_time": 3,
    "tile16_time": 4,
    "tile32_time": 5,
    "tile64_time": 6,
    "class_tile16_time": 7,
    "pair_tile16": 8,
}
ID_RESIDUAL_ORDERS: Final = {value: key for key, value in RESIDUAL_ORDER_IDS.items()}


class HG1Error(RuntimeError):
    """A source, generator, coder, receiver, or custody invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"python_type": type(value).__name__}


def require_file(path: Path, *, byte_count: int, sha256: str, label: str) -> None:
    fact = et1.file_fact(path)
    if fact["bytes"] != byte_count or fact["sha256"] != sha256:
        raise HG1Error(f"{label} identity mismatch: {fact}")


def fact_matches(path: Path, fact: object) -> bool:
    """Return whether a checkpoint fact still names the retained exact bytes."""

    if not isinstance(fact, dict) or not path.is_file():
        return False
    try:
        return (
            int(fact["bytes"]) == path.stat().st_size
            and str(fact["sha256"]) == et1.sha256_path(path)
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def current_git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def retained_inventory(output_root: Path, manifest_path: Path) -> list[dict[str, object]]:
    """Inventory every persisted output other than the self-referential manifest."""

    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append(et1.file_fact(path))
    return rows


def put_uleb(output: bytearray, value: int) -> None:
    if value < 0:
        raise HG1Error("ULEB value must be non-negative")
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def get_uleb(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise HG1Error("truncated ULEB")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise HG1Error("overlong ULEB")


def zigzag(value: int) -> int:
    return value * 2 if value >= 0 else -value * 2 - 1


def unzigzag(value: int) -> int:
    return value // 2 if value % 2 == 0 else -(value // 2) - 1


def unpack_mask(path: Path, *, expected_bytes: int, expected_sha256: str) -> np.ndarray:
    require_file(path, byte_count=expected_bytes, sha256=expected_sha256, label=path.name)
    packed = np.fromfile(path, dtype=np.uint8)
    mask = np.unpackbits(packed, bitorder="little", count=TOTAL_POSITIONS)
    return mask.reshape(TOKEN_SHAPE).astype(bool, copy=False)


def fit_horizon_payload(tokens: np.ndarray) -> bytes:
    rows = np.empty((N_PAIRS, HORIZON_X.size), dtype="<u2")
    for pair in range(N_PAIRS):
        frame = np.asarray(tokens[pair])
        road = frame == CLASS_ROAD
        first = np.argmax(road, axis=0)
        absent = ~np.any(road, axis=0)
        first[absent] = HEIGHT
        first = ndimage.median_filter(first.astype(np.int32), size=17, mode="nearest")
        rows[pair] = np.clip(first[HORIZON_X], 0, HEIGHT).astype("<u2")
    return HORIZON_HEADER.pack(
        HORIZON_MAGIC, 1, N_PAIRS, HEIGHT, WIDTH, HORIZON_X.size
    ) + HORIZON_X.astype("<u2").tobytes() + rows.tobytes()


def render_horizon(payload: bytes, output: np.ndarray) -> None:
    if len(payload) < HORIZON_HEADER.size:
        raise HG1Error("horizon payload truncated")
    magic, version, pairs, height, width, knots = HORIZON_HEADER.unpack_from(payload)
    if (magic, version, pairs, height, width, knots) != (
        HORIZON_MAGIC, 1, N_PAIRS, HEIGHT, WIDTH, HORIZON_X.size
    ):
        raise HG1Error("horizon header mismatch")
    cursor = HORIZON_HEADER.size
    xs = np.frombuffer(payload, dtype="<u2", count=knots, offset=cursor).astype(np.float64)
    cursor += 2 * knots
    rows = np.frombuffer(payload, dtype="<u2", count=pairs * knots, offset=cursor)
    rows = rows.reshape(pairs, knots)
    if cursor + 2 * pairs * knots != len(payload) or np.any(np.diff(xs) <= 0):
        raise HG1Error("horizon payload length/order mismatch")
    x_full = np.arange(WIDTH, dtype=np.float64)
    y_grid = np.arange(HEIGHT, dtype=np.int32)[:, None]
    for pair in range(N_PAIRS):
        horizon = np.rint(np.interp(x_full, xs, rows[pair])).astype(np.int32)
        output[pair] = np.where(y_grid >= horizon[None, :], CLASS_ROAD, CLASS_UNDRIVABLE)


def fit_lane_payload(tokens: np.ndarray) -> tuple[bytes, dict[str, object]]:
    cfg = LaneBandRenderConfig()
    lines, fit = build_lane_band_pairs_from_lstars(tokens, cfg)
    blob, meta = serialize_lane_band_rd_tracked(
        lines, cfg, pack_mode="coherent_slot", smooth="none"
    )
    payload = LANE_MAGIC + struct.pack("<BI", 1, len(blob)) + blob
    return payload, {
        "pair_count": len(lines),
        "fit_summary": json_safe(fit),
        "serializer": json_safe(meta),
    }


def render_lane(payload: bytes, output: np.ndarray) -> None:
    if len(payload) < 9 or payload[:4] != LANE_MAGIC:
        raise HG1Error("lane payload truncated or bad magic")
    version, size = struct.unpack_from("<BI", payload, 4)
    blob = payload[9:]
    if version != 1 or len(blob) != size:
        raise HG1Error("lane payload length/version mismatch")
    decoded, header = deserialize_lane_band_any(blob)
    if len(decoded) != N_PAIRS:
        raise HG1Error("lane payload pair count mismatch")
    cfg = LaneBandRenderConfig(
        softness=float(header["softness"]),
        dash_gate=bool(header["dash_gate"]),
        dash_forward_max_m=float(header["dash_forward_max_m"]),
        v_h=float(header["v_h"]),
        cx=None if header.get("cx") is None else float(header["cx"]),
    )
    for pair, pair_lines in enumerate(decoded):
        coverage = rasterize_lane_coverage_range_dependent(
            pair_lines,
            h=HEIGHT,
            w=WIDTH,
            softness=cfg.softness,
            dash_gate=cfg.dash_gate,
            dash_forward_max_m=cfg.dash_forward_max_m,
            v_h=cfg.v_h,
            cx=cfg.cx,
        )
        output[pair, coverage >= 0.5] = CLASS_LANE


def component_boxes(frame: np.ndarray, minimum_area: int = 10) -> list[tuple[int, int, int, int]]:
    labelled, count = ndimage.label(frame == CLASS_MOVABLE, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labelled, max_label=count)
    boxes: list[tuple[int, int, int, int]] = []
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        area = int(np.count_nonzero(labelled[slices] == label_id))
        if area < minimum_area:
            continue
        y_slice, x_slice = slices
        boxes.append((y_slice.start, x_slice.start, y_slice.stop, x_slice.stop))
    return sorted(boxes, key=lambda box: ((box[1] + box[3]), (box[0] + box[2]), box))


def fit_movable_payload(tokens: np.ndarray) -> tuple[bytes, dict[str, object]]:
    active: dict[int, tuple[float, float, int]] = {}
    next_track = 1
    rows: list[tuple[int, int, int, int, int, int]] = []
    per_frame: list[int] = []
    for pair in range(N_PAIRS):
        boxes = component_boxes(np.asarray(tokens[pair]))
        unused = {track for track, (_, _, last) in active.items() if pair - last <= 3}
        frame_rows = []
        for y0, x0, y1, x1 in boxes:
            cy = 0.5 * (y0 + y1)
            cx = 0.5 * (x0 + x1)
            candidates = [
                (math.hypot(cy - active[track][0], cx - active[track][1]), track)
                for track in unused
            ]
            if candidates and min(candidates)[0] <= 60.0:
                _, track = min(candidates)
                unused.remove(track)
            else:
                track = next_track
                next_track += 1
            active[track] = (cy, cx, pair)
            frame_rows.append((pair, track, y0, x0, y1, x1))
        per_frame.append(len(frame_rows))
        rows.extend(sorted(frame_rows, key=lambda row: row[1:]))
    if next_track > 65_535:
        raise HG1Error("Movable track id overflow")
    header = MOVABLE_HEADER.pack(MOVABLE_MAGIC, 1, N_PAIRS, len(rows), next_track - 1)
    payload = header + b"".join(MOVABLE_ROW.pack(*row) for row in rows)
    return payload, {
        "rows": len(rows),
        "tracks": next_track - 1,
        "components_per_frame_min": min(per_frame),
        "components_per_frame_max": max(per_frame),
        "components_per_frame_mean": float(np.mean(per_frame)),
        "minimum_component_area": 10,
        "tracking_gate_px": 60.0,
    }


def render_movable(payload: bytes, output: np.ndarray) -> None:
    if len(payload) < MOVABLE_HEADER.size:
        raise HG1Error("Movable payload truncated")
    magic, version, pairs, count, _tracks = MOVABLE_HEADER.unpack_from(payload)
    if magic != MOVABLE_MAGIC or version != 1 or pairs != N_PAIRS:
        raise HG1Error("Movable header mismatch")
    if len(payload) != MOVABLE_HEADER.size + count * MOVABLE_ROW.size:
        raise HG1Error("Movable row length mismatch")
    seen: set[tuple[int, int]] = set()
    for index in range(count):
        pair, track, y0, x0, y1, x1 = MOVABLE_ROW.unpack_from(
            payload, MOVABLE_HEADER.size + index * MOVABLE_ROW.size
        )
        if (
            pair >= N_PAIRS
            or not (0 <= y0 < y1 <= HEIGHT and 0 <= x0 < x1 <= WIDTH)
            or (pair, track) in seen
        ):
            raise HG1Error("invalid/duplicate Movable row")
        seen.add((pair, track))
        output[pair, y0:y1, x0:x1] = CLASS_MOVABLE


def fit_mycar_payload(tokens: np.ndarray) -> tuple[bytes, dict[str, object]]:
    counts = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    for pair in range(N_PAIRS):
        counts += np.asarray(tokens[pair]) == CLASS_MYCAR
    mask = counts * 2 >= N_PAIRS
    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    payload = MYCAR_HEADER.pack(MYCAR_MAGIC, 1, HEIGHT, WIDTH, len(packed)) + packed
    return payload, {
        "static_seed_pixels": int(np.count_nonzero(mask)),
        "threshold": "present in at least 300 of 600 pairs",
    }


def render_mycar(payload: bytes, output: np.ndarray) -> None:
    if len(payload) < MYCAR_HEADER.size:
        raise HG1Error("MyCar payload truncated")
    magic, version, height, width, size = MYCAR_HEADER.unpack_from(payload)
    packed = payload[MYCAR_HEADER.size:]
    if (magic, version, height, width, size) != (MYCAR_MAGIC, 1, HEIGHT, WIDTH, len(packed)):
        raise HG1Error("MyCar header mismatch")
    mask = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little", count=HEIGHT * WIDTH)
    output[:, mask.reshape(HEIGHT, WIDTH).astype(bool)] = CLASS_MYCAR


def render_generators(streams: dict[str, bytes], output_path: Path) -> dict[str, object]:
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    generated = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=TOKEN_SHAPE)
    render_horizon(streams["road_undrivable"], generated)
    render_lane(streams["lane"], generated)
    render_movable(streams["movable"], generated)
    render_mycar(streams["mycar"], generated)
    generated.flush()
    del generated
    os.replace(temporary, output_path)
    return et1.file_fact(output_path)


def encode_residual(
    target: np.ndarray,
    generated: np.ndarray,
    output_path: Path,
    protected: np.ndarray | None,
    order: str,
) -> dict[str, object]:
    if order not in RESIDUAL_ORDER_IDS:
        raise HG1Error(f"unknown residual order: {order}")
    records: list[tuple[int, int]] = []
    for pair in range(N_PAIRS):
        mask = np.asarray(target[pair]) != np.asarray(generated[pair])
        if protected is not None:
            mask &= protected[pair]
        positions = np.flatnonzero(mask.reshape(-1))
        flat_target = np.asarray(target[pair]).reshape(-1)
        for position in positions.tolist():
            label = int(flat_target[position])
            if label < 0 or label > CLASS_MYCAR:
                raise HG1Error("residual target class outside [0,4]")
            records.append((pair * HEIGHT * WIDTH + position, label))
    if order == "frame_raster":
        records.sort(key=lambda row: row[0])
    elif order == "class_frame_raster":
        records.sort(key=lambda row: (row[1], row[0]))
    else:
        def tile_key(row: tuple[int, int]) -> tuple[int, ...]:
            address, label = row
            pair, position = divmod(address, HEIGHT * WIDTH)
            y, x = divmod(position, WIDTH)
            if order == "class_tile16_time":
                return label, y // 16, x // 16, pair, (y % 16) * 16 + x % 16
            if order == "pair_tile16":
                return pair, y // 16, x // 16, (y % 16) * 16 + x % 16, label
            tile_size = int(order.removeprefix("tile").removesuffix("_time"))
            return (
                y // tile_size,
                x // tile_size,
                pair,
                (y % tile_size) * tile_size + x % tile_size,
                label,
            )

        records.sort(key=tile_key)
    output = bytearray(
        RESIDUAL_HEADER.pack(
            RESIDUAL_MAGIC,
            2,
            RESIDUAL_ORDER_IDS[order],
            N_PAIRS,
            HEIGHT,
            WIDTH,
            len(records),
        )
    )
    previous = -1
    for address, label in records:
        put_uleb(output, zigzag(address - previous))
        output.append(label)
        previous = address
    et1.atomic_bytes(output_path, bytes(output))
    return {**et1.file_fact(output_path), "corrections": len(records), "order": order}


def apply_residual(payload: bytes, output: np.ndarray) -> int:
    if len(payload) < RESIDUAL_HEADER.size:
        raise HG1Error("residual payload truncated")
    magic, version, order_id, pairs, height, width, count = RESIDUAL_HEADER.unpack_from(payload)
    if (
        magic != RESIDUAL_MAGIC
        or version != 2
        or order_id not in ID_RESIDUAL_ORDERS
        or (pairs, height, width) != (N_PAIRS, HEIGHT, WIDTH)
    ):
        raise HG1Error("residual header mismatch")
    order = ID_RESIDUAL_ORDERS[order_id]
    offset = RESIDUAL_HEADER.size
    previous = -1
    previous_key: tuple[int, ...] | None = None
    flat_output = output.reshape(-1)
    for _ in range(count):
        coded_delta, offset = get_uleb(payload, offset)
        address = previous + unzigzag(coded_delta)
        if address < 0 or address >= TOTAL_POSITIONS or offset >= len(payload):
            raise HG1Error("residual address/label escaped field")
        label = payload[offset]
        offset += 1
        if label > CLASS_MYCAR:
            raise HG1Error("residual label outside [0,4]")
        if order == "frame_raster":
            key = (address,)
        elif order == "class_frame_raster":
            key = (label, address)
        elif order == "class_tile16_time":
            pair, position = divmod(address, HEIGHT * WIDTH)
            y, x = divmod(position, WIDTH)
            key = (label, y // 16, x // 16, pair, (y % 16) * 16 + x % 16)
        elif order == "pair_tile16":
            pair, position = divmod(address, HEIGHT * WIDTH)
            y, x = divmod(position, WIDTH)
            key = (pair, y // 16, x // 16, (y % 16) * 16 + x % 16, label)
        else:
            pair, position = divmod(address, HEIGHT * WIDTH)
            y, x = divmod(position, WIDTH)
            tile_size = int(order.removeprefix("tile").removesuffix("_time"))
            key = (
                y // tile_size,
                x // tile_size,
                pair,
                (y % tile_size) * tile_size + x % tile_size,
                label,
            )
        if previous_key is not None and key <= previous_key:
            raise HG1Error("residual order is non-canonical or contains a duplicate")
        flat_output[address] = label
        previous_key = key
        previous = address
    if offset != len(payload):
        raise HG1Error("residual payload has trailing bytes")
    return count


def coder_race(name: str, raw_path: Path, output_root: Path) -> dict[str, object]:
    raw = raw_path.read_bytes()
    rows: dict[str, object] = {}
    for coder in CODERS:
        directory = output_root / "retained" / "coder_races" / name / coder
        coded_path = directory / "payload.coded"
        repeat_path = directory / "payload.repeat.coded"
        started = time.monotonic()
        resumed = coded_path.is_file() and repeat_path.is_file()
        if resumed:
            coded = coded_path.read_bytes()
            repeated = repeat_path.read_bytes()
        else:
            coded = et1.compress_payload(raw, coder)
            repeated = et1.compress_payload(raw, coder)
            et1.atomic_bytes(coded_path, coded)
            et1.atomic_bytes(repeat_path, repeated)
        if coded != repeated or et1.decompress_payload(coded, coder) != raw:
            if resumed:
                resumed = False
                coded = et1.compress_payload(raw, coder)
                repeated = et1.compress_payload(raw, coder)
                et1.atomic_bytes(coded_path, coded)
                et1.atomic_bytes(repeat_path, repeated)
            if coded != repeated or et1.decompress_payload(coded, coder) != raw:
                raise HG1Error(f"{name}/{coder} failed deterministic exact coder race")
        rows[coder] = {
            "coder": coder,
            "seconds": time.monotonic() - started,
            "resumed_from_retained_payloads": resumed,
            "coded": et1.file_fact(coded_path),
            "repeat": et1.file_fact(repeat_path),
            "deterministic_repeat_equal": True,
            "raw_parseback_equal": True,
        }
    winner = min(CODERS, key=lambda coder: (int(rows[coder]["coded"]["bytes"]), CODERS.index(coder)))
    return {"name": name, "raw": et1.file_fact(raw_path), "coders": rows, "winner": winner}


def build_packet(races: Sequence[dict[str, object]], output_path: Path) -> dict[str, object]:
    rows = bytearray()
    bodies = bytearray()
    for race in races:
        name = str(race["name"])
        winner = str(race["winner"])
        raw_fact = race["raw"]
        coded_fact = race["coders"][winner]["coded"]
        raw_path = Path(str(raw_fact["path"]))
        coded_path = Path(str(coded_fact["path"]))
        raw = raw_path.read_bytes()
        coded = coded_path.read_bytes()
        rows.extend(PACKET_ROW.pack(
            STREAM_IDS[name], et1.CODER_IDS[winner], len(raw), len(coded),
            bytes.fromhex(sha256_bytes(raw)), bytes.fromhex(sha256_bytes(coded)),
        ))
        bodies.extend(coded)
    packet = PACKET_HEADER.pack(PACKET_MAGIC, PACKET_VERSION, len(races), 0) + rows + bodies
    et1.atomic_bytes(output_path, packet)
    return et1.file_fact(output_path)


def parse_packet(packet: bytes) -> dict[str, bytes]:
    if len(packet) < PACKET_HEADER.size:
        raise HG1Error("packet truncated")
    magic, version, count, reserved = PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC or version != PACKET_VERSION or count != len(STREAMS) or reserved:
        raise HG1Error("packet header mismatch")
    cursor = PACKET_HEADER.size
    row_values = []
    for _ in range(count):
        if cursor + PACKET_ROW.size > len(packet):
            raise HG1Error("packet roster truncated")
        row_values.append(PACKET_ROW.unpack_from(packet, cursor))
        cursor += PACKET_ROW.size
    streams: dict[str, bytes] = {}
    for stream_id, coder_id, raw_size, coded_size, raw_sha, coded_sha in row_values:
        if stream_id not in ID_STREAMS or coder_id not in et1.CODER_NAMES:
            raise HG1Error("packet roster enum invalid")
        coded = packet[cursor:cursor + coded_size]
        cursor += coded_size
        if len(coded) != coded_size or sha256_bytes(coded) != coded_sha.hex():
            raise HG1Error("packet coded stream identity mismatch")
        raw = et1.decompress_payload(coded, et1.CODER_NAMES[coder_id])
        if len(raw) != raw_size or sha256_bytes(raw) != raw_sha.hex():
            raise HG1Error("packet raw stream identity mismatch")
        name = ID_STREAMS[stream_id]
        if name in streams:
            raise HG1Error("duplicate packet stream")
        streams[name] = raw
    if cursor != len(packet) or set(streams) != set(STREAMS):
        raise HG1Error("packet roster/trailing-byte mismatch")
    return streams


def decode_packet_to_file(packet: bytes, output_path: Path) -> dict[str, object]:
    streams = parse_packet(packet)
    render_generators(streams, output_path)
    output = np.memmap(output_path, mode="r+", dtype=np.uint8, shape=TOKEN_SHAPE)
    corrections = apply_residual(streams["residual"], output)
    output.flush()
    del output
    return {**et1.file_fact(output_path), "corrections": corrections}


def build_complete_archive(
    output_path: Path,
    packet: bytes,
    semantic: bytes,
    carrier: bytes,
    compact_residual: bytes,
) -> None:
    header = COMPLETE_HEADER.pack(
        COMPLETE_MAGIC, COMPLETE_VERSION, len(semantic), len(carrier), len(compact_residual), len(packet)
    )
    member = header + semantic + carrier + compact_residual + packet
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, mode="w") as archive:
        archive.writestr(info, member)
    os.replace(temporary, output_path)


def parse_complete_archive(path: Path) -> tuple[dict[str, bytes], bytes]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["p"]:
            raise HG1Error("complete archive must contain only member p")
        member = archive.read("p")
    if len(member) < COMPLETE_HEADER.size:
        raise HG1Error("complete archive member truncated")
    magic, version, semantic_n, carrier_n, residual_n, packet_n = COMPLETE_HEADER.unpack_from(member)
    if magic != COMPLETE_MAGIC or version != COMPLETE_VERSION:
        raise HG1Error("complete archive magic/version mismatch")
    if len(member) != COMPLETE_HEADER.size + semantic_n + carrier_n + residual_n + packet_n:
        raise HG1Error("complete archive sections do not close")
    cursor = COMPLETE_HEADER.size
    sections = {}
    for name, size in (("semantic_renderer", semantic_n), ("pose_carrier", carrier_n), ("compact_residual", residual_n)):
        sections[name] = member[cursor:cursor + size]
        cursor += size
    return sections, member[cursor:]


def retain_hg1_framing(archive_path: Path, packet: bytes, output_path: Path) -> dict[str, object]:
    """Retain ZIP, HG1C, and HG1P roster bytes not charged to stream bodies."""

    archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("p")
        member = archive.read("p")
    local_offset = int(info.header_offset)
    if archive_bytes[local_offset:local_offset + 4] != b"PK\x03\x04":
        raise HG1Error("HG1 archive local ZIP header is malformed")
    filename_bytes = int.from_bytes(archive_bytes[local_offset + 26:local_offset + 28], "little")
    extra_bytes = int.from_bytes(archive_bytes[local_offset + 28:local_offset + 30], "little")
    member_start = local_offset + 30 + filename_bytes + extra_bytes
    member_end = member_start + int(info.compress_size)
    roster_bytes = PACKET_HEADER.size + len(STREAMS) * PACKET_ROW.size
    framing = (
        archive_bytes[:member_start]
        + member[:COMPLETE_HEADER.size]
        + packet[:roster_bytes]
        + archive_bytes[member_end:]
    )
    et1.atomic_bytes(output_path, framing)
    return et1.file_fact(output_path)


def mismatch_facts(left: np.ndarray, right: np.ndarray, protected: np.ndarray | None = None) -> dict[str, object]:
    total = 0
    protected_total = 0
    for pair in range(N_PAIRS):
        mismatch = np.asarray(left[pair]) != np.asarray(right[pair])
        total += int(np.count_nonzero(mismatch))
        if protected is not None:
            protected_total += int(np.count_nonzero(mismatch & protected[pair]))
    return {
        "mismatch_positions": total,
        "mismatch_fraction": total / TOTAL_POSITIONS,
        "protected_mismatch_positions": protected_total if protected is not None else None,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source_archive = args.source_archive.resolve()
    source_tokens_path = args.source_tokens.resolve()
    top1_path = args.top1_mask.resolve()
    error_path = args.error_mask.resolve()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.json"
    if args.resume_from is not None and args.resume_from.resolve() != manifest_path:
        raise HG1Error("--resume-from must name this output root's manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < args.minimum_free_bytes:
        raise HG1Error(f"APDataStore storage preflight failed: free={free_bytes}")
    et1.validate_sources(source_archive, source_tokens_path)
    require_file(top1_path, byte_count=TOP1_BYTES, sha256=TOP1_SHA256, label="BL1 top-1% mask")
    require_file(error_path, byte_count=ERROR_BYTES, sha256=ERROR_SHA256, label="MS9 final-error mask")
    section_paths = et1.extract_dx2_sections(source_archive, output_root / "retained" / "source_sections")
    resumed_manifest = False
    manifest: dict[str, object]
    if args.resume_from is not None and manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text())
        if not isinstance(loaded, dict) or loaded.get("schema") != "ddm_hg1_heterogeneous_analytic_generator_gate.v1":
            raise HG1Error("resume manifest schema mismatch")
        expected_inputs = {
            "source_archive": et1.file_fact(source_archive),
            "source_tokens": et1.file_fact(source_tokens_path),
            "top1_mask": et1.file_fact(top1_path),
            "error_mask": et1.file_fact(error_path),
        }
        for key, fact in expected_inputs.items():
            old = loaded.get(key)
            if not isinstance(old, dict) or old.get("sha256") != fact["sha256"] or old.get("bytes") != fact["bytes"]:
                raise HG1Error(f"resume manifest {key} identity mismatch")
        manifest = loaded
        resumed_manifest = True
    else:
        manifest = {
            "schema": "ddm_hg1_heterogeneous_analytic_generator_gate.v1",
            "axis": AXIS,
            "source_archive": et1.file_fact(source_archive),
            "source_tokens": et1.file_fact(source_tokens_path),
            "top1_mask": et1.file_fact(top1_path),
            "error_mask": et1.file_fact(error_path),
            "stages": {},
        }
    manifest["storage_preflight"] = {
        "free_bytes": free_bytes,
        "minimum_free_bytes": args.minimum_free_bytes,
    }
    manifest["resume_from"] = str(manifest_path)
    manifest["resumed_manifest"] = resumed_manifest
    manifest["provenance"] = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "git_head_before_serializer": current_git_head(),
        "runner": et1.file_fact(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": "NONE_NO_RNG",
        "config": {
            "token_shape": list(TOKEN_SHAPE),
            "horizon_step": HORIZON_STEP,
            "movable_minimum_component_area": 10,
            "movable_tracking_gate_px": 60.0,
            "residual_orders": list(RESIDUAL_ORDER_IDS),
            "coders": list(CODERS),
        },
    }
    manifest.setdefault("stages", {})
    manifest["stages"]["00_source_sections"] = {
        name: et1.file_fact(path) for name, path in section_paths.items()
    }
    et1.atomic_json(manifest_path, manifest)

    tokens = np.memmap(source_tokens_path, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
    generator_root = output_root / "retained" / "generators"
    generator_root.mkdir(parents=True, exist_ok=True)
    generator_path = generator_root / "generated_tokens.u8"
    prior_generator_stage = manifest["stages"].get("01_generators", {})
    prior_generator_payloads = (
        prior_generator_stage.get("payloads", {})
        if isinstance(prior_generator_stage, dict)
        else {}
    )
    generator_resumed = (
        isinstance(prior_generator_payloads, dict)
        and all(
            fact_matches(
                generator_root / f"{name}.raw",
                prior_generator_payloads.get(name),
            )
            for name in GENERATOR_STREAMS
        )
        and isinstance(prior_generator_stage, dict)
        and fact_matches(generator_path, prior_generator_stage.get("generated_tokens"))
    )
    if generator_resumed:
        raw_streams = {
            name: (generator_root / f"{name}.raw").read_bytes()
            for name in GENERATOR_STREAMS
        }
        generator_fact = et1.file_fact(generator_path)
        lane_diagnostics = prior_generator_stage["lane_diagnostics"]
        movable_diagnostics = prior_generator_stage["movable_diagnostics"]
        mycar_diagnostics = prior_generator_stage["mycar_diagnostics"]
    else:
        horizon = fit_horizon_payload(tokens)
        et1.atomic_bytes(generator_root / "road_undrivable.raw", horizon)
        lane, lane_diagnostics = fit_lane_payload(tokens)
        et1.atomic_bytes(generator_root / "lane.raw", lane)
        movable, movable_diagnostics = fit_movable_payload(tokens)
        et1.atomic_bytes(generator_root / "movable.raw", movable)
        mycar, mycar_diagnostics = fit_mycar_payload(tokens)
        et1.atomic_bytes(generator_root / "mycar.raw", mycar)
        raw_streams = {
            "road_undrivable": horizon,
            "lane": lane,
            "movable": movable,
            "mycar": mycar,
        }
        generator_fact = render_generators(raw_streams, generator_path)
    generated = np.memmap(generator_path, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
    generator_mismatch = mismatch_facts(tokens, generated)
    manifest["stages"]["01_generators"] = {
        "payloads": {name: et1.file_fact(generator_root / f"{name}.raw") for name in GENERATOR_STREAMS},
        "generated_tokens": generator_fact,
        "mismatch_vs_source": generator_mismatch,
        "lane_diagnostics": lane_diagnostics,
        "movable_diagnostics": movable_diagnostics,
        "mycar_diagnostics": mycar_diagnostics,
        "resumed_from_checkpoint": generator_resumed,
    }
    et1.atomic_json(manifest_path, manifest)

    top1 = unpack_mask(top1_path, expected_bytes=TOP1_BYTES, expected_sha256=TOP1_SHA256)
    final_error = unpack_mask(error_path, expected_bytes=ERROR_BYTES, expected_sha256=ERROR_SHA256)
    protected = top1 | final_error
    protected_positions = int(np.count_nonzero(protected))
    expected_protected = BL1_TOP1_POSITIONS + MS9_ERROR_POSITIONS - MS9_TOP1_ERROR_POSITIONS
    if protected_positions != expected_protected:
        raise HG1Error(f"protected-mask union mismatch: {protected_positions} != {expected_protected}")
    prior_residual_stage = manifest["stages"].get("02_residuals", {})
    prior_residual_payloads = (
        prior_residual_stage.get("payloads", {})
        if isinstance(prior_residual_stage, dict)
        else {}
    )
    residual_resumed = isinstance(prior_residual_payloads, dict)
    for candidate in ("exact", "protected"):
        candidate_rows = prior_residual_payloads.get(candidate, {}) if residual_resumed else {}
        if not isinstance(candidate_rows, dict):
            residual_resumed = False
            break
        for order in RESIDUAL_ORDER_IDS:
            row = candidate_rows.get(order)
            if not fact_matches(generator_root / f"residual_{candidate}_{order}.raw", row):
                residual_resumed = False
                break
    residual_payloads: dict[str, dict[str, dict[str, object]]] = {"exact": {}, "protected": {}}
    if residual_resumed:
        residual_payloads = prior_residual_payloads
    else:
        for candidate, mask in (("exact", None), ("protected", protected)):
            for order in RESIDUAL_ORDER_IDS:
                raw_path = generator_root / f"residual_{candidate}_{order}.raw"
                residual_payloads[candidate][order] = encode_residual(
                    tokens, generated, raw_path, mask, order
                )
    manifest["stages"]["02_residuals"] = {
        "payloads": residual_payloads,
        "allocation": {
            "protected_positions": protected_positions,
            "protected_position_fraction": protected_positions / TOTAL_POSITIONS,
            "approximated_positions": TOTAL_POSITIONS - protected_positions,
            "approximated_position_fraction": 1 - protected_positions / TOTAL_POSITIONS,
            "protected_incumbent_model_bits_lower_bound": BL1_TOP1_BITS + BL1_ERROR_OUTSIDE_TOP1_BITS,
            "protected_incumbent_model_bit_fraction_lower_bound": (BL1_TOP1_BITS + BL1_ERROR_OUTSIDE_TOP1_BITS) / BL1_BITS_TOTAL,
            "current_final_error_positions_protected": MS9_ERROR_POSITIONS,
            "sensitivity_status": "UNMEASURED: BL1/MS9 is spatial error coincidence, not an intervention map",
        },
        "resumed_from_checkpoint": residual_resumed,
    }
    et1.atomic_json(manifest_path, manifest)

    races: dict[str, dict[str, object]] = {}
    for name in GENERATOR_STREAMS:
        races[name] = coder_race(name, generator_root / f"{name}.raw", output_root)
    residual_races: dict[str, dict[str, dict[str, object]]] = {"exact": {}, "protected": {}}
    for candidate in residual_races:
        for order in RESIDUAL_ORDER_IDS:
            race_name = f"residual_{candidate}_{order}"
            residual_races[candidate][order] = coder_race(
                race_name, generator_root / f"{race_name}.raw", output_root
            )
    races["residuals"] = residual_races
    manifest["stages"]["03_coder_races"] = races
    et1.atomic_json(manifest_path, manifest)

    semantic = section_paths["semantic_renderer"].read_bytes()
    carrier = section_paths["pose_carrier"].read_bytes()
    compact = section_paths["compact_residual"].read_bytes()
    candidates: dict[str, object] = {}
    for candidate in ("exact", "protected"):
        candidate_root = output_root / "retained" / "candidates" / candidate
        candidate_root.mkdir(parents=True, exist_ok=True)
        selected_races = [races[name] for name in GENERATOR_STREAMS]
        selected_order = min(
            RESIDUAL_ORDER_IDS,
            key=lambda order: (
                int(
                    residual_races[candidate][order]["coders"]
                    [residual_races[candidate][order]["winner"]]["coded"]["bytes"]
                ),
                tuple(RESIDUAL_ORDER_IDS).index(order),
            ),
        )
        residual_source_race = residual_races[candidate][selected_order]
        residual_race = dict(residual_source_race)
        residual_race["name"] = "residual"
        selected_races.append(residual_race)
        packet_path = candidate_root / f"hg1_{candidate}.packet"
        packet_fact = build_packet(selected_races, packet_path)
        packet = packet_path.read_bytes()
        direct_path = candidate_root / "decoded_tokens.u8"
        direct_fact = decode_packet_to_file(packet, direct_path)
        direct = np.memmap(direct_path, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
        mismatch = mismatch_facts(tokens, direct, protected)
        if candidate == "exact" and mismatch["mismatch_positions"] != 0:
            raise HG1Error("exact heterogeneous packet failed source equality")
        if candidate == "protected" and mismatch["protected_mismatch_positions"] != 0:
            raise HG1Error("protected packet changed a protected concentration/error position")
        archive_path = candidate_root / f"candidate_hg1_{candidate}.zip"
        repeat_path = candidate_root / f"candidate_hg1_{candidate}.repeat.zip"
        build_complete_archive(archive_path, packet, semantic, carrier, compact)
        build_complete_archive(repeat_path, packet, semantic, carrier, compact)
        if archive_path.read_bytes() != repeat_path.read_bytes():
            raise HG1Error("complete archive repeat changed bytes")
        parsed_sections, parsed_packet = parse_complete_archive(archive_path)
        for name, payload in parsed_sections.items():
            if sha256_bytes(payload) != et1.EXPECTED_SECTION_SHA256[name]:
                raise HG1Error(f"complete archive changed inherited {name}")
        if parsed_packet != packet:
            raise HG1Error("complete archive changed HG1 packet")
        archive_decode_path = candidate_root / "archive_parseback_tokens.u8"
        archive_decode_fact = decode_packet_to_file(parsed_packet, archive_decode_path)
        if et1.sha256_path(archive_decode_path) != direct_fact["sha256"]:
            raise HG1Error("complete archive receiver differs from direct receiver")
        generator_bytes = sum(int(races[name]["coders"][races[name]["winner"]]["coded"]["bytes"]) for name in GENERATOR_STREAMS)
        residual_bytes = int(
            residual_source_race["coders"][residual_source_race["winner"]]["coded"]["bytes"]
        )
        framing_bytes = archive_path.stat().st_size - len(semantic) - len(carrier) - len(compact) - generator_bytes - residual_bytes
        framing = retain_hg1_framing(
            archive_path, packet, candidate_root / "container_framing.bin"
        )
        if framing["bytes"] != framing_bytes:
            raise HG1Error("candidate accounting did not close against retained framing")
        candidates[candidate] = {
            "packet": packet_fact,
            "direct_decode": direct_fact,
            "complete_archive": et1.file_fact(archive_path),
            "complete_archive_repeat": et1.file_fact(repeat_path),
            "complete_archive_repeat_equal": True,
            "archive_parseback_decode": archive_decode_fact,
            "mismatch_vs_source": mismatch,
            "generator_coded_bytes": generator_bytes,
            "residual_coded_bytes": residual_bytes,
            "residual_order": selected_order,
            "residual_coder": residual_source_race["winner"],
            "container_framing": framing,
            "archive_bytes": archive_path.stat().st_size,
            "bytes_over_fixed_distortion_cap": archive_path.stat().st_size - 137_986,
            "bytes_over_zero_distortion_cap": archive_path.stat().st_size - 180_218,
            "distortion": "UNCHANGED_BY_EXACT_FIELD" if candidate == "exact" else "UNMEASURED",
            "score_claim": False,
        }
        del direct
    manifest["stages"]["04_candidates"] = candidates
    manifest["final"] = {
        "prediction": "CONFIRMED" if int(candidates["exact"]["archive_bytes"]) >= 137_986 else "REFUTED",
        "prediction_number_bytes": int(candidates["exact"]["archive_bytes"]),
        "prediction_scope": "current-DX2 exact heterogeneous analytic generators plus one unique-home residual",
        "exact": candidates["exact"],
        "protected_byte_only": candidates["protected"],
        "fixed_distortion_cap_bytes": 137_986,
        "zero_distortion_cap_bytes": 180_218,
        "rate_exchange_s_per_byte": 6.658590e-7,
        "typed_member_rows": [
            {
                "member": name,
                "type": "BUILT",
                "raw": races[name]["raw"],
                "coder": races[name]["winner"],
                "coded": races[name]["coders"][races[name]["winner"]]["coded"],
            }
            for name in GENERATOR_STREAMS
        ],
    }
    manifest["stages"]["05_retained_inventory"] = {
        "files": retained_inventory(output_root, manifest_path),
        "exclusion": "manifest.json is self-referential and therefore omitted",
    }
    et1.atomic_json(manifest_path, manifest)
    del protected, final_error, top1, generated, tokens
    return manifest


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-tokens", type=Path, required=True)
    parser.add_argument("--top1-mask", type=Path, required=True)
    parser.add_argument("--error-mask", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--minimum-free-bytes", type=int, default=2 * 1024**3)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = run(args)
    print(json.dumps(result["final"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
