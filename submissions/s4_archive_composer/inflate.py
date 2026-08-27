#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Standalone Task #578 S4 witness receiver (numpy + brotli only).

All video-derived state is read from the monolithic ``0.bin``.  This file is a
generic interpreter: it contains no source frames, scorer weights, target
argmax table, or repository import.  Every binary grammar is exact-consuming
and hash-bound before any output is atomically published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import struct
import zlib
from pathlib import Path

import brotli
import numpy as np

PAIR_COUNT = 600
SCORER_H, SCORER_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
S4_MAGIC = b"S4A1\x00\x00"
S4_PREFIX = struct.Struct(">6sHH")
S4_SECTION = struct.Struct(">HBBQQ32s")
S4_NAMES = (
    "manifest.json",
    "seed.ppcs",
    "base.pbase3",
    "causal.pcr3",
    "events.pce3",
    "components.pcomp3",
)
S4_CODECS = {
    0: "raw",
    1: "zlib9",
    2: "brotli_q11",
    3: "lzma1_raw_1MiB",
    4: "mixed",
    5: "range_static_v1",
}
PPCS_NAMES = (
    "manifest",
    "native_grammar",
    "units",
    "ground_chart",
    "trajectory",
    "movable_tracks",
    "causal_jitter",
    "events",
    "pose_tightening",
    "constraint_seeds",
    "receiver",
    "authority",
)
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]


class DecodeError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
        "ascii"
    )


def parse_s4(payload: bytes) -> dict[str, tuple[bytes, str, int]]:
    if len(payload) < S4_PREFIX.size + 32 or hashlib.sha256(payload[:-32]).digest() != payload[-32:]:
        raise DecodeError("S4 container is truncated or outer-hash-invalid")
    magic, version, count = S4_PREFIX.unpack_from(payload)
    if magic != S4_MAGIC or version != 1 or count != len(S4_NAMES):
        raise DecodeError("S4 container header mismatch")
    cursor, limit, out = S4_PREFIX.size, len(payload) - 32, {}
    declarations = []
    for expected in S4_NAMES:
        if cursor + S4_SECTION.size > limit:
            raise DecodeError("S4 section header is truncated")
        nlen, registry, codec_id, encoded, decoded, digest = S4_SECTION.unpack_from(payload, cursor)
        cursor += S4_SECTION.size
        end = cursor + nlen + encoded
        if not nlen or end > limit or registry != 1 or codec_id not in S4_CODECS:
            raise DecodeError("S4 section metadata is invalid")
        name_bytes = payload[cursor : cursor + nlen]
        cursor += nlen
        body = payload[cursor:end]
        cursor = end
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise DecodeError("S4 section name is not ASCII") from exc
        if name != expected or hashlib.sha256(body).digest() != digest:
            raise DecodeError("S4 section order or digest mismatch")
        out[name] = (body, S4_CODECS[codec_id], decoded)
        if name != "manifest.json":
            declarations.append(
                {
                    "name": name,
                    "codec": S4_CODECS[codec_id],
                    "encoded_bytes": len(body),
                    "decoded_bytes": decoded,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "registry_version": registry,
                }
            )
    if cursor != limit:
        raise DecodeError("S4 container has trailing bytes")
    manifest_bytes = out["manifest.json"][0]
    try:
        manifest = json.loads(manifest_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodeError("S4 manifest is malformed") from exc
    if _canonical(manifest) != manifest_bytes or manifest.get("section_registry") != declarations:
        raise DecodeError("S4 manifest is noncanonical or disagrees with section headers")
    return out


def parse_ppcs(payload: bytes) -> dict:
    prefix, section = struct.Struct(">6sHH"), struct.Struct(">HQ")
    if len(payload) < prefix.size + 32 or hashlib.sha256(payload[:-32]).digest() != payload[-32:]:
        raise DecodeError("PPCS outer digest mismatch")
    magic, version, count = prefix.unpack_from(payload)
    if magic != b"PPCS1\x00" or version != 1 or count != len(PPCS_NAMES):
        raise DecodeError("PPCS header mismatch")
    cursor, limit, rows = prefix.size, len(payload) - 32, {}
    for expected in PPCS_NAMES:
        if cursor + section.size > limit:
            raise DecodeError("PPCS section header is truncated")
        nlen, plen = section.unpack_from(payload, cursor)
        cursor += section.size
        end = cursor + nlen + plen + 32
        if not nlen or end > limit:
            raise DecodeError("PPCS section length is invalid")
        name_bytes = payload[cursor : cursor + nlen]
        cursor += nlen
        raw = payload[cursor : cursor + plen]
        cursor += plen
        digest = payload[cursor : cursor + 32]
        cursor += 32
        try:
            name, value = name_bytes.decode("ascii"), json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DecodeError("PPCS section is malformed") from exc
        if name != expected or _canonical(value) != raw or hashlib.sha256(raw).digest() != digest:
            raise DecodeError("PPCS section order, canonical form, or digest mismatch")
        rows[name] = value
    if cursor != limit:
        raise DecodeError("PPCS has trailing bytes")
    manifest = rows.pop("manifest")
    seed = {
        "schema": manifest["schema"],
        "container": manifest["container"],
        "grammar": rows.pop("native_grammar"),
        "units": rows.pop("units"),
        "ground_chart": rows.pop("ground_chart"),
        "trajectory": rows.pop("trajectory"),
        "movable_tracks": rows.pop("movable_tracks"),
        "boundary_jitter": rows.pop("causal_jitter"),
        "events": rows.pop("events"),
        "pose_tightening": rows.pop("pose_tightening"),
        "constraint_seeds": rows.pop("constraint_seeds"),
        "receiver": rows.pop("receiver"),
        "authority": rows.pop("authority"),
    }
    geometry = seed["ground_chart"]["geometry"]
    if geometry != {
        "camera_height": CAMERA_H,
        "camera_width": CAMERA_W,
        "class_count": 5,
        "scorer_height": SCORER_H,
        "scorer_width": SCORER_W,
    } or seed["receiver"]["pair_count"] != PAIR_COUNT:
        raise DecodeError("PPCS geometry or pair count mismatch")
    return seed


def _terminal(payload: bytes, codec: str) -> bytes:
    if codec == "raw":
        return payload
    if codec == "zlib9":
        return zlib.decompress(payload)
    if codec == "brotli_q11":
        return brotli.decompress(payload)
    if codec == "lzma1_raw_1MiB":
        return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    if codec == "range_static_v1":
        if len(payload) < 10 or payload[:4] != b"RNG1":
            raise DecodeError("range-static terminal header mismatch")
        symbol_count, alphabet_count = struct.unpack_from(">IH", payload, 4)
        table_end = 10 + 4 * alphabet_count
        if not 1 <= alphabet_count <= 256 or table_end > len(payload):
            raise DecodeError("range-static terminal table is invalid")
        frequencies = list(struct.unpack_from(f">{alphabet_count}I", payload, 10))
        return range_decode_static(payload[table_end:], frequencies, symbol_count)
    raise DecodeError(f"unsupported terminal codec {codec!r}")


def range_decode_static(payload: bytes, frequencies: list[int], symbol_count: int) -> bytes:
    """Bit-exact standalone twin of the repository #557 static range decoder."""

    if (
        not frequencies
        or len(frequencies) > 256
        or any(value <= 0 for value in frequencies)
        or symbol_count < 0
    ):
        raise DecodeError("invalid arithmetic model")
    if symbol_count and not payload:
        raise DecodeError("encoded range stream is empty")
    total, cumulative = sum(frequencies), [0]
    for value in frequencies:
        cumulative.append(cumulative[-1] + value)
    byte_index = bit_index = 0

    def read_bit() -> int:
        nonlocal byte_index, bit_index
        if byte_index >= len(payload):
            return 0
        bit = (payload[byte_index] >> (7 - bit_index)) & 1
        bit_index += 1
        if bit_index == 8:
            bit_index = 0
            byte_index += 1
        return bit

    full, half, quarter, three_quarters = 1 << 32, 1 << 31, 1 << 30, 3 << 30
    low, high, code = 0, full - 1, 0
    for _ in range(32):
        code = (code << 1) | read_bit()
    out = bytearray()
    for _ in range(symbol_count):
        current_range = high - low + 1
        scaled = ((code - low + 1) * total - 1) // current_range
        if not 0 <= scaled < total:
            raise DecodeError("arithmetic code is inconsistent with the declared model")
        symbol = int(np.searchsorted(cumulative, scaled, side="right") - 1)
        if not 0 <= symbol < len(frequencies):
            raise DecodeError("arithmetic code is outside the declared model")
        out.append(symbol)
        high = low + current_range * cumulative[symbol + 1] // total - 1
        low += current_range * cumulative[symbol] // total
        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half
                high -= half
                code -= half
            elif low >= quarter and high < three_quarters:
                low -= quarter
                high -= quarter
                code -= quarter
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            code = (code << 1) | read_bit()
    return bytes(out)


def decode_base(payload: bytes) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...], list, dict]:
    if len(payload) < 8:
        raise DecodeError("PBASE3 is truncated")
    static_len, lane_len = struct.unpack_from("<II", payload)
    if len(payload) != 8 + static_len + lane_len:
        raise DecodeError("PBASE3 length mismatch")
    quotient = brotli.decompress(payload[8 : 8 + static_len])
    lane_raw = lzma.decompress(payload[8 + static_len :], format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    if len(quotient) < 10:
        raise DecodeError("PXQ1 is truncated")
    magic, height, width, edge_mask = struct.unpack_from(">4sHHH", quotient)
    count, packed = height * width, (height * width + 7) // 8
    if magic != b"PXQ1" or (height, width) != (SCORER_H, SCORER_W) or len(quotient) != 10 + 3 * packed:
        raise DecodeError("PXQ1 header or length mismatch")
    planes = []
    for index in range(3):
        start = 10 + index * packed
        planes.append(np.unpackbits(np.frombuffer(quotient[start : start + packed], np.uint8), bitorder="little")[:count])
    if np.any(planes[0] & planes[1]):
        raise DecodeError("PXQ1 Road and Undrivable masks overlap")
    ru = (planes[0].astype(np.uint8) + 2 * planes[1].astype(np.uint8)).reshape(height, width)
    hood = planes[2].reshape(height, width).astype(bool)
    edges, bit = [], 0
    for left in range(5):
        for right in range(left + 1, 5):
            if edge_mask & (1 << bit):
                edges.append((left, right))
            bit += 1
    lane_pairs, lane_header = decode_lane(lane_raw)
    return ru, hood, tuple(edges), lane_pairs, lane_header


def decode_lane(payload: bytes) -> tuple[list[list[np.ndarray]], dict]:
    if not payload.startswith(b"LBND2\x00") or len(payload) < 14:
        raise DecodeError("LBND2 header mismatch")
    offset = 6
    hlen = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    if offset + hlen + 4 > len(payload):
        raise DecodeError("LBND2 header is truncated")
    raw_header = payload[offset : offset + hlen]
    offset += hlen
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodeError("LBND2 JSON header is malformed") from exc
    plen = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    if offset + plen > len(payload):
        raise DecodeError("LBND2 presence stream is truncated")
    presence_raw = payload[offset : offset + plen]
    offset += plen
    rd = header["rd"]
    pairs, slots, dslot = int(rd["n_pairs"]), int(rd["K"]), int(rd["d_slot"])
    if pairs != PAIR_COUNT or dslot != 11 or not 0 <= slots <= 32:
        raise DecodeError("LBND2 geometry mismatch")
    expected_words = pairs * slots * dslot
    if len(payload) - offset != expected_words * 4 or plen != (pairs * slots + 7) // 8:
        raise DecodeError("LBND2 stream length mismatch")
    if slots:
        presence = np.unpackbits(np.frombuffer(presence_raw, np.uint8))[: pairs * slots].reshape(pairs, slots)
        encoded = np.frombuffer(payload, np.uint32, expected_words, offset).astype(np.int64).reshape(pairs, -1)
        delta = (encoded >> 1) ^ -(encoded & 1)
        values = np.cumsum(delta, axis=0).astype(np.float64) * np.tile(
            np.asarray(rd["base_steps"], np.float64), slots
        )
    else:
        presence, values = np.zeros((pairs, 0), bool), np.zeros((pairs, 0), np.float64)
    decoded = [
        [values[pair, slot * 11 : (slot + 1) * 11].copy() for slot in range(slots) if presence[pair, slot]]
        for pair in range(pairs)
    ]
    return decoded, header


def lane_mask(lines: list[np.ndarray], header: dict, camera: dict) -> np.ndarray:
    rows = np.arange(SCORER_H, dtype=np.float64)
    cols = np.arange(SCORER_W, dtype=np.float64)[None, :]
    horizon, cx = float(header["v_h"]), float(header["cx"] if header.get("cx") is not None else SCORER_W / 2)
    softness = max(float(header["softness"]), 1e-6)
    coverage = np.zeros((SCORER_H, SCORER_W), np.float64)
    below = rows > horizon + 1.0
    selected = rows[below]
    forward = float(camera["height_m"]) * float(camera["fy_scorer"]) / np.maximum(
        selected - horizon, 1e-3
    )
    for vector in lines:
        center = cx - np.polyval(vector[:4], forward) * float(camera["fx_scorer"]) / forward
        half_width = np.maximum(np.polyval(vector[4:6], selected), 0.5)
        valid = (forward >= vector[9] - 1.0) & (forward <= vector[10] + 5.0)
        dash = np.ones_like(forward, bool)
        if bool(header["dash_gate"]) and vector[6] > 0:
            near = forward < float(header["dash_forward_max_m"])
            phase = np.mod(forward - vector[7], vector[6]) / vector[6]
            dash = np.where(near, phase < vector[8], True)
        signed = half_width[:, None] - np.abs(cols - center[:, None])
        candidate = np.clip(signed / softness + 0.5, 0.0, 1.0) * (valid & dash)[:, None]
        coverage[below] = np.maximum(coverage[below], candidate)
    return coverage >= 0.5


def _uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise DecodeError("varint is truncated or overlong")
        byte, offset = payload[offset], offset + 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
        shift += 7


def decode_events(payload: bytes, codec: str, expected_raw: int) -> list[list[tuple[int, np.ndarray]]]:
    raw = _terminal(payload, codec)
    if len(raw) != expected_raw or len(raw) < 10:
        raise DecodeError("PCE3 decoded length mismatch")
    magic, frames, height, width = struct.unpack_from("<4sHHH", raw)
    if magic != b"PCE3" or (frames, height, width) != (PAIR_COUNT, SCORER_H, SCORER_W):
        raise DecodeError("PCE3 header mismatch")
    offset, previous, result = 10, [[] for _ in range(5)], []
    for _ in range(PAIR_COUNT):
        frame_rows, current = [], []
        for class_id in range(5):
            count, offset = _uvarint(raw, offset)
            class_rows = []
            for _ in range(count):
                if offset >= len(raw):
                    raise DecodeError("PCE3 event is truncated")
                event_type, offset = raw[offset], offset + 1
                if event_type == 1:
                    prior, offset = _uvarint(raw, offset)
                    if prior >= len(previous[class_id]):
                        raise DecodeError("PCE3 prior reference is invalid")
                    base = previous[class_id][prior]
                elif event_type == 0:
                    base = np.empty(0, dtype=np.int64)
                else:
                    raise DecodeError("PCE3 event type is unknown")
                value_count, offset = _uvarint(raw, offset)
                values = []
                if value_count:
                    first, offset = _uvarint(raw, offset)
                    values.append(first)
                    for _ in range(value_count - 1):
                        delta, offset = _uvarint(raw, offset)
                        if delta <= 0:
                            raise DecodeError("PCE3 deltas must be positive")
                        values.append(values[-1] + delta)
                sites = np.setxor1d(base, np.asarray(values, np.int64), assume_unique=True)
                if not len(sites) or sites[-1] >= SCORER_H * SCORER_W:
                    raise DecodeError("PCE3 component is empty or out of range")
                class_rows.append(sites)
                frame_rows.append((class_id, sites))
            current.append(class_rows)
        previous = current
        result.append(frame_rows)
    if offset != len(raw):
        raise DecodeError("PCE3 has trailing bytes")
    return result


def _decode_component_raw(raw: bytes) -> tuple[int, int, np.ndarray]:
    if len(raw) < 12:
        raise DecodeError("PCOMP3 component header is truncated")
    frame, class_id, _stratum, count, first = struct.unpack_from("<HBBII", raw)  # DEAD_BYTES_AUDIT_OK:stratum byte is PCOMP3 stream metadata consumed by sibling receivers (predictor_r4_tailrace routes streams by (class_id, stratum_id)); this receiver reconstructs by (frame, class_id) alone, the byte stays charged in the 12B header
    if frame >= PAIR_COUNT or class_id >= 5 or count == 0:
        raise DecodeError("PCOMP3 component metadata is invalid")
    cursor, values = 12, [first]
    for _ in range(count - 1):
        delta, cursor = _uvarint(raw, cursor)
        if delta <= 0:
            raise DecodeError("PCOMP3 component deltas must be positive")
        values.append(values[-1] + delta)
    if cursor != len(raw) or values[-1] >= SCORER_H * SCORER_W:
        raise DecodeError("PCOMP3 packet has trailing or out-of-grid data")
    return frame, class_id, np.asarray(values, np.int64)


def decode_components(payload: bytes, codec: str, expected_raw: int) -> list[list[tuple[int, np.ndarray]]]:
    frames = [[] for _ in range(PAIR_COUNT)]
    if codec == "range_static_v1":
        raw = _terminal(payload, codec)
        if len(raw) != expected_raw:
            raise DecodeError("range-coded PCOMP3 decoded length mismatch")
        offset = 0
        while offset < len(raw):
            if offset + 4 > len(raw):
                raise DecodeError("range-coded PCOMP3 record length is truncated")
            size = struct.unpack_from("<I", raw, offset)[0]
            offset += 4
            if not size or offset + size > len(raw):
                raise DecodeError("range-coded PCOMP3 record size is invalid")
            frame, class_id, sites = _decode_component_raw(raw[offset : offset + size])
            offset += size
            frames[frame].append((class_id, sites))
        return frames
    if codec != "zlib9":
        raise DecodeError(f"unsupported PCOMP3 codec {codec!r}")
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise DecodeError("PCOMP3 packet length is truncated")
        size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        if not size or offset + size > len(payload):
            raise DecodeError("PCOMP3 packet size is invalid")
        try:
            raw = zlib.decompress(payload[offset : offset + size])
        except zlib.error as exc:
            raise DecodeError("PCOMP3 zlib packet is invalid") from exc
        offset += size
        frame, class_id, sites = _decode_component_raw(raw)
        frames[frame].append((class_id, sites))
    return frames


def _catmull(values: list[int], times: list[int], time: int) -> float:
    if time <= times[0]:
        return float(values[0])
    if time >= times[-1]:
        return float(values[-1])
    right = next(index for index, value in enumerate(times) if value >= time)
    left = right - 1
    p0, p1, p2, p3 = map(float, (values[max(0, left - 1)], values[left], values[right], values[min(len(values) - 1, right + 1)]))
    u = (time - times[left]) / (times[right] - times[left])
    return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3)


def trajectory(seed: dict, time: int) -> tuple[float, float, float]:
    rows = seed["trajectory"]["controls"]
    times = [row["time"] for row in rows]
    residual = next((row for row in seed["trajectory"]["ar_residuals"] if row["time"] == time), {})
    values = []
    for key, delta in (("tx_q", "dtx_q"), ("ty_q", "dty_q"), ("yaw_q", "dyaw_q")):
        values.append(_catmull([row[key] for row in rows], times, time) + residual.get(delta, 0))
    return values[0] / 256.0, values[1] / 256.0, values[2] / 1_048_576.0


def shift(grid: np.ndarray, tx: float, ty: float, yaw: float) -> np.ndarray:
    yy, xx = np.indices(grid.shape, dtype=np.float64)
    cy, cx = (grid.shape[0] - 1) / 2, (grid.shape[1] - 1) / 2
    cosine, sine = math.cos(-yaw), math.sin(-yaw)
    dx, dy = xx - cx - tx, yy - cy - ty
    sx = np.clip(np.rint(cosine * dx - sine * dy + cx).astype(np.int64), 0, grid.shape[1] - 1)
    sy = np.clip(np.rint(sine * dx + cosine * dy + cy).astype(np.int64), 0, grid.shape[0] - 1)
    return grid[sy, sx]


def apply_tracks(field: np.ndarray, tracks: list[dict], time: int) -> None:
    for track in tracks:
        knots = track["knots"]
        if time < knots[0]["time"] or time > knots[-1]["time"]:
            continue
        right = next((index for index, knot in enumerate(knots) if knot["time"] >= time), len(knots) - 1)
        left = max(0, right - 1)
        span = knots[right]["time"] - knots[left]["time"]
        alpha = 0.0 if span == 0 else (time - knots[left]["time"]) / span
        interpolated = {
            key: (1 - alpha) * knots[left][key] + alpha * knots[right][key]
            for key in ("y_q", "x_q", "height_q", "width_q")
        }
        cy, cx = interpolated["y_q"] / 256, interpolated["x_q"] / 256
        height = max(1, round(interpolated["height_q"] / 256))
        width = max(1, round(interpolated["width_q"] / 256))
        y0, x0 = max(0, round(cy - height / 2)), max(0, round(cx - width / 2))
        field[y0 : min(SCORER_H, y0 + height), x0 : min(SCORER_W, x0 + width)] = track["cell_id"]


def support_indices(n_in: int, n_out: int) -> np.ndarray:
    source = np.clip((np.arange(n_out, dtype=np.float64) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
    low = np.floor(source).astype(np.int64)
    high = np.minimum(low + 1, n_in - 1)
    indices = np.stack((low, high), axis=1)
    if np.unique(indices).size != indices.size:
        raise DecodeError("factor-2 supports overlap")
    return indices


def realize(
    field: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    rgb = palette[field]
    frame = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    for row_offset in range(2):
        for col_offset in range(2):
            frame[row_indices[:, row_offset, None], col_indices[None, :, col_offset], :] = rgb
    return frame


def prepare(archive_payload: bytes) -> dict:
    sections = parse_s4(archive_payload)
    manifest = json.loads(sections["manifest.json"][0].decode("ascii"))
    palette = np.asarray(manifest["weight_derived_constants"]["R2_max_margin_palette"]["value_u8"], dtype=np.uint8)
    if palette.shape != (5, 3):
        raise DecodeError("counted weight-derived palette shape mismatch")
    lane_camera = manifest["video_derived_constants"]["lane_camera_intrinsics"]["value"]
    if set(lane_camera) != {"height_m", "fx_scorer", "fy_scorer"}:
        raise DecodeError("counted lane camera constant schema mismatch")
    seed = parse_ppcs(sections["seed.ppcs"][0])
    ru, hood, edges, lanes, lane_header = decode_base(sections["base.pbase3"][0])
    causal = sections["causal.pcr3"]
    if causal[0] or causal[2] != 0:
        raise DecodeError("this receiver version expects the selected zero-parameter causal policy")
    events = decode_events(
        sections["events.pce3"][0], sections["events.pce3"][1], sections["events.pce3"][2]
    )
    components = decode_components(
        sections["components.pcomp3"][0],
        sections["components.pcomp3"][1],
        sections["components.pcomp3"][2],
    )
    constraints = [[] for _ in range(PAIR_COUNT)]
    for row in seed["constraint_seeds"]:
        constraints[row["time"]].append(row)
    return {
        "seed": seed,
        "ru": ru,
        "hood": hood,
        "edges": edges,
        "lanes": lanes,
        "lane_header": lane_header,
        "lane_camera": lane_camera,
        "events": events,
        "components": components,
        "constraints": constraints,
        "palette": palette,
    }


def decode(archive_payload: bytes, output: Path | None, max_pairs: int) -> dict:
    if not 1 <= max_pairs <= PAIR_COUNT:
        raise DecodeError("max_pairs must be in [1,600]")
    state = prepare(archive_payload)
    rows, cols = support_indices(CAMERA_H, SCORER_H), support_indices(CAMERA_W, SCORER_W)
    digest, pair_hashes, previous, previous_pose = hashlib.sha256(), [], None, (0.0, 0.0, 0.0)
    temporary = None if output is None else output.with_name(f".{output.name}.{os.getpid()}.tmp")
    handle = None
    try:
        if temporary is not None:
            temporary.parent.mkdir(parents=True, exist_ok=True)
            handle = temporary.open("wb")
        for pair in range(max_pairs):
            pose = trajectory(state["seed"], pair)
            if previous is None:
                field = np.zeros((SCORER_H, SCORER_W), dtype=np.uint8)
                field[state["ru"] == 2] = 2
            else:
                field = shift(previous, pose[0] - previous_pose[0], pose[1] - previous_pose[1], pose[2] - previous_pose[2])
            static = np.isin(field, (0, 2)) & (state["ru"] != 0)
            field[static] = np.where(state["ru"][static] == 1, 0, 2).astype(np.uint8)
            field[
                lane_mask(state["lanes"][pair], state["lane_header"], state["lane_camera"])
            ] = 1
            field[state["hood"]] = 4
            apply_tracks(field, state["seed"]["movable_tracks"], pair)
            flat = field.reshape(-1)
            for class_id, sites in state["events"][pair]:
                flat[sites] = class_id
            for class_id, sites in state["components"][pair]:
                flat[sites] = class_id
            for row in state["constraints"][pair]:
                field[row["y"], row["x"]] = row["cell_id"]
            frame = realize(field, rows, cols, state["palette"])
            raw = frame.tobytes(order="C")
            pair_digest = hashlib.sha256(raw + raw).hexdigest()
            pair_hashes.append(pair_digest)
            digest.update(raw)
            digest.update(raw)
            if handle is not None:
                handle.write(raw)
                handle.write(raw)
            previous, previous_pose = field.copy(), pose
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            expected = max_pairs * 2 * CAMERA_H * CAMERA_W * 3
            if temporary.stat().st_size != expected:
                raise DecodeError("inflated raw byte count mismatch")
            os.replace(temporary, output)
        return {
            "schema": "s4_standalone_decode_receipt.v1",
            "pairs": max_pairs,
            "output_bytes": max_pairs * 2 * CAMERA_H * CAMERA_W * 3,
            "stream_sha256": digest.hexdigest(),
            "first_pair_sha256": pair_hashes[0],
            "last_pair_sha256": pair_hashes[-1],
            "atomic_output": output is not None,
            "scorer_invocations": 0,
        }
    finally:
        if handle is not None:
            handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--max-pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--hash-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.hash_only and args.output is not None:
        raise DecodeError("--hash-only cannot publish an output file")
    if not args.hash_only and args.output is None:
        raise DecodeError("output path is required unless --hash-only is selected")
    receipt = decode(args.archive.read_bytes(), None if args.hash_only else args.output, args.max_pairs)
    text = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_name(f".{args.receipt.name}.{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, args.receipt)
    print(text, end="")


if __name__ == "__main__":
    main()
