#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_cg3 -- counted-GT surgical correction pricing and bounded realization.

Axis: [macOS-CPU advisory] NON-PROMOTABLE.  This script fires no full n600
scorer job.  Full-population pricing is cache-derived from the live cx1 n600
GT/cx1 argmax arrays; realization is bounded to an explicit sample of pairs and
uses the frozen CPU SegNet on the actual inflated cx1 raw frames.

The correction objects are descriptions, not submissions.  They are priced with
real coders (Brotli-Q11, raw LZMA1, and an R7 SMEVR nibble framing) and then the
description->realized survival ratio is measured separately on a bounded sample.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    import brotli
except ImportError:  # pragma: no cover - environment dependent
    brotli = None  # type: ignore[assignment]

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover - fail closed on the lab env
    raise SystemExit("ddm_cg3 requires scipy.ndimage for real component labels") from exc


REPO = Path(__file__).resolve().parents[1]
ARGMAX_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
GT_ARGMAX = ARGMAX_DIR / "gt_argmax_n600.npy"
CX1_ARGMAX = ARGMAX_DIR / "cx1_argmax_n600.npy"
CX1_RECEIPT = ARGMAX_DIR / "cx1_directed_flip_receipt.json"
RAW_0 = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2/inflated/0.raw")
OUT_DIR = REPO / ".omx" / "research" / "ddm_cg3_20260804"
OUT_JSON = OUT_DIR / "ddm_cg3_counted_gt_recovery_receipt.json"
ARTIFACT_DIR = OUT_DIR / "artifacts"

SEG_H = 384
SEG_W = 512
CAM_H = 874
CAM_W = 1164
N_PAIRS = 600
PIXELS_PER_PAIR = SEG_H * SEG_W
PIXELS_N600 = N_PAIRS * PIXELS_PER_PAIR
RATE_DENOM = 37_545_489
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
EXPECTED_TOTAL_FLIPS = 508_640
EXPECTED_DSEG = 0.004311794704861111
OWN_VEHICLE_S = 0.7910689
OWN_VEHICLE_BYTES = 353_805


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _rate_s(bytes_: int | float) -> float:
    return 25.0 * float(bytes_) / RATE_DENOM


def _seg_s(pixels: int | float, *, n_pairs: int = N_PAIRS) -> float:
    return 100.0 * float(pixels) / (float(n_pairs) * PIXELS_PER_PAIR)


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode a negative integer")
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _packbits(mask: np.ndarray) -> bytes:
    return np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1)).tobytes()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return y0, x0, y1 - y0, x1 - x0


def _pack_bbox(y0: int, x0: int, h: int, w: int) -> bytes:
    if not (0 <= y0 < SEG_H and 0 <= x0 < SEG_W and 0 < h <= SEG_H and 0 < w <= SEG_W):
        raise ValueError(f"bad bbox {(y0, x0, h, w)}")
    return struct.pack("<4H", y0, x0, h, w)


def _serialize_crop_mask(mask: np.ndarray) -> bytes:
    box = _bbox(mask)
    if box is None:
        return b"\x00"
    y0, x0, h, w = box
    crop = np.asarray(mask[y0 : y0 + h, x0 : x0 + w], dtype=np.uint8)
    return b"\x01" + _pack_bbox(y0, x0, h, w) + _packbits(crop)


def _decode_crop_mask(record: bytes) -> np.ndarray:
    out = np.zeros((SEG_H, SEG_W), dtype=bool)
    if record == b"\x00":
        return out
    if not record or record[0] != 1 or len(record) < 9:
        raise ValueError("bad crop record")
    y0, x0, h, w = struct.unpack("<4H", record[1:9])
    n = h * w
    body = record[9:]
    bits = np.unpackbits(np.frombuffer(body, dtype=np.uint8))[:n].reshape(h, w)
    out[y0 : y0 + h, x0 : x0 + w] = bits.astype(bool)
    return out


def _serialize_edge_mask(gt: np.ndarray, rd: np.ndarray, a: int, b: int) -> bytes:
    mask = (gt != rd) & (((gt == a) & (rd == b)) | ((gt == b) & (rd == a)))
    box = _bbox(mask)
    if box is None:
        return b"\x00"
    y0, x0, h, w = box
    crop_mask = np.asarray(mask[y0 : y0 + h, x0 : x0 + w], dtype=np.uint8)
    crop_gt = np.asarray(gt[y0 : y0 + h, x0 : x0 + w], dtype=np.uint8)
    target_bits = crop_gt[crop_mask.astype(bool)] == b
    return (
        b"\x02"
        + struct.pack("<BB", a, b)
        + _pack_bbox(y0, x0, h, w)
        + _varint(int(crop_mask.sum()))
        + _packbits(crop_mask)
        + _packbits(target_bits)
    )


def _decode_edge_record(record: bytes) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((SEG_H, SEG_W), dtype=bool)
    target = np.zeros((SEG_H, SEG_W), dtype=np.uint8)
    if record == b"\x00":
        return mask, target
    if not record or record[0] != 2 or len(record) < 11:
        raise ValueError("bad edge record")
    a = int(record[1])
    b = int(record[2])
    y0, x0, h, w = struct.unpack("<4H", record[3:11])
    off = 11
    shift = 0
    count = 0
    while True:
        raw = record[off]
        off += 1
        count |= (raw & 0x7F) << shift
        if not (raw & 0x80):
            break
        shift += 7
    crop_n = h * w
    bm_len = (crop_n + 7) // 8
    crop_mask = (
        np.unpackbits(np.frombuffer(record[off : off + bm_len], dtype=np.uint8))[:crop_n]
        .reshape(h, w)
        .astype(bool)
    )
    off += bm_len
    side_len = (count + 7) // 8
    side = np.unpackbits(np.frombuffer(record[off : off + side_len], dtype=np.uint8))[:count]
    if int(crop_mask.sum()) != count:
        raise ValueError("edge record count mismatch")
    crop_target = np.full((h, w), a, dtype=np.uint8)
    crop_target[crop_mask] = np.where(side.astype(bool), b, a).astype(np.uint8)
    mask[y0 : y0 + h, x0 : x0 + w] = crop_mask
    target[y0 : y0 + h, x0 : x0 + w] = crop_target
    return mask, target


def _serialize_component_masks(mask: np.ndarray, select_mask: np.ndarray) -> tuple[bytes, int, int]:
    labels, n_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    selected: list[tuple[int, int, int, int, np.ndarray]] = []
    total_pixels = 0
    for label_id in range(1, int(n_labels) + 1):
        comp = labels == label_id
        if not bool(np.any(comp & select_mask)):
            continue
        box = _bbox(comp)
        if box is None:
            continue
        y0, x0, h, w = box
        crop = np.asarray(comp[y0 : y0 + h, x0 : x0 + w], dtype=np.uint8)
        selected.append((y0, x0, h, w, crop))
        total_pixels += int(crop.sum())
    out = bytearray(b"\x03" + _varint(len(selected)))
    for y0, x0, h, w, crop in selected:
        out += _pack_bbox(y0, x0, h, w)
        out += _packbits(crop)
    return bytes(out), len(selected), total_pixels


def _decode_component_record(record: bytes) -> np.ndarray:
    if not record or record[0] != 3:
        raise ValueError("bad component record")
    out = np.zeros((SEG_H, SEG_W), dtype=bool)
    off = 1
    shift = 0
    count = 0
    while True:
        raw = record[off]
        off += 1
        count |= (raw & 0x7F) << shift
        if not (raw & 0x80):
            break
        shift += 7
    for _ in range(count):
        y0, x0, h, w = struct.unpack("<4H", record[off : off + 8])
        off += 8
        n = h * w
        bm_len = (n + 7) // 8
        crop = (
            np.unpackbits(np.frombuffer(record[off : off + bm_len], dtype=np.uint8))[:n]
            .reshape(h, w)
            .astype(bool)
        )
        off += bm_len
        out[y0 : y0 + h, x0 : x0 + w] |= crop
    if off != len(record):
        raise ValueError("component record trailing bytes")
    return out


def _frame_records(records: list[bytes], *, surface_id: str) -> bytes:
    sid = surface_id.encode("utf-8")
    out = bytearray(b"CG3R1" + _varint(len(sid)) + sid + _varint(len(records)))
    for record in records:
        out += _varint(len(record))
        out += record
    return bytes(out)


def _brotli_q11(payload: bytes) -> bytes:
    if brotli is None:
        raise RuntimeError("brotli package is unavailable")
    return bytes(brotli.compress(payload, quality=11))


def _lzma1_raw(payload: bytes) -> bytes:
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=filters)


def _unlzma1_raw(payload: bytes, expected_len: int) -> bytes:
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)
    got = dec.decompress(payload, max_length=expected_len + 1)
    if len(got) != expected_len or not dec.eof or dec.unused_data:
        raise ValueError("LZMA1 roundtrip length/termination mismatch")
    return got


def _load_r7_module():
    path = REPO / "experiments" / "ddm_r7_token_coder.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ddm_r7_token_coder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_R7 = None


def _r7():
    global _R7
    if _R7 is None:
        _R7 = _load_r7_module()
    return _R7


def _bytes_to_nibbles(byte_matrix: np.ndarray) -> np.ndarray:
    hi = byte_matrix >> 4
    lo = byte_matrix & 15
    return np.stack([hi, lo], axis=2).astype(np.uint8)


def _nibbles_to_bytes(nibbles: np.ndarray) -> np.ndarray:
    if nibbles.shape[2] != 2:
        raise ValueError("nibble matrix shape mismatch")
    return ((nibbles[:, :, 0].astype(np.uint16) << 4) | nibbles[:, :, 1].astype(np.uint16)).astype(
        np.uint8
    )


def _smevr_records(records: list[bytes]) -> bytes:
    """Use the landed R7 SMEVR coder as a real nibble-frame codec for records.

    Each pair is a fixed-width byte row: u32 length plus payload, zero padded.
    The row matrix is split only if the R7 production max-value guard requires
    it.  Decode is verified by the caller.
    """

    r7 = _r7()
    n = len(records)
    max_len = max((len(record) for record in records), default=0)
    cols_total = max_len + 4
    matrix = np.zeros((n, cols_total), dtype=np.uint8)
    for i, record in enumerate(records):
        matrix[i, :4] = np.frombuffer(struct.pack("<I", len(record)), dtype=np.uint8)
        matrix[i, 4 : 4 + len(record)] = np.frombuffer(record, dtype=np.uint8)
    max_values = 16_000_000
    max_cols = max(1, max_values // max(1, n * 2))
    chunks: list[bytes] = []
    for start in range(0, cols_total, max_cols):
        part = matrix[:, start : start + max_cols]
        codes = _bytes_to_nibbles(part).reshape(n, part.shape[1], 2, 1)
        frame = r7.encode_token_codes(codes, levels=16, codec="smevr")
        decoded = r7.decode_token_codes(frame, verify=r7.VERIFY_DIGEST)
        if not np.array_equal(decoded, codes):
            raise RuntimeError("SMEVR record frame failed digest decode")
        chunks.append(frame)
    out = bytearray(b"CGSV1" + struct.pack("<IIH", n, cols_total, len(chunks)))
    for chunk in chunks:
        out += struct.pack("<I", len(chunk))
        out += chunk
    return bytes(out)


def _unsmevr_records(payload: bytes) -> list[bytes]:
    if payload[:5] != b"CGSV1":
        raise ValueError("bad CGSV1 magic")
    n, cols_total, chunk_count = struct.unpack("<IIH", payload[5:15])
    off = 15
    parts = []
    r7 = _r7()
    for _ in range(chunk_count):
        (length,) = struct.unpack("<I", payload[off : off + 4])
        off += 4
        frame = payload[off : off + length]
        off += length
        decoded = r7.decode_token_codes(frame, verify=r7.VERIFY_DIGEST)
        parts.append(_nibbles_to_bytes(decoded.reshape(decoded.shape[0], decoded.shape[1], 2)))
    matrix = np.concatenate(parts, axis=1)[:, :cols_total] if parts else np.zeros((n, 0), dtype=np.uint8)
    records: list[bytes] = []
    for row in matrix:
        (length,) = struct.unpack("<I", row[:4].tobytes())
        records.append(row[4 : 4 + length].tobytes())
    return records


@dataclass(frozen=True)
class CoderResult:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str | None


def _race_coders(records: list[bytes], raw: bytes, *, surface_id: str, store: bool) -> list[CoderResult]:
    encoded: dict[str, bytes] = {
        "brotli-q11": _brotli_q11(raw),
        "lzma1-raw": _lzma1_raw(raw),
        "smevr-r7-nibble": _smevr_records(records),
    }
    if brotli is not None and brotli.decompress(encoded["brotli-q11"]) != raw:
        raise RuntimeError(f"{surface_id}: brotli roundtrip failed")
    if _unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise RuntimeError(f"{surface_id}: LZMA1 roundtrip failed")
    if _unsmevr_records(encoded["smevr-r7-nibble"]) != records:
        raise RuntimeError(f"{surface_id}: SMEVR record roundtrip failed")
    results = []
    best_codec = min(encoded, key=lambda key: len(encoded[key]))
    for codec, payload in encoded.items():
        artifact_path = None
        if store and codec == best_codec:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            safe = surface_id.replace("<->", "_").replace(":", "_").replace("/", "_")
            path = ARTIFACT_DIR / f"{safe}.{codec}.bin"
            path.write_bytes(payload)
            artifact_path = str(path)
        results.append(
            CoderResult(codec=codec, bytes=len(payload), sha256=_sha256_bytes(payload), artifact_path=artifact_path)
        )
    results.sort(key=lambda row: row.bytes)
    return results


def _compress_independent(records: list[bytes]) -> dict[str, int]:
    return {
        "brotli-q11": sum(len(_brotli_q11(record)) for record in records),
        "lzma1-raw": sum(len(_lzma1_raw(record)) for record in records),
        "smevr-r7-nibble": sum(len(_smevr_records([record])) for record in records),
    }


def _choose_even_sample(n: int, k: int) -> list[int]:
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def _temporal_iou_for_class(gt: np.ndarray, cls: int, pairs: Iterable[int]) -> float:
    vals = []
    prev = None
    for p in pairs:
        cur = np.asarray(gt[p] == cls)
        if prev is not None:
            inter = int((prev & cur).sum())
            union = int((prev | cur).sum())
            if union:
                vals.append(inter / union)
        prev = cur
    return float(np.mean(vals)) if vals else 0.0


def _detect_class_roles(gt: np.ndarray) -> dict[str, int]:
    sample = _choose_even_sample(int(gt.shape[0]), min(96, int(gt.shape[0])))
    yy = np.arange(SEG_H, dtype=np.float64)[:, None]
    stats = []
    for cls in range(5):
        area_sum = 0
        y_sum = 0.0
        for p in sample:
            mask = np.asarray(gt[p] == cls)
            area = int(mask.sum())
            area_sum += area
            if area:
                y_sum += float((mask * yy).sum())
        area_frac = area_sum / (len(sample) * PIXELS_PER_PAIR)
        centroid_y = y_sum / max(area_sum, 1)
        iou = _temporal_iou_for_class(gt, cls, sample)
        stats.append({"idx": cls, "area_frac": area_frac, "centroid_y": centroid_y, "temporal_iou": iou})
    lane = min(stats, key=lambda row: row["area_frac"])["idx"]
    mycar = max(stats, key=lambda row: (row["centroid_y"], row["area_frac"]))["idx"]
    undrivable = min((row for row in stats if row["idx"] != mycar), key=lambda row: row["centroid_y"])["idx"]
    small = [row for row in stats if row["idx"] not in {lane, mycar, undrivable}]
    movable = min(small, key=lambda row: row["area_frac"])["idx"]
    road = next(row["idx"] for row in stats if row["idx"] not in {lane, mycar, undrivable, movable})
    roles = {"Road": road, "Lane": lane, "Undrivable": undrivable, "Movable": movable, "MyCar": mycar}
    expected = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if roles != expected:
        raise RuntimeError(f"self-detected class roles {roles} differ from canonical source order {expected}")
    return {
        "roles": roles,
        "stats": stats,
        "method": "spatial/static signature: Lane smallest area; MyCar bottom/static; Undrivable top; Movable remaining small mid-band; Road residual.",
        "matched_canonical_order": True,
    }


def _load_inputs() -> tuple[np.memmap, np.memmap, dict[str, Any]]:
    if not GT_ARGMAX.is_file() or not CX1_ARGMAX.is_file():
        raise SystemExit(f"missing argmax cache: {GT_ARGMAX} / {CX1_ARGMAX}")
    gt = np.load(GT_ARGMAX, mmap_mode="r")
    rd = np.load(CX1_ARGMAX, mmap_mode="r")
    receipt = json.loads(CX1_RECEIPT.read_text())
    if tuple(gt.shape) != (N_PAIRS, SEG_H, SEG_W) or tuple(rd.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise SystemExit(f"unexpected argmax shapes: gt={gt.shape}, cx1={rd.shape}")
    return gt, rd, receipt


def _edge_pairs_from_receipt(receipt: dict[str, Any]) -> list[tuple[str, int, int]]:
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    out: list[tuple[str, int, int]] = []
    for row in receipt["undirected_edges_with_asymmetry"]:
        a_name, b_name = row["edge"].split("<->")
        out.append((row["edge"], class_to_idx[a_name], class_to_idx[b_name]))
    return out


def _surface_row(
    *,
    surface_id: str,
    surface_kind: str,
    target_scope: str,
    records: list[bytes],
    raw: bytes,
    fixable_flips: int,
    described_pixels: int,
    extra: dict[str, Any] | None = None,
    store: bool = True,
) -> dict[str, Any]:
    coder_results = _race_coders(records, raw, surface_id=surface_id, store=store)
    best = coder_results[0]
    gross_s = _seg_s(fixable_flips)
    rate_s = _rate_s(best.bytes)
    row = {
        "surface_id": surface_id,
        "surface_kind": surface_kind,
        "target_scope": target_scope,
        "n_pairs": N_PAIRS,
        "fixable_flips": fixable_flips,
        "described_pixels": described_pixels,
        "gross_described_correction_S_if_perfect": gross_s,
        "best_codec": best.codec,
        "best_bytes": best.bytes,
        "best_rate_S": rate_s,
        "net_delta_S_if_perfect_and_pose_unchanged": -gross_s + rate_s,
        "bytes_per_fixable_flip": best.bytes / fixable_flips if fixable_flips else math.inf,
        "coder_race": [_jsonable(result.__dict__) for result in coder_results],
        "raw_framed_bytes": len(raw),
        "raw_framed_sha256": _sha256_bytes(raw),
        "verdict_scope": "INSTANCE",
        "claim_label": "MEASURED description price; DERIVED perfect-realization S arithmetic",
    }
    if extra:
        row.update(extra)
    return row


def _price_surfaces(gt: np.memmap, rd: np.memmap, receipt: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    edge_pairs = _edge_pairs_from_receipt(receipt)
    total_flips = 0
    confusion = np.zeros((5, 5), dtype=np.int64)
    lane_records: list[bytes] = []
    island_records: dict[str, list[bytes]] = {"Lane": [], "Movable": []}
    island_component_counts = {name: 0 for name in island_records}
    island_pixels = {name: 0 for name in island_records}
    island_fixable = {name: 0 for name in island_records}
    edge_records: dict[str, list[bytes]] = {edge: [] for edge, _a, _b in edge_pairs}
    edge_fixable: dict[str, int] = {edge: 0 for edge, _a, _b in edge_pairs}
    edge_described: dict[str, int] = {edge: 0 for edge, _a, _b in edge_pairs}
    edge_idx = {edge: (a, b) for edge, a, b in edge_pairs}
    lane_idx = 1
    movable_idx = 3
    lane_fixable = 0
    lane_pixels = 0

    for p in range(N_PAIRS):
        g = np.asarray(gt[p], dtype=np.uint8)
        r = np.asarray(rd[p], dtype=np.uint8)
        diff = g != r
        total_flips += int(diff.sum())
        np.add.at(confusion, (g[diff].astype(np.int64), r[diff].astype(np.int64)), 1)

        lane_mask = g == lane_idx
        lane_records.append(_serialize_crop_mask(lane_mask))
        lane_fixable += int((lane_mask & diff).sum())
        lane_pixels += int(lane_mask.sum())

        for name, cls in (("Lane", lane_idx), ("Movable", movable_idx)):
            cls_mask = g == cls
            select_mask = cls_mask & diff
            record, n_comp, n_pix = _serialize_component_masks(cls_mask, select_mask)
            decoded = _decode_component_record(record)
            if not np.array_equal(decoded, cls_mask & decoded):
                raise RuntimeError(f"{name} component decode invariant failed at pair {p}")
            # All selected components should be a subset of the GT class and should
            # cover exactly every serialized component pixel.
            if bool(np.any(decoded & ~cls_mask)):
                raise RuntimeError(f"{name} component decode escaped GT class at pair {p}")
            island_records[name].append(record)
            island_component_counts[name] += n_comp
            island_pixels[name] += n_pix
            island_fixable[name] += int((decoded & diff).sum())

        for edge, (a, b) in edge_idx.items():
            record = _serialize_edge_mask(g, r, a, b)
            mask, target = _decode_edge_record(record)
            truth = diff & (((g == a) & (r == b)) | ((g == b) & (r == a)))
            if not np.array_equal(mask, truth):
                raise RuntimeError(f"{edge} edge decode mismatch at pair {p}")
            if bool(mask.any()) and not np.array_equal(target[mask], g[mask]):
                raise RuntimeError(f"{edge} target-side decode mismatch at pair {p}")
            edge_records[edge].append(record)
            count = int(truth.sum())
            edge_fixable[edge] += count
            edge_described[edge] += count

    if total_flips != EXPECTED_TOTAL_FLIPS:
        raise RuntimeError(f"total flip control failed: {total_flips} != {EXPECTED_TOTAL_FLIPS}")

    lane_raw = _frame_records(lane_records, surface_id="rl1_lane_crop")
    rows.append(
        _surface_row(
            surface_id="rl1_lane_crop",
            surface_kind="class_crop",
            target_scope="Lane",
            records=lane_records,
            raw=lane_raw,
            fixable_flips=lane_fixable,
            described_pixels=lane_pixels,
            extra={
                "rl1_compatible_independent_pair_coder_bytes": _compress_independent(lane_records),
                "rl1_projection_brotli_q11_bytes": 272_869,
                "verdict_scope": "INSTANCE (n600 re-price of rl1's n32 survivor object)",
            },
        )
    )

    for name in ("Lane", "Movable"):
        raw = _frame_records(island_records[name], surface_id=f"island_{name.lower()}_components")
        rows.append(
            _surface_row(
                surface_id=f"island_{name.lower()}_components",
                surface_kind="island_components",
                target_scope=name,
                records=island_records[name],
                raw=raw,
                fixable_flips=island_fixable[name],
                described_pixels=island_pixels[name],
                extra={
                    "component_count": island_component_counts[name],
                    "component_selection": "8-connected GT components with at least one current cx1 flip",
                    "verdict_scope": "INSTANCE",
                },
            )
        )

    for edge, _a, _b in edge_pairs:
        raw = _frame_records(edge_records[edge], surface_id=f"interface_{edge}")
        rows.append(
            _surface_row(
                surface_id=f"interface_{edge}",
                surface_kind="interface_edge",
                target_scope=edge,
                records=edge_records[edge],
                raw=raw,
                fixable_flips=edge_fixable[edge],
                described_pixels=edge_described[edge],
                extra={
                    "edge_rank_from_cx1_receipt": [r["edge"] for r in receipt["undirected_edges_with_asymmetry"]].index(edge)
                    + 1,
                    "verdict_scope": "INSTANCE",
                },
            )
        )

    return {
        "total_flips": total_flips,
        "dseg": total_flips / PIXELS_N600,
        "confusion_matrix_gt_by_rendered": confusion.tolist(),
        "surfaces": rows,
    }


def _load_scorer():
    if str(REPO / "upstream") not in sys.path:
        sys.path.insert(0, str(REPO / "upstream"))
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    net = load_frozen_distortion_net(device="cpu")
    net.eval()
    return net


def _seg_argmax_from_pairs(net: torch.nn.Module, pairs_bthwc: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        x = pairs_bthwc.permute(0, 1, 4, 2, 3).float()
        seg_in = net.segnet.preprocess_input(x)
        out = net.segnet(seg_in)
        return out.argmax(dim=1).cpu().numpy().astype(np.uint8)


def _score_grid_frame(last_frame_hwc: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.asarray(last_frame_hwc).copy()).permute(2, 0, 1).unsqueeze(0).float()
    with torch.inference_mode():
        small = torch.nn.functional.interpolate(t, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)
    return small[0].permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).numpy()


def _prototype_colors(score_frame_hwc: np.ndarray, argmax_hw: np.ndarray) -> np.ndarray:
    flat = score_frame_hwc.reshape(-1, 3).astype(np.float64)
    labels = argmax_hw.reshape(-1)
    protos = np.zeros((5, 3), dtype=np.uint8)
    global_mean = np.rint(flat.mean(axis=0)).clip(0, 255).astype(np.uint8)
    for cls in range(5):
        sel = labels == cls
        if np.any(sel):
            protos[cls] = np.rint(flat[sel].mean(axis=0)).clip(0, 255).astype(np.uint8)
        else:
            protos[cls] = global_mean
    return protos


def _paint_score_pixels_to_camera(frame_hwc: np.ndarray, mask: np.ndarray, target: np.ndarray, protos: np.ndarray) -> np.ndarray:
    out = frame_hwc.copy()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        y0 = int(math.floor(y * CAM_H / SEG_H))
        y1 = int(math.floor((y + 1) * CAM_H / SEG_H))
        x0 = int(math.floor(x * CAM_W / SEG_W))
        x1 = int(math.floor((x + 1) * CAM_W / SEG_W))
        if y1 <= y0:
            y1 = y0 + 1
        if x1 <= x0:
            x1 = x0 + 1
        out[y0:y1, x0:x1, :] = protos[int(target[y, x])]
    return out


def _surface_mask_and_target(surface_id: str, gt: np.ndarray, rd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((SEG_H, SEG_W), dtype=np.uint8)
    if surface_id == "rl1_lane_crop":
        mask = gt == 1
        target[mask] = 1
        return mask, target
    if surface_id == "island_lane_components":
        record, _n, _pix = _serialize_component_masks(gt == 1, (gt == 1) & (gt != rd))
        mask = _decode_component_record(record)
        target[mask] = 1
        return mask, target
    if surface_id == "island_movable_components":
        record, _n, _pix = _serialize_component_masks(gt == 3, (gt == 3) & (gt != rd))
        mask = _decode_component_record(record)
        target[mask] = 3
        return mask, target
    if surface_id.startswith("interface_"):
        edge = surface_id.removeprefix("interface_")
        a_name, b_name = edge.split("<->")
        names = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        record = _serialize_edge_mask(gt, rd, names[a_name], names[b_name])
        return _decode_edge_record(record)
    raise KeyError(surface_id)


def _realization_sample(
    gt: np.memmap,
    rd: np.memmap,
    surfaces: list[dict[str, Any]],
    *,
    n_pairs: int,
    batch: int,
) -> dict[str, Any]:
    if n_pairs <= 0:
        return {"measured": False, "reason": "n_pairs <= 0"}
    pair_ids = _choose_even_sample(N_PAIRS, n_pairs)
    raw_size = RAW_0.stat().st_size
    expected = N_PAIRS * 2 * CAM_H * CAM_W * 3
    if raw_size != expected:
        raise RuntimeError(f"raw size {raw_size} != expected {expected}")
    raw = np.memmap(RAW_0, dtype=np.uint8, mode="r", shape=(N_PAIRS * 2, CAM_H, CAM_W, 3))
    net = _load_scorer()

    base_mismatch = 0
    base_total = 0
    base_argmax_by_pair: dict[int, np.ndarray] = {}
    for start in range(0, len(pair_ids), batch):
        ids = pair_ids[start : start + batch]
        pairs = np.stack([np.stack([raw[2 * p], raw[2 * p + 1]], axis=0) for p in ids], axis=0)
        arg = _seg_argmax_from_pairs(net, torch.from_numpy(pairs.copy()))
        for local, p in enumerate(ids):
            cached = np.asarray(rd[p], dtype=np.uint8)
            base_argmax_by_pair[p] = arg[local]
            base_mismatch += int((arg[local] != cached).sum())
            base_total += PIXELS_PER_PAIR
    base_match_rate = 1.0 - (base_mismatch / base_total)

    rows = []
    for surface in surfaces:
        sid = surface["surface_id"]
        described = 0
        fixed = 0
        collateral = 0
        corrected_total = 0
        for start in range(0, len(pair_ids), batch):
            ids = pair_ids[start : start + batch]
            corrected_pairs = []
            masks_targets = []
            for p in ids:
                g = np.asarray(gt[p], dtype=np.uint8)
                base = base_argmax_by_pair[p]
                mask, target = _surface_mask_and_target(sid, g, base)
                debt = mask & (base != g)
                described += int(debt.sum())
                score_frame = _score_grid_frame(np.asarray(raw[2 * p + 1], dtype=np.uint8))
                protos = _prototype_colors(score_frame, base)
                corrected_last = _paint_score_pixels_to_camera(
                    np.asarray(raw[2 * p + 1], dtype=np.uint8), mask, target, protos
                )
                pair = np.stack([raw[2 * p], corrected_last], axis=0)
                corrected_pairs.append(pair)
                masks_targets.append((g, base, debt))
            arg = _seg_argmax_from_pairs(net, torch.from_numpy(np.stack(corrected_pairs, axis=0).copy()))
            for local, (g, base, debt) in enumerate(masks_targets):
                new = arg[local]
                fixed += int((debt & (new == g)).sum())
                collateral += int(((base == g) & (new != g)).sum())
                corrected_total += int((new != g).sum())
        rows.append(
            {
                "surface_id": sid,
                "sample_pairs": len(pair_ids),
                "sample_pair_ids": pair_ids,
                "described_fixable_flips_on_sample": described,
                "realized_fixed_flips_on_sample": fixed,
                "collateral_new_flips_on_sample": collateral,
                "survival_ratio_fixed_over_described": fixed / described if described else None,
                "net_ratio_fixed_minus_collateral_over_described": (fixed - collateral) / described if described else None,
                "sample_corrected_dseg_after_surface": corrected_total / (len(pair_ids) * PIXELS_PER_PAIR),
                "claim_label": "MEASURED bounded realization sample",
                "verdict_scope": "INSTANCE sample, not n600 authority",
            }
        )
    return {
        "measured": True,
        "axis": "[macOS-CPU advisory bounded n<=120] NON-PROMOTABLE",
        "scorer_forwards_scope": "bounded sample only; no full n600 scorer job",
        "n_pairs": len(pair_ids),
        "sample_mode": "EVENLY_STRIDED_ROUNDED_LINSPACE_NOT_PREFIX",
        "pair_ids": pair_ids,
        "base_argmax_match_rate_vs_cache": base_match_rate,
        "base_argmax_mismatched_pixels": base_mismatch,
        "surface_rows": rows,
        "realization_mechanism": (
            "actual cx1 inflated camera RGB; paint described score-grid pixels as target-class "
            "prototype colors expanded to camera rectangles; CPU SegNet re-run on corrected raw."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--realization-pairs", type=int, default=32)
    parser.add_argument("--realization-batch", type=int, default=4)
    parser.add_argument("--skip-realization", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    gt, rd, receipt = _load_inputs()
    class_detection = _detect_class_roles(gt)
    pricing = _price_surfaces(gt, rd, receipt)
    realization = (
        {"measured": False, "reason": "--skip-realization"}
        if args.skip_realization
        else _realization_sample(
            gt,
            rd,
            pricing["surfaces"],
            n_pairs=args.realization_pairs,
            batch=args.realization_batch,
        )
    )
    result = {
        "schema": "ddm_cg3_counted_gt_recovery.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "own_vehicle_frontier": {
            "S": OWN_VEHICLE_S,
            "archive_bytes": OWN_VEHICLE_BYTES,
            "axis": "[macOS-CPU advisory]",
            "status": "UNMOVED",
        },
        "inputs": {
            "gt_argmax": str(GT_ARGMAX),
            "cx1_argmax": str(CX1_ARGMAX),
            "cx1_receipt": str(CX1_RECEIPT),
            "inflated_raw": str(RAW_0),
        },
        "safety": {
            "gt_video_decode": "none in this run; reused cached SegNet GT argmax whose producer used frame_utils.yuv420_to_rgb",
            "full_n600_scorer_job": "not run",
            "realization_pairs_cap": args.realization_pairs,
            "mps": "not used",
        },
        "class_detection": class_detection,
        "positive_control": {
            "total_flips": pricing["total_flips"],
            "expected_total_flips": EXPECTED_TOTAL_FLIPS,
            "d_seg": pricing["dseg"],
            "expected_d_seg": EXPECTED_DSEG,
            "confusion_matrix_gt_by_rendered": pricing["confusion_matrix_gt_by_rendered"],
            "verdict": "ARGMAX_VERIFIED",
        },
        "pricing": {
            "score_formula": "S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489",
            "rate_s_per_byte": 25.0 / RATE_DENOM,
            "surfaces": pricing["surfaces"],
        },
        "realization": realization,
        "wall_seconds": round(time.time() - t0, 1),
    }
    args.out_json.write_text(json.dumps(_jsonable(result), indent=1) + "\n")
    print(json.dumps(_jsonable({
        "out_json": str(args.out_json),
        "total_flips": pricing["total_flips"],
        "surfaces": [
            {
                "surface_id": row["surface_id"],
                "best_codec": row["best_codec"],
                "best_bytes": row["best_bytes"],
                "gross_S": row["gross_described_correction_S_if_perfect"],
                "net_delta_S_if_perfect": row["net_delta_S_if_perfect_and_pose_unchanged"],
            }
            for row in pricing["surfaces"]
        ],
        "realization_measured": realization.get("measured"),
        "wall_seconds": result["wall_seconds"],
    }), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
