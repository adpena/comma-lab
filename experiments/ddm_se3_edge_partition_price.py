#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""SE3 Road/Lane description-side pricing.

This is a scorer-free n600 pricing script for the SE3 charter.  It measures
counted Road/Lane edit streams on the cached GT/current argmax fields, using
real coders only.  The receiver-derived band is the same assumption-scoped
stand-in as sg3: cx1's argmax field supplies the deterministic Road/Lane
coordinate chart here; closure requires re-running against the live generator's
own class field.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage

try:
    import brotli
except ImportError:  # pragma: no cover - pact env has brotli
    brotli = None  # type: ignore[assignment]


REPO = Path(__file__).resolve().parents[1]
ARGMAX_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
GT_ARGMAX = ARGMAX_DIR / "gt_argmax_n600.npy"
CX1_ARGMAX = ARGMAX_DIR / "cx1_argmax_n600.npy"
DEFAULT_OUT = REPO / ".omx/research/ddm_se3_20260804/se3_edge_partition_price.json"
DEFAULT_ARTIFACT_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_se3_20260804/edge_partition_payloads")

SEG_H = 384
SEG_W = 512
N_PAIRS = 600
RATE_DENOM = 37_545_489
ROAD = 0
LANE = 1
ED1_SECTION_BYTES = 169_149
ED1_ARCHIVE_DELTA_BYTES = 169_351
ED1_CAPTURED_TARGETS = 191_005
SG3_ROAD_LANE_BAND_K1_BYTES = 81_365
SG3_ROAD_LANE_BAND_K1_FLIPS = 161_660


@dataclass(frozen=True)
class CoderResult:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str | None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode negative values")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(payload):
            raise ValueError("truncated varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def packbits(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8).reshape(-1), bitorder="big").tobytes()


def unpackbits(payload: bytes, count: int) -> np.ndarray:
    raw = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    if raw.size < count:
        raise ValueError("bit payload is truncated")
    return raw[:count].astype(bool)


def frame_records(records: Iterable[bytes], surface_id: str) -> bytes:
    sid = surface_id.encode("utf-8")
    out = bytearray(b"SE3R1" + varint(len(sid)) + sid)
    records = list(records)
    out += varint(len(records))
    for record in records:
        out += varint(len(record))
        out += record
    return bytes(out)


def load_cg3_module() -> Any:
    path = REPO / "experiments/ddm_cg3_counted_gt_recovery.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ddm_cg3_counted_gt_recovery", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CG3: Any | None = None


def cg3() -> Any:
    global _CG3
    if _CG3 is None:
        _CG3 = load_cg3_module()
    return _CG3


def lzma1_raw(payload: bytes) -> bytes:
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=filters)


def unlzma1_raw(payload: bytes, expected_len: int) -> bytes:
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)
    got = dec.decompress(payload, max_length=expected_len + 1)
    if len(got) != expected_len or not dec.eof or dec.unused_data:
        raise ValueError("LZMA1 roundtrip length/termination mismatch")
    return got


def race_payloads(
    *,
    surface_id: str,
    raw: bytes,
    smevr_records: list[bytes],
    artifact_dir: Path,
    store_best: bool,
) -> list[CoderResult]:
    encoded: dict[str, bytes] = {
        "zlib-9": zlib.compress(raw, 9),
        "lzma1-raw": lzma1_raw(raw),
        "smevr-r7-nibble": cg3()._smevr_records(smevr_records),
    }
    if brotli is not None:
        encoded["brotli-q11"] = bytes(brotli.compress(raw, quality=11))

    if zlib.decompress(encoded["zlib-9"]) != raw:
        raise RuntimeError(f"{surface_id}: zlib roundtrip failed")
    if unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise RuntimeError(f"{surface_id}: lzma roundtrip failed")
    if cg3()._unsmevr_records(encoded["smevr-r7-nibble"]) != smevr_records:
        raise RuntimeError(f"{surface_id}: smevr roundtrip failed")
    if brotli is not None and brotli.decompress(encoded["brotli-q11"]) != raw:
        raise RuntimeError(f"{surface_id}: brotli roundtrip failed")

    best = min(encoded, key=lambda name: len(encoded[name]))
    results: list[CoderResult] = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        artifact_path = None
        if store_best and codec == best:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            safe = surface_id.replace("/", "_").replace(":", "_")
            path = artifact_dir / f"{safe}.{codec}.bin"
            path.write_bytes(payload)
            artifact_path = str(path)
        results.append(CoderResult(codec, len(payload), sha256_bytes(payload), artifact_path))
    return results


def band_for(frame_argmax: np.ndarray, radius: int) -> np.ndarray:
    st3 = ndimage.generate_binary_structure(2, 2)
    road = frame_argmax == ROAD
    lane = frame_argmax == LANE
    return ndimage.binary_dilation(road, st3, radius) & ndimage.binary_dilation(lane, st3, radius)


def sparse_record(indices: np.ndarray, lane_targets: np.ndarray) -> bytes:
    out = bytearray(varint(int(indices.size)))
    prev = 0
    for n, index in enumerate(indices.tolist()):
        delta = int(index) - prev if n else int(index)
        out += varint(delta)
        prev = int(index)
    out += packbits(lane_targets)
    return bytes(out)


def run_record(indices: np.ndarray, lane_targets: np.ndarray) -> bytes:
    out = bytearray()
    if indices.size == 0:
        return b"\x00"
    starts: list[int] = []
    lengths: list[int] = []
    start = int(indices[0])
    prev = start
    length = 1
    for index in indices[1:].tolist():
        index = int(index)
        if index == prev + 1:
            length += 1
        else:
            starts.append(start)
            lengths.append(length)
            start = index
            length = 1
        prev = index
    starts.append(start)
    lengths.append(length)
    out += varint(len(starts))
    prev_start = 0
    for n, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        out += varint(start - prev_start if n else start)
        out += varint(length)
        prev_start = start
    out += packbits(lane_targets)
    return bytes(out)


def decode_sparse_record(record: bytes) -> tuple[np.ndarray, np.ndarray]:
    count, offset = read_varint(record, 0)
    indices = np.empty(count, dtype=np.int64)
    prev = 0
    for n in range(count):
        delta, offset = read_varint(record, offset)
        value = prev + delta if n else delta
        indices[n] = value
        prev = value
    bit_len = (count + 7) // 8
    lane_targets = unpackbits(record[offset : offset + bit_len], count)
    if offset + bit_len != len(record):
        raise ValueError("sparse record trailing bytes")
    return indices, lane_targets


def decode_run_record(record: bytes) -> tuple[np.ndarray, np.ndarray]:
    run_count, offset = read_varint(record, 0)
    indices: list[int] = []
    prev_start = 0
    for n in range(run_count):
        start_delta, offset = read_varint(record, offset)
        length, offset = read_varint(record, offset)
        start = prev_start + start_delta if n else start_delta
        indices.extend(range(start, start + length))
        prev_start = start
    count = len(indices)
    bit_len = (count + 7) // 8
    lane_targets = unpackbits(record[offset : offset + bit_len], count)
    if offset + bit_len != len(record):
        raise ValueError("run record trailing bytes")
    return np.asarray(indices, dtype=np.int64), lane_targets


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, CoderResult):
        return value.__dict__
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def summarize_surface(
    *,
    surface_id: str,
    candidate_kind: str,
    radius: int,
    raw: bytes,
    smevr_records: list[bytes],
    captured_flips: int,
    band_pixels: int,
    total_road_lane_flips: int,
    rate_exchange_bytes_per_flip: float,
    px_n600: int,
    artifact_dir: Path,
    store_best: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coders = race_payloads(
        surface_id=surface_id,
        raw=raw,
        smevr_records=smevr_records,
        artifact_dir=artifact_dir,
        store_best=store_best,
    )
    best = coders[0]
    rate_s = 25.0 * best.bytes / RATE_DENOM
    gross_s = 100.0 * captured_flips / px_n600
    row: dict[str, Any] = {
        "surface_id": surface_id,
        "candidate_kind": candidate_kind,
        "radius": radius,
        "captured_flips": captured_flips,
        "total_road_lane_flips": total_road_lane_flips,
        "capture_fraction": captured_flips / total_road_lane_flips,
        "band_pixels": band_pixels,
        "band_precision": captured_flips / band_pixels,
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "best_codec": best.codec,
        "best_bytes": best.bytes,
        "best_sha256": best.sha256,
        "artifact_path": best.artifact_path,
        "bytes_per_captured_flip": best.bytes / max(captured_flips, 1),
        "break_even_survival_no_pose_no_collateral": best.bytes
        / max(captured_flips * rate_exchange_bytes_per_flip, 1.0),
        "rate_S": rate_s,
        "gross_S_if_100pct_survival": gross_s,
        "net_S_if_100pct_survival_no_pose_no_collateral": rate_s - gross_s,
        "coder_race": [result.__dict__ for result in coders],
        "claim_label": "MEASURED description bytes; DERIVED break-even arithmetic; scorer-free",
        "verdict_scope": "ASSUMPTION(receiver-field) for band coordinate derivation; INSTANCE for this serialized stream",
    }
    if extra:
        row.update(extra)
    return row


def measure(args: argparse.Namespace) -> dict[str, Any]:
    gt = np.load(args.gt_argmax, mmap_mode="r")
    current = np.load(args.current_argmax, mmap_mode="r")
    if tuple(gt.shape) != (N_PAIRS, SEG_H, SEG_W) or tuple(current.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise SystemExit(f"unexpected argmax shapes: gt={gt.shape}, current={current.shape}")

    px_n600 = int(np.prod(gt.shape))
    rate_exchange = 4.0 * RATE_DENOM / px_n600
    target = (gt != current) & (
        ((gt == ROAD) & (current == LANE)) | ((gt == LANE) & (current == ROAD))
    )
    lane_target = gt == LANE
    total = int(target.sum())
    rows: list[dict[str, Any]] = []

    for radius in args.radii:
        edit_records: list[bytes] = []
        explicit_records: list[bytes] = []
        sparse_records: list[bytes] = []
        run_records: list[bytes] = []
        edit_stream = bytearray()
        explicit_stream = bytearray()
        band_pixels = 0
        captured = 0
        for frame in range(N_PAIRS):
            band = band_for(np.asarray(current[frame]), radius)
            frame_target = np.asarray(target[frame])
            edits = np.asarray(frame_target[band], dtype=bool)
            directions = np.asarray(lane_target[frame][band][edits], dtype=bool)
            edit_payload = packbits(edits)
            direction_payload = packbits(directions)
            edit_records.append(edit_payload)
            explicit_records.append(edit_payload + direction_payload)
            edit_stream += edit_payload
            explicit_stream += edit_payload + direction_payload
            positions = np.flatnonzero(edits)
            sparse = sparse_record(positions, directions)
            run = run_record(positions, directions)
            if not np.array_equal(decode_sparse_record(sparse)[0], positions):
                raise RuntimeError("sparse index roundtrip failed")
            if not np.array_equal(decode_sparse_record(sparse)[1], directions):
                raise RuntimeError("sparse direction roundtrip failed")
            if not np.array_equal(decode_run_record(run)[0], positions):
                raise RuntimeError("run index roundtrip failed")
            if not np.array_equal(decode_run_record(run)[1], directions):
                raise RuntimeError("run direction roundtrip failed")
            sparse_records.append(sparse)
            run_records.append(run)
            band_pixels += int(band.sum())
            captured += int(edits.sum())

        rows.append(
            summarize_surface(
                surface_id=f"road_lane_band_r{radius}_edit_bits_side_implied",
                candidate_kind="implicit arclength edit bitstream; target side implied by receiver partition",
                radius=radius,
                raw=bytes(edit_stream),
                smevr_records=edit_records,
                captured_flips=captured,
                band_pixels=band_pixels,
                total_road_lane_flips=total,
                rate_exchange_bytes_per_flip=rate_exchange,
                px_n600=px_n600,
                artifact_dir=args.artifact_dir,
                store_best=args.store_best,
                extra={"direction_bits": "not counted; ASSUMED side is implied by generator-pair partition"},
            )
        )
        rows.append(
            summarize_surface(
                surface_id=f"road_lane_band_r{radius}_edit_plus_direction_bits",
                candidate_kind="implicit arclength edit bitstream plus explicit target-side bits",
                radius=radius,
                raw=bytes(explicit_stream),
                smevr_records=explicit_records,
                captured_flips=captured,
                band_pixels=band_pixels,
                total_road_lane_flips=total,
                rate_exchange_bytes_per_flip=rate_exchange,
                px_n600=px_n600,
                artifact_dir=args.artifact_dir,
                store_best=args.store_best,
                extra={"direction_bits": "counted for every edited band coordinate"},
            )
        )
        rows.append(
            summarize_surface(
                surface_id=f"road_lane_band_r{radius}_sparse_varint_arclength",
                candidate_kind="sparse varint arclength edit indices plus target-side bits",
                radius=radius,
                raw=frame_records(sparse_records, f"road_lane_band_r{radius}_sparse"),
                smevr_records=sparse_records,
                captured_flips=captured,
                band_pixels=band_pixels,
                total_road_lane_flips=total,
                rate_exchange_bytes_per_flip=rate_exchange,
                px_n600=px_n600,
                artifact_dir=args.artifact_dir,
                store_best=args.store_best,
            )
        )
        rows.append(
            summarize_surface(
                surface_id=f"road_lane_band_r{radius}_run_arclength",
                candidate_kind="run-length arclength edit intervals plus target-side bits",
                radius=radius,
                raw=frame_records(run_records, f"road_lane_band_r{radius}_runs"),
                smevr_records=run_records,
                captured_flips=captured,
                band_pixels=band_pixels,
                total_road_lane_flips=total,
                rate_exchange_bytes_per_flip=rate_exchange,
                px_n600=px_n600,
                artifact_dir=args.artifact_dir,
                store_best=args.store_best,
            )
        )

    baseline_rows = {
        "sg3_band_k1": {
            "bytes": SG3_ROAD_LANE_BAND_K1_BYTES,
            "captured_flips": SG3_ROAD_LANE_BAND_K1_FLIPS,
            "break_even_survival_no_pose_no_collateral": SG3_ROAD_LANE_BAND_K1_BYTES
            / (SG3_ROAD_LANE_BAND_K1_FLIPS * rate_exchange),
            "source": ".omx/research/ddm_sg3_cheap_addr.json",
            "scope": "ASSUMPTION(receiver-field), side-implied edit bits",
        },
        "ed1_section": {
            "bytes": ED1_SECTION_BYTES,
            "archive_delta_bytes": ED1_ARCHIVE_DELTA_BYTES,
            "captured_flips": ED1_CAPTURED_TARGETS,
            "break_even_survival_no_pose_no_collateral": ED1_ARCHIVE_DELTA_BYTES
            / (ED1_CAPTURED_TARGETS * rate_exchange),
            "source": ".omx/research/ddm_ed1_per_edge_carrier_20260804.md",
            "scope": "byte-closed receiver-consumed candidate; scorer-unvalidated",
        },
    }

    return {
        "schema": "ddm_se3_edge_partition_price.v1",
        "axis": "[macOS-CPU advisory / cache-derived description bytes]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "selection_mode": "n600 all pairs; no prefix",
        "inputs": {
            "gt_argmax": str(args.gt_argmax),
            "gt_argmax_sha256": sha256_file(args.gt_argmax) if args.hash_inputs else None,
            "current_argmax": str(args.current_argmax),
            "current_argmax_sha256": sha256_file(args.current_argmax) if args.hash_inputs else None,
            "class_order": {"Road": ROAD, "Lane": LANE},
        },
        "meta": {
            "pairs": N_PAIRS,
            "seg_h": SEG_H,
            "seg_w": SEG_W,
            "slots": px_n600,
            "rate_exchange_bytes_per_flip": rate_exchange,
            "total_road_lane_flips": total,
        },
        "coordinate_chart": {
            "description": "receiver-derived Road/Lane band from current argmax stand-in, ordered by deterministic band raster/arclength coordinates",
            "no_absolute_xy_counted": True,
            "closure_requirement": "rerun against the live generator class field; if the receiver cannot derive this chart, use full-position rows instead",
        },
        "candidate_rows": sorted(rows, key=lambda row: row["best_bytes"]),
        "comparison_baselines": baseline_rows,
        "best_candidate_id": min(rows, key=lambda row: row["best_bytes"])["surface_id"],
        "boundaries": [
            "No SegNet/PoseNet scorer forward was run.",
            "No archive.zip was built.",
            "Band coordinates are ASSUMPTION(receiver-field), not a closed receiver proof.",
            "Direction-free rows assume target side is implied by the partition; explicit-direction rows price the safer stream.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-argmax", type=Path, default=GT_ARGMAX)
    parser.add_argument("--current-argmax", type=Path, default=CX1_ARGMAX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--store-best", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    args = parser.parse_args()
    result = measure(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "best_candidate_id": result["best_candidate_id"]}, indent=2))


if __name__ == "__main__":
    main()
