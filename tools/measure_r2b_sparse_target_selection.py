#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure sparse V10 rounding-bin decisions on the real n600 hard oracle.

The receiver starts from the C1 rounded RGB scorer planes.  A charged decision
selects one side of the same rounding bin for one frame-1 scorer cell; the
exact numerator and a bounded uint8 camera block are re-derived without source
pixels.  Source pixels are consulted encode-side only to choose the sign.

Authority: [macOS-CPU advisory], score_claim=false, pointer unmoved.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
import time
import zipfile
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    BlockSolveStatus,
    DisjointResizeOperator,
    solve_bounded_integer_block,
)

SCHEMA = "r2b_sparse_target_selection_receipt.v1"
STREAM_MAGIC = b"R2B1"
PAIR_COUNT = 600
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
CHANNELS = 3
PIXELS = PAIR_COUNT * SCORER_HW[0] * SCORER_HW[1]
SEG_SCORE_PER_FLIP = 100.0 / PIXELS
BYTE_PRICE = 25.0 / 37_545_489
ADMISSION_BYTES = 70_748
BASE_DSEG = 0.00015196
BASE_DPOSE = 0.00010184
GT_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def scorer_binding_sha256(upstream: Path) -> str:
    digest = hashlib.sha256()
    for relative in ("modules.py", "models/posenet.safetensors", "models/segnet.safetensors"):
        path = upstream / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def encode_uleb(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ValueError("ULEB value must be a nonnegative exact integer")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def decode_uleb(payload: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise ValueError("truncated ULEB")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ValueError("overlong ULEB")


def encode_stream(indices: list[int], signs: list[int]) -> bytes:
    if len(indices) != len(signs) or any(s not in (-1, 1) for s in signs):
        raise ValueError("decision indices/signs mismatch")
    if indices != sorted(set(indices)):
        raise ValueError("decision indices must be unique and increasing")
    gaps = bytearray()
    previous = -1
    for index in indices:
        gaps.extend(encode_uleb(index - previous - 1))
        previous = index
    sign_bytes = bytearray((len(signs) + 7) // 8)
    for i, sign in enumerate(signs):
        if sign > 0:
            sign_bytes[i // 8] |= 1 << (i % 8)
    body = struct.pack("<II", len(gaps), len(sign_bytes)) + gaps + sign_bytes
    compressed = brotli.compress(bytes(body), quality=11)
    return STREAM_MAGIC + struct.pack("<II", len(indices), len(compressed)) + compressed


def decode_stream(payload: bytes) -> tuple[list[int], list[int]]:
    if len(payload) < 12 or payload[:4] != STREAM_MAGIC:
        raise ValueError("invalid sparse stream header")
    count, compressed_size = struct.unpack_from("<II", payload, 4)
    if compressed_size != len(payload) - 12:
        raise ValueError("sparse stream length mismatch")
    body = brotli.decompress(payload[12:])
    if len(body) < 8:
        raise ValueError("truncated sparse stream body")
    gap_size, sign_size = struct.unpack_from("<II", body, 0)
    if 8 + gap_size + sign_size != len(body) or sign_size != (count + 7) // 8:
        raise ValueError("sparse stream section lengths mismatch")
    gaps = body[8 : 8 + gap_size]
    sign_payload = body[8 + gap_size :]
    indices: list[int] = []
    offset = 0
    previous = -1
    for _ in range(count):
        gap, offset = decode_uleb(gaps, offset)
        index = previous + gap + 1
        if index <= previous:
            raise ValueError("non-increasing sparse decision index")
        indices.append(index)
        previous = index
    if offset != len(gaps):
        raise ValueError("trailing sparse gap bytes")
    signs = [1 if sign_payload[i // 8] & (1 << (i % 8)) else -1 for i in range(count)]
    if encode_stream(indices, signs) != payload:
        raise ValueError("sparse stream is not canonical")
    return indices, signs


def canonical_archive(stream: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, stream)
    return buffer.getvalue()


def linear_index(pair: int, row: int, col: int) -> int:
    return (pair * SCORER_HW[0] + row) * SCORER_HW[1] + col


def unravel_index(index: int) -> tuple[int, int, int]:
    pair, cell = divmod(index, SCORER_HW[0] * SCORER_HW[1])
    row, col = divmod(cell, SCORER_HW[1])
    if not 0 <= pair < PAIR_COUNT:
        raise ValueError("decision index outside n600 frame-1 surface")
    return pair, row, col


def _rounded_plane_value(frame: np.ndarray, op: DisjointResizeOperator, row: int, col: int) -> np.ndarray:
    rs, cs = op.row_supports[row], op.col_supports[col]
    # C1 canonical realization assigns the same rounded byte to all owned taps.
    return np.asarray(frame[rs.indices[0], cs.indices[0]], dtype=np.uint8)


def signed_rounding_block(
    op: DisjointResizeOperator,
    rounded_rgb: np.ndarray,
    row: int,
    col: int,
    sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive one exact same-rounded-bin block and its three target numerators."""
    if sign not in (-1, 1):
        raise ValueError("rounding sign must be -1 or +1")
    rs, cs = op.row_supports[row], op.col_supports[col]
    coeff = np.outer(rs.numerators, cs.numerators).astype(np.int64).reshape(-1)
    denominator = int(rs.denominator) * int(cs.denominator)
    common_gcd = reduce(gcd, (int(x) for x in coeff))
    half_step = ((denominator - 1) // 2 // common_gcd) * common_gcd
    if half_step <= 0:
        raise ValueError("resize geometry has no signed rounding-bin interior")
    block = np.empty((len(rs.indices), len(cs.indices), CHANNELS), dtype=np.uint8)
    targets = np.empty(CHANNELS, dtype=np.int64)
    for channel, rounded in enumerate(np.asarray(rounded_rgb, dtype=np.int64)):
        target_integer = int(rounded) * denominator + sign * half_step
        if not 0 <= target_integer <= 255 * denominator:
            raise ValueError("signed rounding target escaped uint8 gamut")
        result = solve_bounded_integer_block(
            coeff.tolist(),
            denominator,
            target_integer / denominator,
            target_integer=target_integer,
            preferred=np.full(len(coeff), int(rounded), dtype=np.float64),
            max_nodes=4096,
        )
        if result.status != BlockSolveStatus.FEASIBLE_EXACT or not result.exact_target_rational:
            raise ValueError("signed rounding target is not HARD_ACCEPT exact")
        values = np.asarray(result.values, dtype=np.uint8).reshape(len(rs.indices), len(cs.indices))
        if int(np.dot(coeff, values.reshape(-1).astype(np.int64))) != target_integer:
            raise AssertionError("bounded block numerator verification failed")
        if (target_integer + denominator // 2) // denominator != int(rounded):
            raise AssertionError("signed decision changed the rounded descriptor byte")
        block[:, :, channel] = values
        targets[channel] = target_integer
    return block, targets


def choose_source_closest_sign(
    op: DisjointResizeOperator,
    rounded_rgb: np.ndarray,
    source_frame: np.ndarray,
    row: int,
    col: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    rs, cs = op.row_supports[row], op.col_supports[col]
    source = source_frame[np.ix_(rs.indices, cs.indices, range(CHANNELS))].astype(np.int32)
    options = []
    for sign in (-1, 1):
        try:
            block, targets = signed_rounding_block(op, rounded_rgb, row, col, sign)
        except ValueError:
            continue
        distance = int(np.sum((block.astype(np.int32) - source) ** 2, dtype=np.int64))
        options.append((distance, sign, block, targets))
    if not options:
        raise ValueError("neither signed rounding-bin side is feasible")
    _distance, sign, block, targets = min(options, key=lambda row_: (row_[0], row_[1]))
    return sign, block, targets


def _load_model(upstream: Path):
    sys.path.insert(0, str(upstream))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    model = DistortionNet().eval().to("cpu")
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    return torch, model


def _raw_memmap(path: Path) -> np.memmap:
    expected = PAIR_COUNT * 2 * CAMERA_HW[0] * CAMERA_HW[1] * CHANNELS
    if path.stat().st_size != expected:
        raise ValueError(f"raw byte count {path.stat().st_size} != {expected}")
    return np.memmap(path, mode="r", dtype=np.uint8, shape=(PAIR_COUNT, 2, *CAMERA_HW, CHANNELS))


def score_transition_batches(
    *,
    target_raw: Path,
    candidate_raw: Path,
    cache: np.lib.npyio.NpzFile,
    upstream: Path,
    stage_dir: Path,
    batch_size: int,
    collect_flips: bool,
) -> dict:
    stage_dir.mkdir(parents=True, exist_ok=True)
    target = _raw_memmap(target_raw)
    candidate = _raw_memmap(candidate_raw)
    missing = [
        (start, min(PAIR_COUNT, start + batch_size))
        for start in range(0, PAIR_COUNT, batch_size)
        if not (stage_dir / f"batch-{start:04d}.json").is_file()
    ]
    torch = model = None
    if missing:
        torch, model = _load_model(upstream)
    for start, stop in missing:
        target_batch = torch.from_numpy(np.array(target[start:stop], copy=True))
        candidate_batch = torch.from_numpy(np.array(candidate[start:stop], copy=True))
        with torch.inference_mode():
            target_pose, target_logits = model(target_batch)
            candidate_pose, candidate_logits = model(candidate_batch)
        ref = target_logits.argmax(dim=1).cpu().numpy()
        pred = candidate_logits.argmax(dim=1).cpu().numpy()
        top2 = torch.topk(target_logits, 2, dim=1).values
        margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        target_pose6 = target_pose["pose"][:, :6].cpu().numpy().astype(np.float64)
        candidate_pose6 = candidate_pose["pose"][:, :6].cpu().numpy().astype(np.float64)
        pose_sq = (target_pose6 - candidate_pose6) ** 2
        cache_labels = np.asarray(cache["lstars"][start:stop])
        cache_mismatch = int(np.count_nonzero(ref != cache_labels))
        flips: list[list[float | int]] = []
        if collect_flips:
            for local, row, col in zip(*np.nonzero(ref != pred), strict=True):
                flips.append(
                    [
                        start + int(local),
                        int(row),
                        int(col),
                        int(ref[local, row, col]),
                        int(pred[local, row, col]),
                        float(margins[local, row, col]),
                    ]
                )
        row = {
            "schema": "r2b_hard_oracle_batch.v1",
            "pair_start": start,
            "pair_stop": stop,
            "flips": flips,
            "flip_count": int(np.count_nonzero(ref != pred)),
            "cache_label_mismatches": cache_mismatch,
            "pose_squared_error": pose_sq.tolist(),
        }
        atomic_json(stage_dir / f"batch-{start:04d}.json", row)
        print(f"hard-oracle {start}:{stop} flips={row['flip_count']}", flush=True)
    rows = [json.loads((stage_dir / f"batch-{start:04d}.json").read_text()) for start in range(0, PAIR_COUNT, batch_size)]
    if any(row["pair_start"] != start for row, start in zip(rows, range(0, PAIR_COUNT, batch_size), strict=True)):
        raise ValueError("hard-oracle stage ordering drifted")
    flips = [flip for row in rows for flip in row["flips"]]
    flip_count = sum(int(row["flip_count"]) for row in rows)
    pose_sq = np.concatenate([np.asarray(row["pose_squared_error"], dtype=np.float64) for row in rows])
    return {
        "flips": flips,
        "flip_count": flip_count,
        "d_seg": flip_count / PIXELS,
        "pose_squared_error": pose_sq,
        "d_pose": float(pose_sq.mean()),
        "cache_label_mismatches": sum(int(row["cache_label_mismatches"]) for row in rows),
        "batch_stages": len(rows),
    }


def _edge_flags(labels: np.ndarray, pair: int, row: int, col: int) -> tuple[bool, bool]:
    center = int(labels[pair, row, col])
    neighbors = []
    if row:
        neighbors.append(int(labels[pair, row - 1, col]))
    if row + 1 < SCORER_HW[0]:
        neighbors.append(int(labels[pair, row + 1, col]))
    if col:
        neighbors.append(int(labels[pair, row, col - 1]))
    if col + 1 < SCORER_HW[1]:
        neighbors.append(int(labels[pair, row, col + 1]))
    edge = any(value != center for value in neighbors)
    road_lane = center in (0, 1) and any({center, value} == {0, 1} for value in neighbors)
    return edge, road_lane


def contribution_histogram(flips: list[list[float | int]], pose_sq: np.ndarray, labels: np.ndarray) -> dict:
    class_counts: Counter[int] = Counter()
    margin_counts: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    bands = ((0.0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0), (1.0, float("inf")))
    for pair, row, col, source_class, _pred, margin in flips:
        margin = abs(float(margin))
        class_counts[int(source_class)] += 1
        for lo, hi in bands:
            if lo <= margin < hi:
                margin_counts[f"[{lo:.0e},{hi:.0e})" if math.isfinite(hi) else f"[{lo:.0e},inf)"] += 1
                break
        strata["tie_tight_<1e-3" if margin < 1e-3 else ("interior_>=1" if margin >= 1.0 else "margin_band_[1e-3,1)")] += 1
        edge, road_lane = _edge_flags(labels, int(pair), int(row), int(col))
        edge_counts["road_lane_edge" if road_lane else ("other_edge" if edge else "nonedge")] += 1
    def scored(counter: Counter) -> dict:
        return {str(key): {"flips": value, "seg_score_mass": value * SEG_SCORE_PER_FLIP} for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}
    dim_mse = pose_sq.mean(axis=0)
    pair_mse = pose_sq.mean(axis=1)
    top_pose = np.argsort(pair_mse)[::-1][:20]
    return {
        "seg": {
            "per_flip_score_mass": SEG_SCORE_PER_FLIP,
            "by_source_class": scored(class_counts),
            "by_target_margin": scored(margin_counts),
            "by_tie_interior_stratum": scored(strata),
            "by_edge_stratum": scored(edge_counts),
        },
        "pose": {
            "mse_by_dimension": {str(i): float(value) for i, value in enumerate(dim_mse)},
            "top20_pairs_by_mse": [{"pair": int(i), "mse": float(pair_mse[i])} for i in top_pose],
            "total_pose_term": math.sqrt(10.0 * float(pose_sq.mean())),
        },
    }


def build_curve(indices: list[int], signs: list[int]) -> tuple[list[dict], int]:
    n = len(indices)
    knees = sorted({0, min(64, n), min(256, n), min(1024, n), min(4096, n), n // 2, (3 * n) // 4, n})
    rows = []
    stop_k = 0
    previous_bytes = previous_score = 0.0
    stopped = False
    for k in knees:
        ordered = sorted(zip(indices[:k], signs[:k], strict=True))
        stream_indices = [item[0] for item in ordered]
        stream_signs = [item[1] for item in ordered]
        archive_bytes = 0 if k == 0 else len(canonical_archive(encode_stream(stream_indices, stream_signs)))
        score = k * SEG_SCORE_PER_FLIP
        delta_bytes = archive_bytes - previous_bytes
        delta_score = score - previous_score
        marginal = None if delta_bytes <= 0 else delta_score / delta_bytes
        admitted_segment = k == 0 or (not stopped and marginal is not None and marginal >= BYTE_PRICE)
        if k and not admitted_segment:
            stopped = True
        if admitted_segment:
            stop_k = k
        rows.append({
            "decisions": k,
            "charged_archive_bytes": archive_bytes,
            "scheduled_recovered_seg_score_upper_bound": score,
            "marginal_score_per_byte_from_previous_knee": marginal,
            "kkt_segment_admitted": admitted_segment,
        })
        previous_bytes, previous_score = archive_bytes, score
    return rows, stop_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--target-raw", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("/Users/adpena/Projects/pact/upstream"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-candidate", action="store_true")
    parser.add_argument("--preserve-candidate-raw", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 16:
        raise SystemExit("official CPU batch geometry is fixed at 16")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out_dir).free < 8 * 1024**3:
        raise SystemExit("storage preflight refused: need at least 8 GiB free")
    if sha256_file(args.gt_cache) != GT_CACHE_SHA256:
        raise SystemExit("GT cache SHA-256 drifted")
    baseline_raw_sha = sha256_file(args.baseline_raw)
    target_raw_sha = sha256_file(args.target_raw)
    scorer_sha = scorer_binding_sha256(args.upstream)
    stage_binding = f"{target_raw_sha[:12]}_{baseline_raw_sha[:12]}_{scorer_sha[:12]}"
    cache = np.load(args.gt_cache, mmap_mode="r", allow_pickle=False)
    if int(cache["n_pairs"]) != PAIR_COUNT:
        raise SystemExit("GT cache is not n600")
    started = time.time()
    baseline = score_transition_batches(
        target_raw=args.target_raw,
        candidate_raw=args.baseline_raw,
        cache=cache,
        upstream=args.upstream,
        stage_dir=args.out_dir / f"baseline_stages_{stage_binding}",
        batch_size=args.batch_size,
        collect_flips=True,
    )
    # The sha-pinned cache was produced at batch32; the official CPU evaluator
    # uses batch16.  This scorer is known to carry a three-cell kernel-geometry
    # difference at n600.  Refuse any drift beyond that measured custody class,
    # and keep the live batch16 labels as score authority.
    if baseline["cache_label_mismatches"] > 3:
        raise SystemExit(
            "NO-FAKE target SegNet/cache difference exceeds the measured "
            f"batch-geometry class: {baseline['cache_label_mismatches']} > 3"
        )
    if abs(baseline["d_seg"] - BASE_DSEG) > 5e-9 or abs(baseline["d_pose"] - BASE_DPOSE) > 5e-9:
        raise SystemExit(f"baseline hard-oracle drift: d_seg={baseline['d_seg']} d_pose={baseline['d_pose']}")

    labels = np.asarray(cache["lstars"])
    histogram = contribution_histogram(baseline["flips"], baseline["pose_squared_error"], labels)
    op = DisjointResizeOperator.build(camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1])
    baseline_raw = _raw_memmap(args.baseline_raw)
    target_raw = _raw_memmap(args.target_raw)
    ranked = []
    block_cache: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    for flip in baseline["flips"]:
        pair, row, col, source_class, pred, margin = flip
        edge, road_lane = _edge_flags(labels, int(pair), int(row), int(col))
        idx = linear_index(int(pair), int(row), int(col))
        rounded = _rounded_plane_value(baseline_raw[int(pair), 1], op, int(row), int(col))
        try:
            sign, block, targets = choose_source_closest_sign(op, rounded, target_raw[int(pair), 1], int(row), int(col))
        except ValueError:
            continue
        block_cache[idx] = (sign, block, targets)
        ranked.append((not road_lane, not edge, abs(float(margin)), 0 if int(source_class) == 1 else 1, idx, sign, int(source_class), int(pred)))
    ranked.sort()
    indices = [row[4] for row in ranked]
    signs = [row[5] for row in ranked]
    curve, stop_k = build_curve(indices, signs)
    evaluation_k = stop_k if stop_k or not args.score_candidate else len(indices)
    selected = sorted(zip(indices[:evaluation_k], signs[:evaluation_k], strict=True))
    selected_indices = [item[0] for item in selected]
    selected_signs = [item[1] for item in selected]
    stream = encode_stream(selected_indices, selected_signs)
    if decode_stream(stream) != (selected_indices, selected_signs):
        raise AssertionError("stream parse-back differs")
    stream_path = args.out_dir / "sparse_decisions.r2b"
    stream_path.write_bytes(stream)
    archive_bytes = canonical_archive(stream)
    archive_path = args.out_dir / "archive.zip"
    archive_path.write_bytes(archive_bytes)

    candidate_summary = None
    cleanup = None
    if args.score_candidate and evaluation_k:
        candidate_raw_path = args.out_dir / "candidate" / "0.raw"
        candidate_raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not candidate_raw_path.exists():
            shutil.copyfile(args.baseline_raw, candidate_raw_path)
            candidate_mm = np.memmap(candidate_raw_path, mode="r+", dtype=np.uint8, shape=(PAIR_COUNT, 2, *CAMERA_HW, CHANNELS))
            parsed_indices, parsed_signs = decode_stream(stream)
            for idx, parsed_sign in zip(parsed_indices, parsed_signs, strict=True):
                pair, row, col = unravel_index(idx)
                expected_sign, block, _targets = block_cache[idx]
                if parsed_sign != expected_sign:
                    raise AssertionError("stream sign differs from selected block")
                rs, cs = op.row_supports[row], op.col_supports[col]
                candidate_mm[pair, 1][np.ix_(rs.indices, cs.indices, range(CHANNELS))] = block
            candidate_mm.flush()
            del candidate_mm
        candidate_summary = score_transition_batches(
            target_raw=args.target_raw,
            candidate_raw=candidate_raw_path,
            cache=cache,
            upstream=args.upstream,
            stage_dir=args.out_dir / f"candidate_stages_{stage_binding}_{hashlib.sha256(stream).hexdigest()[:16]}",
            batch_size=args.batch_size,
            collect_flips=False,
        )
        candidate_summary.pop("flips", None)
        candidate_summary.pop("pose_squared_error", None)
        raw_bytes = candidate_raw_path.stat().st_size
        raw_sha = sha256_file(candidate_raw_path)
        cleanup = {
            "schema": "certified_rebuildable_scratch_cleanup.v1",
            "original_path": str(candidate_raw_path),
            "bytes": raw_bytes,
            "sha256": raw_sha,
            "source_baseline_raw": str(args.baseline_raw),
            "source_baseline_raw_sha256": baseline_raw_sha,
            "decision_stream": str(stream_path),
            "decision_stream_sha256": hashlib.sha256(stream).hexdigest(),
            "rebuild_command": " ".join(sys.argv),
            "reason": "candidate raw is deterministic rebuildable scorer scratch; receipt and charged stream are durable",
            "deleted": not args.preserve_candidate_raw,
        }
        atomic_json(args.out_dir / "candidate_cleanup.json", cleanup)
        if not args.preserve_candidate_raw:
            candidate_raw_path.unlink()

    baseline_nonrate = 100.0 * baseline["d_seg"] + math.sqrt(10.0 * baseline["d_pose"])
    candidate_nonrate = None if candidate_summary is None else 100.0 * candidate_summary["d_seg"] + math.sqrt(10.0 * candidate_summary["d_pose"])
    receipt = {
        "schema": SCHEMA,
        "lane_id": "lane_r2b_sparse_target_selection_20260720",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
        "gt_cache": {"path": str(args.gt_cache), "sha256": GT_CACHE_SHA256, "bytes": args.gt_cache.stat().st_size},
        "baseline_raw": {"path": str(args.baseline_raw), "sha256": baseline_raw_sha, "bytes": args.baseline_raw.stat().st_size},
        "target_raw": {"path": str(args.target_raw), "sha256": target_raw_sha, "bytes": args.target_raw.stat().st_size},
        "source_custody": {
            "measurement_tool_sha256": sha256_file(Path(__file__)),
            "scorer_binding_sha256": scorer_sha,
            "stage_binding": stage_binding,
        },
        "hard_oracle_batch_size": args.batch_size,
        "cache_label_geometry_caveat": {
            "live_batch16_vs_sha_pinned_batch32_mismatches": baseline["cache_label_mismatches"],
            "authority": "live batch16 labels for score; cache labels only for edge context",
        },
        "baseline": {key: value for key, value in baseline.items() if key not in ("flips", "pose_squared_error")},
        "baseline_nonrate_score": baseline_nonrate,
        "full_nonrate_gap": baseline_nonrate,
        "contribution_histogram": histogram,
        "feasible_signed_decisions": len(indices),
        "infeasible_signed_decisions": len(baseline["flips"]) - len(indices),
        "ranking": "road-lane edge, other edge, target top1-top2 Fisher-margin, Lane class, stable linear index",
        "curve": curve,
        "byte_price": BYTE_PRICE,
        "kkt_stop_decisions": stop_k,
        "candidate_evaluation_decisions": evaluation_k,
        "candidate_selection_reason": (
            "kkt_admitted_prefix" if stop_k else "bounded_full_knee_falsification_after_kkt_zero"
        ),
        "stream": {"path": str(stream_path), "bytes": len(stream), "sha256": hashlib.sha256(stream).hexdigest()},
        "charged_sparse_archive": {"path": str(archive_path), "bytes": len(archive_bytes), "sha256": hashlib.sha256(archive_bytes).hexdigest()},
        "admission_byte_ceiling": ADMISSION_BYTES,
        "byte_gate_pass": len(archive_bytes) <= ADMISSION_BYTES,
        "candidate": candidate_summary,
        "candidate_nonrate_score": candidate_nonrate,
        "candidate_recovered_score": None if candidate_nonrate is None else baseline_nonrate - candidate_nonrate,
        "candidate_cleanup": cleanup,
        "scheduled_curve_authority": "oracle upper bound from baseline flip contributions; only candidate row is hard-oracle measured",
        "verdict_scope": "one-bit source-sign-chosen, fixed-magnitude same-rounded-bin factor-2 preimage formulation only",
        "elapsed_seconds": time.time() - started,
    }
    receipt["hard_gate_pass"] = bool(
        receipt["byte_gate_pass"]
        and candidate_summary is not None
        and candidate_nonrate is not None
        and candidate_nonrate + BYTE_PRICE * len(archive_bytes) < baseline_nonrate
    )
    atomic_json(args.out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
