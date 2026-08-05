#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ST1 scorer-free surgical reformulations for CQ2 and PE4.

This arm consumes the addendum-5 targeting corpus without running SegNet,
PoseNet, or upstream/evaluate.py. Leg A trains a mask-domain local-context
student only on Road/Lane band cells from cached frozen-scorer argmax arrays.
Leg B prices generator-parameter transport as one context-conditioned stream
and as one static/transient two-stream control, using the same real coder race
as PE1/PE4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_cq1_comma10k_chart_overlap as cq1
import ddm_pe1_per_edge_partition_race as pe1
import ddm_pe4_runtime_and_transport as pe4

SEG_H: Final = pe1.SEG_H
SEG_W: Final = pe1.SEG_W
N_PAIRS: Final = pe1.N_PAIRS
ROAD: Final = pe1.ROAD
LANE: Final = pe1.LANE
W_BYTES_PER_FLIP: Final = 1.2731082153320312
ROAD_LANE_N600_FLIP_MASS: Final = 226_140
CHARTER_N32_DENOMINATOR: Final = 8_670
CHARTER_BAND_RADIUS: Final = 1
OWN_FRONTIER_LINE: Final = (
    "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; "
    "contest pointer borrowed/unmoved."
)

DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_st1_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_st1_20260805")
DEFAULT_GT_ARGMAX: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy")
DEFAULT_CURRENT_ARGMAX: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy"
)
DEFAULT_TRANSPORT_GT_CACHE: Final = pe4.DEFAULT_GT_CACHE

CARRIER_PRICE_ROWS: Final = {
    "pe3_hybrid_75kb_section": 74_408,
    "pe1_explicit_curve_k16_full": 106_465,
    "pe1_explicit_curve_k8_full": 120_577,
    "pe4_independent_generator": 164_831,
    "bf1_lane_crop": 205_196,
}


class ST1Error(ValueError):
    """ST1 measurement failed a fail-closed invariant."""


@dataclass(frozen=True, slots=True)
class CoderRow:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(jsonable(payload), indent=1, sort_keys=True) + "\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, CoderRow):
        return {
            "codec": value.codec,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "artifact_path": value.artifact_path,
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def storage_preflight(path: Path, required_free_bytes: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    ok = int(usage.free) >= int(required_free_bytes)
    return {
        "path": str(path),
        "required_free_bytes": int(required_free_bytes),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "ok": bool(ok),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def road_lane_target(gt_frame: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
    return ((gt_frame == ROAD) & (current_frame == LANE)) | (
        (gt_frame == LANE) & (current_frame == ROAD)
    )


def charter_band_for(current_frame: np.ndarray) -> np.ndarray:
    return cq1.band_for(np.asarray(current_frame), CHARTER_BAND_RADIUS)


def load_argmax_pair(gt_path: Path, current_path: Path) -> tuple[np.memmap, np.memmap]:
    gt = np.load(gt_path, mmap_mode="r")
    current = np.load(current_path, mmap_mode="r")
    if tuple(gt.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise ST1Error(f"unexpected GT argmax shape {gt.shape}")
    if tuple(current.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise ST1Error(f"unexpected current argmax shape {current.shape}")
    return gt, current


def compute_frequency_maps(gt: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    road_lane_freq = np.zeros((SEG_H, SEG_W), dtype=np.uint16)
    all_flip_freq = np.zeros((SEG_H, SEG_W), dtype=np.uint16)
    for pair in range(N_PAIRS):
        g = np.asarray(gt[pair], dtype=np.uint8)
        c = np.asarray(current[pair], dtype=np.uint8)
        road_lane_freq += road_lane_target(g, c).astype(np.uint16)
        all_flip_freq += (g != c).astype(np.uint16)
    return road_lane_freq, all_flip_freq


def gather_band_points(
    *,
    gt: np.ndarray,
    current: np.ndarray,
    pairs: tuple[int, ...] | list[int],
    frequency_map: np.ndarray,
    max_positive: int,
    negative_ratio: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    neg_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for pair in pairs:
        g = np.asarray(gt[pair], dtype=np.uint8)
        c = np.asarray(current[pair], dtype=np.uint8)
        band = charter_band_for(c)
        target = road_lane_target(g, c) & band
        py, px = np.nonzero(target)
        ny, nx = np.nonzero(band & ~target)
        if py.size:
            pos_chunks.append(
                (
                    np.full(py.size, int(pair), dtype=np.int16),
                    py.astype(np.int16),
                    px.astype(np.int16),
                )
            )
        if ny.size:
            neg_chunks.append(
                (
                    np.full(ny.size, int(pair), dtype=np.int16),
                    ny.astype(np.int16),
                    nx.astype(np.int16),
                )
            )
    if not pos_chunks:
        raise ST1Error("no positive Road/Lane band points found")
    pos_pairs = np.concatenate([row[0] for row in pos_chunks])
    pos_y = np.concatenate([row[1] for row in pos_chunks])
    pos_x = np.concatenate([row[2] for row in pos_chunks])
    neg_pairs = np.concatenate([row[0] for row in neg_chunks])
    neg_y = np.concatenate([row[1] for row in neg_chunks])
    neg_x = np.concatenate([row[2] for row in neg_chunks])

    pos_take = min(int(max_positive), int(pos_pairs.size))
    pos_weights = frequency_map[pos_y.astype(np.int64), pos_x.astype(np.int64)].astype(np.float64) + 1.0
    pos_idx = rng.choice(pos_pairs.size, size=pos_take, replace=False, p=pos_weights / pos_weights.sum())
    neg_take = min(int(pos_take * max(1, negative_ratio)), int(neg_pairs.size))
    neg_weights = frequency_map[neg_y.astype(np.int64), neg_x.astype(np.int64)].astype(np.float64) + 1.0
    neg_idx = rng.choice(neg_pairs.size, size=neg_take, replace=False, p=neg_weights / neg_weights.sum())

    pairs_out = np.concatenate([pos_pairs[pos_idx], neg_pairs[neg_idx]]).astype(np.int16)
    y_out = np.concatenate([pos_y[pos_idx], neg_y[neg_idx]]).astype(np.int16)
    x_out = np.concatenate([pos_x[pos_idx], neg_x[neg_idx]]).astype(np.int16)
    target_out = np.concatenate(
        [np.ones(pos_take, dtype=np.uint8), np.zeros(neg_take, dtype=np.uint8)]
    )
    order = rng.permutation(target_out.size)
    return {
        "pairs": pairs_out[order],
        "y": y_out[order],
        "x": x_out[order],
        "target": target_out[order],
        "positive_population": np.asarray([pos_pairs.size], dtype=np.int64),
        "negative_population": np.asarray([neg_pairs.size], dtype=np.int64),
    }


def gather_eval_band_points(
    *,
    gt: np.ndarray,
    current: np.ndarray,
    pairs: tuple[int, ...],
) -> dict[str, np.ndarray]:
    pair_rows: list[dict[str, Any]] = []
    pair_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    x_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    charter_target_chunks: list[np.ndarray] = []
    charter_target_total = 0
    for pair in pairs:
        g = np.asarray(gt[pair], dtype=np.uint8)
        c = np.asarray(current[pair], dtype=np.uint8)
        band = charter_band_for(c)
        endpoint_band = pe1.edge_band(g, ROAD, LANE)
        target = road_lane_target(g, c) & band
        charter_target = target
        charter_target_total += int(charter_target.sum())
        y, x = np.nonzero(band)
        t = target[y, x].astype(np.uint8)
        ct = charter_target[y, x].astype(np.uint8)
        pair_chunks.append(np.full(y.size, int(pair), dtype=np.int16))
        y_chunks.append(y.astype(np.int16))
        x_chunks.append(x.astype(np.int16))
        target_chunks.append(t)
        charter_target_chunks.append(ct)
        pair_rows.append(
            {
                "pair": int(pair),
                "band_pixels": int(y.size),
                "target_flips": int(t.sum()),
                "charter_cq1_captured_target_flips": int(charter_target.sum()),
                "gt_endpoint_band_target_flips": int(np.count_nonzero(road_lane_target(g, c) & endpoint_band)),
            }
        )
    return {
        "pairs": np.concatenate(pair_chunks),
        "y": np.concatenate(y_chunks),
        "x": np.concatenate(x_chunks),
        "target": np.concatenate(target_chunks),
        "charter_target_inside_gt_band": np.concatenate(charter_target_chunks),
        "charter_target_total": np.asarray([charter_target_total], dtype=np.int64),
        "pair_rows": pair_rows,
    }


def context_hashes(
    *,
    current: np.ndarray,
    pairs: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    frequency_map: np.ndarray,
    bucket_count: int,
    radius: int,
) -> np.ndarray:
    if bucket_count <= 0:
        raise ST1Error("bucket_count must be positive")
    h = np.full(pairs.shape, np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    p = pairs.astype(np.int64)
    yy = y.astype(np.int64)
    xx = x.astype(np.int64)
    for dy in range(-radius, radius + 1):
        cy = np.clip(yy + dy, 0, SEG_H - 1)
        for dx in range(-radius, radius + 1):
            cx = np.clip(xx + dx, 0, SEG_W - 1)
            vals = np.asarray(current[p, cy, cx], dtype=np.uint64)
            salt = np.uint64((dy + radius + 1) * 257 + (dx + radius + 1) * 17)
            h ^= vals + salt
            h *= prime
    y_bin = ((yy * 32) // SEG_H).astype(np.uint64)
    x_bin = ((xx * 32) // SEG_W).astype(np.uint64)
    freq_bin = np.minimum(frequency_map[yy, xx], 255).astype(np.uint64)
    for vals, salt in (
        (y_bin, np.uint64(0x9E3779B185EBCA87)),
        (x_bin, np.uint64(0xC2B2AE3D27D4EB4F)),
        (freq_bin, np.uint64(0x165667B19E3779F9)),
    ):
        h ^= vals + salt
        h *= prime
    return np.asarray(h % np.uint64(bucket_count), dtype=np.int64)


def train_bucket_logits(
    *,
    hashes: np.ndarray,
    target: np.ndarray,
    bucket_count: int,
    positive_weight: float,
    smoothing: float,
    qscale: float,
) -> np.ndarray:
    weights = np.where(target.astype(bool), float(positive_weight), 1.0).astype(np.float64)
    pos = np.bincount(hashes, weights=weights * target.astype(np.float64), minlength=bucket_count)
    total = np.bincount(hashes, weights=weights, minlength=bucket_count)
    prior = float((weights * target.astype(np.float64)).sum() / max(weights.sum(), 1.0))
    prob = (pos + smoothing * prior) / np.maximum(total + smoothing, 1e-12)
    logits = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / np.clip(1.0 - prob, 1e-6, 1.0))
    return np.clip(np.rint(logits * qscale), -32768, 32767).astype("<i2")


def bucket_payload(qlogits: np.ndarray, *, bucket_count: int, radius: int, qscale: float) -> bytes:
    header = {
        "schema": "ddm_st1_bucket_student_payload.v1",
        "bucket_count": int(bucket_count),
        "radius": int(radius),
        "qscale": float(qscale),
        "dtype": "int16-le-logit",
        "body_bytes": int(qlogits.astype("<i2", copy=False).nbytes),
    }
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"ST1BUCK1\n" + len(raw_header).to_bytes(8, "little") + raw_header + qlogits.tobytes(order="C")


def decode_bucket_payload(raw: bytes) -> tuple[dict[str, Any], np.ndarray]:
    if not raw.startswith(b"ST1BUCK1\n"):
        raise ST1Error("bad ST1 bucket payload magic")
    header_len = int.from_bytes(raw[9:17], "little")
    header = json.loads(raw[17 : 17 + header_len].decode("utf-8"))
    body = raw[17 + header_len :]
    qlogits = np.frombuffer(body, dtype="<i2").copy()
    if int(header["bucket_count"]) != int(qlogits.size):
        raise ST1Error("bucket payload length mismatch")
    return header, qlogits


def compress_payload(label: str, raw: bytes, artifact_dir: Path) -> tuple[CoderRow, list[CoderRow]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    encoded = {
        "brotli-q11": brotli.compress(raw, quality=11),
        "lzma1-raw": pe1.lzma1_raw(raw),
        "zlib-9": zlib.compress(raw, level=9),
    }
    if brotli.decompress(encoded["brotli-q11"]) != raw:
        raise ST1Error(f"{label}: Brotli roundtrip failed")
    if pe1.unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise ST1Error(f"{label}: LZMA1 roundtrip failed")
    if zlib.decompress(encoded["zlib-9"]) != raw:
        raise ST1Error(f"{label}: zlib roundtrip failed")
    rows: list[CoderRow] = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        path = artifact_dir / f"{label}.{codec}.bin"
        path.write_bytes(payload)
        rows.append(CoderRow(codec, len(payload), sha256_bytes(payload), str(path)))
    return rows[0], rows


def choose_threshold(prob: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    thresholds = np.unique(np.quantile(prob, np.linspace(0.05, 0.95, 37)))
    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "predicted": 0,
        "hits": 0,
    }
    target_bool = target.astype(bool)
    total_pos = int(target_bool.sum())
    for threshold in thresholds.tolist() + [0.5]:
        pred = prob >= float(threshold)
        hits = int(np.count_nonzero(pred & target_bool))
        predicted = int(np.count_nonzero(pred))
        precision = hits / predicted if predicted else 0.0
        recall = hits / total_pos if total_pos else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if (f1, recall, -predicted) > (best["f1"], best["recall"], -best["predicted"]):
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "predicted": predicted,
                "hits": hits,
            }
    return best


def evaluate_bucket_model(
    *,
    qlogits: np.ndarray,
    hashes: np.ndarray,
    target: np.ndarray,
    charter_target: np.ndarray,
    charter_denominator: int,
    qscale: float,
    threshold: float,
) -> dict[str, Any]:
    prob = sigmoid(qlogits[hashes].astype(np.float32) / np.float32(qscale))
    pred = prob >= float(threshold)
    target_bool = target.astype(bool)
    charter_bool = charter_target.astype(bool)
    hits = int(np.count_nonzero(pred & target_bool))
    charter_hits = int(np.count_nonzero(pred & charter_bool))
    predicted = int(np.count_nonzero(pred))
    target_count = int(np.count_nonzero(target_bool))
    charter_inside_count = int(np.count_nonzero(charter_bool))
    union = int(np.count_nonzero(pred | target_bool))
    precision = hits / predicted if predicted else 0.0
    recall = hits / target_count if target_count else 0.0
    return {
        "target_flips": target_count,
        "predicted_pixels": predicted,
        "hits": hits,
        "charter_hits": charter_hits,
        "charter_denominator": int(charter_denominator),
        "charter_inside_gt_band_denominator": charter_inside_count,
        "charter_overlap_fraction": float(charter_hits / charter_denominator)
        if charter_denominator
        else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "band_iou": float(hits / union) if union else 1.0,
        "probability_min": float(prob.min()) if prob.size else 0.0,
        "probability_max": float(prob.max()) if prob.size else 0.0,
    }


def surgical_student_leg(
    *,
    gt: np.ndarray,
    current: np.ndarray,
    frequency_map: np.ndarray,
    ssd_dir: Path,
    bucket_sizes: tuple[int, ...],
    radius: int,
    seed: int,
    max_positive: int,
    negative_ratio: int,
    positive_weight: float,
    qscale: float,
    smoothing: float,
) -> dict[str, Any]:
    eval_pairs = tuple(int(v) for v in cq1.PAIRS)
    eval_pair_set = frozenset(eval_pairs)
    train_pairs = [pair for pair in range(N_PAIRS) if pair not in eval_pair_set]
    sampled = gather_band_points(
        gt=gt,
        current=current,
        pairs=train_pairs,
        frequency_map=frequency_map,
        max_positive=max_positive,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1009)
    order = rng.permutation(sampled["target"].size)
    holdout_count = max(1, int(round(0.2 * order.size)))
    hold_idx = order[:holdout_count]
    train_idx = order[holdout_count:]

    eval_points = gather_eval_band_points(gt=gt, current=current, pairs=eval_pairs)
    eval_hash_cache: dict[int, np.ndarray] = {}
    hold_hash_cache: dict[int, np.ndarray] = {}
    train_hash_cache: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    artifact_dir = ssd_dir / "surgical_student_payloads"

    for bucket_count in bucket_sizes:
        train_hash = train_hash_cache.setdefault(
            bucket_count,
            context_hashes(
                current=current,
                pairs=sampled["pairs"][train_idx],
                y=sampled["y"][train_idx],
                x=sampled["x"][train_idx],
                frequency_map=frequency_map,
                bucket_count=bucket_count,
                radius=radius,
            ),
        )
        hold_hash = hold_hash_cache.setdefault(
            bucket_count,
            context_hashes(
                current=current,
                pairs=sampled["pairs"][hold_idx],
                y=sampled["y"][hold_idx],
                x=sampled["x"][hold_idx],
                frequency_map=frequency_map,
                bucket_count=bucket_count,
                radius=radius,
            ),
        )
        eval_hash = eval_hash_cache.setdefault(
            bucket_count,
            context_hashes(
                current=current,
                pairs=eval_points["pairs"],
                y=eval_points["y"],
                x=eval_points["x"],
                frequency_map=frequency_map,
                bucket_count=bucket_count,
                radius=radius,
            ),
        )
        qlogits = train_bucket_logits(
            hashes=train_hash,
            target=sampled["target"][train_idx],
            bucket_count=bucket_count,
            positive_weight=positive_weight,
            smoothing=smoothing,
            qscale=qscale,
        )
        raw = bucket_payload(qlogits, bucket_count=bucket_count, radius=radius, qscale=qscale)
        best, coder_rows = compress_payload(f"st1_bucket_{bucket_count}", raw, artifact_dir)
        decoded_header, decoded_qlogits = decode_bucket_payload(
            {
                "brotli-q11": brotli.decompress,
                "zlib-9": zlib.decompress,
            }.get(best.codec, lambda payload: pe1.unlzma1_raw(payload, len(raw)))(
                Path(best.artifact_path).read_bytes()
            )
        )
        if not np.array_equal(decoded_qlogits, qlogits):
            raise ST1Error("decoded bucket payload differs from trained qlogits")
        hold_prob = sigmoid(decoded_qlogits[hold_hash].astype(np.float32) / np.float32(qscale))
        threshold = choose_threshold(hold_prob, sampled["target"][hold_idx])
        eval_metrics = evaluate_bucket_model(
            qlogits=decoded_qlogits,
            hashes=eval_hash,
            target=eval_points["target"],
            charter_target=eval_points["charter_target_inside_gt_band"],
            charter_denominator=int(eval_points["charter_target_total"][0]),
            qscale=qscale,
            threshold=threshold["threshold"],
        )
        economics = []
        for label, carrier_bytes in CARRIER_PRICE_ROWS.items():
            total_bytes = int(best.bytes + carrier_bytes)
            economics.append(
                {
                    "carrier": label,
                    "student_bytes": int(best.bytes),
                    "carrier_bytes": int(carrier_bytes),
                    "total_bytes": total_bytes,
                    "break_even_survival_n600_road_lane": float(
                        total_bytes / (W_BYTES_PER_FLIP * ROAD_LANE_N600_FLIP_MASS)
                    ),
                    "matched_byte_delta_vs_carrier": int(best.bytes - carrier_bytes),
                }
            )
        rows.append(
            {
                "bucket_count": int(bucket_count),
                "params": int(bucket_count),
                "feature": {
                    "kind": "hashed current-argmax local context crop plus y/x bins and G4 flip-frequency bin",
                    "radius": int(radius),
                    "source": (
                        "cx1 cached frozen-scorer argmax only; target is GT-vs-cx1 Road/Lane "
                        "flip set inside CQ1 SE3 r1 Road/Lane band"
                    ),
                },
                "payload": {
                    "best": best,
                    "all_coders": coder_rows,
                    "decoded_header": decoded_header,
                    "raw_bytes": int(len(raw)),
                },
                "threshold_from_train_holdout": threshold,
                "eval_n32": eval_metrics,
                "economics_vs_carriers": economics,
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["eval_n32"]["charter_hits"],
            row["eval_n32"]["band_iou"],
            -row["payload"]["best"].bytes,
        ),
    )
    return {
        "schema": "ddm_st1_surgical_student_leg.v1",
        "axis": "[macOS-CPU advisory / scorer-free mask-domain]",
        "selection_rule": "max CQ1 SE3 r1 n32 flip-set hits, then band IoU, then smaller real-coded payload",
        "train_pairs": len(train_pairs),
        "eval_pairs": list(eval_pairs),
        "sampled_training_points": int(sampled["target"].size),
        "sampled_training_positive": int(sampled["target"].sum()),
        "positive_population_train_pairs": int(sampled["positive_population"][0]),
        "negative_population_train_pairs": int(sampled["negative_population"][0]),
        "eval_pair_rows": eval_points["pair_rows"],
        "eval_target_flips_computed": int(eval_points["target"].sum()),
        "eval_charter_cq1_captured_flips_computed": int(eval_points["charter_target_total"][0]),
        "charter_reference_denominator": CHARTER_N32_DENOMINATOR,
        "charter_band_definition": f"cq1.band_for(cx1_argmax, radius={CHARTER_BAND_RADIUS})",
        "rows": rows,
        "selected": selected,
        "verdict_scope": (
            "FORMULATION-scoped mask-domain local-context hashed student; no RGB receiver "
            "and no scorer claim. A loss here is not a neural-student family kill."
        ),
    }


def generator_fields(params: pe1.GeneratorParams) -> tuple[int, ...]:
    y0, x0, y1, x1 = params.bbox
    return (
        int(y0),
        int(x0),
        int(y1 - y0),
        int(x1 - x0),
        int(params.gen_a_q4[0]),
        int(params.gen_a_q4[1]),
        int(params.gen_b_q4[0]),
        int(params.gen_b_q4[1]),
    )


def context_conditioned_record(
    *,
    params: pe1.GeneratorParams,
    track_id: int,
    previous: tuple[int, ...] | None,
    static_track: bool,
) -> bytes:
    fields = generator_fields(params)
    deltas = fields if previous is None else tuple(value - prev for value, prev in zip(fields, previous, strict=True))
    max_abs = max(abs(v) for v in deltas)
    motion_bucket = 0 if max_abs == 0 else 1 if max_abs <= 2 else 2 if max_abs <= 8 else 3
    context = (int(static_track) << 3) | (int(previous is not None) << 2) | motion_bucket
    record = bytearray([params.edge[0], params.edge[1], context])
    record += pe1.varint(track_id)
    for value in deltas:
        record += pe1.write_zigzag(value)
    return bytes(record)


def frame_records_for_context(
    *,
    components: list[pe1.Component],
    params_by_uid: dict[int, pe1.GeneratorParams],
    tracks: dict[int, int],
    track_static: dict[int, bool],
    selected_stratum: bool | None,
) -> tuple[bytes, ...]:
    by_pair: list[list[pe1.Component]] = [[] for _ in range(N_PAIRS)]
    for comp in components:
        if comp.uid in params_by_uid:
            by_pair[comp.pair].append(comp)
    previous_by_track: dict[int, tuple[int, ...]] = {}
    out: list[bytes] = []
    for comps in by_pair:
        frame = bytearray()
        kept = 0
        for comp in comps:
            params = params_by_uid[comp.uid]
            track_id = tracks[comp.uid]
            is_static = bool(track_static[track_id])
            if selected_stratum is not None and is_static != selected_stratum:
                continue
            previous = previous_by_track.get(track_id)
            rec = context_conditioned_record(
                params=params,
                track_id=track_id,
                previous=previous,
                static_track=is_static,
            )
            previous_by_track[track_id] = generator_fields(params)
            frame += pe1.varint(len(rec)) + rec
            kept += 1
        out.append(pe1.varint(kept) + bytes(frame))
    return tuple(out)


def race_transport_coders(label: str, frame_records: tuple[bytes, ...], artifact_dir: Path) -> tuple[CoderRow, list[CoderRow], int]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw = b"".join(frame_records)
    encoded = {
        "brotli-q11": brotli.compress(raw, quality=11),
        "lzma1-raw": pe1.lzma1_raw(raw),
        "smevr-r7-nibble": pe4.bd1.smevr_records(list(frame_records)),
    }
    if brotli.decompress(encoded["brotli-q11"]) != raw:
        raise ST1Error(f"{label}: Brotli roundtrip failed")
    if pe1.unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise ST1Error(f"{label}: LZMA1 roundtrip failed")
    if tuple(pe4.bd1.unsmevr_records(encoded["smevr-r7-nibble"])) != frame_records:
        raise ST1Error(f"{label}: SMEVR roundtrip failed")
    rows: list[CoderRow] = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        path = artifact_dir / f"{label}.{codec}.bin"
        path.write_bytes(payload)
        rows.append(CoderRow(codec, len(payload), sha256_bytes(payload), str(path)))
    return rows[0], rows, len(raw)


def component_static_fraction(comp: pe1.Component, flip_frequency: np.ndarray) -> float:
    ys = comp.flat.astype(np.int64) // SEG_W
    xs = comp.flat.astype(np.int64) % SEG_W
    if ys.size == 0:
        return 0.0
    return float(np.count_nonzero(flip_frequency[ys, xs] >= 2) / ys.size)


def transport_leg(
    *,
    gt_cache: Path,
    current_argmax: Path,
    all_flip_frequency: np.ndarray,
    ssd_dir: Path,
) -> dict[str, Any]:
    artifact_dir = ssd_dir / "transport_payloads"
    components, params_by_uid = pe4.load_component_state(
        gt_cache=gt_cache,
        current_argmax=current_argmax,
    )
    all_ids = frozenset(comp.uid for comp in components)
    independent = pe1.build_generator_representation_from_params(
        components=components,
        params_by_uid=params_by_uid,
        selected_ids=all_ids,
        surface_id="st1_independent_generator_reference",
    )
    tracks = pe1.track_generator_components(
        components,
        params_by_uid,
        max_distance_px=pe4.TRANSPORT_MAX_DISTANCE_PX,
    )
    track_stats: dict[int, dict[str, Any]] = {}
    for comp in components:
        if comp.uid not in tracks:
            continue
        track_id = tracks[comp.uid]
        frac = component_static_fraction(comp, all_flip_frequency)
        row = track_stats.setdefault(
            track_id,
            {
                "track_id": track_id,
                "components": 0,
                "flip_mass": 0,
                "static_weighted_sum": 0.0,
                "weight_sum": 0.0,
            },
        )
        weight = max(1, int(comp.flat.size))
        row["components"] += 1
        row["flip_mass"] += int(comp.flip_mass)
        row["static_weighted_sum"] += frac * weight
        row["weight_sum"] += weight
    track_static = {
        track_id: (row["static_weighted_sum"] / max(row["weight_sum"], 1.0)) >= 0.5
        for track_id, row in track_stats.items()
    }

    independent_best, independent_rows, independent_raw = race_transport_coders(
        "st1_transport_independent",
        independent.frame_records,
        artifact_dir,
    )
    one_stream_records = frame_records_for_context(
        components=components,
        params_by_uid=params_by_uid,
        tracks=tracks,
        track_static=track_static,
        selected_stratum=None,
    )
    context_best, context_rows, context_raw = race_transport_coders(
        "st1_transport_context_conditioned_single_stream",
        one_stream_records,
        artifact_dir,
    )
    static_records = frame_records_for_context(
        components=components,
        params_by_uid=params_by_uid,
        tracks=tracks,
        track_static=track_static,
        selected_stratum=True,
    )
    transient_records = frame_records_for_context(
        components=components,
        params_by_uid=params_by_uid,
        tracks=tracks,
        track_static=track_static,
        selected_stratum=False,
    )
    static_best, static_rows, static_raw = race_transport_coders(
        "st1_transport_stratum_static",
        static_records,
        artifact_dir,
    )
    transient_best, transient_rows, transient_raw = race_transport_coders(
        "st1_transport_stratum_transient",
        transient_records,
        artifact_dir,
    )
    stratum_total = int(static_best.bytes + transient_best.bytes + 16)
    static_tracks = sum(1 for value in track_static.values() if value)
    return {
        "schema": "ddm_st1_context_transport_leg.v1",
        "axis": "[macOS-CPU advisory / scorer-free real-coder byte pricing]",
        "selection_mode": "n600 all PE1 generator tracks; scorer-free; no prefix",
        "component_count": len(components),
        "track_count": len(track_static),
        "static_tracks": int(static_tracks),
        "transient_tracks": int(len(track_static) - static_tracks),
        "rows": [
            {
                "stream": "independent_164831_reference_recomputed",
                "best": independent_best,
                "all_coders": independent_rows,
                "raw_bytes": independent_raw,
                "delta_vs_independent_bytes": 0,
            },
            {
                "stream": "context_conditioned_single_stream",
                "best": context_best,
                "all_coders": context_rows,
                "raw_bytes": context_raw,
                "delta_vs_independent_bytes": int(context_best.bytes - independent_best.bytes),
            },
            {
                "stream": "g4_static_transient_two_stream",
                "best": {
                    "codec": f"static:{static_best.codec}+transient:{transient_best.codec}+container16",
                    "bytes": stratum_total,
                    "sha256": sha256_bytes(
                        Path(static_best.artifact_path).read_bytes()
                        + Path(transient_best.artifact_path).read_bytes()
                    ),
                    "artifact_path": [static_best.artifact_path, transient_best.artifact_path],
                },
                "static_best": static_best,
                "transient_best": transient_best,
                "static_all_coders": static_rows,
                "transient_all_coders": transient_rows,
                "raw_bytes": int(static_raw + transient_raw),
                "delta_vs_independent_bytes": int(stratum_total - independent_best.bytes),
            },
        ],
        "verdict_scope": (
            "FORMULATION-scoped to PE1 generator-parameter records and G4 same-pixel "
            "flip-frequency strata; scorer-free byte pricing only."
        ),
    }


def recall_queries() -> list[dict[str, Any]]:
    return [
        {
            "query": "SURGICAL-FORM/addendum 5",
            "sources": [
                ".omx/tmp/codex_runs/_common_contract.md",
                ".omx/tmp/codex_runs/st1_prompt.md",
                ".omx/research/operator_directive_per_edge_optimality_criteria_20260805.md",
                ".omx/state/main_hot_state.md",
            ],
            "changed_plan": "CQ2/PE4 negatives are re-run only as surgical, targeted, scorer-free reformulations.",
        },
        {
            "query": "g3/g4/pc2/fl1 Road Lane transport student",
            "sources": [
                ".omx/research/codex_findings_ddm_g3_score_atlas_20260722T204813Z_codex.md",
                ".omx/research/codex_findings_ddm_g4_spatial_stationarity_20260722T212138Z_codex.md",
                ".omx/research/ddm_pc2_perclass_road_edges_20260802.md",
                ".omx/research/ddm_fl1_perclass_flicker_floors_20260731.md",
            ],
            "changed_plan": "Target Road/Lane band cells, sample by flip-frequency, and split transport only by static/transient structure.",
        },
        {
            "query": "PE1 PE3 PE4 BF1 carrier prices",
            "sources": [
                ".omx/research/ddm_pe1_20260805/PE1_RECEIPT_20260805.md",
                ".omx/research/ddm_pe3_20260805/PE3_RECEIPT_20260805.md",
                ".omx/research/ddm_pe4_20260805/PE4_RECEIPT_20260805.md",
                ".omx/research/ddm_bf1_20260805/BF1_RECEIPT_20260805.md",
            ],
            "changed_plan": "Use measured real-coder carrier bytes as the comparison bar; do not compare against entropy projections.",
        },
        {
            "query": "canonical equation list --json",
            "sources": ["tools/list_canonical_equations.py --json; targeted grep during recall"],
            "changed_plan": "No current equation already settles ST1's surgical student or context-stream transport rows.",
        },
    ]


def write_markdown_receipt(path: Path, receipt: dict[str, Any]) -> None:
    leg_a = receipt["leg_a_surgical_student"]
    leg_b = receipt["leg_b_context_transport"]
    selected = leg_a["selected"]
    selected_metrics = selected["eval_n32"]
    selected_best = jsonable(selected["payload"]["best"])
    leg_b_independent_best = jsonable(leg_b["rows"][0]["best"])
    leg_b_context_best = jsonable(leg_b["rows"][1]["best"])
    leg_b_stratum_best = jsonable(leg_b["rows"][2]["best"])
    lines = [
        "# ST1 surgical reformulations receipt - 2026-08-05",
        "",
        "Status: **SCORER-FREE / MASK-DOMAIN / REAL-CODER-PRICED / NO FRONTIER MOVE**.",
        "",
        "Axis: `[macOS-CPU advisory / scorer-free mask-domain real-coder pricing]`.",
        "`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.",
        "",
        "## Answer First",
        "",
        (
            "Leg A selected the `{bucket}` hashed local-context Road/Lane band student: "
            "`{bytes}` counted bytes, `{hits}/{den}` charter n32 flip-set hits, "
            "band IoU `{iou:.6f}`."
        ).format(
            bucket=selected["bucket_count"],
            bytes=selected_best["bytes"],
            hits=selected_metrics["charter_hits"],
            den=selected_metrics["charter_denominator"],
            iou=selected_metrics["band_iou"],
        ),
        (
            "Leg B repriced transport as one context-conditioned stream: `{ctx}` B vs "
            "`{ind}` B independent; stratum split: `{stratum}` B."
        ).format(
            ind=leg_b_independent_best["bytes"],
            ctx=leg_b_context_best["bytes"],
            stratum=leg_b_stratum_best["bytes"],
        ),
        "",
        "## RECALL EVIDENCE",
        "",
    ]
    for row in receipt["recall_evidence"]:
        lines.append(
            f"- Query `{row['query']}`; sources: {', '.join(row['sources'])}. Plan impact: {row['changed_plan']}"
        )
    lines.extend(
        [
            "",
            "## Leg A - Surgical Student",
            "",
            "| buckets | params | best codec | counted B | charter hits | charter denom | target hits | target denom | band IoU |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in leg_a["rows"]:
        metrics = row["eval_n32"]
        best = jsonable(row["payload"]["best"])
        lines.append(
            "| {bucket} | {params} | `{codec}` | {bytes} | {chits} | {cden} | {hits} | {den} | {iou:.6f} |".format(
                bucket=row["bucket_count"],
                params=row["params"],
                codec=best["codec"],
                bytes=best["bytes"],
                chits=metrics["charter_hits"],
                cden=metrics["charter_denominator"],
                hits=metrics["hits"],
                den=metrics["target_flips"],
                iou=metrics["band_iou"],
            )
        )
    lines.extend(
        [
            "",
            f"- Charter band definition: `{leg_a['charter_band_definition']}`.",
            f"- Computed CQ1 SE3 r1 n32 target denominator: `{leg_a['eval_charter_cq1_captured_flips_computed']}`; charter reference denominator: `{leg_a['charter_reference_denominator']}`.",
            f"- Training sample: `{leg_a['sampled_training_points']}` points from `{leg_a['train_pairs']}` non-eval pairs; positives `{leg_a['sampled_training_positive']}`.",
            "- verdict_scope: " + leg_a["verdict_scope"],
            "",
            "Selected-row economics vs carrier bars, using n600 Road/Lane flip mass 226,140:",
            "",
            "| carrier | student B | carrier B | total B | break-even survival |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in selected["economics_vs_carriers"]:
        lines.append(
            f"| `{row['carrier']}` | {row['student_bytes']} | {row['carrier_bytes']} | {row['total_bytes']} | {row['break_even_survival_n600_road_lane']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Leg B - Context Transport",
            "",
            "| stream | best codec | best bytes | raw bytes | delta vs independent |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in leg_b["rows"]:
        best = jsonable(row["best"])
        lines.append(
            f"| `{row['stream']}` | `{best['codec']}` | {best['bytes']} | {row['raw_bytes']} | {row['delta_vs_independent_bytes']} |"
        )
    lines.extend(
        [
            "",
            f"- Tracks: `{leg_b['track_count']}` total; static `{leg_b['static_tracks']}`, transient `{leg_b['transient_tracks']}`.",
            "- verdict_scope: " + leg_b["verdict_scope"],
            "",
            "## Boundaries",
            "",
            "- No SegNet/PoseNet scorer forward was run.",
            "- No `upstream/evaluate.py` run was performed.",
            "- Leg A is a mask-domain hashed-context student, not a legal RGB receiver.",
            "- Leg B is generator-record byte pricing, not runtime scorer survival.",
            "- All persisted evidence is under `.omx/research/ddm_st1_20260805/` or `/Volumes/VertigoDataTier/pact/ddm_st1_20260805/`; no `/tmp` evidence is cited.",
            "",
            "## Follow-On Disposition",
            "",
            "- FIRED: surgical local-context student trained and evaluated on held-out CQ1 n32 pairs.",
            "- FIRED: context-conditioned one-stream and static/transient two-stream transport priced through real coders.",
            "- FOLDED: stream-split PE4 negative is not promoted to a family kill; ST1 records formulation-scoped byte rows only.",
            "- QUEUED-WITH-FIRE-ORDER: if MAIN wants a scorer probe, build a receiver-consumed RGB realization first, then queue after the active scorer batch frees the slot.",
            "",
            "## NEXT-IF-RESUMED",
            "",
            "Start from `ST1_RECEIPT_20260805.md` and `ddm_st1_receipt.json`. Do not run a scorer from this arm. The next executable unit is a receiver-consumed realization of the selected surgical targeter or a lower-overhead transport record format that removes per-record track IDs without splitting coder context.",
            "",
            f"Own-vehicle frontier line: `{OWN_FRONTIER_LINE}`",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--gt-argmax", type=Path, default=DEFAULT_GT_ARGMAX)
    parser.add_argument("--current-argmax", type=Path, default=DEFAULT_CURRENT_ARGMAX)
    parser.add_argument("--transport-gt-cache", type=Path, default=DEFAULT_TRANSPORT_GT_CACHE)
    parser.add_argument("--bucket-sizes", type=int, nargs="+", default=[65_536, 32_768, 16_384, 8_192])
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-positive", type=int, default=30_000)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--smoothing", type=float, default=2.0)
    parser.add_argument("--qscale", type=float, default=512.0)
    parser.add_argument("--skip-leg-a", action="store_true")
    parser.add_argument("--skip-leg-b", action="store_true")
    args = parser.parse_args(argv)

    args.research_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_dir.mkdir(parents=True, exist_ok=True)
    storage = storage_preflight(args.ssd_dir, required_free_bytes=256 * 1024 * 1024)
    if not storage["ok"]:
        raise ST1Error("SSD tier lacks required free space for ST1 artifacts")

    gt, current = load_argmax_pair(args.gt_argmax, args.current_argmax)
    gt_size, gt_sha = sha256_file(args.gt_argmax)
    current_size, current_sha = sha256_file(args.current_argmax)
    road_lane_freq, all_flip_freq = compute_frequency_maps(gt, current)

    if args.skip_leg_a:
        raise ST1Error("Leg A is required by the charter; do not skip it for a final ST1 run")
    if args.skip_leg_b:
        raise ST1Error("Leg B is required by the charter; do not skip it for a final ST1 run")

    started = time.time()
    leg_a = surgical_student_leg(
        gt=gt,
        current=current,
        frequency_map=road_lane_freq,
        ssd_dir=args.ssd_dir,
        bucket_sizes=tuple(args.bucket_sizes),
        radius=args.radius,
        seed=args.seed,
        max_positive=args.max_positive,
        negative_ratio=args.negative_ratio,
        positive_weight=args.positive_weight,
        qscale=args.qscale,
        smoothing=args.smoothing,
    )
    leg_b = transport_leg(
        gt_cache=args.transport_gt_cache,
        current_argmax=args.current_argmax,
        all_flip_frequency=all_flip_freq,
        ssd_dir=args.ssd_dir,
    )
    receipt = {
        "schema": "ddm_st1_surgical_reformulations_receipt.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory / scorer-free mask-domain real-coder pricing]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "script": str(Path(__file__).relative_to(REPO)),
        "script_sha256": sha256_file(Path(__file__))[1],
        "storage_preflight": storage,
        "custody": {
            "gt_argmax": {"path": str(args.gt_argmax), "bytes": gt_size, "sha256": gt_sha},
            "current_argmax": {
                "path": str(args.current_argmax),
                "bytes": current_size,
                "sha256": current_sha,
            },
            "road_lane_frequency_nonzero_pixels": int(np.count_nonzero(road_lane_freq)),
            "all_flip_frequency_nonzero_pixels": int(np.count_nonzero(all_flip_freq)),
        },
        "parameters": {
            "bucket_sizes": list(args.bucket_sizes),
            "radius": args.radius,
            "seed": args.seed,
            "max_positive": args.max_positive,
            "negative_ratio": args.negative_ratio,
            "positive_weight": args.positive_weight,
            "smoothing": args.smoothing,
            "qscale": args.qscale,
        },
        "recall_evidence": recall_queries(),
        "leg_a_surgical_student": leg_a,
        "leg_b_context_transport": leg_b,
        "boundaries": [
            "No SegNet/PoseNet scorer forward was run.",
            "No upstream/evaluate.py run was performed.",
            "No archive score claim is made.",
            "Leg A is mask-domain targeting evidence, not a legal RGB receiver.",
            "Leg B is real-coder byte pricing over generator records, not scorer survival.",
            "No /tmp evidence.",
        ],
        "wall_seconds": round(time.time() - started, 3),
        "own_vehicle_frontier_line": OWN_FRONTIER_LINE,
    }
    json_path = args.research_dir / "ddm_st1_receipt.json"
    md_path = args.research_dir / "ST1_RECEIPT_20260805.md"
    next_path = args.research_dir / "NEXT_IF_RESUMED.md"
    atomic_write_json(json_path, receipt)
    write_markdown_receipt(md_path, receipt)
    atomic_write_text(
        next_path,
        "\n".join(
            [
                "# ST1 NEXT-IF-RESUMED",
                "",
                "Start from `ST1_RECEIPT_20260805.md` and `ddm_st1_receipt.json`.",
                "Do not run a scorer from this arm; pe2 owns/owned the active batch in the charter context.",
                "The next executable unit is receiver consumption for whichever targeter/transport row MAIN chooses.",
                "",
                f"Own-vehicle frontier line: `{OWN_FRONTIER_LINE}`",
                "",
            ]
        ),
    )
    print(
        json.dumps(
            {
                "receipt": str(json_path),
                "markdown": str(md_path),
                "leg_a_selected_bucket": receipt["leg_a_surgical_student"]["selected"]["bucket_count"],
                "leg_a_charter_hits": receipt["leg_a_surgical_student"]["selected"]["eval_n32"][
                    "charter_hits"
                ],
                "leg_a_charter_denominator": receipt["leg_a_surgical_student"]["selected"]["eval_n32"][
                    "charter_denominator"
                ],
                "leg_b_context_bytes": jsonable(receipt["leg_b_context_transport"]["rows"][1]["best"])[
                    "bytes"
                ],
                "leg_b_independent_bytes": jsonable(receipt["leg_b_context_transport"]["rows"][0]["best"])[
                    "bytes"
                ],
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
