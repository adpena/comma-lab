#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ST2 scorer-native surgical student for Road/Lane target selection.

This arm consumes cached frozen-scorer products and prior scorer-native
equations without launching SegNet, PoseNet, upstream/evaluate.py, paint, or an
n600 scorer job. It prices only the compact bucket table emitted by the
targeter; scorer-derived fields remain compress-time/training evidence and are
not receiver payload.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import ddm_st1_surgical_reformulations as st1
from tac.canonical_equations.lane_gain_chain_composed_20260716 import (
    COMPOSED_GAIN_MED,
    SKIP_GAIN_RATIO_MED,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    HEAD_PAIR_NORMS,
    SEGNET_WEIGHTS_SHA256,
)

SEG_H: Final = st1.SEG_H
SEG_W: Final = st1.SEG_W
N_PAIRS: Final = st1.N_PAIRS
ROAD: Final = st1.ROAD
LANE: Final = st1.LANE
CHARTER_N32_DENOMINATOR: Final = st1.CHARTER_N32_DENOMINATOR
OWN_FRONTIER_LINE: Final = st1.OWN_FRONTIER_LINE

ST1_BASELINE_BYTES: Final = 9_718
ST1_BASELINE_HITS: Final = 8_336
ST1_BASELINE_DENOMINATOR: Final = 8_670
ST1_BASELINE_BAND_IOU: Final = 0.135509

DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_st2_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_st2_20260805")
DEFAULT_GT_ARGMAX: Final = st1.DEFAULT_GT_ARGMAX
DEFAULT_CURRENT_ARGMAX: Final = st1.DEFAULT_CURRENT_ARGMAX
DEFAULT_GT_MARGIN_F16: Final = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/"
    "targets_n600/gt_segnet_margin.f16"
)
DEFAULT_GT_MARGIN_META: Final = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/"
    "targets_n600/targets_meta.json"
)
DEFAULT_HOPE_CAPACITY_TABLE: Final = (
    REPO / ".omx/research/ddm_hb1_hope_bn_capacity_20260727T0001Z/hope_per_stratum_capacity_table.json"
)

ROAD_LANE_HEAD_NORM: Final = float(HEAD_PAIR_NORMS["Road-Lane"])
ROAD_LANE_COMPOSED_GAIN: Final = float(COMPOSED_GAIN_MED["Road-Lane"])
ROAD_LANE_SKIP_GAIN_RATIO: Final = float(SKIP_GAIN_RATIO_MED["Road-Lane"])
ST2_MAGIC: Final = b"ST2SNAT1\n"


class ST2Error(ValueError):
    """ST2 measurement failed a fail-closed invariant."""


@dataclass(frozen=True, slots=True)
class FeatureBins:
    margin: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10, 0.137, 0.20, 0.30, 0.50, 0.80, 1.25, 2.0, 4.0)
    fisher: tuple[float, ...] = (0.01, 0.03, 0.06, 0.10, 0.16, 0.24, 0.32, 0.40, 0.46, 0.49)
    flipdist: tuple[float, ...] = (0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.14, 0.20, 0.32, 0.50, 0.80)
    margin_grad: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20)
    gain_weighted_fisher: tuple[float, ...] = (1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 13.0, 16.0, 20.0, 23.0)
    frequency: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

    def jsonable(self) -> dict[str, list[float] | list[int]]:
        return {
            "margin": list(self.margin),
            "fisher": list(self.fisher),
            "flipdist": list(self.flipdist),
            "margin_grad": list(self.margin_grad),
            "gain_weighted_fisher": list(self.gain_weighted_fisher),
            "frequency": list(self.frequency),
        }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(st1.jsonable(payload), indent=1, sort_keys=True) + "\n")


def load_margin_memmap(path: Path) -> np.memmap:
    if not path.exists():
        raise ST2Error(f"missing cached GT margin field: {path}")
    expected_bytes = N_PAIRS * SEG_H * SEG_W * np.dtype("<f2").itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ST2Error(f"unexpected margin field size {actual_bytes}; expected {expected_bytes}")
    return np.memmap(path, dtype="<f2", mode="r", shape=(N_PAIRS, SEG_H, SEG_W))


def fisher_trace_from_margin(margin: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(margin, dtype=np.float32), 0.0, 20.0)
    return np.float32(0.5) / np.square(np.cosh(clipped * np.float32(0.5)))


def digitize(values: np.ndarray, bins: tuple[float, ...] | tuple[int, ...]) -> np.ndarray:
    return np.searchsorted(np.asarray(bins, dtype=np.float32), np.asarray(values, dtype=np.float32), side="right")


def margin_spatial_gradient(
    margins: np.ndarray,
    *,
    pairs: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    p = pairs.astype(np.int64)
    yy = y.astype(np.int64)
    xx = x.astype(np.int64)
    yp = np.minimum(yy + 1, SEG_H - 1)
    ym = np.maximum(yy - 1, 0)
    xp = np.minimum(xx + 1, SEG_W - 1)
    xm = np.maximum(xx - 1, 0)
    dx = np.abs(np.asarray(margins[p, yy, xp], dtype=np.float32) - np.asarray(margins[p, yy, xm], dtype=np.float32))
    dy = np.abs(np.asarray(margins[p, yp, xx], dtype=np.float32) - np.asarray(margins[p, ym, xx], dtype=np.float32))
    return np.float32(0.5) * (dx + dy)


def current_neighbor_disagreement(
    current: np.ndarray,
    *,
    pairs: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    p = pairs.astype(np.int64)
    yy = y.astype(np.int64)
    xx = x.astype(np.int64)
    center = np.asarray(current[p, yy, xx], dtype=np.uint8)
    offsets = (
        (np.maximum(yy - 1, 0), xx),
        (np.minimum(yy + 1, SEG_H - 1), xx),
        (yy, np.maximum(xx - 1, 0)),
        (yy, np.minimum(xx + 1, SEG_W - 1)),
    )
    counts = np.zeros(center.shape, dtype=np.uint8)
    for cy, cx in offsets:
        counts += np.asarray(current[p, cy, cx], dtype=np.uint8) != center
    return counts


def load_road_lane_capacity_prior(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ST2Error(f"missing HOPE Road/Lane capacity table: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    for row in obj.get("strata", []):
        if row.get("class_pair_names") != "Road--Lane" and row.get("class_pair") != "0-1":
            continue
        bucket_id = str(row.get("bucket_id", ""))
        boundary = "boundary" if "__boundary__" in bucket_id else "cell"
        temporal = "static" if "static_in_image" in bucket_id else "transient"
        shares = [float(value) for value in row["capacity_share"]]
        top_channels = sorted(range(len(shares)), key=lambda idx: shares[idx], reverse=True)[:5]
        rows[f"{boundary}_{temporal}"] = {
            "bucket_id": bucket_id,
            "support_pixel_count": int(row.get("support_pixel_count", 0)),
            "top_channels": [int(idx) for idx in top_channels],
            "top_capacity_shares": [shares[idx] for idx in top_channels],
            "total_capacity": float(sum(float(v) for v in row["capacity_per_channel"])),
        }
    required = {"boundary_static", "boundary_transient", "cell_static", "cell_transient"}
    missing = sorted(required - set(rows))
    if missing:
        raise ST2Error(f"HOPE Road/Lane capacity table missing strata: {missing}")
    return {
        "source_path": str(path),
        "source_sha256": st1.sha256_file(path)[1],
        "schema": obj.get("schema"),
        "evidence_axis": obj.get("evidence_axis"),
        "score_claim": bool(obj.get("score_claim", False)),
        "top_by_road_lane_stratum": rows,
    }


def capacity_channel_codes(
    *,
    margin: np.ndarray,
    road_lane_frequency: np.ndarray,
    neighbor_disagreement: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    prior: dict[str, Any],
) -> np.ndarray:
    yy = y.astype(np.int64)
    xx = x.astype(np.int64)
    static = road_lane_frequency[yy, xx] >= 2
    boundary = (margin <= np.float32(0.5)) | (neighbor_disagreement > 0)
    rows = prior["top_by_road_lane_stratum"]
    out = np.empty(margin.shape, dtype=np.uint8)
    for key, mask in (
        ("boundary_static", boundary & static),
        ("boundary_transient", boundary & ~static),
        ("cell_static", ~boundary & static),
        ("cell_transient", ~boundary & ~static),
    ):
        top_channel = rows[key]["top_channels"][0]
        out[mask] = np.uint8(top_channel)
    return out


def _mix_hash(h: np.ndarray, values: np.ndarray, salt: int) -> np.ndarray:
    prime = np.uint64(1099511628211)
    h ^= values.astype(np.uint64) + np.uint64(salt)
    h *= prime
    return h


def scorer_native_feature_codes(
    *,
    current: np.ndarray,
    margins: np.ndarray,
    pairs: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    road_lane_frequency: np.ndarray,
    all_flip_frequency: np.ndarray,
    capacity_prior: dict[str, Any],
    bucket_count: int,
    bins: FeatureBins = FeatureBins(),
) -> np.ndarray:
    if bucket_count <= 0:
        raise ST2Error("bucket_count must be positive")
    p = pairs.astype(np.int64)
    yy = y.astype(np.int64)
    xx = x.astype(np.int64)
    margin = np.asarray(margins[p, yy, xx], dtype=np.float32)
    margin = np.nan_to_num(margin, nan=20.0, posinf=20.0, neginf=0.0)
    fisher = fisher_trace_from_margin(margin)
    flipdist = margin / np.float32(ROAD_LANE_HEAD_NORM)
    spatial_grad = margin_spatial_gradient(margins, pairs=pairs, y=y, x=x)
    gain_weighted_fisher = fisher / np.float32(max(ROAD_LANE_COMPOSED_GAIN, 1e-8))
    neighbor = current_neighbor_disagreement(current, pairs=pairs, y=y, x=x)
    current_center = np.asarray(current[p, yy, xx], dtype=np.uint8)
    current_group = np.where(current_center == ROAD, 0, np.where(current_center == LANE, 1, 2)).astype(np.uint8)
    capacity_channel = capacity_channel_codes(
        margin=margin,
        road_lane_frequency=road_lane_frequency,
        neighbor_disagreement=neighbor,
        y=y,
        x=x,
        prior=capacity_prior,
    )

    y_bin = ((yy * 32) // SEG_H).astype(np.uint8)
    x_bin = ((xx * 32) // SEG_W).astype(np.uint8)
    rl_freq_bin = digitize(road_lane_frequency[yy, xx], bins.frequency).astype(np.uint8)
    all_freq_bin = digitize(all_flip_frequency[yy, xx], bins.frequency).astype(np.uint8)
    feature_columns = (
        digitize(margin, bins.margin),
        digitize(fisher, bins.fisher),
        digitize(flipdist, bins.flipdist),
        digitize(spatial_grad, bins.margin_grad),
        digitize(gain_weighted_fisher, bins.gain_weighted_fisher),
        rl_freq_bin,
        all_freq_bin,
        y_bin,
        x_bin,
        neighbor,
        current_group,
        capacity_channel,
    )
    h = np.full(pairs.shape, np.uint64(1469598103934665603), dtype=np.uint64)
    for salt, column in enumerate(feature_columns, start=1):
        h = _mix_hash(h, np.asarray(column), 0x9E3779B185EBCA87 + 0x100000001B3 * salt)
    return np.asarray(h % np.uint64(bucket_count), dtype=np.int64)


def scorer_native_sample_weights(
    *,
    margins: np.ndarray,
    pairs: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    road_lane_frequency: np.ndarray,
    bins: FeatureBins = FeatureBins(),
) -> np.ndarray:
    p = pairs.astype(np.int64)
    yy = y.astype(np.int64)
    xx = x.astype(np.int64)
    margin = np.asarray(margins[p, yy, xx], dtype=np.float32)
    margin = np.nan_to_num(margin, nan=20.0, posinf=20.0, neginf=0.0)
    fisher = fisher_trace_from_margin(margin)
    freq = road_lane_frequency[yy, xx].astype(np.float32)
    low_margin = margin <= np.float32(0.137)
    margin_weight = 1.0 + 2.0 * (fisher / np.float32(0.5))
    frequency_weight = 1.0 + np.minimum(freq, 32.0) / np.float32(32.0)
    label_ambiguity_weight = np.where(low_margin, 1.35, 1.0).astype(np.float32)
    return (margin_weight * frequency_weight * label_ambiguity_weight).astype(np.float64)


def train_bucket_logits_weighted(
    *,
    hashes: np.ndarray,
    target: np.ndarray,
    native_weight: np.ndarray,
    bucket_count: int,
    positive_weight: float,
    smoothing: float,
    qscale: float,
) -> np.ndarray:
    if hashes.shape != target.shape or hashes.shape != native_weight.shape:
        raise ST2Error("hashes, target, and native_weight must have matching shapes")
    base = np.asarray(native_weight, dtype=np.float64)
    if not np.isfinite(base).all() or (base <= 0).any():
        raise ST2Error("native sample weights must be positive finite values")
    weights = base * np.where(target.astype(bool), float(positive_weight), 1.0)
    pos = np.bincount(hashes, weights=weights * target.astype(np.float64), minlength=bucket_count)
    total = np.bincount(hashes, weights=weights, minlength=bucket_count)
    prior = float((weights * target.astype(np.float64)).sum() / max(weights.sum(), 1.0))
    prob = (pos + smoothing * prior) / np.maximum(total + smoothing, 1e-12)
    logits = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / np.clip(1.0 - prob, 1e-6, 1.0))
    return np.clip(np.rint(logits * qscale), -32768, 32767).astype("<i2")


def combined_context_scorer_hashes(
    *,
    current: np.ndarray,
    margins: np.ndarray,
    pairs: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    road_lane_frequency: np.ndarray,
    all_flip_frequency: np.ndarray,
    capacity_prior: dict[str, Any],
    bucket_count: int,
    context_radius: int,
    bins: FeatureBins = FeatureBins(),
) -> np.ndarray:
    large_mod = 2_147_483_647
    context_hash = st1.context_hashes(
        current=current,
        pairs=pairs,
        y=y,
        x=x,
        frequency_map=road_lane_frequency,
        bucket_count=large_mod,
        radius=context_radius,
    )
    scorer_hash = scorer_native_feature_codes(
        current=current,
        margins=margins,
        pairs=pairs,
        y=y,
        x=x,
        road_lane_frequency=road_lane_frequency,
        all_flip_frequency=all_flip_frequency,
        capacity_prior=capacity_prior,
        bucket_count=large_mod,
        bins=bins,
    )
    h = np.full(pairs.shape, np.uint64(1469598103934665603), dtype=np.uint64)
    h = _mix_hash(h, context_hash, 0xC2B2AE3D27D4EB4F)
    h = _mix_hash(h, scorer_hash, 0x165667B19E3779F9)
    return np.asarray(h % np.uint64(bucket_count), dtype=np.int64)


def model_hashes(
    *,
    feature_mode: str,
    current: np.ndarray,
    margins: np.ndarray,
    pairs: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    road_lane_frequency: np.ndarray,
    all_flip_frequency: np.ndarray,
    capacity_prior: dict[str, Any],
    bucket_count: int,
    context_radius: int,
    bins: FeatureBins = FeatureBins(),
) -> np.ndarray:
    if feature_mode == "scorer_native":
        return scorer_native_feature_codes(
            current=current,
            margins=margins,
            pairs=pairs,
            y=y,
            x=x,
            road_lane_frequency=road_lane_frequency,
            all_flip_frequency=all_flip_frequency,
            capacity_prior=capacity_prior,
            bucket_count=bucket_count,
            bins=bins,
        )
    if feature_mode == "scorer_weighted_context":
        return st1.context_hashes(
            current=current,
            pairs=pairs,
            y=y,
            x=x,
            frequency_map=road_lane_frequency,
            bucket_count=bucket_count,
            radius=context_radius,
        )
    if feature_mode == "scorer_native_plus_context":
        return combined_context_scorer_hashes(
            current=current,
            margins=margins,
            pairs=pairs,
            y=y,
            x=x,
            road_lane_frequency=road_lane_frequency,
            all_flip_frequency=all_flip_frequency,
            capacity_prior=capacity_prior,
            bucket_count=bucket_count,
            context_radius=context_radius,
            bins=bins,
        )
    raise ST2Error(f"unknown feature mode {feature_mode!r}")


def feature_description(feature_mode: str, context_radius: int) -> dict[str, str | int]:
    if feature_mode == "scorer_native":
        kind = "scorer-native margin/Fisher/head-distance/gain/stationarity/HOPE bucket code"
        runtime = "scorer-native feature code"
    elif feature_mode == "scorer_weighted_context":
        kind = "ST1 current-argmax context hash with scorer-native sample weighting"
        runtime = f"current-argmax local context radius {context_radius}; scorer-native signal is training-only"
    elif feature_mode == "scorer_native_plus_context":
        kind = "combined current-argmax context and scorer-native feature hash"
        runtime = f"current-argmax local context radius {context_radius} mixed with scorer-native feature code"
    else:
        raise ST2Error(f"unknown feature mode {feature_mode!r}")
    return {
        "mode": feature_mode,
        "kind": kind,
        "runtime_feature_surface": runtime,
        "context_radius": int(context_radius),
        "source": (
            "cached frozen-scorer GT margin field, current argmax CQ1 band state, "
            "G4/ST1 recurrence maps, rank-4 head law, #141 saliency prior, "
            "lg1 Road/Lane gain law, and #725 HOPE Road/Lane BN capacity table"
        ),
        "target_leakage_guard": (
            "feature code does not include GT-vs-current Road/Lane mismatch or GT center class; "
            "the mismatch is used only as the supervised target."
        ),
    }


def scorer_native_payload(
    qlogits: np.ndarray,
    *,
    bucket_count: int,
    qscale: float,
    bins: FeatureBins,
    capacity_prior: dict[str, Any],
    feature_mode: str = "scorer_native",
    context_radius: int = 0,
    scorer_weighting: dict[str, Any] | None = None,
) -> bytes:
    header = {
        "schema": "ddm_st2_scorer_native_bucket_student_payload.v1",
        "bucket_count": int(bucket_count),
        "qscale": float(qscale),
        "dtype": "int16-le-logit",
        "body_bytes": int(qlogits.astype("<i2", copy=False).nbytes),
        "feature_schema": {
            "signals": [
                "cached GT top2 margin",
                "Fisher trace 0.5*sech(m/2)^2",
                "rank4 head flip distance margin/||w_Road-w_Lane||",
                "cached margin-field spatial gradient",
                "Road/Lane composed-gain weighted Fisher prior",
                "G4/ST1 Road-Lane same-pixel frequency",
                "all-flip same-pixel frequency",
                "image-space y/x bins",
                "current-argmax neighbor disagreement",
                "current center class group",
                "HOPE Road/Lane BN top-channel stratum code",
            ],
            "bins": bins.jsonable(),
            "road_lane_head_norm": ROAD_LANE_HEAD_NORM,
            "road_lane_composed_gain": ROAD_LANE_COMPOSED_GAIN,
            "road_lane_skip_gain_ratio": ROAD_LANE_SKIP_GAIN_RATIO,
            "hope_capacity_prior_sha256": capacity_prior["source_sha256"],
        },
        "feature_mode": feature_mode,
        "context_radius": int(context_radius),
        "scorer_native_training_weighting": scorer_weighting or {},
        "receiver_payload_boundary": (
            "This payload contains only trained int16 logits. Cached margin, scorer gradients, "
            "head/BN priors, and argmax fields are compress-time/training evidence, not shipped receiver state."
        ),
    }
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ST2_MAGIC + len(raw_header).to_bytes(8, "little") + raw_header + qlogits.tobytes(order="C")


def decode_scorer_native_payload(raw: bytes) -> tuple[dict[str, Any], np.ndarray]:
    if not raw.startswith(ST2_MAGIC):
        raise ST2Error("bad ST2 scorer-native payload magic")
    header_start = len(ST2_MAGIC)
    header_len = int.from_bytes(raw[header_start : header_start + 8], "little")
    body_start = header_start + 8 + header_len
    header = json.loads(raw[header_start + 8 : body_start].decode("utf-8"))
    qlogits = np.frombuffer(raw[body_start:], dtype="<i2").copy()
    if int(header["bucket_count"]) != int(qlogits.size):
        raise ST2Error("scorer-native payload length mismatch")
    return header, qlogits


def decompress_payload(row: st1.CoderRow, *, raw_len: int) -> bytes:
    payload = Path(row.artifact_path).read_bytes()
    if row.codec == "brotli-q11":
        return brotli.decompress(payload)
    if row.codec == "zlib-9":
        import zlib

        return zlib.decompress(payload)
    if row.codec == "lzma1-raw":
        return st1.pe1.unlzma1_raw(payload, raw_len)
    raise ST2Error(f"unknown coder {row.codec}")


def scorer_native_student_leg(
    *,
    gt: np.ndarray,
    current: np.ndarray,
    margins: np.ndarray,
    road_lane_frequency: np.ndarray,
    all_flip_frequency: np.ndarray,
    capacity_prior: dict[str, Any],
    ssd_dir: Path,
    bucket_sizes: tuple[int, ...],
    feature_modes: tuple[str, ...],
    context_radius: int,
    seed: int,
    max_positive: int,
    negative_ratio: int,
    positive_weight: float,
    qscale: float,
    smoothing: float,
    bins: FeatureBins = FeatureBins(),
) -> dict[str, Any]:
    eval_pairs = tuple(int(v) for v in st1.cq1.PAIRS)
    eval_pair_set = frozenset(eval_pairs)
    train_pairs = [pair for pair in range(N_PAIRS) if pair not in eval_pair_set]
    sampled = st1.gather_band_points(
        gt=gt,
        current=current,
        pairs=train_pairs,
        frequency_map=road_lane_frequency,
        max_positive=max_positive,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 2003)
    order = rng.permutation(sampled["target"].size)
    holdout_count = max(1, int(round(0.2 * order.size)))
    hold_idx = order[:holdout_count]
    train_idx = order[holdout_count:]

    eval_points = st1.gather_eval_band_points(gt=gt, current=current, pairs=eval_pairs)
    artifact_dir = ssd_dir / "scorer_native_payloads"
    rows: list[dict[str, Any]] = []
    native_weights_all = scorer_native_sample_weights(
        margins=margins,
        pairs=sampled["pairs"],
        y=sampled["y"],
        x=sampled["x"],
        road_lane_frequency=road_lane_frequency,
        bins=bins,
    )
    scorer_weighting = {
        "formula": "positive_weight * (1 + 2*fisher/0.5) * (1 + min(freq,32)/32) * low_margin_0p137_bonus",
        "low_margin_bonus": 1.35,
        "source": "#141 margin/saliency plus G4 same-pixel recurrence; training-time only",
        "sample_weight_min": float(native_weights_all.min()),
        "sample_weight_max": float(native_weights_all.max()),
        "sample_weight_mean": float(native_weights_all.mean()),
    }

    for feature_mode in feature_modes:
        for bucket_count in bucket_sizes:
            train_hash = model_hashes(
                feature_mode=feature_mode,
                current=current,
                margins=margins,
                pairs=sampled["pairs"][train_idx],
                y=sampled["y"][train_idx],
                x=sampled["x"][train_idx],
                road_lane_frequency=road_lane_frequency,
                all_flip_frequency=all_flip_frequency,
                capacity_prior=capacity_prior,
                bucket_count=bucket_count,
                context_radius=context_radius,
                bins=bins,
            )
            hold_hash = model_hashes(
                feature_mode=feature_mode,
                current=current,
                margins=margins,
                pairs=sampled["pairs"][hold_idx],
                y=sampled["y"][hold_idx],
                x=sampled["x"][hold_idx],
                road_lane_frequency=road_lane_frequency,
                all_flip_frequency=all_flip_frequency,
                capacity_prior=capacity_prior,
                bucket_count=bucket_count,
                context_radius=context_radius,
                bins=bins,
            )
            eval_hash = model_hashes(
                feature_mode=feature_mode,
                current=current,
                margins=margins,
                pairs=eval_points["pairs"],
                y=eval_points["y"],
                x=eval_points["x"],
                road_lane_frequency=road_lane_frequency,
                all_flip_frequency=all_flip_frequency,
                capacity_prior=capacity_prior,
                bucket_count=bucket_count,
                context_radius=context_radius,
                bins=bins,
            )
            qlogits = train_bucket_logits_weighted(
                hashes=train_hash,
                target=sampled["target"][train_idx],
                native_weight=native_weights_all[train_idx],
                bucket_count=bucket_count,
                positive_weight=positive_weight,
                smoothing=smoothing,
                qscale=qscale,
            )
            raw = scorer_native_payload(
                qlogits,
                bucket_count=bucket_count,
                qscale=qscale,
                bins=bins,
                capacity_prior=capacity_prior,
                feature_mode=feature_mode,
                context_radius=context_radius,
                scorer_weighting=scorer_weighting,
            )
            best, coder_rows = st1.compress_payload(
                f"st2_{feature_mode}_{bucket_count}", raw, artifact_dir
            )
            decoded_header, decoded_qlogits = decode_scorer_native_payload(
                decompress_payload(best, raw_len=len(raw))
            )
            if not np.array_equal(decoded_qlogits, qlogits):
                raise ST2Error("decoded scorer-native payload differs from trained qlogits")
            hold_prob = st1.sigmoid(decoded_qlogits[hold_hash].astype(np.float32) / np.float32(qscale))
            threshold = st1.choose_threshold(hold_prob, sampled["target"][hold_idx])
            eval_metrics = st1.evaluate_bucket_model(
                qlogits=decoded_qlogits,
                hashes=eval_hash,
                target=eval_points["target"],
                charter_target=eval_points["charter_target_inside_gt_band"],
                charter_denominator=int(eval_points["charter_target_total"][0]),
                qscale=qscale,
                threshold=threshold["threshold"],
            )
            economics = []
            for label, carrier_bytes in st1.CARRIER_PRICE_ROWS.items():
                total_bytes = int(best.bytes + carrier_bytes)
                economics.append(
                    {
                        "carrier": label,
                        "student_bytes": int(best.bytes),
                        "carrier_bytes": int(carrier_bytes),
                        "total_bytes": total_bytes,
                        "break_even_survival_n600_road_lane": float(
                            total_bytes / (st1.W_BYTES_PER_FLIP * st1.ROAD_LANE_N600_FLIP_MASS)
                        ),
                        "matched_byte_delta_vs_carrier": int(best.bytes - carrier_bytes),
                    }
                )
            rows.append(
                {
                    "bucket_count": int(bucket_count),
                    "params": int(bucket_count),
                    "feature": feature_description(feature_mode, context_radius),
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
    selected_best = selected["payload"]["best"]
    selected_metrics = selected["eval_n32"]
    return {
        "schema": "ddm_st2_scorer_native_student_leg.v1",
        "axis": "[macOS-CPU advisory / cached frozen-scorer-native real-coder pricing]",
        "selection_rule": "max CQ1 SE3 r1 n32 flip-set hits, then band IoU, then smaller real-coded payload",
        "feature_modes": list(feature_modes),
        "context_radius": int(context_radius),
        "scorer_native_training_weighting": scorer_weighting,
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
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
        "charter_band_definition": f"cq1.band_for(cx1_argmax, radius={st1.CHARTER_BAND_RADIUS})",
        "rows": rows,
        "selected": selected,
        "st1_baseline_comparison": {
            "baseline": {
                "bucket_count": 8192,
                "bytes": ST1_BASELINE_BYTES,
                "hits": ST1_BASELINE_HITS,
                "denominator": ST1_BASELINE_DENOMINATOR,
                "band_iou": ST1_BASELINE_BAND_IOU,
            },
            "selected_bytes_delta_vs_st1": int(selected_best.bytes - ST1_BASELINE_BYTES),
            "selected_hits_delta_vs_st1": int(selected_metrics["charter_hits"] - ST1_BASELINE_HITS),
            "beats_hashed_context_at_matched_bytes": bool(
                selected_metrics["charter_hits"] > ST1_BASELINE_HITS and selected_best.bytes <= ST1_BASELINE_BYTES
            ),
        },
        "verdict_scope": (
            "FORMULATION-scoped scorer-native targeter table for OD2/compress-time routing; "
            "not a receiver, not a legal archive row, and not a family kill if it loses."
        ),
    }


def recall_queries() -> list[dict[str, Any]]:
    return [
        {
            "query": "ST2 charter/common contract/ST1 addendum/ST1 receipt/operator addenda 4 and 5",
            "sources": [
                ".omx/tmp/codex_runs/st2_prompt.md",
                ".omx/tmp/codex_runs/_common_contract.md",
                ".omx/research/ddm_st1_20260805/CHARTER_ADDENDUM.md",
                ".omx/research/ddm_st1_20260805/ST1_RECEIPT_20260805.md",
                ".omx/research/operator_directive_per_edge_optimality_criteria_20260805.md",
            ],
            "changed_plan": (
                "Kept the same n32 CQ1 Road/Lane denominator and real-coder pricing, then compared "
                "scorer-native-only, scorer-weighted-context, and scorer-native-plus-context targeters."
            ),
        },
        {
            "query": "g3/g4/pc2/m91/fl1 Road Lane structure",
            "sources": [
                ".omx/research/codex_findings_ddm_g3_score_atlas_20260722T204813Z_codex.md",
                ".omx/research/codex_findings_ddm_g4_spatial_stationarity_20260722T212138Z_codex.md",
                ".omx/research/ddm_pc2_perclass_road_edges_20260802.md",
                ".omx/research/ddm_fl1_perclass_flicker_floors_20260731.md",
                ".omx/research/ddm_cg1_directed_edge_margin_n600.json",
            ],
            "changed_plan": (
                "Kept Road/Lane as the surgical edge and used same-pixel recurrence, not a class-level "
                "or all-flip average, as the transport prior."
            ),
        },
        {
            "query": "#141 margin field/logit distillation and gradient saliency",
            "sources": [
                "experiments/probe_label_noise_floor_and_margin_saliency.py",
                ".omx/research/label_noise_floor_and_margin_saliency_20260618.json",
                "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/targets_n600/targets_meta.json",
            ],
            "changed_plan": (
                "Loaded the cached GT top-2 margin field and used Fisher/margin-gradient features; "
                "no new frozen-scorer forward was launched."
            ),
        },
        {
            "query": "lg1 rank-4 head hyperplanes and composed Road-Lane gradient/gain",
            "sources": [
                "src/tac/canonical_equations/segnet_head_rank4_flipdist_20260715.py",
                "src/tac/canonical_equations/lane_gain_chain_composed_20260716.py",
                "experiments/results/lane_channel_refactor_20260716/s1_gain_chain.json",
            ],
            "changed_plan": (
                "Added margin divided by the Road/Lane head normal and gain-weighted Fisher as "
                "explicit feature coordinates."
            ),
        },
        {
            "query": "#725 HOPE BN capacity Road/Lane strata",
            "sources": [
                "src/tac/canonical_equations/hope_bn_capacity_per_stratum_20260727.py",
                ".omx/research/ddm_hb1_hope_bn_capacity_20260727T0001Z/hope_per_stratum_capacity_table.json",
            ],
            "changed_plan": (
                "Added a Road/Lane HOPE top-channel stratum code, split by boundary/cell and "
                "static/transient recurrence."
            ),
        },
    ]


def source_custody(
    *,
    gt_argmax: Path,
    current_argmax: Path,
    margin_f16: Path,
    margin_meta: Path,
    capacity_table: Path,
) -> dict[str, Any]:
    margin_meta_payload = json.loads(margin_meta.read_text(encoding="utf-8")) if margin_meta.exists() else {}
    rows: dict[str, dict[str, Any]] = {}
    for label, path in (
        ("gt_argmax_n600", gt_argmax),
        ("cx1_argmax_n600", current_argmax),
        ("gt_segnet_margin_f16", margin_f16),
        ("hope_capacity_table", capacity_table),
    ):
        size, digest = st1.sha256_file(path)
        rows[label] = {"path": str(path), "bytes": int(size), "sha256": digest}
    return {
        "files": rows,
        "margin_meta": margin_meta_payload,
        "segnet_weights_sha256": SEGNET_WEIGHTS_SHA256,
        "gt_decode_rule": "cached source was built from upstream video using frame_utils.yuv420_to_rgb per targets_meta",
    }


def write_markdown_receipt(path: Path, receipt: dict[str, Any]) -> None:
    leg = receipt["leg_st2_scorer_native_student"]
    selected = leg["selected"]
    selected_best = st1.jsonable(selected["payload"]["best"])
    selected_metrics = selected["eval_n32"]
    comparison = leg["st1_baseline_comparison"]
    lines = [
        "# ST2 scorer-native surgical student receipt - 2026-08-05",
        "",
        "Status: **SCORER-NATIVE TARGETER / REAL-CODER-PRICED / NO FRONTIER MOVE**.",
        "",
        "Axis: `[macOS-CPU advisory / cached frozen-scorer-native real-coder pricing]`.",
        "`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.",
        "",
        "## Answer First",
        "",
        (
            "ST2 selected the `{bucket}` scorer-native bucket table: `{bytes}` counted bytes, "
            "`{hits}/{den}` charter n32 flip-set hits, band IoU `{iou:.6f}`."
        ).format(
            bucket=selected["bucket_count"],
            bytes=selected_best["bytes"],
            hits=selected_metrics["charter_hits"],
            den=selected_metrics["charter_denominator"],
            iou=selected_metrics["band_iou"],
        ),
        (
            "Against ST1's 8192 hashed-context row, the selected ST2 row has "
            "`{hit_delta:+d}` hits and `{byte_delta:+d}` bytes."
        ).format(
            hit_delta=comparison["selected_hits_delta_vs_st1"],
            byte_delta=comparison["selected_bytes_delta_vs_st1"],
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
            "## Scorer-Native Inputs",
            "",
            "- Cached GT top-2 margin field: consumed as a training/targeting feature, not shipped.",
            "- Fisher trace: `0.5*sech(m/2)^2` from cached margin.",
            "- Head hyperplane basis: Road/Lane `||w_c-w_c'|| = {:.3f}`.".format(ROAD_LANE_HEAD_NORM),
            "- Gradient prior: #141 boundary saliency plus lg1 Road/Lane composed gain `{:.4f}`.".format(
                ROAD_LANE_COMPOSED_GAIN
            ),
            "- HOPE BN prior: Road/Lane top-channel stratum code from #725.",
            "",
            "## Bucket Sweep",
            "",
            "| mode | buckets | best coder | bytes | hits/8670 | precision | recall | band IoU | threshold |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in leg["rows"]:
        best = st1.jsonable(row["payload"]["best"])
        metrics = row["eval_n32"]
        threshold = row["threshold_from_train_holdout"]["threshold"]
        lines.append(
            "| {mode} | {bucket} | {codec} | {bytes} | {hits}/{den} | {precision:.6f} | {recall:.6f} | "
            "{iou:.6f} | {thr:.6f} |".format(
                mode=row["feature"]["mode"],
                bucket=row["bucket_count"],
                codec=best["codec"],
                bytes=best["bytes"],
                hits=metrics["charter_hits"],
                den=metrics["charter_denominator"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                iou=metrics["band_iou"],
                thr=threshold,
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- No `upstream/evaluate.py`, n600 scorer, PoseNet, paint, or realization candidate was run.",
            "- This is an OD2/compress-time targeter, not a receiver and not a legal archive row.",
            "- Counted bytes are real serialized bucket payload bytes from the Brotli/LZMA/zlib coder race.",
            "- Scorer-derived margin, gradient, head, and BN fields are not hidden in code or payload.",
            "",
            "## Next",
            "",
            (
                "Feed the selected bucket table into OD2 as a ranked target prior. Any promotion requires "
                "a scorer-free receiver/solver that realizes selected cells, then byte-closed exact eval."
            ),
            "",
            OWN_FRONTIER_LINE,
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_resume_note(path: Path, receipt: dict[str, Any]) -> None:
    leg = receipt["leg_st2_scorer_native_student"]
    selected = leg["selected"]
    selected_best = st1.jsonable(selected["payload"]["best"])
    metrics = selected["eval_n32"]
    comparison = leg["st1_baseline_comparison"]
    lines = [
        "# NEXT_IF_RESUMED - ST2 scorer-native surgical student",
        "",
        "Resume from `.omx/research/ddm_st2_20260805/ddm_st2_receipt.json`.",
        "",
        (
            "Selected row: `{bucket}` buckets, `{bytes}` real-coded bytes, `{hits}/{den}` charter hits, "
            "band IoU `{iou:.6f}`."
        ).format(
            bucket=selected["bucket_count"],
            bytes=selected_best["bytes"],
            hits=metrics["charter_hits"],
            den=metrics["charter_denominator"],
            iou=metrics["band_iou"],
        ),
        (
            "ST1 comparison: `{hit_delta:+d}` hits and `{byte_delta:+d}` bytes versus the 8192 hashed "
            "local-context row."
        ).format(
            hit_delta=comparison["selected_hits_delta_vs_st1"],
            byte_delta=comparison["selected_bytes_delta_vs_st1"],
        ),
        "",
        "Next exact-relevant unit:",
        "",
        "1. Consume the selected payload artifact from `selected.payload.best.artifact_path` as an OD2 rank prior.",
        "2. Build only a scorer-free receiver/solver realization; do not ship scorer margin/head/BN fields.",
        "3. If a realized archive survives parse-back, run the governed exact-eval path in a separate lane.",
        "",
        OWN_FRONTIER_LINE,
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def parse_bucket_sizes(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ST2Error("bucket sizes must be positive integers")
    return values


def parse_feature_modes(text: str) -> tuple[str, ...]:
    allowed = {"scorer_native", "scorer_weighted_context", "scorer_native_plus_context"}
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise ST2Error("feature modes cannot be empty")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ST2Error(f"unknown feature modes: {unknown}")
    return values


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-argmax", type=Path, default=DEFAULT_GT_ARGMAX)
    ap.add_argument("--current-argmax", type=Path, default=DEFAULT_CURRENT_ARGMAX)
    ap.add_argument("--gt-margin-f16", type=Path, default=DEFAULT_GT_MARGIN_F16)
    ap.add_argument("--gt-margin-meta", type=Path, default=DEFAULT_GT_MARGIN_META)
    ap.add_argument("--hope-capacity-table", type=Path, default=DEFAULT_HOPE_CAPACITY_TABLE)
    ap.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    ap.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    ap.add_argument("--bucket-sizes", default="65536,32768,16384,8192,4096,2048")
    ap.add_argument(
        "--feature-modes",
        default="scorer_native,scorer_weighted_context,scorer_native_plus_context",
    )
    ap.add_argument("--context-radius", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--max-positive", type=int, default=30000)
    ap.add_argument("--negative-ratio", type=int, default=5)
    ap.add_argument("--positive-weight", type=float, default=16.0)
    ap.add_argument("--qscale", type=float, default=256.0)
    ap.add_argument("--smoothing", type=float, default=4.0)
    ap.add_argument("--required-free-bytes", type=int, default=512 * 1024 * 1024)
    args = ap.parse_args(argv)

    bucket_sizes = parse_bucket_sizes(args.bucket_sizes)
    feature_modes = parse_feature_modes(args.feature_modes)
    storage = st1.storage_preflight(args.ssd_dir, args.required_free_bytes)
    if not storage["ok"]:
        raise ST2Error(f"SSD storage preflight failed: {storage}")
    storage["cleanup_policy"] = (
        "Payload artifacts are small durable evidence. No temporary bulk is retained; "
        "future bulky realization must use this SSD tier with its own cleanup manifest."
    )

    gt, current = st1.load_argmax_pair(args.gt_argmax, args.current_argmax)
    margins = load_margin_memmap(args.gt_margin_f16)
    capacity_prior = load_road_lane_capacity_prior(args.hope_capacity_table)
    road_lane_frequency, all_flip_frequency = st1.compute_frequency_maps(gt, current)
    leg = scorer_native_student_leg(
        gt=gt,
        current=current,
        margins=margins,
        road_lane_frequency=road_lane_frequency,
        all_flip_frequency=all_flip_frequency,
        capacity_prior=capacity_prior,
        ssd_dir=args.ssd_dir,
        bucket_sizes=bucket_sizes,
        feature_modes=feature_modes,
        context_radius=args.context_radius,
        seed=args.seed,
        max_positive=args.max_positive,
        negative_ratio=args.negative_ratio,
        positive_weight=args.positive_weight,
        qscale=args.qscale,
        smoothing=args.smoothing,
    )

    args.research_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": "ddm_st2_receipt.v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "axis": "[macOS-CPU advisory / cached frozen-scorer-native real-coder pricing]",
        "score_claim": False,
        "promotion_eligible": False,
        "frontier_moved": False,
        "n600_scorer_job": False,
        "forbidden_jobs_run": {
            "upstream_evaluate": False,
            "n600_scorer": False,
            "posenet": False,
            "paint_or_realization_candidate": False,
        },
        "storage_preflight": storage,
        "source_custody": source_custody(
            gt_argmax=args.gt_argmax,
            current_argmax=args.current_argmax,
            margin_f16=args.gt_margin_f16,
            margin_meta=args.gt_margin_meta,
            capacity_table=args.hope_capacity_table,
        ),
        "recall_evidence": recall_queries(),
        "leg_st2_scorer_native_student": leg,
        "frontier_line": OWN_FRONTIER_LINE,
    }
    json_path = args.research_dir / "ddm_st2_receipt.json"
    md_path = args.research_dir / "ST2_RECEIPT_20260805.md"
    next_path = args.research_dir / "NEXT_IF_RESUMED.md"
    atomic_write_json(json_path, receipt)
    write_markdown_receipt(md_path, receipt)
    write_resume_note(next_path, receipt)
    print(json.dumps(st1.jsonable({"receipt_json": json_path, "selected": leg["selected"]}), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
