#!/usr/bin/env python
"""XO1 scorer-free cached context/orderer measurement.

This is the control-first leg of ``EU2-X1-10K-context-orderer``.  It consumes
only cached argmax/margin/token surfaces and runs no scorer/evaluator.  The
packed output is a tiny additive named-feature head that can be parsed and
consumed exactly once; it is not a renderer, mask, pixel packet, or score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization import ddm_ix2_archive_container as ix2

ARGMAX_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
FIELD_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730")
TOKENS_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy")
GP1_FLIPS_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_gp1_20260803/gp1_per_pair_flips.npy")
DEFAULT_OUT_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_xo1_20260805")

SEG_H = 384
SEG_W = 512
CELL_H = 24
CELL_W = 32
N_CLASSES = 5
DIST_MAX = 15
DIST_BUCKETS = DIST_MAX + 2
ROW_BUCKETS = 8
COL_BUCKETS = 8
TOKEN_BUCKETS = 8
RATE_DENOMINATOR = 37_545_489
GP1_ORDERING_GAP_BYTES = 106_954
CURRENT_OWN_FRONTIER = "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved."

FEATURE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("dist_to_boundary_bucket", DIST_BUCKETS),
    ("receiver_class", N_CLASSES),
    ("nearest_differing_class_or_interior", N_CLASSES + 1),
    ("row_bucket", ROW_BUCKETS),
    ("col_bucket", COL_BUCKETS),
    ("token_activity_bucket", TOKEN_BUCKETS),
)
FEATURE_OFFSETS = np.cumsum([0] + [size for _name, size in FEATURE_BLOCKS[:-1]]).astype(np.int64)
N_WEIGHTS = sum(size for _name, size in FEATURE_BLOCKS)

PACKET_MAGIC = b"XO1CTL1\0"
PACKET_HEADER = struct.Struct("<8sBBHf h")


class XO1PacketError(ValueError):
    """The XO1 packet failed strict parse-back."""


@dataclass(frozen=True)
class SampleSet:
    features: np.ndarray
    labels: np.ndarray
    margins: np.ndarray
    pair_ids: np.ndarray


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boundary(label: np.ndarray) -> np.ndarray:
    out = np.zeros(label.shape, dtype=bool)
    out[:-1, :] |= label[:-1, :] != label[1:, :]
    out[1:, :] |= label[:-1, :] != label[1:, :]
    out[:, :-1] |= label[:, :-1] != label[:, 1:]
    out[:, 1:] |= label[:, :-1] != label[:, 1:]
    return out


def dilate1(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def dist_to_boundary(label: np.ndarray) -> np.ndarray:
    dist = np.full(label.shape, DIST_MAX + 1, dtype=np.uint8)
    active = boundary(label)
    dist[active] = 0
    for radius in range(1, DIST_MAX + 1):
        expanded = dilate1(active)
        new = expanded & ~active
        dist[new] = radius
        active = expanded
        if active.all():
            break
    return dist


def nearest_diff_label(label: np.ndarray) -> np.ndarray:
    out = np.full(label.shape, N_CLASSES, dtype=np.uint8)
    for cls in range(N_CLASSES - 1, -1, -1):
        hit = np.zeros(label.shape, dtype=bool)
        eq = label == cls
        hit[:-1, :] |= eq[1:, :]
        hit[1:, :] |= eq[:-1, :]
        hit[:, :-1] |= eq[:, 1:]
        hit[:, 1:] |= eq[:, :-1]
        out[hit & (label != cls)] = cls
    return out


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos - 1) / 2.0) / (pos * neg))


def make_static_pixel_buckets() -> tuple[np.ndarray, np.ndarray]:
    rows = (np.arange(SEG_H, dtype=np.int16) * ROW_BUCKETS // SEG_H).astype(np.uint8)
    cols = (np.arange(SEG_W, dtype=np.int16) * COL_BUCKETS // SEG_W).astype(np.uint8)
    return (
        np.broadcast_to(rows[:, None], (SEG_H, SEG_W)),
        np.broadcast_to(cols[None, :], (SEG_H, SEG_W)),
    )


def token_activity_buckets(tokens: np.ndarray) -> np.ndarray:
    _base, delta = ix2._factor_mode_delta(tokens, 16)
    activity = (delta != 0).sum(axis=3).astype(np.float32)
    # Deterministic global quantiles.  The bucket transform is generic and the
    # thresholds are recorded as measured control metadata, not hidden code.
    qs = np.quantile(activity.reshape(-1), np.linspace(0.0, 1.0, TOKEN_BUCKETS + 1)[1:-1])
    buckets = np.searchsorted(qs, activity, side="right").astype(np.uint8)
    return buckets


def require_full_population(n_pairs: int, available: int, label: str) -> None:
    if n_pairs != available:
        raise ValueError(
            f"{label} requires full-population n={available}; "
            f"got n={n_pairs}. XO1 does not bank prefix-smoke subsets."
        )


def pixel_features(
    label: np.ndarray,
    dist: np.ndarray,
    nearest: np.ndarray,
    row_bucket: np.ndarray,
    col_bucket: np.ndarray,
    token_pair_buckets: np.ndarray,
    flat_indices: np.ndarray,
) -> np.ndarray:
    rows = flat_indices // SEG_W
    cols = flat_indices % SEG_W
    token_rows = rows // (SEG_H // CELL_H)
    token_cols = cols // (SEG_W // CELL_W)
    return np.stack(
        (
            dist.reshape(-1)[flat_indices],
            label.reshape(-1)[flat_indices],
            nearest.reshape(-1)[flat_indices],
            row_bucket.reshape(-1)[flat_indices],
            col_bucket.reshape(-1)[flat_indices],
            token_pair_buckets[token_rows, token_cols],
        ),
        axis=1,
    ).astype(np.uint8)


def sample_pair(
    pair: int,
    *,
    gt: np.ndarray,
    rendered: np.ndarray,
    token_buckets: np.ndarray,
    row_bucket: np.ndarray,
    col_bucket: np.ndarray,
    rng: np.random.Generator,
    max_neg_per_pair: int,
    eval_sample_per_pair: int,
    train: bool,
) -> SampleSet:
    g = np.asarray(gt[pair])
    r = np.asarray(rendered[pair])
    diff = (g != r).reshape(-1)
    positives = np.flatnonzero(diff)
    negatives = np.flatnonzero(~diff)
    if train:
        n_neg = min(max_neg_per_pair, max(positives.size * 2, 1), negatives.size)
        neg_sample = rng.choice(negatives, size=n_neg, replace=False)
        flat = np.concatenate((positives, neg_sample))
    else:
        n_pos = min(positives.size, eval_sample_per_pair // 2)
        n_neg = min(negatives.size, eval_sample_per_pair - n_pos)
        pos_sample = rng.choice(positives, size=n_pos, replace=False) if n_pos else positives[:0]
        neg_sample = rng.choice(negatives, size=n_neg, replace=False)
        flat = np.concatenate((pos_sample, neg_sample))
    rng.shuffle(flat)
    dist = dist_to_boundary(r)
    nearest = nearest_diff_label(r)
    feats = pixel_features(r, dist, nearest, row_bucket, col_bucket, token_buckets[pair], flat)
    labels = diff[flat].astype(np.float32)
    with np.load(FIELD_DIR / f"pair-{pair:06d}.npz") as z:
        margins = np.asarray(z["distill_margin"], dtype=np.float32).reshape(-1)[flat]
    pair_ids = np.full(flat.size, pair, dtype=np.int16)
    return SampleSet(feats, labels, margins, pair_ids)


def build_samples(
    *,
    n_pairs: int,
    max_neg_per_pair: int,
    eval_sample_per_pair: int,
    seed: int,
) -> tuple[SampleSet, SampleSet, dict[str, Any]]:
    gt = np.load(ARGMAX_DIR / "gt_argmax_n600.npy", mmap_mode="r")
    rendered = np.load(ARGMAX_DIR / "cx1_argmax_n600.npy", mmap_mode="r")
    tokens = np.load(TOKENS_PATH, mmap_mode="r")
    require_full_population(n_pairs, int(tokens.shape[0]), "token activity buckets")
    token_buckets = token_activity_buckets(np.asarray(tokens))
    row_bucket, col_bucket = make_static_pixel_buckets()
    rng = np.random.default_rng(seed)
    train_parts: list[SampleSet] = []
    eval_parts: list[SampleSet] = []
    t0 = time.time()
    for pair in range(n_pairs):
        part = sample_pair(
            pair,
            gt=gt,
            rendered=rendered,
            token_buckets=token_buckets,
            row_bucket=row_bucket,
            col_bucket=col_bucket,
            rng=rng,
            max_neg_per_pair=max_neg_per_pair,
            eval_sample_per_pair=eval_sample_per_pair,
            train=(pair % 2 == 0),
        )
        if pair % 2 == 0:
            train_parts.append(part)
        else:
            eval_parts.append(part)
        if (pair + 1) % 100 == 0:
            print(f"xo1 sample {pair + 1}/{n_pairs} elapsed={time.time() - t0:.1f}s", flush=True)

    def join(parts: list[SampleSet]) -> SampleSet:
        return SampleSet(
            features=np.concatenate([p.features for p in parts], axis=0),
            labels=np.concatenate([p.labels for p in parts], axis=0),
            margins=np.concatenate([p.margins for p in parts], axis=0),
            pair_ids=np.concatenate([p.pair_ids for p in parts], axis=0),
        )

    meta = {
        "token_bucket_shape": list(token_buckets.shape),
        "token_bucket_histogram": np.bincount(token_buckets.reshape(-1), minlength=TOKEN_BUCKETS).astype(int).tolist(),
        "feature_blocks": [{"name": name, "size": size} for name, size in FEATURE_BLOCKS],
        "seed": seed,
    }
    return join(train_parts), join(eval_parts), meta


def logits_for(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    out = np.full(features.shape[0], bias, dtype=np.float32)
    for col, offset in enumerate(FEATURE_OFFSETS):
        out += weights[offset + features[:, col]]
    return out


def fit_additive_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    batch_size: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    weights = np.zeros(N_WEIGHTS, dtype=np.float32)
    p = float(labels.mean())
    bias = float(math.log(max(p, 1e-6) / max(1.0 - p, 1e-6)))
    trace: list[dict[str, float]] = []
    n = labels.size
    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = features[idx]
            yb = labels[idx]
            z = np.clip(logits_for(xb, weights, bias), -30.0, 30.0)
            pred = 1.0 / (1.0 + np.exp(-z))
            grad = (pred - yb).astype(np.float32)
            total_loss += float(
                -(yb * np.log(np.clip(pred, 1e-6, 1.0)) + (1.0 - yb) * np.log(np.clip(1.0 - pred, 1e-6, 1.0))).sum()
            )
            scale = 1.0 / max(1, idx.size)
            grad_w = np.zeros_like(weights)
            for col, offset in enumerate(FEATURE_OFFSETS):
                np.add.at(grad_w, offset + xb[:, col], grad * scale)
            weights -= learning_rate * (grad_w + l2 * weights)
            bias -= learning_rate * float(grad.mean())
        trace.append({"epoch": float(epoch), "mean_loss": total_loss / n, "bias": bias})
    return weights, bias, trace


def quantize_head(weights: np.ndarray, bias: float) -> tuple[bytes, float, np.ndarray, int]:
    all_values = np.concatenate((weights.astype(np.float32), np.asarray([bias], dtype=np.float32)))
    scale = float(max(np.max(np.abs(all_values)) / 127.0, 1e-6))
    q_weights = np.rint(weights / scale).clip(-127, 127).astype(np.int8)
    q_bias = int(np.rint(bias / scale).clip(-32767, 32767))
    payload = bytearray()
    payload.extend(PACKET_HEADER.pack(PACKET_MAGIC, 1, len(FEATURE_BLOCKS), N_WEIGHTS, scale, q_bias))
    for offset, (name, size) in zip(FEATURE_OFFSETS, FEATURE_BLOCKS, strict=True):
        name_bytes = name.encode("ascii")
        if len(name_bytes) > 255:
            raise ValueError("feature name too long")
        payload.extend(struct.pack("<BH", len(name_bytes), size))
        payload.extend(name_bytes)
        payload.extend(q_weights[offset : offset + size].tobytes())
    return bytes(payload), scale, q_weights, q_bias


def parse_head_packet(payload: bytes) -> dict[str, Any]:
    cursor = 0
    if len(payload) < PACKET_HEADER.size:
        raise XO1PacketError("packet too short")
    magic, version, block_count, n_weights, scale, q_bias = PACKET_HEADER.unpack_from(payload, cursor)
    cursor += PACKET_HEADER.size
    if magic != PACKET_MAGIC or version != 1 or block_count != len(FEATURE_BLOCKS) or n_weights != N_WEIGHTS:
        raise XO1PacketError("packet header differs")
    weights: list[int] = []
    blocks = []
    for expected_name, expected_size in FEATURE_BLOCKS:
        if cursor + 3 > len(payload):
            raise XO1PacketError("feature block header truncated")
        name_len, size = struct.unpack_from("<BH", payload, cursor)
        cursor += 3
        name = payload[cursor : cursor + name_len].decode("ascii", "strict")
        cursor += name_len
        if name != expected_name or size != expected_size:
            raise XO1PacketError("feature block schema differs")
        end = cursor + size
        if end > len(payload):
            raise XO1PacketError("feature weights truncated")
        block_weights = np.frombuffer(payload[cursor:end], dtype=np.int8).astype(int).tolist()
        weights.extend(block_weights)
        blocks.append({"name": name, "size": size})
        cursor = end
    if cursor != len(payload):
        raise XO1PacketError("packet has trailing bytes")
    return {
        "version": version,
        "feature_blocks": blocks,
        "n_weights": n_weights,
        "scale": scale,
        "q_bias": q_bias,
        "weights_sha256": sha256_bytes(np.asarray(weights, dtype=np.int8).tobytes()),
        "final_cursor": cursor,
        "packet_len": len(payload),
        "consumed_exactly_once": True,
    }


def top_overlap(scores: np.ndarray, oracle: np.ndarray, fraction: float) -> dict[str, Any]:
    k = max(1, round(scores.size * fraction))
    pred = np.argpartition(scores, -k)[-k:]
    truth = np.argpartition(oracle, -k)[-k:]
    overlap = len(set(pred.tolist()) & set(truth.tolist()))
    return {"fraction": fraction, "k": k, "overlap": overlap, "overlap_fraction": overlap / k}


def evaluate_head(sample: SampleSet, weights: np.ndarray, bias: float) -> dict[str, Any]:
    scores = logits_for(sample.features, weights, bias)
    oracle_priority = -sample.margins.astype(np.float32)
    return {
        "n_samples": int(sample.labels.size),
        "positive_rate": float(sample.labels.mean()),
        "flip_auc": auc_score(scores, sample.labels),
        "spearman_vs_negative_margin": spearman(scores, oracle_priority),
        "top2pct_overlap_vs_margin": top_overlap(scores, oracle_priority, 0.02),
        "top5pct_overlap_vs_margin": top_overlap(scores, oracle_priority, 0.05),
    }


def pair_scores(
    *,
    n_pairs: int,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    rendered = np.load(ARGMAX_DIR / "cx1_argmax_n600.npy", mmap_mode="r")
    tokens = np.load(TOKENS_PATH, mmap_mode="r")
    require_full_population(n_pairs, int(tokens.shape[0]), "pair scoring")
    token_buckets = token_activity_buckets(np.asarray(tokens))
    row_bucket, col_bucket = make_static_pixel_buckets()
    all_indices = np.arange(SEG_H * SEG_W, dtype=np.int64)
    scores = np.zeros(n_pairs, dtype=np.float64)
    for pair in range(n_pairs):
        label = np.asarray(rendered[pair])
        dist = dist_to_boundary(label)
        nearest = nearest_diff_label(label)
        feats = pixel_features(label, dist, nearest, row_bucket, col_bucket, token_buckets[pair], all_indices)
        logits = logits_for(feats, weights, bias)
        # Mean predicted contested-site mass; monotonic under the logistic link.
        scores[pair] = float(logits.mean())
        if (pair + 1) % 100 == 0:
            print(f"xo1 pair-score {pair + 1}/{n_pairs}", flush=True)
    return scores


def token_order_byte_delta(pair_priority: np.ndarray, *, n_pairs: int) -> dict[str, Any]:
    tokens_all = np.load(TOKENS_PATH, mmap_mode="r")
    require_full_population(n_pairs, int(tokens_all.shape[0]), "token byte delta")
    tokens = np.asarray(tokens_all)
    base_frame = ix2.encode_token_frame(tokens, levels=16)
    base_back = ix2.decode_token_frame(base_frame)
    base_ok = bool(np.array_equal(base_back, tokens))

    pred_order = np.argsort(-pair_priority, kind="mergesort")
    pred_frame = ix2.encode_token_frame(tokens[pred_order], levels=16)
    pred_back = ix2.decode_token_frame(pred_frame)
    inv = np.empty_like(pred_order)
    inv[pred_order] = np.arange(pred_order.size)
    pred_restored = pred_back[inv]

    per_pair_flips_all = np.load(GP1_FLIPS_PATH, mmap_mode="r")
    require_full_population(n_pairs, int(per_pair_flips_all.shape[0]), "GP1 flip order")
    per_pair_flips = np.asarray(per_pair_flips_all)
    oracle_order = np.argsort(-per_pair_flips, kind="mergesort")
    oracle_frame = ix2.encode_token_frame(tokens[oracle_order], levels=16)
    oracle_back = ix2.decode_token_frame(oracle_frame)
    inv_oracle = np.empty_like(oracle_order)
    inv_oracle[oracle_order] = np.arange(oracle_order.size)

    return {
        "coder": "src.tac.optimization.ddm_ix2_archive_container.encode_token_frame(levels=16)",
        "baseline_bytes": len(base_frame),
        "baseline_sha256": sha256_bytes(base_frame),
        "baseline_roundtrip": base_ok,
        "control_order_bytes": len(pred_frame),
        "control_order_sha256": sha256_bytes(pred_frame),
        "control_order_roundtrip_and_inverse_restore": bool(np.array_equal(pred_restored, tokens)),
        "control_saved_bytes_vs_current_order": len(base_frame) - len(pred_frame),
        "oracle_flip_order_bytes": len(oracle_frame),
        "oracle_flip_order_sha256": sha256_bytes(oracle_frame),
        "oracle_flip_order_roundtrip_and_inverse_restore": bool(np.array_equal(oracle_back[inv_oracle], tokens)),
        "oracle_flip_saved_bytes_vs_current_order": len(base_frame) - len(oracle_frame),
        "control_pair_spearman_vs_gp1_flips": spearman(pair_priority, per_pair_flips.astype(np.float64)),
        "top_order_pairs_control_first10": pred_order[:10].astype(int).tolist(),
        "top_order_pairs_oracle_first10": oracle_order[:10].astype(int).tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, heldout, sample_meta = build_samples(
        n_pairs=args.n_pairs,
        max_neg_per_pair=args.max_neg_per_pair,
        eval_sample_per_pair=args.eval_sample_per_pair,
        seed=args.seed,
    )
    weights, bias, trace = fit_additive_logistic(
        train.features,
        train.labels,
        seed=args.seed + 1,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        batch_size=args.batch_size,
    )
    packet, scale, q_weights, q_bias = quantize_head(weights, bias)
    packet_path = out_dir / "xo1_control_head.xo1pkt"
    packet_path.write_bytes(packet)
    parsed = parse_head_packet(packet)
    deq_weights = q_weights.astype(np.float32) * scale
    deq_bias = float(q_bias) * scale
    pair_priority = pair_scores(n_pairs=args.n_pairs, weights=deq_weights, bias=deq_bias)
    byte_delta = token_order_byte_delta(pair_priority, n_pairs=args.n_pairs)

    saved = max(0, int(byte_delta["control_saved_bytes_vs_current_order"]))
    recovered_fraction = saved / GP1_ORDERING_GAP_BYTES
    if packet_path.stat().st_size <= 15_000 and (saved >= 30_000 or recovered_fraction >= 0.30):
        verdict = "GO_10K_CONTROL_ONLY"
    elif packet_path.stat().st_size <= 15_000 and (saved >= 15_000 or recovered_fraction >= 0.15):
        verdict = "WEAK_GO_CONTROL_ONLY"
    else:
        verdict = "NO_GO_CONTROL_ONLY"

    result = {
        "schema": "ddm_xo1_context_orderer_control.v1",
        "axis": "[macOS-CPU scorer-free cached-ordering byte-only]",
        "score_claim": False,
        "scorer_forwards": 0,
        "upstream_evaluate_py": False,
        "n_pairs": args.n_pairs,
        "seed": args.seed,
        "control": {
            "type": "int8 additive named-feature logistic/Rudin-style head",
            "train_samples_even_pairs": int(train.labels.size),
            "train_positive_rate": float(train.labels.mean()),
            "heldout_samples_odd_pairs": int(heldout.labels.size),
            "heldout": evaluate_head(heldout, deq_weights, deq_bias),
            "fit_trace": trace,
            "packet": {
                "path": str(packet_path),
                "bytes": packet_path.stat().st_size,
                "sha256": file_sha256(packet_path),
                "parse_back": parsed,
            },
        },
        "token_context_delta": byte_delta,
        "go_nogo": {
            "verdict": verdict,
            "gp1_ordering_gap_bytes": GP1_ORDERING_GAP_BYTES,
            "control_saved_bytes_nonnegative": saved,
            "ordering_gap_recovered_fraction": recovered_fraction,
            "token_context_saved_bytes": int(byte_delta["control_saved_bytes_vs_current_order"]),
            "packet_le_15000": packet_path.stat().st_size <= 15_000,
            "go_bar": {
                "max_packet_bytes": 15_000,
                "ordering_gap_recovery_fraction": 0.30,
                "token_context_saved_bytes": 30_000,
            },
            "weak_go_bar": {
                "ordering_gap_recovery_fraction": 0.15,
                "token_context_saved_bytes": 15_000,
            },
        },
        "input_artifacts": {
            "gt_argmax_n600": {
                "path": str(ARGMAX_DIR / "gt_argmax_n600.npy"),
                "sha256": file_sha256(ARGMAX_DIR / "gt_argmax_n600.npy"),
            },
            "cx1_argmax_n600": {
                "path": str(ARGMAX_DIR / "cx1_argmax_n600.npy"),
                "sha256": file_sha256(ARGMAX_DIR / "cx1_argmax_n600.npy"),
            },
            "cx1_tokens": {"path": str(TOKENS_PATH), "sha256": file_sha256(TOKENS_PATH)},
            "gp1_per_pair_flips": {"path": str(GP1_FLIPS_PATH), "sha256": file_sha256(GP1_FLIPS_PATH)},
            "margin_field_dir": str(FIELD_DIR),
        },
        "sample_meta": sample_meta,
        "boundaries": {
            "public_comma10k_student_trained": False,
            "student_reason": "control leg only; public fetch is recorded separately when attempted",
            "not_a_score": True,
            "not_a_renderer": True,
            "token_order_integration": "same IX2 token coder surface, order-reencode economics only; no inflate.py integration or scorer authority",
        },
        "frontier_line": CURRENT_OWN_FRONTIER,
    }

    json_path = out_dir / "xo1_control_measurement.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out_dir / "SHA256SUMS").write_text(
        f"{file_sha256(packet_path)}  {packet_path.name}\n{file_sha256(json_path)}  {json_path.name}\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-neg-per-pair", type=int, default=1800)
    parser.add_argument("--eval-sample-per-pair", type=int, default=1600)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["go_nogo"], indent=2, sort_keys=True))
    print(CURRENT_OWN_FRONTIER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
