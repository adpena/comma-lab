#!/usr/bin/env python3
"""Bounded real-cache HSG diagnostics for the Nielsen crosswalk.

This probe is research-only.  It reads 64 deterministic pairs from the existing
frozen-SegNet logit/argmax cache, never writes to the cache, and emits JSON to
stdout.  It does not construct a receiver candidate or claim score authority.
"""

from __future__ import annotations

import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tac.boundary_math.power_diagram_witness import read_frozen_segmentation_head
from tac.lossless.range_coder import decode_static_symbols, encode_static_symbols

SEED = 1234
N_PAIRS = 64
N_CLASSES = 5
HEIGHT = 384
WIDTH = 512
SAMPLES_PER_PAIR_CLASS = 16
CODEBOOK_SIZE = 4

LOGITS_PATH = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/"
    "teacher_logits_n600/gt_segnet_logits.f16"
)
LABELS_PATH = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/"
    "targets_n600/gt_segnet_argmax.u8"
)
HEAD_PATH = Path("/Users/adpena/Projects/pact/upstream/models/segnet.safetensors")

LOGITS_BYTES = 1_179_648_000
LABELS_BYTES = 117_964_800
LOGITS_SHA256_EXISTING_CUSTODY = (
    "41d3ef535f5b5855fe17aab678580114a50309dc48d04948af62c2f563ed3b52"
)
LABELS_SHA256_EXISTING_CUSTODY = (
    "36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68"
)
HEAD_SHA256_EXISTING_CUSTODY = (
    "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
)


def _validate_input(path: Path, expected_bytes: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise ValueError(f"{path} has {actual} bytes; expected {expected_bytes}")


def _variation_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    return np.max(difference, axis=-1) - np.min(difference, axis=-1)


def _log_softmax(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    maximum = np.max(values64, axis=-1, keepdims=True)
    normalizer = maximum + np.log(np.sum(np.exp(values64 - maximum), axis=-1, keepdims=True))
    return values64 - normalizer


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    if np.std(first_rank) == 0.0 or np.std(second_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _sample_real_logits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_input(LOGITS_PATH, LOGITS_BYTES)
    _validate_input(LABELS_PATH, LABELS_BYTES)
    logits = np.memmap(
        LOGITS_PATH,
        dtype=np.float16,
        mode="r",
        shape=(600, N_CLASSES, HEIGHT, WIDTH),
    )
    labels = np.memmap(
        LABELS_PATH,
        dtype=np.uint8,
        mode="r",
        shape=(600, HEIGHT, WIDTH),
    )
    pair_ids = np.linspace(0, 599, N_PAIRS, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    sampled_logits: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []
    sampled_pairs: list[np.ndarray] = []
    for pair_id in pair_ids.tolist():
        pair_labels = np.asarray(labels[pair_id]).reshape(-1)
        pair_logits = np.asarray(logits[pair_id], dtype=np.float32).reshape(N_CLASSES, -1).T
        for class_id in range(N_CLASSES):
            candidates = np.flatnonzero(pair_labels == class_id)
            if candidates.size == 0:
                continue
            count = min(SAMPLES_PER_PAIR_CLASS, int(candidates.size))
            chosen = rng.choice(candidates, size=count, replace=False)
            sampled_logits.append(pair_logits[chosen])
            sampled_labels.append(np.full(count, class_id, dtype=np.int64))
            sampled_pairs.append(np.full(count, pair_id, dtype=np.int64))
    return (
        np.concatenate(sampled_logits, axis=0).astype(np.float64),
        np.concatenate(sampled_labels, axis=0),
        np.concatenate(sampled_pairs, axis=0),
    )


def _pairwise_distance_to_centers(
    points: np.ndarray, centers: np.ndarray, *, metric: str
) -> np.ndarray:
    difference = points[:, None, :] - centers[None, :, :]
    if metric == "hilbert":
        return np.max(difference, axis=2) - np.min(difference, axis=2)
    if metric == "euclidean":
        return np.sqrt(np.sum(np.square(difference), axis=2))
    raise ValueError(metric)


def _farthest_first(points: np.ndarray, *, metric: str, k: int) -> np.ndarray:
    chosen = [0]
    while len(chosen) < k:
        distances = _pairwise_distance_to_centers(points, points[chosen], metric=metric)
        nearest = np.min(distances, axis=1)
        nearest[np.asarray(chosen, dtype=np.int64)] = -1.0
        chosen.append(int(np.argmax(nearest)))
    return np.asarray(chosen, dtype=np.int64)


def _assignment_summary(points: np.ndarray, center_indices: np.ndarray) -> dict[str, object]:
    distances = _pairwise_distance_to_centers(
        points, points[center_indices], metric="hilbert"
    )
    assignment = np.argmin(distances, axis=1).astype(np.int64)
    nearest = distances[np.arange(points.shape[0]), assignment]
    counts = np.bincount(assignment, minlength=center_indices.size).astype(np.int64)
    probabilities = counts / counts.sum()
    positive = probabilities > 0
    entropy_bits = -float(np.sum(probabilities[positive] * np.log2(probabilities[positive])))
    frequencies = (counts + 1).tolist()
    encoded = encode_static_symbols(assignment.tolist(), frequencies=frequencies)
    decoded = decode_static_symbols(encoded, count=assignment.size, frequencies=frequencies)
    if decoded != assignment.tolist():
        raise AssertionError("static range-coder parse-back mismatch")
    return {
        "hilbert_radius_max": float(np.max(nearest)),
        "hilbert_radius_mean": float(np.mean(nearest)),
        "assignment_counts": counts.tolist(),
        "assignment_entropy_bits_per_symbol": entropy_bits,
        "static_range_payload_bytes_frequency_table_excluded": len(encoded),
        "static_range_parseback_exact": True,
    }


def main() -> None:
    logits, gt_class, pair_id = _sample_real_logits()
    if not np.isfinite(logits).all():
        raise ValueError("sampled logits contain non-finite values")
    n = logits.shape[0]
    paired = np.roll(logits, 1, axis=0)
    rng = np.random.default_rng(SEED)

    base_distance = _variation_distance(logits, paired)
    first_ray_shift = rng.normal(0.0, 4.0, size=(n, 1))
    second_ray_shift = rng.normal(0.0, 4.0, size=(n, 1))
    ray_distance = _variation_distance(logits + first_ray_shift, paired + second_ray_shift)

    shared_diagonal = rng.normal(0.0, 0.75, size=(n, N_CLASSES))
    diagonal_distance = _variation_distance(
        logits + shared_diagonal, paired + shared_diagonal
    )
    one_sided_distance = _variation_distance(logits + shared_diagonal, paired)

    log_probability_distance = _variation_distance(
        _log_softmax(logits), _log_softmax(paired)
    )

    order = np.argsort(logits, axis=1, kind="stable")
    winner = order[:, -1]
    rival = order[:, -2]
    rows = np.arange(n)
    margin = logits[rows, winner] - logits[rows, rival]
    boundary = logits.copy()
    boundary[rows, winner] -= 0.5 * margin
    boundary[rows, rival] += 0.5 * margin
    boundary_distance = _variation_distance(logits, boundary)

    weight, _bias = read_frozen_segmentation_head(HEAD_PATH)
    weight = np.asarray(weight, dtype=np.float64).reshape(N_CLASSES, -1)
    pair_normal = weight[winner] - weight[rival]
    pair_normal_norm = np.linalg.norm(pair_normal, axis=1)
    if np.any(pair_normal_norm <= 0.0):
        raise ValueError("frozen head contains a zero winner/rival normal")
    euclidean_feature_flip_distance = margin / pair_normal_norm

    centered = logits - np.mean(logits, axis=1, keepdims=True)
    clustering: list[dict[str, object]] = []
    total_hilbert_bytes = 0
    total_euclidean_bytes = 0
    for class_id in range(N_CLASSES):
        points = centered[gt_class == class_id]
        if points.shape[0] < CODEBOOK_SIZE:
            continue
        hilbert_centers = _farthest_first(points, metric="hilbert", k=CODEBOOK_SIZE)
        euclidean_centers = _farthest_first(points, metric="euclidean", k=CODEBOOK_SIZE)
        hilbert_summary = _assignment_summary(points, hilbert_centers)
        euclidean_summary = _assignment_summary(points, euclidean_centers)
        total_hilbert_bytes += int(
            hilbert_summary["static_range_payload_bytes_frequency_table_excluded"]
        )
        total_euclidean_bytes += int(
            euclidean_summary["static_range_payload_bytes_frequency_table_excluded"]
        )
        clustering.append(
            {
                "gt_class": class_id,
                "points": int(points.shape[0]),
                "codebook_size": CODEBOOK_SIZE,
                "hilbert_farthest_first": hilbert_summary,
                "euclidean_farthest_first_evaluated_in_hilbert_metric": euclidean_summary,
                "hilbert_over_euclidean_max_radius_ratio": float(
                    hilbert_summary["hilbert_radius_max"]
                    / euclidean_summary["hilbert_radius_max"]
                ),
            }
        )

    shared_diagonal_argmax_changed = np.mean(
        np.argmax(logits + shared_diagonal, axis=1) != np.argmax(logits, axis=1)
    )
    result = {
        "schema": "nielsen_hsg_n64_probe.v1",
        "measurement_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "authority": {
            "axis": "[macOS-CPU advisory] real frozen-SegNet cache diagnostic",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "receiver_closed": False,
            "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
        },
        "inputs": {
            "seed": SEED,
            "n_pairs": N_PAIRS,
            "pair_ids": np.unique(pair_id).tolist(),
            "sampled_points": int(n),
            "samples_per_pair_class_cap": SAMPLES_PER_PAIR_CLASS,
            "sampled_gt_class_counts": np.bincount(gt_class, minlength=N_CLASSES).tolist(),
            "logits": {
                "path": str(LOGITS_PATH),
                "bytes": LOGITS_BYTES,
                "sha256_from_existing_custody": LOGITS_SHA256_EXISTING_CUSTODY,
                "rehash_this_probe": False,
            },
            "labels": {
                "path": str(LABELS_PATH),
                "bytes": LABELS_BYTES,
                "sha256_from_existing_custody": LABELS_SHA256_EXISTING_CUSTODY,
                "rehash_this_probe": False,
            },
            "frozen_head": {
                "path": str(HEAD_PATH),
                "sha256_from_existing_custody": HEAD_SHA256_EXISTING_CUSTODY,
            },
        },
        "H1_projective_invariance": {
            "independent_ray_scaling_max_abs_distance_error": float(
                np.max(np.abs(ray_distance - base_distance))
            ),
            "shared_positive_diagonal_projectivity_max_abs_distance_error": float(
                np.max(np.abs(diagonal_distance - base_distance))
            ),
            "one_sided_positive_diagonal_fraction_distance_changed_gt_1e_9": float(
                np.mean(np.abs(one_sided_distance - base_distance) > 1e-9)
            ),
            "shared_positive_diagonal_fraction_argmax_changed": float(
                shared_diagonal_argmax_changed
            ),
            "interpretation": (
                "HSG quotients independent ray scale and is invariant when the same positive "
                "diagonal projectivity acts on both points; one-sided class scaling is not a "
                "gauge, and shared diagonal projectivity need not preserve the fixed argmax label."
            ),
        },
        "H2_tropical_identity": {
            "log_probability_hilbert_vs_logit_variation_max_abs_error": float(
                np.max(np.abs(log_probability_distance - base_distance))
            ),
            "interpretation": (
                "The HSG metric is exactly the max-minus-min tropical projective range on "
                "logits; this identity contains no entropy-coder or archive-byte claim."
            ),
        },
        "H3_argmax_boundary": {
            "n_points": int(n),
            "nielsen_distance_convention": "max_minus_min_without_one_half",
            "boundary_distance_vs_top1_top2_margin_max_abs_error": float(
                np.max(np.abs(boundary_distance - margin))
            ),
            "hilbert_margin_vs_rank4_euclidean_flip_spearman": _spearman(
                margin, euclidean_feature_flip_distance
            ),
            "pair_normal_norm_min": float(np.min(pair_normal_norm)),
            "pair_normal_norm_max": float(np.max(pair_normal_norm)),
            "margin_min": float(np.min(margin)),
            "margin_median": float(np.median(margin)),
            "margin_max": float(np.max(margin)),
            "interpretation": (
                "HSG boundary distance is exactly the already-used top1-top2 margin. The "
                "rank-4 Euclidean feature flip additionally prices the winner/rival head-normal "
                "norm and therefore is the stronger realization geometry."
            ),
        },
        "H5_codebook_proxy": {
            "scope": (
                "real n64 frozen-logit class strata only; not movable-site coordinates, not "
                "a complete #557 packet, and frequency-table/codebook bytes are excluded"
            ),
            "per_class": clustering,
            "static_range_payload_bytes_frequency_table_excluded": {
                "hilbert_farthest_first": total_hilbert_bytes,
                "euclidean_farthest_first": total_euclidean_bytes,
                "delta_hilbert_minus_euclidean": total_hilbert_bytes
                - total_euclidean_bytes,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "warnings": [
            "H4 is not run: no exact DirectDescriptionOpsGrammarMinimizerV1 n64 receiver-secants are present in this branch.",
            "H5 radius and payload-only rows cannot price complete archive bytes or d_seg/d_pose.",
            "The HSG smooth log-sum-exp approximation is a distinct geometry and is not substituted for exact Hilbert distance.",
        ],
    }
    if not math.isfinite(result["H3_argmax_boundary"]["hilbert_margin_vs_rank4_euclidean_flip_spearman"]):
        raise ValueError("non-finite HSG/Euclidean rank correlation")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
