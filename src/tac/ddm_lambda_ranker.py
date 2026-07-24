# SPDX-License-Identifier: MIT
"""Deterministic N600 pair-held-out lambda-ranker refit.

This is a read-only advisory analysis surface.  It joins the exact G3 atlas,
the exact EV1 receiver-closed N600 replay, and fresh scorer-value-oracle
rows.  It does not launch, mutate a run, allocate bytes, or emit a score
claim.

The learned forms are deliberately tagged ``[advisory-heuristic]``.  Their
only admission metric is concatenated out-of-fold NDCG@4, with Spearman as a
tie-break.  Every fitted transform, including decile boundaries, is learned
inside the relevant training fold.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.autopilot_rudin_daubechies.falling_rule_list import (
    FallingRule,
    FallingRuleList,
    PredicateRef,
)
from tac.autopilot_rudin_daubechies.slim_ranker import ProxyPanel
from tac.ddm_costate_law import realized_pair_distortion_delta
from tac.ddm_costate_organ import _g3_atlas, discover_sources
from tac.repo_io import sha256_file
from tac.scorer_value_oracle import OracleRow, ScorerValueOracle

SCHEMA = "ddm_lambda_ranker_n600_refit.v1"
PAIR_ROW_SCHEMA = "ddm_lambda_ranker_oof_pair_row.v1"
SLICE_SCHEMA = "ddm_lambda_ranking_error_slice.v1"
SELF_CHECK_SCHEMA = "ddm_lambda_ranker_self_check.v1"
RUN_ID = "ddm_co3_lambda_refit_full_join_20260724"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
HEURISTIC_TAG = "[advisory-heuristic]"
ADMISSION_NDCG_AT_4 = 0.75
MIXTURE_CLOSE_NDCG = 0.02
PAIR_COUNT = 600
ALPHA_GRID: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
PREREGISTRATION_PATH = (
    ".omx/research/ddm_co3_lambda_refit_full_join_preregistration_20260724.md"
)
RD1_PATH = (
    ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "ddm_rd1_lambda_continuation_frontier_receipt_v5.json"
)
RD1_SCHEMA = "ddm_rd1_lambda_continuation_frontier_receipt.v4"

BASE_FEATURES: tuple[str, ...] = (
    "log_gap",
    "log_visibility",
    "log_helpful_ratio",
    "log_byte_price",
)
MS4D_FEATURES: tuple[str, ...] = (
    *BASE_FEATURES,
    "helpful_ratio",
    "changed_cells_scaled",
    "errors_before_scaled",
    "margin_fisher_proxy",
    "boundary_fraction",
    "pose_activity",
    "pose_registration_delta_scaled",
    "direct_fisher_trace_log",
    "direct_fisher_missing",
    "gap_x_margin_fisher",
    "support_x_stationarity",
    "helpful_x_pose_activity",
    "hardness_x_margin_decile",
)
G4_FEATURES: tuple[str, ...] = (
    *MS4D_FEATURES,
    "g4_static_in_image",
    "g4_static_in_xi_proxy",
    "g4_transient",
    "gap_x_g4_transient",
    "helpful_x_g4_transient",
    "lane_flip_share",
    "movable_flip_share",
    "mycar_flip_share",
    "road_flip_share",
    "undrivable_flip_share",
)
CANDIDATES: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("factorized_refit", BASE_FEATURES, 1),
    ("factorized_ms4d_interactions", MS4D_FEATURES, 2),
    ("g4_regime_conditional", G4_FEATURES, 3),
)


class LambdaRankerError(ValueError):
    """A source, feature, fold, or fitted-model contract failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise LambdaRankerError(f"{name}: non-finite value")
    return result


def _fold_id(pair_id: int) -> int:
    digest = hashlib.sha256(
        f"ddm-co3-n600-v1:{pair_id}".encode("ascii")
    ).hexdigest()
    return int(digest[:8], 16) % 5


def _inner_fold_id(pair_id: int, outer_fold: int) -> int:
    digest = hashlib.sha256(
        f"ddm-co3-inner-v1:{outer_fold}:{pair_id}".encode("ascii")
    ).hexdigest()
    return int(digest[:8], 16) % 3


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_rho(predicted: Sequence[float], realized: Sequence[float]) -> float | None:
    if len(predicted) != len(realized) or len(predicted) < 2:
        return None
    left = _rankdata(predicted)
    right = _rankdata(realized)
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    return None if denominator == 0.0 else float((left @ right) / denominator)


def ndcg_at_k(
    predicted: Sequence[float],
    relevance: Sequence[float],
    *,
    k: int,
) -> float | None:
    if len(predicted) == 0 or len(predicted) != len(relevance) or k < 1:
        return None
    pred = np.asarray(predicted, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)
    limit = min(k, pred.size)
    discounts = 1.0 / np.log2(np.arange(2, limit + 2, dtype=np.float64))
    pred_order = np.argsort(-pred, kind="mergesort")[:limit]
    ideal_order = np.argsort(-rel, kind="mergesort")[:limit]
    ideal = float(rel[ideal_order] @ discounts)
    return None if ideal <= 0.0 else float((rel[pred_order] @ discounts) / ideal)


def _quantile_thresholds(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [
        float(value)
        for value in np.quantile(
            array,
            np.arange(0.1, 1.0, 0.1, dtype=np.float64),
            method="linear",
        )
    ]


def _decile(value: float, thresholds: Sequence[float]) -> float:
    return float(1 + sum(value > threshold for threshold in thresholds))


def _feature_context(
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
) -> dict[str, list[float]]:
    return {
        "margin_thresholds": _quantile_thresholds(
            [_finite(rows[index]["margin"], "margin") for index in train_indices]
        ),
        "hardness_thresholds": _quantile_thresholds(
            [_finite(rows[index]["hardness"], "hardness") for index in train_indices]
        ),
    }


def _feature_value(
    row: Mapping[str, Any],
    name: str,
    context: Mapping[str, Sequence[float]],
) -> float:
    if name == "margin_decile":
        return _decile(
            _finite(row["margin"], "margin"),
            context["margin_thresholds"],
        )
    if name == "hardness_decile":
        return _decile(
            _finite(row["hardness"], "hardness"),
            context["hardness_thresholds"],
        )
    if name == "hardness_x_margin_decile":
        return _decile(
            _finite(row["hardness"], "hardness"),
            context["hardness_thresholds"],
        ) * _decile(
            _finite(row["margin"], "margin"),
            context["margin_thresholds"],
        )
    return _finite(row["features"][name], name)


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    feature_names: Sequence[str],
    context: Mapping[str, Sequence[float]],
) -> np.ndarray:
    array = np.asarray(
        [
            [_feature_value(rows[index], name, context) for name in feature_names]
            for index in indices
        ],
        dtype=np.float64,
    )
    if array.ndim != 2 or not np.isfinite(array).all():
        raise LambdaRankerError("candidate feature matrix is not finite and rectangular")
    return array


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    feature_names: Sequence[str],
    context: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    target_mean = float(y.mean())
    centered = y - target_mean
    gram = z.T @ z
    regularizer = float(alpha) * np.eye(z.shape[1], dtype=np.float64)
    try:
        weights = np.linalg.solve(gram + regularizer, z.T @ centered)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram + regularizer) @ z.T @ centered
    return {
        "alpha": float(alpha),
        "feature_names": list(feature_names),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "standardized_weights": weights.tolist(),
        "target_intercept": target_mean,
        "feature_context": {
            key: [float(value) for value in values]
            for key, values in context.items()
        },
    }


def _predict_ridge(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["standardized_weights"], dtype=np.float64)
    return (
        float(model["target_intercept"])
        + ((x - mean) / scale) @ weights
    )


def _fit_predict(
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    feature_names: Sequence[str],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    context = _feature_context(rows, train_indices)
    x_train = _matrix(rows, train_indices, feature_names, context)
    x_test = _matrix(rows, test_indices, feature_names, context)
    y_train = np.asarray(
        [_finite(rows[index]["target"], "target") for index in train_indices],
        dtype=np.float64,
    )
    model = _fit_ridge(
        x_train,
        y_train,
        alpha=alpha,
        feature_names=feature_names,
        context=context,
    )
    return (
        _predict_ridge(model, x_test),
        _predict_ridge(model, x_train),
        model,
    )


def _select_alpha(
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    *,
    feature_names: Sequence[str],
    outer_fold: int,
) -> tuple[float, dict[str, Any]]:
    results: list[dict[str, Any]] = []
    target = [_finite(rows[index]["target"], "target") for index in train_indices]
    for alpha in ALPHA_GRID:
        predictions = np.zeros(len(train_indices), dtype=np.float64)
        for inner_fold in range(3):
            inner_train = [
                index
                for index in train_indices
                if _inner_fold_id(int(rows[index]["pair_id"]), outer_fold)
                != inner_fold
            ]
            inner_test_positions = [
                position
                for position, index in enumerate(train_indices)
                if _inner_fold_id(int(rows[index]["pair_id"]), outer_fold)
                == inner_fold
            ]
            inner_test = [train_indices[position] for position in inner_test_positions]
            if not inner_train or not inner_test:
                raise LambdaRankerError(
                    f"outer fold {outer_fold}: empty deterministic inner split"
                )
            predicted, _train_predicted, _model = _fit_predict(
                rows,
                inner_train,
                inner_test,
                feature_names=feature_names,
                alpha=alpha,
            )
            predictions[inner_test_positions] = predicted
        rho = spearman_rho(predictions.tolist(), target)
        ndcg = ndcg_at_k(predictions.tolist(), target, k=4)
        results.append(
            {
                "alpha": float(alpha),
                "inner_heldout_spearman_rho": rho,
                "inner_heldout_ndcg_at_4": ndcg,
            }
        )
    selected = max(
        results,
        key=lambda row: (
            -math.inf
            if row["inner_heldout_spearman_rho"] is None
            else row["inner_heldout_spearman_rho"],
            -math.inf
            if row["inner_heldout_ndcg_at_4"] is None
            else row["inner_heldout_ndcg_at_4"],
            -row["alpha"],
        ),
    )
    return float(selected["alpha"]), {
        "selection_metric": "inner_three_fold_spearman_then_ndcg_at_4",
        "selected": selected,
        "grid": results,
    }


def _percentile_against_reference(
    values: Sequence[float],
    reference: Sequence[float],
) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64), kind="mergesort")
    val = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(ref, val, side="left")
    right = np.searchsorted(ref, val, side="right")
    return (left + right) / (2.0 * max(1, ref.size))


def _oracle_meta(
    row: OracleRow,
    *,
    surface_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        **row.to_dict(include_value=False),
        "surface_counts": dict(surface_counts),
    }


def _g4_pair_mixture(
    *,
    boundary_count: int,
    interior_count: int,
    stationarity: Mapping[str, Any],
) -> dict[str, float]:
    classes = ("STATIC_IN_IMAGE", "STATIC_IN_XI_PROXY", "TRANSIENT")
    all_classes = stationarity["all"]["classes"]
    boundary_classes = stationarity["boundaries"]["classes"]
    interior_mass = {
        name: max(
            0,
            int(all_classes[name]["flip_mass"])
            - int(boundary_classes[name]["flip_mass"]),
        )
        for name in classes
    }
    interior_total = max(1, sum(interior_mass.values()))
    pair_total = max(1, boundary_count + interior_count)
    return {
        name: (
            boundary_count * _finite(boundary_classes[name]["fraction"], name)
            + interior_count * interior_mass[name] / interior_total
        )
        / pair_total
        for name in classes
    }


def _dominant_stratum(per_stratum: Mapping[str, Any]) -> str:
    closures = {
        name: int(row["errors_before"]) - int(row["errors_after"])
        for name, row in per_stratum.items()
    }
    best_name, best_value = max(closures.items(), key=lambda item: (item[1], item[0]))
    return best_name if best_value > 0 else "NO_POSITIVE_SEG_STRATUM"


def _direct_fisher_by_pair(
    margin_data: Mapping[str, Any],
) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, float]] = defaultdict(
        lambda: {"block_count": 0.0, "fisher_trace": 0.0, "adjoint_norm": 0.0}
    )
    for row in margin_data["direct_blocks"]:
        pair_id = int(row["pair_id"])
        gram = np.asarray(row["margin_fisher_gram"], dtype=np.float64)
        adjoint = np.asarray(row["composite_r_adjoint_readback"], dtype=np.float64)
        grouped[pair_id]["block_count"] += 1.0
        grouped[pair_id]["fisher_trace"] += float(np.trace(gram))
        grouped[pair_id]["adjoint_norm"] += float(np.linalg.norm(adjoint))
    return dict(grouped)


def _build_feature_rows(
    *,
    atlas: Mapping[int, Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    margin_data: Mapping[str, Any],
    pose_data: Mapping[str, Any],
    stationarity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pose_by_pair = {int(row["pair_id"]): row for row in pose_data["rows"]}
    fisher_by_pair = _direct_fisher_by_pair(margin_data)
    rows: list[dict[str, Any]] = []
    for measured in evidence_rows:
        pair_id = int(measured["source_pair_id"])
        if pair_id not in atlas or pair_id not in pose_by_pair:
            raise LambdaRankerError(f"pair {pair_id}: missing G3 or Pose6 row")
        g3 = atlas[pair_id]
        signal = g3["costate_signal"]
        segmentation = g3["segmentation"]
        geometry = g3["evaluator_response_geometry"][
            "joint_cone_summary_diagnostic_only"
        ]
        pose = pose_by_pair[pair_id]
        gap = _finite(signal["lambda_proxy_score_debt"], "gap")
        visibility = max(
            0.0,
            min(1.0, 1.0 - _finite(geometry["empty_cone_fraction"], "empty")),
        )
        changed = max(1, int(measured["changed_argmax_cells"]))
        helpful_ratio = max(
            0.0,
            min(1.0, int(measured["helpful_flips"]) / changed),
        )
        allocated = max(1.0, _finite(signal["allocated_bytes"], "allocated"))
        margin = _finite(signal["median_rank4_flip_distance"], "margin")
        margin_fisher_proxy = 0.5 / math.cosh(margin / 2.0) ** 2
        boundary_count = int(segmentation["boundary_flip_count"])
        interior_count = int(segmentation["interior_flip_count"])
        boundary_fraction = boundary_count / max(
            1, boundary_count + interior_count
        )
        g4_mix = _g4_pair_mixture(
            boundary_count=boundary_count,
            interior_count=interior_count,
            stationarity=stationarity,
        )
        pose_center = np.asarray(pose["center"], dtype=np.float64)
        tube_radius = max(1e-12, _finite(pose["tube_radius"], "tube_radius"))
        pose_activity = float(np.linalg.norm(pose_center) / tube_radius)
        direct = fisher_by_pair.get(pair_id)
        fisher_trace = 0.0 if direct is None else direct["fisher_trace"]
        flip_count = max(1, int(segmentation["flip_count"]))
        shares = {
            name: int(segmentation["class_flip_counts"].get(name, 0)) / flip_count
            for name in ("Lane", "Movable", "MyCar", "Road", "Undrivable")
        }
        distortion_delta = realized_pair_distortion_delta(
            d_seg_before=_finite(measured["d_seg_before"], "d_seg_before"),
            d_seg_after=_finite(measured["d_seg_after"], "d_seg_after"),
            d_pose_before=_finite(measured["d_pose_before"], "d_pose_before"),
            d_pose_after=_finite(measured["d_pose_after"], "d_pose_after"),
        )
        features = {
            "log_gap": math.log1p(gap),
            "log_visibility": math.log1p(visibility),
            "log_helpful_ratio": math.log1p(helpful_ratio),
            "log_byte_price": math.log1p(1.0 / allocated),
            "helpful_ratio": helpful_ratio,
            "changed_cells_scaled": changed / 1000.0,
            "errors_before_scaled": int(measured["errors_before"]) / 10_000.0,
            "margin_fisher_proxy": margin_fisher_proxy,
            "boundary_fraction": boundary_fraction,
            "pose_activity": pose_activity,
            "pose_registration_delta_scaled": _finite(
                pose["registered_center_max_abs_delta"],
                "registered_center_max_abs_delta",
            )
            * 1_000_000.0,
            "direct_fisher_trace_log": math.log1p(max(0.0, fisher_trace)),
            "direct_fisher_missing": 1.0 if direct is None else 0.0,
            "gap_x_margin_fisher": gap * margin_fisher_proxy,
            "support_x_stationarity": visibility
            * g4_mix["STATIC_IN_IMAGE"],
            "helpful_x_pose_activity": helpful_ratio * pose_activity,
            "g4_static_in_image": g4_mix["STATIC_IN_IMAGE"],
            "g4_static_in_xi_proxy": g4_mix["STATIC_IN_XI_PROXY"],
            "g4_transient": g4_mix["TRANSIENT"],
            "gap_x_g4_transient": gap * g4_mix["TRANSIENT"],
            "helpful_x_g4_transient": helpful_ratio * g4_mix["TRANSIENT"],
            "lane_flip_share": shares["Lane"],
            "movable_flip_share": shares["Movable"],
            "mycar_flip_share": shares["MyCar"],
            "road_flip_share": shares["Road"],
            "undrivable_flip_share": shares["Undrivable"],
        }
        rows.append(
            {
                "pair_id": pair_id,
                "outer_fold": _fold_id(pair_id),
                "target": max(0.0, -distortion_delta),
                "features": features,
                "margin": margin,
                "hardness": gap,
                "dominant_stratum": _dominant_stratum(measured["per_stratum"]),
                "g4_temporal_class": max(
                    g4_mix,
                    key=lambda name: (g4_mix[name], name),
                ),
                "g4_pair_class_status": (
                    "DERIVED_DOMINANT_FROM_G3_BOUNDARY_INTERIOR_COUNTS_X_"
                    "G4_AGGREGATE_CLASS_MASS"
                ),
                "g4_mixture": g4_mix,
                "direct_fisher_trace": (
                    None if direct is None else direct["fisher_trace"]
                ),
                "direct_fisher_block_count": (
                    0 if direct is None else int(direct["block_count"])
                ),
            }
        )
    rows.sort(key=lambda row: int(row["pair_id"]))
    if [row["pair_id"] for row in rows] != list(range(PAIR_COUNT)):
        raise LambdaRankerError("feature join must contain pair IDs 0..599 exactly once")
    return rows


def _candidate_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> dict[str, Any]:
    return {
        "heldout_only": True,
        "n_pairs": len(predictions),
        "spearman_rho": spearman_rho(predictions, targets),
        "ndcg_at_4": ndcg_at_k(predictions, targets, k=4),
    }


def _slice_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = [_finite(row["target"], "target") for row in feature_rows]
    margin_thresholds = _quantile_thresholds(
        [_finite(row["margin"], "margin") for row in feature_rows]
    )
    hardness_thresholds = _quantile_thresholds(
        [_finite(row["hardness"], "hardness") for row in feature_rows]
    )
    dimensions: dict[str, list[str]] = {
        "stratum": [str(row["dominant_stratum"]) for row in feature_rows],
        "g4_class": [str(row["g4_temporal_class"]) for row in feature_rows],
        "margin_decile": [
            f"D{int(_decile(_finite(row['margin'], 'margin'), margin_thresholds)):02d}"
            for row in feature_rows
        ],
        "pair_hardness_decile": [
            f"D{int(_decile(_finite(row['hardness'], 'hardness'), hardness_thresholds)):02d}"
            for row in feature_rows
        ],
    }
    out: list[dict[str, Any]] = []
    for dimension, labels in dimensions.items():
        for label in sorted(set(labels)):
            indices = [index for index, value in enumerate(labels) if value == label]
            pred = [float(predictions[index]) for index in indices]
            actual = [targets[index] for index in indices]
            innovations = [
                actual_value - predicted_value
                for actual_value, predicted_value in zip(actual, pred, strict=True)
            ]
            out.append(
                {
                    "schema": SLICE_SCHEMA,
                    "dimension": dimension,
                    "value": label,
                    "n_pairs": len(indices),
                    "mean_innovation": float(np.mean(innovations)),
                    "mean_abs_innovation": float(np.mean(np.abs(innovations))),
                    "spearman_rho": spearman_rho(pred, actual),
                    "ndcg_at_4": ndcg_at_k(pred, actual, k=4),
                    "verdict_scope": (
                        "INSTANCE:V19_N600_EXACT_RECEIVER_REPLAY_X_G3_X_"
                        "FRESH_ORACLE_ROWS"
                    ),
                }
            )
    innovations = np.asarray(targets, dtype=np.float64) - np.asarray(
        predictions, dtype=np.float64
    )
    if innovations.size >= 2 and innovations.std() > 0:
        lag_one = float(np.corrcoef(innovations[:-1], innovations[1:])[0, 1])
    else:
        lag_one = None
    return out, {
        "status": "MEASURED_HELDOUT_INNOVATIONS",
        "ordered_by": "source_pair_id",
        "n_pairs": len(feature_rows),
        "mean": float(innovations.mean()),
        "sample_std": float(innovations.std(ddof=1)),
        "lag_one_correlation": lag_one,
        "whiteness_verdict": (
            "COLORED_INNOVATIONS_FORM_MISMATCH_SIGNAL"
            if lag_one is not None and abs(lag_one) >= 0.1
            else "NO_LARGE_LAG_ONE_COLOR_DETECTED"
        ),
        "verdict_scope": "lag-one diagnostic only; not a complete innovations whiteness test",
    }


def _falling_rule_explanation(
    *,
    admission_ndcg: float,
    precision_complete: bool,
    j8f_verdict_count: int,
) -> dict[str, Any]:
    rules = FallingRuleList(
        rules=[
            FallingRule(
                name="admitted_ranker_with_complete_precision_and_j8f",
                rule_id="ddm_co3_full_decide",
                predicates=(
                    PredicateRef.parse(
                        f"admission_ndcg_at_4 >= {ADMISSION_NDCG_AT_4}"
                    ),
                    PredicateRef.parse("precision_complete >= 1"),
                    PredicateRef.parse("j8f_verdict_count >= 1"),
                ),
                predicted_score_low=ADMISSION_NDCG_AT_4,
                predicted_score_high=1.0,
            ),
            FallingRule(
                name="heldout_ranker_admitted_measurement_duties_owed",
                rule_id="ddm_co3_admitted_but_blocked",
                predicates=(
                    PredicateRef.parse(
                        f"admission_ndcg_at_4 >= {ADMISSION_NDCG_AT_4}"
                    ),
                ),
                predicted_score_low=ADMISSION_NDCG_AT_4,
                predicted_score_high=1.0,
            ),
        ],
        default_band_low=0.0,
        default_band_high=ADMISSION_NDCG_AT_4,
    )
    metadata = {
        "admission_ndcg_at_4": admission_ndcg,
        "precision_complete": int(precision_complete),
        "j8f_verdict_count": j8f_verdict_count,
    }
    chain = rules.evaluate(ProxyPanel(candidate_id=RUN_ID), metadata)
    return {
        "status": "REUSED_CANONICAL_FALLING_RULE_LIST",
        "surface": (
            "tac.autopilot_rudin_daubechies.falling_rule_list.FallingRuleList"
        ),
        "rule_chain": chain.explain(),
        "metadata": metadata,
        "gosdt_slim_disposition": (
            "REASONED_EXCLUDE: dispatch GOSDT action bands and fixed Taylor "
            "ProxyPanel features are not the N600 lambda feature schema; the "
            "generic canonical falling-rule metadata path is lossless."
        ),
    }


def build_n600_lambda_ranker_receipt(repo_root: Path) -> dict[str, Any]:
    """Fit the preregistered race and return one deterministic receipt."""

    root = repo_root.expanduser().resolve(strict=True)
    preregistration = root / PREREGISTRATION_PATH
    if not preregistration.is_file():
        raise LambdaRankerError("held-out selection preregistration is missing")
    rd1_path = root / RD1_PATH
    if not rd1_path.is_file():
        raise LambdaRankerError("RD1 dual-consistency authority is missing")
    rd1 = json.loads(rd1_path.read_text(encoding="utf-8"))
    if (
        not isinstance(rd1, dict)
        or rd1.get("schema") != RD1_SCHEMA
        or rd1.get("score_claim") is not False
    ):
        raise LambdaRankerError("RD1 dual-consistency authority failed schema/firewall")
    rd1_duals = list(rd1.get("duals") or [])
    actionable_rd1_duals = [
        row for row in rd1_duals if row.get("actionable_for_train_decision") is True
    ]
    sources = discover_sources(root)
    for name in ("g3", "g4", "ev1"):
        if not sources[name]["available"]:
            raise LambdaRankerError(f"required source {name} is unavailable")
    atlas, bulk = _g3_atlas(sources["g3"])
    if bulk.get("status") != "VERIFIED" or len(atlas) != PAIR_COUNT:
        raise LambdaRankerError("G3 N600 bulk atlas failed freshness/completeness")
    evidence_rows = list(sources["ev1"]["payload"]["v19_pair_join"]["rows"])
    if len(evidence_rows) != PAIR_COUNT:
        raise LambdaRankerError("EV1 receiver join is not exact N600")

    oracle = ScorerValueOracle(root)
    margin_row = oracle.margin_fisher()
    pose_row = oracle.pose_reference_and_tube()
    stationarity_row = oracle.stationarity_maps()
    margin_value = margin_row.require_value()
    pose_value = pose_row.require_value()
    stationarity_value = stationarity_row.require_value()
    margin_data = margin_value["data"]
    pose_data = pose_value["data"]
    stationarity = stationarity_value["stationarity_decomposition"]
    feature_rows = _build_feature_rows(
        atlas=atlas,
        evidence_rows=evidence_rows,
        margin_data=margin_data,
        pose_data=pose_data,
        stationarity=stationarity,
    )
    targets = [_finite(row["target"], "target") for row in feature_rows]
    fold_counts = {
        str(fold): sum(int(row["outer_fold"]) == fold for row in feature_rows)
        for fold in range(5)
    }
    if any(count == 0 for count in fold_counts.values()):
        raise LambdaRankerError("outer pair-held-out fold is empty")

    candidate_outputs: dict[str, dict[str, Any]] = {}
    outer_predictions: dict[str, np.ndarray] = {}
    outer_reference_predictions: dict[tuple[str, int], np.ndarray] = {}
    outer_test_predictions: dict[tuple[str, int], np.ndarray] = {}
    candidate_complexity: dict[str, int] = {}
    for candidate_id, feature_names, complexity in CANDIDATES:
        predictions = np.zeros(PAIR_COUNT, dtype=np.float64)
        fold_models: list[dict[str, Any]] = []
        for outer_fold in range(5):
            train_indices = [
                index
                for index, row in enumerate(feature_rows)
                if int(row["outer_fold"]) != outer_fold
            ]
            test_indices = [
                index
                for index, row in enumerate(feature_rows)
                if int(row["outer_fold"]) == outer_fold
            ]
            alpha, inner = _select_alpha(
                feature_rows,
                train_indices,
                feature_names=feature_names,
                outer_fold=outer_fold,
            )
            test_prediction, train_prediction, model = _fit_predict(
                feature_rows,
                train_indices,
                test_indices,
                feature_names=feature_names,
                alpha=alpha,
            )
            predictions[test_indices] = test_prediction
            outer_reference_predictions[(candidate_id, outer_fold)] = train_prediction
            outer_test_predictions[(candidate_id, outer_fold)] = test_prediction
            fold_models.append(
                {
                    "outer_fold": outer_fold,
                    "train_pair_count": len(train_indices),
                    "test_pair_count": len(test_indices),
                    "inner_selection": inner,
                    "model": model,
                }
            )
        metrics = _candidate_metrics(predictions.tolist(), targets)
        candidate_outputs[candidate_id] = {
            "candidate_id": candidate_id,
            "learned_form_tag": HEURISTIC_TAG,
            "feature_names": list(feature_names),
            "outer_fold_models": fold_models,
            "metrics": metrics,
            "status": "RACED_PAIR_HELD_OUT",
        }
        outer_predictions[candidate_id] = predictions
        candidate_complexity[candidate_id] = complexity

    pre_gb_best = max(
        candidate_outputs.values(),
        key=lambda row: (
            row["metrics"]["ndcg_at_4"],
            row["metrics"]["spearman_rho"],
            -candidate_complexity[row["candidate_id"]],
        ),
    )
    candidate_outputs["small_monotone_gb"] = {
        "candidate_id": "small_monotone_gb",
        "learned_form_tag": HEURISTIC_TAG,
        "status": (
            "SKIPPED_PREREGISTERED_PLATEAU_CONDITION_NOT_MET"
            if pre_gb_best["metrics"]["ndcg_at_4"] >= ADMISSION_NDCG_AT_4
            else "BLOCKED_IMPLEMENTATION_OWED_AFTER_CONFIRMED_PLATEAU"
        ),
        "metrics": None,
        "preregistered_trigger": (
            f"all candidates 1-3 held-out NDCG@4 < {ADMISSION_NDCG_AT_4}"
        ),
        "observed_best_candidates_1_3_ndcg_at_4": pre_gb_best["metrics"][
            "ndcg_at_4"
        ],
        "verdict_scope": (
            "no negative on monotone/GB family; optimal bounded-stump form was "
            "not triggered by the preregistered plateau rule"
        ),
    }

    mixture_predictions = np.zeros(PAIR_COUNT, dtype=np.float64)
    mixture_folds: list[dict[str, Any]] = []
    for outer_fold in range(5):
        test_indices = [
            index
            for index, row in enumerate(feature_rows)
            if int(row["outer_fold"]) == outer_fold
        ]
        inner_rank = sorted(
            (
                {
                    "candidate_id": candidate_id,
                    **candidate_outputs[candidate_id]["outer_fold_models"][
                        outer_fold
                    ]["inner_selection"]["selected"],
                }
                for candidate_id, _features, _complexity in CANDIDATES
            ),
            key=lambda row: (
                -(
                    -math.inf
                    if row["inner_heldout_ndcg_at_4"] is None
                    else row["inner_heldout_ndcg_at_4"]
                ),
                -(
                    -math.inf
                    if row["inner_heldout_spearman_rho"] is None
                    else row["inner_heldout_spearman_rho"]
                ),
                candidate_complexity[row["candidate_id"]],
            ),
        )
        best, second = inner_rank[:2]
        close = (
            best["inner_heldout_ndcg_at_4"] is not None
            and second["inner_heldout_ndcg_at_4"] is not None
            and best["inner_heldout_ndcg_at_4"]
            - second["inner_heldout_ndcg_at_4"]
            <= MIXTURE_CLOSE_NDCG
        )
        members = [best["candidate_id"], second["candidate_id"]] if close else [
            best["candidate_id"]
        ]
        mixed = np.mean(
            [
                _percentile_against_reference(
                    outer_test_predictions[(member, outer_fold)],
                    outer_reference_predictions[(member, outer_fold)],
                )
                for member in members
            ],
            axis=0,
        )
        mixture_predictions[test_indices] = mixed
        mixture_folds.append(
            {
                "outer_fold": outer_fold,
                "members": members,
                "equal_weight_normalized_rank": close,
                "inner_candidate_order": inner_rank,
            }
        )
    mixture_metrics = _candidate_metrics(mixture_predictions.tolist(), targets)
    candidate_outputs["close_form_mixture"] = {
        "candidate_id": "close_form_mixture",
        "learned_form_tag": HEURISTIC_TAG,
        "status": "RACED_PAIR_HELD_OUT_NESTED_SELECTION",
        "outer_fold_rules": mixture_folds,
        "metrics": mixture_metrics,
    }
    outer_predictions["close_form_mixture"] = mixture_predictions
    candidate_complexity["close_form_mixture"] = 4

    eligible_candidates = [
        row for row in candidate_outputs.values() if row.get("metrics") is not None
    ]
    winner = max(
        eligible_candidates,
        key=lambda row: (
            row["metrics"]["ndcg_at_4"],
            row["metrics"]["spearman_rho"],
            -candidate_complexity[row["candidate_id"]],
        ),
    )
    selected_id = str(winner["candidate_id"])
    selected_predictions = outer_predictions[selected_id]
    admission_passed = (
        float(winner["metrics"]["ndcg_at_4"]) >= ADMISSION_NDCG_AT_4
    )
    slices, innovations = _slice_rows(feature_rows, selected_predictions.tolist())

    selected_innovations = np.asarray(targets) - selected_predictions
    residual_sigma = float(selected_innovations.std(ddof=1))
    pair_rows: list[dict[str, Any]] = []
    precision_count = 0
    margin_thresholds = _quantile_thresholds(
        [_finite(row["margin"], "margin") for row in feature_rows]
    )
    hardness_thresholds = _quantile_thresholds(
        [_finite(row["hardness"], "hardness") for row in feature_rows]
    )
    for row, prediction in zip(feature_rows, selected_predictions, strict=True):
        fisher_trace = row["direct_fisher_trace"]
        if fisher_trace is not None and float(fisher_trace) > 0.0:
            standard_error = residual_sigma / math.sqrt(float(fisher_trace))
            interval = [
                float(prediction - 1.96 * standard_error),
                float(prediction + 1.96 * standard_error),
            ]
            precision_status = "DERIVED_FROM_MEASURED_DIRECT_MS4D_FISHER"
            precision_count += 1
        else:
            standard_error = None
            interval = None
            precision_status = "AWAITING_PAIR_LEVEL_MS4D_FISHER_PRECISION"
        pair_rows.append(
            {
                "schema": PAIR_ROW_SCHEMA,
                "pair_id": int(row["pair_id"]),
                "outer_fold": int(row["outer_fold"]),
                "prediction": float(prediction),
                "target": float(row["target"]),
                "innovation": float(row["target"] - prediction),
                "dominant_stratum": row["dominant_stratum"],
                "g4_temporal_class": row["g4_temporal_class"],
                "g4_pair_class_status": row["g4_pair_class_status"],
                "margin_decile": int(
                    _decile(float(row["margin"]), margin_thresholds)
                ),
                "pair_hardness_decile": int(
                    _decile(float(row["hardness"]), hardness_thresholds)
                ),
                "fisher_standard_error": standard_error,
                "fisher_95_interval": interval,
                "precision_status": precision_status,
                "learned_form_tag": HEURISTIC_TAG,
                "score_claim": False,
                "actuation": "NONE",
            }
        )
    pair_rows.sort(key=lambda row: (-row["prediction"], row["pair_id"]))
    for rank, row in enumerate(pair_rows, 1):
        row["oof_rank"] = rank
        if row["fisher_95_interval"] is None:
            row["pair_order_status"] = "UNRANKED_PRECISION_OWED"
        elif rank > 1:
            previous = pair_rows[rank - 2]
            previous_interval = previous["fisher_95_interval"]
            if (
                previous_interval is None
                or row["fisher_95_interval"][1] >= previous_interval[0]
            ):
                row["pair_order_status"] = "TIED_OVERLAPPING_OR_MISSING_INTERVAL"
            else:
                row["pair_order_status"] = "ORDERED_NONOVERLAPPING_INTERVAL"
        else:
            row["pair_order_status"] = "LEADER_INTERVAL_ESTIMATE"

    self_checks = [
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "pontryagin_bellman_adjacent_lambda_residual",
            "status": "AWAITING_J8F_MEASUREMENT",
            "value": None,
            "reason": (
                "ordered adjacent J8F costates and realized transition terms are "
                "absent; OOF pair predictions are not a Bellman trajectory"
            ),
        },
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "rd1_organ_dual_consistency",
            "status": (
                "AWAITING_NON_NULL_MATCHED_RD1_DUALS"
                if not actionable_rd1_duals
                else "BLOCKED_MATCHED_PAIR_DIMENSION_CROSSWALK_OWED"
            ),
            "value": None,
            "reason": (
                f"hash-verified RD1 campaign source has {len(actionable_rd1_duals)} "
                f"actionable dimension prices across {len(rd1_duals)} typed rows; "
                "null is not coerced to zero and no comparison band is invented"
            ),
        },
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "wallace_mml_pair_precision",
            "status": (
                "COMPLETE" if precision_count == PAIR_COUNT else "PARTIAL_TYPED"
            ),
            "value": {
                "pair_intervals": precision_count,
                "required": PAIR_COUNT,
            },
            "reason": (
                "only direct pair-indexed positive MS4D Fisher blocks authorize "
                "a pair precision interval"
            ),
        },
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "compression_progress_per_effort",
            "status": "AWAITING_J8F_MEASUREMENT",
            "value": None,
            "reason": (
                "J8F delta_S_per_wall_clock_hour is absent; no proxy effort "
                "currency is promoted"
            ),
        },
    ]
    precision_complete = precision_count == PAIR_COUNT
    explanation = _falling_rule_explanation(
        admission_ndcg=float(winner["metrics"]["ndcg_at_4"]),
        precision_complete=precision_complete,
        j8f_verdict_count=0,
    )
    blocker_ids = [
        "BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY",
        "BLOCKED_PAIR_LEVEL_MS4D_FISHER_PRECISION_"
        f"{PAIR_COUNT - precision_count}",
    ]
    if not admission_passed:
        blocker_ids.append("BLOCKED_LAMBDA_RANKER_HELDOUT_NDCG_ADMISSION")

    source_lineage = {
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": sha256_file(preregistration),
            "bytes": preregistration.stat().st_size,
        },
        "g3_receipt": {
            "path": sources["g3"]["path"],
            "sha256": sources["g3"]["sha256"],
        },
        "g3_bulk_atlas": bulk,
        "ev1_receipt": {
            "path": sources["ev1"]["path"],
            "sha256": sources["ev1"]["sha256"],
        },
        "rd1_dual_authority": {
            "path": RD1_PATH,
            "sha256": sha256_file(rd1_path),
            "schema": RD1_SCHEMA,
            "typed_dual_rows": len(rd1_duals),
            "actionable_dual_rows": len(actionable_rd1_duals),
        },
        "margin_fisher_oracle": _oracle_meta(
            margin_row,
            surface_counts={
                "bucket_rows": len(margin_data["rows"]),
                "direct_blocks": len(margin_data["direct_blocks"]),
                "direct_pair_ids": len(_direct_fisher_by_pair(margin_data)),
            },
        ),
        "pose_tube_oracle": _oracle_meta(
            pose_row,
            surface_counts={
                "pair_rows": len(pose_data["rows"]),
                "converged_pair_rows": sum(
                    bool(row["converged"]) for row in pose_data["rows"]
                ),
            },
        ),
        "stationarity_oracle": _oracle_meta(
            stationarity_row,
            surface_counts={
                "strata": len(stationarity),
                "temporal_classes": len(stationarity["all"]["classes"]),
                "all_flip_mass": int(stationarity["all"]["flip_mass"]),
            },
        ),
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "maturity": "_dev",
        "research_only": True,
        "execution_allowed": False,
        "actuation": "NONE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
        "evidence_axis": EVIDENCE_AXIS,
        "population": {
            "required_pairs": PAIR_COUNT,
            "joined_pairs": len(feature_rows),
            "fold_rule": (
                'SHA256("ddm-co3-n600-v1:" + decimal(pair_id))[0:8] mod 5'
            ),
            "fold_counts": fold_counts,
            "heldout_unit": "source_pair_id",
            "shared_v19_rate_bytes_in_pair_target": False,
        },
        "model_race": list(candidate_outputs.values()),
        "selected_model": {
            "candidate_id": selected_id,
            "learned_form_tag": HEURISTIC_TAG,
            "selection_rule": (
                "max held-out NDCG@4; tie held-out Spearman; tie lower complexity"
            ),
            "metrics": winner["metrics"],
        },
        "admission_gate": {
            "metric": "concatenated_pair_out_of_fold_ndcg_at_4",
            "threshold": ADMISSION_NDCG_AT_4,
            "observed": float(winner["metrics"]["ndcg_at_4"]),
            "passed": admission_passed,
            "duty_ranking_upgrade_eligible": admission_passed,
            "pair_order_precision_complete": precision_complete,
        },
        "ranking_error_slices": slices,
        "innovations": innovations,
        "pair_rankings": pair_rows,
        "self_checks": self_checks,
        "rudin_explanation": explanation,
        "bandit_allocation": {
            "status": "DESIGN_ONLY_NOT_ACTUATED",
            "design": (
                "future regret-bounded duty allocation uses measured compression "
                "progress per effort plus exploration bonus for never-fired levers"
            ),
            "required_evidence": (
                "J8F realized delta_S_per_wall_clock_hour and fired-duty history"
            ),
        },
        "source_lineage": source_lineage,
        "blocker_ids": blocker_ids,
        "j8f_blocker_preserved": True,
        "verdict_scope": (
            "INSTANCE:V19_N600_EXACT_RECEIVER_REPLAY_X_G3_ATLAS_X_FRESH_"
            "MS4D_G4_ORACLE_ROWS; no family negative and no contest score"
        ),
    }
    receipt["content_sha256"] = _payload_sha256(receipt)
    return receipt


def write_receipt_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic canonical JSON through an atomic replacement."""

    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, destination)


__all__ = [
    "ADMISSION_NDCG_AT_4",
    "ALPHA_GRID",
    "BASE_FEATURES",
    "CANDIDATES",
    "EVIDENCE_AXIS",
    "G4_FEATURES",
    "HEURISTIC_TAG",
    "MS4D_FEATURES",
    "PAIR_COUNT",
    "PAIR_ROW_SCHEMA",
    "PREREGISTRATION_PATH",
    "RD1_PATH",
    "RD1_SCHEMA",
    "RUN_ID",
    "SCHEMA",
    "SELF_CHECK_SCHEMA",
    "SLICE_SCHEMA",
    "LambdaRankerError",
    "build_n600_lambda_ranker_receipt",
    "ndcg_at_k",
    "spearman_rho",
    "write_receipt_atomic",
]
