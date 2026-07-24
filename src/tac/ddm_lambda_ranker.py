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
DECIDE_ROW_SCHEMA = "ddm_lambda_ranker_decide_row.v1"
RUN_ID = "ddm_co4_road_local_and_precision_20260724"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
HEURISTIC_TAG = "[advisory-heuristic]"
ADMISSION_NDCG_AT_4 = 0.75
ROAD_ADMISSION_NDCG_AT_4 = 0.60
MIXTURE_CLOSE_NDCG = 0.02
PAIR_COUNT = 600
MIN_EXPERT_TRAIN_PAIRS = 8
ALPHA_GRID: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
PREREGISTRATION_PATH = ".omx/research/ddm_co4_road_local_and_precision_preregistration_20260724.md"
CO3_RECEIPT_PATH = ".omx/research/ddm_co3_lambda_refit_full_join_20260724/ddm_co3_lambda_refit_full_join_receipt.json"
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
ROAD_FEATURES: tuple[str, ...] = (
    *MS4D_FEATURES,
    "road_fisher_trace_log",
    "road_fisher_lambda_max_log",
    "road_fisher_effective_rank",
    "road_fisher_boundary_trace_share",
    "road_support_log",
    "g3_hard_rank_percentile",
    "g3_hard_rank_reciprocal_log",
    "road_g4_transient_fraction",
    "road_g4_static_in_image_fraction",
    "road_boundary_support_fraction",
    "road_interior_support_fraction",
    "road_gap_x_fisher_trace_log",
)
CANDIDATES: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("factorized_refit", BASE_FEATURES, 1),
    ("factorized_ms4d_interactions", MS4D_FEATURES, 2),
    ("g4_regime_conditional", G4_FEATURES, 3),
)
ROAD_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("global_road_conditional", 1),
    ("g3_stratum_experts", 2),
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
    digest = hashlib.sha256(f"ddm-co3-n600-v1:{pair_id}".encode("ascii")).hexdigest()
    return int(digest[:8], 16) % 5


def _inner_fold_id(pair_id: int, outer_fold: int) -> int:
    digest = hashlib.sha256(f"ddm-co3-inner-v1:{outer_fold}:{pair_id}".encode("ascii")).hexdigest()
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
        [[_feature_value(rows[index], name, context) for name in feature_names] for index in indices],
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
    # ``einsum`` avoids spurious floating-status warnings emitted by the
    # Accelerate-backed matmul in the pinned macOS NumPy build for these small
    # well-scaled matrices.
    gram = np.einsum("ni,nj->ij", z, z)
    rhs = np.einsum("ni,n->i", z, centered)
    regularizer = float(alpha) * np.eye(z.shape[1], dtype=np.float64)
    try:
        weights = np.linalg.solve(gram + regularizer, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram + regularizer) @ rhs
    return {
        "alpha": float(alpha),
        "feature_names": list(feature_names),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "standardized_weights": weights.tolist(),
        "target_intercept": target_mean,
        "feature_context": {key: [float(value) for value in values] for key, values in context.items()},
    }


def _predict_ridge(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["standardized_weights"], dtype=np.float64)
    standardized = (x - mean) / scale
    return float(model["target_intercept"]) + np.einsum(
        "ij,j->i",
        standardized,
        weights,
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
                if _inner_fold_id(int(rows[index]["pair_id"]), outer_fold) != inner_fold
            ]
            inner_test_positions = [
                position
                for position, index in enumerate(train_indices)
                if _inner_fold_id(int(rows[index]["pair_id"]), outer_fold) == inner_fold
            ]
            inner_test = [train_indices[position] for position in inner_test_positions]
            if not inner_train or not inner_test:
                raise LambdaRankerError(f"outer fold {outer_fold}: empty deterministic inner split")
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
            -math.inf if row["inner_heldout_spearman_rho"] is None else row["inner_heldout_spearman_rho"],
            -math.inf if row["inner_heldout_ndcg_at_4"] is None else row["inner_heldout_ndcg_at_4"],
            -row["alpha"],
        ),
    )
    return float(selected["alpha"]), {
        "selection_metric": "inner_three_fold_spearman_then_ndcg_at_4",
        "selected": selected,
        "grid": results,
    }


def _fit_predict_stratum_experts(
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    feature_names: Sequence[str],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit target-free G3-stratum experts with a shared global fallback."""

    context = _feature_context(rows, train_indices)
    y_train = np.asarray(
        [_finite(rows[index]["target"], "target") for index in train_indices],
        dtype=np.float64,
    )
    x_train = _matrix(rows, train_indices, feature_names, context)
    global_model = _fit_ridge(
        x_train,
        y_train,
        alpha=alpha,
        feature_names=feature_names,
        context=context,
    )
    experts: dict[str, dict[str, Any]] = {}
    train_by_stratum: dict[str, list[int]] = defaultdict(list)
    for index in train_indices:
        train_by_stratum[str(rows[index]["g3_dominant_stratum"])].append(index)
    for stratum, indices in sorted(train_by_stratum.items()):
        if len(indices) < MIN_EXPERT_TRAIN_PAIRS:
            continue
        experts[stratum] = _fit_ridge(
            _matrix(rows, indices, feature_names, context),
            np.asarray(
                [_finite(rows[index]["target"], "target") for index in indices],
                dtype=np.float64,
            ),
            alpha=alpha,
            feature_names=feature_names,
            context=context,
        )

    def predict(indices: Sequence[int]) -> np.ndarray:
        values = np.zeros(len(indices), dtype=np.float64)
        by_model: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, index in enumerate(indices):
            stratum = str(rows[index]["g3_dominant_stratum"])
            model_id = stratum if stratum in experts else "__GLOBAL_FALLBACK__"
            by_model[model_id].append((position, index))
        for model_id, positioned_indices in by_model.items():
            model = global_model if model_id == "__GLOBAL_FALLBACK__" else experts[model_id]
            positions = [position for position, _index in positioned_indices]
            indices_for_model = [index for _position, index in positioned_indices]
            values[positions] = _predict_ridge(
                model,
                _matrix(rows, indices_for_model, feature_names, context),
            )
        return values

    return (
        predict(test_indices),
        predict(train_indices),
        {
            "router": "target_free_g3_dominant_pre_outcome_class_flip_stratum",
            "minimum_expert_train_pairs": MIN_EXPERT_TRAIN_PAIRS,
            "global_fallback_model": global_model,
            "expert_models": experts,
            "expert_train_counts": {stratum: len(indices) for stratum, indices in sorted(train_by_stratum.items())},
        },
    )


def _select_alpha_stratum_experts(
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    *,
    feature_names: Sequence[str],
    outer_fold: int,
) -> tuple[float, dict[str, Any]]:
    results: list[dict[str, Any]] = []
    targets = [_finite(rows[index]["target"], "target") for index in train_indices]
    for alpha in ALPHA_GRID:
        predictions = np.zeros(len(train_indices), dtype=np.float64)
        for inner_fold in range(3):
            inner_train = [
                index
                for index in train_indices
                if _inner_fold_id(int(rows[index]["pair_id"]), outer_fold) != inner_fold
            ]
            inner_test_positions = [
                position
                for position, index in enumerate(train_indices)
                if _inner_fold_id(int(rows[index]["pair_id"]), outer_fold) == inner_fold
            ]
            inner_test = [train_indices[position] for position in inner_test_positions]
            if not inner_train or not inner_test:
                raise LambdaRankerError(f"outer fold {outer_fold}: empty deterministic expert split")
            predicted, _train_predicted, _model = _fit_predict_stratum_experts(
                rows,
                inner_train,
                inner_test,
                feature_names=feature_names,
                alpha=alpha,
            )
            predictions[inner_test_positions] = predicted
        rho = spearman_rho(predictions.tolist(), targets)
        ndcg = ndcg_at_k(predictions.tolist(), targets, k=4)
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
            -math.inf if row["inner_heldout_spearman_rho"] is None else row["inner_heldout_spearman_rho"],
            -math.inf if row["inner_heldout_ndcg_at_4"] is None else row["inner_heldout_ndcg_at_4"],
            -row["alpha"],
        ),
    )
    return float(selected["alpha"]), {
        "selection_metric": "inner_three_fold_spearman_then_ndcg_at_4",
        "selected": selected,
        "grid": results,
        "router": "target_free_g3_dominant_pre_outcome_class_flip_stratum",
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
            int(all_classes[name]["flip_mass"]) - int(boundary_classes[name]["flip_mass"]),
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
    closures = {name: int(row["errors_before"]) - int(row["errors_after"]) for name, row in per_stratum.items()}
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


def _propagated_fisher_by_pair(
    margin_data: Mapping[str, Any],
    assignment_data: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Propagate bucket Grams through exact PF2 bucket-to-pair support.

    The propagation is a support-weighted additive information approximation,
    not a direct pair measurement.  Validation is deliberately strict so a
    missing or inconsistent foreign key fails closed instead of manufacturing
    precision.
    """

    assignment_rows = {str(row["bucket_id"]): row for row in assignment_data["rows"]}
    if len(assignment_rows) != 1200 or len(assignment_data["rows"]) != 1200 or len(margin_data["rows"]) != 1200:
        raise LambdaRankerError("PF2/MS4D propagation requires exact 1,200 buckets")

    grouped: dict[int, dict[str, Any]] = {
        pair_id: {
            "gram": np.zeros((4, 4), dtype=np.float64),
            "road_gram": np.zeros((4, 4), dtype=np.float64),
            "road_boundary_gram": np.zeros((4, 4), dtype=np.float64),
            "contribution_traces": [],
            "bucket_count": 0,
            "support_count": 0,
            "road_support_count": 0,
            "road_boundary_support_count": 0,
            "road_interior_support_count": 0,
            "road_transient_support_count": 0,
            "road_static_in_image_support_count": 0,
        }
        for pair_id in range(PAIR_COUNT)
    }
    seen: set[str] = set()
    for row in margin_data["rows"]:
        bucket_id = str(row["bucket_id"])
        if bucket_id in seen or bucket_id not in assignment_rows:
            raise LambdaRankerError(f"bucket {bucket_id}: duplicate or missing PF2 assignment")
        seen.add(bucket_id)
        assignment = assignment_rows[bucket_id]
        if int(assignment["event_count"]) != int(row["event_count"]):
            raise LambdaRankerError(f"bucket {bucket_id}: event-count mismatch")
        gram = np.asarray(row["margin_fisher_gram"], dtype=np.float64)
        if gram.shape != (4, 4) or not np.isfinite(gram).all():
            raise LambdaRankerError(f"bucket {bucket_id}: invalid Fisher Gram")
        gram = 0.5 * (gram + gram.T)
        eigenvalues = np.linalg.eigvalsh(gram)
        if float(eigenvalues.min()) < -1e-10:
            raise LambdaRankerError(f"bucket {bucket_id}: Fisher Gram is not PSD")
        if float(eigenvalues.min()) < 0.0:
            vectors = np.linalg.eigh(gram)[1]
            gram = vectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ vectors.T

        event_count = int(row["event_count"])
        support_counts = row.get("pair_support_counts")
        assignment_pairs = sorted(int(value) for value in assignment["pair_ids"])
        if event_count == 0:
            if support_counts not in (None, []) or assignment_pairs:
                raise LambdaRankerError(f"bucket {bucket_id}: nonempty support on zero-event bucket")
            continue
        if not isinstance(support_counts, list) or len(support_counts) != PAIR_COUNT:
            raise LambdaRankerError(f"bucket {bucket_id}: missing exact N600 pair support counts")
        counts = np.asarray(support_counts, dtype=np.int64)
        if np.any(counts < 0) or int(counts.sum()) != event_count:
            raise LambdaRankerError(f"bucket {bucket_id}: pair support does not conserve event mass")
        positive_pairs = np.flatnonzero(counts > 0).tolist()
        if positive_pairs != assignment_pairs:
            raise LambdaRankerError(f"bucket {bucket_id}: MS4D support and PF2 assignment disagree")

        is_road = "Road" in str(row["class_pair"]).split("--")
        is_boundary = str(row["class_stratum"]) == "boundary"
        is_transient = str(row["g4_temporal_class"]) == "TRANSIENT"
        is_static = str(row["g4_temporal_class"]) == "STATIC_IN_IMAGE"
        for pair_id in positive_pairs:
            support = int(counts[pair_id])
            share = support / event_count
            contribution = share * gram
            state = grouped[pair_id]
            state["gram"] += contribution
            state["contribution_traces"].append(float(np.trace(contribution)))
            state["bucket_count"] += 1
            state["support_count"] += support
            if is_road:
                state["road_gram"] += contribution
                if is_boundary:
                    state["road_boundary_gram"] += contribution
                state["road_support_count"] += support
                if is_boundary:
                    state["road_boundary_support_count"] += support
                else:
                    state["road_interior_support_count"] += support
                if is_transient:
                    state["road_transient_support_count"] += support
                if is_static:
                    state["road_static_in_image_support_count"] += support

    if seen != set(assignment_rows):
        raise LambdaRankerError("MS4D and PF2 bucket vocabularies are not identical")

    result: dict[int, dict[str, Any]] = {}
    for pair_id, state in grouped.items():
        traces = np.asarray(state.pop("contribution_traces"), dtype=np.float64)
        gram = state.pop("gram")
        road_gram = state.pop("road_gram")
        road_boundary_gram = state.pop("road_boundary_gram")
        design_effect = 1.0
        if traces.size >= 2 and float(traces.mean()) > 0.0:
            cv = float(traces.std(ddof=1) / traces.mean())
            design_effect += cv * cv
        road_eigenvalues = np.linalg.eigvalsh(road_gram)
        road_trace = float(np.trace(road_gram))
        road_lambda_max = float(max(0.0, road_eigenvalues[-1]))
        road_effective_rank = (
            0.0
            if road_trace <= 0.0
            else float(road_trace * road_trace / max(1e-30, float(np.square(road_eigenvalues).sum())))
        )
        result[pair_id] = {
            **state,
            "fisher_gram": gram.tolist(),
            "fisher_trace": float(np.trace(gram)),
            "design_effect": design_effect,
            "road_fisher_gram": road_gram.tolist(),
            "road_fisher_trace": road_trace,
            "road_fisher_lambda_max": road_lambda_max,
            "road_fisher_effective_rank": road_effective_rank,
            "road_boundary_fisher_trace": float(np.trace(road_boundary_gram)),
        }
    return result


def _build_feature_rows(
    *,
    atlas: Mapping[int, Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    margin_data: Mapping[str, Any],
    assignment_data: Mapping[str, Any],
    pose_data: Mapping[str, Any],
    stationarity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pose_by_pair = {int(row["pair_id"]): row for row in pose_data["rows"]}
    fisher_by_pair = _direct_fisher_by_pair(margin_data)
    propagated_by_pair = _propagated_fisher_by_pair(
        margin_data,
        assignment_data,
    )
    rows: list[dict[str, Any]] = []
    for measured in evidence_rows:
        pair_id = int(measured["source_pair_id"])
        if pair_id not in atlas or pair_id not in pose_by_pair:
            raise LambdaRankerError(f"pair {pair_id}: missing G3 or Pose6 row")
        g3 = atlas[pair_id]
        g3_score_rank = int(g3["score_rank"])
        if not 1 <= g3_score_rank <= PAIR_COUNT:
            raise LambdaRankerError(f"pair {pair_id}: invalid G3 hard rank")
        signal = g3["costate_signal"]
        segmentation = g3["segmentation"]
        geometry = g3["evaluator_response_geometry"]["joint_cone_summary_diagnostic_only"]
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
        boundary_fraction = boundary_count / max(1, boundary_count + interior_count)
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
        propagated = propagated_by_pair[pair_id]
        road_support = int(propagated["road_support_count"])
        road_trace = float(propagated["road_fisher_trace"])
        road_total_trace = max(1e-30, road_trace)
        road_boundary_gram_trace = float(propagated["road_boundary_fisher_trace"])
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
            "support_x_stationarity": visibility * g4_mix["STATIC_IN_IMAGE"],
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
            "road_fisher_trace_log": math.log1p(max(0.0, road_trace)),
            "road_fisher_lambda_max_log": math.log1p(max(0.0, float(propagated["road_fisher_lambda_max"]))),
            "road_fisher_effective_rank": float(propagated["road_fisher_effective_rank"]),
            "road_fisher_boundary_trace_share": (
                road_boundary_gram_trace / road_total_trace if road_trace > 0.0 else 0.0
            ),
            "road_support_log": math.log1p(road_support),
            "g3_hard_rank_percentile": ((PAIR_COUNT - g3_score_rank) / (PAIR_COUNT - 1)),
            "g3_hard_rank_reciprocal_log": math.log1p(PAIR_COUNT / g3_score_rank),
            "road_g4_transient_fraction": (
                int(propagated["road_transient_support_count"]) / road_support if road_support > 0 else 0.0
            ),
            "road_g4_static_in_image_fraction": (
                int(propagated["road_static_in_image_support_count"]) / road_support if road_support > 0 else 0.0
            ),
            "road_boundary_support_fraction": (
                int(propagated["road_boundary_support_count"]) / road_support if road_support > 0 else 0.0
            ),
            "road_interior_support_fraction": (
                int(propagated["road_interior_support_count"]) / road_support if road_support > 0 else 0.0
            ),
            "road_gap_x_fisher_trace_log": gap * math.log1p(max(0.0, road_trace)),
        }
        g3_flip_counts = {name: int(value) for name, value in segmentation["class_flip_counts"].items()}
        if sum(g3_flip_counts.values()) != int(segmentation["flip_count"]):
            raise LambdaRankerError(f"pair {pair_id}: G3 class-flip mass mismatch")
        g3_dominant_stratum = max(
            g3_flip_counts,
            key=lambda name: (g3_flip_counts[name], name),
        )
        rows.append(
            {
                "pair_id": pair_id,
                "outer_fold": _fold_id(pair_id),
                "target": max(0.0, -distortion_delta),
                "features": features,
                "margin": margin,
                "hardness": gap,
                "dominant_stratum": _dominant_stratum(measured["per_stratum"]),
                "g3_dominant_stratum": g3_dominant_stratum,
                "g3_score_rank": g3_score_rank,
                "g4_temporal_class": max(
                    g4_mix,
                    key=lambda name: (g4_mix[name], name),
                ),
                "g4_pair_class_status": ("DERIVED_DOMINANT_FROM_G3_BOUNDARY_INTERIOR_COUNTS_X_G4_AGGREGATE_CLASS_MASS"),
                "g4_mixture": g4_mix,
                "direct_fisher_trace": (None if direct is None else direct["fisher_trace"]),
                "direct_fisher_block_count": (0 if direct is None else int(direct["block_count"])),
                "propagated_fisher_trace": float(propagated["fisher_trace"]),
                "propagated_fisher_gram": propagated["fisher_gram"],
                "precision_design_effect": float(propagated["design_effect"]),
                "precision_bucket_count": int(propagated["bucket_count"]),
                "precision_support_count": int(propagated["support_count"]),
            }
        )
    rows.sort(key=lambda row: int(row["pair_id"]))
    if [row["pair_id"] for row in rows] != list(range(PAIR_COUNT)):
        raise LambdaRankerError("feature join must contain pair IDs 0..599 exactly once")
    if sorted(int(row["g3_score_rank"]) for row in rows) != list(range(1, PAIR_COUNT + 1)):
        raise LambdaRankerError("G3 hard ranks must be the exact permutation 1..600")
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


def _road_slice_metrics(
    feature_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
) -> dict[str, Any]:
    indices = [index for index, row in enumerate(feature_rows) if str(row["dominant_stratum"]) == "Road"]
    if not indices:
        raise LambdaRankerError("Road held-out evaluation slice is empty")
    predicted = [float(predictions[index]) for index in indices]
    targets = [_finite(feature_rows[index]["target"], "target") for index in indices]
    return {
        "heldout_only": True,
        "router_forbidden": True,
        "slice_source": "EV1_REALIZED_CLOSURE_STRATUM_EVALUATION_ONLY",
        "n_pairs": len(indices),
        "spearman_rho": spearman_rho(predicted, targets),
        "ndcg_at_4": ndcg_at_k(predicted, targets, k=4),
    }


def _slice_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = [_finite(row["target"], "target") for row in feature_rows]
    margin_thresholds = _quantile_thresholds([_finite(row["margin"], "margin") for row in feature_rows])
    hardness_thresholds = _quantile_thresholds([_finite(row["hardness"], "hardness") for row in feature_rows])
    dimensions: dict[str, list[str]] = {
        "stratum": [str(row["dominant_stratum"]) for row in feature_rows],
        "g4_class": [str(row["g4_temporal_class"]) for row in feature_rows],
        "margin_decile": [
            f"D{int(_decile(_finite(row['margin'], 'margin'), margin_thresholds)):02d}" for row in feature_rows
        ],
        "pair_hardness_decile": [
            f"D{int(_decile(_finite(row['hardness'], 'hardness'), hardness_thresholds)):02d}" for row in feature_rows
        ],
    }
    out: list[dict[str, Any]] = []
    for dimension, labels in dimensions.items():
        for label in sorted(set(labels)):
            indices = [index for index, value in enumerate(labels) if value == label]
            pred = [float(predictions[index]) for index in indices]
            actual = [targets[index] for index in indices]
            innovations = [
                actual_value - predicted_value for actual_value, predicted_value in zip(actual, pred, strict=True)
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
                    "verdict_scope": ("INSTANCE:V19_N600_EXACT_RECEIVER_REPLAY_X_G3_X_FRESH_ORACLE_ROWS"),
                }
            )
    innovations = np.asarray(targets, dtype=np.float64) - np.asarray(predictions, dtype=np.float64)
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
                    PredicateRef.parse(f"admission_ndcg_at_4 >= {ADMISSION_NDCG_AT_4}"),
                    PredicateRef.parse("precision_complete >= 1"),
                    PredicateRef.parse("j8f_verdict_count >= 1"),
                ),
                predicted_score_low=ADMISSION_NDCG_AT_4,
                predicted_score_high=1.0,
            ),
            FallingRule(
                name="heldout_ranker_admitted_measurement_duties_owed",
                rule_id="ddm_co3_admitted_but_blocked",
                predicates=(PredicateRef.parse(f"admission_ndcg_at_4 >= {ADMISSION_NDCG_AT_4}"),),
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
        "surface": ("tac.autopilot_rudin_daubechies.falling_rule_list.FallingRuleList"),
        "rule_chain": chain.explain(),
        "metadata": metadata,
        "gosdt_slim_disposition": (
            "REASONED_EXCLUDE: dispatch GOSDT action bands and fixed Taylor "
            "ProxyPanel features are not the N600 lambda feature schema; the "
            "generic canonical falling-rule metadata path is lossless."
        ),
    }


def _typed_decide_row(
    *,
    decide_id: str,
    status: str,
    condition_name: str,
    condition_value: bool,
    reason: str,
    authority: str,
) -> dict[str, Any]:
    rules = FallingRuleList(
        rules=[
            FallingRule(
                name=f"{decide_id}_condition_satisfied",
                rule_id=f"{decide_id}_satisfied",
                predicates=(PredicateRef.parse(f"{condition_name} >= 1"),),
                predicted_score_low=1.0,
                predicted_score_high=1.0,
            )
        ],
        default_band_low=0.0,
        default_band_high=0.0,
    )
    metadata = {condition_name: int(condition_value)}
    chain = rules.evaluate(ProxyPanel(candidate_id=decide_id), metadata)
    return {
        "schema": DECIDE_ROW_SCHEMA,
        "decide_id": decide_id,
        "status": status,
        "reason": reason,
        "authority": authority,
        "actuation": "NONE",
        "score_claim": False,
        "rudin_explanation": {
            "status": "REUSED_CANONICAL_FALLING_RULE_LIST",
            "surface": ("tac.autopilot_rudin_daubechies.falling_rule_list.FallingRuleList"),
            "rule_chain": chain.explain(),
            "metadata": metadata,
            "gosdt_slim_disposition": (
                "REASONED_EXCLUDE: no action-band or Taylor ProxyPanel feature "
                "is needed for this typed local advisory decision."
            ),
        },
    }


def _unsound_claim_audit(root: Path) -> dict[str, Any]:
    patterns = (
        "77" + "x",
        "2.71" + "x",
        "params^" + "-0.71",
        "label-noise" + " ceiling",
        "label noise" + " ceiling",
    )
    paths: set[Path] = {root / "src/tac/ddm_campaign_costate.py"}
    for glob_pattern in (
        ".omx/research/ddm_co2*",
        ".omx/research/ddm_co3*",
        ".omx/research/codex_findings_ddm_co2*",
        ".omx/research/codex_findings_ddm_co3*",
    ):
        for candidate in root.glob(glob_pattern):
            if candidate.is_file():
                paths.add(candidate)
            elif candidate.is_dir():
                paths.update(
                    path for path in candidate.rglob("*") if path.is_file() and path.suffix in {".json", ".md"}
                )
    matches: list[dict[str, Any]] = []
    scanned: list[str] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        scanned.append(relative)
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            line_numbers = [
                line_number for line_number, line in enumerate(text.splitlines(), 1) if pattern.lower() in line.lower()
            ]
            if line_numbers:
                matches.append(
                    {
                        "path": relative,
                        "pattern": pattern,
                        "line_numbers": line_numbers,
                    }
                )
    if matches:
        raise LambdaRankerError("unsound scaling or label-noise claims remain in CO2/CO3 scope")
    return {
        "status": "CLEAN_NO_UNSOUND_CLAIMS_IN_CO2_CO3_INPUT_SCOPE",
        "patterns": list(patterns),
        "files_scanned": len(scanned),
        "matches": matches,
        "purged_rows": 0,
        "verdict_scope": "CO2/CO3 ranker and campaign inputs consumed by CO4",
    }


def build_n600_lambda_ranker_receipt(repo_root: Path) -> dict[str, Any]:
    """Fit the preregistered race and return one deterministic receipt."""

    root = repo_root.expanduser().resolve(strict=True)
    preregistration = root / PREREGISTRATION_PATH
    if not preregistration.is_file():
        raise LambdaRankerError("held-out selection preregistration is missing")
    co3_receipt_path = root / CO3_RECEIPT_PATH
    if not co3_receipt_path.is_file():
        raise LambdaRankerError("CO3 historical comparison receipt is missing")
    co3_receipt = json.loads(co3_receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(co3_receipt, dict)
        or co3_receipt.get("schema") != SCHEMA
        or co3_receipt.get("score_claim") is not False
        or co3_receipt.get("actuation") != "NONE"
    ):
        raise LambdaRankerError("CO3 historical receipt failed schema/firewall")
    soundness_audit = _unsound_claim_audit(root)
    rd1_path = root / RD1_PATH
    if not rd1_path.is_file():
        raise LambdaRankerError("RD1 dual-consistency authority is missing")
    rd1 = json.loads(rd1_path.read_text(encoding="utf-8"))
    if not isinstance(rd1, dict) or rd1.get("schema") != RD1_SCHEMA or rd1.get("score_claim") is not False:
        raise LambdaRankerError("RD1 dual-consistency authority failed schema/firewall")
    rd1_duals = list(rd1.get("duals") or [])
    actionable_rd1_duals = [row for row in rd1_duals if row.get("actionable_for_train_decision") is True]
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
    assignment_row = oracle.bucket_assignments()
    pose_row = oracle.pose_reference_and_tube()
    stationarity_row = oracle.stationarity_maps()
    margin_value = margin_row.require_value()
    assignment_data = assignment_row.require_value()
    pose_value = pose_row.require_value()
    stationarity_value = stationarity_row.require_value()
    margin_data = margin_value["data"]
    pose_data = pose_value["data"]
    stationarity = stationarity_value["stationarity_decomposition"]
    feature_rows = _build_feature_rows(
        atlas=atlas,
        evidence_rows=evidence_rows,
        margin_data=margin_data,
        assignment_data=assignment_data,
        pose_data=pose_data,
        stationarity=stationarity,
    )
    co3_pair_rows = list(co3_receipt.get("pair_rankings") or [])
    if len(co3_pair_rows) != PAIR_COUNT or {int(row["pair_id"]) for row in co3_pair_rows} != set(range(PAIR_COUNT)):
        raise LambdaRankerError("CO3 historical receipt lacks exact N600 predictions")
    co3_predictions = np.zeros(PAIR_COUNT, dtype=np.float64)
    feature_target_by_pair = {int(row["pair_id"]): _finite(row["target"], "target") for row in feature_rows}
    for row in co3_pair_rows:
        pair_id = int(row["pair_id"])
        co3_predictions[pair_id] = _finite(row["prediction"], "co3 prediction")
        if not math.isclose(
            _finite(row["target"], "co3 target"),
            feature_target_by_pair[pair_id],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise LambdaRankerError(f"pair {pair_id}: CO3 and fresh CO4 targets disagree")
    targets = [_finite(row["target"], "target") for row in feature_rows]
    fold_counts = {str(fold): sum(int(row["outer_fold"]) == fold for row in feature_rows) for fold in range(5)}
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
            train_indices = [index for index, row in enumerate(feature_rows) if int(row["outer_fold"]) != outer_fold]
            test_indices = [index for index, row in enumerate(feature_rows) if int(row["outer_fold"]) == outer_fold]
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
        "preregistered_trigger": (f"all candidates 1-3 held-out NDCG@4 < {ADMISSION_NDCG_AT_4}"),
        "observed_best_candidates_1_3_ndcg_at_4": pre_gb_best["metrics"]["ndcg_at_4"],
        "verdict_scope": (
            "no negative on monotone/GB family; optimal bounded-stump form was "
            "not triggered by the preregistered plateau rule"
        ),
    }

    mixture_predictions = np.zeros(PAIR_COUNT, dtype=np.float64)
    mixture_folds: list[dict[str, Any]] = []
    for outer_fold in range(5):
        test_indices = [index for index, row in enumerate(feature_rows) if int(row["outer_fold"]) == outer_fold]
        inner_rank = sorted(
            (
                {
                    "candidate_id": candidate_id,
                    **candidate_outputs[candidate_id]["outer_fold_models"][outer_fold]["inner_selection"]["selected"],
                }
                for candidate_id, _features, _complexity in CANDIDATES
            ),
            key=lambda row: (
                -(-math.inf if row["inner_heldout_ndcg_at_4"] is None else row["inner_heldout_ndcg_at_4"]),
                -(-math.inf if row["inner_heldout_spearman_rho"] is None else row["inner_heldout_spearman_rho"]),
                candidate_complexity[row["candidate_id"]],
            ),
        )
        best, second = inner_rank[:2]
        close = (
            best["inner_heldout_ndcg_at_4"] is not None
            and second["inner_heldout_ndcg_at_4"] is not None
            and best["inner_heldout_ndcg_at_4"] - second["inner_heldout_ndcg_at_4"] <= MIXTURE_CLOSE_NDCG
        )
        members = [best["candidate_id"], second["candidate_id"]] if close else [best["candidate_id"]]
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

    historical_candidate_ids = (
        *(candidate_id for candidate_id, _features, _complexity in CANDIDATES),
        "close_form_mixture",
    )
    historical_winner = max(
        (candidate_outputs[candidate_id] for candidate_id in historical_candidate_ids),
        key=lambda row: (
            row["metrics"]["ndcg_at_4"],
            row["metrics"]["spearman_rho"],
            -candidate_complexity[row["candidate_id"]],
        ),
    )

    for candidate_id, complexity in ROAD_CANDIDATES:
        predictions = np.zeros(PAIR_COUNT, dtype=np.float64)
        fold_models: list[dict[str, Any]] = []
        for outer_fold in range(5):
            train_indices = [index for index, row in enumerate(feature_rows) if int(row["outer_fold"]) != outer_fold]
            test_indices = [index for index, row in enumerate(feature_rows) if int(row["outer_fold"]) == outer_fold]
            if candidate_id == "global_road_conditional":
                alpha, inner = _select_alpha(
                    feature_rows,
                    train_indices,
                    feature_names=ROAD_FEATURES,
                    outer_fold=outer_fold,
                )
                test_prediction, _train_prediction, model = _fit_predict(
                    feature_rows,
                    train_indices,
                    test_indices,
                    feature_names=ROAD_FEATURES,
                    alpha=alpha,
                )
                router = "GLOBAL_MODEL_NO_ROUTER"
            else:
                alpha, inner = _select_alpha_stratum_experts(
                    feature_rows,
                    train_indices,
                    feature_names=ROAD_FEATURES,
                    outer_fold=outer_fold,
                )
                test_prediction, _train_prediction, model = _fit_predict_stratum_experts(
                    feature_rows,
                    train_indices,
                    test_indices,
                    feature_names=ROAD_FEATURES,
                    alpha=alpha,
                )
                router = "TARGET_FREE_G3_DOMINANT_PRE_OUTCOME_CLASS_FLIP_STRATUM"
            predictions[test_indices] = test_prediction
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
        road_metrics = _road_slice_metrics(feature_rows, predictions.tolist())
        candidate_outputs[candidate_id] = {
            "candidate_id": candidate_id,
            "learned_form_tag": HEURISTIC_TAG,
            "feature_names": list(ROAD_FEATURES),
            "router": router,
            "outer_fold_models": fold_models,
            "metrics": metrics,
            "road_slice_metrics": road_metrics,
            "status": "RACED_PAIR_HELD_OUT_ROAD_CONDITIONAL",
        }
        outer_predictions[candidate_id] = predictions
        candidate_complexity[candidate_id] = complexity

    road_winner = max(
        (candidate_outputs[candidate_id] for candidate_id, _ in ROAD_CANDIDATES),
        key=lambda row: (
            row["road_slice_metrics"]["ndcg_at_4"],
            row["metrics"]["ndcg_at_4"],
            (
                -math.inf
                if row["road_slice_metrics"]["spearman_rho"] is None
                else row["road_slice_metrics"]["spearman_rho"]
            ),
            -candidate_complexity[row["candidate_id"]],
        ),
    )
    road_gate_passed = (
        float(road_winner["road_slice_metrics"]["ndcg_at_4"]) >= ROAD_ADMISSION_NDCG_AT_4
        and float(road_winner["metrics"]["ndcg_at_4"]) >= ADMISSION_NDCG_AT_4
    )
    winner = road_winner if road_gate_passed else dict(co3_receipt["selected_model"])
    selected_id = str(winner["candidate_id"])
    selected_predictions = outer_predictions[selected_id] if road_gate_passed else co3_predictions
    admission_passed = float(winner["metrics"]["ndcg_at_4"]) >= ADMISSION_NDCG_AT_4 and (
        selected_id not in {candidate_id for candidate_id, _ in ROAD_CANDIDATES} or road_gate_passed
    )
    slices, innovations = _slice_rows(feature_rows, selected_predictions.tolist())

    selected_innovations = np.asarray(targets) - selected_predictions
    residual_sigma = float(selected_innovations.std(ddof=1))
    pair_rows: list[dict[str, Any]] = []
    precision_count = 0
    direct_precision_count = 0
    propagated_precision_count = 0
    unranked_precision_count = 0
    margin_thresholds = _quantile_thresholds([_finite(row["margin"], "margin") for row in feature_rows])
    hardness_thresholds = _quantile_thresholds([_finite(row["hardness"], "hardness") for row in feature_rows])
    for row, prediction in zip(feature_rows, selected_predictions, strict=True):
        fisher_trace = row["direct_fisher_trace"]
        if fisher_trace is not None and float(fisher_trace) > 0.0:
            nominal_standard_error = residual_sigma / math.sqrt(float(fisher_trace))
            standard_error = nominal_standard_error
            interval = [
                float(prediction - 1.96 * standard_error),
                float(prediction + 1.96 * standard_error),
            ]
            precision_status = "DERIVED_FROM_MEASURED_DIRECT_MS4D_FISHER"
            precision_class = "DIRECT"
            precision_design_effect = 1.0
            precision_assumptions = [
                "DIRECT_PAIR_INDEXED_MS4D_FISHER_BLOCK",
                "OOF_RESIDUAL_SCALE_LOCAL_LINEARIZATION",
            ]
            precision_count += 1
            direct_precision_count += 1
        elif float(row["propagated_fisher_trace"]) > 0.0:
            propagated_trace = float(row["propagated_fisher_trace"])
            precision_design_effect = max(
                1.0,
                float(row["precision_design_effect"]),
            )
            nominal_standard_error = residual_sigma / math.sqrt(propagated_trace)
            standard_error = nominal_standard_error * math.sqrt(precision_design_effect)
            interval = [
                float(prediction - 1.96 * standard_error),
                float(prediction + 1.96 * standard_error),
            ]
            precision_status = "PROPAGATED_FROM_MS5_MS6_RG3_PAIR_SUPPORT_X_MS4D_BUCKET_GRAMS"
            precision_class = "PROPAGATED"
            precision_assumptions = [
                "PF2_PAIR_SUPPORT_COUNTS_ARE_EXACT_BUCKET_MEMBERSHIP",
                "WITHIN_BUCKET_EXCHANGEABILITY_OF_EVENT_FISHER",
                "BUCKET_FISHER_INFORMATION_ADDS_BY_SUPPORT_SHARE",
                "ADDITIVE_INDEPENDENT_BUCKET_BLOCKS",
                "NO_UNMEASURED_CROSS_BUCKET_COVARIANCE",
                "DEPENDENCE_RISK_PENALIZED_BY_KISH_CV_DESIGN_EFFECT",
                "OOF_RESIDUAL_SCALE_LOCAL_LINEARIZATION",
                "PROPAGATED_INTERVAL_IS_WIDER_THAN_NOMINAL_BY_SQRT_DESIGN_EFFECT",
            ]
            precision_count += 1
            propagated_precision_count += 1
        else:
            nominal_standard_error = None
            standard_error = None
            interval = None
            precision_status = "UNRANKED_NO_POSITIVE_PAIR_INFORMATION"
            precision_class = "UNRANKED"
            precision_design_effect = None
            precision_assumptions = ["NO_DIRECT_OR_POSITIVE_PROPAGATED_PAIR_FISHER_AUTHORITY"]
            unranked_precision_count += 1
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
                "margin_decile": int(_decile(float(row["margin"]), margin_thresholds)),
                "pair_hardness_decile": int(_decile(float(row["hardness"]), hardness_thresholds)),
                "precision_class": precision_class,
                "nominal_fisher_standard_error": nominal_standard_error,
                "fisher_standard_error": standard_error,
                "fisher_95_interval": interval,
                "precision_status": precision_status,
                "precision_design_effect": precision_design_effect,
                "precision_assumptions": precision_assumptions,
                "direct_fisher_trace": row["direct_fisher_trace"],
                "propagated_fisher_trace": float(row["propagated_fisher_trace"]),
                "precision_bucket_count": int(row["precision_bucket_count"]),
                "precision_support_count": int(row["precision_support_count"]),
                "learned_form_tag": HEURISTIC_TAG,
                "score_claim": False,
                "actuation": "NONE",
            }
        )
    pair_rows.sort(key=lambda row: (-row["prediction"], row["pair_id"]))
    for rank, row in enumerate(pair_rows, 1):
        row["oof_rank"] = rank
        if row["fisher_95_interval"] is None:
            row["pair_order_status"] = "TIED_MISSING_INTERVAL_UNRANKED_PRECISION"
            row["pair_order_class"] = "TIED"
        elif rank > 1:
            previous = pair_rows[rank - 2]
            previous_interval = previous["fisher_95_interval"]
            if previous_interval is None or row["fisher_95_interval"][1] >= previous_interval[0]:
                row["pair_order_status"] = "TIED_OVERLAPPING_OR_MISSING_INTERVAL"
                row["pair_order_class"] = "TIED"
            else:
                row["pair_order_status"] = "ORDERED_NONOVERLAPPING_INTERVAL"
                row["pair_order_class"] = "ORDERED"
        else:
            row["pair_order_status"] = "LEADER_INTERVAL_ESTIMATE"
            row["pair_order_class"] = "LEADER"

    self_checks = [
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "pontryagin_bellman_adjacent_lambda_residual",
            "status": "AWAITING_J8F",
            "value": None,
            "reason": (
                "ordered adjacent J8F costates and realized transition terms are "
                "absent; OOF pair predictions are not a Bellman trajectory"
            ),
        },
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "m34_per_state_dual_consistency",
            "status": "AWAITING_J8F_M34_PER_STATE_DUALS",
            "value": None,
            "reason": (
                f"hash-verified RD1 source has {len(actionable_rd1_duals)} "
                f"aggregate actionable prices across {len(rd1_duals)} typed rows, "
                "but aggregate RD1 prices cannot substitute for J8F M34 per-state "
                "duals; null is not coerced to zero"
            ),
        },
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "wallace_mml_pair_precision",
            "status": ("COMPLETE" if precision_count == PAIR_COUNT else "PARTIAL_TYPED"),
            "value": {
                "pair_intervals": precision_count,
                "direct": direct_precision_count,
                "propagated": propagated_precision_count,
                "unranked": unranked_precision_count,
                "required": PAIR_COUNT,
            },
            "reason": (
                "direct pair blocks override support-propagated PF2/MS4D "
                "information; propagated intervals carry explicit dependence "
                "inflation and missing positive information remains unranked"
            ),
        },
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "compression_progress_per_effort",
            "status": "AWAITING_J8F",
            "value": None,
            "reason": ("J8F delta_S_per_wall_clock_hour is absent; no proxy effort currency is promoted"),
        },
        {
            "schema": SELF_CHECK_SCHEMA,
            "check_id": "unsound_scaling_and_label_noise_claim_audit",
            "status": soundness_audit["status"],
            "value": soundness_audit,
            "reason": (
                "77x, 2.71x, params^-0.71, and label-noise-ceiling claims "
                "have no authority in the consumed CO2/CO3 inputs"
            ),
        },
    ]
    precision_complete = precision_count == PAIR_COUNT
    explanation = _falling_rule_explanation(
        admission_ndcg=float(winner["metrics"]["ndcg_at_4"]),
        precision_complete=precision_complete,
        j8f_verdict_count=0,
    )
    decide_rows = [
        _typed_decide_row(
            decide_id="co4_road_local_candidate_admission",
            status=(
                "DECIDE_ROAD_LOCAL_CANDIDATE_ADMITTED" if road_gate_passed else "DECIDE_RETAIN_CO3_ROAD_GATE_FAILED"
            ),
            condition_name="road_gate_passed",
            condition_value=road_gate_passed,
            reason=(
                f"{road_winner['candidate_id']} held-out Road NDCG@4="
                f"{road_winner['road_slice_metrics']['ndcg_at_4']:.12g} and "
                f"global NDCG@4={road_winner['metrics']['ndcg_at_4']:.12g}; "
                f"thresholds are {ROAD_ADMISSION_NDCG_AT_4} and "
                f"{ADMISSION_NDCG_AT_4}"
            ),
            authority="CO4_PREREGISTERED_PAIR_HELD_OUT_ROAD_RACE",
        ),
        _typed_decide_row(
            decide_id="catalog_611_scorer_recursive_construction",
            status=("BLOCKED_TYPED_COUNTED_SCORER_RECURSIVE_APPLICATION_OPERATOR_OWED"),
            condition_name="typed_counted_application_operator_present",
            condition_value=False,
            reason=(
                "#611 proposals are reactivated for construction, but no typed "
                "counted application operator with receiver parse-back and "
                "realized evaluator-cell custody is present; no actuation is "
                "authorized"
            ),
            authority=("BROADCAST_20260724T230954Z_X_SCORER_RECURSIVE_CONSTRUCTION_BINDING"),
        ),
        _typed_decide_row(
            decide_id="ms2r_immutable_stage_input_mismatch",
            status=("DIAGNOSED_ORACLE_INPUT_SUPERSESSION_NEW_STAGE_NAMESPACE_REQUIRED"),
            condition_name="input_supersession_diagnosed",
            condition_value=True,
            reason=(
                "stored stage 01 had WRAPPED=14 and TYPED-GAP=7; fresh oracle "
                "coverage has WRAPPED=21 and TYPED-GAP=0. The 72 structural "
                "diffs are entirely the seven newly GC2-bound oracle rows, not "
                "CO4 model drift or checkpoint corruption. Preserve the "
                "immutable stage and rerun only under a new stage/config identity"
            ),
            authority=(
                ".omx/research/ddm_ms2r_r3_typed_fisher_g4_waterfill_"
                "20260724T211804Z/stage_checkpoints/01_oracle_admission.json"
            ),
        ),
        _typed_decide_row(
            decide_id="pontryagin_bellman_residual",
            status="AWAITING_J8F",
            condition_name="j8f_adjacent_transition_terms_present",
            condition_value=False,
            reason=(
                "pair-held-out innovations are not an ordered Bellman "
                "trajectory and cannot fill missing J8F transition terms"
            ),
            authority="J8F_REALIZED_TRANSITION_TELEMETRY",
        ),
        _typed_decide_row(
            decide_id="m34_per_state_dual_replacement",
            status="AWAITING_J8F_M34_PER_STATE_DUALS",
            condition_name="j8f_m34_per_state_duals_present",
            condition_value=False,
            reason=("aggregate RD1 prices are retained as lineage but are not substituted for state-indexed M34 duals"),
            authority="J8F_M34_PER_STATE_DUAL_TELEMETRY",
        ),
    ]
    blocker_ids = ["BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY"]
    if unranked_precision_count:
        blocker_ids.append(f"BLOCKED_PAIR_PRECISION_UNRANKED_{unranked_precision_count}")
    if not admission_passed:
        blocker_ids.append("BLOCKED_LAMBDA_RANKER_HELDOUT_NDCG_ADMISSION")

    source_lineage = {
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": sha256_file(preregistration),
            "bytes": preregistration.stat().st_size,
        },
        "co3_historical_receipt": {
            "path": CO3_RECEIPT_PATH,
            "sha256": sha256_file(co3_receipt_path),
            "content_sha256": co3_receipt["content_sha256"],
            "selected_candidate_id": co3_receipt["selected_model"]["candidate_id"],
            "global_ndcg_at_4": co3_receipt["selected_model"]["metrics"]["ndcg_at_4"],
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
        "pf2_bucket_assignment_oracle": _oracle_meta(
            assignment_row,
            surface_counts={
                "bucket_rows": len(assignment_data["rows"]),
                "nonempty_buckets": sum(int(row["event_count"]) > 0 for row in assignment_data["rows"]),
                "pair_ids_with_positive_support": sum(
                    float(row["propagated_fisher_trace"]) > 0.0 for row in feature_rows
                ),
            },
        ),
        "pose_tube_oracle": _oracle_meta(
            pose_row,
            surface_counts={
                "pair_rows": len(pose_data["rows"]),
                "converged_pair_rows": sum(bool(row["converged"]) for row in pose_data["rows"]),
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
            "fold_rule": ('SHA256("ddm-co3-n600-v1:" + decimal(pair_id))[0:8] mod 5'),
            "fold_counts": fold_counts,
            "heldout_unit": "source_pair_id",
            "shared_v19_rate_bytes_in_pair_target": False,
        },
        "model_race": list(candidate_outputs.values()),
        "selected_model": {
            "candidate_id": selected_id,
            "learned_form_tag": HEURISTIC_TAG,
            "prediction_source": (
                "FRESH_CO4_PAIR_HELD_OUT_CANDIDATE" if road_gate_passed else "SEALED_CO3_PAIR_HELD_OUT_RECEIPT"
            ),
            "selection_rule": (
                "Road candidates: max held-out Road NDCG@4, then global "
                "NDCG@4, Road Spearman, lower complexity; admit only if Road "
                "and global thresholds pass, otherwise retain historical CO3"
            ),
            "metrics": winner["metrics"],
        },
        "road_local_gate": {
            "evaluation_slice": "EV1_REALIZED_CLOSURE_STRATUM_ROAD",
            "evaluation_slice_is_router_forbidden": True,
            "router_authority": ("G3_TARGET_FREE_DOMINANT_PRE_OUTCOME_CLASS_FLIP_STRATUM_ONLY"),
            "candidate_id": road_winner["candidate_id"],
            "road_threshold_ndcg_at_4": ROAD_ADMISSION_NDCG_AT_4,
            "road_observed": road_winner["road_slice_metrics"],
            "global_threshold_ndcg_at_4": ADMISSION_NDCG_AT_4,
            "global_observed": road_winner["metrics"],
            "passed": road_gate_passed,
            "failure_action": "RETAIN_CO3_WINNER",
            "selected_candidate_id": selected_id,
            "verdict_scope": ("the two frozen CO4 ridge formulations only; the Road-local ranker family remains open"),
        },
        "historical_comparison": {
            "sealed_co3_receipt_candidate_id": co3_receipt["selected_model"]["candidate_id"],
            "sealed_co3_receipt_metrics": co3_receipt["selected_model"]["metrics"],
            "fresh_recomputed_historical_candidate_id": historical_winner["candidate_id"],
            "fresh_recomputed_historical_metrics": historical_winner["metrics"],
            "history_is_not_rewritten": True,
        },
        "admission_gate": {
            "metric": "concatenated_pair_out_of_fold_ndcg_at_4",
            "threshold": ADMISSION_NDCG_AT_4,
            "observed": float(winner["metrics"]["ndcg_at_4"]),
            "passed": admission_passed,
            "duty_ranking_upgrade_eligible": admission_passed,
            "pair_order_precision_complete": precision_complete,
            "pair_precision_counts": {
                "direct": direct_precision_count,
                "propagated": propagated_precision_count,
                "unranked": unranked_precision_count,
            },
        },
        "ranking_error_slices": slices,
        "innovations": innovations,
        "pair_rankings": pair_rows,
        "self_checks": self_checks,
        "decide_rows": decide_rows,
        "rudin_explanation": explanation,
        "bandit_allocation": {
            "status": "DESIGN_ONLY_NOT_ACTUATED",
            "design": (
                "future regret-bounded duty allocation uses measured compression "
                "progress per effort plus exploration bonus for never-fired levers"
            ),
            "required_evidence": ("J8F realized delta_S_per_wall_clock_hour and fired-duty history"),
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
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(payload) + b"\n")
    os.replace(temporary, destination)


__all__ = [
    "ADMISSION_NDCG_AT_4",
    "ALPHA_GRID",
    "BASE_FEATURES",
    "CANDIDATES",
    "CO3_RECEIPT_PATH",
    "DECIDE_ROW_SCHEMA",
    "EVIDENCE_AXIS",
    "G4_FEATURES",
    "HEURISTIC_TAG",
    "MS4D_FEATURES",
    "PAIR_COUNT",
    "PAIR_ROW_SCHEMA",
    "PREREGISTRATION_PATH",
    "RD1_PATH",
    "RD1_SCHEMA",
    "ROAD_ADMISSION_NDCG_AT_4",
    "ROAD_CANDIDATES",
    "ROAD_FEATURES",
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
