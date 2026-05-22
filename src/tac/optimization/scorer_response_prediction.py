# SPDX-License-Identifier: MIT
"""Out-of-fold prediction helpers for scorer-response datasets."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from tac.optimization.scorer_response_dataset import (
    ScorerResponseDatasetError,
    feature_correlations,
    normalize_legacy_response_dataset_authority,
    summarize_rows,
)

OOF_PREDICTION_SCHEMA = "scorer_response_oof_linear_predictions.v1"
DEFAULT_PREDICTION_FIELD = "ll_predicted_delta_vs_baseline_score"


def attach_out_of_fold_linear_predictions(
    dataset: dict[str, Any],
    *,
    target: str = "delta_vs_baseline_score",
    prediction_field: str = DEFAULT_PREDICTION_FIELD,
    ridge_lambda: float = 1.0e-4,
) -> dict[str, Any]:
    """Attach conservative out-of-fold linear predictions to a dataset.

    The fit leaves out each row's declared ``holdout_fold`` before writing that
    row's prediction. Features are limited to row metadata known before scorer
    response measurement: pair position, archive bytes, and family labels.
    Outcome fields such as observed score, pose, seg, or scorer deltas are not
    used as features.
    """

    if not prediction_field:
        raise ScorerResponseDatasetError("prediction_field must be non-empty")
    if not math.isfinite(float(ridge_lambda)) or float(ridge_lambda) < 0.0:
        raise ScorerResponseDatasetError("ridge_lambda must be finite and non-negative")

    normalized = normalize_legacy_response_dataset_authority(dataset)
    rows = copy.deepcopy(normalized["rows"])
    if not rows:
        raise ScorerResponseDatasetError("dataset rows must be non-empty")

    feature_names, x = _design_matrix(rows)
    y = np.asarray([_required_float(row.get(target), f"{target} for {row.get('row_id')}") for row in rows], dtype=np.float64)
    folds = np.asarray([_required_int(row.get("holdout_fold"), f"holdout_fold for {row.get('row_id')}") for row in rows], dtype=np.int64)
    unique_folds = sorted(int(fold) for fold in np.unique(folds))
    predictions = np.full((len(rows),), np.nan, dtype=np.float64)
    fold_summaries: list[dict[str, Any]] = []

    for fold in unique_folds:
        test_mask = folds == fold
        train_mask = ~test_mask
        if int(np.count_nonzero(test_mask)) == 0:
            continue
        if int(np.count_nonzero(train_mask)) < 2:
            raise ScorerResponseDatasetError(
                f"not enough training rows outside holdout fold {fold}"
            )
        beta = _fit_ridge(x[train_mask], y[train_mask], ridge_lambda=float(ridge_lambda))
        fold_predictions = x[test_mask] @ beta
        predictions[test_mask] = fold_predictions
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": int(np.count_nonzero(train_mask)),
                "test_rows": int(np.count_nonzero(test_mask)),
            }
        )

    if np.isnan(predictions).any():
        raise ScorerResponseDatasetError("failed to produce predictions for all rows")

    for row, prediction in zip(rows, predictions, strict=True):
        row[prediction_field] = float(prediction)

    out = copy.deepcopy(normalized)
    out["rows"] = rows
    out["summary"] = summarize_rows(rows)
    out["feature_correlations"] = feature_correlations(rows)
    out["prediction_fit"] = {
        "schema": OOF_PREDICTION_SCHEMA,
        "target": target,
        "prediction_field": prediction_field,
        "ridge_lambda": float(ridge_lambda),
        "feature_set": "pair_family_archive_linear_v1",
        "feature_names": feature_names,
        "folds": fold_summaries,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return out


def _design_matrix(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    families = sorted({str(row.get("family") or "unknown") for row in rows})
    starts = [_pair_start(row) for row in rows]
    max_start = max(max(starts), 1.0)
    archive_values = [
        _optional_float(row.get("archive_bytes"), default=0.0)
        for row in rows
    ]
    max_archive = max(max(archive_values), 1.0)

    feature_names = [
        "intercept",
        "pair_start_norm",
        "pair_start_norm_sq",
        "pair_start_sin_1",
        "pair_start_cos_1",
        "archive_bytes_norm",
    ]
    feature_names.extend(f"family={family}" for family in families)
    feature_names.extend(f"family={family}:pair_start_norm" for family in families)

    matrix: list[list[float]] = []
    for row, start, archive_bytes in zip(rows, starts, archive_values, strict=True):
        x = float(start) / float(max_start)
        family = str(row.get("family") or "unknown")
        base = [
            1.0,
            x,
            x * x,
            math.sin(2.0 * math.pi * x),
            math.cos(2.0 * math.pi * x),
            float(archive_bytes) / float(max_archive),
        ]
        family_one_hot = [1.0 if family == item else 0.0 for item in families]
        family_interactions = [value * x for value in family_one_hot]
        matrix.append(base + family_one_hot + family_interactions)
    return feature_names, np.asarray(matrix, dtype=np.float64)


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    xty = x.T @ y
    return np.linalg.solve(xtx + penalty, xty)


def _pair_start(row: dict[str, Any]) -> float:
    pair_window = row.get("source_pair_window") or row.get("pair_indices")
    if isinstance(pair_window, list) and pair_window:
        value = _optional_float(pair_window[0], default=None)
        if value is not None:
            return value
    value = _optional_float(row.get("source_start_pair"), default=None)
    if value is not None:
        return value
    return 0.0


def _required_float(value: Any, label: str) -> float:
    parsed = _optional_float(value, default=None)
    if parsed is None:
        raise ScorerResponseDatasetError(f"{label} must be finite")
    return parsed


def _required_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ScorerResponseDatasetError(f"{label} must be an integer") from exc
    return parsed


def _optional_float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


__all__ = [
    "DEFAULT_PREDICTION_FIELD",
    "OOF_PREDICTION_SCHEMA",
    "attach_out_of_fold_linear_predictions",
]
