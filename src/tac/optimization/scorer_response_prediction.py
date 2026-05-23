# SPDX-License-Identifier: MIT
"""Out-of-fold prediction helpers for scorer-response datasets."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import numpy as np

from tac.optimization.scorer_response_dataset import (
    ScorerResponseDatasetError,
    feature_correlations,
    normalize_legacy_response_dataset_authority,
    scorer_response_planning_value_for_target,
    summarize_rows,
)

OOF_PREDICTION_SCHEMA = "scorer_response_oof_linear_predictions.v1"
OOF_MODEL_SELECTION_SCHEMA = "scorer_response_oof_model_selection_predictions.v1"
DEFAULT_PREDICTION_FIELD = "ll_predicted_delta_vs_baseline_score"
BASE_FEATURE_SET = "pair_family_archive_linear_v1"
STRUCTURED_FEATURE_SET = "pair_family_archive_linear_plus_allowlisted_structural_v1"
EXPANDED_FEATURE_SET = "pair_family_archive_expanded_nested_ridge_v1"
LINEAR_MODEL_FAMILY = "linear"
EXPANDED_MODEL_FAMILY = "expanded"
DEFAULT_EXPANDED_RIDGE_LAMBDAS = (
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
)
DECLARED_FOLD_STRATEGY = "declared"
GROUP_HASH_FOLD_STRATEGY = "group_hash"
CANDIDATE_FAMILY_TOP_K = (8, 16, 32)
CANDIDATE_FAMILY_MIN_PEARSON_R = 0.2
CANDIDATE_FAMILY_MIN_TOP_K_OVERLAP = 1
CANDIDATE_FAMILY_MIN_NEGATIVE_PREDICTIONS = 1

EXTRA_NUMERIC_FEATURE_FIELDS = (
    "diagnostic_cycle_pair_index_norm",
    "diagnostic_seg_last_l1",
    "diagnostic_pose_pair_l1",
    "diagnostic_rate_pair_l1",
    "diagnostic_total_pair_l1",
    "diagnostic_seg_share",
    "diagnostic_pose_share",
    "window_baseline_score",
    "window_baseline_avg_posenet_dist",
    "window_baseline_avg_segnet_dist",
    "window_baseline_pose_term",
    "window_baseline_seg_term",
    "window_baseline_scorer_term",
    "decoder_q_delta",
    "decoder_q_q_offset",
    "decoder_q_score_impact_abs_sum",
    "decoder_q_axis_share_seg",
    "decoder_q_axis_share_pose",
    "decoder_q_axis_share_rate",
    "decoder_q_top_byte_count",
    "decoder_q_approx_compressed_start_norm",
    "decoder_q_approx_compressed_length_norm",
)


def attach_out_of_fold_linear_predictions(
    dataset: dict[str, Any],
    *,
    target: str = "delta_vs_baseline_score",
    prediction_field: str = DEFAULT_PREDICTION_FIELD,
    ridge_lambda: float = 1.0e-4,
    model_family: str = LINEAR_MODEL_FAMILY,
    ridge_lambdas: tuple[float, ...] | None = None,
    fold_strategy: str = DECLARED_FOLD_STRATEGY,
    fold_key: str = "source_start_pair",
    n_folds: int = 5,
) -> dict[str, Any]:
    """Attach conservative out-of-fold ridge predictions to a dataset.

    The fit leaves out each row's declared ``holdout_fold`` before writing that
    row's prediction. Features are limited to row metadata known before scorer
    response measurement: pair position, archive bytes, family labels, and
    explicitly allowlisted structural priors when present. Outcome fields such
    as observed score, pose, seg, or scorer deltas are not used as features.

    ``model_family="linear"`` preserves the historical single-design ridge
    model. ``model_family="expanded"`` performs nested model selection inside
    each outer holdout fold over richer pair-local/Fourier/RBF bases and ridge
    strengths. With ``fold_strategy="group_hash"``, rows sharing ``fold_key``
    receive the same deterministic outer fold before fitting, preventing sibling
    rows from the same pair/window from crossing train/test boundaries.
    """

    if not prediction_field:
        raise ScorerResponseDatasetError("prediction_field must be non-empty")
    if not math.isfinite(float(ridge_lambda)) or float(ridge_lambda) < 0.0:
        raise ScorerResponseDatasetError("ridge_lambda must be finite and non-negative")
    if model_family not in {LINEAR_MODEL_FAMILY, EXPANDED_MODEL_FAMILY}:
        raise ScorerResponseDatasetError(
            f"model_family must be one of {LINEAR_MODEL_FAMILY!r}, {EXPANDED_MODEL_FAMILY!r}"
        )
    parsed_ridge_lambdas = _ridge_lambdas_for_model_family(
        model_family,
        ridge_lambda=ridge_lambda,
        ridge_lambdas=ridge_lambdas,
    )
    if fold_strategy not in {DECLARED_FOLD_STRATEGY, GROUP_HASH_FOLD_STRATEGY}:
        raise ScorerResponseDatasetError(
            f"fold_strategy must be one of {DECLARED_FOLD_STRATEGY!r}, {GROUP_HASH_FOLD_STRATEGY!r}"
        )
    if int(n_folds) < 2:
        raise ScorerResponseDatasetError("n_folds must be >= 2")
    if not fold_key:
        raise ScorerResponseDatasetError("fold_key must be non-empty")

    normalized = normalize_legacy_response_dataset_authority(dataset)
    rows = copy.deepcopy(normalized["rows"])
    if not rows:
        raise ScorerResponseDatasetError("dataset rows must be non-empty")

    candidate_designs = _candidate_design_matrices(rows, model_family=model_family)
    feature_names = candidate_designs[0]["feature_names"]
    y = np.asarray(
        [
            _required_float(
                scorer_response_planning_value_for_target(
                    row,
                    target,
                    label=str(row.get("row_id") or "<unknown>"),
                ),
                f"{target} for {row.get('row_id')}",
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    folds = _folds_for_rows(
        rows,
        fold_strategy=fold_strategy,
        fold_key=fold_key,
        n_folds=int(n_folds),
    )
    unique_folds = sorted(int(fold) for fold in np.unique(folds))
    predictions = np.full((len(rows),), np.nan, dtype=np.float64)
    fold_summaries: list[dict[str, Any]] = []
    selected_feature_names: dict[str, list[str]] = {}

    for fold in unique_folds:
        test_mask = folds == fold
        train_mask = ~test_mask
        if int(np.count_nonzero(test_mask)) == 0:
            continue
        if int(np.count_nonzero(train_mask)) < 2:
            raise ScorerResponseDatasetError(
                f"not enough training rows outside holdout fold {fold}"
            )
        selected = _select_design_for_outer_fold(
            candidate_designs,
            y=y,
            folds=folds,
            train_mask=train_mask,
            ridge_lambdas=parsed_ridge_lambdas,
        )
        beta = _fit_ridge(
            selected["x"][train_mask],
            y[train_mask],
            ridge_lambda=float(selected["ridge_lambda"]),
        )
        fold_predictions = selected["x"][test_mask] @ beta
        predictions[test_mask] = fold_predictions
        selected_feature_names[selected["feature_set"]] = selected["feature_names"]
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": int(np.count_nonzero(train_mask)),
                "test_rows": int(np.count_nonzero(test_mask)),
                "feature_set": selected["feature_set"],
                "ridge_lambda": float(selected["ridge_lambda"]),
                "inner_rmse": selected.get("inner_rmse"),
                "inner_pearson_r": selected.get("inner_pearson_r"),
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
    if model_family == LINEAR_MODEL_FAMILY:
        schema = OOF_PREDICTION_SCHEMA
        feature_set = (
            STRUCTURED_FEATURE_SET
            if any(name in feature_names for name in EXTRA_NUMERIC_FEATURE_FIELDS)
            else BASE_FEATURE_SET
        )
        fit_feature_names = feature_names
    else:
        schema = OOF_MODEL_SELECTION_SCHEMA
        feature_set = EXPANDED_FEATURE_SET
        fit_feature_names = sorted(
            {
                name
                for names in selected_feature_names.values()
                for name in names
            }
        )
    out["prediction_fit"] = {
        "schema": schema,
        "target": target,
        "target_value_accessor": "scorer_response_planning_value_for_target",
        "prediction_field": prediction_field,
        "model_family": model_family,
        "ridge_lambda": float(ridge_lambda),
        "ridge_lambdas": [float(value) for value in parsed_ridge_lambdas],
        "fold_strategy": fold_strategy,
        "fold_key": fold_key,
        "n_folds": int(n_folds),
        "feature_set": feature_set,
        "candidate_feature_sets": [
            str(design["feature_set"]) for design in candidate_designs
        ],
        "feature_names": fit_feature_names,
        "folds": fold_summaries,
        "candidate_family_metrics": _candidate_family_metrics(
            rows,
            target=target,
            prediction_field=prediction_field,
        ),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    out["prediction_fit"]["spend_triage_usable"] = any(
        item.get("spend_triage_usable") is True
        for item in out["prediction_fit"]["candidate_family_metrics"].values()
    )
    return out


def _ridge_lambdas_for_model_family(
    model_family: str,
    *,
    ridge_lambda: float,
    ridge_lambdas: tuple[float, ...] | None,
) -> tuple[float, ...]:
    values = ridge_lambdas
    if values is None:
        values = (
            (float(ridge_lambda),)
            if model_family == LINEAR_MODEL_FAMILY
            else DEFAULT_EXPANDED_RIDGE_LAMBDAS
        )
    parsed: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ScorerResponseDatasetError(
                "ridge_lambdas must be finite and non-negative"
            )
        parsed.append(number)
    if not parsed:
        raise ScorerResponseDatasetError("ridge_lambdas must be non-empty")
    return tuple(parsed)


def _folds_for_rows(
    rows: list[dict[str, Any]],
    *,
    fold_strategy: str,
    fold_key: str,
    n_folds: int,
) -> np.ndarray:
    if fold_strategy == DECLARED_FOLD_STRATEGY:
        return np.asarray(
            [
                _required_int(
                    row.get("holdout_fold"),
                    f"holdout_fold for {row.get('row_id')}",
                )
                for row in rows
            ],
            dtype=np.int64,
        )
    folds: list[int] = []
    for row in rows:
        key = _group_key(row, fold_key)
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        fold = int.from_bytes(digest[:8], "big") % n_folds
        row["holdout_fold"] = fold
        folds.append(fold)
    return np.asarray(folds, dtype=np.int64)


def _group_key(row: dict[str, Any], fold_key: str) -> str:
    if fold_key in {"source_start_pair", "pair_start"}:
        return f"{fold_key}:{int(_pair_start(row))}"
    value = row.get(fold_key)
    if value is None:
        pair_window = row.get("source_pair_window")
        if fold_key == "source_pair_window" and isinstance(pair_window, list):
            value = pair_window
    if value is None:
        raise ScorerResponseDatasetError(
            f"{fold_key} missing for row {row.get('row_id')}"
        )
    if isinstance(value, list):
        return f"{fold_key}:{jsonable_list_key(value)}"
    return f"{fold_key}:{value}"


def jsonable_list_key(value: list[Any]) -> str:
    return ",".join(str(item) for item in value)


def _candidate_design_matrices(
    rows: list[dict[str, Any]],
    *,
    model_family: str,
) -> list[dict[str, Any]]:
    feature_names, x = _design_matrix(rows, mode="linear")
    if model_family == LINEAR_MODEL_FAMILY:
        return [
            {
                "feature_set": (
                    STRUCTURED_FEATURE_SET
                    if any(name in feature_names for name in EXTRA_NUMERIC_FEATURE_FIELDS)
                    else BASE_FEATURE_SET
                ),
                "feature_names": feature_names,
                "x": x,
            }
        ]
    designs: list[dict[str, Any]] = [
        {
            "feature_set": (
                STRUCTURED_FEATURE_SET
                if any(name in feature_names for name in EXTRA_NUMERIC_FEATURE_FIELDS)
                else BASE_FEATURE_SET
            ),
            "feature_names": feature_names,
            "x": x,
        }
    ]
    for mode in ("harmonic", "rbf", "hybrid"):
        expanded_names, expanded_x = _design_matrix(rows, mode=mode)
        designs.append(
            {
                "feature_set": f"pair_family_archive_{mode}_ridge_v1",
                "feature_names": expanded_names,
                "x": expanded_x,
            }
        )
    return designs


def _select_design_for_outer_fold(
    candidate_designs: list[dict[str, Any]],
    *,
    y: np.ndarray,
    folds: np.ndarray,
    train_mask: np.ndarray,
    ridge_lambdas: tuple[float, ...],
) -> dict[str, Any]:
    train_folds = sorted(int(fold) for fold in np.unique(folds[train_mask]))
    best: dict[str, Any] | None = None
    for design in candidate_designs:
        x = design["x"]
        for ridge_lambda in ridge_lambdas:
            inner_predictions = np.full((int(np.count_nonzero(train_mask)),), np.nan)
            train_indices = np.flatnonzero(train_mask)
            valid = True
            for inner_fold in train_folds:
                inner_test = train_mask & (folds == inner_fold)
                inner_train = train_mask & (folds != inner_fold)
                if int(np.count_nonzero(inner_test)) == 0:
                    continue
                if int(np.count_nonzero(inner_train)) < 2:
                    valid = False
                    break
                try:
                    beta = _fit_ridge(
                        x[inner_train],
                        y[inner_train],
                        ridge_lambda=float(ridge_lambda),
                    )
                except np.linalg.LinAlgError:
                    valid = False
                    break
                inner_test_indices = np.flatnonzero(inner_test)
                local_positions = np.searchsorted(train_indices, inner_test_indices)
                inner_predictions[local_positions] = x[inner_test] @ beta
            if not valid or np.isnan(inner_predictions).any():
                continue
            truth = y[train_mask]
            rmse = _rmse(truth, inner_predictions)
            pearson = _pearson_r(truth, inner_predictions)
            candidate = {
                **design,
                "ridge_lambda": float(ridge_lambda),
                "inner_rmse": rmse,
                "inner_pearson_r": pearson,
            }
            if best is None or (
                rmse,
                -(-math.inf if pearson is None else pearson),
                str(design["feature_set"]),
                float(ridge_lambda),
            ) < (
                best["inner_rmse"],
                -(-math.inf if best["inner_pearson_r"] is None else best["inner_pearson_r"]),
                str(best["feature_set"]),
                float(best["ridge_lambda"]),
            ):
                best = candidate
    if best is None:
        first = candidate_designs[0]
        return {**first, "ridge_lambda": ridge_lambdas[0], "inner_rmse": None, "inner_pearson_r": None}
    return best


def _design_matrix(
    rows: list[dict[str, Any]],
    *,
    mode: str = "linear",
) -> tuple[list[str], np.ndarray]:
    families = sorted({str(row.get("family") or "unknown") for row in rows})
    starts = [_pair_start(row) for row in rows]
    max_start = max(max(starts), 1.0)
    archive_values = [
        _optional_float(row.get("archive_bytes"), default=0.0)
        for row in rows
    ]
    max_archive = max(max(archive_values), 1.0)
    extra_features = _active_extra_numeric_features(rows)
    extra_max_abs = {
        name: max(
            [
                abs(_optional_float(row.get(name), default=0.0) or 0.0)
                for row in rows
            ]
            + [1.0]
        )
        for name in extra_features
    }

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
    feature_names.extend(extra_features)
    if mode in {"harmonic", "hybrid"}:
        for harmonic in (2, 3, 4, 8, 16, 32):
            feature_names.extend(
                [
                    f"pair_start_sin_{harmonic}",
                    f"pair_start_cos_{harmonic}",
                ]
            )
        feature_names.extend(["pair_start_norm_cu", "pair_start_norm_quartic"])
        for family in families:
            for harmonic in (2, 4, 8, 16):
                feature_names.extend(
                    [
                        f"family={family}:pair_start_sin_{harmonic}",
                        f"family={family}:pair_start_cos_{harmonic}",
                    ]
                )
    if mode in {"rbf", "hybrid"}:
        centers = _rbf_centers()
        for center in centers:
            feature_names.append(f"pair_start_rbf_{center:.4f}")
        for family in families:
            for center in centers:
                feature_names.append(f"family={family}:pair_start_rbf_{center:.4f}")
    if mode in {"harmonic", "rbf", "hybrid"}:
        for family in families:
            for name in extra_features:
                feature_names.append(f"family={family}:{name}")

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
        extra_values = [
            (_optional_float(row.get(name), default=0.0) or 0.0) / extra_max_abs[name]
            for name in extra_features
        ]
        values = base + family_one_hot + family_interactions + extra_values
        if mode in {"harmonic", "hybrid"}:
            harmonics: list[float] = []
            for harmonic in (2, 3, 4, 8, 16, 32):
                harmonics.extend(
                    [
                        math.sin(2.0 * math.pi * harmonic * x),
                        math.cos(2.0 * math.pi * harmonic * x),
                    ]
                )
            harmonics.extend([x * x * x, x * x * x * x])
            values.extend(harmonics)
            for family_value in family_one_hot:
                for harmonic in (2, 4, 8, 16):
                    values.extend(
                        [
                            family_value * math.sin(2.0 * math.pi * harmonic * x),
                            family_value * math.cos(2.0 * math.pi * harmonic * x),
                        ]
                    )
        if mode in {"rbf", "hybrid"}:
            rbf_values = [_rbf(x, center) for center in _rbf_centers()]
            values.extend(rbf_values)
            for family_value in family_one_hot:
                values.extend([family_value * value for value in rbf_values])
        if mode in {"harmonic", "rbf", "hybrid"}:
            for family_value in family_one_hot:
                values.extend([family_value * value for value in extra_values])
        matrix.append(values)
    return feature_names, np.asarray(matrix, dtype=np.float64)


def _rbf_centers() -> tuple[float, ...]:
    return tuple(index / 16.0 for index in range(17))


def _rbf(value: float, center: float) -> float:
    width = 1.0 / 12.0
    z = (value - center) / width
    return math.exp(-0.5 * z * z)


def _active_extra_numeric_features(rows: list[dict[str, Any]]) -> list[str]:
    active: list[str] = []
    for name in EXTRA_NUMERIC_FEATURE_FIELDS:
        values = [
            _optional_float(row.get(name), default=None)
            for row in rows
            if row.get(name) is not None
        ]
        finite_values = [value for value in values if value is not None]
        if not finite_values:
            continue
        if any(abs(value) > 0.0 for value in finite_values):
            active.append(name)
    return active


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    xty = x.T @ y
    return np.linalg.solve(xtx + penalty, xty)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    centered_true = y_true - float(np.mean(y_true))
    centered_pred = y_pred - float(np.mean(y_pred))
    denom = float(np.linalg.norm(centered_true) * np.linalg.norm(centered_pred))
    if denom <= 0.0:
        return None
    return float((centered_true @ centered_pred) / denom)


def _candidate_family_metrics(
    rows: list[dict[str, Any]],
    *,
    target: str,
    prediction_field: str,
) -> dict[str, dict[str, Any]]:
    """Compute candidate-family utility metrics for a prediction field.

    Overall held-out correlation can be inflated by easy control/parent rows.
    These metrics measure whether the same field is useful inside each family
    as a spend-triage selector. They remain advisory and carry no score
    authority.
    """

    families = sorted({str(row.get("family") or "unknown") for row in rows})
    out: dict[str, dict[str, Any]] = {}
    for family in families:
        family_rows = [
            row for row in rows if str(row.get("family") or "unknown") == family
        ]
        predictions: list[float] = []
        observed: list[float] = []
        per_fold: list[dict[str, Any]] = []
        folds = sorted(
            {
                fold
                for row in family_rows
                if (fold := _optional_int(row.get("holdout_fold"))) is not None
            }
        )
        for fold in folds:
            fold_predictions: list[float] = []
            fold_observed: list[float] = []
            for row in family_rows:
                if _optional_int(row.get("holdout_fold")) != fold:
                    continue
                pred = _optional_float(row.get(prediction_field), default=None)
                actual = scorer_response_planning_value_for_target(
                    row,
                    target,
                    label=str(row.get("row_id") or "<unknown>"),
                )
                if pred is None or actual is None:
                    continue
                fold_predictions.append(pred)
                fold_observed.append(actual)
            per_fold.append(
                {
                    "fold": fold,
                    "n": len(fold_predictions),
                    "pearson_r": _pearson_lists(fold_observed, fold_predictions),
                }
            )
            predictions.extend(fold_predictions)
            observed.extend(fold_observed)
        overall = _pearson_lists(observed, predictions)
        top_k_metrics = _top_k_metrics(
            family_rows,
            target=target,
            prediction_field=prediction_field,
        )
        negative_prediction_count = sum(1 for value in predictions if value < 0.0)
        observed_improvement_count = sum(1 for value in observed if value < 0.0)
        top8 = top_k_metrics.get("8", {})
        top8_overlap = int(top8.get("overlap_count") or 0)
        top8_mean_observed = _optional_float(
            top8.get("mean_observed_delta_in_predicted_top_k"),
            default=None,
        )
        spend_triage_usable = (
            len(predictions) >= max(CANDIDATE_FAMILY_TOP_K[0], 3)
            and overall is not None
            and overall >= CANDIDATE_FAMILY_MIN_PEARSON_R
            and negative_prediction_count >= CANDIDATE_FAMILY_MIN_NEGATIVE_PREDICTIONS
            and observed_improvement_count > 0
            and top8_overlap >= CANDIDATE_FAMILY_MIN_TOP_K_OVERLAP
            and top8_mean_observed is not None
            and top8_mean_observed < 0.0
        )
        out[family] = {
            "family": family,
            "prediction_field": prediction_field,
            "target": target,
            "n": len(predictions),
            "overall_pearson_r": overall,
            "folds": per_fold,
            "negative_prediction_count": negative_prediction_count,
            "observed_improvement_count": observed_improvement_count,
            "top_k": top_k_metrics,
            "spend_triage_usable": spend_triage_usable,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
        }
    return out


def _top_k_metrics(
    rows: list[dict[str, Any]],
    *,
    target: str,
    prediction_field: str,
) -> dict[str, dict[str, Any]]:
    pairs = []
    for row in rows:
        pred = _optional_float(row.get(prediction_field), default=None)
        actual = scorer_response_planning_value_for_target(
            row,
            target,
            label=str(row.get("row_id") or "<unknown>"),
        )
        if pred is None or actual is None:
            continue
        pairs.append((str(row.get("row_id")), pred, actual))
    by_pred = sorted(pairs, key=lambda item: (item[1], item[0]))
    by_actual = sorted(pairs, key=lambda item: (item[2], item[0]))
    out: dict[str, dict[str, Any]] = {}
    for k in CANDIDATE_FAMILY_TOP_K:
        limit = min(k, len(pairs))
        predicted_ids = {item[0] for item in by_pred[:limit]}
        observed_ids = {item[0] for item in by_actual[:limit]}
        observed_in_predicted = [item[2] for item in by_pred[:limit]]
        out[str(k)] = {
            "k": k,
            "effective_k": limit,
            "overlap_count": len(predicted_ids & observed_ids),
            "overlap_fraction": (
                None if limit == 0 else len(predicted_ids & observed_ids) / limit
            ),
            "mean_observed_delta_in_predicted_top_k": (
                None
                if not observed_in_predicted
                else float(np.mean(np.asarray(observed_in_predicted)))
            ),
            "best_observed_delta_in_predicted_top_k": (
                None if not observed_in_predicted else min(observed_in_predicted)
            ),
        }
    return out


def _pearson_lists(y_true: list[float], y_pred: list[float]) -> float | None:
    if len(y_true) < 2 or len(y_true) != len(y_pred):
        return None
    return _pearson_r(
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )


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


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


__all__ = [
    "BASE_FEATURE_SET",
    "CANDIDATE_FAMILY_MIN_NEGATIVE_PREDICTIONS",
    "CANDIDATE_FAMILY_MIN_PEARSON_R",
    "CANDIDATE_FAMILY_MIN_TOP_K_OVERLAP",
    "CANDIDATE_FAMILY_TOP_K",
    "DECLARED_FOLD_STRATEGY",
    "DEFAULT_EXPANDED_RIDGE_LAMBDAS",
    "DEFAULT_PREDICTION_FIELD",
    "EXPANDED_FEATURE_SET",
    "EXPANDED_MODEL_FAMILY",
    "EXTRA_NUMERIC_FEATURE_FIELDS",
    "GROUP_HASH_FOLD_STRATEGY",
    "LINEAR_MODEL_FAMILY",
    "OOF_MODEL_SELECTION_SCHEMA",
    "OOF_PREDICTION_SCHEMA",
    "STRUCTURED_FEATURE_SET",
    "attach_out_of_fold_linear_predictions",
]
