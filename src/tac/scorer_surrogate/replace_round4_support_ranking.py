# SPDX-License-Identifier: MIT
"""Convex, class-pair-aware support rankers for REPLACE round 4.

The fitted objects are scalar quadratic heads over the committed round-3 local
pre-squeeze-excite chart.  Targets are exact top-area support labels or exact
positive/negative ranking pairs; no mass-regression target is fitted here.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.scorer_surrogate.replace_round3_fidelity_wall import (
    BASE_FEATURE_COUNT,
    mass_localization_metrics,
    prefix_cell_costate_l2,
    prefix_feature_matrix,
)

SCHEMA = "replace_round4_support_ranking.v1"
AUTHORITY_SCOPE = "local macOS-CPU frozen-teacher gradient research evidence; no score authority"
RESEARCH_ONLY = True
CLASS_COUNT = 5
ORDERED_PAIR_COUNT = CLASS_COUNT * (CLASS_COUNT - 1)
GLOBAL_FEATURE_COUNT = BASE_FEATURE_COUNT + 2 * ORDERED_PAIR_COUNT + 2
BLOCK_FEATURE_COUNT = BASE_FEATURE_COUNT + 2
SOURCE_CLASS_SENSITIVITY = np.asarray((2.2, 32.0, 0.26, 1.0, 0.0), dtype=np.float64)


class Round4RankingError(ValueError):
    """A tensor, convex-custody, calibration, or policy invariant failed."""


def _finite(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise Round4RankingError(f"{name} contains nonfinite values")
    return array


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def ordered_class_pair_ids(labels_hw: Any, logits_nchw: Any) -> np.ndarray:
    """Map each source class to its highest-logit non-source competitor.

    The returned ids are contiguous in ``[0, 20)`` and ordered by source class,
    then by competitor with the diagonal removed.
    """

    labels = np.asarray(labels_hw)
    logits = _finite(logits_nchw, name="logits_nchw", dtype=np.float32)
    if logits.ndim == 3:
        logits = logits[None]
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != CLASS_COUNT:
        raise Round4RankingError("logits must have shape (1,5,H,W)")
    if labels.shape != tuple(logits.shape[2:]) or labels.dtype.kind not in "iu":
        raise Round4RankingError("source labels must be an integer grid matching logits")
    if labels.min() < 0 or labels.max() >= CLASS_COUNT:
        raise Round4RankingError("source labels must lie in [0,5)")
    work = np.array(logits[0], dtype=np.float32, copy=True).reshape(CLASS_COUNT, -1)
    target = labels.astype(np.int64, copy=False).reshape(-1)
    work[target, np.arange(target.size, dtype=np.int64)] = -np.inf
    competitor = np.argmax(work, axis=0).astype(np.int64, copy=False)
    if np.any(competitor == target):
        raise Round4RankingError("non-source competitor construction retained a diagonal pair")
    within_source_rank = competitor - (competitor > target).astype(np.int64)
    pair = target * (CLASS_COUNT - 1) + within_source_rank
    return np.ascontiguousarray(pair.reshape(labels.shape), dtype=np.int16)


def pair_id_to_classes(pair_id: int) -> tuple[int, int]:
    if isinstance(pair_id, bool) or not isinstance(pair_id, (int, np.integer)):
        raise Round4RankingError("pair id must be an integer")
    if not 0 <= int(pair_id) < ORDERED_PAIR_COUNT:
        raise Round4RankingError("pair id lies outside the ordered class-pair chart")
    source = int(pair_id) // (CLASS_COUNT - 1)
    rank = int(pair_id) % (CLASS_COUNT - 1)
    competitor = rank if rank < source else rank + 1
    return source, competitor


def support_feature_matrices(
    prefix_nchw: Any,
    labels_hw: Any,
    margins_hw: Any,
    pair_ids_hw: Any,
    *,
    checkpoint_index: int,
    checkpoint_count: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the sealed global-84 and pair-block-44 charts plus pair ids."""

    base = prefix_feature_matrix(
        prefix_nchw,
        labels_hw,
        margins_hw,
        checkpoint_index=checkpoint_index,
        checkpoint_count=checkpoint_count,
        stride=stride,
    ).astype(np.float64, copy=False)
    prefix = np.asarray(prefix_nchw)
    if prefix.ndim == 4:
        prefix = prefix[0]
    if prefix.ndim != 3 or prefix.shape[0] != 32:
        raise Round4RankingError("prefix geometry drifted")
    height, width = prefix.shape[1:]
    pairs = np.asarray(pair_ids_hw)
    if pairs.shape == (2 * height, 2 * width):
        pairs = pairs[::2, ::2]
    sampled_shape = (len(range(0, height, stride)), len(range(0, width, stride)))
    if pairs.shape == (height, width):
        pair_rows = pairs[::stride, ::stride].reshape(-1).astype(np.int64, copy=False)
    elif pairs.shape == sampled_shape:
        pair_rows = pairs.reshape(-1).astype(np.int64, copy=False)
    else:
        raise Round4RankingError(
            "pair ids must match input, prefix, or registered sampled-prefix geometry"
        )
    if pairs.dtype.kind not in "iu":
        raise Round4RankingError("pair ids must use an integer dtype")
    if pair_rows.size != base.shape[0] or np.any((pair_rows < 0) | (pair_rows >= ORDERED_PAIR_COUNT)):
        raise Round4RankingError("sampled pair ids disagree with the feature chart")
    margins = _finite(margins_hw, name="margins_hw", dtype=np.float32)
    if margins.shape != (2 * height, 2 * width):
        raise Round4RankingError("source margins must match the input grid")
    margin_rows = np.tanh(margins[::2, ::2][::stride, ::stride]).reshape(-1).astype(
        np.float64, copy=False
    )
    labels = np.asarray(labels_hw)[::2, ::2][::stride, ::stride].reshape(-1).astype(np.int64)
    sensitivity = np.log1p(SOURCE_CLASS_SENSITIVITY[labels])
    one_hot_pair = np.eye(ORDERED_PAIR_COUNT, dtype=np.float64)[pair_rows]
    global_features = np.concatenate(
        (
            base,
            one_hot_pair,
            one_hot_pair * margin_rows[:, None],
            sensitivity[:, None],
            (sensitivity * margin_rows)[:, None],
        ),
        axis=1,
    )
    block_features = np.concatenate(
        (base, sensitivity[:, None], (sensitivity * margin_rows)[:, None]), axis=1
    )
    if global_features.shape[1] != GLOBAL_FEATURE_COUNT:
        raise Round4RankingError("global feature width drifted")
    if block_features.shape[1] != BLOCK_FEATURE_COUNT:
        raise Round4RankingError("block feature width drifted")
    return (
        np.ascontiguousarray(global_features, dtype=np.float64),
        np.ascontiguousarray(block_features, dtype=np.float64),
        np.ascontiguousarray(pair_rows, dtype=np.int16),
    )


def exact_support_target(
    input_costate_nchw: Any, *, area_fraction: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return prefix-cell L2-square mass and the deterministic oracle mask."""

    if not 0.0 < float(area_fraction) < 1.0:
        raise Round4RankingError("area fraction must lie strictly between zero and one")
    mass = np.square(prefix_cell_costate_l2(input_costate_nchw), dtype=np.float64)
    count = max(1, math.ceil(float(area_fraction) * mass.size))
    flat = np.arange(mass.size, dtype=np.int64)
    selected = np.lexsort((flat, -mass.reshape(-1)))[:count]
    support = np.zeros(mass.size, dtype=np.bool_)
    support[selected] = True
    return mass, support.reshape(mass.shape), count


@dataclass(frozen=True)
class QuadraticStatistics:
    gram: np.ndarray
    rhs: np.ndarray
    target_square: float
    row_count: int
    state_count: int

    def validate(self) -> None:
        gram = _finite(self.gram, name="gram")
        rhs = _finite(self.rhs, name="rhs")
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            raise Round4RankingError("quadratic gram must be square")
        if rhs.shape != (gram.shape[0],):
            raise Round4RankingError("quadratic rhs width disagrees with gram")
        if self.target_square < 0.0 or not math.isfinite(float(self.target_square)):
            raise Round4RankingError("quadratic target square is invalid")
        if self.row_count < 0 or self.state_count < 0:
            raise Round4RankingError("quadratic counts may not be negative")
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(gram))))
        if not np.allclose(gram, gram.T, rtol=0.0, atol=tolerance):
            raise Round4RankingError("quadratic gram is not symmetric")


def _weighted_rows(features: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(support, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != y.size:
        raise Round4RankingError("features and support target are not row aligned")
    positive = float(y.sum())
    negative = float(y.size - positive)
    if positive <= 0.0 or negative <= 0.0:
        raise Round4RankingError("sampled state must contain support and non-support rows")
    weights = np.where(y > 0.5, 0.5 / positive, 0.5 / negative)
    return y, weights


def weighted_topk_statistics(features: Any, support: Any) -> QuadraticStatistics:
    x = _finite(features, name="features")
    y, weights = _weighted_rows(x, np.asarray(support))
    weighted = x * weights[:, None]
    record = QuadraticStatistics(
        gram=np.ascontiguousarray(x.T @ weighted, dtype=np.float64),
        rhs=np.ascontiguousarray(x.T @ (weights * y), dtype=np.float64),
        target_square=float(np.dot(weights, np.square(y))),
        row_count=x.shape[0],
        state_count=1,
    )
    record.validate()
    return record


def weighted_topk_block_statistics(
    features: Any, pair_ids: Any, support: Any
) -> tuple[QuadraticStatistics, ...]:
    x = _finite(features, name="block_features")
    pair = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    y, weights = _weighted_rows(x, np.asarray(support))
    if pair.shape != y.shape or np.any((pair < 0) | (pair >= ORDERED_PAIR_COUNT)):
        raise Round4RankingError("pair ids and block feature rows disagree")
    records = []
    for block in range(ORDERED_PAIR_COUNT):
        mask = pair == block
        xb = x[mask]
        yb = y[mask]
        wb = weights[mask]
        records.append(
            QuadraticStatistics(
                gram=np.ascontiguousarray(xb.T @ (xb * wb[:, None]), dtype=np.float64),
                rhs=np.ascontiguousarray(xb.T @ (wb * yb), dtype=np.float64),
                target_square=float(np.dot(wb, np.square(yb))),
                row_count=int(mask.sum()),
                state_count=1 if np.any(mask) else 0,
            )
        )
        records[-1].validate()
    return tuple(records)


def pairwise_rank_block_statistics(
    features: Any, pair_ids: Any, support: Any
) -> tuple[QuadraticStatistics, ...]:
    """Implicit exact all-positive-negative-pairs RankRLS sufficient stats."""

    x = _finite(features, name="block_features")
    pair = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    y = np.asarray(support, dtype=np.bool_).reshape(-1)
    if x.ndim != 2 or x.shape[0] != pair.size or pair.shape != y.shape:
        raise Round4RankingError("pairwise rows are not aligned")
    records = []
    width = x.shape[1]
    for block in range(ORDERED_PAIR_COUNT):
        block_rows = pair == block
        xp = x[block_rows & y]
        xn = x[block_rows & ~y]
        np_ = xp.shape[0]
        nn = xn.shape[0]
        if np_ == 0 or nn == 0:
            records.append(
                QuadraticStatistics(
                    gram=np.zeros((width, width), dtype=np.float64),
                    rhs=np.zeros(width, dtype=np.float64),
                    target_square=0.0,
                    row_count=0,
                    state_count=0,
                )
            )
            continue
        sp = xp.sum(axis=0, dtype=np.float64)
        sn = xn.sum(axis=0, dtype=np.float64)
        gram = (
            nn * (xp.T @ xp)
            + np_ * (xn.T @ xn)
            - np.outer(sp, sn)
            - np.outer(sn, sp)
        )
        records.append(
            QuadraticStatistics(
                gram=np.ascontiguousarray(0.5 * (gram + gram.T), dtype=np.float64),
                rhs=np.ascontiguousarray(nn * sp - np_ * sn, dtype=np.float64),
                target_square=float(np_ * nn),
                row_count=int(np_ * nn),
                state_count=1,
            )
        )
        records[-1].validate()
    return tuple(records)


def aggregate_quadratic_statistics(
    records: Sequence[QuadraticStatistics],
) -> QuadraticStatistics:
    if not records:
        raise Round4RankingError("at least one quadratic record is required")
    for record in records:
        record.validate()
    width = records[0].rhs.size
    if any(record.rhs.size != width for record in records):
        raise Round4RankingError("quadratic records use different feature widths")
    result = QuadraticStatistics(
        gram=np.ascontiguousarray(sum((record.gram for record in records), np.zeros((width, width))), dtype=np.float64),
        rhs=np.ascontiguousarray(sum((record.rhs for record in records), np.zeros(width)), dtype=np.float64),
        target_square=float(sum(record.target_square for record in records)),
        row_count=sum(record.row_count for record in records),
        state_count=sum(record.state_count for record in records),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class ConvexScalarFit:
    weights: np.ndarray
    certificate: dict[str, Any]


def fit_exact_quadratic(stats: QuadraticStatistics) -> ConvexScalarFit:
    """Solve the rank-truncated convex quadratic at its float64 MP optimum.

    The preregistered numerical problem discards eigendirections at or below
    ``eps * width * lambda_max``.  Its first-order certificate therefore lives
    in the retained eigenspace.  A full-gradient residual is still reported,
    but it can contain the finite-accumulation component of ``rhs`` in the
    declared numerical nullspace and is not an optimum test for that truncated
    problem.
    """

    stats.validate()
    gram = np.ascontiguousarray(0.5 * (stats.gram + stats.gram.T), dtype=np.float64)
    with np.errstate(all="ignore"):
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    maximum = max(0.0, float(eigenvalues[-1])) if eigenvalues.size else 0.0
    cutoff = np.finfo(np.float64).eps * max(1, gram.shape[0]) * maximum
    retained = eigenvalues > cutoff
    weights = np.zeros(gram.shape[0], dtype=np.float64)
    basis = eigenvectors[:, retained]
    discarded_basis = eigenvectors[:, ~retained]
    with np.errstate(all="ignore"):
        if np.any(retained):
            weights = basis @ ((basis.T @ stats.rhs) / eigenvalues[retained])
        gradient = gram @ weights - stats.rhs
        retained_gradient = basis.T @ gradient
        discarded_rhs = discarded_basis.T @ stats.rhs
        retained_gradient_inf = float(
            np.max(np.abs(retained_gradient), initial=0.0)
        )
        full_gradient_inf = float(np.max(np.abs(gradient), initial=0.0))
        rhs_l2 = float(np.linalg.norm(stats.rhs))
        discarded_rhs_l2 = float(np.linalg.norm(discarded_rhs))
        certificate_scale = (
            1.0 + rhs_l2 + maximum * float(np.linalg.norm(weights))
        )
        tolerance = (
            128.0
            * np.finfo(np.float64).eps
            * max(1, gram.shape[0])
            * certificate_scale
        )
        objective = float(
            weights @ gram @ weights - 2.0 * weights @ stats.rhs + stats.target_square
        )
    finite = bool(
        np.isfinite(weights).all()
        and np.isfinite(eigenvalues).all()
        and np.isfinite(objective)
        and np.isfinite(retained_gradient_inf)
        and np.isfinite(full_gradient_inf)
        and np.isfinite(discarded_rhs_l2)
    )
    certificate = {
        "schema": "replace_round4_exact_quadratic_certificate.v2",
        "fit_dtype": "float64",
        "feature_count": int(weights.size),
        "row_count": stats.row_count,
        "state_count": stats.state_count,
        "numerical_rank": int(retained.sum()),
        "rank_threshold": cutoff,
        "minimum_retained_eigenvalue": (
            float(eigenvalues[retained][0]) if np.any(retained) else None
        ),
        "maximum_eigenvalue": maximum,
        "normal_equation_gradient_inf": full_gradient_inf,
        "retained_space_normal_equation_gradient_inf": retained_gradient_inf,
        "normal_equation_tolerance": tolerance,
        "discarded_space_rhs_l2": discarded_rhs_l2,
        "discarded_space_rhs_fraction": discarded_rhs_l2 / max(rhs_l2, np.finfo(np.float64).tiny),
        "full_gradient_is_diagnostic_only": True,
        "normal_equation_optimum_certified": bool(
            finite and retained_gradient_inf <= tolerance
        ),
        "objective": objective,
        "weights_array_sha256": array_sha256(weights),
        "solver": (
            "symmetric-eigh Moore-Penrose minimum-norm solution of the "
            "preregistered rank-truncated normal equations"
        ),
    }
    if stats.row_count and not certificate["normal_equation_optimum_certified"]:
        raise Round4RankingError("float64 normal-equation optimum failed its certificate")
    return ConvexScalarFit(weights=np.ascontiguousarray(weights), certificate=certificate)


def global_scores(features: Any, weights: Any) -> np.ndarray:
    x = _finite(features, name="global_features")
    w = _finite(weights, name="global_weights")
    if w.shape != (x.shape[1],):
        raise Round4RankingError("global score head width drifted")
    return np.ascontiguousarray(x @ w, dtype=np.float64)


def block_scores(features: Any, pair_ids: Any, weights: Any) -> np.ndarray:
    x = _finite(features, name="block_features")
    pair = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    heads = _finite(weights, name="block_weights")
    if heads.shape != (ORDERED_PAIR_COUNT, x.shape[1]) or pair.size != x.shape[0]:
        raise Round4RankingError("block score head geometry drifted")
    return np.ascontiguousarray(np.einsum("ij,ij->i", x, heads[pair]), dtype=np.float64)


@dataclass(frozen=True)
class IsotonicCalibrator:
    x: np.ndarray
    probability: np.ndarray
    valid: bool
    reason: str
    sample_count: int
    positive_count: int

    def predict(self, scores: Any) -> np.ndarray:
        values = _finite(scores, name="calibration_scores")
        if not self.valid:
            raise Round4RankingError(f"invalid calibrator cannot predict: {self.reason}")
        return np.ascontiguousarray(
            np.interp(values, self.x, self.probability, left=self.probability[0], right=self.probability[-1]),
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "x": self.x.tolist(),
            "probability": self.probability.tolist(),
        }


def fit_isotonic_calibrator(
    scores: Any, labels: Any, *, bin_count: int
) -> IsotonicCalibrator:
    values = _finite(scores, name="calibration_scores").reshape(-1)
    target = np.asarray(labels, dtype=np.bool_).reshape(-1)
    if values.shape != target.shape or values.size < 2:
        raise Round4RankingError("calibration scores and labels must be aligned")
    positive = int(target.sum())
    if positive == 0 or positive == target.size:
        return IsotonicCalibrator(
            x=np.empty(0),
            probability=np.empty(0),
            valid=False,
            reason="one support class absent",
            sample_count=target.size,
            positive_count=positive,
        )
    order = np.argsort(values, kind="stable")
    chunks = [chunk for chunk in np.array_split(order, min(bin_count, values.size)) if chunk.size]
    blocks: list[dict[str, float]] = []
    for chunk in chunks:
        weight = float(chunk.size)
        blocks.append(
            {
                "weight": weight,
                "x_sum": float(values[chunk].sum(dtype=np.float64)),
                "probability": float((target[chunk].sum() + 0.5) / (chunk.size + 1.0)),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["probability"] > blocks[-1]["probability"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            blocks.append(
                {
                    "weight": weight,
                    "x_sum": left["x_sum"] + right["x_sum"],
                    "probability": (
                        left["probability"] * left["weight"]
                        + right["probability"] * right["weight"]
                    )
                    / weight,
                }
            )
    x = np.asarray([row["x_sum"] / row["weight"] for row in blocks], dtype=np.float64)
    probability = np.asarray([row["probability"] for row in blocks], dtype=np.float64)
    unique_x, unique_index = np.unique(x, return_index=True)
    probability = probability[unique_index]
    valid = unique_x.size >= 2
    return IsotonicCalibrator(
        x=unique_x,
        probability=probability,
        valid=valid,
        reason="valid" if valid else "fewer than two distinct calibrated score knots",
        sample_count=target.size,
        positive_count=positive,
    )


def fit_block_calibrators(
    scores: Any, labels: Any, pair_ids: Any, *, bin_count: int
) -> tuple[IsotonicCalibrator, tuple[IsotonicCalibrator, ...]]:
    values = _finite(scores, name="calibration_scores").reshape(-1)
    target = np.asarray(labels, dtype=np.bool_).reshape(-1)
    pair = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    if values.shape != target.shape or pair.shape != target.shape:
        raise Round4RankingError("block calibration rows are not aligned")
    global_calibrator = fit_isotonic_calibrator(values, target, bin_count=bin_count)
    blocks = []
    for block in range(ORDERED_PAIR_COUNT):
        mask = pair == block
        if mask.sum() < 2:
            blocks.append(
                IsotonicCalibrator(
                    x=np.empty(0),
                    probability=np.empty(0),
                    valid=False,
                    reason="fewer than two training rows",
                    sample_count=int(mask.sum()),
                    positive_count=int(target[mask].sum()),
                )
            )
        else:
            blocks.append(fit_isotonic_calibrator(values[mask], target[mask], bin_count=bin_count))
    return global_calibrator, tuple(blocks)


def calibrated_block_scores(
    raw_scores: Any,
    pair_ids: Any,
    *,
    global_calibrator: IsotonicCalibrator,
    block_calibrators: Sequence[IsotonicCalibrator],
) -> tuple[np.ndarray, np.ndarray]:
    raw = _finite(raw_scores, name="raw_scores").reshape(-1)
    pair = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    if raw.shape != pair.shape or len(block_calibrators) != ORDERED_PAIR_COUNT:
        raise Round4RankingError("calibrated block geometry drifted")
    calibrated = np.empty_like(raw)
    fallback = np.zeros(raw.size, dtype=np.bool_)
    for block, calibrator in enumerate(block_calibrators):
        mask = pair == block
        if not np.any(mask):
            continue
        selected = calibrator if calibrator.valid else global_calibrator
        if not selected.valid:
            raise Round4RankingError("neither block nor global calibrator is valid")
        calibrated[mask] = selected.predict(raw[mask])
        fallback[mask] = not calibrator.valid
    return calibrated, fallback


def calibration_diagnostics(
    probability: Any, labels: Any, *, bin_count: int = 10
) -> dict[str, Any]:
    prob = _finite(probability, name="probability").reshape(-1)
    target = np.asarray(labels, dtype=np.bool_).reshape(-1)
    if prob.shape != target.shape or np.any((prob < 0.0) | (prob > 1.0)):
        raise Round4RankingError("calibration probabilities or labels are invalid")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_index = np.minimum(np.searchsorted(edges, prob, side="right") - 1, bin_count - 1)
    bin_index = np.maximum(bin_index, 0)
    rows = []
    ece = 0.0
    for index in range(bin_count):
        mask = bin_index == index
        if not np.any(mask):
            continue
        confidence = float(prob[mask].mean())
        observed = float(target[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - observed)
        rows.append(
            {
                "bin": index,
                "count": int(mask.sum()),
                "mean_probability": confidence,
                "observed_support_rate": observed,
                "absolute_gap": abs(confidence - observed),
            }
        )
    return {
        "sample_count": int(prob.size),
        "support_prevalence": float(target.mean()),
        "expected_calibration_error_10bin": ece,
        "brier_score": float(np.mean(np.square(prob - target.astype(np.float64)))),
        "reliability": rows,
    }


def deterministic_topk_mask(scores: Any, *, count: int) -> np.ndarray:
    values = _finite(scores, name="selection_scores").reshape(-1)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 0 < count < values.size:
        raise Round4RankingError("selection count must be an interior integer")
    flat = np.arange(values.size, dtype=np.int64)
    selected = np.lexsort((flat, -values))[: int(count)]
    mask = np.zeros(values.size, dtype=np.bool_)
    mask[selected] = True
    return mask


def aggregate_mass_localization(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise Round4RankingError("heldout localization requires at least one state")
    total = sum(float(row["exact_costate_l2_square"]) for row in rows)
    retained = sum(float(row["retained_exact_costate_l2_square"]) for row in rows)
    oracle = sum(float(row["oracle_retained_exact_costate_l2_square"]) for row in rows)
    selected = sum(int(row["selected_prefix_cells"]) for row in rows)
    cells = sum(int(row["prefix_cell_count"]) for row in rows)
    fraction = retained / total
    return {
        "state_count": len(rows),
        "selected_prefix_cells": selected,
        "prefix_cell_count": cells,
        "realized_input_area_fraction": selected / cells,
        "exact_costate_l2_square": total,
        "retained_exact_costate_l2_square": retained,
        "oracle_retained_exact_costate_l2_square": oracle,
        "retained_exact_costate_l2_mass_fraction": fraction,
        "oracle_retained_exact_costate_l2_mass_fraction": oracle / total,
        "uplift_over_uniform_area": fraction / (selected / cells),
        "conditional_masked_exact_costate_cosine": math.sqrt(fraction),
        "mean_per_state_mass_fraction": float(
            np.mean([float(row["retained_exact_costate_l2_mass_fraction"]) for row in rows])
        ),
    }


def capture_exact_support_teacher(
    *, segnet: Any, frame_nchw: Any, labels: Any
) -> tuple[Any, Any, np.ndarray, dict[str, float], float]:
    """One exact forward/backward yielding prefix, costate, and ordered pairs."""

    import torch
    import torch.nn.functional as functional

    frame = frame_nchw.detach().requires_grad_(True)
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if "prefix" in captured:
            raise Round4RankingError("registered prefix executed more than once")
        captured["prefix"] = output.detach().clone()

    handle = segnet.encoder.model.blocks[0][0].bn1.register_forward_hook(hook)
    started = time.perf_counter()
    try:
        logits = segnet(frame)
        if "prefix" not in captured:
            raise Round4RankingError("registered prefix hook did not fire")
        loss = functional.cross_entropy(logits, labels)
        input_costate = torch.autograd.grad(loss, frame, retain_graph=False)[0]
    finally:
        handle.remove()
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(input_costate).all()) or not bool(torch.isfinite(logits).all()):
        raise Round4RankingError("exact teacher produced a nonfinite tensor")
    labels_np = labels.detach().cpu().numpy()[0]
    pair_ids = ordered_class_pair_ids(labels_np, logits.detach().cpu().numpy())
    metrics = {
        "ce": float(loss.detach().item()),
        "dseg": float((logits.argmax(1) != labels).float().mean().detach().item()),
    }
    return captured["prefix"], input_costate.detach(), pair_ids, metrics, elapsed


__all__ = [
    "AUTHORITY_SCOPE",
    "BLOCK_FEATURE_COUNT",
    "CLASS_COUNT",
    "GLOBAL_FEATURE_COUNT",
    "ORDERED_PAIR_COUNT",
    "RESEARCH_ONLY",
    "SCHEMA",
    "SOURCE_CLASS_SENSITIVITY",
    "ConvexScalarFit",
    "IsotonicCalibrator",
    "QuadraticStatistics",
    "Round4RankingError",
    "aggregate_mass_localization",
    "aggregate_quadratic_statistics",
    "array_sha256",
    "block_scores",
    "calibrated_block_scores",
    "calibration_diagnostics",
    "capture_exact_support_teacher",
    "deterministic_topk_mask",
    "exact_support_target",
    "fit_block_calibrators",
    "fit_exact_quadratic",
    "fit_isotonic_calibrator",
    "global_scores",
    "mass_localization_metrics",
    "ordered_class_pair_ids",
    "pair_id_to_classes",
    "pairwise_rank_block_statistics",
    "support_feature_matrices",
    "weighted_topk_block_statistics",
    "weighted_topk_statistics",
]
