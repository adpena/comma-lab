# SPDX-License-Identifier: MIT
"""Frozen-replay convex costate head with explicit contraction custody.

This module is deliberately independent of the live witness trainer.  It owns
only the fixed-design ridge objective used by the round-2 replacement probe.
The renderer, SegNet teacher, and exact-metric verdict surfaces remain owned by
the committed task-455 harness.

All fitted weights and predictions use NumPy float32.  Float64 is used only to
measure dot products and the spectrum of the *realized float32 Hessian* so the
reported Euclidean/Frobenius contraction constant is numerically legible.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

SCHEMA = "frozen_replay_convex_head.v1"
AUTHORITY_SCOPE = "numpy-fp32 local training-gradient research evidence; not evaluator score authority"
RESEARCH_ONLY = True

ROUND1_MEASURED_EARLY_COSINE = -0.16153190769629602
ROUND1_POLICY_COSINE_FLOOR = 0.0
# The operator's round-2 rule says "round-1 early-regime bar", which is the
# measured saved-regime comparator.  The old nonnegative admission predicate
# is retained as a diagnostic overlay, not silently substituted as this gate.
DECISION_COSINE_BAR = ROUND1_MEASURED_EARLY_COSINE
FAIL_CLOSED_COSINE_BAR = DECISION_COSINE_BAR
MIN_TEACHER_AMORTIZATION = 5.0


class FrozenReplayError(ValueError):
    """A frozen-replay, convexity, or custody invariant failed."""


def array_sha256(value: Any) -> str:
    """Hash shape, dtype, and C-order bytes rather than bytes alone."""

    array = np.asarray(value)
    header = f"{array.dtype.str}|{array.shape}".encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def _float32(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise FrozenReplayError(f"{name} contains nonfinite values")
    return array


@dataclass(frozen=True)
class ReplayAssignment:
    """One immutable state in the fixed replay distribution."""

    pair_index: int
    checkpoint_index: int
    checkpoint_name: str
    split: Literal["train", "heldout"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_replay_assignments(
    *,
    n_pairs: int,
    checkpoint_names: Sequence[str],
    holdout_period: int,
    seed: int,
) -> tuple[ReplayAssignment, ...]:
    """Stratify all pairs over checkpoints and one held-out residue class.

    For the mission values ``n_pairs=600``, three checkpoints, period five,
    and seed 455, every checkpoint receives exactly 160 train and 40 held-out
    states.  Each pair appears once, so the result is an n600 state cohort,
    not a repeated subset.
    """

    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs < 1:
        raise FrozenReplayError("n_pairs must be an integer >= 1")
    if not checkpoint_names or any(not str(name) for name in checkpoint_names):
        raise FrozenReplayError("checkpoint_names must be nonempty strings")
    if (
        isinstance(holdout_period, bool)
        or not isinstance(holdout_period, int)
        or holdout_period < 2
    ):
        raise FrozenReplayError("holdout_period must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FrozenReplayError("seed must be an integer")
    heldout_residue = seed % holdout_period
    rows = tuple(
        ReplayAssignment(
            pair_index=pair_index,
            checkpoint_index=pair_index % len(checkpoint_names),
            checkpoint_name=str(checkpoint_names[pair_index % len(checkpoint_names)]),
            split="heldout" if pair_index % holdout_period == heldout_residue else "train",
        )
        for pair_index in range(n_pairs)
    )
    if len({row.pair_index for row in rows}) != n_pairs:
        raise FrozenReplayError("replay assignment duplicated a pair")
    return rows


FEATURE_NAMES = (
    "constant",
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "rgb2_r",
    "rgb2_g",
    "rgb2_b",
    "dx_r",
    "dx_g",
    "dx_b",
    "dy_r",
    "dy_g",
    "dy_b",
    "lap_r",
    "lap_g",
    "lap_b",
    "coord_x",
    "coord_y",
    "sin_pi_x",
    "cos_pi_x",
    "sin_pi_y",
    "cos_pi_y",
    "label_0",
    "label_1",
    "label_2",
    "label_3",
    "label_4",
    "tanh_gt_margin",
    "stage_0",
    "stage_1",
    "stage_2",
)


def frozen_feature_matrix(
    frame_nchw: Any,
    labels_hw: Any,
    margins_hw: Any,
    *,
    checkpoint_index: int,
    checkpoint_count: int,
    stride: int,
) -> np.ndarray:
    """Build fixed features that omit the exact target-costate tensor.

    The exact input costate never appears in ``X``.  Ground-truth labels and
    margins do appear: they are source-fixed training inputs, not SegNet outputs
    on the witness.  This is therefore not a claim that ``X`` is independent of
    every signal used to define the costate objective.
    """

    frame = _float32(frame_nchw, name="frame_nchw")
    labels = np.asarray(labels_hw)
    margins = _float32(margins_hw, name="margins_hw")
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise FrozenReplayError("frame_nchw must have shape (3,H,W) or (1,3,H,W)")
    height, width = frame.shape[1:]
    if labels.shape != (height, width) or margins.shape != (height, width):
        raise FrozenReplayError("labels and margins must match the frame grid")
    if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() >= 5:
        raise FrozenReplayError("labels must be integer class ids in [0,5)")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise FrozenReplayError("stride must be an integer >= 1")
    if checkpoint_count != 3 or checkpoint_index not in range(checkpoint_count):
        raise FrozenReplayError("the registered feature chart requires exactly three checkpoint stages")

    rgb = frame / np.float32(255.0) - np.float32(0.5)
    padded = np.pad(rgb, ((0, 0), (1, 1), (1, 1)), mode="edge")
    dx = np.float32(0.5) * (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2])
    dy = np.float32(0.5) * (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1])
    lap = (
        padded[:, 1:-1, 2:]
        + padded[:, 1:-1, :-2]
        + padded[:, 2:, 1:-1]
        + padded[:, :-2, 1:-1]
        - np.float32(4.0) * rgb
    )
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None].repeat(width, axis=1)
    one_hot = np.eye(5, dtype=np.float32)[labels.astype(np.int64)].transpose(2, 0, 1)
    stage = np.zeros((checkpoint_count, height, width), dtype=np.float32)
    stage[checkpoint_index] = np.float32(1.0)
    channels = np.concatenate(
        (
            np.ones((1, height, width), dtype=np.float32),
            rgb,
            np.square(rgb),
            dx,
            dy,
            lap,
            np.stack((x, y, np.sin(np.pi * x), np.cos(np.pi * x), np.sin(np.pi * y), np.cos(np.pi * y))),
            one_hot,
            np.tanh(margins, dtype=np.float32)[None],
            stage,
        ),
        axis=0,
    )
    if channels.shape[0] != len(FEATURE_NAMES):
        raise FrozenReplayError("feature implementation and names disagree")
    sampled = channels[:, ::stride, ::stride].reshape(channels.shape[0], -1).T
    return np.ascontiguousarray(sampled, dtype=np.float32)


def sampled_costate_rows(costate_nchw: Any, *, stride: int) -> np.ndarray:
    costate = _float32(costate_nchw, name="costate_nchw")
    if costate.ndim == 4 and costate.shape[0] == 1:
        costate = costate[0]
    if costate.ndim != 3 or costate.shape[0] != 3:
        raise FrozenReplayError("costate_nchw must have shape (3,H,W) or (1,3,H,W)")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise FrozenReplayError("stride must be an integer >= 1")
    return np.ascontiguousarray(costate[:, ::stride, ::stride].reshape(3, -1).T, dtype=np.float32)


@dataclass(frozen=True)
class StateSufficientStatistics:
    """Lossless cached-label representation for the registered ridge objective."""

    gram: np.ndarray
    rhs: np.ndarray
    target_square_sum: float
    row_count: int
    feature_sha256: str
    target_sha256: str

    def validate(self) -> None:
        gram = _float32(self.gram, name="gram")
        rhs = _float32(self.rhs, name="rhs")
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            raise FrozenReplayError("gram must be square")
        if rhs.shape != (gram.shape[0], 3):
            raise FrozenReplayError("rhs must have shape (n_features,3)")
        if self.row_count < 1 or not math.isfinite(self.target_square_sum) or self.target_square_sum < 0.0:
            raise FrozenReplayError("invalid sufficient-statistic count or target norm")
        if not np.allclose(gram, gram.T, rtol=0.0, atol=8.0 * np.finfo(np.float32).eps):
            raise FrozenReplayError("gram is not symmetric at the fp32 accumulation floor")
        for name, value in (("feature_sha256", self.feature_sha256), ("target_sha256", self.target_sha256)):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise FrozenReplayError(f"{name} is not a lowercase SHA-256")


def cache_exact_label_sufficient_statistics(features: Any, targets: Any) -> StateSufficientStatistics:
    """Cache ``X'X``, ``X'Y``, and ``Y'Y`` after one exact label call."""

    x = _float32(features, name="features")
    y = _float32(targets, name="targets")
    if x.ndim != 2 or y.ndim != 2 or y.shape != (x.shape[0], 3):
        raise FrozenReplayError("features/targets must have aligned shapes (n,d)/(n,3)")
    # Torch/Accelerate may leave stale IEEE exception flags set.  Actual
    # overflow remains fail-closed below through StateSufficientStatistics.
    with np.errstate(all="ignore"):
        gram = np.ascontiguousarray(x.T @ x, dtype=np.float32)
        rhs = np.ascontiguousarray(x.T @ y, dtype=np.float32)
    target_square_sum = float(np.sum(np.square(y.astype(np.float64))))
    result = StateSufficientStatistics(
        gram=gram,
        rhs=rhs,
        target_square_sum=target_square_sum,
        row_count=x.shape[0],
        feature_sha256=array_sha256(x),
        target_sha256=array_sha256(y),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class ReplaySufficientStatistics:
    gram: np.ndarray
    rhs: np.ndarray
    target_square_sum: float
    row_count: int
    state_count: int
    per_state_gram: np.ndarray
    per_state_rhs: np.ndarray
    per_state_rows: np.ndarray

    def validate(self) -> None:
        gram = _float32(self.gram, name="gram")
        rhs = _float32(self.rhs, name="rhs")
        per_gram = _float32(self.per_state_gram, name="per_state_gram")
        per_rhs = _float32(self.per_state_rhs, name="per_state_rhs")
        rows = np.asarray(self.per_state_rows)
        feature_count = gram.shape[0]
        if gram.shape != (feature_count, feature_count) or rhs.shape != (feature_count, 3):
            raise FrozenReplayError("aggregate sufficient-statistic shapes disagree")
        if per_gram.shape != (self.state_count, feature_count, feature_count):
            raise FrozenReplayError("per-state gram shape disagrees")
        if per_rhs.shape != (self.state_count, feature_count, 3) or rows.shape != (self.state_count,):
            raise FrozenReplayError("per-state rhs/count shapes disagree")
        if self.state_count < 1 or self.row_count != int(rows.sum()) or np.any(rows < 1):
            raise FrozenReplayError("per-state row counts do not reconcile")


def aggregate_sufficient_statistics(records: Sequence[StateSufficientStatistics]) -> ReplaySufficientStatistics:
    if not records:
        raise FrozenReplayError("at least one cached label record is required")
    for record in records:
        record.validate()
    feature_count = records[0].gram.shape[0]
    if any(record.gram.shape != (feature_count, feature_count) for record in records):
        raise FrozenReplayError("cached label records use different feature charts")
    per_gram = np.stack([record.gram for record in records]).astype(np.float32, copy=False)
    per_rhs = np.stack([record.rhs for record in records]).astype(np.float32, copy=False)
    per_rows = np.asarray([record.row_count for record in records], dtype=np.int64)
    # The realized authority matrix is explicitly fp32.
    gram = np.sum(per_gram, axis=0, dtype=np.float32)
    rhs = np.sum(per_rhs, axis=0, dtype=np.float32)
    result = ReplaySufficientStatistics(
        gram=np.ascontiguousarray(gram),
        rhs=np.ascontiguousarray(rhs),
        target_square_sum=float(sum(record.target_square_sum for record in records)),
        row_count=int(per_rows.sum()),
        state_count=len(records),
        per_state_gram=np.ascontiguousarray(per_gram),
        per_state_rhs=np.ascontiguousarray(per_rhs),
        per_state_rows=per_rows,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class ContractionCertificate:
    norm: str
    data_curvature_min: float
    data_curvature_max: float
    ridge_lambda: float
    mu: float
    smoothness_L: float
    ideal_step_size_eta: float
    step_size_eta: float
    ideal_contraction_gamma: float
    contraction_gamma: float
    ideal_gamma_upper_bound: float
    fp32_step_rounding_slack: float
    hessian_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_contraction_certificate(
    stats: ReplaySufficientStatistics,
) -> tuple[np.ndarray, np.ndarray, ContractionCertificate]:
    """Seal the realized fp32 ridge/step and derive its operator norm."""

    stats.validate()
    gram_mean = np.ascontiguousarray(stats.gram / np.float32(stats.row_count), dtype=np.float32)
    rhs_mean = np.ascontiguousarray(stats.rhs / np.float32(stats.row_count), dtype=np.float32)
    data_eigenvalues = np.linalg.eigvalsh(gram_mean.astype(np.float64))
    data_lmin, data_lmax = float(data_eigenvalues[0]), float(data_eigenvalues[-1])
    if not math.isfinite(data_lmax) or data_lmax <= 0.0:
        raise FrozenReplayError("feature covariance has no positive spectral scale")
    ridge_fp32 = np.float32(data_lmax)
    if float(ridge_fp32) < data_lmax:
        ridge_fp32 = np.nextafter(ridge_fp32, np.float32(np.inf))
    ridge = float(ridge_fp32)
    hessian = np.ascontiguousarray(
        gram_mean + np.float32(ridge) * np.eye(gram_mean.shape[0], dtype=np.float32),
        dtype=np.float32,
    )
    eigenvalues = np.linalg.eigvalsh(hessian.astype(np.float64))
    mu, smoothness = float(eigenvalues[0]), float(eigenvalues[-1])
    if mu <= 0.0 or smoothness < mu:
        raise FrozenReplayError("ridge Hessian is not strongly convex")
    ideal_eta = 2.0 / (mu + smoothness)
    eta = float(np.float32(ideal_eta))
    ideal_gamma = (smoothness - mu) / (smoothness + mu)
    gamma = max(abs(1.0 - eta * mu), abs(1.0 - eta * smoothness))
    # In real arithmetic, lambda=lambda_max(G) gives kappa(H)<=2 and
    # gamma_ideal<=1/3.  The executed fp32 eta has a separately reported
    # rounding slack; its realized linear-map norm must still be contractive.
    ideal_upper = 1.0 / 3.0
    rounding_slack = max(0.0, gamma - ideal_gamma)
    if ideal_gamma > ideal_upper + 64.0 * np.finfo(np.float32).eps:
        raise FrozenReplayError("spectral-scale ridge violated the ideal one-third bound")
    if not 0.0 <= gamma < 1.0:
        raise FrozenReplayError("realized fp32 update is not a contraction")
    certificate = ContractionCertificate(
        norm="Euclidean parameter norm / Frobenius head norm",
        data_curvature_min=data_lmin,
        data_curvature_max=data_lmax,
        ridge_lambda=ridge,
        mu=mu,
        smoothness_L=smoothness,
        ideal_step_size_eta=ideal_eta,
        step_size_eta=eta,
        ideal_contraction_gamma=ideal_gamma,
        contraction_gamma=gamma,
        ideal_gamma_upper_bound=ideal_upper,
        fp32_step_rounding_slack=rounding_slack,
        hessian_sha256=array_sha256(hessian),
    )
    return hessian, rhs_mean, certificate


def _objective(
    stats: ReplaySufficientStatistics,
    weights: np.ndarray,
    *,
    hessian: np.ndarray,
    rhs_mean: np.ndarray,
) -> float:
    w = weights.astype(np.float64)
    return 0.5 * (
        float(np.sum(w * (hessian @ w)))
        - 2.0 * float(np.sum(w * rhs_mean))
        + stats.target_square_sum / float(stats.row_count)
    )


@dataclass(frozen=True)
class ConvexHeadFit:
    weights: np.ndarray
    optimum_weights: np.ndarray
    certificate: ContractionCertificate
    trace: tuple[dict[str, float | int | None], ...]
    terminal_gradient_norm: float
    actual_parameter_residual: float
    residual_parameter_bound: float
    actual_prediction_rmse_residual: float
    residual_prediction_rmse_bound: float
    actual_objective_gap: float
    objective_gap_bound: float
    residual_bounds_validated: bool
    per_state_gradient_variance: float
    per_state_gradient_second_moment: float
    initial_parameter_error_norm: float
    parameter_ratio_numeric_floor: float
    initial_objective_gap: float
    objective_ratio_numeric_floor: float


def fit_cached_convex_head(stats: ReplaySufficientStatistics, *, epochs: int) -> ConvexHeadFit:
    """Run deterministic full-batch fp32 GD on cached sufficient statistics."""

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise FrozenReplayError("epochs must be an integer >= 1")
    hessian, rhs_mean, certificate = derive_contraction_certificate(stats)
    hessian64 = hessian.astype(np.float64)
    rhs64 = rhs_mean.astype(np.float64)
    gram_mean64 = np.ascontiguousarray(
        stats.gram / np.float32(stats.row_count), dtype=np.float32
    ).astype(np.float64)
    optimum = np.linalg.solve(hessian64, rhs64)
    weights = np.zeros(rhs_mean.shape, dtype=np.float32)
    optimum_objective = _objective(
        stats,
        optimum,
        hessian=hessian64,
        rhs_mean=rhs64,
    )
    trace: list[dict[str, float | int | None]] = []
    previous_distance = float(np.linalg.norm((weights - optimum).astype(np.float64)))
    previous_gap = max(
        0.0,
        _objective(
            stats,
            weights,
            hessian=hessian64,
            rhs_mean=rhs64,
        )
        - optimum_objective,
    )
    initial_parameter_error = previous_distance
    initial_objective_gap = previous_gap
    # A ratio-admission floor must carry the units and scale of its numerator.
    # An absolute ``128*eps32`` floor incorrectly suppresses every ratio when a
    # valid convex problem has a subunit optimum (the real n600 costate head is
    # one such case).  The fixed initial scales keep admission deterministic
    # across all epochs while remaining invariant to a joint rescaling.
    parameter_numeric_floor = 128.0 * np.finfo(np.float32).eps * max(
        initial_parameter_error,
        float(np.finfo(np.float32).tiny),
    )
    objective_numeric_floor = 128.0 * np.finfo(np.float32).eps * max(
        initial_objective_gap,
        float(np.finfo(np.float64).tiny),
    )
    eta = np.float32(certificate.step_size_eta)
    for epoch in range(epochs):
        gradient = np.ascontiguousarray(hessian @ weights - rhs_mean, dtype=np.float32)
        weights = np.ascontiguousarray(weights - eta * gradient, dtype=np.float32)
        distance = float(np.linalg.norm((weights - optimum).astype(np.float64)))
        gap = max(
            0.0,
            _objective(
                stats,
                weights,
                hessian=hessian64,
                rhs_mean=rhs64,
            )
            - optimum_objective,
        )
        trace.append(
            {
                "epoch": epoch + 1,
                "parameter_error_norm": distance,
                "parameter_contraction_ratio": (
                    None
                    if previous_distance <= parameter_numeric_floor
                    else distance / previous_distance
                ),
                "objective_gap": gap,
                "objective_gap_ratio": (
                    None if previous_gap <= objective_numeric_floor else gap / previous_gap
                ),
            }
        )
        previous_distance, previous_gap = distance, gap

    terminal_gradient = hessian64 @ weights.astype(np.float64) - rhs64
    gradient_norm = float(np.linalg.norm(terminal_gradient))
    prediction_lipschitz = math.sqrt(max(0.0, certificate.data_curvature_max))
    parameter_bound = gradient_norm / certificate.mu
    prediction_bound = prediction_lipschitz * parameter_bound
    objective_gap_bound = gradient_norm * gradient_norm / (2.0 * certificate.mu)
    weight_error = weights.astype(np.float64) - optimum
    actual_parameter_residual = float(np.linalg.norm(weight_error))
    prediction_square = float(np.sum(weight_error * (gram_mean64 @ weight_error)))
    actual_prediction_residual = math.sqrt(max(0.0, prediction_square))
    actual_objective_gap = max(
        0.0,
        _objective(
            stats,
            weights,
            hessian=hessian64,
            rhs_mean=rhs64,
        )
        - optimum_objective,
    )
    numeric_slack = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        actual_parameter_residual,
        parameter_bound,
        actual_prediction_residual,
        prediction_bound,
        actual_objective_gap,
        objective_gap_bound,
    )
    if actual_parameter_residual > parameter_bound + numeric_slack:
        raise FrozenReplayError("fp64 parameter residual violated the strong-convexity bound")
    if actual_prediction_residual > prediction_bound + numeric_slack:
        raise FrozenReplayError("fp64 prediction residual violated the fidelity bound")
    if actual_objective_gap > objective_gap_bound + numeric_slack:
        raise FrozenReplayError("fp64 objective gap violated the strong-convexity bound")

    state_gradients = []
    for gram, rhs, rows in zip(stats.per_state_gram, stats.per_state_rhs, stats.per_state_rows, strict=True):
        gram_state_mean = np.ascontiguousarray(
            gram / np.float32(rows), dtype=np.float32
        ).astype(np.float64)
        rhs_state_mean = np.ascontiguousarray(
            rhs / np.float32(rows), dtype=np.float32
        ).astype(np.float64)
        state_gradient = (
            gram_state_mean @ weights.astype(np.float64)
            - rhs_state_mean
            + certificate.ridge_lambda * weights.astype(np.float64)
        )
        state_gradients.append(np.ascontiguousarray(state_gradient, dtype=np.float64))
    stacked = np.stack(state_gradients)
    mean_gradient = stacked.mean(axis=0)
    variance = float(np.mean(np.sum(np.square(stacked - mean_gradient), axis=(1, 2))))
    second_moment = float(np.mean(np.sum(np.square(stacked), axis=(1, 2))))
    return ConvexHeadFit(
        weights=weights,
        optimum_weights=optimum,
        certificate=certificate,
        trace=tuple(trace),
        terminal_gradient_norm=gradient_norm,
        actual_parameter_residual=actual_parameter_residual,
        residual_parameter_bound=parameter_bound,
        actual_prediction_rmse_residual=actual_prediction_residual,
        residual_prediction_rmse_bound=prediction_bound,
        actual_objective_gap=actual_objective_gap,
        objective_gap_bound=objective_gap_bound,
        residual_bounds_validated=True,
        per_state_gradient_variance=variance,
        per_state_gradient_second_moment=second_moment,
        initial_parameter_error_norm=initial_parameter_error,
        parameter_ratio_numeric_floor=parameter_numeric_floor,
        initial_objective_gap=initial_objective_gap,
        objective_ratio_numeric_floor=objective_numeric_floor,
    )


def predict_costate(features: Any, weights: Any, *, height: int, width: int) -> np.ndarray:
    x = _float32(features, name="features")
    w = _float32(weights, name="weights")
    if x.ndim != 2 or w.shape != (x.shape[1], 3):
        raise FrozenReplayError("feature/head shapes disagree")
    if x.shape[0] != height * width:
        raise FrozenReplayError("full-grid feature row count disagrees with height*width")
    # Torch may leave stale IEEE exception flags after the exact teacher.  Do
    # not turn those flags into false NumPy warnings, but still fail closed on
    # any actual non-finite prediction.
    with np.errstate(all="ignore"):
        rows = np.ascontiguousarray(x @ w, dtype=np.float32)
    if not np.all(np.isfinite(rows)):
        raise FrozenReplayError("predicted costate contains nonfinite values")
    return np.ascontiguousarray(rows.T.reshape(1, 3, height, width), dtype=np.float32)


def vector_fidelity(reference: Any, candidate: Any) -> dict[str, float | int | None | bool]:
    """Costate or renderer-gradient fidelity with fp64 reduction custody."""

    ref = _float32(reference, name="reference").astype(np.float64).reshape(-1)
    cand = _float32(candidate, name="candidate").astype(np.float64).reshape(-1)
    if ref.shape != cand.shape:
        raise FrozenReplayError("fidelity vectors have different shapes")
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    dot = float(np.dot(ref, cand))
    cosine = None if ref_norm == 0.0 or cand_norm == 0.0 else dot / (ref_norm * cand_norm)
    relative_l2 = None if ref_norm == 0.0 else float(np.linalg.norm(cand - ref) / ref_norm)
    return {
        "compared_elements": int(ref.size),
        "dot": dot,
        "cosine_similarity": cosine,
        "relative_l2_error": relative_l2,
        "reference_norm": ref_norm,
        "candidate_norm": cand_norm,
        "finite": True,
    }


def teacher_call_accounting(
    *,
    naive_teacher_calls: int,
    fresh_anchor_samples: int,
    paired_difference_samples: int,
    exact_labels_per_difference: int,
    observed_teacher_forwards: int,
) -> dict[str, Any]:
    """Instantiate ``C_teacher=A+c_label*D`` without dropping cache-build calls."""

    values = {
        "naive_teacher_calls": naive_teacher_calls,
        "fresh_anchor_samples": fresh_anchor_samples,
        "paired_difference_samples": paired_difference_samples,
        "exact_labels_per_difference": exact_labels_per_difference,
        "observed_teacher_forwards": observed_teacher_forwards,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise FrozenReplayError("teacher-call counts must be nonnegative integers")
    if naive_teacher_calls < 1:
        raise FrozenReplayError("naive teacher baseline must be positive")
    derived = fresh_anchor_samples + exact_labels_per_difference * paired_difference_samples
    if derived != observed_teacher_forwards:
        raise FrozenReplayError("hook-observed teacher calls do not reconcile with A+c_label*D")
    amortization = naive_teacher_calls / derived if derived else math.inf
    return {
        "law": "C_teacher = A + c_label * D",
        "naive_teacher_calls_N": naive_teacher_calls,
        "fresh_anchor_samples_A": fresh_anchor_samples,
        "paired_difference_samples_D": paired_difference_samples,
        "exact_labels_per_difference_c_label": exact_labels_per_difference,
        "derived_C_teacher": derived,
        "observed_teacher_forwards": observed_teacher_forwards,
        "teacher_calls_per_effective_training_step": derived / naive_teacher_calls,
        "teacher_call_amortization_x": amortization,
        "saving_calls": naive_teacher_calls - derived,
        "saving_fraction": 1.0 - derived / naive_teacher_calls,
        "reconciliation": "PASS",
    }


def derive_mission_verdict(*, heldout_costate_cosine: float, teacher_call_amortization_x: float) -> dict[str, Any]:
    if not math.isfinite(heldout_costate_cosine) or not math.isfinite(teacher_call_amortization_x):
        raise FrozenReplayError("decision metrics must be finite")
    cosine_pass = heldout_costate_cosine >= DECISION_COSINE_BAR
    amortization_pass = teacher_call_amortization_x >= MIN_TEACHER_AMORTIZATION
    return {
        "verdict": "GO" if cosine_pass and amortization_pass else "NO-GO",
        "heldout_costate_cosine": heldout_costate_cosine,
        "round1_measured_early_cosine": ROUND1_MEASURED_EARLY_COSINE,
        "round1_policy_cosine_floor": ROUND1_POLICY_COSINE_FLOOR,
        "operator_literal_early_regime_cosine_bar": DECISION_COSINE_BAR,
        "legacy_nonnegative_policy_overlay_pass": (
            heldout_costate_cosine >= ROUND1_POLICY_COSINE_FLOOR
        ),
        "legacy_nonnegative_policy_overlay_is_decision_gate": False,
        "cosine_gate_pass": cosine_pass,
        "teacher_call_amortization_x": teacher_call_amortization_x,
        "minimum_teacher_call_amortization_x": MIN_TEACHER_AMORTIZATION,
        "amortization_gate_pass": amortization_pass,
        "verdict_scope": (
            "FORMULATION x INSTANCE — fixed three-checkpoint V9 n600 replay distribution, deterministic "
            "RGB/geometry/GT feature chart, full-batch spectral-scale ridge head, seed455, local macOS CPU; "
            "not the nonlinear on-policy family, live activation, evaluator score, contest-CPU, or CUDA"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "AUTHORITY_SCOPE",
    "DECISION_COSINE_BAR",
    "FAIL_CLOSED_COSINE_BAR",
    "FEATURE_NAMES",
    "MIN_TEACHER_AMORTIZATION",
    "RESEARCH_ONLY",
    "ROUND1_MEASURED_EARLY_COSINE",
    "ROUND1_POLICY_COSINE_FLOOR",
    "ContractionCertificate",
    "ConvexHeadFit",
    "FrozenReplayError",
    "ReplayAssignment",
    "ReplaySufficientStatistics",
    "StateSufficientStatistics",
    "aggregate_sufficient_statistics",
    "array_sha256",
    "cache_exact_label_sufficient_statistics",
    "derive_contraction_certificate",
    "derive_mission_verdict",
    "deterministic_replay_assignments",
    "fit_cached_convex_head",
    "frozen_feature_matrix",
    "predict_costate",
    "sampled_costate_rows",
    "teacher_call_accounting",
    "vector_fidelity",
]
