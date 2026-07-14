# SPDX-License-Identifier: MIT
"""Deep warm-start mechanisms for the n=1 costate organ.

This module implements the ``U_hierarchical_physics_residual`` research arm.  It is
deliberately organ-side and advisory: it has no trainer, provider, subprocess, archive,
or witness-DSL surface.  NumPy-fp32 is the reference implementation; MLX is an optional
parity backend.

The statistical object is a conjugate block-ridge posterior around the existing fixed
Q/P physics-prior modes.  The structural response prior is the persistent stream, while
the intercept/state residual is the prediction stream.  This is predictive partial
pooling, not causal attribution and not an RL policy.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from tac.witness_control.lambda_net import (
    N_CLASSES,
    PHI_DIM,
    STATE_DIM,
    CampaignTrajectory,
    Interval,
    _predict_interval,
    build_intervals,
    fit_score_composition,
    lever_features,
    make_model,
)

RECEIPT_SCHEMA = "costate_warmstart_cluster_backtest.v1"
PRIOR_MODES = ("Q_priormean_iso", "P_priormean_aniso")
PRECISION_GRID = (0.01, 0.1, 1.0, 10.0)
# intercept / state drift / class response / other response
BLOCK_MULTIPLIERS = (0.25, 8.0, 1.0, 16.0)
GRADUATION_MIN_RECORDS = 3
MIN_INNER_SELECTION_FOLDS = 4


def _fp32(a: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _design_rows(intervals: Sequence[Interval], phis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, targets = [], []
    for iv in intervals:
        occupancy = _fp32(phis).T @ _fp32(iv.u_mean)
        rows.append(np.concatenate((np.ones(1, dtype=np.float32), _fp32(iv.x0), occupancy)))
        targets.append(_fp32(iv.dxdt()))
    if not rows:
        raise ValueError("at least one interval is required")
    return np.stack(rows).astype(np.float32), np.stack(targets).astype(np.float32)


def _block_diagonal(n_features: int, multipliers: Sequence[float]) -> np.ndarray:
    if len(multipliers) != 4 or any(float(v) <= 0.0 for v in multipliers):
        raise ValueError("block multipliers must contain four positive values")
    expected = 1 + STATE_DIM + PHI_DIM
    if n_features != expected:
        raise ValueError(f"expected {expected} design features, got {n_features}")
    d = np.empty(n_features, dtype=np.float32)
    d[0] = float(multipliers[0])
    d[1:1 + STATE_DIM] = float(multipliers[1])
    occ0 = 1 + STATE_DIM
    d[occ0:occ0 + N_CLASSES] = float(multipliers[2])
    d[occ0 + N_CLASSES:] = float(multipliers[3])
    return d


def _prior_matrix(mode: str, supplied: np.ndarray | None = None) -> np.ndarray:
    if supplied is not None:
        m = _fp32(supplied)
    elif mode == "Q_priormean_iso":
        m = np.eye(N_CLASSES, dtype=np.float32)
    elif mode == "P_priormean_aniso":
        # Lazy import keeps the Q arm cheap and reuses the existing content-custodied
        # physics profiles rather than re-deriving them here.
        from tac.witness_control.aniso_perclass_lambda import (
            aniso_coupled_m0,
            measure_aniso_class_profiles,
            smoothed_grad_per_class,
        )

        m = _fp32(aniso_coupled_m0(
            measure_aniso_class_profiles(), smoothed_grad_per_class()))
    else:
        raise ValueError(f"unknown prior mode {mode!r}; choose from {PRIOR_MODES}")
    if m.shape != (N_CLASSES, N_CLASSES) or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("physics prior matrix must be finite, nonnegative, and shaped (5,5)")
    total = float(m.sum())
    if total <= 0.0:
        raise ValueError("physics prior matrix has zero mass")
    return (m * np.float32(N_CLASSES / total)).astype(np.float32)


def physics_prior_coefficients(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    mode: str,
    prior_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Construct the fixed Q/P structural coefficient prior and its one-dof gain.

    This intentionally matches the existing P/Q gauge and robust median drift.  Only
    the direction is physics supplied; the nonnegative global gain is estimated from
    the prefix.  No held-out row is consumed.
    """
    z, y = _fp32(design), _fp32(targets)
    m0 = _prior_matrix(mode, prior_matrix)
    b0 = np.median(y, axis=0).astype(np.float32)
    occ0 = 1 + STATE_DIM
    occ_class = z[:, occ0:occ0 + N_CLASSES]
    drive = -(occ_class @ m0.T)
    resid = y[:, :N_CLASSES] - b0[None, :N_CLASSES]
    den = float(np.sum(drive * drive, dtype=np.float64))
    num = float(np.sum(drive * resid, dtype=np.float64))
    kappa = max(num / den, 0.0) if den > 0.0 else 0.0
    coef0 = np.zeros((z.shape[1], STATE_DIM), dtype=np.float32)
    coef0[0] = b0
    for response_class in range(N_CLASSES):
        for target_class in range(N_CLASSES):
            coef0[occ0 + target_class, response_class] = np.float32(
                -kappa * float(m0[response_class, target_class]))
    return coef0, float(kappa)


@dataclass(frozen=True)
class PosteriorSolve:
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray  # (state, feature, feature)
    noise_variance: np.ndarray
    precision: float
    effective_degrees_of_freedom: float
    condition_number: float
    backend: str

    def predictive_variance(self, row: np.ndarray) -> np.ndarray:
        z = _fp32(row)
        epistemic = np.asarray(
            [z @ self.coefficient_covariance[c] @ z for c in range(STATE_DIM)],
            dtype=np.float32,
        )
        return np.maximum(epistemic + self.noise_variance, 0.0)

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "effective_degrees_of_freedom": self.effective_degrees_of_freedom,
            "condition_number": self.condition_number,
            "noise_variance": [float(v) for v in self.noise_variance],
            "backend": self.backend,
        }


def posterior_solve_numpy_fp32(
    design: np.ndarray,
    targets: np.ndarray,
    prior: np.ndarray,
    *,
    precision: float,
    block_multipliers: Sequence[float] = BLOCK_MULTIPLIERS,
) -> PosteriorSolve:
    """Reference conjugate block-ridge posterior (NumPy-fp32 authority)."""
    z, y, c0 = _fp32(design), _fp32(targets), _fp32(prior)
    if z.ndim != 2 or y.ndim != 2 or c0.shape != (z.shape[1], y.shape[1]):
        raise ValueError("incompatible design, target, and prior shapes")
    if y.shape[1] != STATE_DIM or precision <= 0.0:
        raise ValueError("targets must have STATE_DIM columns and precision must be positive")
    gram = (z.T @ z).astype(np.float32)
    scale = float(np.mean(np.diag(gram), dtype=np.float64)) or 1.0
    diag = _block_diagonal(z.shape[1], block_multipliers)
    prior_precision = np.diag(diag * np.float32(float(precision) * scale)).astype(np.float32)
    system = (gram + prior_precision).astype(np.float32)
    rhs = (z.T @ y + prior_precision @ c0).astype(np.float32)
    coefficients = np.linalg.solve(system, rhs).astype(np.float32)
    inverse = np.linalg.inv(system).astype(np.float32)
    inverse = ((inverse + inverse.T) * np.float32(0.5)).astype(np.float32)
    residual = y - z @ coefficients
    noise = np.maximum(np.mean(residual * residual, axis=0), np.float32(1e-12)).astype(np.float32)
    covariance = np.stack([inverse * v for v in noise]).astype(np.float32)
    edf = float(np.trace(z @ inverse @ z.T))
    return PosteriorSolve(
        coefficients=coefficients,
        coefficient_covariance=covariance,
        noise_variance=noise,
        precision=float(precision),
        effective_degrees_of_freedom=edf,
        condition_number=float(np.linalg.cond(system.astype(np.float64))),
        backend="numpy-fp32",
    )


def posterior_solve_mlx_fp32(
    design: np.ndarray,
    targets: np.ndarray,
    prior: np.ndarray,
    *,
    precision: float,
    block_multipliers: Sequence[float] = BLOCK_MULTIPLIERS,
) -> PosteriorSolve:
    """MLX implementation of the same posterior; raises ImportError when unavailable."""
    import mlx.core as mx

    z_np, y_np, c0_np = _fp32(design), _fp32(targets), _fp32(prior)
    gram_np = (z_np.T @ z_np).astype(np.float32)
    scale = float(np.mean(np.diag(gram_np), dtype=np.float64)) or 1.0
    diag_np = _block_diagonal(z_np.shape[1], block_multipliers)
    p_np = np.diag(diag_np * np.float32(float(precision) * scale)).astype(np.float32)
    z, y, c0, p = map(mx.array, (z_np, y_np, c0_np, p_np))
    system = z.T @ z + p
    coefficients_mx = mx.linalg.solve(system, z.T @ y + p @ c0)
    inverse_mx = mx.linalg.inv(system)
    mx.eval(coefficients_mx, inverse_mx)
    coefficients = np.asarray(coefficients_mx, dtype=np.float32)
    inverse = np.asarray(inverse_mx, dtype=np.float32)
    inverse = ((inverse + inverse.T) * np.float32(0.5)).astype(np.float32)
    residual = y_np - z_np @ coefficients
    noise = np.maximum(np.mean(residual * residual, axis=0), np.float32(1e-12)).astype(np.float32)
    covariance = np.stack([inverse * v for v in noise]).astype(np.float32)
    edf = float(np.trace(z_np @ inverse @ z_np.T))
    return PosteriorSolve(
        coefficients=coefficients,
        coefficient_covariance=covariance,
        noise_variance=noise,
        precision=float(precision),
        effective_degrees_of_freedom=edf,
        condition_number=float(np.linalg.cond((gram_np + p_np).astype(np.float64))),
        backend="mlx-fp32",
    )


def clipped_difference_targets(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    lipschitz_bound: float,
) -> tuple[np.ndarray, dict]:
    """VR-GHAL-inspired pathwise clipping diagnostic.

    The transfer is intentionally narrow: clip *differences* at ``L ||z_t-z_(t-1)||``.
    It does not claim VR-GHAL's theorem because these rows are not repeated unbiased
    stochastic-operator queries.
    """
    if lipschitz_bound <= 0.0:
        raise ValueError("lipschitz_bound must be positive")
    z, y = _fp32(design), _fp32(targets)
    out = y.copy()
    clipped = 0
    ratios = []
    for t in range(1, len(y)):
        delta, dz = y[t] - y[t - 1], float(np.linalg.norm(z[t] - z[t - 1]))
        radius = float(lipschitz_bound) * dz
        norm = float(np.linalg.norm(delta))
        ratios.append(norm / max(dz, 1e-12))
        if norm > radius and norm > 0.0:
            delta = delta * np.float32(radius / norm)
            clipped += 1
        out[t] = out[t - 1] + delta
    return out, {
        "status": "DIAGNOSTIC_ONLY_NO_STOCHASTIC_ORACLE_THEOREM",
        "lipschitz_bound": float(lipschitz_bound),
        "n_differences": max(len(y) - 1, 0),
        "n_clipped": clipped,
        "observed_ratio_max": max(ratios, default=0.0),
    }


def prefix_lipschitz_bound(design: np.ndarray, targets: np.ndarray, quantile: float = 0.75) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0,1]")
    z, y = _fp32(design), _fp32(targets)
    ratios = []
    for t in range(1, len(y)):
        dz = float(np.linalg.norm(z[t] - z[t - 1]))
        if dz > 0.0:
            ratios.append(float(np.linalg.norm(y[t] - y[t - 1])) / dz)
    return max(float(np.quantile(ratios, quantile)) if ratios else 1.0, 1e-8)


@dataclass(frozen=True)
class SupportCertificate:
    n_intervals: int
    occupancy_ambient_dim: int
    occupancy_rank: int
    occupancy_condition: float
    positive_variation_fraction: float
    fore_status: str
    tofu_status: str
    hcm_causal_status: str
    rl_actor_status: str

    @property
    def causally_identified(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {**self.__dict__, "causally_identified": False}


def support_certificate(intervals: Sequence[Interval], phis: np.ndarray) -> SupportCertificate:
    if not intervals:
        raise ValueError("support certificate needs at least one interval")
    occupancy = np.stack([_fp32(phis).T @ _fp32(iv.u_mean) for iv in intervals])
    centered = occupancy - occupancy.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered.astype(np.float64), compute_uv=False)
    tol = max(centered.shape) * np.finfo(np.float64).eps * (singular[0] if singular.size else 0.0)
    nz = singular[singular > tol]
    rank = len(nz)
    condition = float(nz[0] / nz[-1]) if len(nz) > 1 else (1.0 if len(nz) == 1 else math.inf)
    varying = np.std(occupancy, axis=0) > 1e-6
    return SupportCertificate(
        n_intervals=len(intervals),
        occupancy_ambient_dim=int(occupancy.shape[1]),
        occupancy_rank=rank,
        occupancy_condition=condition,
        positive_variation_fraction=float(np.mean(varying)),
        fore_status="BLOCKED_DISTRIBUTION_CUSTODY_NO_BEHAVIOR_TARGET_RATIOS",
        tofu_status="BLOCKED_PARTIAL_ACTION_CUSTODY_NO_MASKED_OFFERED_SLATES",
        hcm_causal_status="NOT_IDENTIFIED_NO_WITHIN_UNIT_RANDOMIZED_TREATMENT",
        rl_actor_status="DISABLED_NO_PROPENSITIES_EXECUTED_DECISION_ROWS",
    )


class HierarchicalPhysicsResidualAdjoint:
    """U arm: Q/P structural prior plus conjugate block-partial-pooling posterior."""

    name = "U_hierarchical_physics_residual"

    def __init__(
        self,
        *,
        prior_mode: str = "Q_priormean_iso",
        precision: float = 1.0,
        block_multipliers: Sequence[float] = BLOCK_MULTIPLIERS,
        prior_matrix: np.ndarray | None = None,
        difference_clip: bool = False,
    ):
        if prior_mode not in PRIOR_MODES:
            raise ValueError(f"unknown prior mode {prior_mode!r}")
        self.prior_mode = prior_mode
        self.precision = float(precision)
        self.block_multipliers = tuple(float(v) for v in block_multipliers)
        self.prior_matrix = prior_matrix
        self.difference_clip = bool(difference_clip)
        self.posterior: PosteriorSolve | None = None
        self.kappa: float | None = None
        self.clip_diagnostic: dict | None = None

    def fit(self, intervals: Sequence[Interval], phis: np.ndarray, seed: int = 0) -> None:
        del seed  # deterministic closed-form solve
        z, y = _design_rows(intervals, phis)
        prior, self.kappa = physics_prior_coefficients(
            z, y, mode=self.prior_mode, prior_matrix=self.prior_matrix)
        if self.difference_clip and len(y) > 1:
            bound = prefix_lipschitz_bound(z, y)
            y, self.clip_diagnostic = clipped_difference_targets(z, y, lipschitz_bound=bound)
        else:
            self.clip_diagnostic = {
                "status": "DISABLED_DEFAULT_NO_STOCHASTIC_ORACLE_CUSTODY",
                "n_clipped": 0,
            }
        self.posterior = posterior_solve_numpy_fp32(
            z, y, prior, precision=self.precision,
            block_multipliers=self.block_multipliers)

    def _check(self) -> PosteriorSolve:
        if self.posterior is None:
            raise RuntimeError("fit must be called first")
        return self.posterior

    def base(self, x: np.ndarray, ctx: np.ndarray, path: np.ndarray | None = None) -> np.ndarray:
        del ctx, path
        p = self._check().coefficients
        return (p[0] + p[1:1 + STATE_DIM].T @ _fp32(x)).astype(np.float32)

    def response(
        self, x: np.ndarray, ctx: np.ndarray, phi: np.ndarray,
        path: np.ndarray | None = None,
    ) -> np.ndarray:
        del x, ctx, path
        p = self._check().coefficients
        return (p[1 + STATE_DIM:].T @ _fp32(phi)).astype(np.float32)

    def predict_interval(self, interval: Interval, lever_names: Sequence[str]) -> np.ndarray:
        pred = self.base(interval.x0, interval.ctx, interval.path)
        responses = np.stack([
            self.response(interval.x0, interval.ctx, lever_features(name), interval.path)
            for name in lever_names
        ])
        return (pred + responses.T @ _fp32(interval.u_mean)).astype(np.float32)

    def response_variance(self, phi: np.ndarray) -> np.ndarray:
        post = self._check()
        start = 1 + STATE_DIM
        v = _fp32(phi)
        return np.asarray([
            v @ post.coefficient_covariance[c, start:, start:] @ v
            for c in range(STATE_DIM)
        ], dtype=np.float32)


@dataclass(frozen=True)
class FoldResult:
    hold: int
    ep1: float
    prior_mode: str
    precision: float
    inner_mae: float | None
    error: float
    persistence_error: float
    perclass_error: float
    persistence_perclass_error: float
    effective_degrees_of_freedom: float
    posterior_condition: float
    support: SupportCertificate
    aggregate_forecaster: str | None = None
    aggregate_constraint_delta_norm: float = 0.0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["support"] = self.support.to_dict()
        return d


@dataclass(frozen=True)
class WarmstartBacktest:
    architecture: str
    prior_modes: tuple[str, ...]
    precision_grid: tuple[float, ...]
    n_folds: int
    walkforward_mae_model: float
    walkforward_mae_persistence: float
    walkforward_perclass_mae_model: float
    walkforward_perclass_mae_persistence: float
    fold_wins: int
    fold_losses: int
    fold_ties: int
    sign_test_p: float
    folds: tuple[FoldResult, ...]
    field_diagnostics: dict
    continual_learning_status: str
    verdict_scope: str = "INSTANCE_X_FORMULATION"
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    score_claim: bool = False

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "prior_modes": list(self.prior_modes),
            "precision_grid": list(self.precision_grid),
            "folds": [f.to_dict() for f in self.folds],
        }


def _sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, j) for j in range(k + 1)) / (2.0 ** n))


def _candidate_inner_mae(
    intervals: Sequence[Interval],
    lever_names: Sequence[str],
    phis: np.ndarray,
    class_weights: np.ndarray,
    *,
    mode: str,
    precision: float,
    difference_clip: bool,
    prior_matrices: dict[str, np.ndarray] | None,
    block_multipliers: Sequence[float],
) -> float | None:
    errors = []
    for hold in range(2, len(intervals)):
        model = HierarchicalPhysicsResidualAdjoint(
            prior_mode=mode, precision=precision, difference_clip=difference_clip,
            prior_matrix=(prior_matrices or {}).get(mode),
            block_multipliers=block_multipliers)
        model.fit(intervals[:hold], phis)
        pred = model.predict_interval(intervals[hold], lever_names)
        measured = intervals[hold].dxdt()
        errors.append(abs(float(class_weights @ (pred[:N_CLASSES] - measured[:N_CLASSES])))
                      * intervals[hold].dep)
    return float(np.mean(errors)) if errors else None


def select_prefix_candidate(
    intervals: Sequence[Interval],
    lever_names: Sequence[str],
    phis: np.ndarray,
    class_weights: np.ndarray,
    *,
    prior_modes: Iterable[str] = PRIOR_MODES,
    precision_grid: Iterable[float] = PRECISION_GRID,
    difference_clip: bool = False,
    prior_matrices: dict[str, np.ndarray] | None = None,
    block_multipliers: Sequence[float] = BLOCK_MULTIPLIERS,
) -> tuple[str, float, float | None]:
    """Select only on a supplied past prefix; deterministic Q-first tie law."""
    modes, grid = tuple(prior_modes), tuple(float(v) for v in precision_grid)
    if not modes or not grid:
        raise ValueError("prior modes and precision grid must be non-empty")
    # The canonical Q ridge precision (0.01) is the preregistered birth arm.  One or
    # two inner folds are too weak to retune it: the first-form probe demonstrated a
    # one-fold winner that exploded on the next interval.  Selection therefore stays
    # frozen until four complete prequential calibration folds exist.
    default = (modes[0], grid[0], None)
    if len(intervals) - 2 < MIN_INNER_SELECTION_FOLDS:
        return default
    ranked = []
    for mode_index, mode in enumerate(modes):
        for precision in grid:
            error = _candidate_inner_mae(
                intervals, lever_names, phis, class_weights, mode=mode,
                precision=precision, difference_clip=difference_clip,
                prior_matrices=prior_matrices,
                block_multipliers=block_multipliers)
            if error is not None and np.isfinite(error):
                ranked.append((error, mode_index, precision, mode))
    if not ranked:
        return default
    error, _, precision, mode = min(ranked)
    return mode, float(precision), float(error)


def project_score_aggregate(
    prediction: np.ndarray,
    predictive_variance: np.ndarray,
    class_weights: np.ndarray,
    target_aggregate: float,
) -> tuple[np.ndarray, float]:
    """Minimum posterior-Mahalanobis projection onto ``w^T y = target``.

    The existing regime dispatcher supplies only the aggregate total-forcing forecast;
    U retains ownership of the per-class decomposition and lever field.  Diagonal
    posterior predictive variance determines which class coordinates may move.
    """
    pred, var, weights = _fp32(prediction).copy(), _fp32(predictive_variance), _fp32(class_weights)
    if pred.shape != (STATE_DIM,) or var.shape != (STATE_DIM,) or weights.shape != (N_CLASSES,):
        raise ValueError("aggregate projection received invalid shapes")
    direction = np.maximum(var[:N_CLASSES], np.float32(1e-12)) * weights
    denom = float(weights @ direction)
    if not np.isfinite(denom) or denom <= 0.0:
        direction = weights
        denom = float(weights @ weights)
    delta_scale = (float(target_aggregate) - float(weights @ pred[:N_CLASSES])) / denom
    delta = direction * np.float32(delta_scale)
    pred[:N_CLASSES] += delta
    return pred, float(np.linalg.norm(delta))


def _regime_aggregate_forecast(
    past: list[Interval], target: Interval, traj: CampaignTrajectory,
    phis: np.ndarray, class_weights: np.ndarray,
) -> tuple[float, str]:
    """Consume the existing T/persistence dispatcher without duplicating its estimator."""
    from tac.witness_control.regime_dispatch import PERSISTENCE, dispatch_decision

    comp = fit_score_composition(traj.verdicts)
    decision = dispatch_decision(past, comp, traj.lever_names, seed=0,
                                 meta_lambda_guard=True)
    if decision.tool == PERSISTENCE:
        forecast = past[-1].dxdt()
    else:
        model = make_model(decision.tool)
        model.fit(past, phis, seed=0)
        forecast = _predict_interval(model, decision.tool, target, traj.lever_names)
    return float(_fp32(class_weights) @ _fp32(forecast[:N_CLASSES])), decision.tool


def _field_diagnostics(
    model: HierarchicalPhysicsResidualAdjoint,
    traj: CampaignTrajectory,
    class_weights: np.ndarray,
) -> dict:
    grad = np.zeros(STATE_DIM, dtype=np.float32)
    grad[:N_CLASSES] = np.float32(100.0) * _fp32(class_weights)
    rows = []
    resolved = 0
    resolved_identified = 0
    intervals = build_intervals(traj)
    shares = np.stack([iv.u_mean for iv in intervals])
    n_identified = 0
    for lever_index, name in enumerate(traj.lever_names):
        phi = lever_features(name)
        mean = float(grad @ model.response(np.zeros(STATE_DIM), np.zeros(3), phi))
        variance = float(np.sum((grad * grad) * model.response_variance(phi)))
        std = math.sqrt(max(variance, 0.0))
        sign_resolved = abs(mean) > 1.96 * std
        data_identified = bool(
            np.std(shares[:, lever_index]) > 1e-4 and
            np.max(shares[:, lever_index]) > 1e-3)
        resolved += int(sign_resolved)
        n_identified += int(data_identified)
        resolved_identified += int(sign_resolved and data_identified)
        rows.append({"lever": name, "mean_marginal_ds": mean,
                     "posterior_std": std, "sign_resolved_95": sign_resolved,
                     "data_identified_from_share_variation": data_identified,
                     "resolution_source": (
                         "DATA_PLUS_FIXED_PRIOR" if data_identified else
                         "FIXED_PRIOR_FEATURE_STRUCTURE_ONLY")})
    return {
        "per_lever": rows,
        "resolved_sign_fraction_95": resolved / max(len(rows), 1),
        "data_identified_lever_count": n_identified,
        "data_identified_resolved_sign_fraction_95": (
            resolved_identified / max(n_identified, 1)),
        "identifiability": (
            "CONDITIONAL_POSTERIOR_PREDICTIVE_ONLY_NOT_CAUSAL; COVARIANCE CONDITIONS_ON_"
            "FIXED_PRIOR_MODE_PRECISION_AND_IN_SAMPLE_NOISE"),
        "clip_diagnostic": model.clip_diagnostic,
        "posterior": model._check().to_dict(),
    }


def walkforward_backtest(
    traj: CampaignTrajectory,
    *,
    prior_modes: Iterable[str] = PRIOR_MODES,
    precision_grid: Iterable[float] = PRECISION_GRID,
    difference_clip: bool = False,
    aggregate_constraint: bool = False,
    prior_matrices: dict[str, np.ndarray] | None = None,
    block_multipliers: Sequence[float] = BLOCK_MULTIPLIERS,
) -> WarmstartBacktest:
    """Deployment-faithful outer WF with nested, prefix-only mode/precision selection."""
    intervals = build_intervals(traj)
    if len(intervals) < 4:
        raise ValueError(f"need at least four intervals; have {len(intervals)}")
    modes, grid = tuple(prior_modes), tuple(float(v) for v in precision_grid)
    phis = np.stack([lever_features(n) for n in traj.lever_names]).astype(np.float32)
    weights = _fp32(fit_score_composition(traj.verdicts).class_weights)
    folds = []
    for hold in range(2, len(intervals)):
        mode, precision, inner = select_prefix_candidate(
            intervals[:hold], traj.lever_names, phis, weights,
            prior_modes=modes, precision_grid=grid, difference_clip=difference_clip,
            prior_matrices=prior_matrices, block_multipliers=block_multipliers)
        model = HierarchicalPhysicsResidualAdjoint(
            prior_mode=mode, precision=precision, difference_clip=difference_clip,
            prior_matrix=(prior_matrices or {}).get(mode),
            block_multipliers=block_multipliers)
        model.fit(intervals[:hold], phis)
        iv = intervals[hold]
        pred, measured = model.predict_interval(iv, traj.lever_names), iv.dxdt()
        aggregate_forecaster = None
        constraint_delta = 0.0
        if aggregate_constraint:
            row, _ = _design_rows([iv], phis)
            target_aggregate, aggregate_forecaster = _regime_aggregate_forecast(
                intervals[:hold], iv, traj, phis, weights)
            pred, constraint_delta = project_score_aggregate(
                pred, model._check().predictive_variance(row[0]), weights,
                target_aggregate)
        persistence = intervals[hold - 1].dxdt()
        folds.append(FoldResult(
            hold=hold, ep1=float(iv.ep1), prior_mode=mode, precision=precision,
            inner_mae=inner,
            error=abs(float(weights @ (pred[:N_CLASSES] - measured[:N_CLASSES]))) * iv.dep,
            persistence_error=abs(float(weights @ (
                persistence[:N_CLASSES] - measured[:N_CLASSES]))) * iv.dep,
            perclass_error=float(np.mean(np.abs(pred[:N_CLASSES] - measured[:N_CLASSES]))) * iv.dep,
            persistence_perclass_error=float(np.mean(np.abs(
                persistence[:N_CLASSES] - measured[:N_CLASSES]))) * iv.dep,
            effective_degrees_of_freedom=model._check().effective_degrees_of_freedom,
            posterior_condition=model._check().condition_number,
            support=support_certificate(intervals[:hold], phis),
            aggregate_forecaster=aggregate_forecaster,
            aggregate_constraint_delta_norm=constraint_delta,
        ))
    wins = sum(f.error < f.persistence_error for f in folds)
    losses = sum(f.error > f.persistence_error for f in folds)
    ties = len(folds) - wins - losses
    final_mode, final_precision, _ = select_prefix_candidate(
        intervals, traj.lever_names, phis, weights, prior_modes=modes,
        precision_grid=grid, difference_clip=difference_clip, prior_matrices=prior_matrices,
        block_multipliers=block_multipliers)
    final = HierarchicalPhysicsResidualAdjoint(
        prior_mode=final_mode, precision=final_precision, difference_clip=difference_clip,
        prior_matrix=(prior_matrices or {}).get(final_mode),
        block_multipliers=block_multipliers)
    final.fit(intervals, phis)
    return WarmstartBacktest(
        architecture=(
            "U_hierarchical_physics_residual_closed_organ" if aggregate_constraint else
            "U_hierarchical_physics_residual_vrclip" if difference_clip else
            "U_hierarchical_physics_residual"),
        prior_modes=modes, precision_grid=grid, n_folds=len(folds),
        walkforward_mae_model=float(np.mean([f.error for f in folds])),
        walkforward_mae_persistence=float(np.mean([f.persistence_error for f in folds])),
        walkforward_perclass_mae_model=float(np.mean([f.perclass_error for f in folds])),
        walkforward_perclass_mae_persistence=float(np.mean([
            f.persistence_perclass_error for f in folds])),
        fold_wins=wins, fold_losses=losses, fold_ties=ties,
        sign_test_p=_sign_test(wins, losses), folds=tuple(folds),
        field_diagnostics=_field_diagnostics(final, traj, weights),
        continual_learning_status=(
            f"EXTERNAL_POSTERIOR_ONLY_UNTIL_{GRADUATION_MIN_RECORDS}_INDEPENDENT_TRAJECTORIES"),
    )


__all__ = [
    "BLOCK_MULTIPLIERS", "GRADUATION_MIN_RECORDS", "MIN_INNER_SELECTION_FOLDS",
    "PRECISION_GRID", "PRIOR_MODES",
    "RECEIPT_SCHEMA", "FoldResult", "HierarchicalPhysicsResidualAdjoint", "PosteriorSolve",
    "SupportCertificate", "WarmstartBacktest", "clipped_difference_targets",
    "physics_prior_coefficients", "posterior_solve_mlx_fp32", "posterior_solve_numpy_fp32",
    "prefix_lipschitz_bound", "project_score_aggregate", "select_prefix_candidate",
    "support_certificate",
    "walkforward_backtest",
]
