# SPDX-License-Identifier: MIT
"""Disagreement-focused replay for the n=1 costate organ.

This is a warm-start from requential coding, not an implementation of relative
entropy coding (REC).  The organ does not expose normalized generative
teacher/student distributions or a decodable proposal stream.  What it *does*
expose is a NumPy-fp32 Gaussian posterior over a continuous marginal-dS field.
Under that declared Gaussian surrogate, the teacher/student disagreement has an
exact KL in bits.  This module spends a fixed replay mass preferentially on that
disagreement while retaining every real interval.

No synthetic target is treated as evidence.  All fitted targets are measured
trajectory derivatives and all curriculum decisions are made from the available
outer-training prefix only.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tac.witness_control.costate_warmstart_cluster import (
    PRECISION_GRID,
    PRIOR_MODES,
    HierarchicalPhysicsResidualAdjoint,
    _design_rows,
    _field_diagnostics,
    _fp32,
    physics_prior_coefficients,
    posterior_solve_mlx_fp32,
    posterior_solve_numpy_fp32,
    select_prefix_candidate,
    support_certificate,
)
from tac.witness_control.lambda_net import (
    N_CLASSES,
    CampaignTrajectory,
    Interval,
    build_intervals,
    fit_score_composition,
    lever_features,
)

REQUENTIAL_RECEIPT_SCHEMA = "costate_requential_curriculum_backtest.v1"
# Half of every interval's unit replay mass is protected; only the other half is
# reallocated.  This is the strongest finite-n coverage guard that still permits a
# strict disagreement preference.  The two-times cap prevents one of <=9 dependent
# intervals from becoming a de-facto repeated dataset.
PROTECTED_UNIFORM_MASS = 0.5
MAX_REPLAY_WEIGHT = 2.0


@dataclass(frozen=True)
class RequentialCurriculum:
    strategy: str
    replay_weights: np.ndarray
    gaussian_kl_bits: np.ndarray
    aggregate_disagreement: np.ndarray
    posterior_variance: np.ndarray
    cumulative_kl_bits: float
    effective_sample_size: float
    variance_floor_hits: int
    latest_bit_fraction: float
    late_debt_slope: float
    latest_is_peak: bool
    status: str = "GAUSSIAN_KL_CURRICULUM_DIAGNOSTIC_NOT_REC_CODE"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "replay_weights": [float(v) for v in self.replay_weights],
            "gaussian_kl_bits": [float(v) for v in self.gaussian_kl_bits],
            "aggregate_disagreement": [float(v) for v in self.aggregate_disagreement],
            "posterior_variance": [float(v) for v in self.posterior_variance],
            "cumulative_kl_bits": self.cumulative_kl_bits,
            "effective_sample_size": self.effective_sample_size,
            "variance_floor_hits": self.variance_floor_hits,
            "latest_bit_fraction": self.latest_bit_fraction,
            "late_debt_slope": self.late_debt_slope,
            "latest_is_peak": self.latest_is_peak,
            "overfit_indicator_status": (
                "RISING_LATE_DISAGREEMENT_OBSERVED_NOT_VALIDATED_OVERFIT_PREDICTOR"
                if self.late_debt_slope > 0.0 and self.latest_is_peak else
                "NO_RISING_LATE_DISAGREEMENT_ON_THIS_PREFIX"),
            "code_interpretation": (
                "POST_BIRTH_SHARED_GAUSSIAN_KL_PROXY_NOT_PREFIX_FREE_REC_NOT_CAPACITY_FLOOR"),
            "status": self.status,
        }


def gaussian_disagreement_bits(delta: float, variance: float) -> float:
    """KL[N(mu_teacher,v)||N(mu_student,v)] in bits for shared variance ``v``."""
    if not np.isfinite(delta) or not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("finite disagreement and positive variance are required")
    return float(delta * delta / (2.0 * variance * math.log(2.0)))


def _capped_mass_allocation(bits: np.ndarray) -> np.ndarray:
    """Allocate fixed replay mass to disagreement without dropping any interval."""
    b = np.maximum(_fp32(bits), np.float32(0.0))
    n = len(b)
    if n == 0:
        raise ValueError("at least one disagreement bit value is required")
    total = float(b.sum(dtype=np.float64))
    if total <= 0.0:
        return np.ones(n, dtype=np.float32)
    weights = (
        np.float32(PROTECTED_UNIFORM_MASS)
        + np.float32(1.0 - PROTECTED_UNIFORM_MASS) * np.float32(n) * b / np.float32(total)
    )
    # Capped-simplex projection.  Repeated redistribution is deterministic and
    # conserves sum(weights)==n to fp32 tolerance.
    for _ in range(n + 1):
        high = weights > np.float32(MAX_REPLAY_WEIGHT)
        if not np.any(high):
            break
        excess = float(np.sum(weights[high] - np.float32(MAX_REPLAY_WEIGHT), dtype=np.float64))
        weights[high] = np.float32(MAX_REPLAY_WEIGHT)
        free = ~high
        if not np.any(free):
            break
        weights[free] += np.float32(excess / int(np.sum(free)))
    weights *= np.float32(n / float(weights.sum(dtype=np.float64)))
    return weights.astype(np.float32)


def build_requential_curriculum(
    intervals: Sequence[Interval],
    lever_names: Sequence[str],
    phis: np.ndarray,
    class_weights: np.ndarray,
    *,
    prior_mode: str,
    precision: float,
    strategy: str,
) -> RequentialCurriculum:
    """Build a prefix-causal uniform or disagreement replay curriculum.

    For interval ``i``, the student is fitted only on intervals ``[:i]``.  The first
    two birth rows receive zero coding debt because there is not yet a two-row organ;
    their protected mass remains in either curriculum.  The Gaussian variance is the
    student's posterior predictive variance in the analytic score direction.
    """
    if strategy not in {"uniform", "disagreement"}:
        raise ValueError("strategy must be 'uniform' or 'disagreement'")
    if not intervals:
        raise ValueError("at least one interval is required")
    weights = _fp32(class_weights)
    bits = np.zeros(len(intervals), dtype=np.float32)
    deltas = np.zeros(len(intervals), dtype=np.float32)
    variances = np.zeros(len(intervals), dtype=np.float32)
    floor_hits = 0
    for i in range(2, len(intervals)):
        student = HierarchicalPhysicsResidualAdjoint(
            prior_mode=prior_mode, precision=precision)
        student.fit(intervals[:i], phis)
        row, _ = _design_rows([intervals[i]], phis)
        pred = student.predict_interval(intervals[i], lever_names)
        teacher = intervals[i].dxdt()
        delta = float(weights @ (teacher[:N_CLASSES] - pred[:N_CLASSES]))
        pred_var = student._check().predictive_variance(row[0])[:N_CLASSES]
        variance = float(np.sum(weights * weights * pred_var, dtype=np.float64))
        # The posterior may be numerically certain on a two-row prefix.  The fp32
        # floor is a numerical guard, not an evidence-bearing noise estimate.
        if variance < float(np.finfo(np.float32).eps):
            floor_hits += 1
        variance = max(variance, float(np.finfo(np.float32).eps))
        deltas[i] = np.float32(delta)
        variances[i] = np.float32(variance)
        bits[i] = np.float32(gaussian_disagreement_bits(delta, variance))
    replay = (
        np.ones(len(intervals), dtype=np.float32)
        if strategy == "uniform" else _capped_mass_allocation(bits)
    )
    ess = float(np.square(replay.sum(dtype=np.float64)) /
                np.square(replay.astype(np.float64)).sum())
    post_birth = bits[2:].astype(np.float64)
    cumulative = float(post_birth.sum())
    if len(post_birth) >= 2:
        x = np.arange(len(post_birth), dtype=np.float64)
        slope = float(np.polyfit(x, post_birth, 1)[0])
    else:
        slope = 0.0
    return RequentialCurriculum(
        strategy=strategy,
        replay_weights=replay,
        gaussian_kl_bits=bits,
        aggregate_disagreement=deltas,
        posterior_variance=variances,
        cumulative_kl_bits=cumulative,
        effective_sample_size=ess,
        variance_floor_hits=floor_hits,
        latest_bit_fraction=(float(post_birth[-1] / cumulative)
                             if len(post_birth) and cumulative > 0.0 else 0.0),
        late_debt_slope=slope,
        latest_is_peak=bool(len(post_birth) and int(np.argmax(post_birth)) == len(post_birth) - 1),
    )


def weighted_posterior_mlx_fp32(
    design: np.ndarray,
    targets: np.ndarray,
    prior: np.ndarray,
    replay_weights: np.ndarray,
    *,
    precision: float,
):
    """MLX parity surface for the curriculum-weighted posterior."""
    root_w = np.sqrt(_fp32(replay_weights)).astype(np.float32)
    return posterior_solve_mlx_fp32(
        _fp32(design) * root_w[:, None], _fp32(targets) * root_w[:, None], prior,
        precision=precision)


class RequentialCurriculumAdjoint(HierarchicalPhysicsResidualAdjoint):
    """U posterior refitted with fixed-mass disagreement replay."""

    name = "R_requential_disagreement_curriculum"

    def __init__(self, *, strategy: str, class_weights: np.ndarray,
                 prior_mode: str, precision: float):
        super().__init__(prior_mode=prior_mode, precision=precision)
        self.strategy = strategy
        self.class_weights = _fp32(class_weights)
        self.curriculum: RequentialCurriculum | None = None

    def fit(self, intervals: Sequence[Interval], phis: np.ndarray, seed: int = 0) -> None:
        del seed
        self.curriculum = build_requential_curriculum(
            intervals, self._lever_names, phis, self.class_weights,
            prior_mode=self.prior_mode, precision=self.precision,
            strategy=self.strategy)
        z, y = _design_rows(intervals, phis)
        prior, self.kappa = physics_prior_coefficients(z, y, mode=self.prior_mode)
        root_w = np.sqrt(self.curriculum.replay_weights).astype(np.float32)
        self.posterior = posterior_solve_numpy_fp32(
            z * root_w[:, None], y * root_w[:, None], prior,
            precision=self.precision)
        self.clip_diagnostic = {
            "status": self.curriculum.status,
            "strategy": self.strategy,
            "cumulative_gaussian_kl_bits": self.curriculum.cumulative_kl_bits,
        }

    def bind_lever_names(self, lever_names: Sequence[str]) -> None:
        self._lever_names = tuple(lever_names)


@dataclass(frozen=True)
class RequentialFold:
    hold: int
    ep1: float
    prior_mode: str
    precision: float
    error: float
    persistence_error: float
    perclass_error: float
    persistence_perclass_error: float
    cumulative_kl_bits: float
    effective_sample_size: float
    support: dict

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RequentialBacktest:
    architecture: str
    strategy: str
    n_folds: int
    walkforward_mae_model: float
    walkforward_mae_persistence: float
    walkforward_perclass_mae_model: float
    walkforward_perclass_mae_persistence: float
    fold_wins: int
    fold_losses: int
    fold_ties: int
    sign_test_p: float
    folds: tuple[RequentialFold, ...]
    final_curriculum: dict
    field_diagnostics: dict
    verdict_scope: str = "INSTANCE_X_FORMULATION"
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    score_claim: bool = False

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "folds": [f.to_dict() for f in self.folds],
        }


def _sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, j) for j in range(k + 1)) / (2.0 ** n))


def walkforward_requential_backtest(
    traj: CampaignTrajectory,
    *,
    strategy: str,
) -> RequentialBacktest:
    """Compare replay curricula under identical prefix-only candidate selection."""
    intervals = build_intervals(traj)
    if len(intervals) < 4:
        raise ValueError("at least four intervals are required")
    phis = np.stack([lever_features(n) for n in traj.lever_names]).astype(np.float32)
    class_weights = _fp32(fit_score_composition(traj.verdicts).class_weights)
    folds = []
    for hold in range(2, len(intervals)):
        mode, precision, _ = select_prefix_candidate(
            intervals[:hold], traj.lever_names, phis, class_weights,
            prior_modes=PRIOR_MODES, precision_grid=PRECISION_GRID)
        model = RequentialCurriculumAdjoint(
            strategy=strategy, class_weights=class_weights,
            prior_mode=mode, precision=precision)
        model.bind_lever_names(traj.lever_names)
        model.fit(intervals[:hold], phis)
        iv = intervals[hold]
        pred, measured = model.predict_interval(iv, traj.lever_names), iv.dxdt()
        persistence = intervals[hold - 1].dxdt()
        assert model.curriculum is not None
        folds.append(RequentialFold(
            hold=hold, ep1=float(iv.ep1), prior_mode=mode, precision=precision,
            error=abs(float(class_weights @ (pred[:N_CLASSES] - measured[:N_CLASSES]))) * iv.dep,
            persistence_error=abs(float(class_weights @ (
                persistence[:N_CLASSES] - measured[:N_CLASSES]))) * iv.dep,
            perclass_error=float(np.mean(np.abs(pred[:N_CLASSES] - measured[:N_CLASSES]))) * iv.dep,
            persistence_perclass_error=float(np.mean(np.abs(
                persistence[:N_CLASSES] - measured[:N_CLASSES]))) * iv.dep,
            cumulative_kl_bits=model.curriculum.cumulative_kl_bits,
            effective_sample_size=model.curriculum.effective_sample_size,
            support=support_certificate(intervals[:hold], phis).to_dict(),
        ))
    wins = sum(f.error < f.persistence_error for f in folds)
    losses = sum(f.error > f.persistence_error for f in folds)
    ties = len(folds) - wins - losses
    mode, precision, _ = select_prefix_candidate(
        intervals, traj.lever_names, phis, class_weights,
        prior_modes=PRIOR_MODES, precision_grid=PRECISION_GRID)
    final = RequentialCurriculumAdjoint(
        strategy=strategy, class_weights=class_weights,
        prior_mode=mode, precision=precision)
    final.bind_lever_names(traj.lever_names)
    final.fit(intervals, phis)
    assert final.curriculum is not None
    return RequentialBacktest(
        architecture="R_requential_disagreement_curriculum",
        strategy=strategy,
        n_folds=len(folds),
        walkforward_mae_model=float(np.mean([f.error for f in folds])),
        walkforward_mae_persistence=float(np.mean([f.persistence_error for f in folds])),
        walkforward_perclass_mae_model=float(np.mean([f.perclass_error for f in folds])),
        walkforward_perclass_mae_persistence=float(np.mean([
            f.persistence_perclass_error for f in folds])),
        fold_wins=wins, fold_losses=losses, fold_ties=ties,
        sign_test_p=_sign_test(wins, losses), folds=tuple(folds),
        final_curriculum=final.curriculum.to_dict(),
        field_diagnostics=_field_diagnostics(final, traj, class_weights),
    )


__all__ = [
    "MAX_REPLAY_WEIGHT",
    "PROTECTED_UNIFORM_MASS",
    "REQUENTIAL_RECEIPT_SCHEMA",
    "RequentialBacktest",
    "RequentialCurriculum",
    "RequentialCurriculumAdjoint",
    "build_requential_curriculum",
    "gaussian_disagreement_bits",
    "walkforward_requential_backtest",
    "weighted_posterior_mlx_fp32",
]
