# SPDX-License-Identifier: MIT
"""Statistical lens over ``(vec, Phi, X)`` and witness telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .core import LensOperationError, T, TypedResult, immutable_array
from .topology import TopologyLens


@dataclass(frozen=True, slots=True)
class KDEDensity:
    samples: tuple[float, ...]
    points: tuple[float, ...]
    density: tuple[float, ...]
    bandwidth_factor: float


@dataclass(frozen=True, slots=True)
class DistributionDrift:
    wasserstein: float
    ks_statistic: float
    ks_pvalue: float
    mean_shift: float
    baseline_count: int
    current_count: int


@dataclass(frozen=True, slots=True)
class StructureTensorAnisotropy:
    lam_max: np.ndarray
    lam_min: np.ndarray
    dH: np.ndarray
    energy: np.ndarray
    orient_normal: np.ndarray
    sigma: float
    floor: float


@dataclass(frozen=True, slots=True)
class ChangePoint:
    index: int
    score: float
    detected: bool
    left_mean: float
    right_mean: float
    left_slope: float
    right_slope: float
    ema_span: int


def _series(value: Any, *, owner: str, minimum: int = 1) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise LensOperationError(f"{owner} must be a finite numeric series") from exc
    if array.size < minimum:
        raise LensOperationError(f"{owner} requires at least {minimum} value(s)")
    if not np.isfinite(array).all():
        raise LensOperationError(f"{owner} must contain only finite values")
    return array


def _kde(complex_: T, args: dict[str, Any]) -> KDEDensity:
    try:
        from scipy.stats import gaussian_kde  # type: ignore[import-untyped]
    except ImportError as exc:
        raise LensOperationError(
            "kde_density requires the optional analysis dependencies; install tac[analysis]"
        ) from exc
    values = args.pop("values", tuple(element.phi for element in complex_.elements))
    samples = _series(values, owner="KDE values", minimum=2)
    if np.unique(samples).size < 2:
        raise LensOperationError("KDE values must contain at least two distinct values")
    points_value = args.pop("points", None)
    if points_value is None:
        points = np.linspace(float(samples.min()), float(samples.max()), 100)
    else:
        points = _series(points_value, owner="KDE points")
    bandwidth = args.pop("bandwidth", "scott")
    if args:
        raise LensOperationError(f"unexpected kde_density arguments: {sorted(args)}")
    try:
        estimator = gaussian_kde(samples, bw_method=bandwidth)
        density = estimator(points)
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise LensOperationError(f"KDE fit failed: {exc}") from exc
    return KDEDensity(
        samples=tuple(float(value) for value in samples),
        points=tuple(float(value) for value in points),
        density=tuple(float(value) for value in density),
        bandwidth_factor=float(cast("Any", estimator.factor)),
    )


def _drift(args: dict[str, Any]) -> DistributionDrift:
    try:
        from scipy.stats import (  # type: ignore[import-untyped]
            ks_2samp,
            wasserstein_distance,
        )
    except ImportError as exc:
        raise LensOperationError(
            "distribution_drift requires the optional analysis dependencies; "
            "install tac[analysis]"
        ) from exc
    if "baseline" not in args or "current" not in args:
        raise LensOperationError("distribution_drift requires baseline= and current=")
    baseline = _series(args.pop("baseline"), owner="baseline")
    current = _series(args.pop("current"), owner="current")
    if args:
        raise LensOperationError(f"unexpected distribution_drift arguments: {sorted(args)}")
    ks = ks_2samp(baseline, current, alternative="two-sided", method="auto")
    return DistributionDrift(
        wasserstein=float(wasserstein_distance(baseline, current)),
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        mean_shift=float(current.mean() - baseline.mean()),
        baseline_count=int(baseline.size),
        current_count=int(current.size),
    )


def _anisotropy(complex_: T, args: dict[str, Any]) -> StructureTensorAnisotropy:
    try:
        from tac.boundary_math.partition_anisotropy_map import structure_tensor_dH
    except ImportError as exc:
        raise LensOperationError(
            "anisotropy requires the optional analysis dependencies; install tac[analysis]"
        ) from exc
    field = args.pop("field", complex_.metadata.get("field"))
    if field is None:
        raise LensOperationError("anisotropy requires field= or adapter metadata['field']")
    try:
        array = np.asarray(field, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LensOperationError("anisotropy field must be a finite 2-D array") from exc
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise LensOperationError("anisotropy field must be a non-empty finite 2-D array")
    sigma = float(args.pop("sigma", 2.0))
    floor = float(args.pop("floor", 1e-6))
    if not np.isfinite(sigma) or sigma < 0.0:
        raise LensOperationError("anisotropy sigma must be finite and non-negative")
    if not np.isfinite(floor) or floor <= 0.0:
        raise LensOperationError("anisotropy floor must be finite and positive")
    if args:
        raise LensOperationError(f"unexpected anisotropy arguments: {sorted(args)}")
    raw = structure_tensor_dH(array, sigma=sigma, floor=floor)
    outputs: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        outputs[key] = immutable_array(np.asarray(value))
    return StructureTensorAnisotropy(
        lam_max=outputs["lam_max"],
        lam_min=outputs["lam_min"],
        dH=outputs["dH"],
        energy=outputs["energy"],
        orient_normal=outputs["orient_normal"],
        sigma=sigma,
        floor=floor,
    )


def _change_point(args: dict[str, Any]) -> ChangePoint:
    try:
        from tac.witness_control.costate_estimator import slope_with_stderr
        from tac.witness_control.sigma_min_plateau import ema_smooth
    except ImportError as exc:
        raise LensOperationError(
            "change_point requires the optional analysis dependencies; install tac[analysis]"
        ) from exc
    if "values" not in args:
        raise LensOperationError("change_point requires values=")
    values = _series(args.pop("values"), owner="change-point values", minimum=4)
    min_segment = int(args.pop("min_segment", 2))
    ema_span = int(args.pop("ema_span", 1))
    min_score = float(args.pop("min_score", 0.0))
    if min_segment < 2 or values.size < 2 * min_segment:
        raise LensOperationError(
            "change_point requires min_segment>=2 and at least two complete segments"
        )
    if ema_span <= 0:
        raise LensOperationError("ema_span must be positive")
    if not np.isfinite(min_score) or min_score < 0.0:
        raise LensOperationError("min_score must be finite and non-negative")
    if args:
        raise LensOperationError(f"unexpected change_point arguments: {sorted(args)}")
    smoothed = np.asarray(ema_smooth(values.tolist(), ema_span), dtype=np.float64)
    best_index = min_segment
    best_score = -1.0
    for index in range(min_segment, len(smoothed) - min_segment + 1):
        left = smoothed[:index]
        right = smoothed[index:]
        scale = np.sqrt((left.size * right.size) / smoothed.size)
        score = float(abs(right.mean() - left.mean()) * scale)
        if score > best_score:
            best_index = index
            best_score = score
    left = smoothed[:best_index]
    right = smoothed[best_index:]
    left_fit = slope_with_stderr(list(range(len(left))), left.tolist())
    right_fit = slope_with_stderr(list(range(len(right))), right.tolist())
    return ChangePoint(
        index=best_index,
        score=best_score,
        detected=best_score > min_score,
        left_mean=float(left.mean()),
        right_mean=float(right.mean()),
        left_slope=float(left_fit.slope),
        right_slope=float(right_fit.slope),
        ema_span=ema_span,
    )


class StatisticsLens:
    """Density, persistence, drift, anisotropy, and change analysis."""

    name = "statistics"
    operations = frozenset(
        {"kde_density", "persistence", "distribution_drift", "anisotropy", "change_point"}
    )

    def apply(self, complex_: T, op: str, **args: Any) -> TypedResult[Any]:
        if op not in self.operations:
            raise LensOperationError(
                f"statistics operation {op!r} is unsupported; choose {sorted(self.operations)}"
            )
        value: Any
        provenance: tuple[str, ...]
        element_ids: tuple[str, ...]
        if op == "kde_density":
            value = _kde(complex_, args)
            provenance = ("scipy.stats.gaussian_kde over explicit scalar samples",)
            element_ids = tuple(element.id for element in complex_.elements)
        elif op == "persistence":
            topology = TopologyLens().apply(complex_, "persistence", **args)
            value = topology.value
            provenance = (
                *topology.provenance,
                "statistics lens delegates to topology persistence",
            )
            element_ids = topology.element_ids
        elif op == "distribution_drift":
            value = _drift(args)
            provenance = (
                "scipy.stats.wasserstein_distance",
                "scipy.stats.ks_2samp",
            )
            element_ids = ()
        elif op == "anisotropy":
            value = _anisotropy(complex_, args)
            provenance = ("tac.boundary_math.partition_anisotropy_map.structure_tensor_dH",)
            element_ids = tuple(element.id for element in complex_.elements)
        else:
            value = _change_point(args)
            provenance = (
                "tac.witness_control.sigma_min_plateau.ema_smooth",
                "tac.witness_control.costate_estimator.slope_with_stderr",
                "deterministic maximum two-segment mean shift",
            )
            element_ids = ()
        return TypedResult(
            lens=self.name,
            op=op,
            value=value,
            element_ids=element_ids,
            provenance=provenance,
        )
