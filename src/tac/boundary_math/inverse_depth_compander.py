# SPDX-License-Identifier: MIT
"""Softened inverse-depth Riemannian row compander for the projective ground chart.

This module implements the S1 follow-up from
``.omx/research/manifold_geometry_slots_dig_20260713.md``.  The measured n600
receipt selected the row-density profile

``rho(v) = sqrt(g_vv(v)) proportional to (v - v_h + delta)^-2``

with ``v_h=174`` and ``delta=32.5257801441824`` rows.  The compander is a
coordinate reparameterization: it changes *where* a fixed-width coordinate
field spends capacity, never the field width, parameter count, or step count.

The composition is deliberately explicit and non-commutative::

    frame coordinates -> GroundFrameChart -> inverse-depth row compander

The NumPy implementation is the deterministic fp32 reference.  The MLX twin
uses the same scalar operation order; parity authority is the CPU stream.  The
analytic transform consumes no RNG, but ``seed`` is persisted as part of the
chart identity so a future empirical-CDF extension cannot acquire an
untracked random stream.

The fitted profile is video-derived and therefore COUNTED at receiver close.
This module supplies the generic transform only; it does not claim that
embedding the fitted constants in decoder code is free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_COMPANDER_SEED",
    "MARGIN_COMPANDER_RESUME_PREFIX",
    "MEASURED_HORIZON_ROW",
    "MEASURED_SOFTENING_OFFSET_ROWS",
    "S1_RECEIPT_PATH",
    "InverseDepthCompanderError",
    "InverseDepthCompanderProfile",
    "MarginCompandedGroundChart",
    "chart_rows_to_image_rows_mlx",
    "chart_rows_to_image_rows_numpy",
    "compand_coords_mlx",
    "compand_coords_numpy",
    "image_rows_to_chart_rows_mlx",
    "image_rows_to_chart_rows_numpy",
    "uncompand_coords_mlx",
    "uncompand_coords_numpy",
]

S1_RECEIPT_PATH = ".omx/research/manifold_geometry_slots_probe_s1_s2_20260713.json"
MEASURED_HORIZON_ROW = 174.0
MEASURED_SOFTENING_OFFSET_ROWS = 32.5257801441824
DEFAULT_COMPANDER_SEED = 0
MARGIN_COMPANDER_RESUME_PREFIX = "__mcc_"
_PROFILE_VERSION = "softened_inverse_depth_s1_v1"


class InverseDepthCompanderError(ValueError):
    """Raised when a compander profile, input, or resume identity is invalid."""


@dataclass(frozen=True)
class InverseDepthCompanderProfile:
    """Validated identity of the analytic softened-inverse-depth chart.

    ``height`` is the endpoint-inclusive raster height.  ``seed=0`` is a
    DERIVED deterministic family seed; the current analytic map does not draw
    random numbers.  It is nevertheless part of the persisted identity.
    """

    height: int = 384
    horizon_row: float = MEASURED_HORIZON_ROW
    softening_offset_rows: float = MEASURED_SOFTENING_OFFSET_ROWS
    seed: int = DEFAULT_COMPANDER_SEED

    def __post_init__(self) -> None:
        if isinstance(self.height, bool) or int(self.height) != self.height or self.height < 2:
            raise InverseDepthCompanderError(f"height must be an integer >= 2, got {self.height!r}")
        bottom = float(self.height - 1)
        horizon = float(self.horizon_row)
        delta = float(self.softening_offset_rows)
        if not np.isfinite(horizon) or not (0.0 <= horizon < bottom):
            raise InverseDepthCompanderError(
                f"horizon_row must be finite in [0, {bottom}), got {self.horizon_row!r}"
            )
        if not np.isfinite(delta) or delta <= 0.0:
            raise InverseDepthCompanderError(
                f"softening_offset_rows must be positive finite, got {self.softening_offset_rows!r}"
            )
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise InverseDepthCompanderError(
                f"seed must be a nonnegative integer, got {self.seed!r}"
            )

    @property
    def bottom_row(self) -> float:
        return float(self.height - 1)

    def provenance(self) -> dict[str, Any]:
        return {
            "profile_version": _PROFILE_VERSION,
            "height": int(self.height),
            "horizon_row": float(self.horizon_row),
            "softening_offset_rows": float(self.softening_offset_rows),
            "seed": int(self.seed),
            "seed_role": "analytic-family identity; current transform consumes no RNG",
            "value_provenance": S1_RECEIPT_PATH,
            "counting": "fitted profile is video-derived and must be counted in receiver-close bytes",
        }


def _fp32_constants(profile: InverseDepthCompanderProfile) -> tuple[np.float32, ...]:
    horizon = np.float32(profile.horizon_row)
    bottom = np.float32(profile.bottom_row)
    delta = np.float32(profile.softening_offset_rows)
    distance = np.float32(bottom - horizon)
    one = np.float32(1.0)
    inv_delta = np.float32(one / delta)
    inv_end = np.float32(one / np.float32(distance + delta))
    normalizer = np.float32(inv_delta - inv_end)
    end_denominator = np.float32(np.float32(distance + delta) * np.float32(distance + delta))
    end_slope = np.float32(np.float32(distance * np.float32(one / end_denominator)) / normalizer)
    return horizon, bottom, delta, distance, inv_delta, normalizer, end_slope


def image_rows_to_chart_rows_numpy(
    rows: np.ndarray | float,
    profile: InverseDepthCompanderProfile,
) -> np.ndarray:
    """Map image rows to companded rows using the deterministic fp32 reference.

    Rows above the measured horizon are unchanged.  The fitted cumulative map
    acts on the in-raster ground support.  Below the raster, the endpoint
    tangent is continued linearly so projective coordinates remain globally
    finite, strictly monotone, and analytically invertible.
    """

    values = np.asarray(rows, dtype=np.float32)
    horizon, bottom, delta, distance, inv_delta, normalizer, end_slope = _fp32_constants(profile)
    one = np.float32(1.0)
    active_distance = np.where(values > horizon, np.float32(values - horizon), np.float32(0.0))
    reciprocal = np.float32(one / np.float32(active_distance + delta))
    fraction = np.float32(np.float32(inv_delta - reciprocal) / normalizer)
    middle = np.float32(horizon + np.float32(distance * fraction))
    tail = np.float32(bottom + np.float32(end_slope * np.float32(values - bottom)))
    return np.where(values <= horizon, values, np.where(values >= bottom, tail, middle)).astype(
        np.float32,
        copy=False,
    )


def chart_rows_to_image_rows_numpy(
    rows: np.ndarray | float,
    profile: InverseDepthCompanderProfile,
) -> np.ndarray:
    """Analytic inverse of :func:`image_rows_to_chart_rows_numpy` in fp32."""

    values = np.asarray(rows, dtype=np.float32)
    horizon, bottom, delta, distance, inv_delta, normalizer, end_slope = _fp32_constants(profile)
    one = np.float32(1.0)
    active = np.where(values > horizon, np.float32(values - horizon), np.float32(0.0))
    fraction = np.float32(active / distance)
    reciprocal = np.float32(inv_delta - np.float32(fraction * normalizer))
    safe_reciprocal = np.where(values < bottom, reciprocal, one).astype(np.float32, copy=False)
    middle_distance = np.float32(np.float32(one / safe_reciprocal) - delta)
    middle = np.float32(horizon + middle_distance)
    tail = np.float32(bottom + np.float32(np.float32(values - bottom) / end_slope))
    return np.where(values <= horizon, values, np.where(values >= bottom, tail, middle)).astype(
        np.float32,
        copy=False,
    )


def _validate_coords_numpy(coords: np.ndarray) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != 2:
        raise InverseDepthCompanderError(f"coords must have shape (N, 2), got {out.shape}")
    return out


def compand_coords_numpy(
    coords: np.ndarray,
    profile: InverseDepthCompanderProfile,
) -> np.ndarray:
    """Compand only normalized y; x is preserved exactly."""

    values = _validate_coords_numpy(coords)
    scale = np.float32(profile.height - 1)
    one = np.float32(1.0)
    two = np.float32(2.0)
    row = np.float32(np.float32(values[:, 1] + one) * np.float32(scale / two))
    chart_row = image_rows_to_chart_rows_numpy(row, profile)
    y_chart = np.float32(np.float32(chart_row * np.float32(two / scale)) - one)
    return np.stack((values[:, 0], y_chart), axis=-1).astype(np.float32, copy=False)


def uncompand_coords_numpy(
    coords: np.ndarray,
    profile: InverseDepthCompanderProfile,
) -> np.ndarray:
    """Inverse normalized-coordinate map; x is preserved exactly."""

    values = _validate_coords_numpy(coords)
    scale = np.float32(profile.height - 1)
    one = np.float32(1.0)
    two = np.float32(2.0)
    chart_row = np.float32(np.float32(values[:, 1] + one) * np.float32(scale / two))
    row = chart_rows_to_image_rows_numpy(chart_row, profile)
    y_image = np.float32(np.float32(row * np.float32(two / scale)) - one)
    return np.stack((values[:, 0], y_image), axis=-1).astype(np.float32, copy=False)


def _mlx_constants(profile: InverseDepthCompanderProfile) -> tuple[Any, ...]:
    import mlx.core as mx

    return tuple(mx.array(value, dtype=mx.float32) for value in _fp32_constants(profile))


def image_rows_to_chart_rows_mlx(rows_mx: Any, profile: InverseDepthCompanderProfile) -> Any:
    """MLX twin of :func:`image_rows_to_chart_rows_numpy` (CPU-stream authority)."""

    import mlx.core as mx

    values = rows_mx.astype(mx.float32)
    horizon, bottom, delta, distance, inv_delta, normalizer, end_slope = _mlx_constants(profile)
    one = mx.array(np.float32(1.0), dtype=mx.float32)
    active_distance = mx.where(values > horizon, values - horizon, mx.zeros_like(values))
    reciprocal = one / (active_distance + delta)
    fraction = (inv_delta - reciprocal) / normalizer
    middle = horizon + distance * fraction
    tail = bottom + end_slope * (values - bottom)
    return mx.where(values <= horizon, values, mx.where(values >= bottom, tail, middle))


def chart_rows_to_image_rows_mlx(rows_mx: Any, profile: InverseDepthCompanderProfile) -> Any:
    """MLX twin of :func:`chart_rows_to_image_rows_numpy` (CPU-stream authority)."""

    import mlx.core as mx

    values = rows_mx.astype(mx.float32)
    horizon, bottom, delta, distance, inv_delta, normalizer, end_slope = _mlx_constants(profile)
    one = mx.array(np.float32(1.0), dtype=mx.float32)
    active = mx.where(values > horizon, values - horizon, mx.zeros_like(values))
    fraction = active / distance
    reciprocal = inv_delta - fraction * normalizer
    safe_reciprocal = mx.where(values < bottom, reciprocal, one)
    middle = horizon + (one / safe_reciprocal - delta)
    tail = bottom + (values - bottom) / end_slope
    return mx.where(values <= horizon, values, mx.where(values >= bottom, tail, middle))


def _validate_coords_mlx(coords_mx: Any) -> None:
    shape = tuple(int(value) for value in coords_mx.shape)
    if len(shape) != 2 or shape[1] != 2:
        raise InverseDepthCompanderError(f"coords must have shape (N, 2), got {shape}")


def compand_coords_mlx(coords_mx: Any, profile: InverseDepthCompanderProfile) -> Any:
    """MLX normalized-coordinate twin of :func:`compand_coords_numpy`."""

    import mlx.core as mx

    _validate_coords_mlx(coords_mx)
    values = coords_mx.astype(mx.float32)
    one = mx.array(np.float32(1.0), dtype=mx.float32)
    two = mx.array(np.float32(2.0), dtype=mx.float32)
    scale = mx.array(np.float32(profile.height - 1), dtype=mx.float32)
    row = (values[:, 1] + one) * (scale / two)
    chart_row = image_rows_to_chart_rows_mlx(row, profile)
    y_chart = chart_row * (two / scale) - one
    return mx.stack((values[:, 0], y_chart), axis=-1)


def uncompand_coords_mlx(coords_mx: Any, profile: InverseDepthCompanderProfile) -> Any:
    """MLX normalized-coordinate twin of :func:`uncompand_coords_numpy`."""

    import mlx.core as mx

    _validate_coords_mlx(coords_mx)
    values = coords_mx.astype(mx.float32)
    one = mx.array(np.float32(1.0), dtype=mx.float32)
    two = mx.array(np.float32(2.0), dtype=mx.float32)
    scale = mx.array(np.float32(profile.height - 1), dtype=mx.float32)
    chart_row = (values[:, 1] + one) * (scale / two)
    row = chart_rows_to_image_rows_mlx(chart_row, profile)
    y_image = row * (two / scale) - one
    return mx.stack((values[:, 0], y_image), axis=-1)


@dataclass(frozen=True)
class MarginCompandedGroundChart:
    """Composition wrapper: softened inverse depth *after* ``GroundFrameChart``."""

    ground_chart: Any
    profile: InverseDepthCompanderProfile

    def __post_init__(self) -> None:
        grid_hw = tuple(int(value) for value in self.ground_chart.grid_hw)
        if len(grid_hw) != 2 or grid_hw[0] != int(self.profile.height):
            raise InverseDepthCompanderError(
                "compander height must equal GroundFrameChart grid height: "
                f"profile={self.profile.height}, grid_hw={grid_hw}"
            )

    @staticmethod
    def compose(
        ground_chart: Any,
        *,
        horizon_row: float = MEASURED_HORIZON_ROW,
        softening_offset_rows: float = MEASURED_SOFTENING_OFFSET_ROWS,
        seed: int = DEFAULT_COMPANDER_SEED,
    ) -> MarginCompandedGroundChart:
        profile = InverseDepthCompanderProfile(
            height=int(ground_chart.grid_hw[0]),
            horizon_row=float(horizon_row),
            softening_offset_rows=float(softening_offset_rows),
            seed=int(seed),
        )
        return MarginCompandedGroundChart(ground_chart=ground_chart, profile=profile)

    @property
    def grid_hw(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.ground_chart.grid_hw)

    @property
    def n_pairs(self) -> int:
        return int(self.ground_chart.n_pairs)

    def coords_for_pair_numpy(self, coords: np.ndarray, pair_idx: int) -> np.ndarray:
        ground = self.ground_chart.coords_for_pair_numpy(coords, pair_idx)
        return compand_coords_numpy(ground, self.profile)

    def coords_for_pair_mlx(self, coords_mx: Any, pair_idx: int) -> Any:
        ground = self.ground_chart.coords_for_pair_mlx(coords_mx, pair_idx)
        return compand_coords_mlx(ground, self.profile)

    def state_arrays(self, prefix: str) -> dict[str, Any]:
        """Canonical resume-registry state; static identity, no optimizer tensors."""

        if prefix != MARGIN_COMPANDER_RESUME_PREFIX:
            raise InverseDepthCompanderError(
                f"unexpected resume prefix {prefix!r}; expected {MARGIN_COMPANDER_RESUME_PREFIX!r}"
            )
        return {
            f"{prefix}enabled": np.asarray(1, dtype=np.int8),
            f"{prefix}version": np.asarray(_PROFILE_VERSION),
            f"{prefix}height": np.asarray(self.profile.height, dtype=np.int64),
            f"{prefix}horizon_row": np.asarray(self.profile.horizon_row, dtype=np.float64),
            f"{prefix}softening_offset_rows": np.asarray(
                self.profile.softening_offset_rows,
                dtype=np.float64,
            ),
            f"{prefix}seed": np.asarray(self.profile.seed, dtype=np.int64),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict[str, Any]) -> bool:
        """Validate that a persisted chart identity equals this run's DSL identity."""

        enabled_key = f"{prefix}enabled"
        if enabled_key not in cfg:
            return False  # additive legacy sidecar; F2 cfg custody handles present keys
        expected = {
            "enabled": 1,
            "version": _PROFILE_VERSION,
            "height": int(self.profile.height),
            "horizon_row": float(self.profile.horizon_row),
            "softening_offset_rows": float(self.profile.softening_offset_rows),
            "seed": int(self.profile.seed),
        }
        mismatches: list[str] = []
        for name, wanted in expected.items():
            key = f"{prefix}{name}"
            if key not in cfg:
                mismatches.append(f"{name}=missing")
                continue
            got = cfg[key]
            if isinstance(wanted, float):
                try:
                    same = float(got) == wanted
                except (TypeError, ValueError):
                    same = False
            else:
                same = str(got) == str(wanted)
            if not same:
                mismatches.append(f"{name}: ckpt={got!r} != current={wanted!r}")
        if mismatches:
            raise InverseDepthCompanderError(
                "margin compander resume identity diverged: " + "; ".join(mismatches)
            )
        return True
