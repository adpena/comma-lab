# SPDX-License-Identifier: MIT
"""Finite 80-column literal polar-frequency-wedge dictionary.

This module is a clean-room reimplementation of the sealed mathematical
contract in ``curvelet_optimal_form_crux_20260715_SPEC.md``.  It is *not* the
historical spatial-Gabor ``windowed_curvelet`` implementation and it does not
claim the unavailable source hash recorded by the historical receipt.

The four leading columns are the common Q1 partition-of-unity functions.  Each
of the remaining 76 columns is the real inverse Fourier series of an explicit
compact polar mask

    W_j(|q|/2) V_j(wrap_pi(arg(q)-theta_l))

on the half-cycle lattice.  Translation is applied by the exact Fourier shift
phase.  Hence every directional atom is a localized trigonometric polynomial
with period two, rather than a renamed global plane wave.  NumPy-fp32 is the
portable authority; the optional MLX path evaluates the same sparse series.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np

FAMILY = "literal_polar_curvelet"
BASIS_VERSION = "literal-polar-curvelet-finite-v1"
FEATURE_WIDTH = 80
SCALING_COLUMNS = 4
DIRECTIONAL_COLUMNS = FEATURE_WIDTH - SCALING_COLUMNS
SCALING_WIDTH = SCALING_COLUMNS
DIRECTIONAL_WIDTH = DIRECTIONAL_COLUMNS
SEED = 0  # recorded for the common basis-program contract; construction is seed-free
FREQUENCY_LATTICE_RADIUS = 160
FREQUENCY_LATTICE_STEP = 0.5
FREQUENCY_PERIOD = 2.0
SCALES = (0, 2, 4)
ORIENTATION_COUNTS = (4, 8, 16)
RADIAL_CENTERS = (3.0, 12.0, 48.0)
RADIAL_HALF_WIDTHS = (2.0, 8.0, 32.0)
ANGULAR_HALF_WIDTHS = (math.pi / 4.0, math.pi / 8.0, math.pi / 16.0)
PER_SCALE_DIRECTIONAL_BUDGETS = (4, 16, 56)
TRANSLATION_LATTICE_RADIUS = 0.75
COORD_CHUNK = 2048
WINDOW_TOKEN = "cos4_polar_radial_angular_periodized_v2"

# Historical custody is retained only to make accidental relabelling detectable.
LOST_SOURCE_SHA256_NON_AUTHORIZING = (
    "8a2e8befde890f769997f5efdd917cb0eee52219c7e86393dc016604b8674697"
)
HISTORICAL_ATOM_SPEC_SHA256_EXPECTED_NONIDENTICAL = (
    "b4ff750267aab9298f1e5a09c0f05e1f398ccea42dd89c5b4f8c61cd9d3fde91"
)


@dataclass(frozen=True)
class LiteralPolarAtomSpec:
    """One deterministic scalar column in counted coefficient order."""

    column: int
    kind: Literal["scaling", "directional"]
    scale: int | None
    orientation: int | None
    theta: float | None
    radial_center: float | None
    radial_half_width: float | None
    angular_half_width: float | None
    translation_normal_index: int
    translation_tangent_index: int
    center_x: float
    center_y: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Public recovered-contract name.  The longer class name remains descriptive
# while this alias keeps downstream callers independent of the family spelling.
LocalizedAtomSpec = LiteralPolarAtomSpec


@dataclass(frozen=True)
class LocalizedFrameMetadata:
    family: str
    basis_version: str
    feature_width: int
    scaling_atoms: int
    directional_atoms: int
    seed: int
    atom_spec_sha256: str
    source_sha256: str
    historical_atom_spec_sha256_expected_nonidentical: str
    lost_source_sha256_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenuineFrameProofReceipt:
    family: str
    basis_version: str
    atom_spec_sha256: str
    feature_width: int
    scaling_columns: int
    directional_columns: int
    passed: bool
    gates: dict[str, bool]
    proof: dict[str, Any]
    verdict_scope: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(self.proof)
        return row


@dataclass(frozen=True)
class InflateBasisContract:
    family: str
    basis_version: str
    compiled: bool
    entrypoint: str
    feature_width: int
    atom_spec_sha256: str
    source_sha256: str
    generic_regenerated_state: tuple[str, ...]
    counted_state: tuple[str, ...]
    rule118_status: str
    lost_source_sha256_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _translation_slots(scale: int, orientation: int, *, seed: int = SEED) -> tuple[tuple[int, int], ...]:
    """Finite, deterministic truncation of the rotated parabolic lattice.

    The allocation 4 + 16 + 56 gives 76 directional columns while retaining
    every registered orientation.  Indices, not rounded Cartesian centers, are
    authoritative; centers are derived using the exact scale spacings below.
    """

    scale_index = SCALES.index(scale)
    n_orient = ORIENTATION_COUNTS[scale_index]
    budget = PER_SCALE_DIRECTIONAL_BUDGETS[scale_index]
    base, remainder = divmod(budget, n_orient)
    extra_start = (seed + 5 * scale) % n_orient
    extra_orientations = {(extra_start + offset) % n_orient for offset in range(remainder)}
    count = base + int(orientation in extra_orientations)
    a = 2.0 ** (-scale)
    b = 2.0 ** (-0.5 * scale)
    n_limit = math.floor(TRANSLATION_LATTICE_RADIUS / a)
    t_limit = math.floor(TRANSLATION_LATTICE_RADIUS / b)

    def site_key(site: tuple[int, int]) -> tuple[int, int, int]:
        n, t = site
        mixed = (
            ((n + n_limit + 1) * 73856093)
            ^ ((t + t_limit + 1) * 19349663)
            ^ ((seed + 1) * 83492791)
            ^ ((orientation + 1) * 2654435761)
            ^ ((scale + 1) * 97531)
        ) & 0xFFFFFFFF
        return mixed, n, t

    sites = [(n, t) for n in range(-n_limit, n_limit + 1) for t in range(-t_limit, t_limit + 1)]
    return tuple(sorted(sites, key=site_key)[:count])


def _directional_spec(
    column: int,
    scale: int,
    orientation: int,
    tn: int,
    tt: int,
) -> LiteralPolarAtomSpec:
    scale_index = SCALES.index(scale)
    n_orient = ORIENTATION_COUNTS[scale_index]
    theta = math.pi * orientation / n_orient
    normal_spacing = 2.0 ** (-scale)
    tangent_spacing = 2.0 ** (-0.5 * scale)
    normal_offset = tn * normal_spacing
    tangent_offset = tt * tangent_spacing
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    # n=(cos,sin), t=(-sin,cos)
    cx = normal_offset * ct - tangent_offset * st
    cy = normal_offset * st + tangent_offset * ct
    return LiteralPolarAtomSpec(
        column=column,
        kind="directional",
        scale=scale,
        orientation=orientation,
        theta=theta,
        radial_center=RADIAL_CENTERS[scale_index],
        radial_half_width=RADIAL_HALF_WIDTHS[scale_index],
        angular_half_width=ANGULAR_HALF_WIDTHS[scale_index],
        translation_normal_index=tn,
        translation_tangent_index=tt,
        center_x=cx,
        center_y=cy,
    )


def literal_polar_curvelet_atom_specs() -> tuple[LiteralPolarAtomSpec, ...]:
    """Return the immutable 80-column specification in decoder order."""

    atoms: list[LiteralPolarAtomSpec] = [
        LiteralPolarAtomSpec(i, "scaling", None, None, None, None, None, None, 0, 0, 0.0, 0.0)
        for i in range(SCALING_COLUMNS)
    ]
    column = SCALING_COLUMNS
    for scale, n_orient in zip(SCALES, ORIENTATION_COUNTS, strict=True):
        for orientation in range(n_orient):
            for tn, tt in _translation_slots(scale, orientation):
                atoms.append(_directional_spec(column, scale, orientation, tn, tt))
                column += 1
    if len(atoms) != FEATURE_WIDTH:
        raise AssertionError(f"literal atom program generated {len(atoms)} columns, expected 80")
    return tuple(atoms)


ATOM_SPECS = literal_polar_curvelet_atom_specs()


def _canonical_atom_spec_bytes() -> bytes:
    payload = {
        "family": FAMILY,
        "basis_version": BASIS_VERSION,
        "feature_width": FEATURE_WIDTH,
        "frequency_lattice_radius": FREQUENCY_LATTICE_RADIUS,
        "frequency_lattice_step": FREQUENCY_LATTICE_STEP,
        "frequency_period": FREQUENCY_PERIOD,
        "atoms": [asdict(atom) for atom in ATOM_SPECS],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


ATOM_SPEC_SHA256 = hashlib.sha256(_canonical_atom_spec_bytes()).hexdigest()


def module_source_sha256() -> str:
    """Hash the actual reimplementation bytes; never returns the lost-source hash."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def scaling_column_mask() -> np.ndarray:
    mask = np.zeros(FEATURE_WIDTH, dtype=bool)
    mask[:SCALING_COLUMNS] = True
    return mask


def directional_column_mask() -> np.ndarray:
    return ~scaling_column_mask()


def direction_angles() -> np.ndarray:
    return np.asarray(
        [np.nan if atom.theta is None else atom.theta for atom in ATOM_SPECS], dtype=np.float32
    )


def _wrap_axis_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap an unoriented line angle to [-pi/2, pi/2)."""

    return (angle + 0.5 * math.pi) % math.pi - 0.5 * math.pi


def _cos4_window(normalized_coordinate: np.ndarray) -> np.ndarray:
    # ``s`` intentionally retains its incoming dtype.  The historical finite
    # program used float64 xi/geometry but the rounded fp32 pi/2 constant.
    s = np.asarray(normalized_coordinate)
    out = np.zeros_like(s)
    inside = np.abs(s) < 1.0
    out[inside] = np.cos(np.float32(math.pi / 2.0) * s[inside]) ** 4
    return out


@dataclass(frozen=True)
class _SparseSpectrum:
    qx: np.ndarray
    qy: np.ndarray
    amplitude: np.ndarray  # exact fp32 finite-series weights


_SPECTRUM_CACHE: dict[tuple[int, int], _SparseSpectrum] = {}


def _sparse_wedge(scale: int, orientation: int) -> _SparseSpectrum:
    key = (scale, orientation)
    cached = _SPECTRUM_CACHE.get(key)
    if cached is not None:
        return cached
    radius = FREQUENCY_LATTICE_RADIUS
    q = np.arange(-radius, radius + 1, dtype=np.int32)
    qx_grid, qy_grid = np.meshgrid(q, q, indexing="xy")
    xi_x = qx_grid / 2.0
    xi_y = qy_grid / 2.0
    mask = curvelet_frequency_window_numpy(
        xi_x, xi_y, scale=scale, orientation=orientation
    )
    nonzero = mask > 0.0
    qx = qx_grid[nonzero].astype(np.int32, copy=False)
    qy = qy_grid[nonzero].astype(np.int32, copy=False)
    raw = mask[nonzero]
    normalization = np.float32((2.0 ** (0.75 * scale)) / float(np.sum(raw, dtype=np.float64)))
    amplitude = np.asarray(raw * normalization, dtype=np.float32)
    cached = _SparseSpectrum(qx=qx, qy=qy, amplitude=amplitude)
    _SPECTRUM_CACHE[key] = cached
    return cached


def curvelet_frequency_window_numpy(
    xi_x: np.ndarray,
    xi_y: np.ndarray,
    *,
    scale: int,
    orientation: int,
) -> np.ndarray:
    """Literal unnormalized ``W_j(|xi|)V_j(angle)`` polar window."""

    scale_index = SCALES.index(scale)
    n_orient = ORIENTATION_COUNTS[scale_index]
    if not 0 <= orientation < n_orient:
        raise ValueError(f"orientation must be in [0,{n_orient}), got {orientation}")
    x = np.asarray(xi_x)
    y = np.asarray(xi_y)
    if x.shape != y.shape:
        raise ValueError("xi_x and xi_y must have the same shape")
    theta = math.pi * orientation / n_orient
    radial = np.hypot(x, y)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    parallel = x * cos_theta + y * sin_theta
    perpendicular = -x * sin_theta + y * cos_theta
    angle = np.arctan2(np.abs(perpendicular), np.abs(parallel))
    return np.asarray(
        _cos4_window((radial - RADIAL_CENTERS[scale_index]) / RADIAL_HALF_WIDTHS[scale_index])
        * _cos4_window(angle / (math.pi / n_orient)),
        dtype=np.float32,
    )


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"coords must have shape (P,2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coords must be finite")
    return np.asarray(arr, dtype=np.float64)


def _q1_numpy(coords: np.ndarray) -> np.ndarray:
    x, y = coords[:, 0], coords[:, 1]
    result = np.stack(
        (
            0.25 * (1.0 - x) * (1.0 - y),
            0.25 * (1.0 + x) * (1.0 - y),
            0.25 * (1.0 - x) * (1.0 + y),
            0.25 * (1.0 + x) * (1.0 + y),
        ),
        axis=-1,
    )
    inside = (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0)
    return np.where(inside[:, None], result, 0.0)


def _directional_atom_direct_numpy(
    coords: np.ndarray,
    atom: LiteralPolarAtomSpec,
    *,
    point_chunk: int = COORD_CHUNK,
) -> np.ndarray:
    assert atom.scale is not None and atom.orientation is not None
    spectrum = _sparse_wedge(atom.scale, atom.orientation)
    out = np.empty(coords.shape[0], dtype=np.float64)
    xi_x = spectrum.qx.astype(np.float64) * FREQUENCY_LATTICE_STEP
    xi_y = spectrum.qy.astype(np.float64) * FREQUENCY_LATTICE_STEP
    for start in range(0, coords.shape[0], point_chunk):
        stop = min(coords.shape[0], start + point_chunk)
        block = coords[start:stop]
        phase = 2.0 * math.pi * (
            (block[:, 0, None] - atom.center_x) * xi_x[None, :]
            + (block[:, 1, None] - atom.center_y) * xi_y[None, :]
        )
        out[start:stop] = np.sum(
            np.cos(phase) * spectrum.amplitude[None, :], axis=1, dtype=np.float64
        )
    return out


def localized_basis_features_numpy(coords: np.ndarray) -> np.ndarray:
    """Evaluate the authoritative sparse trigonometric program at arbitrary points.

    Args:
        coords: finite ``(P,2)`` coordinates.  The dictionary is period two in
            each directional coordinate; Q1 columns intentionally are not
            periodic because they are the common low-order scaling space.
    Returns:
        ``(P,80)`` NumPy-fp32 features in :data:`ATOM_SPECS` order.
    """

    xy = _validate_coords(coords)
    out = np.empty((xy.shape[0], FEATURE_WIDTH), dtype=np.float32)
    out[:, :SCALING_COLUMNS] = _q1_numpy(xy).astype(np.float32)
    for atom in ATOM_SPECS[SCALING_COLUMNS:]:
        out[:, atom.column] = _directional_atom_direct_numpy(xy, atom).astype(np.float32)
    return out


def atom_feature_numpy(coords: np.ndarray, atom: LocalizedAtomSpec) -> np.ndarray:
    """Evaluate one public atom specification as an authoritative fp32 column."""

    xy = _validate_coords(coords)
    if atom.kind == "scaling":
        if not 0 <= atom.column < SCALING_COLUMNS:
            raise ValueError("scaling atom column must be in [0,4)")
        return _q1_numpy(xy)[:, atom.column].astype(np.float32)
    return _directional_atom_direct_numpy(xy, atom).astype(np.float32)


def deterministic_atom_specs(
    family: str = FAMILY, *, seed: int = SEED
) -> tuple[LocalizedAtomSpec, ...]:
    if family != FAMILY:
        raise ValueError(f"this module implements only {FAMILY!r}, got {family!r}")
    if seed != SEED:
        raise ValueError(f"{BASIS_VERSION} is sealed at seed {SEED}, got {seed}")
    return ATOM_SPECS


def atom_spec_sha256(family: str = FAMILY, *, seed: int = SEED) -> str:
    deterministic_atom_specs(family, seed=seed)
    return ATOM_SPEC_SHA256


def basis_features_numpy(
    coords: np.ndarray, family: str = FAMILY, *, seed: int = SEED
) -> np.ndarray:
    deterministic_atom_specs(family, seed=seed)
    arr = np.asarray(coords)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 4:
        first_y = arr[0, 1]
        changed = np.flatnonzero(arr[:, 1] != first_y)
        width = int(changed[0]) if changed.size else 0
        if width >= 2 and arr.shape[0] % width == 0:
            height = arr.shape[0] // width
            canonical = inclusive_grid_coords(height, width)
            if np.array_equal(arr, canonical):
                return localized_basis_features_grid_numpy(height, width)
    return localized_basis_features_numpy(coords)


def basis_features_mlx(coords: Any, family: str = FAMILY, *, seed: int = SEED) -> Any:
    deterministic_atom_specs(family, seed=seed)
    return localized_basis_features_mlx(coords)


def basis_features(
    coords: Any,
    family: str = FAMILY,
    *,
    seed: int = SEED,
    backend: Literal["numpy", "mlx"] = "numpy",
) -> Any:
    if backend == "numpy":
        return basis_features_numpy(coords, family, seed=seed)
    if backend == "mlx":
        return basis_features_mlx(coords, family, seed=seed)
    raise ValueError(f"backend must be 'numpy' or 'mlx', got {backend!r}")


def inclusive_grid_coords(height: int, width: int) -> np.ndarray:
    """Row-major inclusive Cartesian grid on [-1,1]^2."""

    if isinstance(height, bool) or isinstance(width, bool) or height < 2 or width < 2:
        raise ValueError("height and width must be integers >= 2")
    y = np.linspace(-1.0, 1.0, int(height), dtype=np.float32)
    x = np.linspace(-1.0, 1.0, int(width), dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx, yy), axis=-1).reshape(-1, 2)


def _directional_atom_grid_numpy(
    atom: LiteralPolarAtomSpec, height: int, width: int
) -> np.ndarray:
    """Alias-summed inverse FFT on the nonduplicated base grid."""

    assert atom.scale is not None and atom.orientation is not None
    spectrum = _sparse_wedge(atom.scale, atom.orientation)
    base_h, base_w = height - 1, width - 1
    freq = np.zeros((base_h, base_w), dtype=np.complex64)
    qx = spectrum.qx.astype(np.int64)
    qy = spectrum.qy.astype(np.int64)
    xi_x = spectrum.qx.astype(np.float64) * FREQUENCY_LATTICE_STEP
    xi_y = spectrum.qy.astype(np.float64) * FREQUENCY_LATTICE_STEP
    # x_i=-1+2*i/N: incorporate both translation and the base-grid origin
    # in the one historical complex64 phase expression.
    phase = np.exp(
        -2j
        * np.pi
        * (xi_x * (atom.center_x + 1.0) + xi_y * (atom.center_y + 1.0))
    )
    values = spectrum.amplitude * phase
    np.add.at(freq, (qy % base_h, qx % base_w), values)
    base = np.fft.ifft2(freq * np.float32(base_h * base_w))
    imag_max = float(np.max(np.abs(base.imag), initial=0.0))
    if imag_max > 2e-6:
        raise RuntimeError(f"Hermitian inverse unexpectedly complex: {imag_max:.3g}")
    inclusive = np.empty((height, width), dtype=np.float32)
    inclusive[:-1, :-1] = np.asarray(base.real, dtype=np.float32)
    inclusive[-1, :-1] = inclusive[0, :-1]
    inclusive[:-1, -1] = inclusive[:-1, 0]
    inclusive[-1, -1] = inclusive[0, 0]
    return inclusive


def localized_basis_features_grid_numpy(height: int, width: int) -> np.ndarray:
    """Evaluate a complete inclusive grid via alias-summed inverse FFT.

    Returns a flattened row-major ``(height*width,80)`` array so it is directly
    comparable to :func:`localized_basis_features_numpy` on
    :func:`inclusive_grid_coords`.
    """

    coords = inclusive_grid_coords(height, width)
    out = np.empty((coords.shape[0], FEATURE_WIDTH), dtype=np.float32)
    out[:, :SCALING_COLUMNS] = _q1_numpy(coords).astype(np.float32)
    for atom in ATOM_SPECS[SCALING_COLUMNS:]:
        out[:, atom.column] = _directional_atom_grid_numpy(atom, height, width).reshape(-1).astype(
            np.float32
        )
    return out


def localized_basis_features_mlx(coords: Any) -> Any:
    """MLX mirror of the sparse program; raises softly only when called without MLX."""

    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:  # pragma: no cover - host dependent
        raise RuntimeError(f"MLX unavailable for {FAMILY}: {exc}") from exc
    xy = mx.asarray(coords, dtype=mx.float32)
    if len(xy.shape) != 2 or int(xy.shape[1]) != 2:
        raise ValueError(f"coords must have shape (P,2), got {tuple(xy.shape)}")
    x, y = xy[:, 0], xy[:, 1]
    cols = [
        0.25 * (1.0 - x) * (1.0 - y),
        0.25 * (1.0 + x) * (1.0 - y),
        0.25 * (1.0 - x) * (1.0 + y),
        0.25 * (1.0 + x) * (1.0 + y),
    ]
    inside = (mx.abs(x) <= 1.0) & (mx.abs(y) <= 1.0)
    cols = [mx.where(inside, column, 0.0) for column in cols]
    for atom in ATOM_SPECS[SCALING_COLUMNS:]:
        assert atom.scale is not None and atom.orientation is not None
        spectrum = _sparse_wedge(atom.scale, atom.orientation)
        xi_x = mx.array(spectrum.qx.astype(np.float32) * np.float32(FREQUENCY_LATTICE_STEP))
        xi_y = mx.array(spectrum.qy.astype(np.float32) * np.float32(FREQUENCY_LATTICE_STEP))
        amplitude = mx.array(spectrum.amplitude.astype(np.float32))
        phase = (2.0 * math.pi) * (
            (x[:, None] - atom.center_x) * xi_x[None, :]
            + (y[:, None] - atom.center_y) * xi_y[None, :]
        )
        cols.append(mx.sum(mx.cos(phase) * amplitude[None, :], axis=1))
    return mx.stack(cols, axis=-1).astype(mx.float32)


def mlx_parity_receipt(
    *, height: int = 9, width: int = 11, tolerance: float = 3e-4
) -> dict[str, Any]:
    """Measure MLX parity or return an explicit non-authorizing unavailable receipt."""

    coords = inclusive_grid_coords(height, width)
    ref = localized_basis_features_numpy(coords)
    try:
        got_mx = localized_basis_features_mlx(coords)
        import mlx.core as mx  # type: ignore

        mx.eval(got_mx)
        got = np.asarray(got_mx, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - host dependent
        return {
            "backend": "mlx",
            "status": "UNMEASURED_SOFT_UNAVAILABLE",
            "authority": False,
            "reason": str(exc),
            "atom_spec_sha256": ATOM_SPEC_SHA256,
        }
    max_abs = float(np.max(np.abs(got - ref), initial=0.0))
    return {
        "backend": "mlx",
        "status": "MEASURED_PARITY" if max_abs <= tolerance else "MEASURED_PARITY_FAILURE",
        "authority": False,
        "max_abs_error": max_abs,
        "tolerance": tolerance,
        "passed": bool(max_abs <= tolerance),
        "atom_spec_sha256": ATOM_SPEC_SHA256,
    }


def _spatial_scale_metrics(scale: int, diagnostic_grid_size: int) -> tuple[float, float, float]:
    """Measured aspect, core-energy fraction, and far-tail ratio for one zero-shift atom."""

    atom = next(
        a for a in ATOM_SPECS if a.kind == "directional" and a.scale == scale and a.orientation == 0
    )
    grid = _directional_atom_grid_numpy(atom, diagnostic_grid_size, diagnostic_grid_size)
    coords = inclusive_grid_coords(diagnostic_grid_size, diagnostic_grid_size)
    energy = grid.reshape(-1) ** 2
    energy_sum = float(energy.sum())
    x, y = coords[:, 0], coords[:, 1]
    # Circular distance is the correct second-moment coordinate for a period-two atom.
    dx = (x - atom.center_x + 1.0) % 2.0 - 1.0
    dy = (y - atom.center_y + 1.0) % 2.0 - 1.0
    # For an oscillatory atom the raw spatial moment includes the carrier.  The
    # envelope aspect is measured from the dual spectral covariance and
    # normalized by the coarsest wedge, which is exactly the periodized
    # uncertainty-dual quantity (and gives the declared 1:2:4 law).
    def dual_aspect(j: int) -> float:
        spectrum = _sparse_wedge(j, 0)
        q = np.stack((spectrum.qx, spectrum.qy), axis=-1).astype(np.float64) * 0.5
        weight = spectrum.amplitude.astype(np.float64) ** 2
        with np.errstate(all="ignore"):
            spectral_covariance = (q.T * weight) @ q / float(weight.sum())
        values = np.clip(np.linalg.eigvalsh(spectral_covariance), 1e-30, None)
        return float(math.sqrt(values[1] / values[0]))

    aspect = dual_aspect(scale) / dual_aspect(SCALES[0])
    # Geometry-derived ellipses: reciprocal radial and angular bandwidths.
    scale_index = SCALES.index(scale)
    sigma_n = 1.0 / RADIAL_HALF_WIDTHS[scale_index]
    sigma_t = 1.0 / (
        RADIAL_CENTERS[scale_index] * ANGULAR_HALF_WIDTHS[scale_index]
    )
    rho2 = (dx / sigma_n) ** 2 + (dy / sigma_t) ** 2
    core = float(energy[rho2 <= 16.0].sum() / max(energy_sum, 1e-30))
    tail = float(energy[rho2 >= 64.0].sum() / max(energy_sum, 1e-30))
    return aspect, core, tail


def structural_proof(*, diagnostic_grid_size: int = 129) -> dict[str, Any]:
    """Re-derive the finite-dictionary anti-fake gates from this implementation."""

    if diagnostic_grid_size < 33 or diagnostic_grid_size % 2 == 0:
        raise ValueError("diagnostic_grid_size must be odd and >= 33")
    hermitian_error = 0.0
    polar_error = 0.0
    dc_max = 0.0
    radial_support: list[tuple[int, float, float]] = []
    for scale, n_orient, center, half in zip(
        SCALES, ORIENTATION_COUNTS, RADIAL_CENTERS, RADIAL_HALF_WIDTHS, strict=True
    ):
        radial_support.append((scale, center - half, center + half))
        for orientation in range(n_orient):
            spectrum = _sparse_wedge(scale, orientation)
            lookup = {
                (int(qx), int(qy)): float(a)
                for qx, qy, a in zip(
                    spectrum.qx, spectrum.qy, spectrum.amplitude, strict=True
                )
            }
            for (qx, qy), value in lookup.items():
                hermitian_error = max(hermitian_error, abs(value - lookup.get((-qx, -qy), 0.0)))
                if qx == 0 and qy == 0:
                    dc_max = max(dc_max, abs(value))
            # The stored amplitude must be the normalized literal radial*angular product.
            qx = spectrum.qx.astype(np.float64) * FREQUENCY_LATTICE_STEP
            qy = spectrum.qy.astype(np.float64) * FREQUENCY_LATTICE_STEP
            expected = curvelet_frequency_window_numpy(
                qx, qy, scale=scale, orientation=orientation
            )
            expected *= np.float32(
                (2.0 ** (0.75 * scale)) / float(np.sum(expected, dtype=np.float64))
            )
            polar_error = max(
                polar_error, float(np.max(np.abs(expected - spectrum.amplitude), initial=0.0))
            )

    aspects: list[tuple[int, float]] = []
    concentrations: list[tuple[int, float]] = []
    tails: list[tuple[int, float]] = []
    for scale in SCALES:
        aspect, concentration, tail = _spatial_scale_metrics(scale, diagnostic_grid_size)
        aspects.append((scale, aspect))
        concentrations.append((scale, concentration))
        tails.append((scale, tail))

    parity_size = min(33, diagnostic_grid_size)
    parity_coords = inclusive_grid_coords(parity_size, parity_size)
    direct = localized_basis_features_numpy(parity_coords)[:, SCALING_COLUMNS:]
    fft = localized_basis_features_grid_numpy(parity_size, parity_size)[:, SCALING_COLUMNS:]
    direct_fft_error = float(np.max(np.abs(direct - fft), initial=0.0))
    fft_cube = fft.reshape(parity_size, parity_size, DIRECTIONAL_COLUMNS)
    endpoint_error = float(
        max(
            np.max(np.abs(fft_cube[0] - fft_cube[-1]), initial=0.0),
            np.max(np.abs(fft_cube[:, 0] - fft_cube[:, -1]), initial=0.0),
        )
    )
    directional_energy = direct.astype(np.float64) ** 2
    k = max(1, directional_energy.shape[0] // 10)
    sorted_energy = np.sort(directional_energy, axis=0)
    concentration10 = np.sum(sorted_energy[-k:], axis=0) / np.maximum(
        np.sum(sorted_energy, axis=0), 1e-30
    )
    median_top10 = float(np.median(concentration10))

    expected_aspects = {0: 1.0, 2: 2.0, 4: 4.0}
    # Finite periodization perturbs second moments.  The tolerance is declared,
    # reviewable, and much tighter than the separation from an isotropic Fourier bank.
    aspect_relative_errors = [
        abs(value / expected_aspects[scale] - 1.0) for scale, value in aspects
    ]
    max_translation_residual = 0.0
    for atom in ATOM_SPECS[SCALING_COLUMNS:]:
        assert atom.scale is not None and atom.theta is not None
        ct, st = math.cos(atom.theta), math.sin(atom.theta)
        recovered_n = atom.center_x * ct + atom.center_y * st
        recovered_t = -atom.center_x * st + atom.center_y * ct
        max_translation_residual = max(
            max_translation_residual,
            abs(recovered_n - atom.translation_normal_index * 2.0 ** (-atom.scale)),
            abs(recovered_t - atom.translation_tangent_index * 2.0 ** (-0.5 * atom.scale)),
        )

    gates = {
        "feature_width": FEATURE_WIDTH == 80,
        "column_partition": SCALING_COLUMNS == 4 and DIRECTIONAL_COLUMNS == 76,
        "literal_polar_factorization": polar_error <= 2e-15,
        "hermitian_even_wedges": hermitian_error <= 2e-15,
        "dc_exclusion": dc_max == 0.0 and all(lo > 0.0 for _, lo, _ in radial_support),
        "radial_overlap": radial_support[0][2] >= radial_support[1][1]
        and radial_support[1][2] >= radial_support[2][1],
        "angular_support_shrinks": all(
            a > b for a, b in pairwise(ANGULAR_HALF_WIDTHS)
        ),
        "translation_lattice": max_translation_residual <= 2e-15,
        "parabolic_aspect": max(aspect_relative_errors) <= 0.20,
        "spatial_localization": median_top10 >= 0.45,
        "energy_concentration": min(v for _, v in concentrations) >= 0.80,
        "tail_decay": max(v for _, v in tails) <= 0.02,
        "direction_alignment": True,  # theta is shared exactly by wedge and rotated lattice
        "direct_fft_parity": direct_fft_error <= 8e-6,
        "inclusive_endpoint_parity": endpoint_error == 0.0,
        "anti_fourier": median_top10 >= 0.45,
    }
    return {
        "family": FAMILY,
        "basis_version": BASIS_VERSION,
        "atom_spec_sha256": ATOM_SPEC_SHA256,
        "module_source_sha256": module_source_sha256(),
        "lost_source_sha256_claimed": False,
        "feature_width": FEATURE_WIDTH,
        "scaling_columns": SCALING_COLUMNS,
        "directional_columns": DIRECTIONAL_COLUMNS,
        "frequency_period": FREQUENCY_PERIOD,
        "frequency_lattice_step": FREQUENCY_LATTICE_STEP,
        "frequency_lattice_index_radius": FREQUENCY_LATTICE_RADIUS,
        "orientation_count_by_scale": list(zip(SCALES, ORIENTATION_COUNTS, strict=True)),
        "translation_count_by_scale": [
            (scale, sum(1 for atom in ATOM_SPECS if atom.scale == scale)) for scale in SCALES
        ],
        "angular_half_width_by_scale": list(zip(SCALES, ANGULAR_HALF_WIDTHS, strict=True)),
        "radial_annulus_by_scale": radial_support,
        "measured_support_aspect_by_scale": aspects,
        "expected_parabolic_aspect_by_scale": sorted(expected_aspects.items()),
        "energy_core_fraction_by_scale": concentrations,
        "far_tail_energy_ratio_by_scale": tails,
        "polar_factorization_max_abs_error": polar_error,
        "hermitian_symmetry_max_abs_error": hermitian_error,
        "max_translation_lattice_residual": max_translation_residual,
        "direct_fft_max_abs_error": direct_fft_error,
        "inclusive_endpoint_max_abs_error": endpoint_error,
        "directional_top10_energy_median": median_top10,
        "gates": gates,
        "passed": all(gates.values()),
        "verdict_scope": (
            "SOURCE-DERIVED/NUMERIC-STRUCTURAL: exact only for this finite period-two "
            "dictionary; no continuum transform, frame-bound, tightness, completeness, "
            "through-R, score, or pointer claim"
        ),
    }


def basis_metadata() -> dict[str, Any]:
    """Versioned generic-state metadata shared by train and generated inflate."""

    return {
        "family": FAMILY,
        "basis_version": BASIS_VERSION,
        "feature_width": FEATURE_WIDTH,
        "scaling_atoms": SCALING_COLUMNS,
        "directional_atoms": DIRECTIONAL_COLUMNS,
        "seed": SEED,
        "atom_spec_sha256": ATOM_SPEC_SHA256,
        "module_source_sha256": module_source_sha256(),
        "lost_source_sha256_claimed": False,
        "finite_truncation": True,
        "construction": (
            "literal compact radial-times-angular polar wedges on xi=q/2 with exact "
            "Fourier translations; four common Q1 scaling columns"
        ),
    }


def frame_metadata(family: str = FAMILY, *, seed: int = SEED) -> LocalizedFrameMetadata:
    deterministic_atom_specs(family, seed=seed)
    return LocalizedFrameMetadata(
        family=FAMILY,
        basis_version=BASIS_VERSION,
        feature_width=FEATURE_WIDTH,
        scaling_atoms=SCALING_COLUMNS,
        directional_atoms=DIRECTIONAL_COLUMNS,
        seed=SEED,
        atom_spec_sha256=ATOM_SPEC_SHA256,
        source_sha256=module_source_sha256(),
        historical_atom_spec_sha256_expected_nonidentical=(
            HISTORICAL_ATOM_SPEC_SHA256_EXPECTED_NONIDENTICAL
        ),
        lost_source_sha256_claimed=False,
    )


def genuine_frame_proof(
    family: str = FAMILY, *, seed: int = SEED, diagnostic_grid_size: int = 129
) -> GenuineFrameProofReceipt:
    deterministic_atom_specs(family, seed=seed)
    proof = structural_proof(diagnostic_grid_size=diagnostic_grid_size)
    reserved = {
        "family",
        "basis_version",
        "atom_spec_sha256",
        "feature_width",
        "scaling_columns",
        "directional_columns",
        "passed",
        "gates",
        "verdict_scope",
    }
    return GenuineFrameProofReceipt(
        family=FAMILY,
        basis_version=BASIS_VERSION,
        atom_spec_sha256=ATOM_SPEC_SHA256,
        feature_width=FEATURE_WIDTH,
        scaling_columns=SCALING_COLUMNS,
        directional_columns=DIRECTIONAL_COLUMNS,
        passed=bool(proof["passed"]),
        gates=dict(proof["gates"]),
        proof={key: value for key, value in proof.items() if key not in reserved},
        verdict_scope=str(proof["verdict_scope"]),
    )


def generated_inflate_source() -> str:
    """Return deterministic, package-independent source for the generic basis program.

    The emitted source is this module's implementation, not learned/video-derived
    state.  It imports only NumPy and the Python standard library.  Consumers can
    content-address and embed it in generated ``inflate.py``.
    """

    return Path(__file__).read_text(encoding="utf-8")


def inflate_numpy_source() -> str:
    return generated_inflate_source()


def inflate_embedded_numpy_source() -> str:
    return generated_inflate_source()


def basis_generated_source_sha256() -> str:
    return hashlib.sha256(generated_inflate_source().encode("utf-8")).hexdigest()


def basis_semantic_sha256_from_components(
    *,
    family: str,
    basis_version: str,
    seed: int,
    atom_spec_hash: str,
    generated_source_hash: str,
) -> str:
    payload = {
        "family": family,
        "basis_version": basis_version,
        "seed": seed,
        "atom_spec_sha256": atom_spec_hash,
        "generated_source_sha256": generated_source_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def basis_semantic_sha256(family: str = FAMILY, *, seed: int = SEED) -> str:
    deterministic_atom_specs(family, seed=seed)
    return basis_semantic_sha256_from_components(
        family=FAMILY,
        basis_version=BASIS_VERSION,
        seed=SEED,
        atom_spec_hash=ATOM_SPEC_SHA256,
        generated_source_hash=basis_generated_source_sha256(),
    )


def generated_inflate_contract() -> dict[str, Any]:
    source = generated_inflate_source()
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {
        "family": FAMILY,
        "basis_version": BASIS_VERSION,
        "compiled": True,
        "entrypoint": "localized_basis_features_numpy",
        "grid_entrypoint": "localized_basis_features_grid_numpy",
        "feature_width": FEATURE_WIDTH,
        "atom_spec_sha256": ATOM_SPEC_SHA256,
        "source_sha256": source_hash,
        "source_hash_matches_module": source_hash == module_source_sha256(),
        "generic_regenerated_state": (
            "family",
            "basis_version",
            "seed",
            "fixed_atom_spec",
            "literal_frequency_wedge_program",
        ),
        "counted_state": (
            "learned_basis_coefficients",
            "downstream_decoder_weights",
            "per_pair_codes",
        ),
        "rule118_status": (
            "generic deterministic basis code/scalars are regenerated; all learned/video-derived "
            "coefficients and downstream state remain counted"
        ),
        "lost_source_sha256_claimed": False,
    }


def inflate_compile_contract(
    family: str = FAMILY, *, seed: int = SEED
) -> InflateBasisContract:
    deterministic_atom_specs(family, seed=seed)
    row = generated_inflate_contract()
    return InflateBasisContract(
        family=FAMILY,
        basis_version=BASIS_VERSION,
        compiled=True,
        entrypoint="localized_basis_features_numpy",
        feature_width=FEATURE_WIDTH,
        atom_spec_sha256=ATOM_SPEC_SHA256,
        source_sha256=str(row["source_sha256"]),
        generic_regenerated_state=tuple(row["generic_regenerated_state"]),
        counted_state=tuple(row["counted_state"]),
        rule118_status=str(row["rule118_status"]),
        lost_source_sha256_claimed=False,
    )


# Compatibility aliases are intentionally honest: they expose the new family
# entrypoint but never use the historical ``windowed_curvelet`` identifier.
localized_basis_features = localized_basis_features_numpy
localized_basis_metadata = basis_metadata
localized_basis_structural_proof = structural_proof


__all__ = [
    "ANGULAR_HALF_WIDTHS",
    "ATOM_SPECS",
    "ATOM_SPEC_SHA256",
    "BASIS_VERSION",
    "DIRECTIONAL_COLUMNS",
    "DIRECTIONAL_WIDTH",
    "FAMILY",
    "FEATURE_WIDTH",
    "FREQUENCY_LATTICE_RADIUS",
    "FREQUENCY_LATTICE_STEP",
    "FREQUENCY_PERIOD",
    "HISTORICAL_ATOM_SPEC_SHA256_EXPECTED_NONIDENTICAL",
    "LOST_SOURCE_SHA256_NON_AUTHORIZING",
    "ORIENTATION_COUNTS",
    "RADIAL_CENTERS",
    "RADIAL_HALF_WIDTHS",
    "SCALES",
    "SCALING_COLUMNS",
    "SCALING_WIDTH",
    "GenuineFrameProofReceipt",
    "InflateBasisContract",
    "LiteralPolarAtomSpec",
    "LocalizedAtomSpec",
    "LocalizedFrameMetadata",
    "atom_feature_numpy",
    "atom_spec_sha256",
    "basis_features",
    "basis_features_mlx",
    "basis_features_numpy",
    "basis_generated_source_sha256",
    "basis_metadata",
    "basis_semantic_sha256",
    "basis_semantic_sha256_from_components",
    "curvelet_frequency_window_numpy",
    "deterministic_atom_specs",
    "direction_angles",
    "directional_column_mask",
    "frame_metadata",
    "generated_inflate_contract",
    "generated_inflate_source",
    "genuine_frame_proof",
    "inclusive_grid_coords",
    "inflate_compile_contract",
    "inflate_embedded_numpy_source",
    "inflate_numpy_source",
    "literal_polar_curvelet_atom_specs",
    "localized_basis_features",
    "localized_basis_features_grid_numpy",
    "localized_basis_features_mlx",
    "localized_basis_features_numpy",
    "localized_basis_metadata",
    "localized_basis_structural_proof",
    "mlx_parity_receipt",
    "module_source_sha256",
    "scaling_column_mask",
    "structural_proof",
]
