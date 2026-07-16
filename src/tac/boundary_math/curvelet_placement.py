# SPDX-License-Identifier: MIT
"""Basis-generic placement operators for a directional finite dictionary.

The NumPy functions in this module are the portable fp32 authority.  The MLX
twins use the same operation order and are optional: importing this module does
not require MLX.  Nothing here owns a fixed-point loop or a particular curvelet
dictionary.  Callers supply static per-column scale and angle metadata, making
the operators usable by training, resume, and generated receivers alike.
Taper folding is the one deliberate exception to input casting: the NumPy fold
preserves the checkpoint weight dtype exactly, while its MLX mirror accepts the
deployment fp32 dtype only.

Projective covariance distinguishes vectors from covectors.  For a chart
``y = H(x)`` with Jacobian ``J`` a boundary tangent is pushed forward by ``J``;
a boundary normal covector is pulled through by ``J**-T``.  Conflating these two
operations is incorrect for a non-orthogonal chart.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "NUMPY_AUTHORITY_DTYPE",
    "NUMPY_MLX_PARITY_ATOL",
    "CurveletPlacementError",
    "TaperFoldError",
    "TaperFoldParity",
    "TaperFoldReceipt",
    "apply_orientation_gates_mlx",
    "apply_orientation_gates_numpy",
    "array_sha256",
    "fold_taper_into_in_proj_mlx",
    "fold_taper_into_in_proj_numpy",
    "native_orientation_gates_mlx",
    "native_orientation_gates_numpy",
    "normal_covector_to_chart_mlx",
    "normal_covector_to_chart_numpy",
    "orientation_gates_mlx",
    "orientation_gates_numpy",
    "orientation_metadata_from_atom_specs",
    "projective_jacobian_mlx",
    "projective_jacobian_numpy",
    "projective_map_mlx",
    "projective_map_numpy",
    "tangent_vector_to_chart_mlx",
    "tangent_vector_to_chart_numpy",
    "transform_normal_covector_mlx",
    "transform_normal_covector_numpy",
    "transform_tangent_mlx",
    "transform_tangent_numpy",
    "verify_deploy_taper_fold_receipt",
    "verify_taper_fold_numpy",
    "verify_taper_fold_receipt",
    "verify_taper_fold_receipt_self_consistency",
]

NUMPY_AUTHORITY_DTYPE = np.float32
"""All placement inputs are cast to fp32 before the first arithmetic operation."""

NUMPY_MLX_PARITY_ATOL = 3e-6
"""Absolute/relative tolerance for the explicitly ordered NumPy/MLX fp32 mirrors."""

_EPS = np.float32(1e-12)
_FOLD_VERSION = "curvelet_taper_fold_v1"


class CurveletPlacementError(ValueError):
    """A shape, finiteness, or projective-chart contract violation."""


class TaperFoldError(CurveletPlacementError):
    """A taper fold is ambiguous, inconsistent, or already applied."""


def _numpy_vector(value: Any, name: str) -> np.ndarray:
    out = np.asarray(value)
    if out.ndim < 1 or out.shape[-1] != 2:
        raise CurveletPlacementError(f"{name} must have shape (..., 2); got {out.shape}")
    if not np.issubdtype(out.dtype, np.number) or not np.isfinite(out).all():
        raise CurveletPlacementError(f"{name} must contain finite numeric values")
    return out.astype(NUMPY_AUTHORITY_DTYPE, copy=False)


def _numpy_matrix(value: Any, width: int, name: str) -> np.ndarray:
    out = np.asarray(value)
    if out.ndim < 2 or out.shape[-2:] != (width, width):
        raise CurveletPlacementError(
            f"{name} must have shape (..., {width}, {width}); got {out.shape}"
        )
    if not np.issubdtype(out.dtype, np.number) or not np.isfinite(out).all():
        raise CurveletPlacementError(f"{name} must contain finite numeric values")
    return out.astype(NUMPY_AUTHORITY_DTYPE, copy=False)


def _matrix_for_vectors(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Insert point axes so per-batch matrices broadcast over point grids."""
    while matrix.ndim < vectors.ndim + 1:
        matrix = np.expand_dims(matrix, axis=-3)
    return matrix


def _normalise_numpy(value: np.ndarray, name: str, eps: float) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= eps):
        raise CurveletPlacementError(f"{name} contains a zero or singular direction")
    return np.asarray(value / norm, dtype=NUMPY_AUTHORITY_DTYPE)


def projective_map_numpy(
    points: Any, homography: Any, *, eps: float = _EPS
) -> np.ndarray:
    """Apply a 3x3 projective chart to ``(..., 2)`` points (NumPy authority)."""
    x = _numpy_vector(points, "points")
    h = _matrix_for_vectors(_numpy_matrix(homography, 3, "homography"), x)
    homogeneous = np.concatenate(
        (x, np.ones((*x.shape[:-1], 1), dtype=NUMPY_AUTHORITY_DTYPE)), axis=-1
    )
    mapped_h = np.matmul(h, homogeneous[..., None])[..., 0]
    denominator = mapped_h[..., 2:3]
    if np.any(np.abs(denominator) <= eps):
        raise CurveletPlacementError("projective chart maps a point to infinity")
    mapped = mapped_h[..., :2] / denominator
    if not np.isfinite(mapped).all():
        raise CurveletPlacementError("projective chart produced non-finite coordinates")
    return np.asarray(mapped, dtype=NUMPY_AUTHORITY_DTYPE)


def projective_jacobian_numpy(
    points: Any, homography: Any, *, eps: float = _EPS
) -> np.ndarray:
    """Return ``D H(x)`` for a 3x3 homography at ``(..., 2)`` points."""
    x = _numpy_vector(points, "points")
    h = _matrix_for_vectors(_numpy_matrix(homography, 3, "homography"), x)
    homogeneous = np.concatenate(
        (x, np.ones((*x.shape[:-1], 1), dtype=NUMPY_AUTHORITY_DTYPE)), axis=-1
    )
    mapped_h = np.matmul(h, homogeneous[..., None])[..., 0]
    u, v, w = mapped_h[..., 0], mapped_h[..., 1], mapped_h[..., 2]
    if np.any(np.abs(w) <= eps):
        raise CurveletPlacementError("projective Jacobian is undefined at infinity")
    w2 = w * w
    j00 = (h[..., 0, 0] * w - u * h[..., 2, 0]) / w2
    j01 = (h[..., 0, 1] * w - u * h[..., 2, 1]) / w2
    j10 = (h[..., 1, 0] * w - v * h[..., 2, 0]) / w2
    j11 = (h[..., 1, 1] * w - v * h[..., 2, 1]) / w2
    jacobian = np.stack((np.stack((j00, j01), -1), np.stack((j10, j11), -1)), -2)
    if not np.isfinite(jacobian).all():
        raise CurveletPlacementError("projective Jacobian contains non-finite values")
    return np.asarray(jacobian, dtype=NUMPY_AUTHORITY_DTYPE)


def transform_tangent_numpy(
    tangent: Any, jacobian: Any, *, eps: float = _EPS
) -> np.ndarray:
    """Push a tangent vector forward: ``J t / ||J t||``."""
    t = _numpy_vector(tangent, "tangent")
    j = _matrix_for_vectors(_numpy_matrix(jacobian, 2, "jacobian"), t)
    pushed = np.matmul(j, t[..., None])[..., 0]
    return _normalise_numpy(pushed, "transformed tangent", eps)


def transform_normal_covector_numpy(
    normal: Any, jacobian: Any, *, eps: float = _EPS
) -> np.ndarray:
    """Transform a normal covector: ``J**-T n / ||J**-T n||``."""
    n = _numpy_vector(normal, "normal")
    j = _matrix_for_vectors(_numpy_matrix(jacobian, 2, "jacobian"), n)
    a, b = j[..., 0, 0], j[..., 0, 1]
    c, d = j[..., 1, 0], j[..., 1, 1]
    determinant = a * d - b * c
    if np.any(np.abs(determinant) <= eps):
        raise CurveletPlacementError("normal covector transform needs an invertible Jacobian")
    # J^-T n = [[d,-c],[-b,a]] n / det.  The common det scale is retained
    # until normalization so orientation is also correct for det(J) < 0.
    pulled = np.stack(
        ((d * n[..., 0] - c * n[..., 1]) / determinant,
         (-b * n[..., 0] + a * n[..., 1]) / determinant),
        axis=-1,
    )
    return _normalise_numpy(pulled, "transformed normal covector", eps)


# Descriptive aliases used by the basis-program integration surface.
tangent_vector_to_chart_numpy = transform_tangent_numpy
normal_covector_to_chart_numpy = transform_normal_covector_numpy


def _orientation_metadata(
    scale_ids: Any, angles: Any, scaling_scale_id: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    scales = np.asarray(scale_ids)
    theta = np.asarray(angles, dtype=NUMPY_AUTHORITY_DTYPE)
    if scales.ndim != 1 or theta.ndim != 1 or scales.shape != theta.shape:
        raise CurveletPlacementError("scale_ids and angles must be equal-length 1-D arrays")
    if scales.size == 0:
        raise CurveletPlacementError("orientation metadata cannot be empty")
    if not np.issubdtype(scales.dtype, np.integer):
        raise CurveletPlacementError("scale_ids must be integers")
    directional = scales != scaling_scale_id
    if np.any(directional & ~np.isfinite(theta)):
        raise CurveletPlacementError("directional-column angles must be finite")
    groups = tuple(int(s) for s in np.unique(scales[directional]))
    return scales.astype(np.int64, copy=False), theta, groups


def orientation_metadata_from_atom_specs(
    atom_specs: Any, *, scaling_scale_id: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """Adapt ordered literal-basis atom specs to the generic gate metadata.

    Specs need ``column``, ``kind``, ``scale``, and ``theta`` attributes.  This
    deliberately uses structural typing so the placement layer does not import
    or own a particular basis implementation.
    """
    specs = tuple(atom_specs)
    if not specs:
        raise CurveletPlacementError("atom_specs cannot be empty")
    scales = np.empty(len(specs), dtype=np.int64)
    angles = np.full(len(specs), np.nan, dtype=np.float32)
    for expected_column, spec in enumerate(specs):
        if getattr(spec, "column", None) != expected_column:
            raise CurveletPlacementError("atom_specs must be ordered by contiguous column index")
        kind = getattr(spec, "kind", None)
        if kind == "scaling":
            scales[expected_column] = scaling_scale_id
        elif kind == "directional":
            scale = getattr(spec, "scale", None)
            theta = getattr(spec, "theta", None)
            if not isinstance(scale, (int, np.integer)) or not np.isfinite(theta):
                raise CurveletPlacementError("directional atom specs need integer scale and finite theta")
            if int(scale) == scaling_scale_id:
                raise CurveletPlacementError("directional scale collides with scaling_scale_id")
            scales[expected_column] = int(scale)
            angles[expected_column] = float(theta)
        else:
            raise CurveletPlacementError(f"unknown atom kind {kind!r}")
    return scales, angles


def orientation_gates_numpy(
    normal_covectors: Any,
    scale_ids: Any,
    angles: Any,
    *,
    kappa: float,
    scaling_scale_id: int = -1,
    eps: float = _EPS,
) -> np.ndarray:
    """Build deterministic per-scale projective-angle gates.

    Directional columns use ``exp(kappa*cos(2*(theta-angle(normal))))`` and
    are normalized to unit l2 norm independently within every scale.  Columns
    whose ``scale_id == scaling_scale_id`` are identity gates exactly equal to
    one.  The doubled angle makes the gate projective: ``n`` and ``-n`` agree.
    """
    n = _numpy_vector(normal_covectors, "normal_covectors")
    n = _normalise_numpy(n, "normal_covectors", eps)
    if not np.isfinite(kappa) or kappa < 0:
        raise CurveletPlacementError("kappa must be finite and non-negative")
    scales, theta, groups = _orientation_metadata(scale_ids, angles, scaling_scale_id)
    flip = (n[..., 0] < 0) | ((n[..., 0] == 0) & (n[..., 1] < 0))
    n = np.where(flip[..., None], -n, n)
    normal_angle = np.arctan2(n[..., 1], n[..., 0])
    gates = np.ones((*normal_angle.shape, scales.size), dtype=NUMPY_AUTHORITY_DTYPE)
    for scale in groups:
        index = np.flatnonzero(scales == scale)
        logits = np.float32(kappa) * np.cos(
            np.float32(2.0) * (theta[index] - normal_angle[..., None])
        )
        shift = np.max(logits, axis=-1, keepdims=True)
        unnormalised = np.exp(logits - shift)
        denominator = np.sqrt(np.sum(unnormalised * unnormalised, axis=-1, keepdims=True))
        if np.any(denominator <= eps) or not np.isfinite(denominator).all():
            raise CurveletPlacementError(f"orientation gate scale {scale} cannot be normalized")
        gates[..., index] = unnormalised / denominator
    return gates


native_orientation_gates_numpy = orientation_gates_numpy


def apply_orientation_gates_numpy(features: Any, gates: Any) -> np.ndarray:
    """Multiply same-width basis/gate columns in the fp32 authority dtype."""
    values = np.asarray(features, dtype=NUMPY_AUTHORITY_DTYPE)
    q = np.asarray(gates, dtype=NUMPY_AUTHORITY_DTYPE)
    if values.ndim < 1 or q.ndim < 1 or values.shape[-1] != q.shape[-1]:
        raise CurveletPlacementError(
            f"features and gates need the same final width; got {values.shape} and {q.shape}"
        )
    if not np.isfinite(values).all() or not np.isfinite(q).all():
        raise CurveletPlacementError("features and gates must be finite")
    return np.multiply(values, q, dtype=NUMPY_AUTHORITY_DTYPE)


def _mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise ImportError("MLX placement functions require the optional 'mlx' extra") from exc
    return mx


def _mlx_matrix_for_vectors(mx: Any, matrix: Any, vectors: Any) -> Any:
    while matrix.ndim < vectors.ndim + 1:
        matrix = mx.expand_dims(matrix, axis=-3)
    return matrix


def projective_map_mlx(points: Any, homography: Any, *, eps: float = _EPS) -> Any:
    mx = _mlx()
    x = mx.asarray(points, dtype=mx.float32)
    h = mx.asarray(homography, dtype=mx.float32)
    h = _mlx_matrix_for_vectors(mx, h, x)
    homogeneous = mx.concatenate((x, mx.ones((*x.shape[:-1], 1), dtype=x.dtype)), axis=-1)
    mapped = mx.matmul(h, homogeneous[..., None])[..., 0]
    # A data-dependent exception would force a host sync; callers validate the
    # counted chart once with the NumPy authority before using this mirror.
    denominator = mapped[..., 2:3]
    return mapped[..., :2] / mx.where(mx.abs(denominator) > eps, denominator, float("nan"))


def projective_jacobian_mlx(points: Any, homography: Any, *, eps: float = _EPS) -> Any:
    mx = _mlx()
    x = mx.asarray(points, dtype=mx.float32)
    h = mx.asarray(homography, dtype=mx.float32)
    h = _mlx_matrix_for_vectors(mx, h, x)
    homogeneous = mx.concatenate((x, mx.ones((*x.shape[:-1], 1), dtype=x.dtype)), axis=-1)
    mapped = mx.matmul(h, homogeneous[..., None])[..., 0]
    u, v, w = mapped[..., 0], mapped[..., 1], mapped[..., 2]
    w = mx.where(mx.abs(w) > eps, w, float("nan"))
    w2 = w * w
    row0 = mx.stack(((h[..., 0, 0] * w - u * h[..., 2, 0]) / w2,
                     (h[..., 0, 1] * w - u * h[..., 2, 1]) / w2), axis=-1)
    row1 = mx.stack(((h[..., 1, 0] * w - v * h[..., 2, 0]) / w2,
                     (h[..., 1, 1] * w - v * h[..., 2, 1]) / w2), axis=-1)
    return mx.stack((row0, row1), axis=-2)


def transform_tangent_mlx(tangent: Any, jacobian: Any, *, eps: float = _EPS) -> Any:
    mx = _mlx()
    tangent = mx.asarray(tangent, dtype=mx.float32)
    jacobian = _mlx_matrix_for_vectors(mx, mx.asarray(jacobian, dtype=mx.float32), tangent)
    value = mx.matmul(jacobian, tangent[..., None])[..., 0]
    norm = mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True))
    return value / mx.where(norm > eps, norm, float("nan"))


def transform_normal_covector_mlx(normal: Any, jacobian: Any, *, eps: float = _EPS) -> Any:
    mx = _mlx()
    n = mx.asarray(normal, dtype=mx.float32)
    j = mx.asarray(jacobian, dtype=mx.float32)
    j = _mlx_matrix_for_vectors(mx, j, n)
    a, b = j[..., 0, 0], j[..., 0, 1]
    c, d = j[..., 1, 0], j[..., 1, 1]
    determinant = a * d - b * c
    determinant = mx.where(mx.abs(determinant) > eps, determinant, float("nan"))
    value = mx.stack(((d * n[..., 0] - c * n[..., 1]) / determinant,
                      (-b * n[..., 0] + a * n[..., 1]) / determinant), axis=-1)
    norm = mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True))
    return value / mx.where(norm > eps, norm, float("nan"))


tangent_vector_to_chart_mlx = transform_tangent_mlx
normal_covector_to_chart_mlx = transform_normal_covector_mlx


def orientation_gates_mlx(
    normal_covectors: Any,
    scale_ids: Any,
    angles: Any,
    *,
    kappa: float,
    scaling_scale_id: int = -1,
    eps: float = _EPS,
) -> Any:
    """MLX mirror of :func:`orientation_gates_numpy`."""
    mx = _mlx()
    if not np.isfinite(kappa) or kappa < 0:
        raise CurveletPlacementError("kappa must be finite and non-negative")
    scales, theta, groups = _orientation_metadata(scale_ids, angles, scaling_scale_id)
    n = mx.asarray(normal_covectors, dtype=mx.float32)
    norm = mx.sqrt(mx.sum(n * n, axis=-1, keepdims=True))
    n = n / mx.where(norm > eps, norm, float("nan"))
    flip = (n[..., 0] < 0) | ((n[..., 0] == 0) & (n[..., 1] < 0))
    n = mx.where(flip[..., None], -n, n)
    normal_angle = mx.arctan2(n[..., 1], n[..., 0])
    group_values: dict[int, tuple[Any, np.ndarray]] = {}
    for scale in groups:
        index = np.flatnonzero(scales == scale)
        group_theta = mx.asarray(theta[index], dtype=n.dtype)
        logits = np.float32(kappa) * mx.cos(
            np.float32(2.0) * (group_theta - normal_angle[..., None])
        )
        shift = mx.max(logits, axis=-1, keepdims=True)
        values = mx.exp(logits - shift)
        denominator = mx.sqrt(mx.sum(values * values, axis=-1, keepdims=True))
        group_values[scale] = (
            values / mx.where(denominator > eps, denominator, float("nan")),
            index,
        )
    columns = []
    for column, scale in enumerate(scales):
        if int(scale) == scaling_scale_id:
            columns.append(mx.ones(normal_angle.shape, dtype=n.dtype))
        else:
            values, index = group_values[int(scale)]
            position = int(np.flatnonzero(index == column)[0])
            columns.append(values[..., position])
    return mx.stack(columns, axis=-1)


native_orientation_gates_mlx = orientation_gates_mlx


def apply_orientation_gates_mlx(features: Any, gates: Any) -> Any:
    mx = _mlx()
    if features.shape[-1] != gates.shape[-1]:
        raise CurveletPlacementError(
            f"features and gates need the same final width; got {features.shape} and {gates.shape}"
        )
    return mx.asarray(features, dtype=mx.float32) * mx.asarray(gates, dtype=mx.float32)


def array_sha256(value: Any) -> str:
    """Content hash binding an array's canonical little-endian dtype and shape."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TaperFoldError("object arrays cannot enter a taper-fold receipt")
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    header = json.dumps(
        {"dtype": dtype.str, "shape": list(canonical.shape)}, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TaperFoldReceipt:
    """Content-bound proof that one (and only one) deploy fold occurred."""

    version: str
    folded: bool
    feature_axis: int
    weight_shape: tuple[int, ...]
    taper_length: int
    source_weight_sha256: str
    taper_sha256: str
    folded_weight_sha256: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaperFoldParity:
    """Algebra check for ``W (D_w v) == (W D_w) v``."""

    allclose: bool
    max_abs_error: float
    atol: float
    rtol: float


def _receipt_hash(fields: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(fields), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_fold_receipt(
    source: np.ndarray, taper: np.ndarray, folded: np.ndarray, feature_axis: int
) -> TaperFoldReceipt:
    fields: dict[str, Any] = {
        "version": _FOLD_VERSION,
        "folded": True,
        "feature_axis": feature_axis,
        "weight_shape": list(source.shape),
        "taper_length": int(taper.size),
        "source_weight_sha256": array_sha256(source),
        "taper_sha256": array_sha256(taper),
        "folded_weight_sha256": array_sha256(folded),
    }
    return TaperFoldReceipt(
        version=_FOLD_VERSION,
        folded=True,
        feature_axis=feature_axis,
        weight_shape=tuple(source.shape),
        taper_length=int(taper.size),
        source_weight_sha256=fields["source_weight_sha256"],
        taper_sha256=fields["taper_sha256"],
        folded_weight_sha256=fields["folded_weight_sha256"],
        receipt_sha256=_receipt_hash(fields),
    )


def _validated_receipt_data(
    receipt: TaperFoldReceipt | Mapping[str, Any],
) -> dict[str, Any]:
    data = receipt.to_dict() if isinstance(receipt, TaperFoldReceipt) else dict(receipt)
    claimed = data.pop("receipt_sha256", None)
    if data.get("version") != _FOLD_VERSION or data.get("folded") is not True:
        raise TaperFoldError("receipt is not a recognized folded deploy state")
    # JSON serialization changes tuples to lists; canonicalize before hashing.
    data["weight_shape"] = list(data.get("weight_shape", []))
    if claimed != _receipt_hash(data):
        raise TaperFoldError("taper-fold receipt hash mismatch")
    shape = data["weight_shape"]
    axis = data.get("feature_axis")
    if (
        not shape
        or any(not isinstance(size, int) or size < 0 for size in shape)
        or not isinstance(axis, int)
        or not 0 <= axis < len(shape)
        or data.get("taper_length") != shape[axis]
    ):
        raise TaperFoldError("taper-fold receipt shape/axis metadata is inconsistent")
    for field in ("source_weight_sha256", "taper_sha256", "folded_weight_sha256"):
        digest = data.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise TaperFoldError(f"taper-fold receipt has invalid {field}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise TaperFoldError(f"taper-fold receipt has invalid {field}") from exc
    return data


def verify_taper_fold_receipt_self_consistency(
    receipt: TaperFoldReceipt | Mapping[str, Any],
) -> bool:
    """Verify only the serialized receipt's own schema and canonical hash.

    This helper is intentionally named as a weaker check.  Deploy consumers
    should call :func:`verify_deploy_taper_fold_receipt` so a stale but
    internally valid receipt cannot authorize different checkpoint arrays.
    """
    _validated_receipt_data(receipt)
    return True


def verify_taper_fold_receipt(
    receipt: TaperFoldReceipt | Mapping[str, Any],
    *,
    source_weight: Any | None = None,
    taper: Any | None = None,
    folded_weight: Any | None = None,
    require_array_bindings: bool = False,
) -> bool:
    """Verify receipt integrity and any supplied source/taper/deploy arrays.

    With ``require_array_bindings=True`` all three arrays are mandatory and
    their content hashes, shapes, and exact elementwise fold are checked.
    """
    data = _validated_receipt_data(receipt)
    arrays = (source_weight, taper, folded_weight)
    if require_array_bindings and any(value is None for value in arrays):
        raise TaperFoldError("deploy receipt verification requires source, taper, and folded arrays")
    source = None if source_weight is None else np.asarray(source_weight)
    scale = None if taper is None else np.asarray(taper)
    folded = None if folded_weight is None else np.asarray(folded_weight)
    expected_shape = tuple(data["weight_shape"])
    if source is not None and (
        source.shape != expected_shape or array_sha256(source) != data["source_weight_sha256"]
    ):
        raise TaperFoldError("taper-fold receipt does not bind the supplied source weight")
    if scale is not None and (
        scale.shape != (data["taper_length"],) or array_sha256(scale) != data["taper_sha256"]
    ):
        raise TaperFoldError("taper-fold receipt does not bind the supplied taper")
    if folded is not None and (
        folded.shape != expected_shape or array_sha256(folded) != data["folded_weight_sha256"]
    ):
        raise TaperFoldError("taper-fold receipt does not bind the supplied folded weight")
    if source is not None and scale is not None and folded is not None:
        reshape = [1] * source.ndim
        reshape[data["feature_axis"]] = scale.size
        expected = np.multiply(source, scale.reshape(reshape), dtype=source.dtype)
        if not np.array_equal(expected, folded):
            raise TaperFoldError("supplied arrays do not satisfy the exact taper-fold algebra")
    return True


def verify_deploy_taper_fold_receipt(
    receipt: TaperFoldReceipt | Mapping[str, Any],
    *,
    source_weight: Any,
    taper: Any,
    folded_weight: Any,
) -> bool:
    """Deploy-strength receipt verification with all array bindings required."""
    return verify_taper_fold_receipt(
        receipt,
        source_weight=source_weight,
        taper=taper,
        folded_weight=folded_weight,
        require_array_bindings=True,
    )


def _prepare_fold(
    in_proj_weight: Any,
    taper: Any,
    feature_axis: int,
    existing_receipt: TaperFoldReceipt | Mapping[str, Any] | None,
    expected_source_sha256: str | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if existing_receipt is not None:
        verify_taper_fold_receipt_self_consistency(existing_receipt)
        raise TaperFoldError("refusing to fold taper twice: deploy state is already folded")
    weight = np.asarray(in_proj_weight)
    scale = np.asarray(taper)
    if weight.ndim < 1 or not np.issubdtype(weight.dtype, np.floating):
        raise TaperFoldError("in_proj_weight must be a floating-point array")
    if scale.ndim != 1 or not np.issubdtype(scale.dtype, np.floating):
        raise TaperFoldError("taper must be a 1-D floating-point array")
    if not np.isfinite(weight).all() or not np.isfinite(scale).all():
        raise TaperFoldError("weights and taper must be finite")
    axis = int(feature_axis)
    if axis < 0:
        axis += weight.ndim
    if not 0 <= axis < weight.ndim:
        raise TaperFoldError(f"feature_axis {feature_axis} is invalid for shape {weight.shape}")
    if weight.shape[axis] != scale.size:
        raise TaperFoldError(
            f"taper width {scale.size} does not match weight axis {axis} width {weight.shape[axis]}"
        )
    if expected_source_sha256 is not None and array_sha256(weight) != expected_source_sha256:
        raise TaperFoldError("unfolded in_proj weight hash does not match checkpoint custody")
    return weight, scale, axis


def fold_taper_into_in_proj_numpy(
    in_proj_weight: Any,
    taper: Any,
    *,
    feature_axis: int = 1,
    existing_receipt: TaperFoldReceipt | Mapping[str, Any] | None = None,
    expected_source_sha256: str | None = None,
) -> tuple[np.ndarray, TaperFoldReceipt]:
    """Fold ``D_w`` into an input projection and emit a content-bound receipt.

    ``existing_receipt`` is the deploy-checkpoint state marker.  Passing any
    valid folded receipt refuses the operation, preventing an accidental second
    fold.  The input array is never mutated.
    """
    weight, scale, axis = _prepare_fold(
        in_proj_weight, taper, feature_axis, existing_receipt, expected_source_sha256
    )
    shape = [1] * weight.ndim
    shape[axis] = scale.size
    folded = np.multiply(weight, scale.reshape(shape), dtype=weight.dtype)
    return folded, _make_fold_receipt(weight, scale, folded, axis)


def fold_taper_into_in_proj_mlx(
    in_proj_weight: Any,
    taper: Any,
    *,
    feature_axis: int = 1,
    existing_receipt: TaperFoldReceipt | Mapping[str, Any] | None = None,
    expected_source_sha256: str | None = None,
) -> tuple[Any, TaperFoldReceipt]:
    """MLX mirror of :func:`fold_taper_into_in_proj_numpy`."""
    mx = _mlx()
    source_np, taper_np, axis = _prepare_fold(
        np.asarray(in_proj_weight),
        np.asarray(taper),
        feature_axis,
        existing_receipt,
        expected_source_sha256,
    )
    if source_np.dtype != np.float32 or taper_np.dtype != np.float32:
        raise TaperFoldError("MLX taper folding requires fp32 source weights and taper")
    shape = [1] * source_np.ndim
    shape[axis] = taper_np.size
    folded = mx.asarray(in_proj_weight, dtype=mx.float32) * mx.reshape(
        mx.asarray(taper, dtype=mx.float32), shape
    )
    mx.eval(folded)
    receipt = _make_fold_receipt(source_np, taper_np, np.asarray(folded), axis)
    return folded, receipt


def verify_taper_fold_numpy(
    in_proj_weight: Any,
    taper: Any,
    folded_weight: Any,
    features: Any,
    *,
    atol: float = 2e-6,
    rtol: float = 2e-6,
) -> TaperFoldParity:
    """Check the deploy-fold algebra for a conventional ``(out, in)`` layer."""
    weight = np.asarray(in_proj_weight)
    folded = np.asarray(folded_weight)
    scale = np.asarray(taper)
    values = np.asarray(features)
    if weight.ndim != 2 or folded.shape != weight.shape:
        raise TaperFoldError("fold parity expects equal (out_features, in_features) weights")
    if scale.shape != (weight.shape[1],) or values.shape[-1] != weight.shape[1]:
        raise TaperFoldError("fold parity feature/taper width does not match in_proj input width")
    before = np.matmul(values * scale, weight.T)
    after = np.matmul(values, folded.T)
    difference = np.abs(before - after)
    maximum = float(difference.max(initial=0.0))
    return TaperFoldParity(
        allclose=bool(np.allclose(before, after, atol=atol, rtol=rtol)),
        max_abs_error=maximum,
        atol=float(atol),
        rtol=float(rtol),
    )
