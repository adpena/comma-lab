# SPDX-License-Identifier: MIT
"""Deterministic local-relaxation solver for the V10 scorer-plane program.

This module deliberately separates proposals from authority.  ``seg_q`` is the
``vjp_custody_pair.v1`` aggregate/local diagonal relaxation; it is useful for
constructing a candidate but it is not the full global cell-to-plane Jacobian.
Consequently every realized candidate must pass a caller-supplied native-fp32
hard oracle before it can be admitted.  Nothing in this module is used by the
archive decoder and no scorer or source frame is loaded here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Factor2ExactVerification,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)

PAIR_SCHEMA = "vjp_custody_pair.v1"
RECEIVER_ARITHMETIC = "native_float32_cpu_torch"
ACTIVE_ARRANGEMENT = "cached_winner_native_rival"
REPRESENTATION = "solver_scorer_plane_y_with_camera_adjoint_x"
WINNER_SOURCE = "cached_lstars_verified_against_fresh_native_fp32_logits"
RIVAL_SOURCE = "fresh_native_fp32_logits_highest_nonwinner_not_cached"
SCORER_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_CLASSES = 5
HARD_ORACLE_SCHEMA = "v10_constructive_hard_oracle_decision.v1"
PROJECTION_DTYPE = np.dtype("float32")

EXPECTED_SOURCE_HASHES = {
    "cache_sha256": "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6",
    "modules_sha256": "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa",
    "frame_utils_sha256": "d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90",
    "segnet_weights_sha256": "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
    "posenet_weights_sha256": "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576",
}
_FROZEN_SOURCE_HASHES = tuple(EXPECTED_SOURCE_HASHES.items())

_PAIR_FIELDS = frozenset(
    {
        "pair_id",
        "custody_json",
        "winner",
        "rival",
        "cached_margin",
        "native_margin",
        "head_pair_norms",
        "seg_g_y",
        "seg_g_x",
        "seg_q",
        "seg_local_lipschitz",
        "pose_j_y",
        "pose_j_x",
    }
)


class ConstructiveSolveError(ValueError):
    """Malformed custody, infeasible relaxation, or failed admission."""


@dataclass(frozen=True)
class VJPCustodyPair:
    pair_id: int
    winner: np.ndarray
    rival: np.ndarray
    cached_margin: np.ndarray
    native_margin: np.ndarray
    head_pair_norms: np.ndarray
    seg_g_y: np.ndarray
    seg_g_x: np.ndarray
    seg_q: np.ndarray
    seg_local_lipschitz: np.ndarray
    pose_j_y: np.ndarray
    pose_j_x: np.ndarray
    metadata: Mapping[str, Any]
    npz_sha256: str


@dataclass(frozen=True)
class ProjectionResult:
    delta: np.ndarray
    objective: float
    seg_min_slack: float
    pose_mse: float | None
    iterations: int
    converged: bool
    zero_band_exact: bool
    certificate_scope: str = "proposal_only_local_seg_q_relaxation_requires_hard_oracle"


@dataclass(frozen=True)
class HardOracleDecision:
    admitted: bool
    schema: str
    receiver_arithmetic: str
    realized_frame_sha256s: tuple[str, str]
    d_seg: float
    d_pose: float
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.admitted) is not bool:
            raise ConstructiveSolveError("hard oracle admitted must be an exact bool")
        if self.schema != HARD_ORACLE_SCHEMA or self.receiver_arithmetic != RECEIVER_ARITHMETIC:
            raise ConstructiveSolveError("hard oracle schema/native-fp32 marker mismatch")
        if (
            type(self.realized_frame_sha256s) is not tuple
            or len(self.realized_frame_sha256s) != 2
            or any(not _is_sha256(value) for value in self.realized_frame_sha256s)
        ):
            raise ConstructiveSolveError("hard oracle must bind exactly two realized-frame SHA-256s")
        for label, value in (("d_seg", self.d_seg), ("d_pose", self.d_pose)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
                raise ConstructiveSolveError(f"hard oracle {label} must be finite and nonnegative")
            object.__setattr__(self, label, float(value))
        if not isinstance(self.source_hashes, Mapping) or dict(self.source_hashes) != dict(_FROZEN_SOURCE_HASHES):
            raise ConstructiveSolveError("hard oracle frozen source-hash custody mismatch")
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))


@dataclass(frozen=True)
class LatticeAdmission:
    camera_frames: tuple[np.ndarray, np.ndarray]
    proofs: tuple[Factor2ExactVerification, Factor2ExactVerification]
    hard_oracle: HardOracleDecision


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConstructiveSolveError("value is not canonical-JSON serializable") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _immutable_array_copy(value: np.ndarray) -> np.ndarray:
    """Own values behind a read-only bytes buffer whose flag cannot be reopened."""

    contiguous = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _exact_pair_id(value: Any, label: str) -> int:
    if type(value) is not int or value < 0 or value > np.iinfo(np.int64).max:
        raise ConstructiveSolveError(f"{label} must be an exact non-bool nonnegative int64")
    return value


def load_vjp_custody_pair(
    path: Path,
    *,
    expected_pair_id: int,
    scorer_hw: tuple[int, int] = SCORER_HW,
    camera_hw: tuple[int, int] = CAMERA_HW,
    expected_source_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
) -> VJPCustodyPair:
    """Load one exact ``vjp_custody_pair.v1`` NPZ, refusing schema drift.

    The producer's field set, tensor dtypes/shapes, embedded hashes, source
    hashes, pair id, active arrangement, and native-fp32 receiver arithmetic
    are all part of the custody contract.  Extra NPZ fields are refused: an
    apparently harmless producer addition must be explicitly reviewed here.
    """

    expected_pair_id = _exact_pair_id(expected_pair_id, "expected_pair_id")
    path = Path(path)
    if not path.is_file():
        raise ConstructiveSolveError(f"VJP custody NPZ is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            fields = frozenset(data.files)
            if fields != _PAIR_FIELDS:
                missing = sorted(_PAIR_FIELDS - fields)
                extra = sorted(fields - _PAIR_FIELDS)
                raise ConstructiveSolveError(f"VJP custody NPZ field mismatch: missing={missing}, extra={extra}")
            metadata_raw = np.asarray(data["custody_json"])
            if metadata_raw.shape != () or metadata_raw.dtype.kind not in {"U", "S"}:
                raise ConstructiveSolveError("custody_json must be a scalar string")
            metadata = json.loads(str(metadata_raw.reshape(())))
            if not isinstance(metadata, dict):
                raise ConstructiveSolveError("custody_json must decode to an object")
            arrays = {key: np.asarray(data[key]).copy() for key in _PAIR_FIELDS - {"custody_json", "pair_id"}}
            pair_value = np.asarray(data["pair_id"])
    except ConstructiveSolveError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ConstructiveSolveError(f"cannot parse VJP custody NPZ: {path}") from exc

    if pair_value.dtype != np.int64 or pair_value.shape != ():
        raise ConstructiveSolveError("VJP pair_id must be an int64 scalar")
    pair_id = int(pair_value)
    _exact_pair_id(pair_id, "VJP pair_id")
    try:
        metadata_pair_id = _exact_pair_id(metadata.get("pair_id"), "custody metadata pair_id")
    except ConstructiveSolveError as exc:
        raise ConstructiveSolveError("pair id differs across request, NPZ, and custody metadata") from exc
    if pair_id != expected_pair_id or metadata_pair_id != pair_id:
        raise ConstructiveSolveError("pair id differs across request, NPZ, and custody metadata")
    expected_metadata = {
        "schema": PAIR_SCHEMA,
        "receiver_arithmetic": RECEIVER_ARITHMETIC,
        "active_arrangement": ACTIVE_ARRANGEMENT,
        "winner_source": WINNER_SOURCE,
        "rival_source": RIVAL_SOURCE,
        "representation": REPRESENTATION,
        "source_hashes": dict(expected_source_hashes),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ConstructiveSolveError(f"VJP custody metadata {key} mismatch")

    expected_tensors = {
        "winner": (np.dtype("int8"), scorer_hw),
        "rival": (np.dtype("int8"), scorer_hw),
        "cached_margin": (np.dtype("float32"), scorer_hw),
        "native_margin": (np.dtype("float32"), scorer_hw),
        "head_pair_norms": (np.dtype("float32"), scorer_hw),
        "seg_g_y": (np.dtype("float32"), (*scorer_hw, 3)),
        "seg_g_x": (np.dtype("float32"), (*camera_hw, 3)),
        "seg_q": (np.dtype("float32"), (*scorer_hw, 3)),
        "seg_local_lipschitz": (np.dtype("float32"), scorer_hw),
        "pose_j_y": (np.dtype("float32"), (6, 2, *scorer_hw, 3)),
        "pose_j_x": (np.dtype("float32"), (6, 2, *camera_hw, 3)),
    }
    tensor_metadata = metadata.get("tensors")
    if not isinstance(tensor_metadata, dict) or frozenset(tensor_metadata) != frozenset(expected_tensors):
        raise ConstructiveSolveError("VJP tensor metadata field set mismatch")
    for key, (dtype, shape) in expected_tensors.items():
        value = arrays[key]
        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise ConstructiveSolveError(f"VJP tensor {key} dtype/shape/finiteness mismatch")
        actual = {"dtype": str(dtype), "shape": list(shape), "sha256": _sha256_array(value)}
        if tensor_metadata.get(key) != actual:
            raise ConstructiveSolveError(f"VJP tensor {key} embedded hash metadata mismatch")

    winner, rival = arrays["winner"], arrays["rival"]
    if (
        np.any(winner < 0)
        or np.any(winner >= N_CLASSES)
        or np.any(rival < 0)
        or np.any(rival >= N_CLASSES)
        or np.any(winner == rival)
    ):
        raise ConstructiveSolveError("VJP winner/rival arrangement is invalid")
    if (
        np.any(arrays["cached_margin"] < 0)
        or np.any(arrays["head_pair_norms"] <= 0)
        or np.any(arrays["seg_local_lipschitz"] < 0)
    ):
        raise ConstructiveSolveError("VJP margins/norms leave their valid domains")
    if not np.allclose(arrays["cached_margin"], arrays["native_margin"], rtol=1e-4, atol=1e-5):
        raise ConstructiveSolveError("VJP cached/native margin agreement failed")
    rebuilt = arrays["seg_local_lipschitz"][..., None] * arrays["seg_q"]
    if not np.allclose(rebuilt, arrays["seg_g_y"], rtol=2e-6, atol=2e-7):
        raise ConstructiveSolveError("VJP seg_q/local-Lipschitz factorization failed")
    positive = arrays["seg_local_lipschitz"] > 0
    norms = np.linalg.norm(arrays["seg_q"].astype(np.float64), axis=-1)
    if np.any(np.abs(norms[positive] - 1.0) > 2e-6) or np.any(norms[~positive] != 0):
        raise ConstructiveSolveError("VJP seg_q unit/zero convention failed")
    if any(
        float(np.linalg.norm(arrays[key][row].astype(np.float64))) == 0.0
        for key in ("pose_j_y", "pose_j_x")
        for row in range(6)
    ):
        raise ConstructiveSolveError("VJP pose Jacobian contains an all-zero row")

    return VJPCustodyPair(
        pair_id=pair_id,
        metadata=metadata,
        npz_sha256=_sha256_file(path),
        **arrays,
    )


def _validated_projection_arrays(
    anchor: np.ndarray,
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(np.asarray(value, dtype=PROJECTION_DTYPE) for value in (anchor, q, lower, upper, weights))
    a, normal, lo, hi, w = np.broadcast_arrays(*values)
    if a.size == 0 or not all(np.isfinite(value).all() for value in (a, normal, lo, hi, w)):
        raise ConstructiveSolveError("projection arrays must be nonempty and finite")
    if np.any(lo > hi) or np.any(w <= 0):
        raise ConstructiveSolveError("projection requires lower<=upper and positive weights")
    return (
        np.array(a, dtype=PROJECTION_DTYPE, copy=True),
        np.array(normal, dtype=PROJECTION_DTYPE, copy=True),
        np.array(lo, dtype=PROJECTION_DTYPE, copy=True),
        np.array(hi, dtype=PROJECTION_DTYPE, copy=True),
        np.array(w, dtype=PROJECTION_DTYPE, copy=True),
    )


def project_weighted_box_halfspace(
    anchor: np.ndarray,
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: np.ndarray,
    *,
    margin: float,
    bisection_steps: int = 80,
) -> np.ndarray:
    """Project ``anchor`` in weighted norm onto a box and one half-space.

    The constraint is ``q dot delta >= -margin`` and the monotone solution is
    ``clip(anchor + lambda*q/weights, lower, upper)``.  A finite breakpoint
    bound brackets lambda; fixed-count bisection makes byte-for-byte repeated
    execution deterministic.
    """

    a, normal, lo, hi, w = _validated_projection_arrays(anchor, q, lower, upper, weights)
    if not math.isfinite(float(margin)) or margin < 0:
        raise ConstructiveSolveError("margin must be finite and nonnegative")
    if not isinstance(bisection_steps, int) or bisection_steps < 1 or bisection_steps > 256:
        raise ConstructiveSolveError("bisection_steps must lie in [1,256]")
    clipped = np.clip(a, lo, hi)
    threshold = -np.float32(margin)
    if float(np.dot(normal.ravel(), clipped.ravel())) >= threshold:
        return clipped

    best = np.where(normal > 0, hi, np.where(normal < 0, lo, clipped))
    if float(np.dot(normal.ravel(), best.ravel())) < threshold - 2e-5:
        raise ConstructiveSolveError("box/half-space relaxation is infeasible")
    nonzero = normal != 0
    if not np.any(nonzero):
        raise ConstructiveSolveError("binding half-space has an all-zero normal")
    destination = np.where(normal > 0, hi, lo)
    breakpoints = (destination[nonzero] - a[nonzero]) * w[nonzero] / normal[nonzero]
    upper_lambda = np.float32(max(0.0, float(np.max(breakpoints, initial=np.float32(0.0)))))
    if upper_lambda == 0.0:
        upper_lambda = np.float32(1.0)
    # A nextafter step avoids an endpoint being lost to one rounding operation.
    upper_lambda = np.nextafter(upper_lambda, np.float32(math.inf))
    low_lambda = np.float32(0.0)
    for _ in range(bisection_steps):
        candidate_lambda = np.float32((low_lambda + upper_lambda) * np.float32(0.5))
        candidate = np.clip(a + candidate_lambda * normal / w, lo, hi)
        if float(np.dot(normal.ravel(), candidate.ravel())) >= threshold:
            upper_lambda = candidate_lambda
        else:
            low_lambda = candidate_lambda
    result = np.clip(a + upper_lambda * normal / w, lo, hi)
    if float(np.dot(normal.ravel(), result.ravel())) < threshold - 5e-5:
        raise ConstructiveSolveError("deterministic half-space projection did not close")
    return result


def project_pixelwise_seg_relaxation(
    anchor: np.ndarray,
    seg_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: np.ndarray,
    margins: np.ndarray,
) -> np.ndarray:
    """Project independent RGB pixels onto the local ``seg_q`` relaxation.

    RGB has only three coordinates, so this routine solves the clipped KKT
    root by enumerating the at-most seven intervals induced by the six box
    breakpoints.  It avoids one Python call (and an 80-step bisection) per
    scorer pixel, which is essential for real 384x512 chunks.
    """

    a, q, lo, hi, w = _validated_projection_arrays(anchor, seg_q, lower, upper, weights)
    if a.ndim < 1 or a.shape[-1] != 3:
        raise ConstructiveSolveError("pixelwise Seg relaxation requires a final RGB axis")
    m = np.asarray(margins, dtype=PROJECTION_DTYPE)
    if m.shape != a.shape[:-1] or not np.isfinite(m).all() or np.any(m < 0):
        raise ConstructiveSolveError("pixelwise margins must be finite nonnegative and match HxW")
    result = np.clip(a, lo, hi)
    threshold = -m
    dot = np.sum(q * result, axis=-1)
    binding = dot < threshold
    if not np.any(binding):
        return result
    best = np.where(q > 0, hi, np.where(q < 0, lo, result))
    if np.any(binding & (np.sum(q * best, axis=-1) < threshold - 2e-5)):
        raise ConstructiveSolveError("pixelwise box/Seg relaxation is infeasible")

    direction = q / w
    with np.errstate(divide="ignore", invalid="ignore"):
        lower_crossing = (lo - a) / direction
        upper_crossing = (hi - a) / direction
    crossings = np.concatenate((lower_crossing, upper_crossing), axis=-1)
    crossings = np.where(np.isfinite(crossings) & (crossings > 0), crossings, np.inf)
    crossings.sort(axis=-1)
    unresolved = binding.copy()
    left = np.zeros_like(m)
    for interval in range(7):
        if not np.any(unresolved):
            break
        right = crossings[..., interval] if interval < 6 else np.full_like(m, np.inf)
        raw_sample = np.where(np.isfinite(right), (left + right) * 0.5, left + 1.0)
        sample = np.where(unresolved, raw_sample, 0.0)
        sample_value = np.clip(a + sample[..., None] * direction, lo, hi)
        active = (sample_value > lo) & (sample_value < hi)
        slope = np.sum(np.where(active, q * direction, 0.0), axis=-1)
        intercept = np.sum(q * sample_value, axis=-1) - slope * sample
        candidate = np.divide(
            threshold - intercept,
            slope,
            out=np.full_like(m, np.inf),
            where=slope > 0,
        )
        inside = unresolved & (candidate >= left - 2e-5) & (candidate <= right + 2e-5)
        result[inside] = np.clip(
            a[inside] + candidate[inside][..., None] * direction[inside],
            lo[inside],
            hi[inside],
        )
        unresolved &= ~inside
        left = right
    if np.any(unresolved):
        raise ConstructiveSolveError("pixelwise clipped KKT breakpoint solve did not close")
    if np.any(np.sum(q * result, axis=-1) < threshold - 5e-5):
        raise ConstructiveSolveError("pixelwise local Seg projection violates its half-space")
    return result


# PROJECT_PARITY_WAIVED: encode-side constructive-solver constraint projection; solution stored post-projection
def project_rank6_pose_ellipsoid(
    value: np.ndarray,
    pose_j_y: np.ndarray,
    weights: np.ndarray,
    *,
    tau_pose: float,
    bisection_steps: int = 96,
) -> np.ndarray:
    """Weighted projection onto ``||J delta||^2/6 <= tau_pose``.

    The high-dimensional solve is reduced to the eigensystem of the 6x6 Gram
    matrix ``J W^-1 J^T``.  This is a real rank-six projection, not a norm
    clipping proxy.
    """

    x = np.asarray(value, dtype=PROJECTION_DTYPE)
    w = np.asarray(weights, dtype=PROJECTION_DTYPE)
    jac = np.asarray(pose_j_y, dtype=PROJECTION_DTYPE)
    if x.size == 0 or w.shape != x.shape or jac.shape != (6, *x.shape):
        raise ConstructiveSolveError("pose projection requires J shape (6,*delta.shape)")
    if not all(np.isfinite(v).all() for v in (x, w, jac)) or np.any(w <= 0):
        raise ConstructiveSolveError("pose projection inputs must be finite with positive weights")
    if not math.isfinite(float(tau_pose)) or tau_pose < 0:
        raise ConstructiveSolveError("tau_pose must be finite and nonnegative")
    flat_x, flat_w, flat_j = x.ravel(), w.ravel(), jac.reshape(6, -1)
    pose = flat_j @ flat_x
    limit_sq = np.float32(6.0) * np.float32(tau_pose)
    if float(pose @ pose) <= limit_sq + 1e-20:
        return x.copy()
    gram = (flat_j / flat_w[None, :]) @ flat_j.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    coefficients = eigenvectors.T @ pose
    if tau_pose == 0.0:
        inverse = np.zeros_like(eigenvalues)
        positive = eigenvalues > np.float32(max(1.0, float(eigenvalues[-1])) * 1e-6)
        inverse[positive] = np.float32(1.0) / eigenvalues[positive]
        correction6 = eigenvectors @ (inverse * coefficients)
        projected = flat_x - (flat_j.T @ correction6) / flat_w
    else:

        def residual(multiplier: np.float32) -> float:
            scaled = coefficients / (np.float32(1.0) + multiplier * eigenvalues)
            return float(float(scaled @ scaled) - float(limit_sq))

        low, high = np.float32(0.0), np.float32(1.0)
        while residual(high) > 0.0:
            high = np.float32(high * np.float32(2.0))
            if not math.isfinite(float(high)):
                raise ConstructiveSolveError("pose projection root could not be bracketed")
        for _ in range(bisection_steps):
            middle = np.float32((low + high) * np.float32(0.5))
            if residual(middle) > 0.0:
                low = middle
            else:
                high = middle
        factors = high / (np.float32(1.0) + high * eigenvalues)
        correction6 = eigenvectors @ (factors * coefficients)
        projected = flat_x - (flat_j.T @ correction6) / flat_w
    result = projected.reshape(x.shape)
    closure_tolerance = max(2e-5, 2e-5 * float(limit_sq))
    if float(np.sum((flat_j @ projected) ** 2)) > limit_sq + closure_tolerance:
        raise ConstructiveSolveError("rank-six pose projection did not close")
    return result


def solve_constructive_projection(
    *,
    target_planes: np.ndarray,
    predictor_planes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: np.ndarray,
    seg_q: np.ndarray,
    seg_margins: np.ndarray,
    pose_j_y: np.ndarray | None = None,
    tau_pose: float | None = None,
    max_dykstra_iterations: int = 200,
    tolerance: float = 2e-5,
) -> ProjectionResult:
    """Solve the two-plane box/Seg relaxation, optionally intersecting Pose.

    ``seg_q`` constrains frame 1 only.  It is explicitly a proposal surface;
    the returned result is never an authority certificate.
    """

    target = np.asarray(target_planes)
    predictor = np.asarray(predictor_planes)
    if target.shape != predictor.shape or target.ndim != 4 or target.shape[0] != 2 or target.shape[-1] != 3:
        raise ConstructiveSolveError("target/predictor must be same-shape two-plane RGB arrays")
    if target.dtype.kind not in "iuf" or predictor.dtype.kind not in "iuf":
        raise ConstructiveSolveError("target/predictor planes must be real numeric arrays")
    anchor = predictor.astype(PROJECTION_DTYPE) - target.astype(PROJECTION_DTYPE)
    anchor, _unused, lo, hi, w = _validated_projection_arrays(anchor, np.zeros_like(anchor), lower, upper, weights)
    q = np.asarray(seg_q, dtype=PROJECTION_DTYPE)
    margins = np.asarray(seg_margins, dtype=PROJECTION_DTYPE)
    if q.shape != anchor.shape[1:] or margins.shape != anchor.shape[1:-1]:
        raise ConstructiveSolveError("seg_q/margins must match one scorer plane")
    if not np.isfinite(q).all() or not np.isfinite(margins).all() or np.any(margins < 0):
        raise ConstructiveSolveError("seg_q/margins must be finite and margins nonnegative")
    if not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ConstructiveSolveError("tolerance must be finite and positive")

    jac: np.ndarray | None = None
    if pose_j_y is not None:
        if tau_pose is None:
            raise ConstructiveSolveError("tau_pose is required when pose_j_y is present")
        jac = np.asarray(pose_j_y, dtype=PROJECTION_DTYPE)
        if jac.shape != (6, *anchor.shape) or not np.isfinite(jac).all():
            raise ConstructiveSolveError("pose_j_y must be finite with shape (6,2,H,W,3)")
        if not math.isfinite(float(tau_pose)) or tau_pose < 0:
            raise ConstructiveSolveError("tau_pose must be finite and nonnegative")
    elif tau_pose is not None:
        raise ConstructiveSolveError("tau_pose requires pose_j_y")

    zero_band = bool(np.array_equal(lo, np.zeros_like(lo)) and np.array_equal(hi, np.zeros_like(hi)))
    if zero_band:
        delta = np.zeros_like(anchor)
        pose_mse = None
        if jac is not None:
            pose_mse = 0.0
        return ProjectionResult(
            delta, 0.5 * float(np.sum(w * (delta - anchor) ** 2)), float(np.min(margins)), pose_mse, 0, True, True
        )

    def project_seg_box(value: np.ndarray) -> np.ndarray:
        projected = np.clip(value, lo, hi)
        projected[1] = project_pixelwise_seg_relaxation(value[1], q, lo[1], hi[1], w[1], margins)
        return projected

    if pose_j_y is None:
        delta = project_seg_box(anchor)
        iterations, converged, pose_mse = 1, True, None
    else:
        assert jac is not None and tau_pose is not None
        if not isinstance(max_dykstra_iterations, int) or max_dykstra_iterations < 1:
            raise ConstructiveSolveError("max_dykstra_iterations must be positive")
        delta, seg_residual, pose_residual = anchor.copy(), np.zeros_like(anchor), np.zeros_like(anchor)
        converged = False
        iteration_count = 0
        for iteration in range(1, max_dykstra_iterations + 1):
            iteration_count = iteration
            prior = delta.copy()
            seg_input = delta + seg_residual
            seg_projected = project_seg_box(seg_input)
            seg_residual = seg_input - seg_projected
            pose_input = seg_projected + pose_residual
            delta = project_rank6_pose_ellipsoid(pose_input, jac, w, tau_pose=float(tau_pose))
            pose_residual = pose_input - delta
            if float(np.max(np.abs(delta - prior), initial=0.0)) <= tolerance:
                seg_check = project_seg_box(delta)
                pose_vector = jac.reshape(6, -1) @ delta.ravel()
                if (
                    float(np.max(np.abs(seg_check - delta), initial=0.0)) <= 5 * tolerance
                    and float(np.mean(pose_vector * pose_vector)) <= float(tau_pose) + 5 * tolerance
                ):
                    converged = True
                    break
        if not converged:
            raise ConstructiveSolveError("Dykstra box/Seg/Pose intersection did not converge")
        iterations = iteration_count
        pose_vector = jac.reshape(6, -1) @ delta.ravel()
        pose_mse = float(np.mean(pose_vector * pose_vector))

    seg_slack = np.sum(q * delta[1], axis=-1) + margins
    if float(np.min(seg_slack)) < -5e-5 or np.any(delta < lo - 5e-5) or np.any(delta > hi + 5e-5):
        raise ConstructiveSolveError("constructive proposal violates its local relaxation")
    objective = 0.5 * float(np.sum(w * (delta - anchor) ** 2))
    return ProjectionResult(delta, objective, float(np.min(seg_slack)), pose_mse, iterations, converged, False)


def realize_factor2_and_require_hard_oracle(
    operator: DisjointResizeOperator,
    scorer_planes: np.ndarray,
    hard_oracle: Callable[[tuple[np.ndarray, np.ndarray]], HardOracleDecision] | None,
) -> LatticeAdmission:
    """Realize two uint8 planes and require external native-fp32 admission."""

    planes = np.asarray(scorer_planes)
    expected_shape = (2, operator.scorer_h, operator.scorer_w, 3)
    if planes.dtype != np.uint8 or planes.shape != expected_shape:
        raise ConstructiveSolveError(f"lattice admission requires exact uint8 scorer geometry {expected_shape}")
    if not callable(hard_oracle):
        raise ConstructiveSolveError("a caller-supplied native-fp32 hard oracle is mandatory")
    # Freeze the requested bytes before invoking caller code.  The callback may
    # close over and mutate its original input array; that must not redefine the
    # realization or the post-oracle verification target.
    requested_planes = _immutable_array_copy(planes)
    frames: list[np.ndarray] = []
    proofs: list[Factor2ExactVerification] = []
    try:
        for plane in requested_planes:
            frame = realize_factor2_uint8_scorer_plane(operator, plane)
            proof = verify_factor2_uint8_scorer_plane(operator, frame, plane)
            if not proof.certified_exact or not proof.numerator_exact:
                raise ConstructiveSolveError("factor-2 integer numerator verification failed")
            frames.append(frame)
            proofs.append(proof)
    except Uint8LatticeError as exc:
        raise ConstructiveSolveError("factor-2 lattice realization refused") from exc
    oracle_frames = (_immutable_array_copy(frames[0]), _immutable_array_copy(frames[1]))
    expected_frame_hashes = (_sha256_array(oracle_frames[0]), _sha256_array(oracle_frames[1]))
    decision = hard_oracle(oracle_frames)  # encode-side hook; never archive decode
    if not isinstance(decision, HardOracleDecision):
        raise ConstructiveSolveError("hard oracle returned an invalid decision object")
    if decision.realized_frame_sha256s != expected_frame_hashes:
        raise ConstructiveSolveError("hard oracle realized-frame SHA-256 custody mismatch")
    if tuple(_sha256_array(frame) for frame in oracle_frames) != expected_frame_hashes:
        raise ConstructiveSolveError("hard oracle mutated its realized-frame inputs")
    if not decision.admitted:
        raise ConstructiveSolveError("native-fp32 hard oracle refused the realized proposal")
    post_oracle_proofs: list[Factor2ExactVerification] = []
    try:
        for frame, plane in zip(frames, requested_planes, strict=True):
            proof = verify_factor2_uint8_scorer_plane(operator, frame, plane)
            if not proof.certified_exact or not proof.numerator_exact:
                raise ConstructiveSolveError("post-oracle factor-2 integer verification failed")
            post_oracle_proofs.append(proof)
    except Uint8LatticeError as exc:
        raise ConstructiveSolveError("post-oracle factor-2 lattice verification refused") from exc
    returned_frames = (_immutable_array_copy(frames[0]), _immutable_array_copy(frames[1]))
    return LatticeAdmission(
        returned_frames,
        (post_oracle_proofs[0], post_oracle_proofs[1]),
        decision,
    )


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(config))).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_resumable_pair_chunk(
    *,
    pair_ids: Sequence[int],
    config: Mapping[str, Any],
    state_path: Path,
    stage_dir: Path,
    derive_input_hash: Callable[[int], str],
    solve_pair: Callable[[int], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Run config-hashed write-once pair stages with drift-refusing resume.

    Completed inputs are re-derived on every invocation.  An orphan stage
    atomically written before the state append is validated and recovered,
    never overwritten.
    """

    raw_ids = list(pair_ids)
    if not raw_ids:
        raise ConstructiveSolveError("pair_ids must be a nonempty unique nonnegative sequence")
    ids = [_exact_pair_id(pair_id, "pair_ids entry") for pair_id in raw_ids]
    if len(set(ids)) != len(ids):
        raise ConstructiveSolveError("pair_ids must be a nonempty unique nonnegative sequence")
    cfg_hash = config_hash(config)
    state_path, stage_dir = Path(state_path), Path(stage_dir)
    initial: dict[str, Any] = {
        "schema": "v10_constructive_solver_resume.v1",
        "config_hash": cfg_hash,
        "pair_ids": ids,
        "completed": [],
    }
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConstructiveSolveError("cannot parse constructive solver resume state") from exc
        state_pair_ids = state.get("pair_ids")
        if (
            not isinstance(state_pair_ids, list)
            or any(type(pair_id) is not int or isinstance(pair_id, bool) for pair_id in state_pair_ids)
            or {key: state.get(key) for key in ("schema", "config_hash", "pair_ids")}
            != {key: initial[key] for key in ("schema", "config_hash", "pair_ids")}
        ):
            raise ConstructiveSolveError("constructive solver resume config/pair drift")
        if set(state) != set(initial) or not isinstance(state.get("completed"), list):
            raise ConstructiveSolveError("constructive solver resume completion list is malformed")
    else:
        state = initial

    completed_by_id: dict[int, Mapping[str, Any]] = {}
    for row in state["completed"]:
        if not isinstance(row, dict) or set(row) != {"pair_id", "input_hash", "stage", "stage_sha256"}:
            raise ConstructiveSolveError("constructive solver resume row is malformed")
        pair_id = _exact_pair_id(row["pair_id"], "resume row pair_id")
        if pair_id in completed_by_id or pair_id not in ids:
            raise ConstructiveSolveError("constructive solver resume coverage is malformed")
        completed_by_id[pair_id] = row

    results: list[Mapping[str, Any]] = []
    stage_dir.mkdir(parents=True, exist_ok=True)

    def load_stage(stage_path: Path, pair_id: int, input_hash: str) -> dict[str, Any]:
        try:
            stage = json.loads(stage_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConstructiveSolveError("cannot parse constructive solver pair stage") from exc
        if (
            not isinstance(stage, dict)
            or set(stage) != {"schema", "pair_id", "config_hash", "input_hash", "result"}
            or stage.get("schema") != "v10_constructive_solver_pair_stage.v1"
            or type(stage.get("pair_id")) is not int
            or isinstance(stage.get("pair_id"), bool)
            or stage.get("pair_id") != pair_id
            or stage.get("config_hash") != cfg_hash
            or stage.get("input_hash") != input_hash
            or not isinstance(stage.get("result"), dict)
        ):
            raise ConstructiveSolveError("pair stage schema/config/input custody drift")
        return stage

    for pair_id in ids:
        input_hash = derive_input_hash(pair_id)
        if (
            not isinstance(input_hash, str)
            or len(input_hash) != 64
            or any(c not in "0123456789abcdef" for c in input_hash)
        ):
            raise ConstructiveSolveError("derive_input_hash must return lowercase SHA-256")
        stage_path = stage_dir / f"pair_{pair_id:04d}.{cfg_hash[:16]}.json"
        row = completed_by_id.get(pair_id)
        if row is not None:
            if row["input_hash"] != input_hash or Path(row["stage"]).resolve() != stage_path.resolve():
                raise ConstructiveSolveError("completed pair input/stage custody drift")
            if not stage_path.is_file() or _sha256_file(stage_path) != row["stage_sha256"]:
                raise ConstructiveSolveError("completed pair stage hash custody failed")
            stage = load_stage(stage_path, pair_id, input_hash)
        elif stage_path.exists():
            stage = load_stage(stage_path, pair_id, input_hash)
            row = {
                "pair_id": pair_id,
                "input_hash": input_hash,
                "stage": str(stage_path.resolve()),
                "stage_sha256": _sha256_file(stage_path),
            }
            state["completed"].append(row)
            _atomic_json(state_path, state)
        else:
            result = dict(solve_pair(pair_id))
            stage = {
                "schema": "v10_constructive_solver_pair_stage.v1",
                "pair_id": pair_id,
                "config_hash": cfg_hash,
                "input_hash": input_hash,
                "result": result,
            }
            _atomic_json(stage_path, stage)
            row = {
                "pair_id": pair_id,
                "input_hash": input_hash,
                "stage": str(stage_path.resolve()),
                "stage_sha256": _sha256_file(stage_path),
            }
            state["completed"].append(row)
            _atomic_json(state_path, state)
        stage_result = stage.get("result")
        if not isinstance(stage_result, dict):
            raise ConstructiveSolveError("pair stage result is not a mapping")
        results.append(stage_result)
    return results


__all__ = [
    "ACTIVE_ARRANGEMENT",
    "CAMERA_HW",
    "EXPECTED_SOURCE_HASHES",
    "HARD_ORACLE_SCHEMA",
    "PAIR_SCHEMA",
    "PROJECTION_DTYPE",
    "RECEIVER_ARITHMETIC",
    "REPRESENTATION",
    "SCORER_HW",
    "ConstructiveSolveError",
    "HardOracleDecision",
    "LatticeAdmission",
    "ProjectionResult",
    "VJPCustodyPair",
    "config_hash",
    "load_vjp_custody_pair",
    "project_pixelwise_seg_relaxation",
    "project_rank6_pose_ellipsoid",
    "project_weighted_box_halfspace",
    "realize_factor2_and_require_hard_oracle",
    "run_resumable_pair_chunk",
    "solve_constructive_projection",
]
