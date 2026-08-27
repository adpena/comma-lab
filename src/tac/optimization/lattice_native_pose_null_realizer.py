# SPDX-License-Identifier: MIT
"""Lattice-native realization of frame-1 pose-null scorer corrections.

This module solves the small block problem that the sq1/q31 lineage left
open: a correction is specified on the 384x512 scorer lattice, but the legal
actuator is the 874x1164 uint8 camera lattice.  The resize operator is the
exact disjoint #580 half-pixel kernel; no uniform 0.25 tap assumption is made.

All routines here are local optimizers/realizers, not score authorities.  They
return diagnostics that state the finite search scope and whether any cap
bound.  A pruned CVP search is deliberately labeled as pruned, not as a global
integer optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from tac.optimization.evaluator_invisibility_basis import (
    CAMERA_H,
    CAMERA_W,
    SCORER_INPUT_H,
    SCORER_INPUT_W,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator

POSE_LUMA_WEIGHTS = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)
LATTICE_NATIVE_REALIZER_SCHEMA = "ddm_dk1_lattice_native_pose_null_realizer.v1"


class LatticeNativeRealizerError(ValueError):
    """Fail-closed malformed geometry, target, or solver configuration."""


@dataclass(frozen=True)
class PrivateBlockGeometry:
    """Exact private camera supports for one aligned 2x2 scorer block."""

    scorer_row: int
    scorer_col: int
    row_indices: np.ndarray
    col_indices: np.ndarray
    row_numerators: np.ndarray
    col_numerators: np.ndarray
    row_denominators: np.ndarray
    col_denominators: np.ndarray
    weights: np.ndarray
    coefficients: np.ndarray
    denominator: int

    def __post_init__(self) -> None:
        for name in (
            "row_indices",
            "col_indices",
            "row_numerators",
            "col_numerators",
            "row_denominators",
            "col_denominators",
            "weights",
            "coefficients",
        ):
            arr = np.asarray(getattr(self, name))
            object.__setattr__(self, name, arr.copy())
        if self.row_indices.shape != (2, 2) or self.col_indices.shape != (2, 2):
            raise LatticeNativeRealizerError("private block supports must be 2x2")
        if self.weights.shape != (2, 2, 2, 2):
            raise LatticeNativeRealizerError("weights must have shape (2,2,2,2)")
        if self.coefficients.shape != (2, 2, 2, 2):
            raise LatticeNativeRealizerError("coefficients must have shape (2,2,2,2)")
        if int(self.denominator) <= 0:
            raise LatticeNativeRealizerError("denominator must be positive")
        if not np.allclose(self.coefficients / int(self.denominator), self.weights):
            raise LatticeNativeRealizerError("weights drifted from exact integer coefficients")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorer_row": int(self.scorer_row),
            "scorer_col": int(self.scorer_col),
            "row_indices": self.row_indices.tolist(),
            "col_indices": self.col_indices.tolist(),
            "row_numerators": self.row_numerators.tolist(),
            "col_numerators": self.col_numerators.tolist(),
            "row_denominators": self.row_denominators.tolist(),
            "col_denominators": self.col_denominators.tolist(),
            "weights": self.weights.tolist(),
            "coefficients": self.coefficients.tolist(),
            "denominator": int(self.denominator),
            "assumes_uniform_025": False,
        }


@dataclass(frozen=True)
class LatticeRealizerResult:
    """One realized integer camera-delta block plus its measured local objective."""

    method: str
    camera_delta: np.ndarray
    scorer_delta: np.ndarray
    pose_leakage: np.ndarray
    pose_leakage_sq: float
    seg_discrepancy: float
    changed_camera_values: int
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        cam = np.asarray(self.camera_delta)
        scorer = np.asarray(self.scorer_delta)
        leakage = np.asarray(self.pose_leakage)
        if cam.shape != (2, 2, 2, 2, 3) or cam.dtype.kind not in ("i", "u"):
            raise LatticeNativeRealizerError(
                "camera_delta must be integer (2,2,2,2,3)"
            )
        if scorer.shape != (2, 2, 3) or not np.all(np.isfinite(scorer)):
            raise LatticeNativeRealizerError("scorer_delta must be finite (2,2,3)")
        if leakage.shape != (6,) or not np.all(np.isfinite(leakage)):
            raise LatticeNativeRealizerError("pose_leakage must be finite length 6")
        object.__setattr__(self, "camera_delta", cam.astype(np.int16, copy=True))
        object.__setattr__(self, "scorer_delta", scorer.astype(np.float64, copy=True))
        object.__setattr__(self, "pose_leakage", leakage.astype(np.float64, copy=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "pose_leakage_sq": float(self.pose_leakage_sq),
            "pose_leakage_l2": float(np.sqrt(self.pose_leakage_sq)),
            "seg_discrepancy": float(self.seg_discrepancy),
            "changed_camera_values": int(self.changed_camera_values),
            "diagnostics": _jsonable(self.diagnostics),
        }


def build_default_operator() -> DisjointResizeOperator:
    """Build the contest resize operator with exact integer tap custody."""

    return DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_INPUT_H,
        scorer_w=SCORER_INPUT_W,
    )


def pose_constraint_matrix() -> np.ndarray:
    """Return sq1's 6x12 frame-1 yuv6-null constraint matrix."""

    a = np.zeros((6, 12), dtype=np.float64)
    for p in range(4):
        a[p, 3 * p : 3 * p + 3] = POSE_LUMA_WEIGHTS
        a[4, 3 * p + 0] = 0.25
        a[5, 3 * p + 2] = 0.25
    return a


def pose_null_projector() -> np.ndarray:
    """Return the Euclidean scorer-lattice projector onto ker(A)."""

    a = pose_constraint_matrix()
    p = np.eye(12, dtype=np.float64) - np.linalg.pinv(a) @ a
    if not np.allclose(p @ p, p, atol=1e-12):
        raise LatticeNativeRealizerError("pose-null projector is not idempotent")
    if float(np.abs(a @ p).max()) > 1e-10:
        raise LatticeNativeRealizerError("pose-null projector does not annihilate A")
    if int(np.linalg.matrix_rank(p)) != 6:
        raise LatticeNativeRealizerError("pose-null projector rank drifted from 6")
    return p


# PROJECT_PARITY_WAIVED: encode-side realizer delta projection; output stored post-projection, inflate reads it as-is
def project_scorer_delta_to_pose_null(delta: np.ndarray) -> np.ndarray:
    """Project one 2x2 RGB scorer-lattice delta to sq1's pose-null subspace."""

    x = _target_delta(delta)
    flat = x.reshape(12)
    projected = flat @ pose_null_projector().T
    return projected.reshape(2, 2, 3)


def private_block_geometry(
    operator: DisjointResizeOperator,
    scorer_row: int,
    scorer_col: int,
) -> PrivateBlockGeometry:
    """Extract exact #580 private supports for an aligned 2x2 scorer block."""

    if not isinstance(operator, DisjointResizeOperator):
        raise LatticeNativeRealizerError("operator must be DisjointResizeOperator")
    scorer_row = _int_in_range(scorer_row, "scorer_row", 0, operator.scorer_h - 2)
    scorer_col = _int_in_range(scorer_col, "scorer_col", 0, operator.scorer_w - 2)
    if scorer_row % 2 or scorer_col % 2:
        raise LatticeNativeRealizerError(
            "pose yuv6 blocks must be aligned to even scorer row/col"
        )

    row_supports = operator.row_supports[scorer_row : scorer_row + 2]
    col_supports = operator.col_supports[scorer_col : scorer_col + 2]
    if any(len(s.indices) != 2 for s in row_supports + col_supports):
        raise LatticeNativeRealizerError("dk1 requires disjoint two-tap supports")

    denominators = {
        int(rs.denominator) * int(cs.denominator)
        for rs in row_supports
        for cs in col_supports
    }
    if len(denominators) != 1:
        raise LatticeNativeRealizerError("block lacks a single common denominator")
    denominator = denominators.pop()

    row_indices = np.asarray([s.indices for s in row_supports], dtype=np.int64)
    col_indices = np.asarray([s.indices for s in col_supports], dtype=np.int64)
    row_nums = np.asarray([s.numerators for s in row_supports], dtype=np.int64)
    col_nums = np.asarray([s.numerators for s in col_supports], dtype=np.int64)
    row_dens = np.asarray([s.denominator for s in row_supports], dtype=np.int64)
    col_dens = np.asarray([s.denominator for s in col_supports], dtype=np.int64)
    coeff = row_nums[:, None, :, None] * col_nums[None, :, None, :]
    weights = coeff.astype(np.float64) / float(denominator)
    return PrivateBlockGeometry(
        scorer_row=scorer_row,
        scorer_col=scorer_col,
        row_indices=row_indices,
        col_indices=col_indices,
        row_numerators=row_nums,
        col_numerators=col_nums,
        row_denominators=row_dens,
        col_denominators=col_dens,
        weights=weights,
        coefficients=coeff,
        denominator=int(denominator),
    )


def extract_private_camera_block(frame: np.ndarray, geometry: PrivateBlockGeometry) -> np.ndarray:
    """Return camera values shaped (2,2,2,2,3) for the private supports."""

    x = np.asarray(frame)
    if x.ndim != 3 or x.shape[2] != 3:
        raise LatticeNativeRealizerError("frame must be HWC RGB")
    out = np.empty((2, 2, 2, 2, 3), dtype=x.dtype)
    for br in range(2):
        rows = geometry.row_indices[br]
        for bc in range(2):
            cols = geometry.col_indices[bc]
            out[br, bc] = x[np.ix_(rows, cols, range(3))]
    return out


def add_private_delta_to_frame(
    frame: np.ndarray,
    geometry: PrivateBlockGeometry,
    camera_delta: np.ndarray,
) -> np.ndarray:
    """Apply one integer private-block delta to a uint8 camera frame."""

    x = np.asarray(frame)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise LatticeNativeRealizerError("frame must be uint8 HWC RGB")
    d = _camera_delta(camera_delta)
    out = x.copy()
    for br in range(2):
        rows = geometry.row_indices[br]
        for bc in range(2):
            cols = geometry.col_indices[bc]
            old = out[np.ix_(rows, cols, range(3))].astype(np.int16)
            out[np.ix_(rows, cols, range(3))] = np.clip(old + d[br, bc], 0, 255).astype(
                np.uint8
            )
    return out


def apply_private_delta(
    camera_delta: np.ndarray,
    geometry: PrivateBlockGeometry,
) -> np.ndarray:
    """Apply exact D weights to a private camera delta block."""

    d = _camera_delta(camera_delta).astype(np.float64)
    return np.sum(d * geometry.weights[..., None], axis=(2, 3))


def minimum_norm_private_preimage(
    scorer_delta: np.ndarray,
    geometry: PrivateBlockGeometry,
) -> np.ndarray:
    """Continuous minimum-norm private-window preimage for a scorer delta."""

    y = _target_delta(scorer_delta)
    denom = np.sum(np.square(geometry.weights), axis=(2, 3))
    return geometry.weights[..., None] * (y[:, :, None, None, :] / denom[..., None, None, None])


def evaluate_camera_delta(
    method: str,
    camera_delta: np.ndarray,
    target_delta: np.ndarray,
    geometry: PrivateBlockGeometry,
    *,
    s_metric: np.ndarray | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> LatticeRealizerResult:
    """Evaluate local pose leakage and scorer-delta discrepancy for one result."""

    target = _target_delta(target_delta)
    cam = _camera_delta(camera_delta)
    scorer = apply_private_delta(cam, geometry)
    leakage = pose_constraint_matrix() @ scorer.reshape(12)
    error = (scorer - target).reshape(12)
    seg = _seg_metric_value(error, s_metric)
    return LatticeRealizerResult(
        method=method,
        camera_delta=cam,
        scorer_delta=scorer,
        pose_leakage=leakage,
        pose_leakage_sq=float(np.dot(leakage, leakage)),
        seg_discrepancy=seg,
        changed_camera_values=int(np.count_nonzero(cam)),
        diagnostics=diagnostics or {},
    )


def uniform_round_baseline(
    target_delta: np.ndarray,
    geometry: PrivateBlockGeometry,
    *,
    base_block: np.ndarray | None = None,
    delta_bounds: tuple[int, int] = (-127, 127),
    s_metric: np.ndarray | None = None,
) -> LatticeRealizerResult:
    """sq1-style baseline: snap scorer deltas, then broadcast to all private taps."""

    target = _target_delta(target_delta)
    lo, hi = _bounds_for(base_block, delta_bounds)
    rounded = np.rint(target).astype(np.int16)
    cam = np.broadcast_to(rounded[:, :, None, None, :], (2, 2, 2, 2, 3)).copy()
    cam = np.clip(cam, lo, hi).astype(np.int16)
    return evaluate_camera_delta(
        "uniform_round",
        cam,
        target,
        geometry,
        s_metric=s_metric,
        diagnostics={"actuator": "broadcast rounded scorer delta to all four private taps"},
    )


def dykstra_integer_realize(
    target_delta: np.ndarray,
    geometry: PrivateBlockGeometry,
    *,
    base_block: np.ndarray | None = None,
    delta_bounds: tuple[int, int] = (-127, 127),
    iterations: int = 8,
    s_metric: np.ndarray | None = None,
) -> LatticeRealizerResult:
    """Alternate between continuous pose-nullity and the bounded integer lattice."""

    if iterations < 1:
        raise LatticeNativeRealizerError("iterations must be >= 1")
    target = _target_delta(target_delta)
    lo, hi = _bounds_for(base_block, delta_bounds)
    b = _resize_matrix(geometry)
    c = pose_constraint_matrix() @ b
    cc_t = c @ c.T
    c_pinv = c.T @ np.linalg.pinv(cc_t)

    # PROJECT_PARITY_WAIVED: nested Dykstra-solve iterate projection (encode side); solution stored post-projection
    def project_pose_null(z_flat: np.ndarray) -> np.ndarray:
        return z_flat - c_pinv @ (c @ z_flat)

    z0 = minimum_norm_private_preimage(target, geometry)
    x = np.clip(z0, lo, hi).reshape(48)
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    best: LatticeRealizerResult | None = None
    seen: set[bytes] = set()
    history: list[dict[str, Any]] = []
    stop_reason = "iteration_cap"

    for iteration in range(iterations):
        y = project_pose_null(x + p)
        p = x + p - y
        z = np.clip(np.rint(y + q), lo.reshape(48), hi.reshape(48)).astype(np.int16)
        q = y + q - z.astype(np.float64)
        cand = z.reshape(2, 2, 2, 2, 3)
        result = evaluate_camera_delta(
            "dykstra",
            cand,
            target,
            geometry,
            s_metric=s_metric,
            diagnostics={},
        )
        key = _objective_key(result)
        history.append(
            {
                "iteration": iteration + 1,
                "pose_leakage_sq": result.pose_leakage_sq,
                "seg_discrepancy": result.seg_discrepancy,
                "changed_camera_values": result.changed_camera_values,
            }
        )
        digest = cand.tobytes()
        if best is None or key < _objective_key(best):
            best = result
        elif digest in seen:
            stop_reason = "cycle_detected"
            break
        seen.add(digest)
        x = z.astype(np.float64)

    if best is None:
        raise LatticeNativeRealizerError("dykstra produced no candidate")
    diagnostics = dict(best.diagnostics)
    diagnostics.update(
        {
            "solver": "dykstra_round_project",
            "iterations_requested": int(iterations),
            "iterations_run": len(history),
            "stop_reason": stop_reason,
            "cap_stop": stop_reason == "iteration_cap",
            "history": history,
            "continuous_constraint": "ker(A @ D_private)",
            "integer_projection": "round_then_clip_to_uint8_delta_bounds",
        }
    )
    return evaluate_camera_delta(
        "dykstra",
        best.camera_delta,
        target,
        geometry,
        s_metric=s_metric,
        diagnostics=diagnostics,
    )


def cvp_integer_realize(
    target_delta: np.ndarray,
    geometry: PrivateBlockGeometry,
    *,
    base_block: np.ndarray | None = None,
    delta_bounds: tuple[int, int] = (-127, 127),
    tap_radius: int = 1,
    max_channel_candidates: int = 9,
    max_pixel_candidates: int = 16,
    max_combinations: int = 250_000,
    s_metric: np.ndarray | None = None,
) -> LatticeRealizerResult:
    """Bounded private-window CVP/Babai search around the continuous preimage.

    The returned exactness is for the declared finite search scope.  If channel
    or pixel candidate lists are pruned, diagnostics mark the result as
    ``exact_declared_scope=false`` and describe the Babai-pruned scope.
    """

    target = _target_delta(target_delta)
    tap_radius = _int_min(tap_radius, "tap_radius", 0)
    max_channel_candidates = _int_min(max_channel_candidates, "max_channel_candidates", 1)
    max_pixel_candidates = _int_min(max_pixel_candidates, "max_pixel_candidates", 1)
    max_combinations = _int_min(max_combinations, "max_combinations", 1)
    lo, hi = _bounds_for(base_block, delta_bounds)
    center = minimum_norm_private_preimage(target, geometry)
    weights = geometry.weights.reshape(2, 2, 4)

    all_pixel_candidates: list[list[dict[str, Any]]] = []
    pixel_counts: list[dict[str, Any]] = []
    pruned = False
    for pr in range(2):
        for pc in range(2):
            channel_candidates = []
            channel_counts = []
            for ch in range(3):
                cands, count = _channel_candidates(
                    center[pr, pc, :, :, ch],
                    lo[pr, pc, :, :, ch],
                    hi[pr, pc, :, :, ch],
                    weights[pr, pc],
                    float(target[pr, pc, ch]),
                    tap_radius=tap_radius,
                    max_candidates=max_channel_candidates,
                )
                channel_candidates.append(cands)
                channel_counts.append(count)
                pruned = pruned or count["pruned"]

            pix: list[dict[str, Any]] = []
            for r_cand, g_cand, b_cand in product(*channel_candidates):
                rgb = np.asarray([r_cand["value"], g_cand["value"], b_cand["value"]], dtype=np.float64)
                cam = np.stack(
                    [r_cand["delta"], g_cand["delta"], b_cand["delta"]],
                    axis=-1,
                )
                err = rgb - target[pr, pc]
                pix.append(
                    {
                        "rgb": rgb,
                        "camera_delta": cam.astype(np.int16),
                        "seg_error": float(np.dot(err, err)),
                        "luma_abs": float(abs(np.dot(POSE_LUMA_WEIGHTS, rgb))),
                        "norm": int(np.sum(np.abs(cam))),
                    }
                )
            total_pixel = len(pix)
            pix.sort(key=lambda item: (item["luma_abs"], item["seg_error"], item["norm"]))
            if len(pix) > max_pixel_candidates:
                pruned = True
                pix = pix[:max_pixel_candidates]
            all_pixel_candidates.append(pix)
            pixel_counts.append(
                {
                    "pixel": [pr, pc],
                    "channel_counts": channel_counts,
                    "pixel_candidates_total_before_prune": total_pixel,
                    "pixel_candidates_kept": len(pix),
                    "pixel_pruned": total_pixel > len(pix),
                }
            )

    combo_total = int(np.prod([len(p) for p in all_pixel_candidates], dtype=np.int64))
    if combo_total > max_combinations:
        pruned = True
        keep_each = max(1, int(np.floor(max_combinations ** 0.25)))
        all_pixel_candidates = [p[:keep_each] for p in all_pixel_candidates]
        combo_total = int(np.prod([len(p) for p in all_pixel_candidates], dtype=np.int64))

    best_cam: np.ndarray | None = None
    best_key: tuple[float, float, int] | None = None
    evaluated = 0
    for p00, p01, p10, p11 in product(*all_pixel_candidates):
        evaluated += 1
        pixels = (p00, p01, p10, p11)
        scorer = np.stack([p["rgb"] for p in pixels], axis=0).reshape(2, 2, 3)
        leakage = pose_constraint_matrix() @ scorer.reshape(12)
        err = (scorer - target).reshape(12)
        seg = _seg_metric_value(err, s_metric)
        cam = np.stack([p["camera_delta"] for p in pixels], axis=0).reshape(2, 2, 2, 2, 3)
        key = (float(np.dot(leakage, leakage)), seg, int(np.sum(np.abs(cam))))
        if best_key is None or key < best_key:
            best_key = key
            best_cam = cam.astype(np.int16)

    if best_cam is None:
        raise LatticeNativeRealizerError("cvp search produced no candidate")

    diagnostics = {
        "solver": "bounded_private_window_cvp",
        "tap_radius": int(tap_radius),
        "max_channel_candidates": int(max_channel_candidates),
        "max_pixel_candidates": int(max_pixel_candidates),
        "max_combinations": int(max_combinations),
        "combinations_evaluated": int(evaluated),
        "candidate_scope": (
            "EXHAUSTIVE_TAP_RADIUS_PRODUCT"
            if not pruned
            else "BABAI_PRUNED_TOP_K_WITH_EXACT_ENUMERATION_OF_KEPT_SET"
        ),
        "exact_declared_scope": bool(not pruned),
        "global_integer_optimum_claim": False,
        "pixel_candidate_counts": pixel_counts,
    }
    return evaluate_camera_delta(
        "cvp",
        best_cam,
        target,
        geometry,
        s_metric=s_metric,
        diagnostics=diagnostics,
    )


def realize_lattice_native_block(
    target_delta: np.ndarray,
    geometry: PrivateBlockGeometry,
    *,
    base_block: np.ndarray | None = None,
    delta_bounds: tuple[int, int] = (-127, 127),
    dykstra_iterations: int = 8,
    cvp_tap_radius: int = 1,
    s_metric: np.ndarray | None = None,
) -> dict[str, LatticeRealizerResult]:
    """Race the required dk1 arms on one block."""

    target = project_scorer_delta_to_pose_null(target_delta)
    return {
        "naive": uniform_round_baseline(
            target,
            geometry,
            base_block=base_block,
            delta_bounds=delta_bounds,
            s_metric=s_metric,
        ),
        "dykstra": dykstra_integer_realize(
            target,
            geometry,
            base_block=base_block,
            delta_bounds=delta_bounds,
            iterations=dykstra_iterations,
            s_metric=s_metric,
        ),
        "cvp": cvp_integer_realize(
            target,
            geometry,
            base_block=base_block,
            delta_bounds=delta_bounds,
            tap_radius=cvp_tap_radius,
            s_metric=s_metric,
        ),
    }


def results_to_receipt(results: dict[str, LatticeRealizerResult]) -> dict[str, Any]:
    """Summarize a solver race without serializing bulky camera deltas."""

    if not results:
        raise LatticeNativeRealizerError("no results to summarize")
    return {
        "schema": LATTICE_NATIVE_REALIZER_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "methods": {name: result.to_dict() for name, result in results.items()},
        "best_by_pose_then_seg": min(
            results,
            key=lambda name: _objective_key(results[name]),
        ),
    }


def _resize_matrix(geometry: PrivateBlockGeometry) -> np.ndarray:
    b = np.zeros((12, 48), dtype=np.float64)
    for pr in range(2):
        for pc in range(2):
            pixel = pr * 2 + pc
            for tr in range(2):
                for tc in range(2):
                    tap = tr * 2 + tc
                    w = geometry.weights[pr, pc, tr, tc]
                    for ch in range(3):
                        b[pixel * 3 + ch, ((pixel * 4 + tap) * 3) + ch] = w
    return b


def _channel_candidates(
    center: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    weights: np.ndarray,
    target: float,
    *,
    tap_radius: int,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lo_i = lo.reshape(4).astype(np.int64)
    hi_i = hi.reshape(4).astype(np.int64)
    center_raw_i = np.rint(center.reshape(4)).astype(np.int64)
    center_i = np.clip(center_raw_i, lo_i, hi_i)
    clipped_center_count = int(np.count_nonzero(center_i != center_raw_i))
    weights4 = weights.reshape(4)
    choices: list[list[int]] = []
    for idx in range(4):
        low = max(int(lo_i[idx]), int(center_i[idx]) - tap_radius)
        high = min(int(hi_i[idx]), int(center_i[idx]) + tap_radius)
        if low > high:
            raise LatticeNativeRealizerError("empty tap search interval")
        choices.append(list(range(low, high + 1)))
    by_values: dict[tuple[int, ...], dict[str, Any]] = {}
    for vals in product(*choices):
        arr = np.asarray(vals, dtype=np.int16)
        value = float(np.dot(weights4, arr.astype(np.float64)))
        key = tuple(int(v) for v in arr)
        by_values[key] = {
            "delta": arr.reshape(2, 2),
            "value": value,
            "seg_error": float((value - target) ** 2),
            "norm": int(np.sum(np.abs(arr))),
        }
    cands = list(by_values.values())
    cands.sort(key=lambda item: (item["seg_error"], item["norm"]))
    pruned = len(cands) > max_candidates
    if pruned:
        cands = cands[:max_candidates]
    return cands, {
        "tap_product_candidates": int(np.prod([len(c) for c in choices], dtype=np.int64)),
        "unique_candidates": len(by_values),
        "kept": len(cands),
        "pruned": bool(pruned),
        "center_clipped_to_bounds_count": clipped_center_count,
    }


def _bounds_for(
    base_block: np.ndarray | None,
    delta_bounds: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if base_block is None:
        lo, hi = delta_bounds
        if int(lo) > int(hi):
            raise LatticeNativeRealizerError("delta_bounds low exceeds high")
        return (
            np.full((2, 2, 2, 2, 3), int(lo), dtype=np.int16),
            np.full((2, 2, 2, 2, 3), int(hi), dtype=np.int16),
        )
    base = np.asarray(base_block)
    if base.shape != (2, 2, 2, 2, 3) or base.dtype != np.uint8:
        raise LatticeNativeRealizerError("base_block must be uint8 (2,2,2,2,3)")
    return -base.astype(np.int16), (255 - base.astype(np.int16))


def _camera_delta(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (2, 2, 2, 2, 3) or arr.dtype.kind not in ("i", "u"):
        raise LatticeNativeRealizerError("camera delta must be integer (2,2,2,2,3)")
    if np.any(arr < -255) or np.any(arr > 255):
        raise LatticeNativeRealizerError("camera delta exceeds uint8 delta envelope")
    return arr.astype(np.int16, copy=False)


def _target_delta(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (2, 2, 3) or arr.dtype.kind not in ("i", "u", "f"):
        raise LatticeNativeRealizerError("target delta must be real (2,2,3)")
    out = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(out)):
        raise LatticeNativeRealizerError("target delta contains non-finite values")
    return out


def _seg_metric_value(error: np.ndarray, s_metric: np.ndarray | None) -> float:
    e = np.asarray(error, dtype=np.float64).reshape(12)
    if s_metric is None:
        return float(np.dot(e, e))
    m = np.asarray(s_metric, dtype=np.float64)
    if m.shape == (12,):
        return float(np.dot(m, e * e))
    if m.shape != (12, 12):
        raise LatticeNativeRealizerError("s_metric must be None, length 12, or 12x12")
    return float(e @ m @ e)


def _objective_key(result: LatticeRealizerResult) -> tuple[float, float, int]:
    return (
        float(result.pose_leakage_sq),
        float(result.seg_discrepancy),
        int(result.changed_camera_values),
    )


def _int_min(value: int, name: str, minimum: int) -> int:
    if not isinstance(value, int) or value < minimum:
        raise LatticeNativeRealizerError(f"{name} must be integer >= {minimum}")
    return int(value)


def _int_in_range(value: int, name: str, low: int, high: int) -> int:
    if not isinstance(value, int) or value < low or value > high:
        raise LatticeNativeRealizerError(f"{name} must be integer in [{low}, {high}]")
    return int(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
