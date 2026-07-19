# SPDX-License-Identifier: MIT
"""Score-native utilities for the joint uint8 seg/pose inverse solve.

This module is deliberately scorer-agnostic.  A measurement runner supplies the
frozen SegNet/PoseNet hard-oracle callbacks; this layer owns the interval
lattice projection, rate telemetry, and KKT calculation.  The separation keeps
scorer weights out of decode-time code and makes the predictor contract
auditable.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    BlockSolveStatus,
    DisjointResizeOperator,
    solve_bounded_integer_block,
)


class JointSolveError(ValueError):
    """A fail-closed joint-solve or custody violation."""


@dataclass(frozen=True)
class MarginBandConfig:
    """Convert cached SegNet logit margins into scorer-RGB trust bands.

    ``local_lipschitz`` is an explicitly measured/configured logit-per-RGB-unit
    bound.  The band remains a proposal aid: only a frozen hard-oracle verdict
    can authorize a candidate.
    """

    scale: float
    local_lipschitz: float
    max_rgb_radius: float


@dataclass(frozen=True)
class HyperplaneBand:
    """Conservative RGB-axis inner box of a winner/rival pullback slab."""

    channel_radii: np.ndarray
    winner: np.ndarray
    rival: np.ndarray
    feature_flip_distance: np.ndarray
    equation_id: str = "segnet_head_rank4_linear_flipdist_v1"


@dataclass(frozen=True)
class IntervalSolveTelemetry:
    total_blocks: int
    exact_blocks: int
    repaired_blocks: int
    budget_blocks: int
    target_numerator_l1_shift: int
    maximum_projection_error: float
    binding_counts: Mapping[str, int]


@dataclass(frozen=True)
class IntervalFrameResult:
    frame: np.ndarray
    chosen_numerators: np.ndarray
    band_radius_numerators: np.ndarray
    binding_map: np.ndarray
    telemetry: IntervalSolveTelemetry


def derive_margin_rgb_band(margins: np.ndarray, config: MarginBandConfig) -> np.ndarray:
    """Legacy isotropic helper, valid only for a zero-radius control.

    A positive isotropic band cannot represent the rank-4 winner/rival
    arrangement and is rejected.  Use :func:`derive_hyperplane_channel_band`.
    """

    margin = np.asarray(margins, dtype=np.float64)
    values = (config.scale, config.local_lipschitz, config.max_rgb_radius)
    if not all(math.isfinite(float(v)) for v in values):
        raise JointSolveError("margin-band configuration must be finite")
    if config.scale < 0 or config.local_lipschitz <= 0 or config.max_rgb_radius < 0:
        raise JointSolveError("margin band requires scale>=0, Lip_local>0, max_radius>=0")
    if margin.ndim != 2 or not np.isfinite(margin).all() or np.any(margin < 0):
        raise JointSolveError("cached margins must be a finite nonnegative 2-D field")
    if config.scale != 0.0:
        raise JointSolveError(
            "positive isotropic SegNet bands are forbidden; provide winner/rival "
            "unit-head-normal pullbacks to derive_hyperplane_channel_band"
        )
    return np.zeros((*margin.shape, 3), dtype=np.float64)


def derive_hyperplane_channel_band(
    margins: np.ndarray,
    winner: np.ndarray,
    rival: np.ndarray,
    unit_head_normal_pullback_rgb: np.ndarray,
    pair_norms: np.ndarray,
    config: MarginBandConfig,
) -> HyperplaneBand:
    """Derive an anisotropic inner box of each active hyperplane slab.

    The exact head-space distance is ``margin / ||Delta-w||``.  The supplied
    real-scorer VJP is the RGB pullback of the *unit* winner/rival head normal.
    Channel radius ``d/(3*|q_c|)`` guarantees the axis box lies inside
    ``|<q,delta_rgb>| <= d`` by the triangle inequality.  A zero pullback
    component is unconstrained and clipped by ``max_rgb_radius``.
    """

    margin = np.asarray(margins, dtype=np.float64)
    win, riv = np.asarray(winner), np.asarray(rival)
    pullback = np.asarray(unit_head_normal_pullback_rgb, dtype=np.float64)
    norms = np.asarray(pair_norms, dtype=np.float64)
    if margin.ndim != 2 or win.shape != margin.shape or riv.shape != margin.shape or norms.shape != margin.shape:
        raise JointSolveError("margin/winner/rival/pair-norm geometry mismatch")
    if pullback.shape != (*margin.shape, 3):
        raise JointSolveError("unit-head-normal RGB pullback must have shape HxWx3")
    if not np.isfinite(margin).all() or not np.isfinite(pullback).all() or not np.isfinite(norms).all():
        raise JointSolveError("hyperplane-band inputs must be finite")
    if np.any(margin < 0) or np.any(norms <= 0) or np.any(win == riv):
        raise JointSolveError("invalid active winner/rival hyperplane custody")
    if config.scale < 0 or config.max_rgb_radius < 0 or config.local_lipschitz <= 0:
        raise JointSolveError("invalid hyperplane band configuration")
    distance = config.scale * margin / norms
    denominator = 3.0 * np.abs(pullback) * config.local_lipschitz
    with np.errstate(divide="ignore", invalid="ignore"):
        radii = np.where(denominator > 0, distance[:, :, None] / denominator, config.max_rgb_radius)
    radii = np.minimum(radii, config.max_rgb_radius)
    return HyperplaneBand(radii, win.astype(np.int8), riv.astype(np.int8), distance)


def generated_fill_predictor(operator: DisjointResizeOperator, target: np.ndarray) -> np.ndarray:
    """Decoder-free piecewise-constant fill derived only from the stored target.

    This is a valid rule-118 predictor only when ``target`` (or its compact
    description) is counted in the payload.  No camera-space source is accepted.
    """

    y = np.asarray(target, dtype=np.float64)
    squeeze = y.ndim == 2
    if squeeze:
        y = y[:, :, None]
    if y.shape[:2] != (operator.scorer_h, operator.scorer_w) or y.ndim != 3:
        raise JointSolveError("target geometry does not match resize operator")
    if not np.isfinite(y).all():
        raise JointSolveError("target contains non-finite values")
    out = np.zeros((operator.camera_h, operator.camera_w, y.shape[2]), dtype=np.uint8)
    for oi, rs in enumerate(operator.row_supports):
        for oj, cs in enumerate(operator.col_supports):
            value = np.clip(np.rint(y[oi, oj]), 0, 255).astype(np.uint8)
            out[np.ix_(rs.indices, cs.indices, range(y.shape[2]))] = value
    return out[:, :, 0] if squeeze else out


def _nearest_reachable(value: int, lo: int, hi: int, modulus: int) -> int:
    if lo > hi or modulus <= 0:
        raise JointSolveError("invalid reachable interval")
    clipped = min(max(value, lo), hi)
    down = clipped - (clipped % modulus)
    up = down + modulus
    choices = [x for x in (down, up) if lo <= x <= hi]
    if not choices:
        first = lo + ((-lo) % modulus)
        if first > hi:
            raise JointSolveError("interval contains no reachable numerator")
        return first
    return min(choices, key=lambda x: (abs(x - value), x))


def solve_interval_frame(
    operator: DisjointResizeOperator,
    source_numerators: np.ndarray,
    common_denominator: int,
    rgb_band: np.ndarray,
    *,
    predictor: np.ndarray,
    max_nodes_per_block: int = 4096,
) -> IntervalFrameResult:
    """Pick the reachable in-band scorer point nearest a declared predictor.

    Every selected numerator is solved exactly on the uint8 lattice.  The
    returned binding map uses 0=slack, 1=lower band, 2=upper band.  A budgeted
    or infeasible block fails closed; it is never silently accepted.
    """

    src = np.asarray(source_numerators)
    if src.ndim == 2:
        src = src[:, :, None]
    if src.shape[:2] != (operator.scorer_h, operator.scorer_w) or not np.issubdtype(src.dtype, np.integer):
        raise JointSolveError("source_numerators must be an integer scorer plane")
    pred = np.asarray(predictor)
    if pred.shape != (operator.camera_h, operator.camera_w, src.shape[2]) or pred.dtype != np.uint8:
        raise JointSolveError("predictor must be camera-geometry uint8 with matching channels")
    band = np.asarray(rgb_band, dtype=np.float64)
    if band.shape == (operator.scorer_h, operator.scorer_w):
        band = np.repeat(band[:, :, None], src.shape[2], axis=2)
    if band.shape != (operator.scorer_h, operator.scorer_w, src.shape[2]) or not np.isfinite(band).all() or np.any(band < 0):
        raise JointSolveError("rgb_band must be a finite nonnegative scorer-plane channel field")
    if int(common_denominator) <= 0 or int(max_nodes_per_block) <= 0:
        raise JointSolveError("denominator and node budget must be positive")

    pred_num, pred_den = operator.apply_numerators(pred)
    if int(pred_den) != int(common_denominator):
        raise JointSolveError("predictor/source rational denominators differ")
    if pred_num.ndim == 2:
        pred_num = pred_num[:, :, None]
    radius = np.floor(band * int(common_denominator) + 1e-12).astype(np.int64)
    chosen = np.empty_like(src, dtype=np.int64)
    binding = np.zeros(src.shape, dtype=np.uint8)
    out = pred.copy()
    exact = repaired = budget = numerator_shift = 0

    for oi, rs in enumerate(operator.row_supports):
        for oj, cs in enumerate(operator.col_supports):
            coefficients = tuple(int(x) for x in np.outer(rs.numerators, cs.numerators).reshape(-1))
            modulus = math.gcd(*coefficients)
            max_sum = 255 * sum(coefficients)
            for ch in range(src.shape[2]):
                center = int(src[oi, oj, ch])
                rad = int(radius[oi, oj, ch])
                lo, hi = max(0, center - rad), min(max_sum, center + rad)
                target_integer = _nearest_reachable(int(pred_num[oi, oj, ch]), lo, hi, modulus)
                chosen[oi, oj, ch] = target_integer
                if target_integer == lo and lo != hi:
                    binding[oi, oj, ch] = 1
                elif target_integer == hi and lo != hi:
                    binding[oi, oj, ch] = 2
                numerator_shift += abs(target_integer - center)
                idx = np.ix_(rs.indices, cs.indices, (ch,))
                preferred = pred[idx].reshape(-1).astype(np.float64)
                solved = solve_bounded_integer_block(
                    coefficients,
                    int(common_denominator),
                    target_integer / int(common_denominator),
                    target_integer=target_integer,
                    preferred=preferred,
                    max_nodes=int(max_nodes_per_block),
                )
                if solved.status is BlockSolveStatus.NOT_FOUND_BUDGET:
                    budget += 1
                    raise JointSolveError(f"integer repair exhausted node budget at block {(oi, oj, ch)}")
                if solved.status is not BlockSolveStatus.FEASIBLE_EXACT:
                    raise JointSolveError(f"interval block is not certified exact: {solved.status}")
                values = np.asarray(solved.values, dtype=np.uint8)
                out[idx] = values.reshape(len(rs.indices), len(cs.indices), 1)
                exact += 1
                repaired += int(not np.array_equal(values, pred[idx].reshape(-1)))

    realized_num, realized_den = operator.apply_numerators(out)
    if realized_num.ndim == 2:
        realized_num = realized_num[:, :, None]
    error = float(np.max(np.abs(realized_num.astype(np.int64) - chosen), initial=0)) / int(realized_den)
    if error != 0.0:
        raise JointSolveError(f"integer projection custody failed: max error {error}")
    counts = {
        "slack": int(np.count_nonzero(binding == 0)),
        "lower": int(np.count_nonzero(binding == 1)),
        "upper": int(np.count_nonzero(binding == 2)),
    }
    return IntervalFrameResult(
        frame=out,
        chosen_numerators=chosen,
        band_radius_numerators=radius,
        binding_map=binding,
        telemetry=IntervalSolveTelemetry(
            total_blocks=int(src.size), exact_blocks=exact, repaired_blocks=repaired,
            budget_blocks=budget, target_numerator_l1_shift=numerator_shift,
            maximum_projection_error=error, binding_counts=counts,
        ),
    )


def residual_bytes_and_tiles(frame: np.ndarray, predictor: np.ndarray, *, tile_hw: tuple[int, int] = (64, 64)) -> dict[str, Any]:
    """Measure actual Brotli and Zstandard bytes plus additive tile attribution."""

    import brotli
    def zstd19(payload: bytes) -> bytes:
        try:
            import zstandard as zstd
        except ImportError:
            try:
                proc = subprocess.run(
                    ["zstd", "-19", "--stdout", "--quiet"], input=payload,
                    capture_output=True, check=False,
                )
            except FileNotFoundError as exc:  # pragma: no cover - environment gate
                raise JointSolveError("zstandard module or zstd CLI is required for rate custody") from exc
            if proc.returncode:
                raise JointSolveError(f"zstd CLI failed with rc={proc.returncode}") from None
            return proc.stdout
        return zstd.ZstdCompressor(level=19).compress(payload)
    x, p = np.asarray(frame), np.asarray(predictor)
    if x.shape != p.shape or x.dtype != np.uint8 or p.dtype != np.uint8 or x.ndim != 3:
        raise JointSolveError("frame/predictor must be same-shape uint8 HWC")
    residual = (x.astype(np.int16) - p.astype(np.int16)).astype("<i2", copy=False)
    raw = residual.tobytes(order="C")
    th, tw = tile_hw
    if th <= 0 or tw <= 0:
        raise JointSolveError("tile geometry must be positive")
    rows = []
    for y0 in range(0, x.shape[0], th):
        for x0 in range(0, x.shape[1], tw):
            tile = residual[y0:y0 + th, x0:x0 + tw]
            payload = tile.tobytes(order="C")
            rows.append({
                "y": y0, "x": x0, "h": int(tile.shape[0]), "w": int(tile.shape[1]),
                "brotli_q11_bytes": len(brotli.compress(payload, quality=11)),
                "zstd_19_bytes": len(zstd19(payload)),
                "nonzero_values": int(np.count_nonzero(tile)),
            })
    return {
        "residual_encoding": "signed little-endian int16 HWC",
        "raw_bytes": len(raw),
        "brotli_q11_bytes": len(brotli.compress(raw, quality=11)),
        "zstd_19_bytes": len(zstd19(raw)),
        "residual_sha256": hashlib.sha256(raw).hexdigest(),
        "tile_attribution_convention": "each tile compressed independently; additive but includes per-tile codec overhead",
        "tiles": rows,
    }


def range_payload_bytes_and_tiles(
    chosen_numerators: np.ndarray,
    predictor_numerators: np.ndarray,
    *,
    tile_hw: tuple[int, int] = (32, 32),
) -> dict[str, Any]:
    """Actual codec bytes for the counted range(A) numerator description.

    Camera-space ``ker(A)`` is never serialized or attributed.  The decoder
    regenerates it from the declared predictor and deterministic lattice solve.
    """

    chosen, pred = np.asarray(chosen_numerators), np.asarray(predictor_numerators)
    if chosen.shape != pred.shape or chosen.ndim != 3 or not np.issubdtype(chosen.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise JointSolveError("chosen/predictor numerators must be same-shape integer HWC")
    residual64 = chosen.astype(np.int64) - pred.astype(np.int64)
    if np.any(residual64 < np.iinfo(np.int32).min) or np.any(residual64 > np.iinfo(np.int32).max):
        raise JointSolveError("range numerator residual exceeds signed int32 payload contract")
    words = residual64.astype("<i4", copy=False)
    import brotli

    def zstd19(payload: bytes) -> bytes:
        try:
            import zstandard as zstd
        except ImportError:
            proc = subprocess.run(
                ["zstd", "-19", "--stdout", "--quiet"], input=payload,
                capture_output=True, check=False,
            )
            if proc.returncode:
                raise JointSolveError(f"zstd CLI failed with rc={proc.returncode}") from None
            return proc.stdout
        return zstd.ZstdCompressor(level=19).compress(payload)

    raw = words.tobytes(order="C")
    th, tw = tile_hw
    rows = []
    for y0 in range(0, words.shape[0], th):
        for x0 in range(0, words.shape[1], tw):
            tile = words[y0:y0 + th, x0:x0 + tw]
            payload = tile.tobytes(order="C")
            rows.append({
                "y": y0, "x": x0, "h": int(tile.shape[0]), "w": int(tile.shape[1]),
                "brotli_q11_bytes": len(brotli.compress(payload, quality=11)),
                "zstd_19_bytes": len(zstd19(payload)),
                "nonzero_values": int(np.count_nonzero(tile)),
            })
    return {
        "residual_encoding": "signed little-endian int32 scorer-numerator HWC",
        "raw_bytes": len(raw), "brotli_q11_bytes": len(brotli.compress(raw, quality=11)),
        "zstd_19_bytes": len(zstd19(raw)), "residual_sha256": hashlib.sha256(raw).hexdigest(),
        "range_A_only": True, "ker_A_payload_bytes": 0,
        "tile_attribution_convention": "each scorer-plane tile compressed independently; additive with per-tile codec overhead",
        "tiles": rows,
    }


def pose_score_derivative(d_pose: float) -> float:
    """Return d(sqrt(10*d_pose))/d(d_pose) = 5/sqrt(10*d_pose)."""

    if not math.isfinite(d_pose) or d_pose <= 0:
        return math.inf
    return 5.0 / math.sqrt(10.0 * d_pose)


def solve_measured_waterfill(seg_curve: Sequence[Mapping[str, float]], pose_curve: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Choose the measured adjacent segments with closest score/byte marginals.

    Curves contain ``bytes`` and ``distortion``.  No interpolation beyond the
    measured adjacent secants is promoted; flat/non-monotone data returns an
    explicit instance-scoped inconclusive verdict.
    """

    def segments(curve: Sequence[Mapping[str, float]], family: str) -> list[dict[str, float]]:
        pts = sorted(({"bytes": float(p["bytes"]), "distortion": float(p["distortion"])} for p in curve), key=lambda p: p["bytes"])
        out = []
        for a, b in itertools.pairwise(pts):
            db = b["bytes"] - a["bytes"]
            gain = a["distortion"] - b["distortion"]
            if db <= 0 or gain <= 0:
                continue
            dmid = 0.5 * (a["distortion"] + b["distortion"])
            weight = 100.0 if family == "seg" else pose_score_derivative(dmid)
            out.append({"lo_bytes": a["bytes"], "hi_bytes": b["bytes"], "marginal_score_per_byte": weight * gain / db})
        return out

    ss, ps = segments(seg_curve, "seg"), segments(pose_curve, "pose")
    if not ss or not ps:
        return {"status": "INCONCLUSIVE_FLAT_OR_NOISY", "verdict_scope": "measured curve instance only", "seg_segments": ss, "pose_segments": ps}
    pair = min(((abs(s["marginal_score_per_byte"] - p["marginal_score_per_byte"]), s, p) for s in ss for p in ps), key=lambda x: x[0])
    return {
        "status": "MEASURED_SECANT_KKT_CANDIDATE",
        "verdict_scope": "measured adjacent curve segments only; no continuous optimum claim",
        "marginal_gap": pair[0], "seg_segment": pair[1], "pose_segment": pair[2],
        "derived_pose_seg_crossover_d_pose": 2.5e-4,
    }


__all__ = [
    "HyperplaneBand", "IntervalFrameResult", "JointSolveError", "MarginBandConfig",
    "derive_hyperplane_channel_band", "derive_margin_rgb_band", "generated_fill_predictor", "pose_score_derivative",
    "range_payload_bytes_and_tiles", "residual_bytes_and_tiles", "solve_interval_frame", "solve_measured_waterfill",
]
