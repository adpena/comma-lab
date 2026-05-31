# SPDX-License-Identifier: MIT
"""Per-subband rate-distortion water-fill solver for the Z8 wavelet payload.

#1591 / #1592 — the canonical "open knob" the detail-coeff entropy-headroom
report (``z8_detail_entropy_headroom_*.json``) identified as the only remaining
Z8 rate lever: the per-subband quantization step ``Δ``. The wavelet detail blob
is ~99.5% of the archive and 1000-4000× above its Shannon floor at the current
operating point; the report measured, per Mallat detail subband, a discrete
rate-distortion curve over a sweep of ``Δ`` values. This module consumes those
curves and solves the canonical discrete-RD bit-allocation:

    minimize  D_total(Δ)  subject to  R_total(Δ) <= byte_budget

via the Lagrangian ``J = D_total + λ · R_total`` (Shoham & Gersho 1988). For a
fixed ``λ`` each subband independently picks the operating point minimizing its
local ``D_i·w_i + λ·R_i``; bisecting ``λ`` drives ``R_total`` to the budget. The
discrete RD points are first reduced to their lower-convex-hull (Pareto
frontier) because a Lagrangian can only ever select hull points — this keeps the
solver correct even though the raw RD points are non-monotone in ``Δ`` (the live
codec switches method at coarse ``Δ``).

The emitted artifact is the ``entropy_detail_quantization_steps`` map consumed
directly by :class:`Z8JointCoefficientWaterfillConfig` — the existing executable
actuator. This module is therefore a pure *extend*: it produces the actuator's
existing per-(frame, level, subband) ``Δ`` input rather than re-implementing any
coefficient mutation or archive packing.

All outputs are ``[macOS-CPU advisory]`` planner artifacts — non-promotable
until the chosen ``Δ`` map is materialized through the actuator and the
byte-closed archive is signed by exact contest CPU/CUDA eval. The solver
proposes the operating point; full-video replay ratifies.

Distortion model: for an orthonormal DWT, reconstruction MSE is the sum of
per-subband squared-error contributions (Parseval). We use the
coefficient-count-weighted mean of per-subband MSE,
``Σ_i mse_i·n_i / Σ_i n_i``, as the global distortion proxy so per-subband
distortions compose into one scalar objective.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PER_SUBBAND_RD_WATERFILL_SCHEMA = "z8_per_subband_rd_waterfill_solution.v1"
PER_SUBBAND_RD_WATERFILL_ROLE = (
    "solve_lagrangian_rate_distortion_water_fill_over_per_subband_quant_step_"
    "curves_then_emit_actuator_entropy_detail_quantization_steps_map"
)

# Advisory / non-promotable markers per CLAUDE.md "MPS auth eval is NOISE" +
# Catalog #192 + #341. This is a planner; it emits a config, never a score.
ADVISORY_MARKERS: dict[str, Any] = {
    "axis_tag": "[macOS-CPU advisory]",
    "evidence_grade": "macOS-CPU-advisory",
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}

_VALID_ORIENTATIONS = ("lh", "hl", "hh")


@dataclass(frozen=True)
class SubbandRDPoint:
    """One discrete operating point on a subband's rate-distortion curve.

    ``quant_step`` is ``None`` for the keep-raw baseline (no quantization): the
    coefficients are stored at their current ``current_raw_f32_brotli`` cost with
    zero added distortion. A ``float`` step is a quantized operating point with
    the report-measured ``bytes_per_coeff`` and ``distortion_mse``.
    """

    quant_step: float | None
    bytes_per_coeff: float
    distortion_mse: float

    def __post_init__(self) -> None:
        if self.quant_step is not None:
            if not math.isfinite(self.quant_step) or self.quant_step <= 0.0:
                raise ValueError("quant_step must be a positive finite float or None")
        if not math.isfinite(self.bytes_per_coeff) or self.bytes_per_coeff < 0.0:
            raise ValueError("bytes_per_coeff must be a non-negative finite float")
        if not math.isfinite(self.distortion_mse) or self.distortion_mse < 0.0:
            raise ValueError("distortion_mse must be a non-negative finite float")


@dataclass(frozen=True)
class SubbandRDCurve:
    """A single Mallat detail subband's discrete RD curve.

    ``name`` is the report label (e.g. ``"L0_hh"``); ``level`` and
    ``orientation`` are parsed from it and map onto the actuator's
    ``frame_X_details:level:orientation`` key space.
    """

    name: str
    level: int
    orientation: str
    n_coeffs: int
    points: tuple[SubbandRDPoint, ...]

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("level must be >= 0")
        if self.orientation not in _VALID_ORIENTATIONS:
            raise ValueError(f"orientation must be one of {_VALID_ORIENTATIONS}; got {self.orientation!r}")
        if self.n_coeffs <= 0:
            raise ValueError("n_coeffs must be positive")
        if not self.points:
            raise ValueError("points must not be empty")


@dataclass(frozen=True)
class SubbandChoice:
    """The water-fill verdict for one subband."""

    name: str
    level: int
    orientation: str
    n_coeffs: int
    chosen_quant_step: float | None
    bytes_per_coeff: float
    distortion_mse: float
    subband_bytes: float


@dataclass(frozen=True)
class WaterfillSolution:
    """Result of the per-subband RD water-fill."""

    schema: str
    role: str
    lambda_value: float
    total_bytes: float
    baseline_total_bytes: float
    bytes_saved: float
    weighted_mean_mse: float
    target_total_bytes: float | None
    max_weighted_mse: float | None
    choices: tuple[SubbandChoice, ...]
    advisory_markers: dict[str, Any] = field(default_factory=lambda: dict(ADVISORY_MARKERS))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "role": self.role,
            "lambda_value": self.lambda_value,
            "total_bytes": self.total_bytes,
            "baseline_total_bytes": self.baseline_total_bytes,
            "bytes_saved": self.bytes_saved,
            "weighted_mean_mse": self.weighted_mean_mse,
            "target_total_bytes": self.target_total_bytes,
            "max_weighted_mse": self.max_weighted_mse,
            "choices": [
                {
                    "name": c.name,
                    "level": c.level,
                    "orientation": c.orientation,
                    "n_coeffs": c.n_coeffs,
                    "chosen_quant_step": c.chosen_quant_step,
                    "bytes_per_coeff": c.bytes_per_coeff,
                    "distortion_mse": c.distortion_mse,
                    "subband_bytes": c.subband_bytes,
                }
                for c in self.choices
            ],
            **self.advisory_markers,
        }


def _parse_subband_label(label: str) -> tuple[int, str]:
    """Parse a report subband label like ``"L0_hh"`` into ``(level, orient)``."""

    text = str(label).strip().lower()
    if not text.startswith("l") or "_" not in text:
        raise ValueError(f"unsupported subband label: {label!r} (expected L<level>_<orient>)")
    level_text, orient = text[1:].split("_", 1)
    try:
        level = int(level_text)
    except ValueError as exc:
        raise ValueError(f"subband label level must be an integer: {label!r}") from exc
    if orient not in _VALID_ORIENTATIONS:
        raise ValueError(f"subband label orientation must be one of {_VALID_ORIENTATIONS}: {label!r}")
    return level, orient


def load_subband_rd_curves_from_report(
    report: Mapping[str, Any] | str | Path,
    *,
    rate_field: str = "live_codec_brotli_bytes_per_coeff",
) -> list[SubbandRDCurve]:
    """Build :class:`SubbandRDCurve` list from an entropy-headroom report.

    Accepts a parsed dict or a path to the JSON report. Each ``per_subband``
    row contributes one curve whose points are the keep-raw baseline plus every
    ``quant_sweep`` operating point.
    """

    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise TypeError("report must be a Mapping or a path to a JSON report")
    per_subband = report.get("per_subband")
    if not isinstance(per_subband, Sequence) or not per_subband:
        raise ValueError("report must contain a non-empty 'per_subband' list")

    curves: list[SubbandRDCurve] = []
    for row in per_subband:
        if not isinstance(row, Mapping):
            raise ValueError("each per_subband entry must be a mapping")
        label = str(row["subband"])
        level, orient = _parse_subband_label(label)
        n_coeffs = int(row["n_coeffs"])
        baseline_bpc = float(row["current_raw_f32_brotli_bytes_per_coeff"])
        points: list[SubbandRDPoint] = [
            SubbandRDPoint(quant_step=None, bytes_per_coeff=baseline_bpc, distortion_mse=0.0)
        ]
        sweep = row.get("quant_sweep")
        if not isinstance(sweep, Sequence) or not sweep:
            raise ValueError(f"subband {label!r} must have a non-empty quant_sweep")
        for q in sweep:
            if rate_field in q:
                bytes_per_coeff = float(q[rate_field])
            elif "live_codec_brotli_bytes_per_coeff" in q:
                bytes_per_coeff = float(q["live_codec_brotli_bytes_per_coeff"])
            elif "live_codec_bytes_per_coeff" in q:
                bytes_per_coeff = float(q["live_codec_bytes_per_coeff"])
            else:
                raise ValueError(
                    f"subband {label!r} quant row missing rate field {rate_field!r}"
                )
            points.append(
                SubbandRDPoint(
                    quant_step=float(q["quant_step"]),
                    bytes_per_coeff=bytes_per_coeff,
                    distortion_mse=float(q["distortion_mse"]),
                )
            )
        curves.append(
            SubbandRDCurve(
                name=label,
                level=level,
                orientation=orient,
                n_coeffs=n_coeffs,
                points=tuple(points),
            )
        )
    return curves


def _pareto_frontier(points: Sequence[SubbandRDPoint]) -> list[SubbandRDPoint]:
    """Lower-convex-hull (Pareto) reduction of (rate, distortion) operating points.

    A Lagrangian ``min(D + λR)`` can only ever select points on the lower-left
    convex hull of the RD scatter. We (1) drop dominated points (a point is
    dominated if another has both <= bytes AND <= mse), then (2) keep only the
    convex-hull vertices, so the per-λ argmin is well-defined and monotone.
    Ties on bytes keep the lowest-mse point; ties on mse keep the lowest-bytes.
    """

    # Sort by bytes asc, then mse asc.
    ordered = sorted(points, key=lambda p: (p.bytes_per_coeff, p.distortion_mse))
    # Drop dominated: as bytes increase, mse must strictly decrease to survive.
    nondominated: list[SubbandRDPoint] = []
    best_mse = math.inf
    for p in ordered:
        if p.distortion_mse < best_mse - 1e-18:
            nondominated.append(p)
            best_mse = p.distortion_mse
    if len(nondominated) <= 2:
        return nondominated
    # Convex-hull vertices (lower hull in rate-distortion space): the slope
    # magnitude |ΔD/ΔR| must be monotonically decreasing as bytes increase.
    hull: list[SubbandRDPoint] = []
    for p in nondominated:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # slope a->b vs b->p (both ΔR > 0, ΔD < 0). Keep b only if it is a
            # genuine hull vertex: |slope(a,b)| > |slope(b,p)|.
            dr_ab = b.bytes_per_coeff - a.bytes_per_coeff
            dr_bp = p.bytes_per_coeff - b.bytes_per_coeff
            if dr_ab <= 0.0 or dr_bp <= 0.0:
                hull.pop()
                continue
            slope_ab = (b.distortion_mse - a.distortion_mse) / dr_ab
            slope_bp = (p.distortion_mse - b.distortion_mse) / dr_bp
            # Both slopes negative; b is interior (not a vertex) if the curve
            # does not bend, i.e. slope_ab <= slope_bp (less steep then steeper).
            if slope_ab <= slope_bp + 1e-30:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def _choose_point_for_lambda(
    hull: Sequence[SubbandRDPoint],
    *,
    lambda_value: float,
    distortion_weight: float,
) -> SubbandRDPoint:
    """Per-subband argmin of ``D_i·w_i + λ·R_i`` over the hull points."""

    best = hull[0]
    best_j = best.distortion_mse * distortion_weight + lambda_value * best.bytes_per_coeff
    for p in hull[1:]:
        j = p.distortion_mse * distortion_weight + lambda_value * p.bytes_per_coeff
        if j < best_j - 1e-18:
            best_j = j
            best = p
    return best


def _total_bytes_for_lambda(
    curves: Sequence[SubbandRDCurve],
    hulls: Sequence[list[SubbandRDPoint]],
    *,
    lambda_value: float,
    total_coeffs: int,
) -> tuple[float, float, list[SubbandRDPoint]]:
    """Return (total_bytes, weighted_mean_mse, chosen_points) at a given λ."""

    total_bytes = 0.0
    weighted_mse_sum = 0.0
    chosen: list[SubbandRDPoint] = []
    for curve, hull in zip(curves, hulls):
        w_i = curve.n_coeffs  # distortion weight (Parseval coefficient count)
        pt = _choose_point_for_lambda(
            hull, lambda_value=lambda_value, distortion_weight=float(w_i)
        )
        chosen.append(pt)
        total_bytes += pt.bytes_per_coeff * curve.n_coeffs
        weighted_mse_sum += pt.distortion_mse * curve.n_coeffs
    weighted_mean_mse = weighted_mse_sum / total_coeffs if total_coeffs else 0.0
    return total_bytes, weighted_mean_mse, chosen


def _build_solution(
    curves: Sequence[SubbandRDCurve],
    chosen: Sequence[SubbandRDPoint],
    *,
    lambda_value: float,
    total_coeffs: int,
    target_total_bytes: float | None,
    max_weighted_mse: float | None,
) -> WaterfillSolution:
    total_bytes = 0.0
    weighted_mse_sum = 0.0
    baseline_total_bytes = 0.0
    choices: list[SubbandChoice] = []
    for curve, pt in zip(curves, chosen):
        subband_bytes = pt.bytes_per_coeff * curve.n_coeffs
        total_bytes += subband_bytes
        weighted_mse_sum += pt.distortion_mse * curve.n_coeffs
        # baseline = keep-raw point (the first hull point is keep-raw by load order;
        # recompute from curve.points[0] which is always the baseline).
        baseline_total_bytes += curve.points[0].bytes_per_coeff * curve.n_coeffs
        choices.append(
            SubbandChoice(
                name=curve.name,
                level=curve.level,
                orientation=curve.orientation,
                n_coeffs=curve.n_coeffs,
                chosen_quant_step=pt.quant_step,
                bytes_per_coeff=pt.bytes_per_coeff,
                distortion_mse=pt.distortion_mse,
                subband_bytes=subband_bytes,
            )
        )
    weighted_mean_mse = weighted_mse_sum / total_coeffs if total_coeffs else 0.0
    return WaterfillSolution(
        schema=PER_SUBBAND_RD_WATERFILL_SCHEMA,
        role=PER_SUBBAND_RD_WATERFILL_ROLE,
        lambda_value=lambda_value,
        total_bytes=total_bytes,
        baseline_total_bytes=baseline_total_bytes,
        bytes_saved=baseline_total_bytes - total_bytes,
        weighted_mean_mse=weighted_mean_mse,
        target_total_bytes=target_total_bytes,
        max_weighted_mse=max_weighted_mse,
        choices=tuple(choices),
    )


def solve_per_subband_waterfill(
    curves: Sequence[SubbandRDCurve],
    *,
    target_total_bytes: float | None = None,
    max_weighted_mse: float | None = None,
    lambda_value: float | None = None,
    lambda_lo: float = 1e-9,
    lambda_hi: float = 1e12,
    max_iterations: int = 200,
) -> WaterfillSolution:
    """Solve the per-subband Lagrangian RD water-fill.

    Exactly one operating mode must be selected:

    * ``lambda_value`` set — solve at that fixed λ (returns the per-subband
      argmin allocation; no bisection).
    * ``target_total_bytes`` set — bisect λ so total detail bytes <= the target
      (the smallest λ whose allocation fits under the budget; lower λ = lower
      distortion).
    * ``max_weighted_mse`` set — bisect λ so weighted-mean MSE <= the ceiling
      (the cheapest allocation whose distortion stays under the ceiling).

    Higher λ ⇒ cheaper (coarser) operating points ⇒ fewer bytes / more
    distortion, so both byte-budget and distortion-ceiling targets are monotone
    in λ and bisectable.
    """

    if not curves:
        raise ValueError("curves must not be empty")
    modes_set = sum(
        x is not None for x in (target_total_bytes, max_weighted_mse, lambda_value)
    )
    if modes_set != 1:
        raise ValueError(
            "exactly one of target_total_bytes, max_weighted_mse, lambda_value must be set"
        )

    hulls = [_pareto_frontier(curve.points) for curve in curves]
    total_coeffs = sum(curve.n_coeffs for curve in curves)

    if lambda_value is not None:
        if lambda_value < 0.0 or not math.isfinite(lambda_value):
            raise ValueError("lambda_value must be a non-negative finite float")
        _, _, chosen = _total_bytes_for_lambda(
            curves, hulls, lambda_value=lambda_value, total_coeffs=total_coeffs
        )
        return _build_solution(
            curves,
            chosen,
            lambda_value=lambda_value,
            total_coeffs=total_coeffs,
            target_total_bytes=None,
            max_weighted_mse=None,
        )

    if target_total_bytes is not None:
        if target_total_bytes <= 0.0 or not math.isfinite(target_total_bytes):
            raise ValueError("target_total_bytes must be a positive finite float")

        def fits(lam: float) -> bool:
            tot, _, _ = _total_bytes_for_lambda(
                curves, hulls, lambda_value=lam, total_coeffs=total_coeffs
            )
            return tot <= target_total_bytes

        constraint_kind = "byte_budget"
        target = target_total_bytes
    else:
        assert max_weighted_mse is not None
        if max_weighted_mse < 0.0 or not math.isfinite(max_weighted_mse):
            raise ValueError("max_weighted_mse must be a non-negative finite float")

        def fits(lam: float) -> bool:
            # Lower λ → lower distortion. We want the LOWEST-byte allocation
            # whose distortion stays <= ceiling, i.e. the HIGHEST λ that still
            # satisfies the distortion ceiling.
            _, mse, _ = _total_bytes_for_lambda(
                curves, hulls, lambda_value=lam, total_coeffs=total_coeffs
            )
            return mse <= max_weighted_mse

        constraint_kind = "distortion_ceiling"
        target = max_weighted_mse

    lo, hi = lambda_lo, lambda_hi
    if constraint_kind == "byte_budget":
        # Find smallest λ that fits the byte budget. fits() is monotone:
        # False (too many bytes) at small λ → True at large λ.
        if fits(lo):
            chosen_lambda = lo
        elif not fits(hi):
            # Even the coarsest allocation exceeds the budget; return hi (best effort).
            chosen_lambda = hi
        else:
            for _ in range(max_iterations):
                mid = math.sqrt(lo * hi)
                if fits(mid):
                    hi = mid
                else:
                    lo = mid
                if hi / lo < 1.0 + 1e-9:
                    break
            chosen_lambda = hi
    else:
        # distortion ceiling: want highest λ whose distortion <= ceiling.
        # fits() is True at small λ (low distortion) → False at large λ.
        if not fits(lo):
            chosen_lambda = lo  # even finest allocation violates ceiling; best effort
        elif fits(hi):
            chosen_lambda = hi
        else:
            for _ in range(max_iterations):
                mid = math.sqrt(lo * hi)
                if fits(mid):
                    lo = mid
                else:
                    hi = mid
                if hi / lo < 1.0 + 1e-9:
                    break
            chosen_lambda = lo

    _, _, chosen = _total_bytes_for_lambda(
        curves, hulls, lambda_value=chosen_lambda, total_coeffs=total_coeffs
    )
    return _build_solution(
        curves,
        chosen,
        lambda_value=chosen_lambda,
        total_coeffs=total_coeffs,
        target_total_bytes=target_total_bytes,
        max_weighted_mse=max_weighted_mse,
    )


def emit_actuator_quant_steps(
    solution: WaterfillSolution,
    *,
    frames: Sequence[str] = ("frame_0_details", "frame_1_details"),
) -> dict[str, float]:
    """Convert a solution into the actuator's ``entropy_detail_quantization_steps`` map.

    The report's RD curves are frame-agnostic (aggregated across frames), so the
    solved per-(level, orientation) ``Δ`` is applied to every frame's details at
    that level+orientation. Subbands the solver left at keep-raw (no
    quantization, ``chosen_quant_step is None``) are omitted from the map so the
    actuator stores them losslessly.

    The returned dict is directly consumable by
    ``Z8JointCoefficientWaterfillConfig(entropy_detail_quantization_steps=...)``
    and every key round-trips through ``_parse_entropy_detail_step_key``.
    """

    steps: dict[str, float] = {}
    for choice in solution.choices:
        if choice.chosen_quant_step is None:
            continue
        for frame in frames:
            if frame not in ("frame_0_details", "frame_1_details"):
                raise ValueError(f"unsupported frame key: {frame!r}")
            key = f"{frame}:{choice.level}:{choice.orientation}"
            steps[key] = float(choice.chosen_quant_step)
    return steps


def _source_report_sha256(report: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(report, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _coverage_blockers(
    report: Mapping[str, Any],
    *,
    require_full_archive_coverage: bool,
) -> list[str]:
    blockers: list[str] = []
    pairs_measured = report.get("pairs_measured")
    total_pairs = report.get("total_pairs_in_archive")
    if (
        require_full_archive_coverage
        and pairs_measured is not None
        and total_pairs is not None
        and int(pairs_measured) < int(total_pairs)
    ):
        blockers.append(f"partial_headroom_coverage:{int(pairs_measured)}/{int(total_pairs)}")
    return blockers


def build_rd_waterfill_schedule_from_headroom_report(
    report: Mapping[str, Any] | str | Path,
    *,
    target_total_bytes: float | None = None,
    target_detail_byte_fraction: float | None = None,
    max_weighted_mse: float | None = None,
    lambda_value: float | None = None,
    require_full_archive_coverage: bool = True,
    rate_field: str = "live_codec_brotli_bytes_per_coeff",
) -> dict[str, Any]:
    """Solve a Lagrangian RD schedule and emit the actuator-ready step map."""

    if isinstance(report, (str, Path)):
        report_obj = json.loads(Path(report).read_text(encoding="utf-8"))
    else:
        report_obj = dict(report)
    if not isinstance(report_obj, Mapping):
        raise TypeError("report must be a Mapping or a path to a JSON report")

    curves = load_subband_rd_curves_from_report(report_obj, rate_field=rate_field)
    baseline_total_bytes = sum(
        curve.points[0].bytes_per_coeff * curve.n_coeffs for curve in curves
    )
    if target_detail_byte_fraction is not None:
        if target_total_bytes is not None:
            raise ValueError("target_total_bytes and target_detail_byte_fraction are exclusive")
        if not math.isfinite(target_detail_byte_fraction) or not (0.0 < target_detail_byte_fraction <= 1.0):
            raise ValueError("target_detail_byte_fraction must be in (0, 1]")
        target_total_bytes = baseline_total_bytes * float(target_detail_byte_fraction)
    solution = solve_per_subband_waterfill(
        curves,
        target_total_bytes=target_total_bytes,
        max_weighted_mse=max_weighted_mse,
        lambda_value=lambda_value,
    )
    steps = emit_actuator_quant_steps(solution)
    blockers = _coverage_blockers(
        report_obj,
        require_full_archive_coverage=require_full_archive_coverage,
    )
    if not steps:
        blockers.append("waterfill_selected_keep_raw_for_all_subbands")
    chosen = [
        {
            "aggregate_subband": choice.name,
            "level_idx": int(choice.level),
            "orientation": choice.orientation,
            "quant_step": choice.chosen_quant_step,
            "selection_reason": (
                "lagrangian_rd_waterfill_keep_raw"
                if choice.chosen_quant_step is None
                else "lagrangian_rd_waterfill_selected_quant_step"
            ),
            "distortion_mse": float(choice.distortion_mse),
            "live_codec_method": None,
            "live_codec_brotli_bytes_per_coeff": float(choice.bytes_per_coeff),
            "subband_bytes": float(choice.subband_bytes),
        }
        for choice in solution.choices
    ]
    return {
        "schema": "z8_entropy_delta_schedule.v2",
        "purpose": (
            "Per-subband detail quantization schedule selected by Lagrangian "
            "RD water-fill over measured Z8 entropy headroom curves; consumed "
            "by joint P18/P19 dead-zone materializers as "
            "entropy_detail_quantization_steps."
        ),
        **ADVISORY_MARKERS,
        "schedule_strategy": "per_subband_lagrangian_rd_waterfill",
        "source_report_tool": report_obj.get("tool"),
        "source_archive_path": report_obj.get("archive_path"),
        "source_archive_sha256": report_obj.get("archive_sha256"),
        "source_report_sha256": _source_report_sha256(report_obj),
        "source_archive_total_bytes": report_obj.get("archive_total_bytes"),
        "source_wavelet_blob_bytes": report_obj.get("wavelet_blob_bytes"),
        "pairs_measured": report_obj.get("pairs_measured"),
        "total_pairs_in_archive": report_obj.get("total_pairs_in_archive"),
        "require_full_archive_coverage": bool(require_full_archive_coverage),
        "rate_field": rate_field,
        "target_total_bytes": target_total_bytes,
        "target_detail_byte_fraction": target_detail_byte_fraction,
        "max_weighted_mse": max_weighted_mse,
        "lambda_value": lambda_value,
        "entropy_detail_quantization_steps": steps,
        "chosen_subbands": chosen,
        "waterfill_solution": solution.as_dict(),
        "blockers": blockers,
        "ready_for_materializer": bool(steps and not blockers),
    }


__all__ = [
    "ADVISORY_MARKERS",
    "PER_SUBBAND_RD_WATERFILL_ROLE",
    "PER_SUBBAND_RD_WATERFILL_SCHEMA",
    "SubbandChoice",
    "SubbandRDCurve",
    "SubbandRDPoint",
    "WaterfillSolution",
    "build_rd_waterfill_schedule_from_headroom_report",
    "emit_actuator_quant_steps",
    "load_subband_rd_curves_from_report",
    "solve_per_subband_waterfill",
]
