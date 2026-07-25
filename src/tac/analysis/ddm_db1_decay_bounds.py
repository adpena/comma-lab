# SPDX-License-Identifier: MIT
"""Deterministic, scorer-free decay and SegNet margin-mass bounds for DDM DB1.

This module deliberately separates three objects which are easy to conflate:

* the V19C *accepted proposal-order* curve;
* the fixed SN1 predicted-boundary margin atlas;
* a live descent trajectory.

Only the first two are materialized by the DB1 inputs.  In particular, the SN1
scratch shards retain ordered boundary incidences but not coordinates or target
error membership.  ``unique_count_bounds`` therefore returns the tight bounds
that follow from the globally measured duplicate budget instead of relabelling
incidences as unique pixels.

No scorer, optimizer, archive writer, or campaign launcher is imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
N600_SITES = 117_964_800
MARGIN_KEY = re.compile(r"p(?P<pair>\d{4})_w(?P<winner>\d)_r(?P<rival>\d)")
REPO_ROOT = Path(__file__).resolve().parents[3]


class DB1DecayBoundsError(RuntimeError):
    """Raised when input custody or the analytical domain is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def durable_path_label(path: Path) -> str:
    """Use repo-relative labels for tracked artifacts; keep durable SSD paths absolute."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _quantile(values: list[float], probability: float) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.quantile(finite, probability))


def unique_count_bounds(
    incidence_count: int,
    *,
    total_incidence_count: int,
    unique_total: int,
) -> tuple[int, int]:
    """Bound unique pixels represented by a prefix of ordered incidences.

    Every predicted-boundary pixel appears in at least one ordered
    winner-neighbour array.  Across the complete atlas, the number of duplicate
    incidences is exactly ``total_incidence_count - unique_total``.  At any
    threshold, at most that many selected incidences can be duplicates:

        max(0, I(delta) - D) <= N(delta) <= min(I(delta), B)

    where ``D = I(infinity) - B`` and ``B`` is the measured unique-boundary
    total.  This is much tighter than the generic five-class multiplicity bound.
    """

    if not 0 <= incidence_count <= total_incidence_count:
        raise ValueError("incidence_count must lie within the complete incidence population")
    if not 0 <= unique_total <= total_incidence_count:
        raise ValueError("unique_total must lie within the complete incidence population")
    duplicate_budget = total_incidence_count - unique_total
    return (
        max(0, incidence_count - duplicate_budget),
        min(incidence_count, unique_total),
    )


@dataclass(frozen=True)
class DecayFit:
    family: Literal["power", "exponential"]
    rss: float
    exponent: float
    asymptote: float
    amplitude: float
    aic: float
    aicc: float
    rmse: float

    def predict(self, admission: float) -> float:
        if self.family == "power":
            basis = admission ** (-self.exponent)
        else:
            basis = math.exp(-self.exponent * admission)
        return self.asymptote + self.amplitude * basis

    def target_projection(self, target: float, current_admission: int) -> dict[str, Any]:
        if target <= self.asymptote:
            return {
                "status": "ASYMPTOTE_AT_OR_ABOVE_TARGET",
                "total_admissions": None,
                "additional_admissions": None,
                "log10_total_admissions": None,
            }
        residual = target - self.asymptote
        if self.amplitude <= 0.0:
            return {
                "status": "NO_DECAY_AMPLITUDE",
                "total_admissions": None,
                "additional_admissions": None,
                "log10_total_admissions": None,
            }
        if self.family == "power":
            log_total = math.log10(self.amplitude / residual) / self.exponent
        else:
            ratio = residual / self.amplitude
            if not 0.0 < ratio < 1.0:
                total = 0.0
                log_total = float("-inf")
            else:
                total = -math.log(ratio) / self.exponent
                log_total = math.log10(total)
        if self.family == "power":
            total = 10.0**log_total if log_total < 308.0 else math.inf
        return {
            "status": "FINITE_MODEL_EXTRAPOLATION",
            "total_admissions": total if math.isfinite(total) else None,
            "additional_admissions": (
                max(0.0, total - current_admission) if math.isfinite(total) else None
            ),
            "log10_total_admissions": log_total,
        }


def _constrained_linear_coefficients(
    basis: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, float]:
    """Solve min ||y-c-a*f|| with 0<=c<=min(y), a>=0."""

    minimum = max(0.0, float(np.min(values)))
    design = np.column_stack((np.ones_like(basis), basis))
    unconstrained = np.linalg.lstsq(design, values, rcond=None)[0]
    candidates: list[tuple[float, float, float]] = []

    def add(asymptote: float, amplitude: float) -> None:
        c = float(np.clip(asymptote, 0.0, minimum))
        a = max(0.0, float(amplitude))
        residual = values - (c + a * basis)
        candidates.append((float(np.dot(residual, residual)), c, a))

    add(float(unconstrained[0]), float(unconstrained[1]))
    add(float(np.mean(values)), 0.0)
    add(0.0, float(np.dot(basis, values) / np.dot(basis, basis)))
    add(minimum, float(np.dot(basis, values - minimum) / np.dot(basis, basis)))
    return min(candidates)


def fit_decay_family(
    admissions: np.ndarray,
    values: np.ndarray,
    family: Literal["power", "exponential"],
) -> DecayFit:
    """Fit ``c+a*n^-p`` or ``c+a*exp(-k*n)`` without SciPy.

    For a fixed exponent the amplitude and asymptote are a constrained linear
    least-squares problem.  A log-spaced bracket followed by golden-section
    refinement makes the one-dimensional nonlinear fit deterministic.
    """

    x = np.asarray(admissions, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 8:
        raise ValueError("admissions and values must be equal one-dimensional arrays of size >=8")
    if np.any(x <= 0.0) or np.any(y < 0.0):
        raise ValueError("admissions must be positive and d_seg values non-negative")

    if family == "power":
        exponent_lo, exponent_hi = 1.0e-4, 20.0

        def basis(exponent: float) -> np.ndarray:
            return x ** (-exponent)

    elif family == "exponential":
        exponent_lo, exponent_hi = 1.0e-6, 2.0

        def basis(exponent: float) -> np.ndarray:
            return np.exp(-exponent * x)

    else:
        raise ValueError(f"unsupported family: {family}")

    grid = np.geomspace(exponent_lo, exponent_hi, 301)
    grid_rss = [_constrained_linear_coefficients(basis(float(q)), y)[0] for q in grid]
    best_index = int(np.argmin(grid_rss))
    lo_index = max(0, best_index - 1)
    hi_index = min(grid.size - 1, best_index + 1)
    lo = math.log(float(grid[lo_index]))
    hi = math.log(float(grid[hi_index]))
    golden = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(log_exponent: float) -> tuple[float, float, float, float]:
        exponent = math.exp(log_exponent)
        rss, asymptote, amplitude = _constrained_linear_coefficients(basis(exponent), y)
        return rss, exponent, asymptote, amplitude

    left = hi - golden * (hi - lo)
    right = lo + golden * (hi - lo)
    left_result = objective(left)
    right_result = objective(right)
    for _ in range(48):
        if left_result[0] < right_result[0]:
            hi = right
            right = left
            right_result = left_result
            left = hi - golden * (hi - lo)
            left_result = objective(left)
        else:
            lo = left
            left = right
            left_result = right_result
            right = lo + golden * (hi - lo)
            right_result = objective(right)

    rss, exponent, asymptote, amplitude = min(left_result, right_result)
    sample_count = int(y.size)
    parameter_count = 3
    safe_rss = max(rss, np.finfo(np.float64).tiny)
    aic = sample_count * math.log(safe_rss / sample_count) + 2 * parameter_count
    aicc = aic + (
        2 * parameter_count * (parameter_count + 1)
        / (sample_count - parameter_count - 1)
    )
    return DecayFit(
        family=family,
        rss=rss,
        exponent=exponent,
        asymptote=asymptote,
        amplitude=amplitude,
        aic=aic,
        aicc=aicc,
        rmse=math.sqrt(rss / sample_count),
    )


def bootstrap_decay_fit(
    admissions: np.ndarray,
    values: np.ndarray,
    fit: DecayFit,
    *,
    target: float,
    horizon_admission: int,
    replicates: int,
    seed: int,
    block_length: int = 8,
) -> dict[str, Any]:
    """Circular moving-block residual bootstrap, conditional on proposal order."""

    x = np.asarray(admissions, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    prediction = np.asarray([fit.predict(float(value)) for value in x], dtype=np.float64)
    residual = y - prediction
    residual -= residual.mean()
    rng = np.random.default_rng(seed)
    exponents: list[float] = []
    asymptotes: list[float] = []
    horizon_values: list[float] = []
    finite_target_logs: list[float] = []
    unreachable = 0

    for _ in range(replicates):
        parts: list[np.ndarray] = []
        selected = 0
        while selected < y.size:
            start = int(rng.integers(0, y.size))
            indices = np.arange(start, start + block_length) % y.size
            part = residual[indices]
            parts.append(part)
            selected += part.size
        sampled = np.concatenate(parts)[: y.size]
        fitted = fit_decay_family(x, prediction + sampled, fit.family)
        exponents.append(fitted.exponent)
        asymptotes.append(fitted.asymptote)
        horizon_values.append(fitted.predict(float(horizon_admission)))
        target_projection = fitted.target_projection(target, int(x[-1]))
        log_total = target_projection["log10_total_admissions"]
        if log_total is None:
            unreachable += 1
        else:
            finite_target_logs.append(float(log_total))

    return {
        "method": "circular_moving_block_residual_bootstrap",
        "conditional_on": (
            "observed finite V19C proposal ordering; excludes unseen proposal families, "
            "descent transport, and selection-process uncertainty"
        ),
        "replicates": replicates,
        "seed": seed,
        "block_length": block_length,
        "exponent_ci95": [
            _quantile(exponents, 0.025),
            _quantile(exponents, 0.975),
        ],
        "asymptote_ci95": [
            _quantile(asymptotes, 0.025),
            _quantile(asymptotes, 0.975),
        ],
        "horizon_d_seg_ci95": [
            _quantile(horizon_values, 0.025),
            _quantile(horizon_values, 0.975),
        ],
        "target_unreachable_fraction": unreachable / replicates,
        "finite_target_log10_admissions_ci95_conditional": [
            _quantile(finite_target_logs, 0.025),
            _quantile(finite_target_logs, 0.975),
        ],
    }


def analyze_v19c_curve(
    curve_path: Path,
    *,
    target: float,
    horizon_additional_admissions: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    payload = json.loads(curve_path.read_text(encoding="utf-8"))
    curve = payload["accepted_curve_n600"]
    if len(curve) != 104:
        raise DB1DecayBoundsError(f"expected 104 V19C admissions, found {len(curve)}")
    admissions = np.arange(1, len(curve) + 1, dtype=np.float64)
    values = np.asarray([float(row["d_seg"]) for row in curve], dtype=np.float64)
    horizon_admission = len(curve) + horizon_additional_admissions
    fits: dict[str, Any] = {}
    fit_objects: dict[str, DecayFit] = {}
    for family_index, family in enumerate(("power", "exponential")):
        fitted = fit_decay_family(admissions, values, family)
        fit_objects[family] = fitted
        fits[family] = {
            "equation": (
                "d_seg(n)=c+a*n^(-p)"
                if family == "power"
                else "d_seg(n)=c+a*exp(-k*n)"
            ),
            "parameter_name": "p" if family == "power" else "k",
            "parameter": fitted.exponent,
            "asymptote": fitted.asymptote,
            "amplitude": fitted.amplitude,
            "rss": fitted.rss,
            "rmse": fitted.rmse,
            "aic": fitted.aic,
            "aicc": fitted.aicc,
            "predicted_d_seg_at_horizon": fitted.predict(float(horizon_admission)),
            "target_projection": fitted.target_projection(target, len(curve)),
            "ci": bootstrap_decay_fit(
                admissions,
                values,
                fitted,
                target=target,
                horizon_admission=horizon_admission,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + family_index,
            ),
        }

    selected = min(fit_objects.values(), key=lambda row: row.aicc)
    tail_sensitivity: list[dict[str, Any]] = []
    for window in (70, 52, 35):
        for family in ("power", "exponential"):
            fitted = fit_decay_family(admissions[-window:], values[-window:], family)
            tail_sensitivity.append(
                {
                    "window_admissions": window,
                    "first_admission": int(admissions[-window]),
                    "family": family,
                    "asymptote": fitted.asymptote,
                    "parameter": fitted.exponent,
                    "aicc": fitted.aicc,
                    "predicted_d_seg_at_horizon": fitted.predict(float(horizon_admission)),
                    "target_projection": fitted.target_projection(target, len(curve)),
                }
            )
    return {
        "source": {
            "path": durable_path_label(curve_path),
            "bytes": curve_path.stat().st_size,
            "sha256": sha256_file(curve_path),
        },
        "admission_count": len(curve),
        "first_d_seg": float(values[0]),
        "terminal_d_seg": float(values[-1]),
        "terminal_errors": int(payload["final"]["measurement"]["errors"]),
        "horizon_additional_admissions": horizon_additional_admissions,
        "horizon_total_admission": horizon_admission,
        "target_d_seg": target,
        "fits": fits,
        "selected_by_aicc": selected.family,
        "delta_aicc_power_minus_exponential": (
            fit_objects["power"].aicc - fit_objects["exponential"].aicc
        ),
        "tail_window_sensitivity": tail_sensitivity,
        "scope": (
            "FORMULATION: V19C finite accepted proposal-order curve only; this is not a "
            "live-descent law and does not close scorer-recursive or solve-derived families"
        ),
    }


def _verify_jsonl_output_hash(receipt: dict[str, Any], path: Path) -> None:
    matches = [row for row in receipt["outputs"] if Path(row["path"]).name == path.name]
    if len(matches) != 1:
        raise DB1DecayBoundsError(f"receipt does not identify exactly one output named {path.name}")
    expected = matches[0]
    actual_sha = sha256_file(path)
    if path.stat().st_size != int(expected["bytes"]) or actual_sha != expected["sha256"]:
        raise DB1DecayBoundsError(f"SN1 tracked output custody failed for {path}")


def load_margin_distances(
    sn1_receipt_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and hash-verify all certified SN1 margin shards, converting to AT1 d2."""

    receipt = json.loads(sn1_receipt_path.read_text(encoding="utf-8"))
    norms = receipt["measurement"]["head_norms_exact"]
    artifacts = sorted(
        (
            row
            for row in receipt["scratch_artifacts"]
            if row["path"].endswith("_margins.npz")
        ),
        key=lambda row: row["path"],
    )
    if len(artifacts) != 38:
        raise DB1DecayBoundsError(f"expected 38 SN1 margin shards, found {len(artifacts)}")
    all_distances: list[np.ndarray] = []
    digest_rows: list[str] = []
    coordinate_arrays = 0
    margin_arrays = 0
    for artifact in artifacts:
        path = Path(artifact["path"])
        if path.stat().st_size != int(artifact["bytes"]):
            raise DB1DecayBoundsError(f"SN1 shard byte-count mismatch: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != artifact["sha256"]:
            raise DB1DecayBoundsError(f"SN1 shard SHA-256 mismatch: {path}")
        digest_rows.append(f"{path.name}\t{artifact['bytes']}\t{actual_sha}")
        with np.load(path, allow_pickle=False) as shard:
            if len(shard.files) != int(artifact["array_count"]):
                raise DB1DecayBoundsError(f"SN1 shard array-count mismatch: {path}")
            for key in shard.files:
                match = MARGIN_KEY.fullmatch(key)
                if match is None:
                    coordinate_arrays += 1
                    continue
                winner = int(match.group("winner"))
                rival = int(match.group("rival"))
                if winner == rival:
                    raise DB1DecayBoundsError(f"invalid same-class margin array: {key}")
                values = np.asarray(shard[key], dtype=np.float64)
                if values.ndim != 1 or np.any(values < -1.0e-5):
                    raise DB1DecayBoundsError(f"invalid margin values in {path}:{key}")
                orientation = f"{CLASS_NAMES[winner]}->{CLASS_NAMES[rival]}"
                all_distances.append(values / float(norms[orientation]))
                margin_arrays += 1
    if coordinate_arrays:
        raise DB1DecayBoundsError(
            "unexpected non-margin arrays found; update the coordinate-custody classifier"
        )
    distances = np.sort(np.concatenate(all_distances))
    digest = hashlib.sha256(("\n".join(digest_rows) + "\n").encode("utf-8")).hexdigest()
    return distances, {
        "receipt": {
            "path": durable_path_label(sn1_receipt_path),
            "bytes": sn1_receipt_path.stat().st_size,
            "sha256": sha256_file(sn1_receipt_path),
        },
        "shard_count": len(artifacts),
        "margin_array_count": margin_arrays,
        "coordinate_array_count": coordinate_arrays,
        "all_shards_sha256_verified": True,
        "shard_manifest_digest_sha256": digest,
        "distance_law": "d2=positive_winner_rival_margin/||w_winner-w_rival||_2",
    }


def _read_unique_boundary_total(telemetry_path: Path) -> tuple[int, int]:
    unique_total = 0
    frame_count = 0
    with telemetry_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "boundary_pixel_count" in row:
                telemetry = row
            else:
                telemetry = row.get("telemetry", {})
            if "boundary_pixel_count" not in telemetry:
                continue
            unique_total += int(telemetry["boundary_pixel_count"])
            frame_count += 1
    if frame_count != 600:
        raise DB1DecayBoundsError(f"expected 600 SN1 telemetry rows, found {frame_count}")
    return unique_total, frame_count


def _rank_distance(sorted_distances: np.ndarray, required_incidence_count: int) -> float | None:
    if required_incidence_count <= 0:
        return 0.0
    if required_incidence_count > sorted_distances.size:
        return None
    return float(sorted_distances[required_incidence_count - 1])


def required_delta_bounds(
    sorted_distances: np.ndarray,
    *,
    unique_total: int,
    required_unique_count: int,
) -> dict[str, Any]:
    total_incidence_count = int(sorted_distances.size)
    duplicate_budget = total_incidence_count - unique_total
    if required_unique_count <= 0:
        return {
            "status": "NO_ADDITIONAL_UNIQUE_MASS_REQUIRED",
            "necessary_delta_lower_bound": 0.0,
            "duplicate_budget_sufficient_delta_upper_bound": 0.0,
        }
    if required_unique_count > unique_total:
        return {
            "status": "EXCEEDS_COMPLETE_FIXED_BOUNDARY_SUPPORT",
            "necessary_delta_lower_bound": None,
            "duplicate_budget_sufficient_delta_upper_bound": None,
        }
    return {
        "status": "BOUNDED_FROM_MARGIN_ORDER_STATISTICS",
        "necessary_delta_lower_bound": _rank_distance(
            sorted_distances, required_unique_count
        ),
        "duplicate_budget_sufficient_delta_upper_bound": _rank_distance(
            sorted_distances, required_unique_count + duplicate_budget
        ),
    }


def analyze_margin_mass(
    *,
    sn1_receipt_path: Path,
    telemetry_path: Path,
    at1_manifest_path: Path,
    at1_atlas_path: Path,
    operating_points: dict[str, float],
    targets: dict[str, float],
    opening_delta_d_seg: float,
) -> dict[str, Any]:
    sn1_receipt = json.loads(sn1_receipt_path.read_text(encoding="utf-8"))
    _verify_jsonl_output_hash(sn1_receipt, telemetry_path)
    distances, shard_custody = load_margin_distances(sn1_receipt_path)
    unique_total, frame_count = _read_unique_boundary_total(telemetry_path)
    if unique_total > distances.size:
        raise DB1DecayBoundsError("unique SN1 boundary total exceeds ordered incidences")
    duplicate_budget = int(distances.size - unique_total)
    opening_corrected_errors = int(round(opening_delta_d_seg * N600_SITES))
    opening_delta_interval = required_delta_bounds(
        distances,
        unique_total=unique_total,
        required_unique_count=opening_corrected_errors,
    )
    opening_delta_upper = opening_delta_interval[
        "duplicate_budget_sufficient_delta_upper_bound"
    ]
    if opening_delta_upper is None:
        opening_radius_unique_upper = unique_total
    else:
        opening_radius_incidences = int(
            np.searchsorted(distances, opening_delta_upper, side="right")
        )
        _, opening_radius_unique_upper = unique_count_bounds(
            opening_radius_incidences,
            total_incidence_count=int(distances.size),
            unique_total=unique_total,
        )

    at1_manifest = json.loads(at1_manifest_path.read_text(encoding="utf-8"))
    at1_atlas = json.loads(at1_atlas_path.read_text(encoding="utf-8"))
    at1_atlas_sha = sha256_file(at1_atlas_path)
    at1_payload_sha = hashlib.sha256(
        json.dumps(
            at1_atlas,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if at1_payload_sha != at1_manifest["contraction_atlas_sha256"]:
        raise DB1DecayBoundsError("AT1 contraction-atlas SHA-256 mismatch")
    if int(at1_manifest["counts"]["gaze_pairs"]) != 600:
        raise DB1DecayBoundsError("AT1 manifest is not full n600")

    delta_grid = (
        0.0,
        1e-6,
        2e-6,
        5e-6,
        1e-5,
        2e-5,
        5e-5,
        1e-4,
        2e-4,
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
        1e-1,
        2e-1,
        5e-1,
        1.0,
    )
    cdf_rows: list[dict[str, Any]] = []
    for delta in delta_grid:
        incidence_count = int(np.searchsorted(distances, delta, side="right"))
        lower, upper = unique_count_bounds(
            incidence_count,
            total_incidence_count=int(distances.size),
            unique_total=unique_total,
        )
        cdf_rows.append(
            {
                "delta_d2": delta,
                "ordered_incidence_count": incidence_count,
                "unique_pixel_count_lower": lower,
                "unique_pixel_count_upper": upper,
                "global_site_fraction_lower": lower / N600_SITES,
                "global_site_fraction_upper": upper / N600_SITES,
            }
        )

    target_rows: list[dict[str, Any]] = []
    for operating_name, d_seg in operating_points.items():
        errors = int(round(d_seg * N600_SITES))
        fixed_support_floor_errors = max(0, errors - unique_total)
        per_target: list[dict[str, Any]] = []
        for target_name, target_d_seg in targets.items():
            target_errors = int(round(target_d_seg * N600_SITES))
            correction_debt = max(0, errors - target_errors)
            per_target.append(
                {
                    "target": target_name,
                    "target_d_seg": target_d_seg,
                    "target_errors": target_errors,
                    "correction_debt_errors": correction_debt,
                    "required_delta": required_delta_bounds(
                        distances,
                        unique_total=unique_total,
                        required_unique_count=correction_debt,
                    ),
                }
            )
        target_rows.append(
            {
                "operating_point": operating_name,
                "d_seg": d_seg,
                "errors": errors,
                "fixed_initial_boundary_oracle_floor_errors": fixed_support_floor_errors,
                "fixed_initial_boundary_oracle_floor_d_seg": (
                    fixed_support_floor_errors / N600_SITES
                ),
                "fixed_initial_boundary_terminal_d_seg_band": [
                    fixed_support_floor_errors / N600_SITES,
                    d_seg,
                ],
                "opening_calibrated_fixed_atlas_one_step_beneficial_rate_bound": {
                    "delta_d_seg_lower": 0.0,
                    "delta_d_seg_upper": min(errors, opening_radius_unique_upper)
                    / N600_SITES,
                    "conditioning": (
                        "assumes the first opening net-error delta corresponds to a fixed "
                        "AT1 d2 radius no larger than the duplicate-budget sufficient endpoint; "
                        "zero is the honest lower bound because target-error membership is absent"
                    ),
                },
                "target_rows": per_target,
            }
        )

    return {
        "sn1_custody": {
            **shard_custody,
            "telemetry": {
                "path": durable_path_label(telemetry_path),
                "bytes": telemetry_path.stat().st_size,
                "sha256": sha256_file(telemetry_path),
                "frame_count": frame_count,
            },
        },
        "at1_custody": {
            "manifest_path": str(at1_manifest_path),
            "manifest_sha256": sha256_file(at1_manifest_path),
            "atlas_path": str(at1_atlas_path),
            "atlas_file_sha256": at1_atlas_sha,
            "atlas_canonical_payload_sha256": at1_payload_sha,
            "counts": at1_manifest["counts"],
            "evidence_axis": at1_manifest["evidence_axis"],
        },
        "total_ordered_boundary_incidences": int(distances.size),
        "total_unique_boundary_pixels": unique_total,
        "total_duplicate_incidences": duplicate_budget,
        "incidence_to_unique_ratio": float(distances.size / unique_total),
        "distance_quantiles": {
            str(probability): float(np.quantile(distances, probability))
            for probability in (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "n_delta": cdf_rows,
        "operating_point_bounds": target_rows,
        "opening_anchor": {
            "label": "[naive-menu upper bound]",
            "delta_d_seg": opening_delta_d_seg,
            "net_corrected_errors": opening_corrected_errors,
            "effective_d2_interval_if_every_net_correction_came_from_fixed_boundary_atlas": (
                opening_delta_interval
            ),
            "non_identifiability": (
                "net error reduction does not identify realized head displacement, corrected "
                "pixel membership, beneficial/gross flip counts, or replenished boundary mass"
            ),
        },
        "law": (
            "max(0,I(delta)-(I(infinity)-B)) <= N(delta) <= min(I(delta),B)"
        ),
        "law_scope": (
            "MEASURED fixed SN1 predicted-boundary atlas only; N(delta) counts unique "
            "predicted-boundary pixels within AT1 head-space d2, not error-conditioned "
            "correctable pixels and not a live-descent replenishment process"
        ),
        "missing_for_e7_optimal_stop": [
            "pixel coordinates and target-error membership for each SN1 margin incidence",
            "realized per-step head-space transport and gross beneficial/harmful flip flow",
            "conditional residual flip-field entropy H(F|free decoder context) versus D",
            "same-parent marginal step seconds and marginal counted bytes",
        ],
    }


__all__ = [
    "CLASS_NAMES",
    "DB1DecayBoundsError",
    "DecayFit",
    "N600_SITES",
    "analyze_margin_mass",
    "analyze_v19c_curve",
    "bootstrap_decay_fit",
    "fit_decay_family",
    "required_delta_bounds",
    "sha256_file",
    "durable_path_label",
    "unique_count_bounds",
]
