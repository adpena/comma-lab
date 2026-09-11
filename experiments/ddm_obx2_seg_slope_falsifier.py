# SPDX-License-Identifier: MIT
"""PRE-REGISTERED seg-slope falsifier for the OBX2 w2 stage.

This file is committed BEFORE the w2 epoch data is fitted.  Everything that
decides the verdict — the target, the model family, the fit window, the
flattening test, and the closure rule — is written here first, so the verdict
cannot be chosen after seeing the curve.

The question.  The OBX2 gate is `100*d_seg + sqrt(10*d_pose) < 0.04`.  The
epoch-20 n600 row measured `d_seg = 0.02450141`, whose Seg leg alone is 2.45 —
61× the whole gate.  Seg is now the binding term.  Either the w2 stage's seg
descent reaches the gate inside its own 200 epochs or the base-only mechanism
does not solve Seg at this formulation, and continuing to feed it is spending
days on a settled question.

The instrument.  `mean_expected_flip`, logged per epoch, was MEASURED to track
the parsed object's `d_seg` to 1.0% (0.02474 surrogate against 0.02450141
measured, same object).  That is what makes a per-epoch fit possible at all; it
is a surrogate and never a score.

TARGETS (both reported; neither chosen after the fact)
  * `D_SEG_CEILING` 4.0e-4 — the Seg leg alone exactly fills the gate, leaving
    ZERO room for Pose.  An object at the ceiling cannot pass.
  * `D_SEG_POSE_CONSISTENT` 1.87e-4 — the Seg budget left once Pose sits where a
    scorer-plane-matched render on this grid measured it (`d_pose` 4.52317e-5,
    Pose leg 0.021267).  This is the target an object actually has to hit.

MODEL FAMILY (both fitted; the one with the smaller log-space residual is
reported as primary, and BOTH projections are stated)
  * power law     `log10(flip) = a + b * log10(epoch + 1)`
  * exponential   `log10(flip) = a + b * epoch`

VERDICT RULE, pre-registered
  * CLOSED-AT-FORMULATION-SCOPE for an arm when EITHER
      (a) the primary fit's projected epoch to `D_SEG_CEILING` exceeds
          `STAGE_EPOCHS` (200), the stage the arm actually has; or
      (b) the descent has FLATTENED: fitted over the last `FLATTEN_WINDOW` (20)
          epochs, the curve implies less than `FLATTEN_MIN_FACTOR` (1.5x)
          improvement in `d_seg` over the following 100 epochs.
    Test (b) is only applied once an arm has reached `FLATTEN_MIN_EPOCH` (60),
    because a flattening test on a short window is a test of noise.
  * OPEN otherwise: the arm still has slope and keeps its budget.
  * A projection is reported with its own error: the log-space residual standard
    error is propagated to an epoch band, and a band that straddles
    `STAGE_EPOCHS` is reported as INDETERMINATE rather than forced to a side.

Nothing here reads a scorer, trains, or launches.  It fits numbers a live run
already wrote and prints a verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

SCHEMA = "ddm_obx2_seg_slope_falsifier.v1"

# --- pre-registered constants -------------------------------------------------
D_SEG_CEILING = 4.0e-4
D_SEG_POSE_CONSISTENT = 1.87e-4
STAGE_EPOCHS = 200
FLATTEN_WINDOW = 20
FLATTEN_MIN_EPOCH = 60
FLATTEN_MIN_FACTOR = 1.5
FLATTEN_HORIZON_EPOCHS = 100
MIN_EPOCHS_TO_FIT = 6
SURROGATE_CALIBRATION = {
    "surrogate": "mean_expected_flip",
    "measured_d_seg": 0.02450141059,
    "surrogate_value": 0.02474,
    "relative_error": 0.00974,
    "source": "base-only epoch-20 n600 row against the w2 stage's first logged epoch",
}


class SegSlopeError(RuntimeError):
    """Fail-closed refusal for a falsifier input that cannot support a verdict."""


@dataclass(frozen=True)
class Fit:
    """One fitted model with its own log-space error."""

    family: str
    intercept: float
    slope: float
    residual_standard_error: float
    points: int

    def predict_log10(self, epoch: float) -> float:
        x = math.log10(epoch + 1.0) if self.family == "power" else epoch
        return self.intercept + self.slope * x

    def epoch_reaching(self, target: float) -> float | None:
        """Epoch at which the fit reaches `target`, or None if it never does."""

        if self.slope >= 0.0:
            return None
        goal = math.log10(target)
        if self.family == "power":
            return 10.0 ** ((goal - self.intercept) / self.slope) - 1.0
        return (goal - self.intercept) / self.slope

    def epoch_band(self, target: float) -> tuple[float | None, float | None]:
        """Epoch band from propagating one residual standard error either way."""

        if self.slope >= 0.0:
            return (None, None)
        band = []
        for shift in (-self.residual_standard_error, self.residual_standard_error):
            shifted = Fit(self.family, self.intercept + shift, self.slope, 0.0, self.points)
            band.append(shifted.epoch_reaching(target))
        low, high = band
        if low is None or high is None:
            return (low, high)
        return (min(low, high), max(low, high))


@dataclass
class ArmVerdict:
    arm: str
    epochs: list[int] = field(default_factory=list)
    surrogate: list[float] = field(default_factory=list)
    fits: dict[str, Fit] = field(default_factory=dict)
    primary: str = ""
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)


def _least_squares(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """Ordinary least squares with the residual standard error."""

    n = len(xs)
    if n < 3:
        raise SegSlopeError("a slope needs at least three points")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        raise SegSlopeError("fit inputs have no spread in the epoch axis")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
        rse = math.sqrt(sum(r * r for r in residuals) / (n - 2))
    else:
        rse = 0.0
    return intercept, slope, rse


def fit_family(epochs: Sequence[int], values: Sequence[float], family: str) -> Fit:
    if family not in ("power", "exponential"):
        raise SegSlopeError(f"unknown model family: {family}")
    xs = [math.log10(e + 1.0) if family == "power" else float(e) for e in epochs]
    ys = [math.log10(v) for v in values]
    intercept, slope, rse = _least_squares(xs, ys)
    return Fit(family, intercept, slope, rse, len(epochs))


def read_epoch_rows(log_path: Path) -> tuple[list[int], list[float]]:
    """Epoch index and `mean_expected_flip` from a run log, in order."""

    epochs: list[int] = []
    values: list[float] = []
    for line in log_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{") or '"mean_expected_flip"' not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        flip = float(row["mean_expected_flip"])
        if flip <= 0.0:
            raise SegSlopeError(f"non-positive surrogate in {log_path}: {flip}")
        epochs.append(int(row["epoch"]))
        values.append(flip)
    return epochs, values


def flattening_factor(fit: Fit, last_epoch: int) -> float:
    """Improvement factor the fit implies over the next `FLATTEN_HORIZON_EPOCHS`."""

    now = fit.predict_log10(last_epoch)
    later = fit.predict_log10(last_epoch + FLATTEN_HORIZON_EPOCHS)
    return 10.0 ** (now - later)


def judge(arm: str, epochs: Sequence[int], values: Sequence[float]) -> ArmVerdict:
    """Apply the pre-registered rule.  No branch here was chosen after the data."""

    result = ArmVerdict(arm=arm, epochs=list(epochs), surrogate=list(values))
    if len(epochs) < MIN_EPOCHS_TO_FIT:
        result.verdict = "INSUFFICIENT_EPOCHS"
        result.reasons.append(
            f"{len(epochs)} epochs logged; the pre-registered minimum is {MIN_EPOCHS_TO_FIT}"
        )
        return result
    for family in ("power", "exponential"):
        result.fits[family] = fit_family(epochs, values, family)
    result.primary = min(result.fits, key=lambda name: result.fits[name].residual_standard_error)
    primary = result.fits[result.primary]

    reached = primary.epoch_reaching(D_SEG_CEILING)
    low, high = primary.epoch_band(D_SEG_CEILING)
    if reached is None:
        result.verdict = "CLOSED_AT_FORMULATION_SCOPE"
        result.reasons.append("the primary fit does not descend; it never reaches the ceiling")
        return result
    if low is not None and high is not None and low <= STAGE_EPOCHS <= high:
        result.verdict = "INDETERMINATE"
        result.reasons.append(
            f"projected epoch band [{low:.0f}, {high:.0f}] straddles the stage's {STAGE_EPOCHS}"
        )
        return result
    if reached > STAGE_EPOCHS:
        result.verdict = "CLOSED_AT_FORMULATION_SCOPE"
        result.reasons.append(
            f"projected epoch {reached:.0f} to d_seg {D_SEG_CEILING:.1e} exceeds the stage's {STAGE_EPOCHS}"
        )
        return result

    last_epoch = max(epochs)
    if last_epoch >= FLATTEN_MIN_EPOCH:
        window = [(e, v) for e, v in zip(epochs, values, strict=True) if e > last_epoch - FLATTEN_WINDOW]
        if len(window) >= 3:
            tail = fit_family([e for e, _ in window], [v for _, v in window], result.primary)
            factor = flattening_factor(tail, last_epoch)
            if factor < FLATTEN_MIN_FACTOR:
                result.verdict = "CLOSED_AT_FORMULATION_SCOPE"
                result.reasons.append(
                    f"flattened: the last {FLATTEN_WINDOW} epochs imply {factor:.2f}x over the next "
                    f"{FLATTEN_HORIZON_EPOCHS}, under the {FLATTEN_MIN_FACTOR}x floor"
                )
                return result
            result.reasons.append(f"not flattened: tail implies {factor:.2f}x over {FLATTEN_HORIZON_EPOCHS} epochs")
    else:
        result.reasons.append(
            f"flattening test not applied: last epoch {last_epoch} is under the {FLATTEN_MIN_EPOCH} floor"
        )
    result.verdict = "OPEN"
    result.reasons.append(f"projected epoch {reached:.0f} to d_seg {D_SEG_CEILING:.1e} is inside the stage")
    return result


def describe(result: ArmVerdict) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "arm": result.arm,
        "epochs_fitted": len(result.epochs),
        "epoch_range": [min(result.epochs), max(result.epochs)] if result.epochs else None,
        "latest_surrogate": result.surrogate[-1] if result.surrogate else None,
        "verdict": result.verdict,
        "reasons": result.reasons,
        "primary_family": result.primary,
        "fits": {},
    }
    for family, fit in result.fits.items():
        entry: dict[str, Any] = {
            "intercept": fit.intercept,
            "slope": fit.slope,
            "residual_standard_error_log10": fit.residual_standard_error,
            "points": fit.points,
        }
        for label, target in (("ceiling_4e-4", D_SEG_CEILING), ("pose_consistent_1.87e-4", D_SEG_POSE_CONSISTENT)):
            reached = fit.epoch_reaching(target)
            low, high = fit.epoch_band(target)
            entry[f"epoch_to_{label}"] = reached
            entry[f"epoch_band_to_{label}"] = [low, high]
        payload["fits"][family] = entry
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OBX2 pre-registered seg-slope falsifier")
    parser.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "RUN_LOG"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = []
    for name, log_path in args.arm:
        epochs, values = read_epoch_rows(Path(log_path))
        results.append(describe(judge(name, epochs, values)))
    receipt = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[training-surrogate projection, not a score]",
        "score_claim": False,
        "promotable": False,
        "pre_registered": {
            "d_seg_ceiling": D_SEG_CEILING,
            "d_seg_pose_consistent": D_SEG_POSE_CONSISTENT,
            "stage_epochs": STAGE_EPOCHS,
            "flatten_window": FLATTEN_WINDOW,
            "flatten_min_epoch": FLATTEN_MIN_EPOCH,
            "flatten_min_factor": FLATTEN_MIN_FACTOR,
            "flatten_horizon_epochs": FLATTEN_HORIZON_EPOCHS,
            "min_epochs_to_fit": MIN_EPOCHS_TO_FIT,
            "surrogate_calibration": SURROGATE_CALIBRATION,
        },
        "arms": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
