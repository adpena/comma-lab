#!/usr/bin/env python3
"""Prediction-vs-empirical delta logger.

Closes the prediction loop introduced by the solver-stack wire-in
(commit ``d484507f``). For every substrate that has BOTH a typed prediction
anchor in ``.omx/state/predicted_anchors_solver_stack_wire_in_20260513.jsonl``
AND an empirical contest-CUDA / contest-CPU anchor in
``.omx/state/continual_learning_posterior.json``, compute the delta

.. math::

    \\Delta := \\text{empirical\\_score} - \\text{predicted\\_midpoint}

and report whether the prediction band CONTAINED the empirical score.

The output is a structured JSON + CSV; if matplotlib is available, a plot is
emitted too. The script is a READ-ONLY consumer of state — it never mutates
the predicted_anchors or continual_learning_posterior files.

Per CLAUDE.md "Apples-to-apples evidence discipline":
- Predictions stay tagged ``[prediction; planning_only]``.
- Empirical anchors keep their original axis label (``[contest-CUDA]`` /
  ``[contest-CPU GHA Linux x86_64]`` / ``[macOS-CPU advisory only]``).
- The delta logger NEVER promotes a prediction; the verdict is informational.

Per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in:
- Hook 5 (continual-learning posterior update): this script consumes the
  posterior history; the calibration feedback (tighten/widen prediction
  bands) is operator-routable and surfaced as the final report section.

Lane: ``lane_other_priorities_parallel_sweep_20260513``.

Usage
-----

::

    .venv/bin/python tools/log_prediction_vs_empirical_delta.py \\
        --predicted-anchors .omx/state/predicted_anchors_solver_stack_wire_in_20260513.jsonl \\
        --empirical-posterior .omx/state/continual_learning_posterior.json \\
        --out-dir reports/prediction_vs_empirical_delta \\
        [--plot]

Outputs:
- ``deltas.json`` — per-substrate delta records
- ``deltas.csv`` — flat tabular form (operator-friendly)
- ``calibration_report.json`` — aggregate stats + per-substrate tighten/widen
  recommendations
- ``deltas_plot.png`` (optional, only if matplotlib available + ``--plot``)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_NAME = "tac_prediction_vs_empirical_delta_v1"
"""Top-level schema name for the deltas.json output."""

CALIBRATION_SCHEMA_NAME = "tac_prediction_calibration_report_v1"
"""Schema name for the calibration_report.json output."""

# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRecord:
    """A single typed prediction loaded from the predicted_anchors JSONL."""

    lane_id: str
    predicted_score_band_low: float
    predicted_score_band_high: float
    predicted_archive_bytes: int | None
    predicted_seg_distortion_avg: float | None
    predicted_pose_distortion_avg: float | None
    source: str

    @property
    def predicted_midpoint(self) -> float:
        return 0.5 * (self.predicted_score_band_low + self.predicted_score_band_high)


@dataclass(frozen=True)
class EmpiricalRecord:
    """A single empirical anchor loaded from continual_learning_posterior."""

    architecture_class: str
    axis: str
    evidence_tag: str
    archive_sha256: str | None
    archive_bytes: int
    score_value: float
    hardware_substrate: str
    observed_at_utc: str


@dataclass(frozen=True)
class DeltaRow:
    """Per-substrate delta record."""

    lane_id: str
    matched_empirical_architecture_class: str
    predicted_midpoint: float
    predicted_score_band_low: float
    predicted_score_band_high: float
    empirical_score: float
    delta: float
    """``empirical - predicted_midpoint``. Positive: predicted lower than empirical."""
    band_contained_empirical: bool
    """True iff ``low <= empirical <= high``."""
    over_or_under: str
    """One of: ``over_predicted`` (predicted lower than actual; we were too
    optimistic), ``under_predicted`` (predicted higher than actual; we were
    too pessimistic), ``within_band`` (band contained actual)."""
    empirical_axis: str
    empirical_evidence_tag: str
    empirical_archive_sha256: str | None
    empirical_observed_at_utc: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_predictions(path: Path) -> list[PredictionRecord]:
    """Parse the predicted-anchors JSONL.

    Raises ``FileNotFoundError`` if the file is missing (the caller MUST handle
    or surface this; we never silently coerce missing predictions to ``[]``).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"predicted-anchors JSONL not found: {path}. "
            "Re-run the solver-stack wire-in sweep to regenerate."
        )
    out: list[PredictionRecord] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"predicted-anchors JSONL parse error at {path}:{line_no}: {exc}"
            ) from exc
        if row.get("schema") != "tac_predicted_anchor_v1":
            # Skip foreign rows but log them — never silently treat as a
            # prediction anchor.
            continue
        out.append(
            PredictionRecord(
                lane_id=row["lane_id"],
                predicted_score_band_low=float(row["predicted_score_band_low"]),
                predicted_score_band_high=float(row["predicted_score_band_high"]),
                predicted_archive_bytes=row.get("predicted_archive_bytes"),
                predicted_seg_distortion_avg=row.get("predicted_seg_distortion_avg"),
                predicted_pose_distortion_avg=row.get("predicted_pose_distortion_avg"),
                source=row.get("source", ""),
            )
        )
    return out


def load_empirical(path: Path) -> list[EmpiricalRecord]:
    """Parse the continual_learning_posterior history."""
    if not path.exists():
        raise FileNotFoundError(
            f"continual_learning_posterior not found: {path}."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "tac_continual_learning_posterior_v1":
        raise ValueError(
            f"continual_learning_posterior schema mismatch at {path}: "
            f"got {data.get('schema')!r}"
        )
    out: list[EmpiricalRecord] = []
    for row in data.get("accepted_anchor_history", []):
        out.append(
            EmpiricalRecord(
                architecture_class=row.get("architecture_class", "unknown"),
                axis=row.get("axis", "unknown"),
                evidence_tag=row.get("evidence_tag", "unknown"),
                archive_sha256=row.get("archive_sha256"),
                archive_bytes=int(row.get("archive_bytes", 0)),
                score_value=float(row.get("score_value")),
                hardware_substrate=row.get("hardware_substrate", "unknown"),
                observed_at_utc=row.get("observed_at_utc", "unknown"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Matching (prediction lane_id ↔ empirical architecture_class)
# ---------------------------------------------------------------------------


def _normalize_token(s: str) -> str:
    """Lowercase + replace separators with single underscores."""
    return s.lower().replace("-", "_").replace(" ", "_")


def match_prediction_to_empirical(
    prediction: PredictionRecord,
    empiricals: list[EmpiricalRecord],
) -> EmpiricalRecord | None:
    """Match a prediction lane_id to an empirical anchor by architecture-class
    token overlap.

    Heuristic: the prediction lane_id contains tokens like
    ``"lane_a1_plus_lapose_composition_20260513"``; the empirical anchor's
    ``architecture_class`` is a shorter form like ``"hnerv_ft_microcodec"``
    or ``"a1_host"``. We use substring containment with normalization in
    BOTH directions. If multiple empiricals match, we return the most-recent
    one (by ``observed_at_utc``).

    Returns ``None`` if no empirical anchor matches.
    """
    pred_norm = _normalize_token(prediction.lane_id)
    pred_tokens_all = set(pred_norm.split("_")) - {"lane", ""}
    # Drop common nondiscriminating suffix tokens.
    _GENERIC = {
        "prior", "model", "world", "v1", "v2", "v3", "substrate", "composition",
        "host",
    }
    candidates: list[EmpiricalRecord] = []
    for emp in empiricals:
        emp_norm = _normalize_token(emp.architecture_class)
        # Bidirectional substring match — covers both
        # `a1` ⊂ `lane_a1_plus_lapose` AND
        # `time_traveler_l5_autonomy` ⊂ `lane_time_traveler_l5_autonomy_substrate_...`.
        if emp_norm and (emp_norm in pred_norm or pred_norm.startswith(f"lane_{emp_norm}")):
            candidates.append(emp)
            continue
        # Token-overlap fallback: require >= 2 non-generic tokens of overlap
        # (or >= 1 short-token like 'a1' that uniquely names the substrate).
        emp_tokens = set(emp_norm.split("_")) - {""}
        overlap = (pred_tokens_all & emp_tokens) - _GENERIC
        if len(overlap) >= 2:
            candidates.append(emp)
            continue
        # Single-distinctive-token fallback: identifiers like 'a1' / 'l5' are
        # short but disambiguate uniquely; accept if the overlap contains a
        # token shorter than 4 chars that is not in _GENERIC.
        short_overlap = {t for t in overlap if 1 < len(t) <= 3}
        if short_overlap:
            candidates.append(emp)
    if not candidates:
        return None
    # Return the most-recent by observed_at_utc.
    candidates.sort(key=lambda e: e.observed_at_utc, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Delta computation + verdict
# ---------------------------------------------------------------------------


def compute_delta_row(
    prediction: PredictionRecord,
    empirical: EmpiricalRecord,
) -> DeltaRow:
    delta = empirical.score_value - prediction.predicted_midpoint
    contained = (
        prediction.predicted_score_band_low
        <= empirical.score_value
        <= prediction.predicted_score_band_high
    )
    if contained:
        verdict = "within_band"
    elif empirical.score_value > prediction.predicted_score_band_high:
        # Empirical worse than predicted high — we were too optimistic.
        verdict = "over_predicted"
    else:
        # Empirical better than predicted low — we were too pessimistic.
        verdict = "under_predicted"
    return DeltaRow(
        lane_id=prediction.lane_id,
        matched_empirical_architecture_class=empirical.architecture_class,
        predicted_midpoint=prediction.predicted_midpoint,
        predicted_score_band_low=prediction.predicted_score_band_low,
        predicted_score_band_high=prediction.predicted_score_band_high,
        empirical_score=empirical.score_value,
        delta=delta,
        band_contained_empirical=contained,
        over_or_under=verdict,
        empirical_axis=empirical.axis,
        empirical_evidence_tag=empirical.evidence_tag,
        empirical_archive_sha256=empirical.archive_sha256,
        empirical_observed_at_utc=empirical.observed_at_utc,
    )


def build_delta_rows(
    predictions: Iterable[PredictionRecord],
    empiricals: list[EmpiricalRecord],
) -> tuple[list[DeltaRow], list[str]]:
    """Return ``(matched_rows, unmatched_lane_ids)``."""
    matched: list[DeltaRow] = []
    unmatched: list[str] = []
    for pred in predictions:
        emp = match_prediction_to_empirical(pred, empiricals)
        if emp is None:
            unmatched.append(pred.lane_id)
            continue
        matched.append(compute_delta_row(pred, emp))
    return matched, unmatched


def build_calibration_report(rows: list[DeltaRow]) -> dict[str, Any]:
    """Aggregate per-substrate verdicts into operator-routable recommendations."""
    n = len(rows)
    n_within = sum(1 for r in rows if r.band_contained_empirical)
    n_over = sum(1 for r in rows if r.over_or_under == "over_predicted")
    n_under = sum(1 for r in rows if r.over_or_under == "under_predicted")
    abs_deltas = [abs(r.delta) for r in rows]
    mean_abs = sum(abs_deltas) / n if n else 0.0
    max_abs = max(abs_deltas) if abs_deltas else 0.0
    # Per-substrate recommendation: tighten if band was much wider than the
    # delta magnitude (we were too conservative); widen if the band missed
    # the empirical entirely.
    per_substrate_recommendations: list[dict[str, str]] = []
    for r in rows:
        band_width = r.predicted_score_band_high - r.predicted_score_band_low
        if r.band_contained_empirical and band_width > 4 * abs(r.delta) and band_width > 0.01:
            rec = "tighten_band"
            rationale = (
                f"band_width={band_width:.4f} >> 4*|delta|={4*abs(r.delta):.4f}; "
                "band is much wider than the actual error so prediction confidence "
                "can be tightened."
            )
        elif not r.band_contained_empirical:
            rec = "widen_band"
            rationale = (
                f"empirical {r.empirical_score:.4f} fell outside "
                f"[{r.predicted_score_band_low:.4f}, "
                f"{r.predicted_score_band_high:.4f}]; band must widen to cover "
                f"the {r.over_or_under} drift of {r.delta:+.4f}."
            )
        else:
            rec = "leave_as_is"
            rationale = (
                f"band well-calibrated: contained empirical {r.empirical_score:.4f} "
                f"with delta {r.delta:+.4f}, band_width {band_width:.4f}."
            )
        per_substrate_recommendations.append(
            {
                "lane_id": r.lane_id,
                "recommendation": rec,
                "rationale": rationale,
            }
        )
    return {
        "schema": CALIBRATION_SCHEMA_NAME,
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_matched_substrates": n,
        "n_within_band": n_within,
        "n_over_predicted": n_over,
        "n_under_predicted": n_under,
        "mean_abs_delta": mean_abs,
        "max_abs_delta": max_abs,
        "per_substrate_recommendations": per_substrate_recommendations,
        "evidence_grade": "[prediction-calibration; planning_only]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "notes": (
            "Per CLAUDE.md 'Apples-to-apples evidence discipline': prediction "
            "anchors stay [prediction; planning_only] and empirical anchors "
            "keep their original axis label. This calibration report informs "
            "operator-routable band tighten/widen decisions; it does NOT "
            "promote any prediction to authoritative evidence."
        ),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    rows: list[DeltaRow],
    unmatched: list[str],
    calibration: dict[str, Any],
    out_dir: Path,
    plot: bool = False,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas_json_path = out_dir / "deltas.json"
    deltas_csv_path = out_dir / "deltas.csv"
    calibration_json_path = out_dir / "calibration_report.json"

    deltas_doc = {
        "schema": SCHEMA_NAME,
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_matched": len(rows),
        "n_unmatched": len(unmatched),
        "unmatched_lane_ids": unmatched,
        "rows": [asdict(r) for r in rows],
        "evidence_grade": "[prediction; planning_only]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    deltas_json_path.write_text(json.dumps(deltas_doc, indent=2), encoding="utf-8")
    calibration_json_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    # CSV (flat tabular form for spreadsheet review).
    fieldnames = list(asdict(DeltaRow(  # static field-order reference
        lane_id="", matched_empirical_architecture_class="",
        predicted_midpoint=0.0, predicted_score_band_low=0.0,
        predicted_score_band_high=0.0, empirical_score=0.0, delta=0.0,
        band_contained_empirical=False, over_or_under="",
        empirical_axis="", empirical_evidence_tag="",
        empirical_archive_sha256=None, empirical_observed_at_utc="",
    )).keys())
    with deltas_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    out: dict[str, Path] = {
        "deltas_json": deltas_json_path,
        "deltas_csv": deltas_csv_path,
        "calibration_report": calibration_json_path,
    }
    if plot:
        plot_path = _maybe_plot(rows, out_dir)
        if plot_path is not None:
            out["plot"] = plot_path
    return out


def _maybe_plot(rows: list[DeltaRow], out_dir: Path) -> Path | None:
    """Emit a matplotlib bar chart if the library is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.4)))
    lane_ids = [r.lane_id for r in rows]
    deltas = [r.delta for r in rows]
    colors = [
        "green" if r.band_contained_empirical
        else ("red" if r.over_or_under == "over_predicted" else "orange")
        for r in rows
    ]
    y = list(range(len(rows)))
    ax.barh(y, deltas, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(lane_ids, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("delta = empirical - predicted_midpoint")
    ax.set_title("Prediction-vs-empirical delta (green: in-band; red: over-predicted; orange: under-predicted)")
    fig.tight_layout()
    plot_path = out_dir / "deltas_plot.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "delta logger",
    )
    p.add_argument(
        "--predicted-anchors",
        type=Path,
        default=Path(
            ".omx/state/predicted_anchors_solver_stack_wire_in_20260513.jsonl"
        ),
        help="Path to the predicted-anchors JSONL.",
    )
    p.add_argument(
        "--empirical-posterior",
        type=Path,
        default=Path(".omx/state/continual_learning_posterior.json"),
        help="Path to the continual_learning_posterior.json.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for deltas.json + deltas.csv + calibration_report.json",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Emit deltas_plot.png if matplotlib is available.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions = load_predictions(args.predicted_anchors)
    empiricals = load_empirical(args.empirical_posterior)
    rows, unmatched = build_delta_rows(predictions, empiricals)
    calibration = build_calibration_report(rows)
    outputs = write_outputs(
        rows, unmatched, calibration, args.out_dir, plot=args.plot
    )
    print(
        f"[delta-logger] matched={len(rows)} unmatched={len(unmatched)} "
        f"predictions={len(predictions)} empiricals={len(empiricals)}"
    )
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    if rows:
        print(
            f"[delta-logger] n_within_band={calibration['n_within_band']}/"
            f"{len(rows)} "
            f"mean_abs_delta={calibration['mean_abs_delta']:.4f} "
            f"max_abs_delta={calibration['max_abs_delta']:.4f}"
        )
    if unmatched:
        print(
            f"[delta-logger] unmatched lane_ids "
            f"(no empirical anchor yet): {unmatched[:5]}"
            + (f" ...({len(unmatched) - 5} more)" if len(unmatched) > 5 else "")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
