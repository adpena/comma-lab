#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dykstra alternating-projections solver on the solver-stack wire-in
Pareto constraints manifest.

Per the solver-stack wire-in audit (commit ``d484507f``): 'Re-run Dykstra
alternating-projections with the new Pareto constraints manifest.'

Consumes
--------

``.omx/state/pareto_constraints_solver_stack_wire_in_20260513.json``: typed
substrate constraint manifest with per-substrate (rate_bytes_max,
seg_distortion_band, pose_distortion_band, predicted_score_band) entries
relative to the PR106 r2 baseline anchor.

Mathematical formulation
------------------------

Each substrate row defines a half-space constraint in the 3-D feasible region
``(rate, seg, pose)``:

.. math::

    F_i = \\{ (r, s, p) :
              r \\le R_i \\land
              s \\le S_i^{high} \\land
              p \\le P_i^{high} \\}

The achievable region for a given substrate is the box defined by its
per-axis upper bounds. Dykstra alternating projections compute the
intersection of these per-substrate boxes — i.e., the substrates that
EVERY ONE of the candidates could simultaneously land within (a stacking
upper bound, NOT a prediction that the stack converges).

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: this output is
SOLVER OUTPUT, not score authority. The active-constraint identification
informs operator routing to substrate-priority ordering.

Output
------

``pareto_solver_output.json``: typed achievable-region polytope with
- per-axis intersection bounds (tightest constraint per axis)
- per-substrate active-vs-slack tagging (which substrates are AT each
  binding axis)
- NEWLY-ACTIVE constraints relative to the PR106 r2 baseline anchor
- per-substrate predicted feasibility verdict ("predicted band fits in the
  intersected polytope" vs "predicted band exceeds polytope on axis X")

Per CLAUDE.md "Apples-to-apples evidence discipline": every constraint stays
labeled ``[prediction; planning_only]``; output is informational and does
not promote any candidate to authoritative evidence.

Lane: ``lane_other_priorities_parallel_sweep_20260513``.

Usage
-----

::

    .venv/bin/python tools/rerun_dykstra_pareto_solver_stack_wire_in.py \\
        --pareto-constraints .omx/state/pareto_constraints_solver_stack_wire_in_20260513.json \\
        --out-dir reports/dykstra_solver_stack_wire_in_rerun_20260513
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_NAME = "tac_dykstra_pareto_solver_output_v1"


@dataclass(frozen=True)
class SubstrateConstraint:
    """A single substrate's per-axis upper bound constraints."""

    lane_id: str
    family: str
    rate_bytes_max: int | None
    rate_bytes_typical: int | None
    seg_distortion_band: tuple[float, float] | None
    pose_distortion_band: tuple[float, float] | None
    predicted_score_band: tuple[float, float] | None
    active_constraint_hypothesis: str
    composition_with: str | None
    evidence_grade: str


@dataclass(frozen=True)
class BaselineAnchor:
    """Reference operating-point anchor (e.g., PR106 r2 frontier)."""

    lane_id: str
    score: float
    seg_avg: float
    pose_avg: float
    rate_bytes: int
    axis: str
    operating_point_note: str


@dataclass(frozen=True)
class PolytopeAxis:
    """Per-axis intersection bound + the substrate(s) binding it."""

    axis_name: str
    tightest_bound: float
    binding_substrates: list[str]
    is_newly_active_vs_baseline: bool
    """True iff the intersection bound is tighter than the baseline anchor's
    value on this axis."""


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_constraints_manifest(path: Path) -> tuple[BaselineAnchor, list[SubstrateConstraint]]:
    """Parse the Pareto constraints JSON into typed records."""
    if not path.exists():
        raise FileNotFoundError(f"Pareto constraints manifest not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != "tac_pareto_constraints_v1":
        raise ValueError(
            f"Pareto constraints schema mismatch at {path}: got "
            f"{doc.get('schema')!r}"
        )
    base_doc = doc["baseline_anchor"]
    baseline = BaselineAnchor(
        lane_id=base_doc["lane_id"],
        score=float(base_doc["score"]),
        seg_avg=float(base_doc["components"]["seg_avg"]),
        pose_avg=float(base_doc["components"]["pose_avg"]),
        rate_bytes=int(base_doc["components"]["rate_bytes"]),
        axis=base_doc["axis"],
        operating_point_note=base_doc.get("operating_point_note", ""),
    )
    constraints: list[SubstrateConstraint] = []
    for row in doc.get("substrate_constraints", []):
        constraints.append(
            SubstrateConstraint(
                lane_id=row["lane_id"],
                family=row.get("family", ""),
                rate_bytes_max=row.get("rate_bytes_max"),
                rate_bytes_typical=row.get("rate_bytes_typical"),
                seg_distortion_band=(
                    tuple(row["seg_distortion_predicted_band"])
                    if row.get("seg_distortion_predicted_band")
                    else None
                ),
                pose_distortion_band=(
                    tuple(row["pose_distortion_predicted_band"])
                    if row.get("pose_distortion_predicted_band")
                    else None
                ),
                predicted_score_band=(
                    tuple(row["predicted_score_band"])
                    if row.get("predicted_score_band")
                    else None
                ),
                active_constraint_hypothesis=row.get("active_constraint_hypothesis", ""),
                composition_with=row.get("composition_with"),
                evidence_grade=row.get("evidence_grade", "[prediction]"),
            )
        )
    return baseline, constraints


# ---------------------------------------------------------------------------
# Dykstra alternating-projections (3 axes: rate, seg, pose)
# ---------------------------------------------------------------------------


def _axis_value_or_none(
    constraint: SubstrateConstraint, axis: str
) -> tuple[float | None, bool]:
    """Extract the per-substrate upper bound for ``axis``.

    Returns ``(value, applicable)``: ``applicable=False`` means the substrate
    has no constraint on this axis (treat as +infinity in the intersection).
    """
    if axis == "rate_bytes":
        return (
            (float(constraint.rate_bytes_max), True)
            if constraint.rate_bytes_max not in (None, 0)
            else (None, False)
        )
    if axis == "seg_distortion":
        return (
            (constraint.seg_distortion_band[1], True)
            if constraint.seg_distortion_band is not None
            else (None, False)
        )
    if axis == "pose_distortion":
        return (
            (constraint.pose_distortion_band[1], True)
            if constraint.pose_distortion_band is not None
            else (None, False)
        )
    raise ValueError(f"Unknown axis: {axis!r}")


def compute_intersection_polytope(
    constraints: list[SubstrateConstraint],
    baseline: BaselineAnchor,
) -> list[PolytopeAxis]:
    """Intersect per-substrate per-axis upper bounds.

    For each axis, the intersection bound is ``min_i(upper_i)`` across all
    substrates that impose a constraint on that axis. The binding substrates
    are those at the tight bound. ``is_newly_active_vs_baseline`` flags axes
    where the new intersection is TIGHTER than the baseline anchor's value.
    """
    axes = ("rate_bytes", "seg_distortion", "pose_distortion")
    baseline_values: dict[str, float] = {
        "rate_bytes": float(baseline.rate_bytes),
        "seg_distortion": baseline.seg_avg,
        "pose_distortion": baseline.pose_avg,
    }
    out: list[PolytopeAxis] = []
    for axis in axes:
        per_substrate_bounds: list[tuple[str, float]] = []
        for c in constraints:
            value, applicable = _axis_value_or_none(c, axis)
            if applicable and value is not None:
                per_substrate_bounds.append((c.lane_id, value))
        if not per_substrate_bounds:
            # No substrate constrains this axis — record the baseline as the
            # effective bound (no tightening; no binding).
            out.append(
                PolytopeAxis(
                    axis_name=axis,
                    tightest_bound=baseline_values[axis],
                    binding_substrates=[],
                    is_newly_active_vs_baseline=False,
                )
            )
            continue
        tightest = min(b for _, b in per_substrate_bounds)
        # Binding substrates: within 0.5% of the tightest bound (numerical
        # slack; for rate this is ~few bytes, for seg/pose negligible).
        tol = abs(tightest) * 0.005 if tightest != 0.0 else 1e-12
        binders = [lid for (lid, v) in per_substrate_bounds if abs(v - tightest) <= tol]
        # Newly-active iff the intersection bound is STRICTLY tighter
        # (smaller, since all bounds are upper-bounds) than the baseline.
        newly_active = tightest < baseline_values[axis]
        out.append(
            PolytopeAxis(
                axis_name=axis,
                tightest_bound=tightest,
                binding_substrates=binders,
                is_newly_active_vs_baseline=newly_active,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Per-substrate feasibility verdict
# ---------------------------------------------------------------------------


def per_substrate_feasibility(
    constraints: list[SubstrateConstraint],
    polytope: list[PolytopeAxis],
) -> list[dict[str, Any]]:
    """For each substrate, report whether its predicted band fits inside the
    intersected polytope on each axis.
    """
    poly_by_axis = {a.axis_name: a for a in polytope}
    out: list[dict[str, Any]] = []
    for c in constraints:
        verdict: dict[str, Any] = {
            "lane_id": c.lane_id,
            "family": c.family,
            "active_constraint_hypothesis": c.active_constraint_hypothesis,
        }
        infeasibilities: list[str] = []
        for axis in ("rate_bytes", "seg_distortion", "pose_distortion"):
            value, applicable = _axis_value_or_none(c, axis)
            if not applicable or value is None:
                continue
            bound = poly_by_axis[axis].tightest_bound
            # The substrate's MAX is feasible iff <= polytope bound.
            if value > bound:
                infeasibilities.append(
                    f"{axis}: substrate max {value} > polytope bound {bound}"
                )
        verdict["feasible_within_intersection"] = not infeasibilities
        verdict["infeasibility_reasons"] = infeasibilities
        out.append(verdict)
    return out


# ---------------------------------------------------------------------------
# Build full report
# ---------------------------------------------------------------------------


def build_solver_output(
    baseline: BaselineAnchor,
    constraints: list[SubstrateConstraint],
    polytope: list[PolytopeAxis],
    feasibility: list[dict[str, Any]],
) -> dict[str, Any]:
    newly_active = [a for a in polytope if a.is_newly_active_vs_baseline]
    return {
        "schema": SCHEMA_NAME,
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evidence_grade": "[prediction-derived; planning_only]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "baseline_anchor": asdict(baseline),
        "n_substrate_constraints": len(constraints),
        "achievable_region_polytope": [asdict(a) for a in polytope],
        "newly_active_constraints": [
            {
                "axis": a.axis_name,
                "tightest_bound": a.tightest_bound,
                "binding_substrates": a.binding_substrates,
                "baseline_value": (
                    {
                        "rate_bytes": baseline.rate_bytes,
                        "seg_distortion": baseline.seg_avg,
                        "pose_distortion": baseline.pose_avg,
                    }[a.axis_name]
                ),
            }
            for a in newly_active
        ],
        "per_substrate_feasibility": feasibility,
        "notes": (
            "Per CLAUDE.md 'Meta-Lagrangian/Pareto solver' non-negotiable: this "
            "output is SOLVER OUTPUT, not score authority. Active-constraint "
            "identification informs operator routing to substrate-priority "
            "ordering. Per 'Apples-to-apples evidence discipline': every "
            "constraint stays labeled [prediction; planning_only] and the "
            "report does not promote any candidate to authoritative evidence."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "dykstra solver",
    )
    p.add_argument(
        "--pareto-constraints",
        type=Path,
        default=Path(
            ".omx/state/pareto_constraints_solver_stack_wire_in_20260513.json"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline, constraints = load_constraints_manifest(args.pareto_constraints)
    polytope = compute_intersection_polytope(constraints, baseline)
    feasibility = per_substrate_feasibility(constraints, polytope)
    report = build_solver_output(baseline, constraints, polytope, feasibility)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "pareto_solver_output.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[dykstra-solver] n_substrates={len(constraints)} "
        f"baseline={baseline.lane_id} (score={baseline.score:.4f}) "
        f"newly_active_axes={len(report['newly_active_constraints'])}/3"
    )
    for axis in polytope:
        marker = "*ACTIVE*" if axis.is_newly_active_vs_baseline else "      "
        print(
            f"  {marker} axis={axis.axis_name:<18s} "
            f"tightest_bound={axis.tightest_bound:>14.6g} "
            f"binders={','.join(axis.binding_substrates[:2])}"
            + (f" +{len(axis.binding_substrates) - 2}" if len(axis.binding_substrates) > 2 else "")
        )
    n_infeasible = sum(1 for f in feasibility if not f["feasible_within_intersection"])
    print(
        f"[dykstra-solver] feasible_substrates={len(feasibility) - n_infeasible}/{len(feasibility)} "
        f"output={out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
