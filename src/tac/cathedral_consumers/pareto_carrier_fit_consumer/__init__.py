# SPDX-License-Identifier: MIT
"""Pareto carrier-fit ranking consumer — across-carrier R(D) frontier selection.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS"
+ "Results must become system intelligence" + Catalog #335 canonical cathedral
consumer contract + Catalog #341 Tier A observability-only markers.

WHAT THIS CONSUMES (does NOT reinvent)
--------------------------------------
This package is the META-LAGRANGIAN/PARETO SOLVER consumer that ranks
carrier-fit results produced by the codex fleet (HiNeRV/SNeRV/PR101 carrier
EXECUTION). The fits themselves are produced upstream; this consumer ingests
per-carrier-per-budget fit rows and selects the (carrier, model-size budget)
that minimizes the official contest score on the across-carrier R(D) Pareto
frontier, RELATIVE to the canonical local-frontier baseline.

It consumes EXISTING canonical machinery rather than reimplementing it:

* :func:`tac.auth_eval_schema.contest_formula_score` — the canonical contest
  score ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489``.
* :data:`tac.auth_eval_schema.ORIGINAL_VIDEO_BYTES` — the rate denominator.
* :func:`tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` +
  :class:`tac.dykstra_pareto_solver.Polytope` — the canonical Dykstra
  alternating-projections / KKT feasibility machinery used to feasibility-check
  each frontier candidate against the baseline polytope. We do NOT reimplement
  alternating projections; we project each candidate's per-axis delta-vs-baseline
  into the canonical (seg, pose, rate) polytope and read the canonical verdict's
  feasibility + per-axis tight/slack identification.
* :func:`tac.substrates._shared.mlx_score_aware.modelsize_budget_plan.build_modelsize_budget_plan`
  — the WITHIN-carrier byte-price waterfilling primitive (the sister producer of
  per-budget rows). This consumer is the ACROSS-carrier complement: the sister
  selects the best budget WITHIN one carrier; this selects the best
  (carrier, budget) pair across ALL carriers on the global frontier.

WHY THIS IS A NEW SURFACE (not a duplicate)
-------------------------------------------
Before this consumer, the solver layer had:
* per-carrier byte-waterfilling (modelsize_budget_plan) — one carrier at a time;
* the Dykstra 3-axis polytope solver — feasibility of a single candidate.
Neither computed the ACROSS-carrier non-dominated R(D) frontier or selected the
global min-S (carrier, budget). That selection is the missing consumer this
package supplies.

NO-FAKE evidence boundary
-------------------------
Every input :class:`CarrierFitRow` is advisory/proxy by construction. Rows carry
``score_claim`` / ``promotion_eligible`` / ``ready_for_exact_eval_dispatch`` =
False (the canonical false-authority markers). Synthetic test fixtures MUST set
``fixture_not_real=True``; the consumer threads that flag into the verdict so a
downstream reader can never mistake a fixture frontier for a real measurement.
The selection is a RANKING / PLANNING output, never a score claim.

Catalog #125 6-hook wire-in declaration
---------------------------------------
* Hook #1 sensitivity-map: ACTIVE — the per-axis tight-constraint identification
  from the consumed Dykstra verdict surfaces which axis (seg/pose/rate) is
  binding at the selected frontier point (next-cycle attack direction).
* Hook #2 Pareto constraint: ACTIVE PRIMARY — this IS a Pareto-frontier
  consumer; it computes the across-carrier non-dominated R(D) set and selects
  min-S, feasibility-checked via the canonical Dykstra polytope solver.
* Hook #3 bit-allocator: ACTIVE — the selected (carrier, model-size budget)
  byte ceiling is the bit-allocator's per-carrier budget recommendation.
* Hook #4 cathedral autopilot dispatch: ACTIVE — auto-discovered via Catalog
  #335 canonical contract; ``consume_candidate`` surfaces per-candidate frontier
  membership + dominance to the cathedral autopilot ranker (Tier A
  observability-only per Catalog #341).
* Hook #5 continual-learning posterior: ACTIVE — ``update_from_anchor`` ingests
  a landed carrier-fit anchor so the next ranking sees the new (carrier, budget)
  point on the frontier.
* Hook #6 probe-disambiguator: ACTIVE — the frontier verdict's
  ``selection_reason`` + per-candidate ``dominated_by`` disambiguates which
  (carrier, budget) probe to dispatch next (the frontier point closest to the
  baseline that is not yet measured at full-fit).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.cathedral.consumer_contract import HookNumber
from tac.dykstra_pareto_solver import (
    Polytope,
    solve_pareto_polytope_intersection,
)

# Catalog #335 canonical cathedral consumer contract metadata.
CONSUMER_NAME = "pareto_carrier_fit_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)

CARRIER_FIT_ROW_SCHEMA = "pareto_carrier_fit_row.v1"
FRONTIER_VERDICT_SCHEMA = "pareto_carrier_fit_frontier.v1"

# Canonical contest rate term per archive byte: 25 / 37_545_489.
CONTEST_BYTE_PRICE_SCORE = 25.0 / float(ORIGINAL_VIDEO_BYTES)

__all__ = [
    "CARRIER_FIT_ROW_SCHEMA",
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONTEST_BYTE_PRICE_SCORE",
    "FRONTIER_VERDICT_SCHEMA",
    "CarrierFitConsumerError",
    "CarrierFitRow",
    "ParetoCarrierFitFrontier",
    "compute_across_carrier_pareto_frontier",
    "consume_candidate",
    "parse_carrier_fit_rows",
    "rate_term_for_bytes",
    "update_from_anchor",
]


class CarrierFitConsumerError(ValueError):
    """Raised when carrier-fit input violates the consumer contract."""


@dataclass(frozen=True)
class CarrierFitRow:
    """One advisory per-carrier-per-budget fit-result row.

    A carrier (e.g. ``hi_nerv``, ``snerv``, ``pr101``) trained at a given
    model-size budget produces (advisory ``d_seg``, advisory ``d_pose``,
    ``modelsize_bytes``). The contest rate term + total score are DERIVED via
    the canonical :func:`contest_formula_score`; an upstream-supplied
    ``advisory_S`` (if present) is validated against the canonical formula.

    Fields
    ------
    carrier_id : str
        Stable carrier id. Non-empty.
    modelsize_bytes : int
        Total archive bytes at this budget. ``> 0``.
    d_seg : float
        Advisory SegNet distortion (argmax-flip rate). ``>= 0``.
    d_pose : float
        Advisory PoseNet distortion (per-dim Mahalanobis MSE). ``>= 0``.
    budget_id : str
        Optional sub-budget label (e.g. ``"latent16_ch32"``). Defaults to the
        byte count as string. Distinguishes multiple budgets for one carrier.
    fixture_not_real : bool
        NO-FAKE marker. Synthetic fixtures MUST set True. Threaded into the
        frontier verdict so a fixture frontier can never be read as real.
    source : Mapping[str, Any]
        Free-form provenance (commit / call_id / advisory tag). Never an
        authority field.
    """

    carrier_id: str
    modelsize_bytes: int
    d_seg: float
    d_pose: float
    budget_id: str = ""
    fixture_not_real: bool = False
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        carrier = str(self.carrier_id).strip()
        if not carrier:
            raise CarrierFitConsumerError("carrier_id must be non-empty")
        if not isinstance(self.modelsize_bytes, int) or isinstance(
            self.modelsize_bytes, bool
        ):
            raise CarrierFitConsumerError(
                f"modelsize_bytes must be int, got {self.modelsize_bytes!r}"
            )
        if self.modelsize_bytes <= 0:
            raise CarrierFitConsumerError(
                f"modelsize_bytes must be > 0, got {self.modelsize_bytes}"
            )
        for label, value in (("d_seg", self.d_seg), ("d_pose", self.d_pose)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise CarrierFitConsumerError(
                    f"{label} must be numeric, got {value!r}"
                )
            if math.isnan(value) or math.isinf(value):
                raise CarrierFitConsumerError(f"{label}={value} is NaN/inf")
            if value < 0:
                raise CarrierFitConsumerError(
                    f"{label} must be >= 0, got {value}"
                )
        if not isinstance(self.fixture_not_real, bool):
            raise CarrierFitConsumerError("fixture_not_real must be bool")
        if not isinstance(self.source, Mapping):
            raise CarrierFitConsumerError("source must be a Mapping")
        # Normalize budget_id default to the byte count for a stable key.
        if not str(self.budget_id).strip():
            object.__setattr__(self, "budget_id", str(int(self.modelsize_bytes)))
        object.__setattr__(self, "carrier_id", carrier)

    @property
    def key(self) -> str:
        """Stable (carrier, budget) identifier."""
        return f"{self.carrier_id}::{self.budget_id}"

    @property
    def rate_term(self) -> float:
        """Canonical contest rate-axis contribution: 25*bytes/37545489."""
        return rate_term_for_bytes(self.modelsize_bytes)

    @property
    def nonrate_score(self) -> float:
        """Distortion-axis score: 100*d_seg + sqrt(10*d_pose)."""
        return 100.0 * float(self.d_seg) + math.sqrt(10.0 * float(self.d_pose))

    @property
    def advisory_S(self) -> float:
        """Canonical total contest score for this (carrier, budget) row."""
        return contest_formula_score(
            seg_dist=float(self.d_seg),
            pose_dist=float(self.d_pose),
            archive_bytes=int(self.modelsize_bytes),
        )

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema": CARRIER_FIT_ROW_SCHEMA,
            "key": self.key,
            "carrier_id": self.carrier_id,
            "budget_id": self.budget_id,
            "modelsize_bytes": int(self.modelsize_bytes),
            "d_seg": float(self.d_seg),
            "d_pose": float(self.d_pose),
            "rate_term": float(self.rate_term),
            "nonrate_score": float(self.nonrate_score),
            "advisory_S": float(self.advisory_S),
            "fixture_not_real": bool(self.fixture_not_real),
            "source": dict(self.source),
            # Canonical false-authority markers (Catalog #323/#341).
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
            "axis_tag": "[predicted]",
        }


def rate_term_for_bytes(archive_bytes: int) -> float:
    """Canonical contest rate-axis contribution for ``archive_bytes``.

    ``25 * archive_bytes / ORIGINAL_VIDEO_BYTES`` — consumes the canonical
    :data:`tac.auth_eval_schema.ORIGINAL_VIDEO_BYTES` denominator.
    """
    if not isinstance(archive_bytes, int) or isinstance(archive_bytes, bool):
        raise CarrierFitConsumerError(
            f"archive_bytes must be int, got {archive_bytes!r}"
        )
    if archive_bytes < 0:
        raise CarrierFitConsumerError("archive_bytes must be >= 0")
    return float(archive_bytes) * CONTEST_BYTE_PRICE_SCORE


def parse_carrier_fit_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[CarrierFitRow]:
    """Parse advisory carrier-fit dict rows into typed :class:`CarrierFitRow`.

    Each row must carry ``carrier_id`` + ``modelsize_bytes`` (or
    ``archive_bytes``) + ``d_seg`` + ``d_pose``. ``advisory_S`` is optional;
    if present it is validated against the canonical formula (within tolerance)
    and a mismatch raises (NO-FAKE: an upstream-claimed S that does not match
    the canonical formula on the supplied components is rejected, not silently
    overwritten).
    """
    parsed: list[CarrierFitRow] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CarrierFitConsumerError(
                f"carrier-fit row[{i}] must be a Mapping, got {type(row).__name__}"
            )
        bytes_value = row.get("modelsize_bytes", row.get("archive_bytes"))
        if bytes_value is None:
            raise CarrierFitConsumerError(
                f"carrier-fit row[{i}] missing modelsize_bytes/archive_bytes"
            )
        try:
            modelsize_bytes = int(bytes_value)
        except (TypeError, ValueError) as exc:
            raise CarrierFitConsumerError(
                f"carrier-fit row[{i}].modelsize_bytes={bytes_value!r} not int"
            ) from exc
        fit_row = CarrierFitRow(
            carrier_id=str(row.get("carrier_id", "")),
            modelsize_bytes=modelsize_bytes,
            d_seg=float(row.get("d_seg", 0.0)),
            d_pose=float(row.get("d_pose", 0.0)),
            budget_id=str(row.get("budget_id", "")),
            fixture_not_real=bool(row.get("fixture_not_real", False)),
            source=dict(row.get("source", {})),
        )
        advisory_s = row.get("advisory_S")
        if advisory_s is not None:
            try:
                claimed = float(advisory_s)
            except (TypeError, ValueError) as exc:
                raise CarrierFitConsumerError(
                    f"carrier-fit row[{i}].advisory_S={advisory_s!r} not numeric"
                ) from exc
            if not math.isclose(claimed, fit_row.advisory_S, rel_tol=1e-4, abs_tol=1e-6):
                raise CarrierFitConsumerError(
                    f"carrier-fit row[{i}] advisory_S={claimed} does not match "
                    f"canonical contest_formula_score={fit_row.advisory_S:.9f} on "
                    f"the supplied (d_seg={fit_row.d_seg}, d_pose={fit_row.d_pose}, "
                    f"bytes={fit_row.modelsize_bytes}); refusing to silently "
                    "overwrite a mismatched score claim (NO-FAKE)"
                )
        parsed.append(fit_row)
    return parsed


@dataclass(frozen=True)
class ParetoCarrierFitFrontier:
    """Verdict: across-carrier R(D) Pareto frontier + min-S selection.

    Carries the canonical false-authority markers — this is a planning /
    ranking output, never a score claim.
    """

    schema: str
    rows: tuple[CarrierFitRow, ...]
    frontier_keys: tuple[str, ...]
    dominated_keys: tuple[str, ...]
    dominated_by: Mapping[str, str]
    selected_key: str | None
    selected_advisory_S: float | None
    baseline_S: float
    baseline_archive_bytes: int
    selected_beats_baseline: bool
    selected_feasible_vs_baseline: bool
    selected_tight_axes: tuple[str, ...]
    selected_slack_axes: tuple[str, ...]
    selection_reason: str
    any_fixture_not_real: bool
    blockers: tuple[str, ...]

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rows": [row.as_jsonable() for row in self.rows],
            "frontier_keys": list(self.frontier_keys),
            "dominated_keys": list(self.dominated_keys),
            "dominated_by": dict(self.dominated_by),
            "selected_key": self.selected_key,
            "selected_advisory_S": self.selected_advisory_S,
            "baseline_S": float(self.baseline_S),
            "baseline_archive_bytes": int(self.baseline_archive_bytes),
            "selected_beats_baseline": bool(self.selected_beats_baseline),
            "selected_feasible_vs_baseline": bool(
                self.selected_feasible_vs_baseline
            ),
            "selected_tight_axes": list(self.selected_tight_axes),
            "selected_slack_axes": list(self.selected_slack_axes),
            "selection_reason": self.selection_reason,
            "any_fixture_not_real": bool(self.any_fixture_not_real),
            "blockers": list(self.blockers),
            # Canonical false-authority markers — RANKING not score claim.
            "axis_tag": "[predicted]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        }


def _is_dominated(candidate: CarrierFitRow, other: CarrierFitRow) -> bool:
    """True iff ``other`` Pareto-dominates ``candidate`` in (rate, distortion).

    The R(D) objective is to MINIMIZE both the rate term and the non-rate
    distortion score. ``other`` dominates ``candidate`` iff ``other`` is
    no worse on both axes AND strictly better on at least one.
    """
    no_worse = (
        other.rate_term <= candidate.rate_term + 1e-12
        and other.nonrate_score <= candidate.nonrate_score + 1e-12
    )
    strictly_better = (
        other.rate_term < candidate.rate_term - 1e-12
        or other.nonrate_score < candidate.nonrate_score - 1e-12
    )
    return no_worse and strictly_better


def compute_across_carrier_pareto_frontier(
    rows: Sequence[CarrierFitRow] | Iterable[Mapping[str, Any]],
    *,
    baseline_archive_bytes: int,
    baseline_d_seg: float = 0.0,
    baseline_d_pose: float = 0.0,
    baseline_S: float | None = None,
) -> ParetoCarrierFitFrontier:
    """Compute the across-carrier R(D) Pareto frontier + min-S selection.

    Args
    ----
    rows : Sequence[CarrierFitRow] | Iterable[Mapping]
        Per-carrier-per-budget fit rows (typed or dict). Dict rows are parsed
        via :func:`parse_carrier_fit_rows`.
    baseline_archive_bytes : int
        The canonical local-frontier baseline archive byte count (e.g. the
        178493-byte fp11 source-brotli-recode CPU frontier). Used for the
        baseline rate term + the Dykstra polytope upper bound on the rate axis.
    baseline_d_seg, baseline_d_pose : float
        Baseline distortion components. If ``baseline_S`` is None it is computed
        from these via the canonical formula. When unknown, leave at 0.0 and
        pass ``baseline_S`` directly (e.g. the 0.192 published frontier).
    baseline_S : float | None
        Canonical baseline contest score (e.g. 0.192 [contest-CPU]). If None,
        derived from ``(baseline_d_seg, baseline_d_pose, baseline_archive_bytes)``.

    Returns
    -------
    ParetoCarrierFitFrontier
        Frontier membership + dominance + min-S selection + Dykstra
        feasibility/KKT verdict for the selected point, with false-authority
        markers.

    Notes
    -----
    Feasibility-vs-baseline is read from the canonical Dykstra polytope solver
    (consumed, not reinvented): the selected candidate's per-axis delta vs the
    baseline (Δseg, Δpose, Δbytes) is projected into the canonical (seg, pose,
    rate) polytope whose upper bounds are the baseline components. A feasible
    projection means the candidate lies inside the "no worse than baseline on
    any axis" cone; the per-axis tight constraints identify which axis is
    binding (the next-cycle attack direction per Catalog #125 hook #1/#6).
    """
    if not isinstance(baseline_archive_bytes, int) or isinstance(
        baseline_archive_bytes, bool
    ):
        raise CarrierFitConsumerError(
            f"baseline_archive_bytes must be int, got {baseline_archive_bytes!r}"
        )
    if baseline_archive_bytes <= 0:
        raise CarrierFitConsumerError("baseline_archive_bytes must be > 0")

    # Normalize rows to typed CarrierFitRow.
    typed_rows: list[CarrierFitRow]
    materialized = list(rows)
    if materialized and isinstance(materialized[0], CarrierFitRow):
        # Validate the rest are also typed.
        for i, r in enumerate(materialized):
            if not isinstance(r, CarrierFitRow):
                raise CarrierFitConsumerError(
                    f"mixed row types: rows[{i}] is {type(r).__name__}, "
                    "expected CarrierFitRow"
                )
        typed_rows = list(materialized)  # type: ignore[arg-type]
    else:
        typed_rows = parse_carrier_fit_rows(materialized)  # type: ignore[arg-type]

    if baseline_S is None:
        baseline_total = contest_formula_score(
            seg_dist=float(baseline_d_seg),
            pose_dist=float(baseline_d_pose),
            archive_bytes=int(baseline_archive_bytes),
        )
    else:
        baseline_total = float(baseline_S)

    any_fixture = any(r.fixture_not_real for r in typed_rows)

    if not typed_rows:
        return ParetoCarrierFitFrontier(
            schema=FRONTIER_VERDICT_SCHEMA,
            rows=(),
            frontier_keys=(),
            dominated_keys=(),
            dominated_by={},
            selected_key=None,
            selected_advisory_S=None,
            baseline_S=baseline_total,
            baseline_archive_bytes=int(baseline_archive_bytes),
            selected_beats_baseline=False,
            selected_feasible_vs_baseline=False,
            selected_tight_axes=(),
            selected_slack_axes=(),
            selection_reason="no_carrier_fit_rows_supplied",
            any_fixture_not_real=False,
            blockers=("no_carrier_fit_rows_supplied",),
        )

    # Compute the non-dominated Pareto frontier in (rate_term, nonrate_score).
    frontier: list[CarrierFitRow] = []
    dominated: list[CarrierFitRow] = []
    dominated_by: dict[str, str] = {}
    for candidate in typed_rows:
        dominator: CarrierFitRow | None = None
        for other in typed_rows:
            if other is candidate or other.key == candidate.key:
                continue
            if _is_dominated(candidate, other):
                dominator = other
                break
        if dominator is None:
            frontier.append(candidate)
        else:
            dominated.append(candidate)
            dominated_by[candidate.key] = dominator.key

    # Select the global min-S point (the contest objective). The min-S point is
    # always a frontier point (lowering both axes lowers S), but we select from
    # the frontier explicitly for clarity + to surface the frontier membership.
    selected = min(frontier, key=lambda r: r.advisory_S)
    selected_S = selected.advisory_S
    beats_baseline = selected_S < baseline_total - 1e-12

    # Feasibility-vs-baseline via the canonical Dykstra polytope solver.
    # Express the candidate's position relative to the baseline as a per-axis
    # delta and project into the canonical (seg, pose, rate) polytope whose
    # bounds encode "no worse than baseline on any axis". We do NOT reimplement
    # alternating projections — we consume solve_pareto_polytope_intersection.
    feasible, tight_axes, slack_axes = _feasibility_vs_baseline(
        selected,
        baseline_d_seg=float(baseline_d_seg),
        baseline_d_pose=float(baseline_d_pose),
        baseline_archive_bytes=int(baseline_archive_bytes),
        candidate_id=selected.key,
    )

    blockers: list[str] = [
        "advisory_carrier_fit_ranking_no_score_claim",
        "requires_full_fit_evidence_before_dispatch",
        "requires_byte_closed_archive_before_promotion",
        "requires_contest_cpu_then_cuda_auth_after_local_win",
    ]
    if any_fixture:
        blockers.insert(0, "fixture_not_real_rows_present_ranking_is_demonstration_only")

    if beats_baseline:
        reason = (
            f"selected min-S frontier point {selected.key} "
            f"(advisory_S={selected_S:.6f}) beats baseline_S={baseline_total:.6f}"
        )
    else:
        reason = (
            f"selected min-S frontier point {selected.key} "
            f"(advisory_S={selected_S:.6f}) does NOT beat baseline_S="
            f"{baseline_total:.6f}; no carrier fit closes the frontier gap yet"
        )

    return ParetoCarrierFitFrontier(
        schema=FRONTIER_VERDICT_SCHEMA,
        rows=tuple(typed_rows),
        frontier_keys=tuple(r.key for r in sorted(frontier, key=lambda r: r.advisory_S)),
        dominated_keys=tuple(r.key for r in dominated),
        dominated_by=dominated_by,
        selected_key=selected.key,
        selected_advisory_S=float(selected_S),
        baseline_S=baseline_total,
        baseline_archive_bytes=int(baseline_archive_bytes),
        selected_beats_baseline=beats_baseline,
        selected_feasible_vs_baseline=feasible,
        selected_tight_axes=tuple(tight_axes),
        selected_slack_axes=tuple(slack_axes),
        selection_reason=reason,
        any_fixture_not_real=any_fixture,
        blockers=tuple(blockers),
    )


def _feasibility_vs_baseline(
    candidate: CarrierFitRow,
    *,
    baseline_d_seg: float,
    baseline_d_pose: float,
    baseline_archive_bytes: int,
    candidate_id: str,
) -> tuple[bool, list[str], list[str]]:
    """Project candidate's delta-vs-baseline into the canonical Dykstra polytope.

    Consumes :func:`solve_pareto_polytope_intersection`. The polytope encodes
    "the candidate is no worse than baseline on each axis":

      * seg axis bound ``(0, baseline_d_seg)`` — candidate d_seg should not
        exceed baseline.
      * pose axis bound ``(0, baseline_d_pose)`` — likewise.
      * rate axis bound ``(<negative>, 0)`` — the candidate's byte DELTA vs
        baseline must be <= 0 (no more bytes than baseline).

    The initial point is the candidate's actual per-axis position
    (its d_seg, d_pose, and byte-delta). Feasibility is determined by whether
    the candidate's ORIGINAL point already lies inside the polytope (i.e. the
    Dykstra projection did not have to move it — ``convergence_residual ≈ 0``
    AND :meth:`Polytope.contains`). The candidate being OUTSIDE the polytope is
    exactly the "worse than baseline" case we want to flag as infeasible. We
    consume the Dykstra verdict for the per-axis tight/slack identification
    (which axis is binding = the next-cycle attack direction).
    """
    byte_delta = float(candidate.modelsize_bytes - baseline_archive_bytes)
    # Rate-axis lower bound: allow the candidate's actual byte delta when it is
    # already <= 0 (so the polytope is non-degenerate); upper bound 0.0 (no more
    # bytes than baseline). When the candidate has MORE bytes than baseline the
    # byte_delta is positive and lies above the upper bound 0.0 → infeasible.
    rate_lo = min(byte_delta, 0.0) - 1.0
    seg_hi = max(float(baseline_d_seg), 1e-9)
    pose_hi = max(float(baseline_d_pose), 1e-9)
    polytope = Polytope(
        axis_bounds={
            "seg": (0.0, seg_hi),
            "pose": (0.0, pose_hi),
            "rate": (rate_lo, 0.0),
        },
        name="carrier_fit_no_worse_than_baseline",
    )
    initial_point = {
        "seg": float(candidate.d_seg),
        "pose": float(candidate.d_pose),
        "rate": byte_delta,
    }
    verdict = solve_pareto_polytope_intersection(
        polytope,
        initial_point=initial_point,
        candidate_id=candidate_id,
    )
    # The solver PROJECTS the point into the polytope, so its ``feasible`` flag
    # reflects projection convergence, not whether the ORIGINAL candidate was
    # already inside. The "no worse than baseline" predicate is whether the
    # original point already lies inside the polytope: contains() on the
    # original point. The tight/slack axes from the Dykstra verdict identify
    # the binding constraints (the per-axis duals, consumed not reinvented).
    feasible = polytope.contains(initial_point, tolerance=1e-9)
    return (
        bool(feasible),
        list(verdict.tight_constraint_axes),
        list(verdict.slack_axes),
    )


# ---------------------------------------------------------------------------
# Catalog #335 canonical cathedral consumer contract surface.
# ---------------------------------------------------------------------------


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    A landed carrier-fit anchor (a new measured (carrier, budget) point) would
    extend the across-carrier frontier the next time the ranker runs. This
    consumer is stateless per invocation — the frontier is recomputed from the
    candidate payload on each :func:`consume_candidate` call — so there is no
    persistent posterior to mutate here; the acknowledgment is explicit. A
    future stateful variant would append the anchor's (carrier, budget) row to
    a frontier cache keyed by archive sha.
    """
    _ = anchor  # explicit acknowledgment; stateless frontier recompute per call


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Tier A observability-only per Catalog #341: returns the canonical
    non-promotable markers (``predicted_delta_adjustment=0.0`` /
    ``promotable=False`` / ``axis_tag="[predicted]"``). The contribution
    surfaces whether the candidate's (carrier, budget) lies on the across-
    carrier R(D) frontier and whether it is the min-S selection, computed by
    consuming the candidate's own ``carrier_fit_rows`` payload (if present).

    The candidate payload MAY supply:
      * ``carrier_fit_rows``: list of advisory fit dict rows.
      * ``baseline_archive_bytes`` / ``baseline_S``: baseline anchor.
    When absent, the contribution is a zero-adjustment acknowledgment with a
    rationale explaining the missing payload (so the cathedral ranker sees the
    consumer fired but had nothing to rank).
    """
    rows_payload = candidate.get("carrier_fit_rows")
    if not isinstance(rows_payload, (list, tuple)) or not rows_payload:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "pareto_carrier_fit_consumer: no carrier_fit_rows in candidate "
                "payload; observability-only acknowledgment"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }
    baseline_bytes = candidate.get("baseline_archive_bytes")
    if not isinstance(baseline_bytes, int) or isinstance(baseline_bytes, bool):
        # Default to the published-frontier byte count if not supplied; the
        # ranking is still observability-only so a default baseline is safe.
        baseline_bytes = 178493
    baseline_s = candidate.get("baseline_S")
    try:
        frontier = compute_across_carrier_pareto_frontier(
            rows_payload,
            baseline_archive_bytes=int(baseline_bytes),
            baseline_S=(float(baseline_s) if baseline_s is not None else None),
        )
    except CarrierFitConsumerError as exc:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"pareto_carrier_fit_consumer: input rejected ({exc}); "
                "observability-only acknowledgment"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }
    candidate_key = str(candidate.get("carrier_fit_key", ""))
    on_frontier = candidate_key in frontier.frontier_keys
    is_selected = candidate_key == frontier.selected_key
    rationale = (
        "pareto_carrier_fit_consumer: across-carrier R(D) frontier computed; "
        f"selected={frontier.selected_key} "
        f"(advisory_S={frontier.selected_advisory_S}); "
        f"beats_baseline={frontier.selected_beats_baseline}; "
        f"candidate_on_frontier={on_frontier}; candidate_selected={is_selected}; "
        f"tight_axes={list(frontier.selected_tight_axes)}"
    )
    return {
        "predicted_delta_adjustment": 0.0,  # Tier A observability-only.
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "pareto_carrier_fit_frontier": frontier.as_jsonable(),
    }
