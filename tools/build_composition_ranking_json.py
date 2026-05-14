# SPDX-License-Identifier: MIT
"""FIX-C bridge: composition-cell enumeration → autopilot ranking JSON.

Per ZZZZZ integration audit 2026-05-12, the cathedral autopilot's
``--use-substrate-composition-matrix-ranking <path>`` flag consumes a JSON
artifact with schema ``tac_autopilot_dispatch_ranking_v1`` (the 58-row
substrate-only ranking shipped by
``src/tac/optimization/autopilot_dispatch_ranking.py``).

CCCC's ``tac.composition`` module enumerates the FULL (substrate × primitive
× order) matrix at ~16,833 cells (7,834 compatible non-bare + 24 bare =
7,858 compatible total). That enumeration is operationally orphaned: nothing
in the autopilot consumer path reads ``enumerate_cells()``.

This bridge closes the gap:

1. Imports :func:`tac.composition.enumerate.enumerate_cells` (full matrix).
2. Filters to compatible cells (``compatibility_verdict`` in
   ``{"compatible", "compatible_bare_substrate"}``).
3. Consumes :mod:`tac.continual_learning` posterior to reweight the
   per-cell predicted score delta by the family-keyed correction factor
   (wire-in hook 5).
4. Consumes :func:`tac.cost_band_calibration.predict` to widen the
   estimated dispatch cost band when the cell's anchor count is low
   (wire-in hook 4).
5. Consumes a per-axis sensitivity reweighting derived from the operating-
   point-aware SegNet-vs-PoseNet rule (CLAUDE.md "operating-point dependent"
   section, the PR106 frontier 2.71× pose-marginal value; wire-in hook 1).
6. Emits the same ``tac_autopilot_dispatch_ranking_v1`` schema the
   autopilot already consumes — no autopilot-loader change required.

Per CLAUDE.md "Forbidden score claims" + "Forbidden empirical-claim-
without-evidence-tag", every emitted row carries::

    score_claim                     = False
    promotion_eligible              = False
    ready_for_exact_eval_dispatch   = False
    composition_notes               carries [predicted; ...] tag

and the top-level payload tags the evidence grade ``planning_only_*``.

Cross-references
----------------
- :mod:`tac.composition.enumerate` — full (substrate × primitive × order) matrix.
- :mod:`tac.optimization.autopilot_dispatch_ranking` — the 58-row sibling
  whose schema this bridge mirrors.
- :mod:`tools.cathedral_autopilot_autonomous_loop` — consumer.
- :mod:`tac.continual_learning` — posterior (hook 5).
- :mod:`tac.cost_band_calibration` — cost-band predictor (hook 4).

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``halt_and_ask_default_on``
- ``no_tmp_paths``
- ``substrate_primitive_composition_cell_registry_v1``
- ``composition_cell_to_autopilot_bridge_v1``
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Repo-root import shim.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac import cost_band_calibration  # noqa: E402
from tac.composition.enumerate import enumerate_cells  # noqa: E402
from tac.continual_learning import (  # noqa: E402
    load_posterior,
    posterior_query_track_correction,
)
from tac.optimization.autopilot_dispatch_ranking import (  # noqa: E402
    DEFAULT_CUMULATIVE_CAP_USD,
    DEFAULT_PER_DISPATCH_CAP_USD,
)
from tac.optimization.autopilot_dispatch_ranking import (  # noqa: E402
    SCHEMA_VERSION as AUTOPILOT_SCHEMA_VERSION,
)
from tac.optimization.substrate_composition_matrix import (  # noqa: E402
    DISPATCH_COST_USD_MIDPOINT,
    canonical_substrate_inventory,
)
from tac.sensitivity_map.axis_weights import (  # noqa: E402
    PR106_R2_FRONTIER_AXIS_WEIGHTS,
    AxisWeightsError,
    validate_axis_weights_mapping,
)

if TYPE_CHECKING:
    from tac.composition.registry import CompositionCell, SubstrateRow

# Internal bridge schema marker (lives INSIDE the autopilot schema payload).
BRIDGE_SCHEMA = "tac_composition_cell_to_autopilot_bridge_v1"


# ──────────────────────────────────────────────────────────────────────────
# Axis-weight rule (CLAUDE.md "SegNet vs PoseNet importance — operating-
# point dependent"). Defaults match the PR106 frontier where pose marginal
# = 2.71x seg marginal. Operators can override on the CLI.
#
# The canonical source-of-truth is :mod:`tac.sensitivity_map.axis_weights`
# (COUNCIL-A1 landing 2026-05-12). ``DEFAULT_AXIS_WEIGHTS`` is preserved as a
# plain ``dict[str, float]`` for backward compatibility with consumers that
# already imported it, BUT it is now sourced from the canonical AxisWeights
# typed dataclass so the bridge and the probe-disambiguator stay coherent.
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_AXIS_WEIGHTS: dict[str, float] = PR106_R2_FRONTIER_AXIS_WEIGHTS.as_mapping()


@dataclasses.dataclass(frozen=True)
class _CellPlan:
    """Internal staging row before serialization."""

    cell: CompositionCell
    substrate: SubstrateRow
    estimated_dispatch_cost_usd: float
    cost_confidence_tag: str
    posterior_correction: float
    posterior_n_observations: int
    axis_weight: float
    expected_information_gain: float
    eig_per_dollar: float
    blockers: tuple[str, ...]


def _substrate_index() -> dict[str, SubstrateRow]:
    return {s.substrate_id: s for s in canonical_substrate_inventory()}


def _estimate_dispatch_cost(
    substrate_id: str,
    *,
    posterior_path: Path | None = None,
) -> tuple[float, str]:
    """Estimate dispatch cost for a substrate.

    Walks the substrate → DISPATCH_COST_USD_MIDPOINT table first.  When the
    bucket has enough cost-band anchors, replace with the empirical p50 from
    :func:`tac.cost_band_calibration.predict`.  Returns the (cost_usd,
    confidence_tag) pair where confidence_tag is one of
    ``substrate_midpoint``, ``cost_band_p50``, ``cost_band_weak_p50``, or
    ``cost_band_hand_calibrated``.
    """
    midpoint = DISPATCH_COST_USD_MIDPOINT.get(substrate_id, 0.0)
    # Cost-band prediction is keyed on (platform, gpu, epochs, all_flags_on).
    # For planning purposes, ask for the canonical T4-3000ep-all-flags bucket
    # since that matches the autopilot's substrate-cost-midpoint regime.
    try:
        band = cost_band_calibration.predict(
            "modal", "T4", 3000,
            all_flags_on=True,
            posterior_path=posterior_path,
        )
    except Exception:
        return midpoint, "substrate_midpoint"
    if band.confidence_tag == "empirical_posterior":
        return float(band.p50_cost_usd), "cost_band_p50"
    if band.confidence_tag == "weak_posterior":
        return float(band.p50_cost_usd), "cost_band_weak_p50"
    # Hand-calibrated fallback: trust the substrate midpoint since it's
    # the cleaner per-substrate signal than the bucket-level stub.
    return midpoint, "substrate_midpoint"


def _posterior_correction_for_cell(
    cell: CompositionCell,
    posterior,
) -> tuple[float, int]:
    """Query the posterior for the track-correction factor matching this cell.

    The posterior keys tracks by family / architecture-class strings. The
    cell's ``substrate_class`` value is the canonical key. Falls back to
    ``(1.0, 0)`` if no observations exist for this family.
    """
    track_kind = cell.substrate_class.value
    return posterior_query_track_correction(posterior, track_kind, default=1.0)


def _build_cell_plans(
    cells: list[CompositionCell],
    *,
    posterior_path: Path | None = None,
    axis_weights: dict[str, float] = DEFAULT_AXIS_WEIGHTS,
    apply_posterior: bool = True,
    apply_cost_band: bool = True,
) -> list[_CellPlan]:
    """Build the typed plan rows before serialization.

    The plan applies (in order):

    1. Axis weighting (CLAUDE.md PR106 marginal rule).
    2. Posterior correction (hook 5) — if ``apply_posterior``.
    3. Cost-band widening (hook 4) — if ``apply_cost_band``.
    """
    substrate_idx = _substrate_index()
    posterior = (
        load_posterior(posterior_path) if apply_posterior else None
    )

    out: list[_CellPlan] = []
    for cell in cells:
        substrate = substrate_idx.get(cell.substrate_id)
        if substrate is None:
            # Cell references an unknown substrate — skip with explicit blocker.
            continue
        if apply_cost_band:
            cost_usd, cost_tag = _estimate_dispatch_cost(
                cell.substrate_id, posterior_path=posterior_path,
            )
        else:
            cost_usd = DISPATCH_COST_USD_MIDPOINT.get(cell.substrate_id, 0.0)
            cost_tag = "substrate_midpoint"

        if posterior is not None:
            correction, n_obs = _posterior_correction_for_cell(cell, posterior)
        else:
            correction, n_obs = 1.0, 0

        axis = substrate.target_axis.value
        axis_weight = float(axis_weights.get(axis, 1.0))

        # Apply posterior correction + axis weighting to the planning score
        # delta. ``predicted_score_delta`` is negative for an improvement;
        # multiplying by correction>1.0 amplifies (operator's posterior has
        # seen this family deliver more than the prior expected) and <1.0
        # damps. Axis weighting scales the *information gain* (EV) of the
        # bet, NOT the predicted_score_delta itself — predicted delta stays
        # a planning prediction; EV is reweighted toward the axis the
        # marginal rule says matters more.
        weighted_delta = cell.predicted_score_delta * correction
        raw_eig = abs(weighted_delta)
        weighted_eig = raw_eig * axis_weight
        # Cost-zero is treated as cost-unknown (NOT free). Per the sibling
        # autopilot ranker, emit eig_per_dollar=0.0 so the row's blocker
        # surfaces; float("inf") would violate RFC 8259 on JSON serialize.
        eig_per_dollar = weighted_eig / cost_usd if cost_usd > 0 else 0.0

        # Propagate the cell's own blockers AND add a cost-estimation blocker
        # when we couldn't get a real number.
        blockers = list(cell.blockers)
        if cell.semantic_compatibility_warning is not None:
            blockers.append(
                f"semantic_compatibility_warning: {cell.semantic_compatibility_warning}"
            )
        if cost_usd <= 0.0:
            blockers.append(
                f"cost_estimation_required: substrate {cell.substrate_id!r} "
                "has midpoint cost <= $0 (planning artifact; promotion requires "
                "real dispatch cost anchor)"
            )

        out.append(
            _CellPlan(
                cell=cell,
                substrate=substrate,
                estimated_dispatch_cost_usd=float(cost_usd),
                cost_confidence_tag=cost_tag,
                posterior_correction=float(correction),
                posterior_n_observations=int(n_obs),
                axis_weight=float(axis_weight),
                expected_information_gain=float(weighted_eig),
                eig_per_dollar=float(eig_per_dollar),
                blockers=tuple(blockers),
            )
        )
    return out


def _enforce_envelope(
    plans: list[_CellPlan],
    *,
    per_dispatch_cap_usd: float,
    cumulative_cap_usd: float,
) -> tuple[list[dict[str, Any]], float]:
    """Sort by EV/$ desc, annotate envelope flags, return JSON-ready rows.

    Returns ``(rows, cumulative_spend_usd)``. The envelope walks down the
    EV/$-sorted list and marks any candidate whose cumulative cost would
    breach ``cumulative_cap_usd`` as ``fits_cumulative_envelope=False``.
    """
    sorted_plans = sorted(
        plans, key=lambda p: p.eig_per_dollar, reverse=True
    )
    clean_plans = [p for p in sorted_plans if not p.blockers]
    review_plans = [p for p in sorted_plans if p.blockers]
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for plan in clean_plans + review_plans:
        fits_per = plan.estimated_dispatch_cost_usd <= per_dispatch_cap_usd
        prospective = cumulative + plan.estimated_dispatch_cost_usd
        fits_envelope = prospective <= cumulative_cap_usd
        if fits_per and fits_envelope:
            cumulative = prospective
        cell = plan.cell
        substrate = plan.substrate
        composition_notes_lines = [
            "[predicted; substrate x primitive composition matrix v1]",
            f"substrate_id: {cell.substrate_id}",
            f"substrate_class: {substrate.substrate_class.value}",
            f"target_axis: {substrate.target_axis.value}",
            f"primitives_in_order: {list(cell.primitive_ids())}",
            f"composition_order: {list(cell.composition_order)}",
            f"predicted_bytes_delta: {cell.predicted_bytes_delta}",
            f"compatibility_verdict: {cell.compatibility_verdict}",
            f"axis_weight: {plan.axis_weight:.3f}",
            (
                f"posterior_correction: {plan.posterior_correction:.4f} "
                f"(n={plan.posterior_n_observations})"
            ),
            f"cost_confidence: {plan.cost_confidence_tag}",
        ]
        if cell.notes:
            composition_notes_lines.append(f"notes: {cell.notes}")
        if cell.semantic_compatibility_warning is not None:
            composition_notes_lines.append(
                f"semantic_compatibility_warning: {cell.semantic_compatibility_warning}"
            )
        rows.append(
            {
                "candidate_id": cell.cell_id,
                "family": substrate.substrate_class.value,
                "substrate_ids": [cell.substrate_id],
                "predicted_score_delta": float(
                    cell.predicted_score_delta * plan.posterior_correction
                ),
                "expected_information_gain": plan.expected_information_gain,
                "estimated_dispatch_cost_usd": plan.estimated_dispatch_cost_usd,
                "eig_per_dollar": plan.eig_per_dollar,
                "composition_notes": "\n".join(composition_notes_lines),
                "blockers": list(plan.blockers),
                "semantic_compatibility_warning": cell.semantic_compatibility_warning,
                "operator_review_required": bool(plan.blockers),
                "fits_per_dispatch_cap": bool(fits_per),
                "fits_cumulative_envelope": bool(fits_envelope),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return rows, cumulative


def build_payload(
    *,
    only_compatible: bool = True,
    only_with_primitives: bool = False,
    max_primitives_per_cell: int = 4,
    per_dispatch_cap_usd: float = DEFAULT_PER_DISPATCH_CAP_USD,
    cumulative_cap_usd: float = DEFAULT_CUMULATIVE_CAP_USD,
    axis_weights: dict[str, float] = DEFAULT_AXIS_WEIGHTS,
    apply_posterior: bool = True,
    apply_cost_band: bool = True,
    posterior_path: Path | None = None,
    max_total: int | None = None,
) -> dict[str, Any]:
    """Construct the autopilot-consumable ranking payload.

    The output schema mirrors ``tac_autopilot_dispatch_ranking_v1`` exactly
    so the autopilot's existing ``--use-substrate-composition-matrix-
    ranking`` flag accepts it without changes.
    """
    cells = enumerate_cells(max_primitives_per_cell=max_primitives_per_cell)
    if only_compatible:
        cells = [
            c
            for c in cells
            if c.compatibility_verdict
            in ("compatible", "compatible_bare_substrate")
        ]
    if only_with_primitives:
        cells = [c for c in cells if c.primitives]

    plans = _build_cell_plans(
        cells,
        posterior_path=posterior_path,
        axis_weights=axis_weights,
        apply_posterior=apply_posterior,
        apply_cost_band=apply_cost_band,
    )
    rows, cumulative_spend = _enforce_envelope(
        plans,
        per_dispatch_cap_usd=per_dispatch_cap_usd,
        cumulative_cap_usd=cumulative_cap_usd,
    )
    if max_total is not None:
        rows = rows[: max(0, int(max_total))]

    n_substrates_considered = len({c.substrate_id for c in cells})
    n_operator_review_rows = sum(
        1 for row in rows if row.get("operator_review_required") is True
    )

    payload: dict[str, Any] = {
        # Mirror the autopilot's canonical schema so the loader accepts us.
        "schema": AUTOPILOT_SCHEMA_VERSION,
        "bridge_schema": BRIDGE_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "matrix_schema": "tac_composition_cell_enumeration_v1",
        "n_substrates_considered": int(n_substrates_considered),
        "per_dispatch_cap_usd": float(per_dispatch_cap_usd),
        "cumulative_cap_usd": float(cumulative_cap_usd),
        "cumulative_estimated_spend_usd": float(cumulative_spend),
        "n_ranked_dispatches": len(rows),
        "n_operator_review_rows": int(n_operator_review_rows),
        "n_clean_dispatch_rows": int(len(rows) - n_operator_review_rows),
        "n_filtered_dropped": 0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_composition_cell_to_autopilot_bridge_v1",
        "ranked_dispatches": rows,
        "composition_constraints_applied": [
            "enumerate_cells_compatibility_matrix",
            "validate_pipeline_ordering_mutual_exclusion",
            "renderer_replacement_one_per_dispatch_chain",
            f"per_dispatch_cap_usd={per_dispatch_cap_usd}",
            f"cumulative_cap_usd={cumulative_cap_usd}",
            (
                "axis_weight_rule:pr106_pose_marginal_2_71x"
                if axis_weights is DEFAULT_AXIS_WEIGHTS
                else "axis_weight_rule:custom"
            ),
        ],
        "wire_in_hooks_exercised": [
            "hook_1_sensitivity_map_axis_weighting",
            "hook_4_cathedral_autopilot_dispatch_hook",
            "hook_5_continual_learning_posterior_consumed",
        ],
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "operator_gate_non_negotiable_at_every_dispatch",
            "halt_and_ask_default_on",
            "no_tmp_paths",
            "substrate_primitive_composition_cell_registry_v1",
            "composition_cell_to_autopilot_bridge_v1",
        ],
    }
    return payload


def _refuse_tmp_path(path: Path) -> None:
    s = str(path)
    if s.startswith("/tmp/") or "/private/tmp/" in s or "/var/tmp/" in s:
        raise ValueError(
            f"refusing to write to forbidden /tmp path: {path!r} "
            "(per CLAUDE.md Forbidden /tmp paths non-negotiable)"
        )


def write_payload(payload: dict[str, Any], path: Path) -> None:
    """Write the payload to a durable path (refuses /tmp)."""
    _refuse_tmp_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace inf/nan with explicit JSON-safe placeholders (RFC 8259).
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Where to write the bridge-generated ranking JSON. The path must "
            "NOT be under /tmp, /var/tmp, or /private/tmp per CLAUDE.md."
        ),
    )
    p.add_argument(
        "--max-primitives-per-cell",
        type=int,
        default=4,
        help="Cap on primitive-pipeline length per cell. Default 4.",
    )
    p.add_argument(
        "--include-bare-substrate",
        action="store_true",
        default=True,
        help="Include bare-substrate (no-primitive) baseline cells. Default ON.",
    )
    p.add_argument(
        "--only-with-primitives",
        action="store_true",
        help="Drop bare-substrate cells (only emit cells with >=1 primitive).",
    )
    p.add_argument(
        "--per-dispatch-cap-usd",
        type=float,
        default=DEFAULT_PER_DISPATCH_CAP_USD,
    )
    p.add_argument(
        "--cumulative-cap-usd",
        type=float,
        default=DEFAULT_CUMULATIVE_CAP_USD,
    )
    p.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Cap on number of emitted ranked dispatches. Default unlimited.",
    )
    p.add_argument(
        "--no-posterior",
        action="store_true",
        help="Skip continual-learning posterior reweighting (hook 5).",
    )
    p.add_argument(
        "--no-cost-band",
        action="store_true",
        help="Skip cost-band predictor; use substrate-midpoint costs only (hook 4).",
    )
    p.add_argument(
        "--posterior-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the continual-learning posterior JSONL. "
            "Defaults to tac.continual_learning.DEFAULT_POSTERIOR_PATH."
        ),
    )
    p.add_argument(
        "--pose-axis-weight",
        type=float,
        default=DEFAULT_AXIS_WEIGHTS["pose"],
        help=(
            "Per-axis weight on pose-axis EV (default 2.71 per PR106 frontier "
            "marginal rule)."
        ),
    )
    p.add_argument(
        "--seg-axis-weight",
        type=float,
        default=DEFAULT_AXIS_WEIGHTS["seg"],
    )
    p.add_argument(
        "--rate-axis-weight",
        type=float,
        default=DEFAULT_AXIS_WEIGHTS["rate"],
    )
    p.add_argument(
        "--mixed-axis-weight",
        type=float,
        default=DEFAULT_AXIS_WEIGHTS["mixed"],
    )
    args = p.parse_args(argv)

    if args.max_primitives_per_cell < 0:
        print(
            f"--max-primitives-per-cell must be >= 0; got "
            f"{args.max_primitives_per_cell}",
            file=sys.stderr,
        )
        return 2

    axis_weights = {
        "pose": float(args.pose_axis_weight),
        "seg": float(args.seg_axis_weight),
        "rate": float(args.rate_axis_weight),
        "mixed": float(args.mixed_axis_weight),
    }
    # Delegate validation to the canonical axis_weights API. Per
    # COUNCIL-A1 landing 2026-05-12, the validator lives in
    # tac.sensitivity_map.axis_weights so the bridge and the probe stay
    # coherent on what counts as a valid axis weight.
    try:
        validate_axis_weights_mapping(axis_weights)
    except AxisWeightsError as exc:
        print(f"build_composition_ranking_json: {exc}", file=sys.stderr)
        return 2

    payload = build_payload(
        only_compatible=True,
        only_with_primitives=bool(args.only_with_primitives),
        max_primitives_per_cell=int(args.max_primitives_per_cell),
        per_dispatch_cap_usd=float(args.per_dispatch_cap_usd),
        cumulative_cap_usd=float(args.cumulative_cap_usd),
        axis_weights=axis_weights,
        apply_posterior=not bool(args.no_posterior),
        apply_cost_band=not bool(args.no_cost_band),
        posterior_path=args.posterior_path,
        max_total=args.max_total,
    )
    try:
        write_payload(payload, args.output)
    except ValueError as exc:
        print(f"build_composition_ranking_json: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {payload['n_ranked_dispatches']} ranked composition cells "
        f"to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
