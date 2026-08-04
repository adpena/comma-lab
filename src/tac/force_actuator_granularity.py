"""ACTUATOR GRANULARITY — the one axis `ddm_fl2`'s force ledger cannot express.

Arm `ddm_cg1`, task #809. This is an ANNOTATION LAYER over
`tac.witness_control.force_class_edge_ledger`, NOT a parallel ledger. It imports
fl2's `LEDGER` and adds a single orthogonal column. fl2 owns scope, verb,
verdict, evidence, protection, and magnitude; none of that is duplicated here.

THE HOLE THIS FILLS
-------------------
fl2's own governing law is::

    Asymmetry is a PRIOR on a per-site actuator, never the actuator.

But fl2 has no field for "is this actuator per-site or aggregate?". It encodes
that distinction as a sentinel *inside the verb axis* -- `verb == "AGGREGATE"`
means "this force has no verb channel". That works only for forces with no verb.
It cannot express the case the law is actually about::

    as1.grow_lane.harms : verb="TRANSFER", magnitude +0.2459 S, HARMS

That row has a real verb AND is a per-side aggregate actuator, and being an
aggregate actuator is *precisely why it failed*. fl2 can record the harm but not
the reason. Same for `as1.class_prior_logit_shift` (verb="TRANSFER", +0.2459 S)
and `mg1.barrier_key.neutral` (verb="AMPLITUDE", 0.0000% advantage).

Making granularity its own axis frees the verb axis to mean only "which
production did this act through", and makes the law's prediction QUERYABLE:
`predicted_dead()` returns every force that actuates through an aggregate and
has not yet been measured dead -- i.e. the standing falsifiable prediction.

FALSIFIER
---------
Land ONE working aggregate actuator -- a per-side or global scalar reweighting
with a measured, realized-through-R improvement -- and the law breaks. Eight
instruments have failed to do it so far.

AUTHORITY
---------
Classification only. No row here is a score claim, and no classification is a
measurement: `Granularity` records HOW a force actuates (readable from its
construction), never WHETHER it works (fl2's `verdict` owns that).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from tac.witness_control.force_class_edge_ledger import LEDGER, ForceLedgerRow

__all__ = [
    "GOVERNING_LAW",
    "GRANULARITY",
    "Granularity",
    "coverage",
    "granularity_of",
    "predicted_dead",
    "unclassified_forces",
]

Granularity = Literal[
    # Addresses elements individually -- the family that survives the law.
    "PER_SITE",  # per pixel / per scorer site
    "PER_TILE",  # per fixed spatial cell (e.g. 16x16)
    "PER_COMPONENT",  # per connected component
    # Cashes a quantity through an aggregate -- the family the law predicts dead.
    "PER_SIDE_AGGREGATE",  # one scalar per class or class-pair
    "GLOBAL_SCALAR",  # one scalar for the whole objective
    # Not an actuator at all.
    "STRUCTURAL",  # a property of the frozen scorer/operators
    "FAMILY_VERDICT",  # a verdict about a family, not a force you can fire
    "UNCLASSIFIED",  # construction not established -- reported, never assumed
]

AGGREGATE_GRANULARITIES: frozenset[str] = frozenset(
    {"PER_SIDE_AGGREGATE", "GLOBAL_SCALAR"}
)
PER_ELEMENT_GRANULARITIES: frozenset[str] = frozenset(
    {"PER_SITE", "PER_TILE", "PER_COMPONENT"}
)

GOVERNING_LAW: str = (
    "Asymmetry is a PRIOR on a per-site actuator, never the actuator. Eight "
    "independent instruments have failed to cash one through an aggregate. Any "
    "force whose granularity is aggregate is PREDICTED DEAD until a realized "
    "per-element receipt says otherwise."
)

# Classification is by CONSTRUCTION -- how the force applies its quantity -- and
# is deliberately conservative: a force is listed only where its construction is
# established from the cited arm's own description. Everything else stays
# UNCLASSIFIED and is counted in the denominator rather than assumed benign.
GRANULARITY: dict[str, Granularity] = {
    # --- per-element actuators (the surviving family) --------------------
    "per_site_margin_weight": "PER_SITE",  # mg1: weight applied at each site
    "margin_floor_hinge": "PER_SITE",  # mg1: per-site hinge on |m|
    "margin_hinge_weight": "PER_SITE",
    "squared_hinge": "PER_SITE",
    "seg_focal_gamma": "PER_SITE",
    "fisher_density_weight": "PER_SITE",
    "tau_softplus_tau": "PER_SITE",
    "gated_area_move_16px": "PER_TILE",  # as1: 16x16 gated cells
    "static_spatial_cell_index": "PER_TILE",  # hs1: static cell index
    # THE CANONICAL WORKED EXAMPLE OF THE LAW DONE RIGHT.
    # Named "per_edge_...", and cg1 first misclassified it as an aggregate on the
    # strength of that name -- which surfaced it as an apparent counterexample
    # (the only measured, realized-through-R IMPROVES among per-scope forces).
    # Adjudicating it against ru1's evidence inverted the reading: the actuator is
    # PER-ATLAS-CELL ("+yield in 17/18 cells"); the EDGE only supplies the SIGN
    # ("required RGB direction differs per edge: edge-resolve the existing sign
    # rule"). Edge = prior, cell = actuator. It confirms the law rather than
    # breaking it, and it is the template every other per-edge force should copy:
    # dS -0.046 at ~0 B = 61% of the entire Road<->Undrivable edge (0.075909 S).
    "per_edge_tie_calibration": "PER_TILE",
    "component_existence_production": "PER_COMPONENT",
    "lane_presence_existence_carrier": "PER_COMPONENT",
    "lane_existence_hinge_934": "PER_COMPONENT",
    "stroke_curve_production": "PER_COMPONENT",
    "region_paint_production": "PER_COMPONENT",
    "displacement_carrier": "PER_SITE",
    "phase_field_contour_carrier_bz1": "PER_SITE",
    "shared_grid_token_production": "PER_TILE",
    # --- aggregate actuators (PREDICTED DEAD) ----------------------------
    # Each of these applies ONE scalar across a whole class, side, or objective.
    "grow_lane_into_road_1px": "PER_SIDE_AGGREGATE",  # MEASURED +0.2459 S HARMS
    "class_prior_logit_shift": "PER_SIDE_AGGREGATE",  # MEASURED +0.2459 S HARMS
    "barrier_integral_rank_key": "PER_SIDE_AGGREGATE",  # MEASURED 0.0000% advantage
    "perclass_pair_surface_tension_sigma_ccprime": "PER_SIDE_AGGREGATE",  # #382
    "road_undriv_bulk_field": "PER_SIDE_AGGREGATE",  # BUILT_UNFIRED
    "density_weighted_signed_hinge_b16": "PER_SIDE_AGGREGATE",
    "class_weight_lane": "PER_SIDE_AGGREGATE",
    "lane_guard_lambda_primal_dual": "PER_SIDE_AGGREGATE",  # MEASURED INERT
    "tie_locus_edge_weighted_loss": "PER_SIDE_AGGREGATE",
    "realized_dseg_term": "GLOBAL_SCALAR",  # MEASURED INERT
    "free_generic_extractor_blend": "GLOBAL_SCALAR",  # MEASURED HARMS
    "wr1_tile_support_pricing": "GLOBAL_SCALAR",
    "drop_more_beyond_knee": "GLOBAL_SCALAR",
    "total_archive_ceiling_bytes_200000": "GLOBAL_SCALAR",
    "thr_wall_pose_adoption_gate": "GLOBAL_SCALAR",
    "sR_signed_reachability_field": "GLOBAL_SCALAR",
    "directional_entropy_prior": "GLOBAL_SCALAR",
    "gr1_gsum_gradient_key": "GLOBAL_SCALAR",
    "wr1_766_waterfill_flip_count_key": "GLOBAL_SCALAR",
    "head_natural_grad": "GLOBAL_SCALAR",
    "birth_plateau_knee_conjunct": "GLOBAL_SCALAR",
    # --- not actuators ----------------------------------------------------
    "scorer_d_seg_pricing": "STRUCTURAL",
    "scorer_D_null_space": "STRUCTURAL",
    "tr1_seg_leg_composite": "STRUCTURAL",  # the shipped objective, not a lever
    "any_aggregate_reweighting_of_d_seg": "FAMILY_VERDICT",
    "any_transmitted_location_mask": "FAMILY_VERDICT",
    "directed_asymmetry_as_prior": "FAMILY_VERDICT",
    "per_class_pair_scalar_reweight": "FAMILY_VERDICT",
    "per_pair_seg_correction_carrier": "FAMILY_VERDICT",
    "per_side_weight_any": "FAMILY_VERDICT",
}


def granularity_of(force: str) -> Granularity:
    """How does this force apply its quantity? UNCLASSIFIED when not established."""
    return GRANULARITY.get(force, "UNCLASSIFIED")


def predicted_dead(rows: Iterable[ForceLedgerRow] | None = None) -> list[ForceLedgerRow]:
    """Rows that actuate through an aggregate and are not yet measured dead.

    This is the ledger's standing falsifiable prediction. A row already measured
    HARMS/NEUTRAL/INERT is a CONFIRMATION, not a prediction, so it is excluded --
    predictions and confirmations must not be conflated into one count.
    """
    return [
        r
        for r in (LEDGER if rows is None else rows)
        if granularity_of(r.force) in AGGREGATE_GRANULARITIES
        and r.verdict not in ("HARMS", "NEUTRAL", "INERT")
    ]


def unclassified_forces(rows: Iterable[ForceLedgerRow] | None = None) -> list[str]:
    src = LEDGER if rows is None else rows
    return sorted({r.force for r in src if granularity_of(r.force) == "UNCLASSIFIED"})


def coverage(rows: Iterable[ForceLedgerRow] | None = None) -> dict:
    """Report the DENOMINATOR, not just a count (m50: vacuity must not read as pass)."""
    src = list(LEDGER if rows is None else rows)
    forces = {r.force for r in src}
    by_g: dict[str, int] = {}
    for f in forces:
        g = granularity_of(f)
        by_g[g] = by_g.get(g, 0) + 1
    agg = {f for f in forces if granularity_of(f) in AGGREGATE_GRANULARITIES}
    per_el = {f for f in forces if granularity_of(f) in PER_ELEMENT_GRANULARITIES}
    # The law's own scoreboard: among aggregate actuators that HAVE been measured,
    # how many improved? The law predicts zero.
    measured_agg = [
        r for r in src if granularity_of(r.force) in AGGREGATE_GRANULARITIES and r.measured
    ]
    return {
        "forces_total": len(forces),
        "forces_classified": len(forces) - len(unclassified_forces(src)),
        "forces_unclassified": len(unclassified_forces(src)),
        "by_granularity": dict(sorted(by_g.items())),
        "aggregate_actuator_forces": len(agg),
        "per_element_actuator_forces": len(per_el),
        "rows_predicted_dead": len(predicted_dead(src)),
        "measured_aggregate_rows": len(measured_agg),
        "measured_aggregate_rows_that_improved": sum(
            1 for r in measured_agg if r.verdict == "IMPROVES"
        ),
    }
