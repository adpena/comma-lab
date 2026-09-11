# SPDX-License-Identifier: MIT
"""Canonical equation: a rung price is a property of the CODER STATE, not of the rung (ddm_pc3).

THE FAILURE THIS CODIFIES.  ddm_pc3 priced the pose-carrier lattice rungs on move 44's carrier,
then the pointer moved to 45 -- which was ddm_pc3's OWN predictor refit (-160 B, bit-identical
codes).  Re-pricing the SAME rungs against the SAME shipped codes on the new carrier moved every
price UP, because the refit predictor absorbs less of the halving than the stale one did:

    rung                          move 44      move 45     change
    cheapest per-dimension /2      +54 B       +75 B       +38.9 %
    dearest  per-dimension /2      +76 B       +81 B        +6.6 %
    global /2                     +808 B      +914 B       +13.1 %

So a rung table priced before a pointer move UNDER-CHARGES after it, and it under-charges MOST on
the cheapest rung -- precisely the rung a ranker picks.  A table that ranks correctly can still
price wrongly, and only the price decides admission.

THE MECHANISM, stated so it transfers.  The naive cost of halving a lattice pitch is one Rice bit
per symbol (900 B globally, 75 B per dimension on this 12-dimensional, 600-pair carrier).  The
AR(1)+bias predictor absorbs part of that bit.  A predictor fit to a SUPERSEDED residual absorbs
MORE of it by accident -- it is loose where the new one is tight -- so every structural rung
measured against it reads cheap.  Improving the predictor therefore RAISES the price of every
rung built on it.  The two effects are the same arithmetic seen twice.

    price(rung, coder_state) = naive_rice_cost(rung) - absorbed(predictor, field)

CONSEQUENCE (the operating rule).  Re-price every rung against the CURRENT coder before firing,
and never carry a price across a pointer move.  On move 45 this closed the family on two
independent legs: the n600 ceiling closes the GLOBAL rungs by cost before one is built (914 B
against 349.9 B of payable value for the whole family, CI [284.7, 415.5] -- 2.20x past even the
optimistic edge, and /4 is 4.41x past), and the REALIZED measurement closes the per-dimension
rungs the bound could not reach (thirteen rungs realized from the shipped point; not one nets
below zero; the cheapest is 4.4x short of the mean gain it needs).

Axis ``[exact archive bytes, contest-CUDA T4 n600 for the moves]``; ``score_claim=false`` for the
equation itself.  Registered by ddm_cons1 on 2026-09-11 from ddm_pc3's own §3/§7 tables.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pc3_rung_price_expires_at_pointer_move_v1"

_UTC = "2026-09-11T00:00:00Z"
_AXIS = "[exact archive bytes; contest-CUDA T4 n600 pointer rows]"
_LEDGER = ".omx/research/ddm_pc3_pose_carrier_rate_distortion_curve_on_move44_20260911.md"
_PRODUCER = "experiments/ddm_pc3_pose_carrier_curve.py"
_REFIT_PRODUCER = "experiments/ddm_pc3_predictor_refit.py"

# MEASURED: the same rungs against the same shipped codes, two coder states.
RUNG_PRICE_BYTES = {
    # rung: (move 44 carrier, move 45 carrier after the CPR1 predictor refit)
    "per_dimension_div2_cheapest": (54, 75),
    "per_dimension_div2_dearest": (76, 81),
    "global_div2": (808, 914),
}
NAIVE_RICE_GLOBAL_BYTES = 900
NAIVE_RICE_PER_DIMENSION_BYTES = 75
LATTICE_FAMILY_VALUE_BYTES = 495.6
LATTICE_FAMILY_VALUE_OPTIMISTIC_BYTES = 706.9
CARRIER_DIMENSIONS = 12


def price(rung: str, coder_state: str) -> int:
    """The MEASURED byte price of ``rung`` on ``coder_state`` in {'move_44', 'move_45'}."""
    index = {"move_44": 0, "move_45": 1}
    if rung not in RUNG_PRICE_BYTES:
        raise KeyError(f"no measured price for rung {rung!r}")
    if coder_state not in index:
        raise KeyError(f"rung prices were measured on move_44 and move_45 only, got {coder_state!r}")
    return RUNG_PRICE_BYTES[rung][index[coder_state]]


def price_inflation_ratio(rung: str) -> float:
    """DERIVED (pc3 states the paired prices, not a ratio): move-45 price / move-44 price.

    1.389x on the cheapest per-dimension rung, 1.131x on global /2 -- the stale table
    under-charges MOST on exactly the rung a ranker picks.
    """
    old, new = RUNG_PRICE_BYTES[rung]
    return new / old


def closed_by_the_bound(rung: str, coder_state: str = "move_45") -> bool:
    """True when the rung costs more than the whole lattice family is worth at infinite precision."""
    return price(rung, coder_state) > LATTICE_FAMILY_VALUE_OPTIMISTIC_BYTES


def build_pc3_rung_price_expires_at_pointer_move_v1() -> CanonicalEquation:
    """Build the rung-price-expiry canonical equation (ddm_pc3 pose corner, n600)."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_LEDGER,
        reactivation_criteria=(
            "re-price every rung against the CURRENT coder state before firing; this equation is "
            "re-measured whenever a pointer move changes the carrier's predictor, its codes, or "
            "the field they are fit to. The per-dimension rungs at 75 B are NOT closed -- the "
            "bound cannot reach their granularity and their verdict waits for their own row"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="macOS CPU (exact byte arithmetic) + contest-CUDA T4 for the pointer rows",
        captured_at_utc=_UTC,
    )

    expiry_anchor = EmpiricalAnchor(
        anchor_id="pc3_rung_reprice_move44_vs_move45_20260911",
        measurement_utc=_UTC,
        inputs={
            "object": "the shipped 12-dimensional pose carrier, 600 pairs, identical shipped codes",
            "coder_states": (
                "move 44 carrier vs move 45 carrier (move 45 IS ddm_pc3's own CPR1 AR(1)+bias "
                "predictor refit, -160 B, bit-identical codes)"
            ),
            "rungs": sorted(RUNG_PRICE_BYTES),
            "instrument": "real encode of each rung by the shipped coder; exact archive bytes",
        },
        predicted_output={
            "assumption_being_tested": (
                "a rung's byte price is a property of the rung, so a table priced on move 44 "
                "ranks AND prices correctly on move 45"
            ),
            "expected_change": "0 B",
            "cl3_prior": (
                "coder quality and capacity-change cost are SUBSTITUTES -- pc3 measured that law "
                "on a new axis (the PREDICTOR rather than the entropy coder)"
            ),
        },
        empirical_output={
            "rung_price_bytes_move44_move45": RUNG_PRICE_BYTES,
            # DERIVED by ddm_cons1 from pc3's paired price tables; pc3 states no ratio.
            "price_inflation_ratio_cheapest_per_dimension_DERIVED": 1.389,
            "price_inflation_ratio_dearest_per_dimension_DERIVED": 1.066,
            "price_inflation_ratio_global_div2_DERIVED": 1.131,
            "price_inflation_ratio_global_div4_DERIVED": 1.080,
            "naive_rice_cost": {
                "global_bytes": NAIVE_RICE_GLOBAL_BYTES,
                "per_dimension_bytes": NAIVE_RICE_PER_DIMENSION_BYTES,
            },
            "lattice_family_value_bytes": LATTICE_FAMILY_VALUE_BYTES,
            "lattice_family_value_optimistic_bytes": LATTICE_FAMILY_VALUE_OPTIMISTIC_BYTES,
            "global_rungs_verdict": (
                "CLOSED BY THE BOUND on move 45: 914 B against 706.9 B at the optimistic edge "
                "= 2.20x past it"
            ),
            "per_dimension_rungs_verdict": (
                "CLOSED by the REALIZED measurement, not by the bound (the bound cannot reach "
                "their granularity): thirteen rungs realized from the shipped point, not one nets "
                "below zero; the best, dim5 /2, is +3.855e-05 with its rate cost 4.4x its pose "
                "credit, and the cheapest rung is 4.4x short of the mean gain it needs"
            ),
            "direction": (
                "a better predictor RAISES the price of every structural rung built on it, "
                "because the stale predictor was absorbing the halving by accident"
            ),
        },
        # |predicted 0 B change - measured +21 B on the cheapest rung| / 21 B
        residual=1.0,
        source_artifact=_LEDGER,
        measurement_method="real_encode_of_each_rung_by_the_shipped_coder_on_two_pointer_states",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Rung prices expire at a pointer move (price = naive cost minus predictor absorption)",
        one_line_summary=(
            "A rung's price belongs to the coder state: per-dim /2 went 54 -> 75 B and global /2 "
            "808 -> 914 B across one pointer move, under-charging most on the cheapest rung"
        ),
        latex_form=(
            r"\mathrm{price}(r, c) = \mathrm{naive\_rice}(r) - \mathrm{absorbed}(\hat{p}_c, F_c),"
            r"\quad \hat{p}_{45} \succ \hat{p}_{44} \Rightarrow \mathrm{price}(r,45) > "
            r"\mathrm{price}(r,44)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pc3_rung_price_expiry_20260911:price"
        ),
        domain_of_validity={
            "object": (
                "the shipped pose carrier's lattice rungs (per-dimension and global pitch "
                "halvings) under the CPR1 AR(1)+bias predictor"
            ),
            "generalizes_to": (
                "any structural rung whose price is measured against a LEARNED predictor: "
                "improving the predictor raises the rung's price"
            ),
            "excluded": (
                "rungs whose cost is independent of a predictor (raw side-channel bytes); and "
                "any transfer of these ABSOLUTE byte figures to a different carrier -- the "
                "DIRECTION transfers, the numbers do not"
            ),
            "support": (
                "n = 2 coder states, 4 rung classes (per-dim cheapest/dearest, global /2 and /4), "
                "exact encodes with a byte-exact positive control"
            ),
            "verdict_this_priced": (
                "the coefficient-lattice refinement family is CLOSED at FORMULATION scope on move "
                "45's carrier: the whole family's payable bytes are 349.9 B, CI [284.7, 415.5], so "
                "global /2 at 914 B is 2.20x past even the optimistic edge and global /4 is 4.41x; "
                "not one of thirteen realized rungs nets below zero. NOT closed by this: a "
                "different carrier FORMAT, a rank change, or a solver reaching a DIFFERENT basin"
            ),
        },
        units_in={"rung": "rung name", "coder_state": "pointer-move token"},
        units_out={
            "price": "bytes added to archive.zip",
            "price_inflation_ratio": "dimensionless ratio of new price to stale price (DERIVED)",
        },
        empirical_anchors=(expiry_anchor,),
        predicted_vs_empirical_residual={"rung_price_stability_across_a_pointer_move": 1.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _LEDGER,
            ".omx/research/ddm_pc3_cap1_predictor_refit_move44_contest_cuda_20260911_pointer_move_45_20260911.md",
        ),
        canonical_producers=(_PRODUCER, _REFIT_PRODUCER),
        provenance=provenance,
    )
