# SPDX-License-Identifier: MIT
"""Canonical law: FORMAT-vs-SEARCH attribution, and when codebook design is INADMISSIBLE (ddm_mq1).

``ddm_pw1`` established that a discrete menu is a falsifiable modelling choice and that
occupancy piling at a bound is a measurement request.  The natural next move -- and the
one the operator asked for -- is to design those menus properly: RD-optimal codepoint
placement, conditioning, entropy coding.  This law is the result of trying, and it says
**when that whole programme does not apply**, with two independent measured refusals.

THE ATTRIBUTION.  For any payload parameter whose values are searched and then stored,
the recoverable distortion splits into two DISJOINT parts, both measurable by holding
every other variable fixed and refining one coordinate to its continuous optimum:

    gap_lattice = d(nearest SHIPPABLE point) - d(continuous optimum)
        -> what a finer STORAGE FORMAT could buy.  Unreachable by more search.
    gap_search  = d(shipped) - d(nearest SHIPPABLE point)
        -> what a better SEARCH could buy at TODAY's format.  Independent of storage.

Their ratio says where to spend.  Reporting only the total, or optimising the format
without measuring ``gap_search``, is how a solver-limited payload gets mistaken for a
code-limited one.

MEASURED (ddm_mq1, 2026-08-01, n=48 mass-ordered pairs = 86.5% of population d_pose mass,
live v4d pose chain, canary EXACTLY 0.0, ``[macOS-CPU frozen-PoseNet advisory]``), as a
fraction of the own-vehicle gap to the bar (S 0.9476091 -> 0.172141, gap 0.7754681):

    coordinate            gap_LATTICE     gap_SEARCH     note
    p0 forward              0.0213%         0.1412%      NEGATIVE CONTROL (pw1 bracketed it)
    p1 lateral              0.0128%         0.4694%
    p2 vertical             0.0107%         0.8743%      moved up to 2,985 f16 cells
    beta rolling-shutter      --            0.3358%      over-fine reference, 10x step
    index streams @ H1      0.0106%           --         all three, 123 B total
    ------------------------------------------------------------------------------
    TOTAL                  <=0.056%        >=1.82%       SEARCH beats FORMAT by 33x

The negative control behaved: ``p0``, the one coordinate a self-terminating bracket had
already been run on, has the smallest search gap of the three, so the instrument is
finding real basins and not floating-point noise.

REFUSAL 1 -- ECONOMIC (degenerate lambda).  A codebook exists to trade rate for
distortion.  Here ``dS/dB = 25/37_545_489 = 6.6586e-07`` and ``dS/d(d_pose) =
5/sqrt(10*d_pose) = 18.083``, so ONE ARCHIVE BYTE == 3.68e-08 of mean d_pose, and the
entire addressable index-stream budget is 123 B = 0.0106% of the gap against a distortion
term worth 35.66%.  At that lambda no codepoint placement can be selected on rate: the
optimum is "as many codepoints as you like", i.e. ship the value.  This holds for ANY
weighting -- probability-weighted OR objective-weighted.

REFUSAL 2 -- STATISTICAL (unidentified argmin).  A codebook must be fitted to a
distribution of optima.  MEASURED with a wrong-initialisation positive control (n=16,
restart at the far end of the shipped table): the OBJECTIVE agrees across starts
(median |relative difference| 0.00354; 11/16 within 1%) but the ARGMIN does not --
**only 5/16 agree on the location**, and in 3/16 the wrong-init restart found a STRICTLY
BETTER optimum than the from-shipped search.  The objective is flat and multi-modal, so
there is no identified optimum distribution to fit.  Over-resolution removes MENU
censoring; it does NOT remove SOLVER bias, and this control is what detects the residue.

THE CENSORING THAT LOOKED LIKE A CODEBOOK.  DERIVED FROM SOURCE, not from data: a Swann
doubling bracket started at ``g0`` with step ``s`` reaches only ``g0 +- s*(2^k - 1)``.
Every one of the 13 values in the shipped ``rs_beta_mags`` table is a seed or a point on
that orbit, with no exceptions.  **The table was never a codebook -- it is the SEARCH'S
REACHABLE SET**, and its occupancy histogram is a picture of the search, not of a
solution density.  Its spacing doubles with distance from the seed, so its holes are
widest exactly where the largest wins were: re-running at a 10x finer step plus a
golden-section polish moved one pair -3.500 -> -5.7891 (the orbit jumps -3.5 -> -7.5)
for a gain of 2.26e-02.  Fitting a codebook to such a histogram fits the instrument.

SCOPE.  This law does NOT say codebook design is generally useless.  It gives two
PRE-CONDITIONS that must both hold before a codebook is admissible: rate must bind, and
the argmin must be identified.  Where they hold, the objective-weighted criterion (weight
each region by d(objective)/d(parameter), not by probability mass) is the correct one --
recorded here for the payload where it applies, and NOT used to claim optimality here.

Receipt: .omx/research/ddm_mq1_pose_menu_rd_audit_20260801.md
Producers: tools/mq1_pose_lattice_resolution_probe.py, tools/mq1_beta_overfine_reference.py
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_mq1_format_vs_search_attribution_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / ".omx/research/ddm_mq1_pose_menu_rd_audit_20260801.md"

#: Ratio of search-gap to lattice-gap at or above which the payload is called
#: SEARCH-limited.  Derived, not chosen: the measured separation is 33x aggregate
#: (and 6.6x on the negative control, the WEAKEST coordinate), so any threshold in
#: (1, 6.6] separates the observed classes.  2.0 is the conservative line -- "the
#: search axis is worth at least twice the format axis" -- and deliberately
#: under-reports rather than over-reports a search verdict.
SEARCH_DOMINANCE_THRESHOLD = 2.0

#: Fraction of the distortion at stake below which the addressable byte budget cannot
#: select between codebooks.  Anchored on the measured 0.0106% vs 35.66% (a factor of
#: 3,400); 0.01 (1%) is three orders of magnitude more permissive than the anchor and
#: is the weakest defensible line.
RATE_BINDS_MIN_FRACTION = 0.01

#: Minimum fraction of wrong-init control pairs that must agree on the ARGMIN before an
#: optimum distribution is treated as identified.  Anchored on the measured 5/16 = 0.3125
#: (refused); 0.8 is the conventional agreement bar and is stated as ASSUMED, not derived
#: -- see the anchor's reactivation criteria.
ARGMIN_IDENTIFIED_MIN_AGREEMENT = 0.8

# --- measured anchor constants (ddm_mq1, n600 population / n48 probe) ---------------
GAP_TO_BAR = 0.7754681
LATTICE_PCT_TOTAL = 0.0456          # p0+p1+p2 lattice, % of gap
SEARCH_PCT_TOTAL = 1.4850           # p0+p1+p2 search, % of gap
BETA_SEARCH_PCT = 0.3358            # beta over-fine reference, % of gap
INDEX_STREAM_CEILING_BYTES = 123
BYTE_TO_DPOSE = 3.6822e-08          # one archive byte, in mean-d_pose units
ARGMIN_AGREEMENT_MEASURED = 5.0 / 16.0
OBJECTIVE_AGREEMENT_MEASURED = 11.0 / 16.0


def format_vs_search(
    gap_lattice: float,
    gap_search: float,
    *,
    addressable_bytes: int | None = None,
    distortion_at_stake_S: float | None = None,
    byte_to_S: float = 25.0 / 37_545_489.0,
    argmin_agreement: float | None = None,
) -> dict[str, Any]:
    """Attribute recoverable distortion to FORMAT vs SEARCH, and gate codebook design.

    ``gap_lattice`` / ``gap_search`` are the two disjoint parts defined in the module
    docstring, in ANY consistent unit (both are compared only to each other).

    The two optional gates answer a different question from the attribution -- not
    "where is the distortion?" but "is a codebook even admissible?":

    * ``addressable_bytes`` + ``distortion_at_stake_S`` test REFUSAL 1 (does rate bind?).
    * ``argmin_agreement`` -- the fraction of wrong-init control restarts that agree on
      the optimum's LOCATION -- tests REFUSAL 2 (is the optimum identified?).

    A ``SEARCH_LIMITED`` verdict is a routing decision, not a claim that the format is
    perfect: it says the next unit of effort belongs on the solver.  Omitting a gate's
    inputs returns ``None`` for it and leaves ``codebook_admissible`` undetermined --
    an unrun gate is VACUOUS, never a pass.
    """
    gl, gs = float(gap_lattice), float(gap_search)
    if gl < 0.0 or gs < 0.0:
        raise ValueError("gaps must be non-negative; they are distortion REDUCTIONS")
    if gl == 0.0 and gs == 0.0:
        return {
            "verdict": "UNDETERMINED_EMPTY",
            "search_to_lattice_ratio": None,
            "note": "both gaps zero — VACUOUS scope (report the denominator), "
                    "never a CLOSED verdict",
            "codebook_admissible": None,
        }
    ratio = (gs / gl) if gl > 0.0 else float("inf")
    if ratio >= SEARCH_DOMINANCE_THRESHOLD:
        verdict = "SEARCH_LIMITED"
    elif ratio <= 1.0 / SEARCH_DOMINANCE_THRESHOLD:
        verdict = "FORMAT_LIMITED"
    else:
        verdict = "COMPARABLE_MEASURE_BOTH"

    rate_binds: bool | None = None
    rate_fraction: float | None = None
    if addressable_bytes is not None and distortion_at_stake_S is not None:
        if float(distortion_at_stake_S) <= 0.0:
            raise ValueError("distortion_at_stake_S must be positive")
        rate_fraction = (int(addressable_bytes) * float(byte_to_S)
                         / float(distortion_at_stake_S))
        rate_binds = rate_fraction >= RATE_BINDS_MIN_FRACTION

    argmin_identified: bool | None = None
    if argmin_agreement is not None:
        a = float(argmin_agreement)
        if not 0.0 <= a <= 1.0:
            raise ValueError("argmin_agreement is a fraction in [0,1]")
        argmin_identified = a >= ARGMIN_IDENTIFIED_MIN_AGREEMENT

    gates = (rate_binds, argmin_identified)
    if any(g is None for g in gates):
        admissible: bool | None = None
        reason = "gate_not_run_supply_both_rate_and_argmin_inputs"
    elif not rate_binds:
        admissible, reason = False, "REFUSED_rate_does_not_bind_lambda_degenerate"
    elif not argmin_identified:
        admissible, reason = False, "REFUSED_argmin_unidentified_no_density_to_fit"
    else:
        admissible, reason = True, "admissible_use_OBJECTIVE_weighted_placement"

    return {
        "verdict": verdict,
        "search_to_lattice_ratio": ratio,
        "gap_lattice": gl,
        "gap_search": gs,
        "search_share": gs / (gl + gs),
        "threshold": SEARCH_DOMINANCE_THRESHOLD,
        "rate_binds": rate_binds,
        "rate_fraction_of_distortion_at_stake": rate_fraction,
        "argmin_identified": argmin_identified,
        "codebook_admissible": admissible,
        "codebook_gate_reason": reason,
        "note": "a SEARCH_LIMITED verdict routes the next unit of effort to the solver; "
                "it does not claim the storage format is optimal",
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(inputs)
    allowed = {"gap_lattice", "gap_search", "addressable_bytes",
               "distortion_at_stake_S", "byte_to_S", "argmin_agreement"}
    if not keys <= allowed or not {"gap_lattice", "gap_search"} <= keys:
        raise ValueError(
            "format-vs-search inputs differ from the canonical callable contract "
            "(expected 'gap_lattice' and 'gap_search', optionally "
            "'addressable_bytes'/'distortion_at_stake_S'/'byte_to_S'/'argmin_agreement')"
        )
    kwargs = {k: inputs[k] for k in allowed - {"gap_lattice", "gap_search"} if k in inputs}
    return format_vs_search(inputs["gap_lattice"], inputs["gap_search"], **kwargs)


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_mq1_format_vs_search_attribution_v1() -> CanonicalEquation:
    """Build the format-vs-search attribution law with its measured anchors."""
    provenance = build_provenance_for_research_sidecar(
        RECEIPT,
        reactivation_criteria=(
            "Re-anchor if a payload called SEARCH_LIMITED is re-solved and the realized "
            "gain lands below its measured gap_search (would falsify the decomposition), "
            "if a coordinate called FORMAT_LIMITED is beaten by search alone, if a "
            "codebook REFUSED on argmin grounds is later fitted and pays (would falsify "
            "REFUSAL 2), or if ARGMIN_IDENTIFIED_MIN_AGREEMENT is DERIVED from a "
            "population of controls rather than assumed at the conventional 0.8. The "
            "0.8 agreement bar is the weakest constant in this law and is labelled "
            "ASSUMED; the 33x separation it is applied to is MEASURED."
        ),
        measurement_axis="[macOS-CPU frozen-PoseNet advisory]",
        hardware_substrate="darwin_arm64_cpu_frozen_posenet_n48_of_n600",
        captured_at_utc="2026-08-01T00:00:00Z",
    )
    attribution_anchor = EmpiricalAnchor(
        anchor_id="ddm_mq1_v4d_pose_format_vs_search_n48_20260801",
        measurement_utc="2026-08-01T00:00:00Z",
        inputs={
            "population": "48 mass-ordered pairs of 600 (86.5% of population d_pose mass)",
            "coordinates": ["p0_forward_NEGATIVE_CONTROL", "p1_lateral", "p2_vertical"],
            "chain": "ddm_v4c_resolve -> ddm_v4d_resolve -> inflate_runner_v4d",
            "canary_max_abs_err": 0.0,
            "gap_to_bar": GAP_TO_BAR,
        },
        predicted_output={
            "hypothesis": "the pose menus are badly designed codebooks and RD-optimal "
                          "placement will pay",
        },
        empirical_output={
            "lattice_pct_of_gap_total": LATTICE_PCT_TOTAL,
            "search_pct_of_gap_total": SEARCH_PCT_TOTAL,
            "beta_search_pct_of_gap": BETA_SEARCH_PCT,
            "index_stream_ceiling_bytes": INDEX_STREAM_CEILING_BYTES,
            "search_to_lattice_ratio": SEARCH_PCT_TOTAL / LATTICE_PCT_TOTAL,
            "negative_control_p0_search_pct": 0.1412,
            "negative_control_behaved": True,
            "verdict": "SEARCH_LIMITED",
            "hypothesis_refuted": True,
        },
        residual=0.0,
        source_artifact=str(RECEIPT.relative_to(REPO)),
        measurement_method=(
            "per-coordinate self-terminating Swann bracket plus golden-section polish "
            "at the realized frozen CPU PoseNet, holding every other shipped variable "
            "fixed and accepting only strict decreases; the nearest SHIPPABLE point is "
            "computed in the receiver's own per-column format (offset+f16 for p0, plain "
            "f16 for p1/p2), which is what makes the lattice/search split disjoint"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    # SECOND, INDEPENDENT ANCHOR: the two codebook REFUSALS, measured on the same
    # payload but by different instruments (an entropy census and a wrong-init control).
    refusal_anchor = EmpiricalAnchor(
        anchor_id="ddm_mq1_codebook_inadmissible_two_refusals_20260801",
        measurement_utc="2026-08-01T00:00:00Z",
        inputs={
            "addressable_bytes": INDEX_STREAM_CEILING_BYTES,
            "distortion_at_stake_S": 0.2765034,
            "positive_control": "wrong-init restart at g=-7.5, every 3rd pair, n=16",
            "beta_menu_size": 13,
        },
        predicted_output={
            "hypothesis": "an objective-weighted Lloyd-Max menu can be fitted and shipped",
        },
        empirical_output={
            "rate_fraction_of_distortion_at_stake": (
                INDEX_STREAM_CEILING_BYTES * (25.0 / 37_545_489.0) / 0.2765034),
            "rate_binds": False,
            "objective_agreement_within_1pct": OBJECTIVE_AGREEMENT_MEASURED,
            "argmin_agreement": ARGMIN_AGREEMENT_MEASURED,
            "argmin_identified": False,
            "wrong_init_strictly_better_fraction": 3.0 / 16.0,
            "codebook_admissible": False,
            "beta_table_is_the_bracket_orbit_not_a_codebook": True,
            "beta_orbit_formula": "g0 +- BETA_STEP0*(2^k - 1), BETA_STEP0=0.5",
            "beta_values_off_orbit": 0,
        },
        residual=0.0,
        source_artifact=str(RECEIPT.relative_to(REPO)),
        measurement_method=(
            "REFUSAL 1: first-order conditional entropy census over all three per-pair "
            "index streams against their shipped byte counts.  REFUSAL 2: identical "
            "over-fine search restarted from a deliberately wrong initialisation, "
            "comparing both the objective and the argmin location.  The beta-orbit "
            "identity is DERIVED from bracket_out's source, then verified exhaustively "
            "against all 13 shipped table entries"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Format-vs-search attribution and codebook admissibility",
        one_line_summary=(
            "Recoverable distortion splits into gap_lattice (finer FORMAT) and "
            "gap_search (better SEARCH); their ratio routes effort, and a codebook is "
            "admissible only if rate binds AND the argmin is identified."
        ),
        latex_form=(
            r"D_{\mathrm{rec}} = \underbrace{d(\hat{x}_q)-d(x^\*)}_{\text{gap}_{"
            r"\mathrm{lattice}}} + \underbrace{d(x_0)-d(\hat{x}_q)}_{\text{gap}_{"
            r"\mathrm{search}}};\quad \rho=\frac{\text{gap}_{\mathrm{search}}}{"
            r"\text{gap}_{\mathrm{lattice}}};\quad \text{codebook admissible} "
            r"\iff \frac{\lambda B}{D} \ge \tau_{\mathrm{rate}} \;\wedge\; "
            r"a_{\mathrm{argmin}} \ge \tau_{\mathrm{id}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_mq1_format_vs_search_attribution_20260801:"
            "format_vs_search"
        ),
        domain_of_validity={
            "applies_to": [
                "any payload parameter that is SEARCHED and then STORED at finite "
                "precision, where the receiver's per-column format is known",
            ],
            "requires": [
                "both sides of each gap evaluated at the SAME realized scorer "
                "(otherwise the split is not floor-free)",
                "the shippable-point quantizer must match the receiver's actual "
                "per-column reconstruction format",
                "a wrong-initialisation restart control before any argmin gate verdict",
            ],
            "excluded": [
                "coordinate-wise gaps treated as ADDITIVE — they interact; the totals "
                "attribute effort, they do not predict a joint gain",
                "claiming a codebook is OPTIMAL when either admissibility gate refuses "
                "or was not run (an unrun gate is VACUOUS, never a pass)",
                "any promotion, score, or submission use — the anchors are "
                "[macOS-CPU frozen-PoseNet advisory], score_claim=false",
            ],
        },
        canonical_producers=(
            "tools/mq1_pose_lattice_resolution_probe.py",
            "tools/mq1_beta_overfine_reference.py",
            "tools/mq1_joint_pose_refine_emit.py",
        ),
        canonical_consumers=(
            "tools/mq1_joint_pose_refine_emit.py",
            "experiments/ddm_v4d_resolve.py",
        ),
        empirical_anchors=(attribution_anchor, refusal_anchor),
        units_in={
            "gap_lattice": "distortion units (any, consistent with gap_search)",
            "gap_search": "distortion units (any, consistent with gap_lattice)",
            "addressable_bytes": "archive bytes",
            "distortion_at_stake_S": "score units",
            "argmin_agreement": "dimensionless fraction in [0,1]",
        },
        units_out={
            "verdict": "enum {SEARCH_LIMITED, FORMAT_LIMITED, COMPARABLE_MEASURE_BOTH, "
                       "UNDETERMINED_EMPTY}",
            "search_to_lattice_ratio": "dimensionless ratio in [0, inf]",
            "codebook_admissible": "tri-state {True, False, None=gate not run}",
        },
        predicted_vs_empirical_residual={
            "hypothesis_refuted_no_numeric_residual": 0.0,
        },
        last_calibration_utc="2026-08-01T00:00:00Z",
        provenance=provenance,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
    )
