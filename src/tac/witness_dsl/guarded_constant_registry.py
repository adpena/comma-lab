# SPDX-License-Identifier: MIT
"""Canonical :class:`~tac.witness_dsl.guarded_constant.GuardedConstant` declarations.

One module, one place: every constant that has been MEASURED to be misusable
along the staleness / domain / units-role axes is declared here once, with its
provenance, and consumed from here.  This is the registry half of #533
(anti-duplicate-SoT) for the guard layer — the VALUE itself still lives in the
composed :class:`~tac.witness_dsl.lawref.LawRef` (#351).

Each declaration below is backed by an in-repo MEASURED artifact, cited inline.
Nothing here is a new number: every figure is quoted from an artifact or is an
exact closed form re-derived in this module's tests.

Axis: apparatus.  ``score_claim=false``, ``promotion_eligible=false``.
"""
from __future__ import annotations

from tac.witness_dsl.guarded_constant import (
    ROLE_EXCHANGE_RATE,
    ROLE_THRESHOLD,
    Corroboration,
    DerivationRef,
    DomainCheck,
    DomainSpec,
    GuardedConstant,
    Units,
)
from tac.witness_dsl.lawref import (
    LADDER_HARDCODED_WAIVER,
    LADDER_MEASURED_ANCHOR,
    InputRef,
    LawRef,
)

# ===========================================================================
# (a)+(b)  margin_floor — the frozen-output / wrong-domain instance.
# ===========================================================================
#: MEASURED, ``ddm_rt2`` (`.omx/research/ddm_rt2_realized_dseg_discarded_20260803.md`),
#: cached n600 GT margins, 600 pairs x 384 x 512 = 117,964,800 sites, Lane class
#: SELF-DETECTED by area signature (0.5855%, rarest) — never luma-sorted.
RT2_MEMO = ".omx/research/ddm_rt2_realized_dseg_discarded_20260803.md"
RT2_SITES = 117_964_800
RT2_LANE_SITES = 690_639

#: Lane-RESTRICTED QA80 margin percentiles (the correct domain).  Memo §9.2.
LANE_MARGIN_P5 = 0.0517
LANE_MARGIN_P10 = 0.1047
#: The derivation's own output at pct=10 on that field.  Memo §9.2.
LANE_DERIVED_FLOOR = 0.104748

#: GLOBAL-field QA80 margin percentiles (the WRONG domain — the (b) instance).
#: Memo §4: "p0.01% / p0.1% / p1% / p5% / median = 0.0036 / 0.0353 / 0.3552 /
#: 2.0582 / 5.8934".  Reading the same statistic off this population is what
#: produced the withdrawn "raise the floor ~20x" recommendation.
GLOBAL_MARGIN_P5 = 2.0582
GLOBAL_MARGIN_P50 = 5.8934

#: The value actually shipped today: the frozen literal default at
#: ``src/tac/optimization/direct_description_joint_descent.py`` on
#: ``DirectDescriptionJointDescentMLXModule.__init__``.  MEASURED: none of the 8
#: construction sites (4 files under ``tools/``) passes ``margin_floor``, so this
#: default is the value on every live path.
SHIPPED_MARGIN_FLOOR = 0.1

#: The independent second route: the L7 cross-hardware fp32 portability bound.
#: Real and load-bearing — represented as a corroboration, never resolved away.
L7_FP32_DRIFT_BOUND = 0.096

LANE_MARGIN_DOMAIN = DomainSpec(
    domain_id="qa80_margin__gt_lane_restricted",
    description=(
        "the QA80 scorer margin field RESTRICTED to GT-Lane pixels (class 1, "
        f"self-detected by area 0.5855%); {RT2_LANE_SITES:,} of {RT2_SITES:,} sites "
        f"on cached n600 GT. NOT the global margin field, whose p5 is "
        f"{GLOBAL_MARGIN_P5} = {GLOBAL_MARGIN_P5 / LANE_MARGIN_P5:.1f}x higher"
    ),
    checks=(
        DomainCheck.max_margin_separator(
            kind="quantile",
            param=5.0,
            in_domain_value=LANE_MARGIN_P5,
            out_of_domain_value=GLOBAL_MARGIN_P5,
            provenance=(
                f"both endpoints MEASURED in {RT2_MEMO} on the same cached n600 data "
                f"(lane-restricted p5={LANE_MARGIN_P5}; global p5={GLOBAL_MARGIN_P5})"
            ),
        ),
        DomainCheck.max_margin_separator(
            kind="quantile",
            param=50.0,
            in_domain_value=0.6013,
            out_of_domain_value=GLOBAL_MARGIN_P50,
            provenance=(
                f"both endpoints MEASURED in {RT2_MEMO} (lane-restricted p50=0.6013; "
                f"global median={GLOBAL_MARGIN_P50}, 'the median margin is 5.8934 — 59x "
                "the shipped floor')"
            ),
        ),
    ),
)

MARGIN_FLOOR = GuardedConstant(
    constant_id="seg_margin_hinge_floor",
    lawref=LawRef(
        equation_id="dsl_custodied_scalar_identity_v1",
        inputs={
            "value": InputRef.literal(
                SHIPPED_MARGIN_FLOOR,
                provenance=(
                    "the value SHIPPED today: frozen literal default "
                    "`margin_floor: float = 0.1` on "
                    "DirectDescriptionJointDescentMLXModule.__init__ "
                    "(src/tac/optimization/direct_description_joint_descent.py). "
                    "Custodied here so the frozen value carries its ladder rung; the "
                    "class-1 derived_live route is the DerivationRef below."
                ),
            )
        },
        ladder_class=LADDER_HARDCODED_WAIVER,
        fallback=SHIPPED_MARGIN_FLOOR,
        fallback_waiver_reason=(
            "class-4 custody of the incumbent so migration is byte-identical and the "
            "behaviour change is measured, not assumed; the class-1 route "
            "(derive_margin_floor over the Lane-restricted field) is declared and "
            f"returns {LANE_DERIVED_FLOOR} = "
            f"{LANE_DERIVED_FLOOR / SHIPPED_MARGIN_FLOOR:.3f}x the incumbent"
        ),
    ),
    units=Units.parse("dimensionless"),  # a scorer logit margin: same scale as the logits
    role=ROLE_THRESHOLD,
    domain=LANE_MARGIN_DOMAIN,
    derivation=DerivationRef(
        callable_path="tac.optimization.lane_guard:derive_margin_floor",
        kwargs={"pct": 10.0},
        returns_tuple_index=0,  # returns (floor, provenance_dict)
        invocation_required=True,
        drift_tolerance_ratio=1.10,
        tolerance_provenance=(
            f"the two INDEPENDENT routes to this floor span {L7_FP32_DRIFT_BOUND} "
            f"(L7 fp32 cross-hardware portability bound) .. {LANE_DERIVED_FLOOR} "
            f"(Lane-restricted p10, {RT2_MEMO}) = "
            f"{LANE_DERIVED_FLOOR / L7_FP32_DRIFT_BOUND:.3f}x; the tolerance is set at that "
            "measured spread so agreement between the routes is not read as drift, and "
            "anything wider than the routes' own disagreement IS"
        ),
    ),
    corroborations=(
        Corroboration(
            route_id="l7_fp32_cross_hardware_drift_bound",
            value=L7_FP32_DRIFT_BOUND,
            provenance=(
                "the cross-hardware fp32 portability guard the floor also serves; real "
                "and load-bearing — this route is REPRESENTED, not resolved away"
            ),
        ),
        Corroboration(
            route_id="gt_lane_restricted_margin_p10",
            value=LANE_DERIVED_FLOOR,
            provenance=(
                f"derive_margin_floor(lane_margins, pct=10.0) = {LANE_DERIVED_FLOOR} on "
                f"cached n600 ({RT2_MEMO} §9.2)"
            ),
        ),
    ),
    corroboration_tolerance_ratio=1.10,
    corroboration_tolerance_provenance=(
        f"the two routes MEASURE {L7_FP32_DRIFT_BOUND} and {LANE_DERIVED_FLOOR}; their "
        f"actual spread is {LANE_DERIVED_FLOOR / L7_FP32_DRIFT_BOUND:.4f}x, so the "
        "tolerance is that measured spread. It is not a chosen slack: if either route "
        "moves further than the other has ever been, that is a real conflict"
    ),
    incumbent_literal=SHIPPED_MARGIN_FLOOR,
    literal_site_names=("margin_floor",),
    notes=(
        "ddm_rt2 (34011da8c1) WITHDREW the 'wrong floor' finding: the shipped 0.1 is "
        f"{LANE_DERIVED_FLOOR / SHIPPED_MARGIN_FLOOR:.3f}x the derived value, so NOTHING "
        "FAILS today. That is precisely why this guard exists — the derivation is "
        "documented 'data-derived per run, never a bare constant' and is not invoked at "
        "this call site, so the run cannot adapt and no test would go red if the data moved."
    ),
)

#: The byte-identical migration value for the frozen call site.  Resolving with
#: no sample and no consumer role would REFUSE (invocation_required=True); the
#: module-scope value below is obtained through :meth:`GuardedConstant.audit`,
#: which is the LOUD read path: it returns the custodied incumbent AND the list
#: of violations, so importing this name can never be mistaken for having run
#: the derivation.  A caller that HAS the Lane-restricted field should call
#: ``MARGIN_FLOOR.resolve(consumer_role=ROLE_THRESHOLD, sample=..., \
#: sample_domain_id=LANE_MARGIN_DOMAIN.domain_id)`` and get the live value.
_MARGIN_FLOOR_AUDIT = MARGIN_FLOOR.audit(consumer_role=ROLE_THRESHOLD)
MARGIN_FLOOR_INCUMBENT: float = float(_MARGIN_FLOOR_AUDIT.value)
#: Byte-identity of the migration, PROVEN at import (never assumed).
MARGIN_FLOOR_MIGRATION_IS_BYTE_IDENTICAL: bool = bool(
    _MARGIN_FLOOR_AUDIT.byte_identical_to_incumbent
)


# ===========================================================================
# (c)  W — the flip<->byte EXCHANGE RATE, repeatedly consumed as a COST.
# ===========================================================================
#: EXACT closed form, re-derived this session:
#:     100*flips/(n*H*W) == 25*bytes/DEN   =>   bytes/flip = 4*DEN/(n*H*W)
#:     = 4*37,545,489 / (600*384*512) = 4,171,721/3,276,800 = 1.2731082153320312
#: Both S-legs are linear in their arguments, so this is an identity of the score
#: function, not an empirical property of any mechanism.
CONTEST_RATE_DENOMINATOR = 37_545_489
N_PAIRS = 600
FRAME_H = 384
FRAME_W = 512
W_BYTES_PER_FLIP = 4 * CONTEST_RATE_DENOMINATOR / (N_PAIRS * FRAME_H * FRAME_W)

#: MEASURED mechanism costs, for the record: they span 65x around W —
#: sx2 0.4981 B/flip; mf1 1,854 B / 1,483 components; qd1 32.53 B/flip.
#: Reading W as a mechanism cost silently asserts every mechanism is break-even.
MEASURED_MECHANISM_COST_SPAN = (0.4981, 32.53)

SCORE_FUNCTION_IDENTITY_DOMAIN = DomainSpec(
    domain_id="contest_score_function_identity",
    description=(
        "the contest score function itself (S = 100*d_seg + sqrt(10*d_pose) + "
        f"25*bytes/{CONTEST_RATE_DENOMINATOR}) at n={N_PAIRS} pairs, {FRAME_H}x{FRAME_W}. "
        "This is an exact identity, not a sample of any mechanism's behaviour"
    ),
)

W_EXCHANGE_RATE = GuardedConstant(
    constant_id="flip_byte_exchange_rate_W",
    lawref=LawRef(
        equation_id="dsl_custodied_scalar_identity_v1",
        inputs={
            "value": InputRef.literal(
                W_BYTES_PER_FLIP,
                provenance=(
                    "EXACT closed form 4*DEN/(n*H*W) = "
                    f"4*{CONTEST_RATE_DENOMINATOR}/({N_PAIRS}*{FRAME_H}*{FRAME_W}) = "
                    "4171721/3276800 = 1.2731082153320312; both S-legs are linear so the "
                    "rate is exact, not fitted"
                ),
                config_tags={
                    "n_pairs": str(N_PAIRS),
                    "frame_hw": f"{FRAME_H}x{FRAME_W}",
                    "rate_denominator": str(CONTEST_RATE_DENOMINATOR),
                },
            )
        },
        ladder_class=LADDER_MEASURED_ANCHOR,
    ),
    units=Units.parse("bytes/flip"),
    role=ROLE_EXCHANGE_RATE,
    domain=SCORE_FUNCTION_IDENTITY_DOMAIN,
    notes=(
        "W says what one flip is WORTH. It does NOT say what any mechanism CHARGES: "
        f"measured mechanism costs span {MEASURED_MECHANISM_COST_SPAN[0]} .. "
        f"{MEASURED_MECHANISM_COST_SPAN[1]} B/flip = "
        f"{MEASURED_MECHANISM_COST_SPAN[1] / MEASURED_MECHANISM_COST_SPAN[0]:.0f}x. "
        "Consuming W with role=mechanism_cost REFUSES."
    ),
)


REGISTRY: dict[str, GuardedConstant] = {
    MARGIN_FLOOR.constant_id: MARGIN_FLOOR,
    W_EXCHANGE_RATE.constant_id: W_EXCHANGE_RATE,
}


__all__ = [
    "GLOBAL_MARGIN_P5",
    "GLOBAL_MARGIN_P50",
    "L7_FP32_DRIFT_BOUND",
    "LANE_DERIVED_FLOOR",
    "LANE_MARGIN_DOMAIN",
    "LANE_MARGIN_P5",
    "LANE_MARGIN_P10",
    "MARGIN_FLOOR",
    "MARGIN_FLOOR_INCUMBENT",
    "MARGIN_FLOOR_MIGRATION_IS_BYTE_IDENTICAL",
    "MEASURED_MECHANISM_COST_SPAN",
    "REGISTRY",
    "RT2_MEMO",
    "SCORE_FUNCTION_IDENTITY_DOMAIN",
    "SHIPPED_MARGIN_FLOOR",
    "W_BYTES_PER_FLIP",
    "W_EXCHANGE_RATE",
]
