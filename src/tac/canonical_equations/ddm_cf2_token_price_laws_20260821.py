# SPDX-License-Identifier: MIT
"""ddm_cf2 -- the two token-field PRICING laws the jg/fs lineage measured, registered.

These are the ``rv17`` wave-2 **F2** carried item (the equations leg, deliberately deferred
until the mirror encode measured the REMOVAL-direction constant -- it now has).  Both laws
are ``verdict_scope: FORMULATION``: they govern **jg3-class edit-configuration re-selection
under real prices, on the rc2-lineage token field**.  Neither is a family law and neither is
a score.

**Law (a) -- direction dependence.** The autoregressive ``-log2 p`` rate model may be trusted
in one direction of travel and not the other.  Moving tokens AWAY from the model's argmax,
the realized price is ~0.92x of the ACTUAL flat price the selector charged (the model
*overcharges* by ~8%).  Moving them TOWARD the argmax, the realized credit is ~0.09x of the
modelled credit (the model *overstates* the credit ~11x), because erasing the image's
departure from the model's prior degrades the neighbourhood's own conditioning -- a
second-order recapture of **91.3%** of the first-order credit at ``u = 7.75``, and over 100%
at ``u = 12.0`` where the substitution *costs* 37 bytes.

  ⚠ The ``4.718`` figure paired with these ratios historically is ``ddm_jg1``/``ddm_jg3``'s
  ``LogitPrice`` **RANKER** -- ordering only, never a price, per its own docstring.  Ratios
  against it are *ranker-relative* and are named as such here
  (:func:`ranker_relative_ratio`).  The price jg3 actually charged is a flat
  **4.1379 bits/token**.  Series A (vs the ranker) and Series B (vs the actual price) are
  two different questions and are never mixed.

**Law (b) -- average is not marginal.** The average price of a greedy-admitted edit set is
not the price of its marginal member.  The *arithmetic* half is true by definition and is
framing.  The *empirical, FORMULATION-scoped* half is the *degradation direction*: because
jg3's greedy orders sites by gain, a denser configuration adds exactly the sites it already
ranked worst, so price and yield degrade **together** -- measured 2.24x the set-average price
at 3.98x less yield on the ADD side of the admission cut, and 1.46x the trimmed-set average
on the REMOVE side of the same cut.  A flat prior fitted to a set's average is therefore
simultaneously too dear for the set and too cheap for its margin, and correcting only the
first error admits exactly the configurations the second forbids.

Every constant below was re-derived in code from a retained encode receipt and agrees with
that receipt's own published field.  Axis ``[macOS-CPU advisory / scorer-free EXACT byte
measurement]``; ``score_claim=false``.  Only ``upstream/evaluate.py`` on contest hardware is
a score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tac.provenance import Provenance

MEMO = ".omx/research/ddm_cf2_wave2_carried_items_20260821.md"

_AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"


# ---------------------------------------------------------------------------
# The shared measurement function -- this is how every constant here was derived.
# Every module constant below is COMPUTED from a retained receipt's own (bytes,
# tokens) pair through this function; none is typed as a decimal.
# ---------------------------------------------------------------------------


def bits_per_token(delta_bytes: float, tokens: int) -> float:
    """Realized price/credit of a token-field edit, from an EXACT archive byte delta.

    This is the only instrument either law admits: a REAL re-encode's ``stat`` delta over
    the token count it moved.  ``tokens`` is the count of tokens that CHANGED, which is the
    denominator both laws are stated against; passing a set size when the object is a
    margin (or vice versa) is the law-(b) error itself.
    """
    if tokens == 0:
        raise ValueError("tokens must be non-zero (a price needs a denominator)")
    return float(delta_bytes) * 8.0 / float(tokens)


#: The flat price ``ddm_jg3``'s selector actually CHARGED per changed token, measured by
#: ``ddm_jg2``'s byte-identical re-encoder.  This is a price and may be a denominator.
JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN = 4.1379

#: ``ddm_jg1``/``ddm_jg3``'s ``LogitPrice`` mean.  A **RANKER** -- ordering signal only, per
#: its own docstring.  Retained because published receipts carry
#: ``modelled_bits_per_changed_token_jg1 = 4.718`` and ratios against it circulate; it is
#: NEVER a price and never a trust factor (rv17 W2-F3/F4/F14).
JG1_LOGIT_RANKER_BITS_PER_TOKEN = 4.718

#: jg5's shipped 455-edit set: 4,151 B over 8,654 changed tokens.  Equals the receipt's own
#: ``measured_bits_per_changed_token`` field (3.837300670210307) exactly.
JG5_SHIPPED_455_BITS_PER_TOKEN = bits_per_token(4151, 8654)

#: jg3's own configuration, from ``ddm_jg4``'s retained per-frame code-bit arrays:
#: 5,196.258 B over 10,900 changed tokens (3.8137673394495413).
JG3_CONFIGURATION_BITS_PER_TOKEN = bits_per_token(5196.258, 10900)

#: Law (a), AWAY from the model argmax: realized / ACTUAL price.  A two-point MEASURED band,
#: not one number -- the two points are two different objects on one lineage (jg5's shipped
#: set and jg3's configuration), so the band's WIDTH is object spread, not noise.
AWAY_TRUST_VS_ACTUAL_PRICE_BAND = (
    JG3_CONFIGURATION_BITS_PER_TOKEN / JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN,
    JG5_SHIPPED_455_BITS_PER_TOKEN / JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN,
)

#: Law (a), TOWARD the model argmax: realized / MODELLED credit.  Note the denominator is
#: the MODEL, not the actual price -- the toward direction has no "actual price" because
#: the operation is a credit, not a charge.  ``u = 7.75`` realized 1,022 B of a modelled
#: 11,716.7; ``u = 12.0`` realized -37 B of a modelled 2,546.1 (the substitution COSTS
#: bytes).  The ladder is monotone adverse, so neither point predicts the other.
TOWARD_TRUST_VS_MODEL_U7P75 = 1022 / 11716.7
TOWARD_TRUST_VS_MODEL_U12 = -37 / 2546.1


# ---------------------------------------------------------------------------
# Law (a) -- direction dependence.
# ---------------------------------------------------------------------------

AWAY_FROM_ARGMAX = "away_from_argmax"
TOWARD_ARGMAX = "toward_argmax"
_VALID_DIRECTIONS = (AWAY_FROM_ARGMAX, TOWARD_ARGMAX)


def direction_trust_factor(direction: str) -> float:
    """The MEASURED realized/modelled factor for a direction of travel.

    AWAY returns the midpoint of the two-point band (the band itself is
    :data:`AWAY_TRUST_VS_ACTUAL_PRICE_BAND` and a caller pricing a decision should carry
    both ends).  TOWARD returns the ``u = 7.75`` point, the less adverse of the two
    measured rungs -- so a prediction built on it is the OPTIMISTIC end.

    Raises on any other token: there is no measured third direction, and silently
    defaulting one would be the fake this law exists to prevent.
    """
    if direction == AWAY_FROM_ARGMAX:
        low, high = AWAY_TRUST_VS_ACTUAL_PRICE_BAND
        return (low + high) / 2.0
    if direction == TOWARD_ARGMAX:
        return TOWARD_TRUST_VS_MODEL_U7P75
    raise ValueError(
        f"direction={direction!r} must be one of {_VALID_DIRECTIONS!r}; "
        "no other direction has been measured on this lineage"
    )


def predict_realized_bits_per_token(modelled_bits_per_token: float, direction: str) -> float:
    """Predict the REALIZED bits/token from a modelled one, given the direction of travel.

    The denominators differ by direction and that is the law: AWAY is measured against the
    ACTUAL flat price the selector charged, TOWARD against the MODELLED credit.  Feeding a
    toward-direction model price into the away factor (or the reverse) is a ~10x error, so
    the direction is a required argument, never a default.
    """
    return float(modelled_bits_per_token) * direction_trust_factor(direction)


def second_order_recapture_fraction(realized: float, modelled: float) -> float:
    """Fraction of the first-order credit the model's own conditioning takes back.

    ``1 - realized/modelled``.  At ``u = 7.75`` this is 0.913 (91.3%); at ``u = 12.0`` it
    exceeds 1.0, i.e. the substitution costs more than it was worth.  A value at or below
    zero means the edit realized at least the modelled amount, which is the AWAY regime.
    """
    if modelled == 0:
        raise ValueError("modelled must be non-zero to form a recapture fraction")
    return 1.0 - float(realized) / float(modelled)


def ranker_relative_ratio(
    realized_bits_per_token: float,
    ranker_bits_per_token: float = JG1_LOGIT_RANKER_BITS_PER_TOKEN,
) -> float:
    """Realized price against the ``LogitPrice`` **RANKER** -- an ordering ratio, NOT a trust factor.

    Named so that a ranker-relative number can never be read as a price ratio.  jg5's
    published ``realized_over_modelled`` (0.8133) and the mirror's (0.7722) are both this
    quantity.  To ask "did the model overcharge?", divide by
    :data:`JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN` instead.
    """
    if ranker_bits_per_token == 0:
        raise ValueError("ranker_bits_per_token must be non-zero")
    return float(realized_bits_per_token) / float(ranker_bits_per_token)


# ---------------------------------------------------------------------------
# Law (b) -- greedy set average is not its marginal member's price.
# ---------------------------------------------------------------------------


def marginal_over_average_ratio(marginal: float, average: float) -> float:
    """How far the marginal member's price sits from the set average it is quoted with.

    The ARITHMETIC half of law (b): a ratio != 1 is true by definition for any non-constant
    set and carries no empirical content on its own.  What is MEASURED (and
    FORMULATION-scoped) is that on a greedy-ranked selection this ratio is > 1 -- the margin
    is dearer -- while the yield ratio moves the other way.  See
    :func:`greedy_margin_degrades_both_terms`.
    """
    if average == 0:
        raise ValueError("average must be non-zero")
    return float(marginal) / float(average)


def predict_reselection_delta_bits(delta_tokens: int, marginal_bits_per_token: float) -> float:
    """Price a configuration re-selection at the MARGINAL rate, never the set average.

    A re-selection delta buys or sells only its ``delta_tokens``, and on a gain-ordered
    greedy those are by construction the ranking's worst members.  Round 1 of ``ddm_fs3``
    priced a +300-token re-selection at the 2.6573 b/tok average of the 569 tokens it was
    added to and projected a 7.11x-the-bar win; the real re-encode measured 5.9467 b/tok and
    the candidate archive came back LARGER than the body it came from.
    """
    return float(delta_tokens) * float(marginal_bits_per_token)


def greedy_margin_degrades_both_terms(
    set_price: float, marginal_price: float, set_yield: float, marginal_yield: float
) -> bool:
    """True when the margin is BOTH dearer per token AND lower-yielding than the set.

    This is the empirical, FORMULATION-scoped clause of law (b) as a falsifiable predicate.
    Measured true on the ADD side (2.24x price, 3.98x less yield) and, on price, true again
    on the REMOVE side of the same admission cut (1.46x).  A greedy-selected configuration
    that returns False here has either a non-gain-ordered ranking or a price that does not
    track rank, and reopens this law.
    """
    return marginal_price > set_price and marginal_yield < set_yield


# ---------------------------------------------------------------------------
# Registration.
# ---------------------------------------------------------------------------


def _provenance() -> Provenance:
    return build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "law (a): a real re-encode on this lineage measuring an away-direction ratio "
            "outside 0.92-1.00, or a toward-direction ratio above ~0.2, reopens the "
            "direction split. law (b): a greedy-selected configuration whose marginal "
            "member prices at or below its set average reopens the degradation direction. "
            "BOTH: every constant is rc2-lineage and jg3-class; neither transfers to "
            "another token field, another coder, or another selector without re-measurement"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="apple_macos_cpu",
        captured_at_utc="2026-08-21T00:00:00Z",
    )


def build_token_rate_model_direction_dependence_v1() -> CanonicalEquation:
    """Law (a): the ``-log2 p`` model's trust factor depends on the DIRECTION of the edit."""
    provenance = _provenance()
    jg5_realized = JG5_SHIPPED_455_BITS_PER_TOKEN
    jg3_realized = JG3_CONFIGURATION_BITS_PER_TOKEN
    # Both rungs' bits/token are DERIVED from the memo's own (bytes, flips) pairs.
    u7_modelled = bits_per_token(11716.7, 9106)
    u7_realized = bits_per_token(1022, 9106)
    u12_modelled = bits_per_token(2546.1, 1440)
    u12_realized = bits_per_token(-37, 1440)
    anchors = (
        EmpiricalAnchor(
            anchor_id="cf2_away_jg5_shipped_455_20260820",
            measurement_utc="2026-08-20T00:00:00Z",
            inputs={
                "direction": AWAY_FROM_ARGMAX,
                "modelled_bits_per_token": JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN,
                "delta_bytes": 4151,
                "tokens_changed": 8654,
            },
            predicted_output={
                "realized_bits_per_token": predict_realized_bits_per_token(
                    JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN, AWAY_FROM_ARGMAX
                )
            },
            empirical_output={
                "realized_bits_per_token": jg5_realized,
                "realized_over_actual_price": jg5_realized
                / JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN,
                "ranker_relative_ratio": ranker_relative_ratio(jg5_realized),
            },
            residual=abs(
                predict_realized_bits_per_token(
                    JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN, AWAY_FROM_ARGMAX
                )
                - jg5_realized
            ),
            source_artifact=(
                "/Volumes/APDataStore/pact/ddm_jg5/retained/final/S1_encode_jg5_subset455.json"
            ),
            measurement_method=(
                "ddm_jg5's shipped 455-edit set, priced by ddm_jg2's re-encoder whose "
                "unedited control reproduces the shipped token stream byte-identically "
                "(sha 15054e5da33640bc...). archive 176,429 -> 180,580 B, token stream "
                "109,696 -> 113,847 B (sha b9243abd2e38f9ae...), 8,654 tokens changed: "
                "4,151*8/8,654 = 3.837300670210307 b/tok, which equals the receipt's own "
                "measured_bits_per_changed_token field exactly. Against jg3's ACTUAL flat "
                "price 4.1379 that is 0.92735 (a 7.84% overcharge). The receipt's own "
                "realized_over_modelled 0.8133320623591156 is the RANKER-relative ratio "
                "(vs 4.718) and is NOT this quantity"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
        EmpiricalAnchor(
            anchor_id="cf2_away_jg3_configuration_20260820",
            measurement_utc="2026-08-20T00:00:00Z",
            inputs={
                "direction": AWAY_FROM_ARGMAX,
                "modelled_bits_per_token": JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN,
                "delta_bytes": 5196.258,
                "tokens_changed": 10900,
            },
            predicted_output={
                "realized_bits_per_token": predict_realized_bits_per_token(
                    JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN, AWAY_FROM_ARGMAX
                )
            },
            empirical_output={
                "realized_bits_per_token": jg3_realized,
                "realized_over_actual_price": jg3_realized
                / JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN,
                "overcharge_factor": JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN / jg3_realized,
            },
            residual=abs(
                predict_realized_bits_per_token(
                    JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN, AWAY_FROM_ARGMAX
                )
                - jg3_realized
            ),
            source_artifact=".omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md",
            measurement_method=(
                "ddm_fs3 section 2, from ddm_jg4's retained per-frame code-bit arrays: "
                "5,196.258 B over jg3's own 10,900 changed tokens = 3.8137673394495413 "
                "b/tok, an 8.50% overcharge against the 4.1379 the selector charged "
                "(4.1379/3.813767 = 1.084990). A SECOND object on the same lineage, not a "
                "restatement of the jg5 anchor: it also measures 10.277 B of context bleed "
                "into pairs that were never edited, so per-pair attribution is leaky"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
        EmpiricalAnchor(
            anchor_id="cf2_toward_rung4_u7p75_20260820",
            measurement_utc="2026-08-20T00:00:00Z",
            inputs={
                "direction": TOWARD_ARGMAX,
                "modelled_bits_per_token": u7_modelled,
                "modelled_bytes_saved": 11716.7,
                "p_max_threshold": 0.9953547,
                "flips": 9106,
            },
            predicted_output={
                "realized_bits_per_token": predict_realized_bits_per_token(
                    u7_modelled, TOWARD_ARGMAX
                )
            },
            empirical_output={
                "realized_bits_per_token": u7_realized,
                "realized_bytes_saved": 1022,
                "realized_over_modelled": TOWARD_TRUST_VS_MODEL_U7P75,
                "second_order_recapture_fraction": second_order_recapture_fraction(
                    1022, 11716.7
                ),
            },
            residual=abs(
                predict_realized_bits_per_token(u7_modelled, TOWARD_ARGMAX) - u7_realized
            ),
            source_artifact=".omx/research/ddm_fs2_rc4_drop_carrier_resolve_20260820.md",
            measurement_method=(
                "ddm_fs2 rung 4 at u = 7.75: the confidence-threshold token drop writes the "
                "model's own argmax at 9,106 positions. Modelled 11,716.7 B saved; MEASURED "
                "1,022 B by exact archive stat (180,456 -> 179,434 B), realized/modelled "
                "0.0872 -- the second-order term recaptures 91.3% of the first-order credit "
                "because erasing the image's departure from the model's prior degrades the "
                "autoregressive conditioning of the whole neighbourhood. Not merely the free "
                "corrector losing adaptation: the corrector is worth 2,362 B in total and "
                "the shortfall is 10,694.7 B, 4.5x larger"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
        EmpiricalAnchor(
            anchor_id="cf2_toward_rung4_u12_20260820",
            measurement_utc="2026-08-20T00:00:00Z",
            inputs={
                "direction": TOWARD_ARGMAX,
                "modelled_bits_per_token": u12_modelled,
                "modelled_bytes_saved": 2546.1,
                "p_max_threshold": 0.9997559,
                "flips": 1440,
            },
            predicted_output={
                "realized_bits_per_token": predict_realized_bits_per_token(
                    u12_modelled, TOWARD_ARGMAX
                )
            },
            empirical_output={
                "realized_bits_per_token": u12_realized,
                "realized_bytes_saved": -37,
                "realized_over_modelled": TOWARD_TRUST_VS_MODEL_U12,
                "second_order_recapture_fraction": second_order_recapture_fraction(
                    -37, 2546.1
                ),
            },
            residual=abs(
                predict_realized_bits_per_token(u12_modelled, TOWARD_ARGMAX) - u12_realized
            ),
            source_artifact=".omx/research/ddm_fs2_rc4_drop_carrier_resolve_20260820.md",
            measurement_method=(
                "ddm_fs2's SECOND threshold, u = 12.0, 6.3x sparser in flip count -- encoded "
                "specifically to test whether 0.0872 is a property of the ladder or of one "
                "row. The substitution does not merely under-deliver, it COSTS 37 bytes: "
                "realized/modelled -0.0145, recapture above 100%. Two thresholds spanning "
                "6.3x, both refused, monotone adverse -- which is what scopes law (a) at "
                "FORMULATION rather than INSTANCE. Both bits/token here are derived in code "
                "from the memo's own (bytes, flips) pairs, never typed. This anchor carries "
                "the equation's LARGEST residual on purpose: predicting it with the u=7.75 "
                "factor is off by ~1.44 b/tok, which is the recorded evidence that the "
                "toward-direction factor does not transfer ACROSS thresholds -- only its "
                "sign and order of magnitude do"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
    )
    return CanonicalEquation(
        equation_id="token_rate_model_direction_dependence_v1",
        name="Token-rate model direction dependence",
        one_line_summary=(
            "the -log2 p model's trust factor is DIRECTION-dependent: ~0.92x realized/actual "
            "away from the argmax, ~0.09x realized/modelled toward it (91.3% recapture)"
        ),
        latex_form=(
            r"\frac{b_{\mathrm{real}}}{b_{\mathrm{model}}} = "
            r"\begin{cases} 0.922\!-\!0.927 & \text{away from } \arg\max \\ "
            r"0.087 \ (\to -0.015) & \text{toward } \arg\max \end{cases}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_cf2_token_price_laws_20260821:"
            "predict_realized_bits_per_token"
        ),
        domain_of_validity={
            "included": [
                "the rc2/br1-lineage HPAC token field under its shipped autoregressive coder",
                "jg3-class seg edits that move tokens AWAY from the model argmax "
                "(Series B, vs the ACTUAL flat price 4.1379)",
                "confidence-threshold drops that write the model argmax, i.e. TOWARD it "
                "(vs the MODELLED credit)",
                "sizing a token-field lever BEFORE paying for a real re-encode",
            ],
            "excluded": [
                "any use of 4.718 as a price or a trust-factor denominator -- it is jg1's "
                "LogitPrice RANKER, ordering only, per its own docstring",
                "mixing the two series: away ratios are vs the actual price, toward ratios "
                "vs the model; they answer different questions and their range is not one range",
                "substituting this factor for a real re-encode when the decision is being "
                "MADE -- the consumer rule is price by REAL re-encode, always",
                "predicting one toward-direction threshold from another: the ladder is "
                "monotone adverse (0.0872 at u=7.75, -0.0145 at u=12.0) and the u=12 "
                "anchor's ~1.44 b/tok residual is the recorded proof it does not transfer",
                "another token field, coder, selector, or body without re-measurement",
                "per-pair attribution: the jg3 anchor measured 10.277 B of context bleed "
                "into pairs that were never edited",
            ],
            "authority": _AXIS,
            "verdict_scope": "FORMULATION",
        },
        units_in={
            "modelled_bits_per_token": "bits per changed token",
            "direction": "enum {away_from_argmax, toward_argmax}",
        },
        units_out={"realized_bits_per_token": "bits per changed token"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            "max_anchor_residual": max(a.residual for a in anchors)
        },
        last_calibration_utc="2026-08-21T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments.ddm_fs2_drop_ladder",
            "experiments.ddm_fs2_jg5_on_candidate",
            "experiments.ddm_jg3_joint_solve",
            "tac.ddm_costate_organ",
        ),
        canonical_producers=(
            "experiments.ddm_jg2_tail_reencode",
            "experiments.ddm_fs3_jg5_real_price_reopen",
        ),
        provenance=provenance,
    )


def build_greedy_set_average_vs_marginal_price_v1() -> CanonicalEquation:
    """Law (b): a greedy set's AVERAGE price is not its MARGINAL member's price."""
    provenance = _provenance()
    add_marginal = bits_per_token(223, 300)
    remove_marginal = bits_per_token(664, 997)
    trimmed_average = bits_per_token(3487, 7657)
    anchors = (
        EmpiricalAnchor(
            anchor_id="cf2_marginal_add_round2_reopen_20260820",
            measurement_utc="2026-08-20T00:00:00Z",
            inputs={
                "set_average_bits_per_token": 2.6573,
                "delta_tokens": 300,
                "delta_bytes": 223,
                "set_tokens": 569,
            },
            predicted_output={
                "delta_bits_at_set_average": predict_reselection_delta_bits(300, 2.6573)
            },
            empirical_output={
                "marginal_bits_per_token": add_marginal,
                "delta_bits_realized": 223 * 8,
                "marginal_over_average": marginal_over_average_ratio(add_marginal, 2.6573),
                "set_yield_cells_per_token": 876 / 569,
                "marginal_yield_cells_per_token": 116 / 300,
                "yield_degradation_factor": (876 / 569) / (116 / 300),
            },
            residual=abs(predict_reselection_delta_bits(300, 2.6573) - 223 * 8),
            source_artifact=".omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md",
            measurement_method=(
                "ddm_fs3 round 2. The 38-pair reopen was BUILT and priced by a real "
                "re-encode: token stream 113,847 -> 114,070 B, archive 180,580 -> 180,803 B "
                "-- visibly LARGER than the body it came from -- over +300 tokens, so "
                "223*8/300 = 5.9467 b/tok against the 2.6573 set average the projection "
                "used. Ratio 2.2379. Yields move the other way in the same measurement: the "
                "38 pairs' shipped edits yield 876/569 = 1.5395 cells/token, the marginal "
                "sites 116/300 = 0.3867, a factor of 3.98. Row REFUSED at 16.36x the bar"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
        EmpiricalAnchor(
            anchor_id="cf2_marginal_removal_mirror_drop137_20260821",
            measurement_utc="2026-08-21T00:00:00Z",
            inputs={
                "trimmed_set_average_bits_per_token": trimmed_average,
                "delta_tokens": -997,
                "delta_bytes": -664,
                "trimmed_set_tokens": 7657,
                "pairs_tightened": 137,
            },
            # Signs are consistent with the inputs: a REMOVAL credits negative bits, so
            # both the prediction and the realized figure are negative here.
            predicted_output={
                "delta_bits_at_trimmed_average": predict_reselection_delta_bits(
                    -997, trimmed_average
                )
            },
            empirical_output={
                "marginal_removal_bits_per_token": remove_marginal,
                "delta_bits_realized": -664 * 8,
                "marginal_over_average": marginal_over_average_ratio(
                    remove_marginal, trimmed_average
                ),
                "ranker_relative_ratio": ranker_relative_ratio(trimmed_average),
            },
            residual=abs(
                predict_reselection_delta_bits(-997, trimmed_average) - (-664 * 8)
            ),
            source_artifact=(
                "/Volumes/APDataStore/pact/ddm_fs3/reencode/retained/"
                "S1_encode_fs3_drop137.json"
            ),
            measurement_method=(
                "The MIRROR of round 2, on the same admission cut from the other side: drop "
                "the 137 shipping pairs' over-admitted tokens. Real re-encode against the "
                "sha-receipted 113,847 B shipped-455 baseline (sha b9243abd2e38f9ae...): "
                "token stream -> 113,183 B, -997 tokens, so 664*8/997 = 5.3280 b/dropped-token "
                "-- close to the 5.9467 an ADDITION of marginal tokens cost and nowhere near "
                "the 2.6573 a whole-set removal credited, which is what settles the axis as "
                "average-vs-marginal rather than direction-of-operation. The surviving set "
                "averages 3487*8/7657 = 3.6432022985503463 b/tok and the receipt's own "
                "realized_over_modelled 0.7721920938004125 is that against the 4.718 RANKER. "
                "WARNING that travels with this number: the running credit fell monotonically "
                "7.1862 -> 6.1227 -> 5.9451 -> 5.5617 -> 5.3280 and came to rest 1.9% above "
                "its kill line -- a TREND EDGE, not a plateau. Rate leg survived its "
                "pre-registered falsifier; the ROW it came from (task #1176, the fs3 mirror) "
                "was TERMINALLY REFUSED on the measured pose leg (+3.590433e-02 S, stale "
                "carrier) -- the price stands, the candidate does not"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance=None,
        ),
    )
    return CanonicalEquation(
        equation_id="greedy_set_average_vs_marginal_price_v1",
        name="Greedy set average is not its marginal price",
        one_line_summary=(
            "a greedy-admitted set's average price is not its margin's: measured 2.24x the "
            "average at 3.98x less yield adding, 1.46x the average removing"
        ),
        latex_form=(
            r"b_{\mathrm{marg}} > \bar{b}_{\mathrm{set}} \ \wedge\ "
            r"y_{\mathrm{marg}} < \bar{y}_{\mathrm{set}}\ \ "
            r"(\Delta \mathrm{bits} = \Delta n \cdot b_{\mathrm{marg}})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_cf2_token_price_laws_20260821:"
            "predict_reselection_delta_bits"
        ),
        domain_of_validity={
            "included": [
                "jg3-class edit-configuration re-selection on the rc2-lineage token field",
                "any gain-ordered greedy selection whose price is being re-fitted, in "
                "either direction across the admission cut",
                "pricing a configuration DELTA: only the delta-tokens are bought or sold",
            ],
            "excluded": [
                "pricing a re-selection off a set average -- that is the error itself",
                "reading the arithmetic clause (average != marginal) as the empirical one; "
                "only the DEGRADATION DIRECTION is measured, and only on this formulation",
                "selections whose ranking is not gain-ordered, or whose price does not "
                "track rank -- the mechanism is the ordering, not the arithmetic",
                "carrying 5.3280 as a plateau value: it is a trend edge, still falling when "
                "the encode ended, resting 1.9% above its kill line",
                "another body, coder or selector without re-measurement",
            ],
            "authority": _AXIS,
            "verdict_scope": "FORMULATION",
        },
        units_in={
            "delta_tokens": "count of tokens added or dropped by the re-selection",
            "marginal_bits_per_token": "bits per marginal changed token",
        },
        units_out={"delta_bits": "bits of coded token stream"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            "max_anchor_residual": max(a.residual for a in anchors)
        },
        last_calibration_utc="2026-08-21T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments.ddm_jg3_joint_solve",
            "experiments.ddm_fs3_jg3_repriced_rescreen",
            "experiments.ddm_fs3_compose_reopen_candidate",
            "tac.ddm_costate_organ",
        ),
        canonical_producers=(
            "experiments.ddm_jg2_tail_reencode",
            "experiments.ddm_fs3_jg5_real_price_reopen",
        ),
        provenance=provenance,
    )


ALL_CF2_TOKEN_PRICE_BUILDERS = (
    build_token_rate_model_direction_dependence_v1,
    build_greedy_set_average_vs_marginal_price_v1,
)


def populate_cf2_token_price_laws(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> tuple[CanonicalEquation, ...]:
    """Append both laws through the locked registry helper (never a bare JSONL write)."""
    from tac.canonical_equations.registry import register_canonical_equation

    built = []
    for builder in ALL_CF2_TOKEN_PRICE_BUILDERS:
        equation = builder()
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=(
                "ddm_cf2 rv17 wave-2 carried item F2 (the equations leg); "
                "verdict_scope FORMULATION on the jg3-class/rc2-lineage token field; "
                f"axis {_AXIS}; score_claim=false"
            ),
        )
        built.append(equation)
    return tuple(built)
