# SPDX-License-Identifier: MIT
"""Canonical equation: two token arms compose SUB-additively on seg, in proportion to pair overlap.

THE MISTAKE THIS FORBIDS.  Two arms edit a token field.  Their edits touch **zero common cells**, so
the composition looks additive and the natural move is to add their measured deltas and fire.  ddm_sj1
built exactly that composition -- its 78 move-38 tokens onto rp1's move-39 field -- and MEASURED the
miss.  The naive sum is S **0.1376341479**; the projected composition is **0.1376357708**; the miss is
**+1.62e-06**, i.e. the composition is WORSE than the sum, and the seg term is the largest
contributor to it.  (The memo's printed decomposition does not CLOSE -- see
``named_decomposition_remainder_s`` -- so quote the terms, never their sum.)

WHY, AND IT IS NOT SUBTLE ONCE SAID.  A seg repair is a property of the **render**, not of the token.
A sister's disjoint token writes on a SHARED pair move the render underneath your repair, so your
repair can stop being one.  MEASURED: **80 repairs -> 74**.  All four losing pairs lie inside the
13-pair overlap (565 loses one outright; 237, 293 and 398 net zero).  The 29 non-overlapping pairs
are untouched.  **7.5 % of the arm's seg credit was spent on 4.6 % of its pairs.**

    Delta S_composed  =  Delta S_a + Delta S_b  +  seg_loss(overlap)  +  rate_anti_synergy  ,
    seg_loss > 0 and concentrated ENTIRELY on the shared pairs.

THE RATE LEG IS THE YIELD LAW SEEN IN A MIRROR.  The same 78 tokens cost **+47 B** on move 37's field
and **+50 B** on move 39's: 4.8205 -> 5.1282 bits/token, **+6.4 %**.  rp1's edits REMOVED learned
surprise from that field, so writing surprise back into it costs more per token.  Small beside the
73-85 % clawback rp1 measured on the removal leg, and pointed the other way.

AND THE CONTAINER CAN HAND IT BACK.  The tail splice landed at 180,236 B and the container search
closed it to **180,233 B** -- the +3 B of coder anti-synergy refunded.  fe1's lottery law
(``model_section_edit_container_break_fee_v1``) paying out in this arm's favour for once: sample, do
not search, and never assume the sign.

THE GATES THAT MADE THIS A MEASUREMENT.  Zero position collisions asserted, not assumed; no cell
outside either edit set moved; rp1's shipped field re-encoded through sj1's OWN loop emitted
119,568 B sha ``26e9eba5...`` byte-identical to rp1's retained rider body (two arms, two encoders,
one byte string); the composed twins were identical; pm2's pose-base gate passed at -0.08 %.

CONFIRMED ON THE EXACT AXIS.  The composition became pointer move 40:
**S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]**, sha ``986d536b...``, Delta -3.070463e-05
vs move 39 (1.54x the bar).  The projection missed the T4 print by +2.84e-06 -- so the sub-additivity
this equation prices is larger than the projection's own error, and the direction is the same.

THE OPERATING RULE.  Predict every composition of two token arms as sub-additive on seg in proportion
to their pair overlap, and **re-verify every shared pair by recompute -- never carry a repair across**.

Axis ``[contest-CUDA T4 n600 for the exact rows; exact archive bytes; frozen CPU SegNet for the
repair recount]``.  Registered by ddm_cons2 on 2026-09-11 from ddm_sj1's own section 31.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "token_edit_composition_subadditive_on_pair_overlap_v1"

_UTC = "2026-09-11T00:00:00Z"
_AXIS = "[contest-CUDA T4 n600 exact rows; exact archive bytes; frozen CPU SegNet repair recount]"
_LEDGER = ".omx/research/ddm_sj1_multipass_token_predistortion_20260905.md"
_MOVE40 = ".omx/research/ddm_sj1_t4_compose39_rp1_union_20260910_pointer_move_40_20260910.md"

# MEASURED (sj1 section 31a/31d): the composition's shape.
UNION_PAIRS = 282
OVERLAP_PAIRS = 13
OVERLAP_PAIR_IDS = (7, 118, 165, 167, 202, 237, 293, 398, 502, 546, 565, 584, 593)
POSITION_COLLISIONS = 0
REPAIRS_BEFORE = 80
REPAIRS_AFTER = 74
LOSING_PAIRS = 4
LOSING_PAIRS_INSIDE_OVERLAP = 4

# MEASURED (sj1 section 31f): the miss against the naive sum, decomposed.
NAIVE_SUM_S = 0.1376341479
PROJECTED_COMPOSITION_S = 0.1376357708
MISS_S = 1.62e-06
SEG_TERM_S = 5.086e-06
BYTE_TERM_S = -6.66e-07

# MEASURED (sj1 section 31c): the rate leg, same tokens, two fields.
TOKEN_COST_BYTES = {"move_37_field": 47, "move_39_field": 50}
BITS_PER_TOKEN = {"move_37_field": 4.8205, "move_39_field": 5.1282}
RATE_ANTI_SYNERGY_FRACTION = 0.064
TAIL_SPLICE_BYTES = 180_236
CONTAINER_CLOSED_BYTES = 180_233

# MEASURED (sj1 section 32): the EXACT row the composition became.
MOVE40_EXACT_S = 0.13763861019288715
MOVE40_ARCHIVE_BYTES = 180_233
MOVE40_DELTA_S = -3.070463e-05
PROJECTION_MISS_VS_T4_PRINT_S = 2.84e-06


def seg_credit_lost_fraction() -> float:
    """DERIVED from sj1's repair counts: the share of seg credit the overlap took (7.5 %)."""
    return (REPAIRS_BEFORE - REPAIRS_AFTER) / REPAIRS_BEFORE


def pair_overlap_fraction() -> float:
    """DERIVED: the share of the union's pairs that both arms touch (4.6 %)."""
    return OVERLAP_PAIRS / UNION_PAIRS


def composition_miss_s() -> float:
    """The MEASURED miss of sj1's compose-39 against the naive sum: +1.62e-06 S.

    Positive means the composition is WORSE than the sum.
    """
    return MISS_S


def named_decomposition_remainder_s() -> float:
    """The term sj1's printed decomposition leaves UNNAMED -- and it is bigger than the miss.

    The memo's table lists +5.086e-06 (six seg repairs lost on the overlap), -6.66e-07 (bytes:
    +47 measured vs +48 assumed) and then a third row whose value is the word "remainder"
    ("T4 seg carry + rounding"). The two named terms sum to +4.42e-06 against a measured miss of
    +1.62e-06, so the remainder is **-2.80e-06** -- 1.7x the miss in magnitude. That does not
    weaken the finding (the seg term is still the largest single contributor and still lives
    entirely on the overlap); it means the DECOMPOSITION is not closed, and nobody should quote
    the two named terms as if they added up to the miss.

    Two further restatements in the same memo that its own numbers do not support: "the seg loss
    is four times the byte effect" (5.086e-06 / 6.66e-07 = **7.64x**), and the memory index's
    "six seg repairs lost = +5.09e-6 (four fifths)" -- +5.086e-06 is not four fifths of a
    +1.62e-06 miss. Recorded here so a reader checks the arithmetic instead of inheriting it.
    """
    return MISS_S - (SEG_TERM_S + BYTE_TERM_S)


def predict_subadditive(overlap_pairs: int, union_pairs: int) -> bool:
    """True when a composition must be predicted sub-additive on seg -- i.e. whenever pairs are shared.

    Disjoint CELLS are not enough. sj1 measured zero position collisions and still lost six repairs,
    because a repair belongs to the render and the sister's writes move it.
    """
    if union_pairs <= 0:
        raise ValueError("union_pairs must be positive")
    if not 0 <= overlap_pairs <= union_pairs:
        raise ValueError("overlap_pairs must lie in [0, union_pairs]")
    return overlap_pairs > 0


def build_token_edit_composition_subadditive_on_pair_overlap_v1() -> CanonicalEquation:
    """Build the composition sub-additivity canonical equation (ddm_sj1 compose-39, n600)."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_LEDGER,
        reactivation_criteria=(
            "re-measure the seg leg of EVERY composition by recompute on the shared pairs -- this "
            "equation gives the sign and the locus, never a number to carry. It widens (support "
            "n = 1 composition) when a second cross-arm composition with a different overlap "
            "fraction is measured; a measured composition with shared pairs and NO repair loss "
            "would falsify the mechanism, not merely the coefficient"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="macOS CPU (frozen SegNet recount, exact encodes) + contest-CUDA T4 for move 40",
        captured_at_utc=_UTC,
    )

    anchor = EmpiricalAnchor(
        anchor_id="sj1_compose39_rp1_union_subadditive_seg_n600_20260910",
        measurement_utc="2026-09-10T00:00:00Z",
        inputs={
            "arm_a": "rp1's move 39 -- 473 argmax-neutral token changes over 253 pairs, 180,186 B",
            "arm_b": "sj1's move-38 subset -- 78 token writes over 42 pairs",
            "union_pairs": UNION_PAIRS,
            "overlap_pairs": OVERLAP_PAIRS,
            "overlap_pair_ids": list(OVERLAP_PAIR_IDS),
            "position_collisions": POSITION_COLLISIONS,
            "well_definedness": (
                "MEASURED, not assumed: zero position collisions, and the builder asserts no cell "
                "outside either edit set moved"
            ),
            "cross_arm_control": (
                "rp1's shipped field re-encoded through sj1's OWN loop emits 119,568 B sha "
                "26e9eba5209495a0... byte-identical to rp1's retained rider body"
            ),
        },
        predicted_output={
            "naive_sum_s": NAIVE_SUM_S,
            "assumption": "token-disjoint edits compose additively",
        },
        empirical_output={
            "projected_composition_s": PROJECTED_COMPOSITION_S,
            "miss_s": MISS_S,
            "decomposition_s": {
                "six_seg_repairs_lost_on_the_overlap": SEG_TERM_S,
                "bytes_47_measured_vs_48_assumed": BYTE_TERM_S,
                "unnamed_remainder_t4_seg_carry_and_rounding": -2.80e-06,
                "note": (
                    "the two NAMED terms sum to +4.42e-06 against a measured miss of +1.62e-06, "
                    "so the remainder the memo prints as the word 'remainder' is -2.80e-06 -- "
                    "1.7x the miss. The decomposition does NOT close; the seg term is still the "
                    "largest contributor and still lives entirely on the overlap"
                ),
                "memo_restatements_its_own_numbers_do_not_support": (
                    "'the seg loss is four times the byte effect' is 7.64x; the memory index's "
                    "'four fifths' does not hold for +5.086e-06 against a +1.62e-06 miss"
                ),
            },
            "repairs": {"before": REPAIRS_BEFORE, "after": REPAIRS_AFTER},
            "losing_pairs": LOSING_PAIRS,
            "losing_pairs_inside_the_overlap": LOSING_PAIRS_INSIDE_OVERLAP,
            "seg_credit_lost_fraction": 0.075,
            "pair_overlap_fraction": 0.046,
            "rate_leg": {
                "token_cost_bytes": TOKEN_COST_BYTES,
                "bits_per_token": BITS_PER_TOKEN,
                "anti_synergy_fraction": RATE_ANTI_SYNERGY_FRACTION,
                "mechanism": (
                    "the yield law's mirror -- writing surprise into a field surprise was REMOVED "
                    "from costs more per token"
                ),
                "container_refund": {
                    "tail_splice_bytes": TAIL_SPLICE_BYTES,
                    "closed_bytes": CONTAINER_CLOSED_BYTES,
                    "note": "the +3 B was handed back; fe1's lottery is two-sided",
                },
            },
            "exact_row_the_composition_became": {
                "pointer_move": 40,
                "s": MOVE40_EXACT_S,
                "archive_bytes": MOVE40_ARCHIVE_BYTES,
                "delta_s_vs_move_39": MOVE40_DELTA_S,
                "archive_sha256_prefix": "986d536b",
                "projection_miss_vs_t4_print_s": PROJECTION_MISS_VS_T4_PRINT_S,
            },
        },
        # |projected composition - naive sum| / naive sum: the relative size of the miss.
        residual=1.179e-05,
        source_artifact=_LEDGER,
        measurement_method=(
            "rebuild the composed field, re-encode it through the shipped coder, and RECOMPUTE "
            "every repair through the receiver's own render_frame1 at batch 1 against the frozen "
            "CPU SegNet -- never carrying a repair from either parent"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Token-edit compositions are sub-additive on seg in proportion to pair overlap",
        one_line_summary=(
            "Two token arms with ZERO cell collisions lost 6 of 80 seg repairs, all four losing "
            "pairs inside their 13-pair overlap: +5.086e-06 S of a +1.62e-06 miss"
        ),
        latex_form=(
            r"\Delta S_{a\oplus b} = \Delta S_a + \Delta S_b + \varepsilon_{\mathrm{seg}}"
            r"(\mathcal{P}_a \cap \mathcal{P}_b) + \varepsilon_{\mathrm{rate}},\quad "
            r"\varepsilon_{\mathrm{seg}} > 0 \ \text{and supported on the shared pairs}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.sj1_composition_subadditive_on_pair_overlap_20260911"
            ":predict_subadditive"
        ),
        domain_of_validity={
            "object": (
                "two token-field edit sets composed onto one field, cell-disjoint, each with its "
                "own measured seg and rate legs, under the shipped coder and receiver"
            ),
            "generalizes_to": (
                "any two actuators whose credit is a property of the RENDER rather than of the "
                "symbol they write -- the sign and the locus (the shared pairs) transfer"
            ),
            "excluded": (
                "the NUMBERS. 6-of-80 and 7.5 % are one composition's; do not carry them to a "
                "different overlap, a different field, or cell-COLLIDING edits (a different and "
                "worse case this equation does not cover). Also excluded: the pose leg, which was "
                "near-neutral here (+6.887478e-07) and is not claimed to be sub-additive"
            ),
            "support": (
                "n = 1 cross-arm composition, n600, with a byte-exact cross-arm control and an "
                "EXACT contest-CUDA T4 row (move 40) confirming sign and scale"
            ),
            "verdict_this_priced": (
                "the composition was still WORTH firing -- it became pointer move 40 at 1.54x the "
                "bar. The law is about PREDICTION, not admission: a composition whose margin is "
                "thinner than its overlap's seg loss must be re-measured before it is fired"
            ),
        },
        units_in={"overlap_pairs": "count", "union_pairs": "count"},
        units_out={
            "predict_subadditive": "boolean -- must the seg leg be recomputed?",
            "composition_miss_s": "score units, positive = composition worse than the naive sum",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "naive_additivity_relative_error": 1.179e-05,
            "seg_term_over_byte_term": 7.64,
            "named_decomposition_remainder_abs_s": 2.80e-06,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _MOVE40,
            "experiments/ddm_rp1_rebase.py",
            ".omx/research/ddm_gs3_gestalt_after_submission_20260903.md",
        ),
        canonical_producers=(
            "experiments/ddm_sj1_compose_segcheck.py",
            "experiments/ddm_rp1_segcheck.py",
        ),
        provenance=provenance,
    )
