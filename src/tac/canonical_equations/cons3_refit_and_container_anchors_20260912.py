# SPDX-License-Identifier: MIT
"""Two anchors ddm_tmx1 measured, appended to the equations that already own their laws.

ddm_cons3 registers no new canonical equation here on purpose.  Both facts below belong to laws
that exist, and a duplicate law is worse than a missing one: two ids for one mechanism means a
future arm cites whichever it finds and neither gets the other's anchors.

ANCHOR 1 -> ``hpr1_counted_section_refit_debt_v1`` (n = 1 becomes n = 2, one positive one NEGATIVE).

    hpr1's own bar for calling the refit debt a law was "a SECOND section's refit", and its ranked
    queue named the exact test: the tc1/tc3 tail mixer's 40 counted int8 weights, fit to a field
    that has moved, where 60 B of state prices the 118,511 B stream.  ddm_tmx1 ran that test and it
    LOSES.  Four independent fits, two bases with different HPAC priors, four held-out folds and
    SIX exact 600-frame encodes agree: every refit costs bytes and none saves any.

    The decomposition still holds -- and reading it is the whole finding:

        section          Delta model   Delta tail   Delta B
        hpac  (12,262 B fitted state)      +351      -1,238      -887   PAYS
        mixer (40 B fitted state)            0         +20        +20   LOSES

    ``Delta model`` is ZERO for the mixer because its 60 B rider length is FIXED by the shipped
    schema (``variant(1) + 40 int8 + 19 B geometry``; ``LaneMixer.__init__`` refuses anything but
    ``41 + FORMAT.size``): a refit may change the 40 VALUES and nothing else, it cannot grow.  So
    ``Delta B = Delta tail`` alone, and the tail went the wrong way.

    The surrogate says why, and it is not noise.  Paired against the shipped weights on a 1-in-32
    systematic sample (3,686,368 of 117,964,800 symbols), the full-sample fit gains -90.2 +- 25.4 B
    (move 47) and -98.2 +- 23.9 B (move 48) IN SAMPLE at |t| > 3.5, and reverses to roughly +55 B
    held out in all four fold directions (t +2.16 .. +2.29).  The real coder on the full field then
    prices those same weights at +22 B and +20 B: a sample-to-field transfer loss of 112 B and
    118 B, twice, on two independent bases.  A 40-parameter fit reliably extracts about -94 B of
    fold-specific structure and pays about +55 B for it elsewhere.

ANCHOR 2 -> ``model_section_edit_container_break_fee_v1`` (its EXCLUSION clause, now measured).

    That law's domain already excludes "sections that are NOT range-coded under a match-finding
    compressor -- the mechanism is the re-randomised payload".  ddm_tmx1 measured that exclusion
    instead of reasoning it: the archive holds ONE member, ``p``, at ``compress_type 0`` (STORE),
    with exactly 100 B of ZIP overhead, and the 60 B rider's length is fixed, so a token-stream
    length change reaches archive bytes ONE-FOR-ONE with no re-segmentation and no recompression.
    Every exact row in the arm confirms it by arithmetic: stream +203 B -> archive +203 B, stream
    +22 -> +22, +155 -> +155, +20 -> +20.  The standing +-34.8 B container-break lottery therefore
    does NOT govern this axis, and these deltas are exact deterministic byte counts, not draws.
    (The lottery still governs edits that move a brotli'd member -- which is what hpr1's refit did
    and this one does not.)

Axis for both: ``[macOS-CPU advisory; exact bytes, scorer-free]`` -- exact archive bytes from the
real coder and the real receiver loop, twin encodes agreeing, decoded field byte-identical to the
shipped field ``a92e7d90...``, with NO scorer run.  Registered by ddm_cons3 on 2026-09-12 from
ddm_tmx1's memo, read-only; cons3 launched nothing and measured nothing.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

_UTC = "2026-09-12T00:00:00Z"
_AXIS = "[macOS-CPU advisory; exact bytes, scorer-free]"
_TMX1_MEMO = ".omx/research/ddm_tmx1_refit_tail_mixer_tc1_on_current_field_20260911.md"
_TMX1_FALSIFIERS = ".omx/research/ddm_tmx1_falsifiers_preregistered_20260911.md"

REFIT_ANCHOR_ID = "tmx1_tail_mixer_refit_negative_exact_bytes_20260911"
CONTAINER_ANCHOR_ID = "tmx1_store_member_stream_to_archive_one_for_one_20260911"

REFIT_EQUATION_ID = "hpr1_counted_section_refit_debt_v1"
CONTAINER_EQUATION_ID = "model_section_edit_container_break_fee_v1"

# MEASURED (ddm_tmx1): the exact rows, real coder over all 117,964,800 symbols.
CONTROL47_STREAM_BYTES = 118_511
CONTROL47_ARCHIVE_BYTES = 179_359
CONTROL48_STREAM_BYTES = 118_896
CONTROL48_ARCHIVE_BYTES = 179_111
REFIT47_STRIDE4_DELTA_B = 203
REFIT47_FULL_DELTA_B = 22
REFIT48_STRIDE4_DELTA_B = 155
REFIT48_FULL_DELTA_B = 20

# MEASURED: the fitted state each section holds, and the stream both condition.
HPAC_FITTED_STATE_BYTES = 12_262
MIXER_FITTED_STATE_BYTES = 40
MIXER_RIDER_BYTES = 60
CONDITIONED_STREAM_BYTES = 118_511

# MEASURED: the surrogate, paired, 1-in-32 systematic sample.
SAMPLE_SYMBOLS = 3_686_368
TOTAL_SYMBOLS = 117_964_800
IN_SAMPLE_GAIN_B = {"move47": -90.2, "move48": -98.2}
IN_SAMPLE_SE_B = {"move47": 25.4, "move48": 23.9}
HELD_OUT_LOSS_B = {
    "move47_even": 53.6,
    "move47_odd": 54.2,
    "move48_even": 59.0,
    "move48_odd": 54.2,
}

# MEASURED: the fire bar this family had to clear.
ADMIT_DELTA_S = -2e-5
EXCHANGE_RATE_S_PER_BYTE = 6.658589531221714e-07
ADMIT_DELTA_B = -30.04

# MEASURED: the container facts that make stream bytes reach archive bytes 1:1.
ARCHIVE_MEMBERS = 1
ARCHIVE_MEMBER_NAME = "p"
ARCHIVE_COMPRESS_TYPE = 0  # STORE
ZIP_OVERHEAD_BYTES = 100
CONTAINER_LOTTERY_SD_BYTES = 34.8


def transfer_loss_bytes(in_sample_gain_b: float, exact_delta_b: float) -> float:
    """How much of a fitted gain the real field takes back: measured 112 B and 118 B."""
    return exact_delta_b - in_sample_gain_b


def fitted_capacity_ratio() -> float:
    """The quantity tmx1 proposes as the RANKING one: state bytes, hpac against the mixer."""
    return HPAC_FITTED_STATE_BYTES / MIXER_FITTED_STATE_BYTES


def build_tmx1_low_capacity_refit_negative_anchor() -> EmpiricalAnchor:
    """The second section's refit that hpr1's own bar demanded -- and it LOSES."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_TMX1_MEMO,
        reactivation_criteria=(
            "the refit debt is now n = 2 with one POSITIVE (hpac, 12,262 B of fitted state, "
            "-887 B) and one NEGATIVE (the tail mixer, 40 B of fitted state, +20 B) instance, so "
            "it is NOT a general refit predictor and must not be used to authorise a refit rung. "
            "tmx1's proposal -- rank refit rungs by the FITTED CAPACITY a section holds against "
            "the noise in the objective that fits it, never by the byte mass it conditions -- is "
            "itself at n = 2 and needs a THIRD section's refit before it is a law. The `semantic` "
            "member (29,862 B) is the ranked candidate and its staleness still cannot be dated"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="macOS CPU, real coder + real receiver loop, 600 frames, twin encodes",
        captured_at_utc=_UTC,
    )
    return EmpiricalAnchor(
        anchor_id=REFIT_ANCHOR_ID,
        measurement_utc=_UTC,
        inputs={
            "section": (
                "the tail coder's mixer weights: tc1's 35 (K=5 winning classes x F=7 features) "
                "plus tc3's 5 lane coefficients, shipped as a 60 B rider "
                "(variant(1) + 40 int8 + 19 B geometry), rider sha 76f10171e42d27e6..."
            ),
            "fitted_on": (
                "tc1's 35 on the move-33/34 body composed at move 36; tc3's 5 on move 40 with the "
                "35 FROZEN; the field last moved at move 43 (sj1 pass 6) and is a92e7d90... today"
            ),
            "why_delta_model_is_zero": (
                "the rider's length is FIXED by the shipped schema -- LaneMixer.__init__ refuses "
                "anything but 41 + FORMAT.size -- so a refit may change the 40 VALUES and nothing "
                "else. It cannot grow, so Delta model = 0 and Delta B = Delta tail alone"
            ),
            "search": (
                "four independent fits (stride-4 and full-sample, on move 47 and move 48) over a "
                f"1-in-32 systematic sample ({SAMPLE_SYMBOLS:,} of {TOTAL_SYMBOLS:,} symbols, "
                "hashed on (frame, position) so it does not follow the coding-group lattice), "
                "scored with the coder's own ideal length, replayed BIT-IDENTICALLY to the "
                "shipped mixer (5,000 random rows, exact float32 equality)"
            ),
            "controls": (
                "two live-loop controls re-encoded their bases byte-identically: control47 "
                f"{CONTROL47_STREAM_BYTES:,} B stream / {CONTROL47_ARCHIVE_BYTES:,} B archive at "
                f"Delta 0, control48 {CONTROL48_STREAM_BYTES:,} / "
                f"{CONTROL48_ARCHIVE_BYTES:,} at Delta 0, twins identical, decoded field a92e7d90..."
            ),
        },
        predicted_output={
            "staleness_audit_rank": (
                "#1 on the board by the audit's ranking rule -- '60 B of state prices 118,511 B' "
                "-- i.e. the largest expected repayment of any queued refit rung"
            ),
            "fire_bar": (
                f"Delta S < {ADMIT_DELTA_S} <=> Delta B < {ADMIT_DELTA_B} B at "
                f"{EXCHANGE_RATE_S_PER_BYTE} S/B"
            ),
            "pre_registered_falsifier_F3": (
                "the refit must beat the shipped weights on the held-out fold in BOTH directions"
            ),
        },
        empirical_output={
            "exact_delta_b": {
                "refit47_stride4": REFIT47_STRIDE4_DELTA_B,
                "refit47_full_sample": REFIT47_FULL_DELTA_B,
                "refit48_stride4": REFIT48_STRIDE4_DELTA_B,
                "refit48_full_sample": REFIT48_FULL_DELTA_B,
            },
            "delta_model_bytes": 0,
            "delta_tail_bytes": REFIT48_FULL_DELTA_B,
            "delta_b_bytes": REFIT48_FULL_DELTA_B,
            "in_sample_paired_gain_bytes": IN_SAMPLE_GAIN_B,
            "held_out_loss_bytes": HELD_OUT_LOSS_B,
            "sample_to_field_transfer_loss_bytes": [112, 118],
            "falsifier_F3": "FIRES -- four folds out of four go the wrong way",
            "falsifier_F4_surrogate_sign": (
                "does NOT fire: every surrogate sign matches the exact encode, so the surrogate is "
                "a sound ranker at this object; there is simply nothing to rank into a win"
            ),
            "fitted_state_bytes": {
                "hpac_pays": HPAC_FITTED_STATE_BYTES,
                "mixer_loses": MIXER_FITTED_STATE_BYTES,
            },
            "conditioned_stream_bytes_is_the_same_for_both": CONDITIONED_STREAM_BYTES,
            "mixer_whole_measured_value_when_fresh_bytes": 628,
            "verdict": "DO NOT FIRE -- nothing staged, nothing sealed, no candidate claimed",
            "support_after_this_anchor": "n = 2 sections: one positive, one negative",
        },
        # The ranking rule predicted this rung would repay MORE than any other; the measurement
        # returned the opposite sign at every one of six exact encodes. DERIVED: the best exact row
        # (+20 B) against the bar it had to beat (-30.04 B), normalised by the bar.
        residual=abs(REFIT48_FULL_DELTA_B - ADMIT_DELTA_B) / abs(ADMIT_DELTA_B),
        source_artifact=_TMX1_MEMO,
        measurement_method=(
            "six exact 600-frame encodes with the real coder over all 117,964,800 symbols, twin "
            "encodes agreeing, decoded field byte-identical to a92e7d90..., against two "
            "byte-identical live-loop controls; plus a paired 1-in-32 surrogate with frame-parity "
            "held-out folds whose tolerance is three standard errors DERIVED from the sample"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_tmx1_store_member_container_scope_anchor() -> EmpiricalAnchor:
    """The exclusion clause, measured: a STORE member passes stream bytes through 1:1."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_TMX1_MEMO,
        reactivation_criteria=(
            "re-measure the moment the archive stops holding exactly one STORE member, or the "
            "rider's length stops being fixed by the schema. The fee law returns in full for any "
            "edit that moves a brotli'd member -- the hpac section is the live example, and "
            "hpr1's refit paid there while this arm's did not"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="macOS CPU, real coder, exact ZIP byte accounting",
        captured_at_utc=_UTC,
    )
    return EmpiricalAnchor(
        anchor_id=CONTAINER_ANCHOR_ID,
        measurement_utc=_UTC,
        inputs={
            "container": (
                f"the candidate archive holds {ARCHIVE_MEMBERS} member, '{ARCHIVE_MEMBER_NAME}', "
                f"compress_type {ARCHIVE_COMPRESS_TYPE} (STORE), with exactly "
                f"{ZIP_OVERHEAD_BYTES} B of ZIP overhead"
            ),
            "edit_family": (
                "40 int8 mixer values inside a 60 B rider whose length the schema fixes: no "
                "re-segmentation, no recompression, no length change in the rider itself"
            ),
            "law_clause_under_test": (
                "the fee law's own EXCLUSION -- 'sections that are NOT range-coded under a "
                "match-finding compressor need not pay it' -- which was reasoned, not measured"
            ),
        },
        predicted_output={
            "if_the_lottery_governed": (
                f"an edit's archive delta would be a draw with sd {CONTAINER_LOTTERY_SD_BYTES} B "
                "around its stream delta, and a +20 B stream change could not be called"
            ),
        },
        empirical_output={
            "stream_to_archive_delta_pairs": {
                "refit47_stride4": [REFIT47_STRIDE4_DELTA_B, REFIT47_STRIDE4_DELTA_B],
                "refit47_full_sample": [REFIT47_FULL_DELTA_B, REFIT47_FULL_DELTA_B],
                "refit48_stride4": [REFIT48_STRIDE4_DELTA_B, REFIT48_STRIDE4_DELTA_B],
                "refit48_full_sample": [REFIT48_FULL_DELTA_B, REFIT48_FULL_DELTA_B],
            },
            "pass_through": "one-for-one on every row; the twins prove it reproducible",
            "consequence": (
                "these are exact deterministic byte counts, NOT draws; the standing "
                f"+-{CONTAINER_LOTTERY_SD_BYTES} B container-break lottery does not govern this "
                "axis and must not be quoted as a margin on it"
            ),
            "where_the_lottery_still_governs": (
                "any edit that moves a brotli'd member -- the hpac section, which is what hpr1's "
                "refit moved and this arm's did not"
            ),
        },
        residual=0.0,
        source_artifact=_TMX1_FALSIFIERS,
        measurement_method=(
            "exact ZIP member census on the candidate archive plus four paired "
            "(stream bytes, archive bytes) rows from real 600-frame encodes with twin agreement"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
