# SPDX-License-Identifier: MIT
"""Canonical equation: the CONTAINER-BREAK FEE on an edited RX1 model section, and the
encoder-side container search that pays it off (ddm_fe1, 2026-09-09).

THE OBJECT.  On the live pass-3 body the semantic model section is stored as
``brotli(CK2-2-plane-interleave(RC1 rider))`` -- ``runtime/residual_archive.py:193, 237, 248``,
RX1 ``reserved = 0x7a`` -- and the RC1 rider's payload is a RANGE-CODED bitstream
(``cpr1/rc1_adaptive_model_sections.py`` ``_encode_groups``).  Range coding has no
synchronisation points: changing ONE symbol re-randomises every bit after it.  The brotli layer
above therefore loses whatever matches it had found on the shipped bytes, and it loses them
whether the edit touched one code or two hundred.

THE MEASUREMENT (ddm_fe1, exact re-encodes, scorer-free).  Random single-code edits to
``frame_embed.weight`` (600x8, 3-bit codes) at N = 1, 2, 5, 10, 25, 50, 72, 100, 150, 200, ten
draws each, priced as the archive's own section bytes:

  N changed codes    at the SHIPPED container shape     with a container SEARCH
        1                    +62.5 B                            -7.2 +- 25.4 B
       72                    +77.3 B                           +21.0 +- 29.5 B
      200                   +100.6 B                           +24.8 +- 24.5 B

  least squares over all ten N:
      shipped shape:  dB = 0.2134*N + 58.89   (rms 7.09 B)
      searched:       dB = 0.1677*N -  1.92   (rms 5.26 B)

THE LAW.  The cost of editing a range-coded model section splits into

    dB(N) = BREAK_FEE * [the container shape is not re-searched] + MARGINAL_B_PER_CODE * N

with a MEASURED break fee of **58.9 B** and a marginal of only **0.17-0.21 B per changed code**.
The fee is not a length effect: length-preserving code changes still cost +40 B at the shipped
shape.  It is the loss of brotli's purchase on a re-randomised payload.

WHY IT IS RECOVERABLE.  Brotli streams are self-describing and the CK2 interleave rides in the
RX1 ``reserved`` byte, so the brotli quality, the window size and the interleave are
ENCODER-ONLY choices -- the receiver is never told which were used and its decode path is
unchanged.  Searching q in {9, 10, 11} x lgwin in {16, 18, 20, 22, 24} x {ck2, plain} and
shipping the smallest recovers **69.7 B at N = 1** and **56.3 B at N = 72**.  The live body's own
shape is ``(ck2, 11, 24)`` -- pinned by BYTE identity, because (ck2, 11, 16) compresses the
shipped rider to exactly the same 30,246 B and they are DIFFERENT bytes (brotli records its
window size), a trap that made this arm's first null build reproduce the right byte COUNT with
30,129 wrong bytes.  It is the best shape for the SHIPPED codes and a poor one for any edit.

WHY IT MATTERS.  A one-cell seg repair is worth -8.4771e-07 S and one archive byte costs
+6.6586e-07 S, so an unsearched +70 B fee demands ~55 repaired cells before an edit breaks even
and a searched fee demands ~17 at N = 72.  ddm_fe1's frame-embedding arm was priced BOTH ways:
at the shipped shape it reads dead, and searched it reads live.  Any arm that edits the semantic
or hpac model section on this body -- ddm_pc2, ddm_rc2, any successor to ddm_hp4 -- must price
with the container search or it will kill a live axis on an encoder-side artefact.

Producer: ``experiments/ddm_fe1_pose_price.py rate-law`` via
``.omx/research/ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md``.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "model_section_edit_container_break_fee_v1"

_UTC = "2026-09-09T13:05:00Z"
_AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"
_LEDGER = ".omx/research/ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md"
_CHARTER = (
    ".omx/research/charters/ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md"
)
_PRODUCER = "experiments/ddm_fe1_pose_price.py"
_RECEIPT = "ddm_fe1_frame_embedding_predistortion/admission/RATE_LAW.json"

#: The live pass-3 body this was measured on.
LIVE_ARCHIVE_SHA256 = (
    "06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3"
)
LIVE_ARCHIVE_BYTES = 181_645
SHIPPED_SEMANTIC_SECTION_BYTES = 30_246
SHIPPED_RC1_STREAM_BYTES = 31_792
SHIPPED_CONTAINER_SHAPE = ("ck2", 11, 24)
RX1_RESERVED = 0x7A

#: Least-squares fits over N in {1, 2, 5, 10, 25, 50, 72, 100, 150, 200}, ten draws each.
BREAK_FEE_BYTES = 58.89
SHIPPED_SHAPE_MARGINAL_B_PER_CODE = 0.2134
SHIPPED_SHAPE_FIT_RMS_BYTES = 7.09
SEARCHED_INTERCEPT_BYTES = -1.92
SEARCHED_MARGINAL_B_PER_CODE = 0.1677
SEARCHED_FIT_RMS_BYTES = 5.26

#: Encoder-only container choices; none of them is transmitted.
CONTAINER_QUALITIES = (9, 10, 11)
CONTAINER_LGWINS = (16, 18, 20, 22, 24)
CONTAINER_INTERLEAVES = ("ck2", "plain")

S_PER_BYTE = 25.0 / 37_545_489
S_PER_SEG_CELL = 100.0 / (600 * 384 * 512)


def section_delta_bytes(changed_codes: int, *, container_search: bool = True) -> float:
    """Predicted archive-section growth for ``changed_codes`` edited model codes.

    With ``container_search`` the encoder re-picks the brotli shape and the CK2 interleave
    for the edited bytes, which is legal because none of those choices is transmitted; without
    it the caller pays the full break fee for keeping a shape chosen for different bytes.
    """
    codes = float(changed_codes)
    if codes < 0:
        raise ValueError("changed_codes must not be negative")
    if codes == 0:
        return 0.0
    if container_search:
        return SEARCHED_MARGINAL_B_PER_CODE * codes + SEARCHED_INTERCEPT_BYTES
    return SHIPPED_SHAPE_MARGINAL_B_PER_CODE * codes + BREAK_FEE_BYTES


def container_search_recovers_bytes(changed_codes: int) -> float:
    """Bytes the encoder-side container search buys back at this edit size."""
    return section_delta_bytes(changed_codes, container_search=False) - section_delta_bytes(
        changed_codes, container_search=True
    )


def cells_to_break_even(changed_codes: int, *, container_search: bool = True) -> float:
    """Seg cells an edit must repair before its own section bytes are paid for."""
    delta = section_delta_bytes(changed_codes, container_search=container_search)
    return delta * S_PER_BYTE / S_PER_SEG_CELL


def _anchor_break_fee() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="fe1_semantic_section_container_break_fee_58p6_bytes_20260909",
        measurement_utc=_UTC,
        inputs={
            "archive_sha256": LIVE_ARCHIVE_SHA256,
            "archive_bytes": LIVE_ARCHIVE_BYTES,
            "section": "RX1 semantic, brotli(CK2-interleave(RC1 rider)), 30,246 B shipped",
            "edit": "random single-code changes to frame_embed.weight (600x8, 3-bit codes), steps in {-1, +1, +2}",
            "N": [1, 2, 5, 10, 25, 50, 72, 100, 150, 200],
            "repeats": 10,
            "seed": 20260909,
            "producer": f"{_PRODUCER} rate-law",
            "receipt": _RECEIPT,
        },
        predicted_output={
            "naive_prior": "an edit costs bits proportional to how many codes moved",
        },
        empirical_output={
            "shipped_shape_fit": "dB = 0.2134*N + 58.89 B (rms 7.09)",
            "shipped_shape_at_N1": 62.5,
            "shipped_shape_at_N200": 100.6,
            "length_preserving_edits_still_cost_bytes": 40,
            "reading": (
                "the cost is a fixed container-break fee, not a per-code price: 1 code and "
                "200 codes cost the same to within the draw-to-draw spread, because the RC1 "
                "payload is range-coded and one changed symbol re-randomises every bit after it"
            ),
        },
        residual=float(SHIPPED_SHAPE_FIT_RMS_BYTES),
        source_artifact=_LEDGER,
        measurement_method=(
            "exact re-encode of the whole semantic section through the shipped RC1 coder and "
            "brotli container; the shipped codes reproduce the archive header's 30,246 B"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "a body whose model section is NOT range-coded, or whose container is not "
                "brotli, re-opens the fee -- the mechanism is specific to a range-coded "
                "payload under a match-finding compressor"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _anchor_one_sample_lottery() -> EmpiricalAnchor:
    """The refinement ddm_fe1's ITEM 5 measured: the delta does not accumulate."""
    return EmpiricalAnchor(
        anchor_id="fe1_container_break_delta_is_a_one_sample_lottery_20260909",
        measurement_utc="2026-09-09T12:45:00Z",
        inputs={
            "archive": "pc2 move 34, 181,373 B, sha e138ee09...",
            "candidate_moves": "133 seg-NEUTRAL single frame_embed code moves, one per pair, each measured by the n600 realized search to leave the pair's SegNet argmax flip count exactly unchanged",
            "builds": "250 REAL archive builds with the container searched on every one: 133 singles, then 117 greedy-accumulation and neighbourhood builds",
            "producer": "experiments/ddm_fe1_item5_neutral_rate.py run/verify",
        },
        predicted_output={
            "prior": "a greedy search over 133+ reducing moves compounds and reaches <= -150 B",
        },
        empirical_output={
            "single_move_distribution": "mean +0.1 B, sd 34.8 B; min -61, p5 -50, median -1, max +98; 67 of 133 reduce",
            "best_single": -61,
            "best_subset_after_117_greedy_builds": -61,
            "moves_in_the_winning_set": 1,
            "neutrality_verified": "pair 331 re-rendered, 21 -> 21 flips, delta 0",
            "reading": (
                "the delta is a property of the PERTURBED PAYLOAD, not of the number of "
                "perturbations: a ONE-SAMPLE LOTTERY, not an additive budget. Adding a "
                "second move to a good draw RE-SAMPLES rather than compounds, and 117 "
                "builds conditioned on the best draw never beat it -- there is no "
                "gradient because there is no landscape. It also explains why a "
                "23-cell FiLM edit (-68 B) and a zero-distortion single move (-61 B) "
                "land so close: both are one ticket in the same lottery."
            ),
            "operational": (
                "do not build a byte-search arm on this axis; DO take one cheap sample -- "
                "price a handful of seg-neutral variants of whatever edit you are already "
                "making and ship the smallest"
            ),
        },
        residual=89.0,  # |predicted -150 B minus measured -61 B|
        source_artifact=_LEDGER,
        measurement_method="250 real archive builds on the live pointer, container searched on each; the winning move's seg-neutrality re-verified through the receiver's own render",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "a section whose coder has synchronisation points (so a symbol change "
                "does NOT re-randomise everything downstream) could have an additive "
                "structure this one lacks"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _anchor_container_search() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="fe1_container_search_recovers_63p5_bytes_at_n1_20260909",
        measurement_utc=_UTC,
        inputs={
            "shapes_searched": "q in {9,10,11} x lgwin in {16,18,20,22,24} x {ck2, plain} = 30",
            "transmitted": "none -- brotli streams are self-describing and CK2 rides in RX1 reserved bit 0x2",
            "shipped_shape": list(SHIPPED_CONTAINER_SHAPE),
            "receipt": _RECEIPT,
        },
        predicted_output={
            "prior": "the shipped container shape is the right shape for edited bytes too",
        },
        empirical_output={
            "searched_fit": "dB = 0.1677*N - 1.92 B (rms 5.26)",
            "recovered_at_N1": 69.7,
            "recovered_at_N72": 56.3,
            "searched_at_N1": -7.2,
            "searched_at_N72": 21.0,
            "cells_to_break_even_N72_searched": 16.5,
            "cells_to_break_even_N72_shipped_shape": 60.7,
            "reading": (
                "the shipped shape is optimal for the SHIPPED codes and poor for any edit of "
                "them; re-searching it is an encoder-only change the receiver never sees, and "
                "skipping it prices a model-section edit about 3x too high"
            ),
        },
        residual=float(SEARCHED_FIT_RMS_BYTES),
        source_artifact=_LEDGER,
        measurement_method=(
            "same exact re-encodes, minimum over the 30 container shapes; the SHIPPED codes' "
            "own minimum equals the archive header's 30,246 B, so the search is anchored"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "a receiver that pins the brotli parameters or the interleave would close the "
                "search; nothing in the live receiver does"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_model_section_edit_container_break_fee_v1() -> CanonicalEquation:
    """Build the container-break-fee law for edited RX1 model sections (ddm_fe1)."""
    fee = _anchor_break_fee()
    search = _anchor_container_search()
    lottery = _anchor_one_sample_lottery()
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Container-break fee on an edited range-coded model section -- an edit costs a "
            "fixed ~58.6 B, not a per-code price, and an encoder-side container search pays "
            "most of it back"
        ),
        one_line_summary=(
            "fe1: an edit costs 0.2134*N + 58.89 B at the shipped brotli/CK2 shape but only "
            "0.1677*N - 1.92 B re-searched; 69.7 B recovered at N=1, 56.3 B at N=72"
        ),
        latex_form=(
            r"\Delta B(N)=\underbrace{58.89}_{\text{break fee}}\cdot\mathbb{1}[\text{shape not re-searched}]"
            r"+m N,\quad m_{\text{shipped}}=0.2134,\ m_{\text{searched}}=0.1677;\quad"
            r"\text{cells to break even}=\Delta B\cdot\frac{25/37{,}545{,}489}{100/117{,}964{,}800}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.model_section_edit_container_break_fee_20260909"
            ":section_delta_bytes"
        ),
        domain_of_validity={
            "included": [
                "RX1 model sections stored as brotli(optional CK2 interleave(range-coded rider)) on the live pass-3 body 06c44dc4...",
                "the semantic section's RC1 rider, measured on frame_embed code edits at N = 1..200",
                "encoder-side container choice: brotli quality, lgwin, and the CK2 interleave, none of which is transmitted",
            ],
            "excluded": [
                "use as a d_seg / d_pose claim: this is a scorer-free byte law",
                "sections that are NOT range-coded under a match-finding compressor -- the mechanism is the re-randomised payload, so a raw or Huffman-coded section need not pay it",
                "the hpac section: same container family and the same rider magic, but the fee is NOT measured there (predicted to apply, owed as fe1 ITEM 2)",
                "extrapolation past N = 200 changed codes",
            ],
            "measurement_axis": [_AXIS],
            "result_type": "RATE law; it prices an edit, it does not authorise one",
            "sister_laws": [
                "ddm_hp4 -- frame-embedding CODING is closed (+167..+583 B for every predictor); this law is about the CONTAINER, not the predictor",
                "ddm_rc1 -- the adaptive model-section recode that put the RC1 rider on this body",
                "reordering-pays-iff-the-coder-has-no-context-model -- the sister statement about what a coder's model does to a permutation",
            ],
            "known_boundary": "one section, one body, one edit family; the draw-to-draw spread is +-35 B (sd measured over 133 real single-move builds), so a SPECIFIC edit must still be priced by a real encode -- and the delta does NOT accumulate over edits, see the one-sample-lottery anchor",
            "verdict_scope": "measurement (exact bytes), not a family verdict",
        },
        units_in={
            "changed_codes": "count",
            "container_search": "bool",
        },
        units_out={
            "section_delta_bytes": "bytes",
            "container_search_recovers_bytes": "bytes",
            "cells_to_break_even": "seg_cells",
        },
        empirical_anchors=(fee, search, lottery),
        predicted_vs_empirical_residual={
            fee.anchor_id: fee.residual,
            search.anchor_id: search.residual,
            lottery.anchor_id: lottery.residual,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_LEDGER, _CHARTER),
        canonical_producers=(_PRODUCER, _LEDGER),
        provenance=build_provenance_for_predicted(
            model_id="model_section_edit_container_break_fee.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
    )
