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
        1                    +65.0 +- 39.8 B                    -7.2 +- 25.4 B
       72                    +68.1 +- 29.4 B                   +21.0 +- 29.5 B
      200                    +80.8 +- 29.0 B                   +24.8 +- 24.5 B

  least squares over all ten N:
      shipped shape:  dB = 0.2121*N + 58.58   (rms 7.18 B)
      searched:       dB = 0.1677*N -  1.92   (rms 5.26 B)

THE LAW.  The cost of editing a range-coded model section splits into

    dB(N) = BREAK_FEE * [the container shape is not re-searched] + MARGINAL_B_PER_CODE * N

with a MEASURED break fee of **58.6 B** and a marginal of only **0.17-0.21 B per changed code**.
The fee is not a length effect: length-preserving code changes still cost +40 B at the shipped
shape.  It is the loss of brotli's purchase on a re-randomised payload.

WHY IT IS RECOVERABLE.  Brotli streams are self-describing and the CK2 interleave rides in the
RX1 ``reserved`` byte, so the brotli quality, the window size and the interleave are
ENCODER-ONLY choices -- the receiver is never told which were used and its decode path is
unchanged.  Searching q in {9, 10, 11} x lgwin in {16, 18, 20, 22, 24} x {ck2, plain} and
shipping the smallest recovers **63.5 B at N = 1** and **47.1 B at N = 72**.  The live body's own
shape is ``(ck2, 11, 16)``; it is the best shape for the SHIPPED codes and a poor one for any
edit of them.

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
SHIPPED_CONTAINER_SHAPE = ("ck2", 11, 16)
RX1_RESERVED = 0x7A

#: Least-squares fits over N in {1, 2, 5, 10, 25, 50, 72, 100, 150, 200}, ten draws each.
BREAK_FEE_BYTES = 58.58
SHIPPED_SHAPE_MARGINAL_B_PER_CODE = 0.2121
SHIPPED_SHAPE_FIT_RMS_BYTES = 7.18
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
            "shipped_shape_fit": "dB = 0.2121*N + 58.58 B (rms 7.18)",
            "shipped_shape_at_N1": 65.0,
            "shipped_shape_at_N200": 80.8,
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
            "recovered_at_N1": 63.5,
            "recovered_at_N72": 47.1,
            "searched_at_N1": -7.2,
            "searched_at_N72": 21.0,
            "cells_to_break_even_N72_searched": 16.5,
            "cells_to_break_even_N72_shipped_shape": 53.5,
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
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Container-break fee on an edited range-coded model section -- an edit costs a "
            "fixed ~58.6 B, not a per-code price, and an encoder-side container search pays "
            "most of it back"
        ),
        one_line_summary=(
            "fe1: an edit costs 0.2121*N + 58.58 B at the shipped brotli/CK2 shape but only "
            "0.1677*N - 1.92 B re-searched; 63.5 B recovered at N=1, 47.1 B at N=72"
        ),
        latex_form=(
            r"\Delta B(N)=\underbrace{58.58}_{\text{break fee}}\cdot\mathbb{1}[\text{shape not re-searched}]"
            r"+m N,\quad m_{\text{shipped}}=0.2121,\ m_{\text{searched}}=0.1677;\quad"
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
            "known_boundary": "one section, one body, one edit family; the draw-to-draw spread is +-25..40 B, so a SPECIFIC edit must still be priced by a real encode",
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
        empirical_anchors=(fee, search),
        predicted_vs_empirical_residual={
            fee.anchor_id: fee.residual,
            search.anchor_id: search.residual,
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
