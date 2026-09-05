# SPDX-License-Identifier: MIT
"""Canonical equation: the adaptive-recode ceiling on the two RX1 MODEL sections (ddm_rc1).

THE GAP THIS CODIFIES.  The frontier archive's two MODEL sections -- the SM3R renderer body and
the IHS1 integer probability model -- reached the archive through Brotli alone.  Brotli is a
generic BYTE coder: it never sees a code boundary.  Nobody had ever unpacked the packed 3/4-bit
(semantic) or 0..8-bit (hpac) signed integer codes and driven them through a coder that models
the code alphabet.  The ddm_rc1 charter predicted -2,650..-5,650 container B.  It is -1,733 B,
and the shortfall has a single identifiable cause that this law records so no successor repeats
it.

THE LAW (three clauses; all numbers MEASURED on the cl2 frontier object).

1. CODER CLAUSE.  An adaptive binary-tree model over the code alphabet, one context bank per
   tensor (semantic) or per bit depth (hpac), lands within 1% of the order-0 entropy of the
   very stream it codes:

       semantic:  packed 28,896 B -> coded 24,544 B   vs H0 24,346 B   (+0.81%)
       hpac:      packed 11,156 B -> coded  9,639 B   vs H0  9,608 B   (+0.32%)

   No transmitted table, no float, integer 12-bit bins, DX2's carryless range coder.

2. BROTLI-OVERLAP CLAUSE (why the credit is much smaller than the coder's raw win).  The
   container credit is NOT the raw win.  Brotli already recovers most of a packed code stream's
   order-0 statistics when the code width is CONSTANT, because a fixed-width packing makes each
   byte a stable function of the code alphabet and Brotli's previous-byte literal context sees
   it.  It cannot do so when the width CHANGES every few hundred symbols, because then no byte
   boundary aligns with a code boundary.  Hence

       semantic (constant 3/4-bit widths, 16 tensors):  raw win 4,352 B -> container -610 B
       hpac     (widths 0..8 changing per channel):     raw win 1,517 B -> container -1,123 B

   The hpac credit is 1.84x the semantic credit off a body 2.0x SMALLER.  Predict the credit
   from the width-stability of the packing, never from the raw coder win.

3. CONTEXT-DILUTION CLAUSE (what stays closed, and why).  These bodies DO carry first-order
   structure -- the IHS1 rows measure a Miller-Madow-corrected H1 of ~8,469 B against an H0 of
   9,608 B, so ~1,139 B is real.  FIVE conditioned designs converted essentially none of it
   (previous magnitude bucket, previous bit-length bucket, adaptive unary magnitude, exp-Golomb
   bit-length prefix, row-delta, each at two or three adaptation shifts).  Their costs against
   the unconditioned tree run from -10.3 B to +1,158 B: the single best-conditioned variant
   (previous bit-length, 4 buckets, shift 4) is 10.3 B better, which is 0.9% of the 1,139 B
   available and does not survive as a design.  The mechanism is measurable, not speculative:
   a depth-7 row class holds 3,614 symbols over an alphabet of 120, so the plug-in H1 needs
   1,508 conditional cells and Miller-Madow attributes 0.277 of the 1.363-bit apparent gain to
   overfit.  An adaptive coder must LEARN each cell from the same symbols, and the learning cost
   IS the bias.  The structure is real AND unaffordable at this sample size.

TRANSFER BOUNDARY (binding).  Clause 1 is about a packed-integer weight stream with a stationary
per-group distribution; it does NOT transfer to already-entropy-coded streams (the RC64 token
tail), nor to the carrier (dx1 measured its own recode ceiling at -18 B).  Clause 2's RATIO
transfers as a rule of thumb only; the absolute bytes never do.  Clause 3 is FORMULATION scope --
it closes the five conditioned designs listed, not the family: a semi-static table paying explicit counted
bytes, or a mixture sharing one weight vector across depths, is untested.

ADAPTATION-RATE COROLLARY.  The shift dominates every context choice.  A stationary stream wants
a SLOW shift (semantic: 6, i.e. rate 1/64, worth 344 container B over shift 4); a stream that switches
group every few hundred symbols wants a FAST one (hpac: 5).  Sweep the shift before designing a
context.

VERDICT: a POSITIVE law with a measured ceiling.  The candidate is -1,733 B at zero distortion
(179,982 -> 178,249 B), rate dS -1.153933565760723e-03, projected S 0.14666350774473783.  Axis
``[macOS-CPU advisory / scorer-free EXACT byte measurement]``; ``score_claim=false`` until a T4
row confirms custody.

Producers: ``experiments/ddm_rc1_model_section_adaptive_recode.py`` (the race + the build) and
``experiments/ddm_rc1_adaptive_section_codec.py`` (the coder, encoder AND receiver).
Consumer: ``.omx/research/ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md``.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "model_section_adaptive_recode_ceiling_v1"
_UTC = "2026-09-05T00:00:00Z"
_LEDGER = ".omx/research/ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md"

#: The cl2 frontier object this law was measured on.
BASE_ARCHIVE_SHA256 = "08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e"
BASE_ARCHIVE_BYTES = 179_982
BASE_SCORE = 0.14781744131049854
CANDIDATE_ARCHIVE_SHA256 = (
    "1438049e3655fbcfa8eb289fa51ac58f834d72d8a09586353663cea68e57c122"
)
CANDIDATE_ARCHIVE_BYTES = 178_249
S_PER_BYTE = 25.0 / 37_545_489

#: RAW basis: packed bytes, order-0 entropy bytes, and the coder's realized bytes.
SECTION_RAW = {
    "semantic": {"packed": 28_896, "h0": 24_346.3, "h1": 24_029.4, "coded": 24_544.0},
    "hpac": {"packed": 11_156, "h0": 9_607.8, "h1_miller_madow": 8_469.0, "coded": 9_639.5},
}
#: CONTAINER basis: what archive.zip holds, before and after.
SECTION_CONTAINER = {
    "semantic": {"shipped": 30_856, "candidate": 30_246, "delta": -610, "shift": 6},
    "hpac": {"shipped": 13_466, "candidate": 12_343, "delta": -1_123, "shift": 5},
}
#: The generic measurement baselines, container basis (neither beats the shipped Brotli).
GENERIC_BASELINES = {
    "semantic": {"brotli_ck2_shipped": 30_856, "xz9e": 32_608, "xz9e_ck2": 31_988,
                 "zstd22": 33_224, "zstd22_ck2": 32_624},
    "hpac": {"brotli_shipped": 13_466, "xz9e": 13_536, "xz9e_ck2": 13_720,
             "zstd22": 13_992, "zstd22_ck2": 14_143},
}
#: The order-1 context designs that LOST to the unconditioned tree (hpac, raw bytes).
CLOSED_CONTEXT_DESIGNS = {
    "unconditioned_tree_winner": 9_639.5,
    "prev_magnitude_bucket": (9_640.1, 9_667.2),
    "prev_bitlength_bucket_high_depths": (9_629.2, 9_681.9),
    "adaptive_unary_magnitude": (10_163.4, 10_797.1),
    "expgolomb_bitlength_prefix": (10_289.4, 10_382.7),
    "row_delta": (9_835.1, 9_932.4),
}
#: The charter's admit bar (10x the container-transform noise ck2 measured).
ADMIT_BAR_BYTES = -300


def container_delta_bytes() -> int:
    """Total container credit over both MODEL sections (MEASURED, exact)."""

    return sum(int(item["delta"]) for item in SECTION_CONTAINER.values())


def rate_delta_s(delta_bytes: int | None = None) -> float:
    """Rate-term dS for a byte delta, at the contest's exact 25/37,545,489 S per byte."""

    if delta_bytes is None:
        delta_bytes = container_delta_bytes()
    return delta_bytes * S_PER_BYTE


def coder_gap_fraction(section: str) -> float:
    """How far the adaptive coder lands above its own order-0 bound, as a fraction."""

    row = SECTION_RAW[section]
    return float(row["coded"]) / float(row["h0"]) - 1.0


def admitted(delta_bytes: int | None = None) -> bool:
    """The charter's admit rule: total container credit at or below -300 B."""

    if delta_bytes is None:
        delta_bytes = container_delta_bytes()
    return delta_bytes <= ADMIT_BAR_BYTES


def build_model_section_adaptive_recode_ceiling_v1() -> CanonicalEquation:
    """Build the MODEL-section adaptive-recode ceiling equation (ddm_rc1, 2026-09-05)."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_LEDGER,
        reactivation_criteria=(
            "re-measure when either MODEL section's packing changes (a new SM3R mode, a "
            "different per-tensor depth allocation, a re-fitted IHS1 depth ladder), when the "
            "container's Brotli shape changes, or when any arm MEASURES an order-1 model that "
            "converts part of the ~1,139 B of first-order structure clause 3 records as real "
            "but unaffordable"
        ),
    )

    coder_anchor = EmpiricalAnchor(
        anchor_id="model_section_adaptive_coder_reaches_order0_20260905",
        measurement_utc=_UTC,
        inputs={
            "object": (
                "the cl2 frontier archive's two RX1 MODEL section bodies, archive sha256 "
                + BASE_ARCHIVE_SHA256
            ),
            "semantic_body": "SM3R v1 mode 6, 36,130 raw B, 59,376 codes at 3/4 bits",
            "hpac_body": "IHS1, 17,770 raw B, 20,416 codes at 0..8 bits over 517 channels",
            "coder": (
                "adaptive binary tree over the code alphabet, integer 12-bit bins, DX2's "
                "carryless range coder, no transmitted table"
            ),
            "scorer_runs": 0,
            "training_runs": 0,
        },
        predicted_output={
            "prior_law": (
                "the charter's prediction: quantized trained weights concentrate near zero, so "
                "the semantic codes carry 2.4-2.9 bits/param against Brotli's realized 3.72"
            ),
            "semantic_bits_per_param": "2.4 .. 2.9",
        },
        empirical_output={
            "semantic_h0_bits_per_param": 3.281,
            "semantic_packed_bits_per_param": 3.893,
            "semantic_coder_gap_fraction": 0.00814,
            "hpac_coder_gap_fraction": 0.00330,
            "consequence": (
                "the coder is at its bound and the sections are not; the charter's 3.72 "
                "bits/param divided the WHOLE body's container bytes by the code count, so it "
                "charged the codes for the fp16 scales, fp16 tensors and prune masks too"
            ),
        },
        # |predicted midpoint 2.65 - measured 3.281| / 2.65
        residual=0.2381,
        source_artifact=_LEDGER,
        measurement_method="exact_adaptive_model_code_length_plus_plugin_entropy_over_the_full_bodies",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    container_anchor = EmpiricalAnchor(
        anchor_id="model_section_container_credit_and_width_stability_20260905",
        measurement_utc=_UTC,
        inputs={
            "object": "the cl2 frontier archive, " + str(BASE_ARCHIVE_BYTES) + " B",
            "identity_control": (
                "both shipped container streams re-pack byte-identically from the restored raw "
                "bodies (semantic ck2+Brotli q11 lgwin24 -> 30,856 B; hpac Brotli -> 13,466 B)"
            ),
            "generic_baselines": "xz -9e and zstd --ultra -22, with and without the ck2 plane",
            "decode_control": (
                "every coded body decoded by a FRESH decoder and asserted byte-identical to the "
                "raw body before its bytes were recorded"
            ),
            "scorer_runs": 0,
            "training_runs": 0,
        },
        predicted_output={
            "charter_total_delta_bytes": "-2,650 .. -5,650",
            "charter_semantic_delta_bytes": "-2,050 .. -4,350",
            "charter_hpac_delta_bytes": "-600 .. -1,300",
            "charter_falsifier": (
                "best adaptive total within -300 B of shipped => Brotli already sits at these "
                "bodies' empirical entropy => CLOSE at family scope"
            ),
        },
        empirical_output={
            "semantic_delta_bytes": SECTION_CONTAINER["semantic"]["delta"],
            "hpac_delta_bytes": SECTION_CONTAINER["hpac"]["delta"],
            "total_delta_bytes": container_delta_bytes(),
            "rate_delta_s": rate_delta_s(),
            "candidate_archive_bytes": CANDIDATE_ARCHIVE_BYTES,
            "candidate_archive_sha256": CANDIDATE_ARCHIVE_SHA256,
            "projected_s": BASE_SCORE + rate_delta_s(),
            "generic_baselines_container_bytes": GENERIC_BASELINES,
            "falsifier_fired": False,
            "verdict": (
                "ADMIT at 5.78x the -300 B bar; neither generic coder beats the shipped Brotli "
                "on either body; the credit tracks the packing's WIDTH STABILITY, not the raw "
                "coder win"
            ),
        },
        # |predicted midpoint -4,150 - measured -1,733| / 4,150
        residual=0.5824,
        source_artifact=_LEDGER,
        measurement_method="exact_container_byte_counts_through_the_shipped_packers_with_fresh_decoder_identity",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    dilution_anchor = EmpiricalAnchor(
        anchor_id="order1_context_dilution_closes_five_designs_20260905",
        measurement_utc=_UTC,
        inputs={
            "object": "the IHS1 rows, 20,416 codes over 517 channels at 0..8 bits",
            "designs": sorted(k for k in CLOSED_CONTEXT_DESIGNS if k != "unconditioned_tree_winner"),
            "bias_estimator": "Miller-Madow correction on the plug-in conditional entropy",
            "scorer_runs": 0,
            "training_runs": 0,
        },
        predicted_output={
            "prior_law": (
                "first-order structure visible in the plug-in H1 is convertible by conditioning "
                "the adaptive contexts on the previous symbol"
            ),
            "expected_conversion_bytes": "several hundred of the ~1,139 B measured",
            "note": "the best conditioned variant converted 10.3 B, 0.9% of what is available",
        },
        empirical_output={
            "h0_bytes": SECTION_RAW["hpac"]["h0"],
            "h1_miller_madow_bytes": SECTION_RAW["hpac"]["h1_miller_madow"],
            "real_structure_bytes": 1_138.8,
            "converted_bytes": 10.3,
            "design_costs_raw_bytes": CLOSED_CONTEXT_DESIGNS,
            "mechanism": (
                "depth-7 rows: 3,614 symbols over an alphabet of 120 need 1,508 conditional "
                "cells; Miller-Madow attributes 0.277 of the 1.363-bit apparent gain to overfit. "
                "The learning cost IS the bias, so the structure is real AND unaffordable"
            ),
            "verdict_scope": "FORMULATION -- the five conditioned designs listed, not the family",
        },
        # |predicted midpoint 500 B - measured 10.3 B| / 500
        residual=0.9794,
        source_artifact=_LEDGER,
        measurement_method="exact_adaptive_model_code_length_per_design_plus_miller_madow_bias",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Adaptive-recode ceiling on packed integer MODEL sections, and its "
            "width-stability rule"
        ),
        one_line_summary=(
            "An adaptive per-group tree coder over the MODEL sections' packed codes lands within "
            "1% of their order-0 entropy for -1,733 container B; the credit tracks the packing's "
            "WIDTH STABILITY, not the raw win"
        ),
        latex_form=(
            r"\Delta B_{\text{container}} = \sum_{s} \left(B^{\text{RC1}}_s - "
            r"B^{\text{brotli}}_s\right) = -1{,}733\ \text{B},\quad "
            r"\frac{C_s}{H_0(s)} - 1 \in [0.0033, 0.0081],\quad "
            r"\Delta S = \Delta B \cdot \frac{25}{37{,}545{,}489}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.model_section_adaptive_recode_ceiling_20260905"
            ":container_delta_bytes"
        ),
        units_in={
            "section": "one of 'semantic' (SM3R body) or 'hpac' (IHS1 body)",
            "delta_bytes": "container bytes, signed (negative is a credit)",
        },
        units_out={
            "container_delta_bytes": "bytes of the 179,982 B archive",
            "rate_delta_s": "contest score units, at 25/37,545,489 S per byte",
            "coder_gap_fraction": "dimensionless excess over the stream's own order-0 entropy",
        },
        empirical_anchors=(coder_anchor, container_anchor, dilution_anchor),
        predicted_vs_empirical_residual={
            "semantic_bits_per_param": 0.2381,
            "total_container_delta_bytes": 0.5824,
            "order1_conversion_bytes": 0.9794,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_producers=(
            "experiments/ddm_rc1_model_section_adaptive_recode.py",
            "experiments/ddm_rc1_adaptive_section_codec.py",
        ),
        canonical_consumers=(_LEDGER,),
        provenance=provenance,
        domain_of_validity={
            "object": (
                "the two RX1 MODEL section bodies of the cl2 frontier archive, sha256 "
                + BASE_ARCHIVE_SHA256
                + " @ "
                + str(BASE_ARCHIVE_BYTES)
                + " B"
            ),
            "applies_to": (
                "packed fixed-point integer code streams with a stationary per-group "
                "distribution, reaching the archive through a generic byte coder"
            ),
            "does_not_apply_to": (
                "already-entropy-coded streams (the RC64 token tail); the carrier (dx1 measured "
                "its own recode ceiling at -18 B); any claim that the ORDER-1 structure is "
                "convertible -- clause 3 closes five conditioned designs, not the family"
            ),
            "coder": (
                "adaptive binary tree over the code alphabet, one context bank per tensor "
                "(semantic) or per bit depth (hpac); integer 12-bit bins; DX2's carryless range "
                "coder; no transmitted model"
            ),
            "transfer_rule": (
                "the WIDTH-STABILITY reading transfers (constant-width packings leave Brotli "
                "little to give up, changing-width packings leave a lot); the absolute bytes "
                "never do, and the adaptation shift must be swept before any context is designed"
            ),
        },
    )


__all__ = [
    "ADMIT_BAR_BYTES",
    "CANDIDATE_ARCHIVE_BYTES",
    "CANDIDATE_ARCHIVE_SHA256",
    "CLOSED_CONTEXT_DESIGNS",
    "EQUATION_ID",
    "GENERIC_BASELINES",
    "SECTION_CONTAINER",
    "SECTION_RAW",
    "admitted",
    "build_model_section_adaptive_recode_ceiling_v1",
    "coder_gap_fraction",
    "container_delta_bytes",
    "rate_delta_s",
]
