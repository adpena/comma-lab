# SPDX-License-Identifier: MIT
"""Canonical equation: the section-coding rate axis is measured shut, and the two
preconditions that make any future coding claim admissible.

The closure
-----------
Four mechanisms, all four coded sections of the sz1 pointer archive (sha
``debb025f45bb...`` @ 179,930 B), measured byte-exact through the real coders:

============================================  ==================================  ======
mechanism                                     result                              closes
============================================  ==================================  ======
brotli parameter sweep (3 modes x 15 lgwin    shipped is at-or-beyond our          yes
x 10 lgblock, quality 11)                     encoder on all three model streams
cross-section joint coding (all 24 orders)    +205 B; every ordered conditional    yes
                                              pair NEGATIVE
global byte-plane deinterleave k in 2,3,4,8   +31..+509 B; loses on all three      yes
                                              sections at every k
REGION byte-plane (the sz1 mechanism)         hpac +7, carrier -5, semantic 0      yes
============================================  ==================================  ======

Total honestly available on this axis: **-5 B** on 179,930 B = **3.3e-6 S**, below any
admission bar and not worth a receiver change. ``token_stream`` -- 61.0% of the archive --
was closed by measurement too (every mechanism COSTS bytes, ratio 1.0000-1.0006), converting
the last standing derivation on this axis into a receipt.

Concordant with #996 (coder axis closed vs each section's own memoryless bound) and #1060
(all 38/38 semantic tensors receiver-required, every exact recoding +340 B). Three
independent closures agree. **Rate progress must come from REPRESENTATION -- fewer or
smaller symbols -- not from coding them better.**

PRECONDITION 1: calibrate the instrument, or you cannot state your own bar
--------------------------------------------------------------------------
``ddm_xs1`` could not reproduce the shipped ``hpac`` stream (off by 40 B) and correctly
refused to draw a conclusion. Read at source: ``runtime/residual_archive.py:161`` unpacks
``RX1_MODEL_HEADER`` and the live archive carries ``codec=2 = RX1_CODEC_BROTLI`` -- the
``lzma.FORMAT_XZ`` branch at ``:179`` is dead code on this lineage. With that fixed,
``mode=GENERIC, lgwin=16, q11`` reproduces ``semantic`` to the byte.

``hpac`` still lands +39 B: the brotli CLI also lands 13,555 and lgblock sweeps do not move
it, so the shipped stream was produced by an encoder that found a better parse than
python-brotli 1.2.0 can. **Consequence: on hpac a candidate must beat -39 on our instrument
merely to TIE the bytes already shipped.** A candidate measured at -20 on an uncalibrated
instrument looks like a win and is a 19 B loss. State the bar or the number means nothing.

PRECONDITION 2: a byte-permutation must ship its inverse, in the same measurement
---------------------------------------------------------------------------------
The first composed hpac result read **-215 B** -- a clear win over the -39 bar. It was an
artifact of a LOSSY transform: the compositor read each region from the ORIGINAL buffer
while writing into a copy, so overlapping regions clobbered each other and destroyed
information. It compressed better because it was throwing bytes away.

A verified forward/inverse pair (200 random op-sequences, overlaps included, all
round-tripping byte-identical) plus a greedy search constrained to that pair converges to
**-32** -- i.e. **+7 worse than shipped**. Same grid, same instrument, honest transform, and
the win evaporates.

Live instance of the NO-FAKE class: *a transform that improves the metric by being lossy is
not a rate win.* The cure is ordering, not cleverness -- prove invertibility BEFORE
believing the number.

PRECONDITION 3 (the one I violated on myself): do not cap the search below the known scale
-------------------------------------------------------------------------------------------
The first region search bounded length at 4,096 B -- BELOW sz1's own winning region of
8,284 B. A closure drawn inside a self-imposed bound is a statement about the bound. Re-run
uncapped over ``len in {8192, 8284, 12288, 16384, 24576, 36040} x k in {2,4}`` at 2,048-B
stride: 114 arms, zero negatives, best +11. The cap was not hiding a win -- but that is a
measurement, and it was not knowable before it ran.

``verdict_scope``: **INSTANCE** -- the sz1 pointer archive, measured byte-exact. The three
preconditions are METHOD and carry to any coding claim on any vehicle.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "section_coding_axis_closure_v1"
AXIS = "[byte-exact through the real coders] on the sz1 pointer archive; no entropy estimates"
SOURCE_MEMO = ".omx/research/ddm_bp1_section_coding_axis_closed_20260818.md"

RATE_DENOMINATOR_BYTES = 37_545_489

ARCHIVE_SHA256 = "debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a"
ARCHIVE_BYTES = 179_930

#: section -> (shipped coded bytes, our best same-instrument bytes). The DELTA is the bar.
#: Our encoder is python-brotli 1.2.0 at mode=GENERIC, lgwin=16, quality=11.
SHIPPED_VS_OUR_INSTRUMENT = {
    "semantic_blob": (34_243, 34_243),
    "carrier_blob": (22_161, 22_160),
    "hpac_blob": (13_515, 13_554),
    "token_stream": (109_801, 109_806),
}

#: The best honest result of the REGION byte-plane mechanism, per section, after the
#: invertibility constraint. Positive = costs bytes.
REGION_BYTE_PLANE_BEST = {
    "semantic_blob": 0,
    "carrier_blob": -5,
    "hpac_blob": +7,
    "token_stream": +7,
}

TOTAL_AVAILABLE_BYTES = -5
TOTAL_AVAILABLE_S = 3.33e-06

#: The number the lossy compositor produced, and what the invertible one produces.
FAKE_WIN_LOSSY_BYTES = -215
HONEST_INVERTIBLE_BYTES = -32
ROUND_TRIP_PROOF_SEQUENCES = 200

#: The cap I imposed on my own search, and sz1's own winning region length.
SELF_IMPOSED_CAP_BYTES = 4_096
SZ1_WINNING_REGION_BYTES = 8_284
UNCAPPED_ARMS = 114
UNCAPPED_BEST_BYTES = +11


def instrument_bar_bytes(section: str) -> int:
    """Bytes a candidate must beat ON OUR INSTRUMENT merely to TIE the shipped stream.

    Negative means our encoder is behind the shipped one and the candidate owes that
    much before it has gained anything. ``hpac_blob`` returns -39: the trap this exists
    to close.

    Raises KeyError for an unknown section rather than returning 0 -- a silent zero bar
    would let an uncalibrated section pass as calibrated, which is the exact defect.
    """
    shipped, ours = SHIPPED_VS_OUR_INSTRUMENT[section]
    return shipped - ours


def coding_candidate_admissible(
    section: str,
    candidate_bytes_on_our_instrument: int,
    *,
    inverse_verified: bool,
    search_max_region_bytes: int | None = None,
) -> dict:
    """Fail-closed admissibility for a section-coding candidate.

    Three preconditions, each of which has already produced a wrong number once:

    1. **Bar** -- the candidate must beat ``instrument_bar_bytes(section)``, not zero.
    2. **Invertibility** -- ``inverse_verified`` must be True. A byte permutation without
       a demonstrated inverse and round-trip proof is not measurable; the -215 B "win"
       was a lossy transform.
    3. **Uncensored search** -- if a max region length is declared it must reach at least
       ``SZ1_WINNING_REGION_BYTES``, the known-winning scale on this vehicle.

    Returns the verdict plus every failed precondition, so a caller sees WHICH one bit.
    """
    bar = instrument_bar_bytes(section)
    refusals = []
    if not inverse_verified:
        refusals.append(
            "INVERSE_UNVERIFIED — ship the forward/inverse pair and a round-trip proof "
            "in the same measurement; a lossy transform compresses better by destroying "
            "information (ddm_bp1 §3)"
        )
    if (
        search_max_region_bytes is not None
        and search_max_region_bytes < SZ1_WINNING_REGION_BYTES
    ):
        refusals.append(
            f"SEARCH_CAPPED_BELOW_KNOWN_SCALE — max region {search_max_region_bytes} B is "
            f"under the known-winning {SZ1_WINNING_REGION_BYTES} B; a closure inside a "
            "self-imposed bound is a statement about the bound"
        )
    beats_bar = candidate_bytes_on_our_instrument < bar
    if not beats_bar:
        refusals.append(
            f"DOES_NOT_BEAT_BAR — {candidate_bytes_on_our_instrument:+d} B does not beat "
            f"the {bar:+d} B instrument bar for {section}"
        )
    return {
        "section": section,
        "instrument_bar_bytes": bar,
        "candidate_bytes": candidate_bytes_on_our_instrument,
        "realized_bytes_vs_shipped": candidate_bytes_on_our_instrument - bar,
        "admissible": not refusals,
        "refusals": tuple(refusals),
        "score_claim": False,
    }


def bytes_to_S(delta_bytes: float, *, denominator: int = RATE_DENOMINATOR_BYTES) -> float:
    """Rate-leg S for a byte delta. Linear and axis-independent, forever."""
    return 25.0 * delta_bytes / denominator


def build_section_coding_axis_closure_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        SOURCE_MEMO,
        reactivation_criteria=(
            "re-open only on evidence that the sections' STATISTICS changed — a new "
            "token model, a different carrier codec, or a semantic serialization change. "
            "A better general-purpose compressor is not evidence: the shipped encoder "
            "already beats generic brotli-q11 by 661 B on the model streams."
        ),
        measurement_axis=AXIS,
        hardware_substrate="macos_arm64_local",
        captured_at_utc="2026-08-18T05:30:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="bp1_four_mechanisms_four_sections_closed_20260818",
            measurement_utc="2026-08-18T05:30:00Z",
            inputs={
                "archive_sha256": ARCHIVE_SHA256,
                "archive_bytes": ARCHIVE_BYTES,
                "sections": list(SHIPPED_VS_OUR_INSTRUMENT),
                "coders": "python-brotli 1.2.0 q11 (+ brotli CLI cross-check), lzma XZ 9e",
                "mechanisms": [
                    "brotli parameter sweep 3 modes x 15 lgwin x 10 lgblock",
                    "cross-section joint coding, all 24 orders",
                    "global byte-plane deinterleave k in {2,3,4,8}",
                    "region byte-plane, greedy, invertibility-constrained, uncapped",
                ],
                "payload_custody": "/Volumes/APDataStore/pact/ddm_bp1/ (P0 — persisted before measuring)",
            },
            predicted_output={
                "claim_under_test": (
                    "the two sections xs1 left unprobed (hpac, token_stream) hold a "
                    "coding win the calibrated instrument will reveal"
                )
            },
            empirical_output={
                "instrument_bars": {
                    s: instrument_bar_bytes(s) for s in SHIPPED_VS_OUR_INSTRUMENT
                },
                "region_byte_plane_best": dict(REGION_BYTE_PLANE_BEST),
                "joint_coding_best_of_24_orders_bytes": +205,
                "token_stream_ratios": {
                    "brotli_q11_lgwin16": 1.0001,
                    "brotli_q11_lgwin24": 1.0000,
                    "lzma_xz_9e": 1.0006,
                    "byte_plane_then_brotli": 1.0001,
                },
                "total_available_bytes": TOTAL_AVAILABLE_BYTES,
                "total_available_S": TOTAL_AVAILABLE_S,
                "verdict": (
                    "REFUTED — the axis is shut across all four sections; -5 B total is "
                    "below any admission bar. Rate progress must come from REPRESENTATION."
                ),
                "no_fake_instance": {
                    "lossy_composed_bytes": FAKE_WIN_LOSSY_BYTES,
                    "invertible_composed_bytes": HONEST_INVERTIBLE_BYTES,
                    "round_trip_sequences_proven": ROUND_TRIP_PROOF_SEQUENCES,
                    "mechanism": (
                        "compositor read each region from the ORIGINAL buffer while "
                        "writing into a copy; overlapping regions clobbered each other"
                    ),
                },
                "self_imposed_cap": {
                    "cap_bytes": SELF_IMPOSED_CAP_BYTES,
                    "known_winning_scale_bytes": SZ1_WINNING_REGION_BYTES,
                    "uncapped_arms": UNCAPPED_ARMS,
                    "uncapped_best_bytes": UNCAPPED_BEST_BYTES,
                    "note": "the cap was not hiding a win — but that took a measurement",
                },
            },
            residual=0.0,
            source_artifact="/Volumes/APDataStore/pact/ddm_bp1/BP1_PARAM_SWEEP.json",
            measurement_method=(
                "the shipped streams were extracted from the live archive by parsing "
                "RX1_MODEL_HEADER (codec=2=BROTLI), persisted to the SSD tier BEFORE any "
                "measurement (P0 ALWAYS KEEP THE PAYLOAD), then re-coded byte-exact. No "
                "entropy estimates anywhere; every number is a real coder's output length."
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Section-coding axis closure, with the calibration / invertibility / "
            "uncensored-search preconditions that make a coding claim admissible"
        ),
        one_line_summary=(
            "4 mechanisms x 4 sections leave -5 B = 3.3e-6 S; a coding candidate is "
            "admissible only if it beats its section's INSTRUMENT bar (hpac -39 B), ships "
            "a verified inverse, and searched past 8,284 B"
        ),
        latex_form=r"\Delta S_{rate}=\frac{25\,\Delta B}{37{,}545{,}489},\quad \Delta B \geq -5",
        python_callable_module_path=(
            "tac.canonical_equations.section_coding_axis_closure_20260818"
            ":coding_candidate_admissible"
        ),
        domain_of_validity={
            "axis": AXIS,
            "research_only": False,
            "applies_to": (
                "lossless re-coding of the archive's existing sections on the "
                "PR130/PR135-lineage container: brotli/lzma parameter choice, joint or "
                "shared-dictionary coding, byte-plane permutation"
            ),
            "does_not_apply_to": (
                "REPRESENTATION changes — fewer or smaller symbols. Those move the rate "
                "term and this closure says nothing about them; it is precisely what the "
                "closure routes toward."
            ),
            "preconditions_are_method_not_instance": (
                "the -5 B is INSTANCE-scoped to the sz1 archive; the three preconditions "
                "(calibrated bar, verified inverse, uncensored search) carry to any "
                "coding claim on any vehicle"
            ),
        },
        units_in={
            "candidate_bytes_on_our_instrument": "bytes, signed, vs our own best baseline",
            "search_max_region_bytes": "bytes",
        },
        units_out={"instrument_bar_bytes": "bytes", "S": "contest score units"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"registration": 0.0},
        last_calibration_utc="2026-08-18T05:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "any proposed re-coder / repack / byte-permutation on this lineage: call "
            "coding_candidate_admissible() before believing a byte delta",
            "rate-route charters — the closure is why the route is representation, not "
            "coding (ddm_asym1 needs 11,584 B and none of it comes from the coder)",
            "the NO-FAKE catalog: the lossy-compositor instance is a worked example of a "
            "metric improved by destroying information",
        ),
        canonical_producers=(
            SOURCE_MEMO,
            ".omx/research/ddm_xs1_cross_section_joint_coding_20260818.md",
            "/Volumes/APDataStore/pact/ddm_bp1/ (all receipts + persisted payloads)",
        ),
        provenance=provenance,
    )


def populate_section_coding_axis_closure_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_section_coding_axis_closure_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "ARCHIVE_BYTES",
    "ARCHIVE_SHA256",
    "AXIS",
    "EQUATION_ID",
    "FAKE_WIN_LOSSY_BYTES",
    "HONEST_INVERTIBLE_BYTES",
    "RATE_DENOMINATOR_BYTES",
    "REGION_BYTE_PLANE_BEST",
    "ROUND_TRIP_PROOF_SEQUENCES",
    "SELF_IMPOSED_CAP_BYTES",
    "SHIPPED_VS_OUR_INSTRUMENT",
    "SZ1_WINNING_REGION_BYTES",
    "TOTAL_AVAILABLE_BYTES",
    "TOTAL_AVAILABLE_S",
    "UNCAPPED_ARMS",
    "UNCAPPED_BEST_BYTES",
    "build_section_coding_axis_closure_v1",
    "bytes_to_S",
    "coding_candidate_admissible",
    "instrument_bar_bytes",
    "populate_section_coding_axis_closure_equation",
]
