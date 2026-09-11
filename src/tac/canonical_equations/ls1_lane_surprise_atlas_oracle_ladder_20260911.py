# SPDX-License-Identifier: MIT
"""Canonical equation: the oracle ladder on the shipped token field (ddm_ls1 + ddm_ls2).

WHAT THIS PRICES.  The sub-0.12 rate corner asks for **25,899 B** off the incumbent tail at held
distortion.  ls1 measured, on the SHIPPED move-44 field itself, the very best a context model of a
given information set could do -- an ORACLE, granted its tables for free -- and ls2 then measured
what a real, charged, full-resolution correction of that same shape actually recovers.  Between them
they close the receiver-visible tail family on THIS object by arithmetic rather than by argument.

THE LADDER (ls1, n600, all 117,964,800 symbols, Miller-Madow corrected).  Each rung is a strictly
larger information set, so each is an upper bound on every model that sees no more than it::

    information set                      plug-in upper B   Miller-Madow B   MM shortfall vs 25,899 B
    tc1_joint (the shipped mixer's)          11,568.821        8,218.071            17,680.929
    receiver_lane (what a receiver CAN see)  22,222.524       17,534.216             8,364.784
    granted_previous_row_lane (GRANTED)      29,809.287       24,863.638             1,035.362

Read the third row carefully: it is the only rung that comes near the demand, and it is **granted
geometry the receiver does not have** -- a previous-row Lane distance that is unavailable at decode
time.  The highest RECEIVER-VISIBLE rung falls **8,364.784 B short** with its tables already free.
(Against the optimistic uncorrected plug-in it is short by 3,676.476 B; sparse-cell bias is large
enough here to change the threshold answer, which is why the MM column is the one that decides.)

THE CHARGED REALISATION (ls2, same field, all 600 pairs).  Two declared full-resolution correction
families, fit and then charged for their own parameters: separate distance features + HPAC phase
nets **266.162261 B**; the stronger joint distance feature + phase nets **385.550480 B** -- **1.4887 %**
of the demand.  Held out on a seeded 120-pair split the joint model returns 67.049582 B, 49.049582 B
after the entire parameter charge.  ls2's fire decision is REFUSED_CANDIDATE_FIRE, and not because
the gain is negative: the receiver work (~1.180e9 feature multiplies, 101,600 B of count state) has
no measured native timing, and live move-44 T4 inflate already sits at 1232.418725255 s against a
1260 s gate.

WHY LANE, AND WHY THAT IS NOT A POCKET.  Lane is **691,677 symbols = 0.586342 % of area** carrying
**33.478517 %** of stream surprise -- 57x over-represented.  It is still not a Lane-shaped pocket:
Lane contributes only 11,303.496 B of the strongest whole-population estimate, and the separate ls2
model's total gain is a Road gain that makes Lane *worse* (-88.535929 B signed gross on Lane).

THE INSTRUMENT IS RECONCILED TO THE REAL STREAM.  957,986.402 bits = 119,748.300 B against 119,749
physical RC64 bytes: a framing difference of 0.699757115 B (0.000584353 %), inside a pre-registered
+/-0.5 % falsifier.  That is what makes these oracle numbers comparable to archive bytes at all.

THE OPERATING RULE.  Any claimed tail mechanism on this object must beat 25,899 B **with its own
serialized cost**.  No priced mechanism does, and even unpriced side information falls 1,035 B short.
The explicit Lane override map -- the thing a "counted Lane carrier" would have to ship -- is
594,003 B on its own, which exceeds the entire incumbent tail.  The next question is the OBJECT, not
a better tail model on this one.

Axis ``[macOS-CPU advisory / scorer-free n600 receiver probabilities]``; ``score_claim=false``.
Registered by ddm_cons2 on 2026-09-11 from ls1's oracle ladder and ls2's pricing tables.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "lane_surprise_atlas_oracle_ladder_v1"

_UTC = "2026-09-11T00:00:00Z"
_AXIS = "[macOS-CPU advisory / scorer-free n600 receiver probabilities]"
_LS1_LEDGER = ".omx/research/ddm_ls1_lane_conditioned_surprise_atlas_on_the_shipped_field_20260911.md"
_LS2_LEDGER = ".omx/research/ddm_ls2_full_resolution_lane_probability_bound_20260911.md"

# MEASURED (ls1): the rate demand this ladder is read against, at held distortion on move 44.
#
# The demand is archive_bytes - strict_cap and is therefore true of ONE pointer move only:
# 180,406 - 154,507 = 25,899 at move 44. gdc1's 25,959 B is the same arithmetic at move 43
# (180,466 - 154,507). Do not harmonize them and do not carry either across a move.
RATE_DEMAND_BYTES = 25_899
STRICT_ARCHIVE_CAP_BYTES = 154_507
MOVE44_ARCHIVE_BYTES = 180_406

# MEASURED (ls1): (plug-in upper bound B, Miller-Madow estimate B) per information set.
ORACLE_LADDER_BYTES = {
    "tc1_joint": (11_568.821, 8_218.071),
    "receiver_lane": (22_222.524, 17_534.216),
    "granted_previous_row_lane": (29_809.287, 24_863.638),
}
# Which rungs a real decoder could actually compute. The one that nearly reaches the demand cannot.
RECEIVER_VISIBLE = {
    "tc1_joint": True,
    "receiver_lane": True,
    "granted_previous_row_lane": False,
}
# MEASURED (ls1): Lane's share of each rung's whole-population MM estimate, in bytes.
LANE_MM_BYTES = {
    "tc1_joint": 2_587.622,
    "receiver_lane": 6_686.446,
    "granted_previous_row_lane": 11_303.496,
}

# MEASURED (ls1): the atlas's own reconciliation to the physical stream.
STREAM_BITS = 957_986.402
STREAM_BYTES_FROM_BITS = 119_748.300
STREAM_BYTES_PHYSICAL_RC64 = 119_749
FRAMING_DIFFERENCE_BYTES = 0.699757115
FRAMING_DIFFERENCE_FRACTION = 0.00000584353
FRAMING_FALSIFIER_FRACTION = 0.005

# MEASURED (ls1): Lane's area and surprise shares, and the cost of shipping its override map.
TOTAL_SYMBOLS = 117_964_800
LANE_SYMBOLS = 691_677
LANE_AREA_FRACTION = 0.00586342
LANE_SURPRISE_FRACTION = 0.33478517
LANE_OVERRIDE_MAP_BYTES = 594_003

# MEASURED (ls2): net RC64 byte-equivalents after each family's own parameter charge.
CHARGED_CORRECTION_NET_BYTES = {
    "separate_distance_plus_phase": 266.162261,
    "joint_distance_plus_phase": 385.550480,
}
CHARGED_CORRECTION_PARAM_BYTES = {
    "separate_distance_plus_phase": 23,
    "joint_distance_plus_phase": 18,
}
LS2_HELD_OUT_JOINT_GAIN_BYTES = 67.049582
LS2_HELD_OUT_JOINT_GAIN_AFTER_FULL_PARAM_CHARGE_BYTES = 49.049582
LS2_RECEIVER_STATE_BYTES = 101_600
MOVE44_T4_INFLATE_SECONDS = 1232.418725255
DECODE_GATE_SECONDS = 1260


def oracle_shortfall_bytes(information_set: str, *, estimator: str = "miller_madow") -> float:
    """MEASURED shortfall of ``information_set``'s oracle against the 25,899 B rate demand.

    ``estimator`` is ``'miller_madow'`` (the deciding column) or ``'plug_in'`` (the optimistic
    upper bound). A positive result is how many bytes the oracle is SHORT with free tables.
    """
    index = {"plug_in": 0, "miller_madow": 1}
    if information_set not in ORACLE_LADDER_BYTES:
        raise KeyError(f"no measured oracle for information set {information_set!r}")
    if estimator not in index:
        raise KeyError(f"estimator must be 'plug_in' or 'miller_madow', got {estimator!r}")
    return RATE_DEMAND_BYTES - ORACLE_LADDER_BYTES[information_set][index[estimator]]


def best_receiver_visible_shortfall_bytes(*, estimator: str = "miller_madow") -> float:
    """The shortfall of the best rung a real decoder could compute: 8,364.784 B (MM).

    The 1,035.362 B rung is NOT this one -- it is granted previous-row geometry the receiver does
    not have, and quoting its shortfall as the receiver's is the ladder's one easy misreading.
    """
    visible = [k for k, ok in RECEIVER_VISIBLE.items() if ok]
    return min(oracle_shortfall_bytes(k, estimator=estimator) for k in visible)


def charged_fraction_of_demand(family: str) -> float:
    """DERIVED (ls2 states the net bytes and the 1.4887 % figure): net gain / rate demand."""
    if family not in CHARGED_CORRECTION_NET_BYTES:
        raise KeyError(f"no measured charged correction for family {family!r}")
    return CHARGED_CORRECTION_NET_BYTES[family] / RATE_DEMAND_BYTES


def instrument_is_reconciled() -> bool:
    """True when the atlas's bit accounting matches the physical stream inside its falsifier."""
    return FRAMING_DIFFERENCE_FRACTION < FRAMING_FALSIFIER_FRACTION


def build_lane_surprise_atlas_oracle_ladder_v1() -> CanonicalEquation:
    """Build the oracle-ladder canonical equation (ddm_ls1 atlas + ddm_ls2 charged realisation)."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_LS1_LEDGER,
        reactivation_criteria=(
            "this ladder is measured on the SHIPPED move-44 field; re-measure it on any field a "
            "pointer move changes. It reopens on (a) an information set NOT on the ladder that a "
            "receiver can compute, (b) a charged full-resolution family outside ls2's two declared "
            "ones, or (c) a native receiver whose MEASURED decode wall-clock fits the 1260 s gate, "
            "which is the leg ls2 actually lacked -- its gains were positive, not negative"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="macOS CPU (exact integer-frequency arithmetic through the shipped RC64 coder)",
        captured_at_utc=_UTC,
    )

    ladder_anchor = EmpiricalAnchor(
        anchor_id="ls1_oracle_ladder_shipped_move44_field_n600_20260911",
        measurement_utc=_UTC,
        inputs={
            "object": (
                "the SHIPPED move-44 token field subset6.u8, 117,964,800 symbols, sha "
                "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8, from archive "
                "180,406 B sha 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
            ),
            "information_sets": sorted(ORACLE_LADDER_BYTES),
            "estimator": "plug-in upper bound and Miller-Madow sparse-cell correction, both reported",
            "tables_granted_free": (
                "yes -- these are ORACLES. A real model must additionally pay for its tables; the "
                "DERIVED dense layout sizes are 3,047,232 / 7,861,024 / 10,203,200 B respectively"
            ),
            "rate_demand_bytes": RATE_DEMAND_BYTES,
        },
        predicted_output={
            "charter_framing": (
                "'oracle 17,534 B receiver-visible, 1,035 B short' -- which pairs the "
                "receiver_lane MM VALUE with the granted_previous_row_lane SHORTFALL"
            )
        },
        empirical_output={
            "ladder_plugin_and_mm_bytes": ORACLE_LADDER_BYTES,
            "mm_shortfall_vs_demand_bytes": {
                k: RATE_DEMAND_BYTES - v[1] for k, v in ORACLE_LADDER_BYTES.items()
            },
            "receiver_visible": RECEIVER_VISIBLE,
            "best_receiver_visible_mm_shortfall_bytes": 8_364.784,
            "the_1035_b_rung_is_not_receiver_visible": (
                "granted_previous_row_lane uses a previous-row Lane distance unavailable at decode "
                "time; its 1,035.362 B shortfall is an UNPAID hindsight grant, not a receiver's"
            ),
            "lane_mm_bytes_per_rung": LANE_MM_BYTES,
            "lane_area_fraction": LANE_AREA_FRACTION,
            "lane_surprise_fraction": LANE_SURPRISE_FRACTION,
            "lane_override_map_bytes": LANE_OVERRIDE_MAP_BYTES,
            "lane_map_alone_exceeds_the_whole_incumbent_tail": True,
            "instrument_reconciliation": {
                "bits": STREAM_BITS,
                "bytes_from_bits": STREAM_BYTES_FROM_BITS,
                "physical_rc64_bytes": STREAM_BYTES_PHYSICAL_RC64,
                "framing_difference_bytes": FRAMING_DIFFERENCE_BYTES,
                "framing_difference_fraction": FRAMING_DIFFERENCE_FRACTION,
                "falsifier": "+/-0.5 %, PASSED",
            },
            "verdict": (
                "no constructed mechanism is shown to clear the 25,899 B demand; the strongest "
                "specified oracle crosses it only in the optimistic uncorrected fit and only with "
                "geometry the receiver cannot have"
            ),
        },
        # The charter's framing is off by the difference between two rows of one table:
        # |8,364.784 - 1,035.362| / 8,364.784.
        residual=0.8762,
        source_artifact=_LS1_LEDGER,
        measurement_method=(
            "full n600 decode of the shipped field through the actual receiver, per-symbol integer "
            "frequencies, plug-in and Miller-Madow conditional entropies per information set, with "
            "an independent recount and a byte-identical resumed-checkpoint restart control"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    charged_anchor = EmpiricalAnchor(
        anchor_id="ls2_charged_full_resolution_correction_n600_20260911",
        measurement_utc=_UTC,
        inputs={
            "object": "the same shipped move-44 field; ls1 atlas RESULT sha ddf6a80bd9d49e6f...",
            "families": sorted(CHARGED_CORRECTION_NET_BYTES),
            "coefficients": "signed int8 / 32 in [-4, 127/32]; 8-byte LS2P header; nothing free",
            "held_out": "seed 20260911, 120 of 600 pairs, coefficients fit on the other 480",
        },
        predicted_output={
            "what_the_receiver_visible_oracle_allowed": 17_534.216,
        },
        empirical_output={
            "net_bytes_after_own_parameter_charge": CHARGED_CORRECTION_NET_BYTES,
            "parameter_bytes": CHARGED_CORRECTION_PARAM_BYTES,
            "joint_fraction_of_demand": 0.014887,
            "held_out_joint_gain_bytes_120_pairs": LS2_HELD_OUT_JOINT_GAIN_BYTES,
            "held_out_joint_gain_after_full_parameter_charge_bytes": (
                LS2_HELD_OUT_JOINT_GAIN_AFTER_FULL_PARAM_CHARGE_BYTES
            ),
            "lane_is_not_where_the_separate_model_gains": (
                "separate signed gross on Lane is -88.535929 B: its total is a Road gain that makes "
                "Lane worse. The joint model's Lane gain is only 47.158295 B"
            ),
            "receiver_state_bytes": LS2_RECEIVER_STATE_BYTES,
            "decision": "REFUSED_CANDIDATE_FIRE -- timing admission absent, not a negative gain",
            "timing_context_seconds": {
                "move44_t4_inflate": MOVE44_T4_INFLATE_SECONDS,
                "gate": DECODE_GATE_SECONDS,
                "margin": DECODE_GATE_SECONDS - MOVE44_T4_INFLATE_SECONDS,
            },
        },
        # The realised charged gain is 385.550480 B where the receiver-visible ORACLE allowed
        # 17,534.216 B: the oracle over-promises the realisable model by 45.5x.
        residual=0.97801,
        source_artifact=_LS2_LEDGER,
        measurement_method=(
            "fit two declared feature families on the retained decoder rows, charge every "
            "coefficient byte, price every one of the 117,964,800 symbols through the actual "
            "shipped RC64 frequency conversion, and bound the continuous box by supporting planes"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Oracle ladder on the shipped token field: what a tail model could know, and what it recovers",
        one_line_summary=(
            "The best RECEIVER-VISIBLE oracle is 8,364.784 B short of the 25,899 B demand with "
            "free tables; the charged realisation returns 385.550480 B (1.49 %)"
        ),
        latex_form=(
            r"\Delta B_{\mathrm{oracle}}(\mathcal{I}) = H(X) - H_{\mathrm{MM}}(X \mid \mathcal{I}),"
            r"\quad \max_{\mathcal{I}\ \mathrm{receiver\text{-}visible}} \Delta B = 17{,}534.216"
            r" < 25{,}899,\quad \Delta B_{\mathrm{charged}} = 385.55"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ls1_lane_surprise_atlas_oracle_ladder_20260911"
            ":oracle_shortfall_bytes"
        ),
        domain_of_validity={
            "object": (
                "the SHIPPED move-44 token field under the shipped RC64 coder and its counted "
                "mixer; ls1 measures the oracle, ls2 the charged realisation of one of its rungs"
            ),
            "generalizes_to": (
                "the STRUCTURE transfers: an oracle granted its tables is an upper bound on every "
                "model with the same information set, and a rung built on information the receiver "
                "cannot compute prices nothing"
            ),
            "excluded": (
                "the absolute byte figures on any other field (a pointer move re-measures them); "
                "full-resolution families outside ls2's two declared ones; any NATIVE receiver "
                "timing claim (ls2 has none, and that is why it refused rather than fired); and "
                "reading a Miller-Madow estimate as a certified universal ceiling"
            ),
            "support": (
                "n = 600 pairs / 117,964,800 symbols, full decode, independent recount PASS, "
                "instrument reconciled to the physical stream within 0.000584353 %"
            ),
            "verdict_this_priced": (
                "the RECEIVER-VISIBLE tail family is CLOSED at the measured level on THIS object. "
                "NOT closed by this: a different OBJECT (a born generator's field), a native "
                "receiver whose measured decode wall-clock fits the gate, or an information set "
                "the ladder never measured"
            ),
        },
        units_in={"information_set": "ladder rung name", "estimator": "plug_in | miller_madow"},
        units_out={
            "oracle_shortfall_bytes": "bytes still owed against the 25,899 B rate demand",
            "charged_fraction_of_demand": "dimensionless fraction (DERIVED)",
        },
        empirical_anchors=(ladder_anchor, charged_anchor),
        predicted_vs_empirical_residual={
            "oracle_over_promises_the_charged_realisation": 45.5,
            "charter_framing_paired_two_different_rows": 0.8762,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _LS2_LEDGER,
            ".omx/research/ddm_gs3_gestalt_after_submission_20260903.md",
            ".omx/research/ddm_obx1_successor_object_under_the_cross_design_20260911.md",
        ),
        canonical_producers=(
            "experiments/ddm_ls1_oracle_atlas.py",
            "experiments/ddm_ls1_verify_atlas.py",
            "experiments/ddm_ls2_probability_bound.py",
        ),
        provenance=provenance,
    )
