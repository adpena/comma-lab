# SPDX-License-Identifier: MIT
"""Canonical equation: the refit debt of a counted section fit to a superseded field (ddm_hpr1).

THE PROPOSED LAW.  **A counted section fit to a superseded state of the object is owed a refit
before any structural rung on it is priced.**  Its byte decomposition is exactly two terms:

    Delta B = Delta model + Delta tail

where ``Delta model`` is the counted growth of the section's own weights (they are video-derived,
so every byte of them ships) and ``Delta tail`` is the change in the stream those weights
condition.  A refit is admissible when the sum is negative; nothing structural moves, so the
receiver is unchanged and only the rate term can move.

THE ONE MEASURED INSTANCE (move 47, contest-CUDA T4 n600, real coder, real receiver loop, twin
encodes agreeing, decoded token field byte-identical to the shipped field):

    leg                          move 45        retrained        delta
    hpac member (the model)      11,911 B       12,262 B         Delta model = +351 B
    RLC1 stream (the tail)      119,749 B      118,511 B         Delta tail   = -1,238 B
    archive                     180,246 B      179,359 B         Delta B      =   -887 B

``Delta S`` = -5.906e-4, ``receiver_change: false``, archive sha ``d1fab05d69f31c90...``.  It was
3.45 % of the whole remaining gap to sub-0.12 at move 45, and it was found by ACCIDENT: it was the
CONTROL of a shape experiment whose shape rung was falsified (+812 B against this control).

THE CAUSE, MEASURED.  The shipped prior's weights descend from cl2's lambda=1.0 fit at **move 26**.
The token field then moved EIGHT times -- moves 31, 32, 35, 38, 39, 40, 42 and 43 -- last at move 43
(sj1 pass 6), producing the current ``a92e7d90...``; moves 44-47 are field-invariant by receipt.  So
the prior spent thirteen moves coding a field it was never fit to, and the -887 B was the
accumulated bill.  (The arm first wrote "the field moved at move 32"; that is a FIRST-change
statement and is wrong as a staleness figure.  The correction makes the staleness WORSE.)

WHAT A REFIT IS, OPERATIONALLY.  60 epochs under cl2's own law, warm-started from the shipped
prior itself, in the SHIPPED GEOMETRY -- no shape bit ships, the geometry lives in receiver code.
It replaces all of the prior's weights inside the archive member.  Output-lossless by receipt: the
decoded token field and the cold n600 public decode are byte-identical to the pointer's, so d_seg
and d_pose do not move at all.

NO SECANT.  A secant (Delta tail per Delta model) is the instrument for a CAPACITY rung, and the
arm explicitly refuses to attach one here: "a refit, not a capacity rung".  The only secant hpr1
reports, -1.2066, belongs to the shape rung R1 -- which was FALSIFIED.  cl2's +0.446 capacity
secant does not transfer either.  Registering a secant for this instance would import an
instrument its own producer rejected; the ratio -1238/351 = -3.53 is arithmetic, not a law.

n = 1, SAID OUT LOUD.  ddm_hpr1's staleness audit sets its own bar: register the law "only when a
second section's refit prices it, the hpac refit being n=1".  ddm_cons1 registers the measured
INSTANCE and its decomposition so it stops being an orphan, and carries the n=1 label in the
domain rather than claiming a general refit predictor.  The audit's ranked queue names the test
that would make it a law: the tc1 tail mixer's 35 int8 weights, fit to the move-32 field, 11 moves
stale, where 60 B of state prices 118,511 B.  The ``semantic`` member (29,862 B) is STALE and its
staleness CANNOT EVEN BE DATED -- the store names the checkpoint and not its training data.

Axis ``[contest-CUDA T4 n600]`` for the move-47 row.  Registered by ddm_cons1 on 2026-09-11 from
ddm_hpr1's memos, read-only; cons1 launched nothing and wrote nothing into the hpr1 tree.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "hpr1_counted_section_refit_debt_v1"

_UTC = "2026-09-11T00:00:00Z"
_AXIS = "[contest-CUDA T4 n600]"
_SHAPE_MEMO = ".omx/research/ddm_hpr1_hpac_receptive_field_shape_rung_20260911.md"
_MOVE_MEMO = (
    ".omx/research/ddm_hpr1_retrain_control_move46_contest_cuda_20260911_"
    "pointer_move_47_20260911.md"
)
_AUDIT = ".omx/research/ddm_hpr1_staleness_audit_20260911.md"
_PRODUCER = "experiments/ddm_hpr1_shape_price.py"

# MEASURED, move 47 vs move 45, exact archive bytes.
MODEL_BYTES_BEFORE = 11_911
MODEL_BYTES_AFTER = 12_262
TAIL_BYTES_BEFORE = 119_749
TAIL_BYTES_AFTER = 118_511
ARCHIVE_BYTES_BEFORE = 180_246
ARCHIVE_BYTES_AFTER = 179_359
DELTA_S = -5.906e-4
EXCHANGE_RATE_S_PER_BYTE = 6.658589531221714e-07
SHARE_OF_REMAINING_GAP_TO_SUB_012 = 0.0345
ARCHIVE_SHA256_AFTER = "d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9"

# The staleness itself.
PRIOR_FIT_AT_MOVE = 26
FIELD_MOVED_AT_MOVES = (31, 32, 35, 38, 39, 40, 42, 43)
REFIT_EPOCHS = 60
SUPPORT_N_SECTIONS = 1


def delta_b(delta_model: float, delta_tail: float) -> float:
    """The refit's archive-byte change: the model it adds plus the tail it buys."""
    return delta_model + delta_tail


def refit_is_admissible(delta_model: float, delta_tail: float) -> bool:
    """A refit pays when the counted weights it grows cost less than the stream they buy."""
    return delta_b(delta_model, delta_tail) < 0


def measured_instance() -> dict[str, float]:
    """The single measured instance, re-derived from its own leg bytes (never a quoted total)."""
    d_model = MODEL_BYTES_AFTER - MODEL_BYTES_BEFORE
    d_tail = TAIL_BYTES_AFTER - TAIL_BYTES_BEFORE
    return {
        "delta_model_bytes": d_model,
        "delta_tail_bytes": d_tail,
        "delta_b_bytes": delta_b(d_model, d_tail),
        "delta_b_from_archive_bytes": ARCHIVE_BYTES_AFTER - ARCHIVE_BYTES_BEFORE,
        "delta_s": DELTA_S,
        "staleness_moves": len(FIELD_MOVED_AT_MOVES),
    }


def build_hpr1_counted_section_refit_debt_v1() -> CanonicalEquation:
    """Build the counted-section refit-debt equation (ddm_hpr1, move 47, n=1 section)."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_AUDIT,
        reactivation_criteria=(
            "this is n = 1 and is registered as a measured INSTANCE, not a general refit "
            "predictor. It becomes a law when a SECOND counted section's refit prices it -- the "
            "audit's ranked queue names the test: the tc1 tail mixer's 35 int8 weights, fit to "
            "the move-32 field and 11 moves stale, where 60 B of state prices 118,511 B. The "
            "`semantic` member is stale and its staleness cannot even be dated until the store "
            "records a checkpoint's TRAINING DATA and not only its hash"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="contest-CUDA T4 (n600 exact eval) + real coder/receiver loop",
        captured_at_utc=_UTC,
    )

    refit_anchor = EmpiricalAnchor(
        anchor_id="hpr1_hpac_prior_refit_move47_contest_cuda_n600_20260911",
        measurement_utc=_UTC,
        inputs={
            "section": "the archive's `hpac` member -- the arithmetic coder's IntegerHPAC prior",
            "shipped_prior_fit_at": f"cl2 lambda=1.0 fit at move {PRIOR_FIT_AT_MOVE}",
            "field_moved_at_moves": FIELD_MOVED_AT_MOVES,
            "refit": (
                f"{REFIT_EPOCHS} epochs under cl2's own law, warm-started from the shipped prior, "
                "in the SHIPPED geometry -- no shape bit ships, receiver unchanged"
            ),
            "control_role": (
                "this was the CONTROL of a shape experiment; the shape rung R1 was falsified "
                "(+812 B against this control)"
            ),
            "instrument": "twin encodes, 600 frames, real coder and real receiver loop",
        },
        predicted_output={
            "charter_expectation": (
                "the experiment's value is the SHAPE rung (a receptive-field change); the "
                "retrain control is a null arm expected to move nothing"
            ),
            "expected_control_delta_b": "0 B",
        },
        empirical_output={
            "model_bytes": [MODEL_BYTES_BEFORE, MODEL_BYTES_AFTER],
            "tail_bytes": [TAIL_BYTES_BEFORE, TAIL_BYTES_AFTER],
            "archive_bytes": [ARCHIVE_BYTES_BEFORE, ARCHIVE_BYTES_AFTER],
            "delta_model_bytes": 351,
            "delta_tail_bytes": -1238,
            "delta_b_bytes": -887,
            "delta_s": DELTA_S,
            "share_of_remaining_gap_to_sub_012": SHARE_OF_REMAINING_GAP_TO_SUB_012,
            "receiver_change": False,
            "output_lossless": (
                "decoded token field byte-identical to the shipped field a92e7d90...; the cold "
                "n600 public decode byte-identical to the pointer's raw -- d_seg and d_pose "
                "unmoved, only the rate term moves"
            ),
            "archive_sha256": ARCHIVE_SHA256_AFTER,
            "no_secant": (
                "the arm refuses a secant for a refit ('a refit, not a capacity rung'); the only "
                "secant it reports, -1.2066, belongs to the FALSIFIED shape rung R1, and cl2's "
                "+0.446 capacity secant does not transfer"
            ),
            "verdict": (
                "the shape rung is FALSIFIED at INSTANCE scope (superseded, not un-recoverable); "
                "its CONTROL became pointer move 47"
            ),
        },
        # |predicted 0 B for the control - measured -887 B| / 887 B
        residual=1.0,
        source_artifact=_MOVE_MEMO,
        measurement_method="twin_encodes_600_frames_real_coder_exact_archive_bytes_contest_cuda_n600",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Refit debt of a counted section fit to a superseded field (Delta B = Delta model + Delta tail)",
        one_line_summary=(
            "A counted section fit to a superseded field is owed a refit before any structural "
            "rung on it is priced: hpac, 13 moves stale, paid +351 model for -1,238 tail = -887 B"
        ),
        latex_form=(
            r"\Delta B = \Delta_{\mathrm{model}} + \Delta_{\mathrm{tail}},\quad "
            r"\text{admissible} \iff \Delta B < 0;\quad "
            r"(+351) + (-1238) = -887\ \mathrm{B},\ \Delta S = -5.906\times10^{-4}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.hpr1_counted_section_refit_debt_20260911:delta_b"
        ),
        domain_of_validity={
            "applies_to": (
                "a COUNTED archive section whose contents are learned from the object and whose "
                "refit leaves the receiver unchanged (output-lossless, geometry in receiver code)"
            ),
            "support": (
                f"n = {SUPPORT_N_SECTIONS} section (the hpac prior). The law is PROPOSED, not "
                "established: one measured instance. ddm_hpr1's own bar is a SECOND section's refit"
            ),
            "excluded": (
                "capacity rungs (a section GROWN rather than refit) -- their instrument is the "
                "secant and this equation deliberately carries none; and any refit that changes "
                "the receiver, which must be priced as a receiver change with a decode wall-clock"
            ),
            "known_stale_sections_at_move_47": (
                "`semantic` 29,862 B (stale, UNDATABLE -- the store names the checkpoint, not its "
                "training data); the tc1 tail mixer's 35 int8 weights (fit to the move-32 field)"
            ),
        },
        units_in={"delta_model": "archive bytes", "delta_tail": "archive bytes"},
        units_out={
            "delta_b": "archive bytes",
            "refit_is_admissible": "boolean",
        },
        empirical_anchors=(refit_anchor,),
        predicted_vs_empirical_residual={"control_expected_null_delta_b": 1.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_AUDIT, _MOVE_MEMO, _SHAPE_MEMO),
        canonical_producers=(_PRODUCER, "experiments/ddm_hpr1_shape_inputs.py"),
        provenance=provenance,
    )
