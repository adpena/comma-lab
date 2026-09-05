# SPDX-License-Identifier: MIT
"""Canonical equation: the HPAC prior-capacity slope on the SHIPPED fs2 mixer -- more learned
capacity, bought by lowering the trainer's rate multiplier, does NOT repay itself in stream bytes
(ddm_cl2, 2026-09-05; cl1's preregistered ladder, finally fired on the shipped object).

THE OBJECT.  The shipped token subsystem is a 13,515 B integer HPAC model (IHS1 pack of the
epoch-634 EMA of an rx2_mc36 Metal burn) plus a 113,411 B RC64 stream (fx1 fixed-point logistic
mixer on top) = 126,926 B joint, over the 600-pair semantic token field held bit-identical.  dc1
measured that the best 21-tap hand-context oracle loses to this learned receptive field by
32,057 B and concluded more capacity is "affordable only as learned weights".  cl1 preregistered
the cheapest test of that claim: hold the topology fixed and lower ``--rate-lambda`` (the
model-bit multiplier in the trainer's joint objective) 1.0 -> 0.5 -> 0.25, so the learned
per-channel bit depths may spend more model bytes to lower token cross-entropy.  Break-even for
adjacent rungs: ``delta(stream bytes) / delta(packed model bytes) < -1``.

THE INSTRUMENT (the reproduction gate cl1 owed and jf1 taught).  The same trainer, warm-started
from the shipped epoch-634 EMA state, 60-epoch cosine, seed 20260716, batch 8, QAT 0.5, on the
current field, on local Metal.  Priced through the shipped path itself: IHS1 pack + Brotli race,
``encode_tail`` (the receiver's decode walked with the symbols known; fx1 mixer in the loop),
RC64, receiver-copy decode identity.  Every number below is EXACT bytes on that path.

THE MEASURED LADDER (ddm_cl2, 2026-09-05):

  rung                     device   packed model B   stream B     joint B    d joint vs shipped
  shipped (fs2)              --        13,515        113,411     126,926           0
  jf2 null ep60 (control)   CPU        13,463        113,715     127,178        +252
  lambda 1.0 (control)      MPS        13,466        113,419     126,885         -41
  lambda 0.5                MPS        13,816        113,575     127,391        +465

  adjacent secant 1.0 -> 0.5:  d model +350 B, d stream +156 B, slope +0.446  (bar: < -1)

THE LAW.  On this object the prior is NOT under-capacity in the direction the multiplier can
buy: releasing the model-bit constraint by 2x grows the packed model by +350 B and makes the
stream WORSE by +156 B -- the extra bits go into weight precision the coder does not need, and
the perturbed fit codes the field less well.  The prediction (model <= +1,500 B, stream <= -2x
that, net <= -1,500 B) is falsified at the first rung; the ladder's lambda 0.25 rung is not
fired (the preregistered fire condition, slope < -1, is not met).  Every lower-lambda rung on
this topology is closed at formulation scope.

WHAT DID MOVE.  The control rung itself: re-fitting the shipped weights under the reference law
and re-packing gives -49 B of model for +8 B of stream, -41 B joint, with the field held.  That
is a retrain / pack-size effect, NOT a capacity win, and it is the only pointer-relevant number
this ladder produced (-2.73e-5 S rate-only).  Its admissibility rests on run-to-run bit identity
of the training (the uninterrupted twin), recorded in the memo.

VERDICT.  Prior-law FALSIFIED; lambda ladder CLOSED (formulation: fixed C64/P64/delta2/D8
topology, 60-epoch warm-start law, multiplier in {1, 1/2}).  Axis
``[macOS-CPU advisory / scorer-free EXACT byte measurement]``; training
``[macOS-MPS research-signal]``.  NON-PROMOTABLE by itself; the control candidate is sealed
separately if its twin proves deterministic.

Producer: ``experiments/ddm_cl2_hpac_prior_capacity_ladder.py`` (stages control / price / verify /
parseback / report) via ``.omx/research/ddm_cl2_hpac_prior_capacity_ladder_on_shipped_object_20260905.md``.
Consumers: any charter that proposes buying HPAC capacity through the trainer's rate multiplier,
a wider trunk, or a longer warm-start schedule on the shipped object; sister laws ``ddm_dc1``
(the 21-tap oracle floor this tested), ``ddm_mc1`` (the context-INPUT door, closed 2026-09-05),
``ddm_jf2`` (the joint field+model diagonal, distortion-dead).
"""

from __future__ import annotations

from collections.abc import Mapping

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

EQUATION_ID = "hpac_prior_capacity_slope_v1"

_UTC = "2026-09-05T15:10:00Z"
_AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"
_LEDGER = ".omx/research/ddm_cl2_hpac_prior_capacity_ladder_on_shipped_object_20260905.md"
_CHARTER = ".omx/research/charters/ddm_cl2_hpac_prior_capacity_ladder_on_shipped_object_20260905.md"
_PRODUCER = "experiments/ddm_cl2_hpac_prior_capacity_ladder.py"

# --- the shipped object (fs2 pointer, MEASURED) --------------------------------------------
SHIPPED_MODEL_BYTES = 13_515
SHIPPED_STREAM_BYTES = 113_411
SHIPPED_JOINT_BYTES = SHIPPED_MODEL_BYTES + SHIPPED_STREAM_BYTES  # 126,926
SHIPPED_ARCHIVE_BYTES = 180_023
FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
STREAM_SHA256 = "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
S_PER_BYTE = 25.0 / 37_545_489

# --- the rungs (ddm_cl2, all EXACT bytes through the shipped path) ---------------------------
RUNG_MODEL_BYTES: Mapping[str, int] = {
    "cpu_control_jf2_null": 13_463,
    "lambda_1p0": 13_466,
    "lambda_0p5": 13_816,
}
RUNG_STREAM_BYTES: Mapping[str, int] = {
    "cpu_control_jf2_null": 113_715,
    "lambda_1p0": 113_419,
    "lambda_0p5": 113_575,
}
RUNG_LAMBDA: Mapping[str, float] = {
    "cpu_control_jf2_null": 1.0,
    "lambda_1p0": 1.0,
    "lambda_0p5": 0.5,
}
CONTROL_TOLERANCE_BYTES = 500  # the charter's INSTRUMENT-REFUSED threshold on the control's joint gap
BREAK_EVEN_SLOPE = -1.0  # cl1's adjacent-rung break-even: d stream / d model < -1 pays
PRIOR_LAW_MAX_MODEL_GROWTH_BYTES = 1_500.0
PRIOR_LAW_MIN_STREAM_SAVING_FACTOR = 2.0  # stream must fall by >= 2x the model growth
PRIOR_LAW_NET_JOINT_BYTES = -1_500.0

# --- derived, re-computed by the guards ---------------------------------------------------------
CONTROL_JOINT_DELTA_BYTES = (
    RUNG_MODEL_BYTES["lambda_1p0"] + RUNG_STREAM_BYTES["lambda_1p0"] - SHIPPED_JOINT_BYTES
)  # -41
CPU_CONTROL_JOINT_DELTA_BYTES = (
    RUNG_MODEL_BYTES["cpu_control_jf2_null"] + RUNG_STREAM_BYTES["cpu_control_jf2_null"] - SHIPPED_JOINT_BYTES
)  # +252
LAMBDA_0P5_DELTA_MODEL_BYTES = RUNG_MODEL_BYTES["lambda_0p5"] - RUNG_MODEL_BYTES["lambda_1p0"]  # +350
LAMBDA_0P5_DELTA_STREAM_BYTES = RUNG_STREAM_BYTES["lambda_0p5"] - RUNG_STREAM_BYTES["lambda_1p0"]  # +156
LAMBDA_0P5_SLOPE = LAMBDA_0P5_DELTA_STREAM_BYTES / LAMBDA_0P5_DELTA_MODEL_BYTES  # +0.4457
LAMBDA_0P5_JOINT_DELTA_VS_CONTROL_BYTES = LAMBDA_0P5_DELTA_MODEL_BYTES + LAMBDA_0P5_DELTA_STREAM_BYTES  # +506
LAMBDA_0P5_JOINT_DELTA_VS_SHIPPED_BYTES = (
    RUNG_MODEL_BYTES["lambda_0p5"] + RUNG_STREAM_BYTES["lambda_0p5"] - SHIPPED_JOINT_BYTES
)  # +465


def joint_bytes(model_bytes: int, stream_bytes: int) -> int:
    """The token subsystem's counted bytes: packed model + RC64 stream (the field is held)."""
    return int(model_bytes) + int(stream_bytes)


def control_reproduces_shipped_family(
    control_joint_bytes: int,
    shipped_joint_bytes: int = SHIPPED_JOINT_BYTES,
    tolerance_bytes: int = CONTROL_TOLERANCE_BYTES,
) -> bool:
    """The instrument gate: the lambda=1 control must land within ``tolerance_bytes`` ABOVE the
    shipped joint (below is fine).  Failing it makes the ladder INSTRUMENT-REFUSED, never a
    family verdict (jf1's lesson: an epoch-2 refit was 7,554 B weaker)."""
    return int(control_joint_bytes) - int(shipped_joint_bytes) <= int(tolerance_bytes)


def adjacent_slope(delta_stream_bytes: int, delta_model_bytes: int) -> float | None:
    """cl1's secant between adjacent rungs; ``None`` when the model did not move."""
    if int(delta_model_bytes) == 0:
        return None
    return float(delta_stream_bytes) / float(delta_model_bytes)


def rung_pays(delta_stream_bytes: int, delta_model_bytes: int, break_even: float = BREAK_EVEN_SLOPE) -> bool:
    """A rung that grows the model by ``d_model > 0`` pays only if the stream falls by MORE than
    that (slope < -1).  When the model does not grow there is nothing to repay: pays iff the
    joint fell."""
    slope = adjacent_slope(delta_stream_bytes, delta_model_bytes)
    if int(delta_model_bytes) > 0 and slope is not None:
        return slope < float(break_even)
    return int(delta_model_bytes) + int(delta_stream_bytes) < 0


def next_rung_admitted(delta_stream_bytes: int, delta_model_bytes: int) -> bool:
    """The preregistered fire condition for the next lower lambda: the previous secant paid."""
    return rung_pays(delta_stream_bytes, delta_model_bytes)


def prior_law_prediction_holds(delta_model_bytes: int, delta_stream_bytes: int) -> bool:
    """dc1-derived prediction for lambda 1.0 -> 0.5: model <= +1,500 B and stream <= -2x that."""
    return (
        float(delta_model_bytes) <= PRIOR_LAW_MAX_MODEL_GROWTH_BYTES
        and float(delta_stream_bytes) <= -PRIOR_LAW_MIN_STREAM_SAVING_FACTOR * max(float(delta_model_bytes), 0.0)
        and float(delta_model_bytes) + float(delta_stream_bytes) <= PRIOR_LAW_NET_JOINT_BYTES
    )


def rate_only_delta_s(delta_archive_bytes: int) -> float:
    """Rate-term S change of a byte delta at held distortion (25 / 37,545,489 per byte)."""
    return float(delta_archive_bytes) * S_PER_BYTE


def _anchor_control() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="cl2_control_rung_reproduces_shipped_joint_minus_41_bytes_20260905",
        measurement_utc="2026-09-05T14:50:00Z",
        inputs={
            "field_sha256": FIELD_SHA256,
            "shipped_stream_sha256": STREAM_SHA256,
            "law": "tools/train_ddm_cl1_hpac_capacity.py --profile cl2_shipped_ladder, warm start ep634 EMA (ff2d3e45...), 60-epoch cosine, seed 20260716, batch 8, QAT 0.5, rate_lambda 1.0, device mps",
            "path": "IHS1 pack + Brotli q0..q11 race; jg2.encode_tail on the fs2 fire tree (fx1 mixer, RC64); receiver-copy decode",
            "cpu_control": "jf2's terminal epoch-60 null checkpoint (sha 3aca9dbb...), same law on CPU, priced through the same path",
            "producer": f"{_PRODUCER} price --rung lambda_1p0 / cpu_control_jf2_null",
        },
        predicted_output={
            "charter_gate": f"control joint within +{CONTROL_TOLERANCE_BYTES} B of {SHIPPED_JOINT_BYTES} B, else INSTRUMENT-REFUSED",
            "jf2_prior": "+314 B joint on the dx2 path (CPU)",
        },
        empirical_output={
            "lambda_1p0_model_bytes": RUNG_MODEL_BYTES["lambda_1p0"],
            "lambda_1p0_stream_bytes": RUNG_STREAM_BYTES["lambda_1p0"],
            "lambda_1p0_joint_delta_vs_shipped": CONTROL_JOINT_DELTA_BYTES,
            "cpu_control_model_bytes": RUNG_MODEL_BYTES["cpu_control_jf2_null"],
            "cpu_control_stream_bytes": RUNG_STREAM_BYTES["cpu_control_jf2_null"],
            "cpu_control_joint_delta_vs_shipped": CPU_CONTROL_JOINT_DELTA_BYTES,
            "two_encodes_identical": True,
            "receiver_decode_identity": True,
            "reading": "the Metal control lands 41 B BELOW the shipped joint (-49 B model, +8 B stream); the CPU control +252 B; both inside the +500 B gate -- the instrument is the shipped law on this object",
        },
        residual=abs(float(CONTROL_JOINT_DELTA_BYTES)),
        source_artifact=_LEDGER,
        measurement_method="exact packed bytes and exact RC64 stream bytes on the shipped encode path; receiver-copy decode compared to the retained field byte for byte",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria="a reference law whose control lands outside +500 B would re-open the instrument question, not the family",
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_metal_and_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _anchor_slope() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="cl2_lambda_0p5_secant_positive_prior_law_falsified_20260905",
        measurement_utc="2026-09-05T15:05:00Z",
        inputs={
            "rungs": {"lambda_1p0": 1.0, "lambda_0p5": 0.5},
            "same_law": "identical config, init, cache, seed and device; only --rate-lambda differs",
            "break_even_slope": BREAK_EVEN_SLOPE,
            "producer": f"{_PRODUCER} price --rung lambda_0p5; report",
        },
        predicted_output={
            "prior_law": "model <= +1,500 B, stream <= -2x that, net joint <= -1,500 B; 0.25 continues with a shallower slope",
            "falsifier": "lambda 0.5's net joint >= 0 vs the control",
        },
        empirical_output={
            "lambda_0p5_model_bytes": RUNG_MODEL_BYTES["lambda_0p5"],
            "lambda_0p5_stream_bytes": RUNG_STREAM_BYTES["lambda_0p5"],
            "delta_model_bytes": LAMBDA_0P5_DELTA_MODEL_BYTES,
            "delta_stream_bytes": LAMBDA_0P5_DELTA_STREAM_BYTES,
            "slope_stream_per_model": LAMBDA_0P5_SLOPE,
            "joint_delta_vs_control": LAMBDA_0P5_JOINT_DELTA_VS_CONTROL_BYTES,
            "joint_delta_vs_shipped": LAMBDA_0P5_JOINT_DELTA_VS_SHIPPED_BYTES,
            "lambda_0p25_fired": False,
            "reading": "the model grew +350 B and the stream got WORSE by +156 B (slope +0.446 vs the -1 bar): the multiplier buys weight precision the coder does not use; falsifier FIRED at the first rung",
        },
        # the gap between the prediction's net joint and the measured net joint, in bytes
        residual=float(LAMBDA_0P5_JOINT_DELTA_VS_CONTROL_BYTES - PRIOR_LAW_NET_JOINT_BYTES),
        source_artifact=_LEDGER,
        measurement_method="exact adjacent secant of two same-law rungs through the shipped pack + mixer + RC64 path",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria="a DIFFERENT capacity coordinate (width, frame_dim, receptive-field shape) measured through the same instrument with a control inside +500 B; the multiplier coordinate is closed",
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_metal_and_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_hpac_prior_capacity_slope_v1() -> CanonicalEquation:
    """Build the prior-capacity slope law for the shipped HPAC mixer (ddm_cl2, 2026-09-05)."""
    control = _anchor_control()
    slope = _anchor_slope()
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "HPAC prior-capacity slope on the shipped mixer -- lowering the trainer's model-bit "
            "multiplier grows the packed model and does not lower the exact stream"
        ),
        one_line_summary=(
            "cl2: control reproduces the shipped joint at -41 B (instrument PASS); lambda 1.0 -> 0.5 costs "
            "+350 B of model for +156 B of stream (slope +0.446 vs the -1 bar); prior law falsified, 0.25 not fired"
        ),
        latex_form=(
            r"J(\lambda)=B_{\text{model}}(\lambda)+B_{\text{stream}}(\lambda),\ \text{field held};\quad"
            r"\text{pays}\iff \frac{\Delta B_{\text{stream}}}{\Delta B_{\text{model}}}<-1;\quad"
            r"\text{MEASURED: } \frac{+156}{+350}=+0.446,\ J(1)=126{,}885,\ J(\tfrac12)=127{,}391,\ J_{\text{shipped}}=126{,}926"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.hpac_prior_capacity_slope_20260905:rung_pays"
        ),
        domain_of_validity={
            "included": [
                "the shipped semantic_joint_ctxmix integer HPAC mixer (C64/P64/delta2/D8, fx1 mixer, RC64) on the 600-pair token field cc10a7b0...",
                "capacity bought through the trainer's --rate-lambda multiplier under the 60-epoch warm-start reference law",
                "adjacent-rung secants of exact packed-model and exact stream bytes",
            ],
            "excluded": [
                "use as a d_seg / d_pose / score claim; the field is held so distortion never moved",
                "architectural capacity (width, frame_dim, receptive-field shape): NOT measured here -- a different coordinate",
                "the control's -41 B as a capacity effect: it is a retrain / pack-size effect and is sealed separately on its own determinism proof",
            ],
            "measurement_axis": [_AXIS, "[macOS-MPS research-signal] (training)"],
            "result_type": "CAPACITY-COORDINATE closure at formulation scope; the control candidate is handled by its own seal",
            "sister_laws": [
                "ddm_dc1 -- the 21-tap oracle floor (+32,057 B) that motivated 'affordable only as learned weights'",
                "ddm_mc1 -- motion_compensated_previous_plane_alignment_gate_v1 (the context-INPUT door, CEILING-REFUSED)",
                "ddm_jf2 -- the joint field+model diagonal, byte-alive but distortion-dead",
            ],
            "known_boundary": "two rungs of one multiplier on one object; the CPU control differs from the Metal control by 293 B joint (device numerics), inside the gate",
            "verdict_scope": "formulation (fixed topology, multiplier in {1, 1/2}, warm-start law)",
        },
        units_in={
            "model_bytes": "bytes",
            "stream_bytes": "bytes",
            "delta_model_bytes": "bytes",
            "delta_stream_bytes": "bytes",
            "delta_archive_bytes": "bytes",
        },
        units_out={
            "joint_bytes": "bytes",
            "adjacent_slope": "bytes_per_byte",
            "rung_pays": "bool",
            "control_reproduces_shipped_family": "bool",
            "prior_law_prediction_holds": "bool",
            "rate_only_delta_s": "score_units",
        },
        empirical_anchors=(control, slope),
        predicted_vs_empirical_residual={
            control.anchor_id: control.residual,
            slope.anchor_id: slope.residual,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_LEDGER, _CHARTER),
        canonical_producers=(_PRODUCER, _LEDGER),
        provenance=build_provenance_for_predicted(
            model_id="hpac_prior_capacity_slope.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_metal_and_cpu",
        ),
    )
