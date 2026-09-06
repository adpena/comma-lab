# SPDX-License-Identifier: MIT
"""Canonical equation: model-coder strength SUBSTITUTES for the capacity lever -- improving the
model coder makes every capacity rung measured through model bytes worth LESS, not more
(ddm_cl3, 2026-09-06; measured on one weight change priced through two containers).

THE SETUP.  With the token field held bit-identical, the token subsystem's counted bytes are
``J = B_model + B_stream``.  A capacity rung changes the WEIGHTS: it moves ``B_model`` (the model
section) and, through the changed probabilities, ``B_stream`` (the arithmetic-coded field).  ddm_cl2
priced its ladder when ``B_model`` reached the archive as ``brotli(raw IHS1)``.  ddm_rc1 then
replaced that generic byte coder with an adaptive per-group binary-tree range coder (the ``RC1H``
rider), which alone bought -1,123 B on the SAME weights.  ddm_cl3's lambda = 2.0 rung was therefore
priced on BOTH containers, from one checkpoint, giving a controlled read of what the coder change
does to a capacity measurement.

THE MEASUREMENT (ddm_cl3, one rung, two containers, EXACT bytes):

  basis                       d B_model    d B_stream       d J        reading
  cl2's Brotli (old object)      -659         +681          +22        essentially break-even
  rc1 RC1H (the live object)     -457         +681         +224        clearly loses

``d B_stream`` is IDENTICAL in both rows because it is a property of the weights, not of the model
coder.  That premise is MEASURED, not assumed: the control weights encode to a 113,419 B stream on
ddm_cl2's fs2 tree AND to a 113,419 B stream (sha ``e07274ca...``) on the live pc1 tree, whose model
coder and carrier both differ -- same weights, same stream, two different objects.  Only the
model-side saving moved: the strong coder returned 457 B where the weak coder returned 659 B, i.e.
it captured 69.3% as much for the same weight shrink.

THE LAW.  For a fixed weight change with the field held,

    d J(coder) = d B_model(coder) + d B_stream,     d B_stream independent of the model coder,
    |d B_model(strong)| <= |d B_model(weak)|,

so ``d J`` is MONOTONICALLY WORSE under a stronger model coder.  The mechanism is not subtle: a
strong coder was already coding the larger model near its entropy -- that is what earned it its own
win -- so much of what the capacity lever removes is redundancy the weak coder was FAILING to
capture and the strong coder had ALREADY captured.  The weak coder books that redundancy twice
(once in its baseline, once in the shrink); the strong coder books it once.  **Coder quality and
capacity-lever value draw on the same pool: they are SUBSTITUTES, not complements.**

WHY THIS MATTERS BEYOND THE RUNG.

1. **Byte deltas do not transfer across coders.**  A capacity delta measured on a weak coder
   OVERSTATES what the same weight change is worth once the coder is strong -- here by ~31% on the
   model side, which was the whole margin: +22 B (a coin-flip) became +224 B (a clear loss).
2. **Ordering matters between coder work and capacity work.**  Landing the coder first does not only
   bank its own bytes; it REMOVES headroom from every downstream capacity lever.  An arm pricing
   this ladder before rc1 would have read lambda = 2.0 as break-even and plausibly chased lambda = 4.0
   -- chasing a lever the coder had already spent.
3. It is a falsifier for the reflex "a better coder makes everything cheaper".  It makes the SHIPPED
   object cheaper and every model-byte lever measured against it POORER.

WHAT THIS IS NOT.  It is not a claim about d_seg / d_pose: the field is held bit-identical on every
row, so distortion never moved.  It is not a claim that capacity is worthless -- it bounds how much
of a capacity saving a given coder can return.  It does not say the coder change was wrong: rc1's
-1,123 B is banked and larger than anything this ladder found.

PREDICTED vs MEASURED.  ddm_cl3 pre-registered the OPPOSITE (memo section 4d, 23:33Z, before any
exact number existed): that rc1 would show a LARGER model saving than Brotli, because lambda = 2.0
moves mass into zero-width channels and low depths and the adaptive coder models the code alphabet
directly while Brotli never sees a code boundary.  Measured: 457 B against 659 B -- the prediction
was inverted, by 202 B.  The error was treating coder quality and capacity value as complements.

VERDICT.  Established at INSTANCE scope for these two containers on this object, with the mechanism
stated at family scope and its transfer rule (discount, or re-price) given above.  Axis
``[macOS-CPU advisory / scorer-free EXACT byte measurement]``; training
``[macOS-MPS research-signal]``.  NON-PROMOTABLE by itself: no scorer ran and no pointer moved.

Producer: ``experiments/ddm_cl3_rc1_rung_price.py`` via
``.omx/research/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md``.
Consumers: any charter that transfers a capacity-lever byte delta across a coder change; any
sequencing decision between coder work and capacity work; sister law
``hpac_prior_capacity_slope_v1`` (ddm_cl2), whose ladder deltas were all priced on the weak coder.
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

EQUATION_ID = "coder_strength_substitutes_for_capacity_v1"

_UTC = "2026-09-06T00:40:00Z"
_AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"
_LEDGER = ".omx/research/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md"
_PRODUCER = "experiments/ddm_cl3_rc1_rung_price.py"

# --- the one weight change, priced on two containers (ddm_cl3, EXACT bytes) ------------------
#: The live pointer's model section under each container, on the CONTROL weights (lambda = 1.0).
CONTROL_MODEL_BYTES_WEAK = 13_466  # brotli(raw IHS1), cl2's object
CONTROL_MODEL_BYTES_STRONG = 12_343  # rc1 RC1H rider + ck2 + brotli, the live object
#: The same, on the lambda = 2.0 weights.
RUNG_MODEL_BYTES_WEAK = 12_807
RUNG_MODEL_BYTES_STRONG = 11_886
#: The stream is a property of the WEIGHTS, not of the model coder: one number for both bases.
CONTROL_STREAM_BYTES = 113_419
RUNG_STREAM_BYTES = 114_100
#: The premise, MEASURED rather than assumed: the SAME control weights encode to this same stream
#: on ddm_cl2's fs2 tree and on the live pc1 tree, which differ in BOTH model coder and carrier.
CONTROL_STREAM_BYTES_ON_WEAK_CODER_TREE = 113_419

DELTA_MODEL_WEAK = RUNG_MODEL_BYTES_WEAK - CONTROL_MODEL_BYTES_WEAK  # -659
DELTA_MODEL_STRONG = RUNG_MODEL_BYTES_STRONG - CONTROL_MODEL_BYTES_STRONG  # -457
DELTA_STREAM = RUNG_STREAM_BYTES - CONTROL_STREAM_BYTES  # +681
DELTA_JOINT_WEAK = DELTA_MODEL_WEAK + DELTA_STREAM  # +22
DELTA_JOINT_STRONG = DELTA_MODEL_STRONG + DELTA_STREAM  # +224
#: What the strong coder returned, as a fraction of what the weak coder returned.
CAPTURE_FRACTION = abs(DELTA_MODEL_STRONG) / abs(DELTA_MODEL_WEAK)  # 0.693...
#: The coder's own standalone win on the SAME weights, from ddm_rc1 (context, not an input here).
CODER_OWN_WIN_BYTES = CONTROL_MODEL_BYTES_STRONG - CONTROL_MODEL_BYTES_WEAK  # -1,123


def joint_delta(delta_model_bytes: int, delta_stream_bytes: int) -> int:
    """The token subsystem's joint byte delta with the field held: model + stream."""

    return int(delta_model_bytes) + int(delta_stream_bytes)


def rung_pays(delta_model_bytes: int, delta_stream_bytes: int) -> bool:
    """A capacity rung pays iff the JOINT falls.  Direction-agnostic by construction."""

    return joint_delta(delta_model_bytes, delta_stream_bytes) < 0


def coder_capture_fraction(strong_model_saving: float, weak_model_saving: float) -> float:
    """Fraction of a weak coder's model saving that a strong coder returns for the same weights.

    Both arguments are SAVINGS (positive magnitudes).  A value below 1.0 is the substitution this
    law names; 1.0 would mean the coder change is neutral for the capacity lever; above 1.0 would
    falsify the law.
    """

    if weak_model_saving <= 0.0:
        raise ValueError("weak_model_saving must be a positive magnitude")
    return float(strong_model_saving) / float(weak_model_saving)


def transfer_capacity_delta(weak_basis_model_saving: float, capture_fraction: float = CAPTURE_FRACTION) -> float:
    """Discount a model-side capacity saving measured on a WEAK coder onto a STRONG one.

    This is the transfer rule the law exists to supply: never carry a cl2-basis (Brotli) model
    delta onto the rc1-coded object undiscounted.  The stream side needs no discount -- it does not
    depend on the model coder.
    """

    if not 0.0 < capture_fraction <= 1.0:
        raise ValueError("capture_fraction must lie in (0, 1]")
    return float(weak_basis_model_saving) * float(capture_fraction)


def substitution_holds(strong_model_saving: float, weak_model_saving: float) -> bool:
    """True when the strong coder returns LESS than the weak one for the same weight change."""

    return coder_capture_fraction(strong_model_saving, weak_model_saving) < 1.0


def _anchor_two_container_price() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id="cl3_lambda_2p0_priced_on_both_containers_substitution_20260906",
        measurement_utc="2026-09-06T00:35:02Z",
        inputs={
            "rung": "lambda_2p0 (rate multiplier 2.0, seed 20260716, 60-epoch warm start from the ep-634 EMA init)",
            "terminal_checkpoint_sha256_prefix": "fd686915",
            "field_held_sha256_prefix": "cc10a7b0",
            "weak_container": "brotli(raw IHS1) q0..q11 race -- ddm_cl2's object",
            "strong_container": "brotli(ck2_interleave(apply_hpac(raw, row_counts, shift=5)), q11, lgwin24) -- ddm_rc1, the live object",
            "instrument_control": "the strong container rebuilt the live pointer's hpac section byte-for-byte (12,343 B, sha-identical) from cl2's retained control raw body",
            "producer": f"{_PRODUCER} control; price --rung lambda_2p0",
        },
        predicted_output={
            "P8": "the rc1 (strong) basis shows a LARGER model saving than Brotli, because the rung moves mass into zero-width channels and low depths and the adaptive coder models the code alphabet directly",
            "falsifier": "strong-coder saving <= weak-coder saving",
        },
        empirical_output={
            "delta_model_weak_bytes": DELTA_MODEL_WEAK,
            "delta_model_strong_bytes": DELTA_MODEL_STRONG,
            "delta_stream_bytes_both_bases": DELTA_STREAM,
            "delta_joint_weak_bytes": DELTA_JOINT_WEAK,
            "delta_joint_strong_bytes": DELTA_JOINT_STRONG,
            "capture_fraction": CAPTURE_FRACTION,
            "coder_own_win_bytes": CODER_OWN_WIN_BYTES,
            "reading": "P8 INVERTED: the strong coder returned 457 B where the weak one returned 659 B (69.3%), while the stream tax was identical at +681 B; the rung reads break-even (+22 B) on the weak coder and a clear loss (+224 B) on the object that ships",
        },
        # magnitude by which the measured strong-coder saving fell SHORT of the weak one; P8
        # predicted the strong coder would EXCEED it, so the prediction is inverted, not merely off
        residual=float(abs(abs(DELTA_MODEL_WEAK) - abs(DELTA_MODEL_STRONG))),
        source_artifact=_LEDGER,
        measurement_method="one terminal checkpoint packed through two containers; the stream encoded once through the receiver's own trajectory on the live tree",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria="a THIRD container on the same weights, or the same two containers on a capacity coordinate other than the rate multiplier (width, frame_dim, receptive-field shape), to move this from INSTANCE to family scope",
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_metal_and_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_coder_strength_substitutes_for_capacity_v1() -> CanonicalEquation:
    """Build the coder-strength/capacity substitution law (ddm_cl3, 2026-09-06)."""

    anchor = _anchor_two_container_price()
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "model-coder strength substitutes for the capacity lever -- a stronger model coder "
            "returns less for the same weight shrink, so it makes capacity rungs worth less"
        ),
        one_line_summary=(
            "one weight change, two containers: model -659 B on Brotli vs -457 B on rc1 (69.3%), "
            "stream +681 B in both; the rung is +22 B on the weak coder and +224 B on the shipped one"
        ),
        latex_form=(
            r"\Delta J(\text{coder})=\Delta B_{\text{model}}(\text{coder})+\Delta B_{\text{stream}},\ "
            r"\Delta B_{\text{stream}}\perp\text{coder};\quad "
            r"|\Delta B_{\text{model}}(\text{strong})|\le|\Delta B_{\text{model}}(\text{weak})|"
            r"\ \Rightarrow\ \Delta J(\text{strong})\ge\Delta J(\text{weak});\quad"
            r"\text{MEASURED: }\tfrac{457}{659}=0.693,\ \Delta J:\ +22\to+224"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.coder_strength_substitutes_for_capacity_20260906:transfer_capacity_delta"
        ),
        domain_of_validity={
            "included": [
                "the integer HPAC model section of the semantic_joint_ctxmix object, priced as exact container bytes",
                "transferring a model-side capacity byte delta across a change of MODEL coder",
                "sequencing decisions between model-coder work and model-capacity work",
            ],
            "excluded": [
                "use as a d_seg / d_pose / score claim: the field is held bit-identical on every row, so distortion never moved",
                "the STREAM side: it does not depend on the model coder and needs no discount",
                "a claim that the coder change was wrong -- rc1's own -1,123 B is banked and larger than anything this ladder found",
                "capacity coordinates other than the trainer's rate multiplier: not measured here",
            ],
            "measurement_axis": [_AXIS, "[macOS-MPS research-signal] (training)"],
            "result_type": "TRANSFER RULE plus a sequencing consequence",
            "known_boundary": "one weight change on two containers of one object; a third container or a different capacity coordinate is what moves this past INSTANCE scope",
            "verdict_scope": "instance for the two containers; the mechanism is stated at family scope",
            "sister_laws": [
                "hpac_prior_capacity_slope_v1 -- ddm_cl2's ladder, every rung of which was priced on the WEAK coder and therefore needs this discount",
            ],
        },
        units_in={
            "delta_model_bytes": "bytes",
            "delta_stream_bytes": "bytes",
            "strong_model_saving": "bytes",
            "weak_model_saving": "bytes",
            "capture_fraction": "dimensionless",
        },
        units_out={
            "joint_delta": "bytes",
            "rung_pays": "bool",
            "coder_capture_fraction": "dimensionless",
            "transfer_capacity_delta": "bytes",
            "substitution_holds": "bool",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.anchor_id: anchor.residual},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_LEDGER,),
        canonical_producers=(_PRODUCER, _LEDGER),
        provenance=build_provenance_for_predicted(
            model_id="coder_strength_substitutes_for_capacity.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_metal_and_cpu",
        ),
    )
