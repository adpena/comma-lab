# SPDX-License-Identifier: MIT
"""Canonical equation: the ceiling on BIDIRECTIONAL temporal context for the label field (ddm_bd1).

THE GAP THIS CODIFIES.  Video coding's largest single lever is the B-pyramid: a second
reference plane at +d buys 20-35% over P-only on natural video.  No arm had ever given the
HPAC label-field coder the NEXT pair's field -- ``prepare_frame_context(idx, previous_raw)``
is causal-only -- so the door looked wide open, and the ddm_bd1 charter predicted a 15-30%
cut of the 113,419 B RC64 token stream.  It is worth 2.84-5.68%, and the reason is a single
measurable property of the coded object.

THE LAW (two clauses).

1. TEMPORAL-DECAY CLAUSE.  Let ``P(d)`` be the conditional cost of the current label plane
   given the causal spatial neighbourhood plus the previous plane at distance ``d``.  On this
   field

       P(d) / P(1)  =  1.014 .. 1.129   for d = 2 .. 32                (MEASURED)

   Thirty-two pairs of temporal distance cost only 7-13% more than one.  The SegNet argmax
   label field is piecewise-constant over regions that persist for tens of pairs, so it does
   NOT decorrelate with temporal distance the way pixel intensity does.

2. REDUNDANCY CLAUSE (the consequence).  Because the past plane decays so slowly, the future
   plane is near-redundant with it.  Writing ``PB(d)`` for the cost with BOTH neighbours at
   distance d,

       PB(1) / P(1)  =  0.910 .. 0.927                                 (MEASURED)

   and that 7.3-9.0% IS the supremum of the whole bidirectional family, because no decode
   order can give every pair both neighbours at distance 1 -- some pair must be coded first.
   Every realizable B-pyramid pays a keyframe tax plus longer-distance intermediate levels:

       net(GOP g) = sum_levels  (n_level / N) * cost_level / P(1)
                  = 0.943 .. 0.972   for g in {2, 4, 8, 16, 32}        (MEASURED)

   i.e. a 2.84-5.68% saving, saturating by g = 32 (+0.12 pp per doubling and shrinking).

TRANSFER BOUNDARY (binding).  This law is about a SLOWLY DECORRELATING coded object.  It says
nothing about natural-video B-frames, whose premise (fast intensity decorrelation) this field
does not satisfy, and it does NOT close additional PAST references at several distances -- a
different estimand with a different redundancy structure.  The gain is localised: at d = 1 the
future plane cuts hc1's "no" (wrong-prediction) branch by 9.3-11.6% and the "yes"
(confirmation) branch by only 1.3-1.4%.

INSTRUMENT AND ITS BIAS.  Measured with an adaptive-count model in one model class per arm
(arms differ only by the next-plane taps), read through two brackets: exact sequential KT
(charges the full per-context learning cost -> lower bound on gain) and plug-in (ignores it ->
upper bound).  The counting model codes this field ~1.72x worse than the shipped mixer, and a
stronger base model extracts LESS marginal value from a partially redundant source, so the
measured 2.84-5.68% is an OVER-estimate of what a trained bidirectional mixer would gain.

VERDICT (honest, NO-FAKE): a NEGATIVE / prior-transfer law.  It moves no pointer and is not a
d_seg / d_pose / rate lever; it closes a door cheaply and tells the next charter what to
predict.  Axis ``[exact local bit/byte arithmetic, scorer-free]``; ``score_claim=false``; no
scorer, no Modal, no Metal, no training was run to produce it.

Producer: ``experiments/ddm_bd1_bidirectional_context_screen.py``.
Consumer: ``.omx/research/ddm_bd1_bidirectional_pyramid_context_20260905.md``.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "bidirectional_pyramid_context_gain_v1"

_UTC = "2026-09-05T00:00:00Z"
_AXIS = "[exact local bit/byte arithmetic, scorer-free]"
_LEDGER = ".omx/research/ddm_bd1_bidirectional_pyramid_context_20260905.md"
_PRODUCER = "experiments/ddm_bd1_bidirectional_context_screen.py"

# The coded object (cl2 LADDER_REPORT.json, rung lambda_1p0 -- the frontier archive).
SHIPPED_STREAM_BYTES = 113_419
FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"

# MEASURED 2026-09-05, three context ladders (alpha_s4_p25, beta_s6_p5, gamma_s3_p25),
# 536-pair common window, plug-in reading.  Re-derive at every pointer move.
TEMPORAL_DECAY_P_OVER_P1 = {
    2: (1.0214, 1.0170, 1.0136),
    4: (1.0528, 1.0351, 1.0460),
    8: (1.0889, 1.0531, 1.0942),
    16: (1.0993, 1.0611, 1.1077),
    32: (1.1146, 1.0688, 1.1292),
}
BIDIRECTIONAL_RATIO_AT_D1 = (0.9229, 0.9270, 0.9101)
NO_BRANCH_RATIO_AT_D1 = (0.8996, 0.9068, 0.8842)
YES_BRANCH_RATIO_AT_D1 = (0.9856, 0.9872, 0.9866)

# Exact per-level attribution over the full 600-pair field, per GOP.
PYRAMID_NET_RATIO_BY_GOP = {
    2: (0.97162, 0.97050, 0.96112),
    4: (0.96333, 0.96099, 0.95042),
    8: (0.95937, 0.95531, 0.94697),
    16: (0.95673, 0.95265, 0.94439),
    32: (0.95545, 0.95147, 0.94320),
}
# Unattainable supremum: every pair coded bidirectionally at d = 1.
FAMILY_CEILING_RATIO = (0.922162, 0.925379, 0.909695)

# DERIVED: cost of the conv_future branch at cl2's measured 2.810 bits/param over 38,341 params.
CONV_FUTURE_PARAMS = 2_944
LEVEL_EMBED_PARAMS = 32
SHIPPED_MODEL_BYTES = 13_466
SHIPPED_MODEL_PARAMS = 38_341

# The charter's falsifier.
F1_BAR_FRACTION = 0.08


def temporal_decay_ratio(distance: int, ladder: int = 0) -> float:
    """``P(d) / P(1)`` -- what a reference ``distance`` pairs back costs versus one back."""
    if distance == 1:
        return 1.0
    if distance not in TEMPORAL_DECAY_P_OVER_P1:
        raise KeyError(f"no measured decay ratio at distance {distance}")
    return TEMPORAL_DECAY_P_OVER_P1[distance][ladder]


def pyramid_net_ratio(gop: int, ladder: int = 0) -> float:
    """Net cost of a GOP-``gop`` B-pyramid as a fraction of the shipped causal-d1 coder."""
    if gop not in PYRAMID_NET_RATIO_BY_GOP:
        raise KeyError(f"no measured pyramid ratio at GOP {gop}")
    return PYRAMID_NET_RATIO_BY_GOP[gop][ladder]


def predicted_stream_saving_bytes(gop: int, ladder: int = 0) -> float:
    """GROSS byte saving on the shipped 113,419 B stream, before the model-byte cost."""
    return SHIPPED_STREAM_BYTES * (1.0 - pyramid_net_ratio(gop, ladder))


def conv_future_model_bytes() -> float:
    """DERIVED model cost of the second reference branch at the shipped average bit depth."""
    bits_per_param = SHIPPED_MODEL_BYTES * 8.0 / SHIPPED_MODEL_PARAMS
    return (CONV_FUTURE_PARAMS + LEVEL_EMBED_PARAMS) * bits_per_param / 8.0


def clears_f1(gop: int, ladder: int = 0) -> bool:
    """The charter's screen falsifier: the door stays open only at >= 8% of the stream."""
    return (1.0 - pyramid_net_ratio(gop, ladder)) >= F1_BAR_FRACTION


def build_bidirectional_pyramid_context_gain_v1() -> CanonicalEquation:
    """Build the bidirectional-pyramid context-gain canonical equation (ddm_bd1, 2026-09-05)."""
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_LEDGER,
        reactivation_criteria=(
            "re-measure when the coded object changes (a different field, a residual/token "
            "re-parameterisation, or a coder whose reference planes are not the raw label "
            "field), or if any arm MEASURES conditional mutual information from the future "
            "plane materially above the 9.3-11.6% no-branch figure recorded here"
        ),
    )

    decay_anchor = EmpiricalAnchor(
        anchor_id="label_field_temporal_decay_p_over_p1_20260905",
        measurement_utc=_UTC,
        inputs={
            "object": "600 x 384 x 512 GT SegNet argmax label field, sha256 " + FIELD_SHA256,
            "instrument": "adaptive-count conditional model, 3 context ladders, 536-pair common window",
            "arms": "P-only at distance d in {1, 2, 4, 8, 16, 32}",
            "scorer_runs": 0,
            "training_runs": 0,
        },
        predicted_output={
            "prior_law": (
                "the video-coding prior the charter reasoned from: temporal correlation decays "
                "fast enough with distance that a distance-8 reference is +40..+80% worse than "
                "a distance-1 reference"
            ),
            "p8_over_p1": "1.40 .. 1.80",
        },
        empirical_output={
            "p2_over_p1": TEMPORAL_DECAY_P_OVER_P1[2],
            "p8_over_p1": TEMPORAL_DECAY_P_OVER_P1[8],
            "p32_over_p1": TEMPORAL_DECAY_P_OVER_P1[32],
            "consequence": (
                "the label field is piecewise-constant over regions persisting tens of pairs; "
                "32 pairs of distance cost only 7-13% more than 1, so references at different "
                "distances are near-redundant with each other"
            ),
        },
        # |predicted midpoint 1.60 - measured midpoint 1.079| / 1.60
        residual=0.3256,
        source_artifact=_LEDGER,
        measurement_method="adaptive_count_conditional_entropy_three_ladders_kt_and_plugin_brackets",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    family_anchor = EmpiricalAnchor(
        anchor_id="bidirectional_pyramid_family_ceiling_20260905",
        measurement_utc=_UTC,
        inputs={
            "object": "the cl2 lambda=1.0 frontier RC64 token stream, 113,419 B",
            "family": "every decode-order-realizable B-pyramid, GOP in {2, 4, 8, 16, 32}",
            "supremum": "every pair coded bidirectionally at distance 1 (unattainable)",
            "model_cost_budget_bytes": 1500,
            "scorer_runs": 0,
            "training_runs": 0,
        },
        predicted_output={
            "charter_net_saving_fraction": "0.15 .. 0.30",
            "charter_net_saving_bytes": "17,000 .. 34,000",
            "charter_no_branch_cut_at_d1": "0.30 .. 0.45",
        },
        empirical_output={
            "net_ratio_by_gop": PYRAMID_NET_RATIO_BY_GOP,
            "best_saving_fraction": (0.044550, 0.048527, 0.056796),
            "best_saving_bytes": (5052.82, 5503.84, 6441.74),
            "unattainable_ceiling_saving_fraction": (0.077838, 0.074621, 0.090305),
            "no_branch_ratio_at_d1": NO_BRANCH_RATIO_AT_D1,
            "yes_branch_ratio_at_d1": YES_BRANCH_RATIO_AT_D1,
            "conv_future_model_bytes_derived": 1034.0,
            "f1_bar_fraction": F1_BAR_FRACTION,
            "f1_fired": True,
            "verdict": (
                "CLOSED at family scope: the 8% falsifier bites the unattainable supremum, not "
                "merely the charter's GOP=8 layout; the family saturates by GOP 32"
            ),
        },
        # |predicted midpoint 0.225 - measured best 0.0568| / 0.225
        residual=0.7476,
        source_artifact=_LEDGER,
        measurement_method="exact_per_level_plugin_attribution_over_the_full_600_pair_field",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Bidirectional (B-pyramid) temporal-context gain on a slowly decorrelating label field",
        one_line_summary=(
            "P(d)/P(1)=1.014..1.129 for d<=32, so the future plane is near-redundant with the past: "
            "every B-pyramid saves 2.84-5.68% of the stream, supremum 7.5-9.0%"
        ),
        latex_form=(
            r"\mathrm{net}(g)=\sum_{\ell}\frac{n_\ell}{N}\frac{C_\ell}{P(1)},\quad "
            r"\frac{P(d)}{P(1)}\in[1.014,1.129]\ (d\le 32),\quad "
            r"\sup_{\text{family}}\left(1-\mathrm{net}\right)=\frac{PB(1)}{P(1)}\ \text{complement}"
            r"\in[0.075,0.090]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bidirectional_pyramid_context_gain_20260905:pyramid_net_ratio"
        ),
        domain_of_validity={
            "object": (
                "the GT SegNet argmax label field of the comma video-compression contest "
                "(600 x 384 x 512, 5 classes), sha256 " + FIELD_SHA256
            ),
            "coder": (
                "a context-mixing model whose temporal reference is the raw label plane "
                "(the shipped HPAC integer mixer)"
            ),
            "applies_to": "a SECOND REFERENCE PLANE at +d, any GOP (the bidirectional family)",
            "does_not_apply_to": (
                "natural-video B-frames (fast intensity decorrelation -- the premise this field "
                "does not satisfy); additional PAST references at several distances (a different "
                "estimand); the corrector/model axis; motion-compensated references (mc1's "
                "separate closure)"
            ),
            "transfer_rule": (
                "RATIOS transfer; the counting model's absolute bytes never do, and the "
                "instrument's 1.72x weaker base biases the measured gain UPWARD"
            ),
        },
        units_in={
            "distance": "pairs",
            "gop": "pairs",
            "ladder": "index 0..2 selecting the alpha/beta/gamma context ladder",
        },
        units_out={
            "pyramid_net_ratio": "dimensionless cost ratio against the shipped causal-d1 coder",
            "predicted_stream_saving_bytes": "bytes of the 113,419 B shipped RC64 token stream",
            "conv_future_model_bytes": "bytes added to the 13,466 B packed model",
        },
        empirical_anchors=(decay_anchor, family_anchor),
        predicted_vs_empirical_residual={
            "temporal_decay_p8_over_p1": 0.3256,
            "pyramid_net_saving_fraction": 0.7476,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(_LEDGER,),
        canonical_producers=(_PRODUCER,),
        provenance=provenance,
    )
