# SPDX-License-Identifier: MIT
"""Canonical equation: the shipped semantic renderer's seg->pose COUPLING (ddm_rf1 + ddm_ft1).

THE GAP THIS CODIFIES.  Two arms three weeks apart, on two different kinds of edit,
measured the SAME structural fact about the shipped SM3R semantic renderer: any move
along the seg-only direction pays for its d_seg change with a d_pose change roughly
TWO ORDERS OF MAGNITUDE larger.  Both arms wrote the ratio into their memo and
neither wrote it into the equations leg, so the number that CLOSES the seg-only
renderer formulation lived only in prose.

THE LAW.  For the shipped semantic renderer at its own size, an edit whose gradient
direction prices d_seg ALONE realizes

    |Delta d_pose| = k * |Delta d_seg|,     k in [166.81, 217.30]  (MEASURED, n=2 arms)

k is a property of the DIRECTION, not of the edit's magnitude or of its sign: rf1's
un-retrained structural swap and ft1's trained seg-only fine-tune are different
mechanisms that land 1.303x apart on the same object.

THE CLOSING ARITHMETIC (DERIVED, at Delta B = 0).  The score is
S = 100 d_seg + sqrt(10 d_pose) + 25 B / 37_545_489.  With no byte credit to spend, a
seg cut of |Delta d_seg| funds a pose ceiling

    d_pose_max = (sqrt(10 d_pose_base) + 100 |Delta d_seg|)^2 / 10

and the coupling predicts the pose the cut actually costs.  On the afr1 shipped object
(`d_seg 0.00020139`, `d_pose 6.37e-06`, 180,002 B, S 0.14797617125559104,
[contest-CUDA T4 n600]) a 25% seg cut funds `d_pose_max = 1.694e-05` (2.66x base) while
costing, at the SMALLEST measured coupling, `Delta d_pose = 8.40e-03 = 1,318x` base.
Even the best n600 carrier recovery ever measured (8.0x, jg5) leaves it ~62x over the
ceiling at k = 166.81 and ~81x over at k = 217.30 -- and a renderer weight moves all 600
pairs at once, so there is no per-pair admission lever to buy the gap back.

VERDICT.  `verdict_scope: FORMULATION` -- seg-only gradient direction, shipped SM3R
renderer, this size.  The seg-only renderer fine-tune is CLOSED by arithmetic, at BOTH
ends of the measured band.

WHAT THIS LAW DOES NOT CLOSE (binding, in `domain_of_validity["excluded"]`).  k is the
coupling OF THE SEG-ONLY DIRECTION.  A joint loss that prices pose in-loop does not
accept that direction; it searches for a lower-coupling one.  ft1's own prior for that
branch: w96b ran pose in-loop from step zero and still landed d_pose 1.30e-03 = 204x
base -- an order of magnitude better than 2,366x, and still not payable.  204x vs 2,366x
is the whole remaining question, and this equation does not answer it.

DIRECTION SYMMETRY IS AN ASSUMPTION, NOT A MEASUREMENT.  Both anchors moved d_seg UP.
The closing arithmetic applies |Delta d_pose| = k |Delta d_seg| to a seg DECREASE, i.e.
it assumes the seg-only direction costs pose at rate k in both senses (local linearity
of the realized map around the shipped weights).  That assumption is stated in the
domain and is the single thing a future arm could falsify cheaply.

Producers: `.omx/research/ddm_rf1_renderer_film_rung_20260824.md` -- which publishes the
four COMPONENTS but never prints the ratio; 166.81 is DERIVED from that table, and first
appears in print in ft1, which re-derived it the same way -- and
`.omx/research/ddm_ft1_shipped_renderer_aligned_finetune_20260903.md` with retained
receipt `retained/verdict_ft1_step600.json` under
`/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/` (217.30366,
the receipt's own `delta.coupling_dpose_over_dseg`).
Consumers: the fold-back program
`.omx/research/ddm_fb1_foldback_program_20260903.md` and every future renderer charter --
the ceiling above is the gate such a charter must clear BEFORE it launches.

Memory: `renderer_seg_pose_coupling_170_220_two_arms_20260903`.
"""

from __future__ import annotations

import math

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

EQUATION_ID = "renderer_seg_pose_coupling_shipped_object_v1"

_UTC = "2026-09-03T00:00:00Z"
_AXIS = "[macOS-CPU advisory]"
_RF1_LEDGER = ".omx/research/ddm_rf1_renderer_film_rung_20260824.md"
_FT1_LEDGER = ".omx/research/ddm_ft1_shipped_renderer_aligned_finetune_20260903.md"
_FT1_RECEIPT = (
    "/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/"
    "retained/verdict_ft1_step600.json"
)

# Contest scoring constants (upstream/evaluate.py).
SEG_WEIGHT = 100.0
POSE_WEIGHT = 10.0

# --- The two MEASURED anchors -------------------------------------------------------
# rf1: DERIVED from the memo's published component table (5-6 significant figures there).
RF1_BASE_D_SEG = 0.0003474
RF1_CANDIDATE_D_SEG = 0.00043022
RF1_BASE_D_POSE = 0.00014701
RF1_CANDIDATE_D_POSE = 0.01396208
RF1_COUPLING = 166.80837961844966

# ft1: MEASURED, read straight off the retained receipt's delta.coupling_dpose_over_dseg.
FT1_BASE_D_SEG = 0.00020029703776041667
FT1_CANDIDATE_D_SEG = 0.000269622802734375
FT1_BASE_D_POSE = 9.002495893360466e-06
FT1_CANDIDATE_D_POSE = 0.015073745112341155
FT1_COUPLING = 217.30366224024704

COUPLING_MIN = RF1_COUPLING
COUPLING_MAX = FT1_COUPLING
# Geometric centre: the band is multiplicative (a ratio of ratios), so its centre is too.
COUPLING_CENTRE = math.sqrt(COUPLING_MIN * COUPLING_MAX)  # 190.38926383452008
COUPLING_DISPERSION = COUPLING_MAX / COUPLING_MIN  # 1.3027143045049543

# --- The shipped object the ceiling is priced against -------------------------------
# afr1 [contest-CUDA T4 n600]:
# experiments/results/modal_auth_eval_mirror/
#   contest_auth_eval_modal-ddm_afr1_tile48_groupbin8_cuda_n600_20260831.json
AFR1_D_SEG = 0.00020139
AFR1_D_POSE = 6.37e-06
AFR1_ARCHIVE_BYTES = 180_002
AFR1_S = 0.14797617125559104
# Measured n600 carrier-recovery ceiling for a terminal pose re-solve (fcd2 / jg5).
CARRIER_RECOVERY_MIN = 5.87
CARRIER_RECOVERY_MAX = 8.0


def predicted_delta_d_pose(delta_d_seg: float, *, coupling: float = COUPLING_CENTRE) -> float:
    """|Delta d_pose| the seg-only direction costs for a realized |Delta d_seg|.

    Sign-free by construction: the law is a magnitude ratio measured on the
    seg-only DIRECTION, and the closing arithmetic applies it to a seg cut under the
    stated local-linearity assumption.
    """
    return abs(coupling) * abs(delta_d_seg)


def payable_pose_ceiling(
    delta_d_seg: float,
    *,
    base_d_pose: float = AFR1_D_POSE,
    delta_bytes: int = 0,
    rate_denominator_bytes: int = 37_545_489,
) -> float:
    """The largest post-solve d_pose a seg improvement (plus any byte credit) can fund.

    Solves ``100*Delta d_seg + sqrt(10 d_pose_new) - sqrt(10 d_pose_base)
    + 25*Delta B / 37,545,489 < 0`` for ``d_pose_new``.  ``delta_d_seg`` is taken as a
    magnitude and treated as an improvement (a seg cut); ``delta_bytes`` negative is a
    byte credit.  Returns 0.0 when the move cannot fund any pose at all.
    """
    credit = (
        SEG_WEIGHT * abs(delta_d_seg)
        - 25.0 * float(delta_bytes) / float(rate_denominator_bytes)
    )
    budget = math.sqrt(POSE_WEIGHT * base_d_pose) + credit
    if budget <= 0.0:
        return 0.0
    return budget * budget / POSE_WEIGHT


def seg_only_move_is_payable(
    delta_d_seg: float,
    *,
    coupling: float = COUPLING_MIN,
    base_d_pose: float = AFR1_D_POSE,
    carrier_recovery: float = CARRIER_RECOVERY_MAX,
    delta_bytes: int = 0,
) -> bool:
    """THE GATE: does a seg-only renderer move clear its own pose ceiling?

    Defaults are the MOST FAVOURABLE reading of every measured input -- the smallest
    measured coupling and the best carrier recovery ever measured at n600 -- so a False
    here is a closure that holds a fortiori across the whole measured band.
    """
    cost = base_d_pose + predicted_delta_d_pose(delta_d_seg, coupling=coupling)
    post_solve = cost / max(float(carrier_recovery), 1.0)
    return post_solve <= payable_pose_ceiling(
        delta_d_seg, base_d_pose=base_d_pose, delta_bytes=delta_bytes
    )


def overshoot_multiple(
    delta_d_seg: float,
    *,
    coupling: float = COUPLING_MIN,
    base_d_pose: float = AFR1_D_POSE,
    carrier_recovery: float = CARRIER_RECOVERY_MAX,
) -> float:
    """How many times over the payable ceiling the seg-only move lands (1.0 = exactly at it)."""
    ceiling = payable_pose_ceiling(delta_d_seg, base_d_pose=base_d_pose)
    if ceiling <= 0.0:
        return math.inf
    cost = base_d_pose + predicted_delta_d_pose(delta_d_seg, coupling=coupling)
    return (cost / max(float(carrier_recovery), 1.0)) / ceiling


def coupling_from_components(
    base_d_seg: float, candidate_d_seg: float, base_d_pose: float, candidate_d_pose: float
) -> float:
    """Re-derive k from a receipt's four component numbers (the audit path)."""
    return abs(candidate_d_pose - base_d_pose) / abs(candidate_d_seg - base_d_seg)


def build_renderer_seg_pose_coupling_shipped_object_v1() -> CanonicalEquation:
    """Build the renderer seg->pose coupling canonical equation (rf1 + ft1, 2026-09-03)."""
    rf1_anchor = EmpiricalAnchor(
        anchor_id="rf1_film_amortized_flat_w96_structural_coupling_20260824",
        measurement_utc="2026-08-24T00:00:00Z",
        inputs={
            "object": "film_amortized_flat_w96 vs the matched dx2 CPU base (mst1 advisory_r1)",
            "edit_kind": "un-retrained STRUCTURAL change (per-block FiLM -> one trunk flat FiLM)",
            "base_d_seg": RF1_BASE_D_SEG,
            "candidate_d_seg": RF1_CANDIDATE_D_SEG,
            "base_d_pose": RF1_BASE_D_POSE,
            "candidate_d_pose": RF1_CANDIDATE_D_POSE,
            "base_archive_bytes": 180_368,
            "candidate_archive_bytes": 179_290,
            "candidate_archive_sha256": (
                "34855e3c43e564d48adc492d919afa81662ebff847386d36bbf1a07304b26d21"
            ),
            "n_pairs": 600,
            "precision_note": (
                "the memo publishes 5-6 significant figures per component; k is DERIVED "
                "from that table, not read off a retained full-precision receipt"
            ),
        },
        predicted_output={
            "coupling_centre": COUPLING_CENTRE,
            "prior_law": (
                "rf1 had no prior coupling law -- it is the FIRST anchor; the centre shown "
                "here is the two-anchor band centre this equation now carries"
            ),
        },
        empirical_output={
            "coupling_dpose_over_dseg": RF1_COUPLING,
            "d_seg_ratio": 1.2384,
            "d_pose_ratio": 94.9737,
            "pose_share_of_damage": 0.9759,
            "net_delta_s": 0.342880989,
            "verdict": "REFUSED at 2.7749x the matched base; 97.59% of the damage is pose",
        },
        residual=abs(RF1_COUPLING - COUPLING_CENTRE) / COUPLING_CENTRE,
        source_artifact=_RF1_LEDGER,
        measurement_method=(
            "matched-instrument byte-closed advisory replay of an exported structural variant "
            "through the shipped receiver and upstream SegNet/PoseNet; k re-derived from the "
            "memo's four published components"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_RF1_LEDGER,
            reactivation_criteria=(
                "re-derive k from a full-precision retained receipt if one is ever produced for "
                "this variant; the memo's 5-6 figure components bound k to about +/-0.01%"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    ft1_anchor = EmpiricalAnchor(
        anchor_id="ft1_shipped_renderer_seg_only_finetune_coupling_step600_20260903",
        measurement_utc="2026-09-03T00:00:00Z",
        inputs={
            "object": (
                "shipped SM3R semantic renderer, seg-only aligned fine-tune, step-600 "
                "checkpoint, exported and parsed back through the shipped receiver"
            ),
            "edit_kind": "TRAINED seg-only expected-flip margin loss, no pose term in the loop",
            "base_d_seg": FT1_BASE_D_SEG,
            "candidate_d_seg": FT1_CANDIDATE_D_SEG,
            "base_d_pose": FT1_BASE_D_POSE,
            "candidate_d_pose": FT1_CANDIDATE_D_POSE,
            "pair_selection": "seeded random draw of 200/600 (seed 20260903); NOT a prefix",
            "n_pairs": 200,
            "gt_lineage": "DALI",
            "section_bytes": 36_130,
            "section_sha256": (
                "819c28e8971020fb34990cf32e011eb88f87eb41369eb16393ae5382c3cad407"
            ),
            "parse_back_max_abs_delta": 0.0,
            "size_preserved": True,
        },
        predicted_output={
            "coupling_transfers_from_rf1": True,
            "prior_law": (
                "rf1's 166.8 was measured on an un-retrained structural change; the "
                "pre-registered question was whether it transfers to a small trained fine-tune"
            ),
            "falsifier": "a coupling below ~10 would reopen the seg-only renderer formulation",
        },
        empirical_output={
            "coupling_dpose_over_dseg": FT1_COUPLING,
            "delta_d_seg": 6.932576497395836e-05,
            "delta_d_pose": 0.015064742616447795,
            "d_pose_ratio_vs_base": 1674.3962219919883,
            "delta_s": 0.38569364080686197,
            "transfer_ratio_vs_rf1": COUPLING_DISPERSION,
            "trained_vs_realized_max_abs_delta": 0.002322908490896225,
            "verdict": (
                "FORMULATION CLOSED -- seg-only aligned renderer fine-tune at this size is "
                "unpayable on pose at ANY useful seg gain"
            ),
        },
        residual=abs(FT1_COUPLING - COUPLING_CENTRE) / COUPLING_CENTRE,
        source_artifact=_FT1_RECEIPT,
        measurement_method=(
            "export -> shipped-receiver parse-back -> upstream SegNet/PoseNet on a seeded "
            "random 200/600 draw; k read directly from delta.coupling_dpose_over_dseg"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_FT1_RECEIPT,
            reactivation_criteria=(
                "re-measure k on a JOINT (pose-priced) objective from these same weights -- "
                "that direction is explicitly OUTSIDE this equation's domain and is the one "
                "branch the closure does not reach"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Shipped-renderer seg->pose coupling -- the seg-only direction pays "
            "|Delta d_pose| = k |Delta d_seg| with k in [166.81, 217.30]"
        ),
        one_line_summary=(
            "k = 166.81 (rf1 structural) and 217.30 (ft1 trained), 1.303x apart; at dB=0 a 25% "
            "seg cut funds d_pose 1.694e-05 but costs 8.4e-03 -- seg-only renderer is CLOSED"
        ),
        latex_form=(
            r"|\Delta d_{\mathrm{pose}}| = k\,|\Delta d_{\mathrm{seg}}|,\ "
            r"k\in[166.81,\,217.30];\quad "
            r"d^{\max}_{\mathrm{pose}}=\frac{\left(\sqrt{10 d^{\mathrm{base}}_{\mathrm{pose}}}"
            r"+100|\Delta d_{\mathrm{seg}}|-\frac{25\Delta B}{37545489}\right)^2}{10};\quad "
            r"\mathrm{PAYABLE}\iff \frac{d^{\mathrm{base}}_{\mathrm{pose}}+k|\Delta d_{\mathrm{seg}}|}"
            r"{\rho}\le d^{\max}_{\mathrm{pose}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.renderer_seg_pose_coupling_20260903:seg_only_move_is_payable"
        ),
        domain_of_validity={
            "included": [
                "the shipped SM3R semantic renderer at its own size (36,130 B section)",
                "edits whose gradient direction prices d_seg ALONE (seg-only loss, or an "
                "un-retrained structural swap that was never pose-conditioned)",
                "realized-through-receiver measurement: export -> parse-back -> frozen scorers",
            ],
            "excluded": [
                "JOINT (pose-priced) formulations -- a loss with pose in-loop rejects the "
                "seg-only direction and searches for a lower-coupling one; w96b's 204x (pose "
                "in loop) vs 2,366x (pose absent) is an order of magnitude and is UNTESTED "
                "from these weights",
                "renderers of a different size or family",
                "any use of k as a SCORE, an axis, or a promotion claim -- this is a lever law",
            ],
            "assumption_stated_not_measured": (
                "DIRECTION SYMMETRY: both anchors moved d_seg UP; the closing arithmetic "
                "applies the same k to a seg DECREASE, i.e. it assumes local linearity of the "
                "realized map around the shipped weights. Cheapest falsification available."
            ),
            "measurement_axis": [_AXIS],
            "n_pairs": {"rf1": 600, "ft1": 200},
            "no_per_pair_admission_lever": (
                "a renderer weight change moves all 600 pairs at once, so the per-pair "
                "accept/reject lever that rescues token-level edits does not exist here"
            ),
            "result_type": (
                "LEVER-CLOSURE law; NON-PROMOTABLE; moves no pointer by itself. It gates "
                "renderer charters BEFORE they launch."
            ),
            "priced_against": {
                "object": "afr1 shipped archive [contest-CUDA T4 n600]",
                "d_seg": AFR1_D_SEG,
                "d_pose": AFR1_D_POSE,
                "archive_bytes": AFR1_ARCHIVE_BYTES,
                "S": AFR1_S,
            },
            "carrier_recovery_measured": [CARRIER_RECOVERY_MIN, CARRIER_RECOVERY_MAX],
        },
        units_in={
            "delta_d_seg": "argmax_disagreement_fraction",
            "base_d_pose": "mean_squared_error",
            "coupling": "dimensionless_ratio_dpose_per_dseg",
            "carrier_recovery": "dimensionless_multiple",
            "delta_bytes": "bytes",
        },
        units_out={
            "predicted_delta_d_pose": "mean_squared_error",
            "payable_pose_ceiling": "mean_squared_error",
            "seg_only_move_is_payable": "bool",
            "overshoot_multiple": "dimensionless_multiple",
        },
        empirical_anchors=(rf1_anchor, ft1_anchor),
        predicted_vs_empirical_residual={
            "rf1_film_amortized_flat_w96_structural_coupling_20260824": (
                abs(RF1_COUPLING - COUPLING_CENTRE) / COUPLING_CENTRE
            ),
            "ft1_shipped_renderer_seg_only_finetune_coupling_step600_20260903": (
                abs(FT1_COUPLING - COUPLING_CENTRE) / COUPLING_CENTRE
            ),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _FT1_LEDGER,
            ".omx/research/ddm_fb1_foldback_program_20260903.md",
        ),
        canonical_producers=(_RF1_LEDGER, _FT1_LEDGER),
        provenance=build_provenance_for_predicted(
            model_id="renderer_seg_pose_coupling_shipped_object.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_128gib_cpu",
        ),
    )
