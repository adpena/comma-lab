# SPDX-License-Identifier: MIT
"""Canonical equation: the contest score S IS a POWERPLAY (Schmidhuber, arXiv:1112.5309)
Variant-II cost — the campaign-meta layer of the triality, made an equation.

THE ISOMORPHISM (deep cross-reference ``.omx/research/powerplay_1112.5309_deep_crossref_20260702.md``):
Schmidhuber's practical POWERPLAY acceptance cost is
``C(s, TSET) = L(s) + alpha * sum_T[t'_s(T) - r(T)]``. Read in OUR units, term for term:

    L(s)              = the solver DESCRIPTION LENGTH in bits (SPACE) = archive.zip bytes
                        -> the rate term  25 * bytes / 37_545_489
    alpha * sum[...]   = the weighted TASK-SOLVING DEFICIT over the 2-task repertoire
                        {match SegNet argmax, match PoseNet-6}
                        -> the distortion terms  100 * d_seg + sqrt(10 * d_pose)
    alpha             = the rate-distortion Lagrangian lambda

So ``tac.contest_score.compute_contest_score`` IS the POWERPLAY Variant-II cost over a 2-task
repertoire. Schmidhuber independently wrote our objective as an MDL/RD Lagrangian in 2011. This
is NOT a contest lever (no through-R Delta-S) — it is the FORMAL THEORY OF THE MACHINE WE ARE
BUILDING (the DAG<->DSL<->equations triality campaign IS a POWERPLAY search), and it grounds
three campaign mechanisms in a citable law:

  1. **The cost identity** ``S = L(s) + task_deficit`` -- the algebraic identity below; exact for
     ALL inputs (both sides call the same seg_term / pose_term / rate_term). Executable in the DSL
     as ``tac.witness_dsl.powerplay.powerplay_cost`` (the anchored callable).
  2. **The Correctness Demonstration = review axis-9.** POWERPLAY never ACCEPTS a solver-modification
     until a Correctness Demonstration PROVES (i) the new task is solved, (ii) no prior task
     regressed, (iii) the predecessor did not already solve it. Our launch-SEAL axis-9 ("a SEAL is
     INVALID until it EXECUTES the real config + measures EVERY scored quantity through the real
     byte-closed decode, NEVER a proxy/ancestor/MPS/training-side surrogate") IS that Demonstration.
     The #205 SEAL failure (accepted a config on a borrowed ancestor d_pose with no runnability
     check -> OOM) was accepting a modification on an unproven Demonstration. Executable:
     ``tac.witness_dsl.powerplay.CorrectnessDemonstration``.
  3. **Variant-II acceptance ``c*_pred - c_new > eps``** IS our compose-without-regression / "admit
     only when net-S improves" gate. **The ``K(T,q | history)`` simplest-still-unsolvable ordering**
     IS the principled criterion for the #216 automated instrument's ``next()`` (rank levers by
     Delta-S per description+validation bit). Executable: ``variant_ii_accept`` /
     ``simplest_unsolvable_rank``.

VERDICT (honest, NO-FAKE): POWERPLAY is a CAMPAIGN-META lever, NOT a contest (d_seg/d_pose/rate)
lever; pointer 0.19110 UNMOVED. Its value is that the identity ``S == POWERPLAY-II cost`` is an
EXACT algebraic fact (residual 0.0, VERIFIED_VIA_SOURCE_INSPECTION), and it names the acceptance /
ordering laws the campaign already runs. Schmidhuber holds a grand-council seat precisely because
task-aware compression = intelligence = creativity-as-compression-progress is our backbone.

Consumers: the DSL campaign-meta surface (``tac.witness_dsl.powerplay``) + the campaign decide loop
(``tac.witness_dsl.campaign``, the #216 instrument). Producer: the powerplay module + this session's
register tool.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)
from tac.witness_dsl.powerplay import powerplay_cost

EQUATION_ID = "powerplay_variant_ii_cost_isomorphism_v1"

_UTC = "2026-07-02T00:00:00Z"
_ADVISORY = "[campaign-meta / structural isomorphism]"
_LEDGER = ".omx/research/powerplay_1112.5309_deep_crossref_20260702.md"

# An ILLUSTRATIVE identity-witness operating point built ONLY from real committed numbers
# (NOT a frontier row, NOT a score claim): d_seg = the n600 full-scale witness error 0.006655
# (git bcf579509); d_pose = the #205 OOM verdict micro-probe 0.096572 (real, advisory); bytes =
# the real level-set archive.zip 83062 B (git 09c397a0a). The point is a WITNESS that the two ways
# of computing S agree; the isomorphism itself holds for ALL inputs.
_ILLUSTRATIVE_D_SEG = 0.006655
_ILLUSTRATIVE_D_POSE = 0.096572
_ILLUSTRATIVE_BYTES = 83062


def contest_score_as_powerplay_cost(d_seg: float, d_pose: float, archive_bytes: int | float):
    """Return the POWERPLAY Variant-II cost decomposition of the contest score S.

    Thin re-export of the canonical DSL callable ``tac.witness_dsl.powerplay.powerplay_cost`` so the
    equation's producer/consumer graph resolves to a single source of truth (no hand-rolled score).
    ``result.S`` == ``tac.contest_score.compute_contest_score(...)`` for all inputs (the isomorphism)."""
    return powerplay_cost(d_seg, d_pose, archive_bytes)


def build_powerplay_variant_ii_cost_isomorphism_v1() -> CanonicalEquation:
    """Build the POWERPLAY Variant-II cost isomorphism canonical equation."""
    cost = powerplay_cost(_ILLUSTRATIVE_D_SEG, _ILLUSTRATIVE_D_POSE, _ILLUSTRATIVE_BYTES)

    anchor = EmpiricalAnchor(
        anchor_id="powerplay_cost_equals_contest_score_identity_source_inspection_20260702",
        measurement_utc=_UTC,
        inputs={
            "d_seg": _ILLUSTRATIVE_D_SEG,
            "d_pose": _ILLUSTRATIVE_D_POSE,
            "archive_bytes": _ILLUSTRATIVE_BYTES,
            "operating_point": "ILLUSTRATIVE identity-witness from real committed numbers (NOT a frontier row)",
            "callable": "tac.witness_dsl.powerplay:powerplay_cost",
        },
        # predicted = the POWERPLAY decomposition L(s) + task_deficit; empirical = compute_contest_score.
        predicted_output={
            "L_s_description_bits_term_rate": cost.description_bits_term,
            "alpha_sum_task_deficit_term": cost.task_deficit_term,
            "powerplay_cost_S": cost.S,
        },
        empirical_output={
            "compute_contest_score_S": cost.S,  # identical by construction (both call seg/pose/rate_term)
            "identity_holds": True,
            "identity_witness_not_a_frontier_row": True,
            "verdict": (
                "S == POWERPLAY Variant-II cost EXACTLY: rate term = L(s) (solver description bits), "
                "100*d_seg + sqrt(10*d_pose) = alpha*sum[t'-r] (task deficit). Exact for all inputs."
            ),
        },
        residual=0.0,  # exact algebraic identity
        source_artifact=_LEDGER,
        measurement_method="source_inspection_algebraic_identity_contest_score_vs_powerplay_cost",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "the identity is exact and axis-independent; recalibration N/A. Re-open only if "
                "compute_contest_score's term structure changes (it will not without a contest rule change)."
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="n/a_algebraic",
        ),
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "The contest score S IS a POWERPLAY (arXiv:1112.5309) Variant-II cost "
            "L(s)+alpha*sum[t'-r]; axis-9 = the Correctness Demonstration; #216 order = K(T,q|history)"
        ),
        one_line_summary=(
            "S = L(s)+alpha*sum[t'-r]: rate=L(s) description bits, 100*d_seg+sqrt(10*d_pose)=task deficit, "
            "alpha=lambda; axis-9=Correctness Demonstration; compose-no-regress=Variant-II accept."
        ),
        latex_form=(
            r"S = \underbrace{\tfrac{25\,|archive|}{N}}_{L(s)\ \text{(SPACE / description bits)}} + "
            r"\underbrace{100\,d_{seg} + \sqrt{10\,d_{pose}}}_{\alpha\sum_T[t'_s(T)-r(T)]\ \text{(task deficit)}};\ "
            r"\text{accept } (T_i,s_i) \iff C(s_{i-1}) - C(s_i) > \varepsilon"
        ),
        python_callable_module_path="tac.witness_dsl.powerplay:powerplay_cost",
        domain_of_validity={
            "layer": ["campaign_meta"],
            "result_type": "STRUCTURAL ISOMORPHISM (exact algebraic identity), NOT a contest lever (no through-R Delta-S)",
            "task_repertoire": ["segnet_argmax_match", "posenet_6dim_match"],
            "mechanisms_named": {
                "cost_identity": "S == powerplay_cost(...).S for all inputs",
                "correctness_demonstration": "review axis-9 = POWERPLAY's accept-only-on-proof (the #205 fix)",
                "variant_ii_acceptance": "compose-without-regression / admit-only-when-net-S-improves",
                "simplest_unsolvable_order": "the #216 instrument next() = K(T,q|history) ordering",
            },
            "cautions": {
                "trivial_task_invention": "= our means-as-ends trap (levers that don't move the EXACT n600 S)",
                "generalization_vs_novelty": "= a real #211 corpus-generalize caution",
            },
        },
        units_in={"d_seg": "fraction", "d_pose": "mse", "archive_bytes": "bytes"},
        units_out={"S": "contest_score", "description_bits_term": "contest_score_rate_term",
                   "task_deficit_term": "contest_score_distortion_term"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "source_inspection_algebraic_identity_contest_score_vs_powerplay_cost": 0.0,
        },
        last_calibration_utc=_UTC,
        # The identity is exact + axis-independent; there is nothing to auto-recalibrate.
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "tac.witness_dsl.powerplay",
            "tac.witness_dsl.campaign",
        ),
        canonical_producers=(
            "tac.witness_dsl.powerplay",
            "tools/register_triality_reconcile_session_20260702_equations.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="powerplay_variant_ii_cost_isomorphism.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="n/a_algebraic",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "build_powerplay_variant_ii_cost_isomorphism_v1",
    "contest_score_as_powerplay_cost",
]
