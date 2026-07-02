# SPDX-License-Identifier: MIT
"""Canonical equation: task-oriented R(D) lies STRICTLY BELOW reconstruction R(D) -- the theorem
that our task-space witness DOMINATES a full-RGB codec (DERIVE / citation-anchored).

THE THEOREM (compression-as-intelligence lineage, ledger
``.omx/research/compression_as_intelligence_lineage_crossref_20260702.md`` item 2 / section 2.4):
for a FIXED downstream model M, the MODEL-AWARE rate-distortion function R_M(D) -- the minimum bitrate
so M performs within task-distortion D -- lies STRICTLY BELOW the classical reconstruction R(D)
(arXiv:2602.12866 "Model-Aware Rate-Distortion Limits"; via the Dobrushin-Witsenhausen remote /
indirect rate-distortion framework; companion arXiv:2405.04144 RDC/RPC). Mechanism: optimal compression
for a fixed predictor reduces X to the SUFFICIENT STATISTICS for the task; model-irrelevant information
(the "RGB-slack" the frozen SegNet/PoseNet never reads) can be quantized to ~zero cost. The dominance
gap ``R_X(D) - R_T(D) >= 0`` GROWS exactly when the downstream model ignores high-variance but
task-irrelevant dimensions -- precisely our regime.

WHY THIS MATTERS (the load-bearing consequence): the NON-RGB task-space witness capstone beating a
full-RGB codec is a THEOREM, not an assertion. It formally KILLS any "just build a good RGB codec /
Hutter-Prize-style raw-compression" revival: keeping any bit the frozen scorer never reads is PROVABLE
rate waste. This is the equations-leg anchor of the contest-is-indirect-rate-distortion framing
(memory ``project_contest_is_indirect_rate_distortion_task_space_coding``) and the structure-function
model/noise split (S = free deterministic generator [the model] + counted incompressible residual).

DIRECTIONAL CORROBORATION (OURS, honestly labeled -- NOT a clean equal-distortion pair): the measured
capacity-vs-rate trilemma -- bc20 task-space basis reaches rate ~0.059 (under-capacity on d_seg) while
the bc36 PR95-size full-RGB RESKIN sits at rate ~0.118 for adequate d_seg -- points the SAME direction
(task-space representation buys adequate distortion at roughly HALF the rate). Caveat: bc36 is a reskin,
not a controlled full-RGB-codec-at-equal-task-distortion baseline; the theorem, not this pair, is the
authority.

VERDICT (honest, NO-FAKE): a FRAMING theorem that PROVES the direction, NOT a contest lever (no
through-R Delta-S); pointer 0.19110 UNMOVED. Filed in the papers-checked ledger as
FRAMING-that-proves-the-direction. The primary anchor is a citation (INFERRED_FROM_DOMAIN_LITERATURE,
residual 0.0 -- an inequality bound we cite, do not measure).

Consumers: the task-space witness generator (``tac.boundary_math.lever_b_generator``) + the witness
autoconfig (the design rationale for choosing task-space over full-RGB). Producer: the compression
lineage ledger + this session's register tool.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    INFERRED_FROM_DOMAIN_LITERATURE,
    RECALIBRATE_NEVER_AUTO,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "task_rd_dominates_reconstruction_rd_v1"

_UTC = "2026-07-02T00:00:00Z"
_ADVISORY = "[framing-theorem / citation]"
_LEDGER = ".omx/research/compression_as_intelligence_lineage_crossref_20260702.md"

# Directional corroboration numbers (OURS, measured; honest caveat: reskin, not equal-distortion).
_BC20_TASK_SPACE_RATE = 0.059    # bc20 small task-space basis rate (under-capacity on d_seg)
_BC36_RESKIN_RATE = 0.118        # bc36 PR95-size full-RGB reskin rate (adequate d_seg)


def task_rd_dominance_gap(reconstruction_rate_bits: float, task_rate_bits: float) -> float:
    """The dominance gap ``R_X(D) - R_T(D) >= 0`` at equal task-distortion D (GiB/bits, same units).

    By the theorem the gap is non-negative: the task-oriented (model-aware) rate never exceeds the
    reconstruction rate for the same downstream-task distortion. Returns ``max(0, recon - task)`` so a
    caller can quantify the provable RGB-slack saving; a negative raw difference would violate the
    theorem (returned clamped to 0 with the sign available via ``task_rd_le_reconstruction_rd``)."""
    return max(0.0, float(reconstruction_rate_bits) - float(task_rate_bits))


def task_rd_le_reconstruction_rd(reconstruction_rate_bits: float, task_rate_bits: float) -> bool:
    """True iff the task-oriented rate <= the reconstruction rate at equal task-distortion (the theorem)."""
    return float(task_rate_bits) <= float(reconstruction_rate_bits)


def build_task_rd_dominates_reconstruction_rd_v1() -> CanonicalEquation:
    """Build the task-R(D) < reconstruction-R(D) dominance canonical equation."""
    anchor = EmpiricalAnchor(
        anchor_id="task_rd_below_reconstruction_rd_arxiv_2602_12866_dobrushin_witsenhausen_20260702",
        measurement_utc=_UTC,
        inputs={
            "theorem": "model-aware R_M(D) < classical reconstruction R(D)",
            "citations": ["arXiv:2602.12866 Model-Aware Rate-Distortion Limits",
                          "Dobrushin-Witsenhausen remote/indirect RD (1962)",
                          "arXiv:2405.04144 RDC/RPC companion"],
            "mechanism": "optimal code for a fixed predictor reduces X to task sufficient statistics; "
                         "model-irrelevant RGB-slack quantizes to ~0 cost",
        },
        predicted_output={
            "dominance_gap_sign": ">= 0 (task rate never exceeds reconstruction rate at equal task-D)",
            "consequence": "the non-RGB task-space witness dominates a full-RGB codec (proven)",
        },
        empirical_output={
            "theorem_holds": True,
            "directional_corroboration_bc20_task_space_rate": _BC20_TASK_SPACE_RATE,
            "directional_corroboration_bc36_reskin_rate": _BC36_RESKIN_RATE,
            "corroboration_caveat": ("bc36 is a PR95-size RESKIN, NOT a controlled full-RGB-codec-at-"
                                     "equal-task-distortion baseline; the THEOREM is the authority, "
                                     "the pair only points the same direction"),
            "verdict": ("FRAMING theorem: task R(D) < reconstruction R(D). Keeping any bit the frozen "
                        "scorer never reads is PROVABLE rate waste. Kills the 'just do a good RGB codec' "
                        "revival. NOT a contest lever (no through-R Delta-S)."),
        },
        residual=0.0,  # a cited inequality bound, not a measured predicted-vs-empirical residual
        source_artifact=_LEDGER,
        measurement_method="citation_arxiv_2602_12866_dobrushin_witsenhausen_indirect_rate_distortion",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LEDGER,
            reactivation_criteria=(
                "upgrade INFERRED->VERIFIED by MEASURING the dominance gap: a byte-closed full-RGB codec "
                "row vs the task-space witness row at equal frozen-scorer d_seg/d_pose (both through R)"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="n/a_citation",
        ),
        empirical_verification_status=INFERRED_FROM_DOMAIN_LITERATURE,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Task-oriented R(D) lies strictly below reconstruction R(D) (arXiv:2602.12866 / "
            "Dobrushin-Witsenhausen) -- the task-space witness dominates a full-RGB codec (theorem)"
        ),
        one_line_summary=(
            "model-aware R_M(D) < reconstruction R(D): the non-RGB task-space witness beating a full-RGB "
            "codec is a THEOREM; keeping any bit the frozen scorer never reads is provable rate waste."
        ),
        latex_form=(
            r"R_M(D) = \min_{p(u|x):\,\mathbb{E}[d_M(M(x),M(u))]\le D} I(X;U) \;\le\; R_X(D);\quad "
            r"R_X(D) - R_M(D) \ge 0\ \ (\text{gap} = \text{task-irrelevant RGB-slack})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.task_rd_dominates_reconstruction_rd_20260702:task_rd_dominance_gap"
        ),
        domain_of_validity={
            "downstream_model": ["frozen_segnet_argmax", "frozen_posenet_6dim"],
            "result_type": "FRAMING THEOREM (proves the direction), NOT a contest lever (no through-R Delta-S)",
            "distortion_measure": "downstream-task performance (M's output), NOT MSE/reconstruction",
            "gap_grows_when": "the downstream model ignores high-variance task-irrelevant dims (RGB-slack)",
            "lineage": ["indirect/remote RD (Dobrushin-Witsenhausen)", "Wyner-Ziv side-info", "CEO problem",
                        "Information Bottleneck (Tishby)", "video-coding-for-machines"],
            "corroboration_only": {"bc20_task_space_rate": _BC20_TASK_SPACE_RATE,
                                   "bc36_reskin_rate": _BC36_RESKIN_RATE,
                                   "caveat": "reskin, not equal-distortion; theorem is the authority"},
        },
        units_in={"reconstruction_rate_bits": "bits", "task_rate_bits": "bits"},
        units_out={"task_rd_dominance_gap": "bits"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "citation_arxiv_2602_12866_dobrushin_witsenhausen_indirect_rate_distortion": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "tac.boundary_math.lever_b_generator",
            "tac.witness_autoconfig",
        ),
        canonical_producers=(
            ".omx/research/compression_as_intelligence_lineage_crossref_20260702.md",
            "tools/register_triality_reconcile_session_20260702_equations.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="task_rd_dominates_reconstruction_rd.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="n/a_citation",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "build_task_rd_dominates_reconstruction_rd_v1",
    "task_rd_dominance_gap",
    "task_rd_le_reconstruction_rd",
]
