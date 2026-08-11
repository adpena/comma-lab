# SPDX-License-Identifier: MIT
"""Canonical equation anchors for the 2026-08-10 to 2026-08-11 DDM arc.

Two findings extend existing laws instead of creating twin registries:
``cpu_cuda_score_gap_v1`` receives the LC2 opposite-sign paired-device anchor,
and ``realization_breakeven_bytes_v1`` receives PZ4P-envelope to PZ4R-realized
yield.  PS135 radius-2 multistart escape is a new, advisory-only law.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import (
    eval_cpu_cuda_score_gap,
    eval_radius2_multistart_singleton_escape,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

REPO = Path(__file__).resolve().parents[3]

CPU_CUDA_EQUATION_ID = "cpu_cuda_score_gap_v1"
REALIZATION_EQUATION_ID = "realization_breakeven_bytes_v1"
MULTISTART_EQUATION_ID = "radius2_multistart_singleton_escape_v1"

RC64P_CPU_RECEIPT = (
    REPO
    / ".omx/research/ddm_rc64p_native_cpu_decode_20260810_modal/contest_cpu_diag"
)
PZ4R_MEMO = REPO / ".omx/research/ddm_pz4r_pgq1_receiver_20260811.md"
PS135_PASS2_RECEIPT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/leg_a/passes/pass_02/receipt.json"
)


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def build_lc2_device_delta_anchor(
    *, source_receipt: Path = RC64P_CPU_RECEIPT
) -> EmpiricalAnchor:
    """Build the exact same-archive LC2 CPU/CUDA opposite-sign anchor."""

    score_cpu = 0.20728492781521812
    score_cuda = 0.16959899569230852
    empirical_delta = eval_cpu_cuda_score_gap(
        {"score_cpu": score_cpu, "score_cuda": score_cuda}
    )
    prior_hnerv_delta = 0.033
    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "recalibrate the per-lineage device model after at least three paired exact "
            "anchors for the PR130 semantic-pose lineage"
        ),
        measurement_axis="[contest-CPU] + [contest-CUDA T4]",
        hardware_substrate="linux_x86_64_cpu_plus_t4",
        captured_at_utc="2026-08-11T00:45:00Z",
    )
    return EmpiricalAnchor(
        anchor_id="lc2_identical_bytes_opposite_sign_cpu_cuda_20260811",
        measurement_utc="2026-08-11T00:45:00Z",
        inputs={
            "archive_bytes": 187_226,
            "archive_sha256": "f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45",
            "archive_family": "pr130_semantic_pose_lineage_lc2",
            "score_cpu": score_cpu,
            "score_cuda": score_cuda,
        },
        predicted_output={
            "cuda_minus_cpu_score": prior_hnerv_delta,
            "prediction_scope": "older_hnerv_cluster_only",
        },
        empirical_output={
            "cuda_minus_cpu_score": empirical_delta,
            "cpu_minus_cuda_score": -empirical_delta,
            "cpu_d_seg": 0.00042739,
            "cpu_d_pose": 0.00015904,
            "seg_cpu_over_cuda": 1.45,
            "pose_cpu_over_cuda": 6.6,
            "sign_opposes_pr102_precedent": True,
        },
        residual=abs(empirical_delta - prior_hnerv_delta),
        source_artifact=_relative_or_absolute(source_receipt),
        measurement_method="same_archive_paired_contest_cpu_cuda_exact_eval",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_pz4_realization_yield_anchor(
    *, source_receipt: Path = PZ4R_MEMO
) -> EmpiricalAnchor:
    """Build the PZ4P envelope-to-PZ4R receiver-closed byte-yield anchor."""

    envelope_saving = 19_221
    realized_saving = 4_089
    yield_fraction = realized_saving / envelope_saving
    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "append another envelope/receiver pair only after exact counted archive "
            "realization and receiver parse-back"
        ),
        measurement_axis="[macOS-CPU scorer-free receiver build]",
        hardware_substrate="m5_max_macos_cpu",
        captured_at_utc="2026-08-11T00:00:00Z",
    )
    return EmpiricalAnchor(
        anchor_id="pz4p_envelope_to_pz4r_receiver_closed_yield_corrected_20260811",
        measurement_utc="2026-08-11T00:00:00Z",
        inputs={
            "base_archive_bytes": 187_226,
            "envelope_archive_bytes": 168_005,
            "realized_archive_bytes": 183_137,
            "envelope_saving_bytes": envelope_saving,
            "realized_saving_bytes": realized_saving,
        },
        predicted_output={"realization_fraction": 1.0},
        empirical_output={
            "realization_fraction": yield_fraction,
            "realized_saving_bytes": realized_saving,
            "decoder_required_bytes_restored": 15_132,
            "score_recovery_status": "UNMEASURED",
            "formulation": "CPR1 held decoder-required basis and coefficient-map data",
        },
        residual=1.0 - yield_fraction,
        source_artifact=_relative_or_absolute(source_receipt),
        measurement_method="pz4p_envelope_to_pz4r_receiver_closed_archive_yield",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_ps135_multistart_equation(
    *, source_receipt: Path = PS135_PASS2_RECEIPT
) -> CanonicalEquation:
    """Build the advisory radius-2 multistart singleton-escape equation."""

    inputs: Mapping[str, Any] = {
        "pair_count": 600,
        "accepted_rows": 597,
        "score_before": 0.2072899013894104,
        "score_after": 0.18474482031130968,
        "d_pose_before": 0.00015904,
        "d_pose_after": 0.000030088120534088604,
    }
    pass1 = eval_radius2_multistart_singleton_escape(inputs)
    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "transfer to CUDA only after exact contest-CUDA replay of the final retained archive"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="m5_max_macos_cpu",
        captured_at_utc="2026-08-11T10:34:26Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="ps135_radius2_multistart_pass1_pass2_n600_20260811",
        measurement_utc="2026-08-11T10:34:26Z",
        inputs=dict(inputs),
        predicted_output={
            "escaped": False,
            "accepted_rows": 0,
            "prediction_scope": "shipped_plus_or_minus_one_singleton_optimum",
        },
        empirical_output={
            **pass1,
            "pass1_accepted_rows": 597,
            "pass1_archive_bytes": 187_223,
            "pass1_score": 0.18474482031130968,
            "pass2_accepted_rows": 517,
            "pass2_archive_bytes": 187_221,
            "pass2_archive_sha256": "b8c3b1187cff48eb8208973536c8f94874c78fb4ad68df84e92fbe9418c1b24a",
            "pass2_score": 0.17952896607020802,
            "pass2_d_pose": 0.000014717098986238853,
            "cuda_transfer": "OPEN_HYPOTHESIS",
        },
        residual=597 / 600,
        source_artifact=_relative_or_absolute(source_receipt),
        measurement_method="macos_cpu_advisory_n600_receiver_closed_radius2_multistart",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=MULTISTART_EQUATION_ID,
        name="Radius-2 multistart escape from a shipped singleton optimum",
        one_line_summary=(
            "PS135 radius-2 native, wrong-sign, and projected starts escaped the shipped "
            "singleton optimum on 597/600 rows; CUDA transfer remains open."
        ),
        latex_form=(
            r"E_{r=2}=\mathbf{1}[A_{r=2}>0\land S_{r=2}<S_0],\quad "
            r"q=A_{r=2}/N,\quad \Delta S=S_0-S_{r=2}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.evaluators:eval_radius2_multistart_singleton_escape"
        ),
        domain_of_validity={
            "axis": "[macOS-CPU advisory]",
            "population": "n600 LC2 carrier rows",
            "neighborhood": (
                "radius-2 rank-3 cubes around native GN, wrong-sign GN, and "
                "PR133-projected row-local starts, plus singleton control"
            ),
            "verdict_scope": "INSTANCE(LC2 PS135 GEN-2 passes 1-2)",
            "excluded": [
                "contest-CUDA transfer without exact replay",
                "claim that further passes converge",
                "claim that the plus-or-minus-one singleton result was wrong",
            ],
            "score_claim": False,
        },
        units_in={
            "pair_count": "rows",
            "accepted_rows": "rows",
            "score_before": "score_units",
            "score_after": "score_units",
            "d_pose_before": "mse",
            "d_pose_after": "mse",
        },
        units_out={
            "escaped": "boolean",
            "accepted_fraction": "fraction",
            "score_reduction": "score_units",
            "d_pose_reduction": "mse",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "macos_cpu_advisory_n600_receiver_closed_radius2_multistart": 597 / 600
        },
        last_calibration_utc="2026-08-11T10:34:26Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            ".omx.research.charters.ddm_js1_global_joint_solve",
            ".omx.state.main_hot_state",
        ),
        canonical_producers=(
            "experiments.ddm_ps135_pose_resolve",
            "tac.canonical_equations.ddm_cn4_arc_consolidation_20260811",
        ),
        provenance=provenance,
    )


def populate_ddm_cn4_arc_equations(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str = "codex",
    subagent_id: str = "ddm_cn4",
) -> tuple[str, ...]:
    """Append two anchors and register the PS135 law through locked helpers."""

    from tac.canonical_equations.registry import (
        register_canonical_equation,
        update_equation_with_domain_refinement,
        update_equation_with_empirical_anchor,
    )

    update_equation_with_domain_refinement(
        CPU_CUDA_EQUATION_ID,
        domain_of_validity_extension={
            "archive_classes": ["hnerv_family", "pr130_semantic_pose_lineage"],
            "domain_of_validity_excluded": [
                "universal CPU/CUDA gap sign transfer across archive lineages"
            ],
        },
        rationale=(
            "LC2 exact paired axes reverse the older HNeRV-cluster sign; preserve a "
            "per-archive, per-lineage device delta rather than a universal sign prior."
        ),
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cn4 lc2 opposite-sign device-domain refinement",
    )
    update_equation_with_empirical_anchor(
        CPU_CUDA_EQUATION_ID,
        build_lc2_device_delta_anchor(),
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cn4 lc2 identical-bytes paired CPU/CUDA anchor",
    )
    update_equation_with_empirical_anchor(
        REALIZATION_EQUATION_ID,
        build_pz4_realization_yield_anchor(),
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cn4 pz4p envelope to pz4r receiver-closed yield anchor",
    )
    register_canonical_equation(
        build_ps135_multistart_equation(),
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cn4 ps135 radius-2 multistart singleton-escape law",
    )
    return (CPU_CUDA_EQUATION_ID, REALIZATION_EQUATION_ID, MULTISTART_EQUATION_ID)


__all__ = [
    "CPU_CUDA_EQUATION_ID",
    "MULTISTART_EQUATION_ID",
    "REALIZATION_EQUATION_ID",
    "build_lc2_device_delta_anchor",
    "build_ps135_multistart_equation",
    "build_pz4_realization_yield_anchor",
    "populate_ddm_cn4_arc_equations",
]
