# SPDX-License-Identifier: MIT
"""Canonical law: solved-object realized-S rate dominance (r6cal, real evaluator row).

Registers ``ddm_r6cal_solved_object_realized_rate_dominance_v1`` — the
measured law that a VALUE-MATERIALIZED box solve (per-pixel plane records,
no PREDICT stage) is rate-dominated through the real ``upstream/evaluate.py``
chain, and that its description compression is DEAD at the coder layer:

    S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/N_orig,   N_orig = 37,545,489
    rate_share = (25*bytes/N_orig) / S

Empirical anchor (r6cal, 2026-07-27, [macOS-CPU advisory], rc=0, n600):
the ms2r_r3 solved-seg archive (291,205,400 B, sha e3d0581f...) scored
S = 194.43 = 0.115997 seg + 0.407838 pose + 193.901723 rate — 99.731% of S
is rate while d_seg 0.00115997 is INSIDE the #613 box (<= 0.00116) and the
chain reproduced the solve's own scorer record to |dd_seg| = 2.9e-8 /
|dd_pose| = 6.1e-9. Description-compression floor (companion receipt): the
1,200 plane records are entropy-saturated (H0 = 7.999 / H1 = 7.986 b/B,
already Brotli-Q11 internally; RAW wins 50/50 coder races) — 291,205,400 B
IS the coded floor for this representation. Consequence (the law's teeth):
the ONLY rate lever for a value-materialized solve is the missing
PREDICT/describe stage (descriptor_len = 0 on all records, measured), i.e.
video-derived parametric description, never a better coder. Sister receipt:
the #603 debt table (3,103,689 errors 100.00% owned at 93.8 B/error vs
1.2731 B/error score-optimal = 74x over) prices why no admission rule can
accept this representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_r6cal_solved_object_realized_rate_dominance_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / ".omx/research/r6cal_asbuilt_row_receipt_20260727.json"
FLOOR_RECEIPT = REPO / ".omx/research/r6cal_description_compression_floor_20260727.json"
N_ORIG_BYTES = 37_545_489  # upstream/evaluate.py:63 denominator (frozen contest constant)


def realized_score_decomposition(
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
) -> dict[str, float]:
    """Exact contest-S decomposition + rate share (pure closed form).

    Mirrors ``upstream/evaluate.py:92`` term-for-term; the rate_share output
    is the law's diagnostic — a value-materialized solve sits near 1.0.
    """

    if not (0.0 <= float(d_seg) <= 1.0):
        raise ValueError("d_seg must be a fraction in [0, 1]")
    if float(d_pose) < 0.0:
        raise ValueError("d_pose must be nonnegative")
    if int(archive_bytes) <= 0:
        raise ValueError("archive_bytes must be positive")
    seg_term = 100.0 * float(d_seg)
    pose_term = (10.0 * float(d_pose)) ** 0.5
    rate_term = 25.0 * int(archive_bytes) / N_ORIG_BYTES
    s = seg_term + pose_term + rate_term
    return {
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "S": s,
        "rate_share": rate_term / s if s > 0.0 else 0.0,
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, float]:
    required = {"d_seg", "d_pose", "archive_bytes"}
    if set(inputs) != required:
        raise ValueError("rate-dominance inputs differ from the canonical callable contract")
    return realized_score_decomposition(
        inputs["d_seg"], inputs["d_pose"], inputs["archive_bytes"]
    )


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_r6cal_solved_object_realized_rate_dominance_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the r6cal solved-object rate-dominance law with its measured anchor."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Re-anchor when any archive from the ms2r_r3 lineage gains a PREDICT/"
            "describe stage (descriptor_len > 0) or when a coder beats RAW on the "
            "plane records (would falsify the coding-dead leg); contest-CPU/CUDA "
            "re-measure required before any promotion use."
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_fp32_batch16_threads4",
        captured_at_utc="2026-07-27T23:59:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="r6cal_ms2r_r3_archive_evaluate_py_n600_20260727",
        measurement_utc="2026-07-27T23:59:00Z",
        inputs={
            "d_seg": 0.00115997,
            "d_pose": 0.01663316,
            "archive_bytes": 291_205_400,
            "archive_sha256": "e3d0581ff4a3f475057e77e530374dad444b640a049b058cd66b37563534773e",
        },
        predicted_output={"S": 194.42556, "rate_share": 0.99731},
        empirical_output={
            "S": 194.43,  # evaluator-printed (rounded); components reproduce 194.42556
            "seg_term": 0.115997,
            "pose_term": 0.407838,
            "rate_term": 193.901723,
        },
        residual=0.0,  # identity law evaluated on the evaluator's own printed components
        source_artifact=str(RECEIPT.relative_to(REPO)),
        measurement_method=(
            # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
            "real upstream/evaluate.py n600 --device cpu rc=0 on the exact archive "
            "bytes (sha-pinned); chain fidelity vs solve record |dd_seg|=2.9e-8, "
            "|dd_pose|=6.1e-9; companion coder-floor receipt "
            f"{FLOOR_RECEIPT.name} (H1=7.986 b/B, RAW wins 50/50)"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Solved-object realized-S rate dominance (r6cal evaluator row)",
        one_line_summary=(
            "A value-materialized box solve is 99.73% rate through the real "
            "evaluator with d_seg in-box; its records are entropy-saturated so "
            "the only rate lever is the missing PREDICT/describe stage."
        ),
        latex_form=(
            r"S=100\,d_{seg}+\sqrt{10\,d_{pose}}+\frac{25\,B}{N},\quad "
            r"\mathrm{rate\_share}=\frac{25B/N}{S}\;\xrightarrow{\text{value-materialized}}\;0.997,"
            r"\quad H_1(\text{records})=7.986\,\mathrm{b/B}\Rightarrow B_{\min}^{\text{coder}}=B"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_r6cal_solved_object_rate_dominance_20260728:"
            "realized_score_decomposition"
        ),
        domain_of_validity={
            "archive": "ms2r_r3 solved-seg lineage (sha e3d0581f...), value-materialized plane records",
            # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
            "evaluator": "real upstream/evaluate.py n600, --device cpu (rc=0), exact archive bytes",
            "evidence_axis": "[macOS-CPU advisory] — NOT contest-CPU (Apple Silicon host)",
            "coding_dead_scope": (
                "INSTANCE(this representation): H0=7.999/H1=7.986 b/B, internal Brotli-Q11, "
                "RAW wins 50/50 coder races — NOT a claim about parametric descriptions"
            ),
            "consequence": (
                "rate lever = PREDICT/describe stage only (descriptor_len=0 measured on all "
                "1,200 records); #603 pricing 93.8 B/error vs 1.2731 score-optimal = 74x over"
            ),
            "research_only": True,
            "score_claim": False,
            "verdict_scope": "INSTANCE_VALUE_MATERIALIZED_SOLVE_REPRESENTATION",
            "excluded": [
                "contest score / frontier movement (advisory axis)",
                "coder-race verdicts on parametric/described payloads",
                "any promotion or submission use without contest-CPU/CUDA re-measure",
            ],
        },
        units_in={
            "d_seg": "fraction of disagreeing argmax pixels (evaluator SegNet term)",
            "d_pose": "MSE on first 6 PoseNet dims (evaluator PoseNet term)",
            "archive_bytes": "exact archive.zip size in bytes",
        },
        units_out={
            "seg_term": "score units",
            "pose_term": "score units",
            "rate_term": "score units",
            "S": "score units",
            "rate_share": "fraction of S carried by the rate term",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"S": 0.0},
        last_calibration_utc="2026-07-27T23:59:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.costate_digest",
            "tac.optimization.direct_description_minimizer",
        ),
        canonical_producers=(
            "tools.realize_ddm_m7_relaxed_receiver",
        ),
        provenance=provenance,
    )


def populate_ddm_r6cal_solved_object_realized_rate_dominance_v1(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the r6cal rate-dominance law through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_r6cal_solved_object_realized_rate_dominance_v1(
        source_receipt=source_receipt
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "r6cal owed triality leg registered at the post-merge boundary; advisory "
            "axis; score_claim=false; the gap-is-RATE convergent finding's law form"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "N_ORIG_BYTES",
    "build_ddm_r6cal_solved_object_realized_rate_dominance_v1",
    "populate_ddm_r6cal_solved_object_realized_rate_dominance_v1",
    "realized_score_decomposition",
]
