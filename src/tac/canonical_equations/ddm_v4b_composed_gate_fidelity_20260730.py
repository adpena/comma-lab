# SPDX-License-Identifier: MIT
"""Canonical law: composed-gate instrument fidelity (v4b, real evaluator row).

Registers ``ddm_v4b_composed_gate_instrument_fidelity_v1`` — the measured law
that the per-pair realized composition chain (frozen CPU-torch PoseNet oracle
-> two-plane static-horizon compose -> f16-quantized params -> byte-close ->
receiver decode) predicts the REAL ``upstream/evaluate.py`` composed score to
instrument grade on this vehicle:

    S_pred = 100*d_seg^ship + sqrt(10 * mean_p d_pose^realized(p)) + 25*B/N
    |S_pred - S_measured| <= 3e-6   (measured, v4b gate)

Empirical anchor (v4b composed gate, 2026-07-30, [macOS-CPU advisory], n600):
archive 274,479 B (sha 3b3a4abf...) = Knee-A tokens x full-600 re-solved pose
(best-of single/two-plane) x 75 B selector x s_r=1 static-horizon receiver
(v = 437 DERIVED from intrinsics at decode; ZERO shipped mask bytes).
Real evaluate.py: d_pose 0.06365131 / d_seg 0.00553676 -> recomputed
S = 1.534258 vs predicted 1.534255 (residual 3e-6; d_pose predicted-vs-
measured residual 5.1e-7). Two teeth: (1) INSTRUMENT FIDELITY — per-pair
realized acceptance on this chain is gate-grade, so composed predictions from
it carry pre-gate authority (gates still verify, never claim); (2) PHYSICS-
BEATS-SEMANTICS — the realizable derived-horizon partition (one row from
K^-T [0,-1,0]) outperformed the ILLEGAL GT semantic mask upper bound by
-0.180 S, because the parallax split the warp needs is geometric (depth), not
semantic (class). Sister receipts: qa45 (static beats GT on the full base,
0.543x), ck1 (Knee-A pose damage = stale-params INSTANCE, re-solve parity
0.98x vs full base).
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

EQUATION_ID = "ddm_v4b_composed_gate_instrument_fidelity_v1"
REPO = Path(__file__).resolve().parents[3]
MEMO = REPO / ".omx/research/ddm_v4b_composed_gate_20260730.md"
N_ORIG_BYTES = 37_545_489  # upstream/evaluate.py:63 denominator (frozen contest constant)


def composed_gate_fidelity(
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    predicted_s: float,
) -> dict[str, float]:
    """Contest-S recomposition + prediction residual (pure closed form).

    Mirrors ``upstream/evaluate.py:92`` term-for-term; ``prediction_residual``
    is the law's diagnostic — a gate-grade composition chain sits at ~1e-6.
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
        "prediction_residual": abs(s - float(predicted_s)),
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, float]:
    required = {"d_seg", "d_pose", "archive_bytes", "predicted_s"}
    if set(inputs) != required:
        raise ValueError("composed-gate fidelity inputs differ from the canonical contract")
    return composed_gate_fidelity(inputs["d_seg"], inputs["d_pose"], inputs["archive_bytes"], inputs["predicted_s"])


register_evaluator(EQUATION_ID, _evaluate)


def build_ddm_v4b_composed_gate_instrument_fidelity_v1(
    *,
    source_receipt: Path = MEMO,
) -> CanonicalEquation:
    """Build the v4b composed-gate instrument-fidelity law with its measured anchor."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Re-anchor at every composed gate (v4c onward): if any future gate's "
            "prediction residual exceeds ~1e-3 the fidelity leg is FALSIFIED at that "
            "grammar and the composition chain must be re-audited before its "
            "predictions carry pre-gate authority; contest-CPU/CUDA re-measure "
            "required before any promotion use."
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_fp32_batch16_threads2",
        captured_at_utc="2026-07-30T21:30:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="v4b_composed_gate_evaluate_py_n600_20260730",
        measurement_utc="2026-07-30T21:30:00Z",
        inputs={
            "d_seg": 0.00553676,
            "d_pose": 0.06365131,
            "archive_bytes": 274_479,
            "predicted_s": 1.534255,
            "archive_sha256": ("3b3a4abf65296c2a98871c73a4ac74b5fd2ac6ed9e9ebe691375efa93e13868a"),
        },
        predicted_output={"S": 1.534255, "prediction_residual": 0.0},
        empirical_output={
            "S": 1.534258,  # recomputed from evaluator components (printed 1.53)
            "seg_term": 0.553676,
            "pose_term": 0.797818,
            "rate_term": 0.182764,
            "prediction_residual": 3e-06,
            "d_pose_prediction_residual": 5.1e-07,
        },
        residual=3e-06,
        source_artifact=str(MEMO.relative_to(REPO)),
        measurement_method=(
            # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
            "real upstream/evaluate.py n600 --device cpu on the exact archive bytes "
            "(sha-pinned) in the pfs1 D1 eval_root protocol; prediction = shipped "
            "d_seg (exact, tokens byte-identical to the wr1 Knee-A gate) + per-pair "
            "realized d_pose table (v4b_ship_table.json) + recomputed rate; ops "
            "receipt: clean single detached run after a SIGURG-orphan double-run "
            "was tree-killed (no contaminated numbers consumed)"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    anchor_v4c = EmpiricalAnchor(
        anchor_id="v4c_composed_gate_evaluate_py_n600_20260730",
        measurement_utc="2026-07-30T23:59:00Z",
        inputs={
            "d_seg": 0.00431179,
            "d_pose": 0.01038450,
            "archive_bytes": 359_750,
            "predicted_s": 0.992834,
            "archive_sha256_prefix": "b6365270ddc55fde",
        },
        predicted_output={"S": 0.992834, "prediction_residual": 0.0},
        empirical_output={
            "S": 0.992972,  # recomputed from evaluator components (printed 0.99)
            "seg_term": 0.431179,
            "pose_term": 0.322250,
            "rate_term": 0.239543,
            "prediction_residual": 1.38e-04,
            "d_pose_prediction_residual": 1e-07,
        },
        residual=1.38e-04,
        source_artifact=str(MEMO.relative_to(REPO)),
        measurement_method=(
            # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
            "real upstream/evaluate.py n600 --device cpu, exact archive bytes, same "
            "eval_root protocol; the STANDING RE-ANCHOR DUTY's first exercise: "
            "residual 1.38e-4 < the 1e-3 falsification threshold and lies entirely "
            "inside the pre-declared d_seg evaluate band (+-2.8e-5 d_seg = +-2.8e-3 "
            "S); pose predicted-vs-measured ~1e-7 — law HOLDS on the v4c grammar "
            "(cell_drop50 base + static two-plane re-solve + photometric (a,b) rung "
            "+ lossless trio); first measured sub-1.0 own-vehicle row"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    anchor_v4d = EmpiricalAnchor(
        anchor_id="v4d_composed_gate_evaluate_py_n600_20260731",
        measurement_utc="2026-07-31T00:00:00Z",
        inputs={
            "d_seg": 0.00431179,
            "d_pose": 0.00858145,
            "archive_bytes": 360_238,
            "predicted_s": 0.963986,
            "archive_sha256_prefix": "f1f3288062468e97",
        },
        predicted_output={"S": 0.963986, "prediction_residual": 0.0},
        empirical_output={
            "S": 0.9639878179186275,  # recomputed from evaluator components (printed 0.96)
            "seg_term": 0.431179,
            "pose_term": 0.29294112036380276,
            "rate_term": 0.23986769755482476,
            "prediction_residual": 1.8179186275224524e-06,
        },
        residual=1.8179186275224524e-06,
        source_artifact=str(MEMO.relative_to(REPO)),
        measurement_method=(
            # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
            "real upstream/evaluate.py n600 --device cpu, exact archive bytes, same "
            "eval_root protocol; THIRD anchor, tightest yet (1.82e-6 < v4b 3e-6 < "
            "v4c 1.38e-4): the v4d refinement stack (QA66 per-pair beta + QA65 dim0 "
            "offset-lattice re-solve + (a,b) re-fit, +488 B over v4c) composed over "
            "the byte-identical v4c token base; d_seg reproduced to the digit "
            "(sha-checked token identity), d_pose predicted-vs-measured at the 1e-6 "
            "class; monotone-safe best-of construction — law HOLDS across three "
            "grammar generations"
        ),
        provenance=provenance,
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Composed-gate instrument fidelity (v4b evaluator row)",
        one_line_summary=(
            "Per-pair realized composition predicts the real evaluator to 3e-6 "
            "(d_pose 5.1e-7) — gate-grade; the derived static horizon beat the "
            "GT-mask UB."
        ),
        latex_form=(
            r"S_{pred}=100\,d_{seg}^{ship}+\sqrt{10\,\overline{d_{pose}^{real}}}"
            r"+\frac{25B}{N},\qquad |S_{pred}-S_{eval}|\le 3\times10^{-6}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_v4b_composed_gate_fidelity_20260730:composed_gate_fidelity"
        ),
        domain_of_validity={
            "vehicle": (
                "own-vehicle composed line: Knee-A token base + 6dof f16 pose field + "
                "selector + s_r=1 static-horizon two-plane receiver (v4b grammar)"
            ),
            # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
            "evaluator": "real upstream/evaluate.py n600 --device cpu, exact archive bytes",
            "evidence_axis": "[macOS-CPU advisory] — NOT contest-CPU (Apple Silicon host)",
            "fidelity_scope": (
                "INSTANCE(v4b grammar) with a standing re-anchor duty at every "
                "subsequent composed gate; NOT a claim that arbitrary future grammars "
                "compose confound-free"
            ),
            "physics_beats_semantics_scope": (
                "parallax far/ground partition for the two-plane warp: derived "
                "vanishing-row split >= GT semantic class mask (measured on full base "
                "0.543x UB and knee base -0.180 S); NOT a general mask claim"
            ),
            "research_only": True,
            "score_claim": False,
            "verdict_scope": "INSTANCE_V4B_GRAMMAR_WITH_STANDING_REANCHOR_DUTY",
            "excluded": [
                "contest score / frontier movement (advisory axis; pointer 0.1910828242 UNMOVED)",
                "any promotion or submission use without contest-CPU/CUDA re-measure",
                "gate-skipping: predictions carry PRE-gate authority only; gates still verify",
            ],
        },
        units_in={
            "d_seg": "fraction of disagreeing argmax pixels (evaluator SegNet term)",
            "d_pose": "MSE on first 6 PoseNet dims (evaluator PoseNet term)",
            "archive_bytes": "exact archive.zip size in bytes",
            "predicted_s": "score units (pre-gate composed prediction)",
        },
        units_out={
            "seg_term": "score units",
            "pose_term": "score units",
            "rate_term": "score units",
            "S": "score units",
            "prediction_residual": "score units (|predicted - measured|)",
        },
        empirical_anchors=(anchor, anchor_v4c, anchor_v4d),
        predicted_vs_empirical_residual={"S": 1.8179186275224524e-06},
        last_calibration_utc="2026-07-31T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.costate_digest",
            "experiments.stage_v4b_realized_gate",
        ),
        canonical_producers=("experiments.ddm_ck1_build_composed_archive",),
        provenance=provenance,
    )


def populate_ddm_v4b_composed_gate_instrument_fidelity_v1(
    *,
    source_receipt: Path = MEMO,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the v4b composed-gate fidelity law through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_v4b_composed_gate_instrument_fidelity_v1(source_receipt=source_receipt)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "v4b measured-gate equations leg (drift-detector per-leg); advisory axis; "
            "score_claim=false; instrument-fidelity + physics-beats-semantics anchors"
        ),
    )
    return equation
