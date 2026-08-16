# SPDX-License-Identifier: MIT
"""Canonical equation: wd3 fresh-topology handicap — Adam state carries pose descent;
fresh-init dense w56 asymptotes 2.5x above the warm-lineage seg floor at matched budget.

MEASURED 2026-08-15 on the wd3 scorer-aware width-distillation instrument
(``experiments/ddm_wd3_scorer_aware_width_distillation.py``, MPS training with
``PYTORCH_ENABLE_MPS_FALLBACK=0`` determinism gate; fixed n60 evaluation subset through
the frozen scorers — the SAME instrument for every arm, per the batch-shape-is-part-of-
the-instrument law). Axis ``[Darwin-mps training / n60 advisory]``; research_only; no
score claim; NEVER an exact row.

Two measured facts, one law
---------------------------
1. **Optimizer-state pose carry (W0 A/B, matched weights + subsets + epochs):** from the
   SAME wd2-lineage weights, 4 epochs with the inherited Adam state (W0_warm) moved
   d_pose −43.5% while a fresh Adam (W0_reset) moved −14.4% — the optimizer STATE
   carries ~3x of the per-window pose descent. Seg was a near-tie. Consequence: a fresh
   topology's early pose sluggishness is EXPECTED, never a family negative.
2. **Fresh-init matched-budget endpoint (D56, 65 ep from scorer-free birth):** dense
   w56 lands hard_d_seg 0.002682 — FLAT from ~ep40 (an asymptote, not a slope) — vs the
   W0 floor 0.00106-0.00109 at matched TOTAL budget (2.5x), with d_pose ~0.31
   non-converged (the fact-1 handicap compounded over a whole run plus no wd2
   curriculum inheritance). verdict_scope: instance (D56, n60); per the sealed spec law
   a negative cannot be emitted from n60 — the n120 seeded confirmation is the owed
   discharge before any family word.

The LAW this registers: comparisons across warm-vs-fresh students are CONFOUNDED by the
optimizer-state pose carry unless (a) pose is read per fact 1's discount, and (b) seg is
read at its asymptote, not mid-descent. The D56 seg asymptote is the un-confounded
capacity read; the pose endpoint is not.

Memo: ``.omx/research/ddm_wd3_d56_fresh_dense_verdict_20260815.md``.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "wd3_fresh_topology_pose_carry_v1"
AXIS = "[Darwin-mps training / n60 advisory] frozen-scorer fixed-subset instrument"
MEMO = ".omx/research/ddm_wd3_d56_fresh_dense_verdict_20260815.md"
D56_CONFIG_SHA256 = "f8e2c8e625bd01a8e2e3263e6a7b43cfb226c0fef7add0efff45cb36387f8526"

#: MEASURED 2026-08-15 (W0 A/B, 4-epoch windows, matched weights/subset/epochs).
MEASURED_W0_WARM_DPOSE_WINDOW_DELTA = -0.435
MEASURED_W0_RESET_DPOSE_WINDOW_DELTA = -0.144
#: MEASURED endpoints at matched total budget (ep65-equivalent, n60 subset).
MEASURED_ENDPOINTS_N60 = {
    "W0_warm": {"hard_d_seg": 0.0010857, "d_pose": 0.02294},
    "W0_reset": {"hard_d_seg": 0.0010628, "d_pose": 0.03408},
    "D56_fresh": {"hard_d_seg": 0.002682, "d_pose": 0.3105},
}


def build_wd3_fresh_topology_pose_carry_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=D56_CONFIG_SHA256,
        source_path=MEMO,
        captured_at_utc="2026-08-16T02:20:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="wd3_w0_warm_vs_reset_pose_carry_20260815",
            measurement_utc="2026-08-15T23:30:00Z",
            inputs={
                "arms": "W0_warm (inherited Adam state) vs W0_reset (fresh Adam), "
                        "SAME wd2-lineage weights, matched n60 subset + 4-epoch window",
                "instrument": "wd3 trainer n60 fixed-subset evals, frozen scorers",
            },
            predicted_output={
                "question": "does the optimizer STATE (not the weights) carry pose descent?"
            },
            empirical_output={
                "warm_dpose_window_delta": MEASURED_W0_WARM_DPOSE_WINDOW_DELTA,
                "reset_dpose_window_delta": MEASURED_W0_RESET_DPOSE_WINDOW_DELTA,
                "carry_ratio": 3.02,
                "seg_verdict": "near-tie",
            },
            residual=0.0,
            source_artifact=MEMO,
            measurement_method=(
                "matched-everything A/B: same weights, same n60 subset, same epochs; the "
                "single varied factor is the Adam state; per-epoch eval JSONs retained"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="wd3_d56_fresh_dense_matched_budget_20260815",
            measurement_utc="2026-08-16T01:40:00Z",
            inputs={
                "arm": "D56 dense_d4_w56, scorer-free birth, 65 epochs (matched total "
                       "budget vs W0's 60+5)",
                "config_sha256": D56_CONFIG_SHA256,
                "run_wall_clock_s": 6446,
            },
            predicted_output={
                "question": "does fresh dense w56 reach the warm-lineage floor at "
                            "matched budget?"
            },
            empirical_output={
                "endpoints_n60": {k: dict(v) for k, v in MEASURED_ENDPOINTS_N60.items()},
                "seg_asymptote_from_epoch": 40,
                "seg_ratio_vs_w0_floor": 2.5,
                "verdict": "NEGATIVE-LEANING",
                "verdict_scope": "instance (D56, n60); n120 confirmation OWED",
            },
            residual=0.0,
            source_artifact=MEMO,
            measurement_method=(
                "full 65-epoch governed watched burn; per-epoch n60 evals through the "
                "same frozen-scorer instrument as the W0 arms; all payloads + stage "
                "checkpoints retained (TRAIN_RESULT score_claim=false)"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="wd3 fresh-topology handicap: optimizer-state pose carry + seg asymptote read",
        one_line_summary=(
            "Adam state carries ~3x per-window pose descent (W0 warm vs reset): read fresh "
            "students at the SEG ASYMPTOTE — D56 fresh w56 flattens at 2.5x the warm floor."
        ),
        latex_form=(
            r"\frac{\Delta d_{\mathrm{pose}}^{\mathrm{warm}}}"
            r"{\Delta d_{\mathrm{pose}}^{\mathrm{reset}}}\approx 3,\qquad "
            r"\mathrm{capacity\ read}=\lim_{e\to\infty} d_{\mathrm{seg}}(e)\ \ "
            r"(\mathrm{asymptote,\ not\ slope})"
        ),
        python_callable_module_path=(
            "experiments.ddm_wd3_scorer_aware_width_distillation:main"
        ),
        domain_of_validity={
            "instrument": "wd3 n60 fixed-subset evals, frozen scorers, MPS-deterministic "
                          "training (PYTORCH_ENABLE_MPS_FALLBACK=0)",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": "instance for the D56 endpoint; the pose-carry LAW is the "
                             "registered object (W0 A/B, one window — widen with F64)",
            "spec_law": "a negative cannot be emitted from n60 (sealed wd3 charter)",
        },
        units_in={"window_delta": "relative d_pose change over a matched epoch window"},
        units_out={"carry_ratio": "warm/reset pose-descent ratio (dimensionless)"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"registration": 0.0},
        last_calibration_utc="2026-08-16T02:20:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "wd3 F64/W96 arm adjudication (#1070 fire chain — pose read discounted, "
            "seg read at asymptote)",
            "any future fresh-vs-warm student comparison (distillation family)",
        ),
        canonical_producers=(
            "experiments/ddm_wd3_scorer_aware_width_distillation.py",
        ),
        provenance=provenance,
    )


def populate_wd3_fresh_topology_pose_carry_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of the D56 verdict landing)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_wd3_fresh_topology_pose_carry_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "MEASURED_ENDPOINTS_N60",
    "MEASURED_W0_RESET_DPOSE_WINDOW_DELTA",
    "MEASURED_W0_WARM_DPOSE_WINDOW_DELTA",
    "build_wd3_fresh_topology_pose_carry_v1",
    "populate_wd3_fresh_topology_pose_carry_equation",
]
