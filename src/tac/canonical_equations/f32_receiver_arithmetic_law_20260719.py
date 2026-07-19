# SPDX-License-Identifier: MIT
"""Canonical f32 receiver-arithmetic law (Task #540, operator GO step-3 2026-07-19).

THE LAW: exact rational scorer-plane equality `A(z) = y` is NECESSARY but NOT
SUFFICIENT for an exactness (d_seg = 0) claim — the frozen SegNet's argmax is
evaluated in NATIVE float32 CPU-Torch arithmetic, whose accumulation order and
tie behavior differ from generic float64 at the ULP scale. A cell whose
winner/rival logit margin lies at or below the measured ULP class can flip
under the declared arithmetic even when the plane equality is exact-rational.

Consequences (BINDING on every receiver / exactness verdict):
1. Every receiver/receipt DECLARES its arithmetic: native float32 CPU-Torch
   conv/eval semantics + native argmax tie policy.
2. Every exactness verdict is measured through the DECLARED arithmetic (the
   hard oracle), never inferred from rational equality or float64 replay.
3. Cells with margin at/below the measured ULP class are ARITHMETIC-SENSITIVE:
   an exactness claim over them requires either a hard-oracle pass or an
   explicit deterministic tie-breaking rule baked into the decode.

MEASURED anchors (both single-cell, instance scope — the class floor, not a
rate): frame-195 (power-diagram arm, 2026-07-18) margin 4.76837158203125e-7
with float64 preferring the OTHER class; pair-125 (joint-solve arm, 2026-07-19)
ONE native-f32 argmax mismatch (d_seg 5.086263020833333e-6) despite exact
rational A-equality and zero cached/native winner disagreement on the source
control. NO score/promotion claim; the law constrains VERDICT ADMISSIBILITY.
"""

from __future__ import annotations

from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "f32_receiver_arithmetic_exactness_admissibility_v1"
SOURCE_MEMO = ".omx/research/joint_seg_pose_inverse_solve_20260719_codex.md"
SISTER_MEMO = ".omx/research/v10_power_diagram_byteclose_findings_20260718.md"
_UTC = "2026-07-19T04:30:00Z"

# MEASURED single-cell ULP-class margins (the two receipts named above).
FRAME_195_MARGIN = 4.76837158203125e-7
PAIR_125_DSEG = 5.086263020833333e-6


def exactness_claim_admissibility(
    min_winner_rival_margin: float,
    *,
    hard_oracle_verified: bool,
    declared_arithmetic: str = "native_float32_cpu_torch",
    ulp_class_margin: float = FRAME_195_MARGIN,
    ulp_safety_factor: float = 10.0,
) -> dict[str, Any]:
    """The law's evaluator: is a d_seg-exactness claim ADMISSIBLE?

    A claim is admissible iff (a) the frozen-scorer hard oracle verified it
    under the declared native-f32 arithmetic, OR (b) every winner/rival margin
    clears the measured ULP class by ``ulp_safety_factor`` (then rational
    plane-equality implies argmax equality with margin to spare). Anything
    else is INADMISSIBLE_ARITHMETIC_SENSITIVE — not a wrong claim, an
    unproven one (honest non-knowledge, never rounded to exact).
    """

    if min_winner_rival_margin < 0 or ulp_class_margin <= 0 or ulp_safety_factor < 1:
        raise ValueError(
            "margins must be >=0, ulp_class_margin>0, ulp_safety_factor>=1"
        )
    margin_safe = min_winner_rival_margin > ulp_safety_factor * ulp_class_margin
    admissible = bool(hard_oracle_verified or margin_safe)
    return {
        "admissible": admissible,
        "basis": (
            "hard_oracle" if hard_oracle_verified
            else ("margin_clears_ulp_class" if margin_safe
                  else "INADMISSIBLE_ARITHMETIC_SENSITIVE")
        ),
        "declared_arithmetic": declared_arithmetic,
        "min_winner_rival_margin": float(min_winner_rival_margin),
        "ulp_class_margin": float(ulp_class_margin),
        "ulp_safety_factor": float(ulp_safety_factor),
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_f32_receiver_arithmetic_law_v1() -> CanonicalEquation:
    """Build the law with its two measured single-cell anchors."""

    def _prov(path: str, utc: str):
        return build_provenance_for_research_sidecar(
            sidecar_path=path,
            reactivation_criteria=(
                "a receiver that decodes deterministically WITHOUT hard-oracle "
                "access (contest inflate) must bake explicit tie-breaking rules; "
                "any new measured ULP-class instance appends an anchor and "
                "recalibrates ulp_class_margin (max over instances)."
            ),
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64",
            captured_at_utc=utc,
        )

    anchor_frame195 = EmpiricalAnchor(
        anchor_id="frame195_native_f32_tie_class_20260718",
        measurement_utc="2026-07-18T00:00:00Z",
        inputs={
            "context": "power-diagram byteclose arm, frame 195 class-0/class-1 cell",
            "arithmetic_compared": "native f32 CPU-Torch vs generic float64",
        },
        predicted_output="class 1 (float64 margin 2.5277826765e-7 for class 1)",
        empirical_output=f"class 0 under native f32 (tie; margin {FRAME_195_MARGIN})",
        residual=FRAME_195_MARGIN,
        source_artifact=".omx/research/v10_power_diagram_frame195_diagnostic_20260718.json",
        measurement_method=(
            "paired forward through frozen SegNet head under both arithmetics on "
            "the identical feature input; margin read from logits"
        ),
        provenance=_prov(SISTER_MEMO, "2026-07-18T00:00:00Z"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    anchor_pair125 = EmpiricalAnchor(
        anchor_id="pair125_exact_rational_still_flips_f32_20260719",
        measurement_utc=_UTC,
        inputs={
            "context": "joint seg+pose inverse solve, pair 125, exact rational A-equality",
            "d_pose_same_frames": 3.917505209116712e-11,
            "source_control": "exactly zero on both metrics; cached/native winner disagreement zero",
        },
        predicted_output="d_seg 0.0 (rational plane equality)",
        empirical_output=f"d_seg {PAIR_125_DSEG} (ONE native-f32 argmax cell)",
        residual=PAIR_125_DSEG,
        source_artifact=(
            ".omx/research/joint_seg_pose_inverse_solve_pair125_diagnostic2_stages/"
            "pair_0125.hard_oracle_refusal.json"
        ),
        measurement_method=(
            "hard-oracle native-f32 CPU-Torch SegNet argmax on the exactly-solved "
            "uint8 frames vs cached L*; fail-closed refusal receipt"
        ),
        provenance=_prov(SOURCE_MEMO, _UTC),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="f32 receiver-arithmetic exactness admissibility",
        one_line_summary=(
            "Rational plane equality does not imply native-f32 argmax equality at "
            "ULP-class margins; exactness verdicts need the f32 hard oracle or a "
            "margin clearing the measured ULP class."
        ),
        latex_form=(
            r"\mathrm{ADMISSIBLE}(d_{\mathrm{seg}}{=}0)\;\Longleftrightarrow\;"
            r"\mathrm{oracle}_{f32}(z)=L^{*}\;\lor\;"
            r"\min_{jk}m_{jk}>\kappa\,\mu_{\mathrm{ULP}},\quad"
            r"\mu_{\mathrm{ULP}}=\max(4.768\times10^{-7},\ldots)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.f32_receiver_arithmetic_law_20260719:"
            "exactness_claim_admissibility"
        ),
        domain_of_validity={
            "scorer": "frozen SegNet argmax, native float32 CPU-Torch semantics",
            "claim_type": "d_seg exactness / bit-identity verdicts",
            "instances_measured": ["frame_195", "pair_125"],
            "verdict_scope": (
                "instance-anchored law; the ULP class floor recalibrates as the "
                "max over measured instances"
            ),
        },
        units_in={
            "min_winner_rival_margin": "logit units (frozen head)",
            "ulp_class_margin": "logit units",
        },
        units_out={"admissible": "bool", "basis": "categorical"},
        empirical_anchors=(anchor_frame195, anchor_pair125),
        predicted_vs_empirical_residual={
            "frame195_margin": FRAME_195_MARGIN,
            "pair125_dseg": PAIR_125_DSEG,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/measure_joint_seg_pose_rate.py",
            "src/tac/witness_dsl/v10_compiler_receiver.py (production successor, #543)",
            "tools/measure_uint8_lattice_feasibility.py",
        ),
        canonical_producers=(
            ".omx/research/v10_power_diagram_frame195_diagnostic_20260718.json",
            ".omx/research/joint_seg_pose_inverse_solve_receipt_n24_20260719.json",
        ),
        provenance=_prov(SOURCE_MEMO, _UTC),
    )
