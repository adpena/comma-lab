# SPDX-License-Identifier: MIT
"""Canonical admission law for DDM V15 grammar-parametrized RGB templates.

The law is deliberately narrow: it governs an encode-side shared-template
proposal after exact-R differentiation and realized uint8 replay.  It does not
turn a bounded optimizer negative into a family verdict or confer score
authority on macOS CPU evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

EQUATION_ID: Final = "ddm_v15_grammar_parametrized_scorer_solve_admission_v1"
CONTEST_ARCHIVE_NORMALIZER: Final = 37_545_489
RATE_WEIGHT: Final = 25
BREAK_EVEN_SCORE_PER_BYTE: Final = RATE_WEIGHT / CONTEST_ARCHIVE_NORMALIZER
POINTER: Final = "0.1910828242 [contest-CPU]"

N64_RECEIPT = (
    ".omx/research/ddm_v15_scorer_solved_templates_n64_20260723T011500Z/"
    "ddm_v15_scorer_solved_templates_n64_receipt.json"
)
N64_RECEIPT_SHA256 = "be679f30d913ced637001548e3a8e5d44ec992c64489ce3ee44bc1c4a1849639"
N600_RECEIPT = (
    ".omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z/"
    "ddm_v15_scorer_solved_templates_n600_receipt.json"
)
N600_RECEIPT_SHA256 = "5ed6f830b3749a51e0d300a9104fda9a77e86bbeb3b81428a20e1ec0d3dcfcb8"
TEMPLATE_SHA256 = "515ecd3cbcee5c13251b283d0448b851883a3d917157fb754fee183531be7cdc"


@dataclass(frozen=True, slots=True)
class TemplateAdmission:
    admitted: bool
    reason: str
    score_gain_per_byte: float
    break_even_score_per_byte: float


def fisher_trace_from_winner_rival_margin(margin: float) -> float:
    """Return ``tr(F)=1/2 sech^2(m/2)`` with finite scalar custody."""

    value = float(margin)
    if not math.isfinite(value):
        raise ValueError("winner-rival margin must be finite")
    return 0.5 / math.cosh(max(-40.0, min(40.0, value / 2.0))) ** 2


def admit_realized_template_step(
    *,
    target_error_improvement: int,
    harmful_off_target_flips: int,
    score_gain: float,
    delta_archive_bytes: int,
) -> TemplateAdmission:
    """Apply zero-collateral, realized-gain, and reverse-waterfill gates."""

    if isinstance(target_error_improvement, bool) or not isinstance(target_error_improvement, int):
        raise ValueError("target_error_improvement must be an integer")
    if isinstance(harmful_off_target_flips, bool) or not isinstance(harmful_off_target_flips, int):
        raise ValueError("harmful_off_target_flips must be an integer")
    if isinstance(delta_archive_bytes, bool) or not isinstance(delta_archive_bytes, int):
        raise ValueError("delta_archive_bytes must be an integer")
    gain = float(score_gain)
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("score_gain must be finite and nonnegative")
    if target_error_improvement <= 0:
        return TemplateAdmission(False, "NO_REALIZED_TARGET_IMPROVEMENT", 0.0, BREAK_EVEN_SCORE_PER_BYTE)
    if harmful_off_target_flips != 0:
        return TemplateAdmission(False, "HARD_ZERO_COLLATERAL_VIOLATION", 0.0, BREAK_EVEN_SCORE_PER_BYTE)
    if delta_archive_bytes <= 0:
        raise ValueError("a counted template step requires positive delta_archive_bytes")
    marginal = gain / delta_archive_bytes
    if marginal < BREAK_EVEN_SCORE_PER_BYTE:
        return TemplateAdmission(False, "BELOW_RATE_BREAK_EVEN", marginal, BREAK_EVEN_SCORE_PER_BYTE)
    return TemplateAdmission(True, "ADMITTED", marginal, BREAK_EVEN_SCORE_PER_BYTE)


def derivation_edges() -> tuple[tuple[str, str, str], ...]:
    """Return the triality edges that make the admission law auditable."""

    return (
        ("frozen_SegNet_winner_rival_margin", "fisher_trace", "half_sech_squared_margin"),
        ("camera_RGB_template", "exact_R_adjoint", "scorer_margin_gradient"),
        ("continuous_projected_step", "uint8_lattice_projection", "rounded_template_candidate"),
        ("rounded_template_candidate", "realized_secant_replay", "target_error_improvement"),
        ("off_target_baseline_correct_cells", "hard_oracle", "harmful_off_target_flips"),
        ("score_gain_and_exact_bytes", "reverse_waterfill", "template_admission"),
        ("counted_template_bank", "v14_receiver_extension", "deterministic_camera_RGB"),
    )


def measured_v15_anchor() -> dict[str, object]:
    """Return the immutable advisory n600 disposition without promotion authority."""

    return {
        "equation_id": EQUATION_ID,
        "archive_bytes": 133_941,
        "d_seg": 0.027470296224,
        "d_pose": 163.061327281443,
        "movable_conditional_d_seg": 0.291615222639,
        "lane_conditional_d_seg": 0.435195521828,
        "full_p_camera_byte_identity": True,
        "template_sha256": TEMPLATE_SHA256,
        "fork_passed": False,
        "verdict_scope": "FORMULATION x bounded projected-gradient/secant search; family open",
        "named_successor": "#366 joint predictor/template training",
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
    }
