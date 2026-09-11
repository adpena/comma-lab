# SPDX-License-Identifier: MIT
"""Canonical equation: OBX2 PoseNet distortion as a function of scorer-plane error.

MEASURED on the complete 600-pair population through the frozen CPU scorers, on
move 44's own retained decoded bytes and three deterministic constructions of
them.  `d_pose` follows a sub-quadratic power law in the scorer-plane RMSE — the
RMSE measured AFTER the scorer's own bilinear resize to 384x512, which is the
plane the scorer actually reads:

    d_pose(r) = d_pose_floor + C * r ** p

with `d_pose_floor` the pointer's own measured `d_pose` at `r = 0`.  SECOND CORRECTION, 2026-09-11, and the one that matters.  An eighth rung showed
the relation is **NOT A FUNCTION OF SCORER-PLANE RMSE AT ALL**: it is
NON-MONOTONIC.  `sp384_render_noise_4` at RMSE 2.300 measures `d_pose` 0.0120593
while `grid_384x512` at RMSE **4.544** — twice the error — measures 0.011817,
LESS.  What `d_pose` responds to is the error's STRUCTURE, not its magnitude.

Split by structure, each family is a clean power law:

  * SMOOTH (resampling-like: the `sp_*` and `grid_*` constructions) —
    `C = 8.724e-4`, `p = 1.7439`, four points all within 13%.
  * NOISE (independent per-pixel: the `sp384_render_noise_*` rungs) —
    `C = 1.111e-3`, `p = 2.8125`, three points all within 10%.

Two rungs land at essentially the SAME RMSE and settle this without any fit:
`grid_384x512` (smooth, RMSE 4.544) measures `d_pose` 0.011817, while
`sp384_render_noise_8` (noise, RMSE **4.340** — 4.5% LOWER) measures 0.111126.
**Independent noise does 9.40x the Pose damage at matched magnitude.**  For a
successor generator this is a design constraint, not a curiosity: its render
error must be SMOOTH, because high-frequency error is an order of magnitude more
expensive per unit of RMSE.  The single "law" I reported earlier was a mixture of two physics, which
is why its residual grew from 5% to 13% to 29% to 67% as points accumulated.

What this does to the closure.  Both families' FITTED crossings of the Pose
budget sit close together — 0.253 LSB (smooth) and 0.391 LSB (noise) — so the
"near-lossless" reading survives on the fits.  But the MEASURED bracket I
previously quoted, [0.1992, 0.7604], mixed families: its upper end is a NOISE
rung, and a trained generator's error is structured, not white.  The correct
smooth-family bracket is **[0.1992, 4.5440]** — loose, because no smooth rung
was measured between them.  So the closure rests on the smooth fit, not on a
tight measured bracket, and tightening it needs a smooth construction in that
gap.  Stated plainly because I asserted the tight bracket twice.

Why this equation exists.  The OBX2 burn gate is `distortion < 0.04` at
`<= 122,000 B`.  With the measured `d_seg` of a scorer-plane-matched render
(1.1e-4, contributing 0.011), the Pose budget is `d_pose < 8.4e-5`, and
the admissible scorer-plane RMSE is BRACKETED BY MEASUREMENT between **0.1992
(`sp_640x852`, inside the budget) and 0.7604 (`sp384_render_noise_1`, outside
it)**, with the interpolant placing the crossing near 0.245.  Every point in
that bracket is near-lossless — 0.08% to 0.30% of full scale — so the closure's
direction is robust to the fit's 29% wobble even though its exact value is not.
The rungs that pin it are direct: +/-1 LSB of independent noise on the render
alone takes the object to distortion 0.0875, and +/-2 LSB to 0.1581.  No 122,000 B object encodes 1,200 frames of
384x512 at that fidelity, so matching a photometric teacher is the wrong
objective and the object must be POSE-EQUIVALENT rather than photometrically
faithful.  The law is what turns that from an opinion into arithmetic.

The sister derivation `pose_weight_at_operating_point` is exact, not empirical:
it is the contest objective's own derivative `d/dx sqrt(10 x) = 5 / sqrt(10 x)`,
evaluated at a MEASURED operating point.  It replaces a pinned training constant
with a value derived from the live object's own measured `d_pose`.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "obx2_pose_vs_scorer_plane_rmse_v1"

# MEASURED n600 anchors: (scorer-plane RMSE in uint8 units, d_pose).
# r = 0 is move 44's own decoded bytes; the rest are deterministic constructions
# of them scored in the same frozen CPU process.
POSE_FLOOR = 4.58687e-6
# RECALIBRATED twice on 2026-09-11: four points, then six, now SEVEN.
POWER_LAW_COEFFICIENT = 9.33907e-4
POWER_LAW_EXPONENT = 1.7508

MEASURED_POINTS: tuple[tuple[str, float, float], ...] = (
    ("teacher", 0.0, 4.58687e-6),
    ("sp_384x512", 0.160, 4.52317e-5),
    ("sp_640x852", 0.1992, 5.04831e-5),
    ("sp384_render_noise_1", 0.7604, 5.43636e-4),
    ("sp384_render_noise_2", 1.278, 2.02857e-3),
    ("sp384_render_noise_4", 2.300, 1.20593e-2),
    ("sp384_render_noise_8", 4.340, 1.11126e-1),
    ("grid_384x512", 4.544, 0.011817),
    ("sp_192x256", 8.670, 0.0391067),
)

# The local exponent between consecutive measured points, which is what says
# whether one power law describes this at all: 0.555, 1.839, 2.548, 1.391,
# 1.853.  It wanders by 4.6x.  The fit is an INTERPOLANT with 29% worst
# residual, not a law, and the honest primary claim is the measured bracket
# below rather than any single exponent.
# Split by error STRUCTURE, because the pooled relation is non-monotonic.
SMOOTH_COEFFICIENT = 8.72421e-4
SMOOTH_EXPONENT = 1.7439
SMOOTH_WORST_RELATIVE_ERROR = 0.127
NOISE_COEFFICIENT = 1.08333e-3
NOISE_EXPONENT = 3.0663
NOISE_WORST_RELATIVE_ERROR = 0.155
# MODEL-FREE, the cleanest statement the ladder produced: two measured n600 rows
# at essentially the same scorer-plane RMSE, one smooth and one noise.
MATCHED_RMSE_COMPARISON = {
    "smooth_rung": "grid_384x512",
    "smooth_scorer_plane_rmse": 4.544,
    "smooth_d_pose": 0.011817,
    "noise_rung": "sp384_render_noise_8",
    "noise_scorer_plane_rmse": 4.340,
    "noise_d_pose": 0.111126,
    "noise_penalty": 9.40,
    "note": "the noise rung has 4.5% LOWER RMSE and 9.40x the Pose damage",
}
NOISE_PENALTY_AT_EQUAL_RMSE = 9.40
SMOOTH_RUNGS = ("sp_384x512", "sp_640x852", "grid_384x512", "sp_192x256")
NOISE_RUNGS = (
    "sp384_render_noise_1",
    "sp384_render_noise_2",
    "sp384_render_noise_4",
    "sp384_render_noise_8",
)
WORST_RELATIVE_ERROR = 0.667  # the POOLED fit on eight points, retained to show why pooling fails

# The gate this law is used against (OBX2 burn spec "Exact admission arithmetic").
DISTORTION_GATE = 0.04
SCORER_PLANE_RMSE_FULL_SCALE = 255.0


def predict_d_pose_by_structure(scorer_plane_rmse: float, structure: str) -> float:
    """Predicted `d_pose` for an error of a named STRUCTURE at a given RMSE.

    Use this, not the pooled fit: the pooled relation is non-monotonic because
    independent noise and smooth resampling error of the same magnitude do
    different amounts of damage.
    """

    rmse = float(scorer_plane_rmse)
    if rmse < 0.0 or not math.isfinite(rmse):
        raise ValueError("scorer-plane RMSE must be finite and non-negative")
    if structure == "smooth":
        coefficient, exponent = SMOOTH_COEFFICIENT, SMOOTH_EXPONENT
    elif structure == "noise":
        coefficient, exponent = NOISE_COEFFICIENT, NOISE_EXPONENT
    else:
        raise ValueError("structure must be 'smooth' or 'noise'")
    if rmse == 0.0:
        return POSE_FLOOR
    return POSE_FLOOR + coefficient * rmse**exponent


def is_monotonic_in_rmse() -> bool:
    """Whether `d_pose` rises with RMSE across all measured rungs.  It does not."""

    ordered = sorted(MEASURED_POINTS, key=lambda row: row[1])
    return all(a[2] <= b[2] for a, b in itertools.pairwise(ordered))


def predict_d_pose(scorer_plane_rmse: float) -> float:
    """POOLED interpolant, retained for continuity.  Prefer the structure-aware form.

    The pooled relation mixes two physics and is non-monotonic across them; its
    worst residual is 29% against 13% and 9% for the two families fitted apart.
    """

    rmse = float(scorer_plane_rmse)
    if rmse < 0.0 or not math.isfinite(rmse):
        raise ValueError("scorer-plane RMSE must be finite and non-negative")
    if rmse == 0.0:
        return POSE_FLOOR
    return POSE_FLOOR + POWER_LAW_COEFFICIENT * rmse**POWER_LAW_EXPONENT


def measured_bracket(d_pose_budget: float) -> tuple[float | None, float | None]:
    """The measured rungs that bracket a Pose budget: (highest inside, lowest outside).

    This is the honest primary claim.  It needs no model: it is two measured
    n600 rows and the statement that the crossing lies between them.
    """

    inside = [rmse for _, rmse, d_pose in MEASURED_POINTS if d_pose < d_pose_budget]
    outside = [rmse for _, rmse, d_pose in MEASURED_POINTS if d_pose >= d_pose_budget]
    return (max(inside) if inside else None, min(outside) if outside else None)


def admissible_scorer_plane_rmse(d_pose_budget: float) -> float:
    """INTERPOLATE the fit to a Pose budget.

    The fit carries a 29% worst residual and its local exponent varies 4.6x, so
    this is an interpolation between measured rungs and not a law's prediction.
    Prefer `measured_bracket` for any claim that has to hold.
    """

    budget = float(d_pose_budget)
    if budget <= POSE_FLOOR:
        return 0.0
    return ((budget - POSE_FLOOR) / POWER_LAW_COEFFICIENT) ** (1.0 / POWER_LAW_EXPONENT)


def pose_budget_at_distortion_gate(d_seg: float, gate: float = DISTORTION_GATE) -> float:
    """Pose budget left by a measured `d_seg` under `100*d_seg + sqrt(10*d_pose) < gate`."""

    remaining = float(gate) - 100.0 * float(d_seg)
    if remaining <= 0.0:
        return 0.0
    return remaining * remaining / 10.0


def pose_weight_at_operating_point(d_pose: float) -> float:
    """EXACT derivative of the contest Pose term at a MEASURED operating point.

    The contest objective contributes `sqrt(10 * d_pose)`.  A training loss that
    is linear in the Pose MSE must therefore weight it by
    `d/d(d_pose) sqrt(10*d_pose) = 5 / sqrt(10*d_pose)`, evaluated where the
    object actually sits.  This is arithmetic, not a fit: the only empirical
    input is the operating point itself, which must come from a measured row on
    the LIVE object rather than from an assumed constant.
    """

    operating_point = float(d_pose)
    if not math.isfinite(operating_point) or operating_point <= 0.0:
        raise ValueError("pose operating point must be a positive finite d_pose")
    return 5.0 / math.sqrt(10.0 * operating_point)


def derive_pose_weight(
    *,
    d_pose: float,
    d_seg: float,
    source_artifact: str,
    source_sha256: str,
    receiver: str,
    pair_denominator: int,
) -> dict[str, Any]:
    """Pose weight plus every input pinned, so the derivation is auditable later."""

    if pair_denominator != 600:
        raise ValueError("the OBX2 pose operating point is an n600 measurement")
    if receiver not in ("torch", "numpy"):
        raise ValueError("receiver must name the reference that decoded the scored bytes")
    weight = pose_weight_at_operating_point(d_pose)
    return {
        "law_ref": EQUATION_ID,
        "derivation": "5 / sqrt(10 * d_pose) — the exact derivative of sqrt(10*d_pose)",
        "pose_weight": weight,
        "operating_point_d_pose": float(d_pose),
        "operating_point_d_seg": float(d_seg),
        "operating_point_distortion": 100.0 * float(d_seg) + math.sqrt(10.0 * float(d_pose)),
        "pose_budget_at_gate": pose_budget_at_distortion_gate(float(d_seg)),
        "admissible_scorer_plane_rmse": admissible_scorer_plane_rmse(
            pose_budget_at_distortion_gate(float(d_seg))
        ),
        "source_artifact": source_artifact,
        "source_sha256": source_sha256,
        "receiver": receiver,
        "pair_denominator": pair_denominator,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }


def _anchor(name: str, rmse: float, d_pose: float) -> EmpiricalAnchor:
    report = ".omx/research/ddm_obx2_edge_local_implicit_correction_20260911.md"
    return EmpiricalAnchor(
        anchor_id=f"obx2_stage2a_{name}_n600_20260911",
        measurement_utc="2026-09-11T00:00:00Z",
        inputs={"rung": name, "scorer_plane_rmse": rmse, "n_pairs": 600},
        predicted_output={"d_pose": predict_d_pose(rmse)},
        empirical_output={"d_pose": d_pose},
        residual=abs(predict_d_pose(rmse) - d_pose),
        source_artifact=report,
        measurement_method="n600_frozen_cpu_scorers_on_parsed_or_retained_decoded_bytes",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=report,
            reactivation_criteria=(
                "re-fit when a rung below scorer-plane RMSE 0.16 lands; the small-error regime "
                "is the one the gate lives in and is currently constrained by a single point"
            ),
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )


def build_obx2_pose_vs_scorer_plane_rmse_v1() -> CanonicalEquation:
    """Build the OBX2 Pose-versus-scorer-plane-error canonical equation."""

    anchors = tuple(_anchor(name, rmse, d_pose) for name, rmse, d_pose in MEASURED_POINTS)
    worst_ratio = max(
        abs(predict_d_pose(rmse) / d_pose - 1.0) for _, rmse, d_pose in MEASURED_POINTS if d_pose > 0
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="OBX2 PoseNet distortion versus scorer-plane RMSE",
        one_line_summary=(
            "d_pose is NON-MONOTONIC in scorer-plane RMSE: at matched RMSE (4.340 vs 4.544) "
            "independent noise does 9.40x the Pose damage of smooth resampling error."
        ),
        latex_form=r"d_{pose}(r) \approx d_{pose}^{floor} + C\,r^{p},\quad C = 9.339\times10^{-4},\ p = 1.751",
        python_callable_module_path=(
            "tac.canonical_equations.obx2_pose_vs_scorer_plane_rmse_20260911:predict_d_pose"
        ),
        domain_of_validity={
            "vehicle": ["qbf_coordinate_generator", "move_44_decoded_bytes"],
            "measurement_axis": ["macOS-CPU advisory"],
            "scorer_plane_rmse_range": [0.0, 8.67],
            "monotonic_in_rmse": is_monotonic_in_rmse(),
            "smooth_family": {
                "coefficient": SMOOTH_COEFFICIENT,
                "exponent": SMOOTH_EXPONENT,
                "worst_relative_error": SMOOTH_WORST_RELATIVE_ERROR,
                "rungs": list(SMOOTH_RUNGS),
            },
            "noise_family": {
                "coefficient": NOISE_COEFFICIENT,
                "exponent": NOISE_EXPONENT,
                "worst_relative_error": NOISE_WORST_RELATIVE_ERROR,
                "rungs": list(NOISE_RUNGS),
            },
            "noise_penalty_at_equal_rmse": NOISE_PENALTY_AT_EQUAL_RMSE,
            "matched_rmse_comparison": MATCHED_RMSE_COMPARISON,
            "pooled_worst_relative_error": WORST_RELATIVE_ERROR,
            "status": (
                "the pooled relation is NON-MONOTONIC in RMSE; fit and quote by error structure"
            ),
            "note": (
                "fitted on deterministic constructions of ONE video's decoded bytes; a single "
                "power law summarizes them to 29% and no better, so claims should rest on the "
                "measured bracket rather than on the fitted crossing"
            ),
        },
        units_in={"scorer_plane_rmse": "uint8_levels_rms_in_the_384x512_scorer_plane"},
        units_out={"d_pose": "mean_squared_error_over_600x6_posenet_values"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            "pooled_worst_relative_error": worst_ratio,
            "smooth_family_worst_relative_error": SMOOTH_WORST_RELATIVE_ERROR,
            "noise_family_worst_relative_error": NOISE_WORST_RELATIVE_ERROR,
        },
        last_calibration_utc="2026-09-11T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments/ddm_obx2_trainer.py",
            "experiments/ddm_obx2_edge_local_implicit_correction.py",
        ),
        canonical_producers=("experiments/ddm_obx2_edge_local_implicit_correction.py",),
        provenance=build_provenance_for_predicted(
            model_id="obx2_pose_vs_scorer_plane_rmse.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "DISTORTION_GATE",
    "EQUATION_ID",
    "MATCHED_RMSE_COMPARISON",
    "MEASURED_POINTS",
    "POSE_FLOOR",
    "POWER_LAW_COEFFICIENT",
    "POWER_LAW_EXPONENT",
    "WORST_RELATIVE_ERROR",
    "admissible_scorer_plane_rmse",
    "build_obx2_pose_vs_scorer_plane_rmse_v1",
    "derive_pose_weight",
    "is_monotonic_in_rmse",
    "measured_bracket",
    "pose_budget_at_distortion_gate",
    "pose_weight_at_operating_point",
    "predict_d_pose",
    "predict_d_pose_by_structure",
]
