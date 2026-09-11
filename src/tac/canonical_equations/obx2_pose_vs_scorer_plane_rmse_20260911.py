# SPDX-License-Identifier: MIT
"""Canonical equation: OBX2 PoseNet distortion as a function of scorer-plane error.

MEASURED on the complete 600-pair population through the frozen CPU scorers, on
move 44's own retained decoded bytes and three deterministic constructions of
them.  `d_pose` follows a sub-quadratic power law in the scorer-plane RMSE — the
RMSE measured AFTER the scorer's own bilinear resize to 384x512, which is the
plane the scorer actually reads:

    d_pose(r) = d_pose_floor + C * r ** p

with `d_pose_floor` the pointer's own measured `d_pose` at `r = 0`.  Four n600
points constrain it across three decades of `r` and the fit holds within 5%.

Why this equation exists.  The OBX2 burn gate is `distortion < 0.04` at
`<= 122,000 B`.  With the measured `d_seg` of a scorer-plane-matched render
(1.1e-4, contributing 0.011), the Pose budget is `d_pose < 8.4e-5`, and
inverting the law gives an admissible scorer-plane RMSE of **0.238 of one uint8
LSB — 0.093% of full scale**.  No 122,000 B object encodes 1,200 frames of
384x512 at that fidelity, so matching a photometric teacher is the wrong
objective and the object must be POSE-EQUIVALENT rather than photometrically
faithful.  The law is what turns that from an opinion into arithmetic.

The sister derivation `pose_weight_at_operating_point` is exact, not empirical:
it is the contest objective's own derivative `d/dx sqrt(10 x) = 5 / sqrt(10 x)`,
evaluated at a MEASURED operating point.  It replaces a pinned training constant
with a value derived from the live object's own measured `d_pose`.
"""

from __future__ import annotations

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
POWER_LAW_COEFFICIENT = 9.29365e-4
POWER_LAW_EXPONENT = 1.7121

MEASURED_POINTS: tuple[tuple[str, float, float], ...] = (
    ("teacher", 0.0, 4.58687e-6),
    ("sp_384x512", 0.160, 4.52317e-5),
    ("grid_384x512", 4.544, 0.011817),
    ("sp_192x256", 8.670, 0.0391067),
)

# The gate this law is used against (OBX2 burn spec "Exact admission arithmetic").
DISTORTION_GATE = 0.04
SCORER_PLANE_RMSE_FULL_SCALE = 255.0


def predict_d_pose(scorer_plane_rmse: float) -> float:
    """Predicted `d_pose` at a scorer-plane RMSE, in uint8 units."""

    rmse = float(scorer_plane_rmse)
    if rmse < 0.0 or not math.isfinite(rmse):
        raise ValueError("scorer-plane RMSE must be finite and non-negative")
    if rmse == 0.0:
        return POSE_FLOOR
    return POSE_FLOOR + POWER_LAW_COEFFICIENT * rmse**POWER_LAW_EXPONENT


def admissible_scorer_plane_rmse(d_pose_budget: float) -> float:
    """Invert the law: the largest scorer-plane RMSE that stays inside a Pose budget."""

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
            "d_pose = 4.587e-06 + 9.294e-04 * spRMSE^1.712 over three decades (within 5%); "
            "inverting it puts the admissible scorer-plane RMSE at 0.238 of one uint8 LSB."
        ),
        latex_form=r"d_{pose}(r) = d_{pose}^{floor} + C\,r^{p},\quad C = 9.294\times10^{-4},\ p = 1.712",
        python_callable_module_path=(
            "tac.canonical_equations.obx2_pose_vs_scorer_plane_rmse_20260911:predict_d_pose"
        ),
        domain_of_validity={
            "vehicle": ["qbf_coordinate_generator", "move_44_decoded_bytes"],
            "measurement_axis": ["macOS-CPU advisory"],
            "scorer_plane_rmse_range": [0.0, 8.67],
            "note": (
                "fitted on deterministic constructions of ONE video's decoded bytes; the "
                "small-error regime below 0.16 rests on a single point and the render-noise "
                "rungs are queued to constrain it"
            ),
        },
        units_in={"scorer_plane_rmse": "uint8_levels_rms_in_the_384x512_scorer_plane"},
        units_out={"d_pose": "mean_squared_error_over_600x6_posenet_values"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"worst_relative_error_over_four_n600_points": worst_ratio},
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
    "MEASURED_POINTS",
    "POSE_FLOOR",
    "POWER_LAW_COEFFICIENT",
    "POWER_LAW_EXPONENT",
    "admissible_scorer_plane_rmse",
    "build_obx2_pose_vs_scorer_plane_rmse_v1",
    "derive_pose_weight",
    "pose_budget_at_distortion_gate",
    "pose_weight_at_operating_point",
    "predict_d_pose",
]
