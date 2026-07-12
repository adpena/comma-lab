# SPDX-License-Identifier: MIT
"""Canonical equation: ELM/INR closed-form affine SDF-head seed for #341.

This is a design/source-inspection law, not an empirical score anchor.  With the witness trunk
frozen, the SDF head is affine in its hidden features and its label-smoothed CE-logit target can
be fit by streaming weighted ridge normal equations.  Partitioned local fits are deployable in
the current decoder only after a measured global-affine fold; the projection residual prevents
the POU field from being silently claimed as shipped behavior.
"""

from __future__ import annotations

import hashlib

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_predicted

EQUATION_ID = "elm_inr_affine_sdf_head_seed_v1"
_LAW = (
    "A_s=sum_i w_si x_i x_i^T+lambda*diag(1,...,1,0); "
    "B_s=sum_i w_si x_i y_i^T; beta_s=pinv(A_s)B_s; "
    "y_i=T*(log(q_i)-mean(log(q_i))); fold local POU field to one global affine head"
)
_LAW_SHA256 = hashlib.sha256(_LAW.encode("utf-8")).hexdigest()


def build_elm_inr_affine_sdf_head_seed_v1() -> CanonicalEquation:
    """Build the non-promotable closed-form seed law with no invented empirical anchor."""

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="ELM/INR streaming closed-form affine SDF-head seed with explicit POU fold",
        one_line_summary=(
            "Freeze witness features, solve the affine SDF head by streaming ridge LS, then "
            "measure the POU-to-global projection residual before #341 Gauss-Newton polish."
        ),
        latex_form=(
            r"x_i=[h_i;1],\quad A_s=\sum_i w_{si}x_ix_i^\top+\lambda\,"
            r"\mathrm{diag}(1,\ldots,1,0),\quad B_s=\sum_iw_{si}x_iy_i^\top,\quad"
            r"\beta_s=A_s^+B_s,\quad y_i=T(\log q_i-\overline{\log q_i})"
        ),
        python_callable_module_path=(
            "tac.boundary_math.elm_inr_head_solve:solve_partitioned_affine_head_with_fold"
        ),
        domain_of_validity={
            "vehicle": "v7.5/v9 level-set witness with frozen trunk/features and affine out_sdf",
            "solved_parameters": ["out_sdf.weight", "out_sdf.bias"],
            "frozen_parameters": [
                "coordinate features",
                "trunk",
                "pair code",
                "FiLM",
                "out_tex",
                "palette",
            ],
            "target": "finite centered label-smoothed categorical log-probabilities; not exact argmax",
            "pou_custody": (
                "local subdomain field is not present in the current decoder; it must be projected "
                "to one global affine head; direct-target, local-target, folded-target, and "
                "fold-to-local RMSE must all be reported"
            ),
            "receiver_optimum": (
                "the unregularized direct-global solve is the target-SSE optimum for the current "
                "single affine receiver; a POU field may fit locally better but its global fold "
                "cannot beat that direct target-SSE optimum"
            ),
            "canonical_pair_scope": "full P=600; a smaller pair-limit is diagnostic_slice=true only",
            "terminal_consumer": "#341 exact through-R head Gauss-Newton finisher",
            "settled_predecessor": (
                "tac.boundary_math.lever_b_levelset_generator.fit_out_sdf_to_structured_target; "
                "this law adds streaming/resume plus POU/fold custody"
            ),
            "score_authority": "none; upstream/evaluate.py on exact archive bytes remains owed",
            "verdict_scope": "build/source law; no wall-clock win or d_seg movement asserted",
        },
        units_in={
            "hidden_features": "dimensionless fp32",
            "smoothed_targets": "logit",
            "ridge": "normal_equation_diagonal",
            "pou_weights": "unitless partition weights",
        },
        units_out={
            "out_sdf.weight": "logit per hidden-feature unit",
            "out_sdf.bias": "logit",
            "fit_rmse": "logit",
            "projection_rmse": "logit",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-12T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.elm_inr_head_seed",
            "tools.quadratic_basin_finisher_probe",
        ),
        canonical_producers=("tac.boundary_math.elm_inr_head_solve",),
        provenance=build_provenance_for_predicted(
            model_id=EQUATION_ID,
            inputs_sha256=_LAW_SHA256,
            measurement_axis="[predicted build law]",
            hardware_substrate="numpy-portable",
        ),
    )


def populate_elm_inr_affine_sdf_head_seed_equation(
    *,
    path=None,
    lock_path=None,
    agent=None,
    subagent_id=None,
) -> CanonicalEquation:
    """Idempotently append the equation through the canonical locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_elm_inr_affine_sdf_head_seed_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="elm_inr_affine_head_seed_20260712; design/source law, empirical n600 anchor owed",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_elm_inr_affine_sdf_head_seed_v1",
    "populate_elm_inr_affine_sdf_head_seed_equation",
]
