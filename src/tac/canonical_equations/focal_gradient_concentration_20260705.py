# SPDX-License-Identifier: MIT
"""Canonical equation: FOCAL-GAMMA gradient-budget concentration (council levelset-loss-geometry
symposium 2026-07-05 §equations; MEASURED calibration anchor from the ep50 probe).

Shannon's concentration law for the focal reweight ``(1-p_y)^gamma`` on the base per-pixel seg
loss (stop-grad, mean-1 renormalized — the ``--seg-focal-gamma`` build semantics):

    share_R(gamma) = sum_{p in R} (1-p_y)^gamma / sum_{all p} (1-p_y)^gamma

plus Rudin's interpretability readback (EXACT under stop-grad reweighting; a full-differentiable
focal loss would NOT satisfy it):

    weight(p1) / weight(p2) = ((1-p1) / (1-p2))^gamma      (p=0.5 vs p=0.9  =>  5^gamma)

MEASURED REFINEMENT (the calibration anchor, ep50 of the 2026-07-05 nucleation-fix run,
12-pair gt_n24 subset, witness-alone deploy surface, [macOS-CPU/MLX advisory] NON-PROMOTABLE):
the symposium's "share_isl monotone in gamma" claim holds ONLY when the island pixels are the
uniformly-hardest (lowest-p) tail. On the live checkpoint they are NOT — the hardest tail is
BULK (Road<->Undrivable mid-formation) — so the island weight share is NON-MONOTONE in gamma
(peak near gamma~1: 2.17% pixel share -> 3.23% @gamma=1 -> 2.14% @gamma=3), and the realized
through-R gradient share moves islands by at most +0.13pp (3.12% -> 3.25% @gamma=1) while
monotonically feeding the bulk-BOUNDARY band instead (9.17% -> 10.36% @gamma=3). Focal-gamma is
therefore a (weak) BULK-BOUNDARY reallocator on this vehicle at this stage, NOT an island lever.
Full numbers: ``.omx/research/focal_boundary_calibration_20260705.md``.

means != ends: advisory calibration, NOT a score; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import numpy as np

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "focal_gradient_concentration_v1"

_UTC = "2026-07-05T06:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"
_CALIB_MEMO = ".omx/research/focal_boundary_calibration_20260705.md"
_SYMPOSIUM = ".omx/research/council_grand_symposium_levelset_loss_geometry_20260705.md"

# --- MEASURED ep50 calibration constants (12-pair gt_n24 subset; witness-alone surface) -----------
ISLAND_PX_SHARE_EP50 = 0.0217           # GT classes {1,3} pixel share (== the gamma->0 weight share)
ISLAND_WEIGHT_SHARE_BY_GAMMA_EP50 = {   # analytic share_isl(gamma) — NON-MONOTONE, peak ~gamma=1
    0.5: 0.0312, 1.0: 0.0323, 2.0: 0.0263, 3.0: 0.0214,
}
ISLAND_GRAD_SHARE_EP50 = {              # realized through-R |grad| share on islands
    "current": 0.0312, "g1": 0.0325, "g3": 0.0305,
}
BULK_BOUNDARY_GRAD_SHARE_EP50 = {       # realized through-R |grad| share on the 2px bulk-edge band
    "current": 0.0917, "g1": 0.0975, "g3": 0.1036,
}
ISLAND_RESIDUAL_SHARE_EP50 = 0.0186     # islands' share of d_seg flips (residual) at ep50


def focal_region_share(p_all: np.ndarray, region_mask: np.ndarray, gamma: float) -> float:
    """share_R(gamma) = sum_R (1-p)^gamma / sum_all (1-p)^gamma — the concentration law callable.

    ``p_all`` = realized softmax GT-class probability per pixel; ``region_mask`` boolean same
    shape. gamma -> 0 recovers the region's pixel share (uniform budget)."""
    w = np.power(np.clip(1.0 - np.asarray(p_all, np.float64), 1e-12, 1.0), float(gamma))
    return float(w[np.asarray(region_mask, bool)].sum() / (w.sum() + 1e-300))


def focal_weight_ratio(p1: float, p2: float, gamma: float) -> float:
    """Rudin's readback: the EXACT gradient-weight ratio ((1-p1)/(1-p2))^gamma between two pixels
    under the stop-grad mean-1-renormalized reweight (renorm cancels). (0.5, 0.9) -> 5^gamma."""
    return float(((1.0 - p1) / (1.0 - p2)) ** gamma)


def build_focal_gradient_concentration_v1() -> CanonicalEquation:
    """Build the focal gradient-budget concentration equation with the MEASURED ep50 anchor."""

    anchor = EmpiricalAnchor(
        anchor_id="focal_gamma_ep50_island_share_calibration_20260705",
        measurement_utc=_UTC,
        inputs={
            "checkpoint": "levelset_n600_witness_20260705T015247Z EMA @ep50 (nucleation-fix run)",
            "subset": "12 pairs, gt_n24 stride-2; witness-alone (deploy/verdict) surface",
            "surface_caveat": ("live base CE reads the SEED-COMPOSED frame; seed not persisted in "
                               "checkpoints => composed surface unreconstructable; the wa share is "
                               "an UPPER bound on the composed-surface island share"),
            "loss": "ce + hinge*exp(-margin) GT-margin weight (hinge=4.0, the live config)",
        },
        predicted_output={
            "symposium_claim": "share_isl(gamma) monotone increasing in gamma",
            "gamma_to_zero_share": ISLAND_PX_SHARE_EP50,
        },
        empirical_output={
            "island_weight_share_by_gamma": {str(k): v for k, v in
                                             ISLAND_WEIGHT_SHARE_BY_GAMMA_EP50.items()},
            "island_grad_share_realized": ISLAND_GRAD_SHARE_EP50,
            "bulk_boundary_grad_share_realized": BULK_BOUNDARY_GRAD_SHARE_EP50,
            "island_residual_share": ISLAND_RESIDUAL_SHARE_EP50,
            "verdict": ("NON-MONOTONE island share (peak ~gamma=1; the hardest-p tail is BULK, not "
                        "islands) => focal is a weak BULK-BOUNDARY reallocator here, not an island "
                        "lever; islands already get >= their residual share of gradient "
                        "(3.1% grad vs 1.9% residual) => gamma* = 0 (HOLD) on this trajectory"),
        },
        residual=0.0,
        source_artifact=_CALIB_MEMO,
        measurement_method="probe_focal_gamma_calibration.py grad-share + d_seg decomposition",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_CALIB_MEMO,
            reactivation_criteria=("re-run the probe on a later checkpoint if the pre-registered "
                                   "fire criterion (ep50->100 slope flattening with islands >50% of "
                                   "residual) ever holds; gamma* re-derived from the new shares"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Focal-gamma gradient-budget concentration: share_R(g)=sum_R(1-p)^g/sum(1-p)^g; "
              "weight ratio ((1-p1)/(1-p2))^g (5^g at 0.5/0.9); MEASURED non-monotone island share"),
        one_line_summary=(
            "Focal (1-p)^g reallocates the seg-gradient budget; MEASURED ep50: island share "
            "NON-monotone (peak g~1), bulk-boundary monotone => weak bulk-boundary lever, gamma*=0."
        ),
        latex_form=(
            r"s_R(\gamma)=\frac{\sum_{p\in R}(1-p_y)^\gamma}{\sum_p (1-p_y)^\gamma},\quad"
            r"\frac{w_1}{w_2}=\Big(\frac{1-p_1}{1-p_2}\Big)^\gamma\ \ (5^\gamma\ @\ 0.5/0.9)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.focal_gradient_concentration_20260705:focal_region_share"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": "--seg-focal-gamma (stop-grad mean-1 focal reweight on the base seg loss)",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": ("monotone-in-gamma island concentration holds ONLY when islands are the "
                     "uniformly-hardest tail; measured ep50: they are not (bulk mid-formation) — "
                     "recalibrate per checkpoint before firing"),
        },
        units_in={"p_all": "softmax_gt_prob_0_1", "gamma": "dimensionless"},
        units_out={"share": "fraction_0_1"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            # the symposium's monotone-share prediction vs the measured non-monotone peak (g=3 case):
            # predicted share(3) >= share(1) 0.0323; measured 0.0214 => |residual| = 0.0109.
            "island_share_monotonicity_at_g3": abs(
                ISLAND_WEIGHT_SHARE_BY_GAMMA_EP50[3.0] - ISLAND_WEIGHT_SHARE_BY_GAMMA_EP50[1.0]),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=(
            "experiments/probe_focal_gamma_calibration.py",
            "experiments/train_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="focal_gradient_concentration.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


__all__ = [
    "BULK_BOUNDARY_GRAD_SHARE_EP50",
    "EQUATION_ID",
    "ISLAND_GRAD_SHARE_EP50",
    "ISLAND_PX_SHARE_EP50",
    "ISLAND_RESIDUAL_SHARE_EP50",
    "ISLAND_WEIGHT_SHARE_BY_GAMMA_EP50",
    "build_focal_gradient_concentration_v1",
    "focal_region_share",
    "focal_weight_ratio",
]
