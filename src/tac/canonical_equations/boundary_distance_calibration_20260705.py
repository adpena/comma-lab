# SPDX-License-Identifier: MIT
"""Canonical equation: BOUNDARY-DISTANCE weight calibration law (CE-window intervention
pre-stage 2026-07-05; MEASURED ep100 calibration anchor from the bd probe).

The ``--boundary-distance-weight`` (Kervadec SDF-native placement) term enters the loss
additively, so its share of the (CE + w*BD) total is the EXACT hyperbolic mixing law

    ratio(w) = w * M_bd / (M_ce + w * M_bd)        (M = the measured term magnitudes)

which inverts in closed form to the weight for a target share r:

    w(r) = r * M_ce / ((1 - r) * M_bd)

MEASURED (ep100 EMA of the 2026-07-05 nucleation-fix run, 12-pair gt_n24 subset,
witness-alone deploy surface, [macOS-CPU/MLX advisory] NON-PROMOTABLE):
M_ce = 35.614 (w_seg=100 CE*hinge term), M_bd = 17.723 (raw band-mean |phi gap|), so the
operator 5-15%-of-total window maps to w in [0.106, 0.355]; w=0.2 lands at 9.05% (measured
directly, not interpolated). On the PHI (SDF-field) surface — the ONLY surface common to both
terms (the bd term has ZERO gradient w.r.t. the rendered frame) — the realized |grad| shares
move MONOTONICALLY with w: bulk-boundary 5.65% -> 6.14% (w=0.2) -> 8.45% (w=1.0), island
1.23% -> 1.38% -> 2.08% (RISES — lane edges are inter-class edges inside the band; no island
collapse), interior 93.1% -> 89.5%. Sister diagnostic measured on the same probe: LIVE
(resume-sidecar) weights score CE 0.928x the EMA's on the witness-alone surface — the ep92+
spike-guard deadlock is NOT a seg-surface divergence of the live weights.
Full numbers: ``.omx/research/ce_window_intervention_package_20260705.md``.

means != ends: advisory calibration, NOT a score; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "boundary_distance_weight_calibration_v1"

_UTC = "2026-07-05T08:30:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"
_PACKAGE_MEMO = ".omx/research/ce_window_intervention_package_20260705.md"
_LOSS_SYMPOSIUM = ".omx/research/council_grand_symposium_levelset_loss_geometry_20260705.md"

# --- MEASURED ep100 calibration constants (12-pair gt_n24 subset; witness-alone surface) ----------
CE_TERM_MEAN_EP100 = 35.614      # w_seg=100 * mean(ce * hinge_w), hinge=4.0 (live config)
BD_RAW_MEAN_EP100 = 17.723       # trainer boundary_distance_term_mlx raw value (band-mean |gap|)
CE_TERM_LIVE_MEAN_EP100 = 33.046  # SAME term with the LIVE (resume-sidecar) weights: 0.928x EMA
BULK_BOUNDARY_GRAD_SHARE_BY_W_EP100 = {  # phi-surface realized |grad| share (monotone in w)
    0.0: 0.0565, 0.01: 0.0566, 0.05: 0.0575, 0.1: 0.0587, 0.2: 0.0614, 0.5: 0.0699, 1.0: 0.0845,
}
ISLAND_GRAD_SHARE_BY_W_EP100 = {  # rises with w (band includes lane inter-class edges): no collapse
    0.0: 0.0123, 0.01: 0.0123, 0.05: 0.0126, 0.1: 0.0130, 0.2: 0.0138, 0.5: 0.0164, 1.0: 0.0208,
}


def bd_share_of_total(w: float, m_ce: float = CE_TERM_MEAN_EP100,
                      m_bd: float = BD_RAW_MEAN_EP100) -> float:
    """ratio(w) = w*M_bd / (M_ce + w*M_bd) — the exact additive-term mixing law."""
    tot = m_ce + w * m_bd
    return (w * m_bd / tot) if tot > 0 else 0.0


def bd_weight_for_ratio(r: float, m_ce: float = CE_TERM_MEAN_EP100,
                        m_bd: float = BD_RAW_MEAN_EP100) -> float:
    """Closed-form inverse: the w_bd putting the bd term at share r of (CE + w*BD).
    r=0.05 -> 0.1058, r=0.15 -> 0.3546 at the ep100 anchor (the operator 5-15% window)."""
    if not (0.0 <= r < 1.0):
        raise ValueError(f"target share r must be in [0,1), got {r}")
    return r * m_ce / ((1.0 - r) * m_bd)


def build_boundary_distance_weight_calibration_v1() -> CanonicalEquation:
    """Build the bd-weight calibration equation with the MEASURED ep100 anchor."""

    anchor = EmpiricalAnchor(
        anchor_id="bd_weight_ep100_phi_surface_calibration_20260705",
        measurement_utc=_UTC,
        inputs={
            "checkpoint": ("levelset_n600_witness_20260705T015247Z EMA BEST @ep100 "
                           "(sha256 ab06c6ae...) + resume sidecar (sha256 d09a49b9...)"),
            "subset": "12 pairs, gt_n24 stride-2; witness-alone (deploy/verdict) surface",
            "surface": ("PHI (SDF-field) leaf — the only common surface: bd reads model.sdf "
                        "directly (zero frame-gradient); CE differentiates through "
                        "softmax->palette->sigmoid->R->SegNet from the SAME leaf"),
            "loss": "w_seg=100 * mean(ce * (1+4*exp(-margin))) + w * boundary_distance_term_mlx",
        },
        predicted_output={
            "target_window": "bd term = 5-15% of (CE + w*BD)",
            "w_window_from_inverse_law": [0.1058, 0.3546],
        },
        empirical_output={
            "ce_term_mean": CE_TERM_MEAN_EP100,
            "bd_raw_mean": BD_RAW_MEAN_EP100,
            "ce_term_live_mean": CE_TERM_LIVE_MEAN_EP100,
            "bulk_boundary_grad_share_by_w": {str(k): v for k, v in
                                              BULK_BOUNDARY_GRAD_SHARE_BY_W_EP100.items()},
            "island_grad_share_by_w": {str(k): v for k, v in ISLAND_GRAD_SHARE_BY_W_EP100.items()},
            "verdict": ("w_bd* = 0.2 (measured 9.05% of total, mid-window); shares monotone in w; "
                        "island share RISES (no collapse); live weights NOT seg-damaged "
                        "(live/EMA CE 0.928)"),
        },
        residual=0.0,
        source_artifact=_PACKAGE_MEMO,
        measurement_method=("probe_boundary_distance_calibration.py phi-surface grad shares + "
                            "term magnitudes (chunked, kill-durable; trainer-imported bd functions)"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_PACKAGE_MEMO,
            reactivation_criteria=("re-run the probe on any later checkpoint before firing/retuning "
                                   "w_bd — M_ce falls as training progresses so ratio(w) grows at "
                                   "fixed w; recompute w(r) from the fresh magnitudes"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Boundary-distance weight calibration: ratio(w)=w*M_bd/(M_ce+w*M_bd); "
              "w(r)=r*M_ce/((1-r)*M_bd); MEASURED ep100 M_ce=35.61 M_bd=17.72 => w*=0.2 @9.05%"),
        one_line_summary=(
            "Additive bd term mixes hyperbolically with CE; ep100-measured magnitudes put the "
            "5-15% operator window at w in [0.106,0.355]; w*=0.2; phi-surface shares monotone."
        ),
        latex_form=(
            r"\rho(w)=\frac{w\,M_{bd}}{M_{ce}+w\,M_{bd}},\qquad"
            r"w(\rho)=\frac{\rho\,M_{ce}}{(1-\rho)\,M_{bd}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.boundary_distance_calibration_20260705:bd_weight_for_ratio"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": "--boundary-distance-weight (Kervadec SDF-native band placement, build 535e142be)",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": ("magnitudes are checkpoint-dependent (M_ce falls with training) — the mixing "
                     "law is exact but w(r) must be re-evaluated per checkpoint; grad-share rows "
                     "are phi-surface (NOT comparable to the focal probe's frame-surface shares)"),
        },
        units_in={"w": "loss_weight", "r": "fraction_0_1",
                  "m_ce": "loss_units", "m_bd": "loss_units"},
        units_out={"ratio": "fraction_0_1", "w": "loss_weight"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            # closed-form ratio(0.2)=0.09048 vs the measured-magnitude table row 0.0905 (exact by
            # construction on the means; kept as the self-consistency readback):
            "ratio_law_at_w02": abs(bd_share_of_total(0.2) - 0.0905),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=(
            "experiments/probe_boundary_distance_calibration.py",
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="boundary_distance_weight_calibration.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


__all__ = [
    "BD_RAW_MEAN_EP100",
    "BULK_BOUNDARY_GRAD_SHARE_BY_W_EP100",
    "CE_TERM_LIVE_MEAN_EP100",
    "CE_TERM_MEAN_EP100",
    "EQUATION_ID",
    "ISLAND_GRAD_SHARE_BY_W_EP100",
    "bd_share_of_total",
    "bd_weight_for_ratio",
    "build_boundary_distance_weight_calibration_v1",
]
