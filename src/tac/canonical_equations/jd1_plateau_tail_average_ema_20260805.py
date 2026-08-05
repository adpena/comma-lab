# SPDX-License-Identifier: MIT
"""Canonical source-inspection law for JD1 plateau-tail EMA live weights.

Axis: apparatus / trainer law.  This file makes the dy2 tail-average update
machine-addressable after the dy1 scope-law resolver merge; it is not a score
claim and does not imply a scorer or archive row.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "jd1_plateau_tail_average_ema_v1"
CAPTURED_AT_UTC = "2026-08-05T22:16:26Z"
DY2_RECEIPT = ".omx/research/ddm_dy2_20260805/RECEIPT.md"


def plateau_tail_live_weight(updates_since_anchor: int) -> float:
    """Live-sample weight after the plateau anchor sample is already installed."""
    k = int(updates_since_anchor)
    if k < 0:
        raise ValueError("updates_since_anchor must be >= 0")
    return 1.0 / float(k + 2)


def build_equation() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DY2_RECEIPT,
        reactivation_criteria=(
            "operator may recalibrate only if the JD1 tail-average update equation changes; "
            "score promotion still requires a byte-closed evaluated archive"
        ),
        measurement_axis="[source-inspection apparatus law; no scorer]",
        hardware_substrate="source_reader",
        captured_at_utc=CAPTURED_AT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="dy2_source_inspection_plateau_tail_weight_20260805",
        measurement_utc=CAPTURED_AT_UTC,
        inputs={"updates_since_anchor": [0, 1, 8], "anchor_sample_index": 0},
        predicted_output={"live_weights": [0.5, 1.0 / 3.0, 0.1]},
        empirical_output={
            "trainer_helper": "jd1_ema_tail_average_live_weight",
            "live_weights": [0.5, 1.0 / 3.0, 0.1],
        },
        residual=0.0,
        source_artifact=DY2_RECEIPT,
        measurement_method="source_inspection_dy2_tail_average_update_law",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="JD1 plateau-tail EMA live-sample weight",
        one_line_summary=(
            "After a JD1 plateau anchor, the next live iterate enters the EMA shadow "
            "with weight 1/(updates_since_anchor+2)."
        ),
        latex_form=(
            r"\theta^{EMA}_{n+1}=\theta^{EMA}_{n}+\frac{1}{n+2}"
            r"(\theta^{live}_{n}-\theta^{EMA}_{n})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.jd1_plateau_tail_average_ema_20260805:"
            "plateau_tail_live_weight"
        ),
        domain_of_validity={
            "included": [
                "JD1 plateau_tail_average EMA mode after explicit anchor",
                "scope-law tier T3_LIVE_ADAPTED",
                "shipping EMA shadow update law only",
            ],
            "excluded": [
                "pre-anchor geometric EMA",
                "plateau classifier selection",
                "score, archive, or frontier claims",
            ],
        },
        units_in={"updates_since_anchor": "count_of_live_updates_after_anchor"},
        units_out={"live_weight": "unitless_convex_weight"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "source_inspection_dy2_tail_average_update_law": 0.0,
        },
        last_calibration_utc=CAPTURED_AT_UTC,
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "tac.witness_dsl.scope_laws._eval_jd1_plateau_tail_average_ema",
            "experiments.train_tr1_partition_renderer_mlx.jd1_ema_tail_average_live_weight",
        ),
        canonical_producers=(
            "tools.register_jd1_plateau_tail_average_ema_20260805",
        ),
        provenance=provenance,
    )


__all__ = ["EQUATION_ID", "build_equation", "plateau_tail_live_weight"]
