# SPDX-License-Identifier: MIT
"""Canonical equation: the detection-evasion (UNIWARD dual) ceiling is ~0 — the
frozen-SegNet Fisher-flat interior carries no realistic d_seg.

MEASURED 2026-07-15 on the cached frozen-SegNet per-pixel margin field (``gt_n96.npz``
margins, bit-exact argmax parity vs the frozen SegNet, 0 mismatches / 96 frames). This
is the DUAL of ``segnet_head_rank4_linear_flipdist_v1`` (projection): where projection
cannot land a uint8-reachable flip, the steganographic move is EVASION — hide the
unavoidable error where the detector is Fisher-flat. This law says that move has no room.

The law
-------
The 2-class Fisher trace surrogate is ``tr g = ½ sech²(m/2)`` (``optimal_metric``), a
monotone-decreasing function of the logit margin ``m`` (curvature↔(−margin) Pearson 0.978).
The minimal ALIGNED render-L2 (0-255 units) to flip a pixel of margin ``m`` is::

    cost(m) = m / ||grad m||,   ||grad m|| ≈ G · sech(m/2)  =>  cost(m) = (m/G)·cosh(m/2)

``G ≈ 0.0606`` DERIVED from the closed-form head anchor (median Road-Lane boundary margin
0.516 flips at first-order input-L2 8.8 over the (384,512,3) frame). ``cost(m)`` grows
~``m·e^{m/2}``: flipping the annulus (m→0) is free (max detectable), flipping the interior
costs exponentially (evasion-safe). CONSEQUENCE (MEASURED, exact from the margin
distribution + sech²): a per-element render-error RMS up to 4 uint8-LSB flips ZERO pixels
of margin ≥ 2.0 (the 4.7%-area annulus threshold) — the Fisher-flat interior is NULL, so
there is nowhere to relocate error that is not already hidden. 100% of modeled d_seg is on
the boundary annulus. Evadable-by-relocation fraction ≈ 0: the evasion dual is pre-exhausted
by the scorer's own margin geometry. The residual is boundary-structural (sharper PROJECTION
+ sub-pixel geometry), plus a realization-irreducible sub-LSB tail d_seg ≈ 8e-4 (Lane-dominated).

Scope / honesty: ``interior_null`` is MEASURED (exact from the cached margin field + the exact
sech² law; does not use the gain model). ``cost(m)`` and the realization floor are DERIVED
(first-order pixel-space pullback of the closed-form head gain). The ONE surviving UNIWARD
variant is CONDITIONAL (suppress boundary-CONCENTRATED Gibbs render error, ceiling = boundary
excess over uniform) and is UNMEASURED (owed: witness error↔margin correlation, n600). Axis
``[macOS-CPU advisory]``; research_only; no score claim; pointer UNMOVED.

Artifact: ``experiments/results/adversarial_evasion_fisher_null_20260715/evasion_probe.json``
Memo: ``.omx/research/adversarial_evasion_fisher_null_20260715.md``
Sister: ``segnet_head_rank4_linear_flipdist_v1`` (projection); ``optimal_metric`` (sech² Fisher).
"""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "evasion_ceiling_fisher_null_interior_v1"
AXIS = "[macOS-CPU advisory] cached frozen-SegNet margin field (bit-exact argmax vs gt_n96)"
SEGNET_WEIGHTS_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
PROBE_ARTIFACT = "experiments/results/adversarial_evasion_fisher_null_20260715/evasion_probe.json"
PROBE_ARTIFACT_SHA256 = "ed0b9a98fc9c66b071cee512206bd7c865f71c1f1817d8d93c4c281826f0458c"

#: DERIVED input->margin gain scale (median Road-Lane boundary anchor + 0.978 relation).
G_GAIN = 0.0606
#: MEASURED annulus threshold margin (4.7%-area percentile of the n96 field).
ANNULUS_THRESHOLD_MARGIN = 2.013
#: MEASURED realization-irreducible d_seg floor (quantization-only, 1-2 sigma), Lane-dominated.
REALIZATION_FLOOR_DSEG = (0.000465, 0.000929)


def flip_cost_render_l2(margin: float, g_gain: float = G_GAIN) -> float:
    """Minimal aligned render-L2 (0-255 units) to flip a pixel of the given logit margin.

    cost(m) = (m/G)·cosh(m/2). Monotone, ~ m·e^{m/2}; the UNIWARD detectability cost.
    """
    m = abs(float(margin))
    if g_gain <= 0:
        raise ValueError("g_gain must be > 0")
    return (m / g_gain) * math.cosh(m / 2.0)


def render_rms_to_flip_interior(margin: float = ANNULUS_THRESHOLD_MARGIN, g_gain: float = G_GAIN) -> float:
    """Per-element render RMS (0-255) needed to flip an interior pixel of the given margin.

    = cost(m) treated as a 1-sigma random-error reach: m / ||grad m|| = (m/G)·cosh(m/2).
    At the 4.7%-area annulus threshold (m≈2.0) this is ~51 LSB — catastrophic — hence the
    interior is null under any realistic render error (interior_leak == 0).
    """
    return flip_cost_render_l2(margin, g_gain)


def build_evasion_ceiling_fisher_null_interior_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=SEGNET_WEIGHTS_SHA256,
        source_path=PROBE_ARTIFACT,
        captured_at_utc="2026-07-16T04:10:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="evasion_interior_null_20260715",
            measurement_utc="2026-07-16T04:10:00Z",
            inputs={
                "field": "gt_n96 margins (96,384,512), bit-exact vs frozen SegNet argmax",
                "fisher_law": "tr g = 0.5*sech^2(m/2) (optimal_metric, exact)",
                "annulus_threshold_margin": ANNULUS_THRESHOLD_MARGIN,
            },
            predicted_output={
                "interior_leak_rate": 0.0,
                "reason": "sech^2(m/2) -> 0 in the interior; no realistic render error flips m>=2",
            },
            empirical_output={
                "interior_leak_rate_at_rho_0p25_to_4p0_LSB": 0.0,
                "modeled_dseg_on_annulus_fraction": 1.0,
                "evadable_by_relocation_fraction": 0.0,
            },
            residual=0.0,
            source_artifact=PROBE_ARTIFACT,
            measurement_method=(
                "swept per-element render RMS 0.25-4.0 LSB against the cached margin field; "
                "counted flips at margin>=2.0 (Fisher-flat interior) -> exactly 0 at every level; "
                "exact from the margin distribution + sech^2 law (independent of the gain model)"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="evasion_realization_floor_lane_20260715",
            measurement_utc="2026-07-16T04:10:00Z",
            inputs={
                "half_lsb_rms": 1.0 / math.sqrt(12.0),
                "gain_model": "||grad m_p|| ~= 0.0606*sech(m_p/2) (DERIVED from closed-form head anchor)",
            },
            predicted_output={"law": "d_seg_floor = P(margin < k*half_lsb_jitter); Lane-dominated"},
            empirical_output={
                "d_seg_floor_1sigma": 0.000465,
                "d_seg_floor_2sigma": 0.000929,
                "lane_flip_rate_1sigma": 0.0166,
                "bulk_class_flip_rate_1sigma": "Road 0.0009 / Undrivable 0.0001",
            },
            residual=0.0,
            source_artifact=PROBE_ARTIFACT,
            measurement_method=(
                "quantization-only irreducible residual: margin field MEASURED, half-LSB jitter "
                "scale DERIVED (first-order pixel-space pullback of the closed-form head gain)"
            ),
            provenance=provenance,
            empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Detection-evasion (UNIWARD dual) ceiling: the Fisher-flat interior is null",
        one_line_summary=(
            "No realistic render error flips an interior (m>=2) pixel, so ~100% of d_seg is on "
            "the boundary annulus and evadable-by-relocation ~0 (evasion pre-exhausted)."
        ),
        latex_form=(
            r"\operatorname{cost}(m)=\frac{m}{G}\cosh(m/2),\quad "
            r"\operatorname{tr}g=\tfrac12\operatorname{sech}^2(m/2)\to0,\quad "
            r"\text{interior\_leak}=0\Rightarrow \text{evadable}\approx0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.evasion_ceiling_fisher_null_20260715:flip_cost_render_l2"
        ),
        domain_of_validity={
            "network": "frozen contest SegNet (smp.Unet tu-efficientnet_b2, classes=5, activation=None)",
            "weights_sha256": SEGNET_WEIGHTS_SHA256,
            "space": "per-pixel logit margin field; interior-null MEASURED exact; cost(m)/floor DERIVED",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": "FORMULATION: UNIWARD-relocation dual on n96 field; conditional Gibbs variant UNMEASURED",
        },
        units_in={"margin": "logit"},
        units_out={"flip_cost": "render_L2_0_255"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"interior_leak_rate": 0.0},
        last_calibration_utc="2026-07-16T04:10:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "msal_uni / UNIWARD lever (#268/Lever-4) EV bound: relocation target is null",
            "boundary projection carriers (analytic lane band, appearance-phase endgame) get the residual",
        ),
        canonical_producers=("tools/adversarial_evasion_fisher_null_probe.py",),
        provenance=provenance,
    )


__all__ = [
    "ANNULUS_THRESHOLD_MARGIN",
    "EQUATION_ID",
    "G_GAIN",
    "REALIZATION_FLOOR_DSEG",
    "build_evasion_ceiling_fisher_null_interior_v1",
    "flip_cost_render_l2",
    "render_rms_to_flip_interior",
]
