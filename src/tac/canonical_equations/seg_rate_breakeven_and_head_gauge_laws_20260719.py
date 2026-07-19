# SPDX-License-Identifier: MIT
"""Two small DERIVED laws from the 2026-07-19 wave (#540 equations-leg debt).

1. ``seg_rate_breakeven_v1`` — the KKT indifference price between seg distortion
   and archive bytes, pure arithmetic on the canonical score law
   (upstream/evaluate.py:92): S = 100*d_seg + sqrt(10*d_pose) + 25*B/37_545_489.
   Holding pose fixed, dS = 100*dd_seg + (25/37_545_489)*dB, so the break-even
   byte budget for a seg reduction dd is B* = 100*37_545_489/25 * dd
   = 150_181_956 * dd (~150.18 B per 1e-6 d_seg). Rows on the measured secant
   cheaper than B* are worth taking; costlier rows are not. Consumers: the #536
   waterfill, the seg-secant arm's admission ranking, and #553's rate-equivalence
   framing (a 200 B saving ~ 1.33e-6 d_seg).

2. ``segnet_head_affine_gauge_quotient_v1`` — the description-complexity gauge
   for the frozen rank-4 max-of-affine head (DERIVED, #550 Nielsen crosswalk):
   the argmax over K affine scores l_i(z)=a_i.z+b_i in d dims equals a weighted-
   site power partition (s_i=a_i/2, w_i=b_i+|s_i|^2); adding one affine function
   u.z+v to ALL scores moves nothing, removing d+1 DOF, so margin-preserving
   description complexity is at most (K-1)(d+1) scalars — 20 for K=5,d=4 — and
   partition-only quotients one more positive scale to 19 DOF. Applied to the
   MEASURED 338-byte PDW1 packet this yields the 158/138/134-byte constructions
   (its 180 tie-locus bytes are deterministic functions of sites/weights/edges).
   Upper CONSTRUCTIONS, not information-theoretic minima; exact over the declared
   channel-quotient arithmetic, silent on the spatial/RGB pullback.
"""

from __future__ import annotations

from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

_UTC = "2026-07-19T10:50:00Z"
_ARCHIVE_DENOM = 37_545_489
_RATE_COEFF = 25.0
_SEG_COEFF = 100.0
BREAKEVEN_BYTES_PER_UNIT_DSEG = _SEG_COEFF * _ARCHIVE_DENOM / _RATE_COEFF  # 150_181_956.0


def seg_rate_breakeven_bytes(delta_d_seg: float) -> dict[str, Any]:
    """Byte budget B* that exactly offsets a seg reduction of ``delta_d_seg``.

    Adding fewer than B* bytes while buying ``delta_d_seg`` lowers S; adding
    more raises it. Pure score-law arithmetic (pose held fixed); advisory —
    never a score claim.
    """
    if delta_d_seg < 0:
        raise ValueError("delta_d_seg must be >= 0 (a reduction magnitude)")
    return {
        "breakeven_bytes": BREAKEVEN_BYTES_PER_UNIT_DSEG * delta_d_seg,
        "bytes_per_1e6_d_seg": BREAKEVEN_BYTES_PER_UNIT_DSEG * 1e-6,
        "score_claim": False,
        "promotion_eligible": False,
    }


def head_gauge_description_dof(num_classes: int, feature_dim: int) -> dict[str, Any]:
    """Margin-preserving and partition-only DOF for a max-of-affine head.

    K affine scores over d dims carry K*(d+1) raw scalars; the shared-affine
    gauge (d+1 DOF) is score-difference-invisible, leaving (K-1)*(d+1)
    margin-preserving DOF; partition-only consumers quotient one more common
    positive scale. Exact in the channel quotient; says nothing about the
    spatial pullback that realizes the cells.
    """
    if num_classes < 2 or feature_dim < 1:
        raise ValueError("need num_classes >= 2 and feature_dim >= 1")
    margin_dof = (num_classes - 1) * (feature_dim + 1)
    return {
        "raw_dof": num_classes * (feature_dim + 1),
        "margin_preserving_dof": margin_dof,
        "partition_only_dof": margin_dof - 1,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _prov(sidecar: str, reactivation: str):
    return build_provenance_for_research_sidecar(
        sidecar_path=sidecar,
        reactivation_criteria=reactivation,
        measurement_axis="[derived from canonical score law / frozen head]",
        hardware_substrate="macos_arm64",
        captured_at_utc=_UTC,
    )


def build_seg_rate_breakeven_v1() -> CanonicalEquation:
    anchor = EmpiricalAnchor(
        anchor_id="breakeven_arithmetic_cross_check_20260719",
        measurement_utc=_UTC,
        inputs={"score_law": "upstream/evaluate.py:92", "delta_d_seg": 1e-6},
        predicted_output="150.181956 bytes per 1e-6 d_seg",
        empirical_output=(
            "DERIVED identity 100*37_545_489/25*1e-6 = 150.181956; independently "
            "re-derived by the #550 arm from the pinned score law (memo 'Evidence "
            "language and authority boundary' section)"
        ),
        residual=0.0,
        source_artifact=".omx/research/nielsen_infogeo_crosswalk_20260719_codex.md",
        measurement_method="exact arithmetic on the canonical contest score formula",
        provenance=_prov(
            ".omx/research/nielsen_infogeo_crosswalk_20260719_codex.md",
            "re-derive only if upstream/evaluate.py's coefficients or archive "
            "denominator ever change (they are frozen contest law)",
        ),
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
    )
    return CanonicalEquation(
        equation_id="seg_rate_breakeven_v1",
        name="seg-vs-rate KKT indifference price",
        one_line_summary=(
            "B* = (100/25)*37,545,489 * dd_seg = 150,181,956 bytes per unit "
            "d_seg (~150.18 B per 1e-6): secant rows cheaper than B* pay, "
            "costlier rows do not (pose held fixed)."
        ),
        latex_form=(
            r"B^{*}(\Delta d_{seg}) = \frac{100 \cdot 37{,}545{,}489}{25}\,"
            r"\Delta d_{seg} = 1.50181956\times 10^{8}\,\Delta d_{seg}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.seg_rate_breakeven_and_head_gauge_laws_20260719:"
            "seg_rate_breakeven_bytes"
        ),
        domain_of_validity={
            "score_law": "frozen upstream/evaluate.py:92 coefficients",
            "holding": "pose fixed (pose axis measured inactive, #549)",
            "boundary": (
                "an indifference PRICE, not an achievable secant — the "
                "receiver-closed Delta-bytes/Delta-d_seg curve is the sibling "
                "seg_secant_rd_curve measurement"
            ),
        },
        units_in={"delta_d_seg": "d_seg units (argmax disagreement rate)"},
        units_out={"breakeven_bytes": "archive.zip bytes"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"arithmetic_identity": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "#536 waterfill (admission threshold)",
            "seg_secant_rd_curve arm (row ranking)",
            "#553 PDW2 probe (rate-equivalence framing)",
        ),
        canonical_producers=(
            ".omx/research/v10_flattened_lagrangian_kkt_derivation_20260719.md",
            ".omx/research/nielsen_infogeo_crosswalk_20260719_codex.md",
        ),
        provenance=_prov(
            ".omx/research/v10_flattened_lagrangian_kkt_derivation_20260719.md",
            "frozen contest law; no recalibration expected",
        ),
    )


def build_segnet_head_affine_gauge_quotient_v1() -> CanonicalEquation:
    anchor = EmpiricalAnchor(
        anchor_id="pdw1_gauge_audit_338_to_138_20260719",
        measurement_utc=_UTC,
        inputs={
            "packet": "MEASURED 338-byte PDW1 (12 hdr + 10 ids + 80 sites + 20 weights "
                      "+ 36 edges + 144 tie normals + 36 tie offsets)",
            "K": 5, "d": 4,
        },
        predicted_output="(K-1)(d+1)=20 margin scalars; 19 partition-only DOF",
        empirical_output=(
            "DERIVED constructions: 158B (drop the 180 tie bytes — deterministic "
            "functions of sites/weights/edges) -> 138B (reference-class gauge, "
            "margin-preserving) -> 134B (partition-only). Upper constructions, "
            "not minima; PDW1 remains the only MEASURED packet"
        ),
        residual=0.0,
        source_artifact=".omx/research/nielsen_infogeo_crosswalk_20260719_codex.md",
        measurement_method=(
            "gauge algebra on the frozen rank-4 head (Bregman Voronoi first-type "
            "affine-cell identity, Nielsen-Boissonnat-Nock arXiv 0709.2196) applied "
            "to the byte-audited PDW1 layout"
        ),
        provenance=_prov(
            ".omx/research/nielsen_infogeo_crosswalk_20260719_codex.md",
            "#553 PDW2 $0 format probe MEASURES the construction (5-condition gate "
            "incl. frame-195 near-tie reproduction under declared fp32); its packet "
            "bytes append the MEASURED anchor",
        ),
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
    )
    return CanonicalEquation(
        equation_id="segnet_head_affine_gauge_quotient_v1",
        name="max-of-affine head description gauge quotient",
        one_line_summary=(
            "max-of-affine head = power partition; shared-affine gauge removes "
            "d+1 DOF => margin description <= (K-1)(d+1)=20 scalars (19 "
            "partition-only) — PDW1 338B -> 138/134B derived."
        ),
        latex_form=(
            r"l_i(z)=a_i^{\top}z+b_i \Rightarrow \arg\max_i l_i = \text{PowerCell}"
            r"(s_i{=}a_i/2,\,\omega_i{=}b_i{+}\|s_i\|^2);\ \ l_i \sim l_i+u^{\top}z+v"
            r"\Rightarrow \dim = (K{-}1)(d{+}1)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.seg_rate_breakeven_and_head_gauge_laws_20260719:"
            "head_gauge_description_dof"
        ),
        domain_of_validity={
            "head": "frozen SegNet rank-4 max-of-affine head, channel quotient",
            "arithmetic": "exact over the DECLARED receiver arithmetic; fp32 tie "
                          "reconstruction must be reproduced (frame-195 gate)",
            "boundary": "constructions/upper bounds, NOT information minima; silent "
                        "on the spatial/RGB pullback that realizes cells",
        },
        units_in={"num_classes": "K", "feature_dim": "d"},
        units_out={"margin_preserving_dof": "fp32 scalars", "partition_only_dof": "DOF"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"gauge_algebra_identity": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "#553 PDW2 gauge-fixed packet probe",
            "#539 power-diagram witness parametrization",
            "src/tac/boundary_math/power_diagram_witness.py",
        ),
        canonical_producers=(
            ".omx/research/nielsen_infogeo_crosswalk_20260719_codex.md",
            ".omx/research/segnet_recursive_fractal_factorization_20260715.md",
        ),
        provenance=_prov(
            ".omx/research/nielsen_infogeo_crosswalk_20260719_codex.md",
            "PDW2 measured bytes (or a refutation at the frame-195 gate) recalibrate",
        ),
    )
