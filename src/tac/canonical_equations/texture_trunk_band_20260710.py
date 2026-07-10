# SPDX-License-Identifier: MIT
"""Canonical equation: the TEXTURE TRUNK band-design law (#395 P0, 2026-07-10).

One DERIVED design law, the equations leg of the band-designed per-class stationary texture trunk
(``tac.boundary_math.texture_trunk``; ``.omx/research/texture_trunk_p0_design_20260710.md``).

LAW — ``texture_trunk_band_is_stem_passband_v1``.
The texture trunk's oscillatory feature bank carries frequencies ONLY in the frozen SegNet stem's
transfer PASS-BAND: periods in ``[P_min, band_hi]`` render-px, where ``P_min = 2·stride = 4`` is the
stride-2 stem Nyquist (LAW ``segnet_stem_nyquist_alias_wall_v1``, MEASURED) and ``band_hi`` (default
8.0 ≈ 2·Nyquist) is the ceiling above which texture reads as flat (the palette base's DC job;
period-8/16 tiles read Undrivable in the price list ``segnet_through_r_texture_price_list_v1``).
Periods below ``P_min`` alias away before the first MBConv (dead); periods above ``band_hi`` carry no
in-band oscillation (dead). This is CLAUSE-B minimal-dim: the bank support (⇒ the counted coefficient
dimension F·K·3) is the geometry's bound — the measured stem transfer pass-band — never a guess. The
render grid IS 384×512 = the seg-input grid (``witness_autoconfig`` ``render_h=384, render_w=512``), so
render-px ≡ seg-input px for the surviving band and the periods are in seg-px directly.

DERIVED (from ``segnet_stem_nyquist_alias_wall_v1`` + the price-list breadth) and VERIFIED BY SOURCE
INSPECTION: ``tac.boundary_math.texture_trunk.TextureBandSpec.__post_init__`` REFUSES any period
outside ``[P_min, band_hi]`` at construction, and ``band_limit_report`` proves every built feature's
2-D FFT peak period lands in-band. Rule-118: the Gabor bank STRUCTURE is a generic parametric family
(free, regenerable from (H, W, spec)); only the fitted per-class coefficients W_tex are video-derived
(counted). means != ends: this is a DESIGN law for a witness lever, NOT a score; pointer contest-CPU
0.19110 UNMOVED (the lever moves it only through a byte-closed matched-bytes A/B exact row).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

_UTC = "2026-07-10T07:30:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_MEMO = ".omx/research/texture_trunk_p0_design_20260710.md"
_HW = "apple_m5_max_cpu"

TEXTURE_TRUNK_BAND_EQUATION_ID = "texture_trunk_band_is_stem_passband_v1"
P_MIN_RENDER_PX = 4.0  # 2*stride stem Nyquist (segnet_stem_nyquist_alias_wall_v1)
BAND_HI_DEFAULT_RENDER_PX = 8.0


def _sidecar_prov(reactivation: str) -> object:
    return build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=reactivation,
        measurement_axis=_ADVISORY,
        hardware_substrate=_HW,
    )


def build_texture_trunk_band_is_stem_passband_v1() -> CanonicalEquation:
    """The texture trunk's feature-bank support = the frozen SegNet stem transfer pass-band."""

    anchor = EmpiricalAnchor(
        anchor_id="texture_trunk_band_construction_and_spectral_proof_20260710",
        measurement_utc=_UTC,
        inputs={
            "cited_law": "segnet_stem_nyquist_alias_wall_v1 (P_min = 2*stride = 4 seg-px, MEASURED)",
            "band_hi_default": BAND_HI_DEFAULT_RENDER_PX,
            "method": "TextureBandSpec.__post_init__ refuses out-of-band periods; band_limit_report "
            "2-D FFT peak-period per feature over the render grid",
        },
        predicted_output={
            "bank_support_period_px": [P_MIN_RENDER_PX, BAND_HI_DEFAULT_RENDER_PX],
            "counted_coeff_dim_default": "F*K*3 + K*3 = 24*5*3 + 5*3 = 375",
        },
        empirical_output={
            "all_features_in_band": True,
            "peak_period_min_px": 3.99,
            "peak_period_max_px": 8.02,
            "period_2_refused_at_construction": True,
            "period_16_refused_at_construction": True,
            "bank_is_rule118_free": "bank key ends in _B => excluded from the counted byte-close blob "
            "(measure_witness_blob_bytes / _quantize_blob_from_flat / _load_decoder_params); only "
            "W_tex+bias counted (~2.5e-4 S uncoded at default band)",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="tac.boundary_math.texture_trunk.band_limit_report + TextureBandSpec",
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        provenance=_sidecar_prov(
            "re-derive band_hi if a live matched-bytes A/B shows the trunk wants breadth beyond 8 px "
            "(would contradict the price-list period-8/16-read-flat finding — re-measure the stem)"
        ),
    )
    return CanonicalEquation(
        equation_id=TEXTURE_TRUNK_BAND_EQUATION_ID,
        name=(
            "Texture-trunk feature-bank support = frozen SegNet stem transfer pass-band "
            "[period-4 Nyquist .. band_hi=8] render-px (clause-B minimal-dim; counted coeffs = F*K*3)"
        ),
        one_line_summary=(
            "The band-designed texture trunk carries ONLY periods in [2*stride=4, band_hi=8] render-px "
            "(the measured stem pass-band); out-of-band periods are refused at construction."
        ),
        latex_form=r"\mathrm{supp}(T)=[\,P_{\min}=2s_{\text{stem}}=4,\ \text{band\_hi}=8\,]\ \text{render-px}",
        python_callable_module_path="tac.boundary_math.texture_trunk:band_limit_report",
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "texture_trunk"],
            "perceiver": ["frozen_segnet_efficientnet_b2_stem"],
            "render_grid": ["384x512 = seg-input grid (render-px == seg-px for the band)"],
            "measurement_axis": ["macOS-CPU advisory"],
        },
        units_in={"stem_stride": "seg_output_px_per_step", "band_hi": "render_px"},
        units_out={"period": "render_px", "n_features": "count", "counted_coeff": "count"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"texture_trunk_band": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.boundary_math.texture_trunk",),
        canonical_producers=("tac.boundary_math.texture_trunk",),
        provenance=_sidecar_prov(
            "re-measure the stem pass-band only on frozen-scorer checkpoint change; re-derive band_hi "
            "on a live matched-bytes A/B breadth finding"
        ),
    )
