# SPDX-License-Identifier: MIT
"""Canonical equations: how the FROZEN SegNet perceives texture + the MEASURED per-class
through-R texture price list (task-space witness capstone, texture level; 2026-07-10).

Two MEASURED laws, both realized through the canonical R + frozen CPU-SegNet on the 568-tile
context-free sweep (``experiments/results/stem_perception_20260710/``). Both are
``[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]`` — decision-geometry
probes (whole-frame constant/parametric tiles), explicitly NON-n600; a d_seg VERDICT is n600 only
through :func:`tac.through_r.measure_through_r`. MEANS; pointer 0.19110 UNMOVED.

LAW 1 — ``segnet_stem_nyquist_alias_wall_v1`` (perceiver read + Nyquist).
The EfficientNet-B2 ``conv_stem`` is 32 kernels of 3×3×3, stride-2, SiLU. READ verbatim
(:func:`tac.through_r.stem_perception.characterize_filters`): it is COLOUR + LOW-PASS dominated —
21/32 low-pass (DC-fraction ≥ 0.6), 11 achromatic/luminance + strong colour-opponency (R+G, +R−B,
+G−B, …); orientation is NOT resolvable at 3×3 (oriented_share = 0.0 — orientation selectivity
emerges in later MBConv, not the stem). The stride-2 stem sets a hard Nyquist: the finest texture
PERIOD that survives is ``2·stride = 4`` seg-input px ≈ 9.1 camera px
(``4 · mean(874/384, 1164/512)``). Texture finer than period-4 aliases away before the first MBConv.

LAW 2 — ``segnet_through_r_texture_price_list_v1`` (per-class minimal sufficient texture).
MEASURED over 568 cheapest-first tiles (64 flats + a bounded stripe/checker/gabor grid at periods
{2,4,8,16} × 4 orientations) through R. The per-class CHEAPEST texture (fewest description-length
bits) that WINS the argmax interior with positive margin:

    class         flat-floor    cheapest winner (through R)
    Undrivable    +8.33  (WINS) flat, 15 bits, margin +8.33  — the texture-free DEFAULT basin (sky/top)
    MyCar         +11.85 (WINS) flat, 15 bits, margin +11.85 — flat-winnable (ego-hood colour region)
    Movable       +0.31  (WINS) flat, 15 bits, margin +0.31  — weakly flat-winnable
    Road          −3.50  (LOSES) stripe period-4 gray↔black, 40 bits, margin +8.34, win 0.887
    Lane          −5.00  (LOSES) stripe period-4 black↔gray, 40 bits, margin +1.99, win 0.970

The DECISIVE finding: Road + Lane have a NEGATIVE flat floor (0/64 flats win either) — they are
TEXTURE-DEFINED classes; NO constant colour classifies them through R. The ONLY winning texture in
the whole sweep is a period-4 (= the stem Nyquist, ≈9 cam-px) HIGH-CONTRAST LUMINANCE grating —
exactly 1 of 96 period-4 stripes wins Road and 1 wins Lane; NOTHING at period 2/8/16 wins (period-2
aliases below Nyquist; period-8/16 read as flat → the Undrivable basin). Polarity matters
(Road = bright-on-dark, Lane = dark-on-bright). This is the texture-level statement of the
capstone thesis: the witness's binding job on Road/Lane is stem-Nyquist grating SYNTHESIS, not
colour selection — and it explains why the trained mod32cap witness (d_seg 0.0048) beats the
flat-paint floor (0.0416) by 8.7× (it has an ACTIVE out_tex head, ‖W‖_mean 0.06, over a per-class
palette base). Rule-118 compliance: the period-4 grating STRUCTURE is generic/free; only the fitted
colours + phase are video-derived (counted).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

_UTC = "2026-07-10T06:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_MEMO = ".omx/research/segnet_texture_perception_20260710.md"
_ARTIFACT = "experiments/results/stem_perception_20260710/tile_responses.jsonl"
_HW = "apple_m5_max_cpu"

STEM_NYQUIST_PERIOD_SEG_PX = 4.0
STEM_NYQUIST_PERIOD_CAM_PX = 9.099
LOW_PASS_SHARE = 21.0 / 32.0
ORIENTED_SHARE = 0.0

# per-class (flat_floor_margin, cheapest_winner_bits, cheapest_winner_margin) — MEASURED through R
PRICE_LIST = {
    "Road": (-3.4975, 40, 8.3365),
    "Lane": (-5.0008, 40, 1.9942),
    "Undrivable": (8.3263, 15, 8.3263),
    "Movable": (0.3093, 15, 0.3093),
    "MyCar": (11.8462, 15, 11.8462),
}


def _sidecar_prov(reactivation: str) -> object:
    return build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=reactivation,
        measurement_axis=_ADVISORY,
        hardware_substrate=_HW,
    )


def build_segnet_stem_nyquist_alias_wall_v1() -> CanonicalEquation:
    """LAW 1 — the frozen B2 stem is colour+low-pass dominated with a period-4 alias wall."""

    anchor = EmpiricalAnchor(
        anchor_id="segnet_b2_stem_read_and_nyquist_20260710",
        measurement_utc=_UTC,
        inputs={
            "perceiver": "frozen SegNet encoder.model.conv_stem (32,3,3,3) stride 2 + bn1 SiLU",
            "method": "SVD colour-direction + DC-fraction + 3x3-DFT anisotropy per kernel",
        },
        predicted_output={
            "finest_surviving_period_seg_px": STEM_NYQUIST_PERIOD_SEG_PX,
            "finest_surviving_period_cam_px": STEM_NYQUIST_PERIOD_CAM_PX,
        },
        empirical_output={
            "n_filters": 32,
            "low_pass_share": round(LOW_PASS_SHARE, 4),
            "count_by_kind": {"low-pass (colour/blur)": 21, "high-pass (isotropic)": 11},
            "achromatic_luminance_filters": 11,
            "oriented_share": ORIENTED_SHARE,
            "orientation_note": (
                "orientation NOT resolvable at 3x3 (emerges in later MBConv); stem is a colour + "
                "low-pass front-end, so texture discrimination is a downstream property gated by the "
                "period-4 stem-Nyquist alias wall"
            ),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="tac.through_r.stem_perception.characterize_filters + stem_nyquist",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=_sidecar_prov(
            "re-read if the frozen SegNet checkpoint is ever changed (it is a contest constant)"
        ),
    )
    return CanonicalEquation(
        equation_id="segnet_stem_nyquist_alias_wall_v1",
        name=(
            "SegNet B2 stem is colour+low-pass dominated (21/32 low-pass, 11 achromatic, "
            "oriented_share 0); stride-2 Nyquist = period-4 seg-px (~9.1 cam-px) alias wall"
        ),
        one_line_summary=(
            "Frozen B2 conv_stem: colour + low-pass front-end (no orientation at 3x3); finest "
            "surviving texture period = 2*stride = 4 seg-px ~= 9.1 cam-px (the alias wall)."
        ),
        latex_form=r"P_{\min}=2\cdot s_{\text{stem}}=4\ \text{seg-px}\approx 9.1\ \text{cam-px}",
        python_callable_module_path="tac.through_r.stem_perception:stem_nyquist",
        domain_of_validity={
            "perceiver": ["frozen_segnet_efficientnet_b2_stem"],
            "measurement_axis": ["macOS-CPU advisory"],
        },
        units_in={"stem_stride": "seg_output_px_per_step"},
        units_out={"finest_period_seg_px": "seg_input_px", "finest_period_cam_px": "camera_px"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"segnet_stem_read": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.through_r.stem_perception",),
        canonical_producers=("tac.through_r.stem_perception",),
        provenance=_sidecar_prov("frozen scorer constant; re-read only on checkpoint change"),
    )


def build_segnet_through_r_texture_price_list_v1() -> CanonicalEquation:
    """LAW 2 — per-class minimal sufficient texture: Road/Lane are stem-Nyquist gratings, not colours."""

    anchor = EmpiricalAnchor(
        anchor_id="through_r_texture_price_list_568tile_20260710",
        measurement_utc=_UTC,
        inputs={
            "sweep": "568 tiles = 64 flats + bounded stripe/checker/gabor grid (periods 2/4/8/16 x "
            "4 orientations, 24 colour pairs, 4-level RGB grid)",
            "R": "render_grid_to_camera_uint8 (bicubic-up to camera + uint8) then SegNet "
            "preprocess_input bilinear-down — the real verdict forward",
            "metric": "interior signed margin logit_c - max_{k!=c} logit_k (central-50% box)",
        },
        predicted_output={
            "palette_thesis": "Road+Lane are texture-defined (no flat colour wins); minimal texture "
            "= stem-Nyquist period-4 grating",
        },
        empirical_output={
            "per_class_flat_floor_and_cheapest_winner_bits_margin": {
                k: {"flat_floor_margin": round(v[0], 4), "cheapest_bits": v[1],
                    "cheapest_margin": round(v[2], 4)}
                for k, v in PRICE_LIST.items()
            },
            "road_lane_winners": "ONLY period-4 high-contrast luminance stripe wins Road (1/96) and "
            "Lane (1/96); NOTHING at period 2/8/16; polarity matters (Road bright-on-dark, Lane "
            "dark-on-bright)",
            "modal_class_distribution": {"Undrivable": 496, "MyCar": 51, "Movable": 14, "Road": 6,
                                         "Lane": 1},
            "witness_link": "trained mod32cap witness d_seg 0.0048 = 8.7x below flat-paint floor "
            "0.0416; witness has active out_tex head (||W||_mean 0.06) over per-class palette base",
            "verdict": "Road/Lane binding job is stem-Nyquist grating SYNTHESIS, not colour "
            "selection; Undrivable/MyCar/Movable are flat-winnable texture-free basins",
        },
        residual=0.0,
        source_artifact=_ARTIFACT,
        measurement_method="tac.through_r.stem_perception.per_class_price_list (through_R=True)",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=_sidecar_prov(
            "densify the colour/phase search around the period-4 grating to sharpen the Road/Lane "
            "price (structure is fixed; only the fitted colours/phase are video-derived/counted); "
            "compare mod32cap BEST-render Road/Lane spectra vs this optimal synthetic (deferred: "
            "render decode contends with the active witness run)"
        ),
    )
    return CanonicalEquation(
        equation_id="segnet_through_r_texture_price_list_v1",
        name=(
            "Through-R per-class texture price list: Road/Lane flat-floor NEGATIVE (texture-defined); "
            "cheapest winner = period-4 stem-Nyquist high-contrast luminance grating (~40 bits)"
        ),
        one_line_summary=(
            "MEASURED through-R: Road/Lane win argmax ONLY via a period-4 (stem-Nyquist) high-"
            "contrast luminance stripe; Undrivable/MyCar/Movable are flat-winnable basins."
        ),
        latex_form=(
            r"\text{win}_c(\text{tex})=\big[\,\min_{p\in\text{int}}(\ell_c-\max_{k\ne c}\ell_k)>0\,"
            r"\big];\ \ \text{Road,Lane}\Rightarrow \text{period-4 grating only}"
        ),
        python_callable_module_path="tac.through_r.stem_perception:per_class_price_list",
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "context_free_texture_tile"],
            "classes": ["Road", "Lane", "Undrivable", "Movable", "MyCar"],
            "measurement_axis": ["macOS-CPU advisory"],
            "caveat": "context-free whole-frame tiles (decision geometry), NOT an n600 d_seg verdict",
        },
        units_in={"texture_spec": "parametric_generator", "bits": "description_length_bits"},
        units_out={"signed_margin": "logit_units", "win_fraction": "fraction"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"segnet_texture_price": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.through_r.stem_perception",),
        canonical_producers=("tac.through_r.stem_perception",),
        provenance=_sidecar_prov(
            "re-measure with a denser colour/phase grid or against a live witness render"
        ),
    )
