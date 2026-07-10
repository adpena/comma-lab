# SPDX-License-Identifier: MIT
"""Canonical equation: PALETTE-REALIZATION CEILING is CONTEXT-DOMINATED, not a fixable palette artifact.

Operator probe (2026-07-09): the C1 transfer-ceiling row realized the PERFECT direct partition (GT
``L*``) as per-class MEAN-RGB and pushed it through R -> frozen CPU-torch SegNet, getting realized
d_seg ``F ~= 0.0337`` (16-strided subset; ``negaudit_retests_c1_e5_20260709.md``). *"That seems like a
big number which should be smaller indicating something we are overlooking."* The hypothesis under test:
``F`` is a PALETTE ARTIFACT — R blends two adjacent class colours into a THIRD class's RGB region — so a
mixing-robust palette (a FREE design variable) would drop ``F`` to boundary noise.

MEASURED (n600, canonical ``tac.through_r.measure_through_r``, 7 arms; ``palette_artifact_probe_20260710``):
the hypothesis is REFUTED, and the overlooked thing is the OPPOSITE of a fixable palette artifact.

  * **U1 canary** — real GT frame1 through R gives d_seg ``1.6e-7`` (~0): ALL of ``F`` is from painting.
  * **F_naive (per-pair mean, n600) = 0.048323** (the subset was 0.0337; n600 is the authority value).
  * **Zero-mixing floor** (per-pair palette, hard seg-grid paint, ZERO R mixing) = **0.041622** — i.e.
    R's resolution mixing (bicubic-up + bilinear-down + uint8) contributes only ``0.0067`` (**14%**);
    **86%** of ``F`` survives with zero mixing. R is NOT the enemy (consistent with the U1 canary).
  * **The palette is NOT a free lever.** Every recolour is worse-or-equal: global-mean 0.0709; camera-res
    global-mean 0.0729; scene-anchored mixing-robust ``==`` global-mean (0.0709, no perturbation helps);
    and the ABSTRACT mixing-robust palette (max-logit class colours) is CATASTROPHIC 0.504 (every class
    except Undrivable flips 100%). The BEST palette is the per-pair scene MEAN (floor 0.0416).
  * **ROOT CAUSE — SegNet argmax is CONTEXT/TEXTURE-dominated, not colour-dominated.** The decision-
    geometry probe (216 constant-colour tiles) decodes **195/216 to Undrivable**; **Road=1, Lane=0**,
    Movable=10, MyCar=10 — Road/Lane NEVER win argmax on colour alone. So NO abstract palette can make a
    flat region read as Road/Lane/Movable; a flat per-class-mean partition strips the texture/context the
    argmax depends on -> ``~0.04`` irreducible d_seg regardless of palette.
  * **Two independent lenses AGREE (cross-validation).** The flip decomposition (a flip whose realized
    class is NOT locally present = manufactured/context; else = boundary jitter) splits baseline into
    **85.1% third-class (0.0411) / 14.9% boundary (0.0072)**; the boundary component ``0.0072`` matches
    the resolution-mixing contribution ``0.0067``, and the third-class component ``0.0411`` matches the
    zero-mixing floor ``0.0416``. The "third-class" mass is a flat-paint CONTEXT mis-read, not R mixing.

VERDICT (verdict_scope FORMULATION; the C1 mirage HARDENS, no new kill). The naive-palette C1 "mirage"
caveat is not softened to naive-only — it is SHARPENED to **ALL-palette**: direct-partition -> palette
realization is CONFIRMED SHUT (floor 0.0416, palette-irreducible). The trained-through-R self-orient
witness (realized d_seg 0.0048) is **8.7x BELOW** this palette floor and remains the ONLY viable
realization regime — because it renders real TEXTURED RGB, not a flat palette.

v8 IMPLICATION (design correction). v8's byte-close realization does NOT "inherit the optimal palette for
free": there is no free optimal palette. A flat per-class colour codec (store 5 colours/pair) floors at
~0.042 d_seg (dominated). v8 must carry enough per-pixel TEXTURE/chroma to keep SegNet's context intact.

Mechanism / reference: ``tac.through_r.palette_realization`` (arms + decision geometry + flip decomp).
Authority: ``[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]`` — a realized d_seg
is NEVER a score; pointer contest-CPU 0.19110 UNMOVED (MEANS).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "palette_realization_ceiling_context_dominated_v1"

_UTC = "2026-07-10T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_MEMO = ".omx/research/palette_artifact_probe_20260710.md"

# --- MEASURED n600 arm values (load-bearing, quoted verbatim from the result JSON) --------------------
F_NAIVE_PERPAIR = 0.048323            # arm 1: per-pair mean palette, render-grid, full R (n600)
F_GLOBAL_MEAN = 0.070947             # control: global mean palette, render-grid
F_MIXROBUST_ABSTRACT = 0.504375      # arm 2a: abstract max-logit palette (CATASTROPHIC)
F_MIXROBUST_SCENE = 0.070947         # arm 2b: scene-anchored MR == global mean (no improvement)
F_CAMERA_RES = 0.072868             # arm 3: camera-res global-mean paint (only down-mix)
F_FLOOR_GLOBALMEAN = 0.050274        # arm 4: zero-mix floor, global-mean palette
F_FLOOR_PERPAIR = 0.041622           # arm 4b: zero-mix floor, per-pair mean palette (the BEST floor)
RES_MIXING_CONTRIB = 0.006701        # F_naive - F_floor_perpair (R's resolution mixing = 14%)
U1_CANARY_DSEG = 1.59e-07            # real GT f1 through R (n64) ~ 0 -> all F from painting
DECOMP_THIRD_SHARE = 0.8514          # baseline flips: 85.1% third-class (context mis-read)
DECOMP_BOUNDARY_SHARE = 0.1486       # 14.9% boundary (R jitter)
DECOMP_THIRD_COMPONENT = 0.041142    # ~= zero-mix floor 0.0416 (cross-validation)
DECOMP_BOUNDARY_COMPONENT = 0.007181  # ~= resolution-mixing contribution 0.0067 (cross-validation)
DG_UNDRIVABLE_TILES = 195            # of 216 constant-colour tiles decode to Undrivable
DG_TILES_TOTAL = 216
TRAINED_WITNESS_REALIZED_DSEG = 0.0048  # mod32cap trained-through-R self-orient: 8.7x below the floor


def build_palette_realization_ceiling_v1() -> CanonicalEquation:
    """Build the palette-realization-ceiling law (context-dominated, palette-irreducible)."""

    anchor_measured = EmpiricalAnchor(
        anchor_id="palette_realization_ceiling_n600_7arm_20260710",
        measurement_utc=_UTC,
        inputs={
            "harness": "tac.through_r.measure_through_r (frozen CPU-torch SegNet, chunked, n600)",
            "mechanism": "tac.through_r.palette_realization (arms + decision geometry + flip decomp)",
            "target": "GT L* argmax (gt_n600.npz lstars), 600 pairs",
            "realization": "GT L* palette-painted -> R(bicubic-up->uint8->bilinear-down) -> SegNet argmax",
        },
        predicted_output={
            "hypothesis_under_test": (
                "F~=0.0337 is a PALETTE ARTIFACT (R blends two class colours into a third class); a "
                "mixing-robust palette (a free design variable) would drop F to boundary noise"
            ),
        },
        empirical_output={
            "U1_canary_real_gt_through_R_dseg": U1_CANARY_DSEG,
            "F_naive_perpair_mean_n600": F_NAIVE_PERPAIR,
            "F_zero_mixing_floor_perpair": F_FLOOR_PERPAIR,
            "resolution_mixing_contribution": RES_MIXING_CONTRIB,
            "resolution_mixing_share_of_F": round(RES_MIXING_CONTRIB / F_NAIVE_PERPAIR, 3),
            "F_global_mean": F_GLOBAL_MEAN,
            "F_camera_res_globalmean": F_CAMERA_RES,
            "F_mixrobust_scene_anchored": F_MIXROBUST_SCENE,
            "F_mixrobust_abstract_CATASTROPHIC": F_MIXROBUST_ABSTRACT,
            "F_zero_mixing_floor_globalmean": F_FLOOR_GLOBALMEAN,
            "decision_geometry_undrivable_tiles": f"{DG_UNDRIVABLE_TILES}/{DG_TILES_TOTAL}",
            "decision_geometry_road_lane_win_argmax_on_colour": "Road=1, Lane=0 (context-only classes)",
            "decomp_third_class_share": DECOMP_THIRD_SHARE,
            "decomp_boundary_share": DECOMP_BOUNDARY_SHARE,
            "cross_validation": (
                "decomp boundary component 0.0072 ~= resolution-mixing contribution 0.0067; decomp "
                "third-class component 0.0411 ~= zero-mixing floor 0.0416 (two independent lenses agree)"
            ),
            "trained_witness_realized_dseg": TRAINED_WITNESS_REALIZED_DSEG,
            "verdict": (
                "REFUTED: F is NOT a fixable palette artifact. R mixing is only 14%; the palette is not "
                "a free lever (every recolour worse-or-equal, abstract MR catastrophic 0.504); ROOT "
                "CAUSE is SegNet context/texture-domination (constant colour -> Undrivable 195/216; "
                "Road/Lane never win argmax on colour). The C1 mirage HARDENS naive-only -> ALL-palette: "
                "direct-partition->palette realization is CONFIRMED SHUT (floor 0.0416). The trained-"
                "through-R witness (0.0048) is 8.7x below the floor and is the only viable regime. "
                "means != ends: pointer 0.19110 UNMOVED."
            ),
            "verdict_scope": "FORMULATION (direct-partition->palette realization, ALL palettes; n600)",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method=(
            "n600 realized-through-R d_seg over 7 palette-realization arms + 216-tile SegNet decision-"
            "geometry probe + radius-2 third-class-vs-boundary flip decomposition (canonical harness)"
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "re-open direct-partition->palette realization ONLY if a NEW palette family drops the "
                "zero-mixing floor below ~boundary noise (~0.007); the measured floor is 0.0416 "
                "palette-irreducible (context-dominated), so the transfer path stays SHUT pending that"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Palette-realization ceiling is CONTEXT-DOMINATED: realizing a partition as a flat per-class "
            "palette through R -> SegNet floors at ~0.042 d_seg (palette-irreducible); R mixing is only "
            "14%; SegNet argmax is texture/context-dominated (constant colour -> Undrivable 195/216)"
        ),
        one_line_summary=(
            "F(palette realization) n600 = 0.048; zero-mix floor 0.042 (R mixing only 14%); no palette "
            "rescues it (abstract MR 0.504); SegNet context-dominated -> direct-partition->palette SHUT."
        ),
        latex_form=(
            r"F=\mathrm{d\_seg}\big(R(\mathrm{paint}(L^*,P))\big),\ \ "
            r"F_{\text{floor}}=\min_P F\big|_{\text{no-}R}\approx 0.042\ \ (\text{palette-irreducible}),"
            r"\ \ F-F_{\text{floor}}\approx 0.007\ (R\ \text{mixing},\ 14\%)"
        ),
        python_callable_module_path="tac.through_r.palette_realization:run_arm",
        domain_of_validity={
            "vehicle": ["direct_partition_palette_realization", "coord_inr_seg_witness"],
            "measurement_axis": ["macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE"],
            "note": (
                "n600 realized-through-R (advisory, NON-PROMOTABLE). The floor 0.0416 is measured over "
                "the naive-mean/global-mean/mixing-robust palette family; a fundamentally different "
                "realization (textured RGB, i.e. the trained witness) is the escape, not a better palette."
            ),
        },
        units_in={"L_star": "gt_segnet_argmax", "palette": "per_class_rgb", "R": "resolution_chain"},
        units_out={"F": "realized_dseg_fraction"},
        empirical_anchors=(anchor_measured,),
        predicted_vs_empirical_residual={"palette_artifact_hypothesis_refuted": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.through_r.palette_realization",  # the mechanism this law characterises
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # the escape regime (trained)
        ),
        canonical_producers=("tac.through_r.palette_realization",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="see anchor reactivation_criteria (new palette family beating floor 0.007)",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def populate_palette_realization_ceiling_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the palette-realization-ceiling law (latest-row-wins).

    Equations leg of the palette-artifact probe; mechanism leg =
    ``tac.through_r.palette_realization``; DAG leg = FEED-palette; memo = ``palette_artifact_probe_20260710``.
    """
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_palette_realization_ceiling_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="palette_realization_ceiling_20260710 (equations leg of the palette-artifact probe; "
              "context-dominated, palette-irreducible floor 0.0416; C1 mirage HARDENS to ALL-palette)",
    )
    return eq


__all__ = [
    "DECOMP_BOUNDARY_SHARE",
    "DECOMP_THIRD_SHARE",
    "EQUATION_ID",
    "F_FLOOR_PERPAIR",
    "F_MIXROBUST_ABSTRACT",
    "F_NAIVE_PERPAIR",
    "RES_MIXING_CONTRIB",
    "U1_CANARY_DSEG",
    "build_palette_realization_ceiling_v1",
    "populate_palette_realization_ceiling_equation",
]
