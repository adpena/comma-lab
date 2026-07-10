# SPDX-License-Identifier: MIT
"""Canonical equation: ROAD/LANE GRATING-IN-COMPOSITION REFUTED — the context-free price-list win
does NOT transfer to region placement (#394 UNIT A, MEASURED n600 through-R).

The texture price list (``segnet_texture_perception_20260710``, ``tac.through_r.stem_perception``)
MEASURED that a CONTEXT-FREE whole-frame period-4 luminance grating (bright 160 / dark 0, orient
135°) WINS Road (+8.336, win 0.887) and Lane (+1.994, win 0.970) through R, while NO flat colour
wins either (Road −3.50, Lane −5.00). It PROPOSED that grating as the v8 Road/Lane carrier fill.

#394 UNIT A MEASURED that proposal at n600 through-R by COMPOSING the generator: fill the GT
partition ``L*`` per pair with scene-mean-flat basins + the price-list grating for Road/Lane, push
through the real R + frozen CPU-torch SegNet, per-class d_seg vs L*. The RESULT REFUTES the
grating-in-composition:

    d_seg(scene-flat baseline, n600)  = 0.07095   (Road 0.0165, Lane 0.1204, Movable 0.403, MyCar 0.235)
    d_seg(grating Road/Lane, n600)    = 0.29873   (Road 0.9985, Lane 1.0,    Movable 1.0)
    Δ(texture − flat)                 = +0.22778   ANTAGONISTIC — the grating DESTROYS Road & Lane

WHY (the mechanism, DERIVED from the two measurements). SegNet is a U-Net with a large receptive
field: its per-pixel argmax is CONTEXT-DOMINATED. A whole-frame grating tile evokes Road in
ISOLATION (no competing context), but the SAME grating placed in the REAL road region — bordered by
flat Undrivable/Lane/Movable, within real scene geometry — injects period-4 high-contrast edges
THROUGHOUT the region that the surrounding context re-reads as not-Road (argmax flips 100%). In
composition the scene-mean FLAT road colour ALREADY wins Road (d_seg 0.0165), because the context
supplies the discrimination the isolated tile lacked. So the price-list "Road/Lane need texture"
finding is a CONTEXT-FREE ARTIFACT; it does not survive the transfer to scene composition.

CONSEQUENCE for the v8 route (the constraint carved). PINS the v8 Road/Lane carrier FILL to FLAT
scene colour (Road covered flat at 0.017; the grating is a NO-GO). RELOCATES the coverage residual
to Movable (flat 0.403) + MyCar (0.235) per-frame COLOUR and Lane (0.120) thin-structure boundary
jitter (the #333 annulus) — NONE of which is a texture gap. The 53%-of-enemy "Road/Lane
generator-coverage gap" is therefore NOT closed by a texture primitive; the texture-fill FAMILY is
NOT killed (reformulations: thin-band / boundary-annulus / scene-adapted-colour / lane-tangent
gratings, per-pair flat) — one FORMULATION (whole-region global context-free grating) is refuted.

verdict_scope: FORMULATION. axis: [through-R] n600 (advisory, NON-PROMOTABLE). means != ends:
pointer contest-CPU 0.19110 UNMOVED. Mechanism modules: tac.through_r.roadlane_texture_generator
(the generator + byte account) + tac.boundary_math.movable_site_coder (the Movable sparse-site
carrier: 2145 sites -> 6289 B tracked vs 9094 B raw, box-IoU 0.743). Driver:
experiments/measure_v8_geocoder_close.py. Verdict JSON:
experiments/results/v8_geocoder_close_n600/verdict_v8_geocoder_texture_close.json.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_macos_cpu_advisory,
    build_provenance_for_predicted,
)

EQUATION_ID = "roadlane_grating_composition_refuted_v1"

_UTC = "2026-07-10T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- the MEASURED n600 through-R anchor numbers (load-bearing, quoted verbatim from the verdict) ---
FLAT_BASELINE_AGG_DSEG = 0.07095       # scene-mean-flat all classes, n600 through-R
GRATING_GENERATOR_AGG_DSEG = 0.29873   # scene-flat basins + price-list grating Road/Lane, n600
TEXTURE_MINUS_FLAT = 0.22778           # ANTAGONISTIC delta (grating WORSE)
FLAT_ROAD_DSEG = 0.0165                # Road wins flat in composition
GRATING_ROAD_DSEG = 0.9985             # grating DESTROYS Road
FLAT_LANE_DSEG = 0.1204
GRATING_LANE_DSEG = 1.0                # grating DESTROYS Lane
# The Movable sparse-site carrier (the geometry primitive that DOES work), n600:
MOVABLE_SITES_TOTAL = 2145
MOVABLE_TRACKED_BYTES = 6289           # tracked + presence
MOVABLE_RAW_BYTES = 9094               # per-frame independent
MOVABLE_BOX_IOU = 0.743


def build_roadlane_grating_composition_refuted_v1() -> CanonicalEquation:
    """The MEASURED n600 through-R refutation of the grating-in-composition Road/Lane carrier.

    One anchor, VERIFIED_VIA_EMPIRICAL_ANCHOR: the matched-control A/B (scene-flat baseline vs
    scene-flat basins + price-list grating Road/Lane) at n600, real R + frozen CPU-torch SegNet.
    """

    anchor = EmpiricalAnchor(
        anchor_id="roadlane_grating_composition_refuted_n600_through_r_20260710",
        measurement_utc=_UTC,
        inputs={
            "driver": "experiments/measure_v8_geocoder_close.py --n 600 (governed via tools/safe_run.py)",
            "generator": "tac.through_r.roadlane_texture_generator (scene-flat basins + price-list grating)",
            "grating_spec": "period-4 stripe, orient 135°, bright 160 / dark 0 (Road bright-on-dark, Lane reversed)",
            "measurement": "canonical tac.through_r.palette_realization.run_arm (real R + frozen CPU-torch SegNet)",
            "gt": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz lstars (L*)",
            "source_price_list": "segnet_texture_perception_20260710 (context-free whole-frame tile win)",
        },
        predicted_output={
            "price_list_hypothesis": (
                "filling Road/Lane with the context-free-winning period-4 grating should win them "
                "through R (Road +8.336 / Lane +1.994 as whole-frame tiles)"
            ),
        },
        empirical_output={
            "flat_baseline_agg_dseg": FLAT_BASELINE_AGG_DSEG,
            "grating_generator_agg_dseg": GRATING_GENERATOR_AGG_DSEG,
            "texture_minus_flat": TEXTURE_MINUS_FLAT,
            "flat_Road_dseg": FLAT_ROAD_DSEG,
            "grating_Road_dseg": GRATING_ROAD_DSEG,
            "flat_Lane_dseg": FLAT_LANE_DSEG,
            "grating_Lane_dseg": GRATING_LANE_DSEG,
            "movable_sites_total": MOVABLE_SITES_TOTAL,
            "movable_tracked_bytes": MOVABLE_TRACKED_BYTES,
            "movable_raw_bytes": MOVABLE_RAW_BYTES,
            "movable_box_iou": MOVABLE_BOX_IOU,
            "verdict": (
                "REFUTED (FORMULATION): the context-free price-list grating win does NOT transfer to "
                "region placement in scene composition — it DESTROYS Road (0.0165->0.9985) and Lane "
                "(0.1204->1.0), +0.228 ANTAGONISTIC vs the matched scene-flat baseline. SegNet's U-Net "
                "context already discriminates Road flat (0.017). Pins Road/Lane fill to FLAT scene "
                "colour; relocates the coverage residual to Movable/MyCar per-frame colour + Lane "
                "boundary jitter (NOT a texture gap). Texture-fill FAMILY not killed (reformulations "
                "queued). means != ends: pointer 0.19110 UNMOVED."
            ),
        },
        residual=0.0,
        source_artifact="experiments/results/v8_geocoder_close_n600/verdict_v8_geocoder_texture_close.json",
        measurement_method="matched-control A/B (flat vs grating), n600, real R + frozen CPU-torch SegNet argmax",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_macos_cpu_advisory(
            archive_sha256=str(_dummy_sha()),
            source_path="experiments/results/v8_geocoder_close_n600/v8_geocoder_close_n600.json",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Road/Lane grating-in-composition REFUTED: the context-free whole-frame period-4 grating "
            "that wins Road/Lane in ISOLATION (price list) DESTROYS them when placed in the real region "
            "shapes within scene composition (n600 through-R); SegNet context dominates ⇒ scene-flat wins"
        ),
        one_line_summary=(
            "n600 through-R: scene-flat 0.071 vs grating 0.299 (Road 0.017->0.999, Lane 0.120->1.0, "
            "+0.228 antagonistic) — price-list texture win is a context-free artifact; pin Road/Lane to flat."
        ),
        latex_form=(
            r"d_{seg}^{grating}=0.299 > d_{seg}^{flat}=0.071\;\Rightarrow\;"
            r"\Delta=+0.228\ (\text{antagonistic});\quad "
            r"\text{win}^{iso}_{Road}=0.887\ \text{but}\ d_{seg}^{comp}_{Road}=0.999"
        ),
        python_callable_module_path=(
            "tac.through_r.roadlane_texture_generator:run_composed_generator_arm"
        ),
        domain_of_validity={
            "vehicle": ["v8_perclass_geometric_carrier", "coord_inr_seg_witness"],
            "carrier": "v8 Road/Lane generator FILL primitive",
            "measurement_axis": ["macOS-CPU advisory", "through-R"],
            "note": (
                "FORMULATION-scoped refutation of the WHOLE-REGION GLOBAL CONTEXT-FREE grating fill. "
                "Does NOT kill the texture-fill family: thin-band / boundary-annulus / scene-adapted-colour "
                "/ lane-tangent-oriented gratings + per-pair flat are untested reformulations. The v8 "
                "Road/Lane carrier fill is FLAT scene colour; the residual is Movable/MyCar colour + Lane "
                "boundary jitter, not texture."
            ),
        },
        units_in={
            "grating_spec": "period-4 stripe 160/0 @135deg", "lab": "gt L* argmax partition",
            "R": "render_grid_to_camera_uint8 + SegNet preprocess bilinear-down",
        },
        units_out={"d_seg": "fraction of pixels != L*"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            # the price-list hypothesis (grating wins Road/Lane in composition) is FALSIFIED: predicted
            # win, measured destroy. The recorded residual is the magnitude of the miss (0.228 antagonistic).
            "grating_wins_composition_hypothesis_falsified": TEXTURE_MINUS_FLAT,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.through_r.roadlane_texture_generator",  # the generator this measured
            "tac.boundary_math.movable_site_coder",      # the sibling Movable geometry carrier
        ),
        canonical_producers=(
            "experiments.measure_v8_geocoder_close",
        ),
        provenance=build_provenance_for_predicted(
            model_id="roadlane_grating_composition_refuted.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[through-R]",
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def _dummy_sha() -> str:
    """The verdict JSON's inputs_sha256 (deterministic re-derivation key) as the advisory anchor sha.

    Read lazily from the landed n600 JSON so the anchor's ``source_sha256`` points at the exact
    measured artifact; falls back to a zero digest if the JSON is absent (registration stays idempotent).
    """
    import json
    from pathlib import Path

    p = Path("experiments/results/v8_geocoder_close_n600/v8_geocoder_close_n600.json")
    if p.exists():
        try:
            sha = json.loads(p.read_text()).get("inputs_sha256")
            if isinstance(sha, str) and len(sha) == 64:
                return sha
        except Exception:
            pass
    return "0" * 64


def populate_roadlane_grating_composition_refuted_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the grating-in-composition refutation law (#394 UNIT A
    equations leg; the DSL/mechanism legs are the two generator modules; the DAG leg is FEED-u394a)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_roadlane_grating_composition_refuted_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="roadlane_grating_composition_refuted_20260710 (#394 UNIT A: n600 through-R MEASURED "
              "refutation of the price-list grating as the v8 Road/Lane carrier fill; pins to flat)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "FLAT_BASELINE_AGG_DSEG",
    "GRATING_GENERATOR_AGG_DSEG",
    "MOVABLE_TRACKED_BYTES",
    "TEXTURE_MINUS_FLAT",
    "build_roadlane_grating_composition_refuted_v1",
    "populate_roadlane_grating_composition_refuted_equation",
]
