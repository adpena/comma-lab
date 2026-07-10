# SPDX-License-Identifier: MIT
"""roadlane_texture_generator — the v8 Road/Lane generator-coverage close (texture primitive).

#394 UNIT A. The v8 rate enemy is COVERAGE, not coding (mature-codec audit
``.omx/research/mature_codec_toolbox_audit_20260710.md``): 53% of the 0.074 residual is the
Road/Lane generator not covering the class, NOT a weak coder. The decisive NEW input is the
texture price list (``.omx/research/segnet_texture_perception_20260710.md``,
:mod:`tac.through_r.stem_perception`): Road/Lane are **texture-defined** — a constant colour
LOSES both through R (Road −3.50, Lane −5.00 flat floor), and the ONLY winning texture in the
568-tile through-R sweep is a **period-4 (stem-Nyquist ≈9 cam-px) high-contrast luminance
grating** (Road = bright-on-dark, Lane = dark-on-bright). The trained mod32cap witness beats the
0.0416 flat-paint floor 8.7× BECAUSE it synthesises texture.

THIS module builds the **composed per-class texture generator** and byte-accounts it:

* **basins** (Undrivable / MyCar / Movable) — texture-free FLAT fill (the price list: each wins
  flat through R). Cheap: 1 colour each.
* **Road / Lane** — the period-4 luminance grating primitive (the price-list winner). The grating
  STRUCTURE is a generic parametric family = rule-118 FREE (inflate.py code); only the fitted
  {two colours, phase, orientation, period, duty} per class are video-derived = COUNTED (~40 bits).

The generator fills the GT partition ``L*`` per pair with these SegNet-optimal primitives and is
measured THROUGH R (:func:`tac.through_r.palette_realization.run_arm`) vs the flat-paint floor
(0.0416) and the trained witness (0.0048), per class. This isolates the TEXTURE-coverage question
(does the period-4 grating win the REAL curved/thin Road/Lane region shapes through R at n600?)
from the GEOMETRY-coverage question (does the lane-poly/horizon placement match L*? — that is the
#234 carrier, byte-accounted separately). Using the GT ``L*`` as the PLACEMENT makes this the
texture CEILING of the generator: a texture win here proves the primitive; the geometry carrier
(SPEC_v8.1 §I: horizon 4167 B + lateral curves + lane LBND2) supplies the counted rate.

NO-FAKE / REUSE-not-rederive: the texture families + description-length are the canonical
:mod:`tac.through_r.stem_perception` (``synth_tile`` / ``texture_dl_bits`` / ``TextureSpec``); the
decision-geometry basin colours are :func:`tac.through_r.palette_realization.map_decision_geometry`;
the through-R d_seg is the canonical :func:`palette_realization.run_arm`. This module adds ONLY the
per-class fill composition + byte accounting; it re-derives NOTHING. Every realized number is
``[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]`` — NEVER a score; the
pointer (contest-CPU 0.19110) moves only through byte-closed exact eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.through_r.resolution_chain import SEG_H, SEG_W
from tac.through_r.stem_perception import (
    TextureSpec,
    synth_tile,
    texture_dl_bits,
)
from tac.witness_control.perclass_verdict import CLASS_NAMES, N_CLASSES

__all__ = [
    "BASIN_CLASSES",
    "TEXTURE_CLASSES",
    "ClassFill",
    "RoadLaneTextureError",
    "TextureFillPlan",
    "byte_account_texture_fill",
    "default_roadlane_grating_specs",
    "fill_partition_texture",
    "fit_flat_only_plan",
    "fit_texture_fill_plan",
    "plan_from_palette",
    "run_composed_generator_arm",
]

# Canonical class indices (comma10k canonical order — CLAUDE.md, MEASURED, do NOT luma-sort):
#   0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar
ROAD, LANE, UNDRIVABLE, MOVABLE, MYCAR = 0, 1, 2, 3, 4
# The price-list finding: basins win FLAT through R (Undriv +8.33, MyCar +11.85, Movable +0.31);
# Road/Lane LOSE flat (−3.50 / −5.00) and need the period-4 grating.
BASIN_CLASSES: tuple[int, ...] = (UNDRIVABLE, MYCAR, MOVABLE)
TEXTURE_CLASSES: tuple[int, ...] = (ROAD, LANE)

# The stem-Nyquist period (finest texture surviving the stride-2 stem) — MEASURED
# (stem_perception.stem_nyquist: 2*STEM_STRIDE = 4 seg-input px). The winning grating period.
STEM_NYQUIST_PERIOD = 4

_ADVISORY_LABEL = "[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]"


class RoadLaneTextureError(ValueError):
    """Raised on a mis-shaped / toy / non-authority texture-generator input."""


# --------------------------------------------------------------------------- #
# The per-class fill plan.                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassFill:
    """One class's fill primitive: a :class:`TextureSpec` + its counted description-length.

    ``spec.family == 'flat'`` for basins (1 colour); ``'stripe'`` / ``'gabor'`` for Road/Lane
    (the period-4 grating). ``bits`` = :func:`texture_dl_bits` (the COUNTED generator-parameter
    rate; the deterministic synth code is rule-118 FREE). ``source`` records how the spec was
    obtained (``'price_list'`` / ``'decision_geometry_flat'`` / ``'default_grating'`` / ``'given'``).
    """

    class_idx: int
    class_name: str
    spec: TextureSpec
    bits: int
    source: str


@dataclass
class TextureFillPlan:
    """The GLOBAL (video-invariant) per-class fill plan + its exact byte account.

    ``fills`` maps class index -> :class:`ClassFill`. The plan is video-invariant (one texture per
    class for the whole clip) so its rate is a tiny constant regardless of frame count — the whole
    point of the texture primitive (near-free; the counted rate is the GEOMETRY carrier, elsewhere).
    """

    fills: dict[int, ClassFill]
    total_texture_bits: int
    total_texture_bytes: float
    color_quant: int
    label: str = _ADVISORY_LABEL
    provenance: dict[str, Any] = field(default_factory=dict)

    def spec_for(self, c: int) -> TextureSpec:
        return self.fills[int(c)].spec


def default_roadlane_grating_specs(
    *,
    period: int = STEM_NYQUIST_PERIOD,
    road_dark: tuple[float, float, float] = (0.0, 0.0, 0.0),
    road_bright: tuple[float, float, float] = (160.0, 160.0, 160.0),
    orientation: float = 135.0,
    duty: float = 0.5,
) -> dict[int, TextureSpec]:
    """The DEFAULT period-4 luminance-grating specs for Road/Lane — the MEASURED price-list winner.

    The exact winning tiles from the landed 568-tile through-R sweep
    (``experiments/results/stem_perception_20260710/tile_responses.jsonl``): Road = **bright-on-dark**
    (160-on-0, period 4, orientation **135°**, margin +8.336, win 0.887); Lane = **dark-on-bright**
    (0-on-160, same period/orientation, margin +1.994, win 0.970 — the reversed polarity the price
    list MEASURED matters). Period 4 = the stem-Nyquist (≈9 cam-px, the only surviving scale). The
    STRUCTURE is rule-118 FREE; the fitted colours+phase+orientation are the counted ~40 bits/class.
    """

    if int(period) < 1:
        raise RoadLaneTextureError("grating period must be >= 1")
    road = TextureSpec(
        family="stripe", c_a=road_bright, c_b=road_dark, period=int(period),
        orientation=float(orientation), duty=float(duty),
    )
    # Lane = reversed polarity (dark-on-bright): swap c_a/c_b.
    lane = TextureSpec(
        family="stripe", c_a=road_dark, c_b=road_bright, period=int(period),
        orientation=float(orientation), duty=float(duty),
    )
    return {ROAD: road, LANE: lane}


def fit_texture_fill_plan(
    segnet: Any | None = None,
    *,
    price_list: Any | None = None,
    basin_flat_colors: dict[int, tuple[float, float, float]] | None = None,
    roadlane_specs: dict[int, TextureSpec] | None = None,
    color_quant: int = 5,
    n_classes: int = N_CLASSES,
) -> TextureFillPlan:
    """Build the GLOBAL per-class :class:`TextureFillPlan`.

    Basins (Undrivable/MyCar/Movable) -> optimal FLAT colour. Road/Lane -> the period-4 grating.

    Source resolution (value-provenance ladder):
      * ``basin_flat_colors`` given -> use them (``'given'``).
      * else ``price_list`` given (a :class:`stem_perception.ClassPriceList`) -> the cheapest
        winning FLAT spec per basin (``'price_list'``); a basin with no flat winner falls back to
        the decision-geometry best colour.
      * else ``segnet`` given -> probe the decision geometry
        (:func:`palette_realization.map_decision_geometry`) and take each basin's max-mean-logit
        colour (``'decision_geometry_flat'``).
      * Road/Lane: ``roadlane_specs`` given -> use them; else the ``price_list`` cheapest winner per
        class if it is a grating; else :func:`default_roadlane_grating_specs` (``'default_grating'``).

    The plan is video-invariant; ``total_texture_bytes`` is its whole-clip counted rate.
    """

    fills: dict[int, ClassFill] = {}
    names = CLASS_NAMES if int(n_classes) == N_CLASSES else tuple(
        f"class_{c}" for c in range(int(n_classes))
    )

    # --- basins: optimal flat colour -------------------------------------- #
    dg = None
    for c in BASIN_CLASSES:
        color: tuple[float, float, float] | None = None
        src = ""
        if basin_flat_colors is not None and c in basin_flat_colors:
            color = tuple(float(v) for v in basin_flat_colors[c])  # type: ignore[assignment]
            src = "given"
        elif price_list is not None:
            pt = price_list.per_class_cheapest.get(names[c])
            if pt is not None and pt.spec.family == "flat":
                color = tuple(float(v) for v in pt.spec.c_a)
                src = "price_list"
        if color is None:
            if dg is None:
                from tac.through_r.palette_realization import map_decision_geometry

                dg = map_decision_geometry(segnet)
            color = tuple(float(v) for v in dg.best_color_for_class(c))
            src = "decision_geometry_flat"
        spec = TextureSpec(family="flat", c_a=color)
        fills[c] = ClassFill(
            class_idx=c, class_name=names[c], spec=spec,
            bits=texture_dl_bits(spec, color_quant=color_quant), source=src,
        )

    # --- Road / Lane: the period-4 grating -------------------------------- #
    defaults = default_roadlane_grating_specs()
    for c in TEXTURE_CLASSES:
        spec: TextureSpec | None = None
        src = ""
        if roadlane_specs is not None and c in roadlane_specs:
            spec = roadlane_specs[c]
            src = "given"
        elif price_list is not None:
            pt = price_list.per_class_cheapest.get(names[c])
            if pt is not None and pt.spec.family in ("stripe", "gabor", "checker"):
                spec = pt.spec
                src = "price_list"
        if spec is None:
            spec = defaults[c]
            src = "default_grating"
        fills[c] = ClassFill(
            class_idx=c, class_name=names[c], spec=spec,
            bits=texture_dl_bits(spec, color_quant=color_quant), source=src,
        )

    return byte_account_texture_fill(
        TextureFillPlan(
            fills=fills, total_texture_bits=0, total_texture_bytes=0.0, color_quant=int(color_quant),
            provenance={
                "basin_classes": list(BASIN_CLASSES),
                "texture_classes": list(TEXTURE_CLASSES),
                "grating_period": STEM_NYQUIST_PERIOD,
            },
        )
    )


def fit_flat_only_plan(
    segnet: Any | None = None,
    *,
    dg: Any | None = None,
    color_quant: int = 5,
    n_classes: int = N_CLASSES,
) -> TextureFillPlan:
    """The all-FLAT matched control plan: EVERY class flat at its decision-geometry-optimal colour.

    Reproduces the flat-paint floor IN the same through-R harness (the matched-control arm) so the
    texture generator is measured against a flat baseline on the SAME frames/SegNet, not only
    against the externally-cited 0.0416. Road/Lane get their max-mean-logit colour too (the price
    list MEASURED no flat WINS Road/Lane; this arm makes that visible in composition). Reuse
    :func:`palette_realization.map_decision_geometry` for the colours (``dg`` optional to share the
    probe with :func:`fit_texture_fill_plan`).
    """

    names = CLASS_NAMES if int(n_classes) == N_CLASSES else tuple(
        f"class_{c}" for c in range(int(n_classes))
    )
    if dg is None:
        from tac.through_r.palette_realization import map_decision_geometry

        dg = map_decision_geometry(segnet)
    fills: dict[int, ClassFill] = {}
    for c in range(int(n_classes)):
        color = tuple(float(v) for v in dg.best_color_for_class(c))
        spec = TextureSpec(family="flat", c_a=color)  # type: ignore[arg-type]
        fills[c] = ClassFill(
            class_idx=c, class_name=names[c], spec=spec,
            bits=texture_dl_bits(spec, color_quant=color_quant), source="decision_geometry_flat",
        )
    return byte_account_texture_fill(
        TextureFillPlan(
            fills=fills, total_texture_bits=0, total_texture_bytes=0.0, color_quant=int(color_quant),
            provenance={"arm": "flat_only_control"},
        )
    )


def plan_from_palette(
    palette: np.ndarray,
    *,
    grating_specs: dict[int, TextureSpec] | None = None,
    color_quant: int = 5,
    n_classes: int = N_CLASSES,
) -> TextureFillPlan:
    """Build a plan from a per-class SCENE palette ``(n_classes,3)``, optionally overriding classes
    with gratings.

    Each class is flat at ``palette[c]`` (the REAL scene colour — what actually survives SegNet's
    context, per the flat-paint floor 0.0416) UNLESS ``grating_specs[c]`` is given (Road/Lane -> the
    period-4 grating). This is the HONEST generator: scene-realistic basins so the composition
    context is real, then the texture lever swaps ONLY Road/Lane. ``grating_specs=None`` -> the
    all-flat scene baseline (reproduces the flat-paint floor in-harness).
    """

    palette = np.asarray(palette, dtype=np.float64)
    if palette.shape != (int(n_classes), 3):
        raise RoadLaneTextureError(f"palette must be ({n_classes},3); got {palette.shape}")
    names = CLASS_NAMES if int(n_classes) == N_CLASSES else tuple(
        f"class_{c}" for c in range(int(n_classes))
    )
    grating_specs = grating_specs or {}
    fills: dict[int, ClassFill] = {}
    for c in range(int(n_classes)):
        if c in grating_specs:
            spec = grating_specs[c]
            src = "grating"
        else:
            spec = TextureSpec(family="flat", c_a=tuple(float(v) for v in palette[c]))  # type: ignore[arg-type]
            src = "scene_mean_flat"
        fills[c] = ClassFill(
            class_idx=c, class_name=names[c], spec=spec,
            bits=texture_dl_bits(spec, color_quant=color_quant), source=src,
        )
    return byte_account_texture_fill(
        TextureFillPlan(
            fills=fills, total_texture_bits=0, total_texture_bytes=0.0, color_quant=int(color_quant),
            provenance={"from": "scene_palette", "grating_classes": sorted(grating_specs)},
        )
    )


def byte_account_texture_fill(plan: TextureFillPlan) -> TextureFillPlan:
    """Fill in ``total_texture_bits`` / ``total_texture_bytes`` from the per-class :class:`ClassFill`.

    Exact COUNTED bytes = the sum of the per-class generator-parameter description lengths
    (rule-118: the deterministic synth STRUCTURE is FREE; only the fitted colours+phase+orientation
    are counted). Returns the SAME plan with the totals set (in place + returned for chaining).
    """

    bits = int(sum(f.bits for f in plan.fills.values()))
    plan.total_texture_bits = bits
    plan.total_texture_bytes = bits / 8.0
    return plan


# --------------------------------------------------------------------------- #
# Composition: fill the GT partition with the per-class textures.              #
# --------------------------------------------------------------------------- #
def fill_partition_texture(
    lab: np.ndarray,
    plan: TextureFillPlan,
    *,
    h: int = SEG_H,
    w: int = SEG_W,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    """Fill an ``(h,w)`` label map with each class's :class:`TextureSpec` -> render-grid float ``(h,w,3)``.

    Each class region ``lab==c`` is filled from a full-frame synthesis of ``plan.spec_for(c)`` (so
    the grating phase is GLOBALLY coherent across the frame — a lane at column x reads the same
    grating phase as the road beside it, which is what the frozen SegNet's translation-covariant
    stem sees). A class present in ``lab`` but absent from ``plan.fills`` RAISES (NO silent hole).
    """

    lab = np.asarray(lab)
    if lab.shape != (h, w):
        raise RoadLaneTextureError(f"lab must be ({h},{w}); got {lab.shape}")
    present = np.unique(lab)
    missing = [int(c) for c in present if int(c) not in plan.fills and 0 <= int(c) < int(n_classes)]
    if missing:
        raise RoadLaneTextureError(
            f"lab contains classes {missing} with no fill in the plan (classes {list(plan.fills)}); "
            "NO silent hole."
        )
    frame = np.zeros((h, w, 3), dtype=np.float64)
    for c in present:
        c = int(c)
        if c not in plan.fills:
            continue  # out-of-range label guard (already raised above for in-range)
        tile = synth_tile(plan.spec_for(c), h=h, w=w)
        m = lab == c
        frame[m] = tile[m]
    return frame


def run_composed_generator_arm(
    labs: np.ndarray,
    *,
    plan: TextureFillPlan,
    segnet: Any | None = None,
    verdict_batch: int = 32,
    decompose_radius: int = 2,
    n_classes: int = N_CLASSES,
    allow_subset_reason: str | None = None,
    arm_name: str = "composed_texture_generator",
) -> Any:
    """Fill every GT partition in ``labs`` with the plan's textures, measure THROUGH R -> d_seg.

    Reuses the canonical :func:`palette_realization.run_arm` (through-R + per-class + flip
    decomposition) — this module supplies ONLY the texture-filled render-grid frames. n600
    discipline is inherited (``N != 600`` requires ``allow_subset_reason``). Returns the
    :class:`palette_realization.ArmResult` (the ``run_arm`` verdict object) unchanged.
    """

    from tac.through_r.palette_realization import load_frozen_segnet, run_arm

    labs = np.asarray(labs)
    if labs.ndim != 3 or labs.shape[1:] != (SEG_H, SEG_W):
        raise RoadLaneTextureError(
            f"labs must be (N,{SEG_H},{SEG_W}); got {labs.shape}"
        )
    if segnet is None:
        segnet = load_frozen_segnet("cpu-torch")
    frames = [
        fill_partition_texture(labs[i], plan, n_classes=int(n_classes))
        for i in range(labs.shape[0])
    ]
    return run_arm(
        arm_name,
        frames,
        labs,
        segnet=segnet,
        no_R=False,
        verdict_batch=int(verdict_batch),
        decompose_radius=int(decompose_radius),
        n_classes=int(n_classes),
        allow_subset_reason=allow_subset_reason,
    )
