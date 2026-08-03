# SPDX-License-Identifier: MIT
"""ddm_p4x (#920) — the LANE EXISTENCE PRIMITIVE + per-class BIRTH MATRIX.

The first *verb-native* force in the campaign: it is scored against a VERB
(``ANNIHILATE``), not against a pixel count.

**Why a new shape was required, stated as the measurement that forces it.**
``ddm_cg1r`` (``ee848e88cd``) MEASURED that realized per-flip GT-margin depth is
direction-SYMMETRIC on all nine class edges (Road<->Lane 1.074x) while the COUNT
asymmetry runs to 15.88x.  The "10x lane-erasure discount" is therefore a
VOLUMETRIC / verb-level quantity, **not** per-flip pricing:

    a whole Lane word dies at ~2.5 px of depth because Lane has no interior.

Consequence, and it is the entire design: *a force expressed as another per-pixel
weight is aimed at an already-symmetric quantity and should be expected to measure
null on this channel.*  ``ddm_lg1``'s lane guard composes exactly that way — it is
an ADDITIVE per-pixel addend folded into ``seg_pixel_w`` — which is why its own
ledger row records ``protection=ABSENT`` for the ANNIHILATE verb specifically:
a rim-peel guard up-weights currently-WON support and does nothing whatever for a
whole component being lost.  This module is the missing instrument, and it is a
SEPARATE loss TERM at COMPONENT granularity, never a pixel weight.

**The primitive.**  For each GT connected component ``c`` of a protected class::

    s(c) = logsumexp_beta( m_live(p) for p in c )
    L    = sum_c  w_c * relu(target - s(c))   / n_components

where ``m_live(p) = logit[gt_class(p)] - max_{k != gt_class(p)} logit[k]`` is the
signed live multiclass margin (the SAME idiom the ``margin`` seg form already uses
at ``train_witness_realized_through_R_mlx.py`` L1450-1451, so this reads the
vehicle's real decision surface, not a proxy).

As ``beta -> inf``, ``s(c) -> max_p m_live(p)``: the component's WITNESS pixel.
That is the existence semantics exactly — a word survives argmax iff at least one
of its pixels wins its class.  Protecting the max (not the mean, not the sum) is
what makes this term blind to area and sensitive to EXISTENCE, which is the
property every per-pixel surrogate lacks.

**Cost.**  O(#components), not O(#pixels).  The protected pixel set is small:
MEASURED 1,151 Lane px/frame over 27.64 components/frame, and 2,434 Movable px/frame
over 3.68 components.  So the per-component reduction is a dense ``(K, n_comp)``
masked ``logsumexp`` over PROTECTED PIXELS ONLY (K ~ 3.6k, n_comp ~ 31) — roughly
115k floats.  No scatter primitive, no segment-reduction kernel, exact and
differentiable, and ``mx.logsumexp`` supplies its own max-shift so the reduction is
numerically stable without a separate scatter-max pass.

**Authority.** ``[macOS-CPU advisory]``; ``research_only=True``; ``score_claim=False``;
``promotable=False``.  Contest pointer ``0.1910828242`` [contest-CPU] UNMOVED.
Everything here is MEANS.  No arm in this module has been raced against the scorer,
so no function claims a d_seg effect; the addressable mass is stated as a CEILING
with its denominator, below.

**Addressable mass, with the denominator carried (per the standing law that a
Delta-S quoted without its baseline is unanchored).**  The existence hinge targets
the ANNIHILATE channel *specifically*.  It does NOT address Lane's whole 0.1575 S
ledger debt — 73.0% of Lane's flip PIXELS are ERODE on SURVIVING components, which
is the rim guard's job and is untouched here.  MEASURED from ``gt2_verbs.json``:

    Lane   ANNIHILATE  47,226 px -> 0.040036 S   (9,655 of 16,581 words = 58.23%)
    Movable ANNIHILATE  8,180 px -> 0.006934 S   (361 of 2,207 words = 16.36%)
    Road/Undrivable/MyCar        ->  0.000434 S  (315 + 197 + 0 px; not protected by default)
    ------------------------------------------------------------------
    all-class ANNIHILATE ceiling  ->  0.047403 S  = 7.66% of the 0.6189279 gap

100% capture of the ANNIHILATE channel is therefore numerically ~= the 30% capture
of Lane's total debt that ``gc16`` P4 credits — an independent cross-check that the
two framings agree on magnitude while disagreeing on which pixels are in scope.
The honest figure for THIS instrument is the ANNIHILATE column, and 100% capture is
not a plausible outcome; it is the ceiling that bounds the row.

**The failure is RECALL, not precision** (``gt2``): ANNIHILATE:BIRTH = 10,124:616
words = 16.4x.  A force that suppresses false BIRTH aims at the small side of a
16.4:1 asymmetry.  This term only ever pushes a GT component's witness margin UP,
so it cannot trade recall for precision.

Provenance ladder for every default is declared in ``BIRTH_MATRIX`` and
``derive_beta`` below (constants-are-poison: no bare literal is a default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---- MEASURED scorer-coordinate constants (comma10k canonical order) ----------
# CLAUDE.md, MEASURED from the cached SegNet argmax; NEVER luma-sort re-derived.
ROAD, LANE, UNDRIVABLE, MOVABLE, MYCAR = 0, 1, 2, 3, 4
N_CLASSES = 5
SEG_H, SEG_W = 384, 512  # frozen SegNet argmax plane (upstream modules.py preprocess)
N_PAIRS_N600 = 600

#: S units per single flip on the n600 plane.  DERIVED, never copied:
#: 100 / (600 * 384 * 512).  Cross-checks against the cg1r ledger exactly
#: (185,801 Lane flips * this = 0.157500 S, the ``tr1.lane.annihilate`` magnitude).
S_PER_FLIP = 100.0 / (N_PAIRS_N600 * SEG_H * SEG_W)

#: gt2_verbs.json (ddm_gt2, /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/), n600,
#: substrate gt_argmax_n600.npy + cx1_argmax_n600.npy, zero scorer forwards.
#: LADDER CLASS 3 (measured_anchor).  Caveat, stated because it is load-bearing:
#: the artifact lives on an external volume, so no in-repo gate can verify it and
#: it is unverifiable on another host.  Re-derivation trigger: re-run gt2's verb
#: pass on the current decode corpus if the vehicle's argmax changes.
GT2_VERB_MEASUREMENTS: dict[int, dict[str, float]] = {
    LANE: {
        "gt_px": 690639.0, "gt_components": 16581.0,
        "annihilate_components": 9655.0, "annihilate_px": 47226.0,
        "erode_px": 135683.0, "gouge_px": 2926.0,
        "birth_components": 591.0,
        "frac_px_at_depth_le_1": 0.7504325704166721,
    },
    MOVABLE: {
        "gt_px": 1460325.0, "gt_components": 2207.0,
        "annihilate_components": 361.0, "annihilate_px": 8180.0,
        "erode_px": 53940.0, "gouge_px": 16718.0,
        "birth_components": 5.0,
        "frac_px_at_depth_le_1": None,  # not the discriminating statistic for Movable
    },
}

#: ---- THE CONNECTIVITY GRAMMAR MISMATCH (ddm_p4x MEASURED 2026-08-03) ----------
#: ``gt2``'s word grammar is **4-CONNECTED**.  This arm re-derived it from the same
#: cached corpus and reproduced gt2 EXACTLY under 4-connectivity on every quantity
#: (Lane 16,581 components / 9,655 annihilated / 0.5823 rate / 47,226 px; Movable
#: 2,207 / 361 / 0.1636 / 8,180) and NOT under 8-connectivity (Lane 14,323 = 0.864x).
#: Pixel totals matched 1.0000 for both classes, so the corpus is identical and the
#: divergence is purely the labelling rule.
#:
#: This matters because the DESIGN prescribes 8-connectivity for a physical reason
#: (Rosenfeld: a seed that does not respect 8-connectivity is deleted by the
#: receiver's measured consolidation), while the DEBT is priced on gt2's 4-connected
#: partition.  The two are different grammars and a "word" is not interchangeable
#: between them: for Lane they differ by 13.6%, i.e. 2,258 diagonal joins that
#: 8-connectivity merges into one word and 4-connectivity splits into two.
#:
#: Resolution, and it is deliberately NOT to silently pick one:
#:   * S-arithmetic is connectivity-INVARIANT (pixels are pixels), so any S ceiling
#:     is safe to quote once its grammar is named.
#:   * Per-WORD rates are NOT invariant and must never be quoted across grammars.
#:   * The default stays 8-connected because the receiver constraint is physical,
#:     and the 8-connected denominators are re-derived here so the row is priced in
#:     the grammar it is actually defended in.
#: LADDER CLASS 3 (measured_anchor), this arm, same cached corpus, $0, zero scorer
#: forwards.  Reproduce with: tools/ddm_p4x_connectivity_control.py
GT2_VERB_MEASUREMENTS_8CONN: dict[int, dict[str, float]] = {
    LANE: {
        "gt_components": 14323.0, "annihilate_components": 7789.0,
        "annihilate_px": 43972.0, "annihilation_rate_of_words": 0.5438,
    },
    MOVABLE: {
        "gt_components": 2197.0, "annihilate_components": 356.0,
        "annihilate_px": 8139.0, "annihilation_rate_of_words": 0.1620,
    },
}

#: The connectivity a component index was built under.  Named, never implicit.
CONNECTIVITY_4, CONNECTIVITY_8 = 4, 8
DEFAULT_CONNECTIVITY = CONNECTIVITY_8

#: Default-protected classes: exactly the two with a materially non-zero
#: word-annihilation rate (Lane 54.38%, Movable 16.20% at 8-conn).  Road (5.45%),
#: Undrivable (6.00%) and MyCar (0.00%) are DELIBERATELY excluded -- MyCar
#: annihilates zero components in 600 frames, so a term over it can only add
#: gradient noise.  Excluding them is a measurement, not an oversight.
DEFAULT_PROTECTED_CLASSES: tuple[int, ...] = (LANE, MOVABLE)


def annihilate_ceiling_s(class_id: int, connectivity: int = DEFAULT_CONNECTIVITY) -> float:
    """S recoverable if EVERY annihilated component of ``class_id`` were preserved.

    A CEILING, not a prediction.  Multiply by a capture fraction to price a row,
    and always quote it against the gap denominator (see module docstring).

    ``connectivity`` is REQUIRED to be explicit in spirit even though it defaults:
    the ANNIHILATE pixel mass differs between grammars (Lane 47,226 px 4-conn vs
    43,972 px 8-conn) because merging diagonal fragments lets some merged words
    retain a surviving fragment >= 5% and stop counting as annihilated.  That
    difference (0.002758 S for Lane) does not vanish -- it MOVES to the ERODE/GOUGE
    channel, which is the rim guard's instrument, not this one.
    """
    if connectivity == CONNECTIVITY_8:
        return GT2_VERB_MEASUREMENTS_8CONN[class_id]["annihilate_px"] * S_PER_FLIP
    if connectivity == CONNECTIVITY_4:
        return GT2_VERB_MEASUREMENTS[class_id]["annihilate_px"] * S_PER_FLIP
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def protected_ceiling_s(
    classes: tuple[int, ...] = DEFAULT_PROTECTED_CLASSES,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> float:
    """Total ANNIHILATE-channel ceiling over ``classes``, in S units."""
    return sum(annihilate_ceiling_s(c, connectivity) for c in classes)


def mean_component_area(class_id: int) -> float:
    """MEASURED mean GT component area in px for ``class_id`` (gt2)."""
    m = GT2_VERB_MEASUREMENTS[class_id]
    return m["gt_px"] / m["gt_components"]


def derive_beta(mean_area: float, tolerance: float) -> float:
    """DERIVE the softmax sharpness from the component-size law + a stated tolerance.

    ``logsumexp_beta`` over ``n`` values overestimates their max by at most
    ``log(n) / beta``.  Requiring that slack to stay within ``tolerance`` margin
    units gives ``beta >= log(n) / tolerance``.  Using the class's MEASURED mean
    component area for ``n`` makes the default a DERIVATION with an explicit design
    parameter, not a tuned literal.

    Conservative by construction: annihilated components are far SMALLER than the
    class mean (Lane 4.89 px vs mean 41.65 px), so a beta derived from the mean
    over-satisfies the tolerance on exactly the components this term protects.
    """
    if mean_area <= 1.0:
        return math.log(2.0) / tolerance
    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0 margin units")
    return math.log(mean_area) / tolerance


#: Design parameter, DECLARED not measured: how much soft-max slack (in margin
#: units) we accept versus a true max.  0.5 keeps the surrogate inside half a
#: margin unit of the witness pixel.  LADDER CLASS 2 (derived_at_config from a
#: declared tolerance); it is a knob of the SURROGATE, not of the physics, and it
#: is raced like any other.
BETA_TOLERANCE_MARGIN_UNITS = 0.5


@dataclass(frozen=True)
class ClassExistencePolicy:
    """Per-class instantiation of the ONE existence mechanism.

    The mechanism is identical across classes; only the GEOMETRY differs, and every
    field below is chosen from a MEASURED geometric fact about that class rather
    than swept.  This dataclass IS the "per-class birth matrix".
    """

    class_id: int
    class_name: str
    beta: float
    target: float
    weight_policy: str          # "uniform" | "sqrt_area" | "area"
    connectivity: int           # 8 (Rosenfeld) unless a class demands otherwise
    interior_bearing: bool      # True => the class has interior worth protecting
    geometry_note: str


def _lane_policy() -> ClassExistencePolicy:
    return ClassExistencePolicy(
        class_id=LANE,
        class_name="Lane",
        beta=derive_beta(mean_component_area(LANE), BETA_TOLERANCE_MARGIN_UNITS),
        target=0.0,
        weight_policy="uniform",
        connectivity=8,
        interior_bearing=False,
        geometry_note=(
            "75.04% of Lane GT px sit at depth<=1 and GOUGE is only 2,926 px vs ERODE "
            "135,683 px: Lane has NO INTERIOR. Existence is therefore carried by the "
            "witness pixel alone, and weight_policy='uniform' is the volumetric law "
            "applied literally -- an area weight would re-import the per-pixel pricing "
            "that cg1r MEASURED to be already symmetric (1.074x) and hence null on this "
            "verb. 58.23% word-annihilation rate is the highest of any class."),
    )


def _movable_policy() -> ClassExistencePolicy:
    return ClassExistencePolicy(
        class_id=MOVABLE,
        class_name="Movable",
        beta=derive_beta(mean_component_area(MOVABLE), BETA_TOLERANCE_MARGIN_UNITS),
        target=0.0,
        weight_policy="sqrt_area",
        connectivity=8,
        interior_bearing=True,
        geometry_note=(
            "Movable is the ONLY class where GOUGE (16,718 px) is a large fraction of "
            "ERODE (53,940 px) = 31.0%: it loses INTERIOR, not just rim, so unlike Lane "
            "it is a blob with something inside to keep. sqrt_area (not uniform, not "
            "area) is the compromise the geometry argues for -- a 22.7 px mean "
            "annihilated Movable component is 4.6x a Lane one, so treating the two as "
            "identical words would under-weight Movable; full area weighting would "
            "re-import per-pixel pricing. RACED, not asserted."),
    )


#: THE PER-CLASS BIRTH MATRIX -- ONE mechanism, instantiated per class GEOMETRY.
#: Membership is set by DEFAULT_PROTECTED_CLASSES above (measured, not chosen).
BIRTH_MATRIX: dict[int, ClassExistencePolicy] = {
    LANE: _lane_policy(),
    MOVABLE: _movable_policy(),
}

WEIGHT_POLICIES = ("uniform", "sqrt_area", "area")


@dataclass(frozen=True)
class ExistenceHingeConfig:
    """Existence-hinge configuration.  ``weight == 0.0`` => the lever is OFF.

    OFF is byte-identical by construction: the trainer never builds the term, this
    module is never imported, no state is touched and no RNG is drawn.
    """

    weight: float = 0.0
    protected_classes: tuple[int, ...] = DEFAULT_PROTECTED_CLASSES
    beta_override: float | None = None
    target_override: float | None = None
    weight_policy_override: str | None = None
    max_components_per_pair: int = 4096

    def enabled(self) -> bool:
        return float(self.weight) > 0.0

    def policy_for(self, class_id: int) -> ClassExistencePolicy:
        base = BIRTH_MATRIX[class_id]
        if (self.beta_override is None and self.target_override is None
                and self.weight_policy_override is None):
            return base
        pol = self.weight_policy_override or base.weight_policy
        if pol not in WEIGHT_POLICIES:
            raise ValueError(f"weight_policy must be one of {WEIGHT_POLICIES}, got {pol!r}")
        return ClassExistencePolicy(
            class_id=base.class_id,
            class_name=base.class_name,
            beta=float(self.beta_override) if self.beta_override is not None else base.beta,
            target=float(self.target_override) if self.target_override is not None else base.target,
            weight_policy=pol,
            connectivity=base.connectivity,
            interior_bearing=base.interior_bearing,
            geometry_note=base.geometry_note,
        )

    def validate(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("existence hinge weight must be >= 0 (0.0 is OFF)")
        for c in self.protected_classes:
            if c not in BIRTH_MATRIX:
                raise ValueError(
                    f"class {c} has no BIRTH_MATRIX policy; protecting a class whose "
                    "geometry was never measured is exactly the cargo-cult this module "
                    f"exists to avoid. Measured classes: {sorted(BIRTH_MATRIX)}")
        if self.weight_policy_override is not None and \
                self.weight_policy_override not in WEIGHT_POLICIES:
            raise ValueError(f"weight_policy_override must be in {WEIGHT_POLICIES}")


# ------------------------------------------------------------------------------
# GT component extraction (static per pair -- computed ONCE, cached compactly)
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentIndex:
    """Compact per-pair index of protected GT components.

    ``pixel_flat``  (K,) int32 -- flat H*W indices of protected pixels
    ``comp_of_px``  (K,) int32 -- component ordinal in [0, n_comp) per protected pixel
    ``comp_class``  (n_comp,) int32 -- class id of each component
    ``comp_area``   (n_comp,) int32 -- pixel count of each component

    ~9 KB/pair for Lane+Movable, so the whole n600 corpus is ~5.5 MB -- cacheable in
    RAM without a storage-tier decision.  The dense ``(K, n_comp)`` membership mask
    is rebuilt per step from this (trivial) rather than stored.
    """

    pixel_flat: np.ndarray
    comp_of_px: np.ndarray
    comp_class: np.ndarray
    comp_area: np.ndarray

    @property
    def n_comp(self) -> int:
        return int(self.comp_class.shape[0])

    @property
    def n_px(self) -> int:
        return int(self.pixel_flat.shape[0])


def _label_components(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Component labelling of a boolean mask under an EXPLICIT connectivity.

    8-connectivity is the default for a physical reason, not a stylistic one:
    Rosenfeld's constraint is that seeds must respect 8-connectivity or the
    receiver's measured consolidation deletes them (gt2 MEASURED a FRAGMENT negative
    on 4 of 5 classes).  Under 4-connectivity a diagonal Lane dash splits into two
    "words" and this term would defend a component the receiver cannot represent.

    4-connectivity is still selectable because it is the grammar gt2's published
    per-word rates are denominated in -- so an A/B that wants to speak in gt2's own
    units can, PROVIDED it says so.  What is forbidden is leaving it implicit.
    """
    if connectivity == CONNECTIVITY_8:
        structure = np.ones((3, 3), dtype=np.int32)
    elif connectivity == CONNECTIVITY_4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    try:
        from scipy.ndimage import label as _ndlabel
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "existence_hinge requires scipy.ndimage.label for connected-component "
            "extraction. Install scipy, or supply a precomputed ComponentIndex.") from exc
    lab, n = _ndlabel(mask, structure=structure)
    return lab.astype(np.int32), int(n)


def build_component_index(
    lstar: np.ndarray,
    protected_classes: tuple[int, ...] = DEFAULT_PROTECTED_CLASSES,
    *,
    connectivity: int = DEFAULT_CONNECTIVITY,
    max_components: int = 4096,
    min_area: int = 1,
) -> ComponentIndex:
    """Extract protected GT components from one pair's argmax field.

    ``lstar`` is the (H, W) int GT SegNet argmax.  GT is STATIC per pair index, so
    this is computed once per pair and cached -- it is never on the training step's
    hot path (MEASURED 5.3 ms/frame, 3.2 s for the whole n600 corpus).
    """
    if lstar.ndim != 2:
        raise ValueError(f"lstar must be (H, W); got {lstar.shape}")
    h, w = lstar.shape
    px_chunks: list[np.ndarray] = []
    comp_chunks: list[np.ndarray] = []
    cls_chunks: list[np.ndarray] = []
    area_chunks: list[np.ndarray] = []
    next_ord = 0
    for cid in protected_classes:
        mask = lstar == cid
        if not mask.any():
            continue
        lab, n = _label_components(mask, connectivity)
        if n == 0:
            continue
        flat_lab = lab.reshape(-1)
        sel = np.nonzero(flat_lab)[0]
        labels_here = flat_lab[sel]                      # 1..n
        areas = np.bincount(labels_here, minlength=n + 1)[1:]
        keep_lab = np.nonzero(areas >= max(1, int(min_area)))[0] + 1
        if keep_lab.size == 0:
            continue
        remap = np.full(n + 1, -1, dtype=np.int64)
        remap[keep_lab] = np.arange(keep_lab.size, dtype=np.int64) + next_ord
        ordinals = remap[labels_here]
        keep_px = ordinals >= 0
        px_chunks.append(sel[keep_px].astype(np.int32))
        comp_chunks.append(ordinals[keep_px].astype(np.int32))
        cls_chunks.append(np.full(keep_lab.size, cid, dtype=np.int32))
        area_chunks.append(areas[keep_lab - 1].astype(np.int32))
        next_ord += int(keep_lab.size)
        if next_ord > max_components:
            raise ValueError(
                f"pair produced {next_ord} protected components > max_components="
                f"{max_components}. Refusing rather than silently truncating: a "
                "truncated component set would defend an arbitrary subset and the "
                "resulting verdict would be uninterpretable.")
    if next_ord == 0:
        empty_i = np.zeros((0,), dtype=np.int32)
        return ComponentIndex(empty_i, empty_i, empty_i, empty_i)
    return ComponentIndex(
        pixel_flat=np.concatenate(px_chunks),
        comp_of_px=np.concatenate(comp_chunks),
        comp_class=np.concatenate(cls_chunks),
        comp_area=np.concatenate(area_chunks),
    )


def component_weights(
    index: ComponentIndex, cfg: ExistenceHingeConfig
) -> np.ndarray:
    """Per-component loss weight ``w_c`` from each class's own policy."""
    if index.n_comp == 0:
        return np.zeros((0,), dtype=np.float32)
    w = np.ones((index.n_comp,), dtype=np.float32)
    area = index.comp_area.astype(np.float32)
    for cid in np.unique(index.comp_class):
        pol = cfg.policy_for(int(cid))
        sel = index.comp_class == cid
        if pol.weight_policy == "uniform":
            w[sel] = 1.0
        elif pol.weight_policy == "sqrt_area":
            w[sel] = np.sqrt(area[sel])
        elif pol.weight_policy == "area":
            w[sel] = area[sel]
        else:  # pragma: no cover - guarded by validate()
            raise ValueError(pol.weight_policy)
        s = w[sel].sum()
        if s > 0:
            # Normalize WITHIN each class so the class mix does not silently become a
            # second, unlabelled weighting knob (that is the governance-knob failure
            # mode: a knob that optimizes without being laddered).
            w[sel] = w[sel] * (float(sel.sum()) / float(s))
    return w


def component_betas_targets(
    index: ComponentIndex, cfg: ExistenceHingeConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Per-component ``beta`` and hinge ``target`` from the birth matrix."""
    n = index.n_comp
    betas = np.ones((n,), dtype=np.float32)
    targets = np.zeros((n,), dtype=np.float32)
    for cid in np.unique(index.comp_class) if n else ():
        pol = cfg.policy_for(int(cid))
        sel = index.comp_class == cid
        betas[sel] = pol.beta
        targets[sel] = pol.target
    return betas, targets


# ------------------------------------------------------------------------------
# The reduction (numpy reference -- the deterministic authority for tests)
# ------------------------------------------------------------------------------
def existence_scores_np(
    live_margin: np.ndarray, index: ComponentIndex, betas: np.ndarray
) -> np.ndarray:
    """``s(c) = logsumexp_beta`` over each component's pixels.  numpy reference.

    Stable by explicit max-shift; ``logsumexp`` is shift-invariant so the shift is
    exact rather than approximate.
    """
    if index.n_comp == 0:
        return np.zeros((0,), dtype=np.float32)
    m = live_margin.reshape(-1)[index.pixel_flat].astype(np.float64)
    b = betas.astype(np.float64)[index.comp_of_px]
    n = index.n_comp
    scaled = b * m
    cmax = np.full(n, -np.inf, dtype=np.float64)
    np.maximum.at(cmax, index.comp_of_px, scaled)
    acc = np.zeros(n, dtype=np.float64)
    np.add.at(acc, index.comp_of_px, np.exp(scaled - cmax[index.comp_of_px]))
    out = (cmax + np.log(acc)) / betas.astype(np.float64)
    return out.astype(np.float32)


def existence_hinge_np(
    live_margin: np.ndarray, index: ComponentIndex, cfg: ExistenceHingeConfig
) -> tuple[float, dict[str, Any]]:
    """Scalar existence-hinge loss + a telemetry row.  numpy reference.

    Telemetry is emitted UNCONDITIONALLY when the term is built (score-neutral
    observability defaults on): the per-class at-risk component counts are the only
    way to tell a term that is protecting words from a term that is silently inert.
    """
    if index.n_comp == 0:
        return 0.0, {"n_comp": 0, "at_risk": 0, "loss": 0.0, "per_class": {}}
    betas, targets = component_betas_targets(index, cfg)
    w = component_weights(index, cfg)
    s = existence_scores_np(live_margin, index, betas)
    viol = np.maximum(targets - s, 0.0)
    per_comp = w * viol
    loss = float(per_comp.sum() / max(1, index.n_comp))
    per_class: dict[str, Any] = {}
    for cid in np.unique(index.comp_class):
        sel = index.comp_class == cid
        pol = cfg.policy_for(int(cid))
        per_class[pol.class_name] = {
            "n_comp": int(sel.sum()),
            "at_risk": int((viol[sel] > 0).sum()),
            "mean_witness_margin": float(s[sel].mean()),
            "min_witness_margin": float(s[sel].min()),
        }
    return loss * float(cfg.weight), {
        "n_comp": int(index.n_comp),
        "at_risk": int((viol > 0).sum()),
        "loss": loss,
        "per_class": per_class,
    }


# ------------------------------------------------------------------------------
# The reduction (MLX -- the differentiable training path)
# ------------------------------------------------------------------------------
#: Additive mask value for non-member entries of the dense (K, n_comp) block.
#: Finite (not -inf) so that an all-masked column can never produce inf-inf = NaN.
_NEG_MASK = -1.0e9


def membership_mask_np(index: ComponentIndex) -> np.ndarray:
    """Dense additive (K, n_comp) membership mask: 0.0 for member, -1e9 otherwise."""
    k, n = index.n_px, index.n_comp
    mask = np.full((k, n), _NEG_MASK, dtype=np.float32)
    mask[np.arange(k), index.comp_of_px] = 0.0
    return mask


def live_margin_mlx(seg_logits: Any, lstar_oh: Any, mx: Any) -> Any:
    """Signed live multiclass margin ``gt_logit - runner_up``, shape (1, H, W).

    Deliberately the SAME idiom as the ``margin`` seg form
    (train_witness_realized_through_R_mlx.py L1450-1451) so this term reads the
    vehicle's real decision surface rather than a lookalike.
    """
    gt_logit = mx.sum(seg_logits * lstar_oh, axis=-1)
    runner_up = mx.max(seg_logits + lstar_oh * (-1e9), axis=-1)
    return gt_logit - runner_up


def existence_hinge_mlx(
    seg_logits: Any,
    lstar_oh: Any,
    pixel_flat: Any,
    membership_mask: Any,
    betas: Any,
    targets: Any,
    weights: Any,
    mx: Any,
) -> Any:
    """Differentiable existence-hinge term.  Returns a scalar mx array.

    All of ``pixel_flat`` / ``membership_mask`` / ``betas`` / ``targets`` /
    ``weights`` derive from STATIC GT and carry no gradient; the only differentiable
    input is ``seg_logits``.  Gradient therefore flows to exactly the witness pixel
    of each at-risk component (softly, with sharpness ``beta``), which is the
    intended and auditable behaviour of the primitive.

    ``mx`` is injected rather than imported so the module stays importable -- and
    unit-testable against the numpy reference -- on hosts without MLX.
    """
    n_comp = membership_mask.shape[1]
    if n_comp == 0:
        return mx.zeros(())
    m = live_margin_mlx(seg_logits, lstar_oh, mx).reshape(-1)
    mk = mx.take(m, pixel_flat)                       # (K,)
    blocked = mk[:, None] + membership_mask           # (K, n_comp)
    # mx.logsumexp supplies its own max-shift => stable without a scatter-max pass.
    s = mx.logsumexp(betas[None, :] * blocked, axis=0) / betas
    viol = mx.maximum(targets - s, 0.0)
    return mx.sum(weights * viol) / float(n_comp)
