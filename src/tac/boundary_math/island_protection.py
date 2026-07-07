# SPDX-License-Identifier: MIT
"""EARLY-SEED + CONTAINMENT + AMPLIFICATION — the finest-scale islands-protection stack.

WHAT THIS IS (operator directive 2026-07-01, the "finest-scale islands helpable"
4-lever stack: analytical-lane + EARLY-SEED + CONTAINMENT + AMPLIFICATION). The
level-set TASK-SPACE witness DISCOVERS the finest-scale erasure tail LATE (via
spectral-bias-slow training) or NEVER: the binding d_seg residual is the ISLANDS
— the thin class-1 (Lane) dash band and the small class-3 (Movable) blobs — which
are the lowest-persistence features (error ∝ 1/persistence, DAG FEED-dv/dw). This
module gives those islands three protections so they survive training:

  (a) EARLY-SEED  — a SPARSE, self-detected, per-pair RGB residual seeded at ep0
      from the GT island appearance (lane band + movable blobs), so the islands
      are BORN at step 0 instead of waiting for spectral-bias-slow discovery.
      (task #208 "rare-class-protected structured init, lane+movable seeded at ZERO".)
  (b) CONTAINMENT — a PROTECTED (decoupled) gradient projection on the seed
      residual so the BULK-class (Road/Undrivable/hood, ~97% of pixels) CE gradient
      cannot wash the seeded islands back below the argmax margin (HPRC protected-
      pathway / sparse-protected-residual v3). The protected residual is a SEPARATE
      param group -> composes with MD-Decoupling (#175) + does NOT touch the witness
      weight grads (grouped-backward ~17x unaffected).
  (c) AMPLIFICATION — a target-region-BIRTH loss term that RAISES the weak, low-
      persistence island-class logits ABOVE the argmax margin, riding the SAME
      canonical top1-top2 margin field the margin-saliency map (#141) uses.

SELF-DETECTION (NO-FAKE, never hardcode a class index — the FEED-dn mislabel guard):
  islands = the classes that are BOTH small-area (< area_max) AND temporally
  UNSTABLE (static-IoU < iou_max). On the REAL frozen-SegNet argmax cache (n96,
  MEASURED here 2026-07-01) that is EXACTLY class 1 (Lane: area 0.59%, IoU 0.00)
  and class 3 (Movable: area 1.56%, IoU 0.00) — NOT the bulk (Road 22.9%/IoU0.33,
  Undrivable 49.3%/IoU0.91, MyCar 25.6%/IoU0.96). Lane vs Movable is discriminated
  by THICKNESS (lane is a thin band -> small interior EDT; movable is a blob ->
  larger EDT) + lane-band geometric recall. The comma10k canonical order
  [Road,Lane,Undrivable,Movable,MyCar] is CONFIRMED by the detection but this
  module never assumes it — it detects from the data signature.

RATE HONESTY (respect the HPRC negative, NO-FAKE): a DENSE protected RGB sidecar
that SHIPS is rate-fatal (HPRC 600-pair rate term 2.643, sparse-v3 24,981 B WORSE
after entropy-position). So the EARLY-SEED here is a TRAINING-TIME ACCELERANT, not a
shipped dense residual: it seeds the protected pathway at ep0 so the witness's OWN
weights internalize the islands early; what SHIPS is the trained witness (+ optional
TINY PARAMETRIC geometry — lane centerline coeffs via ``lane_headstart``, movable
boxes — NOT dense RGB). The AMPLIFICATION birth term and the CONTAINMENT projection
are pure training dynamics (ZERO archive bytes). Any ship-cost is MEASURED separately
and is parametric, never the dense seed.

BORROWED-SUBSTRATE (CLAUDE.md NO-FAKE #7):
  * BORROWED / REUSED (composition, NOT duplication): the lane band geometry from
    ``lane_headstart`` (openpilot centerline fit + rasterizer) + ``lane_sdf_component``
    (rasterize_lane_band); the self-detect PATTERN from ``hood_static_component``
    (identify_static_hood_class); the canonical top1-top2 margin field from
    ``margin_saliency_map`` (#141); scipy EDT.
  * OURS-ORIGINAL: the JOINT lane+movable island detection + the training-time
    protected-residual EARLY-SEED + the decoupled CONTAINMENT projection + the
    inverse-persistence-weighted AMPLIFICATION birth term as ONE composable kit for
    the through-R witness.

COMPUTE (operator directive 2026-07-01, compute is a co-equal facet):
  * numpy is the DETERMINISTIC BIT-IDENTICAL AUTHORITY. The MLX (mx) forms of the
    two HOT paths — the amplification birth term (per-pair, per-epoch, on the LIVE
    (H,W,5) seg logits) and the containment grad projection — mirror the numpy
    reference op-for-op (parity >= 0.9997; exact on CPU). Both are fully VECTORIZED
    (no python pixel/pair loops) and the birth term is ``mx.compile``-able.
  * The self-detection + seed rasterization are ONE-TIME precompute (not hot);
    numpy-native (scipy EDT, connected components).

Evidence axis: the seed/mask/detection are deterministic geometry; the island-
SURVIVAL verdict is [contest-CPU advisory] (frozen CPU-torch SegNet argmax on the
real n600 GT via ``experiments/island_protection_survival_smoke.py``). This module
is training-time infrastructure: promotion_eligible=False; the pointer moves ONLY
through a byte-closed ``upstream/evaluate.py`` exact row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# WIRE-IN SPEC (the parent wires; this module is NOT self-installing).
# PRIMARY TARGET = the LIVE level-set trainer
# ``experiments/train_LEVELSET_witness_realized_through_R_mlx.py`` (it imports
# build_witness_module / apply_siren_init / make_loss_fn from the RGB base
# ``train_witness_realized_through_R_mlx.py``). All hooks ADDITIVE / opt-in
# (defaults OFF => byte-identical baseline):
#
#   (0) argparse: add the 9 flags in ISLAND_TRAINER_FLAG_DEFAULTS (booleans as
#       argparse.BooleanOptionalAction; --containment-mode / --amplify-form choices;
#       floats/ints). Defaults = the OFF values (opt-in like --structured-init / --lane-prior-phi1).
#
#   (a) EARLY-SEED — EXTEND the trainer's existing ``--structured-init`` (which today
#       seeds the phi/SDF target for LANE via the openpilot centerline; Movable ABSENT).
#       Use identify_island_classes + build_island_masks to ADD the MOVABLE (class-3)
#       seed target to the structured-init phi (movable = the net-new piece). For an
#       RGB-residual seed variant on the base witness: build_island_seed(gt_seg, masks,
#       base=ep0 render) -> a SEPARATE mx param ``seed_residual[pair]`` composed via the
#       base render_through_R_mlx compose_fn (rgb + seed_residual on the island band).
#
#   (b) CONTAINMENT — between value_and_grad and opt.update, on the SEED / structured-init
#       param grad ONLY (NOT the witness weight grads -> the ~17x grouped-backward is
#       untouched; the seed is its OWN optimizer group -> composes with MD-Decoupling #175):
#         g_seed = contain_protected_grad_mx(grads[<seed_param>], <seed_param>,
#                     ContainmentSpec(mode=args.containment_mode, damp=args.containment_damp,
#                                     protected_mask=masks.any_mask))
#
#   (c) AMPLIFICATION — RIDE THE INLINE LEVER-4 (do NOT add a 2nd margin-saliency / SegNet
#       forward). LEVER-4 already computes the SHARED realized signed margin ``_signed``
#       (R2b-M3) and adds ``msal_w * msal_term`` to ``L``. RIGHT NEXT TO IT, add the ISLAND
#       birth term on the SAME ``_signed`` (island×persistence weight, orthogonal to LEVER-4's
#       fragility×all-class weight):
#         L = L + amplify_weight * island_birth_from_signed_mx(_signed, island_weight,
#                     amplify_margin_target, form=amplify_form)
#       where island_weight (per pair, precomputed like margins) =
#         island_persistence_weight(masks.any_mask, kind=args.amplify_persist)  (0 off-island,
#         mean-1 over island). (amplify_weight 0 => L unchanged.) The base RGB trainer, which
#       has no _signed exposed, instead uses island_birth_term_mx(seg_logits, island_oh, weight,...).
#
# BULK-HARM GUARD (measure post-train): the birth term's weight is 0 off-island and mean-1
# ON island, so it RE-ALLOCATES budget to the finest scale, not adds; but softmax competition
# could still pull a boundary bulk pixel -> RE-MEASURE bulk (Road/Undrivable/hood) d_seg
# through-R after training. The $0 smoke's erasure model does not exercise this (paste-only).
# ---------------------------------------------------------------------------

# Class-signature thresholds MEASURED on the real frozen-SegNet argmax cache
# (experiments/results/mlx_fleet_gt_cache/gt_n96.npz, 2026-07-01). Islands are the
# small-area AND temporally-unstable classes; bulk is large-area OR static.
DEFAULT_AREA_MAX = 0.06        # island area fraction ceiling (Lane 0.006, Movable 0.016 << 0.06 << Road 0.23)
DEFAULT_STATIC_IOU_MAX = 0.20  # island static-IoU ceiling (Lane/Movable 0.00 << 0.20 << Undrivable 0.91)
_EPS = 1e-8


# ===========================================================================
# 1. SELF-DETECTION — never hardcode the class index (NO-FAKE FEED-dn guard).
# ===========================================================================
@dataclass
class IslandClassEvidence:
    """Per-class spatial/temporal signature used to pick the island classes."""

    cls: int
    area_frac: float          # mean per-frame area fraction
    static_iou: float         # IoU of the all-frames mask (staticity; low = unstable = island-like)
    v_centroid_frac: float    # vertical centroid row / H (0=top, 1=bottom)
    mean_thickness_px: float   # mean interior EDT depth of the class region (thin band -> small)
    px_per_frame: float
    is_island: bool
    island_kind: str | None    # "lane" | "movable" | None


@dataclass
class IslandDetection:
    """Result of self-detecting the island (finest-scale) classes."""

    lane_cls: int | None
    movable_cls: int | None
    island_classes: tuple[int, ...]
    evidence: tuple[IslandClassEvidence, ...]
    n_classes: int


def _mean_interior_thickness(mask_stack: np.ndarray) -> float:
    """Mean interior EDT depth (px) over the per-frame class masks — the THICKNESS.

    A thin band (lane) has small interior depth; a compact blob (movable) larger.
    Pure scipy EDT (the SAME construction lane_sdf/hood use). ``mask_stack`` (N,H,W)."""
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:  # pragma: no cover - scipy is a hard dep in-repo
        # Fallback: coarse thickness proxy = px / boundary-px (no scipy).
        a = np.asarray(mask_stack, bool)
        if not a.any():
            return 0.0
        # 4-neighbour boundary count
        b = np.zeros_like(a)
        b[:, 1:, :] |= a[:, 1:, :] & ~a[:, :-1, :]
        b[:, :-1, :] |= a[:, :-1, :] & ~a[:, 1:, :]
        b[:, :, 1:] |= a[:, :, 1:] & ~a[:, :, :-1]
        b[:, :, :-1] |= a[:, :, :-1] & ~a[:, :, 1:]
        return float(a.sum() / max(1, b.sum()))
    a = np.asarray(mask_stack, bool)
    depths: list[float] = []
    for i in range(a.shape[0]):
        if a[i].any():
            d = distance_transform_edt(a[i])
            depths.append(float(d[a[i]].mean()))
    return float(np.mean(depths)) if depths else 0.0


def identify_island_classes(
    lstars: np.ndarray,
    *,
    n_classes: int = 5,
    area_max: float = DEFAULT_AREA_MAX,
    static_iou_max: float = DEFAULT_STATIC_IOU_MAX,
) -> IslandDetection:
    """Detect the finest-scale ISLAND classes from the frozen-SegNet argmax cache.

    islands = classes that are BOTH small-area (< ``area_max``) AND temporally
    unstable (all-frames static-IoU < ``static_iou_max``). Among the detected
    islands, the THINNER (smaller interior EDT) is LANE, the thicker/blobbier is
    MOVABLE (ties broken by area: lane is smaller). Pure MEASUREMENT on the REAL
    L* (``lstars`` (N,H,W) int argmax) — the NO-FAKE guard against a hardcoded
    (possibly wrong) index. Returns ``IslandDetection`` (lane_cls/movable_cls may
    be None if the data has fewer than 2 island classes)."""
    a = np.asarray(lstars)
    if a.ndim != 3:
        raise ValueError(f"lstars must be (N,H,W) integer argmax, got shape {a.shape}")
    n, h, w = a.shape
    ev: list[IslandClassEvidence] = []
    for c in range(int(n_classes)):
        masks = a == c                       # (N,H,W)
        tot = int(masks.sum())
        if tot == 0:
            ev.append(IslandClassEvidence(c, 0.0, 0.0, 0.0, 0.0, 0.0, False, None))
            continue
        area = float(masks.mean())
        union = masks.any(0)
        inter = masks.all(0)
        iou = float(inter.sum() / union.sum()) if union.sum() else 0.0
        col_rows = masks.sum(axis=(0, 2))    # (H,) pixels per row summed over frames
        vcen = float((col_rows * np.arange(h)).sum() / max(1, col_rows.sum())) / h
        thick = _mean_interior_thickness(masks)
        is_isl = (area < area_max) and (iou < static_iou_max)
        ev.append(IslandClassEvidence(c, area, iou, vcen, thick, tot / n, is_isl, None))

    islands = [e for e in ev if e.is_island]
    lane_cls: int | None = None
    movable_cls: int | None = None
    if len(islands) == 1:
        # Single island: classify by thickness (thin band -> lane, else movable).
        only = islands[0]
        if only.mean_thickness_px <= 3.0:
            lane_cls = only.cls
            only.island_kind = "lane"
        else:
            movable_cls = only.cls
            only.island_kind = "movable"
    elif len(islands) >= 2:
        # Lane = thinnest (smallest interior EDT); movable = thickest. Tie -> smaller area = lane.
        by_thick = sorted(islands, key=lambda e: (e.mean_thickness_px, e.area_frac))
        lane_e = by_thick[0]
        movable_e = by_thick[-1]
        lane_e.island_kind = "lane"
        movable_e.island_kind = "movable"
        lane_cls, movable_cls = lane_e.cls, movable_e.cls
        # Any remaining islands beyond the two extremes are left kind=None (detected
        # but not assigned lane/movable) — honest: we protect the two canonical islands.
    return IslandDetection(
        lane_cls=lane_cls, movable_cls=movable_cls,
        island_classes=tuple(e.cls for e in islands),
        evidence=tuple(ev), n_classes=int(n_classes),
    )


# ===========================================================================
# 2. ISLAND MASKS — per-frame boolean masks (+ optional annulus dilation).
# ===========================================================================
@dataclass
class IslandMasks:
    """Per-frame island boolean masks on the argmax grid."""

    lane_mask: np.ndarray | None    # (H,W) bool or None
    movable_mask: np.ndarray | None  # (H,W) bool or None
    any_mask: np.ndarray            # (H,W) bool — union of all island pixels
    lane_cls: int | None
    movable_cls: int | None
    dilate_px: int


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, iterations=int(px))
    except Exception:  # pragma: no cover
        out = mask.copy()
        for _ in range(int(px)):
            s = out.copy()
            s[1:, :] |= out[:-1, :]
            s[:-1, :] |= out[1:, :]
            s[:, 1:] |= out[:, :-1]
            s[:, :-1] |= out[:, 1:]
            out = s
        return out


def build_island_masks(
    lstar: np.ndarray,
    lane_cls: int | None,
    movable_cls: int | None,
    *,
    dilate_px: int = 1,
) -> IslandMasks:
    """Per-frame island masks. ``dilate_px`` grows the mask to cover the codim-1
    argmax boundary annulus (the flip-prone ring around each island). ``lstar`` (H,W)."""
    a = np.asarray(lstar)
    if a.ndim != 2:
        raise ValueError(f"lstar must be (H,W), got {a.shape}")
    lane_m = _dilate(a == lane_cls, dilate_px) if lane_cls is not None else None
    mov_m = _dilate(a == movable_cls, dilate_px) if movable_cls is not None else None
    any_m = np.zeros(a.shape, bool)
    if lane_m is not None:
        any_m |= lane_m
    if mov_m is not None:
        any_m |= mov_m
    return IslandMasks(lane_m, mov_m, any_m, lane_cls, movable_cls, int(dilate_px))


def eased_island_masks(
    lstar: np.ndarray,
    lane_cls: int | None,
    movable_cls: int | None,
    *,
    dilate_px: int = 1,
    vanishing_point: tuple[float, float] | None = None,
) -> IslandMasks:
    """#323 LADDER per-class island homotopy — the manifold-aware drop-in alternative to
    ``build_island_masks``' isotropic ``_dilate`` (which grows BOTH classes the same way).

    The LADDER transfer proof splits the birth homotopy by class geometry:

    - **movable (blob class)** → ``sdf_dilation_eased``: the SDF forward-Euler homotopy
      (1-Lipschitz ⇒ Hausdorff-continuous nested filtration ⇒ bounded step-debt). For a
      blob its footprint equals isotropic dilation, but it is the PROVEN-transfer object.
    - **lane (curve class)** → ``oriented_width_eased``: grows each coherent segment ALONG
      its openpilot VP-tangent (road-forward) via a line structuring element, so the eased
      target stays a thin CURVE on the ~8-dim lane manifold. Isotropic dilation of a curve
      leaves the manifold (measured NO-GO for transfer). openpilot-grounded, rule-118-clean.

    Same return type + fields as ``build_island_masks`` so both amplify/seed call sites can
    switch on it behind a default-OFF flag (byte-identical when unfired). ``dilate_px`` is
    reused as the growth radius (movable) / along-tangent width (lane)."""
    from tac.witness_curriculum.eased_targets import (
        oriented_width_eased as _owe,
        sdf_dilation_eased as _sde,
    )
    a = np.asarray(lstar)
    if a.ndim != 2:
        raise ValueError(f"lstar must be (H,W), got {a.shape}")
    lane_m = None
    if lane_cls is not None:
        lane_m = (_owe(a, lane_cls, int(dilate_px), vanishing_point=vanishing_point) == lane_cls)
    mov_m = None
    if movable_cls is not None:
        mov_m = (_sde(a, movable_cls, int(dilate_px)) == movable_cls)
    any_m = np.zeros(a.shape, bool)
    if lane_m is not None:
        any_m |= lane_m
    if mov_m is not None:
        any_m |= mov_m
    return IslandMasks(lane_m, mov_m, any_m, lane_cls, movable_cls, int(dilate_px))


# ===========================================================================
# 3. EARLY-SEED — the sparse protected RGB residual seeded from GT island appearance.
# ===========================================================================
@dataclass
class IslandSeed:
    """The ep0 sparse protected-residual seed for ONE pair-frame."""

    residual: np.ndarray        # (H,W,3) float32 additive RGB residual (0 off islands)
    mask: np.ndarray            # (H,W) bool — the protected support (island pixels)
    support_frac: float         # fraction of pixels that carry seed
    lane_cls: int | None
    movable_cls: int | None


def build_island_seed(
    gt_frame_segres: np.ndarray,
    masks: IslandMasks,
    *,
    base_render_segres: np.ndarray | None = None,
    blend: float = 1.0,
) -> IslandSeed:
    """Build the sparse additive RGB residual that BIRTHS the islands at ep0.

    At island pixels the composed render should equal the GT frame RGB (which the
    FROZEN SegNet already classifies correctly by construction), so the seed value
    is ``blend*(gt_rgb - base)`` restricted to the island mask; 0 elsewhere. With
    ``base_render_segres=None`` the residual IS the GT island appearance (base=0).
    Deterministic, numpy. ``gt_frame_segres`` (H,W,3) float/uint8 at the argmax grid.

    This is the "sparse protected residual" (HPRC v3): the seed is VIDEO-DERIVED
    (COUNTED if it ships) but SPARSE (only the island band, ~2% of pixels). It is a
    training-time INIT of the protected pathway; the containment keeps it, the
    amplification refines it."""
    gt = np.asarray(gt_frame_segres, dtype=np.float32)
    if gt.ndim != 3 or gt.shape[-1] != 3:
        raise ValueError(f"gt_frame_segres must be (H,W,3), got {gt.shape}")
    m = np.asarray(masks.any_mask, bool)
    if m.shape != gt.shape[:2]:
        raise ValueError(f"mask {m.shape} != frame grid {gt.shape[:2]}")
    base = np.zeros_like(gt) if base_render_segres is None else np.asarray(base_render_segres, dtype=np.float32)
    residual = np.zeros_like(gt)
    residual[m] = float(blend) * (gt[m] - base[m])
    return IslandSeed(
        residual=residual.astype(np.float32), mask=m,
        support_frac=float(m.mean()), lane_cls=masks.lane_cls, movable_cls=masks.movable_cls,
    )


def compose_seed(base_rgb: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Compose base render + protected seed residual (the ep0 render). Exact inverse
    of ``build_island_seed`` when base matches: ``compose_seed(base, seed)[m] == gt[m]``."""
    return (np.asarray(base_rgb, dtype=np.float32) + np.asarray(residual, dtype=np.float32))


# ===========================================================================
# 4. CONTAINMENT — decoupled protected-pathway gradient projection.
# ===========================================================================
_CONTAIN_MODES = ("freeze", "damp", "shield")


@dataclass
class ContainmentSpec:
    """How the protected seed residual is defended from the bulk gradient wash."""

    mode: str = "shield"        # "freeze" | "damp" | "shield"
    damp: float = 0.1           # scale for "damp" (0=freeze, 1=off); "shield" ignores
    protected_mask: np.ndarray | None = None  # (H,W) bool support; None => whole param

    def __post_init__(self) -> None:
        if self.mode not in _CONTAIN_MODES:
            raise ValueError(f"mode must be one of {_CONTAIN_MODES}, got {self.mode!r}")
        if not (0.0 <= float(self.damp) <= 1.0):
            raise ValueError(f"damp must be in [0,1], got {self.damp}")


def contain_protected_grad_np(
    grad: np.ndarray,
    residual: np.ndarray,
    spec: ContainmentSpec,
) -> np.ndarray:
    """Project the SEED-residual gradient so the bulk wash cannot erase the islands.

    NUMPY REFERENCE (authority). ``grad`` and ``residual`` are the protected param's
    gradient and current value (same shape, e.g. (H,W,3)); ``spec.protected_mask``
    (H,W) selects the protected pixels (None => all). Modes:

      freeze : zero the gradient on protected pixels (seed is FROZEN at its GT value).
      damp   : scale the protected gradient by ``spec.damp`` (slow refinement).
      shield : zero ONLY the DESTRUCTIVE gradient component — the part that shrinks
               the seed toward 0 (the bulk-CE wash direction). A gradient-descent
               step is ``r -= lr*g``; it shrinks |r| when ``g`` points along
               ``sign(r)`` (same sign as the residual). ``shield`` removes exactly
               that same-sign, magnitude-shrinking part (``g_shield = g -
               relu(g*sign(r))*sign(r)`` on protected pixels), so refinements that
               GROW / re-shape the seed still flow but the erase-to-bulk direction
               is contained. Off-protected pixels are untouched (bulk trains freely).

    This operates on a SEPARATE (decoupled) param -> composes with MD-Decoupling
    (#175) and never touches the witness weight grads (grouped-backward unaffected).
    """
    g = np.asarray(grad, dtype=np.float32)
    r = np.asarray(residual, dtype=np.float32)
    if g.shape != r.shape:
        raise ValueError(f"grad {g.shape} != residual {r.shape}")
    m = spec.protected_mask
    if m is None:
        sel = np.ones(g.shape[:2], bool) if g.ndim >= 2 else np.ones_like(g, bool)
    else:
        sel = np.asarray(m, bool)
        if sel.shape != g.shape[:sel.ndim]:
            raise ValueError(f"protected_mask {sel.shape} incompatible with grad {g.shape}")
    # broadcast the (H,W) mask across trailing channel dim
    bsel = sel[..., None] if (g.ndim == sel.ndim + 1) else sel
    out = g.copy()
    if spec.mode == "freeze":
        out = np.where(bsel, 0.0, out)
    elif spec.mode == "damp":
        out = np.where(bsel, out * float(spec.damp), out)
    else:  # shield
        sgn = np.sign(r)
        destructive = np.maximum(g * sgn, 0.0) * sgn  # same-sign (shrink-|r|) component
        shielded = g - destructive
        out = np.where(bsel, shielded, out)
    return out.astype(np.float32)


def contain_protected_grad_mx(grad: Any, residual: Any, spec: ContainmentSpec) -> Any:
    """MLX mirror of ``contain_protected_grad_np`` (bit-identical ops; VECTORIZED).

    ``grad``/``residual`` are mx arrays; ``spec.protected_mask`` is (H,W) numpy/bool
    (converted to mx once). No python loops. Cheap elementwise -> negligible next to
    the scorer forward, and it does NOT gate the grouped-backward fast path."""
    import mlx.core as mx

    g = grad
    r = residual
    m = spec.protected_mask
    if m is None:
        bsel = None
    else:
        sel = mx.array(np.asarray(m, bool))
        bsel = sel[..., None] if (len(g.shape) == len(sel.shape) + 1) else sel
    if spec.mode == "freeze":
        out = mx.zeros_like(g) if bsel is None else mx.where(bsel, mx.zeros_like(g), g)
    elif spec.mode == "damp":
        d = float(spec.damp)
        out = g * d if bsel is None else mx.where(bsel, g * d, g)
    else:  # shield
        sgn = mx.sign(r)
        destructive = mx.maximum(g * sgn, 0.0) * sgn
        shielded = g - destructive
        out = shielded if bsel is None else mx.where(bsel, shielded, g)
    return out


# ===========================================================================
# 5. AMPLIFICATION — target-region-BIRTH term (rides the canonical margin field #141).
# ===========================================================================
def island_one_hot(lstar: np.ndarray, island_mask: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """One-hot of the GT island class at island pixels, zero elsewhere. (H,W,C) float32.

    At island pixels the GT argmax IS the island class (lane_cls / movable_cls), so
    the birth term's target = the class the pixel SHOULD win. Zero rows off-island."""
    a = np.asarray(lstar)
    oh = np.eye(int(n_classes), dtype=np.float32)[a]           # (H,W,C)
    oh = oh * np.asarray(island_mask, np.float32)[..., None]    # zero off islands
    return oh.astype(np.float32)


def island_persistence_weight(island_mask: np.ndarray, *, kind: str = "inverse_thickness") -> np.ndarray:
    """Per-pixel AMPLIFICATION weight — HIGH on the lowest-persistence island pixels.

    error ∝ 1/persistence (FEED-dv): the thinnest / finest-scale island pixels are
    erased first, so amplification concentrates there. ``inverse_thickness``: EDT
    interior depth ``d`` (thick=persistent), weight ``1/(1+d)`` on the island, then
    MEAN-1 normalized OVER the island pixels (preserves the birth budget; RE-allocated
    to the finest scale, not added). ``uniform``: 1 on the island. Returns (H,W) f32,
    ZERO off the island. Precomputed from GT (stop-grad prior, NO-FAKE)."""
    m = np.asarray(island_mask, bool)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    if kind == "uniform":
        w = m.astype(np.float32)
    elif kind == "inverse_thickness":
        try:
            from scipy.ndimage import distance_transform_edt
            d = distance_transform_edt(m).astype(np.float32)
        except Exception:  # pragma: no cover
            d = m.astype(np.float32)
        w = np.where(m, 1.0 / (1.0 + d), 0.0).astype(np.float32)
    else:
        raise ValueError(f"kind must be uniform|inverse_thickness, got {kind!r}")
    mean_on = float(w[m].mean()) if m.any() else 0.0
    if mean_on > _EPS:
        w = np.where(m, w / mean_on, 0.0).astype(np.float32)   # mean-1 over island support
    return w.astype(np.float32)


_BIRTH_FORMS = ("hinge", "softplus")


def island_birth_term_np(
    seg_logits: np.ndarray,
    island_oh: np.ndarray,
    weight: np.ndarray,
    margin_target: float,
    *,
    form: str = "hinge",
    tau: float = 0.3,
) -> float:
    """Target-region-BIRTH loss (NUMPY reference / authority).

    Rides the canonical top1-top2 margin field (#141): the SIGNED island margin is
    ``signed = logit[island_cls] - max_{c != island_cls} logit[c]`` (want >
    ``margin_target`` so the island WINS its pixels by a margin). The birth penalty
    RAISES the weak island logit above the argmax margin, weighted by ``weight``
    (inverse-persistence, mean-1 over island), averaged over the island pixels
    (weight is 0 off-island). Same construction as the trainer's ``_live_signed()``
    seg loss + the landed target-region-birth actuator, specialized to the self-
    detected island class. Two forms of the penalty on the deficit ``z = margin_target
    - signed``:

      hinge    : ``relu(z)``              (L1; matches the trainer's ``margin_hinge`` seg form)
      softplus : ``tau*softplus(z/tau)``  (smooth; matches PR95 tau-softplus / the
                 target-region-birth frontier-crossing term — nonzero gradient even
                 slightly PAST the margin, so births keep hardening)

    ``seg_logits`` (...,H,W,C); ``island_oh`` (...,H,W,C); ``weight`` (...,H,W)."""
    if form not in _BIRTH_FORMS:
        raise ValueError(f"form must be one of {_BIRTH_FORMS}, got {form!r}")
    lg = np.asarray(seg_logits, dtype=np.float32)
    oh = np.asarray(island_oh, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    gt_logit = np.sum(lg * oh, axis=-1)                        # (...,H,W) island-class logit
    runner_up = np.max(lg + oh * (-1e9), axis=-1)              # (...,H,W) best competitor
    signed = gt_logit - runner_up
    z = float(margin_target) - signed                         # deficit (>0 => island losing/near-flip)
    if form == "softplus":
        t = max(float(tau), 1e-6)
        birth = t * np.logaddexp(0.0, z / t)                  # tau*softplus(z/tau)
    else:
        birth = np.maximum(z, 0.0)
    num = float(np.sum(birth * w))
    den = float(np.sum(w)) + _EPS
    return num / den


def island_birth_term_mx(seg_logits: Any, island_oh: Any, weight: Any, margin_target: float,
                         *, form: str = "hinge", tau: float = 0.3) -> Any:
    """MLX mirror of ``island_birth_term_np`` (bit-identical ops; VECTORIZED, mx.compile-able).

    Differentiable through the FROZEN MLX SegNet -> the gradient IS the margin-
    saliency-driven birth update (#141). No python loops; one fused reduction."""
    import mlx.core as mx

    gt_logit = mx.sum(seg_logits * island_oh, axis=-1)
    runner_up = mx.max(seg_logits + island_oh * (-1e9), axis=-1)
    signed = gt_logit - runner_up
    z = float(margin_target) - signed
    if form == "softplus":
        t = max(float(tau), 1e-6)
        birth = t * mx.logaddexp(mx.zeros_like(z), z / t)
    else:
        birth = mx.maximum(z, 0.0)
    num = mx.sum(birth * weight)
    den = mx.sum(weight) + _EPS
    return num / den


def island_birth_from_signed_np(
    signed: np.ndarray, weight: np.ndarray, margin_target: float, *, form: str = "hinge", tau: float = 0.3,
) -> float:
    """Island-birth term that RIDES a PRE-COMPUTED signed realized margin (NUMPY authority).

    This is the composition variant for the LEVELSET trainer: its LEVER-4 (margin-saliency)
    already computes the SHARED realized decision margin ``_signed = target_logit -
    max_competing_logit`` (R2b-M3, ONE forward). This term REUSES that exact tensor — it does
    NOT recompute a second margin / saliency / SegNet forward — and adds an ISLAND-weighted
    (inverse-persistence, class-specific) hinge orthogonal to LEVER-4's fragility (all-class)
    weight. ``signed`` (...,H,W); ``weight`` (...,H,W) island×persistence, 0 off-island."""
    if form not in _BIRTH_FORMS:
        raise ValueError(f"form must be one of {_BIRTH_FORMS}, got {form!r}")
    s = np.asarray(signed, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    z = float(margin_target) - s
    if form == "softplus":
        t = max(float(tau), 1e-6)
        birth = t * np.logaddexp(0.0, z / t)
    else:
        birth = np.maximum(z, 0.0)
    return float(np.sum(birth * w)) / (float(np.sum(w)) + _EPS)


def island_birth_from_signed_mx(signed: Any, weight: Any, margin_target: float,
                                *, form: str = "hinge", tau: float = 0.3) -> Any:
    """MLX mirror of ``island_birth_from_signed_np`` — the LEVER-4 composition (rides the
    SHARED ``_signed``; bit-identical ops, VECTORIZED). Add its output onto the trainer's
    loss ``L`` next to ``msal_term``; the gradient rides the same realized margin field."""
    import mlx.core as mx

    z = float(margin_target) - signed
    if form == "softplus":
        t = max(float(tau), 1e-6)
        birth = t * mx.logaddexp(mx.zeros_like(z), z / t)
    else:
        birth = mx.maximum(z, 0.0)
    return mx.sum(birth * weight) / (mx.sum(weight) + _EPS)


def make_island_birth_term_mx_compiled():
    """Return an ``mx.compile``-wrapped birth term ``f(seg_logits, island_oh, weight,
    margin_target_arr) -> scalar`` for the hot inner loop. ``margin_target`` is passed
    as a 0-d mx array so the compiled graph does not recompile per epoch."""
    import mlx.core as mx

    def _f(seg_logits, island_oh, weight, margin_target_arr):
        gt_logit = mx.sum(seg_logits * island_oh, axis=-1)
        runner_up = mx.max(seg_logits + island_oh * (-1e9), axis=-1)
        signed = gt_logit - runner_up
        birth = mx.maximum(margin_target_arr - signed, 0.0)
        return mx.sum(birth * weight) / (mx.sum(weight) + _EPS)

    return mx.compile(_f)


# ===========================================================================
# 6. Recall diagnostics (used by the $0 survival smoke; pure numpy).
# ===========================================================================
def island_recall(pred_argmax: np.ndarray, lstar: np.ndarray, island_cls: int) -> float:
    """Fraction of GT island-``cls`` pixels the prediction also labels ``cls``.

    Recall (not IoU) isolates BIRTH: did the island class survive at its GT pixels?"""
    gt = np.asarray(lstar) == int(island_cls)
    if not gt.any():
        return float("nan")
    pr = np.asarray(pred_argmax) == int(island_cls)
    return float((pr & gt).sum() / gt.sum())


# ===========================================================================
# 7. TRIALITY / DSL leg — the trainer flags this kit is driven by (net-new).
# ===========================================================================
# The canonical trainer-flag set the parent WIRES INTO the argparse of the RGB
# witness trainer (experiments/train_witness_realized_through_R_mlx.py). Each is
# NET-NEW (no existing string collides; confirmed 2026-07-01). Booleans MUST be
# argparse.BooleanOptionalAction (so the DSL can emit --no-<flag> without crashing,
# per curriculum_dsl.real_store_true_flags). This dict is the single source of the
# flag NAMES + the ARGPARSE-DEFAULT VALUES (all OFF => byte-identical baseline, opt-in
# like --lane-prior-phi1). ``island_protection_flags()`` returns the ENABLED (ON)
# values for building an A/B lever -- do NOT confuse the two: DEFAULTS = trainer
# argparse defaults (OFF); island_protection_flags() = the lever that turns it ON.
ISLAND_TRAINER_FLAG_DEFAULTS: dict[str, Any] = {
    "--seed-islands": False,          # BooleanOptionalAction: seed the protected island residual at ep0
    "--seed-island-eased": False,     # BooleanOptionalAction (#323 LADDER): class-aware eased masks (SDF movable / VP-tangent lane) instead of isotropic dilation
    "--island-dilate-px": 1,          # int: annulus dilation of the island masks
    "--containment-mode": "shield",   # choices freeze|damp|shield: how the seed grad is protected
    "--containment-damp": 0.1,        # float: damp factor for --containment-mode damp
    "--amplify-weight": 0.0,          # float: weight of the island-birth term added to the seg loss (0=off)
    "--amplify-form": "hinge",        # choices hinge|softplus: birth penalty form
    "--amplify-margin-target": 1.0,   # float: the margin the island must WIN its pixels by
    "--amplify-persist": "inverse_thickness",  # choices uniform|inverse_thickness: birth weight
}


def island_protection_flags(
    *,
    seed_islands: bool = True,
    seed_island_eased: bool = False,
    dilate_px: int = 1,
    containment_mode: str = "shield",
    containment_damp: float = 0.1,
    amplify_weight: float = 0.5,
    amplify_form: str = "hinge",
    amplify_margin_target: float = 1.0,
    amplify_persist: str = "inverse_thickness",
) -> dict[str, Any]:
    """Return the canonical trainer flag→value dict for this kit (the DSL leg source).

    Validated types match ``ISLAND_TRAINER_FLAG_DEFAULTS``. The DSL renders bools as
    ``--flag`` / ``--no-flag`` and valued flags as ``[flag, str(val)]``."""
    if containment_mode not in _CONTAIN_MODES:
        raise ValueError(f"containment_mode must be {_CONTAIN_MODES}, got {containment_mode!r}")
    if amplify_form not in _BIRTH_FORMS:
        raise ValueError(f"amplify_form must be {_BIRTH_FORMS}, got {amplify_form!r}")
    if amplify_persist not in ("uniform", "inverse_thickness"):
        raise ValueError(f"amplify_persist must be uniform|inverse_thickness, got {amplify_persist!r}")
    return {
        "--seed-islands": bool(seed_islands),
        "--seed-island-eased": bool(seed_island_eased),
        "--island-dilate-px": int(dilate_px),
        "--containment-mode": str(containment_mode),
        "--containment-damp": float(containment_damp),
        "--amplify-weight": float(amplify_weight),
        "--amplify-form": str(amplify_form),
        "--amplify-margin-target": float(amplify_margin_target),
        "--amplify-persist": str(amplify_persist),
    }


def build_island_protection_lever(name: str = "islands_protect", *, epochs_delta: int = 0, **kw):
    """Return a ``tac.witness_dsl.curriculum_dsl.Lever`` for the A/B campaign (DSL leg).

    Lazy import so this module never hard-depends on the DSL. ``kw`` forwards to
    ``island_protection_flags``. The lever's ``overrides`` are the trainer flags;
    the DSL's ``validate()`` structurally proves they exist in the trainer argparse
    (they must be wired per the WIRE-IN spec first)."""
    from tac.witness_dsl.curriculum_dsl import Lever

    return Lever(
        name=name,
        overrides=island_protection_flags(**kw),
        epochs_delta=int(epochs_delta),
        notes="EARLY-SEED + CONTAINMENT + AMPLIFICATION islands-protection stack "
              "(tac.boundary_math.island_protection); FEED-lz.",
    )


# ===========================================================================
# 8. COMPUTE — benchmark + a FLAGGED fused Metal kernel opportunity (#212).
# ===========================================================================
# The hot per-pair path is the birth term's fused (signed -> deficit -> softplus/relu ->
# weighted mean) reduction over (H,W). It is already a single mx.compile'd reduction
# (0.86 ms/pair at (384,512,5); n600 = 0.51 s). If a full-stack sweep makes it a bottleneck,
# fuse the deficit+softplus+weighted-sum into ONE Metal kernel (avoids the intermediate
# (H,W) birth map materialization). FLAGGED, not yet built (mirrors the FEED-pt persistence-
# pool kernel flag). Toggle: env TAC_MLX_CUSTOM_ISLAND_BIRTH=1.
TAC_MLX_CUSTOM_ISLAND_BIRTH_ENV = "TAC_MLX_CUSTOM_ISLAND_BIRTH"


def metal_island_birth_kernel_signature() -> dict[str, Any]:
    """The (proposed) fused Metal kernel contract for the birth term (#212 compute facet).

    inputs : signed (H,W) f32, weight (H,W) f32, margin_target f32, tau f32, form u8 (0=hinge,1=softplus)
    output : scalar f32 = sum(birth(margin_target-signed) * weight) / (sum(weight)+eps)
    grid   : one threadgroup per row-tile; threadgroup reduction -> atomic add to two f32 accumulators
             (num, den). Bit-target: match island_birth_from_signed_np within 1e-5 (fp32 reduction order).
    NOT YET BUILT — the mx.compile'd path is the current authority; this is the fusion spec."""
    return {
        "kernel": "island_birth_reduce",
        "env_flag": TAC_MLX_CUSTOM_ISLAND_BIRTH_ENV,
        "inputs": ["signed:f32[H,W]", "weight:f32[H,W]", "margin_target:f32", "tau:f32", "form:u8"],
        "output": "scalar:f32",
        "reference": "tac.boundary_math.island_protection.island_birth_from_signed_np",
        "parity_target": 1e-5,
        "status": "FLAGGED_NOT_BUILT",
    }


# ===========================================================================
# 9. COMPUTE benchmark — MLX-GPU vs numpy at n600 scale (co-equal facet).
# ===========================================================================
def benchmark_amplification(*, n: int = 600, h: int = 384, w: int = 512, c: int = 5,
                            reps: int = 3, seed: int = 0) -> dict:
    """Benchmark the AMPLIFICATION birth term (the hot per-pair path) MLX-GPU vs numpy
    at n600 scale + report parity. Deterministic synthetic logits; the shapes match
    the real (H,W,C) live seg logits. Returns timings (s/pair) + parity (max relative
    diff). NO scorer, NO GPU dispatch — a local micro-benchmark."""
    rng = np.random.default_rng(seed)
    lstar = np.zeros((h, w), np.int64)
    lstar[h // 3 : 2 * h // 3, w // 2] = 1                     # a lane stripe
    lstar[h // 2 : h // 2 + 6, 10:16] = 3                      # a movable blob
    mask = (lstar == 1) | (lstar == 3)
    oh = island_one_hot(lstar, mask)
    weight = island_persistence_weight(mask)
    logits = rng.standard_normal((h, w, c)).astype(np.float32)

    import time

    # numpy authority
    t = time.time()
    for _ in range(reps * 4):
        _np = island_birth_term_np(logits, oh, weight, 0.5)
    np_s = (time.time() - t) / (reps * 4)

    out: dict[str, Any] = {"n": n, "grid": [h, w, c], "numpy_s_per_pair": round(np_s, 6),
                            "numpy_n600_s": round(np_s * n, 4)}
    try:
        import mlx.core as mx

        f = make_island_birth_term_mx_compiled()
        oh_mx, w_mx, lg_mx, mt = mx.array(oh), mx.array(weight), mx.array(logits), mx.array(0.5, dtype=mx.float32)
        mx.eval(f(lg_mx, oh_mx, w_mx, mt))  # warm compile
        t = time.time()
        for _ in range(reps * 4):
            v = f(lg_mx, oh_mx, w_mx, mt); mx.eval(v)
        mx_s = (time.time() - t) / (reps * 4)
        out["mlx_compiled_s_per_pair"] = round(mx_s, 6)
        out["mlx_n600_s"] = round(mx_s * n, 4)
        out["parity_max_rel_diff"] = float(abs(float(v) - _np) / (abs(_np) + 1e-9))
        out["parity_ok_0.9997"] = bool(out["parity_max_rel_diff"] < 3e-4)
    except Exception as exc:  # pragma: no cover
        out["mlx"] = f"unavailable: {exc}"
    return out


__all__ = [
    "DEFAULT_AREA_MAX",
    "DEFAULT_STATIC_IOU_MAX",
    "IslandClassEvidence",
    "IslandDetection",
    "identify_island_classes",
    "IslandMasks",
    "build_island_masks",
    "eased_island_masks",
    "IslandSeed",
    "build_island_seed",
    "compose_seed",
    "ContainmentSpec",
    "contain_protected_grad_np",
    "contain_protected_grad_mx",
    "island_one_hot",
    "island_persistence_weight",
    "island_birth_term_np",
    "island_birth_term_mx",
    "island_birth_from_signed_np",
    "island_birth_from_signed_mx",
    "make_island_birth_term_mx_compiled",
    "island_recall",
    "benchmark_amplification",
    "metal_island_birth_kernel_signature",
    "TAC_MLX_CUSTOM_ISLAND_BIRTH_ENV",
    "ISLAND_TRAINER_FLAG_DEFAULTS",
    "island_protection_flags",
    "build_island_protection_lever",
]
