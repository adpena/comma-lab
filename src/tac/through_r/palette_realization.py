# SPDX-License-Identifier: MIT
"""palette_realization — is the naive-palette realized ceiling (F=0.0337) a PALETTE ARTIFACT?

Operator probe (2026-07-09): the C1 transfer-ceiling row realized the PERFECT direct partition
(GT ``L*``) as per-pair per-class MEAN-RGB and pushed it through R -> frozen CPU-torch SegNet,
getting realized d_seg ``F = 0.0337`` (``negaudit_retests_c1_e5_20260709.md``). *"That seems like a
big number which should be smaller indicating something we are overlooking."*

THE HYPOTHESIS. ``0.0337`` is a PALETTE ARTIFACT, not an R wall. Proof sketch: painting a partition
introduces TWO flip sources the partition itself does not have — (1) **third-class mixing**: R's
bicubic-UP + bilinear-DOWN + uint8 blend two adjacent class colors ``c_i, c_j`` into a mixture that
lands in a THIRD class ``k``'s RGB decision region (manufactured — a class that is not even locally
present); (2) **genuine boundary-position jitter**: even a perfect palette shifts the argmax boundary
by a sub-pixel (the #333 annulus jitter — real, irreducible under R). The palette is a FREE design
variable we treated as fixed (per-class MEAN). Choosing colors whose pairwise mixtures stay
argmax-correct removes source (1). The U1 canary (real GT frames through the SAME R give d_seg
EXACTLY 0) proves ALL of ``0.0337`` comes from the painting, not R.

THE ARMS (all n600, same L* target, canonical :func:`tac.through_r.measure_through_r` authority path):
  1. **baseline** — per-pair per-class MEAN-RGB, render-grid, full R (reproduce ``0.0337``).
  2. **mixing-robust palette** — ONE global 5-colour palette, optimised (from the SegNet decision
     geometry) so every pairwise mixture ``alpha*c_i+(1-alpha)*c_j`` stays argmax-correct for the
     nearer class as long as possible; render-grid, full R.
  3. **camera-res paint** (#149) — nearest-up ``L*`` -> camera 874x1164 hard palette paint (R.up a
     no-op) so ONLY the bilinear-DOWN mixes; best palette.
  4. **boundary-snapped** — hard palette paint at the SegNet grid directly from ``L*`` (zero
     resolution mixing; SegNet ``preprocess_input`` at seg-res is EXACT identity), best palette. The
     irreducible floor: SegNet's own disagreement with a perfectly hard partition + palette-colour
     context-dependence.

THE DECOMPOSITION IS THE FINDING (per arm): of the realized flips, how many are **third-class-mixing**
(realized class NOT present in the local ``L*`` neighbourhood — manufactured by the palette) vs
**boundary-position** (realized class IS a locally-adjacent class — a real ±1px jitter). The drop
``F_baseline -> F_boundary_snapped`` is the manufactured share; the residual is the real floor.

NO-FAKE / REUSE-not-rederive: the R operator + frozen SegNet + aggregate/per-class compare are the
canonical :mod:`tac.through_r` authority (``render_grid_to_camera_uint8`` +
``harness._segnet_argmax_chunked`` + ``compare_label_stack_to_lstars``). This module adds ONLY the
palette-realisation + decision-geometry + flip-decomposition logic on top; it re-derives NOTHING.
Authority: ``[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]`` — a realized d_seg
is NEVER a score; the pointer (contest-CPU 0.19110) moves only through byte-closed exact eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.through_r.compare import compare_label_stack_to_lstars
from tac.through_r.harness import (
    THROUGH_R_LABEL,
    ThroughRResult,
    _segnet_argmax_chunked,
    load_frozen_segnet,
)
from tac.through_r.resolution_chain import (
    CAMERA_H,
    CAMERA_W,
    SEG_H,
    SEG_W,
    render_grid_to_camera_uint8,
)
from tac.witness_control.perclass_verdict import CLASS_NAMES, N_CLASSES

__all__ = [
    "ArmResult",
    "DecisionGeometry",
    "PaletteRealizationError",
    "classify_color",
    "decompose_flips",
    "map_decision_geometry",
    "mixing_robust_palette",
    "naive_mean_palette",
    "paint_camera_res_uint8",
    "paint_render_grid",
    "paint_seg_grid",
    "realize_argmax_no_R",
    "run_arm",
]


class PaletteRealizationError(ValueError):
    """Raised on a mis-shaped / toy / non-authority palette-realisation input."""


# --------------------------------------------------------------------------- #
# Decision geometry: probe the frozen SegNet with constant-colour tiles.       #
# --------------------------------------------------------------------------- #
@dataclass
class DecisionGeometry:
    """The measured RGB -> SegNet response map (context-free, constant-tile probe).

    ``colors`` ``(K,3)`` float[0,255] probed colours; ``mean_logits`` ``(K, n_classes)`` the mean
    over the SegNet logit field for a constant tile of that colour; ``argmax_class`` ``(K,)`` the
    modal class a constant tile of that colour decodes to (``mean_logits.argmax(1)``).

    A constant colour is R-INVARIANT (bicubic-up of a constant is constant, uint8 exact for integer
    levels, bilinear-down constant), so the constant-tile probe is identical whether measured through
    full R or the identity seg-res path — we use the seg-res path (cheap). The map is CONTEXT-FREE:
    SegNet is a U-Net with spatial receptive field, so a colour's class at a specific position can
    differ. That approximation is DOCUMENTED — the palette is optimised against this map, but every
    arm's F is measured through the REAL SegNet (the authority), so a context effect surfaces as a
    non-zero boundary-snapped floor, never as a hidden fake.
    """

    colors: np.ndarray
    mean_logits: np.ndarray
    argmax_class: np.ndarray
    n_classes: int
    grid_levels: int

    def best_color_for_class(self, c: int) -> np.ndarray:
        """The probed colour that MOST strongly evokes class ``c`` (max mean logit).

        Uses the logit (not argmax) so a context-only class (Lane/Movable, which may NEVER win a
        constant-tile argmax) still gets a representative colour — the colour that maximally pushes
        SegNet toward ``c``.
        """
        return self.colors[int(np.argmax(self.mean_logits[:, int(c)]))].astype(np.float32)


def _seg_logits_mean(segnet: Any, seg_frames: list[np.ndarray], chunk: int) -> np.ndarray:
    """Mean SegNet logit vector per constant/hard seg-res frame ``-> (N, n_classes)``.

    Mirrors :func:`harness._segnet_argmax_chunked` op-for-op (``(b,1,3,H,W)`` -> ``preprocess_input``
    -> forward) but returns the SPATIAL-MEAN logits instead of the argmax, for the decision-geometry
    probe. Seg-res input => ``preprocess_input`` bilinear-resize is EXACT identity (verified).
    """

    import torch

    n = len(seg_frames)
    step = n if int(chunk) <= 0 else int(chunk)
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, n, step):
            block = seg_frames[start : start + step]
            arr = np.stack([np.asarray(f)[None] for f in block], axis=0)  # (b,1,H,W,3)
            xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()  # (b,1,3,H,W)
            seg_in = segnet.preprocess_input(xp)
            logits = segnet(seg_in)  # (b, n_classes, H, W)
            out.append(logits.mean(dim=(2, 3)).cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


def map_decision_geometry(
    segnet: Any | None = None,
    *,
    grid_levels: int = 6,
    verdict_batch: int = 32,
    n_classes: int = N_CLASSES,
    backend: str = "cpu-torch",
) -> DecisionGeometry:
    """Probe the frozen SegNet with a ``grid_levels**3`` cube of constant-colour tiles.

    Returns the :class:`DecisionGeometry` (colours, mean logits, argmax class). Cheap + decisive:
    ``grid_levels=6`` => 216 constant tiles (7 chunks). ``segnet=None`` loads the frozen CPU authority.
    """

    if segnet is None:
        segnet = load_frozen_segnet(backend)
    levels = np.linspace(0.0, 255.0, int(grid_levels), dtype=np.float32)
    colors = np.array([[r, g, b] for r in levels for g in levels for b in levels], dtype=np.float32)
    frames = [np.broadcast_to(c, (SEG_H, SEG_W, 3)).astype(np.float32) for c in colors]
    mean_logits = _seg_logits_mean(segnet, frames, int(verdict_batch))
    if mean_logits.shape[1] != int(n_classes):
        raise PaletteRealizationError(
            f"decision geometry expected {n_classes} logit channels; got {mean_logits.shape[1]}"
        )
    return DecisionGeometry(
        colors=colors,
        mean_logits=mean_logits,
        argmax_class=mean_logits.argmax(axis=1).astype(np.int64),
        n_classes=int(n_classes),
        grid_levels=int(grid_levels),
    )


def classify_color(dg: DecisionGeometry, rgb: np.ndarray) -> int:
    """Nearest-probed-colour class lookup (the cheap proxy the palette optimiser uses)."""
    rgb = np.asarray(rgb, dtype=np.float32).reshape(3)
    d2 = ((dg.colors - rgb[None, :]) ** 2).sum(axis=1)
    return int(dg.argmax_class[int(np.argmin(d2))])


# --------------------------------------------------------------------------- #
# Palettes.                                                                    #
# --------------------------------------------------------------------------- #
def naive_mean_palette(rg_hwc: np.ndarray, lab: np.ndarray, *, n_classes: int = N_CLASSES) -> np.ndarray:
    """Per-pair per-class MEAN RGB ``-> (n_classes, 3)`` float (the baseline realisation).

    ``rg_hwc`` render-grid float RGB ``(SEG_H, SEG_W, 3)``; ``lab`` the ``(SEG_H, SEG_W)`` L* argmax.
    A class absent in ``lab`` gets ``0`` (matches the ceiling script; those pixels never paint).
    """
    rg = np.asarray(rg_hwc, dtype=np.float32)
    lab = np.asarray(lab)
    if rg.shape != (SEG_H, SEG_W, 3):
        raise PaletteRealizationError(f"rg_hwc must be ({SEG_H},{SEG_W},3); got {rg.shape}")
    if lab.shape != (SEG_H, SEG_W):
        raise PaletteRealizationError(f"lab must be ({SEG_H},{SEG_W}); got {lab.shape}")
    palette = np.zeros((int(n_classes), 3), dtype=np.float32)
    for c in range(int(n_classes)):
        m = lab == c
        if m.any():
            palette[c] = rg[m].mean(axis=0)
    return palette


def global_mean_palette(
    frames_rg: list[np.ndarray], labs: np.ndarray, *, n_classes: int = N_CLASSES
) -> np.ndarray:
    """ONE global per-class MEAN RGB over ALL pairs ``-> (n_classes, 3)`` (per-pair vs global control).

    Pixel-count-weighted mean of every class-``c`` pixel across the stack — isolates the
    global-vs-per-pair axis from the mixing-robust axis in the arm comparison.
    """
    labs = np.asarray(labs)
    acc = np.zeros((int(n_classes), 3), dtype=np.float64)
    cnt = np.zeros((int(n_classes),), dtype=np.int64)
    for k, rg in enumerate(frames_rg):
        rg = np.asarray(rg, dtype=np.float32)
        lab = labs[k]
        for c in range(int(n_classes)):
            m = lab == c
            n = int(m.sum())
            if n:
                acc[c] += rg[m].sum(axis=0)
                cnt[c] += n
    out = np.zeros((int(n_classes), 3), dtype=np.float32)
    nz = cnt > 0
    out[nz] = (acc[nz] / cnt[nz, None]).astype(np.float32)
    return out


def _mixing_penalty(
    dg: DecisionGeometry, palette: np.ndarray, *, alpha_samples: int, w_third: float
) -> float:
    """Sum over ordered class pairs of samples along ``c_i->c_j`` that land in a THIRD class.

    A high penalty = the palette's colour segments pass through a foreign class's decision region
    (the manufactured third-class-mixing pathology). Lower is better.
    """
    n = palette.shape[0]
    alphas = np.linspace(0.0, 1.0, int(alpha_samples))
    pen = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for a in alphas:
                mix = a * palette[i] + (1.0 - a) * palette[j]
                k = classify_color(dg, mix)
                if k != i and k != j:
                    pen += float(w_third)
    return pen


def mixing_robust_palette(
    dg: DecisionGeometry,
    *,
    seed_palette: np.ndarray | None = None,
    neighbor_dist: float = 96.0,
    alpha_samples: int = 9,
    w_third: float = 10.0,
    max_rounds: int = 6,
    conf_min_frac: float = 0.5,
    top_candidates: int = 12,
) -> np.ndarray:
    """Optimise ONE global 5-colour palette to minimise pairwise third-class mixing.

    Two seeding modes:

    * ``seed_palette=None`` (abstract): seed = each class's max-mean-logit colour
      (:meth:`DecisionGeometry.best_color_for_class`); candidate pool = the class's argmax-interior
      colours (or top-logit colours for a context-only class). MEASURED to be catastrophic on the real
      SegNet (constant-colour tiles decode ~90% to Undrivable => abstract class colours ignore the
      spatial context SegNet relies on).
    * ``seed_palette`` given (scene-anchored, the fair test): seed = the data-driven scene palette
      (e.g. global-mean); candidate pool per class = probed colours WITHIN ``neighbor_dist`` of that
      class's seed colour — a MINIMAL perturbation that stays near the real scene colour while
      nudging away from third-class mixing. This respects SegNet's context-dominance.

    Coordinate descent minimises the global :func:`_mixing_penalty`; deterministic (nearest-grid
    proxy). ``conf_min_frac`` documents the interior-confidence intent (argmax-win already encodes it).
    """
    n = dg.n_classes
    if seed_palette is not None:
        palette = np.asarray(seed_palette, dtype=np.float32).copy()
        pools = []
        for c in range(n):
            d2 = ((dg.colors - palette[c][None, :]) ** 2).sum(axis=1)
            near = np.where(d2 <= float(neighbor_dist) ** 2)[0]
            if near.size == 0:
                near = np.array([int(np.argmin(d2))])
            pools.append(near[np.argsort(d2[near])][: int(top_candidates)])
    else:
        palette = np.stack([dg.best_color_for_class(c) for c in range(n)], axis=0).astype(np.float32)
        pools = []
        for c in range(n):
            interior = np.where(dg.argmax_class == c)[0]
            if interior.size >= 1:
                order = interior[np.argsort(-dg.mean_logits[interior, c])]
            else:  # context-only class: fall back to top-logit colours (never wins argmax)
                order = np.argsort(-dg.mean_logits[:, c])
            pools.append(order[: int(top_candidates)])

    best_pen = _mixing_penalty(dg, palette, alpha_samples=alpha_samples, w_third=w_third)
    for _ in range(int(max_rounds)):
        improved = False
        for c in range(n):
            cur_idx_pen = best_pen
            cur_color = palette[c].copy()
            for cand in pools[c]:
                trial = palette.copy()
                trial[c] = dg.colors[int(cand)]
                pen = _mixing_penalty(dg, trial, alpha_samples=alpha_samples, w_third=w_third)
                if pen < cur_idx_pen - 1e-9:
                    cur_idx_pen = pen
                    cur_color = dg.colors[int(cand)].astype(np.float32)
            if cur_idx_pen < best_pen - 1e-9:
                palette[c] = cur_color
                best_pen = cur_idx_pen
                improved = True
        if not improved:
            break
    # conf_min_frac is retained for API-compat / documentation of the interior threshold intent;
    # interior membership already encodes argmax-win (the strongest confidence signal available).
    _ = conf_min_frac
    return palette.astype(np.float32)


# --------------------------------------------------------------------------- #
# Painting.                                                                    #
# --------------------------------------------------------------------------- #
def paint_render_grid(palette: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Palette-paint an ``(SEG_H, SEG_W)`` label map -> render-grid float ``(SEG_H, SEG_W, 3)``."""
    palette = np.asarray(palette, dtype=np.float32)
    lab = np.asarray(lab)
    if lab.shape != (SEG_H, SEG_W):
        raise PaletteRealizationError(f"lab must be ({SEG_H},{SEG_W}); got {lab.shape}")
    return palette[lab].astype(np.float32)


def paint_seg_grid(palette: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Alias of :func:`paint_render_grid` used for the no-R hard seg-grid arm (semantic clarity)."""
    return paint_render_grid(palette, lab)


def paint_camera_res_uint8(palette: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Nearest-UP ``L*`` to camera 874x1164 then hard palette paint -> camera uint8 (arm 3).

    Painting at camera res means R's bicubic-UP is a no-op (the harness returns a camera-uint8 frame
    unchanged), so ONLY the SegNet ``preprocess_input`` bilinear-DOWN mixes — isolating how much of
    the flip mass came from the UP stage. Nearest upsample keeps the label hard (no floating blend).
    """
    palette = np.asarray(palette, dtype=np.float32)
    lab = np.asarray(lab)
    if lab.shape != (SEG_H, SEG_W):
        raise PaletteRealizationError(f"lab must be ({SEG_H},{SEG_W}); got {lab.shape}")
    ys = (np.arange(CAMERA_H) * SEG_H // CAMERA_H).clip(0, SEG_H - 1)
    xs = (np.arange(CAMERA_W) * SEG_W // CAMERA_W).clip(0, SEG_W - 1)
    lab_cam = lab[np.ix_(ys, xs)]  # (CAMERA_H, CAMERA_W) nearest-up label
    return np.clip(np.round(palette[lab_cam]), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Realisation.                                                                 #
# --------------------------------------------------------------------------- #
def realize_argmax_no_R(
    segnet: Any, seg_frames: list[np.ndarray], *, verdict_batch: int = 32
) -> np.ndarray:
    """Forward hard seg-res frames through SegNet with ZERO R (arm 4 boundary-snapped).

    ``preprocess_input`` at seg-res ``(SEG_H, SEG_W)`` is EXACT identity (verified), so this is the
    frozen-SegNet argmax of a perfectly hard partition with no resolution mixing at all. Reuses the
    canonical chunked forward (:func:`harness._segnet_argmax_chunked`) — a seg-res frame is not
    (CAMERA_H,CAMERA_W) uint8, but the frames are already at seg-res so we must NOT push them through
    ``render_grid_to_camera_uint8`` (that would bicubic-UP and re-introduce blend); instead we feed
    them straight to the chunked SegNet forward, which resizes seg->seg (identity).
    """
    return _segnet_argmax_chunked(segnet, list(seg_frames), int(verdict_batch))


# --------------------------------------------------------------------------- #
# Flip decomposition: third-class-mixing (manufactured) vs boundary-position.  #
# --------------------------------------------------------------------------- #
def _dilate_presence(binary_hw: np.ndarray, radius: int) -> np.ndarray:
    """Separable box-dilation of a ``(H,W)`` bool by ``radius`` (zero-fill borders)."""
    b = np.asarray(binary_hw, dtype=bool)
    # rows
    out = b.copy()
    for d in range(1, radius + 1):
        out[d:, :] |= b[:-d, :]
        out[:-d, :] |= b[d:, :]
    b2 = out.copy()
    for d in range(1, radius + 1):
        out[:, d:] |= b2[:, :-d]
        out[:, :-d] |= b2[:, d:]
    return out


@dataclass
class FlipDecomposition:
    """Per-arm split of realized flips into manufactured (third-class) vs real (boundary)."""

    total_flips: int
    total_pixels: int
    third_class_flips: int
    boundary_flips: int
    third_class_share: float  # third / total flips
    boundary_share: float  # boundary / total flips
    third_class_frac_of_pixels: float  # third / total pixels (== manufactured d_seg component)
    boundary_frac_of_pixels: float
    radius: int
    per_class_third_share: dict[str, float] = field(default_factory=dict)


def decompose_flips(
    realized: np.ndarray,
    lstars: np.ndarray,
    *,
    radius: int = 2,
    n_classes: int = N_CLASSES,
) -> FlipDecomposition:
    """Split ``(realized != L*)`` flips into third-class-mixing vs boundary-position.

    A flip pixel with realized class ``k`` is **boundary-position** if ``k`` appears anywhere in the
    ``(2*radius+1)^2`` ``L*`` window around it (``k`` is a legitimately-adjacent class => a real
    ±jitter). Otherwise it is **third-class-mixing**: ``k`` is not even locally present => it can only
    have been MANUFACTURED by the palette (two class colours blended into ``k``'s RGB region, or a
    context/palette-colour effect). The third-class share is the manufactured fraction of ``0.0337``.
    """
    realized = np.asarray(realized)
    lstars = np.asarray(lstars)
    if realized.shape != lstars.shape:
        raise PaletteRealizationError(
            f"realized {realized.shape} != lstars {lstars.shape}"
        )
    if realized.ndim != 3:
        raise PaletteRealizationError(f"expected (N,H,W); got {realized.shape}")
    n = realized.shape[0]
    total_pixels = int(realized.size)
    third = 0
    boundary = 0
    per_class_third = np.zeros((int(n_classes),), dtype=np.int64)
    per_class_flip = np.zeros((int(n_classes),), dtype=np.int64)
    for i in range(n):
        r = realized[i]
        gt = lstars[i]
        flip = r != gt
        if not flip.any():
            continue
        # presence[k] = dilated (gt==k); a flip to k is boundary iff presence[k] at that pixel.
        present_classes = np.unique(r[flip])
        for k in present_classes:
            k = int(k)
            pres = _dilate_presence(gt == k, radius)
            fk = flip & (r == k)
            b = int((fk & pres).sum())
            t = int(fk.sum()) - b
            boundary += b
            third += t
            per_class_third[k] += t
            per_class_flip[k] += int(fk.sum())
    total_flips = third + boundary
    names = CLASS_NAMES if int(n_classes) == N_CLASSES else tuple(f"class_{c}" for c in range(int(n_classes)))
    per_class_third_share = {
        names[c]: (float(per_class_third[c] / per_class_flip[c]) if per_class_flip[c] else 0.0)
        for c in range(int(n_classes))
    }
    return FlipDecomposition(
        total_flips=total_flips,
        total_pixels=total_pixels,
        third_class_flips=third,
        boundary_flips=boundary,
        third_class_share=(third / total_flips) if total_flips else 0.0,
        boundary_share=(boundary / total_flips) if total_flips else 0.0,
        third_class_frac_of_pixels=third / total_pixels,
        boundary_frac_of_pixels=boundary / total_pixels,
        radius=int(radius),
        per_class_third_share=per_class_third_share,
    )


# --------------------------------------------------------------------------- #
# Arm driver.                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class ArmResult:
    """One arm's realized F + per-class + flip decomposition (advisory, NON-PROMOTABLE)."""

    arm: str
    agg_dseg: float
    per_class_dseg: dict[str, float]
    n_frames: int
    decomposition: FlipDecomposition
    label: str = THROUGH_R_LABEL
    realized: np.ndarray | None = None
    inputs_sha256: str | None = None
    subset_reason: str | None = None


def run_arm(
    arm: str,
    frames: list[np.ndarray],
    lstars: np.ndarray,
    *,
    segnet: Any,
    no_R: bool = False,
    verdict_batch: int = 32,
    decompose_radius: int = 2,
    n_classes: int = N_CLASSES,
    allow_subset_reason: str | None = None,
) -> ArmResult:
    """Realise ``frames`` (through-R unless ``no_R``) -> frozen SegNet -> F + decomposition.

    ``no_R=True`` (arm 4): ``frames`` are hard seg-res; forwarded via :func:`realize_argmax_no_R`
    (identity preprocess, zero mixing). Otherwise reuse the canonical through-R meter
    (:func:`tac.through_r.measure_through_r`) via its private chunked forward on camera frames.
    n600 discipline mirrors the harness: ``N != 600`` requires an explicit ``allow_subset_reason``.
    """
    lstars = np.asarray(lstars)
    n = len(frames)
    if n != 600 and not (allow_subset_reason and allow_subset_reason.strip()):
        raise PaletteRealizationError(
            f"n600 discipline: a verdict needs N=600; got N={n}. Pass allow_subset_reason for a "
            "labelled non-authority subset."
        )
    if no_R:
        realized = realize_argmax_no_R(segnet, frames, verdict_batch=verdict_batch)
    else:
        cam = [
            f if (np.asarray(f).shape[:2] == (CAMERA_H, CAMERA_W) and np.asarray(f).dtype == np.uint8)
            else render_grid_to_camera_uint8(f)
            for f in frames
        ]
        realized = _segnet_argmax_chunked(segnet, cam, int(verdict_batch))
    gt_list = [lstars[i] for i in range(n)]
    pred_list = [realized[i] for i in range(n)]
    cmp = compare_label_stack_to_lstars(pred_list, gt_list, n_classes=int(n_classes))
    decomp = decompose_flips(realized, lstars[:n], radius=int(decompose_radius), n_classes=int(n_classes))
    import hashlib

    sha = hashlib.sha256(
        np.ascontiguousarray(lstars[:n]).tobytes() + np.ascontiguousarray(realized).tobytes()
    ).hexdigest()
    return ArmResult(
        arm=arm,
        agg_dseg=cmp.agg_dseg,
        per_class_dseg=cmp.per_class_dseg,
        n_frames=n,
        decomposition=decomp,
        realized=realized,
        inputs_sha256=sha,
        subset_reason=(allow_subset_reason.strip() if (n != 600 and allow_subset_reason) else None),
    )


def arm_result_to_measurement_rows(
    res: ArmResult, *, git_sha: str, review_status: str, tool: str = "tac.through_r.palette_realization.run_arm"
) -> list[Any]:
    """Build canonical :class:`tac.verdicts.MeasurementRow` rows for an :class:`ArmResult`.

    Reuses :meth:`ThroughRResult.to_measurement_rows` (one aggregate + per-class ``[through-R]`` rows)
    so the palette arms emit the SAME typed measurement surface as the canonical meter (P1).
    """
    tr = ThroughRResult(
        agg_dseg=res.agg_dseg,
        per_class_dseg=res.per_class_dseg,
        flip_share_by_class={},
        per_pair_dseg=[],
        n_frames=res.n_frames,
        n_classes=N_CLASSES,
        is_n600=res.n_frames == 600,
        backend="cpu-torch",
        input_space="render-grid",
        inputs_sha256=res.inputs_sha256,
        subset_reason=res.subset_reason,
    )
    return tr.to_measurement_rows(
        git_sha=git_sha, review_status=review_status, tool=tool, config_ref=f"arm={res.arm}"
    )
