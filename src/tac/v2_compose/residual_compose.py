# SPDX-License-Identifier: MIT
"""v2_compose.residual_compose — the bit-exact bulk (+) residual composition (Step 4 / Step 6 glue).

THE rate-bearing seam, realized as a SINGLE composition rule shared by the trainer (compress side),
the realized verdict, and the ``inflate.py`` decoder (so train == inflate, NO-FAKE):

    composed_rgb = where(composition_mask, INR_residual_rgb, deterministic_bulk_rgb)

The composition mask is DERIVED FROM THE BULK'S OWN WARPED LABEL MAP, NOT from the GT. This is the
keystone that keeps the rate small: the mask is a deterministic function of the already-generated
bulk partition, so it is REGENERATED at inflate (rule-118 FREE, 0 counted bytes) and is bit-identical
to the compress-side mask. The (correct) bulk shows through everywhere else, so the INR can be FAR
smaller than the full-partition INR.

MASK MODE (the B2 fix, review r1 HIGH-2): the override region must COVER the cells the bulk gets
WRONG (``bulk_argmax_through_R != gt_lstars``), not merely the cells where the bulk predicts a LEARN
class. d_seg flips are argmax instabilities on the codim-1 boundary annulus (small-margin pixels), so
the DEFAULT mask is ``boundary_annulus``: the inter-class boundary of the bulk's OWN warped partition
(dilated ~2px). This is GT-FREE (derived from the bulk's own argmax — available identically at
compress AND inflate), SELF-DETECTED (no SegNet class index is hardcoded; it keys off where the
partition CHANGES), and covers ALL inter-class flips (Road↔Undrivable/sky included), not just the
Lane+Movable subset the legacy ``learn_classes`` mask reached. The legacy ``learn_classes`` mode
(``isin(warped_label, learn_classes)``) is retained for back-compat; ``union`` is annulus ∪
learn-class region. Whatever mode the bundle is built with is carried into the residual manifest so
the inflate re-derives the SAME mask (train == inflate). :func:`measure_composition_coverage` reports
what fraction of the true residual the chosen mask reaches (the explicit pre-fire GEOMETRY gate that
prevents misreading an unreachable-residual ceiling as an INR-capacity wall).

Boundary (rule-118, the NO-FAKE line, sister of ``bulk_generator`` + ``archive_grammar``):
  * COUNTED in archive.zip: the residual-INR weights, the stored keyframes/pose/calib (bulk).
  * FREE in inflate.py: the warp, the SDF/ramp render, the composition mask derivation, the
    composition op. All GENERIC. NO GT mask, NO bulk RGB, and NO scorer weights ship.

``compose_residual_rgb`` + ``derive_composition_mask`` are the ONE math, mirrored op-for-op into
the inflate.py template (``archive_grammar``); :func:`assert_inflate_compose_parity`-style tests
prove the mirror is byte-identical. The GT-derived ``residual_mask`` from
:mod:`tac.v2_compose.residual_target` is the DIAGNOSTIC d_seg floor (which cells the bulk gets
wrong) -- NOT shipped and NOT the composition rule.

Authority: numpy / CPU (NEVER MPS). ``[macOS-CPU advisory] NON-PROMOTABLE``; the pointer is UNMOVED
0.19110 and moves only on a byte-closed ``upstream/evaluate.py`` row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "LEARN_CLASSES",
    "MASK_MODE_BOUNDARY_ANNULUS",
    "MASK_MODE_LEARN_CLASSES",
    "MASK_MODE_UNION",
    "MASK_MODES",
    "DEFAULT_ANNULUS_DILATE",
    "derive_composition_mask",
    "compose_residual_rgb",
    "measure_composition_coverage",
    "CoverageReport",
    "ResidualBundle",
    "build_residual_training_bundle",
    "save_residual_training_bundle",
    "load_residual_training_bundle",
]

# Canonical comma10k order [Road0, Lane1, Undrivable2, Movable3, MyCar4]. The LEARN tier (the
# residual the deterministic bulk cannot generate from pose) is Lane + Movable (the BULK tier is
# Road/Undrivable/MyCar = BULK_IDX [0,2,4]). The legacy composition restricted the INR override to
# exactly Lane+Movable; the boundary-annulus mode (DEFAULT) generalizes it to all inter-class edges.
LEARN_CLASSES: tuple[int, ...] = (1, 3)

# Composition mask modes (the B2 fix). boundary_annulus = inter-class boundary of the bulk's OWN
# warped partition (GT-free, self-detected, covers ALL codim-1 flips); learn_classes = the legacy
# isin(warped_label, learn_classes) region; union = annulus ∪ learn-class region.
MASK_MODE_BOUNDARY_ANNULUS = "boundary_annulus"
MASK_MODE_LEARN_CLASSES = "learn_classes"
MASK_MODE_UNION = "union"
MASK_MODES: tuple[str, ...] = (MASK_MODE_BOUNDARY_ANNULUS, MASK_MODE_LEARN_CLASSES, MASK_MODE_UNION)
# Default annulus half-width (rounds of 4-connectivity dilation around the inter-class boundary). ~2px
# keeps the override region small while covering the small-margin flip band (FACT-2 codim-1 annulus).
DEFAULT_ANNULUS_DILATE = 2


def _dilate_bool(mask: np.ndarray, rounds: int) -> np.ndarray:
    """scipy-FREE 4-connectivity binary dilation (``rounds`` iterations). Deterministic, bit-exact,
    and trivially mirrored in the MLX-free inflate.py. ``rounds == 0`` returns the mask unchanged."""
    m = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    for _ in range(int(rounds)):
        out = m.copy()
        out[:-1, :] |= m[1:, :]   # up
        out[1:, :] |= m[:-1, :]   # down
        out[:, :-1] |= m[:, 1:]   # left
        out[:, 1:] |= m[:, :-1]   # right
        m = out
    return m


def _inter_class_boundary(label: np.ndarray) -> np.ndarray:
    """The codim-1 inter-class boundary of a label map: a cell is True iff any 4-neighbor has a
    DIFFERENT label (both sides of every edge marked). scipy-FREE, deterministic, bit-exactly
    mirrored in the MLX-free inflate.py. SELF-DETECTED (keys off where the partition CHANGES — no
    SegNet class index hardcoded)."""
    lbl = np.asarray(label)
    if lbl.ndim != 2:
        raise ValueError(f"label must be (H,W); got {lbl.shape}")
    b = np.zeros(lbl.shape, dtype=bool)
    ne_v = lbl[:-1, :] != lbl[1:, :]   # vertical neighbor differs
    b[:-1, :] |= ne_v
    b[1:, :] |= ne_v
    ne_h = lbl[:, :-1] != lbl[:, 1:]   # horizontal neighbor differs
    b[:, :-1] |= ne_h
    b[:, 1:] |= ne_h
    return b


def derive_composition_mask(
    warped_label: np.ndarray,
    learn_classes: tuple[int, ...] = LEARN_CLASSES,
    dilate: int = 0,
    mode: str = MASK_MODE_BOUNDARY_ANNULUS,
) -> np.ndarray:
    """The bulk-LABEL-derived override region (True == INR paints here, False == bulk shows through).
    Deterministic function of the bulk's OWN warped partition -- available identically at compress
    AND inflate (rule-118 FREE, 0 counted bytes). Returns a bool (H, W) mask.

    ``mode``:
      * ``boundary_annulus`` (DEFAULT, the B2 fix): the inter-class boundary of ``warped_label``
        (+ ``dilate`` rounds of 4-connectivity dilation). Covers ALL codim-1 flips (where d_seg
        argmax instabilities live), including Road↔Undrivable/sky -- GT-free, self-detected, no
        class index hardcoded.
      * ``learn_classes`` (LEGACY): ``isin(warped_label, learn_classes)`` (+ ``dilate``). Reaches
        ONLY the Lane+Movable region -- the geometry-ceiling bug review r1 HIGH-2 flagged.
      * ``union``: ``boundary_annulus`` ∪ ``learn_classes`` (the full Lane/Movable region AND every
        inter-class edge).
    """
    lbl = np.asarray(warped_label)
    if lbl.ndim != 2:
        raise ValueError(f"warped_label must be (H,W); got {lbl.shape}")
    if mode not in MASK_MODES:
        raise ValueError(f"mode must be one of {MASK_MODES}; got {mode!r}")
    if mode == MASK_MODE_LEARN_CLASSES:
        mask = np.isin(lbl, np.asarray(learn_classes, dtype=lbl.dtype))
    elif mode == MASK_MODE_BOUNDARY_ANNULUS:
        mask = _inter_class_boundary(lbl)
    else:  # union
        mask = _inter_class_boundary(lbl) | np.isin(lbl, np.asarray(learn_classes, dtype=lbl.dtype))
    if int(dilate) > 0:
        mask = _dilate_bool(mask, int(dilate))
    return mask.astype(bool)


@dataclass(frozen=True)
class CoverageReport:
    """How well a composition mask COVERS the true (GT-measured) residual -- the explicit pre-fire
    GEOMETRY gate (B2). ``unreachable_dseg`` is the per-cell d_seg the INR can NEVER fix because the
    flip falls OUTSIDE the override region; it is a HARD lower bound on the composed witness d_seg,
    independent of INR capacity. The GPU GO must be gated on it sitting comfortably below the
    sub-0.15 d_seg budget, else a geometry ceiling would be misread as an INR-capacity wall."""

    coverage: float            # (residual & mask).sum() / residual.sum() -- frac of flips reachable
    unreachable_dseg: float    # (residual & ~mask).mean() -- the d_seg the INR can NEVER reach
    reachable_dseg: float      # (residual & mask).mean()
    residual_dseg: float       # residual.mean() == the deterministic bulk d_seg floor
    override_frac: float       # mask.mean() -- how much of the frame the INR may repaint
    n_residual_cells: int
    n_reachable_cells: int
    dseg_budget: float | None  # the sub-0.15 d_seg budget the gate compares against (None == ungated)
    passes_gate: bool          # unreachable_dseg < dseg_budget (True when ungated)
    authority: str = "[macOS-CPU advisory] NON-PROMOTABLE"

    def to_summary(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "score_claim": False,
            "promotable": False,
            "frontier_pointer": "UNMOVED 0.19110",
            "coverage": self.coverage,
            "unreachable_dseg": self.unreachable_dseg,
            "reachable_dseg": self.reachable_dseg,
            "residual_dseg_floor": self.residual_dseg,
            "override_frac": self.override_frac,
            "n_residual_cells": self.n_residual_cells,
            "n_reachable_cells": self.n_reachable_cells,
            "dseg_budget": self.dseg_budget,
            "passes_gate": self.passes_gate,
            "note": (
                "coverage = frac of (bulk!=GT) cells inside the composition mask. unreachable_dseg "
                "= the d_seg floor the INR can NEVER close (flip OUTSIDE the override region) -- a "
                "GEOMETRY ceiling independent of INR capacity. GATE the GPU GO on unreachable_dseg "
                "< the sub-0.15 d_seg budget (else a geometry ceiling reads as a capacity wall)."
            ),
        }


def measure_composition_coverage(
    residual_mask: np.ndarray,
    composition_mask: np.ndarray,
    *,
    dseg_budget: float | None = None,
) -> CoverageReport:
    """Measure how much of the TRUE residual the composition mask COVERS (the B2 pre-fire gate).

    Args:
        residual_mask: (n, H, W) bool -- the cells the deterministic bulk gets WRONG
            (``bulk_argmax_through_R != gt_lstars``, from :mod:`tac.v2_compose.residual_target`).
        composition_mask: (n, H, W) bool -- the override region the INR may repaint
            (:func:`derive_composition_mask` over the bulk's warped labels).
        dseg_budget: the sub-0.15 d_seg budget the gate compares ``unreachable_dseg`` against
            (None == report only, no GO/NO-GO verdict).

    Returns:
        :class:`CoverageReport`. NO-FAKE: every quantity is measured on the SAME cells through the
        SAME R + frozen SegNet as the byte-close verdict; nothing is asserted.
    """
    res = np.asarray(residual_mask, dtype=bool)
    comp = np.asarray(composition_mask, dtype=bool)
    if res.shape != comp.shape:
        raise ValueError(f"residual_mask {res.shape} != composition_mask {comp.shape}")
    n_res = int(res.sum())
    reachable = res & comp
    n_reach = int(reachable.sum())
    coverage = float(n_reach / n_res) if n_res > 0 else 1.0
    unreachable_dseg = float((res & ~comp).mean())
    passes = True if dseg_budget is None else bool(unreachable_dseg < float(dseg_budget))
    return CoverageReport(
        coverage=coverage,
        unreachable_dseg=unreachable_dseg,
        reachable_dseg=float(reachable.mean()),
        residual_dseg=float(res.mean()),
        override_frac=float(comp.mean()),
        n_residual_cells=n_res,
        n_reachable_cells=n_reach,
        dseg_budget=(None if dseg_budget is None else float(dseg_budget)),
        passes_gate=passes,
    )


def compose_residual_rgb(bulk_rgb: np.ndarray, inr_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """THE composition op (numpy fp; the ONE math mirrored into inflate.py + the MLX trainer):
    ``composed = where(mask, inr_rgb, bulk_rgb)``. ``bulk_rgb`` / ``inr_rgb`` are (H, W, 3); ``mask``
    is (H, W) bool. The bulk is a CONSTANT (no gradient) -- only the INR is trainable, so gradients
    flow ONLY through the masked region (the residual the INR owns)."""
    bulk = np.asarray(bulk_rgb)
    inr = np.asarray(inr_rgb)
    if bulk.shape != inr.shape or bulk.ndim != 3 or bulk.shape[-1] != 3:
        raise ValueError(f"bulk_rgb/inr_rgb must match as (H,W,3); got {bulk.shape} vs {inr.shape}")
    m = np.asarray(mask, dtype=bool)
    if m.shape != bulk.shape[:2]:
        raise ValueError(f"mask must be (H,W) matching the render; got {m.shape} vs {bulk.shape[:2]}")
    return np.where(m[..., None], inr, bulk)


@dataclass(frozen=True)
class ResidualBundle:
    """The residual-only trainer's INPUT (a free TRAINING ARTIFACT -- NOT an archive section).

    Carries the FIXED deterministic bulk render (render res, pre-R) + the bulk-derived composition
    mask per pair, so the trainer composes ``where(mask, INR, bulk)`` and trains the INR on the
    realized-through-R d_seg of the composed witness. The COUNTED bytes are the INR weights the
    trainer PRODUCES (+ the stored bulk keyframes/pose) -- never this bundle."""

    bulk_rgb_render_res: np.ndarray   # (n, H, W, 3) float32 -- the deterministic bulk RGB (pre-R)
    composition_mask: np.ndarray      # (n, H, W) bool -- the bulk-derived override region (INR paints)
    learn_classes: tuple[int, ...]
    dilate: int
    n_pairs: int
    render_h: int
    render_w: int
    mask_mode: str = MASK_MODE_BOUNDARY_ANNULUS
    authority: str = "[macOS-CPU advisory] NON-PROMOTABLE"

    def to_summary(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "score_claim": False,
            "promotable": False,
            "frontier_pointer": "UNMOVED 0.19110",
            "n_pairs": self.n_pairs,
            "render_h": self.render_h,
            "render_w": self.render_w,
            "mask_mode": self.mask_mode,
            "learn_classes": list(self.learn_classes),
            "dilate": self.dilate,
            "composition_override_frac": float(self.composition_mask.mean()),
            "note": (
                "composition_mask = derive_composition_mask(warped_bulk_label, mode=mask_mode) "
                "(+dilate): the INR paints ONLY here; the deterministic bulk shows through "
                "elsewhere. mask_mode=boundary_annulus (default) covers ALL inter-class flips. Mask "
                "is bulk-derived (regenerated FREE at inflate) -- ships 0 bytes. The INR WEIGHTS the "
                "trainer produces are the COUNTED LEARN-tier rate."
            ),
        }


def build_residual_training_bundle(
    bulk_rgb_render_res: np.ndarray,
    warped_labels: np.ndarray,
    *,
    learn_classes: tuple[int, ...] = LEARN_CLASSES,
    dilate: int = 0,
    mode: str = MASK_MODE_BOUNDARY_ANNULUS,
) -> ResidualBundle:
    """Assemble the residual bundle from the deterministic bulk render + its warped label maps.

    Args:
        bulk_rgb_render_res: (n, H, W, 3) the deterministic bulk RGB at render res (pre-R), from
            :func:`tac.v2_compose.bulk_generator.generate_bulk_render_and_labels`.
        warped_labels: (n, H, W) the bulk's warped partition label maps (the mask source).
        learn_classes: the LEARN tier classes (Lane + Movable) the legacy ``learn_classes`` mode
            overrides (carried as metadata in the annulus mode).
        dilate: rounds of 4-connectivity dilation on the override region (0 = exact region).
        mode: composition mask mode (see :func:`derive_composition_mask`). DEFAULT
            ``boundary_annulus`` (the B2 fix -- covers all codim-1 flips, GT-free, self-detected).
    """
    rgb = np.asarray(bulk_rgb_render_res, dtype=np.float32)
    lbl = np.asarray(warped_labels)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"bulk_rgb_render_res must be (n,H,W,3); got {rgb.shape}")
    if lbl.ndim != 3 or lbl.shape != rgb.shape[:3]:
        raise ValueError(f"warped_labels must be (n,H,W) matching the render; got {lbl.shape} vs {rgb.shape[:3]}")
    if mode not in MASK_MODES:
        raise ValueError(f"mode must be one of {MASK_MODES}; got {mode!r}")
    n, H, W, _ = rgb.shape
    masks = np.empty((n, H, W), dtype=bool)
    for i in range(n):
        masks[i] = derive_composition_mask(lbl[i], learn_classes=learn_classes, dilate=dilate, mode=mode)
    return ResidualBundle(
        bulk_rgb_render_res=rgb,
        composition_mask=masks,
        learn_classes=tuple(int(c) for c in learn_classes),
        dilate=int(dilate),
        n_pairs=int(n),
        render_h=int(H),
        render_w=int(W),
        mask_mode=mode,
    )


def save_residual_training_bundle(bundle: ResidualBundle, path: str | Path) -> int:
    """Persist the residual bundle as a compressed npz (the residual-only trainer's --residual-target-npz
    input). Returns the file size in bytes. (Training artifact; NOT an archive section.)"""
    path = Path(path)
    s = str(path)
    if any(t in s for t in ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")):
        raise ValueError(f"refusing to write a durable bundle under a transient /tmp path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bulk_rgb_render_res=bundle.bulk_rgb_render_res.astype(np.float32),
        composition_mask=np.packbits(bundle.composition_mask, axis=None),
        composition_mask_shape=np.asarray(bundle.composition_mask.shape, np.int64),
        learn_classes=np.asarray(bundle.learn_classes, np.int64),
        dilate=np.asarray(bundle.dilate, np.int64),
        n_pairs=np.asarray(bundle.n_pairs, np.int64),
        render_h=np.asarray(bundle.render_h, np.int64),
        render_w=np.asarray(bundle.render_w, np.int64),
        mask_mode=np.asarray(bundle.mask_mode),
    )
    real = path if path.suffix == ".npz" else path.with_suffix(".npz")
    real.with_suffix(".summary.json").write_text(json.dumps(bundle.to_summary(), indent=2))
    return int(real.stat().st_size)


def load_residual_training_bundle(path: str | Path) -> ResidualBundle:
    """Load a residual bundle saved by :func:`save_residual_training_bundle`."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    z = np.load(path, allow_pickle=False)
    shape = tuple(int(x) for x in z["composition_mask_shape"])
    mask = np.unpackbits(z["composition_mask"])[: int(np.prod(shape))].astype(bool).reshape(shape)
    # mask_mode is provenance added with the B2 fix; older bundles (no key) were the legacy mode.
    mask_mode = str(z["mask_mode"]) if "mask_mode" in z.files else MASK_MODE_LEARN_CLASSES
    return ResidualBundle(
        bulk_rgb_render_res=z["bulk_rgb_render_res"].astype(np.float32),
        composition_mask=mask,
        learn_classes=tuple(int(c) for c in z["learn_classes"]),
        dilate=int(z["dilate"]),
        n_pairs=int(z["n_pairs"]),
        render_h=int(z["render_h"]),
        render_w=int(z["render_w"]),
        mask_mode=mask_mode,
    )
