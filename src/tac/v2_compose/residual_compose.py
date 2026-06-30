# SPDX-License-Identifier: MIT
"""v2_compose.residual_compose — the bit-exact bulk (+) residual composition (Step 4 / Step 6 glue).

THE rate-bearing seam, realized as a SINGLE composition rule shared by the trainer (compress side),
the realized verdict, and the ``inflate.py`` decoder (so train == inflate, NO-FAKE):

    composed_rgb = where(composition_mask, INR_residual_rgb, deterministic_bulk_rgb)

The composition mask is DERIVED FROM THE BULK'S OWN WARPED LABEL MAP (the LEARN classes —
Lane + Movable — optionally dilated), NOT from the GT. This is the keystone that keeps the rate
small: the mask is a deterministic function of the already-generated bulk partition, so it is
REGENERATED at inflate (rule-118 FREE, 0 counted bytes) and is bit-identical to the compress-side
mask. The residual INR therefore only paints the Lane+Movable override region; the (correct) bulk
shows through everywhere else, so the INR can be FAR smaller than the full-partition INR.

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
    "derive_composition_mask",
    "compose_residual_rgb",
    "ResidualBundle",
    "build_residual_training_bundle",
    "save_residual_training_bundle",
    "load_residual_training_bundle",
]

# Canonical comma10k order [Road0, Lane1, Undrivable2, Movable3, MyCar4]. The LEARN tier (the
# residual the deterministic bulk cannot generate from pose) is Lane + Movable (the BULK tier is
# Road/Undrivable/MyCar = BULK_IDX [0,2,4]). The composition lets the INR override exactly here.
LEARN_CLASSES: tuple[int, ...] = (1, 3)


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


def derive_composition_mask(
    warped_label: np.ndarray, learn_classes: tuple[int, ...] = LEARN_CLASSES, dilate: int = 0
) -> np.ndarray:
    """The bulk-LABEL-derived override region: ``isin(warped_label, learn_classes)`` (+ ``dilate``
    rounds of 4-connectivity dilation). Deterministic function of the bulk's OWN warped partition --
    available identically at compress AND inflate (rule-118 FREE, 0 counted bytes). Returns a
    bool (H, W) mask: True == the INR paints here, False == the deterministic bulk shows through."""
    lbl = np.asarray(warped_label)
    if lbl.ndim != 2:
        raise ValueError(f"warped_label must be (H,W); got {lbl.shape}")
    mask = np.isin(lbl, np.asarray(learn_classes, dtype=lbl.dtype))
    if int(dilate) > 0:
        mask = _dilate_bool(mask, int(dilate))
    return mask.astype(bool)


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
            "learn_classes": list(self.learn_classes),
            "dilate": self.dilate,
            "composition_override_frac": float(self.composition_mask.mean()),
            "note": (
                "composition_mask = isin(warped_bulk_label, learn_classes) (+dilate): the INR "
                "paints ONLY here; the deterministic bulk shows through elsewhere. Mask is "
                "bulk-derived (regenerated FREE at inflate) -- ships 0 bytes. The INR WEIGHTS the "
                "trainer produces are the COUNTED LEARN-tier rate."
            ),
        }


def build_residual_training_bundle(
    bulk_rgb_render_res: np.ndarray,
    warped_labels: np.ndarray,
    *,
    learn_classes: tuple[int, ...] = LEARN_CLASSES,
    dilate: int = 0,
) -> ResidualBundle:
    """Assemble the residual bundle from the deterministic bulk render + its warped label maps.

    Args:
        bulk_rgb_render_res: (n, H, W, 3) the deterministic bulk RGB at render res (pre-R), from
            :func:`tac.v2_compose.bulk_generator.generate_bulk_render_and_labels`.
        warped_labels: (n, H, W) the bulk's warped partition label maps (the mask source).
        learn_classes: the LEARN tier classes (Lane + Movable) the INR overrides.
        dilate: rounds of 4-connectivity dilation on the override region (0 = exact label region).
    """
    rgb = np.asarray(bulk_rgb_render_res, dtype=np.float32)
    lbl = np.asarray(warped_labels)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"bulk_rgb_render_res must be (n,H,W,3); got {rgb.shape}")
    if lbl.ndim != 3 or lbl.shape != rgb.shape[:3]:
        raise ValueError(f"warped_labels must be (n,H,W) matching the render; got {lbl.shape} vs {rgb.shape[:3]}")
    n, H, W, _ = rgb.shape
    masks = np.empty((n, H, W), dtype=bool)
    for i in range(n):
        masks[i] = derive_composition_mask(lbl[i], learn_classes=learn_classes, dilate=dilate)
    return ResidualBundle(
        bulk_rgb_render_res=rgb,
        composition_mask=masks,
        learn_classes=tuple(int(c) for c in learn_classes),
        dilate=int(dilate),
        n_pairs=int(n),
        render_h=int(H),
        render_w=int(W),
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
    return ResidualBundle(
        bulk_rgb_render_res=z["bulk_rgb_render_res"].astype(np.float32),
        composition_mask=mask,
        learn_classes=tuple(int(c) for c in z["learn_classes"]),
        dilate=int(z["dilate"]),
        n_pairs=int(z["n_pairs"]),
        render_h=int(z["render_h"]),
        render_w=int(z["render_w"]),
    )
