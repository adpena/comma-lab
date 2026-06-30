# SPDX-License-Identifier: MIT
"""v2_compose.residual_target — the keystone seam S3: residual = GT_partition - bulk_through_R.

Step 4 of the v2 pipeline. THE rate-bearing difference vs the existing ``--structured-init``
mechanism: the deterministic bulk is GENERATED at decode (outside the counted INR weights) and
SUBTRACTED from the target, so the residual INR carries ONLY the cells the bulk gets WRONG (the
Lane-survival annulus + small movables) — NOT the whole partition. That is what lets the residual
INR be FAR smaller than the 90 KB full-partition INR (the rate win).

Contrast (the seam the spec calls OPEN, now closed on the compress side):
  * ``--structured-init`` (existing trainer): bakes the bulk SDFs INTO the INR weights as a
    train-time init; the trained INR still encodes the whole partition -> NO rate shrink.
  * THIS module: computes the residual TARGET with the bulk SUBTRACTED. The trained INR only has
    to flip ``residual_mask`` cells; composed at inflate as ``bulk (argmax) where INR inactive,
    INR (argmax) where active`` -> the INR can be sized for the residual only.

NO-FAKE: ``residual_mask[p] = (bulk_argmax_through_R[p] != gt_lstars[p])`` is the EXACT set of
argmax cells the deterministic bulk gets wrong, measured through the SAME R + frozen CPU-torch
SegNet as the byte-close verdict. ``bulk_dseg = mean(residual_mask)`` IS the deterministic d_seg
floor the residual INR must close (the upper bound on what LEARN must achieve). No surrogate.
``[macOS-CPU advisory] NON-PROMOTABLE``; the pointer moves only on a byte-closed exact row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ResidualTarget",
    "compute_residual_target",
    "save_residual_target",
    "load_residual_target",
]

# canonical comma10k class names (self-detected order; index is metadata only).
_CLASS_NAMES = ("Road", "Lane", "Undriv", "Movable", "MyCar")


@dataclass(frozen=True)
class ResidualTarget:
    """The residual the LEARN-tier INR must carry (bulk subtracted)."""

    bulk_argmax_through_R: np.ndarray   # (n, H, W) int64 — deterministic bulk's SegNet argmax
    gt_lstars: np.ndarray               # (n, H, W) int64 — GT SegNet argmax (the target)
    residual_mask: np.ndarray           # (n, H, W) bool  — cells bulk gets WRONG (INR must flip)
    bulk_dseg: float                    # mean(residual_mask) — the deterministic d_seg floor
    n_pairs: int
    per_class_residual: dict[str, float]   # residual fraction attributable to each GT class
    per_class_area: dict[str, float]
    residual_classes_ranked: tuple[str, ...]  # GT classes carrying the most residual (desc)
    authority: str = "[macOS-CPU advisory] NON-PROMOTABLE"

    def to_summary(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "score_claim": False,
            "promotable": False,
            "frontier_pointer": "UNMOVED 0.19110",
            "n_pairs": self.n_pairs,
            "bulk_dseg_floor": self.bulk_dseg,
            "residual_cells_total_frac": float(self.residual_mask.mean()),
            "per_class_residual": dict(self.per_class_residual),
            "per_class_area": dict(self.per_class_area),
            "residual_classes_ranked": list(self.residual_classes_ranked),
            "note": (
                "residual_mask = (bulk_argmax_through_R != gt_lstars): the EXACT cells the "
                "deterministic bulk gets wrong. The residual INR (LEARN tier) trains to flip "
                "ONLY these -> it can be far smaller than the full-partition INR. bulk_dseg is "
                "the deterministic floor (FACT 2) the INR must close to ~6e-4 to beat 0.19110."
            ),
        }


def compute_residual_target(
    bulk_argmax_through_R: np.ndarray,
    gt_lstars: np.ndarray,
    *,
    class_names: tuple[str, ...] = _CLASS_NAMES,
) -> ResidualTarget:
    """Compute the residual target: residual = GT_partition - bulk_through_R (bulk SUBTRACTED).

    Args:
        bulk_argmax_through_R: (n, H, W) int — the deterministic bulk's SegNet argmax through R
            (from :func:`tac.v2_compose.bulk_generator.generate_bulk_argmax_stack`).
        gt_lstars: (n, H, W) int — the GT SegNet argmax (the cache ``lstars``).
        class_names: canonical class-name tuple (index == name position; metadata only).

    Returns:
        :class:`ResidualTarget` with the per-pair residual mask + the deterministic d_seg floor +
        the per-GT-class residual breakdown (which classes the residual INR must carry).
    """
    bulk = np.asarray(bulk_argmax_through_R, dtype=np.int64)
    gt = np.asarray(gt_lstars, dtype=np.int64)
    if bulk.shape != gt.shape:
        raise ValueError(f"shape mismatch: bulk {bulk.shape} vs gt {gt.shape}")
    if bulk.ndim != 3:
        raise ValueError(f"expected (n,H,W); got {bulk.shape}")

    residual_mask = bulk != gt          # the cells the bulk gets WRONG (the INR's job)
    n_pairs = int(bulk.shape[0])
    total = float(gt.size)

    # per-GT-class residual: fraction of ALL cells that are wrong AND whose GT class is c.
    per_class_residual: dict[str, float] = {}
    per_class_area: dict[str, float] = {}
    n_classes = len(class_names)
    for c in range(n_classes):
        gt_is_c = gt == c
        per_class_area[class_names[c]] = float(gt_is_c.mean())
        per_class_residual[class_names[c]] = float(np.sum(residual_mask & gt_is_c) / total)

    ranked = tuple(
        sorted(per_class_residual, key=lambda k: per_class_residual[k], reverse=True)
    )

    return ResidualTarget(
        bulk_argmax_through_R=bulk,
        gt_lstars=gt,
        residual_mask=residual_mask,
        bulk_dseg=float(residual_mask.mean()),
        n_pairs=n_pairs,
        per_class_residual=per_class_residual,
        per_class_area=per_class_area,
        residual_classes_ranked=ranked,
    )


def save_residual_target(rt: ResidualTarget, path: str | Path) -> int:
    """Persist the residual target as a compressed npz (the residual-INR trainer's input).

    Saves the bulk argmax + residual mask (packed bits) + GT lstars so the GPU residual run can
    load them, compute the residual, and train the INR to flip ONLY the residual cells. Returns
    the file size in bytes. (This npz is a TRAINING ARTIFACT, not an archive section — the INR
    WEIGHTS it produces are the COUNTED LEARN bytes.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bulk_argmax_through_R=rt.bulk_argmax_through_R.astype(np.uint8),
        gt_lstars=rt.gt_lstars.astype(np.uint8),
        residual_mask=np.packbits(rt.residual_mask, axis=None),
        residual_mask_shape=np.asarray(rt.residual_mask.shape, np.int64),
        n_pairs=np.asarray(rt.n_pairs, np.int64),
        bulk_dseg=np.asarray(rt.bulk_dseg, np.float64),
    )
    # np.savez_compressed appends .npz if missing
    real = path if path.suffix == ".npz" else path.with_suffix(".npz")
    size = int(real.stat().st_size)
    # also drop a human-readable summary sidecar
    real.with_suffix(".summary.json").write_text(json.dumps(rt.to_summary(), indent=2))
    return size


def load_residual_target(path: str | Path) -> ResidualTarget:
    """Load a residual target saved by :func:`save_residual_target`."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    z = np.load(path, allow_pickle=False)
    shape = tuple(int(x) for x in z["residual_mask_shape"])
    mask = np.unpackbits(z["residual_mask"])[: int(np.prod(shape))].astype(bool).reshape(shape)
    bulk = z["bulk_argmax_through_R"].astype(np.int64)
    gt = z["gt_lstars"].astype(np.int64)
    return compute_residual_target(bulk, gt)
