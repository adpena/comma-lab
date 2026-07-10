# SPDX-License-Identifier: MIT
"""compare — the canonical "label-stack vs L* -> agg + per-class d_seg" back-half.

ONE fact, ONE place (P1 one-fact-one-store, at the code level). Two meters computed
the SAME comparison independently: :func:`tac.through_r.harness.measure_through_r`
(the through-R meter: candidate frames -> R -> SegNet argmax -> compare) and
:func:`tac.inc1a_harness.mask_dseg_meter.measure_mask_dseg` (the mask-level meter:
composite tropical-argmax partitions -> compare). They differ ONLY in the FRONT half
(how the predicted label stack is produced); the BACK half — per-pair
:func:`tac.boundary_math.bitmask_dseg.d_seg_reference` averaged into the aggregate,
plus the canonical per-class decomposition
:func:`tac.witness_control.perclass_verdict.per_class_flip_stats` /
:func:`per_class_dseg_fields` — was duplicated. This module is that shared back half.

NO-FAKE / REUSE-not-rederive: NOTHING is re-derived here. The aggregate is the
authority functional ``(cand != gt).mean()`` averaged over pairs; the per-class rate /
share are the canonical sensor's exact outputs (``sum(flips)/sum(pixels) == d_seg`` by
construction). Delegating both meters to this function is numeric-identity-preserving:
it is literally the same function calls in the same order they made inline.

This module is pure numpy (no torch, no R) — it operates on already-realized INT label
stacks. Shape validation is the CALLER's responsibility (each meter keeps its own
error type + front-half validation); this helper assumes validated equal-shape inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tac.boundary_math.bitmask_dseg import d_seg_reference
from tac.witness_control.perclass_verdict import (
    CLASS_NAMES,
    N_CLASSES,
    per_class_dseg_fields,
    per_class_flip_stats,
)

__all__ = ["LabelStackCompare", "compare_label_stack_to_lstars"]


@dataclass
class LabelStackCompare:
    """The shared compare result: aggregate + per-class d_seg over a label stack vs L*.

    Every field is what BOTH ``measure_through_r`` and ``measure_mask_dseg`` computed
    inline. Callers map these onto their own result dataclasses (keeping their own
    extra-key names, e.g. ``per_pair_std`` vs ``per_frame_std``).
    """

    agg_dseg: float
    per_class_dseg: dict[str, float]
    flip_share_by_class: dict[str, float]
    per_pair_dseg: list[float]
    total_flips: int
    total_pixels: int
    per_pair_std: float
    per_pair: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]


def _class_names(n_classes: int) -> tuple[str, ...]:
    """Canonical 5-class names, else generic ``class_<c>`` (mirrors both meters)."""
    if int(n_classes) == N_CLASSES:
        return CLASS_NAMES
    return tuple(f"class_{c}" for c in range(int(n_classes)))


def compare_label_stack_to_lstars(
    pred_list: list[np.ndarray],
    gt_list: list[np.ndarray],
    *,
    n_classes: int = N_CLASSES,
) -> LabelStackCompare:
    """Compare a predicted INT label stack against L* -> aggregate + per-class d_seg.

    ``pred_list`` / ``gt_list``: equal-length lists of ``(H, W)`` int label maps (the
    predicted argmax and the GT ``L*``). Per-pair :func:`d_seg_reference` averaged ->
    ``agg_dseg``; :func:`per_class_flip_stats` over all pairs -> per-class rate + flip
    share (via :func:`per_class_dseg_fields`). Deterministic; pure numpy.

    Numeric-identity note: ``per_pair`` is a float64 array so ``.mean()`` / ``.std()``
    (ddof=0) reproduce both meters' prior ``float(np.mean(list))`` /
    ``float(arr.mean())`` and ``float(np.std(list))`` / ``float(arr.std())`` exactly.
    """

    n = len(pred_list)
    if n != len(gt_list):
        raise ValueError(f"pred_list ({n}) and gt_list ({len(gt_list)}) length mismatch")
    if n == 0:
        raise ValueError("no frames to compare (empty label stack)")

    per_pair = np.empty(n, dtype=np.float64)
    for i in range(n):
        per_pair[i] = d_seg_reference(pred_list[i], gt_list[i])
    agg = float(per_pair.mean())

    flips, pixels = per_class_flip_stats(pred_list, gt_list, n_classes=int(n_classes))
    fields = per_class_dseg_fields(flips, pixels)
    names = _class_names(int(n_classes))
    per_class = {names[c]: float(fields["d_seg_by_class"][c]) for c in range(int(n_classes))}
    share = {names[c]: float(fields["flip_share_by_class"][c]) for c in range(int(n_classes))}

    return LabelStackCompare(
        agg_dseg=agg,
        per_class_dseg=per_class,
        flip_share_by_class=share,
        per_pair_dseg=[float(x) for x in per_pair],
        total_flips=int(flips.sum()),
        total_pixels=int(pixels.sum()),
        per_pair_std=float(per_pair.std()),
        per_pair=per_pair,
    )
