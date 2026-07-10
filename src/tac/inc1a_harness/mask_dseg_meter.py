# SPDX-License-Identifier: MIT
"""The MASK-level d_seg meter — composite partition vs L\\* on gt_n600, per-class + aggregate.

increment-1a item 2 (SYNTHESIS_v2_v8 §B ``measure_1a``). PAINT-FREE: this measures the
composite tropical-argmax MASK partition against the GT SegNet argmax ``L\\*`` (the
``lstars`` array of ``gt_n600.npz`` — ``(600, 384, 512)`` int64), NOT through R. It is a
NECESSARY-CONDITION proxy (mask-optimal ≠ score-optimal, risk-3).

NO-FAKE / REUSE-not-rederive: the aggregate is the authority functional
:func:`tac.boundary_math.bitmask_dseg.d_seg_reference` (``(cand != gt).mean()``, the exact
form ``evaluate.py`` scores); the per-class decomposition is the canonical sensor
:func:`tac.witness_control.perclass_verdict.per_class_flip_stats` /
:func:`per_class_dseg_fields` (``sum(flips)/sum(pixels) == d_seg`` by construction). No
d_seg is re-derived here.

n600 discipline (OPERATOR PRIORITY — allergic to non-n600 / toys): ``require_n600=True`` by
DEFAULT — the meter REFUSES a subset for any measurement that will inform a verdict. A
subset is admissible only with ``require_n600=False`` (explicitly labeled non-authority).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tac.through_r.compare import compare_label_stack_to_lstars
from tac.witness_control.perclass_verdict import N_CLASSES

N600 = 600


class MaskDsegMeterError(ValueError):
    """Raised when the meter is asked to measure a toy / mis-shaped input."""


@dataclass
class MaskDsegResult:
    """The measured mask-level d_seg for ONE arm's composite partitions vs L\\*."""

    agg_dseg: float  # mean over frames of (partition != L*).mean()
    per_class_dseg: dict[str, float]  # class-name -> flips_c / pixels_c
    flip_share_by_class: dict[str, float]  # class-name -> flips_c / total_flips
    n_frames: int
    n_classes: int
    is_n600: bool
    total_flips: int = 0
    total_pixels: int = 0
    label: str = ""  # provenance tag, e.g. "[analytic-generators, no-trained-fields]"
    extra: dict[str, float] = field(default_factory=dict)


def measure_mask_dseg(
    partitions: list[np.ndarray],
    lstars: np.ndarray,
    *,
    n_classes: int = N_CLASSES,
    require_n600: bool = True,
    label: str = "",
) -> MaskDsegResult:
    """Measure MASK d_seg of ``partitions`` (composite argmax) vs ``lstars`` (L\\*).

    ``partitions``: list of ``(H, W)`` int label maps (one per frame). ``lstars``:
    ``(N, H, W)`` int GT argmax stack (or a list of ``(H, W)``). Per-frame
    :func:`d_seg_reference` averaged → ``agg_dseg``; :func:`per_class_flip_stats` over all
    frames → per-class rate + flip share. RAISES if ``require_n600`` and ``N != 600`` (the
    toy-refusal gate), or on any shape mismatch.
    """

    gt_list = [np.asarray(lstars[i]) for i in range(len(lstars))] if isinstance(
        lstars, np.ndarray
    ) else [np.asarray(g) for g in lstars]
    if len(partitions) != len(gt_list):
        raise MaskDsegMeterError(
            f"n_partitions {len(partitions)} != n_gt {len(gt_list)}"
        )
    n = len(partitions)
    if n == 0:
        raise MaskDsegMeterError("no frames to measure (empty partitions)")
    is_n600 = n == N600
    if require_n600 and not is_n600:
        raise MaskDsegMeterError(
            f"n600 discipline: measurement needs N=600 to inform a verdict; got N={n}. "
            f"Pass require_n600=False ONLY for an explicitly-labeled non-authority subset."
        )

    # Front-half validation (KEEP the mask meter's own error type + shape check); then
    # delegate the compare BACK-half to the canonical through_r helper (P1 one-fact-one-
    # place). Numeric-identity-preserving: the helper runs the SAME d_seg_reference /
    # per_class_flip_stats / per_class_dseg_fields calls this meter ran inline (agg mean +
    # per_frame_std reproduce exactly under float64 ddof=0).
    preds: list[np.ndarray] = []
    for i, (p, g) in enumerate(zip(partitions, gt_list, strict=True)):
        pa = np.asarray(p)
        if pa.shape != g.shape:
            raise MaskDsegMeterError(
                f"frame {i}: partition shape {pa.shape} != L* shape {g.shape}"
            )
        preds.append(pa)

    cmp = compare_label_stack_to_lstars(preds, gt_list, n_classes=int(n_classes))

    return MaskDsegResult(
        agg_dseg=cmp.agg_dseg,
        per_class_dseg=cmp.per_class_dseg,
        flip_share_by_class=cmp.flip_share_by_class,
        n_frames=n,
        n_classes=int(n_classes),
        is_n600=is_n600,
        total_flips=cmp.total_flips,
        total_pixels=cmp.total_pixels,
        label=str(label),
        extra={"per_frame_std": cmp.per_pair_std},
    )
