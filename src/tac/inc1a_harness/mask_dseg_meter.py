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

from tac.boundary_math.bitmask_dseg import d_seg_reference
from tac.witness_control.perclass_verdict import (
    CLASS_NAMES,
    N_CLASSES,
    per_class_dseg_fields,
    per_class_flip_stats,
)

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

    # Aggregate = mean over frames of the exact authority functional (cand != gt).mean().
    per_frame = np.empty(n, dtype=np.float64)
    preds: list[np.ndarray] = []
    for i, (p, g) in enumerate(zip(partitions, gt_list, strict=True)):
        pa = np.asarray(p)
        if pa.shape != g.shape:
            raise MaskDsegMeterError(
                f"frame {i}: partition shape {pa.shape} != L* shape {g.shape}"
            )
        per_frame[i] = d_seg_reference(pa, g)
        preds.append(pa)
    agg = float(per_frame.mean())

    # Per-class via the canonical sensor (flips attributed to GT class; sum identity holds).
    flips, pixels = per_class_flip_stats(preds, gt_list, n_classes=int(n_classes))
    fields = per_class_dseg_fields(flips, pixels)
    names = CLASS_NAMES if int(n_classes) == N_CLASSES else tuple(
        f"class_{c}" for c in range(int(n_classes))
    )
    per_class = {names[c]: float(fields["d_seg_by_class"][c]) for c in range(int(n_classes))}
    share = {names[c]: float(fields["flip_share_by_class"][c]) for c in range(int(n_classes))}

    return MaskDsegResult(
        agg_dseg=agg,
        per_class_dseg=per_class,
        flip_share_by_class=share,
        n_frames=n,
        n_classes=int(n_classes),
        is_n600=is_n600,
        total_flips=int(flips.sum()),
        total_pixels=int(pixels.sum()),
        label=str(label),
        extra={"per_frame_std": float(per_frame.std())},
    )
