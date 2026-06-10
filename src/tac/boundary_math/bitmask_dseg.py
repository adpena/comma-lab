# SPDX-License-Identifier: MIT
"""Exact ``d_seg`` as a bitmask popcount(XOR) functional.

Source spec §3 (the right data structures) + §2 (the combinatorial/SET functional):

    d_seg = mean_pixels [ argmax S(F1) != L* ]

This is EXACTLY a per-pixel inequality of two integer label arrays.  evaluate.py
computes it as ``(argmax(out1) != argmax(out2)).mean()`` (verified against
``upstream/modules.py`` DistortionNet seg term).  Decomposed per-class, the same
quantity is ``popcount(XOR(mask_candidate, mask_gt)) / (2 * N)`` summed over the 5
class one-hot masks — but a flip touches exactly TWO class masks (the old label and
the new label), so the per-class XOR sum double-counts; dividing by 2 recovers the
pixel-flip rate.  Both forms are provided + cross-checked by tests.

NO FAKE: ``d_seg_reference`` is the direct argmax-compare (the authority form);
``d_seg_bitmask`` is the popcount form; tests assert they agree bit-for-bit on real
and synthetic partitions.  Neither is a constant.
"""

from __future__ import annotations

import numpy as np

N_SEG_CLASSES = 5


def class_masks_from_argmax(argmax_hw: np.ndarray, n_classes: int = N_SEG_CLASSES) -> np.ndarray:
    """One-hot the integer label map into ``(n_classes, H, W)`` boolean masks.

    Each pixel ``p`` sets exactly one class plane true (``argmax_hw[p]``).  Class
    ids must be in ``[0, n_classes)``.
    """

    a = np.asarray(argmax_hw)
    if a.ndim != 2:
        raise ValueError(f"argmax must be (H, W); got shape {a.shape}")
    if a.min() < 0 or a.max() >= n_classes:
        raise ValueError(f"class ids must be in [0,{n_classes}); got [{a.min()},{a.max()}]")
    masks = np.zeros((n_classes, *a.shape), dtype=bool)
    for c in range(n_classes):
        masks[c] = a == c
    return masks


def d_seg_reference(candidate_argmax_hw: np.ndarray, gt_argmax_hw: np.ndarray) -> float:
    """The authority form: per-pixel argmax-disagreement rate.

    This is the exact functional evaluate.py scores: ``(cand != gt).mean()``.
    """

    cand = np.asarray(candidate_argmax_hw)
    gt = np.asarray(gt_argmax_hw)
    if cand.shape != gt.shape:
        raise ValueError(f"shape mismatch: candidate {cand.shape} vs gt {gt.shape}")
    return float(np.mean(cand != gt))


def d_seg_bitmask(
    candidate_argmax_hw: np.ndarray,
    gt_argmax_hw: np.ndarray,
    n_classes: int = N_SEG_CLASSES,
) -> float:
    """The bitmask form: ``sum_c popcount(XOR(mask_c, gt_mask_c)) / (2*N)``.

    A single pixel flip from class a -> class b toggles exactly two class planes
    (plane a turns off, plane b turns on), so the per-class XOR popcount sum equals
    ``2 * (#flipped pixels)``; dividing by ``2*N`` recovers the flip RATE.  Equals
    :func:`d_seg_reference` exactly (tested), but operates on the bitmask data
    structure the spec calls for (reusable by STC/RLE contour coders + popcount HW).
    """

    cand_masks = class_masks_from_argmax(candidate_argmax_hw, n_classes)
    gt_masks = class_masks_from_argmax(gt_argmax_hw, n_classes)
    n_pixels = cand_masks.shape[1] * cand_masks.shape[2]
    # XOR over boolean planes; popcount == count of True bits.
    xor_bits = int(np.count_nonzero(cand_masks ^ gt_masks))
    return xor_bits / (2.0 * n_pixels)


def flip_count(candidate_argmax_hw: np.ndarray, gt_argmax_hw: np.ndarray) -> int:
    """Number of flipped pixels (the seg-debt count, not the rate)."""

    cand = np.asarray(candidate_argmax_hw)
    gt = np.asarray(gt_argmax_hw)
    if cand.shape != gt.shape:
        raise ValueError(f"shape mismatch: candidate {cand.shape} vs gt {gt.shape}")
    return int(np.count_nonzero(cand != gt))
