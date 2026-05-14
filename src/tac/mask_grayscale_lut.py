# SPDX-License-Identifier: MIT
"""Selfcomp grayscale-LUT mask codec — lane MM rate-attack primitive.

The Selfcomp paradigm (https://github.com/.../selfcomp inflate.py L25-29 +
L175-180) encodes the 5-class SegNet masks as a SINGLE 8-bit grayscale plane
where each class is mapped to a distinct gray level. At inflate time a
Gaussian-softmax LUT projects gray values back into class-probability space,
which is then fed into the renderer / SegMap.

Why this matters for Lane MM
----------------------------
Our existing canonical mask path is 5-channel one-hot AV1 (mask_codec.py).
On Lane A's full-res 384x512 masks the AV1 stream is ~280KB.
The public Selfcomp submission ships a single 1-channel grayscale AV1 stream that is
empirically ~50% smaller (~140KB) for the same 1200 frames at the same
quality, because:

1. Single-channel video skips chroma planes (3-channel YUV would otherwise
   waste bits on monochrome chroma).
2. The 5 grayscale targets [0, 64, 128, 192, 255] are spread across the full
   8-bit range, so AV1's quantizer has 64-pixel gaps (3) and 63-pixel gap (1) to absorb noise — the
   noisy decode still nearest-neighbours back to the correct class.
3. The Gaussian-softmax LUT (sigma=15) at inflate time matches the public
   Selfcomp reference: softmax(exp(-(gray - target)^2 / (2 sigma^2))). This
   is intentionally much softer than a nearest-neighbour one-hot projection;
   the renderer must be trained against this exact analog distribution if the
   soft map is fed directly at inflate time.

CLAUDE.md compliance
--------------------
- Pure encode/decode primitives; no scorer access.
- Outputs are bit-deterministic (CPU-only int math + nearest neighbours).
- The LUT is constructable without GPU; safe for inflate path.

Class -> gray mapping (matches Selfcomp inflate.py CLASS_TARGETS[5]).
These are contest SegNet class IDs, not a new label taxonomy:
    0 (road)          -> 0
    1 (lane_markings) -> 255
    2 (undrivable)    -> 64
    3 (movable)       -> 192
    4 (my_car)        -> 128
"""
from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

from tac.semantic_label_contract import (
    NUM_CONTEST_SEGNET_CLASSES,
    SELFCOMP_CLASS_TO_GRAY,
)


# CLASS_TO_GRAY mirror of Selfcomp/inflate.py CLASS_TARGETS = [0, 255, 64, 192, 128].
# The list-index is the class id; we publish a dict for explicit lookup.
CLASS_TO_GRAY: Final[dict[int, int]] = dict(SELFCOMP_CLASS_TO_GRAY)

# Grayscale targets in class-id order — used by the LUT builder + decoder.
_CLASS_TARGETS_TENSOR: Final[torch.Tensor] = torch.tensor(
    [CLASS_TO_GRAY[c] for c in range(len(CLASS_TO_GRAY))],
    dtype=torch.float32,
)

NUM_CLASSES: Final[int] = NUM_CONTEST_SEGNET_CLASSES
LUT_DEFAULT_SIGMA: Final[float] = 15.0


def encode_masks_grayscale(class_ids: torch.Tensor) -> torch.Tensor:
    """Encode a per-pixel class-id tensor to grayscale uint8.

    Args:
        class_ids: int64 (N, H, W) tensor with values in {0, 1, 2, 3, 4}.

    Returns:
        uint8 (N, H, W) tensor with the matching grayscale targets.

    Raises:
        ValueError: if class_ids has any value outside [0, NUM_CLASSES).
    """
    if class_ids.dtype != torch.int64:
        raise ValueError(
            f"class_ids must be int64, got {class_ids.dtype}. Cast with .long() if needed."
        )
    if class_ids.min() < 0 or class_ids.max() >= NUM_CLASSES:
        raise ValueError(
            f"class_ids must be in [0, {NUM_CLASSES}); got range "
            f"[{int(class_ids.min())}, {int(class_ids.max())}]"
        )
    targets = _CLASS_TARGETS_TENSOR.to(device=class_ids.device, dtype=torch.uint8)
    return targets[class_ids]


def decode_grayscale_to_classes(gray: torch.Tensor) -> torch.Tensor:
    """Decode grayscale uint8 back to class ids via nearest-neighbour matching.

    Used by nearest-neighbour forensic paths and smoke tests. Production
    soft-LUT SegMap inflates should use ``grayscale_to_probability_map`` so
    runtime input matches the training distribution.

    Args:
        gray: (N, H, W) uint8 or int tensor in [0, 255].

    Returns:
        int64 (N, H, W) class-id tensor.
    """
    if gray.dim() < 1:
        raise ValueError(f"gray must be at least 1-D, got shape {tuple(gray.shape)}")
    targets = _CLASS_TARGETS_TENSOR.to(device=gray.device)
    # |gray - target| computed per class, argmin selects nearest target.
    g = gray.to(torch.float32).unsqueeze(-1)  # (..., 1)
    distances = (g - targets).abs()  # (..., NUM_CLASSES)
    return distances.argmin(dim=-1).to(torch.int64)


def create_gaussian_softmax_lut(
    sigma: float = LUT_DEFAULT_SIGMA,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the inflate-time grayscale-to-class-probability LUT.

    Mirrors Selfcomp inflate.py L175-180 (``create_gaussian_softmax_lut``):
    for each gray value v in [0, 255] and each class c in [0, NUM_CLASSES),
    compute lut[v, c] = softmax_c(exp(-((v - target[c])^2) / (2 sigma^2))).

    Args:
        sigma: Gaussian width controlling how fast the bell values decay away
            from class targets. With the Selfcomp bell-softmax form the exact
            target row is deliberately still soft, so renderers must be trained
            against this distribution instead of assuming one-hot masks.
        targets: Optional custom class-target gray values, shape (NUM_CLASSES,)
            float, all in [0, 255]. If None, uses Selfcomp's default fixed
            CLASS_TO_GRAY mapping. Lane LCT supplies a learnable Parameter
            here so the codebook adapts during SegMap training.

    Returns:
        (256, NUM_CLASSES) float32 tensor with each row summing to 1.

    Raises:
        ValueError: if sigma <= 0; if targets has wrong shape or out-of-range.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if targets is None:
        targets_t = _CLASS_TARGETS_TENSOR
    else:
        targets_t = torch.as_tensor(targets, dtype=torch.float32)
        if targets_t.shape != (NUM_CLASSES,):
            raise ValueError(
                f"targets must have shape ({NUM_CLASSES},), got {tuple(targets_t.shape)}"
            )
        if not torch.all((targets_t >= 0) & (targets_t <= 255)):
            raise ValueError(
                f"targets values must be in [0, 255], got {targets_t.tolist()}"
            )
    x = torch.arange(256, dtype=torch.float32).unsqueeze(1)  # (256, 1)
    targets_b = targets_t.unsqueeze(0)  # (1, NUM_CLASSES)
    squared_diff = (x - targets_b) ** 2
    bell = torch.exp(-squared_diff / (2.0 * sigma * sigma))
    return F.softmax(bell, dim=1)


def grayscale_to_probability_map(
    gray: torch.Tensor,
    *,
    sigma: float = LUT_DEFAULT_SIGMA,
    targets: torch.Tensor | None = None,
    channel_first: bool = True,
) -> torch.Tensor:
    """Project uint8 grayscale masks to Selfcomp-style class probabilities.

    The returned probabilities are the exact LUT distribution produced by
    ``create_gaussian_softmax_lut``. This helper is the train/inflate parity
    point for grayscale lanes: use it whenever a renderer consumes the soft
    analog mask map directly.

    Args:
        gray: 2-D ``(H, W)``, 3-D ``(N, H, W)``, or 4-D ``(B, T, H, W)``
            grayscale tensor. Values are clamped to [0, 255] before lookup.
        sigma: LUT sigma.
        targets: Optional class gray targets.
        channel_first: If true, return ``(C, H, W)``, ``(N, C, H, W)``, or
            ``(B, T, C, H, W)``. If false, keep the embedding channel last.

    Returns:
        Float32 probability tensor with class probabilities summing to one.
    """
    if gray.dim() not in (2, 3, 4):
        raise ValueError(
            f"gray must have 2, 3, or 4 dims, got shape {tuple(gray.shape)}"
        )
    lut = create_gaussian_softmax_lut(sigma=sigma, targets=targets).to(
        device=gray.device
    )
    gray_idx = gray.to(torch.long).clamp(0, 255)
    probs = F.embedding(gray_idx, lut)
    if not channel_first:
        return probs
    if gray.dim() == 2:
        return probs.permute(2, 0, 1).contiguous()
    if gray.dim() == 3:
        return probs.permute(0, 3, 1, 2).contiguous()
    return probs.permute(0, 1, 4, 2, 3).contiguous()


__all__ = [
    "CLASS_TO_GRAY",
    "NUM_CLASSES",
    "LUT_DEFAULT_SIGMA",
    "encode_masks_grayscale",
    "decode_grayscale_to_classes",
    "create_gaussian_softmax_lut",
    "grayscale_to_probability_map",
]
