# SPDX-License-Identifier: MIT
"""Kahan-summation Conv2d for MPS-vs-CUDA reduction-ordering noise correction.

Per slot 9 formalization (`feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`)
the HIGHEST-EV (10x) engineering correction targeting the **dominant** noise
source identified by slot 2's granular analyzer (CV=2.6% uniform across
pairs → reduction-ordering noise, not per-pair-structural noise).

Mathematical background (Higham 2002, *Accuracy and Stability of Numerical
Algorithms*, chapter 4):

    Naive summation of N floats accumulates roundoff at O(eps * sqrt(N))
    when summed in random order; Metal MPS uses parallel reduction whose
    ordering differs from CUDA's atomic-add ordering. Kahan summation
    tracks a running "compensation" sum of the lost precision and adds it
    back at each step, reducing the accumulated error to O(eps) regardless
    of N or ordering.

    For SegNet (Conv2d stride-2 stem on 384x512 frames; ~50 layers; ~140K
    params; accumulation depth N ~ 384*512 = 196,608 per output channel),
    naive reduction error: 5e-8 * sqrt(196608) = 2.2e-5. Kahan reduction
    error: 5e-8 (independent of N). Predicted drift reduction: ~440x in
    the worst layer; ~10x in the aggregate-weighted gap per slot 9 §4.6.

The implementation strategy:

1. Decompose Conv2d into the canonical im2col + matmul + reshape pipeline.
2. For the matmul accumulation phase, replace the standard inner-product
   sum with a Kahan-compensated sum per output element.
3. Return same dtype as input (no precision contamination upstream).

This is a **compress-time only** correction per CLAUDE.md "MPS auth eval is
NOISE" + the strict-scorer-rule (Catalog #6). Inflate-time scorer load
remains forbidden; this helper is for compress-time MPS substrate training
and forward-pass diagnostics.

Public API:
    - `kahan_conv2d(input, weight, bias, ...)` — drop-in replacement for
      `torch.nn.functional.conv2d` with Kahan-compensated accumulation.
    - `patch_conv2d_to_kahan_for_mps_globally()` — monkey-patch
      `torch.nn.functional.conv2d` to route through `kahan_conv2d` when
      `input.device.type == "mps"`. Compress-time only.
    - `restore_torch_conv2d()` — undo monkey-patch.
    - `KAHAN_CONV2D_AXIS_TAG` / `KAHAN_CONV2D_EVIDENCE_GRADE` — Provenance
      tags per Catalog #287/#323.

Sister of:
    - `tac.mps_diagnostic.drift_predictor` (slot 9; predictor whose
      predicted gap reduction this helper realizes).
    - `tac.mps_diagnostic.pinned_softmax` (sister HIGH-EV correction).
    - `tac.mps_diagnostic.fp32_matmul_override` (sister MEDIUM-EV correction).

Per Catalog #229 PV: `torch.nn.functional.conv2d` is a builtin (no Python
signature inspection); we mirror its documented signature from the PyTorch
2.11 docs.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "KAHAN_CONV2D_AXIS_TAG",
    "KAHAN_CONV2D_EVIDENCE_GRADE",
    "kahan_conv2d",
    "kahan_sum",
    "patch_conv2d_to_kahan_for_mps_globally",
    "restore_torch_conv2d",
]

# Provenance tags per Catalog #287 / #323 — every emitted tensor (when used
# in a downstream rate or loss claim) is non-promotable until a paired Linux
# x86_64 anchor lands per CLAUDE.md "MPS auth eval is NOISE".
KAHAN_CONV2D_AXIS_TAG: str = "[macOS-MPS-Kahan-corrected-PyTorch]"
KAHAN_CONV2D_EVIDENCE_GRADE: str = "macOS-MPS-Kahan-corrected-diagnostic"

# Module-level state for the monkey-patch path; preserves the original
# `torch.nn.functional.conv2d` so :func:`restore_torch_conv2d` can undo.
_ORIGINAL_CONV2D: Optional[Callable[..., torch.Tensor]] = None


def kahan_sum(
    summands: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Kahan-compensated sum along a single dimension.

    Implements Higham 2002 chapter 4 algorithm 4.2 (Kahan summation) in
    vectorized form across all OTHER tensor dimensions in parallel.

    For each input position, the compensation `c` is the running
    accumulator of low-order bits lost in the previous addition; it is
    subtracted from the next summand before accumulation.

    Returns the sum with same dtype as ``summands``.
    """
    if dim < 0:
        dim = summands.dim() + dim
    if not 0 <= dim < summands.dim():
        raise ValueError(f"dim {dim} out of range for tensor with {summands.dim()} dims")

    n_terms = summands.shape[dim]
    # Move the summation axis to position 0 for efficient iteration.
    permuted = summands.movedim(dim, 0)
    sum_acc = torch.zeros_like(permuted[0])
    compensation = torch.zeros_like(permuted[0])

    for i in range(n_terms):
        # Algorithm 4.2:
        #   y = x[i] - c
        #   t = sum + y
        #   c = (t - sum) - y
        #   sum = t
        y = permuted[i] - compensation
        t = sum_acc + y
        compensation = (t - sum_acc) - y
        sum_acc = t

    if keepdim:
        # Insert the reduced axis back as size-1 so output has same rank.
        sum_acc = sum_acc.unsqueeze(0).movedim(0, dim)
    return sum_acc


def _normalize_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a scalar-or-pair stride / padding / dilation argument."""
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def kahan_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    *,
    enable_kahan: bool = True,
) -> torch.Tensor:
    """Conv2d with Kahan-compensated accumulation.

    Drop-in replacement for ``torch.nn.functional.conv2d`` whose
    accumulation phase tracks a running compensation per Kahan 1965 /
    Higham 2002. When ``enable_kahan=False`` (or when the input is not
    on MPS), falls back to the standard ``F.conv2d`` path.

    Args:
        input: input tensor of shape ``(N, C_in, H, W)``.
        weight: convolution weights of shape ``(C_out, C_in / groups, kH, kW)``.
        bias: optional bias of shape ``(C_out,)``.
        stride: convolution stride.
        padding: convolution padding.
        dilation: convolution dilation.
        groups: convolution groups.
        enable_kahan: if ``False``, falls through to ``F.conv2d`` unchanged.

    Returns:
        Output tensor with same dtype as ``input``.

    Per slot 9 formalization: predicted ~10x drift reduction vs naive
    F.conv2d on MPS for Conv2d-heavy networks (SegNet, PoseNet, NeRV-class
    renderers).

    Implementation uses ``unfold`` to materialize the im2col patches, then
    accumulates via :func:`kahan_sum`. Memory cost is ``C_in*kH*kW`` times
    the output spatial size; recommended only for compress-time training
    of small substrates (<10M params).
    """
    if not enable_kahan or input.device.type != "mps":
        # Fallback to canonical path; the helper is a no-op for non-MPS
        # tensors and when explicitly disabled.
        original = _ORIGINAL_CONV2D if _ORIGINAL_CONV2D is not None else F.conv2d
        return original(input, weight, bias, stride, padding, dilation, groups)

    if groups != 1:
        # Grouped convolutions are uncommon in scorer networks; fall through
        # to canonical path to avoid a complex split/concat plumbing in the
        # Kahan path. Future work can extend.
        original = _ORIGINAL_CONV2D if _ORIGINAL_CONV2D is not None else F.conv2d
        return original(input, weight, bias, stride, padding, dilation, groups)

    stride_pair = _normalize_pair(stride)
    padding_pair = _normalize_pair(padding)
    dilation_pair = _normalize_pair(dilation)

    n, c_in, _h_in, _w_in = input.shape
    c_out, _c_in_per_group, k_h, k_w = weight.shape

    # im2col: (N, C_in * kH * kW, L) where L = H_out * W_out.
    patches = F.unfold(
        input,
        kernel_size=(k_h, k_w),
        stride=stride_pair,
        padding=padding_pair,
        dilation=dilation_pair,
    )

    # weight: (C_out, C_in * kH * kW)
    weight_flat = weight.reshape(c_out, c_in * k_h * k_w)

    # Standard fused einsum: output[n, c_out, l] = sum_k weight_flat[c_out, k] * patches[n, k, l]
    # Kahan-summed variant: explicitly accumulate over k with compensation.
    # patches: (N, K, L); weight_flat: (C_out, K)
    # Materialize the per-term product (N, C_out, K, L) and Kahan-sum on K.
    n_k = patches.shape[1]
    # Reshape for broadcast: (N, 1, K, L) * (1, C_out, K, 1) -> (N, C_out, K, L)
    per_term = patches.unsqueeze(1) * weight_flat.unsqueeze(0).unsqueeze(-1)
    # Kahan-sum on axis K (dim=2).
    summed = kahan_sum(per_term, dim=2)
    # summed shape: (N, C_out, L)

    if bias is not None:
        summed = summed + bias.view(1, c_out, 1)

    # Reshape (N, C_out, L) -> (N, C_out, H_out, W_out)
    h_out = (input.shape[2] + 2 * padding_pair[0] - dilation_pair[0] * (k_h - 1) - 1) // stride_pair[0] + 1
    w_out = (input.shape[3] + 2 * padding_pair[1] - dilation_pair[1] * (k_w - 1) - 1) // stride_pair[1] + 1
    output = summed.reshape(n, c_out, h_out, w_out)
    return output


def patch_conv2d_to_kahan_for_mps_globally() -> None:
    """Monkey-patch ``torch.nn.functional.conv2d`` to route through
    :func:`kahan_conv2d` when ``input.device.type == "mps"``.

    Use at COMPRESS TIME ONLY. Inflate time must not load scorers per
    CLAUDE.md strict-scorer-rule (Catalog #6).

    Idempotent: calling twice has the same effect as calling once.
    """
    global _ORIGINAL_CONV2D
    if _ORIGINAL_CONV2D is not None:
        # Already patched.
        return
    _ORIGINAL_CONV2D = F.conv2d

    def _dispatched_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> torch.Tensor:
        if input.device.type == "mps":
            return kahan_conv2d(
                input, weight, bias, stride, padding, dilation, groups,
                enable_kahan=True,
            )
        # Non-MPS path: unchanged behavior.
        assert _ORIGINAL_CONV2D is not None  # for type narrowing
        return _ORIGINAL_CONV2D(input, weight, bias, stride, padding, dilation, groups)

    F.conv2d = _dispatched_conv2d  # type: ignore[assignment]


def restore_torch_conv2d() -> None:
    """Undo :func:`patch_conv2d_to_kahan_for_mps_globally`.

    Idempotent: calling when no patch is active is a no-op.
    """
    global _ORIGINAL_CONV2D
    if _ORIGINAL_CONV2D is None:
        return
    F.conv2d = _ORIGINAL_CONV2D  # type: ignore[assignment]
    _ORIGINAL_CONV2D = None
