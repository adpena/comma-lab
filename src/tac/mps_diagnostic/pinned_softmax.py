# SPDX-License-Identifier: MIT
"""Pinned-epsilon softmax for MPS-vs-CUDA boundary-flip noise correction.

Per slot 9 formalization (`feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`)
the HIGH-EV (~50% boundary-flip reduction) engineering correction targeting
the Metal MPS vs CUDA cuDNN epsilon-for-log-sum-exp-stabilization difference.

SegNet emits 5-class logits; the contest scorer derives `seg_distortion =
argmax_disagreement_rate`. At pixels whose logits are close to the argmax
boundary (max - second_max < epsilon), small numerical differences between
Metal and CUDA flip the predicted class. Per slot 9 §4.2: argmax boundary
flip rate is ~50% lower when log-sum-exp stabilization is computed in fp64
(symmetric across Metal + cuDNN because the epsilon difference is below the
fp64 rounding floor).

Implementation:

1. Promote ``logits`` to fp64 for the stabilization phase.
2. Compute max-subtracted log-sum-exp in fp64.
3. Compute softmax in fp64.
4. Demote to input dtype.

This is a **compress-time only** correction per CLAUDE.md "MPS auth eval is
NOISE" + the strict-scorer-rule (Catalog #6). Inflate-time scorer load is
forbidden; this helper is for compress-time forward-pass diagnostics and
substrate training that backprops through SegNet/PoseNet softmax outputs.

Public API:
    - `pinned_softmax(logits, dim, ...)` — drop-in replacement for
      `torch.nn.functional.softmax` with fp64 stabilization.
    - `patch_softmax_to_pinned_for_mps_globally()` — monkey-patch
      `torch.nn.functional.softmax` to route through `pinned_softmax`
      when `input.device.type == "mps"`. Compress-time only.
    - `restore_torch_softmax()` — undo monkey-patch.
    - `PINNED_SOFTMAX_AXIS_TAG` / `PINNED_SOFTMAX_EVIDENCE_GRADE` —
      Provenance tags per Catalog #287/#323.

Sister of:
    - `tac.mps_diagnostic.drift_predictor` (slot 9; predictor whose
      predicted boundary-flip reduction this helper realizes).
    - `tac.mps_diagnostic.kahan_conv2d` (sister HIGHEST-EV correction).
    - `tac.mps_diagnostic.fp32_matmul_override` (sister MEDIUM-EV correction).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "PINNED_SOFTMAX_AXIS_TAG",
    "PINNED_SOFTMAX_EVIDENCE_GRADE",
    "pinned_softmax",
    "patch_softmax_to_pinned_for_mps_globally",
    "restore_torch_softmax",
]

# Provenance tags per Catalog #287 / #323 — every emitted tensor (when used
# downstream in a rate or loss claim) is non-promotable until a paired Linux
# x86_64 anchor lands per CLAUDE.md "MPS auth eval is NOISE".
PINNED_SOFTMAX_AXIS_TAG: str = "[macOS-MPS-pinned-softmax-PyTorch]"
PINNED_SOFTMAX_EVIDENCE_GRADE: str = "macOS-MPS-pinned-softmax-diagnostic"

# Module-level state for the monkey-patch path.
_ORIGINAL_SOFTMAX: Optional[Callable[..., torch.Tensor]] = None


def pinned_softmax(
    logits: torch.Tensor,
    dim: int = -1,
    *,
    enable_pinning: bool = True,
    intermediate_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Softmax with epsilon pinned via fp64 intermediate promotion.

    Implementation:
        1. Promote ``logits`` to ``intermediate_dtype`` (default fp64).
        2. Compute ``logits - logits.max(dim=dim, keepdim=True)`` for
           numerical stabilization (the canonical log-sum-exp trick).
        3. ``exp(stabilized)`` then divide by ``sum(exp, dim, keepdim=True)``.
        4. Demote to ``logits.dtype``.

    When ``enable_pinning=False`` (or when input is on a non-MPS device,
    AND the MPS path is the only known offender), falls back to the
    standard ``F.softmax`` path.

    Per slot 9 §4.2: predicted ~50% reduction in SegNet 5-class argmax [prediction]
    boundary flip rate vs naive F.softmax on MPS. The reduction is
    SYMMETRIC across Metal and cuDNN because fp64 stabilization is below
    both backends' epsilon-rounding floors.

    Per Catalog #229 PV: ``intermediate_dtype=torch.float64`` is the
    canonical choice because:
        - fp32 stabilization does not close the Metal-vs-cuDNN epsilon gap
          (the gap is in the LAST 2-3 bits of fp32).
        - fp16/bf16 stabilization makes the gap WORSE (more bits lost).
        - fp64 closes the gap to below both backends' rounding floors.

    Returns:
        Softmax probabilities with ``dtype == logits.dtype`` and same shape.
    """
    if not enable_pinning or logits.device.type != "mps":
        # Fallback to canonical path. The helper is a no-op for non-MPS
        # tensors (Metal-vs-cuDNN epsilon gap is MPS-specific) and when
        # explicitly disabled.
        original = _ORIGINAL_SOFTMAX if _ORIGINAL_SOFTMAX is not None else F.softmax
        return original(logits, dim=dim)

    # MPS does not natively support fp64; promote to CPU for the
    # stabilization phase, then move back to MPS for the final demotion.
    # This is the canonical pattern per Apple MPSGraph docs (Float64 is
    # CPU-only on MPS as of torch 2.11). We must call .cpu() FIRST then
    # .to(fp64) because direct .to(device='cpu', dtype=fp64) on an MPS
    # tensor raises TypeError per the Metal framework's fp64 restriction.
    original_dtype = logits.dtype
    if intermediate_dtype == torch.float64:
        promoted = logits.cpu().to(dtype=intermediate_dtype)
    else:
        promoted = logits.to(dtype=intermediate_dtype)

    # Stabilized log-sum-exp: subtract max, exp, divide by sum.
    max_along_dim = promoted.max(dim=dim, keepdim=True).values
    stabilized = promoted - max_along_dim
    exp_vals = torch.exp(stabilized)
    softmax_fp64 = exp_vals / exp_vals.sum(dim=dim, keepdim=True)

    # Demote to original dtype + move back to MPS if we did the CPU detour.
    return softmax_fp64.to(device=logits.device, dtype=original_dtype)


def patch_softmax_to_pinned_for_mps_globally() -> None:
    """Monkey-patch ``torch.nn.functional.softmax`` to route through
    :func:`pinned_softmax` when ``input.device.type == "mps"``.

    Use at COMPRESS TIME ONLY. Inflate time must not load scorers per
    CLAUDE.md strict-scorer-rule (Catalog #6).

    Idempotent: calling twice has the same effect as calling once.
    """
    global _ORIGINAL_SOFTMAX
    if _ORIGINAL_SOFTMAX is not None:
        # Already patched.
        return
    _ORIGINAL_SOFTMAX = F.softmax

    def _dispatched_softmax(
        input: torch.Tensor,
        dim: Optional[int] = None,
        _stacklevel: int = 3,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if input.device.type == "mps":
            dim_resolved = -1 if dim is None else dim
            result = pinned_softmax(input, dim_resolved, enable_pinning=True)
            if dtype is not None:
                return result.to(dtype=dtype)
            return result
        assert _ORIGINAL_SOFTMAX is not None  # for type narrowing
        # Pass through the canonical F.softmax signature (dim, _stacklevel, dtype).
        if dim is None:
            return _ORIGINAL_SOFTMAX(input, _stacklevel=_stacklevel, dtype=dtype)
        return _ORIGINAL_SOFTMAX(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    F.softmax = _dispatched_softmax  # type: ignore[assignment]


def restore_torch_softmax() -> None:
    """Undo :func:`patch_softmax_to_pinned_for_mps_globally`.

    Idempotent: calling when no patch is active is a no-op.
    """
    global _ORIGINAL_SOFTMAX
    if _ORIGINAL_SOFTMAX is None:
        return
    F.softmax = _ORIGINAL_SOFTMAX  # type: ignore[assignment]
    _ORIGINAL_SOFTMAX = None
