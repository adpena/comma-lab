# SPDX-License-Identifier: MIT
"""Frontier runtime patches that let the PINNED upstream scorer run on torch-MPS
(the Apple GPU) — WITHOUT editing the upstream snapshot.

Sibling of :mod:`tac.differentiable_eval_roundtrip` (which patches the upstream
``rgb_to_yuv6`` to be DIFFERENTIABLE). This module's concern is orthogonal:
DEVICE COMPATIBILITY — making the upstream PyTorch modules' forward+backward run
on the ``mps`` backend, where a few ops have stride/contiguity gaps that CPU/CUDA
tolerate.

Why this exists (operator 2026-06-12, "do whatever upstream did but for macOS"):
upstream ``modules.py`` itself selects ``cuda -> mps -> cpu`` and carries a
``# TODO run in bfloat16?`` — it is DESIGNED to run on the Apple GPU. Measured:
the upstream ``DistortionNet`` forward+backward at B=8, full camera resolution is
**~104x faster on MPS-fp32 than torch-CPU** (17.3 s/step -> 0.166 s/step), with a
per-step gradient cosine of ~1.0 vs the torch-CPU authority. (fp16/bf16 autocast
are both SLOWER on MPS and have worse gradients — fp32 is the sweet spot.)

The ONLY blocker was a single BatchNorm backward stride quirk:
``NativeBatchNormBackward0`` raises ``view size is not compatible with input
tensor's size and stride`` on MPS when the saved tensor is non-contiguous (it
surfaces in the SegNet smp-Unet decoder's BN layers). Forcing the BN input
contiguous on MPS fixes it with ZERO numerical change (memory layout only).

Authority discipline (CLAUDE.md "MPS auth eval is NOISE"): MPS here is a TRAINING
GRADIENT backend ONLY. The EXACT d_seg/d_pose for any decision remain torch-CPU
(or contest CPU/CUDA). The MPS gradient must still pass the descent-equivalence
acceptance gate (BOTH d_seg AND d_pose) before it is trusted for a real run — the
~1.0 per-step cosine is a first sanity, not the verdict.
"""

from __future__ import annotations

import torch

_BN_PATCH_FLAG = "_tac_bn_mps_contig_patched"


def patch_batchnorm_contiguous_for_mps() -> bool:
    """Wrap ``torch.nn.functional.batch_norm`` so MPS inputs are made contiguous
    before the call. Idempotent. Returns True if it applied the patch (False if
    already patched). Numerics-preserving: ``.contiguous()`` changes only memory
    layout, not values, and is a no-op when the tensor is already contiguous.
    """
    import torch.nn.functional as F

    if getattr(F, _BN_PATCH_FLAG, False):
        return False

    _orig_batch_norm = F.batch_norm

    def _batch_norm_mps_contiguous(inp, *args, **kwargs):
        if isinstance(inp, torch.Tensor) and inp.device.type == "mps" and not inp.is_contiguous():
            inp = inp.contiguous()
        return _orig_batch_norm(inp, *args, **kwargs)

    _batch_norm_mps_contiguous._tac_orig = _orig_batch_norm  # type: ignore[attr-defined]
    F.batch_norm = _batch_norm_mps_contiguous
    setattr(F, _BN_PATCH_FLAG, True)
    return True


def patch_scorer_for_mps() -> dict[str, bool]:
    """Apply ALL known MPS device-compat patches needed to run the upstream
    ``DistortionNet`` forward+backward on the Apple GPU. The single consolidation
    point — extend here (not with scattered one-offs) when a new MPS op gap is
    found. Returns a dict of {patch_name: applied_now}.
    """
    return {"batchnorm_contiguous": patch_batchnorm_contiguous_for_mps()}


__all__ = ["patch_batchnorm_contiguous_for_mps", "patch_scorer_for_mps"]
