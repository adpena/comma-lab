# SPDX-License-Identifier: MIT
"""ATW V2 cooperative-receiver V2 — PyTorch parity reference for Axis 2 drift verification.

Per Phase 3 design memo §8 (MLX drift minimization per primitive) + operator
binding directive #3 ("MLX drift minimization ... we also need adversarial
review against all landing recursive for ... MLX drift minimization").

This module provides a PyTorch implementation of the canonical forward path
for use in MLX↔PyTorch drift verification tests (test_basic.py). The PyTorch
implementation MUST be byte-stable wrt the canonical layer semantics so the
drift measurement isolates MLX-vs-PyTorch numerical differences (not
algorithmic mismatches).

NON-PRODUCTION
==============

This module is for PARITY VERIFICATION ONLY. The substrate's canonical
production paths are:
- MLX renderer: ``mlx_renderer.py`` (Apple Silicon local iteration)
- numpy reference: ``numpy_reference.py`` (portable CPU; bit-exact)
- inflate runtime: ``inflate.py`` (production inflate)

This PyTorch reference is used ONLY in test_basic.py to compare MLX
primitive outputs against PyTorch primitive outputs under identical
weights, to surface drift bounds per primitive.

Cross-references
----------------

* Phase 3 design memo §8 (MLX drift minimization per primitive)
* ``codex_findings_mlx_drift_determinism_online_research_20260522T050151Z_codex.md``
* ``pr95_mlx_pytorch_drift_mitigation_engineering_landed_20260525.md``
* ``pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md``
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover — exercised on test environments with torch installed
    import torch  # type: ignore[import-untyped]
    import torch.nn as nn  # type: ignore[import-untyped]
    import torch.nn.functional as F  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


def is_torch_available() -> bool:
    return _TORCH_IMPORT_ERROR is None and torch is not None


def require_torch_available() -> None:
    if not is_torch_available():
        raise RuntimeError(
            f"PyTorch is not available. Required for _torch_compat_reference. "
            f"Import error: {_TORCH_IMPORT_ERROR!r}. Per Axis 3 portability: "
            f"numpy_reference is the portable fallback (no PyTorch dependency)."
        )


def torch_ego_motion_foe_projection(pose_delta: "torch.Tensor") -> "torch.Tensor":
    """PyTorch reference for ego-motion FOE projection.

    Sister to ``numpy_reference.numpy_ego_motion_foe_projection`` +
    ``mlx_renderer.mlx_ego_motion_foe_projection``. Used in test_basic.py
    for MLX↔PyTorch drift measurement.
    """
    require_torch_available()
    if pose_delta.dim() != 2 or pose_delta.shape[1] != 6:
        raise ValueError(f"pose_delta must be (B, 6); got {tuple(pose_delta.shape)}")
    translation = pose_delta[:, :3]
    rotation = pose_delta[:, 3:]
    eps = 1e-8
    t_norm = torch.norm(translation, dim=1, keepdim=True) + eps
    r_norm = torch.norm(rotation, dim=1, keepdim=True) + eps
    return torch.cat(
        [translation / t_norm, rotation / r_norm],
        dim=1,
    )


def torch_linear(
    x: "torch.Tensor",
    weight: np.ndarray,
    bias: np.ndarray,
) -> "torch.Tensor":
    """PyTorch reference for Linear layer with explicit FP32 weights from numpy."""
    require_torch_available()
    w = torch.from_numpy(weight)
    b = torch.from_numpy(bias)
    return F.linear(x, w, b)


def torch_relu(x: "torch.Tensor") -> "torch.Tensor":
    """PyTorch reference for ReLU."""
    require_torch_available()
    return F.relu(x)


def torch_sigmoid(x: "torch.Tensor") -> "torch.Tensor":
    """PyTorch reference for sigmoid."""
    require_torch_available()
    return torch.sigmoid(x)


def torch_cond_embedding_forward(
    ego_motion_proj: "torch.Tensor",
    cond_embed_weight_1: np.ndarray,
    cond_embed_bias_1: np.ndarray,
    cond_embed_weight_2: np.ndarray,
    cond_embed_bias_2: np.ndarray,
) -> "torch.Tensor":
    """End-to-end PyTorch reference for cond_embed forward pass.

    Sister to numpy_reference.numpy_decode_pair_with_ego_motion_conditioning's
    conditioning embedding portion + mlx_renderer._CondEmbeddingHead forward.
    """
    require_torch_available()
    h = torch_linear(ego_motion_proj, cond_embed_weight_1, cond_embed_bias_1)
    h = torch_relu(h)
    return torch_linear(h, cond_embed_weight_2, cond_embed_bias_2)


__all__ = [
    "is_torch_available",
    "require_torch_available",
    "torch_cond_embedding_forward",
    "torch_ego_motion_foe_projection",
    "torch_linear",
    "torch_relu",
    "torch_sigmoid",
]
