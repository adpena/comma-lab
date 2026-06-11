# SPDX-License-Identifier: MIT
"""Per-stage weight mechanisms for the PR95 curriculum (QAT + sigma-noise + C1a).

These are the three weight-domain mechanisms the PR95 8-stage curriculum turns on
in its later stages (L14 stage-4 QAT, L16 C1a coder-aware entropy, L17 Gaussian
sigma weight-noise). They operate on the decoder's Conv2d/Linear WEIGHT tensors,
so they live here (shared by both the PR95-reference and capstone trainers) rather
than in the inner loop of either trainer:

- :func:`apply_stage_weight_transforms` — applied INSIDE the traced ``mx.vjp``
  forward: fake-quantize (STE) and/or add Gaussian noise to the weight tensors
  BEFORE the render, so the gradient flows back to the live (un-quantized,
  un-noised) weight. This is the exact PR95 ``apply_qat``/``restore_qat`` STE
  pattern (forward sees the transformed weight; backward passes through) plus the
  L17 weight-noise injection.

- :func:`add_c1a_entropy_gradient` — the C1a entropy term is a function of the
  weights ALONE (not the pixels), so its gradient is computed separately via
  ``mx.grad`` over the weight tree and ADDED to the pixel-derived gradient tree
  before the optimizer step. This matches torch ``loss = ... + cat_lambda*ent``
  where ``ent`` is differentiated together with the scorer loss (gradients add).

The weight-tensor selection (:func:`weight_tensor_keys`) mirrors torch's
``isinstance(mod, (nn.Conv2d, nn.Linear))`` filter: the ``*.weight`` entries of
the trainable tree with ndim>=2 (biases are 1D; latents/codebook are excluded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tac.mlx_pr95_port.mlx_losses import (
    apply_sigma_noise_mlx,
    cat_entropy_v2_mlx,
    fake_quantize_mlx,
)

try:  # pragma: no cover - import guard
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = tree_unflatten = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("tac.mlx_pr95_port.curriculum_mechanisms requires mlx.core.")


@dataclass(frozen=True)
class StageMechanisms:
    """The active weight-domain mechanisms for one curriculum stage."""

    use_qat: bool = False
    sigma_weight_noise: float = 0.0
    cat_lambda: float = 0.0
    cat_sigma: float = 0.2

    @property
    def any_weight_transform(self) -> bool:
        """True if the traced forward must transform weights (QAT or sigma-noise)."""
        return self.use_qat or self.sigma_weight_noise > 0.0

    @property
    def c1a_active(self) -> bool:
        """True if the C1a entropy gradient must be added to the grad tree."""
        return self.cat_lambda > 0.0


def weight_tensor_keys(param_tree: Any) -> list[str]:
    """Return the flat-tree keys of the Conv2d/Linear WEIGHT tensors.

    Mirrors torch ``isinstance(mod, (nn.Conv2d, nn.Linear))`` + ``mod.weight``:
    the ``*.weight`` entries with ndim>=2. Biases (1D), latents, and the VQ
    codebook are excluded. Sorted for determinism.
    """
    _require_mlx()
    keys: list[str] = []
    for k, v in tree_flatten(param_tree):
        if k.endswith(".weight") and getattr(v, "ndim", 0) >= 2:
            keys.append(k)
    return sorted(keys)


def apply_stage_weight_transforms(
    param_arrays_by_name: dict[str, Any],
    weight_keys: list[str],
    mech: StageMechanisms,
    *,
    noise_key: Any = None,
) -> dict[str, Any]:
    """Return a NEW name->array map with QAT + sigma-noise applied to weight keys.

    Called INSIDE the traced ``mx.vjp`` forward on the (traced) parameter arrays,
    so the transform is part of the autodiff graph and the gradient flows back to
    the live weight (STE for QAT; noise is a constant w.r.t. the weight). Non-weight
    arrays (biases, latents, codebook) pass through unchanged.

    Args:
        param_arrays_by_name: name -> (traced) MLX array.
        weight_keys: the Conv2d/Linear ``*.weight`` keys (from :func:`weight_tensor_keys`).
        mech: the stage's active mechanisms.
        noise_key: optional MLX random key for the sigma-noise (per-weight split).
    """
    _require_mlx()
    if not mech.any_weight_transform:
        return param_arrays_by_name
    wk = set(weight_keys)
    out: dict[str, Any] = {}
    key = noise_key
    for name, arr in param_arrays_by_name.items():
        if name in wk:
            w = arr
            if mech.use_qat:
                w = fake_quantize_mlx(w)
            if mech.sigma_weight_noise > 0.0:
                if key is not None:
                    key, sub = mx.random.split(key)
                    w = apply_sigma_noise_mlx(w, mech.sigma_weight_noise, rng_key=sub)
                else:
                    w = apply_sigma_noise_mlx(w, mech.sigma_weight_noise)
            out[name] = w
        else:
            out[name] = arr
    return out


def add_c1a_entropy_gradient(
    grads: Any,
    param_tree: Any,
    weight_keys: list[str],
    mech: StageMechanisms,
    *,
    rng_key: Any = None,
) -> Any:
    """Add ``cat_lambda * d(cat_entropy_v2)/dw`` to the gradient tree (in weight keys).

    The C1a entropy is a scalar function of the WEIGHTS alone, so its gradient is
    computed via ``mx.grad`` over just the weight arrays and ADDED to the incoming
    (pixel-derived) gradient tree — exactly as torch adds ``cat_lambda*ent`` to the
    loss before a single ``backward()`` (gradients are linear; the sum-of-losses
    gradient equals the sum of the per-loss gradients).

    Args:
        grads: the pixel-derived gradient tree (nested dict) for the bundle.
        param_tree: the bundle's trainable parameter tree (nested dict).
        weight_keys: the Conv2d/Linear ``*.weight`` keys.
        mech: the stage's active mechanisms (``cat_lambda``, ``cat_sigma``).
        rng_key: optional MLX random key for the C1a soft-histogram subsample.

    Returns:
        The gradient tree with the C1a term added to the weight keys (a new tree).
    """
    _require_mlx()
    if not mech.c1a_active:
        return grads
    flat_params = dict(tree_flatten(param_tree))
    ordered_keys = [k for k in weight_keys if k in flat_params]
    weight_arrays = [flat_params[k] for k in ordered_keys]

    def c1a_scalar(arrays: list[Any]) -> Any:
        return mech.cat_lambda * cat_entropy_v2_mlx(
            arrays, sigma=mech.cat_sigma, rng_key=rng_key
        )

    c1a_grads = mx.grad(c1a_scalar)(weight_arrays)
    grads_flat = dict(tree_flatten(grads))
    for k, g in zip(ordered_keys, c1a_grads, strict=True):
        if k in grads_flat and grads_flat[k] is not None:
            grads_flat[k] = grads_flat[k] + g
        else:
            grads_flat[k] = g
    return tree_unflatten(list(grads_flat.items()))


__all__ = [
    "StageMechanisms",
    "add_c1a_entropy_gradient",
    "apply_stage_weight_transforms",
    "weight_tensor_keys",
]
