# SPDX-License-Identifier: MIT
"""Muon optimizer - Newton-Schulz orthogonalized momentum (Keller Jordan, 2024).

Mathematical formulation
------------------------

For 2-D parameter ``W`` with momentum buffer ``b``, the update step is:

    g_t  = grad L(W_t)
    b_t  = momentum * b_{t-1} + g_t        (Nesterov: g_hat = g + momentum * b)
    W_t  <- W_t - lr * scale * NewtonSchulz5(g_hat)

The Newton-Schulz quintic iteration (5 steps default) computes an approximate
matrix sign / orthogonalization in bfloat16. Empirically Muon converges 2-4x
faster than AdamW for hidden-layer 2-D weights on transformer-shaped problems.

Coefficients ``(a, b, c) = (3.4445, -4.7750, 2.0315)`` are Keller Jordan's
tuned values for 5-step convergence to ~1e-3 orthogonality on typical
weight-spectrum inputs.

Decoupled weight decay
----------------------

Following Chen-Li-Liu (arXiv:2506.15054), Muon's spectral-norm KKT story
requires WD to be active. We apply WD before the orthogonalized update,
mirroring AdamW's decoupled convention.

Per-shape handling
------------------

- ``ndim == 4`` (Conv2d weights): flattened to 2-D ``(out_ch, in_ch * kH * kW)``
  for NS, then unflattened
- ``ndim == 2`` (Linear weights): NS applied directly
- ``ndim == 1`` (biases, gains): skipped - uses momentum SGD without NS

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------

This optimizer produces no contest score by itself. Any empirical score claim
requires the canonical dispatch path (CUDA + Linux x86_64 CPU paired auth eval
on 1:1 contest-CI hardware per CLAUDE.md "Submission auth eval - BOTH CPU AND
CUDA").

Cross-references
----------------

- Reference implementation: https://github.com/KellerJordan/Muon
- PR95 hnerv_muon source: ``experiments/results/public_pr_archive_release_view/
  public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/optim.py``
- Keller Jordan research memo: commit ``d64b17cf``
- Sister optimizers: ``tac.optimization.iglt.InfoGeomLangevinOptimizer`` (Fisher
  preconditioning) and ``tac.optimization.langevin_optimizer.LangevinOptimizer``
- Weight decay rationale: Chen-Li-Liu arXiv:2506.15054

Lane: ``lane_other_priorities_parallel_sweep_20260513``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = [
    "MuonOptimizer",
    "partition_params_for_muon",
    "zeropower_via_newtonschulz5",
]


# Keller Jordan's tuned 5-step NS coefficients (bfloat16 stability).
_NS_COEFFS_A: float = 3.4445
_NS_COEFFS_B: float = -4.7750
_NS_COEFFS_C: float = 2.0315


@torch.no_grad()
def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz iteration approximating the matrix sign / orthogonalization.

    Given gradient ``G`` (ndim >= 2), returns an approximately orthogonal matrix of
    the same shape such that ``||output||_2 ~= 1`` along the leading singular
    direction. Operates in bfloat16 for numerical stability of the quintic
    iteration; casts back to the input dtype at exit.

    Parameters
    ----------
    G : torch.Tensor
        Gradient tensor, ``ndim >= 2``.
    steps : int
        Number of NS iterations (default 5). Higher means tighter orthogonality at
        ~linear-in-steps wall-clock cost.
    eps : float
        Small constant for numerical stability of the initial normalization.

    Returns
    -------
    torch.Tensor
        Orthogonalized tensor of same shape and dtype as ``G``.
    """
    if G.ndim < 2:
        raise ValueError(
            f"zeropower_via_newtonschulz5 requires ndim >= 2, got ndim={G.ndim}"
        )
    a, b, c = _NS_COEFFS_A, _NS_COEFFS_B, _NS_COEFFS_C
    X = G.to(torch.bfloat16) if G.dtype == torch.float32 else G.clone()
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B_ = b * A + c * A @ A
        X = a * X + B_ @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class MuonOptimizer(torch.optim.Optimizer):
    """Muon optimizer (Keller Jordan, 2024).

    Parameters
    ----------
    params : Iterable[torch.nn.Parameter]
        Parameters to optimize. Use :func:`partition_params_for_muon` to split
        a model's params into the Muon-eligible vs AdamW-handled groups.
    lr : float
        Learning rate (default 0.02; higher than AdamW because the NS step
        normalizes the update magnitude).
    momentum : float
        Momentum coefficient (default 0.95).
    nesterov : bool
        Use Nesterov momentum (default ``True``).
    ns_steps : int
        Newton-Schulz iteration count per step (default 5).
    weight_decay : float
        Decoupled weight-decay coefficient applied to the parameter directly
        before the orthogonalized update (default 0.0; recommended ~0.01 for
        Muon-eligible weights per Chen-Li-Liu arXiv:2506.15054).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # Decoupled weight decay: shrink the parameter before the
                # orthogonalized update (AdamW convention).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                gu = g.add(buf, alpha=momentum) if nesterov else buf

                orig_shape = gu.shape
                if gu.ndim == 4:
                    g2d = gu.view(gu.size(0), -1)
                    g_ortho = zeropower_via_newtonschulz5(g2d, steps=ns_steps)
                    scale = max(1.0, (g2d.size(0) / g2d.size(1)) ** 0.5)
                    g_final = (g_ortho * scale).view(orig_shape)
                elif gu.ndim == 2:
                    g_ortho = zeropower_via_newtonschulz5(gu, steps=ns_steps)
                    scale = max(1.0, (gu.size(0) / gu.size(1)) ** 0.5)
                    g_final = g_ortho * scale
                else:
                    # 1-D (biases, gains): plain SGD-with-momentum, no NS.
                    g_final = gu

                p.add_(g_final, alpha=-lr)
        return loss


def partition_params_for_muon(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split ``model.named_parameters()`` into ``(muon_params, adamw_params)``.

    Muon-eligible: 2-D+ weights NOT in the stem and NOT in RGB heads.

    AdamW-handled:
      - All biases (``ndim == 1``)
      - All 1-D params (gains, embeddings without batch dim)
      - Stem ``Linear`` weights (name contains ``"stem"``)
      - RGB head weights (name starts with ``"rgb"`` or contains ``".rgb_"``)

    The split mirrors PR95 hnerv_muon convention. Future substrates with
    different head naming should override via explicit name filtering.

    Parameters
    ----------
    model : torch.nn.Module
        Module whose ``named_parameters()`` will be inspected.

    Returns
    -------
    tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]
        ``(muon_params, adamw_params)``. Both lists contain only params with
        ``requires_grad=True``.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            adamw_params.append(p)
            continue
        low = name.lower()
        if "stem" in low or low.startswith("rgb") or ".rgb_" in low:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    return muon_params, adamw_params
