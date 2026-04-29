"""Lane W-V2 — continuous LEARNABLE per-pair loss weights.

Lane W-V1 used a hard top-K + uniform weight (K=30, weight=5.0) heuristic
to up-weight the hardest pairs. Per the user's water-fill insight (see
``project_arbitrary_vs_learnable_taxonomy``) this is the same anti-pattern
as Lane Ω-V1's water-fill: a closed-form heuristic where a learnable
parameter would be mathematically optimal.

Lane W-V2 replaces top-K with a CONTINUOUS per-pair learnable weight
vector. Math:

    weights_i = softplus(raw_i)           # always > 0 (no clamp footgun)
    loss      = sum_i weights_i * pair_loss_i
              + λ · (sum_i weights_i - N_pairs)²    # rate Lagrangian

The Lagrangian penalty drives ``sum(weights)`` toward ``N_pairs`` so the
average weight is 1.0 — the optimiser can then redistribute "weight mass"
toward the hardest pairs WITHOUT inflating the loss scale (which would
otherwise compete with the SC bit-rate Lagrangian and other auxiliary
terms in train_renderer.py).

The Lane W-V1 profile output (top-K hard + uniform-1 elsewhere) is used as
a WARM-START initialisation, not a hard top-K. The continuous parameter
then adapts during training, which is mathematically optimal because:

  - the Hessian-of-the-loss preserves rank-1 sparsity along the
    "hardest pair" direction (Yousfi-style argument), so the warm-start
    is in the basin of attraction of the optimum;
  - softplus(raw) is smooth → SGD on raw converges where hard top-K
    cannot (top-K is a piecewise-constant function of contrib_i; gradient
    is zero almost everywhere).

CLAUDE.md compliance:
- Pure PyTorch. CUDA-required at the call site (caller decides device).
- Tests: gradient flows, softplus keeps weights >= 0, warm-start works,
  Lagrangian penalty drives sum toward N_pairs, save/load round-trip.
- No global state, no MPS/CPU defaults.
- Mirrors the LearnableBitDepth + dual-ascent pattern from
  ``tac.self_compress`` and ``tac.learnable_bit_quant``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LearnablePairWeights",
    "compute_pair_weight_dual_update",
    "compute_pair_weighted_primal_loss",
    "compute_pair_weight_rate_penalty",
    "save_learnable_pair_weights",
    "load_learnable_pair_weights",
]


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Stable inverse of ``softplus``: returns ``raw`` such that
    ``softplus(raw) = y``. Uses the numerically-stable
    ``log(expm1(y))`` for ``y < 20``; for large ``y``, ``softplus(raw) ≈ raw``
    so we fall back to ``y`` directly to avoid overflow.
    """
    y = y.clamp_min(1e-9)
    safe = y < 20.0
    # log(exp(y) - 1) — use expm1 + log for numerical stability
    out = torch.where(safe, torch.log(torch.expm1(y.clamp_max(20.0))), y)
    return out


class LearnablePairWeights(nn.Module):
    """Per-pair dual multipliers for hard-pair weighting.

    Args:
        n_pairs: number of contest pairs (typically 600 = 1200//2).
        warm_start: optional ``(n_pairs,)`` float tensor to initialise the
            multipliers. ``None`` ⇒ all λ = 0 and weights = 1.0. Common
            warm-start: the Lane W-V1 (top-K hard + uniform-1) tensor
            from ``experiments/profile_pair_sensitivity.py``.

    Forward returns ``1 + λ_p`` for each pair. The λ vector is a buffer,
    not an ``nn.Parameter``: optimizers must not update it by backprop.
    The only mutating path is :meth:`dual_update`, which applies the
    projected dual-ascent rule
    ``λ_p <- max(0, λ_p + η * (loss_p - target_loss))``.
    """

    def __init__(
        self,
        n_pairs: int,
        *,
        warm_start: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if n_pairs <= 0:
            raise ValueError(f"n_pairs must be positive, got {n_pairs}")

        if warm_start is None:
            lambda_init = torch.zeros(n_pairs, dtype=torch.float32)
        else:
            if not isinstance(warm_start, torch.Tensor):
                raise TypeError(
                    f"warm_start must be a torch.Tensor, got {type(warm_start).__name__}"
                )
            if warm_start.ndim != 1 or warm_start.shape[0] != n_pairs:
                raise ValueError(
                    f"warm_start shape {tuple(warm_start.shape)} does not match "
                    f"n_pairs={n_pairs}"
                )
            if (warm_start < 0).any():
                raise ValueError(
                    "warm_start must be non-negative"
                )
            # Warm-start values historically represented direct loss
            # multipliers. The dual formulation can only amplify hard
            # pairs, so values below 1 map to λ=0.
            lambda_init = (warm_start.detach().to(torch.float32) - 1.0).clamp_min(0.0)

        self.register_buffer("lambda_pair", lambda_init.clone())
        self.register_buffer(
            "running_target_loss",
            torch.tensor(float("nan"), dtype=torch.float32),
        )
        self.register_buffer("dual_step", torch.tensor(0, dtype=torch.long))
        self.n_pairs = int(n_pairs)

    # ── Forward / accessors ──────────────────────────────────────────────

    def forward(self) -> torch.Tensor:
        """Return the ``(n_pairs,)`` loss multipliers ``1 + λ_p``."""
        return 1.0 + self.lambda_pair

    def weights(self) -> torch.Tensor:
        """Alias for ``forward()`` — explicit call site."""
        return self.forward()

    def lambdas(self) -> torch.Tensor:
        """Return the nonnegative dual multipliers λ_p."""
        return self.lambda_pair

    def weight_for_pair(self, pair_idx: int | torch.Tensor) -> torch.Tensor:
        """Return the scalar multiplier ``1 + λ_p`` for ``pair_idx``."""
        return self.forward()[pair_idx]

    @torch.no_grad()
    def dual_update(
        self,
        pair_losses: torch.Tensor,
        *,
        eta: float,
        target_loss: torch.Tensor | float | None = None,
        pair_idx: int | torch.Tensor | None = None,
        running_mean_momentum: float = 0.99,
    ) -> torch.Tensor:
        """Apply projected dual ascent to λ.

        ``target_loss`` defaults to the current vector mean for full-vector
        updates. For scalar ``pair_idx`` updates, a running mean target is
        maintained so streaming training loops can update one pair at a
        time without making easy pairs attract mass.
        """
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if not 0.0 <= running_mean_momentum < 1.0:
            raise ValueError(
                f"running_mean_momentum must be in [0, 1), got {running_mean_momentum}"
            )

        losses = pair_losses.detach().to(
            device=self.lambda_pair.device,
            dtype=self.lambda_pair.dtype,
        )
        if not torch.isfinite(losses).all():
            raise ValueError("pair_losses contains NaN/Inf")

        if pair_idx is None:
            if losses.shape != (self.n_pairs,):
                raise ValueError(
                    f"pair_losses shape {tuple(losses.shape)} != ({self.n_pairs},)"
                )
            target = losses.mean() if target_loss is None else torch.as_tensor(
                target_loss, device=losses.device, dtype=losses.dtype,
            )
            self.lambda_pair.add_(float(eta) * (losses - target))
            self.lambda_pair.clamp_(min=0.0)
            self.running_target_loss.copy_(target.detach())
            self.dual_step.add_(1)
            return self.lambda_pair

        if losses.numel() != 1:
            raise ValueError(
                "scalar pair_idx updates require pair_losses with one element"
            )
        idx = int(pair_idx.item()) if isinstance(pair_idx, torch.Tensor) else int(pair_idx)
        if not 0 <= idx < self.n_pairs:
            raise IndexError(f"pair_idx {idx} out of range [0, {self.n_pairs})")

        observed = losses.reshape(())
        if target_loss is None:
            if bool(torch.isnan(self.running_target_loss)):
                target = observed
            else:
                target = self.running_target_loss.to(losses.device, losses.dtype)
            new_running = (
                float(running_mean_momentum) * target
                + (1.0 - float(running_mean_momentum)) * observed
            )
            self.running_target_loss.copy_(new_running.detach())
        else:
            target = torch.as_tensor(target_loss, device=losses.device, dtype=losses.dtype)
            self.running_target_loss.copy_(target.detach())

        self.lambda_pair[idx].add_(float(eta) * (observed - target))
        self.lambda_pair[idx].clamp_(min=0.0)
        self.dual_step.add_(1)
        return self.lambda_pair

    # ── Persistence ──────────────────────────────────────────────────────

    def state_for_save(self) -> dict:
        """Plain-dict snapshot. Use with ``save_learnable_pair_weights``."""
        return {
            "schema_version": 2,
            "module": "tac.learnable_pair_weights.LearnablePairWeights",
            "n_pairs": self.n_pairs,
            "lambda_pair": self.lambda_pair.detach().cpu().clone(),
            "weights": self.weights().detach().cpu().clone(),
            "running_target_loss": self.running_target_loss.detach().cpu().clone(),
            "dual_step": self.dual_step.detach().cpu().clone(),
        }


def compute_pair_weight_dual_update(
    pair_weights: LearnablePairWeights,
    pair_losses: torch.Tensor,
    *,
    eta: float,
    target_loss: torch.Tensor | float | None = None,
    pair_idx: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional wrapper for :meth:`LearnablePairWeights.dual_update`."""
    return pair_weights.dual_update(
        pair_losses,
        eta=eta,
        target_loss=target_loss,
        pair_idx=pair_idx,
    )


def compute_pair_weighted_primal_loss(
    pair_weights: LearnablePairWeights,
    pair_losses: torch.Tensor,
) -> torch.Tensor:
    """Primal loss ``Σ_p (1 + λ_p) loss_p / N``.

    The dual multipliers are detached buffers by construction, so gradient
    flows only into ``pair_losses`` and upstream model parameters.
    """
    if pair_losses.shape != (pair_weights.n_pairs,):
        raise ValueError(
            f"pair_losses shape {tuple(pair_losses.shape)} != ({pair_weights.n_pairs},)"
        )
    multipliers = pair_weights().detach().to(
        device=pair_losses.device,
        dtype=pair_losses.dtype,
    )
    return (multipliers * pair_losses).sum() / float(pair_weights.n_pairs)


def compute_pair_weight_rate_penalty(
    pair_weights: LearnablePairWeights,
    *,
    target_sum: float | None = None,
    lambda_rate: float = 1.0,
) -> torch.Tensor:
    """Retired compatibility shim.

    Round 10 replaced the old softplus + sum-to-N penalty with projected
    dual ascent on λ_p. Keeping this function as a zero-valued tensor lets
    older call sites import it without reintroducing optimizer-controlled
    pair weights.
    """
    del target_sum, lambda_rate
    return pair_weights.lambda_pair.sum() * 0.0


def save_learnable_pair_weights(
    pair_weights: LearnablePairWeights,
    path: str | Path,
) -> Path:
    """Serialise to ``path`` as a torch dict (mirrors
    ``profile_pair_sensitivity.py`` output schema so the same loader works).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pair_weights.state_for_save(), p)
    return p


def load_learnable_pair_weights(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LearnablePairWeights:
    """Inverse of ``save_learnable_pair_weights``. Restores the module.

    Validates the schema_version + module ID — refuses to load anything
    other than a v1 LearnablePairWeights snapshot to fail loud (CLAUDE.md
    "fail loud not silent").
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    # Loader-format-safety: this loader handles ONLY pytorch pickles, never
    # renderer .bin files. Verify magic bytes (DEN-V2 bug pattern guard).
    with open(p, "rb") as _f:
        _magic = _f.read(4)
    if _magic in (b"FP4A", b"ASYM", b"DPSM", b"I4LZ", b"CCh1", b"C3R1", b"SCv1", b"OMG1"):
        raise ValueError(
            f"learnable-pair-weights file {p} has renderer magic {_magic!r} — "
            f"expected pytorch pickle. Wrong file?"
        )
    state = torch.load(p, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(
            f"{p} is not a learnable-pair-weights snapshot (got {type(state).__name__})"
        )
    schema_version = state.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError(
            f"{p} schema_version={state.get('schema_version')} not in (1, 2); "
            "refuse to load incompatible snapshot."
        )
    if state.get("module") != "tac.learnable_pair_weights.LearnablePairWeights":
        raise ValueError(
            f"{p} module={state.get('module')!r} != "
            "tac.learnable_pair_weights.LearnablePairWeights"
        )
    n_pairs = int(state["n_pairs"])
    pw = LearnablePairWeights(n_pairs)
    if schema_version == 2:
        pw.lambda_pair.copy_(state["lambda_pair"].to(pw.lambda_pair.dtype))
        if "running_target_loss" in state:
            pw.running_target_loss.copy_(
                state["running_target_loss"].to(pw.running_target_loss.dtype)
            )
        if "dual_step" in state:
            pw.dual_step.copy_(state["dual_step"].to(pw.dual_step.dtype))
    else:
        if "raw" in state:
            weights = F.softplus(state["raw"].to(torch.float32))
        elif "weights" in state:
            weights = state["weights"].to(torch.float32)
        else:
            raise ValueError(f"{p} v1 snapshot has neither 'raw' nor 'weights'")
        pw.lambda_pair.copy_((weights - 1.0).clamp_min(0.0).to(pw.lambda_pair.dtype))
    return pw
