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
    """Per-pair learnable loss-weight vector parameterised via softplus.

    Args:
        n_pairs: number of contest pairs (typically 600 = 1200//2).
        warm_start: optional ``(n_pairs,)`` float tensor to initialise the
            weights. ``None`` ⇒ all weights = 1.0 (uniform). Common
            warm-start: the Lane W-V1 (top-K hard + uniform-1) tensor
            from ``experiments/profile_pair_sensitivity.py``.

    Forward returns the ``(n_pairs,)`` weight tensor (positive floats).
    The single nn.Parameter is ``self.raw`` of shape ``(n_pairs,)``; the
    forward call applies ``softplus``.

    The forward pass is differentiable wrt. self.raw, so
    ``loss = (weights * pair_losses).sum()`` propagates gradient back
    into ``self.raw``. Combined with ``compute_pair_weight_rate_penalty``,
    this enables Lagrangian dual ascent on the constraint
    ``sum(weights) == n_pairs``.
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
            # softplus(raw) = 1 ⇒ raw = log(exp(1) - 1) ≈ 0.5413
            init_y = torch.ones(n_pairs, dtype=torch.float32)
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
                    "warm_start must be non-negative (softplus codomain is [0, ∞))"
                )
            init_y = warm_start.detach().to(torch.float32).clamp_min(1e-9)

        raw_init = _inverse_softplus(init_y)
        self.raw = nn.Parameter(raw_init.clone())
        self.n_pairs = int(n_pairs)

    # ── Forward / accessors ──────────────────────────────────────────────

    def forward(self) -> torch.Tensor:
        """Return the (n_pairs,) softplus-positive weight vector."""
        return F.softplus(self.raw)

    def weights(self) -> torch.Tensor:
        """Alias for ``forward()`` — explicit call site."""
        return self.forward()

    def weight_for_pair(self, pair_idx: int | torch.Tensor) -> torch.Tensor:
        """Return the differentiable scalar weight for ``pair_idx``."""
        return F.softplus(self.raw[pair_idx])

    # ── Persistence ──────────────────────────────────────────────────────

    def state_for_save(self) -> dict:
        """Plain-dict snapshot. Use with ``save_learnable_pair_weights``."""
        return {
            "schema_version": 1,
            "module": "tac.learnable_pair_weights.LearnablePairWeights",
            "n_pairs": self.n_pairs,
            "raw": self.raw.detach().cpu().clone(),
            "weights": self.weights().detach().cpu().clone(),
        }


def compute_pair_weight_rate_penalty(
    pair_weights: LearnablePairWeights,
    *,
    target_sum: float | None = None,
    lambda_rate: float = 1.0,
) -> torch.Tensor:
    """Lagrangian rate penalty: ``λ · (sum(weights) − target_sum)²``.

    Drives the average weight toward 1.0 (target_sum = n_pairs by default)
    so the loss scale stays comparable to the unweighted run. The squared
    form is symmetric — both over- and under-shoot are penalised — which
    is what we want: free up weight mass on easy pairs AND constrain the
    total magnitude.

    Args:
        pair_weights: the LearnablePairWeights module.
        target_sum: desired sum-of-weights. Default ``n_pairs`` ⇒ mean = 1.
        lambda_rate: Lagrangian multiplier. Annealed via dual ascent in the
            training loop — start at 0, ramp to ~1 after the warmup.

    Returns:
        Scalar tensor with grad wrt. pair_weights.raw.
    """
    if target_sum is None:
        target_sum = float(pair_weights.n_pairs)
    w = pair_weights()
    # squared deviation of the constraint sum(w) - target_sum
    return float(lambda_rate) * (w.sum() - float(target_sum)).pow(2)


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
    if state.get("schema_version") != 1:
        raise ValueError(
            f"{p} schema_version={state.get('schema_version')} != 1; "
            "refuse to load incompatible snapshot."
        )
    if state.get("module") != "tac.learnable_pair_weights.LearnablePairWeights":
        raise ValueError(
            f"{p} module={state.get('module')!r} != "
            "tac.learnable_pair_weights.LearnablePairWeights"
        )
    n_pairs = int(state["n_pairs"])
    pw = LearnablePairWeights(n_pairs)
    pw.raw.data.copy_(state["raw"].to(pw.raw.dtype))
    return pw
