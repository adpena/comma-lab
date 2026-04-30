"""Lane 20 β-variant — sensitivity-conditioned Ballé hyperprior σ prediction.

Paradigm β extension of :mod:`tac.balle_hyperprior_codec` (full-Ballé mode).
Vanilla Ballé predicts per-block σ from a hyper-latent ``z`` via a small
hyper-decoder MLP. The β-variant predicts per-block σ from
``concat(z, sensitivity_block_aggregate)`` so the rate-prediction network
can tighten σ on high-sensitivity blocks (where small reconstruction errors
matter to the score) and relax σ on score-blind blocks.

Math foundation
---------------
The bits-per-block under a Gaussian prior are:

    bits(y_b | σ_b) ≈ −log2 N(y_b ; 0, σ_b²)

The σ-prediction MLP is trained to minimize sum_b bits(y_b | σ_b). The
β-variant feeds ``[z_b, mean(sensitivity_block_b)]`` instead of just z_b.
At training time, this lets the predictor learn a CONDITIONAL relationship:
"high-sensitivity blocks tend to have smaller |y|, predict smaller σ."

Two regimes:
1. ``mode="concat"``: append the per-block sensitivity scalar to z.
   hyper_decoder input dim = z_dim + 1.
2. ``mode="multiplicative"``: use sensitivity as a residual scaling factor
   on the predicted σ:
       σ_β = σ_baseline / (1 + α * sens_normalized)
   where α is a learned scalar. This yields tighter σ on
   high-sensitivity blocks without changing the hyper_decoder
   architecture.

Wire format
-----------
Identical to vanilla ``BHv1`` mode=1 (full Ballé) — the
sensitivity-conditioned pieces (extra concat input or α scalar) are
implementation details of the σ predictor, not the wire format. The
encoder must commit to ONE mode per archive (recorded in metadata) and
the decoder reproduces σ from the same input + the same hyper-decoder
weights.

In ``mode="multiplicative"``, the per-block sensitivity must be embedded
as side-info in the archive (additional ``num_blocks * 4`` bytes float32)
so the decoder can reconstruct σ_β. In ``mode="concat"``, the per-block
sensitivity is part of the hyper-encoder input, so only z (already in
side-info) needs to be in the archive.

CLAUDE.md compliance
--------------------
* Pure-math byte-level encode/decode at inflate time — no scorer load.
* Sensitivity is computed ONCE at compress time and either (a) baked into
  the trained hyper-decoder weights via concat training (no extra archive
  bytes) or (b) shipped as side-info (extra bytes accounted for).
* No silent defaults — all kwargs are required.

References
----------
* :mod:`tac.balle_hyperprior_codec` — vanilla full-Ballé codec.
* :mod:`tac.sensitivity_map` — sensitivity-vector contract.
* ``.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md``
  §"Paradigm Shift β" — math foundation.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class SensitivityWeightedBalleError(ValueError):
    """Raised when sensitivity-conditioned Ballé inputs are malformed."""


def _validate_block_sensitivity(
    sensitivity: torch.Tensor,
    *,
    num_blocks: int,
    name: str,
) -> torch.Tensor:
    """Validate per-block sensitivity (1-D, num_blocks long)."""
    if not torch.is_tensor(sensitivity):
        raise SensitivityWeightedBalleError(
            f"{name}: sensitivity must be a torch.Tensor, got "
            f"{type(sensitivity).__name__}"
        )
    if sensitivity.dim() != 1 or int(sensitivity.shape[0]) != int(num_blocks):
        raise SensitivityWeightedBalleError(
            f"{name}: sensitivity shape {tuple(sensitivity.shape)} does not "
            f"match expected ({num_blocks},)"
        )
    s = sensitivity.detach().to(torch.float32)
    if not torch.isfinite(s).all():
        n_bad = int((~torch.isfinite(s)).sum().item())
        raise SensitivityWeightedBalleError(
            f"{name}: sensitivity contains {n_bad} NaN/Inf value(s)"
        )
    if (s < 0).any():
        raise SensitivityWeightedBalleError(
            f"{name}: sensitivity must be non-negative"
        )
    return s


def aggregate_pixel_sensitivity_to_blocks(
    *,
    pixel_sensitivity: torch.Tensor,
    block_size: int,
    aggregate: Literal["mean", "max", "sum"] = "mean",
) -> torch.Tensor:
    """Aggregate per-pixel sensitivity into per-block sensitivity.

    The y stream is partitioned into contiguous ``block_size`` chunks; this
    helper produces one scalar per block. Used to build the per-block
    sensitivity tensor that the σ predictor consumes.

    Args:
        pixel_sensitivity: 1-D non-negative tensor (length = total y count).
        block_size: positive integer chunk size.
        aggregate: "mean" / "max" / "sum".

    Returns:
        1-D tensor of length ``ceil(num_pixels / block_size)``.

    Raises:
        SensitivityWeightedBalleError on bad inputs.
    """
    if not torch.is_tensor(pixel_sensitivity):
        raise SensitivityWeightedBalleError(
            "aggregate_pixel_sensitivity_to_blocks: input must be a tensor"
        )
    if pixel_sensitivity.dim() != 1:
        raise SensitivityWeightedBalleError(
            f"aggregate_pixel_sensitivity_to_blocks: input must be 1-D; "
            f"got shape {tuple(pixel_sensitivity.shape)}"
        )
    if not isinstance(block_size, int) or block_size <= 0:
        raise SensitivityWeightedBalleError(
            f"block_size must be a positive int; got {block_size!r}"
        )
    if aggregate not in {"mean", "max", "sum"}:
        raise SensitivityWeightedBalleError(
            f"aggregate must be mean/max/sum; got {aggregate!r}"
        )
    s = pixel_sensitivity.detach().to(torch.float32)
    if not torch.isfinite(s).all():
        raise SensitivityWeightedBalleError(
            "aggregate_pixel_sensitivity_to_blocks: input contains NaN/Inf"
        )
    if (s < 0).any():
        raise SensitivityWeightedBalleError(
            "aggregate_pixel_sensitivity_to_blocks: input must be non-negative"
        )
    n = s.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    pad_len = n_blocks * block_size - n
    if pad_len:
        s = torch.cat([s, torch.zeros(pad_len, dtype=s.dtype)], dim=0)
    blocks = s.reshape(n_blocks, block_size)
    if aggregate == "mean":
        return blocks.mean(dim=1)
    if aggregate == "max":
        return blocks.amax(dim=1)
    return blocks.sum(dim=1)


class SensitivityConditionedHyperDecoder(nn.Module):
    """Hyper-decoder that consumes ``[z, per-block-sensitivity]``.

    Uses ``mode="concat"`` semantics: per-block sensitivity is concatenated
    onto z before the MLP. The decoder MUST receive the same per-block
    sensitivity at inflate time — either baked into z (encoder responsibility)
    or read from archive side-info.

    For the multiplicative scaling regime, see
    :func:`apply_multiplicative_sensitivity_to_sigma`.

    Args:
        z_dim: hyper-latent dimension. Required keyword.
        hidden_dim: MLP hidden layer width. Required keyword.
        sigma_min: lower bound for predicted σ. Required keyword.
        sigma_max: upper bound for predicted σ. Required keyword.
    """

    def __init__(
        self,
        *,
        z_dim: int,
        hidden_dim: int,
        sigma_min: float,
        sigma_max: float,
    ) -> None:
        super().__init__()
        if z_dim <= 0 or hidden_dim <= 0:
            raise SensitivityWeightedBalleError(
                f"z_dim and hidden_dim must be > 0; got z_dim={z_dim}, "
                f"hidden_dim={hidden_dim}"
            )
        if not (0.0 < sigma_min < sigma_max):
            raise SensitivityWeightedBalleError(
                f"require 0 < sigma_min < sigma_max; got sigma_min="
                f"{sigma_min}, sigma_max={sigma_max}"
            )
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        # Input dim = z_dim + 1 (sensitivity scalar)
        self.fc1 = nn.Linear(z_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        per_block_sensitivity: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-block σ.

        Args:
            z: ``(num_blocks, z_dim)`` hyper-latent.
            per_block_sensitivity: ``(num_blocks,)`` non-negative scalar.

        Returns:
            ``(num_blocks,)`` σ tensor in ``[sigma_min, sigma_max]``.
        """
        if z.dim() != 2 or int(z.shape[1]) != self.z_dim:
            raise SensitivityWeightedBalleError(
                f"z shape {tuple(z.shape)} does not match (B, {self.z_dim})"
            )
        sens = _validate_block_sensitivity(
            per_block_sensitivity,
            num_blocks=int(z.shape[0]),
            name="per_block_sensitivity",
        ).to(z.device).unsqueeze(1)
        x = torch.cat([z, sens], dim=1)
        h = torch.relu(self.fc1(x))
        sigma_pre = self.fc2(h).squeeze(1)
        # Soft-bound via sigmoid + linear map.
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(
            sigma_pre
        )
        return sigma


def apply_multiplicative_sensitivity_to_sigma(
    *,
    sigma_baseline: torch.Tensor,
    per_block_sensitivity: torch.Tensor,
    alpha: float,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Apply ``σ_β = σ_baseline / (1 + α * sens_norm)`` block-wise.

    The multiplicative regime tightens σ on high-sensitivity blocks
    without changing the hyper-decoder architecture. ``sens_norm`` is
    max-normalized so the worst sensitivity block has factor
    ``1 / (1 + α)``.

    Args:
        sigma_baseline: ``(num_blocks,)`` predicted σ from a vanilla
            hyper-decoder. Required keyword.
        per_block_sensitivity: ``(num_blocks,)`` non-negative tensor.
            Required keyword.
        alpha: positive scalar; larger α tightens σ more on high-sensitivity
            blocks. Required keyword.
        sigma_min: lower bound on σ_β. Required keyword.
        sigma_max: upper bound on σ_β. Required keyword.

    Returns:
        ``(num_blocks,)`` σ_β tensor clamped to ``[sigma_min, sigma_max]``.

    Raises:
        SensitivityWeightedBalleError on bad inputs.
    """
    if not torch.is_tensor(sigma_baseline):
        raise SensitivityWeightedBalleError(
            "sigma_baseline must be a torch.Tensor"
        )
    if sigma_baseline.dim() != 1:
        raise SensitivityWeightedBalleError(
            f"sigma_baseline must be 1-D; got shape {tuple(sigma_baseline.shape)}"
        )
    n_blocks = int(sigma_baseline.shape[0])
    sens = _validate_block_sensitivity(
        per_block_sensitivity,
        num_blocks=n_blocks,
        name="per_block_sensitivity",
    ).to(sigma_baseline.device)
    if alpha <= 0.0:
        raise SensitivityWeightedBalleError(
            f"alpha must be > 0; got {alpha}"
        )
    if not (0.0 < sigma_min < sigma_max):
        raise SensitivityWeightedBalleError(
            f"require 0 < sigma_min < sigma_max; got sigma_min={sigma_min}, "
            f"sigma_max={sigma_max}"
        )
    denom_max = sens.max().clamp_min(1e-12)
    sens_norm = sens / denom_max
    sigma_beta = sigma_baseline / (1.0 + float(alpha) * sens_norm)
    return sigma_beta.clamp(min=sigma_min, max=sigma_max)


__all__ = [
    "SensitivityWeightedBalleError",
    "aggregate_pixel_sensitivity_to_blocks",
    "SensitivityConditionedHyperDecoder",
    "apply_multiplicative_sensitivity_to_sigma",
]
