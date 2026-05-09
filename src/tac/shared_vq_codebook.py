"""T17 — Shared VQ-VAE codebook (renderer + quantizer + auxiliary scorer).

Council provenance
------------------
This module operationalizes Track T17 from the Fields-Medal Grand Council
Eureka session (memory ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md``,
§Selfcomp eureka).

Selfcomp's lateral leap: instead of three independent latent spaces
(renderer's per-pixel emission, quantizer's per-tensor codes, auxiliary
scorer's per-pixel features), use a SINGLE shared 256-entry VQ-VAE
codebook with van den Oord persistent EMA updates. This unifies what
Hinton 2014 (knowledge distillation) + van den Oord 2017 (VQ-VAE) +
Selfcomp's block-FP achieve separately:

  - **Reduces archive bytes**: one shared 256-entry codebook (~16 KB)
    replaces three smaller per-component codebooks.
  - **Increases distillation quality**: shared latent space means the
    auxiliary scorer's features and the renderer's emissions live in
    the same discrete index space.
  - **Enables joint training**: optimizer updates one codebook; gradient
    flow is end-to-end through the codebook lookup.

Mathematical core
-----------------
van den Oord 2017 VQ-VAE update (persistent EMA buffer form):

    Encoder produces continuous z_e ∈ R^D
    Codebook is C entries: e_1, ..., e_C ∈ R^D
    Quantize: z_q = e_k where k = argmin_i ||z_e - e_i||²

    Persistent buffers per codebook entry i:
        N_i (count): tracked occurrences of entry i across batches
        m_i (sum): sum of z_e values that mapped to entry i

    EMA update at decay γ (default 0.99 per van den Oord §3.2):
        N_i' = γ · N_i + (1-γ) · n_i     (n_i = batch count for entry i)
        m_i' = γ · m_i + (1-γ) · sum_i   (sum_i = batch sum for entry i)
        e_i' = m_i' / N_i'

    The codebook entries are NOT optimized via gradient descent (they
    are buffers); the encoder is optimized via straight-through estimator
    that backprops the gradient of z_q through the (non-differentiable)
    argmin.

CLAUDE.md compliance
--------------------
- **Codebook EMA decay = 0.99** (van den Oord canonical) — distinct
  from the 0.997 weight-EMA rule because codebooks adapt FASTER than
  weights by design (per CLAUDE.md "EMA — NON-NEGOTIABLE" exception
  clause for VQ-VAE codebook EMA).
- **No silent defaults**: ``SharedCodebookConfig`` validates all fields.
- **No score claims**: this module produces nn.Modules + byte estimators;
  any score claim from a renderer using this codebook MUST be tagged.
- **Strict-scorer-rule**: this module performs NO scorer load.
- **No /tmp paths**: in-memory state only.

Public API
----------
- ``SharedCodebookConfig`` — dataclass + validation
- ``SharedCodebook`` — nn.Module factory; persistent EMA buffers + STE quant
- ``quantize_via_shared_codebook`` — pure helper (alias for forward)
- ``shared_codebook_state_bytes`` — closed-form archive byte cost
- ``compute_codebook_perplexity`` — diagnostic for codebook utilization

Cross-references
----------------
- Council eureka memo: ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md`` (Selfcomp eureka)
- Phase 2 launch memo: ``feedback_paradigm_dezeta_phase2_architectural_launch_20260509.md``
- van den Oord 2017 — VQ-VAE persistent EMA codebook
- T10 IB-Lagrangian aux scorer: :mod:`tac.ib_lagrangian_aux_scorer`
- T18 Ballé nonlinear transform: :mod:`tac.balle_nonlinear_transform`
- Existing related: ``learnable_class_targets``, ``vqvae_codec`` (already
  use the persistent EMA pattern)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch


__all__ = [
    "SharedCodebookConfig",
    "SharedCodebook",
    "quantize_via_shared_codebook",
    "shared_codebook_state_bytes",
    "compute_codebook_perplexity",
    "SharedCodebookError",
]


class SharedCodebookError(RuntimeError):
    """Raised on configuration or runtime invariant violations."""


_BYTES_PER_PARAM_BY_QUANT: dict[str, float] = {
    "fp4": 0.5,
    "fp8": 1.0,
    "fp16": 2.0,
    "fp32": 4.0,
}


@dataclass(frozen=True)
class SharedCodebookConfig:
    """Config for the shared VQ-VAE codebook.

    Attributes
    ----------
    num_entries
        Codebook size ``C``. Council canon = 256 (van den Oord
        §4.2 baseline). MUST be > 1.
    entry_dim
        Embedding dimension ``D`` of each codebook entry. Quantizr-class
        renderers use D=64 to match feature_channels. MUST be > 0.
    ema_decay
        Persistent-EMA decay γ. Per CLAUDE.md EMA exception clause for
        VQ-VAE codebooks: default 0.99 (NOT 0.997). MUST be in
        [0.9, 1.0). < 0.9 over-adapts to recent batches; >= 1.0 freezes.
    epsilon_laplace
        Laplace smoothing constant for the EMA count update. Prevents
        division-by-zero when an entry has never been used. van den Oord
        §3.2 default = 1e-5. MUST be > 0.
    quantization
        Codebook entry quantization for archive byte estimation.
        Default ``"fp16"`` (codebooks are usually shipped at higher
        precision than weights). Must be in ``_BYTES_PER_PARAM_BY_QUANT``.
    label
        Operator label. Required, non-empty.
    """

    num_entries: int
    entry_dim: int
    ema_decay: float
    epsilon_laplace: float
    quantization: str
    label: str

    def __post_init__(self) -> None:
        if not (self.num_entries > 1):
            raise SharedCodebookError(
                f"num_entries must be > 1; got {self.num_entries}"
            )
        if not (self.entry_dim > 0):
            raise SharedCodebookError(
                f"entry_dim must be > 0; got {self.entry_dim}"
            )
        if math.isnan(self.ema_decay) or math.isinf(self.ema_decay):
            raise SharedCodebookError(
                f"ema_decay must be finite; got {self.ema_decay}"
            )
        if not (0.9 <= self.ema_decay < 1.0):
            raise SharedCodebookError(
                f"ema_decay must be in [0.9, 1.0); got {self.ema_decay} "
                "(van den Oord canon = 0.99; codebooks adapt faster than weights)"
            )
        if not (self.epsilon_laplace > 0):
            raise SharedCodebookError(
                f"epsilon_laplace must be > 0; got {self.epsilon_laplace}"
            )
        if math.isnan(self.epsilon_laplace) or math.isinf(self.epsilon_laplace):
            raise SharedCodebookError("epsilon_laplace must be finite")
        if self.quantization not in _BYTES_PER_PARAM_BY_QUANT:
            raise SharedCodebookError(
                f"quantization must be one of "
                f"{sorted(_BYTES_PER_PARAM_BY_QUANT.keys())}; "
                f"got {self.quantization!r}"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise SharedCodebookError("label must be a non-empty string")

    @classmethod
    def vandenoord_canonical(
        cls,
        *,
        label: str,
        num_entries: int = 256,
        entry_dim: int = 64,
    ) -> "SharedCodebookConfig":
        """Council canonical (256 entries × 64 dim, EMA=0.99, FP16)."""
        return cls(
            num_entries=num_entries,
            entry_dim=entry_dim,
            ema_decay=0.99,
            epsilon_laplace=1e-5,
            quantization="fp16",
            label=label,
        )


def _require_torch() -> "tuple[Any, Any]":
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except ImportError as exc:
        raise SharedCodebookError(
            "torch is required for SharedCodebook instantiation."
        ) from exc
    return torch, nn


def SharedCodebook(config: SharedCodebookConfig) -> Any:  # noqa: N802
    """Factory: build the shared VQ-VAE codebook nn.Module.

    The returned module exposes:

      - ``forward(z_e) -> (z_q, indices, commitment_loss)``:
        ``z_e`` shape ``(..., entry_dim)`` (any leading dims).
        Returns the quantized latent (with straight-through gradient),
        the codebook indices, and the commitment loss
        ``||sg(z_e) - z_q||²``.

      - ``update_ema(z_e, indices)``: explicit codebook EMA update.
        Should be called AFTER each training step (van den Oord §3.2).

      - ``state_dict()``: standard nn.Module API; the codebook entries
        are stored as a buffer (not parameter), so they survive
        torch.save/load via state_dict.

    The codebook entries are persistent BUFFERS (not gradient-trained
    parameters); EMA updates are explicit per van den Oord 2017.
    """
    if not isinstance(config, SharedCodebookConfig):
        raise SharedCodebookError(
            f"SharedCodebook requires SharedCodebookConfig; "
            f"got {type(config).__name__}"
        )
    torch, nn = _require_torch()

    num_entries = config.num_entries
    entry_dim = config.entry_dim
    ema_decay = config.ema_decay
    epsilon_laplace = config.epsilon_laplace

    class _SharedCodebookModule(nn.Module):
        """Shared VQ-VAE codebook (van den Oord persistent EMA form)."""

        def __init__(self) -> None:
            super().__init__()
            # Codebook entries: (num_entries, entry_dim).
            init_entries = torch.randn(num_entries, entry_dim) * 0.1
            self.register_buffer("codebook", init_entries)
            # EMA persistent buffers: count + sum per entry.
            self.register_buffer("ema_count", torch.zeros(num_entries))
            self.register_buffer(
                "ema_sum", torch.zeros(num_entries, entry_dim)
            )
            # Initialize ema_count with a small value so first batch's
            # division doesn't divide by zero before update.
            self.ema_count.fill_(1.0)
            self.ema_sum.copy_(init_entries.clone())

        def _encode_indices(self, z_e: Any) -> Any:
            """Find the closest codebook index for each input vector."""
            # Flatten leading dims; final dim is entry_dim.
            flat = z_e.reshape(-1, entry_dim)
            # Distances: ||z_e - e_i||² = ||z_e||² + ||e_i||² - 2 * z_e · e_i.
            z_norm = (flat * flat).sum(dim=1, keepdim=True)
            e_norm = (self.codebook * self.codebook).sum(dim=1)
            cross = flat @ self.codebook.t()
            distances = z_norm + e_norm - 2 * cross
            return distances.argmin(dim=1)

        def forward(
            self, z_e: Any
        ) -> "tuple[Any, Any, Any]":
            """Quantize z_e via the codebook with straight-through gradient.

            Parameters
            ----------
            z_e
                Continuous input ``(..., entry_dim)``.

            Returns
            -------
            (z_q, indices, commitment_loss)
                ``z_q``: same shape as ``z_e``, but quantized to nearest entry
                    (with straight-through gradient: ``z_q.grad`` flows to
                    ``z_e``).
                ``indices``: long tensor of codebook indices, leading shape
                    matching ``z_e[..., 0]``.
                ``commitment_loss``: scalar ``||sg(z_e) - z_q||²`` per
                    van den Oord §3.2 commitment cost (encoder-side
                    regularizer).
            """
            if z_e.shape[-1] != entry_dim:
                raise SharedCodebookError(
                    f"z_e last dim {z_e.shape[-1]} != entry_dim {entry_dim}"
                )
            indices = self._encode_indices(z_e)
            # Look up quantized vectors.
            z_q_flat = self.codebook[indices]
            z_q = z_q_flat.reshape(z_e.shape)
            # Straight-through gradient: forward = z_q, backward = identity to z_e.
            z_q_st = z_e + (z_q - z_e).detach()
            # Commitment loss (encoder must commit to its assigned entry).
            commitment_loss = ((z_e - z_q.detach()) ** 2).mean()
            indices_shape = z_e.shape[:-1]
            indices = indices.reshape(indices_shape) if indices_shape else indices
            return z_q_st, indices, commitment_loss

        @torch.no_grad()
        def update_ema(self, z_e: Any, indices: Any) -> None:
            """Persistent EMA update of codebook entries.

            Per van den Oord §3.2:
                N_i' = γ · N_i + (1-γ) · n_i
                m_i' = γ · m_i + (1-γ) · sum_i
                e_i' = m_i' / N_i' (with Laplace smoothing)

            Should be called once per training step, AFTER the forward
            pass and (optionally) the optimizer step on the encoder.
            """
            flat_z = z_e.reshape(-1, entry_dim).detach()
            flat_idx = indices.reshape(-1).detach()
            # Per-entry counts.
            one_hot = torch.zeros(
                flat_idx.shape[0], num_entries,
                device=z_e.device, dtype=z_e.dtype,
            )
            one_hot.scatter_(1, flat_idx.unsqueeze(1), 1.0)
            batch_count = one_hot.sum(dim=0)
            batch_sum = one_hot.t() @ flat_z
            # EMA update.
            self.ema_count.mul_(ema_decay).add_(
                batch_count, alpha=(1.0 - ema_decay)
            )
            self.ema_sum.mul_(ema_decay).add_(
                batch_sum, alpha=(1.0 - ema_decay)
            )
            # Laplace smoothing + recompute codebook entries.
            n = self.ema_count.sum().item()
            smoothed_count = (
                (self.ema_count + epsilon_laplace)
                / (n + num_entries * epsilon_laplace)
                * n
            )
            self.codebook.copy_(self.ema_sum / smoothed_count.unsqueeze(1))

    return _SharedCodebookModule()


def quantize_via_shared_codebook(
    codebook: Any, z_e: Any
) -> "tuple[Any, Any, Any]":
    """Public alias for the codebook forward pass.

    Mirrors ``codebook(z_e)`` but exists as a stable public name so
    external callers don't depend on the private inner-class.
    """
    return codebook(z_e)


def shared_codebook_state_bytes(config: SharedCodebookConfig) -> int:
    """Closed-form: archive byte cost of the codebook entries.

    Codebook entries are ``num_entries × entry_dim`` floats.
    ``ema_count`` and ``ema_sum`` are TRAINING state and need NOT be
    shipped (operator can re-derive from a single-pass at inference).

    Returns int (rounded up).
    """
    params = config.num_entries * config.entry_dim
    bytes_per_param = _BYTES_PER_PARAM_BY_QUANT[config.quantization]
    return int(math.ceil(params * bytes_per_param))


def compute_codebook_perplexity(
    indices: Any, num_entries: int
) -> float:
    """Diagnostic: codebook entry perplexity (utilization).

    Perplexity = exp(H(p)) where p_i = (count of entry i) / N. Higher
    perplexity means more entries are being used (target ≈ num_entries).
    Low perplexity = codebook collapse (most entries unused).

    Returns a float in [1, num_entries].
    """
    torch, _ = _require_torch()
    if not (num_entries > 1):
        raise SharedCodebookError(
            f"num_entries must be > 1; got {num_entries}"
        )
    flat_idx = indices.reshape(-1)
    n = flat_idx.shape[0]
    if n == 0:
        return 1.0
    one_hot = torch.zeros(n, num_entries, device=indices.device)
    one_hot.scatter_(1, flat_idx.unsqueeze(1), 1.0)
    p = one_hot.mean(dim=0)
    # H(p) with safe log on zero probs.
    p_safe = p.clamp_min(1e-12)
    h = -(p * torch.log(p_safe)).sum().item()
    return float(math.exp(h))
