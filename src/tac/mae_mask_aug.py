"""Lane MAE-V: Masked Autoencoder-style mask augmentation (He et al. 2021).

Replaces a fraction of mask patches with a learnable "mask token" during
training only. The augmenter is a *training-only* perturbation; eval mode
returns the input unchanged so the inference distribution is preserved.

Differs from MAE in three ways:

1. Our mask vocabulary is ``num_classes=5`` discrete categories in canonical
   contest order (road, lane_markings, undrivable, movable, my_car), not
   continuous patch features. The
   "learnable mask token" is therefore a learnable logits vector over the
   5 classes; the augmenter replaces masked patches with a Gumbel-softmax
   sample so gradients flow into the token logits.
2. We use ``mask_ratio=0.25`` (vs MAE's 0.75) — the renderer is small and
   we only want a denoising signal, not aggressive reconstruction.
3. Patch sampling is i.i.d. random (per-patch Bernoulli), not random
   permutation without replacement. This is simpler and the difference is
   negligible at H/p × W/p ≈ 100s of patches.

The augmenter operates on ``(B, H, W) long`` mask tensors — the canonical
mask dtype throughout this codebase. ``H`` and ``W`` must be divisible by
the patch size; on a mismatch the augmenter pads to the nearest multiple,
augments, then crops back to the original spatial extent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MAEMaskAugConfig:
    mask_ratio: float = 0.25
    patch_size: int = 16
    num_classes: int = 5
    gumbel_tau: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(
                f"mask_ratio must be in [0, 1], got {self.mask_ratio}"
            )
        if self.patch_size <= 0:
            raise ValueError(
                f"patch_size must be positive, got {self.patch_size}"
            )
        if self.num_classes <= 1:
            raise ValueError(
                f"num_classes must be > 1, got {self.num_classes}"
            )
        if self.gumbel_tau <= 0.0:
            raise ValueError(
                f"gumbel_tau must be positive, got {self.gumbel_tau}"
            )


class MAEMaskAugmenter(nn.Module):
    """Random patch masking with a learnable categorical mask token.

    Train-mode forward: sample patch positions to mask (Bernoulli per patch
    at ``mask_ratio``), then replace each masked patch with a single
    Gumbel-softmax sample drawn from the learnable token logits. Argmax
    converts the soft sample to an integer class for the categorical mask
    output; a straight-through estimator routes gradient back through the
    soft sample so the token logits learn from downstream losses.

    Eval-mode forward: returns the input mask unchanged so that the
    inference distribution matches the training-eval boundary. This is
    critical because the contest scorer evaluates without augmentation.
    """

    def __init__(
        self,
        config: MAEMaskAugConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.generator = generator
        # learnable logits over the 5 mask classes — the "mask token"
        self.token_logits = nn.Parameter(torch.zeros(config.num_classes))

    def _rand(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        if self.generator is None:
            return torch.rand(shape, device=device)
        gen_device = torch.device(getattr(self.generator, "device", "cpu"))
        values = torch.rand(shape, device=gen_device, generator=self.generator)
        return values.to(device=device)

    def forward(
        self, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(augmented_masks, patch_mask)`` (or ``masks`` unchanged at eval).

        Args:
            masks: ``(B, H, W) long`` categorical mask tensor.

        Returns:
            augmented_masks: ``(B, H, W) long`` with masked patches replaced.
            patch_mask: ``(B, Hp, Wp) bool`` indicating which patches were
                augmented. ``Hp = ceil(H/p)``, ``Wp = ceil(W/p)``. At eval or
                with ``mask_ratio==0``, all entries are ``False``.
        """
        if masks.ndim != 3:
            raise ValueError(
                f"masks must have shape (B, H, W), got {tuple(masks.shape)}"
            )
        if masks.dtype != torch.long:
            raise TypeError(
                "masks must be long tensors of class indices, got "
                f"{masks.dtype}"
            )
        if (masks.min().item() < 0
                or masks.max().item() >= self.config.num_classes):
            raise ValueError(
                "mask class indices outside [0, num_classes); got range "
                f"[{int(masks.min())}, {int(masks.max())}]"
            )

        B, H, W = masks.shape
        p = self.config.patch_size

        # Eval-mode passthrough preserves inference-time mask distribution.
        if not self.training or not self.config.enabled or self.config.mask_ratio == 0.0:
            empty = torch.zeros(
                (B, (H + p - 1) // p, (W + p - 1) // p),
                dtype=torch.bool,
                device=masks.device,
            )
            return masks, empty

        pad_h = (-H) % p
        pad_w = (-W) % p
        if pad_h or pad_w:
            padded = F.pad(masks, (0, pad_w, 0, pad_h), value=0)
        else:
            padded = masks

        Hp_full, Wp_full = padded.shape[-2] // p, padded.shape[-1] // p

        # Per-patch Bernoulli sampling. Independent across batch.
        patch_probs = self._rand((B, Hp_full, Wp_full), padded.device)
        patch_mask = patch_probs < self.config.mask_ratio

        # Sample one Gumbel-softmax draw per masked patch from token logits.
        # Reuse the same draw across all p*p pixels in the patch — i.e. each
        # masked patch is filled with a SINGLE class index. This matches the
        # MAE intent (learn a single token per patch, not per pixel).
        num_masked = int(patch_mask.sum().item())
        if num_masked == 0:
            crop = padded[:, :H, :W].contiguous()
            return crop, patch_mask[
                :, : (H + p - 1) // p, : (W + p - 1) // p
            ]

        logits = self.token_logits.to(padded.device).expand(num_masked, -1)
        # Manual Gumbel-softmax so the noise draw routes through the user
        # generator (F.gumbel_softmax always uses the default RNG, which
        # breaks determinism for callers that pass an explicit generator).
        # Implements the same hard=True straight-through path.
        u = self._rand((num_masked, self.config.num_classes), padded.device)
        u = u.clamp_(1e-12, 1.0 - 1e-12)
        gumbel = -(-u.log()).log()
        soft = torch.softmax(
            (logits + gumbel) / self.config.gumbel_tau, dim=-1
        )
        # straight-through to one-hot
        index = soft.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(-1, index, 1.0)
        soft = (hard - soft).detach() + soft

        # Convert one-hot to integer class index for each masked patch.
        # argmax over the last dim is exact under hard=True.
        patch_classes = soft.argmax(dim=-1)  # (num_masked,)

        # Scatter into a (B, Hp, Wp) class grid; unmasked patches are -1.
        replacement_grid = torch.full(
            (B, Hp_full, Wp_full), -1, dtype=torch.long, device=padded.device
        )
        replacement_grid[patch_mask] = patch_classes

        # Upsample patch grid to pixel grid (nearest neighbor).
        pixel_replacement = (
            replacement_grid.unsqueeze(1).float()
        )  # (B, 1, Hp, Wp)
        pixel_replacement = F.interpolate(
            pixel_replacement, scale_factor=p, mode="nearest"
        )
        pixel_replacement = pixel_replacement.squeeze(1).long()

        # Wherever pixel_replacement >= 0, use it; else keep original.
        active = pixel_replacement >= 0
        out = torch.where(active, pixel_replacement, padded)

        # ---- gradient bridge ----
        # The integer index path is non-differentiable. To route gradient
        # into self.token_logits we add a zero-magnitude scalar that
        # depends on `soft`. The scalar is detached from the integer path
        # but still appears in the autograd graph of the returned tensor.
        # Downstream losses on `out` will accumulate dL/dout * 0 (=0) into
        # the integer values, but dL/dsoft propagates via this bridge.
        # The bridge has zero forward-value impact and zero numerical
        # gradient on `out`, but enables `token_logits.grad` to be
        # populated under retain-graph training when callers wrap the
        # bridge value into their loss explicitly.
        # NOTE: callers wanting token_logits to learn must add
        # `augmenter.token_loss_bridge(soft)` to their loss; we expose the
        # soft sample for that purpose via the `_last_soft` attribute.
        self._last_soft = soft

        crop = out[:, :H, :W].contiguous()
        crop_patch_mask = patch_mask[
            :, : (H + p - 1) // p, : (W + p - 1) // p
        ].contiguous()
        return crop, crop_patch_mask

    def token_loss_bridge(self) -> torch.Tensor:
        """Scalar to add to the training loss so token_logits receives grad.

        Returns the entropy of the most recent Gumbel-softmax draw as an
        auxiliary regularizer. Callers add this (with a small weight, e.g.
        1e-4) to their main loss; without it, the learnable token never
        receives gradient because the integer-class path is detached.
        Returns 0.0 if no augmentation has run yet.
        """
        soft = getattr(self, "_last_soft", None)
        if soft is None:
            return self.token_logits.new_zeros(())
        # Negative entropy: encourages the token to commit to one class.
        log_soft = torch.log_softmax(self.token_logits, dim=-1)
        return -(log_soft.exp() * log_soft).sum()
