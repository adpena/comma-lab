"""LoRA (Low-Rank Adaptation) for MaskRenderer TTO at compress time.

Instead of full-model TTO (optimizing all 287K params) or per-pixel TTO
(optimizing 1200×874×1164×3 values), LoRA adapts only low-rank weight deltas
on the renderer's conv layers. This gives model-level adaptation with ~4K
trainable parameters, stored as ~8KB fp16 in the archive.

Key insight: at compress time, we have unlimited compute but limited archive
space. LoRA lets us capture per-video adaptations that the global renderer
misses, with negligible rate cost.

Architecture:
    For each target Conv2d layer with weights W ∈ R^{C_out × C_in × k × k}:
    - Add low-rank factors A ∈ R^{r × C_in × 1 × 1} and B ∈ R^{C_out × r × 1 × 1}
    - Effective weight becomes: W + (B @ A).view(C_out, C_in, 1, 1) (broadcast to k×k)
    - Only A, B are trainable; W is frozen

    With rank r=4 and the 4 target layers:
    - stem_conv: (8, 36) → A: 4×8=32, B: 36×4=144 → 176 params
    - down_conv: (36, 60) → A: 4×36=144, B: 60×4=240 → 384 params
    - up_conv: (60, 36) → A: 4×60=240, B: 36×4=144 → 384 params (transposed)
    - head: (36, 3) → A: 4×36=144, B: 3×4=12 → 156 params
    Total: ~1100 params → 2.2KB fp16

Usage:
    from tac.lora import apply_lora, extract_lora_state, load_lora_state

    # At compress time:
    model = MaskRenderer(...)
    model.load_state_dict(base_weights)
    apply_lora(model, rank=4)  # Injects LoRA adapters
    # ... optimize model (only LoRA params have requires_grad) ...
    lora_delta = extract_lora_state(model)
    torch.save(lora_delta, "lora_delta.pt")

    # At inflate time:
    model = MaskRenderer(...)
    model.load_state_dict(base_weights)
    apply_lora(model, rank=4)
    load_lora_state(model, torch.load("lora_delta.pt"))
"""
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """LoRA adapter for Conv2d layers.

    Wraps an existing Conv2d and adds low-rank adaptation:
        output = original_conv(x) + lora_B(lora_A(x)) * scale

    The original conv weights are frozen. Only A and B are trainable.

    Args:
        original: the Conv2d to adapt
        rank: rank of the low-rank factorization
        scale: scaling factor for the LoRA output (alpha/rank convention)
    """

    def __init__(self, original: nn.Conv2d, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scale = scale

        c_in = original.in_channels
        c_out = original.out_channels

        # Low-rank factors as 1x1 convolutions (efficient, no kernel size dependency)
        # Match stride of original in lora_A so spatial dims align with base_out
        stride = original.stride if isinstance(original.stride, int) else original.stride[0]
        self.lora_A = nn.Conv2d(c_in, rank, kernel_size=1, stride=stride, bias=False)
        self.lora_B = nn.Conv2d(rank, c_out, kernel_size=1, bias=False)

        # Init: A ~ N(0, 1/sqrt(rank)), B = 0 → LoRA output starts at zero
        nn.init.kaiming_normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original output + scaled LoRA adaptation."""
        base_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scale
        return base_out + lora_out

    @property
    def trainable_params(self) -> int:
        """Number of trainable parameters in this adapter."""
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])


class LoRAConvTranspose2d(nn.Module):
    """LoRA adapter for ConvTranspose2d layers.

    Same principle as LoRAConv2d but for transposed convolutions.
    The LoRA branch uses regular 1x1 convs (not transposed) since the
    adaptation operates in the output space.

    Args:
        original: the ConvTranspose2d to adapt
        rank: rank of the low-rank factorization
        scale: scaling factor for the LoRA output
    """

    def __init__(self, original: nn.ConvTranspose2d, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scale = scale

        c_in = original.in_channels
        c_out = original.out_channels

        # LoRA in output space: project from input channels → rank → output channels
        # Applied AFTER the transposed conv's spatial upsampling
        self.lora_A = nn.Conv2d(c_in, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, c_out, kernel_size=1, bias=False)

        nn.init.kaiming_normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original transposed conv + scaled LoRA bypass."""
        base_out = self.original(x)
        # LoRA bypass: downsample via A, upsample via interpolate, apply B
        lora_low = self.lora_A(x)
        # Match spatial dims of base_out
        if lora_low.shape[2:] != base_out.shape[2:]:
            lora_low = F.interpolate(
                lora_low, size=base_out.shape[2:], mode="bilinear", align_corners=False
            )
        lora_out = self.lora_B(lora_low) * self.scale
        return base_out + lora_out

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])


# ---------------------------------------------------------------------------
# Application and extraction
# ---------------------------------------------------------------------------

# Default target layers in MaskRenderer (the 4 largest conv layers)
DEFAULT_TARGET_LAYERS = ["stem_conv", "down_conv", "up_conv", "head"]


def apply_lora(
    model: nn.Module,
    rank: int = 4,
    scale: float = 1.0,
    target_layers: list[str] | None = None,
) -> int:
    """Inject LoRA adapters into a MaskRenderer's conv layers.

    Replaces target Conv2d/ConvTranspose2d layers with LoRA-wrapped versions.
    All non-LoRA parameters are frozen.

    Args:
        model: MaskRenderer instance
        rank: LoRA rank (higher = more capacity but larger archive)
        scale: LoRA scaling factor
        target_layers: list of attribute names to adapt (default: stem, down, up, head)

    Returns:
        Total number of trainable LoRA parameters.
    """
    if target_layers is None:
        target_layers = DEFAULT_TARGET_LAYERS

    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Determine target device from model parameters
    device = next(model.parameters()).device

    total_params = 0
    for name in target_layers:
        original = getattr(model, name, None)
        if original is None:
            continue

        if isinstance(original, nn.ConvTranspose2d):
            adapter = LoRAConvTranspose2d(original, rank=rank, scale=scale)
        elif isinstance(original, nn.Conv2d):
            adapter = LoRAConv2d(original, rank=rank, scale=scale)
        else:
            raise TypeError(f"Layer {name} is {type(original)}, expected Conv2d or ConvTranspose2d")

        # Move adapter to same device as the model
        adapter = adapter.to(device)
        setattr(model, name, adapter)
        total_params += adapter.trainable_params

    return total_params


def extract_lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the LoRA adapter weights from a model.

    Returns a minimal state dict containing only the low-rank factors.
    This is what gets stored in the archive.

    Args:
        model: model with LoRA adapters applied

    Returns:
        Dict mapping layer names to their LoRA A/B weight tensors.
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRAConv2d, LoRAConvTranspose2d)):
            state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load LoRA adapter weights into a model with adapters already applied.

    Args:
        model: model with LoRA adapters applied via apply_lora()
        state: dict from extract_lora_state()
    """
    for name, module in model.named_modules():
        if isinstance(module, (LoRAConv2d, LoRAConvTranspose2d)):
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"
            if a_key in state:
                module.lora_A.weight.data.copy_(state[a_key])
            if b_key in state:
                module.lora_B.weight.data.copy_(state[b_key])


def lora_archive_size_bytes(model: nn.Module, dtype: torch.dtype = torch.float16) -> int:
    """Compute the archive size of LoRA weights in bytes.

    Args:
        model: model with LoRA adapters applied
        dtype: storage dtype (default fp16)

    Returns:
        Total bytes needed to store all LoRA weights at the given dtype.
    """
    total = 0
    for module in model.modules():
        if isinstance(module, (LoRAConv2d, LoRAConvTranspose2d)):
            total += module.lora_A.weight.numel() * torch.finfo(dtype).bits // 8
            total += module.lora_B.weight.numel() * torch.finfo(dtype).bits // 8
    return total
