# SPDX-License-Identifier: MIT
"""LoRA / DoRA adapter wrappers over PR95's frozen HNeRVDecoder.

Mathematical contract (per Hu et al. 2021 LoRA, Liu et al. 2024 DoRA, Meng
et al. 2024 PiSSA):

    LoRA: W_eff = W_frozen + (alpha / r) * B @ A
          where A: (r, in_dim), B: (out_dim, r), alpha is the scaling factor.
          Init: A ~ Kaiming uniform, B = 0 (so W_eff at step 0 == W_frozen).

    DoRA: W_eff = m * (V / ||V||_col) where V = W_frozen + (alpha / r) * B @ A.
          m is per-output-channel magnitude; init m = ||W_frozen||_col.
          ||·||_col denotes column-wise L2 norm (over input dim of the flattened
          conv weight). At step 0, W_eff == W_frozen.

    PiSSA init (opt-in): A, B initialized from top-r SVD of W_frozen instead
          of (Kaiming, 0). Frozen residual is W_frozen - (alpha/r) * B_init @ A_init.
          Convergence is faster on the principal directions.

For Conv2d (out_ch, in_ch, k_h, k_w), we flatten the inner dims to
(out_ch, in_ch * k_h * k_w) and apply LoRA there; the adapter ΔW is reshaped
back to (out_ch, in_ch, k_h, k_w) before adding to the frozen weight. This is
the standard LoRA-for-CNN extension (see Aghajanyan et al. 2020,
Houlsby et al. 2019 lineage).

Targets are partitioned into tiers (per the Fisher/SVD deconstruction memo):
    Tier C: 6 upsample conv blocks (LoRA, large parameter savings)
    Tier B: refine + skip layers (full fine-tune, too small for LoRA overhead)
    Tier A: RGB heads + biases (full fine-tune, high leverage)

Per CLAUDE.md "Forbidden device-selection defaults" — this module never
defaults `device='mps'`. Caller selects device explicitly.

Imports the PR95 base via vendored byte-faithful re-port at
`tac.substrates.pr95_lora_dora.pr95_base` (next file).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tier definitions per `.omx/research/pr95_artifact_deconstruction_20260513.md`.
# The 6 upsample conv blocks where LoRA r=8 saves 84-92% of original params.
DEFAULT_TIER_C_TARGETS: tuple[str, ...] = (
    "blocks.0",
    "blocks.1",
    "blocks.2",
    "blocks.3",
    "blocks.4",
    "blocks.5",
)

# Refine + skip layers; small enough that LoRA overhead dominates; full FT.
DEFAULT_TIER_B_TARGETS: tuple[str, ...] = (
    "refine.0",
    "refine.1",
    "skips.2",
    "skips.3",
    "skips.4",
)

# RGB heads + their biases; high leverage; full FT.
DEFAULT_TIER_A_TARGETS: tuple[str, ...] = (
    "rgb_0",
    "rgb_1",
)


@dataclass
class AdapterConfig:
    """Per-tensor adapter configuration.

    `kind` ∈ {"lora", "dora"}.
    `rank` is the bottleneck width r.
    `alpha` is the LoRA scaling factor (defaults to `rank` so alpha/r == 1.0).
    `init` ∈ {"zero_b", "pissa"}. "zero_b" is canonical LoRA (B=0); "pissa"
        initializes from top-r SVD of the frozen base.
    """

    name: str
    kind: str = "lora"
    rank: int = 8
    alpha: float | None = None
    init: str = "zero_b"

    def __post_init__(self) -> None:
        if self.kind not in {"lora", "dora"}:
            raise ValueError(f"AdapterConfig.kind must be 'lora' or 'dora', got {self.kind!r}")
        if self.rank < 1:
            raise ValueError(f"AdapterConfig.rank must be >= 1, got {self.rank}")
        if self.init not in {"zero_b", "pissa"}:
            raise ValueError(f"AdapterConfig.init must be 'zero_b' or 'pissa', got {self.init!r}")
        if self.alpha is None:
            self.alpha = float(self.rank)


def _flatten_weight(w: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten a weight tensor to (out, in*) for LoRA, preserving original shape."""
    original_shape = tuple(w.shape)
    if w.dim() == 2:
        return w, original_shape
    if w.dim() == 4:
        out_ch = w.shape[0]
        return w.reshape(out_ch, -1), original_shape
    raise ValueError(f"_flatten_weight expects 2D or 4D tensor, got shape {original_shape}")


class LoRAAdapter(nn.Module):
    """Standard LoRA adapter: ΔW = (alpha / r) * B @ A.

    A: (r, in_dim) — Kaiming-uniform init OR top-r right singular vectors (PiSSA).
    B: (out_dim, r) — zero init OR top-r left singular vectors * sqrt(S) (PiSSA).
    """

    def __init__(self, frozen_weight: torch.Tensor, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.original_shape = tuple(frozen_weight.shape)
        flat, _ = _flatten_weight(frozen_weight)
        out_dim, in_dim = flat.shape

        r = config.rank
        self.A = nn.Parameter(torch.empty(r, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, r))
        # scale = alpha / r; expose so DoRA/inflate can read it
        self.scale = float(config.alpha) / float(r)  # type: ignore[arg-type]

        self._init_adapter(flat, config.init)

    def _init_adapter(self, flat_frozen: torch.Tensor, init: str) -> None:
        if init == "zero_b":
            # Canonical LoRA: A ~ Kaiming uniform, B = 0
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        elif init == "pissa":
            # PiSSA: top-r SVD of frozen base
            r = self.config.rank
            U, S, Vh = torch.linalg.svd(flat_frozen.float(), full_matrices=False)
            r_eff = min(r, S.shape[0])
            sqrt_s = torch.sqrt(S[:r_eff].clamp_min(0.0))
            # B = U[:, :r] * sqrt(S[:r])  (scaled left singular vectors)
            # A = sqrt(S[:r]) * Vh[:r, :]
            B_init = U[:, :r_eff] * sqrt_s.unsqueeze(0)
            A_init = sqrt_s.unsqueeze(1) * Vh[:r_eff, :]
            # Account for the alpha/r scale applied at forward — divide by scale
            # so the effective delta at init equals the principal SVD components.
            if self.scale != 0.0:
                B_init = B_init / math.sqrt(self.scale)
                A_init = A_init / math.sqrt(self.scale)
            with torch.no_grad():
                self.A.copy_(A_init)
                if r_eff < r:
                    # pad B with zeros for the unused rank slots
                    pad = torch.zeros(self.B.shape[0], r - r_eff)
                    self.B.copy_(torch.cat([B_init, pad], dim=1))
                else:
                    self.B.copy_(B_init)
        else:  # pragma: no cover — guarded by AdapterConfig.__post_init__
            raise ValueError(init)

    def delta_flat(self) -> torch.Tensor:
        """Return the flattened ΔW = scale * B @ A. Shape (out_dim, in_dim)."""
        return self.scale * (self.B @ self.A)

    def delta(self) -> torch.Tensor:
        """Return ΔW shaped to the frozen tensor's original shape."""
        return self.delta_flat().reshape(self.original_shape)


class DoRAAdapter(nn.Module):
    """DoRA adapter: W_eff = m * (W_frozen + scale·BA) / ||W_frozen + scale·BA||_col.

    `m` is a learnable per-output-channel magnitude. At init,
    m = ||W_frozen||_col so W_eff == W_frozen.

    The norm is column-wise over the flattened (in_dim) axis.
    """

    def __init__(self, frozen_weight: torch.Tensor, config: AdapterConfig):
        super().__init__()
        if config.kind != "dora":
            raise ValueError(f"DoRAAdapter requires config.kind=='dora', got {config.kind!r}")
        self.config = config
        self.original_shape = tuple(frozen_weight.shape)
        flat, _ = _flatten_weight(frozen_weight)
        out_dim, in_dim = flat.shape

        # LoRA part (re-use)
        lora_cfg = AdapterConfig(name=config.name, kind="lora", rank=config.rank,
                                 alpha=config.alpha, init=config.init)
        self.lora = LoRAAdapter(frozen_weight, lora_cfg)

        # Magnitude: per-output-channel scalar, init from column norms of frozen
        with torch.no_grad():
            init_mag = torch.linalg.norm(flat.float(), dim=1)  # (out_dim,)
        self.magnitude = nn.Parameter(init_mag.clone())

    def apply_to(self, frozen_flat: torch.Tensor) -> torch.Tensor:
        """Apply DoRA decomposition to the frozen flat weight.

        Returns the effective flat weight (out_dim, in_dim).
        """
        delta = self.lora.delta_flat()
        V = frozen_flat + delta  # (out_dim, in_dim)
        V_norm = torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(1e-12)
        return self.magnitude.unsqueeze(1) * (V / V_norm)


class PR95LoRADoRADecoder(nn.Module):
    """Frozen PR95 HNeRVDecoder + LoRA/DoRA adapters + tier B/A trainable params.

    Pretrained `decoder_state_dict` is loaded into a fresh HNeRVDecoder and ALL
    parameters frozen by default. Adapters from `adapter_configs` are attached
    to the matching `<target_module>.weight` tensors. Tier B and Tier A targets
    are UNFROZEN for full fine-tune.

    On forward, the effective weight of each adapted module is computed as
        W_frozen + ΔW_lora       (for LoRA)
        m * (W_frozen + ΔW_lora) / ||W_frozen + ΔW_lora||_col   (for DoRA)
    and substituted into the conv/linear forward via a custom forward hook.
    """

    def __init__(
        self,
        decoder_state_dict: dict[str, torch.Tensor],
        meta: dict,
        adapter_configs: Iterable[AdapterConfig] = (),
        tier_b_targets: tuple[str, ...] = DEFAULT_TIER_B_TARGETS,
        tier_a_targets: tuple[str, ...] = DEFAULT_TIER_A_TARGETS,
    ):
        super().__init__()
        # Import here to avoid circular import (pr95_base imports from this file
        # would create a cycle if we put it at module top).
        from .pr95_base import HNeRVDecoder

        self.meta = dict(meta)
        self.base = HNeRVDecoder(
            latent_dim=meta["latent_dim"],
            base_channels=meta["base_channels"],
            eval_size=tuple(meta["eval_size"]),
        )
        self.base.load_state_dict(decoder_state_dict)

        # Freeze EVERYTHING in base by default
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Build adapters keyed by target module name (e.g. "blocks.0")
        self.adapters = nn.ModuleDict()
        self._adapter_kinds: dict[str, str] = {}
        for cfg in adapter_configs:
            mod = self._get_target_module(cfg.name)
            if not hasattr(mod, "weight"):
                raise ValueError(f"Adapter target {cfg.name!r} has no .weight attribute")
            frozen_weight = mod.weight.data
            if cfg.kind == "lora":
                adapter: nn.Module = LoRAAdapter(frozen_weight, cfg)
            elif cfg.kind == "dora":
                adapter = DoRAAdapter(frozen_weight, cfg)
            else:  # pragma: no cover — guarded by AdapterConfig.__post_init__
                raise ValueError(cfg.kind)
            # ModuleDict requires `.` -> `_` munging
            key = cfg.name.replace(".", "__")
            self.adapters[key] = adapter
            self._adapter_kinds[cfg.name] = cfg.kind

        # Unfreeze tier B + tier A modules for full FT
        self.tier_b_targets = tuple(tier_b_targets)
        self.tier_a_targets = tuple(tier_a_targets)
        for name in (*tier_b_targets, *tier_a_targets):
            try:
                mod = self._get_target_module(name)
            except AttributeError:
                continue
            for p in mod.parameters():
                p.requires_grad_(True)

        # Override conv/linear forward to inject effective weights
        self._install_adapter_hooks()

    def _get_target_module(self, dotted_name: str) -> nn.Module:
        """Resolve `blocks.0` -> self.base.blocks[0] etc."""
        obj: nn.Module = self.base
        for part in dotted_name.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)  # type: ignore[index]
        return obj

    def _install_adapter_hooks(self) -> None:
        """Patch the .forward of each adapted Conv2d/Linear to use W_eff."""
        for name in list(self._adapter_kinds.keys()):
            mod = self._get_target_module(name)
            adapter_key = name.replace(".", "__")
            kind = self._adapter_kinds[name]

            # Capture closure variables
            def make_forward(_mod: nn.Module, _adapter_key: str, _kind: str):
                base_forward = _mod.forward

                def adapted_forward(x: torch.Tensor) -> torch.Tensor:
                    adapter = self.adapters[_adapter_key]
                    frozen_weight = _mod.weight
                    if _kind == "lora":
                        delta = adapter.delta()  # type: ignore[union-attr]
                        eff_weight = frozen_weight + delta
                    else:  # dora
                        flat, original_shape = _flatten_weight(frozen_weight)
                        eff_flat = adapter.apply_to(flat)  # type: ignore[union-attr]
                        eff_weight = eff_flat.reshape(original_shape)
                    # Re-issue the conv/linear with effective weight
                    if isinstance(_mod, nn.Conv2d):
                        return F.conv2d(
                            x, eff_weight, _mod.bias,
                            stride=_mod.stride, padding=_mod.padding,
                            dilation=_mod.dilation, groups=_mod.groups,
                        )
                    if isinstance(_mod, nn.Linear):
                        return F.linear(x, eff_weight, _mod.bias)
                    return base_forward(x)

                return adapted_forward

            mod.forward = make_forward(mod, adapter_key, kind)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.base(z)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
