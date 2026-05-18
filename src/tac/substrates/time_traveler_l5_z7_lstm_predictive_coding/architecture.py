# SPDX-License-Identifier: MIT
"""Z7 GRU recurrent predictive-coding scaffold.

This module is intentionally smaller than the eventual Z7PCWM1 substrate. It
lands only the substrate-distinguishing recurrent predictor primitive needed by
the pre-build smoke surface:

    forward(z_prev, ego_motion) -> z_pred

The archive grammar, inflate runtime, and training/export smoke now exist as
false-authority prebuild surfaces. Score-aware full training and exact eval
remain blocked behind the Z7 Wave N+1 council path. Keeping this predictor
reusable and shape-compatible with Z6 lets queue/readiness tooling replace
"missing package" with a precise trained-packet/exact-eval blocker without
granting score or dispatch authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tac.substrates.time_traveler_l5_z6.architecture import _Z6Decoder

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width)."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

CONTEXT_CONDITIONING_MODES: tuple[str, ...] = ("none", "latent_affine")
"""Opt-in Z7 context-conditioning modes understood by train and inflate."""


@dataclass(frozen=True)
class Z7GruPredictiveCodingConfig:
    """Static design-time parameters for the Z7 recurrent predictor scaffold.

    Args:
        latent_dim: per-pair latent dimensionality, matching Z6 by default.
        ego_motion_dim: ego-motion vector dimension.
        gru_hidden_dim: hidden state width for each GRU layer.
        gru_num_layers: stacked GRUCell layer count.
        stateful: when True, recurrent state persists across forward calls.
        identity_predictor: when True, returns z_prev unchanged with no params.
        beta_ib: beta-IB placeholder for future full trainer wiring.
    """

    latent_dim: int = 24
    ego_motion_dim: int = 8
    gru_hidden_dim: int = 128
    gru_num_layers: int = 1
    stateful: bool = True
    identity_predictor: bool = False
    beta_ib: float = 1.0
    num_pairs: int = NUM_PAIRS
    decoder_embed_dim: int = 32
    decoder_initial_grid_h: int = 24
    decoder_initial_grid_w: int = 32
    decoder_channels: tuple[int, ...] = (32, 24, 16, 12)
    decoder_num_upsample_blocks: int = 4
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    latent_init_std: float = 0.02
    context_conditioning_mode: str = "none"
    context_affine_strength: float = 0.125

    @property
    def predictor_input_dim(self) -> int:
        return self.latent_dim + self.ego_motion_dim


def normalize_context_conditioning_mode(value: str) -> str:
    """Normalize and validate the Z7 decoder-context conditioning mode."""

    mode = str(value).strip().lower().replace("-", "_")
    if mode not in CONTEXT_CONDITIONING_MODES:
        allowed = ", ".join(CONTEXT_CONDITIONING_MODES)
        raise ValueError(f"context_conditioning_mode must be one of: {allowed}")
    return mode


class GruRecurrentPredictor(nn.Module):
    """GRU-bound Z7 next-latent predictor.

    The public forward signature matches the Z6 predictor family so the future
    trainer can swap predictor mechanisms without changing the outer
    predictive-coding loop. Stateful mode is the Z7 distinguishing feature:
    hidden state carries temporal context across the 600-pair sequence.
    """

    def __init__(self, config: Z7GruPredictiveCodingConfig) -> None:
        super().__init__()
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if config.ego_motion_dim <= 0:
            raise ValueError("ego_motion_dim must be positive")
        if config.gru_hidden_dim <= 0:
            raise ValueError("gru_hidden_dim must be positive")
        if config.gru_num_layers <= 0:
            raise ValueError("gru_num_layers must be positive")
        if config.beta_ib < 0:
            raise ValueError("beta_ib must be non-negative")

        self.config = config
        self.latent_dim = config.latent_dim
        self.ego_motion_dim = config.ego_motion_dim
        self.identity_predictor = config.identity_predictor
        self.stateful = config.stateful
        self._h: list[torch.Tensor] | None = None

        if config.identity_predictor:
            return

        cells: list[nn.GRUCell] = []
        input_dim = config.predictor_input_dim
        for _ in range(config.gru_num_layers):
            cells.append(nn.GRUCell(input_dim, config.gru_hidden_dim))
            input_dim = config.gru_hidden_dim
        self.gru_cells = nn.ModuleList(cells)
        self.output_projection = nn.Linear(config.gru_hidden_dim, config.latent_dim)

    def reset_state(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Reset hidden state for a new sequence."""

        if self.identity_predictor:
            self._h = None
            return
        self._h = [
            torch.zeros(
                batch_size,
                self.config.gru_hidden_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.config.gru_num_layers)
        ]

    def forward(self, z_prev: torch.Tensor, ego_motion: torch.Tensor) -> torch.Tensor:
        """Predict z_t from z_{t-1} and ego-motion side information."""

        if z_prev.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z_prev last dim {z_prev.shape[-1]} != latent_dim {self.latent_dim}"
            )
        if ego_motion.shape[-1] != self.ego_motion_dim:
            raise ValueError(
                f"ego_motion last dim {ego_motion.shape[-1]} != ego_motion_dim "
                f"{self.ego_motion_dim}"
            )
        if z_prev.shape[0] != ego_motion.shape[0]:
            raise ValueError(
                f"batch mismatch: z_prev batch {z_prev.shape[0]} != "
                f"ego_motion batch {ego_motion.shape[0]}"
            )
        if self.identity_predictor:
            return z_prev

        batch = z_prev.shape[0]
        if self._h is None or not self.stateful or self._h[0].shape[0] != batch:
            self.reset_state(batch, device=z_prev.device, dtype=z_prev.dtype)
        assert self._h is not None

        x = torch.cat([z_prev, ego_motion], dim=-1)
        new_state: list[torch.Tensor] = []
        for layer_index, cell in enumerate(self.gru_cells):
            h_t = cell(x, self._h[layer_index])
            new_state.append(h_t)
            x = h_t
        self._h = new_state if self.training or self.stateful else None
        return self.output_projection(x)

    def sequence_forward(
        self,
        z0: torch.Tensor,
        ego_motion_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregress over a batch-first ego-motion sequence.

        Args:
            z0: initial latent, shape ``(B, latent_dim)``.
            ego_motion_sequence: ``(B, T, ego_motion_dim)``.

        Returns:
            Predicted latent sequence, shape ``(B, T, latent_dim)``.
        """

        if ego_motion_sequence.ndim != 3:
            raise ValueError("ego_motion_sequence must have shape (B, T, ego_dim)")
        if ego_motion_sequence.shape[0] != z0.shape[0]:
            raise ValueError("z0 and ego_motion_sequence batch dimensions differ")
        if ego_motion_sequence.shape[-1] != self.ego_motion_dim:
            raise ValueError(
                f"ego_motion_sequence last dim {ego_motion_sequence.shape[-1]} "
                f"!= ego_motion_dim {self.ego_motion_dim}"
            )
        self.reset_state(z0.shape[0], device=z0.device, dtype=z0.dtype)
        z = z0
        outs: list[torch.Tensor] = []
        for t in range(ego_motion_sequence.shape[1]):
            z = self(z, ego_motion_sequence[:, t, :])
            outs.append(z)
        return torch.stack(outs, dim=1)

    def num_parameters(self) -> int:
        """Count trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_z6_compatible_signature(self) -> str:
        """Return a compact human-readable signature/custody string."""

        return (
            "GruRecurrentPredictor canonical signature: "
            f"forward(z_prev: (B, {self.latent_dim}), "
            f"ego_motion: (B, {self.ego_motion_dim})) -> "
            f"(B, {self.latent_dim}); hidden_dim={self.config.gru_hidden_dim}; "
            f"layers={self.config.gru_num_layers}; stateful={self.stateful}; "
            f"identity_predictor={self.identity_predictor}"
        )


class LatentAffineContextConditioner(nn.Module):
    """Bounded affine modulation of replayed latents from recurrent context.

    This is the smallest byte-closed branch that makes Z7 test a contextual
    decoder mechanism instead of only a raw predictive residual channel.
    """

    def __init__(self, config: Z7GruPredictiveCodingConfig) -> None:
        super().__init__()
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if config.context_affine_strength < 0:
            raise ValueError("context_affine_strength must be non-negative")
        self.latent_dim = int(config.latent_dim)
        self.strength = float(config.context_affine_strength)
        self.proj = nn.Linear(self.latent_dim, 2 * self.latent_dim)

    def forward(self, latents: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """Return context-modulated latents with the same shape as ``latents``."""

        if latents.shape != contexts.shape:
            raise ValueError(
                f"latents shape {tuple(latents.shape)} must match contexts shape "
                f"{tuple(contexts.shape)}"
            )
        if latents.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latents last dim {latents.shape[-1]} != latent_dim {self.latent_dim}"
            )
        scale, shift = self.proj(contexts.to(dtype=latents.dtype)).chunk(2, dim=-1)
        scale = torch.tanh(scale) * self.strength
        shift = torch.tanh(shift) * self.strength
        return latents * (1.0 + scale) + shift

    def num_parameters(self) -> int:
        """Count trainable conditioner parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Z7GruPredictiveCodingSubstrate(nn.Module):
    """Z7 prebuild substrate: GRU recurrence + Z6-compatible RGB decoder.

    This binds the Z7 distinguishing recurrent predictor to byte-exportable
    latent, residual, ego-motion, and decoder streams. It is intentionally
    score-neutral: training code must still tag outputs as proxy/prebuild until
    paired exact eval lands.
    """

    def __init__(self, config: Z7GruPredictiveCodingConfig) -> None:
        super().__init__()
        if config.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if config.decoder_num_upsample_blocks <= 0:
            raise ValueError("decoder_num_upsample_blocks must be positive")
        if len(config.decoder_channels) < config.decoder_num_upsample_blocks:
            raise ValueError(
                "decoder_channels must have at least decoder_num_upsample_blocks "
                "entries"
            )
        if config.output_height <= 0 or config.output_width <= 0:
            raise ValueError("output_height/output_width must be positive")
        if config.latent_init_std < 0:
            raise ValueError("latent_init_std must be non-negative")
        context_mode = normalize_context_conditioning_mode(
            config.context_conditioning_mode
        )
        if config.context_affine_strength < 0:
            raise ValueError("context_affine_strength must be non-negative")

        self.config = config
        self.context_conditioning_mode = context_mode
        self.predictor = GruRecurrentPredictor(config)
        self.context_conditioner: LatentAffineContextConditioner | None = None
        if context_mode == "latent_affine":
            self.context_conditioner = LatentAffineContextConditioner(config)
        self.decoder = _Z6Decoder(
            latent_dim=config.latent_dim,
            embed_dim=config.decoder_embed_dim,
            initial_grid_h=config.decoder_initial_grid_h,
            initial_grid_w=config.decoder_initial_grid_w,
            decoder_channels=config.decoder_channels,
            num_upsample_blocks=config.decoder_num_upsample_blocks,
            output_height=config.output_height,
            output_width=config.output_width,
        )
        self.latent_init = nn.Parameter(
            torch.randn(config.latent_dim) * float(config.latent_init_std)
        )
        self.residuals = nn.Parameter(
            torch.zeros(config.num_pairs, config.latent_dim)
        )
        self.register_buffer(
            "ego_motion_buffer",
            torch.zeros(config.num_pairs, config.ego_motion_dim),
            persistent=True,
        )

    def replay_latents_and_contexts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay recurrent latents plus pre-residual temporal contexts."""

        z = self.latent_init.view(1, self.config.latent_dim)
        self.predictor.reset_state(1, device=z.device, dtype=z.dtype)
        outs: list[torch.Tensor] = []
        contexts: list[torch.Tensor] = []
        for t in range(self.config.num_pairs):
            ego_t = self.ego_motion_buffer[t : t + 1].to(dtype=z.dtype)
            pred = self.predictor(z, ego_t)
            z = pred + self.residuals[t : t + 1]
            contexts.append(pred.squeeze(0))
            outs.append(z.squeeze(0))
        return torch.stack(outs, dim=0), torch.stack(contexts, dim=0)

    def replay_latents(self) -> torch.Tensor:
        """Replay the full recurrent latent sequence."""

        latents, _contexts = self.replay_latents_and_contexts()
        return latents

    def condition_latents(
        self,
        latents: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured context-conditioning branch before decoding."""

        if self.context_conditioner is None:
            return latents
        return self.context_conditioner(latents, contexts)

    def reconstruct_all_pairs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode every pair in sequence order."""

        latents, contexts = self.replay_latents_and_contexts()
        decoder_latents = self.condition_latents(latents, contexts)
        rgb_0, rgb_1 = self.decoder(decoder_latents)
        return rgb_0, rgb_1, latents

    def reconstruct_pair(
        self,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode selected pair indices after deterministic sequence replay."""

        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.numel() == 0:
            raise ValueError("pair_indices must be non-empty")
        if (
            pair_indices.min().item() < 0
            or pair_indices.max().item() >= self.config.num_pairs
        ):
            raise ValueError(
                f"pair_indices out of range [0, {self.config.num_pairs}); "
                f"got [{pair_indices.min().item()}, {pair_indices.max().item()}]"
            )
        rgb_0, rgb_1, latents = self.reconstruct_all_pairs()
        return rgb_0[pair_indices], rgb_1[pair_indices], latents[pair_indices]

    def decoder_metadata(self) -> dict[str, int | list[int] | float | str | bool]:
        """Return metadata required by the Z7PCWM1 inflate runtime."""

        return {
            "decoder_embed_dim": int(self.config.decoder_embed_dim),
            "decoder_initial_grid_h": int(self.config.decoder_initial_grid_h),
            "decoder_initial_grid_w": int(self.config.decoder_initial_grid_w),
            "decoder_channels": [int(c) for c in self.config.decoder_channels],
            "decoder_num_upsample_blocks": int(
                self.config.decoder_num_upsample_blocks
            ),
            "output_height": int(self.config.output_height),
            "output_width": int(self.config.output_width),
            "latent_init_std": float(self.config.latent_init_std),
            "context_conditioning_mode": self.context_conditioning_mode,
            "context_affine_strength": float(self.config.context_affine_strength),
            "context_conditioner_state_dict_in_encoder_blob": (
                self.context_conditioner is not None
            ),
        }

    def num_parameters(self) -> int:
        """Count trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_breakdown(self) -> dict[str, int]:
        """Return decoder/predictor/latent/residual parameter counts."""

        return {
            "decoder": self.decoder.num_parameters(),
            "predictor": self.predictor.num_parameters(),
            "context_conditioner": (
                0
                if self.context_conditioner is None
                else self.context_conditioner.num_parameters()
            ),
            "latent_init": self.latent_init.numel(),
            "residuals": self.residuals.numel(),
            "total": self.num_parameters(),
        }


__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "CONTEXT_CONDITIONING_MODES",
    "GruRecurrentPredictor",
    "LatentAffineContextConditioner",
    "Z7GruPredictiveCodingConfig",
    "Z7GruPredictiveCodingSubstrate",
    "normalize_context_conditioning_mode",
]
