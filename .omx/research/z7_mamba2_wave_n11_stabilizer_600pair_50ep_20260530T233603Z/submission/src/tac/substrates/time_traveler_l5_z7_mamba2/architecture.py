# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 substrate architecture — Mamba2 selective state-space + Z6-compatible RGB decoder.

This module wires the canonical Mamba2Predictor primitive
(``tac.optimization.mamba2_predictor.Mamba2Predictor``) into a full
substrate-distinguishing architecture: per-pair latent autoregression via
Mamba-2 selective state-space recurrence, followed by a Z6-compatible
PixelShuffle RGB decoder. Sister to the Z7-GRU/LSTM canonical pattern
(``tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture``)
where the *predictor primitive* is swapped from GRUCell to Mamba-2 SSM.

Per the Z7-Mamba-2 design memo
(``.omx/research/z7_mamba2_substrate_design_memo_20260518.md``):
this is the TOP-5 #2 FULL substrate per the deep-research wave
(``.omx/research/comprehensive_research_wave_20260518.md`` §0 + §2.2 + §3.6).

Canonical-vs-unique decision per layer
--------------------------------------

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

1. **Predictor primitive (UNIQUE-FORK)**: Mamba2Predictor (selective
   state-space, O(N) selectivity, continuous-time-discretized
   recurrence). Replaces Z7-LSTM/GRU primitive. Different mathematical
   structure → genuine class-shift candidate.
2. **Decoder (CANONICAL-ADOPT)**: ``_Z6Decoder`` from
   ``tac.substrates.time_traveler_l5_z6.architecture``. Sister-substrate
   parity per HNeRV parity L5 (full renderer not single-component slot)
   + clean composition with Z7-LSTM for paired-comparison.
3. **Context conditioner (CANONICAL-ADOPT)**: ``LatentAffineContextConditioner``
   from Z7-LSTM sister; identical canonical sister.
4. **Substrate skeleton (CANONICAL-ADOPT)**: per-pair autoregressive
   replay + residual + ego-motion buffer matches Z7-LSTM/GRU canonical.
5. **Latent dim (CANONICAL-ADOPT)**: 24 matches Z6/Z7-GRU sister.
6. **Ego motion dim (CANONICAL-ADOPT)**: 8 matches Z7-GRU PoseNet-projection.
7. **Mamba-2 d_state (UNIQUE-DEFAULT)**: 16 per upstream canonical for
   language; CC-9 CARGO-CULTED-PENDING-VERIFICATION per design memo §2.
8. **Mamba-2 d_model (UNIQUE-DEFAULT)**: 64 per design memo §7 (sister
   to GRU hidden_dim=128 halved for parameter parity).
9. **Mamba-2 expand (CANONICAL-ADOPT)**: 2 from upstream reference.
10. **Stateful (UNIQUE-FORK semantic)**: True (Wyner-Ziv implicit
    side-info channel pattern per Catalog #311); ablation control
    available via ``stateful=False``.

6-hook wire-in declaration per Catalog #125
-------------------------------------------

1. Sensitivity-map: Mamba-2 selective-projection gradient norms (A_proj,
   B_proj, C_proj) ARE the per-tensor importance signal; registered at
   ``tac.sensitivity_map.time_traveler_l5_z7_mamba2`` at first dispatch.
2. Pareto constraint: ``mamba2_residual_entropy ≤ ε_residual`` adds
   bound to convex feasibility region.
3. Bit-allocator hook: per-pair residual bit allocation derives from
   Mamba-2 selectivity-matrix amplitude.
4. Cathedral autopilot dispatch: recipe loaded by autopilot ranker.
5. Continual-learning posterior: every Z7-Mamba-2 empirical anchor
   seeds posterior via ``posterior_update_locked``.
6. Probe-disambiguator: ``identity_predictor=True`` mode IS the probe
   (no learning; sister to Z6/Z7-GRU identity_predictor pattern).

[verified-against: tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture]
[verified-against: tac.substrates.time_traveler_l5_z6.architecture._Z6Decoder]
[verified-against: tac.optimization.mamba2_predictor.Mamba2Predictor]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tac.optimization.mamba2_predictor import (
    MAMBA_SSM_AVAILABLE,
    Mamba2Predictor,
    Mamba2PredictorConfig,
)
from tac.substrates.time_traveler_l5_z6.architecture import _Z6Decoder
from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture import (
    CONTEXT_CONDITIONING_MODES,
    EVAL_HW,
    NUM_PAIRS,
    LatentAffineContextConditioner,
    normalize_context_conditioning_mode,
)

__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "CONTEXT_CONDITIONING_MODES",
    "MAMBA_SSM_AVAILABLE",
    "Z7Mamba2PredictiveCodingConfig",
    "Z7Mamba2PredictiveCodingSubstrate",
    "normalize_context_conditioning_mode",
]


@dataclass(frozen=True)
class Z7Mamba2PredictiveCodingConfig:
    """Static design-time parameters for the Z7-Mamba-2 substrate.

    Defaults match the parent design memo §7 architectural specification.

    Args:
        latent_dim: per-pair latent dimensionality (24 = Z6 sister parity).
        ego_motion_dim: ego-motion vector dim (8 = Z7-GRU PoseNet-projection).
        d_model: Mamba-2 internal model dim (64; sister to GRU hidden_dim=128
            halved for parameter parity).
        d_state: Mamba-2 selective state-space dim (16 canonical; CC-9
            CARGO-CULTED-PENDING-VERIFICATION for dashcam 600-pair).
        expand: Mamba-2 expansion factor (2 from upstream).
        d_conv: Mamba-2 conv1d kernel size (4 canonical).
        backend: ``"auto"`` (default), ``"mamba_ssm"``,
            ``"reference_torch"`` (Mamba-1 S6 reference), or
            ``"ssd_reference"`` (canonical Mamba-2 SSD via
            :mod:`tac.substrates._shared.mamba2_ssd` tri-backend
            NUMPY/PYTORCH/MLX helper). The ``ssd_reference`` backend
            is the canonical $0 macOS MLX path per CLAUDE.md 8th
            MLX-first standing directive (added 2026-05-30; commit
            ``b2936fb81`` canonical helper landed with 33 byte-stable
            parity tests). The ``reference_torch`` backend is
            preserved for backward compat + cite-chain per CLAUDE.md
            HISTORICAL_PROVENANCE Catalog #110/#113 (existing
            ``z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1``
            canonical equation anchors cite reference_torch).
        ssd_nheads: number of SSD heads when ``backend='ssd_reference'``.
            Default ``None`` derives ``nheads = d_inner // ssd_headdim``
            from ``ssd_headdim``; explicit override required if
            ``d_inner % ssd_headdim != 0``. Ignored for non-SSD backends.
        ssd_headdim: per-head feature dim when ``backend='ssd_reference'``;
            default 64 (canonical from state-spaces/mamba upstream). Used
            with ``ssd_nheads`` to construct the canonical helper config.
            Ignored for non-SSD backends.
        stateful: True (default) maintains hidden state across the
            600-pair sequence (Wyner-Ziv implicit side-info channel).
        identity_predictor: True returns z_prev unchanged (Catalog #125
            hook #6 probe-disambiguator control).
        beta_ib: beta-IB-Lagrangian placeholder; canonical Wave-N+1
            anchor from C6 IBPS Phase 2 empirical beta-optimal.
        num_pairs: contest pair count (600).
        decoder_*: Z6Decoder canonical parameters.
        context_conditioning_mode: ``"none"`` or ``"latent_affine"``.
        context_affine_strength: affine modulation strength (0.125).
    """

    latent_dim: int = 24
    ego_motion_dim: int = 8
    d_model: int = 64
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    backend: str = "auto"
    ssd_nheads: int | None = None
    ssd_headdim: int = 64
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
    def d_inner(self) -> int:
        """Mamba-2 inner dim after expansion."""
        return self.expand * self.d_model

    @property
    def predictor_input_dim(self) -> int:
        """Concat dim for (z_prev, ego_motion) input."""
        return self.latent_dim + self.ego_motion_dim

    def to_mamba2_predictor_config(self) -> Mamba2PredictorConfig:
        """Build the underlying Mamba2PredictorConfig from this config.

        Threads the new ``ssd_nheads`` / ``ssd_headdim`` fields through
        to the underlying ``Mamba2PredictorConfig`` so the canonical
        Mamba-2 SSD backend (per :mod:`tac.substrates._shared.mamba2_ssd`)
        is reachable from the Z7 config surface. The Mamba2PredictorConfig
        will ignore these fields when ``backend != 'ssd_reference'``.
        """
        return Mamba2PredictorConfig(
            latent_dim=self.latent_dim,
            ego_motion_dim=self.ego_motion_dim,
            d_model=self.d_model,
            d_state=self.d_state,
            expand=self.expand,
            d_conv=self.d_conv,
            backend=self.backend,  # type: ignore[arg-type]
            ssd_nheads=self.ssd_nheads,
            ssd_headdim=self.ssd_headdim,
            stateful=self.stateful,
            identity_predictor=self.identity_predictor,
        )


class Z7Mamba2PredictiveCodingSubstrate(nn.Module):
    """Z7-Mamba-2 substrate: Mamba2Predictor recurrence + Z6-compatible RGB decoder.

    This binds the Z7-Mamba-2 distinguishing recurrent predictor to byte-
    exportable latent, residual, ego-motion, and decoder streams. The
    full forward path autoregresses through the Mamba-2 selective
    state-space across the 600-pair sequence, then renders RGB via the
    Z6-compatible PixelShuffle decoder.

    Sister of ``Z7GruPredictiveCodingSubstrate`` with the predictor
    primitive swapped from GRUCell to Mamba2Predictor; everything else
    (decoder / context conditioner / latent init / residuals / ego_motion
    buffer / autoregressive replay) follows the canonical sister pattern
    for clean paired-comparison.

    Per the design memo §7 architectural specification:

        z_0 = latent_init  (trainable; shape (latent_dim,))
        for t in range(num_pairs):
            ego_t = ego_motion_buffer[t]  (shape (ego_motion_dim,))
            pred_t = Mamba2Predictor(z_{t-1}, ego_t)
                     (selective state-space; hidden state h_{t-1} -> h_t)
            z_t = pred_t + residuals[t]  (residual learning; shape (latent_dim,))
        rgb_0, rgb_1 = Z6Decoder(z_seq)
                       (PixelShuffle dual-RGB head; shape (B, 3, 384, 512) each)

    The recurrent state is internally maintained by Mamba2Predictor; when
    ``stateful=True``, state persists across the 600-pair sequence (Wyner-Ziv
    implicit side-info channel pattern per Catalog #311 Ballard verbatim);
    when ``stateful=False``, state resets every pair (ablation control).
    """

    def __init__(self, config: Z7Mamba2PredictiveCodingConfig) -> None:
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
        if config.beta_ib < 0:
            raise ValueError("beta_ib must be non-negative")
        context_mode = normalize_context_conditioning_mode(
            config.context_conditioning_mode
        )
        if config.context_affine_strength < 0:
            raise ValueError("context_affine_strength must be non-negative")

        self.config = config
        self.context_conditioning_mode = context_mode

        # Predictor: Mamba2Predictor (replaces GRU primitive in sister).
        self.predictor = Mamba2Predictor(config.to_mamba2_predictor_config())

        # Optional context conditioner (sister-canonical).
        self.context_conditioner: LatentAffineContextConditioner | None = None
        if context_mode == "latent_affine":
            # Adapt to Z7-LSTM/GRU sister config shape for canonical reuse.
            from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture import (  # noqa: E501
                Z7GruPredictiveCodingConfig,
            )
            adapter_cfg = Z7GruPredictiveCodingConfig(
                latent_dim=config.latent_dim,
                ego_motion_dim=config.ego_motion_dim,
                num_pairs=config.num_pairs,
                context_conditioning_mode="latent_affine",
                context_affine_strength=config.context_affine_strength,
            )
            self.context_conditioner = LatentAffineContextConditioner(adapter_cfg)

        # Decoder: Z6-compatible (canonical sister).
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

        # Trainable latent init + per-pair residuals (sister-canonical).
        self.latent_init = nn.Parameter(
            torch.randn(config.latent_dim) * float(config.latent_init_std)
        )
        self.residuals = nn.Parameter(
            torch.zeros(config.num_pairs, config.latent_dim)
        )
        # Ego-motion side-info buffer (filled by trainer from real video).
        self.register_buffer(
            "ego_motion_buffer",
            torch.zeros(config.num_pairs, config.ego_motion_dim),
            persistent=True,
        )

    def replay_latents_and_contexts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregress through Mamba-2 + residual stream; return (latents, pre-residual contexts)."""
        z = self.latent_init.view(1, self.config.latent_dim)
        self.predictor.reset_state(1, device=z.device)
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
        """Replay only the latent sequence (no pre-residual contexts)."""
        latents, _contexts = self.replay_latents_and_contexts()
        return latents

    def condition_latents(
        self,
        latents: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Apply configured context-conditioning branch before decoding."""
        if self.context_conditioner is None:
            return latents
        return self.context_conditioner(latents, contexts)

    def reconstruct_all_pairs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay full sequence + decode every pair.

        Returns:
            ``(rgb_0, rgb_1, latents)`` where rgb_{0,1} are
            ``(num_pairs, 3, H, W)`` unit-domain RGB and latents are
            ``(num_pairs, latent_dim)``.
        """
        latents, contexts = self.replay_latents_and_contexts()
        decoder_latents = self.condition_latents(latents, contexts)
        rgb_0, rgb_1 = self.decoder(decoder_latents)
        return rgb_0, rgb_1, latents

    def reconstruct_pair(
        self,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode selected pair indices after deterministic sequence replay.

        Per Catalog #218 mini-batch reconstruct (defense against D4 T4
        OOM bug class): caller selects specific pair indices to bound
        peak activation memory.
        """
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
        """Return metadata required by the Z7MCM2 inflate runtime."""
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
            "mamba2_d_model": int(self.config.d_model),
            "mamba2_d_state": int(self.config.d_state),
            "mamba2_expand": int(self.config.expand),
            "mamba2_d_conv": int(self.config.d_conv),
            "mamba2_backend_active": str(self.predictor.backend_active),
        }

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_breakdown(self) -> dict[str, int]:
        """Return decoder/predictor/latent/residual parameter counts."""
        ctx_params = (
            0 if self.context_conditioner is None
            else sum(
                p.numel()
                for p in self.context_conditioner.parameters()
                if p.requires_grad
            )
        )
        return {
            "decoder": self.decoder.num_parameters(),
            "predictor": self.predictor.num_parameters(),
            "context_conditioner": ctx_params,
            "latent_init": self.latent_init.numel(),
            "residuals": self.residuals.numel(),
            "total": self.num_parameters(),
        }
