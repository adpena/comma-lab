# SPDX-License-Identifier: MIT
"""C1 substrate architecture -- world-model + foveated decoder.

Per the C1 long-term campaign ledger
``.omx/research/campaign_c1_world_model_foveation_20260514.md`` the
substrate composes three orthogonal building blocks:

1. ``WorldModelModule`` -- latent dynamics mode (identity/no-world-model by
   default on executable dispatch surfaces after the 2026-05-14 probe verdict;
   GRU/LSTM/Transformer remain explicit opt-ins) that propagates a compact
   latent state ``z_t`` across the temporal axis.

2. ``FoveatedDecoderModule`` -- per-frame decoder ``D(z_t)`` that emits
   full ``(3, 384, 512)`` RGB but with per-pixel bit allocation modulated
   by a foveation map ``M_t``. The center 96x128 region (NEAR-FOV) carries
   full-detail bits; periphery is heavily quantized. Per Atick-Redlich
   1990 (Neural Computation 2), foveation matched to ego-motion trajectory
   gives 2-10x bit savings on stationary-ergodic driving.

3. ``FoveationMapModule`` -- emits the foveation map ``M_t`` per frame from
   the ego-motion-conditioned latent. The map is differentiable so it
   trains end-to-end with the world-model.

This module is INTENTIONALLY a scaffold (HNeRV parity discipline lesson
L12: single-LOC-per-LOC review discipline). The forward pass is a small
torch.nn graph; the real complexity lives in the score-aware Lagrangian
and the multi-stage training schedule (gated by Phase 3 council approval).

Catalog #124 archive-grammar 8 fields are declared inline in __init__.py at
module level so the AST walker observes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

NUM_PAIRS: int = 600
"""Contest pair count (frame_0 + frame_1 per pair = 1200 frames total)."""

N_FRAMES: int = 1200
"""Total contest frame count (= 2 * NUM_PAIRS)."""

EVAL_HW: tuple[int, int] = (384, 512)
"""Scorer-resolution (height, width). The C1 decoder emits at this resolution."""

DECODER_LATENT_DIM: int = 64
"""Default decoder latent dimensionality (the z_t in the world model).

Small enough that the GRU+decoder param budget stays under 80 KB FP4; large
enough to capture inter-frame structure. Council can override via config.
"""

# Archive byte targets (substrate-engineering scope; the C1 archive carries
# the full end-to-end renderer: world-model + decoder + z_init + foveation
# meta + residual surprise).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 100_000
"""Lower bound on packed archive bytes; tight residual + small world-model."""

TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 180_000
"""Upper bound on packed archive bytes; loose residual + larger world-model."""

PER_FRAME_RESIDUAL_BYTES_TARGET: int = 50
"""Target bytes per-frame for residual surprise blob.

Per the campaign ledger §3.2 the predictive-coding residual asymptotes to
<100 bytes/frame for stationary-ergodic driving. ~50 B/frame * 1200 frames
= 60 KB residual blob; combined with ~50 KB decoder + ~10 KB world-model
+ ~16 KB z_init + ~1 KB foveation meta = ~137 KB total.
"""

FOVEATION_BIT_ATTENUATION_DEFAULT: float = 0.5
"""Default foveation attenuation factor.

Per-pixel bit cost := base_bit_cost * (1 + foveation_attenuation * (1 - M_t(x, y))).
At attenuation=0.5, periphery pixels (M=0) cost 50% MORE bits than center pixels
(M=1). Lowering attenuation toward 0.0 -> uniform bit allocation (no foveation
savings). Raising attenuation toward 1.0 -> aggressive periphery quantization.
"""


class WorldModelRecurrenceMode(Enum):
    """The world-model recurrence design tension (probe-disambiguator hook).

    Per the design-tension memo
    ``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md``
    we ship BOTH defensible interpretations and let the math arbitrate via
    the probe at ``tools/probe_c1_world_model_vs_independent_frames_disambiguator.py``.
    """

    GRU = "gru"
    """1-layer GRU; ~12K params at latent_dim=64; cheapest forward.

    Hafner et al. (RSSM/Dreamer) use GRU as the deterministic recurrent
    state predictor. Best when latent dynamics are highly periodic
    (driving = forward motion + small steering corrections)."""

    LSTM = "lstm"
    """1-layer LSTM; ~16K params at latent_dim=64; better long-range memory.

    Ha & Schmidhuber 2018 used LSTM in the original world-model paper.
    Pays a small param-budget penalty for explicit gating, but the cell
    state can carry slow-varying context (e.g. ambient lighting, lane
    layout) across the full 1200-frame sequence."""

    TRANSFORMER = "transformer"
    """1-layer 2-head transformer; ~25K params; best for non-local context.

    Most expensive forward (O(T^2) attention) but captures long-range
    dependencies the GRU/LSTM may miss. Council expectation: defer until
    GRU/LSTM anchors land, then probe whether attention helps."""

    IDENTITY_NO_WORLD_MODEL = "identity_no_world_model"
    """Zero-parameter identity/no-world-model mode.

    Added after the 2026-05-14 adversarial review and fair C1 probe v2
    falsified GRU/LSTM recurrence as the default campaign action for the C1
    archive class. This mode repeats ``z_init`` for every frame, preserving an
    executable no-world-model/foveation baseline while keeping GRU/LSTM/
    Transformer available only as explicit opt-ins.
    """


class FoveationStrategy(Enum):
    """The foveation-strategy design tension (probe-disambiguator hook).

    Probe: ``tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py``.
    """

    UNIFORM = "uniform"
    """No foveation; uniform bit allocation. Control / ablation baseline."""

    EGO_MOTION_RADIAL = "ego_motion_radial"
    """Radial foveation centered on the ego-motion vanishing point.

    Per Atick-Redlich 1990 the cortical magnification factor for driving is
    well-approximated by a 2D Gaussian centered on the forward-motion FOV
    center. Computed from PoseNet-predicted ego-motion direction at
    inflate time (no learned params)."""

    LEARNED_PER_PIXEL = "learned_per_pixel"
    """Learned per-pixel attention map from world-model latent.

    More expressive but adds ~3 KB foveation_meta bytes per archive.
    Council expectation: probe whether the learned map outperforms the
    geometric ego-motion radial map empirically."""


@dataclass(frozen=True)
class WorldModelConfig:
    """Static design-time parameters for the world-model recurrence.

    Args:
        recurrence_mode: identity-no-world-model / GRU / LSTM / TRANSFORMER
            (probe-disambiguator hook).
        latent_dim: per-step latent z dimensionality.
        hidden_dim: GRU/LSTM cell hidden state size.
        transformer_heads: only used when recurrence_mode=TRANSFORMER.
        transformer_seq_len: full sequence length (not chunked).
        route_compute_to_z5: when True, the substrate routes recurrent
            compute through Z5's ``HierarchicalPredictor`` (Z5's canonical
            Hafner DreamerV3 + Rao-Ballard predictive-coding pattern with
            per-step ``identity_predictor`` mode-arbitration). Per the
            2026-05-14 grand-council reconvening Decision 1 (UNANIMOUS β
            11/11) — Option β routes C1 compute INTO Z5 wholesale so the
            dispositive Hafner DreamerV3 vs identity test happens as a
            side-effect of the already-planned Z5 dispatch.

            This flag is OPT-IN (default False): the original
            ``WorldModelModule`` (Decision δ REJECTED 11/11 — DO NOT
            REMOVE) remains the C1 v1 forward path. Setting this flag to
            True at trainer/dispatch time selects the Z5-routed forward
            path WITHOUT modifying the substrate scaffold; the council's
            "PAUSE WorldModelModule removal" verdict is preserved.
        z5_predictor_hidden_dim: hidden_dim passed to Z5's
            ``HierarchicalPredictor`` when ``route_compute_to_z5=True``.
            Defaults to ``hidden_dim`` so the routing is parameter-budget
            comparable to the native GRU/LSTM cell.
        z5_predictor_num_layers: predictor depth (2 or 3) per Z5's design.
        z5_predictor_ego_motion_dim: ego-motion projection dim for Z5's
            predictor. Default 8 matches Z5's
            ``PredictiveCodingConfig.predictor_ego_motion_dim`` default.
        z5_identity_predictor_mode: when True, Z5's predictor runs in
            ``identity_predictor`` regime (predict z_t = z_{t-1}). This is
            the dispositive ablation per the council's Decision 1 β:
            ``predictive_world_model`` (False) vs ``identity_predictor``
            (True). Default False (full Hafner DreamerV3 pattern).
    """

    recurrence_mode: WorldModelRecurrenceMode = WorldModelRecurrenceMode.GRU
    latent_dim: int = DECODER_LATENT_DIM
    hidden_dim: int = DECODER_LATENT_DIM  # GRU/LSTM hidden state size
    transformer_heads: int = 2  # only used when recurrence_mode=TRANSFORMER
    transformer_seq_len: int = N_FRAMES  # full sequence; not chunked
    # ── 2026-05-14 council Decision 1 (UNANIMOUS β 11/11): C1 -> Z5 routing ──
    route_compute_to_z5: bool = False
    z5_predictor_hidden_dim: int | None = None
    z5_predictor_num_layers: int = 2
    z5_predictor_ego_motion_dim: int = 8
    z5_identity_predictor_mode: bool = False


@dataclass(frozen=True)
class WorldModelFoveationConfig:
    """Static design-time parameters for the full C1 substrate.

    Args:
        world_model_cfg: WorldModelConfig sub-config.
        foveation_strategy: UNIFORM / EGO_MOTION_RADIAL / LEARNED_PER_PIXEL.
        output_height: scorer-resolution height (default 384).
        output_width: scorer-resolution width (default 512).
        num_pairs: contest pair count (default 600).
        residual_loss_weight: scalar weight on residual L^2 penalty.
        foveation_loss_weight: scalar L^1 penalty on foveation-map sparsity
            (encourages the map to concentrate bits in fewer regions,
            improving byte savings).
        decoder_channels: per-stage decoder channel widths. Default
            (32, 16, 8) gives a tiny FP4-friendly decoder.
        archive_byte_target: target archive packed bytes; informs the
            rate-term Lagrangian budget. Default 140 KB.
    """

    world_model_cfg: WorldModelConfig = WorldModelConfig()
    foveation_strategy: FoveationStrategy = FoveationStrategy.EGO_MOTION_RADIAL
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    num_pairs: int = NUM_PAIRS
    residual_loss_weight: float = 0.1
    foveation_loss_weight: float = 0.01
    decoder_channels: tuple[int, int, int] = (32, 16, 8)
    archive_byte_target: int = 140_000

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)

    @property
    def n_frames(self) -> int:
        return 2 * self.num_pairs


class WorldModelModule(nn.Module):
    """Recurrent latent dynamics: z_t = WorldModel(z_{t-1}, action_t).

    For C1 V1 the "action" is a zero placeholder -- the world-model is
    autonomous (no explicit action signal). In future versions ego-motion
    estimates from PoseNet could be fed as actions; for now we keep the
    scaffold simple.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.recurrence_mode == WorldModelRecurrenceMode.IDENTITY_NO_WORLD_MODEL:
            self.cell = nn.Identity()
        elif cfg.recurrence_mode == WorldModelRecurrenceMode.GRU:
            self.cell: nn.Module = nn.GRUCell(
                input_size=cfg.latent_dim,
                hidden_size=cfg.hidden_dim,
            )
        elif cfg.recurrence_mode == WorldModelRecurrenceMode.LSTM:
            self.cell = nn.LSTMCell(
                input_size=cfg.latent_dim,
                hidden_size=cfg.hidden_dim,
            )
        else:  # TRANSFORMER
            # 1-layer 2-head transformer encoder layer; we feed the sequence
            # all at once in the forward pass (not per-step).
            self.cell = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.transformer_heads,
                dim_feedforward=cfg.hidden_dim * 2,
                batch_first=True,
            )

    def unroll(
        self,
        z_init: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Unroll the world-model for n_steps starting from z_init.

        Args:
            z_init: ``(latent_dim,)`` or ``(1, latent_dim)`` initial latent.
            n_steps: number of recurrent steps to unroll.

        Returns:
            ``(n_steps, latent_dim)`` tensor of latent states z_0..z_{n_steps-1}.
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(0)
        if z_init.shape[-1] != self.cfg.latent_dim:
            raise ValueError(
                f"z_init last dim {z_init.shape[-1]} != latent_dim "
                f"{self.cfg.latent_dim}"
            )

        if self.cfg.recurrence_mode == WorldModelRecurrenceMode.IDENTITY_NO_WORLD_MODEL:
            return z_init.expand(n_steps, -1).contiguous()

        if self.cfg.recurrence_mode == WorldModelRecurrenceMode.TRANSFORMER:
            # Tile z_init across the sequence and let the encoder process it.
            # This gives a sequence-shaped (1, n_steps, latent_dim) tensor.
            seq = z_init.unsqueeze(1).expand(-1, n_steps, -1).contiguous()
            out = self.cell(seq)  # (1, n_steps, latent_dim)
            return out.squeeze(0)

        # GRU / LSTM recurrent unroll
        z_t = z_init  # (1, latent_dim)
        outputs: list[torch.Tensor] = []
        cell_state: torch.Tensor | None = None
        zero_action = torch.zeros_like(z_t)
        for _ in range(n_steps):
            if self.cfg.recurrence_mode == WorldModelRecurrenceMode.GRU:
                # GRUCell input is "input" (action) and state z_t
                z_t = self.cell(zero_action, z_t)
            else:  # LSTM
                if cell_state is None:
                    cell_state = torch.zeros_like(z_t)
                z_t, cell_state = self.cell(zero_action, (z_t, cell_state))
            outputs.append(z_t)
        return torch.cat(outputs, dim=0)  # (n_steps, latent_dim)


class Z5RoutedWorldModel(nn.Module):
    """C1 -> Z5 routing adapter: delegate recurrent compute to Z5's
    ``HierarchicalPredictor``.

    Per the 2026-05-14 grand-council reconvening Decision 1 (UNANIMOUS β
    11/11) — Option β routes C1 compute INTO Z5 wholesale rather than
    duplicating world-model plumbing in two places (Hotz: "stop building
    plumbing in two places"; Time-Traveler peer: "Z5 IS C1 modulo scale").

    This adapter exposes the SAME ``unroll(z_init, n_steps) -> (n_steps,
    latent_dim)`` interface as :class:`WorldModelModule`, so it is a
    drop-in replacement at substrate-build time when
    ``WorldModelConfig.route_compute_to_z5=True``. Internally it invokes
    Z5's :class:`HierarchicalPredictor` autoregressively, with ego-motion
    set to a zero placeholder (matching the C1 v1 "no explicit action
    signal" semantic) so the routing is observation-equivalent to C1's
    GRU/LSTM forward pass at the top-level interface contract.

    The wrapper preserves :class:`WorldModelModule` (Decision δ REJECTED
    11/11 — DO NOT REMOVE the existing class). Selecting the Z5-routed
    path is opt-in via :attr:`WorldModelConfig.route_compute_to_z5`.

    Mode arbitration:
        * ``identity_predictor=False``: full Z5 Hafner DreamerV3 +
          Rao-Ballard predictive-coding pattern (z_t = predictor(z_{t-1},
          ego_motion_t)). This is the ``predictive_world_model`` regime
          per the council ledger.
        * ``identity_predictor=True``: Z5's identity branch (z_t =
          z_{t-1}). This is the ``identity_predictor`` regime — the
          dispositive ablation that self-arbitrates the world-model
          class hypothesis at trainer scale.

    Cross-references:
        * :mod:`tac.substrates.z5_predictive_coding_world_model.architecture`
        * Council ledger
          ``.omx/research/grand_council_c1_post_probe_v2_reconvene_20260514.md``
        * Memory file
          ``feedback_c1_council_reconvene_post_probe_v2_landed_20260514.md``

    Args:
        cfg: parent :class:`WorldModelConfig`. Reads ``latent_dim``,
            ``z5_predictor_hidden_dim`` (defaults to ``hidden_dim``),
            ``z5_predictor_num_layers``, ``z5_predictor_ego_motion_dim``,
            and ``z5_identity_predictor_mode``.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        # Lazy import per CLAUDE.md "Beauty, simplicity, and developer
        # experience": Z5's architecture module is heavy (encoder + decoder
        # + predictor); only import when routing is enabled.
        from tac.substrates.z5_predictive_coding_world_model.architecture import (
            HierarchicalPredictor,
        )

        self.cfg = cfg
        hidden_dim = (
            cfg.z5_predictor_hidden_dim
            if cfg.z5_predictor_hidden_dim is not None
            else cfg.hidden_dim
        )
        self._z5_predictor_hidden_dim = hidden_dim
        self._ego_motion_dim = cfg.z5_predictor_ego_motion_dim
        self._identity_predictor_mode = cfg.z5_identity_predictor_mode
        self.predictor = HierarchicalPredictor(
            latent_dim=cfg.latent_dim,
            hidden_dim=hidden_dim,
            num_layers=cfg.z5_predictor_num_layers,
            ego_motion_dim=cfg.z5_predictor_ego_motion_dim,
            identity_predictor=cfg.z5_identity_predictor_mode,
        )

    def unroll(
        self,
        z_init: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Unroll Z5's predictor for ``n_steps`` from ``z_init``.

        Matches :meth:`WorldModelModule.unroll` signature exactly so the
        substrate forward pass is interface-equivalent at
        ``WorldModelConfig.route_compute_to_z5=True``.

        Args:
            z_init: ``(latent_dim,)`` or ``(1, latent_dim)`` initial latent.
            n_steps: number of recurrent steps to unroll.

        Returns:
            ``(n_steps, latent_dim)`` tensor of latent states z_0..z_{n_steps-1}.
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(0)
        if z_init.shape[-1] != self.cfg.latent_dim:
            raise ValueError(
                f"z_init last dim {z_init.shape[-1]} != latent_dim "
                f"{self.cfg.latent_dim}"
            )

        # Per the C1 v1 contract (line 196-200 docstring of
        # WorldModelModule.unroll), the "action" is a zero placeholder. We
        # mirror that here so the routing is observation-equivalent at the
        # interface contract: identical I/O shape; the ONLY change is which
        # recurrent kernel runs the per-step prediction.
        zero_action = torch.zeros(
            1, self._ego_motion_dim, device=z_init.device, dtype=z_init.dtype
        )
        z_t = z_init
        outputs: list[torch.Tensor] = []
        for _ in range(n_steps):
            z_t = self.predictor(z_t, zero_action)
            outputs.append(z_t)
        return torch.cat(outputs, dim=0)

    @property
    def identity_predictor_mode(self) -> bool:
        """Read-only accessor for the routed predictor's identity-mode flag."""
        return self._identity_predictor_mode


class FoveatedDecoderModule(nn.Module):
    """Per-frame decoder D(z_t) -> RGB ``(3, H, W)`` with foveation-aware out.

    Architecture is intentionally tiny: latent_dim -> 32 -> 16 -> 8 -> 3 via
    a small stack of conv + upsample blocks. The full 384x512 emission goes
    through a learnable upsample-then-conv ladder; per-pixel bit-cost
    modulation happens at archive build time (the foveation map controls
    per-pixel quantization step in the residual codec).
    """

    def __init__(self, cfg: WorldModelFoveationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c0 = cfg.world_model_cfg.latent_dim
        c1, c2, c3 = cfg.decoder_channels
        # Start at 8x8 and upsample to (output_height, output_width) in 4 steps.
        # 8 -> 16 -> 32 -> 64 -> 384 (with final F.interpolate to exact dims).
        self.linear_to_8x8 = nn.Linear(c0, c0 * 8 * 8)
        self.block1 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.to_rgb = nn.Conv2d(c3, 3, kernel_size=3, padding=1)

    def decode(self, z_t: torch.Tensor) -> torch.Tensor:
        """Decode a single latent ``(latent_dim,)`` or ``(B, latent_dim)``
        to RGB ``(B, 3, H, W)`` in unit range.
        """
        import torch.nn.functional as F

        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        b = z_t.shape[0]
        c0 = self.cfg.world_model_cfg.latent_dim
        x = self.linear_to_8x8(z_t).reshape(b, c0, 8, 8)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.block3(x)
        x = F.interpolate(
            x,
            size=self.cfg.output_hw,
            mode="bilinear",
            align_corners=False,
        )
        rgb = self.to_rgb(x)
        # Map to [0, 1] via sigmoid (unit range; trainer scales to [0, 255]
        # before scorer_pair_components which expects 255-domain).
        rgb = torch.sigmoid(rgb)
        return rgb


class FoveationMapModule(nn.Module):
    """Emit the per-frame foveation map M_t ``(1, H, W)`` in [0, 1].

    Three strategies (probe-disambiguator hook):

    - UNIFORM: returns constant 1.0 (no foveation; control)
    - EGO_MOTION_RADIAL: 2D Gaussian centered on the vanishing point
      (no learned params; computed from ego-motion seed in archive meta)
    - LEARNED_PER_PIXEL: tiny conv head on z_t emits per-pixel attention
    """

    def __init__(self, cfg: WorldModelFoveationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.foveation_strategy == FoveationStrategy.LEARNED_PER_PIXEL:
            c0 = cfg.world_model_cfg.latent_dim
            self.learned_head = nn.Sequential(
                nn.Linear(c0, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
            )
        else:
            self.learned_head = None

    def map(self, z_t: torch.Tensor) -> torch.Tensor:
        """Return ``(1, H, W)`` foveation map in [0, 1].

        Args:
            z_t: ``(latent_dim,)`` or ``(B, latent_dim)`` latent for the
                current frame. UNIFORM/EGO_MOTION_RADIAL ignore this.

        Returns:
            ``(B, 1, H, W)`` foveation map (B=1 if z_t was 1D).
        """
        h, w = self.cfg.output_hw
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        b = z_t.shape[0]
        if self.cfg.foveation_strategy == FoveationStrategy.UNIFORM:
            return torch.ones(b, 1, h, w, device=z_t.device, dtype=z_t.dtype)
        if self.cfg.foveation_strategy == FoveationStrategy.EGO_MOTION_RADIAL:
            # 2D Gaussian centered on (h/2, w/2) -- vanishing point at frame
            # center for forward driving. Future versions could thread the
            # ego-motion seed from archive meta to shift the center.
            yy = torch.arange(h, device=z_t.device, dtype=z_t.dtype)
            xx = torch.arange(w, device=z_t.device, dtype=z_t.dtype)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            cy, cx = h / 2.0, w / 2.0
            sigma = min(h, w) / 4.0
            gauss = torch.exp(
                -((grid_y - cy) ** 2 + (grid_x - cx) ** 2) / (2.0 * sigma ** 2)
            )
            # Tile across batch.
            return gauss.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w).contiguous()
        # LEARNED_PER_PIXEL
        assert self.learned_head is not None
        # Tiny scalar per-frame attention; broadcast to full image. This is
        # the simplest learned-per-pixel scaffold -- a future version could
        # emit a per-pixel logit map, but for L1 we keep it minimal.
        scalar = torch.sigmoid(self.learned_head(z_t))  # (B, 1)
        return scalar.unsqueeze(-1).unsqueeze(-1).expand(b, 1, h, w).contiguous()


class WorldModelFoveationSubstrate(nn.Module):
    """The full C1 substrate: world-model + foveated decoder + foveation map.

    Forward pass:

    1. Unroll the world-model from z_init for n_frames steps -> z_0..z_{T-1}
    2. For each z_t: emit RGB frame via foveated decoder D(z_t)
    3. Emit foveation map M_t per frame from the foveation strategy

    Returns the full ``(T, 3, H, W)`` frame stack. The trainer is responsible
    for staging consecutive frames into pairs ``(rgb_0, rgb_1)`` for the
    score-aware Lagrangian per the standard contest pair contract.
    """

    def __init__(self, cfg: WorldModelFoveationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Per the 2026-05-14 grand-council reconvening Decision 1
        # (UNANIMOUS β 11/11): when ``route_compute_to_z5`` is set on the
        # world_model_cfg, the recurrent kernel is Z5's
        # ``HierarchicalPredictor`` (Hafner DreamerV3 + Rao-Ballard
        # predictive-coding) with self-arbitration of the
        # ``predictive_world_model`` vs ``identity_predictor`` regime.
        # Otherwise, the canonical :class:`WorldModelModule` (Decision δ
        # REJECTED 11/11 — DO NOT REMOVE) provides the C1 v1 GRU/LSTM/
        # Transformer recurrence. Both kernels expose the SAME ``unroll``
        # interface so downstream forward paths are unchanged.
        self.world_model: nn.Module
        if cfg.world_model_cfg.route_compute_to_z5:
            self.world_model = Z5RoutedWorldModel(cfg.world_model_cfg)
        else:
            self.world_model = WorldModelModule(cfg.world_model_cfg)
        self.decoder = FoveatedDecoderModule(cfg)
        self.foveation = FoveationMapModule(cfg)
        # Initial latent z_init -- trainable parameter that the world-model
        # unrolls from. ~16 KB at FP32 / latent_dim=64.
        self.z_init = nn.Parameter(torch.zeros(cfg.world_model_cfg.latent_dim))

    @property
    def routes_compute_to_z5(self) -> bool:
        """Whether this substrate's world-model kernel is the Z5 routing wrapper."""
        return isinstance(self.world_model, Z5RoutedWorldModel)

    def render_all_frames(
        self,
        n_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render the full T-frame stack.

        Args:
            n_frames: how many frames to render. Defaults to cfg.n_frames
                (= 2 * cfg.num_pairs).

        Returns:
            (rgb_stack, foveation_stack):
                rgb_stack:        (T, 3, H, W) in [0, 1] unit range
                foveation_stack:  (T, 1, H, W) foveation maps M_t in [0, 1]
        """
        t = n_frames if n_frames is not None else self.cfg.n_frames
        latents = self.world_model.unroll(self.z_init, t)  # (T, latent_dim)
        rgb = self.decoder.decode(latents)  # (T, 3, H, W)
        fov = self.foveation.map(latents)  # (T, 1, H, W)
        return rgb, fov

    def render_pair(
        self,
        pair_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render a single contest pair ``(frame_0, frame_1)`` at pair_idx.

        Mini-batched alternative to ``render_all_frames`` to stay under T4
        memory pressure per Catalog #218 (no full-N OOM). Internally
        unrolls the world model only enough to reach frame ``2 * pair_idx + 1``
        which is O(pair_idx) work; a future optimization could checkpoint
        intermediate latents to amortize.
        """
        if not (0 <= pair_idx < self.cfg.num_pairs):
            raise ValueError(
                f"pair_idx {pair_idx} outside [0, {self.cfg.num_pairs})"
            )
        n_steps = 2 * pair_idx + 2
        latents = self.world_model.unroll(self.z_init, n_steps)
        z_0 = latents[2 * pair_idx : 2 * pair_idx + 1]
        z_1 = latents[2 * pair_idx + 1 : 2 * pair_idx + 2]
        rgb_0 = self.decoder.decode(z_0)  # (1, 3, H, W)
        rgb_1 = self.decoder.decode(z_1)  # (1, 3, H, W)
        return rgb_0, rgb_1


__all__ = [
    "DECODER_LATENT_DIM",
    "EVAL_HW",
    "FOVEATION_BIT_ATTENUATION_DEFAULT",
    "FoveatedDecoderModule",
    "FoveationMapModule",
    "FoveationStrategy",
    "N_FRAMES",
    "NUM_PAIRS",
    "PER_FRAME_RESIDUAL_BYTES_TARGET",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "WorldModelConfig",
    "WorldModelFoveationConfig",
    "WorldModelFoveationSubstrate",
    "WorldModelModule",
    "WorldModelRecurrenceMode",
    "Z5RoutedWorldModel",
]
