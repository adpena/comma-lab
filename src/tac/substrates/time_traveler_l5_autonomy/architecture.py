# SPDX-License-Identifier: MIT
"""Time-Traveler L5 Autonomy architecture — differentiable world model.

Implements the five first-principles design moves from the
``.omx/research/time_traveler_architecture_reverse_engineered_20260513.md``
synthesis memo:

1. **World model** (Stage 1, ~55-70 KB encoded once): a small coordinate-MLP
   renderer (sub-60K params at FP16 ~ 35 KB) + log-polar foveation grid
   (~2 KB) + ego-pose dynamics prior (~3 KB Markov-1 over SE(3) deltas).
2. **Per-pair side info** (Stage 2, ~25-35 KB at 45 B/pair x 600 pairs):
   SE(3) Lie-algebra pose delta + boundary-only segnet residual + HF
   residual byte-stuffing + prediction-error residual.
3. **Predictive decoding**: per-pair RGB rendered by composing
   ``world_model(pair_index, foveation_coords) + per_pair_residual``.

The architecture is INTENTIONALLY simple. The score gains come from the
score-aware loss (Atick-Redlich cooperative-receiver) and the world-model
prior, not from architectural complexity. Per HNeRV parity discipline L12
(single-LOC-per-LOC review discipline): every line below is reviewable in
30 seconds.

**Catalog #124 archive-grammar 8 fields** declared inline so the AST
walker observes them:

* ``archive_grammar``: monolithic single-file ``0.bin``
* ``parser_section_manifest``: TT5L header + decoder + side-info + AC state + meta
* ``inflate_runtime_loc_budget``: <= 200 LOC substrate-engineering waiver
* ``runtime_dep_closure``: torch + brotli only
* ``export_format``: TT5L
* ``score_aware_loss``: cooperative-receiver + predictive-coding (Atick-Redlich + Rao-Ballard)
* ``bolt_on_loc_budget``: lane_class=substrate_engineering
* ``no_op_detector_planned``: emit/parse roundtrip preserves bytes byte-for-byte
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Scorer resolution (height, width). The world model renders at this resolution."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 per pair)."""

CAMERA_HW: tuple[int, int] = (874, 1164)
"""Contest camera native resolution; inflate.py upsamples renderer output."""

# Archive size budget (per design memo Stage 1+2+3+4 = 95-110 KB).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 95_000
TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 110_000
PER_PAIR_SIDE_INFO_TARGET_BYTES: int = 45
"""Per-pair side info budget: 12 B SE(3) Lie + 18 B seg boundary + 6 B HF + 9 B residual."""


@dataclass(frozen=True)
class TimeTravelerConfig:
    """Static design-time parameters for the Time-Traveler substrate.

    Defaults aim for ~35 KB FP16 decoder + ~25 KB world-model side info +
    ~30 KB per-pair side info = ~90 KB before brotli, with brotli typically
    closing the final ~5-10 KB to land in 95-110 KB.

    Args:
        hidden_dim: Width of the renderer MLP. Default 64; smaller than SIREN's
            default 128 because the Tikhonov prior + world model carry the
            inductive bias.
        num_hidden_layers: Depth of the renderer MLP. Default 4.
        coord_dim: Input coordinate dim per pair (x, y, t) + foveation = 4.
        pose_dim: SE(3) Lie-algebra dim. 6 (3 translation + 3 rotation).
        per_pair_side_info_bytes: Target bytes per pair after int8 quantization.
        foveation_grid_size: log-polar foveation grid resolution (e.g., 16x24
            gives 384 cells; 8-bit quant => 384 B; <=2 KB target).
        first_omega: SIREN-style first-layer omega for the renderer MLP.
        hidden_omega: SIREN-style downstream omega.
        num_pairs: Contest pair count (600).
        output_height, output_width: Renderer eval resolution (384, 512).
        markov_transition_band: Number of past pose deltas the Markov-1 prior
            uses (default 4; transitions encoded as int8 deltas).
        boundary_only_segmap_classes: SegNet 5-class output; we encode
            classes 0/1 (road/lane) as boundary-only deltas via int4 per cell.
    """

    hidden_dim: int = 64
    num_hidden_layers: int = 4
    coord_dim: int = 4
    pose_dim: int = 6
    per_pair_side_info_bytes: int = PER_PAIR_SIDE_INFO_TARGET_BYTES
    foveation_grid_h: int = 16
    foveation_grid_w: int = 24
    first_omega: float = 30.0
    hidden_omega: float = 1.0
    num_pairs: int = NUM_PAIRS
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    markov_transition_band: int = 4
    boundary_only_segmap_classes: tuple[int, ...] = (0, 1)
    coord_feature_freqs: int = 4
    """Positional encoding bands for (x, y) — adds sin/cos basis."""

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {self.hidden_dim}")
        if self.num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive; got {self.num_hidden_layers}"
            )
        if self.coord_dim != 4:
            raise ValueError(f"coord_dim must be 4 (x, y, t, foveation); got {self.coord_dim}")
        if self.pose_dim != 6:
            raise ValueError(f"pose_dim must be 6 (SE(3) Lie algebra); got {self.pose_dim}")
        if self.per_pair_side_info_bytes <= 0 or self.per_pair_side_info_bytes > 256:
            raise ValueError(
                f"per_pair_side_info_bytes must be in (0, 256]; got "
                f"{self.per_pair_side_info_bytes}"
            )
        if self.foveation_grid_h <= 0 or self.foveation_grid_w <= 0:
            raise ValueError("foveation_grid dims must be positive")
        if self.num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive; got {self.num_pairs}")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output dims must be positive")
        if self.first_omega <= 0.0 or self.hidden_omega <= 0.0:
            raise ValueError("omega values must be positive")
        if self.markov_transition_band <= 0:
            raise ValueError("markov_transition_band must be positive")
        if self.coord_feature_freqs <= 0:
            raise ValueError("coord_feature_freqs must be positive")

    def predict_decoder_param_count(self) -> int:
        """Closed-form prediction of renderer-MLP parameter count.

        Useful for budget planning before instantiation. Each Linear layer has
        ``in*out + out`` params; we have ``input -> hidden -> ... -> output``.
        """
        input_dim = self.coord_dim + 2 * self.coord_feature_freqs * 2  # x,y bands
        output_dim = 6  # rgb_0 + rgb_1
        layers: list[tuple[int, int]] = [(input_dim, self.hidden_dim)]
        for _ in range(self.num_hidden_layers - 1):
            layers.append((self.hidden_dim, self.hidden_dim))
        layers.append((self.hidden_dim, output_dim))
        return sum(i * o + o for i, o in layers)


def _siren_init_(linear: nn.Linear, *, is_first: bool, omega: float) -> None:
    """SIREN initialization scheme (Sitzmann et al. NeurIPS 2020).

    First layer: ``U(-1/in, 1/in)``; downstream: ``U(-sqrt(6/in)/omega, +...)``.
    Both rescaled so ``sin(omega * x)`` preserves the distribution.
    """
    fan_in = linear.in_features
    with torch.no_grad():
        bound = 1.0 / fan_in if is_first else math.sqrt(6.0 / fan_in) / omega
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.zero_()


class LogPolarFoveationGrid(nn.Module):
    """Log-polar foveation grid centered on the camera focus-of-expansion.

    The grid is a deterministic spatial weighting map produced from a small
    learned ``(grid_h, grid_w)`` int8-quantized weight table. The map is
    bilinear-upsampled to ``EVAL_HW`` at training time.

    Per the design memo: 2 KB encodes the spatial weighting map; gives 5-10x
    effective resolution gain on score-relevant regions (vehicles, lane
    markers near vanishing point) vs uniform.

    The grid is APPLIED to the renderer's output as a learned per-pixel gain
    modulating the predictive decoder. The grid itself is part of the world
    model (encoded once in Stage 1 of the archive).
    """

    def __init__(self, cfg: TimeTravelerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Learned grid weights in [0, 2] (1.0 = neutral). int8 quantize at archive time.
        self.grid_weights = nn.Parameter(
            torch.ones(cfg.foveation_grid_h, cfg.foveation_grid_w)
            + 0.02 * torch.randn(cfg.foveation_grid_h, cfg.foveation_grid_w)
        )

    def forward(self) -> torch.Tensor:
        """Return the full-resolution foveation weighting map ``(H, W)``."""
        import torch.nn.functional as F

        # Add batch and channel dims for interpolate.
        grid = self.grid_weights.unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(
            grid,
            size=(self.cfg.output_height, self.cfg.output_width),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.squeeze(0).squeeze(0).clamp(0.1, 4.0)

    def estimate_int8_bytes(self) -> int:
        """Closed-form: ``grid_h * grid_w`` int8 + 8 B header."""
        return self.cfg.foveation_grid_h * self.cfg.foveation_grid_w + 8


class EgoMotionDynamicsPrior(nn.Module):
    """Markov-1 ego-pose dynamics prior in SE(3) Lie algebra.

    Learns a small ``(pose_dim, pose_dim)`` transition matrix and a per-band
    delta-coding lookup. Encodes the prior that highway dashcam ego-motion is
    locally near-linear (constant velocity + small turn rate).

    Per the design memo: 3 KB encodes the dynamics prior. The actual SE(3)
    pose at each pair is stored in per-pair side info (12 B/pair); this module
    holds the learned PREDICTOR ``pose_{t+1} = A @ pose_t + b`` so the side-info
    only needs to encode the residual.
    """

    def __init__(self, cfg: TimeTravelerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Identity + small perturbation. Linear AR-1 over SE(3) Lie-algebra.
        eye = torch.eye(cfg.pose_dim)
        self.transition = nn.Parameter(eye + 0.01 * torch.randn(cfg.pose_dim, cfg.pose_dim))
        self.bias = nn.Parameter(torch.zeros(cfg.pose_dim))

    def predict_next(self, prev_pose: torch.Tensor) -> torch.Tensor:
        """Predict ``pose_{t+1}`` from ``pose_t`` via Markov-1 AR(1)."""
        return prev_pose @ self.transition.T + self.bias

    def estimate_fp16_bytes(self) -> int:
        """Closed-form: ``pose_dim**2 + pose_dim`` FP16 + 8 B header."""
        return 2 * (self.cfg.pose_dim**2 + self.cfg.pose_dim) + 8


class PredictiveRenderer(nn.Module):
    """Sub-60K-param SIREN-style MLP renderer for the world model.

    Inputs: per-pair coordinate ``(x, y, t, foveation)`` -> RGB pair (6 channels).
    The renderer is the "differentiable physics op + predictive decoder"
    component of Stage 1 (~35 KB at FP16, ~60K params target).

    Note: at NUM_PAIRS=600 with ~32K params, scoring at 384x512 requires
    600 * 384 * 512 = ~118M coord evaluations. Caller should chunk by pair.
    """

    def __init__(self, cfg: TimeTravelerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Positional encoding adds 2 * freqs * 2 dims (sin/cos on x, y).
        input_dim = cfg.coord_dim + 2 * cfg.coord_feature_freqs * 2
        layers: list[nn.Module] = []
        in_features = input_dim
        for layer_idx in range(cfg.num_hidden_layers):
            lin = nn.Linear(in_features, cfg.hidden_dim)
            omega = cfg.first_omega if layer_idx == 0 else cfg.hidden_omega
            _siren_init_(lin, is_first=(layer_idx == 0), omega=omega)
            layers.append(lin)
            in_features = cfg.hidden_dim
        output_layer = nn.Linear(in_features, 6)
        _siren_init_(output_layer, is_first=False, omega=cfg.hidden_omega)
        # Initialize output bias to mid-gray (0.5 in [0,1] space = 127.5/255).
        with torch.no_grad():
            output_layer.bias.fill_(0.5)
        self.hidden = nn.ModuleList(layers)
        self.output_layer = output_layer

    def _positional_encode(self, xy: torch.Tensor) -> torch.Tensor:
        """Append sin/cos positional encoding to ``(x, y)`` columns of input."""
        # xy: (..., 2)
        freqs = torch.tensor(
            [2.0**k * math.pi for k in range(self.cfg.coord_feature_freqs)],
            device=xy.device,
            dtype=xy.dtype,
        )
        scaled = xy.unsqueeze(-1) * freqs  # (..., 2, freqs)
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return encoded.flatten(-2)  # (..., 2 * freqs * 2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Render RGB pair from coordinates.

        Args:
            coords: ``(..., 4)`` with columns (x, y, t, foveation).

        Returns:
            ``(..., 6)`` RGB pair output in ``[0, 1]`` after sigmoid.
        """
        xy = coords[..., :2]
        pe = self._positional_encode(xy)
        x = torch.cat([coords, pe], dim=-1)
        omega_schedule = [self.cfg.first_omega] + [self.cfg.hidden_omega] * (
            self.cfg.num_hidden_layers - 1
        )
        for layer, omega in zip(self.hidden, omega_schedule, strict=True):
            x = torch.sin(omega * layer(x))
        return torch.sigmoid(self.output_layer(x))


class Z5RoutedLatentPredictor(nn.Module):
    """Time-Traveler -> Z5 routing adapter for per-pair latent prediction.

    Per the 2026-05-14 grand-council reconvening Decision 1 (UNANIMOUS β
    11/11), Time-Traveler is a sister-substrate of Z5
    (``z5_predictive_coding_world_model``); the sister-routing pattern
    keeps the latent-prediction primitive consistent across the two L5
    autonomy substrates so cathedral-autopilot composition logic
    (Catalog #227 ``adjust_predicted_delta_for_composition_alpha``) sees
    the same predictor kernel under stacking.

    This adapter wraps Z5's :class:`HierarchicalPredictor` for use by
    Time-Traveler's per-pair latent recurrence WHEN a future revision of
    :class:`TimeTravelerSubstrate` opts into the routed path. The current
    Time-Traveler v1 substrate does NOT use this adapter (the canonical
    SIREN renderer + Markov-1 ego-pose dynamics + log-polar foveation
    grid composition is preserved per CLAUDE.md "Forbidden premature
    KILL"); the adapter is provided as a NEUTRAL routing primitive so a
    future training run can compare the SIREN-rendered baseline against
    a Z5-routed predicted-residual variant without duplicating Z5's
    Hafner DreamerV3 + Rao-Ballard predictive-coding plumbing inside
    this module.

    Mode arbitration matches Z5 directly:
        * ``identity_predictor=False``: Z5's full predictive-coding kernel
          (z_t = predictor(z_{t-1}, ego_motion_t)).
        * ``identity_predictor=True``: identity (z_t = z_{t-1}).

    Cross-references:
        * :mod:`tac.substrates.z5_predictive_coding_world_model.architecture`
        * :class:`tac.substrates.c1_world_model_foveation.architecture.Z5RoutedWorldModel`
          (sister adapter on the C1 side)
        * Council ledger
          ``.omx/research/grand_council_c1_post_probe_v2_reconvene_20260514.md``

    Args:
        latent_dim: per-pair latent dimensionality.
        hidden_dim: predictor hidden dim (passed to Z5's HierarchicalPredictor).
        num_layers: predictor depth (2 or 3 per Z5's design).
        ego_motion_dim: ego-motion projection dim. Time-Traveler's pose_dim
            is 6 (SE(3) Lie); ego-motion conditioning here is a learned
            small projection, default 8.
        identity_predictor: when True, predictor is identity (Z5
            ablation control).
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        ego_motion_dim: int = 8,
        identity_predictor: bool = False,
    ) -> None:
        super().__init__()
        # Lazy import per CLAUDE.md "Beauty, simplicity, and developer
        # experience" — Z5's architecture is heavy, only import when used.
        from tac.substrates.z5_predictive_coding_world_model.architecture import (
            HierarchicalPredictor,
        )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ego_motion_dim = ego_motion_dim
        self._identity_predictor_mode = identity_predictor
        self.predictor = HierarchicalPredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            ego_motion_dim=ego_motion_dim,
            identity_predictor=identity_predictor,
        )

    def predict_next(
        self,
        z_prev: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next latent given previous latent + ego-motion.

        Args:
            z_prev: ``(B, latent_dim)``.
            ego_motion: ``(B, ego_motion_dim)``.

        Returns:
            ``(B, latent_dim)`` predicted next latent.
        """
        if z_prev.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z_prev last dim {z_prev.shape[-1]} != latent_dim {self.latent_dim}"
            )
        if ego_motion.shape[-1] != self.ego_motion_dim:
            raise ValueError(
                f"ego_motion last dim {ego_motion.shape[-1]} != "
                f"ego_motion_dim {self.ego_motion_dim}"
            )
        return self.predictor(z_prev, ego_motion)

    @property
    def identity_predictor_mode(self) -> bool:
        """Read-only accessor for the routed predictor's identity-mode flag."""
        return self._identity_predictor_mode


class TimeTravelerSubstrate(nn.Module):
    """Composite world-model substrate.

    Composes the four world-model components:

    * ``PredictiveRenderer`` — small MLP
    * ``LogPolarFoveationGrid`` — spatial weighting map
    * ``EgoMotionDynamicsPrior`` — Markov-1 SE(3) AR(1)
    * Per-pair Lie-algebra pose codes (the side info; one row per pair)

    ``render_pair(pair_index)`` returns the predicted RGB pair without
    applying per-pair residuals (that's added at archive-build / inflate
    time by ``TimeTravelerArchive``).
    """

    def __init__(self, cfg: TimeTravelerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.renderer = PredictiveRenderer(cfg)
        self.foveation = LogPolarFoveationGrid(cfg)
        self.dynamics = EgoMotionDynamicsPrior(cfg)
        # Per-pair SE(3) Lie-algebra code, initialized small (small motion prior).
        self.pose_codes = nn.Parameter(
            0.01 * torch.randn(cfg.num_pairs, cfg.pose_dim)
        )

    def _build_coord_grid(self, device: torch.device) -> torch.Tensor:
        """Return ``(H * W, 2)`` grid of (x, y) in [-1, 1]."""
        ys = torch.linspace(-1.0, 1.0, self.cfg.output_height, device=device)
        xs = torch.linspace(-1.0, 1.0, self.cfg.output_width, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    def _motion_conditioned_coord_grid(
        self,
        *,
        pair_idx: int,
        coord_grid_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a small SE(3)-derived ego-motion warp to renderer coordinates.

        TT5L charges per-pair pose codes and a Markov dynamics prior. The
        renderer must therefore consume both, or the bytes are a false
        mechanism. This lightweight warp maps the six Lie coordinates onto a
        bounded 2-D affine perturbation: translation, scale, shear, and yaw.
        """
        pose = self.pose_codes[pair_idx]
        prev_idx = max(0, pair_idx - 1)
        predicted_pose = self.dynamics.predict_next(
            self.pose_codes[prev_idx].unsqueeze(0)
        ).squeeze(0)
        motion = pose + 0.5 * predicted_pose
        translation = 0.05 * torch.tanh(motion[:2])
        scale = 1.0 + 0.02 * torch.tanh(motion[2])
        shear = 0.02 * torch.tanh(motion[3:5])
        yaw = 0.05 * torch.tanh(motion[5])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x = coord_grid_xy[:, 0]
        y = coord_grid_xy[:, 1]
        warped_x = scale * (cos_yaw * x - sin_yaw * y) + shear[0] * y + translation[0]
        warped_y = scale * (sin_yaw * x + cos_yaw * y) + shear[1] * x + translation[1]
        return torch.stack([warped_x, warped_y], dim=-1).clamp(-1.25, 1.25)

    def render_pair(self, pair_idx: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one pair as ``(rgb_0, rgb_1)`` both shaped ``(1, 3, H, W)``.

        Returns RGB in [0, 1] for downstream eval-roundtrip / scorer ingestion.
        """
        idx = int(pair_idx.flatten()[0].item()) if isinstance(pair_idx, torch.Tensor) else int(pair_idx)
        if not 0 <= idx < self.cfg.num_pairs:
            raise IndexError(
                f"pair_idx {idx} out of range [0, {self.cfg.num_pairs})"
            )
        device = self.pose_codes.device
        H, W = self.cfg.output_height, self.cfg.output_width
        coord_grid_xy = self._motion_conditioned_coord_grid(
            pair_idx=idx,
            coord_grid_xy=self._build_coord_grid(device),
        )  # (H*W, 2)
        t = idx / max(1, self.cfg.num_pairs - 1)
        foveation_map = self.foveation()  # (H, W)
        foveation_flat = foveation_map.flatten().unsqueeze(-1)  # (H*W, 1)
        t_col = torch.full(
            (H * W, 1), t, device=device, dtype=coord_grid_xy.dtype
        )
        coords = torch.cat([coord_grid_xy, t_col, foveation_flat], dim=-1)  # (H*W, 4)
        # MLP eval; output (H*W, 6) in [0, 1].
        out6 = self.renderer(coords)
        rgb6 = out6.t().reshape(1, 6, H, W)
        rgb_0 = rgb6[:, :3]
        rgb_1 = rgb6[:, 3:]
        return rgb_0, rgb_1

    def parameter_count(self) -> int:
        """Total trainable parameters across all world-model components."""
        return sum(p.numel() for p in self.parameters())

    def estimate_world_model_bytes(self) -> int:
        """Closed-form: 2 * renderer_params (FP16) + foveation + dynamics + per-pair codes.

        Per-pair pose codes are part of the per-pair side info, not the
        world model, but they are counted here for total-budget sanity.
        """
        renderer_bytes = 2 * sum(p.numel() for p in self.renderer.parameters())  # FP16
        foveation_bytes = self.foveation.estimate_int8_bytes()
        dynamics_bytes = self.dynamics.estimate_fp16_bytes()
        pose_codes_bytes = 2 * self.cfg.num_pairs * self.cfg.pose_dim  # FP16
        return renderer_bytes + foveation_bytes + dynamics_bytes + pose_codes_bytes


__all__ = [
    "CAMERA_HW",
    "EVAL_HW",
    "NUM_PAIRS",
    "PER_PAIR_SIDE_INFO_TARGET_BYTES",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "EgoMotionDynamicsPrior",
    "LogPolarFoveationGrid",
    "PredictiveRenderer",
    "TimeTravelerConfig",
    "TimeTravelerSubstrate",
    "Z5RoutedLatentPredictor",
]
