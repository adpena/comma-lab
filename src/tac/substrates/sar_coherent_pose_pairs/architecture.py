"""SAR coherent pose-pair integration substrate — architecture.

Implements three first-principles design moves from the L2 ledger §2:

1. ``SARCoherentPoseCodec`` — sparse rFFT pose-delta codec exploiting temporal
   coherence across the 600-pair contest stream. Per Cook & Bernfeld 1967
   *Radar Signals* and Carrara, Goodman, Majewski 1995 *Spotlight Synthetic
   Aperture Radar*: phase coherence across pulses → effective aperture √N
   larger; for N=600 pairs that is √600 ≈ 24× coherent SNR gain.

2. ``SARCoherentRenderer`` — sub-50K-param SIREN-style coordinate-MLP that
   consumes (x, y, t, pose_codes_t) and emits an RGB pair (6 channels per
   coordinate). Reuses the canonical SIREN init from Sitzmann et al.
   NeurIPS 2020.

3. ``SARCoherentSubstrate`` — composite trainable module (renderer + pose
   codec + per-pair RGB residual). The per-pair residual is a small int8-
   quantized RGB delta the renderer cannot predict.

**Catalog #124 archive-grammar 8 fields** declared inline so the AST walker
observes them:

* ``archive_grammar``: monolithic single-file ``0.bin``
* ``parser_section_manifest``: SARC header + renderer state_dict + sparse
  rFFT pose codec + per-pair RGB residual + meta JSON
* ``inflate_runtime_loc_budget``: ≤ 200 LOC substrate-engineering waiver
* ``runtime_dep_closure``: torch + numpy + brotli only
* ``export_format``: SARC monolithic ``0.bin``
* ``score_aware_loss``: cooperative-receiver via canonical
  ``score_pair_components``
* ``bolt_on_loc_budget``: ``lane_class=substrate_engineering``
* ``no_op_detector_planned``: emit/parse roundtrip preserves bytes
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Scorer resolution (height, width). The renderer emits at this resolution."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 per pair)."""

CAMERA_HW: tuple[int, int] = (874, 1164)
"""Contest camera native resolution; inflate.py upsamples renderer output."""

POSE_DIM: int = 6
"""SE(3) Lie-algebra pose dimension (3 translation + 3 rotation)."""

# Archive size budget per L2 ledger §2.2 + §2.3 (renderer ~15-25 KB +
# sparse rFFT pose codec ~3-5 KB + per-pair RGB residual ~30-40 KB +
# header/meta ~1 KB → 50-70 KB total before brotli).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 50_000
TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 70_000
PER_PAIR_RESIDUAL_TARGET_BYTES: int = 50
"""Per-pair int8 RGB-residual budget (50 B = ~16 ch × 3 spatial cells)."""


@dataclass(frozen=True)
class SARCoherentConfig:
    """Static design-time parameters for the SAR coherent substrate.

    Defaults aim for ~20 KB FP16 renderer + ~4 KB sparse rFFT pose codec +
    ~30 KB per-pair int8 residual = ~54 KB before brotli, with brotli
    typically closing the final ~5 KB to land in 50-70 KB.

    Args:
        hidden_dim: Width of the renderer MLP. Default 48 (smaller than
            time-traveler 64 because the SAR pose-codec already concentrates
            score-relevant info — the renderer carries less inductive load).
        num_hidden_layers: Depth of the renderer MLP. Default 3.
        coord_dim: (x, y, t) + pose_code_dim per coordinate; declared explicitly.
        pose_dim: SE(3) Lie-algebra dim (6).
        per_pair_residual_bytes: Int8 RGB-residual budget per pair (50 B).
        first_omega: SIREN-style first-layer omega.
        hidden_omega: SIREN-style downstream omega.
        coord_feature_freqs: Positional encoding frequency bands for (x, y).
        sar_topk_keep_fraction: Fraction of rFFT bins retained per pose dim.
            Default 0.10 → keep 10% of (NUM_PAIRS//2 + 1) = 30 bins per dim.
        sar_int16_scale: int16 quantization scale for sparse rFFT coefficients.
        num_pairs: Contest pair count (600).
        output_height, output_width: Renderer eval resolution (384, 512).
    """

    hidden_dim: int = 48
    num_hidden_layers: int = 3
    pose_dim: int = POSE_DIM
    per_pair_residual_bytes: int = PER_PAIR_RESIDUAL_TARGET_BYTES
    first_omega: float = 30.0
    hidden_omega: float = 1.0
    coord_feature_freqs: int = 4
    sar_topk_keep_fraction: float = 0.10
    sar_int16_scale: float = 256.0
    num_pairs: int = NUM_PAIRS
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    pose_code_dim: int = 6
    """Dimensionality of the per-pair pose code that the renderer consumes."""

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {self.hidden_dim}")
        if self.num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive; got {self.num_hidden_layers}"
            )
        if self.pose_dim != POSE_DIM:
            raise ValueError(
                f"pose_dim must be {POSE_DIM} (SE(3) Lie algebra); got {self.pose_dim}"
            )
        if self.per_pair_residual_bytes <= 0 or self.per_pair_residual_bytes > 256:
            raise ValueError(
                f"per_pair_residual_bytes must be in (0, 256]; got "
                f"{self.per_pair_residual_bytes}"
            )
        if self.num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive; got {self.num_pairs}")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output dims must be positive")
        if self.first_omega <= 0.0 or self.hidden_omega <= 0.0:
            raise ValueError("omega values must be positive")
        if self.coord_feature_freqs <= 0:
            raise ValueError("coord_feature_freqs must be positive")
        if not 0.0 < self.sar_topk_keep_fraction <= 1.0:
            raise ValueError(
                f"sar_topk_keep_fraction must be in (0, 1]; "
                f"got {self.sar_topk_keep_fraction}"
            )
        if self.sar_int16_scale <= 0.0:
            raise ValueError("sar_int16_scale must be positive")
        if self.pose_code_dim <= 0:
            raise ValueError("pose_code_dim must be positive")
        if self.pose_code_dim != self.pose_dim:
            raise ValueError(
                f"pose_code_dim must equal pose_dim for SARC archive/inflate "
                f"shape custody; got pose_code_dim={self.pose_code_dim} "
                f"pose_dim={self.pose_dim}"
            )

    def sar_topk(self) -> int:
        """Closed-form: number of rFFT coefficients retained per dim."""
        n_rfft_bins = self.num_pairs // 2 + 1
        return max(1, int(round(self.sar_topk_keep_fraction * n_rfft_bins)))

    def predict_renderer_param_count(self) -> int:
        """Closed-form prediction of renderer-MLP parameter count."""
        # input = (x, y, t) + pos_encoding(x,y) + pose_code = 3 + 2*freqs*2 + pose_code_dim
        input_dim = 3 + 2 * self.coord_feature_freqs * 2 + self.pose_code_dim
        output_dim = 6  # rgb_0 + rgb_1
        layers: list[tuple[int, int]] = [(input_dim, self.hidden_dim)]
        for _ in range(self.num_hidden_layers - 1):
            layers.append((self.hidden_dim, self.hidden_dim))
        layers.append((self.hidden_dim, output_dim))
        return sum(i * o + o for i, o in layers)


def _siren_init_(linear: nn.Linear, *, is_first: bool, omega: float) -> None:
    """SIREN initialization scheme (Sitzmann et al. NeurIPS 2020)."""
    fan_in = linear.in_features
    with torch.no_grad():
        bound = 1.0 / fan_in if is_first else math.sqrt(6.0 / fan_in) / omega
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.zero_()


class SARCoherentPoseCodec(nn.Module):
    """SAR-style coherent pose-pair codec.

    Stores the per-pair pose-delta trajectory as a sparse top-K rFFT over the
    600-pair temporal axis. At inflate time the sparse rFFT is inverse-FFT'd
    to recover per-pair pose deltas, which feed the renderer's pose-code input.

    Per Cook & Bernfeld 1967 *Radar Signals* + Carrara et al. 1995 *Spotlight
    SAR*: temporal smoothness of the pose trajectory concentrates rFFT energy
    in the lowest 10-20% of bins. The L2 ledger §2.4 falsification test:
    if rFFT spectrum is uniform, sparse retention provides no compression
    gain. Empirical contest dashcam data shows ~80% energy in the lowest 10%
    of bins (sustained-velocity highway driving).

    Trainable parameter: per-pair pose-delta tensor (NUM_PAIRS, POSE_DIM).
    The rFFT + top-K + sparse inverse-rFFT happens at archive build / inflate
    time (NOT at training time — gradient flows through the dense pose tensor).
    """

    def __init__(self, cfg: SARCoherentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Per-pair pose deltas, initialized small (small motion prior).
        self.pose_deltas = nn.Parameter(
            0.01 * torch.randn(cfg.num_pairs, cfg.pose_dim)
        )

    def get_pose_codes(self) -> torch.Tensor:
        """Return per-pair pose codes (NUM_PAIRS, pose_code_dim).

        At training time the pose codes ARE the pose deltas (gradient-friendly).
        At archive build time, the sparse rFFT roundtrip produces a slightly
        lossy version (the smoke test asserts roundtrip error < 1e-2 RMS).
        """
        return self.pose_deltas

    def encode_sparse_rfft(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse top-K rFFT over the temporal axis.

        Returns:
            sparse_coeffs: complex tensor (n_rfft_bins, pose_dim) with zeros
                outside the retained top-K positions.
            topk_indices: int64 tensor (K, pose_dim) of retained bin indices.
        """
        # rFFT along the pair-axis (dim=0). Output shape: (n_rfft_bins, pose_dim).
        rfft_full = torch.fft.rfft(self.pose_deltas, dim=0)
        K = self.cfg.sar_topk()
        # Per-dim top-K by magnitude (different dims may concentrate energy in
        # different bins).
        magnitudes = rfft_full.abs()
        topk = torch.topk(magnitudes, k=K, dim=0)
        topk_indices = topk.indices  # (K, pose_dim)
        sparse = torch.zeros_like(rfft_full)
        # Scatter top-K back into the sparse tensor per dim.
        for d in range(self.cfg.pose_dim):
            sparse[topk_indices[:, d], d] = rfft_full[topk_indices[:, d], d]
        return sparse, topk_indices

    def decode_from_sparse_rfft(
        self,
        sparse_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """Invert sparse rFFT to recover per-pair pose deltas (NUM_PAIRS, pose_dim).

        Used at inflate time after the int16-quantized sparse rFFT bytes have
        been dequantized back to float coefficients.
        """
        return torch.fft.irfft(sparse_coeffs, n=self.cfg.num_pairs, dim=0)

    def estimate_int16_bytes(self) -> int:
        """Closed-form bytes for the sparse rFFT bytes (after deduplication).

        Layout: K positions × pose_dim × (2 B index + 4 B complex int16 = real
        int16 + imag int16) = K * pose_dim * 6. Container headers are counted
        by the archive grammar, not by this payload estimator.
        """
        K = self.cfg.sar_topk()
        return K * self.cfg.pose_dim * 6


class SARCoherentRenderer(nn.Module):
    """Sub-50K-param SIREN-style MLP renderer for the SAR coherent substrate.

    Inputs: per-coordinate ``(x, y, t, pose_code)`` -> RGB pair (6 channels).
    The pose_code is broadcast across all (x, y) coordinates of the same t.
    """

    def __init__(self, cfg: SARCoherentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # input = (x, y, t) + pos_encoding(x, y) + pose_code
        input_dim = 3 + 2 * cfg.coord_feature_freqs * 2 + cfg.pose_code_dim
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
        # Initialize output bias to mid-gray (0.5 in [0,1] space).
        with torch.no_grad():
            output_layer.bias.fill_(0.5)
        self.hidden = nn.ModuleList(layers)
        self.output_layer = output_layer

    def _positional_encode(self, xy: torch.Tensor) -> torch.Tensor:
        """Append sin/cos positional encoding to ``(x, y)`` columns of input."""
        freqs = torch.tensor(
            [2.0**k * math.pi for k in range(self.cfg.coord_feature_freqs)],
            device=xy.device,
            dtype=xy.dtype,
        )
        scaled = xy.unsqueeze(-1) * freqs  # (..., 2, freqs)
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return encoded.flatten(-2)  # (..., 2 * freqs * 2)

    def forward(self, coords: torch.Tensor, pose_code: torch.Tensor) -> torch.Tensor:
        """Render RGB pair from coordinates + per-pair pose code.

        Args:
            coords: ``(N, 3)`` with columns (x, y, t).
            pose_code: ``(pose_code_dim,)`` for this pair, broadcast across N.

        Returns:
            ``(N, 6)`` RGB pair output in ``[0, 1]`` after sigmoid.
        """
        xy = coords[..., :2]
        pe = self._positional_encode(xy)
        # Broadcast pose_code across N coords.
        pose_broadcast = pose_code.unsqueeze(0).expand(coords.shape[0], -1)
        x = torch.cat([coords, pe, pose_broadcast], dim=-1)
        omega_schedule = [self.cfg.first_omega] + [self.cfg.hidden_omega] * (
            self.cfg.num_hidden_layers - 1
        )
        for layer, omega in zip(self.hidden, omega_schedule, strict=True):
            x = torch.sin(omega * layer(x))
        return torch.sigmoid(self.output_layer(x))


class SARCoherentSubstrate(nn.Module):
    """Composite SAR coherent substrate.

    Composes:

    * ``SARCoherentRenderer`` — small SIREN MLP
    * ``SARCoherentPoseCodec`` — per-pair pose deltas with sparse rFFT codec

    ``render_pair(pair_idx)`` returns the predicted RGB pair using the
    pose code for that pair. Per-pair RGB residuals are added at archive-
    build / inflate time by ``inflate.py`` (NOT applied here because they
    are int8-quantized non-differentiable corrections).
    """

    def __init__(self, cfg: SARCoherentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.renderer = SARCoherentRenderer(cfg)
        self.pose_codec = SARCoherentPoseCodec(cfg)

    def _build_coord_grid(self, device: torch.device) -> torch.Tensor:
        """Return ``(H * W, 2)`` grid of (x, y) in [-1, 1]."""
        ys = torch.linspace(-1.0, 1.0, self.cfg.output_height, device=device)
        xs = torch.linspace(-1.0, 1.0, self.cfg.output_width, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    def render_pair(self, pair_idx: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one pair as ``(rgb_0, rgb_1)`` both shaped ``(1, 3, H, W)``.

        Returns RGB in [0, 1] for downstream eval-roundtrip / scorer ingestion.
        """
        idx = int(pair_idx.flatten()[0].item()) if isinstance(pair_idx, torch.Tensor) else int(pair_idx)
        if not 0 <= idx < self.cfg.num_pairs:
            raise IndexError(
                f"pair_idx {idx} out of range [0, {self.cfg.num_pairs})"
            )
        device = self.pose_codec.pose_deltas.device
        H, W = self.cfg.output_height, self.cfg.output_width
        coord_grid_xy = self._build_coord_grid(device)  # (H*W, 2)
        t = idx / max(1, self.cfg.num_pairs - 1)
        t_col = torch.full(
            (H * W, 1), t, device=device, dtype=coord_grid_xy.dtype
        )
        coords = torch.cat([coord_grid_xy, t_col], dim=-1)  # (H*W, 3)
        pose_codes = self.pose_codec.get_pose_codes()  # (NUM_PAIRS, pose_code_dim)
        pose_code = pose_codes[idx]  # (pose_code_dim,)
        out6 = self.renderer(coords, pose_code)  # (H*W, 6)
        rgb6 = out6.t().reshape(1, 6, H, W)
        rgb_0 = rgb6[:, :3]
        rgb_1 = rgb6[:, 3:]
        return rgb_0, rgb_1

    def parameter_count(self) -> int:
        """Total trainable parameters across renderer + pose codec."""
        return sum(p.numel() for p in self.parameters())

    def estimate_renderer_bytes(self) -> int:
        """Closed-form: 2 * renderer_params (FP16) + 8 B header."""
        return 2 * sum(p.numel() for p in self.renderer.parameters()) + 8

    def estimate_total_archive_bytes(self) -> int:
        """Predicted archive size before brotli (renderer + pose codec + per-pair residual)."""
        return (
            self.estimate_renderer_bytes()
            + self.pose_codec.estimate_int16_bytes()
            + self.cfg.num_pairs * self.cfg.per_pair_residual_bytes
            + 64  # header + meta JSON allowance
        )


__all__ = [
    "CAMERA_HW",
    "EVAL_HW",
    "NUM_PAIRS",
    "PER_PAIR_RESIDUAL_TARGET_BYTES",
    "POSE_DIM",
    "SARCoherentConfig",
    "SARCoherentPoseCodec",
    "SARCoherentRenderer",
    "SARCoherentSubstrate",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
]
