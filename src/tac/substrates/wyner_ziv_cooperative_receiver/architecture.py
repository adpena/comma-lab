# SPDX-License-Identifier: MIT
"""Wyner-Ziv cooperative-receiver architecture (alien-tech N3).

Implements the DISCUS construction (Pradhan-Ramchandran 2003) of Wyner-Ziv
1976 source coding with side information at the decoder, specialized to
the comma video compression contest:

* **Source X**: ground-truth RGB pair tensors at scorer resolution.
* **Side information Y**: a learned ``SideInfoPredictor`` that predicts
  the source from a small per-pair coordinate code + global pose code.
  At inflate time, the SAME predictor regenerates Y deterministically.
* **Coset index U**: a small integer per pair indicating which coset of
  a structured code the source falls into. Transmitted in the archive.
* **Reconstruction X_hat**: at inflate time, the receiver enumerates
  members of coset U and picks the candidate nearest to Y.

The architecture is INTENTIONALLY simple: the score gains come from the
score-aware Wyner-Ziv loss + the conditional-rate term ``H(X | Y)`` that
penalizes wasted bits, not from architectural complexity. Per HNeRV
parity discipline lesson L12 (single-LOC-per-LOC review discipline):
every line below is reviewable in 30 seconds.

**Catalog #124 archive-grammar 8 fields** declared inline so the AST
walker observes them:

* ``archive_grammar``: monolithic single-file ``0.bin``
* ``parser_section_manifest``: WZ1 header + renderer + coset indices +
  side-info predictor + meta JSON
* ``inflate_runtime_loc_budget``: <= 200 LOC substrate-engineering waiver
* ``runtime_dep_closure``: torch + brotli only
* ``export_format``: WZ1
* ``score_aware_loss``: cooperative-receiver + Wyner-Ziv conditional rate
* ``bolt_on_loc_budget``: lane_class=substrate_engineering
* ``no_op_detector_planned``: emit/parse roundtrip preserves bytes byte-for-byte
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Scorer resolution (height, width). Renderer outputs at this resolution."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 per pair)."""

CAMERA_HW: tuple[int, int] = (874, 1164)
"""Contest camera native resolution; inflate.py upsamples renderer output."""

# Archive size budget (renderer ~ 30 KB + side-info predictor ~ 18 KB +
# coset indices ~ 6 KB + headers/meta ~ 2 KB = ~ 56 KB before brotli;
# brotli typically closes the final 5-15 KB to land in 50-70 KB target,
# significantly below PR101 baseline ~ 114 KB by trading bits-per-pair
# for shared decoder structure per Wyner-Ziv R_WZ(D) advantage).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 50_000
TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 70_000
PER_PAIR_COSET_INDEX_BITS: int = 8
"""Coset-index bit budget per pair (DISCUS uses small log2(num_cosets) per
pair; default 8 bits => 256 cosets; provides 1-byte-per-pair archive cost
for the coset stream)."""


@dataclass(frozen=True)
class WynerZivConfig:
    """Static design-time parameters for the Wyner-Ziv substrate.

    Defaults aim for ~30 KB FP16 renderer + ~18 KB FP16 side-info predictor
    + ~6 KB coset indices (1 B per pair * 600 pairs = 600 B compressed by
    brotli further) + ~2 KB header/meta = ~56 KB before brotli, target
    final size 50-70 KB.

    Args:
        hidden_dim: Width of the renderer MLP. Default 48 (smaller than
            time-traveler since the side-info predictor carries half the
            structural burden).
        num_hidden_layers: Depth of the renderer MLP. Default 3.
        side_info_hidden_dim: Width of the side-info predictor MLP. Smaller
            than the main renderer because Y only needs to disambiguate
            cosets, not reconstruct full RGB.
        side_info_num_layers: Depth of the side-info predictor MLP.
        coord_dim: Input coordinate dim per pair (x, y, t).
        pose_dim: Per-pair pose code dim (small SE(3)-inspired Lie algebra).
        coset_index_bits: Bits per pair allocated to the coset index
            (DISCUS small-bin budget). 8 = 256 cosets per pair.
        num_cosets: Derived as ``2 ** coset_index_bits``.
        first_omega: SIREN-style first-layer omega for the renderer MLP.
        hidden_omega: SIREN-style downstream omega.
        num_pairs: Contest pair count (600).
        output_height, output_width: Renderer eval resolution (384, 512).
        coord_feature_freqs: Positional-encoding bands for (x, y).
        wyner_ziv_dither_std: Std-dev of dither added to the source before
            coset binning during training (reduces quantization-edge
            instability; 0 at eval).
    """

    hidden_dim: int = 48
    num_hidden_layers: int = 3
    side_info_hidden_dim: int = 32
    side_info_num_layers: int = 2
    coord_dim: int = 3
    pose_dim: int = 6
    coset_index_bits: int = PER_PAIR_COSET_INDEX_BITS
    first_omega: float = 30.0
    hidden_omega: float = 1.0
    num_pairs: int = NUM_PAIRS
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    coord_feature_freqs: int = 4
    wyner_ziv_dither_std: float = 0.005

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {self.hidden_dim}")
        if self.num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive; got {self.num_hidden_layers}"
            )
        if self.side_info_hidden_dim <= 0:
            raise ValueError(
                f"side_info_hidden_dim must be positive; got {self.side_info_hidden_dim}"
            )
        if self.side_info_num_layers <= 0:
            raise ValueError(
                f"side_info_num_layers must be positive; got {self.side_info_num_layers}"
            )
        if self.coord_dim != 3:
            raise ValueError(f"coord_dim must be 3 (x, y, t); got {self.coord_dim}")
        if self.pose_dim != 6:
            raise ValueError(f"pose_dim must be 6; got {self.pose_dim}")
        if not 1 <= self.coset_index_bits <= 16:
            raise ValueError(
                f"coset_index_bits must be in [1, 16]; got {self.coset_index_bits}"
            )
        if self.num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive; got {self.num_pairs}")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output dims must be positive")
        if self.first_omega <= 0.0 or self.hidden_omega <= 0.0:
            raise ValueError("omega values must be positive")
        if self.coord_feature_freqs <= 0:
            raise ValueError("coord_feature_freqs must be positive")
        if self.wyner_ziv_dither_std < 0.0:
            raise ValueError("wyner_ziv_dither_std must be >= 0")

    @property
    def num_cosets(self) -> int:
        """``2 ** coset_index_bits`` — DISCUS coset count per pair."""
        return 1 << self.coset_index_bits

    def predict_renderer_param_count(self) -> int:
        """Closed-form prediction of renderer-MLP parameter count."""
        input_dim = self.coord_dim + 2 * self.coord_feature_freqs * 2
        output_dim = 6  # rgb_0 + rgb_1 = 2 * 3
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


def _positional_encode(
    xy: torch.Tensor, *, freqs: int
) -> torch.Tensor:
    """Append sin/cos positional encoding to ``(..., 2)`` columns of input.

    Returns ``(..., 2 * freqs * 2)`` flattened encoding.
    """
    bands = torch.tensor(
        [2.0**k * math.pi for k in range(freqs)],
        device=xy.device,
        dtype=xy.dtype,
    )
    scaled = xy.unsqueeze(-1) * bands  # (..., 2, freqs)
    encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
    return encoded.flatten(-2)


class _SirenMLP(nn.Module):
    """Small SIREN-style MLP shared by renderer and side-info predictor."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        first_omega: float,
        hidden_omega: float,
        output_bias_init: float,
    ) -> None:
        super().__init__()
        self.first_omega = first_omega
        self.hidden_omega = hidden_omega
        layers: list[nn.Module] = []
        in_features = input_dim
        for layer_idx in range(num_hidden_layers):
            lin = nn.Linear(in_features, hidden_dim)
            omega = first_omega if layer_idx == 0 else hidden_omega
            _siren_init_(lin, is_first=(layer_idx == 0), omega=omega)
            layers.append(lin)
            in_features = hidden_dim
        output_layer = nn.Linear(in_features, output_dim)
        _siren_init_(output_layer, is_first=False, omega=hidden_omega)
        with torch.no_grad():
            output_layer.bias.fill_(output_bias_init)
        self.hidden = nn.ModuleList(layers)
        self.output_layer = output_layer
        self._num_hidden = num_hidden_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega_schedule = [self.first_omega] + [self.hidden_omega] * (
            self._num_hidden - 1
        )
        for layer, omega in zip(self.hidden, omega_schedule, strict=True):
            x = torch.sin(omega * layer(x))
        return torch.sigmoid(self.output_layer(x))


class SideInfoPredictor(nn.Module):
    """Wyner-Ziv decoder-side predictor producing Y from coordinate + pose.

    Implements the receiver's ``Y = predictor(coords, pose_code)`` view.
    Both encoder and decoder evaluate this predictor on the SAME per-pair
    pose code, so Y is deterministic given the (small) pose code that lives
    in the archive.

    The coset disambiguation step picks the coset member nearest to Y.
    """

    def __init__(self, cfg: WynerZivConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Input: positional-encoded (x, y) + t + pose_code (6 dims).
        input_dim = (
            cfg.coord_dim + 2 * cfg.coord_feature_freqs * 2 + cfg.pose_dim
        )
        self.mlp = _SirenMLP(
            input_dim=input_dim,
            hidden_dim=cfg.side_info_hidden_dim,
            num_hidden_layers=cfg.side_info_num_layers,
            output_dim=6,
            first_omega=cfg.first_omega,
            hidden_omega=cfg.hidden_omega,
            output_bias_init=0.5,
        )

    def forward(
        self, coords: torch.Tensor, pose_code: torch.Tensor
    ) -> torch.Tensor:
        """Predict ``Y`` from coords + pose_code.

        Args:
            coords: ``(..., 3)`` with columns (x, y, t).
            pose_code: ``(pose_dim,)`` per-pair pose code.

        Returns:
            ``(..., 6)`` predicted RGB pair in ``[0, 1]`` (sigmoid output).
        """
        xy = coords[..., :2]
        pe = _positional_encode(xy, freqs=self.cfg.coord_feature_freqs)
        n = coords.shape[0]
        pose_expanded = pose_code.unsqueeze(0).expand(n, -1)
        x = torch.cat([coords, pe, pose_expanded], dim=-1)
        return self.mlp(x)


class _Renderer(nn.Module):
    """Sub-30K-param SIREN-style MLP renderer for the source X.

    Inputs: per-pair coordinate ``(x, y, t)`` -> RGB pair (6 channels).
    The renderer is the substrate's reconstruction path; receiver applies
    coset disambiguation against side-info Y.
    """

    def __init__(self, cfg: WynerZivConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.coord_dim + 2 * cfg.coord_feature_freqs * 2
        self.mlp = _SirenMLP(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            output_dim=6,
            first_omega=cfg.first_omega,
            hidden_omega=cfg.hidden_omega,
            output_bias_init=0.5,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        xy = coords[..., :2]
        pe = _positional_encode(xy, freqs=self.cfg.coord_feature_freqs)
        x = torch.cat([coords, pe], dim=-1)
        return self.mlp(x)


def slepian_wolf_coset_index(
    source: torch.Tensor, *, num_cosets: int
) -> torch.Tensor:
    """DISCUS coset binning: quantize source values into reachable cosets.

    The receiver enumerates scalar representatives and applies the SAME
    quantizer. This must be reachable and monotone; a hash can make valid
    transmitted indices unrecoverable when the decoder search grid is smaller
    than the coset count.

    Args:
        source: float tensor in ``[0, 1]`` (e.g., the mean of a small RGB
            patch acting as the coset key).
        num_cosets: power-of-two coset count (matches ``cfg.num_cosets``).

    Returns:
        Long-int tensor of coset indices in ``[0, num_cosets)``.
    """
    if num_cosets <= 0 or (num_cosets & (num_cosets - 1)) != 0:
        raise ValueError(f"num_cosets must be a positive power of 2; got {num_cosets}")
    if num_cosets == 1:
        return torch.zeros_like(source, dtype=torch.long)
    scaled = source.detach().float().clamp(0.0, 1.0) * float(num_cosets - 1)
    return scaled.round().to(torch.long).clamp(0, num_cosets - 1)


def disambiguate_coset(
    side_info: torch.Tensor,
    *,
    coset_index: int,
    num_cosets: int,
    search_grid: int = 32,
) -> torch.Tensor:
    """Pick the coset member nearest to ``side_info`` matching ``coset_index``.

    Implementation: enumerate ``search_grid`` candidate values uniformly in
    ``[0, 1]``, quantize each through ``slepian_wolf_coset_index``, and pick
    the nearest matching candidate. The grid must be at least as large as the
    coset count so every transmitted index is decodable.

    Args:
        side_info: ``(...,)`` predicted Y in ``[0, 1]``.
        coset_index: integer in ``[0, num_cosets)`` identifying the coset.
        num_cosets: matching ``cfg.num_cosets``.
        search_grid: number of candidates to enumerate.

    Returns:
        Tensor with same shape as ``side_info`` containing the disambiguated
        reconstruction.
    """
    if coset_index < 0 or coset_index >= num_cosets:
        raise ValueError(
            f"coset_index must be in [0, {num_cosets}); got {coset_index}"
        )
    if search_grid < num_cosets:
        raise ValueError(
            f"search_grid must be >= num_cosets so every transmitted coset is "
            f"reachable; got search_grid={search_grid} num_cosets={num_cosets}"
        )
    candidates = torch.linspace(0.0, 1.0, search_grid, device=side_info.device)
    cand_indices = slepian_wolf_coset_index(candidates, num_cosets=num_cosets)
    matching = candidates[cand_indices == coset_index]
    if matching.numel() == 0:
        raise ValueError(
            f"no candidate representative for coset_index={coset_index}; "
            f"search_grid={search_grid} num_cosets={num_cosets}"
        )
    diffs = (matching.unsqueeze(0) - side_info.unsqueeze(-1)).abs()
    pick = diffs.argmin(dim=-1)
    return matching[pick]


class WynerZivSubstrate(nn.Module):
    """Composite Wyner-Ziv substrate (renderer + side-info predictor + pose codes).

    ``render_pair(pair_idx)`` returns the predicted RGB pair from the renderer
    alone (no coset disambiguation — that happens at archive-build / inflate
    time by ``WynerZivArchive``).

    ``predict_side_info(pair_idx)`` returns the receiver's ``Y`` view from
    the SideInfoPredictor + per-pair pose code.
    """

    def __init__(self, cfg: WynerZivConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.renderer = _Renderer(cfg)
        self.side_info = SideInfoPredictor(cfg)
        # Per-pair pose code (small SE(3)-inspired Lie algebra). Encoder and
        # decoder both see this; lives in the archive.
        self.pose_codes = nn.Parameter(
            0.01 * torch.randn(cfg.num_pairs, cfg.pose_dim)
        )

    def _build_coord_grid(self, device: torch.device) -> torch.Tensor:
        """Return ``(H * W, 2)`` grid of (x, y) in [-1, 1]."""
        ys = torch.linspace(-1.0, 1.0, self.cfg.output_height, device=device)
        xs = torch.linspace(-1.0, 1.0, self.cfg.output_width, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    def _make_coords(self, pair_idx: int, device: torch.device) -> torch.Tensor:
        """Build ``(H * W, 3)`` coords with (x, y, t) for the given pair."""
        H, W = self.cfg.output_height, self.cfg.output_width
        coord_grid_xy = self._build_coord_grid(device)
        t = pair_idx / max(1, self.cfg.num_pairs - 1)
        t_col = torch.full(
            (H * W, 1), t, device=device, dtype=coord_grid_xy.dtype
        )
        return torch.cat([coord_grid_xy, t_col], dim=-1)

    def render_pair(
        self, pair_idx: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one pair as ``(rgb_0, rgb_1)`` both shaped ``(1, 3, H, W)``."""
        idx = (
            int(pair_idx.flatten()[0].item())
            if isinstance(pair_idx, torch.Tensor)
            else int(pair_idx)
        )
        if not 0 <= idx < self.cfg.num_pairs:
            raise IndexError(
                f"pair_idx {idx} out of range [0, {self.cfg.num_pairs})"
            )
        device = self.pose_codes.device
        H, W = self.cfg.output_height, self.cfg.output_width
        coords = self._make_coords(idx, device)
        out6 = self.renderer(coords)
        rgb6 = out6.t().reshape(1, 6, H, W)
        return rgb6[:, :3], rgb6[:, 3:]

    def predict_side_info(
        self, pair_idx: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run side-info predictor and return ``(Y_0, Y_1)`` shaped ``(1, 3, H, W)``."""
        idx = (
            int(pair_idx.flatten()[0].item())
            if isinstance(pair_idx, torch.Tensor)
            else int(pair_idx)
        )
        if not 0 <= idx < self.cfg.num_pairs:
            raise IndexError(
                f"pair_idx {idx} out of range [0, {self.cfg.num_pairs})"
            )
        device = self.pose_codes.device
        H, W = self.cfg.output_height, self.cfg.output_width
        coords = self._make_coords(idx, device)
        pose_code = self.pose_codes[idx]
        out6 = self.side_info(coords, pose_code)
        rgb6 = out6.t().reshape(1, 6, H, W)
        return rgb6[:, :3], rgb6[:, 3:]

    def parameter_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def estimate_substrate_bytes(self) -> int:
        """Closed-form: 2 * (renderer + side_info_pred) FP16 + pose codes FP16."""
        renderer_bytes = 2 * sum(p.numel() for p in self.renderer.parameters())
        side_info_bytes = 2 * sum(p.numel() for p in self.side_info.parameters())
        pose_codes_bytes = 2 * self.cfg.num_pairs * self.cfg.pose_dim
        return renderer_bytes + side_info_bytes + pose_codes_bytes


__all__ = [
    "CAMERA_HW",
    "EVAL_HW",
    "NUM_PAIRS",
    "PER_PAIR_COSET_INDEX_BITS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "SideInfoPredictor",
    "WynerZivConfig",
    "WynerZivSubstrate",
    "disambiguate_coset",
    "slepian_wolf_coset_index",
]
