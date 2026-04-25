"""Low-complexity overfitted renderers inspired by Cool-Chic and C3.

These are experimental GPU-lane modules:

* ``CoolChicLatentRenderer`` uses shared learned multi-resolution latent grids
  plus a tiny synthesis network. The decoder is intentionally small; most
  rate is in quantizable latents/weights.
* ``C3ResidualRenderer`` wraps any mask renderer and adds a zero-initialized
  coordinate residual head. This lets us test the C3 idea as a residual codec
  without replacing the proven mask-renderer pipeline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.renderer import MotionPredictor, PairGenerator


class _CoordGridMixin:
    """Small bounded coordinate-grid cache shared by the experimental renderers."""

    def __init__(self) -> None:
        self._coord_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def _coord_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device)
        if key not in self._coord_cache:
            if len(self._coord_cache) >= 4:
                self._coord_cache.pop(next(iter(self._coord_cache)))
            yy = torch.linspace(-1.0, 1.0, h, device=device)
            xx = torch.linspace(-1.0, 1.0, w, device=device)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            self._coord_cache[key] = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        return self._coord_cache[key]


class CoolChicLatentRenderer(nn.Module, _CoordGridMixin):
    """Shared multi-resolution latent renderer with a tiny synthesis decoder.

    This is not a literal Cool-Chic bitstream implementation. It imports the
    relevant architectural bias for our experiment: overfit compact latents,
    decode with a low-complexity per-pixel synthesis network, and keep the
    forward path deterministic.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_embed_dim: int = 6,
        latent_ch: int = 8,
        hidden: int = 32,
        latent_shapes: tuple[tuple[int, int], ...] = ((6, 8), (12, 16), (24, 32)),
    ):
        nn.Module.__init__(self)
        _CoordGridMixin.__init__(self)
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.latent_ch = latent_ch
        self.hidden = hidden
        self.latent_shapes = tuple((int(h), int(w)) for h, w in latent_shapes)

        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.latents = nn.ParameterList(
            [nn.Parameter(torch.empty(1, latent_ch, h, w)) for h, w in self.latent_shapes]
        )

        in_ch = 2 + class_embed_dim + latent_ch * len(self.latent_shapes)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for latent in self.latents:
            nn.init.normal_(latent, mean=0.0, std=0.02)
        nn.init.normal_(self.class_embed.weight, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.decoder[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.decoder[0].bias)
        nn.init.kaiming_uniform_(self.decoder[2].weight, a=math.sqrt(5))
        nn.init.zeros_(self.decoder[2].bias)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        b, h, w = masks.shape
        device = masks.device

        coords = self._coord_grid(h, w, device).expand(b, -1, -1, -1)
        class_features = self.class_embed(masks).permute(0, 3, 1, 2).contiguous()
        latent_features = [
            F.interpolate(latent, size=(h, w), mode="bilinear", align_corners=False).expand(b, -1, -1, -1)
            for latent in self.latents
        ]
        features = torch.cat([coords, class_features, *latent_features], dim=1)
        logits = self.decoder(features)
        return 255.0 * torch.sigmoid(logits / 50.0)

    def decoder_param_count(self) -> int:
        return sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class C3ResidualRenderer(nn.Module, _CoordGridMixin):
    """Coordinate-conditioned residual head on top of a base mask renderer."""

    def __init__(
        self,
        base_renderer: nn.Module,
        num_classes: int = 5,
        class_embed_dim: int = 4,
        residual_hidden: int = 32,
        residual_layers: int = 2,
        residual_scale: float = 16.0,
        num_bands: int = 6,
    ):
        nn.Module.__init__(self)
        _CoordGridMixin.__init__(self)
        self.base_renderer = base_renderer
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.residual_hidden = residual_hidden
        self.residual_layers = residual_layers
        self.residual_scale = residual_scale
        self.num_bands = num_bands

        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        freqs = 2.0 ** torch.arange(num_bands).float() * math.pi
        self.register_buffer("freqs", freqs)

        coord_ch = 2 + 4 * num_bands
        in_ch = coord_ch + class_embed_dim + 3
        layers: list[nn.Module] = []
        for i in range(residual_layers):
            layers.append(nn.Conv2d(in_ch if i == 0 else residual_hidden, residual_hidden, 1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(residual_hidden, 3, 1, bias=True))
        self.residual_net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.class_embed.weight, mean=0.0, std=0.02)
        for module in self.residual_net[:-1]:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def _encoded_coords(self, h: int, w: int, device: torch.device, batch: int) -> torch.Tensor:
        coords = self._coord_grid(h, w, device)
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        x_scaled = x * self.freqs.view(1, -1, 1, 1)
        y_scaled = y * self.freqs.view(1, -1, 1, 1)
        encoded = torch.cat(
            [x, y, torch.sin(x_scaled), torch.cos(x_scaled), torch.sin(y_scaled), torch.cos(y_scaled)],
            dim=1,
        )
        return encoded.expand(batch, -1, -1, -1)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        base = self.base_renderer(masks)
        b, _, h, w = base.shape
        coord_features = self._encoded_coords(h, w, masks.device, b)
        class_features = self.class_embed(masks).permute(0, 3, 1, 2).contiguous()
        residual_in = torch.cat([base / 255.0, coord_features, class_features], dim=1)
        residual = self.residual_scale * torch.tanh(self.residual_net(residual_in))
        return (base + residual).clamp(0.0, 255.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _build_motion(
    num_classes: int,
    embed_dim: int,
    motion_hidden: int,
) -> MotionPredictor:
    return MotionPredictor(num_classes=num_classes, embed_dim=embed_dim, hidden=motion_hidden)


def build_coolchic_renderer(
    num_classes: int = 5,
    embed_dim: int = 6,
    latent_ch: int = 8,
    hidden: int = 32,
    motion_hidden: int = 32,
    latent_shapes: tuple[tuple[int, int], ...] = ((6, 8), (12, 16), (24, 32)),
    blend_mode: str = "scalar",
    noise_mode: str = "deterministic",
) -> PairGenerator:
    renderer = CoolChicLatentRenderer(
        num_classes=num_classes,
        class_embed_dim=embed_dim,
        latent_ch=latent_ch,
        hidden=hidden,
        latent_shapes=latent_shapes,
    )
    motion = _build_motion(num_classes, embed_dim, motion_hidden)
    pair_gen = PairGenerator(renderer, motion, blend_mode=blend_mode, noise_mode=noise_mode)
    print(
        f"[coolchic_renderer] Built PairGenerator: {pair_gen.param_count():,} params "
        f"(renderer={renderer.param_count():,}, decoder={renderer.decoder_param_count():,})"
    )
    return pair_gen


def build_c3_residual_renderer(
    num_classes: int = 5,
    embed_dim: int = 6,
    latent_ch: int = 6,
    hidden: int = 24,
    motion_hidden: int = 32,
    residual_hidden: int = 32,
    residual_layers: int = 2,
    residual_scale: float = 16.0,
    latent_shapes: tuple[tuple[int, int], ...] = ((6, 8), (12, 16), (24, 32)),
    blend_mode: str = "scalar",
    noise_mode: str = "deterministic",
) -> PairGenerator:
    base = CoolChicLatentRenderer(
        num_classes=num_classes,
        class_embed_dim=embed_dim,
        latent_ch=latent_ch,
        hidden=hidden,
        latent_shapes=latent_shapes,
    )
    renderer = C3ResidualRenderer(
        base,
        num_classes=num_classes,
        class_embed_dim=max(2, embed_dim // 2),
        residual_hidden=residual_hidden,
        residual_layers=residual_layers,
        residual_scale=residual_scale,
    )
    motion = _build_motion(num_classes, embed_dim, motion_hidden)
    pair_gen = PairGenerator(renderer, motion, blend_mode=blend_mode, noise_mode=noise_mode)
    print(
        f"[c3_residual_renderer] Built PairGenerator: {pair_gen.param_count():,} params "
        f"(base={base.param_count():,}, residual={renderer.param_count() - base.param_count():,})"
    )
    return pair_gen
