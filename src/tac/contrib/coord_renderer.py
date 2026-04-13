"""Coordinate-based neural field renderer (INRIA COOL approach).

Instead of a convolutional U-Net, this renderer uses a per-pixel MLP:
    input: (x, y, class_id) with positional encoding
    output: (R, G, B) color at that pixel

This is the smallest possible renderer (~50K params) because it has no
spatial convolutions -- each pixel is processed independently.

Positional encoding with 10 frequency bands provides spatial structure
that the MLP can use to generate smooth color fields per class.

Architecture:
    PE(x, y) [42d] + class_embed [8d] = 50d input
    -> Linear(50, 64) -> ReLU
    -> Linear(64, 64) -> ReLU
    -> Linear(64, 3) -> Sigmoid * 255

Classes:
    - PositionalEncoding: sin/cos frequency encoding for spatial coords
    - CoordRenderer: the coordinate-based renderer MLP
    - CoordPairGenerator: wraps CoordRenderer + MotionPredictor for pair output
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tac.renderer import MotionPredictor, warp_with_flow


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for spatial coordinates.

    Maps 2D coordinates (x, y) to a higher-dimensional space using
    sin/cos at multiple frequency bands. This gives the MLP enough
    spatial structure to learn smooth color fields.

    Output dimension: 2 * (2 * num_bands + 1) = 2 + 4 * num_bands
    With 10 bands: 42 dimensions.

    Args:
        num_bands: number of frequency bands (default 10)
    """

    def __init__(self, num_bands: int = 10):
        super().__init__()
        self.num_bands = num_bands
        # Frequencies: 2^0, 2^1, ..., 2^(num_bands-1)
        freqs = 2.0 ** torch.arange(num_bands).float() * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        """Output dimensionality: raw (2) + sin/cos per band (4 * num_bands)."""
        return 2 + 4 * self.num_bands

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode 2D coordinates with sinusoidal positional encoding.

        Args:
            coords: (..., 2) tensor of (x, y) in [-1, 1]

        Returns:
            (..., output_dim) encoded coordinates
        """
        # coords: (..., 2)
        x, y = coords[..., 0:1], coords[..., 1:2]

        # Apply each frequency band to both x and y
        # freqs: (num_bands,) -> broadcast over spatial dims
        x_scaled = x * self.freqs  # (..., num_bands)
        y_scaled = y * self.freqs  # (..., num_bands)

        encoded = torch.cat(
            [
                x,
                y,  # raw coordinates (2)
                torch.sin(x_scaled),  # sin(f * x) for each freq (num_bands)
                torch.cos(x_scaled),  # cos(f * x) for each freq (num_bands)
                torch.sin(y_scaled),  # sin(f * y) for each freq (num_bands)
                torch.cos(y_scaled),  # cos(f * y) for each freq (num_bands)
            ],
            dim=-1,
        )

        return encoded


class CoordRenderer(nn.Module):
    """Neural field renderer: per-pixel MLP maps (x, y, class) to RGB.

    The smallest possible renderer architecture (~50K params). No spatial
    convolutions -- each pixel is processed independently through a shared MLP.
    Spatial coherence comes from positional encoding + class-conditional generation.

    Args:
        num_classes: segmentation classes (5 for comma SegNet)
        class_embed_dim: dimension of learned class embeddings
        hidden_dim: MLP hidden layer width
        num_bands: positional encoding frequency bands
        num_layers: number of hidden layers in MLP
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_embed_dim: int = 8,
        hidden_dim: int = 64,
        num_bands: int = 10,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Positional encoding for (x, y)
        self.pe = PositionalEncoding(num_bands)

        # Class embedding
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)

        # MLP: PE_dim + class_embed_dim -> hidden -> ... -> 3 (RGB)
        in_dim = self.pe.output_dim + class_embed_dim
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)])
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_dim, 3))  # output RGB
        self.mlp = nn.Sequential(*layers)

        # Zero-init output layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Cache coordinate grid
        self._coord_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def _get_coord_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or create cached coordinate grid.

        Returns:
            (H, W, 2) tensor with (x, y) in [-1, 1]
        """
        key = (H, W, device)
        if key not in self._coord_cache:
            # Bounded cache (maxsize=4) matching renderer.py _coord_grid_cache convention
            if len(self._coord_cache) >= 4:
                self._coord_cache.pop(next(iter(self._coord_cache)))
            yy = torch.linspace(-1.0, 1.0, H, device=device)
            xx = torch.linspace(-1.0, 1.0, W, device=device)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            self._coord_cache[key] = torch.stack([grid_x, grid_y], dim=-1)
        return self._coord_cache[key]

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Render RGB frames from segmentation masks using per-pixel MLP.

        Args:
            masks: (B, H, W) long tensor with values in [0, num_classes)

        Returns:
            (B, 3, H, W) float tensor in [0, 255]
        """
        B, H, W = masks.shape
        device = masks.device

        # Get coordinate grid: (H, W, 2)
        coords = self._get_coord_grid(H, W, device)

        # Positional encoding: (H, W, 2) -> (H, W, PE_dim)
        pe_encoded = self.pe(coords)  # (H, W, PE_dim)
        pe_encoded = pe_encoded.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, PE_dim)

        # Class embedding: (B, H, W) -> (B, H, W, class_embed_dim)
        class_features = self.class_embed(masks)

        # Concatenate: (B, H, W, PE_dim + class_embed_dim)
        mlp_input = torch.cat([pe_encoded, class_features], dim=-1)

        # Reshape for MLP: (B*H*W, input_dim)
        flat_input = mlp_input.reshape(B * H * W, -1)

        # Forward through MLP: (B*H*W, 3)
        flat_rgb = self.mlp(flat_input)

        # Reshape back: (B, H, W, 3)
        rgb_hwc = flat_rgb.reshape(B, H, W, 3)

        # Soft sigmoid output, permute to (B, 3, H, W)
        rgb = 255.0 * torch.sigmoid(rgb_hwc / 50.0)
        return rgb.permute(0, 3, 1, 2).contiguous()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CoordPairGenerator(nn.Module):
    """Generate PoseNet-compatible frame pairs using CoordRenderer.

    Wraps CoordRenderer + MotionPredictor into the same interface as
    PairGenerator but with the coordinate-based renderer.

    Args:
        renderer: CoordRenderer instance
        motion: MotionPredictor instance
    """

    def __init__(self, renderer: CoordRenderer, motion: MotionPredictor):
        super().__init__()
        self.renderer = renderer
        self.motion = motion
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Generate frame pair from mask pair.

        Args:
            mask_t: (B, H, W) long
            mask_t1: (B, H, W) long

        Returns:
            (B, 2, H, W, 3) float tensor in [0, 255]
        """
        frame_t = self.renderer(mask_t)
        frame_t1 = self.renderer(mask_t1)

        flow = self.motion(mask_t, mask_t1)
        frame_t1_warped = warp_with_flow(frame_t, flow)

        alpha = torch.sigmoid(self.blend_logit)
        frame_t1_blended = (alpha * frame_t1_warped + (1.0 - alpha) * frame_t1).clamp(0.0, 255.0)

        f_t_hwc = frame_t.permute(0, 2, 3, 1)
        f_t1_hwc = frame_t1_blended.permute(0, 2, 3, 1)
        return torch.stack([f_t_hwc, f_t1_hwc], dim=1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_coord_renderer(
    num_classes: int = 5,
    class_embed_dim: int = 8,
    hidden_dim: int = 64,
    num_bands: int = 10,
    num_layers: int = 3,
    motion_hidden: int = 32,
    motion_embed_dim: int = 6,
) -> CoordPairGenerator:
    """Build coordinate-based renderer with motion predictor.

    Args:
        num_classes: segmentation classes
        class_embed_dim: class embedding dimension for renderer
        hidden_dim: MLP hidden width
        num_bands: positional encoding frequency bands
        num_layers: MLP hidden layers
        motion_hidden: MotionPredictor hidden width
        motion_embed_dim: MotionPredictor embedding dimension

    Returns:
        CoordPairGenerator wrapping CoordRenderer + MotionPredictor
    """
    renderer = CoordRenderer(
        num_classes=num_classes,
        class_embed_dim=class_embed_dim,
        hidden_dim=hidden_dim,
        num_bands=num_bands,
        num_layers=num_layers,
    )
    motion = MotionPredictor(
        num_classes=num_classes,
        embed_dim=motion_embed_dim,
        hidden=motion_hidden,
    )
    pair_gen = CoordPairGenerator(renderer, motion)

    total = pair_gen.param_count()
    r_count = renderer.param_count()
    m_count = motion.param_count()
    print(
        f"[coord_renderer] Built CoordPairGenerator: {total:,} params "
        f"(renderer={r_count:,}, motion={m_count:,}, blend=1)"
    )

    return pair_gen
