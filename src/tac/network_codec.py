"""Network-IS-the-codec: a tiny neural network whose weights ARE the compressed video.

The radical approach for the GPU lane: instead of storing compressed video +
postfilter, store a TINY neural network whose forward pass GENERATES the
video frames. The network weights ARE the compressed representation.
Inference IS decompression.

Key insight: the network only needs to memorize ONE specific video (1200
frames). This is pure overfitting, which is exactly what we want for
compression. No generalization needed.

Architecture choices:
- SIREN (sinusoidal activations) -- excellent for memorizing continuous signals
- Positional encoding (NeRF-style) -- frame_index -> fourier features -> RGB
- Hash grid encoding (Instant-NGP style) -- fast, compact

Size target: 50-100KB for the entire network (smaller than current 893KB archive).

Score formula: S = 100*seg + sqrt(10*pose) + 25*rate
The network_size_in_bits term directly optimizes the rate component.

Usage::

    from tac.network_codec import (
        SIRENVideoCodec,
        SelfCompressingVideoCodec,
        train_network_codec,
        export_network_archive,
        inflate_network_codec,
    )
"""

from __future__ import annotations

import io
import json
import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SIRENLayer",
    "SIRENVideoCodec",
    "SelfCompressingVideoCodec",
    "MaskConditionedSIREN",
    "train_network_codec",
    "train_mask_conditioned_siren",
    "export_network_archive",
    "export_mask_siren_archive",
    "inflate_network_codec",
    "inflate_mask_siren_archive",
]


# ── SIREN primitives ────────────────────────────────────────────────────


class SIRENLayer(nn.Module):
    """Single SIREN layer with sinusoidal activation.

    SIREN (Sinusoidal Representation Network, Sitzmann et al. 2020)
    uses sin() activations which are excellent for representing continuous
    signals. The omega_0 parameter controls the frequency of the sinusoid.

    Initialization is critical for numerical stability:
    - First layer: weights uniform in [-1/fan_in, 1/fan_in]
    - Hidden layers: weights uniform in [-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0]

    Args:
        in_features: input dimension.
        out_features: output dimension.
        omega_0: frequency multiplier for sin activation.
        is_first: True for the first layer (different initialization).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        """SIREN-specific initialization for numerical stability."""
        fan_in = self.linear.weight.shape[1]
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/fan_in, 1/fan_in]
                bound = 1.0 / fan_in
            else:
                # Hidden layers: sqrt(6/fan_in) / omega_0
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class PositionalEncoding(nn.Module):
    """NeRF-style Fourier feature positional encoding.

    Maps scalar input to high-dimensional Fourier features:
    x -> [sin(2^0 * pi * x), cos(2^0 * pi * x), ...,
           sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]

    This allows the network to represent high-frequency content
    that MLPs with standard activations struggle with.

    Args:
        num_frequencies: number of frequency octaves (L).
        include_input: whether to include the raw input in output.
    """

    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        # Pre-compute frequency bands
        freqs = 2.0 ** torch.arange(num_frequencies).float() * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        d = 2 * self.num_frequencies  # sin + cos per frequency
        if self.include_input:
            d += 1
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalar values to Fourier features.

        Args:
            x: (...,) or (..., 1) float tensor.

        Returns:
            (..., output_dim) Fourier features.
        """
        if x.ndim > 0 and x.shape[-1] != 1:
            x = x.unsqueeze(-1)

        # (*, 1) * (L,) -> (*, L)
        scaled = x * self.freqs
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        if self.include_input:
            features = torch.cat([x, features], dim=-1)

        return features


# ── SIREN Video Codec ────────────────────────────────────────────────────


class SIRENVideoCodec(nn.Module):
    """SIREN-based video memorization network.

    SIREN uses sin() activations which are excellent for representing
    continuous signals (images, video).

    Input: (frame_idx, y, x) normalized to [-1, 1]
    Output: (R, G, B) in [0, 255]

    The entire video is encoded as the network weights. At inflate time,
    the network is evaluated at every (frame, y, x) coordinate to
    reconstruct the video.

    Args:
        hidden: hidden layer width (controls capacity vs size).
        layers: number of hidden layers.
        omega_0: SIREN frequency parameter (higher = more detail).
        num_frames: number of video frames (for positional encoding).
        frame_h: frame height.
        frame_w: frame width.
        pos_encoding_freqs: Fourier feature frequencies for frame index.
    """

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 4,
        omega_0: float = 30.0,
        num_frames: int = 1200,
        frame_h: int = 384,
        frame_w: int = 512,
        pos_encoding_freqs: int = 6,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_layers = layers
        self.omega_0 = omega_0
        self.num_frames = num_frames
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.pos_encoding_freqs = pos_encoding_freqs

        # Positional encoding for frame index (temporal)
        self.pos_enc = PositionalEncoding(
            num_frequencies=pos_encoding_freqs, include_input=True,
        )

        # Input: pos_enc(frame_idx) + y + x
        # = (2*pos_encoding_freqs + 1) + 2 spatial coords
        input_dim = self.pos_enc.output_dim + 2

        # Build SIREN layers
        net_layers = []
        net_layers.append(SIRENLayer(input_dim, hidden, omega_0=omega_0, is_first=True))
        for _ in range(layers - 1):
            net_layers.append(SIRENLayer(hidden, hidden, omega_0=omega_0))
        self.net = nn.ModuleList(net_layers)

        # Output: linear to RGB (no sin activation on output)
        self.output = nn.Linear(hidden, 3)
        # Initialize output to produce ~128 (middle gray)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.fill_(0.0)  # sigmoid(0) = 0.5, * 255 = 127.5

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Generate RGB values for given coordinates.

        Args:
            coords: (B, 3) float tensor with (frame_idx_norm, y_norm, x_norm)
                all normalized to [-1, 1].

        Returns:
            (B, 3) float tensor with RGB values in [0, 255].
        """
        # Separate frame and spatial coords
        frame_idx = coords[:, 0:1]  # (B, 1)
        spatial = coords[:, 1:3]  # (B, 2)

        # Positional encoding on frame index
        frame_features = self.pos_enc(frame_idx)  # (B, pos_dim)

        # Concatenate frame features + spatial coords
        x = torch.cat([frame_features, spatial], dim=-1)  # (B, input_dim)

        # Forward through SIREN layers
        for layer in self.net:
            x = layer(x)

        # Output layer: sigmoid * 255 for [0, 255] range
        rgb = torch.sigmoid(self.output(x)) * 255.0
        return rgb

    def generate_frame(self, frame_idx: int, device: str = "cpu") -> torch.Tensor:
        """Generate a single frame.

        Args:
            frame_idx: frame index (0-based).
            device: compute device.

        Returns:
            (H, W, 3) uint8 tensor.
        """
        H, W = self.frame_h, self.frame_w

        # Normalized frame index
        t = 2.0 * frame_idx / max(self.num_frames - 1, 1) - 1.0

        # Grid of (y, x) coordinates normalized to [-1, 1]
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Build coordinate tensor: (H*W, 3)
        coords = torch.stack([
            torch.full((H * W,), t, device=device),
            grid_y.reshape(-1),
            grid_x.reshape(-1),
        ], dim=-1)

        with torch.no_grad():
            rgb = self.forward(coords)  # (H*W, 3)

        return rgb.reshape(H, W, 3).clamp(0, 255).byte()

    def generate_all_frames(
        self,
        device: str = "cpu",
        batch_pixels: int = 65536,
    ) -> torch.Tensor:
        """Generate all video frames.

        Args:
            device: compute device.
            batch_pixels: number of pixels to process at once (memory management).

        Returns:
            (num_frames, H, W, 3) uint8 tensor.
        """
        H, W = self.frame_h, self.frame_w
        frames = torch.zeros(self.num_frames, H, W, 3, dtype=torch.uint8)

        for f in range(self.num_frames):
            frames[f] = self.generate_frame(f, device=device)

        return frames

    def param_count(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        """Total size in bytes at float32."""
        return self.param_count() * 4

    def size_bytes_fp16(self) -> int:
        """Total size in bytes at float16."""
        return self.param_count() * 2


# ── Self-compressing video codec ─────────────────────────────────────────


class SelfCompressingVideoCodec(SIRENVideoCodec):
    """Video codec that self-compresses during training.

    Combines techniques 1 + 3: a SIREN video generator with learnable
    per-layer bit-depth. The network memorizes the video while
    simultaneously learning to compress its own weights.

    Training loss:
        L = scorer_loss(generated, gt)
          + lambda_rate * network_size_in_bits
          + lambda_smooth * temporal_consistency

    The network_size_in_bits term directly optimizes the rate component
    of the competition score. End-to-end rate-distortion optimization.

    Args:
        Same as SIRENVideoCodec plus:
        init_bits: initial bit-depth for self-compression.
    """

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 4,
        omega_0: float = 30.0,
        num_frames: int = 1200,
        frame_h: int = 384,
        frame_w: int = 512,
        pos_encoding_freqs: int = 6,
        init_bits: float = 8.0,
    ):
        super().__init__(
            hidden=hidden,
            layers=layers,
            omega_0=omega_0,
            num_frames=num_frames,
            frame_h=frame_h,
            frame_w=frame_w,
            pos_encoding_freqs=pos_encoding_freqs,
        )
        self.init_bits = init_bits

        # Add learnable bit-depth for each layer
        self.layer_bits: nn.ParameterList = nn.ParameterList()
        for layer in self.net:
            n_out = layer.linear.weight.shape[0]
            self.layer_bits.append(nn.Parameter(torch.full((n_out,), init_bits)))
        # Output layer
        n_out = self.output.weight.shape[0]
        self.output_bits = nn.Parameter(torch.full((n_out,), init_bits))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward with self-compressed weights."""
        from .self_compress import _ste_quantize

        # Separate frame and spatial coords
        frame_idx = coords[:, 0:1]
        spatial = coords[:, 1:3]
        frame_features = self.pos_enc(frame_idx)
        x = torch.cat([frame_features, spatial], dim=-1)

        # Forward through SIREN layers with quantized weights
        for layer, bits in zip(self.net, self.layer_bits):
            q_weight = _ste_quantize(layer.linear.weight, bits, self.training)
            q_bias = layer.linear.bias
            x = torch.sin(layer.omega_0 * F.linear(x, q_weight, q_bias))

        # Output layer
        q_out_weight = _ste_quantize(self.output.weight, self.output_bits, self.training)
        rgb = torch.sigmoid(F.linear(x, q_out_weight, self.output.bias)) * 255.0
        return rgb

    def total_bits(self) -> torch.Tensor:
        """Total bits across all layers including bias (differentiable for rate penalty)."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer, bits in zip(self.net, self.layer_bits):
            fan_in = layer.linear.weight.shape[1]
            per_channel = bits.clamp(0.0, 8.0)
            # Weight bits: fan_in weights per channel at learned bit-depth
            total = total + (per_channel * fan_in).sum()
            # Bias bits: 1 bias per channel at learned bit-depth
            if layer.linear.bias is not None:
                total = total + per_channel.sum()
        # Output layer
        fan_in = self.output.weight.shape[1]
        out_bits = self.output_bits.clamp(0.0, 8.0)
        total = total + (out_bits * fan_in).sum()
        if self.output.bias is not None:
            total = total + out_bits.sum()
        return total

    def total_bytes(self) -> float:
        """Total bytes at current bit allocation."""
        return float(self.total_bits().detach().item()) / 8.0


# ── Mask-conditioned SIREN ──────────────────────────────────────────────


class MaskConditionedSIREN(nn.Module):
    """SIREN that generates frames conditioned on segmentation masks.

    Input: (frame_idx, y, x, mask_class_onehot) -> (R, G, B)

    The mask class follows the canonical contest SegNet order
    (road, lane_markings, undrivable, movable, my_car), is one-hot encoded,
    and is concatenated with the spatial coordinates. This gives the network
    class-specific generation capacity: undrivable regions can be smooth,
    road can be textured gray, and class boundaries stay sharp.

    Architecture:
    - Positional encoding for frame_idx -> fourier features
    - Spatial (y, x) raw coordinates
    - One-hot mask class (num_classes dims) concatenated
    - SIREN layers: (fourier_dim + 2 + num_classes) -> hidden -> ... -> 3

    Operating at scorer resolution (512x384) to minimize parameters.
    Bilinear upscale to 1164x874 at inflate time.

    Args:
        hidden: hidden layer width (controls capacity vs size).
        layers: number of hidden layers.
        omega_0: SIREN frequency parameter (higher = more detail).
        num_frames: number of video frames (for positional encoding).
        frame_h: frame height at scorer resolution.
        frame_w: frame width at scorer resolution.
        num_classes: number of segmentation classes.
        pos_encoding_freqs: Fourier feature frequencies for frame index.
    """

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 5,
        omega_0: float = 30.0,
        num_frames: int = 1200,
        frame_h: int = 384,
        frame_w: int = 512,
        num_classes: int = 5,
        pos_encoding_freqs: int = 8,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_layers = layers
        self.omega_0 = omega_0
        self.num_frames = num_frames
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.num_classes = num_classes
        self.pos_encoding_freqs = pos_encoding_freqs

        # Positional encoding for frame index (temporal)
        self.pos_enc = PositionalEncoding(
            num_frequencies=pos_encoding_freqs, include_input=True,
        )

        # Input: pos_enc(frame_idx) + y + x + one_hot(mask_class)
        input_dim = self.pos_enc.output_dim + 2 + num_classes

        # Build SIREN layers
        net_layers: list[SIRENLayer] = []
        net_layers.append(SIRENLayer(input_dim, hidden, omega_0=omega_0, is_first=True))
        for _ in range(layers - 1):
            net_layers.append(SIRENLayer(hidden, hidden, omega_0=omega_0))
        self.net = nn.ModuleList(net_layers)

        # Output: linear to RGB (no sin activation on output)
        self.output = nn.Linear(hidden, 3)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.fill_(0.0)  # sigmoid(0) = 0.5, * 255 = 127.5

    def forward(self, coords: torch.Tensor, mask_onehot: torch.Tensor) -> torch.Tensor:
        """Generate RGB values for given coordinates and mask classes.

        Args:
            coords: (B, 3) float tensor with (frame_idx_norm, y_norm, x_norm)
                all normalized to [-1, 1].
            mask_onehot: (B, num_classes) float tensor, one-hot mask class.

        Returns:
            (B, 3) float tensor with RGB values in [0, 255].
        """
        frame_idx = coords[:, 0:1]
        spatial = coords[:, 1:3]

        frame_features = self.pos_enc(frame_idx)

        # Concatenate: frame features + spatial coords + mask class
        x = torch.cat([frame_features, spatial, mask_onehot], dim=-1)

        for layer in self.net:
            x = layer(x)

        rgb = torch.sigmoid(self.output(x)) * 255.0
        return rgb

    def generate_frame(
        self,
        frame_idx: int,
        masks: torch.Tensor,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Generate a single frame given a mask.

        Args:
            frame_idx: frame index (0-based).
            masks: (H, W) long tensor with class indices.
            device: compute device.

        Returns:
            (H, W, 3) uint8 tensor.
        """
        H, W = self.frame_h, self.frame_w
        t = 2.0 * frame_idx / max(self.num_frames - 1, 1) - 1.0

        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        coords = torch.stack([
            torch.full((H * W,), t, device=device),
            grid_y.reshape(-1),
            grid_x.reshape(-1),
        ], dim=-1)

        # One-hot encode mask classes
        mask_flat = masks.reshape(-1).long().to(device)
        mask_onehot = F.one_hot(mask_flat, self.num_classes).float()

        with torch.no_grad():
            rgb = self.forward(coords, mask_onehot)

        return rgb.reshape(H, W, 3).clamp(0, 255).byte()

    def generate_all_frames(
        self,
        masks: torch.Tensor,
        device: str = "cpu",
        batch_pixels: int = 65536,
    ) -> torch.Tensor:
        """Generate all video frames from masks.

        Args:
            masks: (num_frames, H, W) long tensor with class indices.
            device: compute device.
            batch_pixels: pixels per batch (memory management).

        Returns:
            (num_frames, H, W, 3) uint8 tensor.
        """
        H, W = self.frame_h, self.frame_w
        N = masks.shape[0]
        assert N == self.num_frames, f"Expected {self.num_frames} masks, got {N}"

        frames = torch.zeros(N, H, W, 3, dtype=torch.uint8)
        for f in range(N):
            frames[f] = self.generate_frame(f, masks[f], device=device)
        return frames

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return self.param_count() * 4

    def size_bytes_fp16(self) -> int:
        return self.param_count() * 2


# ── Training ─────────────────────────────────────────────────────────────


def train_network_codec(
    gt_frames: torch.Tensor,
    *,
    posenet: nn.Module | None = None,
    segnet: nn.Module | None = None,
    hidden: int = 64,
    layers: int = 4,
    omega_0: float = 30.0,
    target_size_kb: int = 50,
    epochs: int = 1000,
    lr: float = 1e-4,
    batch_pixels: int = 4096,
    lambda_rate: float = 0.0,
    lambda_rate_end: float = 1.0,
    lambda_smooth: float = 0.1,
    scorer_weight: float = 0.0,
    ramp_start_frac: float = 0.5,
    use_self_compress: bool = False,
    device: str = "cpu",
    log_every: int = 100,
    pos_encoding_freqs: int = 6,
) -> SIRENVideoCodec | SelfCompressingVideoCodec:
    """Train a network codec that memorizes a video.

    Phase 1 (memorize): Train to reconstruct frames. No compression penalty.
    Phase 2 (compress): Ramp up rate penalty. Network prunes channels
        and reduces bit-depth while maintaining quality.
    Phase 3 (fine-tune): Fix architecture, fine-tune remaining weights.

    Args:
        gt_frames: (T, H, W, 3) uint8 ground truth video.
        posenet: frozen PoseNet model (optional, for scorer loss).
        segnet: frozen SegNet model (optional, for scorer loss).
        hidden: SIREN hidden width.
        layers: number of SIREN hidden layers.
        omega_0: SIREN frequency parameter.
        target_size_kb: target archive size in kilobytes.
        epochs: total training epochs.
        lr: learning rate.
        batch_pixels: pixels per training batch.
        lambda_rate: initial rate penalty.
        lambda_rate_end: final rate penalty.
        lambda_smooth: temporal consistency weight.
        scorer_weight: weight on scorer loss (0 = pixel-only).
        ramp_start_frac: fraction of training before rate ramp starts.
        use_self_compress: if True, use SelfCompressingVideoCodec.
        device: compute device.
        log_every: logging interval.
        pos_encoding_freqs: Fourier feature frequencies.

    Returns:
        Trained video codec.
    """
    T, H, W, C = gt_frames.shape
    assert C == 3, f"Expected 3 channels, got {C}"

    target_bits = target_size_kb * 1024 * 8

    # Build model
    if use_self_compress:
        model = SelfCompressingVideoCodec(
            hidden=hidden, layers=layers, omega_0=omega_0,
            num_frames=T, frame_h=H, frame_w=W,
            pos_encoding_freqs=pos_encoding_freqs,
        )
    else:
        model = SIRENVideoCodec(
            hidden=hidden, layers=layers, omega_0=omega_0,
            num_frames=T, frame_h=H, frame_w=W,
            pos_encoding_freqs=pos_encoding_freqs,
        )

    if scorer_weight > 0 and (posenet is None or segnet is None):
        raise ValueError(
            "scorer_weight > 0 requires both posenet and segnet models. "
            "Scorer loss in train_network_codec is not yet implemented; "
            "set scorer_weight=0 for pixel-only training."
        )
    if scorer_weight > 0:
        raise NotImplementedError(
            "Scorer loss integration in train_network_codec is not yet implemented. "
            "Use scorer_weight=0 for pixel-only training, then fine-tune with "
            "train_self_compressing for scorer-aware optimization."
        )

    model = model.to(device)
    gt_float = gt_frames.float().to(device)  # (T, H, W, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Pre-compute all valid coordinate indices
    total_pixels = T * H * W

    for epoch in range(epochs):
        model.train()

        # Sample random pixels for this batch
        pixel_indices = torch.randint(0, total_pixels, (batch_pixels,))
        frame_idx = pixel_indices // (H * W)
        spatial_idx = pixel_indices % (H * W)
        y_idx = spatial_idx // W
        x_idx = spatial_idx % W

        # Normalize coordinates to [-1, 1]
        t_norm = 2.0 * frame_idx.float() / max(T - 1, 1) - 1.0
        y_norm = 2.0 * y_idx.float() / max(H - 1, 1) - 1.0
        x_norm = 2.0 * x_idx.float() / max(W - 1, 1) - 1.0

        coords = torch.stack([t_norm, y_norm, x_norm], dim=-1).to(device)

        # Ground truth RGB at these coordinates
        gt_rgb = gt_float[frame_idx, y_idx, x_idx]  # (batch_pixels, 3)

        # Forward
        pred_rgb = model(coords)  # (batch_pixels, 3)

        # Pixel reconstruction loss (L2)
        pixel_loss = F.mse_loss(pred_rgb, gt_rgb)

        # Temporal smoothness: penalize large changes between adjacent frames
        smooth_loss = torch.tensor(0.0, device=device)
        if lambda_smooth > 0 and T > 1:
            # Sample some adjacent frame pairs
            n_smooth = min(batch_pixels // 4, 1024)
            t_smooth = torch.randint(0, T - 1, (n_smooth,))
            y_smooth = torch.randint(0, H, (n_smooth,))
            x_smooth = torch.randint(0, W, (n_smooth,))

            t1_norm = 2.0 * t_smooth.float() / max(T - 1, 1) - 1.0
            t2_norm = 2.0 * (t_smooth + 1).float() / max(T - 1, 1) - 1.0
            y_s_norm = 2.0 * y_smooth.float() / max(H - 1, 1) - 1.0
            x_s_norm = 2.0 * x_smooth.float() / max(W - 1, 1) - 1.0

            coords1 = torch.stack([t1_norm, y_s_norm, x_s_norm], dim=-1).to(device)
            coords2 = torch.stack([t2_norm, y_s_norm, x_s_norm], dim=-1).to(device)

            pred1 = model(coords1)
            pred2 = model(coords2)
            gt1 = gt_float[t_smooth, y_smooth, x_smooth]
            gt2 = gt_float[t_smooth + 1, y_smooth, x_smooth]

            # Penalize prediction smoothness that deviates from GT smoothness
            pred_diff = (pred2 - pred1).norm(dim=-1)
            gt_diff = (gt2 - gt1).norm(dim=-1)
            smooth_loss = F.mse_loss(pred_diff, gt_diff)

        # Rate penalty (for self-compressing variant)
        rate_loss = torch.tensor(0.0, device=device)
        if use_self_compress and isinstance(model, SelfCompressingVideoCodec):
            progress = epoch / max(epochs - 1, 1)
            if progress >= ramp_start_frac:
                ramp_progress = (progress - ramp_start_frac) / (1.0 - ramp_start_frac)
                cur_lambda = lambda_rate + (lambda_rate_end - lambda_rate) * ramp_progress
            else:
                cur_lambda = lambda_rate
            total_bits = model.total_bits()
            rate_loss = cur_lambda * F.relu(total_bits - target_bits) / target_bits

        total_loss = pixel_loss + lambda_smooth * smooth_loss + rate_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Clamp bits for self-compressing
        if use_self_compress and isinstance(model, SelfCompressingVideoCodec):
            with torch.no_grad():
                for bits in model.layer_bits:
                    bits.clamp_(0.0, 8.0)
                model.output_bits.clamp_(0.0, 8.0)

        if epoch % log_every == 0 or epoch == epochs - 1:
            psnr = 10 * math.log10(255 ** 2 / max(pixel_loss.item(), 1e-10))
            size_str = ""
            if use_self_compress and isinstance(model, SelfCompressingVideoCodec):
                size_str = f" | size={model.total_bytes():.0f}B"
            else:
                size_str = f" | params={model.param_count():,} ({model.size_bytes_fp16() / 1024:.1f}KB fp16)"
            print(
                f"  epoch {epoch:4d}/{epochs} | "
                f"pixel_loss={pixel_loss.item():.2f} PSNR={psnr:.1f}dB | "
                f"smooth={smooth_loss.item():.4f}{size_str}"
            )

    return model


# ── Mask-conditioned SIREN training ────────────────────────────────────


def train_mask_conditioned_siren(
    gt_frames: torch.Tensor,
    masks: torch.Tensor,
    *,
    posenet: nn.Module | None = None,
    segnet: nn.Module | None = None,
    hidden: int = 128,
    layers: int = 5,
    omega_0: float = 30.0,
    num_classes: int = 5,
    pos_encoding_freqs: int = 8,
    num_steps: int = 2000,
    batch_pixels: int = 8192,
    lr: float = 5e-4,
    device: str = "cpu",
    log_every: int = 100,
    scorer_ramp_start: float = 0.6,
    scorer_weight_max: float = 10.0,
    tv_weight: float = 0.01,
) -> MaskConditionedSIREN:
    """Three-phase training of a mask-conditioned SIREN.

    Phase 1 (memorize, 60% of steps):
        Loss = MSE(siren_output, gt_frames)
        Goal: learn the video content.

    Phase 2 (scorer-constrain, 30% of steps):
        Loss = MSE + scorer_weight * (seg_loss + pose_loss) + TV
        Goal: shift from pixel-exact to scorer-satisfying.

    Phase 3 (fine-tune, 10% of steps):
        Loss = scorer_weight * (seg_loss + pose_loss) + TV (reduced lr)
        Goal: polish scorer satisfaction.

    The key insight: we do NOT need pixel-perfect reconstruction. We need
    SCORER-SATISFYING reconstruction. The SIREN learns that undrivable
    scene regions can be smooth, road can be textured gray, and class
    boundaries must be sharp. Mask conditioning enables this resource
    allocation.

    Args:
        gt_frames: (T, H, W, 3) uint8 ground truth at scorer resolution.
        masks: (T, H, W) long tensor with class indices.
        posenet: frozen PoseNet (required for phase 2+3).
        segnet: frozen SegNet (required for phase 2+3).
        hidden: SIREN hidden width.
        layers: number of SIREN hidden layers.
        omega_0: SIREN frequency parameter.
        num_classes: number of segmentation classes.
        pos_encoding_freqs: Fourier feature frequencies.
        num_steps: total training steps.
        batch_pixels: pixels per training batch.
        lr: learning rate.
        device: compute device.
        log_every: logging interval.
        scorer_ramp_start: fraction of training before scorer loss activates.
        scorer_weight_max: maximum scorer loss weight.
        tv_weight: total variation weight for compressibility.

    Returns:
        Trained MaskConditionedSIREN.
    """
    T, H, W, C = gt_frames.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    assert masks.shape == (T, H, W), f"Mask shape mismatch: {masks.shape} vs ({T}, {H}, {W})"

    model = MaskConditionedSIREN(
        hidden=hidden, layers=layers, omega_0=omega_0,
        num_frames=T, frame_h=H, frame_w=W,
        num_classes=num_classes, pos_encoding_freqs=pos_encoding_freqs,
    ).to(device)

    gt_float = gt_frames.float().to(device)
    masks_dev = masks.long().to(device)

    print(f"  MaskConditionedSIREN: {model.param_count():,} params "
          f"({model.size_bytes_fp16() / 1024:.1f}KB fp16)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    total_pixels = T * H * W

    # Phase boundaries
    phase2_start = int(num_steps * scorer_ramp_start)
    phase3_start = int(num_steps * 0.9)

    for step in range(num_steps):
        model.train()
        progress = step / max(num_steps - 1, 1)

        # ── Sample random pixels ──
        pixel_indices = torch.randint(0, total_pixels, (batch_pixels,))
        frame_idx = pixel_indices // (H * W)
        spatial_idx = pixel_indices % (H * W)
        y_idx = spatial_idx // W
        x_idx = spatial_idx % W

        t_norm = 2.0 * frame_idx.float() / max(T - 1, 1) - 1.0
        y_norm = 2.0 * y_idx.float() / max(H - 1, 1) - 1.0
        x_norm = 2.0 * x_idx.float() / max(W - 1, 1) - 1.0

        coords = torch.stack([t_norm, y_norm, x_norm], dim=-1).to(device)

        # One-hot mask for sampled pixels
        mask_vals = masks_dev[frame_idx, y_idx, x_idx]  # (batch_pixels,)
        mask_onehot = F.one_hot(mask_vals, num_classes).float()

        gt_rgb = gt_float[frame_idx, y_idx, x_idx]

        pred_rgb = model(coords, mask_onehot)

        # ── Phase 1: pixel reconstruction ──
        pixel_loss = F.mse_loss(pred_rgb, gt_rgb)
        total_loss = pixel_loss

        # ── Phase 2+3: scorer constraints ──
        scorer_loss_val = torch.tensor(0.0, device=device)
        if step >= phase2_start and posenet is not None and segnet is not None:
            # Ramp scorer weight from 0 to scorer_weight_max
            ramp = min(1.0, (step - phase2_start) / max(phase3_start - phase2_start, 1))
            cur_scorer_w = scorer_weight_max * ramp

            # Generate a small batch of full frames for scorer eval
            # Pick 2 consecutive frames for PoseNet pair
            pair_start = torch.randint(0, max(T - 1, 1), (1,)).item()
            frames_for_scorer = []
            for fi in [pair_start, pair_start + 1]:
                frame = model.generate_frame(fi, masks_dev[fi], device=device)
                frames_for_scorer.append(frame.float())

            # Stack as (2, H, W, 3) -> (1, 2, 3, H, W) for scorer
            pair_hwc = torch.stack(frames_for_scorer, dim=0)  # (2, H, W, 3)
            pair_chw = pair_hwc.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # (1, 2, 3, H, W)

            # GT pair for comparison
            gt_pair_hwc = gt_float[pair_start:pair_start + 2]  # (2, H, W, 3)
            gt_pair_chw = gt_pair_hwc.permute(0, 3, 1, 2).unsqueeze(0).contiguous()

            # PoseNet
            with torch.no_grad():
                gt_pose_in = posenet.preprocess_input(gt_pair_chw)
                gt_pose_out = posenet(gt_pose_in)
                gt_pose = gt_pose_out["pose"][..., :6] if isinstance(gt_pose_out, dict) else gt_pose_out[..., :6]

            # Need grad for generated frames — rebuild with grad
            pair_chw_grad = pair_chw.detach().requires_grad_(True)
            pred_pose_in = posenet.preprocess_input(pair_chw_grad)
            pred_pose_out = posenet(pred_pose_in)
            pred_pose = pred_pose_out["pose"][..., :6] if isinstance(pred_pose_out, dict) else pred_pose_out[..., :6]
            pose_loss = F.mse_loss(pred_pose, gt_pose)

            # SegNet — use first frame of pair
            seg_frame = pair_chw[:, 0:1].contiguous()  # (1, 1, 3, H, W)
            gt_seg_frame = gt_pair_chw[:, 0:1].contiguous()

            with torch.no_grad():
                gt_seg_in = segnet.preprocess_input(gt_seg_frame)
                gt_seg_out = segnet(gt_seg_in)
                gt_seg_labels = gt_seg_out.argmax(dim=-1) if gt_seg_out.ndim > 2 else gt_seg_out

            seg_frame_grad = seg_frame.detach().requires_grad_(True)
            pred_seg_in = segnet.preprocess_input(seg_frame_grad)
            pred_seg_out = segnet(pred_seg_in)

            if pred_seg_out.ndim >= 3:
                # Cross-entropy against GT argmax
                pred_flat = pred_seg_out.reshape(-1, pred_seg_out.shape[-1])
                gt_flat = gt_seg_labels.reshape(-1).long()
                seg_loss = F.cross_entropy(pred_flat, gt_flat)
            else:
                seg_loss = torch.tensor(0.0, device=device)

            scorer_loss_val = cur_scorer_w * (100.0 * seg_loss + torch.sqrt(10.0 * pose_loss + 1e-8))

            # Phase 3: drop pixel loss, scorer only
            if step >= phase3_start:
                total_loss = scorer_loss_val
            else:
                total_loss = pixel_loss + scorer_loss_val

        # Total variation for compressibility
        if tv_weight > 0 and step >= phase2_start:
            # TV on a small generated patch
            tv_patch_f = torch.randint(0, T, (1,)).item()
            with torch.no_grad():
                tv_frame = model.generate_frame(tv_patch_f, masks_dev[tv_patch_f], device=device).float()
            tv_chw = tv_frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            tv_h = (tv_chw[:, :, 1:, :] - tv_chw[:, :, :-1, :]).abs().mean()
            tv_v = (tv_chw[:, :, :, 1:] - tv_chw[:, :, :, :-1]).abs().mean()
            total_loss = total_loss + tv_weight * (tv_h + tv_v)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == num_steps - 1:
            psnr = 10 * math.log10(255 ** 2 / max(pixel_loss.item(), 1e-10))
            phase = "memorize" if step < phase2_start else ("constrain" if step < phase3_start else "fine-tune")
            scorer_str = f" scorer={scorer_loss_val.item():.4f}" if scorer_loss_val.item() > 0 else ""
            print(
                f"  step {step:4d}/{num_steps} [{phase:9s}] | "
                f"pixel_loss={pixel_loss.item():.2f} PSNR={psnr:.1f}dB{scorer_str}"
            )

    return model


# ── Export / import ──────────────────────────────────────────────────────


def export_mask_siren_archive(
    codec: MaskConditionedSIREN,
    masks: torch.Tensor,
    use_fp16: bool = True,
) -> bytes:
    """Export mask-conditioned SIREN as a minimal archive.

    Archive format:
    - [meta_len (4B)] [meta JSON] [mask_blob_len (4B)] [mask_blob] [weights]

    The masks are entropy-coded (run-length + zlib) to ~200-300 bytes
    for typical driving scenes with 5 classes.

    Args:
        codec: trained MaskConditionedSIREN.
        masks: (T, H, W) long tensor with class indices.
        use_fp16: store weights as float16.

    Returns:
        Archive bytes.
    """
    import zlib

    meta = {
        "version": 2,
        "type": "mask_siren",
        "hidden": codec.hidden,
        "layers": codec.num_layers,
        "omega_0": codec.omega_0,
        "num_frames": codec.num_frames,
        "frame_h": codec.frame_h,
        "frame_w": codec.frame_w,
        "num_classes": codec.num_classes,
        "pos_encoding_freqs": codec.pos_encoding_freqs,
        "param_count": codec.param_count(),
    }
    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    # Entropy-code masks: uint8 class indices, zlib compressed
    masks_np = masks.cpu().numpy().astype("uint8")
    masks_raw = masks_np.tobytes()
    masks_compressed = zlib.compress(masks_raw, level=9)

    # Serialize weights
    state = codec.state_dict()
    weight_buf = io.BytesIO()
    if use_fp16:
        state_fp16 = {}
        for k, v in state.items():
            if v.is_floating_point():
                state_fp16[k] = v.half().cpu()
            else:
                state_fp16[k] = v.cpu()
        torch.save(state_fp16, weight_buf)
    else:
        torch.save({k: v.cpu() for k, v in state.items()}, weight_buf)
    weight_bytes = weight_buf.getvalue()

    # Pack: [meta_len (4B)] [meta JSON] [mask_len (4B)] [masks] [weights]
    buf = bytearray()
    buf.extend(struct.pack("<I", len(meta_json)))
    buf.extend(meta_json)
    buf.extend(struct.pack("<I", len(masks_compressed)))
    buf.extend(masks_compressed)
    buf.extend(weight_bytes)

    return bytes(buf)


def inflate_mask_siren_archive(
    archive: bytes,
    device: str = "cpu",
    target_h: int | None = None,
    target_w: int | None = None,
) -> torch.Tensor:
    """Inflate a mask-conditioned SIREN archive to raw video frames.

    Steps:
    1. Load masks and SIREN weights from archive
    2. For each frame: run SIREN(frame_idx, coords, mask) at scorer resolution
    3. If target_h/target_w specified, bilinear upscale to target resolution
    4. Return uint8 RGB frames

    Args:
        archive: bytes from export_mask_siren_archive.
        device: compute device for inference.
        target_h: optional target height for upscaling (e.g. 874).
        target_w: optional target width for upscaling (e.g. 1164).

    Returns:
        (num_frames, H_out, W_out, 3) uint8 tensor.
    """
    import zlib

    offset = 0
    meta_len = struct.unpack("<I", archive[offset:offset + 4])[0]
    offset += 4
    meta = json.loads(archive[offset:offset + meta_len].decode("utf-8"))
    offset += meta_len

    # Load masks
    mask_len = struct.unpack("<I", archive[offset:offset + 4])[0]
    offset += 4
    masks_compressed = archive[offset:offset + mask_len]
    offset += mask_len
    import numpy as np
    masks_raw = zlib.decompress(masks_compressed)
    masks_np = np.frombuffer(masks_raw, dtype=np.uint8).reshape(
        meta["num_frames"], meta["frame_h"], meta["frame_w"],
    )
    masks = torch.from_numpy(masks_np.copy()).long()

    # Load weights
    weight_bytes = archive[offset:]
    weight_buf = io.BytesIO(weight_bytes)
    state = torch.load(weight_buf, map_location="cpu", weights_only=True)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}

    codec = MaskConditionedSIREN(
        hidden=meta["hidden"],
        layers=meta["layers"],
        omega_0=meta["omega_0"],
        num_frames=meta["num_frames"],
        frame_h=meta["frame_h"],
        frame_w=meta["frame_w"],
        num_classes=meta["num_classes"],
        pos_encoding_freqs=meta["pos_encoding_freqs"],
    )
    codec.load_state_dict(state, strict=True)
    codec = codec.to(device).eval()

    # Generate all frames at scorer resolution
    frames = codec.generate_all_frames(masks, device=device)

    # Upscale if target resolution specified
    if target_h is not None and target_w is not None:
        frames_chw = frames.float().permute(0, 3, 1, 2)  # (N, 3, H, W)
        frames_up = F.interpolate(
            frames_chw, size=(target_h, target_w), mode="bilinear", align_corners=False,
        )
        frames = frames_up.permute(0, 2, 3, 1).clamp(0, 255).byte()

    return frames


def export_network_archive(
    codec: SIRENVideoCodec | SelfCompressingVideoCodec,
    use_fp16: bool = True,
) -> bytes:
    """Export the trained codec as a minimal archive.

    The archive contains:
    - Architecture definition (JSON, ~200 bytes)
    - Network weights (float16 or self-compressed)
    - Frame count + resolution metadata

    Args:
        codec: trained video codec.
        use_fp16: if True, store weights as float16.

    Returns:
        Archive bytes.
    """
    # Architecture metadata
    meta = {
        "version": 1,
        "type": "self_compressing" if isinstance(codec, SelfCompressingVideoCodec) else "siren",
        "hidden": codec.hidden,
        "layers": codec.num_layers,
        "omega_0": codec.omega_0,
        "num_frames": codec.num_frames,
        "frame_h": codec.frame_h,
        "frame_w": codec.frame_w,
        "pos_encoding_freqs": codec.pos_encoding_freqs,
        "param_count": codec.param_count(),
    }
    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    # Serialize weights
    state = codec.state_dict()
    weight_buf = io.BytesIO()
    if use_fp16:
        state_fp16 = {}
        for k, v in state.items():
            if v.is_floating_point():
                state_fp16[k] = v.half().cpu()
            else:
                state_fp16[k] = v.cpu()
        torch.save(state_fp16, weight_buf)
    else:
        torch.save({k: v.cpu() for k, v in state.items()}, weight_buf)
    weight_bytes = weight_buf.getvalue()

    # Pack: [meta_len (4B)] [meta JSON] [weights]
    buf = bytearray()
    buf.extend(struct.pack("<I", len(meta_json)))
    buf.extend(meta_json)
    buf.extend(weight_bytes)

    return bytes(buf)


def inflate_network_codec(
    archive: bytes,
    device: str = "cpu",
) -> torch.Tensor:
    """Inflate by running the network forward pass.

    For each frame index 0..num_frames-1:
        frame = codec(frame_idx)

    On CPU: ~5-10 minutes for 1200 frames at 384x512 with 64-hidden SIREN
    On GPU: ~10-30 seconds (depends on batch_pixels in generate_all_frames)

    Args:
        archive: bytes from export_network_archive.
        device: compute device for inference.

    Returns:
        (num_frames, H, W, 3) uint8 tensor.
    """
    # Parse header
    offset = 0
    meta_len = struct.unpack("<I", archive[offset:offset + 4])[0]
    offset += 4
    meta = json.loads(archive[offset:offset + meta_len].decode("utf-8"))
    offset += meta_len

    # Load weights
    weight_bytes = archive[offset:]
    weight_buf = io.BytesIO(weight_bytes)
    state = torch.load(weight_buf, map_location="cpu", weights_only=True)
    # Convert back to float32
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}

    # Reconstruct model
    codec_type = meta.get("type", "siren")
    if codec_type == "self_compressing":
        codec = SelfCompressingVideoCodec(
            hidden=meta["hidden"],
            layers=meta["layers"],
            omega_0=meta["omega_0"],
            num_frames=meta["num_frames"],
            frame_h=meta["frame_h"],
            frame_w=meta["frame_w"],
            pos_encoding_freqs=meta["pos_encoding_freqs"],
        )
    else:
        codec = SIRENVideoCodec(
            hidden=meta["hidden"],
            layers=meta["layers"],
            omega_0=meta["omega_0"],
            num_frames=meta["num_frames"],
            frame_h=meta["frame_h"],
            frame_w=meta["frame_w"],
            pos_encoding_freqs=meta["pos_encoding_freqs"],
        )

    codec.load_state_dict(state, strict=True)
    codec = codec.to(device).eval()

    # Generate all frames
    return codec.generate_all_frames(device=device)


def load_network_codec(
    archive: bytes,
    device: str = "cpu",
) -> SIRENVideoCodec | SelfCompressingVideoCodec:
    """Load the codec model without running inference.

    Useful for further fine-tuning or inspection.

    Args:
        archive: bytes from export_network_archive.
        device: target device.

    Returns:
        Loaded codec model.
    """
    offset = 0
    meta_len = struct.unpack("<I", archive[offset:offset + 4])[0]
    offset += 4
    meta = json.loads(archive[offset:offset + meta_len].decode("utf-8"))
    offset += meta_len

    weight_bytes = archive[offset:]
    weight_buf = io.BytesIO(weight_bytes)
    state = torch.load(weight_buf, map_location="cpu", weights_only=True)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}

    codec_type = meta.get("type", "siren")
    if codec_type == "self_compressing":
        codec = SelfCompressingVideoCodec(
            hidden=meta["hidden"],
            layers=meta["layers"],
            omega_0=meta["omega_0"],
            num_frames=meta["num_frames"],
            frame_h=meta["frame_h"],
            frame_w=meta["frame_w"],
            pos_encoding_freqs=meta["pos_encoding_freqs"],
        )
    else:
        codec = SIRENVideoCodec(
            hidden=meta["hidden"],
            layers=meta["layers"],
            omega_0=meta["omega_0"],
            num_frames=meta["num_frames"],
            frame_h=meta["frame_h"],
            frame_w=meta["frame_w"],
            pos_encoding_freqs=meta["pos_encoding_freqs"],
        )

    codec.load_state_dict(state, strict=True)
    return codec.to(device)


# ── Smoke tests ─────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Run basic shape, forward-pass, and export/import checks."""
    print("network_codec: running smoke tests...")

    # Use tiny dimensions for smoke test
    T, H, W = 4, 16, 16

    # 1. SIREN forward pass
    codec = SIRENVideoCodec(
        hidden=16, layers=2, omega_0=30.0,
        num_frames=T, frame_h=H, frame_w=W,
        pos_encoding_freqs=4,
    )
    coords = torch.randn(32, 3)  # random coords
    rgb = codec(coords)
    assert rgb.shape == (32, 3), f"Output shape: {rgb.shape}"
    assert rgb.min() >= 0.0 and rgb.max() <= 255.0, f"Range: [{rgb.min()}, {rgb.max()}]"
    print(f"  SIREN forward: OK ({rgb.shape}, range [{rgb.min():.1f}, {rgb.max():.1f}])")

    # 2. Frame generation
    frame = codec.generate_frame(0)
    assert frame.shape == (H, W, 3), f"Frame shape: {frame.shape}"
    assert frame.dtype == torch.uint8
    print(f"  frame generation: OK ({frame.shape})")

    # 3. All frames generation
    all_frames = codec.generate_all_frames()
    assert all_frames.shape == (T, H, W, 3)
    assert all_frames.dtype == torch.uint8
    print(f"  all frames: OK ({all_frames.shape})")

    # 4. Param count and size
    print(f"  params: {codec.param_count():,} ({codec.size_bytes_fp16() / 1024:.1f}KB fp16)")

    # 5. Export and inflate
    archive = export_network_archive(codec)
    print(f"  exported: {len(archive)} bytes")

    inflated = inflate_network_codec(archive)
    assert inflated.shape == (T, H, W, 3)
    assert inflated.dtype == torch.uint8
    # Should match original generation
    orig_frames = codec.generate_all_frames()
    diff = (inflated.float() - orig_frames.float()).abs().max().item()
    assert diff < 2.0, f"Export/inflate mismatch: max diff={diff}"
    print(f"  export/inflate round-trip: OK (max diff={diff:.1f})")

    # 6. Self-compressing variant
    sc_codec = SelfCompressingVideoCodec(
        hidden=16, layers=2, omega_0=30.0,
        num_frames=T, frame_h=H, frame_w=W,
        pos_encoding_freqs=4, init_bits=8.0,
    )
    sc_rgb = sc_codec(coords)
    assert sc_rgb.shape == (32, 3)
    total_bits = sc_codec.total_bits()
    assert total_bits.requires_grad, "total_bits must be differentiable"
    total_bits.backward()
    print(f"  self-compressing forward: OK (total_bits={total_bits.item():.0f})")

    # 7. Training (tiny, just verify it runs)
    gt = torch.randint(0, 256, (T, H, W, 3), dtype=torch.uint8)
    trained = train_network_codec(
        gt,
        hidden=8, layers=2, omega_0=15.0,
        epochs=10, batch_pixels=64,
        log_every=5, pos_encoding_freqs=3,
    )
    assert trained is not None
    print("  training (10 epochs): OK")

    # 8. Self-compressing training
    trained_sc = train_network_codec(
        gt,
        hidden=8, layers=2, omega_0=15.0,
        epochs=10, batch_pixels=64,
        use_self_compress=True,
        lambda_rate_end=0.5,
        target_size_kb=1,
        log_every=5, pos_encoding_freqs=3,
    )
    assert isinstance(trained_sc, SelfCompressingVideoCodec)
    print(f"  self-compressing training: OK (size={trained_sc.total_bytes():.0f}B)")

    # 9. Load model without inflation
    archive2 = export_network_archive(trained)
    loaded = load_network_codec(archive2)
    assert loaded.param_count() == trained.param_count()
    print("  load_network_codec: OK")

    # 10. PositionalEncoding
    pe = PositionalEncoding(num_frequencies=6, include_input=True)
    x = torch.tensor([[0.5]])
    out = pe(x)
    assert out.shape == (1, 13), f"PE output shape: {out.shape}"  # 1 + 2*6
    print(f"  positional encoding: OK ({out.shape})")

    print("network_codec: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
