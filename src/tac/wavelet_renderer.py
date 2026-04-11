"""Wavelet-domain renderer: segment masks -> RGB frames via iDWT synthesis.

Architecture inspired by INRIA DP-SIMS-Lite: instead of predicting RGB pixels
directly from segmentation masks, predict wavelet coefficients at 2
decomposition levels. The inverse wavelet transform (iDWT) is parameter-free,
so the neural network is smaller (~100-200K params vs 312K pixel-domain).

Pipeline:
    5-class mask (B, H, W) at 384x512
    -> class embedding (B, embed_dim, H, W)
    -> downsample to each wavelet scale
    -> coarse CNN predicts LL2 coefficients at 96x128
    -> mid CNN predicts LH2/HL2/HH2 detail at 96x128
    -> fine CNN predicts LH1/HL1/HH1 detail at 192x256
    -> 2-level inverse Haar DWT -> RGB at 384x512

Key insight: class-conditional coefficient prediction. Road pixels need
different wavelet patterns than sky or vehicle pixels. The per-class
embedding lets the network specialize wavelet templates per semantic class.

Advantages over pixel-domain:
    - Network operates at half/quarter resolution (fewer computations)
    - iDWT provides free spatial upsampling with perfect reconstruction
    - Class-conditional coefficient prediction separates coarse structure
      from fine detail naturally
    - Significantly smaller parameter count (~100-200K vs 312K)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Haar wavelet transforms (parameter-free) ─────────────────────────


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D Haar discrete wavelet transform (analysis).

    Decomposes an image into four subbands at half resolution:
        LL: low-frequency approximation (coarse structure)
        LH: horizontal detail (horizontal edges)
        HL: vertical detail (vertical edges)
        HH: diagonal detail (diagonal edges)

    Args:
        x: (B, C, H, W) — input image or feature map. H, W must be even.

    Returns:
        Tuple of (LL, LH, HL, HH), each (B, C, H/2, W/2)
    """
    # Even/odd rows and columns
    x00 = x[:, :, 0::2, 0::2]  # even row, even col
    x01 = x[:, :, 0::2, 1::2]  # even row, odd col
    x10 = x[:, :, 1::2, 0::2]  # odd row, even col
    x11 = x[:, :, 1::2, 1::2]  # odd row, odd col

    # Separable 2D Haar: row transform then column transform
    # No normalization in forward — inverse applies 1/4
    ll = x00 + x01 + x10 + x11
    lh = x00 + x01 - x10 - x11
    hl = x00 - x01 + x10 - x11
    hh = x00 - x01 - x10 + x11  # note: +x11 (separable product of two sign flips)
    return ll, lh, hl, hh


def haar_idwt2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """2D inverse Haar DWT (synthesis) — PARAMETER FREE.

    Reconstructs an image from its four wavelet subbands. This is the
    exact inverse of haar_dwt2d: haar_idwt2d(*haar_dwt2d(x)) == x.

    Args:
        ll: (B, C, H, W) — low-frequency approximation
        lh: (B, C, H, W) — horizontal detail
        hl: (B, C, H, W) — vertical detail
        hh: (B, C, H, W) — diagonal detail

    Returns:
        (B, C, H*2, W*2) — reconstructed image at double resolution
    """
    B, C, H, W = ll.shape
    out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
    # Inverse: 1/4 normalization compensates forward's unnormalized sums
    out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.25
    out[:, :, 0::2, 1::2] = (ll + lh - hl - hh) * 0.25
    out[:, :, 1::2, 0::2] = (ll - lh + hl - hh) * 0.25
    out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.25
    return out


# ── Coefficient predictor blocks ─────────────────────────────────────


class _CoarsePredictor(nn.Module):
    """Predict LL2 wavelet coefficients (coarse RGB appearance).

    Operates at quarter resolution (96x128 for 384x512 input).
    This has the most capacity since LL encodes the coarse structure
    that dominates visual appearance and scorer perception.
    """

    def __init__(self, in_ch: int, hidden: int, out_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DetailPredictor(nn.Module):
    """Predict detail wavelet coefficients (LH, HL, HH for one level).

    Lighter than coarse predictor — detail bands are sparse (mostly zero)
    because natural images have smooth regions between edges.

    Outputs 3 detail bands x out_ch (e.g., 3 bands x 3 RGB = 9 channels).
    """

    def __init__(self, in_ch: int, hidden: int, out_ch: int = 3):
        super().__init__()
        self.out_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch * 3, 3, padding=1, bias=True),  # 3 detail bands
        )
        # Zero-init: detail bands start at zero (smooth image)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (LH, HL, HH) each with out_ch channels."""
        out = self.net(x)
        c = self.out_ch
        return out[:, 0:c], out[:, c : 2 * c], out[:, 2 * c : 3 * c]


# ── WaveletRenderer ──────────────────────────────────────────────────


class WaveletRenderer(nn.Module):
    """Render RGB frames from segmentation masks via wavelet-domain synthesis.

    Instead of predicting pixels directly, predicts wavelet coefficients
    at 2 decomposition levels. The inverse DWT (iDWT) is parameter-free.

    Advantages over pixel-domain:
    - Network operates at half/quarter resolution (fewer computations)
    - iDWT provides free spatial upsampling with perfect reconstruction
    - Class-conditional coefficient prediction naturally separates
      coarse structure from fine detail

    For 384x512 input:
        Level 2: LL2 (96x128), LH2/HL2/HH2 (96x128)
        Level 1: LH1/HL1/HH1 (192x256)
        Output: RGB (384x512) via 2-level iDWT

    Args:
        num_classes: segmentation classes (5 for comma SegNet)
        embed_dim: per-class embedding dimension
        hidden: base hidden channel width for predictors
        embedding: optional pre-created Embedding (for weight sharing)
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 8,
        hidden: int = 48,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Class embedding: each of 5 classes -> learned embed_dim-dimensional vector
        self.embedding = embedding if embedding is not None else nn.Embedding(num_classes, embed_dim)

        # Coarse predictor: quarter-res embedding -> LL2 (96x128, 3ch RGB)
        # This is the main capacity — predicts coarse RGB appearance
        self.coarse_net = _CoarsePredictor(embed_dim, hidden, out_ch=3)

        # Detail predictor (level 2): quarter-res embedding -> LH2/HL2/HH2
        # At quarter resolution, these capture coarse edges and texture
        self.detail_coarse_net = _DetailPredictor(embed_dim, hidden // 2, out_ch=3)

        # Detail predictor (level 1): half-res embedding -> LH1/HL1/HH1
        # At half resolution, these capture fine edges — lightest network
        self.detail_fine_net = _DetailPredictor(embed_dim, hidden // 4, out_ch=3)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Generate RGB from mask via wavelet synthesis.

        Args:
            masks: (B, H, W) long tensor, class labels 0..num_classes-1

        Returns:
            (B, 3, H, W) float RGB in [0, 255]
        """
        B, H, W = masks.shape

        # Embed mask at full resolution: (B, H, W) -> (B, embed_dim, H, W)
        emb = self.embedding(masks).permute(0, 3, 1, 2).contiguous()

        # Downsample embedding to each wavelet scale via nearest neighbor
        # (preserves class boundaries — no interpolation artifacts)
        H2, W2 = H // 2, W // 2
        H4, W4 = H // 4, W // 4
        emb_half = F.interpolate(emb, size=(H2, W2), mode="nearest")
        emb_quarter = F.interpolate(emb, size=(H4, W4), mode="nearest")

        # Predict wavelet coefficients at each level
        # Level 2 LL: coarse appearance (96x128 for 384x512 input)
        ll2 = self.coarse_net(emb_quarter)  # (B, 3, H/4, W/4)

        # Level 2 details: coarse edges (96x128)
        lh2, hl2, hh2 = self.detail_coarse_net(emb_quarter)  # each (B, 3, H/4, W/4)

        # Level 1 details: fine edges (192x256)
        lh1, hl1, hh1 = self.detail_fine_net(emb_half)  # each (B, 3, H/2, W/2)

        # Inverse DWT: reconstruct from coarse to fine
        # Level 2 -> Level 1 LL: iDWT(LL2, LH2, HL2, HH2) -> (B, 3, H/2, W/2)
        ll1 = haar_idwt2d(ll2, lh2, hl2, hh2)

        # Level 1 -> Full resolution: iDWT(LL1, LH1, HL1, HH1) -> (B, 3, H, W)
        rgb = haar_idwt2d(ll1, lh1, hl1, hh1)

        # Soft sigmoid output: always-flowing gradients, no dead zones
        # sigmoid(0/50) = 0.5 -> 127.5 at init (mid-gray)
        return 255.0 * torch.sigmoid(rgb / 50.0)

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── WaveletPairGenerator ─────────────────────────────────────────────


class WaveletPairGenerator(nn.Module):
    """Generate PoseNet-compatible frame pairs from segmentation masks.

    Combines WaveletRenderer and MotionPredictor (reused from renderer.py)
    to produce the (B, 2, H, W, 3) HWC pair format the scorer expects.

    Same interface as PairGenerator from renderer.py — drop-in replacement.

    Args:
        renderer: WaveletRenderer instance
        motion: MotionPredictor instance
    """

    def __init__(self, renderer: WaveletRenderer, motion):
        super().__init__()
        self.renderer = renderer
        self.motion = motion
        # Learned blend weight: sigmoid(raw) gives alpha in [0, 1]
        # Initialize to 0 -> sigmoid(0) = 0.5 -> equal blend
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a scored frame pair from two consecutive masks.

        Args:
            mask_t: (B, H, W) long — mask at time t
            mask_t1: (B, H, W) long — mask at time t+1

        Returns:
            (B, 2, H, W, 3) float tensor in [0, 255] — HWC pair format
        """
        # Render both frames directly via wavelet synthesis
        frame_t = self.renderer(mask_t)  # (B, 3, H, W)
        frame_t1 = self.renderer(mask_t1)  # (B, 3, H, W)

        # Predict flow and warp frame_t -> frame_t1_warped
        from tac.renderer import warp_with_flow

        flow = self.motion(mask_t, mask_t1)  # (B, 2, H, W)
        frame_t1_warped = warp_with_flow(frame_t, flow)

        # Blend warped and directly-rendered frame_t+1
        alpha = torch.sigmoid(self.blend_logit)
        frame_t1_blended = (alpha * frame_t1_warped + (1.0 - alpha) * frame_t1).clamp(0.0, 255.0)

        # Pack to HWC pair format: (B, 2, H, W, 3)
        f_t_hwc = frame_t.permute(0, 2, 3, 1)
        f_t1_hwc = frame_t1_blended.permute(0, 2, 3, 1)
        return torch.stack([f_t_hwc, f_t1_hwc], dim=1)

    def param_count(self) -> int:
        """Total trainable parameter count (renderer + motion + blend)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ──────────────────────────────────────────────────────────


def build_wavelet_renderer(
    num_classes: int = 5,
    embed_dim: int = 8,
    hidden: int = 48,
    motion_hidden: int = 32,
) -> WaveletPairGenerator:
    """Build the full wavelet mask-to-pair rendering pipeline.

    Default settings target ~100-200K params (significantly smaller than
    the 312K pixel-domain U-Net renderer).

    Features:
        - Shared embedding between renderer and motion predictor
        - 2-level Haar wavelet synthesis (parameter-free iDWT)
        - Class-conditional coefficient prediction
        - Soft sigmoid output for always-flowing gradients

    Args:
        num_classes: segmentation classes (5 for comma)
        embed_dim: per-class embedding dimension
        hidden: base hidden width for coefficient predictors
        motion_hidden: MotionPredictor hidden channels

    Returns:
        WaveletPairGenerator wrapping WaveletRenderer + MotionPredictor
    """
    from tac.renderer import MotionPredictor

    # Shared embedding: renderer and motion predictor learn a single
    # class representation, reducing parameters and improving coherence
    shared_embed = nn.Embedding(num_classes, embed_dim)

    renderer = WaveletRenderer(
        num_classes=num_classes,
        embed_dim=embed_dim,
        hidden=hidden,
        embedding=shared_embed,
    )
    motion = MotionPredictor(
        num_classes=num_classes,
        embed_dim=embed_dim,
        hidden=motion_hidden,
        embedding=shared_embed,
    )
    pair_gen = WaveletPairGenerator(renderer, motion)

    # Verify embedding is truly shared
    assert renderer.embedding is motion.embedding, "Embedding sharing failed"

    total = pair_gen.param_count()
    r_count = renderer.param_count()
    m_count = motion.param_count()
    print(
        f"[wavelet_renderer] Built WaveletPairGenerator: {total:,} params "
        f"(renderer={r_count:,}, motion={m_count:,}, blend=1, "
        f"shared_embed={shared_embed.weight.numel()}, wavelet=Haar 2-level)"
    )

    return pair_gen
