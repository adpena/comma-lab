"""Cross-frame attention for temporal coherence in neural renderers (Trick 20).

Lightweight cross-attention module that attends from current-frame features
to previous-frame features.  This gives the renderer an explicit mechanism
to copy or adapt previous-frame information, improving:

    1. Temporal consistency (reduces flicker between consecutive frames).
    2. PoseNet score (PoseNet compares *pairs* — consistent features help).
    3. Efficiency (the renderer can reuse previous computation instead of
       re-deriving everything from the mask alone).

The attention is multi-head with a gated residual connection (initialized
to zero so the renderer starts as if attention were absent).

Architecture::

    Q = W_q(current_features)    shape: (B, C, H, W)
    K = W_k(previous_features)   shape: (B, C, H, W)
    V = W_v(previous_features)   shape: (B, C, H, W)

    # Spatial attention: per-pixel, channel-reduced dot product
    attn = softmax(Q @ K^T / sqrt(d_head))  over channel dim per pixel
    out = attn * V
    output = current_features + gate * W_o(out)

The gate starts at 0, so training is stable even when prepended to an
already-trained renderer.

Usage::

    attn = CrossFrameAttention(dim=64, num_heads=4)
    out = attn(current_feats, previous_feats)

    # Or wrap an entire renderer:
    wrapped = CrossFrameRenderer(base_renderer, dim=64)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFrameAttention(nn.Module):
    """Lightweight cross-attention: current features attend to previous features.

    Multi-head attention with per-pixel spatial operation (no quadratic
    attention over spatial locations — efficient for high-resolution features).

    Each head computes:
        score = sum(q * k, dim=channel) / sqrt(d_head)
        attn_weight = sigmoid(score)      # per-pixel, per-head
        attended = attn_weight * v

    This is O(B * num_heads * H * W * d_head), much cheaper than
    full spatial self-attention which is O(B * C * (H*W)^2).

    Args:
        dim: feature channel dimension (must be divisible by num_heads).
        num_heads: number of attention heads (default 4).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads

        # Q from current, K/V from previous
        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)

        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=True)

        # Gated residual: starts at zero (no effect at init)
        self.gate = nn.Parameter(torch.zeros(1))

        # Zero-init output projection for clean residual start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        current_features: torch.Tensor,
        previous_features: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend from current to previous features.

        Args:
            current_features: (B, C, H, W) current frame's features.
            previous_features: (B, C, H, W) previous frame's features.
                Must have same spatial dimensions.

        Returns:
            (B, C, H, W) attended features (current + gated attention output).
        """
        B, C, H, W = current_features.shape

        # Project to Q, K, V
        q = self.to_q(current_features)   # (B, C, H, W)
        k = self.to_k(previous_features)  # (B, C, H, W)
        v = self.to_v(previous_features)  # (B, C, H, W)

        # Reshape into multi-head: (B, num_heads, d_head, H, W)
        q = q.reshape(B, self.num_heads, self.d_head, H, W)
        k = k.reshape(B, self.num_heads, self.d_head, H, W)
        v = v.reshape(B, self.num_heads, self.d_head, H, W)

        # Per-pixel, per-head attention score
        scale = math.sqrt(self.d_head)
        # (B, num_heads, d_head, H, W) * (B, num_heads, d_head, H, W) -> sum over d_head
        score = (q * k).sum(dim=2, keepdim=True) / scale  # (B, num_heads, 1, H, W)
        attn = torch.sigmoid(score)  # (B, num_heads, 1, H, W)

        # Attend: element-wise multiply attention weights with values
        attended = attn * v  # (B, num_heads, d_head, H, W)

        # Reshape back: (B, C, H, W)
        attended = attended.reshape(B, C, H, W)

        # Gated output projection
        out = self.out_proj(attended)
        return current_features + self.gate * out


class CrossFrameRenderer(nn.Module):
    """Wrapper that adds cross-frame attention to any base renderer.

    Intercepts the base renderer's output features (before the RGB head)
    and applies CrossFrameAttention between consecutive frames.

    For renderers that output RGB directly, this wrapper:
        1. Runs the base renderer to get current RGB.
        2. Projects RGB to a feature space via a small encoder.
        3. Cross-attends to the previous frame's features.
        4. Decodes back to RGB via a small decoder.

    The encoder/decoder are initialized as near-identity so the wrapper
    starts as a pass-through.

    Args:
        base_renderer: any nn.Module that takes masks and returns (B, 3, H, W) RGB.
        dim: intermediate feature dimension for cross-attention.
        num_heads: attention heads.
    """

    def __init__(
        self,
        base_renderer: nn.Module,
        dim: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.base_renderer = base_renderer
        self.dim = dim

        # Lightweight encoder: RGB -> features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, dim),
            nn.ReLU(inplace=True),
        )

        # Cross-frame attention
        self.attention = CrossFrameAttention(dim=dim, num_heads=num_heads)

        # Lightweight decoder: features -> RGB residual
        self.decoder = nn.Conv2d(dim, 3, 3, padding=1, bias=True)

        # Zero-init decoder for pass-through at start
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        # State for autoregressive rendering
        self._prev_features: torch.Tensor | None = None

    def forward(
        self,
        masks: torch.Tensor,
        prev_features: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render current frame with cross-attention to previous features.

        Args:
            masks: (B, H, W) long — current frame's segmentation mask.
            prev_features: (B, dim, H, W) or None — previous frame's features.
                If None, no attention is applied (first frame).
            **kwargs: forwarded to base_renderer.

        Returns:
            (rgb, features) tuple:
                - rgb: (B, 3, H, W) rendered frame in [0, 255].
                - features: (B, dim, H, W) current frame's features (pass to next call).
        """
        # Base renderer generates initial RGB
        base_rgb = self.base_renderer(masks, **kwargs)

        # Encode to feature space
        features = self.encoder(base_rgb / 255.0)  # normalize to [0, 1]

        # Cross-attend to previous features if available
        if prev_features is not None:
            features = self.attention(features, prev_features)

        # Decode residual and add to base RGB
        residual = self.decoder(features)
        rgb = (base_rgb + residual).clamp(0.0, 255.0)

        return rgb, features

    def render_sequence(
        self,
        masks_seq: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Auto-regressively render a sequence with cross-frame attention.

        Args:
            masks_seq: (B, T, H, W) long — masks for all frames.
            **kwargs: forwarded to base_renderer.

        Returns:
            (B, T, 3, H, W) rendered RGB frames.
        """
        B, T, H, W = masks_seq.shape
        frames = []
        prev_feat = None

        for t in range(T):
            rgb, feat = self.forward(masks_seq[:, t], prev_features=prev_feat, **kwargs)
            frames.append(rgb)
            prev_feat = feat.detach()  # detach to prevent backprop through entire sequence

        return torch.stack(frames, dim=1)

    def param_count(self) -> int:
        """Total trainable parameter count (including base renderer)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Smoke tests ───────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Run basic shape and forward-pass checks."""
    B, H, W = 2, 32, 32
    dim = 16
    num_heads = 4

    # Test CrossFrameAttention standalone
    attn = CrossFrameAttention(dim=dim, num_heads=num_heads)
    curr = torch.randn(B, dim, H, W)
    prev = torch.randn(B, dim, H, W)

    out = attn(curr, prev)
    assert out.shape == (B, dim, H, W), f"Expected (B, dim, H, W), got {out.shape}"

    # At init, gate=0 so output should equal input
    diff = (out - curr).abs().max()
    assert diff < 1e-6, f"At init, output should equal input, got max diff {diff:.6f}"

    # Gradient flows through
    loss = out.sum()
    loss.backward()
    assert attn.gate.grad is not None, "Gate should have gradient"
    assert attn.to_q.weight.grad is not None, "Q projection should have gradient"

    # Test CrossFrameRenderer with a tiny base renderer
    class TinyRenderer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(5, 3, 1)
            nn.init.constant_(self.conv.weight, 0.0)
            nn.init.constant_(self.conv.bias, 128.0)

        def forward(self, masks, **kwargs):
            B, H, W = masks.shape
            oh = torch.zeros(B, 5, H, W, device=masks.device, dtype=torch.float32)
            oh.scatter_(1, masks.unsqueeze(1).clamp(0, 4), 1.0)
            return torch.sigmoid(self.conv(oh)) * 255.0

    base = TinyRenderer()
    wrapped = CrossFrameRenderer(base, dim=dim, num_heads=num_heads)

    masks = torch.randint(0, 5, (B, H, W))

    # First frame (no previous features)
    rgb1, feat1 = wrapped(masks)
    assert rgb1.shape == (B, 3, H, W), f"RGB shape wrong: {rgb1.shape}"
    assert feat1.shape == (B, dim, H, W), f"Feature shape wrong: {feat1.shape}"
    assert rgb1.min() >= 0.0 and rgb1.max() <= 255.0

    # Second frame (with previous features)
    rgb2, feat2 = wrapped(masks, prev_features=feat1)
    assert rgb2.shape == (B, 3, H, W)

    # Sequence rendering
    masks_seq = torch.randint(0, 5, (B, 4, H, W))
    seq = wrapped.render_sequence(masks_seq)
    assert seq.shape == (B, 4, 3, H, W), f"Sequence shape wrong: {seq.shape}"

    # Param count
    n_params = wrapped.param_count()
    assert n_params > 0

    print(f"cross_frame_attention: all smoke tests passed ({n_params} total params)")


if __name__ == "__main__":
    _smoke_test()
