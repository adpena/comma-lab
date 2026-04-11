"""Temporal delta renderer — frame[t] as frame[t-1] + delta[t] (Trick 15).

Instead of generating each frame independently from its mask, predict
only the *delta* (residual) relative to the previous frame.  The base
renderer generates the delta conditioned on:
    - mask[t] (current segmentation)
    - mask[t-1] (previous segmentation)
    - prev_frame (the already-generated previous frame)

Advantages:
    - Temporal consistency: small mask changes → small deltas → smooth video.
    - Computation savings: the base renderer only needs to predict a
      small residual, not the full RGB signal.
    - Better for PoseNet: optical flow is implicitly encoded in the delta,
      so consecutive pairs have more consistent motion signals.

Theory: most codec distortion is temporal flicker — independent per-frame
generation introduces high-frequency temporal noise.  By construction,
delta rendering suppresses flicker because frame[t] = frame[t-1] + delta[t],
and the delta is regularized to be small.

Usage::

    base_renderer = MaskRenderer(...)
    delta_renderer = TemporalDeltaRenderer(base_renderer)
    frames = delta_renderer.render_sequence(masks, initial_frame)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaConditioner(nn.Module):
    """Fuse current mask, previous mask, and previous frame into a conditioning tensor.

    Concatenates one-hot masks and previous frame, then projects through a
    small conv to produce a conditioning input for the delta predictor.

    Args:
        num_classes: number of segmentation classes (5 for comma).
        frame_channels: RGB channels (3).
        out_channels: output conditioning channels.
    """

    def __init__(
        self,
        num_classes: int = 5,
        frame_channels: int = 3,
        out_channels: int = 16,
    ):
        super().__init__()
        in_ch = num_classes * 2 + frame_channels  # curr_mask_oh + prev_mask_oh + prev_frame
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )
        self.num_classes = num_classes

    def forward(
        self,
        mask_curr: torch.Tensor,
        mask_prev: torch.Tensor,
        prev_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Produce conditioning tensor.

        Args:
            mask_curr: (B, H, W) long — current frame's segmentation mask.
            mask_prev: (B, H, W) long — previous frame's segmentation mask.
            prev_frame: (B, 3, H, W) float — previous generated frame in [0, 255].

        Returns:
            (B, out_channels, H, W) conditioning features.
        """
        B, H, W = mask_curr.shape
        device = mask_curr.device

        # One-hot encode both masks
        oh_curr = _one_hot(mask_curr, self.num_classes, device)  # (B, K, H, W)
        oh_prev = _one_hot(mask_prev, self.num_classes, device)
        # Normalize frame to [0, 1]
        frame_norm = prev_frame / 255.0

        combined = torch.cat([oh_curr, oh_prev, frame_norm], dim=1)
        return self.conv(combined)


def _one_hot(mask: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """One-hot encode a (B, H, W) long mask to (B, K, H, W) float."""
    B, H, W = mask.shape
    oh = torch.zeros(B, num_classes, H, W, device=device, dtype=torch.float32)
    oh.scatter_(1, mask.unsqueeze(1).clamp(0, num_classes - 1), 1.0)
    return oh


class DeltaPredictor(nn.Module):
    """Lightweight residual predictor — 3 conv layers producing an RGB delta.

    The delta is tanh-activated and scaled, so it stays bounded.
    At initialization, the output is zero (identity: frame[t] = frame[t-1]).

    Args:
        cond_channels: input channels from DeltaConditioner.
        hidden: hidden layer width.
        delta_scale: maximum absolute delta per pixel (default 30.0).
    """

    def __init__(
        self,
        cond_channels: int = 16,
        hidden: int = 32,
        delta_scale: float = 30.0,
    ):
        super().__init__()
        self.delta_scale = delta_scale
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )
        # Zero-init last conv for identity at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Predict RGB delta from conditioning features.

        Args:
            cond: (B, cond_channels, H, W) conditioning tensor.

        Returns:
            (B, 3, H, W) delta in [-delta_scale, +delta_scale].
        """
        raw = self.net(cond)
        return torch.tanh(raw) * self.delta_scale


class TemporalDeltaRenderer(nn.Module):
    """Wraps a delta predictor to render frames as prev_frame + delta.

    Generates frame[t] = clamp(frame[t-1] + delta[t], 0, 255) where
    delta[t] is predicted from (mask[t], mask[t-1], frame[t-1]).

    Can also wrap an existing base renderer: the base renderer generates
    an initial RGB proposal, and the delta is added as a temporal correction.

    Args:
        num_classes: number of segmentation classes.
        cond_channels: intermediate conditioning width.
        hidden: delta predictor hidden width.
        delta_scale: max absolute delta per pixel.
    """

    def __init__(
        self,
        num_classes: int = 5,
        cond_channels: int = 16,
        hidden: int = 32,
        delta_scale: float = 30.0,
    ):
        super().__init__()
        self.conditioner = DeltaConditioner(
            num_classes=num_classes,
            frame_channels=3,
            out_channels=cond_channels,
        )
        self.predictor = DeltaPredictor(
            cond_channels=cond_channels,
            hidden=hidden,
            delta_scale=delta_scale,
        )

    def forward(
        self,
        mask_curr: torch.Tensor,
        mask_prev: torch.Tensor,
        prev_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Generate current frame as prev_frame + predicted delta.

        Args:
            mask_curr: (B, H, W) long — current segmentation.
            mask_prev: (B, H, W) long — previous segmentation.
            prev_frame: (B, 3, H, W) float in [0, 255].

        Returns:
            (B, 3, H, W) rendered frame in [0, 255].
        """
        cond = self.conditioner(mask_curr, mask_prev, prev_frame)
        delta = self.predictor(cond)
        return (prev_frame + delta).clamp(0.0, 255.0)

    def render_sequence(
        self,
        masks: torch.Tensor,
        initial_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Auto-regressively render a sequence of frames.

        Args:
            masks: (B, T, H, W) long — segmentation masks for all frames.
            initial_frame: (B, 3, H, W) float — the first frame (given).

        Returns:
            (B, T, 3, H, W) rendered frames including the initial frame
            at position 0.
        """
        B, T, H, W = masks.shape
        frames = [initial_frame]

        for t in range(1, T):
            prev = frames[-1]
            frame_t = self.forward(masks[:, t], masks[:, t - 1], prev)
            frames.append(frame_t)

        return torch.stack(frames, dim=1)

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Smoke tests ───────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Run basic shape and forward-pass checks."""
    B, T, H, W = 2, 5, 64, 64
    num_classes = 5

    renderer = TemporalDeltaRenderer(num_classes=num_classes, hidden=16)

    # Single frame forward
    mask_curr = torch.randint(0, num_classes, (B, H, W))
    mask_prev = torch.randint(0, num_classes, (B, H, W))
    prev_frame = torch.rand(B, 3, H, W) * 255.0

    out = renderer(mask_curr, mask_prev, prev_frame)
    assert out.shape == (B, 3, H, W), f"Expected (B, 3, H, W), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 255.0, "Output out of [0, 255] range"

    # At initialization, delta should be ~zero (identity)
    diff = (out - prev_frame).abs().max()
    assert diff < 1.0, f"At init, delta should be ~0, got max diff {diff:.4f}"

    # Sequence rendering
    masks = torch.randint(0, num_classes, (B, T, H, W))
    initial = torch.rand(B, 3, H, W) * 255.0
    seq = renderer.render_sequence(masks, initial)
    assert seq.shape == (B, T, 3, H, W), f"Sequence shape wrong: {seq.shape}"

    # First frame should be identical to initial
    assert torch.allclose(seq[:, 0], initial), "First frame should equal initial_frame"

    # Param count
    n_params = renderer.param_count()
    assert n_params > 0, "Should have trainable parameters"
    assert n_params < 100_000, f"Should be lightweight, got {n_params}"

    print(f"temporal_delta: all smoke tests passed ({n_params} params)")


if __name__ == "__main__":
    _smoke_test()
