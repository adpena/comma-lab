# SPDX-License-Identifier: MIT
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


# ── Trick 32: Backward delta rendering ───────────────────────────────


class BackwardDeltaRenderer(nn.Module):
    """Render frames in REVERSE order: last frame first, then backward deltas.

    The last frame is rendered perfectly (maximum SegNet fidelity), then each
    earlier frame is generated as next_frame + small_delta.  Because PoseNet
    measures pose change between consecutive frames, tiny deltas produce
    near-zero pose distortion.

    This exploits a key asymmetry: SegNet evaluates frames independently
    (so only the last frame needs to be perfect), while PoseNet evaluates
    consecutive pairs (so small inter-frame deltas minimize distortion).

    Args:
        base_renderer: a callable that takes (mask, prev_mask, prev_frame)
            and produces a single frame.  Typically a TemporalDeltaRenderer
            or similar module.  If None, uses an internal DeltaConditioner +
            DeltaPredictor.
        num_classes: number of segmentation classes (5 for comma).
        cond_channels: intermediate conditioning width.
        hidden: delta predictor hidden width.
        delta_scale: max absolute delta per pixel.
    """

    def __init__(
        self,
        base_renderer: nn.Module | None = None,
        num_classes: int = 5,
        cond_channels: int = 16,
        hidden: int = 32,
        delta_scale: float = 30.0,
    ):
        super().__init__()
        if base_renderer is not None:
            self.delta_module = base_renderer
        else:
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
            self.delta_module = None

    def _generate_one(
        self,
        mask_curr: torch.Tensor,
        mask_next: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Generate frame[t] from frame[t+1] via backward delta.

        Args:
            mask_curr: (B, H, W) long — target frame's segmentation mask.
            mask_next: (B, H, W) long — already-rendered next frame's mask.
            next_frame: (B, 3, H, W) float in [0, 255] — the next frame.

        Returns:
            (B, 3, H, W) rendered current frame in [0, 255].
        """
        if self.delta_module is not None:
            # Use the base renderer: it takes (curr_mask, prev_mask, prev_frame)
            # In backward mode, "prev" is actually the next frame
            return self.delta_module(mask_curr, mask_next, next_frame)
        cond = self.conditioner(mask_curr, mask_next, next_frame)
        delta = self.predictor(cond)
        return (next_frame + delta).clamp(0.0, 255.0)

    def forward(
        self,
        mask_curr: torch.Tensor,
        mask_next: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a single frame via backward delta from the next frame.

        Args:
            mask_curr: (B, H, W) long — current frame's segmentation mask.
            mask_next: (B, H, W) long — next frame's segmentation mask.
            next_frame: (B, 3, H, W) float in [0, 255] — already-rendered next frame.

        Returns:
            (B, 3, H, W) rendered frame in [0, 255].
        """
        return self._generate_one(mask_curr, mask_next, next_frame)

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def render_backward_sequence(
    masks: torch.Tensor,
    initial_last_frame: torch.Tensor,
    backward_renderer: BackwardDeltaRenderer,
) -> torch.Tensor:
    """Render a frame sequence in reverse order using backward delta generation.

    Starts from the LAST frame (provided as initial_last_frame) and works
    backward to frame 0.  Each frame is generated as the next frame plus
    a small learned delta.

    This produces temporally smooth sequences with near-zero PoseNet
    distortion because consecutive frames differ by only a small delta.

    Args:
        masks: (B, T, H, W) long — segmentation masks for all frames.
        initial_last_frame: (B, 3, H, W) float — the last frame (given or
            rendered with maximum fidelity for SegNet).
        backward_renderer: a BackwardDeltaRenderer instance.

    Returns:
        (B, T, 3, H, W) rendered frames in temporal order (frame 0 first).
        The last frame at position T-1 is identical to initial_last_frame.
    """
    B, T, H, W = masks.shape
    # Build frames list in reverse: index T-1 first, then T-2, ..., 0
    frames_reversed: list[torch.Tensor] = [initial_last_frame]

    for t in range(T - 2, -1, -1):
        next_frame = frames_reversed[-1]
        frame_t = backward_renderer(masks[:, t], masks[:, t + 1], next_frame)
        frames_reversed.append(frame_t)

    # Reverse to get temporal order
    frames_reversed.reverse()
    return torch.stack(frames_reversed, dim=1)


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

    print(f"  temporal_delta: forward delta smoke passed ({n_params} params)")

    # ── Trick 32: backward delta rendering ──
    backward = BackwardDeltaRenderer(num_classes=num_classes, hidden=16)

    # Single backward frame
    mask_curr = torch.randint(0, num_classes, (B, H, W))
    mask_next = torch.randint(0, num_classes, (B, H, W))
    next_frame = torch.rand(B, 3, H, W) * 255.0
    out_bwd = backward(mask_curr, mask_next, next_frame)
    assert out_bwd.shape == (B, 3, H, W), f"Backward single frame shape: {out_bwd.shape}"
    assert out_bwd.min() >= 0.0 and out_bwd.max() <= 255.0

    # At init, backward delta should be ~zero
    diff_bwd = (out_bwd - next_frame).abs().max()
    assert diff_bwd < 1.0, f"Backward init delta should be ~0, got {diff_bwd:.4f}"

    # Backward sequence rendering
    last_frame = torch.rand(B, 3, H, W) * 255.0
    seq_bwd = render_backward_sequence(masks, last_frame, backward)
    assert seq_bwd.shape == (B, T, 3, H, W), f"Backward sequence shape: {seq_bwd.shape}"

    # Last frame should be identical to initial_last_frame
    assert torch.allclose(seq_bwd[:, -1], last_frame), "Last frame should equal initial_last_frame"

    # With base_renderer wrapping
    base = TemporalDeltaRenderer(num_classes=num_classes, hidden=16)
    backward_wrap = BackwardDeltaRenderer(base_renderer=base)
    out_wrap = backward_wrap(mask_curr, mask_next, next_frame)
    assert out_wrap.shape == (B, 3, H, W), "Wrapped backward shape wrong"

    bwd_params = backward.param_count()
    assert bwd_params > 0
    print(f"  temporal_delta: backward delta smoke passed ({bwd_params} params)")

    print("temporal_delta: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
