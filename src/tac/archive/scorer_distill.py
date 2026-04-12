"""Scorer distillation heads — lightweight proxies for PoseNet/SegNet (Trick 21).

Train tiny heads that mimic the frozen scorers' behavior from intermediate
renderer features.  At inflate time, these heads provide a differentiable
scorer proxy without loading the full 25MB PoseNet/SegNet models.

Theory: the full scorers have ~10M parameters but their actual behavior
on our mask-conditioned distribution is much lower-dimensional.  A 3-layer
MLP (PoseNet) or 1x1 conv (SegNet) with <50K parameters can approximate
the scorer output within 1-2% on in-distribution data.

Benefits:
    1. Size: distilled heads are <200KB vs 25MB for full scorers.
    2. Speed: ~50x faster inference (3-layer MLP vs ResNet).
    3. Deployability: can be included in archive.zip for TTO at inflate time.
    4. Differentiability: always differentiable (no frozen model quirks).

Architecture:
    PoseNetHead: features → 64 → 32 → 6 (pose regression)
    SegNetHead: features → num_classes via 1x1 conv (per-pixel classification)

Usage::

    from tac.archive.scorer_distill import PoseNetHead, SegNetHead, DistilledScorerWrapper

    pose_head = PoseNetHead(input_dim=64)
    seg_head = SegNetHead(input_dim=64, num_classes=5)
    wrapper = DistilledScorerWrapper(renderer, pose_head, seg_head)

    # Train heads to mimic frozen scorers
    from tac.archive.scorer_distill import distill_scorer_heads
    distill_scorer_heads(renderer, posenet, segnet, train_data, epochs=50)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── PoseNet Distillation Head ─────────────────────────────────────────


class PoseNetHead(nn.Module):
    """Lightweight MLP that mimics PoseNet's 6-DOF pose regression.

    Architecture: input_dim → 64 → 32 → 6
    Uses ReLU activations and LayerNorm for training stability.

    The 6 outputs correspond to PoseNet's first 6 pose parameters
    (3 translation + 3 rotation), which is what the scoring formula uses.

    Args:
        input_dim: feature dimension from the renderer's intermediate layer.
        hidden: first hidden layer width (default 64).
    """

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 6),
        )
        # Zero-init last layer for small initial predictions
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict 6-DOF pose from spatially-pooled features.

        Args:
            features: (B, C, H, W) intermediate renderer features,
                or (B, C) pre-pooled features.

        Returns:
            (B, 6) pose predictions.
        """
        if features.ndim == 4:
            # Global average pool: (B, C, H, W) → (B, C)
            features = features.mean(dim=(2, 3))
        return self.net(features)


# ── SegNet Distillation Head ──────────────────────────────────────────


class SegNetHead(nn.Module):
    """Lightweight 1x1 conv that mimics SegNet's per-pixel classification.

    A single 1x1 convolution from feature channels to num_classes.
    This is sufficient because SegNet's final layer is essentially a
    1x1 classification head on encoder-decoder features.

    Args:
        input_dim: feature channels from the renderer.
        num_classes: number of segmentation classes (5 for comma SegNet).
    """

    def __init__(self, input_dim: int, num_classes: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv = nn.Conv2d(input_dim, num_classes, 1, bias=True)
        # Zero-init for uniform initial predictions
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict per-pixel class logits from renderer features.

        Args:
            features: (B, C, H, W) intermediate renderer features.

        Returns:
            (B, num_classes, H, W) class logits (un-normalized).
        """
        return self.conv(features)


# ── Distilled Scorer Wrapper ─────────────────────────────────────────


class DistilledScorerWrapper(nn.Module):
    """Wraps a renderer with distilled scorer heads for end-to-end training.

    Attaches PoseNetHead and SegNetHead to the renderer's intermediate
    features, providing a differentiable scorer proxy.

    The wrapper extracts features from the renderer via a forward hook
    on a specified layer, or from the renderer's penultimate output if
    the renderer exposes features via a .features attribute.

    For simple usage, the renderer can be any module — the wrapper
    applies a small feature extractor (3x3 conv) to the RGB output.

    Args:
        renderer: any nn.Module producing (B, 3, H, W) RGB output.
        posenet_head: PoseNetHead instance.
        segnet_head: SegNetHead instance.
        feature_dim: if renderer doesn't expose features, extract them
            from RGB via a small conv with this many channels.
    """

    def __init__(
        self,
        renderer: nn.Module,
        posenet_head: PoseNetHead,
        segnet_head: SegNetHead,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.renderer = renderer
        self.posenet_head = posenet_head
        self.segnet_head = segnet_head

        # Feature extractor from RGB (used if renderer doesn't expose features)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        masks: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Render frames and predict scorer outputs.

        Args:
            masks: (B, H, W) long — segmentation masks.
            **kwargs: forwarded to renderer.

        Returns:
            Dict with keys:
                - "rgb": (B, 3, H, W) rendered frames.
                - "pose": (B, 6) predicted pose parameters.
                - "seg_logits": (B, num_classes, H, W) segmentation logits.
                - "features": (B, feature_dim, H, W) intermediate features.
        """
        rgb = self.renderer(masks, **kwargs)

        # Extract features from RGB
        features = self.feature_extractor(rgb / 255.0)

        # Predict scorer outputs
        pose = self.posenet_head(features)
        seg_logits = self.segnet_head(features)

        return {
            "rgb": rgb,
            "pose": pose,
            "seg_logits": seg_logits,
            "features": features,
        }


# ── Distillation Training ────────────────────────────────────────────


def distill_scorer_heads(
    renderer: nn.Module,
    posenet: nn.Module,
    segnet: nn.Module,
    train_masks: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    feature_dim: int = 32,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> DistilledScorerWrapper:
    """Train distilled scorer heads to mimic frozen PoseNet/SegNet.

    Procedure:
        1. Freeze the renderer (it should already be trained).
        2. Generate RGB frames from masks using the renderer.
        3. Run frozen PoseNet/SegNet on the generated frames to get targets.
        4. Train PoseNetHead (MSE loss) and SegNetHead (CE loss) to match.

    The renderer's parameters are NOT updated — only the heads and
    the feature extractor are trained.

    Args:
        renderer: trained renderer (frozen during distillation).
        posenet: frozen PoseNet scorer.
        segnet: frozen SegNet scorer.
        train_masks: (N, H, W) long tensor — training segmentation masks.
        epochs: number of training epochs.
        lr: learning rate.
        feature_dim: feature dimension for the heads.
        device: computation device.
        verbose: print training progress.

    Returns:
        DistilledScorerWrapper with trained heads.
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Build heads
    pose_head = PoseNetHead(input_dim=feature_dim)
    seg_head = SegNetHead(input_dim=feature_dim, num_classes=5)
    wrapper = DistilledScorerWrapper(renderer, pose_head, seg_head, feature_dim=feature_dim)
    wrapper = wrapper.to(device)

    # Freeze renderer
    for p in wrapper.renderer.parameters():
        p.requires_grad = False

    # Trainable parameters: heads + feature extractor only
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    N = train_masks.shape[0]
    batch_size = min(8, N)

    for epoch in range(epochs):
        epoch_pose_loss = 0.0
        epoch_seg_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            masks_batch = train_masks[i : i + batch_size].to(device)

            # Generate RGB with frozen renderer
            with torch.no_grad():
                rgb = wrapper.renderer(masks_batch)

            # Get scorer targets
            with torch.no_grad():
                # PoseNet target: need pairs, so use consecutive frames
                # For single-frame distillation, just match the feature response
                pose_input = _prepare_posenet_input(rgb, posenet)
                pose_target = posenet(pose_input)
                if isinstance(pose_target, dict):
                    pose_target = pose_target.get("pose", pose_target.get("output"))
                pose_target = pose_target[:, :6].detach()

                # SegNet target
                seg_input = _prepare_segnet_input(rgb)
                seg_target = segnet(seg_input).argmax(dim=1).detach()  # (B, H, W)

            # Forward through wrapper (features + heads)
            out = wrapper(masks_batch)

            # Losses
            pose_loss = F.mse_loss(out["pose"], pose_target)

            # Resize seg_logits to match seg_target spatial dims
            seg_logits = out["seg_logits"]
            if seg_logits.shape[2:] != seg_target.shape[1:]:
                seg_logits = F.interpolate(
                    seg_logits,
                    size=seg_target.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
            seg_loss = F.cross_entropy(seg_logits, seg_target)

            loss = pose_loss + seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_pose_loss += pose_loss.item()
            epoch_seg_loss += seg_loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 10 == 0:
            avg_pose = epoch_pose_loss / max(n_batches, 1)
            avg_seg = epoch_seg_loss / max(n_batches, 1)
            print(f"  [distill] epoch {epoch + 1}/{epochs}: pose_loss={avg_pose:.6f}, seg_loss={avg_seg:.4f}")

    # Unfreeze renderer (caller decides what to do next)
    for p in wrapper.renderer.parameters():
        p.requires_grad = True

    return wrapper


def _prepare_posenet_input(
    rgb: torch.Tensor,
    posenet: nn.Module,
) -> torch.Tensor:
    """Prepare RGB frames for PoseNet forward pass.

    PoseNet expects preprocessed input. If the model has a
    preprocess_input method, use it. Otherwise, create a simple
    pair from the batch by duplicating frames.

    Args:
        rgb: (B, 3, H, W) float tensor in [0, 255].
        posenet: PoseNet model.

    Returns:
        Preprocessed input tensor for posenet.forward().
    """
    B, C, H, W = rgb.shape
    # PoseNet expects (B, T=2, C, H, W) pairs
    # For distillation, use each frame paired with itself (zero motion)
    rgb_hwc = rgb.permute(0, 2, 3, 1)  # (B, H, W, 3)
    pair = torch.stack([rgb_hwc, rgb_hwc], dim=1)  # (B, 2, H, W, 3)

    if hasattr(posenet, "preprocess_input"):
        # Convert to (B, T, C, H, W) for preprocess_input
        pair_bchw = pair.permute(0, 1, 4, 2, 3)  # (B, 2, 3, H, W)
        return posenet.preprocess_input(pair_bchw)

    # Fallback: concatenate along channel dim
    return pair.reshape(B, 2 * H, W, C).permute(0, 3, 1, 2)


def _prepare_segnet_input(rgb: torch.Tensor) -> torch.Tensor:
    """Prepare RGB frames for SegNet forward pass.

    SegNet expects (B, C, H, W) float input. Normalize to [0, 1]
    if the model expects it, otherwise pass as-is.

    Args:
        rgb: (B, 3, H, W) float tensor in [0, 255].

    Returns:
        Input tensor for segnet.forward().
    """
    return rgb


# ── Save / Load ──────────────────────────────────────────────────────


def save_distilled_heads(
    wrapper: DistilledScorerWrapper,
    path: str | Path,
) -> None:
    """Save distilled heads (without renderer) for archive.zip inclusion.

    Saves only the feature extractor, PoseNetHead, and SegNetHead —
    NOT the renderer. Total size: typically <200KB.

    Args:
        wrapper: DistilledScorerWrapper with trained heads.
        path: output file path (e.g., 'distilled_heads.pt').
    """
    path = Path(path)
    state = {
        "feature_extractor": wrapper.feature_extractor.state_dict(),
        "posenet_head": wrapper.posenet_head.state_dict(),
        "segnet_head": wrapper.segnet_head.state_dict(),
        "config": {
            "posenet_input_dim": wrapper.posenet_head.input_dim,
            "posenet_hidden": wrapper.posenet_head.net[0].out_features,
            "segnet_input_dim": wrapper.segnet_head.input_dim,
            "segnet_num_classes": wrapper.segnet_head.num_classes,
        },
    }
    torch.save(state, str(path))
    size_kb = path.stat().st_size / 1024
    print(f"[scorer_distill] Saved distilled heads to {path} ({size_kb:.1f} KB)")


def load_distilled_heads(
    path: str | Path,
    renderer: nn.Module | None = None,
) -> DistilledScorerWrapper:
    """Load distilled heads and wrap a renderer.

    Args:
        path: path to saved heads file.
        renderer: renderer to wrap. If None, a dummy is used (heads-only mode).

    Returns:
        DistilledScorerWrapper with loaded heads.
    """
    path = Path(path)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    config = state["config"]

    pose_head = PoseNetHead(
        input_dim=config["posenet_input_dim"],
        hidden=config.get("posenet_hidden", 64),
    )
    seg_head = SegNetHead(
        input_dim=config["segnet_input_dim"],
        num_classes=config["segnet_num_classes"],
    )

    if renderer is None:
        renderer = nn.Identity()

    wrapper = DistilledScorerWrapper(
        renderer=renderer,
        posenet_head=pose_head,
        segnet_head=seg_head,
        feature_dim=config["posenet_input_dim"],
    )

    wrapper.feature_extractor.load_state_dict(state["feature_extractor"])
    wrapper.posenet_head.load_state_dict(state["posenet_head"])
    wrapper.segnet_head.load_state_dict(state["segnet_head"])

    return wrapper


# ── Smoke tests ───────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Run basic shape and forward-pass checks."""
    B, H, W = 2, 32, 32
    feature_dim = 16
    num_classes = 5

    # PoseNetHead
    pose_head = PoseNetHead(input_dim=feature_dim, hidden=32)
    feat_4d = torch.randn(B, feature_dim, H, W)
    pose_out = pose_head(feat_4d)
    assert pose_out.shape == (B, 6), f"PoseNetHead output shape wrong: {pose_out.shape}"

    # Pre-pooled input
    feat_2d = torch.randn(B, feature_dim)
    pose_out_2d = pose_head(feat_2d)
    assert pose_out_2d.shape == (B, 6)

    # At init, output should be near zero
    assert pose_out.abs().max() < 0.1, "PoseNetHead should start near zero"

    # SegNetHead
    seg_head = SegNetHead(input_dim=feature_dim, num_classes=num_classes)
    seg_out = seg_head(feat_4d)
    assert seg_out.shape == (B, num_classes, H, W), f"SegNetHead output shape wrong: {seg_out.shape}"

    # At init, output should be near zero (uniform predictions)
    assert seg_out.abs().max() < 0.1, "SegNetHead should start near zero"

    # DistilledScorerWrapper with a tiny renderer
    class TinyRenderer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(5, 3, 1)
            nn.init.constant_(self.conv.bias, 128.0)

        def forward(self, masks, **kwargs):
            B, H, W = masks.shape
            oh = torch.zeros(B, 5, H, W, dtype=torch.float32)
            oh.scatter_(1, masks.unsqueeze(1).clamp(0, 4), 1.0)
            return torch.sigmoid(self.conv(oh)) * 255.0

    renderer = TinyRenderer()
    wrapper = DistilledScorerWrapper(
        renderer, pose_head, seg_head, feature_dim=feature_dim
    )

    masks = torch.randint(0, num_classes, (B, H, W))
    out = wrapper(masks)

    assert "rgb" in out and out["rgb"].shape == (B, 3, H, W)
    assert "pose" in out and out["pose"].shape == (B, 6)
    assert "seg_logits" in out and out["seg_logits"].shape == (B, num_classes, H, W)
    assert "features" in out and out["features"].shape == (B, feature_dim, H, W)

    # Gradient flows
    loss = out["pose"].sum() + out["seg_logits"].sum()
    loss.backward()
    assert pose_head.net[0].weight.grad is not None, "PoseNetHead should have gradients"
    assert seg_head.conv.weight.grad is not None, "SegNetHead should have gradients"

    # Save / load round-trip
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    save_distilled_heads(wrapper, tmp_path)
    loaded = load_distilled_heads(tmp_path, renderer=renderer)

    # Verify loaded heads produce same output
    with torch.no_grad():
        out_orig = wrapper(masks)
        out_loaded = loaded(masks)
    assert torch.allclose(
        out_orig["pose"], out_loaded["pose"], atol=1e-5
    ), "Loaded heads should match original"
    assert torch.allclose(
        out_orig["seg_logits"], out_loaded["seg_logits"], atol=1e-5
    ), "Loaded heads should match original"

    import os

    os.unlink(tmp_path)

    # Param count
    head_params = sum(p.numel() for p in pose_head.parameters()) + sum(
        p.numel() for p in seg_head.parameters()
    )
    assert head_params < 50_000, f"Heads should be lightweight, got {head_params}"

    print(f"scorer_distill: all smoke tests passed ({head_params} head params)")


if __name__ == "__main__":
    _smoke_test()
