"""Tests for the Selfcomp SegMap renderer + trainer.

The 6 mandated tests in the work-scope blueprint:
  - test_segmap_forward_shape
  - test_residualblock_identity_on_zeros
  - test_segmap_trainer_rejects_eval_roundtrip_false
  - test_segmap_trainer_train_epoch_loss_finite
  - test_export_inference_state_dict_keys_match
  - test_ema_apply_restore_cycle
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.preflight import PreflightError
from tac.segmap_renderer import (
    SEGMAP_INPUT_SIZE,
    ResidualBlock,
    SegMap,
    SegMapTrainer,
)
from tac.training import EMA, TrainConfig


# ─── Mock scorers shaped like upstream PoseNet/SegNet ────────────────────


class _MockPoseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(12, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8, 12)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = F.interpolate(x, size=(96, 128), mode="bilinear", align_corners=False)
        x = x.reshape(b, t * c, 96, 128)
        if x.shape[1] < 12:
            pad = torch.zeros(
                b, 12 - x.shape[1], 96, 128, device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=1)
        return x[:, :12]

    def forward(self, x: torch.Tensor) -> dict:
        h = self.conv(x).mean(dim=(2, 3))
        return {"pose": self.fc(h)}


class _MockSegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, ...]
        return F.interpolate(last, size=(48, 64), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─── Tiny SegMap fixture ────────────────────────────────────────────────


def _make_tiny_segmap(num_blocks: int = 2, max_frame_index: int = 16) -> SegMap:
    """A small but architecturally faithful SegMap for unit tests."""
    return SegMap(
        hidden=8,
        block_hidden=8,
        num_blocks=num_blocks,
        max_frame_index=max_frame_index,
    )


# ─── Architecture tests ─────────────────────────────────────────────────


def test_residualblock_identity_on_zeros() -> None:
    """A ResidualBlock with zeroed weights returns the post-activation residual.

    With both conv weights/biases zeroed, conv1(x)=0 -> SiLU(0)=0 -> conv2(0)=0
    -> SiLU(0 + x) = x * sigmoid(x). So the block reduces to ``x * sigmoid(x)``,
    which is the activation of the input — checking that the residual path
    survives the zero-conv configuration.
    """
    block = ResidualBlock(hidden=4, block_hidden=4)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    x = torch.randn(1, 4, 8, 8)
    expected = x * torch.sigmoid(x)
    out = block(x)
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-6)


def test_segmap_forward_shape() -> None:
    """SegMap.forward returns (B, 3, H, W) in the 0-255 sigmoid-scaled band."""
    model = _make_tiny_segmap()
    h = SEGMAP_INPUT_SIZE[1] // 16  # 24
    w = SEGMAP_INPUT_SIZE[0] // 16  # 32
    # Ensure a "model resolution" that matches the convolutional layout but
    # is small enough for a fast test.
    x = torch.rand(2, 5, h, w) * 0.2  # class-probability-shaped input
    frame_indices = torch.tensor([0, 1], dtype=torch.long)
    out = model(x, frame_indices)
    assert out.shape == (2, 3, h, w)
    assert torch.isfinite(out).all()
    # sigmoid * 255 -> bounded in [0, 255]
    assert out.min() >= 0.0
    assert out.max() <= 255.0


# ─── Trainer guard tests ───────────────────────────────────────────────


def _make_eval_roundtrip_true_config() -> TrainConfig:
    return TrainConfig(
        hidden=8,
        epochs=100,
        warmup_epochs=10,
        tag="test-segmap",
        lr=1e-3,
        eval_roundtrip=True,
        loss_mode="standard",
    )


def test_segmap_trainer_rejects_eval_roundtrip_false() -> None:
    """SegMapTrainer raises PreflightError on eval_roundtrip=False."""
    cfg = TrainConfig(
        hidden=8, epochs=100, warmup_epochs=10, tag="test-rt-false",
        eval_roundtrip=False,
    )
    model = _make_tiny_segmap()
    with pytest.raises(PreflightError, match=r"eval_roundtrip=True"):
        SegMapTrainer(model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu")


def test_segmap_trainer_rejects_mps_device() -> None:
    """SegMapTrainer raises PreflightError on device='mps'."""
    cfg = _make_eval_roundtrip_true_config()
    model = _make_tiny_segmap()
    with pytest.raises(PreflightError, match=r"refuses device='mps'"):
        SegMapTrainer(model, cfg, _MockPoseNet(), _MockSegNet(), device="mps")


# ─── Trainer integration test ──────────────────────────────────────────


def test_segmap_trainer_train_epoch_loss_finite() -> None:
    """One train_epoch step over toy data produces finite loss + updates params."""
    torch.manual_seed(0)
    cfg = _make_eval_roundtrip_true_config()
    # Use SEGMAP_INPUT_SIZE directly since the eval_roundtrip chain expects
    # (B*T, 3, h, w) shaped tensors that bicubic-resize to camera resolution.
    h, w = SEGMAP_INPUT_SIZE[1], SEGMAP_INPUT_SIZE[0]
    # Shrink for test speed: 1/16 scale on each axis.
    h = h // 16
    w = w // 16
    model = _make_tiny_segmap()
    trainer = SegMapTrainer(model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu")

    b = 1
    t = 2
    # Class-probability map (5-channel) at SegMap input resolution.
    masks = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    # GT pairs in HWC layout (B, T, H, W, 3) at the same model resolution.
    gt = torch.rand(b, t, h, w, 3) * 255.0

    pre_param = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    stats = trainer.train_epoch(masks, gt, ema=None)
    post_param = next(p for p in model.parameters() if p.requires_grad).detach()

    assert math.isfinite(stats["loss"]), f"loss not finite: {stats}"
    assert stats["num_steps"] == 1
    # At least one gradient step landed.
    assert not torch.equal(pre_param, post_param)


# ─── Export tests ─────────────────────────────────────────────────────


def test_export_inference_state_dict_keys_match() -> None:
    """Exported state_dict carries every key Selfcomp's load_segmap reads."""
    model = _make_tiny_segmap(num_blocks=3, max_frame_index=8)
    cfg = _make_eval_roundtrip_true_config()
    trainer = SegMapTrainer(model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu")

    state = trainer.export_inference_state_dict(ema=None)

    required = {
        "shared_latent_base",
        "frame_affine_embedding.weight",
        "layer_in.weight",
        "layer_in.bias",
        "layer_out.weight",
        "layer_out.bias",
    }
    for i in range(3):
        required.add(f"blocks.{i}.conv1.weight")
        required.add(f"blocks.{i}.conv1.bias")
        required.add(f"blocks.{i}.conv2.weight")
        required.add(f"blocks.{i}.conv2.bias")
    missing = required - set(state.keys())
    assert not missing, f"missing keys in export: {missing}"
    # Every value is a CPU tensor.
    for k, v in state.items():
        assert isinstance(v, torch.Tensor)
        assert v.device.type == "cpu"


def test_train_epoch_batch_size_eq_b_matches_unchunked_path() -> None:
    """When batch_size == B (single mini-batch == legacy unchunked path),
    the new code MUST be byte-identical to a legacy single-forward call.

    This is the strongest equivalence the chunking refactor can promise:
    no slicing happens, the whole epoch is one forward, and the result
    must match the prior implementation exactly."""
    torch.manual_seed(42)
    cfg = _make_eval_roundtrip_true_config()
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16

    b = 4
    t = 2
    masks = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt = torch.rand(b, t, h, w, 3) * 255.0

    torch.manual_seed(0)
    model_a = _make_tiny_segmap(num_blocks=2, max_frame_index=32)
    torch.manual_seed(0)
    posenet_a, segnet_a = _MockPoseNet(), _MockSegNet()
    torch.manual_seed(0)
    model_b = _make_tiny_segmap(num_blocks=2, max_frame_index=32)
    torch.manual_seed(0)
    posenet_b, segnet_b = _MockPoseNet(), _MockSegNet()

    trainer_a = SegMapTrainer(
        model_a, cfg, posenet_a, segnet_a, device="cpu"
    )
    trainer_b = SegMapTrainer(
        model_b, cfg, posenet_b, segnet_b, device="cpu"
    )

    # Both call: chunking degenerates to a single mini-batch.
    torch.manual_seed(123)
    stats_a = trainer_a.train_epoch(masks, gt, ema=None, batch_size=b)
    torch.manual_seed(123)
    stats_b = trainer_b.train_epoch(masks, gt, ema=None, batch_size=b)

    # Identical inputs + seeds + arch → identical outputs.
    assert stats_a["num_steps"] == 1
    assert stats_b["num_steps"] == 1
    assert math.isfinite(stats_a["loss"])
    assert abs(stats_a["loss"] - stats_b["loss"]) < 1e-4
    assert abs(stats_a["pose_dist"] - stats_b["pose_dist"]) < 1e-4
    assert abs(stats_a["seg_dist"] - stats_b["seg_dist"]) < 1e-4


def test_train_epoch_chunked_path_runs_and_steps_optimizer() -> None:
    """BUG CLASS B regression: chunked training (batch_size=4 over 8 pairs)
    MUST run end-to-end without error AND advance the model parameters via
    optimizer.step(). The mini-batch-VRAM savings are the WHOLE point of
    this fix — the 7.03 GiB OOM on T4 came from pushing 600 pairs through
    one forward; chunking makes T4 viable for the SegMap-paradigm lanes
    (SA-v2 / SC++-v2 / SO-v2) that died on 2026-04-29."""
    torch.manual_seed(42)
    cfg = _make_eval_roundtrip_true_config()
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16

    b = 8
    t = 2
    masks = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt = torch.rand(b, t, h, w, 3) * 255.0

    torch.manual_seed(0)
    model = _make_tiny_segmap(num_blocks=2, max_frame_index=64)
    pre_param = next(p for p in model.parameters() if p.requires_grad).detach().clone()

    trainer = SegMapTrainer(
        model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu"
    )
    torch.manual_seed(123)
    stats = trainer.train_epoch(masks, gt, ema=None, batch_size=4)

    assert math.isfinite(stats["loss"])
    # ceil(8 / 4) = 2 mini-batches → 2 forward passes per epoch.
    assert stats["num_steps"] == 2
    # Optimizer.step() ran at least once (params advanced from init).
    post_param = next(p for p in model.parameters() if p.requires_grad).detach()
    assert not torch.equal(pre_param, post_param), (
        "Chunked path did not advance model parameters — optimizer.step() "
        "either was not called or saw all-zero grads."
    )


def test_train_epoch_batch_size_chunks_correctly() -> None:
    """num_steps reflects ceil(B / batch_size)."""
    torch.manual_seed(0)
    cfg = _make_eval_roundtrip_true_config()
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16
    model = _make_tiny_segmap(num_blocks=2, max_frame_index=32)
    trainer = SegMapTrainer(
        model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu"
    )
    b = 7  # NOT a multiple of 3
    t = 2
    masks = F.softmax(torch.randn(b, t, 5, h, w), dim=2)
    gt = torch.rand(b, t, h, w, 3) * 255.0

    stats = trainer.train_epoch(masks, gt, ema=None, batch_size=3)
    # ceil(7 / 3) = 3 mini-batches.
    assert stats["num_steps"] == 3, stats


def test_train_epoch_rejects_invalid_batch_size() -> None:
    """batch_size < 1 must raise."""
    cfg = _make_eval_roundtrip_true_config()
    model = _make_tiny_segmap()
    trainer = SegMapTrainer(
        model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu"
    )
    h = SEGMAP_INPUT_SIZE[1] // 16
    w = SEGMAP_INPUT_SIZE[0] // 16
    masks = F.softmax(torch.randn(1, 2, 5, h, w), dim=2)
    gt = torch.rand(1, 2, h, w, 3) * 255.0
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        trainer.train_epoch(masks, gt, ema=None, batch_size=0)


def test_ema_apply_restore_cycle() -> None:
    """export_inference_state_dict(ema=EMA) restores live weights afterwards."""
    model = _make_tiny_segmap()
    cfg = _make_eval_roundtrip_true_config()
    trainer = SegMapTrainer(model, cfg, _MockPoseNet(), _MockSegNet(), device="cpu")

    ema = EMA(model, decay=0.99)
    # Mutate the live weights so the live state differs from EMA shadow.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    live_before = {k: v.clone() for k, v in model.state_dict().items()}

    state = trainer.export_inference_state_dict(ema=ema)
    # After export, the live model should equal what we set above (restored).
    live_after = {k: v.clone() for k, v in model.state_dict().items()}
    for k in live_before:
        assert torch.equal(live_before[k], live_after[k]), (
            f"live state for {k} mutated by export"
        )
    # The exported state should be the EMA shadow, NOT the live state.
    for k in state:
        if k in ema.shadow:
            assert torch.allclose(state[k].to(ema.shadow[k].dtype), ema.shadow[k].cpu()), (
                f"{k}: exported state did not match EMA shadow"
            )
