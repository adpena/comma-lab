"""Tests for learnable Selfcomp grayscale-LUT class targets."""
from __future__ import annotations

import pytest
import torch

from tac.learnable_class_targets import LearnableClassTargets
from tac.mask_grayscale_lut import CLASS_TO_GRAY, NUM_CLASSES


def test_default_init_matches_hardcoded() -> None:
    targets = LearnableClassTargets()
    expected = torch.tensor([CLASS_TO_GRAY[c] for c in range(NUM_CLASSES)], dtype=torch.float32)
    assert torch.equal(targets(), expected)


def test_forward_clamps_to_0_255_range() -> None:
    targets = LearnableClassTargets(torch.tensor([-10.0, 12.0, 64.0, 192.0, 300.0]))
    out = targets()
    assert out.min() >= 0.0
    assert out.max() <= 255.0
    assert out.tolist() == [0.0, 12.0, 64.0, 192.0, 255.0]


def test_enforce_separation_min_gap_32() -> None:
    targets = LearnableClassTargets(torch.tensor([128.0, 128.0, 129.0, 130.0, 131.0]))
    returned = targets.enforce_separation(min_gap=32.0)
    assert returned is targets
    sorted_targets = torch.sort(targets()).values
    assert torch.all(sorted_targets[1:] - sorted_targets[:-1] >= 32.0 - 1e-5)
    assert sorted_targets.min() >= 0.0
    assert sorted_targets.max() <= 255.0


def test_ema_update_moves_targets_toward_assignments() -> None:
    targets = LearnableClassTargets(
        torch.tensor([0.0, 255.0, 64.0, 192.0, 128.0]),
        ema_decay=0.5,
    )
    before = targets().clone()
    assignments = torch.tensor([2, 2, 2, 3, 3, 4], dtype=torch.long)
    gray_values = torch.tensor([90.0, 92.0, 94.0, 170.0, 172.0, 140.0])
    targets.ema_update(assignments, gray_values)
    after = targets()
    assert after[2] > before[2]
    assert after[3] < before[3]
    assert after[4] > before[4]


def test_serialize_deserialize_roundtrip_fp16() -> None:
    source = LearnableClassTargets(torch.tensor([0.0, 255.0, 64.25, 192.5, 128.75]))
    restored = LearnableClassTargets.deserialize_from_bytes(source.serialize_to_bytes())
    assert torch.allclose(restored(), source().to(torch.float16).to(torch.float32))


def test_serialize_size_is_10_bytes() -> None:
    assert len(LearnableClassTargets().serialize_to_bytes()) == 10


def test_mps_device_rejected_at_construction() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable on this runner")
    with pytest.raises(RuntimeError, match="MPS"):
        LearnableClassTargets(device=torch.device("mps"))


def test_gradient_flow_through_targets() -> None:
    targets = LearnableClassTargets()
    loss = ((targets() - torch.tensor([10.0, 245.0, 70.0, 180.0, 130.0])) ** 2).sum()
    loss.backward()
    assert targets.raw_values.grad is not None
    assert torch.isfinite(targets.raw_values.grad).all()
    assert targets.raw_values.grad.abs().sum() > 0
