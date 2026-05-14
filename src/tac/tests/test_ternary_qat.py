# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch

from tac.optimization.ternary_qat import (
    TernaryQATConfig,
    calibrate_ternary_tensor,
    dequantize_ternary_tensor,
    pack_ternary_tensor,
    quantize_ternary_tensor,
    ternary_ste,
    unpack_ternary_tensor,
)


def test_score_sensitivity_changes_rate_distortion_threshold() -> None:
    tensor = torch.tensor([0.25, 1.0])
    cfg = TernaryQATConfig(rate_lambda=0.05)

    protect_small = calibrate_ternary_tensor(
        tensor,
        score_sensitivity=torch.tensor([100.0, 1.0]),
        config=cfg,
    )
    protect_large = calibrate_ternary_tensor(
        tensor,
        score_sensitivity=torch.tensor([1.0, 100.0]),
        config=cfg,
    )

    assert protect_small.nonzero_count == 2
    assert protect_large.nonzero_count == 1
    assert protect_small.to_dict()["score_claim"] is False
    assert protect_small.to_dict()["promotion_eligible"] is False


def test_ternary_pack_unpack_roundtrip_is_deterministic() -> None:
    tensor = torch.tensor([[-2.0, -0.1, 0.0], [0.2, 3.0, 4.0]])
    quantized = quantize_ternary_tensor(tensor)

    blob1 = pack_ternary_tensor(quantized)
    blob2 = quantized.pack()
    restored = unpack_ternary_tensor(blob1)

    assert blob1 == blob2
    assert torch.equal(restored.codes, quantized.codes)
    assert restored.calibration == quantized.calibration
    assert torch.allclose(
        dequantize_ternary_tensor(restored),
        quantized.dequantize(),
    )


def test_ternary_ste_uses_quantized_forward_identity_backward() -> None:
    tensor = torch.tensor([-2.0, -0.1, 3.0], requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0])
    y = ternary_ste(tensor)

    assert not torch.allclose(y.detach(), tensor.detach())
    (y * weights).sum().backward()
    assert torch.allclose(tensor.grad, weights)


def test_ternary_all_zero_tensor_is_stable() -> None:
    quantized = quantize_ternary_tensor(torch.zeros(2, 3))

    assert quantized.calibration.nonzero_count == 0
    assert quantized.calibration.scale == 1.0
    assert torch.equal(quantized.codes, torch.zeros(2, 3, dtype=torch.int8))


def test_ternary_empty_tensor_is_packable_noop() -> None:
    quantized = quantize_ternary_tensor(torch.empty(0))
    blob = quantized.pack()
    restored = unpack_ternary_tensor(blob)

    assert restored.calibration.numel == 0
    assert restored.calibration.score_weight_sum == 0.0
    assert restored.codes.numel() == 0
    assert restored.dequantize().numel() == 0


def test_ternary_rejects_bad_inputs_and_corrupt_blobs() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        quantize_ternary_tensor(torch.tensor([1, 2], dtype=torch.int64))
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_ternary_tensor(
            torch.ones(2),
            score_sensitivity=torch.tensor([1.0, -1.0]),
        )

    quantized = quantize_ternary_tensor(torch.tensor([1.0, -1.0]))
    blob = bytearray(quantized.pack())
    blob[0:4] = b"BAD!"
    with pytest.raises(ValueError, match="magic"):
        unpack_ternary_tensor(bytes(blob))
