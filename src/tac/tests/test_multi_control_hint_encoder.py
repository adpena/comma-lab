# SPDX-License-Identifier: MIT
import torch

from tac.contrib.multi_control_hint_encoder import MultiControlHintEncoder


def test_shape_correctness_for_long_masks():
    encoder = MultiControlHintEncoder()
    masks = torch.randint(0, 5, (2, 12, 16), dtype=torch.long)

    out = encoder(masks)

    assert out.shape == (2, 256, 12, 16)


def test_zero_init_output_is_zero():
    encoder = MultiControlHintEncoder()
    masks = torch.randint(0, 5, (1, 8, 10), dtype=torch.long)

    out, weight = encoder(masks, return_weight_map=True)

    assert torch.count_nonzero(out) == 0
    assert torch.count_nonzero(weight) == 0


def test_gradient_flows_to_zero_projection():
    encoder = MultiControlHintEncoder()
    masks = torch.nn.functional.one_hot(
        torch.randint(0, 5, (1, 8, 8)), num_classes=5
    ).permute(0, 3, 1, 2).float()

    out = encoder(masks)
    out.sum().backward()

    grads = [
        p.grad.abs().sum().item()
        for p in encoder.parameters()
        if p.grad is not None
    ]
    assert any(g > 0 for g in grads)


def test_out_channels_configurable():
    encoder = MultiControlHintEncoder(out_channels=64)
    masks = torch.randint(0, 5, (1, 6, 7), dtype=torch.long)

    out = encoder(masks)

    assert out.shape == (1, 64, 6, 7)


def test_works_with_batch_greater_than_one():
    encoder = MultiControlHintEncoder(out_channels=32)
    masks = torch.randint(0, 5, (4, 9, 11), dtype=torch.long)

    out, weight = encoder(masks, return_weight_map=True)

    assert out.shape[0] == 4
    assert weight.shape == (4, 1, 9, 11)


def test_output_dtype_is_float32():
    encoder = MultiControlHintEncoder()
    masks = torch.randint(0, 5, (1, 5, 6), dtype=torch.long)

    out, weight = encoder(masks, return_weight_map=True)

    assert out.dtype == torch.float32
    assert weight.dtype == torch.float32


def test_default_parameter_count_stays_near_lane_budget():
    encoder = MultiControlHintEncoder()

    n_params = sum(p.numel() for p in encoder.parameters())

    assert 5_000 <= n_params <= 8_000
