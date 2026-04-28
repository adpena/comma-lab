from __future__ import annotations

import torch

from tac.entropy_bottleneck import EntropyBottleneck


def test_forward_returns_correct_shapes() -> None:
    eb = EntropyBottleneck(num_channels=3)
    y = torch.randn(2, 3, 4, 5)

    y_hat, bits = eb(y)

    assert y_hat.shape == y.shape
    assert bits.shape == torch.Size([])


def test_bits_per_element_is_positive_scalar() -> None:
    eb = EntropyBottleneck(num_channels=4)
    y = torch.randn(2, 4, 3, 3)

    _, bits = eb(y)

    assert bits.ndim == 0
    assert bits.item() > 0.0


def test_training_mode_uses_noisy_y_not_rounded() -> None:
    torch.manual_seed(123)
    eb = EntropyBottleneck(num_channels=1)
    eb.train()
    y = torch.full((16, 1, 2, 2), 0.25)

    y_hat, _ = eb(y)

    assert not torch.equal(y_hat, y.round())
    assert torch.all((y_hat - y) >= -0.5)
    assert torch.all((y_hat - y) <= 0.5)


def test_eval_mode_uses_rounded_y() -> None:
    eb = EntropyBottleneck(num_channels=2)
    eb.eval()
    y = torch.tensor([[[[0.25]], [[1.60]]]])

    y_hat, _ = eb(y)

    assert torch.equal(y_hat, y.round())


def test_rate_loss_matches_bits_from_forward() -> None:
    eb = EntropyBottleneck(num_channels=3)
    y = torch.randn(2, 3, 3, 3)

    _, bits = eb(y)

    assert torch.equal(eb.rate_loss(), bits)


def test_gradients_flow_through_y_hat_in_training_mode() -> None:
    torch.manual_seed(77)
    eb = EntropyBottleneck(num_channels=2)
    eb.train()
    y = torch.randn(2, 2, 3, 3, requires_grad=True)

    y_hat, _ = eb(y)
    y_hat.sum().backward()

    assert y.grad is not None
    assert torch.count_nonzero(y.grad).item() == y.numel()


def test_two_dim_input_uses_channel_dim_only() -> None:
    """Lane EBR must work for plain (B, C) latents, not just (B, C, H, W)."""
    eb = EntropyBottleneck(num_channels=4)
    y = torch.randn(8, 4)

    y_hat, bits = eb(y)

    assert y_hat.shape == y.shape
    assert torch.isfinite(bits).item()
    assert bits.item() > 0.0


def test_overflow_safety_under_large_logits() -> None:
    """Ballé bottlenecks NaN at sigmoid extremes; verify finite bits at 1e6."""
    eb = EntropyBottleneck(num_channels=2)
    y = torch.tensor([[[[1.0e6]], [[-1.0e6]]]])

    y_hat, bits = eb(y)

    assert torch.isfinite(y_hat).all().item()
    assert torch.isfinite(bits).item()
    # Saturated CDF gives bounded but huge bits; cap is the float32 ceiling.
    assert bits.item() < 100.0


def test_collapsed_scale_does_not_emit_nan() -> None:
    """Tiny softplus(raw_scale) used to break the (upper-lower) clamp."""
    eb = EntropyBottleneck(num_channels=1)
    with torch.no_grad():
        eb.raw_scale.fill_(-50.0)
    eb.eval()
    y = torch.tensor([[[[0.1]]]])

    _, bits = eb(y)

    assert torch.isfinite(bits).item()


def test_train_eval_quantization_gap_matches_round() -> None:
    """eval mode must produce torch.round(y); train mode must NOT."""
    torch.manual_seed(0)
    eb = EntropyBottleneck(num_channels=2)
    y = torch.tensor([[[[0.49]], [[1.51]]]])

    eb.train()
    y_train, _ = eb(y)
    eb.eval()
    y_eval, _ = eb(y)

    assert not torch.equal(y_train, y.round())
    assert torch.equal(y_eval, y.round())


def test_state_dict_roundtrip_preserves_bits() -> None:
    """serialize/deserialize via state_dict must preserve the rate prediction."""
    torch.manual_seed(11)
    eb_a = EntropyBottleneck(num_channels=4)
    eb_a.eval()
    # Move parameters away from init so the test is non-trivial.
    with torch.no_grad():
        eb_a.loc.copy_(torch.randn(4))
        eb_a.raw_scale.copy_(torch.randn(4))
        eb_a.raw_shape.copy_(torch.randn(4))

    state = eb_a.state_dict()
    eb_b = EntropyBottleneck(num_channels=4)
    eb_b.load_state_dict(state)
    eb_b.eval()

    y = torch.randn(2, 4, 3, 3)
    _, bits_a = eb_a(y)
    _, bits_b = eb_b(y)

    assert torch.allclose(bits_a, bits_b)


def test_rate_decreases_under_optimization_loop() -> None:
    """Integration test: optimizing the EB params on fixed latents must reduce bits.

    This proves the bottleneck actually learns — a regression here would catch
    a silently-broken gradient path or a frozen prior.
    """
    torch.manual_seed(2026)
    eb = EntropyBottleneck(num_channels=4, init_scale=10.0)
    eb.train()
    # Fixed latents with small variance — the optimal logistic prior should
    # tighten its scale and drive bits down.
    latents = torch.randn(16, 4, 8, 8) * 0.5
    optim = torch.optim.Adam(eb.parameters(), lr=5e-2)

    bit_history: list[float] = []
    for _ in range(40):
        optim.zero_grad()
        _, bits = eb(latents)
        bit_history.append(bits.item())
        bits.backward()
        optim.step()

    first = sum(bit_history[:5]) / 5.0
    last = sum(bit_history[-5:]) / 5.0
    # First 5 vs last 5 average — must drop materially (>=20%).
    assert last < first * 0.8, (
        f"Lane EBR did not learn: first5={first:.4f} last5={last:.4f} "
        f"history_min={min(bit_history):.4f} history_max={max(bit_history):.4f}"
    )
