"""Tests for src/tac/learnable_bit_quant.py — Lane Ω-V2 LearnablePerElementBitDepth.

Pins:
  1. Per-element bits are learnable (registered as nn.Parameter).
  2. Gradient flows through BOTH the weight AND the bits.raw parameter
     when calling .backward() on a function of the quantized output.
  3. 8-bit init ≈ identity: max |q − w| < scale / 127.
  4. 1-bit forced: clusters to ±max per output channel (no zeros).
  5. Per-output-channel scale: max(|q|) per row equals max(|w|) per row.
  6. softplus parameterization: bits stay >= 1 even with very negative raw.
  7. Warm-start tensor: forward bits ≈ warm_start values.
  8. bits_rounded() → uint8 in [1, 8].
  9. LearnableBitConv2d wraps a Conv2d, forward matches manual fake-quant.
 10. Determinism: same seed → identical outputs.
 11. Round 13 (C-1): controller + explicit target raises ValueError.
 12. Round 13 (C-2): rate penalty zero-tensor inherits the model's device.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.learnable_bit_quant import (
    LagrangianRateController,
    LearnableBitConv2d,
    LearnablePerElementBitDepth,
    _PerElementSTEQuantize,
    _zero_on_device,
    compute_learnable_bit_rate_penalty,
    renderer_total_learnable_weight_bits,
)


# ── LearnablePerElementBitDepth basics ───────────────────────────────────


def test_bits_is_learnable_parameter():
    """raw must be an nn.Parameter (not a buffer)."""
    bd = LearnablePerElementBitDepth((4, 3, 3, 3), init_bits=8.0)
    assert isinstance(bd.raw, nn.Parameter)
    assert bd.raw.requires_grad


def test_bits_used_in_range():
    """bits_used() always returns values in [1, 8]."""
    bd = LearnablePerElementBitDepth((2, 3), init_bits=4.0)
    bits = bd.bits_used()
    assert (bits >= 1.0).all() and (bits <= 8.0).all()


def test_softplus_keeps_bits_nonnegative():
    """Even when raw is very negative, softplus pulls bits toward 0+; the
    clamp(min=1) keeps the forward representable. Verifies the
    parameterisation NEVER allows bits < 1."""
    bd = LearnablePerElementBitDepth((2, 3), init_bits=8.0)
    with torch.no_grad():
        bd.raw.fill_(-100.0)  # softplus(-100) ≈ 0
    bits = bd.bits_used()
    assert (bits >= 1.0).all(), (
        "bits must stay >= 1 even with very negative raw (clamp + softplus)"
    )


def test_bits_rounded_in_range_and_uint8():
    bd = LearnablePerElementBitDepth((2, 3), init_bits=4.5)
    bits_int = bd.bits_rounded()
    assert bits_int.dtype == torch.uint8
    assert (bits_int >= 1).all() and (bits_int <= 8).all()


def test_warm_start_tensor_initialises_bits_correctly():
    """When warm_start is provided, bits_used() ≈ warm_start (within softplus
    rounding noise — the inverse-softplus init is exact in float64)."""
    target = torch.tensor([[1.5, 4.0, 7.0], [2.0, 6.5, 3.0]])
    bd = LearnablePerElementBitDepth((2, 3), init_bits=8.0, warm_start=target)
    bits = bd.bits_used()
    diff = (bits - target).abs()
    assert diff.max().item() < 1e-3, (
        f"warm_start mismatch: max diff {diff.max().item()}"
    )


def test_warm_start_shape_mismatch_raises():
    bad = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="warm_start shape"):
        LearnablePerElementBitDepth((2, 3), warm_start=bad)


def test_forward_shape_mismatch_raises():
    bd = LearnablePerElementBitDepth((2, 3))
    with pytest.raises(ValueError, match="weight shape"):
        bd(torch.zeros(3, 3))


# ── 8-bit identity ───────────────────────────────────────────────────────


def test_8bit_init_near_identity():
    torch.manual_seed(42)
    w = torch.randn(8, 16) * 0.5
    bd = LearnablePerElementBitDepth(w.shape, init_bits=8.0)
    q = bd(w)
    # 8-bit max diff = scale / 127 per channel
    scale = w.reshape(8, -1).abs().max(dim=1).values
    step = scale.max().item() / 127.0
    assert (q - w).abs().max().item() < step + 1e-6


# ── 1-bit clustering ─────────────────────────────────────────────────────


def test_1bit_forced_clusters_to_pm_scale():
    torch.manual_seed(0)
    w = torch.randn(4, 8) * 0.3
    # Force 1-bit on every element by setting raw to inverse-softplus(1)
    bd = LearnablePerElementBitDepth(w.shape, init_bits=1.0)
    q = bd(w)
    # Per row, every element should be ±max(|row|) — no zeros
    scale = w.reshape(4, -1).abs().max(dim=1).values
    for row in range(4):
        unique = torch.unique(q[row].abs()).tolist()
        assert len(unique) == 1, (
            f"row {row}: 1-bit unique abs should be 1, got {unique}"
        )
        assert abs(unique[0] - scale[row].item()) < 1e-5
    assert (q == 0).sum().item() == 0


def test_1bit_preserves_sign():
    torch.manual_seed(2)
    w = torch.randn(2, 4)
    bd = LearnablePerElementBitDepth(w.shape, init_bits=1.0)
    q = bd(w)
    same_sign = (q.sign() == w.sign()) | (w == 0)
    assert same_sign.all()


# ── Per-output-channel scale ─────────────────────────────────────────────


def test_per_channel_max_preserved():
    torch.manual_seed(5)
    w = torch.randn(4, 6) * 0.7
    for init_b in (2.0, 4.0, 8.0):
        bd = LearnablePerElementBitDepth(w.shape, init_bits=init_b)
        q = bd(w)
        max_w = w.reshape(4, -1).abs().max(dim=1).values
        max_q = q.reshape(4, -1).abs().max(dim=1).values
        assert torch.allclose(max_q, max_w, atol=1e-4), (
            f"init_bits={init_b}: per-channel max should be preserved"
        )


# ── Gradient flow (THE load-bearing property) ────────────────────────────


def test_gradient_flows_through_weight():
    torch.manual_seed(11)
    w = torch.randn(3, 5, requires_grad=True)
    bd = LearnablePerElementBitDepth(w.shape, init_bits=4.0)
    q = bd(w)
    (q * 2.0).sum().backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    # STE: when no saturation, grad ≈ upstream (2.0). Some saturation
    # near edges; pin a softer property.
    assert w.grad.abs().mean().item() > 0


def test_gradient_flows_through_bits():
    """The single most important property of Lane Ω-V2: the bits
    parameter receives gradient from the quantized output, so Lagrangian
    dual ascent on λ + Adam on bits.raw can converge to the KKT optimum.

    Round 26 IMPORTANT finding fix: also assert SIGN/direction (not just
    finiteness) — Round 21 caught a sign-inversion bug that 4 prior rounds
    missed precisely because tests like this only checked grad-not-None.
    """
    torch.manual_seed(13)
    w = torch.randn(3, 5, requires_grad=False)
    bd = LearnablePerElementBitDepth(w.shape, init_bits=4.0)
    q = bd(w)
    (q.sum()).backward()
    assert bd.raw.grad is not None
    assert torch.isfinite(bd.raw.grad).all()
    # Direction anchor: at least one element must have non-zero gradient
    # (proves the path actually fires; an always-zero backward would silently
    # pass the finiteness check above).
    assert bd.raw.grad.abs().max().item() > 0, (
        "bits gradient is identically zero — STE backward not flowing"
    )


def test_higher_bits_means_lower_quant_error():
    """Sanity: at 8-bit init, max diff < scale/127. At 2-bit init, max
    diff > scale/127 (much coarser). Verifies bit-depth actually
    affects fidelity (no dead path)."""
    torch.manual_seed(0)
    w = torch.randn(4, 16) * 0.5
    bd_high = LearnablePerElementBitDepth(w.shape, init_bits=8.0)
    bd_low = LearnablePerElementBitDepth(w.shape, init_bits=2.0)
    q_high = bd_high(w)
    q_low = bd_low(w)
    err_high = (q_high - w).abs().mean().item()
    err_low = (q_low - w).abs().mean().item()
    assert err_low > err_high, (
        f"low-bit error should be > high-bit error; got "
        f"low={err_low:.4f} high={err_high:.4f}"
    )


# ── LearnableBitConv2d wrapper ───────────────────────────────────────────


def test_learnable_bit_conv2d_forward_shape():
    layer = LearnableBitConv2d(3, 4, 3, padding=1)
    x = torch.randn(2, 3, 8, 8)
    y = layer(x)
    assert y.shape == (2, 4, 8, 8)


def test_learnable_bit_conv2d_8bit_close_to_fp32_conv():
    """At 8-bit init, output should be within fp16-quant noise of a plain
    Conv2d with the same weights."""
    torch.manual_seed(7)
    layer = LearnableBitConv2d(3, 4, 3, padding=1, init_bits=8.0)
    plain = nn.Conv2d(3, 4, 3, padding=1)
    with torch.no_grad():
        plain.weight.copy_(layer.conv.weight)
        plain.bias.copy_(layer.conv.bias)
    x = torch.randn(2, 3, 8, 8)
    y_quant = layer(x)
    y_plain = plain(x)
    # 8-bit per-channel quant error bounded by scale * |x| / 127 per output
    diff = (y_quant - y_plain).abs().mean().item()
    assert diff < 0.2, f"8-bit quant should be near-identity; got mean diff {diff}"


def test_learnable_bit_conv2d_gradient_flows_through_bits():
    torch.manual_seed(9)
    layer = LearnableBitConv2d(3, 4, 3, padding=1)
    x = torch.randn(1, 3, 8, 8)
    y = layer(x)
    y.sum().backward()
    assert layer.conv.weight.grad is not None
    assert layer.bit_depth.raw.grad is not None
    # Round 28 magnitude anchor: gradient must be NON-ZERO, not just present.
    # Round-26-class gap: presence-only checks pass with all-zero grads.
    assert layer.bit_depth.raw.grad.abs().max().item() > 0
    # The bias also gets grad
    assert layer.conv.bias.grad is not None


def test_learnable_bit_conv2d_total_weight_bits_differentiable():
    layer = LearnableBitConv2d(3, 4, 3)
    total = layer.total_weight_bits()
    assert total.requires_grad
    total.backward()
    assert layer.bit_depth.raw.grad is not None


def test_learnable_bit_conv2d_stride_and_dilation_work():
    """Lane S regression — the wrapper has to support all Conv2d kwargs
    so it can drop into the renderer's downsample / dilated paths."""
    layer = LearnableBitConv2d(8, 16, 3, stride=2, padding=1)
    y = layer(torch.randn(1, 8, 16, 16))
    assert y.shape == (1, 16, 8, 8)

    layer2 = LearnableBitConv2d(8, 8, 3, padding=2, dilation=2)
    y2 = layer2(torch.randn(1, 8, 16, 16))
    assert y2.shape == (1, 8, 16, 16)


def test_learnable_bit_conv2d_replicate_padding_works():
    layer = LearnableBitConv2d(4, 4, 3, padding=1, padding_mode="replicate")
    y = layer(torch.randn(1, 4, 8, 8))
    assert y.shape == (1, 4, 8, 8)


# ── Determinism ──────────────────────────────────────────────────────────


def test_deterministic_for_seed():
    torch.manual_seed(123)
    layer1 = LearnableBitConv2d(3, 4, 3, padding=1)
    x = torch.randn(1, 3, 8, 8)
    y1 = layer1(x)

    torch.manual_seed(123)
    layer2 = LearnableBitConv2d(3, 4, 3, padding=1)
    y2 = layer2(x)

    assert torch.equal(y1, y2)


# ── Finiteness ────────────────────────────────────────────────────────────


def test_outputs_finite_at_all_bit_levels():
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    for b in (1.0, 2.0, 4.0, 8.0):
        bd = LearnablePerElementBitDepth(w.shape, init_bits=b)
        q = bd(w)
        assert torch.isfinite(q).all()


# ── Round 10 upstream-signed bit-gradient chain rule ─────────────────────


def test_round10_zero_bin_mse_upstream_preserves_sign():
    """w=0.1, b=2 gives q=0, q_bplus=0, q_bminus=+1.

    A small bit decrease increases the local proxy MSE, so the STE
    gradient wrt bits must be negative and SGD must raise bits.

    Round 21 derivation (anti-arbitrariness):
      step = 1/(2^(b-1)-1) = 1 at b=2, so q = round(0.1)*1 = 0.
      q_bplus  (b=3): step=1/3, round(0.3)*1/3 = 0.
      q_bminus (b=1): sign quantizer, w=0.1>=0 → q_bminus = +1.
      grad_output (∂L/∂q for L=½(q-w)²) = q-w = -0.1.
      sq_err_bplus  = (0  - 0.1)² = 0.01
      sq_err_bminus = (1  - 0.1)² = 0.81
      local_diff    = (0.01 - 0.81)/2 = -0.40
      grad_bits     = |-0.1| · -0.40 = -0.04   ← expected
      True L(b=1)=0.405, L(b=2)=0.005, L(b=3)=0.005. SGD on negative
      gradient pushes b up; b=2 is already optimal so step is small.
    """
    scale = torch.ones(1, dtype=torch.float64)
    weight = torch.tensor([[[[0.1]]]], dtype=torch.float64, requires_grad=True)
    bits = torch.tensor([[[[2.0]]]], dtype=torch.float64, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)

    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()

    with torch.no_grad():
        q_decrease = _PerElementSTEQuantize.apply(weight.detach(), scale, bits.detach() - 1.0)
        loss_decrease = 0.5 * ((q_decrease - weight.detach()) ** 2).sum()

    assert bits.grad is not None
    assert loss_decrease.item() > loss.item()
    assert bits.grad.item() < 0.0
    assert bits.grad.item() == pytest.approx(-0.04, abs=1e-12)


def test_round10_synthetic_ce_like_upstream_sign_flips_grad_bits():
    """The MSE proxy uses abs(upstream) as a scale; equal-magnitude
    synthetic upstream signs should not invert the bit-depth direction.

    Round 21 derivation (anti-arbitrariness):
      Two elements, both w=0.1, b=2 → same q, q_bplus, q_bminus as above.
      sq_err_bplus  = 0.01, sq_err_bminus = 0.81, local_diff = -0.40.
      Element 0: grad_output=+2.0 → grad_bits = |+2.0| · -0.40 = -0.80.
      Element 1: grad_output=-2.0 → grad_bits = |-2.0| · -0.40 = -0.80.
      Both negative because the local distortion landscape rewards
      raising bits regardless of upstream sign — the upstream is just
      a magnitude scaler. This is the canonical Round 7 invariant the
      Round 10 linearisation broke (it returned ±1.0 instead of -0.80).
    """
    scale = torch.ones(2, dtype=torch.float64)
    weight = torch.tensor([[[[0.1]]], [[[0.1]]]], dtype=torch.float64, requires_grad=True)
    bits = torch.full_like(weight, 2.0, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)

    q.backward(torch.tensor([[[[2.0]]], [[[-2.0]]]], dtype=torch.float64))

    assert bits.grad is not None
    assert bits.grad[0].item() == pytest.approx(-0.8, abs=1e-12)
    assert bits.grad[1].item() == pytest.approx(-0.8, abs=1e-12)


def test_round10_random_bit_gradient_descent_reduces_local_loss_expectation():
    g = torch.Generator().manual_seed(2026_04_28)
    scale = torch.ones(20, dtype=torch.float64)
    weight = (torch.rand(20, 1, 1, 1, generator=g, dtype=torch.float64) * 1.6 - 0.8)
    bits = (2.0 + 2.0 * torch.rand_like(weight, generator=g)).detach()

    def loss_at(bits_value: torch.Tensor) -> torch.Tensor:
        q = _PerElementSTEQuantize.apply(weight, scale, bits_value)
        return 0.5 * ((q - weight) ** 2).mean()

    initial = float(loss_at(bits).item())
    bits_param = bits.clone().requires_grad_(True)
    for _ in range(80):
        if bits_param.grad is not None:
            bits_param.grad.zero_()
        loss = loss_at(bits_param)
        loss.backward()
        with torch.no_grad():
            bits_param.sub_(0.2 * bits_param.grad)
            bits_param.clamp_(1.0, 8.0)

    final = float(loss_at(bits_param.detach()).item())
    assert final <= initial


# ── Round 13 R12-C1: controller + explicit target is an error ───────────


def _build_tiny_swapped_model() -> nn.Module:
    """Tiny renderer-shaped Sequential with a swapped LearnableBitConv2d
    so the rate-penalty / total-bits helpers have something to count."""
    return nn.Sequential(
        LearnableBitConv2d(4, 4, 3, padding=1, init_bits=8.0),
        nn.ReLU(),
        LearnableBitConv2d(4, 4, 3, padding=1, init_bits=8.0),
    )


def test_r13_c1_controller_with_explicit_target_raises_value_error():
    """C-1 regression: passing a controller AND an explicit
    ``target_bits_per_weight`` is now a hard ValueError. Previously the
    explicit value was silently discarded — making downstream call sites
    look correct in code review while the function ignored their input.
    """
    model = _build_tiny_swapped_model()
    ctrl = LagrangianRateController(target_bits_per_weight=2.0, eta=0.1)
    with pytest.raises(ValueError, match="LagrangianRateController"):
        compute_learnable_bit_rate_penalty(
            model,
            target_bits_per_weight=4.0,  # contradicts the controller
            lambda_rate=ctrl,
        )


def test_r13_c1_controller_with_none_target_works():
    """C-1 contract: passing target=None with a controller is the
    canonical happy path. The controller's stored target (2.0) drives
    the residual."""
    model = _build_tiny_swapped_model()
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, eta=0.1, initial_lambda=1.0,
    )
    pen = compute_learnable_bit_rate_penalty(
        model, target_bits_per_weight=None, lambda_rate=ctrl,
    )
    # mean_bits ≈ 8 → residual ≈ 6 → pen ≈ 1.0 · 6 = 6.
    assert pen.item() == pytest.approx(6.0, rel=1e-4)


def test_r13_c1_legacy_float_path_requires_explicit_target():
    """C-1 contract: legacy float-multiplier callers must STILL pass an
    explicit ``target_bits_per_weight`` (no controller to fall back on)."""
    model = _build_tiny_swapped_model()
    with pytest.raises(ValueError, match="legacy float"):
        compute_learnable_bit_rate_penalty(
            model, target_bits_per_weight=None, lambda_rate=1.0,
        )


# ── Round 13 R12-C2: zero-tensor early-exit respects device ─────────────


def test_r13_c2_zero_on_device_helper_default():
    """``_zero_on_device(None)`` returns a zero-dim tensor on the default
    (typically CPU) device."""
    z = _zero_on_device(None)
    assert z.dim() == 0
    assert z.item() == 0.0


def test_r13_c2_zero_on_device_helper_meta():
    """``_zero_on_device('meta')`` allows callers to construct a
    device-specific zero without instantiating a full tensor on
    accelerator hardware."""
    z = _zero_on_device(torch.device("meta"))
    assert z.dim() == 0
    assert z.device.type == "meta"


def test_r13_c2_renderer_total_bits_lives_on_layer_device():
    """C-2 regression: when a model is moved to a device, the total-bits
    helper must return a tensor on the SAME device — otherwise
    ``loss_on_X + total_bits`` raises a device mismatch."""
    model = _build_tiny_swapped_model().to(torch.device("meta"))
    total = renderer_total_learnable_weight_bits(model)
    assert total.device.type == "meta", (
        f"total_bits should follow layer device; got {total.device}"
    )


def test_r13_c2_rate_penalty_lives_on_layer_device():
    """C-2 regression: the rate penalty's residual + scalar arithmetic
    must stay on the layer's raw.device. Previously the no-layer
    early-exit and the in-loop ``torch.zeros((), device=device)`` were
    out of sync — the early-exit forced CPU."""
    model = _build_tiny_swapped_model().to(torch.device("meta"))
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, initial_lambda=1.0,
    )
    pen = compute_learnable_bit_rate_penalty(
        model, target_bits_per_weight=None, lambda_rate=ctrl,
    )
    assert pen.device.type == "meta", (
        f"rate penalty should follow layer device; got {pen.device}"
    )


def test_r13_c2_rate_penalty_no_layers_returns_zero_tensor():
    """C-2 contract: the no-layer early-exit must STILL return a
    zero-dim tensor so callers can ``loss + penalty`` unconditionally."""
    plain_model = nn.Sequential(nn.Linear(4, 4))
    pen = compute_learnable_bit_rate_penalty(
        plain_model, target_bits_per_weight=2.0, lambda_rate=1.0,
    )
    assert pen.dim() == 0
    assert pen.item() == 0.0


def test_r13_c2_total_bits_no_layers_returns_zero_tensor():
    """C-2 contract: empty-model early-exit returns a zero-dim tensor."""
    plain_model = nn.Sequential(nn.Linear(4, 4))
    total = renderer_total_learnable_weight_bits(plain_model)
    assert total.dim() == 0
    assert total.item() == 0.0


# ── Round 14 finding 3 (R15) — zero-layer model device inference ────────


def test_round15_finding3_zero_layer_total_bits_inherits_model_device():
    """R14-3 regression: a model with ZERO LearnableBitConv2d layers
    that has been moved to a non-CPU device must still have its
    ``renderer_total_learnable_weight_bits`` return a tensor on the
    model's device — otherwise ``loss_on_X + total_bits`` raises
    a device-mismatch crash. Round 13 fixed this for the LAYERED
    path; Round 14 closes the no-layer early-exit gap.
    """
    plain_model = nn.Sequential(nn.Linear(4, 4)).to(torch.device("meta"))
    total = renderer_total_learnable_weight_bits(plain_model)
    assert total.device.type == "meta", (
        f"zero-layer total_bits should follow model device; got "
        f"{total.device}"
    )


def test_round15_finding3_zero_layer_rate_penalty_inherits_model_device():
    """R14-3 regression: same as above but for
    ``compute_learnable_bit_rate_penalty`` — the no-layer early-exit
    must return a tensor on the model's device so a CUDA-resident
    model doesn't crash on ``cuda_loss + cpu_zero``.
    """
    plain_model = nn.Sequential(nn.Linear(4, 4)).to(torch.device("meta"))
    pen = compute_learnable_bit_rate_penalty(
        plain_model, target_bits_per_weight=2.0, lambda_rate=1.0,
    )
    assert pen.device.type == "meta", (
        f"zero-layer rate penalty should follow model device; got "
        f"{pen.device}"
    )


def test_round15_finding3_completely_empty_model_falls_back_to_cpu():
    """R14-3 contract: a model with NO parameters AND NO buffers
    (e.g. a freshly-constructed empty Sequential) cannot have its
    device inferred — fall back to the default CPU device rather
    than raising. The caller's downstream binary op may still
    crash, but at least the helper itself does not.
    """
    empty_model = nn.Sequential()
    total = renderer_total_learnable_weight_bits(empty_model)
    assert total.dim() == 0
    assert total.item() == 0.0
    # Default device — for a typical workstation this is CPU.
    pen = compute_learnable_bit_rate_penalty(
        empty_model, target_bits_per_weight=2.0, lambda_rate=1.0,
    )
    assert pen.dim() == 0
    assert pen.item() == 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only regression test for the device-mismatch crash.",
)
def test_round15_finding3_zero_layer_model_on_cuda_no_device_mismatch():
    """R14-3 regression: the actual real-CUDA path. A zero-layer model
    on CUDA, plus a CUDA loss, plus the rate penalty — must NOT raise
    a device-mismatch. This is the user-facing crash that finding 3
    catches: under the previous code ``cuda_loss + _zero_on_device(None)``
    crashed because the zero was on CPU.
    """
    cuda = torch.device("cuda")
    plain_model = nn.Sequential(nn.Linear(4, 4)).to(cuda)
    cuda_loss = torch.tensor(1.0, device=cuda)
    total = renderer_total_learnable_weight_bits(plain_model)
    pen = compute_learnable_bit_rate_penalty(
        plain_model, target_bits_per_weight=2.0, lambda_rate=1.0,
    )
    # The load-bearing assertion: this addition must not crash.
    out_total = cuda_loss + total
    out_pen = cuda_loss + pen
    assert out_total.device == cuda
    assert out_pen.device == cuda
