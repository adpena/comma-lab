"""Bug 1 (codex Round 3) regression: bit-depth STE must produce a
non-zero gradient for weights that quantize to the zero bin.

Scenario
--------
A weight whose magnitude is *small but non-zero* (say |w| < step / 2)
gets rounded to ``q = 0`` by the per-element fake-quantizer at the
current bit-depth. The previous backward formula
``grad_bits = grad_output · (-ln2 · q)`` is then exactly zero — the
rate-distortion allocator therefore receives no signal that allocating
more bits to this weight would let it round to a non-zero level (i.e.
let the weight participate in the layer's representation at all).
Combined with the rate penalty that pushes bits down, allocation is
broken: small-but-important weights cannot earn more bits.

The fix uses ``grad_bits = grad_output · (-ln2 · w_unquantized)`` (the
ratio ``w/step`` times ``∂step/∂bits = -ln2 · step`` cancels the step,
leaving a signal proportional to the original weight). This survives
the zero-bin rounding because ``w`` itself is non-zero.

What this test asserts
----------------------
1. Set up a small weight tensor that *will* round to zero at low bit
   depth (e.g. magnitude 0.1 · scale at 1-bit).
2. Compute the bit gradient via backward.
3. Verify ``|grad_bits|`` for those zero-bin elements is strictly
   greater than zero, so the rate-distortion allocator has signal.
4. Verify the *sign* of the gradient is consistent with the direction
   that decreases distortion when bits increase: a positive distortion
   gradient (grad_output > 0) on a positive weight should produce a
   *negative* grad_bits (since ``∂q/∂bits ∝ +w``, by the standard
   ``-ln2 · w`` surrogate). Concretely: increasing bits → finer step →
   smaller quantization error.
"""
from __future__ import annotations

import torch

from tac.learnable_bit_quant import _PerElementSTEQuantize


def _make_weight_with_subbin_elements(
    *,
    n_chan: int = 4,
    h: int = 1,
    w: int = 1,
    bits: float = 1.0,
    sub_bin_frac: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(weight, scale, bits_continuous)`` where the absolute
    values of every element are below the half-step threshold (i.e.
    they will quantize to zero in the general path).

    For the 1-bit branch we keep the same setup so the *general* code
    path's zero-bin behavior is exercised even though 1-bit itself uses
    the sign quantizer.
    """
    # Per-channel scale = 1.0 for simplicity.
    scale = torch.ones(n_chan, dtype=torch.float64)
    # Each element is sub_bin_frac · scale → well under one step at any
    # bit depth >= 2 (where step = scale / (2^(b-1) - 1) ≥ scale).
    sign = torch.tensor([+1.0, -1.0, +1.0, -1.0], dtype=torch.float64)
    weight = (sign * sub_bin_frac).reshape(n_chan, 1, 1, 1).expand(
        n_chan, 1, h, w
    ).clone()
    bits_continuous = torch.full_like(weight, bits, dtype=torch.float64)
    return weight, scale, bits_continuous


def test_zero_bin_elements_have_nonzero_bit_gradient_general_path():
    """Bug 1 core test: weights that round to the zero bin (general
    quantizer path, bits >= 2) must still receive non-zero bit gradient.
    The fix is to use the unquantized weight in the surrogate."""
    weight, scale, bits = _make_weight_with_subbin_elements(
        bits=2.0, sub_bin_frac=0.1,
    )
    # Make the inputs gradient-ready.
    weight.requires_grad_(True)
    bits_param = bits.detach().clone().requires_grad_(True)

    # Forward through the STE — the result q must be all zeros for our
    # sub-bin weights (validates our test setup).
    q = _PerElementSTEQuantize.apply(weight, scale, bits_param)
    # Sanity: at 2-bit, levels = 2^(2-1) - 1 = 1, step = scale / 1 = 1.0,
    # round(0.1 / 1.0) = 0 → q = 0 everywhere.
    assert torch.allclose(q, torch.zeros_like(q)), (
        f"test setup failure: q expected all zeros, got {q.tolist()}"
    )

    # Backward: pretend the upstream gradient is +1 everywhere (we just
    # want to measure the magnitude of the bit gradient).
    grad_out = torch.ones_like(q)
    q.backward(grad_out)

    # The fix's load-bearing assertion: the bit gradient is NON-ZERO
    # for these zero-bin elements. Pre-fix value was exactly 0 (because
    # ``-ln2 · q`` is zero when ``q = 0``).
    assert bits_param.grad is not None
    assert torch.all(bits_param.grad.abs() > 0), (
        f"bit gradient is zero on zero-bin elements: "
        f"{bits_param.grad.tolist()} — Bug 1 not fixed."
    )


def test_zero_bin_bit_gradient_sign_reduces_distortion():
    """Sign check: increasing bits should *reduce* per-element distortion,
    so the bit gradient under a positive upstream gradient must point in
    the direction that increases ``q`` toward the unquantized ``w``.

    Concretely: with ``∂q/∂bits ≈ -ln2 · w``, a positive upstream
    grad_output on a *positive* weight produces a *negative* grad_bits
    (gradient descent then INCREASES bits, which is the desired
    direction — bits has a softplus(raw) parameterisation so SGD on
    -grad pushes raw up → bits up → finer step → q non-zero → smaller
    quantization error).
    """
    weight, scale, bits = _make_weight_with_subbin_elements(
        bits=2.0, sub_bin_frac=0.1,
    )
    weight.requires_grad_(True)
    bits_param = bits.detach().clone().requires_grad_(True)

    q = _PerElementSTEQuantize.apply(weight, scale, bits_param)
    # Upstream gradient = +1 (so the loss surface "wants q to decrease"
    # — call this "negative distortion direction" for elements where
    # raising bits would let q approach the true positive weight).
    grad_out = torch.ones_like(q)
    q.backward(grad_out)

    # Per the fix: grad_bits = grad_out · (-ln2 · weight)
    # = +1 · (-ln2 · ±0.1). On positive weights → negative grad_bits;
    # on negative weights → positive grad_bits. The product
    # (grad_bits · sign(weight)) must therefore be UNIFORMLY NEGATIVE
    # (the gradient descends in the direction that *raises* bits when
    # |weight| is non-zero, i.e. allocates rate to the load-bearing
    # weights).
    sign_w = torch.sign(weight.detach())
    gb = bits_param.grad
    product = gb * sign_w
    assert torch.all(product < 0), (
        f"bit-gradient sign convention violated: grad·sign(w) = "
        f"{product.tolist()} — should be all negative so SGD on bits "
        f"raises bit-depth on load-bearing weights."
    )


def test_one_bit_branch_keeps_grad_signal_on_quantized_magnitude():
    """1-bit elements use the sign quantizer (out = sign(w) · scale) — the
    grid surrogate doesn't apply directly. The fix substitutes
    ``-ln2 · q`` (the post-quantization magnitude) for that branch, so
    the bit gradient is still non-zero (driven by ``q = ±scale``)."""
    weight, scale, bits = _make_weight_with_subbin_elements(
        bits=1.0, sub_bin_frac=0.1,
    )
    weight.requires_grad_(True)
    bits_param = bits.detach().clone().requires_grad_(True)

    q = _PerElementSTEQuantize.apply(weight, scale, bits_param)
    # 1-bit: q = sign(w) · scale → ±1.0 everywhere.
    assert torch.allclose(q.abs(), torch.ones_like(q)), (
        f"test setup failure: 1-bit q expected magnitude 1, got {q.abs().tolist()}"
    )
    grad_out = torch.ones_like(q)
    q.backward(grad_out)

    # Post-fix 1-bit grad: grad_bits = +1 · (-ln2 · q) = -ln2 · sign(w)
    # → magnitude = ln2 ≈ 0.693, non-zero everywhere.
    assert bits_param.grad is not None
    assert torch.all(bits_param.grad.abs() > 0.5), (
        f"1-bit branch grad_bits magnitudes too small: "
        f"{bits_param.grad.abs().tolist()}"
    )


def test_zero_bin_signal_persists_in_higher_bit_setting():
    """Verify the fix is not specific to 2-bit: at 4-bit a sub-bin weight
    (magnitude well below step/2) also receives bit-gradient signal."""
    # At 4-bit: levels = 2^(4-1) - 1 = 7, step = 1/7 ≈ 0.143.
    # A weight at 0.05 < step/2 ≈ 0.071 still rounds to 0.
    weight, scale, bits = _make_weight_with_subbin_elements(
        bits=4.0, sub_bin_frac=0.05,
    )
    weight.requires_grad_(True)
    bits_param = bits.detach().clone().requires_grad_(True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits_param)
    assert torch.allclose(q, torch.zeros_like(q))
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert torch.all(bits_param.grad.abs() > 0), (
        "4-bit zero-bin weights still receive zero gradient — Bug 1 fix "
        "does not generalise across bit-depths."
    )


def test_above_bin_elements_keep_gradient_signal():
    """Sanity / non-regression: weights that *don't* round to zero (large
    enough to land on a non-zero quantization level) must also have
    non-zero bit gradient. The fix should only ADD signal in the zero
    bin, not remove it elsewhere."""
    # Set up: at 2-bit, step = 1.0 (with scale = 1.0), so a weight of
    # 0.6 rounds to round(0.6 / 1.0) = 1 → q = 1.0.
    n_chan = 4
    scale = torch.ones(n_chan, dtype=torch.float64)
    weight = torch.tensor(
        [0.6, -0.6, 0.6, -0.6], dtype=torch.float64
    ).reshape(n_chan, 1, 1, 1).clone()
    bits = torch.full_like(weight, 2.0, dtype=torch.float64)
    weight.requires_grad_(True)
    bits_param = bits.detach().clone().requires_grad_(True)

    q = _PerElementSTEQuantize.apply(weight, scale, bits_param)
    # Sanity: q should be ±1 (not ±0).
    assert torch.allclose(q.abs(), torch.ones_like(q)), (
        f"setup: expected |q|=1, got {q.abs().tolist()}"
    )
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert torch.all(bits_param.grad.abs() > 0), (
        "non-zero-bin weights lost their bit-gradient signal — fix "
        "broke the high-bit path."
    )
