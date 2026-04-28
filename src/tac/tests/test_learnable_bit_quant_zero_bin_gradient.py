"""Bug 1 (codex Round 4) regression: bit-depth STE must produce a
gradient that points in the *distortion-reducing* direction for every
weight whose quantization residual ``q − w`` is non-zero, including the
zero-bin case.

Round 3 attempted to fix the original "``-ln2 · q`` is exactly zero
when q=0" bug by substituting the raw weight ``-ln2 · w``. That had
the wrong sign for the dominant case: under the canonical
reconstruction loss ``L = ½ (q − w)²`` the upstream gradient is
``∂L/∂q = q − w`` (negative when q under-shoots a positive weight in
the zero bin), and chaining with ``-ln2 · w`` produces a *positive*
``grad_bits`` — SGD then DECREASES bits, the opposite of what
minimizes distortion.

Round 4 fix: use the HAWQ first-order surrogate
    ∂q/∂bits ≈ -ln2 · (q − w)
which guarantees three load-bearing properties under MSE:

  1. Magnitude tracks the *quantization error*, not the raw weight.
  2. Sign chains with ``∂L/∂q = (q − w)`` to give
     ``∂L/∂bits = -ln2 · (q − w)² ≤ 0`` ALWAYS — SGD raises bits
     whenever any distortion exists. Direction is unconditionally
     correct.
  3. Zero distortion → zero gradient (no spurious signal under
     perfect reconstruction).

The tests below pin all three properties using REAL MSE loss against
the unquantized weight, not synthetic ``grad_output = +1`` (which the
Round 3 fix happened to satisfy by coincidence on positive weights).
"""
from __future__ import annotations

import math

import torch

from tac.learnable_bit_quant import _PerElementSTEQuantize


LN2 = math.log(2.0)


def _quantize_with_grad(
    weight_val: float,
    *,
    bits: float,
    scale_val: float = 1.0,
    n_chan: int = 4,
    sign_pattern: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (weight, scale, bits) tensors with ``weight = sign · weight_val``,
    push them through ``_PerElementSTEQuantize.apply``, and return
    ``(weight, bits_param, q)`` for downstream gradient checks.

    ``sign_pattern`` defaults to ``(+, -, +, -)`` so each test exercises
    both signs simultaneously.
    """
    if sign_pattern is None:
        sign_pattern = (+1.0, -1.0, +1.0, -1.0)
    if len(sign_pattern) != n_chan:
        raise ValueError(
            f"sign_pattern length {len(sign_pattern)} != n_chan {n_chan}"
        )
    scale = torch.full((n_chan,), float(scale_val), dtype=torch.float64)
    sign = torch.tensor(list(sign_pattern), dtype=torch.float64)
    weight = (sign * float(weight_val)).reshape(n_chan, 1, 1, 1).clone()
    weight.requires_grad_(True)
    bits_t = torch.full_like(weight, float(bits), dtype=torch.float64)
    bits_param = bits_t.detach().clone().requires_grad_(True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits_param)
    return weight, bits_param, q


# ── Bug 1 sanity: w=0.1, scale=1, b=2 (forces q=0) ──────────────────────


def test_zero_bin_grad_bits_is_negative_under_mse_loss():
    """w=0.1, scale=1, b=2 → q=0. Under MSE the upstream gradient is
    ``∂L/∂q = q − w = -0.1``; the Round 4 surrogate gives
    ``grad_bits = (q − w) · (-ln2) · (q − w) = -ln2 · 0.01 < 0``
    so SGD raises bits. The Round 3 surrogate (``-ln2 · weight``)
    produces ``+0.00693`` for w=0.1 (positive → SGD lowers bits, WRONG).
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.1, bits=2.0, scale_val=1.0,
    )
    # At b=2, levels = 2^(2-1) - 1 = 1, step = scale / 1 = 1.0,
    # round(0.1 / 1.0) = 0 → q = 0 everywhere (load-bearing test setup).
    assert torch.allclose(q, torch.zeros_like(q)), (
        f"setup failure: q expected all zeros, got {q.tolist()}"
    )

    # REAL MSE distortion: L = 0.5 · sum((q - w)²).
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()

    assert bits_param.grad is not None
    # grad_bits MUST be negative everywhere — SGD then raises bits which
    # is the only way to reduce the per-element distortion.
    assert torch.all(bits_param.grad < 0), (
        f"zero-bin grad_bits must be negative under MSE; got "
        f"{bits_param.grad.tolist()}"
    )
    # Magnitude check: per-element grad_bits = -ln2 · (q − w)² = -ln2 · 0.01.
    expected = -LN2 * (0.1 ** 2)
    assert torch.allclose(
        bits_param.grad,
        torch.full_like(bits_param.grad, expected),
        atol=1e-12,
    ), (
        f"zero-bin grad_bits magnitude wrong: got {bits_param.grad.tolist()}, "
        f"expected ≈ {expected}"
    )


def test_nonzero_bin_grad_bits_is_negative_under_mse_loss():
    """w=0.5, scale=1, b=2 (q nonzero). The Round 4 surrogate must STILL
    produce negative grad_bits under MSE — distortion can only decrease
    when bits increase, regardless of which bin we land in.

    At b=2, step=1.0, round(0.5 / 1.0) = 0 (since 0.5 rounds to 0 with
    banker's-or-half-to-even convention in PyTorch); use a value clearly
    above the half-step boundary so the test is unambiguous.
    """
    # Pick a value that rounds to a non-zero level even under either
    # rounding convention. At step=1.0, w=0.7 → round(0.7)=1 → q=1.0.
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.7, bits=2.0, scale_val=1.0,
    )
    # |q| should be 1 here (nonzero bin).
    assert torch.allclose(q.abs(), torch.ones_like(q)), (
        f"setup failure: |q|=1 expected, got {q.abs().tolist()}"
    )

    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()

    assert bits_param.grad is not None
    # All grad_bits must be ≤ 0 (and < 0 since residual ≠ 0).
    assert torch.all(bits_param.grad < 0), (
        f"nonzero-bin grad_bits must be negative under MSE; got "
        f"{bits_param.grad.tolist()}"
    )
    # Magnitude check: per-element grad_bits = -ln2 · (q − w)² for each
    # element. Residual on positive weight: q=+1, w=+0.7 → r=+0.3.
    # Residual on negative weight: q=-1, w=-0.7 → r=-0.3. Square is the
    # same → all elements have the same |grad|.
    expected = -LN2 * (0.3 ** 2)
    assert torch.allclose(
        bits_param.grad,
        torch.full_like(bits_param.grad, expected),
        atol=1e-12,
    ), (
        f"nonzero-bin grad_bits magnitude wrong: got {bits_param.grad.tolist()}, "
        f"expected ≈ {expected}"
    )


def test_zero_distortion_grad_bits_is_zero():
    """w=0 (no distortion possible) → grad_bits must be exactly 0.
    The Round 4 surrogate ``-ln2 · (q − w)`` is exactly 0 when q=w=0.
    This pins that the gradient correctly says "no signal" when the
    weight is already perfectly representable.
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.0, bits=2.0, scale_val=1.0,
        sign_pattern=(0.0, 0.0, 0.0, 0.0),
    )
    # Zero weights → zero quantized output → zero distortion.
    assert torch.allclose(q, torch.zeros_like(q))
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()
    assert bits_param.grad is not None
    assert torch.all(bits_param.grad == 0), (
        f"zero-distortion grad_bits must be exactly 0; got "
        f"{bits_param.grad.tolist()}"
    )


# ── Generality across bit depths and bin types ──────────────────────────


def test_zero_bin_signal_persists_at_higher_bits():
    """At b=4, step ≈ 0.143; w=0.05 still rounds to 0 → q=0. Under MSE
    the Round 4 surrogate must give a strictly negative bit gradient."""
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.05, bits=4.0, scale_val=1.0,
    )
    assert torch.allclose(q, torch.zeros_like(q))
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()
    assert bits_param.grad is not None
    assert torch.all(bits_param.grad < 0), (
        "4-bit zero-bin grad_bits not negative under MSE — bug 1 fix "
        "does not generalise across bit-depths."
    )


def test_one_bit_branch_grad_bits_is_negative_under_mse_loss():
    """1-bit elements use the sign quantizer (out = sign(w) · scale).
    The Round 4 residual surrogate uses ``(q − w) = sign(w)·scale − w``,
    which is non-zero whenever ``|w| < scale``. Under MSE the chained
    grad must be negative on every element.
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.1, bits=1.0, scale_val=1.0,
    )
    # 1-bit: |q| = scale = 1.
    assert torch.allclose(q.abs(), torch.ones_like(q)), (
        f"setup failure: 1-bit |q|=1 expected, got {q.abs().tolist()}"
    )
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()
    assert bits_param.grad is not None
    # Residual on positive w: q=+1, w=+0.1 → r=+0.9 → grad = -ln2 · 0.81.
    # Residual on negative w: q=-1, w=-0.1 → r=-0.9 → grad = -ln2 · 0.81.
    # Both negative.
    assert torch.all(bits_param.grad < 0), (
        f"1-bit grad_bits must be negative under MSE; got "
        f"{bits_param.grad.tolist()}"
    )
    expected = -LN2 * (0.9 ** 2)
    assert torch.allclose(
        bits_param.grad,
        torch.full_like(bits_param.grad, expected),
        atol=1e-12,
    )


# ── Direct surrogate sanity check (no upstream loss) ────────────────────


def test_grad_bits_surrogate_has_correct_sign_under_synthetic_upstream():
    """Direct sign check: with synthetic ``grad_output = +1`` and
    ``residual = q − w``, the Round 4 surrogate gives
    ``grad_bits = +1 · (-ln2 · residual) = -ln2 · residual``.

    For w=+0.1, q=0 → residual = -0.1 → grad_bits = +ln2 · 0.1 (positive).
    For w=-0.1, q=0 → residual = +0.1 → grad_bits = -ln2 · 0.1 (negative).

    Crucially, ``grad_bits · sign(residual)`` is uniformly NEGATIVE — i.e.
    SGD on bits is descended along the residual axis, which is the
    correct direction whenever the *upstream* gradient itself is
    distortion-aligned (the canonical case under MSE).
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.1, bits=2.0, scale_val=1.0,
    )
    assert torch.allclose(q, torch.zeros_like(q))
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    residual = (q.detach() - weight.detach())
    sign_r = torch.sign(residual)
    product = bits_param.grad * sign_r
    assert torch.all(product < 0), (
        f"surrogate sign convention violated: grad·sign(residual) = "
        f"{product.tolist()} — should be uniformly negative."
    )


def test_above_bin_elements_keep_gradient_signal():
    """Sanity / non-regression: weights that don't round to zero (large
    enough to land on a non-zero quantization level) must also have
    non-zero bit gradient. The Round 4 surrogate is only zero when
    residual is zero — non-grid weights always have residual ≠ 0.
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.6, bits=2.0, scale_val=1.0,
    )
    # |q| should be 1 (nonzero bin).
    assert torch.allclose(q.abs(), torch.ones_like(q)), (
        f"setup: expected |q|=1, got {q.abs().tolist()}"
    )
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()
    assert bits_param.grad is not None
    assert torch.all(bits_param.grad.abs() > 0), (
        "non-zero-bin weights lost their bit-gradient signal — Round 4 "
        "fix broke the high-bit path."
    )
    # And under MSE the sign must still be negative.
    assert torch.all(bits_param.grad < 0), (
        f"non-zero-bin grad_bits must be negative under MSE; got "
        f"{bits_param.grad.tolist()}"
    )
