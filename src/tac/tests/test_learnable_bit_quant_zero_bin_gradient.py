# SPDX-License-Identifier: MIT
"""Bit-depth STE surrogate regression tests — Rounds 4, 6 & 7.

Round 4 (HAWQ residual surrogate): the bit-depth gradient must produce
a *direction* that escapes the zero bin (increase bits when ``q=0``
under-shoots a non-zero weight).

Round 6 (q-delta finite-difference STE): the residual surrogate is
provably WRONG when the integer grids at adjacent bit-depths are NOT
NESTED. ``levels(b) = 2^(b-1) - 1 = {1, 3, 7, 15, …}`` is a sequence
of co-prime denominators, so the b=4 grid is NOT a refinement of the
b=3 grid. The Round 6 fix replaced the residual approximation with::

    ∂q/∂bits ≈ (q(b+1) − q(b−1)) / 2
    ∂L/∂bits = grad_output · (qp − qm) / 2

Round 7 (LOSS-DELTA finite-difference STE): the Round 6 q-delta FD
passes synthetic-upstream tests but INVERTS the sign update under the
realistic MSE chain. Worked example::

    w = 0.1, scale = 1, b = 2 → q = 0 (zero bin)
    q_bplus(b=3)  = 0          (step=1/3 > 0.1, still rounds to 0)
    q_bminus(b=1) = +1         (sign quantizer)
    grad_output (MSE) = q − w = -0.1
    Round 6 grad_bits = -0.1 · (0 − 1)/2 = +0.05   → SGD LOWERS bits
    bits=1 → q=+1 → MSE = 0.81 (vs MSE=0.01 at b=2 — STRICTLY WORSE)

Round 7 replaces the q-delta FD with a LOSS-DELTA FD chained with the
upstream gradient as a magnitude scaler::

    L(q) = ½(q − w)²                     (canonical local proxy)
    local_diff = [L(q_bplus) − L(q_bminus)] / 2
    grad_bits  = |grad_output| · local_diff

The sign of the bit update now comes from the local loss landscape, not
from a chain product that can invert under MSE. Verification of the
fix at the bug case::

    sq_err_bplus  = (0  − 0.1)² = 0.01
    sq_err_bminus = (1  − 0.1)² = 0.81
    local_diff    = (0.01 − 0.81)/2 = -0.40
    grad_bits     = |-0.1| · -0.40 = -0.04    → SGD RAISES bits ✓

Round 7 also tightens the zero-distortion short-circuit from
``weight == 0`` (which incorrectly suppressed the b=1 sign-quantizer
escape at w=0) to ``q == weight`` (element-wise zero distortion at the
current bit-depth).

Properties tested below:
  * Zero-bin escape (Rounds 4 & 7): under SYNTHETIC ``+1`` upstream,
    ``grad_bits`` is NEGATIVE at the zero bin for both signs of the
    weight (the local loss landscape always favors raising bits when
    a finer grid would reduce distortion).
  * Sweet-spot detection (Round 6/7): the FD surrogate captures the
    actual local-loss landscape; under the MSE chain at a true sweet
    spot the magnitude is dominated by ``|grad_output|`` which is
    small, so SGD barely moves regardless of FD sign.
  * Zero distortion → zero gradient (Round 7 fix uses ``q == weight``,
    not ``weight == 0``).
  * Sign correctness vs the actual loss-delta direction across a sweep.
  * NEW Round 7 math-safety checks under REAL MSE CHAIN at the four
    canonical operating points.
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


def _local_mse(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """½ (q − w)² — the canonical local distortion proxy used by HAWQ."""
    return 0.5 * (q - w) ** 2


def _quantize_at_bits(
    weight: torch.Tensor, scale: torch.Tensor, bits_val: float
) -> torch.Tensor:
    """Run the same quantizer formula at a fixed bit-depth (no autograd).

    Used by the loss-delta tests to compute ``L(b+1)`` and ``L(b−1)``
    at known bit-depths without re-routing through autograd.
    """
    bits_t = torch.full_like(weight, float(bits_val), dtype=weight.dtype)
    with torch.no_grad():
        out = _PerElementSTEQuantize.apply(
            weight.detach(), scale, bits_t
        )
    return out


# ── Round 4: zero-bin escape (synthetic upstream) ───────────────────────


def test_zero_bin_grad_bits_points_toward_more_bits_synthetic():
    """w=±0.1, scale=1, b=2 → q=0 (zero bin).

    Round 7 LOSS-DELTA FD under synthetic ``+1`` upstream::

        q_bplus(b=3)  = 0  (step=1/3 > 0.1, still rounds to 0)
        q_bminus(b=1) = sign(w)·1  (sign quantizer)
        sq_err_bplus  = (0 − w)² = 0.01           (BOTH signs)
        sq_err_bminus = (sign(w) − w)² = 0.81     (BOTH signs)
        local_diff    = (0.01 − 0.81)/2 = -0.40
        grad_bits     = |+1| · -0.40 = -0.40      (NEGATIVE for BOTH signs)

    Unlike the Round 6 q-delta FD (which gave grad_bits = -sign(w)·0.5,
    flipping with weight sign), the Round 7 loss-delta FD produces a
    SIGN-INVARIANT bit gradient: the local loss is symmetric in w, so
    the FD direction depends only on which adjacent grid is *closer to
    w*. At the zero bin the FINER grid is always closer (the b+1 grid
    can represent values nearer to 0), so the FD points to "raise bits"
    regardless of weight sign.
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.1, bits=2.0, scale_val=1.0,
    )
    assert torch.allclose(q, torch.zeros_like(q)), (
        f"setup: q expected all zeros, got {q.tolist()}"
    )
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    # Round 7: grad_bits is NEGATIVE for both signs (sign-invariant loss).
    assert torch.all(bits_param.grad < 0), (
        f"zero-bin grad_bits direction wrong: got {bits_param.grad.tolist()}, "
        f"should all be NEGATIVE (escape zero bin via finer grid)."
    )


def test_nonzero_bin_grad_bits_under_synthetic_upstream():
    """w=±0.7, scale=1, b=2 → q=±1 (saturated).

    Round 7 LOSS-DELTA FD under synthetic ``+1`` upstream::

        q_bplus(b=3)  for w=±0.7: ±0.667    (closer to w)
        q_bminus(b=1) for w=±0.7: ±1.0      (sign quantizer = current q)
        sq_err_bplus  = (0.667 − 0.7)² ≈ 0.00109
        sq_err_bminus = (1.0   − 0.7)² = 0.09
        local_diff    ≈ (0.00109 − 0.09)/2 ≈ -0.0445
        grad_bits     = |+1| · -0.0445 = -0.0445  (NEGATIVE for BOTH signs)

    The b=3 grid is closer to |w|=0.7 than the b=1 sign quantizer, so
    the FD correctly points to raising bits. Sign-invariant (symmetric
    loss in w).
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.7, bits=2.0, scale_val=1.0,
    )
    assert torch.allclose(q.abs(), torch.ones_like(q)), (
        f"setup: |q|=1 expected, got {q.abs().tolist()}"
    )
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    # Round 7: grad_bits is NEGATIVE for both signs (raise bits → finer grid).
    assert torch.all(bits_param.grad < 0), (
        f"nonzero-bin grad_bits direction wrong: got {bits_param.grad.tolist()}, "
        f"should all be NEGATIVE (raise bits → finer grid closer to w)."
    )


def test_zero_distortion_grad_bits_is_zero():
    """w=0 → q=0 at every bit-depth → q_bplus=q_bminus=0 → grad_bits=0.
    The FD surrogate correctly says "no signal" when the weight is
    perfectly representable.
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.0, bits=2.0, scale_val=1.0,
        sign_pattern=(0.0, 0.0, 0.0, 0.0),
    )
    assert torch.allclose(q, torch.zeros_like(q))
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    assert torch.all(bits_param.grad == 0), (
        f"zero-distortion grad_bits must be exactly 0; got "
        f"{bits_param.grad.tolist()}"
    )


def test_zero_bin_signal_persists_at_higher_bits():
    """At b=4 with w=0.05 the value still rounds to 0 under the b=4 grid
    (step=1/7≈0.143; round(0.05/0.143)=round(0.35)=0). The b=5 grid has
    step=1/15; round(0.05/0.0667)=round(0.75)=1; q_bplus=1/15≈0.0667.
    The b=3 grid has step=1/3; round(0.05/0.333)=round(0.15)=0; q_bminus=0.
    Central FD: (1/15 − 0)/2 = 1/30 ≈ 0.033 (non-zero signal).

    The Round 4 surrogate had this signal hard-coded toward "raise bits"
    via the residual approximation; the Round 6 FD surrogate produces a
    real direction reflecting the actual q(b) staircase. Here the FD
    direction is positive (q_bplus > q_bminus), meaning ``q`` rises with
    bits — under MSE chain (grad_output = q-w = -0.05) the chained
    grad_bits = -0.05 × 1/30 = -0.00167 (negative → SGD raises bits, the
    correct escape direction). This test pins that the FD signal SURVIVES
    at higher bit-depths (the Round 4 fix didn't have a "high-bit signal
    decay" failure mode and Round 6 mustn't introduce one).
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.05, bits=4.0, scale_val=1.0,
    )
    assert torch.allclose(q, torch.zeros_like(q))
    # Test 1: synthetic upstream — non-zero signal.
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    assert torch.all(bits_param.grad.abs() > 0), (
        f"4-bit zero-bin lost signal under FD surrogate; got "
        f"{bits_param.grad.tolist()}"
    )

    # Test 2: under MSE chain, grad_bits must be NEGATIVE everywhere
    # (raises bits, escape direction). For both signs of w, the FD
    # direction (qp − qm) flips sign in lockstep with w, so the product
    # ``(q − w) · (qp − qm)`` is uniformly negative.
    #   w=+0.05: q=0, q_bplus=+1/15, q_bminus=0 → FD=+1/30. grad_output=-0.05.
    #            grad_bits = -0.05 · 1/30 = -1.67e-3 (raises bits) ✓
    #   w=-0.05: q=0, q_bplus=-1/15, q_bminus=0 → FD=-1/30. grad_output=+0.05.
    #            grad_bits = +0.05 · -1/30 = -1.67e-3 (raises bits) ✓
    weight2, bits_param2, q2 = _quantize_with_grad(
        weight_val=0.05, bits=4.0, scale_val=1.0,
    )
    loss = 0.5 * ((q2 - weight2) ** 2).sum()
    loss.backward()
    assert bits_param2.grad is not None
    assert torch.all(bits_param2.grad < 0), (
        f"4-bit zero-bin under MSE: grad_bits = {bits_param2.grad.tolist()}, "
        f"should be negative everywhere (escape direction = raise bits)."
    )


def test_one_bit_branch_grad_bits_under_synthetic():
    """1-bit elements use the sign quantizer (out = sign(w)·scale).

    Round 7 LOSS-DELTA FD: for w=±0.1, b=1::

        q             = ±1 (sign quantizer)
        q_bplus(b=2)  = round(±0.1 / 1) · 1 = 0
        q_bminus(b=1) = ±1     (clamped to b=1)
        sq_err_bplus  = (0 − 0.1)² = 0.01
        sq_err_bminus = (1 − 0.1)² = 0.81
        local_diff    = (0.01 − 0.81)/2 = -0.40
        grad_bits     = |+1| · -0.40 = -0.40   (NEGATIVE for BOTH signs)

    SGD raises bits to escape the sign quantizer, sign-invariant.
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.1, bits=1.0, scale_val=1.0,
    )
    assert torch.allclose(q.abs(), torch.ones_like(q)), (
        f"setup: 1-bit |q|=1 expected, got {q.abs().tolist()}"
    )
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    # Round 7: grad_bits is NEGATIVE for both signs (escape sign quantizer).
    assert torch.all(bits_param.grad < 0), (
        f"1-bit grad_bits direction wrong: got {bits_param.grad.tolist()}, "
        f"should all be NEGATIVE (raise bits to escape sign quantizer)."
    )


# ── Round 6: sweet-spot detection (the new fix) ─────────────────────────


def test_grad_bits_handles_b3_b4_sweet_spot():
    """w = 1/3 + 0.001, scale = 1, b = 3 → q = ±1/3 (sweet spot,
    MSE ≈ 5e-7). Going to b=4 produces q = 2/7 ≈ 0.2857 (MSE ≈ 2.4e-3,
    WORSE). Going to b=2 produces q = 0 (MSE ≈ 0.112, MUCH WORSE).

    Round 7 LOSS-DELTA FD under SYNTHETIC ``+1`` upstream::

        sq_err_bplus  = (2/7 − (1/3+0.001))² ≈ 2.36e-3
        sq_err_bminus = (0   − (1/3+0.001))² ≈ 0.112
        local_diff    = (2.36e-3 − 0.112)/2 ≈ -0.0547
        grad_bits     = |+1| · -0.0547 = -0.0547  (NEGATIVE)

    Under SYNTHETIC upstream the FD direction is "raise bits" because
    going UP is much less bad than going DOWN (b=2 is far worse than
    b=4). However sweet-spot PROTECTION in Round 7 happens through the
    ``|grad_output|`` MAGNITUDE scaler in the realistic MSE chain — at
    the sweet spot ``q − w = -0.001``, so the MSE-chained
    ``grad_bits = 0.001 · -0.0547 ≈ -5.5e-5`` is tiny and SGD barely
    moves. See ``test_sweet_spot_w_one_third_b3_no_push_to_b4`` below
    for the MSE-chain magnitude check.

    This test pins the SYNTHETIC-upstream chain-rule formula rather
    than the previous ``grad·sign(w) > 0`` invariant (which assumed
    Round 6 q-delta semantics).
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=1.0 / 3.0 + 0.001, bits=3.0, scale_val=1.0,
    )
    expected_q = torch.full_like(q, 1.0 / 3.0)
    expected_q = expected_q * torch.sign(weight.detach())
    assert torch.allclose(q, expected_q, atol=1e-12), (
        f"sweet-spot setup: q expected ±1/3, got {q.flatten().tolist()}"
    )
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    # Round 7: grad_bits is NEGATIVE for both signs (loss landscape
    # symmetric in w, b=2 is much worse than b=4).
    assert torch.all(bits_param.grad < 0), (
        f"sweet-spot synthetic-upstream grad direction wrong: "
        f"got {bits_param.grad.tolist()}, should be NEGATIVE under "
        f"loss-delta FD (b=2 much worse than b=4)."
    )
    # Magnitude: |grad_bits| ≈ |sq_err_bplus − sq_err_bminus|/2.
    # sq_err_bplus  = (2/7 − w)², sq_err_bminus = (0 − w)² = w².
    w_val = 1.0 / 3.0 + 0.001
    sq_err_bplus = (2.0 / 7.0 - w_val) ** 2
    sq_err_bminus = w_val ** 2
    expected_mag = abs((sq_err_bplus - sq_err_bminus) / 2.0)
    assert torch.allclose(
        bits_param.grad.abs(),
        torch.full_like(bits_param.grad, expected_mag),
        atol=1e-12,
    ), (
        f"sweet-spot grad magnitude wrong: got {bits_param.grad.abs().tolist()}, "
        f"expected ≈ {expected_mag}"
    )


def test_grad_bits_handles_b4_b5_transition():
    """At b=4 (levels=7, step=1/7), a weight just above 1/7 has
    ``q = 1/7`` — a near-sweet-spot. b=5 (levels=15, step=1/15) gives
    ``q = 2/15 ≈ 0.133`` (CLOSER to w=1/7+ε ≈ 0.144). b=3 (levels=3,
    step=1/3) gives ``q = 0`` (FURTHER from w).

    Round 7 LOSS-DELTA FD::

        sq_err_bplus  = (2/15 − w)² ≈ 0.00011
        sq_err_bminus = (0    − w)² ≈ 0.0207
        local_diff    = (0.00011 − 0.0207)/2 ≈ -0.01029
        grad_bits     = |+1| · -0.01029 (NEGATIVE; raises bits)

    The FD direction agrees with the actual loss landscape: raising bits
    from b=4 to b=5 IS beneficial here (L5 < L4 < L3). The test pins
    that the FD surrogate matches the loss-delta formula and gives
    finite, non-zero output.
    """
    w_val = 1.0 / 7.0 + 0.001  # just above the b=4 grid point 1/7
    weight, bits_param, q = _quantize_with_grad(
        weight_val=w_val, bits=4.0, scale_val=1.0,
    )
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    # Compute actual loss at b=3, b=4, b=5 to verify FD direction.
    scale = torch.ones(weight.shape[0], dtype=weight.dtype)
    q_b3 = _quantize_at_bits(weight, scale, 3.0)
    q_b4 = _quantize_at_bits(weight, scale, 4.0)
    q_b5 = _quantize_at_bits(weight, scale, 5.0)
    L3 = float(_local_mse(q_b3, weight.detach()).sum().item())
    L4 = float(_local_mse(q_b4, weight.detach()).sum().item())
    L5 = float(_local_mse(q_b5, weight.detach()).sum().item())
    assert L4 < L3, (
        f"b=4 should beat b=3 here (L4={L4}, L3={L3}); test setup wrong"
    )
    # Round 7 expected: grad_bits = |grad_output| · (sq_err_bplus − sq_err_bminus)/2
    sq_err_bplus = (q_b5 - weight.detach()) ** 2
    sq_err_bminus = (q_b3 - weight.detach()) ** 2
    expected_grad = grad_out.abs() * (sq_err_bplus - sq_err_bminus) / 2.0
    assert torch.allclose(bits_param.grad, expected_grad, atol=1e-12), (
        f"b=4 transition loss-FD wrong: got {bits_param.grad.tolist()}, "
        f"expected {expected_grad.tolist()}"
    )
    # And the FD direction agrees with raising bits (b=5 is closer than b=3).
    assert torch.all(bits_param.grad < 0), (
        f"b=4→5 transition should yield negative grad (raise bits); "
        f"got {bits_param.grad.tolist()}"
    )


def test_grad_bits_finite_difference_matches_actual_loss_delta():
    """Round 7 contract: across a sweep of weight values, the LOSS-DELTA
    FD surrogate must MATCH the formula::

        grad_bits = |grad_output| · (sq_err_bplus − sq_err_bminus) / 2

    where ``sq_err_b* = (q_b* − w)²``. This is the new load-bearing
    formula that fixes the Round 6 sign-inversion under MSE chain.
    """
    # Sweep across a range of weight values that exercise grid sweet
    # spots, off-grid points, and zero-bin cases.
    test_values = [
        0.05, 0.1, 0.15, 0.2, 0.25, 1.0 / 3.0 + 0.001, 0.4, 0.5, 0.6,
        0.7, 1.0 / 7.0 + 0.001,
    ]
    for b in (2.0, 3.0, 4.0, 5.0):
        for w_val in test_values:
            weight, bits_param, q = _quantize_with_grad(
                weight_val=w_val, bits=b, scale_val=1.0,
            )
            scale = torch.ones(weight.shape[0], dtype=weight.dtype)
            # Compute local L at b±1 (clamped to [1,8]).
            b_plus = min(b + 1.0, 8.0)
            b_minus = max(b - 1.0, 1.0)
            q_bp = _quantize_at_bits(weight, scale, b_plus)
            q_bm = _quantize_at_bits(weight, scale, b_minus)
            # FD surrogate with synthetic +1 upstream.
            grad_out = torch.ones_like(q)
            q.backward(grad_out)
            assert bits_param.grad is not None
            # Round 7 expected formula.
            sq_err_bp = (q_bp - weight.detach()) ** 2
            sq_err_bm = (q_bm - weight.detach()) ** 2
            expected = grad_out.abs() * (sq_err_bp - sq_err_bm) / 2.0
            # Apply the zero-distortion mask (q == weight → grad_bits=0).
            zero_mask = (q.detach() == weight.detach())
            expected = torch.where(
                zero_mask, torch.zeros_like(expected), expected
            )
            assert torch.allclose(bits_param.grad, expected, atol=1e-12), (
                f"Round 7 loss-FD chain-rule mismatch at w={w_val}, b={b}: "
                f"got {bits_param.grad.tolist()}, expected {expected.tolist()}"
            )
            # And the surrogate must be FINITE (no NaN/Inf).
            assert torch.isfinite(bits_param.grad).all(), (
                f"FD grad non-finite at w={w_val}, b={b}: {bits_param.grad}"
            )
            del weight, bits_param, q


def test_above_bin_elements_keep_gradient_signal():
    """Sanity / non-regression: weights large enough to land on a
    non-zero quantization level have non-zero bit gradient under the
    FD surrogate. The gradient is zero only when ``q_bplus == q_bminus``
    which requires a flat region of the q(b) staircase — rare for
    off-grid weights.
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.6, bits=2.0, scale_val=1.0,
    )
    assert torch.allclose(q.abs(), torch.ones_like(q)), (
        f"setup: expected |q|=1, got {q.abs().tolist()}"
    )
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    assert torch.all(bits_param.grad.abs() > 0), (
        "non-zero-bin weights lost their bit-gradient signal — Round 6 "
        "FD surrogate broke the high-bit path."
    )


# ── MSE-chain regression: behavior changed in Round 6 ───────────────────
#
# Under MSE upstream the chain product ``(q − w) · (q_bplus − q_bminus)/2``
# has more nuanced sign than the Round 4 residual surrogate (which was
# uniformly negative). The cases below pin the new behavior so future
# regressions can be detected. These are NOT guarantees of "raises bits"
# under MSE — that property held only by accident in Round 4 because the
# residual surrogate hard-coded the direction; the FD surrogate is more
# honest about the actual non-monotonic landscape.


def test_zero_bin_under_mse_chain_round7_behavior():
    """Round 7 LOSS-DELTA FD restores the Round 4 zero-bin escape
    invariant under MSE chain (which Round 6 q-delta FD broke).

    For w=+0.1, b=2: q=0, q_bplus(b=3)=0, q_bminus(b=1)=+1.
    grad_output = q − w = -0.1, |grad_output| = 0.1.
    sq_err_bplus  = (0 − 0.1)² = 0.01
    sq_err_bminus = (1 − 0.1)² = 0.81
    local_diff    = (0.01 − 0.81)/2 = -0.40
    grad_bits     = 0.1 · -0.40 = -0.04   (NEGATIVE → SGD raises bits)

    Round 6 gave +0.05 at this point (SGD lowers bits → MSE goes from
    0.01 to 0.81). Round 7 restores the correct direction.
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.1, bits=2.0, scale_val=1.0,
        sign_pattern=(+1.0, +1.0, +1.0, +1.0),
    )
    assert torch.allclose(q, torch.zeros_like(q))
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()
    assert bits_param.grad is not None
    # Round 7 expected: -0.04 (sign-invariant under MSE because |g| folds).
    expected = torch.full_like(bits_param.grad, -0.04)
    assert torch.allclose(bits_param.grad, expected, atol=1e-12), (
        f"MSE-chain zero-bin grad (Round 7): got {bits_param.grad.tolist()}, "
        f"expected ≈ {expected.tolist()}"
    )


def test_grad_bits_surrogate_has_correct_sign_under_synthetic_upstream():
    """Round 7: the LOSS-DELTA FD is SIGN-INVARIANT in w because
    ``L(q) = ½(q − w)²`` is symmetric under joint sign-flip of (w, q).

    For w=±0.1, b=2: q=0, q_bplus=0, q_bminus=±1.
    sq_err_bplus  = (0 − w)² = 0.01    (BOTH signs)
    sq_err_bminus = (sign(w) − w)² = 0.81  (BOTH signs)
    local_diff    = -0.40              (BOTH signs)
    grad_bits     = |+1| · -0.40 = -0.40   (NEGATIVE for BOTH signs)

    SGD raises bits regardless of weight sign — the correct escape
    direction. Round 6 had grad_bits flipping sign with w (which gave
    grad·sign(w) < 0 uniformly); Round 7 has grad_bits constant in sign
    (which gives grad < 0 uniformly).
    """
    weight, bits_param, q = _quantize_with_grad(
        weight_val=0.1, bits=2.0, scale_val=1.0,
    )
    assert torch.allclose(q, torch.zeros_like(q))
    grad_out = torch.ones_like(q)
    q.backward(grad_out)
    assert bits_param.grad is not None
    # Round 7: grad_bits is uniformly NEGATIVE under synthetic +1 at
    # the zero bin, regardless of weight sign.
    assert torch.all(bits_param.grad < 0), (
        f"Round 7 surrogate sign convention violated: grad = "
        f"{bits_param.grad.tolist()} — should be uniformly negative "
        f"under synthetic +1 upstream at the zero bin."
    )


# ── Math safety check coverage (the four codex Round 6 invariants) ──────


def test_math_safety_check_zero_bin():
    """Safety check 1: For w=0.1, scale=1, b=2 (zero-bin), under synthetic
    positive upstream, grad_bits should be NEGATIVE (raise bits → escape
    zero bin) for positive weight. Round 4 invariant preserved in the
    synthetic-upstream regime.
    """
    scale = torch.ones(1, dtype=torch.float64)
    weight = torch.tensor([[[[+0.1]]]], dtype=torch.float64, requires_grad=True)
    bits = torch.tensor([[[[2.0]]]], dtype=torch.float64, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)
    q.backward(torch.ones_like(q))
    assert bits.grad is not None
    assert bits.grad.item() < 0, (
        f"Safety check 1 (zero bin escape) failed: grad_bits = "
        f"{bits.grad.item()}, expected NEGATIVE."
    )


def test_math_safety_check_sweet_spot():
    """Safety check 2 (Round 7 update): For w=1/3+0.001, scale=1, b=3
    (sweet spot), under SYNTHETIC +1 upstream the LOSS-DELTA FD is
    NEGATIVE (b=2 is much worse than b=4, so local FD says raise bits).

    Under MSE chain at the sweet spot the bit-update is gated by the
    tiny ``|grad_output| = |q − w| = 0.001`` magnitude scaler — see
    ``test_sweet_spot_w_one_third_b3_no_push_to_b4`` for the chain-
    magnitude check that ensures SGD barely moves at the sweet spot.
    """
    scale = torch.ones(1, dtype=torch.float64)
    weight = torch.tensor(
        [[[[1.0 / 3.0 + 0.001]]]], dtype=torch.float64, requires_grad=True
    )
    bits = torch.tensor([[[[3.0]]]], dtype=torch.float64, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)
    q.backward(torch.ones_like(q))
    assert bits.grad is not None
    # Round 7 synthetic-upstream sign at sweet spot: NEGATIVE
    # (loss-delta dominated by b=2 being far worse than b=4).
    assert bits.grad.item() < 0, (
        f"Safety check 2 (sweet spot synthetic) failed: grad_bits = "
        f"{bits.grad.item()}, expected NEGATIVE under loss-delta FD."
    )


# ── Round 7 NEW math safety checks under REAL MSE CHAIN ─────────────────


def test_zero_bin_under_real_mse_chain_round7():
    """Round 7 Bug 1 fix: w=+0.1, scale=1, b=2 under the REAL MSE chain
    (the chain that broke Round 6).

    grad_output (MSE) = q − w = -0.1
    sq_err_bplus  = (0 − 0.1)² = 0.01
    sq_err_bminus = (1 − 0.1)² = 0.81
    local_diff    = -0.40
    grad_bits     = |-0.1| · -0.40 = -0.04   (NEGATIVE → SGD raises bits ✓)

    Under Round 6 q-delta FD this was +0.05 → SGD lowered bits → MSE
    went from 0.01 to 0.81 (Pareto-dominated). Round 7 restores the
    correct escape direction.
    """
    scale = torch.ones(1, dtype=torch.float64)
    weight = torch.tensor([[[[+0.1]]]], dtype=torch.float64, requires_grad=True)
    bits = torch.tensor([[[[2.0]]]], dtype=torch.float64, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)
    # Real MSE chain — half-MSE so dL/dq = q - w.
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()
    assert bits.grad is not None
    assert bits.grad.item() < 0, (
        f"Round 7 Bug 1 fix failed: under MSE chain w=+0.1 b=2, "
        f"grad_bits = {bits.grad.item()}, expected NEGATIVE "
        f"(Round 6 would have given +0.05 — pushing bits DOWN → MSE 0.01→0.81)."
    )
    # Pin exact value: -0.04
    assert abs(bits.grad.item() - (-0.04)) < 1e-12, (
        f"Round 7 zero-bin MSE-chain grad value mismatch: "
        f"got {bits.grad.item()}, expected -0.04"
    )


def test_w_zero_b1_escapes_via_bit_gradient():
    """Round 7 Bug 2 fix: w=0, b=1. The 1-bit sign quantizer is
    ASYMMETRIC at w=0 — ``weight >= 0`` returns +scale, so q=+1 even
    though w=0. Distortion exists; SGD should be told to raise bits so
    b≥2 can map w=0 → q=0.

    Round 6 used ``weight == 0`` short-circuit which suppressed grad_bits
    here, leaving SGD stuck at b=1. Round 7 uses ``q == weight`` instead;
    at w=0, b=1 we have q=+1 ≠ 0 = weight, so the gradient flows.

    For w=0, scale=1, b=1: q=+1 (sign), q_bplus(b=2)=round(0/1)*1=0,
    q_bminus(b=1)=+1.
    grad_output (MSE) = q − w = +1.0.
    sq_err_bplus  = (0 − 0)² = 0
    sq_err_bminus = (1 − 0)² = 1
    local_diff    = (0 − 1)/2 = -0.5
    grad_bits     = |+1| · -0.5 = -0.5   (NEGATIVE → SGD raises bits ✓)
    """
    scale = torch.ones(1, dtype=torch.float64)
    weight = torch.tensor([[[[0.0]]]], dtype=torch.float64, requires_grad=True)
    bits = torch.tensor([[[[1.0]]]], dtype=torch.float64, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)
    # Sanity: sign quantizer maps 0 → +scale.
    assert q.item() == 1.0, (
        f"setup: 1-bit q at w=0 should be +scale=+1 (asymmetric sign quantizer); "
        f"got {q.item()}"
    )
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()
    assert bits.grad is not None
    # Round 7 fix: gradient should NOW flow (not suppressed by
    # ``weight == 0`` mask) and point to RAISING bits.
    assert bits.grad.item() < 0, (
        f"Round 7 Bug 2 fix failed: w=0 b=1 grad_bits = {bits.grad.item()}, "
        f"expected NEGATIVE (escape b=1 sign quantizer where q=+1 ≠ 0 = w). "
        f"Round 6 ``weight == 0`` mask incorrectly suppressed this signal."
    )


def test_sweet_spot_w_one_third_b3_no_push_to_b4():
    """Round 7 sweet-spot protection comes from |grad_output| magnitude
    scaler, not the FD direction.

    For w=1/3+0.001, scale=1, b=3 under MSE chain:
    grad_output = q − w = -0.001 (TINY)
    |grad_output| = 0.001
    sq_err_bplus  = (2/7 − w)²  ≈ 2.36e-3
    sq_err_bminus = (0   − w)²  ≈ 0.112
    local_diff    ≈ -0.0547
    grad_bits     = 0.001 · -0.0547 ≈ -5.47e-5  (TINY magnitude)

    The FD direction is "raise bits" but the magnitude is so small that
    SGD with any reasonable LR barely moves bits — preserving the sweet
    spot. The KEY contract: at the sweet spot ``|grad_bits| < 1e-3``
    so SGD doesn't push us into the worse b=4 grid.
    """
    scale = torch.ones(1, dtype=torch.float64)
    w_val = 1.0 / 3.0 + 0.001
    weight = torch.tensor(
        [[[[w_val]]]], dtype=torch.float64, requires_grad=True
    )
    bits = torch.tensor([[[[3.0]]]], dtype=torch.float64, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)
    # Sanity: q at sweet spot is 1/3.
    assert abs(q.item() - 1.0 / 3.0) < 1e-12, (
        f"setup: sweet-spot q expected 1/3, got {q.item()}"
    )
    loss = 0.5 * ((q - weight) ** 2).sum()
    loss.backward()
    assert bits.grad is not None
    # Sweet-spot protection: magnitude must be tiny under MSE chain.
    assert abs(bits.grad.item()) < 1e-3, (
        f"Round 7 sweet-spot protection failed: |grad_bits| = "
        f"{abs(bits.grad.item())} under MSE chain at w=1/3+0.001, b=3; "
        f"expected < 1e-3 so SGD doesn't push to worse b=4 grid."
    )


def test_math_safety_check_zero_distortion():
    """Safety check 3: For w=0 (no distortion), grad_bits = 0."""
    scale = torch.ones(1, dtype=torch.float64)
    weight = torch.tensor([[[[0.0]]]], dtype=torch.float64, requires_grad=True)
    bits = torch.tensor([[[[2.0]]]], dtype=torch.float64, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)
    q.backward(torch.ones_like(q))
    assert bits.grad is not None
    assert bits.grad.item() == 0.0, (
        f"Safety check 3 (zero distortion) failed: grad_bits = "
        f"{bits.grad.item()}, expected exactly 0."
    )


def test_math_safety_check_well_quantized_small_magnitude():
    """Safety check 4: For w=0.5, scale=1, b=2, grad_bits should be
    "small" (no urgent need to change bits). With central FD:
        q (b=2) = round(0.5/1)*1 = 0
        q_bplus (b=3) = round(0.5/0.333)*0.333 = 2/3 ≈ 0.667
        q_bminus (b=1) = sign(+0.5)·1 = +1
        (qp − qm)/2 = (0.667 − 1)/2 = -0.167
    Magnitude 0.167 is "small" relative to scale=1.
    """
    scale = torch.ones(1, dtype=torch.float64)
    weight = torch.tensor([[[[+0.5]]]], dtype=torch.float64, requires_grad=True)
    bits = torch.tensor([[[[2.0]]]], dtype=torch.float64, requires_grad=True)
    q = _PerElementSTEQuantize.apply(weight, scale, bits)
    q.backward(torch.ones_like(q))
    assert bits.grad is not None
    # "Small" defined as |grad_bits| < 0.5 (less than half the scale).
    assert abs(bits.grad.item()) < 0.5, (
        f"Safety check 4 (well-quantized small grad) failed: |grad_bits| = "
        f"{abs(bits.grad.item())}, expected < 0.5."
    )
