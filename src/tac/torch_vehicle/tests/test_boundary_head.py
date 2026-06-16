# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the boundary-conditioned / depthwise-separable HEAD variant.

The contracts proven (REAL behavior, never constants):
  1. PARITY: default-off is bit-identical to the parent ConfigurableTaperHNeRVDecoder
     (same state_dict keys, same forward) -> inherits the vendored parity contract.
  2. The exact param-count formulas equal the ACTUAL nn parameter counts.
  3. The vendored full ``refine`` is O(C^2)-in-WIDTH; the depthwise-separable refine is
     (a) strictly cheaper than the full refine at every width, AND (b) grows EXACTLY
     LINEARLY along its DEPTH axis (n_stages) -> capacity-at-linear-cost mechanism.
  4. The enabled forward is finite, correct shape (2,2,3,384,512), and TRAINS (grad flows
     to the separable refine AND the low-rank head residual).
  5. The optional low-rank head residual is zero-init -> enabling it starts as a no-op,
     and it actually shifts the rgb_1 output once its params move.

Every test would FAIL if a body were replaced by a constant return.
"""
from __future__ import annotations

import pytest
import torch

from tac.torch_vehicle.boundary_head import (
    BoundaryHeadTaperDecoder,
    BoundaryLowRankHeadResidual,
    DepthwiseSeparableRefine,
    full_refine_param_count,
    lowrank_head_residual_param_count,
    separable_refine_param_count,
)
from tac.torch_vehicle.configurable_taper_decoder import (
    ConfigurableTaperHNeRVDecoder,
    vendored_taper,
)


# ---- 1. PARITY (default-off bit-identical) ---------------------------------
def test_default_off_is_bit_identical_to_parent():
    """boundary_head_enabled=False -> same state_dict keys + bit-identical forward."""
    ch = vendored_taper(20)
    torch.manual_seed(0)
    parent = ConfigurableTaperHNeRVDecoder(latent_dim=28, channels=ch)
    child = BoundaryHeadTaperDecoder(latent_dim=28, channels=ch, boundary_head_enabled=False)
    assert set(parent.state_dict().keys()) == set(child.state_dict().keys()), (
        "default-off must not add/remove any state_dict keys"
    )
    child.load_state_dict(parent.state_dict())
    z = torch.randn(2, 28)
    with torch.no_grad():
        a, b = parent(z), child(z)
    assert torch.equal(a, b), "default-off forward is not bit-identical to the parent"
    assert a.shape == (2, 2, 3, 384, 512)


def test_lowrank_only_enabled_path_equals_parent_at_zero_init():
    """When ONLY the low-rank residual is on (separable refine OFF -> vendored full refine),
    the forward takes the enabled branch but uses the vendored refine + a ZERO-INIT residual
    -> it must be bit-identical to the parent. This guards the enabled-path forward against
    drifting from the parent's verbatim-vendored forward (the only difference allowed at
    init is the zero residual, which is a no-op)."""
    ch = vendored_taper(20)
    torch.manual_seed(3)
    parent = ConfigurableTaperHNeRVDecoder(latent_dim=28, channels=ch)
    child = BoundaryHeadTaperDecoder(
        latent_dim=28, channels=ch,
        boundary_head_enabled=False, lowrank_residual_enabled=True, lowrank_rank=4,
    )
    child.load_state_dict(parent.state_dict(), strict=False)
    z = torch.randn(2, 28)
    with torch.no_grad():
        a, b = parent(z), child(z)
    assert torch.equal(a, b), "lowrank-only zero-init enabled path drifted from parent forward"


def test_default_off_round_trips_vendored_weights_bit_identical():
    """Cross-load REAL vendored decoder weights -> default-off child -> bit-identical
    forward. This is the inherited faithful-generalization gate (real cross-load, not a
    self-roundtrip)."""
    from tac.torch_vehicle.driver import import_vendored_bundle

    v = import_vendored_bundle()
    vd = v.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    child = BoundaryHeadTaperDecoder(
        latent_dim=28, channels=vendored_taper(20), boundary_head_enabled=False
    )
    child.load_state_dict(vd.state_dict())
    z = torch.randn(2, 28)
    with torch.no_grad():
        a, b = vd(z), child(z)
    assert torch.equal(a, b), "default-off does not match vendored decoder bit-for-bit"


# ---- 2. exact param-count formulas ----------------------------------------
@pytest.mark.parametrize("cf", [8, 10, 16, 20, 32])
def test_full_refine_param_count_is_exact(cf):
    """full_refine_param_count == actual params of the vendored refine block."""
    refine = torch.nn.Sequential(
        torch.nn.Conv2d(cf, cf // 2, 3, padding=2, dilation=2),
        torch.nn.Conv2d(cf // 2, cf, 3, padding=1),
    )
    actual = sum(p.numel() for p in refine.parameters())
    assert full_refine_param_count(cf) == actual


@pytest.mark.parametrize("cf", [8, 16, 32])
@pytest.mark.parametrize("ns", [1, 2, 3])
def test_separable_refine_param_count_is_exact(cf, ns):
    m = DepthwiseSeparableRefine(cf, n_stages=ns)
    actual = sum(p.numel() for p in m.parameters())
    assert separable_refine_param_count(cf, ns) == actual


@pytest.mark.parametrize("cf", [8, 16, 32])
def test_lowrank_head_residual_param_count_is_exact(cf):
    m = BoundaryLowRankHeadResidual(cf, rank=4)
    actual = sum(p.numel() for p in m.parameters())
    assert lowrank_head_residual_param_count(cf, 4) == actual


# ---- 3. the capacity-at-linear-cost mechanism -----------------------------
def test_full_refine_is_quadratic_in_width():
    """Doubling final_ch ~quadruples the vendored full-refine params (O(C^2))."""
    p16 = full_refine_param_count(16)
    p32 = full_refine_param_count(32)
    p64 = full_refine_param_count(64)
    # ratio approaches 4x as width doubles (quadratic). Allow a band for the //2 + bias terms.
    assert 3.5 < p32 / p16 < 4.5, f"full refine not ~quadratic 16->32: {p32/p16:.2f}"
    assert 3.5 < p64 / p32 < 4.5, f"full refine not ~quadratic 32->64: {p64/p32:.2f}"


def test_separable_refine_depth_axis_is_exactly_linear():
    """At fixed width, separable refine params grow EXACTLY linearly in n_stages -
    capacity along DEPTH at O(stages) (the mechanism this module exists to provide)."""
    cf = 16
    p = [separable_refine_param_count(cf, ns) for ns in (1, 2, 3, 4)]
    deltas = [p[i + 1] - p[i] for i in range(len(p) - 1)]
    assert len(set(deltas)) == 1, f"depth axis not exactly linear: deltas={deltas}"
    assert deltas[0] == separable_refine_param_count(cf, 1), "per-stage delta != one-stage cost"


def test_separable_refine_strictly_cheaper_than_full_at_every_width():
    """The separable refine (2 stages) is strictly cheaper than the vendored full refine
    at every contest-relevant width -> it factorizes the k^2 out of the width-mixing."""
    for cf in (8, 16, 20, 32, 64):
        assert separable_refine_param_count(cf, 2) < full_refine_param_count(cf), (
            f"separable not cheaper than full at cf={cf}"
        )


def test_separable_refine_advantage_grows_with_width():
    """The separable savings RATIO improves as width grows (the C^2 trap is worse at large
    width, exactly where you'd be tempted to widen the head)."""
    ratio16 = separable_refine_param_count(16, 2) / full_refine_param_count(16)
    ratio64 = separable_refine_param_count(64, 2) / full_refine_param_count(64)
    assert ratio64 < ratio16, (
        f"separable advantage should grow with width: {ratio64:.3f} !< {ratio16:.3f}"
    )


# ---- 4. enabled forward: finite, correct shape, TRAINS --------------------
def test_enabled_separable_forward_shape_and_finite():
    ch = vendored_taper(20)
    m = BoundaryHeadTaperDecoder(
        latent_dim=28, channels=ch, boundary_head_enabled=True, separable_n_stages=2
    )
    z = torch.randn(2, 28)
    out = m(z)
    assert out.shape == (2, 2, 3, 384, 512)
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 255.0).all(), "output not in sigmoid*255 range"


def test_enabled_forward_grad_flows_to_separable_refine_and_residual():
    """Real gradient flow to BOTH the separable refine and the low-rank residual when
    both are enabled (it TRAINS)."""
    ch = vendored_taper(20)
    m = BoundaryHeadTaperDecoder(
        latent_dim=28,
        channels=ch,
        boundary_head_enabled=True,
        separable_n_stages=2,
        lowrank_residual_enabled=True,
        lowrank_rank=4,
    )
    z = torch.randn(2, 28)
    out = m(z)
    out.mean().backward()
    # separable refine params got grad
    refine_grad = sum(
        p.grad.abs().sum().item()
        for p in m.refine.parameters()
        if p.grad is not None
    )
    assert refine_grad > 0, "no gradient reached the separable refine"
    # low-rank residual V got grad (U starts zero-init so its grad may route via V/U chain)
    v_grad = sum(
        p.grad.abs().sum().item()
        for p in m.rgb_1_boundary.v.parameters()
        if p.grad is not None
    )
    u_grad = sum(
        p.grad.abs().sum().item()
        for p in m.rgb_1_boundary.u.parameters()
        if p.grad is not None
    )
    assert (v_grad + u_grad) > 0, "no gradient reached the low-rank head residual"


# ---- 5. low-rank residual zero-init -> graceful A/B start ------------------
def test_lowrank_residual_is_zero_init_no_op_then_shifts_output():
    """Zero-init U -> the residual is EXACTLY zero at construction (enabling it does not
    perturb a warm-started head). After perturbing U it actually shifts rgb_1 output."""
    cf = 16
    res = BoundaryLowRankHeadResidual(cf, rank=4)
    x = torch.randn(2, cf, 8, 8)
    out0 = res(x)
    assert torch.count_nonzero(out0) == 0, "zero-init residual must be exactly zero"
    # perturb U -> residual becomes non-trivial (it really computes U(V(x)))
    with torch.no_grad():
        res.u.weight.add_(torch.randn_like(res.u.weight))
    out1 = res(x)
    assert out1.abs().sum() > 0, "residual did not respond to a real weight change"
    assert out1.shape == (2, 3, 8, 8)


def test_lowrank_enabled_decoder_starts_bit_identical_to_separable_only():
    """With separable refine on + low-rank residual on (zero-init), the frame-1 output is
    IDENTICAL to separable-refine-only at init (the residual is a no-op until trained) ->
    enabling the boundary residual is a graceful, side-effect-free A/B start on rgb_1."""
    ch = vendored_taper(20)
    torch.manual_seed(7)
    a = BoundaryHeadTaperDecoder(
        latent_dim=28, channels=ch, boundary_head_enabled=True, separable_n_stages=2,
        lowrank_residual_enabled=False,
    )
    b = BoundaryHeadTaperDecoder(
        latent_dim=28, channels=ch, boundary_head_enabled=True, separable_n_stages=2,
        lowrank_residual_enabled=True, lowrank_rank=4,
    )
    # copy shared weights so only the (zero-init) residual differs
    b.load_state_dict(a.state_dict(), strict=False)
    z = torch.randn(2, 28)
    with torch.no_grad():
        oa, ob = a(z), b(z)
    # frame 1 (index 1) is where the residual is added; it must match at zero-init
    assert torch.allclose(oa[:, 1], ob[:, 1], atol=1e-5), (
        "zero-init boundary residual must not change rgb_1 output at init"
    )
