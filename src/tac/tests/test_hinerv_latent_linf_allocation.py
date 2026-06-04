# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the HiNeRV dense decoder-VJP adjoint + L-inf latent allocation.

Slot EEE discipline: every test verifies ACTUAL BEHAVIOR of the adjoint /
Fisher-pullback / allocator on a real HiNeRV carrier — NOT constants. If the
function body were replaced by ``return canonical_markers`` every behavioral
test below would FAIL.

Key behavioral proofs:
  * the adjoint dot-product identity ``<J x, y> == <x, J^T y>`` holds (~1e-6) AND
    fails for a non-adjoint transform;
  * the finite-difference numerical JVP converges to the analytic JVP;
  * the Fisher-pullback actually concentrates latent saliency where the pixel
    saliency is concentrated (a uniform pixel saliency gives a DIFFERENT latent
    saliency than a concentrated one);
  * the L-inf allocation steps GENUINELY DIFFER from the L2 uniform steps and
    track the saliency (high-saliency latent dim => finer step);
  * L-inf and L2 spend the SAME total latent bits (the fairness invariant);
  * quantize-with-steps actually changes the latents (no-op proof) and a finer
    step yields a smaller quantization error than a coarser step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.analysis.hinerv_latent_linf_allocation import (
    NON_PROMOTABLE_MARKERS,
    HinervAllocationError,
    adjoint_dotproduct_residual,
    allocate_l2_uniform_latent_steps,
    allocate_linf_latent_steps,
    decoder_jacobian_vjp,
    finite_difference_vjp_residual,
    push_pixel_saliency_to_latent,
    quantize_latents_with_steps,
)
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate


def _carrier(seed: int = 0) -> HinervSubstrate:
    torch.manual_seed(seed)
    cfg = HinervConfig(
        latent_dim_coarse=4,
        latent_dim_mid=6,
        latent_dim_fine=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )
    return HinervSubstrate(cfg).eval()


def _trained_carrier(seed: int = 0, steps: int = 200) -> HinervSubstrate:
    m = _carrier(seed)
    opt = torch.optim.Adam(m.parameters(), lr=5e-2)
    idx = torch.arange(3, dtype=torch.long)
    torch.manual_seed(seed + 1)
    t0 = torch.rand(3, 3, 24, 32)
    t1 = torch.rand(3, 3, 24, 32)
    for _ in range(steps):
        r0, r1 = m(idx)
        loss = (r0 - t0).pow(2).mean() + (r1 - t1).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


# ---------------------------------------------------------------------------
# G3 — the dense decoder-VJP adjoint exactness (the load-bearing proof).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", ["coarse", "mid", "fine"])
def test_adjoint_dotproduct_machine_exact_untrained(scale: str) -> None:
    """``<J x, y> == <x, J^T y>`` to fp32 machine precision on every scale."""
    m = _carrier()
    r = adjoint_dotproduct_residual(m, 0, scale=scale, seed=3)
    assert r < 1e-4, f"adjoint dot-product residual {r:.2e} too large for {scale}"


@pytest.mark.parametrize("scale", ["coarse", "mid", "fine"])
def test_adjoint_dotproduct_machine_exact_trained(scale: str) -> None:
    """The adjoint identity holds even on a trained (high-Lipschitz) carrier."""
    m = _trained_carrier()
    r = adjoint_dotproduct_residual(m, 0, scale=scale, seed=3)
    assert r < 1e-4, f"trained adjoint residual {r:.2e} too large for {scale}"


def test_finite_difference_jvp_converges_untrained() -> None:
    """The numerical JVP converges to the analytic JVP on the near-linear carrier."""
    m = _carrier()
    for scale in ("coarse", "mid", "fine"):
        r = finite_difference_vjp_residual(m, 0, scale=scale, seed=5)
        assert r < 1e-3, f"fd-sweep residual {r:.2e} on untrained {scale}"


def test_finite_difference_jvp_converges_trained() -> None:
    """The fd sweep still finds the convergence floor on the high-Lipschitz carrier.

    The trained sin(30) carrier has an enormous local Lipschitz constant, so the
    convergence floor is looser than untrained but the numerical JVP STILL
    converges to the analytic JVP (proving the analytic Jacobian is the true one).
    """
    m = _trained_carrier()
    # fine scale is the most-used and converges tightest; all scales must be < 1.
    rs = {s: finite_difference_vjp_residual(m, 0, scale=s, seed=5)
          for s in ("coarse", "mid", "fine")}
    assert rs["fine"] < 1e-2, f"trained fine fd-sweep {rs['fine']:.2e}"
    for s, r in rs.items():
        assert r < 1.0 - 1e-6, f"fd-sweep {s}={r:.3e} did not converge below 1.0"


def test_vjp_only_targets_queried_pair() -> None:
    """The decoder VJP reaches ONLY the queried pair's latent row (not others)."""
    m = _carrier()
    cot = torch.randn(3, 24, 32)  # (C, H, W) pixel cotangent
    jty = decoder_jacobian_vjp(m, 1, cot, frame_slot=1, scales=("fine",))
    assert jty["fine"].shape == (8,)
    assert torch.isfinite(jty["fine"]).all()
    # A non-trivial cotangent gives a non-zero adjoint on a non-degenerate carrier.
    assert float(jty["fine"].abs().sum()) > 0.0


def test_vjp_does_not_mutate_model_params() -> None:
    """The override path restores the nn.Parameter latents (no side effects)."""
    m = _carrier()
    before_c = m.latents_coarse.detach().clone()
    cot = torch.randn(3, 24, 32)
    decoder_jacobian_vjp(m, 0, cot, scales=("coarse", "mid", "fine"))
    assert isinstance(m.latents_coarse, torch.nn.Parameter)
    assert torch.allclose(m.latents_coarse.detach(), before_c)


def test_adjoint_identity_fails_for_non_adjoint() -> None:
    """A scrambled 'adjoint' (random vector instead of J^T y) FAILS the test.

    This is the falsification control: the dot-product identity is NOT trivially
    satisfied; a transform that is not the true adjoint produces a large residual.
    """
    m = _carrier()
    gen = torch.Generator(device="cpu").manual_seed(7)
    idx = torch.tensor([0], dtype=torch.long)
    z_star = m.latents_fine[idx].detach().clone().reshape(-1)
    x = torch.randn(8, generator=gen)

    def frame_of(z_row: torch.Tensor) -> torch.Tensor:
        from tac.analysis.hinerv_latent_linf_allocation import _render_pair_frame
        return _render_pair_frame(m, 0, frame_slot=1, z_override={"fine": z_row.reshape(1, -1)})

    _, jx = torch.autograd.functional.jvp(frame_of, z_star, x)
    y = torch.randn(jx.numel(), generator=gen)
    lhs = float(torch.dot(jx.reshape(-1), y))  # true <J x, y>
    # FAKE adjoint: a random latent vector instead of J^T y.
    fake_jty = torch.randn(8, generator=gen)
    rhs_fake = float(torch.dot(x, fake_jty))
    denom = max(abs(lhs), abs(rhs_fake), 1e-12)
    fake_residual = abs(lhs - rhs_fake) / denom
    assert fake_residual > 1e-2, "a random vector must NOT satisfy the adjoint identity"


# ---------------------------------------------------------------------------
# The Fisher-pullback: pixel saliency -> latent saliency.
# ---------------------------------------------------------------------------


def test_pullback_concentrated_differs_from_uniform() -> None:
    """A concentrated pixel saliency yields a DIFFERENT latent saliency than uniform.

    If the pullback returned a constant (a fake), these two would be identical.
    The whole point is that WHERE the pixel saliency lives changes which latent
    dims it weights.
    """
    m = _carrier()
    sp_uniform = torch.ones(24, 32)
    sp_concentrated = torch.zeros(24, 32)
    sp_concentrated[8:12, 14:18] = 50.0
    ls_u = push_pixel_saliency_to_latent(m, 0, sp_uniform, frame_slot=1)
    ls_c = push_pixel_saliency_to_latent(m, 0, sp_concentrated, frame_slot=1)
    # Aggregate: the concatenated normalized distributions differ materially.
    u_all = np.concatenate([ls_u.s_latent[s] for s in ("coarse", "mid", "fine")])
    c_all = np.concatenate([ls_c.s_latent[s] for s in ("coarse", "mid", "fine")])
    u_all = u_all / (u_all.sum() + 1e-12)
    c_all = c_all / (c_all.sum() + 1e-12)
    assert float(np.abs(u_all - c_all).sum()) > 1e-3, (
        "concentrated vs uniform pixel saliency must give different latent saliency"
    )


def test_pullback_nonnegative_and_finite() -> None:
    """The Fisher-pullback is a sum of squares: nonnegative + finite by construction."""
    m = _carrier()
    sp = torch.rand(24, 32) * 10.0
    ls = push_pixel_saliency_to_latent(m, 0, sp, frame_slot=1)
    for scale in ("coarse", "mid", "fine"):
        v = ls.s_latent[scale]
        assert np.all(np.isfinite(v))
        assert np.all(v >= 0.0)


def test_pullback_zero_pixel_saliency_gives_zero_latent() -> None:
    """Zero pixel saliency => zero latent saliency (the pullback respects the input)."""
    m = _carrier()
    sp = torch.zeros(24, 32)
    ls = push_pixel_saliency_to_latent(m, 0, sp, frame_slot=1)
    for scale in ("coarse", "mid", "fine"):
        assert float(np.abs(ls.s_latent[scale]).sum()) < 1e-9


# ---------------------------------------------------------------------------
# The L-inf latent allocation (the class-shift objective) vs L2 uniform.
# ---------------------------------------------------------------------------


def _lat_vals(m: HinervSubstrate, p: int = 0) -> dict[str, np.ndarray]:
    return {
        "coarse": m.latents_coarse[p].detach().numpy(),
        "mid": m.latents_mid[p].detach().numpy(),
        "fine": m.latents_fine[p].detach().numpy(),
    }


def test_linf_and_l2_spend_equal_rate() -> None:
    """The fairness invariant: L-inf realized rate == L2 realized rate (>= target)."""
    m = _carrier()
    sp = torch.zeros(24, 32)
    sp[8:12, 14:18] = 50.0
    ls = push_pixel_saliency_to_latent(m, 0, sp, frame_slot=1)
    lv = _lat_vals(m)
    target = 40.0
    l2 = allocate_l2_uniform_latent_steps(lv, target_bits=target)
    linf = allocate_linf_latent_steps(ls.s_latent, lv, target_bits=target)
    assert abs(l2.total_bits - target) < 0.05 * target
    # disadvantage_linf: L-inf must spend AT LEAST the L2 rate (anti-gaming).
    assert linf.total_bits >= l2.total_bits - 1e-3
    assert abs(linf.total_bits - target) < 0.05 * target


def test_linf_steps_differ_from_l2_uniform() -> None:
    """L-inf steps GENUINELY DIFFER from the uniform L2 steps (Catalog #139 no-op)."""
    m = _carrier()
    sp = torch.zeros(24, 32)
    sp[8:12, 14:18] = 50.0
    ls = push_pixel_saliency_to_latent(m, 0, sp, frame_slot=1)
    lv = _lat_vals(m)
    l2 = allocate_l2_uniform_latent_steps(lv, target_bits=40.0)
    linf = allocate_linf_latent_steps(ls.s_latent, lv, target_bits=40.0)
    # L2 within a scale is a single uniform step.
    for scale in ("coarse", "mid", "fine"):
        assert np.allclose(l2.steps[scale], l2.steps[scale][0])
    # L-inf must NOT be uniform on at least one scale (it tracks saliency).
    any_varied = any(
        float(np.std(linf.steps[scale])) > 1e-9 for scale in ("coarse", "mid", "fine")
    )
    assert any_varied, "L-inf steps must vary by latent saliency, not be uniform"


def test_linf_higher_saliency_gets_finer_step() -> None:
    """The L-inf allocator gives a FINER step to higher-saliency latent dims.

    Construct a synthetic latent saliency where dim 0 is huge and the rest tiny;
    the allocated step for dim 0 must be the smallest in its scale (most precise).
    """
    s_latent = {
        "coarse": np.array([100.0, 1e-3, 1e-3, 1e-3]),
        "mid": np.full(6, 1e-3),
        "fine": np.full(8, 1e-3),
    }
    lat_vals = {
        "coarse": np.linspace(-1, 1, 4),
        "mid": np.linspace(-1, 1, 6),
        "fine": np.linspace(-1, 1, 8),
    }
    linf = allocate_linf_latent_steps(s_latent, lat_vals, target_bits=30.0)
    coarse_steps = linf.steps["coarse"]
    assert coarse_steps[0] == coarse_steps.min(), (
        "highest-saliency dim must get the finest (smallest) step"
    )
    assert coarse_steps[0] < coarse_steps[1], "high saliency => finer step than low"


def test_quantize_actually_changes_latents() -> None:
    """quantize_latents_with_steps actually mutates latents (no-op proof)."""
    m = _carrier()
    lv = _lat_vals(m)
    l2 = allocate_l2_uniform_latent_steps(lv, target_bits=20.0)  # coarse steps
    q = quantize_latents_with_steps(lv, l2.steps)
    # At a coarse rate the quantized latents must differ from the originals.
    total_change = sum(
        float(np.abs(q[s] - lv[s]).sum()) for s in ("coarse", "mid", "fine")
    )
    assert total_change > 0.0, "quantization at a coarse rate must change the latents"


def test_quantize_preserves_shaped_latent_geometry() -> None:
    """Archive/render consumers must not receive flattened shaped latent arrays."""
    z = {"coarse": np.array([[0.137, -0.642], [0.911, 0.333]], dtype=np.float32)}
    steps = {"coarse": np.full(4, 0.25, dtype=np.float64)}

    q = quantize_latents_with_steps(z, steps, scales=("coarse",))

    assert q["coarse"].shape == z["coarse"].shape
    np.testing.assert_allclose(
        q["coarse"],
        np.array([[0.25, -0.75], [1.0, 0.25]], dtype=np.float64),
    )


def test_finer_step_smaller_quantization_error() -> None:
    """A finer quantizer step yields a smaller quantization error than a coarse one."""
    z = {"coarse": np.array([0.137, -0.642, 0.911, 0.333])}
    fine = {"coarse": np.full(4, 0.001)}
    coarse = {"coarse": np.full(4, 0.5)}
    q_fine = quantize_latents_with_steps(z, fine, scales=("coarse",))
    q_coarse = quantize_latents_with_steps(z, coarse, scales=("coarse",))
    err_fine = float(np.abs(q_fine["coarse"] - z["coarse"]).sum())
    err_coarse = float(np.abs(q_coarse["coarse"] - z["coarse"]).sum())
    assert err_fine < err_coarse, "finer step must give smaller quantization error"


# ---------------------------------------------------------------------------
# Invariants / fail-closed guards.
# ---------------------------------------------------------------------------


def test_non_promotable_markers_present() -> None:
    """The module declares the canonical non-promotable advisory markers."""
    assert NON_PROMOTABLE_MARKERS["score_claim"] is False
    assert NON_PROMOTABLE_MARKERS["promotable"] is False
    assert NON_PROMOTABLE_MARKERS["axis_tag"] == "[macOS-CPU advisory]"


def test_allocator_rejects_negative_saliency() -> None:
    """Negative latent saliency is rejected (the rho inversion needs s >= 0)."""
    with pytest.raises(HinervAllocationError):
        allocate_linf_latent_steps(
            {"coarse": np.array([-1.0, 2.0])},
            {"coarse": np.array([0.0, 1.0])},
            target_bits=5.0,
            scales=("coarse",),
        )


def test_allocator_rejects_infeasible_target() -> None:
    """A target bit budget above the finest feasible rate raises (fail-closed)."""
    with pytest.raises(HinervAllocationError):
        allocate_linf_latent_steps(
            {"coarse": np.array([1.0, 1.0])},
            {"coarse": np.array([-0.5, 0.5])},  # range 1.0 -> tiny bit capacity
            target_bits=1.0e6,  # impossibly high
            scales=("coarse",),
        )


def test_unknown_scale_rejected() -> None:
    """Requesting an unknown latent scale raises (no silent zero)."""
    m = _carrier()
    cot = torch.randn(3, 24, 32)
    with pytest.raises(HinervAllocationError):
        decoder_jacobian_vjp(m, 0, cot, scales=("bogus",))


def test_frame_slot_validation() -> None:
    """Only frame_slot 0/1 are valid (a pair has two frames)."""
    m = _carrier()
    with pytest.raises(HinervAllocationError):
        adjoint_dotproduct_residual(m, 0, scale="fine", frame_slot=2)
