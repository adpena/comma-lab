# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the 2026-06-24 next-gen INR activation expansion.

Per the SUPREME RULE (no fake implementations): each test verifies REAL behavior of
the named activation — that the functional form is actually applied to the input, that
every hyperparameter actually modulates the output, and that "siren" stays the
bit-identical control. A test here FAILS if any activation degenerates to a no-op or
ignores its hyperparameter (constants-not-behavior is forbidden).

These cover the FIXED-FORM families (gauss/hosc/sinc/rcgauss/finer_gauss) at the
stateless apply_activation_family surface, and the LEARNABLE families
(step_basis/fkan) at the nn.Module surface. The torch_vehicle decoder-level parity
(siren bit-identical to vendored; non-siren genuinely differs) is covered in
tac/torch_vehicle/tests/test_configurable_taper_decoder.py.
"""
from __future__ import annotations

import math

import pytest
import torch

from tac.substrates.siren.activation_family import (
    FIXED_FORM_ACTIVATION_IDS,
    FourierKANActivation,
    LearnableStepBasis,
    apply_activation_family,
    is_learnable_activation,
    make_learnable_activation,
    normalize_activation_family,
)

_OMEGA = 1.0
_WIRE = 1.0


def _apply(fam: str, x: torch.Tensor, **kw) -> torch.Tensor:
    return apply_activation_family(
        x, activation_family=normalize_activation_family(fam), omega=_OMEGA,
        wire_scale=_WIRE, **kw,
    )


# ---- exact-form verification (REAL math, not a renamed sin) -----------------
def test_siren_is_exactly_sin_omega_x():
    """siren MUST be exactly torch.sin(omega*x) (the bit-identical control)."""
    x = torch.linspace(-3, 3, 257)
    assert torch.equal(_apply("siren", x), torch.sin(_OMEGA * x))


def test_gauss_is_exact_gaussian_bump():
    """gauss == exp(-(s*x)^2): a smooth bump peaking 1 at x=0, no oscillation."""
    x = torch.linspace(-3, 3, 257)
    out = _apply("gauss", x)
    assert torch.allclose(out, torch.exp(-torch.square(_WIRE * x)))
    # peak at 0 == 1, monotone-decreasing magnitude away from 0 (no ringing).
    assert torch.isclose(out[128], torch.tensor(1.0))
    assert (out >= 0).all() and out.max() <= 1.0 + 1e-6


def test_hosc_is_exact_tanh_beta_sin():
    """hosc == tanh(beta*sin(omega*x)) and beta sharpens toward a step train."""
    x = torch.linspace(-3, 3, 257)
    out = _apply("hosc", x, hosc_beta=4.0)
    assert torch.allclose(out, torch.tanh(4.0 * torch.sin(_OMEGA * x)))
    # larger beta => sharper (closer to +-1 square wave) => larger mean |value|.
    lo = _apply("hosc", x, hosc_beta=1.0).abs().mean()
    hi = _apply("hosc", x, hosc_beta=16.0).abs().mean()
    assert hi > lo, "hosc_beta did not sharpen the activation (no-op hyperparam!)"


def test_sinc_is_cardinal_sine_with_unit_peak():
    """sinc == sin(omega*x)/(omega*x) with the x=0 singularity patched to 1."""
    x = torch.linspace(-6, 6, 513)
    out = _apply("sinc", x)
    # match unnormalized sin(t)/t (== torch.sinc(t/pi)) elementwise where t != 0.
    t = _OMEGA * x
    expected = torch.sinc(t / math.pi)
    assert torch.allclose(out, expected, atol=1e-6)
    # sinc(0) == 1 (removable singularity), and it oscillates (sign changes exist).
    mid = out[256]
    assert torch.isclose(mid, torch.tensor(1.0), atol=1e-5)
    assert (out < 0).any(), "sinc should oscillate (have negative lobes)"


def test_rcgauss_is_finite_localized_and_unit_peak():
    """rcgauss (FLAIR band-localized) is finite everywhere (the RC singularities are
    patched), peaks ~1 at x=0, and DECAYS in the tails (Gaussian-enveloped)."""
    x = torch.linspace(-8, 8, 1025)
    out = _apply("rcgauss", x)
    assert torch.isfinite(out).all(), "rcgauss produced non-finite values (RC singularity!)"
    assert torch.isclose(out[512], torch.tensor(1.0), atol=1e-3)
    # tail magnitude << center (spatial localization by the Gaussian envelope).
    assert out[:50].abs().mean() < 0.05 * out[512].abs()


def test_finer_gauss_is_finer_times_gaussian():
    """custom #1 == sin(omega*(|x|+1)*x) * exp(-0.5*(s*x)^2): FINER carrier, Gauss envelope."""
    x = torch.linspace(-3, 3, 257)
    out = _apply("finer_gauss", x)
    finer = torch.sin(_OMEGA * (torch.abs(x) + 1.0) * x)
    window = torch.exp(-0.5 * torch.square(_WIRE * x))
    assert torch.allclose(out, finer * window)
    # it is NOT pure FINER (the envelope damps the tails) and NOT pure siren.
    assert not torch.equal(out, finer)
    assert not torch.equal(out, torch.sin(_OMEGA * x))


# ---- NO-FAKE: every non-siren fixed-form family genuinely differs from sin ----
@pytest.mark.parametrize(
    "fam", ("gauss", "hosc", "sinc", "rcgauss", "finer_gauss")
)
def test_fixed_form_family_differs_from_siren(fam: str):
    """Each new fixed-form family must produce DIFFERENT output than siren at the same
    input (FAILS if the branch degenerated to torch.sin)."""
    x = torch.linspace(-2.5, 2.5, 129)
    siren = _apply("siren", x)
    out = _apply(fam, x)
    assert out.shape == siren.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, siren, atol=1e-4), f"{fam} == siren (no-op!)"


@pytest.mark.parametrize("fam", ("wire", "gauss", "rcgauss", "finer_gauss"))
def test_wire_scale_actually_modulates(fam: str):
    """For families that consume wire_scale, two different scales => different outputs."""
    x = torch.linspace(-3, 3, 257)
    a = apply_activation_family(
        x, activation_family=normalize_activation_family(fam), omega=_OMEGA, wire_scale=0.5
    )
    b = apply_activation_family(
        x, activation_family=normalize_activation_family(fam), omega=_OMEGA, wire_scale=2.0
    )
    assert not torch.allclose(a, b, atol=1e-5), f"{fam} ignored wire_scale (no-op hyperparam!)"


def test_learnable_families_raise_in_stateless_apply():
    """step_basis / fkan must REFUSE the stateless path (they carry parameters)."""
    x = torch.randn(8)
    for fam in ("step_basis", "fkan"):
        with pytest.raises(ValueError, match="LEARNABLE"):
            _apply(fam, x)
        assert is_learnable_activation(fam)
    for fam in FIXED_FORM_ACTIVATION_IDS:
        assert not is_learnable_activation(fam)


# ---- LEARNABLE: step-basis (custom #2) --------------------------------------
def test_step_basis_forward_is_real_tanh_step_sum():
    """LearnableStepBasis == sum_k a_k*tanh(g_k*(x-c_k)), and the params are real."""
    m = LearnableStepBasis(num_steps=4, init_gain=1.0).eval()
    x = torch.linspace(-3, 3, 257)
    with torch.no_grad():
        out = m(x)
        expected = (m.a * torch.tanh(m.g * (x.unsqueeze(-1) - m.c))).sum(dim=-1)
    assert torch.allclose(out, expected)
    # 3K params (a, g, c each of length K).
    assert sum(p.numel() for p in m.parameters()) == 3 * 4


def test_step_basis_is_quasi_constant_between_steps_no_gibbs():
    """The key argmax-edge property: away from the step centers the activation is
    quasi-FLAT (tanh saturates), i.e. its derivative ~0 in the tails — NOT oscillating
    like a sinusoid. This is what lets it represent a step without Gibbs ringing."""
    m = LearnableStepBasis(num_steps=4, init_gain=2.0).eval()
    x = torch.linspace(-8, 8, 1601, requires_grad=True)
    out = m(x)
    g = torch.autograd.grad(out.sum(), x)[0]
    # far tails: |derivative| is tiny (flat plateau), vs near center it is large.
    tail = g[:80].abs().mean()
    center = g[760:840].abs().mean()
    assert tail < 0.05 * center, "step-basis tails are not flat (would ring like a sine)"


def test_step_basis_params_actually_move_output():
    """NO-FAKE: perturbing a_k, g_k, c_k each changes the output (no dead params)."""
    torch.manual_seed(0)
    m = LearnableStepBasis(num_steps=4).eval()
    x = torch.linspace(-3, 3, 129)
    with torch.no_grad():
        base = m(x).clone()
        m.a.add_(0.3)
        after_a = m(x).clone()
        m.a.sub_(0.3)
        m.g.add_(0.5)
        after_g = m(x).clone()
        m.g.sub_(0.5)
        m.c.add_(0.4)
        after_c = m(x).clone()
    assert not torch.equal(base, after_a), "a_k is a dead parameter"
    assert not torch.equal(base, after_g), "g_k is a dead parameter"
    assert not torch.equal(base, after_c), "c_k is a dead parameter"


def test_step_basis_k_changes_param_count():
    for k in (2, 4, 8):
        m = LearnableStepBasis(num_steps=k)
        assert sum(p.numel() for p in m.parameters()) == 3 * k


# ---- LEARNABLE: FKAN (custom #3) --------------------------------------------
def test_fkan_init_equals_sin_omega_x():
    """At init (a_1=1, rest 0) the FKAN activation == sin(omega*x) — the SIREN basis,
    so it inherits SIREN's spectral prior at epoch 0 (well-conditioned start)."""
    m = FourierKANActivation(num_harmonics=5, omega=1.0).eval()
    x = torch.linspace(-3, 3, 257)
    with torch.no_grad():
        out = m(x)
    assert torch.allclose(out, torch.sin(1.0 * x), atol=1e-6), "FKAN init != sin (bad prior)"


def test_fkan_forward_is_real_fourier_series():
    """FKAN == sum_k [a_k sin(k*omega*x) + b_k cos(k*omega*x)] with real params."""
    torch.manual_seed(1)
    m = FourierKANActivation(num_harmonics=4, omega=1.0).eval()
    with torch.no_grad():
        m.a.copy_(torch.tensor([1.0, 0.5, -0.3, 0.2]))
        m.b.copy_(torch.tensor([0.0, 0.4, 0.1, -0.2]))
    x = torch.linspace(-3, 3, 257)
    with torch.no_grad():
        out = m(x)
        k = torch.arange(1, 5, dtype=torch.float32)
        phase = (k * x.unsqueeze(-1))
        expected = (m.a * torch.sin(phase) + m.b * torch.cos(phase)).sum(dim=-1)
    assert torch.allclose(out, expected)
    assert sum(p.numel() for p in m.parameters()) == 2 * 4  # a, b each length K


def test_fkan_higher_harmonics_actually_engage():
    """NO-FAKE: setting a higher harmonic coefficient changes the output (the harmonic
    is really synthesized, not ignored)."""
    m = FourierKANActivation(num_harmonics=5, omega=1.0).eval()
    x = torch.linspace(-3, 3, 257)
    with torch.no_grad():
        base = m(x).clone()
        m.a[3] = 0.7  # engage the 4th harmonic
        after = m(x).clone()
        m.a[3] = 0.0
        m.b[4] = 0.6  # engage the 5th harmonic cosine
        after_b = m(x).clone()
    assert not torch.equal(base, after), "high harmonic a_k ignored (dead)"
    assert not torch.equal(base, after_b), "high harmonic b_k ignored (dead)"


def test_fkan_omega_matters():
    """Different base omega => different harmonic spacing => different output."""
    m1 = FourierKANActivation(num_harmonics=4, omega=1.0).eval()
    m2 = FourierKANActivation(num_harmonics=4, omega=3.0).eval()
    with torch.no_grad():
        # engage a few harmonics identically so the only difference is omega.
        for m in (m1, m2):
            m.a.copy_(torch.tensor([1.0, 0.5, 0.3, 0.2]))
    x = torch.linspace(-3, 3, 257)
    with torch.no_grad():
        assert not torch.allclose(m1(x), m2(x), atol=1e-5), "FKAN ignored omega"


def test_make_learnable_activation_dispatch():
    sb = make_learnable_activation("step_basis", step_basis_k=6)
    fk = make_learnable_activation("fkan", fkan_k=7, omega=2.0)
    assert isinstance(sb, LearnableStepBasis) and sb.num_steps == 6
    assert isinstance(fk, FourierKANActivation) and fk.num_harmonics == 7 and fk.omega == 2.0
    with pytest.raises(ValueError, match="not learnable"):
        make_learnable_activation("siren")


def test_learnable_modules_grad_flows():
    """Gradients flow to every learnable parameter (trainable, not frozen)."""
    for m in (LearnableStepBasis(num_steps=3), FourierKANActivation(num_harmonics=3)):
        x = torch.linspace(-2, 2, 64)
        out = m(x)
        out.pow(2).mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name} no grad"


def test_invalid_learnable_sizes_fail_closed():
    with pytest.raises(ValueError, match="num_steps must be positive"):
        LearnableStepBasis(num_steps=0)
    with pytest.raises(ValueError, match="num_harmonics must be positive"):
        FourierKANActivation(num_harmonics=0)


def test_normalize_aliases_for_new_families():
    """New aliases route to the canonical id (so --activation gaussian etc. work)."""
    assert normalize_activation_family("gaussian") == "gauss"
    assert normalize_activation_family("flair") == "rcgauss"
    assert normalize_activation_family("rc-gauss") == "rcgauss"
    assert normalize_activation_family("cardinal_sine") == "sinc"
    assert normalize_activation_family("ada_hosc") == "hosc"
    assert normalize_activation_family("sinekan") == "fkan"
    assert normalize_activation_family("soft-step") == "step_basis"
    assert normalize_activation_family("finer-gauss") == "finer_gauss"
