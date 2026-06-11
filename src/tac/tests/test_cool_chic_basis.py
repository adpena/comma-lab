# SPDX-License-Identifier: MIT
"""Tests for the Cool-Chic non-conv basis (lane_cool_chic_score_aware_basis).

NO-FAKE discipline (Slot EEE classes): tests verify ACTUAL behavior, not
constants. The parity gate feeds NON-zero grids+weights (the grid-PE fake-parity
lesson: a zero-init tensor renders identically on every backend, so zeros pass
any parity check vacuously).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.residual_basis.cool_chic_carrier import CoolChicGridSpec, CoolChicPairCarrier
from tac.residual_basis.cool_chic_synthesis_numpy import (
    SynthesisWeightsNumpy,
    bilinear_upsample_numpy,
    laplacian_rate_bits_numpy,
    synthesize_rgb_numpy,
    total_synthesis_param_count,
)


def _real_grids(rng: np.random.Generator, c: int, n_grids: int, h: int, w: int):
    """NON-zero multiresolution grids (real inputs for the parity gate)."""
    return [
        rng.standard_normal((c, max(h >> lv, 1), max(w >> lv, 1))).astype(np.float64)
        for lv in range(n_grids)
    ]


def test_bilinear_upsample_matches_torch_align_corners_false():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((2, 5, 7)).astype(np.float64)
    ref = bilinear_upsample_numpy(g, 11, 13)
    t = torch.nn.functional.interpolate(
        torch.from_numpy(g).unsqueeze(0), size=(11, 13),
        mode="bilinear", align_corners=False,
    ).squeeze(0).numpy()
    # numpy ref and torch must agree (the portability contract for upsample).
    assert np.allclose(ref, t, atol=1e-5), float(np.abs(ref - t).max())


def test_laplacian_rate_is_nonnegative_and_grows_with_deviation():
    z = np.array([0.0, 1.0, 5.0])
    mu = np.zeros(3)
    log_scale = np.zeros(3)  # scale = 1
    bits = laplacian_rate_bits_numpy(z, mu, log_scale, quant_step=1.0)
    assert (bits >= 0).all()
    # larger |z-mu| => more bits (NOT a constant).
    assert bits[2] > bits[1] > bits[0]


def test_numpy_synth_output_in_unit_range_and_depends_on_weights():
    rng = np.random.default_rng(1)
    n_grids, c = 3, 2
    grids = _real_grids(rng, c, n_grids, 8, 8)
    c_in = c * n_grids
    w = SynthesisWeightsNumpy(
        w1=rng.standard_normal((6, c_in)) * 0.5,
        b1=rng.standard_normal(6) * 0.1,
        w2=rng.standard_normal((3, 6)) * 0.5,
        b2=rng.standard_normal(3) * 0.1,
    )
    out = synthesize_rgb_numpy(grids, w, 8, 8)
    assert out.shape == (3, 8, 8)
    assert (out >= 0).all() and (out <= 1).all()
    # Perturbing a weight MUST change the output (not a constant render).
    w2 = SynthesisWeightsNumpy(w1=w.w1 + 1.0, b1=w.b1, w2=w.w2, b2=w.b2)
    out2 = synthesize_rgb_numpy(grids, w2, 8, 8)
    assert not np.allclose(out, out2)


def test_mlx_synth_matches_numpy_reference_on_real_inputs():
    """PORTABILITY CONTRACT: MLX fast path == numpy reference on NON-zero inputs."""
    pytest.importorskip("mlx.core")
    from tac.residual_basis.cool_chic_synthesis_mlx import synthesize_rgb_mlx

    rng = np.random.default_rng(2)
    n_grids, c = 4, 2
    grids = _real_grids(rng, c, n_grids, 12, 16)
    c_in = c * n_grids
    w1 = rng.standard_normal((10, c_in)) * 0.4
    b1 = rng.standard_normal(10) * 0.2
    w2 = rng.standard_normal((3, 10)) * 0.4
    b2 = rng.standard_normal(3) * 0.2
    ref = synthesize_rgb_numpy(
        grids, SynthesisWeightsNumpy(w1=w1, b1=b1, w2=w2, b2=b2), 24, 32
    )
    got = synthesize_rgb_mlx(grids, w1, b1, w2, b2, 24, 32, on_cpu=True)
    # fp32 MLX vs fp64 numpy: tolerance must catch a REAL divergence but allow
    # float32 rounding. Max abs error on [0,1] sigmoid output.
    err = float(np.abs(ref - got).max())
    assert err < 2e-3, f"MLX<->numpy synth parity FAILED: max abs err {err}"
    # And it must NOT be vacuously equal because everything is ~0.5.
    assert ref.std() > 1e-3, "reference output is ~constant; parity is vacuous"


def test_carrier_reconstruct_pair_shape_and_frame1_differs_from_frame0():
    spec = CoolChicGridSpec(base_h=12, base_w=16, n_grids=3, channels_per_grid=2)
    carrier = CoolChicPairCarrier(n_pairs=4, spec=spec, synth_hidden=8, out_hw=(48, 64))
    idx = torch.arange(3)
    rgb0, rgb1 = carrier.reconstruct_pair(idx)
    assert rgb0.shape == (3, 3, 48, 64)
    assert rgb1.shape == (3, 3, 48, 64)
    assert float(rgb0.min()) >= 0.0 and float(rgb0.max()) <= 255.0
    # frame1 carries the per-pair delta => must differ from frame0 (NOT a copy).
    assert not torch.allclose(rgb0, rgb1)
    # different pairs => different frame1 (the per-pair delta is real).
    assert not torch.allclose(rgb1[0], rgb1[1])


def test_carrier_torch_synth_matches_numpy_reference():
    """The torch carrier synthesis == numpy reference (cross-backend parity)."""
    torch.manual_seed(0)
    spec = CoolChicGridSpec(base_h=8, base_w=8, n_grids=3, channels_per_grid=2)
    carrier = CoolChicPairCarrier(n_pairs=2, spec=spec, synth_hidden=6, out_hw=(16, 16))
    feat0 = carrier._upsample_concat(None)
    torch_out = carrier._synth(feat0).detach().numpy()
    grids_np = [g.detach().numpy().astype(np.float64) for g in carrier.latent_grids]
    w = SynthesisWeightsNumpy(
        w1=carrier.w1.detach().numpy().astype(np.float64),
        b1=carrier.b1.detach().numpy().astype(np.float64),
        w2=carrier.w2.detach().numpy().astype(np.float64),
        b2=carrier.b2.detach().numpy().astype(np.float64),
    )
    np_out = synthesize_rgb_numpy(grids_np, w, 16, 16)
    err = float(np.abs(torch_out - np_out).max())
    assert err < 1e-4, f"torch<->numpy synth parity FAILED: {err}"


def test_charged_bytes_scale_with_grid_resolution():
    """The latent-byte term MUST grow with grid resolution (the basis lever)."""
    small = CoolChicPairCarrier(
        n_pairs=2, spec=CoolChicGridSpec(8, 8, 3, 2), synth_hidden=8, out_hw=(32, 32)
    )
    big = CoolChicPairCarrier(
        n_pairs=2, spec=CoolChicGridSpec(32, 32, 4, 2), synth_hidden=8, out_hw=(32, 32)
    )
    cs, cb = small.charged_bytes(), big.charged_bytes()
    assert cb["latent_count"] > cs["latent_count"]
    assert cb["latent_bytes"] > cs["latent_bytes"]
    # weight bytes are tiny + fixed-ish (the non-conv-basis claim).
    assert cs["weight_param_count"] < 1000
    assert cb["weight_param_count"] < 1000


def test_latent_rate_bytes_is_real_arm_estimate_not_constant():
    """The ARM rate MUST depend on the latent values (not a fixed count)."""
    torch.manual_seed(0)
    spec = CoolChicGridSpec(16, 16, 3, 2)
    carrier = CoolChicPairCarrier(n_pairs=2, spec=spec, synth_hidden=8, out_hw=(32, 32))
    r0 = carrier.latent_rate_bytes()
    # zero out the grids => Laplacian rate at mu~0 should change materially.
    with torch.no_grad():
        for g in carrier.latent_grids:
            g.mul_(5.0)  # inflate magnitudes => higher rate
    r1 = carrier.latent_rate_bytes()
    assert r1 != r0, "ARM rate did not respond to latent magnitude (fake rate)"
    assert r0 > 0 and r1 > 0


def test_synth_param_count_helper():
    assert total_synthesis_param_count(8, 12) == 8 * 12 + 12 + 12 * 3 + 3


def test_carrier_plugs_into_score_aware_trainer_interface():
    """The carrier exposes exactly what ScoreAwareTrainer consumes."""
    spec = CoolChicGridSpec(12, 16, 3, 2)
    carrier = CoolChicPairCarrier(n_pairs=4, spec=spec, synth_hidden=8, out_hw=(48, 64))
    # named_parameters must include 'latent' so the trainer's latent-LR group fires.
    names = [n for n, _ in carrier.named_parameters()]
    assert any("latent" in n.lower() for n in names)
    # reconstruct_pair(idx) -> (B,3,H,W),(B,3,H,W) is the trainer contract.
    out = carrier.forward(torch.arange(2))
    assert out.shape == (2, 2, 3, 48, 64)


def test_gradient_flows_to_latents_and_synth():
    """A loss on the render MUST produce gradients on grids AND synth weights."""
    spec = CoolChicGridSpec(8, 8, 3, 2)
    carrier = CoolChicPairCarrier(n_pairs=2, spec=spec, synth_hidden=6, out_hw=(16, 16))
    rgb0, rgb1 = carrier.reconstruct_pair(torch.arange(2))
    loss = (rgb1 - 128.0).pow(2).mean() + rgb0.mean()
    loss.backward()
    assert carrier.latent_grids[0].grad is not None
    assert carrier.w1.grad is not None
    assert carrier.frame1_delta.grad is not None
    assert float(carrier.latent_grids[0].grad.abs().sum()) > 0
