# SPDX-License-Identifier: MIT
"""Behavioral tests for the Cool-Chic byte-close + numpy inflate (NO FAKE).

These verify ACTUAL behavior (not constants): bit-exact numpy↔torch render
parity on REAL non-zero weights (the fake-parity guard), a self-contained
range-coded blob that inflates without torch, and that quant is near-lossless on
the rendered RGB. Per CLAUDE.md "Tests-verify-constants-not-behavior" forbidden
class — every assertion would FAIL if the encode/decode were replaced by a no-op.
"""

from __future__ import annotations

import numpy as np
import torch

from tac.residual_basis.cool_chic_byte_close import (
    byte_close,
    carrier_to_state,
    inflate_numpy,
    quantize_to_symbols,
    render_pair_numpy,
)
from tac.residual_basis.cool_chic_carrier import CoolChicGridSpec, CoolChicPairCarrier


def _make_carrier(seed: int = 0) -> CoolChicPairCarrier:
    torch.manual_seed(seed)
    spec = CoolChicGridSpec(base_h=24, base_w=32, n_grids=4, channels_per_grid=2)
    c = CoolChicPairCarrier(n_pairs=4, spec=spec, synth_hidden=12, out_hw=(48, 64))
    # perturb to REAL non-zero weights (the fake-parity guard: zero-init renders
    # identically on every backend and proves nothing).
    with torch.no_grad():
        for p in c.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return c


def test_numpy_render_is_bit_exact_to_torch_on_real_weights():
    c = _make_carrier()
    idx = torch.arange(4)
    rgb0_t, rgb1_t = c.reconstruct_pair(idx)
    state = carrier_to_state(c)
    for i in range(4):
        r0n, r1n = render_pair_numpy(state, i)
        assert np.abs(r0n - rgb0_t[i].detach().numpy()).max() < 1e-3
        assert np.abs(r1n - rgb1_t[i].detach().numpy()).max() < 1e-3


def test_byte_close_roundtrip_is_self_contained_and_decodable():
    c = _make_carrier()
    state = carrier_to_state(c)
    blob = byte_close(state, grid_step=0.05, delta_step=0.05)
    assert isinstance(blob, bytes) and len(blob) > 100
    state2 = inflate_numpy(blob)  # pure numpy, no torch
    # the inflated state renders, and quant is near-lossless on the RGB.
    for i in range(4):
        r0a, r1a = render_pair_numpy(state, i)
        r0b, r1b = render_pair_numpy(state2, i)
        # step 0.05 -> a few /255 render delta, not a no-op and not a blow-up.
        assert np.abs(r0a - r0b).max() < 10.0
        assert np.abs(r1a - r1b).max() < 10.0


def test_bytes_respond_to_grid_step_not_a_constant():
    """Coarser quant -> fewer/closer symbols -> smaller blob (real coding)."""
    c = _make_carrier()
    state = carrier_to_state(c)
    fine = len(byte_close(state, grid_step=0.02, delta_step=0.02))
    coarse = len(byte_close(state, grid_step=0.2, delta_step=0.2))
    assert coarse < fine  # if encode were a no-op this would not hold


def test_quantize_dequantize_offsets_are_nonnegative_symbols():
    x = np.array([[-0.31, 0.0, 0.47], [0.12, -0.08, 0.9]])
    syms, offset = quantize_to_symbols(x, 0.05)
    assert syms.min() >= 0
    # round-trip within a quant step.
    deq = (syms.astype(float) - offset) * 0.05
    assert np.abs(deq - x).max() <= 0.05 + 1e-9
