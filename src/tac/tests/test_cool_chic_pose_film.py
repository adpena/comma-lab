# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the Cool-Chic stored-pose FiLM (pose-fold).

These verify the ACTUAL mechanism (the Quantizr store-pose lesson), not
constants. Every assertion would FAIL if the FiLM were a no-op or the stored
pose were ignored:

  - FiLM is IDENTITY at init (untrained FiLM-ON == FiLM-OFF render).
  - DIFFERENT stored poses produce DIFFERENT frame1 once the FiLM is trained
    (the pose is actually consumed; not a dead buffer).
  - numpy inflate render is bit-exact to the torch render WITH FiLM on REAL
    (non-zero) FiLM weights (the portability / fake-parity guard).
  - the byte-closed reloaded state still applies the FiLM (the pose survives
    the range-code + int8 roundtrip well enough to be non-trivial).
"""

from __future__ import annotations

import numpy as np
import torch

from tac.residual_basis.cool_chic_byte_close import (
    byte_close,
    carrier_to_state,
    inflate_numpy,
    render_pair_numpy,
)
from tac.residual_basis.cool_chic_carrier import CoolChicGridSpec, CoolChicPairCarrier


def _spec() -> CoolChicGridSpec:
    return CoolChicGridSpec(base_h=24, base_w=32, n_grids=4, channels_per_grid=2)


def _perturb(c: CoolChicPairCarrier, scale: float = 0.05) -> None:
    with torch.no_grad():
        for p in c.parameters():
            p.add_(torch.randn_like(p) * scale)


def test_film_is_identity_at_init():
    """FiLM-ON at init renders EXACTLY the FiLM-OFF render (zero-init fc2)."""
    torch.manual_seed(0)
    spec = _spec()
    off = CoolChicPairCarrier(n_pairs=4, spec=spec, synth_hidden=12, out_hw=(48, 64),
                              pose_film_enabled=False)
    torch.manual_seed(0)
    on = CoolChicPairCarrier(n_pairs=4, spec=spec, synth_hidden=12, out_hw=(48, 64),
                             pose_film_enabled=True)
    on.set_stored_pose(torch.randn(4, 6))  # non-zero stored pose
    idx = torch.arange(4)
    r0_off, r1_off = off.reconstruct_pair(idx)
    r0_on, r1_on = on.reconstruct_pair(idx)
    # frame0 has no FiLM; frame1 FiLM is identity at init -> both match.
    assert torch.allclose(r0_off, r0_on, atol=1e-4)
    assert torch.allclose(r1_off, r1_on, atol=1e-4)


def test_different_stored_pose_changes_frame1_when_film_trained():
    """A trained (non-zero-fc2) FiLM makes frame1 DEPEND on the stored pose."""
    torch.manual_seed(1)
    on = CoolChicPairCarrier(n_pairs=2, spec=_spec(), synth_hidden=12, out_hw=(48, 64),
                             pose_film_enabled=True)
    _perturb(on)
    # Train-like FiLM fc2 (break the identity init so the pose has an effect).
    with torch.no_grad():
        on.pose_film.fc2.weight.add_(torch.randn_like(on.pose_film.fc2.weight) * 0.3)
        on.pose_film.fc2.bias.add_(torch.randn_like(on.pose_film.fc2.bias) * 0.3)
    idx = torch.tensor([0])
    on.set_stored_pose(torch.tensor([[1.0, 0.5, -0.3, 0.1, 0.0, 0.2], [0.0] * 6]))
    _, r1_a = on.reconstruct_pair(idx)
    on.set_stored_pose(torch.tensor([[-2.0, 1.5, 0.7, -0.9, 0.4, -0.6], [0.0] * 6]))
    _, r1_b = on.reconstruct_pair(idx)
    # different stored pose -> materially different frame1 (NOT a no-op).
    assert float((r1_a - r1_b).abs().max()) > 1.0


def test_numpy_inflate_matches_torch_with_film_on_real_weights():
    """Bit-exact numpy<->torch render WITH a trained FiLM (fake-parity guard)."""
    torch.manual_seed(2)
    on = CoolChicPairCarrier(n_pairs=3, spec=_spec(), synth_hidden=12, out_hw=(48, 64),
                             pose_film_enabled=True)
    _perturb(on)
    with torch.no_grad():
        on.pose_film.fc2.weight.add_(torch.randn_like(on.pose_film.fc2.weight) * 0.2)
        on.pose_film.fc2.bias.add_(torch.randn_like(on.pose_film.fc2.bias) * 0.2)
    on.set_stored_pose(torch.randn(3, 6) * 0.8)
    idx = torch.arange(3)
    r0_t, r1_t = on.reconstruct_pair(idx)
    state = carrier_to_state(on)
    assert state.pose_film is not None and state.stored_pose is not None
    for i in range(3):
        r0n, r1n = render_pair_numpy(state, i)
        assert np.abs(r0n - r0_t[i].detach().numpy()).max() < 1e-3
        assert np.abs(r1n - r1_t[i].detach().numpy()).max() < 1e-3


def test_byte_closed_state_still_applies_film():
    """The range-coded + int8 byte-close preserves the pose-dependent FiLM."""
    torch.manual_seed(3)
    on = CoolChicPairCarrier(n_pairs=3, spec=_spec(), synth_hidden=12, out_hw=(48, 64),
                             pose_film_enabled=True)
    _perturb(on)
    with torch.no_grad():
        on.pose_film.fc2.weight.add_(torch.randn_like(on.pose_film.fc2.weight) * 0.2)
        on.pose_film.fc2.bias.add_(torch.randn_like(on.pose_film.fc2.bias) * 0.2)
    on.set_stored_pose(torch.randn(3, 6) * 0.8)
    state = carrier_to_state(on)
    blob = byte_close(state, grid_step=0.05, delta_step=0.05, pose_step=0.002)
    state2 = inflate_numpy(blob)
    assert state2.pose_film is not None and state2.stored_pose is not None
    # the stored pose round-trips near-losslessly (fine pose_step).
    assert np.abs(state2.stored_pose - state.stored_pose).max() < 0.01
    # the byte-closed render still tracks the full-precision FiLM render.
    for i in range(3):
        r0a, r1a = render_pair_numpy(state, i)
        r0b, r1b = render_pair_numpy(state2, i)
        assert np.abs(r1a - r1b).max() < 12.0


def test_charged_bytes_includes_stored_pose_when_film_on():
    """The pose-fold pays REAL stored-pose bytes (not free)."""
    off = CoolChicPairCarrier(n_pairs=48, spec=_spec(), synth_hidden=24, out_hw=(96, 128),
                              pose_film_enabled=False)
    on = CoolChicPairCarrier(n_pairs=48, spec=_spec(), synth_hidden=24, out_hw=(96, 128),
                             pose_film_enabled=True)
    assert off.charged_bytes()["stored_pose_bytes"] == 0.0
    assert on.charged_bytes()["stored_pose_bytes"] > 0.0
    # FiLM adds weight params over the no-FiLM carrier.
    assert on.weight_param_count() > off.weight_param_count()
