# SPDX-License-Identifier: MIT
"""NO-FAKE tests for Lever C — per-dim pose Mahalanobis/AIL weighting.

Verify the weighted MSE actually re-allocates per-dim pull (would FAIL if ``weighted_pose_mse`` ignored its
weights) AND that the uniform/None defaults are byte-identical to the plain MSE.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tac.torch_vehicle.pose_dim_weights import (
    N_SCORED_POSE_DIMS,
    measure_pose_dim_weights_from_targets,
    normalise_pose_dim_weights,
    weighted_pose_mse,
    zero_low_sensitivity_dims,
)


def test_normalise_uniform_is_identity() -> None:
    assert normalise_pose_dim_weights((1, 1, 1, 1, 1, 1)) == (1.0,) * 6


def test_normalise_scales_to_mean_one() -> None:
    w = normalise_pose_dim_weights((2, 2, 2, 2, 2, 2))
    assert w == (1.0,) * 6  # mean 1.0
    w2 = normalise_pose_dim_weights((1, 2, 3, 4, 5, 6))
    assert sum(w2) / 6 == pytest.approx(1.0)


def test_normalise_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        normalise_pose_dim_weights((1, 1, 1))  # wrong length
    with pytest.raises(ValueError):
        normalise_pose_dim_weights((1, -1, 1, 1, 1, 1))  # negative
    with pytest.raises(ValueError):
        normalise_pose_dim_weights((0, 0, 0, 0, 0, 0))  # all-zero


def test_uniform_weighted_mse_equals_plain_mse() -> None:
    torch.manual_seed(0)
    pred = torch.randn(8, 6)
    tgt = torch.randn(8, 6)
    plain = F.mse_loss(pred, tgt)
    assert weighted_pose_mse(pred, tgt, None) == pytest.approx(float(plain), abs=1e-7)
    assert weighted_pose_mse(pred, tgt, (1, 1, 1, 1, 1, 1)) == pytest.approx(float(plain), abs=1e-7)


def test_nonuniform_weight_tilts_loss_toward_weighted_dim() -> None:
    # error ONLY in dim 0 → up-weighting dim 0 INCREASES the loss; down-weighting DECREASES it.
    pred = torch.zeros(4, 6)
    tgt = torch.zeros(4, 6)
    tgt[:, 0] = 1.0  # squared error 1.0 in dim 0 only
    base = float(weighted_pose_mse(pred, tgt, None))  # = (1+0+0+0+0+0)/6 = 1/6
    assert base == pytest.approx(1.0 / 6.0)
    up = float(weighted_pose_mse(pred, tgt, (6, 0, 0, 0, 0, 0)))  # all weight on dim 0
    # normalised (6,0,..)→ mean 1 → (6,0,0,0,0,0); sum w_k·sq_k /6 = 6*1/6 = 1.0
    assert up == pytest.approx(1.0)
    down = float(weighted_pose_mse(pred, tgt, (0.1, 2, 2, 2, 2, 2)))
    assert down < base  # dim 0 (the only error) is down-weighted → loss drops


def test_measure_weights_from_targets_inv_var() -> None:
    # construct targets: dim 0 has HIGH variance, dim 5 ~0 variance.
    torch.manual_seed(1)
    t = torch.zeros(100, 6)
    t[:, 0] = torch.randn(100) * 5.0  # high var
    t[:, 1] = torch.randn(100) * 1.0
    # dims 2..5 ~constant (low var)
    w = measure_pose_dim_weights_from_targets(t, mode="inv_var", floor=0.0)
    assert len(w) == N_SCORED_POSE_DIMS
    assert sum(w) / 6 == pytest.approx(1.0)
    # inverse-variance: the HIGH-variance dim 0 gets the SMALLEST weight.
    assert w[0] < w[1]
    assert w[0] < w[5]


def test_measure_weights_uniform_mode_is_uniform() -> None:
    t = torch.randn(50, 6)
    assert measure_pose_dim_weights_from_targets(t, mode="uniform") == (1.0,) * 6


def test_measure_weights_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        measure_pose_dim_weights_from_targets(torch.randn(10, 3))  # < 6 dims


def test_zero_low_sensitivity_dims_drops_dead_dim() -> None:
    # dim 3 has ~0 Jacobian row energy → dropped; survivors renormalise to mean 1.0 over 5 dims.
    w = (1, 1, 1, 1, 1, 1)
    energy = [1.0, 1.0, 1.0, 1e-9, 1.0, 1.0]
    out, zeroed = zero_low_sensitivity_dims(w, energy, rel_tol=1e-3)
    assert zeroed == [3]
    assert out[3] == 0.0
    surv = [out[k] for k in range(6) if k != 3]
    assert sum(surv) / 5 == pytest.approx(1.0)


def test_zero_low_sensitivity_no_drop_when_all_sensitive() -> None:
    out, zeroed = zero_low_sensitivity_dims((1,) * 6, [1.0] * 6, rel_tol=1e-3)
    assert zeroed == []
    assert out == (1.0,) * 6


def test_zero_low_sensitivity_all_zero_energy_keeps_uniform() -> None:
    # all energies 0 → mx==0 → NO dim is below rel_tol*mx → nothing dropped → uniform survivors.
    out, zeroed = zero_low_sensitivity_dims((1,) * 6, [0.0] * 6, rel_tol=1e-3)
    assert zeroed == []
    assert out == (1.0,) * 6


def test_zero_low_sensitivity_drops_multiple_dead_dims() -> None:
    # two dead dims (1 and 4) dropped; survivors (0,2,3,5) renormalise to mean 1.0 over 4 dims.
    energy = [1.0, 1e-9, 1.0, 1.0, 1e-9, 1.0]
    out, zeroed = zero_low_sensitivity_dims((1,) * 6, energy, rel_tol=1e-3)
    assert zeroed == [1, 4]
    assert out[1] == 0.0 and out[4] == 0.0
    surv = [out[k] for k in (0, 2, 3, 5)]
    assert sum(surv) / 4 == pytest.approx(1.0)


def test_zero_low_sensitivity_survivors_zero_weight_falls_back_uniform() -> None:
    # dim 0 has energy (survives) but its weight is 0; the lone survivor falls back to uniform-over-survivors.
    energy = [1.0, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9]  # only dim 0 survives
    out, zeroed = zero_low_sensitivity_dims((0.0, 1, 1, 1, 1, 1), energy, rel_tol=1e-3)
    assert zeroed == [1, 2, 3, 4, 5]
    assert out[0] == pytest.approx(1.0)  # fell back to uniform over the single survivor


def test_grad_flows_through_weighted_mse() -> None:
    pred = torch.zeros(4, 6, requires_grad=True)
    tgt = torch.ones(4, 6)
    loss = weighted_pose_mse(pred, tgt, (1, 2, 3, 4, 5, 6))
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    # the gradient on each dim scales with its (normalised) weight.
    g = pred.grad.abs().mean(dim=0)
    assert g[5] > g[0]  # dim 5 has the largest weight → largest gradient
