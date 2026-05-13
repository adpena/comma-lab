from __future__ import annotations

import numpy as np
import torch

from tac.optimization.zen_state_frontier import (
    InformationGeometricLangevinOptimizer,
    brenier_quantile_quantize_1d,
    mps_decompose,
    mps_reconstruct,
    onsager_importance_weights,
    sinkhorn_transport_plan,
    tropical_lora_forward,
    wasserstein_barycenter_diagonal_gaussians,
)


def test_onsager_importance_weights_are_positive_normalized_and_sensitive() -> None:
    weights = onsager_importance_weights([0.0, 2.0, 8.0], floor=0.5)

    assert np.all(weights > 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[2] > weights[1] > weights[0]


def test_wasserstein_barycenter_diagonal_gaussians_uses_std_not_variance_mean() -> None:
    mean, var = wasserstein_barycenter_diagonal_gaussians(
        means=[[0.0, 2.0], [2.0, 4.0]],
        variances=[[1.0, 4.0], [9.0, 16.0]],
        weights=[0.25, 0.75],
    )

    np.testing.assert_allclose(mean, [1.5, 3.5])
    # std* = .25*[1,2] + .75*[3,4] = [2.5,3.5].
    np.testing.assert_allclose(var, [6.25, 12.25])


def test_brenier_quantile_quantize_1d_is_monotone_and_uses_bin_means() -> None:
    values = np.array([4.0, 0.0, 2.0, 8.0])
    quantized, levels = brenier_quantile_quantize_1d(values, n_levels=2)

    np.testing.assert_allclose(levels, [1.0, 6.0])
    np.testing.assert_allclose(quantized, [6.0, 1.0, 1.0, 6.0])
    sorted_q = quantized[np.argsort(values)]
    assert np.all(sorted_q[:-1] <= sorted_q[1:])


def test_sinkhorn_transport_plan_matches_marginals() -> None:
    plan = sinkhorn_transport_plan(
        [[0.0, 2.0], [2.0, 0.0]],
        source_weights=[0.4, 0.6],
        target_weights=[0.5, 0.5],
        epsilon=0.5,
        n_iters=500,
    )

    np.testing.assert_allclose(plan.sum(axis=1), [0.4, 0.6], atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), [0.5, 0.5], atol=1e-6)
    assert plan[0, 0] > plan[0, 1]
    assert plan[1, 1] > plan[1, 0]


def test_mps_decompose_reconstructs_exact_when_rank_unbounded() -> None:
    tensor = np.arange(24.0).reshape(2, 3, 4)
    train = mps_decompose(tensor)
    recon = mps_reconstruct(train)

    np.testing.assert_allclose(recon, tensor, atol=1e-10)
    assert train.ranks[0] == 1
    assert train.ranks[-1] == 1


def test_mps_decompose_obeys_rank_cap() -> None:
    tensor = np.arange(32.0).reshape(2, 4, 4)
    train = mps_decompose(tensor, max_rank=1)

    assert max(train.ranks) == 1
    assert mps_reconstruct(train).shape == tensor.shape


def test_tropical_lora_forward_exact_and_smoothed() -> None:
    base = torch.tensor([1.0, 2.0, -1.0])
    residual_a = torch.tensor([0.5, -10.0, 1.0])
    residual_b = torch.tensor([-2.0, 0.25, 0.0])

    exact = tropical_lora_forward(base, [residual_a, residual_b])
    torch.testing.assert_close(exact, torch.tensor([1.5, 2.25, 0.0]))
    smooth = tropical_lora_forward(base, [residual_a, residual_b], temperature=0.5)
    assert torch.all(smooth >= exact)


def test_information_geometric_langevin_preconditions_by_fisher_diag() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    optimizer = InformationGeometricLangevinOptimizer(
        [param],
        lr=0.1,
        T_init=0.0,
        T_final=0.0,
        n_steps=1,
        fisher_diag=[torch.tensor([4.0, 1.0])],
        fisher_damping=1e-9,
    )
    param.grad = torch.tensor([4.0, 1.0])

    optimizer.step()

    # Natural-gradient step: theta -= lr * grad / fisher.
    torch.testing.assert_close(param.detach(), torch.tensor([0.9, 0.9]), atol=1e-6, rtol=0)
