# SPDX-License-Identifier: MIT
"""Focused tests for canonical frontier composition primitives."""

from __future__ import annotations

import pytest
import torch

from tac.composition.frontier_primitives import (
    CompositionPrimitiveError,
    DiagonalGaussian,
    bregman_barycenter,
    build_mera_hierarchy_metadata,
    checkpoint_diagonal_gaussian_barycenter,
    normalize_weights,
    sinkhorn_transport_plan,
    wasserstein_diagonal_gaussian_barycenter,
)


def test_normalize_weights_uniform_and_rejects_bad_inputs() -> None:
    assert normalize_weights(None, 4) == (0.25, 0.25, 0.25, 0.25)
    assert normalize_weights((2.0, 1.0), 2) == pytest.approx((2 / 3, 1 / 3))
    with pytest.raises(CompositionPrimitiveError):
        normalize_weights((1.0,), 2)
    with pytest.raises(CompositionPrimitiveError):
        normalize_weights((-1.0, 2.0), 2)
    with pytest.raises(CompositionPrimitiveError):
        normalize_weights((0.0, 0.0), 2)


def test_wasserstein_diagonal_gaussian_barycenter_matches_w2_formula() -> None:
    g0 = DiagonalGaussian(
        mean=torch.tensor([0.0, 2.0]),
        variance=torch.tensor([1.0, 9.0]),
    )
    g1 = DiagonalGaussian(
        mean=torch.tensor([4.0, 6.0]),
        variance=torch.tensor([9.0, 25.0]),
    )
    out = wasserstein_diagonal_gaussian_barycenter((g0, g1), (0.25, 0.75))
    assert torch.allclose(out.mean, torch.tensor([3.0, 5.0]))
    # W2 diagonal barycenter averages standard deviations, then squares.
    expected_std = 0.25 * torch.tensor([1.0, 3.0]) + 0.75 * torch.tensor([3.0, 5.0])
    assert torch.allclose(out.variance, expected_std.square())


def test_wasserstein_barycenter_is_group_associative_for_diagonal_case() -> None:
    gaussians = tuple(
        DiagonalGaussian(
            mean=torch.tensor([float(i)]),
            variance=torch.tensor([float((i + 1) ** 2)]),
        )
        for i in range(3)
    )
    direct = wasserstein_diagonal_gaussian_barycenter(gaussians, (0.2, 0.3, 0.5))
    first_group = wasserstein_diagonal_gaussian_barycenter(gaussians[:2], (0.4, 0.6))
    grouped = wasserstein_diagonal_gaussian_barycenter(
        (first_group, gaussians[2]),
        (0.5, 0.5),
    )
    assert torch.allclose(grouped.mean, direct.mean)
    assert torch.allclose(grouped.variance, direct.variance)


def test_checkpoint_barycenter_validates_key_and_shape_contracts() -> None:
    c0 = {"a": torch.tensor([1.0, 3.0]), "b": torch.ones(2, 2)}
    c1 = {"a": torch.tensor([3.0, 5.0]), "b": torch.full((2, 2), 3.0)}
    out = checkpoint_diagonal_gaussian_barycenter((c0, c1), weights=(1.0, 3.0))
    assert tuple(sorted(out.mean_state)) == ("a", "b")
    assert torch.allclose(out.mean_state["a"], torch.tensor([2.5, 4.5]))
    assert out.to_metadata()["score_claim"] is False
    with pytest.raises(CompositionPrimitiveError):
        checkpoint_diagonal_gaussian_barycenter((c0, {"a": torch.ones(2)}))
    with pytest.raises(CompositionPrimitiveError):
        checkpoint_diagonal_gaussian_barycenter(
            (c0, {"a": torch.ones(3), "b": torch.ones(2, 2)}),
        )


def test_mera_hierarchy_metadata_is_deterministic_and_fail_closed() -> None:
    meta_a = build_mera_hierarchy_metadata(
        {"layer_b": (2, 4), "layer_a": (4, 4)},
        source_checkpoint_ids=("ckpt1", "ckpt2"),
        max_bond_dim=3,
    )
    meta_b = build_mera_hierarchy_metadata(
        {"layer_a": (4, 4), "layer_b": (2, 4)},
        source_checkpoint_ids=("ckpt1", "ckpt2"),
        max_bond_dim=3,
    )
    assert meta_a.to_json() == meta_b.to_json()
    assert meta_a.sha256() == meta_b.sha256()
    assert meta_a.to_dict()["ready_for_exact_eval_dispatch"] is False
    with pytest.raises(CompositionPrimitiveError):
        build_mera_hierarchy_metadata(
            {"bad": (0, 4)},
            source_checkpoint_ids=("ckpt1",),
            max_bond_dim=3,
        )


def test_bregman_squared_barycenter_is_weighted_mean_and_associative() -> None:
    xs = (torch.tensor([0.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor([8.0, 10.0]))
    direct = bregman_barycenter(xs, (0.25, 0.25, 0.5))
    group = bregman_barycenter(xs[:2], (0.5, 0.5))
    grouped = bregman_barycenter((group, xs[2]), (0.5, 0.5))
    assert torch.allclose(direct, torch.tensor([5.0, 7.0]))
    assert torch.allclose(grouped, direct)


def test_bregman_kl_modes_normalize_and_reject_negative_mass() -> None:
    p = torch.tensor([0.8, 0.2])
    q = torch.tensor([0.2, 0.8])
    forward = bregman_barycenter((p, q), divergence="kl_forward")
    reverse = bregman_barycenter((p, q), divergence="kl_reverse")
    assert torch.allclose(forward.sum(), torch.tensor(1.0))
    assert torch.allclose(reverse.sum(), torch.tensor(1.0))
    assert torch.allclose(forward, torch.tensor([0.5, 0.5]), atol=1e-6)
    assert torch.allclose(reverse, torch.tensor([0.5, 0.5]), atol=1e-6)
    with pytest.raises(CompositionPrimitiveError):
        bregman_barycenter((torch.tensor([1.0, -1.0]), q), divergence="kl_forward")


def test_sinkhorn_transport_plan_matches_marginals_and_validates_shape() -> None:
    source = torch.tensor([0.5, 0.5])
    target = torch.tensor([0.25, 0.75])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = sinkhorn_transport_plan(
        source,
        target,
        cost,
        epsilon=0.5,
        max_iters=500,
        tol=1e-7,
    )
    assert result.converged
    assert torch.allclose(result.plan.sum(dim=1), result.row_marginal, atol=1e-5)
    assert torch.allclose(result.plan.sum(dim=0), result.col_marginal, atol=1e-5)
    with pytest.raises(CompositionPrimitiveError):
        sinkhorn_transport_plan(source, target, torch.ones(3, 2))
