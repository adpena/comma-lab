# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.elm_inr_head_solve import (
    StreamingPartitionedRidge,
    StreamingRidgeNormalEquations,
    extract_levelset_hidden_numpy,
    partitioned_affine_predict,
    rectangular_pou_weights,
    smoothed_ce_logit_targets,
    solve_partitioned_affine_head_with_fold,
    verify_seed_checkpoint_preservation,
    write_seed_checkpoint_atomic,
)
from tac.boundary_math.lever_b_levelset_generator import (
    fit_out_sdf_to_structured_target,
    levelset_rgb_forward_numpy,
)


def test_exact_synthetic_affine_recovery_and_mlx_orientation() -> None:
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(256, 6))
    weight = rng.normal(size=(4, 6))
    bias = rng.normal(size=(4,))
    targets = hidden @ weight.T + bias

    normal = StreamingRidgeNormalEquations(6, 4, ridge=0.0)
    normal.update(hidden, targets)
    beta, diagnostics = normal.solve()

    assert diagnostics.rank == 7
    np.testing.assert_allclose(beta[:-1].T, weight, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(beta[-1], bias, rtol=1e-10, atol=1e-10)
    assert normal.residual_rmse(beta) == pytest.approx(0.0, abs=2e-7)


def test_rectangular_pou_is_nonnegative_and_sums_to_one_across_boundaries() -> None:
    coords = np.asarray(
        [
            [-1.0, -1.0],
            [-0.5, -0.5],
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    weights = rectangular_pou_weights(coords, grid_shape=(3, 4))
    assert weights.shape == (coords.shape[0], 12)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)


def test_streaming_accumulation_equals_one_shot_normal_equations() -> None:
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(79, 5))
    targets = rng.normal(size=(79, 3))
    weights = rng.uniform(0.0, 1.0, size=79)

    one = StreamingRidgeNormalEquations(5, 3, ridge=0.031)
    one.update(hidden, targets, weights)
    streamed = StreamingRidgeNormalEquations(5, 3, ridge=0.031)
    for start, stop in ((0, 7), (7, 31), (31, 48), (48, 79)):
        streamed.update(hidden[start:stop], targets[start:stop], weights[start:stop])

    np.testing.assert_allclose(streamed.gram, one.gram, rtol=2e-15, atol=2e-14)
    np.testing.assert_allclose(streamed.rhs, one.rhs, rtol=2e-15, atol=2e-14)
    assert streamed.target_square_sum == pytest.approx(one.target_square_sum, rel=2e-15)
    assert streamed.weight_sum == pytest.approx(one.weight_sum, rel=2e-15)


def test_singular_system_is_finite_and_deterministic() -> None:
    hidden = np.ones((32, 4), dtype=np.float64)
    targets = np.column_stack([np.linspace(-1.0, 1.0, 32), np.ones(32)])

    solved = []
    diagnostics = []
    for _ in range(2):
        normal = StreamingRidgeNormalEquations(4, 2, ridge=0.0)
        normal.update(hidden, targets)
        beta, diag = normal.solve()
        solved.append(beta)
        diagnostics.append(diag)
    assert diagnostics[0].rank < diagnostics[0].dimension
    assert np.all(np.isfinite(solved[0]))
    np.testing.assert_array_equal(solved[0], solved[1])


def _witness_fixture(*, optional_pair_layers: bool) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    n_pixels, in_features, hidden, depth, mod_dim, classes = 37, 5, 7, 2, 3, 4
    params: dict[str, np.ndarray] = {
        "in_proj.weight": rng.normal(scale=0.2, size=(hidden, in_features)).astype(np.float32),
        "in_proj.bias": rng.normal(scale=0.1, size=hidden).astype(np.float32),
        "film.weight": rng.normal(scale=0.03, size=(depth * 2 * hidden, mod_dim)).astype(np.float32),
        "film.bias": rng.normal(scale=0.02, size=depth * 2 * hidden).astype(np.float32),
        "out_sdf.weight": rng.normal(scale=0.2, size=(classes, hidden)).astype(np.float32),
        "out_sdf.bias": rng.normal(scale=0.1, size=classes).astype(np.float32),
        "out_tex.weight": rng.normal(scale=0.1, size=(3, hidden)).astype(np.float32),
        "out_tex.bias": rng.normal(scale=0.1, size=3).astype(np.float32),
        "palette": rng.normal(scale=0.2, size=(classes, 3)).astype(np.float32),
    }
    for layer in range(depth):
        params[f"hidden.{layer}.weight"] = rng.normal(scale=0.2, size=(hidden, hidden)).astype(np.float32)
        params[f"hidden.{layer}.bias"] = rng.normal(scale=0.1, size=hidden).astype(np.float32)
        if optional_pair_layers:
            params[f"film_pl.{layer}.weight"] = rng.normal(
                scale=0.02, size=(2 * hidden, mod_dim)
            ).astype(np.float32)
            params[f"film_pl.{layer}.bias"] = rng.normal(scale=0.01, size=2 * hidden).astype(np.float32)
            params[f"concat_pl.{layer}.weight"] = rng.normal(
                scale=0.02, size=(hidden, mod_dim)
            ).astype(np.float32)
            params[f"concat_pl.{layer}.bias"] = rng.normal(scale=0.01, size=hidden).astype(np.float32)
    features = rng.normal(size=(n_pixels, in_features)).astype(np.float32)
    code = rng.normal(size=mod_dim).astype(np.float32)
    return params, features, code


@pytest.mark.parametrize("optional_pair_layers", [False, True])
def test_hidden_feature_forward_matches_canonical_numpy_phi(optional_pair_layers: bool) -> None:
    params, features, code = _witness_fixture(optional_pair_layers=optional_pair_layers)
    common = {
        "n_hidden": 2,
        "hidden_dim": 7,
        "activation": "hosc",
        "wire_w0": 20.0,
        "wire_s0": 10.0,
        "hosc_beta": 2.5,
        "hosc_omega": 1.0,
    }
    hidden = extract_levelset_hidden_numpy(params, features, code, **common)
    reconstructed_phi = hidden.astype(np.float64) @ params["out_sdf.weight"].astype(np.float64).T
    reconstructed_phi += params["out_sdf.bias"].astype(np.float64)
    _, canonical_phi = levelset_rgb_forward_numpy(
        params,
        features,
        code,
        **common,
        n_classes=4,
        softmax_temp=0.7,
        chroma=True,
    )
    np.testing.assert_allclose(reconstructed_phi, canonical_phi, rtol=0.0, atol=2e-7)


def test_atomic_checkpoint_preserves_every_non_sdf_array_and_metadata(tmp_path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "seed.npz"
    arrays = {
        "out_sdf.weight": np.arange(15, dtype=np.float32).reshape(3, 5),
        "out_sdf.bias": np.arange(3, dtype=np.float32),
        "out_tex.weight": np.arange(10, dtype=np.float32).reshape(2, 5),
        "palette": np.arange(9, dtype=np.float32).reshape(3, 3),
        "__cfg_activation": np.asarray("hosc"),
        "__epoch": np.asarray(650, np.int64),
    }
    np.savez(source, **arrays)
    new_weight = np.full((3, 5), 7.0, np.float32)
    new_bias = np.full(3, -2.0, np.float32)

    write_seed_checkpoint_atomic(source, output, weight=new_weight, bias=new_bias)
    preservation = verify_seed_checkpoint_preservation(source, output)
    assert preservation == {
        "source_key_count": 6,
        "non_head_key_count": 4,
        "changed_keys": ["out_sdf.bias", "out_sdf.weight"],
        "all_non_head_arrays_exact": True,
    }
    with np.load(output, allow_pickle=True) as seeded:
        assert set(seeded.files) == set(arrays)
        np.testing.assert_array_equal(seeded["out_sdf.weight"], new_weight)
        np.testing.assert_array_equal(seeded["out_sdf.bias"], new_bias)
        for key in arrays.keys() - {"out_sdf.weight", "out_sdf.bias"}:
            np.testing.assert_array_equal(seeded[key], arrays[key])


def _solve_partitioned_pipeline(
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    interrupt_after: int | None,
) -> np.ndarray:
    local = StreamingPartitionedRidge(4, 3, grid_shape=(2, 3), ridge=1e-4)
    for index, (hidden, targets, coords) in enumerate(batches):
        local.update(hidden, targets, coords)
        if interrupt_after is not None and index + 1 == interrupt_after:
            local = StreamingPartitionedRidge.from_state_dict(local.state_dict())
    local_beta, _ = local.solve()
    global_fold = StreamingRidgeNormalEquations(4, 3, ridge=1e-4)
    for index, (hidden, _targets, coords) in enumerate(batches):
        prediction = partitioned_affine_predict(hidden, coords, local_beta, grid_shape=(2, 3))
        global_fold.update(hidden, prediction)
        if interrupt_after is not None and index + 1 == interrupt_after:
            global_fold = StreamingRidgeNormalEquations.from_state_dict(global_fold.state_dict("g"), "g")
    return global_fold.solve()[0]


def test_interrupted_resume_matches_uninterrupted_final_seed() -> None:
    rng = np.random.default_rng(23)
    batches = []
    for _ in range(5):
        hidden = rng.normal(size=(43, 4))
        coords = rng.uniform(-1.0, 1.0, size=(43, 2))
        labels = rng.integers(0, 3, size=43)
        targets = smoothed_ce_logit_targets(
            labels,
            n_classes=3,
            smoothing=0.1,
            temperature=0.8,
        )
        batches.append((hidden, targets, coords))
    uninterrupted = _solve_partitioned_pipeline(batches, interrupt_after=None)
    resumed = _solve_partitioned_pipeline(batches, interrupt_after=2)
    np.testing.assert_array_equal(resumed, uninterrupted)


def test_one_subdomain_is_directly_deployable_global_affine_solve() -> None:
    rng = np.random.default_rng(29)
    hidden = rng.normal(size=(101, 3))
    coords = rng.uniform(-1.0, 1.0, size=(101, 2))
    targets = rng.normal(size=(101, 2))
    local = StreamingPartitionedRidge(3, 2, grid_shape=(1, 1), ridge=0.02)
    local.update(hidden, targets, coords)
    local_beta, _ = local.solve()
    direct = StreamingRidgeNormalEquations(3, 2, ridge=0.02)
    direct.update(hidden, targets)
    direct_beta, _ = direct.solve()
    np.testing.assert_array_equal(local_beta[0], direct_beta)


def test_full_partitioned_callable_reports_direct_local_fold_and_receiver_optimum() -> None:
    rng = np.random.default_rng(301)
    hidden = rng.normal(size=(511, 7))
    coords = rng.uniform(-1.0, 1.0, size=(511, 2))
    # A coordinate-local perturbation gives the POU field something real to fit that a single
    # affine receiver cannot represent exactly.
    base_weight = rng.normal(size=(4, 7))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        targets = hidden @ base_weight.T
    assert np.all(np.isfinite(targets))
    targets[:, 0] += 0.35 * (coords[:, 0] > 0.0)
    targets[:, 1] -= 0.25 * (coords[:, 1] > 0.0)

    first = solve_partitioned_affine_head_with_fold(
        hidden,
        targets,
        coords,
        grid_shape=(2, 2),
        ridge=0.0,
    )
    second = solve_partitioned_affine_head_with_fold(
        hidden,
        targets,
        coords,
        grid_shape=(2, 2),
        ridge=0.0,
    )

    np.testing.assert_array_equal(first.local_beta, second.local_beta)
    np.testing.assert_array_equal(first.direct_global_beta, second.direct_global_beta)
    np.testing.assert_array_equal(first.folded_global_beta, second.folded_global_beta)
    assert first.fold_second_solve_applied is True
    assert first.pou_local_target_rmse < first.direct_global_target_rmse
    assert first.folded_global_target_rmse >= first.direct_global_target_rmse - 1e-10
    assert first.fold_vs_local_rmse > 0.0


def test_grid1_bypasses_second_ridge_instead_of_double_shrinking() -> None:
    rng = np.random.default_rng(307)
    hidden = rng.normal(size=(193, 5))
    coords = rng.uniform(-1.0, 1.0, size=(193, 2))
    targets = rng.normal(size=(193, 3))
    solution = solve_partitioned_affine_head_with_fold(
        hidden,
        targets,
        coords,
        grid_shape=(1, 1),
        ridge=3.0,
    )

    assert solution.fold_second_solve_applied is False
    assert solution.fold_diagnostics is None
    assert solution.fold_vs_local_rmse == 0.0
    np.testing.assert_array_equal(solution.folded_global_beta, solution.local_beta[0])
    # The direct comparator is unregularized target-SSE authority; positive local ridge is
    # allowed to differ but cannot have lower raw target SSE.
    assert solution.folded_global_target_rmse >= solution.direct_global_target_rmse - 1e-10


def test_one_by_one_zero_ridge_extends_settled_global_helper_without_rederivation() -> None:
    """The new 1x1 stream is the settled helper plus bounded/resumable custody, not new algebra."""

    rng = np.random.default_rng(31)
    hidden = rng.normal(size=(177, 6)).astype(np.float32)
    targets = rng.normal(size=(177, 4)).astype(np.float32)
    coords = rng.uniform(-1.0, 1.0, size=(177, 2)).astype(np.float32)
    settled_weight, settled_bias = fit_out_sdf_to_structured_target(hidden, targets, ridge=0.0)
    streamed = StreamingPartitionedRidge(6, 4, grid_shape=(1, 1), ridge=0.0)
    for start in range(0, hidden.shape[0], 23):
        streamed.update(hidden[start : start + 23], targets[start : start + 23], coords[start : start + 23])
    beta, _ = streamed.solve()
    np.testing.assert_allclose(beta[0, :-1].T, settled_weight, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(beta[0, -1], settled_bias, rtol=2e-6, atol=2e-6)


def test_smoothed_target_refuses_exact_one_hot_infinite_logit_request() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        smoothed_ce_logit_targets(np.asarray([0, 1]), n_classes=2, smoothing=0.0, temperature=1.0)
