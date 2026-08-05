from __future__ import annotations

import numpy as np

from experiments import ddm_sm2_sum_surrogate_race as sm2


def test_affine_sum_fit_recovers_combined_delta_model():
    rows = []
    for e, s in [(-2.0, 1.0), (-1.0, -3.0), (0.5, 4.0), (2.0, -2.0), (4.0, 3.0)]:
        rows.append({"d_surr_entropy": e, "d_surr_smevr": s, "d_bytes": 7.0 + 3.0 * e - 5.0 * s})

    fit = sm2.fit_affine(
        "synthetic",
        rows,
        predictors=("d_surr_entropy", "d_surr_smevr"),
        target="d_bytes",
    )

    assert fit.n_rows == 5
    assert fit.rmse < 1e-9
    assert abs(fit.intercept - 7.0) < 1e-9
    assert abs(fit.coefficients["d_surr_entropy"] - 3.0) < 1e-9
    assert abs(fit.coefficients["d_surr_smevr"] + 5.0) < 1e-9


def test_pair_permutation_leaves_marginal_entropy_blind_but_moves_smevr_proxy():
    codes = np.array(
        [
            [[[0], [15]]],
            [[[1], [14]]],
            [[[2], [13]]],
            [[[13], [2]]],
            [[[14], [1]]],
            [[[15], [0]]],
        ],
        dtype=np.uint8,
    )
    identity = sm2.token_surrogates(codes, levels=16)
    permuted_codes = codes[[0, 2, 5, 1, 4, 3]]
    permuted_row = sm2.token_surrogates(permuted_codes, levels=16)

    assert abs(identity["surr_entropy_bits"] - permuted_row["surr_entropy_bits"]) < 1e-12
    assert identity["surr_smevr_surrogate_bits"] != permuted_row["surr_smevr_surrogate_bits"]


def test_full_transform_rows_emit_live_blind_subspace_deltas():
    codes = np.arange(8 * 2 * 2 * 1, dtype=np.uint8).reshape(8, 2, 2, 1) % 16
    rows = sm2.full_transform_rows(
        codes,
        levels=16,
        codec="smevr",
        temp=0.15,
        random_perms=1,
        seed=3,
    )

    identity = rows[0]
    assert identity["row_id"] == "identity"
    assert identity["d_bytes"] == 0
    assert identity["d_surr_entropy"] == 0.0
    assert identity["d_surr_smevr"] == 0.0
    permuted = [row for row in rows[1:] if row["row_id"] == "reverse_pairs"][0]
    assert abs(permuted["d_surr_entropy"]) < 1e-12
    assert "d_surr_smevr" in permuted
    assert "d_bytes" in permuted


def test_rank_metrics_returns_none_for_constant_predictor():
    metrics = sm2.rank_metrics([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    assert metrics["pearson"] is None
    assert metrics["spearman"] is None
