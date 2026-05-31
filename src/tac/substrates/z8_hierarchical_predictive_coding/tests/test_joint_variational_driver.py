# SPDX-License-Identifier: MIT
"""Z8 joint variational driver contract tests."""

from __future__ import annotations

import pytest

from tac.substrates.z8_hierarchical_predictive_coding.joint_variational_driver import (
    Z8JointVariationalDriverConfig,
    build_z8_joint_variational_driver_metadata,
    expected_categorical_archive_rate_score,
)

try:
    import mlx.core as _mx
except ImportError:  # pragma: no cover - non-Apple CI path
    _mx = None

mlx_only = pytest.mark.skipif(_mx is None, reason="MLX required")


def test_joint_variational_metadata_is_non_authority_and_score_grounded() -> None:
    cfg = Z8JointVariationalDriverConfig(
        archive_rate_weight=2.0,
        argmax_commitment_weight=0.25,
    )

    metadata = build_z8_joint_variational_driver_metadata(
        cfg,
        archive_export_enabled=True,
    )

    assert metadata["schema"] == "z8_joint_variational_driver.v1"
    assert metadata["archive_rate_weight"] == 2.0
    assert metadata["argmax_commitment_weight"] == 0.25
    assert "25.0 * expected_categorical_bytes / 37545489" in metadata["rate_formula"]
    assert metadata["ste_boundary"] == "gumbel_softmax_argmax_indices_to_archive"
    assert metadata["archive_export_enabled"] is True
    assert metadata["implicit_dykstra_allocator_diff_status"] == "pending_follow_on"
    assert "score_claim" not in metadata
    assert "promotion_eligible" not in metadata
    assert "ready_for_exact_eval_dispatch" not in metadata


def test_joint_variational_config_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="archive_rate_weight"):
        Z8JointVariationalDriverConfig(archive_rate_weight=-1.0)

    with pytest.raises(ValueError, match="argmax_commitment_weight"):
        Z8JointVariationalDriverConfig(argmax_commitment_weight=-1.0)


@mlx_only
def test_expected_categorical_rate_score_rewards_sharper_archive_posteriors() -> None:
    import mlx.core as mx

    class _Cfg:
        num_pairs = 2

    class _Model:
        cfg = _Cfg()

        def __init__(self, logits) -> None:
            self.logits_per_level = [logits]

    pair_indices = mx.array([0, 1], dtype=mx.int32)
    uniform = _Model(mx.zeros((2, 2, 4)))
    sharp = _Model(
        mx.array(
            [
                [[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]],
                [[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]],
            ]
        )
    )

    uniform_score = expected_categorical_archive_rate_score(uniform, pair_indices)
    sharp_score = expected_categorical_archive_rate_score(sharp, pair_indices)
    mx.eval(uniform_score, sharp_score)

    assert float(sharp_score.item()) < float(uniform_score.item())
