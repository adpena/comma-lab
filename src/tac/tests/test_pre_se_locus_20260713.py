# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.scorer_surrogate.pre_se_locus_20260713 import (
    PreSECutCostLedger,
    PreSEPairGatedMLPWeights,
    pre_se_feature_snapshot,
    pre_se_pair_block_features,
    pre_se_pair_gated_logits_numpy,
    verify_pre_se_taps,
)
from tac.witness_dsl.pre_se_locus_policy_20260713 import (
    LOCUS_SPECS,
    PreSELocusPolicy,
)


def test_pre_se_policy_changes_only_the_locus_contract() -> None:
    policy = PreSELocusPolicy()
    contract = policy.compile_measurement_contract()
    assert policy.base.n_pairs == 600
    assert policy.base.retained_mass_bar == 0.47
    assert policy.base.realized_area_fraction == 2311 / (192 * 256)
    assert contract["deep_feature_count_by_locus"] == {
        "block2-pre-se": 188,
        "block3-pre-se": 332,
    }
    assert contract["fresh_heldout_exact_calls"] == 120
    assert contract["inherited_train_exact_targets"] == 480
    assert contract["live_trainer_argv"] == []


@pytest.mark.parametrize(
    ("locus", "channels", "divisor", "feature_count"),
    (
        ("block2-pre-se", 144, 2, 188),
        ("block3-pre-se", 288, 4, 332),
    ),
)
def test_pre_se_chart_has_locus_specific_width_and_same_pair_rows(
    locus: str, channels: int, divisor: int, feature_count: int
) -> None:
    prefix = np.arange(32 * 8 * 12, dtype=np.float32).reshape(1, 32, 8, 12) / 1000.0
    feature = np.arange(
        channels * (8 // divisor) * (12 // divisor), dtype=np.float32
    ).reshape(1, channels, 8 // divisor, 12 // divisor)
    labels = np.indices((16, 24)).sum(axis=0).astype(np.int64) % 5
    margins = np.linspace(-2.0, 2.0, 16 * 24, dtype=np.float32).reshape(16, 24)
    pair_ids = np.arange(8 * 12, dtype=np.int16).reshape(8, 12) % 20
    rows, pairs = pre_se_pair_block_features(
        prefix,
        feature,
        labels,
        margins,
        pair_ids,
        locus=locus,
        checkpoint_index=1,
        checkpoint_count=3,
        stride=2,
    )
    assert rows.shape == (24, feature_count)
    assert pairs.shape == (24,)
    assert np.array_equal(pairs, pair_ids[::2, ::2].reshape(-1))


def test_pre_se_pair_gated_numpy_accepts_both_locus_widths() -> None:
    for spec in LOCUS_SPECS:
        features = np.ones((4, spec.feature_count), dtype=np.float32)
        input_weight = np.zeros((3, spec.feature_count), dtype=np.float32)
        input_bias = np.array((1.0, 2.0, 3.0), dtype=np.float32)
        output_weight = np.zeros((20, 3), dtype=np.float32)
        output_weight[2] = (1.0, 0.0, 0.0)
        output_weight[7] = (0.0, 1.0, 0.0)
        output_bias = np.arange(20, dtype=np.float32)
        weights = PreSEPairGatedMLPWeights(
            input_weight, input_bias, output_weight, output_bias
        )
        observed = pre_se_pair_gated_logits_numpy(
            features, np.array((2, 7, 2, 7)), weights
        )
        assert np.array_equal(
            observed, np.array((3.0, 9.0, 3.0, 9.0), dtype=np.float32)
        )


def test_pre_se_cut_predicate_excludes_own_se_and_projection() -> None:
    predicate = PreSECutCostLedger._under_pre_se_cut
    assert predicate("encoder.model.conv_stem", 1, 2)
    assert predicate("encoder.model.blocks.1.1.se.conv_expand", 1, 2)
    assert predicate("encoder.model.blocks.1.2.conv_pw", 1, 2)
    assert predicate("encoder.model.blocks.1.2.conv_dw", 1, 2)
    assert not predicate("encoder.model.blocks.1.2.se.conv_reduce", 1, 2)
    assert not predicate("encoder.model.blocks.1.2.conv_pwl", 1, 2)


def test_real_efficientnet_taps_are_pre_own_se_but_have_upstream_se() -> None:
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    torch.set_num_threads(1)
    segnet = smp.Unet(
        encoder_name="tu-efficientnet_b2",
        encoder_weights=None,
        in_channels=3,
        classes=5,
    ).eval()
    frame = torch.zeros(1, 3, 384, 512)
    prefix, block2, block3 = pre_se_feature_snapshot(segnet, frame)
    assert tuple(prefix.shape) == (1, 32, 192, 256)
    assert tuple(block2.shape) == (1, 144, 96, 128)
    assert tuple(block3.shape) == (1, 288, 48, 64)
    verification = verify_pre_se_taps(segnet, frame)
    assert verification["block2-pre-se"]["upstream_se_global_reduction_count"] == 4
    assert verification["block3-pre-se"]["upstream_se_global_reduction_count"] == 7
    for row in verification.values():
        assert row["equals_depthwise_activation_immediately_before_own_se"] is True
        assert row["own_se_applied_to_capture"] is False
        assert row["strict_end_to_end_independently_tileable_from_rgb"] is False
