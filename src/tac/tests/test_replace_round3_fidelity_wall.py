# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.scorer_surrogate.replace_round3_fidelity_wall import (
    BASE_FEATURE_COUNT,
    RFF_FEATURE_COUNT,
    ConvMacLedger,
    aggregate_multi_target_statistics,
    cache_multi_target_sufficient_statistics,
    capture_exact_teacher_with_prefix_adjoint,
    deterministic_rff_parameters,
    exact_prefix_vjp,
    fit_multi_target_ridge,
    local_prefix_feature_snapshot,
    log_costate_mass_target_rows,
    mass_localization_metrics,
    prefix_feature_matrix,
    rff_lift,
    source_margin_risk_scores,
)
from tac.witness_dsl.replace_round3_fidelity_wall_policy import (
    PREREGISTERED_INPUT_COSTATE_COSINE_BAR,
    ReplaceRound3FidelityWallPolicy,
)


def test_policy_seals_new_instance_and_honest_teacher_accounting() -> None:
    policy = ReplaceRound3FidelityWallPolicy()
    contract = policy.compile_measurement_contract()
    assert contract["input_costate_cosine_bar"] == pytest.approx(
        PREREGISTERED_INPUT_COSTATE_COSINE_BAR
    )
    assert contract["label_only_teacher_amortization_x"] == 15.0
    assert contract["inclusive_teacher_amortization_x"] == 12.0
    assert contract["localizer_mass_fraction_bar"] == 0.47
    assert contract["fore_weighting_enabled"] is False
    with pytest.raises(ValueError, match="preregistered"):
        ReplaceRound3FidelityWallPolicy(rff_frequency_count=32)


def test_prefix_chart_and_rff_are_fixed_and_deterministic() -> None:
    generator = np.random.default_rng(455)
    prefix = generator.standard_normal((1, 32, 4, 5), dtype=np.float32)
    labels = generator.integers(0, 5, size=(8, 10), dtype=np.int64)
    margins = generator.standard_normal((8, 10), dtype=np.float32)
    base = prefix_feature_matrix(
        prefix, labels, margins, checkpoint_index=1, checkpoint_count=3, stride=1
    )
    assert base.shape == (20, BASE_FEATURE_COUNT)
    assert np.all(base[:, 0] == 1.0)
    lifted_a = rff_lift(base)
    lifted_b = rff_lift(base)
    assert lifted_a.shape == (20, RFF_FEATURE_COUNT)
    assert np.array_equal(lifted_a, lifted_b)
    assert np.array_equal(deterministic_rff_parameters(), deterministic_rff_parameters())


def test_multi_target_ridge_reuses_round2_contraction_for_arbitrary_width() -> None:
    generator = np.random.default_rng(455)
    records = []
    for _state in range(4):
        features = generator.standard_normal((32, 7), dtype=np.float32)
        targets = generator.standard_normal((32, 5), dtype=np.float32)
        records.append(cache_multi_target_sufficient_statistics(features, targets))
    stats = aggregate_multi_target_statistics(records)
    fit = fit_multi_target_ridge(stats, epochs=15)
    summary = fit.summary()
    assert fit.weights.shape == (7, 5)
    assert summary["three_channel_block_count"] == 2
    assert summary["certificate"]["contraction_gamma"] < 1.0
    assert summary["certificate"]["ideal_gamma_upper_bound"] == pytest.approx(1.0 / 3.0)
    assert summary["residual_bounds_all_validated"] is True


def test_sufficient_statistics_suppress_stale_flags_but_refuse_real_overflow() -> None:
    features = np.full((4, 2), np.finfo(np.float32).max, dtype=np.float32)
    targets = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="nonfinite"):
        cache_multi_target_sufficient_statistics(features, targets)


def test_mass_target_and_localizers_obey_prefix_cell_geometry() -> None:
    costate = np.zeros((1, 3, 4, 4), dtype=np.float32)
    costate[:, :, :2, :2] = 2.0
    log_rows = log_costate_mass_target_rows(costate, stride=1)
    assert log_rows.shape == (4, 1)
    scores = np.array([[4.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    metrics = mass_localization_metrics(costate, scores, area_fraction=0.24)
    assert metrics["selected_prefix_cells"] == 1
    assert metrics["retained_exact_costate_l2_mass_fraction"] == 1.0
    assert metrics["conditional_masked_exact_costate_cosine"] == 1.0
    margins = np.array(
        [[0.01, 0.02, 2.0, 2.0], [0.03, 0.04, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]],
        dtype=np.float32,
    )
    margin_scores = source_margin_risk_scores(margins)
    assert np.unravel_index(np.argmax(margin_scores), margin_scores.shape) == (0, 0)


def test_exact_local_prefix_vjp_and_mac_ledger() -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_dw = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
            self.bn1 = torch.nn.Sequential(torch.nn.BatchNorm2d(32), torch.nn.SiLU())
            self.aa = torch.nn.Identity()

    class EncoderModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_stem = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
            self.bn1 = torch.nn.Sequential(torch.nn.BatchNorm2d(32), torch.nn.SiLU())
            self.blocks = torch.nn.Sequential(torch.nn.Sequential(Block()))

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = EncoderModel()

    class FakeSegNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.head = torch.nn.Conv2d(32, 5, 1)

        def forward(self, value):
            model = self.encoder.model
            block = model.blocks[0][0]
            prefix = block.aa(block.bn1(block.conv_dw(model.bn1(model.conv_stem(value)))))
            return self.head(functional.interpolate(prefix, size=value.shape[-2:], mode="nearest"))

    torch.manual_seed(455)
    segnet = FakeSegNet().eval()
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)
    frame = torch.randn(1, 3, 8, 10)
    labels = torch.zeros((1, 8, 10), dtype=torch.long)
    with ConvMacLedger(segnet) as mac_ledger:
        prefix, prefix_adjoint, input_costate, metrics, _elapsed = (
            capture_exact_teacher_with_prefix_adjoint(
                segnet=segnet, frame_nchw=frame, labels=labels
            )
        )
    cost = mac_ledger.summary()
    predicted, _elapsed = exact_prefix_vjp(
        segnet=segnet, frame_nchw=frame, prefix_adjoint_nchw=prefix_adjoint
    )
    assert prefix.shape == prefix_adjoint.shape == (1, 32, 4, 5)
    assert torch.equal(prefix, local_prefix_feature_snapshot(segnet, frame))
    assert predicted.shape == input_costate.shape == frame.shape
    assert metrics.keys() == {"ce", "dseg"}
    assert 0.0 < cost["prefix_fraction_of_full_teacher_conv_flops"] < 1.0
