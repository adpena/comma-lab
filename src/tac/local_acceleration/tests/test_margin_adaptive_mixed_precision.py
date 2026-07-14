# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tac.local_acceleration.margin_adaptive_mixed_precision import (
    ProfileCertificate,
    build_margin_adaptive_int64_model,
    derive_capped_precision_map,
    interval_argmax_certificate_mask,
    solve_finite_profile_waterfill,
    validate_precision_map,
    weighted_average_bits,
)
from tac.local_acceleration.weight_l1_int64_fixedpoint_scorer import (
    WeightL1Int64Conv2d,
)


def _model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(4, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 5, 1),
    ).eval()


def test_capped_map_is_nested_and_covers_every_conv() -> None:
    model = _model()
    low = derive_capped_precision_map(model, cap_bits=8)
    high = derive_capped_precision_map(model, cap_bits=20)
    assert set(low) == {"0", "2"}
    assert all(low[path] <= high[path] for path in low)
    assert set(low.values()) == {8}


def test_arbitrary_map_builds_exact_int64_per_channel_twin() -> None:
    model = _model()
    precision = {"0": 8, "2": 12}
    candidate, manifest = build_margin_adaptive_int64_model(
        model,
        precision_by_path=precision,
    )
    assert isinstance(candidate[0], WeightL1Int64Conv2d)
    assert isinstance(candidate[2], WeightL1Int64Conv2d)
    assert dict(manifest.precision_by_path) == precision
    assert manifest.converted_conv2d_count == 2
    assert manifest.accumulation == "exact_signed_int64"
    assert manifest.scale_granularity.startswith("per_output_channel")
    assert dict(manifest.integer_storage_bits_by_path) == {"0": 8, "2": 16}
    assert dict(manifest.integer_storage_histogram) == {8: 1, 16: 1}
    assert manifest.region_runtime_claim is False
    assert manifest.native_speed_claim is False
    with torch.inference_mode():
        output = candidate(torch.randn(1, 4, 5, 6))
    assert output.shape == (1, 5, 5, 6)
    assert torch.isfinite(output).all()


def test_precision_map_refuses_partial_or_unsafe_maps() -> None:
    model = _model()
    with pytest.raises(ValueError, match="coverage differs"):
        validate_precision_map(model, {"0": 8})
    with pytest.raises(ValueError, match="outside signed-int64-safe"):
        validate_precision_map(model, {"0": 32, "2": 8})


def test_weighted_average_bits_is_explicit_about_work_measure() -> None:
    assert weighted_average_bits({"a": 8, "b": 16}, {"a": 3, "b": 1}) == 10.0
    with pytest.raises(ValueError, match="coverage differs"):
        weighted_average_bits({"a": 8}, {"a": 1, "b": 1})


def test_interval_certificate_uses_strict_top1_rival_separation() -> None:
    # NCHW, two pixels.  Pixel 0 remains separated under its radius.  Pixel 1
    # has an exact tie and therefore cannot receive a strict certificate.
    logits = np.asarray([[[[3.0, 2.0]], [[1.0, 2.0]], [[0.0, 0.0]]]], dtype=np.float32)
    radius = np.asarray([[[[0.25, 0.0]], [[0.25, 0.0]], [[0.25, 0.0]]]], dtype=np.float32)
    mask = interval_argmax_certificate_mask(logits, radius, class_axis=1)
    np.testing.assert_array_equal(mask, np.asarray([[[True, False]]]))


def test_interval_certificate_can_check_a_supplied_reference_winner() -> None:
    logits = np.asarray([[[[3.0]], [[1.0]], [[0.0]]]], dtype=np.float32)
    radius = np.full_like(logits, 0.1)
    expected = np.asarray([[[0]]], dtype=np.int64)
    assert bool(
        interval_argmax_certificate_mask(
            logits,
            radius,
            expected_argmax=expected,
            class_axis=1,
        )[0, 0, 0]
    )
    wrong = np.asarray([[[1]]], dtype=np.int64)
    assert not bool(
        interval_argmax_certificate_mask(
            logits,
            radius,
            expected_argmax=wrong,
            class_axis=1,
        )[0, 0, 0]
    )


def test_finite_ladder_waterfill_is_pointwise_minimum_over_profiles() -> None:
    low = ProfileCertificate(
        name="cap8",
        average_bits=8.0,
        certified_mask=np.asarray([[True, False, False]]),
    )
    middle = ProfileCertificate(
        name="cap12",
        average_bits=12.0,
        certified_mask=np.asarray([[True, True, False]]),
    )
    high = ProfileCertificate(
        name="cap16",
        average_bits=16.0,
        certified_mask=np.asarray([[True, True, True]]),
    )
    result = solve_finite_profile_waterfill([high, low, middle])
    np.testing.assert_array_equal(result.selected_profile_index, [[0, 1, 2]])
    assert result.profile_order == ("cap8", "cap12", "cap16")
    assert dict(result.profile_histogram) == {"cap8": 1, "cap12": 1, "cap16": 1}
    assert result.average_selected_bits == 12.0
    assert result.native_region_execution_claim is False
    assert result.to_summary()["certified_fraction"] == 1.0
