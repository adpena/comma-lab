# SPDX-License-Identifier: MIT
"""Tests for Lane 17 IMP β-variant — sensitivity-weighted pruning."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.imp_sensitivity_weighted import (
    SensitivityWeightedIMPError,
    classify_layers_by_sensitivity,
    prune_with_sensitivity_weighting,
)


def _three_conv_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 4, 1),
    )


def _make_sensitivity_high_low_mid(model: nn.Module) -> dict[str, torch.Tensor]:
    """Conv 0 = high (protected), Conv 1 = mid (standard), Conv 2 = low (aggressive)."""
    return {
        "0.weight": torch.full((8,), 1e-2),  # high - protect
        "2.weight": torch.full((16,), 5e-4),  # mid - standard
        "4.weight": torch.full((4,), 1e-7),  # low - aggressive
    }


def test_classify_protect_standard_aggressive() -> None:
    model = _three_conv_model()
    sens = _make_sensitivity_high_low_mid(model)
    cls = classify_layers_by_sensitivity(
        model=model,
        sensitivities=sens,
        protect_threshold=1e-3,
        aggressive_threshold=1e-5,
    )
    assert cls["0.weight"] == "protect"
    assert cls["2.weight"] == "standard"
    assert cls["4.weight"] == "aggressive"


def test_classify_max_aggregation_protects_layer_with_one_high_channel() -> None:
    model = _three_conv_model()
    sens = {
        "0.weight": torch.full((8,), 1e-7),
        "2.weight": torch.full((16,), 1e-7),
        "4.weight": torch.full((4,), 1e-7),
    }
    sens["2.weight"][3] = 5e-3  # one channel with high sensitivity
    cls = classify_layers_by_sensitivity(
        model=model,
        sensitivities=sens,
    )
    # Conv 1 should be protected because its max channel is high
    assert cls["2.weight"] == "protect"


def test_classify_threshold_validation_rejects_bad_inputs() -> None:
    model = _three_conv_model()
    sens = _make_sensitivity_high_low_mid(model)
    with pytest.raises(SensitivityWeightedIMPError, match="must be > 0"):
        classify_layers_by_sensitivity(
            model=model, sensitivities=sens, protect_threshold=-1
        )
    with pytest.raises(SensitivityWeightedIMPError, match="< protect"):
        classify_layers_by_sensitivity(
            model=model,
            sensitivities=sens,
            protect_threshold=1e-3,
            aggressive_threshold=1e-2,
        )


def test_classify_missing_sensitivity_raises() -> None:
    model = _three_conv_model()
    with pytest.raises(SensitivityWeightedIMPError, match="missing sensitivity"):
        classify_layers_by_sensitivity(
            model=model,
            sensitivities={"0.weight": torch.ones(8)},
        )


def test_prune_protected_layer_unchanged() -> None:
    torch.manual_seed(2026)
    model = _three_conv_model()
    sens = _make_sensitivity_high_low_mid(model)
    new_mask = prune_with_sensitivity_weighting(
        model=model,
        sensitivities=sens,
        sparsity_increment=0.20,
    )
    # Protected layer mask = full True (nothing pruned)
    assert new_mask["0.weight"].all().item() is True


def test_prune_aggressive_layer_more_sparse_than_standard() -> None:
    torch.manual_seed(2026)
    model = _three_conv_model()
    sens = {
        "0.weight": torch.full((8,), 5e-4),  # standard
        "2.weight": torch.full((16,), 5e-4),  # standard
        "4.weight": torch.full((4,), 1e-7),  # aggressive
    }
    new_mask = prune_with_sensitivity_weighting(
        model=model,
        sensitivities=sens,
        sparsity_increment=0.20,
        aggressive_multiplier=1.5,
    )
    # Aggressive layer has 4 channels × 16 channels × 1 × 1 = 64 weights
    # Pruned at 0.30 → ~19 should be pruned.
    # Standard layer 0.weight has 8×3×3×3 = 216 weights, pruned at 0.20 → ~43 pruned
    aggressive_sparsity = float(
        (~new_mask["4.weight"]).sum() / new_mask["4.weight"].numel()
    )
    standard_sparsity = float(
        (~new_mask["0.weight"]).sum() / new_mask["0.weight"].numel()
    )
    # The aggressive layer should be more sparse (within tolerance)
    assert aggressive_sparsity > standard_sparsity
    # Aggressive should be near 0.30, standard near 0.20
    assert 0.20 <= aggressive_sparsity <= 0.40
    assert 0.10 <= standard_sparsity <= 0.30


def test_prune_with_existing_mask_monotone() -> None:
    torch.manual_seed(2026)
    model = _three_conv_model()
    sens = {
        "0.weight": torch.full((8,), 5e-4),  # standard
        "2.weight": torch.full((16,), 5e-4),  # standard
        "4.weight": torch.full((4,), 5e-4),  # standard
    }
    mask1 = prune_with_sensitivity_weighting(
        model=model,
        sensitivities=sens,
        sparsity_increment=0.20,
    )
    mask2 = prune_with_sensitivity_weighting(
        model=model,
        sensitivities=sens,
        sparsity_increment=0.20,
        current_mask=mask1,
    )
    # Cumulative sparsity must be monotone (already-pruned weights stay pruned)
    for key in mask1:
        # mask2[key] => mask1[key] (only weights surviving in mask1 can be alive in mask2)
        assert (mask2[key] & ~mask1[key]).sum().item() == 0


def test_prune_missing_current_mask_entry_uses_subset_weight_shape() -> None:
    torch.manual_seed(2026)
    model = _three_conv_model()
    sens = {
        "0.weight": torch.full((8,), 5e-4),
        "2.weight": torch.full((16,), 5e-4),
        "4.weight": torch.full((4,), 1e-7),
    }
    current_mask = {
        "0.weight": torch.ones_like(model[0].weight, dtype=torch.bool),
    }

    new_mask = prune_with_sensitivity_weighting(
        model=model,
        sensitivities=sens,
        sparsity_increment=0.20,
        current_mask=current_mask,
    )

    assert new_mask["2.weight"].shape == model[2].weight.shape
    assert new_mask["4.weight"].shape == model[4].weight.shape


def test_prune_aggressive_multiplier_capped() -> None:
    torch.manual_seed(2026)
    model = _three_conv_model()
    sens = {
        "0.weight": torch.full((8,), 1e-7),
        "2.weight": torch.full((16,), 1e-7),
        "4.weight": torch.full((4,), 1e-7),
    }
    # Multiplier × increment should be capped at 0.95 effective.
    new_mask = prune_with_sensitivity_weighting(
        model=model,
        sensitivities=sens,
        sparsity_increment=0.50,
        aggressive_multiplier=10.0,  # would otherwise blow past 1.0
    )
    # All layers aggressive; effective increment = min(0.95, 0.50*10) = 0.95.
    # Global magnitude pruning lets small layers go fully sparse, but the
    # GLOBAL aggregate should not exceed the cap.
    total = sum(int(m.numel()) for m in new_mask.values())
    pruned = sum(int((~m).sum().item()) for m in new_mask.values())
    global_sparsity = pruned / total
    # Capped at 0.95 (the prune fn allows at most n-1, so slightly less).
    assert global_sparsity <= 0.95 + 1e-6
    assert global_sparsity > 0.5  # something was pruned


def test_prune_invalid_sparsity_increment_raises() -> None:
    model = _three_conv_model()
    sens = _make_sensitivity_high_low_mid(model)
    with pytest.raises(SensitivityWeightedIMPError, match="must be in"):
        prune_with_sensitivity_weighting(
            model=model,
            sensitivities=sens,
            sparsity_increment=1.5,
        )


def test_prune_returns_only_prunable_layers() -> None:
    model = _three_conv_model()
    sens = _make_sensitivity_high_low_mid(model)
    new_mask = prune_with_sensitivity_weighting(
        model=model,
        sensitivities=sens,
        sparsity_increment=0.20,
    )
    # Should have masks for all 3 conv layers, nothing else (no activations etc.)
    assert set(new_mask) == {"0.weight", "2.weight", "4.weight"}


def test_classify_canonical_and_bare_keys_both_work() -> None:
    model = _three_conv_model()
    # Use bare keys (no .weight suffix) — should still resolve via fallback
    sens = {
        "0": torch.full((8,), 1e-2),
        "2": torch.full((16,), 5e-4),
        "4": torch.full((4,), 1e-7),
    }
    cls = classify_layers_by_sensitivity(model=model, sensitivities=sens)
    assert cls["0.weight"] == "protect"
    assert cls["2.weight"] == "standard"
    assert cls["4.weight"] == "aggressive"
