# SPDX-License-Identifier: MIT
"""Tests for the Wyner-Ziv → per-byte sensitivity-map wire-in (Catalog #125 hook #1).

Validates the canonical helper
:func:`tac.sensitivity_map.axis_level_reweight` and the end-to-end sister
:func:`tac.sensitivity_map.update_sensitivity_map_from_master_gradient_anchor`.

Lane: lane_wire_in_2_wyner_ziv_covariance_to_sensitivity_map_20260517.
Memory: feedback_wire_in_2_wyner_ziv_to_sensitivity_map_landed_20260517.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tac.master_gradient_consumers import (
    WynerZivSideInfoClassification,
    wyner_ziv_side_info_covariance,
)
from tac.sensitivity_map import (
    MIXED_SENSITIVITY_BASELINE,
    PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT,
    SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT,
    WYNER_ZIV_REWEIGHT_OPERATING_POINT_TAG,
    WynerZivAxisLevelReweightError,
    axis_level_reweight,
)
from tac.sensitivity_map import wyner_ziv_reweight as _wyner_ziv_reweight_module

REAL_VALIDATE_NPY = Path(
    ".omx/tmp/master_gradient_per_pair_8pair_fp64_validate.npy"
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


def _build_classification(
    *,
    shared: tuple[int, ...] = (0, 1, 2),
    specific: tuple[int, ...] = (7, 8, 9),
    mixed: tuple[int, ...] = (3, 4, 5),
    n_bytes: int = 10,
    n_pairs: int = 8,
    aggregate_mean: float = 0.5,
    threshold_high: float = 0.8,
    threshold_low: float = 0.2,
) -> WynerZivSideInfoClassification:
    """Build a synthetic classification with disjoint sets."""
    return WynerZivSideInfoClassification(
        candidate_shared_prior_byte_indices=shared,
        pair_specific_byte_indices=specific,
        mixed_byte_indices=mixed,
        aggregate_byte_pair_correlation_mean=aggregate_mean,
        n_bytes=n_bytes,
        n_pairs=n_pairs,
        correlation_threshold_high=threshold_high,
        correlation_threshold_low=threshold_low,
        archive_sha256="deadbeef" * 8,
        estimated_wyner_ziv_gain_bytes=len(shared),
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Module surface + constants                                                    #
# ──────────────────────────────────────────────────────────────────────────── #


def test_canonical_constants_pinned():
    """The canonical bias factors must keep their documented values.

    Downstream consumers depend on these defaults — silent reordering would
    change downstream EV multipliers without notice.
    """
    assert SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT == 0.1
    assert PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT == 2.0
    assert MIXED_SENSITIVITY_BASELINE == 1.0
    assert WYNER_ZIV_REWEIGHT_OPERATING_POINT_TAG == "wyner_ziv_side_info_covariance_v1"


def test_module_public_api():
    """The package re-exports the canonical wire-in surface."""
    from tac import sensitivity_map as sm

    for name in (
        "axis_level_reweight",
        "update_sensitivity_map_from_master_gradient_anchor",
        "WynerZivAxisLevelReweightError",
        "SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT",
        "PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT",
        "MIXED_SENSITIVITY_BASELINE",
        "WYNER_ZIV_REWEIGHT_OPERATING_POINT_TAG",
        "wyner_ziv_reweight",
    ):
        assert hasattr(sm, name), f"sensitivity_map missing public API: {name!r}"
        assert name in sm.__all__, f"{name!r} missing from __all__"
    # Dotted submodule access
    assert sm.wyner_ziv_reweight is _wyner_ziv_reweight_module


# ──────────────────────────────────────────────────────────────────────────── #
# axis_level_reweight — synthetic inputs                                        #
# ──────────────────────────────────────────────────────────────────────────── #


def test_synthetic_shared_prior_bytes_get_downweighted():
    """Shared-prior bytes get the 10x downweight default."""
    c = _build_classification()
    weights = axis_level_reweight(c)
    for idx in c.candidate_shared_prior_byte_indices:
        assert weights[idx] == pytest.approx(SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT)


def test_synthetic_pair_specific_bytes_get_upweighted():
    """Pair-specific bytes get the 2x upweight default."""
    c = _build_classification()
    weights = axis_level_reweight(c)
    for idx in c.pair_specific_byte_indices:
        assert weights[idx] == pytest.approx(PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT)


def test_synthetic_mixed_bytes_stay_at_baseline():
    """Mixed bytes (between thresholds) retain the baseline weight."""
    c = _build_classification()
    weights = axis_level_reweight(c)
    for idx in c.mixed_byte_indices:
        assert weights[idx] == pytest.approx(MIXED_SENSITIVITY_BASELINE)


def test_synthetic_unclassified_bytes_default_to_baseline():
    """Bytes inside n_bytes but in NO classification set keep the baseline."""
    c = _build_classification(
        shared=(0,),
        specific=(9,),
        mixed=(),  # leaves 1-8 unclassified
        n_bytes=10,
    )
    weights = axis_level_reweight(c)
    for idx in range(1, 9):
        assert weights[idx] == pytest.approx(MIXED_SENSITIVITY_BASELINE)


def test_returned_dict_covers_full_byte_range():
    """The returned dict has exactly n_bytes entries, keyed 0..n_bytes-1."""
    c = _build_classification(n_bytes=10)
    weights = axis_level_reweight(c)
    assert sorted(weights.keys()) == list(range(10))
    assert len(weights) == 10


def test_determinism():
    """Same input → same output dict (insertion order + values)."""
    c = _build_classification()
    w1 = axis_level_reweight(c)
    w2 = axis_level_reweight(c)
    assert w1 == w2
    assert list(w1.items()) == list(w2.items())


def test_all_shared_classification_downweights_everything():
    """Edge case: every byte is shared-prior → every byte downweighted."""
    c = _build_classification(
        shared=tuple(range(10)),
        specific=(),
        mixed=(),
        n_bytes=10,
    )
    weights = axis_level_reweight(c)
    for v in weights.values():
        assert v == pytest.approx(SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT)


def test_all_specific_classification_upweights_everything():
    """Edge case: every byte is pair-specific → every byte upweighted."""
    c = _build_classification(
        shared=(),
        specific=tuple(range(10)),
        mixed=(),
        n_bytes=10,
    )
    weights = axis_level_reweight(c)
    for v in weights.values():
        assert v == pytest.approx(PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT)


def test_empty_classification_returns_baseline_for_all():
    """Edge case: no bytes classified into any set → all baseline."""
    c = _build_classification(
        shared=(),
        specific=(),
        mixed=(),
        n_bytes=10,
    )
    weights = axis_level_reweight(c)
    for v in weights.values():
        assert v == pytest.approx(MIXED_SENSITIVITY_BASELINE)


def test_zero_n_bytes_returns_empty_dict():
    """Edge case: n_bytes=0 → empty dict (legal degenerate input)."""
    c = _build_classification(shared=(), specific=(), mixed=(), n_bytes=0)
    weights = axis_level_reweight(c)
    assert weights == {}


# ──────────────────────────────────────────────────────────────────────────── #
# axis_level_reweight — base_weights pre-existing dict                          #
# ──────────────────────────────────────────────────────────────────────────── #


def test_base_weights_multiplied_at_classified_indices():
    """When base_weights is supplied, bias factors MULTIPLY the existing values."""
    c = _build_classification()
    base = {i: 5.0 for i in range(10)}
    weights = axis_level_reweight(c, base_weights=base)
    for idx in c.candidate_shared_prior_byte_indices:
        assert weights[idx] == pytest.approx(5.0 * SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT)
    for idx in c.pair_specific_byte_indices:
        assert weights[idx] == pytest.approx(5.0 * PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT)
    for idx in c.mixed_byte_indices:
        # mixed bytes preserve the base weight (no bias applied)
        assert weights[idx] == pytest.approx(5.0)


def test_base_weights_input_is_not_mutated():
    """The caller's base_weights mapping must not be mutated in place."""
    c = _build_classification()
    base = {i: 5.0 for i in range(10)}
    base_snapshot = dict(base)
    _ = axis_level_reweight(c, base_weights=base)
    assert base == base_snapshot


def test_sparse_base_weights_backfill_missing_keys_with_baseline():
    """Sparse base_weights → missing keys in [0, n_bytes) default to baseline."""
    c = _build_classification(
        shared=(),
        specific=(),
        mixed=(),
        n_bytes=10,
    )
    base = {0: 5.0, 5: 7.0}  # sparse
    weights = axis_level_reweight(c, base_weights=base)
    assert weights[0] == pytest.approx(5.0)
    assert weights[5] == pytest.approx(7.0)
    for idx in (1, 2, 3, 4, 6, 7, 8, 9):
        assert weights[idx] == pytest.approx(MIXED_SENSITIVITY_BASELINE)


def test_base_weights_preserves_out_of_range_keys():
    """base_weights keys outside [0, n_bytes) survive unchanged."""
    c = _build_classification(n_bytes=10)
    base = {i: 5.0 for i in range(10)}
    base[42] = 999.0  # outside range
    weights = axis_level_reweight(c, base_weights=base)
    assert weights[42] == pytest.approx(999.0)


# ──────────────────────────────────────────────────────────────────────────── #
# axis_level_reweight — custom bias factors                                     #
# ──────────────────────────────────────────────────────────────────────────── #


def test_custom_downweight_upweight_baseline():
    """Custom bias-factor kwargs override the canonical defaults.

    The bias factor is MULTIPLICATIVE on top of the baseline (or pre-existing
    base_weights value), so shared-prior bytes end up at
    ``baseline * downweight``, pair-specific at ``baseline * upweight``, and
    mixed at ``baseline``.
    """
    c = _build_classification()
    weights = axis_level_reweight(c, downweight=0.05, upweight=4.0, baseline=2.5)
    for idx in c.candidate_shared_prior_byte_indices:
        assert weights[idx] == pytest.approx(2.5 * 0.05)
    for idx in c.pair_specific_byte_indices:
        assert weights[idx] == pytest.approx(2.5 * 4.0)
    for idx in c.mixed_byte_indices:
        assert weights[idx] == pytest.approx(2.5)


def test_zero_downweight_zeroes_shared_prior_bytes():
    """downweight=0 → shared-prior bytes get exactly zero sensitivity."""
    c = _build_classification()
    weights = axis_level_reweight(c, downweight=0.0)
    for idx in c.candidate_shared_prior_byte_indices:
        assert weights[idx] == 0.0


# ──────────────────────────────────────────────────────────────────────────── #
# axis_level_reweight — validation / error paths                                #
# ──────────────────────────────────────────────────────────────────────────── #


def test_non_classification_input_raises():
    """Passing the wrong type raises WynerZivAxisLevelReweightError."""
    with pytest.raises(WynerZivAxisLevelReweightError, match="WynerZivSideInfoClassification"):
        axis_level_reweight("not a classification")  # type: ignore[arg-type]


def test_negative_downweight_raises():
    """Negative bias factor is rejected."""
    c = _build_classification()
    with pytest.raises(WynerZivAxisLevelReweightError, match="downweight"):
        axis_level_reweight(c, downweight=-0.1)


def test_nan_upweight_raises():
    """NaN bias factor is rejected."""
    c = _build_classification()
    with pytest.raises(WynerZivAxisLevelReweightError, match="upweight.*finite"):
        axis_level_reweight(c, upweight=float("nan"))


def test_inf_baseline_raises():
    """Infinite bias factor is rejected."""
    c = _build_classification()
    with pytest.raises(WynerZivAxisLevelReweightError, match="baseline.*finite"):
        axis_level_reweight(c, baseline=math.inf)


def test_negative_base_weight_value_raises():
    """A negative value in base_weights is rejected."""
    c = _build_classification()
    base = {i: -1.0 for i in range(10)}
    with pytest.raises(WynerZivAxisLevelReweightError, match="non-negative"):
        axis_level_reweight(c, base_weights=base)


def test_non_int_base_weight_key_raises():
    """A non-int key in base_weights is rejected."""
    c = _build_classification()
    base = {"0": 1.0}  # type: ignore[dict-item]
    with pytest.raises(WynerZivAxisLevelReweightError, match="must be int"):
        axis_level_reweight(c, base_weights=base)


def test_overlapping_classification_sets_raises():
    """A defensively-malformed classification (overlapping sets) is refused."""
    # Bypass dataclass-frozen check by constructing the overlap directly.
    c = WynerZivSideInfoClassification(
        candidate_shared_prior_byte_indices=(0, 1, 2),
        pair_specific_byte_indices=(2, 3, 4),  # overlaps shared at index 2
        mixed_byte_indices=(),
        aggregate_byte_pair_correlation_mean=0.5,
        n_bytes=10,
        n_pairs=8,
        correlation_threshold_high=0.8,
        correlation_threshold_low=0.2,
        archive_sha256="deadbeef" * 8,
        estimated_wyner_ziv_gain_bytes=3,
    )
    with pytest.raises(WynerZivAxisLevelReweightError, match="disjoint"):
        axis_level_reweight(c)


def test_negative_byte_index_in_classification_raises():
    """A negative byte index in any classification set is refused."""
    c = WynerZivSideInfoClassification(
        candidate_shared_prior_byte_indices=(-1,),
        pair_specific_byte_indices=(),
        mixed_byte_indices=(),
        aggregate_byte_pair_correlation_mean=0.5,
        n_bytes=10,
        n_pairs=8,
        correlation_threshold_high=0.8,
        correlation_threshold_low=0.2,
        archive_sha256="deadbeef" * 8,
        estimated_wyner_ziv_gain_bytes=0,
    )
    with pytest.raises(WynerZivAxisLevelReweightError, match="negative byte index"):
        axis_level_reweight(c)


def test_out_of_range_byte_index_in_classification_raises():
    """A byte index >= n_bytes in any classification set is refused."""
    c = WynerZivSideInfoClassification(
        candidate_shared_prior_byte_indices=(15,),
        pair_specific_byte_indices=(),
        mixed_byte_indices=(),
        aggregate_byte_pair_correlation_mean=0.5,
        n_bytes=10,
        n_pairs=8,
        correlation_threshold_high=0.8,
        correlation_threshold_low=0.2,
        archive_sha256="deadbeef" * 8,
        estimated_wyner_ziv_gain_bytes=0,
    )
    with pytest.raises(WynerZivAxisLevelReweightError, match="n_bytes"):
        axis_level_reweight(c)


# ──────────────────────────────────────────────────────────────────────────── #
# Real per-pair gradient tensor round-trip                                      #
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.mark.skipif(
    not REAL_VALIDATE_NPY.exists(),
    reason="real per-pair anchor not present (skip — fixture is workstation-local)",
)
def test_real_per_pair_gradient_round_trip():
    """End-to-end: real per-pair tensor → wyner_ziv classifier → reweight.

    Confirms the helper correctly processes the canonical fp64 8-pair
    validate fixture at .omx/tmp/master_gradient_per_pair_8pair_fp64_validate.npy
    (shape (178417, 8, 3); produced by `tools/extract_master_gradient.py`
    on Modal CPU per FEC6 master gradient extraction).
    """
    per_pair = np.load(REAL_VALIDATE_NPY)
    assert per_pair.ndim == 3 and per_pair.shape[-1] == 3
    n_bytes, n_pairs, _ = per_pair.shape

    classification = wyner_ziv_side_info_covariance(
        per_pair,
        archive_sha256="0" * 64,
        measurement_axis="pose",
        measurement_hardware="linux_x86_64_modal_cpu",
        sample_axis=1,
        write_sidecar=False,
    )
    assert classification.n_bytes == n_bytes
    assert classification.n_pairs == n_pairs

    weights = axis_level_reweight(classification)
    # Full byte coverage
    assert len(weights) == n_bytes
    # All weights finite + non-negative
    for v in weights.values():
        assert math.isfinite(v) and v >= 0.0
    # Shared-prior bytes (if any) sit at downweight value
    for idx in classification.candidate_shared_prior_byte_indices[:5]:
        assert weights[idx] == pytest.approx(SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT)
    # Pair-specific bytes (if any) sit at upweight value
    for idx in classification.pair_specific_byte_indices[:5]:
        assert weights[idx] == pytest.approx(PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT)


@pytest.mark.skipif(
    not REAL_VALIDATE_NPY.exists(),
    reason="real per-pair anchor not present (skip — fixture is workstation-local)",
)
def test_real_per_pair_round_trip_is_deterministic():
    """Running the round-trip twice on the same tensor produces identical weights."""
    per_pair = np.load(REAL_VALIDATE_NPY)
    classification = wyner_ziv_side_info_covariance(
        per_pair,
        archive_sha256="0" * 64,
        measurement_axis="pose",
        measurement_hardware="linux_x86_64_modal_cpu",
        sample_axis=1,
        write_sidecar=False,
    )
    w1 = axis_level_reweight(classification)
    w2 = axis_level_reweight(classification)
    assert w1 == w2
