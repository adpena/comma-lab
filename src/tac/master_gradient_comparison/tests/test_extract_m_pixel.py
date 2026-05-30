# SPDX-License-Identifier: MIT
"""Tests for Phase A canonical helper — ``extract_M_pixel`` + sister broadcast.

Verifies the canonical contract for per-pixel scorer-sensitivity extraction
from ``ContestGradientTensor`` / ``InflatedGradientTensor``, plus the
``broadcast_sensitivity_map_to_channels`` adapter that Z8 M7 Path B
consumes for ``(B, C, H, W)`` Protocol compatibility.

Per Catalog #287 evidence-tag discipline: every test claim is paired with
adjacent source/observed evidence. Per CLAUDE.md "Premise-verification-
before-edit" Catalog #229: tests verify the canonical math contract
empirically, not just structural shape.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tac.master_gradient import OperatingPoint
from tac.master_gradient_comparison.multi_granularity import (
    ContestGradientTensor,
    InflatedGradientTensor,
    LEGAL_PIXEL_REDUCTIONS,
    M_PIXEL_PROVENANCE_KIND,
    MultiGranularityComparisonError,
    PerPixelSensitivityMap,
    broadcast_sensitivity_map_to_channels,
    extract_M_pixel,
)


# ----------------------------------------------------------------------
# Fixtures: synthetic gradient tensors persisted to a temp dir.
# ----------------------------------------------------------------------


def _make_synthetic_contest_tensor(
    tmp_path: Path,
    *,
    n_pairs: int = 3,
    h: int = 4,
    w: int = 6,
    seed: int = 0,
) -> ContestGradientTensor:
    """Persist a synthetic (N, 3, H, W) gradient + return wrapper.

    Sister of the canonical extract_M_contest emission shape per
    ``ContestGradientTensor.shape()``. Axis 1 = (seg, pose, rate) per the
    canonical scorer-gradient layout.
    """
    rng = np.random.RandomState(seed)
    arr = rng.randn(n_pairs, 3, h, w).astype(np.float32)
    arr_path = tmp_path / "m_contest_synthetic.npy"
    np.save(arr_path, arr)
    import hashlib

    sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    return ContestGradientTensor(
        array_path=str(arr_path),
        array_sha256=sha,
        n_pairs=n_pairs,
        height=h,
        width=w,
        contest_video_sha256="a" * 64,
        scorer_seg_sha256="b" * 64,
        scorer_pose_sha256="c" * 64,
        operating_point=op,
        captured_at_utc="2026-05-30T00:00:00+00:00",
        measurement_axis="[predicted]",
    )


def _make_synthetic_inflated_tensor(
    tmp_path: Path,
    *,
    n_pairs: int = 2,
    h: int = 5,
    w: int = 7,
    seed: int = 1,
) -> InflatedGradientTensor:
    """Sister of ``_make_synthetic_contest_tensor`` for the inflated surface."""
    rng = np.random.RandomState(seed)
    arr = rng.randn(n_pairs, 3, h, w).astype(np.float32)
    arr_path = tmp_path / "m_inflated_synthetic.npy"
    np.save(arr_path, arr)
    import hashlib

    sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    return InflatedGradientTensor(
        array_path=str(arr_path),
        array_sha256=sha,
        n_pairs=n_pairs,
        height=h,
        width=w,
        inflated_video_sha256="d" * 64,
        archive_sha256="e" * 64,
        scorer_seg_sha256="f" * 64,
        scorer_pose_sha256="0" * 64,
        operating_point=op,
        captured_at_utc="2026-05-30T00:00:00+00:00",
        measurement_axis="[predicted]",
    )


# ----------------------------------------------------------------------
# Section A: Module-level constants + provenance kind.
# ----------------------------------------------------------------------


def test_provenance_kind_canonical_value() -> None:
    """The canonical M_PIXEL_PROVENANCE_KIND matches the helper's import path."""
    assert M_PIXEL_PROVENANCE_KIND == "tac.master_gradient_comparison.extract_M_pixel"


def test_legal_reductions_canonical_set() -> None:
    assert LEGAL_PIXEL_REDUCTIONS == frozenset({"l2_norm", "l1_norm", "max"})


# ----------------------------------------------------------------------
# Section B: PerPixelSensitivityMap dataclass invariants.
# ----------------------------------------------------------------------


def test_per_pixel_sensitivity_map_shape_contract(tmp_path: Path) -> None:
    """shape() returns (N_pairs, H, W) per dataclass docstring."""
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    result = extract_M_pixel(m_contest, reduction="l2_norm")
    assert isinstance(result, PerPixelSensitivityMap)
    assert result.shape() == (3, 4, 6)


def test_per_pixel_sensitivity_map_rejects_invalid_n_pairs(tmp_path: Path) -> None:
    arr_path = tmp_path / "tmp.npy"
    np.save(arr_path, np.zeros((1, 4, 6), dtype=np.float32))
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    with pytest.raises(MultiGranularityComparisonError, match="n_pairs"):
        PerPixelSensitivityMap(
            array_path=str(arr_path),
            array_sha256="x" * 64,
            n_pairs=0,
            height=4,
            width=6,
            source_video_sha256="a" * 64,
            source_kind="m_contest",
            reduction="l2_norm",
            operating_point=op,
            captured_at_utc="2026-05-30T00:00:00+00:00",
        )


def test_per_pixel_sensitivity_map_rejects_invalid_reduction(tmp_path: Path) -> None:
    arr_path = tmp_path / "tmp.npy"
    np.save(arr_path, np.zeros((1, 4, 6), dtype=np.float32))
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    with pytest.raises(MultiGradientGuardError := MultiGranularityComparisonError, match="reduction"):
        PerPixelSensitivityMap(
            array_path=str(arr_path),
            array_sha256="x" * 64,
            n_pairs=1,
            height=4,
            width=6,
            source_video_sha256="a" * 64,
            source_kind="m_contest",
            reduction="bogus",
            operating_point=op,
            captured_at_utc="2026-05-30T00:00:00+00:00",
        )


def test_per_pixel_sensitivity_map_rejects_invalid_source_kind(tmp_path: Path) -> None:
    arr_path = tmp_path / "tmp.npy"
    np.save(arr_path, np.zeros((1, 4, 6), dtype=np.float32))
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    with pytest.raises(MultiGranularityComparisonError, match="source_kind"):
        PerPixelSensitivityMap(
            array_path=str(arr_path),
            array_sha256="x" * 64,
            n_pairs=1,
            height=4,
            width=6,
            source_video_sha256="a" * 64,
            source_kind="bogus",
            reduction="l2_norm",
            operating_point=op,
            captured_at_utc="2026-05-30T00:00:00+00:00",
        )


def test_per_pixel_sensitivity_map_load_roundtrip(tmp_path: Path) -> None:
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    result = extract_M_pixel(m_contest, reduction="l2_norm")
    loaded = result.load()
    assert loaded.shape == (3, 4, 6)
    assert loaded.dtype == np.float32


# ----------------------------------------------------------------------
# Section C: extract_M_pixel math + persistence + provenance.
# ----------------------------------------------------------------------


def test_extract_M_pixel_l2_norm_canonical_math(tmp_path: Path) -> None:
    """L2 norm reduction matches sqrt(sum(M^2)) across axis 1 EXACTLY.

    This is the Fridrich UNIWARD-analog canonical scorer-blindness inverse
    per the module docstring.
    """
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    result = extract_M_pixel(m_contest, reduction="l2_norm")

    # Math witness: recompute from the source array directly.
    src = m_contest.load()
    expected = np.sqrt(np.sum(src.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    got = result.load()
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_extract_M_pixel_l1_norm_canonical_math(tmp_path: Path) -> None:
    """L1 norm matches sum(|M|) across axis 1."""
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    result = extract_M_pixel(m_contest, reduction="l1_norm")
    src = m_contest.load()
    expected = np.sum(np.abs(src.astype(np.float64)), axis=1).astype(np.float32)
    np.testing.assert_allclose(result.load(), expected, rtol=1e-6, atol=1e-7)


def test_extract_M_pixel_max_canonical_math(tmp_path: Path) -> None:
    """Max reduction matches max(|M|) across axis 1 — conservative dominant-axis bound."""
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    result = extract_M_pixel(m_contest, reduction="max")
    src = m_contest.load()
    expected = np.max(np.abs(src.astype(np.float64)), axis=1).astype(np.float32)
    np.testing.assert_allclose(result.load(), expected, rtol=1e-6, atol=1e-7)


def test_extract_M_pixel_all_reductions_non_negative(tmp_path: Path) -> None:
    """All reductions must yield non-negative weights — the Yousfi cost-map invariant."""
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    for reduction in ("l2_norm", "l1_norm", "max"):
        result = extract_M_pixel(m_contest, reduction=reduction)
        loaded = result.load()
        assert (loaded >= 0).all(), f"{reduction} produced negative values"


def test_extract_M_pixel_inflated_source_kind(tmp_path: Path) -> None:
    """Polymorphic on InflatedGradientTensor; source_kind reflects choice."""
    m_inflated = _make_synthetic_inflated_tensor(tmp_path)
    result = extract_M_pixel(m_inflated, reduction="l2_norm")
    assert result.source_kind == "m_inflated"
    assert result.source_video_sha256 == "d" * 64
    assert result.shape() == (2, 5, 7)


def test_extract_M_pixel_contest_source_kind(tmp_path: Path) -> None:
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    result = extract_M_pixel(m_contest, reduction="l2_norm")
    assert result.source_kind == "m_contest"
    assert result.source_video_sha256 == "a" * 64


def test_extract_M_pixel_rejects_invalid_reduction(tmp_path: Path) -> None:
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    with pytest.raises(MultiGranularityComparisonError, match="reduction"):
        extract_M_pixel(m_contest, reduction="bogus")


def test_extract_M_pixel_rejects_non_tensor_input(tmp_path: Path) -> None:
    with pytest.raises(MultiGranularityComparisonError, match="must be ContestGradientTensor"):
        extract_M_pixel("not_a_tensor")  # type: ignore[arg-type]


def test_extract_M_pixel_persists_npy_and_meta(tmp_path: Path) -> None:
    """Result is persisted to _PERSIST_ROOT with a sidecar .meta.json."""
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    cache_path = tmp_path / "m_pixel_test.npy"
    result = extract_M_pixel(m_contest, reduction="l2_norm", cache_path=cache_path)
    assert cache_path.exists()
    meta_path = cache_path.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    # Provenance discipline per Catalog #323:
    assert meta["canonical_helper_invocation"] == M_PIXEL_PROVENANCE_KIND
    assert meta["schema"] == "m_pixel_meta_v1"
    assert meta["source_kind"] == "m_contest"
    assert meta["reduction"] == "l2_norm"
    assert meta["array_sha256"] == result.array_sha256


def test_extract_M_pixel_persists_under_canonical_root_by_default(tmp_path: Path, monkeypatch) -> None:
    """When cache_path is None, persistence routes under _PERSIST_ROOT (never /tmp)."""
    # We don't want to litter the live state dir — use chdir to tmp_path so
    # the relative _PERSIST_ROOT resolves there.
    import os

    monkeypatch.chdir(tmp_path)
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    result = extract_M_pixel(m_contest, reduction="l2_norm")
    # Per Catalog #245 sister discipline: never /tmp.
    assert "/tmp/" not in result.array_path
    assert ".omx/state/master_gradient_comparison" in result.array_path


def test_extract_M_pixel_measurement_axis_is_predicted(tmp_path: Path) -> None:
    """Per Catalog #192 + #317: the map is NEVER promotable; tag is [predicted]."""
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    result = extract_M_pixel(m_contest, reduction="l2_norm")
    assert result.measurement_axis == "[predicted]"


# ----------------------------------------------------------------------
# Section D: broadcast_sensitivity_map_to_channels adapter (Z8 M7 Path B).
# ----------------------------------------------------------------------


def test_broadcast_shape_default(tmp_path: Path) -> None:
    """Default batch_size=1 + num_channels=3 → (1, 3, H, W)."""
    arr = np.random.RandomState(0).uniform(0, 5, size=(2, 8, 12)).astype(np.float32)
    out = broadcast_sensitivity_map_to_channels(arr)
    assert out.shape == (1, 3, 8, 12)


def test_broadcast_shape_custom_batch_and_channels() -> None:
    arr = np.random.RandomState(0).uniform(0, 5, size=(2, 4, 6)).astype(np.float32)
    out = broadcast_sensitivity_map_to_channels(arr, batch_size=4, num_channels=1)
    assert out.shape == (4, 1, 4, 6)


def test_broadcast_channel_uniformity_fridrich_invariant() -> None:
    """All channels at a given pixel get the SAME weight per Fridrich UNIWARD
    per-spatial-location cost convention."""
    arr = np.random.RandomState(0).uniform(0, 5, size=(2, 4, 6)).astype(np.float32)
    out = broadcast_sensitivity_map_to_channels(arr, batch_size=3, num_channels=5)
    # For every (b, h, w), all C values must be equal.
    for b in range(3):
        for c in range(1, 5):
            np.testing.assert_array_equal(out[b, c], out[b, 0])


def test_broadcast_pair_aggregation_first(tmp_path: Path) -> None:
    """'first' aggregation returns pair 0's map verbatim per docstring."""
    arr = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],  # pair 0
            [[100.0, 200.0], [300.0, 400.0]],  # pair 1 (huge values)
        ],
        dtype=np.float32,
    )
    out = broadcast_sensitivity_map_to_channels(arr, batch_size=1, num_channels=1, pair_aggregation="first")
    # Should match pair 0 exactly, not pair 1.
    np.testing.assert_array_equal(out[0, 0], arr[0])


def test_broadcast_pair_aggregation_mean(tmp_path: Path) -> None:
    arr = np.array(
        [
            [[2.0, 4.0]],
            [[6.0, 8.0]],
        ],
        dtype=np.float32,
    )
    out = broadcast_sensitivity_map_to_channels(arr, batch_size=1, num_channels=1, pair_aggregation="mean")
    np.testing.assert_array_equal(out[0, 0], np.array([[4.0, 6.0]], dtype=np.float32))


def test_broadcast_pair_aggregation_max() -> None:
    arr = np.array(
        [
            [[2.0, 8.0]],
            [[6.0, 4.0]],
        ],
        dtype=np.float32,
    )
    out = broadcast_sensitivity_map_to_channels(arr, batch_size=1, num_channels=1, pair_aggregation="max")
    np.testing.assert_array_equal(out[0, 0], np.array([[6.0, 8.0]], dtype=np.float32))


def test_broadcast_rejects_invalid_aggregation() -> None:
    arr = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(MultiGranularityComparisonError, match="pair_aggregation"):
        broadcast_sensitivity_map_to_channels(arr, pair_aggregation="bogus")


def test_broadcast_rejects_invalid_batch_size() -> None:
    arr = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(MultiGranularityComparisonError, match="batch_size"):
        broadcast_sensitivity_map_to_channels(arr, batch_size=0)


def test_broadcast_rejects_invalid_num_channels() -> None:
    arr = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(MultiGranularityComparisonError, match="num_channels"):
        broadcast_sensitivity_map_to_channels(arr, num_channels=-1)


def test_broadcast_rejects_wrong_ndim() -> None:
    arr = np.zeros((4,), dtype=np.float32)
    with pytest.raises(MultiGranularityComparisonError, match="shape"):
        broadcast_sensitivity_map_to_channels(arr)


def test_broadcast_accepts_persistivity_map_object(tmp_path: Path) -> None:
    """Polymorphic input: accepts both raw array AND PerPixelSensitivityMap."""
    m_contest = _make_synthetic_contest_tensor(tmp_path)
    sensitivity = extract_M_pixel(m_contest, reduction="l2_norm")
    out = broadcast_sensitivity_map_to_channels(sensitivity, batch_size=2, num_channels=3)
    assert out.shape == (2, 3, m_contest.height, m_contest.width)
    assert (out >= 0).all()


def test_broadcast_custom_dtype() -> None:
    arr = np.ones((2, 4, 6), dtype=np.float32)
    out = broadcast_sensitivity_map_to_channels(arr, dtype=np.float64)
    assert out.dtype == np.float64


# ----------------------------------------------------------------------
# Section E: Z8 M8 ScoreAwareLevelLoss Protocol invariant witness.
# ----------------------------------------------------------------------


def test_uniform_sensitivity_reduces_loss_to_l2_invariant(tmp_path: Path) -> None:
    """The M8 Protocol invariant per binding_contract.py:467-470:
    when sensitivity_map == 1 everywhere, the weighted-L2 loss reduces to
    standard L2. Z8 M7 Path A's uniform map already satisfies this; this
    test verifies broadcast preserves it.
    """
    uniform_pixel_map = np.ones((1, 8, 12), dtype=np.float32)
    sensitivity = broadcast_sensitivity_map_to_channels(
        uniform_pixel_map, batch_size=1, num_channels=3
    )
    rng = np.random.RandomState(0)
    reconstruction = rng.randn(1, 3, 8, 12).astype(np.float32)
    target = rng.randn(1, 3, 8, 12).astype(np.float32)
    weighted_l2 = (sensitivity * (reconstruction - target) ** 2).sum()
    standard_l2 = ((reconstruction - target) ** 2).sum()
    assert np.isclose(weighted_l2, standard_l2)


# ----------------------------------------------------------------------
# Phase C tests (decompose_M_contest_per_level + MallatDyadicMismatchError)
# 2026-05-30 — Yousfi-ordered cap=1-per-turn cascade per CLAUDE.md
# "Forbidden premature KILL without research exhaustion" + Rao-Ballard
# 1999 hierarchical predictive coding mean-pooling canonical.
# ----------------------------------------------------------------------


def test_decompose_per_level_provenance_kind_canonical_value() -> None:
    """Per Catalog #323 canonical Provenance umbrella + Catalog #335 contract."""
    from tac.master_gradient_comparison.multi_granularity import (
        M_CONTEST_PER_LEVEL_PROVENANCE_KIND,
    )

    assert M_CONTEST_PER_LEVEL_PROVENANCE_KIND == (
        "tac.master_gradient_comparison.decompose_M_contest_per_level"
    )


def test_decompose_per_level_legal_reductions_canonical_set() -> None:
    """The 3 canonical Mallat dyadic projection reductions are pinned."""
    from tac.master_gradient_comparison.multi_granularity import (
        LEGAL_LEVEL_PROJECTION_REDUCTIONS,
    )

    assert LEGAL_LEVEL_PROJECTION_REDUCTIONS == frozenset({"mean", "max", "sum"})


def test_decompose_per_level_mallat_dyadic_error_is_notimplementederror() -> None:
    """Sister of Phase D's ResolutionMismatchAwaitingPerLevelProjectionError.

    Per CLAUDE.md "Forbidden premature KILL" — honest deferral via
    NotImplementedError subclass naming Phase B as the reactivation
    criterion for non-dyadic / non-uniform-stride projections.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        MallatDyadicMismatchError,
    )

    assert issubclass(MallatDyadicMismatchError, NotImplementedError)


def test_decompose_per_level_identity_short_circuit(tmp_path: Path) -> None:
    """Identity projection (level_shape == native shape) returns input unchanged.

    Sister of the M8 Protocol identity-resolution invariant — no copy,
    no new sidecar, no provenance chain perturbation.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    result = decompose_M_contest_per_level(m_pixel, level_shape=(8, 12))
    # Identity short-circuit: same object returned (no copy).
    assert result is m_pixel


def test_decompose_per_level_mean_canonical_math(tmp_path: Path) -> None:
    """Rao-Ballard 1999 mean-pooling canonical default.

    Each coarse cell is the MEAN of its dyadic block. Verified against
    explicit numpy block-reduce; sister-checks against the canonical
    Mallat dyadic stride (uniform 2x in both H and W).
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    m_native = m_pixel.load()  # (2, 8, 12)
    result = decompose_M_contest_per_level(
        m_pixel, level_shape=(4, 6), reduction="mean"
    )
    assert result.shape() == (2, 4, 6)
    # Canonical mean over 2x2 dyadic blocks.
    expected = m_native.reshape(2, 4, 2, 6, 2).mean(axis=(2, 4))
    np.testing.assert_allclose(result.load(), expected.astype(np.float32), rtol=1e-6)


def test_decompose_per_level_max_canonical_math(tmp_path: Path) -> None:
    """Per Fridrich UNIWARD conservative-bound convention.

    Max-pooling preserves UNIWARD-style high-sensitivity locations per
    the canonical scorer-blindness inverse interpretation.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    m_native = m_pixel.load()
    result = decompose_M_contest_per_level(m_pixel, level_shape=(4, 6), reduction="max")
    expected = m_native.reshape(2, 4, 2, 6, 2).max(axis=(2, 4))
    np.testing.assert_allclose(result.load(), expected.astype(np.float32), rtol=1e-6)


def test_decompose_per_level_sum_canonical_math(tmp_path: Path) -> None:
    """Sum-pooling preserves total contribution per coarse cell.

    Proportional to mean by stride^2 — useful when total per-cell
    sensitivity matters more than average.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    m_native = m_pixel.load()
    result = decompose_M_contest_per_level(m_pixel, level_shape=(4, 6), reduction="sum")
    expected = m_native.reshape(2, 4, 2, 6, 2).sum(axis=(2, 4))
    np.testing.assert_allclose(result.load(), expected.astype(np.float32), rtol=1e-6)


def test_decompose_per_level_invalid_reduction_rejected(tmp_path: Path) -> None:
    """Reductions outside LEGAL_LEVEL_PROJECTION_REDUCTIONS raise."""
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    with pytest.raises(MultiGranularityComparisonError, match="reduction must be"):
        decompose_M_contest_per_level(
            m_pixel, level_shape=(4, 6), reduction="bogus_reduction"
        )


def test_decompose_per_level_invalid_level_shape_rejected(tmp_path: Path) -> None:
    """Malformed level_shape (non-tuple / wrong arity / non-int / non-positive) raises."""
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    with pytest.raises(MultiGranularityComparisonError, match="level_shape must be"):
        decompose_M_contest_per_level(m_pixel, level_shape=(4, 6, 1))  # type: ignore[arg-type]
    with pytest.raises(MultiGranularityComparisonError, match="level_shape must be"):
        decompose_M_contest_per_level(m_pixel, level_shape=[4, 6])  # type: ignore[arg-type]
    with pytest.raises(MultiGranularityComparisonError, match="must be positive"):
        decompose_M_contest_per_level(m_pixel, level_shape=(0, 6))
    with pytest.raises(MultiGranularityComparisonError, match="must be positive"):
        decompose_M_contest_per_level(m_pixel, level_shape=(4, -1))


def test_decompose_per_level_non_dyadic_raises_mallat_mismatch(tmp_path: Path) -> None:
    """Non-integer-divisible projections raise MallatDyadicMismatchError.

    Per CLAUDE.md "Forbidden premature KILL" honest deferral: Phase C
    only supports Mallat dyadic; non-dyadic raises with explicit Phase B
    reactivation criterion.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        MallatDyadicMismatchError,
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    # (8, 12) -> (3, 5): non-integer-divisible.
    with pytest.raises(MallatDyadicMismatchError, match="integer-divisible"):
        decompose_M_contest_per_level(m_pixel, level_shape=(3, 5))


def test_decompose_per_level_non_uniform_stride_raises_mallat_mismatch(
    tmp_path: Path,
) -> None:
    """Non-uniform stride (e.g. 384x512 -> 192x128 with stride_h=2 != stride_w=4)
    raises MallatDyadicMismatchError per the canonical Mallat dyadic invariant.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        MallatDyadicMismatchError,
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    # (8, 12) -> (4, 3): stride_h=2 != stride_w=4.
    with pytest.raises(MallatDyadicMismatchError, match="uniform stride"):
        decompose_M_contest_per_level(m_pixel, level_shape=(4, 3))


def test_decompose_per_level_canonical_z8_mallat_hierarchy(tmp_path: Path) -> None:
    """Canonical Z8 Mallat dyadic hierarchy: 384x512 -> 192x256 -> ... -> 24x32.

    Sister of the canonical contest video resolution per CLAUDE.md
    masks contract. Each level halves both dims; uniform stride preserved.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    # Smaller proxy for the 384x512 contest video (preserves dyadic structure
    # at lower compute: 32x48 -> 16x24 -> 8x12 -> 4x6).
    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=32, w=48)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    # Level 1 (1/2 stride).
    l1 = decompose_M_contest_per_level(m_pixel, level_shape=(16, 24))
    assert l1.shape() == (1, 16, 24)
    # Level 2 (1/4 stride).
    l2 = decompose_M_contest_per_level(m_pixel, level_shape=(8, 12))
    assert l2.shape() == (1, 8, 12)
    # Level 3 (1/8 stride).
    l3 = decompose_M_contest_per_level(m_pixel, level_shape=(4, 6))
    assert l3.shape() == (1, 4, 6)


def test_decompose_per_level_provenance_chain_preserves_source(tmp_path: Path) -> None:
    """Source video sha + source_kind + scorer-axis reduction + operating
    point + measurement_axis preserved through projection. Sister
    .meta.json sidecar carries predecessor_array_sha256 forensic chain.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    result = decompose_M_contest_per_level(
        m_pixel, level_shape=(4, 6), reduction="mean"
    )

    assert result.source_video_sha256 == m_pixel.source_video_sha256
    assert result.source_kind == m_pixel.source_kind  # "m_contest"
    assert result.reduction == m_pixel.reduction  # "l2_norm" preserved
    assert result.operating_point == m_pixel.operating_point
    assert result.measurement_axis == m_pixel.measurement_axis  # "[predicted]"
    assert result.array_sha256 != m_pixel.array_sha256  # NEW bytes after projection

    # Sidecar provenance chain via .meta.json
    meta_path = Path(result.array_path).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["schema"] == "m_pixel_per_level_meta_v1"
    assert meta["canonical_helper_invocation"] == (
        "tac.master_gradient_comparison.decompose_M_contest_per_level"
    )
    assert meta["predecessor_array_sha256"] == m_pixel.array_sha256
    assert meta["level_projection_reduction"] == "mean"
    assert meta["native_shape"] == [8, 12]
    assert meta["level_shape"] == [4, 6]
    assert meta["mallat_dyadic_stride"] == 2


def test_decompose_per_level_non_negativity_preserved(tmp_path: Path) -> None:
    """All Phase A reductions produce non-negative maps; Phase C projection
    (mean/max/sum of non-negative inputs) preserves non-negativity.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    for axis_reduction in ("l2_norm", "l1_norm", "max"):
        m_pixel = extract_M_pixel(gt, reduction=axis_reduction)
        for projection_reduction in ("mean", "max", "sum"):
            result = decompose_M_contest_per_level(
                m_pixel, level_shape=(4, 6), reduction=projection_reduction
            )
            arr = result.load()
            assert (arr >= 0).all(), (
                f"non-negativity violated for "
                f"axis={axis_reduction}, projection={projection_reduction}"
            )


def test_decompose_per_level_uniform_input_preserves_uniformity(tmp_path: Path) -> None:
    """M8 Protocol invariant sister: uniform input map produces uniform output
    map (every coarse cell equals the uniform constant for mean / max; for
    sum equals constant * stride^2). This sister-of-M8-uniform-invariant
    holds the canonical Path A reduction-to-standard-L2 contract intact
    through Phase C projection.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
    )

    # Build a uniform-1.0 PerPixelSensitivityMap directly (bypassing
    # extract_M_pixel which produces data-dependent values).
    n_pairs, h, w = 2, 8, 12
    uniform_arr = np.ones((n_pairs, h, w), dtype=np.float32)
    uniform_path = tmp_path / "uniform.npy"
    np.save(uniform_path, uniform_arr)
    import hashlib

    uniform_sha = hashlib.sha256(
        np.ascontiguousarray(uniform_arr).tobytes()
    ).hexdigest()
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    uniform_map = PerPixelSensitivityMap(
        array_path=str(uniform_path),
        array_sha256=uniform_sha,
        n_pairs=n_pairs,
        height=h,
        width=w,
        source_video_sha256="d" * 64,
        source_kind="m_contest",
        reduction="l2_norm",
        operating_point=op,
        captured_at_utc="2026-05-30T00:00:00+00:00",
        measurement_axis="[predicted]",
    )

    result_mean = decompose_M_contest_per_level(
        uniform_map, level_shape=(4, 6), reduction="mean"
    )
    assert np.allclose(result_mean.load(), 1.0)
    result_max = decompose_M_contest_per_level(
        uniform_map, level_shape=(4, 6), reduction="max"
    )
    assert np.allclose(result_max.load(), 1.0)
    result_sum = decompose_M_contest_per_level(
        uniform_map, level_shape=(4, 6), reduction="sum"
    )
    # stride^2 = 2x2 = 4.
    assert np.allclose(result_sum.load(), 4.0)


# ----------------------------------------------------------------------
# Section J: Phase B — decompose_M_contest_per_subband
#
# Canonical 2D separable Daubechies wavelet decomposition per Mallat 1989
# §7.7. Sister of Phase C (per-level dyadic projection); where Phase C
# produces a single (N, H_level, W_level) tensor, Phase B produces 4
# subbands {LL, LH, HL, HH} per the canonical Daubechies multi-resolution
# analysis.
#
# Test discipline per Catalog #229 (premise-verification-before-edit):
# each test is paired with an explicit canonical identity OR shape contract
# the helper must satisfy. Per Catalog #287: tests verify behavior, not
# just structural existence.
# ----------------------------------------------------------------------


def test_phase_b_provenance_kind_canonical_value() -> None:
    """Canonical provenance kind matches the helper import path."""
    from tac.master_gradient_comparison.multi_granularity import (
        M_CONTEST_PER_SUBBAND_PROVENANCE_KIND,
    )

    assert M_CONTEST_PER_SUBBAND_PROVENANCE_KIND == (
        "tac.master_gradient_comparison.decompose_M_contest_per_subband"
    )


def test_phase_b_legal_wavelet_families_canonical_set() -> None:
    """Pinned canonical wavelet families per Daubechies 1992 + Haar."""
    from tac.master_gradient_comparison.multi_granularity import (
        LEGAL_WAVELET_FAMILIES,
    )

    assert LEGAL_WAVELET_FAMILIES == frozenset({"db1", "db2", "db4", "haar"})


def test_phase_b_legal_subband_reductions_canonical_set() -> None:
    """Pinned canonical subband reductions: abs (L1), square (L2), magnitude (alias)."""
    from tac.master_gradient_comparison.multi_granularity import (
        LEGAL_SUBBAND_REDUCTIONS,
    )

    assert LEGAL_SUBBAND_REDUCTIONS == frozenset({"abs", "square", "magnitude"})


def test_phase_b_wavelet_decomposition_error_is_multi_granularity_error() -> None:
    """WaveletDecompositionError inherits from MultiGranularityComparisonError so
    callers catching the broader contract-violation class capture Phase B failures.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        WaveletDecompositionError,
    )

    assert issubclass(WaveletDecompositionError, MultiGranularityComparisonError)


def test_phase_b_level_zero_rejected(tmp_path: Path) -> None:
    """level=0 is identity per Phase C; Phase B requires level >= 1 by contract."""
    from tac.master_gradient_comparison.multi_granularity import (
        WaveletDecompositionError,
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    with pytest.raises(WaveletDecompositionError, match="level must be >= 1"):
        decompose_M_contest_per_subband(m_pixel, level=0, cache_path=tmp_path)


def test_phase_b_negative_level_rejected(tmp_path: Path) -> None:
    from tac.master_gradient_comparison.multi_granularity import (
        WaveletDecompositionError,
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    with pytest.raises(WaveletDecompositionError, match="level must be >= 1"):
        decompose_M_contest_per_subband(m_pixel, level=-1, cache_path=tmp_path)


def test_phase_b_unknown_wavelet_rejected(tmp_path: Path) -> None:
    from tac.master_gradient_comparison.multi_granularity import (
        WaveletDecompositionError,
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    with pytest.raises(WaveletDecompositionError, match="wavelet must be one of"):
        decompose_M_contest_per_subband(
            m_pixel, level=1, wavelet="biorthogonal", cache_path=tmp_path
        )


def test_phase_b_unknown_subband_reduction_rejected(tmp_path: Path) -> None:
    from tac.master_gradient_comparison.multi_granularity import (
        WaveletDecompositionError,
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    with pytest.raises(
        WaveletDecompositionError, match="subband_reduction must be one of"
    ):
        decompose_M_contest_per_subband(
            m_pixel,
            level=1,
            subband_reduction="rectified_linear",
            cache_path=tmp_path,
        )


def test_phase_b_odd_spatial_dim_rejected(tmp_path: Path) -> None:
    """Daubechies 1-level DWT requires even axis length per Mallat §7.5."""
    from tac.master_gradient_comparison.multi_granularity import (
        WaveletDecompositionError,
        decompose_M_contest_per_subband,
    )

    # (8, 9) is odd along W; level=1 cannot decompose.
    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=9)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    with pytest.raises(WaveletDecompositionError, match="even spatial dims"):
        decompose_M_contest_per_subband(m_pixel, level=1, cache_path=tmp_path)


def test_phase_b_level_2_requires_even_at_level_2(tmp_path: Path) -> None:
    """Level 2 needs even spatial dims at level 1 (i.e., dims div by 4)."""
    from tac.master_gradient_comparison.multi_granularity import (
        WaveletDecompositionError,
        decompose_M_contest_per_subband,
    )

    # (10, 12): level 1 yields (5, 6); 5 is odd so level 2 cannot decompose.
    # Need to construct a map with non-Phase-A path because Phase A enforces
    # n_pairs * h * w shape; use the raw PerPixelSensitivityMap dataclass.
    n_pairs, h, w = 1, 10, 12
    arr = np.ones((n_pairs, h, w), dtype=np.float32)
    arr_path = tmp_path / "level2_odd.npy"
    np.save(arr_path, arr)
    import hashlib

    sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    m_pixel = PerPixelSensitivityMap(
        array_path=str(arr_path),
        array_sha256=sha,
        n_pairs=n_pairs,
        height=h,
        width=w,
        source_video_sha256="d" * 64,
        source_kind="m_contest",
        reduction="l2_norm",
        operating_point=op,
        captured_at_utc="2026-05-30T00:00:00+00:00",
        measurement_axis="[predicted]",
    )
    with pytest.raises(WaveletDecompositionError, match="even spatial dims"):
        decompose_M_contest_per_subband(m_pixel, level=2, cache_path=tmp_path)


def test_phase_b_canonical_db2_default_produces_4_subbands(tmp_path: Path) -> None:
    """Canonical default wavelet is db2; produces 4 subbands at (N, H/2, W/2)."""
    from tac.master_gradient_comparison.multi_granularity import (
        SubbandSensitivityDecomposition,
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    decomp = decompose_M_contest_per_subband(m_pixel, level=1, cache_path=tmp_path)

    assert isinstance(decomp, SubbandSensitivityDecomposition)
    assert decomp.wavelet_family == "db2"
    assert decomp.level == 1
    assert decomp.subband_reduction == "abs"
    expected_shape = (2, 4, 6)
    assert decomp.approximation.shape() == expected_shape
    assert decomp.detail_horizontal.shape() == expected_shape
    assert decomp.detail_vertical.shape() == expected_shape
    assert decomp.detail_diagonal.shape() == expected_shape
    assert decomp.subband_shape() == expected_shape


def test_phase_b_all_subbands_non_negative_across_reductions(tmp_path: Path) -> None:
    """All 3 subband_reduction modes produce non-negative scorer-sensitivity maps
    (Yousfi/Fridrich-grounded contract per Phase A docstring).
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")

    for reduction in ("abs", "square", "magnitude"):
        decomp = decompose_M_contest_per_subband(
            m_pixel,
            level=1,
            subband_reduction=reduction,
            cache_path=tmp_path,
        )
        for name, sub in (
            ("LL", decomp.approximation),
            ("LH", decomp.detail_horizontal),
            ("HL", decomp.detail_vertical),
            ("HH", decomp.detail_diagonal),
        ):
            arr = sub.load()
            assert (arr >= 0).all(), (
                f"non-negativity violated: subband={name}, reduction={reduction}"
            )


def test_phase_b_canonical_z8_mallat_hierarchy_levels(tmp_path: Path) -> None:
    """Canonical Z8 Mallat dyadic hierarchy: 32x48 -> 16x24 -> 8x12 -> 4x6.

    Smaller proxy for the 384x512 contest video (preserves dyadic structure
    at lower compute). Each level halves both dims.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=32, w=48)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")

    # Level 1: 32x48 -> 16x24 per subband
    l1 = decompose_M_contest_per_subband(m_pixel, level=1, cache_path=tmp_path)
    assert l1.subband_shape() == (1, 16, 24)

    # Level 2: 32x48 -> 8x12 per subband
    l2 = decompose_M_contest_per_subband(m_pixel, level=2, cache_path=tmp_path)
    assert l2.subband_shape() == (1, 8, 12)

    # Level 3: 32x48 -> 4x6 per subband
    l3 = decompose_M_contest_per_subband(m_pixel, level=3, cache_path=tmp_path)
    assert l3.subband_shape() == (1, 4, 6)


def test_phase_b_provenance_chain_preserves_source_metadata(tmp_path: Path) -> None:
    """All 4 subbands preserve source video sha + source_kind + reduction +
    operating point + measurement_axis. .meta.json sidecar carries
    predecessor_array_sha256 forensic chain per Catalog #323.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=2, h=8, w=12)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    decomp = decompose_M_contest_per_subband(
        m_pixel, level=1, wavelet="db2", cache_path=tmp_path
    )

    for name, sub in (
        ("LL", decomp.approximation),
        ("LH", decomp.detail_horizontal),
        ("HL", decomp.detail_vertical),
        ("HH", decomp.detail_diagonal),
    ):
        assert sub.source_video_sha256 == m_pixel.source_video_sha256
        assert sub.source_kind == m_pixel.source_kind
        assert sub.reduction == m_pixel.reduction
        assert sub.operating_point == m_pixel.operating_point
        assert sub.measurement_axis == m_pixel.measurement_axis
        # Each subband has its OWN array_sha256 (new bytes after decomp).
        assert sub.array_sha256 != m_pixel.array_sha256, name
        # Sister sidecar provenance.
        meta_path = Path(sub.array_path).with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        assert meta["schema"] == "m_pixel_per_subband_meta_v1"
        assert meta["canonical_helper_invocation"] == (
            "tac.master_gradient_comparison.decompose_M_contest_per_subband"
        )
        assert meta["predecessor_array_sha256"] == m_pixel.array_sha256
        assert meta["wavelet_family"] == "db2"
        assert meta["level"] == 1
        assert meta["subband_reduction"] == "abs"
        assert meta["native_shape"] == [8, 12]
        assert meta["subband_shape"] == [4, 6]


def test_phase_b_meta_json_schema_pinned(tmp_path: Path) -> None:
    """The sidecar schema string is the canonical machine-readable contract."""
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    decomp = decompose_M_contest_per_subband(m_pixel, level=1, cache_path=tmp_path)
    meta = json.loads(Path(decomp.approximation.array_path).with_suffix(".meta.json").read_text())
    # Canonical schema literal for downstream consumers.
    assert meta["schema"] == "m_pixel_per_subband_meta_v1"


def test_phase_b_haar_ll_identity_vs_phase_c_mean(tmp_path: Path) -> None:
    """Sister-check Phase B vs Phase C identity for Haar wavelet.

    For orthonormal Haar (db1): LL coefficient at coarse cell (h, w) equals
    ``(input[2h,2w]+input[2h,2w+1]+input[2h+1,2w]+input[2h+1,2w+1]) / 2``
    per Mallat §7.5. Therefore ``Phase_B_LL_haar = 2 * Phase_C_mean``
    exactly (for non-negative input where ``subband_reduction='abs'`` is
    a no-op on the LL band).

    Empirical verification anchor for the Phase B canonical contract: the
    Haar wavelet's analytically-known LL identity must hold to fp32 numerical
    precision when the canonical implementation is correct.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_level,
        decompose_M_contest_per_subband,
    )

    # Use a non-negative random map (Phase A maps are non-negative by construction).
    n_pairs, h, w = 1, 8, 8
    rng = np.random.RandomState(42)
    rand_arr = rng.rand(n_pairs, h, w).astype(np.float32)
    rand_path = tmp_path / "rand_haar.npy"
    np.save(rand_path, rand_arr)
    import hashlib

    sha = hashlib.sha256(np.ascontiguousarray(rand_arr).tobytes()).hexdigest()
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    m_pixel = PerPixelSensitivityMap(
        array_path=str(rand_path),
        array_sha256=sha,
        n_pairs=n_pairs,
        height=h,
        width=w,
        source_video_sha256="d" * 64,
        source_kind="m_contest",
        reduction="l2_norm",
        operating_point=op,
        captured_at_utc="2026-05-30T00:00:00+00:00",
        measurement_axis="[predicted]",
    )

    decomp_haar = decompose_M_contest_per_subband(
        m_pixel, level=1, wavelet="db1", cache_path=tmp_path
    )
    phase_c_mean = decompose_M_contest_per_level(
        m_pixel, level_shape=(h // 2, w // 2), reduction="mean"
    )

    ll = decomp_haar.approximation.load()
    mean_c = phase_c_mean.load()
    # Canonical Haar identity: LL = 2 * mean (orthonormal Haar low-pass).
    np.testing.assert_allclose(ll, 2.0 * mean_c, atol=1e-5)


def test_phase_b_haar_alias_equivalent_to_db1(tmp_path: Path) -> None:
    """The 'haar' wavelet name is a canonical alias for 'db1'."""
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")

    decomp_db1 = decompose_M_contest_per_subband(
        m_pixel, level=1, wavelet="db1", cache_path=tmp_path
    )
    decomp_haar = decompose_M_contest_per_subband(
        m_pixel, level=1, wavelet="haar", cache_path=tmp_path
    )
    # Different wavelet_family stored (alias preserved for operator
    # readability), but actual LL bytes equal.
    np.testing.assert_allclose(
        decomp_db1.approximation.load(),
        decomp_haar.approximation.load(),
        atol=1e-7,
    )


def test_phase_b_parseval_energy_preservation_orthonormal(tmp_path: Path) -> None:
    """Per Mallat §7.5 + Daubechies 1988: orthonormal Daubechies wavelets
    preserve total energy. ``sum(LL² + LH² + HL² + HH²) == sum(input²)`` to
    fp64 numerical precision (atol ~1e-6 at fp32 storage).

    This is the canonical near-Parseval cross-check between Phase B's 4
    subbands and the input signal energy.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    # Use random non-negative input; energy preservation holds for ANY input.
    n_pairs, h, w = 1, 8, 8
    rng = np.random.RandomState(123)
    rand_arr = rng.rand(n_pairs, h, w).astype(np.float32)
    rand_path = tmp_path / "rand_parseval.npy"
    np.save(rand_path, rand_arr)
    import hashlib

    sha = hashlib.sha256(np.ascontiguousarray(rand_arr).tobytes()).hexdigest()
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    m_pixel = PerPixelSensitivityMap(
        array_path=str(rand_path),
        array_sha256=sha,
        n_pairs=n_pairs,
        height=h,
        width=w,
        source_video_sha256="d" * 64,
        source_kind="m_contest",
        reduction="l2_norm",
        operating_point=op,
        captured_at_utc="2026-05-30T00:00:00+00:00",
        measurement_axis="[predicted]",
    )

    # Verify across db2 and db4 (both orthonormal); subband_reduction="square"
    # IS the per-coefficient energy.
    for wavelet in ("db1", "db2", "db4"):
        decomp = decompose_M_contest_per_subband(
            m_pixel,
            level=1,
            wavelet=wavelet,
            subband_reduction="square",
            cache_path=tmp_path,
        )
        energy_subbands = (
            decomp.approximation.load().sum()
            + decomp.detail_horizontal.load().sum()
            + decomp.detail_vertical.load().sum()
            + decomp.detail_diagonal.load().sum()
        )
        energy_input = np.sum(rand_arr.astype(np.float64) ** 2)
        # Tolerance per fp32 storage of subband coefficients (input is fp32).
        np.testing.assert_allclose(
            float(energy_subbands), float(energy_input), rtol=1e-5,
            err_msg=f"Parseval violated for wavelet={wavelet}",
        )


def test_phase_b_integrates_with_extract_M_pixel(tmp_path: Path) -> None:
    """Phase B canonically consumes Phase A's output without intermediate
    transforms — the canonical pipeline is Phase A -> Phase B sister to
    Phase A -> Phase C.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=3, h=16, w=24)
    # Verify every Phase A reduction feeds Phase B cleanly.
    for axis_reduction in ("l2_norm", "l1_norm", "max"):
        m_pixel = extract_M_pixel(gt, reduction=axis_reduction)
        decomp = decompose_M_contest_per_subband(
            m_pixel, level=1, cache_path=tmp_path
        )
        # Subband shape halves spatial dims.
        assert decomp.subband_shape() == (3, 8, 12)
        # Phase A reduction is preserved in the per-subband meta.
        assert decomp.approximation.reduction == axis_reduction


def test_phase_b_dataclass_invariants_rejected(tmp_path: Path) -> None:
    """SubbandSensitivityDecomposition __post_init__ enforces canonical contract."""
    from tac.master_gradient_comparison.multi_granularity import (
        SubbandSensitivityDecomposition,
        WaveletDecompositionError,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    valid_decomp = decompose_M_contest_per_subband(
        m_pixel, level=1, cache_path=tmp_path
    )

    # Invalid wavelet_family.
    with pytest.raises(WaveletDecompositionError, match="wavelet_family must"):
        SubbandSensitivityDecomposition(
            approximation=valid_decomp.approximation,
            detail_horizontal=valid_decomp.detail_horizontal,
            detail_vertical=valid_decomp.detail_vertical,
            detail_diagonal=valid_decomp.detail_diagonal,
            wavelet_family="biorthogonal",  # invalid
            level=1,
            subband_reduction="abs",
            predecessor_array_sha256="x" * 64,
        )

    # Invalid level (< 1).
    with pytest.raises(WaveletDecompositionError, match="level must be >= 1"):
        SubbandSensitivityDecomposition(
            approximation=valid_decomp.approximation,
            detail_horizontal=valid_decomp.detail_horizontal,
            detail_vertical=valid_decomp.detail_vertical,
            detail_diagonal=valid_decomp.detail_diagonal,
            wavelet_family="db2",
            level=0,  # invalid: identity not stored
            subband_reduction="abs",
            predecessor_array_sha256="x" * 64,
        )

    # Invalid subband_reduction.
    with pytest.raises(WaveletDecompositionError, match="subband_reduction must"):
        SubbandSensitivityDecomposition(
            approximation=valid_decomp.approximation,
            detail_horizontal=valid_decomp.detail_horizontal,
            detail_vertical=valid_decomp.detail_vertical,
            detail_diagonal=valid_decomp.detail_diagonal,
            wavelet_family="db2",
            level=1,
            subband_reduction="rectified_linear",  # invalid
            predecessor_array_sha256="x" * 64,
        )

    # Invalid predecessor_array_sha256 length.
    with pytest.raises(
        WaveletDecompositionError, match="predecessor_array_sha256 must be 64"
    ):
        SubbandSensitivityDecomposition(
            approximation=valid_decomp.approximation,
            detail_horizontal=valid_decomp.detail_horizontal,
            detail_vertical=valid_decomp.detail_vertical,
            detail_diagonal=valid_decomp.detail_diagonal,
            wavelet_family="db2",
            level=1,
            subband_reduction="abs",
            predecessor_array_sha256="short",  # invalid
        )


def test_phase_b_subband_shape_mismatch_rejected(tmp_path: Path) -> None:
    """Mallat 4-subband invariant: all 4 subbands must share spatial shape.

    A mis-constructed SubbandSensitivityDecomposition with mismatched subband
    shapes is refused by __post_init__ per the canonical Mallat contract.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        SubbandSensitivityDecomposition,
        WaveletDecompositionError,
        decompose_M_contest_per_subband,
    )

    gt8 = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel8 = extract_M_pixel(gt8, reduction="l2_norm")
    decomp = decompose_M_contest_per_subband(m_pixel8, level=1, cache_path=tmp_path)
    # Build a sister at a DIFFERENT shape.
    gt16 = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=16, w=16)
    m_pixel16 = extract_M_pixel(gt16, reduction="l2_norm")
    decomp16 = decompose_M_contest_per_subband(m_pixel16, level=1, cache_path=tmp_path)

    # Mix shapes — fails Mallat 4-subband invariant.
    with pytest.raises(WaveletDecompositionError, match="4-subband invariant"):
        SubbandSensitivityDecomposition(
            approximation=decomp.approximation,  # (1, 4, 4)
            detail_horizontal=decomp16.detail_horizontal,  # (1, 8, 8) — mismatch
            detail_vertical=decomp.detail_vertical,
            detail_diagonal=decomp.detail_diagonal,
            wavelet_family="db2",
            level=1,
            subband_reduction="abs",
            predecessor_array_sha256="x" * 64,
        )


def test_phase_b_inflated_source_kind_supported(tmp_path: Path) -> None:
    """Phase B works with InflatedGradientTensor source per the polymorphic
    Phase A contract (source_kind="m_inflated" preserved through decomp).
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    gt_inflated = _make_synthetic_inflated_tensor(tmp_path, n_pairs=2, h=8, w=8)
    m_pixel = extract_M_pixel(gt_inflated, reduction="l2_norm")
    assert m_pixel.source_kind == "m_inflated"

    decomp = decompose_M_contest_per_subband(m_pixel, level=1, cache_path=tmp_path)
    for sub in (
        decomp.approximation,
        decomp.detail_horizontal,
        decomp.detail_vertical,
        decomp.detail_diagonal,
    ):
        assert sub.source_kind == "m_inflated"


def test_phase_b_predicted_grade_non_promotable(tmp_path: Path) -> None:
    """Per Catalog #192 + Catalog #317: ALL 4 subbands are ``[predicted]``
    grade sensitivity surfaces, NEVER promotable as contest scores. The
    measurement_axis MUST be lane-tagged.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    gt = _make_synthetic_contest_tensor(tmp_path, n_pairs=1, h=8, w=8)
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    decomp = decompose_M_contest_per_subband(m_pixel, level=1, cache_path=tmp_path)
    for sub in (
        decomp.approximation,
        decomp.detail_horizontal,
        decomp.detail_vertical,
        decomp.detail_diagonal,
    ):
        assert sub.measurement_axis.startswith("["), sub.measurement_axis
        assert sub.measurement_axis == "[predicted]"


def test_phase_b_canonical_z8_contest_resolution_smoke(tmp_path: Path) -> None:
    """Canonical Z8 contest-resolution-equivalent surface: 384x512 input
    decomposes to (192, 256) per subband at level 1 (sister of M5 milestone
    canonical Z8 LevelDimensionContract).

    Uses a uniform sentinel-tiny constant to keep the test cheap; the
    shape contract is the canonical assertion.
    """
    from tac.master_gradient_comparison.multi_granularity import (
        decompose_M_contest_per_subband,
    )

    # Build a small ContestGradientTensor at the canonical contest shape.
    n_pairs, h, w = 1, 384, 512
    rng = np.random.RandomState(0)
    arr = rng.randn(n_pairs, 3, h, w).astype(np.float32)
    arr_path = tmp_path / "contest_resolution.npy"
    np.save(arr_path, arr)
    import hashlib

    sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    op = OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.119, score=0.192)
    gt = ContestGradientTensor(
        array_path=str(arr_path),
        array_sha256=sha,
        n_pairs=n_pairs,
        height=h,
        width=w,
        contest_video_sha256="a" * 64,
        scorer_seg_sha256="b" * 64,
        scorer_pose_sha256="c" * 64,
        operating_point=op,
        captured_at_utc="2026-05-30T00:00:00+00:00",
        measurement_axis="[predicted]",
    )
    m_pixel = extract_M_pixel(gt, reduction="l2_norm")
    assert m_pixel.shape() == (1, 384, 512)

    decomp = decompose_M_contest_per_subband(m_pixel, level=1, cache_path=tmp_path)
    # Canonical Z8 M5 level-1 subband shape: (H/2, W/2).
    assert decomp.subband_shape() == (1, 192, 256)
