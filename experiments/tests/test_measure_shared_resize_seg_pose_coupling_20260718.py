# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from experiments.measure_shared_resize_seg_pose_coupling_20260718 import (
    MeasurementError,
    aggregate_response_metrics,
    deterministic_stride_sample,
    inspect_checkpoint_payload_keys,
    mmap_stored_npy_member,
    realized_lsb_counts,
    sample_mean_product_gram,
    target_cache_mismatch_metrics,
    topk_sign_direction,
)


def test_stride_sample_is_unique_sorted_and_reproducible() -> None:
    first = deterministic_stride_sample(600, 8, 538)
    assert first == deterministic_stride_sample(600, 8, 538)
    assert first == tuple(sorted(first))
    assert len(set(first)) == 8
    assert first != deterministic_stride_sample(600, 8, 539)


def test_product_gram_uses_sample_mean_block_scaling() -> None:
    assert sample_mean_product_gram(32.0, 16.0, 8.0, n_sample=4) == [
        [2.0, 1.0],
        [1.0, 0.5],
    ]


def test_checkpoint_payload_inspection_refuses_carriers() -> None:
    custody = inspect_checkpoint_payload_keys(("code", "in_proj.weight", "__cfg_n_pairs"))
    assert custody["carrier_absent"] is True
    assert custody["base_inr_only"] is True
    with pytest.raises(MeasurementError, match="pose-carrier"):
        inspect_checkpoint_payload_keys(("code", "pose_carrier.dxi"))
    with pytest.raises(MeasurementError, match="pose-carrier"):
        inspect_checkpoint_payload_keys(("code", "liveP__pose_carrier.xi_stored"))
    with pytest.raises(MeasurementError, match="pose-carrier"):
        inspect_checkpoint_payload_keys(("code", "__cfg_pose_carrier_residual_scale"))


def test_target_cache_mismatch_metrics_are_exact() -> None:
    metrics = target_cache_mismatch_metrics(
        np.array([[[0, 1], [1, 0]]]),
        np.array([[[0, 1], [0, 0]]]),
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.0, 2.5]], dtype=np.float32),
    )
    assert metrics["seg_mismatched_pixels"] == 1
    assert metrics["seg_total_pixels"] == 4
    assert metrics["seg_mismatch_fraction"] == 0.25
    assert metrics["pose_mismatched_elements"] == 1
    assert metrics["pose_max_abs"] == 0.5
    assert metrics["pose_mse"] == 0.125


def test_topk_direction_has_stable_ties_and_camera_layout() -> None:
    gradient = np.array([[[[2.0], [-2.0]], [[1.0], [1.0]]]], dtype=np.float32)
    direction = topk_sign_direction(gradient, 0.5)
    np.testing.assert_array_equal(
        direction,
        np.array([[[[-1], [1]], [[0], [0]]]], dtype=np.int8),
    )
    with pytest.raises(ValueError, match="non-finite"):
        topk_sign_direction(np.array([np.nan]), 1.0)


def test_realized_counts_conserve_requests_and_detect_clipping() -> None:
    pair = (
        np.array([[[0], [5]]], dtype=np.uint8),
        np.array([[[255], [10]]], dtype=np.uint8),
    )
    direction = np.array([[[[-1], [1]]], [[[1], [-1]]]], dtype=np.int8)
    counts = realized_lsb_counts(pair, direction, +1)
    assert counts == {"nonzero_requested": 4, "realized_changed": 2, "boundary_clipped": 2}


def test_response_classifies_failed_target_as_uninformative() -> None:
    result = aggregate_response_metrics(
        baseline={"d_seg": 0.2, "d_pose": 0.4},
        seg_plus={"d_seg": 0.19, "d_pose": 0.41},
        seg_minus={"d_seg": 0.21, "d_pose": 0.39},
        pose_plus={"d_seg": 0.18, "d_pose": 0.42},
        pose_minus={"d_seg": 0.22, "d_pose": 0.38},
        joint_plus={"d_seg": 0.19, "d_pose": 0.39},
    )
    classified = result["measured_direction_classification"]
    assert classified["seg_direction"]["quality"] == "MEASURED_HELP"
    assert classified["seg_direction"]["cross_d_pose_effect"] == "MEASURED_HARM"
    assert classified["pose_direction"]["quality"] == "UNINFORMATIVE_DIRECTION"
    assert classified["pose_direction"]["cross_d_seg_effect"] == "UNINFORMATIVE_DIRECTION"


def test_mmap_stored_member_reads_without_extracting(tmp_path) -> None:
    archive = tmp_path / "cache.npz"
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez(archive, values=values)
    mapped = mmap_stored_npy_member(archive, "values.npy")
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, values)


def test_mmap_rejects_compressed_member(tmp_path) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.arange(3, dtype=np.int16))
    archive = tmp_path / "compressed.npz"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("values.npy", buffer.getvalue())
    with pytest.raises(MeasurementError, match="not ZIP_STORED"):
        mmap_stored_npy_member(archive, "values.npy")
