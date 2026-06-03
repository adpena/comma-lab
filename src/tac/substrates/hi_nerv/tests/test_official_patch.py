# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.hi_nerv.official_patch import (
    HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF,
    OfficialPatchIndexError,
    official_compute_pixel_idx_3d,
    official_flat_patch_index_to_thw,
    official_patch_index_contract,
    official_patch_to_video,
    official_video_to_patch,
    official_vidx_to_pidx,
)


def test_official_video_patch_roundtrip_matches_source_order() -> None:
    video = np.arange(2 * 4 * 6 * 8 * 3, dtype=np.int32).reshape(2, 4, 6, 8, 3)

    patches = official_video_to_patch(video, patch_size=(2, 3, 4))

    assert patches.shape == (16, 2, 3, 4, 3)
    np.testing.assert_array_equal(patches[0], video[0, 0:2, 0:3, 0:4, :])
    np.testing.assert_array_equal(patches[1], video[0, 0:2, 0:3, 4:8, :])
    np.testing.assert_array_equal(patches[2], video[0, 0:2, 3:6, 0:4, :])

    roundtrip = official_patch_to_video(patches, video_size=(4, 6, 8))
    np.testing.assert_array_equal(roundtrip, video)


def test_official_vidx_to_pidx_expands_child_patch_grid() -> None:
    vidx = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)

    pidx = official_vidx_to_pidx(
        vidx,
        vidx_max=(2, 2, 2),
        pidx_max=(4, 4, 6),
    )

    expected_first = np.array(
        [
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [0, 3, 0],
            [0, 3, 1],
            [0, 3, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [1, 3, 0],
            [1, 3, 1],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(pidx[:12], expected_first)
    assert pidx.shape == (24, 3)


def test_official_compute_pixel_idx_3d_matches_padding_and_clipping_contract() -> None:
    idx = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)

    result = official_compute_pixel_idx_3d(
        idx,
        idx_max=(2, 3, 4),
        sizes=(4, 9, 8),
        padding=(1, 0, 2),
        clipped=True,
        return_mask=True,
    )

    t_idx, h_idx, w_idx = result.pixel_indices
    t_mask, h_mask, w_mask = result.masks or (None, None, None)
    np.testing.assert_array_equal(t_idx[0], np.array([0, 0, 1, 2]))
    np.testing.assert_array_equal(h_idx[1], np.array([6, 7, 8]))
    np.testing.assert_array_equal(w_idx[1], np.array([4, 5, 6, 7, 7, 7]))
    np.testing.assert_array_equal(t_mask[0], np.array([False, True, True, True]))
    np.testing.assert_array_equal(h_mask[1], np.array([True, True, True]))
    np.testing.assert_array_equal(w_mask[1], np.array([True, True, True, True, False, False]))


def test_official_flat_patch_index_to_thw_matches_dataset_mapping() -> None:
    flat = np.array([0, 1, 3, 4, 11], dtype=np.int64)

    thw = official_flat_patch_index_to_thw(flat, num_patches=(2, 2, 3))

    expected = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 1, 2],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(thw, expected)


def test_official_patch_contract_is_false_authority() -> None:
    contract = official_patch_index_contract()

    assert HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF in contract["proof_marker"]
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False


def test_official_patch_helpers_fail_closed_on_invalid_shapes() -> None:
    with pytest.raises(OfficialPatchIndexError, match="shape"):
        official_vidx_to_pidx(
            np.zeros((3, 2), dtype=np.int64),
            vidx_max=(1, 1, 1),
            pidx_max=(1, 1, 1),
        )
    with pytest.raises(OfficialPatchIndexError, match="divisible"):
        official_video_to_patch(np.zeros((1, 3, 5, 5, 1)), patch_size=(2, 5, 5))
