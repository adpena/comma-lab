from __future__ import annotations

import numpy as np
import pytest

from tac.local_acceleration.ane_unlock_followup_20260713 import (
    flip_summary,
    require_real_n600,
    stored_npy_memmap,
)


def test_stored_npy_member_is_zero_copy_mappable(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.savez(path, gt_f1=data)
    mapped = stored_npy_memmap(path, "gt_f1.npy")
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, data)


def test_compressed_member_refuses_mapping(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    np.savez_compressed(path, gt_f1=np.zeros((1, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="compressed"):
        stored_npy_memmap(path, "gt_f1.npy")


def test_n600_and_worst_pair_contract() -> None:
    require_real_n600(600)
    with pytest.raises(ValueError, match="exactly n600"):
        require_real_n600(24)
    ref = np.zeros((2, 2, 1, 2), dtype=np.float32)
    pred = ref.copy()
    ref[:, 0] = 1.0
    pred[1, 0, 0, 0] = 0.0
    pred[1, 1, 0, 0] = 2.0
    row = flip_summary(ref, pred)
    assert row["worst_pair_index"] == 1
    assert row["worst_pair_flip_fraction"] == 0.5
