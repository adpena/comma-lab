from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_minimizer import (
    DirectDescriptionError,
    compile_direct_description_archive_v2,
    receive_direct_description_archive_v2,
)
from tac.optimization.direct_description_real_target_rung0 import (
    DirectDescriptionRealTargetRung0ConfigV1,
    RealTargetSubsetV1,
    _block_mean_projection,
    _initial_description,
    _objective,
    _pose6_ordinal_codes,
    load_real_target_subset,
)


def test_integer_block_projection_is_exact_and_geometry_bound() -> None:
    planes = np.zeros((2, 384, 512, 3), dtype=np.uint8)
    planes[0, :48, :64] = 7
    planes[1] = 255
    projected = _block_mean_projection(planes)
    assert projected.shape == (2, 8, 8, 3)
    assert projected.dtype == np.uint8
    assert np.all(projected[0, 0, 0] == 7)
    assert np.all(projected[0, 0, 1] == 0)
    assert np.all(projected[1] == 255)
    with pytest.raises(DirectDescriptionError, match="projection"):
        _block_mean_projection(np.zeros((1, 8, 8, 3), dtype=np.uint8))


def test_pose6_ordinal_codes_are_monotone_and_pair_tie_broken() -> None:
    poses = np.zeros((600, 6), dtype=np.float64)
    poses[:, 0] = np.arange(600, dtype=np.float64)
    poses[:, 1] = 1.0
    codes = _pose6_ordinal_codes(poses)
    assert codes.shape == (600, 6)
    assert codes.dtype == np.uint8
    assert np.all(np.diff(codes[:, 0].astype(np.int16)) >= 0)
    assert np.all(np.diff(codes[:, 1].astype(np.int16)) >= 0)
    assert codes[0, 0] == 0 and codes[-1, 0] == 255
    poses[0, 0] = np.nan
    with pytest.raises(DirectDescriptionError, match="finite"):
        _pose6_ordinal_codes(poses)


def test_real_objective_uses_plane_target_and_pose6_stream(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text("{}")
    config = DirectDescriptionRealTargetRung0ConfigV1(
        target_receipt_path=str(target_path), target_receipt_sha256="0" * 64
    )
    z = _initial_description(config)
    receiver = receive_direct_description_archive_v2(compile_direct_description_archive_v2(z).archive)
    target = RealTargetSubsetV1(
        receipt=None,  # type: ignore[arg-type]
        receipt_path=target_path,
        receipt_sha256="0" * 64,
        projection=np.zeros((64, 2, 8, 8, 3), dtype=np.uint8),
        pose6_codes=np.zeros((64, 6), dtype=np.uint8),
    )
    before = _objective(receiver, target)
    pose = z.pose6_dxi_residuals.payload
    mutated = z.replace_stream_byte("pose6_dxi_residuals", 0, pose[0] ^ 1)
    after = _objective(
        receive_direct_description_archive_v2(compile_direct_description_archive_v2(mutated).archive),
        target,
    )
    assert before["pose6_integer_l1_debt"] != after["pose6_integer_l1_debt"]
    assert before["plane_integer_l1_debt"] != after["plane_integer_l1_debt"]


def test_config_and_receipt_hash_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        DirectDescriptionRealTargetRung0ConfigV1(
            target_receipt_path=str(tmp_path / "receipt.json"), target_receipt_sha256="bad"
        )
    path = tmp_path / "receipt.json"
    path.write_bytes(b"{}\n")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(DirectDescriptionError, match="SHA-256"):
        load_real_target_subset(path, "0" * 64)
    with pytest.raises((DirectDescriptionError, ValueError)):
        load_real_target_subset(path, observed)
