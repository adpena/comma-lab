from __future__ import annotations

from pathlib import Path

import pytest

from tac.boundary_math.shared_receiver_admission import (
    MAX_ARCHIVE_BYTES,
    MAX_D_SEG,
    SCORE_BYTES_NORMALIZER,
)
from tools.measure_r1b_boundary_generator_n600 import (
    FIXED_C1_CAP_BYTES,
    R1BMeasurementError,
    _source_hashes,
    compose_score,
    gate_summary,
    storage_preflight,
)


def test_score_decomposes_all_three_authority_terms() -> None:
    row = compose_score(d_seg=0.001, d_pose=0.004, archive_bytes=100_000)
    assert row["seg_component"] == pytest.approx(0.1)
    assert row["pose_component"] == pytest.approx((10.0 * 0.004) ** 0.5)
    assert row["rate_component"] == pytest.approx(25.0 * 100_000 / SCORE_BYTES_NORMALIZER)
    assert row["score"] == pytest.approx(
        row["seg_component"] + row["pose_component"] + row["rate_component"]
    )


def test_gate_summary_keeps_task_gate_and_fixed_c1_cap_distinct() -> None:
    row = gate_summary(archive_bytes=FIXED_C1_CAP_BYTES + 1, d_seg=MAX_D_SEG)
    assert row["archive_gate_pass"] is True
    assert row["fixed_c1_cap_pass"] is False
    assert row["d_seg_gate_pass"] is True
    assert row["joint_task_gate_pass"] is True
    assert row["joint_fixed_c1_gate_pass"] is False
    assert MAX_ARCHIVE_BYTES > FIXED_C1_CAP_BYTES


def test_score_and_gate_refuse_invalid_values() -> None:
    with pytest.raises(R1BMeasurementError):
        compose_score(d_seg=float("nan"), d_pose=0.0, archive_bytes=1)
    with pytest.raises(R1BMeasurementError):
        compose_score(d_seg=0.0, d_pose=0.0, archive_bytes=0)


def test_storage_preflight_refuses_non_ssd_location(tmp_path: Path) -> None:
    with pytest.raises(R1BMeasurementError, match="scratch_root"):
        storage_preflight(tmp_path)


def test_scorer_hashes_follow_upstream_models_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    for relative in (
        "modules.py",
        "frame_utils.py",
        "models/posenet.safetensors",
        "models/segnet.safetensors",
    ):
        (tmp_path / relative).write_bytes(relative.encode("ascii"))
    hashes = _source_hashes(tmp_path)
    assert set(hashes) == {
        "modules.py",
        "frame_utils.py",
        "posenet.safetensors",
        "segnet.safetensors",
    }
