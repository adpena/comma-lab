from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.canonical_equations.ddm_runtime_export_identity_20260723 import (
    EQUATION_ID,
    PAINT_JACOBIAN_EQUATION_ID,
    describe,
    export_identity,
    score_row,
    semantic_paint_jacobian_summary,
    sha256_chunks,
)


def test_chunk_hash_and_identity() -> None:
    payload = b"camera-bytes"
    count, digest = sha256_chunks([payload[:4], payload[4:]])
    assert count == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    result = export_identity(
        pair_count=600,
        source_bytes=count,
        source_sha256=digest,
        packaged_bytes=count,
        packaged_sha256=digest,
    )
    assert result.byte_identical


def test_identity_rejects_length_or_hash_drift() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    assert not export_identity(
        pair_count=1,
        source_bytes=1,
        source_sha256=digest,
        packaged_bytes=2,
        packaged_sha256=digest,
    ).byte_identical
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        export_identity(
            pair_count=1,
            source_bytes=1,
            source_sha256="bad",
            packaged_bytes=1,
            packaged_sha256=digest,
        )


def test_score_row_and_scope() -> None:
    row = score_row(archive_bytes=177_169, d_seg=0.000545578002735662, d_pose=0.00002930755865188909)
    assert row["total"] == pytest.approx(0.18964681567130603)
    description = describe()
    assert description["equation_id"] == EQUATION_ID
    assert description["score_claim"] is False
    assert (
        description["empirical_verification_status"]
        == "MEASURED_EXACT_N600_BUILD_636_SOURCE_AND_PACKAGED_RAW_SHA256_IDENTICAL"
    )


def test_semantic_paint_jacobian_closes_exact_camera_preimages() -> None:
    labels = np.array([[[1, 2], [0, 1]]], dtype=np.uint8)
    row = semantic_paint_jacobian_summary(
        labels,
        [[0, 0, 0], [10, 20, 30], [255, 40, 50]],
        camera_hw=(4, 4),
        frames_per_pair=2,
    )
    assert row["equation_id"] == PAINT_JACOBIAN_EQUATION_ID
    assert row["label_assignment_preimage_camera_pixels"] == {
        "max_per_cell_all_frames": 8,
        "min_per_cell_all_frames": 8,
        "sum_per_pair_all_frames": 32,
    }
    assert row["painted_camera_pixels_all_pairs_all_frames"] == 24
    assert [
        coefficient["unit_perturbation_output_bytes_changed_per_channel"]
        for coefficient in row["coefficient_rows"]
    ] == [16, 8]
    assert row["coefficient_rows"][1]["unit_perturbation_direction_rgb"] == [
        -1,
        1,
        1,
    ]
