from __future__ import annotations

import numpy as np

from tac.engineered_corrections import unpack_sparse_corrections
from tac.optimization.inflate_postprocess_surface import RawVideoShape
from tac.optimization.sparse_residual_oracle import (
    SparseResidualOracleConfig,
    authority_payload,
    select_sparse_residual_plan,
    write_charge_proxy_archive,
    write_sparse_residual_candidate,
)


def _write_raw(path, array: np.ndarray) -> None:
    path.write_bytes(array.astype(np.uint8).tobytes())


def test_selects_and_applies_top_sparse_residual_pixels(tmp_path) -> None:
    shape = RawVideoShape(frames=2, height=2, width=2, channels=3)
    baseline = np.full((2, 2, 2, 3), 10, dtype=np.uint8)
    target = baseline.copy()
    target[0, 0, 0] = [13, 10, 10]
    target[1, 1, 1] = [10, 7, 10]
    baseline_raw = tmp_path / "baseline.raw"
    target_raw = tmp_path / "target.raw"
    output_raw = tmp_path / "candidate.raw"
    correction_bin = tmp_path / "corr.bin"
    _write_raw(baseline_raw, baseline)
    _write_raw(target_raw, target)

    config = SparseResidualOracleConfig(top_k_pixels=2, max_abs_delta=2, chunk_frames=1)
    plan = select_sparse_residual_plan(
        baseline_raw=baseline_raw,
        target_raw=target_raw,
        shape=shape,
        config=config,
    )
    result = write_sparse_residual_candidate(
        baseline_raw=baseline_raw,
        output_raw=output_raw,
        correction_bin=correction_bin,
        plan=plan,
        shape=shape,
    )

    decoded = np.frombuffer(output_raw.read_bytes(), dtype=np.uint8).reshape(baseline.shape)
    assert plan.sparse["n_kept"] == 2
    assert result.changed_pixel_count == 2
    assert result.changed_byte_count == 2
    assert result.max_abs_delta_applied == 2
    assert decoded[0, 0, 0].tolist() == [12, 10, 10]
    assert decoded[1, 1, 1].tolist() == [10, 8, 10]
    assert result.passed_visible_change is True
    unpacked = unpack_sparse_corrections(correction_bin.read_bytes(), compressed=True)
    assert unpacked["n_kept"] == 2


def test_frame_selector_limits_selection_to_odd_frames(tmp_path) -> None:
    shape = RawVideoShape(frames=3, height=1, width=2, channels=3)
    baseline = np.full((3, 1, 2, 3), 50, dtype=np.uint8)
    target = baseline.copy()
    target[0, 0, 0] = [60, 60, 60]
    target[1, 0, 1] = [55, 50, 50]
    baseline_raw = tmp_path / "baseline.raw"
    target_raw = tmp_path / "target.raw"
    _write_raw(baseline_raw, baseline)
    _write_raw(target_raw, target)

    plan = select_sparse_residual_plan(
        baseline_raw=baseline_raw,
        target_raw=target_raw,
        shape=shape,
        config=SparseResidualOracleConfig(
            top_k_pixels=1,
            max_abs_delta=1,
            frame_selector="odd",
            chunk_frames=2,
        ),
    )

    pixels_per_frame = shape.height * shape.width
    assert plan.sparse["n_kept"] == 1
    assert int(plan.sparse["indices"][0]) // pixels_per_frame == 1


def test_charge_proxy_archive_adds_exact_packed_bytes(tmp_path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"PK\x03\x04baseline")
    payload = b"charged-correction"
    out = tmp_path / "proxy.zip"

    meta = write_charge_proxy_archive(
        baseline_archive=archive,
        correction_payload=payload,
        output_archive=out,
    )

    assert out.stat().st_size == archive.stat().st_size + len(payload)
    assert meta["extra_bytes_exact"] == len(payload)
    assert meta["is_valid_submission_archive_claim"] is False


def test_authority_payload_is_never_promotable() -> None:
    payload = authority_payload()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "correction_bytes_not_yet_consumed_by_live_inflate_py" in payload["promotion_blockers"]
