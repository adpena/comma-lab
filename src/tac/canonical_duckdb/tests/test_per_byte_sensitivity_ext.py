# SPDX-License-Identifier: MIT
"""Tests for per_byte_sensitivity_ext extension table backfill."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_duckdb.per_byte_sensitivity_ext import (
    CLASS_NAMES,
    QUANTILE_BOUNDARIES,
    bootstrap_per_byte_sensitivity_table,
    compute_sensitivity_classes,
    refresh_per_byte_sensitivity,
    refresh_per_byte_sensitivity_table,
)
from tac.canonical_duckdb.schema import connect


def test_compute_sensitivity_classes_canonical_4_class() -> None:
    """4-class quantile partition: top_2pct + top_5pct + top_20pct + tail."""
    n = 1000
    rng = np.random.default_rng(seed=42)
    sens = rng.uniform(size=n).astype(np.float32)
    rank, cls = compute_sensitivity_classes(sens)
    assert rank.shape == (n,)
    assert cls.shape == (n,)
    # Class counts
    counts = {c: int(np.sum(cls == c)) for c in CLASS_NAMES}
    assert counts["top_2pct"] == 20  # 2% of 1000
    assert counts["top_5pct"] == 30  # 5% - 2% = 3%
    assert counts["top_20pct"] == 150  # 20% - 5% = 15%
    assert counts["tail"] == 800  # 80%


def test_compute_sensitivity_classes_top_assigned_highest() -> None:
    """top_2pct class must contain the highest sensitivity bytes."""
    n = 100
    sens = np.arange(n, dtype=np.float32)  # 0, 1, 2, ..., 99
    rank, cls = compute_sensitivity_classes(sens)
    # Top 2% = 2 elements (n*0.02 = 2); highest are sens[98], sens[99]
    top_indices = np.where(cls == "top_2pct")[0]
    assert len(top_indices) == 2
    # The highest 2 sensitivities must be in top_2pct
    assert 98 in top_indices and 99 in top_indices


def test_compute_sensitivity_classes_rejects_wrong_names_length() -> None:
    """class_names length must equal boundaries + 1."""
    sens = np.ones(10, dtype=np.float32)
    with pytest.raises(ValueError, match="must equal"):
        compute_sensitivity_classes(
            sens,
            boundaries=(0.02, 0.05),
            class_names=("a", "b", "c", "d"),  # wrong: 4 names vs 3 boundaries + 1 = 3
        )


def test_bootstrap_creates_per_byte_sensitivity_table(tmp_path: Path) -> None:
    """bootstrap_per_byte_sensitivity_table is idempotent."""
    db_path = tmp_path / "test.duckdb"
    con = connect(db_path)
    bootstrap_per_byte_sensitivity_table(con)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "per_byte_sensitivity" in tables
    # Idempotent
    bootstrap_per_byte_sensitivity_table(con)
    con.close()


def test_refresh_per_byte_sensitivity_empty_repo_returns_zero(tmp_path: Path) -> None:
    """No master_gradient_anchors.jsonl → 0 rows."""
    db_path = tmp_path / "test.duckdb"
    con = connect(db_path)
    bootstrap_per_byte_sensitivity_table(con)
    n = refresh_per_byte_sensitivity(con, tmp_path)
    assert n == 0
    con.close()


def test_refresh_per_byte_sensitivity_synthetic_archive(tmp_path: Path) -> None:
    """Backfill correctly from a synthetic master_gradient anchor + .npy."""
    repo = tmp_path
    state_dir = repo / ".omx/state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic gradient: 100 bytes, deterministic
    n = 100
    rng = np.random.default_rng(seed=42)
    arr = rng.standard_normal((n, 3)).astype(np.float32) * 1e-6
    npy_path = state_dir / "syn_grad.npy"
    np.save(npy_path, arr)

    anchor = {
        "archive_sha256": "sha_test_" + "a" * 56,
        "operating_point": {"d_seg": 1e-3, "d_pose": 1e-4, "rate": 5e-3, "score": 0.2},
        "gradient_array_path": ".omx/state/syn_grad.npy",
        "n_bytes": n,
        "measurement_method": "synthetic",
        "measurement_axis": "[predicted]",
        "measurement_hardware": "synthetic",
        "measurement_call_id": "test",
        "measurement_utc": "2026-05-18T00:00:00Z",
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "schema_version": "master_gradient_anchor_v1",
    }
    jsonl_path = state_dir / "master_gradient_anchors.jsonl"
    jsonl_path.write_text(json.dumps(anchor) + "\n")

    db_path = repo / "test.duckdb"
    con = connect(db_path)
    bootstrap_per_byte_sensitivity_table(con)
    inserted = refresh_per_byte_sensitivity(con, repo)
    assert inserted == n

    rows = con.execute(
        "SELECT archive_sha256, byte_idx, sensitivity_class FROM per_byte_sensitivity "
        "WHERE archive_sha256 = ?",
        [anchor["archive_sha256"]],
    ).fetchall()
    assert len(rows) == n
    # 4 distinct classes present
    classes = {r[2] for r in rows}
    assert "top_2pct" in classes
    assert "top_5pct" in classes
    assert "top_20pct" in classes
    assert "tail" in classes
    con.close()


def test_refresh_per_byte_sensitivity_skips_false_contest_axis_anchor(
    tmp_path: Path,
) -> None:
    """DuckDB backfill must not materialize rows from mislabeled contest anchors."""
    repo = tmp_path
    state_dir = repo / ".omx/state"
    state_dir.mkdir(parents=True, exist_ok=True)
    arr = np.ones((20, 3), dtype=np.float32)
    npy_path = state_dir / "false_axis.npy"
    np.save(npy_path, arr)
    anchor = {
        "archive_sha256": "sha_test_" + "c" * 56,
        "operating_point": {"d_seg": 1e-3, "d_pose": 1e-4, "rate": 5e-3, "score": 0.2},
        "gradient_array_path": ".omx/state/false_axis.npy",
        "n_bytes": 20,
        "measurement_method": "autograd_per_parameter_projected_8pair_subset",
        "measurement_axis": "[contest-CUDA]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_mps_advisory",
        "measurement_call_id": "test",
        "measurement_utc": "2026-05-18T00:00:00Z",
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "schema_version": "master_gradient_anchor_v1",
    }
    (state_dir / "master_gradient_anchors.jsonl").write_text(json.dumps(anchor) + "\n")

    db_path = repo / "test.duckdb"
    con = connect(db_path)
    bootstrap_per_byte_sensitivity_table(con)
    assert refresh_per_byte_sensitivity(con, repo) == 0
    con.close()


def test_refresh_per_byte_sensitivity_uses_advisory_correction_row(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    state_dir = repo / ".omx/state"
    state_dir.mkdir(parents=True, exist_ok=True)
    arr = np.ones((20, 3), dtype=np.float32)
    npy_path = state_dir / "advisory.npy"
    np.save(npy_path, arr)
    base = {
        "archive_sha256": "sha_test_" + "d" * 56,
        "operating_point": {"d_seg": 1e-3, "d_pose": 1e-4, "rate": 5e-3, "score": 0.2},
        "gradient_array_path": ".omx/state/advisory.npy",
        "n_bytes": 20,
        "measurement_call_id": "test",
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "schema_version": "master_gradient_anchor_v1",
    }
    stale = {
        **base,
        "measurement_method": "autograd_per_parameter_projected_8pair_subset",
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_utc": "2026-05-18T00:00:00Z",
    }
    correction = {
        **base,
        "measurement_method": "autograd_per_parameter_projected_8pair_subset_axis_correction",
        "measurement_axis": "[macOS-CPU advisory]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_utc": "2026-05-18T01:00:00Z",
    }
    (state_dir / "master_gradient_anchors.jsonl").write_text(
        json.dumps(stale) + "\n" + json.dumps(correction) + "\n"
    )

    db_path = repo / "test.duckdb"
    con = connect(db_path)
    bootstrap_per_byte_sensitivity_table(con)
    assert refresh_per_byte_sensitivity(con, repo) == 20
    rows = con.execute(
        "SELECT DISTINCT source_measurement_axis, evidence_grade, "
        "source_anchor_authoritative, promotion_eligible "
        "FROM per_byte_sensitivity WHERE archive_sha256 = ?",
        [base["archive_sha256"]],
    ).fetchall()
    assert rows == [("[macOS-CPU advisory]", "diagnostic", False, False)]
    con.close()


def test_refresh_per_byte_sensitivity_marks_derived_contest_rows_non_promotable(
    tmp_path: Path,
) -> None:
    """Per-byte sensitivity rows are planning rows even from authoritative sources."""
    repo = tmp_path
    state_dir = repo / ".omx/state"
    state_dir.mkdir(parents=True, exist_ok=True)
    arr = np.ones((10, 3), dtype=np.float32)
    npy_path = state_dir / "contest.npy"
    np.save(npy_path, arr)
    archive_sha = "e" * 64
    anchor = {
        "archive_sha256": archive_sha,
        "scored_archive_sha256": archive_sha,
        "scored_archive_bytes": 12345,
        "operating_point": {"d_seg": 1e-3, "d_pose": 1e-4, "rate": 5e-3, "score": 0.2},
        "gradient_array_path": ".omx/state/contest.npy",
        "n_bytes": 10,
        "measurement_method": "autograd_per_parameter_projected_full",
        "measurement_axis": "[contest-CUDA]",
        "measurement_hardware": "linux_x86_64_t4_cuda",
        "measurement_call_id": "fc-test",
        "measurement_utc": "2026-05-18T00:00:00Z",
        "n_pairs_used": 600,
        "n_pairs_total": 600,
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "schema_version": "master_gradient_anchor_v1",
    }
    (state_dir / "master_gradient_anchors.jsonl").write_text(json.dumps(anchor) + "\n")

    db_path = repo / "test.duckdb"
    con = connect(db_path)
    bootstrap_per_byte_sensitivity_table(con)
    assert refresh_per_byte_sensitivity(con, repo) == 10
    rows = con.execute(
        "SELECT DISTINCT evidence_grade, source_anchor_authoritative, "
        "promotion_eligible FROM per_byte_sensitivity WHERE archive_sha256 = ?",
        [archive_sha],
    ).fetchall()
    assert rows == [("diagnostic_from_contest_authoritative_source", True, False)]
    con.close()


def test_refresh_per_byte_sensitivity_table_under_canonical_lock(tmp_path: Path) -> None:
    """The operator-facing entry uses the canonical lock + returns expected shape."""
    repo = tmp_path
    state_dir = repo / ".omx/state"
    state_dir.mkdir(parents=True, exist_ok=True)

    arr = np.ones((50, 3), dtype=np.float32) * 1e-5
    arr[0] *= 100  # spike on byte 0
    npy_path = state_dir / "syn.npy"
    np.save(npy_path, arr)

    anchor = {
        "archive_sha256": "sha_test_" + "b" * 56,
        "operating_point": {"d_seg": 1e-3, "d_pose": 1e-4, "rate": 5e-3, "score": 0.2},
        "gradient_array_path": ".omx/state/syn.npy",
        "n_bytes": 50,
        "measurement_method": "synthetic",
        "measurement_axis": "[predicted]",
        "measurement_hardware": "synthetic",
        "measurement_call_id": "test",
        "measurement_utc": "2026-05-18T00:00:00Z",
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "schema_version": "master_gradient_anchor_v1",
    }
    (state_dir / "master_gradient_anchors.jsonl").write_text(json.dumps(anchor) + "\n")

    result = refresh_per_byte_sensitivity_table(repo, db_path=repo / "canonical.duckdb")
    assert result["table"] == "per_byte_sensitivity"
    assert result["row_count"] == 50
    assert "refreshed_at_utc" in result
    assert result["source_of_truth"] == "master_gradient_anchors.jsonl + .npy sidecars"


def test_quantile_boundaries_canonical() -> None:
    """The canonical QUANTILE_BOUNDARIES matches the sensitivity_mask_aware_quantizr_v1 design."""
    assert QUANTILE_BOUNDARIES == (0.02, 0.05, 0.20)
    assert CLASS_NAMES == ("top_2pct", "top_5pct", "top_20pct", "tail")
