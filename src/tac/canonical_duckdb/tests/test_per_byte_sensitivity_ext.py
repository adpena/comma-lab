# SPDX-License-Identifier: MIT
"""Tests for per_byte_sensitivity_ext extension table backfill."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_duckdb.per_byte_sensitivity_ext import (
    CLASS_NAMES,
    QUANTILE_BOUNDARIES,
    MultiGranularitySensitivityRun,
    bootstrap_multi_granularity_sensitivity_table,
    bootstrap_per_byte_sensitivity_table,
    compute_sensitivity_classes,
    query_multi_granularity_sensitivity_top_cells,
    refresh_multi_granularity_sensitivity,
    refresh_multi_granularity_sensitivity_table,
    refresh_per_byte_sensitivity,
    refresh_per_byte_sensitivity_table,
    upsert_multi_granularity_sensitivity_rows,
)
from tac.canonical_duckdb.query import audit_table_provenance
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


def test_bootstrap_creates_multi_granularity_sensitivity_tables(tmp_path: Path) -> None:
    """ITEM_8 bootstrap creates cells, runs, and joined read-model view."""
    con = connect(tmp_path / "test.duckdb")
    bootstrap_multi_granularity_sensitivity_table(con)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "multi_granularity_sensitivity" in tables
    assert "multi_granularity_sensitivity_runs" in tables
    assert "multi_granularity_sensitivity_with_runs" in tables
    bootstrap_multi_granularity_sensitivity_table(con)
    con.close()


def test_multi_granularity_upsert_requires_planning_only_run(tmp_path: Path) -> None:
    """ITEM_8 rows must be explicit run/provenance diagnostics, never score claims."""
    con = connect(tmp_path / "test.duckdb")
    run = MultiGranularitySensitivityRun(
        run_id="run_test",
        archive_sha256="a" * 64,
        gradient_tensor_kind="per_pair_per_byte_v1",
        gradient_array_path=".omx/state/per_pair.npy",
        class_basis="synthetic_segnet_class_weights",
        source_measurement_axis="[macOS-CPU advisory]",
        evidence_grade="diagnostic",
        source_anchor_authoritative=False,
        row_count=1,
    )
    inserted = upsert_multi_granularity_sensitivity_rows(
        con,
        [
            {
                "run_id": "run_test",
                "archive_sha256": "a" * 64,
                "pair_id": 0,
                "byte_offset": 2,
                "class_id": 3,
                "axis": "pose",
                "raw_gradient_fp64": -0.5,
                "marginal_coefficient_fp64": 10.0,
                "class_weight_fp64": 2.0,
                "derived_at_utc": "2026-05-19T00:00:00Z",
            }
        ],
        run=run,
    )
    assert inserted == 1
    rows = query_multi_granularity_sensitivity_top_cells(
        con,
        "a" * 64,
        axis="pose",
        class_id=3,
    )
    assert rows[0]["signed_score_delta_unit_fp64"] == -10.0
    assert rows[0]["sensitivity_fp64"] == 10.0
    run_row = con.execute(
        "SELECT score_claim, promotion_eligible, ready_for_exact_eval_dispatch "
        "FROM multi_granularity_sensitivity_runs WHERE run_id = 'run_test'"
    ).fetchone()
    assert run_row == (False, False, False)

    bad_run = {**run.__dict__, "run_id": "bad_run", "score_claim": True}
    with pytest.raises(ValueError, match="score_claim must be false"):
        upsert_multi_granularity_sensitivity_rows(con, [], run=bad_run)
    bad_authority = {
        **run.__dict__,
        "run_id": "bad_authority",
        "source_anchor_authoritative": True,
    }
    with pytest.raises(ValueError, match="contest measurement axis"):
        upsert_multi_granularity_sensitivity_rows(con, [], run=bad_authority)
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


def _write_per_pair_anchor(
    repo: Path,
    arr: np.ndarray,
    *,
    archive_sha: str = "f" * 64,
    tensor_kind: str = "per_pair_per_byte_v1",
    measurement_axis: str = "[predicted]",
    measurement_hardware: str = "synthetic",
    measurement_method: str = "synthetic_per_pair",
) -> Path:
    state_dir = repo / ".omx/state"
    state_dir.mkdir(parents=True, exist_ok=True)
    npy_path = state_dir / "per_pair.npy"
    np.save(npy_path, arr)
    anchor = {
        "archive_sha256": archive_sha,
        "operating_point": {"d_seg": 0.01, "d_pose": 0.025, "rate": 0.0, "score": 0.2},
        "gradient_array_path": ".omx/state/per_pair.npy",
        "n_bytes": 4,
        "n_pairs": 2,
        "measurement_method": measurement_method,
        "measurement_axis": measurement_axis,
        "measurement_hardware": measurement_hardware,
        "measurement_call_id": "test-call",
        "measurement_utc": "2026-05-19T00:00:00Z",
        "gradient_tensor_kind": tensor_kind,
        "schema_version": "master_gradient_anchor_v1",
    }
    (state_dir / "master_gradient_anchors.jsonl").write_text(json.dumps(anchor) + "\n")
    return npy_path


def _write_class_source(repo: Path, weights: list[float] | None = None) -> Path:
    path = repo / ".omx/state/class_weights.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "multi_granularity_class_weights_v1",
                "class_basis": "synthetic_segnet_class_weights",
                "class_weights": weights or [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            sort_keys=True,
        )
    )
    return path


def test_refresh_multi_granularity_sensitivity_from_bounded_per_pair_anchor(
    tmp_path: Path,
) -> None:
    """Bounded ITEM_8 refresh preserves pair x byte x class x axis signal."""
    repo = tmp_path
    arr = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3) / 100.0
    _write_per_pair_anchor(repo, arr)
    class_source = _write_class_source(repo)
    con = connect(repo / "test.duckdb")

    inserted = refresh_multi_granularity_sensitivity(
        con,
        repo,
        class_source_path=class_source,
        byte_offsets=[1, 3],
        refreshed_at="2026-05-19T01:00:00Z",
    )
    assert inserted == 2 * 2 * 5 * 3

    row = con.execute(
        """
        SELECT raw_gradient_fp64, marginal_coefficient_fp64, class_weight_fp64,
               signed_score_delta_unit_fp64, sensitivity_fp64
        FROM multi_granularity_sensitivity
        WHERE pair_id = 0 AND byte_offset = 1 AND class_id = 2 AND axis = 'pose'
        """
    ).fetchone()
    raw = float(arr[1, 0, 1])
    assert row == pytest.approx((raw, 10.0, 3.0, raw * 10.0 * 3.0, abs(raw * 30.0)))
    run = con.execute(
        "SELECT row_count, class_basis, score_claim, promotion_eligible "
        "FROM multi_granularity_sensitivity_runs"
    ).fetchone()
    assert run == (inserted, "synthetic_segnet_class_weights", False, False)
    con.close()


def test_refresh_multi_granularity_records_blocker_without_class_source(
    tmp_path: Path,
) -> None:
    """Missing class attribution must not duplicate fake per-class rows."""
    repo = tmp_path
    arr = np.ones((4, 2, 3), dtype=np.float64)
    _write_per_pair_anchor(repo, arr)
    con = connect(repo / "test.duckdb")

    assert (
        refresh_multi_granularity_sensitivity(con, repo, byte_offsets=[0])
        == 0
    )
    rows = con.execute("SELECT COUNT(*) FROM multi_granularity_sensitivity").fetchone()[0]
    assert rows == 0
    blocker = con.execute(
        "SELECT blocker_reason, row_count FROM multi_granularity_sensitivity_runs"
    ).fetchone()
    assert blocker == ("class_source_required_for_class_specific_sensitivity", 0)
    con.close()


def test_refresh_multi_granularity_skips_aggregate_and_false_contest_axis(
    tmp_path: Path,
) -> None:
    """Only usable per_pair_per_byte_v1 anchors can populate ITEM_8 cells."""
    repo = tmp_path
    con = connect(repo / "test.duckdb")
    aggregate = np.ones((4, 3), dtype=np.float64)
    _write_per_pair_anchor(repo, aggregate, tensor_kind="aggregate_per_byte_v1")
    class_source = _write_class_source(repo)
    assert (
        refresh_multi_granularity_sensitivity(
            con,
            repo,
            class_source_path=class_source,
            byte_offsets=[0],
        )
        == 0
    )

    false_axis = np.ones((4, 2, 3), dtype=np.float64)
    _write_per_pair_anchor(
        repo,
        false_axis,
        measurement_axis="[contest-CPU]",
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        measurement_method="synthetic_per_pair_subset",
    )
    assert (
        refresh_multi_granularity_sensitivity(
            con,
            repo,
            class_source_path=class_source,
            byte_offsets=[0],
        )
        == 0
    )
    con.close()


def test_refresh_multi_granularity_rejects_shape_and_cardinality_hazards(
    tmp_path: Path,
) -> None:
    """Shape authority and row caps block accidental wrong-domain materialization."""
    repo = tmp_path
    class_source = _write_class_source(repo)
    transposed = np.ones((2, 4, 3), dtype=np.float64)
    _write_per_pair_anchor(repo, transposed)
    con = connect(repo / "test.duckdb")
    with pytest.raises(ValueError, match="shape"):
        refresh_multi_granularity_sensitivity(
            con,
            repo,
            class_source_path=class_source,
            byte_offsets=[0],
        )

    proper = np.ones((4, 2, 3), dtype=np.float64)
    _write_per_pair_anchor(repo, proper)
    with pytest.raises(ValueError, match="max_rows"):
        refresh_multi_granularity_sensitivity(
            con,
            repo,
            class_source_path=class_source,
            byte_offsets=[0, 1, 2, 3],
            max_rows=100,
        )
    con.close()


def test_refresh_multi_granularity_table_bootstrap_operator_entry(
    tmp_path: Path,
) -> None:
    """Operator-facing refresh makes ITEM_8 discoverable without dense materialization."""
    result = refresh_multi_granularity_sensitivity_table(
        tmp_path,
        db_path=tmp_path / "canonical.duckdb",
    )
    assert result["table"] == "multi_granularity_sensitivity"
    assert result["row_count"] == 0
    assert result["source_of_truth"] == (
        "per_pair master_gradient anchor + explicit class source sidecar"
    )


def test_multi_granularity_audit_table_provenance_accepts_extension(
    tmp_path: Path,
) -> None:
    """Canonical provenance audit recognizes ITEM_8 as derived, not source-of-truth."""
    db_path = tmp_path / "canonical.duckdb"
    con = connect(db_path)
    bootstrap_multi_granularity_sensitivity_table(con)
    con.close()
    audit = audit_table_provenance(
        "multi_granularity_sensitivity",
        db_path=db_path,
    )
    assert audit["row_count"] == 0
    assert audit["duckdb_is_source_of_truth"] is False
    assert "class source" in audit["source_of_truth"]


def test_multi_granularity_sensitivity_upsert_query_round_trip(tmp_path: Path) -> None:
    """The ITEM_8 public upsert/query path must be executable, not schema-only."""
    db_path = tmp_path / "test.duckdb"
    con = connect(db_path)
    bootstrap_multi_granularity_sensitivity_table(con)

    archive_sha = "f" * 64
    inserted = upsert_multi_granularity_sensitivity_rows(
        con,
        [
            {
                "pair_id": 7,
                "byte_offset": 11,
                "class_id": 3,
                "axis": "seg",
                "raw_gradient_fp64": -0.25,
                "marginal_coefficient_fp64": 2.0,
                "class_weight_fp64": 0.5,
                "signed_score_delta_unit_fp64": -0.25,
                "sensitivity_fp64": 0.25,
                "derived_at_utc": "2026-05-19T00:00:00Z",
            }
        ],
        run={
            "run_id": "mgs-test-run",
            "archive_sha256": archive_sha,
            "gradient_tensor_kind": "per_pair_per_byte_v1",
            "gradient_array_path": ".omx/state/test_per_pair_gradient.npy",
            "class_basis": "segnet_5class",
            "source_measurement_axis": "[macOS-CPU advisory]",
            "evidence_grade": "diagnostic",
            "source_anchor_authoritative": False,
            "row_count": 1,
        },
    )

    assert inserted == 1
    rows = query_multi_granularity_sensitivity_top_cells(con, archive_sha)
    assert rows == [
        {
            "run_id": "mgs-test-run",
            "archive_sha256": archive_sha,
            "pair_id": 7,
            "byte_offset": 11,
            "class_id": 3,
            "axis": "seg",
            "raw_gradient_fp64": -0.25,
            "marginal_coefficient_fp64": 2.0,
            "class_weight_fp64": 0.5,
            "signed_score_delta_unit_fp64": -0.25,
            "sensitivity_fp64": 0.25,
            "derived_at_utc": rows[0]["derived_at_utc"],
            "class_basis": "segnet_5class",
            "source_measurement_axis": "[macOS-CPU advisory]",
            "evidence_grade": "diagnostic",
            "source_anchor_authoritative": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "blocker_reason": None,
        }
    ]
    con.close()


def test_multi_granularity_query_defaults_to_latest_run_but_preserves_history(
    tmp_path: Path,
) -> None:
    """Historical runs remain stored while normal consumers see latest cells."""
    con = connect(tmp_path / "test.duckdb")
    archive_sha = "a" * 64
    base_run = {
        "archive_sha256": archive_sha,
        "gradient_tensor_kind": "per_pair_per_byte_v1",
        "gradient_array_path": ".omx/state/test_per_pair_gradient.npy",
        "class_basis": "segnet_5class",
        "source_measurement_axis": "[macOS-CPU advisory]",
        "evidence_grade": "diagnostic",
        "source_anchor_authoritative": False,
        "row_count": 1,
    }
    for run_id, utc, sensitivity in (
        ("old-run", "2026-05-19T00:00:00Z", 0.1),
        ("new-run", "2026-05-19T01:00:00Z", 0.2),
    ):
        upsert_multi_granularity_sensitivity_rows(
            con,
            [
                {
                    "pair_id": 0,
                    "byte_offset": 1,
                    "class_id": 2,
                    "axis": "seg",
                    "raw_gradient_fp64": sensitivity,
                    "marginal_coefficient_fp64": 1.0,
                    "class_weight_fp64": 1.0,
                    "sensitivity_fp64": sensitivity,
                    "derived_at_utc": utc,
                }
            ],
            run={**base_run, "run_id": run_id, "source_anchor_utc": utc},
        )

    assert (
        con.execute("SELECT COUNT(*) FROM multi_granularity_sensitivity").fetchone()[0]
        == 2
    )
    latest = query_multi_granularity_sensitivity_top_cells(con, archive_sha)
    assert [row["run_id"] for row in latest] == ["new-run"]
    all_runs = query_multi_granularity_sensitivity_top_cells(
        con,
        archive_sha,
        include_all_runs=True,
    )
    assert {row["run_id"] for row in all_runs} == {"old-run", "new-run"}
    con.close()
