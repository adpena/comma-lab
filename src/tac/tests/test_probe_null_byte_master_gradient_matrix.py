# SPDX-License-Identifier: MIT
"""Tests for tools/probe_null_byte_master_gradient_matrix.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.probe_null_byte_master_gradient_matrix import probe_all_anchors


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_anchor(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_probe_all_anchors_matrix_handles_ok_and_missing_npy(tmp_path: Path) -> None:
    grad_path = tmp_path / "grad.npy"
    grad = np.ones((6, 3), dtype=np.float64)
    grad[[0, 2, 4], :] = 0.0
    np.save(grad_path, grad)

    anchors_path = tmp_path / "anchors.jsonl"
    _write_anchor(
        anchors_path,
        [
            {
                "archive_sha256": "6bae0201fb" + "0" * 54,
                "gradient_array_path": str(grad_path),
                "measurement_axis": "[contest-CUDA]",
                "measurement_hardware": "linux_x86_64_t4_modal",
                "n_bytes": 6,
                "n_pairs_used": 1,
            },
            {
                "archive_sha256": "9cb989cef5" + "1" * 54,
                "gradient_array_path": str(tmp_path / "missing.npy"),
                "measurement_axis": "[macOS-CPU advisory]",
                "measurement_hardware": "m5_max",
                "n_bytes": 6,
            },
        ],
    )

    matrix = probe_all_anchors(anchors_jsonl_path=anchors_path, epsilon=1e-9)

    assert matrix["score_claim"] is False
    assert matrix["promotable"] is False
    assert matrix["n_anchors_scanned"] == 2
    assert matrix["n_anchors_probed_ok"] == 1
    assert matrix["n_anchors_missing_npy"] == 1
    ok = matrix["per_anchor"][0]
    assert ok["status"] == "OK"
    assert ok["n_null_bytes"] == 3
    assert ok["substrate_label"] == "pr101_fec6_frontier"
    assert ok["predicted_delta_s_per_seed_budget"]["K=16"] == 0.0


def test_probe_null_byte_master_gradient_matrix_cli(tmp_path: Path) -> None:
    grad_path = tmp_path / "grad.npy"
    grad = np.ones((20, 3), dtype=np.float64)
    grad[:18, :] = 0.0
    np.save(grad_path, grad)

    anchors_path = tmp_path / "anchors.jsonl"
    _write_anchor(
        anchors_path,
        [
            {
                "archive_sha256": "6bae0201fb" + "0" * 54,
                "gradient_array_path": str(grad_path),
                "measurement_axis": "[contest-CUDA]",
                "measurement_hardware": "linux_x86_64_t4_modal",
                "n_bytes": 20,
                "n_pairs_used": 1,
            }
        ],
    )
    output_dir = tmp_path / "matrix"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "probe_null_byte_master_gradient_matrix.py"),
            "--anchors-jsonl",
            str(anchors_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "probed_ok=1" in completed.stderr
    matrix = json.loads((output_dir / "null_byte_matrix.json").read_text(encoding="utf-8"))
    assert matrix["n_anchors_probed_ok"] == 1
    assert matrix["score_claim"] is False
    assert matrix["promotion_eligible"] is False
    assert matrix["rank_or_kill_eligible"] is False
    assert matrix["promotable"] is False
    assert matrix["axis_tag"] == "[predicted]"
    top = matrix["top5_replacement_candidates"][0]
    for key in (
        "substrate_label",
        "codec_family",
        "scored_archive_sha256",
        "axis",
        "anchor_index",
        "n_null_bytes",
        "null_fraction",
        "predicted_delta_s_per_seed_budget",
    ):
        assert key in top
    assert "K=16" in top["predicted_delta_s_per_seed_budget"]
    assert matrix["provenance"]["score_claim"] is False
    assert (output_dir / "null_byte_matrix.md").read_text(encoding="utf-8").startswith(
        "# Null-byte master-gradient probe matrix"
    )


def test_codec_family_rollups_aggregate_correctly(tmp_path: Path) -> None:
    """Multiple anchors of same family produce mean+stddev rollup with correct counts."""
    grad1 = tmp_path / "hn1.npy"
    grad2 = tmp_path / "hn2.npy"
    grad3 = tmp_path / "pr106.npy"
    for path, n_null in [(grad1, 5), (grad2, 10), (grad3, 8)]:
        arr = np.ones((100, 3), dtype=np.float64)
        arr[list(range(n_null)), :] = 0.0
        np.save(path, arr)

    anchors_path = tmp_path / "anchors.jsonl"
    _write_anchor(
        anchors_path,
        [
            {
                "archive_sha256": "6bae0201fb" + "0" * 54,  # hnerv
                "gradient_array_path": str(grad1),
                "measurement_axis": "[contest-CUDA]",
                "measurement_hardware": "linux_x86_64_t4_modal",
                "n_bytes": 100,
                "n_pairs_used": 1,
            },
            {
                "archive_sha256": "6bae0201fb" + "0" * 54,  # hnerv (same family)
                "gradient_array_path": str(grad2),
                "measurement_axis": "[contest-CUDA]",
                "measurement_hardware": "linux_x86_64_t4_modal",
                "n_bytes": 100,
                "n_pairs_used": 1,
            },
            {
                "archive_sha256": "9cb989cef5" + "1" * 54,  # pr106
                "gradient_array_path": str(grad3),
                "measurement_axis": "[macOS-CPU advisory]",
                "measurement_hardware": "m5_max",
                "n_bytes": 100,
                "n_pairs_used": 1,
            },
        ],
    )

    matrix = probe_all_anchors(anchors_jsonl_path=anchors_path, epsilon=1e-9)
    rollups = matrix["codec_family_rollups"]
    assert "hnerv_family" in rollups
    assert "pr106_format0d_family" in rollups
    assert rollups["hnerv_family"]["anchor_count"] == 2
    # Mean of 0.05 and 0.10 = 0.075
    assert abs(rollups["hnerv_family"]["null_fraction_mean"] - 0.075) < 1e-9
    assert rollups["pr106_format0d_family"]["anchor_count"] == 1
    assert abs(rollups["pr106_format0d_family"]["null_fraction_mean"] - 0.08) < 1e-9


def test_empty_anchors_ledger_handled(tmp_path: Path) -> None:
    """Empty anchors JSONL must produce empty matrix (not crash)."""
    anchors_path = tmp_path / "empty.jsonl"
    anchors_path.write_text("")

    matrix = probe_all_anchors(anchors_jsonl_path=anchors_path, epsilon=1e-9)
    assert matrix["n_anchors_scanned"] == 0
    assert matrix["n_anchors_probed_ok"] == 0
    assert matrix["per_anchor"] == []
    assert matrix["top5_replacement_candidates"] == []
    assert matrix["codec_family_rollups"] == {}


def test_live_repo_11_anchor_ledger_probes_canonical_fec6_frontier(tmp_path: Path) -> None:
    """Live regression: the canonical 11-anchor ledger probes cleanly AND the
    fec6 frontier [contest-CUDA] anchor matches the canonical equation's
    registered empirical anchor (16292 null bytes / 9.1314% null fraction).

    Per task #1111: pins the empirical anchor underlying
    ``tac.canonical_equations.master_gradient_null_space_byte_fraction_v1``.
    """
    import pytest

    ledger_path = REPO_ROOT / ".omx" / "state" / "master_gradient_anchors.jsonl"
    if not ledger_path.is_file():
        pytest.skip(f"canonical ledger not present at {ledger_path}")

    matrix = probe_all_anchors(anchors_jsonl_path=ledger_path, epsilon=1e-9)
    assert matrix["n_anchors_scanned"] >= 8, (
        f"expected >=8 anchors in canonical ledger; got {matrix['n_anchors_scanned']}"
    )
    # Most npys must be reachable
    assert matrix["n_anchors_probed_ok"] >= matrix["n_anchors_scanned"] - 2, (
        "more than 2 anchors are MISSING_NPY; investigate ledger or extraction wave"
    )
    # Pin the canonical empirical anchor: every fec6 frontier OP3-V3 contest-CUDA
    # anchor MUST land at 16292 null bytes per the canonical equation registration
    fec6_contest_cuda_rows = [
        r
        for r in matrix["per_anchor"]
        if r["status"] == "OK"
        and r["scored_archive_sha256"].startswith("6bae0201fb08")
        and r["axis"] == "[contest-CUDA]"
    ]
    if fec6_contest_cuda_rows:
        anchor = fec6_contest_cuda_rows[0]
        assert anchor["n_null_bytes"] == 16292, (
            f"fec6 frontier [contest-CUDA] expected n_null=16292 per canonical "
            f"equation registration; got {anchor['n_null_bytes']}"
        )
        assert abs(anchor["null_fraction"] - 0.09131416849291267) < 1e-9, (
            f"fec6 frontier [contest-CUDA] null_fraction drift: "
            f"expected 9.131416% per canonical equation registration; "
            f"got {anchor['null_fraction']*100:.4f}%"
        )
