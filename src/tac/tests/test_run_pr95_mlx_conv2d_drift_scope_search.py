# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PR95_SOURCE_MODEL = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src/model.py"
)
PR95_RELEASE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
    / "archive.zip"
)


def test_pr95_mlx_conv2d_drift_scope_search_cli_baseline_only(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlx.core")
    pytest.importorskip("torch")
    if not PR95_SOURCE_MODEL.is_file() or not PR95_RELEASE_ARCHIVE.is_file():
        pytest.skip("public PR95 source artifacts are unavailable")

    output_dir = tmp_path / "scope_search"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_conv2d_drift_scope_search.py"),
            "--archive-zip",
            str(PR95_RELEASE_ARCHIVE),
            "--public-pr95-source-model",
            str(PR95_SOURCE_MODEL),
            "--output-dir",
            str(output_dir),
            "--mlx-device",
            "cpu",
            "--no-presets",
            "--no-single-blocks",
            "--no-prefix-blocks",
            "--atol-max",
            "0.01",
            "--atol-mean",
            "0.001",
            "--cliff-threshold",
            "10",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    summary = json.loads(
        (output_dir / "scope_search_summary.json").read_text(encoding="utf-8")
    )

    assert stdout["ok"] is True
    assert summary["schema"] == "pr95_hnerv_mlx_conv2d_drift_scope_search.v1"
    assert summary["candidate_count"] == 1
    assert summary["best_by_delta_candidate"]["candidate_id"] == "baseline_optimized"
    assert summary["rows"][0]["candidate_id"] == "baseline_optimized"
    assert summary["minimal_passed_candidate"]["candidate_id"] == "baseline_optimized"
    assert summary["minimal_no_cliff_candidate"]["candidate_id"] == "baseline_optimized"
    assert summary["exact_readiness_refusal"]["ready"] is False
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
