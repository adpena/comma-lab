# SPDX-License-Identifier: MIT
from __future__ import annotations

import zipfile
from pathlib import Path

from tac.optimization.repair_autonomous_multi_archive_runner import (
    REPAIR_AUTONOMOUS_MULTI_ARCHIVE_RUNNER_SCHEMA,
    run_repair_autonomous_multi_archive_runner,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_archive(path: Path, payload: bytes) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", payload)
        zf.writestr("inflate.py", b"print('decode-only')\n")
    return path


def _write_runtime_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    inflate_sh = path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\npython inflate.py \"$@\"\n", encoding="utf-8")
    inflate_sh.chmod(0o755)
    (path / "inflate.py").write_text("print('decode-only')\n", encoding="utf-8")
    return path


def test_multi_archive_runner_executes_and_closes_runtime_custody(
    tmp_path: Path,
) -> None:
    archive_a = _write_archive(tmp_path / "candidate_a.zip", b"A" * 96)
    archive_b = _write_archive(tmp_path / "candidate_b.zip", b"B" * 128)
    runtime_dir = _write_runtime_dir(tmp_path / "runtime")

    summary = run_repair_autonomous_multi_archive_runner(
        archives=[archive_a, archive_b],
        source_labels=["unit_a", "unit_b"],
        source_runtime_dirs=[runtime_dir],
        output_dir=tmp_path / "multi_runner",
        repo_root=REPO_ROOT,
        chain_id="unit_multi_archive_repair",
        queue_id="unit_multi_archive_repair_materialization",
        execute_local=True,
        close_runtime_custody=True,
        worker_max_experiments_per_iteration=10,
        max_steps_per_iteration=120,
        posterior_path=None,
        overwrite=True,
    )

    assert summary["schema"] == REPAIR_AUTONOMOUS_MULTI_ARCHIVE_RUNNER_SCHEMA
    assert summary["archive_count"] == 2
    assert summary["typed_response_count"] == 10
    assert summary["ready_experiment_count"] == 10
    assert summary["queue_validation"]["returncode"] == 0
    assert summary["exact_ready_bridge_candidate_count"] == 10
    assert summary["exact_ready_bridge_runtime_content_tree_custody_proven_count"] == 10
    assert summary["archive_bound_exact_handoff_candidate_count"] == 10
    assert summary["runtime_closure"]["closure_report_count"] == 10
    assert summary["posterior_appended_count_total"] == 0
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["score_claim"] is False
