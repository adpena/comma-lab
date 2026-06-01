# SPDX-License-Identifier: MIT
"""Tests for the Z8 full ``inflate.sh`` runtime benchmark."""

from __future__ import annotations

from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.inflate_runtime_benchmark import (
    benchmark_z8_submission_inflate_runtime,
)


def _write_runtime(path: Path, *, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_full_inflate_benchmark_runs_receiver_shell_and_hashes_outputs(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "submission"
    archive_dir.mkdir()
    (archive_dir / "0.bin").write_bytes(b"z8-payload")
    inflate_sh = _write_runtime(
        archive_dir / "inflate.sh",
        body="""#!/usr/bin/env bash
set -euo pipefail
archive_dir="$1"
output_dir="$2"
file_list="$3"
while IFS= read -r name; do
  [ -z "$name" ] && continue
  target="$output_dir/$name.raw"
  mkdir -p "$(dirname "$target")"
  cat "$archive_dir/0.bin" > "$target"
  printf '%s' "$name" >> "$target"
done < "$file_list"
""",
    )
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0\nnested/1\n", encoding="utf-8")

    report = benchmark_z8_submission_inflate_runtime(
        inflate_sh=inflate_sh,
        archive_dir=archive_dir,
        file_list=file_list,
        output_dir=tmp_path / "inflate_bench",
        repeat=1,
        timeout_seconds=5.0,
        auth_eval_window_seconds=1800.0,
        inflate_device="cpu",
    )

    assert report["schema"] == "z8_submission_inflate_runtime_benchmark.v1"
    assert report["benchmark_scope"] == "full_submission_inflate_sh_runtime"
    assert report["receiver_path_exercised"] is True
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["successful_runs"] == 1
    assert report["inflate_seconds_best"] is not None
    assert report["output_retention_policy"] == "manifest_then_delete"
    assert report["large_artifact_cleanup_default"] is True
    assert report["file_list_entries"] == ["0", "nested/1"]
    assert report["archive_member_manifest"]["file_count"] == 2
    assert report["runs"][0]["returncode"] == 0
    assert report["runs"][0]["output_manifest"]["file_count"] == 2
    assert report["runs"][0]["output_manifest"]["total_bytes"] > 0
    assert report["runs"][0]["output_retained"] is False
    assert report["runs"][0]["output_cleanup_blocker"] is None
    assert not (tmp_path / "inflate_bench" / "run_000").exists()
    assert "auth_evaluator_not_run" in report["blockers"]
    assert "contest_cpu_cuda_score_not_measured" in report["blockers"]
    assert "inflate_sh_returned_nonzero" not in report["blockers"]
    assert "inflate_output_count_below_file_list_count" not in report["blockers"]


def test_full_inflate_benchmark_fails_closed_on_receiver_error(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "submission"
    archive_dir.mkdir()
    (archive_dir / "0.bin").write_bytes(b"z8-payload")
    inflate_sh = _write_runtime(
        archive_dir / "inflate.sh",
        body="""#!/usr/bin/env bash
set -euo pipefail
echo 'receiver failed' >&2
exit 9
""",
    )
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0\n", encoding="utf-8")

    report = benchmark_z8_submission_inflate_runtime(
        inflate_sh=inflate_sh,
        archive_dir=archive_dir,
        file_list=file_list,
        output_dir=tmp_path / "inflate_bench",
        repeat=1,
        timeout_seconds=5.0,
        inflate_device="cpu",
    )

    assert report["successful_runs"] == 0
    assert report["inflate_seconds_best"] is None
    assert report["runs"][0]["returncode"] == 9
    assert "receiver failed" in report["runs"][0]["stderr_tail"]
    assert "inflate_sh_returned_nonzero" in report["blockers"]
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
