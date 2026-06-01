from __future__ import annotations

import json
import zipfile
from pathlib import Path

from comma_lab.local_submission_replay import (
    run_local_submission_replay,
    stage_local_replay_submission,
)


def _write_fake_upstream(root: Path, *, fail_after_raw: bool = False) -> None:
    (root / ".venv" / "bin").mkdir(parents=True)
    python_bin = root / ".venv" / "bin" / "python"
    python_bin.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    python_bin.chmod(0o755)
    (root / "public_test_video_names.txt").write_text("0.mkv\n", encoding="utf-8")
    fail_block = "exit 17" if fail_after_raw else ""
    (root / "evaluate.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
SUBMISSION_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submission-dir) SUBMISSION_DIR="$2"; shift 2 ;;
    --video-names-file) shift 2 ;;
    --device) shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$SUBMISSION_DIR/inflated" "$SUBMISSION_DIR/archive"
printf raw > "$SUBMISSION_DIR/inflated/0.raw"
FAIL_BLOCK
cat > "$SUBMISSION_DIR/report.txt" <<'EOF'
=== Evaluation results over 1 samples ===
  Average PoseNet Distortion: 0.00040000
  Average SegNet Distortion: 0.00100000
  Submission file size: 1,234 bytes
  Original uncompressed size: 10,000 bytes
  Compression Rate: 0.12340000
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 3.25
EOF
""".replace("FAIL_BLOCK", fail_block),
        encoding="utf-8",
    )
    (root / "evaluate.sh").chmod(0o755)


def test_stage_local_replay_submission_copies_runtime_and_archive(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (runtime / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"payload")

    submission = stage_local_replay_submission(
        runtime_submission_dir=runtime,
        archive_zip_path=archive,
        output_dir=tmp_path / "replay",
    )

    assert (submission / "inflate.sh").is_file()
    assert (submission / "inflate.py").is_file()
    assert (submission / "archive.zip").read_bytes() == archive.read_bytes()


def test_run_local_submission_replay_cleans_raw_scratch(tmp_path: Path, monkeypatch) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr("comma_lab.local_submission_replay.repo_root", lambda: fake_repo)
    upstream = tmp_path / "upstream"
    _write_fake_upstream(upstream)

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"payload")
    submission = stage_local_replay_submission(
        runtime_submission_dir=runtime,
        archive_zip_path=archive,
        output_dir=tmp_path / "replay",
    )

    summary = run_local_submission_replay(
        submission_dir=submission,
        source_runtime_submission_dir=runtime,
        archive_zip_path=archive,
        upstream_root=upstream,
    )

    assert summary.evaluation_passed is True
    assert summary.inflated_dir_cleanup == "deleted_after_success"
    assert not Path(summary.inflated_dir).exists()
    assert summary.archive_extract_dir_cleanup == "deleted_after_success"
    cleanup_manifest = Path(summary.scratch_cleanup_manifest_path)
    assert cleanup_manifest.is_file()
    cleanup = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
    assert cleanup["schema"] == "local_submission_replay_scratch_cleanup_manifest.v1"
    assert cleanup["cleanup_reason"] == "deleted_after_success"
    assert cleanup["inflated_total_bytes"] == 3
    assert cleanup["files"]["inflated"][0]["relative_path"] == "0.raw"
    assert cleanup["files"]["inflated"][0]["sha256"]
    assert summary.local_score_estimate is not None
    assert summary.axis_tag == "[macOS-CPU advisory]"
    assert summary.score_claim is False
    assert summary.ready_for_exact_eval_dispatch is False


def test_run_local_submission_replay_retains_raw_scratch_after_failure_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr("comma_lab.local_submission_replay.repo_root", lambda: fake_repo)
    upstream = tmp_path / "upstream"
    _write_fake_upstream(upstream, fail_after_raw=True)

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"payload")
    submission = stage_local_replay_submission(
        runtime_submission_dir=runtime,
        archive_zip_path=archive,
        output_dir=tmp_path / "replay",
    )

    summary = run_local_submission_replay(
        submission_dir=submission,
        source_runtime_submission_dir=runtime,
        archive_zip_path=archive,
        upstream_root=upstream,
    )

    assert summary.evaluation_passed is False
    assert "local_replay_returncode:17" in summary.blockers
    assert summary.inflated_dir_cleanup == "retained_after_failed_replay"
    assert Path(summary.inflated_dir).exists()
    assert summary.archive_extract_dir_cleanup == "retained_after_failed_replay"
    cleanup_manifest = Path(summary.scratch_cleanup_manifest_path)
    assert cleanup_manifest.is_file()
    cleanup = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
    assert cleanup["cleanup_reason"] == "retained_after_failed_replay"
    assert cleanup["deleted_after_manifest"] is False
    assert cleanup["returncode"] == 17
    assert cleanup["stdout"]["sha256"]
    assert cleanup["stderr"]["sha256"]
    assert cleanup["selected_replay_env"]["COMMA_CHALLENGE_ROOT"] == str(upstream)
    assert cleanup["source_runtime_file_count"] >= 1


def test_run_local_submission_replay_certified_failed_cleanup_deletes_scratch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr("comma_lab.local_submission_replay.repo_root", lambda: fake_repo)
    upstream = tmp_path / "upstream"
    _write_fake_upstream(upstream, fail_after_raw=True)

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"payload")
    submission = stage_local_replay_submission(
        runtime_submission_dir=runtime,
        archive_zip_path=archive,
        output_dir=tmp_path / "replay",
    )

    summary = run_local_submission_replay(
        submission_dir=submission,
        source_runtime_submission_dir=runtime,
        archive_zip_path=archive,
        upstream_root=upstream,
        cleanup_failed_scratch=True,
        certify_failed_scratch_rebuildable=True,
    )

    assert summary.evaluation_passed is False
    assert summary.inflated_dir_cleanup == "deleted_after_failed_replay_certified_rebuildable"
    assert not Path(summary.inflated_dir).exists()
    assert (
        summary.archive_extract_dir_cleanup
        == "deleted_after_failed_replay_certified_rebuildable"
    )
    cleanup = json.loads(
        Path(summary.scratch_cleanup_manifest_path).read_text(encoding="utf-8")
    )
    assert cleanup["cleanup_reason"] == "deleted_after_failed_replay_certified_rebuildable"
    assert cleanup["deleted_after_manifest"] is True


def test_run_local_submission_replay_rejects_uncertified_failed_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr("comma_lab.local_submission_replay.repo_root", lambda: fake_repo)
    upstream = tmp_path / "upstream"
    _write_fake_upstream(upstream)

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"payload")
    submission = stage_local_replay_submission(
        runtime_submission_dir=runtime,
        archive_zip_path=archive,
        output_dir=tmp_path / "replay",
    )

    try:
        run_local_submission_replay(
            submission_dir=submission,
            source_runtime_submission_dir=runtime,
            archive_zip_path=archive,
            upstream_root=upstream,
            cleanup_failed_scratch=True,
        )
    except ValueError as exc:
        assert "certify_failed_scratch_rebuildable" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("uncertified failed cleanup should be rejected")
