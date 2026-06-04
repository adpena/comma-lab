from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from comma_lab.evaluate import evaluate_external_submission_dir


def test_external_upstream_eval_gate_hashes_then_cleans_inflated(tmp_path: Path) -> None:
    upstream = _write_fake_upstream(tmp_path)
    submission = _write_fake_submission(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    summary = evaluate_external_submission_dir(
        submission_dir=submission,
        upstream_root=upstream,
        device="cpu",
        artifact_dir=artifact_dir,
        min_free_bytes=0,
        require_upstream_venv=False,
    )
    payload = summary.to_dict()

    assert payload["returncode"] == 0
    assert payload["parsed_report"]["submission_bytes"] == (submission / "archive.zip").stat().st_size
    assert payload["inflated_outputs_manifest"]["file_count"] == 1
    assert payload["inflated_outputs_manifest"]["files"][0]["path"] == "0.raw"
    assert payload["inflated_outputs_manifest"]["certified_rebuildable"] is True
    assert payload["inflated_dir_retained"] is False
    assert payload["inflated_dir_cleanup"] == "deleted_after_success_with_manifest_certificate"
    assert not (submission / "inflated").exists()
    assert (artifact_dir / "report.txt").is_file()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False


def test_snerv_upstream_eval_gate_cli_consumes_bundle_json(tmp_path: Path) -> None:
    upstream = _write_fake_upstream(tmp_path)
    _write_fake_upstream_venv(upstream)
    submission = _write_fake_submission(tmp_path)
    bundle_json = tmp_path / "bundle.json"
    output_json = tmp_path / "snerv_gate.json"
    artifact_dir = tmp_path / "snerv_gate_artifacts"
    bundle_json.write_text(
        json.dumps(
            {
                "schema": "snerv_upstream_submission_bundle_materialization.v1",
                "output_submission_dir": submission.as_posix(),
                "archive_zip": {"path": (submission / "archive.zip").as_posix(), "data_only": True},
                "upstream_contest_contract": {"runtime_source_outside_archive_zip": True},
                "receiver_proof": {"runtime_consumption_proof_passed": True},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_snerv_upstream_eval_gate.py",
            "--bundle-json",
            bundle_json.as_posix(),
            "--upstream-root",
            upstream.as_posix(),
            "--artifact-dir",
            artifact_dir.as_posix(),
            "--output-json",
            output_json.as_posix(),
            "--min-free-bytes",
            "0",
        ],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "snerv_upstream_eval_gate.v1" in result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "snerv_upstream_eval_gate.v1"
    assert payload["evaluation"]["returncode"] == 0
    assert payload["evaluation"]["archive_zip"]["bytes"] == (submission / "archive.zip").stat().st_size
    assert payload["evaluation"]["inflated_dir_retained"] is False
    assert "paired_contest_cpu_cuda_auth_eval_missing" in payload["blockers"]
    assert "pre_submission_compliance_gate_missing" in payload["blockers"]
    assert payload["score_claim"] is False
    feedback_path = Path(payload["candidate_feedback_row_path"])
    assert feedback_path.is_file()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert feedback["schema"] == "nerv_candidate_feedback_row.v1"
    assert feedback["feedback_kind"] == "upstream_eval_gate"
    assert feedback["feedback_scope"] == "full600_upstream_cpu_eval"
    assert feedback["family"] == "snerv"
    assert feedback["measured_archive_bytes"] == (submission / "archive.zip").stat().st_size
    assert feedback["upstream_eval_score"] == 19.58
    assert feedback["scope_matches_candidate"] is False
    assert feedback["context_only"] is True
    assert "snerv_upstream_eval_gate_score_bad" in feedback["direct_feedback_blockers"]
    assert feedback["score_claim"] is False
    assert not (submission / "inflated").exists()


def test_harvest_snerv_upstream_eval_gate_feedback_cli(tmp_path: Path) -> None:
    upstream = _write_fake_upstream(tmp_path)
    _write_fake_upstream_venv(upstream)
    submission = _write_fake_submission(tmp_path)
    bundle_json = tmp_path / "bundle.json"
    gate_json = tmp_path / "snerv_gate.json"
    feedback_json = tmp_path / "snerv_feedback.json"
    artifact_dir = tmp_path / "snerv_gate_artifacts"
    bundle_json.write_text(
        json.dumps(
            {
                "schema": "snerv_upstream_submission_bundle_materialization.v1",
                "output_submission_dir": submission.as_posix(),
                "archive_zip": {"path": (submission / "archive.zip").as_posix(), "data_only": True},
                "upstream_contest_contract": {"runtime_source_outside_archive_zip": True},
                "receiver_proof": {"runtime_consumption_proof_passed": True},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "tools/run_snerv_upstream_eval_gate.py",
            "--bundle-json",
            bundle_json.as_posix(),
            "--upstream-root",
            upstream.as_posix(),
            "--artifact-dir",
            artifact_dir.as_posix(),
            "--output-json",
            gate_json.as_posix(),
            "--min-free-bytes",
            "0",
        ],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/harvest_snerv_upstream_eval_gate_feedback.py",
            "--gate-json",
            gate_json.as_posix(),
            "--output-json",
            feedback_json.as_posix(),
        ],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "upstream_eval_gate" in result.stdout
    feedback = json.loads(feedback_json.read_text(encoding="utf-8"))
    assert feedback["schema"] == "nerv_candidate_feedback_row.v1"
    assert feedback["feedback_kind"] == "upstream_eval_gate"
    assert feedback["upstream_eval_gate_path"] == gate_json.resolve(strict=False).as_posix()
    assert feedback["measured_archive_bytes"] == (submission / "archive.zip").stat().st_size
    assert "snerv_upstream_eval_gate_score_bad" in feedback["direct_feedback_blockers"]
    assert feedback["ready_for_exact_eval_dispatch"] is False


def _write_fake_upstream(tmp_path: Path) -> Path:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "evaluate.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
SUBMISSION_DIR=""
DEVICE="cpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submission-dir|--submission_dir)
      SUBMISSION_DIR="${2%/}"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    *)
      shift ;;
  esac
done
ARCHIVE_ZIP="${SUBMISSION_DIR}/archive.zip"
ARCHIVE_DIR="${SUBMISSION_DIR}/archive"
INFLATED_DIR="${SUBMISSION_DIR}/inflated"
rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR" "$INFLATED_DIR"
unzip -q -o "$ARCHIVE_ZIP" -d "$ARCHIVE_DIR"
cp "$ARCHIVE_DIR/0.bin" "$INFLATED_DIR/0.raw"
SIZE="$(wc -c < "$ARCHIVE_ZIP" | tr -d ' ')"
cat > "${SUBMISSION_DIR}/report.txt" <<EOF
=== Evaluation results over 1 samples ===
  Average PoseNet Distortion: 2.50000000
  Average SegNet Distortion: 0.12500000
  Submission file size: ${SIZE} bytes
  Original uncompressed size: 1,000 bytes
  Compression Rate: 0.12300000
  Final score: formula = 19.58
EOF
echo "fake upstream evaluated on ${DEVICE}"
""",
        encoding="utf-8",
    )
    os.chmod(upstream / "evaluate.sh", 0o755)
    return upstream


def _write_fake_upstream_venv(upstream: Path) -> None:
    venv_bin = upstream / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    python_path = venv_bin / "python"
    try:
        python_path.symlink_to(sys.executable)
    except OSError:
        python_path.write_text("#!/usr/bin/env sh\nexec python \"$@\"\n", encoding="utf-8")
        os.chmod(python_path, 0o755)


def _write_fake_submission(tmp_path: Path) -> Path:
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (submission / "inflate.py").write_text("pass\n", encoding="utf-8")
    with zipfile.ZipFile(submission / "archive.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", b"candidate-raw")
    return submission
