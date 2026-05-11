from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "tools" / "build_factorized_hnerv_archive.py"
COMPLIANCE = REPO_ROOT / "scripts" / "pre_submission_compliance_check.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_build_closes_submission_surface(tmp_path: Path) -> None:
    out_dir = tmp_path / "factorized"
    proc = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--synthetic-substrate",
            "--output-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    submission_dir = out_dir / "submission_dir"
    archive_zip = submission_dir / "archive.zip"
    report = submission_dir / "report.txt"
    archive_manifest = submission_dir / "archive_manifest.json"
    build_manifest = out_dir / "build_manifest.json"
    assert archive_zip.is_file()
    assert report.is_file()
    assert archive_manifest.is_file()
    assert build_manifest.is_file()

    archive_sha = _sha256(archive_zip)
    archive_bytes = archive_zip.stat().st_size
    manifest = json.loads(archive_manifest.read_text(encoding="utf-8"))
    build = json.loads(build_manifest.read_text(encoding="utf-8"))
    assert manifest["archive_sha256"] == archive_sha
    assert manifest["archive_size_bytes"] == archive_bytes
    assert build["archive_sha256"] == archive_sha
    assert build["archive_zip_sha256"] == archive_sha
    assert build["submission_custody"]["archive_zip_sha256"] == archive_sha
    assert f"archive_sha256: {archive_sha}" in report.read_text(encoding="utf-8")
    assert f"archive_size_bytes: {archive_bytes}" in report.read_text(encoding="utf-8")

    with zipfile.ZipFile(archive_zip) as zf:
        assert zf.namelist() == ["0.bin"]
        inner = zf.read("0.bin")
    assert build["archive_payload_sha256"] == hashlib.sha256(inner).hexdigest()
    assert manifest["archive_payload_sha256"] == build["archive_payload_sha256"]

    compliance_json = out_dir / "pre_submission_compliance.nonfinal.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(COMPLIANCE),
            "--strict",
            "--submission-dir",
            str(submission_dir),
            "--archive",
            str(archive_zip),
            "--expected-archive-sha256",
            archive_sha,
            "--expected-archive-size-bytes",
            str(archive_bytes),
            "--expect-single-member",
            "0.bin",
            "--json-out",
            str(compliance_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    compliance = json.loads(compliance_json.read_text(encoding="utf-8"))
    assert compliance["passed"] is True
    failed_errors = [
        row["name"]
        for row in compliance["checks"]
        if not row["passed"] and row["severity"] == "error"
    ]
    assert failed_errors == []
