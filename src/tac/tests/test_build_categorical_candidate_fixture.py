from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.categorical_candidate_readiness import (
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    RUNTIME_LOADER_PARITY_CONTRACT,
    audit_categorical_candidate_manifest,
)
from tac.repo_io import read_json, sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_build_categorical_candidate_fixture_is_deterministic_and_blocked(
    tmp_path: Path,
) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    source_sha = "c" * 64

    for out_dir in (out_a, out_b):
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "build_categorical_candidate_fixture.py"),
                "--out-dir",
                str(out_dir),
                "--source-archive-sha256",
                source_sha,
            ],
            check=True,
            cwd=REPO,
            text=True,
        )

    archive_a = out_a / "archive.zip"
    archive_b = out_b / "archive.zip"
    assert sha256_file(archive_a) == sha256_file(archive_b)

    candidate = read_json(out_a / "candidate.json")
    readiness = audit_categorical_candidate_manifest(
        candidate,
        repo_root=REPO,
        manifest_dir=out_a,
    )
    assert candidate["fixture_only"] is True
    assert candidate["candidate_manifest_contract"] == CANDIDATE_MANIFEST_CONTRACT
    assert (
        candidate["runtime_loader_parity"]["runtime_loader_parity_contract"]
        == RUNTIME_LOADER_PARITY_CONTRACT
    )
    assert readiness["runtime_loader_parity"]["accepted"] is True
    assert candidate["score_claim"] is False
    assert readiness["fixture_only"] is True
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert "fixture_only_candidate_not_dispatchable" in readiness["dispatch_blockers"]

    with zipfile.ZipFile(archive_a) as archive:
        member_order = [
            "categorical_payload.bin",
            "class_codebook.json",
            "inflate.sh",
            "runtime_decoder.py",
        ]
        assert archive.namelist() == [
            *member_order,
        ]
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_STORED
        modes = {info.filename: info.external_attr >> 16 for info in archive.infolist()}
        assert modes["inflate.sh"] == 0o755
        assert modes["categorical_payload.bin"] == 0o644

    archive_member_manifest = read_json(out_a / "archive_member_manifest.json")
    assert (
        archive_member_manifest["archive_member_manifest_contract"]
        == ARCHIVE_MEMBER_MANIFEST_CONTRACT
    )
    assert archive_member_manifest["member_order"] == member_order
    assert archive_member_manifest["member_count"] == len(member_order)


def test_build_categorical_candidate_fixture_records_tool_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixture"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_categorical_candidate_fixture.py"),
            "--out-dir",
            str(out_dir),
            "--source-archive-sha256",
            "d" * 64,
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "categorical_candidate_fixture_build"
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["tool_run_manifest"]["tool"] == "tools/build_categorical_candidate_fixture.py"
    assert "fixture_only_candidate_not_dispatchable" in summary["readiness_blockers"]
