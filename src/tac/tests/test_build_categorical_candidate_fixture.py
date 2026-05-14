# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.categorical_candidate_plan import (
    CATEGORICAL_CLASS_CODEBOOK_CONTRACT,
    CATEGORICAL_CONSTRUCTION_PLAN_CONTRACT,
)
from tac.categorical_candidate_readiness import (
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    RUNTIME_LOADER_PARITY_CONTRACT,
    audit_categorical_candidate_manifest,
)
from tac.categorical_label_atoms import build_categorical_typed_label_atoms
from tac.categorical_label_prior_payload_manifest import (
    LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
    LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
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
    assert candidate["runtime_loader_parity"]["runtime_loader_parity_contract"] == RUNTIME_LOADER_PARITY_CONTRACT
    assert readiness["runtime_loader_parity"]["accepted"] is False
    assert "runtime_execution_proof_artifact_missing" in readiness["runtime_loader_parity"]["blockers"]
    assert readiness["label_prior_payload_manifest"]["accepted"] is True
    assert readiness["label_prior_payload_manifest"]["member"] == LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER
    assert readiness["candidate_construction_plan"]["accepted"] is True
    assert readiness["candidate_construction_plan"]["ready_for_exact_eval_dispatch"] is False
    assert candidate["score_claim"] is False
    assert readiness["fixture_only"] is True
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert "fixture_only_candidate_not_dispatchable" in readiness["dispatch_blockers"]
    construction_plan = read_json(out_a / "construction_plan.json")
    assert construction_plan == candidate["candidate_construction_plan"]
    assert construction_plan["construction_plan_contract"] == CATEGORICAL_CONSTRUCTION_PLAN_CONTRACT
    assert construction_plan["candidate_construction_ready"] is True
    assert construction_plan["ready_for_exact_eval_dispatch"] is False
    assert "real_byte_closed_archive_parity_missing" in construction_plan["dispatch_blockers"]
    assert [row["name"] for row in construction_plan["class_rows"]] == [
        "road",
        "lane_markings",
        "undrivable",
        "movable",
        "my_car",
    ]
    assert construction_plan["class_rows"][1]["charged_label_member"] == "class_codebook.json"
    assert construction_plan["class_rows"][1]["openpilot_prior_hint"] == "lane_marking_track_prior"
    assert construction_plan["typed_label_atoms"] == build_categorical_typed_label_atoms()

    with zipfile.ZipFile(archive_a) as archive:
        member_order = [
            "categorical_payload.bin",
            "class_codebook.json",
            "inflate.sh",
            LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
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
    assert archive_member_manifest["archive_member_manifest_contract"] == ARCHIVE_MEMBER_MANIFEST_CONTRACT
    assert archive_member_manifest["member_order"] == member_order
    assert archive_member_manifest["member_count"] == len(member_order)
    with zipfile.ZipFile(archive_a) as archive:
        class_codebook = json.loads(archive.read("class_codebook.json").decode("utf-8"))
        label_prior_payload_manifest = json.loads(
            archive.read(LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER).decode("utf-8")
        )
    assert class_codebook["class_codebook_contract"] == CATEGORICAL_CLASS_CODEBOOK_CONTRACT
    assert class_codebook["classes"][0]["name"] == "road"
    assert class_codebook["classes"][1]["default_quant_bits"] == 8
    assert class_codebook["typed_label_atoms"] == build_categorical_typed_label_atoms()
    assert (
        label_prior_payload_manifest["label_prior_payload_manifest_contract"]
        == LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT
    )
    assert label_prior_payload_manifest["conditioning_priors"] == candidate["conditioning_priors"]
    assert (
        label_prior_payload_manifest["typed_label_atoms"]
        == build_categorical_typed_label_atoms()
    )


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
    assert summary["paths"]["construction_plan"].endswith("construction_plan.json")
