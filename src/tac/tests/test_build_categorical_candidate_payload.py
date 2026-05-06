from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from tac.categorical_candidate_plan import CATEGORICAL_CLASS_CODEBOOK_CONTRACT
from tac.categorical_candidate_readiness import (
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    DECODE_REENCODE_PARITY_CONTRACT,
    HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
    RUNTIME_LOADER_PARITY_CONTRACT,
    audit_categorical_candidate_manifest,
)
from tac.categorical_payload_candidate import (
    MEMBER_ORDER,
    RUNTIME_PROOF_SKELETON_CONTRACT,
)
from tac.pr91_hpm1_codec import build_hpm1_mask_segment
from tac.repo_io import read_json, sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_build_categorical_candidate_payload_is_byte_closed_and_fail_closed(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "categorical_payload.bin"
    payload_path.write_bytes(b"HPM1_local_categorical_payload_bytes\n")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    for out_dir in (out_a, out_b):
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "build_categorical_candidate_payload.py"),
                "--out-dir",
                str(out_dir),
                "--payload-source",
                "file",
                "--categorical-payload",
                str(payload_path),
                "--source-archive-sha256",
                "e" * 64,
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
    readiness_file = read_json(out_a / "readiness.json")
    summary = read_json(out_a / "summary.json")

    assert candidate["fixture_only"] is False
    assert candidate["candidate_manifest_contract"] == CANDIDATE_MANIFEST_CONTRACT
    assert candidate["runtime_loader_parity"]["runtime_loader_parity_contract"] == RUNTIME_LOADER_PARITY_CONTRACT
    assert candidate["decode_reencode_parity"]["decode_reencode_parity_contract"] == DECODE_REENCODE_PARITY_CONTRACT
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["fixture_only"] is False
    assert readiness["candidate_archive"]["untracked_members"] == []
    assert readiness["candidate_archive"]["member_order_matches_manifest"] is True
    assert readiness["candidate_archive"]["zip_determinism_contract"]["passed"] is True
    assert readiness["archive_member_manifest"]["member_order_matches_charged_members"] is True
    assert readiness["archive_member_manifest"]["member_count_matches_charged_members"] is True
    assert readiness["runtime_loader_parity"]["accepted"] is False
    assert readiness["runtime_loader_parity"]["sidecar_free"] is True
    assert readiness["decode_reencode_parity"]["accepted"] is False
    assert readiness_file["ready_for_exact_eval_dispatch"] is False
    assert readiness_file["tool_run_manifest"]["tool"] == "tools/build_categorical_candidate_payload.py"
    blockers = set(readiness["dispatch_blockers"])
    assert "runtime_loader_parity_not_passed" in blockers
    assert "decode_reencode_parity_not_passed" in blockers
    assert "decode_reencode_full_decode_not_proven" in blockers
    assert "decode_reencode_byte_exact_reencode_not_proven" in blockers
    assert "no_op_control_not_passed:decode_reencode_identity_control" in blockers
    assert "no_op_control_not_passed:runtime_consumes_conditioning_control" in blockers

    archive_member_manifest = read_json(out_a / "archive_member_manifest.json")
    assert archive_member_manifest["archive_member_manifest_contract"] == ARCHIVE_MEMBER_MANIFEST_CONTRACT
    assert archive_member_manifest["member_order"] == list(MEMBER_ORDER)
    assert archive_member_manifest["member_count"] == len(MEMBER_ORDER)
    assert {record["name"] for record in archive_member_manifest["members"]} == set(MEMBER_ORDER)

    with zipfile.ZipFile(archive_a) as archive:
        assert archive.namelist() == list(MEMBER_ORDER)
        class_codebook = json.loads(archive.read("class_codebook.json").decode("utf-8"))
        proof_skeleton = json.loads(
            archive.read("runtime_consumer_proof_skeleton.json").decode("utf-8")
        )
    assert class_codebook["class_codebook_contract"] == CATEGORICAL_CLASS_CODEBOOK_CONTRACT
    assert proof_skeleton["runtime_consumer_proof_skeleton_contract"] == RUNTIME_PROOF_SKELETON_CONTRACT
    assert proof_skeleton["ready_for_exact_eval_dispatch"] is False
    assert proof_skeleton["proof_status"]["archive_contains_payload_codebook_and_runtime"] is True
    assert proof_skeleton["proof_status"]["full_decode_reencode_parity"] is False
    assert proof_skeleton["proof_status"]["runtime_output_parity"] is False

    assert summary["kind"] == "categorical_byte_closed_local_candidate_build"
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["tool_run_manifest"]["tool"] == "tools/build_categorical_candidate_payload.py"
    assert summary["archive_sha256"] == sha256_file(archive_a)
    assert "runtime_loader_parity_not_passed" in summary["readiness_blockers"]


def test_build_categorical_candidate_payload_emits_hpm1_structural_inventory(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "categorical_payload.bin"
    payload = build_hpm1_mask_segment(
        (np.arange(16, dtype=np.uint32) % 5).tobytes(),
        b"synthetic-hpac-ppmd",
        N=2,
        H=4,
        W=4,
        P=2,
        delta=1,
        ch=4,
        use_spm=True,
        hpac_d_film=2,
    )
    payload_path.write_bytes(payload)
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_categorical_candidate_payload.py"),
            "--out-dir",
            str(out_dir),
            "--payload-source",
            "file",
            "--categorical-payload",
            str(payload_path),
            "--source-archive-sha256",
            "e" * 64,
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    candidate = read_json(out_dir / "candidate.json")
    readiness = read_json(out_dir / "readiness.json")
    summary = read_json(out_dir / "summary.json")
    inventory = read_json(out_dir / "hpm1_structural_inventory.json")

    inventory_record = candidate["hpm1_structural_decode_inventory"]
    assert inventory_record["contract"] == HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT
    assert inventory_record["payload_member_sha256"] == candidate["decode_reencode_parity"][
        "payload_member_sha256"
    ]
    assert inventory_record["structural_reencode_matches_source"] is True
    assert inventory_record["full_decode_proven"] is False
    assert inventory_record["byte_exact_semantic_reencode_proven"] is False
    assert inventory["structural_reencode"]["matches_source_segment"] is True
    assert inventory["full_decode"]["passed"] is False
    assert inventory["byte_exact_semantic_reencode"]["passed"] is False
    assert readiness["hpm1_structural_decode_inventory"]["accepted"] is True
    assert readiness["hpm1_structural_decode_inventory"][
        "structural_reencode_matches_source"
    ] is True
    assert "hpm1_structural_inventory" in summary["paths"]
    assert (
        "decode_reencode_full_decode_not_proven"
        in readiness["dispatch_blockers"]
    )
    assert (
        "decode_reencode_byte_exact_reencode_not_proven"
        in readiness["dispatch_blockers"]
    )
