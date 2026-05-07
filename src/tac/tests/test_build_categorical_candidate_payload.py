from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.categorical_candidate_plan import CATEGORICAL_CLASS_CODEBOOK_CONTRACT
from tac.categorical_candidate_readiness import (
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    DECODE_REENCODE_INDEPENDENT_PROOF_KIND,
    DECODE_REENCODE_PARITY_CONTRACT,
    HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
    RUNTIME_EXECUTION_PROOF_KIND,
    RUNTIME_LOADER_PARITY_CONTRACT,
    audit_categorical_candidate_manifest,
)
from tac.categorical_label_atoms import build_categorical_typed_label_atoms
from tac.categorical_label_prior_payload_manifest import (
    LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
    LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
)
from tac.categorical_payload_candidate import (
    DECODE_REENCODE_BLOCKED_PROOF_FILENAME,
    LABEL_PERMUTATION_CONTROL_FILENAME,
    MEMBER_ORDER,
    RUNTIME_EXECUTION_PROOF_FILENAME,
    RUNTIME_PROOF_SKELETON_CONTRACT,
)
from tac.pr91_hpm1_codec import build_hpm1_mask_segment
from tac.repo_io import read_json, sha256_file

REPO = Path(__file__).resolve().parents[3]


def _valid_hpm1_hpac_payload() -> bytes:
    torch = pytest.importorskip("torch")
    pyppmd = pytest.importorskip("pyppmd")
    from tac.pr86_hpac_codec import PPMD_MAX_ORDER, PPMD_MEM_SIZE, HPACMini

    model = HPACMini(num_pairs=2, P=2, delta=1, ch=4, d_film=2, use_spm=True)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        for buffer in model.buffers():
            if torch.is_floating_point(buffer):
                buffer.zero_()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return pyppmd.compress(
        buf.getvalue(),
        max_order=PPMD_MAX_ORDER,
        mem_size=PPMD_MEM_SIZE,
    )


def test_build_categorical_candidate_payload_is_byte_closed_and_fail_closed(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "categorical_payload.bin"
    payload_path.write_bytes(b"opaque_local_categorical_payload_bytes\n")
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
    assert candidate["runtime_loader_parity"]["passed"] is True
    assert readiness["runtime_loader_parity"]["accepted"] is True
    assert readiness["runtime_loader_parity"]["sidecar_free"] is True
    assert readiness["runtime_loader_parity"]["runtime_execution_proof"]["accepted"] is True
    assert (
        readiness["runtime_loader_parity"]["runtime_execution_proof"][
            "hpm1_runtime_consumer_proof"
        ]["required"]
        is False
    )
    assert readiness["decode_reencode_parity"]["accepted"] is False
    assert readiness["conditioning_prior_contract"]["passed"] is True
    assert readiness["label_prior_payload_manifest"]["accepted"] is True
    assert readiness["label_prior_payload_manifest"]["member"] == LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER
    assert readiness_file["ready_for_exact_eval_dispatch"] is False
    assert readiness_file["tool_run_manifest"]["tool"] == "tools/build_categorical_candidate_payload.py"
    blockers = set(readiness["dispatch_blockers"])
    assert "runtime_loader_parity_not_passed" not in blockers
    assert "runtime_execution_proof_artifact_missing" not in blockers
    assert "decode_reencode_parity_not_passed" in blockers
    assert "decode_reencode_full_decode_not_proven" in blockers
    assert "decode_reencode_byte_exact_reencode_not_proven" in blockers
    assert "decode_reencode_independent_proof_artifact_missing" not in blockers
    assert "decode_reencode_independent_proof_full_decode_not_proven" in blockers
    assert "decode_reencode_independent_proof_byte_exact_reencode_not_proven" in blockers
    assert "no_op_control_not_passed:decode_reencode_identity_control" in blockers
    assert "no_op_control_not_passed:label_permutation_fail_closed_control" not in blockers
    assert "no_op_control_not_passed:runtime_consumes_conditioning_control" not in blockers
    assert candidate["no_op_controls"]["label_permutation_fail_closed_control"]["passed"] is True

    archive_member_manifest = read_json(out_a / "archive_member_manifest.json")
    assert archive_member_manifest["archive_member_manifest_contract"] == ARCHIVE_MEMBER_MANIFEST_CONTRACT
    assert archive_member_manifest["member_order"] == list(MEMBER_ORDER)
    assert archive_member_manifest["member_count"] == len(MEMBER_ORDER)
    assert {record["name"] for record in archive_member_manifest["members"]} == set(MEMBER_ORDER)
    charged_by_name = {record["name"]: record for record in archive_member_manifest["members"]}

    prior_by_name = {row["name"]: row for row in candidate["conditioning_priors"]}
    qma9_prior = prior_by_name["local_categorical_payload"]
    assert qma9_prior["charged_member"] == "categorical_payload.bin"
    assert qma9_prior["charged_member_sha256"] == charged_by_name["categorical_payload.bin"]["sha256"]
    assert qma9_prior["source_provenance"]["kind"] == "charged_archive_member"
    assert qma9_prior["source_provenance"]["charged_member"] == "categorical_payload.bin"
    assert qma9_prior["source_provenance"]["sha256"] == charged_by_name["categorical_payload.bin"]["sha256"]
    clade_prior = prior_by_name["canonical_class_codebook_conditioning"]
    assert clade_prior["charged_member"] == "class_codebook.json"
    assert clade_prior["charged_member_sha256"] == charged_by_name["class_codebook.json"]["sha256"]
    assert clade_prior["source_provenance"]["kind"] == "charged_archive_member"
    openpilot_prior = prior_by_name["ego_lane_atom_ranker"]
    assert openpilot_prior["runtime_consumed"] is False
    assert openpilot_prior["source_provenance"]["kind"] == "compression_time_only_derivation"
    label_prior_manifest_record = candidate["label_prior_payload_manifest"]
    assert label_prior_manifest_record["member"] == LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER
    assert label_prior_manifest_record["contract"] == LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT

    with zipfile.ZipFile(archive_a) as archive:
        assert archive.namelist() == list(MEMBER_ORDER)
        class_codebook = json.loads(archive.read("class_codebook.json").decode("utf-8"))
        label_prior_payload_manifest = json.loads(
            archive.read(LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER).decode("utf-8")
        )
        proof_skeleton = json.loads(archive.read("runtime_consumer_proof_skeleton.json").decode("utf-8"))
    assert class_codebook["class_codebook_contract"] == CATEGORICAL_CLASS_CODEBOOK_CONTRACT
    assert class_codebook["typed_label_atoms"] == build_categorical_typed_label_atoms()
    assert (
        label_prior_payload_manifest["label_prior_payload_manifest_contract"]
        == LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT
    )
    assert (
        label_prior_payload_manifest["typed_label_atoms"]
        == build_categorical_typed_label_atoms()
    )
    assert label_prior_payload_manifest["conditioning_priors"] == candidate["conditioning_priors"]
    assert label_prior_payload_manifest["label_contract"] == "contest_zero_based_comma10k_order"
    assert proof_skeleton["runtime_consumer_proof_skeleton_contract"] == RUNTIME_PROOF_SKELETON_CONTRACT
    assert proof_skeleton["ready_for_exact_eval_dispatch"] is False
    assert proof_skeleton["proof_status"]["archive_contains_payload_codebook_and_runtime"] is True
    assert proof_skeleton["proof_status"]["charged_label_prior_payload_manifest"] is True
    assert proof_skeleton["proof_status"]["runtime_consumes_charged_members"] is True
    assert proof_skeleton["proof_status"]["full_decode_reencode_parity"] is False

    assert summary["kind"] == "categorical_byte_closed_local_candidate_build"
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["tool_run_manifest"]["tool"] == "tools/build_categorical_candidate_payload.py"
    assert summary["archive_sha256"] == sha256_file(archive_a)
    assert "runtime_loader_parity_not_passed" not in summary["readiness_blockers"]
    runtime_proof = read_json(out_a / RUNTIME_EXECUTION_PROOF_FILENAME)
    assert runtime_proof["kind"] == RUNTIME_EXECUTION_PROOF_KIND
    assert runtime_proof["independent_proof"] is True
    assert runtime_proof["executed_archive_inflate"] is True
    assert runtime_proof["expected_fail_closed_exit"] is True
    assert runtime_proof["runtime_executed"] is True
    assert runtime_proof["sidecar_free"] is True
    assert runtime_proof["fallback_used"] is False
    assert runtime_proof["runtime_report_summary"]["typed_label_atom_count"] == 5
    assert runtime_proof["runtime_report_summary"]["payload_codec"] == "opaque_categorical_payload"
    label_control = read_json(out_a / LABEL_PERMUTATION_CONTROL_FILENAME)
    assert label_control["kind"] == "categorical_label_permutation_fail_closed_control"
    assert label_control["passed"] is True
    assert label_control["mutation"]["operation"] == "reverse_classes_order"
    assert label_control["failure_contract"]["fail_closed"] is True
    assert summary["label_permutation_control"]["passed"] is True
    blocked_proof = read_json(out_a / DECODE_REENCODE_BLOCKED_PROOF_FILENAME)
    assert blocked_proof["kind"] == DECODE_REENCODE_INDEPENDENT_PROOF_KIND
    assert blocked_proof["independent_proof"] is True
    assert blocked_proof["score_claim"] is False
    assert blocked_proof["dispatch_attempted"] is False
    assert blocked_proof["proof_scope"] == "full_decode_reencode"
    assert blocked_proof["full_decode"]["passed"] is False
    assert blocked_proof["byte_exact_reencode"]["passed"] is False
    assert blocked_proof["sidecar_free"] is True
    assert summary["decode_reencode_blocked_proof"]["kind"] == DECODE_REENCODE_INDEPENDENT_PROOF_KIND


def test_build_categorical_candidate_payload_emits_hpm1_structural_inventory(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "categorical_payload.bin"
    hpac = _valid_hpm1_hpac_payload()
    payload = build_hpm1_mask_segment(
        (np.arange(16, dtype=np.uint32) % 5).tobytes(),
        hpac,
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
    runtime_proof = read_json(out_dir / RUNTIME_EXECUTION_PROOF_FILENAME)

    inventory_record = candidate["hpm1_structural_decode_inventory"]
    assert inventory_record["contract"] == HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT
    assert inventory_record["payload_member_sha256"] == candidate["decode_reencode_parity"]["payload_member_sha256"]
    assert inventory_record["structural_reencode_matches_source"] is True
    assert inventory_record["full_decode_proven"] is False
    assert inventory_record["byte_exact_semantic_reencode_proven"] is False
    assert inventory["structural_reencode"]["matches_source_segment"] is True
    assert inventory["full_decode"]["passed"] is False
    assert inventory["byte_exact_semantic_reencode"]["passed"] is False
    assert readiness["hpm1_structural_decode_inventory"]["accepted"] is True
    assert readiness["hpm1_structural_decode_inventory"]["structural_reencode_matches_source"] is True
    assert readiness["runtime_loader_parity"]["accepted"] is True
    assert readiness["runtime_loader_parity"]["runtime_execution_proof"]["accepted"] is True
    hpm1_runtime = readiness["runtime_loader_parity"]["runtime_execution_proof"][
        "hpm1_runtime_consumer_proof"
    ]
    assert hpm1_runtime["required"] is True
    assert hpm1_runtime["accepted"] is True
    assert hpm1_runtime["payload_codec"] == "HPM1"
    assert hpm1_runtime["hpm1_structural_reencode_passed"] is True
    assert hpm1_runtime["hpm1_hpac_model_load_passed"] is True
    assert hpm1_runtime["expected_fail_closed_exit"] is True
    assert readiness["dispatch_blockers"].count(
        "no_op_control_not_passed:label_permutation_fail_closed_control"
    ) == 0
    assert runtime_proof["runtime_report_summary"]["payload_codec"] == "HPM1"
    assert runtime_proof["runtime_report_summary"]["hpm1_structural_reencode_passed"] is True
    assert "hpm1_structural_inventory" in summary["paths"]
    assert "decode_reencode_blocked_proof" in summary["paths"]
    assert "decode_reencode_full_decode_not_proven" in readiness["dispatch_blockers"]
    assert "decode_reencode_byte_exact_reencode_not_proven" in readiness["dispatch_blockers"]
    assert "decode_reencode_independent_proof_artifact_missing" not in readiness["dispatch_blockers"]
    blocked_proof = read_json(out_dir / DECODE_REENCODE_BLOCKED_PROOF_FILENAME)
    assert blocked_proof["negative_proof"]["structural_inventory_attached"] is True
    assert blocked_proof["negative_proof"]["structural_reencode_matches_source"] is True
    assert "hpac_autoregressive_probability_rows" in blocked_proof["negative_proof"]["unsupported_wire_constructs"]
