from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.analysis.lapose_foveation_atoms import build_foveation_transport_atom_manifest
from tac.analysis.lapose_foveation_payload import (
    PAYLOAD_MEMBER,
    build_lapose_foveation_tuple_payload_artifact,
    pack_lapose_foveation_tuple_payload,
)
from tac.lapose_foveation_payload_candidate import (
    ARCHIVE_MEMBER_MANIFEST_CONTRACT,
    CANDIDATE_MANIFEST_CONTRACT,
    MEMBER_ORDER,
    RUNTIME_LOADER_PARITY_CONTRACT,
    RUNTIME_PROOF_SKELETON_CONTRACT,
    audit_lapose_foveation_payload_candidate,
    build_lapose_foveation_payload_archive_candidate,
)
from tac.lapose_foveation_runtime_skeleton import (
    PROOF_MEMBER,
    RUNTIME_EFFECT_CONTROLS_CONTRACT,
    build_runtime_effect_control_report,
)
from tac.repo_io import read_json, sha256_bytes, sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]


def test_build_lapose_foveation_payload_archive_is_byte_closed_and_fail_closed(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "lapose_foveation_tuples.lfv1"
    source_readiness_path = tmp_path / "lfv1_readiness.json"
    source_readiness = build_lapose_foveation_tuple_payload_artifact(
        _manifest(),
        payload_path=payload_path,
        repo_root=tmp_path,
        max_atoms=2,
    )
    write_json(source_readiness_path, source_readiness)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    for out_dir in (out_a, out_b):
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "build_lapose_foveation_payload_archive.py"),
                "--out-dir",
                str(out_dir),
                "--lfv1-payload",
                str(payload_path),
                "--source-readiness-json",
                str(source_readiness_path),
            ],
            check=True,
            cwd=REPO,
            text=True,
        )

    archive_a = out_a / "archive.zip"
    archive_b = out_b / "archive.zip"
    assert sha256_file(archive_a) == sha256_file(archive_b)

    candidate = read_json(out_a / "candidate.json")
    readiness = audit_lapose_foveation_payload_candidate(
        candidate,
        repo_root=REPO,
        manifest_dir=out_a,
    )
    readiness_file = read_json(out_a / "readiness.json")
    summary = read_json(out_a / "summary.json")

    assert candidate["candidate_manifest_contract"] == CANDIDATE_MANIFEST_CONTRACT
    assert candidate["score_claim"] is False
    assert candidate["dispatch_attempted"] is False
    assert candidate["ready_for_exact_eval_dispatch"] is False
    assert candidate["lfv1_payload"]["member"] == PAYLOAD_MEMBER
    assert candidate["lfv1_payload"]["bytes"] == payload_path.stat().st_size
    assert candidate["lfv1_payload"]["sha256"] == sha256_file(payload_path)
    assert candidate["runtime_loader_parity"]["runtime_loader_parity_contract"] == (
        RUNTIME_LOADER_PARITY_CONTRACT
    )
    assert readiness["payload_member_proven"] is True
    assert readiness["byte_closed_local_archive"] is True
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["candidate_archive"]["untracked_members"] == []
    assert readiness["candidate_archive"]["member_order_matches_manifest"] is True
    assert readiness["candidate_archive"]["zip_determinism_contract"]["passed"] is True
    assert readiness["archive_member_manifest"]["member_order_matches_manifest"] is True
    assert readiness["runtime_loader_parity"]["sidecar_free"] is True
    assert readiness["runtime_loader_parity"]["accepted"] is False
    assert readiness["lfv1_payload_decode"]["accepted"] is True
    assert readiness["runtime_effect_controls"]["accepted"] is True
    assert readiness["runtime_effect_controls"]["contract"] == RUNTIME_EFFECT_CONTROLS_CONTRACT
    assert readiness["runtime_consumption_audit"]["structural_runtime_consumption"]["passed"] is True
    assert readiness["runtime_consumption_audit"]["scored_runtime_output_parity"]["passed"] is False
    assert readiness_file["tool_run_manifest"]["tool"] == (
        "tools/build_lapose_foveation_payload_archive.py"
    )
    assert readiness_file["ready_for_exact_eval_dispatch"] is False

    blockers = set(readiness["dispatch_blockers"])
    assert "runtime_loader_parity_not_passed" in blockers
    assert "exact_cuda_auth_eval_missing" in blockers
    assert not any(blocker.startswith("no_op_control_not_passed:") for blocker in blockers)
    assert not any(blocker.startswith("runtime_effect_controls_") for blocker in blockers)
    assert readiness["no_op_controls"]["failed_controls"] == []
    assert set(readiness["no_op_controls"]["passed_controls"]) == {
        "lfv1_identity_decode_control",
        "lfv1_tuple_mutation_runtime_output_control",
        "charged_member_presence_control",
        "runtime_consumes_foveation_tuple_control",
    }

    archive_member_manifest = read_json(out_a / "archive_member_manifest.json")
    assert archive_member_manifest["archive_member_manifest_contract"] == (
        ARCHIVE_MEMBER_MANIFEST_CONTRACT
    )
    assert archive_member_manifest["member_order"] == list(MEMBER_ORDER)
    assert archive_member_manifest["member_count"] == len(MEMBER_ORDER)
    payload_proof = next(
        proof
        for proof in readiness["archive_member_manifest"]["member_sha256_proofs"]
        if proof["name"] == PAYLOAD_MEMBER
    )
    assert payload_proof["manifest_bytes"] == payload_path.stat().st_size
    assert payload_proof["actual_sha256"] == sha256_file(payload_path)
    assert payload_proof["sha256_match"] is True

    with zipfile.ZipFile(archive_a) as archive:
        assert archive.namelist() == list(MEMBER_ORDER)
        proof_skeleton = json.loads(archive.read(PROOF_MEMBER).decode("utf-8"))
    assert proof_skeleton["runtime_consumer_proof_skeleton_contract"] == (
        RUNTIME_PROOF_SKELETON_CONTRACT
    )
    assert proof_skeleton["ready_for_exact_eval_dispatch"] is False
    assert proof_skeleton["proof_status"]["archive_contains_payload_and_runtime"] is True
    assert proof_skeleton["proof_status"]["runtime_output_parity"] is False
    assert proof_skeleton["proof_status"]["structural_runtime_consumption"] is True
    assert proof_skeleton["proof_status"]["scored_runtime_output_parity"] is False
    assert proof_skeleton["proof_status"]["noop_controls"] is True

    assert summary["kind"] == "lapose_foveation_byte_closed_local_candidate_build"
    assert summary["archive_sha256"] == sha256_file(archive_a)
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["tool_run_manifest"]["tool"] == (
        "tools/build_lapose_foveation_payload_archive.py"
    )
    assert "runtime_loader_parity_not_passed" in summary["readiness_blockers"]


def test_lapose_foveation_runtime_skeleton_verifies_members_then_exits_fail_closed(
    tmp_path: Path,
) -> None:
    payload = _lfv1_payload()
    result = build_lapose_foveation_payload_archive_candidate(
        out_dir=tmp_path / "candidate",
        lfv1_payload=payload,
        payload_source={
            "kind": "fixture_lfv1_payload",
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
        },
        repo_root=REPO,
    )
    extract_dir = tmp_path / "extract"
    with zipfile.ZipFile(tmp_path / "candidate" / "archive.zip") as archive:
        archive.extractall(extract_dir)

    proc = subprocess.run(
        [
            sys.executable,
            str(extract_dir / "runtime_consumer.py"),
            "--archive-root",
            str(extract_dir),
        ],
        check=False,
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    report = json.loads(proc.stdout)
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["lfv1_payload_decode"] == result["candidate_manifest"]["lfv1_payload"]["decoded"]
    assert report["runtime_effect_controls"]["passed"] is True
    assert report["runtime_effect_controls"]["runtime_effect_controls_contract"] == (
        RUNTIME_EFFECT_CONTROLS_CONTRACT
    )
    assert report["structural_runtime_consumption_proven"] is True
    assert report["runtime_output_parity_proven"] is False
    assert report["scored_runtime_output_parity_proven"] is False
    assert report["noop_controls_proven"] is True
    assert "exact_cuda_auth_eval_missing" in report["dispatch_blockers"]
    assert "lapose_foveation_scored_runtime_output_parity_missing" in report["dispatch_blockers"]


def test_lfv1_runtime_effect_controls_identity_and_structural_mutation() -> None:
    payload = _lfv1_payload()

    report = build_runtime_effect_control_report(payload)

    identity = report["lfv1_identity_decode_control"]
    mutation = report["lfv1_tuple_mutation_runtime_output_control"]
    runtime_consumes = report["runtime_consumes_foveation_tuple_control"]
    assert report["passed"] is True
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert identity["passed"] is True
    assert identity["byte_exact"] is True
    assert identity["source_payload_sha256"] == identity["reencoded_payload_sha256"]
    assert mutation["passed"] is True
    assert mutation["structural_output_changed"] is True
    assert (
        mutation["source_structural_output_sha256"]
        != mutation["mutated_structural_output_sha256"]
    )
    assert runtime_consumes["passed"] is True
    assert runtime_consumes["source_structural_output"]["route_count"] == 2
    assert report["structural_runtime_consumption"]["passed"] is True
    assert report["scored_runtime_output_parity"]["passed"] is False


def test_lapose_foveation_candidate_audit_fails_closed_on_payload_member_mismatch(
    tmp_path: Path,
) -> None:
    payload = _lfv1_payload()
    out = tmp_path / "candidate"
    result = build_lapose_foveation_payload_archive_candidate(
        out_dir=out,
        lfv1_payload=payload,
        payload_source={
            "kind": "fixture_lfv1_payload",
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
        },
        repo_root=REPO,
    )
    archive_path = out / "archive.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    mutated = bytearray(members[PAYLOAD_MEMBER])
    mutated[-1] ^= 1
    members[PAYLOAD_MEMBER] = bytes(mutated)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in MEMBER_ORDER:
            archive.writestr(name, members[name], compress_type=zipfile.ZIP_STORED)

    readiness = audit_lapose_foveation_payload_candidate(
        result["candidate_manifest"],
        repo_root=REPO,
        manifest_dir=out,
    )

    blockers = set(readiness["dispatch_blockers"])
    assert readiness["payload_member_proven"] is False
    assert readiness["byte_closed_local_archive"] is False
    assert "candidate_archive_sha256_mismatch" in blockers
    assert f"archive_member_manifest_member_sha256_mismatch:{PAYLOAD_MEMBER}" in blockers
    assert "lfv1_payload_member_sha256_mismatch" in blockers
    assert "lfv1_payload_decoded_preview_mismatch" in blockers


def _lfv1_payload() -> bytes:
    payload, _pack = pack_lapose_foveation_tuple_payload(_manifest(), max_atoms=2)
    return payload


def _manifest() -> dict:
    return build_foveation_transport_atom_manifest(
        _records(),
        base_pose_dist=0.02,
        source="fixture",
        frame_width=320,
        frame_height=200,
        foveal_center=(160.0, 90.0),
        center_gain=(12.0, 8.0),
    )


def _records() -> list[dict]:
    return [
        {
            "pair_index": 10,
            "latent_action": [-0.3, 0.0, 1.0, 0.1, -0.2, 0.3, 0.02, -0.03],
            "expected_seg_dist_delta": -0.00001,
            "expected_pose_dist_delta": -0.00002,
            "confidence": 0.6,
            "class_support": [1],
            "pair_support": [10],
            "geometry_priors": ["lane_boundary"],
        },
        {
            "pair_index": 75,
            "hard_pair_rank": 0,
            "latent_action": [0.0, 1.0, 0.0, 1.2, 0.8, 1.5, 0.4, 0.2],
            "expected_seg_dist_delta": -0.0002,
            "expected_pose_dist_delta": -0.00005,
            "confidence": 0.8,
            "hard_pair_score": 4.2,
            "pair_support": [75],
            "hard_pair_support": [75],
            "class_support": [2, 3],
            "geometry_priors": ["lane_boundary"],
            "openpilot_priors": ["ego_motion"],
        },
        {
            "pair_index": 127,
            "latent_action": [0.4, 0.5, -0.5, -0.3, 0.9, 0.5, -0.2, 0.4],
            "expected_seg_dist_delta": -0.00005,
            "expected_pose_dist_delta": -0.00001,
            "confidence": 0.7,
            "class_support": [3],
            "openpilot_priors": ["yaw_rate"],
            "evidence_grade": "planning_lapose_foveation_transport",
        },
    ]
