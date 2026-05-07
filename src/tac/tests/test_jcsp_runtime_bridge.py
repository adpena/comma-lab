"""Fail-closed tests for the robust_current JCSP runtime bridge."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.joint_codec_stack_orchestrator import (
    JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
    JCSP_LOCAL_SKELETON_SCHEMA,
    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
    KIND_ARITHMETIC_STATIC,
    StreamSource,
    pack_jcsp_local_skeleton_container,
    run_sequential_codec_stack,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BRIDGE_PATH = REPO_ROOT / "submissions" / "robust_current" / "jcsp_runtime_bridge.py"
INFLATE_SH_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate.sh"


def _load_bridge_module():
    spec = importlib.util.spec_from_file_location("jcsp_runtime_bridge_test", BRIDGE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["jcsp_runtime_bridge_test"] = module
    spec.loader.exec_module(module)
    return module


def _real_jcsp_bytes() -> bytes:
    stream = StreamSource(
        name="tiny",
        qints=np.array([0, 1, -1, 2], dtype=np.int8),
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    return run_sequential_codec_stack(streams=[stream]).container_bytes


def _jcsk_preview_bytes() -> bytes:
    return pack_jcsp_local_skeleton_container(
        manifest={
            "schema": JCSP_LOCAL_SKELETON_SCHEMA,
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_runtime_loader": False,
            "ready_for_submission_runtime_consumption": False,
            "ready_for_exact_eval_dispatch": False,
            "stream_count": 0,
            "streams": [],
        }
    )


def test_runtime_bridge_detects_real_jcsp_but_refuses_consumption(
    tmp_path: Path,
) -> None:
    bridge = _load_bridge_module()
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "jcsp.bin").write_bytes(_real_jcsp_bytes())
    inflated_dir = tmp_path / "inflated"
    inflated_dir.mkdir()
    names_file = tmp_path / "names.txt"
    names_file.write_text("2018-07-27--06-03-57/10/video.hevc\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    first = bridge.probe_jcsp_runtime_bridge(
        archive_dir,
        inflated_dir=inflated_dir,
        video_names_file=names_file,
        manifest_json=manifest_path,
    )
    second = bridge.probe_jcsp_runtime_bridge(
        archive_dir,
        inflated_dir=inflated_dir,
        video_names_file=names_file,
        manifest_json=manifest_path,
    )

    assert first == second
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == first
    assert first["schema"] == bridge.JCSP_RUNTIME_BRIDGE_PROBE_SCHEMA
    assert first["score_claim"] is False
    assert first["detected_real_jcsp_member"] is True
    assert first["refused_preview_member"] is False
    assert first["ready_for_runtime_loader"] is True
    assert first["consumes_required_member"] is False
    assert first["ready_for_submission_runtime_consumption"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["container_magic"] == "JCSP"
    assert first["stream_count"] == 1
    assert first["streams"][0]["payload_magic"] == "AQv1"
    assert first["runtime_action"] == "refuse_until_jcsp_stream_consumer_implemented"
    output_contract = first["contest_output_contract"]
    assert output_contract["schema"] == bridge.JCSP_RUNTIME_OUTPUT_CONTRACT_SCHEMA
    assert output_contract["expected_raw_outputs"] == [
        "2018-07-27--06-03-57/10/video.raw"
    ]
    parity_contract = output_contract["raw_output_parity_contract"]
    assert parity_contract["schema"] == (
        bridge.JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA
    )
    assert parity_contract["required_proof_schema"] == (
        bridge.JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA
    )
    assert parity_contract["required_candidate_output_source"] == (
        "jcsp_runtime_bridge_emitted_rawvideo"
    )
    assert parity_contract["expected_raw_output_count"] == 1
    assert parity_contract["preexisting_raw_outputs_are_not_parity_proof"] is True
    assert parity_contract["ready_for_output_parity"] is False
    assert output_contract["bridge_emits_contest_raw_outputs"] is False
    assert output_contract["raw_output_emission_attempted"] is False
    assert output_contract["output_parity_checked"] is False
    assert output_contract["ready_for_submission_runtime_consumption"] is False
    assert output_contract["existing_raw_output_count"] == 0
    assert bridge.JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER in (
        output_contract["dispatch_blockers"]
    )
    assert JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER in first["dispatch_blockers"]
    assert bridge.JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER in first["dispatch_blockers"]
    assert "jcsp_stream_decode_emit_frames_missing" in first["dispatch_blockers"]
    assert len(first["manifest_sha256"]) == 64


def test_runtime_bridge_marks_preexisting_raw_outputs_unproven(
    tmp_path: Path,
) -> None:
    bridge = _load_bridge_module()
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "jcsp.bin").write_bytes(_real_jcsp_bytes())
    inflated_dir = tmp_path / "inflated"
    raw_path = inflated_dir / "route" / "video.raw"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"stale-output")
    names_file = tmp_path / "names.txt"
    names_file.write_text("route/video.hevc\n", encoding="utf-8")

    manifest = bridge.probe_jcsp_runtime_bridge(
        archive_dir,
        inflated_dir=inflated_dir,
        video_names_file=names_file,
    )

    output_contract = manifest["contest_output_contract"]
    assert output_contract["existing_raw_output_count"] == 1
    [row] = output_contract["existing_raw_outputs_at_probe"]
    assert row["path"] == "route/video.raw"
    assert row["exists"] is True
    assert row["is_file"] is True
    assert row["bytes"] == len(b"stale-output")
    assert row["sha256"] is None
    assert row["sha256_status"] == "not_hashed_pre_dispatch_probe"
    assert row["parity_proof_source"] == "preexisting_raw_output_unproven"
    assert "jcsp_existing_raw_outputs_unproven" in (
        output_contract["dispatch_blockers"]
    )
    parity_contract = output_contract["raw_output_parity_contract"]
    assert parity_contract["existing_raw_output_count_at_probe"] == 1
    assert parity_contract["preexisting_raw_outputs_are_not_parity_proof"] is True
    assert parity_contract["ready_for_submission_runtime_consumption"] is False
    assert bridge.JCSP_RUNTIME_OUTPUT_PARITY_BLOCKER in (
        output_contract["dispatch_blockers"]
    )
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_runtime_raw_output_parity_proof_requires_bridge_emission(
    tmp_path: Path,
) -> None:
    bridge = _load_bridge_module()
    candidate_dir = tmp_path / "candidate"
    reference_dir = tmp_path / "reference"
    candidate_path = candidate_dir / "route" / "video.raw"
    reference_path = reference_dir / "route" / "video.raw"
    candidate_path.parent.mkdir(parents=True)
    reference_path.parent.mkdir(parents=True)
    payload = b"rgb-raw-bytes"
    candidate_path.write_bytes(payload)
    reference_path.write_bytes(payload)
    manifest_path = tmp_path / "raw_output_parity.json"

    proof = bridge.prove_jcsp_runtime_raw_output_parity(
        ["route/video.raw"],
        candidate_raw_dir=candidate_dir,
        reference_raw_dir=reference_dir,
        candidate_outputs_emitted_by_bridge=False,
        manifest_json=manifest_path,
    )

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == proof
    assert proof["schema"] == bridge.JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA
    assert proof["score_claim"] is False
    assert proof["dispatch_attempted"] is False
    assert proof["byte_exact_raw_output_parity"] is True
    assert proof["ready_for_output_parity"] is False
    assert proof["ready_for_submission_runtime_consumption"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
    assert "jcsp_candidate_outputs_not_emitted_by_bridge" in (
        proof["dispatch_blockers"]
    )
    [row] = proof["outputs"]
    assert row["candidate_bytes"] == len(payload)
    assert row["candidate_sha256"] == hashlib.sha256(payload).hexdigest()
    assert row["reference_sha256"] == row["candidate_sha256"]
    assert row["byte_exact_match"] is True

    emitted_proof = bridge.prove_jcsp_runtime_raw_output_parity(
        ["route/video.raw"],
        candidate_raw_dir=candidate_dir,
        reference_raw_dir=reference_dir,
        candidate_outputs_emitted_by_bridge=True,
    )

    assert emitted_proof["ready_for_output_parity"] is True
    assert emitted_proof["ready_for_submission_runtime_consumption"] is True
    assert emitted_proof["ready_for_exact_eval_dispatch"] is False
    assert emitted_proof["dispatch_blockers"] == []


def test_runtime_raw_output_parity_proof_rejects_unsafe_paths(
    tmp_path: Path,
) -> None:
    bridge = _load_bridge_module()
    with pytest.raises(ValueError, match="unsafe raw output path"):
        bridge.prove_jcsp_runtime_raw_output_parity(
            ["../video.raw"],
            candidate_raw_dir=tmp_path,
            reference_raw_dir=tmp_path,
            candidate_outputs_emitted_by_bridge=True,
        )


def test_runtime_bridge_refuses_jcsk_preview_member(tmp_path: Path) -> None:
    bridge = _load_bridge_module()
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "jcsp.bin").write_bytes(_jcsk_preview_bytes())

    manifest = bridge.probe_jcsp_runtime_bridge(archive_dir)

    assert manifest["member_present"] is True
    assert manifest["detected_real_jcsp_member"] is False
    assert manifest["refused_preview_member"] is True
    assert manifest["ready_for_runtime_loader"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["container_magic"] == "JCSK"
    assert manifest["runtime_action"] == "refuse_jcsk_preview_member"
    assert JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER in manifest["dispatch_blockers"]
    assert JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER in manifest["dispatch_blockers"]


def test_runtime_bridge_cli_fails_closed_when_jcsp_member_present(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "jcsp.bin").write_bytes(_real_jcsp_bytes())
    inflated_dir = tmp_path / "inflated"
    inflated_dir.mkdir()
    names_file = tmp_path / "names.txt"
    names_file.write_text("route/video.hevc\n", encoding="utf-8")
    manifest_path = tmp_path / "probe.json"

    result = subprocess.run(
        [
            sys.executable,
            str(BRIDGE_PATH),
            str(archive_dir),
            "--inflated-dir",
            str(inflated_dir),
            "--video-names-file",
            str(names_file),
            "--manifest-json",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 44
    assert "FATAL" in result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["detected_real_jcsp_member"] is True
    assert manifest["consumes_required_member"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["contest_output_contract"]["expected_raw_outputs"] == [
        "route/video.raw"
    ]
    assert manifest["contest_output_contract"]["bridge_emits_contest_raw_outputs"] is False
    assert not (inflated_dir / "route" / "video.raw").exists()


def test_inflate_sh_probes_jcsp_before_branch_dispatch() -> None:
    text = INFLATE_SH_PATH.read_text(encoding="utf-8")
    hook = text.index("jcsp_runtime_bridge.py")
    branch_dispatch = text.index("while IFS= read -r rel")

    assert hook < branch_dispatch
    assert "--inflated-dir \"$INFLATED_DIR\"" in text
    assert "--video-names-file \"$VIDEO_NAMES_FILE\"" in text
    assert "--manifest-json \"$JCSP_RUNTIME_PROBE_MANIFEST\"" in text
