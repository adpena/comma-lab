"""Fail-closed tests for the robust_current JCSP runtime bridge."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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
    manifest_path = tmp_path / "manifest.json"

    first = bridge.probe_jcsp_runtime_bridge(
        archive_dir,
        manifest_json=manifest_path,
    )
    second = bridge.probe_jcsp_runtime_bridge(
        archive_dir,
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
    assert JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER in first["dispatch_blockers"]
    assert "jcsp_stream_decode_emit_frames_missing" in first["dispatch_blockers"]
    assert len(first["manifest_sha256"]) == 64


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
    manifest_path = tmp_path / "probe.json"

    result = subprocess.run(
        [
            sys.executable,
            str(BRIDGE_PATH),
            str(archive_dir),
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


def test_inflate_sh_probes_jcsp_before_branch_dispatch() -> None:
    text = INFLATE_SH_PATH.read_text(encoding="utf-8")
    hook = text.index("jcsp_runtime_bridge.py")
    branch_dispatch = text.index("while IFS= read -r rel")

    assert hook < branch_dispatch
    assert "--manifest-json \"$JCSP_RUNTIME_PROBE_MANIFEST\"" in text
