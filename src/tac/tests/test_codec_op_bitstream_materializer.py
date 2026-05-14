# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.codec_op_bitstream_materializer import (
    BITSTREAM_MAGIC,
    BITSTREAM_MAGIC_TEXT,
    CodecOpBitstreamMaterializerError,
    materialize_codec_op_bitstream,
    parse_materialized_codec_op_bitstream,
)
from tac.repo_io import sha256_bytes, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]


def _decode_validation() -> dict[str, object]:
    return {
        "expected_tensor_count": 2,
        "matched_tensor_count": 2,
        "missing_tensor_keys": [],
        "non_tensor_decoded_keys": [],
        "shape_mismatch_tensor_keys": [],
        "dtype_mismatch_tensor_keys": [],
        "decode_coverage_status": "full",
    }


def _source(payload: bytes, **overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "candidate_id": "codec_op_fixture_q5",
        "op_module": "tac.fixture",
        "op_class": "FixtureCodecOp",
        "op_name": "fixture_codec_op",
        "stream_name": "codec_op:fixture_codec_op",
        "op_params": {"quality": 5, "window": 18},
        "bytes_out": len(payload),
        "blob_sha256": sha256_bytes(payload),
        "blob_base64": base64.b64encode(payload).decode("ascii"),
        "decode_coverage_status": "full",
        "decode_validation": _decode_validation(),
        "evidence_semantics": "codec_op_archive_member_byte_custody",
        "evidence_grade": "byte_custody_only",
        "exact_cuda_auth_eval": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_sha256": sha256_bytes(b"source-archive"),
        "archive_bytes": 1234,
    }
    source.update(overrides)
    return source


def test_materializes_deterministic_codec_op_blob_and_golden_vector(tmp_path: Path) -> None:
    payload = b"codec-op-payload-v1"
    out = tmp_path / "codec_op.cobm"
    manifest_path = tmp_path / "manifest.json"

    manifest = materialize_codec_op_bitstream(
        _source(payload),
        output_blob=out,
        manifest_output=manifest_path,
    )

    assert out.read_bytes().startswith(BITSTREAM_MAGIC)
    parsed = parse_materialized_codec_op_bitstream(out.read_bytes())
    assert parsed["payload"] == payload
    assert parsed["header"]["codec_magic"] == BITSTREAM_MAGIC_TEXT
    assert parsed["header"]["deterministic_params"]["op_params"] == {
        "quality": 5,
        "window": 18,
    }
    assert manifest["codec_magic"] == BITSTREAM_MAGIC_TEXT
    assert manifest["score_claim"] is False
    assert manifest["dispatchable"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["ready_for_archive_substitution"] is True
    assert manifest["archive_substitution_blockers"] == []
    assert manifest["materialized_charged_byte_artifact"] is True
    assert manifest["charged_byte_blob"]["payload_sha256"] == sha256_bytes(payload)
    assert manifest["charged_byte_blob"]["sha256"] == sha256_file(out)
    assert manifest["roundtrip"]["status"] == "passed"
    assert manifest["archive_identity"]["present"] is True
    assert "missing_exact_cuda_auth_eval" in manifest["blockers"]
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["golden_vector"] == manifest["golden_vector"]


def test_materialized_blob_is_stable_across_output_paths(tmp_path: Path) -> None:
    payload = b"stable-materialized-bytes"
    out_a = tmp_path / "a.cobm"
    out_b = tmp_path / "b.cobm"

    manifest_a = materialize_codec_op_bitstream(_source(payload), output_blob=out_a)
    manifest_b = materialize_codec_op_bitstream(_source(payload), output_blob=out_b)

    assert out_a.read_bytes() == out_b.read_bytes()
    assert manifest_a["golden_vector"]["blob_sha256"] == manifest_b["golden_vector"]["blob_sha256"]


def test_missing_payload_bytes_fails_closed_before_writing(tmp_path: Path) -> None:
    source = _source(b"payload")
    source.pop("blob_base64")

    with pytest.raises(CodecOpBitstreamMaterializerError, match="payload bytes are missing"):
        materialize_codec_op_bitstream(source, output_blob=tmp_path / "out.cobm")

    assert not (tmp_path / "out.cobm").exists()


def test_decode_roundtrip_failure_rejected_before_writing(tmp_path: Path) -> None:
    payload = b"payload"
    bad_validation = _decode_validation()
    bad_validation["matched_tensor_count"] = 1
    bad_validation["missing_tensor_keys"] = ["b.bias"]
    source = _source(
        payload,
        decode_coverage_status="failed",
        decode_validation=bad_validation,
    )

    with pytest.raises(CodecOpBitstreamMaterializerError, match="did not reconstruct"):
        materialize_codec_op_bitstream(source, output_blob=tmp_path / "out.cobm")

    assert not (tmp_path / "out.cobm").exists()


def test_cpu_only_and_missing_archive_identity_are_fail_closed_blockers(
    tmp_path: Path,
) -> None:
    payload = b"cpu-planning-payload"
    source = _source(
        payload,
        evidence_semantics="cpu_codec_op_admm_bridge_planning_only",
        archive_sha256=None,
        archive_bytes=None,
    )

    manifest = materialize_codec_op_bitstream(source, output_blob=tmp_path / "out.cobm")

    assert manifest["source_evidence"]["cpu_only"] is True
    assert manifest["archive_identity"]["present"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["ready_for_archive_substitution"] is False
    assert "cpu_only_evidence_not_score_or_dispatch_evidence" in manifest["blockers"]
    assert "archive_identity_absent" in manifest["blockers"]


def test_payload_custody_mismatch_rejected(tmp_path: Path) -> None:
    source = _source(b"payload", bytes_out=999)

    with pytest.raises(CodecOpBitstreamMaterializerError, match="byte count mismatch"):
        materialize_codec_op_bitstream(source, output_blob=tmp_path / "out.cobm")


def test_cli_materializes_payload_path_from_manifest_custody(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"path-backed-codec-op-payload")
    source = _source(payload_path.read_bytes())
    source.pop("blob_base64")
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    out = tmp_path / "cli.cobm"
    manifest_path = tmp_path / "cli_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/materialize_codec_op_bitstream.py",
            "--input-json",
            str(input_json),
            "--payload",
            str(payload_path),
            "--output-blob",
            str(out),
            "--manifest-output",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parsed = parse_materialized_codec_op_bitstream(out.read_bytes())
    assert parsed["payload"] == payload_path.read_bytes()
    assert manifest["source_payload"]["source"]["kind"] == "explicit_payload_path"
    assert manifest["charged_byte_blob"]["payload_sha256"] == sha256_file(payload_path)


def test_materializer_accepts_standardized_materialized_payload_aliases(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "materialized.section"
    payload_path.write_bytes(b"standardized-materialized-payload")
    source = _source(payload_path.read_bytes())
    source.pop("blob_base64")
    source.pop("bytes_out")
    source.pop("blob_sha256")
    source.update(
        {
            "materialized_payload_path": payload_path.name,
            "materialized_payload_bytes": payload_path.stat().st_size,
            "materialized_payload_sha256": sha256_file(payload_path),
        }
    )
    source_json = tmp_path / "source.json"
    source_json.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")

    manifest = materialize_codec_op_bitstream(
        source,
        source_manifest_path=source_json,
        output_blob=tmp_path / "materialized.cobm",
    )

    assert manifest["source_payload"]["source"]["kind"] == (
        "manifest_payload_path:materialized_payload_path"
    )
    assert manifest["charged_byte_blob"]["payload_sha256"] == sha256_file(payload_path)
