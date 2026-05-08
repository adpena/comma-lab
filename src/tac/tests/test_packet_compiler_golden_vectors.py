from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "build_packet_compiler_golden_vectors.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("build_packet_compiler_golden_vectors_test", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_vector(root: Path, label: str) -> dict[str, Any]:
    return json.loads((root / "vectors" / f"{label}.json").read_text(encoding="utf-8"))


def test_builtin_suite_emits_byte_identical_json_across_output_dirs(tmp_path: Path) -> None:
    script = _load_script()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_index = script.build_golden_vector_suite(output_dir=first_dir)
    second_index = script.build_golden_vector_suite(output_dir=second_dir)

    assert first_index == second_index
    assert (first_dir / script.INDEX_NAME).read_bytes() == (second_dir / script.INDEX_NAME).read_bytes()
    for row in first_index["vectors"]:
        rel = Path(row["vector_path"])
        assert (first_dir / rel).read_bytes() == (second_dir / rel).read_bytes()


def test_stored_single_member_vector_pins_headers_and_charged_payload_bytes(tmp_path: Path) -> None:
    script = _load_script()
    output_dir = tmp_path / "vectors"

    script.build_golden_vector_suite(output_dir=output_dir)
    vector = _load_vector(output_dir, "stored_single_member_ok")

    assert vector["schema_version"] == "packet_compiler_golden_vector.v1"
    assert vector["score_claim"] is False
    assert vector["input_archive"]["bytes"] == 113
    assert vector["input_archive"]["sha256"] == "7cae837c71aa1abbc55b52dcdb51487a847725bb97cb507d5761ac23c344bf86"
    assert bytes.fromhex(vector["input_archive"]["hex"]) == (
        output_dir / "fixtures" / "stored_single_member_ok" / "archive.zip"
    ).read_bytes()

    compiler = vector["compiler_manifest"]
    assert compiler["contest_compliance"]["blockers"] == []
    assert compiler["archive"]["zip_strict"] is True
    assert compiler["archive"]["members"][0]["payload_sha256"] == hashlib.sha256(b"payload-bytes").hexdigest()

    header = vector["zip_header_manifest"]
    assert header["charged_byte_accounting"] == {
        "contest_charged_archive_bytes": 113,
        "charged_member_payload_bytes": 13,
        "zip_container_overhead_bytes": 100,
        "accounted_zip_span_bytes": 113,
        "accounted_zip_span_matches_archive_bytes": True,
        "notes": [
            "Contest rate term charges archive.zip bytes.",
            "charged_member_payload_bytes is the sum of compressed ZIP member payload spans.",
        ],
    }
    member = header["members"][0]
    assert member["local_header"]["offset"] == 0
    assert member["local_header"]["bytes"] == 31
    assert member["compressed_payload"]["offset"] == 31
    assert member["compressed_payload"]["bytes"] == 13
    assert member["compressed_payload"]["charged_payload_bytes"] == 13
    assert member["central_directory_header"]["central_header_offset"] == 44
    assert member["central_directory_header"]["central_header_bytes"] == 47
    assert header["eocd"]["offset"] == 91
    assert header["eocd"]["bytes"] == 22


def test_duplicate_member_vector_records_fail_closed_case(tmp_path: Path) -> None:
    script = _load_script()
    output_dir = tmp_path / "vectors"

    script.build_golden_vector_suite(output_dir=output_dir)
    vector = _load_vector(output_dir, "duplicate_member_fail_closed")

    compiler = vector["compiler_manifest"]
    blockers = compiler["contest_compliance"]["blockers"]
    assert compiler["contest_compliance"]["contest_compliant_packet_shape"] is False
    assert compiler["archive"]["zip_strict"] is False
    assert compiler["archive"]["duplicate_member_names"] == ["x"]
    assert "archive:duplicate_archive_member:x" in blockers
    assert compiler["score_dispatch_gate"]["score_claim"] is False
    assert compiler["score_dispatch_gate"]["dispatchable"] is False

    assert vector["expectation"] == {
        "expected_contest_compliant_packet_shape": False,
        "expected_blockers": ["archive:duplicate_archive_member:x"],
        "observed_expected_blockers": ["archive:duplicate_archive_member:x"],
        "must_fail_closed": True,
        "status": "matched",
    }
    header = vector["zip_header_manifest"]
    assert header["duplicate_member_names"] == ["x"]
    assert len(header["members"]) == 2
    assert header["charged_byte_accounting"]["charged_member_payload_bytes"] == 2
    assert header["charged_byte_accounting"]["accounted_zip_span_matches_archive_bytes"] is True


def test_cli_writes_index_and_vectors(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-vectors"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "vectors=2" in completed.stdout
    index = json.loads((output_dir / "packet_compiler_golden_vectors_index.json").read_text(encoding="utf-8"))
    assert [row["label"] for row in index["vectors"]] == [
        "stored_single_member_ok",
        "duplicate_member_fail_closed",
    ]
