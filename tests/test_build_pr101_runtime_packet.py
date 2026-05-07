from __future__ import annotations

import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tools.build_pr101_runtime_packet import (
    _canonical_json_sha256,
    build_packet,
    run_local_inflate_parity,
)


def _write_file(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)


def _write_zip(path: Path, member: str, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, data)


def _runtime_fixture(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    _write_file(runtime / "inflate.sh", "#!/usr/bin/env bash\nset -euo pipefail\n", 0o755)
    _write_file(runtime / "inflate.py", "print('inflate')\n")
    _write_file(runtime / "src/model.py", "MODEL = 1\n", 0o755)
    _write_file(runtime / "README.md", "runtime notes\n")
    _write_file(runtime / "__pycache__/inflate.cpython-312.pyc", "cache")
    _write_file(runtime / "src/__pycache__/model.cpython-312.pyc", "cache")
    _write_file(runtime / ".DS_Store", "finder")
    return runtime


def test_build_packet_copies_runtime_and_candidate_archive_with_custody(tmp_path: Path) -> None:
    runtime = _runtime_fixture(tmp_path)
    candidate_archive = tmp_path / "candidate/archive.zip"
    source_archive = tmp_path / "source/archive.zip"
    _write_zip(candidate_archive, "x", b"candidate-bytes")
    _write_zip(source_archive, "x", b"source-bytes")

    packet_dir = tmp_path / "packet"
    payload = build_packet(
        source_runtime_dir=runtime,
        candidate_archive=candidate_archive,
        packet_dir=packet_dir,
        source_archive=source_archive,
        candidate_id="unit-pr101",
        recorded_at_utc=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
    )

    manifest = json.loads((packet_dir / "runtime_custody_manifest.json").read_text())
    assert payload == manifest
    assert (packet_dir / "archive.zip").read_bytes() == candidate_archive.read_bytes()
    assert (packet_dir / "inflate.sh").is_file()
    assert (packet_dir / "src/model.py").is_file()
    assert not (packet_dir / "__pycache__").exists()
    assert not (packet_dir / "src/__pycache__").exists()
    assert not (packet_dir / ".DS_Store").exists()

    runtime_files = manifest["runtime_custody"]["runtime_files"]
    assert [row["relpath"] for row in runtime_files] == [
        "README.md",
        "inflate.py",
        "inflate.sh",
        "src/model.py",
    ]
    assert {row["relpath"]: row["mode"] for row in runtime_files} == {
        "README.md": "0644",
        "inflate.py": "0644",
        "inflate.sh": "0755",
        "src/model.py": "0755",
    }
    assert manifest["packet_archive"]["members"][0]["name"] == "x"
    assert manifest["packet_archive"]["members"][0]["sha256"] != manifest["source_archive"]["members"][0]["sha256"]
    assert manifest["runtime_checks"]["inflate_sh_bash_n"]["passed"] is True

    without_self = {key: value for key, value in manifest.items() if key != "manifest_sha256_excluding_self"}
    assert manifest["manifest_sha256_excluding_self"] == _canonical_json_sha256(without_self)


def test_build_packet_runtime_tree_hash_is_stable_across_output_dirs(tmp_path: Path) -> None:
    runtime = _runtime_fixture(tmp_path)
    candidate_archive = tmp_path / "candidate/archive.zip"
    _write_zip(candidate_archive, "x", b"candidate-bytes")

    first = build_packet(
        source_runtime_dir=runtime,
        candidate_archive=candidate_archive,
        packet_dir=tmp_path / "packet-a",
        source_archive=None,
        candidate_id="unit-pr101",
        recorded_at_utc=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
    )
    second = build_packet(
        source_runtime_dir=runtime,
        candidate_archive=candidate_archive,
        packet_dir=tmp_path / "packet-b",
        source_archive=None,
        candidate_id="unit-pr101",
        recorded_at_utc=datetime(2026, 5, 7, 13, 0, tzinfo=UTC),
    )

    assert first["runtime_custody"]["runtime_tree_sha256"] == second["runtime_custody"]["runtime_tree_sha256"]
    assert first["runtime_custody"]["runtime_files"] == second["runtime_custody"]["runtime_files"]


def test_build_packet_refuses_non_empty_output_without_force(tmp_path: Path) -> None:
    runtime = _runtime_fixture(tmp_path)
    candidate_archive = tmp_path / "candidate/archive.zip"
    _write_zip(candidate_archive, "x", b"candidate-bytes")
    packet_dir = tmp_path / "packet"
    _write_file(packet_dir / "partner-note.txt", "do not replace")

    with pytest.raises(ValueError, match="not empty"):
        build_packet(
            source_runtime_dir=runtime,
            candidate_archive=candidate_archive,
            packet_dir=packet_dir,
            source_archive=None,
            candidate_id="unit-pr101",
            recorded_at_utc=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
        )

    assert (packet_dir / "partner-note.txt").read_text(encoding="utf-8") == "do not replace"


def test_local_inflate_parity_precreates_nested_output_parent_dirs(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    _write_file(
        runtime / "inflate.sh",
        """#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  cat "${DATA_DIR}/x" > "${OUTPUT_DIR}/${BASE}.raw"
done < "$FILE_LIST"
""",
        0o755,
    )
    _write_file(runtime / "inflate.py", "unused\n")

    source_archive = tmp_path / "source/archive.zip"
    candidate_archive = tmp_path / "candidate/archive.zip"
    _write_zip(source_archive, "x", b"same-bytes")
    _write_zip(candidate_archive, "x", b"same-bytes")
    packet_dir = tmp_path / "packet"
    build_packet(
        source_runtime_dir=runtime,
        candidate_archive=candidate_archive,
        packet_dir=packet_dir,
        source_archive=source_archive,
        candidate_id="unit-pr101",
        recorded_at_utc=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
    )
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("nested/segment/video.hevc\n", encoding="utf-8")

    parity = run_local_inflate_parity(
        source_runtime_dir=runtime,
        source_archive=source_archive,
        packet_dir=packet_dir,
        file_list=file_list,
        parity_dir=tmp_path / "parity",
        timeout_seconds=30,
    )

    assert parity["passed"] is True
    assert parity["outputs_retained"] is False
    assert parity["parity_dir_removed_after_hashing"] is True
    assert parity["expected_output_relpaths"] == ["nested/segment/video.raw"]
    assert parity["comparisons"] == [
        {
            "candidate_bytes": 10,
            "candidate_sha256": parity["candidate_outputs"][0]["sha256"],
            "relpath": "nested/segment/video.raw",
            "sha256_equal": True,
            "source_bytes": 10,
            "source_sha256": parity["source_outputs"][0]["sha256"],
        }
    ]
