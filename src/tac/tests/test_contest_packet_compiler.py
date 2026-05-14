# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.submission_archive import DETERMINISTIC_ZIP_DATE_TIME, DETERMINISTIC_ZIP_FILE_MODE
from tac.submission_packet_compiler import inspect_packet

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_member(
    zf: zipfile.ZipFile,
    name: str,
    payload: bytes,
    *,
    date_time: tuple[int, int, int, int, int, int] = DETERMINISTIC_ZIP_DATE_TIME,
    mode: int = DETERMINISTIC_ZIP_FILE_MODE,
    create_system: int = 3,
) -> None:
    info = zipfile.ZipInfo(name, date_time=date_time)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = mode << 16
    info.create_system = create_system
    zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def _write_packet(root: Path) -> Path:
    root.mkdir()
    with zipfile.ZipFile(root / "archive.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        _write_member(zf, "renderer.bin", b"renderer")
        _write_member(zf, "masks.mkv", b"masks")
    inflate = root / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\npython inflate.py\n", encoding="utf-8")
    inflate.chmod(0o755)
    (root / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    return root


def test_contest_packet_compiler_records_zip_and_runtime_manifest(
    tmp_path: Path,
) -> None:
    packet = _write_packet(tmp_path / "packet")

    manifest = inspect_packet(packet)

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["dispatchable"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["contest_compliance"]["blockers"] == []
    assert manifest["score_dispatch_gate"]["blockers"] == [
        "score_claim_forbidden_without_exact_cuda_auth_eval",
        "dispatch_readiness_forbidden_without_byte_closed_archive_and_exact_cuda_auth_eval",
        "level2_dispatch_claim_required_before_any_remote_exact_eval",
        "pre_submission_compliance_check_required_before_release_or_promotion",
    ]

    archive = manifest["archive"]
    assert archive["zip_metadata"]["member_order"] == ["renderer.bin", "masks.mkv"]
    assert archive["zip_metadata"]["all_timestamps_canonical"] is True
    assert archive["zip_metadata"]["all_permissions_canonical"] is True

    renderer = archive["members"][0]
    assert renderer["member_order_index"] == 0
    assert renderer["date_time"] == list(DETERMINISTIC_ZIP_DATE_TIME)
    assert renderer["unix_permissions"] == DETERMINISTIC_ZIP_FILE_MODE
    assert renderer["payload_bytes"] == len(b"renderer")
    assert renderer["payload_sha256"] == hashlib.sha256(b"renderer").hexdigest()
    assert renderer["compressed_payload_sha256"] == hashlib.sha256(b"renderer").hexdigest()
    assert renderer["local_header"]["data_offset"] == renderer["data_offset"]

    runtime = manifest["runtime_tree_manifest"]
    assert runtime["schema_version"] == "runtime_tree_manifest.v1"
    assert runtime["hooks"]["pre_submission_compliance_check_arg"] == (
        "--expected-runtime-tree-sha256"
    )
    assert runtime["tree_sha256"] == manifest["golden_vectors"]["runtime_tree_sha256"]
    inflate = next(row for row in runtime["files"] if row["path"] == "inflate.sh")
    assert inflate["mode"] & stat.S_IXUSR
    assert inflate["blockers"] == []


def test_contest_packet_compiler_blocks_nondeterministic_zip_metadata(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    with zipfile.ZipFile(packet / "archive.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        _write_member(
            zf,
            "x",
            b"payload",
            date_time=(2026, 5, 7, 12, 34, 56),
            mode=0o755,
            create_system=0,
        )
    inflate = packet / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    inflate.chmod(0o755)

    manifest = inspect_packet(packet)

    blockers = manifest["contest_compliance"]["blockers"]
    assert "archive:x:noncanonical_zip_timestamp:2026-05-07T12:34:56" in blockers
    assert "archive:x:noncanonical_zip_permissions:0o755" in blockers
    assert "archive:x:noncanonical_zip_create_system:0" in blockers
    assert manifest["archive"]["zip_metadata"]["all_timestamps_canonical"] is False
    assert manifest["archive"]["zip_metadata"]["all_permissions_canonical"] is False
    assert manifest["contest_compliance"]["contest_compliant_packet_shape"] is False


def test_contest_packet_compiler_cli_alias_writes_manifest(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path / "packet")
    json_out = tmp_path / "contest_packet_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "contest_packet_compiler.py"),
            str(packet),
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "score_claim=false dispatchable=false" in completed.stdout
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["schema_version"] == "submission_packet_compiler.v1"
    assert written["runtime_tree_manifest"]["tree_sha256"]
    assert written["score_dispatch_gate"]["dispatchable"] is False
