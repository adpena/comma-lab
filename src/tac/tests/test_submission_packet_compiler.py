from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from tac.submission_packet_compiler import (
    MANIFEST_NAME,
    PacketCompilerError,
    compile_packet,
    inspect_packet,
)


def _write_archive(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    return path


def _write_packet(root: Path) -> Path:
    root.mkdir()
    _write_archive(root / "archive.zip", [("x", b"payload-bytes")])
    inflate = root / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\npython inflate.py\n", encoding="utf-8")
    inflate.chmod(0o755)
    (root / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    return root


def test_inspect_packet_emits_deterministic_golden_vectors(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path / "packet")

    first = inspect_packet(packet, target_profile="contest_one_video_replay")
    second = inspect_packet(packet, target_profile="contest_one_video_replay")

    assert first == second
    assert first["schema_version"] == "submission_packet_compiler.v1"
    assert first["target_profile"] == "contest_one_video_replay"
    assert first["score_claim"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["contest_compliance"]["blockers"] == []
    member = first["archive"]["members"][0]
    assert member["name"] == "x"
    assert member["local_header_name"] == "x"
    assert member["local_central_name_match"] is True
    assert member["payload_sha256"]


def test_identity_mode_copies_packet_bytes_and_writes_manifest(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path / "packet")
    output = tmp_path / "identity"

    manifest = compile_packet(packet, mode="identity", output_dir=output)

    assert (output / "archive.zip").read_bytes() == (packet / "archive.zip").read_bytes()
    assert (output / "inflate.sh").read_bytes() == (packet / "inflate.sh").read_bytes()
    assert os.access(output / "inflate.sh", os.X_OK)
    assert manifest["identity_rewrite"]["byte_identical_to_input_tree"] is True
    written = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert written["identity_rewrite"]["copied_file_count"] == 3


def test_duplicate_zip_members_fail_closed_in_manifest(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    with zipfile.ZipFile(packet / "archive.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"a")
        zf.writestr("x", b"b")
    (packet / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    manifest = inspect_packet(packet)

    blockers = manifest["contest_compliance"]["blockers"]
    assert "archive:duplicate_archive_member:x" in blockers
    assert manifest["archive"]["duplicate_member_names"] == ["x"]


def test_archive_only_packet_is_inspectable_but_not_contest_packet_shape(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path / "archive.zip", [("x", b"payload")])

    manifest = inspect_packet(archive)

    assert manifest["archive"]["zip_strict"] is True
    assert "archive_only_packet_runtime_missing" in manifest["contest_compliance"]["blockers"]


@pytest.mark.parametrize("mode", ["canonicalize", "optimize"])
def test_unimplemented_rewrite_modes_fail_closed(tmp_path: Path, mode: str) -> None:
    packet = _write_packet(tmp_path / "packet")

    with pytest.raises(PacketCompilerError, match=f"{mode} mode is not implemented"):
        compile_packet(packet, mode=mode, output_dir=tmp_path / "out")


def test_unknown_target_profile_rejected(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path / "packet")

    with pytest.raises(PacketCompilerError, match="unknown target_profile"):
        inspect_packet(packet, target_profile="ambiguous")
