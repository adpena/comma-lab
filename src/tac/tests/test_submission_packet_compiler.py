from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tac.submission_packet_compiler import (
    MANIFEST_NAME,
    TARGET_PROFILE_POLICIES,
    TARGET_PROFILES,
    PacketCompilerError,
    compile_packet,
    inspect_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _zipwire_archive_from_python_archive(archive: dict[str, object]) -> dict[str, object]:
    members = []
    for member in archive["members"]:  # type: ignore[index]
        assert isinstance(member, dict)
        members.append(
            {
                "name": member["name"],
                "local_header_name": member["local_header_name"],
                "local_central_name_match": member["local_central_name_match"],
                "member_order_index": member["member_order_index"],
                "header_offset": member["header_offset"],
                "data_offset": member["data_offset"],
                "compress_type": member["compress_type"],
                "compressed_bytes": member["compressed_bytes"],
                "uncompressed_bytes": member["uncompressed_bytes"],
                "crc32": member["crc32"],
                "payload_sha256": member["payload_sha256"],
                "compressed_payload_sha256": member["compressed_payload_sha256"],
                "date_time": member["date_time"],
                "flag_bits": member["flag_bits"],
                "external_attr": member["external_attr"],
                "create_system": member["create_system"],
                "unix_permissions": member["unix_permissions"],
                "blockers": member["blockers"],
                "local_header": {
                    "flag_bits": member["flag_bits"],
                    "compress_type": member["compress_type"],
                    "crc32": member["crc32"],
                    "compressed_bytes": member["compressed_bytes"],
                    "uncompressed_bytes": member["uncompressed_bytes"],
                },
            }
        )
    return {
        "path": archive["path"],
        "bytes": archive["bytes"],
        "sha256": archive["sha256"],
        "member_count": archive["member_count"],
        "duplicate_member_names": archive["duplicate_member_names"],
        "members": members,
        "blockers": archive["blockers"],
        "zip_strict": archive["zip_strict"],
    }


def _write_fake_zipwire(
    path: Path,
    payload: dict[str, object],
    *,
    return_code: int = 0,
) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if len(sys.argv) != 2:\n"
        "    print('usage: fake_zipwire <archive.zip>', file=sys.stderr)\n"
        "    raise SystemExit(2)\n"
        f"sys.stdout.write({json.dumps(payload, sort_keys=True)!r} + '\\n')\n"
        f"raise SystemExit({return_code})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_inspect_packet_emits_deterministic_golden_vectors(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path / "packet")

    first = inspect_packet(packet, target_profile="contest_one_video_replay")
    second = inspect_packet(packet, target_profile="contest_one_video_replay")

    assert first == second
    assert first["schema_version"] == "submission_packet_compiler.v1"
    assert first["target_profile"] == "contest_one_video_replay"
    assert first["target_profile_policy"]["contest_dispatch_candidate"] is True
    assert first["target_profile_policy"]["allows_one_video_replay"] is True
    assert first["score_claim"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["native_zipwire"]["status"] == "not_requested"
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


def test_native_zipwire_match_records_conformance_details(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path / "packet")
    python_archive = inspect_packet(packet)["archive"]
    assert python_archive is not None
    fake_zipwire = _write_fake_zipwire(
        tmp_path / "fake_zipwire",
        _zipwire_archive_from_python_archive(python_archive),
    )

    manifest = inspect_packet(packet, zipwire_bin=fake_zipwire)

    native = manifest["native_zipwire"]
    assert native["requested"] is True
    assert native["status"] == "matched"
    assert native["matched"] is True
    assert native["comparison"]["mismatches"] == []
    assert native["process"]["return_code"] == 0
    assert manifest["contest_compliance"]["blockers"] == []


def test_native_zipwire_mismatch_marks_blockers(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path / "packet")
    python_archive = inspect_packet(packet)["archive"]
    assert python_archive is not None
    native_payload = _zipwire_archive_from_python_archive(python_archive)
    assert isinstance(native_payload["members"], list)
    native_payload["members"][0]["crc32"] = "00000000"  # type: ignore[index]
    fake_zipwire = _write_fake_zipwire(tmp_path / "fake_zipwire", native_payload)

    manifest = inspect_packet(packet, zipwire_bin=fake_zipwire)

    native = manifest["native_zipwire"]
    assert native["status"] == "mismatch"
    assert native["matched"] is False
    assert native["comparison"]["mismatches"] == [
        {
            "path": "members[0].crc32",
            "python": python_archive["members"][0]["crc32"],
            "native": "00000000",
        }
    ]
    assert "native_zipwire:mismatch:members[0].crc32" in manifest["contest_compliance"]["blockers"]
    assert manifest["contest_compliance"]["contest_compliant_packet_shape"] is False


def test_cli_zipwire_flag_writes_native_section(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path / "packet")
    python_archive = inspect_packet(packet)["archive"]
    assert python_archive is not None
    fake_zipwire = _write_fake_zipwire(
        tmp_path / "fake_zipwire",
        _zipwire_archive_from_python_archive(python_archive),
    )
    json_out = tmp_path / "manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "submission_packet_compiler.py"),
            str(packet),
            "--zipwire-bin",
            str(fake_zipwire),
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["native_zipwire"]["requested"] is True
    assert written["native_zipwire"]["status"] == "matched"


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


def test_target_profiles_are_explicit_and_have_dispatch_policies(
    tmp_path: Path,
) -> None:
    packet = _write_packet(tmp_path / "packet")

    assert set(TARGET_PROFILES) == set(TARGET_PROFILE_POLICIES)
    assert "contest_generalized" in TARGET_PROFILES
    assert "production_edge_adaptive" in TARGET_PROFILES

    contest = inspect_packet(packet, target_profile="contest_generalized")
    assert contest["target_profile_policy"]["contest_dispatch_candidate"] is True
    assert contest["target_profile_policy"]["allows_one_video_replay"] is False
    assert contest["target_profile_policy"]["requires_cross_video_generalization"] is True

    edge = inspect_packet(packet, target_profile="production_edge_adaptive")
    assert edge["target_profile_policy"]["contest_dispatch_candidate"] is False
    assert edge["target_profile_policy"]["allows_optional_device_learning"] is True
    assert edge["target_profile_policy"]["requires_cross_video_generalization"] is True
