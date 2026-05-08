from __future__ import annotations

import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tools.build_track2_identity_packet import (
    Track2IdentityPacketError,
    _canonical_json_sha256,
    build_track2_identity_packet,
)


def _write_file(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def _packet_fixture(tmp_path: Path, members: list[tuple[str, bytes]] | None = None) -> Path:
    packet = tmp_path / "source_packet"
    _write_file(
        packet / "inflate.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\npython inflate.py \"$@\"\n",
        0o755,
    )
    _write_file(packet / "inflate.py", "print('identity runtime')\n")
    _write_file(packet / "decoder/runtime.py", "VALUE = 7\n")
    _write_zip(packet / "archive.zip", members or [("p", b"payload-bytes")])
    return packet


def test_build_track2_identity_packet_records_byte_closed_manifest(tmp_path: Path) -> None:
    source_packet = _packet_fixture(tmp_path, [("p", b"payload-bytes"), ("meta.json", b"{}")])
    output_packet = tmp_path / "track2_identity_packet"

    payload = build_track2_identity_packet(
        source_packet_dir=source_packet,
        output_packet_dir=output_packet,
        candidate_id="unit-track2",
        recorded_at_utc=datetime(2026, 5, 8, 14, 0, tzinfo=UTC),
    )

    manifest = json.loads((output_packet / "track2_identity_manifest.json").read_text())
    assert manifest == payload
    assert (output_packet / "archive.zip").read_bytes() == (source_packet / "archive.zip").read_bytes()
    assert (output_packet / "inflate.sh").read_bytes() == (source_packet / "inflate.sh").read_bytes()
    assert (output_packet / "decoder/runtime.py").read_bytes() == (
        source_packet / "decoder/runtime.py"
    ).read_bytes()

    assert manifest["schema"] == "track2_custom_decoder_identity_packet_v1"
    assert manifest["candidate_id"] == "unit-track2"
    assert manifest["recorded_at_utc"] == "2026-05-08T14:00:00Z"
    assert manifest["status"] == {
        "optimization_applied": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "score_claim": False,
        "ranking_claim": False,
        "promotion_eligible": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "byte_custody_only",
        "classification": "non_optimized_identity_scaffold",
    }
    assert manifest["archive"]["identity"]["byte_identical"] is True
    assert manifest["byte_closure"] == {
        "archive_copied_byte_for_byte": True,
        "runtime_copied_byte_for_byte": True,
        "runtime_tree_sha256_identical": True,
        "identity_roundtrip_passed": True,
        "inflate_output_roundtrip_attempted": False,
        "inflate_output_roundtrip_reason": (
            "scorers and inflate execution are intentionally not run by this "
            "identity scaffold; exact CUDA auth eval is the promotion gate"
        ),
    }
    assert manifest["archive"]["source"]["member_names"] == ["p", "meta.json"]
    assert manifest["archive"]["source"]["members"][0]["sha256"]
    assert (
        manifest["runtime_manifest"]["source"]["runtime_tree_sha256"]
        == manifest["runtime_manifest"]["output"]["runtime_tree_sha256"]
    )
    assert "exact_cuda_auth_eval_missing" in manifest["remaining_blockers"]
    assert any(
        check["name"] == "inflate_entrypoint_bash_syntax" and check["passed"] is True
        for check in manifest["fail_closed_checks"]
    )

    without_self = {key: value for key, value in manifest.items() if key != "manifest_sha256_excluding_self"}
    assert manifest["manifest_sha256_excluding_self"] == _canonical_json_sha256(without_self)


def test_build_track2_identity_packet_refuses_missing_inflate_entrypoint(tmp_path: Path) -> None:
    packet = tmp_path / "source_packet"
    _write_zip(packet / "archive.zip", [("p", b"payload")])

    with pytest.raises(Track2IdentityPacketError, match="inflate_entrypoint_present"):
        build_track2_identity_packet(
            source_packet_dir=packet,
            output_packet_dir=tmp_path / "out",
            recorded_at_utc=datetime(2026, 5, 8, 14, 0, tzinfo=UTC),
        )

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("members", "match"),
    [
        ([("../escape", b"x")], "zip_member_safe"),
        ([(".DS_Store", b"x")], "zip_member_safe"),
        ([("__MACOSX/._payload", b"x")], "zip_member_safe"),
    ],
)
def test_build_track2_identity_packet_rejects_unsafe_archive_members(
    tmp_path: Path,
    members: list[tuple[str, bytes]],
    match: str,
) -> None:
    packet = _packet_fixture(tmp_path, members)

    with pytest.raises(Track2IdentityPacketError, match=match):
        build_track2_identity_packet(
            source_packet_dir=packet,
            output_packet_dir=tmp_path / "out",
            recorded_at_utc=datetime(2026, 5, 8, 14, 0, tzinfo=UTC),
        )


def test_build_track2_identity_packet_rejects_duplicate_payload_containers(tmp_path: Path) -> None:
    packet = _packet_fixture(
        tmp_path,
        [("p", b"payload"), ("renderer_payload.bin", b"second-container")],
    )

    with pytest.raises(Track2IdentityPacketError, match="zip_at_most_one_payload_container"):
        build_track2_identity_packet(
            source_packet_dir=packet,
            output_packet_dir=tmp_path / "out",
            recorded_at_utc=datetime(2026, 5, 8, 14, 0, tzinfo=UTC),
        )


def test_build_track2_identity_packet_rejects_duplicate_zip_members(tmp_path: Path) -> None:
    packet = _packet_fixture(tmp_path, [("payload.bin", b"first"), ("payload.bin", b"second")])

    with pytest.raises(Track2IdentityPacketError, match="zip_no_duplicate_members"):
        build_track2_identity_packet(
            source_packet_dir=packet,
            output_packet_dir=tmp_path / "out",
            recorded_at_utc=datetime(2026, 5, 8, 14, 0, tzinfo=UTC),
        )


def test_build_track2_identity_packet_refuses_non_empty_output_without_force(tmp_path: Path) -> None:
    source_packet = _packet_fixture(tmp_path)
    output_packet = tmp_path / "out"
    _write_file(output_packet / "partner-note.txt", "keep me")

    with pytest.raises(Track2IdentityPacketError, match="not empty"):
        build_track2_identity_packet(
            source_packet_dir=source_packet,
            output_packet_dir=output_packet,
            recorded_at_utc=datetime(2026, 5, 8, 14, 0, tzinfo=UTC),
        )

    assert (output_packet / "partner-note.txt").read_text(encoding="utf-8") == "keep me"


def test_build_track2_identity_packet_refuses_output_ancestor_of_source_even_with_force(
    tmp_path: Path,
) -> None:
    source_packet = _packet_fixture(tmp_path / "outer")

    with pytest.raises(Track2IdentityPacketError, match="must not contain source packet directory"):
        build_track2_identity_packet(
            source_packet_dir=source_packet,
            output_packet_dir=tmp_path / "outer",
            force=True,
            recorded_at_utc=datetime(2026, 5, 8, 14, 0, tzinfo=UTC),
        )

    assert (source_packet / "archive.zip").is_file()
