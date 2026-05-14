# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

from tac.codec_op_archive_substitution import (
    ArchiveSubstitutionError,
    build_archive_substitution_candidate,
    inspect_archive_for_substitution,
)
from tac.repo_io import sha256_bytes, sha256_file


def _zip_info(
    name: str,
    *,
    date_time: tuple[int, int, int, int, int, int] = (2024, 5, 7, 12, 30, 0),
    compress_type: int = zipfile.ZIP_DEFLATED,
    mode: int = 0o640,
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=date_time)
    info.compress_type = compress_type
    info.external_attr = (mode & 0xFFFF) << 16
    info.internal_attr = 0
    info.create_system = 3
    return info


def _write_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", allowZip64=False) as zf:
        for name, payload in members:
            zf.writestr(
                _zip_info(name),
                payload,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def test_substitutes_physical_member_deterministically_and_records_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    replacement = b"codec-op-new-pose-stream"
    _write_archive(
        source,
        [
            ("inflate.sh", b"#!/bin/sh\n"),
            ("optimized_poses.bin", b"old-pose-stream"),
            ("renderer.bin", b"renderer"),
        ],
    )
    old_member_sha = sha256_bytes(b"old-pose-stream")
    out1 = tmp_path / "candidate1.zip"
    out2 = tmp_path / "candidate2.zip"

    manifest = build_archive_substitution_candidate(
        source_archive=source,
        expected_source_archive_sha256=sha256_file(source),
        expected_source_archive_bytes=source.stat().st_size,
        target_member_name="optimized_poses.bin",
        expected_target_member_sha256=old_member_sha,
        expected_target_member_bytes=len(b"old-pose-stream"),
        replacement_substream=replacement,
        output_archive=out1,
        candidate_id="codec_op_pose_k2",
    )
    manifest2 = build_archive_substitution_candidate(
        source_archive=source,
        expected_source_archive_sha256=sha256_file(source),
        expected_source_archive_bytes=source.stat().st_size,
        target_member_name="optimized_poses.bin",
        expected_target_member_sha256=old_member_sha,
        expected_target_member_bytes=len(b"old-pose-stream"),
        replacement_substream=replacement,
        output_archive=out2,
        candidate_id="codec_op_pose_k2",
    )

    assert sha256_file(out1) == sha256_file(out2)
    assert manifest["archive"]["old_archive_sha256"] == sha256_file(source)
    assert manifest["archive"]["new_archive_sha256"] == sha256_file(out1)
    assert manifest["archive"]["archive_byte_delta"] == (
        out1.stat().st_size - source.stat().st_size
    )
    assert manifest["target_member"]["old_bytes"] == len(b"old-pose-stream")
    assert manifest["target_member"]["new_bytes"] == len(replacement)
    assert manifest["target_member"]["member_byte_delta"] == (
        len(replacement) - len(b"old-pose-stream")
    )
    assert manifest["target_member"]["new_sha256"] == sha256_bytes(replacement)
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "exact_runtime_parity_not_supplied" in manifest["dispatch_readiness"]["blockers"]
    assert "matching_lane_dispatch_claim_not_supplied" in manifest["dispatch_readiness"]["blockers"]
    assert manifest2["archive"]["new_archive_sha256"] == manifest["archive"]["new_archive_sha256"]

    source_info = inspect_archive_for_substitution(source)
    candidate_info = inspect_archive_for_substitution(out1)
    roundtrip_member = candidate_info.find_member("optimized_poses.bin")
    assert candidate_info.member_names() == [
        "inflate.sh",
        "optimized_poses.bin",
        "renderer.bin",
    ]
    assert roundtrip_member.payload == replacement
    for before, after in zip(source_info.members, candidate_info.members, strict=True):
        assert before.name == after.name
        assert before.date_time == after.date_time
        assert before.external_attr == after.external_attr
        assert before.compress_type == after.compress_type

    out3 = tmp_path / "candidate3.zip"
    ready_manifest = build_archive_substitution_candidate(
        source_archive=source,
        expected_source_archive_sha256=sha256_file(source),
        expected_source_archive_bytes=source.stat().st_size,
        target_member_name="optimized_poses.bin",
        expected_target_member_sha256=old_member_sha,
        expected_target_member_bytes=len(b"old-pose-stream"),
        replacement_substream=replacement,
        output_archive=out3,
        candidate_id="codec_op_pose_k2",
        exact_runtime_parity={
            "safe_for_exact_eval_dispatch": True,
            "source_archive_sha256": sha256_file(source),
            "candidate_archive_sha256": sha256_file(out1),
        },
        lane_claim={
            "lane_id": "codec-op-pose-k2",
            "status": "active",
            "source_archive_sha256": sha256_file(source),
            "candidate_archive_sha256": sha256_file(out1),
        },
    )
    assert ready_manifest["ready_for_exact_eval_dispatch"] is True
    assert ready_manifest["dispatch_readiness"]["blockers"] == []


def test_rejects_source_archive_identity_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _write_archive(source, [("target.bin", b"old")])

    with pytest.raises(ArchiveSubstitutionError, match="archive byte count mismatch"):
        build_archive_substitution_candidate(
            source_archive=source,
            expected_source_archive_sha256=sha256_file(source),
            expected_source_archive_bytes=source.stat().st_size + 1,
            target_member_name="target.bin",
            expected_target_member_sha256=sha256_bytes(b"old"),
            expected_target_member_bytes=3,
            replacement_substream=b"new",
            output_archive=tmp_path / "out.zip",
            candidate_id="bad-source",
        )


def test_no_op_replacement_stays_non_dispatchable_even_with_claims(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    payload = b"same-bytes"
    _write_archive(source, [("target.bin", payload)])
    out = tmp_path / "out.zip"

    manifest = build_archive_substitution_candidate(
        source_archive=source,
        expected_source_archive_sha256=sha256_file(source),
        expected_source_archive_bytes=source.stat().st_size,
        target_member_name="target.bin",
        expected_target_member_sha256=sha256_bytes(payload),
        expected_target_member_bytes=len(payload),
        replacement_substream=payload,
        output_archive=out,
        candidate_id="no-op",
        exact_runtime_parity={
            "safe_for_exact_eval_dispatch": True,
        },
        lane_claim={
            "lane_id": "codec-op-no-op",
            "status": "active",
        },
    )

    assert manifest["target_member"]["no_op_payload"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "replacement_payload_matches_source_member" in manifest["dispatch_readiness"]["blockers"]


@pytest.mark.parametrize(
    ("members", "match"),
    [
        ([("target.bin", b"a"), ("target.bin", b"b")], "duplicate archive member"),
        ([("._target.bin", b"a")], "resource fork archive member"),
        ([("../target.bin", b"a")], "zip-slip archive member path"),
        ([(".hidden", b"a")], "hidden archive member"),
    ],
)
def test_rejects_duplicate_hidden_resource_and_traversal_members(
    tmp_path: Path,
    members: list[tuple[str, bytes]],
    match: str,
) -> None:
    source = tmp_path / "bad.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write_archive(source, members)

    with pytest.raises(ArchiveSubstitutionError, match=match):
        inspect_archive_for_substitution(source)


def test_rejects_member_extra_metadata(tmp_path: Path) -> None:
    source = tmp_path / "extra.zip"
    info = _zip_info("target.bin")
    info.extra = b"\x01\x00\x00\x00"
    with zipfile.ZipFile(source, "w", allowZip64=False) as zf:
        zf.writestr(info, b"old", compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    with pytest.raises(ArchiveSubstitutionError, match="extra metadata unsupported"):
        inspect_archive_for_substitution(source)


def test_rejects_packed_payload_container_without_explicit_override(
    tmp_path: Path,
) -> None:
    source = tmp_path / "packed.zip"
    _write_archive(source, [("p", b"packed-payload")])

    with pytest.raises(ArchiveSubstitutionError, match="packed payload container"):
        build_archive_substitution_candidate(
            source_archive=source,
            expected_source_archive_sha256=sha256_file(source),
            expected_source_archive_bytes=source.stat().st_size,
            target_member_name="p",
            expected_target_member_sha256=sha256_bytes(b"packed-payload"),
            expected_target_member_bytes=len(b"packed-payload"),
            replacement_substream=b"new-logical-stream-only",
            output_archive=tmp_path / "out.zip",
            candidate_id="packed-logical-unsafe",
        )


def test_cli_builds_candidate_and_cites_codec_op_manifest(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    source = tmp_path / "source.zip"
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"codec-op-bytes")
    _write_archive(source, [("target.bin", b"old-bytes")])
    sweep_manifest = tmp_path / "sweep.json"
    sweep_manifest.write_text(
        json.dumps(
            {
                "schema_version": "codec_op_param_sweep_manifest.v1",
                "candidates": [
                    {
                        "candidate_id": "codec_op_candidate",
                        "op_module": "tac.fake",
                        "op_class": "OpFake",
                        "op_params": {"k": 2},
                        "candidate_substream_bytes": replacement.stat().st_size,
                        "ready_for_exact_eval_dispatch": False,
                        "score_claim": False,
                        "evidence_semantics": "unit_test",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "candidate.zip"
    manifest_path = tmp_path / "manifest.json"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/build_codec_op_archive_substitution_candidate.py",
            "--source-archive",
            str(source),
            "--expected-source-archive-sha256",
            sha256_file(source),
            "--expected-source-archive-bytes",
            str(source.stat().st_size),
            "--target-member",
            "target.bin",
            "--expected-target-member-sha256",
            sha256_bytes(b"old-bytes"),
            "--expected-target-member-bytes",
            str(len(b"old-bytes")),
            "--replacement-substream",
            str(replacement),
            "--expected-replacement-sha256",
            sha256_file(replacement),
            "--expected-replacement-bytes",
            str(replacement.stat().st_size),
            "--output-archive",
            str(out),
            "--manifest-output",
            str(manifest_path),
            "--candidate-id",
            "codec_op_candidate",
            "--codec-op-manifest",
            str(sweep_manifest),
            "--codec-op-candidate-id",
            "codec_op_candidate",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert out.is_file()
    assert manifest["codec_op_manifest"]["candidate_id"] == "codec_op_candidate"
    assert manifest["replacement_substream"]["path"] == replacement.as_posix()
    assert manifest["target_member"]["new_sha256"] == sha256_file(replacement)
