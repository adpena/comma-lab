# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import pytest

from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
)
from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle
from tac.public_frontier_intake import (
    PublicFrontierIntakeError,
    profile_public_frontier_archive,
    render_markdown,
    write_outputs,
)


def _segments(*, mask: bytes = b"QMA9mask", randmulti: bytes = b"rand") -> dict[str, bytes]:
    rows = {
        "mask": mask,
        "model": b"model",
        "pose": b"pose",
        "post": b"post",
        "shift": b"shift",
        "frac": b"frac",
        "frac2": b"frac2",
        "frac3": b"frac3",
        "bias": b"b" * 223,
        "region": b"r" * 273,
        "randmulti": randmulti,
    }
    return {name: rows[name] for name in SEGMENT_ORDER}


def _write_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)


def _patch_first_local_header(
    path: Path,
    *,
    flag_bits: int | None = None,
    compress_type: int | None = None,
    mod_time: int | None = None,
    mod_date: int | None = None,
    crc32: int | None = None,
    compressed_size: int | None = None,
    uncompressed_size: int | None = None,
) -> None:
    raw = bytearray(path.read_bytes())
    fields = list(struct.unpack("<IHHHHHIIIHH", raw[:30]))
    assert fields[0] == 0x04034B50
    if flag_bits is not None:
        fields[2] = flag_bits
    if compress_type is not None:
        fields[3] = compress_type
    if mod_time is not None:
        fields[4] = mod_time
    if mod_date is not None:
        fields[5] = mod_date
    if crc32 is not None:
        fields[6] = crc32
    if compressed_size is not None:
        fields[7] = compressed_size
    if uncompressed_size is not None:
        fields[8] = uncompressed_size
    raw[:30] = struct.pack("<IHHHHHIIIHH", *fields)
    path.write_bytes(bytes(raw))


def test_profile_detects_charged_side_info_and_baseline_segment_delta(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.zip"
    candidate = tmp_path / "candidate.zip"
    baseline_x = pack_pr85_bundle(_segments(randmulti=b"old-rand"))
    candidate_x = pack_pr85_bundle(_segments(randmulti=b"RMB1new-rand"))
    _write_archive(baseline, [("x", baseline_x)])
    _write_archive(candidate, [("x", candidate_x), ("a", b"tiny-side-info")])

    report = profile_public_frontier_archive(
        candidate,
        label="PR92_like",
        baselines={"baseline": baseline},
    )

    assert report["score_claim"] is False
    assert report["strict_zip"]["valid"] is True
    assert report["primary_member"]["name"] == "x"
    assert report["side_info"]["charged_bytes"] == len(b"tiny-side-info")
    assert report["side_info"]["requires_runtime_contract_review"] is True
    assert report["archive_bound_candidate_contract_schema"] == (
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    )
    assert report["archive_bound_candidate_contract_surface_schema"] == (
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA
    )
    contract = report["archive_bound_candidate_contract"]
    assert contract["schema"] == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    assert contract["byte_closed_candidate_materialized"] is True
    assert contract["receiver_contract_satisfied"] is False
    assert {"public_frontier", "zip_container", "zip_ordering"}.issubset(
        set(contract["archive_substrate_tags"])
    )
    diff = report["baseline_diffs"][0]
    changed_names = {row["segment"] for row in diff["changed_segments"]}
    assert changed_names == {"randmulti"}
    randmulti = diff["changed_segments"][0]
    assert randmulti["baseline_codec"] == "opaque_pr85_segment"
    assert randmulti["candidate_codec"] == "RMB1_side_info_backed_randmulti"
    assert "Charged Side Info" in render_markdown(report)


def test_profile_rejects_central_local_name_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "mismatch.zip"
    x = pack_pr85_bundle(_segments())
    _write_archive(archive, [("x", x)])

    raw = bytearray(archive.read_bytes())
    signature, *_rest, name_len, _extra_len = struct.unpack("<IHHHHHIIIHH", raw[:30])
    assert signature == 0x04034B50
    assert name_len == 1
    raw[30:31] = b"y"
    archive.write_bytes(bytes(raw))

    report = profile_public_frontier_archive(archive, label="bad")

    assert report["strict_zip"]["valid"] is False
    assert "x:central_local_name_mismatch" in report["strict_zip"]["blockers"]


def test_profile_rejects_local_central_zip_header_field_mismatches(tmp_path: Path) -> None:
    archive = tmp_path / "header-mismatch.zip"
    x = pack_pr85_bundle(_segments())
    _write_archive(archive, [("x", x)])

    _patch_first_local_header(
        archive,
        flag_bits=0x0800,
        compress_type=zipfile.ZIP_DEFLATED,
        mod_time=1,
        mod_date=34,
        crc32=0,
        compressed_size=len(x) + 1,
        uncompressed_size=len(x) + 2,
    )

    report = profile_public_frontier_archive(archive, label="bad-header")

    assert report["strict_zip"]["valid"] is False
    blockers = set(report["strict_zip"]["blockers"])
    assert "x:central_local_compression_method_mismatch" in blockers
    assert "x:central_local_general_purpose_flags_mismatch" in blockers
    assert "x:central_local_mod_time_mismatch" in blockers
    assert "x:central_local_mod_date_mismatch" in blockers
    assert "x:central_local_crc32_mismatch" in blockers
    assert "x:central_local_compressed_size_mismatch" in blockers
    assert "x:central_local_uncompressed_size_mismatch" in blockers


def test_profile_fails_on_bad_zip_and_writes_outputs(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")

    with pytest.raises(PublicFrontierIntakeError, match="bad ZIP archive"):
        profile_public_frontier_archive(bad, label="bad")

    archive = tmp_path / "archive.zip"
    json_out = tmp_path / "out.json"
    markdown_out = tmp_path / "out.md"
    _write_archive(archive, [("x", pack_pr85_bundle(_segments()))])
    report = profile_public_frontier_archive(archive, label="ok")
    write_outputs(report, json_out=json_out, markdown_out=markdown_out)

    assert json.loads(json_out.read_text())["schema"] == "public_frontier_archive_intake_v1"
    assert "# Public Frontier Archive Intake" in markdown_out.read_text()
