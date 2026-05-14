# SPDX-License-Identifier: MIT
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

from tac.endgame_archive_decision import build_endgame_decision_profile, render_markdown
from tac.pr85_bundle import PR85_HEADERLESS_RANDMULTI_SPECS, SEGMENT_ORDER, pack_pr85_bundle


def _br(data: bytes) -> bytes:
    import pytest

    brotli = pytest.importorskip("brotli")
    return brotli.compress(data, quality=5)


def _qma9(bitstream: bytes = b"\x00" * 16) -> bytes:
    return b"QMA9" + struct.pack("<IIII", 600, 512, 384, len(bitstream)) + bitstream


def _legacy_randmulti() -> bytes:
    return _br(b"legacy-randmulti-payload")


def _rmb1_randmulti() -> bytes:
    raw_mask = b"\x00" * (75 * sum(spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS))
    mask_br = _br(raw_mask)
    vals_br = _br(b"")
    return b"RMB1" + len(mask_br).to_bytes(2, "little") + mask_br + vals_br


def _rsb1_actions(count: int = 600) -> bytes:
    raw = bytes([idx % 4 for idx in range(count)])
    return b"RSB1" + count.to_bytes(2, "little") + bytes([1, 0]) + _br(raw)


def _segments(*, randmulti: bytes | None = None) -> dict[str, bytes]:
    rows = {
        "mask": _qma9(),
        "model": _br(b"QH0\0model"),
        "pose": _br(b"P1D1pose"),
        "post": _br(b"\x00" * 2400),
        "shift": _br(b"SD4" + b"\0"),
        "frac": _br(b"FV1" + b"\0"),
        "frac2": _br(b"FH2" + b"\0"),
        "frac3": _br(b"FD3" + b"\0"),
        "bias": _br(b"BD1" + b"\0"),
        "region": _br(b"RH1" + b"\0"),
        "randmulti": randmulti if randmulti is not None else _legacy_randmulti(),
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


def test_endgame_profile_estimates_rmb1_side_info_transplant_delta(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.zip"
    candidate = tmp_path / "candidate.zip"
    baseline_x = pack_pr85_bundle(_segments(), header_mode="explicit_30")
    candidate_x = pack_pr85_bundle(_segments(randmulti=_rmb1_randmulti()), header_mode="explicit_30")
    _write_archive(baseline, [("x", baseline_x)])
    _write_archive(candidate, [("x", candidate_x), ("a", _rsb1_actions())])

    profile = build_endgame_decision_profile(
        {"frontier": baseline, "candidate": candidate},
        frontier_label="frontier",
    )

    assert profile["score_claim"] is False
    archive_rows = {row["label"]: row for row in profile["archives"]}
    assert archive_rows["candidate"]["side_info"]["members"][0]["validation"]["status"] == "ok"
    comparison = profile["comparisons_to_frontier"][0]
    estimate = next(row for row in comparison["transplant_estimates"] if row["segment"] == "randmulti")
    assert estimate["requires_candidate_side_info"] is True
    assert estimate["estimated_archive_delta_bytes"] == comparison["archive_delta_bytes"]
    assert "Endgame Archive Decision Profile" in render_markdown(profile)


def test_endgame_profile_fails_closed_on_bad_rsb1_side_info(tmp_path: Path) -> None:
    archive = tmp_path / "bad_side.zip"
    x_payload = pack_pr85_bundle(_segments(), header_mode="explicit_30")
    _write_archive(archive, [("x", x_payload), ("a", b"RSB1" + b"\x58\x02\x01\x00not-brotli")])

    profile = build_endgame_decision_profile({"bad": archive}, frontier_label="bad")
    report = profile["archives"][0]

    assert report["decision_support"]["valid_for_byte_decision"] is False
    assert "side_member_validation_failed:a#1" in report["decision_support"]["blockers"]
    assert report["side_info"]["members"][0]["validation"]["status"] == "failed"


def test_endgame_profile_rejects_central_local_name_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "mismatch.zip"
    x_payload = pack_pr85_bundle(_segments(), header_mode="explicit_30")
    _write_archive(archive, [("x", x_payload)])

    raw = bytearray(archive.read_bytes())
    signature, *_rest, name_len, _extra_len = struct.unpack("<IHHHHHIIIHH", raw[:30])
    assert signature == 0x04034B50
    assert name_len == 1
    raw[30:31] = b"y"
    archive.write_bytes(bytes(raw))

    profile = build_endgame_decision_profile({"bad": archive}, frontier_label="bad")
    report = profile["archives"][0]

    assert report["strict_zip"]["valid"] is False
    assert "x:central_local_name_mismatch" in report["strict_zip"]["blockers"]
    assert report["decision_support"]["valid_for_byte_decision"] is False
