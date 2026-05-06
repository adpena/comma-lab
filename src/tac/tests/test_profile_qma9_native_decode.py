from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from experiments.profile_qma9_native_decode import (
    SCHEMA,
    extract_qma9_payload,
    profile_qma9_native_decode,
)


PYTHON_QMA9_FIXTURE = bytes(
    [
        0x51,
        0x4D,
        0x41,
        0x39,
        0x02,
        0x00,
        0x00,
        0x00,
        0x03,
        0x00,
        0x00,
        0x00,
        0x04,
        0x00,
        0x00,
        0x00,
        0x11,
        0x00,
        0x00,
        0x00,
        0x03,
        0x84,
        0xE7,
        0x13,
        0xBF,
        0x51,
        0x70,
        0xA3,
        0xD4,
        0x3B,
        0x40,
        0xD3,
        0x52,
        0xFD,
        0x5D,
        0x1E,
        0x00,
    ]
)
PYTHON_QMA9_RAW = bytes(
    [
        0,
        2,
        4,
        1,
        1,
        3,
        0,
        2,
        2,
        4,
        1,
        3,
        1,
        3,
        0,
        2,
        2,
        4,
        1,
        3,
        3,
        0,
        2,
        4,
    ]
)


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _large_qma9_placeholder() -> bytes:
    return b"QMA9" + (1).to_bytes(4, "little") + (1).to_bytes(4, "little") + (
        1
    ).to_bytes(4, "little") + (1001).to_bytes(4, "little") + (b"\0" * 1001)


def _compact_v5_bundle(mask: bytes) -> bytes:
    lengths = {
        "mask": len(mask),
        "model": 1001,
        "pose": 101,
        "post": 4,
        "shift": 5,
        "frac": 6,
        "frac2": 7,
        "frac3": 8,
    }
    header = b"".join(_u24(lengths[name]) for name in lengths)
    return (
        header
        + mask
        + b"M" * lengths["model"]
        + b"P" * lengths["pose"]
        + b"o" * lengths["post"]
        + b"s" * lengths["shift"]
        + b"f" * lengths["frac"]
        + b"g" * lengths["frac2"]
        + b"h" * lengths["frac3"]
        + b"B" * 223
        + b"R" * 273
        + b"randmulti"
    )


def test_extracts_qma9_mask_slice_from_zip_compact_v5_member(tmp_path: Path) -> None:
    mask = _large_qma9_placeholder()
    bundle = _compact_v5_bundle(mask)
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", bundle)
        zf.writestr("a", b"sideinfo")

    extracted, source = extract_qma9_payload(archive, input_format="auto")

    assert extracted == mask
    assert source["extraction"] == "compact_v5_micro_mask_slice"
    assert source["qma9_offset"] == 24
    assert source["compact_v5_micro_header"]["mask_bytes"] == len(mask)
    assert source["zip"]["member_name"] == "x"
    assert source["zip"]["all_file_members"] == ["x", "a"]


def test_native_qma9_profiler_reports_repeat_sha_and_no_score_claim(tmp_path: Path) -> None:
    qma9 = tmp_path / "fixture.qma9"
    qma9.write_bytes(PYTHON_QMA9_FIXTURE)
    output_json = tmp_path / "timing.json"
    expected_raw_sha = hashlib.sha256(PYTHON_QMA9_RAW).hexdigest()

    report = profile_qma9_native_decode(
        input_path=qma9,
        output_json=output_json,
        decoder_path=None,
        build_profile="debug",
        member=None,
        input_format="qma9",
        prefix_frames=None,
        repeat=2,
        expected_decoded_sha256=expected_raw_sha,
    )

    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert report == written
    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["remote_jobs_dispatched"] is False
    assert report["determinism"]["passed"] is True
    assert report["determinism"]["unique_decoded_sha256"] == [expected_raw_sha]
    assert report["determinism"]["matches_expected_decoded_sha256"] is True
    assert report["qma9_payload"]["header"]["decoded_mask_bytes"] == len(PYTHON_QMA9_RAW)
    assert report["metadata_timing"]["elapsed_seconds"] >= 0
    assert report["decode"]["repeat"] == 2
    assert report["decode"]["timings"][0]["decoded_bytes"] == len(PYTHON_QMA9_RAW)
    assert report["no_op_detection"]["decoded_output_nonempty"] is True
    assert report["no_op_detection"]["decoded_sha256_equals_encoded_sha256"] is False
