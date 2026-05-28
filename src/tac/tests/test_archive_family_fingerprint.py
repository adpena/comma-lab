# SPDX-License-Identifier: MIT
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

from tac.optimization.archive_family_fingerprint import (
    ARCHIVE_FAMILY_COVERAGE_REPORT_SCHEMA,
    ARCHIVE_FAMILY_FINGERPRINT_SCHEMA,
    build_archive_family_coverage_report,
    fingerprint_archive_family,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _fp11_selector_payload(selector_magic: bytes) -> bytes:
    source = b"hnerv-source"
    selector = selector_magic + b"\x00\x01payload"
    return b"FP11" + struct.pack("<I", len(source)) + source + struct.pack("<H", len(selector)) + selector


def test_archive_family_fingerprint_distinguishes_selector_generations(tmp_path: Path) -> None:
    fec3 = _write_zip(tmp_path / "fec3.zip", {"x": _fp11_selector_payload(b"FEC3")})
    fec6 = _write_zip(tmp_path / "fec6.zip", {"x": _fp11_selector_payload(b"FEC6")})

    fec3_probe = fingerprint_archive_family(fec3, repo_root=tmp_path)
    fec6_probe = fingerprint_archive_family(fec6, repo_root=tmp_path)

    assert fec3_probe["schema"] == ARCHIVE_FAMILY_FINGERPRINT_SCHEMA
    assert "fec3_compact_selector" in fec3_probe["detected_archive_families"]
    assert fec3_probe["score_affecting_adapter_implemented"] is False
    assert "fec3_compact_selector" in fec3_probe["unsupported_score_affecting_families"]
    assert "fec6_fixed_huffman_k16_selector" in fec6_probe["detected_archive_families"]
    assert fec6_probe["score_affecting_adapter_implemented"] is True
    assert fec6_probe["implemented_score_affecting_families"] == [
        "fec6_fixed_huffman_k16_selector"
    ]


def test_archive_family_fingerprint_covers_non_fec_payload_families(tmp_path: Path) -> None:
    psv4 = _write_zip(tmp_path / "psv4.zip", {"0.bin": b"PSV4\x01payload", "inflate.py": b""})
    dfl1 = _write_zip(tmp_path / "dfl1.zip", {"p": b"DFL1payload"})
    rpk1 = _write_zip(tmp_path / "rpk1.zip", {"p": b"\x00\x01\x02\x03RPK1payload"})
    hdm = _write_zip(tmp_path / "hdm.zip", {"x": b"\xfe\r\x80\xd5HDM9SO payload"})

    probes = {
        path.name: fingerprint_archive_family(path, repo_root=tmp_path)[
            "detected_archive_families"
        ]
        for path in (psv4, dfl1, rpk1, hdm)
    }

    assert "pact_nerv_selector_v4_packet" in probes["psv4.zip"]
    assert "multi_member_runtime_archive" in probes["psv4.zip"]
    assert "renderer_dfl1_payload" in probes["dfl1.zip"]
    assert "renderer_rpk1_payload" in probes["rpk1.zip"]
    assert "hnerv_latent_sidecar_hdm" in probes["hdm.zip"]


def test_archive_family_coverage_report_rolls_up_adapter_gaps(tmp_path: Path) -> None:
    archives = [
        _write_zip(tmp_path / "fec5.zip", {"x": _fp11_selector_payload(b"FEC5")}),
        _write_zip(tmp_path / "fec6.zip", {"x": _fp11_selector_payload(b"FEC6")}),
        _write_zip(tmp_path / "psv4.zip", {"0.bin": b"PSV4\x01payload"}),
    ]

    report = build_archive_family_coverage_report(archives, repo_root=tmp_path)

    assert report["schema"] == ARCHIVE_FAMILY_COVERAGE_REPORT_SCHEMA
    assert report["archive_count"] == 3
    assert report["implemented_score_affecting_family_counts"] == {
        "fec6_fixed_huffman_k16_selector": 1
    }
    assert report["unsupported_score_affecting_family_counts"]["fec5_fixed_huffman_k8_selector"] == 1
    assert report["unsupported_score_affecting_family_counts"]["pact_nerv_selector_v4_packet"] == 1
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
