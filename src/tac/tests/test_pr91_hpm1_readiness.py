from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tac.pr91_hpm1_codec import (
    DEFAULT_PR91_ARCHIVE,
    EXPECTED_PR91_ARCHIVE_SHA256,
    EXPECTED_PR91_HPM1_HPAC_SHA256,
    EXPECTED_PR91_HPM1_MASK_SHA256,
    EXPECTED_PR91_HPM1_TOKENS_SHA256,
    EXPECTED_PR91_MEMBER_X_SHA256,
    analyze_pr91_hpm1_runtime_sources,
)
from tac.pr91_hpm1_readiness import audit_pr91_hpm1_readiness

REPO = Path(__file__).resolve().parents[3]


def test_pr91_hpm1_readiness_fails_closed_on_missing_archive(tmp_path: Path) -> None:
    report = audit_pr91_hpm1_readiness(archive=tmp_path / "missing.zip")

    assert report["kind"] == "pr91_hpm1_readiness"
    assert report["score_claim"] is False
    assert report["dispatch_attempted"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["promotion_eligible"] is False
    assert report["gates"]["static_archive_custody"]["status"] == "missing"
    assert "static_archive_custody" in report["dispatch_blockers"]
    assert "full_hpm1_decode_600_frames" in report["dispatch_blockers"]


def test_pr91_hpm1_readiness_rejects_non_pr91_single_x_archive(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"not a pr85 bundle")

    report = audit_pr91_hpm1_readiness(archive=archive)

    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["member_x"]["exists"] is True
    assert report["member_x"]["matches_expected"] is False
    assert report["hpm1_mask_segment"]["exists"] is False
    assert "member_x_custody" in report["dispatch_blockers"]
    assert "hpm1_segment_custody" in report["dispatch_blockers"]


def test_pr91_hpm1_readiness_rejects_duplicate_zip_members(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr("x", b"second")

    report = audit_pr91_hpm1_readiness(archive=archive)

    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["member_x"]["zip_report"]["duplicates"] == ["x"]
    assert "member_x_custody" in report["dispatch_blockers"]
    assert "zip_wire_contract" in report["dispatch_blockers"]


def test_pr91_hpm1_readiness_rejects_central_local_name_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"payload")
    raw = bytearray(archive.read_bytes())
    # Local file header name starts at byte 30 for this minimal archive. The
    # central directory still names member x, so strict ZIP custody must fail.
    raw[30] = ord("y")
    archive.write_bytes(raw)

    report = audit_pr91_hpm1_readiness(archive=archive)
    wire = report["member_x"]["zip_report"]["wire_contract"]

    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["member_x"]["zip_report"]["status"] == "zip_member_read_failed"
    assert wire["passed"] is False
    assert wire["central_local_name_mismatches"]
    assert "member_x_custody" in report["dispatch_blockers"]
    assert "zip_wire_contract" in report["dispatch_blockers"]


def test_pr91_hpm1_readiness_rejects_unsafe_zip_member_name(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("../x", b"payload")

    report = audit_pr91_hpm1_readiness(archive=archive)
    wire = report["member_x"]["zip_report"]["wire_contract"]

    assert report["ready_for_exact_eval_dispatch"] is False
    assert wire["passed"] is False
    assert wire["unsafe_names"] == ["../x", "../x"]
    assert "member_x_custody" in report["dispatch_blockers"]
    assert "zip_wire_contract" in report["dispatch_blockers"]


@pytest.mark.skipif(not DEFAULT_PR91_ARCHIVE.is_file(), reason="PR91 public archive not present")
def test_pr91_hpm1_readiness_static_real_archive_passes_but_dispatch_stays_blocked() -> None:
    report = audit_pr91_hpm1_readiness()

    assert report["source_archive"]["matches_expected"] is True
    assert report["source_archive"]["sha256"] == EXPECTED_PR91_ARCHIVE_SHA256
    assert report["member_x"]["matches_expected"] is True
    assert report["member_x"]["sha256"] == EXPECTED_PR91_MEMBER_X_SHA256
    assert report["member_x"]["zip_report"]["wire_contract"]["passed"] is True
    assert report["gates"]["zip_wire_contract"]["passed"] is True
    assert report["runtime_source_inventory"]["required_source_files_present"] is True
    assert report["runtime_source_inventory"]["pycache_only"] is False
    runtime_paths = {row["path"] for row in report["runtime_source_inventory"]["files"]}
    assert {"inflate.py", "pr86_hpac.py"} <= runtime_paths
    assert report["gates"]["runtime_source_inventory"]["required_for_dispatch"] is True
    assert report["gates"]["runtime_source_inventory"]["passed"] is True
    assert report["hpm1_mask_segment"]["matches_expected"] is True
    assert report["hpm1_mask_segment"]["sha256"] == EXPECTED_PR91_HPM1_MASK_SHA256
    assert report["hpm1_payload"]["tokens_sha256"] == EXPECTED_PR91_HPM1_TOKENS_SHA256
    assert report["hpm1_payload"]["hpac_sha256"] == EXPECTED_PR91_HPM1_HPAC_SHA256
    assert report["gates"]["static_archive_custody"]["passed"] is True
    assert report["gates"]["hpm1_token_hpac_custody"]["passed"] is True
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["dispatch_blockers"] == [
        "byte_exact_hpm1_reencode",
        "exact_cuda_auth_eval_after_parity",
        "full_hpm1_decode_600_frames",
        "runtime_hpm1_loader_without_sidecars",
    ]


def test_pr91_hpm1_runtime_inventory_rejects_pycache_only_sources(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    pycache = runtime_dir / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "inflate.cpython-312.pyc").write_bytes(b"pyc")
    (pycache / "pr86_hpac.cpython-312.pyc").write_bytes(b"pyc")

    report = analyze_pr91_hpm1_runtime_sources(source_dir=runtime_dir)

    assert report["status"] == "failed_closed_missing_required_runtime_sources"
    assert report["required_source_files_present"] is False
    assert report["pycache_only"] is True
    assert report["missing_required_source_files"] == ["inflate.py", "pr86_hpac.py"]


@pytest.mark.skipif(not DEFAULT_PR91_ARCHIVE.is_file(), reason="PR91 public archive not present")
def test_pr91_hpm1_readiness_accepts_decode_reencode_parity_report_but_keeps_exact_eval_blocked() -> None:
    base = audit_pr91_hpm1_readiness()
    parity_report = _parity_report(
        archive_sha=base["source_archive"]["sha256"],
        hpm1_sha=base["hpm1_mask_segment"]["sha256"],
    )

    report = audit_pr91_hpm1_readiness(parity_report=parity_report)

    assert report["decode_reencode_parity"]["accepted"] is True
    assert report["gates"]["full_hpm1_decode_600_frames"]["passed"] is True
    assert report["gates"]["byte_exact_hpm1_reencode"]["passed"] is True
    assert report["gates"]["runtime_hpm1_loader_without_sidecars"]["passed"] is True
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["dispatch_blockers"] == ["exact_cuda_auth_eval_after_parity"]


def test_pr91_hpm1_readiness_rejects_bad_parity_report(tmp_path: Path) -> None:
    report = audit_pr91_hpm1_readiness(
        archive=tmp_path / "missing.zip",
        parity_report={"schema": "wrong", "score_claim": True},
    )

    parity = report["decode_reencode_parity"]
    assert parity["accepted"] is False
    assert "parity_report_schema_mismatch" in parity["blockers"]
    assert "parity_report_score_claim_must_be_false" in parity["blockers"]
    assert "full_hpm1_decode_600_frames" in report["dispatch_blockers"]


def test_audit_pr91_hpm1_readiness_cli_records_tool_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"not a pr85 bundle")
    out = tmp_path / "readiness.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_pr91_hpm1_readiness.py"),
            "--archive",
            str(archive),
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    tool_run = payload["tool_run_manifest"]
    assert payload["kind"] == "pr91_hpm1_readiness"
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False
    assert tool_run["tool"] == "tools/audit_pr91_hpm1_readiness.py"
    assert tool_run["score_claim"] is False
    assert tool_run["input_files"][0]["path"].endswith("archive.zip")


def _parity_report(*, archive_sha: str, hpm1_sha: str) -> dict:
    return {
        "schema": "pr91_hpm1_decode_reencode_parity_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "archive_sha256": archive_sha,
        "hpm1_mask_sha256": hpm1_sha,
        "device_contract": {"resolved_device": "cpu"},
        "full_decode": {
            "passed": True,
            "frame_count": 600,
            "decoded_masks_sha256": "d" * 64,
        },
        "byte_exact_reencode": {
            "passed": True,
            "reencoded_hpm1_sha256": hpm1_sha,
        },
        "runtime_loader": {
            "sidecar_free": True,
            "fallback_used": False,
        },
    }
