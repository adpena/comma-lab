from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from experiments.profile_pr85_qh0_record_anatomy import (
    build_profile,
    parse_qh0_records,
)


REPO = Path(__file__).resolve().parents[3]


def test_parse_qh0_records_fails_closed_on_bad_magic() -> None:
    with pytest.raises(Exception, match="unsupported renderer magic"):
        parse_qh0_records(b"BAD")


def test_real_pr85_qh0_record_anatomy_is_byte_closed() -> None:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    if not archive.is_file():
        pytest.skip("public PR85 intake archive is not present")

    profile = build_profile(archive)
    anatomy = profile["anatomy"]

    assert profile["planning_only"] is True
    assert profile["score_claim"] is False
    assert profile["dispatch_performed"] is False
    assert profile["renderer_member"]["bytes"] == anatomy["payload_bytes"]
    assert anatomy["magic"] == "QH0"
    assert anatomy["consumed_bytes"] == anatomy["payload_bytes"]
    assert anatomy["byte_accounting"]["records_plus_magic_bytes"] == anatomy["payload_bytes"]
    assert anatomy["record_count"] > 20
    assert anatomy["top_records_by_bytes"]


def test_profile_uses_single_member_archive_contract(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a", b"x")
        zf.writestr("b", b"y")

    with pytest.raises(Exception, match="expected one member"):
        build_profile(archive)
