"""Tests for the canonical payload retention helper (task #1001, P0)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.payload_retention import (
    PayloadRetentionError,
    retain_candidates,
    retain_payload,
    retention_root,
)
from tac.payload_retention_gate import scan_source


def test_retain_payload_writes_bytes_and_reports_custody(tmp_path: Path) -> None:
    blob = b"\x00\x01payload-bytes\xff"
    record = retain_payload(tmp_path / "p.bin", blob)
    assert (tmp_path / "p.bin").read_bytes() == blob
    assert record["bytes"] == len(blob)
    assert record["sha256"] == hashlib.sha256(blob).hexdigest()
    assert record["path"] == str(tmp_path / "p.bin")


def test_retain_payload_creates_parents_and_leaves_no_temp(tmp_path: Path) -> None:
    record = retain_payload(tmp_path / "a" / "b" / "p.bin", b"xy")
    assert Path(str(record["path"])).exists()
    assert not list(tmp_path.rglob(".*tmp"))


def test_retain_payload_overwrites_atomically(tmp_path: Path) -> None:
    target = tmp_path / "p.bin"
    retain_payload(target, b"first")
    second = retain_payload(target, b"second-and-longer")
    assert target.read_bytes() == b"second-and-longer"
    assert second["bytes"] == len(b"second-and-longer")


def test_retain_candidates_keeps_every_candidate_not_only_the_winner(tmp_path: Path) -> None:
    payloads = {"zlib9": b"a" * 40, "lzma": b"b" * 10, "brotli": b"c" * 25}
    rows = retain_candidates(tmp_path, payloads)
    assert set(rows) == set(payloads)
    for name, blob in payloads.items():
        assert (tmp_path / f"{name}.bin").read_bytes() == blob
        assert rows[name]["sha256"] == hashlib.sha256(blob).hexdigest()


def test_retention_root_picks_first_tier_that_fits(tmp_path: Path) -> None:
    full = tmp_path / "full_tier"
    roomy = tmp_path / "roomy_tier"
    full.mkdir()
    roomy.mkdir()
    root = retention_root("ddm_kp2", need_bytes=0, tiers=[full, roomy])
    # Both tiers are on the same filesystem here, so the FIRST is chosen.
    assert root == full / "ddm_kp2" / "retained"
    assert root.is_dir()


def test_retention_root_fails_closed_when_nothing_fits(tmp_path: Path) -> None:
    with pytest.raises(PayloadRetentionError) as excinfo:
        retention_root("ddm_kp2", need_bytes=1 << 62, tiers=[tmp_path])
    message = str(excinfo.value)
    assert "ALWAYS KEEP THE PAYLOAD" in message
    assert "storage-ROUTING" in message
    assert "GiB free" in message


def test_retention_root_skips_absent_tiers(tmp_path: Path) -> None:
    missing = Path("/nonexistent-volume-for-tests/pact")
    root = retention_root("ddm_kp2", tiers=[missing, tmp_path])
    assert root == tmp_path / "ddm_kp2" / "retained"


def test_helper_usage_clears_the_retention_gate() -> None:
    """The cure must actually clear the detector — otherwise it is not a cure."""
    cured = (
        "def race(q):\n"
        "    raw = q.tobytes()\n"
        "    blob = zlib.compress(raw, 9)\n"
        "    record = retain_payload(root / 'z.bin', blob)\n"
        "    return {'zlib9': len(blob), 'sha': record['sha256']}\n"
    )
    assert scan_source(cured, "cured.py") == []


def test_gate_still_fires_without_the_helper() -> None:
    """Two materializations are discarded here — the raw buffer AND the compressed blob."""
    uncured = (
        "def race(q):\n"
        "    raw = q.tobytes()\n"
        "    blob = zlib.compress(raw, 9)\n"
        "    return {'zlib9': len(blob)}\n"
    )
    findings = scan_source(uncured, "uncured.py")
    assert {finding.producer for finding in findings} == {"tobytes", "compress"}
