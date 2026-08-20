from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tools.measure_ep725_xcodec_bridge_n600 import (
    G28Error,
    _assert_no_upstream_bytecode,
    _chunk_rows,
    _extract_single_member,
    _proof_contains_archive,
    _validate_checkpoint,
)


def test_upstream_bytecode_guard_refuses_recursive_tree_contamination(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    assert _assert_no_upstream_bytecode(upstream, stage="test")["passed"] is True
    cache = upstream / "__pycache__"
    cache.mkdir()
    (cache / "modules.cpython-313.pyc").write_bytes(b"not-authority")
    with pytest.raises(G28Error, match="bytecode contamination"):
        _assert_no_upstream_bytecode(upstream, stage="test")


def test_chunk_rows_cover_exact_n600_without_overlap() -> None:
    rows = _chunk_rows(12, 31)
    assert len(rows) == 50
    assert [pair for row in rows for pair in row["pair_ids"]] == list(range(600))
    assert rows[0]["byte_offset"] == 0
    assert rows[-1]["byte_offset"] + rows[-1]["byte_length"] == 600 * 2 * 31


def test_archive_proof_requires_identity_in_one_typed_row() -> None:
    digest = "a" * 64
    assert _proof_contains_archive(
        {"selected": {"archive_bytes": 80_295, "archive_sha256": digest}},
        archive_bytes=80_295,
        archive_sha256=digest,
    )
    assert not _proof_contains_archive(
        {"bytes_elsewhere": 80_295, "hash_elsewhere": digest},
        archive_bytes=80_295,
        archive_sha256=digest,
    )


def test_single_member_archive_is_reopened_exactly(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("0.bin", b"payload")
    member, receipt = _extract_single_member(archive)
    assert member == b"payload"
    assert receipt["member"]["sha256"] == hashlib.sha256(member).hexdigest()

    invalid = tmp_path / "invalid.zip"
    with zipfile.ZipFile(invalid, "w") as zf:
        zf.writestr("x", b"payload")
    with pytest.raises(G28Error, match=r"exactly one 0\.bin"):
        _extract_single_member(invalid)


def test_decode_checkpoint_rehashes_exact_range(tmp_path: Path) -> None:
    raw = tmp_path / "0.raw"
    raw.write_bytes(b"0123456789")
    row = {"index": 0, "pair_ids": [0], "byte_offset": 2, "byte_length": 4}
    value = {
        "schema": "tac.g28_ep725_bridge_decode_checkpoint.v1",
        "manifest_sha256": "b" * 64,
        **row,
        "range_sha256": hashlib.sha256(b"2345").hexdigest(),
    }
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    assert _validate_checkpoint(
        checkpoint,
        row=row,
        manifest_sha256="b" * 64,
        raw_path=raw,
    )["range_sha256"] == value["range_sha256"]

    raw.write_bytes(b"01x3456789")
    with pytest.raises(G28Error, match="checkpoint drift"):
        _validate_checkpoint(
            checkpoint,
            row=row,
            manifest_sha256="b" * 64,
            raw_path=raw,
        )
