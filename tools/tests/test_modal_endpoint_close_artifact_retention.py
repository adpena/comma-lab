# SPDX-License-Identifier: MIT
"""Self-protection for returned-artifact retention (task #1105, landing L2).

ALWAYS KEEP THE PAYLOAD applies to receipts. Before this fix persist_remote_result
wrote only ``bytes`` artifacts and recorded every other type as the bare label
``{"embedded_value_type": "str"}`` — the measured TYPE with the content discarded,
which is the canonical measure-and-discard signature. The rr4 CUDA row returned
``contest_auth_eval.json`` and ``report.txt`` as ``str``; both were dropped (rv2
finding FO-2) and had to be recovered from the Modal result cache.

Executed both-direction control, 2026-08-18, same three-artifact fixture:
  before — materialized_artifacts {"contest_auth_eval.json": {"embedded_value_type":
           "str"}, "report.txt": {...same...}}; returned_artifacts/ held ONLY
           provenance.json (the one bytes artifact).
  after  — all three on disk with bytes+sha256; provenance.json's sha256
           68c759aa7338cc7355759ed51f8194f0c3cca8e4ade93a7d83b180ac7700aa3e
           UNCHANGED, so the bytes path did not move.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import modal_endpoint_close as close  # noqa: E402

# The exact rr4 shape: the two score-bearing receipts arrive as str, not bytes.
REPORT_TXT = "seg 0.0006\npose 3.4e-05\n"
AUTH_JSON = json.dumps({"final_score": 0.15816036933414834}, sort_keys=True)


def _persist(artifacts: dict, out: Path) -> dict:
    safe, _ = close.persist_remote_result({"ok": True, "artifacts": artifacts}, out)
    return safe["materialized_artifacts"]


def test_str_artifacts_are_retained_byte_for_byte(tmp_path):
    """The rv2 FO-2 case: a str receipt must reach disk with its exact bytes."""

    records = _persist({"report.txt": REPORT_TXT, "contest_auth_eval.json": AUTH_JSON}, tmp_path)

    assert (tmp_path / "returned_artifacts" / "report.txt").read_bytes() == REPORT_TXT.encode()
    assert (
        tmp_path / "returned_artifacts" / "contest_auth_eval.json"
    ).read_bytes() == AUTH_JSON.encode()
    for name in ("report.txt", "contest_auth_eval.json"):
        assert records[name]["source_value_type"] == "str"
        assert records[name]["persisted_encoding"] == "utf-8"


def test_bytes_artifacts_are_unchanged_regression(tmp_path):
    """The pre-existing bytes path must not move: same file, same sha."""

    payload = b'{"device": "cpu"}'
    records = _persist({"provenance.json": payload}, tmp_path)

    assert (tmp_path / "returned_artifacts" / "provenance.json").read_bytes() == payload
    assert records["provenance.json"]["persisted_encoding"] == "bytes"
    assert (
        records["provenance.json"]["sha256"]
        == "68c759aa7338cc7355759ed51f8194f0c3cca8e4ade93a7d83b180ac7700aa3e"
    )


def test_no_artifact_is_recorded_as_a_bare_type_label(tmp_path):
    """The measure-and-discard signature itself: a record with no bytes on disk.

    This is the positive invariant, not merely the absence of the old key name: EVERY
    record must name a real file with a size and a digest.
    """

    records = _persist(
        {
            "a.txt": "text",
            "b.bin": b"bytes",
            "c.json": {"nested": [1, 2, 3]},
            "d.dat": bytearray(b"mutable"),
            "e.dat": memoryview(b"view"),
            "f.num": 12345,
        },
        tmp_path,
    )

    assert len(records) == 6
    for name, record in records.items():
        assert "embedded_value_type" not in record, f"{name} kept only a type label"
        assert record["sha256"] and record["bytes"] > 0
        assert Path(record["path"]).is_file()


@pytest.mark.parametrize(
    "payload,expected_bytes,encoding",
    [
        (b"raw", b"raw", "bytes"),
        (bytearray(b"raw"), b"raw", "bytes"),
        (memoryview(b"raw"), b"raw", "bytes"),
        ("text", b"text", "utf-8"),
        ("café ✓", "café ✓".encode(), "utf-8"),
        ({"k": 1}, None, "json"),
        ([1, 2], None, "json"),
        (42, None, "json"),
        (None, None, "json"),
    ],
)
def test_artifact_payload_bytes_never_returns_empty_handed(payload, expected_bytes, encoding):
    blob, enc = close.artifact_payload_bytes(payload)
    assert enc == encoding
    assert isinstance(blob, bytes) and blob
    if expected_bytes is not None:
        assert blob == expected_bytes


def test_unserializable_artifact_is_kept_lossily_not_dropped(tmp_path):
    """Raising would discard a whole paid harvest over one bad entry. Keep it, loudly."""

    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque sentinel>"

    records = _persist({"weird.dat": Opaque()}, tmp_path)

    assert records["weird.dat"]["lossy_repr"] is True
    assert records["weird.dat"]["persisted_encoding"] == "repr"
    assert (tmp_path / "returned_artifacts" / "weird.dat").read_bytes() == b"<Opaque sentinel>"


def test_nan_payload_falls_back_to_repr_rather_than_vanishing(tmp_path):
    """canonical_json_bytes uses allow_nan=False, so NaN must take the repr path."""

    records = _persist({"nan.json": {"x": float("nan")}}, tmp_path)

    assert records["nan.json"]["persisted_encoding"] == "repr"
    assert (tmp_path / "returned_artifacts" / "nan.json").read_bytes() == b"{'x': nan}"


def test_name_normalization_collisions_keep_both_payloads(tmp_path):
    """"a-b.txt" and "a_b.txt" both normalize to "a_b.txt"; neither may be overwritten."""

    records = _persist({"a-b.txt": "first", "a_b.txt": "second"}, tmp_path)

    assert len(records) == 2
    written = {p.read_bytes() for p in (tmp_path / "returned_artifacts").iterdir()}
    assert written == {b"first", b"second"}
    # Each record still names the artifact the remote actually returned.
    assert {r["source_name"] for r in records.values()} == {"a-b.txt", "a_b.txt"}


def test_unsafe_artifact_names_are_still_refused(tmp_path):
    """Retention must not widen the name-safety hole."""

    with pytest.raises(close.EndpointClosureError, match="unsafe returned artifact name"):
        _persist({"../escape.txt": "x"}, tmp_path)


def test_non_mapping_artifacts_are_still_refused(tmp_path):
    with pytest.raises(close.EndpointClosureError, match="artifacts must be a mapping"):
        _persist([("a", b"b")], tmp_path)
