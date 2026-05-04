"""Tests for the recovery-stub submissions/robust_current/unpack_renderer_payload.py.

Covers the safe-stub behavior: API contract preserved, fail-loud on real payloads
rather than silently mis-decoding. The full RPK1 byte-format parser will be
restored once experiments/build_sjkl_c067_archive.py is recovered (its packer
is the inverse — once it lands, byte format is fully recoverable).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_UNPACKER_PATH = (
    Path(__file__).resolve().parents[3]
    / "submissions" / "robust_current" / "unpack_renderer_payload.py"
)
spec = importlib.util.spec_from_file_location("unpack_renderer_payload", _UNPACKER_PATH)
unpacker = importlib.util.module_from_spec(spec)
sys.modules["unpack_renderer_payload"] = unpacker
spec.loader.exec_module(unpacker)


def test_module_has_required_api():
    """submission_archive.py expects _parse_payload."""
    assert hasattr(unpacker, "_parse_payload")
    assert callable(unpacker._parse_payload)


def test_module_has_public_alias():
    assert hasattr(unpacker, "parse_payload")


def test_empty_payload_returns_empty_dicts():
    """Per SAFE STUB contract: empty payload yields empty results without raising."""
    header, members = unpacker._parse_payload(b"")
    assert header == {"version": 0, "layout": "empty", "member_order": []}
    assert members == {}


def test_real_payload_raises_not_implemented():
    """Per SAFE STUB contract: real payloads must fail loud, not silently mis-decode."""
    fake_payload = b"\x5b\x98\x68\x43" + b"\x00" * 100  # mimics C067 p header
    with pytest.raises(NotImplementedError) as exc_info:
        unpacker._parse_payload(fake_payload)
    msg = str(exc_info.value)
    # Recovery pointer must be in the error message
    assert "recovery" in msg.lower() or "stub" in msg.lower()
    assert "build_sjkl_c067_archive" in msg or "subagent" in msg


def test_non_bytes_input_rejected():
    with pytest.raises(TypeError):
        unpacker._parse_payload("not bytes")
    with pytest.raises(TypeError):
        unpacker._parse_payload(None)
    with pytest.raises(TypeError):
        unpacker._parse_payload([1, 2, 3])


def test_bytearray_accepted():
    """bytearray is a valid bytes-like input."""
    header, members = unpacker._parse_payload(bytearray())
    assert members == {}


def test_caller_can_catch_and_record_error():
    """Verifies the caller-pattern that submission_archive.py uses works:
    catch the exception, record it as a validation error, continue.
    """
    errors: list[str] = []
    payload = b"some_packed_payload_bytes" * 10
    try:
        unpacker._parse_payload(payload)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"unpack failed: {exc}")
    assert len(errors) == 1
    assert "unpack failed" in errors[0]
