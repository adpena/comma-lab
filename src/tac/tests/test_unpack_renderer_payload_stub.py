# SPDX-License-Identifier: MIT
"""Regression tests for the live robust renderer payload unpacker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_UNPACKER_PATH = (
    Path(__file__).resolve().parents[3]
    / "submissions"
    / "robust_current"
    / "unpack_renderer_payload.py"
)
spec = importlib.util.spec_from_file_location("unpack_renderer_payload", _UNPACKER_PATH)
assert spec is not None and spec.loader is not None
unpacker = importlib.util.module_from_spec(spec)
sys.modules["unpack_renderer_payload"] = unpacker
spec.loader.exec_module(unpacker)


def test_module_has_required_api() -> None:
    assert hasattr(unpacker, "_parse_payload")
    assert callable(unpacker._parse_payload)


def test_module_has_public_alias() -> None:
    assert hasattr(unpacker, "parse_payload")
    assert unpacker.parse_payload is not None


def test_empty_payload_fails_loud() -> None:
    with pytest.raises(ValueError, match="too short"):
        unpacker._parse_payload(b"")


def test_unknown_real_payload_fails_loud() -> None:
    fake_payload = b"\x5b\x98\x68\x43" + b"\x00" * 100
    with pytest.raises(ValueError, match="bad renderer payload magic"):
        unpacker._parse_payload(fake_payload)


def test_non_bytes_input_rejected() -> None:
    for payload in ("not bytes", None, [1, 2, 3]):
        with pytest.raises((TypeError, ValueError)):
            unpacker._parse_payload(payload)


def test_public_alias_matches_private_parser() -> None:
    with pytest.raises(ValueError, match="too short"):
        unpacker.parse_payload(b"")
