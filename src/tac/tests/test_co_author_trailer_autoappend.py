# SPDX-License-Identifier: MIT
"""Unit tests for FIX-3 Co-Authored-By trailer auto-append (META-META 2026-05-08).

The serializer auto-appends the canonical Co-Authored-By trailer to every commit
message unless --no-co-author is passed. The transformation must be idempotent
so that subagents that retry a failed commit don't accumulate duplicate trailers.

Bug class: feedback_meta_meta_commit_machinery_protections_20260508 — three
recent commits (00896b43, c6d09bbb, 89d6eba2) shipped without the Co-Authored-By
trailer because subagents forgot to add it manually. Auto-append + idempotency
makes the trailer a structural property of every serialized commit.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_serializer():
    path = REPO / "tools" / "subagent_commit_serializer.py"
    spec = importlib.util.spec_from_file_location("_subagent_commit_serializer_caat", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trailer_appended_to_message_without_one() -> None:
    mod = _load_serializer()
    out = mod._append_co_author_trailer("fix: bug")
    assert out.endswith("Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>\n")
    assert "fix: bug" in out


def test_trailer_idempotent_when_already_present() -> None:
    mod = _load_serializer()
    msg_with_trailer = "fix: bug\n\n" + mod.CO_AUTHOR_TRAILER + "\n"
    out = mod._append_co_author_trailer(msg_with_trailer)
    # Exactly one trailer line, no duplication.
    assert out.count(mod.CO_AUTHOR_TRAILER) == 1
    assert out == msg_with_trailer


def test_trailer_separator_two_newlines_when_no_trailing_newline() -> None:
    mod = _load_serializer()
    out = mod._append_co_author_trailer("subject only")
    assert "subject only\n\n" + mod.CO_AUTHOR_TRAILER + "\n" == out


def test_trailer_keeps_blank_line_when_message_already_ends_in_newline() -> None:
    mod = _load_serializer()
    out = mod._append_co_author_trailer("subject only\n")
    # Git convention: body and trailers are separated by a blank line, i.e.
    # two consecutive newlines. The implementation collapses to a single
    # extra "\n" when the message already ends in one.
    assert "subject only\n\n" + mod.CO_AUTHOR_TRAILER + "\n" == out


def test_trailer_canonical_value_pinned() -> None:
    mod = _load_serializer()
    # Pinned canonical value — must match CLAUDE.md commit guidance and
    # subagent prompt templates exactly.
    assert mod.CO_AUTHOR_TRAILER == (
        "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
    )


def test_trailer_present_anywhere_treated_idempotent() -> None:
    """Idempotency must hold even if trailer is buried mid-body (operator-edited)."""
    mod = _load_serializer()
    msg = (
        "fix: complex change\n\n"
        + mod.CO_AUTHOR_TRAILER + "\n\n"
        + "Additional note from operator after trailer.\n"
    )
    out = mod._append_co_author_trailer(msg)
    assert out == msg
    assert out.count(mod.CO_AUTHOR_TRAILER) == 1
