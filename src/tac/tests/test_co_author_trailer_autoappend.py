# SPDX-License-Identifier: MIT
"""Unit tests for the NO-co-author-trailer contract (operator NON-NEGOTIABLE 2026-05-31).

Operator NON-NEGOTIABLE 2026-05-31 verbatim: "there should be no co-author
trailer ever in our commit history."

This INVERTS the prior FIX-3 auto-append behavior (META-META 2026-05-08): the
serializer NO LONGER appends any Co-Authored-By trailer. ``_append_co_author_trailer``
is now a no-op kept only for backward-compat with callers/tests that import it;
``final_message`` in the serializer is the operator's message verbatim.

Catalog #119 was inverted from require-trailer to forbid-trailer
(``check_subagent_commits_have_no_co_author_trailer``); these tests pin the
serializer side of that contract.
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


def test_append_helper_is_noop_for_message_without_trailer() -> None:
    """The trailer-append helper must be a no-op (operator NON-NEGOTIABLE 2026-05-31)."""
    mod = _load_serializer()
    out = mod._append_co_author_trailer("fix: bug")
    assert out == "fix: bug"
    assert "Co-Authored-By" not in out


def test_append_helper_does_not_introduce_trailer_for_subject_only() -> None:
    mod = _load_serializer()
    out = mod._append_co_author_trailer("subject only")
    assert out == "subject only"
    assert "Co-Authored-By" not in out


def test_append_helper_preserves_message_ending_in_newline() -> None:
    mod = _load_serializer()
    out = mod._append_co_author_trailer("subject only\n")
    assert out == "subject only\n"
    assert "Co-Authored-By" not in out


def test_append_helper_does_not_strip_an_existing_trailer() -> None:
    """No-op means verbatim: it must not append, but it also must not edit/strip.

    A pre-existing operator-typed trailer would be flagged by Catalog #119 at
    commit-scan time, not silently rewritten here.
    """
    mod = _load_serializer()
    pre = "fix: bug\n\nCo-Authored-By: somebody <x@y.z>\n"
    out = mod._append_co_author_trailer(pre)
    assert out == pre


def test_append_helper_idempotent_under_repeat_application() -> None:
    mod = _load_serializer()
    once = mod._append_co_author_trailer("fix: bug")
    twice = mod._append_co_author_trailer(once)
    assert once == twice == "fix: bug"
