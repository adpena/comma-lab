"""Tests for Catalog #208 — check_docs_no_local_absolute_paths.

OSS v0.2.0-rc1 release prep audit finding D-2 self-protect (2026-05-14).

Bug class: tracked docs files containing hardcoded local absolute paths
(e.g. ``/Users/<operator>/Projects/pact/...``, ``/home/<user>/``,
``C:\\Users\\<user>\\...``, ``/private/var/``, ``/tmp/``) leak local
environment state into public-facing documentation. Per CLAUDE.md "Public
Disclosure Hygiene" non-negotiable.

Anchor: OSS v0.2.0-rc1 audit commit c293ba425 found 2 tracked files in
``docs/superpowers/**`` leaking ``/Users/adpena/Projects/pact/...`` paths
into the OSS release surface. This STRICT gate (warn-only at landing —
strict-flip pending bulk backfill) refuses any docs/**/*.md state that
re-introduces the pattern.

Sister of Catalog #109 (public PR intake clone source-provenance
corruption gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_docs_no_local_absolute_paths,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _make_corpus(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a fake repo with ``docs/`` populated by the given files."""
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────
# Positive cases — gate should flag
# ─────────────────────────────────────────────────────────────────────────


def test_macos_home_path_is_violation(tmp_path):
    """``/Users/<word>/...`` is a violation."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Run: `python -m foo --repo-root /Users/alice/Projects/pact`\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1
    assert "docs/example.md:1" in vs[0]


def test_linux_home_path_is_violation(tmp_path):
    """``/home/<word>/...`` is a violation."""
    root = _make_corpus(tmp_path, {
        "docs/example.md": "See [foo](/home/bob/projects/foo/data.json).\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_private_var_path_is_violation(tmp_path):
    """``/private/var/...`` is a violation."""
    root = _make_corpus(tmp_path, {
        "docs/example.md": "Output: /private/var/folders/.../T/tmp.txt\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_tmp_path_is_violation(tmp_path):
    """``/tmp/...`` is a violation (per FORBIDDEN /tmp pattern in CLAUDE.md)."""
    root = _make_corpus(tmp_path, {
        "docs/example.md": "Run `cp /tmp/scratch.json output.json`\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_windows_home_path_is_violation(tmp_path):
    """``C:\\Users\\<word>\\...`` is a violation."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "On Windows: C:\\Users\\alice\\Documents\\foo.txt\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_markdown_link_with_local_path_is_violation(tmp_path):
    """Local path inside a markdown link is a violation."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "- [reports/x.json](/Users/alice/Projects/pact/reports/x.json)\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_multiple_violations_per_file(tmp_path):
    """Each violating line contributes a violation row."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Path 1: /Users/alice/x\n"
            "Path 2: /home/bob/y\n"
            "Path 3: /tmp/zzz\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 3


def test_nested_doc_paths_scanned(tmp_path):
    """Nested docs subdirs (e.g. docs/superpowers/plans/) are scanned."""
    root = _make_corpus(tmp_path, {
        "docs/superpowers/plans/foo.md":
            "Run: `python --root /Users/alice/Projects/pact`\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1
    assert "docs/superpowers/plans/foo.md" in vs[0]


# ─────────────────────────────────────────────────────────────────────────
# Negative cases — gate should accept
# ─────────────────────────────────────────────────────────────────────────


def test_relative_path_is_accepted(tmp_path):
    """Repo-relative path is the canonical pattern; not a violation."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Run: `python -m foo --repo-root <repo-root>`\n"
            "See [results.jsonl](../reports/results.jsonl)\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_placeholder_path_is_accepted(tmp_path):
    """``<repo-root>/`` placeholder is the canonical replacement."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Output written to `<repo-root>/reports/foo.json`.\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_no_docs_dir_no_violations(tmp_path):
    """No docs/ directory → no violations."""
    vs = check_docs_no_local_absolute_paths(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert vs == []


def test_non_markdown_file_not_scanned(tmp_path):
    """Files outside ``*.md`` glob are not scanned."""
    root = _make_corpus(tmp_path, {
        "docs/example.txt":
            "Run: `python --root /Users/alice/Projects/pact`\n",
        "docs/example.json":
            '{"path": "/Users/alice/x"}\n',
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_unrelated_paths_not_flagged(tmp_path):
    """Paths NOT matching the local-absolute patterns are ignored."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "src/tac/foo.py is the canonical helper.\n"
            "Run from the repo root: `python -m tools.bar`.\n"
            "`uv pip install` is preferred.\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_tmp_substring_inside_word_not_flagged(tmp_path):
    """``attempt/tmp_path`` (word-boundary) does NOT match the /tmp/ pattern."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "The test fixture uses `attempt/tmpfile` inside the body.\n"
            "Note: `/some/other/tmp_path/` is also ignored.\n",
    })
    # The first line: `attempt/tmp` does NOT contain `/tmp/`
    # The second: `/some/other/tmp_path/` - we look for `/tmp/` preceded by
    # a non-path char, so it should match... actually it WILL match if the
    # word boundary includes the alpha char before. Let's verify behavior:
    # `e/tmp_path/` — the char before `/tmp/` is `r` (path char), but the
    # text is actually `tmp_path` not `tmp/`. Let me clarify the test:
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    # The string `/tmp_path/` does NOT contain `/tmp/`.
    assert vs == []


# ─────────────────────────────────────────────────────────────────────────
# Waiver mechanism
# ─────────────────────────────────────────────────────────────────────────


def test_html_comment_waiver_respected(tmp_path):
    """Same-line ``<!-- DOCS_LOCAL_PATH_OK:<reason> -->`` waives."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Run: `python --root /Users/alice/Projects/pact` "
            "<!-- DOCS_LOCAL_PATH_OK:operator-example-from-2026-04-10 -->\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_hash_comment_waiver_respected(tmp_path):
    """Same-line ``# DOCS_LOCAL_PATH_OK:<reason>`` waives (rare; markdown
    comment isn't standard but operators may use it inside fenced blocks)."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Path: /Users/alice/x  # DOCS_LOCAL_PATH_OK:test-fixture\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_placeholder_reason_literal_rejected(tmp_path):
    """Literal ``DOCS_LOCAL_PATH_OK:<reason>`` placeholder does NOT waive."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Path: /Users/alice/x "
            "<!-- DOCS_LOCAL_PATH_OK:<reason> -->\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1


def test_demo_block_marker_allows_fenced_paths(tmp_path):
    """``<!-- DEMO_LOCAL_PATH_OK -->`` 3-line-preceding marker waives
    literal paths inside the next code-fenced block."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Shell demo (uses real path for clarity):\n"
            "<!-- DEMO_LOCAL_PATH_OK -->\n"
            "```bash\n"
            "cd /Users/alice/Projects/pact\n"
            "python -m tools.foo\n"
            "```\n"
            "After the fence: paths return to flagged.\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


def test_demo_block_does_not_persist_past_fence(tmp_path):
    """The demo waiver does NOT extend past the code fence."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "<!-- DEMO_LOCAL_PATH_OK -->\n"
            "```bash\n"
            "cd /Users/alice/Projects/pact\n"
            "```\n"
            "But /Users/alice/Projects/other is flagged.\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 1
    assert ":5" in vs[0]


# ─────────────────────────────────────────────────────────────────────────
# Strict mode
# ─────────────────────────────────────────────────────────────────────────


def test_strict_mode_raises_on_violation(tmp_path):
    """strict=True raises PreflightError on any violation."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Run: `python --root /Users/alice/Projects/pact`\n",
    })
    with pytest.raises(PreflightError) as exc_info:
        check_docs_no_local_absolute_paths(
            repo_root=root, strict=True, verbose=False,
        )
    assert "docs/example.md:1" in str(exc_info.value)
    assert "Public Disclosure Hygiene" in str(exc_info.value)


def test_strict_mode_silent_on_clean_corpus(tmp_path):
    """strict=True is silent on clean corpus."""
    root = _make_corpus(tmp_path, {
        "docs/example.md":
            "Run: `python --root <repo-root>`\n",
    })
    vs = check_docs_no_local_absolute_paths(
        repo_root=root, strict=True, verbose=False,
    )
    assert vs == []


# ─────────────────────────────────────────────────────────────────────────
# D-2 anchor file regression — the 2 sanitized docs stay clean
# ─────────────────────────────────────────────────────────────────────────


def test_d2_plan_file_is_clean():
    """Catalog #208 D-2 anchor #1: anti-drift hardening plan stays clean.

    The plan file at docs/superpowers/plans/2026-04-10-anti-drift-runtime-
    hardening.md previously contained 3 hardcoded
    ``/Users/adpena/Projects/pact`` references (lines 121, 126, 135).
    These were sanitized to ``<repo-root>`` per OSS v0.2.0-rc1 audit D-2.
    """
    repo_root = Path(__file__).resolve().parents[3]
    plan = repo_root / "docs/superpowers/plans/2026-04-10-anti-drift-runtime-hardening.md"
    assert plan.is_file()
    text = plan.read_text(encoding="utf-8")
    assert "/Users/adpena/Projects/pact" not in text, (
        "D-2 regression: plan file re-introduced /Users/adpena/Projects/pact"
    )


def test_d2_spec_file_is_clean():
    """Catalog #208 D-2 anchor #2: anti-drift design spec stays clean.

    The spec file at docs/superpowers/specs/2026-04-10-anti-drift-runtime-
    design.md previously contained 13 hardcoded ``/Users/adpena/Projects/
    pact`` references in markdown link URLs (Context + Canonical Data Model
    sections). These were sanitized to ``../../../`` repo-relative links
    per OSS v0.2.0-rc1 audit D-2.
    """
    repo_root = Path(__file__).resolve().parents[3]
    spec = repo_root / "docs/superpowers/specs/2026-04-10-anti-drift-runtime-design.md"
    assert spec.is_file()
    text = spec.read_text(encoding="utf-8")
    assert "/Users/adpena/Projects/pact" not in text, (
        "D-2 regression: spec file re-introduced /Users/adpena/Projects/pact"
    )


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard (warn-only-aware)
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_d2_files_are_clean():
    """Regression guard: the 2 D-2 anchor files MUST remain clean.

    Catalog #208 is currently WARN-ONLY because 5 other docs files
    (outside D-2 scope) contain pre-existing violations. The 2 D-2
    anchor files were sanitized in the landing commit batch and must
    not regress.
    """
    vs = check_docs_no_local_absolute_paths(
        repo_root=None, strict=False, verbose=False,
    )
    d2_files = (
        "docs/superpowers/plans/2026-04-10-anti-drift-runtime-hardening.md",
        "docs/superpowers/specs/2026-04-10-anti-drift-runtime-design.md",
    )
    for d2 in d2_files:
        assert not any(d2 in v for v in vs), (
            f"D-2 regression: {d2} re-introduced a local-absolute-path leak.\n"
            f"Violations: {[v[:200] for v in vs if d2 in v]}"
        )
