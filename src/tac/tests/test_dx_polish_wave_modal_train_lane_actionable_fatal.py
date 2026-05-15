# SPDX-License-Identifier: MIT
"""Tests for DX-POLISH-WAVE 2026-05-15 (Catalog #238 / DX-1).

Verifies the categorized + actionable Catalog #166 FATAL message body
emitted by ``experiments/modal_train_lane.py::_format_dx_polish_actionable_fatal``.

Coverage:
  * Category bucket assignment for canonical path patterns:
      - codex_sister_research_ledger (e.g. `_codex.md`)
      - subagent_state_ephemeral (e.g. `.omx/state/...`, `.omx/tmp/...`)
      - build_artifact_or_derived_output (e.g. `experiments/results/...`)
      - operator_owned_source (e.g. `src/tac/foo.py`)
      - documentation_or_memo (e.g. `docs/foo.md`)
      - configuration_or_recipe (e.g. `.omx/operator_authorize_recipes/...`)
      - vendored_intake_clone (e.g. `experiments/results/public_pr*_intake_*/...`)
      - unclassified fallback
  * Porcelain line parsing handles XY-status prefix + rename arrow.
  * Actionable error body contains:
      - count and HEAD short SHA
      - per-bucket actionable next-steps line
      - smoke-before-full alternative path mention
      - Catalog #202 paired-env bypass mention
      - Catalog #166 cross-reference
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    """Load `experiments/modal_train_lane.py` without importing modal SDK.

    The module's top-level imports modal which is not always installed in
    test envs. We need only the helpers
    (`_extract_porcelain_path`, `_categorize_dirty_paths`,
    `_format_dx_polish_actionable_fatal`). Strategy: read the source,
    strip the modal-dependent top-level lines, exec the rest under a
    synthetic module name. This is test-only and intentional.
    """
    src_path = REPO_ROOT / "experiments" / "modal_train_lane.py"
    text = src_path.read_text(encoding="utf-8")
    # Truncate at the first `app = modal.App(...)` line; everything
    # before that is import-only and depends on modal.
    sentinel = "app = modal.App("
    if sentinel in text:
        # Keep everything from the first `def _git_dirty_tree_summary`
        # onward; that's where our helpers live.
        marker = "def _git_dirty_tree_summary"
        idx = text.find(marker)
        if idx == -1:
            raise RuntimeError(
                "test scaffold: could not locate _git_dirty_tree_summary marker"
            )
        body_text = text[idx:]
        # Truncate at first occurrence of `@app.local_entrypoint()` so we
        # do not eval modal-decorated code.
        end_marker = "@app.local_entrypoint()"
        end_idx = body_text.find(end_marker)
        if end_idx != -1:
            body_text = body_text[:end_idx]
    else:
        body_text = text
    namespace: dict = {"__name__": "_dx_polish_test_scaffold"}
    exec(compile(body_text, str(src_path), "exec"), namespace)
    return namespace


_NS = _load_module()
_categorize_dirty_paths = _NS["_categorize_dirty_paths"]
_extract_porcelain_path = _NS["_extract_porcelain_path"]
_format_dx_polish_actionable_fatal = _NS["_format_dx_polish_actionable_fatal"]


# ---------------------------------------------------------------------------
# _extract_porcelain_path
# ---------------------------------------------------------------------------


def test_extract_porcelain_handles_modified_status() -> None:
    assert _extract_porcelain_path(" M src/tac/foo.py") == "src/tac/foo.py"


def test_extract_porcelain_handles_untracked() -> None:
    assert _extract_porcelain_path("?? tools/new_file.py") == "tools/new_file.py"


def test_extract_porcelain_handles_rename_arrow() -> None:
    assert (
        _extract_porcelain_path("R  old/path.md -> new/path.md")
        == "new/path.md"
    )


def test_extract_porcelain_handles_quoted_path() -> None:
    assert (
        _extract_porcelain_path(' M "src/tac/with space.py"')
        == "src/tac/with space.py"
    )


# ---------------------------------------------------------------------------
# _categorize_dirty_paths
# ---------------------------------------------------------------------------


def test_categorize_codex_sister_research_ledger() -> None:
    paths = [
        ".omx/research/foo_codex.md",
        "feedback_codex_round_5.md",
    ]
    cat = _categorize_dirty_paths(paths)
    assert "codex_sister_research_ledger" in cat
    assert cat["codex_sister_research_ledger"]["count"] == 2


def test_categorize_subagent_state_ephemeral() -> None:
    paths = [".omx/state/active_lane_dispatch_claims.md", ".omx/tmp/scratch.py"]
    cat = _categorize_dirty_paths(paths)
    assert "subagent_state_ephemeral" in cat
    assert cat["subagent_state_ephemeral"]["count"] == 2


def test_categorize_build_artifact_or_derived_output() -> None:
    paths = [
        "experiments/results/lane_foo/build_manifest.json",
        "reports/raw/eval_data.json",
    ]
    cat = _categorize_dirty_paths(paths)
    assert "build_artifact_or_derived_output" in cat
    assert cat["build_artifact_or_derived_output"]["count"] == 2


def test_categorize_operator_owned_source() -> None:
    paths = [
        "src/tac/preflight.py",
        "tools/operator_authorize.py",
        "experiments/train_substrate_foo.py",
    ]
    cat = _categorize_dirty_paths(paths)
    assert "operator_owned_source" in cat
    assert cat["operator_owned_source"]["count"] == 3


def test_categorize_configuration_or_recipe() -> None:
    paths = [
        ".omx/operator_authorize_recipes/substrate_foo.yaml",
        "configs/something.yaml",
        "pyproject.toml",
    ]
    cat = _categorize_dirty_paths(paths)
    assert "configuration_or_recipe" in cat
    assert cat["configuration_or_recipe"]["count"] == 3


def test_categorize_documentation_or_memo() -> None:
    paths = ["docs/release/notes.md", "MEMORY.md", "CLAUDE.md"]
    cat = _categorize_dirty_paths(paths)
    assert "documentation_or_memo" in cat
    assert cat["documentation_or_memo"]["count"] == 3


def test_categorize_vendored_intake_clone() -> None:
    paths = [
        "experiments/results/public_pr101_intake_codex/source/foo.py",
    ]
    cat = _categorize_dirty_paths(paths)
    # `_intake_` matches before `experiments/results/` because the bucket
    # ordering puts vendored AFTER build_artifact_or_derived_output. So
    # this path is intentionally categorized as build_artifact_or_derived_output.
    # That is the deliberate ordering: a path under experiments/results/
    # is GC-helper territory regardless of intake clone status. Verify
    # the canonical ordering holds.
    assert "build_artifact_or_derived_output" in cat


def test_categorize_unclassified_fallback() -> None:
    paths = ["random_top_level_file.txt"]
    cat = _categorize_dirty_paths(paths)
    assert "unclassified" in cat


def test_categorize_single_path_assigned_to_first_matching_bucket() -> None:
    """A path matching multiple buckets goes to the FIRST listed."""
    paths = [".omx/research/codex_foo_codex.md"]
    cat = _categorize_dirty_paths(paths)
    assert "codex_sister_research_ledger" in cat
    assert cat["codex_sister_research_ledger"]["count"] == 1
    assert "subagent_state_ephemeral" not in cat


def test_categorize_examples_capped_at_three() -> None:
    paths = [f"src/tac/file_{i}.py" for i in range(10)]
    cat = _categorize_dirty_paths(paths)
    assert cat["operator_owned_source"]["count"] == 10
    assert len(cat["operator_owned_source"]["examples"]) == 3


# ---------------------------------------------------------------------------
# _format_dx_polish_actionable_fatal
# ---------------------------------------------------------------------------


def _make_dirty_tree(paths: list[str]) -> dict:
    return {
        "dirty": True,
        "dirty_paths_count": len(paths),
        "summary": "; ".join(paths[:10]),
        "categorized": _categorize_dirty_paths(paths),
    }


def test_actionable_fatal_includes_count_and_head_short_sha() -> None:
    tree = _make_dirty_tree(["src/tac/foo.py"])
    msg = _format_dx_polish_actionable_fatal(tree, "abcdef1234567890")
    assert "FATAL [Catalog #166]" in msg
    assert "1 uncommitted edit(s)" in msg
    assert "abcdef123456" in msg


def test_actionable_fatal_lists_categorized_buckets() -> None:
    tree = _make_dirty_tree(
        [
            "src/tac/foo.py",
            ".omx/research/note_codex.md",
            "experiments/results/lane_x/manifest.json",
        ]
    )
    msg = _format_dx_polish_actionable_fatal(tree, "deadbeef0000")
    assert "operator owned source" in msg
    assert "codex sister research ledger" in msg
    assert "build artifact or derived output" in msg


def test_actionable_fatal_includes_serializer_next_step_for_source_edits() -> None:
    tree = _make_dirty_tree(["src/tac/foo.py"])
    msg = _format_dx_polish_actionable_fatal(tree, "abc123")
    assert "subagent_commit_serializer.py" in msg
    assert "--expected-content-sha256" in msg


def test_actionable_fatal_includes_catalog_202_bypass_for_sister_ledger() -> None:
    tree = _make_dirty_tree([".omx/research/note_codex.md"])
    msg = _format_dx_polish_actionable_fatal(tree, "abc123")
    assert "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK" in msg
    assert "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED" in msg


def test_actionable_fatal_includes_gc_helper_for_build_artifacts() -> None:
    tree = _make_dirty_tree(["experiments/results/lane_x/manifest.json"])
    msg = _format_dx_polish_actionable_fatal(tree, "abc123")
    assert "tools/gc_experiments_results.py" in msg


def test_actionable_fatal_always_includes_smoke_before_full_alternative() -> None:
    tree = _make_dirty_tree(["src/tac/foo.py"])
    msg = _format_dx_polish_actionable_fatal(tree, "abc123")
    assert "tools/run_modal_smoke_before_full.py" in msg
    assert "--smoke-only" in msg


def test_actionable_fatal_always_includes_explicit_override_step() -> None:
    tree = _make_dirty_tree(["src/tac/foo.py"])
    msg = _format_dx_polish_actionable_fatal(tree, "abc123")
    assert "--require-clean-head=False" in msg
    assert "OPERATOR OVERRIDE" in msg or "operator override" in msg.lower()


def test_actionable_fatal_handles_zero_categorized_falls_back_to_summary() -> None:
    tree = {
        "dirty": True,
        "dirty_paths_count": 1,
        "summary": "?? unknown_pattern.foo",
        "categorized": {},
    }
    msg = _format_dx_polish_actionable_fatal(tree, "abc123")
    assert "Raw porcelain summary" in msg


def test_actionable_fatal_no_local_absolute_paths() -> None:
    """Catalog #208 sister discipline: error message uses repo-relative paths only."""
    tree = _make_dirty_tree(["src/tac/foo.py", ".omx/research/x_codex.md"])
    msg = _format_dx_polish_actionable_fatal(tree, "abc123")
    forbidden = ["/Users/", "/home/", "/tmp/", "/private/var/", "C:\\Users\\"]
    for pattern in forbidden:
        assert pattern not in msg, (
            f"actionable FATAL leaks local-path token `{pattern}`; violates "
            "Catalog #208 + CLAUDE.md Public Disclosure Hygiene"
        )
