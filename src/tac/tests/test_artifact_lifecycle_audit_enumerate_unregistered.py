# SPDX-License-Identifier: MIT
"""Unit tests for FIX-4 META gate enumerate-unregistered (META-META 2026-05-08).

``audit_unregistered_long_lived_artifacts()`` enumerates tracked files under
LONG_LIVED_ARTIFACT_ROOTS and flags any that classify as UNKNOWN by the
registry AND are not in the explicit allowlist. This closes the codex
verification HIGH-3 gap where new long-lived artifacts could slip past the
META gate entirely simply by not matching any registered pattern.

Bug class: pre-FIX-4, ``run_meta_lifecycle_audit`` only iterated REGISTERED
patterns. New artifact paths outside the registry slipped past the META gate
entirely, recreating the provenance-vs-state-confusion bug class.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tac.artifact_lifecycle import (
    ALLOWLIST_RELPATH,
    LARGE_RESEARCH_JSON_MANIFEST_SCHEMA,
    LONG_LIVED_ARTIFACT_ROOTS,
    ArtifactClassifier,
    ArtifactKind,
    audit_large_rebuildable_research_json_artifacts,
    audit_unregistered_long_lived_artifacts,
    run_meta_lifecycle_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.email", "codex@example.invalid")
    _git(repo, "config", "user.name", "Codex")
    _git(repo, "config", "commit.gpgsign", "false")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _write_registry(repo: Path, yaml_text: str) -> None:
    state_dir = repo / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "artifact_kind_registry.yaml").write_text(yaml_text, encoding="utf-8")


def _write_allowlist(repo: Path, paths_with_reasons: list[tuple[str, str]]) -> None:
    state_dir = repo / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "artifact_classification_allowlist.json").write_text(
        json.dumps({
            "allowlisted_paths": [
                {"path": p, "reason": r} for p, r in paths_with_reasons
            ],
        }, indent=2),
        encoding="utf-8",
    )


def test_unregistered_long_lived_artifact_flagged_as_unknown(tmp_path: Path) -> None:
    """A tracked file under reports/ with no registry pattern flagged."""
    _init_repo(tmp_path)
    _write_registry(tmp_path, "")  # empty registry
    target = tmp_path / "reports" / "raw" / "mystery_output.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"score": 1}\n', encoding="utf-8")
    _commit_all(tmp_path, "tracked unregistered artifact")

    violations = audit_unregistered_long_lived_artifacts(tmp_path)
    assert any("reports/raw/mystery_output.json" in v for v in violations)
    assert any("UNKNOWN long-lived artifact" in v for v in violations)


def test_classified_long_lived_artifact_not_flagged(tmp_path: Path) -> None:
    """A tracked file matched by registry pattern is not surfaced."""
    _init_repo(tmp_path)
    _write_registry(
        tmp_path,
        '''
- pattern: "reports/raw/**/*.json"
  kind: DERIVED_OUTPUT
  rationale: "test classified"
''',
    )
    # fnmatch treats `**` as `.*`; need at least one intermediate dir for the
    # pattern to match, so place the file at reports/raw/<dir>/summary.json.
    target = tmp_path / "reports" / "raw" / "sweep" / "summary.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"score": 1}\n', encoding="utf-8")
    _commit_all(tmp_path, "tracked classified artifact")

    violations = audit_unregistered_long_lived_artifacts(tmp_path)
    # The summary.json matches the registry; should not be flagged.
    assert not any("reports/raw/sweep/summary.json" in v for v in violations)


def test_allowlisted_path_not_flagged(tmp_path: Path) -> None:
    """An UNKNOWN file in the allowlist is not surfaced."""
    _init_repo(tmp_path)
    _write_registry(tmp_path, "")
    _write_allowlist(
        tmp_path,
        [("reports/raw/mystery.json", "intentional one-off snapshot from incident")],
    )
    target = tmp_path / "reports" / "raw" / "mystery.json"
    target.parent.mkdir(parents=True)
    target.write_text('{}\n', encoding="utf-8")
    _commit_all(tmp_path, "allowlisted unregistered artifact")

    violations = audit_unregistered_long_lived_artifacts(tmp_path)
    assert not any("reports/raw/mystery.json" in v for v in violations)


def test_path_outside_long_lived_roots_not_enumerated(tmp_path: Path) -> None:
    """src/foo.py outside LONG_LIVED_ARTIFACT_ROOTS is never enumerated."""
    _init_repo(tmp_path)
    _write_registry(tmp_path, "")
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n", encoding="utf-8")
    _commit_all(tmp_path, "non-long-lived path")

    violations = audit_unregistered_long_lived_artifacts(tmp_path)
    assert not any("src/foo.py" in v for v in violations)


def test_untracked_file_not_enumerated(tmp_path: Path) -> None:
    """Untracked (not in git ls-files) artifacts are never enumerated."""
    _init_repo(tmp_path)
    _write_registry(tmp_path, "")
    # Create initial commit so git ls-files works.
    (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _commit_all(tmp_path, "seed")

    # Now write an untracked artifact; not committed.
    target = tmp_path / "reports" / "raw" / "untracked.json"
    target.parent.mkdir(parents=True)
    target.write_text('{}\n', encoding="utf-8")

    violations = audit_unregistered_long_lived_artifacts(tmp_path)
    assert not any("untracked.json" in v for v in violations)


def test_path_filter_restricts_audit(tmp_path: Path) -> None:
    """path_filter limits the audit to specified relpaths."""
    _init_repo(tmp_path)
    _write_registry(tmp_path, "")
    a = tmp_path / "reports" / "raw" / "a.json"
    b = tmp_path / "reports" / "raw" / "b.json"
    a.parent.mkdir(parents=True)
    a.write_text("{}\n", encoding="utf-8")
    b.write_text("{}\n", encoding="utf-8")
    _commit_all(tmp_path, "two unregistered artifacts")

    violations = audit_unregistered_long_lived_artifacts(
        tmp_path, path_filter={"reports/raw/a.json"},
    )
    # a.json should be flagged; b.json filtered out.
    assert any("reports/raw/a.json" in v for v in violations)
    assert not any("reports/raw/b.json" in v for v in violations)


def test_large_research_json_requires_compact_manifest(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_registry(
        tmp_path,
        '''
- pattern: ".omx/research/**/*.json"
  kind: HISTORICAL_PROVENANCE
  rationale: "test research json"
''',
    )
    target = tmp_path / ".omx" / "research" / "run" / "huge_plan.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"payload": "large"}\n', encoding="utf-8")
    _commit_all(tmp_path, "large research json")

    violations = audit_large_rebuildable_research_json_artifacts(
        tmp_path,
        threshold_bytes=8,
    )

    assert any("huge_plan.json" in v for v in violations)
    assert any("no sibling compact manifest" in v for v in violations)


def test_large_research_json_accepts_compact_manifest(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_registry(
        tmp_path,
        '''
- pattern: ".omx/research/**/*.json"
  kind: HISTORICAL_PROVENANCE
  rationale: "test research json"
''',
    )
    target = tmp_path / ".omx" / "research" / "run" / "huge_plan.json"
    summary = tmp_path / ".omx" / "research" / "run" / "huge_plan.compact_summary.json"
    manifest = tmp_path / ".omx" / "research" / "run" / "huge_plan.compact_manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"payload": "large"}\n', encoding="utf-8")
    summary.write_text('{"summary": "compact"}\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema": LARGE_RESEARCH_JSON_MANIFEST_SCHEMA,
                "source_json_path": ".omx/research/run/huge_plan.json",
                "source_json_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "source_json_bytes": target.stat().st_size,
                "rebuild_command": "python tools/rebuild_huge_plan.py",
                "compact_summary_path": ".omx/research/run/huge_plan.compact_summary.json",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "large research json manifest")

    violations = audit_large_rebuildable_research_json_artifacts(
        tmp_path,
        threshold_bytes=8,
    )

    assert violations == []


def test_large_research_json_rejects_stale_compact_manifest(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_registry(
        tmp_path,
        '''
- pattern: ".omx/research/**/*.json"
  kind: HISTORICAL_PROVENANCE
  rationale: "test research json"
''',
    )
    target = tmp_path / ".omx" / "research" / "run" / "huge_plan.json"
    summary = tmp_path / ".omx" / "research" / "run" / "huge_plan.compact_summary.json"
    manifest = tmp_path / ".omx" / "research" / "run" / "huge_plan.compact_manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"payload": "large"}\n', encoding="utf-8")
    summary.write_text('{"summary": "compact"}\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema": LARGE_RESEARCH_JSON_MANIFEST_SCHEMA,
                "source_json_path": ".omx/research/run/huge_plan.json",
                "source_json_sha256": "0" * 64,
                "source_json_bytes": 1,
                "source_inputs": ["source_state.json"],
                "compact_summary_path": ".omx/research/run/huge_plan.compact_summary.json",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "large research json stale manifest")

    violations = audit_large_rebuildable_research_json_artifacts(
        tmp_path,
        threshold_bytes=8,
    )

    assert any("source_json_bytes" in v for v in violations)
    assert any("source_json_sha256 is stale or missing" in v for v in violations)


def test_run_meta_audit_with_enumerate_unregistered_true_flags_unknown(tmp_path: Path) -> None:
    """When enumerate_unregistered=True, run_meta_lifecycle_audit also calls FIX-4."""
    _init_repo(tmp_path)
    _write_registry(tmp_path, "")
    target = tmp_path / "reports" / "raw" / "unknown.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    _commit_all(tmp_path, "unknown artifact")

    # Default behavior: enumerate_unregistered=False → no flag from FIX-4.
    violations_default = run_meta_lifecycle_audit(tmp_path)
    assert not any("reports/raw/unknown.json" in v for v in violations_default)

    # With FIX-4 enabled: should flag.
    violations_enabled = run_meta_lifecycle_audit(tmp_path, enumerate_unregistered=True)
    assert any("reports/raw/unknown.json" in v for v in violations_enabled)


def test_long_lived_artifact_roots_pinned() -> None:
    """The LONG_LIVED_ARTIFACT_ROOTS tuple is pinned to known canonical roots."""
    expected = {
        "experiments/results/",
        ".omx/state/",
        ".omx/research/",
        "reports/",
        "docs/",
        "runtime-rs/",
        "cuda/",
        "submissions/robust_current/",
    }
    assert set(LONG_LIVED_ARTIFACT_ROOTS) == expected


def test_allowlist_relpath_pinned() -> None:
    assert ALLOWLIST_RELPATH == ".omx/state/artifact_classification_allowlist.json"


def test_live_registry_classifies_top_level_omx_research_jsonl() -> None:
    cls = ArtifactClassifier(REPO_ROOT).classify_path(
        ".omx/research/cuda_cpu_axis_profile_updates.jsonl"
    )
    assert cls.kind == ArtifactKind.HISTORICAL_PROVENANCE
    assert cls.matched_pattern == ".omx/research/*.jsonl"


def test_corrupt_allowlist_treated_as_empty(tmp_path: Path) -> None:
    """Malformed allowlist JSON must not crash; treated as no allowlisted paths."""
    _init_repo(tmp_path)
    _write_registry(tmp_path, "")
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "artifact_classification_allowlist.json").write_text(
        "not valid json {", encoding="utf-8",
    )
    target = tmp_path / "reports" / "raw" / "x.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    _commit_all(tmp_path, "with corrupt allowlist")

    # Expectation: the function does not raise; it just treats allowlist as empty.
    violations = audit_unregistered_long_lived_artifacts(tmp_path)
    assert any("reports/raw/x.json" in v for v in violations)
