from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.artifact_lifecycle import (
    CommittedRangeProvenanceGuard,
    run_meta_lifecycle_audit,
)
from tac.preflight import _artifact_lifecycle_changed_paths


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.email", "codex@example.invalid")
    _git(repo, "config", "user.name", "Codex")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def test_committed_range_provenance_guard_flags_scalar_mutation(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({"archive_sha256": "aaa", "status": "created"}) + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "base")
    provenance.write_text(
        json.dumps({"archive_sha256": "bbb", "status": "created"}) + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "mutate provenance")

    result = CommittedRangeProvenanceGuard(tmp_path).check(
        "provenance.json",
        base_ref="HEAD~1",
    )

    assert any("archive_sha256" in violation for violation in result.violations)
    assert any("committed range HEAD~1..HEAD" in violation for violation in result.violations)


def test_committed_range_provenance_guard_allows_append_only_growth(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    provenance = tmp_path / "recovery_metadata.json"
    provenance.write_text(
        json.dumps({"status": "created", "attempts": [{"started": "a"}]}) + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "base")
    provenance.write_text(
        json.dumps(
            {
                "status": "created",
                "attempts": [{"started": "a"}, {"started": "b"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "append attempt")

    result = CommittedRangeProvenanceGuard(tmp_path).check(
        "recovery_metadata.json",
        append_fields=("attempts",),
        base_ref="HEAD~1",
    )

    assert result.violations == []


def test_committed_range_provenance_guard_rejects_append_field_rewrite(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    provenance = tmp_path / "recovery_metadata.json"
    provenance.write_text(
        json.dumps({"status": "created", "attempts": [{"started": "a"}]}) + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "base")
    provenance.write_text(
        json.dumps({"status": "created", "attempts": [{"started": "rewritten"}]}) + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "rewrite attempt")

    result = CommittedRangeProvenanceGuard(tmp_path).check(
        "recovery_metadata.json",
        append_fields=("attempts",),
        base_ref="HEAD~1",
    )

    assert any("mutated existing entries" in violation for violation in result.violations)


def test_meta_lifecycle_audit_uses_committed_range_base_ref(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "artifact_kind_registry.yaml").write_text(
        """
- pattern: "provenance.json"
  kind: HISTORICAL_PROVENANCE
  rationale: "unit test provenance"
  append_fields: []
""",
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({"archive_sha256": "aaa", "status": "created"}) + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "base")
    provenance.write_text(
        json.dumps({"archive_sha256": "bbb", "status": "created"}) + "\n",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "mutate provenance")

    violations = run_meta_lifecycle_audit(tmp_path, base_ref="HEAD~1")

    assert any("archive_sha256" in violation for violation in violations)


def test_meta_lifecycle_audit_skips_untracked_historical_provenance(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "artifact_kind_registry.yaml").write_text(
        """
- pattern: "experiments/results/**/provenance.json"
  kind: HISTORICAL_PROVENANCE
  rationale: "unit test provenance"
  append_fields: []
""",
        encoding="utf-8",
    )
    _commit_all(tmp_path, "registry only")
    provenance = tmp_path / "experiments" / "results" / "scratch" / "provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        json.dumps({"archive_sha256": "untracked", "status": "scratch"}) + "\n",
        encoding="utf-8",
    )

    violations = run_meta_lifecycle_audit(tmp_path, base_ref="HEAD")

    assert violations == []


def test_meta_lifecycle_audit_flags_tracked_live_state(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "artifact_kind_registry.yaml").write_text(
        """
- pattern: ".omx/state/runtime.lock"
  kind: LIVE_STATE
  rationale: "unit test live lock"
""",
        encoding="utf-8",
    )
    (state_dir / "runtime.lock").write_text("locked\n", encoding="utf-8")
    _commit_all(tmp_path, "tracked live state")

    violations = run_meta_lifecycle_audit(tmp_path)

    assert any("LIVE_STATE file" in violation for violation in violations)


def test_meta_lifecycle_audit_honors_path_filter(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "artifact_kind_registry.yaml").write_text(
        """
- pattern: "reports/raw/**/*.json"
  kind: DERIVED_OUTPUT
  rationale: "unit test derived output"
""",
        encoding="utf-8",
    )
    legacy = tmp_path / "reports" / "raw" / "legacy" / "summary.json"
    current = tmp_path / "reports" / "raw" / "current" / "summary.json"
    legacy.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    legacy.write_text('{"score": 1}\n', encoding="utf-8")
    current.write_text('{"score": 2}\n', encoding="utf-8")
    _commit_all(tmp_path, "tracked derived outputs")

    violations = run_meta_lifecycle_audit(
        tmp_path,
        path_filter={"reports/raw/current/summary.json"},
    )

    assert len(violations) == 1
    assert "reports/raw/current/summary.json" in violations[0]


def test_artifact_lifecycle_changed_paths_falls_back_to_head_commit(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    first = tmp_path / "first.txt"
    first.write_text("first\n", encoding="utf-8")
    _commit_all(tmp_path, "first")
    second = tmp_path / "second.txt"
    second.write_text("second\n", encoding="utf-8")
    _commit_all(tmp_path, "second")

    paths = _artifact_lifecycle_changed_paths(tmp_path, "HEAD")

    assert paths == {"second.txt"}
