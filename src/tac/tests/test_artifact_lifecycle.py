from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.artifact_lifecycle import (
    CommittedRangeProvenanceGuard,
    run_meta_lifecycle_audit,
)


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
