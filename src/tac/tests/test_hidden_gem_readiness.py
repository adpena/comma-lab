from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tac.hidden_gem_readiness import (
    DISPATCH_BLOCKER_REGISTRY_ONLY,
    audit_hidden_gems,
    hidden_gem_readiness_to_dict,
    readiness_payload,
    render_markdown,
)
from tac.hidden_gems import HiddenGemEntry

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "audit_hidden_gem_readiness.py"


def test_hidden_gem_readiness_classifies_missing_paths_and_hashes(tmp_path: Path) -> None:
    evidence = tmp_path / "docs" / "evidence.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("signal\n", encoding="utf-8")
    target = tmp_path / "src" / "tac" / "live_target.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")
    entry = _entry(
        status="ready_for_patch",
        evidence_paths=("docs/evidence.md", "docs/missing.md"),
        integration_targets=("src/tac/live_target.py", "src/tac/missing_target.py"),
    )

    (row,) = audit_hidden_gems(repo_root=tmp_path, entries=(entry,))
    payload = hidden_gem_readiness_to_dict(row)

    assert row.readiness_status == "blocked_missing_integration_targets"
    assert row.eligible_for_local_patch is False
    assert row.ready_for_exact_eval_dispatch is False
    assert row.missing_evidence_paths == ("docs/missing.md",)
    assert row.missing_integration_targets == ("src/tac/missing_target.py",)
    assert DISPATCH_BLOCKER_REGISTRY_ONLY in row.dispatch_blockers
    probe = next(item for item in payload["evidence"] if item["path"] == "docs/evidence.md")
    assert probe["exists"] is True
    assert probe["kind"] == "file"
    assert probe["bytes"] == len("signal\n")
    assert probe["sha256"] == hashlib.sha256(b"signal\n").hexdigest()


def test_hidden_gem_readiness_ready_for_local_patch_stays_dispatch_closed(tmp_path: Path) -> None:
    evidence = tmp_path / "docs" / "evidence.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("custody\n", encoding="utf-8")
    target = tmp_path / "src" / "tac" / "target.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 2\n", encoding="utf-8")
    entry = _entry(
        status="ready_for_patch",
        evidence_paths=("docs/evidence.md",),
        integration_targets=("src/tac/target.py",),
    )

    (row,) = audit_hidden_gems(repo_root=tmp_path, entries=(entry,))

    assert row.readiness_status == "ready_for_local_patch"
    assert row.eligible_for_local_patch is True
    assert row.ready_for_exact_eval_dispatch is False
    assert row.dispatch_blockers == (DISPATCH_BLOCKER_REGISTRY_ONLY,)


def test_hidden_gem_readiness_payload_summary_and_markdown_are_stable(tmp_path: Path) -> None:
    live = tmp_path / "src" / "tac" / "target.py"
    live.parent.mkdir(parents=True)
    live.write_text("VALUE = 3\n", encoding="utf-8")
    entry = _entry(
        status="prototype",
        evidence_paths=("docs/missing.md",),
        integration_targets=("src/tac/target.py",),
    )

    payload = readiness_payload(repo_root=tmp_path, entries=(entry,))
    markdown = render_markdown(audit_hidden_gems(repo_root=tmp_path, entries=(entry,)))

    assert payload["schema_version"] == 1
    assert payload["summary"]["entry_count"] == 1
    assert payload["summary"]["ready_for_exact_eval_dispatch_count"] == 0
    assert payload["summary"]["readiness_status_counts"] == {"blocked_missing_evidence": 1}
    assert markdown.startswith("# Hidden-Gem Readiness Audit\n")
    assert "| `test_hidden_gem` | `prototype` | `blocked_missing_evidence` |" in markdown


def test_audit_hidden_gem_readiness_cli_json_output() -> None:
    proc = _run_cli("--format", "json", "--status", "ready_for_patch")

    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["audit"] == "hidden_gem_readiness"
    assert payload["schema_version"] == 1
    assert payload["summary"]["entry_count"] > 0
    assert payload["summary"]["ready_for_exact_eval_dispatch_count"] == 0
    assert all(entry["ready_for_exact_eval_dispatch"] is False for entry in payload["entries"])


def _entry(
    *,
    status: str,
    evidence_paths: tuple[str, ...],
    integration_targets: tuple[str, ...],
) -> HiddenGemEntry:
    return HiddenGemEntry(
        key="test_hidden_gem",
        title="Test hidden gem",
        category="archive_packing",
        status=status,
        summary="Test row",
        evidence_paths=evidence_paths,
        integration_targets=integration_targets,
        next_patch="Patch the test row",
        contest_compliance_notes=("No dispatch from this planning row.",),
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
