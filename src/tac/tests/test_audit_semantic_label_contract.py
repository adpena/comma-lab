"""Tests for the active semantic-label contract audit tool."""

from __future__ import annotations

from pathlib import Path

from tools.audit_semantic_label_contract import (
    audit_semantic_label_contract,
    main,
)


def test_semantic_label_contract_audit_passes_current_checkout() -> None:
    result = audit_semantic_label_contract()
    assert result.ok, result


def test_semantic_label_contract_audit_reports_stale_phrase(tmp_path: Path) -> None:
    path = tmp_path / "stale.py"
    path.write_text("# Only 5 semantic classes: road, lane, vehicle, sky, background\n")
    result = audit_semantic_label_contract(repo_root=tmp_path, rel_paths=("stale.py",))
    assert not result.ok
    assert len(result.findings) == 1
    assert result.findings[0].path == "stale.py"


def test_semantic_label_contract_cli_json() -> None:
    assert main(["--format", "json"]) == 0
