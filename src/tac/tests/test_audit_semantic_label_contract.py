"""Tests for the active semantic-label contract audit tool."""

from __future__ import annotations

from pathlib import Path

from tools.audit_semantic_label_contract import (
    ADVISORY_REL_PATHS,
    CORE_REL_PATHS,
    audit_semantic_label_contract,
    main,
)


def test_semantic_label_contract_audit_passes_current_checkout() -> None:
    result = audit_semantic_label_contract()
    assert result.ok, result


def test_semantic_label_contract_audit_reports_stale_phrase(tmp_path: Path) -> None:
    path = tmp_path / "stale.py"
    path.write_text("# Only 5 semantic classes: road, lane, vehicle, sky, background\n")
    result = audit_semantic_label_contract(repo_root=tmp_path, rel_paths=("stale.py",), advisory_rel_paths=())
    assert not result.ok
    assert len(result.blocking_findings) == 1
    assert result.blocking_findings[0].path == "stale.py"


def test_semantic_label_contract_audit_reports_regex_patterns(tmp_path: Path) -> None:
    path = tmp_path / "stale.py"
    path.write_text(
        "\n".join(
            (
                "# class 4: sky",
                "# class 2 = vehicle",
                "# default weights are road,lane,bg,vehicle,sky",
            )
        )
    )
    result = audit_semantic_label_contract(repo_root=tmp_path, rel_paths=("stale.py",), advisory_rel_paths=())
    assert not result.ok
    assert {finding.pattern for finding in result.blocking_findings} == {
        "class 4 sky",
        "class 2 vehicle",
        "legacy class order",
    }


def test_semantic_label_contract_audit_keeps_advisory_nonblocking(tmp_path: Path) -> None:
    core = tmp_path / "core.py"
    advisory = tmp_path / "advisory.py"
    core.write_text("# canonical class names live elsewhere\n")
    advisory.write_text("# class 4: sky\n")

    result = audit_semantic_label_contract(
        repo_root=tmp_path,
        rel_paths=("core.py",),
        advisory_rel_paths=("advisory.py",),
    )

    assert result.ok
    assert not result.blocking_findings
    assert len(result.advisory_findings) == 1
    assert result.advisory_findings[0].severity == "advisory"


def test_semantic_label_contract_cli_json() -> None:
    assert main(["--format", "json"]) == 0


def test_semantic_label_contract_cli_can_fail_on_advisory(tmp_path: Path) -> None:
    for rel_path in CORE_REL_PATHS + ADVISORY_REL_PATHS:
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# canonical class names live elsewhere\n")
    (tmp_path / ADVISORY_REL_PATHS[0]).write_text("# class 4: sky\n")

    assert (
        main(
            [
                "--repo-root",
                str(tmp_path),
                "--format",
                "json",
                "--fail-on-advisory",
            ]
        )
        == 2
    )
