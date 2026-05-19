#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check canonical TAC/comma-lab terminology in public docs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

CANONICAL_FILES = (
    "README.md",
    "docs/contest_compliance_authority.md",
    "docs/terminology_and_boundaries.md",
    "src/tac/README.md",
    "src/comma_lab/README.md",
    "src/tac/__init__.py",
    "src/comma_lab/__init__.py",
    "pyproject.toml",
)

FORBIDDEN_TAC_DEFINITION_RE = re.compile(
    r"`?\b(?:TAC|tac)\b`?\s*(?:=|:|means|stands\s+for|is)\s+"
    r"(?:\*\*)?Task[- ]Aware Codec(?:\*\*)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    """One terminology violation."""

    path: str
    message: str
    line: int | None = None

    def render(self) -> str:
        if self.line is None:
            return f"{self.path}: {self.message}"
        return f"{self.path}:{self.line}: {self.message}"


def _read_required(root: Path, relpath: str, findings: list[Finding]) -> str:
    path = root / relpath
    if not path.is_file():
        findings.append(Finding(relpath, "required terminology file is missing"))
        return ""
    return path.read_text(encoding="utf-8")


def _require_contains(
    findings: list[Finding],
    *,
    relpath: str,
    text: str,
    needle: str,
    rationale: str,
) -> None:
    if needle not in text:
        findings.append(Finding(relpath, f"missing {rationale}: {needle!r}"))


def _forbidden_definition_findings(relpath: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if FORBIDDEN_TAC_DEFINITION_RE.search(line):
            findings.append(
                Finding(
                    relpath,
                    "TAC must expand to Task-Aware Compression, not Task-Aware Codec",
                    lineno,
                )
            )
    return findings


def check_repo(root: Path) -> list[Finding]:
    """Return terminology findings for a repository root."""

    root = root.resolve()
    findings: list[Finding] = []
    texts = {relpath: _read_required(root, relpath, findings) for relpath in CANONICAL_FILES}
    for relpath, text in texts.items():
        findings.extend(_forbidden_definition_findings(relpath, text))

    root_readme = texts["README.md"]
    _require_contains(
        findings,
        relpath="README.md",
        text=root_readme,
        needle="`tac` means **Task-Aware Compression**",
        rationale="canonical TAC expansion",
    )
    for needle in (
        "src/tac/README.md",
        "src/comma_lab/README.md",
        "docs/terminology_and_boundaries.md",
        "docs/contest_compliance_authority.md",
    ):
        _require_contains(
            findings,
            relpath="README.md",
            text=root_readme,
            needle=needle,
            rationale="public docs pointer",
        )

    tac_readme = texts["src/tac/README.md"]
    for needle, rationale in (
        ("# tac - Task-Aware Compression", "package title"),
        ("Video coding for machines", "VCM terminology"),
        ("Feature coding for machines", "FCM terminology"),
        ("Use **compression** for the project and research program", "codec/compression distinction"),
        ("procedural generation from archive-contained seeds or weights", "procedural generation boundary"),
        ("docs/contest_compliance_authority.md", "contest authority pointer"),
    ):
        _require_contains(
            findings,
            relpath="src/tac/README.md",
            text=tac_readme,
            needle=needle,
            rationale=rationale,
        )

    tac_init = texts["src/tac/__init__.py"]
    for needle, rationale in (
        ("Task-Aware Compression (tac)", "package docstring TAC expansion"),
        ("compression optimized against\ndownstream machine-perception tasks", "package scope"),
        ('"codec" names a concrete encoder/decoder or wire\nformat', "codec/package distinction"),
    ):
        _require_contains(
            findings,
            relpath="src/tac/__init__.py",
            text=tac_init,
            needle=needle,
            rationale=rationale,
        )

    comma_readme = texts["src/comma_lab/README.md"]
    for needle, rationale in (
        ("It is intentionally not the compression engine.", "comma_lab boundary"),
        ("`tac`: Task-Aware Compression library", "related package pointer"),
        ("`comma_lab.task_codec`", "legacy task_codec boundary"),
        ("Never create score authority from `comma_lab` alone", "authority boundary"),
        ("docs/contest_compliance_authority.md", "contest authority pointer"),
    ):
        _require_contains(
            findings,
            relpath="src/comma_lab/README.md",
            text=comma_readme,
            needle=needle,
            rationale=rationale,
        )

    comma_init = texts["src/comma_lab/__init__.py"]
    for needle, rationale in (
        ("repository operations for Task-Aware Compression research", "operations-layer scope"),
        ("lossless_review_tracker", "state/review export"),
        ("state_sync", "state/review export"),
    ):
        _require_contains(
            findings,
            relpath="src/comma_lab/__init__.py",
            text=comma_init,
            needle=needle,
            rationale=rationale,
        )

    terminology = texts["docs/terminology_and_boundaries.md"]
    for needle, rationale in (
        ("This document is the canonical naming and package-boundary reference", "authority declaration"),
        ('Never expand TAC as "Task-Aware Codec."', "negative expansion rule"),
        ("`comma_lab.task_codec` is a legacy compatibility namespace", "legacy task_codec boundary"),
        ("Contest Compliance Boundary", "contest compliance section"),
    ):
        _require_contains(
            findings,
            relpath="docs/terminology_and_boundaries.md",
            text=terminology,
            needle=needle,
            rationale=rationale,
        )
    for needle in (
        "Procedural generation from an archive-contained seed",
        "Constants in `inflate.py` may describe how to decode a charged payload",
        "Package Ownership",
        "docs/contest_compliance_authority.md",
    ):
        _require_contains(
            findings,
            relpath="docs/terminology_and_boundaries.md",
            text=terminology,
            needle=needle,
            rationale="terminology source-of-truth content",
        )

    compliance = texts["docs/contest_compliance_authority.md"]
    for needle, rationale in (
        ("# Contest Compliance Authority", "compliance authority title"),
        ("Authority Ladder", "source hierarchy"),
        ("Public PR Precedents", "public precedent section"),
        ("archive_seeded", "procedural archive-seeded mode"),
        ("runtime_constant", "procedural runtime-constant mode"),
        ("score-bearing information must be charged through `archive.zip`", "payload relocation guard"),
        ("#35 tensor_inversion", "scorer inflate precedent"),
        ("#68 loophole_v2", "script-payload loophole precedent"),
        ("#78 qzs3_script_payload_r147", "withdrawn payload-relocation precedent"),
    ):
        _require_contains(
            findings,
            relpath="docs/contest_compliance_authority.md",
            text=compliance,
            needle=needle,
            rationale=rationale,
        )

    pyproject = texts["pyproject.toml"]
    for needle, rationale in (
        ('description = "Task-Aware Compression:', "project description"),
        ('"task-aware-compression"', "PyPI keyword"),
        ('"video-coding-for-machines"', "VCM PyPI keyword"),
    ):
        _require_contains(
            findings,
            relpath="pyproject.toml",
            text=pyproject,
            needle=needle,
            rationale=rationale,
        )

    return findings


def _payload(findings: list[Finding]) -> dict[str, object]:
    return {
        "schema": "tac_terminology_check_v1",
        "ok": not findings,
        "finding_count": len(findings),
        "findings": [{"path": finding.path, "line": finding.line, "message": finding.message} for finding in findings],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable findings.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when findings exist.")
    args = parser.parse_args(argv)

    findings = check_repo(args.repo_root)
    if args.json:
        print(json.dumps(_payload(findings), indent=2, sort_keys=True))
    elif findings:
        print("TAC terminology check failed:")
        for finding in findings:
            print(f"  - {finding.render()}")
    else:
        print("TAC terminology check passed.")

    return 1 if args.strict and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
