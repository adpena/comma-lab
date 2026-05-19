#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check canonical TAC/comma-lab terminology in public docs."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

CANONICAL_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "HANDOFF.md",
    "PROGRAM.md",
    "SYSTEM_MAP.md",
    "docs/README.md",
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
FORBIDDEN_TAC_PARENTHETICAL_RE = re.compile(
    r"(?:Task[- ]Aware Codec\s*\(\s*(?:TAC|tac)\s*\)|"
    r"\(\s*(?:TAC|tac)\s*\)\s*Task[- ]Aware Codec)",
    re.IGNORECASE,
)
AMBIGUOUS_TAC_HEADING_RE = re.compile(r"^#{1,6}\s+.*\bTAC\b")

STALE_PUBLIC_PHRASES: tuple[tuple[str, str], ...] = (
    (
        "Comma Lab Compression Challenge library",
        "describe `tac` as the Task-Aware Compression library",
    ),
    (
        "reusable codec / runtime library",
        "describe `tac` as the reusable Task-Aware Compression runtime-contract library",
    ),
    (
        "https://github.com/adpena/tac",
        "point public docs to src/tac/README.md or the current comma-lab package boundary instead of the stale tac-only repository",
    ),
    (
        "The frontier today is",
        "durable docs must point to reports/latest.md instead of hard-coding a live frontier",
    ),
    (
        "Today's public frontier",
        "durable docs must point to reports/latest.md instead of hard-coding a live frontier",
    ),
    (
        "current frontier state",
        "durable docs must say frontier snapshot or point to reports/latest.md",
    ),
    (
        "The current leader",
        "public docs must avoid hard-coded live-leader claims; point to reports/latest.md or mark as historical",
    ),
    (
        "**Current results:**",
        "public docs must mark result rows as historical or evidence-grade, not current by prose",
    ),
    (
        "current submitted Apogee",
        "paper docs must mark Apogee packet notes as historical or evidence-grade",
    ),
    (
        "architecture maps directly to comma's production data pipeline",
        "public docs must frame contest-to-production transfer as a hypothesis unless evidence-grade production validation exists",
    ),
    (
        "ArXiv writeup track",
        "public docs must avoid venue commitments; use paper/writeup draft language",
    ),
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
        if FORBIDDEN_TAC_DEFINITION_RE.search(line) or FORBIDDEN_TAC_PARENTHETICAL_RE.search(line):
            findings.append(
                Finding(
                    relpath,
                    "TAC must expand to Task-Aware Compression, not Task-Aware Codec",
                    lineno,
                )
            )
    return findings


def _public_terminology_scan_paths(root: Path) -> list[Path]:
    """Return public docs/package paths where stale TAC expansion must not hide."""

    paths: set[Path] = set()
    for relpath in (
        "README.md",
        "CONTRIBUTING.md",
        "HANDOFF.md",
        "PROGRAM.md",
        "SYSTEM_MAP.md",
        "CHANGELOG.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "src/tac/README.md",
        "src/tac/__init__.py",
        "src/comma_lab/README.md",
        "src/comma_lab/__init__.py",
    ):
        path = root / relpath
        if path.is_file():
            paths.add(path)
    docs = root / "docs"
    if docs.is_dir():
        for path in docs.rglob("*.md"):
            relpath = path.relative_to(root).as_posix()
            if relpath.startswith("docs/superpowers/"):
                continue
            paths.add(path)
    for path in _public_readme_paths(root):
        paths.add(path)
    return sorted(paths)


def _public_readme_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        try:
            rel_dir = current_path.relative_to(root).as_posix()
        except ValueError:
            continue
        dirs[:] = [
            dirname
            for dirname in dirs
            if not _skip_public_readme_scan(
                f"{dirname}/" if rel_dir == "." else f"{rel_dir}/{dirname}/"
            )
        ]
        if "README.md" not in files:
            continue
        path = current_path / "README.md"
        relpath = path.relative_to(root).as_posix()
        if not _skip_public_readme_scan(relpath):
            paths.append(path)
    return paths


def _skip_public_readme_scan(relpath: str) -> bool:
    return relpath.startswith(
        (
            ".git/",
            ".mypy_cache/",
            ".omx/",
            ".pytest_cache/",
            ".ruff_cache/",
            ".venv/",
            "build/",
            "data/",
            "docs/superpowers/",
            "experiments/results/",
            "htmlcov/",
            "reports/raw/",
            "upstream/",
            "vendored/",
        )
    )


def _ambiguous_tac_heading_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted((root / "docs").rglob("*.md")):
        relpath = path.relative_to(root).as_posix()
        if relpath == "docs/terminology_and_boundaries.md":
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if AMBIGUOUS_TAC_HEADING_RE.match(line) and "Task-Aware Compression" not in line:
                findings.append(
                    Finding(
                        relpath,
                        "TAC headings must expand to Task-Aware Compression (`tac`) or be rewritten",
                        lineno,
                    )
                )
    return findings


def _stale_public_phrase_findings(root: Path) -> list[Finding]:
    """Catch public-facing phrasing that weakens TAC/comma-lab authority."""

    findings: list[Finding] = []
    scan_roots = [
        root / "README.md",
        root / "CONTRIBUTING.md",
        root / "HANDOFF.md",
        root / "PROGRAM.md",
        root / "SYSTEM_MAP.md",
        root / "docs",
        root / "src" / "tac" / "README.md",
        root / "src" / "comma_lab" / "README.md",
    ]
    paths: list[Path] = []
    for scan_root in scan_roots:
        if scan_root.is_file():
            paths.append(scan_root)
        elif scan_root.is_dir():
            paths.extend(scan_root.rglob("*.md"))
    for path in sorted(set(paths)):
        relpath = path.relative_to(root).as_posix()
        if "docs/superpowers/" in relpath:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for phrase, replacement in STALE_PUBLIC_PHRASES:
                if phrase in line:
                    findings.append(
                        Finding(
                            relpath,
                            f"stale public wording {phrase!r}; {replacement}",
                            lineno,
                        )
                    )
    return findings


def check_repo(root: Path) -> list[Finding]:
    """Return terminology findings for a repository root."""

    root = root.resolve()
    findings: list[Finding] = []
    texts = {relpath: _read_required(root, relpath, findings) for relpath in CANONICAL_FILES}
    public_texts = dict(texts)
    for path in _public_terminology_scan_paths(root):
        relpath = path.relative_to(root).as_posix()
        public_texts.setdefault(relpath, path.read_text(encoding="utf-8"))
    for relpath, text in public_texts.items():
        findings.extend(_forbidden_definition_findings(relpath, text))
    findings.extend(_ambiguous_tac_heading_findings(root))
    findings.extend(_stale_public_phrase_findings(root))

    root_readme = texts["README.md"]
    _require_contains(
        findings,
        relpath="README.md",
        text=root_readme,
        needle="# comma-lab",
        rationale="canonical public repository title",
    )
    _require_contains(
        findings,
        relpath="README.md",
        text=root_readme,
        needle="`tac` means **Task-Aware Compression**",
        rationale="canonical TAC expansion",
    )
    _require_contains(
        findings,
        relpath="README.md",
        text=root_readme,
        needle="The local checkout may still be named `pact`",
        rationale="pact alias containment",
    )
    _require_contains(
        findings,
        relpath="README.md",
        text=root_readme,
        needle="This is an active research and engineering repo.",
        rationale="active-repo status",
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

    contributing = texts["CONTRIBUTING.md"]
    for needle, rationale in (
        ("`tac` means Task-Aware Compression", "contributing TAC expansion"),
        ("Use `codec` only for concrete encoders", "contributing codec/compression distinction"),
        ("`comma_lab` owns lab operations", "contributing comma_lab boundary"),
        ("src/tac/ src/comma_lab/", "contributing quality checks include both packages"),
        ("docs/terminology_and_boundaries.md", "contributing terminology pointer"),
        ("docs/contest_compliance_authority.md", "contributing contest authority pointer"),
    ):
        _require_contains(
            findings,
            relpath="CONTRIBUTING.md",
            text=contributing,
            needle=needle,
            rationale=rationale,
        )

    docs_readme = texts["docs/README.md"]
    for needle, rationale in (
        ("# Documentation Index", "docs index title"),
        ("Start with the files that describe the current public repository contract", "current docs routing"),
        ("Historical And Internal Plans", "historical docs routing"),
        ("docs/superpowers/", "historical superpowers routing"),
        ("tools/check_tac_terminology.py --strict", "terminology guard command"),
    ):
        _require_contains(
            findings,
            relpath="docs/README.md",
            text=docs_readme,
            needle=needle,
            rationale=rationale,
        )

    handoff = texts["HANDOFF.md"]
    for needle, rationale in (
        ("`tac` means Task-Aware Compression", "operator handoff TAC expansion"),
        ("`comma_lab` is the lab\noperations", "operator handoff comma_lab boundary"),
        ("docs/terminology_and_boundaries.md", "operator handoff terminology pointer"),
        (
            "100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * archive_bytes /\n37,545,489",
            "official score formula",
        ),
    ):
        _require_contains(
            findings,
            relpath="HANDOFF.md",
            text=handoff,
            needle=needle,
            rationale=rationale,
        )

    program = texts["PROGRAM.md"]
    for needle, rationale in (
        ("`tac` means Task-Aware Compression", "program TAC expansion"),
        ("A codec is a concrete encoder/decoder or wire format", "program codec/compression distinction"),
        ("`comma_lab` owns lab operations", "program comma_lab boundary"),
        ("docs/terminology_and_boundaries.md", "program terminology pointer"),
    ):
        _require_contains(
            findings,
            relpath="PROGRAM.md",
            text=program,
            needle=needle,
            rationale=rationale,
        )

    system_map = texts["SYSTEM_MAP.md"]
    for needle, rationale in (
        ("# The canonical Task-Aware Compression", "system-map TAC expansion"),
        ("# library and compression engine", "system-map compression engine boundary"),
    ):
        _require_contains(
            findings,
            relpath="SYSTEM_MAP.md",
            text=system_map,
            needle=needle,
            rationale=rationale,
        )

    tac_readme = texts["src/tac/README.md"]
    for needle, rationale in (
        ("# tac - Task-Aware Compression", "package title"),
        ("`TAC` is a repository/package acronym", "repository-local acronym boundary"),
        ("Video coding for machines", "VCM terminology"),
        ("Feature coding for machines", "FCM terminology"),
        ("Semantic communication", "adjacent semantic/goal-oriented communication framing"),
        ("Use **compression** for the project and research program", "codec/compression distinction"),
        ("procedural generation from archive-contained seeds or charged archive", "procedural generation boundary"),
        ("weight-derived promotion paths", "procedural weight-derived authority boundary"),
        ("procedural-seed authority packets", "procedural authority packet boundary"),
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
        ("Authority Model", "terminology authority model"),
        ("`TAC` itself is a repository-local acronym", "repository-local acronym rule"),
        ("`comma_lab.task_codec` is a legacy compatibility namespace", "legacy task_codec boundary"),
        ("Contest Compliance Boundary", "contest compliance section"),
        ("build_procedural_seed_authority_packet", "procedural authority packet helper"),
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
        "derived from already charged archive bytes",
        "Constants in `inflate.py` may describe how to decode a charged payload",
        "archive-seeded and weight-derived versions are the",
        "canonical promotion paths",
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
        ("weight_derived", "procedural weight-derived mode"),
        ("runtime_constant", "procedural runtime-constant mode"),
        ("score-bearing information must be charged", "payload relocation guard"),
        ("through `archive.zip`", "archive charging surface"),
        ("How To Establish Authority", "procedural authority packet protocol"),
        ("build_procedural_seed_authority_packet", "procedural authority packet helper"),
        ("#35 tensor_inversion", "scorer inflate precedent"),
        ("#68 loophole_v2", "script-payload loophole precedent"),
        ("#78 qzs3_script_payload_r147", "withdrawn payload-relocation precedent"),
        ("Task-Aware Compression (`tac`) design path", "expanded procedural generation TAC phrasing"),
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
        ('"task-oriented-compression"', "task-oriented compression PyPI keyword"),
        ('"coding-for-machines"', "coding for machines PyPI keyword"),
        ('"video-coding-for-machines"', "VCM PyPI keyword"),
        ('"feature-coding-for-machines"', "FCM PyPI keyword"),
        ('comma_lab = ["py.typed"]', "comma_lab typed package marker"),
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
