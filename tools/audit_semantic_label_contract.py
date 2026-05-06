#!/usr/bin/env python3
"""Audit the active semantic-label contract and selected live callsites."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac import camera  # noqa: E402
from tac.mask_grayscale_lut import CLASS_TO_GRAY  # noqa: E402
from tac.semantic_label_contract import (  # noqa: E402
    CONTEST_SEGNET_CLASS_NAME_TUPLE,
    CONTEST_SEGNET_CLASS_NAMES,
    NUM_CONTEST_SEGNET_CLASSES,
    SELFCOMP_CLASS_TO_GRAY,
    validate_contest_class_table,
)
from tac.semantic_quantization import CLASS_NAMES, DEFAULT_CLASS_BITS  # noqa: E402


CORE_REL_PATHS: tuple[str, ...] = (
    "src/tac/semantic_label_contract.py",
    "src/tac/semantic_quantization.py",
    "src/tac/mask_grayscale_lut.py",
    "src/tac/camera.py",
    "src/tac/mask_prior.py",
    "src/tac/mini_scorer.py",
    "src/tac/depth_motion.py",
    "experiments/diagnose_nerv_geometry.py",
    "experiments/constrained_gen_research/inflate.py",
)

STALE_EXACT_PHRASES: tuple[str, ...] = (
    "road, lane, vehicle, sky, background",
    "0: \"background\"",
    "0 (background)",
    "class 0: background",
    "class 4: sky",
    "4 = vehicle",
    "4 = background",
)


@dataclass(frozen=True)
class AuditFinding:
    path: str
    line: int
    phrase: str
    text: str


@dataclass(frozen=True)
class AuditResult:
    contract_ok: bool
    findings: tuple[AuditFinding, ...]

    @property
    def ok(self) -> bool:
        return self.contract_ok and not self.findings


def _line_findings(path: Path, rel_path: str) -> list[AuditFinding]:
    text = path.read_text(encoding="utf-8")
    findings: list[AuditFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for phrase in STALE_EXACT_PHRASES:
            if phrase in line:
                findings.append(
                    AuditFinding(
                        path=rel_path,
                        line=line_no,
                        phrase=phrase,
                        text=line.strip(),
                    )
                )
    return findings


def audit_semantic_label_contract(
    repo_root: Path = REPO_ROOT,
    rel_paths: tuple[str, ...] = CORE_REL_PATHS,
) -> AuditResult:
    validate_contest_class_table()
    contract_ok = (
        camera.NUM_CLASSES == NUM_CONTEST_SEGNET_CLASSES
        and camera.SEGNET_CLASS_NAMES == CONTEST_SEGNET_CLASS_NAME_TUPLE
        and CLASS_TO_GRAY == SELFCOMP_CLASS_TO_GRAY
        and CLASS_NAMES == CONTEST_SEGNET_CLASS_NAMES
        and DEFAULT_CLASS_BITS[0] == 8
        and DEFAULT_CLASS_BITS[1] == 8
    )

    findings: list[AuditFinding] = []
    for rel_path in rel_paths:
        path = repo_root / rel_path
        if not path.exists():
            findings.append(
                AuditFinding(
                    path=rel_path,
                    line=0,
                    phrase="missing",
                    text="curated semantic-label audit path is missing",
                )
            )
            continue
        findings.extend(_line_findings(path, rel_path))

    return AuditResult(contract_ok=contract_ok, findings=tuple(findings))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_semantic_label_contract(repo_root=args.repo_root)
    if args.format == "json":
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "contract_ok": result.contract_ok,
                    "findings": [asdict(finding) for finding in result.findings],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif result.ok:
        print("semantic label contract audit: OK")
    else:
        print("semantic label contract audit: FAILED")
        if not result.contract_ok:
            print("- live contract exports disagree")
        for finding in result.findings:
            loc = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"- {loc}: stale phrase {finding.phrase!r}: {finding.text}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
