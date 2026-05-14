#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PCC10 — Anti-arbitrariness scanner for prediction-logic magic numbers.

Council Q4 prescription. Every numeric literal bound to an identifier that
implies prediction logic (`predict|score|band|delta|threshold|gain|loss_coef|
rate_coef|...`) MUST carry one of these provenance tags on the same line OR
the immediately preceding line:

    # [contest-defined]              — fixed by upstream/evaluate.py contest rules
    # [calibration:<source>]         — derived from a calibration anchor or fit
    # [empirical:<artifact-path>]    — measured via real artifact (non-trivial)
    # [heuristic:<reason>]           — judgment call with stated rationale

Untagged numeric literals are flagged. The user mandate 2026-05-05 ("no
arbitrariness") drives this — magic constants ship into production via
prediction logic with no evidence trail.

Scoped to:
    experiments/build_*.py / experiments/sweep_*.py / experiments/repack_*.py
    tools/apogee_intN_*.py / tools/predispatch_sanity.py
    src/tac/predictor/*.py

Excluded:
    Test files (already covered by acceptance asserts)
    Generated artifacts (build_metadata.json — JSON not Python)
    Constants flagged contest-defined in CLAUDE.md (already documented)

Exit codes:
    0    no untagged literals
    1    untagged literals found (only when --strict)
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

SCANNED_GLOBS = (
    "experiments/build_*.py",
    "experiments/sweep_*.py",
    "experiments/repack_*.py",
    "tools/apogee_intN_*.py",
    "tools/predispatch_sanity.py",
    "src/tac/predictor/*.py",
)

# Identifier substrings that imply prediction-logic context.
PREDICTION_NAME_PATTERNS = (
    re.compile(r"predict", re.IGNORECASE),
    re.compile(r"score", re.IGNORECASE),
    re.compile(r"band", re.IGNORECASE),
    re.compile(r"^delta", re.IGNORECASE),  # avoids "deltas" being all-paths
    re.compile(r"threshold", re.IGNORECASE),
    re.compile(r"gain", re.IGNORECASE),
    re.compile(r"loss_coef", re.IGNORECASE),
    re.compile(r"rate_coef", re.IGNORECASE),
    re.compile(r"calibration", re.IGNORECASE),
    re.compile(r"sensitivity", re.IGNORECASE),
    re.compile(r"_pct", re.IGNORECASE),
    re.compile(r"_bits", re.IGNORECASE),
    re.compile(r"_weight", re.IGNORECASE),
)

PROVENANCE_TAG_PATTERN = re.compile(
    r"\[\s*(?:contest-defined|calibration\s*:|empirical\s*:|heuristic\s*:|inherited\s*:|external\s*:)",
    re.IGNORECASE,
)

# Whitelist: numeric literals that don't need tags.
# Whitelisted: 0, 1, -1, 2 (loop indices, default counts, common shapes).
WHITELIST_LITERALS = {0, 1, -1, 2, 0.0, 1.0, -1.0, 2.0}


@dataclass(frozen=True)
class UntaggedLiteral:
    path: str
    lineno: int
    target_name: str
    literal_value: str
    line_text: str


def _name_implies_prediction(name: str) -> bool:
    return any(p.search(name) for p in PREDICTION_NAME_PATTERNS)


def _line_or_above_has_tag(lines: list[str], lineno: int) -> bool:
    """Check the same line and the immediately preceding line for a provenance tag."""
    indices = [lineno - 1, lineno - 2]
    for idx in indices:
        if 0 <= idx < len(lines):
            if PROVENANCE_TAG_PATTERN.search(lines[idx]):
                return True
    return False


def _ast_extract_target_names(target: ast.AST) -> list[str]:
    """Extract target names from an Assign target (Name or Tuple of Names)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_ast_extract_target_names(elt))
        return names
    if isinstance(target, ast.Attribute):
        return [target.attr]
    return []


def _scan_file(path: Path, lines: list[str]) -> list[UntaggedLiteral]:
    """Walk AST; for any assignment of a numeric literal to a prediction-named
    target, require a provenance tag on the same line or immediately above."""
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return []
    findings: list[UntaggedLiteral] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        # Get assignment target names + value
        if isinstance(node, ast.Assign):
            target_names: list[str] = []
            for t in node.targets:
                target_names.extend(_ast_extract_target_names(t))
            value = node.value
        else:  # ast.AnnAssign
            target_names = _ast_extract_target_names(node.target)
            value = node.value
            if value is None:
                continue
        if not target_names:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, (int, float)):
            continue
        if value.value in WHITELIST_LITERALS:
            continue
        # Filter by name relevance
        relevant = [n for n in target_names if _name_implies_prediction(n)]
        if not relevant:
            continue
        # Check provenance tag
        lineno = node.lineno
        if _line_or_above_has_tag(lines, lineno):
            continue
        line_text = lines[lineno - 1] if 0 <= lineno - 1 < len(lines) else ""
        findings.append(UntaggedLiteral(
            path=str(path),
            lineno=lineno,
            target_name=relevant[0],
            literal_value=repr(value.value),
            line_text=line_text.strip()[:140],
        ))
    return findings


def scan(repo_root: Path) -> list[UntaggedLiteral]:
    findings: list[UntaggedLiteral] = []
    for glob in SCANNED_GLOBS:
        for path in sorted(repo_root.glob(glob)):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lines = text.splitlines()
            findings.extend(_scan_file(path, lines))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    findings = scan(args.repo_root)

    if args.json:
        import json
        print(json.dumps({
            "schema": "calibration_provenance_v1",
            "findings": [
                {"path": f.path, "lineno": f.lineno, "target_name": f.target_name,
                 "literal_value": f.literal_value, "line_text": f.line_text}
                for f in findings
            ],
            "count": len(findings),
        }, indent=2))
    else:
        for f in findings:
            print(f"{f.path}:{f.lineno}: {f.target_name} = {f.literal_value}", file=sys.stderr)
            print(f"  → {f.line_text}", file=sys.stderr)
            print("  → missing provenance tag (one of [contest-defined], [calibration:<src>], "
                  "[empirical:<artifact>], [heuristic:<reason>])", file=sys.stderr)
        if findings:
            print(f"\n[PCC10] {len(findings)} untagged numeric literal(s) in prediction logic "
                  f"across {len({f.path for f in findings})} file(s).", file=sys.stderr)
        else:
            print("[PCC10] OK: 0 untagged prediction-logic literals", file=sys.stderr)

    if findings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
