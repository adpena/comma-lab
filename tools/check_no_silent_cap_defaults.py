#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refuse newly introduced silent iteration-cap defaults.

Caps are legitimate.  The defect is a cap-shaped default in an executable that
cannot report why the run stopped.  This checker derives the denominator from
Python source instead of from a memo: every argparse integer default whose flag
looks like an iteration/step/pass cap is a site.  A site is "silent" when its
file has no stop-reporting vocabulary such as ``stop_reason``.

Use ``--baseline`` to fail only NEW silent sites while still reporting the full
current population.  The baseline is a JSON report emitted by this tool.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

CAP_FLAG_RE = re.compile(
    r"^--(?:(?:max|num|n)-)?(?:steps?|iters?|iterations?|passes?|relin-bound)$"
)
STOP_REPORT_MARKERS = (
    "CapStopReceipt",
    "build_cap_stop_receipt",
    "stop_reason",
    "stop_reasons",
    "stop_status",
    "early_stop_reason",
    "termination_census",
    "convergence_verdict",
    "convergence_decidable",
)
DEFAULT_ROOTS = ("experiments", "tools", "src/tac")
SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "site-packages",
    "venv",
}


@dataclass(frozen=True)
class CapDefaultSite:
    path: str
    line: int
    scope: str
    flag: str
    default: int
    reports_stop_reason: bool

    @property
    def key(self) -> str:
        return f"{self.path}|{self.scope}|{self.flag}|{self.default}"

    @property
    def status(self) -> str:
        return "reports_stop_reason" if self.reports_stop_reason else "silent_cap_default"


def _numeric_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -int(node.operand.value)
    return None


def _has_stop_reporting(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in STOP_REPORT_MARKERS:
            return True
        if isinstance(node, ast.Attribute) and node.attr in STOP_REPORT_MARKERS:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(marker in node.value for marker in STOP_REPORT_MARKERS):
                return True
    return False


class _CapVisitor(ast.NodeVisitor):
    def __init__(self, path: str, reports_stop_reason: bool) -> None:
        self.path = path
        self.reports_stop_reason = reports_stop_reason
        self.scope_stack: list[str] = []
        self.sites: list[CapDefaultSite] = []

    @property
    def scope(self) -> str:
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            cap_flags = [flag for flag in flags if CAP_FLAG_RE.match(flag)]
            if cap_flags:
                default = None
                for kw in node.keywords:
                    if kw.arg == "default":
                        default = _numeric_int(kw.value)
                        break
                if default is not None and default > 0:
                    for flag in cap_flags:
                        self.sites.append(
                            CapDefaultSite(
                                path=self.path,
                                line=int(getattr(node, "lineno", 0)),
                                scope=self.scope,
                                flag=flag,
                                default=default,
                                reports_stop_reason=self.reports_stop_reason,
                            )
                        )
        self.generic_visit(node)


def _iter_python_files(roots: list[Path], *, include_tests: bool) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel_parts = path.relative_to(REPO).parts if path.is_relative_to(REPO) else path.parts
            if any(part in SKIP_DIR_NAMES for part in rel_parts):
                continue
            if len(rel_parts) >= 2 and rel_parts[:2] == ("experiments", "results"):
                continue
            if not include_tests and ("tests" in rel_parts or path.name.startswith("test_")):
                continue
            files.append(path)
    return sorted(files)


def scan_cap_defaults(roots: list[Path] | None = None, *, include_tests: bool = False) -> dict[str, Any]:
    roots = roots or [REPO / r for r in DEFAULT_ROOTS]
    sites: list[CapDefaultSite] = []
    parse_errors: list[dict[str, str]] = []
    files = _iter_python_files(roots, include_tests=include_tests)
    for path in files:
        rel = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append({"path": rel, "error": str(exc)})
            continue
        visitor = _CapVisitor(rel, _has_stop_reporting(tree))
        visitor.visit(tree)
        sites.extend(visitor.sites)

    silent = [site for site in sites if not site.reports_stop_reason]
    return {
        "schema": "silent_cap_default_scan.v1",
        "roots": [str(r.relative_to(REPO)) if r.is_relative_to(REPO) else str(r) for r in roots],
        "include_tests": include_tests,
        "files_scanned": len(files),
        "parse_errors": parse_errors,
        "cap_default_sites": len(sites),
        "silent_cap_default_sites": len(silent),
        "allowed_silent_cap_default_keys": [site.key for site in silent],
        "sites": [
            {**asdict(site), "key": site.key, "status": site.status}
            for site in sites
        ],
    }


def _load_allowed_keys(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "allowed_silent_cap_default_keys" in obj:
        return {str(k) for k in obj["allowed_silent_cap_default_keys"]}
    if "sites" in obj:
        return {
            str(site["key"])
            for site in obj["sites"]
            if site.get("status") == "silent_cap_default"
        }
    raise ValueError(f"baseline has no allowed silent-cap keys: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path,
                    help="roots to scan; defaults to experiments tools src/tac")
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--baseline", type=Path,
                    help="existing scan report whose silent keys are grandfathered")
    ap.add_argument("--write-baseline", type=Path,
                    help="write the current scan JSON here")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    roots = [p if p.is_absolute() else REPO / p for p in args.roots] if args.roots else None
    report = scan_cap_defaults(roots, include_tests=args.include_tests)

    new_silent: list[str] = []
    if args.baseline is not None:
        allowed = _load_allowed_keys(args.baseline)
        current = {
            site["key"]
            for site in report["sites"]
            if site["status"] == "silent_cap_default"
        }
        new_silent = sorted(current - allowed)
        report["baseline"] = str(args.baseline)
        report["new_silent_cap_default_sites"] = len(new_silent)
        report["new_silent_cap_default_keys"] = new_silent

    if args.write_baseline is not None:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(
            "silent-cap scan: files={files_scanned} cap_defaults={cap_default_sites} "
            "silent={silent_cap_default_sites}".format(**report)
        )
        if args.baseline is not None:
            print(f"new_silent={len(new_silent)}")
            for key in new_silent:
                print(f"  NEW {key}")

    if report["parse_errors"]:
        return 2
    if new_silent:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
