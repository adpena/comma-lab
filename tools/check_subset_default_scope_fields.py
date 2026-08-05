#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit subset defaults whose emitted verdicts omit subset scope fields.

This is the repo-wide, warn-only sister to :mod:`tac.subset_selection_gate`.
The hook guard refuses newly added ``pairs[:n_pairs]``-style prefixes.  This
scanner answers the SS1 question across the live source tree: if a tool defaults
to an under-sampled pair/frame subset and emits a verdict or receipt, does that
output carry enough scope to prevent a prefix/subset row from being read as
population truth?

The required shape is intentionally small:

* ``n`` / population count,
* a selection mode (prefix, stratified, seeded-random, etc.),
* an axis-bias or subset/population caveat.

Missing any of those is reported as ``silent_verdict_subset_default``.  Sites in
files that do not emit verdict-like output are still inventoried as
``dormant_no_verdict`` rather than treated as clean.
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

PAIR_POPULATION = 600
FRAME_POPULATION = 1200

SUBSET_FLAG_RE = re.compile(
    r"^--(?:(?:verdict|sample|batch|num|n|limit)-)?(?:pairs?|frames?|pair-count|frame-count)$"
)

PAIR_COUNT_NAMES = {
    "n",
    "n_pairs",
    "num_pairs",
    "npairs",
    "sample_pairs",
    "verdict_pairs",
    "n_verdict_pairs",
    "pair_count",
    "limit_pairs",
    "max_pairs",
    "n_frames",
    "num_frames",
    "nframes",
    "frame_count",
    "limit_frames",
    "max_frames",
}

VERDICT_MARKERS = (
    "verdict",
    "receipt",
    "score",
    "d_seg",
    "d_pose",
    "delta_s",
    "delta score",
    "frontier",
    "pointer",
    "eta",
)
SINK_NAMES = {
    "dump",
    "dumps",
    "print",
    "write",
    "writelines",
    "write_text",
    "writerow",
    "writerows",
}

N_SCOPE_MARKERS = (
    '"n"',
    "'n'",
    "n_subset",
    "n_pairs",
    "verdict_pairs",
    "sample_pairs",
    "population",
    "n_population",
)
MODE_SCOPE_MARKERS = (
    "selection_mode",
    "selection mode",
    "pair_selection",
    "pair selection",
    "mode=",
    "video_order_prefix",
    "seeded_random",
    "seeded-random",
    "stratified",
    "strided",
    "prefix",
)
BIAS_SCOPE_MARKERS = (
    "axis_bias",
    "axis-bias",
    "axis bias",
    "governing_ratio",
    "governing_ratios",
    "population_matched",
    "different population",
    "subset/population",
    "subset vs population",
    "prefix bias",
    "pose axis",
    "seg axis",
)


@dataclass(frozen=True)
class SubsetDefaultSite:
    path: str
    line: int
    scope: str
    kind: str
    hint: str
    default: int | None
    emits_verdict: bool
    reports_scope_fields: bool
    status: str

    @property
    def key(self) -> str:
        return f"{self.path}|{self.line}|{self.scope}|{self.kind}|{self.hint}"


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


def _attribute_tail(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _source_hint(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - best-effort display only
        return node.__class__.__name__


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _subset_population_for_label(label: str) -> int:
    return FRAME_POPULATION if "frame" in label else PAIR_POPULATION


def _is_subset_default(label: str, default: int) -> bool:
    return 0 < default < _subset_population_for_label(label)


def _argparse_dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _intrinsic_subset_limit_name(name: str) -> bool:
    return name.startswith(("sample_", "verdict_", "limit_", "max_")) or name in {
        "n_verdict_pairs",
    }


def _target_suggests_pairs(node: ast.AST) -> bool:
    tail = _attribute_tail(node)
    if tail is None:
        return False
    lower = tail.lower()
    return "pair" in lower or "frame" in lower


def _range_bound_is_subset_expr(node: ast.AST, subset_default_names: set[str]) -> bool:
    tail = _attribute_tail(node)
    if tail is None or tail == "n" or tail not in PAIR_COUNT_NAMES:
        return False
    return tail in subset_default_names or _intrinsic_subset_limit_name(tail)


def _has_any(text_lower: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text_lower for marker in markers)


def _reports_scope_fields(text_lower: str) -> bool:
    return (
        _has_any(text_lower, N_SCOPE_MARKERS)
        and _has_any(text_lower, MODE_SCOPE_MARKERS)
        and _has_any(text_lower, BIAS_SCOPE_MARKERS)
    )


def _emits_verdict_like_output(tree: ast.AST, text_lower: str) -> bool:
    if not _has_any(text_lower, VERDICT_MARKERS):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) in SINK_NAMES:
            return True
    return False


def _status(*, emits_verdict: bool, reports_scope_fields: bool) -> str:
    if not emits_verdict:
        return "dormant_no_verdict"
    if reports_scope_fields:
        return "scope_reported"
    return "silent_verdict_subset_default"


class _SubsetDefaultVisitor(ast.NodeVisitor):
    def __init__(
        self,
        path: str,
        *,
        emits_verdict: bool,
        reports_scope_fields: bool,
        subset_default_names: set[str],
    ) -> None:
        self.path = path
        self.emits_verdict = emits_verdict
        self.reports_scope_fields = reports_scope_fields
        self.subset_default_names = subset_default_names
        self.scope_stack: list[str] = []
        self.sites: list[SubsetDefaultSite] = []

    @property
    def scope(self) -> str:
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"

    def _append(
        self,
        node: ast.AST,
        *,
        kind: str,
        hint: str,
        default: int | None = None,
    ) -> None:
        self.sites.append(
            SubsetDefaultSite(
                path=self.path,
                line=int(getattr(node, "lineno", 0)),
                scope=self.scope,
                kind=kind,
                hint=hint,
                default=default,
                emits_verdict=self.emits_verdict,
                reports_scope_fields=self.reports_scope_fields,
                status=_status(
                    emits_verdict=self.emits_verdict,
                    reports_scope_fields=self.reports_scope_fields,
                ),
            )
        )

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
            subset_flags = [flag for flag in flags if SUBSET_FLAG_RE.match(flag)]
            if subset_flags:
                default = None
                for kw in node.keywords:
                    if kw.arg == "default":
                        default = _numeric_int(kw.value)
                        break
                if default is not None:
                    for flag in subset_flags:
                        if _is_subset_default(flag, default):
                            self._append(
                                node,
                                kind="cli_subset_default",
                                hint=f"{flag}={default}",
                                default=default,
                            )
        if _call_name(node.func) == "range" and node.args:
            upper = node.args[0] if len(node.args) == 1 else node.args[1]
            if _range_bound_is_subset_expr(upper, self.subset_default_names):
                self._append(
                    node,
                    kind="prefix_range",
                    hint=f"range({_source_hint(upper)})",
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        sl = node.slice
        if isinstance(sl, ast.Slice) and sl.lower is None and sl.step is None and sl.upper is not None:
            upper = _attribute_tail(sl.upper)
            bare_n_on_pair_target = upper == "n" and _target_suggests_pairs(node.value)
            subset_bound = (
                upper in self.subset_default_names
                or (upper is not None and _intrinsic_subset_limit_name(upper))
                or bare_n_on_pair_target
            )
            if upper in PAIR_COUNT_NAMES and subset_bound:
                self._append(
                    node,
                    kind="prefix_slice",
                    hint=f"{_source_hint(node.value)}[:{_source_hint(sl.upper)}]",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        default = _numeric_int(node.value)
        if default is not None:
            for target in node.targets:
                self._maybe_assignment_default(node, target, default)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value is not None:
            default = _numeric_int(node.value)
            if default is not None:
                self._maybe_assignment_default(node, node.target, default)
        self.generic_visit(node)

    def _maybe_assignment_default(self, node: ast.AST, target: ast.AST, default: int) -> None:
        tail = _attribute_tail(target)
        if tail in PAIR_COUNT_NAMES and tail != "n" and _is_subset_default(tail, default):
            self._append(
                node,
                kind="assigned_subset_default",
                hint=f"{_source_hint(target)}={default}",
                default=default,
            )


class _SubsetDefaultNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            subset_flags = [flag for flag in flags if SUBSET_FLAG_RE.match(flag)]
            if subset_flags:
                default = None
                dest = None
                for kw in node.keywords:
                    if kw.arg == "default":
                        default = _numeric_int(kw.value)
                    elif kw.arg == "dest" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        dest = kw.value.value
                if default is not None:
                    for flag in subset_flags:
                        if _is_subset_default(flag, default):
                            self.names.add(dest or _argparse_dest(flag))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        default = _numeric_int(node.value)
        if default is not None:
            for target in node.targets:
                self._maybe_add(target, default)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value is not None:
            default = _numeric_int(node.value)
            if default is not None:
                self._maybe_add(node.target, default)
        self.generic_visit(node)

    def _maybe_add(self, target: ast.AST, default: int) -> None:
        tail = _attribute_tail(target)
        if tail in PAIR_COUNT_NAMES and tail != "n" and _is_subset_default(tail, default):
            self.names.add(tail)


def _subset_default_names(tree: ast.AST) -> set[str]:
    collector = _SubsetDefaultNameCollector()
    collector.visit(tree)
    return collector.names


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


def scan_subset_default_scope_fields(
    roots: list[Path] | None = None,
    *,
    include_tests: bool = False,
) -> dict[str, Any]:
    roots = roots or [REPO / r for r in DEFAULT_ROOTS]
    sites: list[SubsetDefaultSite] = []
    parse_errors: list[dict[str, str]] = []
    files = _iter_python_files(roots, include_tests=include_tests)
    files_matched: set[str] = set()

    for path in files:
        rel = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append({"path": rel, "error": str(exc)})
            continue
        text_lower = text.lower()
        visitor = _SubsetDefaultVisitor(
            rel,
            emits_verdict=_emits_verdict_like_output(tree, text_lower),
            reports_scope_fields=_reports_scope_fields(text_lower),
            subset_default_names=_subset_default_names(tree),
        )
        visitor.visit(tree)
        if visitor.sites:
            files_matched.add(rel)
        sites.extend(visitor.sites)

    status_counts = {
        "scope_reported": sum(site.status == "scope_reported" for site in sites),
        "silent_verdict_subset_default": sum(
            site.status == "silent_verdict_subset_default" for site in sites
        ),
        "dormant_no_verdict": sum(site.status == "dormant_no_verdict" for site in sites),
    }
    return {
        "schema": "subset_default_scope_fields_scan.v1",
        "roots": [str(r.relative_to(REPO)) if r.is_relative_to(REPO) else str(r) for r in roots],
        "include_tests": include_tests,
        "files_scanned": len(files),
        "files_matched": len(files_matched),
        "parse_errors": parse_errors,
        "subset_default_sites": len(sites),
        **status_counts,
        "allowed_silent_subset_default_keys": [
            site.key for site in sites if site.status == "silent_verdict_subset_default"
        ],
        "sites": [
            {**asdict(site), "key": site.key}
            for site in sites
        ],
    }


def _load_allowed_keys(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "allowed_silent_subset_default_keys" in obj:
        return {str(k) for k in obj["allowed_silent_subset_default_keys"]}
    if "sites" in obj:
        return {
            str(site["key"])
            for site in obj["sites"]
            if site.get("status") == "silent_verdict_subset_default"
        }
    raise ValueError(f"baseline has no allowed subset-default keys: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path, help="roots to scan")
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--write-baseline", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    roots = [p if p.is_absolute() else REPO / p for p in args.roots] if args.roots else None
    report = scan_subset_default_scope_fields(roots, include_tests=args.include_tests)

    new_silent: list[str] = []
    if args.baseline is not None:
        allowed = _load_allowed_keys(args.baseline)
        current = {
            site["key"]
            for site in report["sites"]
            if site["status"] == "silent_verdict_subset_default"
        }
        new_silent = sorted(current - allowed)
        report["baseline"] = str(args.baseline)
        report["new_silent_subset_default_sites"] = len(new_silent)
        report["new_silent_subset_default_keys"] = new_silent

    if args.write_baseline is not None:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(
            "subset-default scope scan: files={files_scanned} matched={files_matched} "
            "sites={subset_default_sites} scope_reported={scope_reported} "
            "silent={silent_verdict_subset_default} dormant={dormant_no_verdict}".format(
                **report
            )
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
    raise SystemExit(main(sys.argv[1:]))
