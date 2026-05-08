#!/usr/bin/env python3
"""B4 — Naming-vs-implementation drift scanner ("ADMM-named-but-not-ADMM").

Bug class: a class/function/file named "ADMM" or "primal_dual" implements
λ-bisection over independent per-tensor argmin (NOT iterative ADMM with
consensus updates).

Real instance: Path B step 5+6 tools used "ADMM" naming for what is actually
Lagrangian per-tensor allocation.

Detection: scan ``tools/`` and ``experiments/`` Python files. For each
function/class/file whose name contains ``admm`` (case-insensitive) or
``primal_dual`` / ``primal-dual``, verify the body contains AT LEAST ONE of:

  * ``rho`` updates (penalty parameter — distinguishes Lagrangian penalty
    method from pure dual-decomposition / pure bisection),
  * a ``z`` (or ``z_k``) consensus variable update,
  * a ``u`` (or ``u_k`` / ``y`` / ``lam_k``) explicit dual-variable update
    that is explicitly assigned inside an iteration loop (not just bisection),
  * an explicit ``# ADMM_WAIVED:<reason>`` marker (e.g.
    ``# ADMM_WAIVED:lambda-bisection-only`` documenting that the name is
    aspirational).

If none of the above is present, the file/function/class is naming-vs-impl
mismatched.

Memory ref: ``feedback_review_math_council_4_landings_20260508.md``
(Dykstra MEDIUM finding).
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

ADMM_NAME_RE = re.compile(r"admm|primal[_\-]dual", re.IGNORECASE)
WAIVER_RE = re.compile(r"#\s*ADMM_WAIVED\s*:", re.IGNORECASE)

# Markers for genuine iterative ADMM/consensus implementation. We look for
# explicit assignments to these names INSIDE a for/while loop body. Scan is
# regex-based per body text — full-AST control-flow analysis is overkill.
ADMM_BODY_TOKEN_RE = re.compile(
    r"\brho\s*[*+\-]?=\s*|"  # rho update
    r"\bz\s*=\s*|\bz_k\s*=\s*|\bz_new\s*=\s*|"  # z (consensus) update
    r"\bu\s*=\s*|\bu_k\s*=\s*|\blam_k\s*=\s*|\by_dual\s*=\s*|"  # dual update
    r"\bx\s*=\s*[^=].*\bz\b|"  # x-update referring to z
    r"\bproj_consensus|consensus_update|admm_step",
    re.MULTILINE,
)


@dataclass
class Finding:
    rel_path: str
    name: str
    kind: str  # "file" / "class" / "function"
    lineno: int
    reason: str


def _has_iterative_admm_body(text: str) -> bool:
    """Quick: text contains an admm/consensus-style assignment + a loop."""
    if not ADMM_BODY_TOKEN_RE.search(text):
        return False
    # Look for a for/while loop somewhere in the body.
    return ("for " in text) or ("while " in text)


def _scan_tree(path: Path, repo_root: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    rel = str(path.relative_to(repo_root))
    findings: list[Finding] = []
    file_admm_named = bool(ADMM_NAME_RE.search(path.stem))
    if WAIVER_RE.search(text):
        return []  # whole-file waiver
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    if file_admm_named and not _has_iterative_admm_body(text):
        findings.append(
            Finding(
                rel_path=rel,
                name=path.stem,
                kind="file",
                lineno=1,
                reason=(
                    "filename contains 'admm'/'primal_dual' but body lacks "
                    "rho/z/u/consensus iterative updates. Either add real "
                    "ADMM updates, rename to 'lagrangian_*' / 'bisection_*', "
                    "or add `# ADMM_WAIVED:<reason>` at file top."
                ),
            )
        )
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not ADMM_NAME_RE.search(node.name):
                continue
            try:
                body_text = ast.unparse(node)
            except (AttributeError, TypeError):
                # Older Pythons or unusual nodes
                continue
            if WAIVER_RE.search(body_text):
                continue
            if _has_iterative_admm_body(body_text):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            findings.append(
                Finding(
                    rel_path=rel,
                    name=node.name,
                    kind=kind,
                    lineno=getattr(node, "lineno", 0),
                    reason=(
                        f"{kind} name 'admm/primal_dual' but body lacks "
                        f"rho/z/u/consensus iterative updates. Either add "
                        f"real ADMM updates, rename ('lagrangian_'/"
                        f"'bisection_'), or annotate "
                        f"`# ADMM_WAIVED:<reason>` inside the body."
                    ),
                )
            )
    return findings


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = repo_root or REPO_ROOT_DEFAULT
    repo = repo.resolve()
    findings: list[Finding] = []
    scan_roots = [repo / "tools", repo / "experiments", repo / "src" / "tac"]
    for root in scan_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            # Skip vendored / harvested / env trees + test files
            parts = set(path.parts)
            if any(
                p in parts
                for p in (
                    "uv_project_env",
                    "site-packages",
                    "__pycache__",
                    "vast_harvest",
                    "tests",
                )
            ):
                continue
            if not ADMM_NAME_RE.search(path.read_text(encoding="utf-8", errors="ignore")[:4096]):
                # Cheap pre-filter: skip files that don't mention admm at all.
                continue
            findings.extend(_scan_tree(path, repo))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    findings = scan(repo)
    if findings:
        print(
            f"[B4-admm-naming-impl-mismatch] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings[:20]:
            print(
                f"  • {f.rel_path}:{f.lineno} {f.kind}={f.name}: {f.reason}",
                file=sys.stderr,
            )
        if len(findings) > 20:
            print(f"  … (+{len(findings) - 20} more)", file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("[B4-admm-naming-impl-mismatch] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
