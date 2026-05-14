#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""B5 — Inflate wire format dead-bytes scanner.

Bug class: ``inflate.py`` reads N-byte side-info from the wire format that's
never used in state-dict assembly.

Real instance: ADMM step 6 ``inflate.py:117-121`` reads 28 K bytes; comment
claims they're charged for audit/reproducibility, but
``reconstructed_q = chunk`` is a no-op.

Detection: scan ``submissions/exact_current/inflate.py`` AND every
``submission_dir/inflate.py`` under ``experiments/results/``. For each
inflate file, find variables read from the archive byte stream
(struct.unpack OR ``read(n_bytes)`` assignments). Then verify each captured
variable name is actually used downstream in:

  * ``state_dict`` assembly (load_state_dict, model.load_state_dict, …),
  * a tensor reconstruction expression (any ``torch.`` / ``np.`` /
    ``.reshape()`` / ``.view()`` containing the variable),
  * an explicit ``# DEAD_BYTES_AUDIT_OK:<reason>`` marker on the same or
    next line documenting why bytes are charged but unused.

A read whose target name is later only assigned (e.g. ``reconstructed_q =
chunk`` where ``chunk`` is then never consumed) is a violation.

Memory ref: ``feedback_review_engineering_council_4_landings_20260508.md``
(REVIEW-ENG C2).

NOTE: ``submissions/exact_current/inflate.py`` is in the "must not edit"
mutation frontier — this scanner reads but never writes it.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

DEAD_BYTES_WAIVER_RE = re.compile(
    r"#\s*DEAD_BYTES_AUDIT_OK\s*(?::\s*([^\n]*))?",
    re.IGNORECASE,
)


@dataclass
class Finding:
    rel_path: str
    lineno: int
    var_name: str
    reason: str


class _UsageVisitor(ast.NodeVisitor):
    """Collect Name nodes referenced anywhere (load context) plus method calls."""

    def __init__(self) -> None:
        self.names_loaded: set[str] = set()
        self.attr_targets: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names_loaded.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # foo.bar(...) — record `foo`
        if isinstance(node.value, ast.Name):
            self.attr_targets.add(node.value.id)
        self.generic_visit(node)


def _scan_inflate_file(path: Path, repo_root: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    rel = str(path.relative_to(repo_root))
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    # Find all Name targets bound from a struct.unpack or fp.read() expression.
    # Track lineno so we can associate waiver comments. The RHS may be the
    # call directly OR a subscript/index of the call (e.g.
    # `audit_token = struct.unpack("<I", buf)[0]`).
    def _rhs_is_unpack_or_read(rhs: ast.AST) -> bool:
        # Unwrap Subscript / Attribute / call-chain: walk inside until we
        # find a Call whose func is .unpack/.unpack_from/.read/.frombuffer.
        for sub in ast.walk(rhs):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr in (
                    "unpack",
                    "unpack_from",
                    "read",
                    "frombuffer",
                )
            ):
                return True
        return False

    candidate_targets: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        # Catalog #168 fix 2026-05-12: handle both `var = struct.unpack(...)`
        # (Assign) and `var: bytes = struct.unpack(...)` (AnnAssign) so the
        # dead-bytes audit doesn't silently miss annotated wire-format reads.
        if isinstance(node, ast.Assign):
            rhs = node.value
            target_iter = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            rhs = node.value
            target_iter = [node.target]
        else:
            continue
        if not _rhs_is_unpack_or_read(rhs):
            continue
        for tgt in target_iter:
            if isinstance(tgt, ast.Name):
                candidate_targets.append((tgt.id, node.lineno))
            elif isinstance(tgt, ast.Tuple):
                for elt in tgt.elts:
                    if isinstance(elt, ast.Name):
                        candidate_targets.append((elt.id, node.lineno))

    # Collect every Name usage in Load context across the file (post-binding).
    usage = _UsageVisitor()
    usage.visit(tree)

    # For each candidate target, verify it's used somewhere.
    lines = text.splitlines()
    findings: list[Finding] = []
    for var_name, lineno in candidate_targets:
        # Trivial single-character / underscore targets (_ ignored chunks)
        if var_name == "_" or var_name.startswith("__"):
            continue
        # Direct usage anywhere = OK.
        if var_name in usage.names_loaded:
            continue
        if var_name in usage.attr_targets:
            continue
        # Waiver on same line or next line.
        idx = lineno - 1
        window = "\n".join(lines[max(0, idx - 1) : min(len(lines), idx + 2)])
        if DEAD_BYTES_WAIVER_RE.search(window):
            continue
        findings.append(
            Finding(
                rel_path=rel,
                lineno=lineno,
                var_name=var_name,
                reason=(
                    f"`{var_name}` bound from struct.unpack/read but never "
                    f"loaded later. Either consume it in state-dict assembly "
                    f"OR annotate `# DEAD_BYTES_AUDIT_OK:<reason>` on the "
                    f"same line documenting why those bytes are charged."
                ),
            )
        )
    return findings


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    candidates: list[Path] = []

    # Submissions
    sub = repo / "submissions"
    if sub.is_dir():
        for p in sub.rglob("inflate.py"):
            candidates.append(p)

    # Generated submission_dirs (own builds only — exclude vendored
    # public-PR intakes; their inflate.py is forensic input only and is
    # covered by separate public-frontier audits).
    res_root = repo / "experiments" / "results"
    if res_root.is_dir():
        for p in res_root.rglob("inflate.py"):
            # skip __pycache__
            if "__pycache__" in p.parts:
                continue
            parts = set(p.parts)
            if any(
                token in parts
                for token in (
                    "public_pr_archive_release_view",
                    "public_pr_archive_kaggle_mirror",
                    "public_pr_intake_full",
                )
            ):
                continue
            # also exclude paths whose any-component contains _intake_ marker
            if any("_intake_" in part for part in p.parts):
                continue
            candidates.append(p)

    for c in candidates:
        findings.extend(_scan_inflate_file(c, repo))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    findings = scan(repo)
    if findings:
        print(f"[B5-inflate-dead-bytes] {len(findings)} violation(s):", file=sys.stderr)
        for f in findings[:20]:
            print(f"  • {f.rel_path}:{f.lineno} ({f.var_name}): {f.reason}", file=sys.stderr)
        if len(findings) > 20:
            print(f"  … (+{len(findings) - 20} more)", file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("[B5-inflate-dead-bytes] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
