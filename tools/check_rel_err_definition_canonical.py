#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""rel_err — canonical-definition discipline scanner.

Bug class: re-implementing ``rel_err`` inline silently in heterogeneous
forms (L1 vs L2 vs RMS vs per-element percentage) and feeding the result
into the Lagrangian per-tensor allocator's squared penalty
``cost = bytes + λ · w · rel_err²``. The squared penalty is only
mathematically a rate-MSE Lagrangian when ``rel_err`` is an L2-style
quantity (RMS / L2 ratio); L1 and per-element forms make the dual
ill-posed.

Detection: scan ``src/tac/`` and ``tools/`` for assignments of the form
``rel_err = …``. Allow:

* the canonical helper module ``src/tac/codec/rel_err.py``;
* call sites that explicitly use ``compute_rel_err(…)`` (the canonical
  helper) — they may still bind the result to ``rel_err``;
* sites carrying a same-line or preceding-comment waiver
  ``# REL_ERR_NON_CANONICAL_OK:<reason>``;
* generated/cached/vendored trees (``__pycache__``, ``site-packages``,
  ``uv_project_env``, ``vast_harvest``).

Anything else is flagged as a potential silent re-implementation and
should either route through ``tac.codec.rel_err.compute_rel_err`` or
declare its non-canonical form via the waiver comment.

Memory ref: ``feedback_rel_err_l1_rms_canonicalization_20260508.md``
+ ``.omx/research/rel_err_inconsistency_audit_20260508_claude.md``.

Promotion plan: starts ``strict=False``. After every existing live
violation in ``tools/pr101_*.py`` / ``tools/build_admm_*.py`` /
``tools/build_pr106_*.py`` / ``src/tac/optimization/*`` is annotated with
either a waiver or a routing through the canonical helper, flip to
STRICT in ``preflight_all()``.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

# Match `rel_err = …` (also rel_errs / rel_err_smoke / rel_err_pct etc.)
# only when the LHS is an unindexed bare assignment. Skip dict/list lookups
# like `r["rel_err"]` which are reads, not definitions.
REL_ERR_ASSIGN_RE = re.compile(
    r"""^(?P<indent>\s*)             # leading whitespace
        (?P<name>rel_err            # bare rel_err
            (?:_pct|_smoke|_pct_per_weight|_form|_fp32_smoke|_vs_quantized_fp32)?
        )
        \s*=\s*                      # assignment
        (?!=)                        # not ==
    """,
    re.VERBOSE,
)

# Allowed call patterns that DELEGATE to the canonical helper.
CANONICAL_CALL_RE = re.compile(
    r"compute_rel_err\s*\(|"
    r"aggregate_rel_err\s*\(|"
    r"\.rel_err\b|"  # `.rel_err` attribute access from AllocationResult
    r"result\.rel_err|"
    r"res\.rel_err"
)

# Same-line waiver
WAIVER_SAMELINE_RE = re.compile(r"#\s*REL_ERR_NON_CANONICAL_OK\s*:")

# Files exempt from scanning (they ARE the canonical implementations).
EXEMPT_REL_PATHS: frozenset[str] = frozenset(
    {
        "src/tac/codec/rel_err.py",
        # The encoder primitives intentionally compute their own forms; their
        # docstrings already declare the form via REL_ERR_FORM_* constants.
        "src/tac/codec/per_tensor_codecs.py",
    }
)

# Read-from-dict heuristics: skip lines like `rel_err = chosen["rel_err"]`
# (the LHS is a destructuring of an existing curve row, not a fresh
# computation). Routed-through-canonical accepts any line where the RHS
# already references compute_rel_err / aggregate_rel_err.
READ_FROM_DICT_RE = re.compile(
    r"=\s*[a-zA-Z_][\w\.]*\s*\[\s*[\"']rel_err[\"']\s*\]"
)
# Read-from-attribute (e.g., `rel_err = res.rel_err` or
# `rel_err = encoded["rel_err"]`).
READ_FROM_ATTR_RE = re.compile(
    r"=\s*[a-zA-Z_][\w\.]*\.rel_err\b"
)

# Numeric / proxy assignments: `rel_err = 0.0`, `rel_err = float(x)`,
# `rel_err = enc["rel_err"]`, `rel_err = abs_err / abs_orig if … else 0.0`,
# etc. We try to detect "computed from a difference" patterns so the
# scanner only flags fresh inline definitions.
COMPUTE_PATTERNS = (
    re.compile(r"abs_err"),
    re.compile(r"abs_orig"),
    re.compile(r"np\.linalg\.norm"),
    re.compile(r"np\.abs"),
    re.compile(r"np\.sqrt"),
    re.compile(r"\.norm\("),
)


@dataclass
class Finding:
    rel_path: str
    lineno: int
    reason: str


def _is_exempt(rel: str) -> bool:
    return rel in EXEMPT_REL_PATHS or rel.startswith(("src/tac/tests/",))


def _is_canonical_routed(line_text: str) -> bool:
    return bool(CANONICAL_CALL_RE.search(line_text))


def _is_read_from_existing(line_text: str) -> bool:
    """Lines that destructure an existing dict/object, not a fresh compute."""
    return bool(
        READ_FROM_DICT_RE.search(line_text)
        or READ_FROM_ATTR_RE.search(line_text)
    )


def _is_constant_assign(line_text: str) -> bool:
    """`rel_err = 0.0` / `rel_err = float("inf")` etc. — sentinels.

    Also matches array-init forms like ``np.zeros_like(abs_err)`` /
    ``np.zeros(N)`` / ``np.empty_like(...)`` which are buffer-allocation, not
    fresh distortion computations. The actual elementwise assignment that
    follows uses indexed LHS (``rel_err_pct[mask] = …``) and is not matched
    by ``REL_ERR_ASSIGN_RE``.
    """
    rhs = line_text.split("=", 1)[1] if "=" in line_text else ""
    rhs = rhs.split("#", 1)[0].strip()
    if rhs in {"0.0", "0", "1.0", '0', 'float("inf")'}:
        return True
    if re.fullmatch(r"float\(['\"]\s*inf\s*['\"]\)", rhs):
        return True
    # numpy array-init / sentinel-init forms
    if re.match(
        r"np\.(zeros|empty|ones|zeros_like|empty_like|ones_like|full|full_like)\b",
        rhs,
    ):
        return True
    return False


def _is_explicit_compute(line_text: str) -> bool:
    """Heuristic for fresh inline definition — references abs/norm/sqrt of diffs."""
    return any(p.search(line_text) for p in COMPUTE_PATTERNS)


def _scan_file(path: Path, repo_root: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    rel = str(path.relative_to(repo_root))
    if _is_exempt(rel):
        return []

    findings: list[Finding] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        m = REL_ERR_ASSIGN_RE.match(line)
        if not m:
            continue
        # Walking common false-positive cases first.
        if WAIVER_SAMELINE_RE.search(line):
            continue
        if _is_canonical_routed(line):
            continue
        if _is_read_from_existing(line):
            continue
        if _is_constant_assign(line):
            continue
        # Only flag when the RHS actually looks like an inline distortion
        # computation (np.abs, np.linalg.norm, abs_err / abs_orig).
        if not _is_explicit_compute(line):
            continue
        findings.append(
            Finding(
                rel_path=rel,
                lineno=i,
                reason=(
                    "inline rel_err computation lacks routing through "
                    "tac.codec.rel_err.compute_rel_err and lacks a "
                    "`# REL_ERR_NON_CANONICAL_OK:<reason>` waiver. The "
                    "Lagrangian penalty `λ · w · rel_err²` is a rate-MSE "
                    "dual only when rel_err is RMS/L2; declare the form "
                    "explicitly or route through the canonical helper."
                ),
            )
        )
    return findings


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    scan_roots = [repo / "src" / "tac", repo / "tools"]
    findings: list[Finding] = []
    for root in scan_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            parts = set(path.parts)
            if any(
                p in parts
                for p in (
                    "__pycache__",
                    "uv_project_env",
                    "site-packages",
                    "vast_harvest",
                )
            ):
                continue
            findings.extend(_scan_file(path, repo))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    findings = scan(repo)
    if findings:
        print(
            f"[rel-err-canonical] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        if args.verbose:
            for f in findings:
                print(f"  • {f.rel_path}:{f.lineno}: {f.reason}", file=sys.stderr)
        else:
            for f in findings[:20]:
                print(f"  • {f.rel_path}:{f.lineno}: {f.reason}", file=sys.stderr)
            if len(findings) > 20:
                print(f"  … (+{len(findings) - 20} more)", file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("[rel-err-canonical] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
