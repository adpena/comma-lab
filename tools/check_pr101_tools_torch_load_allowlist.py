#!/usr/bin/env python3
"""B8 — torch.load(weights_only=False) allowlist for cross-paradigm tools.

Bug class: ``torch.load(<path>, weights_only=False)`` without an explicit
allowlist comment marking the input as trusted. The canonical
``preflight_loader_format_safety`` covers the renderer/checkpoint loader
shape; this sister check extends to the cross-paradigm encoder tools that
operate on PR101 state-dict bytes (a different name surface).

Real instances (REVIEW-ENG C4):
  * ``tools/pr101_cross_paradigm_hstack_vstack_empirical.py:212``
  * ``tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py:143``

Detection rule: in any tool under ``tools/pr101_*.py`` that calls
``torch.load(..., weights_only=False)``, require ONE of:

  * a same-line or 5-line-window comment matching
    ``# WEIGHTS_ONLY_FALSE_OK:<reason>`` documenting why the input is trusted,
  * an explicit ``magic-byte`` validation OR a SHA-256 verification call in
    the 30 lines preceding the torch.load.

Memory ref: ``feedback_codex_adversarial_review_4_landings_20260508.md``.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

WEIGHTS_ONLY_FALSE_RE = re.compile(
    r"torch\.load\([^)]*weights_only\s*=\s*False",
)
WAIVER_RE = re.compile(r"#\s*WEIGHTS_ONLY_FALSE_OK\s*:", re.IGNORECASE)
PRECEDING_VALIDATION_RE = re.compile(
    r"sha256|hashlib|magic\s*=|magic_bytes|expected_sha|verify_sha",
    re.IGNORECASE,
)

SCANNED_GLOBS: tuple[str, ...] = (
    "tools/pr101_*.py",
    "tools/build_admm_*.py",
    "tools/build_cross_paradigm_*.py",
)


@dataclass
class Finding:
    rel_path: str
    lineno: int
    reason: str


def _scan_file(path: Path, repo_root: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    lines = text.splitlines()
    rel = str(path.relative_to(repo_root))
    findings: list[Finding] = []
    for i, line in enumerate(lines):
        if not WEIGHTS_ONLY_FALSE_RE.search(line):
            continue
        # 5-line window around the call for waiver
        win_start = max(0, i - 5)
        win_end = min(len(lines), i + 6)
        window = "\n".join(lines[win_start:win_end])
        if WAIVER_RE.search(window):
            continue
        # 30-line preceding window for validation (sha/magic check)
        pre_start = max(0, i - 30)
        pre = "\n".join(lines[pre_start:i])
        if PRECEDING_VALIDATION_RE.search(pre):
            continue
        findings.append(
            Finding(
                rel_path=rel,
                lineno=i + 1,
                reason=(
                    "torch.load(..., weights_only=False) without "
                    "`# WEIGHTS_ONLY_FALSE_OK:<reason>` waiver and without a "
                    "preceding sha256/magic-byte validation. B8."
                ),
            )
        )
    return findings


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    for glob in SCANNED_GLOBS:
        for p in sorted(repo.glob(glob)):
            if not p.is_file():
                continue
            findings.extend(_scan_file(p, repo))
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
            f"[B8-pr101-torch-load-allowlist] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings:
            print(f"  • {f.rel_path}:{f.lineno}: {f.reason}", file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("[B8-pr101-torch-load-allowlist] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
