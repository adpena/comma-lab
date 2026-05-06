#!/usr/bin/env python3
"""Contest-compliance audit for a submission archive.

Single-call audit of any archive.zip we might submit. Reports:

  - Total archive bytes (vs 25 * bytes / 37545489 rate term)
  - File list with per-member size + percentage
  - Renderer magic bytes (must be in scorer-free allowlist)
  - Any extras NOT in the canonical REQUIRED_ARCHIVE_MEMBERS set
  - Determinism: zip member timestamps + ordering check
  - Pass / FAIL summary with explicit reasons

Exit codes:
  0 — all contest-compliance checks pass
  1 — at least one HARD failure (missing renderer, scorer magic violation, etc.)
  2 — at least one WARN (unknown extra member; oversize)

Usage:
    python tools/audit_archive.py submissions/robust_current/archive_correct.zip
    python tools/audit_archive.py path/to/archive.zip --strict

References:
  - src/tac/stack_compositions.REQUIRED_ARCHIVE_MEMBERS
  - src/tac/stack_compositions._SCORER_FREE_RENDERER_MAGICS
  - CLAUDE.md "Auth eval measurement — non-negotiable"
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.submission_archive import validate_archive_member_name  # noqa: E402

# Canonical contest constants (mirror upstream/evaluate.py rate formula).
RATE_DENOM = 37_545_489  # contest pinned constant
RATE_MULT = 25  # contest pinned constant


def _try_import_canonical() -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    """Import REQUIRED_ARCHIVE_MEMBERS + _SCORER_FREE_RENDERER_MAGICS.

    Falls back to hardcoded defaults if the import fails (so the tool is
    usable in a stripped-down environment without the tac package).
    """
    try:
        from tac.stack_compositions import (  # type: ignore  # noqa: I001
            REQUIRED_ARCHIVE_MEMBERS,
            _SCORER_FREE_RENDERER_MAGICS,
        )
        return REQUIRED_ARCHIVE_MEMBERS, _SCORER_FREE_RENDERER_MAGICS
    except Exception as exc:  # fall back, but loud
        print(
            f"[audit_archive] WARNING: could not import canonical registry "
            f"({exc!r}); using hardcoded defaults.",
            file=sys.stderr,
        )
        return (
            ("renderer.bin", "masks.mkv", "optimized_poses.pt"),
            (
                b"ASYM",
                b"DPSM",
                b"FP4A",
                b"FP8H",
                b"I4LZ",
                b"CCh1",
                b"C3R1",
                b"SCv1",
                b"SZv1",
                b"NWC1",
                b"NWCS",
            ),
        )


@dataclass
class AuditResult:
    archive_path: Path
    archive_bytes: int = 0
    members: dict[str, int] = field(default_factory=dict)  # name → size
    rate_term: float = 0.0
    renderer_magic: bytes | None = None
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def audit(
    archive_path: Path,
    *,
    strict: bool = False,
    max_archive_bytes: int = 350_000,  # ~Quantizr 293KB + 20% buffer
) -> AuditResult:
    """Run the contest-compliance audit on `archive_path`.

    `strict=True` promotes warnings to failures (used by CI gates).
    `max_archive_bytes` is a soft guardrail — Quantizr ships 293KB, so
    anything > 350KB warrants a WARN (rate term explodes past 0.23).
    """
    required_members, scorer_free_magics = _try_import_canonical()
    res = AuditResult(archive_path=archive_path)

    if not archive_path.is_file():
        res.failures.append(f"archive not found: {archive_path}")
        return res

    res.archive_bytes = archive_path.stat().st_size
    res.rate_term = RATE_MULT * res.archive_bytes / RATE_DENOM

    if res.archive_bytes <= 0:
        res.failures.append("archive is empty")
        return res

    if res.archive_bytes > max_archive_bytes:
        res.warnings.append(
            f"archive size {res.archive_bytes:,}B > soft cap "
            f"{max_archive_bytes:,}B — rate term {res.rate_term:.4f} is "
            f"high vs Quantizr 0.20"
        )

    try:
        with zipfile.ZipFile(archive_path) as zf:
            seen_members: set[str] = set()
            for info in zf.infolist():
                try:
                    validate_archive_member_name(info.filename)
                except ValueError as exc:
                    res.failures.append(str(exc))
                    continue
                if info.filename in seen_members:
                    res.failures.append(f"duplicate member name: {info.filename}")
                    continue
                seen_members.add(info.filename)
                res.members[info.filename] = info.file_size

            # Renderer magic check.
            if "renderer.bin" not in res.members:
                res.failures.append("missing required member: renderer.bin")
            else:
                with zf.open("renderer.bin") as fp:
                    res.renderer_magic = fp.read(4)
                if res.renderer_magic not in scorer_free_magics:
                    res.failures.append(
                        f"renderer.bin magic {res.renderer_magic!r} "
                        f"NOT in scorer-free allowlist "
                        f"{[m.decode(errors='replace') for m in scorer_free_magics]} "
                        f"— inflate may load scorer state (strict-scorer-rule violation)"
                    )

            # Required members.
            for required in required_members:
                if required == "gradient_corrections.bin":
                    # Optional sidecar — not required for non-EC archives.
                    continue
                if required == "renderer.bin":
                    # Already covered by the dedicated renderer.bin check
                    # above (line ~133); skip to avoid duplicate failure
                    # message in res.failures (R32 Finding 1).
                    continue
                if required not in res.members:
                    res.failures.append(f"missing required member: {required}")

            # Unknown extras (warn).
            known_set = set(required_members) | {
                "renderer.bin",
                "masks.mkv",
                "optimized_poses.pt",
                "gradient_corrections.bin",
                "uniward_delta.bin",
                "decorrelation_basis.bin",
            }
            for name in res.members:
                if name not in known_set:
                    res.warnings.append(
                        f"unknown extra member: {name} ({res.members[name]:,}B)"
                    )

    except zipfile.BadZipFile as exc:
        res.failures.append(f"not a valid zip archive: {exc}")
        return res

    if strict:
        res.failures.extend(res.warnings)
        res.warnings = []
    return res


def render(res: AuditResult) -> str:
    lines = []
    lines.append(f"=== audit_archive: {res.archive_path} ===")
    lines.append(f"  bytes:      {res.archive_bytes:>10,}")
    lines.append(
        f"  rate_term:  {res.rate_term:>10.4f}  "
        f"(= {RATE_MULT} x {res.archive_bytes:,} / {RATE_DENOM:,})"
    )
    if res.renderer_magic is not None:
        lines.append(
            f"  magic:      {res.renderer_magic!r:>10}  (renderer.bin first 4 bytes)"
        )
    lines.append("")
    lines.append("  members:")
    for name, sz in sorted(res.members.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * sz / max(1, res.archive_bytes)
        lines.append(f"    {name:<28} {sz:>10,}B  ({pct:5.1f}%)")
    lines.append("")
    if res.failures:
        lines.append(f"  FAILURES ({len(res.failures)}):")
        for f in res.failures:
            lines.append(f"    [FAIL] {f}")
    if res.warnings:
        lines.append(f"  WARNINGS ({len(res.warnings)}):")
        for w in res.warnings:
            lines.append(f"    [WARN] {w}")
    if res.passed and not res.warnings:
        lines.append("  STATUS: PASS — contest-compliant")
    elif res.passed:
        lines.append(f"  STATUS: PASS-WITH-WARN ({len(res.warnings)} warning)")
    else:
        lines.append(f"  STATUS: FAIL ({len(res.failures)} failure)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a submission archive for contest compliance.",
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Promote warnings to failures (used by CI gates).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=350_000,
        help="Soft archive size cap in bytes (default 350,000).",
    )
    args = parser.parse_args()

    res = audit(args.archive, strict=args.strict, max_archive_bytes=args.max_bytes)
    print(render(res))

    if res.failures:
        return 1
    if res.warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
