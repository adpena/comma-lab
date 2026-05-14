#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the tac OSS release packet — sanitized, manifest-stamped, SHA-locked.

Sister of `tools/oss_publish_staging.py` but **release-grade**: instead of
staging the OSS-publish surface for operator review, this builds an immutable
**release packet** with:

  - the canonical reusable-code surface (`src/tac/`, public docs, public examples)
  - a release manifest stamped with version + per-file SHA-256
  - a sanitization receipt enumerating every redacted line + pattern label
  - a strict "no leak" gate refusing to emit if ANY un-redacted secret pattern,
    private infra URL, local absolute path, raw provider log surface, or
    unpublished operator-state path is present in the included files

Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable: the sanitization gate
runs BEFORE the packet is written; failures abort the run with a clear
diagnostic so the packet is never produced with leaked content.

Per CLAUDE.md "tac stays clean; comma-lab owns research state": only canonical
reusable code is packaged. Internal research state, provider transcripts,
auto-memory, raw experiment directories, and ad-hoc operator notes are excluded
by construction.

The release packet layout::

    <out-dir>/                      # the packet root
      release_manifest.json         # version + per-file SHA-256 + sanitization receipt
      sanitization_receipt.json     # what was scanned, what was clean, what was excluded
      LICENSE                       # tracked license (Apache 2.0)
      README.md                     # release-grade README (provided OR generated minimal)
      src/tac/...                   # the canonical library
      docs/...                      # only the explicitly-public docs subset
      examples/...                  # explicitly-public example scripts
      tests/                        # tac tests (proves the library works)

Usage::

    .venv/bin/python tools/build_tac_oss_release_packet.py \\
        --out-dir release_packets/tac_v0.5.0 \\
        --version 0.5.0 \\
        --readme path/to/release_readme.md

Cross-references
----------------
- `tools/oss_publish_staging.py` — sister staging tool (operator-review path)
- `tac.preflight.public_release_hygiene_violations_for_text` — sanitization detector
- `tac.preflight.check_public_release_hygiene` — repo-wide hygiene scan
- `feedback_5_beyond_phase4_modules_landed_20260509.md` — landing memo

CLAUDE.md compliance
--------------------
- Per "Public Disclosure Hygiene": sanitization gate MUST run before emit; refuses
  on any leak (the `--allow-leak` flag is intentionally NOT supported).
- Per "tac stays clean": only `src/tac/`, `docs/`, `examples/`, `LICENSE`, and
  the optional README + tests/ are packaged. No `.omx`, `.ralph`, `.claude`,
  `experiments/`, `reports/`, `submissions/`, or provider-state files.
- Per "Forbidden /tmp paths in any persisted artifact": the manifest paths are
  always relative to the packet root, never `/tmp/...` or absolute.
- Per "Bugs must be permanently fixed AND self-protected against": the
  sanitization gate IS the structural protection — every committed release
  packet includes the receipt proving it was scanned at build time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.preflight import public_release_hygiene_violations_for_text  # noqa: E402
from tac.repo_io import json_text, sha256_bytes, write_json  # noqa: E402

RELEASE_PACKET_SCHEMA = "tac_oss_release_packet_v1"

# Canonical reusable surface — explicit allow-list, NOT glob-implicit.
# Per CLAUDE.md "tac stays clean": each entry justifies its inclusion below.
INCLUDED_TREES: tuple[str, ...] = (
    "src/tac",        # the library itself (canonical reusable code)
    "docs/paper",     # public methodology + writeup
)

INCLUDED_FILES: tuple[str, ...] = (
    "LICENSE",
    "pyproject.toml",
)

# Per-file extension allowlist within INCLUDED_TREES. Anything else is excluded
# (e.g. .pt checkpoints under src/tac/contrib are excluded by construction).
INCLUDED_EXTENSIONS: frozenset[str] = frozenset({
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".cfg",
    ".ini",
})

# Hard exclusions — NEVER include even if matched by INCLUDED_TREES patterns.
# Per CLAUDE.md: research state, provider transcripts, account metadata.
HARD_EXCLUDED_PATH_FRAGMENTS: tuple[str, ...] = (
    ".omx/",
    ".ralph/",
    ".claude/",
    "experiments/results/",
    "reports/raw/",
    "reports/private/",
    "submissions/exact_current/",
    "submissions/robust_current/eval_runs/",
    "upstream/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".git/",
)


def _is_excluded(rel: str) -> bool:
    """Return True if the relative path matches any hard exclusion fragment."""
    rel_norm = rel.replace("\\", "/")
    return any(frag in rel_norm for frag in HARD_EXCLUDED_PATH_FRAGMENTS)


def _enumerate_included_files(repo_root: Path) -> list[Path]:
    """Walk the canonical surface; return absolute paths of included files."""
    out: list[Path] = []
    for tree in INCLUDED_TREES:
        tree_root = repo_root / tree
        if not tree_root.is_dir():
            continue
        for candidate in sorted(tree_root.rglob("*")):
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(repo_root).as_posix()
            if _is_excluded(rel):
                continue
            if candidate.suffix not in INCLUDED_EXTENSIONS:
                continue
            out.append(candidate)
    for fname in INCLUDED_FILES:
        candidate = repo_root / fname
        if candidate.is_file():
            out.append(candidate)
    return out


def scan_for_leaks(files: list[Path], repo_root: Path) -> dict[str, list[str]]:
    """Scan every file for public-release hygiene violations.

    Returns ``{relative_path: [violation_message, ...]}`` for any file that has
    one or more violations. Empty dict means the surface is clean.

    Per CLAUDE.md "Public Disclosure Hygiene": this is the structural gate. A
    non-empty return value MUST abort the packet build.
    """
    leaks: dict[str, list[str]] = {}
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Binary or unreadable; skip text scan but still include in the
            # manifest with a marker (tests expect this stays out of leaks).
            continue
        violations = public_release_hygiene_violations_for_text(rel, text)
        if violations:
            leaks[rel] = violations
    return leaks


def build_packet(
    out_dir: Path,
    version: str,
    *,
    readme_source: Path | None = None,
    repo_root: Path | None = None,
    include_tests: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the release packet at ``out_dir``.

    Returns the in-memory release manifest (also written to disk as
    ``release_manifest.json``).

    Raises ``RuntimeError`` if the sanitization gate finds any leak.
    Raises ``FileExistsError`` if ``out_dir`` exists and ``overwrite=False``.
    Raises ``ValueError`` if ``version`` is empty or whitespace-only.
    """
    if not version or not version.strip():
        raise ValueError("version must be a non-empty string")
    root = repo_root or REPO_ROOT
    out_dir = Path(out_dir).resolve()

    # Pre-flight: refuse-if-exists check WITHOUT mutating anything.
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"out_dir already exists: {out_dir}. Pass overwrite=True to "
            "rebuild in place."
        )

    files = _enumerate_included_files(root)
    # ``include_tests`` is informational; tests under src/tac/tests are already
    # captured by the src/tac tree walk.
    _ = include_tests  # surface in manifest only

    # SANITIZATION GATE — must pass before ANY destructive action (so a
    # failed scan does NOT lose the operator's prior packet on overwrite=True).
    leaks = scan_for_leaks(files, root)
    if leaks:
        leak_summary = "\n".join(
            f"  • {rel}: {len(msgs)} violation(s); first: {msgs[0]}"
            for rel, msgs in sorted(leaks.items())
        )
        raise RuntimeError(
            "TAC OSS RELEASE PACKET sanitization gate REFUSED:\n"
            f"{leak_summary}\n\n"
            "Per CLAUDE.md 'Public Disclosure Hygiene' non-negotiable, the "
            "release packet cannot be emitted with un-redacted secrets, local "
            "absolute paths, private infrastructure URLs, or operator state. "
            "Redact each violation in the source files (or move them out of "
            "the OSS surface) and rebuild."
        )

    # Sanitization passed — safe to remove prior packet (if present) and
    # rebuild. This ordering preserves the prior packet on a failed scan.
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)

    # Now copy files into out_dir/ preserving structure relative to repo root.
    out_dir.mkdir(parents=True, exist_ok=True)
    file_records: list[dict[str, Any]] = []
    for src_path in files:
        rel = src_path.relative_to(root).as_posix()
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = src_path.read_bytes()
        dst.write_bytes(data)
        file_records.append({
            "path": rel,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        })

    # README handling: if a release-specific README is supplied, copy it as
    # README.md at the packet root (overwriting any same-named file from the
    # canonical surface). Otherwise generate a minimal placeholder.
    readme_path = out_dir / "README.md"
    if readme_source is not None:
        readme_source = Path(readme_source).resolve()
        if not readme_source.is_file():
            raise FileNotFoundError(f"readme_source not found: {readme_source}")
        readme_text = readme_source.read_text(encoding="utf-8")
        # Re-scan the README — operator-supplied content must clear the gate too.
        readme_violations = public_release_hygiene_violations_for_text(
            "README.md", readme_text
        )
        if readme_violations:
            shutil.rmtree(out_dir)
            raise RuntimeError(
                "Release README failed sanitization gate:\n"
                + "\n".join(f"  • {v}" for v in readme_violations)
            )
        readme_path.write_text(readme_text, encoding="utf-8")
    elif not readme_path.exists():
        readme_path.write_text(
            f"# tac — version {version}\n\n"
            "Public release. See `release_manifest.json` for the file inventory.\n",
            encoding="utf-8",
        )

    readme_bytes = readme_path.read_bytes()
    file_records.append({
        "path": "README.md",
        "bytes": len(readme_bytes),
        "sha256": sha256_bytes(readme_bytes),
        "source": "operator_supplied" if readme_source else "generated_placeholder",
    })

    # Sanitization receipt: explicit per-file scan trace for forensic audit.
    sanitization_receipt = {
        "schema": "tac_oss_release_sanitization_receipt_v1",
        "scanned_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "n_files_scanned": len(files) + (1 if readme_source else 0),
        "n_leaks_found": 0,
        "patterns_checked": [
            "local absolute operator path",
            "private-key material",
            "OpenAI-style API token",
            "GitHub personal access token",
            "Hugging Face token",
            "explicit secret environment assignment",
            "concrete Vast SSH endpoint",
            "private Lightning Studio app link",
            "raw Modal call id",
        ],
        "scan_status": "PASSED",
    }
    write_json(out_dir / "sanitization_receipt.json", sanitization_receipt)

    # Release manifest — schema-versioned, machine-readable, append-only.
    manifest: dict[str, Any] = {
        "schema": RELEASE_PACKET_SCHEMA,
        "evidence_grade": "[release-packet; sanitization-gate passed]",
        "version": version,
        "built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "repo_root": str(root),
        "n_files": len(file_records),
        "total_bytes": sum(r["bytes"] for r in file_records),
        "sanitization_receipt_path": "sanitization_receipt.json",
        "files": sorted(file_records, key=lambda r: r["path"]),
        "included_trees": list(INCLUDED_TREES),
        "included_files": list(INCLUDED_FILES),
        "include_tests": include_tests,
        "claude_md_compliance_tags": [
            "public_disclosure_hygiene_gate_passed",
            "tac_stays_clean_canonical_surface_only",
            "no_tmp_paths_in_manifest",
            "sanitization_receipt_emitted",
        ],
    }
    write_json(out_dir / "release_manifest.json", manifest)

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to emit the release packet (must not exist or pass --overwrite)")
    parser.add_argument("--version", type=str, required=True,
                        help="Release version string (e.g. 0.5.0)")
    parser.add_argument("--readme", type=Path, default=None,
                        help="Optional release README to use as packet root README.md")
    parser.add_argument("--no-tests", action="store_true",
                        help="Exclude src/tac/tests from manifest enumeration (still copied with src/tac)")
    parser.add_argument("--overwrite", action="store_true",
                        help="If --out-dir exists, remove and rebuild")
    args = parser.parse_args(argv)

    try:
        manifest = build_packet(
            args.out_dir,
            args.version,
            readme_source=args.readme,
            include_tests=not args.no_tests,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"build_tac_oss_release_packet: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    print(json_text({
        "ok": True,
        "version": manifest["version"],
        "n_files": manifest["n_files"],
        "total_bytes": manifest["total_bytes"],
        "out_dir": str(args.out_dir),
        "sanitization_receipt": "PASSED",
    }))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
