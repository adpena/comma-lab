#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""B3 — Build-manifest archive custody check.

Bug class: a ``build_manifest.json`` references ``archive_relpath`` but the
archive is gitignored AND no CI rebuild-and-SHA-assert smoke exists.

Real instance: Path B step 6 ``archive.zip`` at
``experiments/results/admm_x_lossy_coarsening_path_b_step6_20260508T060435Z/
archive.zip`` exists on dirty disk only.

Detection: walk all ``build_manifest.json`` (or ``manifest.json`` carrying
``schema_version`` ending in ``_build.v1``) under ``experiments/results/``.
For each manifest that declares ``archive_relpath`` + ``archive_sha256``,
verify EITHER:
  * the archive exists in ``git ls-tree HEAD`` at the recorded relpath, OR
  * a verifier script under ``tools/verify_*_archive_sha256*.py`` /
    ``scripts/verify_*_archive_sha256*.sh`` references either the manifest
    relpath OR the SHA-256 string, OR
  * the manifest carries ``custody_status`` set to one of:
    ``"published"``, ``"committed-binary"``, ``"ci-rebuildable"``,
    ``"transient-allowed"`` (with explicit reason).

Memory ref: ``feedback_codex_adversarial_review_4_landings_20260508.md``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

ALLOWED_CUSTODY_STATUSES = frozenset(
    {
        "published",
        "committed-binary",
        "ci-rebuildable",
        "transient-allowed",
    }
)


@dataclass
class Finding:
    manifest_rel: str
    archive_relpath: str
    archive_sha256: str
    reason: str


@lru_cache(maxsize=1)
def _git_tracked_paths(repo_root_str: str) -> frozenset[str]:
    try:
        out = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=repo_root_str,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return frozenset()
    return frozenset(line.strip() for line in out.stdout.splitlines() if line.strip())


@lru_cache(maxsize=1)
def _verifier_scripts_text(repo_root_str: str) -> str:
    parts: list[str] = []
    repo = Path(repo_root_str)
    for sub in ("tools", "scripts"):
        d = repo / sub
        if not d.is_dir():
            continue
        for p in d.rglob("verify_*archive*sha*.*"):
            try:
                parts.append(p.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
        for p in d.rglob("rebuild_*archive*.*"):
            try:
                parts.append(p.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
    return "\n".join(parts)


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = repo_root or REPO_ROOT_DEFAULT
    repo = repo.resolve()
    tracked = _git_tracked_paths(str(repo))
    verifier_text = _verifier_scripts_text(str(repo))
    findings: list[Finding] = []

    results_root = repo / "experiments" / "results"
    if not results_root.is_dir():
        return findings

    for manifest_path in results_root.rglob("build_manifest.json"):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        archive_relpath = data.get("archive_relpath")
        archive_sha = data.get("archive_sha256")
        if not isinstance(archive_relpath, str) or not isinstance(archive_sha, str):
            continue
        # Allow explicit custody-status override.
        custody = str(data.get("custody_status", "")).strip().lower()
        if custody in ALLOWED_CUSTODY_STATUSES:
            continue
        # Tracked-in-git: archive committed.
        if archive_relpath in tracked:
            continue
        # Verifier references the manifest relpath OR the sha256.
        if (archive_relpath in verifier_text) or (archive_sha in verifier_text):
            continue
        rel_manifest = manifest_path.relative_to(repo)
        findings.append(
            Finding(
                manifest_rel=str(rel_manifest),
                archive_relpath=archive_relpath,
                archive_sha256=archive_sha,
                reason=(
                    "build manifest references an archive that is NOT "
                    "tracked in git AND no CI rebuild-and-verify-SHA script "
                    "references either the relpath or the SHA-256. Add a "
                    "tools/verify_<lane>_archive_sha256.py, set "
                    "`custody_status` to one of "
                    f"{sorted(ALLOWED_CUSTODY_STATUSES)}, or commit the "
                    "archive. B3 (custody hole)."
                ),
            )
        )
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
            f"[B3-build-manifest-custody] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings[:10]:
            print(
                f"  • {f.manifest_rel}: {f.archive_relpath} ({f.archive_sha256[:16]}…) "
                f"— {f.reason}",
                file=sys.stderr,
            )
        if len(findings) > 10:
            print(f"  … (+{len(findings) - 10} more)", file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("[B3-build-manifest-custody] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
