#!/usr/bin/env python3
"""Materialize a sanitized public comma-lab export tree.

This creates a fresh out-of-tree copy from tracked git blobs, not from the
current repository history. It is the safe path for publishing community docs,
historical records, and reproducibility tooling without flipping this private
working repository public.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.preflight import check_public_release_hygiene  # noqa: E402
from tac.repo_io import json_text, sha256_bytes, write_json  # noqa: E402
from tools.audit_public_publish_links import audit_public_publish_links  # noqa: E402

DEFAULT_REF = "HEAD"

DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = (
    "README.md",
    "AGENTS.md",
    "docs/*.md",
    "docs/**/*.md",
    "reports/graphs/build_public_site_bundle.py",
    "reports/graphs/test_build_public_site_bundle.py",
    "submissions/apogee/**/*.md",
    "submissions/apogee/**/*.sh",
    "src/comma_lab/*.py",
    "src/comma_lab/**/*.py",
    "src/tac/preflight.py",
    "src/tac/repo_io.py",
    "src/tac/submission_archive.py",
    "src/tac/tests/test_materialize_comma_lab_public_export.py",
    "tools/materialize_comma_lab_public_export.py",
    "pyproject.toml",
    "uv.lock",
)

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    ".git/**",
    ".hypothesis/**",
    ".omx/state/**",
    ".omx/status/**",
    ".omx/auto_memory_snapshot_*/**",
    ".omx/logs/**",
    ".omx/tmp/**",
    "docs/superpowers/**",
    "reports/graphs/site/**",
    "reports/raw/**",
    "reports/private/**",
    "experiments/results/**",
    "reverse_engineering/orphan_pyc_recovery_*/**",
    "upstream/videos/**",
    "upstream/models/**",
    "**/__pycache__/**",
    "**/*.pyc",
)


def _git_lines(repo_root: Path, args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return [line for line in proc.stdout.splitlines() if line]


def _git_blob(repo_root: Path, ref: str, path: str) -> bytes:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip())
    return proc.stdout


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def selected_export_paths(
    tracked_paths: list[str],
    *,
    include_patterns: tuple[str, ...] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[str]:
    selected: list[str] = []
    for path in sorted(tracked_paths):
        if _matches_any(path, exclude_patterns):
            continue
        if _matches_any(path, include_patterns):
            selected.append(path)
    return selected


def materialize_public_export(
    repo_root: Path,
    out_dir: Path,
    *,
    ref: str = DEFAULT_REF,
    allow_private_repo_links: bool = False,
    strict_hygiene: bool = True,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise SystemExit(f"FATAL: output directory is not empty: {out_dir}")
    else:
        out_dir.mkdir(parents=True)

    tracked = _git_lines(repo_root, ["ls-tree", "-r", "--name-only", ref])
    selected = selected_export_paths(tracked)
    copied: list[dict[str, object]] = []
    for path in selected:
        data = _git_blob(repo_root, ref, path)
        dst = out_dir / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        copied.append({"path": path, "bytes": len(data), "sha256": sha256_bytes(data)})

    manifest = {
        "schema_version": 1,
        "produced_by": "tools/materialize_comma_lab_public_export.py",
        "source_ref": ref,
        "source_head": _git_lines(repo_root, ["rev-parse", ref])[0],
        "output": "${PUBLIC_EXPORT_ROOT}",
        "include_patterns": list(DEFAULT_INCLUDE_PATTERNS),
        "exclude_patterns": list(DEFAULT_EXCLUDE_PATTERNS),
        "copied_count": len(copied),
        "copied_bytes": sum(int(item["bytes"]) for item in copied),
        "copied": copied,
        "publish_warning": (
            "This export is a sanitized tree, not a cleaned git history. "
            "Do not flip the private working repo public."
        ),
    }
    write_json(out_dir / "PUBLIC_EXPORT_MANIFEST.json", manifest)

    hygiene_violations = check_public_release_hygiene(
        repo_root=repo_root,
        strict=strict_hygiene,
        verbose=False,
        scan_paths=[out_dir],
    )
    link_payload = (
        {"violation_count": 0, "violations": []}
        if allow_private_repo_links
        else audit_public_publish_links([out_dir], base_root=out_dir, live=False)
    )
    link_violations = [
        "{path}:{line}: {kind}: {url} ({detail})".format(**violation)
        for violation in link_payload["violations"]
    ]
    if link_violations and strict_hygiene:
        raise SystemExit(
            "PUBLIC LINK HYGIENE violations:\n"
            + "\n".join(f"  - {violation}" for violation in link_violations[:40])
        )
    manifest["hygiene_violation_count"] = len(hygiene_violations)
    manifest["public_link_violation_count"] = len(link_violations)
    manifest["public_link_count"] = int(link_payload.get("link_count", 0))
    write_json(out_dir / "PUBLIC_EXPORT_MANIFEST.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--allow-private-repo-links", action="store_true")
    parser.add_argument("--no-strict-hygiene", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.force and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    manifest = materialize_public_export(
        args.repo_root,
        args.out_dir,
        ref=args.ref,
        allow_private_repo_links=args.allow_private_repo_links,
        strict_hygiene=not args.no_strict_hygiene,
    )
    print(json_text(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
