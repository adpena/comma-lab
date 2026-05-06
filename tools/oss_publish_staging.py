#!/usr/bin/env python3
"""Stage the tac OSS-publish subset into a clean directory tree.

Per CLAUDE.md Strategic Secrecy Rule: this script COPIES the OSS-relevant
subset of the repo into an out-of-tree staging directory so the operator can
preview before `gh repo create adpena/tac --public --push`.

INCLUDES (OSS-publish surface):
    src/tac/                          — the tac library
    src/tac/tests/                    — tests prove it works
    tools/                            — public utilities (claim_lane_dispatch,
                                        subagent_commit_serializer, lane_maturity,
                                        oss_publish_staging itself)
    experiments/extract_pr106_decoder.py
    experiments/repack_pr106_with_*.py
    experiments/build_sjkl_*.py
    experiments/prepare_sjkl_pair_tensors.py
    experiments/block_fp_int*_codec_sketch.py
    submissions/apogee_v2/             — Lane Ω-W-V3 inflate adapter
    submissions/apogee_intN/           — Lane #04 inflate adapter
    scripts/remote_lane_*.sh           — dispatch wrappers (public methodology)
    scripts/ensure_remote_uv.sh        — canonical uv installer
    docs/recovery_report_20260504.md
    docs/pr106_stacking_decision_table_20260504.md
    docs/paper/                        — methodology + 4-results + related work
    pyproject.toml                     — package metadata
    LICENSE                            — proposed Apache 2.0 (operator confirms)
    README.md                          — generated from ${TAC_OSS_README}
                                         OR provided via --readme

EXCLUDES (per CLAUDE.md Strategic Secrecy Rule + practical bloat):
    .omx/                              — operator state + research notes (PRIVATE)
    .ralph/                            — research log (PRIVATE)
    .claude/                           — Claude Code config (PRIVATE)
    experiments/results/               — large empirical artifacts (LFS or skip)
    reports/                           — raw eval outputs + viz
    upstream/                          — pinned contest snapshot (don't redistribute)
    submissions/exact_current/         — per CLAUDE.md mutation frontier
    submissions/robust_current/        — internal track (not OSS-ready surface)
    runtime-rs/                        — separate Rust crate (own repo)
    cuda/, jax/, mojo/                 — separate language bindings
    .git/                              — staging is a fresh tree, not a clone
    *.pyc, __pycache__/                — bytecode caches
    *.egg-info/                        — install caches

Output:
    <out-dir>/                         — fresh staging tree, ready for git init + gh repo create
    <out-dir>/MANIFEST.json            — what was copied + what was excluded + sha-256 per file
    <out-dir>/README.md                — operator-reviewable copy of the OSS README
    <out-dir>/.gitignore               — minimal stop-list for the new repo

Usage:
    .venv/bin/python tools/oss_publish_staging.py \\
        --out-dir ${TAC_OSS_STAGING_ROOT} \\
        --readme ${TAC_OSS_README}

Then operator reviews:
    ls ${TAC_OSS_STAGING_ROOT}/
    cat ${TAC_OSS_STAGING_ROOT}/MANIFEST.json | jq .
    less ${TAC_OSS_STAGING_ROOT}/README.md

Then publishes:
    cd ${TAC_OSS_STAGING_ROOT}
    git init -b main
    git add .
    git commit -m "Initial public release of tac"
    gh repo create adpena/tac --public --source . --push

Per CLAUDE.md Strategic Secrecy: this script does NOT execute any
`gh repo create` or push commands. Operator runs those manually after review.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.preflight import check_public_release_hygiene  # noqa: E402
from tac.repo_io import sha256_bytes, write_json  # noqa: E402
from tools.audit_public_publish_links import audit_public_publish_links  # noqa: E402

# Files / directories to INCLUDE in the OSS staging copy. Globs are evaluated
# against REPO_ROOT. Order matters for human readability of MANIFEST.json.
_OSS_INCLUDE_PATTERNS: tuple[str, ...] = (
    # Library + tests
    "src/tac/**/*.py",
    # Public-facing utilities
    "tools/claim_lane_dispatch.py",
    "tools/subagent_commit_serializer.py",
    "tools/lane_maturity.py",
    "tools/oss_publish_staging.py",
    "tools/review_tracker.py",
    "tools/tool_bootstrap.py",
    "tools/audit_public_publish_links.py",
    # Lane Ω-W-V3 producer + adapter
    "experiments/extract_pr106_decoder.py",
    "experiments/build_sensitivity_map_pr106.py",
    "experiments/repack_pr106_with_water_filling.py",
    "submissions/apogee_v2/inflate.py",
    "submissions/apogee_v2/inflate.sh",
    "submissions/apogee_v2/src/model.py",
    "submissions/apogee_v2/src/codec.py",
    # Lane #04 producer + adapter
    "experiments/repack_pr106_with_int4_block_fp.py",
    "experiments/repack_pr106_with_intN_block_fp.py",
    "experiments/block_fp_int4_codec_sketch.py",
    "experiments/block_fp_intN_codec_sketch.py",
    "submissions/apogee_intN/inflate.py",
    "submissions/apogee_intN/inflate.sh",
    "submissions/apogee_intN/src/model.py",
    "submissions/apogee_intN/src/codec.py",
    "submissions/apogee_intN/src/intn_codec.py",
    # Lane SJ-KL producer pipeline (5 modules)
    "experiments/prepare_sjkl_pair_tensors.py",
    "experiments/build_sjkl_residual.py",
    "experiments/build_sjkl_c067_archive.py",
    # SJ-KL runtime helpers (also relied on by inflate paths)
    "submissions/robust_current/unpack_renderer_payload.py",
    # Dispatch wrappers (public methodology)
    "scripts/remote_lane_omega_w_v3_pr106.sh",
    "scripts/remote_lane_apogee_intN.sh",
    "scripts/remote_lane_sjkl_c067.sh",
    "scripts/ensure_remote_uv.sh",
    "scripts/ensure_remote_pip.sh",
    "scripts/remote_archive_only_eval.sh",
    # OSS-publish docs
    "docs/recovery_report_20260504.md",
    "docs/pr106_stacking_decision_table_20260504.md",
    "docs/paper/01_introduction.md",
    "docs/paper/02_method.md",
    "docs/paper/03_gradient_bug.md",
    "docs/paper/04_results.md",
    "docs/paper/05_production.md",
    "docs/paper/06_related_work.md",
    "docs/paper/07_discussion.md",
    # Package metadata
    "pyproject.toml",
)


# Patterns to EXCLUDE (matched against any path component or full relative path)
_OSS_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    ".git",
    ".omx",
    ".ralph",
    ".claude",
    ".hypothesis",
    ".venv",
    "node_modules",
    "experiments/results",
    "reports/raw",
    "upstream",  # pinned contest snapshot — don't redistribute
)


_README_FALLBACK = """\
# tac

Research code for the comma.ai video compression challenge.

(README staging placeholder — operator should provide a polished README via
`--readme <path>` before publishing. Suggested source:
`${TAC_OSS_README}` from the local release preparation work.)

For methodology see `docs/paper/`. For dispatch wrappers see `scripts/`.
For the codec library see `src/tac/`.
"""


_LICENSE_FALLBACK = """\
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

(Operator: confirm Apache 2.0 is the intended license; replace this file with
the full Apache 2.0 text + your copyright statement before publishing. The
short stub here is a placeholder so `gh repo create` doesn't choke on a
missing LICENSE file.)
"""


_GITIGNORE = """\
# tac OSS — staging-clean .gitignore
__pycache__/
*.pyc
*.egg-info/
.venv/
.hypothesis/
.pytest_cache/
.coverage
*.log
*.tmp
build/
dist/
"""


def _git_lines(repo_root: Path, args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if line]


def _git_blob(repo_root: Path, ref: str, path: str) -> bytes:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git show {ref}:{path} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_stage_output_path(repo_root: Path, out_dir: Path) -> Path:
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    if out_dir == repo_root or _is_relative_to(out_dir, repo_root):
        raise SystemExit(f"FATAL: --out-dir must be outside the repository: {out_dir}")
    if _is_relative_to(repo_root, out_dir):
        raise SystemExit(f"FATAL: --out-dir would contain the repository: {out_dir}")
    return out_dir


def _path_excluded(rel: Path) -> bool:
    """True if any path component or the full relative path matches any exclude pattern."""
    parts = rel.parts
    rel_str = str(rel)
    for pat in _OSS_EXCLUDE_PATTERNS:
        if pat in parts:
            return True
        if pat in rel_str:
            return True
        if rel.match(pat):
            return True
    return False


def _matches_include_pattern(path: str, pattern: str) -> bool:
    if fnmatch.fnmatchcase(path, pattern):
        return True
    if "/**/" in pattern:
        direct_pattern = pattern.replace("/**/", "/")
        return fnmatch.fnmatchcase(path, direct_pattern)
    return False


def selected_oss_paths(tracked_paths: list[str]) -> list[str]:
    """Select tracked git paths that belong in the tac OSS staging tree."""
    selected: list[str] = []
    for path in sorted(tracked_paths):
        rel = Path(path)
        if _path_excluded(rel):
            continue
        if any(_matches_include_pattern(path, pattern) for pattern in _OSS_INCLUDE_PATTERNS):
            selected.append(path)
    return selected


def stage_oss_publish(
    out_dir: Path,
    repo_root: Path,
    readme_path: Path | None = None,
    *,
    ref: str = "HEAD",
    strict_hygiene: bool = True,
) -> dict:
    """Materialize the OSS subset from git blobs; write MANIFEST.json + release scaffolding."""
    repo_root = repo_root.resolve()
    out_dir = _validate_stage_output_path(repo_root, out_dir)
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise SystemExit(
                f"FATAL: --out-dir {out_dir} is not empty. "
                "Refusing to clobber. Pick a fresh path or rm -rf first."
            )
    else:
        out_dir.mkdir(parents=True)

    files = selected_oss_paths(_git_lines(repo_root, ["ls-tree", "-r", "--name-only", ref]))
    if not files:
        raise SystemExit("FATAL: no files matched include patterns — sanity-check _OSS_INCLUDE_PATTERNS")

    copied: list[dict] = []
    for rel_str in files:
        data = _git_blob(repo_root, ref, rel_str)
        rel = Path(rel_str)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        copied.append({
            "path": rel_str,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        })

    # README
    readme_dst = out_dir / "README.md"
    if readme_path and readme_path.is_file():
        shutil.copy2(readme_path, readme_dst)
        readme_source = "operator_supplied_readme"
    else:
        readme_dst.write_text(_README_FALLBACK)
        readme_source = "fallback (operator should provide --readme)"

    # LICENSE
    license_dst = out_dir / "LICENSE"
    license_dst.write_text(_LICENSE_FALLBACK)

    # .gitignore
    gitignore_dst = out_dir / ".gitignore"
    gitignore_dst.write_text(_GITIGNORE)

    manifest = {
        "schema_version": 1,
        "produced_at_utc": dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "produced_by": "tools/oss_publish_staging.py",
        "source_ref": ref,
        "source_head": _git_lines(repo_root, ["rev-parse", ref])[0],
        "staging_root": "${TAC_OSS_STAGING_ROOT}",
        "n_files_copied": len(copied),
        "total_bytes_copied": sum(c["bytes"] for c in copied),
        "include_patterns": list(_OSS_INCLUDE_PATTERNS),
        "exclude_patterns": list(_OSS_EXCLUDE_PATTERNS),
        "readme_source": readme_source,
        "license": "Apache 2.0 (placeholder — operator should replace LICENSE with full text)",
        "next_steps": [
            "cd ${TAC_OSS_STAGING_ROOT}",
            "git init -b main",
            "git add .",
            'git commit -m "Initial public release of tac"',
            "gh repo create adpena/tac --public --source . --push",
        ],
    }
    manifest_dst = out_dir / "MANIFEST.json"
    write_json(manifest_dst, manifest)

    hygiene_violations = check_public_release_hygiene(
        repo_root=repo_root,
        strict=False,
        verbose=False,
        scan_paths=[out_dir],
    )
    if hygiene_violations and strict_hygiene:
        raise SystemExit(
            "PUBLIC RELEASE HYGIENE violations:\n"
            + "\n".join(f"  - {violation}" for violation in hygiene_violations[:40])
        )
    link_payload = audit_public_publish_links([out_dir], base_root=out_dir, live=False)
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
    write_json(manifest_dst, manifest)

    print(f"[oss-stage] copied {len(copied)} files ({sum(c['bytes'] for c in copied)} bytes total) → {out_dir}", file=sys.stderr)
    print(f"[oss-stage] wrote MANIFEST.json + README.md + LICENSE + .gitignore", file=sys.stderr)
    print(f"[oss-stage] readme source: {readme_source}", file=sys.stderr)
    print(f"[oss-stage] next steps printed in MANIFEST.json", file=sys.stderr)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Fresh out-of-tree staging directory (must be empty or non-existent).")
    parser.add_argument("--readme", type=Path, default=None,
                        help="Polished README markdown to copy as README.md. Falls back to placeholder.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help="Source repo root (default: auto-detected from this script's path).")
    parser.add_argument("--ref", default="HEAD",
                        help="Git ref to materialize from (default: HEAD).")
    parser.add_argument("--no-strict-hygiene", action="store_true",
                        help="Record public hygiene violations instead of failing.")
    args = parser.parse_args(argv)
    stage_oss_publish(
        args.out_dir,
        args.repo_root,
        readme_path=args.readme,
        ref=args.ref,
        strict_hygiene=not args.no_strict_hygiene,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
