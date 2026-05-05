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
    README.md                          — generated from /tmp/tac_oss_readme_draft.md
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
        --out-dir /tmp/tac_oss_staging \\
        --readme /tmp/tac_oss_readme_draft.md

Then operator reviews:
    ls /tmp/tac_oss_staging/
    cat /tmp/tac_oss_staging/MANIFEST.json | jq .
    less /tmp/tac_oss_staging/README.md

Then publishes:
    cd /tmp/tac_oss_staging
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
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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
`/tmp/tac_oss_readme_draft.md` from the prior session work.)

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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _expand_include_patterns(repo_root: Path) -> list[Path]:
    """Resolve include patterns to a deduplicated, sorted list of file paths."""
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in _OSS_INCLUDE_PATTERNS:
        for p in sorted(repo_root.glob(pat)):
            if not p.is_file():
                continue
            rel = p.relative_to(repo_root)
            if _path_excluded(rel):
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def stage_oss_publish(out_dir: Path, repo_root: Path, readme_path: Path | None = None) -> dict:
    """Copy the OSS subset into out_dir; write MANIFEST.json + README + LICENSE + .gitignore."""
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise SystemExit(
                f"FATAL: --out-dir {out_dir} is not empty. "
                "Refusing to clobber. Pick a fresh path or rm -rf first."
            )
    else:
        out_dir.mkdir(parents=True)

    files = _expand_include_patterns(repo_root)
    if not files:
        raise SystemExit("FATAL: no files matched include patterns — sanity-check _OSS_INCLUDE_PATTERNS")

    copied: list[dict] = []
    for src in files:
        rel = src.relative_to(repo_root)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({
            "path": str(rel),
            "bytes": dst.stat().st_size,
            "sha256": _sha256_file(dst),
        })

    # README
    readme_dst = out_dir / "README.md"
    if readme_path and readme_path.is_file():
        shutil.copy2(readme_path, readme_dst)
        readme_source = str(readme_path)
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
        "out_dir": str(out_dir),
        "repo_root": str(repo_root),
        "n_files_copied": len(copied),
        "total_bytes_copied": sum(c["bytes"] for c in copied),
        "include_patterns": list(_OSS_INCLUDE_PATTERNS),
        "exclude_patterns": list(_OSS_EXCLUDE_PATTERNS),
        "readme_source": readme_source,
        "license": "Apache 2.0 (placeholder — operator should replace LICENSE with full text)",
        "next_steps": [
            f"cd {out_dir}",
            "git init -b main",
            "git add .",
            'git commit -m "Initial public release of tac"',
            "gh repo create adpena/tac --public --source . --push",
        ],
    }
    manifest_dst = out_dir / "MANIFEST.json"
    manifest_dst.write_text(json.dumps(manifest, indent=2))

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
    args = parser.parse_args(argv)
    stage_oss_publish(args.out_dir, args.repo_root, readme_path=args.readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
