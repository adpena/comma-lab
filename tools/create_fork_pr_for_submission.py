#!/usr/bin/env python3
"""Auto-create a draft fork PR with a submission_dir for GHA CPU eval.

Background — the bug class this fixes
─────────────────────────────────────
Memory ref: ``pr102_cpu_eval_gha_runtime_contract_failure_20260508_codex.md``
+ this session's roadmap iteration 2.

The fork's ``eval.yml`` workflow downloads only ``archive.zip`` from
``submission_url`` and then executes ``bash evaluate.sh
--submission-dir ./submissions/<submission_name>``. The runtime files,
especially ``inflate.sh``, must already exist in the checked-out fork. For
non-baseline submissions that means the dispatcher needs ``--pr-number``
pointing at a fork PR whose merge ref provides ``submissions/<name>/``.

Until now those fork PRs had to be created manually. This tool automates the
full create-fork-PR-with-submission_dir flow so a non-baseline GHA-CPU
dispatch becomes a single shell call.

Workflow
────────
1. Validate ``submission_dir`` contains ``inflate.sh``.
2. Clone the fork into a fresh tmpdir.
3. Branch ``add-submission-<name>-<timestamp>`` off ``main``.
4. Copy ``submission_dir/*`` to ``submissions/<name>/`` on the branch.
5. Commit + push to the fork.
6. Open a draft PR via ``gh pr create --draft``.
7. Print the PR number to stdout.
8. Write a manifest JSON for forensic traceability.

Operator selected this fix path in /loop iteration 2 of session 2026-05-08
("Auto-create fork PRs"). Public-release hygiene still applies: only the
runtime files required for the GHA CPU eval should be copied into the fork.
``--draft`` is the default because draft PRs are easier to clean up after the
eval completes, but the operator can pass ``--no-draft`` when intentional.

Usage
─────
    python tools/create_fork_pr_for_submission.py \\
        --submission-dir experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T091112Z/submission_dir \\
        --submission-name pr101_lossy_coarsening_admm_step6_20260508 \\
        --output-dir experiments/results/fork_pr_lossy_coarsening_20260508/

Returns the PR number on stdout (e.g. ``12``). Manifest at
``<output-dir>/fork_pr_manifest.json`` records every artifact for audit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_FORK_REPO = "adpena/comma_video_compression_challenge"
DEFAULT_BASE_BRANCH = "master"
EXCLUDED_SUBMISSION_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
EXCLUDED_SUBMISSION_FILE_NAMES = frozenset({".DS_Store"})
EXCLUDED_SUBMISSION_SUFFIXES = frozenset({".pyc", ".pyo"})


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Thin subprocess wrapper with consistent error surfacing."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        sys.stderr.write(
            f"[command-failed] {' '.join(cmd)} (cwd={cwd}, rc={result.returncode})\n"
            f"  stdout: {result.stdout!r}\n  stderr: {result.stderr!r}\n"
        )
        sys.exit(result.returncode or 4)
    return result


def now_utc_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_submission_dir(submission_dir: Path) -> None:
    """Refuse if the submission_dir is missing inflate.sh."""
    if not submission_dir.exists():
        sys.stderr.write(f"[fatal] submission_dir does not exist: {submission_dir}\n")
        sys.exit(2)
    if not submission_dir.is_dir():
        sys.stderr.write(f"[fatal] submission_dir is not a directory: {submission_dir}\n")
        sys.exit(2)
    inflate_sh = submission_dir / "inflate.sh"
    if not inflate_sh.is_file():
        sys.stderr.write(
            f"[fatal] submission_dir missing inflate.sh: {submission_dir}\n"
            f"        The GHA workflow expects submissions/<name>/inflate.sh "
            f"to be present in the fork checkout.\n"
        )
        sys.exit(2)


def clone_fork(fork_repo: str, target_dir: Path) -> None:
    """gh repo clone the fork into target_dir."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["gh", "repo", "clone", fork_repo, str(target_dir)])


def branch_exists_remote(clone_dir: Path, branch: str) -> bool:
    """Return True if the branch already exists on origin (the fork)."""
    res = run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=clone_dir,
        check=False,
    )
    return bool(res.stdout.strip())


def find_existing_pr(fork_repo: str, branch: str) -> int | None:
    """Return open PR number for branch on fork, or None."""
    res = run(
        [
            "gh", "pr", "list",
            "--repo", fork_repo,
            "--head", branch,
            "--json", "number,state",
            "--state", "open",
        ],
        check=False,
    )
    try:
        prs = json.loads(res.stdout or "[]")
    except json.JSONDecodeError:
        return None
    for pr in prs:
        if pr.get("state") == "OPEN":
            return int(pr["number"])
    return None


def should_copy_submission_path(path: Path) -> bool:
    """Return whether a runtime file should be copied into the fork PR.

    The GHA fork PR is a public runtime surface, not a local custody snapshot.
    Rebuildable caches and host-specific metadata must be filtered at the
    canonical helper so every caller gets the same hygiene guarantee.
    """

    if any(part in EXCLUDED_SUBMISSION_DIR_NAMES for part in path.parts):
        return False
    if path.name in EXCLUDED_SUBMISSION_FILE_NAMES:
        return False
    if path.suffix in EXCLUDED_SUBMISSION_SUFFIXES:
        return False
    return True


def _copy_submission_tree(src: Path, dst: Path, *, root: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir(), key=lambda item: item.name):
        rel = entry.relative_to(root)
        if not should_copy_submission_path(rel):
            continue
        target = dst / entry.name
        if entry.is_dir():
            _copy_submission_tree(entry, target, root=root)
        else:
            shutil.copy2(entry, target)


def copy_submission_dir(src: Path, clone_dir: Path, submission_name: str) -> Path:
    """Copy src/* to <clone_dir>/submissions/<submission_name>/."""
    target = clone_dir / "submissions" / submission_name
    if target.exists():
        # Preserve repository state by removing to ensure clean copy.
        shutil.rmtree(target)
    target.mkdir(parents=True)
    _copy_submission_tree(src, target, root=src)
    return target


def detect_default_branch(clone_dir: Path) -> str:
    """Return the default branch of origin (main or master)."""
    res = run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=clone_dir,
        check=False,
    )
    out = (res.stdout or "").strip()
    # e.g. refs/remotes/origin/master → master
    if out.startswith("refs/remotes/origin/"):
        return out[len("refs/remotes/origin/") :]
    return DEFAULT_BASE_BRANCH


def build_pr_body(
    submission_name: str,
    archive_path: Path | None,
    archive_sha256: str | None,
) -> str:
    """Minimal PR body — submission name + (optional) archive SHA for traceability."""
    lines = [
        f"Adds `submissions/{submission_name}/` for GHA CPU auth eval workflow.",
        "",
        f"- created_at_utc: {now_utc_iso()}",
        "- purpose: enable `--submission-name " + submission_name + "` in eval.yml workflow_dispatch",
    ]
    if archive_path is not None:
        lines.append(f"- intended_archive: {archive_path.name}")
    if archive_sha256:
        lines.append(f"- archive_sha256: {archive_sha256}")
    lines.append("")
    lines.append("Auto-created by `tools/create_fork_pr_for_submission.py`.")
    return "\n".join(lines)


def open_pr(
    fork_repo: str,
    branch: str,
    base_branch: str,
    title: str,
    body: str,
    draft: bool,
) -> int:
    """gh pr create returning the PR number."""
    cmd = [
        "gh", "pr", "create",
        "--repo", fork_repo,
        "--head", branch,
        "--base", base_branch,
        "--title", title,
        "--body", body,
    ]
    if draft:
        cmd.append("--draft")
    res = run(cmd)
    # gh pr create prints the PR URL on success; extract the number.
    url = (res.stdout or "").strip().splitlines()[-1] if res.stdout else ""
    pr_number = parse_pr_number_from_url(url)
    if pr_number is None:
        sys.stderr.write(f"[fatal] could not parse PR number from gh output: {url!r}\n")
        sys.exit(4)
    return pr_number


def parse_pr_number_from_url(url: str) -> int | None:
    """Extract the trailing /pull/<N> integer from a gh PR URL."""
    parts = url.rstrip("/").rsplit("/", 1)
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1])
    except (ValueError, TypeError):
        return None


def write_manifest(
    output_dir: Path,
    *,
    fork_repo: str,
    submission_name: str,
    branch: str,
    pr_number: int,
    pr_url: str,
    base_branch: str,
    submission_dir_src: Path,
    archive_path: Path | None,
    archive_sha256: str | None,
    reused_existing_pr: bool,
    draft: bool,
) -> Path:
    """Write fork_pr_manifest.json for forensic traceability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "fork_pr_manifest.json"
    manifest = {
        "tool": "tools/create_fork_pr_for_submission.py",
        "created_at_utc": now_utc_iso(),
        "fork_repo": fork_repo,
        "submission_name": submission_name,
        "branch": branch,
        "base_branch": base_branch,
        "pr_number": pr_number,
        "pr_url": pr_url,
        "draft": draft,
        "reused_existing_pr": reused_existing_pr,
        "submission_dir_src_relpath": str(submission_dir_src),
        "intended_archive_relpath": (
            str(archive_path) if archive_path is not None else None
        ),
        "intended_archive_sha256": archive_sha256,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def derive_branch_name(submission_name: str, timestamp: str) -> str:
    return f"add-submission-{submission_name}-{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--submission-dir", required=True, type=Path,
        help="Local path to submission directory containing inflate.sh.",
    )
    parser.add_argument(
        "--submission-name", required=True,
        help="Target submission name in the fork (will appear at submissions/<name>/).",
    )
    parser.add_argument(
        "--fork-repo", default=DEFAULT_FORK_REPO,
        help=f"Fork repo slug (default: {DEFAULT_FORK_REPO}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write fork_pr_manifest.json (default: experiments/results/fork_pr_<name>_<timestamp>/).",
    )
    parser.add_argument(
        "--base-branch", default=None,
        help="Base branch for the PR (default: auto-detect from origin/HEAD).",
    )
    parser.add_argument(
        "--branch-name", default=None,
        help="Branch name (default: add-submission-<name>-<timestamp>).",
    )
    parser.add_argument(
        "--no-draft", action="store_true",
        help="Open as a regular (non-draft) PR. Default is draft for easier cleanup post-eval.",
    )
    parser.add_argument(
        "--archive-path", type=Path, default=None,
        help="Optional intended-archive path (recorded in PR body for traceability).",
    )
    parser.add_argument(
        "--archive-sha256", default=None,
        help="Optional SHA-256 of the intended archive (recorded in PR body for traceability).",
    )
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="If a branch + PR with same name already exists, reuse the existing PR number "
             "instead of failing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without cloning, branching, or pushing.",
    )
    parser.add_argument(
        "--commit-message", default=None,
        help="Override the commit message (default: 'add submissions/<name>/ for GHA CPU eval').",
    )
    args = parser.parse_args()

    submission_dir: Path = args.submission_dir.resolve()
    submission_name: str = args.submission_name
    timestamp = now_utc_compact()

    if args.output_dir is None:
        args.output_dir = (
            Path("experiments/results") / f"fork_pr_{submission_name}_{timestamp}"
        )
    output_dir: Path = args.output_dir.resolve()

    branch = args.branch_name or derive_branch_name(submission_name, timestamp)
    commit_message = (
        args.commit_message
        or f"add submissions/{submission_name}/ for GHA CPU eval"
    )

    validate_submission_dir(submission_dir)

    if args.dry_run:
        print(
            f"[dry-run] would clone {args.fork_repo} → tmpdir, "
            f"branch={branch}, commit={commit_message!r}, "
            f"submission_name={submission_name}, draft={not args.no_draft}, "
            f"output_dir={output_dir}"
        )
        return 0

    pr_url = ""
    pr_number: int | None = None
    reused_existing = False
    base_branch = args.base_branch

    with tempfile.TemporaryDirectory(prefix="fork_pr_clone_") as tmp:
        clone_dir = Path(tmp) / "fork_clone"
        clone_fork(args.fork_repo, clone_dir)

        # Resolve base branch from origin/HEAD if not specified.
        if base_branch is None:
            base_branch = detect_default_branch(clone_dir)

        # Check for existing branch on origin (the fork).
        if branch_exists_remote(clone_dir, branch):
            if args.reuse_existing:
                existing = find_existing_pr(args.fork_repo, branch)
                if existing is not None:
                    sys.stderr.write(
                        f"[reuse] branch {branch!r} + open PR #{existing} already exist; "
                        f"reusing.\n"
                    )
                    pr_number = existing
                    pr_url = f"https://github.com/{args.fork_repo}/pull/{existing}"
                    reused_existing = True
                else:
                    sys.stderr.write(
                        f"[fatal] branch {branch!r} exists on origin but no open PR found. "
                        f"Either delete the branch on the fork or pass a different "
                        f"--branch-name.\n"
                    )
                    sys.exit(4)
            else:
                sys.stderr.write(
                    f"[fatal] branch {branch!r} already exists on {args.fork_repo}. "
                    f"Pass --reuse-existing to reuse the existing PR, or "
                    f"--branch-name to pick a different name.\n"
                )
                sys.exit(4)

        if pr_number is None:
            # Fresh branch path: branch off base, copy submission_dir, commit, push, PR.
            run(["git", "checkout", base_branch], cwd=clone_dir)
            run(["git", "checkout", "-b", branch], cwd=clone_dir)

            target_path = copy_submission_dir(submission_dir, clone_dir, submission_name)
            sys.stderr.write(f"[copied] {submission_dir} → {target_path}\n")

            run(["git", "add", f"submissions/{submission_name}"], cwd=clone_dir)
            commit_env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "claude-fork-pr-tool",
                "GIT_AUTHOR_EMAIL": "fork-pr-tool@noreply.local",
                "GIT_COMMITTER_NAME": "claude-fork-pr-tool",
                "GIT_COMMITTER_EMAIL": "fork-pr-tool@noreply.local",
            }
            commit_proc = subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=clone_dir, env=commit_env,
                capture_output=True, text=True,
            )
            if commit_proc.returncode != 0:
                sys.stderr.write(
                    f"[fatal] git commit failed: rc={commit_proc.returncode}\n"
                    f"  stdout: {commit_proc.stdout!r}\n  stderr: {commit_proc.stderr!r}\n"
                )
                sys.exit(commit_proc.returncode or 4)
            run(["git", "push", "-u", "origin", branch], cwd=clone_dir)

            title = f"add submissions/{submission_name}/ for GHA CPU eval"
            body = build_pr_body(
                submission_name, args.archive_path, args.archive_sha256,
            )
            pr_number = open_pr(
                args.fork_repo, branch, base_branch, title, body,
                draft=not args.no_draft,
            )
            pr_url = f"https://github.com/{args.fork_repo}/pull/{pr_number}"

    manifest_path = write_manifest(
        output_dir,
        fork_repo=args.fork_repo,
        submission_name=submission_name,
        branch=branch,
        pr_number=pr_number,
        pr_url=pr_url,
        base_branch=base_branch or DEFAULT_BASE_BRANCH,
        submission_dir_src=submission_dir,
        archive_path=args.archive_path,
        archive_sha256=args.archive_sha256,
        reused_existing_pr=reused_existing,
        draft=not args.no_draft,
    )
    sys.stderr.write(f"[manifest] {manifest_path}\n")
    sys.stderr.write(f"[pr-url] {pr_url}\n")

    print(pr_number)
    return 0


if __name__ == "__main__":
    sys.exit(main())
