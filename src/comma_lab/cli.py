from __future__ import annotations

import argparse
import os
import subprocess
import shutil
import sys
from pathlib import Path

from .bootstrap import bootstrap_upstream
from .evaluate import evaluate_submission
from .install import install_submission
from .paths import default_upstream_root, repo_root, upstream_snapshot_path
from .smoke import smoke_submission
from .snapshot import load_snapshot
from .tracks.exact_current import create_minimal_archive


def cmd_bootstrap_upstream(args: argparse.Namespace) -> int:
    dest = Path(args.dest) if args.dest else default_upstream_root()
    snapshot = bootstrap_upstream(dest=dest, do_lfs=not args.no_lfs)
    print(f"Upstream ready at: {dest}")
    print(f"Snapshot written to: {upstream_snapshot_path()}")
    print(f"Commit: {snapshot.get('commit')}")
    print(f"Public test video names: {snapshot.get('public_test_video_names')}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    snap = load_snapshot(upstream_snapshot_path())
    print(f"Repo root: {repo_root()}")
    print(f"Default upstream root: {default_upstream_root()}")
    print(f"Snapshot file: {upstream_snapshot_path()}")
    if not snap:
        print("No upstream snapshot yet.")
        return 0

    print(f"Upstream commit: {snap.get('commit')}")
    print(f"Last verified at: {snap.get('last_verified_at')}")
    print(f"Public test video names: {snap.get('public_test_video_names')}")
    for rel, digest in snap.get("files", {}).items():
        print(f"  {rel}: {digest}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    tools = ["git", "python3", "curl", "ffmpeg", "git-lfs", "node", "npm", "uv", "omx"]
    print("Tool status:")
    for tool in tools:
        found = shutil.which(tool)
        print(f"  {tool:<10} {'OK' if found else 'MISSING'}{f' -> {found}' if found else ''}")

    prompt_path = repo_root() / "PROMPT.md"
    print(f"Prompt path: {prompt_path}")
    print(f"Prompt exists: {prompt_path.exists()}")
    print(f"Upstream root exists: {default_upstream_root().exists()}")
    print(f"Snapshot exists: {upstream_snapshot_path().exists()}")
    return 0


def cmd_install_submission(args: argparse.Namespace) -> int:
    dst = install_submission(
        args.name,
        upstream_root=Path(args.upstream_root) if args.upstream_root else None,
        force=True,
    )
    print(f"Installed submission '{args.name}' to: {dst}")
    return 0


def cmd_show_prompt(args: argparse.Namespace) -> int:
    if args.name == "top":
        prompt_path = repo_root() / "PROMPT.md"
    else:
        prompt_path = repo_root() / "prompts" / f"{args.name}.md"
    if not prompt_path.exists():
        print(f"Prompt not found: {prompt_path}", file=sys.stderr)
        return 1
    print(prompt_path.read_text())
    return 0



def cmd_eval_submission(args: argparse.Namespace) -> int:
    summary = evaluate_submission(
        args.name,
        device=args.device,
        upstream_root=Path(args.upstream_root) if args.upstream_root else None,
        sync=not args.no_sync,
        package=args.package,
        report_copy=Path(args.report_copy) if args.report_copy else None,
    )
    print(summary.to_json())
    return 0


def cmd_smoke_submission(args: argparse.Namespace) -> int:
    summary = smoke_submission(
        args.name,
        upstream_root=Path(args.upstream_root) if args.upstream_root else None,
        sync=not args.no_sync,
        package=args.package,
    )
    print(summary.to_json())
    return 0


def cmd_package_submission(args: argparse.Namespace) -> int:
    sub_dir = repo_root() / "submissions" / args.name
    if args.name == "exact_current":
        archive = create_minimal_archive(sub_dir / "archive.zip")
        print(f"Updated exact-current archive: {archive}")
        return 0
    if args.name == "robust_current":
        upstream_root = Path(args.upstream_root) if args.upstream_root else default_upstream_root()
        env = os.environ.copy()
        env["COMMA_CHALLENGE_ROOT"] = str(upstream_root)
        subprocess.run(["bash", str(sub_dir / "compress.sh")], cwd=repo_root(), env=env, check=True)
        archive = sub_dir / "archive.zip"
        print(f"Updated robust-current archive: {archive}")
        return 0
    print(f"No package action implemented for '{args.name}'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="comma video lab helper CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bootstrap-upstream", help="clone or refresh the upstream challenge repo and write a snapshot")
    p.add_argument("--dest", default=None, help="destination directory for the upstream repo")
    p.add_argument("--no-lfs", action="store_true", help="skip git-lfs pull")
    p.set_defaults(func=cmd_bootstrap_upstream)

    p = sub.add_parser("status", help="show local starter-pack status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="check local tool availability and key repo files")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("install-submission", help="copy a starter-pack submission into the upstream repo")
    p.add_argument("name", choices=["exact_current", "robust_current"])
    p.add_argument("--upstream-root", default=None)
    p.set_defaults(func=cmd_install_submission)

    p = sub.add_parser("show-prompt", help="print a prompt file")
    p.add_argument("name", help="prompt stem under prompts/, or 'top' for PROMPT.md")
    p.set_defaults(func=cmd_show_prompt)

    p = sub.add_parser("eval-submission", help="package/sync/evaluate a submission via the upstream .venv and print a JSON summary")
    p.add_argument("name", choices=["exact_current", "robust_current"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--upstream-root", default=None)
    p.add_argument("--package", action="store_true", help="package the submission before evaluation")
    p.add_argument("--no-sync", action="store_true", help="skip copying the submission into upstream before evaluation")
    p.add_argument("--report-copy", default=None, help="optional repo-local path to copy the raw report to after evaluation")
    p.set_defaults(func=cmd_eval_submission)

    p = sub.add_parser("smoke-submission", help="package/sync/inflate a submission and verify raw output count/geometry before scorer runs")
    p.add_argument("name", choices=["exact_current", "robust_current"])
    p.add_argument("--upstream-root", default=None)
    p.add_argument("--package", action="store_true", help="package the submission before the smoke check")
    p.add_argument("--no-sync", action="store_true", help="skip copying the submission into upstream before the smoke check")
    p.set_defaults(func=cmd_smoke_submission)

    p = sub.add_parser("package-submission", help="refresh a submission artifact in-place")
    p.add_argument("name", choices=["exact_current", "robust_current"])
    p.add_argument("--upstream-root", default=None, help="override the challenge root used during packaging")
    p.set_defaults(func=cmd_package_submission)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
