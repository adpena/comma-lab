from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path

from .bootstrap import bootstrap_upstream
from .evaluate import evaluate_submission
from .install import install_submission
from .lock import submission_lock
from .lossless_review_tracker import (
    doctor_repo as lossless_review_doctor_repo,
    scan_repo as lossless_review_scan_repo,
    status_payload as lossless_review_status_payload,
    sync_repo as lossless_review_sync_repo,
)
from .lossless_state_sync import (
    doctor_repo as lossless_doctor_repo,
    promote_record as lossless_promote_record,
    sync_repo as lossless_sync_repo,
)
from .paths import default_upstream_root, repo_root, upstream_snapshot_path
from .scheduler.registry import load_platform_registry
from .scheduler.reporting import build_budget_report, build_status_report, select_result_records
from .scheduler.repository import collect_run_records
from .smoke import smoke_submission
from .snapshot import load_snapshot
from .state_sync import doctor_repo, promote_record, sync_repo
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
    upstream_root = Path(args.upstream_root) if args.upstream_root else default_upstream_root()
    with submission_lock(args.name, upstream_root):
        if args.name == "exact_current":
            archive = create_minimal_archive(sub_dir / "archive.zip")
            print(f"Updated exact-current archive: {archive}")
            return 0
        if args.name == "robust_current":
            env = os.environ.copy()
            env["COMMA_CHALLENGE_ROOT"] = str(upstream_root)
            subprocess.run(["bash", str(sub_dir / "compress.sh")], cwd=repo_root(), env=env, check=True)
            archive = sub_dir / "archive.zip"
            print(f"Updated robust-current archive: {archive}")
            return 0
        print(f"No package action implemented for '{args.name}'.")
        return 0


def _scheduler_repo_root(args: argparse.Namespace) -> Path:
    return Path(args.repo_root) if getattr(args, "repo_root", None) else repo_root()


def _scheduler_registry_path(root: Path, explicit_path: str | None) -> Path | None:
    if explicit_path:
        return Path(explicit_path)
    default_path = root / "configs" / "platforms.json"
    if default_path.exists():
        return default_path
    return None


def _load_scheduler_registry(args: argparse.Namespace, *, required: bool) -> object:
    root = _scheduler_repo_root(args)
    registry_path = _scheduler_registry_path(root, getattr(args, "registry", None))
    if registry_path is None:
        if required:
            raise ValueError("`sched budget` requires --registry or configs/platforms.json")
        return None
    return load_platform_registry(registry_path)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_sched_status(args: argparse.Namespace) -> int:
    root = _scheduler_repo_root(args)
    registry = _load_scheduler_registry(args, required=False)
    records = collect_run_records(root, registry)
    report = build_status_report(root, records)
    if args.json:
        _print_json(report.to_dict())
        return 0

    print(f"Repo root: {root}")
    print(f"Measured results: {report.result_count}")
    print(f"Total run records: {report.run_record_count}")
    if not report.tracks:
        print("No measured results found.")
    else:
        print("Latest by track:")
        for item in report.tracks:
            latest = item.latest_result
            score = "n/a" if latest.score is None else f"{latest.score:.4f}"
            print(f"  {item.track}: {score} ({latest.run_id})")
    if report.active_runs:
        print("Active runs:")
        for run in report.active_runs:
            print(f"  {run.platform}: {run.run_id} [{run.status}]")
    return 0


def cmd_sched_results(args: argparse.Namespace) -> int:
    root = _scheduler_repo_root(args)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    registry = _load_scheduler_registry(args, required=False)
    records = collect_run_records(root, registry)
    results = select_result_records(records, track=args.track, limit=args.limit)
    if args.json:
        _print_json({"results": [record.to_dict() for record in results]})
        return 0

    if not results:
        print("No matching measured results found.")
        return 0
    for record in results:
        score = "n/a" if record.score is None else f"{record.score:.4f}"
        finished = record.finished_at or "unknown-time"
        track = record.track or "unknown-track"
        print(f"{finished}  {track:<16} {record.platform:<8} {score:<8} {record.run_id}")
    return 0


def cmd_sched_budget(args: argparse.Namespace) -> int:
    root = _scheduler_repo_root(args)
    registry = _load_scheduler_registry(args, required=True)
    records = collect_run_records(root, registry)
    report = build_budget_report(registry, records)
    if args.json:
        _print_json(report.to_dict())
        return 0

    for name in sorted(report.platforms):
        item = report.platforms[name]
        status = "OVER" if item.over_budget else "OK"
        print(f"{name} ({item.platform.kind}) [{status}]")
        print(
            "  "
            f"runs={item.usage.total_runs} active={item.usage.active_runs} "
            f"failed={item.usage.failed_runs} archive_bytes={item.usage.archive_bytes}"
        )
    return 0


def cmd_state_doctor(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    report = doctor_repo(root)
    if args.json:
        _print_json(report.to_dict())
        return 0

    if not report.findings:
        print("state doctor: no drift found")
        return 0
    print(f"state doctor: {len(report.findings)} finding(s)")
    for finding in report.findings:
        print(f"  [{finding.severity}] {finding.code} {finding.path}: {finding.message}")
    return 0


def cmd_state_sync(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    result = sync_repo(root)
    if args.json:
        _print_json(result.to_dict())
        return 0

    print(f"state sync: changed {len(result.changed_paths)} path(s)")
    for path in result.changed_paths:
        print(f"  {path}")
    return 0


def cmd_state_promote(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    result = promote_record(root, record_path=args.record)
    if args.json:
        _print_json(result.to_dict())
        return 0

    print(f"state promote: changed {len(result.changed_paths)} path(s)")
    for path in result.changed_paths:
        print(f"  {path}")
    return 0


def cmd_lossless_state_doctor(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    report = lossless_doctor_repo(root)
    if args.json:
        _print_json(report.to_dict())
        return 0

    if not report.findings:
        print("lossless-state doctor: no drift found")
        return 0
    print(f"lossless-state doctor: {len(report.findings)} finding(s)")
    for finding in report.findings:
        print(f"  [{finding.severity}] {finding.code} {finding.path}: {finding.message}")
    return 0


def cmd_lossless_state_sync(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    result = lossless_sync_repo(root)
    if args.json:
        _print_json(result.to_dict())
        return 0

    print(f"lossless-state sync: changed {len(result.changed_paths)} path(s)")
    for path in result.changed_paths:
        print(f"  {path}")
    return 0


def cmd_lossless_state_promote(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    result = lossless_promote_record(root, record_path=args.record)
    if args.json:
        _print_json(result.to_dict())
        return 0

    print(f"lossless-state promote: changed {len(result.changed_paths)} path(s)")
    for path in result.changed_paths:
        print(f"  {path}")
    return 0


def cmd_lossless_review_doctor(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    report = lossless_review_doctor_repo(root)
    if args.json:
        _print_json(report.to_dict())
        return 0

    if not report.findings:
        print("lossless-review doctor: no drift found")
        return 0
    print(f"lossless-review doctor: {len(report.findings)} finding(s)")
    for finding in report.findings:
        print(f"  [{finding.severity}] {finding.code} {finding.path}: {finding.message}")
    return 0


def cmd_lossless_review_sync(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    result = lossless_review_sync_repo(root)
    if args.json:
        _print_json(result.to_dict())
        return 0

    print(f"lossless-review sync: changed {len(result.changed_paths)} path(s)")
    for path in result.changed_paths:
        print(f"  {path}")
    return 0


def cmd_lossless_review_scan(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    result = lossless_review_scan_repo(root)
    if args.json:
        _print_json(result.to_dict())
        return 0

    print(f"lossless-review scan: changed {len(result.changed_paths)} path(s)")
    for path in result.changed_paths:
        print(f"  {path}")
    return 0


def cmd_lossless_review_status(args: argparse.Namespace) -> int:
    root = Path(args.repo_root) if args.repo_root else repo_root()
    payload = lossless_review_status_payload(root)
    if args.json:
        _print_json(payload)
        return 0

    counts = payload["counts"]
    print(f"lossless-review status: {payload['tracker_path']}")
    print(f"  last_scan={payload['last_scan']}")
    print(
        "  "
        f"total={counts['total']} reviewed={counts['reviewed']} "
        f"unreviewed={counts['unreviewed']} stale={counts['stale']} needs_fix={counts['needs_fix']}"
    )
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

    p = sub.add_parser("smoke-submission", help="package/sync/inflate a submission and verify raw output count, geometry, and sampled RGB semantics before scorer runs")
    p.add_argument("name", choices=["exact_current", "robust_current"])
    p.add_argument("--upstream-root", default=None)
    p.add_argument("--package", action="store_true", help="package the submission before the smoke check")
    p.add_argument("--no-sync", action="store_true", help="skip copying the submission into upstream before the smoke check")
    p.set_defaults(func=cmd_smoke_submission)

    p = sub.add_parser("package-submission", help="refresh a submission artifact in-place")
    p.add_argument("name", choices=["exact_current", "robust_current"])
    p.add_argument("--upstream-root", default=None, help="override the challenge root used during packaging")
    p.set_defaults(func=cmd_package_submission)

    p = sub.add_parser("sched", help="inspect scheduler-friendly repo state")
    sched_sub = p.add_subparsers(dest="sched_cmd", required=True)

    sp = sched_sub.add_parser("status", help="summarize latest measured results and active discovered runs")
    sp.add_argument("--repo-root", default=None, help="override the repo root to inspect")
    sp.add_argument("--registry", default=None, help="platform registry path; defaults to configs/platforms.json when present")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_sched_status)

    sp = sched_sub.add_parser("results", help="list measured scorer results from reports/results.jsonl")
    sp.add_argument("--repo-root", default=None, help="override the repo root to inspect")
    sp.add_argument("--registry", default=None, help="optional platform registry path for device-to-platform mapping")
    sp.add_argument("--track", default=None, help="optional track filter")
    sp.add_argument("--limit", type=int, default=10, help="maximum number of results to show")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_sched_results)

    sp = sched_sub.add_parser("budget", help="compare discovered run usage against a platform registry budget")
    sp.add_argument("--repo-root", default=None, help="override the repo root to inspect")
    sp.add_argument("--registry", default=None, help="platform registry path; defaults to configs/platforms.json when present")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_sched_budget)

    p = sub.add_parser("state", help="inspect and repair canonical promoted state")
    state_sub = p.add_subparsers(dest="state_cmd", required=True)

    sp = state_sub.add_parser("doctor", help="audit promoted-state drift and stale managed-session manifests")
    sp.add_argument("--repo-root", default=None, help="override the repo root to inspect")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_state_doctor)

    sp = state_sub.add_parser("sync", help="reproject canonical promoted truth into summaries, ledgers, and docs")
    sp.add_argument("--repo-root", default=None, help="override the repo root to repair")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_state_sync)

    sp = state_sub.add_parser("promote", help="validate a promoted-result record, install it as canonical state, and sync mirrors")
    sp.add_argument("--repo-root", default=None, help="override the repo root to repair")
    sp.add_argument("--record", default=None, help="path to a promoted-result JSON record; defaults to .omx/state/promoted_result.json")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_state_promote)

    p = sub.add_parser("lossless-state", help="inspect and repair canonical lossless promoted state")
    lossless_state_sub = p.add_subparsers(dest="lossless_state_cmd", required=True)

    sp = lossless_state_sub.add_parser("doctor", help="audit lossless promoted-state drift")
    sp.add_argument("--repo-root", default=None, help="override the repo root to inspect")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_lossless_state_doctor)

    sp = lossless_state_sub.add_parser("sync", help="reproject canonical lossless promoted truth into ledgers and docs")
    sp.add_argument("--repo-root", default=None, help="override the repo root to repair")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_lossless_state_sync)

    sp = lossless_state_sub.add_parser(
        "promote",
        help="validate a lossless promoted-result record, install it as canonical state, and sync mirrors",
    )
    sp.add_argument("--repo-root", default=None, help="override the repo root to repair")
    sp.add_argument(
        "--record",
        default=None,
        help="path to a lossless promoted-result JSON record; defaults to .omx/state/lossless_promoted_result.json",
    )
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_lossless_state_promote)

    p = sub.add_parser("lossless-review", help="project the hardened review tracker onto the lossless slice")
    lossless_review_sub = p.add_subparsers(dest="lossless_review_cmd", required=True)

    sp = lossless_review_sub.add_parser("doctor", help="audit lossless review tracker drift")
    sp.add_argument("--repo-root", default=None, help="override the repo root to inspect")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_lossless_review_doctor)

    sp = lossless_review_sub.add_parser("sync", help="refresh the lossless review tracker projection from the global tracker")
    sp.add_argument("--repo-root", default=None, help="override the repo root to repair")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_lossless_review_sync)

    sp = lossless_review_sub.add_parser("scan", help="rescan the global review tracker, then refresh the lossless projection")
    sp.add_argument("--repo-root", default=None, help="override the repo root to repair")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_lossless_review_scan)

    sp = lossless_review_sub.add_parser("status", help="summarize the lossless review tracker projection")
    sp.add_argument("--repo-root", default=None, help="override the repo root to inspect")
    sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sp.set_defaults(func=cmd_lossless_review_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
