#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "graphs" / "report_history.json"
FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"
DEFAULT_GLOBS = [
    "reports/graphs/*.md",
    ".omx/state/*.md",
    ".omx/research/*.md",
    ".ralph/*.md",
    "reports/latest.md",
    "reports/lossless_latest.md",
]


def run_git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def categorize_path(rel_path: str) -> str:
    if rel_path.startswith("reports/graphs/"):
        return "report_graph"
    return "durable_state"


def resolve_default_surface_paths(repo_root: Path) -> list[str]:
    surfaces: set[str] = set()
    for pattern in DEFAULT_GLOBS:
        for candidate in repo_root.glob(pattern):
            if not candidate.is_file():
                continue
            rel_path = repo_relative(candidate, repo_root)
            if rel_path.startswith("reports/graphs/site/"):
                continue
            surfaces.add(rel_path)
    return sorted(surfaces)


def load_blob_text(repo_root: Path, commit: str, rel_path: str) -> tuple[str, bool]:
    result = run_git(repo_root, "show", f"{commit}:{rel_path}", check=False)
    if result.returncode != 0:
        return "", False
    return result.stdout, True


def collect_file_history(repo_root: Path, rel_path: str) -> list[dict[str, object]]:
    format_spec = (
        f"%H{FIELD_SEP}%P{FIELD_SEP}%ct{FIELD_SEP}%cI{FIELD_SEP}%an{FIELD_SEP}%ae"
        f"{FIELD_SEP}%s{FIELD_SEP}%b{RECORD_SEP}"
    )
    result = run_git(
        repo_root,
        "log",
        "--follow",
        f"--format={format_spec}",
        "--",
        rel_path,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    snapshots: list[dict[str, object]] = []
    for raw_record in result.stdout.split(RECORD_SEP):
        record = raw_record.strip("\n")
        if not record:
            continue
        fields = record.split(FIELD_SEP, 7)
        if len(fields) < 8:
            continue
        commit, parents, unix_time, iso_time, author_name, author_email, subject, body = fields
        content, content_available = load_blob_text(repo_root, commit, rel_path)
        normalized_content = content.rstrip("\n")
        snapshots.append(
            {
                "commit": commit,
                "short_commit": commit[:7],
                "parents": [item for item in parents.split() if item],
                "timestamp_unix": int(unix_time),
                "timestamp_utc": iso_time,
                "author_name": author_name,
                "author_email": author_email,
                "subject": subject,
                "body": body.strip(),
                "content": normalized_content,
                "content_available": content_available,
                "byte_length": len(content.encode("utf-8")),
                "line_count": 0 if not normalized_content else normalized_content.count("\n") + 1,
            }
        )
    return snapshots


def collect_history(repo_root: Path, *, include_paths: list[str] | None = None) -> dict[str, object]:
    surfaces = sorted(dict.fromkeys(include_paths or resolve_default_surface_paths(repo_root)))
    files: list[dict[str, object]] = []
    snapshot_count = 0
    for rel_path in surfaces:
        snapshots = collect_file_history(repo_root, rel_path)
        if not snapshots:
            continue
        snapshot_count += len(snapshots)
        files.append(
            {
                "path": rel_path,
                "category": categorize_path(rel_path),
                "snapshot_count": len(snapshots),
                "latest_timestamp_utc": snapshots[0]["timestamp_utc"],
                "snapshots": snapshots,
            }
        )

    head = run_git(repo_root, "rev-parse", "HEAD", check=False)
    repo_head = head.stdout.strip() if head.returncode == 0 else ""
    return {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_root": str(repo_root.resolve()),
        "repo_head": repo_head,
        "file_count": len(files),
        "snapshot_count": snapshot_count,
        "files": files,
    }


def write_history_json(
    repo_root: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT,
    include_paths: list[str] | None = None,
) -> dict[str, object]:
    payload = collect_history(repo_root, include_paths=include_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build git-backed report history JSON for the reports/graphs viewer.")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--path",
        action="append",
        dest="include_paths",
        default=None,
        help="repo-relative path to include; repeat to override default surfaces",
    )
    args = parser.parse_args()

    payload = write_history_json(
        args.repo_root.resolve(),
        output_path=args.output.resolve(),
        include_paths=args.include_paths,
    )
    print("report_history_ready", args.output.resolve(), payload["snapshot_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
