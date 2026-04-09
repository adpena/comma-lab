#!/usr/bin/env python3
"""Build static timeline data from git history across multiple machines.

Extracts markdown file history from local and remote git repos,
sanitizes sensitive data, and outputs a JSON bundle for the static viewer.

Usage:
    # Local only
    python tools/md_timeline_build.py --repo . --out tools/timeline_site/data.json

    # With remote machines
    python tools/md_timeline_build.py \
        --repo . \
        --remote bat00:/home/adpena/pact \
        --remote tertiary:/Users/adpena/Projects/pact \
        --out tools/timeline_site/data.json

    # With log file aggregation
    python tools/md_timeline_build.py \
        --repo . \
        --logs .ralph/run_log.md \
        --logs .omx/state/current_focus.md \
        --remote bat00:/home/adpena/pact \
        --out tools/timeline_site/data.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ── Sanitization ────────────────────────────────────────────────────────


# Patterns to redact from content and metadata
SANITIZE_PATTERNS = [
    # Email addresses
    (re.compile(r"[\w.+-]+@[\w.-]+\.\w+"), "[email]"),
    # IP addresses (private ranges)
    (re.compile(r"\b(?:192\.168|10\.\d+|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b"), "[ip]"),
    # Usernames in GitHub URLs, prose, etc.
    (re.compile(r"\badpena\b"), "[user]"),
    # Internal hostnames (bat00, tertiary, etc.)
    (re.compile(r"\bbat00(?:\.local)?\b"), "[host]"),
    (re.compile(r"\btertiary(?:\.local)?\b"), "[host]"),
    # SSH hostnames that look internal
    (re.compile(r"ssh\s+[\w.-]+(?:\.local|\.internal|\.lan)\b"), "ssh [host]"),
    # Absolute paths with usernames
    (re.compile(r"/(?:Users|home)/\w+/"), "/~/"),
    # SSH config references
    (re.compile(r"~/\.ssh/\w+"), "[ssh-config]"),
    # Targeted credential patterns (not overly broad)
    (re.compile(r"(?:Bearer|token|key|secret|password)[=:\s]+\S{16,}", re.IGNORECASE), "[credential]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[aws-key]"),
    (re.compile(r"\bKGAT_[A-Za-z0-9]{20,}\b"), "[api-token]"),
    # Hex strings 41+ chars (not git hashes which are exactly 40)
    (re.compile(r"\b[0-9a-f]{41,}\b", re.IGNORECASE), "[token]"),
]

# Paths to never include (even if tracked)
EXCLUDE_PATHS = {
    ".env", ".envrc", "credentials.json", "secrets.yaml",
    ".ssh/", ".gnupg/", ".aws/",
}


def sanitize_text(text: str) -> str:
    """Strip sensitive patterns from text."""
    for pattern, replacement in SANITIZE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def should_exclude_file(filepath: str) -> bool:
    """Check if a file path should be excluded from the bundle."""
    for excl in EXCLUDE_PATHS:
        if excl in filepath:
            return True
    return False


# ── Git Operations ──────────────────────────────────────────────────────


def run_git(args: list[str], cwd: str) -> str:
    """Run git command locally."""
    result = subprocess.run(
        ["git"] + args, cwd=cwd,
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip()


def run_remote_git(host: str, repo_path: str, args: list[str]) -> str:
    """Run git command on a remote machine via SSH."""
    cmd = f"cd {shlex.quote(repo_path)} && git {' '.join(shlex.quote(a) for a in args)}"
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, cmd],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  Warning: SSH to {host} failed: {result.stderr.strip()}", file=sys.stderr)
        return ""
    return result.stdout.strip()


@dataclass
class CommitInfo:
    hash: str
    short_hash: str
    date: str
    author: str
    message: str
    source: str  # machine name


@dataclass
class FileVersion:
    commit_hash: str
    content: str


@dataclass
class FileDiff:
    from_hash: str
    to_hash: str
    additions: int
    deletions: int
    diff_html: str


@dataclass
class FileTimeline:
    path: str
    commits: list[CommitInfo] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)  # hash -> content
    diffs: list[FileDiff] = field(default_factory=list)


def extract_commits(cwd: str, source: str = "local") -> list[CommitInfo]:
    """Extract all commits from a repo."""
    raw = run_git(
        ["log", "--all", "--format=%H|%ai|%an|%s", "--date=iso"],
        cwd,
    )
    commits = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commits.append(CommitInfo(
            hash=parts[0],
            short_hash=parts[0][:7],
            date=parts[1].strip(),
            author=sanitize_text(parts[2].strip()),
            message=sanitize_text(parts[3].strip()),
            source=source,
        ))
    return commits


def extract_remote_commits(host: str, repo_path: str) -> list[CommitInfo]:
    """Extract commits from a remote machine."""
    raw = run_remote_git(
        host, repo_path,
        ["log", "--all", "--format=%H|%ai|%an|%s", "--date=iso"],
    )
    commits = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commits.append(CommitInfo(
            hash=parts[0],
            short_hash=parts[0][:7],
            date=parts[1].strip(),
            author=sanitize_text(parts[2].strip()),
            message=sanitize_text(parts[3].strip()),
            source=host,
        ))
    return commits


def get_md_files(cwd: str) -> list[str]:
    """List all .md files tracked in the repo."""
    raw = run_git(["ls-tree", "-r", "--name-only", "HEAD"], cwd)
    return sorted(
        f for f in raw.splitlines()
        if f.endswith(".md") and not should_exclude_file(f)
    )


def get_file_commits(cwd: str, filepath: str, source: str = "local") -> list[CommitInfo]:
    """Get commits that touched a file."""
    raw = run_git(
        ["log", "--all", "--follow", "--format=%H|%ai|%an|%s", "--", filepath],
        cwd,
    )
    commits = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commits.append(CommitInfo(
            hash=parts[0],
            short_hash=parts[0][:7],
            date=parts[1].strip(),
            author=sanitize_text(parts[2].strip()),
            message=sanitize_text(parts[3].strip()),
            source=source,
        ))
    return commits


def get_file_content_at(cwd: str, commit_hash: str, filepath: str) -> str | None:
    """Get sanitized file content at a commit."""
    result = subprocess.run(
        ["git", "show", f"{commit_hash}:{filepath}"],
        cwd=cwd, capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return None
    return sanitize_text(result.stdout)


def compute_inline_diff(old_text: str, new_text: str) -> tuple[str, int, int] | None:
    """Compute an HTML inline diff between two text versions.

    Returns (html, additions, deletions) or None if no changes.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines, lineterm="",
    ))

    if not diff_lines:
        return None

    html_parts = []
    additions = 0
    deletions = 0
    for line in diff_lines:
        if line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("@@"):
            html_parts.append(f'<span class="hunk-header">{_esc(line)}</span>\n')
        elif line.startswith("+"):
            html_parts.append(f'<ins>{_esc(line[1:])}</ins>\n')
            additions += 1
        elif line.startswith("-"):
            html_parts.append(f'<del>{_esc(line[1:])}</del>\n')
            deletions += 1
        else:
            html_parts.append(_esc(line[1:] if line.startswith(" ") else line) + "\n")

    return "".join(html_parts), additions, deletions


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Build ───────────────────────────────────────────────────────────────


def build_timeline(
    repo_dir: str,
    remotes: list[tuple[str, str]],
    tracked_files: list[str] | None = None,
) -> dict:
    """Build the complete timeline data bundle.

    Args:
        repo_dir: local git repo path
        remotes: list of (hostname, remote_repo_path) tuples
        tracked_files: specific files to track (None = all .md files)
    """
    print("Extracting local git history...")
    all_commits = extract_commits(repo_dir, "local")
    print(f"  {len(all_commits)} local commits")

    for host, path in remotes:
        print(f"Extracting from {host}:{path}...")
        remote_commits = extract_remote_commits(host, path)
        print(f"  {len(remote_commits)} commits from {host}")
        all_commits.extend(remote_commits)

    # Deduplicate by hash, keep first (which preserves source info)
    seen_hashes = set()
    unique_commits = []
    for c in all_commits:
        if c.hash not in seen_hashes:
            seen_hashes.add(c.hash)
            unique_commits.append(c)

    # Sort by date descending
    unique_commits.sort(key=lambda c: c.date, reverse=True)

    # Get files to track
    if tracked_files is None:
        md_files = get_md_files(repo_dir)
    else:
        md_files = tracked_files

    print(f"Tracking {len(md_files)} markdown files")

    file_timelines = {}
    for filepath in md_files:
        print(f"  Processing {filepath}...")
        fc = get_file_commits(repo_dir, filepath, "local")
        if not fc:
            continue

        versions = {}
        diffs = []
        for i, commit in enumerate(fc):
            content = get_file_content_at(repo_dir, commit.hash, filepath)
            if content is not None:
                versions[commit.short_hash] = content

                # Compute diff against previous version
                if i + 1 < len(fc):
                    prev_content = get_file_content_at(repo_dir, fc[i + 1].hash, filepath)
                    if prev_content is not None:
                        result = compute_inline_diff(prev_content, content)
                        if result:
                            diff_html, adds, dels = result
                            diffs.append({
                                "from": fc[i + 1].short_hash,
                                "to": commit.short_hash,
                                "additions": adds,
                                "deletions": dels,
                                "html": diff_html,
                            })

        file_timelines[filepath] = {
            "commits": [asdict(c) for c in fc],
            "versions": versions,
            "diffs": diffs,
        }

    # Build the bundle
    bundle = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "sources": ["local"] + [h for h, _ in remotes],
        "total_commits": len(unique_commits),
        "commits": [asdict(c) for c in unique_commits[:100]],  # cap at 100
        "files": sorted(file_timelines.keys()),
        "timelines": file_timelines,
    }

    return bundle


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build static timeline data from git history"
    )
    parser.add_argument("--repo", default=".", help="Local git repo path")
    parser.add_argument(
        "--remote", action="append", default=[],
        help="Remote in host:path format (e.g., bat00:/home/user/pact)"
    )
    parser.add_argument(
        "--file", action="append", default=[],
        help="Specific files to track (default: all .md)"
    )
    parser.add_argument(
        "--out", default="tools/timeline_site/data.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--sanitize", action="store_true", default=True,
        help="Sanitize sensitive data (default: true)"
    )
    parser.add_argument(
        "--no-sanitize", action="store_false", dest="sanitize",
        help="Disable sanitization (for local preview)"
    )
    args = parser.parse_args()

    remotes = []
    for r in args.remote:
        if ":" not in r:
            print(f"Error: remote must be host:path format, got '{r}'", file=sys.stderr)
            sys.exit(1)
        host, path = r.split(":", 1)
        remotes.append((host, path))

    tracked_files = args.file if args.file else None

    bundle = build_timeline(
        repo_dir=os.path.abspath(args.repo),
        remotes=remotes,
        tracked_files=tracked_files,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2))

    size_kb = out_path.stat().st_size / 1024
    print(f"\nBundle written to {out_path} ({size_kb:.1f} KB)")
    print(f"  {bundle['total_commits']} commits, {len(bundle['files'])} files")
    print(f"  Sources: {', '.join(bundle['sources'])}")


if __name__ == "__main__":
    main()
