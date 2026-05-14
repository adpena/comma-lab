# SPDX-License-Identifier: MIT
"""Reusable reproducibility metadata for contest packets and reports."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from tac.repo_io import repo_relative, sha256_bytes, sha256_file


def normalize_git_remote_url(remote_url: str | None) -> str | None:
    """Return a browser-friendly repository URL when the remote is recognized."""

    if not remote_url:
        return None
    remote = remote_url.strip()
    if not remote:
        return None
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.removeprefix("git@github.com:")
    if remote.startswith("ssh://git@github.com/"):
        remote = "https://github.com/" + remote.removeprefix("ssh://git@github.com/")
    if remote.endswith(".git"):
        remote = remote[:-4]
    return remote


def _run_git(repo_root: Path, args: Sequence[str], *, binary: bool = False) -> str | bytes | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_text(repo_root: Path, args: Sequence[str]) -> str | None:
    output = _run_git(repo_root, args, binary=False)
    if output is None:
        return None
    return str(output).strip()


def _git_bytes(repo_root: Path, args: Sequence[str]) -> bytes:
    output = _run_git(repo_root, args, binary=True)
    if output is None:
        return b""
    return bytes(output)


def _path_record(path: Path, repo_root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": repo_relative(path, repo_root),
        "exists": path.exists(),
    }
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return record


def collect_source_transparency(
    *,
    repo_root: str | Path,
    source_paths: Iterable[str | Path] = (),
    artifact_paths: Iterable[str | Path] = (),
    commands: Iterable[Sequence[str] | str] = (),
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Collect source, artifact, and command custody for a generated packet.

    The returned object is intentionally conservative: dirty and untracked
    source states are surfaced rather than hidden, and proxy/eval tooling can
    embed it without making a score claim.
    """

    root = Path(repo_root).resolve()
    generated_at = generated_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    head_commit = _git_text(root, ["rev-parse", "HEAD"])
    branch = _git_text(root, ["branch", "--show-current"])
    remote = _git_text(root, ["config", "--get", "remote.origin.url"])
    online_url = normalize_git_remote_url(remote)
    status_text = _git_text(root, ["status", "--porcelain=v1"]) or ""
    status_lines = [line for line in status_text.splitlines() if line]
    staged_diff = _git_bytes(root, ["diff", "--cached", "--binary"])
    unstaged_diff = _git_bytes(root, ["diff", "--binary"])
    tracked_diff_blob = staged_diff + b"\n---unstaged---\n" + unstaged_diff
    fingerprint_blob = status_text.encode("utf-8") + b"\n---diff---\n" + tracked_diff_blob

    normalized_commands: list[list[str]] = []
    for command in commands:
        if isinstance(command, str):
            normalized_commands.append([command])
        else:
            normalized_commands.append([str(part) for part in command])

    commit_url = f"{online_url}/commit/{head_commit}" if online_url and head_commit else None
    tree_url = f"{online_url}/tree/{head_commit}" if online_url and head_commit else None
    untracked = [line[3:] for line in status_lines if line.startswith("?? ")]

    return {
        "schema": "tac_source_transparency_v1",
        "generated_at_utc": generated_at,
        "repo": {
            "root": repo_relative(root, root),
            "remote_origin_url": remote,
            "online_url": online_url,
            "head_commit": head_commit,
            "branch": branch,
            "dirty": bool(status_lines),
            "status_porcelain": status_lines,
            "untracked_files": untracked,
            "tracked_dirty_diff_sha256": sha256_bytes(tracked_diff_blob),
            "working_tree_fingerprint_sha256": sha256_bytes(fingerprint_blob),
        },
        "online_links": {
            "repository": online_url,
            "commit": commit_url,
            "tree": tree_url,
        },
        "source_paths": [_path_record(Path(path), root) for path in source_paths],
        "artifact_paths": [_path_record(Path(path), root) for path in artifact_paths],
        "reproduction_commands": normalized_commands,
        "release_contract": {
            "include_in_submission_packets": True,
            "include_in_writeups": True,
            "include_in_reports": True,
            "deterministic_reproduction_required": True,
            "score_claim_from_metadata": False,
        },
    }


def transparency_report_markdown(transparency: dict[str, Any]) -> str:
    """Render a compact Markdown block for packet READMEs and writeups."""

    repo = transparency.get("repo", {})
    links = transparency.get("online_links", {})
    lines = [
        "## Source Transparency",
        "",
        f"- repo: `{repo.get('remote_origin_url') or 'unknown'}`",
        f"- online: `{links.get('repository') or 'unavailable'}`",
        f"- commit: `{repo.get('head_commit') or 'unknown'}`",
        f"- commit_url: `{links.get('commit') or 'unavailable'}`",
        f"- dirty: `{repo.get('dirty')}`",
        f"- working_tree_fingerprint_sha256: "
        f"`{repo.get('working_tree_fingerprint_sha256')}`",
        "",
        "### Reproduction Commands",
        "",
    ]
    commands = transparency.get("reproduction_commands") or []
    if commands:
        lines.append("```bash")
        for command in commands:
            lines.append(" ".join(str(part) for part in command))
        lines.append("```")
    else:
        lines.append("No command recorded.")
    return "\n".join(lines)


__all__ = [
    "collect_source_transparency",
    "normalize_git_remote_url",
    "transparency_report_markdown",
]
