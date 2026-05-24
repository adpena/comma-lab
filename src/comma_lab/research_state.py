"""Audit research-state files for git tracking and external artifact custody.

The contest lab has three different kinds of useful state:

* small durable research records that should be tracked in git;
* large or rebuildable artifacts that should be hosted externally with a
  committed manifest;
* private provider/operator state that should stay local but be summarized.

This module owns that boundary for the lab layer. The reusable ``tac``
Task-Aware Compression library should not need to know about Claude, OMX,
provider logs, or release custody policy.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".tsv"}
SMALL_TEXT_LIMIT_BYTES = 1_000_000
LARGE_ARTIFACT_LIMIT_BYTES = 5_000_000
LOCAL_RESEARCH_CONTROL_PATTERNS = (
    ".omx/research/*_storage_plan_*.json",
    ".omx/research/*_proactive_cleanup_*.json",
    ".omx/research/*_proactive_cleanup_*.json.journal.jsonl",
    ".omx/research/*_artifact_retention_*.json",
    ".omx/research/*_artifact_retention_*.json.journal.jsonl",
)

DEFAULT_SCAN_ROOTS = (
    ".omx",
    ".claude",
    "docs",
    "reports",
)


@dataclass(frozen=True)
class ResearchStateRecord:
    relpath: str
    bytes: int
    git_status: str
    category: str
    disposition: str
    target: str
    reason: str


def _run_git(root: Path, args: list[str], *, stdin: str | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode not in {0, 1}:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _tracked_paths(root: Path) -> set[str]:
    return set(filter(None, _run_git(root, ["ls-files"]).splitlines()))


def _ignored_paths(root: Path, relpaths: list[str]) -> set[str]:
    if not relpaths:
        return set()
    # `git check-ignore` normally treats paths inside gitlinks/submodules as a
    # fatal pathspec error. Lab raw-intake trees intentionally contain public
    # repo snapshots, so use --no-index to apply ignore rules without asking git
    # to resolve those paths as normal tracked working-tree members.
    output = _run_git(root, ["check-ignore", "--no-index", "--stdin"], stdin="\n".join(relpaths) + "\n")
    return set(filter(None, output.splitlines()))


def iter_candidate_files(root: Path, scan_roots: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in scan_roots:
        path = root / Path(raw)
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        files.extend(item for item in path.rglob("*") if item.is_file())
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def classify_relpath(relpath: str, size: int, git_status: str) -> tuple[str, str, str, str]:
    path = Path(relpath)
    suffix = path.suffix.lower()
    is_small_text = suffix in TEXT_SUFFIXES and size <= SMALL_TEXT_LIMIT_BYTES

    if "__pycache__/" in relpath or suffix in {".pyc", ".pyo"}:
        return (
            "cache",
            "delete_cache_after_manifest",
            "not tracked",
            "Python cache output is rebuildable and should not be custody state.",
        )

    if relpath.startswith(".claude/settings.local") or relpath.endswith(".lock"):
        return (
            "private_operator_state",
            "keep_private_local",
            "local only",
            "Local Claude settings and lock files can contain operator-specific state.",
        )

    if relpath.startswith(".omx/auto_memory_snapshot_"):
        return (
            "memory_snapshot",
            "canonicalize_to_research_ledger",
            ".omx/research/ or docs/paper/ara/",
            "Memory snapshots are interesting, but raw snapshots are bulky operator backups.",
        )

    if relpath.startswith(".omx/research/artifacts/"):
        return (
            "research_artifact",
            "externalize_with_manifest",
            "external artifact store plus committed manifest",
            "Generated audit/profile artifacts are rebuildable custody outputs; summarize or manifest them instead of committing every run output.",
        )

    if any(fnmatch.fnmatch(relpath, pattern) for pattern in LOCAL_RESEARCH_CONTROL_PATTERNS):
        return (
            "queue_local_control_artifact",
            "keep_private_local",
            "local only plus dated research memo for durable findings",
            "Queue-owned storage, cleanup, and retention JSON can contain machine-local volume choices and should not be tracked directly.",
        )

    if relpath.startswith(".omx/research/"):
        if is_small_text:
            return (
                "research_ledger",
                "track_in_git",
                relpath,
                "Small research ledgers and structured summaries are durable lab knowledge.",
            )
        return (
            "research_artifact",
            "externalize_with_manifest",
            "external artifact store plus committed manifest",
            "Large or binary research artifacts should not bloat git history.",
        )

    if relpath in {
        ".omx/state/current_focus.md",
        ".omx/state/next_experiments.md",
        ".omx/state/active_lane_dispatch_claims.md",
    }:
        return (
            "operator_control_plane",
            "track_in_git",
            relpath,
            "Small markdown control-plane state is part of cross-agent continuity.",
        )

    if relpath.startswith(".omx/state/"):
        if suffix == ".md" and is_small_text:
            return (
                "operator_control_plane",
                "track_in_git",
                relpath,
                "Small markdown state is reviewable and low-risk to track.",
            )
        return (
            "provider_or_runtime_state",
            "summarize_to_research_ledger",
            ".omx/research/<dated summary>.md",
            "JSON/log provider state may leak account, host, or transient queue details.",
        )

    if relpath.startswith(".omx/logs/") or relpath.startswith(".omx/tmp/"):
        return (
            "ephemeral_operator_log",
            "summarize_to_research_ledger",
            ".omx/research/<dated summary>.md",
            "Raw logs are useful for forensics but should be summarized before public git.",
        )

    if relpath.startswith("reports/raw/") or relpath.startswith("reports/private/"):
        return (
            "raw_or_private_report_artifact",
            "externalize_with_manifest",
            "external artifact store plus committed manifest",
            "Raw/private report artifacts are usually large, rebuildable, or sensitive.",
        )

    if relpath.startswith("reports/graphs/public_site/.wrangler/"):
        return (
            "hosted_supplement_cache",
            "keep_private_local",
            "local only",
            "Wrangler cache is account-local hosting state, not a public source artifact.",
        )

    if relpath.startswith("reports/graphs/public_site/"):
        return (
            "hosted_supplement_build",
            "externalize_with_manifest",
            "docs/site source plus external hosted supplement manifest",
            "Generated public-site bundles should be hosted or rebuilt from source, not committed wholesale.",
        )

    if size > LARGE_ARTIFACT_LIMIT_BYTES:
        return (
            "large_artifact",
            "externalize_with_manifest",
            "external artifact store plus committed manifest",
            "Large artifacts belong in Hugging Face, Lightning, or another artifact store.",
        )

    if relpath.startswith("docs/") or relpath.startswith("reports/"):
        if is_small_text:
            return (
                "public_lab_doc",
                "track_in_git",
                relpath,
                "Small docs and report summaries are public lab output.",
            )
        return (
            "derived_report_artifact",
            "externalize_with_manifest",
            "external artifact store plus committed manifest",
            "Derived media should be linked by manifest instead of committed blindly.",
        )

    if git_status == "tracked":
        return (
            "tracked_other",
            "track_in_git",
            relpath,
            "Already tracked; keep unless a later cleanup classifies it as private or large.",
        )

    return (
        "uncategorized",
        "manual_review",
        "comma_lab research-state triage",
        "No durable policy matched this path.",
    )


def audit_research_state(root: Path, scan_roots: Iterable[str | Path] = DEFAULT_SCAN_ROOTS) -> list[ResearchStateRecord]:
    files = iter_candidate_files(root, scan_roots)
    relpaths = [path.relative_to(root).as_posix() for path in files]
    tracked = _tracked_paths(root)
    ignored = _ignored_paths(root, relpaths)

    records: list[ResearchStateRecord] = []
    for path, relpath in zip(files, relpaths, strict=True):
        if relpath in tracked:
            git_status = "tracked"
        elif relpath in ignored:
            git_status = "ignored"
        else:
            git_status = "untracked"
        category, disposition, target, reason = classify_relpath(relpath, path.stat().st_size, git_status)
        records.append(
            ResearchStateRecord(
                relpath=relpath,
                bytes=path.stat().st_size,
                git_status=git_status,
                category=category,
                disposition=disposition,
                target=target,
                reason=reason,
            )
        )
    return records


def render_markdown(records: list[ResearchStateRecord]) -> str:
    by_disposition = Counter(record.disposition for record in records)
    by_category = Counter(record.category for record in records)
    lines = [
        "# Research State Tracking Audit",
        "",
        f"- total_files: `{len(records)}`",
        "",
        "## Disposition Counts",
        "",
        "| disposition | files |",
        "|---|---:|",
    ]
    for key, count in sorted(by_disposition.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    lines.extend(["", "## Category Counts", "", "| category | files |", "|---|---:|"])
    for key, count in sorted(by_category.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    def table(title: str, dispositions: set[str], limit: int = 80) -> None:
        rows = [record for record in records if record.disposition in dispositions]
        lines.extend(["", f"## {title}", "", "| path | git | bytes | target | reason |", "|---|---|---:|---|---|"])
        for record in rows[:limit]:
            lines.append(
                f"| `{record.relpath}` | `{record.git_status}` | {record.bytes} | "
                f"`{record.target}` | {record.reason} |"
            )
        if len(rows) > limit:
            lines.append(f"| ... | ... | ... | ... | {len(rows) - limit} more omitted; see JSON. |")

    table("Should Be Tracked In Git", {"track_in_git"})
    table("Canonicalize Or Summarize", {"canonicalize_to_research_ledger", "summarize_to_research_ledger"})
    table("Externalize With Manifest", {"externalize_with_manifest"})
    table("Keep Private Or Delete Cache", {"keep_private_local", "delete_cache_after_manifest"})
    table("Manual Review", {"manual_review"})
    lines.append("")
    return "\n".join(lines)


def build_summary(records: list[ResearchStateRecord]) -> dict[str, object]:
    return {
        "total_files": len(records),
        "disposition_counts": dict(Counter(record.disposition for record in records)),
        "category_counts": dict(Counter(record.category for record in records)),
        "records": [asdict(record) for record in records],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--scan-root", action="append", default=None, help="Path to scan; repeatable.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--fail-on-untracked-trackable", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve()
    scan_roots = args.scan_root or list(DEFAULT_SCAN_ROOTS)
    records = audit_research_state(root, scan_roots)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(build_summary(records), indent=2, sort_keys=True) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(records))

    untracked_trackable = [
        record for record in records if record.disposition == "track_in_git" and record.git_status != "tracked"
    ]
    if args.fail_on_untracked_trackable and untracked_trackable:
        for record in untracked_trackable[:50]:
            print(f"trackable state is not tracked: {record.relpath}")
        if len(untracked_trackable) > 50:
            print(f"... {len(untracked_trackable) - 50} more")
        return 1

    print(json.dumps(build_summary(records), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
