#!/usr/bin/env python3
"""Audit the git index/worktree split before public-release commits.

This guard protects a common dirty-worktree failure mode:

* the index stages a rollback or deletion;
* the working tree still contains the desired file contents;
* normal local checks pass because they read the working tree;
* a commit would silently publish the staged rollback.

It also keeps provider/runtime custody state out of public release commits.
Small markdown control-plane files are allowed; JSON, database, and provider
state should be summarized into dated research ledgers instead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ALLOWED_STAGED_OMX_STATE = {
    ".omx/state/active_lane_dispatch_claims.md",
    ".omx/state/current_focus.md",
    ".omx/state/next_experiments.md",
}

LOCAL_CUSTODY_PREFIXES = (
    "experiments/results/",
    "reports/raw/",
    "reverse_engineering/orphan_pyc_recovery_",
)


@dataclass(frozen=True)
class IndexRecord:
    xy: str
    path: str
    kind: str
    severity: str
    detail: str
    documented_by: str | None = None


def _git(root: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _git_stdout_or_none(root: Path, args: list[str]) -> str | None:
    code, stdout, _stderr = _git(root, args)
    if code != 0:
        return None
    return stdout.strip()


def parse_status_porcelain(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in text.splitlines():
        if not raw:
            continue
        xy = raw[:2]
        path = raw[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        rows.append((xy, path))
    return rows


def _hashes(root: Path, path: str) -> tuple[str | None, str | None, str | None]:
    head = _git_stdout_or_none(root, ["rev-parse", f"HEAD:{path}"])
    index = _git_stdout_or_none(root, ["rev-parse", f":{path}"])
    worktree: str | None = None
    full_path = root / path
    if full_path.is_file():
        worktree = _git_stdout_or_none(root, ["hash-object", path])
    return head, index, worktree


def load_local_custody_rules(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid local custody manifest {path}: {exc}") from exc
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"local custody manifest must contain a rules list: {path}")
    parsed: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"local custody rule #{index} must be an object")
        rule_id = rule.get("id")
        match = rule.get("match")
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError(f"local custody rule #{index} missing nonempty id")
        if not isinstance(match, dict):
            raise ValueError(f"local custody rule {rule_id!r} missing match object")
        parsed.append(rule)
    return parsed


def _rule_matches(record: IndexRecord, rule: dict[str, Any]) -> bool:
    match = rule.get("match", {})
    kind = match.get("kind")
    if isinstance(kind, str) and record.kind != kind:
        return False
    path = match.get("path")
    if isinstance(path, str) and record.path != path:
        return False
    paths = match.get("paths")
    if isinstance(paths, list) and record.path not in paths:
        return False
    prefix = match.get("path_prefix")
    if isinstance(prefix, str) and not record.path.startswith(prefix):
        return False
    prefixes = match.get("path_prefixes")
    if not isinstance(prefixes, list):
        return True
    return any(isinstance(item, str) and record.path.startswith(item) for item in prefixes)


def document_local_custody(record: IndexRecord, rules: list[dict[str, Any]]) -> IndexRecord:
    if record.severity != "warning":
        return record
    for rule in rules:
        if _rule_matches(record, rule):
            return IndexRecord(
                xy=record.xy,
                path=record.path,
                kind=record.kind,
                severity="info",
                detail=record.detail,
                documented_by=str(rule["id"]),
            )
    return record


def audit_release_index_split(root: Path, *, local_custody_rules: list[dict[str, Any]] | None = None) -> list[IndexRecord]:
    code, stdout, stderr = _git(root, ["status", "--porcelain=v1"])
    if code != 0:
        raise RuntimeError(f"git status failed: {stderr.strip()}")

    records: list[IndexRecord] = []
    for xy, path in parse_status_porcelain(stdout):
        index_status, worktree_status = xy[0], xy[1]
        if index_status == "?" and worktree_status == "?":
            continue

        head, index, worktree = _hashes(root, path)
        if head and index and worktree and worktree == head and index != head:
            records.append(
                IndexRecord(
                    xy=xy,
                    path=path,
                    kind="shadowed_staged_rollback",
                    severity="blocker",
                    detail=(
                        "working tree equals HEAD but the index differs; local checks "
                        "will read the preserved working tree while a commit would publish "
                        "the staged rollback"
                    ),
                )
            )

        if path.startswith(".omx/state/") and path not in ALLOWED_STAGED_OMX_STATE:
            if index_status != " ":
                records.append(
                    IndexRecord(
                        xy=xy,
                        path=path,
                        kind="staged_private_runtime_state",
                        severity="blocker",
                        detail=(
                            "provider/runtime state must stay local or be summarized in "
                            ".omx/research before public release"
                        ),
                    )
                )
            elif worktree_status != " ":
                records.append(
                    IndexRecord(
                        xy=xy,
                        path=path,
                        kind="unstaged_private_runtime_state",
                        severity="warning",
                        detail="private runtime state is preserved locally and not staged",
                    )
                )
        elif path.startswith(LOCAL_CUSTODY_PREFIXES) and worktree_status != " ":
            records.append(
                IndexRecord(
                    xy=xy,
                    path=path,
                    kind="unstaged_local_custody_snapshot",
                    severity="warning",
                    detail=(
                        "raw/generated custody state is preserved locally; promote only "
                        "through a canonical source file, release manifest, or dated ledger"
                    ),
                )
            )
    rules = local_custody_rules or []
    return [document_local_custody(record, rules) for record in records]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--local-custody-manifest",
        type=Path,
        default=None,
        help="Manifest documenting intentionally local provider/raw/generated custody paths.",
    )
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    try:
        rules = load_local_custody_rules(args.local_custody_manifest)
    except ValueError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    records = audit_release_index_split(root, local_custody_rules=rules)
    payload = {
        "summary": {
            "record_count": len(records),
            "blocker_count": sum(1 for record in records if record.severity == "blocker"),
            "warning_count": sum(1 for record in records if record.severity == "warning"),
            "documented_count": sum(1 for record in records if record.severity == "info"),
        },
        "records": [asdict(record) for record in records],
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        visible_records = [record for record in records if record.severity != "info"]
        if visible_records:
            for record in visible_records:
                print(f"{record.severity}: {record.path}: {record.kind}: {record.detail}")
        else:
            documented = payload["summary"]["documented_count"]
            suffix = f" ({documented} documented local custody record(s))" if documented else ""
            print(f"release index split: PASS{suffix}")

    if args.strict and payload["summary"]["blocker_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
