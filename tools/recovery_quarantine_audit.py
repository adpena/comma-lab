#!/usr/bin/env python3
"""Classify recovered/quarantined files without moving or deleting them.

This is a signal-preservation tool. It compares a quarantine tree against the
live repo, records hashes, detects recovery stubs/specs, and emits a disposition
table for human review before anything is promoted or removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuarantineRecord:
    relpath: str
    category: str
    disposition: str
    quarantine_sha256: str
    quarantine_bytes: int
    live_exists: bool
    live_same_sha256: bool
    live_sha256: str | None
    is_recovery_stub: bool
    is_recovery_spec: bool
    is_incomplete_decompile: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_text_recovery_stub(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    try:
        prefix = path.read_text(errors="replace")[:512]
    except OSError:
        return False
    return "RECOVERY STUB" in prefix or "__recovery_status__" in prefix


def is_incomplete_decompile(path: Path) -> bool:
    if path.suffix not in {".py", ".sh"}:
        return False
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return False
    markers = (
        "# Source Generated with Decompyle++",
        "# WARNING: Decompyle incomplete",
        "<NODE:",
        "pass# WARNING",
    )
    return any(marker in text for marker in markers)


def category_for(relpath: str) -> str:
    parts = Path(relpath).parts
    name = Path(relpath).name
    if relpath.startswith(".omx/auto_memory_snapshot_"):
        return "memory_snapshot"
    if name.endswith(".recovery_spec.json"):
        return "recovery_spec"
    if relpath.startswith(".omx/state/modal_"):
        return "provider_state"
    if relpath.startswith(".omx/state/"):
        return "state"
    if relpath.startswith("experiments/results/"):
        return "public_or_experiment_intake"
    if relpath.startswith("experiments/"):
        return "experiment_tool"
    if relpath.startswith("scripts/"):
        return "script"
    if relpath.startswith("tools/"):
        return "tool"
    if relpath.startswith("src/tac/tests/"):
        return "test"
    if relpath.startswith("src/tac/"):
        return "tac_module"
    if relpath.startswith("submissions/"):
        return "submission_runtime"
    if relpath.startswith("docs/") or relpath.startswith("reports/") or name.endswith(".md"):
        return "docs"
    if "pr" in relpath.lower() and any(part.startswith("public_") for part in parts):
        return "public_intake"
    return "other"


def is_blocked_recovery_name(relpath: str) -> bool:
    return relpath.endswith(".PREFLIGHT_DEBT") or relpath.endswith(".QUARANTINED")


def disposition_for(
    *,
    relpath: str,
    category: str,
    live_exists: bool,
    live_same_sha256: bool,
    is_stub: bool,
    is_spec: bool,
    is_incomplete: bool,
) -> str:
    if live_same_sha256:
        return "duplicate_same_safe_to_delete_after_manifest_commit"
    if category == "memory_snapshot":
        return "preserve_as_custody_then_extract_to_memory_or_ledger"
    if is_spec:
        return "preserve_until_matching_source_is_canonical_then_delete"
    if is_stub or is_incomplete:
        return "do_not_promote_incomplete_recovery_preserve_for_manual_rehydration"
    if is_blocked_recovery_name(relpath):
        return "blocked_recovery_input_needs_canonicalization_before_promotion"
    if live_exists:
        return "compare_by_hand_live_diff_before_merge_or_delete"
    if category in {"tool", "script", "experiment_tool", "tac_module", "test"}:
        return "review_for_promotion_to_main_with_tests"
    if category in {"public_or_experiment_intake", "provider_state", "submission_runtime"}:
        return "preserve_as_forensic_artifact_unless_redundant"
    if category in {"docs", "state"}:
        return "review_for_ledger_or_public_release_hygiene"
    return "manual_review"


def iter_records(repo: Path, quarantine: Path) -> list[QuarantineRecord]:
    records: list[QuarantineRecord] = []
    for path in sorted(item for item in quarantine.rglob("*") if item.is_file()):
        relpath = path.relative_to(quarantine).as_posix()
        live_path = repo / relpath
        q_sha = sha256_file(path)
        live_exists = live_path.is_file()
        live_sha = sha256_file(live_path) if live_exists else None
        live_same = live_sha == q_sha if live_sha is not None else False
        is_spec = Path(relpath).name.endswith(".recovery_spec.json")
        is_stub = is_text_recovery_stub(path)
        is_incomplete = is_incomplete_decompile(path)
        category = category_for(relpath)
        records.append(
            QuarantineRecord(
                relpath=relpath,
                category=category,
                disposition=disposition_for(
                    category=category,
                    relpath=relpath,
                    live_exists=live_exists,
                    live_same_sha256=live_same,
                    is_stub=is_stub,
                    is_spec=is_spec,
                    is_incomplete=is_incomplete,
                ),
                quarantine_sha256=q_sha,
                quarantine_bytes=path.stat().st_size,
                live_exists=live_exists,
                live_same_sha256=live_same,
                live_sha256=live_sha,
                is_recovery_stub=is_stub,
                is_recovery_spec=is_spec,
                is_incomplete_decompile=is_incomplete,
            )
        )
    return records


def render_markdown(records: list[QuarantineRecord], quarantine: Path) -> str:
    by_category = Counter(record.category for record in records)
    by_disposition = Counter(record.disposition for record in records)
    lines = [
        "# Recovery Quarantine Audit",
        "",
        f"- quarantine: `{quarantine}`",
        f"- total_files: `{len(records)}`",
        "",
        "## Category Counts",
        "",
        "| category | files |",
        "|---|---:|",
    ]
    for key, count in sorted(by_category.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Disposition Counts", "", "| disposition | files |", "|---|---:|"])
    for key, count in sorted(by_disposition.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "## Promotion Queue",
            "",
            "| relpath | category | bytes | sha256 |",
            "|---|---|---:|---|",
        ]
    )
    for record in records:
        if record.disposition != "review_for_promotion_to_main_with_tests":
            continue
        lines.append(
            f"| `{record.relpath}` | `{record.category}` | {record.quarantine_bytes} | "
            f"`{record.quarantine_sha256[:16]}` |"
        )
    lines.extend(
        [
            "",
            "## Live Diff Queue",
            "",
            "| relpath | category | bytes | quarantine sha | live sha |",
            "|---|---|---:|---|---|",
        ]
    )
    for record in records:
        if record.disposition != "compare_by_hand_live_diff_before_merge_or_delete":
            continue
        lines.append(
            f"| `{record.relpath}` | `{record.category}` | {record.quarantine_bytes} | "
            f"`{record.quarantine_sha256[:16]}` | `{(record.live_sha256 or '')[:16]}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    quarantine = args.quarantine.resolve()
    if not quarantine.is_dir():
        raise SystemExit(f"quarantine directory does not exist: {quarantine}")

    records = iter_records(repo, quarantine)
    payload = {
        "schema_version": 1,
        "tool": "recovery_quarantine_audit",
        "repo": str(repo),
        "quarantine": str(quarantine),
        "records": [asdict(record) for record in records],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(records, quarantine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
