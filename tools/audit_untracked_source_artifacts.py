#!/usr/bin/env python3
"""Audit untracked source/research artifacts that may represent signal loss.

The repository intentionally has large rebuildable artifacts, provider state,
and recovered forensics. This tool separates those from source-like untracked
files that should be either tracked, moved to a canonical recovery queue, or
ignored explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.audit_contract import AuditReport, audit_exit_code  # noqa: E402
from tac.repo_io import json_text, repo_relative  # noqa: E402

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".zig",
}
SOURCE_ROOTS = (
    ".omx/research",
    "docs",
    "experiments",
    "reports",
    "reverse_engineering",
    "scripts",
    "src",
    "submissions",
    "tests",
    "tools",
)
GENERATED_ROOTS = (
    ".omx/research/artifacts",
    ".omx/state",
    "experiments/results",
    "outputs",
    "reports/raw",
    "reports/private",
)
RUNTIME_PACKET_DIR_NAMES = frozenset(
    {
        "full_runtime_packet",
        "replay_submission",
        "runtime_packet",
        "submission_dir",
    }
)
RUNTIME_ENTRYPOINT_NAMES = frozenset(
    {
        "compress.py",
        "compress.sh",
        "inflate.py",
        "inflate.sh",
    }
)
RUNTIME_SOURCE_BASELINE_ALGORITHM = "pact_runtime_source_set_v1"


@dataclass(frozen=True)
class UntrackedRecord:
    path: str
    classification: str
    reason: str


VALID_DISPOSITIONS = frozenset(
    {
        "track",
        "recovery_queue",
        "ignore_private",
        "ignore_rebuildable",
    }
)
VALID_PATH_KINDS = frozenset({"exact", "prefix"})


def parse_git_status_porcelain(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if status == "??" and path:
            records.append((status, path))
    return records


def parse_git_status_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if path:
            records.append((status, path))
    return records


def _is_under_root(posix: str, roots: tuple[str, ...]) -> bool:
    return any(posix == root or posix.startswith(root + "/") for root in roots)


def _has_source_suffix(posix: str) -> bool:
    return Path(posix).suffix.lower() in SOURCE_SUFFIXES


def _is_generated_custody_path(posix: str) -> bool:
    return _is_under_root(posix, GENERATED_ROOTS)


def _is_generated_custody_prefix(posix: str) -> bool:
    stripped = posix.rstrip("/")
    return any(stripped == root or stripped.startswith(root + "/") for root in GENERATED_ROOTS)


def _is_source_like_path(path: str) -> bool:
    posix = path.replace("\\", "/")
    if not _has_source_suffix(posix):
        return False
    return _is_under_root(posix, SOURCE_ROOTS) or _is_generated_custody_path(posix)


def _is_runtime_source_like_path(path: str) -> bool:
    posix = path.replace("\\", "/")
    if not _is_source_like_path(posix) or not _is_generated_custody_path(posix):
        return False
    parts = tuple(part for part in posix.split("/") if part)
    if not parts:
        return False
    if parts[-1] in RUNTIME_ENTRYPOINT_NAMES:
        return True
    return any(part in RUNTIME_PACKET_DIR_NAMES for part in parts)


def _default_disposition_for_record(record: UntrackedRecord) -> str:
    if record.classification != "generated_custody_source_untracked":
        return "track"
    if record.path.startswith((".omx/state/", "reports/raw/", "reports/private/")):
        return "ignore_private"
    return "ignore_rebuildable"


def classify_untracked_path(path: str) -> UntrackedRecord | None:
    posix = path.replace("\\", "/")
    if not _is_source_like_path(posix):
        return None
    if _is_generated_custody_path(posix):
        return UntrackedRecord(
            posix,
            "generated_custody_source_untracked",
            "generated custody source-like artifact needs explicit disposition before promotion",
        )
    if posix.startswith(".omx/research/"):
        return UntrackedRecord(posix, "research_untracked", "research markdown/state should be tracked or promoted")
    if posix.startswith("reverse_engineering/"):
        return UntrackedRecord(posix, "reverse_engineering_untracked", "recovered/deconstruction source needs disposition")
    return UntrackedRecord(posix, "source_untracked", "source-like artifact is untracked")


def _validate_runtime_source_baseline(path: Path, index: int, baseline: Any) -> dict[str, object]:
    if not isinstance(baseline, dict):
        raise ValueError(f"{path}: entries[{index}].runtime_source_baseline must be an object")
    algorithm = baseline.get("algorithm", RUNTIME_SOURCE_BASELINE_ALGORITHM)
    count = baseline.get("count")
    sha256 = baseline.get("sha256")
    if algorithm != RUNTIME_SOURCE_BASELINE_ALGORITHM:
        raise ValueError(
            f"{path}: entries[{index}].runtime_source_baseline.algorithm={algorithm!r}; "
            f"expected {RUNTIME_SOURCE_BASELINE_ALGORITHM!r}"
        )
    if not isinstance(count, int) or count < 0:
        raise ValueError(f"{path}: entries[{index}].runtime_source_baseline.count must be a nonnegative int")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in sha256)
    ):
        raise ValueError(f"{path}: entries[{index}].runtime_source_baseline.sha256 must be lowercase sha256 hex")
    return {"algorithm": algorithm, "count": count, "sha256": sha256}


def load_disposition_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected top-level entries list")
    out: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entries[{index}] is not an object")
        relpath = entry.get("path")
        disposition = entry.get("disposition")
        note = entry.get("note", "")
        path_kind = entry.get("path_kind", "exact")
        if not isinstance(relpath, str) or not relpath:
            raise ValueError(f"{path}: entries[{index}].path must be a nonempty string")
        if not isinstance(path_kind, str) or path_kind not in VALID_PATH_KINDS:
            raise ValueError(
                f"{path}: entries[{index}].path_kind={path_kind!r}; "
                f"expected one of {sorted(VALID_PATH_KINDS)}"
            )
        if path_kind == "prefix" and not relpath.endswith("/"):
            raise ValueError(f"{path}: entries[{index}].path must end in '/' when path_kind is prefix")
        if path_kind == "prefix" and not _is_generated_custody_prefix(relpath):
            raise ValueError(
                f"{path}: entries[{index}].path_kind=prefix is only valid "
                f"under generated custody roots {GENERATED_ROOTS}; got {relpath!r}"
            )
        if disposition not in VALID_DISPOSITIONS:
            raise ValueError(
                f"{path}: entries[{index}].disposition={disposition!r}; "
                f"expected one of {sorted(VALID_DISPOSITIONS)}"
            )
        if not isinstance(note, str) or not note:
            raise ValueError(f"{path}: entries[{index}].note must be a nonempty string")
        loaded_entry: dict[str, Any] = {
            "disposition": disposition,
            "note": note,
            "path_kind": path_kind,
        }
        if "runtime_source_baseline" in entry:
            if path_kind != "prefix":
                raise ValueError(
                    f"{path}: entries[{index}].runtime_source_baseline is only valid for prefix entries"
                )
            loaded_entry["runtime_source_baseline"] = _validate_runtime_source_baseline(
                path, index, entry["runtime_source_baseline"]
            )
        out[relpath] = loaded_entry
    return out


def _is_known_audit_prefix(prefix: str) -> bool:
    return _is_generated_custody_prefix(prefix)


def _git_status(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "git status failed")
    return proc.stdout


def _git_tracked_files(repo_root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        stdout = proc.stdout.decode(errors="replace").strip()
        raise RuntimeError(stderr or stdout or "git ls-files failed")
    return {path for path in proc.stdout.decode().split("\0") if path}


def _generated_source_filesystem_records(
    repo_root: Path,
    *,
    tracked_paths: set[str],
) -> list[UntrackedRecord]:
    """Return source-like files under ignored generated custody roots.

    ``git status --untracked-files=all`` omits ignored files, which is exactly
    where many raw custody/runtime packets live. A direct bounded walk over
    generated roots keeps those source-like artifacts visible to the
    disposition and runtime-source-baseline gates.
    """

    records: list[UntrackedRecord] = []
    for root in GENERATED_ROOTS:
        root_path = repo_root / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            rel = repo_relative(path, repo_root)
            posix = rel.replace("\\", "/")
            if posix in tracked_paths:
                continue
            record = classify_untracked_path(posix)
            if record is not None:
                records.append(record)
    return records


def find_invalid_disposition_paths(
    dispositions: dict[str, dict[str, Any]],
    *,
    live_untracked_paths: set[str],
    tracked_source_like_paths: set[str],
) -> list[str]:
    valid_paths = live_untracked_paths | tracked_source_like_paths
    invalid: list[str] = []
    for relpath, entry in dispositions.items():
        path_kind = entry.get("path_kind", "exact")
        if path_kind == "prefix":
            if _is_known_audit_prefix(relpath) or any(
                path.startswith(relpath) for path in valid_paths
            ):
                continue
            invalid.append(relpath)
            continue
        if relpath not in valid_paths:
            invalid.append(relpath)
    return sorted(invalid)


def find_disposition_for_path(
    dispositions: dict[str, dict[str, Any]], relpath: str
) -> dict[str, Any] | None:
    exact = dispositions.get(relpath)
    if exact is not None and exact.get("path_kind", "exact") == "exact":
        return exact
    prefix_matches = [
        (prefix, entry)
        for prefix, entry in dispositions.items()
        if entry.get("path_kind", "exact") == "prefix" and relpath.startswith(prefix)
    ]
    if not prefix_matches:
        return None
    return max(prefix_matches, key=lambda item: len(item[0]))[1]


def find_exact_disposition_for_path(
    dispositions: dict[str, dict[str, Any]], relpath: str
) -> dict[str, Any] | None:
    exact = dispositions.get(relpath)
    if exact is not None and exact.get("path_kind", "exact") == "exact":
        return exact
    return None


def find_prefix_disposition_for_path(
    dispositions: dict[str, dict[str, Any]], relpath: str
) -> tuple[str, dict[str, Any]] | None:
    prefix_matches = [
        (prefix, entry)
        for prefix, entry in dispositions.items()
        if entry.get("path_kind", "exact") == "prefix" and relpath.startswith(prefix)
    ]
    if not prefix_matches:
        return None
    return max(prefix_matches, key=lambda item: len(item[0]))


def _runtime_source_file_fingerprint(repo_root: Path, relpath: str) -> str:
    path = repo_root / relpath
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_runtime_source_baseline(repo_root: Path, relpaths: list[str] | tuple[str, ...]) -> dict[str, object]:
    digest = hashlib.sha256()
    digest.update((RUNTIME_SOURCE_BASELINE_ALGORITHM + "\n").encode())
    runtime_paths = sorted(path.replace("\\", "/") for path in relpaths if _is_runtime_source_like_path(path))
    for relpath in runtime_paths:
        file_path = repo_root / relpath
        size = file_path.stat().st_size
        file_sha256 = _runtime_source_file_fingerprint(repo_root, relpath)
        digest.update(f"{relpath}\0{size}\0{file_sha256}\n".encode())
    return {
        "algorithm": RUNTIME_SOURCE_BASELINE_ALGORITHM,
        "count": len(runtime_paths),
        "sha256": digest.hexdigest(),
    }


def find_runtime_source_custody_blockers(
    dispositions: dict[str, dict[str, Any]],
    runtime_source_paths: list[str] | tuple[str, ...],
    *,
    repo_root: Path,
) -> tuple[list[str], list[dict[str, object]]]:
    blockers: list[str] = []
    prefix_runtime_paths: dict[str, list[str]] = {}
    prefix_entries: dict[str, dict[str, Any]] = {}

    for relpath in sorted(path.replace("\\", "/") for path in runtime_source_paths):
        if find_exact_disposition_for_path(dispositions, relpath) is not None:
            continue
        prefix_match = find_prefix_disposition_for_path(dispositions, relpath)
        if prefix_match is None:
            continue
        prefix, entry = prefix_match
        baseline = entry.get("runtime_source_baseline")
        if baseline is None:
            blockers.append(
                f"{relpath}: runtime source-like artifact matched prefix {prefix} "
                "without exact disposition or runtime_source_baseline"
            )
            continue
        prefix_entries[prefix] = entry
        prefix_runtime_paths.setdefault(prefix, []).append(relpath)

    baseline_summaries: list[dict[str, object]] = []
    for prefix, paths in sorted(prefix_runtime_paths.items()):
        expected = prefix_entries[prefix]["runtime_source_baseline"]
        actual = build_runtime_source_baseline(repo_root, paths)
        status = (
            "matched"
            if expected["count"] == actual["count"] and expected["sha256"] == actual["sha256"]
            else "mismatch"
        )
        summary = {
            "prefix": prefix,
            "status": status,
            "expected_count": expected["count"],
            "expected_sha256": expected["sha256"],
            "actual_count": actual["count"],
            "actual_sha256": actual["sha256"],
            "sample_paths": paths[:20],
        }
        baseline_summaries.append(summary)
        if status != "matched":
            blockers.append(
                f"{prefix}: runtime_source_baseline mismatch "
                f"(expected count={expected['count']} sha256={expected['sha256']}; "
                f"actual count={actual['count']} sha256={actual['sha256']})"
            )
    return blockers, baseline_summaries


def audit_untracked_source_artifacts(
    repo_root: Path,
    *,
    disposition_manifest: Path | None = None,
) -> AuditReport:
    dispositions = load_disposition_manifest(disposition_manifest)
    status_text = _git_status(repo_root)
    records: list[UntrackedRecord] = []
    all_status_records = parse_git_status_records(status_text)
    tracked_files = _git_tracked_files(repo_root)
    for _status, path in parse_git_status_porcelain(status_text):
        record = classify_untracked_path(path)
        if record is not None:
            records.append(record)
    by_path = {record.path: record for record in records}
    for record in _generated_source_filesystem_records(
        repo_root,
        tracked_paths=tracked_files,
    ):
        by_path.setdefault(record.path, record)
    records = sorted(by_path.values(), key=lambda item: item.path)

    untracked_paths = {record.path for record in records}
    tracked_source_like_paths = {
        path.replace("\\", "/")
        for path in tracked_files
        if _is_source_like_path(path)
    }
    source_like_deletes = sorted(
        path.replace("\\", "/")
        for status, path in all_status_records
        if "D" in status and _is_source_like_path(path)
    )
    shadowed_delete_paths = sorted(set(source_like_deletes) & untracked_paths)
    runtime_source_paths = sorted(record.path for record in records if _is_runtime_source_like_path(record.path))

    by_class: dict[str, int] = {}
    by_disposition: dict[str, int] = {}
    undispositioned: list[UntrackedRecord] = []
    invalid_disposition_paths = find_invalid_disposition_paths(
        dispositions,
        live_untracked_paths=untracked_paths,
        tracked_source_like_paths=tracked_source_like_paths,
    )
    for record in records:
        by_class[record.classification] = by_class.get(record.classification, 0) + 1
        disposition = find_disposition_for_path(dispositions, record.path)
        if disposition is None:
            undispositioned.append(record)
        else:
            disposition_name = disposition["disposition"]
            by_disposition[disposition_name] = by_disposition.get(disposition_name, 0) + 1
    runtime_source_blockers, runtime_source_baselines = find_runtime_source_custody_blockers(
        dispositions,
        runtime_source_paths,
        repo_root=repo_root,
    )

    blockers = tuple(
        [f"{r.path}: {r.classification}: undispositioned source-like artifact" for r in undispositioned]
        + [f"{path}: disposition entry does not match a live untracked source path" for path in invalid_disposition_paths]
        + runtime_source_blockers
    )
    return AuditReport(
        audit="untracked_source_artifacts",
        readiness_key="ready_for_no_signal_loss_canonicalization",
        ready=not blockers,
        blockers=blockers,
        summary={
            "by_class": by_class,
            "by_disposition": by_disposition,
            "disposition_manifest": str(disposition_manifest) if disposition_manifest else None,
            "disposition_exact_entry_count": sum(
                1 for entry in dispositions.values() if entry.get("path_kind", "exact") == "exact"
            ),
            "disposition_prefix_entry_count": sum(
                1 for entry in dispositions.values() if entry.get("path_kind", "exact") == "prefix"
            ),
            "dispositioned_count": len(records) - len(undispositioned),
            "generated_custody_source_like_count": sum(
                1 for record in records if record.classification == "generated_custody_source_untracked"
            ),
            "resolved_tracked_disposition_count": len(set(dispositions) & tracked_source_like_paths),
            "runtime_source_baseline_blocker_count": len(runtime_source_blockers),
            "runtime_source_baselines": runtime_source_baselines,
            "runtime_source_exact_disposition_count": sum(
                1 for path in runtime_source_paths if find_exact_disposition_for_path(dispositions, path) is not None
            ),
            "runtime_source_like_count": len(runtime_source_paths),
            "shadowed_index_delete_count": len(shadowed_delete_paths),
            "shadowed_index_delete_paths": shadowed_delete_paths[:50],
            "source_like_delete_count": len(source_like_deletes),
            "untracked_source_like_count": len(records),
            "undispositioned_count": len(undispositioned),
            "invalid_disposition_count": len(invalid_disposition_paths),
            "sample": [record.__dict__ for record in records[:50]],
            "policy": "track, canonicalize into recovery queue, or add explicit ignore rule",
        },
        metadata={
            "repo_root": repo_relative(repo_root, repo_root),
        },
    )


def write_disposition_template(repo_root: Path, output_path: Path) -> None:
    tracked_files = _git_tracked_files(repo_root)
    by_path = {
        record.path: record
        for _status, path in parse_git_status_porcelain(_git_status(repo_root))
        if (record := classify_untracked_path(path)) is not None
    }
    for record in _generated_source_filesystem_records(
        repo_root,
        tracked_paths=tracked_files,
    ):
        by_path.setdefault(record.path, record)
    records = sorted(by_path.values(), key=lambda item: item.path)
    payload = {
        "schema": "pact_untracked_source_dispositions_v1",
        "policy": "Each entry must be track, recovery_queue, ignore_private, or ignore_rebuildable.",
        "entries": [
            {
                "path": record.path,
                "classification": record.classification,
                "disposition": _default_disposition_for_record(record),
                "note": "default template; review before treating as final disposition",
            }
            for record in records
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_text(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--disposition-manifest", type=Path)
    parser.add_argument("--write-disposition-template", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if args.write_disposition_template:
        write_disposition_template(repo_root, args.write_disposition_template)
    report = audit_untracked_source_artifacts(
        repo_root,
        disposition_manifest=args.disposition_manifest,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(report.to_dict()))
    if args.format == "json":
        print(json_text(report.to_dict()), end="")
    else:
        print(
            "untracked source audit: "
            f"{report.summary['untracked_source_like_count']} source-like untracked file(s); "
            f"{report.summary['undispositioned_count']} undispositioned"
        )
        for record in report.summary["sample"]:
            print(f"  - {record['path']}: {record['classification']}")
    if args.strict:
        return audit_exit_code(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
