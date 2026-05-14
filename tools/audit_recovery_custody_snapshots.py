#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit unique recovery custody snapshots for no-signal-loss preservation.

The recovery cleanup produced two local forensic surfaces that should not be
deleted casually:

* ``.omx/state/orphan_pyc_recovery_20260505`` records bytecode-derived source
  recovery from local ``__pycache__`` files.
* ``.omx/state/signal_loss_audit_20260505T1439Z`` records the broader
  quarantine/signal-loss sweep.

This audit does not promote recovered code or make score claims. It verifies
that the custody snapshots are structurally intact, that tracked-source loss
is not recorded in the signal snapshot, and that unresolved buckets remain
explicitly visible for later hand disposition.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
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
from tac.repo_io import json_text, read_json, repo_relative  # noqa: E402

PYC_RECOVERY_ROOT = ".omx/state/orphan_pyc_recovery_20260505"
SIGNAL_LOSS_ROOT = ".omx/state/signal_loss_audit_20260505T1439Z"
ORPHAN_RECOVERY_MIRROR_ROOT = "reverse_engineering/orphan_pyc_recovery_20260505_codex"
PUBLIC_RUNTIME_RECOVERY_MIRROR_ROOT = (
    "reverse_engineering/public_frontier/recovered_runtime/"
    "experiments_results_20260505_pyc_recovery"
)
RESOLVED_DISPOSITIONS_MANIFEST = (
    ".omx/research/recovery_custody_resolved_dispositions_20260506_codex.json"
)
INCOMPLETE_REHYDRATION_MANIFEST = (
    ".omx/research/recovery_custody_incomplete_rehydration_dispositions_20260506_codex.json"
)

EXPECTED_PYC_TOTAL = 97
EXPECTED_PYC_STUBS = 76
EXPECTED_PYC_AST_OK = 21
EXPECTED_PYC_FULL_DECOMPILE = 2

EXPECTED_SIGNAL_DISPOSITIONS = {
    "duplicate_same_safe_to_delete_after_manifest_commit": 566,
    "do_not_promote_incomplete_recovery_preserve_for_manual_rehydration": 88,
    "preserve_until_matching_source_is_canonical_then_delete": 87,
    "blocked_recovery_input_needs_canonicalization_before_promotion": 3,
    "compare_by_hand_live_diff_before_merge_or_delete": 1,
}

EXPECTED_BLOCKED_RECOVERY_INPUTS = (
    "scripts/remote_lane_pr79_segaction_search.sh.PREFLIGHT_DEBT",
    "scripts/remote_lane_q_faithful_jointgen.sh.PREFLIGHT_DEBT",
    "scripts/remote_lane_sjkl_c067.sh.QUARANTINED",
)
EXPECTED_LIVE_DIFF_PATHS = ("docs/paper/ara/trace/events.jsonl",)
INCOMPLETE_REHYDRATION_DISPOSITION = (
    "do_not_promote_incomplete_recovery_preserve_for_manual_rehydration"
)
PRIVATE_FORENSIC_INCOMPLETE_CATEGORIES = (
    "provider_state",
    "public_or_experiment_intake",
)


@dataclass(frozen=True)
class RecoveryCustodyConfig:
    pyc_recovery_root: str = PYC_RECOVERY_ROOT
    signal_loss_root: str = SIGNAL_LOSS_ROOT
    resolved_dispositions_manifest: str | None = RESOLVED_DISPOSITIONS_MANIFEST
    incomplete_rehydration_manifest: str | None = INCOMPLETE_REHYDRATION_MANIFEST


def _nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected list JSON at {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"expected dict row {index} in {path}")
        rows.append(row)
    return rows


def _load_quarantine_records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"expected quarantine audit records in {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload["records"]):
        if not isinstance(row, dict):
            raise ValueError(f"expected dict quarantine record {index} in {path}")
        rows.append(row)
    return rows


def _load_resolved_dispositions(path: Path | None) -> dict[str, set[str]]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError(f"expected resolved-disposition entries in {path}")
    resolved: dict[str, set[str]] = {}
    for index, entry in enumerate(payload["entries"]):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entries[{index}] is not an object")
        disposition = entry.get("historical_disposition")
        relpath = entry.get("relpath")
        resolution = entry.get("resolution")
        evidence = entry.get("evidence")
        if not isinstance(disposition, str) or not disposition:
            raise ValueError(f"{path}: entries[{index}].historical_disposition must be nonempty")
        if not isinstance(relpath, str) or not relpath:
            raise ValueError(f"{path}: entries[{index}].relpath must be nonempty")
        if not isinstance(resolution, str) or not resolution:
            raise ValueError(f"{path}: entries[{index}].resolution must be nonempty")
        if not isinstance(evidence, str) or not evidence:
            raise ValueError(f"{path}: entries[{index}].evidence must be nonempty")
        resolved.setdefault(disposition, set()).add(relpath)
    return resolved


def _load_incomplete_rehydration_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected incomplete-rehydration object in {path}")
    if payload.get("schema") != "recovery_custody_incomplete_rehydration_dispositions_v1":
        raise ValueError(f"{path}: unsupported schema")
    if payload.get("historical_disposition") != INCOMPLETE_REHYDRATION_DISPOSITION:
        raise ValueError(f"{path}: historical_disposition mismatch")
    if payload.get("expected_total") != EXPECTED_SIGNAL_DISPOSITIONS[INCOMPLETE_REHYDRATION_DISPOSITION]:
        raise ValueError(f"{path}: expected_total mismatch")
    entries = payload.get("explicit_noncurrent_first_party_resolutions")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: explicit_noncurrent_first_party_resolutions must be a list")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: explicit entries[{index}] is not an object")
        for key in ("relpath", "resolution", "current_replacement", "evidence"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f"{path}: explicit entries[{index}].{key} must be nonempty")
        relpath = entry["relpath"]
        if relpath in seen:
            raise ValueError(f"{path}: duplicate explicit relpath {relpath}")
        seen.add(relpath)
    return payload


def _tracked_paths(repo_root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return {item for item in proc.stdout.decode("utf-8").split("\0") if item}


def _recovery_spec_path(relpath: str) -> str:
    return str(Path(relpath).with_suffix(".recovery_spec.json")).replace("\\", "/")


def _pyc_recovery_mirror_candidates(orphan_rel: str) -> list[str]:
    relpath = orphan_rel.replace("\\", "/").lstrip("/")
    candidates = [f"{ORPHAN_RECOVERY_MIRROR_ROOT}/{relpath}"]
    if relpath.startswith("experiments/results/"):
        candidates.append(
            f"{PUBLIC_RUNTIME_RECOVERY_MIRROR_ROOT}/"
            f"{relpath.removeprefix('experiments/results/')}"
        )
    results_marker = "/results/"
    if relpath.startswith(".omx/state/") and results_marker in relpath:
        candidates.append(
            f"{PUBLIC_RUNTIME_RECOVERY_MIRROR_ROOT}/"
            f"{relpath.split(results_marker, 1)[1]}"
        )
    return candidates


def _tracked_file_exists(repo_root: Path, tracked_paths: set[str], relpath: str) -> bool:
    normalized = relpath.replace("\\", "/").lstrip("/")
    return normalized in tracked_paths and (repo_root / normalized).is_file()


def _pyc_recovery_evidence(
    row: dict[str, Any],
    *,
    repo_root: Path,
    tracked_paths: set[str],
) -> dict[str, Any]:
    orphan_rel = str(row.get("orphan_rel") or "")
    pyc_path = str(row.get("pyc_path") or "")
    original_pyc_present = bool(pyc_path and Path(pyc_path).is_file())
    current_tracked_source = bool(
        orphan_rel and _tracked_file_exists(repo_root, tracked_paths, orphan_rel)
    )
    mirror_sources = [
        candidate
        for candidate in _pyc_recovery_mirror_candidates(orphan_rel)
        if _tracked_file_exists(repo_root, tracked_paths, candidate)
    ]
    mirror_specs = [
        _recovery_spec_path(candidate)
        for candidate in _pyc_recovery_mirror_candidates(orphan_rel)
        if _tracked_file_exists(repo_root, tracked_paths, _recovery_spec_path(candidate))
    ]
    source_present = current_tracked_source or bool(mirror_sources)
    spec_present = bool(mirror_specs)
    if original_pyc_present:
        durable = True
        evidence_class = "original_pyc_present"
    elif row.get("stub_written") is True:
        durable = source_present and spec_present
        evidence_class = "stub_source_and_spec" if durable else "stub_missing_source_or_spec"
    elif row.get("ast_parse_ok") is True or row.get("decompiled_ok") is True:
        durable = source_present
        evidence_class = "tracked_rehydrated_source" if durable else "ast_without_tracked_source"
    else:
        durable = source_present and spec_present
        evidence_class = "source_and_spec" if durable else "missing_recovery_evidence"
    return {
        "orphan_rel": orphan_rel,
        "original_pyc_present": original_pyc_present,
        "current_tracked_source": current_tracked_source,
        "mirror_sources": mirror_sources,
        "mirror_specs": mirror_specs,
        "durable": durable,
        "evidence_class": evidence_class,
    }


def audit_recovery_custody_snapshots(
    repo_root: Path,
    *,
    config: RecoveryCustodyConfig | None = None,
) -> AuditReport:
    config = config or RecoveryCustodyConfig()
    blockers: list[str] = []
    pyc_root = repo_root / config.pyc_recovery_root
    signal_root = repo_root / config.signal_loss_root
    resolved_manifest = (
        repo_root / config.resolved_dispositions_manifest
        if config.resolved_dispositions_manifest is not None
        else None
    )
    incomplete_manifest = (
        repo_root / config.incomplete_rehydration_manifest
        if config.incomplete_rehydration_manifest is not None
        else None
    )
    try:
        resolved_dispositions = _load_resolved_dispositions(resolved_manifest)
    except (OSError, ValueError) as exc:
        blockers.append(f"{config.resolved_dispositions_manifest}: {exc}")
        resolved_dispositions = {}
    try:
        incomplete_rehydration = _load_incomplete_rehydration_manifest(incomplete_manifest)
    except (OSError, ValueError) as exc:
        blockers.append(f"{config.incomplete_rehydration_manifest}: {exc}")
        incomplete_rehydration = {}
    try:
        tracked_paths = _tracked_paths(repo_root)
    except (OSError, subprocess.CalledProcessError):
        tracked_paths = set()

    pyc_required = (
        "RECOVERY_INDEX.md",
        "per_file_table.md",
        "per_file_table.tsv",
        "recover_orphans.py",
        "recovery_results.json",
    )
    missing_pyc = [name for name in pyc_required if not (pyc_root / name).is_file()]
    if missing_pyc:
        blockers.append(f"{config.pyc_recovery_root}: missing file(s): {', '.join(missing_pyc)}")

    pyc_rows: list[dict[str, Any]] = []
    pyc_summary: dict[str, Any] = {
        "root": config.pyc_recovery_root,
        "required_files_present": not missing_pyc,
    }
    if not missing_pyc:
        try:
            pyc_rows = _load_json_list(pyc_root / "recovery_results.json")
        except (OSError, ValueError) as exc:
            blockers.append(f"{config.pyc_recovery_root}/recovery_results.json: {exc}")
        if pyc_rows:
            evidence_rows = [
                _pyc_recovery_evidence(row, repo_root=repo_root, tracked_paths=tracked_paths)
                for row in pyc_rows
            ]
            stub_count = sum(1 for row in pyc_rows if row.get("stub_written") is True)
            ast_ok_count = sum(1 for row in pyc_rows if row.get("ast_parse_ok") is True)
            full_count = sum(1 for row in pyc_rows if row.get("decompiled_ok") is True)
            pyc_present_count = sum(1 for row in evidence_rows if row["original_pyc_present"])
            durable_evidence_count = sum(1 for row in evidence_rows if row["durable"])
            missing_pyc_with_durable_evidence_count = sum(
                1
                for row in evidence_rows
                if not row["original_pyc_present"] and row["durable"]
            )
            missing_durable_evidence = [row for row in evidence_rows if not row["durable"]]
            source_present_count = sum(
                1
                for row in pyc_rows
                if row.get("orphan_rel") and (repo_root / str(row["orphan_rel"])).exists()
            )
            spec_present_count = sum(
                1 for row in pyc_rows if row.get("spec_path") and Path(str(row["spec_path"])).exists()
            )
            tracked_source_present_count = sum(1 for row in evidence_rows if row["current_tracked_source"])
            mirror_source_present_count = sum(1 for row in evidence_rows if row["mirror_sources"])
            mirror_spec_present_count = sum(1 for row in evidence_rows if row["mirror_specs"])
            evidence_classes = Counter(str(row["evidence_class"]) for row in evidence_rows)
            pyc_summary.update(
                {
                    "recovery_result_count": len(pyc_rows),
                    "stub_written_count": stub_count,
                    "ast_parse_ok_count": ast_ok_count,
                    "full_decompile_count": full_count,
                    "pyc_present_count": pyc_present_count,
                    "durable_recovery_evidence_count": durable_evidence_count,
                    "missing_pyc_with_durable_recovery_evidence_count": (
                        missing_pyc_with_durable_evidence_count
                    ),
                    "source_present_count": source_present_count,
                    "spec_present_count": spec_present_count,
                    "tracked_source_present_count": tracked_source_present_count,
                    "mirror_source_present_count": mirror_source_present_count,
                    "mirror_spec_present_count": mirror_spec_present_count,
                    "evidence_class_counts": dict(evidence_classes),
                    "missing_durable_recovery_evidence_paths": [
                        str(row["orphan_rel"]) for row in missing_durable_evidence
                    ],
                    "top_level_roots": _counter(
                        [
                            {
                                "root": str(row.get("orphan_rel", "")).split("/", 1)[0]
                                if row.get("orphan_rel")
                                else ""
                            }
                            for row in pyc_rows
                        ],
                        "root",
                    ),
                }
            )
            if len(pyc_rows) != EXPECTED_PYC_TOTAL:
                blockers.append(f"pyc recovery row count drifted: {len(pyc_rows)} != {EXPECTED_PYC_TOTAL}")
            if stub_count != EXPECTED_PYC_STUBS:
                blockers.append(f"pyc stub count drifted: {stub_count} != {EXPECTED_PYC_STUBS}")
            if ast_ok_count != EXPECTED_PYC_AST_OK:
                blockers.append(f"pyc ast-ok count drifted: {ast_ok_count} != {EXPECTED_PYC_AST_OK}")
            if full_count != EXPECTED_PYC_FULL_DECOMPILE:
                blockers.append(
                    f"pyc full-decompile count drifted: {full_count} != {EXPECTED_PYC_FULL_DECOMPILE}"
                )
            if durable_evidence_count != len(pyc_rows):
                blockers.append(
                    "pyc recovery lost bytecode custody without durable recovery evidence: "
                    f"{durable_evidence_count}/{len(pyc_rows)} rows covered"
                )

    signal_required = (
        "counts.txt",
        "deleted_tracked_files.txt",
        "modified_tracked_files.txt",
        "quarantine_audit.json",
        "quarantine_audit.md",
        "git_status_short.txt",
        "untracked_files.txt",
        "worktree_list.txt",
    )
    missing_signal = [name for name in signal_required if not (signal_root / name).is_file()]
    if missing_signal:
        blockers.append(f"{config.signal_loss_root}: missing file(s): {', '.join(missing_signal)}")

    signal_rows: list[dict[str, Any]] = []
    deleted_tracked = _nonempty_lines(signal_root / "deleted_tracked_files.txt")
    modified_tracked = _nonempty_lines(signal_root / "modified_tracked_files.txt")
    signal_summary: dict[str, Any] = {
        "root": config.signal_loss_root,
        "required_files_present": not missing_signal,
        "deleted_tracked_count": len(deleted_tracked),
        "modified_tracked_count": len(modified_tracked),
    }
    if deleted_tracked:
        blockers.append(f"signal-loss snapshot records deleted tracked files: {len(deleted_tracked)}")
    if modified_tracked:
        blockers.append(f"signal-loss snapshot records modified tracked files: {len(modified_tracked)}")

    if not missing_signal:
        try:
            signal_rows = _load_quarantine_records(signal_root / "quarantine_audit.json")
        except (OSError, ValueError) as exc:
            blockers.append(f"{config.signal_loss_root}/quarantine_audit.json: {exc}")
        if signal_rows:
            dispositions = Counter(str(row.get("disposition")) for row in signal_rows)
            categories = Counter(str(row.get("category")) for row in signal_rows)
            blocked_inputs = sorted(
                str(row.get("relpath"))
                for row in signal_rows
                if row.get("disposition") == "blocked_recovery_input_needs_canonicalization_before_promotion"
            )
            live_diff_paths = sorted(
                str(row.get("relpath"))
                for row in signal_rows
                if row.get("disposition") == "compare_by_hand_live_diff_before_merge_or_delete"
            )
            resolved_blocked = sorted(
                resolved_dispositions.get("blocked_recovery_input_needs_canonicalization_before_promotion", set())
            )
            resolved_live_diff = sorted(
                resolved_dispositions.get("compare_by_hand_live_diff_before_merge_or_delete", set())
            )
            unresolved_blocked = sorted(set(blocked_inputs) - set(resolved_blocked))
            unresolved_live_diff = sorted(set(live_diff_paths) - set(resolved_live_diff))
            incomplete_rows = [
                row for row in signal_rows if row.get("disposition") == INCOMPLETE_REHYDRATION_DISPOSITION
            ]
            explicit_replacements = {
                str(entry["relpath"]): entry
                for entry in incomplete_rehydration.get(
                    "explicit_noncurrent_first_party_resolutions",
                    [],
                )
            }
            incomplete_classes: Counter[str] = Counter()
            unresolved_incomplete: list[str] = []
            current_tracked_incomplete: list[str] = []
            private_forensic_incomplete: list[str] = []
            explicit_noncurrent_resolved: list[str] = []
            for row in incomplete_rows:
                relpath = str(row.get("relpath"))
                category = str(row.get("category"))
                exists = (repo_root / relpath).exists()
                tracked = relpath in tracked_paths
                if exists and tracked:
                    incomplete_classes["resolved_by_current_main_tracked_source"] += 1
                    current_tracked_incomplete.append(relpath)
                    continue
                if category in PRIVATE_FORENSIC_INCOMPLETE_CATEGORIES:
                    incomplete_classes[f"private_forensic_{category}"] += 1
                    private_forensic_incomplete.append(relpath)
                    continue
                replacement = explicit_replacements.get(relpath)
                if replacement is not None:
                    replacement_path = str(replacement["current_replacement"])
                    if replacement_path not in tracked_paths or not (repo_root / replacement_path).exists():
                        blockers.append(
                            "incomplete-rehydration replacement is not current tracked source: "
                            f"{relpath} -> {replacement_path}"
                        )
                    incomplete_classes["resolved_by_explicit_noncurrent_first_party_replacement"] += 1
                    explicit_noncurrent_resolved.append(relpath)
                    continue
                incomplete_classes["unresolved_manual_rehydration"] += 1
                unresolved_incomplete.append(relpath)

            historical_incomplete_paths = {str(row.get("relpath")) for row in incomplete_rows}
            unexpected_explicit = sorted(set(explicit_replacements) - historical_incomplete_paths)
            if unexpected_explicit:
                blockers.append(
                    "incomplete-rehydration manifest references non-historical path(s): "
                    + ", ".join(unexpected_explicit[:10])
                )
            if incomplete_rehydration:
                expected_classes = incomplete_rehydration.get("expected_classification_counts")
                if not isinstance(expected_classes, dict):
                    blockers.append("incomplete-rehydration manifest missing expected_classification_counts")
                elif dict(incomplete_classes) != {
                    str(key): int(value) for key, value in expected_classes.items()
                }:
                    blockers.append(
                        "incomplete-rehydration classification drifted: "
                        f"{dict(incomplete_classes)} != {expected_classes}"
                    )
            signal_summary.update(
                {
                    "quarantine_record_count": len(signal_rows),
                    "category_counts": dict(categories),
                    "disposition_counts": dict(dispositions),
                    "blocked_recovery_inputs": blocked_inputs,
                    "live_diff_paths": live_diff_paths,
                    "resolved_manifest": str(resolved_manifest) if resolved_manifest is not None else None,
                    "resolved_blocked_recovery_inputs": resolved_blocked,
                    "resolved_live_diff_paths": resolved_live_diff,
                    "unresolved_blocked_recovery_inputs": unresolved_blocked,
                    "unresolved_live_diff_paths": unresolved_live_diff,
                    "incomplete_rehydration_manifest": (
                        str(incomplete_manifest) if incomplete_manifest is not None else None
                    ),
                    "incomplete_rehydration_classification_counts": dict(incomplete_classes),
                    "incomplete_rehydration_current_tracked_count": len(current_tracked_incomplete),
                    "incomplete_rehydration_private_forensic_count": len(private_forensic_incomplete),
                    "incomplete_rehydration_explicit_noncurrent_resolved": sorted(
                        explicit_noncurrent_resolved
                    ),
                    "unresolved_incomplete_manual_rehydration_paths": sorted(unresolved_incomplete),
                }
            )
            for disposition, expected_count in EXPECTED_SIGNAL_DISPOSITIONS.items():
                actual = dispositions.get(disposition, 0)
                if actual != expected_count:
                    blockers.append(
                        f"signal-loss disposition count drifted for {disposition}: "
                        f"{actual} != {expected_count}"
                    )
            if tuple(blocked_inputs) != tuple(sorted(EXPECTED_BLOCKED_RECOVERY_INPUTS)):
                blockers.append("signal-loss blocked recovery input set drifted")
            if tuple(live_diff_paths) != tuple(sorted(EXPECTED_LIVE_DIFF_PATHS)):
                blockers.append("signal-loss live-diff path set drifted")
            unexpected_resolved: list[str] = []
            for disposition, relpaths in resolved_dispositions.items():
                historical_paths = {
                    str(row.get("relpath"))
                    for row in signal_rows
                    if row.get("disposition") == disposition
                }
                for relpath in relpaths:
                    if relpath not in historical_paths:
                        unexpected_resolved.append(f"{disposition}:{relpath}")
            if unexpected_resolved:
                blockers.append(
                    "resolved-disposition manifest references non-historical path(s): "
                    + ", ".join(sorted(unexpected_resolved)[:10])
                )

    return AuditReport(
        audit="recovery_custody_snapshots",
        readiness_key="ready_for_recovery_custody_preservation",
        ready=not blockers,
        blockers=tuple(blockers),
        summary={
            "pyc_recovery": pyc_summary,
            "signal_loss": signal_summary,
            "next_required_dispositions": {
                "pyc_incomplete_manual_rehydration_records": EXPECTED_SIGNAL_DISPOSITIONS[
                    "do_not_promote_incomplete_recovery_preserve_for_manual_rehydration"
                ]
                if "unresolved_incomplete_manual_rehydration_paths" not in signal_summary
                else len(signal_summary["unresolved_incomplete_manual_rehydration_paths"]),
                "blocked_recovery_inputs": signal_summary.get(
                    "unresolved_blocked_recovery_inputs",
                    list(EXPECTED_BLOCKED_RECOVERY_INPUTS),
                ),
                "live_diff_paths": signal_summary.get(
                    "unresolved_live_diff_paths",
                    list(EXPECTED_LIVE_DIFF_PATHS),
                ),
            },
        },
        metadata={"repo_root": repo_relative(repo_root, repo_root)},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    report = audit_recovery_custody_snapshots(args.repo_root.resolve())
    payload = report.to_dict()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.format == "json":
        print(json_text(payload), end="")
    else:
        pyc = report.summary["pyc_recovery"]
        signal = report.summary["signal_loss"]
        detail = (
            f"(pyc_rows={pyc.get('recovery_result_count', 0)}; "
            f"pyc_present={pyc.get('pyc_present_count', 0)}; "
            f"pyc_durable={pyc.get('durable_recovery_evidence_count', 0)}; "
            f"quarantine_records={signal.get('quarantine_record_count', 0)}; "
            f"tracked_loss={signal.get('deleted_tracked_count', 0)} deleted/"
            f"{signal.get('modified_tracked_count', 0)} modified)"
        )
        print(report.render_text(pass_detail=detail))
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
