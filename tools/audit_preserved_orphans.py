#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit local preserved-orphan shadows against canonical tracked files.

The filter-repo and pyc-recovery passes intentionally parked a few source-like
files in ``.omx/state/orphans_preserved`` while live replacements were
reviewed. This guard makes that custody explicit: exact duplicates and reviewed
superseded shadows are safe to delete after the disposition is recorded; any
unknown preserved source file blocks cleanup.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.audit_contract import AuditReport, audit_exit_code  # noqa: E402
from tac.repo_io import json_text, repo_relative, sha256_file  # noqa: E402

PRESERVED_ORPHAN_ROOT = ".omx/state/orphans_preserved"


@dataclass(frozen=True)
class PreservedOrphanContract:
    orphan_name: str
    canonical_path: str
    reviewed_status: str | None = None
    rationale: str | None = None


PRESERVED_ORPHAN_CONTRACTS: tuple[PreservedOrphanContract, ...] = (
    PreservedOrphanContract(
        orphan_name="remote_lane_pr79_segaction_search_no_provenance_20260505.sh.orphan",
        canonical_path="scripts/remote_lane_pr79_segaction_search.sh",
        reviewed_status="superseded_by_hardened_canonical_script",
        rationale=(
            "canonical PR79 script adds public commit pinning, provenance, "
            "heartbeat, typed payload parsing, and no-score proxy markers"
        ),
    ),
    PreservedOrphanContract(
        orphan_name="remote_lane_q_faithful_jointgen_drift_20260505.sh.orphan",
        canonical_path="scripts/remote_lane_q_faithful_jointgen.sh",
    ),
    PreservedOrphanContract(
        orphan_name="test_yousfi_3_variance_noise_unresolved_imports_20260505.py.orphan",
        canonical_path="src/tac/tests/test_yousfi_3_variance_noise.py",
    ),
    PreservedOrphanContract(
        orphan_name="test_yousfi_5_uncertainty_unresolved_imports_20260505.py.orphan",
        canonical_path="src/tac/tests/test_yousfi_5_uncertainty.py",
    ),
)


def _contract_by_name(
    contracts: tuple[PreservedOrphanContract, ...],
) -> dict[str, PreservedOrphanContract]:
    return {contract.orphan_name: contract for contract in contracts}


def audit_preserved_orphans(
    repo_root: Path,
    *,
    orphan_root: str = PRESERVED_ORPHAN_ROOT,
    contracts: tuple[PreservedOrphanContract, ...] = PRESERVED_ORPHAN_CONTRACTS,
) -> AuditReport:
    root = repo_root / orphan_root
    blockers: list[str] = []
    rows: list[dict[str, object]] = []
    contract_by_name = _contract_by_name(contracts)

    if not root.exists():
        return AuditReport(
            audit="preserved_orphan_canonicalization",
            readiness_key="ready_for_preserved_orphan_cleanup",
            ready=True,
            summary={
                "orphan_root": orphan_root,
                "present_count": 0,
                "duplicate_count": 0,
                "superseded_count": 0,
                "unknown_count": 0,
                "missing_canonical_count": 0,
                "rows": [],
            },
            metadata={"repo_root": repo_relative(repo_root, repo_root)},
        )

    for path in sorted(candidate for candidate in root.iterdir() if candidate.is_file()):
        contract = contract_by_name.get(path.name)
        row: dict[str, object] = {
            "orphan_path": repo_relative(path, repo_root),
            "orphan_name": path.name,
            "orphan_sha256": sha256_file(path),
            "orphan_bytes": path.stat().st_size,
        }
        if contract is None:
            row["status"] = "unknown_preserved_orphan"
            blockers.append(f"{repo_relative(path, repo_root)}: unknown preserved orphan")
            rows.append(row)
            continue

        canonical = repo_root / contract.canonical_path
        row["canonical_path"] = contract.canonical_path
        row["reviewed_status"] = contract.reviewed_status
        row["rationale"] = contract.rationale
        if not canonical.is_file():
            row["status"] = "missing_canonical"
            blockers.append(
                f"{repo_relative(path, repo_root)}: canonical path missing: {contract.canonical_path}"
            )
            rows.append(row)
            continue

        row["canonical_sha256"] = sha256_file(canonical)
        row["canonical_bytes"] = canonical.stat().st_size
        if row["orphan_sha256"] == row["canonical_sha256"]:
            row["status"] = "duplicate_of_canonical"
        elif contract.reviewed_status is not None:
            row["status"] = contract.reviewed_status
        else:
            row["status"] = "unreviewed_content_drift"
            blockers.append(
                f"{repo_relative(path, repo_root)}: differs from canonical "
                f"{contract.canonical_path} without reviewed supersession"
            )
        rows.append(row)

    unknown_count = sum(1 for row in rows if row.get("status") == "unknown_preserved_orphan")
    missing_count = sum(1 for row in rows if row.get("status") == "missing_canonical")
    duplicate_count = sum(1 for row in rows if row.get("status") == "duplicate_of_canonical")
    superseded_count = sum(
        1 for row in rows if row.get("status") == "superseded_by_hardened_canonical_script"
    )
    return AuditReport(
        audit="preserved_orphan_canonicalization",
        readiness_key="ready_for_preserved_orphan_cleanup",
        ready=not blockers,
        blockers=tuple(blockers),
        summary={
            "orphan_root": orphan_root,
            "present_count": len(rows),
            "duplicate_count": duplicate_count,
            "superseded_count": superseded_count,
            "unknown_count": unknown_count,
            "missing_canonical_count": missing_count,
            "rows": rows,
        },
        metadata={"repo_root": repo_relative(repo_root, repo_root)},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    report = audit_preserved_orphans(args.repo_root.resolve())
    payload = report.to_dict()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.format == "json":
        print(json_text(payload), end="")
    else:
        detail = (
            f"({report.summary['duplicate_count']} duplicate; "
            f"{report.summary['superseded_count']} superseded; "
            f"{report.summary['unknown_count']} unknown)"
        )
        print(report.render_text(pass_detail=detail))
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
