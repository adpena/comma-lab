#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit recovered remote lane scripts for canonical custody markers.

Recovered lane scripts are useful only if operators can tell whether they are
score-grade, proxy-only, or legacy-forensic before dispatch. This audit keeps
that classification executable: it checks that each recovered script exists,
has valid shell syntax, and preserves the fail-closed markers that distinguish
exact CUDA custody from proxy or prediction-only work.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - exercised under pytest package import
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.audit_contract import AuditReport, audit_exit_code  # noqa: E402
from tac.repo_io import json_text, repo_relative  # noqa: E402


@dataclass(frozen=True)
class LaneScriptContract:
    lane_id: str
    path: str
    classification: str
    required_markers: tuple[str, ...]


LANE_CONTRACTS: tuple[LaneScriptContract, ...] = (
    LaneScriptContract(
        lane_id="lane_sjkl_c067",
        path="scripts/remote_lane_sjkl_c067.sh",
        classification="canonical_exact_eval_delegated",
        required_markers=(
            "lane_sjkl_c067",
            "claim `lane_sjkl_c067`",
            "scripts/remote_archive_only_eval.sh",
            'export KEEP_EVAL_WORK="${KEEP_EVAL_WORK:-1}"',
            'CONTEST_AUTH_EVAL_CUSTODY_FLAGS="--keep-work-dir --work-dir"',
            "contest_auth_eval.json",
            "score claim until contest_auth_eval.json",
        ),
    ),
    LaneScriptContract(
        lane_id="lane_pr79_segaction_search",
        path="scripts/remote_lane_pr79_segaction_search.sh",
        classification="proxy_search_only_until_exact_eval",
        required_markers=(
            "lane_pr79_segaction_search",
            "PUBLIC_COMMIT=",
            'git checkout --detach "$PUBLIC_COMMIT"',
            "CLONED_PUBLIC_COMMIT",
            "PATCH_DIFF_SHA256",
            "BROTLI_PACKAGE",
            "parse_pr79_archive",
            "write_pr79_single_member_archive",
            '"score_claim": False',
            "remote_cuda_proxy_search_only_until_exact_t4_auth_eval",
            "archive.zip -> inflate.sh -> upstream/evaluate.py",
        ),
    ),
    LaneScriptContract(
        lane_id="lane_q_faithful_jointgen_88k",
        path="scripts/remote_lane_q_faithful_jointgen.sh",
        classification="legacy_recovered_exact_eval_runtime",
        required_markers=(
            "lane_q_faithful_jointgen_88k",
            "experiments/contest_auth_eval.py",
            "--keep-work-dir",
            '--work-dir "$LOG_DIR/eval_work"',
            "predicted band: [0.40, 0.80] [contest-CUDA]",
        ),
    ),
)


def _shell_syntax_ok(path: Path) -> tuple[bool, str]:
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def audit_recovered_remote_lanes(
    repo_root: Path,
    *,
    run_shell_syntax: bool = True,
) -> AuditReport:
    blockers: list[str] = []
    lane_summaries: list[dict[str, object]] = []

    for contract in LANE_CONTRACTS:
        path = repo_root / contract.path
        lane_summary: dict[str, object] = {
            "classification": contract.classification,
            "lane_id": contract.lane_id,
            "path": contract.path,
            "shell_syntax_checked": run_shell_syntax,
        }
        if not path.is_file():
            blockers.append(f"{contract.path}: missing recovered lane script")
            lane_summary["present"] = False
            lane_summaries.append(lane_summary)
            continue

        lane_summary["present"] = True
        text = path.read_text(encoding="utf-8")
        missing_markers = tuple(marker for marker in contract.required_markers if marker not in text)
        if missing_markers:
            blockers.append(
                f"{contract.path}: missing required marker(s): "
                + ", ".join(repr(marker) for marker in missing_markers)
            )
        lane_summary["missing_markers"] = list(missing_markers)
        if "git clone --depth 1 --branch" in text and "git checkout --detach" not in text:
            blockers.append(f"{contract.path}: public clone is branch-only without detached commit pin")
            lane_summary["branch_only_public_clone"] = True
        else:
            lane_summary["branch_only_public_clone"] = False

        if run_shell_syntax:
            syntax_ok, syntax_output = _shell_syntax_ok(path)
            lane_summary["shell_syntax_ok"] = syntax_ok
            if syntax_output:
                lane_summary["shell_syntax_output"] = syntax_output
            if not syntax_ok:
                blockers.append(f"{contract.path}: bash -n failed: {syntax_output}")
        else:
            lane_summary["shell_syntax_ok"] = None

        lane_summaries.append(lane_summary)

    return AuditReport(
        audit="recovered_remote_lane_canonicalization",
        readiness_key="ready_for_operator_visibility",
        ready=not blockers,
        blockers=tuple(blockers),
        summary={
            "lane_count": len(LANE_CONTRACTS),
            "scripts": lane_summaries,
            "runbook": "docs/runbooks/recovered_remote_lanes.md",
            "required_dispatch_rule": "claim lane before remote GPU launch",
        },
        metadata={
            "repo_root": repo_relative(repo_root, repo_root),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true", help="Accepted for preflight symmetry.")
    parser.add_argument(
        "--skip-shell-syntax",
        action="store_true",
        help="Skip bash -n checks. Intended only for unit tests on synthetic fixtures.",
    )
    args = parser.parse_args(argv)

    report = audit_recovered_remote_lanes(
        args.repo_root.resolve(),
        run_shell_syntax=not args.skip_shell_syntax,
    )
    payload = report.to_dict()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.format == "json":
        print(json_text(payload), end="")
    else:
        if report.ready:
            print(report.render_text(pass_detail=f"({len(LANE_CONTRACTS)} recovered scripts checked)"))
        else:
            print(report.render_text())
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
