#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""B6 — Pre-dispatch retired-config redispatch scanner.

Bug class: a config previously dispatched with
``measured_config_status=measured_config_retired_exact_cuda_negative`` (or
``contest_dispatch_verdict=measured_config_retired_exact_cuda_negative``) is
about to be dispatched again, wasting paid GPU time on a known-bad config.

Real instance: ``reports/cathedral_autopilot_evidence.jsonl:35`` shows
``lossy_coarsening @ T0312-noproject`` scored 0.3517 retired; a Lightning
bootstrap fix would re-dispatch the SAME config and waste $0.30-0.60.

Detection mode (preflight, static):
  * Scan ``reports/cathedral_autopilot_evidence.jsonl`` and
    ``reports/raw/pr101_omega_opt_evidence.jsonl`` for rows whose
    ``contest_dispatch_verdict`` or ``measured_config_status`` flags as
    ``measured_config_retired_*``.
  * Build the index of retired ``(technique, lane_id, archive_sha256)`` keys.
  * Verify each retired row also carries
    ``dispatch_blockers`` containing
    ``"reactivation_required_before_new_dispatch"`` AND a
    ``reactivation_criteria`` non-empty array. Without those guards, the
    actuator (``tools/parallel_dispatch_top_k.py``,
    ``scripts/launch_lane_on_vastai.py``) cannot block a redispatch.

If any retired row is missing reactivation guards, it's a B6 violation
(silent re-dispatch surface).

Memory ref: ``feedback_codex_adversarial_review_4_landings_20260508.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

EVIDENCE_FILES: tuple[str, ...] = (
    "reports/cathedral_autopilot_evidence.jsonl",
    "reports/raw/pr101_omega_opt_evidence.jsonl",
)

RETIRED_TOKENS = (
    "measured_config_retired_exact_cuda_negative",
    "measured_config_retired",
    "exact_cuda_negative_retired",
)

REQUIRED_BLOCKER_TOKEN = "reactivation_required_before_new_dispatch"


@dataclass
class Finding:
    file_rel: str
    line_number: int
    technique: str
    reason: str


def _is_retired(row: dict) -> bool:
    for fld in (
        "contest_dispatch_verdict",
        "measured_config_status",
        "evidence_grade",
    ):
        v = str(row.get(fld, "")).lower()
        for tok in RETIRED_TOKENS:
            if tok in v:
                return True
    return False


def _has_reactivation_guards(row: dict) -> bool:
    blockers = row.get("dispatch_blockers", [])
    if not isinstance(blockers, list):
        return False
    if not any(REQUIRED_BLOCKER_TOKEN in str(b) for b in blockers):
        return False
    crit = row.get("reactivation_criteria", [])
    if not isinstance(crit, list) or len(crit) == 0:
        return False
    return True


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    for rel in EVIDENCE_FILES:
        path = repo / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if not _is_retired(row):
                continue
            if _has_reactivation_guards(row):
                continue
            findings.append(
                Finding(
                    file_rel=rel,
                    line_number=lineno,
                    technique=str(row.get("technique", "<unknown>")),
                    reason=(
                        "row marked retired (measured_config_retired*) but "
                        "missing dispatch_blockers containing "
                        f"'{REQUIRED_BLOCKER_TOKEN}' AND/OR non-empty "
                        "reactivation_criteria array. Without these guards "
                        "the parallel-dispatch actuator cannot block "
                        "redispatch of a known-bad config. B6."
                    ),
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    findings = scan(repo)
    if findings:
        print(
            f"[B6-retired-config-redispatch-guard] {len(findings)} "
            f"violation(s):",
            file=sys.stderr,
        )
        for f in findings[:20]:
            print(
                f"  • {f.file_rel}:{f.line_number} technique={f.technique}: "
                f"{f.reason}",
                file=sys.stderr,
            )
        if args.strict:
            return 1
    else:
        print("[B6-retired-config-redispatch-guard] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
