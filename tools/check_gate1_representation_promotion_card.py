#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 1 — Representation promotion card discipline.

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #1.

Required fields for any "representation promotion card" claim (a manifest
row or build_manifest.json entry that introduces a learned representation
and intends to enter the contest exact-eval surface):

  representation_name, target_modes, source_artifact, archive_builder,
  inflate_consumer, runtime_manifest, changed_payload_paths,
  old_new_sha256s, component_risk_plan, exact_eval_command, owner,
  next_unblock_action.

A "promotion card" is detected when a row/object sets one of:
  * ``promotion_card`` (the explicit card itself)
  * ``representation_promotion_card``
  * ``representation_card``

OR a manifest path matches ``representation_card.json``.

Rule: every detected promotion card must contain ALL 12 required fields
with non-empty (truthy) values. Missing/empty fields are violations.

Scope (search roots):
  * ``experiments/results/**/representation_card.json``
  * ``experiments/results/**/build_manifest.json`` (only the ones that set
    one of the promotion-card markers)
  * ``reports/raw/**/manifest.json`` (only the ones with markers)
  * ``reports/cathedral_autopilot_evidence.jsonl`` rows containing markers

The check is **opt-in**: only manifests/rows that explicitly claim to be
promotion cards must obey the schema. Rows that don't claim promotion-card
status (the vast majority of existing artifacts) are ignored. Live count
on landing: 0 (no existing manifests claim promotion-card status yet);
ships warn-only by default and is intended to flip strict=True
immediately once the first card is created.

Memory ref: ``feedback_representation_integration_gap_audit_20260508_codex.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS: tuple[str, ...] = (
    "representation_name",
    "target_modes",
    "source_artifact",
    "archive_builder",
    "inflate_consumer",
    "runtime_manifest",
    "changed_payload_paths",
    "old_new_sha256s",
    "component_risk_plan",
    "exact_eval_command",
    "owner",
    "next_unblock_action",
)

PROMOTION_CARD_MARKERS = (
    "promotion_card",
    "representation_promotion_card",
    "representation_card",
)


@dataclass
class Finding:
    file_rel: str
    line_number: int
    representation_name: str
    reason: str


def _row_is_promotion_card(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    for marker in PROMOTION_CARD_MARKERS:
        v = row.get(marker)
        if v is True:
            return True
        if isinstance(v, dict):
            return True
        if isinstance(v, str) and v.strip():
            return True
    # Heuristic: if a row explicitly sets representation_name AND any of the
    # other required fields, it's a card-shaped object even without the
    # explicit marker. We require all 12 fields present.
    return bool(row.get("representation_name") and any(row.get(k) is not None for k in REQUIRED_FIELDS[1:]))


def _missing_or_empty(row: dict, field: str) -> bool:
    v = row.get(field)
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return bool(isinstance(v, (list, dict)) and len(v) == 0)


def _missing_fields(row: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if _missing_or_empty(row, f)]


def _scan_json_file(path: Path, repo: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    if not _row_is_promotion_card(obj):
        return []
    missing = _missing_fields(obj)
    if not missing:
        return []
    rel = path.relative_to(repo).as_posix()
    return [
        Finding(
            file_rel=rel,
            line_number=1,
            representation_name=str(obj.get("representation_name", "<unknown>")),
            reason=(
                f"promotion-card manifest missing required fields: "
                f"{','.join(missing)}. Gate 1 (representation promotion card)."
            ),
        )
    ]


def _scan_evidence_jsonl(path: Path, repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return findings
    rel = path.relative_to(repo).as_posix()
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
        if not _row_is_promotion_card(row):
            continue
        missing = _missing_fields(row)
        if not missing:
            continue
        findings.append(
            Finding(
                file_rel=rel,
                line_number=lineno,
                representation_name=str(
                    row.get("representation_name", "<unknown>")
                ),
                reason=(
                    f"promotion-card row missing required fields: "
                    f"{','.join(missing)}. Gate 1 (representation promotion "
                    f"card)."
                ),
            )
        )
    return findings


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []

    # 1. JSON manifests under experiments/results and reports/raw.
    json_globs = (
        "experiments/results/*/representation_card.json",
        "experiments/results/*/*/representation_card.json",
        "experiments/results/*/build_manifest.json",
        "reports/raw/*/manifest.json",
        "reports/raw/*/*/manifest.json",
    )
    for pattern in json_globs:
        for path in repo.glob(pattern):
            if not path.is_file():
                continue
            findings.extend(_scan_json_file(path, repo))

    # 2. Evidence ledgers
    evidence_paths = (
        repo / "reports" / "cathedral_autopilot_evidence.jsonl",
        repo / "reports" / "dual_layer_stc_av1_evidence.jsonl",
    )
    for path in evidence_paths:
        if path.is_file():
            findings.extend(_scan_evidence_jsonl(path, repo))

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
            f"[gate1-representation-promotion-card] {len(findings)} "
            f"violation(s):",
            file=sys.stderr,
        )
        for f in findings[:20]:
            print(
                f"  • {f.file_rel}:{f.line_number} "
                f"representation={f.representation_name}: {f.reason}",
                file=sys.stderr,
            )
        if args.strict:
            return 1
    else:
        print("[gate1-representation-promotion-card] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
