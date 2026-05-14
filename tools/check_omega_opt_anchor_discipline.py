#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scan Omega-OPT claim surfaces for exact-anchor promotion discipline.

The Omega-OPT nested score numbers are hypotheses until a matching 1:1
archive/eval artifact exists. This guard fails closed on evidence rows,
lane-registry promotion, and the dated ledger so proxy, CPU, MPS, design, or
planning rows cannot become score/promotion/ranking evidence by accident.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.omega_opt_claims import (  # noqa: E402
    CLAIMS_BY_ID,
    FAIL_CLOSED_FIELDS,
    LANE_IDS_BY_CLAIM,
    OMEGA_OPT_CLAIMS,
    has_exact_1to1_anchor,
    text_mentions_omega_opt,
    validate_omega_opt_claim_table,
    validate_omega_opt_ledger_text,
    validate_omega_opt_row,
)

DEFAULT_LEDGER = REPO_ROOT / ".omx/research/omega_opt_anchor_discipline_20260508_codex.md"
DEFAULT_EVIDENCE_JSONL = REPO_ROOT / "reports/cathedral_autopilot_evidence.jsonl"
DEFAULT_LANE_REGISTRY = REPO_ROOT / ".omx/state/lane_registry.json"


@dataclass(frozen=True)
class OmegaOptAnchorFinding:
    """One fail-closed discipline finding."""

    surface: str
    path: str
    claim_id: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "surface": self.surface,
            "path": self.path,
            "claim_id": self.claim_id,
            "reason": self.reason,
        }

    def as_text(self) -> str:
        return f"{self.surface}:{self.path}:{self.claim_id}: {self.reason}"


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return [row for row in data if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _claim_id_for_row(row: dict[str, Any]) -> str:
    raw = str(row.get("claim_id") or row.get("technique") or "omega_opt_unknown")
    if raw in CLAIMS_BY_ID:
        return raw
    for claim in OMEGA_OPT_CLAIMS:
        if claim.claim_id in raw:
            return claim.claim_id
    return raw


def scan_evidence_jsonl(path: Path, repo_root: Path) -> list[OmegaOptAnchorFinding]:
    """Scan cathedral/autopilot evidence JSONL rows."""
    findings: list[OmegaOptAnchorFinding] = []
    for row in _load_jsonl(path):
        if not text_mentions_omega_opt(
            row.get("technique"),
            row.get("claim_id"),
            row.get("source"),
            row.get("evidence_semantics"),
        ):
            continue
        claim_id = _claim_id_for_row(row)
        for reason in validate_omega_opt_row(row):
            findings.append(OmegaOptAnchorFinding(
                surface="evidence_jsonl",
                path=_rel(path, repo_root),
                claim_id=claim_id,
                reason=reason,
            ))
    return findings


def scan_ledger(path: Path, repo_root: Path) -> list[OmegaOptAnchorFinding]:
    """Scan the dated markdown ledger."""
    if not path.is_file():
        return [
            OmegaOptAnchorFinding(
                surface="ledger",
                path=_rel(path, repo_root),
                claim_id="omega_opt_all",
                reason="dated_omega_opt_anchor_discipline_ledger_missing",
            )
        ]
    text = path.read_text(encoding="utf-8")
    findings: list[OmegaOptAnchorFinding] = []
    for item in validate_omega_opt_ledger_text(text):
        claim_id, _, reason = item.partition(": ")
        findings.append(OmegaOptAnchorFinding(
            surface="ledger",
            path=_rel(path, repo_root),
            claim_id=claim_id,
            reason=reason or item,
        ))
    return findings


def _lane_has_any_gate(lane: dict[str, Any]) -> bool:
    gates = lane.get("gates", {})
    if not isinstance(gates, dict):
        return False
    return any(
        isinstance(gate, dict) and gate.get("status") is True
        for gate in gates.values()
    )


def scan_lane_registry(path: Path, repo_root: Path) -> list[OmegaOptAnchorFinding]:
    """Scan lane maturity registry entries for unanchored promotion."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            OmegaOptAnchorFinding(
                surface="lane_registry",
                path=_rel(path, repo_root),
                claim_id="omega_opt_all",
                reason=f"invalid_json:{exc}",
            )
        ]
    lanes = data.get("lanes", [])
    if not isinstance(lanes, list):
        return []

    findings: list[OmegaOptAnchorFinding] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("id", ""))
        claim_id = LANE_IDS_BY_CLAIM.get(lane_id)
        if claim_id is None:
            continue
        promoted = int(lane.get("level", 0)) > 0 or _lane_has_any_gate(lane)
        if not promoted:
            continue
        evidence_text = json.dumps(lane.get("gates", {}), sort_keys=True)
        row = {
            "claim_id": claim_id,
            "evidence_grade": lane.get("notes", ""),
            "source": evidence_text,
            "notes": lane.get("notes", ""),
        }
        if not has_exact_1to1_anchor(row):
            findings.append(OmegaOptAnchorFinding(
                surface="lane_registry",
                path=_rel(path, repo_root),
                claim_id=claim_id,
                reason="lane_registry_gate_or_level_promoted_without_exact_1to1_anchor",
            ))
    return findings


def scan_plan_manifest(path: Path, repo_root: Path) -> list[OmegaOptAnchorFinding]:
    """Scan a generated codec-stack planner manifest, when supplied."""
    if not path.is_file():
        return [
            OmegaOptAnchorFinding(
                surface="plan_manifest",
                path=_rel(path, repo_root),
                claim_id="omega_opt_all",
                reason="plan_manifest_missing",
            )
        ]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            OmegaOptAnchorFinding(
                surface="plan_manifest",
                path=_rel(path, repo_root),
                claim_id="omega_opt_all",
                reason=f"invalid_json:{exc}",
            )
        ]
    score_band = (
        payload.get("metadata", {})
        .get("nested_optimization", {})
        .get("score_band_prediction", {})
    )
    claims = score_band.get("claims")
    if not isinstance(claims, list):
        return [
            OmegaOptAnchorFinding(
                surface="plan_manifest",
                path=_rel(path, repo_root),
                claim_id="omega_opt_all",
                reason="missing_nested_score_band_claim_table",
            )
        ]
    findings: list[OmegaOptAnchorFinding] = []
    for item in validate_omega_opt_claim_table(claims):
        claim_id, _, reason = item.partition(": ")
        findings.append(OmegaOptAnchorFinding(
            surface="plan_manifest",
            path=_rel(path, repo_root),
            claim_id=claim_id,
            reason=reason or item,
        ))
    return findings


def scan(
    *,
    repo_root: str | Path | None = None,
    ledger: str | Path | None = None,
    evidence_jsonl: str | Path | None = None,
    lane_registry: str | Path | None = None,
    plan_manifest: str | Path | None = None,
) -> list[OmegaOptAnchorFinding]:
    """Scan the configured Omega-OPT discipline surfaces."""
    root = Path(repo_root or REPO_ROOT)
    ledger_path = Path(ledger) if ledger else root / DEFAULT_LEDGER.relative_to(REPO_ROOT)
    evidence_path = (
        Path(evidence_jsonl)
        if evidence_jsonl
        else root / DEFAULT_EVIDENCE_JSONL.relative_to(REPO_ROOT)
    )
    registry_path = (
        Path(lane_registry)
        if lane_registry
        else root / DEFAULT_LANE_REGISTRY.relative_to(REPO_ROOT)
    )

    findings: list[OmegaOptAnchorFinding] = []
    findings.extend(scan_ledger(ledger_path, root))
    findings.extend(scan_evidence_jsonl(evidence_path, root))
    findings.extend(scan_lane_registry(registry_path, root))
    if plan_manifest is not None:
        findings.extend(scan_plan_manifest(Path(plan_manifest), root))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--evidence-jsonl", type=Path, default=None)
    parser.add_argument("--lane-registry", type=Path, default=None)
    parser.add_argument("--plan-manifest", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    findings = scan(
        repo_root=args.repo_root,
        ledger=args.ledger,
        evidence_jsonl=args.evidence_jsonl,
        lane_registry=args.lane_registry,
        plan_manifest=args.plan_manifest,
    )

    if args.format == "json":
        print(json.dumps({
            "ok": not findings,
            "finding_count": len(findings),
            "fail_closed_fields": list(FAIL_CLOSED_FIELDS),
            "findings": [finding.as_dict() for finding in findings],
        }, indent=2, sort_keys=True))
    elif findings:
        print(f"OMEGA-OPT ANCHOR DISCIPLINE FINDINGS ({len(findings)}):")
        for finding in findings:
            print(f"  - {finding.as_text()}")
    else:
        print("Omega-OPT anchor discipline: PASS")

    if findings and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
