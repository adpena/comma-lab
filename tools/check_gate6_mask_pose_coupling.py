#!/usr/bin/env python3
"""Gate 6 — Mask/pose coupling gate.

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #6.

Rule: any mask representation replacement must record decoded mask
SHA-256s, mask disagreement, pose-regeneration status, geometry
diagnostics, and the exact component-risk plan. Smaller mask bytes
alone are insufficient.

Detection (static):
  Scan build manifests + evidence rows for any artifact that:
    * sets ``mask_representation_changed=true``, OR
    * replaces ``masks.mkv`` / ``masks.nrv`` (filename appears in
      ``changed_payload_paths``), OR
    * has ``representation_name`` containing ``mask`` and is at
      level >= 1, OR
    * is a NeRV mask codec lane (lane_id contains ``nerv_mask`` or
      ``mask_nerv``).

  For every such artifact, REQUIRE ALL of:
    * ``decoded_mask_sha256s`` (list of 1199+ hashes for the 600/1199
      decoded masks)
    * ``mask_disagreement`` (numeric or per-frame array)
    * ``pose_regeneration_status`` (one of: ``regenerated_against_new_masks``,
      ``inherited_old_pose_with_proof``, ``not_required_explanation``)
    * ``geometry_diagnostics`` (path or inline summary)
    * ``component_risk_plan`` (path or inline summary)

  Files annotated ``# MASK_POSE_COUPLING_OK:<reason>`` (in the
  build_manifest sibling notes or a top-level annotation) are exempt.

Memory ref: ``feedback_representation_integration_gap_audit_20260508_codex.md``.
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
    "reports/dual_layer_stc_av1_evidence.jsonl",
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "decoded_mask_sha256s",
    "mask_disagreement",
    "pose_regeneration_status",
    "geometry_diagnostics",
    "component_risk_plan",
)

MASK_FILES = ("masks.mkv", "masks.nrv", "masks.bin")


@dataclass
class Finding:
    file_rel: str
    line_number: int
    technique: str
    reason: str


def _is_mask_representation(row: dict, source_lane_id: str = "") -> bool:
    if row.get("mask_representation_changed") is True:
        return True
    changed = row.get("changed_payload_paths", [])
    if isinstance(changed, list):
        for p in changed:
            if any(mf in str(p) for mf in MASK_FILES):
                return True
    rep = str(row.get("representation_name", "")).lower()
    if "mask" in rep:
        return True
    lane_id = str(row.get("lane_id", source_lane_id)).lower()
    if "nerv_mask" in lane_id or "mask_nerv" in lane_id:
        return True
    technique = str(row.get("technique", "")).lower()
    return bool("mask_codec" in technique or "mask_replacement" in technique)


def _has_waiver(row: dict) -> bool:
    notes = str(row.get("notes", ""))
    if "MASK_POSE_COUPLING_OK" in notes:
        return True
    return row.get("mask_pose_coupling_waived") is True


def _missing_fields(row: dict) -> list[str]:
    missing: list[str] = []
    for f in REQUIRED_FIELDS:
        v = row.get(f)
        if v is None:
            missing.append(f)
            continue
        if isinstance(v, str) and not v.strip():
            missing.append(f)
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            missing.append(f)
    return missing


def _scan_build_manifests(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    patterns = (
        "experiments/results/*/build_manifest.json",
        "experiments/results/*/*/build_manifest.json",
    )
    for pattern in patterns:
        for path in repo.glob(pattern):
            relpath = path.relative_to(repo).as_posix()
            if "public_pr" in relpath and "intake" in relpath:
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict):
                continue
            lane_id = str(manifest.get("lane_id", ""))
            if not _is_mask_representation(manifest, lane_id):
                continue
            if _has_waiver(manifest):
                continue
            missing = _missing_fields(manifest)
            if not missing:
                continue
            findings.append(
                Finding(
                    file_rel=relpath,
                    line_number=0,
                    technique=lane_id or "<unknown>",
                    reason=(
                        f"mask-representation manifest missing required "
                        f"coupling-gate fields: {','.join(missing)}. "
                        f"Mask replacements MUST record decoded mask "
                        f"SHA-256s, disagreement, pose-regen status, "
                        f"geometry diagnostics, component-risk plan. "
                        f"Gate 6 (mask/pose coupling)."
                    ),
                )
            )
    return findings


def _scan_evidence(repo: Path) -> list[Finding]:
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
            if not _is_mask_representation(row):
                continue
            if _has_waiver(row):
                continue
            # Only enforce when the row claims dispatch/score; pure
            # planning/proxy rows are exempt.
            if not (
                row.get("score_claim") is True
                or row.get("ready_for_exact_eval_dispatch") is True
            ):
                continue
            missing = _missing_fields(row)
            if not missing:
                continue
            findings.append(
                Finding(
                    file_rel=rel,
                    line_number=lineno,
                    technique=str(row.get("technique", "<unknown>")),
                    reason=(
                        f"mask-representation row claims score/dispatch "
                        f"but missing coupling-gate fields: "
                        f"{','.join(missing)}. Gate 6 (mask/pose coupling)."
                    ),
                )
            )
    return findings


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    findings.extend(_scan_build_manifests(repo))
    findings.extend(_scan_evidence(repo))
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
            f"[gate6-mask-pose-coupling] {len(findings)} violation(s):",
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
        print("[gate6-mask-pose-coupling] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
