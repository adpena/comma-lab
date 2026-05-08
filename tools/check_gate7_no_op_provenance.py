#!/usr/bin/env python3
"""Gate 7 — No-op and provenance gate.

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #7.

Rule: every byte-level experiment MUST prove the targeted payload
changed, MUST prove the scored runtime consumed it, AND MUST preserve
old/new archive SHA-256s. Reuse, decode/re-encode, provenance-only
changes, and cosmetic ZIP repacks stay forensic until this proof exists.

Detection (static):
  Scan build manifests + evidence rows that claim a byte-level transform
  (any of):
    * ``transform_kind`` containing ``byte_level`` / ``repack`` /
      ``codec_swap`` / ``brotli_param`` / ``encoder_swap``
    * ``technique`` containing ``repack`` / ``brotli_q_search`` /
      ``codec_substitution``
    * ``empirical_archive_bytes`` set with a non-trivial baseline delta

  Where a byte-level claim is made AND the row claims dispatch readiness
  (``score_claim=true`` OR ``ready_for_exact_eval_dispatch=true``),
  REQUIRE ALL of:
    * ``old_archive_sha256`` non-empty
    * ``new_archive_sha256`` non-empty (or ``archive_sha256``)
    * ``payload_change_proof`` non-empty (path to a diff/test that
      proves the targeted payload changed) OR
      ``no_op_detector_passed=true``
    * ``runtime_consumption_proof`` non-empty (path to a runtime log
      proving inflate.sh consumed the new bytes) OR
      ``runtime_closure_verified=true``

  Sister gate to B5 (``check_inflate_wire_format_no_dead_bytes``); this
  gate enforces the *external* (manifest-level) provenance contract.

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

BYTE_LEVEL_TOKENS = (
    "byte_level",
    "repack",
    "codec_swap",
    "brotli_param",
    "encoder_swap",
    "brotli_q_search",
    "codec_substitution",
)


@dataclass
class Finding:
    file_rel: str
    line_number: int
    technique: str
    reason: str


def _is_byte_level_row(row: dict) -> bool:
    transform = str(row.get("transform_kind", "")).lower()
    technique = str(row.get("technique", "")).lower()
    if any(tok in transform or tok in technique for tok in BYTE_LEVEL_TOKENS):
        return True
    has_bytes = isinstance(row.get("empirical_archive_bytes"), int) or isinstance(row.get("archive_bytes"), int)
    has_baseline = isinstance(row.get("baseline_archive_bytes"), int) or isinstance(
        row.get("baseline_bytes"), int
    )
    return has_bytes and has_baseline


def _claims_dispatch(row: dict) -> bool:
    if row.get("score_claim") is True:
        return True
    return row.get("ready_for_exact_eval_dispatch") is True


def _has_old_new_sha(row: dict) -> bool:
    new_sha = row.get("new_archive_sha256") or row.get("archive_sha256")
    old_sha = row.get("old_archive_sha256") or row.get("baseline_archive_sha256")
    if not isinstance(new_sha, str) or not new_sha.strip():
        return False
    return not (not isinstance(old_sha, str) or not old_sha.strip())


def _has_payload_change_proof(row: dict) -> bool:
    p = row.get("payload_change_proof")
    if isinstance(p, str) and p.strip():
        return True
    return row.get("no_op_detector_passed") is True


def _has_runtime_consumption_proof(row: dict) -> bool:
    p = row.get("runtime_consumption_proof")
    if isinstance(p, str) and p.strip():
        return True
    return row.get("runtime_closure_verified") is True


def _missing_fields(row: dict) -> list[str]:
    missing: list[str] = []
    if not _has_old_new_sha(row):
        missing.append("old_archive_sha256+new_archive_sha256")
    if not _has_payload_change_proof(row):
        missing.append("payload_change_proof OR no_op_detector_passed")
    if not _has_runtime_consumption_proof(row):
        missing.append(
            "runtime_consumption_proof OR runtime_closure_verified"
        )
    return missing


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
            if not _is_byte_level_row(row):
                continue
            if not _claims_dispatch(row):
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
                        f"byte-level transform row claims score/dispatch "
                        f"but missing provenance proofs: {','.join(missing)}. "
                        f"Cosmetic repacks stay forensic until proof exists. "
                        f"Gate 7 (no-op + provenance)."
                    ),
                )
            )
    return findings


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
            if not _is_byte_level_row(manifest):
                continue
            if not _claims_dispatch(manifest):
                continue
            missing = _missing_fields(manifest)
            if not missing:
                continue
            findings.append(
                Finding(
                    file_rel=relpath,
                    line_number=0,
                    technique=str(manifest.get("lane_id", "<unknown>")),
                    reason=(
                        f"byte-level transform manifest claims score/"
                        f"dispatch but missing provenance proofs: "
                        f"{','.join(missing)}. Gate 7 (no-op + provenance)."
                    ),
                )
            )
    return findings


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    findings.extend(_scan_evidence(repo))
    findings.extend(_scan_build_manifests(repo))
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
            f"[gate7-no-op-provenance] {len(findings)} violation(s):",
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
        print("[gate7-no-op-provenance] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
