#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 2 — No naked bytes.

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #2.

Rule: if bytes are not inside a scored archive member or parser section
that is consumed by ``inflate.sh``, the artifact MUST set
``score_claim=false`` AND ``ready_for_exact_eval_dispatch=false``. It can
be valuable empirical/forensic evidence, but it cannot be a candidate.

Detection (static):
  Scan canonical evidence ledgers for rows that satisfy ANY of:

    * ``score_claim=true``
    * ``ready_for_exact_eval_dispatch=true``
    * ``contest_dispatch_verdict`` value contains a positive token
      (``positive`` / ``promote`` / ``frontier`` / ``exact_cuda_positive``).

  For every such row, REQUIRE at least ONE of the following byte-closure
  proofs:

    * non-empty ``archive_path`` AND non-empty ``archive_sha256``
    * non-empty ``inflate_consumer`` (path to inflate.sh / inflate.py
      that consumes the bytes)
    * ``parser_section_manifest`` non-empty
  Rows that lack ALL byte-closure proofs but claim score/dispatch are
  Gate 2 violations.

  Sister gate to B2 (``check_evidence_row_archive_bytes_has_provenance``):
  B2 catches phantom byte counts; this gate catches phantom score-claims
  that lack archive bytes.

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

POSITIVE_DISPATCH_TOKENS = (
    "positive",
    "promote",
    "frontier",
    "exact_cuda_positive",
    "contest_positive",
)


@dataclass
class Finding:
    file_rel: str
    line_number: int
    technique: str
    reason: str


def _row_claims_score(row: dict) -> bool:
    if row.get("score_claim") is True:
        return True
    if row.get("ready_for_exact_eval_dispatch") is True:
        return True
    verdict = str(row.get("contest_dispatch_verdict", "")).lower()
    return any(tok in verdict for tok in POSITIVE_DISPATCH_TOKENS)


def _row_has_byte_closure(row: dict) -> bool:
    archive_path = row.get("archive_path")
    archive_sha = row.get("archive_sha256")
    if (
        isinstance(archive_path, str)
        and archive_path.strip()
        and isinstance(archive_sha, str)
        and archive_sha.strip()
    ):
        return True
    inflate = row.get("inflate_consumer")
    if isinstance(inflate, str) and inflate.strip():
        return True
    parser_manifest = row.get("parser_section_manifest")
    if isinstance(parser_manifest, (dict, list)) and len(parser_manifest) > 0:
        return True
    return bool(isinstance(parser_manifest, str) and parser_manifest.strip())


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
            if not _row_claims_score(row):
                continue
            if _row_has_byte_closure(row):
                continue
            findings.append(
                Finding(
                    file_rel=rel,
                    line_number=lineno,
                    technique=str(row.get("technique", "<unknown>")),
                    reason=(
                        "row claims score/dispatch readiness "
                        "(score_claim=true OR ready_for_exact_eval_dispatch="
                        "true OR positive contest_dispatch_verdict) but "
                        "lacks ALL byte-closure proofs (archive_path+sha256, "
                        "inflate_consumer, or parser_section_manifest). "
                        "CUDA status text is not byte closure. Gate 2 "
                        "(no naked bytes)."
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
            f"[gate2-no-naked-bytes] {len(findings)} violation(s):",
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
        print("[gate2-no-naked-bytes] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
