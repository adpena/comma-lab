#!/usr/bin/env python3
"""B2 — Phantom byte-proxy claim scanner.

Bug class: an evidence row claims ``empirical_archive_bytes=N`` from
``len(blob)`` of an unwrapped encode (e.g. ``pipeline.encode(x,
skip_validate=True)``) without ZIP packaging + inflate roundtrip.

Real instance: cross-paradigm 137,531 B winner was ``len(blob_op1)`` not a
byte-closed archive.

Rule: every evidence row in the canonical evidence ledgers that sets
``empirical_archive_bytes`` must satisfy AT LEAST ONE of:
  1. set ``archive_sha256`` (a real bytestream that can be re-verified), OR
  2. set ``byte_proxy_only=true`` AND ``cuda_eval_worth_testing=false`` AND
     ``ready_for_exact_eval_dispatch=false`` (explicitly self-tagged proxy
     with ALL guards), OR
  3. set ``measured_config_status`` to a value containing ``proxy`` /
     ``byte_anchor`` / ``cpu_prep`` (well-known proxy tags), OR
  4. include ``[CPU-prep`` / ``[byte-anchor`` / ``[empirical:`` / ``[contest-CUDA``
     in the row's ``source`` field (textual provenance tag).

Memory ref: ``feedback_review_engineering_council_4_landings_20260508.md``.

Exit codes:
    0    no untagged byte-proxy rows
    1    untagged byte-proxy rows found (only with --strict)
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

PROXY_SOURCE_TOKENS = (
    "[CPU-prep",
    "[byte-anchor",
    "[empirical:",
    "[contest-CUDA",
    "[MPS-research-signal]",
    "[predicted",
    "[scorer-basin-parity:",
)

PROXY_STATUS_TOKENS = (
    "proxy",
    "byte_anchor",
    "byte-anchor",
    "cpu_prep",
    "cpu-prep",
    "measured_config_proxy",
)


@dataclass
class Finding:
    file_rel: str
    line_number: int
    technique: str
    reason: str


def _row_has_provenance(row: dict) -> bool:
    # 1. Real archive SHA?
    if isinstance(row.get("archive_sha256"), str) and row["archive_sha256"].strip():
        return True
    # 2. Explicit byte-proxy-only with ALL guards
    if row.get("byte_proxy_only") is True:
        if (
            row.get("cuda_eval_worth_testing") is False
            and row.get("ready_for_exact_eval_dispatch") is False
        ):
            return True
    # 3. measured_config_status hints proxy
    status = str(row.get("measured_config_status", ""))
    for tok in PROXY_STATUS_TOKENS:
        if tok in status.lower():
            return True
    # 4. textual provenance tag in source
    source = str(row.get("source", ""))
    for tok in PROXY_SOURCE_TOKENS:
        if tok in source:
            return True
    # 5. evidence_grade explicit + score_claim=false combo (well-known
    #    promotion-blocked CPU/MPS proxy form)
    grade = str(row.get("evidence_grade", "")).lower()
    if (
        ("cpu-prep" in grade or "mps-research-signal" in grade or "byte-anchor" in grade)
        and row.get("score_claim") is False
    ):
        return True
    return False


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = repo_root or REPO_ROOT_DEFAULT
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
            if "empirical_archive_bytes" not in row:
                continue
            if _row_has_provenance(row):
                continue
            findings.append(
                Finding(
                    file_rel=rel,
                    line_number=lineno,
                    technique=str(row.get("technique", "<unknown>")),
                    reason=(
                        "empirical_archive_bytes set but no archive_sha256, "
                        "byte_proxy_only-with-guards, proxy status tag, or "
                        "textual provenance tag (CPU-prep/byte-anchor/"
                        "empirical/contest-CUDA) found. B2 (phantom "
                        "byte-proxy claim)."
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
            f"[B2-evidence-archive-bytes-provenance] {len(findings)} "
            f"violation(s):",
            file=sys.stderr,
        )
        for f in findings:
            print(
                f"  • {f.file_rel}:{f.line_number} technique={f.technique}: "
                f"{f.reason}",
                file=sys.stderr,
            )
        if args.strict:
            return 1
    else:
        print("[B2-evidence-archive-bytes-provenance] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
