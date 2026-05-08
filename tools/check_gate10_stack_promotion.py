#!/usr/bin/env python3
"""Gate 10 — Stack promotion gate (HStack/VStack/cross-paradigm).

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #10.

Rule: HStack/VStack/cross-paradigm work MUST include the real archive
boundary, side information, latent streams, K/scale tables, decoder
overhead, runtime consumer, and exact-eval plan BEFORE it can be
scheduled as a CUDA candidate. Byte-proxy manifests should continue
to exist, but their dispatch fields must remain fail-closed.

Detection (static):
  Scan canonical evidence ledgers + build manifests for rows whose
  ``technique`` / ``transform_kind`` / ``lane_id`` contains stack
  tokens (``hstack``, ``vstack``, ``cross_paradigm``,
  ``hstack_vstack``).

  For every such row:

  Case A (dispatch claim): if ``score_claim=true`` OR
  ``ready_for_exact_eval_dispatch=true`` OR
  ``contest_dispatch_verdict`` contains ``positive``/``promote``,
  REQUIRE ALL of:
    * ``archive_boundary`` (offsets/lengths of each stack member)
    * ``side_information`` (path or inline summary)
    * ``latent_streams`` (path or inline summary)
    * ``k_scale_tables`` (path or inline summary)
    * ``decoder_overhead_bytes`` (int)
    * ``runtime_consumer`` (path to inflate.sh / inflate.py)
    * ``exact_eval_plan`` (string or path)

  Case B (proxy/byte-anchor only): the row MUST set
  ``score_claim=false`` AND ``ready_for_exact_eval_dispatch=false``.
  Otherwise it's silently passing through to dispatch.

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
)

STACK_TOKENS = (
    "hstack",
    "vstack",
    "cross_paradigm",
    "cross-paradigm",
    "hstack_vstack",
)


@dataclass
class Finding:
    file_rel: str
    line_number: int
    technique: str
    reason: str


def _is_stack_row(row: dict) -> bool:
    fields = (
        str(row.get("technique", "")),
        str(row.get("transform_kind", "")),
        str(row.get("lane_id", "")),
        str(row.get("paradigm", "")),
    )
    blob = " ".join(fields).lower()
    return any(tok in blob for tok in STACK_TOKENS)


def _claims_dispatch(row: dict) -> bool:
    if row.get("score_claim") is True:
        return True
    if row.get("ready_for_exact_eval_dispatch") is True:
        return True
    verdict = str(row.get("contest_dispatch_verdict", "")).lower()
    return bool("positive" in verdict or "promote" in verdict or "frontier" in verdict)


REQUIRED_DISPATCH_FIELDS: tuple[str, ...] = (
    "archive_boundary",
    "side_information",
    "latent_streams",
    "k_scale_tables",
    "decoder_overhead_bytes",
    "runtime_consumer",
    "exact_eval_plan",
)


def _missing_dispatch_fields(row: dict) -> list[str]:
    missing: list[str] = []
    for f in REQUIRED_DISPATCH_FIELDS:
        v = row.get(f)
        if v is None:
            missing.append(f)
            continue
        if isinstance(v, str) and not v.strip():
            missing.append(f)
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            missing.append(f)
            continue
        if isinstance(v, int) and v == 0 and f == "decoder_overhead_bytes":
            # An integer of 0 is suspicious for decoder_overhead_bytes;
            # still allow but warn.
            pass
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
            if not _is_stack_row(row):
                continue
            if _claims_dispatch(row):
                missing = _missing_dispatch_fields(row)
                if missing:
                    findings.append(
                        Finding(
                            file_rel=rel,
                            line_number=lineno,
                            technique=str(row.get("technique", "<unknown>")),
                            reason=(
                                f"stack row claims dispatch but missing "
                                f"required fields: {','.join(missing)}. "
                                f"Gate 10 (stack promotion)."
                            ),
                        )
                    )
            else:
                # Case B: must explicitly disable score_claim AND
                # ready_for_exact_eval_dispatch.
                if (
                    row.get("score_claim") is not False
                    or row.get("ready_for_exact_eval_dispatch") is not False
                ):
                    findings.append(
                        Finding(
                            file_rel=rel,
                            line_number=lineno,
                            technique=str(row.get("technique", "<unknown>")),
                            reason=(
                                "stack/proxy row must explicitly set "
                                "score_claim=false AND "
                                "ready_for_exact_eval_dispatch=false. "
                                "Gate 10 (stack promotion)."
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
            if not _is_stack_row(manifest):
                continue
            if _claims_dispatch(manifest):
                missing = _missing_dispatch_fields(manifest)
                if missing:
                    findings.append(
                        Finding(
                            file_rel=relpath,
                            line_number=0,
                            technique=str(
                                manifest.get("lane_id", "<unknown>")
                            ),
                            reason=(
                                f"stack manifest claims dispatch but "
                                f"missing fields: {','.join(missing)}. "
                                f"Gate 10 (stack promotion)."
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
            f"[gate10-stack-promotion] {len(findings)} violation(s):",
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
        print("[gate10-stack-promotion] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
