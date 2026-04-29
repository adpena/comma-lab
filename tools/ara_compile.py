#!/usr/bin/env python3
"""ara_compile.py — Agent-Native Research Artifact compiler for the pact lab.

Implements the Ara Compiler paradigm from arXiv 2604.24658
("The Last Human-Written Paper: Agent-Native Research Artifacts") as a
semi-automated pipeline tailored to this lab's existing artifacts.

This module is INTENTIONALLY conservative: it does not overwrite the
hand-curated layer files under docs/paper/ara/. It only:

    1. Walks experiments/results/**/contest_auth_eval.json and emits a
       normalized provenance index at docs/paper/ara/evidence/results_index.json
    2. Walks the topic-indexed agent memory under
       .claude/projects/-Users-adpena-Projects-pact/memory/ and produces a
       chronological events file at docs/paper/ara/trace/events.jsonl
    3. Validates Ara Seal Level 1 (structural integrity): every claim id in
       logic/claims.md has a forensic binding that resolves to a real path.

It is a scaffold. A future pass will add Live Research Manager hooks
(decision/observation/dead_end events emitted during the development flow)
and Ara Seal Level 3 (execution-reproducibility) checks.

Usage:
    python tools/ara_compile.py                  # full compile + validate
    python tools/ara_compile.py --validate-only  # just run the seal
    python tools/ara_compile.py --evidence-only  # just rebuild the index

The compiler is read-mostly. It writes to:
    docs/paper/ara/evidence/results_index.json
    docs/paper/ara/trace/events.jsonl
    docs/paper/ara/trace/seal_report.json

It does NOT modify any hand-curated layer file. If the hand-curated content
goes stale relative to the evidence, the seal report flags it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
ARA_ROOT = REPO_ROOT / "docs" / "paper" / "ara"
EVIDENCE_DIR = ARA_ROOT / "evidence"
TRACE_DIR = ARA_ROOT / "trace"
LOGIC_DIR = ARA_ROOT / "logic"

EXPERIMENTS_RESULTS = REPO_ROOT / "experiments" / "results"
MEMORY_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-adpena-Projects-pact"
    / "memory"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvidenceRecord:
    """A normalized pointer to a single contest-CUDA or advisory eval file."""

    lane: str  # e.g. "lane_g_v3_landed"
    score_lane: str  # "[contest-CUDA]" / "[Modal-T4-CUDA]" / "[advisory only]"
    score: float | None
    archive_bytes: int | None
    posenet: float | None
    segnet: float | None
    rate: float | None
    artifact_path: str  # repo-relative
    captured_at: str | None  # ISO timestamp from the file mtime if not embedded


@dataclass
class TraceEvent:
    """A research event extracted from agent memory.

    Aligns with the Ara Live Research Manager event taxonomy:
    decision | claim | experiment | heuristic | dead_end | observation | pivot.
    """

    timestamp: str
    event_type: str
    title: str
    source_path: str  # repo-relative or memory-relative
    summary: str  # one-line


@dataclass
class SealFinding:
    severity: str  # "ok" | "warn" | "error"
    code: str
    message: str
    pointer: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_load_json(path: Path) -> dict | None:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _classify_score_lane(payload: dict, hint_path: Path) -> str:
    """Best-effort label per the lane-tag rule in CLAUDE.md.

    The repo does not always tag scores in-band, so we use lightweight
    heuristics:
      - presence of 'contest_cuda' / 'inflate' in the file path -> contest-CUDA
      - 'modal' in path -> Modal-T4-CUDA
      - 'mps' in path -> MPS-PROXY
      - else advisory.
    """
    lower = str(hint_path).lower()
    if "modal" in lower:
        return "[Modal-T4-CUDA]"
    if "mps" in lower:
        return "[MPS-PROXY]"
    if "contest_auth_eval" in lower or "/contest" in lower or "inflate" in lower:
        return "[contest-CUDA]"
    return "[advisory only]"


def _coerce_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _isoformat(ts_seconds: float) -> str:
    return (
        datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# Evidence ingestion
# ---------------------------------------------------------------------------


def walk_evidence(root: Path) -> list[EvidenceRecord]:
    if not root.exists():
        return []

    records: list[EvidenceRecord] = []
    for path in sorted(root.rglob("*.json")):
        if path.name not in {"contest_auth_eval.json", "auth_eval.json"}:
            # also accept files that look like modal_auth_eval_*.json
            if not path.name.startswith("modal_auth_eval"):
                continue

        payload = _safe_load_json(path)
        if not isinstance(payload, dict):
            continue

        # Pull common fields if present; fall back to None.
        score = _coerce_float(
            payload.get("score")
            or payload.get("total_score")
            or payload.get("final_score"),
        )
        archive_bytes = _coerce_int(
            payload.get("archive_bytes")
            or payload.get("archive_size")
            or payload.get("bytes"),
        )
        posenet = _coerce_float(
            payload.get("posenet_distortion")
            or payload.get("posenet")
            or payload.get("pose_distortion"),
        )
        segnet = _coerce_float(
            payload.get("segnet_distortion")
            or payload.get("segnet")
            or payload.get("seg_distortion"),
        )
        rate = _coerce_float(payload.get("rate"))

        captured_at = payload.get("captured_at") or payload.get("timestamp")
        if not captured_at:
            try:
                captured_at = _isoformat(path.stat().st_mtime)
            except OSError:
                captured_at = None

        lane = path.parent.name if path.parent != root else path.stem
        rec = EvidenceRecord(
            lane=lane,
            score_lane=_classify_score_lane(payload, path),
            score=score,
            archive_bytes=archive_bytes,
            posenet=posenet,
            segnet=segnet,
            rate=rate,
            artifact_path=str(path.relative_to(REPO_ROOT)),
            captured_at=captured_at,
        )
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Memory event ingestion
# ---------------------------------------------------------------------------

# Order matters: more specific rules first. A "landed_runge_phenomenon"
# memory file must classify as dead_end, not experiment.
EVENT_TYPE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"feedback_.*", re.IGNORECASE), "heuristic"),
    (re.compile(r".*council.*kill.*", re.IGNORECASE), "decision"),
    (re.compile(r".*regression.*|.*runge.*|.*crash.*", re.IGNORECASE), "dead_end"),
    (re.compile(r".*paradigm.*shift.*|.*pivot.*", re.IGNORECASE), "pivot"),
    (re.compile(r".*landed.*", re.IGNORECASE), "experiment"),
    (re.compile(r"project_.*", re.IGNORECASE), "observation"),
]


# Disclosure-policy redaction list. Per CLAUDE.md the following lane names
# and recipes are competitively-load-bearing and MUST NOT appear in any
# artifact intended for public consumption. Although docs/paper/ara/ is
# currently a private internal artifact, the compiler is conservative by
# default: it redacts the title field when a memory file references one of
# these terms. The redacted record still keeps the source_path so an
# operator with repo access can read the unredacted file.
PRIVATE_TERMS = (
    "Lane W",
    "Lane Ω",
    "Lane Omega",
    "Lane DARTS",
    "DARTS-S",
    "FR-Ω",
    "FR-Omega",
    "Hessian-aware",
    "hard-pair",
    "Hessian quantization",
    "SO Hessian",
)


def _redact_private(text: str) -> str:
    """Return text with private-term sentences replaced by a placeholder.

    Conservative behaviour: if any private term appears, drop the title
    entirely and substitute a generic placeholder. Operators can recover
    the original via source_path.
    """
    lower = text.lower()
    for term in PRIVATE_TERMS:
        if term.lower() in lower:
            return "[REDACTED — private lane reference; see source_path]"
    return text


def _classify_memory(name: str) -> str:
    for pattern, kind in EVENT_TYPE_RULES:
        if pattern.match(name):
            return kind
    return "observation"


def walk_memory(root: Path) -> list[TraceEvent]:
    if not root.exists():
        return []

    events: list[TraceEvent] = []
    for path in sorted(root.rglob("*.md")):
        try:
            with path.open() as fh:
                first_line = fh.readline().strip()
        except OSError:
            continue

        title = first_line.lstrip("# ").strip() or path.stem
        title = _redact_private(title)
        try:
            mtime = _isoformat(path.stat().st_mtime)
        except OSError:
            mtime = ""

        events.append(
            TraceEvent(
                timestamp=mtime,
                event_type=_classify_memory(path.stem),
                title=title,
                source_path=str(path),
                summary=_redact_private(path.stem),
            )
        )

    events.sort(key=lambda e: e.timestamp, reverse=True)
    return events


# ---------------------------------------------------------------------------
# Ara Seal — Level 1 structural integrity
# ---------------------------------------------------------------------------


CLAIM_BLOCK_RE = re.compile(
    r"^## (?P<id>C\d+) — (?P<title>.+?)$.*?"
    r"\*\*evidence\*\*: (?P<evidence>.+?)$.*?"
    r"\*\*code\*\*: (?P<code>.+?)$",
    re.MULTILINE | re.DOTALL,
)


def seal_level_1(claims_path: Path, evidence_index: list[EvidenceRecord]) -> list[SealFinding]:
    findings: list[SealFinding] = []
    if not claims_path.exists():
        findings.append(
            SealFinding("error", "missing_claims", f"{claims_path} not found", None)
        )
        return findings

    text = claims_path.read_text()
    matches = list(CLAIM_BLOCK_RE.finditer(text))
    if not matches:
        findings.append(
            SealFinding(
                "warn",
                "no_claim_blocks_matched",
                "claims.md regex returned 0 blocks; check formatting",
                str(claims_path),
            )
        )
        return findings

    evidence_pointers = {r.artifact_path for r in evidence_index}
    for m in matches:
        claim_id = m.group("id")
        evidence_ref = m.group("evidence").strip().rstrip(",.;").strip("`")
        code_ref = m.group("code").strip().rstrip(",.;").strip("`")

        # Only check for repository-resident evidence pointers.
        # `evidence/.../foo.json` lives under docs/paper/ara/evidence in the
        # ideal compiled state, but in the skeleton we accept a relative
        # pointer if it resolves to ANY file in the repo.
        if "(regenerate)" in evidence_ref:
            findings.append(
                SealFinding(
                    "warn",
                    "evidence_regen_pending",
                    f"{claim_id} evidence is marked (regenerate); not yet compiled",
                    evidence_ref,
                )
            )
            continue

        repo_path = (ARA_ROOT / evidence_ref).resolve()
        if not repo_path.exists():
            # try resolving as repo-relative
            alt = (REPO_ROOT / evidence_ref).resolve()
            if not alt.exists():
                findings.append(
                    SealFinding(
                        "error",
                        "evidence_dangling",
                        f"{claim_id} evidence pointer does not resolve",
                        evidence_ref,
                    )
                )
                continue

        code_path = (REPO_ROOT / code_ref).resolve()
        if not code_path.exists():
            findings.append(
                SealFinding(
                    "warn",
                    "code_dangling",
                    f"{claim_id} code pointer does not resolve",
                    code_ref,
                )
            )

    if not findings:
        findings.append(
            SealFinding("ok", "level_1_pass", "all claim bindings resolve", None)
        )
    return findings


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_evidence_index(records: Iterable[EvidenceRecord]) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_DIR / "results_index.json"
    payload = {
        "ara_version": "0.1",
        "compiled_at": _isoformat(datetime.now(tz=timezone.utc).timestamp()),
        "record_count": sum(1 for _ in records),
    }
    # We need to consume records once; build a list.
    records = list(records)
    payload["record_count"] = len(records)
    payload["records"] = [asdict(r) for r in records]
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


def write_events(events: Iterable[TraceEvent]) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    out = TRACE_DIR / "events.jsonl"
    with out.open("w") as fh:
        for ev in events:
            fh.write(json.dumps(asdict(ev), sort_keys=True) + "\n")
    return out


def write_seal_report(findings: list[SealFinding]) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    out = TRACE_DIR / "seal_report.json"
    payload = {
        "ara_version": "0.1",
        "level": "1 (structural integrity)",
        "compiled_at": _isoformat(datetime.now(tz=timezone.utc).timestamp()),
        "summary": {
            "ok": sum(1 for f in findings if f.severity == "ok"),
            "warn": sum(1 for f in findings if f.severity == "warn"),
            "error": sum(1 for f in findings if f.severity == "error"),
        },
        "findings": [asdict(f) for f in findings],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile the Ara research artifact for the pact lab."
    )
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    log = (lambda *_: None) if args.quiet else print

    records: list[EvidenceRecord] = []
    if not args.validate_only:
        records = walk_evidence(EXPERIMENTS_RESULTS)
        out = write_evidence_index(records)
        log(f"[ara_compile] evidence: {len(records)} records -> {out}")

    if args.evidence_only:
        return 0

    if not args.validate_only:
        events = walk_memory(MEMORY_DIR)
        out = write_events(events)
        log(f"[ara_compile] trace events: {len(events)} -> {out}")

    findings = seal_level_1(LOGIC_DIR / "claims.md", records)
    out = write_seal_report(findings)
    error_count = sum(1 for f in findings if f.severity == "error")
    log(
        f"[ara_compile] seal level 1: "
        f"{sum(1 for f in findings if f.severity == 'ok')} ok, "
        f"{sum(1 for f in findings if f.severity == 'warn')} warn, "
        f"{error_count} error -> {out}",
    )

    return 0 if error_count == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
