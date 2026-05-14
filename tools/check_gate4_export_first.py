#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 4 — Export-first gate (declare export format BEFORE long training).

Source: ``.omx/research/representation_integration_gap_audit_20260508_codex.md``
prevent-recurrence gate #4.

Rule: a trainable renderer or implicit representation must declare its
export format **before** long training begins. If the variant is non-FP4A,
non-int4, or otherwise outside the current archive exporter, the run is
research-only until a packet exporter exists.

Detection (static):
  Two complementary scans:

  1. Lane registry: any lane with ``trainable=true`` (or whose lane-id
     contains a representation token like ``coolchic``, ``c3``,
     ``nerv``, ``hnerv``, ``mnerv``) at level >= 1 must have a recorded
     ``export_format`` field, OR a ``research_only=true`` annotation, OR
     a ``deploy_runbook`` evidence string referring to a recognized
     packet exporter.

  2. Training-script annotation: any new training script under
     ``experiments/`` or ``src/tac/experiments/`` that mentions a
     long-training profile (epochs >= 100 or ``--total-iterations``
     >= 1000) must have a docstring or sibling JSON declaring
     ``EXPORT_FORMAT`` from the canonical set
     ``{FP4A, int4, FP4A+brotli, packed_decoder_brotli, GRAY_LUT,
     research_only_no_export}`` AND must mark research-only variants
     accordingly.

  Files annotated ``# EXPORT_FORMAT_OK:<format>:<reason>`` are exempt.

Live count on landing: known violations include the ``coolchic``/``c3``
training paths flagged in the audit (renderer-only research variants).
Ships warn-only initially; flip to STRICT after authoring
``# EXPORT_FORMAT_OK:research_only_no_export:<reason>`` in those files.

Memory ref: ``feedback_representation_integration_gap_audit_20260508_codex.md``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

REPRESENTATION_TOKENS = (
    "coolchic",
    "cool_chic",
    "cool-chic",
    "c3_render",
    "c3-render",
    "nerv",
    "hnerv",
    "mnerv",
    "implicit_neural",
)

RECOGNIZED_EXPORT_FORMATS = (
    "FP4A",
    "int4",
    "int6",
    "int8",
    "FP4A+brotli",
    "packed_decoder_brotli",
    "GRAY_LUT",
    "research_only_no_export",
    "CCh1",
    "C3R1",
)

EXPORT_OK_PATTERN = re.compile(r"#\s*EXPORT_FORMAT_OK\s*:\s*(\S+?)\s*:\s*(.+)")
EXPORT_FORMAT_DECLARATION_PATTERN = re.compile(
    r"EXPORT_FORMAT\s*[:=]\s*['\"]([^'\"]+)['\"]"
)


@dataclass
class Finding:
    file_rel: str
    line_number: int
    representation: str
    reason: str


def _has_export_ok_waiver(text: str) -> tuple[bool, str | None]:
    m = EXPORT_OK_PATTERN.search(text)
    if not m:
        return False, None
    fmt = m.group(1).strip()
    return True, fmt


def _has_export_format_declaration(text: str) -> tuple[bool, str | None]:
    m = EXPORT_FORMAT_DECLARATION_PATTERN.search(text)
    if not m:
        return False, None
    return True, m.group(1)


def _is_long_training_script(text: str) -> bool:
    """Heuristic: any script that mentions an epoch arg with a default >= 100
    or a total-iterations arg with default >= 1000."""
    epoch_match = re.search(
        r"--(?:epochs|total-iterations|max-steps)[^,)\n]*?default\s*=\s*(\d+)",
        text,
    )
    if epoch_match:
        try:
            n = int(epoch_match.group(1))
            return n >= 100
        except ValueError:
            return False
    return False


def _mentions_representation(text: str) -> str | None:
    lower = text.lower()
    for tok in REPRESENTATION_TOKENS:
        if tok in lower:
            return tok
    return None


def _scan_training_script(path: Path, repo: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    rep = _mentions_representation(text)
    if rep is None:
        return []
    has_waiver, _waiver_fmt = _has_export_ok_waiver(text)
    if has_waiver:
        return []
    has_decl, decl_fmt = _has_export_format_declaration(text)
    if has_decl and decl_fmt in RECOGNIZED_EXPORT_FORMATS:
        return []
    if not _is_long_training_script(text):
        return []
    rel = path.relative_to(repo).as_posix()
    return [
        Finding(
            file_rel=rel,
            line_number=1,
            representation=rep,
            reason=(
                "long-training script mentions representation "
                f"({rep!r}) but lacks EXPORT_FORMAT declaration in "
                "{recognized list} AND lacks "
                "# EXPORT_FORMAT_OK:<format>:<reason> waiver. Declare "
                "export format before long training. Gate 4 "
                "(export-first)."
            ),
        )
    ]


def _scan_lane_registry(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    path = repo / ".omx" / "state" / "lane_registry.json"
    if not path.is_file():
        return findings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return findings
    rel = path.relative_to(repo).as_posix()
    lanes = data.get("lanes", [])
    if not isinstance(lanes, list):
        return findings
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("id", ""))
        rep_tok = None
        for tok in REPRESENTATION_TOKENS:
            if tok in lane_id.lower():
                rep_tok = tok
                break
        if rep_tok is None:
            continue
        level = lane.get("level", 0)
        if not isinstance(level, int) or level < 1:
            continue
        # Required: export_format OR research_only=true
        export_format = lane.get("export_format")
        research_only = lane.get("research_only")
        if isinstance(export_format, str) and export_format.strip():
            continue
        if research_only is True:
            continue
        # Acceptable substitute: notes/evidence containing
        # "research_only_no_export" or one of the recognized formats.
        notes = str(lane.get("notes", ""))
        if any(fmt in notes for fmt in RECOGNIZED_EXPORT_FORMATS):
            continue
        findings.append(
            Finding(
                file_rel=rel,
                line_number=0,
                representation=lane_id,
                reason=(
                    f"lane '{lane_id}' (representation token {rep_tok!r}) "
                    f"is at level={level} but lacks export_format field, "
                    f"research_only=true flag, OR a recognized export "
                    f"format token in `notes`. Declare export format "
                    f"before scaling. Gate 4 (export-first)."
                ),
            )
        )
    return findings


def scan(repo_root: Path | None = None) -> list[Finding]:
    repo = (repo_root or REPO_ROOT_DEFAULT).resolve()
    findings: list[Finding] = []
    findings.extend(_scan_lane_registry(repo))

    # Training-script scan
    candidate_dirs = (
        repo / "experiments",
        repo / "src" / "tac" / "experiments",
    )
    for d in candidate_dirs:
        if not d.is_dir():
            continue
        for py_file in d.rglob("*.py"):
            # Skip vendored public PR intakes.
            relpath = py_file.relative_to(repo).as_posix()
            if "public_pr" in relpath and "intake" in relpath:
                continue
            if "/tests/" in relpath or relpath.endswith("_test.py"):
                continue
            findings.extend(_scan_training_script(py_file, repo))

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
            f"[gate4-export-first] {len(findings)} violation(s):",
            file=sys.stderr,
        )
        for f in findings[:20]:
            print(
                f"  • {f.file_rel}:{f.line_number} "
                f"representation={f.representation}: {f.reason}",
                file=sys.stderr,
            )
        if args.strict:
            return 1
    else:
        print("[gate4-export-first] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
